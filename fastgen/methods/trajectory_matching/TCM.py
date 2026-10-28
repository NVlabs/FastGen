# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import Dict, Any, Callable, TYPE_CHECKING, Optional

import torch
from torch import Tensor

from fastgen.methods import CMModel
from fastgen.utils.tree_utils import assert_same
from fastgen.utils.distributed.collectives import sync_all, sync_any
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.networks.network import FastGenNetwork


class TCMPrecond(torch.nn.Module):
    """Two-stage Consistency Model Preconditioner.

    This module implements a two-stage consistency model that switches between
    a teacher network (stage-1) and a student network (stage-2) based on the
    transition time step. For timesteps below the transition threshold, it uses
    the teacher network; for timesteps above, it uses the student network.

    The implementation handles distributed training scenarios with proper
    synchronization across ranks to ensure consistent behavior in FSDP setups.

    Args:
        net_t: Teacher network for stage-1 (t < transition_t)
        net_s: Student network for stage-2 (t >= transition_t)
        transition_t: Transition time step threshold
    """

    def __init__(
        self,
        net_t: FastGenNetwork,
        net_s: FastGenNetwork,
        transition_t: float,
    ) -> None:
        super().__init__()

        self.net_t = net_t
        self.net_s = net_s
        self.transition_t = transition_t

        assert hasattr(self.net_s, "noise_scheduler")
        if not self.net_s.noise_scheduler.min_t <= transition_t <= self.net_s.noise_scheduler.max_t:
            raise ValueError(f"transition_t must be in [min_t, max_t], got {transition_t}")

        # every net should have pred_type and noise_scheduler
        self.net_pred_type = self.net_s.net_pred_type
        self.noise_scheduler = self.net_s.noise_scheduler

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Any] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Forward pass through the two-stage consistency model.

        Args:
            x_t: Noisy input tensor at time t
            t: Time step tensor
            condition: Optional conditioning inputs
            **model_kwargs: Additional keyword arguments passed to networks

        Returns:
            Model output (x0 prediction)
        """
        device = x_t.device

        # Determine which samples are in second stage (t >= transition_t)
        second_stage_mask = t >= self.transition_t

        # Synchronize mask information across distributed ranks for FSDP consistency
        local_all_second_stage = second_stage_mask.all().item()
        local_any_second_stage = second_stage_mask.any().item()

        global_all_second_stage = sync_all(local_all_second_stage, device)
        global_any_second_stage = sync_any(local_any_second_stage, device)

        # Optimization: If all samples are in second stage, only use student network
        if global_all_second_stage:
            return self.net_s(x_t, t, condition=condition, **model_kwargs)

        # Get teacher network predictions (always needed as base)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            with torch.no_grad():
                out_t = self.net_t(x_t, t, condition=condition, **model_kwargs)

        # If some samples are in second stage, get student predictions and blend
        if global_any_second_stage:
            out_s = self.net_s(x_t, t, condition=condition, **model_kwargs)

            # Replace teacher outputs with student outputs for second stage samples
            out_t[second_stage_mask] = out_s[second_stage_mask]

        return out_t


class TCMModel(CMModel):
    @staticmethod
    def _mean_or_zero(values: Tensor) -> Tensor:
        """Return the mean, or a scalar zero for an empty subset."""
        return values.mean() if values.numel() else values.new_zeros(())

    def build_model(self):
        super().build_model()

        if not self.config.inference:
            # instantiate the cm teacher (stage-1 model)
            self.cm_teacher = copy.deepcopy(self.net)
            self.register_modules(cm_teacher=self.cm_teacher)

            # Below transition_t the output is the frozen stage-1 teacher's and carries no gradient, so the sampled
            # times must not fall below it; a rank-local runtime check would desynchronize the collective forwards.
            sample_t_cfg, min_t = self.config.sample_t_cfg, self.net.noise_scheduler.min_t
            if sample_t_cfg.min_t is not None:
                min_t = max(min_t, sample_t_cfg.min_t)
            if min_t < self.config.transition_t:
                raise ValueError(
                    f"sample_t_cfg.min_t must be at least transition_t={self.config.transition_t} "
                    f"(no gradient below it), got an effective lower sampling bound of {min_t}"
                )

            self.net_tcm = TCMPrecond(net_t=self.cm_teacher, net_s=self.net, transition_t=self.config.transition_t)

    def on_train_begin(self, fsdp_context=None):
        super().on_train_begin(fsdp_context=fsdp_context)

        # The frozen stage-1 target is prepared through module_dict with every other root, but it
        # keeps the train-mode behaviour (e.g. dropout) it was trained with.
        self.cm_teacher.train()

    def init_optimizers(self):
        super().init_optimizers()
        # Runs after the trainer loads trainer.checkpointer.pretrained_ckpt_path, which sets resume_iter.
        if not hasattr(self, "resume_iter"):
            logger.warning(
                "TCM starts its second stage without the stage-1 consistency model; set "
                "trainer.checkpointer.pretrained_ckpt_path to its checkpoint."
            )

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for the TCM model.

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """
        # Prepare training data and conditions
        real_data = data["real"]
        condition = data.get("condition")
        neg_condition = data.get("neg_condition")

        # build_model ensures these times are at least transition_t, below which net_tcm's output has no gradient.
        t = self.net.noise_scheduler.sample_t(
            real_data.shape[0], **self.sample_t_cfg.sample_t_kwargs(), device=self.device
        )

        # sample the boundary time step with the `boundary_prob` probability
        num_elements_to_mask = int(t.shape[0] * self.config.boundary_prob)
        if num_elements_to_mask == 0:
            mask_t = torch.rand(t.shape[0], device=t.device) < self.config.boundary_prob
        else:
            indices = torch.randperm(t.shape[0])
            mask_indices = indices[:num_elements_to_mask]
            mask_t = torch.zeros(t.shape[0], device=t.device, dtype=torch.bool)
            mask_t[mask_indices] = True

        # set the boundary time steps to the transition time step
        t[mask_t] = self.config.transition_t  # routed to the student by TCMPrecond's t >= transition_t

        cm_loss, loss_unweighted, D_yt = self._compute_cm_loss(
            self.net_tcm, real_data=real_data, t=t, condition=condition, neg_condition=neg_condition
        )

        # get boundary loss
        assert_same(shape=True, device=self.device, cm_loss=cm_loss, t=t, mask_t=mask_t)
        loss_boundary = self._mean_or_zero(cm_loss[mask_t])
        cm_loss = self._mean_or_zero(cm_loss[~mask_t])
        loss_unweighted = self._mean_or_zero(loss_unweighted[~mask_t])

        loss = cm_loss + self.config.w_boundary * loss_boundary
        loss_map = {
            "total_loss": loss,
            "cm_loss": cm_loss,
            "loss_boundary": loss_boundary,
            "unweighted_cm_loss": loss_unweighted,
        }
        outputs = self._get_outputs(D_yt, condition=condition, neg_condition=neg_condition)

        # compute rewards
        loss_map = self.apply_reward(D_yt, data, loss_map)

        return loss_map, outputs
