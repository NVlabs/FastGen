# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AnyFlow — any-step video diffusion with flow maps and on-policy distillation.

AnyFlow trains a single flow-map model ``u_theta(x_t, t, r)`` predicting the
average velocity from ``t`` to ``r`` (``r <= t``), so one model serves any
inference NFE: each Euler-like step picks its own integration interval.

Stage 1 (flow-map pretrain) is MeanFlow with AnyFlow's hyperparameters, run on
``MeanFlowModel`` directly (``configs/experiments/WanT2V/config_anyflow.py``) —
there is no AnyFlow-specific pretrain code. Stage 2 (this module) is DMD2 on the
pretrained flow-map weights, deviating from stock DMD2 twice: the student
generates via a compressed flow-map rollout with gradient through all segments
(``gen_data_from_net``), and every student update co-trains the Stage-1 flow-map
loss on the real batch (``cotrain_pretrain_weight``).
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import torch

from fastgen.methods.consistency_model.mean_flow import FlowMapLossMixin
from fastgen.methods.distribution_matching.dmd2 import DMD2Model
from fastgen.networks.noise_schedule import SHIFTED_TIME_DIST_TYPES
from fastgen.utils.distributed import world_size
import fastgen.utils.logging_utils as logger


if TYPE_CHECKING:
    from typing import Dict, Callable

    from fastgen.configs.methods.config_anyflow import ModelConfig


class AnyFlowModel(FlowMapLossMixin, DMD2Model):
    """AnyFlow on-policy stage: DMD2 with a flow-map rollout student.

    ``FlowMapLossMixin`` supplies the co-trained Stage-1 objective and the
    flow-map validation sample loop, which integrates with ``r = t_next``.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self._init_flow_map_loss()

        logger.info(
            f"AnyFlow on-policy: student_sample_steps_list={self.config.student_sample_steps_list}, "
            f"student_update_freq={self.config.student_update_freq}, "
            f"cotrain_pretrain_weight={self.config.cotrain_pretrain_weight}, "
            f"gan_loss_weight_gen={self.config.gan_loss_weight_gen}"
        )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _generate_noise_and_time(
        self, real_data: torch.Tensor, iteration: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The student rollout always starts from pure noise at ``max_t``."""
        batch_size = real_data.shape[0]

        eps_student = torch.randn(batch_size, *self.input_shape, device=self.device, dtype=real_data.dtype)
        t_student = torch.full(
            (batch_size,),
            self.net.noise_scheduler.max_t,
            device=self.device,
            dtype=self.net.noise_scheduler.t_precision,
        )
        input_student = self.net.noise_scheduler.latents(noise=eps_student)

        t = self._sample_noising_time(batch_size, iteration)
        eps = torch.randn_like(real_data, device=self.device, dtype=real_data.dtype)
        return input_student, t_student, t, eps

    def _broadcast_choice(self, high: int) -> int:
        """Pick an index in [0, high) on rank 0 and broadcast it."""
        idx = torch.randint(0, high, (1,), device=self.device, dtype=torch.long)
        if world_size() > 1:
            torch.distributed.broadcast(idx, src=0)
        return int(idx.item())

    def _sample_rollout_steps(self) -> int:
        """Sample this iteration's rollout NFE, as the reference does
        (``random.choice(num_inference_steps_list)``, rank-0 broadcast in both
        the generator and fake-score updates). Falls back to the fixed
        ``student_sample_steps`` when no list is set.
        """
        steps_list = self.config.student_sample_steps_list
        if not steps_list:
            return int(self.config.student_sample_steps)
        return int(steps_list[self._broadcast_choice(len(steps_list))])

    def _rollout_t_list(self, num_steps: int) -> torch.Tensor:
        """Shifted timestep schedule for ``num_steps`` rollout steps.

        Equivalent to the reference scheduler's ``set_timesteps``: a uniform grid
        mapped through ``shift * s / (1 + (shift - 1) * s)``, first entry clamped
        to ``max_t``. Computed here rather than read from ``sample_t_cfg.t_list``
        because the rollout NFE varies per iteration; that list stays the fixed
        *validation* schedule used at ``student_sample_steps``.
        """
        ns = self.net.noise_scheduler
        shift = self.sample_t_cfg.shift if self.sample_t_cfg.time_dist_type in SHIFTED_TIME_DIST_TYPES else 1.0
        s = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64, device=self.device)
        s = shift * s / (1 + (shift - 1) * s)
        return s.clamp(max=float(ns.max_t)).to(ns.t_precision)

    def gen_data_from_net(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> torch.Tensor:
        """Flow-map rollout compressed into at most three network forwards.

        Mirrors the reference ``WanAnyFlowPipeline.training_rollout``: on a
        ``num_steps``-step schedule ``t_0 > ... > t_N = 0`` with a sampled fine-step
        position ``g``, jump ``t_0 -> t_g``, take the fine step ``t_g -> t_{g+1}``,
        then jump ``t_{g+1} -> 0``, each one forward of mean-velocity sampling
        ``u_theta(x_t, t, r=t_next)``. Gradient flows through all segments; the
        fake-score update wraps this call in ``no_grad`` at the caller.
        """
        del t_student  # the rollout schedule is built below

        num_steps = self._sample_rollout_steps()
        if num_steps < 1:
            raise ValueError(f"rollout steps must be >= 1, got {num_steps}")
        grad_step = self._broadcast_choice(num_steps)
        t_list = self._rollout_t_list(num_steps)

        # The leading jump exists only for grad_step > 0 and the trailing one
        # only for grad_step + 1 < num_steps.
        seg_t = [t_list[0]] if grad_step > 0 else []
        seg_t += [t_list[grad_step], t_list[grad_step + 1]]
        if grad_step + 1 < num_steps:
            seg_t.append(t_list[-1])

        return self._student_sample_loop(
            self.net,
            input_student,
            t_list=torch.stack(seg_t),
            condition=condition,
            student_sample_type="ode",
        )

    # ------------------------------------------------------------------
    # Training step — DMD2 plus the co-trained Stage-1 flow-map loss
    # ------------------------------------------------------------------

    def single_train_step(
        self, data: "Dict[str, Any]", iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, "torch.Tensor | Callable"]]:
        real_data, condition, neg_condition = self._prepare_training_data(data)
        self._setup_grad_requirements(iteration)
        input_student, t_student, t, eps = self._generate_noise_and_time(real_data, iteration=iteration)

        if iteration % self.config.student_update_freq == 0:
            loss_map, outputs = self._student_update_step(
                input_student, t_student, t, eps, data, condition=condition, neg_condition=neg_condition
            )
            if self.config.cotrain_pretrain_weight > 0:
                # Reference cotrain_forward_kl: every generator update also
                # runs the full Stage-1 bidirection (flow-map) loss on the
                # real batch.
                t_mf, r_mf, r_eq_t_mask = self._sample_t_r_buckets(real_data.shape[0])
                mf_outputs = self._compute_mf_loss(
                    real_data=real_data,
                    t=t_mf,
                    r=r_mf,
                    iteration=iteration,
                    condition=condition,
                    neg_condition=neg_condition,
                )
                bidirection_loss = self._reduce_mf_loss(mf_outputs[0], r_eq_t_mask)
                loss_map["bidirection_loss"] = bidirection_loss
                loss_map["total_loss"] = loss_map["total_loss"] + self.config.cotrain_pretrain_weight * bidirection_loss
            return loss_map, outputs

        return self._fake_score_discriminator_update_step(
            input_student, t_student, t, eps, real_data, condition=condition
        )
