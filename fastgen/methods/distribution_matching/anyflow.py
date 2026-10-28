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

from fastgen.methods.trajectory_matching.mean_flow import FlowMapLossMixin
from fastgen.methods.distribution_matching.dmd2 import DMD2Model, Generation, TrainingInputs
from fastgen.utils.distributed.collectives import broadcast_choice
from fastgen.utils.basic_utils import safe_update
from fastgen.utils.tree_utils import noise_from_shape
import fastgen.utils.logging_utils as logger


if TYPE_CHECKING:
    from typing import Dict, Callable

    from fastgen.configs.methods.config_anyflow import ModelConfig


class AnyFlowModel(FlowMapLossMixin, DMD2Model):
    """AnyFlow on-policy stage: DMD2 with a flow-map rollout student.

    ``FlowMapLossMixin`` supplies the co-trained Stage-1 objective and the
    flow-map student-step hook, which integrates with ``r = t_next`` through
    the shared sample loop.

    AnyFlow requires ``backward_simulation="default"`` because its student always
    starts from fresh noise. Unlike base DMD2, it completes the rollout in the
    current iteration and does not emit cross-iteration carry state. Real data
    is required only when the optional co-trained flow-map objective is active.
    """

    # The flow-map rollout's clean endpoint is the generation.
    student_generations = ("x0",)

    def __init__(self, config: ModelConfig):
        if config.backward_simulation != "default":
            raise ValueError(
                "AnyFlowModel requires backward_simulation='default': it always starts from fresh noise "
                "and performs its complete flow-map rollout within the current iteration, carrying nothing."
            )
        if config.max_steps_rollout <= 0:
            raise ValueError(f"max_steps_rollout must be positive, got {config.max_steps_rollout}.")
        super().__init__(config)
        # The co-trained flow-map loss draws its (t, r) from its own config, not from
        # `sample_t_cfg` -- DMD2 keeps that one for the noising time.
        self._init_flow_map_loss(config.cotrain_sample_t_cfg, config.cotrain_sample_r_overrides)

        logger.info(
            f"AnyFlow on-policy: student_steps_list={self.config.student_steps_list}, "
            f"max_steps_rollout={self.config.max_steps_rollout}, "
            f"student_update_freq={self.config.student_update_freq}, "
            f"cotrain_pretrain_weight={self.config.cotrain_pretrain_weight}, "
            f"gan_loss_weight_gen={self.config.gan_loss_weight_gen}"
        )

    def init_optimizers(self):
        super().init_optimizers()
        # Runs after the trainer loads trainer.checkpointer.pretrained_ckpt_path, which sets resume_iter.
        if not hasattr(self, "resume_iter"):
            logger.warning(
                "AnyFlow's on-policy stage starts without its Stage-1 flow-map checkpoint; set "
                "trainer.checkpointer.pretrained_ckpt_path to it, whose iteration also offsets the EMA start."
            )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout_paths(self, num_steps: int) -> list[list[int]]:
        """Enumerate the bounded windows of one inference grid."""
        max_steps_rollout = self.config.max_steps_rollout
        if max_steps_rollout == 1 or num_steps == 1:
            return [[0, num_steps]]
        if max_steps_rollout == 2:
            return [[0, split, num_steps] for split in range(1, num_steps)]

        # Expose consecutive fine steps and collapse the optional prefix and
        # suffix into one forward each. A cap of 3 reproduces the reference
        # prefix / selected fine step / suffix construction.
        num_exposed_steps = min(max_steps_rollout - 2, num_steps)
        # Neighbouring starts share one window once it spans every interior node; keep each window once, in order.
        exposed_windows = dict.fromkeys(
            tuple(step_idx for step_idx in range(start, start + num_exposed_steps + 1) if 0 < step_idx < num_steps)
            for start in range(num_steps - num_exposed_steps + 1)
        )
        return [[0, *window, num_steps] for window in exposed_windows]

    def _training_inputs(self, data: "Dict[str, Any]", iteration: int = 1) -> TrainingInputs:
        """Construct data-free inputs for AnyFlow's in-iteration rollout.

        Shape and dtype come from the model configuration. Real data, when
        present, is retained only for the optional GAN and flow-map co-training
        losses. Returning no student step or trajectory step expresses
        that the rollout constructs its complete schedule in-iteration and
        carries no state across iterations.
        """
        del iteration
        batch_size = self._batch_size(data)
        noise = noise_from_shape(self.input_shape, batch_size, self.device, self.precision)
        student_grid = self.net.noise_scheduler.student_grid(device=self.device)
        input_student = self.net.noise_scheduler.latents(noise=noise, t_init=student_grid[0])
        t, eps = self._sample_t_and_eps(batch_size)
        return TrainingInputs(input_student=input_student, student_step=None, t=t, eps=eps)

    def gen_data_from_net(
        self,
        input_student: torch.Tensor,
        student_step: Optional[torch.Tensor],
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> Generation:
        """Flow-map rollout bounded by ``max_steps_rollout`` network calls.

        On a ``num_steps``-step schedule ``t_0 > ... > t_N = 0``, a contiguous
        window of fine steps is exposed while its optional prefix and suffix
        are collapsed into jumps. The default cap of 3 mirrors the reference
        ``WanAnyFlowPipeline.training_rollout``. Gradient flows through every
        segment; the fake-score update wraps this call in ``no_grad``.
        """
        del student_step, neg_condition  # the complete rollout schedule is built below

        # Draw the training rollout NFE (rank-0 broadcast in both the
        # generator and fake-score updates).
        steps_list = self.config.student_steps_list
        if steps_list:
            num_steps = steps_list[broadcast_choice(len(steps_list), device=self.device)]
        else:
            num_steps = self.net.noise_scheduler.student_grid_cfg.steps
        student_grid = self.net.noise_scheduler.student_grid(num_steps, device=self.device)
        paths = self._rollout_paths(num_steps)
        step_indices = paths[0] if len(paths) == 1 else paths[broadcast_choice(len(paths), device=self.device)]
        sample_grid = student_grid[step_indices]

        gen_data = self._student_sample_loop(
            self.net,
            input_student,
            student_grid=sample_grid,
            condition=condition,
        )
        generation_metrics = {
            "student_steps": torch.tensor(float(num_steps), device=self.device),
            "num_rollout_forwards": torch.tensor(float(len(step_indices) - 1), device=self.device),
        }
        return Generation(data=gen_data, x0=gen_data, metrics=generation_metrics)

    # ------------------------------------------------------------------
    # Training step — DMD2 plus the co-trained Stage-1 flow-map loss
    # ------------------------------------------------------------------

    def single_train_step(
        self, data: "Dict[str, Any]", iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, "torch.Tensor | Callable"]]:
        real_data = data.get("real")
        is_student_update = iteration % self.config.student_update_freq == 0
        if self.config.cotrain_pretrain_weight > 0 and real_data is None:
            raise ValueError(
                "AnyFlow cotrain_pretrain_weight > 0 requires data['real']; set "
                "cotrain_pretrain_weight=0 for data-free training."
            )

        loss_map, outputs = super().single_train_step(data, iteration)
        if is_student_update and self.config.cotrain_pretrain_weight > 0:
            assert real_data is not None
            # Reference cotrain_forward_kl: every generator update also runs
            # the full Stage-1 flow-map loss on the real batch.
            cotrain_loss_map = self._flow_map_loss_step(data, iteration).loss_map
            cotrain_metrics = {"cotrain_loss": cotrain_loss_map.pop("total_loss")}
            safe_update(cotrain_metrics, {f"cotrain_{key}": value for key, value in cotrain_loss_map.items()})
            safe_update(loss_map, cotrain_metrics)
            loss_map["total_loss"] = (
                loss_map["total_loss"] + self.config.cotrain_pretrain_weight * loss_map["cotrain_loss"]
            )
        return loss_map, outputs
