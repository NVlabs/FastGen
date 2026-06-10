# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AnyFlow — any-step video diffusion with flow maps and on-policy distillation.

AnyFlow trains a single flow-map model :math:`u_\\theta(x_t, t, r)` that
predicts the average velocity from time ``t`` to ``r`` (with ``r \\le t``).
Once trained, the same model supports arbitrary inference step counts: each
Euler-like sampling step picks its own integration interval
``(t \\rightarrow r)``.

Training has two stages:

* **Flow-map pretrain** (paper Stage 2) is MeanFlow with AnyFlow's
  hyperparameters — fixed ``beta08`` per-timestep loss weighting,
  finite-difference JVP, a ``consistency_ratio`` fraction of the batch pinned
  to ``r = min_t``, and shifted timestep sampling. It is configured directly
  on :class:`~fastgen.methods.consistency_model.mean_flow.MeanFlowModel`
  (see ``configs/experiments/WanT2V/config_anyflow.py``); there is no
  AnyFlow-specific pretrain code.

* **On-policy** (paper Stage 3, this module) is DMD2 on top of the pretrained
  flow-map weights. The only differences from stock DMD2 are the two narrow
  overrides below:

  - :meth:`AnyFlowModel._generate_noise_and_time` — the student always starts
    from pure noise at ``max_t`` (DMD2's multi-step branch would noise real
    data instead);
  - :meth:`AnyFlowModel.gen_data_from_net` — the student generates via a
    multi-step Euler-flow rollout with ``r = t_next`` (mean-velocity sampling)
    and gradient enabled at one randomly-chosen step, matching AnyFlow's
    ``WanAnyFlowPipeline.training_rollout``.

  The teacher and fake_score are flow-map networks queried at the
  instantaneous velocity ``r = t`` — for gated-fusion Wan networks this is the
  network-level default when ``r`` is not passed (see
  ``fastgen/networks/Wan/network.py``), so DMD2's student / fake-score /
  discriminator update steps are reused unchanged.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import torch

from fastgen.methods.consistency_model.mean_flow import MeanFlowModel
from fastgen.methods.distribution_matching.dmd2 import DMD2Model
from fastgen.utils.basic_utils import convert_cfg_to_dict
import fastgen.utils.logging_utils as logger


if TYPE_CHECKING:
    from fastgen.configs.methods.config_anyflow import ModelConfig


class AnyFlowModel(DMD2Model):
    """AnyFlow on-policy stage: DMD2 with a multi-step rollout-with-gradient student."""

    # Validation sampling integrates the flow map with r = t_next (ode) just
    # like MeanFlow; FastGenModel's default x0-prediction loop never passes r.
    _student_sample_loop = MeanFlowModel._student_sample_loop

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        logger.info(
            f"AnyFlow on-policy: student_sample_steps={self.config.student_sample_steps}, "
            f"student_update_freq={self.config.student_update_freq}, "
            f"gan_loss_weight_gen={self.config.gan_loss_weight_gen}"
        )

    def _sample_grad_step(self, num_steps: int) -> int:
        """Pick one rollout step index in ``[0, num_steps - 1]`` to enable gradients on.

        Broadcast from rank 0 in distributed runs so all ranks agree on the
        same window — matches AnyFlow's reference (``training_rollout`` in
        ``pipeline_wan_anyflow.py``, ``broadcast(sample_step, src=0)``).
        """
        idx = torch.randint(0, num_steps, (1,), device=self.device, dtype=torch.long)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(idx, src=0)
        return int(idx.item())

    def _generate_noise_and_time(
        self, real_data: torch.Tensor
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

        t = self.net.noise_scheduler.sample_t(
            batch_size, **convert_cfg_to_dict(self.config.sample_t_cfg), device=self.device
        )
        eps = torch.randn_like(real_data, device=self.device, dtype=real_data.dtype)
        return input_student, t_student, t, eps

    def gen_data_from_net(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> torch.Tensor:
        """Multi-step Euler-flow rollout with one gradient-enabled step.

        Mirrors AnyFlow's ``WanAnyFlowPipeline.training_rollout``: runs
        ``student_sample_steps`` Euler-flow updates with ``r = t_next``
        (mean-velocity sampling, AnyFlow's ``use_mean_velocity=True`` default)
        starting from ``input_student`` (pure noise at ``max_t``), and enables
        gradients at one randomly-chosen step so the DMD generator update
        receives a gradient through one full denoising forward. When the
        caller wraps this in ``no_grad`` (the fake-score / discriminator
        update), all steps run gradient-free.
        """
        del t_student  # the rollout schedule comes from t_list below
        num_steps = int(self.config.student_sample_steps)
        if num_steps < 1:
            raise ValueError(f"student_sample_steps must be >= 1, got {num_steps}")

        ns = self.net.noise_scheduler
        batch_size = input_student.shape[0]
        grad_step = self._sample_grad_step(num_steps)

        # Timestep schedule. Use config-provided t_list when set (matches
        # AnyFlow's hand-tuned step lists, e.g. [0.999, 0.937, 0.833, 0.624, 0.0]
        # for 4-step Wan); otherwise fall back to the scheduler's default.
        if self.config.sample_t_cfg.t_list is not None:
            t_list = torch.tensor(self.config.sample_t_cfg.t_list, device=self.device, dtype=ns.t_precision)
            if len(t_list) != num_steps + 1:
                raise ValueError(
                    f"sample_t_cfg.t_list has {len(t_list)} entries, "
                    f"expected {num_steps + 1} for student_sample_steps={num_steps}"
                )
        else:
            t_list = ns.get_t_list(sample_steps=num_steps, device=self.device)

        x = input_student
        for step in range(num_steps):
            t_cur = t_list[step].expand(batch_size).to(ns.t_precision)
            t_next = t_list[step + 1].expand(batch_size).to(ns.t_precision)

            enable_grad = (step == grad_step) and torch.is_grad_enabled()
            with torch.set_grad_enabled(enable_grad):
                # Mean-velocity flow prediction: u_theta(x_t, t, r=t_next)
                flow_pred = self.net(x, t_cur, r=t_next, condition=condition, fwd_pred_type="flow")

            # Euler-flow step. Steps run under no_grad add no graph nodes, and
            # keeping ``x`` attached preserves the gradient installed by the
            # grad-enabled step.
            delta_t = (t_cur - t_next).view(batch_size, *([1] * (x.ndim - 1))).to(x.dtype)
            x = x - delta_t * flow_pred

        return x
