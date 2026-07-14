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
  finite-difference JVP, prediction-side guidance fusion
  (``guidance_fuse_scale``), global rebalancing of non-diffusion losses
  (``rebalance_to_diffusion``), and a ``consistency_ratio`` fraction of the
  batch pinned to ``r = 0``. It is configured directly on
  :class:`~fastgen.methods.consistency_model.mean_flow.MeanFlowModel`
  (see ``configs/experiments/WanT2V/config_anyflow.py``); there is no
  AnyFlow-specific pretrain code.

* **On-policy** (paper Stage 3, this module) is DMD2 on top of the pretrained
  flow-map weights, with the reference's three deviations from stock DMD2:

  - the student generates via a flow-map rollout compressed into at most three
    network forwards — one jump from ``t_0`` to ``t_g``, one fine step
    ``t_g \\rightarrow t_{g+1}``, one jump to 0 — with the inference step
    count sampled per iteration from ``student_sample_steps_list`` and the
    fine-step position ``g`` sampled uniformly (both rank-0 broadcast).
    Gradient flows through ALL segments
    (:meth:`AnyFlowModel.gen_data_from_net`, mirroring
    ``WanAnyFlowPipeline.training_rollout``);
  - the student always starts from pure noise at ``max_t``
    (:meth:`AnyFlowModel._generate_noise_and_time`);
  - every student update co-trains the Stage-2 flow-map loss on the real
    batch (``cotrain_pretrain_weight``, mirroring the reference's
    ``cotrain_forward_kl``), borrowing MeanFlow's loss machinery.

  The teacher and fake_score are flow-map networks queried at the
  instantaneous velocity ``r = t`` — for gated-fusion Wan networks this is
  the network-level default when ``r`` is not passed (see
  ``fastgen/networks/Wan/network.py``), so DMD2's update steps run unchanged.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import torch

from fastgen.methods.consistency_model.mean_flow import MeanFlowModel
from fastgen.methods.distribution_matching.dmd2 import DMD2Model
from fastgen.utils import basic_utils, expand_like
from fastgen.utils.basic_utils import convert_cfg_to_dict
from fastgen.utils.distributed import world_size
import fastgen.utils.logging_utils as logger


if TYPE_CHECKING:
    from typing import Dict, Callable

    from fastgen.configs.methods.config_anyflow import ModelConfig


class AnyFlowModel(DMD2Model):
    """AnyFlow on-policy stage: DMD2 with a flow-map rollout student."""

    # Validation sampling integrates the flow map with r = t_next (ode) just
    # like MeanFlow; FastGenModel's default x0-prediction loop never passes r.
    _student_sample_loop = MeanFlowModel._student_sample_loop

    # MeanFlow loss machinery borrowed for the co-trained Stage-2 flow-map
    # loss (the reference's cotrain_forward_kl runs the full bidirection loss
    # at every generator step).
    _sample_t_r_buckets = MeanFlowModel._sample_t_r_buckets
    _reduce_mf_loss = MeanFlowModel._reduce_mf_loss
    _compute_mf_loss = MeanFlowModel._compute_mf_loss
    _drop_condition = MeanFlowModel._drop_condition
    _jvp = MeanFlowModel._jvp
    _estimate_jvp_finite_difference = MeanFlowModel._estimate_jvp_finite_difference
    _mf_pred_to_loss = MeanFlowModel._mf_pred_to_loss
    _compute_weight = MeanFlowModel._compute_weight
    _timestep_weight_raw = MeanFlowModel._timestep_weight_raw
    _get_timestep_weight = MeanFlowModel._get_timestep_weight

    @torch.no_grad()
    def _get_velocity(self, x, z, t, condition=None, neg_condition=None):
        """Raw data velocity for the co-trained flow-map loss.

        The DMD teacher CFG (``config.guidance_scale``) must not leak into the
        co-trained loss — AnyFlow's guidance lives in
        ``loss_config.guidance_fuse_scale`` (prediction-side fusion, handled
        inside ``_compute_mf_loss``), so MeanFlow's target-side eq. 19 path is
        deliberately not used here.
        """
        del neg_condition
        return condition, self.net.noise_scheduler.cond_velocity(x=x, eps=z, t=t)

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        # Attributes required by the borrowed MeanFlow methods.
        self.sample_t_cfg = self.config.sample_t_cfg
        self.sample_r_cfg = self.config.sample_r_cfg
        self.loss_config = self.config.loss_config
        if self.config.precision_amp_jvp is None or self.config.precision_amp_jvp == self.precision_amp:
            self.precision_amp_jvp = None
        else:
            self.precision_amp_jvp = basic_utils.PRECISION_MAP[self.config.precision_amp_jvp]
        self._timestep_weight_scale = None

        logger.info(
            f"AnyFlow on-policy: student_sample_steps_list={self.config.student_sample_steps_list}, "
            f"student_update_freq={self.config.student_update_freq}, "
            f"cotrain_pretrain_weight={self.config.cotrain_pretrain_weight}, "
            f"gan_loss_weight_gen={self.config.gan_loss_weight_gen}"
        )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _broadcast_choice(self, high: int) -> int:
        """Pick an index in [0, high) on rank 0 and broadcast it."""
        idx = torch.randint(0, high, (1,), device=self.device, dtype=torch.long)
        if world_size() > 1:
            torch.distributed.broadcast(idx, src=0)
        return int(idx.item())

    def _sample_rollout_steps(self) -> int:
        """Sample the rollout NFE for this iteration.

        Matches the reference: ``random.choice(num_inference_steps_list)``
        broadcast from rank 0 in both the generator and fake-score updates.
        Falls back to the fixed ``student_sample_steps`` when no list is set.
        """
        steps_list = self.config.student_sample_steps_list
        if not steps_list:
            return int(self.config.student_sample_steps)
        return int(steps_list[self._broadcast_choice(len(steps_list))])

    def _rollout_t_list(self, num_steps: int) -> torch.Tensor:
        """Shifted timestep schedule for ``num_steps`` rollout steps.

        Equivalent to the reference scheduler's ``set_timesteps``: a uniform
        grid mapped through ``shift * s / (1 + (shift - 1) * s)``. The first
        entry is clamped to the scheduler's ``max_t``. A config-provided
        ``t_list`` takes precedence when its length matches.
        """
        ns = self.net.noise_scheduler
        cfg_t_list = self.config.sample_t_cfg.t_list
        if cfg_t_list is not None and len(cfg_t_list) == num_steps + 1:
            return torch.tensor(cfg_t_list, device=self.device, dtype=ns.t_precision)

        shift = self.sample_t_cfg.shift if self.sample_t_cfg.time_dist_type == "shifted" else 1.0
        s = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64, device=self.device)
        s = shift * s / (1 + (shift - 1) * s)
        return s.clamp(max=float(ns.max_t)).to(ns.t_precision)

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
        """Flow-map rollout compressed into at most three network forwards.

        Mirrors the reference ``WanAnyFlowPipeline.training_rollout``: with an
        ``num_steps``-step schedule ``t_0 > ... > t_N = 0`` and a sampled fine
        step position ``g``, the model jumps ``t_0 -> t_g`` in ONE flow-map
        forward, takes the single fine step ``t_g -> t_{g+1}``, then jumps
        ``t_{g+1} -> 0`` in one forward. Gradient flows through all segments
        (the reference applies no ``no_grad`` inside ``training_rollout``);
        the fake-score update wraps this call in ``no_grad`` at the caller.
        Each step uses mean-velocity sampling ``u_theta(x_t, t, r=t_next)``.
        """
        del t_student  # the rollout schedule is built below
        ns = self.net.noise_scheduler
        batch_size = input_student.shape[0]

        num_steps = self._sample_rollout_steps()
        if num_steps < 1:
            raise ValueError(f"rollout steps must be >= 1, got {num_steps}")
        grad_step = self._broadcast_choice(num_steps)
        t_list = self._rollout_t_list(num_steps)

        segments = (
            (t_list[0], t_list[grad_step]),
            (t_list[grad_step], t_list[grad_step + 1]),
            (t_list[grad_step + 1], t_list[-1]),
        )

        x = input_student
        for t_cur, t_next in segments:
            if float(t_cur) == float(t_next):
                continue
            t_batch = t_cur.expand(batch_size).to(ns.t_precision)
            r_batch = t_next.expand(batch_size).to(ns.t_precision)
            flow_pred = self.net(x, t_batch, r=r_batch, condition=condition, fwd_pred_type="flow")
            x = x - expand_like(t_batch - r_batch, x).to(x.dtype) * flow_pred

        return x

    # ------------------------------------------------------------------
    # Training step — DMD2 plus the co-trained Stage-2 flow-map loss
    # ------------------------------------------------------------------

    def single_train_step(
        self, data: "Dict[str, Any]", iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, "torch.Tensor | Callable"]]:
        real_data, condition, neg_condition = self._prepare_training_data(data)
        self._setup_grad_requirements(iteration)
        input_student, t_student, t, eps = self._generate_noise_and_time(real_data)

        if iteration % self.config.student_update_freq == 0:
            loss_map, outputs = self._student_update_step(
                input_student, t_student, t, eps, data, condition=condition, neg_condition=neg_condition
            )
            if self.config.cotrain_pretrain_weight > 0:
                # Reference cotrain_forward_kl: every generator update also
                # runs the full Stage-2 bidirection (flow-map) loss on the
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
