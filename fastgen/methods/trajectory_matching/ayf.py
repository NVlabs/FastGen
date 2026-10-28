# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Align Your Flow (AYF) distillation, EMD variant.

Distills a flow-map student ``u(x_t, t, r)`` from a base flow-matching model
(``r`` is the repo-wide second timestep; the paper calls it ``s``). Per pair t >= r:

    v_t        = teacher velocity at t (CFG-guided, no ``r``)
    model_pred = student velocity at (t, r)
    jvp_out    = d/d(tangent) model_pred, tangent = (v_t, 1, 0)
    emd_target = -(v_t - model_pred + tangent_scale * (r - t) * jvp_out)
    loss       = (model_pred - sg(model_pred) + normalize(emd_target).detach()) ** 2

``normalize(x) = x / clamp(RMS(x), min=rms_clamp_min)`` per sample, so the backward
gradient is ``2 * normalize(emd_target)``.

The teacher follows ``model.net_adapter``: a separate frozen network by default, or
the same network with adapters disabled for a LoRA student (the FLUX/Qwen recipe).
The sampled guidance drives it embedded (FLUX) or as a true-CFG scale (Qwen), per
``teacher_guidance_type``.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import torch

from fastgen.methods import MeanFlowModel
from fastgen.utils.tensor import expand_like, to_high_precision
from fastgen.utils.tree_utils import assert_same
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance

if TYPE_CHECKING:
    from fastgen.configs.methods.config_ayf import ModelConfig


class AYFModel(MeanFlowModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tangent_scale = config.loss_config.tangent_scale
        self.rms_clamp_min = config.loss_config.rms_clamp_min
        self.teacher_guidance_range = config.teacher_guidance_range
        self.teacher_guidance_type = config.teacher_guidance_type
        if self.teacher_guidance_type not in ("embedded", "cfg"):
            raise ValueError(f"bad teacher_guidance_type: {config.teacher_guidance_type}")
        self.guidance_distill_warmup_steps = config.guidance_distill_warmup_steps

    # ------------------------------------------------------------------
    # Teacher velocity
    # ------------------------------------------------------------------

    def _sample_teacher_guidance(self, batch_size: int) -> Optional[torch.Tensor]:
        """One guidance value per sample, shared by teacher, JVP, and student forwards.
        ``None`` (range unset / net has no guidance_scale) means "use the net's default"."""
        if self.teacher_guidance_range is None or getattr(self.net, "guidance_scale", None) is None:
            return None
        lo, hi = self.teacher_guidance_range
        return torch.empty(batch_size, device=self.device).uniform_(lo, hi)

    def _cfg_at(self, scale: float | torch.Tensor) -> GuidanceConfig:
        """The model's ``guidance_cfg`` recipe with only the CFG scale replaced
        (``Tensor[B]`` scales broadcast through ``apply_guidance``)."""
        guidance_cfg = copy.deepcopy(self.config.guidance_cfg)
        guidance_cfg.cfg.scale = scale if isinstance(scale, torch.Tensor) else float(scale)
        return guidance_cfg

    @torch.no_grad()
    def _teacher_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``v_t``: the teacher's guided flow prediction (no ``r``).
        "embedded" feeds ``guidance`` to the teacher's embedded input; "cfg" runs
        true CFG with it as the scale."""
        guidance_cfg = self.config.guidance_cfg
        fwd_kwargs: Dict[str, Any] = {}
        if self.teacher_guidance_type == "cfg":
            if guidance is not None:
                guidance_cfg = self._cfg_at(guidance)
        elif guidance is not None:
            fwd_kwargs["guidance"] = guidance

        return predict_with_guidance(
            self.teacher,
            x_t,
            t,
            guidance_cfg=guidance_cfg,
            condition=condition,
            neg_condition=neg_condition,
            skip_inactive=False,  # DDP-safe: uniform forwards across ranks
            fwd_pred_type="flow",
            **fwd_kwargs,
        )

    # ------------------------------------------------------------------
    # EMD loss
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Divide by the per-sample RMS, clamped at ``rms_clamp_min``.
        Performs a bit better than the paper's ``x / (||x|| + 0.1)`` normalization.
        """
        dims = list(range(1, x.ndim))
        rms = to_high_precision(x).pow(2).mean(dim=dims, keepdim=True).sqrt()
        return x / rms.clamp(min=self.rms_clamp_min).detach().to(x.dtype)

    def _emd_pred_to_loss(
        self,
        model_pred: torch.Tensor,
        v_t: torch.Tensor,
        jvp_out: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        iteration: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Checked before the cast, since a cast made under no_grad never requires grad.
        assert model_pred.requires_grad is torch.is_grad_enabled(), "model_pred requires grad iff grad enabled"
        model_pred, v_t, jvp_out = to_high_precision(model_pred, v_t, jvp_out)
        t, r = to_high_precision(t, r)
        r_minus_t = expand_like((r - t).to(dtype=model_pred.dtype), model_pred)

        assert not any(x.requires_grad for x in [v_t, jvp_out, r_minus_t]), "targets must not require gradients"

        # Optional JVP-term ramp-in (the reference uses none).
        if self.loss_config.tangent_warmup_steps > 0:
            warmup = min(1.0, iteration / self.loss_config.tangent_warmup_steps)
        else:
            warmup = 1.0

        emd_target = -(v_t - model_pred + warmup * self.tangent_scale * r_minus_t * jvp_out)
        clipped = self.normalize(emd_target).detach()

        # Stop-gradient form: value == clipped**2, gradient == 2 * clipped. MEAN
        # reduction keeps the loss O(1) (sum over C*H*W would inflate the gradient
        # by numel and change the effective lr).
        loss = (model_pred - model_pred.detach() + clipped).pow(2)
        loss = torch.mean(loss, dim=list(range(1, loss.ndim)))

        # Detached target + per-sample normalization weight (monitoring).
        emd_target = emd_target.detach()
        rms = emd_target.flatten(1).pow(2).mean(dim=-1).sqrt()
        weight = 1.0 / rms.clamp(min=self.rms_clamp_min)

        assert not emd_target.requires_grad and not weight.requires_grad
        return loss, emd_target, weight

    def _compute_emd_loss(
        self,
        real_data: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        iteration: int,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        skip_jvp: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        noise = torch.randn_like(real_data)
        x_t = self.net.noise_scheduler.forward_process(real_data, noise, t)

        guidance = self._sample_teacher_guidance(real_data.shape[0])
        guidance_kwargs = {} if guidance is None else {"guidance": guidance}
        v_t = self._teacher_velocity(x_t, t, condition=condition, neg_condition=neg_condition, guidance=guidance)
        if skip_jvp:
            # Warmup: with r = t the (r - t) * jvp term vanishes; skip it.
            jvp_out = torch.zeros_like(v_t)
        else:
            jvp_out = self._jvp(x_t, t, r, v_t, condition=condition, fwd_kwargs=guidance_kwargs)
        assert not jvp_out.requires_grad, "jvp_out must not require gradients"

        # Student forward with gradients. During warmup no ``r`` is passed, so the
        # zero-init r_embedder stays at its no-op init until flow-map training.
        student_r = None if skip_jvp else r
        # The teacher pass (a shared-base teacher runs the student's trainable base modules) and a
        # finite-difference JVP ran under no_grad; drop their detached autocast casts so this forward makes
        # its own.
        torch.clear_autocast_cache()
        model_pred = self.net(x_t, t, r=student_r, condition=condition, fwd_pred_type="flow", **guidance_kwargs)
        assert_same(dtype=True, device=True, x_t=x_t, model_pred=model_pred)
        emd_loss, emd_target, weight = self._emd_pred_to_loss(model_pred, v_t, jvp_out, t, r, iteration)

        # Flow-matching loss and x0 approximation (monitoring / reward only); the latter is the flow-map jump to
        # time 0, with u(t, r) in place of u(t, 0).
        cond_velocity = self.net.noise_scheduler.cond_velocity(real_data, noise, t)
        v_loss = torch.mean((model_pred - cond_velocity) ** 2, dim=tuple(range(1, model_pred.ndim)))
        x0_approx = x_t - expand_like(t.to(x_t.dtype), x_t) * model_pred.detach()
        return emd_loss, v_loss, jvp_out, emd_target, v_t, x0_approx, weight

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        real_data = data["real"]
        condition = data.get("condition")
        neg_condition = data.get("neg_condition")
        batch_size = real_data.shape[0]

        t, r, _ = self._sample_t_r_buckets(batch_size)

        # Guidance-distillation warmup: r = t degenerates the EMD loss to
        # guided-velocity distillation; the JVP is skipped.
        distill_only = iteration <= self.guidance_distill_warmup_steps
        if distill_only:
            r = t

        emd_loss, v_loss, jvp_out, emd_target, v_t, x0_approx, weight = self._compute_emd_loss(
            real_data=real_data,
            t=t,
            r=r,
            iteration=iteration,
            condition=condition,
            neg_condition=neg_condition,
            skip_jvp=distill_only,
        )

        loss = emd_loss.mean()
        emd_target = to_high_precision(emd_target)
        loss_map = {
            "total_loss": loss,
            "emd_loss": loss,
            "emd_target": emd_target.pow(2).mean(),
            "emd_target_norm": emd_target.pow(2).mean(dim=list(range(1, emd_target.ndim))).sqrt().mean(),
            "v_loss": v_loss.mean(),
            "jvp_norm": torch.linalg.vector_norm(jvp_out.flatten(1), dim=-1).mean(),
            "v_norm": torch.linalg.vector_norm(v_t.flatten(1), dim=-1).mean(),
            "loss_weight": weight.mean(),
        }
        if self.guidance_distill_warmup_steps > 0:
            loss_map["guidance_distill_phase"] = torch.tensor(float(distill_only), device=self.device)
        outputs = self._get_outputs(x0_approx, condition=condition)
        loss_map = self.apply_reward(x0_approx, data, loss_map)
        return loss_map, outputs

    # Few-step sampling runs MeanFlow's flow-map step, FlowMapLossMixin._advance_student_step.
