# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AnyFlow — any-step video diffusion with flow maps and on-policy distillation.

AnyFlow trains a single model :math:`u_\\theta(x_t, t, r)` that predicts the
average velocity from time ``t`` to ``r`` (with ``r \\le t``). Once trained,
the same model supports arbitrary inference step counts: each Euler-like
sampling step picks its own integration interval ``(t \\rightarrow r)``.

Training has two stages, selected via ``config.loss_config.training_stage``:

* ``"pretrain"`` — flow-map prediction with a central-difference target

      v(x_t, t) = eps - x_0                  # instantaneous flow (flow matching)
      dF/dt    ~=  (u_theta(x_{t+d}, t+d, r) - u_theta(x_{t-d}, t-d, r)) / 2d
      target   =  v - (t - r) * dF/dt
      loss     =  weight(t) * MSE(u_theta(x_t, t, r), target)

  Per-batch sampling assigns ``r = t`` for a ``diffusion_ratio`` fraction
  (recovering plain flow matching), ``r = 0`` for a ``consistency_ratio``
  fraction (forcing consistency to clean data), and a uniform random pair
  otherwise — matching the AnyFlow paper.

* ``"onpolicy"`` — distribution-matching distillation on top of pretrained
  flow-map weights. Inherits DMD2's fake_score / teacher / discriminator
  machinery and alternating-step optimisation, but conditions all forwards
  on ``r = 0`` (predicting the full flow from ``t`` to clean). Multi-step
  rollout-with-gradient is intentionally deferred to a follow-up PR.

The network must support a secondary timestep argument ``r`` (Wan with
``r_timestep=True`` does; MeanFlow already exercises this same code path).
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastgen.methods.distribution_matching.anyflow_scheduler import FlowMapDiscreteScheduler
from fastgen.methods.distribution_matching.dmd2 import DMD2Model
from fastgen.utils import expand_like
import fastgen.utils.logging_utils as logger


def remap_anyflow_keys(state_dict: dict) -> dict:
    """Remap an AnyFlow HF release state_dict to FastGen's Wan layout.

    AnyFlow's ``FAR_Wan_Transformer3DModel`` stores the r-pathway inside the
    main ``condition_embedder`` as ``delta_embedder``, and uses ONE shared
    ``time_proj`` for both t and (t, r). FastGen exposes a separate top-level
    ``r_embedder`` with its own ``time_embedder`` + ``time_proj``. The two
    layouts are functionally equivalent (FastGen's ``r_embedder.time_proj``
    starts as a deepcopy of ``condition_embedder.time_proj`` per
    :meth:`Wan.init_embedder`), so we just rename / duplicate the tensors.

    The function is a no-op when no ``condition_embedder.delta_embedder.*``
    keys are present, so it's safe to call unconditionally.
    """
    delta_keys = [k for k in state_dict if k.startswith("condition_embedder.delta_embedder.")]
    if not delta_keys:
        return state_dict
    new_sd = dict(state_dict)
    for k in delta_keys:
        # condition_embedder.delta_embedder.linear_1.weight
        #   -> r_embedder.time_embedder.linear_1.weight
        new_k = k.replace("condition_embedder.delta_embedder.", "r_embedder.time_embedder.")
        new_sd[new_k] = new_sd.pop(k)
    # AnyFlow's gated fusion shares the final time_proj. FastGen has a
    # separate r_embedder.time_proj that mathematically substitutes for the
    # shared one when fusion="gated"; copy the weights across so the two
    # projections start identical (and AnyFlow's training never diverges them).
    for sub in ("weight", "bias"):
        src = f"condition_embedder.time_proj.{sub}"
        dst = f"r_embedder.time_proj.{sub}"
        if src in new_sd and dst not in new_sd:
            new_sd[dst] = new_sd[src].clone()
    logger.info(
        f"remap_anyflow_keys: rewrote {len(delta_keys)} delta_embedder tensors "
        "and duplicated time_proj weights into r_embedder."
    )
    return new_sd


if TYPE_CHECKING:
    from fastgen.configs.methods.config_anyflow import ModelConfig


class AnyFlowModel(DMD2Model):
    """AnyFlow training method.

    See module docstring for the algorithm.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.loss_config = self.config.loss_config

        if self.loss_config.training_stage not in ("pretrain", "onpolicy"):
            raise ValueError(
                f"training_stage must be 'pretrain' or 'onpolicy', got {self.loss_config.training_stage!r}"
            )

        # Standalone scheduler used for inference and for the per-timestep
        # training weight in the pretrain stage. Training noising still goes
        # through ``self.net.noise_scheduler`` to stay compatible with DMD2.
        self._flowmap_scheduler = FlowMapDiscreteScheduler(
            num_train_timesteps=self.loss_config.num_train_timesteps,
            shift=self.loss_config.shift,
            weight_type=self.loss_config.weight_type,
        )

        if self.loss_config.training_stage == "pretrain":
            logger.info(
                f"AnyFlow pretrain stage: epsilon={self.loss_config.jvp_finite_diff_eps}, "
                f"diffusion_ratio={self.loss_config.diffusion_ratio}, "
                f"consistency_ratio={self.loss_config.consistency_ratio}, "
                f"weight_type={self.loss_config.weight_type}"
            )
        else:
            logger.info(
                f"AnyFlow on-policy stage: student_update_freq={self.config.student_update_freq}, "
                f"gan_loss_weight_gen={self.config.gan_loss_weight_gen}"
            )

    # ------------------------------------------------------------------
    # Build / optimisation overrides — skip DMD2 plumbing in pretrain
    # ------------------------------------------------------------------

    def build_model(self):
        """In pretrain mode skip fake_score / discriminator entirely."""
        if self.config.loss_config.training_stage == "pretrain":
            # Bypass DMD2Model.build_model — pretrain only needs the student.
            # Call grandparent's build_model (FastGenModel) directly.
            super(DMD2Model, self).build_model()
            self.load_student_weights_and_ema()
            return
        super().build_model()

    def init_optimizers(self):
        """Pretrain skips the DMD2 fake_score / discriminator optimisers."""
        if self.config.loss_config.training_stage == "pretrain":
            # Bypass DMD2Model.init_optimizers — only the student optimiser exists.
            super(DMD2Model, self).init_optimizers()
            return
        super().init_optimizers()

    @property
    def model_dict(self):
        if self.config.loss_config.training_stage == "pretrain":
            return super(DMD2Model, self).model_dict
        return super().model_dict

    @property
    def optimizer_dict(self):
        if self.config.loss_config.training_stage == "pretrain":
            return super(DMD2Model, self).optimizer_dict
        return super().optimizer_dict

    @property
    def scheduler_dict(self):
        if self.config.loss_config.training_stage == "pretrain":
            return super(DMD2Model, self).scheduler_dict
        return super().scheduler_dict

    # ------------------------------------------------------------------
    # Pretrain stage
    # ------------------------------------------------------------------

    def _sample_pair_timesteps(
        self, batch_size: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``(t, r)`` with ``t >= r``, plus a per-sample diffusion mask.

        Implements AnyFlow's partitioning of the batch:
        * a ``diffusion_ratio`` fraction has ``r = t`` (pure flow matching)
        * a ``consistency_ratio`` fraction has ``r = min_t``
        * the rest gets a uniform random pair
        """
        ns = self.net.noise_scheduler
        t_dtype = ns.t_precision

        t_1 = torch.rand(batch_size, device=self.device, dtype=t_dtype)
        t_2 = torch.rand(batch_size, device=self.device, dtype=t_dtype)
        t_norm = torch.maximum(t_1, t_2)
        r_norm = torch.minimum(t_1, t_2)

        # Shift to match the flow-matching schedule (Wan default uses shift=5).
        t_norm = self._flowmap_scheduler.apply_shift(t_norm)
        r_norm = self._flowmap_scheduler.apply_shift(r_norm)

        # Rescale unit-interval timesteps into the noise scheduler's [min_t, max_t].
        max_t = float(ns.max_t)
        min_t = float(ns.min_t)
        scale = max_t - min_t
        t = t_norm * scale + min_t
        r = r_norm * scale + min_t

        # Per-batch bucket assignment. We shuffle so the buckets are randomly
        # distributed within the local batch — this matches the paper's intent
        # without requiring global cross-rank coordination.
        n_diffusion = int(round(self.loss_config.diffusion_ratio * batch_size))
        n_consistency = int(round(self.loss_config.consistency_ratio * batch_size))
        n_diffusion = min(n_diffusion, batch_size)
        n_consistency = min(n_consistency, batch_size - n_diffusion)

        perm = torch.randperm(batch_size, device=self.device)
        is_diffusion = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        is_consistency = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        is_diffusion[perm[:n_diffusion]] = True
        is_consistency[perm[n_diffusion : n_diffusion + n_consistency]] = True

        r = torch.where(is_diffusion, t, r)
        r = torch.where(is_consistency, torch.full_like(r, min_t), r)

        return t.to(dtype=t_dtype), r.to(dtype=t_dtype), is_diffusion

    @torch.no_grad()
    def _compute_central_difference_target(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        condition: Optional[Any],
    ) -> torch.Tensor:
        """Compute the AnyFlow flow-map target ``v - (t - r) * dF/dt``.

        ``dF/dt`` is estimated by central difference at ``(t ± delta, r)``.
        Boundary cases near ``min_t`` / ``max_t`` fall back to one-sided
        differences so the estimate stays valid for the whole timestep range.
        """
        ns = self.net.noise_scheduler
        max_t = float(ns.max_t)
        min_t = float(ns.min_t)
        delta = float(self.loss_config.jvp_finite_diff_eps)

        # Validity masks for each finite-difference direction.
        is_fwd_valid = (t + delta) <= max_t
        is_bwd_valid = ((t - delta) >= min_t) & ((t - delta) > r)

        use_central = is_fwd_valid & is_bwd_valid
        use_fwd_only = is_fwd_valid & ~is_bwd_valid
        use_bwd_only = ~is_fwd_valid & is_bwd_valid

        # Build per-sample (t_plus, t_minus, denom) with broadcasting-safe shapes.
        t_plus = torch.where(is_fwd_valid, t + delta, t)
        t_minus = torch.where(is_bwd_valid, t - delta, t)
        denom = torch.where(
            use_central,
            torch.full_like(t, 2 * delta),
            torch.where(use_fwd_only | use_bwd_only, torch.full_like(t, delta), torch.full_like(t, 1.0)),
        )

        # Linear-path extrapolation along the flow direction:
        # x_t = t * eps + (1 - t) * x_0  =>  d x_t / d t = eps - x_0 = v
        x_t_plus = x_t + expand_like(t_plus - t, x_t) * v
        x_t_minus = x_t + expand_like(t_minus - t, x_t) * v

        F_plus = self.net(x_t_plus, t_plus, r=r, condition=condition, fwd_pred_type="flow")
        F_minus = self.net(x_t_minus, t_minus, r=r, condition=condition, fwd_pred_type="flow")

        dF_dt = (F_plus - F_minus) / expand_like(denom, x_t)

        # Where no finite-difference direction was valid (extremely rare with a
        # small delta), fall back to dF/dt = 0 so we recover pure flow matching.
        no_diff_valid = ~(use_central | use_fwd_only | use_bwd_only)
        if no_diff_valid.any():
            dF_dt = torch.where(expand_like(no_diff_valid, dF_dt), torch.zeros_like(dF_dt), dF_dt)

        target = v - expand_like(t - r, x_t) * dF_dt
        return target

    def _pretrain_single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """Single training step for AnyFlow pretrain."""
        real_data, condition, _ = self._prepare_training_data(data)
        batch_size = real_data.shape[0]

        t, r, is_diffusion = self._sample_pair_timesteps(batch_size, dtype=real_data.dtype)

        # Forward noising along the linear flow-matching path.
        eps = torch.randn_like(real_data)
        x_t = self.net.noise_scheduler.forward_process(real_data, eps, t)

        # Ground-truth instantaneous flow direction at time t.
        v = eps - real_data

        # Central-difference target (no grad through the finite-difference probes).
        target = self._compute_central_difference_target(x_t, t, r, v, condition)

        # Student forward — this is the term whose gradient flows.
        u_theta = self.net(x_t, t, r=r, condition=condition, fwd_pred_type="flow")

        # Per-sample MSE in float for numerical stability under bf16/fp16 AMP.
        sq_err = (u_theta.float() - target.float()).pow(2)
        loss_per_sample = sq_err.flatten(1).mean(dim=-1)

        # Per-timestep weight from the flow-map scheduler (beta08 default).
        weight = self._flowmap_scheduler.get_train_weight(t).to(loss_per_sample.device, loss_per_sample.dtype)
        loss = (loss_per_sample * weight).mean()

        # x0 approximation (monitoring only).
        with torch.no_grad():
            x0_approx = self.net.noise_scheduler.flow_to_x0(x_t, u_theta.detach(), t)

        loss_map = {
            "total_loss": loss,
            "anyflow_loss": loss,
            "flow_matching_loss": loss_per_sample[is_diffusion].mean() if is_diffusion.any() else loss.detach() * 0,
            "dF_dt_target_norm": (target - v).flatten(1).norm(dim=-1).mean(),
        }
        outputs = self._get_outputs(x0_approx, input_student=x_t, condition=condition)
        return loss_map, outputs

    # ------------------------------------------------------------------
    # On-policy stage — DMD2 with r=0 conditioning
    # ------------------------------------------------------------------

    def _zeros_like_t(self, t: torch.Tensor) -> torch.Tensor:
        ns = self.net.noise_scheduler
        return torch.full_like(t, float(ns.min_t))

    def _onpolicy_student_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        data: Dict[str, Any],
        condition: Optional[Any],
        neg_condition: Optional[Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        r_zero = self._zeros_like_t(t_student)
        r_zero_t = self._zeros_like_t(t)

        # Student rollout to a single x0 estimate, conditioned on r=0.
        gen_data = self.net(input_student, t_student, r=r_zero, condition=condition, fwd_pred_type="x0")
        perturbed_data = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        with torch.no_grad():
            fake_score_x0 = self.fake_score(perturbed_data, t, r=r_zero_t, condition=condition, fwd_pred_type="x0")

        # Teacher prediction + optional GAN loss for the generator. Mirrors
        # DMD2._compute_teacher_prediction_gan_loss but with r=0 conditioning.
        if self.config.gan_loss_weight_gen > 0:
            teacher_x0, fake_feat = self.teacher(
                perturbed_data,
                t,
                r=r_zero_t,
                condition=condition,
                feature_indices=self.discriminator.feature_indices,
                fwd_pred_type="x0",
            )
            from fastgen.methods.common_loss import gan_loss_generator

            gan_loss_gen = gan_loss_generator(self.discriminator(fake_feat))
        else:
            teacher_x0 = self.teacher(perturbed_data, t, r=r_zero_t, condition=condition, fwd_pred_type="x0")
            gan_loss_gen = torch.tensor(0.0, device=self.device, dtype=teacher_x0.dtype)
        teacher_x0 = teacher_x0.detach()

        # Optional CFG on the teacher.
        if self.config.guidance_scale is not None:
            with torch.no_grad():
                teacher_x0_neg = self.teacher(
                    perturbed_data, t, r=r_zero_t, condition=neg_condition, fwd_pred_type="x0"
                )
            teacher_x0 = teacher_x0 + (self.config.guidance_scale - 1) * (teacher_x0 - teacher_x0_neg)

        from fastgen.methods.common_loss import variational_score_distillation_loss

        vsd_loss = variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0)
        loss = vsd_loss + self.config.gan_loss_weight_gen * gan_loss_gen

        loss_map = {
            "total_loss": loss,
            "vsd_loss": vsd_loss,
            "gan_loss_gen": gan_loss_gen,
        }
        outputs = self._get_outputs(gen_data, input_student, condition=condition)
        return loss_map, outputs

    def _onpolicy_fake_score_discriminator_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        real_data: torch.Tensor,
        condition: Optional[Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        r_zero = self._zeros_like_t(t_student)
        r_zero_t = self._zeros_like_t(t)

        with torch.no_grad():
            gen_data = self.net(input_student, t_student, r=r_zero, condition=condition, fwd_pred_type="x0")
            x_t_sg = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        from fastgen.methods.common_loss import (
            denoising_score_matching_loss,
            gan_loss_discriminator,
        )

        fake_score_pred_type = self.config.fake_score_pred_type or self.teacher.net_pred_type
        fake_score_pred = self.fake_score(
            x_t_sg, t, r=r_zero_t, condition=condition, fwd_pred_type=fake_score_pred_type
        )
        loss_fakescore = denoising_score_matching_loss(
            fake_score_pred_type,
            net_pred=fake_score_pred,
            noise_scheduler=self.net.noise_scheduler,
            x0=gen_data,
            eps=eps,
            t=t,
        )

        gan_loss_disc = torch.zeros_like(loss_fakescore)
        gan_loss_ar1 = torch.zeros_like(loss_fakescore)
        if self.config.gan_loss_weight_gen > 0:
            with torch.no_grad():
                fake_feat = self.teacher(
                    x_t_sg,
                    t,
                    r=r_zero_t,
                    condition=condition,
                    return_features_early=True,
                    feature_indices=self.discriminator.feature_indices,
                )
                # Real data path — mirror DMD2._compute_real_feat but pass r=0.
                from fastgen.utils.basic_utils import convert_cfg_to_dict

                if self.config.gan_use_same_t_noise:
                    t_real, eps_real = t, eps
                else:
                    t_real = self.net.noise_scheduler.sample_t(
                        real_data.shape[0],
                        **convert_cfg_to_dict(self.config.sample_t_cfg),
                        device=self.device,
                    )
                    eps_real = torch.randn_like(real_data)
                perturbed_real = self.net.noise_scheduler.forward_process(real_data, eps_real, t_real)
                r_zero_real = self._zeros_like_t(t_real)
                real_feat = self.teacher(
                    perturbed_real,
                    t_real,
                    r=r_zero_real,
                    condition=condition,
                    return_features_early=True,
                    feature_indices=self.discriminator.feature_indices,
                )

            real_feat_logit = self.discriminator(real_feat)
            gan_loss_disc = gan_loss_discriminator(real_feat_logit, self.discriminator(fake_feat))

            if self.config.gan_r1_reg_weight > 0:
                perturbed_real_alpha = real_data.add(self.config.gan_r1_reg_alpha * torch.randn_like(real_data))
                with torch.no_grad():
                    real_feat_alpha = self.teacher(
                        perturbed_real_alpha,
                        t_real,
                        r=r_zero_real,
                        condition=condition,
                        return_features_early=True,
                        feature_indices=self.discriminator.feature_indices,
                    )
                real_feat_alpha_logit = self.discriminator(real_feat_alpha)
                gan_loss_ar1 = F.mse_loss(real_feat_logit, real_feat_alpha_logit, reduction="mean")

        loss = loss_fakescore + gan_loss_disc + self.config.gan_r1_reg_weight * gan_loss_ar1
        loss_map = {
            "total_loss": loss,
            "fake_score_loss": loss_fakescore,
            "gan_loss_disc": gan_loss_disc,
        }
        if self.config.gan_loss_weight_gen > 0 and self.config.gan_r1_reg_weight > 0:
            loss_map["gan_loss_ar1"] = gan_loss_ar1
        outputs = self._get_outputs(gen_data, input_student, condition=condition)
        return loss_map, outputs

    def _onpolicy_single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        real_data, condition, neg_condition = self._prepare_training_data(data)
        self._setup_grad_requirements(iteration)
        input_student, t_student, t, eps = self._generate_noise_and_time(real_data)

        if iteration % self.config.student_update_freq == 0:
            return self._onpolicy_student_update_step(
                input_student, t_student, t, eps, data, condition=condition, neg_condition=neg_condition
            )
        return self._onpolicy_fake_score_discriminator_update_step(
            input_student, t_student, t, eps, real_data, condition=condition
        )

    # ------------------------------------------------------------------
    # FastGenModel interface
    # ------------------------------------------------------------------

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: Optional[torch.Tensor] = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        # Pretrain stage uses a direct x0 approximation tensor.
        if self.loss_config.training_stage == "pretrain":
            assert input_student is not None, "input_student must be provided"
            ns = self.net.noise_scheduler
            noise = input_student / (ns.max_sigma if hasattr(ns, "max_sigma") else 1.0)
            return {"gen_rand": gen_data, "input_rand": noise}

        # On-policy stage delegates to DMD2's get_outputs path so multi-step
        # generators are produced consistently with the rest of the family.
        if self.config.student_sample_steps == 1:
            assert input_student is not None, "input_student must be provided"
            ns = self.net.noise_scheduler
            noise = input_student / (ns.max_sigma if hasattr(ns, "max_sigma") else 1.0)
            return {"gen_rand": gen_data, "input_rand": noise}

        noise = torch.randn_like(gen_data, dtype=self.precision)
        gen_rand_func = partial(
            self.generator_fn,
            net=self.net_inference,
            noise=noise,
            condition=condition,
            student_sample_steps=self.config.student_sample_steps,
            student_sample_type=self.config.student_sample_type,
            t_list=self.config.sample_t_cfg.t_list,
            precision_amp=self.precision_amp_infer,
        )
        return {"gen_rand": gen_rand_func, "input_rand": noise, "gen_rand_train": gen_data}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        if self.loss_config.training_stage == "pretrain":
            return self._pretrain_single_train_step(data, iteration)
        return self._onpolicy_single_train_step(data, iteration)

    def get_optimizers(self, iteration: int) -> list:
        """Pretrain stage uses only the student optimizer.

        On-policy stage inherits DMD2's alternating optimisation.
        """
        if self.loss_config.training_stage == "pretrain":
            return [self.net_optimizer]
        return super().get_optimizers(iteration)

    def get_lr_schedulers(self, iteration: int) -> list:
        if self.loss_config.training_stage == "pretrain":
            return [self.net_lr_scheduler]
        return super().get_lr_schedulers(iteration)
