# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Any, Callable, TYPE_CHECKING, Tuple, Optional

import numpy as np
import torch

from fastgen.methods import CMModel
from fastgen.methods.common_loss import finite_difference_points
from fastgen.networks.network import FastGenNetwork
from fastgen.utils.guidance import predict_with_guidance
from fastgen.utils.tensor import (
    as_tensor,
    expand_like,
    high_precision_dtype,
    precision_to_dtype,
    promote_dtype,
    temp_disable_efficient_attn,
    to_high_precision,
)
from fastgen.utils.tree_utils import assert_same, cast_like
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.methods.config_scm import ModelConfig


class TrigFlowPrecond(FastGenNetwork):
    """
    Convert denoiser net (x0-prediction) to TrigFlow's F_\theta(x_t/sigma_data, t)
    """

    def __init__(
        self,
        net: FastGenNetwork,  # denoiser net
        sigma_data: float = 0.5,
    ):
        super().__init__(net_pred_type="flow", schedule_type="trig", t_precision=net.noise_scheduler.t_precision)
        self.net = net
        self.sigma_data = sigma_data

    @property
    def predict_with_guidance(self):
        # Native recipes use ctx.predict, which stays bound to this wrapper.
        return getattr(self.net, "predict_with_guidance", None)

    def _convert_output(self, prediction, x_t, t, condition, fwd_pred_type, **context):
        """``F_theta`` is the trig flow over ``sigma_data``, which no schedule conversion handles."""
        if fwd_pred_type != self.net_pred_type:
            raise NotImplementedError(f"TrigFlowPrecond predicts F_theta only, got fwd_pred_type={fwd_pred_type!r}")
        return super()._convert_output(prediction, x_t, t, condition, fwd_pred_type, **context)

    def _condition_output(self, prediction, x_t, t, condition, fwd_pred_type, **context):
        """A wrapped output constraint cannot be applied in ``F_theta``, so finalizing guidance over a constrained
        network raises unless ``apply_output_conditioning=False``."""
        if getattr(self.net, "_condition_output", None) is not None:
            raise NotImplementedError(
                f"sCM does not support output conditioning of {type(self.net).__name__}; "
                "call predict_with_guidance with apply_output_conditioning=False"
            )
        return prediction

    def trigflow_t(self, t: torch.Tensor) -> torch.Tensor:
        """The TrigFlow time with the SNR of the wrapped network's time *t*, which ``_convert_trigflow_to_net_input``
        maps back to *t*; in the dtype of *t*, clamped to the TrigFlow range after its rounding (float32 pi/2 exceeds
        pi/2).

        ``atan2`` is more stable than ``self.noise_scheduler.sqrt_snr_to_t(self.net.noise_scheduler.sqrt_snr(t) *
        self.sigma_data)``.
        """
        alpha, sigma = self.net.noise_scheduler.alpha_sigma(t)
        t_hat = torch.atan2(sigma, alpha * self.sigma_data)
        trig = self.noise_scheduler
        return trig.safe_clamp(t_hat, min=trig.min_t, max=trig.max_t, keep_grad=True)

    def _convert_trigflow_to_net_input(self, x_t_hat, t_hat):
        """
        Maps the student's trigonometric time (t_hat) back to the teacher's
        noise schedule time (t) based on SNR matching.

        Returns the scaled input and the teacher time.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        dtype = x_t_hat.dtype
        x_t_hat = to_high_precision(x_t_hat)

        # Convert t_hat to t by matching SNR, in the dtype of t_hat: each schedule checks a time in its own precision.
        sqrt_snr = self.noise_scheduler.sqrt_snr(t_hat) / self.sigma_data
        t = self.net.noise_scheduler.sqrt_snr_to_t(sqrt_snr)
        alpha, sigma = self.net.noise_scheduler.alpha_sigma(to_high_precision(t))

        # Calculate scaling coefficient such that x_t = x_t_hat * coeff
        coeff = torch.hypot(alpha, sigma / self.sigma_data)
        x_t = x_t_hat * expand_like(coeff.to(x_t_hat.dtype), x_t_hat)
        return x_t.to(dtype), t

    def forward(
        self,
        x_t_hat,
        t_hat,
        condition=None,
        return_logvar=False,
        return_x0_pred=False,
        apply_output_conditioning=True,
        output_dtype="same",
        **model_kwargs,
    ):
        """``F_theta`` from the wrapped network's x0, in ``output_dtype`` or the state dtype.

        ``F_theta`` is this wrapper's native prediction. The wrapped network's output constraint applies to its x0
        unless ``apply_output_conditioning=False``, as guidance sets it: the constraint cannot be applied in F_theta.
        The x0 stays unrounded until the conversion, which divides its rounding by ``sigma_data * sin(t_hat)``; a
        returned x0 is then rounded like ``F_theta``.
        """
        x_t, t = self._convert_trigflow_to_net_input(x_t_hat, t_hat)

        net_outputs = self.net(
            x_t,
            t,
            condition=condition,
            return_logvar=return_logvar,
            fwd_pred_type="x0",
            apply_output_conditioning=apply_output_conditioning,
            output_dtype=None,
            **model_kwargs,
        )

        if return_logvar:
            x0_pred, logvar = net_outputs[0], net_outputs[1]
        else:
            x0_pred = net_outputs

        flow_unscaled = self.noise_scheduler.x0_to_flow(x_t_hat, x0_pred, t_hat, output_dtype=output_dtype)
        F_theta = flow_unscaled / self.sigma_data
        if return_x0_pred:
            x0_pred = cast_like(x0_pred, x_t_hat, output_dtype)

        if return_x0_pred and return_logvar:
            return F_theta, logvar, x0_pred
        elif return_x0_pred:
            return F_theta, x0_pred
        elif return_logvar:
            return F_theta, logvar
        return F_theta


class SCMModel(CMModel):
    def __init__(self, config: ModelConfig):
        """

        Args:
            config (ModelConfig): The configuration for the sCM model
        """
        super().__init__(config)
        self.sigma_data = self.sample_t_cfg.sigma_data

        # None disables autocast in the JVP region: it always sets the autocast state,
        # so the enclosing training autocast does not carry over into it. The JVP widens the
        # state to this dtype but never narrows it: the networks cast it to their precision_model
        # promoted with the active autocast dtype (cast_to_model_precision), and TrigFlowPrecond's
        # x0 -> F_theta conversion divides the state's rounding by sigma_data * sin(t_hat),
        # which vanishes at low noise.
        self.precision_amp_jvp = precision_to_dtype(self.config.precision_amp_jvp)
        logger.critical(f"Using AMP precision {self.precision_amp_jvp} for JVP")

        # Convert EDM networks to TrigFlow networks
        teacher = getattr(self, "teacher", None)
        if not self.config.loss_config.use_cd:
            assert teacher is None
        # TODO: Support teachers with output conditioning, which guidance cannot finalize in TrigFlow's F_theta.
        if teacher is not None and getattr(teacher, "_condition_output", None) is not None:
            raise NotImplementedError(
                f"sCM does not support teachers with output conditioning yet: {type(teacher).__name__}"
            )
        self.teacher_trigflow = None if teacher is None else TrigFlowPrecond(teacher, sigma_data=self.sigma_data)

        self.net_trigflow = TrigFlowPrecond(self.net, sigma_data=self.sigma_data)

    def _estimate_jvp_finite_difference(self, net_trigflow_wrapper, x_t_hat, t_hat, dxt_dt):
        x_dtype = promote_dtype(x_t_hat, self.precision_amp_jvp)
        t_dtype = t_hat.dtype

        # perform finite difference computations in higher precision
        x_t_hat, dxt_dt = to_high_precision(x_t_hat, dxt_dt)
        t_hat = to_high_precision(t_hat)
        dtype = x_t_hat.dtype

        v_t = self._compute_vt(t_hat)
        eps_t = torch.full_like(t_hat, self.loss_config.jvp_finite_diff_eps)

        # Steps stay within the TrigFlow times of the wrapped schedule's range: the network maps every evaluation's
        # time back into that schedule, whose validity check rejects times outside it.
        schedule = self.net_trigflow.net.noise_scheduler
        low = self.net_trigflow.trigflow_t(as_tensor(schedule.min_t, t_hat.device, t_hat.dtype))
        high = self.net_trigflow.trigflow_t(as_tensor(schedule.max_t, t_hat.device, t_hat.dtype))
        # Step along the exact JVP's tangent (dxt_dt, 1): the teacher velocity under consistency distillation.
        x_t_hat_plus, t_hat_plus, x_t_hat_minus, t_hat_minus, norm_factor = finite_difference_points(
            x_t_hat, t_hat, dxt_dt, eps_t, t_hat + eps_t <= high, t_hat - eps_t >= low
        )

        # Both evaluations start from the RNG state of the forked student forward. F_theta_minus stays unforked so
        # that, as with the exact JVP, the step advances the RNG once rather than rewinding it.
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            F_theta_plus = net_trigflow_wrapper(x_t_hat_plus.to(x_dtype), t_hat_plus.to(t_dtype))
        F_theta_minus = net_trigflow_wrapper(x_t_hat_minus.to(x_dtype), t_hat_minus.to(t_dtype))

        factor = expand_like((v_t * norm_factor).to(dtype), F_theta_plus)
        F_theta_jvp = (F_theta_plus.to(dtype=dtype) - F_theta_minus.to(dtype=dtype)) * factor
        return F_theta_jvp

    def _compute_vt(self, t_hat: torch.Tensor) -> torch.Tensor:
        """
        Computes the Jacobian factor to convert a network's time-derivative into a noise-level derivative.
        """
        alpha_t, sigma_t = self.net_trigflow.noise_scheduler.alpha_sigma(t_hat)
        return alpha_t * sigma_t

    @torch.no_grad()
    def _jvp(
        self,
        x_t_hat: torch.Tensor,
        t_hat: torch.Tensor,
        dxt_dt: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> torch.Tensor:
        def net_trigflow_wrapper(x_t_hat, t_hat):
            out = self.net_trigflow(x_t_hat, t_hat, condition=condition, return_logvar=False, return_x0_pred=False)
            assert isinstance(out, torch.Tensor), f"Expected torch.Tensor, but got {type(out)}"
            return out

        with torch.autocast(
            dtype=self.precision_amp_jvp, device_type=self.device.type, enabled=self.precision_amp_jvp is not None
        ):
            if self.loss_config.use_jvp_finite_diff:
                # The cache is keyed by weight only, so this region would be served the student's casts of
                # the enclosing dtype if the two differ. No student forward follows the JVP in this step, so
                # the detached casts these forwards leave behind need no clear.
                torch.clear_autocast_cache()
                F_theta_jvp = self._estimate_jvp_finite_difference(net_trigflow_wrapper, x_t_hat, t_hat, dxt_dt)
            else:
                x_t_hat = x_t_hat.to(promote_dtype(x_t_hat, self.precision_amp_jvp))

                # Tangents match each primal's dtype; t_hat keeps its high precision.
                v_t = self._compute_vt(t_hat)
                dxt_dt = dxt_dt.to(high_precision_dtype(x_t_hat, dxt_dt))
                v_x = (expand_like(v_t.to(dxt_dt.dtype), dxt_dt) * dxt_dt).to(x_t_hat.dtype)
                with temp_disable_efficient_attn(self.device.type):
                    # ``torch.func.jvp`` never consults or fills the autocast weight cache.
                    F_theta_jvp = torch.func.jvp(net_trigflow_wrapper, (x_t_hat, t_hat), (v_x, v_t))[1]

        return F_theta_jvp

    def _compute_scm_loss(
        self,
        real_data: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        iteration: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        t_hat = self.net_trigflow.trigflow_t(t)

        # Generate z and x_t
        z = torch.randn_like(real_data) * self.sigma_data
        x_t_hat = self.net_trigflow.noise_scheduler.forward_process(real_data, z, t_hat)

        if self.loss_config.use_cd:
            with torch.no_grad():
                guided_F_theta = predict_with_guidance(
                    self.teacher_trigflow,
                    x_t_hat,
                    t_hat,
                    guidance_cfg=self.config.guidance_cfg,
                    condition=condition,
                    neg_condition=neg_condition,
                    skip_inactive=False,  # training: FSDP-safe (uniform forwards across ranks)
                    output_dtype=None,
                )
                # Guidance uses the wrapper's F_theta units.
                dxt_dt = self.sigma_data * guided_F_theta
            # Drop the teacher pass's detached casts (a shared-base teacher runs the student's trainable
            # base modules) between the teacher pass and the student forward below, which then makes its own
            # grad-enabled casts.
            torch.clear_autocast_cache()
        else:
            # alternatively, one can use dxt_dt = self.net_trigflow.noise_scheduler.x0_to_flow(x_t_hat, real_data, t_hat)
            dxt_dt = self.net_trigflow.noise_scheduler.cond_velocity(real_data, z, t_hat, output_dtype=None)

        # The tangent stays in high precision: the JVP casts it to the state's dtype and the loss upcasts it.
        assert_same(shape=True, device=self.device, real_data=real_data, x_t_hat=x_t_hat, dxt_dt=dxt_dt)
        assert_same(dtype=self.precision, real_data=real_data, x_t_hat=x_t_hat)

        # Fork the RNG so the JVP forwards see this forward's dropout masks (one realization for F_theta and dF/dt).
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            F_theta, logvar, x0_pred = self.net_trigflow(
                x_t_hat, t_hat, condition=condition, return_logvar=True, return_x0_pred=True
            )
        F_theta_jvp = self._jvp(x_t_hat, t_hat, dxt_dt, condition=condition)

        scm_loss, scm_loss_unweighted, warmup_weight = self._scm_pred_to_loss(
            F_theta=F_theta,
            F_theta_jvp=F_theta_jvp,
            x_t_hat=x_t_hat,
            dxt_dt=dxt_dt,
            logvar=logvar,
            t_hat=t_hat,
            iteration=iteration,
        )
        return scm_loss, scm_loss_unweighted, logvar, x0_pred, warmup_weight

    def _scm_pred_to_loss(
        self,
        F_theta: torch.Tensor,
        F_theta_jvp: torch.Tensor,
        x_t_hat: torch.Tensor,
        dxt_dt: torch.Tensor,
        logvar: torch.Tensor,
        t_hat: torch.Tensor,
        iteration: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        # perform loss computations in higher precision
        F_theta, F_theta_jvp, x_t_hat, dxt_dt = to_high_precision(F_theta, F_theta_jvp, x_t_hat, dxt_dt)
        t_hat = to_high_precision(t_hat)
        F_theta_ = F_theta.detach()

        # Warmup steps for the tangent term
        if self.loss_config.tangent_warmup_steps > 0:
            r = min(1.0, iteration / self.loss_config.tangent_warmup_steps)
        else:
            r = 1.0

        alpha_hat, sigma_hat = self.net_trigflow.noise_scheduler.alpha_sigma(t_hat)
        g1_coeff = expand_like((-alpha_hat * alpha_hat).to(F_theta.dtype), x_t_hat)
        g1 = g1_coeff * (self.sigma_data * F_theta_ - dxt_dt)
        g2_coeff = expand_like((alpha_hat * sigma_hat).to(F_theta.dtype), x_t_hat)
        g2 = -(g2_coeff * x_t_hat + self.sigma_data * F_theta_jvp)
        g = g1 + r * g2

        # Tangent normalization (or we can perform Tangent clipping: g = torch.clamp(g, min=-1, max=1))
        g_norm = torch.linalg.vector_norm(g, dim=tuple(range(1, g.ndim)), keepdim=True)
        if self.loss_config.g_norm_spatial_invariance:
            g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())  # Make the norm to be invariant to spatial size
        g = g / (g_norm + self.loss_config.tangent_warmup_const)

        # The mean over non-batch dimensions already supplies the paper's 1 / D normalization.
        scm_loss_unweighted = torch.mean(torch.square(F_theta - F_theta_ - g), dim=tuple(range(1, F_theta.ndim)))
        # The prior weighting 1 / (sigma_d tan(t_hat)) is the wrapped schedule's sqrt SNR alpha_t / sigma_t.
        prior_weight = self.net_trigflow.noise_scheduler.sqrt_snr(t_hat) / self.sigma_data
        # Per-sample values take the widest dtype, also that of the times.
        scm_loss_unweighted, logvar, prior_weight = to_high_precision(
            scm_loss_unweighted, logvar.squeeze(-1), prior_weight
        )
        weight = prior_weight if self.loss_config.prior_weighting_enabled else 1.0

        assert_same(
            shape=True,
            dtype=True,
            device=self.device,
            logvar=logvar,
            scm_loss_unweighted=scm_loss_unweighted,
            prior_weight=prior_weight,
        )
        scm_loss = weight / torch.exp(logvar) * scm_loss_unweighted + logvar

        return scm_loss, scm_loss_unweighted, r

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for the SCM model.

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

        t = self.net.noise_scheduler.sample_t(
            real_data.shape[0], **self.sample_t_cfg.sample_t_kwargs(), device=self.device
        )
        scm_loss, scm_loss_unweighted, logvar, x0_pred, warmup_weight = self._compute_scm_loss(
            real_data, t, condition=condition, neg_condition=neg_condition, iteration=iteration
        )

        loss = scm_loss.mean()
        loss_map = {
            "total_loss": loss,
            "scm_loss": loss,
            "unweighted_scm_loss": scm_loss_unweighted.mean(),
            "logvar_loss": logvar.mean(),
            "warmup_weight": torch.tensor(warmup_weight, device=self.device),
        }
        outputs = self._get_outputs(x0_pred, condition=condition, neg_condition=neg_condition)

        # compute rewards
        loss_map = self.apply_reward(x0_pred, data, loss_map)

        return loss_map, outputs
