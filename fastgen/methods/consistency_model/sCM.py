# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Any, Callable, TYPE_CHECKING, Tuple, Optional

import numpy as np
import torch

from fastgen.methods import CMModel
from fastgen.networks.network import FastGenNetwork
from fastgen.utils.basic_utils import convert_cfg_to_dict, PRECISION_MAP
from fastgen.utils import expand_like
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
        super().__init__(net_pred_type="flow", schedule_type="trig")
        self.net = net
        self.sigma_data = sigma_data

    def _convert_trigflow_to_net_input(self, x_t_hat, t_hat):
        """
        Maps the student's trigonometric time (t_hat) back to the teacher's
        noise schedule time (t) based on SNR matching.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        x_t_hat_dtype = x_t_hat.dtype
        t_hat_dtype = t_hat.dtype
        x_t_hat = x_t_hat.double()
        t_hat = t_hat.double()

        # Convert t_hat to t by matching SNR
        sqrt_snr_t = self.noise_scheduler.sqrt_snr(t_hat)
        t = self.net.noise_scheduler.sqrt_snr_to_t(sqrt_snr_t / self.sigma_data)

        # Calculate scaling coefficient such that x_t = x_t_hat * coeff
        alpha_t = self.net.noise_scheduler.alpha(t)
        sigma_t = self.net.noise_scheduler.sigma(t)
        coeff = (alpha_t**2 + (sigma_t / self.sigma_data) ** 2).sqrt()
        x_t = x_t_hat * expand_like(coeff, x_t_hat)
        return x_t.to(x_t_hat_dtype), t.to(t_hat_dtype)

    def forward(self, x_t_hat, t_hat, condition=None, return_logvar=False, return_x0_pred=False, **model_kwargs):
        x_t, t = self._convert_trigflow_to_net_input(x_t_hat, t_hat)

        net_outputs = self.net(
            x_t, t, condition=condition, return_logvar=return_logvar, fwd_pred_type="x0", **model_kwargs
        )

        if return_logvar:
            x0_pred, logvar = net_outputs[0], net_outputs[1]
        else:
            x0_pred = net_outputs

        flow_unscaled = self.noise_scheduler.x0_to_flow(x_t_hat, x0_pred, t_hat)
        F_theta = flow_unscaled / self.sigma_data

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
        self.config = config
        self.sample_t_cfg = self.config.sample_t_cfg
        self.loss_config = self.config.loss_config
        self.sigma_data = self.sample_t_cfg.sigma_data

        # Precision for JVP. None disables autocast in the JVP region: it always sets the
        # autocast state, so the enclosing training autocast does not carry over into it.
        if self.config.precision_amp_jvp is None:
            self.precision_amp_jvp = None
        else:
            self.precision_amp_jvp = PRECISION_MAP[self.config.precision_amp_jvp]
            logger.critical(f"Using precision {self.precision_amp_jvp} for JVP")

        # Convert EDM networks to TrigFlow networks
        if not self.config.loss_config.use_cd:
            assert getattr(self, "teacher", None) is None
            self.teacher_trigflow = None
        else:
            self.teacher_trigflow = TrigFlowPrecond(self.teacher, sigma_data=self.sigma_data)

        self.net_trigflow = TrigFlowPrecond(self.net, sigma_data=self.sigma_data)

    def _estimate_jvp_finite_difference(self, net_trigflow_wrapper, real_data, z, t_hat):
        x_dtype = real_data.dtype if self.precision_amp_jvp is None else self.precision_amp_jvp
        t_dtype = t_hat.dtype

        # perform finite difference computations in higher precision
        dtype = torch.float64
        t_hat = t_hat.to(dtype=dtype)
        real_data = real_data.to(dtype=dtype)
        z = z.to(dtype=dtype)

        t_hat = t_hat.clamp(-torch.pi / 2 + 1e-5, torch.pi / 2 - 1e-5)
        v_t = self._compute_vt(t_hat)
        eps_t = (self.loss_config.jvp_finite_diff_eps * t_hat.abs()).clamp(min=1e-6)

        t_hat_plus = (t_hat + eps_t).clamp(max=torch.pi / 2 - 1e-5)
        t_hat_minus = (t_hat - eps_t).clamp(min=-torch.pi / 2 + 1e-5)

        x_t_hat_plus = self.net_trigflow.noise_scheduler.forward_process(real_data, z, t_hat_plus)
        x_t_hat_minus = self.net_trigflow.noise_scheduler.forward_process(real_data, z, t_hat_minus)

        # ! Important: ensure to use fork_rng to avoid different dropout masks
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            F_theta_plus = net_trigflow_wrapper(x_t_hat_plus.to(x_dtype), t_hat_plus.to(t_dtype))
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            F_theta_minus = net_trigflow_wrapper(x_t_hat_minus.to(x_dtype), t_hat_minus.to(t_dtype))

        factor = expand_like(v_t / (2 * eps_t), F_theta_plus)
        F_theta_jvp = (F_theta_plus.to(dtype=dtype) - F_theta_minus.to(dtype=dtype)) * factor
        return F_theta_jvp

    def _compute_vt(self, t_hat: torch.Tensor) -> torch.Tensor:
        """
        Computes the Jacobian factor to convert a network's time-derivative into a noise-level derivative.
        """
        alpha_t = self.net_trigflow.noise_scheduler.alpha(t_hat)
        sigma_t = self.net_trigflow.noise_scheduler.sigma(t_hat)
        return alpha_t * sigma_t

    @torch.no_grad()
    def _jvp(
        self,
        real_data: torch.Tensor,
        z: torch.Tensor,
        x_t_hat: torch.Tensor,
        t_hat: torch.Tensor,
        dxt_dt: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> torch.Tensor:
        def net_trigflow_wrapper(x_t_hat, t_hat, eps=1e-4):
            t_hat = t_hat.clamp(min=-torch.pi / 2 + eps, max=torch.pi / 2 - eps)
            out = self.net_trigflow(x_t_hat, t_hat, condition=condition, return_logvar=False, return_x0_pred=False)
            assert isinstance(out, torch.Tensor), f"Expected torch.Tensor, but got {type(out)}"
            return out

        with torch.autocast(
            dtype=self.precision_amp_jvp, device_type=self.device.type, enabled=self.precision_amp_jvp is not None
        ):
            if self.loss_config.use_jvp_finite_diff:
                F_theta_jvp = self._estimate_jvp_finite_difference(net_trigflow_wrapper, real_data, z, t_hat)

            else:
                if self.precision_amp_jvp is not None:
                    dxt_dt = dxt_dt.to(self.precision_amp_jvp)
                    x_t_hat = x_t_hat.to(self.precision_amp_jvp)

                v_t = self._compute_vt(t_hat).to(dxt_dt.dtype)
                v_x = expand_like(v_t, dxt_dt) * dxt_dt
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
        if self.sample_t_cfg.quantize:
            # quantize t
            assert hasattr(self.net, "noise_scheduler")
            sigma_t = self.net.noise_scheduler.sigma(t)
            sigma_idx = self.net.noise_scheduler.closest_sigma_idx(sigma_t)
            t = self.net.noise_scheduler.sigma_idx_to_t(sigma_idx).to(t)

        sigma_t = self.net.noise_scheduler.sigma(t)
        alpha_t = self.net.noise_scheduler.alpha(t)

        # Convert to t_hat in trigflow
        # This is more stable than self.net_trigflow.noise_scheduler.sqrt_snr_to_t(self.net.noise_scheduler.sqrt_snr(t) * self.sigma_data)
        t_hat = torch.atan2(sigma_t, alpha_t * self.sigma_data)

        # Generate z and x_t
        z = torch.randn_like(real_data) * self.sigma_data
        x_t_hat = self.net_trigflow.noise_scheduler.forward_process(real_data, z, t_hat)

        if self.loss_config.use_cd:
            with torch.no_grad():
                dxt_dt = self.sigma_data * self.teacher_trigflow(x_t_hat, t_hat, condition=condition)  # CD
                if self.config.guidance_scale is not None:
                    kwargs = {"condition": neg_condition}
                    if self.config.skip_layers is not None:
                        kwargs["skip_layers"] = self.config.skip_layers
                    neg_dxt_dt = self.sigma_data * self.teacher_trigflow(x_t_hat, t_hat, **kwargs)
                    dxt_dt = dxt_dt + (self.config.guidance_scale - 1.0) * (dxt_dt - neg_dxt_dt)
        else:
            # alternatively, one can use dxt_dt = self.net_trigflow.noise_scheduler.x0_to_flow(x_t_hat, real_data, t_hat)
            dxt_dt = self.net_trigflow.noise_scheduler.cond_velocity(real_data, z, t_hat)

        assert x_t_hat.dtype == real_data.dtype, f"x_t_hat.dtype: {x_t_hat.dtype}, real_data.dtype: {real_data.dtype}"
        F_theta, logvar, x0_pred = self.net_trigflow(
            x_t_hat, t_hat, condition=condition, return_logvar=True, return_x0_pred=True
        )
        F_theta_jvp = self._jvp(real_data, z, x_t_hat, t_hat, dxt_dt, condition=condition)

        scm_loss, scm_loss_unweighted, warmup_weight = self._scm_pred_to_loss(
            F_theta=F_theta,
            F_theta_jvp=F_theta_jvp,
            x_t_hat=x_t_hat,
            dxt_dt=dxt_dt,
            logvar=logvar,
            sigma_t=sigma_t,
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
        sigma_t: torch.Tensor,
        t_hat: torch.Tensor,
        iteration: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        # perform loss computations in higher precision
        dtype = torch.float64
        F_theta = F_theta.to(dtype)
        F_theta_ = F_theta.detach()
        F_theta_jvp = F_theta_jvp.to(dtype)
        logvar = logvar.squeeze(-1).to(dtype)
        t_hat = t_hat.to(dtype)

        # Warmup steps for the tangent term
        r = min(1.0, iteration / self.loss_config.tangent_warmup_steps)
        # Calculate the tangent g using JVP rearrangement
        alpha_hat = self.net_trigflow.noise_scheduler.alpha(t_hat)
        sigma_hat = self.net_trigflow.noise_scheduler.sigma(t_hat)

        g1_coeff = expand_like(-alpha_hat * alpha_hat, x_t_hat).to(F_theta.dtype)
        g1 = g1_coeff * (self.sigma_data * F_theta_ - dxt_dt)
        g2_coeff = expand_like(alpha_hat * sigma_hat, x_t_hat).to(F_theta.dtype)
        g2 = -(g2_coeff * x_t_hat + self.sigma_data * F_theta_jvp)
        g = g1 + r * g2

        # Tangent normalization (or we can perform Tangent clipping: g = torch.clamp(g, min=-1, max=1))
        g_norm = torch.linalg.vector_norm(g, dim=tuple(range(1, g.ndim)), keepdim=True)
        if self.loss_config.g_norm_spatial_invariance:
            g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())  # Make the norm to be invariant to spatial size
        g = g / (g_norm + self.loss_config.tangent_warmup_const)

        # Calculate loss with adaptive weighting
        weight = 1.0 / sigma_t if self.loss_config.prior_weighting_enabled else 1.0
        D = x_t_hat[0, :].numel() if self.loss_config.divide_x_0_spatial_dim else 1.0
        scm_loss_unweighted = torch.mean(torch.square(F_theta - F_theta_ - g), dim=tuple(range(1, F_theta.ndim)))

        assert (
            logvar.shape == scm_loss_unweighted.shape == t_hat.shape
        ), f"log_var.shape: {logvar.shape}, scm_loss_unweighted.shape: {scm_loss_unweighted.shape}, t_hat.shape: {t_hat.shape}"
        scm_loss = weight / (torch.exp(logvar) * D) * scm_loss_unweighted + logvar

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
        real_data, condition, neg_condition = self._prepare_training_data(data)

        t = self.net.noise_scheduler.sample_t(
            real_data.shape[0], **convert_cfg_to_dict(self.sample_t_cfg), device=self.device
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
        outputs = self._get_outputs(x0_pred, condition=condition)

        return loss_map, outputs
