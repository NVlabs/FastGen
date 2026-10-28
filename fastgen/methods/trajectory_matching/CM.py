# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Any, Callable, TYPE_CHECKING, Tuple, Optional

import torch

from fastgen.methods import FastGenModel
import fastgen.utils.logging_utils as logger
from fastgen.utils.tensor import expand_like, to_high_precision
from fastgen.utils.tree_utils import assert_same
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance

if TYPE_CHECKING:
    from fastgen.configs.methods.config_cm import ModelConfig
    from fastgen.networks.network import FastGenNetwork
    from fastgen.networks.noise_schedule import BaseNoiseSchedule


def get_edm_c_out(noise_scheduler, t, sigma_data=0.5):
    # SNR matching to EDM schedule
    alpha, sigma = noise_scheduler.alpha_sigma(t)
    edm_sigma = sigma / noise_scheduler.non_zero_clamp(alpha)
    return edm_sigma * sigma_data / (edm_sigma**2 + sigma_data**2).sqrt()


def t_to_r_sigmoid(t, ratio, min_r=1e-6):
    # Sigmoid delta t proposed by the ECT paper (https://github.com/locuslab/ect)
    # The effective ratio is an increasing function w.r.t t
    r = t - t * (1 - ratio) * (1 + 8 * torch.sigmoid(-t))
    r = torch.clamp(r, min=min_r)
    return r


@torch.no_grad()
def ode_solver(
    net,
    x_t,
    t,
    t_next,
    condition=None,
    neg_condition=None,
    guidance_cfg: GuidanceConfig | None = None,
    pred_type: str | None = None,
):
    """Single DDIM step of ``net`` from ``t`` to ``t_next`` on its prediction of type ``pred_type``.

    ``pred_type=None`` uses ``net.net_pred_type``; ``"flow"`` on unshifted rectified flow is the Euler step.
    ``guidance_cfg=None`` runs just the cond pass (``predict_with_guidance``
    treats ``None`` as an empty config).
    """
    pred_type = pred_type or net.net_pred_type
    prediction = predict_with_guidance(
        net,
        x_t,
        t,
        guidance_cfg=guidance_cfg,
        condition=condition,
        neg_condition=neg_condition,
        skip_inactive=False,  # training: FSDP-safe (uniform forwards across ranks)
        output_dtype=None,
        fwd_pred_type=pred_type,
    )
    return net.noise_scheduler.step(x_t, prediction, t, t_next, pred_type=pred_type)


class CMModel(FastGenModel):
    def __init__(self, config: ModelConfig):
        """

        Args:
            config (ModelConfig): The configuration for the CM model
        """
        super().__init__(config)
        self.sample_t_cfg = self.config.sample_t_cfg
        self.loss_config = self.config.loss_config
        self.ratio = 0.0
        if (
            getattr(self.sample_t_cfg, "min_r", None) is not None
            and self.sample_t_cfg.min_r < self.net.noise_scheduler.min_t
        ):
            logger.warning(
                f"sample_t_cfg.min_r {self.sample_t_cfg.min_r} is smaller than net.noise_scheduler.min_t {self.net.noise_scheduler.min_t} and will be set to {self.net.noise_scheduler.min_t}"
            )

    def build_model(self):
        super().build_model()
        if self.config.loss_config.use_cd:
            self.build_teacher()
        self.load_student_weights_and_ema()

    def _compute_cm_loss(
        self,
        net: FastGenNetwork,
        real_data: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the consistency model loss
        Args:
            net (torch.nn.Module): The consistency model network
            real_data (torch.Tensor): The real data
            t (torch.Tensor): The time step for student net
            condition (Any): The condition to generate
            neg_condition (Any): The negative condition to generate (default: None)
        Return:
            cm_loss (torch.Tensor): The consistency model loss
            loss_unweighted (torch.Tensor): The unweighted cm loss
            D_yt (torch.Tensor): The denoised output of the student net
        """
        # sample r from a pre-defined mapping function
        min_r = max(net.noise_scheduler.min_t, self.sample_t_cfg.min_r)
        r = t_to_r_sigmoid(t, self.ratio, min_r=min_r)

        # safety check: adjust t values where r > t - eps
        needs_shift = r >= (t - net.noise_scheduler.clamp_min)
        if needs_shift.any():
            logger.warning(
                f"Adjusting {needs_shift.sum()} t values where r > t - eps. "
                "Adjust sample_t_cfg.min_t/min_r to avoid this."
            )
            t = torch.where(needs_shift, r + net.noise_scheduler.clamp_min, t)

        # Shared noise direction
        eps = torch.randn_like(real_data)
        y_t = net.noise_scheduler.forward_process(real_data, eps, t)
        if self.loss_config.use_cd:
            if self.teacher is None:
                raise ValueError("There must exist a teacher network when use_cd=True")
            # consistency distillation (CD)
            y_r = ode_solver(
                self.teacher,
                x_t=y_t,
                t=t,
                t_next=r,
                condition=condition,
                neg_condition=neg_condition,
                guidance_cfg=self.config.guidance_cfg,
                pred_type=self.config.teacher_pred_type,
            )
            # Drop the teacher pass's detached casts (a shared-base teacher runs the student's trainable
            # base modules) here, between the teacher and the student forward: the student's grad-enabled
            # casts are made after this point and stay cached for its no_grad ``D_yr`` pass.
            torch.clear_autocast_cache()
        else:
            # consistency training (CT)
            y_r = net.noise_scheduler.forward_process(real_data, eps, r)

        # Shared Dropout Mask with FSDP-safe RNG handling
        assert_same(
            shape=True,
            dtype=self.precision,
            device=self.device,
            real_data=real_data,
            y_t=y_t,
            y_r=y_r,
        )
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            D_yt = net(y_t, t, condition=condition, fwd_pred_type="x0")
        with torch.no_grad():
            D_yr_candidate = net(y_r, r, condition=condition, fwd_pred_type="x0")

        r_is_noisy = r > 0
        r_is_noisy = expand_like(r_is_noisy, real_data)  # Expand mask to match tensor dimensions
        if not D_yr_candidate.isfinite().all():
            logger.warning("Predicted output contains non-finite values. Adjusting them using torch.nan_to_num.")
        D_yr = r_is_noisy * torch.nan_to_num(D_yr_candidate) + (~r_is_noisy) * real_data
        assert_same(
            shape=True,
            dtype=self.precision,
            device=self.device,
            real_data=real_data,
            D_yt=D_yt,
            D_yr=D_yr,
        )

        cm_loss, loss_unweighted = self._pred_to_loss(
            noise_scheduler=net.noise_scheduler, D_yt=D_yt, D_yr=D_yr, t=t, r=r
        )
        return cm_loss, loss_unweighted, D_yt

    def _pred_to_loss(
        self,
        noise_scheduler: BaseNoiseSchedule,
        D_yt: torch.Tensor,
        D_yr: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # perform loss computations in higher precision
        D_yt, D_yr = to_high_precision(D_yt, D_yr)

        # L2 Loss
        l2_distance = torch.linalg.vector_norm(D_yt - D_yr, dim=tuple(range(1, D_yt.ndim)), keepdim=False)

        # Per sample from here on, so in the widest dtype, also that of the times.
        l2_distance, t, r = to_high_precision(l2_distance, t, r)

        # Huber Loss if needed
        c = self.loss_config.huber_const
        if c > 0:
            loss_unweighted = torch.sqrt(l2_distance**2 + c**2) - c
        elif self.loss_config.use_squared_l2:
            loss_unweighted = l2_distance**2
        else:
            loss_unweighted = l2_distance

        weighting = self.loss_config.weighting_ct_loss
        assert_same(
            shape=True,
            dtype=True,
            device=self.device,
            loss_unweighted=loss_unweighted,
            t=t,
            r=r,
        )
        if weighting == "default":
            cm_loss = loss_unweighted / (t - r)
        elif weighting == "c_out":
            cm_loss = loss_unweighted / get_edm_c_out(noise_scheduler, t)
        elif weighting == "c_out_sq":
            cm_loss = loss_unweighted / get_edm_c_out(noise_scheduler, t) ** 2
        elif weighting == "sigma_sq":
            cm_loss = loss_unweighted / noise_scheduler.sigma(t) ** 2
        elif weighting == "sqrt":
            cm_loss = loss_unweighted / (t - r) ** 0.5
        elif weighting == "one":
            cm_loss = loss_unweighted
        else:
            raise ValueError(
                f"Weighting function {weighting!r} not implemented; choose one of "
                "default, c_out, c_out_sq, sigma_sq, sqrt, one."
            )
        return cm_loss, loss_unweighted

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for the CM model.

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
        cm_loss, loss_unweighted, D_yt = self._compute_cm_loss(
            net=self.net, real_data=real_data, t=t, condition=condition, neg_condition=neg_condition
        )

        loss = cm_loss.mean()
        loss_map = {
            "total_loss": loss,
            "cm_loss": loss,
            "unweighted_cm_loss": loss_unweighted.mean(),
        }
        outputs = self._get_outputs(D_yt, condition=condition, neg_condition=neg_condition)

        # compute rewards
        loss_map = self.apply_reward(D_yt, data, loss_map)

        return loss_map, outputs
