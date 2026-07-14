# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from typing import Dict, Any, Callable, TYPE_CHECKING, Tuple, Optional

import numpy as np
import torch
from fastgen.methods import CMModel
from fastgen.utils import basic_utils, expand_like
from fastgen.utils.basic_utils import convert_cfg_to_dict
from fastgen.utils.distributed import get_rank, world_size
import fastgen.utils.logging_utils as logger


if TYPE_CHECKING:
    from fastgen.configs.methods.config_mean_flow import ModelConfig
    from fastgen.networks.network import FastGenNetwork


@contextlib.contextmanager
def temp_disable_efficient_attn(device_type: str = "cuda"):
    # save the current state of the efficient attention
    if device_type == "cuda":
        backend = getattr(torch.backends, device_type)
        flash_sdp_enabled = backend.flash_sdp_enabled()
        mem_efficient_sdp_enabled = backend.mem_efficient_sdp_enabled()
        cudnn_sdp_enabled = backend.cudnn_sdp_enabled()
        math_sdp_enabled = backend.math_sdp_enabled()

        # disable the efficient attention
        backend.enable_flash_sdp(False)
        backend.enable_mem_efficient_sdp(False)
        backend.enable_cudnn_sdp(False)
        backend.enable_math_sdp(True)

        try:
            yield
        finally:
            # restore the original state of the efficient attention
            backend.enable_flash_sdp(flash_sdp_enabled)
            backend.enable_mem_efficient_sdp(mem_efficient_sdp_enabled)
            backend.enable_cudnn_sdp(cudnn_sdp_enabled)
            backend.enable_math_sdp(math_sdp_enabled)
    else:
        # For non-CUDA devices (CPU), just yield without modifying backends
        yield


class MeanFlowModel(CMModel):
    def __init__(self, config: ModelConfig):
        """

        Args:
            config (ModelConfig): The configuration for the MeanFlow model
        """
        super().__init__(config)
        self.config = config
        self.sample_t_cfg = self.config.sample_t_cfg
        self.sample_r_cfg = self.config.sample_r_cfg
        self.loss_config = self.config.loss_config

        # Precision for JVP
        if self.config.precision_amp_jvp is None or self.config.precision_amp_jvp == self.precision_amp:
            self.precision_amp_jvp = None
        else:
            self.precision_amp_jvp = basic_utils.PRECISION_MAP[self.config.precision_amp_jvp]
            logger.critical(f"Using precision {self.precision_amp_jvp} for JVP")

        # Normalization constant for the fixed per-timestep loss weighting
        # (computed lazily on first use, see _get_timestep_weight).
        self._timestep_weight_scale: Optional[float] = None

    def _mix_condition(
        self,
        condition: Any,
        neg_condition: torch.Tensor,
        dxt_dt: torch.Tensor,
        guided_dxt_dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.cond_dropout_prob is None:
            return condition, dxt_dt

        batch_size = dxt_dt.shape[0]
        # Decide how many to drop first.
        num_to_drop = (torch.rand(batch_size, device=dxt_dt.device) < self.config.cond_dropout_prob).sum()
        # Create the mask to keep all but the first samples (the order is important to ensures most dropout happens at flow matching loss)
        mask = torch.arange(batch_size, device=dxt_dt.device) >= num_to_drop
        dxt_dt = torch.where(expand_like(mask, dxt_dt), guided_dxt_dt, dxt_dt)

        if isinstance(condition, torch.Tensor):
            condition = torch.where(expand_like(mask, condition), condition, neg_condition)
        elif isinstance(condition, dict):
            condition = condition.copy()
            keys_no_drop = self.config.cond_keys_no_dropout
            assert set(keys_no_drop).issubset(
                condition.keys()
            ), f"keys_no_drop: {keys_no_drop} not in {condition.keys()}"
            for k in condition.keys() - keys_no_drop:
                condition[k] = torch.where(expand_like(mask, condition[k]), condition[k], neg_condition[k])
        else:
            raise TypeError(f"Unsupported type: {type(condition)}")

        return condition, dxt_dt

    def _drop_condition(self, condition: Any, neg_condition: Any) -> Any:
        """Replace the condition with neg_condition for a random per-sample subset.

        Plain text dropout (used by the prediction-side guidance fusion); the
        dropped samples then train the unconditional pathway.
        """
        if self.config.cond_dropout_prob is None or neg_condition is None:
            return condition
        ref = neg_condition if isinstance(neg_condition, torch.Tensor) else next(iter(neg_condition.values()))
        keep = torch.rand(ref.shape[0], device=ref.device) >= self.config.cond_dropout_prob
        if isinstance(condition, torch.Tensor):
            return torch.where(expand_like(keep, condition), condition, neg_condition)
        if isinstance(condition, dict):
            condition = condition.copy()
            for k in condition.keys() - set(self.config.cond_keys_no_dropout):
                condition[k] = torch.where(expand_like(keep, condition[k]), condition[k], neg_condition[k])
            return condition
        raise TypeError(f"Unsupported condition type: {type(condition)}")

    @torch.no_grad()
    def _get_velocity(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = self.net.noise_scheduler.forward_process(x, z, t)

        if self.loss_config.use_cd:
            dxt_dt = self.teacher(x_t, t, condition=condition, fwd_pred_type="flow")
            if self.config.guidance_scale is not None:
                guidance_scale = torch.where(
                    ((t >= self.config.guidance_t_start) & (t <= self.config.guidance_t_end)),
                    self.config.guidance_scale,
                    1.0,
                )
                guidance_scale = expand_like(guidance_scale, x_t).to(dtype=x_t.dtype)
                neg_dxt_dt = self.teacher(x_t, t, condition=neg_condition, fwd_pred_type="flow")
                dxt_dt = dxt_dt + (guidance_scale - 1.0) * (dxt_dt - neg_dxt_dt)
        else:
            dxt_dt = self.net.noise_scheduler.cond_velocity(x=x, eps=z, t=t)

            # unconditional score estimation from meanflow eq (19)
            if self.config.guidance_scale is not None or self.config.guidance_mixture_ratio is not None:
                # Turn off dropout
                self.net.eval()
                neg_dxt_dt = self.net(x_t, t, r=t, condition=neg_condition, fwd_pred_type="flow")
                guidance_scale = self.config.guidance_scale or 1.0
                guidance_scale = torch.where(
                    ((t >= self.config.guidance_t_start) & (t <= self.config.guidance_t_end)),
                    guidance_scale,
                    1.0,
                )
                guidance_scale = expand_like(guidance_scale, x_t).to(dtype=x_t.dtype)

                if self.config.guidance_mixture_ratio is None:
                    guided_dxt_dt = neg_dxt_dt + guidance_scale * (dxt_dt - neg_dxt_dt)
                else:
                    guidance_mixture_ratio = torch.where(
                        ((t >= self.config.guidance_t_start) & (t <= self.config.guidance_t_end)),
                        self.config.guidance_mixture_ratio,
                        0.0,
                    )
                    guidance_mixture_ratio = expand_like(guidance_mixture_ratio, x_t).to(dtype=x_t.dtype)
                    cond_dxt_dt = self.net(x_t, t, r=t, condition=condition, fwd_pred_type="flow")
                    guided_dxt_dt = (
                        guidance_scale * dxt_dt
                        + (1.0 - guidance_scale - guidance_mixture_ratio) * neg_dxt_dt
                        + guidance_mixture_ratio * cond_dxt_dt
                    )

                self.net.train()
                condition, dxt_dt = self._mix_condition(condition, neg_condition, dxt_dt, guided_dxt_dt)

        return condition, dxt_dt

    def _estimate_jvp_finite_difference(
        self,
        net_wrapper: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        dxt_dt: torch.Tensor,
    ) -> torch.Tensor:
        x_dtype = x_t.dtype
        t_dtype = t.dtype

        # perform finite difference computations in higher precision
        dtype = torch.float64
        t = t.to(dtype=dtype)
        r = r.to(dtype=dtype)
        x_t = x_t.to(dtype=dtype)
        dxt_dt = dxt_dt.to(dtype=dtype)
        eps = torch.ones_like(t) * self.config.loss_config.jvp_finite_diff_eps

        is_fwd_step_valid = t + eps <= self.net.noise_scheduler.max_t
        is_bwd_step_valid = (t - eps >= self.net.noise_scheduler.min_t) & (t - eps > r)

        use_center_diff = is_fwd_step_valid & is_bwd_step_valid
        use_fwd_diff = is_fwd_step_valid & ~is_bwd_step_valid
        use_bwd_diff = ~is_fwd_step_valid & is_bwd_step_valid

        t_plus = torch.zeros_like(t)
        t_minus = torch.zeros_like(t)
        norm_factor = torch.zeros_like(t)

        # Central difference
        t_plus[use_center_diff] = t[use_center_diff] + eps[use_center_diff]
        t_minus[use_center_diff] = t[use_center_diff] - eps[use_center_diff]
        norm_factor[use_center_diff] = 1.0 / (2 * eps[use_center_diff])

        t_plus[use_fwd_diff] = t[use_fwd_diff] + eps[use_fwd_diff]
        t_minus[use_fwd_diff] = t[use_fwd_diff]
        norm_factor[use_fwd_diff] = 1.0 / eps[use_fwd_diff]

        t_plus[use_bwd_diff] = t[use_bwd_diff]
        t_minus[use_bwd_diff] = t[use_bwd_diff] - eps[use_bwd_diff]
        norm_factor[use_bwd_diff] = 1.0 / eps[use_bwd_diff]

        # Use float64 tensors consistently
        x_t_plus = x_t + expand_like(t_plus - t, dxt_dt) * dxt_dt
        x_t_minus = x_t + expand_like(t_minus - t, dxt_dt) * dxt_dt

        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            u_theta_plus = net_wrapper(x_t_plus.to(dtype=x_dtype), t_plus.to(dtype=t_dtype), r.to(dtype=t_dtype))
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            u_theta_minus = net_wrapper(x_t_minus.to(dtype=x_dtype), t_minus.to(dtype=t_dtype), r.to(dtype=t_dtype))

        u_theta_jvp_fd = (u_theta_plus.to(dtype=dtype) - u_theta_minus.to(dtype=dtype)) * expand_like(
            norm_factor, u_theta_plus
        )

        return u_theta_jvp_fd

    @torch.no_grad()
    def _jvp(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        dxt_dt: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.precision_amp_jvp is not None:
            x_t = x_t.to(dtype=self.precision_amp_jvp)

        def net_wrapper(x_t, t, r):
            pred = self.net(x_t, t, r=r, condition=condition, fwd_pred_type="flow")
            return pred

        with torch.autocast(
            device_type=self.device.type, dtype=self.precision_amp_jvp, enabled=self.precision_amp_jvp is not None
        ):
            if self.loss_config.use_jvp_finite_diff:
                u_theta_jvp = self._estimate_jvp_finite_difference(net_wrapper, x_t, t, r, dxt_dt)
            else:
                with torch.random.fork_rng(
                    devices=[self.device] if self.device.type == "cuda" else []
                ), temp_disable_efficient_attn(self.device.type):
                    tangents = (
                        dxt_dt.to(dtype=x_t.dtype),
                        torch.ones_like(t, dtype=x_t.dtype),
                        torch.zeros_like(r, dtype=x_t.dtype),
                    )
                    _, u_theta_jvp = torch.func.jvp(net_wrapper, (x_t, t, r), tangents, has_aux=False)

        return u_theta_jvp

    def _sample_t_r_buckets(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (t, r) with t >= r and assign the per-batch buckets.

        Returns ``(t, r, r_eq_t_mask)`` where ``r_eq_t_mask`` marks the
        samples with ``r = t`` (pure flow matching).

        With ``consistency_ratio == 0`` (MeanFlow default) the flow-matching
        head size is stochastic, as before. With ``consistency_ratio > 0``
        (AnyFlow) the buckets are assigned deterministically over the GLOBAL
        batch by rank index, exactly like the AnyFlow reference
        (``sample_timestep`` in ``trainer_wan_anyflow_pretrain.py``): the
        first ``round((1 - r_sample_ratio) * global_bsz)`` samples get
        ``r = t``, the next ``round(consistency_ratio * global_bsz)`` get
        ``r = 0`` (consistency to clean data), and the rest keep the sampled
        random pair.
        """
        t_sample_kwargs = convert_cfg_to_dict(self.sample_t_cfg)
        t = self.net.noise_scheduler.sample_t(batch_size, **t_sample_kwargs, device=self.device)
        r_sample_kwargs = convert_cfg_to_dict(self.sample_r_cfg) if self.sample_r_cfg.enabled else t_sample_kwargs
        r = self.net.noise_scheduler.sample_t(batch_size, **r_sample_kwargs, device=self.device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        assert torch.all(t >= r), "r cannot be larger than t"

        if self.sample_t_cfg.consistency_ratio > 0:
            global_bsz = world_size() * batch_size
            n_flow_matching = round((1.0 - self.sample_t_cfg.r_sample_ratio) * global_bsz)
            n_consistency = round(self.sample_t_cfg.consistency_ratio * global_bsz)
            if (n_flow_matching == 0 or n_consistency == 0) and not getattr(self, "_bucket_warned", False):
                self._bucket_warned = True
                logger.warning(
                    f"The deterministic (t, r) bucket partition is degenerate: with "
                    f"world_size * batch_size = {global_bsz}, r_sample_ratio="
                    f"{self.sample_t_cfg.r_sample_ratio} and consistency_ratio="
                    f"{self.sample_t_cfg.consistency_ratio} yield bucket sizes "
                    f"({n_flow_matching}, {n_consistency}). The partition spans ranks but not "
                    f"gradient-accumulation rounds (as in the AnyFlow reference), so empty "
                    f"buckets stay empty every iteration."
                )
            global_idx = get_rank() * batch_size + torch.arange(batch_size, device=self.device)
            r_eq_t_mask = global_idx < n_flow_matching
            is_consistency = (global_idx >= n_flow_matching) & (global_idx < n_flow_matching + n_consistency)
            r = torch.where(r_eq_t_mask, t, r)
            r = torch.where(is_consistency, torch.zeros_like(r), r)
        else:
            # set t=r (flow matching loss) for a subset of the batch
            flow_matching_size = (torch.rand(batch_size, device=self.device) >= self.sample_t_cfg.r_sample_ratio).sum()
            r_eq_t_mask = torch.arange(batch_size, device=self.device) < flow_matching_size
            r = torch.where(r_eq_t_mask, t, r)

        return t, r, r_eq_t_mask

    def _reduce_mf_loss(self, mf_loss: torch.Tensor, r_eq_t_mask: torch.Tensor) -> torch.Tensor:
        """Reduce the per-sample loss to a scalar, optionally rebalancing.

        With ``loss_config.rebalance_to_diffusion`` set (AnyFlow), every
        flow-map / consistency (r < t) sample's loss is multiplied by the
        detached factor ``mean(global flow-matching losses) / (own loss + 1e-5)``,
        matching the AnyFlow reference (``train_bidirection``): flow-map /
        consistency gradients are self-normalized and rescaled to the global
        flow-matching-loss mean.
        """
        if getattr(self.loss_config, "rebalance_to_diffusion", False) and (~r_eq_t_mask).any():
            with torch.no_grad():
                # The global flow-matching-loss mean only needs the global sum
                # and count, so reduce two scalars instead of gathering the
                # per-sample losses (equivalent to the reference's
                # cat(all_gather(loss))[mask].mean(), and independent of the
                # per-rank batch sizes).
                fm_loss_sum = torch.where(r_eq_t_mask, mf_loss, torch.zeros_like(mf_loss)).sum()
                fm_count = r_eq_t_mask.sum().to(mf_loss.dtype)
                if world_size() > 1:
                    torch.distributed.all_reduce(fm_loss_sum)
                    torch.distributed.all_reduce(fm_count)
                scale = torch.ones_like(mf_loss)
                if fm_count > 0:
                    scale[~r_eq_t_mask] = (fm_loss_sum / fm_count) / (mf_loss[~r_eq_t_mask] + 1e-5)
            mf_loss = mf_loss * scale
        return mf_loss.mean()

    def _timestep_weight_raw(self, t: torch.Tensor) -> torch.Tensor:
        """Unnormalized per-timestep weight as a direct function of t in [0, 1]."""
        weight_type = self.loss_config.weight_type
        if weight_type == "beta08":
            return t * (1 - t).clamp(min=0).sqrt()
        if weight_type == "gaussian":
            # exp(-2 (t - 1/2)^2), shifted so the minimum over [0, 1] is zero.
            return (torch.exp(-2 * (t - 0.5) ** 2) - float(torch.exp(torch.tensor(-0.5)))).clamp(min=0)
        if weight_type == "uniform":
            return torch.ones_like(t)
        raise ValueError(f"Invalid weight_type: {weight_type!r}")

    @torch.no_grad()
    def _get_timestep_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Fixed per-timestep loss weight w(t) (AnyFlow-style).

        The weight is evaluated directly as a function of t. The normalization
        constant matches the AnyFlow reference scheduler, which normalizes the
        weights over its shifted discrete timestep grid (``set_timesteps``:
        1000 points excluding t=0); we compute the same constant once and
        cache it.
        """
        if self._timestep_weight_scale is None:
            shift = self.sample_t_cfg.shift if self.sample_t_cfg.time_dist_type == "shifted" else 1.0
            grid = torch.linspace(1.0, 0.0, 1001, dtype=torch.float64)[:-1]
            grid = shift * grid / (1 + (shift - 1) * grid)
            self._timestep_weight_scale = float(1000.0 / self._timestep_weight_raw(grid).sum())
        return self._timestep_weight_raw(t) * self._timestep_weight_scale

    @torch.no_grad()
    def _compute_weight(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Per-sample loss weight: the adaptive normalization (``norm_method``,
        a function of the per-sample loss; ``None`` disables it) times the
        optional fixed per-timestep weight (``weight_type``, a function of t).
        """
        if self.loss_config.norm_method is None:
            weight = torch.ones_like(tensor)
        else:
            norm_method, *norm_args = self.loss_config.norm_method.split("_")

            if norm_method == "poly":
                power = float(norm_args[0])
                assert len(norm_args) == 1, "poly norm method requires 1 argument"
                weight = 1 / (tensor + self.loss_config.norm_const).pow(power)
            elif norm_method == "exp":
                assert len(norm_args) == 2, "exp norm method requires 2 arguments"
                const, scale = float(norm_args[0]), float(norm_args[1])
                weight = const * torch.exp(scale * tensor + self.loss_config.norm_const)
            else:
                raise ValueError(f"Invalid norm method: {self.loss_config.norm_method}")

        if self.loss_config.weight_type is not None:
            weight = weight * self._get_timestep_weight(t)

        assert (
            weight.shape == tensor.shape
        ), f"weight and input tensor must have the same shape: {weight.shape} != {tensor.shape}"
        return weight

    def _mf_pred_to_loss(
        self,
        u_theta: torch.Tensor,
        u_theta_jvp: torch.Tensor,
        x_t: torch.Tensor,
        dxt_dt: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        iteration: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        # perform loss computations in higher precision
        dtype = torch.float64
        u_theta = u_theta.to(dtype=dtype)
        u_theta_jvp = u_theta_jvp.to(dtype=dtype)
        x_t = x_t.to(dtype=dtype)
        dxt_dt = dxt_dt.to(dtype=dtype)
        t = t.to(dtype=dtype)
        r = r.to(dtype=dtype)
        delta_t = expand_like((t - r).clip(self.net.noise_scheduler.min_t, self.net.noise_scheduler.max_t), u_theta)

        assert u_theta.requires_grad is torch.is_grad_enabled(), "u_theta requires gradients if gradients are enabled"
        assert not any(
            t.requires_grad for t in [u_theta_jvp, x_t, dxt_dt, delta_t]
        ), "u_theta_jvp, x_t, v_t, dxt_dt, delta_t should not require gradients"

        # warmup weight
        if self.loss_config.tangent_warmup_steps > 0:
            warmup_weight = min(1.0, iteration / self.loss_config.tangent_warmup_steps)
        else:
            warmup_weight = 1.0

        # use l2 loss as in the original paper
        if self.loss_config.loss_type == "l2":
            tangent = dxt_dt - warmup_weight * delta_t * u_theta_jvp
            loss = (u_theta - tangent).pow(2)
            if self.loss_config.norm_method is None:
                # Without adaptive normalization the reduction matters: the
                # per-element mean matches the AnyFlow reference loss scale.
                loss = torch.mean(loss, dim=list(range(1, loss.ndim)))
            else:
                loss = torch.sum(loss, dim=list(range(1, loss.ndim)))
            weight = self._compute_weight(loss, t)
            loss = loss * weight

        # use explicit gradient
        elif self.loss_config.loss_type == "opt_grad":
            u_theta_ = u_theta.detach()
            tangent = dxt_dt - u_theta_ - warmup_weight * delta_t * u_theta_jvp

            if self.loss_config.tangent_spatial_invariance:
                # scale tangent to be dimension invariant
                sample_dim_inv = np.sqrt(tangent.shape[0] / tangent.numel())
                tangent = tangent * sample_dim_inv

            opt_grad_norm = torch.linalg.vector_norm(tangent.flatten(1), dim=-1)
            weight = self._compute_weight(opt_grad_norm, t)
            weight = expand_like(weight, tangent)
            loss = (u_theta - (u_theta + tangent * weight).detach()).pow(2)
            loss = torch.sum(loss, dim=list(range(1, loss.ndim)))

        else:
            raise ValueError(f"Invalid loss type: {self.loss_config.loss_type}")

        assert not tangent.requires_grad, "Tangent should not require gradients"
        assert not weight.requires_grad, "Loss weight should not require gradients"
        return loss, tangent, weight, warmup_weight

    @classmethod
    def _student_sample_loop(
        cls,
        net: FastGenNetwork,
        x: torch.Tensor,
        t_list: torch.Tensor,
        condition: Any = None,
        student_sample_type: str = "sde",
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample loop for the student network.

        Args:
            net: The FastGenNetwork network
            x: The latents to start from
            t_list: Timesteps to sample
            condition: Optional conditioning information
            student_sample_type: Type of student multistep sampling

        Returns:
            The sampled data
        """
        batch_size = x.shape[0]

        # Multistep sampling loop
        for t_cur, t_next in zip(t_list[:-1], t_list[1:]):
            t_cur_batch = t_cur.expand(batch_size)
            t_next_batch = t_next.expand(batch_size)
            if student_sample_type == "sde":
                delta_t = expand_like(t_cur, x).to(x.dtype)
                x = x - delta_t * net(
                    x, t=t_cur_batch, condition=condition, r=torch.zeros_like(t_next_batch), fwd_pred_type="flow"
                )
                if t_next > 0:
                    eps_infer = torch.randn_like(x)
                    x = net.noise_scheduler.forward_process(x, eps_infer, t_next_batch)
            elif student_sample_type == "ode":
                delta_t = expand_like(t_cur - t_next, x).to(x.dtype)
                x = x - delta_t * net(x, t=t_cur_batch, condition=condition, r=t_next_batch, fwd_pred_type="flow")
            else:
                raise NotImplementedError(
                    f"student_sample_type must be one of 'sde', 'ode' but got {student_sample_type}"
                )

        return x

    def _compute_mf_loss(
        self,
        real_data: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        iteration: int,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
    ]:
        # Generate z and x_t
        z = torch.randn_like(real_data)
        x_t = self.net.noise_scheduler.forward_process(real_data, z, t)

        guidance_fuse_scale = getattr(self.loss_config, "guidance_fuse_scale", None)
        if guidance_fuse_scale is not None:
            # AnyFlow-style guidance distillation: the guidance is fused on the
            # PREDICTION side — the conditional output is trained to be the
            # guided flow directly. Matches the reference ``train_bidirection``:
            # the regression target stays the raw data velocity, the
            # unconditional branch is queried at the SAME (t, r) flow-map
            # slice, and the effective prediction is
            # (u_cond + (g - 1) * u_uncond) / g. Text dropout replaces the
            # condition with neg_condition for a random subset beforehand.
            g = float(guidance_fuse_scale)
            assert g > 0, f"guidance_fuse_scale must be > 0, got {g} (set it to None to disable guidance fusion)"
            assert (
                neg_condition is not None
            ), "guidance_fuse_scale requires neg_condition: the unconditional branch is queried at the same (t, r)"
            condition = self._drop_condition(condition, neg_condition)
            dxt_dt = self.net.noise_scheduler.cond_velocity(x=real_data, eps=z, t=t)
            torch.clear_autocast_cache()
            # dF/dt of the fused prediction: finite difference of the
            # conditional output divided by g (the unconditional derivative is
            # dropped, as in the reference's compute_central_difference).
            u_theta_jvp = self._jvp(x_t, t, r, dxt_dt, condition=condition) / g
            assert not u_theta_jvp.requires_grad, "u_theta_jvp should not require gradients"

            assert x_t.dtype == real_data.dtype, f"x_t.dtype: {x_t.dtype}, real_data.dtype: {real_data.dtype}"
            u_theta = self.net(x_t, t, r=r, condition=condition, fwd_pred_type="flow")
            with torch.no_grad():
                u_uncond = self.net(x_t, t, r=r, condition=neg_condition, fwd_pred_type="flow")
            u_theta = (u_theta + (g - 1.0) * u_uncond) / g
        else:
            condition, dxt_dt = self._get_velocity(real_data, z, t, condition=condition, neg_condition=neg_condition)
            # prevent JVP to use cached conversions (which can break the computational graph) that were created in the no_grad context of _get_velocity
            torch.clear_autocast_cache()
            u_theta_jvp = self._jvp(x_t, t, r, dxt_dt, condition=condition)
            assert not u_theta_jvp.requires_grad, "u_theta_jvp should not require gradients"

            # additional forward pass to get u_theta with gradient; see also https://github.com/Gsunshine/py-meanflow?tab=readme-ov-file#note-on-jvp
            assert x_t.dtype == real_data.dtype, f"x_t.dtype: {x_t.dtype}, real_data.dtype: {real_data.dtype}"
            u_theta = self.net(
                x_t,
                t,
                r=r,
                condition=condition,
                fwd_pred_type="flow",
            )
        mf_loss, tangent, loss_weight, warmup_weight = self._mf_pred_to_loss(
            u_theta=u_theta, u_theta_jvp=u_theta_jvp, x_t=x_t, dxt_dt=dxt_dt, t=t, r=r, iteration=iteration
        )

        # flow matching loss and x0 approx. (for monitoring only)
        v_loss = torch.mean((u_theta - (z - real_data)) ** 2, dim=tuple(range(1, u_theta.ndim)), keepdim=False)
        x0_approx = self.net.noise_scheduler.flow_to_x0(x_t, u_theta.detach(), t)

        return (
            mf_loss,
            v_loss,
            u_theta_jvp,
            tangent,
            dxt_dt,
            x0_approx,
            loss_weight,
            warmup_weight,
        )

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for the MeanFlow model.

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output
        """
        # Prepare training data and conditions
        real_data, condition, neg_condition = self._prepare_training_data(data)
        batch_size = real_data.shape[0]

        # sample t and r, with per-batch buckets (flow matching / consistency)
        t, r, r_eq_t_mask = self._sample_t_r_buckets(batch_size)

        (
            mf_loss,
            v_loss,
            u_theta_jvp,
            tangent,
            dxt_dt,
            x0_approx,
            loss_weight,
            warmup_weight,
        ) = self._compute_mf_loss(
            real_data=real_data,
            t=t,
            r=r,
            iteration=iteration,
            condition=condition,
            neg_condition=neg_condition,
        )

        loss = self._reduce_mf_loss(mf_loss, r_eq_t_mask)
        loss_map = {
            "total_loss": loss,
            "mf_loss": loss,
            "v_loss": v_loss.mean(),
            "jvp_norm": torch.linalg.vector_norm(u_theta_jvp.flatten(1), dim=-1).mean(),
            "tangent_norm": torch.linalg.vector_norm(tangent.flatten(1), dim=-1).mean(),
            "v_norm": torch.linalg.vector_norm(dxt_dt.flatten(1), dim=-1).mean(),
            "loss_weight": loss_weight.mean(),
            "tangent_warmup_weight": torch.tensor(warmup_weight, device=self.device).mean(),
        }
        outputs = self._get_outputs(x0_approx, condition=condition)

        return loss_map, outputs
