# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Any, Callable, TYPE_CHECKING, Tuple, Optional

import attrs
import numpy as np
import torch
from fastgen.methods import CMModel
from fastgen.methods.common_loss import flow_matching_target, finite_difference_points
from fastgen.utils import basic_utils
from fastgen.utils.distributed import get_rank, world_size
from fastgen.utils.guidance import predict_with_guidance
import fastgen.utils.logging_utils as logger
from fastgen.utils.tensor import (
    expand_like,
    high_precision_dtype,
    precision_to_dtype,
    promote_dtype,
    temp_disable_efficient_attn,
    to_high_precision,
)
from fastgen.utils.tree_utils import assert_same, tmap


if TYPE_CHECKING:
    from fastgen.configs.methods.config_mean_flow import ModelConfig
    from fastgen.networks.network import FastGenNetwork


@attrs.frozen(eq=False)
class FlowMapStepResult:
    """Shared flow-map losses and the clean estimate used for outputs and rewards."""

    loss_map: dict[str, torch.Tensor]
    x0_approx: torch.Tensor
    condition: Any
    condition_dropout_mask: torch.Tensor | None


class FlowMapLossMixin:
    """The MeanFlow flow-map regression objective, shared across methods.

    Holds everything needed to turn a real batch into the flow-map loss
    ``|| u_theta(x_t, t, r) - sg(v - (t - r) du/dt) ||^2``: the (t, r) sampler
    with its flow-matching / consistency buckets, the JVP estimate, the loss
    weighting, and the reduction. ``MeanFlowModel`` uses it as its whole
    training objective; distribution-matching methods can co-train it alongside
    their own objective.

    The host class must provide ``net``, ``device``, ``config`` and ``precision``
    (plus ``teacher`` when ``loss_config.use_cd``), and call ``_init_flow_map_loss``
    from its ``__init__``.
    """

    # The flow-map rollout's own ODE and SDE rules, see ``_advance_student_step``.
    student_sample_types = ("ode", "sde")

    def _init_flow_map_loss(self, sample_t_cfg: Any, sample_r_overrides: Optional[Dict[str, Any]]) -> None:
        """Bind the flow-map loss / sampling configs and the JVP precision.

        The (t, r) configs are passed in rather than read off ``self.config``: the host
        may draw the flow-map times from a different density than its own
        ``sample_t_cfg`` (AnyFlow does -- see ``AnyFlowModel``), and ``CMModel`` already
        owns a ``self.sample_t_cfg`` attribute that must not be rebound here.

        Args:
            sample_t_cfg: Config for sampling the flow-map ``t``.
            sample_r_overrides: Changes to ``sample_t_cfg`` for sampling the flow-map ``r``; None samples ``r``
                like ``t``.
        """
        self.flow_map_sample_t_cfg = sample_t_cfg
        self.flow_map_sample_r_overrides = sample_r_overrides
        self.loss_config = self.config.loss_config

        # None runs the JVP with autocast disabled: the region always sets the autocast
        # state, so the enclosing training autocast does not carry over into it. The JVP
        # widens the state to this dtype but never narrows it: the network casts it to its
        # precision_model promoted with the active autocast dtype (cast_to_model_precision).
        self.precision_amp_jvp = precision_to_dtype(self.config.precision_amp_jvp)
        logger.critical(f"Using AMP precision {self.precision_amp_jvp} for JVP")

        # A new flow-map density needs a new normalization of the fixed per-timestep weight.
        self._timestep_weight_scale_value: Optional[float] = None

        self._check_cond_dropout_and_fusion_config()

    def _check_cond_dropout_and_fusion_config(self) -> None:
        """Validate ``guidance_fuse_scale`` and ``cond_dropout_prob``, and warn when the dropout never applies:
        ``_get_velocity`` drops the condition only under prediction-side fusion or, without CD, alongside
        target-side guidance."""
        # Every host runs this, but AYF's config declares neither knob.
        fuse_scale = getattr(self.config, "guidance_fuse_scale", None)
        cond_dropout_prob = getattr(self.config, "cond_dropout_prob", None)
        if fuse_scale is not None and not fuse_scale > 0:
            raise ValueError(f"guidance_fuse_scale must be > 0, got {fuse_scale} (None disables fusion)")
        if cond_dropout_prob is None:
            return
        if not 0 <= cond_dropout_prob <= 1:
            raise ValueError(f"cond_dropout_prob must be in [0, 1], got {cond_dropout_prob} (None disables dropout)")
        drops_condition = fuse_scale is not None or (
            not self.loss_config.use_cd and len(self.config.guidance_cfg.active_pred_keys()) > 1
        )
        if cond_dropout_prob > 0 and not drops_condition:
            logger.warning(
                f"cond_dropout_prob={cond_dropout_prob} has no effect: the condition is dropped only under "
                "guidance_fuse_scale or, without use_cd, an active guidance_cfg"
            )

    def _drop_condition(
        self, condition: Any, neg_condition: Any, batch_size: int, device: torch.device
    ) -> Tuple[Any, torch.Tensor | None]:
        """Replace the condition with neg_condition for a per-sample subset.

        Returns ``(condition, dropout_mask)``; the boolean ``[B]`` mask is ``True``
        for dropped samples and ``None`` when no dropout applies. As in
        ``SFTModel._mix_condition``, a network with structured conditions owns
        the mixing through ``mix_condition``; otherwise the tensor leaves are
        swapped, through dict values that may be ``None`` or tuples/lists of
        tensors.

        ``deterministic_buckets`` decides whether an index carries bucket
        information: if so the buckets are cut on the GLOBAL index and an
        index-based rule would only hit flow matching on rank 0, so draw per
        sample; otherwise drop the first ``num_to_drop``.
        """
        # Dropout disabled, or no negative condition to swap in: every sample
        # stays conditional.
        if self.config.cond_dropout_prob is None or neg_condition is None:
            return condition, None
        if hasattr(self.net, "mix_condition"):
            return self.net.mix_condition(condition, neg_condition, self.config.cond_dropout_prob)

        if self.flow_map_sample_t_cfg.deterministic_buckets:
            dropped = torch.rand(batch_size, device=device) < self.config.cond_dropout_prob
        else:
            num_to_drop = (torch.rand(batch_size, device=device) < self.config.cond_dropout_prob).sum()
            dropped = torch.arange(batch_size, device=device) < num_to_drop

        def drop_leaf(leaf: torch.Tensor, neg_leaf: torch.Tensor) -> torch.Tensor:
            return torch.where(expand_like(dropped, leaf), neg_leaf, leaf)

        if isinstance(condition, torch.Tensor):
            return drop_leaf(condition, neg_condition), dropped
        if isinstance(condition, dict):
            keys_no_drop = set(self.config.cond_keys_no_dropout)
            if not keys_no_drop.issubset(condition.keys()):
                raise ValueError(f"cond_keys_no_dropout: {keys_no_drop} not in {condition.keys()}")
            if keys_no_drop == condition.keys():
                return condition, None
            droppable = {k: v for k, v in condition.items() if k not in keys_no_drop}
            return condition | tmap(drop_leaf, droppable, {k: neg_condition[k] for k in droppable}), dropped
        raise TypeError(f"Unsupported condition type: {type(condition)}")

    @torch.no_grad()
    def _get_velocity(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        x_t: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor | None]:
        """Regression target for the flow-map loss, the condition it was built from,
        and the optional ``[B]`` mask of samples whose condition was dropped.
        ``x_t`` is the caller's ``forward_process(x, z, t)``. The target stays in high precision: the JVP casts it
        to the state's dtype as its tangent, and the loss upcasts it.

        Two independent choices:

        * ``loss_config.use_cd`` picks the target SOURCE -- the teacher, or the
          conditional data velocity.
        * ``guidance_fuse_scale`` guides the *prediction* in
          ``_compute_mf_loss``, so the target needs no guidance of its own and we only
          drop the condition. Otherwise we guide the target too: through the teacher, or
          -- without one -- by the net's own cond/uncond pass.
        """
        fuse_scale = self.config.guidance_fuse_scale
        if fuse_scale is not None:
            if neg_condition is None:
                raise ValueError(
                    "guidance_fuse_scale requires neg_condition: the unconditional branch is queried at the same (t, r)"
                )
            condition, condition_dropout_mask = self._drop_condition(condition, neg_condition, x.shape[0], x.device)
        else:
            condition_dropout_mask = None

        # All t-windowing is encoded as piecewise ``scale`` + ``boundary`` on
        # the user's ``guidance_cfg`` (resolved per-sample inside
        # ``predict_with_guidance``).  The 3-way teacher mixture (MeanFlow
        # paper eq. 21) ``ω·dxt_dt + κ·pos + (1-ω-κ)·neg`` — where
        # ``dxt_dt`` is the cond_velocity, ``pos``/``neg`` are
        # the network cond/uncond passes, ω is the coefficient on
        # ``dxt_dt`` and κ is the coefficient on ``pos`` — is the algebraic
        # composition of ``cfg.scale=ω`` and ``mix.scale=κ+1``.  Set both
        # to enable it (``pos_mix`` is auto-computed as a fresh cond forward).
        if self.loss_config.use_cd:
            dxt_dt = predict_with_guidance(
                self.teacher,
                x_t,
                t,
                # `None` is an empty ``GuidanceConfig``: under fusion the target must stay
                # unguided from EITHER source, or it would be guided twice.
                guidance_cfg=None if fuse_scale is not None else self.config.guidance_cfg,
                condition=condition,
                neg_condition=neg_condition,
                skip_inactive=False,  # training: FSDP-safe (uniform forwards across ranks)
                output_dtype=None,
                fwd_pred_type="flow",
            )
        else:
            dxt_dt = self.net.noise_scheduler.cond_velocity(x=x, eps=z, t=t, output_dtype=None)
            # unconditional score estimation from meanflow eq (19).
            # ``"pos"`` is the conditional target in the net's native prediction
            # type, the space guidance caches its branches in, and ``"pos_mix"``
            # (when ``mix.scale`` is set) is auto-computed by
            # ``predict_with_guidance`` as a fresh cond forward.
            # Skipped under prediction-side fusion: that is this same guidance, moved onto
            # the prediction, and the dropout it needs already ran above.
            if fuse_scale is None:
                if len(self.config.guidance_cfg.active_pred_keys()) > 1:
                    pos = flow_matching_target(self.net.net_pred_type, x, z, self.net.noise_scheduler, t)
                    # Turn off dropout
                    with basic_utils.train_mode(self.net, mode=False):
                        guided_dxt_dt = predict_with_guidance(
                            self.net,
                            x_t,
                            t,
                            guidance_cfg=self.config.guidance_cfg,
                            condition=condition,
                            neg_condition=neg_condition,
                            precomputed={"pos": pos},
                            skip_inactive=False,  # training: FSDP-safe (uniform forwards across ranks)
                            output_dtype=None,
                            r=t,
                            fwd_pred_type="flow",
                        )
                    condition, condition_dropout_mask = self._drop_condition(
                        condition, neg_condition, x_t.shape[0], x_t.device
                    )
                    # Same subset: a kept sample is conditional + guided, a dropped one
                    # unconditional + unguided.
                    dxt_dt = (
                        guided_dxt_dt
                        if condition_dropout_mask is None
                        else torch.where(expand_like(condition_dropout_mask, dxt_dt), dxt_dt, guided_dxt_dt)
                    )

        return condition, dxt_dt, condition_dropout_mask

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
        x_t, dxt_dt = to_high_precision(x_t, dxt_dt)
        t, r = to_high_precision(t, r)
        dtype = x_t.dtype
        eps = torch.full_like(t, self.loss_config.jvp_finite_diff_eps)

        is_fwd_step_valid = t + eps <= self.net.noise_scheduler.max_t
        is_bwd_step_valid = (t - eps >= self.net.noise_scheduler.min_t) & (t - eps > r)
        x_t_plus, t_plus, x_t_minus, t_minus, norm_factor = finite_difference_points(
            x_t, t, dxt_dt, eps, is_fwd_step_valid, is_bwd_step_valid
        )

        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            u_theta_plus = net_wrapper(x_t_plus.to(dtype=x_dtype), t_plus.to(dtype=t_dtype), r.to(dtype=t_dtype))
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            u_theta_minus = net_wrapper(x_t_minus.to(dtype=x_dtype), t_minus.to(dtype=t_dtype), r.to(dtype=t_dtype))

        norm_factor = expand_like(norm_factor.to(dtype), u_theta_plus)
        u_theta_jvp_fd = (u_theta_plus.to(dtype=dtype) - u_theta_minus.to(dtype=dtype)) * norm_factor

        return u_theta_jvp_fd

    @torch.no_grad()
    def _jvp(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        dxt_dt: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        fwd_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        x_t = x_t.to(dtype=promote_dtype(x_t, self.precision_amp_jvp))
        fwd_kwargs = fwd_kwargs or {}

        def net_wrapper(x_t, t, r):
            pred = self.net(x_t, t, r=r, condition=condition, fwd_pred_type="flow", **fwd_kwargs)
            return pred

        with torch.autocast(
            device_type=self.device.type, dtype=self.precision_amp_jvp, enabled=self.precision_amp_jvp is not None
        ):
            if self.loss_config.use_jvp_finite_diff:
                # The cache is keyed by weight only, so this region would be served the target pass's casts of
                # the enclosing dtype if the two differ. The caller clears again before its gradient forward.
                torch.clear_autocast_cache()
                u_theta_jvp = self._estimate_jvp_finite_difference(net_wrapper, x_t, t, r, dxt_dt)
            else:
                with torch.random.fork_rng(
                    devices=[self.device] if self.device.type == "cuda" else []
                ), temp_disable_efficient_attn(self.device.type):
                    # Tangents match each primal's dtype; t/r keep their high precision.
                    primals = (x_t, t, r)
                    tangents = (dxt_dt.to(dtype=x_t.dtype), torch.ones_like(t), torch.zeros_like(r))
                    # ``torch.func.jvp`` never consults or fills the autocast weight cache.
                    _, u_theta_jvp = torch.func.jvp(net_wrapper, primals, tangents, has_aux=False)
        return u_theta_jvp

    def _warn_on_degenerate_buckets(self, n_flow_matching: int, n_consistency: int, global_bsz: int) -> None:
        """Warn once if a requested bucket rounds away under the deterministic partition."""
        requested_but_empty = (self.flow_map_sample_t_cfg.flow_matching_ratio > 0 and n_flow_matching == 0) or (
            self.flow_map_sample_t_cfg.consistency_ratio > 0 and n_consistency == 0
        )
        if not requested_but_empty or getattr(self, "_bucket_warned", False):
            return
        self._bucket_warned = True
        logger.warning(
            f"The deterministic (t, r) bucket partition is degenerate: with "
            f"world_size * batch_size = {global_bsz}, flow_matching_ratio="
            f"{self.flow_map_sample_t_cfg.flow_matching_ratio} and consistency_ratio="
            f"{self.flow_map_sample_t_cfg.consistency_ratio} yield bucket sizes "
            f"({n_flow_matching}, {n_consistency}). The partition spans ranks but not "
            f"gradient-accumulation rounds, so empty buckets stay empty every iteration."
        )

    def _sample_t_r_buckets(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (t, r) with t >= r and assign the per-batch buckets.

        Returns ``(t, r, r_eq_t_mask)`` where ``r_eq_t_mask`` marks the
        samples with ``r = t`` (pure flow matching).

        The batch splits three ways: a ``flow_matching_ratio`` fraction gets
        ``r = t``, a ``consistency_ratio`` fraction gets ``r = 0``
        (consistency to clean data), and the rest keep the sampled random pair.
        """
        t = self.net.noise_scheduler.sample_t(
            batch_size, **self.flow_map_sample_t_cfg.sample_t_kwargs(), device=self.device
        )
        r_sample_cfg = attrs.evolve(self.flow_map_sample_t_cfg, **(self.flow_map_sample_r_overrides or {}))
        r = self.net.noise_scheduler.sample_t(batch_size, **r_sample_cfg.sample_t_kwargs(), device=self.device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        flow_matching_ratio = self.flow_map_sample_t_cfg.flow_matching_ratio
        consistency_ratio = self.flow_map_sample_t_cfg.consistency_ratio
        if not flow_matching_ratio + consistency_ratio <= 1.0:
            raise ValueError(
                "flow_matching_ratio + consistency_ratio must be <= 1, "
                f"got {flow_matching_ratio} + {consistency_ratio}"
            )

        # Both policies produce two bucket sizes and index the batch the same
        # way; they differ only in how the sizes are obtained.
        if self.flow_map_sample_t_cfg.deterministic_buckets:
            global_bsz = world_size() * batch_size
            n_flow_matching = round(flow_matching_ratio * global_bsz)
            n_consistency = round(consistency_ratio * global_bsz)
            self._warn_on_degenerate_buckets(n_flow_matching, n_consistency, global_bsz)
            position = get_rank() * batch_size + torch.arange(batch_size, device=self.device)
        else:
            # Both sizes come from one uniform draw, cut at either end of [0, 1).
            # The two events are disjoint by the check above, so the buckets
            # cannot overlap.
            uniform = torch.rand(batch_size, device=self.device)
            n_flow_matching = (uniform >= 1.0 - flow_matching_ratio).sum()
            n_consistency = (uniform < consistency_ratio).sum()
            position = torch.arange(batch_size, device=self.device)

        r_eq_t_mask = position < n_flow_matching
        is_consistency = (position >= n_flow_matching) & (position < n_flow_matching + n_consistency)
        r = torch.where(r_eq_t_mask, t, r)
        r = torch.where(is_consistency, torch.zeros_like(r), r)

        return t, r, r_eq_t_mask

    def _reduce_mf_loss(self, mf_loss: torch.Tensor, r_eq_t_mask: torch.Tensor) -> torch.Tensor:
        """Reduce the per-sample loss to a scalar, optionally rebalancing.

        With ``loss_config.rebalance_to_flow_matching`` set, every flow-map /
        consistency (r < t) sample's loss is multiplied by the detached factor
        ``mean(global flow-matching losses) / (own loss + 1e-5)``, so those
        gradients are self-normalized and rescaled to the global
        flow-matching-loss mean.
        """
        # No rank-local condition here: the branch must be taken (or not) by
        # every rank so the collective below cannot deadlock. Applying the
        # scale to an empty ~r_eq_t_mask selection is a no-op.
        if self.loss_config.rebalance_to_flow_matching:
            with torch.no_grad():
                # The global flow-matching-loss mean only needs the global sum
                # and count, so reduce two scalars instead of gathering the
                # per-sample losses (equivalent, and independent of the
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

    def _timestep_weight_scale(self) -> Optional[float]:
        """The constant that normalizes the fixed per-timestep weight to mean one over one draw of the flow-map t
        density, ``None`` without a weight: a quadrature over its quantiles at probabilities 1, 1 - 1/N, ..., 1/N for
        N = 1000. For the shifted density on [0, max_t] these are the shifted schedule grid, and for max_t = 1 the
        reference's training-timestep table.

        It is computed on first use, not at construction, where meta initialization leaves a table schedule
        unmaterialized on all ranks but rank 0. A method, since ``nn.Module.__getattr__`` would mask an
        ``AttributeError`` raised inside a property.
        """
        if self.loss_config.weight_type is None:
            return None
        if self._timestep_weight_scale_value is None:
            noise_scheduler = self.net.noise_scheduler
            p = torch.linspace(1.0, 0.0, 1000 + 1, dtype=high_precision_dtype(noise_scheduler.t_precision))[:-1]
            t = to_high_precision(noise_scheduler.sample_t_icdf(p, **self.flow_map_sample_t_cfg.sample_t_kwargs()))
            weight_sum = self._timestep_weight_raw(t).sum()
            if not torch.isfinite(weight_sum) or weight_sum <= 0:
                raise ValueError(
                    f"weight_type={self.loss_config.weight_type!r} has no finite normalization: its weights sum to "
                    f"{weight_sum.item()} over the quantiles of the flow-map t density"
                )
            self._timestep_weight_scale_value = float(len(t) / weight_sum)
        return self._timestep_weight_scale_value

    def _timestep_weight_raw(self, t: torch.Tensor) -> torch.Tensor:
        """Unnormalized per-timestep weight as a function of the noise fraction ``tau = sigma / (alpha + sigma)`` at
        time t: in [0, 1] on every schedule, and t itself on rectified flow without a schedule shift.

        tau is the rectified-flow time with the same signal-to-noise ratio only for unit-variance data, such as the
        Wan latents AnyFlow's weights were tuned on. For data with standard deviation ``sigma_data`` that time is
        ``sigma / (sigma + alpha * sigma_data)``; on EDM (``sigma_data = 0.5``) the weights shift to noisier levels.
        """
        alpha, sigma = self.net.noise_scheduler.alpha_sigma(t)
        tau = sigma / (alpha + sigma)
        weight_type = self.loss_config.weight_type
        if weight_type == "beta08":
            return tau * (1 - tau).clamp(min=0).sqrt()
        if weight_type == "gaussian":
            # exp(-2 (tau - 1/2)^2), shifted so the minimum over [0, 1] is zero.
            return (torch.exp(-2 * (tau - 0.5) ** 2) - float(torch.exp(torch.tensor(-0.5)))).clamp(min=0)
        raise ValueError(f"Invalid weight_type: {weight_type!r}")

    @torch.no_grad()
    def _compute_weight(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Per-sample loss weight: the adaptive normalization (``norm_method``,
        a function of the per-sample loss; ``None`` disables it) times the
        optional fixed per-timestep weight (``weight_type``, a function of the
        noise fraction at t).
        """
        # Per sample, so in the widest dtype, also that of the times.
        tensor, t = to_high_precision(tensor, t)
        if self.loss_config.norm_method is None:
            weight = torch.ones_like(tensor)
        else:
            norm_method, *norm_args = self.loss_config.norm_method.split("_")

            if norm_method == "poly":
                if len(norm_args) != 1:
                    raise ValueError("poly norm method requires 1 argument")
                power = float(norm_args[0])
                weight = 1 / (tensor + self.loss_config.norm_const).pow(power)
            elif norm_method == "exp":
                if len(norm_args) != 2:
                    raise ValueError("exp norm method requires 2 arguments")
                const, scale = float(norm_args[0]), float(norm_args[1])
                weight = const * torch.exp(scale * tensor + self.loss_config.norm_const)
            else:
                raise ValueError(f"Invalid norm method: {self.loss_config.norm_method}")

        if self.loss_config.weight_type is not None:
            weight = weight * (self._timestep_weight_raw(t) * self._timestep_weight_scale())

        assert_same(
            shape=True,
            dtype=True,
            device=self.device,
            tensor=tensor,
            weight=weight,
        )
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
        u_theta, u_theta_jvp, x_t, dxt_dt = to_high_precision(u_theta, u_theta_jvp, x_t, dxt_dt)
        t, r = to_high_precision(t, r)
        delta_t = expand_like((t - r).to(u_theta.dtype), u_theta)

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
                # per-element mean keeps the loss scale independent of the
                # sample dimensionality.
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
            weight = expand_like(weight.to(u_theta.dtype), tangent)
            loss = (u_theta - (u_theta + tangent * weight).detach()).pow(2)
            loss = torch.sum(loss, dim=list(range(1, loss.ndim)))

        else:
            raise ValueError(f"Invalid loss type: {self.loss_config.loss_type}")

        assert not tangent.requires_grad, "Tangent should not require gradients"
        assert not weight.requires_grad, "Loss weight should not require gradients"
        return loss, tangent, weight, warmup_weight

    def _advance_student_step(
        self,
        net: FastGenNetwork,
        state: torch.Tensor,
        step_index: int,
        student_grid: torch.Tensor,
        condition: Any = None,
    ) -> torch.Tensor:
        """Advance one MeanFlow interval with the configured ODE or SDE rule."""
        num_steps = len(student_grid) - 1
        if not 0 <= step_index < num_steps:
            raise ValueError(f"step_index must be an int in [0, {num_steps - 1}], got {step_index!r}.")

        batch_size = state.shape[0]
        student_sample_type = self.config.student_sample_type
        t_cur = student_grid[step_index]
        t_next = student_grid[step_index + 1]
        t_cur_batch = t_cur.expand(batch_size)
        t_next_batch = t_next.expand(batch_size)

        if student_sample_type == "sde":
            delta_t = expand_like(t_cur.to(state.dtype), state)
            next_state = state - delta_t * net(
                state,
                t=t_cur_batch,
                condition=condition,
                r=torch.zeros_like(t_next_batch),
                fwd_pred_type="flow",
            )
            if t_next > 0:
                eps_infer = torch.randn_like(next_state)
                next_state = net.noise_scheduler.forward_process(next_state, eps_infer, t_next_batch)
            return next_state
        if student_sample_type == "ode":
            delta_t = expand_like((t_cur - t_next).to(state.dtype), state)
            return state - delta_t * net(
                state,
                t=t_cur_batch,
                condition=condition,
                r=t_next_batch,
                fwd_pred_type="flow",
            )
        raise NotImplementedError(f"student_sample_type must be one of 'sde', 'ode' but got {student_sample_type}")

    def _compute_mf_loss(
        self,
        real_data: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        iteration: int,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> tuple[torch.Tensor | float | None, ...]:
        # Generate z and x_t
        z = torch.randn_like(real_data)
        x_t = self.net.noise_scheduler.forward_process(real_data, z, t)

        condition, dxt_dt, condition_dropout_mask = self._get_velocity(
            real_data, z, t, x_t, condition=condition, neg_condition=neg_condition
        )
        u_theta_jvp = self._jvp(x_t, t, r, dxt_dt, condition=condition)
        assert not u_theta_jvp.requires_grad, "u_theta_jvp should not require gradients"

        # The target pass and a finite-difference JVP ran the student under no_grad; drop their detached
        # autocast casts so this gradient forward makes its own.
        torch.clear_autocast_cache()
        # Extra forward for the gradient path: ``_jvp`` runs under no_grad (FSDP2's post-backward autograd
        # function has no forward-AD rule and is only inserted with grad enabled), so its primal output
        # carries no graph; ``fork_rng`` in ``_jvp`` gives this forward the same dropout masks. See
        # https://github.com/Gsunshine/py-meanflow?tab=readme-ov-file#note-on-jvp.
        assert_same(
            shape=True,
            dtype=self.precision,
            device=self.device,
            real_data=real_data,
            x_t=x_t,
        )
        u_theta = self.net(
            x_t,
            t,
            r=r,
            condition=condition,
            fwd_pred_type="flow",
        )

        guidance_fuse_scale = self.config.guidance_fuse_scale
        if guidance_fuse_scale is not None:
            # Guidance distillation on the PREDICTION side (see `_get_velocity`): the
            # conditional output learns the guided flow directly, so only the prediction
            # changes. The uncond branch is queried at the SAME (t, r) flow-map slice,
            # giving (u_cond + (g - 1) * u_uncond) / g; dF/dt is then the finite
            # difference over g on conditional samples, with the unconditional
            # derivative dropped.
            u_theta_jvp = (
                u_theta_jvp / guidance_fuse_scale
                if condition_dropout_mask is None
                else torch.where(
                    expand_like(condition_dropout_mask, u_theta_jvp), u_theta_jvp, u_theta_jvp / guidance_fuse_scale
                )
            )
            # Turn off dropout
            with basic_utils.train_mode(self.net, mode=False), torch.no_grad():
                u_uncond = self.net(x_t, t, r=r, condition=neg_condition, fwd_pred_type="flow")
            u_theta = (u_theta + (guidance_fuse_scale - 1.0) * u_uncond) / guidance_fuse_scale

        mf_loss, tangent, loss_weight, warmup_weight = self._mf_pred_to_loss(
            u_theta=u_theta, u_theta_jvp=u_theta_jvp, x_t=x_t, dxt_dt=dxt_dt, t=t, r=r, iteration=iteration
        )

        # flow matching loss (for monitoring only)
        cond_velocity = self.net.noise_scheduler.cond_velocity(real_data, z, t)
        v_loss = torch.mean((u_theta - cond_velocity) ** 2, dim=tuple(range(1, u_theta.ndim)), keepdim=False)

        if hasattr(self, "reward"):
            # x0 approx. with additional forward pass for rewards. Only an
            # optimized reward consumes this graph via total_loss; otherwise
            # recording it would orphan the graph in the DDP sink (see
            # DDPWrapper.single_train_step).
            with torch.set_grad_enabled(self.config.optimize_reward and torch.is_grad_enabled()):
                x0_approx = x_t - expand_like(t.to(x_t.dtype), x_t) * self.net(
                    x_t, t=t, r=torch.zeros_like(t), condition=condition, fwd_pred_type="flow"
                )
        else:
            # x0 approx. for monitoring only: the same flow-map jump to time 0, with u(t, r) in place of u(t, 0)
            x0_approx = x_t - expand_like(t.to(x_t.dtype), x_t) * u_theta.detach().to(x_t.dtype)

        return (
            mf_loss,
            v_loss,
            u_theta_jvp,
            tangent,
            dxt_dt,
            x0_approx,
            loss_weight,
            warmup_weight,
            condition_dropout_mask,
        )

    def _flow_map_loss_step(self, data: Dict[str, Any], iteration: int) -> FlowMapStepResult:
        """Compute the complete MeanFlow objective and monitoring metrics.

        This is the reusable, loss-only part of ``MeanFlowModel.single_train_step``.
        It deliberately leaves output construction and rewards to the host method,
        since distribution-matching hosts have their own generated outputs and have
        already applied their reward objective.
        The result includes the dropout mask used for the clean estimate's reward pass.
        """
        real_data = data["real"]
        condition = data.get("condition")
        neg_condition = data.get("neg_condition")
        t, r, r_eq_t_mask = self._sample_t_r_buckets(real_data.shape[0])

        (
            mf_loss,
            v_loss,
            u_theta_jvp,
            tangent,
            dxt_dt,
            x0_approx,
            loss_weight,
            warmup_weight,
            condition_dropout_mask,
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
        return FlowMapStepResult(
            loss_map=loss_map,
            x0_approx=x0_approx,
            condition=condition,
            condition_dropout_mask=condition_dropout_mask,
        )


class MeanFlowModel(FlowMapLossMixin, CMModel):
    def __init__(self, config: ModelConfig):
        """

        Args:
            config (ModelConfig): The configuration for the MeanFlow model
        """
        super().__init__(config)
        self._init_flow_map_loss(config.sample_t_cfg, config.sample_r_overrides)

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
        result = self._flow_map_loss_step(data, iteration)
        outputs = self._get_outputs(
            result.x0_approx,
            condition=result.condition,
            neg_condition=data.get("neg_condition"),
        )

        # compute rewards
        loss_map = self.apply_reward(
            result.x0_approx, data, result.loss_map, condition_dropout_mask=result.condition_dropout_mask
        )

        return loss_map, outputs
