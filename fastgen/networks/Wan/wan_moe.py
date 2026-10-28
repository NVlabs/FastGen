# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WanMoE: MoE wrapper that composes two Wan networks (high-noise and low-noise experts)."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Set, Tuple, Type, Union

import torch
from torch.distributed.fsdp import FSDPModule
from tqdm.auto import tqdm

from diffusers.models import attention_dispatch as _diffusers_attn_dispatch

from fastgen.networks.network import FastGenNetwork
from fastgen.utils.tree_utils import OutputDtype, assert_same
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.networks.Wan.network import Wan
from fastgen.utils.distributed.fsdp import make_fully_shard_kwargs


def _patch_diffusers_native_attention_for_empty_batch() -> None:
    """Give diffusers' ``_native_attention`` a graph-connected result for empty (batch_size==0) inputs.

    cuDNN SDPA, the default backend for Wan's bf16 shapes on H100 and GB200, returns ``None`` for a
    batch-0 input and diffusers then fails at ``out.permute(...)``. MoE routing feeds an expert an empty
    batch on every rank whose samples all went to the other expert, so training hits this path.

    The replacement has to keep the autograd graph of the populated case, not just the shape. FSDP2
    issues a parameter group's reduce-scatter from an autograd node on the group's first module inputs;
    for Wan's per-block ``[attn1, attn2, ffn]`` groups that is ``attn1``'s ``hidden_states``, whose only
    consumers are the Q/K/V projections. A detached substitute such as ``torch.zeros_like(query)`` makes
    that node unreachable. The zero-weighted Q/K/V term below instead keeps every projection in the graph,
    so all parameters receive zero gradients without relying on any SDPA backend's handling of empty
    inputs. Since both experts execute on every sharded rank, this connected zero keeps their parameter
    hooks and collectives ordered even when expert-local activation batch sizes differ between ranks.
    """
    orig = _diffusers_attn_dispatch._native_attention
    if getattr(orig, "_fastgen_empty_batch_safe", False):
        return  # idempotent; module may be imported many times

    def _empty_batch_safe_native_attention(query, key, value, *args, **kwargs):
        # diffusers dispatches with keyword arguments; ``return_lse`` requests are left to the original
        # function so its unsupported-argument error still surfaces.
        if query.shape[0] == 0 and not kwargs.get("return_lse", False):
            # Same layout and dtype as a populated result ([B, S_q, heads, head_dim]); the zero-weighted
            # scalar keeps Q, K and V connected without calling an attention kernel.
            return query.new_zeros(query.shape) + 0 * (query.sum() + key.sum() + value.sum())
        return orig(query, key, value, *args, **kwargs)

    _empty_batch_safe_native_attention._fastgen_empty_batch_safe = True
    _diffusers_attn_dispatch._native_attention = _empty_batch_safe_native_attention
    # The backend registry stored a direct reference to the original at
    # import time; re-register so dispatch picks up the wrapper.
    backend_name = _diffusers_attn_dispatch.AttentionBackendName.NATIVE
    _diffusers_attn_dispatch._AttentionBackendRegistry._backends[backend_name] = _empty_batch_safe_native_attention


class WanMoE(FastGenNetwork):
    """MoE wrapper that holds two Wan networks and routes by timestep.

    Wan2.2 T2V/I2V A14B use two experts: high-noise (t >= guidance_boundary) and
    low-noise (t < guidance_boundary). This wrapper composes two full Wan networks,
    sharing VAE/text/image encoders. Parameters are automatically registered.
    Works with Wan and WanI2V. FSDP-sharded experts always execute in the same
    order on every rank, including when an expert receives an empty local batch.
    """

    DEFAULT_EXPERT_OVERRIDES: dict[str, dict[str, Any]] = {
        "high": {"transformer_subfolder": "transformer"},
        "low": {"transformer_subfolder": "transformer_2"},
    }
    # The two-expert Wan2.2 A14B checkpoints all use the Wan2.1 VAE.
    latent_compression_factors = (4, 8, 8)

    def __init__(
        self,
        wan_cls: Type,
        model_id_or_local_path: str,
        guidance_boundary: float = 0.875,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        enable_single_expert_shortcut: bool = False,
        cond_keys_no_routing: Optional[Set[str]] = None,
        expert_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
        **kwargs,
    ):
        """Create MoE from two Wan networks.

        Args:
            wan_cls: Wan subclass (Wan or WanI2V).
            model_id_or_local_path: HuggingFace model ID or local path.
            guidance_boundary: t >= guidance_boundary uses high expert, t < uses low expert.
            net_pred_type: Prediction type used by wrapper and both experts.
            schedule_type: Noise schedule type used by wrapper and both experts.
            enable_single_expert_shortcut: Allow one-expert shortcut even in training when both
                expert transformer roots are unsharded. FSDP-sharded experts always use the
                collective-aligned two-expert path.
            cond_keys_no_routing: Condition keys that should never be split by routing.
            expert_overrides: Per-expert constructor keyword overrides, keyed by ``"high"`` and
                ``"low"``. Defaults to ``DEFAULT_EXPERT_OVERRIDES`` and takes precedence
                over the common arguments and ``kwargs``. Both experts must retain the wrapper's
                ``net_pred_type`` so routed native predictions have one output convention.
            **kwargs: Common keyword arguments passed to both experts.
        """
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type)

        # Empty (batch-0) attention inputs are only produced by MoE routing, so install the
        # diffusers native-attention batch-0 workaround here (idempotent) rather than at import
        # time -- this keeps it off non-MoE (and non-Wan) runs that merely import net.py.
        _patch_diffusers_native_attention_for_empty_batch()

        self.guidance_boundary = guidance_boundary
        self.enable_single_expert_shortcut = enable_single_expert_shortcut
        self.cond_keys_no_routing = set(cond_keys_no_routing or set())

        if expert_overrides is None:
            expert_overrides = self.DEFAULT_EXPERT_OVERRIDES
        unknown_roles = set(expert_overrides) - {"high", "low"}
        if unknown_roles:
            raise ValueError(f"Unknown expert override roles: {sorted(unknown_roles)}")
        for role, overrides in expert_overrides.items():
            if not isinstance(overrides, Mapping):
                raise TypeError(f"expert_overrides[{role!r}] must be a mapping")
            if overrides.get("net_pred_type", net_pred_type) != net_pred_type:
                raise ValueError(
                    f"expert_overrides[{role!r}] must keep net_pred_type={net_pred_type!r}; "
                    "WanMoE merges both experts in the wrapper's native prediction space"
                )

        common_expert_kwargs = dict(
            model_id_or_local_path=model_id_or_local_path,
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            **kwargs,
        )
        high_kwargs = {**common_expert_kwargs, **expert_overrides.get("high", {})}
        low_kwargs = {**common_expert_kwargs, **expert_overrides.get("low", {})}
        self.network_high = wan_cls(**high_kwargs)
        self.network_low = wan_cls(**low_kwargs)
        # Mirror the high expert's schedule kwargs so ``_schedule_kwargs`` describes the scheduler
        # ``_set_noise_schedule`` shares.
        self._schedule_kwargs = dict(self.network_high._schedule_kwargs)
        self._set_noise_schedule(**self._schedule_kwargs)

        # The wrapper and high-noise expert expose the same unregistered preprocessor registry.
        self.preprocessors = self.network_high.preprocessors

    def _set_noise_schedule(self, schedule_type: Optional[str] = None, **schedule_kwargs) -> None:
        """Share the high-noise expert's scheduler.

        Schedule kwargs reach only the experts, and the high expert's scheduler also builds the sampling grid
        (``sample_scheduler_and_grid``). Both paths reach one scheduler module with one set of
        non-persistent buffer slots, which preparation supports without creating aliased slots.
        """
        assert not self.is_prepared, "Initialize the noise schedule before preparation"
        network_high = self._modules.get("network_high")
        if network_high is None:
            # ``FastGenNetwork.__init__`` calls this before the experts exist; ``__init__`` shares the high
            # expert's scheduler once they do.
            self.noise_scheduler = None
            return
        if schedule_type not in (None, self.schedule_type) or schedule_kwargs != self._schedule_kwargs:
            raise NotImplementedError(
                f"{type(self).__name__}._set_noise_schedule does not support reconfiguring the high-noise "
                "expert's shared schedule; pass schedule kwargs to the experts instead"
            )
        self.noise_scheduler = network_high.noise_scheduler

    def sample_scheduler_and_grid(self, sample_steps: int, flow_shift: float, device: Optional[torch.device] = None):
        return self.network_high.sample_scheduler_and_grid(sample_steps, flow_shift, device=device)

    @property
    def fps(self):
        """Frame rate shared by both Wan experts."""
        fps = self.network_high.fps
        assert self.network_low.fps == fps, "Wan MoE experts must use the same FPS"
        return fps

    @property
    def concat_mask(self) -> bool:
        return getattr(self.network_high, "concat_mask", False)

    def resolve_sample_settings(
        self, input_shape: Sequence[int], sample_steps: Optional[int] = None, shift: Optional[float] = None
    ) -> Tuple[int, float]:
        """The wrapped I2V checkpoint's ``WanI2V.resolve_sample_settings``."""
        return self.network_high.resolve_sample_settings(input_shape, sample_steps, shift)

    def init_vae(self) -> None:
        """Initialize VAE (e.g. for decoding in callbacks); delegates to high expert and exposes on wrapper."""
        self.network_high.init_vae()

    def init_text_encoder(self) -> None:
        self.network_high.init_text_encoder()

    def init_image_encoder(self) -> None:
        self.network_high.init_image_encoder()

    def init_preprocessors(self) -> None:
        self.network_high.init_preprocessors()

    def reset_parameters(self) -> None:
        # Expert resets rebuild their tables in place; the wrapper keeps sharing the high scheduler.
        self.network_high.reset_parameters()
        self.network_low.reset_parameters()

    def fully_shard(self, **kwargs) -> None:
        # Both experts run forward and backward on every rank. The batch-0 attention path keeps the
        # full autograd graph when a rank has no local samples for an expert. Each expert is a nested
        # FastGen module and shards under its own precision policy.
        for expert in (self.network_high, self.network_low):
            expert_kwargs = make_fully_shard_kwargs(
                kwargs, param_dtype=expert.precision_model, storage_dtype=self.precision_opt, enabled=True
            )
            expert.fully_shard(**(expert_kwargs or kwargs))

    def _mask_select(self, mask: torch.Tensor, out_high: Any, out_low: Any) -> Any:
        """Merge the scattered expert outputs per element or per frame via ``torch.where``.

        With per-element timesteps the index sets are disjoint: a sample's row in the other expert's
        output is a zero written by ``_scatter_output``, so the merge selects only values the
        responsible expert produced and the backward reaches only that expert. With per-frame timesteps
        a sample whose frames straddle the boundary is evaluated by both experts on all of its frames;
        the merge then picks each frame from the responsible expert and ``torch.where`` sends zero
        gradient into the masked-out frames of the other one. That zero gradient does not make the
        masked-out frames harmless for the backward: a non-finite *activation* inside an expert on a
        frame it does not own can still poison that expert's parameter gradients through ``0 * NaN`` in
        its own backward. Sanitizing the merged output cannot prevent this: unselected values are
        already absent from the merged forward value after ``torch.where``, while output sanitization
        only conceals selected non-finite values. This router therefore performs no numerical repair on purpose.
        Selected failures propagate to the loss; masked-frame failures may first surface in the
        gradients and must be reported or handled at the training level (finiteness checks on the loss
        and gradients), not here.
        """
        if torch.is_tensor(out_high):
            m = mask
            while m.ndim > out_high.ndim:
                m = m.squeeze(-1) if m.shape[-1] == 1 else m.any(dim=-1)
            while m.ndim < out_high.ndim:
                m = m.unsqueeze(-1)
            return torch.where(m.expand_as(out_high), out_high, out_low)
        if isinstance(out_high, tuple):
            assert isinstance(out_low, tuple) and len(out_high) == len(out_low)
            return tuple(self._mask_select(mask, oh, ol) for oh, ol in zip(out_high, out_low))
        if isinstance(out_high, list):
            assert isinstance(out_low, list) and len(out_high) == len(out_low)
            return [self._mask_select(mask, oh, ol) for oh, ol in zip(out_high, out_low)]
        raise TypeError(f"Unsupported output type for MoE merge: {type(out_high)}")

    def _select_batch_elements(self, data: Any, indices: torch.Tensor) -> Any:
        """Index dim-0 of every tensor inside a nested structure."""
        if data is None:
            return None
        if torch.is_tensor(data):
            return data if data.dim() == 0 else data[indices]
        if isinstance(data, dict):
            return {
                k: (v if k in self.cond_keys_no_routing else self._select_batch_elements(v, indices))
                for k, v in data.items()
            }
        if isinstance(data, (tuple, list)):
            return type(data)(self._select_batch_elements(v, indices) for v in data)
        return data

    def _select_kwargs_batch(self, kwargs: dict, indices: torch.Tensor) -> dict:
        """Return a copy of forward *kwargs* with batch-indexed entries sub-selected."""
        out = dict(kwargs)
        out["condition"] = self._select_batch_elements(kwargs["condition"], indices)
        if kwargs.get("r") is not None:
            out["r"] = kwargs["r"][indices]
        return out

    def _empty_output_like(self, reference: Any, batch_size: int) -> Any:
        """Zero-filled output with same nested structure as *reference* but *batch_size* rows."""
        if reference is None:
            return None
        if torch.is_tensor(reference):
            return reference.new_zeros((batch_size,) + reference.shape[1:])
        if isinstance(reference, tuple):
            return tuple(self._empty_output_like(r, batch_size) for r in reference)
        if isinstance(reference, list):
            return [self._empty_output_like(r, batch_size) for r in reference]
        raise TypeError(f"Unsupported output type for empty allocation: {type(reference)}")

    def _scatter_output(self, dst: Any, src: Any, indices: torch.Tensor) -> Any:
        """Scatter *src* (subset) into *dst* (full batch) at batch *indices*.

        Uses out-of-place ``index_copy`` so the result is differentiable.
        """
        if dst is None:
            return None
        if torch.is_tensor(dst):
            # Expert indices are unique, and index_copy directly restores the
            # selected batch rows while also supporting an empty subset.
            return dst.index_copy(0, indices, src)
        if isinstance(dst, tuple):
            return tuple(self._scatter_output(d, s, indices) for d, s in zip(dst, src))
        if isinstance(dst, list):
            return [self._scatter_output(d, s, indices) for d, s in zip(dst, src)]
        raise TypeError(f"Unsupported output type for scatter: {type(dst)}")

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Any] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        unpatchify_features: bool = True,
        fwd_pred_type: Optional[str] = None,
        skip_layers: Optional[list] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], list]:
        num_features = min(
            len(self.network_high.transformer.blocks),
            len(self.network_low.transformer.blocks),
        )
        self._validate_forward_kwargs(
            fwd_kwargs,
            r=r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            skip_layers=skip_layers,
            supports_r_timestep=all(
                getattr(network.transformer, "r_embedder", None) is not None
                for network in (self.network_high, self.network_low)
            ),
            supports_skip_layers=True,
            num_features=num_features,
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        assert_same(
            shape=True,
            dtype=self.noise_scheduler.t_precision,
            device=True,
            t=t,
            r=r,
        )
        # Per-frame routing: each (batch, frame) uses high expert iff t >= guidance_boundary.
        # Shape [B, 1, 1, 1, 1] for per-batch t; [B, 1, T, 1, 1] for per-frame t.
        # Both broadcast against [B, C, T, H, W] outputs in torch.where.
        if t.numel() in [t.shape[0], 1]:
            high_mask = (t.flatten() >= self.guidance_boundary).view(-1, 1, 1, 1, 1)
        else:
            high_mask = (t >= self.guidance_boundary).unsqueeze(1).unsqueeze(3).unsqueeze(4)
        assert high_mask.shape[:2] == (x_t.shape[0], 1) and high_mask.dim() == 5

        kwargs = dict(
            condition=condition,
            r=r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            unpatchify_features=unpatchify_features,
            fwd_pred_type=fwd_pred_type,
            apply_output_conditioning=apply_output_conditioning,
            output_dtype=output_dtype,
            skip_layers=skip_layers,
        )

        # Per-batch-element expert needs (collapse per-frame mask to per-element).
        batch_needs_high = high_mask.flatten(1).any(dim=1)  # [B]
        batch_needs_low = (~high_mask).flatten(1).any(dim=1)  # [B]

        # Fast path: entire batch routed to a single expert.
        # Skipping an expert's forward() prevents its FSDP2 allgather/reduce-scatter
        # hooks from firing, which can deadlock the collective or produce incorrect
        # outputs (stale/zero param shards) across ranks when different ranks pick
        # different experts. Only allow the shortcut when:
        #   - explicitly opted in via enable_single_expert_shortcut, OR
        #   - running under inference_mode;
        # and both expert transformer roots are currently unsharded. Check the live
        # roots rather than caching this state: FSDP2 wrapping happens after construction.
        experts_are_sharded = isinstance(self.network_high.transformer, FSDPModule) or isinstance(
            self.network_low.transformer, FSDPModule
        )
        allow_shortcut = (
            self.enable_single_expert_shortcut or torch.is_inference_mode_enabled()
        ) and not experts_are_sharded
        if allow_shortcut and not batch_needs_low.any():
            return self.network_high(x_t, t, **kwargs)
        if allow_shortcut and not batch_needs_high.any():
            return self.network_low(x_t, t, **kwargs)

        # ---- Batch-split: feed each expert only the elements that need it ----
        # Both experts are always called (even with empty input) so that FSDP2
        # pre-forward hooks fire the allgather on every rank.
        idx_high = batch_needs_high.nonzero(as_tuple=False).view(-1)
        idx_low = batch_needs_low.nonzero(as_tuple=False).view(-1)

        kwargs_h = self._select_kwargs_batch(kwargs, idx_high)
        kwargs_l = self._select_kwargs_batch(kwargs, idx_low)

        out_high = self.network_high(x_t[idx_high], t[idx_high], **kwargs_h)
        out_low = self.network_low(x_t[idx_low], t[idx_low], **kwargs_l)

        # Scatter each expert's subset output into a full-batch tensor (zeros
        # where that expert was not called), then mask-merge.  Works for both
        # disjoint and overlapping index sets: the mask selects the correct
        # expert at every (batch, frame) position.
        B = x_t.shape[0]
        full_high = self._scatter_output(self._empty_output_like(out_high, B), out_high, idx_high)
        full_low = self._scatter_output(self._empty_output_like(out_low, B), out_low, idx_low)

        # Per element: disjoint routing, the other expert's row is constant zeros. Per frame: a
        # straddling sample is evaluated by both experts and merged frame-wise, with zero gradient into
        # the masked-out frames (see _mask_select for the non-finite caveat).
        return self._mask_select(high_mask, full_high, full_low)

    @property
    def _condition_output(self):
        """The output constraint both experts share, applied after routing."""
        return self.network_high._condition_output

    def default_guidance_cfg(self) -> GuidanceConfig:
        """Use the wrapped Wan architecture's default guidance recipe."""
        return self.network_high.default_guidance_cfg()

    def sample(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Call base-class sampler with WanMoE self for routed forward()."""
        sample_impl = self.network_high.__class__.sample
        return sample_impl(self, *args, **kwargs)

    def build_conditioning(self, data, input_shape=None, precision=None):
        """Delegate the complete conditioning request to the high-noise expert once.

        Both experts share the same conditioning contract (text + I2V
        first-frame [+ camera] fields), and the wrapper shares the high expert's
        preprocessor dictionary (VAE / text / image encoders). The selected
        expert therefore owns raw-text resolution as well as task enrichment.
        """
        return self.network_high.build_conditioning(data, input_shape=input_shape, precision=precision)


class WanMoET2V(WanMoE):
    """WanMoE for T2V (Wan2.2 T2V A14B)."""

    DEFAULT_GUIDANCE_BOUNDARY = 0.875

    def __init__(
        self,
        model_id_or_local_path: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        guidance_boundary: float = DEFAULT_GUIDANCE_BOUNDARY,
        **kwargs,
    ):
        super().__init__(
            wan_cls=Wan,
            model_id_or_local_path=model_id_or_local_path,
            guidance_boundary=guidance_boundary,
            **kwargs,
        )

    @classmethod
    def default_guidance_cfg(cls, guidance_boundary: Optional[float] = None) -> GuidanceConfig:
        """Default guidance recipe: piecewise CFG ``[3.0, 4.0]`` at the expert-routing boundary.

        Wan2.2 T2V A14B uses CFG 3.0 for the low-noise expert (``t < boundary``)
        and 4.0 for the high-noise expert (``t >= boundary``). ``guidance_boundary``
        defaults to the model's base ``DEFAULT_GUIDANCE_BOUNDARY``. Experiments
        may override via ``config.model.guidance_cfg``.
        """
        g = GuidanceConfig()
        g.cfg.scale = [3.0, 4.0]
        g.cfg.boundary = [cls.DEFAULT_GUIDANCE_BOUNDARY if guidance_boundary is None else guidance_boundary]
        return g

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        sample_steps: int = 40,
        sample_type: str = "ode",
        shift: float = 12.0,
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Multistep sample with MoE routing and CFG via ``predict_with_guidance``.

        ``predict_with_guidance(self, ...)`` runs the cond / cfg-uncond forwards
        through ``forward``, so the per-sample high/low expert routing is
        applied to every guidance pass. Skip-layer guidance is driven by
        ``guidance_cfg.cfg.skip_layers`` (+ boundary).
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )

        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg(self.guidance_boundary)

        scheduler, t_grid = self.sample_scheduler_and_grid(sample_steps, flow_shift=shift, device=noise.device)
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])

        for sched_step, t in tqdm(zip(scheduler.timesteps, t_grid, strict=True), total=len(t_grid), desc="Sampling"):
            t = t.expand(latents.shape[0])

            flow_pred = predict_with_guidance(
                self,
                latents,
                t,
                guidance_cfg=guidance_cfg,
                condition=condition,
                neg_condition=neg_condition,
                fwd_pred_type="flow",
            )

            latents = scheduler.step(flow_pred, sched_step, latents, return_dict=False)[0]

        return latents
