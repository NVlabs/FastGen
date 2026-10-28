# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel Decoding Distillation (PDD) network wrapper.

``make_pdd`` turns any compatible ``FastGenNetwork`` whose final output layer
is an ``nn.Linear`` into its PDD variant; a small ``PDDSpec`` supplies the
architecture-specific details (where the final linear lives and how the PDD
head axis is interleaved into its output features)::

    WanPDD = make_pdd(
        Wan,
        PDDSpec(
            final_linear="transformer.proj_out",
            channel_layout="patch_major",
            out_channels_path="transformer.config.out_channels",  # optional; where C lives
        ),
        module=__name__,
    )
    net = WanPDD(pdd_steps=128, student_grid_cfg=..., model_id_or_local_path=...)

The result is ``type("WanPDD", (PDDMixin, Wan), ...)``, so it inherits the
base network's behavior (FSDP sharding, meta-device init, checkpoint loading,
conditioning, ``to``) and ``isinstance(net, Wan)`` keeps working. Multimodal
networks (e.g. LTX-2) pass ``{modality: path}`` mappings; every listed
projection is widened and the dict output is split per modality.

PDD heads are parameterized as each modality's native RF direction
``q = dx/dsigma_m = eps - x0`` at the start of their fixed base-grid interval.
When teacher parameters are copied into the widened projection, the constant
per-head factor from the teacher's output convention to ``dx/dsigma_m(t_j)``
is folded into head ``j``'s parameters (``pdd_teacher_head_scale``).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Union, Tuple
import contextlib

import torch
import torch.nn.functional as F

from einops import rearrange

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import ShiftedRFNoiseSchedule
from fastgen.utils.tensor import assert_tensor, einsum_without_autocast, expand_like, high_precision_dtype
from fastgen.utils.basic_utils import nested_get, nested_set
from fastgen.utils.tree_utils import (
    DICT_TYPES,
    cast_like,
    get_batch_size,
    get_device,
    get_dtype,
    per_modality,
    to,
)
import fastgen.utils.logging_utils as logger


# Supported output-channel layouts for the widened final linear.
#   - "patch_major":   teacher rows are laid out as (patch, C); heads are inserted
#                       between them  -> (patch, heads, C).
#   - "channel_major": teacher rows are laid out as (C, patch); heads are inserted
#                       before the channel axis -> (heads, C, patch).
# In both cases the rule is: the head axis is inserted immediately *before* the
# channel axis so that, after the architecture's own unpatchify/unpack, the
# output channel dim reads ``(heads, C)`` and can be split with
# ``output.unflatten(channel_dim, (grid_size, -1))``.
CHANNEL_LAYOUTS = ("patch_major", "channel_major")

# Single-predecessor explicit RK tableaus on the shared primary clock. Each
# stage ``(fraction, source)`` evaluates the teacher at the shared time whose
# primary sigma is ``sigma_p + fraction * delta_sigma_p`` after an Euler step
# along the ``source``-th previously computed native velocity;
# ``output_weights`` mix all stage velocities into the interval mean.
# ``euler`` is the zero-stage tableau: one native evaluation at the interval
# start. ``kutta4`` is 4th order only when every modality shares the primary
# warp; with different shifts the native quadrature is 2nd order like
# ``heun``. ``terminal_fallback_stage`` handles solvers whose trailing
# ``fraction == 1.0`` stage would supervise from a teacher evaluation at
# ``t = 0``: samples ending on the clean endpoint blend to that earlier stage
# instead (Euler for heun, the midpoint stage for kutta4).
RK_STAGES: Dict[str, Dict[str, Any]] = {
    "euler": {
        "stages": (),
        "output_weights": (1.0,),
        "terminal_fallback_stage": None,
    },
    "midpoint": {
        "stages": ((0.5, 0),),
        "output_weights": (0.0, 1.0),
        "terminal_fallback_stage": None,
    },
    "heun": {
        "stages": ((1.0, 0),),
        "output_weights": (0.5, 0.5),
        "terminal_fallback_stage": 0,
    },
    "kutta4": {
        "stages": ((0.5, 0), (0.5, 1), (1.0, 2)),
        "output_weights": (1 / 6, 2 / 6, 2 / 6, 1 / 6),
        "terminal_fallback_stage": 1,
    },
}


def _validate_finite_nonzero_tensor(value: torch.Tensor, name: str = "value") -> None:
    assert_tensor(
        torch.all(torch.isfinite(value) & (value != 0)),
        f"{name} must be finite and non-zero",
        error_type=ValueError,
    )


@dataclass
class PDDSpec:
    """Per-architecture adapter describing the final linear layer(s) for PDD.

    Every field accepts a single value (unimodal network, tensor output) or a
    ``{modality: value}`` mapping (one output projection per modality, dict
    output); single values broadcast across a mapping-valued ``final_linear``.

    Args:
        final_linear: Dotted attribute path to the final ``nn.Linear`` from
            the network instance (e.g. ``"transformer.proj_out"``), or
            ``{modality: path}`` for a multimodal network.
        channel_layout: Axis order of the teacher's output features:
            ``"patch_major"`` (patch axis before channel) or
            ``"channel_major"`` (channel before patch). Determines how
            the PDD head axis is interleaved so the output channel axis reads
            ``(heads, C)``. Required, because ``patch_major`` additionally
            needs ``out_channels`` while ``channel_major`` needs nothing.
        out_channels_path: Optional dotted path to the latent channel count
            ``C`` (e.g. ``"transformer.config.out_channels"``). Used only for
            ``patch_major``, unless overridden at construction.
    """

    final_linear: Union[str, Mapping[str, str]]
    channel_layout: Union[str, Mapping[str, str]]
    out_channels_path: Optional[Union[str, Mapping[str, str]]] = None

    def __post_init__(self):
        for key in self.final_linears:
            layout = self.resolve(self.channel_layout, key)
            if layout not in CHANNEL_LAYOUTS:
                raise ValueError(
                    f"channel_layout must be one of {CHANNEL_LAYOUTS}, got {layout!r}"
                    + (f" for modality {key!r}" if key is not None else "")
                )

    @property
    def final_linears(self) -> Dict[Optional[str], str]:
        """``{modality: path}``; the sole key is ``None`` for a unimodal spec."""
        if isinstance(self.final_linear, str):
            return {None: self.final_linear}
        return dict(self.final_linear)

    @staticmethod
    def resolve(value: Any, key: Optional[str]) -> Any:
        """Read a per-modality spec field for *key*, or broadcast a single value."""
        if key is not None and isinstance(value, Mapping):
            return value.get(key)
        return value


class PDDLinear(torch.nn.Linear):
    """Final linear layer for Parallel Decoding Distillation.

    Widens a teacher ``nn.Linear`` so its output features are repeated
    ``grid_size`` times (one native head prediction per base-grid interval,
    evaluated at that interval's start node).
    An optional ``teacher_scale`` multiplies each copied head's parameters by a
    fixed per-head factor, converting the teacher's output convention into the
    heads' native prediction space at their interval-start times.
    The layer can also return a *fused* output: a weighted average
    of the weight slices for a contiguous block ``[step_index, next_step_index)``
    of base-grid intervals, collapsing that block into a single effective linear.

    Fusion is activated by the ``fuse`` context manager. Its temporary state is
    read by an ordinary ``forward(input)`` call, so inline final projections need
    no architecture-specific seam. It is FSDP-safe because the fused weight is
    computed inside ``forward`` where the full weight is all-gathered.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        patch_size: int,
        grid_size: int,
        bias: bool = True,
        channel_layout: str = "patch_major",
        teacher_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if channel_layout not in CHANNEL_LAYOUTS:
            raise ValueError(f"channel_layout must be one of {CHANNEL_LAYOUTS}, got {channel_layout!r}")
        super().__init__(in_features, out_features * grid_size, bias, **kwargs)
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.channel_layout = channel_layout

        if teacher_scale is None:
            self._teacher_scale = None
        else:
            # Explicit CPU placement keeps the fold real inside ``with torch.device("meta")``,
            # and a plain tuple keeps it out of state_dict.
            scale = torch.as_tensor(teacher_scale, device="cpu", dtype=torch.float64)
            if scale.shape != (grid_size,):
                raise ValueError(f"teacher_scale must have shape [{grid_size}], got {tuple(scale.shape)}")
            _validate_finite_nonzero_tensor(scale, "teacher_scale")
            self._teacher_scale = tuple(scale.tolist())

        # Transient fusion state is a plain tuple -- not a registered buffer --
        # so it stays out of state_dict and can be restored as one value.
        self._fusion_state: Optional[
            Tuple[Optional[Union[int, torch.Tensor]], Optional[Union[int, torch.Tensor]], torch.Tensor, torch.dtype]
        ] = None

    @contextlib.contextmanager
    def fuse(
        self,
        step_index: Optional[Union[int, torch.Tensor]],
        next_step_index: Optional[Union[int, torch.Tensor]],
        weights: torch.Tensor,
        precision_decoding: torch.dtype,
    ):
        """Temporarily make ``forward(input)`` return the fused output.

        *weights* are this modality's per-interval weights over the whole base
        grid, normalized by the block's total weight: the layer emits the
        heads' weighted mean and the caller scales it by that total.
        ``[grid_size]`` uses the shared contiguous slice named by the int indices;
        ``[batch, grid_size]`` is already masked per sample, so its indices go
        unused and may be ``None``. ``precision_decoding`` is used only for the
        fused-parameter head-axis accumulation, which runs outside autocast; the
        weights keep their incoming scheduler precision until then, and the resulting
        fused parameters are restored to the layer's model dtype before the projection runs.

        The previous fusion state is restored on exit, so nested/re-entrant use
        is safe.
        """
        previous = self._fusion_state
        self._fusion_state = (step_index, next_step_index, weights, precision_decoding)
        try:
            yield self
        finally:
            self._fusion_state = previous

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Repeat a single-head teacher checkpoint; leave a multi-head PDD checkpoint unchanged."""
        single_out_features = self.out_features // self.grid_size

        names = ["weight"] if self.bias is None else ["weight", "bias"]
        for name in names:
            key = prefix + name
            if key not in state_dict:
                continue
            tensor = state_dict[key]
            if tensor.shape[0] == single_out_features:
                logger.info(f"Repeating {key} grid_size={self.grid_size} times ({self.channel_layout})")
                state_dict[key] = self._repeat_teacher_parameter(tensor)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _repeat_teacher_parameter(self, tensor: torch.Tensor) -> torch.Tensor:
        """Repeat one teacher parameter in this linear's output layout, applying the fold.

        Both layouts insert the head axis immediately before the channel axis;
        ``patch_major`` groups rows as ``(patch, heads, C)`` and
        ``channel_major`` is its ``patch = 1`` case, ``(heads, C, patch)``.
        """
        patch = self.patch_size if self.channel_layout == "patch_major" else 1
        wide = torch.stack([tensor.unflatten(0, (patch, -1))] * self.grid_size, dim=1)
        if self._teacher_scale is not None:
            dtype = high_precision_dtype(tensor)
            scale = torch.tensor(self._teacher_scale, device=tensor.device, dtype=dtype)
            scale = scale.reshape(1, self.grid_size, *((1,) * (wide.ndim - 2)))
            wide = (wide.to(dtype) * scale).to(tensor.dtype)
        return wide.flatten(0, 2)

    def _fused_params(
        self,
        step_index: Optional[Union[int, torch.Tensor]],
        next_step_index: Optional[Union[int, torch.Tensor]],
        weights: torch.Tensor,
        precision_decoding: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute a shared or per-sample fused weight (and bias) for a block.

        *weights* arrive normalized by their block total (see ``fuse``). ``weights[N]`` uses a
        single contiguous slice and returns an ordinary ``[O, I]`` linear weight. ``weights[B, N]``
        is already masked per sample and returns ``[B, O, I]``. The latter keeps mixed-timestep
        batches fused without materialising ``B x N`` head outputs.
        """
        if weights.ndim not in (1, 2) or weights.shape[-1] != self.grid_size:
            raise ValueError(
                f"fusion weights must have shape [grid_size] or [batch, grid_size], got {tuple(weights.shape)}"
            )
        if not weights.is_floating_point():
            raise TypeError(f"fusion weights must be floating point, got {weights.dtype}")
        batched = weights.ndim == 2
        if not batched and not (
            step_index is not None
            and next_step_index is not None
            and 0 <= step_index < next_step_index <= self.grid_size
        ):
            raise ValueError(
                "shared fusion weights require 0 <= step_index < next_step_index <= grid_size, "
                f"got {step_index=}, {next_step_index=}, grid_size={self.grid_size}"
            )

        # A shared block carries the full integration grid plus its contiguous
        # head slice. Mixed batches are already masked per row and therefore
        # retain the full head axis. Applying one selector to both tensors keeps
        # their last/head axes aligned from this point onward.
        head_slice = slice(None) if batched else slice(step_index, next_step_index)

        if self.channel_layout == "patch_major":
            # In the unflatten output features, patch appears before channel axis
            weight = rearrange(self.weight, "(p n c) ... -> n (p c) ...", p=self.patch_size, n=self.grid_size)
        else:  # channel_major: grid is outermost -> only peel n=grid off the front
            weight = rearrange(self.weight, "(n rest) ... -> n rest ...", n=self.grid_size)
        weight = weight[head_slice]

        # The caller normalized the weights in scheduler precision and multiplies
        # the emitted mean by the block total, so together they represent
        # ``sum_j W_j pred_j``. Only the fused-parameter accumulation is cast.
        weights = weights[..., head_slice].to(dtype=precision_decoding)
        equation = "bn,n... -> b..." if batched else "n,n... -> ..."

        def fuse_parameter(parameter: torch.Tensor) -> torch.Tensor:
            fused = einsum_without_autocast(equation, weights, parameter.to(dtype=precision_decoding))
            return fused.to(dtype=parameter.dtype)

        fused_weight = fuse_parameter(weight)

        if self.bias is None:
            fused_bias = None
        else:
            if self.channel_layout == "patch_major":
                bias = rearrange(self.bias, "(p n c) -> n (p c)", p=self.patch_size, n=self.grid_size)
            else:  # channel_major
                bias = rearrange(self.bias, "(n rest) -> n rest", n=self.grid_size)
            fused_bias = fuse_parameter(bias[head_slice])

        return fused_weight, fused_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        fusion_state = self._fusion_state
        if fusion_state is None:
            return super().forward(input)

        fused_weight, fused_bias = self._fused_params(*fusion_state)
        if fused_weight.ndim == self.weight.ndim:
            return F.linear(input, fused_weight, fused_bias)

        if input.shape[0] != fused_weight.shape[0]:
            raise ValueError(
                f"batched fused weight has batch {fused_weight.shape[0]}, input has batch {input.shape[0]}"
            )
        output = torch.einsum("b...i,boi->b...o", input, fused_weight)
        if fused_bias is not None:
            output = output + fused_bias.reshape(fused_bias.shape[0], *((1,) * (output.ndim - 2)), fused_bias.shape[1])
        return output


class PDDMixin:
    """Mixin that turns a ``FastGenNetwork`` into its PDD variant.

    Combine with a base network via ``make_pdd``. The mixin:
      1. widens each final ``nn.Linear`` named by the spec into a
         ``PDDLinear`` (out_features multiplied by ``pdd_steps``), repeating
         teacher weights in a layout-aware way and folding in the per-head
         factor that makes head ``j`` emit ``dx/dsigma_m`` at its base-grid
         interval's start time;
      2. owns the fixed number of PDD intervals/output heads;
      3. implements ``forward(step_index=...)`` for base-grid PDD training and
         optional integer-addressed block fusion;

    A multimodal spec names one projection per modality; all of them are widened
    and fused together, and the base network's ``{modality: Tensor}`` output is
    split per modality. Modalities the base network did not enable (via its
    ``modalities`` attribute) are skipped.
    """

    # Set by ``make_pdd``; may also be passed as ``pdd_spec=`` to ``__init__``.
    PDD_SPEC: Optional[PDDSpec] = None

    def __init__(
        self,
        pdd_steps: int,
        out_channels: Optional[Union[int, Mapping[str, int]]] = None,
        *args,
        pdd_spec: Optional[PDDSpec] = None,
        **kwargs,
    ):
        """Build the base network, then widen it to ``pdd_steps`` output heads."""
        if not isinstance(pdd_steps, int) or isinstance(pdd_steps, bool) or pdd_steps < 2:
            raise ValueError(f"pdd_steps must be an int >= 2, got {pdd_steps!r}")

        # Build the underlying network first (mixin precedes the base in MRO).
        super().__init__(*args, **kwargs)

        spec = pdd_spec if pdd_spec is not None else type(self).PDD_SPEC
        if spec is None:
            raise ValueError(
                "PDDMixin requires a PDDSpec. Build the class via make_pdd(base, spec) "
                "or pass pdd_spec=... to the constructor."
            )

        self._pdd_spec: PDDSpec = spec
        self.pdd_steps = pdd_steps
        self._validate_pdd_schedule_contract()
        student_base_steps = self.noise_scheduler.student_base_steps
        if student_base_steps is None:
            raise ValueError(
                f"{type(self).__name__} requires an anchored student grid: configure "
                "student_grid_cfg.grid_kwargs.base_steps or explicit timesteps."
            )
        if self.pdd_steps != student_base_steps:
            raise ValueError(
                f"{type(self).__name__} requires pdd_steps={self.pdd_steps} to match the scheduler's "
                f"student_base_steps={student_base_steps}."
            )

        # The base network may have been created with parameters on the meta device.
        is_meta = self._is_in_meta_context()

        # Copy the base projection only when the base loaded pretrained weights and is not on meta; a later
        # single-head checkpoint load repeats and folds it anyway (PDDLinear._load_from_state_dict).
        load_pretrained = kwargs.get("load_pretrained", True)

        # We deliberately keep paths rather than the layers as nn.Module attributes here,
        # which would double-register them as child modules and duplicate their params in
        # state_dict / FSDP. The stored paths preserve the spec's scalar-or-dict modality
        # structure so ``per_modality`` can dispatch over the resolved layers uniformly.
        final_linear_paths: Dict[Optional[str], str] = {}
        enabled = getattr(self, "modalities", None)
        for key, path in spec.final_linears.items():
            if key is not None and enabled is not None and key not in enabled:
                logger.info(f"[PDD] Skipping '{path}': modality '{key}' is not enabled on this network")
                continue
            self._widen_final_linear(
                key,
                path,
                out_channels=spec.resolve(out_channels, key),
                load_pretrained=load_pretrained,
                is_meta=is_meta,
            )
            final_linear_paths[key] = path
        if not final_linear_paths:
            raise ValueError(
                f"PDDSpec named no widenable output projection for {type(self).__name__} "
                f"(spec paths {sorted(spec.final_linears.values())}, "
                f"enabled modalities {enabled})."
            )
        self._pdd_final_linear_paths: Union[str, Dict[str, str]] = (
            final_linear_paths[None]
            if isinstance(spec.final_linear, str)
            else {key: path for key, path in final_linear_paths.items() if key is not None}
        )

    def pdd_base_grid(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Build the canonical PDD base grid in scheduler timestep precision."""
        return self.noise_scheduler.student_grid(
            self.pdd_steps,
            device=device,
            dtype=self.noise_scheduler.t_precision,
        )

    def pdd_teacher_head_scale(self, key: Optional[str]) -> Optional[torch.Tensor]:
        """Per-head factor taking the teacher's native output to ``dx/dsigma_m(t_j)``.

        A flow-emitting teacher needs ``1 / sigma'_m(t_j)`` folded into the
        copied parameters of interval head ``j``. An unwarped modality needs
        neither a fold nor a grid lookup here; its base-grid configuration is
        therefore validated by the first grid-consuming operation instead.
        """
        schedules = self.noise_scheduler.schedules
        schedule = schedules[key] if isinstance(schedules, DICT_TYPES) else schedules
        if schedule.is_unwarped:
            return None
        # The fold is computed on CPU so it stays real inside ``with torch.device("meta")``.
        base_grid = self.pdd_base_grid(device=torch.device("cpu"))
        return 1.0 / schedule.sigma_prime(base_grid[:-1])

    def _make_pdd_linear(
        self,
        key: Optional[str],
        orig: torch.nn.Linear,
        patch_factor: int,
        layout: str,
    ) -> PDDLinear:
        """Construct one widened head with its teacher-parameter fold."""
        return PDDLinear(
            orig.in_features,
            orig.out_features,
            patch_size=patch_factor,
            grid_size=self.pdd_steps,
            bias=orig.bias is not None,
            channel_layout=layout,
            teacher_scale=self.pdd_teacher_head_scale(key),
            dtype=orig.weight.dtype,
            device=orig.weight.device,
        )

    def _widen_final_linear(
        self,
        key: Optional[str],
        path: str,
        out_channels: Optional[int],
        load_pretrained: bool,
        is_meta: bool,
    ) -> None:
        """Replace the ``nn.Linear`` at *path* with a ``pdd_steps``-wide ``PDDLinear``."""
        spec = self._pdd_spec
        layout = spec.resolve(spec.channel_layout, key)

        orig = nested_get(self, path, separator=".")
        if not isinstance(orig, torch.nn.Linear):
            raise TypeError(
                f"PDD final layer at '{path}' must be nn.Linear, "
                f"got {type(orig).__name__}. PDD only supports linear final layers."
            )

        if layout == "channel_major":
            # grid is the outermost axis -> patch grouping is not needed
            patch_factor = 1
        else:  # patch_major: C is required to place grid between patch and channels
            out_channels_path = spec.resolve(spec.out_channels_path, key)
            if out_channels is None and out_channels_path is not None:
                out_channels = nested_get(self, out_channels_path, separator=".")
            if out_channels is None:
                raise ValueError(
                    f"patch_major layout needs out_channels: pass it at init or set "
                    f"out_channels_path in the spec (final_linear='{path}')."
                )
            if orig.out_features % out_channels != 0:
                raise ValueError(
                    f"out_features={orig.out_features} is not divisible by " f"out_channels={out_channels}."
                )
            patch_factor = orig.out_features // out_channels

        new_linear = self._make_pdd_linear(key, orig, patch_factor, layout)

        if load_pretrained and not is_meta:
            # Triggers PDDLinear._load_from_state_dict -> layout-aware repeat.
            new_linear.load_state_dict(orig.state_dict())
            init_mode = "pretrained"
        elif is_meta:
            init_mode = "meta (weights via FSDP sync)"
        else:
            init_mode = "random (no pretrained weights)"

        logger.info(
            f"[PDD] Replaced '{path}': {orig.out_features} -> {new_linear.out_features} "
            f"(pdd_steps={self.pdd_steps}, patch_factor={patch_factor}, "
            f"layout={layout}, init={init_mode})"
        )

        nested_set(self, path, new_linear, separator=".")

    @property
    def pdd_linear(self) -> Union[PDDLinear, Dict[str, PDDLinear]]:
        """The widened final linear, preserving the spec's unimodal or multimodal structure."""
        return per_modality(
            lambda path: nested_get(self, path, separator="."),
            self._pdd_final_linear_paths,
        )

    def _validate_pdd_schedule_contract(self) -> None:
        """Validate this mixin's RF-flow contract.

        Other noise schedules or prediction types need their own
        scheduler-specific mixin, which overrides this hook while reusing
        ``PDDMixin``'s projection widening and forward machinery.
        """
        owner = type(self).__name__
        if self.net_pred_type != "flow":
            raise ValueError(
                f"{owner} requires net_pred_type='flow', got {self.net_pred_type!r}. "
                "Other prediction types need their own scheduler-specific PDD mixin."
            )

        schedules = self.noise_scheduler.schedules
        items = schedules.items() if isinstance(schedules, DICT_TYPES) else [(None, schedules)]
        for key, schedule in items:
            modality = "" if key is None else f" for modality {key!r}"
            if not isinstance(schedule, ShiftedRFNoiseSchedule):
                raise ValueError(
                    f"{owner} requires an 'rf' or 'shifted_rf' schedule{modality}, got {type(schedule).__name__}. "
                    "Other noise schedules need their own scheduler-specific PDD mixin."
                )
            if schedule.min_t != 0.0:
                raise ValueError(
                    f"{owner} requires min_t=0{modality} because the PDD grid "
                    f"ends at the canonical zero endpoint, got {schedule.min_t}."
                )

    def _validate_pdd_indices(
        self,
        step_index: Union[int, torch.Tensor],
        next_step_index: Optional[Union[int, torch.Tensor]] = None,
        allow_empty: bool = False,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Validate shared-int or batched-long base-grid indices.

        With only ``step_index``, checks one index ``0 <= i < pdd_steps``; with
        ``next_step_index``, checks a half-open block
        ``0 <= start < end <= pdd_steps`` (``start <= end`` when
        ``allow_empty``). Ints are validated in Python without a device sync;
        batched ``long`` tensors are validated on-device after structural
        asserts against ``batch_size``/``device`` (defaulting to
        ``step_index``'s own shape and device).
        """
        indices = (step_index,) if next_step_index is None else (step_index, next_step_index)
        shared = isinstance(step_index, int)
        if any(isinstance(index, int) != shared for index in indices):
            raise TypeError("PDD block endpoints must both be integers or both be batched tensors")
        relation = "<=" if allow_empty else "<"

        if shared:
            if next_step_index is None:
                if not 0 <= step_index < self.pdd_steps:
                    raise ValueError(f"step_index must be in [0, {self.pdd_steps - 1}], got {step_index}")
                return
            ordered = step_index <= next_step_index if allow_empty else step_index < next_step_index
            if not 0 <= step_index or not ordered or next_step_index > self.pdd_steps:
                raise ValueError(
                    "PDD block indices must satisfy "
                    f"0 <= step_index {relation} next_step_index <= {self.pdd_steps}, got "
                    f"{step_index=} and {next_step_index=}"
                )
            return

        expected_shape = indices[0].shape if batch_size is None else (batch_size,)
        expected_device = indices[0].device if device is None else device
        for index in indices:
            if not isinstance(index, torch.Tensor) or index.dtype != torch.long:
                raise TypeError(f"Batched PDD block indices must be long tensors, got {type(index).__name__}")
            if index.ndim != 1 or index.shape != expected_shape:
                raise ValueError(
                    f"Batched PDD block indices must have shape {tuple(expected_shape)}, got {tuple(index.shape)}"
                )
            if index.device != expected_device:
                raise ValueError(f"Batched PDD block indices must live on {expected_device}, got {index.device}")

        if next_step_index is None:
            valid = torch.all((step_index >= 0) & (step_index < self.pdd_steps))
            message = f"step_index must be in [0, {self.pdd_steps - 1}]"
        else:
            ordered = step_index <= next_step_index if allow_empty else step_index < next_step_index
            valid = torch.all((step_index >= 0) & ordered & (next_step_index <= self.pdd_steps))
            message = f"Every PDD block must satisfy 0 <= start {relation} end <= {self.pdd_steps}"
        assert_tensor(valid, message, error_type=ValueError)

    @property
    def pdd_primary_schedule(self) -> ShiftedRFNoiseSchedule:
        """RF schedule defining the synchronous coordinate used by teacher RK.

        A composite network has one shared model time but may attach a distinct
        RF warp to every modality. Its configured primary schedule supplies the
        single sigma coordinate used for all joint teacher evaluations;
        unimodal networks use their sole schedule.
        """
        schedules = self.noise_scheduler.schedules
        return self.noise_scheduler.primary if isinstance(schedules, DICT_TYPES) else self.noise_scheduler

    def pdd_advance_prediction(self, x_t: Any, prediction: Any, weight: Any) -> Any:
        """Apply ``state + weight * dx/dsigma`` at the state's arithmetic boundary."""
        weight = to(weight, device=get_device(x_t), dtype=get_dtype(x_t))
        return per_modality(
            lambda state, value, scale: state + value * expand_like(scale, value),
            x_t,
            prediction,
            weight,
        )

    def pdd_block_weights(
        self,
        step_index: Union[int, torch.Tensor],
        next_step_index: Union[int, torch.Tensor],
        base_grid: torch.Tensor,
        allow_empty: bool = False,
    ) -> Tuple[Any, Any]:
        """Return each block's interval weights normalized by its total, and that total.

        A PDD head stores each modality's native RF direction
        ``q = dx/dsigma_m = eps - x0``; freezing it on interval ``j`` advances
        the state by ``w_j = sigma_m(t_{j+1}) - sigma_m(t_j)``. The fused and the
        materialized advance both average the heads with the returned
        ``w_j / total`` and scale the mean by the total. The ``w_j`` telescope,
        so the total is the block endpoints' ``sigma_m(t_end) - sigma_m(t_start)``:
        being elementwise rather than a sum, it is the same float for a shared,
        per-sample or multi-endpoint block.

        Int endpoints give ``[pdd_steps]`` weights and ``[batch]`` endpoints
        ``[batch, pdd_steps]``; a ``[batch, num_endpoints]`` ``next_step_index``
        (several endpoints sharing each row's start) gives
        ``[batch, num_endpoints, pdd_steps]``. Weights outside a block are zero,
        and an empty block, allowed only with ``allow_empty``, also totals zero.
        """
        # A mixed int/tensor pair is not multi-endpoint; ``_validate_pdd_indices`` rejects it below.
        multi_endpoint = (
            isinstance(step_index, torch.Tensor)
            and isinstance(next_step_index, torch.Tensor)
            and next_step_index.ndim == 2
        )
        if multi_endpoint:
            if next_step_index.shape[0] != step_index.shape[0]:
                raise ValueError(
                    f"next_step_index must have shape [{step_index.shape[0]}, num_endpoints], "
                    f"got {tuple(next_step_index.shape)}"
                )
            self._validate_pdd_indices(
                step_index.repeat_interleave(next_step_index.shape[1]),
                next_step_index.reshape(-1),
                allow_empty=allow_empty,
                device=base_grid.device,
            )
            step_index = step_index[:, None]
        else:
            self._validate_pdd_indices(
                step_index,
                next_step_index,
                allow_empty=allow_empty,
                device=base_grid.device,
            )

        start_node, end_node = base_grid[step_index], base_grid[next_step_index]
        grid_index = torch.arange(self.pdd_steps, device=base_grid.device)
        if isinstance(step_index, int):
            active = (grid_index >= step_index) & (grid_index < next_step_index)
        else:
            active = (grid_index >= step_index[..., None]) & (grid_index < next_step_index[..., None])

        total = per_modality(
            lambda schedule: schedule.determinant(start_node, end_node),
            self.noise_scheduler.schedules,
        )

        def normalize(schedule: ShiftedRFNoiseSchedule, block_total: torch.Tensor) -> torch.Tensor:
            interval_weights = schedule.determinant(base_grid[:-1], base_grid[1:])
            _validate_finite_nonzero_tensor(interval_weights, "PDD integration weight")
            # An empty block's weights are all masked, so dividing them by one keeps them zero.
            divisor = torch.where(active.any(dim=-1), block_total, torch.ones_like(block_total))
            _validate_finite_nonzero_tensor(divisor, "PDD block total weight")
            return torch.where(active, interval_weights, torch.zeros_like(interval_weights)) / divisor[..., None]

        return per_modality(normalize, self.noise_scheduler.schedules, total), total

    def pdd_advance_blocks(
        self,
        x_t: Any,
        pred: Any,
        step_index: Union[int, torch.Tensor],
        advance_index: Union[int, torch.Tensor],
        base_grid: torch.Tensor,
        fused: bool = False,
        allow_empty: bool = False,
        precision_decoding: Optional[torch.dtype] = None,
    ) -> Any:
        """Advance one RF block to one or several endpoints.

        The endpoint layout selects the mode:

        * int ``step_index`` and ``advance_index``: one half-open block shared by
          the batch; ``fused=True`` accepts a fused mean sigma-velocity
          instead of materialized heads.
        * ``[batch]`` long tensors: per-sample blocks with one endpoint each.
        * ``[batch, num_endpoints]`` ``advance_index``: several (possibly
          repeated, unsorted) endpoints sharing the batched start. The
          returned states gain an endpoint dim with ``result[:, k]`` matching
          the single-endpoint advance to ``advance_index[:, k]``; requires
          materialized heads.

        Materialized heads are reduced to this RF mixin's weighted mean
        sigma-velocity. Endpoints equal to the start are allowed only when
        ``allow_empty=True`` and produce an identity update.

        ``precision_decoding`` scopes the block arithmetic: state and
        prediction are promoted for the integration and the advanced state is
        cast back to the incoming state's dtype. By default nothing is cast.
        The head reduction runs outside autocast, so AMP does not lower it;
        FP32 matmuls still follow the global TF32 setting.
        """
        multi_endpoint = isinstance(advance_index, torch.Tensor) and advance_index.ndim == 2
        if multi_endpoint and fused:
            raise ValueError("Multi-endpoint PDD advances require materialized heads, not a fused prediction.")

        input_dtype = get_dtype(x_t)
        if precision_decoding is not None:
            x_t = to(x_t, dtype=precision_decoding)
            pred = to(pred, dtype=precision_decoding)

        state_dtype = get_dtype(x_t)
        pred_dtype = get_dtype(pred)
        assert (
            state_dtype == pred_dtype
        ), f"PDD state and prediction must have the same dtype, got {state_dtype} and {pred_dtype}"
        # A mixed int/tensor pair is not shared; ``pdd_block_weights`` rejects it below.
        shared_block = isinstance(step_index, int) and isinstance(advance_index, int)
        if shared_block and step_index == advance_index:
            self._validate_pdd_indices(step_index, advance_index, allow_empty=allow_empty)
            return x_t if precision_decoding is None else to(x_t, dtype=input_dtype)
        weights, block_scale = self.pdd_block_weights(
            step_index,
            advance_index,
            base_grid,
            allow_empty=allow_empty,
        )
        if shared_block:
            head_slice, equation = slice(step_index, advance_index), "n,bn...->b..."
        else:
            head_slice = slice(None)
            equation = "bkn,bn...->bk..." if multi_endpoint else "bn,bn...->b..."

        if not fused:

            def reduce_heads(prediction: torch.Tensor, normalized_weights: torch.Tensor) -> torch.Tensor:
                if prediction.ndim < 2 or prediction.shape[1] != self.pdd_steps:
                    raise ValueError(
                        "Materialized PDD prediction must have shape [batch, pdd_steps, ...], "
                        f"got {tuple(prediction.shape)}"
                    )
                head_weights = normalized_weights[..., head_slice].to(dtype=prediction.dtype)
                return einsum_without_autocast(equation, head_weights, prediction[:, head_slice])

            mean_sigma_velocity = per_modality(reduce_heads, pred, weights)
        else:
            mean_sigma_velocity = pred

        # A total valid in scheduler precision can still under- or overflow
        # the state dtype, so validate the cast scale that the advance applies.
        # ``pdd_block_weights`` above already rejected a zero total for every
        # nonempty block, so a zero total marks an empty block's identity
        # update, which stays exempt.
        state_scale = to(block_scale, dtype=state_dtype)
        per_modality(
            lambda scale, total: _validate_finite_nonzero_tensor(
                torch.where(total != 0, scale, torch.ones_like(scale)),
                "PDD block scale in state dtype",
            ),
            state_scale,
            block_scale,
        )
        if multi_endpoint:
            x_t = per_modality(lambda state: state[:, None], x_t)
        advanced = self.pdd_advance_prediction(x_t, mean_sigma_velocity, state_scale)
        return advanced if precision_decoding is None else to(advanced, dtype=input_dtype)

    def pdd_teacher_target(
        self,
        x_t: Any,
        step_index: Union[int, torch.Tensor],
        base_grid: torch.Tensor,
        solver: str,
        predict_teacher: Callable[..., Any],
    ) -> Any:
        """Build the native sigma-velocity PDD target for one base-grid interval.

        ``predict_teacher`` owns guidance and the model/decoding precision
        boundary and returns the teacher's shared-time flow; the target is
        returned in the heads' native space: each modality's ``dx/dsigma_m``
        averaged over the interval, directly comparable to one student head.

        Multi-stage solvers place their stages at shared times chosen by the
        primary clock, while every modality keeps its native scaling and its
        own exact sigma increment per stage, so the mean is native by
        construction. With one warp for every modality the tableau order is
        preserved; ``kutta4`` degrades to 2nd order under different shifts
        (see ``RK_STAGES``).
        """
        t = base_grid[step_index]
        t_next = base_grid[step_index + 1]
        batch_size = get_batch_size(x_t)
        if t.ndim == 0:
            t = t.expand(batch_size)
            t_next = t_next.expand(batch_size)
        schedules = self.noise_scheduler.schedules

        def sigma_velocity(state: Any, time: torch.Tensor) -> Any:
            def convert(flow: torch.Tensor, schedule: ShiftedRFNoiseSchedule) -> torch.Tensor:
                dtype = high_precision_dtype(flow)
                sigma_prime = expand_like(schedule.sigma_prime(time).to(dtype=dtype), flow)
                return (flow.to(dtype=dtype) / sigma_prime).to(dtype=flow.dtype)

            return per_modality(convert, predict_teacher(state, time, fwd_pred_type=self.net_pred_type), schedules)

        if solver not in RK_STAGES:
            raise ValueError(f"RK type: {solver} is undefined")
        return self._pdd_rk_mean_velocity(x_t, t, t_next, base_grid, solver, sigma_velocity)

    def _pdd_rk_mean_velocity(
        self,
        x_t: Any,
        t: torch.Tensor,
        t_next: torch.Tensor,
        base_grid: torch.Tensor,
        solver: str,
        sigma_velocity: Callable[[Any, torch.Tensor], Any],
    ) -> Any:
        """Integrate one interval with an ``RK_STAGES`` tableau on the shared clock.

        Returns the per-modality interval mean of the native ``dx/dsigma_m``.
        Stage times come from the primary clock; each stage state advances
        every modality by its exact schedule-defined sigma increment to that
        shared time. The schedule reparameterization is therefore evaluated
        exactly, while the teacher dynamics retain the approximation order of
        the selected RK tableau. Samples whose interval
        ends on the clean endpoint blend to the solver's terminal fallback
        stage: the trailing stage still runs, so every rank issues identical
        forwards, but a teacher evaluation at ``t = 0`` never supervises.
        """
        tableau = RK_STAGES[solver]
        stages, output_weights = tableau["stages"], tableau["output_weights"]
        schedules = self.noise_scheduler.schedules
        primary = self.pdd_primary_schedule
        delta_sigma = primary.determinant(t, t_next)
        _validate_finite_nonzero_tensor(delta_sigma, "PDD primary-sigma interval")

        def stage_time(fraction: float) -> torch.Tensor:
            if fraction == 1.0:
                return t_next
            if primary.is_unwarped:
                return t + delta_sigma * fraction
            return primary.sigma_to_t(primary.sigma(t) + delta_sigma * fraction)

        velocities = [sigma_velocity(x_t, t)]
        if stages:
            x_t = cast_like(x_t, velocities[0])
        for fraction, source in stages:
            time = stage_time(fraction)
            stage_delta = per_modality(lambda schedule: schedule.determinant(t, time), schedules)
            stage_state = self.pdd_advance_prediction(x_t, velocities[source], stage_delta)
            velocities.append(sigma_velocity(stage_state, time))

        mean_sigma_velocity = per_modality(
            lambda *values: sum(weight * value for weight, value in zip(output_weights, values)),
            *velocities,
        )
        fallback_stage = tableau["terminal_fallback_stage"]
        if fallback_stage is None:
            return mean_sigma_velocity
        # The grid is descending; the terminal interval ends at ``base_grid[-1]``.
        mask_last_step = t_next <= base_grid[-1]
        return per_modality(
            lambda fallback, combined: torch.where(expand_like(mask_last_step, combined), fallback, combined),
            velocities[fallback_stage],
            mean_sigma_velocity,
        )

    def forward(
        self,
        x_t: Any,
        step_index: Union[int, torch.Tensor],
        condition: Optional[Any] = None,
        fused: bool = False,
        next_step_index: Optional[Union[int, torch.Tensor]] = None,
        precision_decoding: Optional[torch.dtype] = None,
        apply_output_conditioning: bool = False,
        **fwd_kwargs,
    ) -> Any:
        """Run PDD at a base-grid index, optionally fusing one interval block.

        ``fused=False`` is the training path and returns every native PDD head,
        unflattened as ``(pdd_steps, C, ...)``. ``fused=True`` requires
        ``next_step_index`` and returns the selected heads' block-weighted mean
        native sigma-velocity. PDD outputs are not ordinary single-time model
        predictions, so this API accepts no ``fwd_pred_type`` override, and the
        base network's output constraint, defined on such predictions, never
        applies: ``apply_output_conditioning`` must be False. Shared endpoints
        are Python integers; per-sample endpoints are ``torch.long[batch]``
        tensors.

        A fused prediction has ``x_t``'s dtype. ``precision_decoding`` controls
        only the fused-head parameter accumulation; when omitted, it defaults
        to ``x_t``'s dtype. Unfused heads retain the base network's native
        output dtype; consumers promote them at their loss, integration, or
        inversion precision boundary.
        """
        if "fwd_pred_type" in fwd_kwargs:
            raise TypeError(
                "PDD forward does not accept fwd_pred_type: heads and fused block "
                "predictions are native sigma-velocities, not single-time model outputs"
            )
        if apply_output_conditioning:
            raise NotImplementedError(
                "PDD forward does not apply output conditioning: the base network's constraint acts on "
                "single-time model outputs, not on native sigma-velocity heads"
            )
        if not isinstance(fused, bool):
            raise TypeError(f"fused must be bool, got {type(fused).__name__}")
        if not fused and next_step_index is not None:
            raise ValueError("next_step_index is only valid when fused=True")
        if fused and next_step_index is None:
            raise ValueError("fused PDD forward requires next_step_index")
        if not fused and precision_decoding is not None:
            raise ValueError("precision_decoding is only valid when fused=True")
        state_device = get_device(x_t)
        state_dtype = get_dtype(x_t)
        batch_size = get_batch_size(x_t)
        self._validate_pdd_indices(step_index, batch_size=batch_size, device=state_device)
        base_grid = self.pdd_base_grid(device=state_device)
        t = base_grid[step_index]
        if t.ndim == 0:
            t = t.expand(batch_size)

        with contextlib.ExitStack() as stack:
            if fused:
                # A shared int block passes full-grid weights plus its slice
                # indices; a batched block passes per-sample masked weights and
                # its tensor indices, which ``PDDLinear._fused_params`` ignores.
                weights, _ = self.pdd_block_weights(step_index, next_step_index, base_grid)
                if precision_decoding is None:
                    precision_decoding = state_dtype
                per_modality(
                    lambda linear, modality_weights: stack.enter_context(
                        linear.fuse(step_index, next_step_index, modality_weights, precision_decoding)
                    ),
                    self.pdd_linear,
                    weights,
                )
            # The wrapped base network still emits its configured shared-time
            # flow. Widened projection folds convert that output into the PDD
            # heads' native sigma-velocity coordinate.
            output = super().forward(
                x_t,
                t,
                condition,
                fwd_pred_type=self.net_pred_type,
                apply_output_conditioning=False,
                **fwd_kwargs,
            )

        if not isinstance(output, (torch.Tensor, dict)):
            raise TypeError(
                "PDD output prediction requires a tensor or per-modality dict from the base "
                f"network; got {type(output).__name__}. Feature/logvar returns are unsupported."
            )
        if not fused:
            return per_modality(lambda out: out.unflatten(1, (self.pdd_steps, -1)), output)

        return to(output, dtype=state_dtype)


def make_pdd(
    base_cls: type,
    spec: PDDSpec,
    name: Optional[str] = None,
    mixin_cls: type = PDDMixin,
    module: Optional[str] = None,
) -> type:
    """Create a PDD variant of ``base_cls`` bound to ``spec``.

    The result takes the base network arguments plus required ``pdd_steps=``
    and optional ``out_channels=``. Module-level wrappers should pass
    ``module=__name__`` so the generated class is importable/serializable as a
    network ``_target_`` from the module where it is bound.
    RF PDD use also needs an anchored ``student_grid_cfg`` whose base resolution
    matches ``pdd_steps``; a shifted schedule consumes that grid while folding
    the teacher projection during construction. ``mixin_cls`` defaults to the
    RF-specific ``PDDMixin``; another scheduler-specific mixin can reuse this
    factory.
    """
    if not issubclass(base_cls, FastGenNetwork):
        raise TypeError(f"PDD can only wrap a FastGenNetwork, got {base_cls.__name__}")
    if not isinstance(mixin_cls, type):
        raise TypeError(f"mixin_cls must be a class, got {type(mixin_cls).__name__}")
    if module is not None and not isinstance(module, str):
        raise TypeError(f"module must be a string or None, got {type(module).__name__}")
    name = name or f"{base_cls.__name__}PDD"
    namespace = {"PDD_SPEC": spec}
    if module is not None:
        namespace["__module__"] = module
    return type(name, (mixin_cls, base_cls), namespace)
