# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guidance schemas, schedulers, and the canonical ``predict_with_guidance`` applier.

Every guidance-bearing network's ``sample()`` and every training-time
teacher pass is one ``predict_with_guidance(net, x_t, t, ...)`` call.

``net`` is a ``FastGenNetwork``, or the ``SharedBaseTeacher`` view of one, which
runs it as its base model. Guidance composes its native predictions, as
a forward with the default ``fwd_pred_type`` and
``apply_output_conditioning=False`` returns them, and applies its
``transform_output`` once to the result: the conversion to the caller's
``fwd_pred_type`` and the output constraint. The terms are differences of
branches, so they compose the same in any prediction space; the rescale is not
linear and runs on the native predictions, unless a network's recipe rescales
in another space (LTX-2 rescales x0).

Cached predictions (``precomputed`` and ``return_cache``) are therefore native
and unconstrained. Code that computes a prediction for reuse, such as DMD2's
shared teacher pass, calls the network the same way. A
transformed prediction cannot be reused: a partial-strength blend would be
applied twice, and a full-strength blend cannot be inverted.

Branches, recipes and the cache use a working dtype per modality, the
``high_precision_dtype`` of its state and requested ``output_dtype``;
transforming the result rounds it once to ``output_dtype``: each modality's
state dtype for ``"same"``, and the working dtype itself for ``None``.
Branch forwards return the working dtype, and precomputed branches are cast to
it when they enter the cache. ``ctx.predict`` returns the cached branch
itself, so recipes must not modify branches in place. ``precomputed``,
``contexts`` and the cache use the branch keys of ``GuidanceConfig.pred_keys``;
``GuidanceContext`` validates these keys and dtypes.

Shared guidance terms define a schema, a slot on ``GuidanceConfig``, a branch
key in ``PRED_KEYS``, and a block helper. A term that only one network
implements is a ``GuidanceTerm`` field of a ``GuidanceConfig`` subclass, and
only that network's recipe accepts it; the subclass extends the methods of
``GuidanceConfig`` with its terms and their branches. The subclass's other
fields are options of that recipe. Networks own their complete recipe and may
reuse any of these helpers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Dict, Optional, Union

import attrs
import torch

from fastgen.networks.lora_utils import SharedBaseTeacher
from fastgen.networks.network import FastGenNetwork
from fastgen.utils.tensor import expand_like, high_precision_dtype
from fastgen.utils.tree_utils import (
    DICT_TYPES,
    DtypeOrDict,
    OutputDtype,
    TensorOrDict,
    assert_same,
    cast_like,
    get_device,
    per_modality,
    tmap,
    validate_output_dtype,
)


def cfg_rescale(
    guided: torch.Tensor,
    cond: torch.Tensor,
    rescale: Optional[float],
    eps: float = 1e-5,
    rescale_type: str = "std",
    rescale_dims: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Rescale *guided* toward the std/norm of *cond* on a single modality.

    ``factor = rescale * cond_stat / guided_stat + (1 - rescale)``;
    ``rescale=1.0`` is a pure-ratio rescale, ``rescale=0.0`` is a no-op.
    A ``"std"`` reduction over a single element has no spread to match and
    raises ``ValueError``.

    Multimodal inputs are not handled here; callers wrap with
    ``per_modality(cfg_rescale, guided, cond, rescale,
    rescale_type=rescale_type, rescale_dims=rescale_dims)``.  A scalar
    *rescale* broadcasts; pass ``{key: scale | None}`` for per-modality
    rescale (entries can be ``None`` to disable a single modality).
    *rescale_type* / *rescale_dims* always broadcast (kwargs are not
    indexed unless dict-shaped, and these are typically uniform).

    Args:
        guided: Guided prediction (typically the cfg-combined output).
        cond: Reference prediction whose statistic *guided* is rescaled toward.
        rescale: Strength in [0, 1]; ``None`` returns *guided* unchanged.
        eps: Floor on the guided statistic to avoid division by zero.
        rescale_type: ``"std"`` (default) or ``"norm"`` (per-row L2).
        rescale_dims: Axes the statistic is reduced over.  ``None`` →
            all non-batch axes (``range(1, guided.ndim)``).  Set to
            ``(-1,)`` for per-token rescaling in patchified
            ``[B, seq, D]`` space (e.g. QwenImage).
    """
    if rescale is None or rescale == 0:
        return guided

    assert_same(shape=True, dtype=True, device=True, guided=guided, cond=cond)

    dims = tuple(range(1, guided.ndim)) if rescale_dims is None else tuple(rescale_dims)
    if rescale_type == "std" and all(guided.size(d) == 1 for d in dims):
        raise ValueError(
            f"std rescale needs more than one element per statistic, got axes {dims} of shape {tuple(guided.shape)}"
        )

    if rescale_type == "std":
        cond_stat = cond.std(dim=dims, keepdim=True, correction=0)
        guided_stat = guided.std(dim=dims, keepdim=True, correction=0).clamp(min=eps)
    elif rescale_type == "norm":
        cond_stat = torch.linalg.vector_norm(cond, dim=dims, keepdim=True)
        guided_stat = torch.linalg.vector_norm(guided, dim=dims, keepdim=True).clamp(min=eps)
    else:
        raise ValueError(f"rescale_type must be 'std' or 'norm', got {rescale_type!r}")
    factor = rescale * cond_stat / guided_stat + (1 - rescale)
    return guided * factor


def apply_guidance(
    state: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    scale: Optional[Union[float, torch.Tensor]],
) -> torch.Tensor:
    """``state + (scale - 1) * (pos - neg)`` on a single modality.

    Returns *state* unchanged when *scale* is ``None``.  A ``Tensor[B]``
    *scale* is broadcast to the prediction shape via ``expand_like``
    and cast to the prediction dtype, in which the arithmetic runs.
    """
    if scale is None:
        return state

    assert_same(shape=True, dtype=True, device=True, state=state, pos=pos, neg=neg)

    if isinstance(scale, torch.Tensor):
        scale = expand_like(scale.to(state.dtype), state)
    return state + (scale - 1) * (pos - neg)


# ───────────────── Schemas ───────────────────────────────────────────────────
#
# ``attrs`` fields hold ``Any`` for OmegaConf compatibility; the
# concrete shapes are:
#   * scale / boundary  : None | float | List[float] | Dict[str, ...same...]
#                         (one dict layer max — per-modality only)
#   * skip_layers       : None | List[int] | List[Optional[List[int]]]
#                         (always-on flat list, or piecewise list paired
#                         with a boundary)


@attrs.define
class GuidanceTerm:
    """Base of every guidance term: a ``scale`` scheduled piecewise over ``boundary`` (see ``resolve_piecewise``).

    A term is off when its ``scale`` is ``None``.
    """

    scale: Any = None
    boundary: Any = None

    def is_ever_active(self) -> bool:
        """Whether ``GuidanceContext.scale`` keeps the term at some time under ``skip_inactive``: some piece of its
        schedule, for some modality, ``is_active``."""

        def any_piece_active(scale, boundary) -> bool:
            pieces = [scale] if scale is None or boundary is None else scale
            return any(is_active(piece) for piece in pieces)

        active = per_modality(any_piece_active, self.scale, self.boundary)
        return any(active.values()) if isinstance(active, dict) else active


@attrs.define
class CFGTerm(GuidanceTerm):
    """Classifier-free guidance: ``state + (scale - 1) * (cond - uncond)``.

    The uncond pass uses ``neg_condition`` (and any ``skip_layers``)
    and is cached under ``precomputed["neg"]`` — the ``MixGuidanceTerm``
    reuses that cache entry when both are enabled.
    """

    skip_layers: Any = None
    skip_layers_boundary: Any = None


@attrs.define
class MixGuidanceTerm(GuidanceTerm):
    """Additive guidance from a (possibly distinct) cond reference.

    Adds ``(scale - 1) * (cache['pos_mix'] - x_uncond)``, where
    ``x_uncond`` is the cfg-uncond reference (the ``"neg"`` branch shared
    with ``CFGTerm``, computed on demand unless ``precomputed``).

    ``cache['pos_mix']`` is a fresh cond forward
    (``net(x_t, t, condition=condition, **fwd_kwargs)``, the arguments
    of the ``"pos"`` pass) unless ``precomputed`` supplies it. It differs
    from ``"pos"`` when the caller precomputes either one: mean_flow's
    3-way teacher mixture passes the analytical conditional target
    (``flow_matching_target`` in the network's native prediction type) as
    ``precomputed['pos']`` and keeps the network forward as ``"pos_mix"``.
    """


@attrs.define
class GuidanceConfig:
    """At most one of each term + an optional final ``rescale`` toward the std/norm of ``x_cond``.

    All slots are present (factory-defaulted); a term is "off" when its
    ``scale`` is ``None`` (``predict_with_guidance`` short-circuits).
    Same-type chaining is disallowed by construction — two CFG terms
    with scales ``s_1, s_2`` collapse to a single term with scale
    ``(s_1 + s_2 - 1)``.

    The rescale is *not* a term; it is a non-linear post-step that runs
    after every linear term has been composed (see ``cfg_rescale``):

    * ``rescale``: strength in [0, 1] (or per-modality dict); ``None`` disables.
    * ``rescale_type``: ``"std"`` (default, original CFG-rescale paper) or
      ``"norm"`` (per-row L2; used by QwenImage's per-token rescale in
      patchified ``[B, seq, D]`` space).
    * ``rescale_dims``: axes the statistic is reduced over; ``None`` →
      all non-batch axes.  Set to ``(-1,)`` for per-token rescaling.

    A network with terms or options of its own subclasses this config. A subclass with terms extends ``validate``,
    so that other networks reject them when configured, and ``pred_keys`` and ``active_pred_keys`` with their
    branches. ``condition_for_pred_key`` runs every branch but ``"neg"`` with the positive condition unless the
    subclass overrides it too.
    """

    cfg: CFGTerm = attrs.field(factory=CFGTerm)
    mix: MixGuidanceTerm = attrs.field(factory=MixGuidanceTerm)
    rescale: Any = None  # None | float | per-modality dict; multiplicative final rescaling
    rescale_type: str = "std"  # "std" | "norm"
    rescale_dims: Any = None  # None → all non-batch axes; else tuple of ints

    def pred_keys(self) -> list[str]:
        """Branch keys that the cache, ``precomputed`` and ``contexts`` may use: here ``PRED_KEYS``."""
        return list(PRED_KEYS)

    def active_pred_keys(self) -> list[str]:
        """Branch keys that ``predict_with_guidance`` computes at some time.

        Mirrors its term-by-term gating under the default ``skip_inactive``: a term counts unless its scale is
        inactive (``None`` or 1) in every piece and modality. ``"pos"`` is always present; CFG adds ``"neg"``, and
        mix adds ``"neg"`` and ``"pos_mix"`` (a fresh cond forward unless the caller overrides it via
        ``precomputed``).
        """
        keys = ["pos"]
        if self.cfg.is_ever_active():
            keys.append("neg")
        if self.mix.is_ever_active():
            keys += [key for key in ("neg", "pos_mix") if key not in keys]
        return keys

    def condition_for_pred_key(self, key: str, condition: Any, neg_condition: Any) -> Any:
        """Condition object passed to ``net`` for the branch *key*.

        Mirrors ``predict_with_guidance``: only ``"neg"`` uses *neg_condition*, and every other branch uses
        *condition*. Keys outside ``pred_keys`` raise ``ValueError``.
        """
        keys = self.pred_keys()
        if key not in keys:
            raise ValueError(f"unknown guidance branch key {key!r}; expected a key from {keys}")
        return neg_condition if key == "neg" else condition

    def validate(
        self, supported_terms: Sequence[str], supports_cfg_skip_layers: bool = False, network_name: str = "network"
    ) -> None:
        """Reject configured terms that a network does not support, before any forward and in any time window.

        A term is configured when its scale is not ``None``, even a scale of 1 that never runs a forward.
        A subclass with terms extends this, so that a network rejects the terms of another network's subclass.
        ``cfg.skip_layers`` count only with a configured CFG term, since only its negative forward receives them.
        """
        for name, term in (("cfg", self.cfg), ("mix", self.mix)):
            if name not in supported_terms and term.scale is not None:
                raise NotImplementedError(f"{network_name} does not support guidance_cfg.{name}")
        if "rescale" not in supported_terms and self.rescale is not None:
            raise NotImplementedError(f"{network_name} does not support guidance_cfg.rescale")
        if not supports_cfg_skip_layers and self.cfg.scale is not None and self.cfg.skip_layers is not None:
            raise NotImplementedError(f"{network_name} does not support guidance_cfg.cfg.skip_layers")


# ───────────────── Helpers ───────────────────────────────────────────────────


def resolve_piecewise(value, t: torch.Tensor, boundary, sequence: bool = False):
    """Generic piecewise-constant t→value lookup.

    | ``value`` shape | ``boundary`` shape | Semantics |
    |---|---|---|
    | ``None`` | (ignored) | disabled, returns ``None`` |
    | anything | ``None`` | always-on, returned as-is |
    | ``Sequence[N]`` | scalar (``N == 2``) | binary switch at that ``t`` |
    | ``Sequence[N]`` | ``Sequence[N-1]`` | piecewise over ``t`` |

    Boundaries ascend, and a time on a boundary takes the later piece:
    ``value[i]`` applies for ``boundary[i-1] <= t < boundary[i]``.
    By default every piece must be a number (e.g. a scale) and the result
    is a per-sample ``Tensor[B]``.  With ``sequence=True`` (e.g.
    ``[None, [5, 10]]`` for ``skip_layers``), every piece must be a
    sequence or ``None`` and the single piece selected by ``t`` is
    returned; this requires every batch sample to fall in the same bucket
    and raises ``ValueError`` otherwise (the typical sample-time case
    ``t.expand(B)`` always satisfies this).  Scalar value + boundary is a
    programmer error and raises late from ``len()``.
    """
    if value is None:
        return None
    if boundary is None:
        # Always-on / pass-through; works uniformly for scalars,
        # always-on Sequences (e.g. ``skip_layers=[5, 10]``), and any
        # other always-on data.
        return value
    boundary_seq = boundary if isinstance(boundary, Sequence) else [boundary]
    if len(value) != len(boundary_seq) + 1:
        raise ValueError(f"len(value) must be len(boundary)+1, got {len(value)} vs {len(boundary_seq)}")
    if sequence and not all(v is None or isinstance(v, Sequence) for v in value):
        raise ValueError(f"a sequence schedule with a boundary needs a sequence or None per piece, got {value!r}")
    boundaries = torch.tensor(boundary_seq, dtype=t.dtype, device=t.device)
    idx = torch.bucketize(t, boundaries, right=True)
    if not sequence:
        return torch.tensor(list(value), dtype=t.dtype, device=t.device)[idx]
    if idx.numel() > 1 and not (idx == idx.flatten()[0]).all():
        raise ValueError(
            "a sequence schedule requires all batch samples to fall in the same bucket; "
            f"got bucket indices {idx.flatten().tolist()[:8]}"
        )
    return value[int(idx.flatten()[0])]


def is_active(s_eff) -> bool:
    """True iff the resolved scale ``s_eff`` contributes a non-no-op perturbation.

     Each guidance term composes additively as
     ``x_pred + (s_eff - 1) * (cond - uncond)``, so ``s_eff == 1``
     everywhere is a no-op.  Returns ``False`` in that case (and for
     ``None``); ``True`` otherwise.

    ``predict_with_guidance`` uses this to skip a term's forward pass and
    ``apply_guidance`` call when the term is uniformly inactive — see its
    ``skip_inactive`` kwarg for the bypass needed for FSDP-based training.
    """
    if s_eff is None:
        return False
    if isinstance(s_eff, DICT_TYPES):
        return any(is_active(v) for v in s_eff.values())
    if torch.is_tensor(s_eff):
        return (s_eff != 1.0).any()
    return s_eff != 1.0


# ───────────────── Forward context and composable guidance blocks ────────────


PRED_KEYS = ("pos", "neg", "pos_mix")


@attrs.define
class GuidanceContext:
    """Per-call forward cache shared by the guidance blocks.

    This context does not decide which terms to compose. The generic entry
    point or a network-owned implementation chooses the blocks and their order.
    """

    net: FastGenNetwork
    x_t: TensorOrDict
    t: torch.Tensor
    guidance_cfg: GuidanceConfig
    condition: Any
    neg_condition: Any
    cache: Dict[str, TensorOrDict]
    contexts: Dict[str, Dict[str, Any]]
    fwd_kwargs: Dict[str, Any]
    skip_inactive: bool
    fwd_pred_type: Optional[str]
    output_dtype: OutputDtype
    apply_output_conditioning: bool
    dtype: DtypeOrDict
    batched: bool
    cat_fn: Optional[Callable]

    def predict(self, key: str, **kwargs: Any) -> TensorOrDict:
        """Return the cached native branch for ``key`` in the working dtype, running its forward first if needed.

        This is the cached tensor itself, so recipes must not modify it in place.
        """
        if key in ("pos", "neg") and key not in self.cache:
            self._predict_cfg_pair()
        if key not in self.cache:
            self.cache[key] = self.net(self.x_t, self.t, **self.forward_kwargs(key, **kwargs))
            self._validate()
        return self.cache[key]

    def _predict_cfg_pair(self) -> None:
        """Batch an uncached CFG pair on demand, after the recipe's validation."""
        cfg = self.guidance_cfg.cfg
        if (
            not self.batched
            or self.scale(cfg) is None
            or any(key in self.cache or self.contexts.get(key) for key in ("pos", "neg"))
        ):
            return
        if not isinstance(self.x_t, torch.Tensor):
            raise TypeError("batched guidance requires a Tensor x_t; use batched=False for multimodal data")
        if cfg.skip_layers is not None:
            raise ValueError("batched guidance cannot apply cfg.skip_layers; use batched=False")
        try:
            c_in = (self.cat_fn or _default_cat)(self.neg_condition, self.condition)
        except Exception as e:
            raise ValueError(
                f"CFG cat_fn failed (guidance_cfg.cfg.scale={cfg.scale!r}); "
                "pass matching condition and neg_condition structures."
            ) from e
        prediction = self.net(
            torch.cat([self.x_t, self.x_t], dim=0),
            torch.cat([self.t, self.t], dim=0),
            **self.forward_kwargs("pos", condition=c_in),
        )
        negative, positive = prediction.chunk(2)
        self.cache.update(pos=positive, neg=negative)
        self._validate()

    def forward_kwargs(self, key: str, **kwargs: Any) -> Dict[str, Any]:
        """Resolve branch overrides for a forward returning the native, unconstrained prediction in the working dtype."""
        kwargs = {**self.fwd_kwargs, **kwargs, **self.contexts.get(key, {})}
        return {**kwargs, "apply_output_conditioning": False, "output_dtype": self.dtype}

    def finalize(self, prediction: TensorOrDict) -> TensorOrDict:
        """Transform the composed native prediction once, which rounds it to the result dtype, and validate it.

        The transform reads the condition and forward context of the positive branch.
        """
        self._assert_dtype_and_device(self.dtype, prediction=prediction)
        pos_kwargs = {**self.fwd_kwargs, "condition": self.condition, **self.contexts.get("pos", {})}
        prediction = self.net.transform_output(
            prediction,
            self.x_t,
            self.t,
            fwd_pred_type=self.fwd_pred_type,
            output_dtype=self.output_dtype,
            apply_output_conditioning=self.apply_output_conditioning,
            **pos_kwargs,
        )
        self._validate(prediction)
        return prediction

    def _validate(self, prediction: Optional[TensorOrDict] = None) -> None:
        """Check branch keys, that the cache has the working dtype, and that ``prediction`` has the result dtype."""
        keys = self.guidance_cfg.pred_keys()
        unknown = sorted((self.cache.keys() | self.contexts.keys()) - set(keys))
        if unknown:
            raise ValueError(f"unknown guidance branch keys {unknown}; expected keys from {keys}")
        self._assert_dtype_and_device(self.dtype, **self.cache)
        if prediction is None:
            return
        if self.output_dtype == "same":
            assert_same(dtype=True, device=True, x_t=self.x_t, prediction=prediction)
        else:
            dtype = self.dtype if self.output_dtype is None else self.output_dtype
            self._assert_dtype_and_device(dtype, prediction=prediction)

    def _assert_dtype_and_device(self, dtype: DtypeOrDict, **structures: TensorOrDict) -> None:
        """Check that *structures* have the modalities of a mapping *dtype*, and that each modality has its dtype and
        the device of ``x_t``."""
        device = get_device(self.x_t)
        for name, structure in structures.items():
            validate_output_dtype(dtype, structure)
            per_modality(lambda x, d: assert_same(dtype=d, device=device, **{name: x}), structure, dtype)

    def scale(self, term) -> Any:
        """Resolve a term's schedule; None means no forward/composition is needed."""
        if term.scale is None:
            return None
        scale = per_modality(resolve_piecewise, term.scale, self.t, term.boundary)
        return None if self.skip_inactive and not is_active(scale) else scale


def _default_cat(neg: Any, pos: Any) -> Any:
    """Concatenate corresponding tensor leaves of nested conditions, negative first."""

    def cat_leaf(p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        assert_same(shape=True, dtype=True, device=True, pos=p, neg=n)
        return torch.cat([n, p], dim=0)

    return tmap(cat_leaf, pos, neg)


def get_guidance_context(
    net: FastGenNetwork,
    x_t: TensorOrDict,
    t: torch.Tensor,
    condition: Any,
    guidance_cfg: Optional[GuidanceConfig] = None,
    neg_condition: Any = None,
    precomputed: Optional[Dict[str, TensorOrDict]] = None,
    contexts: Optional[Dict[str, Dict[str, Any]]] = None,
    skip_inactive: bool = True,
    fwd_pred_type: Optional[str] = None,
    output_dtype: OutputDtype = "same",
    apply_output_conditioning: bool = True,
    batched: bool = False,
    cat_fn: Optional[Callable[[Any, Any], Any]] = None,
    **fwd_kwargs: Any,
) -> GuidanceContext:
    """Prepare a lazy forward context for the network *net*.

    Only missing predictions are computed. A cached CFG branch or branch-specific
    context uses sequential forwards; neither is overwritten to enable batching.
    Other guidance terms use this same cache and are evaluated on demand.
    No forwards run until a recipe calls ``ctx.predict``. Schedules use the
    caller's ``t`` and cached predictions use the network's native prediction space.
    """
    if not isinstance(net.base_net if isinstance(net, SharedBaseTeacher) else net, FastGenNetwork):
        raise TypeError(f"guidance needs a FastGenNetwork or its SharedBaseTeacher, got {type(net).__name__}")
    for kwargs in (contexts or {}).values():
        owned = sorted({"fwd_pred_type", "output_dtype", "apply_output_conditioning"} & kwargs.keys())
        if owned:
            raise ValueError(
                f"guidance owns {owned}: branch forwards return unconstrained native predictions in the working "
                "dtype; set these for the guided result instead"
            )
    validate_output_dtype(output_dtype, x_t)
    dtype = per_modality(high_precision_dtype, x_t, None if output_dtype == "same" else output_dtype)

    ctx = GuidanceContext(
        net=net,
        x_t=x_t,
        t=t,
        guidance_cfg=guidance_cfg if guidance_cfg is not None else GuidanceConfig(),
        condition=condition,
        neg_condition=neg_condition,
        cache={},
        contexts=contexts or {},
        fwd_kwargs=fwd_kwargs,
        skip_inactive=skip_inactive,
        fwd_pred_type=fwd_pred_type,
        output_dtype=output_dtype,
        apply_output_conditioning=apply_output_conditioning,
        dtype=dtype,
        batched=batched,
        cat_fn=cat_fn,
    )
    if precomputed:
        ctx.cache.update({key: cast_like(branch, x_t, dtype) for key, branch in precomputed.items()})
    ctx._validate()
    return ctx


def apply_cfg(ctx: GuidanceContext, prediction: TensorOrDict) -> TensorOrDict:
    """Add CFG using the cached conditional and negative predictions."""
    scale = ctx.scale(ctx.guidance_cfg.cfg)
    if scale is None:
        return prediction
    term = ctx.guidance_cfg.cfg
    skip_layers = resolve_piecewise(term.skip_layers, ctx.t, term.skip_layers_boundary, sequence=True)
    return per_modality(
        apply_guidance,
        prediction,
        ctx.predict("pos", condition=ctx.condition),
        ctx.predict("neg", condition=ctx.neg_condition, skip_layers=skip_layers),
        scale,
    )


def apply_mix(ctx: GuidanceContext, prediction: TensorOrDict) -> TensorOrDict:
    """Add a distinct conditional-reference term, reusing the CFG negative cache."""
    scale = ctx.scale(ctx.guidance_cfg.mix)
    if scale is None:
        return prediction
    return per_modality(
        apply_guidance,
        prediction,
        ctx.predict("pos_mix", condition=ctx.condition),
        ctx.predict("neg", condition=ctx.neg_condition),
        scale,
    )


def apply_rescale(
    ctx: GuidanceContext,
    prediction: TensorOrDict,
    condition_prediction: Optional[TensorOrDict] = None,
) -> TensorOrDict:
    """Apply final rescaling; callers may provide predictions in a custom packed space."""
    g = ctx.guidance_cfg
    if g.rescale is None:
        return prediction
    if condition_prediction is None:
        condition_prediction = ctx.predict("pos", condition=ctx.condition)
    return per_modality(
        cfg_rescale,
        prediction,
        condition_prediction,
        g.rescale,
        rescale_type=g.rescale_type,
        rescale_dims=g.rescale_dims,
    )


# ───────────────── Guided-prediction entry point ──────────────────────


def predict_with_guidance(
    net: FastGenNetwork,
    x_t: TensorOrDict,
    t: torch.Tensor,
    condition: Any,
    guidance_cfg: Optional[GuidanceConfig] = None,
    neg_condition: Any = None,
    precomputed: Optional[Dict[str, TensorOrDict]] = None,
    contexts: Optional[Dict[str, Dict[str, Any]]] = None,
    return_cache: bool = False,
    skip_inactive: bool = True,
    fwd_pred_type: Optional[str] = None,
    output_dtype: OutputDtype = "same",
    apply_output_conditioning: bool = True,
    batched: bool = False,
    cat_fn: Optional[Callable[[Any, Any], Any]] = None,
    **fwd_kwargs: Any,
) -> TensorOrDict:
    """Return the guided prediction of the network *net*.

    The network's optional ``predict_with_guidance(ctx)`` recipe obtains the
    native branches through ``ctx.predict`` and returns the composed, rescaled
    native prediction in the working dtype ``ctx.dtype``, a mapping of one per
    modality for a multimodal state. It must not transform it itself.
    Otherwise, the standard recipe applies CFG, mix, then rescale, and rejects
    every other configured term. The dispatcher transforms either recipe's result once and handles caches.

    Args:
        net: The network, or its SharedBaseTeacher; any other provider raises TypeError.
        x_t: Noisy input tensor or multimodal mapping.
        t: Per-sample timestep tensor, used to resolve guidance schedules.
        guidance_cfg: Full guidance recipe. None means an empty config.
        condition / neg_condition: Conditional and negative forward inputs.
        precomputed: Cached predictions keyed by a branch key of guidance_cfg.pred_keys():
            native and unconstrained, as a forward with the default fwd_pred_type and
            apply_output_conditioning=False returns them; its output_dtype=None avoids
            rounding. They are cast to the working dtype of the computed branches. Other keys
            raise ValueError. Cache hits bypass forwards and their modifiers; the caller
            owns consistency of inputs, timesteps, conditions and prediction space, which
            guidance cannot check. Transformed predictions cannot in general be unblended,
            especially at strength one. "pos_mix" defaults to a fresh conditional forward.
        contexts: Extra per-branch forward kwargs, keyed like precomputed, overriding
            common and term kwargs. Cache hits take precedence. Branch-specific
            contexts prevent CFG batching. They cannot set fwd_pred_type, output_dtype
            or apply_output_conditioning, which describe the result.
        return_cache: Return (transformed_prediction, cache). Cache entries are native
            and untransformed, in the working dtype, so reuse never applies a
            conditioning blend twice.
        skip_inactive: Skip terms whose resolved scale is uniformly 1. Set False
            during distributed training to keep forwards uniform across ranks.
        fwd_pred_type: Prediction type of the result; None keeps the network's native type. The
            final transform converts to it once.
        output_dtype: Dtype of the result, or a mapping of one per modality: "same" is each
            modality's state dtype, and None keeps the working dtype, per modality the
            high_precision_dtype of its state and requested dtype. The final transform rounds
            once to it. Other values, or a mapping without a torch.dtype for each modality of
            x_t, raise ValueError before any forward.
        apply_output_conditioning: Apply the network's output constraint in the final
            transform. False still converts to fwd_pred_type; caches and the result dtype
            are unaffected.
        batched: Evaluate the conditional and CFG-negative pair in one forward
            when neither is cached and neither has its own context. Requires a
            tensor input and no CFG skip_layers; other terms remain sequential.
        cat_fn: Combine (neg_condition, condition) for a batched forward. The
            default concatenates corresponding tensor leaves of nested structures.
        **fwd_kwargs: Common kwargs forwarded to every network call and the final
            transform.
    """
    ctx = get_guidance_context(
        net,
        x_t,
        t,
        guidance_cfg=guidance_cfg,
        condition=condition,
        neg_condition=neg_condition,
        precomputed=precomputed,
        contexts=contexts,
        skip_inactive=skip_inactive,
        fwd_pred_type=fwd_pred_type,
        output_dtype=output_dtype,
        apply_output_conditioning=apply_output_conditioning,
        batched=batched,
        cat_fn=cat_fn,
        **fwd_kwargs,
    )
    if net.predict_with_guidance is not None:
        prediction = net.predict_with_guidance(ctx)
    else:
        ctx.guidance_cfg.validate(
            supported_terms=("cfg", "mix", "rescale"), supports_cfg_skip_layers=True, network_name=type(net).__name__
        )
        prediction = ctx.predict("pos", condition=condition)
        prediction = apply_cfg(ctx, prediction)
        prediction = apply_mix(ctx, prediction)
        prediction = apply_rescale(ctx, prediction)
    prediction = ctx.finalize(prediction)
    return (prediction, ctx.cache) if return_cache else prediction
