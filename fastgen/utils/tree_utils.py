# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytree / multimodal utilities.

Two layers, used distinctly:

1. **Per-modality data operations** — operate on ``TensorOrDict`` (a
   single ``Tensor`` for unimodal, or ``Dict[str, Tensor]`` for
   multimodal).  Built on top of the ``per_modality`` dispatcher and do
   *not* recurse into nested lists / tuples / namedtuples.  Parameter
   name convention: ``data``.  Members:

   * Dispatcher: ``per_modality``.
   * Data ops: ``get_shape``, ``noise_from_shape``, ``randn_like``,
     ``cast_like``, ``validate_output_dtype``, ``index``, ``slice_dim``, ``to_dict``.

2. **Generic tree traversal** — walk arbitrary nested PyTrees (Tensor,
   dict / ModuleDict, list, tuple, namedtuple).  Parameter name
   convention: ``structure`` (or ``*structures``).  Members: ``tmap``,
   ``to``, ``detach``, ``expand_first_dim``, ``first_leaf``,
   ``get_device``, ``get_dtype``, ``get_batch_size``, ``assert_same``.

Most callers in ``fastgen`` should reach for the per-modality layer;
``tmap`` and friends are reserved for the few cases where the data is
genuinely a nested PyTree (e.g. preserving namedtuple containers across
``to`` / ``detach``).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch

from fastgen.utils.basic_utils import cast
from fastgen.utils.tensor import resolve_device

TensorOrDict = Union[torch.Tensor, Dict[str, torch.Tensor]]
DtypeOrDict = Union[torch.dtype, Dict[str, torch.dtype]]
# The ``output_dtype`` of conversions, forwards and guidance: ``"same"`` rounds the result to each modality's
# input dtype, ``None`` returns it uncast, in the precision it was computed in, and a dtype, or a mapping of one
# per modality, casts to it.
OutputDtype = Union[DtypeOrDict, Literal["same"], None]
# ``torch.nn.ModuleDict`` implements dict-like methods but is NOT a ``Mapping``
# subclass, so we keep an explicit tuple. ``CompositeNoiseSchedule.schedules``
# is a ``ModuleDict`` and must be traversable.
DICT_TYPES = (Mapping, torch.nn.ModuleDict)


# ---------------------------------------------------------------------------
# Per-modality dispatcher (canonical multi-arg primitive)
# ---------------------------------------------------------------------------


def per_modality(fn: Callable, primary: Any, *args: Any, **kwargs: Any) -> Any:
    """Apply *fn* per-modality, broadcasting non-dict args/kwargs as-is.

    *primary* drives the output structure:

    * **Non-dict primary** (``Tensor``, scalar, ``None``, ``Callable``, …):
      short-circuits to ``fn(primary, *args, **kwargs)``.
    * **Dict / ``ModuleDict`` primary**: iterates ``for k in primary`` and
      calls ``fn(primary[k], *idx_args, **idx_kwargs)`` per key.  For each
      element of ``args`` / ``kwargs``: if it is a dict (``Mapping`` or
      ``ModuleDict``) it is **indexed per-key** (``v[k]``); otherwise it is
      **broadcast as-is** to every per-modality call.

    Per-key indexing applies symmetrically to positional and keyword args.
    Mismatched dict keys raise ``KeyError`` at the per-modality call.

    ``per_modality`` is the canonical primitive for multimodal-or-unimodal
    data (``TensorOrDict``); it does *not* recurse into nested lists /
    tuples / namedtuples — that is the job of ``tmap``.
    """
    if not isinstance(primary, DICT_TYPES):
        return fn(primary, *args, **kwargs)

    def _idx(v: Any, k: Any) -> Any:
        return v[k] if isinstance(v, DICT_TYPES) else v

    return {
        k: fn(
            primary[k],
            *(_idx(a, k) for a in args),
            **{kk: _idx(v, k) for kk, v in kwargs.items()},
        )
        for k in primary
    }


# ---------------------------------------------------------------------------
# Per-modality data operations
#
# These all consume ``TensorOrDict`` (or a ``TensorOrDict``-shaped spec
# in the case of ``noise_from_shape``) and dispatch via ``per_modality``.
# They do NOT recurse into nested lists / tuples / namedtuples — that is
# the job of ``tmap`` (next section).
# ---------------------------------------------------------------------------


def get_shape(data: TensorOrDict) -> Union[torch.Size, Dict[str, torch.Size]]:
    """Structural shape of *data*.

    * ``Tensor`` → ``torch.Size``.
    * ``Dict[str, Tensor]`` → ``{key: torch.Size}`` (per-modality).

    Useful for logging / debugging.  Multimodal call sites use ``get_shape``
    rather than ``data.shape`` so the per-modality structure survives.
    """
    return per_modality(lambda x: x.shape, data)


def noise_from_shape(
    input_shape: Union[List[int], Dict[str, List[int]]],
    batch_size: int,
    device: Union[torch.device, str],
    dtype: torch.dtype,
) -> TensorOrDict:
    """Create Gaussian noise matching an *input_shape* specification.

    *input_shape* is either a flat list ``[C, ...]`` (unimodal) or a dict
    of such lists ``{"video": [C, ...], "audio": [C, ...]}`` (multimodal).
    The output mirrors ``input_shape``'s structure and prepends *batch_size*
    to every tensor's leading dimension.
    """
    return per_modality(
        lambda s: torch.randn(batch_size, *s, device=device, dtype=dtype),
        input_shape,
    )


def randn_like(data: TensorOrDict, dtype: Optional[torch.dtype] = None) -> TensorOrDict:
    """Sample i.i.d. Gaussian noise matching *data*'s shapes and device.

    Returned structure mirrors *data*: ``Tensor → Tensor``,
    ``Dict[str, Tensor] → Dict[str, Tensor]``.
    """
    return per_modality(lambda x: torch.randn_like(x, dtype=dtype), data)


def validate_output_dtype(dtype: Any, data: TensorOrDict) -> None:
    """Raise ``ValueError`` unless *dtype* is an ``OutputDtype`` for *data*.

    A mapping must give a ``torch.dtype`` for exactly the modalities of a mapping *data*.
    """
    if dtype is None or isinstance(dtype, torch.dtype) or dtype == "same":
        return
    if not isinstance(dtype, Mapping):
        raise ValueError(f"output dtype must be a torch.dtype, a mapping of them, 'same' or None, got {dtype!r}")
    if not isinstance(data, Mapping):
        raise ValueError(f"a per-modality dtype {dtype!r} needs multimodal data, got a {type(data).__name__}")
    if dtype.keys() != data.keys() or not all(isinstance(d, torch.dtype) for d in dtype.values()):
        raise ValueError(f"a per-modality dtype needs a torch.dtype per modality of {sorted(data)}, got {dtype!r}")


def cast_like(data: TensorOrDict, reference: TensorOrDict, dtype: OutputDtype = "same") -> TensorOrDict:
    """Cast *data* to *dtype*: ``"same"`` is the dtype of each modality of *reference*, and ``None`` keeps *data*.

    Only the dtype changes. A ``Tensor`` *reference* applies to every modality, a mapping *dtype* gives each
    modality's dtype, and integer and boolean tensors are cast too, as in ``to``. ``validate_output_dtype``
    rejects any other *dtype*.
    """
    validate_output_dtype(dtype, data)
    if dtype is None:
        return data
    if dtype == "same":
        dtype = per_modality(lambda ref: ref.dtype, reference)
    return per_modality(lambda x, d: cast(x, dtype=d), data, dtype)


def index(data: TensorOrDict, idx: Any) -> TensorOrDict:
    """Index every tensor in *data*, i.e. ``data[idx]`` per modality.

    *idx* is anything ``Tensor.__getitem__`` accepts -- an ``int``, an index tensor, or a tuple for
    several axes (``index(x, (batch_idx, k))`` is ``x[batch_idx, k]``).
    """
    return per_modality(lambda x, i: x[i], data, idx)


def slice_dim(
    data: TensorOrDict,
    *args: Optional[int],
    dim: int = 0,
) -> TensorOrDict:
    """Slice along axis *dim*, broadcasting over modalities.

    Positional ``*args`` are forwarded verbatim to Python's builtin
    ``slice``, mirroring its 1-/2-/3-arg overloads:

    * ``slice_dim(data, stop)`` → first ``stop`` entries along ``dim``.
    * ``slice_dim(data, start, stop)`` → ``[start:stop]`` along ``dim``.
    * ``slice_dim(data, start, stop, step)`` → ``[start:stop:step]`` along ``dim``.

    Any of *start* / *stop* / *step* may be ``None`` for open-ended slicing.
    *dim* defaults to ``0`` (the batch dim) and supports negative indexing
    (e.g. ``dim=-1`` for the last axis).
    """
    s = slice(*args)
    if dim >= 0:
        idx: tuple = (slice(None),) * dim + (s,)
    else:
        idx = (Ellipsis, s) + (slice(None),) * (-dim - 1)
    return index(data, idx)


def to_dict(data: TensorOrDict, suffix: str = "") -> Dict[str, torch.Tensor]:
    """Flatten *data* into ``{key + suffix: tensor, ...}``.

    * ``Tensor`` → ``{suffix: tensor}``.
    * ``Dict[str, Tensor]`` → ``{f"{k}_{suffix}": v, ...}``.

    Used to format per-modality losses for downstream logging.
    """
    if isinstance(data, torch.Tensor):
        return {suffix: data}
    return {f"{k}_{suffix}": v for k, v in data.items()}


# ---------------------------------------------------------------------------
# Generic tree traversal
#
# These walk arbitrary nested PyTrees (Tensor, dict / ModuleDict, list,
# tuple, namedtuple).  Use these only when the caller genuinely needs
# nested-sequence support; for ``TensorOrDict`` data, prefer the
# per-modality helpers above.
# ---------------------------------------------------------------------------


def first_leaf(structure: Any) -> torch.Tensor:
    """Return the first ``torch.Tensor`` leaf reachable from *structure*.

    Walks arbitrary nested PyTrees (Tensor, dict / ``ModuleDict``, list,
    tuple, namedtuple) using the same container layout as ``tmap``.
    Iteration order is dict insertion order then sequence order.  Used by
    the leaf introspection helpers (``get_device``, ``get_dtype``,
    ``get_batch_size``) — they all assume leaf-uniform attributes.
    Raises ``ValueError`` if no tensor leaf is reachable (including on an
    empty dict, an empty sequence, or a non-container scalar).
    """

    def _find(node: Any) -> Optional[torch.Tensor]:
        if isinstance(node, torch.Tensor):
            return node
        if isinstance(node, DICT_TYPES):
            children: Any = node.values()
        elif isinstance(node, (tuple, list)):
            children = node
        else:
            return None
        for child in children:
            found = _find(child)
            if found is not None:
                return found
        return None

    leaf = _find(structure)
    if leaf is None:
        raise ValueError(f"first_leaf: no Tensor leaf found in structure of type {type(structure).__name__!r}")
    return leaf


def get_device(structure: Any) -> torch.device:
    """Device of *structure*'s first tensor leaf (assumed leaf-uniform)."""
    return first_leaf(structure).device


def get_dtype(structure: Any) -> torch.dtype:
    """Dtype of *structure*'s first tensor leaf (assumed leaf-uniform)."""
    return first_leaf(structure).dtype


def get_batch_size(structure: Any) -> int:
    """Batch size (dim 0) of *structure*'s first tensor leaf (assumed leaf-uniform)."""
    return first_leaf(structure).shape[0]


def tmap(fn: Callable, *structures: Any) -> Any:
    """Map *fn* over corresponding leaves of one or more parallel structures.

    Recursion is driven by the first argument's container layout (Tensor,
    dict / ``ModuleDict``, list, tuple, namedtuple).  At each leaf of the
    first argument:

    * If the leaf is a ``torch.Tensor``, *fn* is called with the
      corresponding leaves from **all** ``structures`` (passed positionally).
    * Otherwise the leaf is returned as-is.

    **Strict structural alignment**: every container position must match
    across all ``structures`` (same dict keys, same sequence length); a
    non-container at a container position triggers an assertion.  This is
    intentionally stricter than ``per_modality`` (which broadcasts
    non-dict args at dict positions); callers wanting per-modality
    broadcast semantics over ``TensorOrDict`` should use ``per_modality``
    instead.
    """
    first = structures[0]
    if isinstance(first, torch.Tensor):
        return fn(*structures)
    elif isinstance(first, DICT_TYPES):
        keys = set(first.keys())
        assert all(
            isinstance(s, DICT_TYPES) and set(s.keys()) == keys for s in structures[1:]
        ), f"tmap: dict key mismatch — expected {sorted(first.keys())}"
        return {k: tmap(fn, *(s[k] for s in structures)) for k in first}
    elif isinstance(first, (list, tuple)):
        n = len(first)
        assert all(
            isinstance(s, (list, tuple)) and len(s) == n for s in structures[1:]
        ), f"tmap: sequence length mismatch — expected length {n}"
        items = [tmap(fn, *(s[i] for s in structures)) for i in range(n)]
        if hasattr(type(first), "_fields"):  # NamedTuple
            return type(first)(*items)
        return type(first)(items)
    else:
        return first


def assert_same(
    shape: bool | tuple[int, ...] | None = None,
    dtype: bool | torch.dtype | None = None,
    device: bool | torch.device | None = None,
    **structures: Any,
) -> None:
    """Assert selected metadata for tensor leaves.

    Each metadata argument may skip the property (``None`` / ``False``),
    compare corresponding leaves (``True``), or require a concrete value;
    skipping all of them checks nothing, as ``dtype=module.precision_model``
    does for a module without one. A bare ``cuda`` device target means the
    current CUDA device.

    A required value is checked on every tensor leaf of each structure on its
    own, so those structures need no common layout. Comparisons walk the trees
    together with ``tmap``, so they must share one container layout (dict /
    ``ModuleDict``, list, tuple, namedtuple) and the first structure drives the
    walk: positions where it holds a tensor are compared across all structures,
    positions where it holds ``None`` or metadata are skipped.  ``None``
    structures and ``None`` leaves are omitted from both checks, and each leaf
    position is compared independently, so modalities may use different dtypes
    or shapes.  Keyword names label the failure message.
    """
    attributes = {
        name: value
        for name, value in (("shape", shape), ("dtype", dtype), ("device", device))
        if value is not None and value is not False
    }
    if not attributes:
        return
    structures = {name: value for name, value in structures.items() if value is not None}
    if not structures:
        return
    compared = [attribute for attribute, target in attributes.items() if target is True]
    required = {attribute: target for attribute, target in attributes.items() if target is not True}
    if "device" in required:
        required["device"] = resolve_device(required["device"])

    def require(name: str, leaf: torch.Tensor) -> torch.Tensor:
        for attribute, expected in required.items():
            actual = getattr(leaf, attribute)
            assert actual == expected, f"{name}.{attribute} ({actual}) must be {expected}"
        return leaf

    def compare(*leaves: Any) -> torch.Tensor:
        tensors = [(name, leaf) for name, leaf in zip(structures, leaves) if leaf is not None]
        for name, leaf in tensors:
            if not isinstance(leaf, torch.Tensor):
                raise TypeError(f"assert_same expected {name} to be a Tensor or None, got {type(leaf).__name__}")
        (ref_name, ref), *peers = tensors
        for attribute in compared:
            expected = getattr(ref, attribute)
            for name, tensor in peers:
                actual = getattr(tensor, attribute)
                assert (
                    actual == expected
                ), f"{name}.{attribute} ({actual}) must match {ref_name}.{attribute} ({expected})"
        return leaves[0]

    if required:
        for name, structure in structures.items():
            if not isinstance(structure, (torch.Tensor, *DICT_TYPES, list, tuple)):
                raise TypeError(
                    f"assert_same expected {name} to be a Tensor, a container, or None, got {type(structure).__name__}"
                )
            tmap(partial(require, name), structure)
    if compared:
        tmap(compare, *structures.values())


def to(
    structure: Any,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    cast_ints: bool = True,
) -> Any:
    """Recursively cast every tensor in *structure* to *device* / *dtype*.

    Walks any container supported by ``tmap``. If both arguments are ``None``,
    returns *structure* unchanged. Pass ``cast_ints=False`` to keep the dtype of
    integer and boolean tensors, such as class indices and attention masks; a
    float mask would be added to the attention scores rather than mask them.
    """
    if device is None and dtype is None:
        return structure
    is_cpu = (isinstance(device, str) and device == "cpu") or (
        isinstance(device, torch.device) and device.type == "cpu"
    )
    return tmap(lambda x: cast(x, device, dtype, non_blocking=not is_cpu, cast_ints=cast_ints), structure)


def detach(structure: Any) -> Any:
    """Recursively ``detach`` every tensor in *structure*; non-tensors pass through."""
    return tmap(torch.Tensor.detach, structure)


def expand_first_dim(structure: Any, batch_size: int) -> Any:
    """Expand the first dimension of every tensor in *structure* to ``batch_size``.

    Walks any container supported by ``tmap``; uses ``Tensor.expand`` so
    no memory is copied.  Non-tensor leaves (e.g. ``None``) pass through.
    """
    return tmap(lambda x: x.expand(batch_size, *x.shape[1:]), structure)
