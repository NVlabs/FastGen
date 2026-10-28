# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 sharding of FastGen module roots through ``FastGenModule.prepare_module``.

Custom ``fully_shard`` hooks must make the same collective calls on every rank. When any rank holds
meta state, rank 0's state is exported before wrapping and loaded into the shards afterwards. That
path rejects shared submodules with parameters or persistent buffers, because PyTorch's broadcast
loader omits their duplicate paths. Shared modules without such state, tied parameters in distinct
modules, and preparation with real state on every rank are unaffected.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import functools
from functools import partial
import copy
import time
from typing import TYPE_CHECKING, Iterable, Optional

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


from fastgen.networks.network import FastGenModule
from fastgen.utils.distributed.collectives import (
    raise_across_ranks,
    sync_any,
    validate_object_across_ranks,
)
from fastgen.utils.distributed.module import (
    broadcast_module,
    validate_module_state_across_ranks,
    validate_module_sync_source,
)
from fastgen.utils.distributed import is_rank0, world_size
import fastgen.utils.logging_utils as logger
from fastgen.utils.module import (
    ModuleStateSnapshot,
    has_meta_state,
    materialize_meta_module,
    validate_state_dict_export,
    validate_module_state_preserved,
)

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


@torch.no_grad()
def _move_registered_buffers_(module: torch.nn.Module, device: torch.device | str) -> None:
    """Place validated buffers for FSDP CPU offload without moving parameter shards.

    Preparation must already have rejected unsupported aliases. Each physical owner is visited
    once; raw slot updates preserve persistence and ``None`` registrations. As with the rest of
    preparation, a failed conversion may leave partially moved state and the module must be discarded.
    """
    for owner in module.modules():
        for name, tensor in owner._buffers.items():
            if tensor is not None:
                owner._buffers[name] = tensor.to(device=device)


def reshard_fsdp_modules(modules: Iterable[torch.nn.Module]) -> None:
    """Release full FSDP2 parameters below registered roots; ``reshard`` is per group and idempotent."""
    for root in modules:
        for module in root.modules():
            if isinstance(module, FSDPModule):
                module.reshard()


def _plan_fsdp_groups(module: torch.nn.Module, min_num_params: int) -> list[tuple[torch.nn.Module, torch.nn.Module]]:
    """Plan generic groups, inner first, as ``(module to wrap, precision owner)`` pairs.

    A nested FastGen module starts a boundary when its policy differs from its enclosing boundary.
    Within each boundary, the size threshold can select smaller groups; multi-owner parameters keep
    their owners together in the enclosing group. Ties across boundaries require an architecture-owned
    layout, even if the policies happen to match. Existing FSDP subtrees retain their installed groups.
    This entire plan is validated before wrapping replaces any parameter registrations.
    """
    # A physical module that needs a boundary on any path owns that boundary on every path.
    # In particular, a shared FastGen child may also be reached through a parent of the same precision.
    boundaries = {module}
    pending = [(module, None)]
    visited = set()
    while pending:
        current, parent_policy = pending.pop()
        if isinstance(current, FSDPModule):
            continue
        policy = (
            (current.precision_model, current.precision_opt) if isinstance(current, FastGenModule) else parent_policy
        )
        if policy != parent_policy:
            boundaries.add(current)
        if (current, policy) in visited:
            continue
        visited.add((current, policy))
        pending.extend((child, policy) for child in current.children())

    registrations: dict[int, list[tuple[str, torch.nn.Module, torch.nn.Module]]] = {}
    subtree_parameters: dict[tuple[torch.nn.Module, torch.nn.Module], set[int]] = {}
    candidates: list[tuple[torch.nn.Module, torch.nn.Module, set[int]]] = []

    def visit(current: torch.nn.Module, path: str, enclosing: torch.nn.Module) -> set[int]:
        if isinstance(current, FSDPModule):
            return set()
        boundary = current if current in boundaries else enclosing

        # A shared owner is visited once per boundary. Distinct boundaries must both register its
        # parameters below so their ownership conflict is detected before either can wrap it.
        key = (current, boundary)
        if key not in subtree_parameters:
            owned = subtree_parameters[key] = set()
            for local_name, parameter in current.named_parameters(recurse=False, remove_duplicate=False):
                name = f"{path}.{local_name}" if path else local_name
                registrations.setdefault(id(parameter), []).append((name, current, boundary))
                owned.add(id(parameter))
            for child_name, child in current.named_children():
                owned.update(visit(child, f"{path}.{child_name}" if path else child_name, boundary))

            own_num_params = sum(parameter.numel() for parameter in current.parameters(recurse=False))
            if (
                current is module
                or (current is boundary and owned)
                or (own_num_params and own_num_params >= min_num_params)
            ):
                candidates.append((current, boundary, owned))

        # Parameters managed by a separate precision boundary do not belong to its enclosing group.
        return subtree_parameters[key] if boundary is enclosing else set()

    visit(module, "", module)
    conflicts = [entries for entries in registrations.values() if len({boundary for _, _, boundary in entries}) > 1]
    if conflicts:
        names = [name for entries in conflicts for name, _, _ in entries]
        raise RuntimeError(
            f"Cannot generically FSDP-shard {type(module).__name__}: a parameter is registered under separate "
            "precision boundaries. Provide an architecture-owned fully_shard() layout that keeps its owners "
            f"in one group. First 8 conflicting registrations: {names[:8]}"
        )
    shared_parameters = {
        parameter_id for parameter_id, entries in registrations.items() if len({owner for _, owner, _ in entries}) > 1
    }
    return [
        (candidate, boundary)
        for candidate, boundary, owned in candidates
        if candidate is boundary or not shared_parameters.intersection(owned)
    ]


def _validate_fsdp_pins(module: FastGenModule) -> None:
    """Require every fixed-precision pinned parameter under ``module`` to compute in its pinned dtype.

    A pure check of the installed FSDP2 groups, run after wrapping and before any state is loaded, for
    generic and architecture-owned layouts alike. Buffers are not checked: FSDP never casts them, so their
    pinned storage already holds.
    """
    expected: dict[tuple[torch.nn.Module, str], tuple[str, torch.dtype]] = {}
    for prefix, owner in module._named_fastgen_modules():
        for local_name, dtype in owner._fixed_precision_overrides().items():
            holder_path, _, param_name = local_name.rpartition(".")
            holder = owner.get_submodule(holder_path) if holder_path else owner
            if param_name in holder._parameters:
                expected[(holder, param_name)] = (f"{prefix}.{local_name}" if prefix else local_name, dtype)
    if not expected:
        return

    actual: dict[tuple[torch.nn.Module, str], tuple[Optional[torch.dtype], torch.dtype]] = {}
    for candidate in module.modules():
        if not isinstance(candidate, FSDPModule):
            continue
        group = fully_shard.state(candidate)._fsdp_param_group
        for fsdp_param in group.fsdp_params if group is not None else ():
            info = fsdp_param._module_info
            dtypes = (group.mp_policy.param_dtype, fsdp_param.sharded_param.dtype)
            holders = zip((info.module, *info.shared_modules), (info.param_name, *info.shared_param_names))
            actual.update({holder: dtypes for holder in holders})

    violations = []
    for key, (name, pin) in expected.items():
        if key not in actual:
            violations.append(f"{name}: pinned {pin} but not managed by any FSDP group")
            continue
        param_dtype, storage_dtype = actual[key]
        compute_dtype = storage_dtype if param_dtype is None else param_dtype
        if compute_dtype != pin:
            violations.append(f"{name}: computes in {compute_dtype}, pinned {pin}")
    if violations:
        raise RuntimeError(
            f"FSDP layout of {type(module).__name__} violates fixed-precision pins: {violations[:8]}. Carve pinned "
            "state into its own group (make_fully_shard_kwargs) or provide an architecture-owned fully_shard()."
        )


def make_fully_shard_kwargs(
    fsdp_kwargs: dict,
    param_dtype: torch.dtype,
    storage_dtype: Optional[torch.dtype],
    enabled: bool,
    cast_forward_inputs: Optional[bool] = None,
    reshard_after_forward: Optional[bool] = None,
) -> Optional[dict]:
    """Return kwargs for an explicit precision/storage boundary, or ``None`` if unneeded.

    ``storage_dtype`` is the enclosing root's default storage dtype before sharding. A separate
    group is required when either that storage or the parent compute policy differs from the
    island's ``param_dtype``. Set ``cast_forward_inputs=False`` for a group whose callable owner
    still runs in the parent dtype, such as Wan's direct scale/shift tables.
    """
    mp_policy = fsdp_kwargs.get("mp_policy")
    active = (
        enabled
        and mp_policy is not None
        and (storage_dtype != param_dtype or mp_policy.param_dtype not in (None, param_dtype))
    )
    if not active:
        return None

    updated_kwargs = {
        **fsdp_kwargs,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=mp_policy.reduce_dtype,
            output_dtype=mp_policy.output_dtype,
            cast_forward_inputs=(mp_policy.cast_forward_inputs if cast_forward_inputs is None else cast_forward_inputs),
        ),
    }
    if reshard_after_forward is not None:
        updated_kwargs["reshard_after_forward"] = reshard_after_forward

    return updated_kwargs


def apply_fsdp_checkpointing(module: torch.nn.Module, block_types: type | tuple[type, ...]) -> int:
    """Wrap every ``block_types`` instance below ``module`` in a non-reentrant checkpoint wrapper.

    Follows https://pytorch.org/blog/maximizing-training/ without adaptive selection. Call this
    before ``fully_shard``-ing the blocks: the wrappers replace the blocks in their containers, so
    sharding loops that run afterwards place the FSDP groups outside the checkpoint boundary.
    Returns the number of wrapped modules.
    """
    non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    # ``apply_activation_checkpointing`` wraps children only, never ``module`` itself.
    num_blocks = sum(isinstance(submodule, block_types) for submodule in module.modules() if submodule is not module)
    apply_activation_checkpointing(
        module,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: isinstance(submodule, block_types),
    )
    type_names = ", ".join(t.__name__ for t in (block_types if isinstance(block_types, tuple) else (block_types,)))
    logger.info(
        f"Applied FSDP activation checkpointing to {num_blocks} {type_names} block(s) in {type(module).__name__}"
    )
    return num_blocks


def patch_fsdp2_mp_dtypes() -> bool:
    """
    Make FSDP2 resolve group mixed-precision dtypes even when the first forward freezes all params.

    FSDP2 derives each param group's ``_orig_dtype`` (master / optimizer dtype) and
    ``_reduce_dtype`` lazily at its first forward, from the *trainable* params only. A net
    frozen at that point (DMD2 teacher, not-yet-active student) leaves both ``None``, so once
    it trains the grad reduce skips the cast back to the master dtype (diverges/crashes when
    precision_model != precision_opt) and ignores precision_fsdp_reduce. This wraps
    ``_init_mp_dtypes`` to re-derive both from the group's uniform per-param dtype attrs,
    which are set for every param, not just the trainable ones.

    Removable once https://github.com/pytorch/pytorch/issues/186998 is fixed upstream. Torch is pinned,
    so changed internals fail loudly here rather than through a fallback.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup as cls

    if getattr(cls._init_mp_dtypes, "_mp_allparams_patch", False):
        return True

    orig = cls._init_mp_dtypes

    @functools.wraps(orig)
    def _init_mp_dtypes(self):
        orig(self)
        # _orig_dtype is None only when no params were trainable at lazy init.
        if self._orig_dtype is None and self.fsdp_params:
            orig_dtypes = {p.orig_dtype for p in self.fsdp_params}
            reduce_dtypes = {p.reduce_dtype for p in self.fsdp_params}
            if len(orig_dtypes) == 1:
                self._orig_dtype = next(iter(orig_dtypes))
            if len(reduce_dtypes) == 1:
                self._reduce_dtype = next(iter(reduce_dtypes))
            # A field is unresolved only when the group's params disagree on it. A None
            # reduce dtype is a valid, resolved value: upstream clamps a param's
            # reduce_dtype to None when it equals the compute dtype (no cast needed).
            unresolved = [
                f"{name}={sorted(str(dtype) for dtype in dtypes)}"
                for name, dtypes in (("orig", orig_dtypes), ("reduce", reduce_dtypes))
                if len(dtypes) != 1
            ]
            if self._reduce_dtype is None and len(reduce_dtypes) == 1:
                reduce_repr = "None (reduce in the compute dtype)"
            else:
                reduce_repr = str(self._reduce_dtype)
            msg = (
                "FSDP2 mp-dtype fallback applied (no trainable params at lazy init) "
                f"for group {getattr(self, '_module_fqn', '?')}: "
                f"orig={self._orig_dtype} reduce={reduce_repr}"
            )
            # A frozen group (frozen submodules, teacher, not-yet-active student) is
            # expected and resolves fully, so keep it at debug.
            if unresolved:
                logger.info(f"{msg} (unresolved: per-param dtypes disagree within the group: {'; '.join(unresolved)})")
            else:
                logger.debug(msg)

    _init_mp_dtypes._mp_allparams_patch = True
    cls._init_mp_dtypes = _init_mp_dtypes
    return True


def _load_rank0_state(
    module: torch.nn.Module,
    state_dict: Optional[dict],
    state_device: torch.device,
    cpu_offload: bool,
) -> None:
    """Materialize the sharded meta state on every rank, then load rank 0's preserved values into it.

    ``state_dict`` is rank 0's pre-shard copy and ``None`` elsewhere. Materialization runs uniformly,
    so every callable root ``reset_parameters`` hook executes on every rank; rank 0's values then
    replace every parameter and buffer, including the non-persistent buffers the state dict omits.
    """
    with raise_across_ranks("FSDP state synchronization"):
        materialize_meta_module(module, state_device)
        snapshot = ModuleStateSnapshot.capture(module, "FSDP state load")
    broadcast_module(
        module,
        parameters=False,
        persistent_buffers=False,
        nonpersistent_buffers=True,
        operation="FSDP state synchronization",
    )
    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=cpu_offload)
    logger.debug("Broadcasting the state dict to all ranks")
    # Capture failures have already reached every rank before this collective full-state load.
    with raise_across_ranks("FSDP state load"), snapshot.validate_after():
        set_model_state_dict(module, model_state_dict=state_dict, options=options)


def _export_rank0_state(module: FastGenModule) -> Optional[dict]:
    """Copy rank 0's post-cast state before ``fully_shard`` replaces parameters; ``None`` elsewhere."""
    state_dict = None
    with raise_across_ranks("FSDP source state export"):
        validate_module_sync_source(module, src=0, operation="FSDP state synchronization")
        if is_rank0():
            with validate_module_state_preserved(module, "FSDP source state export"):
                # Both exports run hooks, so validate preservation across the whole export phase;
                # hidden registered state would otherwise stay uninitialized.
                state_dict = module.state_dict()
                state_with_vars = module.state_dict(keep_vars=True)
                validate_state_dict_export(module, state_with_vars, state_dict)
                # TODO: This can be slow. Instead, shard before loading pre-trained weights.
                state_dict = copy.deepcopy(state_dict)
            logger.info(f"  [Rank 0] Extracted state dict with {len(state_dict)} tensors")
    return state_dict


def _validate_pre_wrap(module: FastGenModule, *, needs_sync: bool) -> None:
    """Require every rank to wrap the same topology, layout policy, and parameter order.

    Runs before FSDP mutates registrations and before rank 0's export, which must preserve them,
    so rejections precede the state copy. Shared owners and tied parameters may produce different
    groups despite equal shapes, and FSDP packs parameters in traversal order. Meta synchronization
    cannot load repeated module paths with persistent state. Context-wide settings were already
    compared by ``create_fsdp_context``.
    """
    validate_module_state_across_ranks(module, operation="FSDP pre-wrap topology", include_device=False)
    if world_size() <= 1:
        return
    with raise_across_ranks("FSDP pre-wrap policy"):
        modules = []
        owners = {}
        for name, child in module.named_modules(remove_duplicate=False):
            first_name = owners.setdefault(id(child), name)
            if (
                needs_sync
                and first_name != name
                and (
                    any(parameter is not None for parameter in child._parameters.values())
                    or any(
                        buffer is not None and key not in child._non_persistent_buffers_set
                        for key, buffer in child._buffers.items()
                    )
                )
            ):
                raise RuntimeError(
                    "FSDP meta-state synchronization does not support shared submodules with parameters or "
                    f"persistent buffers: {first_name!r} and {name!r} refer to the same module. "
                    "Register the module once or initialize real state on every rank."
                )
            modules.append(
                (
                    name,
                    bool(getattr(child, "gradient_checkpointing", False)),
                    (child.precision_model, child.precision_opt, child._fixed_precision_overrides())
                    if isinstance(child, FastGenModule)
                    else None,
                )
            )
        parameter_order = tuple(name for name, _ in module.named_parameters(remove_duplicate=False))
    validate_object_across_ranks((modules, parameter_order), operation="FSDP pre-wrap policy")


@dataclass(eq=False)
class FSDPContext:
    """Run-wide FSDP2 configuration with a one-shot per-root sharding operation.

    The mesh, reduction precision, and offload policy are prepared once. ``_shard_module``
    performs the meta-state capture, FSDP wrapping, fixed-precision pin validation, shard placement,
    and optional rank-0 state broadcast as one operation. The same context may be passed to every selected
    ``FastGenModule.prepare_module`` call, which owns lifecycle validation and completion markers.
    Modules do not retain this context. Phase-boundary reshards discover physical FSDP
    modules from the model's registered roots instead of retaining roots here.
    """

    mesh: torch.distributed.device_mesh.DeviceMesh
    min_num_params: int
    reduce_dtype: torch.dtype
    offload_policy: Optional[CPUOffloadPolicy]
    cpu_offload: bool

    @property
    def device(self) -> torch.device:
        """Rank-local execution device selected by the FSDP mesh."""
        device_type = self.mesh.device_type
        if device_type == "cuda":
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device(device_type)

    def _shard_module(self, module: FastGenModule) -> FastGenModule:
        """Internal sharding phase after ``prepare_module`` validates lifecycle and state."""
        # The dispatcher rejects an already-wrapped exact root but intentionally permits bottom-up
        # composition with already-sharded descendants. All ranks must enter with the same
        # logical-root state and module topology. A distributed preparation failure is terminal for
        # these live objects; retrying a rank-asymmetric partial result is unsupported.

        num_params = sum(parameter.numel() for parameter in module.parameters()) / 1e9
        logger.info(f"Starting FSDP2 preparation for {type(module).__name__} ({num_params:.2f}B params)...")
        t0 = time.time()

        # Each FastGenModule root computes in the precision established immediately before
        # this call. ``None`` intentionally keeps the module's storage dtype.
        mp_policy = MixedPrecisionPolicy(
            param_dtype=module.precision_model,
            reduce_dtype=self.reduce_dtype,
            # FastGen owns the input casts so state and timestep precision remain independent.
            output_dtype=None,
            cast_forward_inputs=False,
        )

        needs_sync = sync_any(has_meta_state(module), device=self.device)
        _validate_pre_wrap(module, needs_sync=needs_sync)
        # Rank 0 owns real weights in meta-init mode: keep its post-cast state before fully_shard
        # replaces parameters with DTensors.
        state_dict = _export_rank0_state(module) if needs_sync else None

        with raise_across_ranks("FSDP wrapping"):
            _fully_shard_module(
                module,
                min_num_params=self.min_num_params,
                mesh=self.mesh,
                mp_policy=mp_policy,
                offload_policy=self.offload_policy,
            )
            _validate_fsdp_pins(module)
        logger.info("Completed sharding")

        state_device = torch.device("cpu") if self.cpu_offload else self.device
        if needs_sync:
            _load_rank0_state(module, state_dict, state_device, cpu_offload=self.cpu_offload)
            del state_dict

        with raise_across_ranks("FSDP preparation completion"):
            if not needs_sync:
                module.to(state_device)
            if self.cpu_offload:
                _move_registered_buffers_(module, self.device)
            if needs_sync:
                torch.cuda.empty_cache()

        logger.debug(
            f"Placed {type(module).__name__} state: parameter shards on {state_device}, buffers on {self.device}"
        )

        logger.info(f"FSDP2 prepared {type(module).__name__} in {time.time() - t0:.1f}s")
        return module


def create_fsdp_context(
    reduce_dtype: torch.dtype,
    min_num_params: int = 10_000_000,
    apply_cpu_offload: bool = False,
    sharding_group_size: Optional[int] = None,
) -> FSDPContext:
    """Prepare the run-wide context used to shard individual module roots.

    Args:
        reduce_dtype: Gradient reduction dtype for every root sharded with this context, normally the
            model's resolved ``precision_fsdp_reduce``.
        min_num_params: Generic wrapping threshold for roots without their own ``fully_shard`` hook: a
            submodule becomes its own FSDP unit when the parameters it owns directly
            (``parameters(recurse=False)``) reach this count; containers whose parameters live in their
            children are never wrapped; a parameterized nested FastGen module with its own precision is
            wrapped first as its own unit. Networks with a ``fully_shard`` hook choose their own units.
        apply_cpu_offload: Whether to offload parameters to CPU.
        sharding_group_size: Number of ranks in each FSDP shard group. Defaults to the world size.

    Memory-Efficient Loading with Meta Device:
        To use memory-efficient loading for large models (14B+):
        1. Set `model.meta_init=True`
        2. Use a supported ``FastGenModule`` network class; its construction should ensure:
           - Rank 0: load full weights
           - Other ranks: use `with torch.device("meta"): load_model()` for ZERO memory allocation
        3. Pass this context to each selected module's ``prepare_module`` call. It detects meta
           parameters collectively, shards them, materializes them, and broadcasts weights from rank 0.

        This reduces initialization from N*model_size RAM to 1*model_size RAM,
        and avoids N parallel disk reads (major I/O contention). The saving covers construction-time
        loads, which read weights on rank 0 only (e.g. ``FastGenModel._load_pretrained_model``).
        Monolithic ``Checkpointer.load`` runs before FSDP wrapping and materializes a full model on
        every rank, so it does not retain this host-memory saving. Use DCP checkpoints to load directly
        into prepared shards, including for FSDP training resumes.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("FSDP preparation requires an initialized distributed process group.")

    with raise_across_ranks("FSDP context initialization"):
        # A first forward with frozen parameters still needs the group's reduction dtypes initialized.
        # Install the compatibility patch before any rank enters mesh construction.
        patch_fsdp2_mp_dtypes()
        total_world_size = world_size()
        mesh_shape, mesh_dim_names = (total_world_size,), None
        if sharding_group_size is not None:
            if type(sharding_group_size) is not int or sharding_group_size <= 0:
                raise ValueError(f"sharding_group_size must be a positive integer, got {sharding_group_size!r}.")
            if total_world_size % sharding_group_size != 0:
                raise ValueError(
                    f"World size {total_world_size} must be divisible by shard group size {sharding_group_size}"
                )
            mesh_shape = (total_world_size // sharding_group_size, sharding_group_size)
            mesh_dim_names = ("replicate", "shard")

    validate_object_across_ranks(
        (mesh_shape, mesh_dim_names, reduce_dtype, min_num_params, apply_cpu_offload),
        operation="FSDP context plan",
    )
    logger.info(f"FSDP2 grad-dtype mp patch enabled; creating CUDA mesh {mesh_shape}.")
    device_mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=mesh_dim_names)

    return FSDPContext(
        mesh=device_mesh,
        min_num_params=min_num_params,
        reduce_dtype=reduce_dtype,
        offload_policy=CPUOffloadPolicy() if apply_cpu_offload else None,
        cpu_offload=apply_cpu_offload,
    )


def _fully_shard_module(module: torch.nn.Module, min_num_params: int, **fsdp_kwargs) -> None:
    """Fully shard a module while honoring FastGen-module-owned boundaries."""
    if isinstance(module, FSDPModule):
        # Preparation rejects repeats before dispatch. Reaching this point means an external or
        # partial operation already wrapped this root. Already-sharded descendants remain supported.
        raise RuntimeError(f"Cannot FSDP-prepare {type(module).__name__}: the exact root is already an FSDPModule.")

    if isinstance(module, FastGenModule) and hasattr(module, "fully_shard"):
        # A FastGen module with this hook owns its module boundaries. In particular,
        # sharding some Python wrapper roots can fail for architectures with multiple inheritance.
        module.fully_shard(**fsdp_kwargs)
        return

    groups = _plan_fsdp_groups(module, min_num_params)
    for candidate, precision_owner in groups:
        kwargs = fsdp_kwargs
        if precision_owner is not module:
            kwargs = (
                make_fully_shard_kwargs(
                    fsdp_kwargs,
                    param_dtype=precision_owner.precision_model,
                    storage_dtype=module.precision_opt if isinstance(module, FastGenModule) else None,
                    enabled=True,
                )
                or fsdp_kwargs
            )
        if candidate is not module:
            kwargs = {**kwargs, "reshard_after_forward": True}
        # The root keeps PyTorch's default retention across repeated calls in one phase.
        fully_shard(candidate, **kwargs)
    logger.info(f"  Sharded {len(groups) - 1} submodules")


@contextmanager
def fsdp_sync_grad(model: FastGenModel, enabled: bool):
    """
    Context manager to enable/disable gradient synchronization for FSDP2 modules.

    This mirrors DDP's no_sync behavior for gradient accumulation: for non-last
    microbatches, set_requires_gradient_sync(False) to skip communication.
    """
    # Custom ``fully_shard`` implementations may shard an inner transformer instead of the FastGen
    # wrapper (e.g. Wan), so the FSDP2 modules are collected below each root rather than assumed to
    # be the root itself. Frozen groups are toggled too: PyTorch's own accumulation pattern sets the
    # flags on the complete tree, a group without gradients never reduces, and filtering on
    # ``requires_grad`` here would read the previous iteration's roles for methods that flip them
    # inside ``single_train_step`` (DMD2).
    fsdp_modules = [m for root in model.module_dict.values() for m in root.modules() if isinstance(m, FSDPModule)]

    def set_sync(flag: bool) -> None:
        for module in fsdp_modules:
            module.set_requires_gradient_sync(flag, recurse=False)
            module.set_is_last_backward(flag)

    set_sync(enabled)
    try:
        yield
    finally:
        if not enabled:
            set_sync(True)
