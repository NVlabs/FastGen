# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-rank validation and synchronization of module state.

Rank 0 is the source of truth for meta-initialized state and must hold the complete real state,
including nonpersistent buffers, even in single-process execution: a checkpoint cannot supply
nonpersistent buffers or missing keys. Every rank calls these collectives in the same order.
"""

from typing import Collection, Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate

# DistributedDataParallel's transport for module states (FSDP1's ``sync_module_states`` uses it too):
# one collective per bucket, flattening and ``copy_`` back into the passed tensors done in C++. Private
# API, stable across releases; the bucket size is DDP's ``broadcast_bucket_size``.
from torch.distributed.utils import _sync_params_and_buffers

import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed.collectives import raise_across_ranks, sync_any, validate_object_across_ranks
from fastgen.utils.distributed import get_rank, world_size
from fastgen.utils.module import (
    TensorKind,
    has_meta_state,
    local_tensor,
    materialize_meta_module,
    module_state_tensors,
    module_topology_signature,
    reject_unsupported_state_aliases,
    registration_signature,
    tensor_registrations,
)


_BROADCAST_BUCKET_BYTES = 250 * 1024 * 1024


def validate_module_sync_source(
    module: torch.nn.Module,
    src: int,
    operation: str = "state synchronization",
) -> None:
    """Locally require real state on the source rank before materialization and synchronization.

    Call inside the surrounding local phase's ``raise_across_ranks`` block, before its next
    collective. This check performs no communication and also applies to single-process execution:
    allocating meta state alone cannot recover its values, including nonpersistent buffers.
    """
    if get_rank() == src and has_meta_state(module):
        raise RuntimeError(
            f"{operation}: Rank {src} must hold the complete real state for {module.__class__.__name__} "
            "to synchronize meta-initialized state, including in single-process execution."
        )


def _state_descriptor(
    module: torch.nn.Module, operation: str, nonpersistent_buffers: bool = True, include_device: bool = True
) -> dict:
    """Describe module topology and registered-state metadata for comparison with rank 0's."""
    reject_unsupported_state_aliases(module, operation, allow_buffer_aliases=True)
    registrations = {
        key: registration
        for key, registration in tensor_registrations(module).items()
        if key[0] == "parameter" or registration.persistent or nonpersistent_buffers
    }
    return {
        "topology": module_topology_signature(module),
        "registrations": registration_signature(registrations, include_device=include_device),
    }


def validate_module_state_across_ranks(
    module: torch.nn.Module,
    operation: str = "state synchronization",
    nonpersistent_buffers: bool = True,
    include_device: bool = True,
) -> None:
    """Collectively require rank 0's module topology and registered-state metadata on every rank.

    Parameter-buffer aliases are rejected and selected metadata must match. Copy-only checkpoint
    loads can exclude nonpersistent caches and device types, since neither is transported by that
    operation. All module paths, types, and shared-owner groups must agree, including stateless
    descendants. Buffer-only aliases are supported because synchronization copies values in place.
    This function checks metadata only and does not alter module state.
    """
    if world_size() <= 1:
        return
    with raise_across_ranks(operation):
        descriptor = _state_descriptor(module, operation, nonpersistent_buffers, include_device)
    validate_object_across_ranks(descriptor, operation=operation)


def validate_module_values_across_ranks(
    module: torch.nn.Module,
    parameters: bool = False,
    persistent_buffers: bool = True,
    nonpersistent_buffers: bool = True,
    operation: str = "value synchronization check",
) -> None:
    """Collectively require rank 0's registered-state metadata and selected value fingerprints.

    DTensor parameters are compared only when every placement is ``Replicate``; sharded and partial
    local values legitimately differ and are skipped. ``torch.hash_tensor`` computes a
    device-native 64-bit XOR fingerprint without cloning tensor contents. This is a consistency
    canary, not proof of equality: permutations and some repeated-value changes can collide. Every
    rank must call this with the same selection.
    """
    if world_size() <= 1:
        return
    with raise_across_ranks(operation):
        descriptor = _state_descriptor(module, operation)
        selected = [
            (name, local_tensor(tensor))
            for kind, name, tensor in module_state_tensors(
                module,
                parameters=parameters,
                persistent_buffers=persistent_buffers,
                nonpersistent_buffers=nonpersistent_buffers,
            )
            if not (
                kind == "parameter"
                and isinstance(tensor, DTensor)
                and not all(isinstance(placement, Replicate) for placement in tensor.placements)
            )
        ]
        fingerprints = _tensor_fingerprints([tensor for _name, tensor in selected])
        descriptor["fingerprints"] = dict(zip((name for name, _tensor in selected), fingerprints))
    validate_object_across_ranks(descriptor, operation=operation)


@torch.no_grad()
def _tensor_fingerprints(tensors: list[torch.Tensor]) -> tuple[int, ...]:
    """Return device-native XOR fingerprints, transferring only the scalar results to the host."""
    fingerprints: list[int | None] = [None] * len(tensors)
    by_device: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
    for index, tensor in enumerate(tensors):
        # Empty tensors have no values to fingerprint; the descriptor compares their shapes.
        # Avoid backend-specific empty-reduction behavior.
        if tensor.numel() == 0:
            fingerprints[index] = 0
            continue
        fingerprint = torch.hash_tensor(tensor.detach())
        by_device.setdefault(fingerprint.device, []).append((index, fingerprint))

    # One transfer per device avoids synchronizing once per CUDA tensor. Stacking only the uint64
    # scalar results never copies parameter or buffer contents.
    for device_fingerprints in by_device.values():
        indices, values = zip(*device_fingerprints)
        stacked = torch.stack(values)
        if stacked.device.type != "cpu":
            stacked = stacked.cpu()
        for index, value in zip(indices, stacked.tolist()):
            fingerprints[index] = int(value)
    return tuple(fingerprint for fingerprint in fingerprints if fingerprint is not None)


def _validate_broadcast_selection(
    module: torch.nn.Module,
    selected: list[tuple[TensorKind, str, torch.Tensor]],
    src: int,
    operation: str,
) -> dict:
    """Validate the local selection and describe the broadcast for rank agreement; perform no communication."""
    descriptor = _state_descriptor(module, operation)
    nonreplicated = [
        name
        for _kind, name, tensor in selected
        if isinstance(tensor, DTensor) and not all(isinstance(p, Replicate) for p in tensor.placements)
    ]
    if nonreplicated:
        raise RuntimeError(f"{operation}: cannot broadcast non-replicated DTensors: {nonreplicated[:8]}.")
    # Coalesced transport matches by position. It rejects an invalid source once every rank agrees on it.
    descriptor.update(src=src, payload=[(kind, name) for kind, name, _tensor in selected])
    return descriptor


@torch.no_grad()
def broadcast_module(
    module: torch.nn.Module,
    src: int = 0,
    parameters: bool = True,
    persistent_buffers: bool = True,
    nonpersistent_buffers: bool = True,
    names: Optional[Collection[str]] = None,
    operation: str = "broadcast_module",
) -> None:
    """Broadcast selected registered state of ``module`` from ``src`` to all ranks.

    The three flags select by kind; ``names`` additionally restricts the selection to those qualified
    names (any alias of a shared tensor qualifies). Registrations, ``src``, and the ordered selection
    are compared with rank 0's first, then the selected tensors are broadcast in canonical order
    through coalesced buckets, exactly like ``DistributedDataParallel`` synchronizes its module
    states. Every rank must call this with the same selection. Selected DTensors must be fully
    replicated; broadcasting local shards would overwrite each rank's distinct portion of the global
    tensor.
    """
    if world_size() <= 1:
        return

    with raise_across_ranks(operation):
        selected = list(
            module_state_tensors(
                module,
                parameters=parameters,
                persistent_buffers=persistent_buffers,
                nonpersistent_buffers=nonpersistent_buffers,
                names=names,
            )
        )
        descriptor = _validate_broadcast_selection(module, selected, src, operation)
        payloads = [local_tensor(tensor).detach() for _kind, _name, tensor in selected]
    validate_object_across_ranks(descriptor, operation=operation)
    if payloads:
        _sync_params_and_buffers(dist.group.WORLD, payloads, _BROADCAST_BUCKET_BYTES, src)


@torch.no_grad()
def move_and_sync_module(
    module: torch.nn.Module,
    device: torch.device | str,
) -> None:
    """Move a module to ``device``, synchronizing materialized meta state.

    Meta ranks materialize locally, then every rank receives rank zero's parameters and buffers.
    Callers that synchronize every parameter and buffer themselves can use ``materialize_meta_module``
    directly instead. Loading a checkpoint alone does not cover missing keys or nonpersistent buffers.
    Storage dtypes are kept, so the broadcast requires every rank to already hold rank zero's dtypes;
    cast beforehand, as ``prepare_module`` does through ``_set_precision``. With meta state on any rank,
    every rank must match rank zero's topology and registered metadata before anything materializes.
    Real-only moves are rank-local: nothing is compared or broadcast, so ranks may hold different modules.
    """
    is_meta = has_meta_state(module)
    any_meta = sync_any(is_meta, device=device)

    # Callers validate placement safety first, as ``prepare_module`` does.
    if any_meta:
        with raise_across_ranks("move_and_sync_module"):
            validate_module_sync_source(module, src=0, operation="move_and_sync_module")
        validate_module_state_across_ranks(module, operation="move_and_sync_module topology", include_device=False)
    with raise_across_ranks("move_and_sync_module placement"):
        # Materialization already places every tensor, real ones included, on ``device``.
        if is_meta:
            materialize_meta_module(module, device)
        else:
            module.to(device=device)

    if any_meta:
        logger.debug(f"{module.__class__.__name__}: broadcasting from rank 0 to all ranks")
        broadcast_module(module, src=0)
