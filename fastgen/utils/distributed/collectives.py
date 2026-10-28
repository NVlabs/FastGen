# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
from typing import Any, Callable, Mapping, Optional

import torch
import torch.distributed as dist

import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed import get_rank, world_size


def synchronize():
    """Synchronize all ranks when FastGen's multi-rank process group is active (see ``world_size``)."""
    num_ranks = world_size()
    if num_ranks <= 1:
        return
    logger.debug(f"Synchronizing all devices with world size {num_ranks}")
    dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
    logger.debug(f"Synchronized all devices with world size {num_ranks}")


def clean_up():
    if dist.is_available() and dist.is_initialized():
        try:
            logger.info("Distributed clean up.")
            dist.destroy_process_group()
        except ValueError as error:
            logger.error(f"Error destroying default process group: {error}")


def _all_reduce_scalar(
    local_value: bool | int,
    device: torch.device | str,
    dtype: torch.dtype,
    op,
) -> int:
    """All-reduce one boolean or integer scalar and return its integer value."""
    result = torch.tensor([local_value], dtype=dtype, device=device)
    if world_size() > 1:
        dist.all_reduce(result, op=op)
    return int(result.item())


def sync_all(local_all: bool, device: torch.device | str) -> bool:
    """Synchronize local all across distributed ranks."""
    return bool(
        _all_reduce_scalar(
            local_all,
            device=device,
            dtype=torch.uint8,
            op=dist.ReduceOp.MIN,
        )
    )


def sync_any(local_any: bool, device: torch.device | str) -> bool:
    """Synchronize local any across distributed ranks."""
    return bool(
        _all_reduce_scalar(
            local_any,
            device=device,
            dtype=torch.uint8,
            op=dist.ReduceOp.MAX,
        )
    )


def sync_max(local_value: int, device: torch.device | str) -> int:
    """Synchronize the maximum of a local scalar across distributed ranks."""
    return _all_reduce_scalar(
        local_value,
        device=device,
        dtype=torch.int64,
        op=dist.ReduceOp.MAX,
    )


def sync_min(local_value: int, device: torch.device | str) -> int:
    """Synchronize the minimum of a local scalar across distributed ranks."""
    return _all_reduce_scalar(
        local_value,
        device=device,
        dtype=torch.int64,
        op=dist.ReduceOp.MIN,
    )


def broadcast_choice(
    high: int,
    num_samples: int = 1,
    device: torch.device | str = "cpu",
    src: int = 0,
) -> int | list[int]:
    """Sample and broadcast one or more integers in ``[0, high)``.

    Every rank performs the local draw before the broadcast so this helper consumes the same
    amount of random state everywhere, while only the source rank's values determine the shared
    result. One sample returns an integer; multiple samples return a list.
    """
    assert high > 0 and num_samples > 0 and src >= 0, (high, num_samples, src)
    choices = torch.randint(0, high, (num_samples,), device=device, dtype=torch.long)
    if world_size() > 1:
        dist.broadcast(choices, src=src)
    return int(choices.item()) if num_samples == 1 else choices.tolist()


def broadcast_object(
    value,
    src: int = 0,
    device: Optional[torch.device] = None,
):
    """Broadcast one small, trusted, picklable object and return it on every rank.

    ``device`` selects the tensor transport used internally by PyTorch's object collective.
    FastGen's mixed process group uses CPU/Gloo by default; callers may pass a CUDA device to route
    the serialized bytes through NCCL instead.
    """
    if world_size() <= 1:
        return value

    values = [value]
    dist.broadcast_object_list(values, src=src, device=device)
    return values[0]


def validate_object_across_ranks(value, operation: str) -> None:
    """Require equal small, serializable metadata on all ranks, using a fixed comparison source.

    Build ``value`` inside a preceding error guard if that work can fail locally. Values must
    support ordinary Python equality (no tensors). This check precedes collectives whose source,
    payload selection, or participation depends on the metadata being compared. Mapping mismatches
    report only the differing top-level fields.
    """
    if world_size() <= 1:
        return
    source = broadcast_object(value, src=0, device=torch.device("cuda", torch.cuda.current_device()))
    with raise_across_ranks(operation):
        if value != source:
            detail = ""
            if isinstance(value, Mapping) and isinstance(source, Mapping):
                fields = [
                    key
                    for key in value.keys() | source.keys()
                    if key not in value or key not in source or value[key] != source[key]
                ]
                detail = f"; differing fields: {sorted(fields, key=repr)[:8]}"
            raise RuntimeError(f"{operation} mismatch with rank 0{detail}")


def warn_on_rng_state_drift(operation: str, cpu: bool) -> None:
    """Warn when a rank's generators differ from rank 0's or show draws on a GPU other than its own.

    The current device's CUDA generator and, with ``cpu``, the CPU generator are compared with rank 0's by
    full state (seed and offset), without changing them; Python's and NumPy's generators are not. CUDA
    ordinals are rank-local and each rank draws only on its own device, so the generators of a rank's other
    visible devices are checked locally instead: seeding resets a CUDA generator's offset to zero and every
    draw advances it. Every rank gathers the same results and decides alike.
    """
    if world_size() <= 1:
        return
    current = torch.cuda.current_device()
    device = torch.device("cuda", current)
    generators = torch.cuda.default_generators
    compared = {"CUDA": generators[current].get_state()}
    if cpu:
        compared["CPU"] = torch.default_generator.get_state()

    mismatches = []
    for name, state in compared.items():
        local_state = state.to(device=device)
        gathered = [torch.empty_like(local_state) for _ in range(world_size())]
        dist.all_gather(gathered, local_state)
        ranks = [rank for rank, other in enumerate(gathered) if not torch.equal(other, gathered[0])]
        if ranks:
            mismatches.append(f"{name} on ranks {ranks[:8]}")
    if mismatches:
        logger.warning(
            f"{operation}: RNG states differ from rank 0 ({', '.join(mismatches)}); some ranks consumed randomness "
            "the others did not, e.g. a reset_parameters() hook during meta materialization."
        )

    drawn = [
        f"cuda:{index}" for index, generator in enumerate(generators) if index != current and generator.get_offset()
    ]
    drawn_by_rank = [None] * world_size()
    dist.all_gather_object(drawn_by_rank, drawn)
    foreign = [f"rank {rank}: {', '.join(names)}" for rank, names in enumerate(drawn_by_rank) if names]
    if foreign:
        logger.warning(
            f"{operation}: ranks drew random values on GPUs other than their own ({'; '.join(foreign[:8])}), "
            "e.g. by creating tensors on a hard-coded device."
        )


@contextlib.contextmanager
def raise_across_ranks(operation: str = "collective operation"):
    """Run the block on every rank and raise the first failing rank's error on every rank.

    Wrap work that can fail on a strict subset of ranks (rank-0-only I/O, comparisons against a
    source rank, meta-rank materialization) and that the other ranks would otherwise wait for in
    their next collective. Checks that fail on every rank alike do not need it. Every rank must
    enter the block when distributed execution is active, even with an empty body, so the failure
    poll lines up; leaving the block on success therefore also acts as a barrier. Outside
    distributed execution the block runs unchanged. The failing rank chains its original exception
    as cause. Only ``Exception`` subclasses are propagated; a ``BaseException`` such as
    ``KeyboardInterrupt`` leaves the peers to the launcher's teardown.

    Group consecutive local work in one block, ending before the next collective. This guard
    cannot rescue peers already waiting inside a collective in its body, and an extra barrier
    immediately after a successful block is redundant. Put structural postconditions inside the
    guard too, e.g. ``with raise_across_ranks(operation), snapshot.validate_after(): ...``. One
    phase-wide check is sufficient when no intervening collective, write, or baseline refresh
    requires the validated state sooner.
    """
    if world_size() <= 1:
        yield
        return

    caught = None
    try:
        yield
    except Exception as error:
        caught = error

    device = torch.device("cuda", torch.cuda.current_device())
    rank, num_ranks = get_rank(), world_size()
    offending_rank = sync_min(rank if caught is not None else num_ranks, device=device)
    if offending_rank == num_ranks:
        return

    failure = None if caught is None else f"{type(caught).__name__}: {caught}"
    failure = broadcast_object(failure, src=offending_rank, device=device)
    raise RuntimeError(f"{operation}: rank {offending_rank} failed: {failure}") from caught


def broadcast_rank0_result(fn: Callable[[], Any], operation: str, device: Optional[torch.device] = None) -> Any:
    """Run ``fn`` on rank 0 only and return its result on every rank.

    Every rank must call this together, so never call it from code that only rank 0 runs. A failure
    in ``fn`` is raised on every rank, as by ``raise_across_ranks``, before the result broadcast. The
    result must suit ``broadcast_object``, whose transport ``device`` selects.
    """
    result = None
    with raise_across_ranks(operation):
        if get_rank() == 0:
            result = fn()
    return broadcast_object(result, device=device)


def synced_iter(iterable, device: torch.device | str):
    """Advance together across ranks, raising on unequal exhaustion or iterator errors.

    Each fetch costs one reduction while every rank has an item. Iteration ends normally
    when every rank is exhausted. Outside distributed execution, yield from the iterable
    unchanged. Call from rank processes, never from DataLoader workers, and advance on
    every rank in the same collective order. Apply limits to the returned iterator to
    avoid fetching unused items while keeping iterator initialization inside the guard.
    """
    if world_size() <= 1:
        yield from iterable
        return
    with raise_across_ranks("Initializing synchronized iterator"):
        iterator = iter(iterable)
    exhausted = object()
    while True:
        item, caught = exhausted, None
        try:
            item = next(iterator, exhausted)
        except Exception as error:
            caught = error
        if not sync_all(item is not exhausted, device=device):
            break
        yield item

    # A failed fetch also has no item, so report failures before comparing exhaustion.
    with raise_across_ranks("Advancing synchronized iterator"):
        if caught is not None:
            raise caught
    if sync_any(item is not exhausted, device=device):
        raise ValueError(
            "Iterables yielded different numbers of items across ranks. "
            "Use equal-length iterables or a limit every rank can supply."
        )
