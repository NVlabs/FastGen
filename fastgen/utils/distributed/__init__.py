# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight access to distributed rank information.

Collective and module-state operations intentionally live in explicit submodules so importing
rank-aware logging does not load them or create a reverse dependency on the logging facade.
"""

import functools
import os
from typing import Callable, Optional

import torch
import torch.distributed as dist


def world_size():
    """Get the world size.

    Returns 1 outside an initialized process group and when CUDA is unavailable: FastGen's
    collectives assume the mixed NCCL/Gloo group created by ``ddp.init``, and every collective helper
    in this package (including ``synchronize`` and ``broadcast_object``) gates on this one predicate,
    while ``get_rank`` and ``is_rank0`` report the group's rank regardless.
    """
    if dist.is_initialized() and torch.cuda.is_available():
        return dist.get_world_size()
    return 1


def local_world_size() -> int:
    """Get the number of ranks running on this node."""
    return int(os.environ.get("LOCAL_WORLD_SIZE", 0)) or max(torch.cuda.device_count(), 1)


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get the rank of the worker, or zero outside an initialized process group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group)
    return 0


def is_rank0() -> bool:
    """Return whether this is rank zero."""
    return get_rank() == 0


def rank0_only(func: Callable) -> Callable:
    """Run the decorated function only on rank zero."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_rank0():
            return func(*args, **kwargs)
        return None

    return wrapper
