# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from functools import partial
from contextlib import contextmanager
import time
from typing import TYPE_CHECKING, Optional, Callable

import torch
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
)

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


from fastgen.networks.network import FastGenNetwork
from fastgen.utils.distributed import world_size, synchronize, is_rank0
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


def apply_fsdp_checkpointing(module: torch.nn.Module, check_fn: Optional[Callable[[torch.nn.Module], bool]] = None):
    """
    Apply FSDP checkpointing to a module.

    Follows overall approach outlined in https://pytorch.org/blog/maximizing-training/, without adaptive selection.

    Args:
        module: The module to wrap with activation checkpointing.
        check_fn: A function (Module -> bool) that takes a module and returns True if it should be wrapped.
            If None, wraps transformer blocks (Block in class name) only.
    """
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    if check_fn is None:
        # Default: only checkpoint block-level modules  (e.g. WanTransformerBlock in Wan network)
        # This is the correct granularity for FSDP + checkpointing
        def check_fn(submodule):
            # Check for common transformer block class names
            class_name = submodule.__class__.__name__
            return "Block" in class_name

        logger.info("Using default check_fn: checkpointing modules with 'Block' in class name")

    apply_activation_checkpointing(module, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)


def model_to_fsdp(
    model: FastGenModel,
    min_num_params: int = 10_000_000,
    apply_cpu_offload: bool = False,
    sync_module_states: bool = False,
    sharding_group_size: Optional[int] = None,
):
    """Convert model to FSDP.

    Args:
        model: The FastGen model to wrap with FSDP.
        min_num_params: Minimum number of parameters for a module to be wrapped as a separate FSDP unit.
            Default is 10M, which wraps large models quite finely.
            For a 14B model, each transformer block has ~300-400M params, so a larger value may
            be preferable (e.g., 100M-500M).
        apply_cpu_offload: Whether to offload parameters to CPU.
        sync_module_states: If True, broadcast module states from rank 0 to all other ranks.
            Enable this when using memory-efficient loading where only rank 0 loads weights.

    Memory-Efficient Loading with Meta Device:
        To use memory-efficient loading for large models (14B+):
        1. Set `fsdp_meta_init=True` in trainer config
        2. Network classes should check `is_rank0()` and:
           - Rank 0: load full weights
           - Other ranks: use `with torch.device("meta"): load_model()` for ZERO memory allocation
        4. FSDP will materialize meta tensors and broadcast weights from rank 0

        This reduces initialization from N*model_size RAM to 1*model_size RAM,
        and avoids N parallel disk reads (major I/O contention).
    """
    fsdp_dict = model.fsdp_dict

    total_world_size = world_size()
    if sharding_group_size is None:
        # Full sharding over the world
        logger.info(f"Fully sharding model with {total_world_size} ranks...")
        device_mesh = init_device_mesh("cuda", (total_world_size,))
    else:
        if total_world_size % sharding_group_size != 0:
            raise ValueError(
                f"World size {total_world_size} must be divisible by shard group size {sharding_group_size}"
            )
        replica_group_size = total_world_size // sharding_group_size
        logger.info(f"Sharding model with {replica_group_size} sharding groups of size {sharding_group_size}...")
        device_mesh = init_device_mesh(
            "cuda", (replica_group_size, sharding_group_size), mesh_dim_names=("replicate", "shard")
        )

    # Mixed precision policy for FSDP2
    mp_policy = MixedPrecisionPolicy(
        param_dtype=model.precision,
        reduce_dtype=model.precision_fsdp,
        # We avoid casting all inputs so we can control t precision etc.
        output_dtype=None,
        cast_forward_inputs=False,
    )

    offload_policy = CPUOffloadPolicy() if apply_cpu_offload else None

    for k, v in fsdp_dict.items():
        if k.startswith("ema"):
            logger.warning("EMA network stored in fsdp_dict will be skipped during FSDP2 wrap.")
            continue

        num_params = sum(p.numel() for p in v.parameters()) / 1e9
        logger.info(f"Starting FSDP2 wrap for '{k}' ({num_params:.2f}B params)...")
        t0 = time.time()

        # Step 1: Extract full state dict from rank 0 BEFORE sharding
        # Rank 0 has real weights, other ranks have meta tensors
        if sync_module_states:
            if is_rank0():
                # Rank 0 has the real weights - extract them
                # TODO: This can be slow. Instead, we should shard before loading pre-trained weights.
                state_dict = copy.deepcopy(v.state_dict())
                logger.info(f"  [Rank 0] Extracted state dict with {len(state_dict)} tensors")
            else:
                state_dict = None
        else:
            state_dict = None

        # Step 2: Apply fully_shard to create DTensor structure
        if isinstance(v, FastGenNetwork):
            # We use the network's custom fully_shard method
            v.fully_shard(
                mesh=device_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
            )
        else:
            # Fall back to size-based auto-wrap policy
            modules_to_shard = _get_submodules_to_shard(v, min_num_params)
            logger.info(f"  Sharding {len(modules_to_shard)} submodules")

            for submodule in modules_to_shard:
                fully_shard(
                    submodule,
                    mesh=device_mesh,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                    reshard_after_forward=True,
                )

            fully_shard(
                v,
                mesh=device_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=True,
            )
        logger.info("Completed sharding")

        # Step 3: Move parameters to correct device and optionally broadcast state dict
        # With CPU offloading, use CPU as target device so FSDP can manage GPU placement
        # Without CPU offloading, use CUDA directly
        target_device = "cpu" if apply_cpu_offload else torch.cuda.current_device()

        if sync_module_states:
            logger.info("Syncing module states from rank 0 to all ranks")
            options = StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                cpu_offload=apply_cpu_offload,
            )

            logger.debug(f"Moving all ranks to target device: {target_device}")
            v.to_empty(device=target_device)
            if hasattr(v, "reset_parameters"):
                # We need this to reinitialize non-persistent buffers like RoPE freqs_cos/freqs_sin
                # These aren't stored in the state dict, so we need to reinitialize them after to_empty()
                v.reset_parameters()
            else:
                logger.warning(
                    f"Network {v.__class__.__name__} does not implement the reset_parameters method. "
                    "This may cause unexpected behavior with FSDP2, like non-persistent buffers not "
                    "being initialized correctly."
                )
            synchronize()
            logger.debug("Moved all ranks to target devices")

            logger.debug("Broadcasting the state dict to all ranks")

            set_model_state_dict(v, model_state_dict=state_dict, options=options)
            torch.cuda.empty_cache()
        else:
            v.to(device=target_device)
        synchronize()
        logger.info(f"FSDP2 wrapped {k} in {time.time() - t0:.1f}s")

    return model


def _get_submodules_to_shard(module: torch.nn.Module, min_num_params: int) -> list[torch.nn.Module]:
    """Get list of submodules that should be sharded based on parameter count."""
    modules_to_shard = []

    def _count_params(m: torch.nn.Module) -> int:
        return sum(p.numel() for p in m.parameters(recurse=False))

    def _recurse(m: torch.nn.Module) -> None:
        for child in m.children():
            _recurse(child)
        own_params = _count_params(m)
        if own_params >= min_num_params:
            modules_to_shard.append(m)

    _recurse(module)
    return modules_to_shard


@contextmanager
def fsdp_sync_grad(model: FastGenModel, enabled: bool):
    """
    Context manager to enable/disable gradient synchronization for FSDP2 modules.

    This mirrors DDP's no_sync behavior for gradient accumulation: for non-last
    microbatches, set_requires_gradient_sync(False) to skip communication.
    """
    fsdp_modules = [
        m
        for m in model.fsdp_dict.values()
        if hasattr(m, "set_requires_gradient_sync") and any(p.requires_grad for p in m.parameters())
    ]
    if fsdp_modules:
        for module in fsdp_modules:
            module.set_requires_gradient_sync(enabled, recurse=True)
            if hasattr(module, "set_is_last_backward"):
                module.set_is_last_backward(enabled)
    try:
        yield
    finally:
        if fsdp_modules and not enabled:
            for module in fsdp_modules:
                module.set_requires_gradient_sync(True, recurse=True)
                if hasattr(module, "set_is_last_backward"):
                    module.set_is_last_backward(True)
