# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
FSDP tests for all supported FastGen networks.

These tests verify:
1. Forward pass rank consistency - all ranks produce same output
2. Backward pass with gradient checkpointing (where supported)
3. Proper FSDP sharding and weight synchronization

Each test performs forward + backward pass, which inherently tests:
- Forward pass correctness and rank consistency
- Gradient computation
- Gradient checkpointing (where supported)

Tests require at least 2 GPUs; the larger models need 4 or 8 (see each test's ``RunIf`` marker).

Usage:
    pytest tests/test_network_fsdp.py -v

    # Run specific network tests:
    pytest tests/test_network_fsdp.py::test_fsdp_edm_cifar10 -v
    pytest tests/test_network_fsdp.py::test_fsdp_wan_1_3b -v
"""

import contextlib
import copy
import json
import os
import tempfile
from functools import partial
from typing import Dict
from unittest.mock import patch
from urllib.error import HTTPError

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)

from fastgen.configs.instantiate import LazyCall as L, instantiate
from tests.utils import RunIf, run_distributed_test
from fastgen.utils.basic_utils import clear_gpu_memory
from fastgen.utils.env import set_env_vars
from fastgen.utils.tree_utils import slice_dim
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.cosmos_predict2.network import CosmosPredict2


# =============================================================================
# FSDP Utilities
# =============================================================================


def _get_meta_init_context(meta_init: bool):
    """Get context manager for FSDP meta initialization."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    use_meta = meta_init and rank != 0
    if use_meta:
        return torch.device("meta")
    return contextlib.nullcontext()


def broadcast_state_dict_and_sync(network, broadcast_state_dict):
    """Materialize meta tensors and broadcast state dict from rank 0.

    ``cpu_offload=True`` keeps the broadcast source on CPU so we don't need
    a duplicate full-precision copy of large models (e.g. LTX-2 ~13B) on
    the rank-0 GPU.
    """
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=True,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()


def check_rank_consistency(output, tolerance=1e-5):
    """Check that output is consistent across all ranks."""
    if hasattr(output, "full_tensor"):
        output_full = output.full_tensor()
    else:
        output_full = output

    rank0_output = output_full.clone().contiguous()
    dist.broadcast(rank0_output, src=0)
    diff = (output_full - rank0_output).abs().max().item()

    return diff < tolerance, diff


# =============================================================================
# Network Creation Helpers
# =============================================================================


def create_network_with_fsdp(
    config,
    device_mesh,
    meta_init: bool = True,
    apply_checkpointing: bool = False,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    sharded_dtype: torch.dtype = torch.float32,
):
    """Create a network and apply FSDP sharding.

    Args:
        config: Network configuration
        device_mesh: FSDP device mesh
        meta_init: Whether to use meta device initialization
        apply_checkpointing: Whether to enable gradient checkpointing
        param_dtype: Dtype for parameters during computation (default bfloat16).
        reduce_dtype: Dtype for gradient reduction (default float32).
        sharded_dtype: Dtype of sharded weights (default float32).

    Returns:
        The FSDP-wrapped network.
    """
    rank = dist.get_rank()

    # Clone config
    config = copy.deepcopy(config)

    # Enable/disable gradient checkpointing if the config supports it
    if hasattr(config, "disable_grad_ckpt"):
        config.disable_grad_ckpt = not apply_checkpointing

    # Use meta device context for non-rank-0 processes
    with _get_meta_init_context(meta_init):
        network = instantiate(config)

    # Ensure all ranks have finished loading before continuing
    dist.barrier()

    # Set dtype of sharded weights before FSDP wrapping
    network.to(dtype=sharded_dtype)
    for module in network.modules():
        if isinstance(module, FastGenNetwork):
            # FSDP owns compute precision here; retain the explicitly selected
            # sharded storage dtype until wrapping.
            module._record_precision(precision_model=param_dtype, precision_opt=sharded_dtype)

    # Snapshot state dict from rank 0 BEFORE FSDP sharding. ``state_dict()``
    # returns a fresh dict but its values still alias the parameter tensors,
    # so this keeps the unsharded weights alive for the broadcast without
    # paying for a full deep copy (which OOMs the runner for large
    # models like LTX-2). ``set_model_state_dict`` mutates the dict in
    # place, so the original tensors are released as soon as broadcasting
    # finishes.
    if rank == 0:
        broadcast_state_dict = network.state_dict()
    else:
        broadcast_state_dict = None

    dist.barrier()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=None,
        cast_forward_inputs=False,
    )
    network.fully_shard(mesh=device_mesh, mp_policy=mp_policy)

    # Materialize and sync
    broadcast_state_dict_and_sync(network, broadcast_state_dict)

    # Final sync before returning
    dist.barrier()

    return network


@contextlib.contextmanager
def _capture_timestep_boundary_dtypes(network):
    """Record timesteps at the base-model and reference phase boundaries."""
    captured = []
    handles = []

    positional_targets = {
        "EDMPrecond": ("model", 1),
        "EDM2Precond": ("unet", 1),
        "StableDiffusion15": ("unet", 1),
        "StableDiffusionXL": ("unet", 1),
        "CogVideoX": ("transformer", 1),
        "Wan": ("transformer", 1),
        "CausalWan": ("transformer", 1),
        "CausalVACEWan": ("transformer", 1),
        "CausalWanI2V": ("transformer", 1),
    }
    keyword_targets = {
        "Flux": ("transformer", ("timestep", "r")),
        "Flux2": ("transformer", ("timestep",)),
        "QwenImage": ("transformer", ("timestep", "r")),
        "VACEWan": ("transformer", ("timestep", "r_timestep")),
        "WanI2V": ("transformer", ("timestep", "r_timestep")),
        "WanCamI2V": ("transformer", ("timestep", "r_timestep")),
        "Cosmos3": ("transformer", ("t",)),
        "CosmosPredict2": ("transformer", ("timesteps_B_T",)),
        "LTX2": ("transformer", ("timestep", "audio_timestep", "sigma", "audio_sigma")),
        "MiniMaxH3": ("transformer", ("timestep",)),
    }
    phase_targets = {
        "EDMPrecond": ("model.map_noise",),
        "EDM2Precond": ("unet.emb_fourier",),
        "StableDiffusion15": ("unet.time_proj",),
        "StableDiffusionXL": ("unet.time_proj",),
        "CogVideoX": ("transformer.time_proj",),
        "Flux": ("transformer.time_text_embed.time_proj",),
        "Flux2": ("transformer.time_guidance_embed.time_proj",),
        "QwenImage": ("transformer.time_text_embed.time_proj",),
        "Wan": ("transformer.condition_embedder.timesteps_proj",),
        "CausalWan": ("transformer.condition_embedder.timesteps_proj",),
        "VACEWan": ("transformer.condition_embedder.timesteps_proj",),
        "CausalVACEWan": ("transformer.condition_embedder.timesteps_proj",),
        "WanI2V": ("transformer.condition_embedder.timesteps_proj",),
        "CausalWanI2V": ("transformer.condition_embedder.timesteps_proj",),
        "WanCamI2V": ("transformer.condition_embedder.timesteps_proj",),
        "Cosmos3": ("transformer.time_proj",),
        "CosmosPredict2": ("transformer.t_embedder.0",),
        "LTX2": ("transformer.time_embed.emb.time_proj", "transformer.audio_time_embed.emb.time_proj"),
        "MiniMaxH3": ("transformer.time_proj",),
    }

    def record(label, name, value):
        if torch.is_tensor(value) and value.is_floating_point():
            captured.append((label, name, value.dtype))

    for fastgen_network in network.modules():
        if not isinstance(fastgen_network, FastGenNetwork):
            continue
        label = type(fastgen_network).__name__
        if label == "DiT":
            target = fastgen_network.t_embedder

            def positional_hook(module, args, kwargs, label=label):
                del module, kwargs
                record(label, "t", args[0])

            handles.append(target.register_forward_pre_hook(positional_hook, with_kwargs=True))
        elif label in positional_targets:
            attr, index = positional_targets[label]
            target = getattr(fastgen_network, attr)

            def positional_hook(module, args, kwargs, label=label, index=index):
                del module
                record(label, "t", args[index])
                record(label, "r", kwargs.get("r_noise_labels", kwargs.get("r_timestep")))

            handles.append(target.register_forward_pre_hook(positional_hook, with_kwargs=True))
        elif label in keyword_targets:
            attr, names = keyword_targets[label]
            target = getattr(fastgen_network, attr)

            def keyword_hook(module, args, kwargs, label=label, names=names):
                del module, args
                for name in names:
                    record(label, name, kwargs.get(name))

            handles.append(target.register_forward_pre_hook(keyword_hook, with_kwargs=True))

        for path in phase_targets.get(label, ()):
            target = fastgen_network
            for part in path.split("."):
                target = target[int(part)] if part.isdigit() else getattr(target, part)
            seen = []

            def phase_hook(module, args, kwargs, label=label, path=path, seen=seen):
                del module, kwargs
                # Shared projectors may subsequently embed guidance or ``r``.
                # The first invocation is the primary diffusion timestep.
                if not seen:
                    record(label, f"phase:{path}", args[0])
                    seen.append(True)

            handles.append(target.register_forward_pre_hook(phase_hook, with_kwargs=True))

    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


# =============================================================================
# Generic FSDP Test Implementation
# =============================================================================


def _generic_fsdp_test_impl(
    world_size: int,
    config,
    generate_inputs_fn,
    apply_checkpointing: bool = True,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    sharded_dtype: torch.dtype = torch.float32,
    extra_result_fn=None,
    post_backward_result_fn=None,
) -> Dict:
    """Generic implementation for FSDP forward + backward pass test.

    Args:
        world_size: Total number of processes
        config: Network configuration
        generate_inputs_fn: Function that takes (network, device, dtype) and returns inputs dict
        apply_checkpointing: Whether to enable gradient checkpointing
        param_dtype: Dtype for parameters during computation (default bfloat16).
        reduce_dtype: Dtype for gradient reduction (default float32).
        sharded_dtype: Dtype of sharded weights (default float32).
        extra_result_fn: Optional ``(network, inputs) -> dict`` for network-specific checks that
            need the wrapped module. Called *before* the forward pass and merged into the result
            *after* the backward, so a forward hook can fill its dict in place -- which is how
            MiniMax-H3 observes the compute dtype of its float32 carve-out. Lets a network add
            assertions without a second wrap of a model too large to shard twice.
        post_backward_result_fn: Optional ``(network, inputs) -> dict`` called immediately after
            backward for checks that must inspect materialized gradients.

    Returns:
        Dict with test results
    """
    device_mesh = init_device_mesh("cuda", (world_size,))
    device = torch.device("cuda", torch.cuda.current_device())

    network = create_network_with_fsdp(
        config,
        device_mesh,
        meta_init=True,
        apply_checkpointing=apply_checkpointing,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        sharded_dtype=sharded_dtype,
    )
    network.train()

    storage_dtypes = {param.dtype for param in network.parameters() if param.is_floating_point()}
    assert storage_dtypes == {sharded_dtype}

    # Generate inputs
    # State, conditioning that joins it, losses, and scheduler times stay full precision;
    # each FastGen network boundary casts only the state to param_dtype. Generators build
    # the remaining conditioning in the dtype ``build_conditioning`` gives it,
    # ``network.precision_model`` (LTX-2's text in its ``connector_precision``).
    inputs = generate_inputs_fn(network, device, torch.float32)
    assert inputs["t"].dtype in (torch.float32, torch.float64)

    extra_result = extra_result_fn(network, inputs) if extra_result_fn is not None else {}

    # Forward pass (with gradients). Check each architecture's reference cast
    # point instead of imposing one timestep policy on every family.
    with _capture_timestep_boundary_dtypes(network) as timestep_boundaries:
        output = network(**inputs)
    assert timestep_boundaries, f"No timestep boundary was captured for {type(network).__name__}"

    fp32_time_families = {"EDMPrecond", "EDM2Precond", "CosmosPredict2", "MiniMaxH3"}
    model_time_families = {"Flux", "Flux2", "QwenImage"}

    def expected_timestep_dtype(entry):
        label, boundary, _dtype = entry
        if label in fp32_time_families:
            return torch.float32
        if boundary.startswith("phase:") and label in model_time_families:
            return param_dtype
        return inputs["t"].dtype

    wrong_timestep_dtypes = [
        (*entry, expected_timestep_dtype(entry))
        for entry in timestep_boundaries
        if entry[2] != expected_timestep_dtype(entry)
    ]
    assert not wrong_timestep_dtypes, (
        "Timestep boundary differs from the architecture reference; "
        f"got (family, boundary, actual, expected)={wrong_timestep_dtypes}"
    )

    # Handle tuple outputs
    if isinstance(output, tuple):
        output = output[0]

    if isinstance(output, dict):
        assert all(value.dtype == torch.float32 for value in output.values())
    else:
        assert output.dtype == torch.float32

    # Handle dict outputs (multimodal networks: LTX2 with composite scheduler
    # returns ``{"video": tensor, ...}``).  Sum per-modality means so every
    # modality contributes to the backward.
    if isinstance(output, dict):
        loss = sum(v.mean() for v in output.values())
    else:
        loss = output.mean()
    loss.backward()
    post_backward_result = post_backward_result_fn(network, inputs) if post_backward_result_fn is not None else {}

    # Check that gradients exist and are finite
    grad_info = {"has_grads": False, "all_finite": True, "num_params_with_grad": 0}

    for _, param in network.named_parameters():
        if param.grad is not None:
            grad_info["has_grads"] = True
            grad_info["num_params_with_grad"] += 1
            if not torch.isfinite(param.grad).all():
                grad_info["all_finite"] = False

    # Check rank consistency of output (concatenate dict outputs into a single
    # flat tensor for the cross-rank check; per-modality rank consistency is
    # equivalent to flat-cat consistency).
    with torch.no_grad():
        if isinstance(output, dict):
            output_detached = torch.cat([v.detach().flatten() for v in output.values()])
            output_shape = {k: tuple(v.shape) for k, v in output.items()}
        else:
            output_detached = output.detach()
            output_shape = tuple(output.shape)
    is_consistent, diff = check_rank_consistency(output_detached)

    result = {
        "rank_consistent": is_consistent,
        "rank_consistency_diff": diff,
        "output_shape": output_shape,
        "has_grads": grad_info["has_grads"],
        "all_grads_finite": grad_info["all_finite"],
        "num_params_with_grad": grad_info["num_params_with_grad"],
        "backward_success": grad_info["has_grads"] and grad_info["all_finite"],
        "output_dtype": ({k: v.dtype for k, v in output.items()} if isinstance(output, dict) else output.dtype),
        **extra_result,
        **post_backward_result,
    }

    # Cleanup
    del network
    torch.cuda.empty_cache()

    # Final barrier to ensure all ranks complete before returning
    dist.barrier()

    return result


# =============================================================================
# Input Generation Functions for Each Network Type
# =============================================================================


def generate_edm_cifar10_inputs(network, device, dtype):
    """Generate inputs for EDM CIFAR10."""
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    labels = torch.zeros(batch_size, 10, dtype=dtype, device=device)
    labels[:, 0] = 1.0
    return {"x_t": x, "t": t, "condition": labels}


def generate_edm_imagenet64_inputs(network, device, dtype):
    """Generate inputs for EDM ImageNet64."""
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    labels = torch.zeros(batch_size, 1000, dtype=dtype, device=device)
    labels[:, 0] = 1.0
    return {"x_t": x, "t": t, "condition": labels}


def generate_edm2_inputs(network, device, dtype):
    """Generate inputs for EDM2."""
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    labels = torch.zeros(batch_size, 1000, dtype=dtype, device=device)
    labels[:, 0] = 1.0
    return {"x_t": x, "t": t, "condition": labels}


def generate_dit_inputs(network, device, dtype):
    """Generate inputs for DiT."""
    batch_size = 2
    # DiT operates on latent space
    x = torch.randn(batch_size, 4, 32, 32, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    labels = torch.zeros(batch_size, 1000, dtype=dtype, device=device)
    labels[:, 0] = 1.0
    return {"x_t": x, "t": t, "condition": labels}


def generate_sd15_inputs(network, device, dtype):
    """Generate inputs for Stable Diffusion 1.5."""
    batch_size = 2
    x = torch.randn(batch_size, 4, 8, 8, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    # Use random dummy embeddings instead of text encoder (CLIP ViT-L/14: seq_len=77, hidden_dim=768)
    embeddings = torch.randn(batch_size, 77, 768, device=device, dtype=network.precision_model)
    attention_mask = torch.ones(batch_size, 77, device=device, dtype=torch.bool)
    return {"x_t": x, "t": t, "condition": (embeddings, attention_mask)}


def generate_sdxl_inputs(network, device, dtype):
    """Generate inputs for Stable Diffusion XL."""
    batch_size = 1
    x = torch.randn(batch_size, 4, 8, 8, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    text_embeddings = torch.randn(batch_size, 77, 2048, device=device, dtype=network.precision_model)
    pooled_embeddings = torch.randn(batch_size, 1280, device=device, dtype=network.precision_model)
    return {"x_t": x, "t": t, "condition": (text_embeddings, pooled_embeddings)}


def generate_cogvideox_inputs(network, device, dtype):
    """Generate inputs for CogVideoX."""
    batch_size = 1
    C, T, H, W = 16, 2, 4, 4
    x = torch.randn(batch_size, C, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    # Use random dummy embeddings instead of text encoder (T5-XXL: seq_len=226, hidden_dim=4096)
    condition = torch.randn(batch_size, 226, 4096, device=device, dtype=network.precision_model)
    return {"x_t": x, "t": t, "condition": condition}


def generate_wan_1_3b_inputs(network, device, dtype):
    """Generate inputs for Wan 1.3B."""
    batch_size = 1
    T, H, W = 2, 4, 4
    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    condition = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)
    return {"x_t": x, "t": t, "condition": condition}


def check_wan_precision(network, inputs):
    """Check every Wan parameter's sharding and mixed-precision compute dtype, and the time path's.

    The parameter dtypes outside a forward are the sharded/master dtypes, not
    the mixed-precision all-gather dtypes. Hooks therefore record every direct
    parameter when its owning module executes, after FSDP has materialized the
    corresponding group for compute. A root post-hook summarizes any missing
    or incorrectly cast parameters after the complete forward. Every parameter
    computes in ``precision_model``; with ``keep_original_prec`` the time
    embedding, its projection and the output projection compute from them in
    at least FP32 under autocast, like original Wan.
    """
    del inputs  # inputs stay full precision; each network casts its neural inputs to precision_model
    root = network.transformer
    compute_dtype = str(network.precision_model).replace("torch.", "")
    time_path_dtype = network.precision_model
    if network._keep_original_prec:
        time_path_dtype = torch.promote_types(time_path_dtype, torch.float32)
    all_parameters = dict(root.named_parameters())
    canonical_name = {id(parameter): name for name, parameter in all_parameters.items()}
    seen = {}

    def spy(parameter_names):
        def hook(module, args):
            del args
            for local_name, name in parameter_names.items():
                parameter = module._parameters[local_name]
                seen.setdefault(name, str(parameter.dtype).replace("torch.", ""))

        return hook

    for module in root.modules():
        parameter_names = {
            local_name: canonical_name[id(parameter)]
            for local_name, parameter in module.named_parameters(recurse=False)
        }
        if parameter_names:
            module.register_forward_pre_hook(spy(parameter_names))

    time_path = {}
    root.condition_embedder.register_forward_hook(
        lambda module, args, output: time_path.update(temb=output[0].dtype, timestep_proj=output[1].dtype)
    )
    root.proj_out.register_forward_hook(lambda module, args, output: time_path.update(head=output.dtype))
    summary = {}

    def summarize(module, args, output):
        del module, args, output
        summary.update(
            {
                "parameter_count": len(all_parameters),
                "observed_parameter_count": len(seen),
                "missing_compute_dtypes": sorted(set(all_parameters) - set(seen)),
                "wrong_compute_dtypes": {
                    name: {"expected": compute_dtype, "actual": dtype}
                    for name, dtype in seen.items()
                    if dtype != compute_dtype
                },
                "time_path_outputs": sorted(time_path),
                "wrong_time_path_dtypes": {
                    name: {"expected": str(time_path_dtype), "actual": str(dtype)}
                    for name, dtype in time_path.items()
                    if dtype != time_path_dtype
                },
            }
        )

    root.register_forward_hook(summarize)
    return {
        "wan_parameter_precision": summary,
        "wan_wrong_sharded_dtypes": {
            name: str(parameter.dtype).replace("torch.", "")
            for name, parameter in all_parameters.items()
            if parameter.dtype != torch.float32
        },
        "wan_wrong_sharding": {
            name: [type(placement).__name__ for placement in getattr(parameter, "placements", ())]
            for name, parameter in all_parameters.items()
            if [type(placement).__name__ for placement in getattr(parameter, "placements", ())] != ["Shard"]
        },
    }


def _assert_wan_precision_result(result):
    precision = result["wan_parameter_precision"]
    assert precision["parameter_count"] > 0
    assert precision["observed_parameter_count"] == precision["parameter_count"], precision
    assert not precision["missing_compute_dtypes"], precision["missing_compute_dtypes"]
    assert not precision["wrong_compute_dtypes"], precision["wrong_compute_dtypes"]
    assert precision["time_path_outputs"] == ["head", "temb", "timestep_proj"], precision
    assert not precision["wrong_time_path_dtypes"], precision["wrong_time_path_dtypes"]
    assert not result["wan_wrong_sharded_dtypes"], result["wan_wrong_sharded_dtypes"]
    assert not result["wan_wrong_sharding"], result["wan_wrong_sharding"]


def generate_causal_wan_1_3b_inputs(network, device, dtype):
    """Generate inputs for CausalWan 1.3B."""
    batch_size = 1
    C, T, H, W = 16, 3, 4, 4
    x = torch.randn(batch_size, C, T, H, W, device=device, dtype=dtype)
    t_1d = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    t = t_1d.unsqueeze(1).expand(batch_size, T)  # 2D timesteps
    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    condition = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)
    return {"x_t": x, "t": t, "condition": condition}


def generate_wan22_5b_inputs(network, device, dtype):
    """Generate inputs for Wan 2.2 5B."""
    batch_size = 1
    T, H, W = 2, 4, 4
    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    condition = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)
    return {"x_t": x, "t": t, "condition": condition}


def generate_vace_wan_1_3b_inputs(network, device, dtype):
    """Generate inputs for VACE Wan 1.3B."""
    batch_size = 1
    T_latent, H_latent, W_latent = 2, 4, 4
    x = torch.randn(batch_size, 16, T_latent, H_latent, W_latent, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="logitnormal", device=device)
    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)
    # Use random dummy vid_context instead of prepare_vid_conditioning (shape: [B, 96, T, H, W])
    vid_context = torch.randn(
        batch_size, 96, T_latent, H_latent, W_latent, device=device, dtype=network.precision_model
    )
    condition = {"text_embeds": text_embeds, "vid_context": vid_context}
    return {"x_t": x, "t": t, "condition": condition}


def generate_causal_vace_wan_1_3b_inputs(network, device, dtype):
    """Generate inputs for causal VACE Wan 1.3B."""
    batch_size = 1
    num_frames, height, width = 3, 4, 4
    x = torch.randn(batch_size, 16, num_frames, height, width, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    t = t.unsqueeze(1).expand(batch_size, num_frames)
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)
    vid_context = torch.randn(batch_size, 96, num_frames, height, width, device=device, dtype=network.precision_model)
    return {"x_t": x, "t": t, "condition": {"text_embeds": text_embeds, "vid_context": vid_context}}


def generate_wan21_14b_inputs(network, device, dtype):
    """Generate inputs for Wan 2.1 14B T2V."""
    batch_size = 1
    T, H, W = 2, 4, 4
    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    condition = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)
    return {"x_t": x, "t": t, "condition": condition}


def generate_cosmos3_inputs(network, device, dtype):
    """Generate inputs for Cosmos3-Nano.

    Uses random Qwen2 token IDs instead of the tokenizer (the joint transformer embeds
    raw IDs — there is no separate text encoder), matching the dummy-embedding pattern
    of the other generators.
    """
    batch_size = 1
    T, H, W = 2, 4, 4
    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    input_ids = [torch.randint(0, 1000, (16,), device=device, dtype=torch.long) for _ in range(batch_size)]
    return {"x_t": x, "t": t, "condition": {"input_ids": input_ids}}


def generate_cosmos_predict2_inputs(network, device, dtype):
    """Generate inputs for the tiny real CosmosPredict2 FSDP boundary test."""
    batch_size = 1
    x = torch.randn(batch_size, 2, 2, 2, 2, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    condition = torch.randn(batch_size, 3, 8, device=device, dtype=network.precision_model)
    fps = torch.full((batch_size,), 24.0, device=device, dtype=dtype)
    padding_mask = torch.ones(batch_size, 2, 2, device=device, dtype=torch.bool)
    return {"x_t": x, "t": t, "condition": condition, "fps": fps, "padding_mask": padding_mask}


def generate_cosmos_predict2_2b_inputs(network, device, dtype):
    """Generate small-spatial inputs for the released 2B architecture."""
    batch_size = 1
    x = torch.randn(batch_size, 16, 2, 4, 4, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    condition = torch.randn(batch_size, 3, 100352, device=device, dtype=network.precision_model)
    fps = torch.full((batch_size,), 16.0, device=device, dtype=dtype)
    padding_mask = torch.ones(batch_size, 4, 4, device=device, dtype=torch.bool)
    return {"x_t": x, "t": t, "condition": condition, "fps": fps, "padding_mask": padding_mask}


def generate_wan22_i2v_5b_inputs(network, device, dtype):
    """Generate inputs for Wan 2.2 I2V 5B."""
    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 16, width // 16

    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)

    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)

    # Use random dummy first_frame_cond instead of VAE encoding (shape: [B, 48, 1, H, W])
    first_frame_cond = torch.randn(batch_size, 48, 1, H, W, device=device, dtype=dtype)

    condition = {"text_embeds": text_embeds, "first_frame_cond": first_frame_cond}
    return {"x_t": x, "t": t, "condition": condition}


def generate_causal_wan22_i2v_5b_inputs(network, device, dtype):
    """Generate inputs for CausalWan 2.2 I2V 5B."""
    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 16, width // 16

    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=dtype)
    t_1d = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    t = t_1d.unsqueeze(1).expand(batch_size, T)

    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)

    # Use random dummy first_frame_cond instead of VAE encoding (shape: [B, 48, 1, H, W])
    first_frame_cond = torch.randn(batch_size, 48, 1, H, W, device=device, dtype=dtype)

    condition = {"text_embeds": text_embeds, "first_frame_cond": first_frame_cond}
    return {"x_t": x, "t": t, "condition": condition}


def generate_wan21_i2v_14b_inputs(network, device, dtype):
    """Generate inputs for Wan 2.1 I2V 14B."""
    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 8, width // 8

    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)

    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)

    # Use random dummy image encoder hidden states (CLIP ViT-H/14: seq_len=257, hidden_dim=1280)
    encoder_hidden_states_image = torch.randn(batch_size, 257, 1280, device=device, dtype=network.precision_model)

    # Use random dummy first_frame_cond instead of VAE encoding (shape: [B, 16, T, H, W])
    first_frame_cond = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)

    condition = {
        "text_embeds": text_embeds,
        "first_frame_cond": first_frame_cond,
        "encoder_hidden_states_image": encoder_hidden_states_image,
    }
    return {"x_t": x, "t": t, "condition": condition}


def generate_causal_wan21_i2v_14b_inputs(network, device, dtype):
    """Generate inputs for CausalWan 2.1 I2V 14B."""
    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 8, width // 8

    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    t_1d = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    t = t_1d.unsqueeze(1).expand(batch_size, T)

    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)

    # Use random dummy image encoder hidden states (CLIP ViT-H/14: seq_len=257, hidden_dim=1280)
    encoder_hidden_states_image = torch.randn(batch_size, 257, 1280, device=device, dtype=network.precision_model)

    # Use random dummy first_frame_cond instead of VAE encoding (shape: [B, 16, T, H, W])
    first_frame_cond = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)

    condition = {
        "text_embeds": text_embeds,
        "first_frame_cond": first_frame_cond,
        "encoder_hidden_states_image": encoder_hidden_states_image,
    }
    return {"x_t": x, "t": t, "condition": condition}


def generate_wan22_t2v_a14b_inputs(network, device, dtype):
    """Generate inputs for Wan 2.2 T2V A14B (``WanMoET2V``).

    ``batch_size=2`` with timesteps that straddle ``guidance_boundary`` so the
    MoE routes one element to the high-noise expert and one to the low-noise
    expert. This forces the batch-split forward (both experts run; neither is
    skipped via the single-expert shortcut). Every rank feeds the same batch, so
    routing is symmetric across ranks;
    ``test_fsdp_wan_moe_rank_consistent_empty_expert_routing`` covers the
    rank-symmetric batch-size-1 case where one expert receives an empty batch.
    """
    batch_size = 2
    T, H, W = 2, 4, 4
    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    boundary = float(network.guidance_boundary)
    # One element above the boundary (-> high expert), one below (-> low expert).
    # Timesteps travel in the scheduler precision (as ``sample_t`` returns them), not the input dtype.
    t = torch.tensor(
        [min(boundary + 0.05, 0.99), max(boundary - 0.2, 0.05)],
        device=device,
        dtype=network.noise_scheduler.t_precision,
    )
    # Random dummy UMT5 embeddings (seq_len=512, hidden_dim=4096).
    condition = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)
    return {"x_t": x, "t": t, "condition": condition}


def generate_wan22_i2v_a14b_inputs(network, device, dtype):
    """Generate inputs for Wan 2.2 I2V A14B (``WanMoEI2V``).

    Wan 2.2 I2V A14B uses concat-mask first-frame conditioning and no CLIP
    image encoder (``use_image_encoder=False``), so the condition is just text
    embeddings + a first-frame latent. ``batch_size=2`` with straddling
    timesteps exercises per-element MoE routing across both experts.
    """
    batch_size = 2
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 8, width // 8
    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    boundary = float(network.guidance_boundary)
    # Timesteps travel in the scheduler precision (as ``sample_t`` returns them), not the input dtype.
    t = torch.tensor(
        [min(boundary + 0.05, 0.99), max(boundary - 0.2, 0.05)],
        device=device,
        dtype=network.noise_scheduler.t_precision,
    )
    # Random dummy UMT5 embeddings (seq_len=512, hidden_dim=4096).
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)
    # Random dummy first-frame latent (16-channel Wan VAE): [B, 16, T, H, W].
    first_frame_cond = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    condition = {"text_embeds": text_embeds, "first_frame_cond": first_frame_cond}
    return {"x_t": x, "t": t, "condition": condition}


def generate_lingbot_world_inputs(network, device, dtype):
    """Generate routed A14B MoE inputs that also exercise LingBot camera conditioning."""
    inputs = generate_wan22_i2v_a14b_inputs(network, device, dtype)
    x = inputs["x_t"]
    inputs["condition"]["c2ws_plucker_emb"] = torch.randn(
        x.shape[0], 384, x.shape[2], x.shape[3], x.shape[4], device=device, dtype=network.precision_model
    )
    return inputs


def generate_wancami2v_5b_inputs(network, device, dtype):
    """Generate inputs for WanCamI2V (Wan2.2-TI2V-5B + camera Plücker conditioning).

    Adds a random ``c2ws_plucker_emb`` to the WanI2V condition so the camera
    branch runs under FSDP: the global camera embedding
    (``camera_patch_embedding`` / ``camera_hidden_layer{1,2}``) and the
    per-block camera injection. The Plücker channel count follows the model's
    ``latent_compression_factors`` (``network.plucker_channels``: 6 * 16 * 16 for
    the 16x Wan2.2 VAE) and the spatial dims match the latent grid (patch_size (1, 2, 2)).
    """
    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 16, width // 16
    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    # Random dummy UMT5 embeddings (seq_len=512, hidden_dim=4096).
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)
    # Random dummy first-frame latent (48-channel Wan2.2 VAE): [B, 48, 1, H, W].
    first_frame_cond = torch.randn(batch_size, 48, 1, H, W, device=device, dtype=dtype)
    # Plücker camera embedding: [B, plucker_channels, lat_f, lat_h, lat_w].
    c2ws_plucker_emb = torch.randn(
        batch_size, network.plucker_channels, T, H, W, device=device, dtype=network.precision_model
    )
    condition = {
        "text_embeds": text_embeds,
        "first_frame_cond": first_frame_cond,
        "c2ws_plucker_emb": c2ws_plucker_emb,
    }
    return {"x_t": x, "t": t, "condition": condition}


def generate_flux_inputs(network, device, dtype):
    """Generate inputs for Flux."""
    batch_size = 1
    # Flux operates on latent space: [B, C, H, W] where C=16 for Flux VAE
    x = torch.randn(batch_size, 16, 8, 8, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)

    # Use random dummy embeddings instead of text encoder
    # CLIP pooled embeddings: [B, 768]
    pooled_prompt_embeds = torch.randn(batch_size, 768, device=device, dtype=network.precision_model)
    # T5 text embeddings: [B, seq_len, 4096] (T5-XXL hidden dim is 4096)
    prompt_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=network.precision_model)

    condition = (pooled_prompt_embeds, prompt_embeds)

    # Flux uses embedded guidance (guidance distillation)
    guidance = torch.full((batch_size,), 3.5, device=device, dtype=dtype)

    return {"x_t": x, "t": t, "condition": condition, "guidance": guidance}


def generate_qwen_image_inputs(network, device, dtype):
    """Generate inputs for QwenImage."""
    batch_size = 1
    # QwenImage latent: [B, 16, H, W], H/W must be even for 2x2 packing
    x = torch.randn(batch_size, 16, 8, 8, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)

    # Use random dummy embeddings instead of Qwen2.5-VL text encoder (hidden_dim=3584)
    encoder_hidden_states = torch.randn(batch_size, 512, 3584, device=device, dtype=network.precision_model)
    encoder_hidden_states_mask = torch.ones(batch_size, 512, device=device, dtype=torch.long)
    condition = (encoder_hidden_states, encoder_hidden_states_mask)

    return {"x_t": x, "t": t, "condition": condition}


def _generate_flux2_inputs(network, device, dtype, image_edit: bool = False):
    """Generate inputs for any FLUX.2 variant.

    Avoids materialising the (Mistral / Qwen3) text encoder by reading
    ``joint_attention_dim`` straight off the transformer config and
    synthesising random embeddings of the matching shape. ``image_edit``
    bundles a reference latent into a dict ``condition``.
    """
    batch_size = 1
    # Flux.2 VAE has 32 latent channels; H/W must be even (2x2 patchify).
    x = torch.randn(batch_size, 32, 8, 8, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)

    # joint_attention_dim = len(hidden_state_layers) * text_encoder_hidden_size.
    # dev: 5120 * 3 = 15360; klein-base-4B: 2560 * 3 = 7680;
    # klein-base-9B: 4096 * 3 = 12288. Pull dynamically so the same
    # generator works for every variant.
    joint_attention_dim = network.transformer.config.joint_attention_dim
    prompt_embeds = torch.randn(batch_size, 64, joint_attention_dim, device=device, dtype=network.precision_model)
    # Flux.2 has no pooled CLIP embedding; the slot is always None.
    text_condition = (None, prompt_embeds)

    if image_edit:
        # Reference (source) latent shares the noisy target's shape.
        ref_latent = torch.randn(batch_size, 32, 8, 8, device=device, dtype=network.precision_model)
        condition = {"text_embeds": text_condition, "condition_image": ref_latent}
    else:
        condition = text_condition

    inputs = {"x_t": x, "t": t, "condition": condition}
    # Only guidance-distilled variants (dev) take an embedded guidance
    # vector; klein-base sets `guidance_scale=None`.
    if getattr(network, "guidance_scale", None) is not None:
        inputs["guidance"] = torch.full((batch_size,), float(network.guidance_scale), device=device, dtype=dtype)
    return inputs


def generate_flux2_inputs(network, device, dtype):
    """Generate inputs for FLUX.2 text-to-image (no reference image)."""
    return _generate_flux2_inputs(network, device, dtype, image_edit=False)


def generate_flux2_image_edit_inputs(network, device, dtype):
    """Generate inputs for FLUX.2 image-editing (with reference latent)."""
    return _generate_flux2_inputs(network, device, dtype, image_edit=True)


def generate_ltx2_inputs(network, device, dtype):
    """Generate inputs for LTX-2 T2V."""
    batch_size = 1
    F, H, W = 2, 4, 4
    C = network.transformer.config.in_channels
    x = torch.randn(batch_size, C, F, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)

    connector_in_dim = network.connectors.text_proj_in.in_features
    seq_len = 128
    prompt_embeds = torch.randn(batch_size, seq_len, connector_in_dim, device=device, dtype=network.connector_precision)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    condition = (prompt_embeds, attention_mask)

    return {"x_t": {"video": x}, "t": t, "condition": condition}


def generate_minimax_h3_inputs(network, device, dtype):
    """Generate inputs for MiniMax-H3 (video and audio in one packed sequence).

    Latent frames follow ``F = 5n + 2``, the shape of the 17-frames-to-5-latents chunking, and
    the audio shape is derived from the video one on the shared 40 Hz clock. The conditioner is
    a 62 GB Qwen3-VL, so the prompt embedding is synthesized: what the transformer reads from it
    is a hidden state plus the per-row modality tags.
    """
    from fastgen.networks.MiniMaxH3.packing import MINIMAX_H3_TEXT_TAG

    batch_size, seq_len = 1, 16
    shape = network.infer_shape({"video": [network.video_latent_channels, 7, 4, 4], "audio": None})
    x_t = {name: torch.randn(batch_size, *dims, device=device, dtype=dtype) for name, dims in shape.items()}
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    condition = {
        "prompt_embeds": torch.randn(
            batch_size, seq_len, network.transformer.config.text_dim, device=device, dtype=dtype
        ),
        "text_token_tags": torch.full((seq_len,), MINIMAX_H3_TEXT_TAG, dtype=torch.long, device=device),
    }

    return {"x_t": x_t, "t": t, "condition": condition}


def check_minimax_h3_precision(network, inputs):
    """Record MiniMax-H3's explicit FP32 groups from the wrapped module.

    Its float32 carve-out has to *compute* in float32: those modules get their own FSDP groups
    with a float32 ``param_dtype`` so the release's mixed precision survives training, and folded
    into the bfloat16 groups the only symptom would be quietly worse numerics -- against a float32
    reference forward the carve-out roughly halves the error.

    The hooks fill their dict in place, which is why the generic implementation merges this
    function's return value only after the backward.
    """
    seen = {}

    def spy(name):
        def hook(module, args):
            # ``time_embedder`` is an MLP rather than a Linear, so read the first parameter; by
            # this point FSDP's own pre-forward hook has all-gathered it at the group's dtype.
            seen.setdefault(name, str(next(module.parameters()).dtype).replace("torch.", ""))

        return hook

    carve_out = [
        ("proj_in", network.transformer.proj_in),
        ("audio_proj_in", network.transformer.audio_proj_in),
        ("time_embedder", network.transformer.time_embedder),
        ("proj_out", network.transformer.proj_out),
        ("audio_proj_out", network.transformer.audio_proj_out),
    ]
    for name, module in carve_out:
        module.register_forward_pre_hook(spy(name))

    return {
        "carve_out_compute_dtype": seen,
        "carve_out_sharded": {
            name: all(type(parameter).__name__ == "DTensor" for parameter in module.parameters())
            for name, module in carve_out
        },
    }


def check_minimax_h3_gradients(network, inputs):
    """Verify that every explicit FP32 group participates in the backward."""
    del inputs
    return {
        "carve_out_has_grads": {
            name: all(parameter.grad is not None for parameter in module.parameters())
            for name, module in (
                ("proj_in", network.transformer.proj_in),
                ("audio_proj_in", network.transformer.audio_proj_in),
                ("time_embedder", network.transformer.time_embedder),
                ("proj_out", network.transformer.proj_out),
                ("audio_proj_out", network.transformer.audio_proj_out),
            )
        }
    }


def generate_ltx2_i2v_inputs(network, device, dtype):
    """Generate inputs for LTX-2 I2V."""
    batch_size = 1
    F, H, W = 2, 4, 4
    C = network.transformer.config.in_channels
    x = torch.randn(batch_size, C, F, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)

    connector_in_dim = network.connectors.text_proj_in.in_features
    seq_len = 128
    prompt_embeds = torch.randn(batch_size, seq_len, connector_in_dim, device=device, dtype=network.connector_precision)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    text_embeds = (prompt_embeds, attention_mask)

    frame_cond = torch.randn(batch_size, C, 1, H, W, device=device, dtype=dtype)
    condition = {"text_embeds": text_embeds, "frame_cond": frame_cond}

    return {"x_t": {"video": x}, "t": t, "condition": condition}


# =============================================================================
# Test Implementation Functions
# =============================================================================


def _test_edm_cifar10_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import EDM_CIFAR10_Config

    return _generic_fsdp_test_impl(
        world_size,
        EDM_CIFAR10_Config,
        generate_edm_cifar10_inputs,
        apply_checkpointing=False,  # EDM doesn't have grad checkpointing
    )


def _test_edm_imagenet64_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import EDM_ImageNet64_Config

    return _generic_fsdp_test_impl(
        world_size, EDM_ImageNet64_Config, generate_edm_imagenet64_inputs, apply_checkpointing=False
    )


def _test_edm2_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import EDM2_IN64_S_Config

    return _generic_fsdp_test_impl(world_size, EDM2_IN64_S_Config, generate_edm2_inputs, apply_checkpointing=False)


def _record_parameters_unsharded_after_forward(network, inputs):
    """Fill, once the network's forward returns, the names of the parameters FSDP still holds gathered."""
    from torch.distributed.tensor import DTensor

    del inputs
    result = {}

    def record(module, args, output):
        result["unsharded_after_forward"] = [
            name for name, parameter in module.named_parameters() if not isinstance(parameter, DTensor)
        ]

    network.register_forward_hook(record)
    return result


def _test_dit_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import DiT_IN256_S_Config

    return _generic_fsdp_test_impl(
        world_size,
        DiT_IN256_S_Config,
        generate_dit_inputs,
        apply_checkpointing=False,
        extra_result_fn=_record_parameters_unsharded_after_forward,
    )


def _test_logvar_head_fsdp_impl(rank: int, world_size: int, architecture: str, cpu_offload: bool) -> Dict:
    """Compare a sharded uncertainty head against the equivalent global-batch gradient."""
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    from fastgen.utils.distributed.fsdp import create_fsdp_context

    device = torch.device("cuda", rank)
    torch.manual_seed(20260914)

    if architecture == "dit":
        from fastgen.networks.DiT.network import DiT

        network = DiT(
            input_size=2,
            patch_size=1,
            in_channels=1,
            hidden_size=8,
            depth=1,
            num_heads=1,
            mlp_ratio=1.0,
            enable_class_dropout=False,
            label_dim=2,
            scale_t=False,
        )
        x_t = torch.full((1, 1, 2, 2), 0.25 * (rank + 1), device=device)
        timestep = torch.tensor([0.2 + 0.6 * rank], device=device, dtype=network.noise_scheduler.t_precision)
        condition = torch.tensor([rank], device=device, dtype=torch.long)
    elif architecture == "edm2":
        from fastgen.networks.EDM2.network import EDM2Precond

        network = EDM2Precond(
            img_resolution=4,
            img_channels=1,
            label_dim=2,
            logvar_channels=8,
            model_channels=8,
            channel_mult=[1],
            num_blocks=1,
            attn_resolutions=[],
            dropout=0.0,
        )
        x_t = torch.full((1, 1, 4, 4), 0.25 * (rank + 1), device=device)
        timestep = torch.tensor([0.5 + rank], device=device, dtype=network.noise_scheduler.t_precision)
        condition = torch.zeros(1, 2, device=device)
        condition[:, rank] = 1.0
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # This copy is an exact, unsharded oracle for the uncertainty head. The
    # full-network input to it is captured below, so the comparison isolates
    # FSDP's ownership and gradient reduction from the rest of the architecture.
    reference_head = copy.deepcopy(network.logvar_linear).to(device=device, dtype=torch.float32)
    fsdp_context = create_fsdp_context(
        reduce_dtype=torch.float32,
        min_num_params=1,
        apply_cpu_offload=cpu_offload,
    )
    network.prepare_module(
        device=device,
        precision_model=torch.float32,
        precision_opt=torch.float32,
        trainable=True,
        fsdp_context=fsdp_context,
    )

    head = network.logvar_linear
    head_parameters = dict(head.named_parameters(recurse=False))
    reference_parameters = dict(reference_head.named_parameters(recurse=False))
    head_is_fsdp = isinstance(head, FSDPModule)
    parameters_are_dtensors = bool(head_parameters) and all(
        isinstance(parameter, DTensor) for parameter in head_parameters.values()
    )
    initial_parameter_error = max(
        float(
            (parameter.full_tensor().detach().float().cpu() - reference_parameters[name].detach().float().cpu())
            .abs()
            .max()
            .item()
        )
        for name, parameter in head_parameters.items()
    )
    shard_devices_before_forward = {
        parameter.to_local().device.type if isinstance(parameter, DTensor) else parameter.device.type
        for parameter in head_parameters.values()
    }

    captured_head_inputs = []
    handle = head.register_forward_pre_hook(lambda _module, args: captured_head_inputs.append(args[0].detach().clone()))
    try:
        prediction, logvar = network(
            x_t=x_t,
            t=timestep,
            condition=condition,
            return_logvar=True,
        )
    finally:
        handle.remove()
    assert len(captured_head_inputs) == 1

    # Keep the prediction in the same backward while giving the uncertainty
    # branch its own nonzero objective, as it has in the training loss. A fixed
    # target keeps this valid if an architecture later zero-initializes its head.
    def uncertainty_loss(value):
        return (value.float() - 1.0).square().mean()

    prediction_loss = prediction.float().square().mean()
    logvar_loss = uncertainty_loss(logvar)
    (prediction_loss + logvar_loss).backward()

    gradients_are_dtensors = True
    gradients_are_finite = True
    shard_devices_after_backward = set()
    actual_gradients = {}
    for name, parameter in head_parameters.items():
        gradient = parameter.grad
        gradients_are_dtensors &= isinstance(gradient, DTensor)
        if gradient is None:
            gradients_are_finite = False
            continue
        local_gradient = gradient.to_local() if isinstance(gradient, DTensor) else gradient
        gradients_are_finite &= bool(torch.isfinite(local_gradient).all().item())
        shard_devices_after_backward.add(local_gradient.device.type)
        actual_gradients[name] = (
            gradient.full_tensor().detach().float().cpu()
            if isinstance(gradient, DTensor)
            else gradient.detach().float().cpu()
        )

    # Rank-local timesteps deliberately produce different head gradients. A
    # correct FSDP reduction must equal the reference loss over their union.
    local_head_input = captured_head_inputs[0]
    all_head_inputs = [torch.empty_like(local_head_input) for _ in range(world_size)]
    dist.all_gather(all_head_inputs, local_head_input)
    reference_parameter_values = tuple(reference_parameters.values())
    local_reference_loss = uncertainty_loss(reference_head(local_head_input))
    local_reference_gradients = torch.autograd.grad(local_reference_loss, reference_parameter_values)
    local_reference_flat = torch.cat([gradient.detach().flatten() for gradient in local_reference_gradients])
    all_local_reference_gradients = [torch.empty_like(local_reference_flat) for _ in range(world_size)]
    dist.all_gather(all_local_reference_gradients, local_reference_flat)
    local_gradient_difference = max(
        float((other - all_local_reference_gradients[0]).abs().max().item())
        for other in all_local_reference_gradients[1:]
    )

    global_reference_loss = uncertainty_loss(reference_head(torch.cat(all_head_inputs)))
    global_reference_gradients = torch.autograd.grad(global_reference_loss, reference_parameter_values)
    expected_gradients = {
        name: gradient.detach().float().cpu()
        for name, gradient in zip(reference_parameters, global_reference_gradients)
    }
    max_gradient_error = max(
        float((actual_gradients[name] - expected).abs().max().item()) for name, expected in expected_gradients.items()
    )
    max_gradient_relative_error = max(
        float(((actual_gradients[name] - expected).norm() / expected.norm().clamp_min(1e-7)).item())
        for name, expected in expected_gradients.items()
    )

    local_result = {
        "rank": rank,
        "head_is_fsdp": head_is_fsdp,
        "parameters_are_dtensors": parameters_are_dtensors,
        "initial_parameter_error": initial_parameter_error,
        "gradients_are_dtensors": gradients_are_dtensors,
        "gradients_are_finite": gradients_are_finite,
        "local_gradient_difference": local_gradient_difference,
        "max_gradient_error": max_gradient_error,
        "max_gradient_relative_error": max_gradient_relative_error,
        "prediction_is_finite": bool(torch.isfinite(prediction).all().item()),
        "logvar_is_finite": bool(torch.isfinite(logvar).all().item()),
        "logvar_loss": float(logvar_loss.detach().item()),
        "shard_devices_before_forward": sorted(shard_devices_before_forward),
        "shard_devices_after_backward": sorted(shard_devices_after_backward),
    }
    per_rank = [None] * world_size
    dist.all_gather_object(per_rank, local_result)
    return {"architecture": architecture, "cpu_offload": cpu_offload, "per_rank": per_rank}


def _assert_logvar_head_fsdp_result(result: Dict, architecture: str, cpu_offload: bool) -> None:
    assert result is not None
    assert result["architecture"] == architecture
    assert result["cpu_offload"] is cpu_offload
    assert [entry["rank"] for entry in result["per_rank"]] == [0, 1]
    expected_shard_device = ["cpu" if cpu_offload else "cuda"]
    for entry in result["per_rank"]:
        assert entry["head_is_fsdp"], entry
        assert entry["parameters_are_dtensors"], entry
        assert entry["initial_parameter_error"] == 0.0, entry
        assert entry["gradients_are_dtensors"], entry
        assert entry["gradients_are_finite"], entry
        assert entry["prediction_is_finite"], entry
        assert entry["logvar_is_finite"], entry
        assert entry["logvar_loss"] > 0.0, entry
        assert entry["local_gradient_difference"] > 1e-6, entry
        assert entry["max_gradient_error"] < 1e-5, entry
        assert entry["max_gradient_relative_error"] < 1e-5, entry
        assert entry["shard_devices_before_forward"] == expected_shard_device, entry
        assert entry["shard_devices_after_backward"] == expected_shard_device, entry


def _test_sd15_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import SD15Config

    return _generic_fsdp_test_impl(world_size, SD15Config, generate_sd15_inputs, apply_checkpointing=False)


def _test_sdxl_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import SDXLConfig

    return _generic_fsdp_test_impl(world_size, SDXLConfig, generate_sdxl_inputs, apply_checkpointing=False)


def _test_cogvideox_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CogVideoXConfig

    # CogVideoX 2B was trained in fp16
    return _generic_fsdp_test_impl(
        world_size,
        CogVideoXConfig,
        generate_cogvideox_inputs,
        apply_checkpointing=True,
        param_dtype=torch.float16,
    )


def _test_cogvideox_5b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CogVideoX5BConfig

    # CogVideoX 5B was trained in bf16 and does not require fp32 for numerical stability
    return _generic_fsdp_test_impl(
        world_size,
        CogVideoX5BConfig,
        generate_cogvideox_inputs,
        apply_checkpointing=True,
    )


def _test_wan_1_3b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan_1_3B_Config

    return _generic_fsdp_test_impl(
        world_size,
        Wan_1_3B_Config,
        generate_wan_1_3b_inputs,
        apply_checkpointing=True,
        extra_result_fn=check_wan_precision,
    )


def _test_causal_wan_1_3b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CausalWan_1_3B_Config

    return _generic_fsdp_test_impl(
        world_size,
        CausalWan_1_3B_Config,
        generate_causal_wan_1_3b_inputs,
        apply_checkpointing=True,
    )


def _test_wan22_5b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan22_T2V_5B_Config

    return _generic_fsdp_test_impl(
        world_size,
        Wan22_T2V_5B_Config,
        generate_wan22_5b_inputs,
        apply_checkpointing=True,
    )


def _test_vace_wan_1_3b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import VACE_Wan_1_3B_Config

    return _generic_fsdp_test_impl(
        world_size,
        VACE_Wan_1_3B_Config,
        generate_vace_wan_1_3b_inputs,
        apply_checkpointing=True,
        extra_result_fn=check_wan_precision,
    )


def _test_causal_vace_wan_1_3b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CausalVACE_Wan_1_3B_Config

    return _generic_fsdp_test_impl(
        world_size,
        CausalVACE_Wan_1_3B_Config,
        generate_causal_vace_wan_1_3b_inputs,
        apply_checkpointing=True,
    )


def _test_wan21_14b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan21_T2V_14B_Config

    return _generic_fsdp_test_impl(
        world_size,
        Wan21_T2V_14B_Config,
        generate_wan21_14b_inputs,
        apply_checkpointing=True,
    )


def _test_wan22_i2v_5b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan22_I2V_5B_Config

    return _generic_fsdp_test_impl(
        world_size,
        Wan22_I2V_5B_Config,
        generate_wan22_i2v_5b_inputs,
        apply_checkpointing=True,
        extra_result_fn=check_wan_precision,
    )


def _test_causal_wan22_i2v_5b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CausalWan22_I2V_5B_Config

    return _generic_fsdp_test_impl(
        world_size,
        CausalWan22_I2V_5B_Config,
        generate_causal_wan22_i2v_5b_inputs,
        apply_checkpointing=True,
    )


def _test_wan21_i2v_14b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan21_I2V_14B_480P_Config

    return _generic_fsdp_test_impl(
        world_size,
        Wan21_I2V_14B_480P_Config,
        generate_wan21_i2v_14b_inputs,
        apply_checkpointing=True,
    )


def _test_cosmos3_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Cosmos3_Nano_Config

    return _generic_fsdp_test_impl(
        world_size,
        Cosmos3_Nano_Config,
        generate_cosmos3_inputs,
        apply_checkpointing=True,
    )


def _test_cosmos_predict2_impl(rank: int, world_size: int, use_wan_fp32_strategy: bool = False) -> Dict:
    config = L(CosmosPredict2)(
        model_channels=48,
        num_blocks=1,
        num_heads=4,
        max_img_h=4,
        max_img_w=4,
        max_frames=4,
        in_channels=2,
        out_channels=2,
        patch_spatial=1,
        patch_temporal=1,
        concat_padding_mask=False,
        add_video_condition_mask=False,
        mlp_ratio=2.0,
        crossattn_emb_channels=8,
        use_adaln_lora=False,
        enable_logvar_linear=False,
        use_crossattn_projection=False,
        use_wan_fp32_strategy=use_wan_fp32_strategy,
    )
    return _generic_fsdp_test_impl(
        world_size,
        config,
        generate_cosmos_predict2_inputs,
        apply_checkpointing=False,
    )


def _test_cosmos_predict2_2b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CosmosPredict2_2B_Config

    del rank
    return _generic_fsdp_test_impl(
        world_size,
        CosmosPredict2_2B_Config,
        generate_cosmos_predict2_2b_inputs,
        apply_checkpointing=False,
    )


def _test_causal_wan21_i2v_14b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CausalWan21_I2V_14B_480P_Config

    return _generic_fsdp_test_impl(
        world_size,
        CausalWan21_I2V_14B_480P_Config,
        generate_causal_wan21_i2v_14b_inputs,
        apply_checkpointing=True,
    )


def _test_wan22_t2v_a14b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan22_T2V_14B_Config

    return _generic_fsdp_test_impl(
        world_size,
        Wan22_T2V_14B_Config,
        generate_wan22_t2v_a14b_inputs,
        apply_checkpointing=True,
    )


def _test_wan22_i2v_a14b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan22_I2V_14B_Config

    return _generic_fsdp_test_impl(
        world_size,
        Wan22_I2V_14B_Config,
        generate_wan22_i2v_a14b_inputs,
        apply_checkpointing=True,
    )


def _test_lingbot_world_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import LingBot_World_Config

    del rank
    config = copy.deepcopy(LingBot_World_Config)
    config.load_pretrained = False
    return _generic_fsdp_test_impl(
        world_size,
        config,
        generate_lingbot_world_inputs,
        apply_checkpointing=True,
    )


def _test_wancami2v_5b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan_Kwargs
    from fastgen.networks.WanI2V.network_cam import WanCamI2V
    from fastgen.configs.instantiate import LazyCall as L

    # Built inline (no net.py config): WanCamI2V on the 5B base with camera
    # conditioning enabled. Mirrors Wan22_I2V_5B_Config's kwargs (Wan_Kwargs).
    config = L(WanCamI2V)(
        model_id_or_local_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        disable_efficient_attn=False,
        disable_grad_ckpt=False,
        **Wan_Kwargs,
    )
    return _generic_fsdp_test_impl(
        world_size,
        config,
        generate_wancami2v_5b_inputs,
        apply_checkpointing=True,
    )


def _test_flux_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import FluxConfig

    return _generic_fsdp_test_impl(world_size, FluxConfig, generate_flux_inputs, apply_checkpointing=True)


def _test_qwen_image_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import QwenImageConfig

    return _generic_fsdp_test_impl(world_size, QwenImageConfig, generate_qwen_image_inputs, apply_checkpointing=True)


def _test_flux2_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Flux2Config

    return _generic_fsdp_test_impl(world_size, Flux2Config, generate_flux2_inputs, apply_checkpointing=True)


def _test_flux2_klein_base_4b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Flux2KleinBase4BConfig

    return _generic_fsdp_test_impl(world_size, Flux2KleinBase4BConfig, generate_flux2_inputs, apply_checkpointing=True)


def _test_flux2_image_edit_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Flux2KleinBase4BImageEditConfig

    return _generic_fsdp_test_impl(
        world_size, Flux2KleinBase4BImageEditConfig, generate_flux2_image_edit_inputs, apply_checkpointing=True
    )


def _test_ltx2_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import LTX2Config

    # Keep sharded weights in bf16 to avoid OOM-kills on rank 0.
    return _generic_fsdp_test_impl(
        world_size,
        LTX2Config,
        generate_ltx2_inputs,
        apply_checkpointing=True,
        sharded_dtype=torch.bfloat16,
    )


def _test_minimax_h3_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import MiniMaxH3_Config

    # The net config loads bfloat16 (what inference wants) while the training configs load
    # float32, so leaving ``sharded_dtype`` at its float32 default exercises the pre-FSDP cast
    # that has to lift the masters -- the path the trainer takes with ``precision_opt``.
    return _generic_fsdp_test_impl(
        world_size,
        MiniMaxH3_Config,
        generate_minimax_h3_inputs,
        apply_checkpointing=True,
        extra_result_fn=check_minimax_h3_precision,
        post_backward_result_fn=check_minimax_h3_gradients,
    )


def _test_ltx2_i2v_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import LTX2I2VConfig

    return _generic_fsdp_test_impl(
        world_size,
        LTX2I2VConfig,
        generate_ltx2_i2v_inputs,
        apply_checkpointing=True,
        sharded_dtype=torch.bfloat16,
    )


# =============================================================================
# Pytest Test Functions - EDM Models (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2)
def test_fsdp_edm_cifar10():
    """Test FSDP forward+backward pass for EDM CIFAR10."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_edm_cifar10_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_edm_imagenet64():
    """Test FSDP forward+backward pass for EDM ImageNet64."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_edm_imagenet64_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_edm2():
    """Test FSDP forward+backward pass for EDM2."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_edm2_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_edm2_logvar_head_gradient_is_synchronized():
    """EDM2's uncertainty head is a real FSDP unit with globally averaged gradients."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_logvar_head_fsdp_impl,
        world_size=2,
        timeout=120,
        setup_fn=set_env_vars,
        architecture="edm2",
        cpu_offload=False,
    )
    _assert_logvar_head_fsdp_result(result, architecture="edm2", cpu_offload=False)
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - DiT (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2)
def test_fsdp_dit():
    """Test FSDP forward+backward pass for DiT."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_dit_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    # DiT's groups are separate FSDP roots; each must still free its gathered parameters after forward.
    assert result["unsharded_after_forward"] == [], result["unsharded_after_forward"]
    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_dit_logvar_head_gradient_is_synchronized_with_cpu_offload():
    """DiT's uncertainty head remains sharded and synchronized under CPU offload."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_logvar_head_fsdp_impl,
        world_size=2,
        timeout=120,
        setup_fn=set_env_vars,
        architecture="dit",
        cpu_offload=True,
    )
    _assert_logvar_head_fsdp_result(result, architecture="dit", cpu_offload=True)
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - SD15 (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_sd15():
    """Test FSDP forward+backward pass for SD15."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_sd15_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - SDXL (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_sdxl():
    """Test split-precision FSDP forward+backward for SDXL."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_sdxl_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - Flux (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2, min_gpu_mem=60)
@pytest.mark.large_model
def test_fsdp_flux():
    """Test FSDP forward+backward pass for Flux with gradient checkpointing."""
    clear_gpu_memory()

    # Check if Flux model is accessible before launching distributed test
    # Flux is a gated model that requires HuggingFace authentication
    try:
        from diffusers.models import FluxTransformer2DModel

        FluxTransformer2DModel.load_config(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
        )
    except Exception as e:
        if "not a valid model identifier" in str(e) or "token" in str(e).lower() or "gated" in str(e).lower():
            pytest.skip(f"Test skipped: Flux model not accessible (requires HuggingFace authentication): {e}")
        raise

    result = run_distributed_test(
        test_fn=_test_flux_impl,
        world_size=2,
        timeout=900,  # Flux model loading can take 9+ minutes
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - Flux.2 (2 GPUs)
# =============================================================================


def _check_flux2_accessible(model_id: str, variant_label: str):
    """Skip the test if the FLUX.2 transformer config can't be downloaded.

    All FLUX.2 variants ship under ``black-forest-labs/FLUX.2-*`` on HF; dev
    and klein-base-9B are gated, klein-base-4B is Apache-2.0. Use
    ``load_config`` (same trick as ``test_fsdp_flux``) so we surface a
    gating / auth failure as a skip rather than running 8 FSDP ranks just
    to crash inside ``from_pretrained``.
    """
    try:
        from diffusers.models import Flux2Transformer2DModel

        Flux2Transformer2DModel.load_config(model_id, subfolder="transformer")
    except Exception as e:
        msg = str(e).lower()
        if any(s in msg for s in ("gated", "401", "token", "not a valid model identifier", "is not a local folder")):
            pytest.skip(f"Test skipped: Flux.2 {variant_label} not accessible (HF auth / gating): {e}")
        raise


@RunIf(min_gpus=4, min_gpu_mem=80)
@pytest.mark.large_model
def test_fsdp_flux2():
    """Test FSDP forward+backward pass for FLUX.2-dev with gradient checkpointing."""
    clear_gpu_memory()
    _check_flux2_accessible("black-forest-labs/FLUX.2-dev", "dev")
    result = run_distributed_test(
        test_fn=_test_flux2_impl,
        # Float32 parameter and gradient shards for 32B parameters need four 80 GB GPUs.
        world_size=4,
        timeout=900,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=2, min_gpu_mem=40)
@pytest.mark.large_model
def test_fsdp_flux2_klein_base_4b():
    """Test FSDP forward+backward pass for FLUX.2-klein-base-4B with gradient checkpointing."""
    clear_gpu_memory()
    _check_flux2_accessible("black-forest-labs/FLUX.2-klein-base-4B", "klein-base-4B")
    result = run_distributed_test(
        test_fn=_test_flux2_klein_base_4b_impl,
        world_size=2,
        timeout=600,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=2, min_gpu_mem=40)
@pytest.mark.large_model
def test_fsdp_flux2_image_edit():
    """Test FSDP forward+backward pass for FLUX.2-klein-base-4B image-edit branch.

    Exercises ``enable_image_edit=True`` under FSDP: the reference latent
    is concatenated along the sequence dimension and the discriminator-
    visible target tokens are sliced out before the output reshape.
    """
    clear_gpu_memory()
    _check_flux2_accessible("black-forest-labs/FLUX.2-klein-base-4B", "klein-base-4B image-edit")
    result = run_distributed_test(
        test_fn=_test_flux2_image_edit_impl,
        world_size=2,
        timeout=600,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - QwenImage (8 GPUs)
# =============================================================================


@RunIf(min_gpus=8)
@pytest.mark.large_model
def test_fsdp_qwen_image():
    """Test FSDP forward+backward pass for QwenImage (~20B) with gradient checkpointing."""
    clear_gpu_memory()

    try:
        from diffusers.models import QwenImageTransformer2DModel

        QwenImageTransformer2DModel.load_config(
            "Qwen/Qwen-Image",
            subfolder="transformer",
        )
    except (HTTPError, OSError) as e:
        err_msg = str(e).lower()
        if "not a valid model identifier" in err_msg or "token" in err_msg or "gated" in err_msg:
            pytest.skip(f"Test skipped: QwenImage model not accessible: {e}")
        raise

    result = run_distributed_test(
        test_fn=_test_qwen_image_impl,
        world_size=8,
        timeout=900,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - CogVideoX (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_cogvideox():
    """Test FSDP forward+backward pass for CogVideoX with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_cogvideox_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_cogvideox_5b():
    """Test FSDP forward+backward pass for CogVideoX-5B with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_cogvideox_5b_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - Wan 1.3B (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_wan_1_3b():
    """Test FSDP forward+backward pass for Wan 1.3B with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan_1_3b_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    _assert_wan_precision_result(result)
    clear_gpu_memory()


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_causal_wan_1_3b():
    """Test FSDP forward+backward pass for CausalWan 1.3B with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_causal_wan_1_3b_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - Wan 5B (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_wan22_5b():
    """Test FSDP forward+backward pass for Wan 2.2 5B with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan22_5b_impl,
        world_size=2,
        timeout=600,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_wan22_i2v_5b():
    """Test FSDP forward+backward pass for Wan 2.2 I2V 5B with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan22_i2v_5b_impl,
        world_size=2,
        timeout=600,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    _assert_wan_precision_result(result)
    clear_gpu_memory()


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_causal_wan22_i2v_5b():
    """Test FSDP forward+backward pass for CausalWan 2.2 I2V 5B with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_causal_wan22_i2v_5b_impl,
        world_size=2,
        timeout=600,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - WanCamI2V camera conditioning (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_wancami2v_5b():
    """FSDP forward+backward for WanCamI2V (Wan2.2-TI2V-5B + camera conditioning).

    Exercises the camera path under FSDP: the global camera embedding
    (``camera_patch_embedding`` / ``camera_hidden_layer{1,2}``), the per-block
    zero-init camera injection, and the monkey-patched ``classify_forward`` that
    threads ``c2ws_plucker_emb`` through the blocks. The per-block injection is
    zero-initialised (identity forward) but still receives finite gradients, so
    the camera parameters are covered by the backward check. The single-GPU
    ``test_network_wancami2v_5b`` covers the same wiring without sharding.
    """
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wancami2v_5b_impl,
        world_size=2,
        timeout=600,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - VACE Wan 1.3B (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_vace_wan_1_3b():
    """Test FSDP forward+backward pass for VACE Wan 1.3B with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_vace_wan_1_3b_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    _assert_wan_precision_result(result)
    clear_gpu_memory()


# CausalVACE has a separate forward boundary and per-frame timestep path.
@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_causal_vace_wan_1_3b():
    """Test split-precision FSDP forward+backward for causal VACE Wan."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_causal_vace_wan_1_3b_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# CosmosPredict2 uses a separate structured video boundary. A tiny real model
# covers its FSDP mechanics without allocating the 2B checkpoint architecture.
@RunIf(min_gpus=2)
@pytest.mark.parametrize("use_wan_fp32_strategy", [False, True])
def test_fsdp_cosmos_predict2(use_wan_fp32_strategy):
    """Test split-precision FSDP forward+backward for CosmosPredict2."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=partial(_test_cosmos_predict2_impl, use_wan_fp32_strategy=use_wan_fp32_strategy),
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=8)
@pytest.mark.large_model
def test_fsdp_cosmos_predict2_2b():
    """Test the full 2B architecture with split-precision FSDP forward+backward."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_cosmos_predict2_2b_impl,
        world_size=8,
        timeout=900,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - Cosmos3-Nano (8 GPUs)
# =============================================================================


@RunIf(min_gpus=8)
@pytest.mark.large_model
def test_fsdp_cosmos3():
    """Test FSDP forward+backward pass for Cosmos3-Nano with gradient checkpointing.

    Covers the Cosmos3-specific ``fully_shard`` (per-MoT-decoder-layer sharding without
    wrapping ``self``) and the packed [text || vision] forward under FSDP2.
    """
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_cosmos3_impl,
        world_size=8,
        timeout=900,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - Wan 14B (8 GPUs)
# =============================================================================


@RunIf(min_gpus=8)
@pytest.mark.large_model
def test_fsdp_wan21_14b():
    """Test FSDP forward+backward pass for Wan 2.1 14B with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan21_14b_impl,
        world_size=8,
        timeout=900,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=8)
@pytest.mark.large_model
def test_fsdp_wan21_i2v_14b():
    """Test FSDP forward+backward pass for Wan 2.1 I2V 14B with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan21_i2v_14b_impl,
        world_size=8,
        timeout=900,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=8)
@pytest.mark.large_model
def test_fsdp_causal_wan21_i2v_14b():
    """Test FSDP forward+backward pass for CausalWan 2.1 I2V 14B with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_causal_wan21_i2v_14b_impl,
        world_size=8,
        timeout=900,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - Wan 2.2 A14B MoE (8 GPUs)
# =============================================================================


@RunIf(min_gpus=8)
@pytest.mark.large_model
def test_fsdp_wan22_t2v_a14b():
    """FSDP forward+backward for the Wan 2.2 T2V A14B MoE (``WanMoET2V``).

    Covers ``WanMoE.fully_shard`` (sharding both experts) plus the batch-split
    routed forward/backward with both experts active (timesteps straddle
    ``guidance_boundary``). ``LingBotWorld`` reuses this same MoE machinery.
    """
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan22_t2v_a14b_impl,
        world_size=8,
        timeout=900,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=8)
@pytest.mark.large_model
def test_fsdp_wan22_i2v_a14b():
    """FSDP forward+backward for the Wan 2.2 I2V A14B MoE (``WanMoEI2V``).

    Same MoE coverage as the T2V variant, on the image-to-video experts
    (concat-mask first-frame conditioning, no CLIP image encoder).
    """
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan22_i2v_a14b_impl,
        world_size=8,
        timeout=900,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=8)
@pytest.mark.large_model
def test_fsdp_lingbot_world():
    """FSDP forward/backward for LingBot's camera-conditioned A14B MoE."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_lingbot_world_impl,
        world_size=8,
        timeout=1200,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


# =============================================================================
# Wan MoE: FSDP routing (2 GPUs)
# =============================================================================

_WAN_MOE_TINY_TRANSFORMER_CONFIG = {
    "_class_name": "WanTransformer3DModel",
    "_diffusers_version": "0.35.0.dev0",
    "added_kv_proj_dim": None,
    "attention_head_dim": 16,
    "cross_attn_norm": True,
    "eps": 1e-06,
    "ffn_dim": 96,
    "freq_dim": 256,
    "image_dim": None,
    "in_channels": 36,
    "num_attention_heads": 2,
    "num_layers": 3,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "pos_embed_seq_len": None,
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 1024,
    "text_dim": 32,
}


def _write_tiny_wan22_i2v_model_dir(root: str) -> str:
    """Write ``transformer``/``transformer_2`` configs of a 3-block Wan 2.2 I2V A14B look-alike.

    Like A14B's, the configs make the concat-mask first-frame conditioning (a wider input than output) and no CLIP
    image encoder (no ``image_dim``).
    """
    model_dir = os.path.join(root, "Wan2.2-I2V-A14B-Diffusers")
    for subfolder in ("transformer", "transformer_2"):
        os.makedirs(os.path.join(model_dir, subfolder), exist_ok=True)
        with open(os.path.join(model_dir, subfolder, "config.json"), "w") as f:
            json.dump(_WAN_MOE_TINY_TRANSFORMER_CONFIG, f)
    return model_dir


def _build_tiny_wan_moe_pair(device, storage_dtype, compute_dtype, sdpa_no_cudnn_blocks, network_cls=None):
    """Return a to-be-sharded tiny Wan network and a non-sharded copy with identical weights.

    ``network_cls`` defaults to ``WanMoEI2V`` (two experts); ``WanI2V`` gives a single expert. Both copies
    are built from the same seed, so no broadcast is needed. The copy is cast the way FSDP will compute
    (``compute_dtype`` with Wan's FP32 rotary tables), so it serves as the gradient reference.
    """
    from fastgen.networks.Wan.wan_moe import WanMoE
    from fastgen.networks.WanI2V.wan_moe import WanMoEI2V

    network_cls = network_cls or WanMoEI2V

    def build():
        torch.manual_seed(0)
        network_kwargs = dict(
            model_id_or_local_path=model_dir,
            load_pretrained=False,
            enable_logvar_linear=False,
            r_timestep=False,
            keep_original_prec=True,
        )
        if issubclass(network_cls, WanMoE):
            expert_overrides = copy.deepcopy(network_cls.DEFAULT_EXPERT_OVERRIDES)
            if sdpa_no_cudnn_blocks is not None:
                expert_overrides["low"]["sdpa_no_cudnn_blocks"] = sdpa_no_cudnn_blocks
            network_kwargs["expert_overrides"] = expert_overrides
        else:
            network_kwargs["sdpa_no_cudnn_blocks"] = sdpa_no_cudnn_blocks
        network = network_cls(**network_kwargs)
        return network.to(device=device, dtype=storage_dtype)

    with tempfile.TemporaryDirectory(prefix="wan_moe_routing_") as root:
        model_dir = _write_tiny_wan22_i2v_model_dir(root)
        network, reference = build(), build()
    for module in network.modules():
        if isinstance(module, FastGenNetwork):
            module._record_precision(precision_model=compute_dtype)
    reference._set_precision(precision_model=compute_dtype)
    reference.train()
    return network, reference


def _shard_tiny_wan_moe(network, world_size, compute_dtype, reduce_dtype):
    mesh = init_device_mesh("cuda", (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=compute_dtype, reduce_dtype=reduce_dtype, output_dtype=None, cast_forward_inputs=False
    )
    network.fully_shard(mesh=mesh, mp_policy=mp_policy)
    network.train()


def _tiny_wan_moe_rank_consistent_batches(network, device, dtype, rank, world_size, both_experts):
    """Return unique local data with identical per-expert route counts on every rank.

    ``both_experts=False`` gives each rank one high-noise sample so the low expert
    receives a rank-symmetric empty batch. ``both_experts=True`` gives each rank
    one high- and one low-noise sample. The union batch stores those rank blocks
    in order so callers can evaluate either the union or the original rank slices.
    """
    generator = torch.Generator(device=device).manual_seed(1234)
    samples_per_rank = 2 if both_experts else 1
    num_samples, frames, height, width = samples_per_rank * world_size, 2, 8, 8
    x_t = torch.randn(num_samples, 16, frames, height, width, device=device, dtype=dtype, generator=generator)
    first_frame = torch.randn(num_samples, 16, frames, height, width, device=device, dtype=dtype, generator=generator)
    text = torch.randn(
        num_samples, 16, _WAN_MOE_TINY_TRANSFORMER_CONFIG["text_dim"], device=device, dtype=dtype, generator=generator
    ).to(dtype=network.precision_model)
    if both_experts:
        t = torch.tensor(
            [0.95, 0.30] * world_size,
            device=device,
            dtype=network.noise_scheduler.t_precision,
        )
        assert (t[0::2] >= network.guidance_boundary).all()
        assert (t[1::2] < network.guidance_boundary).all()
        local_indices = [2 * rank, 2 * rank + 1]
    else:
        t = torch.full(
            (world_size,),
            0.95,
            device=device,
            dtype=network.noise_scheduler.t_precision,
        )
        assert (t >= network.guidance_boundary).all()
        local_indices = [rank]

    def batch(indices):
        return x_t[indices], t[indices], {"text_embeds": text[indices], "first_frame_cond": first_frame[indices]}

    return batch(local_indices), batch(list(range(num_samples)))


def _tiny_wan_moe_unequal_route_batches(network, device, dtype, rank, world_size, route_pattern):
    """Return equal-size rank-local batches with unequal expert-local counts.

    ``1_to_2`` gives rank 0 ``(high, low)=(1, 2)`` and rank 1 ``(2, 1)``.
    ``0_to_2`` gives rank 0 ``(0, 2)`` and rank 1 ``(2, 0)``. Keeping the
    total batch size equal on both ranks makes the average of the two local mean
    losses equal to the global sample mean while retaining each rank's local
    batch composition in the non-sharded reference.
    """
    if world_size != 2:
        raise ValueError(f"unequal-route test requires world_size=2, got {world_size}")
    layouts = {
        "1_to_2": ((1, 2), (2, 1)),
        "0_to_2": ((0, 2), (2, 0)),
    }
    if route_pattern not in layouts:
        raise ValueError(f"unknown route pattern {route_pattern!r}; expected one of {sorted(layouts)}")

    rank_layouts = layouts[route_pattern]
    samples_per_rank = sum(rank_layouts[0])
    assert all(sum(counts) == samples_per_rank for counts in rank_layouts)
    num_samples, frames, height, width = samples_per_rank * world_size, 2, 8, 8
    generator = torch.Generator(device=device).manual_seed(5678)
    x_t = torch.randn(num_samples, 16, frames, height, width, device=device, dtype=dtype, generator=generator)
    first_frame = torch.randn(num_samples, 16, frames, height, width, device=device, dtype=dtype, generator=generator)
    text = torch.randn(
        num_samples,
        16,
        _WAN_MOE_TINY_TRANSFORMER_CONFIG["text_dim"],
        device=device,
        dtype=dtype,
        generator=generator,
    ).to(dtype=network.precision_model)
    t_values = []
    for high_count, low_count in rank_layouts:
        t_values.extend([0.95] * high_count)
        t_values.extend([0.30] * low_count)
    t = torch.tensor(t_values, device=device, dtype=network.noise_scheduler.t_precision)
    assert int((t >= network.guidance_boundary).sum()) == sum(counts[0] for counts in rank_layouts)
    assert int((t < network.guidance_boundary).sum()) == sum(counts[1] for counts in rank_layouts)

    def batch(indices):
        return x_t[indices], t[indices], {"text_embeds": text[indices], "first_frame_cond": first_frame[indices]}

    start = rank * samples_per_rank
    local_indices = list(range(start, start + samples_per_rank))
    return batch(local_indices), batch(list(range(num_samples))), rank_layouts


def _full_grads(network):
    """Gathered (unsharded) gradients keyed by the reference parameter name; ``None`` for missing grads."""
    grads = {}
    for name, param in network.named_parameters():
        grad = param.grad
        if grad is not None and hasattr(grad, "full_tensor"):
            grad = grad.full_tensor()
        grads[name.replace("._checkpoint_wrapped_module", "")] = grad
    return grads


def generate_tiny_wan22_i2v_inputs(network, device, dtype):
    """Inputs for the 3-block Wan 2.2 I2V look-alike (concat-mask conditioning, no CLIP encoder)."""
    batch_size, frames, height, width = 2, 2, 8, 8
    x = torch.randn(batch_size, 16, frames, height, width, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device)
    text_embeds = torch.randn(
        batch_size, 16, _WAN_MOE_TINY_TRANSFORMER_CONFIG["text_dim"], device=device, dtype=network.precision_model
    )
    condition = {"text_embeds": text_embeds, "first_frame_cond": torch.randn_like(x)}
    return {"x_t": x, "t": t, "condition": condition}


def _test_tiny_wan22_i2v_impl(rank: int, world_size: int) -> Dict:
    """Checkpoint-free run of the generic Wan FSDP contract (meta-init, broadcast, compute dtypes)."""
    from fastgen.networks.WanI2V.network import WanI2V

    with tempfile.TemporaryDirectory(prefix=f"tiny_wan22_i2v_rank{rank}_") as root:
        config = L(WanI2V)(
            model_id_or_local_path=_write_tiny_wan22_i2v_model_dir(root),
            load_pretrained=False,
            enable_logvar_linear=False,
            r_timestep=False,
            keep_original_prec=True,
            sdpa_no_cudnn_blocks=[1],
        )
        return _generic_fsdp_test_impl(
            world_size,
            config,
            generate_tiny_wan22_i2v_inputs,
            apply_checkpointing=True,
            extra_result_fn=check_wan_precision,
        )


@RunIf(min_gpus=2)
def test_fsdp_tiny_wan22_i2v():
    """Wan's FSDP contract on a 3-block Wan 2.2 I2V look-alike built from a local config.

    Runs the same contract as the checkpoint-backed Wan tests (meta-init on non-zero ranks, rank-0
    broadcast, activation checkpointing, rank-consistent forward, finite backward) and asserts that every
    parameter computes in bfloat16 while the time path and the output projection compute in float32, and
    that sharding and master dtypes are right. Needs no checkpoint.
    """
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_tiny_wan22_i2v_impl,
        world_size=2,
        timeout=600,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    _assert_wan_precision_result(result)
    clear_gpu_memory()


def _test_tiny_wan_i2v_reused_checkpoint_impl(rank: int, world_size: int) -> Dict:
    """Two differentiable forwards of the tiny Wan I2V before one backward (the AnyFlow rollout pattern).

    Each block's FSDP group wraps its checkpoint wrapper, so every checkpoint replay runs after that use's
    pre-backward unshard. Compared in float64 against a non-sharded copy fed the same two inputs.
    """
    from fastgen.networks.WanI2V.network import WanI2V

    device = torch.device("cuda", torch.cuda.current_device())
    network, reference = _build_tiny_wan_moe_pair(
        device, torch.float64, torch.float64, sdpa_no_cudnn_blocks=None, network_cls=WanI2V
    )
    _shard_tiny_wan_moe(network, world_size, torch.float64, torch.float64)

    generator = torch.Generator(device=device).manual_seed(77)
    segments = []
    for t_value in (0.7, 0.3):
        x_t = torch.randn(1, 16, 2, 8, 8, device=device, dtype=torch.float64, generator=generator)
        first_frame = torch.randn(1, 16, 2, 8, 8, device=device, dtype=torch.float64, generator=generator)
        text = torch.randn(
            1, 16, _WAN_MOE_TINY_TRANSFORMER_CONFIG["text_dim"], device=device, dtype=torch.float64, generator=generator
        )
        t = torch.full((1,), t_value, device=device, dtype=network.noise_scheduler.t_precision)
        segments.append((x_t, t, {"text_embeds": text, "first_frame_cond": first_frame}))

    def two_segment_loss(model):
        return sum(model(x_t, t, condition=condition).double().mean() for x_t, t, condition in segments)

    failure = None
    try:
        two_segment_loss(network).backward()
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001 - reported to the parent process
        failure = f"{type(error).__name__}: {error}"
    two_segment_loss(reference).backward()

    params_without_grad, max_rel_grad_err = [], 0.0
    if failure is None:
        reference_grads = {name: param.grad for name, param in reference.named_parameters()}
        for name, grad in _full_grads(network).items():
            if grad is None:
                params_without_grad.append(name)
                continue
            ref = reference_grads[name].double()
            max_rel_grad_err = max(
                max_rel_grad_err, ((grad.double() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            )
    result = {
        "failure": failure,
        "params_without_grad": params_without_grad,
        "max_rel_grad_err": max_rel_grad_err,
    }
    dist.barrier()
    return result


@RunIf(min_gpus=2)
def test_fsdp_tiny_wan_i2v_reused_activation_checkpoint():
    """Two forwards before one backward on Wan's checkpointed FSDP layout match the non-sharded float64 reference."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_tiny_wan_i2v_reused_checkpoint_impl,
        world_size=2,
        timeout=600,
        nccl_timeout=180,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["failure"] is None, result["failure"]
    assert not result["params_without_grad"], f"parameters without gradient: {result['params_without_grad']}"
    assert (
        result["max_rel_grad_err"] < 1e-6
    ), f"two-segment gradients deviate from the reference: {result['max_rel_grad_err']:.3e}"
    clear_gpu_memory()


def _test_wan_moe_rank_consistent_empty_expert_impl(rank: int, world_size: int) -> Dict:
    """Every rank routes one unique sample high, leaving the low expert empty everywhere.

    This rank-consistent batch-size-1 case records FSDP2's collective enqueue order per rank and checks
    that every rank issues the same all-gather/reduce-scatter sequence, with one reduce-scatter per
    parameter group (one group per block plus the root of each expert), that every parameter receives a
    gradient on every rank, and that the averaged gradients match a non-sharded reference on the union
    batch. A detached batch-0 attention output would make the empty expert's block post-backward hooks
    unreachable and defer its reduce-scatters to FSDP2's end-of-backward callback.
    """
    import torch.distributed.fsdp._fully_shard._fsdp_param_group as fsdp_param_group
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    device = torch.device("cuda", torch.cuda.current_device())
    network, reference = _build_tiny_wan_moe_pair(device, torch.float32, torch.bfloat16, sdpa_no_cudnn_blocks=[1])

    # Record the host-side enqueue order of FSDP2 collectives: (op, first parameter of the group, numel).
    sequence = []
    orig_all_gather, orig_reduce = fsdp_param_group.foreach_all_gather, fsdp_param_group.foreach_reduce

    def logged_all_gather(fsdp_params, *args, **kwargs):
        numel = sum(param.sharded_param.numel() for param in fsdp_params) * world_size
        sequence.append(("AG", fsdp_params[0]._param_fqn, numel))
        return orig_all_gather(fsdp_params, *args, **kwargs)

    def logged_reduce(fsdp_params, unsharded_grads, *args, **kwargs):
        sequence.append(("RS", fsdp_params[0]._param_fqn, sum(grad.numel() for grad in unsharded_grads)))
        return orig_reduce(fsdp_params, unsharded_grads, *args, **kwargs)

    fsdp_param_group.foreach_all_gather, fsdp_param_group.foreach_reduce = logged_all_gather, logged_reduce
    try:
        _shard_tiny_wan_moe(network, world_size, torch.bfloat16, torch.float32)
        param_groups = {
            id(state._fsdp_param_group)
            for module in network.modules()
            if (state := _get_module_fsdp_state(module)) is not None and state._fsdp_param_group is not None
        }
        (x_local, t_local, cond_local), (x_union, t_union, cond_union) = _tiny_wan_moe_rank_consistent_batches(
            network,
            device,
            torch.float32,
            rank,
            world_size,
            both_experts=False,
        )
        network(x_local, t_local, condition=cond_local).float().mean().backward()
        torch.cuda.synchronize()
        reference(x_union, t_union, condition=cond_union).float().mean().backward()
    finally:
        fsdp_param_group.foreach_all_gather, fsdp_param_group.foreach_reduce = orig_all_gather, orig_reduce

    sequences = [None] * world_size
    dist.all_gather_object(sequences, sequence)
    first_divergence = next(
        (
            (index, [seq[index] if index < len(seq) else None for seq in sequences])
            for index in range(max(len(seq) for seq in sequences))
            if len({seq[index] if index < len(seq) else None for seq in sequences}) > 1
        ),
        None,
    )

    reference_grads = {name: param.grad for name, param in reference.named_parameters()}
    params_without_grad, max_rel_grad_err = [], 0.0
    for name, grad in _full_grads(network).items():
        if grad is None:
            params_without_grad.append(name)
            continue
        if not torch.isfinite(grad).all():
            params_without_grad.append(f"{name} (non-finite)")
        ref = reference_grads[name].float()
        max_rel_grad_err = max(max_rel_grad_err, ((grad.float() - ref).norm() / ref.norm().clamp_min(1e-6)).item())

    result = {
        "sequences_identical": first_divergence is None,
        "first_divergence": first_divergence,
        "num_reduce_scatters": sum(1 for op, _, _ in sequence if op == "RS"),
        "num_param_groups": len(param_groups),
        "params_without_grad": params_without_grad,
        "max_rel_grad_err": max_rel_grad_err,
    }
    dist.barrier()
    return result


def _test_wan_moe_unequal_route_counts_impl(
    rank: int,
    world_size: int,
    route_pattern: str,
    precision_mode: str,
) -> Dict:
    """Exercise unequal expert-local batch counts through the production forward path.

    The test verifies WanMoE/FSDP against a non-sharded per-rank-batch reference.
    It checks actual expert input sizes, collective enqueue order, gradient
    presence/finiteness, and whole-model plus per-expert gradient error.
    """
    import torch.distributed.fsdp._fully_shard._fsdp_param_group as fsdp_param_group
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    device = torch.device("cuda", torch.cuda.current_device())
    if precision_mode == "bf16":
        storage_dtype, compute_dtype, reduce_dtype, loss_dtype = (
            torch.float32,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
        )
        sdpa_no_cudnn_blocks = [1]
    elif precision_mode == "float64":
        storage_dtype = compute_dtype = reduce_dtype = loss_dtype = torch.float64
        sdpa_no_cudnn_blocks = None
    else:
        raise ValueError(f"unknown precision mode {precision_mode!r}")

    network, reference = _build_tiny_wan_moe_pair(
        device,
        storage_dtype,
        compute_dtype,
        sdpa_no_cudnn_blocks=sdpa_no_cudnn_blocks,
    )
    _shard_tiny_wan_moe(network, world_size, compute_dtype, reduce_dtype)
    expert_roots_are_fsdp = (
        isinstance(network.network_high.transformer, FSDPModule),
        isinstance(network.network_low.transformer, FSDPModule),
    )
    (x_local, t_local, cond_local), (x_union, t_union, cond_union), expected_counts = (
        _tiny_wan_moe_unequal_route_batches(
            network,
            device,
            storage_dtype,
            rank,
            world_size,
            route_pattern=route_pattern,
        )
    )

    sequence = []
    orig_all_gather, orig_reduce = fsdp_param_group.foreach_all_gather, fsdp_param_group.foreach_reduce

    def logged_all_gather(fsdp_params, *args, **kwargs):
        numel = sum(param.sharded_param.numel() for param in fsdp_params) * world_size
        sequence.append(("AG", fsdp_params[0]._param_fqn, numel))
        return orig_all_gather(fsdp_params, *args, **kwargs)

    def logged_reduce(fsdp_params, unsharded_grads, *args, **kwargs):
        sequence.append(("RS", fsdp_params[0]._param_fqn, sum(grad.numel() for grad in unsharded_grads)))
        return orig_reduce(fsdp_params, unsharded_grads, *args, **kwargs)

    observed_counts = {}

    def record_expert_batch(name):
        def hook(_module, args):
            observed_counts[name] = int(args[0].shape[0])

        return hook

    handles = [
        network.network_high.register_forward_pre_hook(record_expert_batch("high")),
        network.network_low.register_forward_pre_hook(record_expert_batch("low")),
    ]
    fsdp_param_group.foreach_all_gather, fsdp_param_group.foreach_reduce = logged_all_gather, logged_reduce
    try:
        param_groups = {
            id(state._fsdp_param_group)
            for module in network.modules()
            if (state := _get_module_fsdp_state(module)) is not None and state._fsdp_param_group is not None
        }
        network(x_local, t_local, condition=cond_local).to(loss_dtype).mean().backward()
        torch.cuda.synchronize()
        samples_per_rank = x_union.shape[0] // world_size
        reference_loss = torch.zeros((), device=device, dtype=loss_dtype)
        for reference_rank in range(world_size):
            start = reference_rank * samples_per_rank
            stop = start + samples_per_rank
            rank_condition = slice_dim(cond_union, start, stop)
            reference_loss = reference_loss + (
                reference(x_union[start:stop], t_union[start:stop], condition=rank_condition).to(loss_dtype).mean()
                / world_size
            )
        reference_loss.backward()
    finally:
        fsdp_param_group.foreach_all_gather, fsdp_param_group.foreach_reduce = orig_all_gather, orig_reduce
        for handle in handles:
            handle.remove()

    sequences = [None] * world_size
    counts_per_rank = [None] * world_size
    dist.all_gather_object(sequences, sequence)
    dist.all_gather_object(counts_per_rank, (observed_counts["high"], observed_counts["low"]))
    first_divergence = next(
        (
            (index, [seq[index] if index < len(seq) else None for seq in sequences])
            for index in range(max(len(seq) for seq in sequences))
            if len({seq[index] if index < len(seq) else None for seq in sequences}) > 1
        ),
        None,
    )

    def normalized_parameter_name(name):
        return name.replace("._checkpoint_wrapped_module", "")

    local_grad_presence = {
        normalized_parameter_name(name): param.grad is not None for name, param in network.named_parameters()
    }
    grad_presence_per_rank = [None] * world_size
    dist.all_gather_object(grad_presence_per_rank, local_grad_presence)
    params_without_grad = sorted(
        name
        for name in local_grad_presence
        if not all(rank_presence.get(name, False) for rank_presence in grad_presence_per_rank)
    )

    reference_grads = {name: param.grad for name, param in reference.named_parameters()}
    non_finite_grads = []
    squared_error = {"all": 0.0, "high": 0.0, "low": 0.0}
    squared_reference = {"all": 0.0, "high": 0.0, "low": 0.0}
    if not params_without_grad:
        for name, grad in _full_grads(network).items():
            if not torch.isfinite(grad).all():
                non_finite_grads.append(name)
                continue
            ref = reference_grads[name]
            if ref is None:
                params_without_grad.append(f"reference:{name}")
                continue
            error_norm_sq = (grad.double() - ref.double()).square().sum().item()
            reference_norm_sq = ref.double().square().sum().item()
            groups = ["all"]
            if name.startswith("network_high."):
                groups.append("high")
            elif name.startswith("network_low."):
                groups.append("low")
            for group in groups:
                squared_error[group] += error_norm_sq
                squared_reference[group] += reference_norm_sq
    relative_grad_error = {
        group: squared_error[group] ** 0.5 / max(squared_reference[group] ** 0.5, 1e-12) for group in squared_error
    }

    result = {
        "route_pattern": route_pattern,
        "precision_mode": precision_mode,
        "expected_counts": expected_counts,
        "counts_per_rank": counts_per_rank,
        "expert_roots_are_fsdp": expert_roots_are_fsdp,
        "sequences_identical": first_divergence is None,
        "first_divergence": first_divergence,
        "num_reduce_scatters": sum(1 for op, _, _ in sequence if op == "RS"),
        "num_param_groups": len(param_groups),
        "params_without_grad": params_without_grad,
        "non_finite_grads": non_finite_grads,
        "reference_grad_norm": {group: norm_sq**0.5 for group, norm_sq in squared_reference.items()},
        "relative_grad_error": relative_grad_error,
    }
    dist.barrier()
    return result


def _test_wan_moe_rank_consistent_routing_gradcheck_impl(rank: int, world_size: int) -> Dict:
    """Directional finite differences for FSDP gradients under balanced, rank-consistent routing.

    The tiny MoE runs in float64 apart from Wan's FP32 rotary tables, so central differences are accurate
    to at least float32 round-off. Each rank
    compares ``<g_fsdp, d>`` with ``(L(theta + eps d) - L(theta - eps d)) / (2 eps)``, where ``L`` averages
    rank-sized forwards on a non-sharded copy and ``g_fsdp`` is the gathered FSDP gradient. Preserving the
    rank-local batch shape matters because GEMMs are not bitwise batch-composition invariant. Every rank receives one high- and one low-noise sample, so neither expert is empty and both
    expert directions remain nonzero. Directions cover all parameters, each expert separately, and random.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    network, reference = _build_tiny_wan_moe_pair(device, torch.float64, torch.float64, sdpa_no_cudnn_blocks=None)
    _shard_tiny_wan_moe(network, world_size, torch.float64, torch.float64)
    (x_local, t_local, cond_local), (x_union, t_union, cond_union) = _tiny_wan_moe_rank_consistent_batches(
        network,
        device,
        torch.float64,
        rank,
        world_size,
        both_experts=True,
    )

    network(x_local, t_local, condition=cond_local).double().mean().backward()
    torch.cuda.synchronize()
    fsdp_grads = _full_grads(network)
    params = dict(reference.named_parameters())
    baseline_params = {name: param.detach().clone() for name, param in params.items()}
    params_without_grad = [name for name, grad in fsdp_grads.items() if grad is None]
    fsdp_grads = {name: grad.double() for name, grad in fsdp_grads.items() if grad is not None}

    def rank_composed_loss() -> float:
        with torch.no_grad():
            samples_per_rank = x_union.shape[0] // world_size
            loss = torch.zeros((), device=device, dtype=torch.float64)
            for reference_rank in range(world_size):
                start = reference_rank * samples_per_rank
                stop = start + samples_per_rank
                rank_condition = slice_dim(cond_union, start, stop)
                loss = (
                    loss
                    + reference(
                        x_union[start:stop],
                        t_union[start:stop],
                        condition=rank_condition,
                    )
                    .double()
                    .mean()
                    / world_size
                )
            return loss.item()

    def restore_baseline():
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(baseline_params[name])

    def set_perturbation(direction, scale):
        # Construct both finite-difference points independently from exact clones, so round-off from
        # incremental +/- updates cannot bias later directions.
        restore_baseline()
        with torch.no_grad():
            for name, d in direction.items():
                params[name].copy_((baseline_params[name].double() + scale * d).to(params[name].dtype))

    def central_difference(direction, eps):
        try:
            set_perturbation(direction, eps)
            loss_plus = rank_composed_loss()
            set_perturbation(direction, -eps)
            loss_minus = rank_composed_loss()
            return (loss_plus - loss_minus) / (2 * eps)
        finally:
            restore_baseline()

    def unit(vectors):
        norm = torch.sqrt(sum((v**2).sum() for v in vectors.values()))
        return {name: v / norm for name, v in vectors.items()}

    generator = torch.Generator(device=device).manual_seed(4321)
    directions = {
        "grad_all": unit(fsdp_grads),
        "grad_high": unit({n: g for n, g in fsdp_grads.items() if n.startswith("network_high.")}),
        "grad_low": unit({n: g for n, g in fsdp_grads.items() if n.startswith("network_low.")}),
        "random": unit(
            {
                n: torch.randn(g.shape, device=device, dtype=torch.float64, generator=generator)
                for n, g in fsdp_grads.items()
            }
        ),
    }
    checks = {}
    for label, direction in directions.items():
        analytic = sum((fsdp_grads[name] * d).sum() for name, d in direction.items()).item()
        numerical = central_difference(direction, eps=1e-3)
        checks[label] = {
            "analytic": analytic,
            "numerical": numerical,
            "rel_err": abs(analytic - numerical) / max(abs(numerical), 1e-12),
        }

    per_rank = [None] * world_size
    dist.all_gather_object(per_rank, {"checks": checks, "params_without_grad": params_without_grad})
    dist.barrier()
    return {"per_rank": per_rank}


@RunIf(min_gpus=2)
def test_fsdp_wan_moe_rank_consistent_empty_expert_routing():
    """Wan MoE under FSDP2 with the same empty expert on every rank.

    Uses a 3-block Wan 2.2 I2V A14B look-alike built from a local config, so it needs no checkpoint.
    The supported batch-size-1 route exercises the empty-batch graph and requires one collective per
    parameter group in identical order on every rank.
    """
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan_moe_rank_consistent_empty_expert_impl,
        world_size=2,
        timeout=600,
        nccl_timeout=180,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["sequences_identical"], f"collective order differs across ranks at {result['first_divergence']}"
    assert (
        result["num_reduce_scatters"] == result["num_param_groups"]
    ), f"{result['num_reduce_scatters']} reduce-scatters for {result['num_param_groups']} parameter groups"
    assert not result["params_without_grad"], f"parameters without gradient: {result['params_without_grad']}"
    assert result["max_rel_grad_err"] < 0.05, f"gradients deviate from the reference: {result['max_rel_grad_err']:.4f}"
    clear_gpu_memory()


@RunIf(min_gpus=2)
@pytest.mark.parametrize("route_pattern", ["1_to_2", "0_to_2"], ids=["1-to-2", "0-to-2"])
@pytest.mark.parametrize("precision_mode", ["bf16", "float64"], ids=["bf16", "float64"])
def test_fsdp_wan_moe_unequal_expert_batch_sizes(route_pattern, precision_mode):
    """Unequal per-rank expert batches preserve FSDP order and per-rank-objective gradients.

    This calls the validator-free production WanMoE forward path directly,
    without replacing routing or either expert.
    """
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan_moe_unequal_route_counts_impl,
        world_size=2,
        timeout=600,
        nccl_timeout=180,
        setup_fn=set_env_vars,
        route_pattern=route_pattern,
        precision_mode=precision_mode,
    )
    assert result is not None, "Test did not return a result"
    assert result["counts_per_rank"] == list(result["expected_counts"]), result
    assert all(result["expert_roots_are_fsdp"]), "both expert transformer roots must be FSDP-wrapped"
    assert result["sequences_identical"], f"collective order differs across ranks at {result['first_divergence']}"
    assert result["num_param_groups"] > 0, "Wan MoE fixture did not create any FSDP parameter groups"
    assert (
        result["num_reduce_scatters"] == result["num_param_groups"]
    ), f"{result['num_reduce_scatters']} reduce-scatters for {result['num_param_groups']} parameter groups"
    assert not result["params_without_grad"], f"parameters without gradient: {result['params_without_grad']}"
    assert not result["non_finite_grads"], f"non-finite gradients: {result['non_finite_grads']}"
    assert result["reference_grad_norm"]["high"] > 0, "high-expert reference gradient is zero"
    assert result["reference_grad_norm"]["low"] > 0, "low-expert reference gradient is zero"
    # Only Wan's FP32 rotary tables sit outside FP64 here; the 2e-4 bound predates the removal of the
    # sharded/checkpointed FP32 carve-out groups, whose GEMMs set a ~1e-4 floor against this reference.
    tolerance = 0.01 if precision_mode == "bf16" else 2e-4
    assert all(
        error < tolerance for error in result["relative_grad_error"].values()
    ), f"FSDP gradients deviate from the per-rank-batch reference: {result['relative_grad_error']}"
    clear_gpu_memory()


def _test_wan_moe_gradient_accumulation_impl(rank: int, world_size: int) -> Dict:
    """Two accumulated microbatches under ``fsdp_sync_grad`` with routing that flips between microbatches.

    Every rank routes its first unique sample high and its second unique sample low, so route counts
    remain shard-group consistent while one expert receives an empty batch per microbatch. The first
    microbatch must issue no reduce-scatter while ``fsdp_sync_grad`` toggles
    ``reduce_grads``/``is_last_backward`` on every FSDP module (frozen or not), the last microbatch must
    reduce-scatter once per parameter group, and the accumulated FSDP gradients must equal a
    non-sharded FP64 reference composed from the same rank-local microbatches with the trainer's
    ``1 / grad_accum_rounds`` scaling. Exercises Wan's FSDP layout, activation checkpointing, the connected
    batch-0 attention and ``fsdp_sync_grad`` together.
    """
    from types import SimpleNamespace

    import torch.distributed.fsdp._fully_shard._fsdp_param_group as fsdp_param_group
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    from fastgen.utils.distributed import fsdp as fsdp_utils

    device = torch.device("cuda", torch.cuda.current_device())
    network, reference = _build_tiny_wan_moe_pair(device, torch.float64, torch.float64, sdpa_no_cudnn_blocks=None)
    _shard_tiny_wan_moe(network, world_size, torch.float64, torch.float64)
    model = SimpleNamespace(module_dict={"net": network})
    fsdp_states = [state for module in network.modules() if (state := _get_module_fsdp_state(module)) is not None]
    param_groups = {id(state._fsdp_param_group) for state in fsdp_states if state._fsdp_param_group is not None}

    reduce_scatters = []
    orig_reduce = fsdp_param_group.foreach_reduce

    def logged_reduce(fsdp_params, unsharded_grads, *args, **kwargs):
        reduce_scatters.append(fsdp_params[0]._param_fqn)
        return orig_reduce(fsdp_params, unsharded_grads, *args, **kwargs)

    # One unique high-noise sample and one unique low-noise sample per rank.
    generator = torch.Generator(device=device).manual_seed(2024)
    num_samples, frames, height, width = 2 * world_size, 2, 8, 8
    x_t = torch.randn(num_samples, 16, frames, height, width, device=device, dtype=torch.float64, generator=generator)
    first_frame = torch.randn(
        num_samples, 16, frames, height, width, device=device, dtype=torch.float64, generator=generator
    )
    text = torch.randn(
        num_samples,
        16,
        _WAN_MOE_TINY_TRANSFORMER_CONFIG["text_dim"],
        device=device,
        dtype=torch.float64,
        generator=generator,
    )
    t = torch.cat(
        (
            torch.full((world_size,), 0.95, device=device),
            torch.full((world_size,), 0.30, device=device),
        )
    ).to(dtype=network.noise_scheduler.t_precision)
    assert (t[:world_size] >= network.guidance_boundary).all()
    assert (t[world_size:] < network.guidance_boundary).all()

    def batch(indices):
        return x_t[indices], t[indices], {"text_embeds": text[indices], "first_frame_cond": first_frame[indices]}

    grad_accum_rounds = 2
    # All ranks use the same expert in a microbatch, then flip experts together.
    local_samples = [[rank], [world_size + rank]]
    flags_in_first_microbatch = {}
    fsdp_param_group.foreach_reduce = logged_reduce
    try:
        for microbatch, indices in enumerate(local_samples):
            sync = microbatch == grad_accum_rounds - 1
            with fsdp_utils.fsdp_sync_grad(model, sync):
                x_local, t_local, cond_local = batch(indices)
                out = network(x_local, t_local, condition=cond_local)
                if not sync:
                    flags_in_first_microbatch = {
                        "reduce_grads": [
                            state._fsdp_param_group.reduce_grads
                            for state in fsdp_states
                            if state._fsdp_param_group is not None
                        ],
                        "is_last_backward": [state._state_ctx.is_last_backward for state in fsdp_states],
                    }
                (out.double().mean() / grad_accum_rounds).backward()
                torch.cuda.synchronize()
                reduce_scatters.append(("microbatch_end", microbatch))
    finally:
        fsdp_param_group.foreach_reduce = orig_reduce
    boundary = reduce_scatters.index(("microbatch_end", 0))
    first_microbatch_reduce_scatters = boundary
    last_microbatch_reduce_scatters = len(reduce_scatters) - boundary - 2

    # Reference: average rank-sized forwards for each microbatch, then apply the trainer's
    # 1/grad_accum_rounds scaling. Concatenating rank samples into one expert forward changes the kernel
    # shape, and GEMMs round differently per shape even though the objective is algebraically identical,
    # which would measure batch composition instead of gradient accumulation.
    for microbatch_samples in (list(range(world_size)), list(range(world_size, 2 * world_size))):
        samples_per_rank = len(microbatch_samples) // world_size
        reference_loss = torch.zeros((), device=device, dtype=torch.float64)
        for reference_rank in range(world_size):
            start = reference_rank * samples_per_rank
            rank_samples = microbatch_samples[start : start + samples_per_rank]
            x_rank, t_rank, cond_rank = batch(rank_samples)
            reference_loss = (
                reference_loss + reference(x_rank, t_rank, condition=cond_rank).double().mean() / world_size
            )
        (reference_loss / grad_accum_rounds).backward()
    reference_params = dict(reference.named_parameters())
    params_without_grad, per_param = [], []
    for name, grad in _full_grads(network).items():
        if grad is None:
            params_without_grad.append(name)
            continue
        ref = reference_params[name].grad.double()
        per_param.append(
            {
                "name": name,
                "dtype": str(reference_params[name].dtype),
                "abs_err": (grad.double() - ref).norm().item(),
                "ref_norm": ref.norm().item(),
            }
        )
    global_ref_norm = float(sum(entry["ref_norm"] ** 2 for entry in per_param) ** 0.5)
    global_rel_err = float(sum(entry["abs_err"] ** 2 for entry in per_param) ** 0.5) / max(global_ref_norm, 1e-12)
    for entry in per_param:
        entry["rel_err"] = entry["abs_err"] / max(entry["ref_norm"], 1e-12)
    worst = sorted(per_param, key=lambda entry: entry["abs_err"], reverse=True)[:5]

    result = {
        "first_microbatch_reduce_scatters": first_microbatch_reduce_scatters,
        "last_microbatch_reduce_scatters": last_microbatch_reduce_scatters,
        "num_param_groups": len(param_groups),
        "all_reduce_grads_off_in_first_microbatch": not any(flags_in_first_microbatch["reduce_grads"]),
        "all_is_last_backward_off_in_first_microbatch": not any(flags_in_first_microbatch["is_last_backward"]),
        "params_without_grad": params_without_grad,
        "global_rel_grad_err": global_rel_err,
        "global_ref_norm": global_ref_norm,
        "worst_params": worst,
    }
    dist.barrier()
    return result


@RunIf(min_gpus=2)
def test_fsdp_wan_moe_gradient_accumulation_rank_consistent_routing():
    """Gradient accumulation with ``fsdp_sync_grad`` while all ranks flip experts together.

    Pins the interaction of ``fsdp_sync_grad`` (flags toggled on every FSDP module) with Wan's FSDP
    layout, activation checkpointing and the connected batch-0 attention: no reduce-scatter before the
    last microbatch, one per parameter group on the last, and gradients equal to a rank-composed
    non-sharded FP64 reference with the same per-expert batch shapes.
    """
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan_moe_gradient_accumulation_impl,
        world_size=2,
        timeout=600,
        nccl_timeout=180,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    # Without this the flag lists and the reduce-scatter counts below are all empty or zero, and
    # every assertion that follows would pass without observing a single parameter group.
    assert result["num_param_groups"] > 0, result
    assert result["all_reduce_grads_off_in_first_microbatch"], "fsdp_sync_grad left reduce_grads on"
    assert result["all_is_last_backward_off_in_first_microbatch"], "fsdp_sync_grad left is_last_backward on"
    assert result["first_microbatch_reduce_scatters"] == 0, result
    assert result["last_microbatch_reduce_scatters"] == result["num_param_groups"], result
    assert not result["params_without_grad"], f"parameters without gradient: {result['params_without_grad']}"
    # FP64 compute apart from Wan's FP32 rotary tables: accumulation agrees to measured 1e-8 round-off;
    # a lost microbatch, a missing 1/grad_accum_rounds or a cross-mixed expert would be O(1).
    assert result["global_rel_grad_err"] < 1e-6, f"accumulated gradients deviate from the reference: {result}"
    abs_tolerance = 1e-6 * result["global_ref_norm"]
    assert all(
        entry["abs_err"] <= abs_tolerance for entry in result["worst_params"]
    ), f"parameter gradients deviate from the reference: {result['worst_params']}"


@RunIf(min_gpus=2)
def test_fsdp_wan_moe_rank_consistent_routing_gradcheck():
    """Numerical gradient check with one high- and one low-noise sample on every rank.

    Central differences of the average of rank-sized reference forwards along the gradient directions
    (all parameters, high expert only, low expert only) must reproduce the gathered FSDP gradients on
    each rank; a random direction is checked with a looser tolerance because its derivative is small
    relative to the float32 round-off in the loss.
    """
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan_moe_rank_consistent_routing_gradcheck_impl,
        world_size=2,
        timeout=600,
        nccl_timeout=180,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    # Observed on H100: ~1e-6 along the gradient directions and ~1e-3 along the random direction.
    tolerances = {"grad_all": 1e-3, "grad_high": 1e-3, "grad_low": 1e-3, "random": 5e-2}
    for rank, entry in enumerate(result["per_rank"]):
        assert not entry[
            "params_without_grad"
        ], f"rank {rank}: parameters without gradient: {entry['params_without_grad']}"
        for label, check in entry["checks"].items():
            assert check["rel_err"] < tolerances[label], (
                f"rank {rank}, direction {label}: FSDP <g, d>={check['analytic']:.6f} vs "
                f"finite difference {check['numerical']:.6f} (rel err {check['rel_err']:.3e})"
            )
    analytic = {label: [entry["checks"][label]["analytic"] for entry in result["per_rank"]] for label in tolerances}
    for label, values in analytic.items():
        assert max(values) - min(values) <= 1e-6 * max(
            1.0, abs(values[0])
        ), f"ranks disagree on <g, d> for {label}: {values}"
    clear_gpu_memory()


# =============================================================================
# SD15 U-Net checkpoint conversion (2 GPUs)
# =============================================================================


_TINY_UNET2D_CONFIG = {
    "sample_size": 8,
    "in_channels": 4,
    "out_channels": 4,
    "layers_per_block": 1,
    "block_out_channels": (8, 16),
    "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
    "cross_attention_dim": 8,
    "attention_head_dim": 2,
    "norm_num_groups": 8,
}


def _build_tiny_sd15(device, dtype):
    """``StableDiffusion15`` around a tiny U-Net, built through the real constructor.

    Only the hub lookups are replaced (the U-Net config and SD1.5's scheduler config, which is
    reconstructed locally), so the network keeps its ``classify_forward`` binding and its
    ``logvar_linear`` head exactly as in training.
    """
    from diffusers import DDIMScheduler, UNet2DConditionModel

    from fastgen.networks.SD15.network import StableDiffusion15

    sd15_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(UNet2DConditionModel, "load_config", return_value=dict(_TINY_UNET2D_CONFIG)))
        stack.enter_context(patch.object(DDIMScheduler, "from_pretrained", return_value=sd15_scheduler))
        network = StableDiffusion15(load_pretrained=False)
    return network.to(device=device, dtype=dtype)


def _test_sd15_unet_checkpointing_parity_impl(rank: int, world_size: int) -> Dict:
    """Forward and backward through the sharded, checkpoint-converted SD15 U-Net against an unsharded copy.

    Every rank uses the same batch, so the reduce-scattered FSDP gradients must equal the reference
    gradients. Float64 keeps the comparison at round-off level.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    network = _build_tiny_sd15(device, torch.float64)
    reference = _build_tiny_sd15(device, torch.float64)
    reference.load_state_dict(network.state_dict())

    # The constructor enables diffusers' block-level checkpointing; ``fully_shard`` must convert it.
    assert network.unet.is_gradient_checkpointing
    mesh = init_device_mesh("cuda", (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float64, reduce_dtype=torch.float64, output_dtype=None, cast_forward_inputs=False
    )
    network.fully_shard(mesh=mesh, mp_policy=mp_policy)
    unet = network.unet
    blocks = [*unet.down_blocks, unet.mid_block, *unet.up_blocks]
    all_wrapped = all(isinstance(block, CheckpointWrapper) for block in blocks)
    network.train()
    reference.train()

    generator = torch.Generator(device=device).manual_seed(1234)
    x_t = torch.randn(2, 4, 8, 8, device=device, dtype=torch.float64, generator=generator)
    t = torch.rand(2, device=device, dtype=torch.float64, generator=generator) * 0.8 + 0.1
    prompt_embeddings = torch.randn(2, 5, 8, device=device, dtype=torch.float64, generator=generator)
    prompt_mask = torch.ones(2, 5, device=device, dtype=torch.bool)
    condition = (prompt_embeddings, prompt_mask)

    output = network(x_t, t, condition=condition)
    output.square().mean().backward()
    reference_output = reference(x_t, t, condition=condition)
    reference_output.square().mean().backward()
    torch.cuda.synchronize()

    output_rel_err = ((output - reference_output).norm() / reference_output.norm()).item()
    fsdp_grads = _full_grads(network)
    reference_grads = {name: param.grad for name, param in reference.named_parameters()}
    missing_grads = sorted(
        name for name, grad in reference_grads.items() if grad is not None and fsdp_grads.get(name) is None
    )
    squared_error = 0.0
    squared_reference = 0.0
    worst = ("", 0.0)
    num_compared = 0
    for name, reference_grad in reference_grads.items():
        if reference_grad is None or fsdp_grads.get(name) is None:
            continue
        num_compared += 1
        abs_err = (fsdp_grads[name] - reference_grad).norm().item()
        squared_error += abs_err**2
        squared_reference += reference_grad.norm().item() ** 2
        if abs_err > worst[1]:
            worst = (name, abs_err)
    result = {
        "all_wrapped": all_wrapped,
        "num_blocks": len(blocks),
        "unet_flag_checkpointing_after_forward": unet.is_gradient_checkpointing,
        "output_rel_err": output_rel_err,
        "missing_grads": missing_grads,
        "num_params_compared": num_compared,
        "global_ref_grad_norm": squared_reference**0.5,
        "global_rel_grad_err": squared_error**0.5 / max(squared_reference**0.5, 1e-12),
        "worst_param": worst,
    }
    # Every rank reports; rank asymmetry (a wrapper or gradient missing on one rank only) must fail.
    per_rank = [None] * world_size
    dist.all_gather_object(per_rank, result)
    dist.barrier()
    return {"per_rank": per_rank}


@RunIf(min_gpus=2)
def test_fsdp_sd15_unet_checkpointing_matches_unsharded_reference():
    """``StableDiffusion15.fully_shard`` converts block-level diffusers checkpointing and shards the wrappers.

    The U-Net runs FastGen's ``classify_forward``, which drives the blocks by keyword, reads block
    attributes (``has_cross_attention``, ``resnets``) and passes residual tuples; all of that now goes
    through ``CheckpointWrapper`` instances that are FSDP groups. Outputs and gathered gradients must
    match an unsharded, unconverted copy at float64 round-off.
    """
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_sd15_unet_checkpointing_parity_impl,
        world_size=2,
        timeout=600,
        nccl_timeout=180,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert len(result["per_rank"]) == 2, result
    for rank, entry in enumerate(result["per_rank"]):
        context = f"rank {rank}: {entry}"
        assert entry["all_wrapped"], context
        # A forward through the converted U-Net must not re-enable diffusers' block flags inside the
        # wrappers; that would recompute every block twice.
        assert not entry["unet_flag_checkpointing_after_forward"], context
        assert not entry["missing_grads"], f"rank {rank}: parameters without FSDP gradient: {entry['missing_grads']}"
        # Guard against a vacuous comparison: the tiny U-Net has dozens of trained tensors.
        assert entry["num_params_compared"] > 20 and entry["global_ref_grad_norm"] > 0, context
        assert entry["output_rel_err"] < 1e-10, context
        assert entry["global_rel_grad_err"] < 1e-8, context
        assert entry["worst_param"][1] <= 1e-8 * entry["global_ref_grad_norm"], context
    clear_gpu_memory()


# =============================================================================
# Pytest Test Functions - LTX-2 (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_ltx2():
    """Test FSDP forward+backward pass for LTX-2 T2V with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_ltx2_impl,
        world_size=2,
        timeout=900,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()


@RunIf(min_gpus=4, min_gpu_mem=80)
@pytest.mark.large_model
def test_fsdp_minimax_h3():
    """Test FSDP forward+backward for MiniMax-H3, including its float32 carve-out groups.

    Four ranks rather than two: float32 master weights for 33B parameters are 132 GB, and the
    shards have to leave room for the all-gather buffers and the activations.

    Beyond the usual rank-consistency and gradient checks, this asserts that the modules the
    checkpoint keeps in float32 really compute in float32 under FSDP. They get their own groups
    with a float32 ``param_dtype``, which is what reproduces the release's mixed precision during
    training -- and which only works because ``forward`` returns each prediction in the dtype of
    the latents it was handed; a float32 prediction against bfloat16 data dies in the backward
    with ``Found dtype BFloat16 but expected Float``.
    """
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_minimax_h3_impl,
        world_size=4,
        timeout=1800,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"

    carve_out = result["carve_out_compute_dtype"]
    assert set(carve_out) == set(result["carve_out_sharded"]), "the forward missed a float32 carve-out module"
    assert set(carve_out.values()) == {"float32"}, f"carve-out did not compute in float32: {carve_out}"
    assert all(result["carve_out_sharded"].values()), result["carve_out_sharded"]
    assert all(result["carve_out_has_grads"].values()), result["carve_out_has_grads"]
    clear_gpu_memory()


@RunIf(min_gpus=2)
@pytest.mark.large_model
def test_fsdp_ltx2_i2v():
    """Test FSDP forward+backward pass for LTX-2 I2V with gradient checkpointing."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_ltx2_i2v_impl,
        world_size=2,
        timeout=900,
        setup_fn=set_env_vars,
    )
    assert result is not None, "Test did not return a result"
    assert result["rank_consistent"], f"Ranks not consistent: diff={result['rank_consistency_diff']}"
    assert result[
        "backward_success"
    ], f"Backward failed: has_grads={result['has_grads']}, finite={result['all_grads_finite']}"
    clear_gpu_memory()
