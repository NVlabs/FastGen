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

Tests require at least 2 GPUs (except Wan 14B models which need 8 GPUs).

Usage:
    pytest tests/test_network_fsdp.py -v

    # Run specific network tests:
    pytest tests/test_network_fsdp.py::test_fsdp_edm_cifar10 -v
    pytest tests/test_network_fsdp.py::test_fsdp_wan_1_3b -v
"""

import contextlib
import copy
from typing import Dict
from unittest.mock import patch, MagicMock
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

from fastgen.utils import instantiate
from fastgen.utils.test_utils import RunIf, run_distributed_test
from fastgen.utils.basic_utils import clear_gpu_memory
from fastgen.utils.io_utils import set_env_vars


# =============================================================================
# FSDP Utilities
# =============================================================================


def _get_meta_init_context(fsdp_meta_init: bool):
    """Get context manager for FSDP meta initialization."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    use_meta = fsdp_meta_init and rank != 0
    if use_meta:
        return torch.device("meta")
    return contextlib.nullcontext()


def broadcast_state_dict_and_sync(network, broadcast_state_dict):
    """Materialize meta tensors and broadcast state dict from rank 0."""
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
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
    fsdp_meta_init: bool = True,
    apply_checkpointing: bool = False,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    sharded_dtype: torch.dtype = torch.float32,
):
    """Create a network and apply FSDP sharding.

    Args:
        config: Network configuration
        device_mesh: FSDP device mesh
        fsdp_meta_init: Whether to use meta device initialization
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
    with _get_meta_init_context(fsdp_meta_init):
        network = instantiate(config)

    # Ensure all ranks have finished loading before continuing
    dist.barrier()

    # Set dtype of sharded weights before FSDP wrapping
    network.to(dtype=sharded_dtype)

    # Extract state dict from rank 0 BEFORE FSDP sharding
    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
    else:
        broadcast_state_dict = None

    dist.barrier()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=param_dtype,
        cast_forward_inputs=True,
    )
    network.fully_shard(mesh=device_mesh, mp_policy=mp_policy)

    # Materialize and sync
    broadcast_state_dict_and_sync(network, broadcast_state_dict)

    # Final sync before returning
    dist.barrier()

    return network


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

    Returns:
        Dict with test results
    """
    device_mesh = init_device_mesh("cuda", (world_size,))
    device = torch.device("cuda", torch.cuda.current_device())

    network = create_network_with_fsdp(
        config,
        device_mesh,
        fsdp_meta_init=True,
        apply_checkpointing=apply_checkpointing,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        sharded_dtype=sharded_dtype,
    )
    network.train()

    # Generate inputs
    inputs = generate_inputs_fn(network, device, param_dtype)

    # Forward pass (with gradients)
    output = network(**inputs)

    # Handle tuple outputs
    if isinstance(output, tuple):
        output = output[0]

    # Compute loss and backward
    loss = output.mean()
    loss.backward()

    # Check that gradients exist and are finite
    grad_info = {"has_grads": False, "all_finite": True, "num_params_with_grad": 0}

    for _, param in network.named_parameters():
        if param.grad is not None:
            grad_info["has_grads"] = True
            grad_info["num_params_with_grad"] += 1
            if not torch.isfinite(param.grad).all():
                grad_info["all_finite"] = False

    # Check rank consistency of output
    with torch.no_grad():
        output_detached = output.detach()
    is_consistent, diff = check_rank_consistency(output_detached)

    result = {
        "rank_consistent": is_consistent,
        "rank_consistency_diff": diff,
        "output_shape": tuple(output.shape),
        "has_grads": grad_info["has_grads"],
        "all_grads_finite": grad_info["all_finite"],
        "num_params_with_grad": grad_info["num_params_with_grad"],
        "backward_success": grad_info["has_grads"] and grad_info["all_finite"],
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
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    labels = torch.zeros(batch_size, 10, dtype=dtype, device=device)
    labels[:, 0] = 1.0
    return {"x_t": x, "t": t, "condition": labels}


def generate_edm_imagenet64_inputs(network, device, dtype):
    """Generate inputs for EDM ImageNet64."""
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    labels = torch.zeros(batch_size, 1000, dtype=dtype, device=device)
    labels[:, 0] = 1.0
    return {"x_t": x, "t": t, "condition": labels}


def generate_edm2_inputs(network, device, dtype):
    """Generate inputs for EDM2."""
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    labels = torch.zeros(batch_size, 1000, dtype=dtype, device=device)
    labels[:, 0] = 1.0
    return {"x_t": x, "t": t, "condition": labels}


def generate_dit_inputs(network, device, dtype):
    """Generate inputs for DiT."""
    batch_size = 2
    # DiT operates on latent space
    x = torch.randn(batch_size, 4, 32, 32, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    labels = torch.zeros(batch_size, 1000, dtype=dtype, device=device)
    labels[:, 0] = 1.0
    return {"x_t": x, "t": t, "condition": labels}


def generate_sd15_inputs(network, device, dtype):
    """Generate inputs for Stable Diffusion 1.5."""
    batch_size = 2
    x = torch.randn(batch_size, 4, 8, 8, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    # Use random dummy embeddings instead of text encoder (CLIP ViT-L/14: seq_len=77, hidden_dim=768)
    embeddings = torch.randn(batch_size, 77, 768, device=device, dtype=dtype)
    attention_mask = torch.ones(batch_size, 77, device=device, dtype=dtype)
    return {"x_t": x, "t": t, "condition": (embeddings, attention_mask)}


def generate_cogvideox_inputs(network, device, dtype):
    """Generate inputs for CogVideoX."""
    batch_size = 1
    C, T, H, W = 16, 2, 4, 4
    x = torch.randn(batch_size, C, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    # Use random dummy embeddings instead of text encoder (T5-XXL: seq_len=226, hidden_dim=4096)
    condition = torch.randn(batch_size, 226, 4096, device=device, dtype=dtype)
    return {"x_t": x, "t": t, "condition": condition}


def generate_wan_1_3b_inputs(network, device, dtype):
    """Generate inputs for Wan 1.3B."""
    batch_size = 1
    T, H, W = 2, 4, 4
    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    condition = torch.randn(batch_size, 512, 4096, device=device, dtype=dtype)
    return {"x_t": x, "t": t, "condition": condition}


def generate_causal_wan_1_3b_inputs(network, device, dtype):
    """Generate inputs for CausalWan 1.3B."""
    batch_size = 1
    C, T, H, W = 16, 3, 4, 4
    x = torch.randn(batch_size, C, T, H, W, device=device, dtype=dtype)
    t_1d = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    t = t_1d.unsqueeze(1).expand(batch_size, T)  # 2D timesteps
    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    condition = torch.randn(batch_size, 512, 4096, device=device, dtype=dtype)
    return {"x_t": x, "t": t, "condition": condition}


def generate_wan22_5b_inputs(network, device, dtype):
    """Generate inputs for Wan 2.2 5B."""
    batch_size = 1
    T, H, W = 2, 4, 4
    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    condition = torch.randn(batch_size, 512, 4096, device=device, dtype=dtype)
    return {"x_t": x, "t": t, "condition": condition}


def generate_vace_wan_1_3b_inputs(network, device, dtype):
    """Generate inputs for VACE Wan 1.3B."""
    batch_size = 1
    T_latent, H_latent, W_latent = 2, 4, 4
    x = torch.randn(batch_size, 16, T_latent, H_latent, W_latent, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="logitnormal", device=device, dtype=dtype)
    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=dtype)
    # Use random dummy vid_context instead of prepare_vid_conditioning (shape: [B, 96, T, H, W])
    vid_context = torch.randn(batch_size, 96, T_latent, H_latent, W_latent, device=device, dtype=dtype)
    condition = {"text_embeds": text_embeds, "vid_context": vid_context}
    return {"x_t": x, "t": t, "condition": condition}


def generate_wan21_14b_inputs(network, device, dtype):
    """Generate inputs for Wan 2.1 14B T2V."""
    batch_size = 1
    T, H, W = 2, 4, 4
    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    condition = torch.randn(batch_size, 512, 4096, device=device, dtype=dtype)
    return {"x_t": x, "t": t, "condition": condition}


def generate_wan22_i2v_5b_inputs(network, device, dtype):
    """Generate inputs for Wan 2.2 I2V 5B."""
    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 16, width // 16

    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)

    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=dtype)

    # Use random dummy first_frame_cond instead of VAE encoding (shape: [B, 16, 1, H, W])
    first_frame_cond = torch.randn(batch_size, 48, 1, H, W, device=device, dtype=dtype)

    condition = {"text_embeds": text_embeds, "first_frame_cond": first_frame_cond}
    return {"x_t": x, "t": t, "condition": condition}


def generate_causal_wan22_i2v_5b_inputs(network, device, dtype):
    """Generate inputs for CausalWan 2.2 I2V 5B."""
    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 16, width // 16

    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=dtype)
    t_1d = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    t = t_1d.unsqueeze(1).expand(batch_size, T)

    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=dtype)

    # Use random dummy first_frame_cond instead of VAE encoding (shape: [B, 16, 1, H, W])
    first_frame_cond = torch.randn(batch_size, 48, 1, H, W, device=device, dtype=dtype)

    condition = {"text_embeds": text_embeds, "first_frame_cond": first_frame_cond}
    return {"x_t": x, "t": t, "condition": condition}


def generate_wan21_i2v_14b_inputs(network, device, dtype):
    """Generate inputs for Wan 2.1 I2V 14B."""
    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 8, width // 8

    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)

    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=dtype)

    # Use random dummy image encoder hidden states (CLIP ViT-H/14: seq_len=257, hidden_dim=1280)
    encoder_hidden_states_image = torch.randn(batch_size, 257, 1280, device=device, dtype=dtype)

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
    t_1d = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)
    t = t_1d.unsqueeze(1).expand(batch_size, T)

    # Use random dummy embeddings instead of text encoder (UMT5: seq_len=512, hidden_dim=4096)
    text_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=dtype)

    # Use random dummy image encoder hidden states (CLIP ViT-H/14: seq_len=257, hidden_dim=1280)
    encoder_hidden_states_image = torch.randn(batch_size, 257, 1280, device=device, dtype=dtype)

    # Use random dummy first_frame_cond instead of VAE encoding (shape: [B, 16, T, H, W])
    first_frame_cond = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)

    condition = {
        "text_embeds": text_embeds,
        "first_frame_cond": first_frame_cond,
        "encoder_hidden_states_image": encoder_hidden_states_image,
    }
    return {"x_t": x, "t": t, "condition": condition}


def generate_flux_inputs(network, device, dtype):
    """Generate inputs for Flux."""
    batch_size = 1
    # Flux operates on latent space: [B, C, H, W] where C=16 for Flux VAE
    x = torch.randn(batch_size, 16, 8, 8, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)

    # Use random dummy embeddings instead of text encoder
    # CLIP pooled embeddings: [B, 768]
    pooled_prompt_embeds = torch.randn(batch_size, 768, device=device, dtype=dtype)
    # T5 text embeddings: [B, seq_len, 4096] (T5-XXL hidden dim is 4096)
    prompt_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=dtype)

    condition = (pooled_prompt_embeds, prompt_embeds)

    # Flux uses embedded guidance (guidance distillation)
    guidance = torch.full((batch_size,), 3.5, device=device, dtype=dtype)

    return {"x_t": x, "t": t, "condition": condition, "guidance": guidance}


def generate_qwen_image_inputs(network, device, dtype):
    """Generate inputs for QwenImage."""
    batch_size = 1
    # QwenImage latent: [B, 16, H, W], H/W must be even for 2x2 packing
    x = torch.randn(batch_size, 16, 8, 8, device=device, dtype=dtype)
    t = network.noise_scheduler.sample_t(batch_size, time_dist_type="uniform", device=device, dtype=dtype)

    # Use random dummy embeddings instead of Qwen2.5-VL text encoder (hidden_dim=3584)
    encoder_hidden_states = torch.randn(batch_size, 512, 3584, device=device, dtype=dtype)
    encoder_hidden_states_mask = torch.ones(batch_size, 512, device=device, dtype=torch.long)
    condition = (encoder_hidden_states, encoder_hidden_states_mask)

    return {"x_t": x, "t": t, "condition": condition}


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


def _test_dit_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import DiT_IN256_S_Config

    with patch("diffusers.AutoencoderKL.from_pretrained") as mock_vae:
        mock_vae.return_value = MagicMock()
        return _generic_fsdp_test_impl(world_size, DiT_IN256_S_Config, generate_dit_inputs, apply_checkpointing=False)


def _test_sd15_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import SD15Config

    return _generic_fsdp_test_impl(world_size, SD15Config, generate_sd15_inputs, apply_checkpointing=False)


def _test_cogvideox_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CogVideoXConfig

    set_env_vars()
    # CogVideoX 2B was trained in fp16 and has numerical issues in backward pass
    # Use float32 for stability until the issue is resolved
    return _generic_fsdp_test_impl(
        world_size,
        CogVideoXConfig,
        generate_cogvideox_inputs,
        apply_checkpointing=True,
        param_dtype=torch.float32,
    )


def _test_cogvideox_5b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CogVideoX5BConfig

    set_env_vars()
    # CogVideoX 5B was trained in bf16 and does not require fp32 for numerical stability
    return _generic_fsdp_test_impl(
        world_size,
        CogVideoX5BConfig,
        generate_cogvideox_inputs,
        apply_checkpointing=True,
    )


def _test_wan_1_3b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan_1_3B_Config

    set_env_vars()
    return _generic_fsdp_test_impl(world_size, Wan_1_3B_Config, generate_wan_1_3b_inputs, apply_checkpointing=True)


def _test_causal_wan_1_3b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CausalWan_1_3B_Config

    set_env_vars()
    return _generic_fsdp_test_impl(
        world_size,
        CausalWan_1_3B_Config,
        generate_causal_wan_1_3b_inputs,
        apply_checkpointing=True,
    )


def _test_wan22_5b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan22_T2V_5B_Config

    set_env_vars()
    return _generic_fsdp_test_impl(
        world_size,
        Wan22_T2V_5B_Config,
        generate_wan22_5b_inputs,
        apply_checkpointing=True,
    )


def _test_vace_wan_1_3b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import VACE_Wan_1_3B_Config

    set_env_vars()
    return _generic_fsdp_test_impl(
        world_size,
        VACE_Wan_1_3B_Config,
        generate_vace_wan_1_3b_inputs,
        apply_checkpointing=True,
    )


def _test_wan21_14b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan21_T2V_14B_Config

    set_env_vars()
    return _generic_fsdp_test_impl(
        world_size,
        Wan21_T2V_14B_Config,
        generate_wan21_14b_inputs,
        apply_checkpointing=True,
    )


def _test_wan22_i2v_5b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan22_I2V_5B_Config

    set_env_vars()
    return _generic_fsdp_test_impl(
        world_size,
        Wan22_I2V_5B_Config,
        generate_wan22_i2v_5b_inputs,
        apply_checkpointing=True,
    )


def _test_causal_wan22_i2v_5b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CausalWan22_I2V_5B_Config

    set_env_vars()
    return _generic_fsdp_test_impl(
        world_size,
        CausalWan22_I2V_5B_Config,
        generate_causal_wan22_i2v_5b_inputs,
        apply_checkpointing=True,
    )


def _test_wan21_i2v_14b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import Wan21_I2V_14B_480P_Config

    set_env_vars()
    return _generic_fsdp_test_impl(
        world_size,
        Wan21_I2V_14B_480P_Config,
        generate_wan21_i2v_14b_inputs,
        apply_checkpointing=True,
    )


def _test_causal_wan21_i2v_14b_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import CausalWan21_I2V_14B_480P_Config

    set_env_vars()
    return _generic_fsdp_test_impl(
        world_size,
        CausalWan21_I2V_14B_480P_Config,
        generate_causal_wan21_i2v_14b_inputs,
        apply_checkpointing=True,
    )


def _test_flux_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import FluxConfig

    set_env_vars()
    return _generic_fsdp_test_impl(world_size, FluxConfig, generate_flux_inputs, apply_checkpointing=True)


def _test_qwen_image_impl(rank: int, world_size: int) -> Dict:
    from fastgen.configs.net import QwenImageConfig

    set_env_vars()
    return _generic_fsdp_test_impl(
        rank, world_size, QwenImageConfig, generate_qwen_image_inputs, apply_checkpointing=True
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
# Pytest Test Functions - Flux (2 GPUs)
# =============================================================================


@RunIf(min_gpus=2)
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
    except HTTPError as e:
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
