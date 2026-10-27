# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Modules and layers for Cosmos Predict2 network.

This module contains all the building blocks used by the main network classes:
- Selective activation checkpointing utilities
- Normalization layers (RMSNorm)
- Attention mechanisms
- Positional embeddings (RoPE, learnable)
- Timestep embeddings
- Patch embeddings
- Transformer blocks
"""

import collections
import contextlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from fastgen.utils.tensor import assert_tensor, promote_dtype

# Try to import checkpoint utilities
try:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    HAS_SELECTIVE_CHECKPOINT = True
except ImportError:
    HAS_SELECTIVE_CHECKPOINT = False
    CheckpointPolicy = None

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
    )

    HAS_CHECKPOINT_WRAPPER = True
except ImportError:
    HAS_CHECKPOINT_WRAPPER = False
    ptd_checkpoint_wrapper = None


# ---------------------- Selective Activation Checkpointing -----------------------


class CheckpointMode(str, Enum):
    """Checkpoint modes for selective activation checkpointing."""

    NONE = "none"
    BLOCK_WISE = "block_wise"
    AGGRESSIVE = "aggressive"
    SAVE_FLASH_ATTN = "save_flash_attn"
    RECOMPUTE_ALL = "recompute_all"

    def __str__(self) -> str:
        return self.value


@dataclass
class SACConfig:
    """Configuration for Selective Activation Checkpointing.

    Args:
        mode: The checkpoint mode to use
        every_n_blocks: Apply checkpointing to every N blocks
        checkpoint_final_layer: Whether to checkpoint the final layer
    """

    mode: CheckpointMode = CheckpointMode.NONE
    every_n_blocks: int = 1
    checkpoint_final_layer: bool = True

    def get_context_fn(self) -> Optional[Callable]:
        """Get the context function for the checkpoint mode."""
        if not HAS_SELECTIVE_CHECKPOINT:
            return None

        if self.mode == CheckpointMode.NONE:
            return None
        elif self.mode == CheckpointMode.BLOCK_WISE:
            return block_wise_context_fn
        elif self.mode == CheckpointMode.AGGRESSIVE:
            return aggressive_context_fn
        elif self.mode == CheckpointMode.SAVE_FLASH_ATTN:
            return save_flash_attn_context_fn
        elif self.mode == CheckpointMode.RECOMPUTE_ALL:
            return recompute_all_context_fn
        else:
            return None


def block_wise_context_fn():
    """
    Block-wise checkpointing: saves matmul and attention outputs, recomputes the rest.
    Good balance between memory and compute.
    """
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"

        # Save matmul outputs
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 8:
                return CheckpointPolicy.MUST_SAVE

        # Save flash attention outputs
        if "flash_attn" in str(func) or "scaled_dot_product" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE

        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def aggressive_context_fn():
    """
    Aggressive memory saving: only saves flash attention outputs.
    Maximizes memory savings at the cost of more recomputation.
    """

    def policy_fn(ctx, func, *args, **kwargs):
        # Save the output of attention - most expensive to recompute
        if "flash_attn" in str(func) or "scaled_dot_product" in str(func):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def save_flash_attn_context_fn():
    """
    Save flash attention and linear outputs for self-attention.
    Good for models with many attention layers.
    """
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"

        # Always save matmul outputs
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_SAVE

        # Save every other flash attention
        if "flash_attn" in str(func) or "scaled_dot_product" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE

        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def recompute_all_context_fn():
    """
    Most aggressive: recomputes everything during backward pass.
    Maximum memory savings but highest compute overhead.
    """

    def policy_fn(ctx, func, *args, **kwargs):
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


# ---------------------- Basic Layers -----------------------


class FP32Linear(nn.Linear):
    """A reference FP32 operation over the caller's parameter storage dtype.

    Explicit operand casts apply the same policy on CPU and CUDA, independent of
    an enclosing autocast region. Higher-precision parameters/inputs retain their
    precision, and gradients flow through the casts back to the stored parameters.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dtype = promote_dtype(input, self.weight, torch.float32)
        with torch.amp.autocast(device_type=input.device.type, enabled=False):
            bias = None if self.bias is None else self.bias.to(dtype)
            return F.linear(input.to(dtype), self.weight.to(dtype), bias)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused normalization multiplies by the weight before rounding to BF16/FP16.
        # Rounding the normalized activation first differs from the native TE kernel.
        dtype = promote_dtype(x, self.weight)
        x, weight = x.to(dtype), self.weight.to(dtype)
        return F.rms_norm(x, (weight.numel(),), weight, self.eps)


# ---------------------- Feed Forward Network -----------------------


class GPT2FeedForward(nn.Module):
    """GPT-2 style feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


# ---------------------- Attention Mechanism -----------------------


def apply_rotary_pos_emb_simple(x: torch.Tensor, rope_emb: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor using SPLIT convention.

    Evaluate the split-half rotation in at least FP32 before restoring the input dtype.

    Args:
        x: Input tensor of shape (B, S, H, D) where D is head_dim
        rope_emb: RoPE embeddings of shape (S, 1, 1, D) containing frequency angles
                  The tensor contains [angles, angles] (repeated for first and second half)

    Returns:
        Tensor with rotary embeddings applied
    """
    # rope_emb shape: (S, 1, 1, D) where D = head_dim
    seq_len = x.shape[1]
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    # Reshape rope_emb from (S, 1, 1, D) to (1, S, 1, D) for proper broadcasting
    rope_emb = rope_emb.squeeze(1).squeeze(1)  # (S, D)
    rope_emb = rope_emb.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)

    # The angles are in the first half (and repeated in second half). Like Transformer Engine's fused kernel,
    # compute cos/sin and the rotation in at least FP32 and round to the input dtype once; low-precision angles
    # garble large positions.
    dtype = promote_dtype(x, torch.float32)
    angles = rope_emb[:, :seq_len, :, :half_dim].to(dtype)  # (1, S, 1, D/2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # SPLIT convention: first half and second half of the dimension
    # x1 = x[..., :half_dim], x2 = x[..., half_dim:]
    x1 = x[..., :half_dim].to(dtype)  # (B, S, H, D/2)
    x2 = x[..., half_dim:].to(dtype)  # (B, S, H, D/2)

    # Apply rotation: standard 2D rotation matrix
    # [cos -sin] [x1]   [x1 * cos - x2 * sin]
    # [sin  cos] [x2] = [x1 * sin + x2 * cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Concatenate back: [out1, out2]
    out = torch.cat([out1, out2], dim=-1).to(x.dtype)

    return out


class Attention(nn.Module):
    """
    Multi-head attention with support for self-attention and cross-attention.

    Args:
        query_dim: Dimensionality of query vectors
        context_dim: Dimensionality of context (key/value) vectors. If None, uses self-attention
        n_heads: Number of attention heads
        head_dim: Dimension of each attention head
        dropout: Dropout probability
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in [self.q_norm, self.k_norm, self.v_norm]:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(self, x, context=None, rope_emb=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)

        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        if self.is_selfattn and rope_emb is not None:
            # WAN fp32 strategy, as in the reference; RoPE computes in FP32 and rounds once either way
            original_dtype = q.dtype
            if self.use_wan_fp32_strategy:
                q = q.to(torch.float32)
                k = k.to(torch.float32)
            q = apply_rotary_pos_emb_simple(q, rope_emb)
            k = apply_rotary_pos_emb_simple(k, rope_emb)
            if self.use_wan_fp32_strategy:
                q = q.to(original_dtype)
                k = k.to(original_dtype)

        return q, k, v

    def compute_attention(self, q, k, v):
        # Rearrange for torch SDPA: (B, S, H, D) -> (B, H, S, D)
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        # Ensure all tensors have the same dtype for SDPA
        dtype = q.dtype
        k = k.to(dtype)
        v = v.to(dtype)

        result = F.scaled_dot_product_attention(q, k, v)
        result = rearrange(result, "b h s d -> b s (h d)")
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ):
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v)


# ---------------------- Positional Embeddings -----------------------


class VideoRopePosition3DEmb(nn.Module):
    """3D Rotary Position Embeddings for video (temporal + spatial)."""

    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h

        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq.copy_(
            torch.arange(
                max(self.max_h, self.max_w, self.max_t),
                dtype=torch.float32,
                device=self.seq.device,
            )
        )
        self.dim_spatial_range.copy_(
            torch.arange(0, dim_h, 2, dtype=torch.float32, device=self.dim_spatial_range.device)[: (dim_h // 2)] / dim_h
        )
        self.dim_temporal_range.copy_(
            torch.arange(0, dim_t, 2, dtype=torch.float32, device=self.dim_temporal_range.device)[: (dim_t // 2)]
            / dim_t
        )

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_T_H_W_C = x_B_T_H_W_C.shape
        return self.generate_embeddings(B_T_H_W_C, fps=fps)

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
    ):
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.float())
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.float())
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.float())

        B, T, H, W, _ = B_T_H_W_C

        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            if fps is None:
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                # Reference guard: the first sample's fps positions the whole batch.
                if B > 1 and T > 1:
                    assert_tensor((fps == fps[0]).all(), "non-uniform fps needs batch size 1")
                half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


class LearnablePosEmbAxis(nn.Module):
    """Learnable positional embeddings along temporal and spatial axes."""

    def __init__(
        self,
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        **kwargs,
    ):
        super().__init__()
        self.interpolation = interpolation
        self.model_channels = model_channels

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.model_channels)
        nn.init.trunc_normal_(self.pos_emb_h, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.pos_emb_w, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.pos_emb_t, std=std, a=-3 * std, b=3 * std)

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_T_H_W_C = x_B_T_H_W_C.shape
        return self.generate_embeddings(B_T_H_W_C, fps=fps)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps: Optional[torch.Tensor] = None) -> torch.Tensor:
        del fps  # Unused, kept for API consistency with VideoRopePosition3DEmb
        B, T, H, W, _ = B_T_H_W_C
        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:T]
            emb = (
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
                + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
                + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
            )
        else:
            raise ValueError(f"Unknown interpolation method {self.interpolation}")

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(1e-6, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return emb / norm.to(emb.dtype)


# ---------------------- Timestep Embeddings -----------------------


class Timesteps(nn.Module):
    """Sinusoidal timestep embeddings."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2
        in_dtype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb.to(dtype=in_dtype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    """MLP for timestep embedding projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_adaln_lora: bool = False,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        linear = FP32Linear if use_wan_fp32_strategy else nn.Linear
        self.linear_1 = linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = linear(out_features, out_features, bias=False)

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.out_dim)
        nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mlp_input = sample if self.use_wan_fp32_strategy else sample.to(self.linear_1.weight.dtype)
        emb = self.linear_1(mlp_input)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = mlp_input
        else:
            emb_B_T_D = emb
            adaln_lora_B_T_3D = None

        return emb_B_T_D, adaln_lora_B_T_3D


# ---------------------- Patch Embedding -----------------------


class PatchEmbed(nn.Module):
    """Patch embedding for video input."""

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W)

        Returns:
            Embedded patches of shape (B, T, H, W, D)
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


# ---------------------- Final Layer -----------------------


class FinalLayer(nn.Module):
    """Final layer of video DiT with AdaLN modulation."""

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        linear = FP32Linear if use_wan_fp32_strategy else nn.Linear
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim

        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                linear(hidden_size, adaln_lora_dim, bias=False),
                linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            nn.init.zeros_(self.adaln_modulation[1].weight)
        self.layer_norm.reset_parameters()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d")
        scale_B_T_1_1_D = rearrange(scale_B_T_D, "b t d -> b t 1 1 d")
        final_context = (
            torch.amp.autocast(device_type=x_B_T_H_W_D.device.type, enabled=False)
            if self.use_wan_fp32_strategy
            else contextlib.nullcontext()
        )
        with final_context:
            if self.use_wan_fp32_strategy:
                x_B_T_H_W_D = x_B_T_H_W_D.to(promote_dtype(x_B_T_H_W_D, torch.float32))
            else:
                shift_B_T_1_1_D = shift_B_T_1_1_D.type_as(x_B_T_H_W_D)
                scale_B_T_1_1_D = scale_B_T_1_1_D.type_as(x_B_T_H_W_D)
            x_B_T_H_W_D = self.layer_norm(x_B_T_H_W_D) * (1 + scale_B_T_1_1_D) + shift_B_T_1_1_D
            x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


# ---------------------- Transformer Block -----------------------


class Block(nn.Module):
    """
    Transformer block with self-attention, cross-attention and MLP with AdaLN modulation.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(x_dim, context_dim, num_heads, x_dim // num_heads)

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        linear = FP32Linear if use_wan_fp32_strategy else nn.Linear
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                linear(x_dim, adaln_lora_dim, bias=False),
                linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                linear(x_dim, adaln_lora_dim, bias=False),
                linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                linear(x_dim, adaln_lora_dim, bias=False),
                linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), linear(x_dim, 3 * x_dim, bias=False))

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        crossattn_gate_scale: float = 1.0,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        # Match native FP32 modulation, including when parameter storage is BF16.
        modulation_context = (
            torch.amp.autocast(device_type=x_B_T_H_W_D.device.type, enabled=False)
            if self.use_wan_fp32_strategy
            else contextlib.nullcontext()
        )
        with modulation_context:
            if self.use_adaln_lora:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
            else:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                )
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape modulation tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        B, T, H, W, D = x_B_T_H_W_D.shape

        # Self-attention
        normalized_x = (
            self.layer_norm_self_attn(x_B_T_H_W_D) * (1 + scale_self_attn_B_T_1_1_D) + shift_self_attn_B_T_1_1_D
        )
        result_B_T_H_W_D = rearrange(
            self.self_attn(
                rearrange(normalized_x, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        # Cross-attention
        normalized_x = (
            self.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + scale_cross_attn_B_T_1_1_D) + shift_cross_attn_B_T_1_1_D
        )
        result_B_T_H_W_D = rearrange(
            self.cross_attn(
                rearrange(normalized_x, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=None,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * (gate_cross_attn_B_T_1_1_D * crossattn_gate_scale) + x_B_T_H_W_D

        # MLP
        normalized_x = self.layer_norm_mlp(x_B_T_H_W_D) * (1 + scale_mlp_B_T_1_1_D) + shift_mlp_B_T_1_1_D
        result_B_T_H_W_D = self.mlp(normalized_x)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D

        return x_B_T_H_W_D
