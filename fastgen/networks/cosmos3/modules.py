# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# This module is a clean-room adaptation of the Apache-2.0 reference implementation
# `diffusers.models.transformers.transformer_cosmos3.Cosmos3OmniTransformer` and the
# packing/position-id helpers in `diffusers.pipelines.cosmos.pipeline_cosmos3_omni`
# (Copyright 2025 The NVIDIA Team and The HuggingFace Team, Apache-2.0). It is restricted to
# the *video* generation pathway (text understanding + vision generation), dropping the
# sound/action towers, and rewritten as a plain ``nn.Module`` (no diffusers ConfigMixin /
# ModelMixin) so it plugs into FastGen's native-network archetype (external weight loading,
# FSDP2 meta-init via ``reset_parameters``). The generation math (dual und/gen pathways,
# two-way joint attention, interleaved 3D-mRoPE, patchify/timestep scatter) is preserved so
# the released ``nvidia/Cosmos3-*`` checkpoint weights map onto it 1:1.

import math
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import RMSNorm

# =====================================================================================
# 3D-mRoPE position-id builders (mirrors pipeline_cosmos3_omni.get_3d_mrope_ids_*).
# These produce the [3, N] (temporal, height, width) position-id tensor consumed by
# Cosmos3VLTextRotaryEmbedding. Kept here (not in the pipeline) so both the FastGen
# network's forward() and sample() can build the joint sequence.
# =====================================================================================


def get_3d_mrope_ids_text_tokens(
    num_tokens: int,
    temporal_offset: float,
    use_float_positions: bool = False,
) -> Tuple[torch.Tensor, float]:
    """3D-mRoPE ids for text tokens: all three axes share the same monotonically
    increasing ids starting at ``temporal_offset``. Returns ``(ids[3, num_tokens], next_offset)``."""
    if use_float_positions:
        ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    else:
        ids = torch.arange(num_tokens, dtype=torch.long) + int(temporal_offset)
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: float,
    reset_spatial_indices: bool = True,
    fps: Optional[float] = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    start_frame_offset: int = 0,
) -> Tuple[torch.Tensor, float]:
    """3D-mRoPE ids for VAE vision tokens (T-major flattened ``grid_t x grid_h x grid_w``).

    When ``fps`` is given and ``grid_t > 1`` (video), temporal positions are fps-modulated
    float values; otherwise integer positions. Returns ``(ids[3, grid_t*grid_h*grid_w], next_offset)``.
    """
    fps_modulation_enabled = fps is not None and grid_t > 1

    if fps_modulation_enabled:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / temporal_compression_factor
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset

    if fps_modulation_enabled:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    next_temporal_offset = math.ceil(mrope_ids.max().item()) + 1
    return mrope_ids, next_temporal_offset


# =====================================================================================
# Backbone sub-modules
# =====================================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Cosmos3VLTextRotaryEmbedding(nn.Module):
    """Interleaved 3D-mRoPE rotary embedding (Qwen3-VL style).

    Consumes a ``[3, N]`` (or ``[3, B, N]``) position-id tensor and returns ``(cos, sin)``
    each shaped ``[N, head_dim]`` for the joint packed sequence.
    """

    def __init__(self, head_dim: int, rope_theta: float, rope_axes_dim: Tuple[int, int, int]):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rope_axes_dim = list(rope_axes_dim)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_inv_freq(self):
        """Recompute ``inv_freq`` without changing its prepared storage dtype or device."""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.inv_freq.device) / self.head_dim)
        )
        self.inv_freq.copy_(inv_freq)

    def apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """Reorganize chunked [TTT...HHH...WWW] frequencies into interleaved THTHWH...
        layout, preserving frequency continuity across the 3 grids. ``freqs``: [3, B, N, D/2]."""
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.rope_axes_dim[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, position_ids: torch.Tensor, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [3, N] -> [3, 1, N]
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(1)
        inv_freq = self.inv_freq.float()
        position_ids = position_ids.to(device)
        # This is a scalar outer product, not a real reduction. Using matmul here lets CUDA TF32 choose
        # shape-dependent kernels, which can change RoPE cos/sin for identical positions at T=3 vs T=21.
        freqs = position_ids[:, :, :, None].float() * inv_freq[None, None, None, :]  # [3, B, N, D/2]
        freqs = self.apply_interleaved_mrope(freqs)  # [B, N, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, N, D]
        cos = emb.cos().to(dtype=dtype).squeeze(0)  # [N, D]
        sin = emb.sin().to(dtype=dtype).squeeze(0)
        return cos, sin


class Cosmos3VLTextMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Cosmos3PackedMoTAttention(nn.Module):
    """Dual-pathway packed attention. Understanding (``to_*``) tokens self-attend causally;
    generation (``add_*_proj``) tokens attend over the full ``[und || gen]`` keys/values."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # Understanding (text) pathway
        self.to_q = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.to_k = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_v = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_out = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.norm_q = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.norm_k = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)

        # Generation (vision) pathway
        self.add_q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.add_k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.add_v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_add_out = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.norm_added_q = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.norm_added_k = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)

    @staticmethod
    def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
        # q: [N_q, H, D], k/v: [N_kv, H_kv, D]  ->  [N_q, H*D]
        q = q.transpose(0, 1).unsqueeze(0)  # [1, H, N_q, D]
        k = k.transpose(0, 1).unsqueeze(0)  # [1, H_kv, N_kv, D]
        v = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=True)  # [1, H, N_q, D]
        return out.squeeze(0).transpose(0, 1).flatten(-2, -1)  # [N_q, H*D]

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_und = self.norm_q(self.to_q(und_seq).view(-1, self.num_attention_heads, self.head_dim))
        k_und = self.norm_k(self.to_k(und_seq).view(-1, self.num_key_value_heads, self.head_dim))
        v_und = self.to_v(und_seq).view(-1, self.num_key_value_heads, self.head_dim)
        q_gen = self.norm_added_q(self.add_q_proj(gen_seq).view(-1, self.num_attention_heads, self.head_dim))
        k_gen = self.norm_added_k(self.add_k_proj(gen_seq).view(-1, self.num_key_value_heads, self.head_dim))
        v_gen = self.add_v_proj(gen_seq).view(-1, self.num_key_value_heads, self.head_dim)

        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        cos_und, sin_und = cos_und.unsqueeze(1), sin_und.unsqueeze(1)  # [N,1,D]
        q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        cos_gen, sin_gen = cos_gen.unsqueeze(1), sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen

        # Understanding: causal self-attention over text only.
        und_out = self.to_out(self._sdpa(q_und, k_und, v_und, is_causal=True))
        # Generation: full bidirectional attention over [und || gen] keys/values.
        all_k = torch.cat([k_und, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        gen_out = self.to_add_out(self._sdpa(q_gen, all_k, all_v, is_causal=False))
        return und_out, gen_out


class Cosmos3VLTextMoTDecoderLayer(nn.Module):
    """Qwen3-VL MoT decoder layer with dual (understanding / generation) pathways."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        attention_bias: bool,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.self_attn = Cosmos3PackedMoTAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = Cosmos3VLTextMLP(hidden_size, intermediate_size)
        self.mlp_moe_gen = Cosmos3VLTextMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.input_layernorm_moe_gen = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.post_attention_layernorm_moe_gen = RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False
        )

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        und_attn, gen_attn = self.self_attn(
            self.input_layernorm(und_seq), self.input_layernorm_moe_gen(gen_seq), rotary_emb
        )
        residual_und = und_seq + und_attn
        residual_gen = gen_seq + gen_attn
        out_und = residual_und + self.mlp(self.post_attention_layernorm(residual_und))
        out_gen = residual_gen + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))
        return out_und, out_gen


# =====================================================================================
# Backbone
# =====================================================================================


class Cosmos3Backbone(nn.Module):
    """Cosmos3 video-generation backbone (understanding text tower + diffusion vision tower).

    Plain ``nn.Module`` adaptation of ``Cosmos3OmniTransformer`` restricted to text+vision.
    Parameter names match the released flat checkpoint layout (``embed_tokens``, ``layers.{i}``,
    ``norm``, ``norm_moe_gen``, ``proj_in``, ``proj_out``, ``time_embedder``) so a remapped
    Cosmos3 state-dict loads directly.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 12288,
        latent_channel: int = 48,
        latent_patch_size: int = 2,
        patch_latent_dim: int = 192,
        vocab_size: int = 151936,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 5000000.0,
        rope_axes_dim: Tuple[int, int, int] = (24, 20, 20),
        attention_bias: bool = False,
        enable_logvar_linear: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.latent_channel = latent_channel
        self.latent_patch_size = latent_patch_size
        self.patch_latent_dim = patch_latent_dim
        self.gradient_checkpointing = gradient_checkpointing

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                Cosmos3VLTextMoTDecoderLayer(
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    intermediate_size=intermediate_size,
                    attention_bias=attention_bias,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.norm_moe_gen = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.rotary_emb = Cosmos3VLTextRotaryEmbedding(head_dim, rope_theta, rope_axes_dim)

        self.proj_in = nn.Linear(patch_latent_dim, hidden_size, bias=True)
        self.proj_out = nn.Linear(hidden_size, patch_latent_dim, bias=True)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)

        # Optional uncertainty head for distillation (DMD2). Computed from a scalar-timestep embedding.
        self.logvar_linear = nn.Linear(hidden_size, 1) if enable_logvar_linear else None

    # ----- pure-tensor pack/unpack helpers (mirror the diffusers reference) -----

    def _patchify_and_pack_latents(
        self, tokens_vision: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        p, c = self.latent_patch_size, self.latent_channel
        packed, original_shapes = [], []
        for latent in tokens_vision:
            latent = latent.squeeze(0)  # [C, T, H, W]
            _, t, h, w = latent.shape
            original_shapes.append((t, h, w))
            h_pad = ((h + p - 1) // p) * p
            w_pad = ((w + p - 1) // p) * p
            if h_pad != h or w_pad != w:
                padded = torch.zeros((c, t, h_pad, w_pad), device=latent.device, dtype=latent.dtype)
                padded[:, :, :h, :w] = latent
                latent = padded
            hp, wp = h_pad // p, w_pad // p
            latent = latent.reshape(c, t, hp, p, wp, p)
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * c)
            packed.append(latent)
        return torch.cat(packed, dim=0), original_shapes

    def _apply_timestep_embeds_to_noisy_tokens(
        self,
        packed_tokens: torch.Tensor,
        packed_timestep_embeds: torch.Tensor,
        noisy_frame_indexes: List[torch.Tensor],
        token_shapes: List[Tuple[int, ...]],
    ) -> torch.Tensor:
        start = 0
        flat_idx = []
        for noisy_i, shape_i in zip(noisy_frame_indexes, token_shapes):
            spatial = math.prod(shape_i[1:])
            spatial_idx = torch.arange(spatial, device=packed_tokens.device)
            offsets = (noisy_i * spatial).unsqueeze(-1) + spatial_idx + start
            flat_idx.append(offsets.flatten())
            start += shape_i[0] * spatial
        flattened = torch.cat(flat_idx, dim=0).unsqueeze(-1).expand(-1, packed_tokens.shape[1])
        # Autocast can give the projected vision tokens its AMP dtype while the
        # timestep embedding retains the model/embedding dtype.  scatter_add
        # requires its floating-point operands to match.
        packed_timestep_embeds = packed_timestep_embeds.to(dtype=packed_tokens.dtype)
        return packed_tokens.scatter_add(dim=0, index=flattened, src=packed_timestep_embeds)

    def _unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
        noisy_frame_indexes_vision: List[torch.Tensor],
        original_latent_shapes: List[Tuple[int, int, int]],
    ) -> List[torch.Tensor]:
        p = self.latent_patch_size
        out_channels = packed_mse_preds.shape[-1] // (p * p)
        assert (
            out_channels * p * p == packed_mse_preds.shape[-1]
        ), f"Packed prediction dim {packed_mse_preds.shape[-1]} is not divisible by patch area {p * p}."
        outs, start = [], 0
        for token_shape, noisy_idx, original in zip(
            token_shapes_vision, noisy_frame_indexes_vision, original_latent_shapes
        ):
            t_c = token_shape[0]
            _, h_orig, w_orig = original
            hp = (((h_orig + p - 1) // p) * p) // p
            wp = (((w_orig + p - 1) // p) * p) // p
            t_n = len(noisy_idx)
            out = torch.zeros(
                (out_channels, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )
            num_patches = t_n * hp * wp
            if num_patches > 0:
                patches = packed_mse_preds[start : start + num_patches].reshape(t_n, hp, wp, p, p, out_channels)
                latent = torch.einsum("thwpqc->cthpwq", patches).reshape(out_channels, t_n, hp * p, wp * p)
                out[:, noisy_idx] = latent[:, :, :h_orig, :w_orig]
                start += num_patches
            outs.append(out.unsqueeze(0))
        return outs

    def compute_t_embed(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Timestep embedding [B, hidden] from per-sample timesteps [B].

        time_embedder is kept in fp32 to match the reference; the sinusoidal time_proj output
        is fp32, so cast to the embedder's weight dtype.
        """
        te_dtype = self.time_embedder.linear_1.weight.dtype
        return self.time_embedder(self.time_proj(timesteps).to(te_dtype))

    def compute_logvar(self, t_embed: torch.Tensor) -> torch.Tensor:
        """Per-sample logvar [B, 1] from a ``compute_t_embed`` embedding, used by DMD2
        score losses."""
        return self.logvar_linear(t_embed.to(self.logvar_linear.weight.dtype))

    # ----- single-sample forward over a pre-built packed sequence -----

    def forward(
        self,
        input_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        position_ids: torch.Tensor,
        und_len: int,
        sequence_length: int,
        vision_tokens: List[torch.Tensor],
        vision_token_shapes: List[Tuple[int, int, int]],
        vision_sequence_indexes: torch.Tensor,
        vision_mse_loss_indexes: torch.Tensor,
        num_noisy_vision_tokens: int,
        vision_noisy_frame_indexes: List[torch.Tensor],
        t: torch.Tensor,
        feature_indices: Optional[Set[int]] = None,
        return_features_early: bool = False,
        return_logvar: bool = False,
    ):
        """Run one denoising-step forward over a single packed (text||vision) sequence.

        Returns ``(preds_vision, features, logvar)``: per-item velocity latents
        ``[1, C, T, H, W]`` (zero on conditioning frames); the per-layer generation-stream
        features captured after the selected decoder layers when ``feature_indices`` is not
        ``None`` (including ``[]`` for an empty set), else ``None``; and the ``[1, 1]``
        logvar when ``return_logvar``, else ``None``.
        """
        t_embed = self.compute_t_embed(t)
        logvar = self.compute_logvar(t_embed) if return_logvar else None

        packed_text = self.embed_tokens(input_ids)
        packed_vis, original_shapes = self._patchify_and_pack_latents(vision_tokens)
        packed_vis = self.proj_in(packed_vis)

        target_dtype = packed_vis.dtype
        hidden_states = packed_vis.new_zeros((sequence_length, self.hidden_size))
        hidden_states[text_indexes] = packed_text.to(dtype=target_dtype)
        # All noisy tokens of a sample share one t: broadcast the single embedding row.
        t_embeds = t_embed.to(target_dtype).expand(num_noisy_vision_tokens, -1).contiguous()
        packed_vis = self._apply_timestep_embeds_to_noisy_tokens(
            packed_vis, t_embeds, vision_noisy_frame_indexes, vision_token_shapes
        )
        hidden_states[vision_sequence_indexes] = packed_vis

        cos, sin = self.rotary_emb(position_ids, device=hidden_states.device, dtype=hidden_states.dtype)
        und_seq, gen_seq = hidden_states[:und_len], hidden_states[und_len:]
        rotary_emb = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])

        features: List[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            # DMD2's frozen teacher is kept in eval mode, but the GAN-generator
            # loss still needs gradients with respect to the teacher input.
            # Gate on autograd state so that path is checkpointed too.
            if self.gradient_checkpointing and torch.is_grad_enabled():
                und_seq, gen_seq = torch.utils.checkpoint.checkpoint(
                    layer, und_seq, gen_seq, rotary_emb, use_reentrant=False
                )
            else:
                und_seq, gen_seq = layer(und_seq, gen_seq, rotary_emb)
            if feature_indices is not None and idx in feature_indices:
                features.append(gen_seq)
                if return_features_early and len(features) == len(feature_indices):
                    return None, features, logvar

        # ``vision_mse_loss_indexes`` address the packed [und || gen] sequence and are all >= und_len (noisy vision
        # tokens only), so index the generation stream directly. ``norm`` (the reference's lm_head input) stays
        # registered only so checkpoints load.
        gen_out = self.norm_moe_gen(gen_seq)
        preds_packed = self.proj_out(gen_out[vision_mse_loss_indexes - und_len])
        preds_vision = self._unpatchify_and_unpack_latents(
            preds_packed, vision_token_shapes, vision_noisy_frame_indexes, original_shapes
        )
        return preds_vision, features if feature_indices is not None else None, logvar

    # ----- init / FSDP meta-init -----

    def reset_parameters(self):
        """Rebuild the non-persistent rotary buffer after meta materialization."""
        self.rotary_emb.reset_inv_freq()
