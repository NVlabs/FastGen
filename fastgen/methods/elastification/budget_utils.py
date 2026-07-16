# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Analytical parameter count for Wan sub-networks under Elastification.

Ports Megatron-LM's `flex_budget_utils.py::get_num_parameters`.

Router-controlled widths (`hidden_size`, `ffn_hidden_size`) may be either
plain ints (for a concrete sub-net) or `torch.Tensor` (for the soft expected
value produced by `soft_probs @ candidate_list_tensor`, giving a
differentiable budget-loss term).

The formula is exact against `diffusers.WanTransformer3DModel` for both the
1.3B (1,418,996,800) and 14B (14,288,491,584) reference configs at full
width. Every parameter tensor of the full model is accounted for.

Wan-scope drops relative to Megatron:
  - No Mamba, no MoE → drops `mamba_*`, `moe_*` inputs, `flex_hetero_mamba`
    / `flex_hetero_moe_expert` branches, `moe_active` vs `moe_all`
    distinction, and returns a single count (not the `(all, active)` tuple).
  - No hybrid layer-pattern string → iterates `range(num_layers)` directly.
  - No GQA → drops `kv_channels` + `num_query_groups` split.
  - No top-level `final_layernorm` (Wan does not have one).

Wan additions (no Megatron analog):
  - `text_encoder_dim` — dim of the frozen text encoder's output; enters
    ONLY through `condition_embedder.text_embedder.linear_1`, not through
    `attn2.to_k`/`to_v`, because text is projected to `hidden_size` by
    `text_embedder` before the transformer blocks see it.
  - `in_channels` / `out_channels` / `patch_dim` — patch-embed + proj-out.
  - `time_freq_dim` — sinusoidal-embedding intermediate.
  - `time_proj` (h → 6·h) inside condition_embedder for AdaLN modulation.
  - Cross-attention block (`attn2`) per WanTransformerBlock.
  - Per-block `scale_shift_table` `[6, h]` + top-level `scale_shift_table`
    `[2, h]` for AdaLN.
  - All linear layers carry biases (attn Q/K/V/out, ffn fc1/fc2,
    condition_embedder linears, patch_embedding, proj_out).
  - `norm2` inside each block is affine (from `cross_attn_norm=True`);
    `norm1` and `norm3` are parameter-free.
  - `norm_q` / `norm_k` inside each attention module are `hidden`-sized
    (rms_norm_across_heads), not `head_dim`-sized.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


ScalarOrTensor = Union[int, float, torch.Tensor]


def get_num_parameters(
    num_layers: int = 0,
    num_attention_heads: int = 0,
    head_dim: int = 0,
    hidden_size: ScalarOrTensor = 0,
    ffn_hidden_size: ScalarOrTensor = 0,
    # ── Wan additions ─────────────────────────────────────────────────
    text_encoder_dim: int = 0,
    in_channels: int = 0,
    out_channels: int = 0,
    patch_dim: int = 0,
    time_freq_dim: int = 256,
    layer_skip_probs: Optional[torch.Tensor] = None,
) -> ScalarOrTensor:

    norm_multiplier = 1
    h = hidden_size

    # ── Top-level (not per-block) ─────────────────────────────────────
    # Wan replaces Megatron's `embedding + final_layernorm + output_layer`
    # with a patch-embed on the input, a condition-embedder that scales
    # with hidden_size, and a proj-out to patch space. There is no
    # top-level LayerNorm in Wan.

    # patch_embedding: Conv3d(in_channels, hidden, kernel=patch_size)
    patch_embed = in_channels * patch_dim * h + h

    # condition_embedder.text_embedder: text_encoder_dim → h → h
    text_embed = (text_encoder_dim * h + h) + (h * h + h)

    # condition_embedder.time_embedder: freq_dim → h → h
    time_embed = (time_freq_dim * h + h) + (h * h + h)

    # condition_embedder.time_proj: h → 6·h for AdaLN modulation
    time_proj = h * (6 * h) + 6 * h

    # Top-level scale_shift_table [2, h]
    top_scale_shift = 2 * h

    # proj_out: h → patch_dim · out_channels
    proj_out = h * (patch_dim * out_channels) + (patch_dim * out_channels)

    # ── Hetero-FFN detection (mirrors Megatron) ───────────────────────
    if isinstance(ffn_hidden_size, int):
        flex_hetero_ffn = False
    else:
        flex_hetero_ffn = ffn_hidden_size.shape[0] != 1

    # ── FFN block ─────────────────────────────────────────────────────
    # Mirrors Megatron's `moe_all` structure, collapsed to a single value
    # because Wan has no MoE. Wan's `WanTransformerBlock.ffn` is a diffusers
    # `FeedForward` with `net.0` (linear+GELU) + `net.2` (linear), both
    # with biases.

    if flex_hetero_ffn:
        ffn_all = []
        for i in range(ffn_hidden_size.shape[0]):
            linear_fc1 = ffn_hidden_size[i] * h + ffn_hidden_size[i]  # weight + bias
            linear_fc2 = h * ffn_hidden_size[i] + h                    # weight + bias
            ffn_all.append(linear_fc1 + linear_fc2)
    else:
        linear_fc1 = ffn_hidden_size * h + ffn_hidden_size
        linear_fc2 = h * ffn_hidden_size + h
        ffn_all = linear_fc1 + linear_fc2

    # ── ATT block ─────────────────────────────────────────────────────
    # attn1 (self-attn) and attn2 (cross-attn) have identical parameter
    # shapes: four hidden×hidden linears (Q/K/V/out) each with bias, plus
    # two hidden-sized RMSNorm weights (norm_q, norm_k). Wan does full
    # attention with `num_attention_heads * head_dim = h`, no GQA — so
    # Megatron's `(num_heads + 2*num_query_groups) * kv_channels * h`
    # collapses to `4 * h * h`. Cross-attn K/V input is `h` too (not
    # `text_encoder_dim`), because `text_embedder` above already projected
    # the text encoder output to `h`.
    linear_qkv_out_weight = 4 * h * h
    linear_qkv_out_bias = 4 * h
    norm_qk = 2 * h
    att = linear_qkv_out_weight + linear_qkv_out_bias + norm_qk

    # ── Norms + AdaLN scale-shift ─────────────────────────────────────
    # Only `norm2` is affine (from `cross_attn_norm=True`); `norm1` and
    # `norm3` are parameter-free LayerNorms.
    norm2 = 2 * norm_multiplier * h

    # Per-block scale_shift_table [6, h]
    adaln = 6 * h

    # ── Per-block loop ────────────────────────────────────────────────
    # Mirrors Megatron's `for i, c in enumerate(hybrid_pattern):` but Wan
    # has no pattern — every block is the same type. Optional
    # `layer_skip_probs[i]` weights each block's contribution by
    # `(1 - skip_prob)` (the "phantom" skip axis).

    all_params = 0
    for i in range(num_layers):
        if flex_hetero_ffn:
            block = 2 * att + norm2 + adaln + ffn_all[i]
        else:
            block = 2 * att + norm2 + adaln + ffn_all

        if layer_skip_probs is not None:
            block = block * (1 - layer_skip_probs[i])

        all_params += block

    return (
        patch_embed
        + text_embed
        + time_embed
        + time_proj
        + top_scale_shift
        + proj_out
        + all_params
    )
