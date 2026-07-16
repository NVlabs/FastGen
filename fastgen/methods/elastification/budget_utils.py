# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Analytical parameter count for Wan sub-networks under Elastification.

Ports Megatron-LM's `flex_budget_utils.py::get_num_parameters`.

Router-controlled widths (`hidden_size`, `mlp_hidden_size`) may be either
plain ints (for a concrete sub-net) or 0-d/1-d `torch.Tensor` (for the soft
expected value produced by `soft_probs @ candidate_list_tensor`, giving a
differentiable budget-loss term).

Wan-scope drops relative to Megatron:
  - No Mamba, no MoE (Wan has neither) → drops `mamba_*`, `moe_*` inputs,
    `flex_hetero_moe_expert`/`flex_hetero_mamba` branches, `moe_active` vs
    `moe_all` distinction, and returns a single count (not a tuple).
  - No hybrid layer pattern string → num_layers + layer_skip_probs directly.
  - No `kv_channels` / `num_query_groups` split — Wan does full attention
    without GQA, so `num_attention_heads * head_dim = hidden_size`.

Wan-specific additions:
  - `text_encoder_dim` for attn2's cross-attention K/V input width.
  - `patch_dim` (product of the 3D patch sizes) and `in_channels` / `out_channels`
    for the patch-embed and proj-out linear-equivalent layers.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


ScalarOrTensor = Union[int, float, torch.Tensor]


def get_num_parameters(
    num_layers: int,
    hidden_size: ScalarOrTensor,
    mlp_hidden_size: ScalarOrTensor,
    num_attention_heads: int,
    head_dim: int,
    text_encoder_dim: int,
    in_channels: int,
    out_channels: int,
    patch_dim: int,
    time_freq_dim: int = 256,
    layer_skip_probs: Optional[torch.Tensor] = None,
) -> ScalarOrTensor:
    """Total parameter count for a Wan sub-network at the given widths.

    Args:
        num_layers: Number of `WanTransformerBlock` in the transformer stack.
        hidden_size: Model residual-stream width. Router-controlled; may be a
            0-d tensor (soft expected value).
        mlp_hidden_size: FFN hidden width. Router-controlled; may be a tensor.
        num_attention_heads: Fixed. `hidden_size = num_attention_heads * head_dim`
            when using the full model.
        head_dim: Fixed. Attention head dimension.
        text_encoder_dim: Fixed. Text encoder output dim (attn2's K/V input).
        in_channels: Fixed. Input latent channels.
        out_channels: Fixed. Output latent channels (before unpatchify).
        patch_dim: Fixed. Product of 3D patch sizes `p_t * p_h * p_w`.
        time_freq_dim: Fixed. Sinusoidal-embedding frequency dim before the
            two-layer projection to `hidden_size`.
        layer_skip_probs: Optional `[num_layers]` tensor of per-layer skip
            probabilities in `[0, 1]` from the router's skip axis. Each block's
            parameters are weighted by `(1 - skip_prob)`. When `None`, all
            blocks count fully.

    Returns:
        Total parameter count. Same type as `hidden_size` / `mlp_hidden_size`
        (scalar int / float when both are ints; 0-d tensor otherwise).
    """

    # ── Input / output "boilerplate" that scales with hidden_size only ──
    # Patch embed: Conv3d weight equivalent = in_channels * patch_volume * hidden.
    patch_embed = in_channels * patch_dim * hidden_size

    # Condition embedder — approximate as two hidden×hidden linears each for
    # (a) the timestep-frequency → hidden projection and (b) the text-encoder
    # → hidden projection.
    time_embed = time_freq_dim * hidden_size + hidden_size * hidden_size
    text_embed = text_encoder_dim * hidden_size + hidden_size * hidden_size

    # Final projection back to patch space.
    proj_out = hidden_size * (patch_dim * out_channels)

    # `norm_out` is affine LayerNorm on hidden_size.
    norm_out = 2 * hidden_size

    # ── Per-block params (attn1 + attn2 + ffn + adaLN + head/block norms) ─
    #
    # attn1 (self-attn): to_q + to_k + to_v + to_out.0, each hidden × hidden.
    attn1 = 4 * hidden_size * hidden_size

    # attn2 (cross-attn):
    #   to_q, to_out.0 : hidden × hidden
    #   to_k, to_v     : text_encoder_dim × hidden (text side untouched)
    attn2 = 2 * hidden_size * hidden_size + 2 * text_encoder_dim * hidden_size

    # FFN: fc1 (hidden → mlp_hidden) + fc2 (mlp_hidden → hidden)
    ffn = 2 * hidden_size * mlp_hidden_size

    # AdaLN modulation table: [6, hidden]
    adaln = 6 * hidden_size

    # norm_q / norm_k on head_dim inside attn1 and attn2 (4 total),
    # plus norm1 / norm2 / norm3 on hidden inside the block.
    norms = 4 * head_dim + 3 * hidden_size

    per_block = attn1 + attn2 + ffn + adaln + norms

    # ── Layer-skip weighting ────────────────────────────────────────────
    if layer_skip_probs is not None:
        assert layer_skip_probs.shape == (num_layers,), (
            f"layer_skip_probs shape {tuple(layer_skip_probs.shape)} does not "
            f"match num_layers={num_layers}"
        )
        total_blocks = ((1 - layer_skip_probs) * per_block).sum()
    else:
        total_blocks = num_layers * per_block

    return patch_embed + time_embed + text_embed + total_blocks + norm_out + proj_out
