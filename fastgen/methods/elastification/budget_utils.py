# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Analytical parameter count for Wan sub-networks under Elastification.

Ports Megatron-LM's `flex_budget_utils.py::get_num_parameters`.

Router-controlled widths (`hidden_size`, `ffn_hidden_size`) may be either
plain ints (for a concrete sub-net) or `torch.Tensor` (for the soft expected
value produced by `soft_probs @ candidate_list_tensor`, giving a
differentiable budget-loss term).

Wan-scope drops:
  - No Mamba, no MoE → drops `mamba_*`, `moe_*` inputs, `flex_hetero_mamba`
    / `flex_hetero_moe_expert` branches, `moe_active` vs `moe_all` distinction,
    and returns a single count (not the `(all, active)` tuple).
  - No hybrid layer-pattern string → iterates `range(num_layers)` directly.
  - No GQA → drops `kv_channels` + `num_query_groups` split; uses `head_dim`
    such that `num_attention_heads * head_dim = hidden_size` at full width.

Wan additions (no Megatron analog):
  - `text_encoder_dim` — attn2 cross-attention K/V input width.
  - `in_channels` / `out_channels` / `patch_dim` — patch-embed + proj-out.
  - `time_freq_dim` — sinusoidal-embedding intermediate.
  - Cross-attention block per WanTransformerBlock.
  - AdaLN modulation table `[6, hidden]` per block.
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

    # Wan input / output boilerplate. Megatron LLMs have `embedding` +
    # `final_layernorm` + `output_layer` here; Wan has patch-embed on the
    # input side and a projection back to patch space on the output side,
    # plus condition-embedder projections that scale with hidden_size.
    patch_embed = in_channels * patch_dim * hidden_size
    time_embed = time_freq_dim * hidden_size + hidden_size * hidden_size
    text_embed = text_encoder_dim * hidden_size + hidden_size * hidden_size
    final_layernorm = hidden_size * 1
    proj_out = hidden_size * (patch_dim * out_channels)

    if isinstance(ffn_hidden_size, int):
        flex_hetero_ffn = False
    else:
        flex_hetero_ffn = ffn_hidden_size.shape[0] != 1

    # FFN (mirrors Megatron's `moe_all`/`moe_active`, collapsed to a single
    # value because Wan has no MoE).

    if flex_hetero_ffn:
        ffn_all = []
        for i in range(ffn_hidden_size.shape[0]):
            pre_mlp_ln = norm_multiplier * hidden_size
            linear_fc1 = ffn_hidden_size[i] * hidden_size
            linear_fc2 = ffn_hidden_size[i] * hidden_size
            ffn_all.append(pre_mlp_ln + linear_fc1 + linear_fc2)
    else:
        pre_mlp_ln = norm_multiplier * hidden_size
        linear_fc1 = ffn_hidden_size * hidden_size
        linear_fc2 = ffn_hidden_size * hidden_size
        ffn_all = pre_mlp_ln + linear_fc1 + linear_fc2

    # ATT (self-attention — `attn1`). Wan uses full attention with
    # `num_attention_heads * head_dim = hidden_size`, so we compute
    # `linear_qkv = 3 * num_attention_heads * head_dim * hidden_size` directly
    # rather than the `(num_heads + 2*num_query_groups) * kv_channels * hidden`
    # GQA-aware formula Megatron uses.

    input_ln = norm_multiplier * hidden_size
    linear_proj = num_attention_heads * head_dim * hidden_size
    linear_qkv = 3 * num_attention_heads * head_dim * hidden_size
    att = input_ln + linear_proj + linear_qkv

    # CROSS-ATT (`attn2`) — new relative to Megatron.
    # `linear_q` and `linear_out` both project on the video-side hidden;
    # `linear_k` and `linear_v` project from the frozen text-encoder output.

    input_ln_cross = norm_multiplier * hidden_size
    linear_q_cross = num_attention_heads * head_dim * hidden_size
    linear_proj_cross = num_attention_heads * head_dim * hidden_size
    linear_kv_cross = 2 * text_encoder_dim * hidden_size
    cross = input_ln_cross + linear_q_cross + linear_proj_cross + linear_kv_cross

    # AdaLN modulation `scale_shift_table` `[6, hidden]` per block.
    adaln = 6 * hidden_size

    # Per-block loop mirrors Megatron's `for i, c in enumerate(hybrid_pattern):`
    # but Wan has no pattern — every block is the same type. Optional
    # `layer_skip_probs[i]` weights each block's contribution by
    # `(1 - skip_prob)` (the "phantom" skip axis).

    all_params = 0
    for i in range(num_layers):
        if flex_hetero_ffn:
            block = att + cross + adaln + ffn_all[i]
        else:
            block = att + cross + adaln + ffn_all

        if layer_skip_probs is not None:
            block = block * (1 - layer_skip_probs[i])

        all_params += block

    return (
        patch_embed
        + time_embed
        + text_embed
        + all_params
        + final_layernorm
        + proj_out
    )
