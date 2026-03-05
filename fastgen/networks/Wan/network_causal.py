# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# See licenses/diffusers/LICENSE for more details.
# -----------------------------------------------------------------------------

import types
from typing import Optional, List, Set, Any, Dict, Tuple, Union
import os
from functools import partial

from tqdm.auto import tqdm
import math
import torch

from diffusers.utils import USE_PEFT_BACKEND, unscale_lora_layers
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import (
    WanAttnProcessor,
    WanAttention,
    _get_qkv_projections,
    _get_added_kv_projections,
)
from fastgen.networks.network import CausalFastGenNetwork
from fastgen.networks.Wan.network import Wan, classify_forward, normalize, flatten_timestep, unflatten_timestep_proj
from fastgen.networks.noise_schedule import NET_PRED_TYPES
import fastgen.utils.logging_utils as logger


# Optional FlexAttention (falls back to SDP if unavailable)
# You can disable FlexAttention entirely by setting FASTGEN_DISABLE_FLEX_ATTENTION=1
_disable_flex_env = os.environ.get("FASTGEN_DISABLE_FLEX_ATTENTION", "0") == "1"
try:
    if _disable_flex_env:
        raise ImportError("FlexAttention disabled via FASTGEN_DISABLE_FLEX_ATTENTION=1")
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask

    FLEX_ATTENTION_AVAILABLE = True
    # Wan 1.3B requires max-autotune to work well with flexattention, see PyTorch issue #133254
    # Turn off Dynamo DDP optimizer when using flex attention (higher-order ops trigger NotImplemented)
    try:  # pragma: no cover
        import torch._dynamo as _dynamo

        _dynamo.config.optimize_ddp = False
    except Exception:
        pass

    # Obey TORCH_COMPILE_MODE if provided; allow disabling compile to avoid autotune benchmark
    _compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
    _disable_compile_wrap = (
        os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1" or os.environ.get("FASTGEN_FLEX_COMPILE", "1") == "0"
    )
    if not _disable_compile_wrap:
        flex_attention = torch.compile(flex_attention, dynamic=False, mode=_compile_mode)
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    create_block_mask = None  # type: ignore
    flex_attention = None  # type: ignore

    class BlockMask:  # type: ignore
        pass


def _rope_forward_with_time_offset(
    self, hidden_states: torch.Tensor, start_frame: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary embedding with a temporal offset for autoregressive causal decoding.

    start_frame is in units of frame-patches, i.e., how many frame
    blocks of length (H_patches * W_patches) have already been generated.
    """
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    # split by (t, h, w)
    split_sizes = [
        self.attention_head_dim - 2 * (self.attention_head_dim // 3),
        self.attention_head_dim // 3,
        self.attention_head_dim // 3,
    ]

    freqs_cos_t, freqs_cos_h, freqs_cos_w = self.freqs_cos.split(split_sizes, dim=1)
    freqs_sin_t, freqs_sin_h, freqs_sin_w = self.freqs_sin.split(split_sizes, dim=1)

    # --- temporal offset handling ---
    total_f = freqs_cos_t.shape[0]  # max_seq_len S
    start = max(0, start_frame)
    end = start + ppf
    if start >= total_f:
        cos_f = freqs_cos_t[-1:].expand(ppf, -1)
        sin_f = freqs_sin_t[-1:].expand(ppf, -1)
    else:
        end = min(end, total_f)
        cos_f, sin_f = freqs_cos_t[start:end], freqs_sin_t[start:end]
        if cos_f.shape[0] < ppf:  # pad if not enough
            pad_rows = ppf - cos_f.shape[0]
            cos_f = torch.cat([cos_f, freqs_cos_t[end - 1 : end].expand(pad_rows, -1)], dim=0)
            sin_f = torch.cat([sin_f, freqs_sin_t[end - 1 : end].expand(pad_rows, -1)], dim=0)

    freqs_cos_f = cos_f.view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_sin_f = sin_f.view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)

    freqs_cos_h = freqs_cos_h[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_sin_h = freqs_sin_h[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)

    freqs_cos_w = freqs_cos_w[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
    freqs_sin_w = freqs_sin_w[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

    freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
    freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

    return freqs_cos, freqs_sin


def _prepare_blockwise_causal_attn_mask(
    self,
    device: torch.device | str,
    num_frames: int = 21,
    frame_seqlen: int = 1560,
    chunk_size: int = 1,
) -> BlockMask | None:
    """
    Construct a block-wise causal attention mask over a sequence consisting of
    contiguous frame tokens laid out as [frame0 | frame1 | ...], where each frame
    contributes `frame_seqlen` tokens. The mask allows each token within a block of
    `chunk_size` frames to attend to all tokens in that block up to the
    block's end (inclusive of self).

    The sequence is right-padded to a multiple of 128 tokens for FlexAttention.
    Returns None if FlexAttention is unavailable.
    """
    if not FLEX_ATTENTION_AVAILABLE:
        return None

    logger.info("creating blockwise causal attn mask for teacher-forcing or diffusion-forcing")

    total_length = num_frames * frame_seqlen

    # right pad to multiple of 128
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

    # Front-load remaining frames into the first chunk, like CogVideoX forward
    num_chunks = num_frames // chunk_size
    remaining_size = num_frames % chunk_size

    # Determine frames per chunk
    frame_counts: List[int] = []
    if num_frames > 0:
        if num_chunks == 0:
            # All frames go into a single (partial) chunk
            frame_counts.append(remaining_size)
        else:
            # First chunk is larger by remaining_size, then uniform chunks
            frame_counts.append(chunk_size + remaining_size)
            frame_counts.extend([chunk_size] * max(num_chunks - 1, 0))

    # Fill block ends for each block of tokens
    current_start = 0
    for frames_in_chunk in frame_counts:
        chunk_len_tokens = frames_in_chunk * frame_seqlen
        ends[current_start : current_start + chunk_len_tokens] = current_start + chunk_len_tokens
        current_start += chunk_len_tokens

    def attention_mask(b, h, q_idx, kv_idx) -> torch.Tensor:
        # allow attending to tokens strictly before the block end, plus self
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )  # type: ignore

    return block_mask


class CausalWanAttnProcessor(WanAttnProcessor):
    """
    Causal attention processor for Wan transformer with self-forcing capabilities.

    This processor handles both self-attention and cross-attention with external KV caching
    for causal generation. It supports Image-to-Video (I2V) tasks, FlexAttention for
    efficient training, and gradient preservation for self-forcing training.

    Key Features:
    - External KV caching for causal autoregressive generation
    - Static cross-attention caching for text conditioning
    - Image branch processing for I2V tasks
    - FlexAttention support with block masks
    - Gradient preservation for self-forcing
    """

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_mask: Optional[BlockMask] = None,
        cache_tag: str = "pos",
        store_kv: bool = False,
        attention_cache_kwargs: Optional[Dict[str, Any]] = None,
        cache_start_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass for causal Wan attention with KV caching support.

        Args:
            attn: The WanAttention module to process
            hidden_states: Input hidden states [B, L, D]
            encoder_hidden_states: Optional conditioning states [B, L_enc, D] for cross-attention
            attention_mask: Optional attention mask for sequence masking
            rotary_emb: Optional rotary positional embeddings (cos, sin) tuple
            block_mask: Optional block mask for FlexAttention causal patterns
            cache_tag: Cache identifier for KV storage ("pos" for positive, "neg" for negative prompts)
            store_kv: Whether to store computed KV pairs in external cache for future use
            attention_cache_kwargs: Additional arguments including external_self_kv and external_cross_kv caches

        Returns:
            torch.Tensor: Processed attention output [B, L, D]
        """
        # Initialize attention_cache_kwargs if None
        attention_cache_kwargs = attention_cache_kwargs or {}

        TEXT_ENCODER_CONTEXT_LENGTH = 512  # Wan text encoder uses 512 token context

        # === Step 1: Handle I2V encoder state separation ===
        encoder_hidden_states_img = None
        is_cross_attn = encoder_hidden_states is not None  # Cross-attention when we have conditioning
        if is_cross_attn and attn.add_k_proj is not None:
            # I2V task: split combined encoder states into image and text components
            image_context_length = encoder_hidden_states.shape[1] - TEXT_ENCODER_CONTEXT_LENGTH
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # === Step 2: Compute and normalize QKV projections ===
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        # Apply layer normalization to Q and K
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Reshape for multi-head attention: (B, L, H, D) where H=num_heads, D=head_dim
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # === Step 3: Apply rotary positional embeddings ===
        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ) -> torch.Tensor:
                """Apply rotary positional embedding to input tensor."""
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)
        # Ensure type hasn't changed after normalization
        query = query.type_as(hidden_states)
        key = key.type_as(hidden_states)

        # === Step 4: Process image branch for I2V tasks ===
        hidden_states_img = None
        # Only process image branch if we have both image states AND the required projection layers
        if (
            encoder_hidden_states_img is not None
            and encoder_hidden_states_img.shape[1] > 0  # Non-empty image context
            and attn.add_k_proj is not None
            and attn.add_v_proj is not None
        ):
            # Compute image branch attention (separate from main text attention)
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            # Reshape for multi-head attention
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            # Compute image attention using query from main branch
            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # === Step 5: Compute main attention (self-attention or cross-attention) ===
        # Get external KV caches from attention_cache_kwargs
        external_self_kv = attention_cache_kwargs.get("external_self_kv", {})
        external_cross_kv = attention_cache_kwargs.get("external_cross_kv", {})

        # Choose attention computation path based on attention type
        if is_cross_attn:
            # --- CROSS-ATTENTION PATH ---
            # Use external cross-attention cache if available (static K/V from text conditioning)
            k_use, v_use = key, value  # Default: use current K/V
            use_external_cache = bool(external_cross_kv)

            if use_external_cache:
                per_tag = external_cross_kv.get(cache_tag, {})
                cache_initialized = per_tag and bool(per_tag.get("is_init", False))
                grad_enabled = torch.is_grad_enabled()

                if cache_initialized and not grad_enabled:
                    # Use cached K/V only in no-grad phases
                    k_use = per_tag["k"]
                    v_use = per_tag["v"]
                elif store_kv:
                    # Refresh cache with current K/V (allows grad flow when enabled)
                    external_cross_kv[cache_tag] = {"k": key, "v": value, "is_init": True}

            # Compute cross-attention
            hidden_states = dispatch_attention_fn(
                query,
                k_use,
                v_use,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )

        else:
            # --- SELF-ATTENTION PATH ---
            bsz, seqlen, nheads, head_dim = query.shape  # query format: [B, L, H, D] after unflatten
            per_tag = external_self_kv.get(cache_tag, {}) if bool(external_self_kv) else {}
            # Use external cache if it's available and there is no block mask provided
            use_external_cache = (
                bool(per_tag) and "k" in per_tag and "v" in per_tag and "len" in per_tag
            ) and block_mask is None
            # Use flex attention if it's available and we aren't using the external cache
            use_flex_attention = not use_external_cache and FLEX_ATTENTION_AVAILABLE and not is_cross_attn

            if block_mask is not None and not use_flex_attention and not is_cross_attn:
                logger.warning(
                    "Block mask provided for self-attention but flex_attention unavailable. "
                    f"use_external_cache={use_external_cache}, FLEX_ATTENTION_AVAILABLE={FLEX_ATTENTION_AVAILABLE}"
                )

            if use_external_cache:
                # Use external KV buffers for causal generation (append-style caching)
                k_buf, v_buf = per_tag["k"], per_tag["v"]  # shape: [bsz, max_len, nheads, head_dim]
                if cache_start_idx is None:
                    cache_start_idx = int(per_tag["len"])
                end_idx = cache_start_idx + seqlen

                # Store current K/V in cache
                if store_kv:
                    per_tag["len"] = end_idx
                    k_buf[:, cache_start_idx:end_idx, :, :] = key.detach()
                    v_buf[:, cache_start_idx:end_idx, :, :] = value.detach()

                # Use accumulated KV cache for attention computation
                if cache_start_idx == 0:
                    # First chunk: just use current key/value
                    k_full = key
                    v_full = value
                else:
                    # Subsequent chunks: combine cached + current
                    # Only current chunk (key/value) should contribute to gradients during generation
                    with torch.no_grad():
                        k_cached = k_buf[:, :cache_start_idx, :, :]
                        v_cached = v_buf[:, :cache_start_idx, :, :]
                    k_full = torch.cat([k_cached, key], dim=1)
                    v_full = torch.cat([v_cached, value], dim=1)

                hidden_states = dispatch_attention_fn(
                    query,
                    k_full,
                    v_full,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                )

            elif use_flex_attention:
                # Use FlexAttention for efficient training with causal block masks
                # FlexAttention requires sequence length to be multiple of 128
                FLEX_PADDING_SIZE = 128
                padded_len = (math.ceil(seqlen / FLEX_PADDING_SIZE) * FLEX_PADDING_SIZE) - seqlen

                if padded_len > 0:
                    # Add zero padding to meet FlexAttention requirements
                    pad_shape = (bsz, padded_len, nheads, head_dim)
                    pad_q = torch.zeros(pad_shape, device=query.device, dtype=query.dtype)
                    pad_k = torch.zeros(pad_shape, device=key.device, dtype=key.dtype)
                    pad_v = torch.zeros(pad_shape, device=value.device, dtype=value.dtype)

                    query = torch.cat([query, pad_q], dim=1)
                    key = torch.cat([key, pad_k], dim=1)
                    value = torch.cat([value, pad_v], dim=1)

                # Compute FlexAttention with block mask
                # Flex attention expects swapped sequence and head dims
                # Note: .contiguous() is required to avoid flex_attention autotuning failures
                # Note: permute is not needed for a significant perf hit in PyTorch version >=2.8, we will
                #    remove it in the next version
                query, key, value = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
                hidden_states = flex_attention(query=query, key=key, value=value, block_mask=block_mask)  # type: ignore
                hidden_states = hidden_states.permute(0, 2, 1, 3)

                # Remove padding if it was added
                if padded_len > 0:
                    hidden_states = hidden_states[:, :-padded_len, :, :]

            else:
                # Standard self-attention (no caching, no FlexAttention)
                hidden_states = dispatch_attention_fn(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _wan_block_forward_inline_cache(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    norm_temb: bool,
    block_mask: BlockMask | None = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    cache_start_idx: Optional[int] = None,
) -> torch.Tensor:
    assert temb.ndim == 4, f"temb.ndim should be 4 in causal networks, got a shape {temb.shape}"
    # temb: batch_size, num_frames, 6, inner_dim -> batch_size, num_frames, 1, inner_dim
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        self.scale_shift_table.unsqueeze(0) + temb.float()
    ).chunk(6, dim=2)

    # Follow sCM to normalize the AdaLN condition
    if norm_temb:
        shift_msa = normalize(shift_msa)
        scale_msa = normalize(scale_msa)
        c_shift_msa = normalize(c_shift_msa)
        c_scale_msa = normalize(c_scale_msa)

    attention_kwargs = attention_kwargs or {}
    cache_tag = attention_kwargs.get("cache_tag", "pos")
    store_kv = attention_kwargs.get("store_kv", False)
    external_self_kv = attention_kwargs.get("external_self_kv", {})
    external_cross_kv = attention_kwargs.get("external_cross_kv", {})

    # Per-frame modulation: reshape tokens into [B, F, T, C]
    seq_len = hidden_states.shape[1]
    num_frames = scale_msa.shape[1]
    frame_seqlen = seq_len // num_frames

    # 1. Self-attention
    norm_hidden_states = (
        (self.norm1(hidden_states.float()).unflatten(1, (num_frames, frame_seqlen)) * (1 + scale_msa) + shift_msa)
        .flatten(1, 2)
        .type_as(hidden_states)
    )
    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=rotary_emb,
        block_mask=block_mask,
        cache_tag=cache_tag,
        store_kv=store_kv,
        attention_cache_kwargs={"external_self_kv": external_self_kv},
        cache_start_idx=cache_start_idx,
    )
    hidden_states = (
        hidden_states.float() + (attn_output.unflatten(1, (num_frames, frame_seqlen)) * gate_msa).flatten(1, 2)
    ).type_as(hidden_states)

    # 2. Cross-attention
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=None,
        rotary_emb=None,
        block_mask=block_mask,
        cache_tag=cache_tag,
        store_kv=store_kv,
        attention_cache_kwargs={"external_cross_kv": external_cross_kv},
    )
    hidden_states = hidden_states + attn_output

    # 3. Feed-forward
    norm_hidden_states = (
        (self.norm3(hidden_states.float()).unflatten(1, (num_frames, frame_seqlen)) * (1 + c_scale_msa) + c_shift_msa)
        .flatten(1, 2)
        .type_as(hidden_states)
    )
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (
        hidden_states.float() + (ff_output.float().unflatten(1, (num_frames, frame_seqlen)) * c_gate_msa).flatten(1, 2)
    ).type_as(hidden_states)

    return hidden_states


def _wan_set_attn_processor(self, processor) -> None:
    """
    Sets the attention processor to use to compute attention.

    Parameters:
        processor (only `AttentionProcessor`):
            The instantiated processor class that will be set as the processor for all `Attention` layers.
    """
    assert not isinstance(processor, dict), "processor should be a single AttentionProcessor, rather than a dict"

    def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor) -> None:
        if hasattr(module, "set_processor"):
            module.set_processor(processor)

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in self.named_children():
        fn_recursive_attn_processor(name, module, processor)


def classify_forward_prepare(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    prepare the input for the block forward passes inside the classify_forward function.
    will be used as the prepare_op function in the classify_forward function.

    args:
        self: WanTransformer3DModel
        hidden_states: [B, C, F, H, W]
        timestep: [B]
        encoder_hidden_states: [B, L_c, D_c]
        r_timestep: [B]
        encoder_hidden_states_image: [B, L_img, D_img]
        attention_kwargs: Optional[Dict[str, Any]] = None

    returns:
        hidden_states: [B, N_tokens, D_model], where N_tokens=(F/p_t)*(H/p_h)*(W/p_w)
        timestep_proj: [B, 6, N_frames * H * W, D_model] if Wan2.2 TI2V 5B else [B, 6, N_frames, D_model],
            where N_frames=F/p_t
        encoder_hidden_states: [B, L_c, D_model]
        encoder_hidden_states_image: [B, L_img, D_model] if provided else None
        temb: [B, N_frames * H * W, D_model] if Wan2.2 TI2V 5B else [B, N_frames, D_model]
        rotary_emb: [B, N_tokens, D_model]
    """
    batch_size, _, num_frames, height, width = hidden_states.shape
    _, p_h, p_w = self.config.patch_size
    frame_seqlen = (height // p_h) * (width // p_w)

    attention_kwargs = attention_kwargs or {}
    chunk_size = attention_kwargs.get("chunk_size", 3)
    cache_tag = attention_kwargs.get("cache_tag", "pos")
    cur_start_frame = attention_kwargs.get("cur_start_frame", 0)
    total_num_frames = attention_kwargs.get("total_num_frames", 21)
    attention_kwargs["frame_seqlen"] = frame_seqlen

    # Use causal rotary embedding
    if cur_start_frame == 0:
        cached_tokens = 0
        kv_container = getattr(self, "_external_self_kv_list", None)
        if isinstance(kv_container, list) and len(kv_container) == len(self.blocks):
            for per_block in kv_container:
                if not isinstance(per_block, dict):
                    continue
                tag_entry = per_block.get(cache_tag, None)
                if isinstance(tag_entry, dict):
                    cached_tokens = int(tag_entry.get("len", 0))
                    if cached_tokens > 0:
                        break
        cur_start_frame = cached_tokens // frame_seqlen
    rotary_emb = self.rope(hidden_states, start_frame=cur_start_frame)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
    timestep, ts_seq_len = flatten_timestep(timestep)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )
    timestep_proj = unflatten_timestep_proj(timestep_proj, ts_seq_len)

    if self.r_embedder is not None and r_timestep is not None:
        if self.time_cond_type == "abs":
            r_timestep = r_timestep
        elif self.time_cond_type == "diff":
            r_timestep = timestep - r_timestep
        else:
            raise ValueError(f"Invalid time condition: {self.time_cond_type}")

        r_timestep, rs_seq_len = flatten_timestep(r_timestep)
        r_timestep = self.r_embedder.timesteps_proj(r_timestep)
        time_embedder_dtype = next(iter(self.r_embedder.time_embedder.parameters())).dtype
        if r_timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            r_timestep = r_timestep.to(time_embedder_dtype)

        remb = self.r_embedder.time_embedder(r_timestep).type_as(encoder_hidden_states)
        r_timestep_proj = self.r_embedder.time_proj(self.r_embedder.act_fn(remb))
        r_timestep_proj = unflatten_timestep_proj(r_timestep_proj, rs_seq_len)

        if self.encoder_depth is None:
            timestep_proj = timestep_proj + r_timestep_proj
            temb = temb + remb
        else:
            temb = remb
    else:
        r_timestep_proj = None

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    # Construct block mask once only when:
    # 1) teacher-forcing or diffusion-forcing is enabled (e.g., num_frames == total_num_frames)
    # 2) FlexAttention is available
    if getattr(self, "block_mask", None) is None and num_frames == total_num_frames and FLEX_ATTENTION_AVAILABLE:
        self.block_mask = self._prepare_blockwise_causal_attn_mask(
            hidden_states.device,
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            chunk_size=chunk_size,
        )

    # Preallocate external KV buffers per block when self-forcing (or student-forcing) is enabled
    if num_frames < total_num_frames:
        self._create_external_caches(
            hidden_states,
            encoder_hidden_states,
            num_frames=num_frames,
            total_num_frames=total_num_frames,
            cache_tag=cache_tag,
            cur_start_frame=cur_start_frame,
        )

    return (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
        temb,
        rotary_emb,
    )


def _create_external_caches(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    num_frames: int,
    total_num_frames: int = 21,
    cache_tag: str = "pos",
    cur_start_frame: int = 0,
) -> None:
    """
    Create and initialize external KV caches for causal attention processing.

    This method preallocates KV buffers for all transformer blocks to enable efficient
    causal generation with append-style caching. It creates separate cache structures
    for self-attention (growing buffers) and cross-attention (static buffers).

    The caches are stored in:
    - self._external_self_kv_list: List of per-block self-attention caches
    - self._external_cross_kv_list: List of per-block cross-attention caches

    Cache Structure:
    - Self-attention cache: {"k": tensor, "v": tensor, "len": current_length}
    - Cross-attention cache: {"k": tensor, "v": tensor, "is_init": bool}

    Args:
        hidden_states: Input hidden states [B, N_tokens, D_model] used for shape inference
        encoder_hidden_states: Text/image conditioning states [B, L_enc, D_enc] for cross-attention cache sizing
        num_frames: Current number of frames being processed
        total_num_frames: Total expected frames for the full sequence (default: 21)
        cache_tag: Cache identifier for this generation session ("pos", "neg", etc.)
        cur_start_frame: Starting frame index for resuming generation (default: 0)

    Note:
        - Caches are automatically reallocated if shapes change between calls
        - Each transformer block gets its own separate cache instance
        - Self-attention caches grow dynamically as frames are added
        - Cross-attention caches are static (text conditioning doesn't change)
    """
    bsz = hidden_states.shape[0]
    frame_seqlen = hidden_states.shape[1] // num_frames if num_frames > 0 else hidden_states.shape[1]
    capacity_tokens = frame_seqlen * total_num_frames

    if not hasattr(self, "_external_self_kv_list"):
        self._external_self_kv_list = []
    if not hasattr(self, "_external_cross_kv_list"):
        self._external_cross_kv_list = []
    self_list = self._external_self_kv_list
    cross_list = self._external_cross_kv_list

    # Check if batch size changed - need to reallocate caches
    need_reinit = len(self_list) != len(self.blocks) or len(cross_list) != len(self.blocks)
    if not need_reinit and len(self_list) > 0 and self_list[0] is not None:
        # Check if existing cache has different batch size
        existing_cache = self_list[0].get(cache_tag, {}).get("k", None)
        if existing_cache is not None and existing_cache.shape[0] != bsz:
            logger.info(f"Batch size changed from {existing_cache.shape[0]} to {bsz}, reallocating caches")
            need_reinit = True

    if need_reinit:
        logger.info("creating external caches when self-forcing (or student-forcing) is enabled")
        self_list.clear()
        cross_list.clear()
        for _ in self.blocks:
            self_list.append({})
            cross_list.append({})

    if any((cache_tag not in cache for cache in self_list)):
        for i, blk in enumerate(self.blocks):
            heads = getattr(blk.attn1, "heads", None)
            kv_inner_dim = (
                blk.attn1.to_k.out_features
                if hasattr(blk.attn1, "to_k") and hasattr(blk.attn1.to_k, "out_features")
                else None
            )
            if heads is None or kv_inner_dim is None:
                heads, head_dim = 12, 128
            else:
                head_dim = kv_inner_dim // int(heads)
                heads = int(heads)

            want_shape = (bsz, capacity_tokens, heads, head_dim)
            need_alloc = False
            if cache_tag not in self_list[i]:
                need_alloc = True
            else:
                k_buf = self_list[i][cache_tag].get("k", None)
                v_buf = self_list[i][cache_tag].get("v", None)
                if (
                    not isinstance(k_buf, torch.Tensor)
                    or not isinstance(v_buf, torch.Tensor)
                    or tuple(k_buf.shape) != want_shape
                    or tuple(v_buf.shape) != want_shape
                ):
                    need_alloc = True
                    logger.warning(
                        f"Self-attn KV cache shape mismatch for block {i}, reallocating. Expected shape={want_shape}"
                    )
            if need_alloc:
                logger.debug(f"Allocating KV cache for {cache_tag} tag")
                want_shape_cross = (bsz, encoder_hidden_states.shape[1], heads, head_dim)
                device = hidden_states.device
                dtype = hidden_states.dtype
                self_list[i][cache_tag] = {
                    "k": torch.zeros(want_shape, device=device, dtype=dtype),
                    "v": torch.zeros(want_shape, device=device, dtype=dtype),
                    "len": cur_start_frame * frame_seqlen,
                }
                cross_list[i][cache_tag] = {
                    "k": torch.zeros(want_shape_cross, device=device, dtype=dtype),
                    "v": torch.zeros(want_shape_cross, device=device, dtype=dtype),
                    "is_init": False,
                }


def classify_forward_block_forward(
    self,
    hidden_states: torch.Tensor,
    timestep_proj: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    r_timestep_proj: Optional[torch.Tensor] = None,
    skip_layers: Optional[List[int]] = None,
    feature_indices: Optional[Set[int]] = None,
    return_features_early: Optional[bool] = False,
    lora_scale: Optional[float] = 1.0,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    block forward pass inside the classify_forward function, wired to the inline-cache processor.
    will be used as the block_forward_op function in the classify_forward function.

    args:
        self: WanTransformer3DModel
        hidden_states: [B, N_tokens, D_model]
        timestep_proj: [B, 6, N_frames * H * W, D_model] if Wan2.2 TI2V 5B else [B, 6, N_frames, D_model],
            where N_frames=F/p_t
        encoder_hidden_states: [B, L_c, D_model]
        rotary_emb: [B, N_tokens, D_model]
        r_timestep_proj: [B, 6, N_frames * H * W, D_model] if Wan2.2 TI2V 5B else [B, 6, N_frames, D_model],
            where N_frames=F/p_t
        skip_layers: Optional[List[int]]
        feature_indices: Optional[Set[int]]
        attention_kwargs: Optional[Dict[str, Any]]

    returns:
        hidden_states: [B, N_tokens, D_model]
        features: List[[B, N_tokens, D_model]]
    """
    attention_kwargs = attention_kwargs or {}
    is_ar = attention_kwargs.get("is_ar", False)
    cache_tag = attention_kwargs.get("cache_tag", "pos")
    cur_start_frame = attention_kwargs.get("cur_start_frame", 0)
    frame_seqlen = attention_kwargs.get("frame_seqlen", 0)
    current_block_mask = None if is_ar else self.block_mask

    features = []
    for idx, block in enumerate(self.blocks):
        if skip_layers is not None and idx in skip_layers:
            continue
        if self.encoder_depth is not None and idx == self.encoder_depth and r_timestep_proj is not None:
            timestep_proj = r_timestep_proj

        # Important!!! Reset cache length to proper position and create isolated snapshots
        attn_snapshot_kwargs = dict(attention_kwargs)
        proper_cache_len = None
        if hasattr(self, "_external_self_kv_list") and idx < len(self._external_self_kv_list):
            original_cache = self._external_self_kv_list[idx]
            if cache_tag in original_cache:
                # Reset cache length to current frame position (not accumulated!)
                proper_cache_len = cur_start_frame * frame_seqlen
                original_cache[cache_tag]["len"] = proper_cache_len

                # Create isolated snapshot with correct length
                stable_cache = {
                    cache_tag: {
                        "k": original_cache[cache_tag]["k"],
                        "v": original_cache[cache_tag]["v"],
                        "len": proper_cache_len,  # Same corrected length
                    }
                }
                attn_snapshot_kwargs["external_self_kv"] = stable_cache
            else:
                attn_snapshot_kwargs["external_self_kv"] = original_cache

        if hasattr(self, "_external_cross_kv_list") and idx < len(self._external_cross_kv_list):
            attn_snapshot_kwargs["external_cross_kv"] = self._external_cross_kv_list[idx]

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # We need to fix the start, as store_kv=True will update the length
            block_fixed_start = partial(block, cache_start_idx=proper_cache_len)
            hidden_states = self._gradient_checkpointing_func(
                block_fixed_start,
                hidden_states,
                timestep_proj,
                encoder_hidden_states,
                rotary_emb,
                self.norm_temb,
                current_block_mask,
                attn_snapshot_kwargs,
            )
        else:
            # Always pass cache_start_idx explicitly to avoid reading from mutable cache dict
            hidden_states = block(
                hidden_states,
                timestep_proj,
                encoder_hidden_states,
                rotary_emb,
                self.norm_temb,
                current_block_mask,
                attn_snapshot_kwargs,
                cache_start_idx=proper_cache_len,
            )
        if feature_indices is not None and idx in feature_indices:
            features.append(hidden_states)

        # If we have all the features, we can exit early
        if return_features_early and len(features) == len(feature_indices):
            if USE_PEFT_BACKEND:
                # Clean up LoRA scaling before early exit
                unscale_lora_layers(self, lora_scale)
            return hidden_states, features

    return hidden_states, features


class CausalWan(CausalFastGenNetwork, Wan):
    """
    Wan with inline-caching attention (internal per-attn caches, append-style).

    This class inherits from both CausalFastGenNetwork (for causal generation capabilities)
    and Wan (for the Wan transformer architecture). It provides causal inference with
    KV caching for efficient autoregressive generation.
    """

    def __init__(
        self,
        model_id_or_local_path: str = Wan.MODEL_ID_1_3B,
        r_timestep: bool = False,
        disable_efficient_attn: bool = False,
        disable_grad_ckpt: bool = False,
        enable_logvar_linear: bool = True,
        r_embedder_init: str = "pretrained",
        time_cond_type: str = "diff",
        norm_temb: bool = False,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        encoder_depth: int | None = None,
        load_pretrained: bool = True,
        use_fsdp_checkpoint: bool = True,
        chunk_size: int = 3,
        total_num_frames: int = 21,
        delete_cache_on_clear: bool = False,
        **model_kwargs,
    ) -> None:
        """Causal Wan model constructor.

        Args:
            model_id_or_local_path (str, optional): The huggingface model ID or local path to load.
                Defaults to "Wan-AI/Wan2.1-T2V-1.3B-Diffusers".
            r_timestep (bool): Whether to support meanflow-like models with r timestep. Defaults to False.
            disable_efficient_attn (bool, optional): Whether to disable efficient attention. Defaults to False.
            disable_grad_ckpt (bool, optional): Whether to disable checkpoints during training. Defaults to False.
            enable_logvar_linear (bool, optional): Whether to enable logvar linear prediction. Defaults to True.
            r_embedder_init (str, optional): Initialization method for the r embedder. Defaults to "pretrained".
            time_cond_type (str, optional): Time condition type for r timestep. Defaults to "diff".
            norm_temb (bool, optional): Whether to normalize the time embeddings. Defaults to False.
            net_pred_type (str, optional): Prediction type. Defaults to "flow".
            schedule_type (str, optional): Schedule type. Defaults to "rf".
            encoder_depth (int, optional): The depth of the encoder (i.e. the number of blocks taking in t embeddings).
                Defaults to None, meaning all blocks take in [t embeddings + r embeddings].
            load_pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
            use_fsdp_checkpoint (bool, optional): Whether to use FSDP gradient checkpointing. Defaults to True.
            chunk_size (int, optional): The chunk size for the transformer. Defaults to 3.
            total_num_frames (int, optional): The total number of frames. Defaults to 21.
            delete_cache_on_clear (bool, optional): Whether to delete the cache on clear. Defaults to False.
            **model_kwargs: Additional keyword arguments to pass to the FastGenNetwork constructor.
        """
        super().__init__(
            model_id_or_local_path=model_id_or_local_path,
            r_timestep=r_timestep,
            disable_efficient_attn=disable_efficient_attn,
            disable_grad_ckpt=disable_grad_ckpt,
            enable_logvar_linear=enable_logvar_linear,
            r_embedder_init=r_embedder_init,
            time_cond_type=time_cond_type,
            norm_temb=norm_temb,
            encoder_depth=encoder_depth,
            load_pretrained=load_pretrained,
            use_fsdp_checkpoint=use_fsdp_checkpoint,
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            chunk_size=chunk_size,
            total_num_frames=total_num_frames,
            **model_kwargs,
        )
        self._delete_cache_on_clear = delete_cache_on_clear

    def override_transformer_forward(self, inner_dim: int) -> None:
        # Patch rope offset and helper methods
        if hasattr(self.transformer, "rope"):
            self.transformer.rope.forward = types.MethodType(_rope_forward_with_time_offset, self.transformer.rope)
        self.transformer._prepare_blockwise_causal_attn_mask = types.MethodType(
            _prepare_blockwise_causal_attn_mask, self.transformer
        )
        self.transformer._create_external_caches = types.MethodType(_create_external_caches, self.transformer)
        self.transformer.set_attn_processor = types.MethodType(_wan_set_attn_processor, self.transformer)
        self.transformer.classify_forward_prepare = types.MethodType(classify_forward_prepare, self.transformer)
        self.transformer.classify_forward_block_forward = types.MethodType(
            classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Patch block forward to inline-cache path
        for block in self.transformer.blocks:
            block.forward = types.MethodType(_wan_block_forward_inline_cache, block)

        # Install attention processor with inline caches
        self.transformer.set_attn_processor(CausalWanAttnProcessor())
        self.transformer.block_mask = None

        # Override timesteps_proj to use official WAN sinusoidal embedding (if configured)
        if self._use_wan_official_sinusoidal:
            logger.info("Using official WAN sinusoidal embedding")
            self._override_timesteps_proj(self.transformer.condition_embedder.timesteps_proj)
            if self.transformer.r_embedder is not None:
                self._override_timesteps_proj(self.transformer.r_embedder.timesteps_proj)

    def clear_caches(self) -> None:
        if self._delete_cache_on_clear:
            if hasattr(self.transformer, "_external_self_kv_list"):
                del self.transformer._external_self_kv_list
            if hasattr(self.transformer, "_external_cross_kv_list"):
                del self.transformer._external_cross_kv_list
            torch.cuda.empty_cache()
            return

        # Reset external KV lengths if present (mapping by tag)
        if hasattr(self.transformer, "_external_self_kv_list"):
            for kvb in self.transformer._external_self_kv_list:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["len"] = 0
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])
        # Reset external cross-attn cache init flags if present (mapping by tag)
        if hasattr(self.transformer, "_external_cross_kv_list"):
            for kvb in self.transformer._external_cross_kv_list:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["is_init"] = False
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])

    def _compute_timestep_inputs(self, timestep: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute timestep input used for Causal Wan models.
        Optionally Expand or mask the timestep input for Wan 2.2 TI2V models.
            - I2V: Apply a mask that zeroes out the timestep for the first latent frame.
            - T2V: Use a mask tensor filled with ones.
        Different from Wan/network.py that expands timestep to [B, num_tokens] for Wan 2.2 TI2V models only,
            we expand timestep to [B, num_frames] for all Wan models to perform causal training/inference.

        Args:
            timestep (torch.Tensor): shape: (B, T) or (B, )
            mask (torch.Tensor): shape: (B, T, H, W)

        Return:
            timestep (torch.Tensor): shape: (B, T)
        """
        timestep = self.noise_scheduler.rescale_t(timestep)

        p_t, _, _ = self.transformer.config.patch_size
        if timestep.ndim == 1:
            timestep = timestep.view(-1, 1)
        timestep = mask[:, ::p_t, 0, 0].to(timestep.dtype) * timestep
        return timestep

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        unpatchify_features: bool = True,
        fwd_pred_type: Optional[str] = None,
        skip_layers: Optional[List[int]] = None,
        cache_tag: str = "pos",
        cur_start_frame: int = 0,
        store_kv: bool = False,
        is_ar: bool = False,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the Wan diffusion score model.

        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (Dict[str, torch.Tensor]): The conditioning information. Defaults to None.
            r (torch.Tensor): Another timestep mainly used by meanflow.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. If the set is non-empty, then features will be returned.
                By default, feature_indices=None means extract no features.
            return_logvar: If true, the forward pass returns the logvar.
            unpatchify_features: If true, the features will be unpatchified and returned in shape of [B, T, H, W, C].
            fwd_pred_type: Update the network prediction type, must be in ['x0', 'eps', 'v', 'flow'].
                None means using the original net_pred_type.
            skip_layers: Apply skip-layer guidance by skipping layers of the unconditional network during forward pass.

        Returns:
            torch.Tensor: The score model output.
        """
        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            # Exit immediately if user requested this.
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"

        condition = torch.stack(condition, dim=0) if isinstance(condition, list) else condition
        timestep_mask = torch.ones_like(x_t[:, 0])  # shape: [batch_size, num_latent_frames, H, W]
        timestep = self._compute_timestep_inputs(t, timestep_mask)  # shape: [batch_size, num_latent_frames]
        r_timestep = None if r is None else self._compute_timestep_inputs(r, timestep_mask)

        attention_kwargs = {
            "cache_tag": cache_tag,
            "chunk_size": self.chunk_size,
            "store_kv": store_kv,
            "is_ar": is_ar,
            "cur_start_frame": cur_start_frame,
            "total_num_frames": self.total_num_frames,
        }

        model_outputs = self.transformer(
            hidden_states=x_t,
            timestep=timestep,
            encoder_hidden_states=condition,
            r_timestep=r_timestep,
            attention_kwargs=attention_kwargs,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            skip_layers=skip_layers,
        )

        if return_features_early:
            assert len(model_outputs) == len(feature_indices)
            return self._unpatchify_features(x_t, model_outputs) if unpatchify_features else model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        # Handle both 1D and 2D timestep formats for noise scheduler
        t_converted = t[:, None, :, None, None] if t.ndim == 2 else t

        if len(feature_indices) == 0:
            assert isinstance(out, torch.Tensor)
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t_converted, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            assert isinstance(out, list)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t_converted, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            out[1] = self._unpatchify_features(x_t, out[1]) if unpatchify_features else out[1]

        if return_logvar:
            return out, logvar
        return out

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 5.0,
        sample_steps: Optional[int] = 50,
        shift: float = 5.0,
        context_noise: float = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive sampling for teacher network with CFG and unipc_scheduler

        Args:
            noise (torch.Tensor): The pure noise to start from (usually a zero-mean, unit-variance Gaussian)
            condition (Optional[Dict[str, torch.Tensor]]): Conditioning information. Defaults to None.
            neg_condition (Optional[Dict[str, torch.Tensor]]): Optional negative conditioning information. Defaults to None.
            guidance_scale (Optional[float]): Scale of guidance. None means no guidance. Defaults to 5.0.
            sample_steps (Optional[int]): Number of time steps to sample. Defaults to 4.
            shift (Optional[float]): Shift value of timestep scheduler. Defaults to 5.0.
            context_noise (Optional[float]): Scale of context noise in the range [0, 1]. Defaults to 0.

        Returns:
            torch.Tensor: The sample output.
        """
        assert self.schedule_type == "rf", f"{self.schedule_type} is not supported"
        # Ensure caches are clean at the start of sampling
        self.clear_caches()

        batch_size = noise.shape[0]
        num_frames = noise.shape[2]

        num_chunks = num_frames // self.chunk_size
        remaining_size = num_frames % self.chunk_size
        time_rescale_factor = self.unipc_scheduler.config.num_train_timesteps

        # Create initial latents
        self.unipc_scheduler.config.flow_shift = shift
        self.unipc_scheduler.set_timesteps(num_inference_steps=sample_steps, device=noise.device)
        timesteps = self.unipc_scheduler.timesteps
        t_init = timesteps[0] / time_rescale_factor
        x = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        for i in range(max(1, num_chunks)):
            if num_chunks == 0:
                start, end = 0, remaining_size
            else:
                start = 0 if i == 0 else self.chunk_size * i + remaining_size
                end = self.chunk_size * (i + 1) + remaining_size

            x_next = x[:, :, start:end]
            # Reset scheduler state (model_outputs, lower_order_nums, etc.)
            self.unipc_scheduler.set_timesteps(num_inference_steps=sample_steps, device=noise.device)
            for timestep in tqdm(timesteps, total=sample_steps - 1):
                t = (timestep / time_rescale_factor).expand(batch_size)
                x_cur = x_next
                flow_pred = self(
                    x_cur,
                    t,
                    condition=condition,
                    cache_tag="pos",  # kv-cache positive
                    cur_start_frame=start,
                    store_kv=False,
                    is_ar=True,
                )
                if guidance_scale is not None:
                    flow_uncond = self(
                        x_cur,
                        t,
                        condition=neg_condition,
                        cache_tag="neg",  # kv-cache negative
                        cur_start_frame=start,
                        store_kv=False,
                        is_ar=True,
                    )
                    flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)
                x_next = self.unipc_scheduler.step(flow_pred, timestep, x_next, return_dict=False)[0]

            x[:, :, start:end] = x_next

            # rerun with timestep zero or context_noise to finalize the cache slice
            x_cache = x_next
            t_cache = torch.full((batch_size,), 0, device=x.device, dtype=x.dtype)
            if context_noise > 0:
                # Add context noise to denoised frames before caching
                t_cache = torch.full((batch_size,), context_noise, device=x.device, dtype=x.dtype)
                x_cache = self.noise_scheduler.forward_process(x_next, torch.randn_like(x_next), t_cache)

            _ = self(
                x_cache,
                t_cache,
                condition=condition,
                cache_tag="pos",
                cur_start_frame=start,
                store_kv=True,
                is_ar=True,
            )
            if guidance_scale is not None:
                _ = self(
                    x_cache,
                    t_cache,
                    condition=neg_condition,
                    cache_tag="neg",
                    cur_start_frame=start,
                    store_kv=True,
                    is_ar=True,
                )

        # cleanup caches after full sampling
        self.clear_caches()
        return x
