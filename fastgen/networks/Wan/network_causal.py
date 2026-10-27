# SPDX-FileCopyrightText: Copyright (c) 2025 The Wan Team and The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from typing import Optional, List, Set, Tuple, Union
import types

from tqdm.auto import tqdm
import math
import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import (
    WanAttnProcessor,
    WanAttention,
    _get_qkv_projections,
    _get_added_kv_projections,
)
from fastgen.networks.kv_cache import FLEX_ATTENTION_AVAILABLE, KVCacheEngine, flex_attention
from fastgen.networks.lora_utils import scaled_lora_layers
from fastgen.networks.network_causal import CausalContext, CausalFastGenNetwork, causal_chunk_ranges
from fastgen.networks.Wan.network import (
    Wan,
    apply_r_embedding,
    flatten_timestep,
    normalize,
    project_logvar,
    project_output,
    unflatten_timestep_proj,
)
from fastgen.utils.tree_utils import OutputDtype, assert_same
from fastgen.utils.guidance import (
    GuidanceConfig,
    predict_with_guidance,
)
import fastgen.utils.logging_utils as logger


TEXT_ENCODER_CONTEXT_LENGTH = 512  # Wan's text encoder context; image tokens (Wan2.1 I2V) precede it


def _get_cross_kv_projections(
    attn: WanAttention, encoder_hidden_states: torch.Tensor, image_context_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized K/V ``[B, L_enc, H, D]`` of a whole cross-attention context, in context order.

    The first ``image_context_length`` tokens are Wan2.1 I2V image tokens and go through ``add_k_proj`` /
    ``add_v_proj``; the rest are text tokens. The causal cross cache stores this pair unchanged, so it is sized
    by the full context length.
    """
    text_states = encoder_hidden_states[:, image_context_length:]
    if attn.fused_projections:
        key, value = attn.to_kv(text_states).chunk(2, dim=-1)
    else:
        key, value = attn.to_k(text_states), attn.to_v(text_states)
    key = attn.norm_k(key)
    if image_context_length > 0:
        key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states[:, :image_context_length])
        key = torch.cat([attn.norm_added_k(key_img), key], dim=1)
        value = torch.cat([value_img, value], dim=1)
    return key.unflatten(2, (attn.heads, -1)), value.unflatten(2, (attn.heads, -1))


def _get_kv_cache_spec_for_blocks(
    blocks: torch.nn.ModuleList,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    num_frames: int,
    total_num_frames: int,
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Build the exact KV cache spec for a Wan-style block collection."""
    attn1 = blocks[0].attn1
    heads = attn1.heads
    head_dim = attn1.to_k.out_features // heads
    frame_seqlen = hidden_states.shape[1] // num_frames
    self_shape = (hidden_states.shape[0], frame_seqlen * total_num_frames, heads, head_dim)
    cross_shape = (hidden_states.shape[0], encoder_hidden_states.shape[1], heads, head_dim)
    return len(blocks), self_shape, cross_shape


def block_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    norm_temb: bool,
    causal_ctx: Optional[CausalContext] = None,
) -> torch.Tensor:
    """Causal Wan transformer block forward with direct causal_ctx threading."""
    core_dtype = encoder_hidden_states.dtype

    if temb.ndim == 4:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
    else:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

    if norm_temb:
        shift_msa = normalize(shift_msa)
        scale_msa = normalize(scale_msa)
        c_shift_msa = normalize(c_shift_msa)
        c_scale_msa = normalize(c_scale_msa)

    seq_len = hidden_states.shape[1]
    per_frame = temb.ndim == 4 and scale_msa.shape[1] != seq_len
    if per_frame:
        num_frames = scale_msa.shape[1]
        frame_seqlen = seq_len // num_frames

    # 1. Self-attention
    if per_frame:
        norm_hidden_states = (
            (self.norm1(hidden_states).float().unflatten(1, (num_frames, frame_seqlen)) * (1 + scale_msa) + shift_msa)
            .flatten(1, 2)
            .to(core_dtype)
        )
    else:
        if temb.ndim == 4:
            scale_msa = scale_msa.squeeze(2)
            shift_msa = shift_msa.squeeze(2)
        norm_hidden_states = (self.norm1(hidden_states).float() * (1 + scale_msa) + shift_msa).to(core_dtype)

    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=rotary_emb,
        causal_ctx=causal_ctx,
    )

    if per_frame:
        hidden_states = hidden_states.float() + (
            attn_output.unflatten(1, (num_frames, frame_seqlen)) * gate_msa
        ).flatten(1, 2)
    else:
        if temb.ndim == 4:
            gate_msa = gate_msa.squeeze(2)
        hidden_states = hidden_states.float() + attn_output * gate_msa

    # 2. Cross-attention
    norm_hidden_states = self.norm2(hidden_states.float()).to(core_dtype)
    attn_output = self.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states.to(core_dtype),
        attention_mask=None,
        rotary_emb=None,
        causal_ctx=causal_ctx,
    )
    hidden_states = hidden_states + attn_output

    # 3. Feed-forward
    if per_frame:
        norm_hidden_states = (
            (
                self.norm3(hidden_states.float()).unflatten(1, (num_frames, frame_seqlen)) * (1 + c_scale_msa)
                + c_shift_msa
            )
            .flatten(1, 2)
            .to(core_dtype)
        )
    else:
        if temb.ndim == 4:
            c_scale_msa = c_scale_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).to(core_dtype)
    ff_output = self.ffn(norm_hidden_states)

    if per_frame:
        hidden_states = hidden_states.float() + (
            ff_output.float().unflatten(1, (num_frames, frame_seqlen)) * c_gate_msa
        ).flatten(1, 2)
    else:
        if temb.ndim == 4:
            c_gate_msa = c_gate_msa.squeeze(2)
        hidden_states = hidden_states.float() + ff_output.float() * c_gate_msa

    if not getattr(self, "fp32_residual", True):
        hidden_states = hidden_states.to(core_dtype)
    return hidden_states


def compute_rotary_emb_with_time_offset(
    rope_module: torch.nn.Module, hidden_states: torch.Tensor, start_frame: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Wan rotary embeddings with a temporal offset."""
    _, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = rope_module.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    split_sizes = [rope_module.t_dim, rope_module.h_dim, rope_module.w_dim]
    freqs_cos_t, freqs_cos_h, freqs_cos_w = rope_module.freqs_cos.split(split_sizes, dim=1)
    freqs_sin_t, freqs_sin_h, freqs_sin_w = rope_module.freqs_sin.split(split_sizes, dim=1)

    total_f = freqs_cos_t.shape[0]
    start = max(0, start_frame)
    end = start + ppf
    if start >= total_f:
        cos_f = freqs_cos_t[-1:].expand(ppf, -1)
        sin_f = freqs_sin_t[-1:].expand(ppf, -1)
    else:
        end = min(end, total_f)
        cos_f, sin_f = freqs_cos_t[start:end], freqs_sin_t[start:end]
        if cos_f.shape[0] < ppf:
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


def classify_forward_prepare(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    causal_ctx: Optional[CausalContext] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare inputs for causal classify_forward, including cache/mask setup when provided.

    For teacher forcing (``causal_ctx.mask_kind == "teacher_forcing"``),
    ``hidden_states``/``timestep`` carry the clean context frames concatenated
    with the noisy target frames; the two halves share frame positions, so the
    masks and RoPE are built for half the input frames.
    """
    batch_size, _, num_frames, height, width = hidden_states.shape
    teacher_forcing_active = causal_ctx is not None and causal_ctx.mask_kind == "teacher_forcing"

    if teacher_forcing_active:
        if num_frames % 2 != 0:
            raise ValueError(
                f"Teacher forcing expects concatenated clean+noisy frames, got odd num_frames={num_frames}"
            )
        num_frames = num_frames // 2

    if causal_ctx is not None:
        if causal_ctx.get_engine("main") is None:
            # link the main engine to the causal context
            causal_ctx.set_engine("main", self.kv_cache_engines["main"])
        _, p_h, p_w = self.rope.patch_size
        frame_seqlen = (height // p_h) * (width // p_w)
        start_frame = causal_ctx.resolve_start_frame(frame_seqlen, engine_key="main")
        rope_states = hidden_states[:, :, :num_frames] if teacher_forcing_active else hidden_states
        rotary_emb = compute_rotary_emb_with_time_offset(self.rope, rope_states, start_frame)
        if teacher_forcing_active:
            # Clean and noisy halves represent the same latent frame/grid
            # positions, so duplicate RoPE rather than treating the noisy half
            # as later frames.
            rotary_emb = (
                torch.cat([rotary_emb[0], rotary_emb[0]], dim=1),
                torch.cat([rotary_emb[1], rotary_emb[1]], dim=1),
            )
    else:
        rotary_emb = self.rope(hidden_states)

    assert_same(
        device=True,
        hidden_states=hidden_states,
        freqs_cos=rotary_emb[0],
        freqs_sin=rotary_emb[1],
    )

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # Contiguous before the block loop: mirrors diffusers #14236 (~8% faster block loop).
    hidden_states = hidden_states.contiguous()

    timestep, ts_seq_len = flatten_timestep(timestep)
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )
    timestep_proj = unflatten_timestep_proj(timestep_proj, ts_seq_len)

    temb, timestep_proj, r_timestep_proj = apply_r_embedding(self, timestep, r_timestep, temb, timestep_proj)

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    if causal_ctx is not None:
        causal_ctx.attach_missing_engines(getattr(self, "kv_cache_engines", {}))

    if causal_ctx is not None and causal_ctx.is_ar:
        total_num_frames = causal_ctx.total_num_frames or num_frames
        block_count, self_shape, cross_shape = _get_kv_cache_spec_for_blocks(
            blocks=self.blocks,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            num_frames=num_frames,
            total_num_frames=total_num_frames,
        )
        causal_ctx.ensure_kv_cache_capacity(
            engine_key="main",
            block_count=block_count,
            self_shape=self_shape,
            cross_shape=cross_shape,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif causal_ctx is not None:
        causal_ctx.prepare_block_mask(device=hidden_states.device, num_frames=num_frames, engine_key="main")

    return (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        temb,
        rotary_emb,
    )


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
    causal_ctx: Optional[CausalContext] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run transformer blocks for causal classify_forward, with cache index stability for ckpt recompute."""

    features = []
    for idx, block in enumerate(self.blocks):
        if self.encoder_depth is not None and idx == self.encoder_depth and r_timestep_proj is not None:
            timestep_proj = r_timestep_proj
        # A skipped block acts as the identity; still record its feature so each requested index yields one.
        if skip_layers is None or idx not in skip_layers:
            causal_ctx_snapshot = causal_ctx.snapshot() if causal_ctx is not None else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def _block_with_ctx(
                    hidden_states_in: torch.Tensor,
                    timestep_proj_in: torch.Tensor,
                    encoder_hidden_states_in: torch.Tensor,
                    rotary_emb_in: torch.Tensor,
                    norm_temb_in: bool,
                    _block: torch.nn.Module = block,
                    _causal_ctx: Optional[CausalContext] = causal_ctx_snapshot,
                ) -> torch.Tensor:
                    return _block(
                        hidden_states_in,
                        timestep_proj_in,
                        encoder_hidden_states_in,
                        rotary_emb_in,
                        norm_temb_in,
                        causal_ctx=_causal_ctx,
                    )

                hidden_states = self._gradient_checkpointing_func(
                    _block_with_ctx,
                    hidden_states,
                    timestep_proj,
                    encoder_hidden_states,
                    rotary_emb,
                    self.norm_temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    timestep_proj,
                    encoder_hidden_states,
                    rotary_emb,
                    self.norm_temb,
                    causal_ctx=causal_ctx_snapshot,
                )

            if (
                causal_ctx_snapshot is not None
                and causal_ctx_snapshot.store_kv
                and causal_ctx_snapshot.grad_through_kv_cache
            ):
                cache_engine = causal_ctx_snapshot.get_engine("main")
                attn1 = getattr(block, "attn1", None)
                if cache_engine is not None and attn1 is not None and cache_engine.is_registered(attn1):
                    cache_engine.anchor_last_self_kv_to_output(
                        block_idx=cache_engine.resolve_block_idx(attn1),
                        cache_tag=causal_ctx_snapshot.cache_tag,
                        output=hidden_states,
                    )

        if feature_indices is not None and idx in feature_indices:
            features.append(hidden_states)
            if return_features_early and len(features) == len(feature_indices):
                break

    return hidden_states, features


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    causal_ctx: Optional[CausalContext] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
    skip_layers: Optional[List[int]] = None,
    **kwargs,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Causal transformer forward using explicit causal_ctx.

    For teacher forcing, ``hidden_states``/``timestep`` carry the clean context
    frames concatenated with the noisy target frames.
    """
    lora_scale = float(kwargs.get("scale", 1.0))

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    with scaled_lora_layers(self, lora_scale):
        (
            hidden_states,
            timestep_proj,
            r_timestep_proj,
            encoder_hidden_states,
            temb,
            rotary_emb,
        ) = self.classify_forward_prepare(
            hidden_states,
            timestep,
            encoder_hidden_states,
            r_timestep,
            encoder_hidden_states_image,
            causal_ctx=causal_ctx,
        )

        hidden_states, features = self.classify_forward_block_forward(
            hidden_states,
            timestep_proj,
            encoder_hidden_states,
            rotary_emb,
            r_timestep_proj,
            skip_layers,
            feature_indices,
            return_features_early,
            causal_ctx=causal_ctx,
        )

        if return_features_early:
            assert feature_indices is not None
            assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
            return features

        # 5. Output norm, projection & unpatchify
        if temb.dim() == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)

            if shift.shape[1] == hidden_states.shape[1]:
                hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
            else:
                frame_seqlen = post_patch_height * post_patch_width
                hs_norm_out = self.norm_out(hidden_states.float()).unflatten(1, (post_patch_num_frames, frame_seqlen))
                hidden_states = (
                    (hs_norm_out * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)).flatten(1, 2).type_as(hidden_states)
                )
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)
            hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = project_output(self, hidden_states, encoder_hidden_states.dtype)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        out: Union[torch.Tensor, List[torch.Tensor]] = [output, features] if feature_indices is not None else output

        if return_logvar:
            assert hasattr(
                self, "logvar_linear"
            ), "logvar_linear layer is required when return_logvar=True. Set enable_logvar_linear=True in model config."
            # One value per sample: average the time embeddings of the noised frames.
            logvar = project_logvar(self, temb, timestep, encoder_hidden_states.dtype)
            return out, logvar
        return out


class CausalWanAttnProcessor(WanAttnProcessor):
    """
    Causal attention processor for Wan transformer with self-forcing capabilities.

    This processor handles both self-attention and cross-attention with external KV caching
    for causal generation. It supports Image-to-Video (I2V) tasks, FlexAttention for
    efficient training, and gradient preservation for self-forcing training.

    Key Features:
    - External KV caching for causal autoregressive generation
    - Static cross-attention caching for the text and (Wan2.1 I2V) image conditioning
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
        causal_ctx: Optional[CausalContext] = None,
    ) -> torch.Tensor:
        """
        Forward pass for causal Wan attention with KV caching support.

        Args:
            attn: The WanAttention module to process
            hidden_states: Input hidden states [B, L, D]
            encoder_hidden_states: Optional conditioning states [B, L_enc, D] for cross-attention
            attention_mask: Optional attention mask for sequence masking
            rotary_emb: Optional rotary positional embeddings (cos, sin) tuple
            causal_ctx: CausalContext carrying cache-engine state for engine-based KV caching.
                Block mask for FlexAttention is read from the `"main"` branch.

        Returns:
            torch.Tensor: Processed attention output [B, L, D]
        """
        # Extract engine-based params from CausalContext (module identity resolves branch + block index).
        kv_cache_engine = None
        block_idx = None
        block_mask = None
        cache_tag = "pos"
        store_kv = False
        cache_start_idx = None
        grad_through_kv_cache = False
        if causal_ctx is not None:
            cache_tag = causal_ctx.cache_tag
            store_kv = causal_ctx.store_kv
            grad_through_kv_cache = causal_ctx.grad_through_kv_cache
            cache_start_idx = causal_ctx.cache_start_idx
            engine_key, kv_cache_engine, block_idx = causal_ctx.resolve_registered_engine(attn)
            if kv_cache_engine is None:
                engine_key = "main"
                kv_cache_engine = causal_ctx.get_engine(engine_key)
            if engine_key is not None:
                block_mask = causal_ctx.active_block_mask(engine_key)

        use_engine_cache = kv_cache_engine is not None and block_idx is not None

        is_cross_attn = encoder_hidden_states is not None  # Cross-attention when we have conditioning
        if is_cross_attn:
            # --- CROSS-ATTENTION PATH ---
            # Wan2.1 I2V blocks receive the image tokens followed by the text tokens
            image_context_length = 0
            if attn.add_k_proj is not None:
                image_context_length = max(encoder_hidden_states.shape[1] - TEXT_ENCODER_CONTEXT_LENGTH, 0)

            query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1)).type_as(hidden_states)

            cross_entry = None
            if use_engine_cache:
                assert kv_cache_engine is not None and block_idx is not None
                cross_entry = kv_cache_engine.get_cross_kv(block_idx=block_idx, cache_tag=cache_tag)
            if cross_entry is not None and cross_entry[2] and not torch.is_grad_enabled():
                # Use cached K/V only in no-grad phases
                key, value, _ = cross_entry
            else:
                key, value = _get_cross_kv_projections(attn, encoder_hidden_states, image_context_length)
                key = key.type_as(hidden_states)
                if store_kv and cross_entry is not None:
                    # Refresh cache with current K/V (allows grad flow when enabled)
                    kv_cache_engine.set_cross_kv(block_idx=block_idx, cache_tag=cache_tag, key=key, value=value)

            # Wan2.1 I2V attends the image and the text tokens with separate softmaxes, not one over the context
            hidden_states = dispatch_attention_fn(
                query,
                key[:, image_context_length:],
                value[:, image_context_length:],
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            if image_context_length > 0:
                hidden_states = hidden_states + dispatch_attention_fn(
                    query,
                    key[:, :image_context_length],
                    value[:, :image_context_length],
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                )

        else:
            # --- SELF-ATTENTION PATH ---
            # === Step 1: Compute and normalize QKV projections ===
            query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

            # Apply layer normalization to Q and K
            query = attn.norm_q(query)
            key = attn.norm_k(key)

            # Reshape for multi-head attention: (B, L, H, D) where H=num_heads, D=head_dim
            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))

            # === Step 2: Apply rotary positional embeddings ===
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

            # === Step 3: Compute attention ===
            bsz, seqlen, nheads, head_dim = query.shape  # query format: [B, L, H, D] after unflatten
            engine_self_kv = None
            if use_engine_cache:
                assert kv_cache_engine is not None and block_idx is not None
                engine_self_kv = kv_cache_engine.prepare_self_attention_kv(
                    block_idx=block_idx,
                    cache_tag=cache_tag,
                    requested_start_idx=cache_start_idx,
                    key=key,
                    value=value,
                    store_kv=store_kv,
                    allow_cache_grad=grad_through_kv_cache and torch.is_grad_enabled(),
                    block_mask=block_mask,
                )

            use_engine_external_cache = engine_self_kv is not None

            # Flex attention runs the block mask, which also hides its zero padding; without a mask or the
            # external cache, attention runs unpadded below.
            use_flex_attention = block_mask is not None and not use_engine_external_cache and FLEX_ATTENTION_AVAILABLE

            if block_mask is not None and not use_flex_attention and not is_cross_attn:
                logger.warning(
                    "Block mask provided for self-attention but flex_attention unavailable. "
                    f"use_external_cache={use_engine_external_cache}, "
                    f"FLEX_ATTENTION_AVAILABLE={FLEX_ATTENTION_AVAILABLE}"
                )

            if use_engine_external_cache:
                assert engine_self_kv is not None
                k_full, v_full = engine_self_kv

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
                # Unmasked self-attention (no block mask, no cache)
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

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CausalWan(CausalFastGenNetwork, Wan):
    """
    Wan with block-causal attention over frame chunks and an external KV cache for autoregressive generation.

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
        enable_logvar_linear: bool = False,
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
        **kwargs,
    ) -> None:
        """Causal Wan model constructor.

        Args:
            model_id_or_local_path (str, optional): The huggingface model ID or local path to load.
                Defaults to "Wan-AI/Wan2.1-T2V-1.3B-Diffusers".
            r_timestep (bool): Whether to support meanflow-like models with r timestep. Defaults to False.
            disable_efficient_attn (bool, optional): Whether to disable efficient attention. Defaults to False.
            disable_grad_ckpt (bool, optional): Whether to disable checkpoints during training. Defaults to False.
            enable_logvar_linear (bool, optional): Whether to enable logvar linear prediction. Defaults to False.
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
            **kwargs: Additional keyword arguments passed to ``Wan``.
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
            **kwargs,
        )

        # initialize the cache engine and register blocks
        self.transformer.kv_cache_engines = {"main": KVCacheEngine()}
        main_engine = self.transformer.kv_cache_engines["main"]
        for idx, block in enumerate(self.transformer.blocks):
            main_engine.register(block.attn1, block_idx=idx)
            main_engine.register(block.attn2, block_idx=idx)

    def override_transformer_forward(self) -> None:
        # Base Wan patches: non-causal classify_forward + sinusoidal override behavior.
        super().override_transformer_forward()

        # Rebind Wan transformer methods to causal implementations
        for idx, block in enumerate(self.transformer.blocks):
            block.forward = types.MethodType(block_forward, block)
        self.transformer.classify_forward_prepare = types.MethodType(classify_forward_prepare, self.transformer)
        self.transformer.classify_forward_block_forward = types.MethodType(
            classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Install causal attention processor (replaces diffusers' default WanAttnProcessor).
        self.transformer.set_attn_processor(CausalWanAttnProcessor())

    def clear_caches(self, delete_cache: bool = False) -> None:
        self.transformer.kv_cache_engines["main"].clear(delete=delete_cache)

        if delete_cache:
            torch.cuda.empty_cache()

    def _compute_timestep_inputs(self, timestep: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
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
            timestep (torch.Tensor): shape: (B, T), or (B, 1) for a (B, ) timestep without a mask
        """
        timestep = self.noise_scheduler.rescale_t(timestep)
        if timestep.ndim == 1:
            timestep = timestep.view(-1, 1)
        if mask is not None:
            p_t, _, _ = self.transformer.config.patch_size
            timestep = mask[:, ::p_t, 0, 0].to(timestep.dtype) * timestep
        return timestep

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        unpatchify_features: bool = True,
        fwd_pred_type: Optional[str] = None,
        skip_layers: Optional[List[int]] = None,
        causal_ctx: Optional[CausalContext] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the Wan diffusion score model.

        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (torch.Tensor): The text conditioning embeddings. Defaults to None.
            r (torch.Tensor): Another timestep mainly used by meanflow.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. Any provided set returns a feature list alongside the output,
                including an empty list for an empty set. None returns only the model output.
            return_logvar: If true, the forward pass also returns the logvar, shape [B, 1].
            unpatchify_features: If true, the features are unpatchified to [B, C', T, H, W] (C' = hidden size / patch
                volume); otherwise they stay packed as [B, N_tokens, D].
            fwd_pred_type: Update the network prediction type, must be in ['x0', 'eps', 'v', 'flow'].
                None means using the original net_pred_type.
            skip_layers: Apply skip-layer guidance by skipping layers of the unconditional network during forward pass.
            causal_ctx: Causal rollout state for the KV-cache path. None attends to the whole sequence without a
                mask; causal training and sampling always pass a context.
            apply_output_conditioning: Apply the output constraint after the conversion.
            output_dtype: ``"same"`` rounds the prediction to the state dtype; ``None`` keeps it in high precision.
        Returns:
            torch.Tensor: The score model output.
        """
        self._validate_forward_kwargs(
            fwd_kwargs,
            r=r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            skip_layers=skip_layers,
            supports_r_timestep=getattr(self.transformer, "r_embedder", None) is not None,
            supports_skip_layers=True,
            num_features=len(self.transformer.blocks),
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        assert isinstance(condition, torch.Tensor), "condition must be a tensor"
        x_t_model = self.cast_to_model_precision(x_t)
        assert_same(dtype=self.precision_model, device=x_t_model.device, condition=condition)
        assert_same(
            shape=True,
            dtype=self.noise_scheduler.t_precision,
            device=True,
            t=t,
            r=r,
        )
        timestep_mask = torch.ones_like(x_t[:, 0])  # shape: [batch_size, num_latent_frames, H, W]
        timestep = self._compute_timestep_inputs(t, timestep_mask)  # shape: [batch_size, num_latent_frames]
        r_timestep = None if r is None else self._compute_timestep_inputs(r, timestep_mask)

        # Pass the primary tensor inputs positionally so FSDP2 input handling is
        # preserved across activation-checkpoint recompute.
        model_outputs = self.transformer(
            x_t_model,
            timestep,
            condition,
            r_timestep=r_timestep,
            causal_ctx=causal_ctx,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            skip_layers=skip_layers,
        )

        if return_features_early:
            assert len(model_outputs) == len(feature_indices)
            return self._unpatchify_features(x_t_model, model_outputs) if unpatchify_features else model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        if feature_indices is None:
            assert isinstance(out, torch.Tensor)
            out = self.transform_output(out, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning)
        else:
            assert isinstance(out, list)
            out[0] = self.transform_output(
                out[0], x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
            )
            out[1] = self._unpatchify_features(x_t_model, out[1]) if unpatchify_features else out[1]

        if return_logvar:
            return out, logvar.to(dtype=x_t.dtype)
        return out

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        sample_steps: int = 50,
        sample_type: str = "ode",
        shift: float = 5.0,
        context_noise: Optional[float] = None,
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Autoregressive sampling for teacher network with CFG and UniPC.

        Args:
            noise (torch.Tensor): The pure noise to start from (usually a zero-mean, unit-variance Gaussian).
            condition (Optional[torch.Tensor]): The text conditioning embeddings. Defaults to None.
            neg_condition (Optional[torch.Tensor]): Optional negative text conditioning embeddings.
            sample_steps (int): Number of time steps to sample. Defaults to 50.
            sample_type: Integration policy; only ``"ode"`` is supported.
            shift (float): Shift value of timestep scheduler. Defaults to 5.0.
            context_noise (Optional[float]): Time in [0, 1] to which each finished chunk is re-noised before
                caching; None caches the clean chunk. Defaults to None.
            guidance_cfg (Optional[GuidanceConfig]): Composite guidance config; ``None`` →
                ``Wan.default_guidance_cfg()`` (CFG at 5.0).

        Returns:
            torch.Tensor: The sample output.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )

        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()
        # Ensure caches are clean at the start of sampling
        self.clear_caches()

        kv_cache_engine = self.transformer.kv_cache_engines["main"]

        # One ``(cache_key, ctx, cond)`` triple per active guidance forward —
        # drives both the per-key ``contexts`` dict for
        # ``predict_with_guidance`` and the cache-finalization loop.
        def new_ctx(cache_tag):
            return CausalContext(
                cache_tag=cache_tag,
                cache_engines={"main": kv_cache_engine},
                chunk_size=self.chunk_size,
                total_num_frames=self.total_num_frames,
                is_ar=True,
            )

        ctxs = [
            (key, new_ctx(key), guidance_cfg.condition_for_pred_key(key, condition, neg_condition))
            for key in guidance_cfg.active_pred_keys()
        ]

        scheduler, t_grid = self.sample_scheduler_and_grid(sample_steps, flow_shift=shift, device=noise.device)
        x = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])
        batch_size = x.shape[0]
        num_frames = x.shape[2]

        for start, end in causal_chunk_ranges(num_frames, self.chunk_size):
            x_next = x[:, :, start:end]
            # Reset the UniPC multistep state for this chunk; the grid (``timesteps`` / ``t_grid``) is unchanged.
            self.sample_scheduler_and_grid(sample_steps, flow_shift=shift, device=noise.device)

            # Configure all contexts for denoising (read-only cache access)
            for _, ctx, _ in ctxs:
                ctx.cur_start_frame = start
                ctx.store_kv = False

            for sched_step, t in tqdm(
                zip(scheduler.timesteps, t_grid, strict=True), total=len(t_grid), desc="Sampling"
            ):
                t = t.expand(batch_size)
                x_cur = x_next
                flow_pred = predict_with_guidance(
                    self,
                    x_cur,
                    t,
                    guidance_cfg=guidance_cfg,
                    condition=condition,
                    neg_condition=neg_condition,
                    fwd_pred_type="flow",
                    contexts={key: {"causal_ctx": ctx} for key, ctx, _ in ctxs},
                )
                x_next = scheduler.step(flow_pred, sched_step, x_next, return_dict=False)[0]

            x[:, :, start:end] = x_next

            # rerun with timestep zero or context_noise to finalize the cache slice
            x_cache = x_next
            t_cache = torch.full((batch_size,), 0, device=x.device, dtype=self.noise_scheduler.t_precision)
            if context_noise is not None:
                # Add context noise to denoised frames before caching
                t_cache = torch.full(
                    (batch_size,), context_noise, device=x.device, dtype=self.noise_scheduler.t_precision
                )
                x_cache = self.noise_scheduler.forward_process(x_next, torch.randn_like(x_next), t_cache)

            # Store KV cache for this chunk (one finalization pass per active ctx).
            for _, ctx, cond in ctxs:
                with ctx.set_store_kv(True):
                    _ = self(x_cache, t_cache, condition=cond, causal_ctx=ctx, apply_output_conditioning=False)

        # cleanup caches after full sampling
        self.clear_caches()
        return x
