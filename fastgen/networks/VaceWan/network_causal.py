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
from functools import partial
from typing import Any, Optional, List, Set, Dict, Union, Tuple

import torch
from diffusers.models import WanVACETransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, unscale_lora_layers
from tqdm.auto import tqdm

from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.networks.network import CausalFastGenNetwork
from fastgen.networks.Wan.network import flatten_timestep, unflatten_timestep_proj
from fastgen.networks.VaceWan.network import (
    VACEWan,
    vace_classify_forward,
)
from fastgen.networks.Wan.network_causal import (
    _rope_forward_with_time_offset,
    _prepare_blockwise_causal_attn_mask,
    _create_external_caches,
    _wan_set_attn_processor,
    _wan_block_forward_inline_cache,
    CausalWanAttnProcessor,
)
import fastgen.utils.logging_utils as logger


def _wan_vace_block_forward_inline_cache(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    control_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
    norm_temb: bool,
    block_mask=None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    cache_start_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inline-cache causal forward for WanVACETransformerBlock.

    This mirrors _wan_block_forward_inline_cache but operates on the VACE control branch and
    returns (conditioning_states, control_hidden_states).
    """
    # Optional input projection and residual from hidden_states
    if self.proj_in is not None:
        control_hidden_states = self.proj_in(control_hidden_states)
        control_hidden_states = control_hidden_states + hidden_states

    # temb is shape [B, N_frames, 6, D] in causal path
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        self.scale_shift_table.unsqueeze(0) + temb.float()
    ).chunk(6, dim=2)

    # Follow sCM to normalize the AdaLN condition
    if norm_temb:
        from fastgen.networks.Wan.network import normalize  # local import to avoid cycle at top-level

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
    seq_len = control_hidden_states.shape[1]
    num_frames = scale_msa.shape[1]
    frame_seqlen = seq_len // max(num_frames, 1)

    # 1. Self-attention on control branch
    norm_hidden_states = (
        (
            self.norm1(control_hidden_states.float()).unflatten(1, (max(num_frames, 1), frame_seqlen)) * (1 + scale_msa)
            + shift_msa
        )
        .flatten(1, 2)
        .type_as(control_hidden_states)
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
    control_hidden_states = (
        control_hidden_states.float()
        + (attn_output.unflatten(1, (max(num_frames, 1), frame_seqlen)) * gate_msa).flatten(1, 2)
    ).type_as(control_hidden_states)

    # 2. Cross-attention to text/image
    norm_hidden_states = self.norm2(control_hidden_states.float()).type_as(control_hidden_states)
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
    control_hidden_states = control_hidden_states + attn_output

    # 3. Feed-forward with per-frame gating
    norm_hidden_states = (
        (
            self.norm3(control_hidden_states.float()).unflatten(1, (max(num_frames, 1), frame_seqlen))
            * (1 + c_scale_msa)
            + c_shift_msa
        )
        .flatten(1, 2)
        .type_as(control_hidden_states)
    )
    ff_output = self.ffn(norm_hidden_states)
    control_hidden_states = (
        control_hidden_states.float()
        + (ff_output.float().unflatten(1, (max(num_frames, 1), frame_seqlen)) * c_gate_msa).flatten(1, 2)
    ).type_as(control_hidden_states)

    conditioning_states = None
    if self.proj_out is not None:
        conditioning_states = self.proj_out(control_hidden_states)

    return conditioning_states, control_hidden_states


def _create_external_caches_vace(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    num_frames: int,
    total_num_frames: int = 21,
    cache_tag: str = "pos",
    cur_start_frame: int = 0,
) -> None:
    """Allocate external KV caches for VACE control blocks.

    Creates per-VACE-block caches analogous to _create_external_caches used for main blocks,
    stored in `self._external_self_kv_list_vace` and `self._external_cross_kv_list_vace`.
    """
    bsz = hidden_states.shape[0]
    frame_seqlen = hidden_states.shape[1] // num_frames if num_frames > 0 else hidden_states.shape[1]
    capacity_tokens = frame_seqlen * total_num_frames

    if not hasattr(self, "_external_self_kv_list_vace"):
        self._external_self_kv_list_vace = []
    if not hasattr(self, "_external_cross_kv_list_vace"):
        self._external_cross_kv_list_vace = []
    self_list = self._external_self_kv_list_vace
    cross_list = self._external_cross_kv_list_vace

    # Check if batch size changed - need to reallocate caches
    need_reinit = len(self_list) != len(self.blocks) or len(cross_list) != len(self.blocks)
    if not need_reinit and len(self_list) > 0 and self_list[0] is not None:
        # Check if existing cache has different batch size
        existing_cache = self_list[0].get(cache_tag, {}).get("k", None)
        if existing_cache is not None and existing_cache.shape[0] != bsz:
            logger.info(f"Batch size changed from {existing_cache.shape[0]} to {bsz}, reallocating caches in vace")
            need_reinit = True

    if need_reinit:
        logger.info("creating external caches when self-forcing (or student-forcing) is enabled in vace")
        self_list.clear()
        cross_list.clear()
        for _ in self.blocks:
            self_list.append({})
            cross_list.append({})

    if any((cache_tag not in cache for cache in self_list)):
        for i, blk in enumerate(self.vace_blocks):
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
            want_shape_cross = (bsz, encoder_hidden_states.shape[1], heads, head_dim)
            device = hidden_states.device
            dtype = hidden_states.dtype

            self_cache = self_list[i].get(cache_tag)
            need_alloc_self = True
            if isinstance(self_cache, dict):
                k_buf = self_cache.get("k")
                v_buf = self_cache.get("v")
                if (
                    isinstance(k_buf, torch.Tensor)
                    and isinstance(v_buf, torch.Tensor)
                    and tuple(k_buf.shape) == want_shape
                    and tuple(v_buf.shape) == want_shape
                    and k_buf.dtype == dtype
                    and v_buf.dtype == dtype
                    and k_buf.device == device
                    and v_buf.device == device
                ):
                    need_alloc_self = False
                else:
                    logger.warning(
                        f"Self-attn KV cache shape/dtype/device mismatch for block {i}, reallocating. "
                        f"Expected shape={want_shape}, dtype={dtype}, device={device}"
                    )

            if need_alloc_self:
                self_list[i][cache_tag] = {
                    "k": torch.zeros(want_shape, device=device, dtype=dtype, requires_grad=False),
                    "v": torch.zeros(want_shape, device=device, dtype=dtype, requires_grad=False),
                    "len": cur_start_frame * frame_seqlen,
                }

            cross_cache = cross_list[i].get(cache_tag)
            need_alloc_cross = True
            if isinstance(cross_cache, dict):
                k_buf = cross_cache.get("k")
                v_buf = cross_cache.get("v")
                if (
                    isinstance(k_buf, torch.Tensor)
                    and isinstance(v_buf, torch.Tensor)
                    and tuple(k_buf.shape) == want_shape_cross
                    and tuple(v_buf.shape) == want_shape_cross
                    and k_buf.dtype == dtype
                    and v_buf.dtype == dtype
                    and k_buf.device == device
                    and v_buf.device == device
                ):
                    need_alloc_cross = False
                else:
                    logger.warning(
                        f"Cross-attn KV cache shape/dtype/device mismatch for block {i}, reallocating. "
                        f"Expected shape={want_shape_cross}, dtype={dtype}, device={device}"
                    )

            if need_alloc_cross:
                cross_list[i][cache_tag] = {
                    "k": torch.zeros(want_shape_cross, device=device, dtype=dtype, requires_grad=False),
                    "v": torch.zeros(want_shape_cross, device=device, dtype=dtype, requires_grad=False),
                    "is_init": False,
                }


def vace_causal_classify_forward_prepare(
    self: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    control_hidden_states: Optional[torch.Tensor] = None,
    control_hidden_states_scale: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[List[torch.Tensor]],
]:
    """Causal prepare step with VACE control handling and KV cache setup."""

    attention_kwargs = attention_kwargs or {}
    batch_size, _, num_frames, height, width = hidden_states.shape
    _, p_h, p_w = self.config.patch_size
    frame_seqlen = (height // p_h) * (width // p_w)

    chunk_size = attention_kwargs.get("chunk_size", 3)
    cache_tag = attention_kwargs.get("cache_tag", "pos")
    cur_start_frame = attention_kwargs.get("cur_start_frame", 0)
    total_num_frames = attention_kwargs.get("total_num_frames", num_frames)
    # frame_offset is the segment-level offset for extrapolation.
    # global_start_frame = frame_offset + cur_start_frame gives the global frame index
    # (e.g., for depth conditioning), which may differ from cur_start_frame during extrapolation
    frame_offset = attention_kwargs.get("frame_offset", 0)
    global_start_frame = frame_offset + cur_start_frame
    attention_kwargs["frame_seqlen"] = frame_seqlen

    if control_hidden_states is not None and control_hidden_states_scale is None:
        control_hidden_states_scale = control_hidden_states.new_ones(len(self.config.vace_layers))

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

    # Ensure RoPE buffers are on the same device as input, useful for cpu_offloading
    if isinstance(rotary_emb, (tuple, list)):
        rotary_emb = tuple(emb.to(device=hidden_states.device, dtype=hidden_states.dtype) for emb in rotary_emb)
    elif isinstance(rotary_emb, torch.Tensor):
        rotary_emb = rotary_emb.to(device=hidden_states.device, dtype=hidden_states.dtype)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    processed_control_states: Optional[torch.Tensor] = None
    processed_control_scales: Optional[List[torch.Tensor]] = None
    if control_hidden_states is not None:
        # Slice control frames to current chunk to ensure token counts align with hidden_states
        # Use global_start_frame (= frame_offset + cur_start_frame) instead of cur_start_frame
        # to correctly index into the depth conditioning during extrapolation
        ctrl_total_frames = control_hidden_states.shape[2]
        start_f = min(global_start_frame, ctrl_total_frames)
        end_f = min(start_f + num_frames, ctrl_total_frames)
        control_slice = control_hidden_states[:, :, start_f:end_f]

        processed_control_states = self.vace_patch_embedding(control_slice)
        processed_control_states = processed_control_states.flatten(2).transpose(1, 2)

        processed_control_tokens = processed_control_states.size(1)
        main_tokens = hidden_states.size(1)
        if processed_control_tokens > main_tokens:
            processed_control_states = processed_control_states[:, :main_tokens, :]
        elif processed_control_tokens < main_tokens:
            pad_len = main_tokens - processed_control_tokens
            padding = processed_control_states.new_zeros(batch_size, pad_len, processed_control_states.size(2))
            processed_control_states = torch.cat([processed_control_states, padding], dim=1)

        if control_hidden_states_scale is not None:
            processed_control_scales = list(torch.unbind(control_hidden_states_scale))

    timestep, ts_seq_len = flatten_timestep(timestep)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image,
        timestep_seq_len=ts_seq_len,
    )
    timestep_proj = unflatten_timestep_proj(timestep_proj, ts_seq_len)

    r_timestep_proj: Optional[torch.Tensor] = None
    if self.r_embedder is not None and r_timestep is not None:
        if self.time_cond_type == "abs":
            time_for_r = r_timestep
        elif self.time_cond_type == "diff":
            time_for_r = timestep - r_timestep
        else:
            raise ValueError(f"Invalid time condition: {self.time_cond_type}")

        time_for_r, rs_seq_len = flatten_timestep(time_for_r)
        r_timestep = self.r_embedder.timesteps_proj(time_for_r)
        time_embedder_dtype = next(iter(self.r_embedder.time_embedder.parameters())).dtype
        if r_timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            r_timestep = r_timestep.to(time_embedder_dtype)

        remb = self.r_embedder.time_embedder(r_timestep).type_as(encoder_hidden_states)
        r_timestep_proj = self.r_embedder.time_proj(self.r_embedder.act_fn(remb))
        r_timestep_proj = unflatten_timestep_proj(r_timestep_proj, rs_seq_len)

        if getattr(self, "encoder_depth", None) is None:
            timestep_proj = timestep_proj + r_timestep_proj
            temb = temb + remb
        else:
            temb = remb

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    if getattr(self, "block_mask", None) is None and num_frames == total_num_frames:
        self.block_mask = self._prepare_blockwise_causal_attn_mask(
            hidden_states.device,
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            chunk_size=chunk_size,
        )

    if num_frames < total_num_frames:
        # Allocate caches for main blocks
        self._create_external_caches(
            hidden_states,
            encoder_hidden_states,
            num_frames=num_frames,
            total_num_frames=total_num_frames,
            cache_tag=cache_tag,
            cur_start_frame=cur_start_frame,
        )
        # Allocate caches for VACE control blocks
        self._create_external_caches_vace(
            processed_control_states if processed_control_states is not None else hidden_states,
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
        temb,
        rotary_emb,
        processed_control_states,
        processed_control_scales,
    )


def vace_causal_classify_forward_block_forward(
    self: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep_proj: torch.Tensor,
    r_timestep_proj: Optional[torch.Tensor],
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    control_hidden_states: Optional[torch.Tensor] = None,
    control_hidden_states_scale: Optional[List[torch.Tensor]] = None,
    skip_layers: Optional[List[int]] = None,
    feature_indices: Optional[Set[int]] = None,
    return_features_early: bool = False,
    lora_scale: float = 1.0,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Causal block forward with optional feature collection and VACE control."""

    attention_kwargs = attention_kwargs or {}
    is_ar = attention_kwargs.get("is_ar", False)
    cache_tag = attention_kwargs.get("cache_tag", "pos")
    cur_start_frame = attention_kwargs.get("cur_start_frame", 0)
    frame_seqlen = attention_kwargs.get("frame_seqlen", 0)
    proper_cache_len = cur_start_frame * frame_seqlen
    current_block_mask = None if is_ar else getattr(self, "block_mask", None)
    norm_temb_flag = getattr(self, "norm_temb", False)

    features: List[torch.Tensor] = []
    feature_indices = feature_indices or set()
    idx = 0

    has_control = control_hidden_states is not None and hasattr(self, "vace_blocks")

    # Process VACE blocks
    control_hidden_states_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
    current_control_states = control_hidden_states
    if has_control:
        assert control_hidden_states_scale is not None
        for block_idx, block in enumerate(self.vace_blocks):
            if skip_layers is not None and idx in skip_layers:
                continue
            if (
                getattr(self, "encoder_depth", None) is not None
                and idx == getattr(self, "encoder_depth", None)
                and r_timestep_proj is not None
            ):
                timestep_proj = r_timestep_proj

            # Snapshot attention kwargs for VACE blocks with corrected cache length
            attn_snapshot_kwargs = dict(attention_kwargs)
            if hasattr(self, "_external_self_kv_list_vace") and block_idx < len(self._external_self_kv_list_vace):
                original_cache = self._external_self_kv_list_vace[block_idx]
                if isinstance(original_cache, dict) and cache_tag in original_cache:
                    original_cache[cache_tag]["len"] = proper_cache_len
                    stable_cache = {
                        cache_tag: {
                            "k": original_cache[cache_tag]["k"],
                            "v": original_cache[cache_tag]["v"],
                            "len": proper_cache_len,
                        }
                    }
                    attn_snapshot_kwargs["external_self_kv"] = stable_cache
                else:
                    attn_snapshot_kwargs["external_self_kv"] = original_cache

            if hasattr(self, "_external_cross_kv_list_vace") and block_idx < len(self._external_cross_kv_list_vace):
                attn_snapshot_kwargs["external_cross_kv"] = self._external_cross_kv_list_vace[block_idx]

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                block_with_cache = partial(block, cache_start_idx=proper_cache_len)
                conditioning_states, current_control_states = self._gradient_checkpointing_func(
                    block_with_cache,
                    hidden_states,
                    encoder_hidden_states,
                    current_control_states,
                    timestep_proj,
                    rotary_emb,
                    norm_temb_flag,
                    current_block_mask,
                    attn_snapshot_kwargs,
                )  # type: ignore
            else:
                # Always pass cache_start_idx explicitly to avoid reading from mutable cache dict
                conditioning_states, current_control_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    current_control_states,
                    timestep_proj,
                    rotary_emb,
                    norm_temb_flag,
                    current_block_mask,
                    attn_snapshot_kwargs,
                    cache_start_idx=proper_cache_len,
                )
            control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[block_idx]))
        control_hidden_states_list = control_hidden_states_list[::-1]

    # Process main blocks
    for block_idx, block in enumerate(self.blocks):
        if skip_layers is not None and idx in skip_layers:
            continue
        if (
            getattr(self, "encoder_depth", None) is not None
            and idx == getattr(self, "encoder_depth", None)
            and r_timestep_proj is not None
        ):
            timestep_proj = r_timestep_proj

        attn_snapshot_kwargs = dict(attention_kwargs)
        if hasattr(self, "_external_self_kv_list") and block_idx < len(self._external_self_kv_list):
            original_cache = self._external_self_kv_list[block_idx]
            if isinstance(original_cache, dict) and cache_tag in original_cache:
                original_cache[cache_tag]["len"] = proper_cache_len
                stable_cache = {
                    cache_tag: {
                        "k": original_cache[cache_tag]["k"],
                        "v": original_cache[cache_tag]["v"],
                        "len": proper_cache_len,
                    }
                }
                attn_snapshot_kwargs["external_self_kv"] = stable_cache
            else:
                attn_snapshot_kwargs["external_self_kv"] = original_cache

        if hasattr(self, "_external_cross_kv_list") and block_idx < len(self._external_cross_kv_list):
            attn_snapshot_kwargs["external_cross_kv"] = self._external_cross_kv_list[block_idx]

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            block_with_cache = partial(block, cache_start_idx=proper_cache_len)
            hidden_states = self._gradient_checkpointing_func(
                block_with_cache,
                hidden_states,
                timestep_proj,
                encoder_hidden_states,
                rotary_emb,
                norm_temb_flag,
                current_block_mask,
                attn_snapshot_kwargs,
            )  # type: ignore
        else:
            # Always pass cache_start_idx explicitly to avoid reading from mutable cache dict
            hidden_states = block(
                hidden_states,
                timestep_proj,
                encoder_hidden_states,
                rotary_emb,
                norm_temb_flag,
                current_block_mask,
                attention_kwargs=attn_snapshot_kwargs,
                cache_start_idx=proper_cache_len,
            )

        if has_control and block_idx in self.config.vace_layers:
            assert control_hidden_states_list, "Control states depleted before applying to configured layers"
            control_hint, scale = control_hidden_states_list.pop()
            hidden_states = hidden_states + control_hint * scale

        if idx in feature_indices:
            features.append(hidden_states)
        idx += 1

    if return_features_early:
        assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        return hidden_states, features

    return hidden_states, features


class CausalVACEWan(CausalFastGenNetwork, VACEWan):
    """VACE-WAN model with causal (autoregressive) extensions."""

    def __init__(
        self,
        depth_model_path: Optional[str] = None,
        context_scale: float = 1.0,
        model_id_or_local_path: str = VACEWan.MODEL_ID_1_3B,
        r_timestep: bool = False,
        disable_efficient_attn: bool = False,
        disable_grad_ckpt: bool = False,
        enable_logvar_linear: bool = True,
        r_embedder_init: str = "pretrained",
        time_cond_type: str = "abs",  # default to absolute for back compatibility
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
        """Causal VACE-WAN model constructor.

        Args:
            depth_model_path (str, optional): Path to depth model. Defaults to None.
            context_scale (float, optional): Scale factor for context influence. Defaults to 1.0.
            model_id_or_local_path (str, optional): The huggingface model ID or local path to load.
                Defaults to "Wan-AI/Wan2.1-VACE-1.3B-diffusers".
            r_timestep (bool): Whether to support meanflow-like models with r timestep. Defaults to False.
            disable_efficient_attn (bool, optional): Whether to disable efficient attention. Defaults to False.
            disable_grad_ckpt (bool, optional): Whether to disable checkpoints during training. Defaults to False.
            enable_logvar_linear (bool, optional): Whether to enable logvar linear prediction. Defaults to True.
            r_embedder_init (str, optional): Initialization method for the r embedder. Defaults to "pretrained".
            time_cond_type (str, optional): Time condition type. Defaults to "abs".
            norm_temb (bool, optional): Whether to normalize the time embeddings. Defaults to False.
            net_pred_type (str, optional): Prediction type. Defaults to "flow".
            schedule_type (str, optional): Schedule type. Defaults to "rf".
            encoder_depth (int, optional): The depth of the encoder (i.e. the number of blocks taking in t embeddings).
            load_pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
            use_fsdp_checkpoint (bool, optional): Whether to use FSDP gradient checkpointing. Defaults to True.
            chunk_size (int, optional): The chunk size for the transformer. Defaults to 3.
            total_num_frames (int, optional): The total number of frames. Defaults to 21.
            delete_cache_on_clear (bool, optional): Whether to delete the cache on clear. Defaults to False.
            **model_kwargs: Additional keyword arguments to pass to the FastGenNetwork constructor.
        """
        super().__init__(
            depth_model_path=depth_model_path,
            context_scale=context_scale,
            model_id_or_local_path=model_id_or_local_path,
            r_timestep=r_timestep,
            disable_efficient_attn=disable_efficient_attn,
            disable_grad_ckpt=disable_grad_ckpt,
            enable_logvar_linear=enable_logvar_linear,
            r_embedder_init=r_embedder_init,
            time_cond_type=time_cond_type,
            norm_temb=norm_temb,
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            encoder_depth=encoder_depth,
            load_pretrained=load_pretrained,
            use_fsdp_checkpoint=use_fsdp_checkpoint,
            chunk_size=chunk_size,
            total_num_frames=total_num_frames,
            **model_kwargs,
        )
        self._delete_cache_on_clear = delete_cache_on_clear

    def override_transformer_forward(self, inner_dim: int) -> None:
        if hasattr(self.transformer, "rope"):
            self.transformer.rope.forward = types.MethodType(_rope_forward_with_time_offset, self.transformer.rope)
        self.transformer._prepare_blockwise_causal_attn_mask = types.MethodType(
            _prepare_blockwise_causal_attn_mask, self.transformer
        )
        self.transformer._create_external_caches = types.MethodType(_create_external_caches, self.transformer)
        # Bind VACE-specific cache creator
        self.transformer._create_external_caches_vace = types.MethodType(_create_external_caches_vace, self.transformer)
        self.transformer.set_attn_processor = types.MethodType(_wan_set_attn_processor, self.transformer)
        self.transformer.classify_forward_prepare = types.MethodType(
            vace_causal_classify_forward_prepare, self.transformer
        )
        self.transformer.classify_forward_block_forward = types.MethodType(
            vace_causal_classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(vace_classify_forward, self.transformer)

        for block in self.transformer.blocks:
            block.forward = types.MethodType(_wan_block_forward_inline_cache, block)
        # Patch VACE control blocks to use inline-cached causal forward
        if hasattr(self.transformer, "vace_blocks"):
            for block in self.transformer.vace_blocks:
                block.forward = types.MethodType(_wan_vace_block_forward_inline_cache, block)

        self.transformer.set_attn_processor(CausalWanAttnProcessor())
        self.transformer.block_mask = None

    def clear_caches(self) -> None:
        if self._delete_cache_on_clear:
            if hasattr(self.transformer, "_external_self_kv_list"):
                del self.transformer._external_self_kv_list
            if hasattr(self.transformer, "_external_cross_kv_list"):
                del self.transformer._external_cross_kv_list
            if hasattr(self.transformer, "_external_self_kv_list_vace"):
                del self.transformer._external_self_kv_list_vace
            if hasattr(self.transformer, "_external_cross_kv_list_vace"):
                del self.transformer._external_cross_kv_list_vace
            torch.cuda.empty_cache()
            return
        if hasattr(self.transformer, "_external_self_kv_list"):
            for kvb in self.transformer._external_self_kv_list:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["len"] = 0
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])
        if hasattr(self.transformer, "_external_cross_kv_list"):
            for kvb in self.transformer._external_cross_kv_list:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["is_init"] = False
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])
        # Also clear VACE control caches
        if hasattr(self.transformer, "_external_self_kv_list_vace"):
            for kvb in self.transformer._external_self_kv_list_vace:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["len"] = 0
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])
        if hasattr(self.transformer, "_external_cross_kv_list_vace"):
            for kvb in self.transformer._external_cross_kv_list_vace:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["is_init"] = False
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])

    def _compute_timestep_inputs(self, timestep: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
        fwd_pred_type: Optional[str] = None,
        skip_layers: Optional[List[int]] = None,
        unpatchify_features: bool = True,
        cache_tag: str = "pos",
        cur_start_frame: int = 0,
        frame_offset: int = 0,
        store_kv: bool = False,
        is_ar: bool = False,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(condition, dict), "condition must be a dict"
        assert "text_embeds" in condition, "condition must contain 'text_embeds'"
        assert "vid_context" in condition, "condition must contain 'vid_context'"

        if feature_indices is None:
            feature_indices = set()
        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"

        timestep_mask = torch.ones_like(x_t[:, 0])
        timestep_inputs = self._compute_timestep_inputs(t, timestep_mask)
        r_timestep_inputs = None if r is None else self._compute_timestep_inputs(r, timestep_mask)

        text_embeds = condition["text_embeds"]
        control_hidden_states = condition["vid_context"]

        control_hidden_states_scale = self.context_scale
        if isinstance(control_hidden_states_scale, (int, float)):
            control_hidden_states_scale = [control_hidden_states_scale] * len(self.transformer.config.vace_layers)
        if isinstance(control_hidden_states_scale, list):
            if len(control_hidden_states_scale) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `control_hidden_states_scale` {len(control_hidden_states_scale)} does not match number of layers "
                    f"{len(self.transformer.config.vace_layers)}."
                )
            control_hidden_states_scale = torch.tensor(control_hidden_states_scale, device=x_t.device, dtype=x_t.dtype)
        if isinstance(control_hidden_states_scale, torch.Tensor):
            if control_hidden_states_scale.size(0) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `control_hidden_states_scale` {control_hidden_states_scale.size(0)} does not match number of layers "
                    f"{len(self.transformer.config.vace_layers)}."
                )
            control_hidden_states_scale = control_hidden_states_scale.to(device=x_t.device, dtype=x_t.dtype)

        encoder_hidden_states = torch.stack(text_embeds) if isinstance(text_embeds, list) else text_embeds

        attention_kwargs = {
            "cache_tag": cache_tag,
            "chunk_size": self.chunk_size,
            "store_kv": store_kv,
            "is_ar": is_ar,
            "cur_start_frame": cur_start_frame,
            # frame_offset is the segment-level offset (e.g., for depth conditioning).
            # Global frame index is computed as: frame_offset + cur_start_frame
            "frame_offset": frame_offset,
            "total_num_frames": self.total_num_frames,
        }

        model_outputs = self.transformer(
            hidden_states=x_t,
            timestep=timestep_inputs,
            encoder_hidden_states=encoder_hidden_states,
            r_timestep=r_timestep_inputs,
            encoder_hidden_states_image=None,
            control_hidden_states=control_hidden_states,
            control_hidden_states_scale=control_hidden_states_scale,
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

        t_for_scheduler = t[:, None, :, None, None] if t.ndim == 2 else t

        if len(feature_indices) == 0:
            assert isinstance(out, torch.Tensor)
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t_for_scheduler, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            assert isinstance(out, list)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t_for_scheduler, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
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
            condition (Optional[Dict[str, torch.Tensor]]): conditioning information.
            neg_condition (Optional[dict[str, torch.Tensor]]): Optional negative conditioning information.
                Defaults to None.
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

        x = noise
        batch_size = x.shape[0]
        num_frames = x.shape[2]

        num_chunks = num_frames // self.chunk_size
        remaining_size = num_frames % self.chunk_size
        time_rescale_factor = self.unipc_scheduler.config.num_train_timesteps
        for i in range(max(1, num_chunks)):
            if num_chunks == 0:
                start, end = 0, remaining_size
            else:
                start = 0 if i == 0 else self.chunk_size * i + remaining_size
                end = self.chunk_size * (i + 1) + remaining_size

            x_next = x[:, :, start:end]
            self.unipc_scheduler.config.flow_shift = shift
            self.unipc_scheduler.set_timesteps(num_inference_steps=sample_steps, device=noise.device)
            timesteps = self.unipc_scheduler.timesteps
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
