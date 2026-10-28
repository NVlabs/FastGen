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

import types
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from einops import rearrange
from tqdm.auto import tqdm

from diffusers.models import WanVACETransformer3DModel

from fastgen.networks.kv_cache import KVCacheEngine
from fastgen.networks.lora_utils import scaled_lora_layers
from fastgen.networks.network_causal import CausalContext, causal_chunk_ranges
from fastgen.networks.VaceWan.modules.vace_depth_annotator import DEPTH_ANYTHING_V2_VITL_URL
from fastgen.networks.VaceWan.network import VACEWan
from fastgen.networks.Wan.network import (
    apply_r_embedding,
    flatten_timestep,
    normalize,
    project_logvar,
    project_output,
    unflatten_timestep_proj,
)
from fastgen.networks.Wan.network_causal import (
    CausalWan,
    _get_kv_cache_spec_for_blocks,
    compute_rotary_emb_with_time_offset,
)
from fastgen.utils.tree_utils import OutputDtype, assert_same
from fastgen.utils.guidance import (
    GuidanceConfig,
    predict_with_guidance,
)


def _maybe_anchor_last_self_kv(
    causal_ctx: Optional[CausalContext],
    engine_key: str,
    block: torch.nn.Module,
    output: torch.Tensor,
) -> None:
    """Anchor cached self-KV to `output` when gradient-through-cache is enabled."""
    if causal_ctx is None or not causal_ctx.store_kv or not causal_ctx.grad_through_kv_cache:
        return

    engine = causal_ctx.get_engine(engine_key)
    attn1 = getattr(block, "attn1", None)
    if engine is None or attn1 is None or not engine.is_registered(attn1):
        return

    engine.anchor_last_self_kv_to_output(
        block_idx=engine.resolve_block_idx(attn1),
        cache_tag=causal_ctx.cache_tag,
        output=output,
    )


def _wan_vace_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    control_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
    norm_temb: bool,
    causal_ctx: Optional[CausalContext] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal forward for VACE control blocks using the shared Wan processor."""
    core_dtype = encoder_hidden_states.dtype
    if self.proj_in is not None:
        control_hidden_states = self.proj_in(control_hidden_states.to(core_dtype))
        control_hidden_states = control_hidden_states + hidden_states.to(core_dtype)

    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        self.scale_shift_table.unsqueeze(0) + temb.float()
    ).chunk(6, dim=2)

    if norm_temb:
        shift_msa = normalize(shift_msa)
        scale_msa = normalize(scale_msa)
        c_shift_msa = normalize(c_shift_msa)
        c_scale_msa = normalize(c_scale_msa)

    seq_len = control_hidden_states.shape[1]
    num_frames = scale_msa.shape[1]
    frame_seqlen = seq_len // max(num_frames, 1)

    norm_hidden_states = (
        (
            self.norm1(control_hidden_states).float().unflatten(1, (max(num_frames, 1), frame_seqlen)) * (1 + scale_msa)
            + shift_msa
        )
        .flatten(1, 2)
        .to(core_dtype)
    )
    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=rotary_emb,
        causal_ctx=causal_ctx,
    )
    control_hidden_states = control_hidden_states.float() + (
        attn_output.unflatten(1, (max(num_frames, 1), frame_seqlen)) * gate_msa
    ).flatten(1, 2)

    norm_hidden_states = self.norm2(control_hidden_states.float()).to(core_dtype)
    attn_output = self.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states.to(core_dtype),
        attention_mask=None,
        rotary_emb=None,
        causal_ctx=causal_ctx,
    )
    control_hidden_states = control_hidden_states + attn_output

    norm_hidden_states = (
        (
            self.norm3(control_hidden_states.float()).unflatten(1, (max(num_frames, 1), frame_seqlen))
            * (1 + c_scale_msa)
            + c_shift_msa
        )
        .flatten(1, 2)
        .to(core_dtype)
    )
    ff_output = self.ffn(norm_hidden_states)
    control_hidden_states = control_hidden_states.float() + (
        ff_output.float().unflatten(1, (max(num_frames, 1), frame_seqlen)) * c_gate_msa
    ).flatten(1, 2)
    if not getattr(self, "fp32_residual", True):
        control_hidden_states = control_hidden_states.to(core_dtype)

    conditioning_states = None
    if self.proj_out is not None:
        conditioning_states = self.proj_out(control_hidden_states.to(core_dtype))

    return conditioning_states, control_hidden_states


def vace_causal_classify_forward_prepare(
    self: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    control_hidden_states: Optional[torch.Tensor] = None,
    control_hidden_states_scale: Optional[torch.Tensor] = None,
    causal_ctx: Optional[CausalContext] = None,
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
    """Prepare causal VACE inputs, including per-branch cache setup."""
    _, _, num_frames, height, width = hidden_states.shape
    _, p_h, p_w = self.config.patch_size

    if control_hidden_states is not None and control_hidden_states_scale is None:
        control_hidden_states_scale = control_hidden_states.new_ones(len(self.config.vace_layers))

    if causal_ctx is not None:
        if causal_ctx.get_engine("main") is None:
            causal_ctx.set_engine("main", self.kv_cache_engines["main"])
        if control_hidden_states is not None and causal_ctx.get_engine("vace") is None:
            causal_ctx.set_engine("vace", self.kv_cache_engines["vace"])
        _, p_h, p_w = self.rope.patch_size
        frame_seqlen = (height // p_h) * (width // p_w)
        cur_start_frame = causal_ctx.resolve_start_frame(frame_seqlen, engine_key="main")
        rotary_emb = compute_rotary_emb_with_time_offset(self.rope, hidden_states, cur_start_frame)
    else:
        rotary_emb = self.rope(hidden_states)
        cur_start_frame = 0

    assert_same(
        device=True,
        hidden_states=hidden_states,
        freqs_cos=rotary_emb[0],
        freqs_sin=rotary_emb[1],
    )

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    processed_control_states: Optional[torch.Tensor] = None
    processed_control_scales: Optional[List[torch.Tensor]] = None
    if control_hidden_states is not None:
        start_f = (causal_ctx.frame_offset if causal_ctx is not None else 0) + cur_start_frame
        end_f = start_f + num_frames
        if end_f > control_hidden_states.shape[2]:
            raise ValueError(
                f"VACE control has {control_hidden_states.shape[2]} latent frames, but this pass reads frames "
                f"[{start_f}, {end_f})"
            )
        processed_control_states = self.vace_patch_embedding(control_hidden_states[:, :, start_f:end_f])
        processed_control_states = processed_control_states.flatten(2).transpose(1, 2)
        assert_same(
            shape=True, device=True, hidden_states=hidden_states, processed_control_states=processed_control_states
        )

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

        if processed_control_states is not None:
            block_count, self_shape, cross_shape = _get_kv_cache_spec_for_blocks(
                blocks=self.vace_blocks,
                hidden_states=processed_control_states,
                encoder_hidden_states=encoder_hidden_states,
                num_frames=num_frames,
                total_num_frames=total_num_frames,
            )
            causal_ctx.ensure_kv_cache_capacity(
                engine_key="vace",
                block_count=block_count,
                self_shape=self_shape,
                cross_shape=cross_shape,
                device=processed_control_states.device,
                dtype=processed_control_states.dtype,
            )
    elif causal_ctx is not None:
        causal_ctx.prepare_block_mask(device=hidden_states.device, num_frames=num_frames, engine_key="main")
        if processed_control_states is not None:
            causal_ctx.prepare_block_mask(device=hidden_states.device, num_frames=num_frames, engine_key="vace")

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
    causal_ctx: Optional[CausalContext] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run causal VACE control and main blocks with shared CausalContext state."""

    features: List[torch.Tensor] = []
    has_control = control_hidden_states is not None and hasattr(self, "vace_blocks")

    control_hidden_states_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
    current_control_states = control_hidden_states
    if has_control:
        assert control_hidden_states_scale is not None
        for block_idx, block in enumerate(self.vace_blocks):
            vace_ctx_snapshot = causal_ctx.snapshot() if causal_ctx is not None else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def _vace_block(
                    hidden_states_in: torch.Tensor,
                    encoder_hidden_states_in: torch.Tensor,
                    control_hidden_states_in: torch.Tensor,
                    timestep_proj_in: torch.Tensor,
                    rotary_emb_in: torch.Tensor,
                    norm_temb_in: bool,
                    _block: torch.nn.Module = block,
                    _causal_ctx: Optional[CausalContext] = vace_ctx_snapshot,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    return _block(
                        hidden_states_in,
                        encoder_hidden_states_in,
                        control_hidden_states_in,
                        timestep_proj_in,
                        rotary_emb_in,
                        norm_temb_in,
                        causal_ctx=_causal_ctx,
                    )

                conditioning_states, current_control_states = self._gradient_checkpointing_func(
                    _vace_block,
                    hidden_states,
                    encoder_hidden_states,
                    current_control_states,
                    timestep_proj,
                    rotary_emb,
                    self.norm_temb,
                )
            else:
                conditioning_states, current_control_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    current_control_states,
                    timestep_proj,
                    rotary_emb,
                    self.norm_temb,
                    causal_ctx=vace_ctx_snapshot,
                )

            _maybe_anchor_last_self_kv(
                causal_ctx=vace_ctx_snapshot,
                engine_key="vace",
                block=block,
                output=current_control_states,
            )
            control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[block_idx]))

        control_hidden_states_list = control_hidden_states_list[::-1]

    for block_idx, block in enumerate(self.blocks):
        if self.encoder_depth is not None and block_idx == self.encoder_depth and r_timestep_proj is not None:
            timestep_proj = r_timestep_proj

        # Skip only the backbone block; still apply its VACE hint and record its feature so both stay aligned.
        if skip_layers is None or block_idx not in skip_layers:
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

            _maybe_anchor_last_self_kv(
                causal_ctx=causal_ctx_snapshot,
                engine_key="main",
                block=block,
                output=hidden_states,
            )

        if has_control and block_idx in self.config.vace_layers:
            assert control_hidden_states_list, "Control states depleted before applying to configured layers"
            control_hint, scale = control_hidden_states_list.pop()
            hidden_states = hidden_states + control_hint * scale

        if feature_indices is not None and block_idx in feature_indices:
            features.append(hidden_states)
            if return_features_early and len(features) == len(feature_indices):
                break

    return hidden_states, features


def vace_causal_classify_forward(
    self: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    control_hidden_states: Optional[torch.Tensor] = None,
    control_hidden_states_scale: Optional[torch.Tensor] = None,
    causal_ctx: Optional[CausalContext] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
    skip_layers: Optional[List[int]] = None,
    **kwargs,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Causal VACE transformer forward aligned with the shared Wan causal path."""
    lora_scale = float(kwargs.get("scale", 1.0))

    batch_size, _, num_frames, height, width = hidden_states.shape
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
            processed_control_states,
            processed_control_scales,
        ) = self.classify_forward_prepare(
            hidden_states,
            timestep,
            encoder_hidden_states,
            r_timestep,
            encoder_hidden_states_image,
            control_hidden_states,
            control_hidden_states_scale,
            causal_ctx=causal_ctx,
        )

        hidden_states, features = self.classify_forward_block_forward(
            hidden_states,
            timestep_proj,
            r_timestep_proj,
            encoder_hidden_states,
            rotary_emb,
            processed_control_states,
            processed_control_scales,
            skip_layers,
            feature_indices,
            bool(return_features_early),
            causal_ctx=causal_ctx,
        )

        if return_features_early:
            assert feature_indices is not None
            assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
            return features

        if temb.dim() == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)

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
        output = rearrange(hidden_states, "b t h w pt ph pw c -> b c (t pt) (h ph) (w pw)")

        out: Union[torch.Tensor, List[torch.Tensor]] = [output, features] if feature_indices is not None else output

        if return_logvar:
            assert hasattr(
                self, "logvar_linear"
            ), "logvar_linear layer is required when return_logvar=True. Set enable_logvar_linear=True in model config."
            # One value per sample: average the time embeddings of the noised frames.
            logvar = project_logvar(self, temb, timestep, encoder_hidden_states.dtype)
            return out, logvar
        return out


class CausalVACEWan(CausalWan, VACEWan):
    """VACE-WAN model with causal (autoregressive) extensions."""

    def __init__(
        self,
        depth_model_path: str = DEPTH_ANYTHING_V2_VITL_URL,
        context_scale: float = 1.0,
        model_id_or_local_path: str = VACEWan.MODEL_ID_1_3B,
        r_timestep: bool = False,
        disable_efficient_attn: bool = False,
        disable_grad_ckpt: bool = False,
        enable_logvar_linear: bool = False,
        r_embedder_init: str = "pretrained",
        time_cond_type: str = "abs",
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
        """Initialize the causal VACE-WAN model.

        Args:
            depth_model_path: Local path, S3 URI, or HTTP(S) URL for the depth model.
            context_scale: Scale factor for context influence.
            model_id_or_local_path: HuggingFace model ID or local path for the base Wan model.
            r_timestep: Whether to support meanflow-like models with r timestep. Defaults to False.
            disable_efficient_attn: Whether to disable efficient attention. Defaults to False.
            disable_grad_ckpt: Whether to disable checkpoints during training. Defaults to False.
            enable_logvar_linear: Whether to enable logvar linear prediction. Defaults to False.
            r_embedder_init: Initialization method for the r embedder. Defaults to "pretrained".
            time_cond_type: Time condition type. Defaults to "abs".
            norm_temb: Whether to normalize the time embeddings of the main and control blocks. Defaults to False.
            net_pred_type: Prediction type. Defaults to "flow".
            schedule_type: Schedule type. Defaults to "rf".
            encoder_depth: Number of blocks taking in t embeddings. Defaults to None, meaning all blocks
                take in t + r embeddings.
            load_pretrained: Whether to load the model from pretrained. Defaults to True.
            use_fsdp_checkpoint: Whether to use FSDP gradient checkpointing. Defaults to True.
            chunk_size: Number of frames per causal chunk. Defaults to 3.
            total_num_frames: Total frames in a full video rollout. Defaults to 21.
            **kwargs: Additional keyword arguments passed to ``VACEWan``.
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
            **kwargs,
        )

        # CausalWan registers the main blocks in the "main" engine.
        vace_engine = KVCacheEngine()
        self.transformer.kv_cache_engines["vace"] = vace_engine
        for idx, block in enumerate(getattr(self.transformer, "vace_blocks", [])):
            vace_engine.register(block.attn1, block_idx=idx)
            vace_engine.register(block.attn2, block_idx=idx)

    def override_transformer_forward(self) -> None:
        # VACE and base Wan patches first, then CausalWan's causal main blocks and attention processor; the VACE
        # causal classify forwards and VACE blocks are rebound below. ``WanVACETransformer3DModel`` derives from
        # diffusers' ``AttentionMixin``, whose recursive ``set_attn_processor`` reaches the main and VACE blocks
        # alike (pinned by the VACE processor tests).
        super().override_transformer_forward()
        self.transformer.classify_forward_prepare = types.MethodType(
            vace_causal_classify_forward_prepare, self.transformer
        )
        self.transformer.classify_forward_block_forward = types.MethodType(
            vace_causal_classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(vace_causal_classify_forward, self.transformer)

        for block in getattr(self.transformer, "vace_blocks", []):
            block.forward = types.MethodType(_wan_vace_block_forward, block)

    def clear_caches(self, delete_cache: bool = False) -> None:
        for engine in self.transformer.kv_cache_engines.values():
            engine.clear(delete=delete_cache)
        if delete_cache:
            torch.cuda.empty_cache()

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
        causal_ctx: Optional[CausalContext] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
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

        assert isinstance(condition, dict), "condition must be a dict"
        assert "text_embeds" in condition, "condition must contain 'text_embeds'"
        assert "vid_context" in condition, "condition must contain 'vid_context'"

        assert_same(
            shape=True,
            dtype=self.noise_scheduler.t_precision,
            device=True,
            t=t,
            r=r,
        )
        timestep_mask = torch.ones_like(x_t[:, 0])
        timestep_inputs = self._compute_timestep_inputs(t, timestep_mask)
        r_timestep_inputs = None if r is None else self._compute_timestep_inputs(r, timestep_mask)
        x_t_model = self.cast_to_model_precision(x_t)

        encoder_hidden_states = condition["text_embeds"]
        control_hidden_states = condition["vid_context"]
        assert_same(
            dtype=self.precision_model,
            device=x_t_model.device,
            text_embeds=encoder_hidden_states,
            vid_context=control_hidden_states,
        )

        control_hidden_states_scale = self.context_scale
        if isinstance(control_hidden_states_scale, (int, float)):
            control_hidden_states_scale = [control_hidden_states_scale] * len(self.transformer.config.vace_layers)
        if isinstance(control_hidden_states_scale, list):
            if len(control_hidden_states_scale) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `control_hidden_states_scale` {len(control_hidden_states_scale)} does not match number of layers "
                    f"{len(self.transformer.config.vace_layers)}."
                )
            control_hidden_states_scale = torch.tensor(
                control_hidden_states_scale, device=x_t_model.device, dtype=x_t_model.dtype
            )
        if isinstance(control_hidden_states_scale, torch.Tensor):
            if control_hidden_states_scale.size(0) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `control_hidden_states_scale` {control_hidden_states_scale.size(0)} does not match number of layers "
                    f"{len(self.transformer.config.vace_layers)}."
                )
            control_hidden_states_scale = control_hidden_states_scale.to(device=x_t_model.device, dtype=x_t_model.dtype)

        model_outputs = self.transformer(
            x_t_model,
            timestep_inputs,
            encoder_hidden_states,
            r_timestep=r_timestep_inputs,
            encoder_hidden_states_image=None,
            control_hidden_states=control_hidden_states,
            control_hidden_states_scale=control_hidden_states_scale,
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
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        sample_steps: int = 50,
        sample_type: str = "ode",
        shift: float = 3.0,
        context_noise: Optional[float] = None,
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Autoregressive sampling for causal VACE-WAN with CFG and UniPC.

        ``shift`` defaults to 3.0, the ``flow_shift`` in the Diffusers VACE checkpoints' scheduler config. The
        official Wan2.1 ``generate.py`` and VACE inference script sample VACE at shift 16.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )
        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()
        self.clear_caches()

        scheduler, t_grid = self.sample_scheduler_and_grid(sample_steps, flow_shift=shift, device=noise.device)

        # Scaled working copy: generated chunks are written into ``x``, never into the caller's ``noise``.
        x = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])
        batch_size = x.shape[0]
        num_frames = x.shape[2]

        cache_engines = {
            "main": self.transformer.kv_cache_engines["main"],
            "vace": self.transformer.kv_cache_engines["vace"],
        }

        # One ``(cache_key, ctx, cond)`` triple per active guidance forward.
        def new_ctx(cache_tag):
            return CausalContext(
                cache_tag=cache_tag,
                cache_engines=cache_engines,
                chunk_size=self.chunk_size,
                total_num_frames=self.total_num_frames,
                is_ar=True,
            )

        ctxs = [
            (key, new_ctx(key), guidance_cfg.condition_for_pred_key(key, condition, neg_condition))
            for key in guidance_cfg.active_pred_keys()
        ]

        for start, end in causal_chunk_ranges(num_frames, self.chunk_size):
            x_next = x[:, :, start:end]
            # Reset the UniPC multistep state for this chunk; the grid (``timesteps`` / ``t_grid``) is unchanged.
            self.sample_scheduler_and_grid(sample_steps, flow_shift=shift, device=noise.device)

            # Configure all contexts for denoising (read-only cache access).
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

            x_cache = x_next
            t_cache = torch.full((batch_size,), 0, device=x.device, dtype=self.noise_scheduler.t_precision)
            if context_noise is not None:
                t_cache = torch.full(
                    (batch_size,), context_noise, device=x.device, dtype=self.noise_scheduler.t_precision
                )
                x_cache = self.noise_scheduler.forward_process(x_next, torch.randn_like(x_next), t_cache)

            # Store KV cache for this chunk (one finalization pass per active ctx).
            for _, ctx, cond in ctxs:
                with ctx.set_store_kv(True):
                    _ = self(x_cache, t_cache, condition=cond, causal_ctx=ctx, apply_output_conditioning=False)

        self.clear_caches()
        return x
