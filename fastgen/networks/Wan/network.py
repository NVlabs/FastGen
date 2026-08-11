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

import copy
from typing import Any, Optional, List, Set, Dict, Union, Tuple, Mapping
import os
import types
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch import dtype
from torch.distributed.fsdp import fully_shard
from diffusers import UniPCMultistepScheduler

from diffusers.models import WanTransformer3DModel, AutoencoderKLWan
from diffusers.models.transformers.transformer_wan import WanTransformerBlock, WanRotaryPosEmbed
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from transformers import AutoTokenizer, UMT5EncoderModel

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.Wan.utils import convert_wan_official_state_dict, remap_anyflow_keys
from fastgen.networks.noise_schedule import NET_PRED_TYPES

from fastgen.utils.basic_utils import prompt_clean, str2bool
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger


def flatten_timestep(timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Flatten timestep and return seq_len if timestep has a shape of [batch_size, seq_len]
    """
    ts_seq_len = None
    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()  # batch_size * seq_len

    return timestep, ts_seq_len


def unflatten_timestep_proj(timestep_proj: torch.Tensor, ts_seq_len: Optional[int] = None) -> torch.Tensor:
    """
    Unflatten timestep, taking expand_timesteps (seq_len) into account
    """
    if ts_seq_len is not None:
        # batch_size, seq_len, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        # batch_size, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    return timestep_proj


def normalize(x, dim=None, eps=1e-4):
    dtype = torch.float64 if x.dtype is torch.float64 else torch.float32
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=dtype)
    norm = torch.add(eps, norm, alpha=1.0)
    return x / norm.to(x.dtype)


def sinusoidal_embedding_1d_wan(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Sinusoidal embedding matching original WAN implementation. In diffusers, the cos and sin are swapped.

    Key differences from diffusers Timesteps:
    - Uses 10000^(-i/half) instead of 10000^(-i/(half-1))
    - Order is [cos, sin] instead of [sin, cos]

    Args:
        dim: Total embedding dimension (will be split into half for cos and sin)
        position: Input positions/timesteps tensor

    Returns:
        Sinusoidal embedding of shape [len(position), dim]
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half, device=position.device).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.float()


def block_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    norm_temb: bool,
):
    if temb.ndim == 4:
        # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        # batch_size, seq_len, 1, inner_dim
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)
    else:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

    # Follow sCM to normalize the AdaLN condition
    if norm_temb:
        shift_msa = normalize(shift_msa)
        scale_msa = normalize(scale_msa)
        c_shift_msa = normalize(c_shift_msa)
        c_scale_msa = normalize(c_scale_msa)

    # 1. Self-attention
    norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
    attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
    hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

    # 2. Cross-attention
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
    hidden_states = hidden_states + attn_output

    # 3. Feed-forward
    norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

    return hidden_states


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
    skip_layers: Optional[List[int]] = None,
    **kwargs,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    # get the sizes
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    # parepare the input for the block forward passes
    (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
        temb,
        rotary_emb,
    ) = self.classify_forward_prepare(
        hidden_states,
        timestep,
        encoder_hidden_states,
        r_timestep,
        encoder_hidden_states_image,
        attention_kwargs,
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
        lora_scale,
        attention_kwargs,
    )

    # If we have all the features, we can exit early
    if return_features_early:
        assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
        return features

    # 5. Output norm, projection & unpatchify
    if temb.dim() == 3:
        # Per-frame (or per-token) modulation
        # Compute per-frame (or per-token) shift/scale from model-level table and per-frame (or per-token) temb
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)

        # Two possible cases:
        #  - Wan 2.1: Per-frame modulation (causal architecture)
        #  - Wan 2.2 IT2V 5B: Per-token modulation (default)
        if shift.shape[1] == hidden_states.shape[1]:
            # Wan 2.2 IT2V 5B model
            # Apply shift/scale per token on normalized tokens
            hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        else:
            # Apply shift/scale per frame on normalized tokens
            frame_seqlen = post_patch_height * post_patch_width
            hs_norm_out = self.norm_out(hidden_states.float()).unflatten(1, (post_patch_num_frames, frame_seqlen))
            hidden_states = (
                (hs_norm_out * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)).flatten(1, 2).type_as(hidden_states)
            )
    else:
        # Global modulation
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if len(feature_indices) == 0:
        # no features requested, return only the model output
        out = output
    else:
        # score and features； score, features
        out = [output, features]

    if return_logvar:
        assert hasattr(
            self, "logvar_linear"
        ), "logvar_linear layer is required when return_logvar=True. Set enable_logvar_linear=True in model config."
        logvar = self.logvar_linear(temb)
        return out, logvar
    return out


def _fuse_r_embedding(
    self,
    temb: torch.Tensor,
    timestep_proj: torch.Tensor,
    remb: torch.Tensor,
    rs_seq_len: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Combine the t- and r-time embeddings into the final timestep projection.

    * ``additive``: project each embedding separately and sum — ``temb + remb``,
      ``timestep_proj + r_embedder.time_proj(...)``.
    * ``gated``: interpolate first, ``(1-g)·temb + g·remb``, then project the
      blend through the condition_embedder's own ``time_proj`` — shared with the
      t-only path, so ``Wan.__init__`` drops the unused ``r_embedder.time_proj``
      rather than let it drift during training.

    Returns ``(temb, timestep_proj, r_timestep_proj)``: ``temb`` [B, D] modulates
    the output head (``scale_shift_table + temb``), ``timestep_proj`` [B, 6, D] is
    the AdaLN modulation every block consumes, and ``r_timestep_proj`` is what the
    blocks switch to at ``encoder_depth`` — ``None`` when there is no switch.
    """
    fusion = getattr(self.r_embedder, "fusion_mode", "additive")

    if fusion == "gated":
        gate = self.r_embedder.gate_value
        rt_emb = (1 - gate) * temb + gate * remb
        rt_ts_proj = self.condition_embedder.time_proj(self.condition_embedder.act_fn(rt_emb))
        rt_ts_proj = unflatten_timestep_proj(rt_ts_proj, rs_seq_len)
        if self.encoder_depth is None:
            return rt_emb, rt_ts_proj, None
        return rt_emb, timestep_proj, rt_ts_proj

    # additive — MeanFlow original path, bit-identical
    r_ts_proj = self.r_embedder.time_proj(self.r_embedder.act_fn(remb))
    r_ts_proj = unflatten_timestep_proj(r_ts_proj, rs_seq_len)
    if self.encoder_depth is None:
        return temb + remb, timestep_proj + r_ts_proj, None
    return remb, timestep_proj, r_ts_proj


def classify_forward_prepare(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = {},
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

    Returns:
        hidden_states: [B, N_tokens, D_model], where N_tokens=(F/p_t)*(H/p_h)*(W/p_w)
        timestep_proj: [B, 6, N_frames * H * W, D_model] if Wan2.2 TI2V 5B else [B, 6, N_frames, D_model], where N_frames=F/p_t
        encoder_hidden_states: [B, K, N_frames, D_model], K in {6, 12}
        encoder_hidden_states_image: [B, K_img, N_frames, D_model] if provided else None
        temb: [B, N_frames * H * W, D_model] if Wan2.2 TI2V 5B else [B, N_frames, D_model]
        rotary_emb: [B, N_tokens, D_model]
    """
    # Ensure RoPE buffers are on the same device as input
    if self.rope.freqs_cos.device != hidden_states.device:
        self.rope.freqs_cos = self.rope.freqs_cos.to(hidden_states.device)
        self.rope.freqs_sin = self.rope.freqs_sin.to(hidden_states.device)

    rotary_emb = self.rope(hidden_states)

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
            # Keep the r_timestep as is
            pass
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

        temb, timestep_proj, r_timestep_proj = self._fuse_r_embedding(temb, timestep_proj, remb, rs_seq_len)
    elif r_timestep is not None:
        # Raise an error here, otherwise we silently ignore the r_timestep
        raise ValueError("r_timestep provided but no r_embedder is present")
    else:
        r_timestep_proj = None

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    return (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
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
    lora_scale: Optional[float] = 1.0,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    block forward pass inside the classify_forward function.
    will be used as the block_forward_op function in the classify_forward function.

    args:
        self: WanTransformer3DModel
        hidden_states: [B, N_tokens, D_model]
        timestep_proj: [B, 6, N_frames, D_model]
        encoder_hidden_states: [B, K, N_frames, D_model]
        rotary_emb: [B, N_tokens, D_model]
        r_timestep_proj: [B, 6, N_frames, D_model]
        skip_layers: Optional[List[int]]
        feature_indices: Optional[Set[int]]
        attention_kwargs: Optional[Dict[str, Any]]

    Returns:
        hidden_states: [B, N_tokens, D_model]
        features: List[[B, N_tokens, D_model]]
    """
    features = []
    for idx, block in enumerate(self.blocks):
        if skip_layers is not None and idx in skip_layers:
            continue
        if self.encoder_depth is not None and idx == self.encoder_depth and r_timestep_proj is not None:
            timestep_proj = r_timestep_proj
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, timestep_proj, encoder_hidden_states, rotary_emb, self.norm_temb
            )
        else:
            hidden_states = block(hidden_states, timestep_proj, encoder_hidden_states, rotary_emb, self.norm_temb)
        if feature_indices is not None and idx in feature_indices:
            features.append(hidden_states)

        # If we have all the features, we can exit early
        if return_features_early and len(features) == len(feature_indices):
            if USE_PEFT_BACKEND:
                # Clean up LoRA scaling before early exit
                unscale_lora_layers(self, lora_scale)
            return hidden_states, features

    return hidden_states, features


class WanTextEncoder:
    def __init__(self, model_id_or_local_path):
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id_or_local_path,
            cache_dir=os.environ["HF_HOME"],
            subfolder="text_encoder",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.text_encoder.eval().requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_local_path,
            cache_dir=os.environ["HF_HOME"],
            subfolder="tokenizer",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )

    def encode(self, conditioning: Optional[Any] = None, precision: dtype = torch.float32) -> torch.Tensor:
        conditioning = [conditioning] if isinstance(conditioning, str) else conditioning
        conditioning = [prompt_clean(u) for u in conditioning]

        text_inputs = self.tokenizer(
            conditioning,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device), mask.to(self.text_encoder.device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=precision, device=self.text_encoder.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        return prompt_embeds

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        self.text_encoder.to(*args, **kwargs)
        return self


class WanVideoEncoder:
    def __init__(
        self,
        model_id_or_local_path: str,
    ):
        self.vae: AutoencoderKLWan = AutoencoderKLWan.from_pretrained(
            model_id_or_local_path,
            cache_dir=os.environ["HF_HOME"],
            subfolder="vae",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            torch_dtype=torch.float32,
        )
        # We never update the encoder, so freeze it
        self.vae.eval().requires_grad_(False)

    def encode(self, real_images: torch.Tensor, mode="sample") -> torch.Tensor:
        # Ensure real_images is on the same device as VAE to avoid device mismatch
        real_images = real_images.to(device=self.vae.device, dtype=self.vae.dtype)
        if mode == "sample":
            latent_images = self.vae.encode(real_images, return_dict=False)[0].sample()
        elif mode == "argmax":
            latent_images = self.vae.encode(real_images, return_dict=False)[0].mode()
        else:
            raise ValueError(f"Invalid mode: {mode}. Supported modes: ['sample', 'argmax']")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(real_images.device, real_images.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            real_images.device, real_images.dtype
        )

        return (latent_images - latents_mean) * latents_std

    def decode(self, latent_images: torch.Tensor) -> torch.Tensor:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latent_images.device, latent_images.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latent_images.device, latent_images.dtype
        )
        latents = latent_images / latents_std + latents_mean
        # Ensure latents is on the same device and dtype as VAE to avoid device mismatch
        latents = latents.to(device=self.vae.device, dtype=self.vae.dtype)
        videos = self.vae.decode(latents, return_dict=False)[0].clip_(-1.0, 1.0)
        return videos

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        self.vae.to(*args, **kwargs)
        return self


class Wan(FastGenNetwork):
    """A Wan teacher model for text-to-video diffusion distillation."""

    MODEL_ID_14B = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    MODEL_ID_1_3B = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    MODEL_ID_VER_2_2_TI2V_5B_720P = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID_1_3B,
        r_timestep: bool = False,
        disable_efficient_attn: bool = False,
        disable_grad_ckpt: bool = False,
        enable_logvar_linear: bool = True,  # Enable logvar_linear if necessary
        r_embedder_init: str = "pretrained",
        time_cond_type: str = "diff",
        norm_temb: bool = False,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        encoder_depth: int | None = None,
        load_pretrained: bool = True,
        use_fsdp_checkpoint: bool = True,
        use_wan_official_sinusoidal: bool = False,
        r_embedder_fusion: str = "additive",
        r_embedder_gate_value: float = 0.25,
        **model_kwargs,
    ):
        """Wan2.1/2.2 model constructor.

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
            load_pretrained (bool, optional): Whether to load the model from pretrained.
                If False, the model will be initialized from the config.
                Defaults to True.
            use_fsdp_checkpoint (bool, optional): Whether to use FSDP gradient checkpointing. Defaults to True.
            use_wan_official_sinusoidal (bool, optional): If True, use official WAN sinusoidal embedding.
                If False (default), use diffusers default sinusoidal embedding.
            **model_kwargs: Additional keyword arguments to pass to the FastGenNetwork constructor.
        """
        # Initialize FastGenNetwork with Wan-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        if disable_efficient_attn:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

        self._use_fsdp_checkpoint = use_fsdp_checkpoint
        self._use_wan_official_sinusoidal = use_wan_official_sinusoidal

        model_id, inner_dim = self._initialize_network(model_id_or_local_path, load_pretrained)
        self.model_id = model_id

        # Add logvar_linear layer (conditional based on enable_logvar_linear flag)
        if enable_logvar_linear:
            self.transformer.logvar_linear = torch.nn.Linear(inner_dim, 1)
            logger.info("Added logvar_linear layer")
        else:
            logger.info("Skipped logvar_linear layer (disabled by enable_logvar_linear=False)")
        self.transformer.norm_temb = norm_temb
        self.transformer.encoder_depth = encoder_depth
        self.transformer.time_cond_type = time_cond_type
        if r_timestep:
            logger.info(f"Initializing r embedder with {r_embedder_init}")
            self.transformer.r_embedder = self.init_embedder(
                r_embedder_init, r_embedder_fusion, r_embedder_gate_value
            )
        else:
            self.transformer.r_embedder = None

        # core functionality to override forward function and other methods in the transformer
        self.override_transformer_forward(inner_dim=inner_dim)

        # Only the base Wan forward routes the r-embedding through
        # _fuse_r_embedding (installed just above); the causal / VACE variants
        # install their own (additive-only) classify_forward_prepare, where
        # "gated" would be silently ignored.
        if r_timestep and r_embedder_fusion == "gated" and not hasattr(self.transformer, "_fuse_r_embedding"):
            raise ValueError(
                f"r_embedder_fusion='gated' is not supported by {type(self).__name__} "
                "(its transformer forward does not implement gated t/r fusion)."
            )

        # Use lazy initialization as this wont work in a meta context when doing FSDP2.
        self._unipc_scheduler = None

        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    @property
    def unipc_scheduler(self) -> UniPCMultistepScheduler:
        """Lazily initialize the scheduler."""
        if self._unipc_scheduler is None:
            self._unipc_scheduler = UniPCMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        return self._unipc_scheduler

    def _initialize_network(self, model_id_or_local_path: str, load_pretrained: bool) -> Tuple[str, int]:
        """Initialize the network.
        Args:
            model_id_or_local_path (str): The model ID or local path to load. This can be either a HuggingFace
                model ID or a local directory holding the model weights.
            load_pretrained (bool): Whether to load the model from pretrained. If False, the model will be initialized from the config.

        Returns:
            Tuple[str, int]: The model ID and the inner dimension.
                model_id (str): The model ID (extracted from local path if needed).
                inner_dim (int): The inner dimension.
        """
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading Wan transformer")
            self.transformer: WanTransformer3DModel = WanTransformer3DModel.from_pretrained(
                model_id_or_local_path,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
        else:
            # Load config and create model structure
            # If we're in a meta context, tensors will automatically be on meta device
            config = WanTransformer3DModel.load_config(
                model_id_or_local_path,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
            if in_meta_context:
                logger.info(
                    "Initializing Wan transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing Wan transformer from config (no pretrained weights)")
                logger.warning("Wan Transformer being initializated from config. No weights are loaded!")
            self.transformer: WanTransformer3DModel = WanTransformer3DModel.from_config(config)
        model_id = Wan.get_model_id(model_id_or_local_path)

        if model_id == self.MODEL_ID_14B:
            inner_dim = 5120
            self.expand_timesteps = False
        elif model_id == self.MODEL_ID_VER_2_2_TI2V_5B_720P:
            inner_dim = 3072
            self.expand_timesteps = True
        else:
            inner_dim = 1536
            self.expand_timesteps = False
        return model_id, inner_dim

    def reset_parameters(self):
        """Reinitialize non-persistent buffers that are computed from model config.

        This is required when using meta device initialization for FSDP2. Non-persistent
        buffers (like RoPE freqs_cos/freqs_sin) are not included in state_dict, so they
        must be recomputed on all ranks after materializing from meta device.

        Call this method AFTER to_empty() and BEFORE or AFTER set_model_state_dict(),
        since these buffers are independent of the trainable parameters.

        Example usage in FSDP initialization:
            model.to_empty(device=torch.cuda.current_device())
            model.reset_parameters()  # Reinitialize RoPE buffers
            set_model_state_dict(model, state_dict, options=options)
        """
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.transformer.patch_embedding.weight.flatten(1))
        for m in self.transformer.condition_embedder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        if self.transformer.r_embedder is not None:
            for m in self.transformer.r_embedder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.transformer.proj_out.weight)

        # Reinitialize RoPE buffers by re-instantiating the rope module
        # We need this as they are no persistent, and wont be synced by set_model_state_dict
        # when using FSDP2
        # The rope module computes freqs_cos/freqs_sin from the transformer config
        if hasattr(self.transformer, "rope") and self.transformer.rope is not None:
            config = self.transformer.config
            rope_config = {
                "attention_head_dim": config.attention_head_dim,
                "patch_size": config.patch_size,
                "max_seq_len": config.rope_max_seq_len,
                "theta": getattr(config, "rope_theta", 10000.0),
            }

            # Get the current device (after to_empty, buffers are on the target device)
            device = next(self.transformer.rope.buffers()).device

            # Re-instantiate rope on the correct device to compute buffers
            new_rope = WanRotaryPosEmbed(**rope_config).to(device)

            # Copy the computed buffers to the existing rope module
            # This preserves the module identity (important for FSDP)
            with torch.no_grad():
                self.transformer.rope.freqs_cos.copy_(new_rope.freqs_cos)
                self.transformer.rope.freqs_sin.copy_(new_rope.freqs_sin)

            del new_rope
            logger.debug("Reinitialized RoPE buffers (freqs_cos, freqs_sin)")

        # Reinitialize noise scheduler (its _sigmas tensor is not a registered buffer)
        super().reset_parameters()

    def fully_shard(self, **kwargs):
        """Fully shard the network.

        Note: We shard `self.transformer` instead of `self` because the network wrapper
        class (e.g., CausalWanI2V) may have complex multiple inheritance with ABC,
        which causes Python's __class__ assignment to fail due to incompatible memory layouts.
        FSDP2's fully_shard works by dynamically modifying __class__, so we apply it only
        to the transformer submodule which is a standard torch.nn.Module.
        """
        # Note: Checkpointing has to happen first, for proper casting during backward pass recomputation.
        if self.transformer.gradient_checkpointing and self._use_fsdp_checkpoint:
            # Disable the built-in gradient checkpointing (which uses torch.utils.checkpoint)
            self.transformer.disable_gradient_checkpointing()
            # Apply FSDP-compatible activation checkpointing to the transformer
            # This will wrap each block (child of transformer.blocks) with checkpoint_wrapper
            apply_fsdp_checkpointing(self.transformer, check_fn=lambda block: isinstance(block, WanTransformerBlock))
            logger.info("Applied FSDP activation checkpointing to transformer blocks")

        # Apply FSDP sharding
        for block in self.transformer.blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

    @classmethod
    def get_model_id(cls, model_path_or_id: str | os.PathLike) -> str:
        """Extract the model ID from a string that represents the ID or a local directory."""
        is_local = os.path.isdir(model_path_or_id)

        if not is_local:
            return model_path_or_id

        # Match and extract
        idx_start = model_path_or_id.rfind("Wan2.")
        if idx_start == -1:
            raise ValueError("unable to extract model id from path. Expects substring `Wan2.X-`")
        name = model_path_or_id[idx_start:].split("/")[0]
        return f"Wan-AI/{name}"

    def init_embedder(
        self,
        embedder_init: str,
        r_embedder_fusion: str = "additive",
        r_embedder_gate_value: float = 0.25,
    ) -> torch.nn.Module:
        """Build the r-embedder and stash its fusion config on the module.

        The fusion settings live on the module (not on `self`) so the
        method-bound forward override can branch on them without changing its
        signature. They are set before the meta-device early return so both
        paths carry them.

        For gated fusion the r_embedder's own `time_proj` / `act_fn` are dropped
        here (the fused embedding goes through the condition_embedder's shared
        projection instead). Whether the transformer's forward actually supports
        gated fusion can only be checked after `override_transformer_forward`
        installs `_fuse_r_embedding`, so that check lives in `__init__`.
        """
        embedder = copy.deepcopy(self.transformer.condition_embedder)
        del embedder.text_embedder

        if r_embedder_fusion not in ("additive", "gated"):
            raise ValueError(f"r_embedder_fusion must be 'additive' or 'gated', got {r_embedder_fusion!r}")
        embedder.fusion_mode = r_embedder_fusion
        # Plain float (not a buffer) so FSDP's reset_parameters does not need to
        # re-materialize it after meta-device init.
        embedder.gate_value = float(r_embedder_gate_value)
        logger.info(
            f"r_embedder fusion={r_embedder_fusion}"
            + (f" gate_value={r_embedder_gate_value}" if r_embedder_fusion == "gated" else "")
        )

        # Skip initialization if using meta device (weights will be broadcast via FSDP)
        if self._is_in_meta_context():
            logger.info("Skipping r_embedder initialization on meta device (will receive weights via FSDP sync)")
        # zero init the r_embedder
        elif embedder_init == "zero":
            for param in embedder.parameters():
                param.data.zero_()
        elif embedder_init == "random":
            # following https://github.com/Wan-Video/Wan2.1/blob/7c81b2f27defa56c7e627a4b6717c8f2292eee58/wan/modules/model.py#L609
            for param in embedder.modules():
                if isinstance(param, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(param.weight)
                    if param.bias is not None:
                        torch.nn.init.zeros_(param.bias)
            for param in embedder.time_embedder.modules():
                if isinstance(param, torch.nn.Linear):
                    torch.nn.init.normal_(param.weight, std=0.02)

            # to have zero remb initially
            torch.nn.init.zeros_(embedder.time_embedder.linear_2.weight)
            torch.nn.init.zeros_(embedder.time_embedder.linear_2.bias)
            # to have zero r_timestep_proj initially
            torch.nn.init.zeros_(embedder.time_proj.weight)
            torch.nn.init.zeros_(embedder.time_proj.bias)
        elif embedder_init == "pretrained":
            # Since the r_embedder is already copied from the condition_embedder, we don't need to do anything
            pass
        else:
            raise ValueError(f"Invalid embedder_init: {embedder_init}")

        if r_embedder_fusion == "gated":
            # The gated path projects the fused (t, r) embedding through the
            # condition_embedder's own time_proj, shared with the t-only path.
            # Drop the r_embedder's copy so it cannot linger as an untrained
            # dead parameter. Done after the init branches above because
            # "random" initializes time_proj.
            del embedder.time_proj
            del embedder.act_fn
        return embedder

    def override_transformer_forward(self, inner_dim: int) -> None:
        # Override transformer forward methods with custom implementations
        for block in self.transformer.blocks:
            block.forward = types.MethodType(block_forward, block)
        self.transformer._fuse_r_embedding = types.MethodType(_fuse_r_embedding, self.transformer)
        self.transformer.classify_forward_prepare = types.MethodType(classify_forward_prepare, self.transformer)
        self.transformer.classify_forward_block_forward = types.MethodType(
            classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Override timesteps_proj to use official WAN sinusoidal embedding (if configured)
        if self._use_wan_official_sinusoidal:
            logger.info("Using official WAN sinusoidal embedding")
            self._override_timesteps_proj(self.transformer.condition_embedder.timesteps_proj)
            if self.transformer.r_embedder is not None:
                self._override_timesteps_proj(self.transformer.r_embedder.timesteps_proj)

    def _override_timesteps_proj(self, timesteps_proj_module) -> None:
        """Override the timesteps_proj forward to use original WAN sinusoidal embedding."""
        num_channels = timesteps_proj_module.num_channels

        def new_forward(self, timesteps):
            # Use original WAN sinusoidal embedding instead of diffusers default sinusoidal embedding
            return sinusoidal_embedding_1d_wan(num_channels, timesteps)

        timesteps_proj_module.forward = types.MethodType(new_forward, timesteps_proj_module)

    def init_preprocessors(self):
        """Initialize the text and video encoders for the Wan model."""
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder for Wan model."""
        self.text_encoder = WanTextEncoder(model_id_or_local_path=self.model_id)

    def init_vae(self):
        """Initialize the video encoder for Wan model."""
        self.vae = WanVideoEncoder(model_id_or_local_path=self.model_id)

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)

        return self

    def _unpatchify_features(self, x_t: torch.Tensor, features: List[torch.Tensor]) -> List[torch.Tensor]:
        B, C, T, H, W = x_t.shape
        p_t, p_h, p_w = self.transformer.config.patch_size
        post_patch_num_frames = T // p_t
        post_patch_height = H // p_h
        post_patch_width = W // p_w
        feats = []
        for feat in features:
            feat = feat.reshape(B, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1)
            feat = feat.permute(0, 7, 1, 4, 2, 5, 3, 6)
            feat = feat.flatten(6, 7).flatten(4, 5).flatten(2, 3)
            feats.append(feat)
        return feats

    def _compute_timestep_inputs(self, timestep: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        Compute timestep input used for Wan models.
        Optionally Expand or mask the timestep input for Wan 2.2 TI2V models.
            - I2V: Apply a mask that zeroes out the timestep for the first latent frame.
            - T2V: Use a mask tensor filled with ones.
        """
        timestep = self.noise_scheduler.rescale_t(timestep)

        if self.expand_timesteps and mask is not None:
            p_t, p_h, p_w = self.transformer.config.patch_size
            # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
            timestep = (mask[:, ::p_t, ::p_h, ::p_w].to(timestep.dtype) * timestep.view(-1, 1, 1, 1)).flatten(1)
        return timestep

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 50,
        shift: float = 5.0,
        skip_layers: Optional[List[int]] = None,
        skip_layers_start_fraction: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """Multistep sample using the UniPC method

        Args:
            noise (torch.Tensor): The noisy latents to start from.
            condition (Dict[str, torch.Tensor], optional): The condition
            neg_condition (Dict[str, torch.Tensor], optional): The negative condition
            guidance_scale (Optional[float]): the guidance scale. None means no guidance.
            num_steps (int): The number of sampling steps.
            shift (float): Noise schedule shift parameter. Affects temporal dynamics.
            skip_layers (Optional[List[int]]): List of transformer layers to skip (used by SLG) during sampling.
            skip_layers_start_fraction (float): Fraction in [0, 1] of the sampling steps to complete
                before skip-layer guidance becomes active.

        Returns:
            torch.Tensor: The sample output.
        """
        assert self.schedule_type == "rf", f"{self.schedule_type} is not supported"

        self.unipc_scheduler.config.flow_shift = shift
        self.unipc_scheduler.set_timesteps(num_inference_steps=num_steps, device=noise.device)
        timesteps = self.unipc_scheduler.timesteps

        # Initialize latents with proper scaling based on the initial timestep
        t_init = timesteps[0] / self.unipc_scheduler.config.num_train_timesteps
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        # main sampling loop
        for idx, timestep in enumerate(tqdm(timesteps)):
            t = (timestep / self.unipc_scheduler.config.num_train_timesteps).expand(latents.shape[0])
            t = self.noise_scheduler.safe_clamp(t, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t).to(
                latents.dtype
            )

            flow_pred = self(
                latents,
                t,
                condition=condition,
                r=None,
                return_features_early=False,
                feature_indices={},
                return_logvar=False,
            )

            if guidance_scale is not None:
                flow_uncond = self(
                    latents,
                    t,
                    condition=neg_condition,
                    r=None,
                    return_features_early=False,
                    feature_indices={},
                    return_logvar=False,
                    skip_layers=skip_layers if idx >= skip_layers_start_fraction * num_steps else None,
                )
                flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)

            latents = self.unipc_scheduler.step(flow_pred, timestep, latents, return_dict=False)[0]

        return latents

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        """Load state dict with automatic extraction for original Wan checkpoint formats.

        Also handles checkpoints where the model state is nested under a common key,
        i.e., {'generator': state_dict}.
        """
        if self._use_wan_official_sinusoidal and not any(k.startswith("transformer.") for k in state_dict.keys()):
            state_dict = convert_wan_official_state_dict(state_dict)

        # Some third-party checkpoints store the r-pathway as
        # condition_embedder.delta_embedder; remap to FastGen's r_embedder
        # layout (no-op for all other checkpoints).
        state_dict = remap_anyflow_keys(state_dict)

        return super().load_state_dict(state_dict, **kwargs)

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
            fwd_pred_type: Update the network prediction type, must be in ['x0', 'eps', 'v', 'flow'].
                None means using the original net_pred_type.
            skip_layers: Apply skip-layer guidance by skipping layers of the unconditional network during forward pass.
            unpatchify_features: If true, the features will be unpatchified and returned in shape of [B, T, H, W, C].

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
        timestep = self._compute_timestep_inputs(t, timestep_mask)
        r_timestep = None if r is None else self._compute_timestep_inputs(r, timestep_mask)

        model_outputs = self.transformer(
            hidden_states=x_t,
            timestep=timestep,
            encoder_hidden_states=condition,
            r_timestep=r_timestep,
            attention_kwargs=None,
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

        if len(feature_indices) == 0:
            assert isinstance(out, torch.Tensor)
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            assert isinstance(out, list)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            out[1] = self._unpatchify_features(x_t, out[1]) if unpatchify_features else out[1]

        if return_logvar:
            return out, logvar
        return out
