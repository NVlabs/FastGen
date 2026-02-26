# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 FastGen network implementation.

Architecture verified against:
  - diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2.py
  - diffusers/src/diffusers/pipelines/ltx2/connectors.py
  - diffusers/src/diffusers/models/transformers/transformer_ltx2.py

Follows the FastGen network pattern established by Flux and Wan:
  - Inherits from FastGenNetwork
  - Monkey-patches classify_forward onto self.transformer
  - forward() handles video-only latent for distillation (audio flows through but is ignored for loss)
  - feature_indices extracts video hidden_states only
"""

import copy
import types
from typing import Any, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from diffusers.models.transformers import LTX2VideoTransformer3DModel
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformerBlock
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from torch.distributed.fsdp import fully_shard
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger


# ---------------------------------------------------------------------------
# Helpers (mirrors of diffusers pipeline static methods)
# ---------------------------------------------------------------------------

def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    """Pack video latents [B, C, F, H, W] → [B, F//pt * H//p * W//p, C*pt*p*p]."""
    B, C, F, H, W = latents.shape
    pF = F // patch_size_t
    pH = H // patch_size
    pW = W // patch_size
    latents = latents.reshape(B, C, pF, patch_size_t, pH, patch_size, pW, patch_size)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int,
    patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    """Unpack video latents [B, T, D] → [B, C, F, H, W]."""
    B = latents.size(0)
    latents = latents.reshape(B, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack audio latents [B, C, L, M] → [B, L, C*M]."""
    return latents.transpose(1, 2).flatten(2, 3)


def _unpack_audio_latents(latents: torch.Tensor, latent_length: int, num_mel_bins: int) -> torch.Tensor:
    """Unpack audio latents [B, L, C*M] -> [B, C, L, M]."""
    return latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)


def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std  = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - mean) * scaling_factor / std


def _denormalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std  = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * std / scaling_factor + mean


def _normalize_audio_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    mean = latents_mean.to(latents.device, latents.dtype)
    std  = latents_std.to(latents.device, latents.dtype)
    return (latents - mean) / std


def _denormalize_audio_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    mean = latents_mean.to(latents.device, latents.dtype)
    std  = latents_std.to(latents.device, latents.dtype)
    return latents * std + mean


def _pack_text_embeds(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    device: torch.device,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Stack all Gemma hidden-state layers, normalize per-batch/per-layer over
    non-padded positions, and pack into [B, T, H * num_layers].
    """
    B, T, H, L = text_hidden_states.shape
    original_dtype = text_hidden_states.dtype

    token_indices = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    else:  # left
        start = T - sequence_lengths[:, None]
        mask = token_indices >= start
    mask = mask[:, :, None, None]  # [B, T, 1, 1]

    masked = text_hidden_states.masked_fill(~mask, 0.0)
    num_valid = (sequence_lengths * H).view(B, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (num_valid + eps)

    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

    normed = (text_hidden_states - mean) / (x_max - x_min + eps) * scale_factor
    normed = normed.flatten(2)                          # [B, T, H*L]
    mask_flat = mask.squeeze(-1).expand(-1, -1, H * L)
    normed = normed.masked_fill(~mask_flat, 0.0)
    return normed.to(original_dtype)


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 1024,
    max_seq_len: int = 4096,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    """Mirrors the pipeline's calculate_shift — defaults match LTX-2 scheduler config."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, mu=None):
    """Call scheduler.set_timesteps, forwarding mu when dynamic shifting is enabled."""
    kwargs = {}
    if mu is not None:
        kwargs["mu"] = mu
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


# ---------------------------------------------------------------------------
# classify_forward — monkey-patched onto self.transformer
# ---------------------------------------------------------------------------

def classify_forward(
    self,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    audio_encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    audio_timestep: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    audio_encoder_attention_mask: Optional[torch.Tensor] = None,
    num_frames: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    fps: float = 24.0,
    audio_num_frames: Optional[int] = None,
    video_coords: Optional[torch.Tensor] = None,
    audio_coords: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[dict] = None,
    return_dict: bool = True,           # accepted for API compatibility; always ignored
    # FastGen distillation kwargs
    return_features_early: bool = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],                              # (video_out, audio_out)
    Tuple[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]],  # ((video_out, audio_out), features)
    List[torch.Tensor],                                             # features only (early exit)
]:
    """
    Drop-in replacement for LTX2VideoTransformer3DModel.forward that adds FastGen
    distillation support (feature extraction, early exit, logvar).

    Audio always flows through every block unchanged — we never short-circuit it —
    but only video hidden_states are stored as features for the discriminator.

    Returns
    -------
    Normal mode  (feature_indices empty, return_features_early False):
        (video_output, audio_output)  — identical to the original forward

    Feature mode (feature_indices non-empty, return_features_early False):
        ((video_output, audio_output), List[video_feature_tensors])

    Early-exit mode (return_features_early True):
        List[video_feature_tensors]   — forward stops as soon as all features collected
    """
    # LoRA scale handling — mirrors the @apply_lora_scale decorator in upstream
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    if feature_indices is None:
        feature_indices = set()

    if return_features_early and len(feature_indices) == 0:
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        return []

    # ------------------------------------------------------------------ #
    # Steps 1-4: identical to the original forward (no changes)
    # ------------------------------------------------------------------ #
    audio_timestep = audio_timestep if audio_timestep is not None else timestep

    # Convert attention masks to additive bias form
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
        audio_encoder_attention_mask = (
            1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)
        ) * -10000.0
        audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

    batch_size = hidden_states.size(0)

    # 1. RoPE positional embeddings
    if video_coords is None:
        video_coords = self.rope.prepare_video_coords(
            batch_size, num_frames, height, width, hidden_states.device, fps=fps
        )
    if audio_coords is None:
        audio_coords = self.audio_rope.prepare_audio_coords(
            batch_size, audio_num_frames, audio_hidden_states.device
        )

    video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
    audio_rotary_emb = self.audio_rope(audio_coords, device=audio_hidden_states.device)

    video_cross_attn_rotary_emb = self.cross_attn_rope(video_coords[:, 0:1, :], device=hidden_states.device)
    audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
        audio_coords[:, 0:1, :], device=audio_hidden_states.device
    )

    # 2. Patchify input projections
    hidden_states = self.proj_in(hidden_states)
    audio_hidden_states = self.audio_proj_in(audio_hidden_states)

    # 3. Timestep embeddings and modulation parameters
    timestep_cross_attn_gate_scale_factor = (
        self.config.cross_attn_timestep_scale_multiplier / self.config.timestep_scale_multiplier
    )

    temb, embedded_timestep = self.time_embed(
        timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype,
    )
    temb = temb.view(batch_size, -1, temb.size(-1))
    embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

    temb_audio, audio_embedded_timestep = self.audio_time_embed(
        audio_timestep.flatten(), batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype,
    )
    temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
    audio_embedded_timestep = audio_embedded_timestep.view(batch_size, -1, audio_embedded_timestep.size(-1))

    video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
        timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype,
    )
    video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
        timestep.flatten() * timestep_cross_attn_gate_scale_factor,
        batch_size=batch_size, hidden_dtype=hidden_states.dtype,
    )
    video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
        batch_size, -1, video_cross_attn_scale_shift.shape[-1]
    )
    video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(
        batch_size, -1, video_cross_attn_a2v_gate.shape[-1]
    )

    audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
        audio_timestep.flatten(), batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype,
    )
    audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
        audio_timestep.flatten() * timestep_cross_attn_gate_scale_factor,
        batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype,
    )
    audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
        batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
    )
    audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(
        batch_size, -1, audio_cross_attn_v2a_gate.shape[-1]
    )

    # 4. Prompt embeddings
    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

    audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
    audio_encoder_hidden_states = audio_encoder_hidden_states.view(
        batch_size, -1, audio_hidden_states.size(-1)
    )

    # ------------------------------------------------------------------ #
    # Step 5: Block loop with video-only feature extraction
    # Audio always flows through every block — we never skip it.
    # ------------------------------------------------------------------ #
    features: List[torch.Tensor] = []

    for idx, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states, audio_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                audio_encoder_hidden_states,
                temb,
                temb_audio,
                video_cross_attn_scale_shift,
                audio_cross_attn_scale_shift,
                video_cross_attn_a2v_gate,
                audio_cross_attn_v2a_gate,
                video_rotary_emb,
                audio_rotary_emb,
                video_cross_attn_rotary_emb,
                audio_cross_attn_rotary_emb,
                encoder_attention_mask,
                audio_encoder_attention_mask,
            )
        else:
            hidden_states, audio_hidden_states = block(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
                temb_ca_scale_shift=video_cross_attn_scale_shift,
                temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                temb_ca_gate=video_cross_attn_a2v_gate,
                temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=audio_rotary_emb,
                ca_video_rotary_emb=video_cross_attn_rotary_emb,
                ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
            )

        # Video-only feature extraction at requested block indices
        if idx in feature_indices:
            features.append(hidden_states.clone())  # [B, T_v, D_v] — packed video tokens

        # Early exit once all requested features are collected
        if return_features_early and len(features) == len(feature_indices):
            return features

    # ------------------------------------------------------------------ #
    # Step 6: Output layers (video + audio) — unchanged from original
    # ------------------------------------------------------------------ #
    scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
    shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
    hidden_states = self.norm_out(hidden_states)
    hidden_states = hidden_states * (1 + scale) + shift
    video_output = self.proj_out(hidden_states)

    audio_scale_shift_values = self.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
    audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]
    audio_hidden_states = self.audio_norm_out(audio_hidden_states)
    audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
    audio_output = self.audio_proj_out(audio_hidden_states)

    # ------------------------------------------------------------------ #
    # Assemble output following FastGen convention
    # ------------------------------------------------------------------ #
    if return_features_early:
        # Should have been caught above; guard for safety
        assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
        return features

    # Logvar (optional — requires logvar_linear to be added to the transformer)
    logvar = None
    if return_logvar:
        assert hasattr(self, "logvar_linear"), (
            "logvar_linear is required when return_logvar=True. "
            "It is added by LTX2.__init__."
        )
        # temb has shape [B, T_tokens, inner_dim]; take mean over tokens for a scalar logvar per sample
        logvar = self.logvar_linear(temb.mean(dim=1))  # [B, 1]

    if len(feature_indices) == 0:
        out = (video_output, audio_output)
    else:
        out = [(video_output, audio_output), features]

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if return_logvar:
        return out, logvar
    return out


# ---------------------------------------------------------------------------
# Text encoder wrapper
# ---------------------------------------------------------------------------

class LTX2TextEncoder(nn.Module):
    """
    Wraps Gemma3ForConditionalGeneration for LTX-2 text conditioning.

    Returns both the packed prompt embeddings AND the tokenizer attention mask,
    which is required by LTX2TextConnectors.
    """

    def __init__(self, model_id: str):
        super().__init__()
        self.tokenizer = GemmaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, subfolder="text_encoder"
        )
        self.text_encoder.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(
        self,
        prompt: Union[str, List[str]],
        precision: torch.dtype = torch.bfloat16,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompt(s) into packed Gemma hidden states.

        Returns
        -------
        prompt_embeds : torch.Tensor  [B, T, H * num_layers]
        attention_mask : torch.Tensor  [B, T]
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        device = next(self.text_encoder.parameters()).device

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids      = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Stack all hidden states: [B, T, H, num_layers]
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        sequence_lengths = attention_mask.sum(dim=-1)

        prompt_embeds = _pack_text_embeds(
            hidden_states,
            sequence_lengths,
            device=device,
            padding_side=self.tokenizer.padding_side,
            scale_factor=scale_factor,
        ).to(precision)

        return prompt_embeds, attention_mask

    def to(self, *args, **kwargs):
        self.text_encoder.to(*args, **kwargs)
        return self


# ---------------------------------------------------------------------------
# Main LTX-2 network — follows FastGen pattern (Flux / Wan)
# ---------------------------------------------------------------------------

class LTX2(FastGenNetwork):
    """
    FastGen wrapper for LTX-2 audio-video generation.

    Distillation targets video only:
      - forward() receives and returns video latents [B, C, F, H, W]
      - Audio is generated internally but not used for the distillation loss
      - classify_forward extracts video hidden_states at requested block indices

    Component layout:
        text_encoder → connectors → transformer (patched) → vae → audio_vae → vocoder
    """

    MODEL_ID = "Lightricks/LTX-2"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        disable_grad_ckpt: bool = False,
        load_pretrained: bool = True,
        **model_kwargs,
    ):
        """
        LTX-2 constructor.

        Args:
            model_id: HuggingFace model ID or local path. Defaults to "Lightricks/LTX-2".
            net_pred_type: Prediction type. Defaults to "flow" (flow matching).
            schedule_type: Schedule type. Defaults to "rf" (rectified flow).
            disable_grad_ckpt: Disable gradient checkpointing during training.
                Set True when using FSDP to avoid memory access errors.
            load_pretrained: Load pretrained weights. If False, initialises from config only.
        """
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        self.model_id = model_id
        self._disable_grad_ckpt = disable_grad_ckpt

        self._initialize_network(model_id, load_pretrained)

        # Monkey-patch classify_forward onto self.transformer (same pattern as Flux / Wan)
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Gradient checkpointing
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_network(self, model_id: str, load_pretrained: bool) -> None:
        """Initialize the transformer and supporting modules."""
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading LTX-2 transformer from pretrained")
            self.transformer: LTX2VideoTransformer3DModel = LTX2VideoTransformer3DModel.from_pretrained(
                model_id, subfolder="transformer"
            )
        else:
            config = LTX2VideoTransformer3DModel.load_config(model_id, subfolder="transformer")
            if in_meta_context:
                logger.info(
                    "Initializing LTX-2 transformer on meta device "
                    "(zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing LTX-2 transformer from config (no pretrained weights)")
                logger.warning("LTX-2 transformer being initialized from config. No weights are loaded!")
            self.transformer: LTX2VideoTransformer3DModel = LTX2VideoTransformer3DModel.from_config(config)

        # inner_dim = num_attention_heads * attention_head_dim
        inner_dim = (
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )

        # Add logvar_linear for uncertainty weighting (DMD2 / f-distill)
        # temb mean has shape [B, inner_dim] → logvar scalar per sample
        self.transformer.logvar_linear = nn.Linear(inner_dim, 1)
        logger.info(f"Added logvar_linear ({inner_dim} → 1) to LTX-2 transformer")

        # Connectors: top-level sibling of transformer (NOT nested inside it)
        if should_load_weights:
            self.connectors: LTX2TextConnectors = LTX2TextConnectors.from_pretrained(
                model_id, subfolder="connectors"
            )
        else:
            # Connectors are lightweight; always load if pretrained is skipped for the transformer
            logger.warning("Skipping connector pretrained load (meta context or load_pretrained=False)")
            self.connectors = None  # will be loaded lazily via init_preprocessors

        # Cache compression ratios used by forward() and sample()
        if should_load_weights:
            # VAEs (needed for sample(); not for the training forward pass)
            self.vae: AutoencoderKLLTX2Video = AutoencoderKLLTX2Video.from_pretrained(
                model_id, subfolder="vae"
            )
            self.vae.eval().requires_grad_(False)

            self.audio_vae: AutoencoderKLLTX2Audio = AutoencoderKLLTX2Audio.from_pretrained(
                model_id, subfolder="audio_vae"
            )
            self.audio_vae.eval().requires_grad_(False)

            self.vocoder: LTX2Vocoder = LTX2Vocoder.from_pretrained(
                model_id, subfolder="vocoder"
            )
            self.vocoder.eval().requires_grad_(False)

            self._cache_vae_constants()

        # Scheduler (used in sample())
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

    def _cache_vae_constants(self) -> None:
        """Cache VAE spatial/temporal compression constants for use in forward() / sample()."""
        self.vae_spatial_compression_ratio   = self.vae.spatial_compression_ratio
        self.vae_temporal_compression_ratio  = self.vae.temporal_compression_ratio
        self.transformer_spatial_patch_size  = self.transformer.config.patch_size
        self.transformer_temporal_patch_size = self.transformer.config.patch_size_t

        self.audio_sampling_rate             = self.audio_vae.config.sample_rate
        self.audio_hop_length                = self.audio_vae.config.mel_hop_length
        self.audio_vae_temporal_compression  = self.audio_vae.temporal_compression_ratio
        self.audio_vae_mel_compression_ratio = self.audio_vae.mel_compression_ratio

    # ------------------------------------------------------------------
    # Preprocessor initialisation (lazy, matches Flux / Wan pattern)
    # ------------------------------------------------------------------

    def init_preprocessors(self):
        """Initialize text encoder and connectors."""
        if not hasattr(self, "text_encoder") or self.text_encoder is None:
            self.init_text_encoder()
        if self.connectors is None:
            self.connectors = LTX2TextConnectors.from_pretrained(
                self.model_id, subfolder="connectors"
            )

    def init_text_encoder(self):
        """Initialize the Gemma3 text encoder for LTX-2."""
        self.text_encoder = LTX2TextEncoder(model_id=self.model_id)

    # ------------------------------------------------------------------
    # Device movement
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "connectors") and self.connectors is not None:
            self.connectors.to(*args, **kwargs)
        if hasattr(self, "vae") and self.vae is not None:
            self.vae.to(*args, **kwargs)
        if hasattr(self, "audio_vae") and self.audio_vae is not None:
            self.audio_vae.to(*args, **kwargs)
        if hasattr(self, "vocoder") and self.vocoder is not None:
            self.vocoder.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # FSDP
    # ------------------------------------------------------------------

    def fully_shard(self, **kwargs):
        """Fully shard the LTX-2 transformer for FSDP2.

        Shards self.transformer (not self) to avoid ABC __class__ assignment issues.
        """
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(
                self.transformer,
                check_fn=lambda block: isinstance(block, LTX2VideoTransformerBlock),
            )
            logger.info("Applied FSDP activation checkpointing to LTX-2 transformer blocks")

        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

    # ------------------------------------------------------------------
    # reset_parameters (required for FSDP meta device init)
    # ------------------------------------------------------------------

    def reset_parameters(self):
        """Reinitialise parameters after meta device materialisation (FSDP2)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        super().reset_parameters()
        logger.debug("Reinitialized LTX-2 parameters")

    # ------------------------------------------------------------------
    # Audio latent sizing helper (shared by forward and sample)
    # ------------------------------------------------------------------

    def _compute_audio_shape(
        self, latent_f: int, fps: float, device: torch.device, dtype: torch.dtype
    ) -> Tuple[int, int, int]:
        """
        Compute audio latent dimensions from video latent frame count.

        Returns (audio_num_frames, latent_mel_bins, num_audio_ch).
        """
        pixel_frames = (latent_f - 1) * self.vae_temporal_compression_ratio + 1
        duration_s   = pixel_frames / fps

        audio_latents_per_second = (
            self.audio_sampling_rate
            / self.audio_hop_length
            / float(self.audio_vae_temporal_compression)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        num_mel_bins     = self.audio_vae.config.mel_bins
        latent_mel_bins  = num_mel_bins // self.audio_vae_mel_compression_ratio
        num_audio_ch     = self.audio_vae.config.latent_channels
        return audio_num_frames, latent_mel_bins, num_audio_ch

    # ------------------------------------------------------------------
    # forward() — video-only distillation interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,           # unused, kept for API compatibility
        fps: float = 24.0,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        audio_latents: Optional[torch.Tensor] = None,
        return_audio: bool = False,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass — video latents in, video latents out.

        Follows the Flux/Wan FastGen pattern. Audio is optional:
          - Distillation (default): ``audio_latents=None, return_audio=False``.
            Random noise is used for audio; only video prediction is returned.
          - Inference via sample(): ``audio_latents=<packed>, return_audio=True``.
            The current denoising audio latents are passed in and the audio
            noise prediction is returned alongside the video prediction so that
            sample() can step both schedulers.

        Args:
            x_t: Video latents [B, C, F, H, W].
            t:   Timestep [B] — scheduler sigmas passed directly to time_embed.
            condition: Tuple of (prompt_embeds [B, T, D], attention_mask [B, T]).
            r:   Unused (kept for FastGen API compatibility).
            fps: Frames per second (needed for RoPE coordinate computation).
            return_features_early: Return video features as soon as collected.
            feature_indices: Set of transformer block indices to extract video features from.
            return_logvar: Return log-variance estimate alongside the output.
            fwd_pred_type: Override prediction type.
            audio_latents: Optional packed audio latents [B, T_a, C_a] from sample().
                           When None, fresh random noise is generated internally.
            return_audio: When True, return (video_out, audio_packed) instead of
                          just video_out. Only used by sample().

        Returns:
            Normal:        video_out [B, C, F, H, W]
            return_audio:  (video_out, audio_packed [B, T_a, C_a])
            With features: (video_out, List[video_feature_tensors])
            Early exit:    List[video_feature_tensors]
            With logvar:   (above, logvar [B, 1])
        """
        if feature_indices is None:
            feature_indices = set()
        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported"

        batch_size = x_t.shape[0]
        _, _, latent_f, latent_h, latent_w = x_t.shape

        # Unpack text conditioning
        prompt_embeds, attention_mask = condition

        # ---- Run connectors to get per-modality encoder hidden states ----
        additive_mask = (1 - attention_mask.to(prompt_embeds.dtype)) * -1_000_000.0
        connector_video_embeds, connector_audio_embeds, connector_attn_mask = self.connectors(
            prompt_embeds, additive_mask, additive_mask=True
        )

        # ---- Timestep: [B] — time_embed handles scale internally ----
        timestep = t.to(x_t.dtype).expand(batch_size)  # [B]

        # ---- Pack video latents ----
        hidden_states = _pack_latents(
            x_t, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )

        # ---- Audio latents: use provided (from sample()) or generate random noise ----
        audio_num_frames, latent_mel_bins, num_audio_ch = self._compute_audio_shape(
            latent_f, fps, x_t.device, x_t.dtype
        )
        if audio_latents is not None:
            # Already packed [B, T_a, C_a] — passed in by sample()
            audio_hidden_states = audio_latents.to(x_t.dtype)
        else:
            audio_hidden_states = _pack_audio_latents(
                torch.randn(
                    batch_size, num_audio_ch, audio_num_frames, latent_mel_bins,
                    device=x_t.device, dtype=x_t.dtype,
                )
            )

        # ---- RoPE coordinates ----
        video_coords = self.transformer.rope.prepare_video_coords(
            batch_size, latent_f, latent_h, latent_w, x_t.device, fps=fps
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            batch_size, audio_num_frames, x_t.device
        )

        # ---- Transformer forward (classify_forward, monkey-patched) ----
        model_outputs = self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=connector_video_embeds,
            audio_encoder_hidden_states=connector_audio_embeds,
            encoder_attention_mask=connector_attn_mask,
            audio_encoder_attention_mask=connector_attn_mask,
            timestep=timestep,
            num_frames=latent_f,
            height=latent_h,
            width=latent_w,
            fps=fps,
            audio_num_frames=audio_num_frames,
            video_coords=video_coords,
            audio_coords=audio_coords,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
        )

        # ---- Early exit: list of video feature tensors ----
        if return_features_early:
            return model_outputs  # List[Tensor], each [B, T_v, D_v]

        # ---- Unpack logvar if requested ----
        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        # ---- Extract video prediction; capture audio for sample() if requested ----
        if len(feature_indices) == 0:
            # out is (video_output, audio_output)
            video_packed = out[0]  # [B, T_v, C_packed]
            audio_packed = out[1]  # [B, T_a, C_packed] — only used when return_audio=True
            features = None
        else:
            # out is [(video_output, audio_output), features]
            video_packed = out[0][0]  # [B, T_v, C_packed]
            audio_packed = out[0][1]  # [B, T_a, C_packed]
            features = out[1]         # List[Tensor]

        # ---- Unpack video tokens → [B, C, F, H, W] ----
        video_out = _unpack_latents(
            video_packed, latent_f, latent_h, latent_w,
            self.transformer_spatial_patch_size, self.transformer_temporal_patch_size,
        )

        # ---- Convert model output to requested prediction type ----
        video_out = self.noise_scheduler.convert_model_output(
            x_t, video_out, t,
            src_pred_type=self.net_pred_type,
            target_pred_type=fwd_pred_type,
        )

        # ---- Re-pack output following FastGen convention ----
        if features is not None:
            out = [video_out, features]
        else:
            out = video_out

        # Return audio noise pred alongside video when called from sample()
        if return_audio:
            out = (out, audio_packed.float())

        if return_logvar:
            return out, logvar
        return out

    # ------------------------------------------------------------------
    # sample() — full denoising loop for inference
    # Follows pipeline_ltx2.py exactly (verified working logic preserved)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        neg_condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        guidance_scale: float = 4.0,
        num_steps: int = 40,
        fps: float = 24.0,
        frame_rate: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        Run the full denoising loop for text-to-video generation (audio always None).

        Returns
        -------
        (video_latents, None):
            video: [B, C, F, H, W] denormalised video latents
            audio: always None
        """
        fps = frame_rate if frame_rate is not None else fps
        do_cfg = neg_condition is not None and guidance_scale > 1.0

        transformer_dtype = self.transformer.dtype
        transformer_device = next(self.transformer.parameters()).device

        # Move latents to transformer device and dtype
        video_latents = noise.to(device=transformer_device, dtype=transformer_dtype)

        # Build combined condition for CFG so connectors are called once per step
        if do_cfg:
            neg_embeds, neg_mask = neg_condition
            cond_embeds, cond_mask = condition
            combined_condition = (
                torch.cat([neg_embeds, cond_embeds], dim=0).to(device=transformer_device, dtype=transformer_dtype),
                torch.cat([neg_mask, cond_mask], dim=0).to(device=transformer_device),
            )
        else:
            embeds, mask = condition
            combined_condition = (
                embeds.to(device=transformer_device, dtype=transformer_dtype),
                mask.to(device=transformer_device),
            )

        # ---- Scheduler timesteps ----
        B, C, latent_f, latent_h, latent_w = video_latents.shape
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        video_seq_len = latent_f * latent_h * latent_w
        mu = _calculate_shift(
            video_seq_len,
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        timesteps, num_steps = _retrieve_timesteps(
            self.scheduler, num_steps, transformer_device, sigmas=sigmas, mu=mu
        )

        # ---- Denoising loop ----
        for t in timesteps:
            latent_input = torch.cat([video_latents] * 2) if do_cfg else video_latents
            t_input = t.to(dtype=transformer_dtype, device=transformer_device).expand(latent_input.shape[0])

            noise_pred = self(
                latent_input,
                t_input,
                condition=combined_condition,
                fps=fps,
                fwd_pred_type="flow",
            )

            if do_cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            video_latents = self.scheduler.step(noise_pred, t, video_latents, return_dict=False)[0]

        # ---- Denormalise ----
        video_latents = _denormalize_latents(
            video_latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

        return video_latents, None