# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 FastGen network implementation (video-only).

Uses the local customized pipeline_ltx2.py and transformer_ltx2.py which
support audio_enabled=False, so no audio weights are allocated and no audio
ops run during training or inference.

Follows the FastGen network pattern established by Flux and Wan:
  - Inherits from FastGenNetwork
  - Monkey-patches classify_forward onto self.transformer
  - forward() operates entirely in video latent space [B, C, F, H, W]
  - sample() calls self() (i.e. forward()) — NOT self.transformer directly
  - feature_indices extracts video hidden_states for the discriminator
"""

import types
from typing import Any, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

# ---- Local customized modules (same folder as this file) ----
from .pipeline_ltx2 import LTX2Pipeline
from .transformer_ltx2 import LTX2VideoTransformer3DModel, LTX2VideoTransformerBlock

from diffusers.models.autoencoders import AutoencoderKLLTX2Video
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger


# ---------------------------------------------------------------------------
# Latent pack / unpack helpers (video only)
# ---------------------------------------------------------------------------

def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    """[B, C, F, H, W] → [B, F//pt * H//p * W//p, C*pt*p*p]"""
    B, C, F, H, W = latents.shape
    latents = latents.reshape(B, C, F // patch_size_t, patch_size_t,
                               H // patch_size, patch_size,
                               W // patch_size, patch_size)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int,
    patch_size: int = 1, patch_size_t: int = 1,
) -> torch.Tensor:
    """[B, T, D] → [B, C, F, H, W]"""
    B = latents.size(0)
    latents = latents.reshape(B, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


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


def _pack_text_embeds(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    device: torch.device,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    B, T, H, L = text_hidden_states.shape
    original_dtype = text_hidden_states.dtype

    token_indices = torch.arange(T, device=device).unsqueeze(0)
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    else:
        start = T - sequence_lengths[:, None]
        mask = token_indices >= start
    mask = mask[:, :, None, None]

    masked = text_hidden_states.masked_fill(~mask, 0.0)
    num_valid = (sequence_lengths * H).view(B, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (num_valid + eps)
    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

    normed = (text_hidden_states - mean) / (x_max - x_min + eps) * scale_factor
    normed = normed.flatten(2)
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
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, mu=None):
    kwargs = {}
    if mu is not None:
        kwargs["mu"] = mu
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


# ---------------------------------------------------------------------------
# classify_forward — monkey-patched onto self.transformer (video-only)
# ---------------------------------------------------------------------------

def classify_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    num_frames: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    fps: float = 24.0,
    video_coords: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[dict] = None,
    return_dict: bool = False,
    # FastGen distillation kwargs
    return_features_early: bool = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: bool = False,
) -> Union[
    torch.Tensor,                                    # video_output only
    Tuple[torch.Tensor, List[torch.Tensor]],         # (video_output, features)
    List[torch.Tensor],                              # features only (early exit)
]:
    """
    Video-only classify_forward monkey-patched onto LTX2VideoTransformer3DModel.

    Since the transformer is built with audio_enabled=False, all audio arguments
    are absent. Only video hidden_states are processed and stored as features.

    Returns
    -------
    Normal mode  (feature_indices empty, return_features_early False):
        video_output  [B, T_v, C_out]

    Feature mode (feature_indices non-empty, return_features_early False):
        (video_output, List[video_feature_tensors])

    Early-exit mode (return_features_early True):
        List[video_feature_tensors]
    """
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

    # -- Attention mask conversion --
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    batch_size = hidden_states.size(0)

    # 1. RoPE
    if video_coords is None:
        video_coords = self.rope.prepare_video_coords(
            batch_size, num_frames, height, width, hidden_states.device, fps=fps
        )
    video_rotary_emb = self.rope(video_coords, device=hidden_states.device)

    # 2. Patchify
    hidden_states = self.proj_in(hidden_states)

    # 3. Timestep embeddings
    temb, embedded_timestep = self.time_embed(
        timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype,
    )
    temb = temb.view(batch_size, -1, temb.size(-1))
    embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

    # 4. Prompt embeddings
    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

    # 5. Block loop — video only, with feature extraction
    features: List[torch.Tensor] = []

    for idx, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states, _ = self._gradient_checkpointing_func(
                block,
                hidden_states,
                None,   # audio_hidden_states — not used
                encoder_hidden_states,
                None,   # audio_encoder_hidden_states — not used
                temb,
                None,   # temb_audio
                None,   # video_cross_attn_scale_shift
                None,   # audio_cross_attn_scale_shift
                None,   # video_cross_attn_a2v_gate
                None,   # audio_cross_attn_v2a_gate
                video_rotary_emb,
                None,   # audio_rotary_emb
                None,   # video_cross_attn_rotary_emb
                None,   # audio_cross_attn_rotary_emb
                encoder_attention_mask,
                None,   # audio_encoder_attention_mask
            )
        else:
            hidden_states, _ = block(
                hidden_states=hidden_states,
                audio_hidden_states=None,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=None,
                temb=temb,
                temb_audio=None,
                temb_ca_scale_shift=None,
                temb_ca_audio_scale_shift=None,
                temb_ca_gate=None,
                temb_ca_audio_gate=None,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=None,
                ca_video_rotary_emb=None,
                ca_audio_rotary_emb=None,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=None,
                audio_enabled=False,
            )

        if idx in feature_indices:
            features.append(hidden_states.clone())

        if return_features_early and len(features) == len(feature_indices):
            if USE_PEFT_BACKEND:
                unscale_lora_layers(self, lora_scale)
            return features

    # 6. Output layer
    scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
    shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
    hidden_states = self.norm_out(hidden_states)
    hidden_states = hidden_states * (1 + scale) + shift
    video_output = self.proj_out(hidden_states)

    # -- Logvar (optional) --
    logvar = None
    if return_logvar:
        assert hasattr(self, "logvar_linear"), (
            "logvar_linear must exist on transformer. It is added by LTX2.__init__."
        )
        logvar = self.logvar_linear(temb.mean(dim=1))  # [B, 1]

    # -- Assemble output --
    if len(feature_indices) == 0:
        out = video_output
    else:
        out = (video_output, features)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if return_logvar:
        return out, logvar
    return out


# ---------------------------------------------------------------------------
# Text encoder wrapper
# ---------------------------------------------------------------------------

class LTX2TextEncoder(nn.Module):
    """Wraps Gemma3 text encoder for LTX-2 conditioning."""

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
# Main LTX-2 network (video-only)
# ---------------------------------------------------------------------------

class LTX2(FastGenNetwork):
    """
    FastGen wrapper for LTX-2, video-only distillation.

    Uses the local customized transformer_ltx2.py (audio_enabled=False) and
    pipeline_ltx2.py (generate_audio=False) so no audio weights are allocated
    and no audio ops run at any point.

    Distillation targets video only:
      - forward() receives and returns video latents [B, C, F, H, W]
      - classify_forward extracts video hidden_states at requested block indices
      - sample() calls self() (forward()) — the pipeline is used only for its
        helper utilities (latent prep, scheduler config)
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
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        self.model_id = model_id
        self._disable_grad_ckpt = disable_grad_ckpt

        self._initialize_network(model_id, load_pretrained)

        # Monkey-patch classify_forward (video-only version)
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_network(self, model_id: str, load_pretrained: bool) -> None:
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and not in_meta_context

        # -- Transformer: audio_enabled=False → no audio weights allocated --
        if should_load_weights:
            logger.info("Loading LTX-2 transformer from pretrained (audio_enabled=False)")
            # Load pretrained AV checkpoint then drop audio keys via strict=False
            av_transformer = LTX2VideoTransformer3DModel.from_pretrained(
                model_id, subfolder="transformer"
            )
            config = av_transformer.config
            # Build video-only transformer
            self.transformer = LTX2VideoTransformer3DModel.from_config(config, audio_enabled=False)
            missing, unexpected = self.transformer.load_state_dict(
                av_transformer.state_dict(), strict=False
            )
            assert len(missing) == 0, f"Missing video keys: {missing}"
            logger.info(f"Dropped {len(unexpected)} audio keys from pretrained checkpoint")
            del av_transformer
        else:
            config = LTX2VideoTransformer3DModel.load_config(model_id, subfolder="transformer")
            if in_meta_context:
                logger.info("Initializing LTX-2 transformer on meta device (audio_enabled=False)")
            else:
                logger.warning("LTX-2 transformer initialized from config only — no pretrained weights!")
            self.transformer = LTX2VideoTransformer3DModel.from_config(config, audio_enabled=False)

        # inner_dim for logvar_linear
        inner_dim = (
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )
        self.transformer.logvar_linear = nn.Linear(inner_dim, 1)
        logger.info(f"Added logvar_linear ({inner_dim} → 1) to transformer")

        # -- Connectors --
        if should_load_weights:
            self.connectors: LTX2TextConnectors = LTX2TextConnectors.from_pretrained(
                model_id, subfolder="connectors"
            )
        else:
            logger.warning("Skipping connector pretrained load")
            self.connectors = None

        # -- VAE (video only — no audio_vae, no vocoder) --
        if should_load_weights:
            self.vae: AutoencoderKLLTX2Video = AutoencoderKLLTX2Video.from_pretrained(
                model_id, subfolder="vae"
            )
            self.vae.eval().requires_grad_(False)
            self._cache_vae_constants()

        # -- Scheduler --
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

    def _cache_vae_constants(self) -> None:
        self.vae_spatial_compression_ratio  = self.vae.spatial_compression_ratio
        self.vae_temporal_compression_ratio = self.vae.temporal_compression_ratio
        self.transformer_spatial_patch_size  = self.transformer.config.patch_size
        self.transformer_temporal_patch_size = self.transformer.config.patch_size_t

    # ------------------------------------------------------------------
    # Preprocessors (lazy)
    # ------------------------------------------------------------------

    def init_preprocessors(self):
        if not hasattr(self, "text_encoder") or self.text_encoder is None:
            self.init_text_encoder()
        if self.connectors is None:
            self.connectors = LTX2TextConnectors.from_pretrained(
                self.model_id, subfolder="connectors"
            )

    def init_text_encoder(self):
        self.text_encoder = LTX2TextEncoder(model_id=self.model_id)

    # ------------------------------------------------------------------
    # Device movement
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for attr in ("text_encoder", "connectors", "vae"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # FSDP
    # ------------------------------------------------------------------

    def fully_shard(self, **kwargs):
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(
                self.transformer,
                check_fn=lambda b: isinstance(b, LTX2VideoTransformerBlock),
            )
            logger.info("Applied FSDP activation checkpointing to transformer blocks")

        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

    # ------------------------------------------------------------------
    # reset_parameters (FSDP meta device)
    # ------------------------------------------------------------------

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        super().reset_parameters()

    # ------------------------------------------------------------------
    # forward() — video-only distillation interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,
        fps: float = 24.0,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple]:
        """
        Training forward pass: video latents [B, C, F, H, W] → video latents.

        Args:
            x_t:       Noisy video latents [B, C, F, H, W].
            t:         Timesteps [B].
            condition: (prompt_embeds [B, T, D], attention_mask [B, T]).
            fps:       Frames per second for RoPE coords.
            return_features_early: Exit once all feature_indices are collected.
            feature_indices:       Block indices to extract hidden states from.
            return_logvar:         Return log-variance alongside output.
            fwd_pred_type:         Override prediction type.

        Returns:
            Normal:        video_out [B, C, F, H, W]
            With features: (video_out, List[feature_tensors])
            Early exit:    List[feature_tensors]
            With logvar:   (above, logvar [B, 1])
        """
        if feature_indices is None:
            feature_indices = set()
        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"Unsupported pred type: {fwd_pred_type}"

        batch_size = x_t.shape[0]
        _, _, latent_f, latent_h, latent_w = x_t.shape

        # -- Text conditioning --
        prompt_embeds, attention_mask = condition
        additive_mask = (1 - attention_mask.to(prompt_embeds.dtype)) * -1_000_000.0
        # Connectors return (video_embeds, audio_embeds, attn_mask);
        # we only use the video branch since audio_enabled=False.
        connector_video_embeds, _, connector_attn_mask = self.connectors(
            prompt_embeds, additive_mask, additive_mask=True
        )

        # -- Timestep --
        timestep = t.to(x_t.dtype).expand(batch_size)

        # -- Pack video latents --
        hidden_states = _pack_latents(
            x_t, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )

        # -- RoPE video coords --
        video_coords = self.transformer.rope.prepare_video_coords(
            batch_size, latent_f, latent_h, latent_w, x_t.device, fps=fps
        )

        # -- Transformer forward (our patched classify_forward, video-only) --
        model_outputs = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=connector_video_embeds,
            encoder_attention_mask=connector_attn_mask,
            timestep=timestep,
            num_frames=latent_f,
            height=latent_h,
            width=latent_w,
            fps=fps,
            video_coords=video_coords,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
        )

        # -- Early exit --
        if return_features_early:
            return model_outputs  # List[Tensor]

        # -- Unpack logvar --
        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        # -- Separate video output from features --
        if len(feature_indices) == 0:
            video_packed = out          # [B, T_v, C]
            features = None
        else:
            video_packed, features = out[0], out[1]

        # -- Unpack → [B, C, F, H, W] --
        video_out = _unpack_latents(
            video_packed, latent_f, latent_h, latent_w,
            self.transformer_spatial_patch_size, self.transformer_temporal_patch_size,
        )

        # -- Prediction type conversion --
        video_out = self.noise_scheduler.convert_model_output(
            x_t, video_out, t,
            src_pred_type=self.net_pred_type,
            target_pred_type=fwd_pred_type,
        )

        # -- Assemble final output --
        if features is not None:
            out = (video_out, features)
        else:
            out = video_out

        if return_logvar:
            return out, logvar
        return out

    # ------------------------------------------------------------------
    # sample() — full denoising loop; calls self() (forward())
    # Works entirely in unpacked [B, C, F, H, W] space so forward() is
    # called with the correct input shape at every step.
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
        Denoising loop for video generation. Calls self() at each step.

        noise must be unpacked video latents [B, C, F, H, W].

        Returns
        -------
        (video_latents [B, C, F, H, W], None)
            Denormalised video latents ready for VAE decode. Audio is always None.
        """
        fps = frame_rate if frame_rate is not None else fps
        do_cfg = neg_condition is not None and guidance_scale > 1.0

        transformer_dtype = self.transformer.dtype
        transformer_device = next(self.transformer.parameters()).device

        # noise must arrive as [B, C, F, H, W] (unpacked)
        assert noise.ndim == 5, "sample() expects unpacked latents [B, C, F, H, W]"
        video_latents = noise.to(device=transformer_device, dtype=transformer_dtype)

        # -- Build combined CFG condition (processed once, reused every step) --
        if do_cfg:
            neg_embeds, neg_mask = neg_condition
            cond_embeds, cond_mask = condition
            combined_condition = (
                torch.cat([neg_embeds, cond_embeds], dim=0).to(
                    device=transformer_device, dtype=transformer_dtype
                ),
                torch.cat([neg_mask, cond_mask], dim=0).to(device=transformer_device),
            )
        else:
            embeds, mask = condition
            combined_condition = (
                embeds.to(device=transformer_device, dtype=transformer_dtype),
                mask.to(device=transformer_device),
            )

        # -- Scheduler timesteps with mu shift --
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

        # -- Denoising loop (unpacked latents throughout) --
        for t in timesteps:
            # Duplicate along batch for CFG
            latent_input = torch.cat([video_latents] * 2) if do_cfg else video_latents
            t_input = (
                t.to(dtype=transformer_dtype, device=transformer_device)
                .expand(latent_input.shape[0])
            )

            # self() → forward() — expects and returns [B, C, F, H, W]
            noise_pred = self(
                latent_input,
                t_input,
                condition=combined_condition,
                fps=fps,
                fwd_pred_type="flow",
            )

            if do_cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            # Scheduler step — operates on packed tokens internally but we keep
            # video_latents unpacked; use the pipeline's _pack/_unpack helpers.
            video_packed = _pack_latents(
                video_latents,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
            noise_pred_packed = _pack_latents(
                noise_pred,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
            stepped_packed = self.scheduler.step(
                noise_pred_packed, t, video_packed, return_dict=False
            )[0]
            video_latents = _unpack_latents(
                stepped_packed, latent_f, latent_h, latent_w,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )

        # -- Denormalise --
        video_latents = _denormalize_latents(
            video_latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

        return video_latents, None
