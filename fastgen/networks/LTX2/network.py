# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 FastGen network implementation.

Architecture verified against:
  - diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2.py
  - diffusers/src/diffusers/pipelines/ltx2/connectors.py
  - diffusers/src/diffusers/models/transformers/transformer_ltx2.py
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from diffusers.models.transformers import LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast


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
    """Unpack audio latents [B, L, C*M] -> [B, C, L, M].
    Mirrors pipeline: latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
    unflatten splits last dim 128 -> [C=8, M=16], giving [B, L, C, M],
    then transpose(1, 2) -> [B, C, L, M].
    """
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

    Args:
        text_hidden_states: [B, T, H, num_layers]  (stacked output from Gemma)
        sequence_lengths:   [B]                    (number of non-padded tokens)
    Returns:
        [B, T, H * num_layers]
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

    @torch.no_grad()
    def encode(
        self,
        prompt: str | list[str],
        precision: torch.dtype = torch.bfloat16,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompt(s) into packed Gemma hidden states.

        Returns
        -------
        prompt_embeds : torch.Tensor  [B, T, H * num_layers]
            Normalised, packed text embeddings ready for LTX2TextConnectors.
        attention_mask : torch.Tensor  [B, T]
            Binary padding mask (1 = real token, 0 = pad) from the tokenizer.
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

        # Bug 1 fix: return BOTH embeds and mask
        return prompt_embeds, attention_mask


# ---------------------------------------------------------------------------
# Main LTX-2 network
# ---------------------------------------------------------------------------

class LTX2(nn.Module):
    """
    FastGen wrapper for LTX-2 audio-video generation.

    Component layout (mirrors the diffusers model_cpu_offload_seq):
        text_encoder → connectors → transformer → vae → audio_vae → vocoder

    Notes
    -----
    * ``connectors`` lives as a sibling of ``transformer``, NOT nested inside
      it (Bug 5 fix).  Connectors process the text embeddings once before the
      denoising loop and produce separate video and audio encoder hidden states.
    * No monkey-patching of the transformer is needed: LTX2VideoTransformer3DModel
      handles its own block loop with the correct audio/video dual-branch logic
      internally (Bug 3 fix).
    """

    def __init__(self, model_id: str = "Lightricks/LTX-2", load_pretrained: bool = True):
        super().__init__()
        self.model_id = model_id
        self._initialized = False
        if load_pretrained:
            self._initialize_network()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_network(self):
        model_id = self.model_id

        # Text encoder (Gemma3)
        self.text_encoder = LTX2TextEncoder(model_id)

        # Bug 5 fix: connectors is a TOP-LEVEL sibling of transformer,
        # NOT attached to self.transformer.
        self.connectors = LTX2TextConnectors.from_pretrained(
            model_id, subfolder="connectors"
        )

        # Transformer (LTX2VideoTransformer3DModel handles all block logic)
        self.transformer = LTX2VideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer"
        )

        # VAEs
        self.vae = AutoencoderKLLTX2Video.from_pretrained(
            model_id, subfolder="vae"
        )
        self.audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
            model_id, subfolder="audio_vae"
        )

        # Vocoder (mel spectrogram → waveform)
        self.vocoder = LTX2Vocoder.from_pretrained(
            model_id, subfolder="vocoder"
        )

        # Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

        # Cache compression ratios
        self.vae_spatial_compression_ratio   = self.vae.spatial_compression_ratio
        self.vae_temporal_compression_ratio  = self.vae.temporal_compression_ratio
        self.transformer_spatial_patch_size  = self.transformer.config.patch_size
        self.transformer_temporal_patch_size = self.transformer.config.patch_size_t

        # Audio VAE constants.
        # sample_rate / mel_hop_length live in config; the compression ratios are
        # instance attributes (LATENT_DOWNSAMPLE_FACTOR = 4) set in __init__, not in config.
        self.audio_sampling_rate             = self.audio_vae.config.sample_rate       # 16000
        self.audio_hop_length                = self.audio_vae.config.mel_hop_length    # 160
        self.audio_vae_temporal_compression  = self.audio_vae.temporal_compression_ratio  # 4
        self.audio_vae_mel_compression_ratio = self.audio_vae.mel_compression_ratio    # 4

        self._initialized = True

    def init_preprocessors(self):
        """No-op placeholder (preprocessors are initialised in _initialize_network)."""
        pass

    # ------------------------------------------------------------------
    # Forward pass (single denoising step)
    # ------------------------------------------------------------------

    def forward(
        self,
        # packed video latents  [B, T_v, C_v]
        hidden_states: torch.Tensor,
        # packed audio latents  [B, T_a, C_a]
        audio_hidden_states: torch.Tensor,
        # already-projected by connectors
        encoder_hidden_states: torch.Tensor,        # video text embeds [B, T_t, D_v]
        audio_encoder_hidden_states: torch.Tensor,  # audio text embeds [B, T_t, D_a]
        encoder_attention_mask: torch.Tensor,       # [B, T_t]
        timestep: torch.Tensor,
        # spatial metadata for RoPE
        num_frames: int,
        height: int,
        width: int,
        audio_num_frames: int,
        fps: float = 24.0,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single denoising step through the transformer.

        Returns
        -------
        (noise_pred_video, noise_pred_audio) both in packed token format.
        """
        noise_pred_video, noise_pred_audio = self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=encoder_attention_mask,  # same mask for audio
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            audio_num_frames=audio_num_frames,
            video_coords=video_coords,
            audio_coords=audio_coords,
            return_dict=False,
        )
        return noise_pred_video, noise_pred_audio

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        condition: tuple[torch.Tensor, torch.Tensor],
        neg_condition: tuple[torch.Tensor, torch.Tensor] | None = None,
        guidance_scale: float = 4.0,
        num_steps: int = 40,
        fps: float = 24.0,
        frame_rate: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full denoising loop for text-to-video+audio generation.

        Follows pipeline_ltx2.py exactly:
          - latents kept in float32 throughout; only cast to prompt dtype for transformer
          - noise predictions cast to float32 before CFG and scheduler step
          - connectors called ONCE on combined [uncond, cond] batch (not twice)
          - audio duration derived from pixel-frame count, not latent frames
          - transformer wrapped in cache_context("cond_uncond")

        Returns
        -------
        (video_latents, audio_latents):
            video: [B, C, F, H, W] denormalised, ready for vae.decode()
            audio: [B, C, L, M]    denormalised, ready for audio_vae.decode()
        """
        fps = frame_rate if frame_rate is not None else fps
        do_cfg = neg_condition is not None and guidance_scale > 1.0

        device = noise.device
        B, C, latent_f, latent_h, latent_w = noise.shape

        # ----------------------------------------------------------------
        # Latent dimensions
        # ----------------------------------------------------------------
        # Audio duration uses pixel-frame count (pipeline line 1003):
        #   duration_s = num_frames / frame_rate
        # where num_frames is the original pixel count. For a causal VAE:
        #   pixel_frames = (latent_f - 1) * temporal_compression + 1
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

        # ----------------------------------------------------------------
        # Pack latents — keep in float32 to match pipeline
        # ----------------------------------------------------------------
        video_latents = _pack_latents(
            noise.float(), self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )

        audio_shape   = (B, num_audio_ch, audio_num_frames, latent_mel_bins)
        audio_latents = torch.randn(audio_shape, device=device, dtype=torch.float32)
        audio_latents = _pack_audio_latents(audio_latents)

        # ----------------------------------------------------------------
        # Text conditioning — run connectors ONCE on combined [uncond, cond]
        # batch, exactly as the pipeline does (lines 959-966)
        # ----------------------------------------------------------------
        prompt_embeds, attention_mask = condition

        if do_cfg:
            neg_embeds, neg_mask = neg_condition
            # Stack [uncond, cond] before connectors — single forward pass
            combined_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
            combined_mask   = torch.cat([neg_mask, attention_mask], dim=0)
        else:
            combined_embeds = prompt_embeds
            combined_mask   = attention_mask

        additive_mask = (1 - combined_mask.to(combined_embeds.dtype)) * -1_000_000.0
        connector_video_embeds, connector_audio_embeds, connector_attn_mask = self.connectors(
            combined_embeds, additive_mask, additive_mask=True
        )

        # ----------------------------------------------------------------
        # Pre-compute RoPE coordinates (pipeline lines 1078-1087)
        # Compute for single batch, then repeat for CFG
        # ----------------------------------------------------------------
        video_coords = self.transformer.rope.prepare_video_coords(
            B, latent_f, latent_h, latent_w, device, fps=fps
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            B, audio_num_frames, device
        )
        if do_cfg:
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))

        # ----------------------------------------------------------------
        # Scheduler timesteps (pipeline lines 1042-1067)
        # ----------------------------------------------------------------
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)

        video_seq_len = latent_f * latent_h * latent_w
        mu = _calculate_shift(
            video_seq_len,
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )

        audio_scheduler = copy.deepcopy(self.scheduler)
        _retrieve_timesteps(audio_scheduler, num_steps, device, sigmas=sigmas, mu=mu)
        timesteps, num_steps = _retrieve_timesteps(self.scheduler, num_steps, device, sigmas=sigmas, mu=mu)

        # ----------------------------------------------------------------
        # Denoising loop (pipeline lines 1091-1154)
        # ----------------------------------------------------------------
        # prompt_embeds.dtype drives the cast for transformer input
        prompt_dtype = connector_video_embeds.dtype

        for t in timesteps:
            # Cast latents to prompt dtype for transformer; keep float32 for scheduler
            latent_input       = torch.cat([video_latents] * 2) if do_cfg else video_latents
            audio_latent_input = torch.cat([audio_latents] * 2) if do_cfg else audio_latents
            latent_input       = latent_input.to(prompt_dtype)
            audio_latent_input = audio_latent_input.to(prompt_dtype)

            timestep = t.expand(latent_input.shape[0])

            # Wrap in cache_context as the pipeline does (CacheMixin optimisation)
            with self.transformer.cache_context("cond_uncond"):
                noise_pred_video, noise_pred_audio = self.forward(
                    hidden_states=latent_input,
                    audio_hidden_states=audio_latent_input,
                    encoder_hidden_states=connector_video_embeds,
                    audio_encoder_hidden_states=connector_audio_embeds,
                    encoder_attention_mask=connector_attn_mask,
                    timestep=timestep,
                    num_frames=latent_f,
                    height=latent_h,
                    width=latent_w,
                    audio_num_frames=audio_num_frames,
                    fps=fps,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                )

            # Cast noise preds to float32 before CFG and scheduler step
            # (pipeline lines 1127-1128)
            noise_pred_video = noise_pred_video.float()
            noise_pred_audio = noise_pred_audio.float()

            if do_cfg:
                video_uncond, video_cond = noise_pred_video.chunk(2)
                noise_pred_video = video_uncond + guidance_scale * (video_cond - video_uncond)

                audio_uncond, audio_cond = noise_pred_audio.chunk(2)
                noise_pred_audio = audio_uncond + guidance_scale * (audio_cond - audio_uncond)

            # Scheduler steps operate on float32 latents
            video_latents = self.scheduler.step(noise_pred_video, t, video_latents, return_dict=False)[0]
            audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]

        # ----------------------------------------------------------------
        # Unpack and denormalise (pipeline lines 1172-1187)
        # ----------------------------------------------------------------
        video_latents = _unpack_latents(
            video_latents, latent_f, latent_h, latent_w,
            self.transformer_spatial_patch_size, self.transformer_temporal_patch_size,
        )
        video_latents = _denormalize_latents(
            video_latents, self.vae.latents_mean, self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

        # Denormalise while still packed [B, L, 128], then unpack to [B, C, L, M]
        # (pipeline lines 1184-1187: _denormalize then _unpack)
        audio_latents = _denormalize_audio_latents(
            audio_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        audio_latents = _unpack_audio_latents(audio_latents, audio_num_frames, latent_mel_bins)

        return video_latents, audio_latents