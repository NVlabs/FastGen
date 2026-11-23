# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kandinsky 6's Qwen/CLIP conditioning and native video/audio codecs."""

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from diffusers import MMAudioVAE, MMAudioVocoder
from diffusers.models.autoencoders.autoencoder_mmaudio import (
    MMAudioAttnBlock1D,
    MMAudioMPConv1d,
    MMAudioResnetBlock1D,
    mp_silu,
    mp_sum,
)
from diffusers.pipelines.kandinsky6.pipeline_kandinsky6_ti2va import Kandinsky6TI2VAPipeline
from transformers import AutoProcessor, CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration

from fastgen.networks.module import FastGenModule
from fastgen.networks.network import FastGenNetwork
from fastgen.utils.distributed import is_rank0
from fastgen.utils.env import hf_load

from .vae import AutoencoderKLHunyuanVideo


def load_component(component, source, subfolder, revision, dtype):
    """Only rank zero loads weights; other ranks construct the same registered meta tree."""
    if FastGenNetwork._is_in_meta_context() or not is_rank0():
        config_class = getattr(component, "config_class", None)
        loader = config_class.from_pretrained if config_class is not None else component.load_config
        config = hf_load(loader, source, subfolder=subfolder, revision=revision)
        with torch.device("meta"):
            if config_class is not None:
                return component._from_config(config, dtype=dtype)
            return component.from_config(config)
    dtype_keyword = "dtype" if getattr(component, "config_class", None) is not None else "torch_dtype"
    return hf_load(component.from_pretrained, source, subfolder=subfolder, revision=revision, **{dtype_keyword: dtype})


def normalize_audio_tensor(value, dim=None, eps=1e-4):
    """Preserve the native fused denominator arithmetic before FP32 audio convolutions."""
    if dim is None:
        dim = list(range(1, value.ndim))
    norm = torch.linalg.vector_norm(value, dim=dim, keepdim=True, dtype=torch.float32)
    denominator = torch.add(eps, norm, alpha=math.sqrt(norm.numel() / value.numel()))
    return value / denominator.to(value.dtype)


class NativeAudioResnetBlock(MMAudioResnetBlock1D):
    def forward(self, value):
        value = normalize_audio_tensor(value, dim=1)
        hidden = self.conv1(mp_silu(value))
        hidden = self.conv2(mp_silu(hidden))
        if self.in_channels != self.out_channels:
            value = self.nin_shortcut(value)
        return mp_sum(value, hidden, t=0.3)


class NativeAudioAttnProcessor:
    """Keep native strided Q/K/V; contiguous staging changes GPU SDPA arithmetic."""

    def __call__(self, attention, value):
        batch, channels, length = value.shape
        projected = attention.qkv(value).reshape(batch, attention.num_heads, -1, 3, length)
        query, key, val = normalize_audio_tensor(projected, dim=2).unbind(3)
        query, key, val = (item.permute(0, 1, 3, 2) for item in (query, key, val))
        hidden = F.scaled_dot_product_attention(query, key, val)
        hidden = hidden.permute(0, 1, 3, 2).reshape(batch, channels, length)
        return attention.proj_out(hidden)


def install_native_audio_math(vae):
    """Attach native arithmetic to this codec without changing its state or other Diffusers models."""
    for module in vae.modules():
        if isinstance(module, MMAudioResnetBlock1D):
            module.__class__ = NativeAudioResnetBlock
        elif isinstance(module, MMAudioAttnBlock1D):
            module.set_processor(NativeAudioAttnProcessor())


def normalize_audio_checkpoint(vae):
    """Native MMAudio re-normalizes loaded MPConv weights, even exported checkpoints.

    This epsilon-sensitive transform runs once on loaded weights. Meta ranks
    receive the transformed state; materialization and reset must not repeat it.
    """
    with torch.no_grad():
        for module in vae.modules():
            if isinstance(module, MMAudioMPConv1d) and not module.weight.is_meta:
                weight = module.weight.float()
                normalized = normalize_audio_tensor(weight) / math.sqrt(weight[0].numel())
                module.weight.copy_(normalized.to(module.weight.dtype))


class FP32CLIPTextModel(FastGenModule, CLIPTextModel):
    """Native FP32 CLIP; its own precision also gives it an FSDP group apart from BF16 Qwen."""

    @property
    def precision_model(self):
        return torch.float32


class Kandinsky6TextEncoder(FastGenModule):
    """Native Qwen BF16 and CLIP FP32 conditioning, including meta-init buffer restoration."""

    @property
    def precision_model(self):
        return torch.bfloat16

    def __init__(self, model_id_or_local_path, revision=None, max_sequence_length=1024):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.text_encoder = load_component(
            Qwen2_5_VLForConditionalGeneration, model_id_or_local_path, "text_encoder", revision, torch.bfloat16
        )
        self.text_encoder_2 = load_component(
            CLIPTextModel, model_id_or_local_path, "text_encoder_2", revision, torch.float32
        )
        self.text_encoder_2.__class__ = FP32CLIPTextModel
        self.tokenizer = hf_load(
            AutoProcessor.from_pretrained, model_id_or_local_path, subfolder="tokenizer", revision=revision
        )
        self.tokenizer_2 = hf_load(
            CLIPTokenizer.from_pretrained, model_id_or_local_path, subfolder="tokenizer_2", revision=revision
        )

    def _fixed_precision_overrides(self):
        return {
            name: torch.float32
            for name, value in self.text_encoder.named_buffers(prefix="text_encoder")
            if value.is_floating_point()
        }

    def reset_parameters(self):
        """Rebuild Qwen rotary frequencies and CLIP positions without touching learned state."""
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        with torch.no_grad():
            for module in self.modules():
                if type(module).__name__ == "Qwen2_5_VisionRotaryEmbedding":
                    indices = torch.arange(0, module.dim, 2, dtype=torch.float32, device="cpu")
                    module.inv_freq.copy_(1.0 / (module.theta ** (indices / module.dim)))
                elif type(module).__name__ == "Qwen2_5_VLRotaryEmbedding":
                    initializer = (
                        module.compute_default_rope_parameters
                        if module.rope_type == "default"
                        else ROPE_INIT_FUNCTIONS[module.rope_type]
                    )
                    with torch.device("cpu"):
                        values, module.attention_scaling = initializer(module.config, device="cpu")
                    module.inv_freq.copy_(values)
                    module.original_inv_freq.copy_(values)
                elif type(module).__name__ == "CLIPTextEmbeddings":
                    positions = torch.arange(module.position_ids.shape[-1], device="cpu").unsqueeze(0)
                    module.position_ids.copy_(positions)

    def encode(self, prompts):
        return self(prompts)

    def forward(self, prompts):
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        with torch.autocast(self.device.type, enabled=False):
            tokens, pooled, mask = Kandinsky6TI2VAPipeline._get_prompt_embeds(
                prompts,
                self.tokenizer,
                self.text_encoder,
                self.tokenizer_2,
                self.text_encoder_2,
                self.max_sequence_length,
                self.device,
            )
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, pooled, mask


class Kandinsky6VideoVAE(FastGenModule):
    """Native FP16 video codec; latent scaling precedes the decoder precision cast."""

    @property
    def precision_model(self):
        return torch.float16

    def __init__(self, source, revision=None):
        super().__init__()
        # The native 6.0 release uses the same memory-aware Hunyuan VAE as 5.0.
        self.vae = load_component(AutoencoderKLHunyuanVideo, source, "vae", revision, torch.float16)

    def forward(self, value, operation, posterior="sample", generator=None):
        with torch.autocast(value.device.type, enabled=False):
            if operation == "encode":
                if value.ndim == 4:
                    value = value.unsqueeze(2)
                distribution = self.vae.encode(value.half(), opt_tiling=value.shape[2] > 1).latent_dist
                latent = distribution.sample(generator=generator) if posterior == "sample" else distribution.mode()
                return latent * self.vae.config.scaling_factor
            if operation == "decode":
                # Native normalization happens in the incoming latent dtype, before FP16 staging.
                return self.vae.decode((value / self.vae.config.scaling_factor).half()).sample
        raise ValueError(f"Unknown video VAE operation: {operation}")


class Kandinsky6AudioVAE(FastGenModule):
    """FP32 MMAudio codec with native weight normalization, attention and scalar state."""

    @property
    def precision_model(self):
        return torch.float32

    def __init__(self, source, revision=None):
        super().__init__()
        self.vae = load_component(MMAudioVAE, source, "audio_vae", revision, torch.float32)
        normalize_audio_checkpoint(self.vae)
        install_native_audio_math(self.vae)
        self.vocoder = load_component(MMAudioVocoder, source, "vocoder", revision, torch.float32)
        # FastGen freezes preprocessors. Persistent buffers keep native scalar
        # checkpoint shapes and arithmetic without FSDP trying to shard 0-D gains.
        for module in self.modules():
            for name, parameter in tuple(module.named_parameters(recurse=False)):
                if parameter.ndim == 0:
                    delattr(module, name)
                    module.register_buffer(name, parameter.detach())

    def forward(self, value, operation):
        with torch.autocast(value.device.type, enabled=False):
            if operation == "encode":
                if value.ndim == 3:
                    if value.shape[1] not in (1, 2):
                        raise ValueError("Kandinsky audio must be mono or stereo waveform data")
                    value = value.float().mean(dim=1)
                if value.ndim != 2:
                    raise ValueError("Kandinsky audio expects [B, samples] or [B, channels, samples]")
                # Match generated audio geometry: one latent per 1024 waveform samples, rounded up.
                hop = self.vae.latent_hop_length
                value = F.pad(value.float(), (0, (-value.shape[-1]) % hop))
                # Native audio encoding uses the posterior mean.
                return self.vae.encode(value).latent_dist.mode() * self.vae.config.scaling_factor
            if operation == "decode":
                latent = (value / self.vae.config.scaling_factor).float()
                mel = self.vae.decode(latent, return_dict=False)[0]
                return self.vocoder(mel, return_dict=False)[0].float()
        raise ValueError(f"Unknown audio VAE operation: {operation}")


class Kandinsky6VAE(FastGenModule):
    """Return native BCTHW video and mono B1S audio, preserving the selected modality keys."""

    latent_compression_factors = (4, 8, 8)
    vocoder_sample_rate = 44100

    def __init__(self, source, modalities=("video", "audio"), revision=None):
        super().__init__()
        self.modalities = tuple(modalities)
        self.video = Kandinsky6VideoVAE(source, revision)
        self.audio = Kandinsky6AudioVAE(source, revision) if "audio" in self.modalities else None

    def encode(self, data, posterior="sample", generator=None):
        """Sample the video posterior by default; native audio encoding uses its mean."""
        return self(data, "encode", posterior, generator)

    def decode(self, data):
        return self(data, "decode")

    def encode_video(self, data, posterior="sample", generator=None):
        return self(data, "encode_video", posterior, generator)

    def forward(self, data, operation, posterior="sample", generator=None):
        if posterior not in ("sample", "mode"):
            raise ValueError("posterior must be 'sample' or 'mode'")
        if operation == "encode_video":
            return self.video(data, "encode", posterior, generator)
        if not isinstance(data, Mapping):
            if self.modalities != ("video",):
                raise ValueError("Joint Kandinsky VAE input requires video/audio modality keys")
            return self.video(data, operation, posterior, generator)
        if not data or not set(data) <= set(self.modalities):
            raise ValueError(f"VAE modalities must be a nonempty subset of {self.modalities}")
        return {
            name: self.audio(value, operation)
            if name == "audio"
            else self.video(value, operation, posterior, generator)
            for name, value in data.items()
        }


def audio_shape(video_shape, fps=24.0, sample_rate=44100, hop_length=1024):
    frames = (video_shape[1] - 1) * 4 + 1
    return [40, math.ceil(frames / fps * sample_rate / hop_length)]
