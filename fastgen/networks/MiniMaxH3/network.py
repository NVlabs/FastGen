# SPDX-FileCopyrightText: Copyright (c) 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
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

"""MiniMax-H3 FastGen network: joint video + audio generation.

MiniMax-H3 denoises **one packed 1-D sequence** holding the text conditioning, optional keyframe
rows and the target audio and video rows, with full self-attention over all of it: no
cross-attention, no per-modality weights, no vocoder.  That sequence's geometry -- row order, the
shared 40-units-per-second rotary clock, the per-row modality tags -- is the checkpoint's contract
and lives in ``fastgen.networks.MiniMaxH3.packing`` (vendored from diffusers) rather than being
reimplemented here.

The supported ``transformer/`` partition is text-only plus optional first/last keyframe
conditioning. Audio is output-only and comes back from ``vae.decode`` as a stereo waveform.

Time convention
---------------
The transformer consumes ``t_h3 in [0, 1]`` with ``t_h3 = 1`` *clean* and predicts a data-ward
velocity (``x0 = x_t + (1 - t_h3) * v_h3``); FastGen's rectified flow has ``t = 0`` clean and a
noise-ward velocity.  The two coincide under ``t_h3 = 1 - t`` and ``v = -v_h3``, which is all
``forward`` does about it, so FastGen's ``rf`` schedule is exact as it stands.

Latent layout
-------------
``x_t`` is a modality dict: video ``[B, 24, F, H, W]``, audio ``[B, 32, 2, L]`` (channels, stereo
axis, 40 Hz latent axis), the channel axis staying at position 1 for both.
"""

import copy
import math
import os
import types
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.autoencoders import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.models.transformers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TransformerBlock,
)
from fastgen.networks.lora_utils import scaled_lora_layers
from fastgen.networks.MiniMaxH3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_KEYFRAME_TIMESTEP,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedSequence,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from torch.distributed.fsdp import fully_shard
from transformers import Qwen2TokenizerFast, Qwen3VLModel, Qwen3VLProcessor

import fastgen.utils.logging_utils as logger
from fastgen.networks.network import FastGenNetwork, FastGenModule, pixel_shape_from_factors, validate_conditioning
from fastgen.networks.noise_schedule import CompositeNoiseSchedule
from fastgen.utils.tensor import expand_like, high_precision_dtype, pixels_to_uint8
from fastgen.utils.distributed import is_rank0, world_size
from fastgen.utils.distributed.fsdp import (
    apply_fsdp_checkpointing,
    make_fully_shard_kwargs,
)
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.module import matches_fp32_declaration, module_state_names
from fastgen.utils.env import hf_download
from fastgen.utils.tree_utils import OutputDtype, TensorOrDict, assert_same, get_batch_size, get_device, to

# The converted subfolders of the supported t2va partition.
MINIMAX_H3_SUBFOLDERS = (
    "modular_model_index.json",
    "transformer/*",
    "text_encoder/*",
    "tokenizer/*",
    "processor/*",
    "vae/*",
    "audio_vae/*",
    "scheduler/*",
    "audio_scheduler/*",
)


def _pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack audio latents ``[B, C, A, L]`` -> channel-major rows ``[B, A * L, C]``."""
    return latents.permute(0, 2, 3, 1).flatten(1, 2)


def _unpack_audio_latents(rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    """Unpack channel-major rows ``[B, A * L, C]`` -> ``[B, C, A, L]``.

    ``C`` is read off the tensor rather than the config so a widened projection flows through
    unchanged.
    """
    rows = rows.unflatten(1, (MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents))
    return rows.permute(0, 3, 1, 2)


# Video defines the shared clock. Its shift is applied to the sampling grid; the audio schedule
# stores only the relative warp, so ``f_(3/12) . f_12 = f_3``.
MINIMAX_H3_FLOW_SHIFT = 12.0
MINIMAX_H3_AUDIO_SHIFT = 3.0
MINIMAX_H3_RELATIVE_AUDIO_SHIFT = MINIMAX_H3_AUDIO_SHIFT / MINIMAX_H3_FLOW_SHIFT


# ---------------------------------------------------------------------------
# classify_forward -- monkey-patched onto self.transformer
# ---------------------------------------------------------------------------


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
    position_ids: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    text_indices: torch.Tensor,
    attention_kwargs: Optional[dict] = None,
    return_dict: bool = True,  # accepted for API compatibility; always ignored
    # FastGen distillation kwargs
    return_features_early: bool = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: bool = False,
    skip_layers: Optional[List[int]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[List]]:
    """Drop-in ``MiniMaxH3Transformer3DModel.forward`` with FastGen's distillation hooks:
    intermediate features, early exit, logvar and skip-layer guidance.

    Everything else follows upstream verbatim, including the casts that align each input with
    its own projection's dtype -- the checkpoint keeps those projections float32 while the
    block stack is bfloat16.

    Returns a 4-tuple ``(video_output, audio_output, logvar, features)``, each ``None`` when
    not applicable (an early exit returns neither output).
    """
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    with scaled_lora_layers(self, lora_scale):
        sequence_length = position_ids.shape[0]
        rotary_emb = self.rope(position_ids)

        # 1. Project each modality and scatter its rows into the packed sequence buffer. The
        # text stream sets the dtype of the packed sequence.
        video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds)

        packed = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        packed = packed.index_copy(1, text_indices, text_embeds)
        packed = packed.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed = packed.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        # 2. One timestep embedding per distinct noise level, shared by every AdaLN projection.
        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))

        # The packed layout has no padding rows, so no tag is negative and no attention mask is needed.
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

        features: List[torch.Tensor] = []
        for idx, block in enumerate(self.transformer_blocks):
            # A skipped block acts as the identity; still record its feature so each requested index yields one.
            if skip_layers is None or idx not in skip_layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    packed = self._gradient_checkpointing_func(block, packed, temb, adaln_indices, rotary_emb)
                else:
                    packed = block(packed, temb, adaln_indices, rotary_emb)

            if feature_indices is not None and idx in feature_indices:
                features.append(packed.clone())
            if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
                return None, None, None, features

        if return_features_early:
            assert feature_indices is not None
            assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
            return None, None, None, features

        # Both output heads share a parameter dtype, so cast their shared activation once.
        normed = self.norm_out(packed, temb, timestep_indices)
        normed = normed.to(self.proj_out.weight.dtype)
        video_output = self.proj_out(normed).index_select(1, video_indices)
        audio_output = self.audio_proj_out(normed).index_select(1, audio_indices)

        logvar = None
        if return_logvar:
            assert hasattr(
                self, "logvar_linear"
            ), "logvar_linear is required when return_logvar=True. It is added by MiniMaxH3.__init__."
            # One row of ``temb`` per distinct noise level, so the logvar head reads the mean
            # over the levels present in the sequence.
            logvar = self.logvar_linear(temb.mean(dim=0, keepdim=True).to(self.logvar_linear.weight.dtype))
            logvar = logvar.expand(packed.shape[0], -1)

    return video_output, audio_output, logvar, features if feature_indices is not None else None


# ---------------------------------------------------------------------------
# Text encoder wrapper
# ---------------------------------------------------------------------------


class MiniMaxH3TextEncoder(FastGenModule):
    """Wraps the Qwen3-VL conditioner MiniMax-H3 reads at its 50th decoder layer.

    Returns both the hidden state and the per-row modality tags: the rows of a reference's
    vision block are tagged as *video* rather than text, which is what the transformer's
    AdaLN modulation keys off, so the tags are part of the conditioning rather than a
    detail of the tokenizer. In a distributed process group, rank 0 loads the weights and
    peers retain matching meta state until the model lifecycle prepares this wrapper eagerly
    or through FSDP.
    """

    def __init__(
        self,
        model_root: str,
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        """
        Args:
            model_root: Local directory holding the converted subfolders (see
                ``MiniMaxH3.model_root``).
            dtype: Data type for the conditioner's weights.  Defaults to bfloat16,
                matching the released checkpoint.
        """
        super().__init__()
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(model_root, "tokenizer"))
        self.processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_root, "processor"))
        text_encoder_path = os.path.join(model_root, "text_encoder")

        # The released ``text_encoder/`` partition is a ``Qwen3VLForConditionalGeneration``
        # checkpoint whose only key outside ``model.*`` is ``lm_head.weight``. MiniMax-H3 reads
        # ``hidden_states[50]`` and never evaluates that head, so the backbone class consumes
        # every weight this conditioner needs and leaves its 778M vocabulary projection unread.
        rank0_only = world_size() > 1
        if rank0_only and not is_rank0():
            logger.debug("Meta-init Qwen3-VL (weights will be synchronized during module preparation)...")
            config = Qwen3VLModel.config_class.from_pretrained(text_encoder_path)
            with torch.device("meta"):
                # Match rank 0's mixed parameter/buffer dtype map for byte-safe broadcast.
                self.text_encoder = Qwen3VLModel._from_config(config, dtype=dtype)
        else:
            logger.debug("Loading Qwen3-VL from pretrained...")
            self.text_encoder = Qwen3VLModel.from_pretrained(text_encoder_path, dtype=dtype)

        num_layers = self.text_encoder.config.text_config.num_hidden_layers
        if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
            raise ValueError(
                f"MiniMax-H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] of its Qwen3-VL "
                f"conditioner, but text_encoder has only {num_layers} decoder layers."
            )

    # -- presentation ----------------------------------------------------

    def _vision_ids(self, pad_token: str, num_tokens: int) -> Tuple[List[int], List[int]]:
        token_ids = (
            [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [self.tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
            + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        return token_ids, [MINIMAX_H3_VIDEO_TAG] * len(token_ids)

    def _text_ids(self, value: str) -> Tuple[List[int], List[int]]:
        token_ids = self.tokenizer(value, add_special_tokens=False)["input_ids"]
        return token_ids, [MINIMAX_H3_TEXT_TAG] * len(token_ids)

    def _keyframe_presentation(self, prompt: str, keyframes: list) -> Tuple[List[int], List[int], dict]:
        """MiniMax-H3's ``t2va`` / ``fl2va`` presentation: a label plus a vision block per
        keyframe, then the prompt verbatim, with no chat template and no special tokens."""
        token_ids: List[int] = []
        token_tags: List[int] = []
        vision: dict = {}
        if keyframes:
            processed = self.processor.image_processor(images=keyframes, return_tensors="pt")
            vision = {"pixel_values": processed["pixel_values"], "image_grid_thw": processed["image_grid_thw"]}
            merge_size = self.processor.image_processor.merge_size**2
            for index in range(len(keyframes)):
                num_image_tokens = int(vision["image_grid_thw"][index].prod()) // merge_size
                for segment in (
                    self._text_ids(f"<Picture {index + 1}>: "),
                    self._vision_ids("<|image_pad|>", num_image_tokens),
                ):
                    token_ids += segment[0]
                    token_tags += segment[1]
        prompt_ids, prompt_tags = self._text_ids(prompt)
        return token_ids + prompt_ids, token_tags + prompt_tags, vision

    # -- encode ----------------------------------------------------------

    def encode(
        self,
        prompt: Union[str, List[str]],
        keyframes: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode MiniMax-H3's presentation of a request.

        MiniMax-H3 packs one request into one sequence and the text length shifts the whole
        media rotary clock, so there is no padding to batch different prompts with: every
        prompt in *prompt* must tokenize to the same presentation length.  In practice the
        video configs run one sample per GPU, where this is free.

        Args:
            prompt: The prompt(s) to encode.
            keyframes: ``fl2va`` keyframes already put onto the target canvas, in packed
                order, as ``[3, H, W]`` ``uint8`` RGB tensors or PIL images.

        Returns:
            ``{"prompt_embeds": [B, T, 5120], "text_token_tags": [T]}``.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        per_prompt = [self._keyframe_presentation(text, keyframes or []) for text in prompt]

        lengths = {len(token_ids) for token_ids, _, _ in per_prompt}
        if len(lengths) != 1:
            raise ValueError(
                "MiniMax-H3 has no padding row: the text length sets the origin of the media rotary clock, so "
                f"every prompt in a batch must tokenize to the same length, got {sorted(lengths)}. Use one "
                "prompt per forward (batch_size=1), which is what the video configs do."
            )

        device = self.device
        model_inputs = []
        for token_ids, _, vision in per_prompt:
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            mm_token_type_ids = torch.tensor(
                self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
            )
            vision_inputs = {
                key: value.to(device=device, dtype=self.precision_model)
                if key.startswith("pixel_values")
                else value.to(device)
                for key, value in vision.items()
            }
            model_inputs.append((input_ids, mm_token_type_ids, vision_inputs))

        prompt_embeds = self(model_inputs)
        return {
            "prompt_embeds": prompt_embeds,
            "text_token_tags": torch.tensor(per_prompt[0][1], dtype=torch.long, device=device),
        }

    def forward(
        self,
        model_inputs: list,
    ) -> torch.Tensor:
        embeds = []
        for input_ids, mm_token_type_ids, vision in model_inputs:
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=True,
                **vision,
            )
            embeds.append(outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER])

        return torch.cat(embeds)


# ---------------------------------------------------------------------------
# VAE wrapper (video + audio)
# ---------------------------------------------------------------------------


def _align_waveform_length(audio: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Trim or zero-pad the trailing (sample) axis to ``num_samples``."""
    length = audio.shape[-1]
    if length > num_samples:
        return audio[..., :num_samples]
    if length < num_samples:
        return torch.nn.functional.pad(audio, (0, num_samples - length))
    return audio


class MiniMaxH3VAE(FastGenModule):
    """VAE wrapper for MiniMax-H3's video and audio latents.

    Two things about the video VAE are handled here rather than by the caller: its pixels are
    ImageNet-normalized RGB over a ``[0, 1]`` base range (FastGen passes ``[-1, 1]``, so both
    directions convert), and spatial tiling is **on by default**, because the released frames
    are the blended-tile ones.  Audio round-trips as a stereo waveform ``[B, 2, samples]`` at
    32 kHz; the VAE itself is mono and takes the two channels as two batch items.

    MiniMax's checkpoint repository ships the VAE implementations and loaders, but not a
    MiniMax-owned local sampling loop; it recommends several external runtimes and uses SGLang
    for its worked deployment example. That SGLang integration keeps both VAEs FP32-resident and
    keeps its sampler and external ``latents_mean/std`` transforms in FP32, while using FP16
    autocast for the video decoder forward. This wrapper preserves that published split explicitly
    around the raw Diffusers autoencoders.
    """

    def __init__(
        self,
        model_root: str,
        enable_tiling: bool = True,
        tile_sample_min_size: Optional[int] = None,
        fps: float = float(MINIMAX_H3_FPS),
    ):
        super().__init__()
        self._fps = float(fps)
        self._video_vae: AutoencoderKLMiniMaxH3 = AutoencoderKLMiniMaxH3.from_pretrained(
            os.path.join(model_root, "vae")
        )
        if enable_tiling:
            self._video_vae.enable_tiling(
                tile_sample_min_height=tile_sample_min_size,
                tile_sample_min_width=tile_sample_min_size,
            )
        else:
            self._video_vae.disable_tiling()

        self._audio_vae: AutoencoderKLMiniMaxH3Audio = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            os.path.join(model_root, "audio_vae")
        )

    @property
    def precision_model(self) -> torch.dtype:
        """MiniMax-H3's VAEs always consume FP32 inputs."""
        return torch.float32

    def fully_shard(self, **kwargs):
        """Shard both autoencoders as a single group behind this wrapper's ``forward``.

        ``forward`` is the only boundary this VAE is entered through: the two autoencoders are
        driven by ``encode``/``decode`` rather than ``__call__``, and Diffusers' audio attention
        processor reads ``pre_block.attn.qkv.weight`` through ``F.linear`` instead of calling
        ``qkv.forward``. A smaller group anywhere above those reads would never run its unshard
        hook and would expose a DTensor weight to an ordinary activation.
        """
        fully_shard(self, **kwargs)

    def forward(
        self,
        data,
        operation: str,
        mode: str = "sample",
        seed: Optional[int] = None,
    ):
        if operation == "encode":
            if isinstance(data, torch.Tensor):
                return self._encode_video(data, mode)

            if "video" not in data:
                return {"audio": self._encode_audio(data["audio"])}

            video = data["video"]
            # ``MiniMaxH3.infer_shape`` derives the audio row count from the video *latent* count,
            # which only inverts for ``17 * n + 5`` frames. This raises for anything else, so the
            # two paths cannot disagree on how many audio rows the packed sequence carries.
            video_latent_num_frames(video.shape[2])
            out = {"video": self._encode_video(video, mode)}

            # The contract covers exactly ``target`` latents (``round(frames / 24 * 40)``, the count the
            # reference generates and decodes to ``target * 800`` samples). Align the waveform to that
            # span so the right-padding encoder returns the count by construction.
            target = audio_latent_num_frames(video.shape[2], self._fps)
            num_samples = target * self._audio_vae.hop_length
            audio = data.get("audio")
            if audio is None:
                audio = video.new_zeros(video.shape[0], MINIMAX_H3_AUDIO_CHANNELS, num_samples)
            else:
                audio = _align_waveform_length(audio, num_samples)
            audio_latents = self._encode_audio(audio)
            if audio_latents.shape[-1] != target:
                raise ValueError(
                    "MiniMax-H3 audio VAE produced an unexpected latent duration: "
                    f"got {audio_latents.shape[-1]} frames for {num_samples} waveform samples, expected {target}."
                )
            out["audio"] = audio_latents
            return out

        if operation == "decode":
            if isinstance(data, torch.Tensor):
                return self._decode_video(data)

            if "video" not in data:
                return {"audio": self._decode_audio(data["audio"])}

            out = {"video": self._decode_video(data["video"])}
            if "audio" in data:
                out["audio"] = self._decode_audio(data["audio"])
            return out

        if operation == "encode_reference_video":
            if mode not in {"sample", "argmax"}:
                raise ValueError(f"Invalid mode: {mode}. Supported modes: ['sample', 'argmax']")
            pixels = self._pixels_to_vae_space(data)
            if pixels.shape[2] != 1:
                raise ValueError(f"A MiniMax-H3 keyframe must contain one frame, got {pixels.shape[2]}.")
            conditions = []
            for keyframe in pixels.split(1):
                distr = self._video_vae.encode(keyframe, return_dict=False)[0]
                latents = self._select_distr(distr, mode, seed)
                # Match Diffusers' conditioning recipe: quantize through FP16, then restore the
                # VAE precision before normalizing.
                latents = latents.to(torch.float16).to(dtype=self.precision_model)
                conditions.append(self._normalize_video_latents(latents))
            return torch.cat(conditions)

        raise ValueError(f"Unsupported preprocessor operation: {operation!r}")

    # -- geometry --------------------------------------------------------

    @property
    def latent_compression_factors(self) -> tuple[int, int, int]:
        """Pixel-to-latent downsampling factors of the video VAE: ``(T, H, W)``.

        Frame counts do not follow ``(F - 1) * T + 1``: the 17-frame chunking maps ``17 * n + 5``
        frames to ``5 * n + 2`` latent frames (``MiniMaxH3.pixel_shape``).
        """
        ratio = self._video_vae.spatial_compression_ratio
        return (self._video_vae.temporal_compression_ratio, ratio, ratio)

    @property
    def video_latent_channels(self) -> int:
        return self._video_vae.config.latent_channels

    @property
    def audio_latent_channels(self) -> int:
        return self._audio_vae.config.latent_channels

    @property
    def audio_sampling_rate(self) -> int:
        return self._audio_vae.config.sampling_rate

    @property
    def vocoder_sample_rate(self) -> int:
        """Sample rate of the decoded waveform.

        Named for the attribute the inference script probes; there is no separate vocoder, the
        audio VAE decodes straight to 32 kHz samples.
        """
        return self.audio_sampling_rate

    # -- encode ----------------------------------------------------------

    def encode(self, data, mode: str = "sample"):
        """Encode pixel / waveform input to latents.

        A bare video tensor returns its latents alone. A dict holding a video always returns
        **both** modalities, since both output heads run over one packed sequence: a clip with
        no audio stream (the loaders signal it by omitting the key) is encoded against a silence
        of its own duration, which is the clip's actual ground truth rather than a missing
        target.  The waveform is trimmed or zero-padded to exactly the span the video's duration
        implies on the 40 Hz grid (``round(frames / 24 * 40)`` latents of 800 samples), so the
        encoder returns that count by construction.
        """
        return self(data, mode=mode, operation="encode")

    def _pixels_to_vae_space(self, video: torch.Tensor) -> torch.Tensor:
        """FastGen's ``[-1, 1]`` RGB, or ``uint8`` RGB, -> the VAE's ImageNet-normalized ``[0, 1]`` space."""
        if video.dtype == torch.uint8:
            # The reference's ``uint8 / 255``, which ``[-1, 1]`` floats do not map back to exactly.
            video = video.to(dtype=self.precision_model) / 255.0
        else:
            video = video.to(dtype=self.precision_model).mul(0.5).add(0.5)
        mean = video.new_tensor(MINIMAX_H3_PIXEL_MEAN).view(1, -1, 1, 1, 1)
        std = video.new_tensor(MINIMAX_H3_PIXEL_STD).view(1, -1, 1, 1, 1)
        return (video - mean) / std

    def _normalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self._video_vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
        std = torch.tensor(self._video_vae.config.latents_std, device=latents.device, dtype=latents.dtype)
        return (latents - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1)

    def _encode_video(self, video: torch.Tensor, mode: str = "sample") -> torch.Tensor:
        """Encode ``[B, 3, T, H, W]`` pixels in ``[-1, 1]`` to ``[B, 24, F, H/16, W/16]``."""
        pixels = self._pixels_to_vae_space(video)
        distr = self._video_vae.encode(pixels, return_dict=False)[0]
        latents = self._select_distr(distr, mode)
        return self._normalize_video_latents(latents.to(dtype=self.precision_model))

    @staticmethod
    def _select_distr(distr, mode: str, seed: Optional[int] = None) -> torch.Tensor:
        if mode == "sample":
            generator = None if seed is None else torch.Generator().manual_seed(seed)
            return distr.sample(generator=generator)
        if mode == "argmax":
            return distr.mode()
        raise ValueError(f"Invalid mode: {mode}. Supported modes: ['sample', 'argmax']")

    def encode_reference_video(
        self,
        video: torch.Tensor,
        mode: str = "sample",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode H3 keyframes from sampled or argmax VAE distributions.

        ``video`` holds ``[K, 3, 1, H, W]`` keyframes as ``uint8`` RGB, the reference's input, or
        as FastGen's ``[-1, 1]`` RGB. Sampling uses the active RNG unless a seed is supplied. A supplied seed recreates a fresh
        generator for every keyframe, matching Diffusers' deterministic encoding. Equal-shaped
        keyframes therefore share the same posterior noise sample. Both modes retain the
        FP16 quantization followed by restoration to the VAE's model precision.
        """
        return self(video, mode=mode, seed=seed, operation="encode_reference_video")

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode a stereo waveform ``[B, 2, samples]`` to ``[B, 32, 2, L]``.

        MiniMax-H3 always consumes the distribution **mean**; the ``logs_proj`` head is never
        evaluated.
        """
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        batch_size = audio.shape[0]
        if audio.shape[1] == 1:
            audio = audio.expand(-1, MINIMAX_H3_AUDIO_CHANNELS, -1)
        mono = audio.reshape(batch_size * MINIMAX_H3_AUDIO_CHANNELS, 1, -1).to(dtype=self.precision_model)
        distr = self._audio_vae.encode(mono, return_dict=False)[0]
        latents = distr.mode().to(dtype=self.precision_model)  # [B * 2, C, L]
        latents = latents.unflatten(0, (batch_size, MINIMAX_H3_AUDIO_CHANNELS)).permute(0, 2, 1, 3)
        mean = torch.tensor(self._audio_vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
        std = torch.tensor(self._audio_vae.config.latents_std, device=latents.device, dtype=latents.dtype)
        return (latents - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

    # -- decode ----------------------------------------------------------

    def decode(self, data):
        """Decode latents to pixels / waveform, mirroring ``encode``'s key set."""
        return self(data, operation="decode")

    def _decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode ``[B, 24, F, h, w]`` to ``[B, 3, T, H, W]`` pixels in ``[-1, 1]``.

        On CUDA, the decode runs under float16 autocast even though the VAE weights are float32,
        matching the reference pipeline.
        """
        model_latents = latents.to(dtype=self.precision_model)
        mean = model_latents.new_tensor(self._video_vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = model_latents.new_tensor(self._video_vae.config.latents_std).view(1, -1, 1, 1, 1)
        model_latents = model_latents * std + mean

        with torch.autocast(
            device_type=model_latents.device.type,
            dtype=torch.float16,
            enabled=model_latents.device.type == "cuda",
        ):
            video = self._video_vae.decode(model_latents, return_dict=False)[0]

        pixel_mean = model_latents.new_tensor(MINIMAX_H3_PIXEL_MEAN).view(1, -1, 1, 1, 1)
        pixel_std = model_latents.new_tensor(MINIMAX_H3_PIXEL_STD).view(1, -1, 1, 1, 1)
        video = (video.to(dtype=self.precision_model) * pixel_std + pixel_mean).clamp(0, 1)
        # FastGen's media writers expect [-1, 1].
        return video.mul(2.0).sub(1.0)

    def _decode_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode ``[B, 32, 2, L]`` to a stereo waveform ``[B, 2, samples]``."""
        model_latents = latents.to(dtype=self.precision_model)
        mean = model_latents.new_tensor(self._audio_vae.config.latents_mean).view(1, -1, 1, 1)
        std = model_latents.new_tensor(self._audio_vae.config.latents_std).view(1, -1, 1, 1)
        model_latents = model_latents * std + mean

        batch_size = model_latents.shape[0]
        mono = model_latents.permute(0, 2, 1, 3).flatten(0, 1)  # [B * 2, C, L]
        waveform = self._audio_vae.decode(mono, return_dict=False)[0]  # [B * 2, 1, samples]
        return waveform.reshape(batch_size, MINIMAX_H3_AUDIO_CHANNELS, -1)


# ---------------------------------------------------------------------------
# Main MiniMax-H3 network
# ---------------------------------------------------------------------------


class MiniMaxH3(FastGenNetwork):
    """FastGen wrapper for MiniMax-H3 joint video + audio generation.

    Condition format (produced by ``MiniMaxH3TextEncoder.encode`` and extended by
    ``build_conditioning``)::

        condition = {
            "prompt_embeds":    Tensor [B, T, 5120],   # Qwen3-VL hidden_states[50]
            "text_token_tags":  LongTensor [T],        # 1 text, 0 for a vision block's rows
            # keyframe conditioning, present only when the request carries keyframes
            "cond_video_rows":  Tensor [B, R, 24 * 4], # float32, noise-augmented, in packed order
            "keyframe_anchors": Tuple[str, ...],       # "first" / "last" per keyframe, placing its rows
        }

    The ``transformer/`` checkpoint partition supports text-only generation and optional
    first/last keyframe conditioning. ``is_fl2va`` selects between those tasks explicitly;
    audio is an output-only modality.
    """

    MODEL_ID = "MiniMaxAI/MiniMax-H3"
    TRANSFORMER_SUBFOLDER = "transformer"
    # From the converted repository's VAE config; ``pixel_shape`` replaces the frame rule with the chunked one.
    latent_compression_factors = (4, 16, 16)

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID,
        modalities: Tuple[str, ...] = ("video", "audio"),
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        disable_grad_ckpt: bool = False,
        load_pretrained: bool = True,
        enable_logvar_linear: bool = False,
        fps: float = float(MINIMAX_H3_FPS),
        enable_vae_tiling: bool = True,
        vae_tile_sample_min_size: Optional[int] = None,
        is_fl2va: bool = False,
        keep_in_fp32: bool = True,
        freeze_adaln: bool = False,
        relative_audio_shift: float = MINIMAX_H3_RELATIVE_AUDIO_SHIFT,
        **schedule_kwargs,
    ):
        """
        Args:
            model_id_or_local_path: HuggingFace model ID or local path of the converted repository.
            modalities: Enabled modalities.  MiniMax-H3 always runs both output heads over
                the one packed sequence, so ``("video", "audio")`` is the only supported
                value; the argument exists so generic callers can read it.
            net_pred_type / schedule_type: FastGen prediction space.  MiniMax-H3 is
                rectified flow; ``forward`` maps its reversed time and data-ward velocity
                onto FastGen's convention.  A non-``rf`` ``schedule_type`` is rejected when
                ``load_pretrained``, the released weights being rectified flow; without it the
                choice is yours, but ``_to_flow_convention`` still assumes
                ``alpha' = -sigma'`` and has nothing to offer another family.
            disable_grad_ckpt: Disable activation checkpointing. FSDP needs no override:
                ``fully_shard`` replaces the Diffusers checkpointing with FSDP-compatible wrappers.
            load_pretrained: Load pretrained weights.  ``False`` (or a meta-device context)
                initialises from the config alone.
            enable_logvar_linear: Add a logvar head for uncertainty weighting (sCM).  Off by default.
            fps: Frame rate of the data.  The released rotary clock is trained at 24 fps, so
                anything else warns when ``load_pretrained`` -- it degrades the output rather than
                being unrepresentable -- and passes quietly otherwise.
            enable_vae_tiling: Spatial tiling of the video VAE.  **On by default**: the
                released frames are the blended-tile ones, so turning it off changes the
                output.
            vae_tile_sample_min_size: Override the tile edge in pixels (default 256).
            is_fl2va: Select first/last-keyframe-to-video+audio conditioning. ``False`` selects
                text-to-video+audio.
            keep_in_fp32: Honour the float32 carve-out -- the two patch projections, the timestep
                MLP and the two output heads -- on every ``to(dtype=...)``, the part diffusers
                does *not* do (``ModelMixin.to`` warns against casting after the load, then
                casts everything).  Upstream can afford that because its callers own the load;
                ours do not, so the carve-out is applied here rather than hoping nobody casts.
                ``False`` is stock behaviour -- the load still pins what diffusers pins, and the
                first cast takes those tensors with it -- which costs roughly a doubling of the
                error against a float32 reference forward.
            freeze_adaln: Hold the per-block AdaLN projections fixed, training only the rest of
                the transformer.  Set it on the ``teacher`` config to reach DMD2's fake score,
                which is instantiated from it (the teacher itself is frozen wholesale anyway).
            relative_audio_shift: Audio's sigma shift *relative to video's*, and the only warp the
                network holds.  Defaults to the release's ``3/12``.  The clock -- video's own shift
                of 12 -- belongs to whoever lays the grid down instead, so it is an argument of
                ``sample`` rather than of the network; composing the two puts audio on the
                released absolute shift of 3.
        """
        if tuple(modalities) != ("video", "audio"):
            raise ValueError(
                "MiniMax-H3 denoises video and audio in one packed sequence and always runs both output heads, "
                f"so `modalities` must be ('video', 'audio'), got {modalities}. The order matters too: video "
                "is the clock, so it has to come first."
            )
        # These constraints apply only to released weights.
        if load_pretrained:
            if float(fps) != float(MINIMAX_H3_FPS):
                logger.warning(
                    f"MiniMax-H3's rotary clock is trained at {MINIMAX_H3_FPS} fps; running the released "
                    f"weights at {fps} puts the media rotary embedding off the rate it learned."
                )
            if schedule_type != "rf":
                # ``_to_flow_convention`` reads ``sigma'`` off the sub-schedules and relies on
                # ``alpha' = -sigma'`` to turn the heads' ``eps - x0`` into ``dx/dt``.
                raise ValueError(
                    f"MiniMax-H3's released weights are rectified flow; `schedule_type` must be 'rf', "
                    f"got {schedule_type!r}."
                )

        self.modalities = tuple(modalities)
        # ``_set_noise_schedule`` runs in ``super().__init__`` and reads this value.
        self.relative_audio_shift = float(relative_audio_shift)

        # The native sampler grids start at sigma 1.0, so the raw-time bound defaults to 1.0 (configs may lower it).
        schedule_kwargs.setdefault("max_t", 1.0)
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)
        self.model_id_or_local_path = model_id_or_local_path
        # Components load from this local root rather than from a ``repo_id`` plus ``subfolder``, which is
        # load-bearing: ``Qwen3VLProcessor.from_pretrained`` loads its own inner tokenizer without carrying the
        # ``subfolder`` down, so it would look for a ``config.json`` at the repository root.
        self.model_root = hf_download(model_id_or_local_path, allow_patterns=list(MINIMAX_H3_SUBFOLDERS))
        self.fps = float(fps)
        self._enable_vae_tiling = enable_vae_tiling
        self._vae_tile_sample_min_size = vae_tile_sample_min_size
        self.is_fl2va = bool(is_fl2va)
        self.freeze_adaln = bool(freeze_adaln)
        self._keep_in_fp32 = keep_in_fp32
        self._layout_cache: dict = {}

        self._initialize_network(load_pretrained, enable_logvar_linear)

        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_network(
        self,
        load_pretrained: bool,
        enable_logvar_linear: bool,
    ) -> None:
        path = os.path.join(self.model_root, self.TRANSFORMER_SUBFOLDER)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        # The checkpoint contains only BF16 and FP32 floating tensors. Both branches start
        # uniformly FP32 for byte-safe FSDP weight sync: the loaded branch requests it explicitly,
        # while ``float()`` normalizes config-only construction away from the process-wide default
        # dtype. Runtime compute precision is applied later.
        if should_load_weights:
            logger.info(f"Loading MiniMax-H3 transformer from pretrained ({path})")
            self.transformer: MiniMaxH3Transformer3DModel = MiniMaxH3Transformer3DModel.from_pretrained(
                path, dtype=torch.float32
            )
        else:
            config = MiniMaxH3Transformer3DModel.load_config(path)
            if in_meta_context:
                logger.info("Initializing MiniMax-H3 transformer on meta device (weights arrive via FSDP sync)")
            else:
                logger.warning("MiniMax-H3 transformer initialized from config. No weights are loaded!")
            self.transformer = MiniMaxH3Transformer3DModel.from_config(config).float()

        if enable_logvar_linear:
            time_embed_dim = self.transformer.config.time_embed_dim
            # Keep loaded and config-only ranks on the same dtype map.
            self.transformer.logvar_linear = nn.Linear(time_embed_dim, 1, dtype=self.transformer.dtype)
            logger.debug(f"Added logvar_linear ({time_embed_dim} -> 1, {self.transformer.dtype})")

        self.patch_size = tuple(self.transformer.config.patch_size)
        self.video_latent_channels = self.transformer.config.in_channels
        self.audio_latent_channels = self.transformer.config.audio_in_channels

    def init_preprocessors(self):
        """Initialize the conditioner and the two VAEs (lazily, on first use)."""
        if "text_encoder" not in self.preprocessors:
            self.init_text_encoder()
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_text_encoder(self, dtype: Optional[torch.dtype] = torch.bfloat16):
        self.preprocessors["text_encoder"] = MiniMaxH3TextEncoder(
            model_root=self.model_root,
            dtype=dtype,
        )

    def init_vae(self):
        self.preprocessors["vae"] = MiniMaxH3VAE(
            self.model_root,
            enable_tiling=self._enable_vae_tiling,
            tile_sample_min_size=self._vae_tile_sample_min_size,
            fps=self.fps,
        )

    # ------------------------------------------------------------------
    # Device movement / FSDP
    # ------------------------------------------------------------------

    def _fixed_precision_overrides(self) -> Dict[str, torch.dtype]:
        """Return the transformer's FP32 state overrides for ``_set_precision``.

        FastGen must retain the checkpoint's declared float32 modules while selecting storage
        precision before DDP/FSDP. Ordinary ``nn.Module.to(dtype)`` calls after that boundary are
        intentionally uniform.
        """
        if not self._keep_in_fp32:
            return {}
        return {
            name: torch.float32
            for name in module_state_names(self)
            if matches_fp32_declaration(name, self.transformer, prefix="transformer.")
        }

    def _freeze_adaln_(self) -> None:
        """Set ``requires_grad=False`` on every per-block AdaLN projection.

        The modulation lives in ``transformer_blocks.<i>.adaln_proj`` (100 of the transformer's
        638 parameters).  Freezing it also drops their grads and AdamW state, since the optimizer
        is built from the ``requires_grad`` parameters at ``trainer.init_optimizers``.
        """
        frozen = 0
        for name, param in self.transformer.named_parameters():
            if "adaln" in name:
                param.requires_grad_(False)
                frozen += 1
        logger.info(f"freeze_adaln: held {frozen} AdaLN parameters fixed")

    def requires_grad_(self, requires_grad: bool = True):
        """Intercept module-level ``requires_grad_(True)`` so the AdaLN stays frozen.

        ``build_model`` enables the whole fake score once, and DMD2's ``_setup_grad_requirements``
        re-enables it on every fake-score iteration; without this the freeze would survive only
        until the first of those calls.  ``requires_grad_(False)`` still freezes everything.
        """
        super().requires_grad_(requires_grad)
        if requires_grad and getattr(self, "freeze_adaln", False):
            self._freeze_adaln_()
        return self

    def fully_shard(self, **kwargs):
        """Shard the MiniMax-H3 transformer for FSDP2.

        Shards ``self.transformer`` (see ``FastGenNetwork.fully_shard``) as one group per block,
        one for the root, and one for each float32 module.  Those extra groups are the point.
        Blocks and root run at the trainer's ``mp_policy`` -- float32 masters, bfloat16 compute --
        while the carve-out gets the same policy at a float32 ``param_dtype``, so it *computes* in
        float32 as the release does.  Measured on the real checkpoint at the SFT geometry that costs
        neither step time nor peak memory, and it halves the error against a float32 reference
        forward.

        It depends on ``forward`` returning each prediction in the dtype of the latents it was
        handed: float32 heads feeding a loss that holds bfloat16 data kill the backward with
        ``RuntimeError: Found dtype BFloat16 but expected Float``.
        """
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(self.transformer, MiniMaxH3TransformerBlock)

        fp32_kwargs = make_fully_shard_kwargs(
            kwargs,
            param_dtype=torch.float32,
            storage_dtype=self.precision_opt,
            enabled=self._keep_in_fp32,
        )

        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)
        if fp32_kwargs is not None:
            for module in (
                self.transformer.proj_in,
                self.transformer.audio_proj_in,
                self.transformer.time_embedder,
                self.transformer.proj_out,
                self.transformer.audio_proj_out,
            ):
                fully_shard(module, **fp32_kwargs)
            logger.info("Sharded MiniMax-H3 input/output projections and time_embedder with a float32 param_dtype")

        fully_shard(self.transformer, **kwargs)
        # Defensive re-application: FSDP2 preserves ``requires_grad`` on the sharded parameters, and the
        # ``requires_grad_`` override above is what keeps the AdaLN frozen through ``build_model`` and the
        # DMD2 fake-score toggles before the optimizer is built at ``trainer.init_optimizers``.
        if getattr(self, "freeze_adaln", False):
            self._freeze_adaln_()

    def _set_noise_schedule(self, schedule_type=None, **schedule_kwargs):
        """Build the composite schedule with video as the clock.

        Video's sub-schedule is the unshifted ``schedule_type`` and audio's is ``"shifted_rf"`` with
        ``relative_audio_shift``, because the shared time *is* video's noise level: the grid is what warps by
        the clock, and the warps compose, so ``f_(3/12) . f_12 = f_3`` lands audio exactly where the release
        does.  Nothing here needs the clock's value, which is why the network does not hold one.
        """
        assert not self.is_prepared, "Initialize the noise schedule before preparation"
        # "rf" rejects a schedule ``shift``: audio's is relative_audio_shift, the clock's is sample(flow_shift=...).
        if schedule_type is not None:
            self.schedule_type = schedule_type

        self.noise_scheduler = CompositeNoiseSchedule(
            schedules={
                "video": {"name": self.schedule_type, **schedule_kwargs},
                "audio": {"name": "shifted_rf", **schedule_kwargs, "shift": self.relative_audio_shift},
            },
            primary_key="video",
        )

    def _unpatchify_features(
        self,
        features: List[torch.Tensor],
        layout: MiniMaxH3PackedSequence,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Split packed feature rows ``[B, N, inner_dim]`` into per-modality grids.

        Restores each modality the way ``forward`` restores the output heads: video to
        ``[B, inner_dim // prod(patch_size), F, H, W]``, audio to ``[B, inner_dim, 2, L]``.
        Conditioning video rows are dropped -- they are inputs, not generated content.
        """
        unpatchified: List[Dict[str, torch.Tensor]] = []
        for feat in features:
            video_rows = feat.index_select(1, layout.video_indices)[:, layout.num_condition_video_rows :]
            # Read the channel count off the tensor, as the output path does, so a widened inner
            # dimension flows through unchanged.
            channels = video_rows.shape[-1] // math.prod(self.patch_size)
            video = unpatchify_video_tokens(
                video_rows.flatten(0, 1),
                num_latent_frames,
                latent_height,
                latent_width,
                channels,
                self.patch_size,
            )
            audio = _unpack_audio_latents(feat.index_select(1, layout.audio_indices), num_audio_latents)
            unpatchified.append({"video": video, "audio": audio})
        return unpatchified

    def flow_convention_scale(self, t: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """Return each head's scale from ``eps - x0`` to FastGen flow at ``t``.

        That scale is ``sigma'(t)``, or ``None`` for a modality whose sub-schedule is unwarped.
        """
        return {
            modality: None if schedule.is_unwarped else schedule.sigma_prime(t)
            for modality, schedule in self.noise_scheduler.subschedules.items()
        }

    def _to_flow_convention(self, model_output: Dict[str, torch.Tensor], t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Put each head's ``eps - x0`` into FastGen's ``flow``, at the shared time ``t``.

        FastGen's ``flow`` is ``dx/dt``, and for rectified flow that is ``sigma'(t)(eps - x0)``.
        Video is the clock, so ``sigma' = 1`` there and its head already emits ``dx/dt``; only
        audio, on the relative warp, is actually scaled. A scaled head stays in high precision,
        so ``convert_model_output`` rounds it once to the state's dtype.

        Args:
            model_output: ``{modality: eps - x0}``, one entry per head.
            t: Shared timestep ``[B]`` from the primary schedule.
        """
        scales = self.flow_convention_scale(t)
        converted: Dict[str, torch.Tensor] = {}
        for modality, out in model_output.items():
            scale = scales[modality]
            if scale is None:
                converted[modality] = out
            else:
                dtype = high_precision_dtype(out)
                converted[modality] = out.to(dtype) * expand_like(scale.to(dtype), out)
        return converted

    def reset_parameters(self):
        """Rebuild the non-persistent rotary buffers after meta-device materialisation.

        Being non-persistent, the rotary ``inv_freq`` is absent from the state dict, and FSDP
        meta-init restores only persistent state. Left unrecomputed it is uninitialized memory on
        every rank, which neither raises nor necessarily produces a NaN loss -- it silently
        destroys the positional structure.
        """
        self._reset_rope_buffers()
        super().reset_parameters()
        logger.debug("Reinitialized MiniMax-H3 rotary buffers")

    def _reset_rope_buffers(self) -> None:
        """Recompute the rotary ``inv_freq`` exactly as ``MiniMaxH3RotaryPosEmbed`` does."""
        rope_theta = float(self.transformer.config.rope_theta)
        for module in self.transformer.modules():
            if not isinstance(module, MiniMaxH3RotaryPosEmbed):
                continue
            freq_dim = module.rope_freq_dim
            inv_freq = 1.0 / (
                rope_theta
                ** (
                    torch.arange(0, 2 * freq_dim, 2, dtype=torch.float32, device=module.inv_freq.device)
                    / (2 * freq_dim)
                )
            )
            module.inv_freq.copy_(inv_freq)
            logger.debug(f"Recomputed rotary inv_freq ({freq_dim} frequencies, theta {rope_theta})")

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def pixel_shape(self, latent_shape: Sequence[int]) -> Tuple[int, ...]:
        return self.pixel_shape_for(latent_shape)

    @classmethod
    def pixel_shape_for(cls, latent_shape: Sequence[int]) -> Tuple[int, ...]:
        """``FastGenNetwork.pixel_shape_for`` with the VAE's chunked frame count.

        ``17 * n + 5`` frames encode to ``5 * n + 2`` latent frames.
        """
        _, height, width = pixel_shape_from_factors(latent_shape, cls.latent_compression_factors)
        num_latent_frames = latent_shape[1]
        if num_latent_frames < 2 or (num_latent_frames - 2) % MINIMAX_H3_LATENTS_PER_CHUNK:
            raise ValueError(
                f"MiniMax-H3's video VAE produces `5 * n + 2` latent frames, got {num_latent_frames}. "
                "Valid counts are 2, 7, 12, 17, ... (i.e. 5, 22, 39, 56, ... frames)."
            )
        chunks = (num_latent_frames - 2) // MINIMAX_H3_LATENTS_PER_CHUNK
        return (chunks * MINIMAX_H3_FRAMES_PER_CHUNK + MINIMAX_H3_LATENTS_PER_CHUNK, height, width)

    def infer_shape(self, input_shape: dict) -> dict:
        """Populate a ``None`` ``"audio"`` entry from the video shape.

        The audio latent count is the one that covers the video's duration on the 40 Hz
        latent grid, which is exactly what makes the two modalities share one rotary clock.
        """
        input_shape = dict(input_shape)
        if input_shape.get("audio") is None:
            video_shape = input_shape["video"]
            assert video_shape is not None, "Cannot infer the audio shape without the video shape"
            num_frames = self.pixel_shape(video_shape)[0]
            num_audio_latents = audio_latent_num_frames(num_frames, self.fps)
            input_shape["audio"] = [self.audio_latent_channels, MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents]
        return input_shape

    # ------------------------------------------------------------------
    # Packed layout
    # ------------------------------------------------------------------

    def _layout(
        self,
        text_token_tags: torch.Tensor,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        keyframe_anchors: Tuple[str, ...],
        device: torch.device,
    ):
        """Return the packed layout cached by shapes, tags, anchors, and device.

        The layout depends only on structural metadata, never on the latent
        values, so one build serves every step of a sampling loop and every iteration that
        keeps the same geometry.
        """
        key = (
            int(text_token_tags.numel()),
            tuple(text_token_tags.tolist()),
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            keyframe_anchors,
            str(device),
        )
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached

        # Layout metadata must be ordinary tensors, but its construction is never differentiable.
        with torch.inference_mode(False), torch.no_grad():
            layout = build_packed_sequence(
                text_token_tags.cpu(),
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                self.patch_size,
                keyframe_anchors,
            )

            layout = copy.copy(layout)
            layout.position_ids = layout.position_ids.to(device)
            layout.token_tags = layout.token_tags.to(device)
            layout.video_indices = layout.video_indices.to(device)
            layout.audio_indices = layout.audio_indices.to(device)
            layout.text_indices = layout.text_indices.to(device)
        # One layout per geometry is plenty; a stale cache would only pin fp64 grids.
        if len(self._layout_cache) > 8:
            self._layout_cache.clear()
        self._layout_cache[key] = layout
        return layout

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    @staticmethod
    def _endpoint_keyframes(
        video: torch.Tensor, keyframe_anchors: Optional[Tuple[str, ...]] = None
    ) -> tuple[torch.Tensor, Tuple[str, ...]]:
        """Select reference-compatible endpoint keyframes and return them in packed order."""
        if keyframe_anchors is None:
            keyframe_anchors = ("first", "last") if video.shape[2] > 1 else ("first",)
        else:
            keyframe_anchors = tuple(keyframe_anchors)
        if keyframe_anchors not in {("first",), ("last",), ("first", "last")}:
            raise ValueError(
                "MiniMax-H3 keyframe_anchors must be ('first',), ('last',), or "
                f"('first', 'last'), got {keyframe_anchors!r}."
            )
        endpoints = {"first": video[:, :, :1], "last": video[:, :, -1:]}
        return torch.cat([endpoints[anchor] for anchor in keyframe_anchors], dim=0), keyframe_anchors

    def build_conditioning(self, data, input_shape=None, precision=None):
        """Resolve T2VA text or build FL2VA's keyframe-aware condition.

        T2VA delegates ordinary raw/encoded resolution to the base builder. FL2VA accepts prompts
        only under ``*_raw`` because the current reference frames are part of the Qwen-VL
        presentation; plain condition fields cannot prove that provenance and must be absent.
        The keyframes are the endpoints of ``real_raw`` that ``data["keyframe_anchors"]`` selects,
        by default the first and last frame (the first alone for a single frame). Positive and
        negative branches share the noisy VAE keyframe rows, which stay float32 whatever the state
        ``precision``. Complete FL2VA conditions are not accepted as re-entrant inputs.
        """
        absent_mapping_keys = ("cond_video_rows", "keyframe_anchors")

        if not self.is_fl2va:
            out = super().build_conditioning(data, input_shape=input_shape, precision=precision)
            validate_conditioning(
                out,
                required_fields=("condition",),
                require_mapping=True,
                absent_mapping_keys=absent_mapping_keys,
            )
            return out

        validate_conditioning(
            data,
            required_fields=("condition_raw", "real_raw"),
            absent_fields=("condition", "neg_condition"),
        )

        real_raw = data["real_raw"]

        video = real_raw["video"] if isinstance(real_raw, Mapping) else real_raw
        if video.shape[0] != 1:
            raise ValueError(
                "MiniMax-H3 packs one request into one sequence, so conditioning is only supported at "
                f"batch size 1, got {video.shape[0]}."
            )

        frames, keyframe_anchors = self._endpoint_keyframes(video, data.get("keyframe_anchors"))
        # The VAE encodes the very uint8 keyframes the text encoder is shown, as the reference does.
        uint8_frames = pixels_to_uint8(frames)
        # Qwen's image processor runs on the frames' device. Its CUDA resize differs from the reference's CPU one, so
        # CUDA frames match the reference's PIL keyframes exactly only on canvases it leaves unresized, whose sides
        # are multiples of 32.
        conditions = self._resolve_conditions(
            data, dtype=self.precision_model, keyframes=list(uint8_frames.transpose(1, 2).flatten(0, 1))
        )
        validate_conditioning(
            conditions,
            required_fields=("condition",),
            require_mapping=True,
            absent_mapping_keys=absent_mapping_keys,
        )

        # Use the active RNG for stochastic conditioning latents.
        vae = self.get_preprocessor("vae", "encode the reference keyframes")
        cond_latents = vae.encode_reference_video(uint8_frames, mode="sample")
        num_latent_frames, latent_height, latent_width = cond_latents.shape[2:]
        rows = patchify_video_latents(cond_latents, self.patch_size)

        # MiniMax-H3's reversed `t_h3 = 0.999` mixes in only `0.001 * noise`. These rows
        # participate in every transformer pass, but their predictions are discarded and the
        # same rows are reinserted at every denoising step.
        noise = keyframe_condition_noise(
            ((num_latent_frames, latent_height, latent_width),) * len(keyframe_anchors),
            self.patch_size,
            self.video_latent_channels,
            device=rows.device,
            dtype=rows.dtype,
        )
        # Mixed and kept in float32: the reference feeds float32 keyframe rows to its float32 input
        # projection, so they are never rounded to the state precision.
        assert_same(dtype=torch.float32, rows=rows)
        keyframe_timestep = rows.new_tensor(MINIMAX_H3_KEYFRAME_TIMESTEP)
        rows = keyframe_timestep * rows + (1.0 - keyframe_timestep) * noise

        conditioning_extras = {
            "cond_video_rows": rows[None],
            # Layout metadata remains a tuple through tensor-tree transforms.
            "keyframe_anchors": keyframe_anchors,
        }

        return {key: {**value, **conditioning_extras} for key, value in conditions.items()}

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    @staticmethod
    def _scalar_timestep(t: torch.Tensor, name: str = "sigma(t)") -> float:
        """The one timestep of a request, in MiniMax-H3's ``t_h3 = 1 - t`` convention.

        The batch axis is pure replication -- one packed layout and one row-to-timestep
        assignment for every item -- so a per-sample timestep would need one layout per sample
        and is rejected rather than silently collapsed.
        """
        flat = t.flatten()
        if flat.numel() > 1 and not bool((flat == flat[0]).all()):
            raise ValueError(
                f"MiniMax-H3 assigns one timestep per row of a single packed sequence, so `{name}` must be "
                f"uniform across the batch; got {flat[:8].tolist()}. Use batch_size=1, which is what the "
                "video configs do."
            )
        return 1.0 - float(flat[0])

    def forward(
        self,
        x_t: TensorOrDict,
        t: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
        fps: Optional[float] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        unpatchify_features: bool = True,
        skip_layers: Optional[List[int]] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[TensorOrDict, List[torch.Tensor], Tuple]:
        """Forward pass over the packed sequence.

        Args:
            x_t: ``{"video": [B, 24, F, H, W], "audio": [B, 32, 2, L]}``.  Both modalities
                are required: MiniMax-H3 runs both output heads over one sequence.
            t: Shared timestep ``[B]`` in FastGen's convention (``0`` clean).
            condition: See the class docstring.
            unpatchify_features: Return features as ``{"video": [B, C, F, H, W],
                "audio": [B, C, 2, L]}`` per layer, which ``CompositeDiscriminator`` consumes.
                ``False`` returns the raw packed rows ``[B, N, inner_dim]``.
            skip_layers: Transformer layer indices to skip entirely (SLG).
            apply_output_conditioning: Apply the output constraint after the conversion.
            output_dtype: A dtype, or a mapping of one per modality, casts the prediction to it; ``"same"`` rounds
                it to each modality's state dtype, and ``None`` keeps it in high precision.
        Returns:
            ``{"video": ..., "audio": ...}`` in *fwd_pred_type* space, plus the features /
            logvar extras when requested.
        """
        self._validate_forward_kwargs(
            fwd_kwargs,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            skip_layers=skip_layers,
            supports_skip_layers=True,
            num_features=len(self.transformer.transformer_blocks),
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        if not isinstance(x_t, Mapping) or len(x_t) != 2 or "video" not in x_t or "audio" not in x_t:
            raise ValueError(
                "MiniMax-H3 denoises video and audio in one packed sequence, so x_t must be a dict with both "
                f"'video' and 'audio' keys; got {type(x_t).__name__}"
                + (f" with keys {sorted(x_t.keys())}" if isinstance(x_t, Mapping) else "")
                + "."
            )
        if fps is not None and float(fps) != self.fps:
            raise ValueError(f"MiniMax-H3's rotary clock is fixed at {self.fps} fps, got {fps}.")
        required_condition_keys = ("prompt_embeds", "text_token_tags")
        if not isinstance(condition, Mapping) or any(key not in condition for key in required_condition_keys):
            raise ValueError(
                "MiniMax-H3's condition is a dict with 'prompt_embeds' and 'text_token_tags' "
                f"(see MiniMaxH3TextEncoder.encode); got {type(condition).__name__}."
            )

        video_latents, audio_latents = x_t["video"], x_t["audio"]
        batch_size = video_latents.shape[0]
        device = video_latents.device
        _, _, num_latent_frames, latent_height, latent_width = video_latents.shape
        num_audio_latents = audio_latents.shape[-1]

        prompt_embeds = condition["prompt_embeds"]
        text_token_tags = condition["text_token_tags"]
        # ``build_packed_sequence`` casts the tags to long, so a non-integer dtype would
        # truncate into a valid-looking tag (0.9 text into 0, a video row) instead of failing.
        if text_token_tags.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"condition['text_token_tags'] must have an int32 or int64 dtype, got {text_token_tags.dtype}"
            )
        # A row-count mismatch fails where the rows are scattered into the packed sequence, naming
        # both lengths. A per-batch tags tensor instead reaches the layout cache, whose key holds
        # ``tags.tolist()``, and only surfaces as an unhashable dict key.
        if text_token_tags.ndim != 1:
            raise ValueError(
                "condition['text_token_tags'] is one shared sequence of per-row tags, not a batched "
                f"tensor; got shape {tuple(text_token_tags.shape)}."
            )
        cond_video_rows = condition.get("cond_video_rows")
        keyframe_anchors = tuple(condition.get("keyframe_anchors", ()))
        # The anchors place the keyframe rows in the sequence, so a condition carries both or neither.
        if (cond_video_rows is None) != (not keyframe_anchors):
            raise ValueError(
                "MiniMax-H3 keyframe conditioning needs both condition['cond_video_rows'] and the "
                f"condition['keyframe_anchors'] that place them; got rows={cond_video_rows is not None}, "
                f"anchors={keyframe_anchors}."
            )

        layout = self._layout(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            keyframe_anchors,
            device,
        )

        # ---- Pack the rows: conditioning first, then the generated ones ----
        video_rows = patchify_video_latents(video_latents, self.patch_size).unflatten(0, (batch_size, -1))
        if layout.num_condition_video_rows:
            # The keyframe rows reach the float32 input projection unrounded, as in the reference; the
            # generated rows join them losslessly, whatever the state precision.
            assert_same(dtype=torch.float32, device=video_rows.device, cond_video_rows=cond_video_rows)
            video_rows = torch.cat([cond_video_rows, video_rows.float()], dim=1)
        audio_rows = _pack_audio_latents(audio_latents)

        # ---- Row-to-timestep plan ----
        # The shared FastGen time is video's noise level: video's sub-schedule is the identity,
        # while audio applies the relative 3/12 warp. During sampling, the shared grid is first
        # shifted by 12, giving effective shifts of 12 for video and 3 for audio. MiniMax-H3 uses
        # reversed time, so each noise level becomes the transformer timestep `t_h3 = 1 - sigma`.
        sigmas = self.noise_scheduler.sigma(t)
        video_timestep = self._scalar_timestep(sigmas["video"], "sigma(t)['video']")
        audio_timestep = self._scalar_timestep(sigmas["audio"], "sigma(t)['audio']")
        unique_timesteps, timestep_indices = build_row_timesteps(
            layout,
            video_timestep,
            audio_timestep,
            # Use the keyframe timestep unless the generated video's H3 timestep is higher.
            max(video_timestep, MINIMAX_H3_KEYFRAME_TIMESTEP),
        )

        outputs = self.transformer(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=prompt_embeds,
            timestep=unique_timesteps.to(device),
            timestep_indices=timestep_indices.to(device),
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            skip_layers=skip_layers,
        )
        video_packed, audio_packed, logvar, features = outputs

        if return_features_early:
            return (
                self._unpatchify_features(
                    features, layout, num_latent_frames, latent_height, latent_width, num_audio_latents
                )
                if unpatchify_features
                else features
            )

        # ---- Drop the conditioning rows and unpack ----
        video_packed = video_packed[:, layout.num_condition_video_rows :]

        # The channel count is read off the tensor so a widened projection flows through unchanged.
        channels = video_packed.shape[-1] // math.prod(self.patch_size)
        video_out = unpatchify_video_tokens(
            video_packed.flatten(0, 1),
            num_latent_frames,
            latent_height,
            latent_width,
            channels,
            self.patch_size,
        )
        audio_out = _unpack_audio_latents(audio_packed, num_audio_latents)

        # Reverse the data-ward velocity.
        model_output = self._to_flow_convention({"video": -video_out, "audio": -audio_out}, t)
        model_output = self.transform_output(
            model_output, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
        )

        if features is not None and unpatchify_features:
            features = self._unpatchify_features(
                features, layout, num_latent_frames, latent_height, latent_width, num_audio_latents
            )
        out = model_output if features is None else [model_output, features]
        if return_logvar:
            return out, logvar.to(dtype=video_latents.dtype)
        return out

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------

    @staticmethod
    def default_guidance_cfg(*args, **kwargs) -> GuidanceConfig:
        """MiniMax-H3's released guidance recipe: one conditional branch.

        The released checkpoints are guidance-distilled, so an empty ``GuidanceConfig`` makes
        ``predict_with_guidance`` perform a single conditional forward per step.
        """
        del args, kwargs
        return GuidanceConfig()

    def sample(
        self,
        noise: TensorOrDict,
        condition: Optional[Dict[str, Any]] = None,
        neg_condition: Optional[Any] = None,
        sample_steps: int = 49,
        sample_type: str = "ode",
        flow_shift: float = MINIMAX_H3_FLOW_SHIFT,
        fps: Optional[float] = None,
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> TensorOrDict:
        """Run MiniMax-H3's denoising loop for joint video + audio generation.

        Sampling first puts the shared grid on video's shift-12 clock. Video's identity
        sub-schedule consumes that grid directly, while audio's relative 3/12 sub-schedule gives
        it an effective shift of 3.

        Args:
            noise: ``{"video": [B, 24, F, H, W], "audio": [B, 32, 2, L]}``.
            condition: See the class docstring.
            neg_condition: Passed through to ``predict_with_guidance`` and used when the supplied
                guidance policy requests a negative branch.
            sample_steps: Number of denoising steps, i.e. model evaluations (forwards). The
                grid spans ``sample_steps + 1`` sigma points, the terminal ``0`` included.
                Defaults to 49: the release's ``num_inference_steps=50`` counts those points.
            sample_type: Integration policy; only ``"ode"`` is supported.
            flow_shift: The clock -- video's sigma shift, and so where this call's grid points
                land.  Defaults to the release's 12.  Audio follows through the network's
                ``relative_audio_shift``, so changing this moves both modalities together and
                keeps the released ratio between them.
            fps: Forwarded to ``forward``, which accepts only the network's fixed clock.
            guidance_cfg: Defaults to ``default_guidance_cfg`` (one conditional forward); a
                supplied config is honoured verbatim.

        Returns:
            ``{"video": [B, 24, F, H, W], "audio": [B, 32, 2, L]}`` clean latents in FP32.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )
        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()

        batch_size = get_batch_size(noise)
        device = get_device(noise)
        ns = self.noise_scheduler

        # Video's shifted grid is shared; sub-schedules map it to modality noise levels.
        grid = ns.primary.shift_t(ns.grid(sample_steps, device=device), shift=flow_shift)
        # Run sampling loop in FP32 (transformer casts latents to each input projection's dtype).
        latents = ns.latents(to(noise, dtype=torch.float32), t_init=grid[0])

        for step in range(sample_steps):
            t = grid[step].expand(batch_size)
            flow = predict_with_guidance(
                self,
                latents,
                t,
                guidance_cfg=guidance_cfg,
                condition=condition,
                neg_condition=neg_condition,
                fps=fps,
                output_dtype=None,
                fwd_pred_type="flow",
            )
            # For rectified flow, the deterministic step on the flow is exactly the reference's
            # ``ratio * x_t + (1 - ratio) * x0`` blend. Keep schedule arithmetic in
            # the scheduler so it owns the coefficient precision and modality mapping.
            latents = ns.step(latents, flow, t, grid[step + 1], pred_type="flow")

        return latents
