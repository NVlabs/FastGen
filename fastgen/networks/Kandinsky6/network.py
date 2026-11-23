# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kandinsky 6 Pro video and synchronized audio/video flow matching.

The native Pro preset uses 121 frames at 24 fps, 480x864 pixels, CFG 5 and
50 Euler steps with shift 5; the pi-Flow (distilled) checkpoints are rejected.
Video latents are BCTHW; joint generation adds
BCS audio latents on the same flow clock. Image conditioning appends a clean
reference frame at the tail, with temporal position zero and token role one;
it does not replace the generated first frame. This is the reference's
``tail_cond_first_frame`` scheme of its Python API and Diffusers; the reference
CLI preset still defaults to the legacy ``pretrain`` scheme. Seed audio noise independently
with video_seed + 1 when comparing against the native pipeline. Inference prepares
the conditioning image with the native bilinear resize/crop; parity also needs the same
sampled image posterior, not merely the same diffusion seed.

Diffusers supplies the architecture; native precision, codecs and conditioning
remain authoritative. Discriminators read the fused blocks' video and audio tokens.
Secondary-time and causal methods require additional architecture support.
"""

import types
from collections.abc import Mapping
from numbers import Real

import torch
from diffusers import Kandinsky6Transformer3DModel

from fastgen.networks.network import FastGenNetwork, validate_conditioning
from fastgen.networks.noise_schedule import CompositeNoiseSchedule
from fastgen.utils.env import hf_load
from fastgen.utils.guidance import GuidanceConfig
from fastgen.utils.tree_utils import tmap

from .precision import install_native_precision, reset_deterministic_buffers, rope_precision_overrides
from .preprocessors import Kandinsky6TextEncoder, Kandinsky6VAE, audio_shape

MODEL_ID = "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"


def classify_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    pooled_projections,
    timestep,
    audio_hidden_states=None,
    visual_rope_pos=None,
    encoder_attention_mask=None,
    visual_token_type_ids=None,
    return_features_early=False,
    feature_indices=None,
):
    """``Kandinsky6Transformer3DModel.forward`` of Diffusers 0.42.1 with fused-block features.

    Returns ``(video, audio, features)``, with ``audio=None`` without audio input. ``features`` holds one
    block output per requested index in traversal order: the ``[B, T, H, W, C]`` video tokens, paired with the
    ``[B, L, C]`` audio tokens as ``{"video", "audio"}`` in joint mode. An early exit returns only ``features``.
    """
    if visual_token_type_ids is not None and self.visual_token_type_num_embeddings == 0:
        raise ValueError("`visual_token_type_ids` requires `visual_token_type_num_embeddings > 0`")
    checkpoint = torch.is_grad_enabled() and self.gradient_checkpointing
    device = hidden_states.device

    text_pos = torch.arange(encoder_hidden_states.shape[1], device=device)
    video_text_rope = self.video_text_rope_embeddings(text_pos)
    audio_text_rope = self.audio_text_rope_embeddings(text_pos)

    video_text_embed = self.video_text_embeddings(encoder_hidden_states)
    video_temb = self.video_time_embeddings(timestep) + self.video_pooled_text_embeddings(pooled_projections)
    for block in self.video_text_transformer_blocks:
        args = (video_text_embed, video_temb, video_text_rope, encoder_attention_mask)
        video_text_embed = self._gradient_checkpointing_func(block, *args) if checkpoint else block(*args)

    audio_text_embed = self.audio_text_embeddings(encoder_hidden_states)
    audio_temb = self.audio_time_embeddings(timestep) + self.audio_pooled_text_embeddings(pooled_projections)
    for block in self.audio_text_transformer_blocks:
        args = (audio_text_embed, audio_temb, audio_text_rope, encoder_attention_mask)
        audio_text_embed = self._gradient_checkpointing_func(block, *args) if checkpoint else block(*args)

    visual_embed = self.visual_embeddings(hidden_states)
    if visual_token_type_ids is not None:
        token_types = self.visual_token_type_embeddings(visual_token_type_ids)
        visual_embed = visual_embed + token_types[:, :, None, None, :]
    visual_shape = visual_embed.shape[1:4]
    if visual_rope_pos is None:
        visual_rope_pos = tuple(torch.arange(size, device=device) for size in visual_shape)
    visual_rope = self.visual_rope_embeddings(visual_rope_pos, self.scale_factor)
    visual_embed = visual_embed.flatten(1, 3)
    visual_rope = visual_rope.flatten(0, 2)

    audio_embed = audio_rope = None
    if audio_hidden_states is not None:
        audio_embed = self.audio_embeddings(audio_hidden_states)
        audio_rope = self.audio_rope_embeddings(torch.arange(audio_hidden_states.shape[1], device=device))

    features = []
    for index, block in enumerate(self.visual_transformer_blocks):
        args = (
            visual_embed,
            audio_embed,
            video_text_embed,
            audio_text_embed,
            (video_temb, audio_temb),
            visual_rope,
            audio_rope,
            encoder_attention_mask,
        )
        visual_embed, audio_embed = self._gradient_checkpointing_func(block, *args) if checkpoint else block(*args)
        if feature_indices is not None and index in feature_indices:
            video_feature = visual_embed.reshape(-1, *visual_shape, visual_embed.shape[-1])
            features.append(video_feature if audio_embed is None else {"video": video_feature, "audio": audio_embed})
        if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
            return features

    visual_embed = visual_embed.reshape(-1, *visual_shape, visual_embed.shape[-1])
    video_out = self.out_layer(visual_embed, video_temb)
    audio_out = self.audio_out_layer(audio_embed, audio_temb) if audio_embed is not None else None
    return video_out, audio_out, features


class Kandinsky6(FastGenNetwork):
    """Kandinsky 6 Pro for text/image-conditioned video, optionally with synchronized audio."""

    latent_compression_factors = (4, 8, 8)
    audio_sampling_rate = 44100

    def __init__(
        self,
        model_id_or_local_path=MODEL_ID,
        revision=None,
        modalities=("video", "audio"),
        image_conditioned=False,
        load_pretrained=True,
        gradient_checkpointing=True,
        fps=24.0,
        scheduler_scale=5.0,
        max_sequence_length=1024,
        **schedule_kwargs,
    ):
        """
        Args:
            model_id_or_local_path: Diffusers repository ID or local path of the Pro checkpoint.
            revision: Hub revision of ``model_id_or_local_path``.
            modalities: ``("video",)`` or joint ``("video", "audio")``.
            image_conditioned: Condition on a clean reference frame at the tail (see the module docstring).
            load_pretrained: Load the pretrained transformer; otherwise, or under meta init, build it from its config.
            gradient_checkpointing: Checkpoint the transformer blocks' activations.
            fps: Video frame rate, which sets the synchronized audio length.
            scheduler_scale: Shift of the native sampling time grid.
            max_sequence_length: Maximum number of Qwen prompt tokens.
            **schedule_kwargs: Noise schedule options; ``max_t`` defaults to 1.
        """
        self.modalities = tuple(modalities)
        if self.modalities not in (("video",), ("video", "audio")):
            raise ValueError("Kandinsky6 supports video or joint video/audio, in that order")
        if fps <= 0 or scheduler_scale <= 0 or max_sequence_length < 1:
            raise ValueError("fps, scheduler_scale and max_sequence_length must be positive")
        schedule_kwargs.setdefault("max_t", 1.0)
        super().__init__(net_pred_type="flow", schedule_type="rf", **schedule_kwargs)
        self.model_id_or_local_path = model_id_or_local_path
        self.revision = revision
        self.image_conditioned = image_conditioned
        self.fps = fps
        self.scheduler_scale = scheduler_scale
        self.max_sequence_length = max_sequence_length
        if load_pretrained and not self._is_in_meta_context():
            self.transformer = hf_load(
                Kandinsky6Transformer3DModel.from_pretrained,
                model_id_or_local_path,
                subfolder="transformer",
                revision=revision,
                torch_dtype=torch.bfloat16,
            )
            # The native checkpoint loader rounds *all* parameters, including
            # Diffusers' FP32 exceptions. Keep rotary buffers in FP32.
            for parameter in self.transformer.parameters():
                parameter.data = parameter.data.to(torch.bfloat16)
        else:
            config = hf_load(
                Kandinsky6Transformer3DModel.load_config,
                model_id_or_local_path,
                subfolder="transformer",
                revision=revision,
            )
            self.transformer = Kandinsky6Transformer3DModel.from_config(config)
        config = self.transformer.config
        if (config.out_visual_dim, config.out_audio_dim) != (config.in_visual_dim, config.in_audio_dim):
            raise ValueError(
                f"{model_id_or_local_path!r} predicts {config.out_visual_dim} video and {config.out_audio_dim} audio "
                f"channels for {config.in_visual_dim} and {config.in_audio_dim} latent channels, a pi-Flow policy of a "
                "distilled checkpoint; Kandinsky6 supports the flow-matching checkpoints"
            )
        install_native_precision(self.transformer)
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

    def _fixed_precision_overrides(self):
        """Keep the rotary angle buffers in FP32, as the native model does."""
        return rope_precision_overrides(self.transformer)

    def reset_parameters(self):
        """Also rebuild the non-persistent rotary angles and time-embedding frequencies."""
        super().reset_parameters()
        reset_deterministic_buffers(self.transformer)

    def _set_noise_schedule(self, schedule_type=None, **schedule_kwargs):
        """Give joint video and audio the same schedule, with video as the primary clock."""
        if len(self.modalities) == 1:
            return super()._set_noise_schedule(schedule_type, **schedule_kwargs)
        if schedule_type is not None:
            self.schedule_type = schedule_type
        config = {"name": self.schedule_type, **schedule_kwargs}
        self.noise_scheduler = CompositeNoiseSchedule(
            schedules={name: config for name in self.modalities}, primary_key="video"
        )

    @staticmethod
    def default_guidance_cfg():
        """Native Pro guidance: constant CFG 5."""
        config = GuidanceConfig()
        config.cfg.scale = 5.0
        return config

    def init_preprocessors(self):
        """Initialize the Qwen/CLIP text encoder and the VAE bundle."""
        if "text_encoder" not in self.preprocessors:
            self.preprocessors["text_encoder"] = Kandinsky6TextEncoder(
                self.model_id_or_local_path, self.revision, self.max_sequence_length
            )
        self.init_vae()

    def init_vae(self):
        """Initialize the video VAE and, for joint generation, the audio codec."""
        if "vae" not in self.preprocessors:
            self.preprocessors["vae"] = Kandinsky6VAE(self.model_id_or_local_path, self.modalities, self.revision)

    def encode_real(self, real_raw):
        """Encode raw clips with the VAE; a joint clip without audio encodes silence of its duration."""
        if len(self.modalities) > 1 and isinstance(real_raw, Mapping) and "audio" not in real_raw:
            # The audio-video loader omits a silent clip's audio. Encode silence of the clip's duration so
            # every rank runs the sharded audio codec and the joint state stays complete.
            video = real_raw["video"]
            samples = round(video.shape[2] / self.fps * self.audio_sampling_rate)
            real_raw = {**real_raw, "audio": video.new_zeros(video.shape[0], samples)}
        return super().encode_real(real_raw)

    def build_conditioning(self, data, input_shape=None, precision=None):
        """Resolve the text branches and, when image-conditioned, add the reference frame latent to each.

        The latent is ``condition_image``, or the VAE encoding of ``condition_image_raw`` or else of the first frame
        of ``real_raw``.
        """
        validate_conditioning(data)
        branches = self._resolve_conditions(data, dtype=self.precision_model)
        reference = None
        if self.image_conditioned:
            reference = data.get("condition_image")
            if reference is None:
                image = data.get("condition_image_raw")
                if image is None:
                    image = data.get("real_raw")
                    image = image.get("video") if isinstance(image, Mapping) else image
                if image is None:
                    raise ValueError("Image conditioning requires condition_image or normalized condition_image_raw")
                if image.ndim == 5:
                    image = image[:, :, :1]
                if image.ndim not in (4, 5):
                    raise ValueError("condition_image_raw must have BCHW or BCTHW layout")
                if "vae" not in self.preprocessors:
                    raise ValueError("Initialize the VAE before encoding image conditioning")
                reference = self.preprocessors["vae"].encode_video(image)
            if reference.ndim != 5 or reference.shape[2] != 1:
                raise ValueError("condition_image must contain one BCTHW latent frame")
            reference = reference.to(dtype=precision) if precision is not None else reference
        return {
            name: {"text": text, **({"reference_latents": reference} if self.image_conditioned else {})}
            for name, text in branches.items()
        }

    def forward(
        self,
        x_t,
        t,
        condition=None,
        r=None,
        return_features_early=False,
        feature_indices=None,
        return_logvar=False,
        fwd_pred_type=None,
        apply_output_conditioning=True,
        output_dtype="same",
        **kwargs,
    ):
        """Predict the flow of video or joint video/audio latents; see ``FastGenNetwork.forward``.

        Image conditioning appends ``condition["reference_latents"]`` as a tail frame at temporal position zero with
        token role one; the prediction drops it. ``feature_indices`` adds the fused blocks' discriminator features.
        """
        self._validate_forward_kwargs(
            kwargs,
            r=r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            num_features=len(self.transformer.visual_transformer_blocks),
        )
        if return_logvar:
            raise NotImplementedError("Kandinsky6 has no learned variance head")
        if return_features_early and feature_indices is not None and not feature_indices:
            return []
        joint = len(self.modalities) > 1
        if joint:
            if not isinstance(x_t, Mapping) or set(x_t) != set(self.modalities):
                raise ValueError("Joint Kandinsky6 state requires exactly video and audio tensors")
            video, audio = x_t["video"], x_t["audio"]
        else:
            if not torch.is_tensor(x_t):
                raise TypeError("Video-only Kandinsky6 state must be a tensor")
            video, audio = x_t, None
        hidden = self.cast_to_model_precision(video).permute(0, 2, 3, 4, 1)
        text = condition["text"] if isinstance(condition, Mapping) else condition
        if not isinstance(text, (tuple, list)) or len(text) != 3:
            raise ValueError("Kandinsky6 text conditioning requires Qwen, pooled CLIP and padding mask")
        tokens, pooled, attention_mask = text
        if self.precision_model is not None and any(value.dtype != self.precision_model for value in (tokens, pooled)):
            raise TypeError("Text conditioning must use the prepared model precision")
        if attention_mask.dtype != torch.bool:
            raise TypeError("Text padding mask must have boolean dtype")
        positions = roles = None
        original_frames = hidden.shape[1]
        if self.image_conditioned:
            if not isinstance(condition, Mapping) or "reference_latents" not in condition:
                raise ValueError("Image-conditioned Kandinsky6 requires reference_latents")
            reference = condition["reference_latents"]
            expected = (video.shape[0], video.shape[1], 1, *video.shape[-2:])
            if tuple(reference.shape) != expected:
                raise ValueError(f"Reference latent shape must be {expected}, got {tuple(reference.shape)}")
            hidden = torch.cat((hidden, self.cast_to_model_precision(reference).permute(0, 2, 3, 4, 1)), dim=1)
            patch_t, patch_h, patch_w = self.transformer.config.patch_size
            if patch_t != 1:
                raise ValueError("Tail-frame conditioning requires temporal patch size one")
            positions = (
                torch.cat(
                    (
                        torch.arange(original_frames, device=hidden.device),
                        torch.zeros(1, device=hidden.device, dtype=torch.long),
                    )
                ),
                torch.arange(hidden.shape[2] // patch_h, device=hidden.device),
                torch.arange(hidden.shape[3] // patch_w, device=hidden.device),
            )
            roles = torch.zeros(hidden.shape[:2], device=hidden.device, dtype=torch.long)
            roles[:, -1] = 1
        mask = torch.zeros_like(hidden[..., :1])
        if self.image_conditioned:
            mask[:, -1] = 1
        hidden = torch.cat((hidden, torch.zeros_like(hidden), mask), dim=-1)
        dtype = self.precision_model or hidden.dtype
        with torch.autocast(hidden.device.type, dtype=dtype, enabled=dtype in (torch.bfloat16, torch.float16)):
            # Match the native contiguous BSC input and its fused biased-linear CUDA path.
            audio_hidden = self.cast_to_model_precision(audio).transpose(1, 2).contiguous() if joint else None
            outputs = self.transformer(
                hidden_states=hidden,
                encoder_hidden_states=tokens,
                pooled_projections=pooled,
                timestep=(t * 1000).float(),
                audio_hidden_states=audio_hidden,
                visual_rope_pos=positions,
                encoder_attention_mask=attention_mask,
                visual_token_type_ids=roles,
                return_features_early=return_features_early,
                feature_indices=feature_indices,
            )
        if return_features_early:
            return self._discriminator_features(outputs, original_frames)
        video_out, audio_out, features = outputs
        prediction = video_out[:, :original_frames].permute(0, 4, 1, 2, 3)
        if joint:
            prediction = {"video": prediction, "audio": audio_out.transpose(1, 2)}
        output = self.transform_output(
            prediction, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
        )
        return (
            [output, self._discriminator_features(features, original_frames)] if feature_indices is not None else output
        )

    @staticmethod
    def _discriminator_features(features, frames):
        """Video tokens to ``[B, C, T, H, W]`` without the reference tail; audio tokens to ``[B, C, L]``."""

        def video(value):
            return value[:, :frames].permute(0, 4, 1, 2, 3)

        return [
            video(feature)
            if torch.is_tensor(feature)
            else {"video": video(feature["video"]), "audio": feature["audio"].transpose(1, 2)}
            for feature in features
        ]

    def sample(
        self,
        noise,
        condition=None,
        neg_condition=None,
        sample_steps=50,
        sample_type="ode",
        guidance_cfg=None,
        scheduler_scale=None,
    ):
        """Native Euler sampler: ``sample_steps`` steps on the ``scheduler_scale``-shifted grid, constant scalar CFG."""
        self._validate_sample_kwargs(sample_type, ("rf",), ("ode",))
        guidance_cfg = guidance_cfg or self.default_guidance_cfg()
        guidance_cfg.validate(supported_terms=("cfg",), network_name="Kandinsky6 native sampler")
        scale = guidance_cfg.cfg.scale
        if guidance_cfg.cfg.boundary is not None or (scale is not None and not isinstance(scale, Real)):
            raise NotImplementedError("Kandinsky6 native sampling requires constant scalar CFG")
        use_cfg = scale is not None and abs(scale - 1.0) > 1e-6
        if use_cfg and neg_condition is None:
            raise ValueError("Kandinsky6 native CFG sampling requires negative conditioning")
        shift = self.scheduler_scale if scheduler_scale is None else scheduler_scale
        if sample_steps < 1 or shift <= 0:
            raise ValueError("sample_steps and scheduler_scale must be positive")
        video = noise["video"] if isinstance(noise, Mapping) else noise
        times = torch.linspace(1, 0, sample_steps + 1, device=video.device, dtype=torch.float32)
        times = shift * times / (1 + (shift - 1) * times)
        state = tmap(lambda value: self.cast_to_model_precision(value).clone(), noise)
        for time, delta in zip(times[:-1], times.diff()):
            t = time.to(self.noise_scheduler.t_precision).expand(video.shape[0])
            velocity = self(state, t, condition, output_dtype=self.precision_model)
            if use_cfg:
                negative = self(state, t, neg_condition, output_dtype=self.precision_model)
                # Match the native pipeline's separate BF16 CFG rounding operations.
                velocity = tmap(lambda pos, neg: neg + scale * (pos - neg), velocity, negative)
            # A model-precision prediction keeps the native update; an unprepared network's high-precision one would
            # otherwise promote the state and the next forward's inputs.
            state = tmap(lambda value, prediction: (value + delta * prediction).to(value.dtype), state, velocity)
        return state

    def fully_shard(self, **kwargs):
        """Shard each text and fused block, then the transformer root, with FSDP-compatible checkpointing."""
        from torch.distributed.fsdp import fully_shard
        from diffusers.models.transformers.transformer_kandinsky6 import (
            Kandinsky6FusedTransformerDecoderBlock,
            Kandinsky6TransformerEncoderBlock,
        )
        from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing

        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(
                self.transformer, (Kandinsky6TransformerEncoderBlock, Kandinsky6FusedTransformerDecoderBlock)
            )
        for block in (
            *self.transformer.video_text_transformer_blocks,
            *self.transformer.audio_text_transformer_blocks,
            *self.transformer.visual_transformer_blocks,
        ):
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

    def infer_shape(self, input_shape):
        """Infer the joint audio shape from the video shape and ``fps``, checking a given one; video passes through."""
        if len(self.modalities) == 1:
            if isinstance(input_shape, Mapping):
                raise ValueError("Video-only Kandinsky6 requires a flat video shape")
            return input_shape
        if not isinstance(input_shape, Mapping) or set(input_shape) != set(self.modalities):
            raise ValueError("Joint Kandinsky6 requires video and audio input_shape keys")
        video = input_shape["video"]
        if video is None or len(video) != 4:
            raise ValueError("A CTHW video shape is required to infer synchronized audio")
        expected_audio = audio_shape(video, self.fps)
        if input_shape["audio"] is not None and list(input_shape["audio"]) != expected_audio:
            raise ValueError(f"Audio shape must match video duration: expected {expected_audio}")
        return {"video": list(video), "audio": expected_audio}
