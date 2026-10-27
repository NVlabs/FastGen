# SPDX-FileCopyrightText: Copyright (c) 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Mapping, Optional, List, Sequence, Set, Union, Tuple
import types

import numpy as np
import torch
import torch.utils.checkpoint
from torch.distributed.fsdp import fully_shard

from diffusers import FlowMatchEulerDiscreteScheduler

from diffusers.models import Flux2Transformer2DModel
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, LlamaTokenizerFast, Qwen2Tokenizer

from fastgen.networks.network import FastGenNetwork, FastGenModule, validate_conditioning
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.env import hf_load
from fastgen.utils.tree_utils import OutputDtype, assert_same, tmap, to
import fastgen.utils.logging_utils as logger


# ---------------------------------------------------------------------------
# Text Encoders
# ---------------------------------------------------------------------------


# System message used by BFL's official Flux.2 text encoder. Must match what
# the model was trained against, otherwise the Mistral VLM produces hidden
# states from a completely different distribution than the Flux.2 transformer
# was conditioned on -- the symptom is "iter 1 student samples look like noise"
# even though the pretrained weights are fine.
# Source:
#   diffusers.pipelines.flux2.system_messages.SYSTEM_MESSAGE
#   https://github.com/black-forest-labs/flux2/blob/main/src/flux2/system_messages.py
FLUX2_SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured "
    "responses focusing on object relationships, object\nattribution and actions "
    "without speculation."
)


class Flux2TextEncoder(FastGenModule):
    """Text encoder for Flux.2-dev using Mistral-Small-3.2-24B-Instruct.

    Flux.2-dev replaces Flux.1's dual CLIP + T5 encoders with a single
    vision-language model. Hidden states are extracted from multiple
    intermediate layers and concatenated to form the prompt embeddings.

    The default extraction layers [10, 20, 30] produce a joint_attention_dim
    of 15360 (5120 per layer x 3 layers) for Flux.2-dev.

    The Flux.2 transformer takes ONLY the per-token sequence embedding from
    this encoder; there is no separate pooled CLIP-style embedding. We still
    return ``(None, prompt_embeds)`` so the existing
    ``condition: Tuple[pooled, prompt]`` contract used elsewhere in FastGen
    keeps working unchanged -- callers must simply not consume the first
    element.
    """

    def __init__(
        self,
        model_id_or_local_path: str,
        hidden_state_layers: Optional[List[int]] = None,
        system_message: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        """Initialize the Mistral VLM text encoder.

        ``dtype`` defaults to bfloat16 because running this encoder in
        float32 produces embeddings noticeably different from the bfloat16
        path that BFL trained against, which degrades downstream quality
        (same observation as ``QwenImageTextEncoder``).
        """
        super().__init__()
        self.hidden_state_layers = hidden_state_layers or [10, 20, 30]
        self.system_message = system_message if system_message is not None else FLUX2_SYSTEM_MESSAGE

        self.processor = hf_load(LlamaTokenizerFast.from_pretrained, model_id_or_local_path, subfolder="tokenizer")
        self.text_encoder = hf_load(
            AutoModel.from_pretrained, model_id_or_local_path, subfolder="text_encoder", dtype=dtype
        )

    def _format_messages(self, prompts: List[str]) -> List[List[dict]]:
        """Wrap each prompt in the (system, user) chat structure Flux.2 expects.

        Mirrors ``diffusers.pipelines.flux2.pipeline_flux2.format_input``.
        """
        # Strip [IMG] tokens to avoid Pixtral validation issues on truncation,
        # matching diffusers' behavior.
        cleaned = [p.replace("[IMG]", "") for p in prompts]
        return [
            [
                {"role": "system", "content": [{"type": "text", "text": self.system_message}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned
        ]

    def encode(
        self,
        conditioning: Optional[Any] = None,
        max_sequence_length: int = 512,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Encode text prompts to embeddings.

        Args:
            conditioning: Text prompt(s) to encode.
            max_sequence_length: Maximum sequence length for tokenization.

        Returns:
            Tuple of ``(None, prompt_embeds)``:
            - The first element is ``None`` because Flux.2 has no separate
              pooled embedding (kept in the tuple for API compatibility).
            - prompt_embeds: ``[B, seq_len, joint_attention_dim]``
              (joint_attention_dim = len(hidden_state_layers) * hidden_size,
              i.e. 3 * 5120 = 15360 for Flux.2-dev).
        """
        if isinstance(conditioning, str):
            conditioning = [conditioning]

        # Apply Mistral chat template with the Flux.2 system message. This is
        # required: the encoder is an instruction-tuned VLM, and Flux.2 was
        # trained against hidden states produced by chat-template-formatted
        # input. Skipping this gives the transformer out-of-distribution
        # conditioning and the samples come out as noise.
        messages_batch = self._format_messages(conditioning)
        inputs = self.processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        return self(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Stack the requested intermediate layers and interleave them per
        # token (matches diffusers' Flux2 path):
        #   [B, K, L, D] --(permute)--> [B, L, K, D] --(reshape)--> [B, L, K*D]
        stacked = torch.stack([outputs.hidden_states[i] for i in self.hidden_state_layers], dim=1)  # [B, K, L, D]
        B, K, L, D = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, L, K * D)

        # Flux.2 has no pooled embedding; return None in the first slot for
        # API compatibility with the rest of FastGen's condition handling.
        return None, prompt_embeds


class Flux2KleinTextEncoder(FastGenModule):
    """Text encoder for Flux.2-klein using Qwen3.

    Klein variants use a Qwen3 causal LM as text encoder with multi-layer
    hidden state extraction, similar to the dev variant but with a smaller
    model and a different chat-template contract.

    The default extraction layers [9, 18, 27] produce a joint_attention_dim
    of 7680 (2560 per layer x 3 layers) for Flux.2-klein-base-4B.

    Like ``Flux2TextEncoder``, this returns ``(None, prompt_embeds)``: Flux.2
    has no separate pooled embedding -- the transformer only consumes the
    per-token sequence. The ``None`` keeps API compatibility with the rest of
    FastGen's ``condition: Tuple[pooled, prompt]`` plumbing.

    Implementation contract (must match diffusers' ``pipelines.flux2.pipeline_flux2_klein
    ._get_qwen3_prompt_embeds``, otherwise the pretrained Klein transformer
    receives out-of-distribution conditioning and produces noise):

    1. Wrap each prompt as ``[{"role": "user", "content": prompt}]``
       (Qwen3 user-only template; no system message, unlike Mistral/dev).
    2. Apply ``apply_chat_template(..., add_generation_prompt=True,
       enable_thinking=False)`` -- ``add_generation_prompt`` adds the trailing
       assistant-turn marker, and ``enable_thinking=False`` suppresses Qwen3's
       ``<think>...</think>`` scratchpad which Flux.2 was NOT trained against.
    3. Tokenize the resulting strings with ``padding="max_length"`` /
       ``truncation=True`` to a fixed sequence length.
    4. Stack hidden states from the requested layers and interleave them
       per-token: ``[B, K, L, D] -> [B, L, K, D] -> [B, L, K*D]``.
    """

    def __init__(
        self,
        model_id_or_local_path: str,
        hidden_state_layers: Optional[List[int]] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        """Initialize the Qwen3 text encoder.

        ``dtype`` defaults to bfloat16 because running this encoder in
        float32 produces embeddings noticeably different from the bfloat16
        path that BFL trained against, which degrades downstream quality
        (same observation as ``QwenImageTextEncoder``).
        """
        super().__init__()
        self.hidden_state_layers = hidden_state_layers or [9, 18, 27]

        # CRITICAL: the tokenizer must be loaded from the `tokenizer/`
        # subfolder, NOT `text_encoder/`. BFL's FLUX.2 repos ship the model
        # weights in `text_encoder/` and the Qwen3 tokenizer (with a
        # bundled `chat_template.jinja`) in a separate `tokenizer/`
        # subfolder. `AutoTokenizer.from_pretrained(..., subfolder="text_encoder")`
        # silently returns a 1-token Qwen2 fallback that maps every prompt
        # to an empty token sequence, so the Qwen3 hidden states fed to the
        # Flux.2 transformer are constant garbage and the iter-0 student
        # samples come out as pure noise (loss still goes down because the
        # model slowly learns to predict mean velocity).  Layout has been
        # verified against all four current variants:
        #   FLUX.2-dev, FLUX.2-klein-{4B,base-4B,base-9B}.
        self.tokenizer = hf_load(Qwen2Tokenizer.from_pretrained, model_id_or_local_path, subfolder="tokenizer")
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError(
                f"Tokenizer loaded from {model_id_or_local_path}/tokenizer has no chat_template. "
                "BFL's FLUX.2 repos always bundle `chat_template.jinja` here; "
                "this likely means the repo layout has changed or the wrong "
                "subfolder was queried."
            )
        self.text_encoder = hf_load(
            AutoModelForCausalLM.from_pretrained, model_id_or_local_path, subfolder="text_encoder", dtype=dtype
        )

    def _format_and_tokenize(self, prompts: List[str], max_sequence_length: int):
        """Apply Qwen3 chat template per prompt, then tokenize with padding.

        Mirrors diffusers' ``_get_qwen3_prompt_embeds`` in
        ``pipelines/flux2/pipeline_flux2_klein.py``.
        """
        all_input_ids = []
        all_attention_masks = []
        for single_prompt in prompts:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])
        return torch.cat(all_input_ids, dim=0), torch.cat(all_attention_masks, dim=0)

    def encode(
        self,
        conditioning: Optional[Any] = None,
        max_sequence_length: int = 512,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Encode text prompts to embeddings.

        Args:
            conditioning: Text prompt(s) to encode.
            max_sequence_length: Maximum sequence length for tokenization.

        Returns:
            Tuple of ``(None, prompt_embeds)``. The first slot is always
            ``None`` (Flux.2 has no pooled embedding); the second has shape
            ``[B, seq_len, joint_attention_dim]`` where joint_attention_dim
            is ``len(hidden_state_layers) * text_encoder.config.hidden_size``.
        """
        if isinstance(conditioning, str):
            conditioning = [conditioning]

        input_ids, attention_mask = self._format_and_tokenize(conditioning, max_sequence_length)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,  # only hidden_states is used; skips the full-vocabulary lm_head projection
        )

        stacked = torch.stack([outputs.hidden_states[i] for i in self.hidden_state_layers], dim=1)  # [B, K, L, D]
        B, K, L, D = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, L, K * D)

        return None, prompt_embeds


# ---------------------------------------------------------------------------
# Image Encoder (VAE)
# ---------------------------------------------------------------------------


class Flux2ImageEncoder(FastGenModule):
    """FLUX.2 VAE wrapper using ``AutoencoderKLFlux2``.

    The VAE emits 32-channel latents at 8x spatial compression. Its
    ``patch_size=(2, 2)`` describes the post-encode BatchNorm layout, not
    additional VAE downsampling. BFL's ``AutoEncoder`` owns this patchify/BN
    transform; Diffusers externalizes it to the pipeline. FastGen applies the
    equivalent tiled statistics to unpatchified latents and patchifies later
    in ``Flux2._pack_latents``.

    FLUX.2-dev and all Klein variants share BFL's FP32 autoencoder. The native
    CLI calls it under ``torch.no_grad()`` without autocast. Encode is FP32;
    during decode, combining the BF16 sampler output with the FP32 BatchNorm
    mean/std produces the FP32 input consumed by the FP32 decoder. Thus both
    VAE forwards and the statistics transforms run in FP32.
    """

    def __init__(self, model_id_or_local_path: str):
        super().__init__()
        self.vae: AutoencoderKLFlux2 = hf_load(
            AutoencoderKLFlux2.from_pretrained, model_id_or_local_path, subfolder="vae"
        )

        # FLUX.2 normalizes via the VAE's BatchNorm2d running stats.
        patch = tuple(self.vae.config.patch_size)
        if len(patch) != 2:
            raise ValueError(f"Expected a 2D Flux.2 VAE patch_size, got {patch}.")
        self._patch_h, self._patch_w = int(patch[0]), int(patch[1])
        self._latent_channels = int(self.vae.config.latent_channels)
        self._bn_eps = float(self.vae.config.batch_norm_eps)

    @property
    def precision_model(self) -> Optional[torch.dtype]:
        """Honor the released VAE's FP32/``force_upcast`` interface contract."""
        if getattr(self.vae.config, "force_upcast", False):
            return torch.float32
        return super().precision_model

    @property
    def latent_compression_factors(self) -> tuple[int, int]:
        """Pixel-to-latent ``(H, W)`` downsampling of the VAE; every down block but the last halves resolution.

        8x for Flux.2 (a 1024x1024 image -> 128x128 latent); ``patch_size`` adds none (see the class docstring).
        """
        factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return (factor, factor)

    def _bn_stats_per_position(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Return BN ``(mean, std)`` reshaped to ``[C, patch_h, patch_w]``.

        The VAE's ``bn`` tracks one running mean and variance per *flat*
        post-patchify channel ``c * (patch_h * patch_w) + p * patch_w + q``.
        Reshaping that to ``[C, patch_h, patch_w]`` gives a stat indexed by
        ``(latent_channel, h_subpos, w_subpos)``, which is what the tiled
        broadcast below needs.
        """
        c, ph, pw = self._latent_channels, self._patch_h, self._patch_w
        mean = self.vae.bn.running_mean.to(dtype=dtype).view(c, ph, pw)
        # Match Flux2Pipeline: sqrt is evaluated in the checkpoint buffer dtype,
        # then the resulting standard deviation is cast to the latent dtype.
        std = torch.sqrt(self.vae.bn.running_var.view(c, ph, pw) + self._bn_eps).to(dtype=dtype)
        return mean, std

    def _tile_to_latent_grid(self, stat_2x2: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Tile a ``[C, patch_h, patch_w]`` stat across the latent grid.

        Returns ``[1, C, height, width]`` such that position ``(y, x)`` gets
        ``stat_2x2[:, y % patch_h, x % patch_w]``. This is exactly what
        running the BN in patchified space and then unpatchifying produces,
        but without the round trip.
        """
        if height % self._patch_h or width % self._patch_w:
            raise ValueError(
                f"Latent spatial dims ({height}, {width}) must be divisible by "
                f"patch_size ({self._patch_h}, {self._patch_w}) for Flux.2 BN normalization."
            )
        tiled = stat_2x2.repeat(1, height // self._patch_h, width // self._patch_w)
        return tiled.unsqueeze(0)

    def encode(self, real_images: torch.Tensor, mode: str = "argmax") -> torch.Tensor:
        """Encode images to latent space.

        Pipeline (matching diffusers' Flux.2 ``_encode_vae_image`` semantics):

            1. VAE encode  -> ``[B, 32, H/8, W/8]`` (8x spatial compression)
            2. Per-(channel, patch-position) BN normalization using the VAE's
               stored running stats, applied in pre-patchify space.

        Args:
            real_images: Input images in [-1, 1] range, shape ``[B, 3, H, W]``.
            mode: ``"argmax"`` takes the posterior mean, as BFL's ``AutoEncoder.encode``
                and diffusers' Flux.2 pipelines and training script do for both
                training images and image-edit references; ``"sample"`` draws from
                the posterior.

        Returns:
            torch.Tensor: BN-normalized latents ``[B, 32, H//8, W//8]``. The
                2x2 patchification is performed downstream in
                ``Flux2._pack_latents`` rather than here.
        """
        return self(real_images, mode=mode, operation="encode")

    def decode(self, latent_images: torch.Tensor) -> torch.Tensor:
        """Decode latents to images (inverse of ``encode``).

        Args:
            latent_images: BN-normalized latents from ``encode``.

        Returns:
            torch.Tensor: Decoded images in [-1, 1] range.
        """
        return self(latent_images, operation="decode")

    def forward(self, data: torch.Tensor, operation: str, mode: str = "argmax") -> torch.Tensor:
        if operation == "encode":
            dist = self.vae.encode(data.to(dtype=self.precision_model), return_dict=False)[0]
            if mode == "sample":
                latent_images = dist.sample()
            elif mode == "argmax":
                latent_images = dist.mode()
            else:
                raise ValueError(f"Invalid mode: {mode}. Supported modes: ['sample', 'argmax']")

            h, w = latent_images.shape[-2:]
            mean, std = self._bn_stats_per_position(latent_images.dtype)
            mean = self._tile_to_latent_grid(mean, h, w)
            std = self._tile_to_latent_grid(std, h, w)
            return (latent_images - mean) / std

        if operation == "decode":
            model_latents = data.to(dtype=self.precision_model)
            h, w = model_latents.shape[-2:]
            mean, std = self._bn_stats_per_position(model_latents.dtype)
            mean = self._tile_to_latent_grid(mean, h, w)
            std = self._tile_to_latent_grid(std, h, w)

            latents = model_latents * std + mean
            images = self.vae.decode(latents, return_dict=False)[0]
            return images.clip(-1.0, 1.0)

        raise ValueError(f"Unsupported preprocessor operation: {operation!r}")


# ---------------------------------------------------------------------------
# Modified Forward with Feature Extraction
# ---------------------------------------------------------------------------


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    target_grid: Tuple[int, int],
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[dict] = None,
    return_features_early: bool = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Modified forward pass for Flux2Transformer2DModel with feature extraction.

    This replaces the default forward to support:
    - Intermediate feature extraction at specified block indices (for discriminators)
    - Log-variance estimation via an appended linear layer
    - Early return once all requested features are collected

    Block index mapping for Flux.2-dev (56 total):
      0-7: joint transformer blocks (8 blocks)
      8-55: single transformer blocks (48 blocks)

    Args:
        hidden_states: Packed latent states [B, seq_len, in_channels].
        target_grid: Packed ``(rows, cols)`` of the noisy target image, whose tokens precede any reference-image
            tokens (image-edit mode); feature extraction reshapes them to ``[B, C, rows, cols]``.
        encoder_hidden_states: Text encoder hidden states.
        timestep: Current timestep.
        img_ids: Image position IDs.
        txt_ids: Text position IDs.
        guidance: Guidance scale embedding (None for klein/schnell).
        joint_attention_kwargs: Additional attention kwargs.
        return_features_early: If True, return features as soon as collected.
        feature_indices: Set of block indices to extract features from.
        return_logvar: If True, return log variance estimate.

    Returns:
        Model output, optionally with features or logvar.
    """
    idx, features = 0, []

    num_txt_tokens = encoder_hidden_states.shape[1]

    # 1. Time + guidance embedding and modulation
    # BFL builds t/guidance at the model-input dtype (``img.dtype``; bf16 in its pipeline: bf16
    # ``torch.randn`` noise and a bf16 model), scales by 1000 there, embeds in fp32 and casts back
    # to it. ``hidden_states`` is still that unprojected input here, and ``time_guidance_embed``
    # runs the fp32 sinusoid and casts back to ``timestep.dtype``.
    timestep_scaled = timestep.to(hidden_states.dtype) * 1000
    guidance_scaled = None if guidance is None else guidance.to(hidden_states.dtype) * 1000
    temb = self.time_guidance_embed(timestep_scaled, guidance_scaled)

    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)

    # 2. Input projections
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # 3. Positional embeddings (separate for text and image, then concatenated)
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    # 4. Joint transformer blocks (text-image interaction)
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                double_stream_mod_img,
                double_stream_mod_txt,
                concat_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if feature_indices is not None and idx in feature_indices:
            # Discriminator features should describe the *target* image only
            # (not the reference). Slice off any concatenated reference
            # tokens before reshaping.
            grid_h, grid_w = target_grid
            feat = hidden_states[:, : grid_h * grid_w, ...].clone()
            B, S, C = feat.shape
            feat = feat.permute(0, 2, 1).reshape(B, C, grid_h, grid_w)
            features.append(feat)

        if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
            return features

        idx += 1

    # 5. Concatenate text and image streams for single-block processing
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # 6. Single transformer blocks
    for block in self.single_transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                None,
                single_stream_mod,
                concat_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if feature_indices is not None and idx in feature_indices:
            # Drop the text portion AND any reference-image portion -- the
            # discriminator only scores the target image.
            grid_h, grid_w = target_grid
            img_hidden = hidden_states[:, num_txt_tokens : num_txt_tokens + grid_h * grid_w, ...]
            feat = img_hidden.clone()
            B, S, C = feat.shape
            feat = feat.permute(0, 2, 1).reshape(B, C, grid_h, grid_w)
            features.append(feat)

        if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
            return features

        idx += 1

    # 7. Remove text tokens and final projection
    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if return_features_early:
        assert feature_indices is not None
        assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
        return features

    out = [output, features] if feature_indices is not None else output

    if return_logvar:
        logvar = self.logvar_linear(temb)
        return out, logvar

    return out


# ---------------------------------------------------------------------------
# Main Network Class
# ---------------------------------------------------------------------------


class Flux2(FastGenNetwork):
    """Flux.2 network for text-to-image generation and image editing.

    Flux.2 is a second-generation flow matching model from Black Forest Labs,
    trained from scratch with major architectural changes over Flux.1:

    - 32B parameter transformer (dev): 8 double-stream + 48 single-stream blocks
    - Hidden size 6144 with 48 attention heads (dev), SwiGLU activation
    - 32-channel VAE with 8x spatial compression (vs Flux.1's 16ch / 8x);
      a 2x2 patchification is then applied at pack time, so the transformer
      sees `4 * latent_channels = 128` channels per packed token.
    - Single VLM text encoder: Mistral-3 24B (dev) or Qwen3 (klein)
    - 4-axis RoPE with dims (32, 32, 32, 32)
    - Shared AdaLN-Zero modulation across blocks of the same type

    Supported variants:
    - FLUX.2-dev (32B): Full model with guidance distillation (50 steps by default
      in BFL/diffusers; 28 is BFL's suggested quality/speed trade-off)
    - FLUX.2-klein-base-4B: Undistilled 4B foundation model, Apache 2.0
      (50 inference steps; ideal for SFT / LoRA / DMD2 distillation)
    - FLUX.2-klein-base-9B: Undistilled 9B foundation model, FLUX
      Non-Commercial License v2.1 (50 inference steps; higher quality
      target for fine-tuning / distillation than klein-base-4B)

    Reference: https://huggingface.co/black-forest-labs/FLUX.2-dev
    """

    MODEL_ID = "black-forest-labs/FLUX.2-dev"
    latent_compression_factors = (8, 8)

    # Flux.2-dev:           8 joint + 48 single = 56 blocks, hidden_size = 6144
    # Flux.2-klein-base-4B: 5 joint + 20 single = 25 blocks, hidden_size = 3072
    # Flux.2-klein-base-9B: 8 joint + 24 single = 32 blocks, hidden_size = 4096

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        disable_grad_ckpt: bool = False,
        guidance_scale: Optional[float] = 4.0,
        load_pretrained: bool = True,
        enable_image_edit: bool = False,
        ref_image_t_scale: int = 10,
        **schedule_kwargs,
    ):
        """Flux.2 constructor.

        Args:
            model_id_or_local_path: HuggingFace model ID or local path.
                Defaults to "black-forest-labs/FLUX.2-dev".
            net_pred_type: Prediction type. Defaults to "flow" for flow matching.
            schedule_type: Schedule type. Defaults to "rf" (rectified flow).
            disable_grad_ckpt: Disable activation checkpointing. FSDP needs no override: ``fully_shard``
                replaces the Diffusers checkpointing with FSDP-compatible wrappers.
            guidance_scale: Default guidance scale for guidance-distilled models.
                Use 4.0 for Flux.2-dev (which was trained with embedded guidance
                distillation); set to None for the undistilled klein-base
                variants (which were trained without an embedded guidance
                vector). Defaults to 4.0.
            load_pretrained: Whether to load pretrained weights.
            enable_image_edit: Enable image-editing / multi-reference mode.
                When True, the forward pass / sampler accept a
                ``condition_image`` reference latent (single tensor or list)
                that is concatenated with the noisy target along the
                sequence dimension, with non-zero ``T`` RoPE positions per
                reference (matching diffusers' ``Flux2Pipeline`` reference
                handling). This enables instruction-tuned image-editing
                distillation.
                Also sets ``self.is_image_edit = True`` so that
                ``Flux2.build_conditioning`` bundles the reference
                image into the structured ``condition`` /
                ``neg_condition`` dicts the forward pass expects.
            ref_image_t_scale: ``T``-axis offset between consecutive
                reference images. The i-th reference uses ``T = scale * (i + 1)``,
                matching diffusers' ``_prepare_image_ids`` (default scale=10).
                Should not need to be changed; exposed as a knob for future
                multi-reference experiments.
        """
        # The native sampler grids start at sigma 1.0, so the raw-time bound defaults to 1.0 (configs may lower it).
        schedule_kwargs.setdefault("max_t", 1.0)
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)

        self.model_id_or_local_path = model_id_or_local_path
        self.guidance_scale = guidance_scale
        self.is_image_edit = bool(enable_image_edit)
        self.ref_image_t_scale = int(ref_image_t_scale)
        logger.debug(f"Flux.2 guidance_scale={guidance_scale}, image_edit={self.is_image_edit}")

        # Initialize the transformer (handles meta device and pretrained loading)
        self._initialize_network(model_id_or_local_path, load_pretrained)
        if guidance_scale is not None and not self.transformer.config.guidance_embeds:
            raise ValueError(f"{model_id_or_local_path} has no guidance embedder; set guidance_scale=None")

        # Override forward with classify_forward for feature extraction support
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Disable cuDNN SDPA backend — known to cause mha_graph->execute errors
        # during backward with Flux-family transformers.
        if torch.backends.cuda.is_built():
            torch.backends.cuda.enable_cudnn_sdp(False)
            logger.info("Disabled cuDNN SDPA backend for Flux.2 compatibility")

        # Gradient checkpointing configuration
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    def _initialize_network(self, model_id_or_local_path: str, load_pretrained: bool) -> None:
        """Initialize the Flux.2 transformer network.

        Args:
            model_id_or_local_path: HuggingFace model ID or local path.
            load_pretrained: Whether to load pretrained weights.
        """
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading Flux.2 transformer from pretrained")
            self.transformer: Flux2Transformer2DModel = hf_load(
                Flux2Transformer2DModel.from_pretrained, model_id_or_local_path, subfolder="transformer"
            )
        else:
            config = hf_load(Flux2Transformer2DModel.load_config, model_id_or_local_path, subfolder="transformer")
            if in_meta_context:
                logger.info(
                    "Initializing Flux.2 transformer on meta device "
                    "(zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing Flux.2 transformer from config (no pretrained weights)")
                logger.warning("Flux.2 transformer being initialized from config. No weights are loaded!")
            self.transformer: Flux2Transformer2DModel = Flux2Transformer2DModel.from_config(config)

        # logvar_linear consumes ``temb`` from ``time_guidance_embed`` (see
        # classify_forward).
        temb_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.transformer.logvar_linear = torch.nn.Linear(temb_dim, 1)

    def fully_shard(self, **kwargs):
        """Shard ``self.transformer``; see ``FastGenNetwork.fully_shard``.

        Flux.2 has two types of transformer blocks:
        - transformer_blocks: Joint attention blocks for text-image interaction
        - single_transformer_blocks: Parallel QKV+FF blocks for image processing
        """
        # Detect block types from the actual model (avoids hard-coded class imports)
        joint_block_type = type(self.transformer.transformer_blocks[0])
        single_block_type = type(self.transformer.single_transformer_blocks[0])

        # Checkpointing has to happen first, for proper casting during backward pass recomputation.
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(self.transformer, (joint_block_type, single_block_type))

        # Shard joint transformer blocks
        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)

        # Shard single transformer blocks
        for block in self.transformer.single_transformer_blocks:
            fully_shard(block, **kwargs)

        fully_shard(self.transformer, **kwargs)

    # ------------------------------------------------------------------
    # Preprocessor initialization (lazy, for inference / visualization)
    # ------------------------------------------------------------------

    def init_preprocessors(self):
        """Initialize text and image encoders."""
        if "text_encoder" not in self.preprocessors:
            self.init_text_encoder()
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder the checkpoint ships: Qwen3 for the klein variants, Mistral-3 for dev."""
        text_encoder_config = hf_load(AutoConfig.from_pretrained, self.model_id_or_local_path, subfolder="text_encoder")
        encoder_cls = Flux2KleinTextEncoder if text_encoder_config.model_type == "qwen3" else Flux2TextEncoder
        self.preprocessors["text_encoder"] = encoder_cls(model_id_or_local_path=self.model_id_or_local_path)

    def init_vae(self):
        """Initialize the Flux.2 VAE."""
        self.preprocessors["vae"] = Flux2ImageEncoder(model_id_or_local_path=self.model_id_or_local_path)

    # ------------------------------------------------------------------
    # Position ID preparation
    # ------------------------------------------------------------------

    def _prepare_latent_image_ids(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare image position IDs for the Flux.2 transformer.

        Flux.2 uses 4-axis RoPE with dims (32, 32, 32, 32) corresponding to
        axes ``(T, H, W, L)`` -- time, height, width, and a sequence-length
        axis used for text tokens. For text-to-image we only populate axes 1
        and 2 (the spatial H and W). This matches diffusers' ``_prepare_latent_ids``
        in ``pipelines/flux2/pipeline_flux2.py`` -- the pretrained transformer
        weights are tied to this exact axis assignment.

        Args:
            height: Latent height (before 2x2 packing).
            width: Latent width (before 2x2 packing).
            device: Target device.
            dtype: Target dtype.

        Returns:
            torch.Tensor: Image position IDs ``[(H//2)*(W//2), 4]``.
        """
        packed_height = height // 2
        packed_width = width // 2
        latent_image_ids = torch.zeros(packed_height, packed_width, 4, device=device, dtype=dtype)
        # Axis 1 = H, axis 2 = W (NOT axes 2/3 -- that puts spatial info into
        # the L sub-band of RoPE and silently destroys image structure).
        latent_image_ids[..., 1] = torch.arange(packed_height, device=device, dtype=dtype)[:, None]
        latent_image_ids[..., 2] = torch.arange(packed_width, device=device, dtype=dtype)[None, :]
        return latent_image_ids.reshape(packed_height * packed_width, 4)

    def _prepare_reference_image_ids(
        self,
        ref_shapes: List[Tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare position IDs for one or more reference (edit-source) images.

        Each reference image gets a unique non-zero ``T``-axis offset so the
        transformer can attend to it as a distinct "frame" without confusing
        its tokens with the noisy target image (whose ``T=0``). This mirrors
        diffusers' ``Flux2Pipeline._prepare_image_ids``:

            T_i  = ref_image_t_scale * (i + 1)            # i-th reference
            H_i  = packed-row index in [0, H_i//2)
            W_i  = packed-col index in [0, W_i//2)
            L_i  = 0

        Args:
            ref_shapes: List of ``(height, width)`` tuples for each
                reference latent (pre-pack, i.e. raw VAE-output spatial dims).
            device: Target device.
            dtype: Target dtype.

        Returns:
            Concatenated reference image position IDs ``[sum_i (H_i//2)*(W_i//2), 4]``.
        """
        all_ids: List[torch.Tensor] = []
        for i, (h, w) in enumerate(ref_shapes):
            ph, pw = h // 2, w // 2
            ids = torch.zeros(ph, pw, 4, device=device, dtype=dtype)
            ids[..., 0] = float(self.ref_image_t_scale * (i + 1))  # T offset
            ids[..., 1] = torch.arange(ph, device=device, dtype=dtype)[:, None]
            ids[..., 2] = torch.arange(pw, device=device, dtype=dtype)[None, :]
            all_ids.append(ids.reshape(ph * pw, 4))
        if not all_ids:
            return torch.zeros(0, 4, device=device, dtype=dtype)
        return torch.cat(all_ids, dim=0)

    def _prepare_text_ids(
        self,
        seq_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare text position IDs.

        Each text token gets a unique position along the L axis (axis 3).
        This matches diffusers' ``_prepare_text_ids`` in
        ``pipelines/flux2/pipeline_flux2.py`` (cartesian_prod with
        ``l = arange(L)`` and ``t = h = w = [0]``).

        Returning all-zeros makes every text token share the same RoPE
        position, which collapses the text positional information and
        silently corrupts conditioning.
        """
        ids = torch.zeros(seq_length, 4, device=device, dtype=dtype)
        ids[:, 3] = torch.arange(seq_length, device=device, dtype=dtype)
        return ids

    # ------------------------------------------------------------------
    # Latent packing / unpacking
    # ------------------------------------------------------------------

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack latents from [B, C, H, W] to [B, (H//2)*(W//2), C*4].

        Flux.2 uses the same 2x2 patch packing as Flux.1, but with C=32
        latent channels from AutoencoderKLFlux2, producing C*4=128 packed
        channels (matching transformer in_channels=128).

        Args:
            latents: Input latents [B, C, H, W].

        Returns:
            Packed latents [B, (H//2)*(W//2), C*4].
        """
        batch_size, channels, height, width = latents.shape
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)
        return latents

    def _unpack_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Unpack latents from [B, (H//2)*(W//2), C*4] to [B, C, H, W].

        Args:
            latents: Packed latents [B, (H//2)*(W//2), C*4].
            height: Target height (original H before packing).
            width: Target width (original W before packing).

        Returns:
            Unpacked latents [B, C, H, W].
        """
        batch_size = latents.shape[0]
        channels = latents.shape[2] // 4
        latents = latents.reshape(batch_size, height // 2, width // 2, channels, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels, height, width)
        return latents

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    def build_conditioning(
        self,
        data: dict,
        input_shape: Optional[Union[Sequence[int], Mapping[str, Optional[Sequence[int]]]]] = None,
        precision: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Resolve text and build the optional Flux.2 image-edit condition.

        The base builder preserves encoded text tuples or encodes missing
        ``condition_raw`` / ``neg_condition_raw`` branches. For plain T2I those
        resolved branches are returned directly.

        For image-editing (``is_image_edit=True``), ``condition_image`` is an
        authoritative precomputed reference latent (or a list of them). If it is
        absent, VAE-encode the pixel-space ``condition_image_raw`` (a tensor or a
        list) in deterministic ``mode="argmax"``, matching diffusers' Flux.2
        pipelines. The resulting latents are bundled, in ``precision_model``, with
        the text embeddings and prepended to the noisy target by the forward pass.
        The negative branch reuses the same reference images.

        Complete rich image-edit conditions are outputs of this method and are
        not accepted as re-entrant inputs. ``"neg_condition"`` is omitted when
        the resolved negative text branch is absent; a present branch
        receives the same reference image as the positive.

        Raises:
            ValueError: in image-edit mode when neither ``condition_image`` nor
                ``condition_image_raw`` is available.
        """
        out = super().build_conditioning(data, input_shape=input_shape, precision=precision)
        required_fields = ("condition",) if self.is_image_edit else ()
        validate_conditioning(out, required_fields=required_fields)
        text_embeds = out.get("condition")
        if not self.is_image_edit:
            return out

        condition_image = data.get("condition_image")
        if condition_image is None:
            condition_image_raw = data.get("condition_image_raw")
            if condition_image_raw is None:
                raise ValueError(
                    f"{type(self).__name__} is in image-edit mode (enable_image_edit=True) and requires "
                    "a precomputed data['condition_image'] latent or pixel-space "
                    "data['condition_image_raw']. Supply one source image per prompt "
                    "(e.g. --input_image_file at inference / the image-edit key_map in training), "
                    "or use a plain-T2I config (enable_image_edit=False)."
                )
            vae = self.get_preprocessor("vae", "encode the reference image")
            # One reference image or a list of them (multi-reference editing), as ``forward`` takes them.
            condition_image = tmap(lambda raw: vae.encode(raw, mode="argmax"), condition_image_raw)
            assert_same(device=True, condition_image_raw=condition_image_raw, condition_image=condition_image)
        condition_image = to(condition_image, dtype=self.precision_model)
        out["condition"] = {
            "text_embeds": text_embeds,
            "condition_image": condition_image,
        }
        if out.get("neg_condition") is not None:
            out["neg_condition"] = {
                "text_embeds": out["neg_condition"],
                "condition_image": condition_image,
            }
        return out

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        guidance: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of Flux.2 diffusion model.

        Args:
            x_t: The diffused data sample [B, C, H, W]. C=32 for Flux.2 VAE latents.
            t: The current timestep in [0, 1].
            condition: Tuple of ``(pooled_prompt_embeds, prompt_embeds)`` from
                the text encoder. ``pooled_prompt_embeds`` is always ``None``
                for Flux.2 (no pooled CLIP embedding exists; the slot is kept
                for API compatibility with FastGen's other networks). The
                Flux.2 transformer only consumes ``prompt_embeds``.
            guidance: Optional guidance scale override. If None, uses self.guidance_scale.
            return_features_early: If True, return features once collected.
            feature_indices: Set of block indices for feature extraction.
            return_logvar: If True, return the logvar.
            fwd_pred_type: Override network prediction type.
            apply_output_conditioning: Apply the output constraint after the conversion.
            output_dtype: ``"same"`` rounds the prediction to the state dtype; ``None`` keeps it in high precision.
        Returns:
            Model output tensor or tuple with logvar/features.
        """
        self._validate_forward_kwargs(
            fwd_kwargs,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            num_features=len(self.transformer.transformer_blocks) + len(self.transformer.single_transformer_blocks),
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        x_t_model = self.cast_to_model_precision(x_t)
        batch_size = x_t_model.shape[0]
        height, width = x_t_model.shape[2], x_t_model.shape[3]
        n_target_tokens = (height // 2) * (width // 2)

        # Image-edit / multi-reference branch: ``condition`` may be a dict
        # bundling text embeddings and a pre-VAE-encoded reference latent (or
        # a list of reference latents) under the standard FastGen contract
        # used by other multi-modal networks (Wan I2V, Cosmos video2world).
        condition_image: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
        if isinstance(condition, dict):
            text_embeds = condition.get("text_embeds")
            condition_image = condition.get("condition_image", None)
            if text_embeds is None:
                raise ValueError("Flux2 condition dict is missing the required 'text_embeds' key.")
            condition = text_embeds

        # Unpack condition. The pooled slot is intentionally discarded -- it's
        # always None for Flux.2 (see `Flux2TextEncoder.encode`); the
        # transformer's `encoder_hidden_states` is the per-token sequence only.
        _pooled_prompt_embeds, prompt_embeds = condition
        assert _pooled_prompt_embeds is None, "Flux2 does not consume pooled prompt embeddings"
        assert_same(dtype=self.precision_model, device=x_t_model.device, prompt_embeds=prompt_embeds)
        if guidance is None and self.guidance_scale is not None:
            guidance = torch.full((batch_size,), self.guidance_scale, device=x_t_model.device, dtype=x_t_model.dtype)
        assert_same(shape=True, device=True, t=t, guidance=guidance)
        # Prepare 4-axis Flux.2 position IDs directly in float32. BFL's reference constructs int64 IDs,
        # which Diffusers converts to float32 before RoPE; both paths are exact for these coordinate ranges.
        img_ids = self._prepare_latent_image_ids(height, width, x_t_model.device, torch.float32)
        txt_ids = self._prepare_text_ids(prompt_embeds.shape[1], x_t_model.device, torch.float32)

        # Pack target latents: [B, 32, H, W] -> [B, (H//2)*(W//2), 128]
        hidden_states = self._pack_latents(x_t_model)

        # Concatenate reference (edit-source) latents along the sequence
        # dimension, with non-zero T-axis RoPE positions per reference. This
        # matches diffusers' ``Flux2Pipeline.__call__``: the transformer
        # consumes (target || ref_0 || ref_1 || ...) tokens jointly, then we
        # slice the output back to the target slot.
        if condition_image is not None:
            ref_list = condition_image if isinstance(condition_image, (list, tuple)) else [condition_image]
            ref_shapes: List[Tuple[int, int]] = []
            packed_refs: List[torch.Tensor] = []
            for ref in ref_list:
                if ref.ndim != 4:
                    raise ValueError(f"Each reference image latent must be [B, C, H, W]; got shape {tuple(ref.shape)}.")
                if ref.shape[0] != batch_size:
                    raise ValueError(f"Reference image batch ({ref.shape[0]}) must match noisy target ({batch_size}).")
                assert_same(dtype=self.precision_model, device=x_t_model.device, condition_image=ref)
                ref_shapes.append((ref.shape[2], ref.shape[3]))
                packed_refs.append(self._pack_latents(ref.to(dtype=x_t_model.dtype)))
            ref_ids = self._prepare_reference_image_ids(ref_shapes, img_ids.device, img_ids.dtype)

            hidden_states = torch.cat([hidden_states, *packed_refs], dim=1)
            img_ids = torch.cat([img_ids, ref_ids], dim=0)

        model_outputs = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=prompt_embeds,
            timestep=t,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            target_grid=(height // 2, width // 2),
        )

        if return_features_early:
            return model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        # Drop reference-image tokens from the prediction (they were only
        # there to condition the attention; we never denoise them).
        def _slice_to_target(tokens: torch.Tensor) -> torch.Tensor:
            return tokens[:, :n_target_tokens, ...] if tokens.shape[1] != n_target_tokens else tokens

        # Unpack output: [B, seq_len, C*4] -> [B, C, H, W]
        if isinstance(out, torch.Tensor):
            out = _slice_to_target(out)
            out = self._unpack_latents(out, height, width)
            out = self.transform_output(out, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning)
        else:
            out[0] = _slice_to_target(out[0])
            out[0] = self._unpack_latents(out[0], height, width)
            out[0] = self.transform_output(
                out[0], x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
            )

        if return_logvar:
            return out, logvar.to(dtype=x_t.dtype)
        return out

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _calculate_shift(self, image_seq_len: int, num_steps: int = 28) -> float:
        """Resolution- AND step-count-dependent shift for the FM scheduler.

        Flux.2 uses an empirical shift that depends on BOTH the packed token
        count and the number of sampling steps -- not the linear formula that
        Flux.1 uses. Coefficients lifted from BFL's official `flux2.sampling`
        and mirrored in
        ``diffusers.pipelines.flux2.pipeline_flux2.compute_empirical_mu``.

        For a 1024x1024 image (image_seq_len=4096) with num_steps=28 this
        returns mu ~= 2.15, vs. the Flux.1 formula's 1.15 -- ~2x off. Using
        the wrong shift quietly miscalibrates which timesteps the model is
        queried at and degrades sample quality.
        """
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        return float(a * num_steps + b)

    @staticmethod
    def default_guidance_cfg() -> GuidanceConfig:
        """Default traditional-CFG recipe at scale 4.0 (klein-base only).

        Consulted by ``sample()`` only when ``self.guidance_scale is None``
        (the CFG branch, used by the undistilled klein-base variants).
        """
        g = GuidanceConfig()
        g.cfg.scale = 4.0
        return g

    def sample_scheduler_and_grid(
        self, sample_steps: int, image_seq_len: int, device: Optional[torch.device] = None
    ) -> Tuple[FlowMatchEulerDiscreteScheduler, torch.Tensor]:
        """Return a new flow-matching Euler scheduler on BFL's ``flux2.sampling.get_schedule`` grid.

        That is ``linspace(1, 0, steps + 1)`` under the exponential shift, with ``mu`` from the Flux.2 empirical formula
        in ``image_seq_len`` (the token count after 2x2 packing) and ``sample_steps``. diffusers applies this shift
        only with ``use_dynamic_shifting`` on an explicit grid; ``shift=mu`` would shift multiplicatively instead.

        Returns:
            The scheduler, set to begin at the first step, and the times to query the network at, one per step: its
            sigmas in ``t_precision``.
        """
        mu = self._calculate_shift(image_seq_len, num_steps=sample_steps)
        scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
        scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / sample_steps, sample_steps), mu=mu, device=device)
        # Start at step 0: high shifts repeat the first timestep, and the lookup (a device sync) takes the second match.
        scheduler.set_begin_index(0)
        t_grid = self.noise_scheduler.safe_clamp(
            scheduler.sigmas[:-1].to(device=device, dtype=self.noise_scheduler.t_precision),
            min=self.noise_scheduler.min_t,
            max=self.noise_scheduler.max_t,
            warn=True,
        )
        return scheduler, t_grid

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        neg_condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sample_steps: int = 28,
        sample_type: str = "ode",
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Generate samples using Euler flow matching.

        Two internal modes (selected by ``self.guidance_scale``):

        1. **Embedded Guidance** (``self.guidance_scale`` is set,
           default for Flux.2-dev): a single forward whose embedded
           distillation-guidance scalar is filled by ``self.forward``
           from ``self.guidance_scale`` (set via
           ``config.model.net.guidance_scale``). ``neg_condition``
           and ``guidance_cfg`` are not consulted.
        2. **CFG mode** (``self.guidance_scale is None``, klein-base):
           uses ``predict_with_guidance`` to fuse the cond +
           cfg-uncond pair into one batched forward.

        Args:
            noise: Initial noise tensor [B, 32, H, W].
            condition: Tuple of (pooled_prompt_embeds, prompt_embeds), or an
                image-edit dict bundling that tuple under ``"text_embeds"``
                together with a reference latent under ``"condition_image"``.
            neg_condition: Negative condition for CFG mode (klein-base);
                ignored in embedded guidance mode.
            sample_steps: Number of sampling steps; defaults to 28, BFL's suggested
                trade-off for dev (BFL/diffusers default to 50). 4 for distilled
                klein, 50 for klein-base-{4B,9B}.
            sample_type: Integration policy; only ``"ode"`` is supported.
            guidance_cfg: Composite guidance config consulted only in CFG
                mode; ``None`` → ``default_guidance_cfg()`` (CFG scale 4.0).

        Returns:
            Generated latent samples [B, 32, H, W].
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )
        batch_size, channels, height, width = noise.shape

        # After 2x2 packing: seq_len = (H // 2) * (W // 2)
        image_seq_len = (height // 2) * (width // 2)

        scheduler, t_grid = self.sample_scheduler_and_grid(sample_steps, image_seq_len, device=noise.device)

        # Scale initial noise
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])

        # Resolve the CFG config once up front; loop-invariant. Only consulted
        # on the CFG branch (klein-base), so skip resolution entirely under
        # distillation.
        cfg = (
            (guidance_cfg if guidance_cfg is not None else self.default_guidance_cfg())
            if self.guidance_scale is None
            else None
        )

        # Sampling loop
        for sched_step, t in zip(scheduler.timesteps, t_grid, strict=True):
            t = t.expand(batch_size)

            if self.guidance_scale is None:
                # CFG mode (klein-base) — fuse cond + uncond into one
                # batched forward via predict_with_guidance.
                noise_pred = predict_with_guidance(
                    self,
                    latents,
                    t,
                    batched=True,
                    guidance_cfg=cfg,
                    condition=condition,
                    neg_condition=neg_condition,
                    fwd_pred_type="flow",
                )
            else:
                # Guidance distillation mode (Flux.2-dev). forward() fills
                # the embedded guidance tensor from self.guidance_scale.
                noise_pred = self(
                    latents,
                    t,
                    condition,
                    fwd_pred_type="flow",
                )

            # Euler step
            latents = scheduler.step(noise_pred, sched_step, latents, return_dict=False)[0]

        return latents
