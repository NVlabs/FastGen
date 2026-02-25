# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import types

import numpy as np
import torch
import torch.utils.checkpoint
from torch import dtype
from torch.distributed.fsdp import fully_shard

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKLQwenImage
from diffusers.models import QwenImageTransformer2DModel
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, PretrainedConfig

from fastgen.networks.network import FastGenNetwork

from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils.basic_utils import str2bool
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger


# Workaround for transformers<4.50 bug: Qwen2.5-VL config has sub-configs as plain dicts
# but GenerationConfig.from_model_config() expects .to_dict() on them.
_sub_config_names = ("decoder", "generator", "text_config", "text_encoder")
_orig_get_text_config = PretrainedConfig.get_text_config


def _patched_get_text_config(self, decoder=False):
    for name in _sub_config_names:
        val = getattr(self, name, None)
        if isinstance(val, dict):
            setattr(self, name, PretrainedConfig(**val))
    return _orig_get_text_config(self, decoder=decoder)


PretrainedConfig.get_text_config = _patched_get_text_config


class QwenImageTextEncoder:
    """Text encoder for Qwen-Image using Qwen2.5-VL-7B-Instruct.

    Uses a single VLM (Qwen2.5-VL-7B) to encode text prompts into embeddings
    with a prompt template. Returns (prompt_embeds, prompt_embeds_mask).
    """

    PROMPT_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    PROMPT_TEMPLATE_DROP_IDX = 34

    def __init__(self, model_id: str, torch_dtype: Optional[dtype] = torch.bfloat16):
        """
        Args:
            model_id: The HuggingFace model ID to load.
            torch_dtype: The data type for the output embeddings. Default to bfloat16 for Qwen image text encoder.
            Note that setting it to float32 will produce embeddings with large
            differences compared to bfloat16, leading to poor performance.
        """
        local_files_only = str2bool(os.getenv("LOCAL_FILES_ONLY", "false"))
        cache_dir = os.environ["HF_HOME"]

        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            subfolder="text_encoder",
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        self.text_encoder.eval().requires_grad_(False)
        self.max_sequence_length = 1024

    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> list:
        """Extract non-padded hidden states per sample using attention mask."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return list(torch.split(selected, valid_lengths.tolist(), dim=0))

    def encode(
        self,
        conditioning: Optional[Any] = None,
        precision: dtype = torch.float32,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts to embeddings.

        Args:
            conditioning: Text prompt(s) to encode.
            precision: Data type for the output embeddings.
            max_sequence_length: Maximum sequence length for tokenization.

        Returns:
            Tuple of (prompt_embeds, prompt_embeds_mask) tensors.
            - prompt_embeds: [B, seq_len, 3584]
            - prompt_embeds_mask: [B, seq_len] long tensor
        """
        if isinstance(conditioning, str):
            conditioning = [conditioning]

        drop_idx = self.PROMPT_TEMPLATE_DROP_IDX
        txt = [self.PROMPT_TEMPLATE.format(p) for p in conditioning]

        txt_tokens = self.tokenizer(
            txt,
            max_length=max_sequence_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.text_encoder.device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=txt_tokens.input_ids,
                attention_mask=txt_tokens.attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]

        max_seq_len = max(e.size(0) for e in split_hidden_states)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        # Truncate to max_sequence_length
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        encoder_attention_mask = encoder_attention_mask[:, :max_sequence_length]

        return prompt_embeds.to(dtype=precision), encoder_attention_mask

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        self.text_encoder.to(*args, **kwargs)
        return self


class QwenImageImageEncoder:
    """VAE encoder/decoder for Qwen-Image (Wan 2.1 based).

    Uses per-channel latents_mean/latents_std normalization instead of
    Flux's scalar shift_factor/scaling_factor. The VAE operates on 5D
    tensors [B, C, T, H, W] so we add/remove the temporal dimension.
    """

    def __init__(self, model_id: str):
        self.vae: AutoencoderKLQwenImage = AutoencoderKLQwenImage.from_pretrained(
            model_id,
            cache_dir=os.environ["HF_HOME"],
            subfolder="vae",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.vae.eval().requires_grad_(False)

        # Per-channel normalization vectors (16 elements each)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1)
        self.latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1)

    def encode(self, real_images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space.

        Args:
            real_images: Input images in [-1, 1] range, shape [B, C, H, W].

        Returns:
            torch.Tensor: Normalized latent representations [B, 16, H//8, W//8].
        """
        # VAE expects 5D: [B, C, T, H, W] with T=1 for images
        x = real_images.unsqueeze(2)
        latents = self.vae.encode(x, return_dict=False)[0].sample()
        # Remove temporal dim: [B, 16, 1, H', W'] -> [B, 16, H', W']
        latents = latents.squeeze(2)
        # Normalize per-channel: (latents - mean) / std
        mean = self.latents_mean.to(latents.device, latents.dtype)
        std = self.latents_std.to(latents.device, latents.dtype)
        latents = (latents - mean) / std
        return latents

    def decode(self, latent_images: torch.Tensor) -> torch.Tensor:
        """Decode latents to images.

        Args:
            latent_images: Normalized latent representations [B, 16, H', W'].

        Returns:
            torch.Tensor: Decoded images in [-1, 1] range, shape [B, C, H, W].
        """
        # De-normalize per-channel: latents * std + mean
        mean = self.latents_mean.to(latent_images.device, latent_images.dtype)
        std = self.latents_std.to(latent_images.device, latent_images.dtype)
        latents = latent_images * std + mean
        # Add temporal dim: [B, C, H', W'] -> [B, C, 1, H', W']
        latents = latents.unsqueeze(2)
        images = self.vae.decode(latents, return_dict=False)[0]
        # Remove temporal dim: [B, C, T, H, W] -> [B, C, H, W]
        images = images[:, :, 0]
        return images.clip(-1.0, 1.0)

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        self.vae.to(*args, **kwargs)
        return self


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List] = None,
    txt_seq_lens: Optional[List[int]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_features_early: bool = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Modified forward pass for QwenImageTransformer2DModel with feature extraction support.

    Replicates the exact forward flow of QwenImageTransformer2DModel but adds:
    - Feature extraction at specified block indices (for discriminator training)
    - Early return once all features are collected
    - Optional logvar estimation

    Args:
        hidden_states: Input packed latent states [B, seq_len, in_channels].
        encoder_hidden_states: Text encoder hidden states [B, text_seq_len, 3584].
        encoder_hidden_states_mask: Text attention mask [B, text_seq_len].
        timestep: Current timestep in [0, 1] range.
        img_shapes: Image spatial shapes for RoPE, e.g. [[(1, H, W)]].
        txt_seq_lens: Text sequence lengths per batch item.
        attention_kwargs: Additional attention kwargs.
        return_features_early: If True, return features as soon as collected.
        feature_indices: Set of block indices to extract features from.
        return_logvar: If True, return log variance estimate.

    Returns:
        Model output, optionally with features or logvar.
    """
    if feature_indices is None:
        feature_indices = set()

    if return_features_early and len(feature_indices) == 0:
        return []

    features = []

    # Store spatial size for feature reshaping
    # hidden_states: [B, seq_len, C*4] where seq_len = (H//2) * (W//2)
    seq_len = hidden_states.shape[1]
    spatial_size = (
        (img_shapes[0][0][1], img_shapes[0][0][2]) if img_shapes is not None else (int(seq_len**0.5), int(seq_len**0.5))
    )

    # 1. Patch embedding
    hidden_states = self.img_in(hidden_states)

    # 2. Text processing
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    # 3. Time embedding (QwenTimestepProjEmbeddings takes timestep and hidden_states for dtype)
    temb = self.time_text_embed(timestep, hidden_states)

    # 4. RoPE (handled internally by QwenEmbedRope)
    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

    # 5. Transformer blocks (single list of 60 dual-stream blocks)
    for idx, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                image_rotary_emb,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )

        # Extract features at specified block indices
        if idx in feature_indices:
            feat = hidden_states.clone()
            B, S, C = feat.shape
            feat = feat.permute(0, 2, 1).reshape(B, C, spatial_size[0], spatial_size[1])
            features.append(feat)

        if return_features_early and len(features) == len(feature_indices):
            return features

    # 6. Final projection
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if return_features_early:
        assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
        return features

    # Prepare output
    if len(feature_indices) == 0:
        out = output
    else:
        out = [output, features]

    if return_logvar:
        logvar = self.logvar_linear(temb)
        return out, logvar

    return out


class QwenImage(FastGenNetwork):
    """Qwen-Image network for text-to-image generation.

    A ~20B parameter MMDiT model using flow matching with true classifier-free guidance.
    Uses Qwen2.5-VL-7B as text encoder and a Wan2.1-based VAE.

    Reference: https://huggingface.co/Qwen/Qwen-Image
    """

    MODEL_ID = "Qwen/Qwen-Image"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        disable_grad_ckpt: bool = False,
        load_pretrained: bool = True,
        **model_kwargs,
    ):
        """QwenImage constructor.

        Args:
            model_id: The HuggingFace model ID to load. Defaults to "Qwen/Qwen-Image".
            net_pred_type: Prediction type. Defaults to "flow" for flow matching.
            schedule_type: Schedule type. Defaults to "rf" (rectified flow).
            disable_grad_ckpt: Whether to disable gradient checkpointing during training.
                Defaults to False. Set to True when using FSDP to avoid memory access errors.
            load_pretrained: Whether to load pretrained weights.
        """
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        self.model_id = model_id
        self._disable_grad_ckpt = disable_grad_ckpt

        self._initialize_network(model_id, load_pretrained)

        # Override forward with classify_forward for feature extraction support
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Gradient checkpointing configuration
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    def _initialize_network(self, model_id: str, load_pretrained: bool) -> None:
        """Initialize the transformer network.

        Args:
            model_id: The HuggingFace model ID or local path.
            load_pretrained: Whether to load pretrained weights.
        """
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        local_files_only = str2bool(os.getenv("LOCAL_FILES_ONLY", "false"))

        if should_load_weights:
            logger.info("Loading QwenImage transformer from pretrained")
            self.transformer: QwenImageTransformer2DModel = QwenImageTransformer2DModel.from_pretrained(
                model_id,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=local_files_only,
            )
        else:
            config = QwenImageTransformer2DModel.load_config(
                model_id,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=local_files_only,
            )
            if in_meta_context:
                logger.info(
                    "Initializing QwenImage transformer on meta device "
                    "(zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing QwenImage transformer from config (no pretrained weights)")
                logger.warning("QwenImage transformer being initialized from config. No weights are loaded!")
            self.transformer: QwenImageTransformer2DModel = QwenImageTransformer2DModel.from_config(config)

        # Add logvar linear layer for variance estimation - QwenImage uses 3072-dim time embeddings
        self.transformer.logvar_linear = torch.nn.Linear(3072, 1)

    def reset_parameters(self):
        """Reinitialize parameters for FSDP meta device initialization."""

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)

        # Reinitialize QwenEmbedRope tensors — these are plain tensors (not nn.Parameters)
        # that end up on meta device during FSDP meta init and don't get materialized by FSDP.
        for m in self.modules():
            if isinstance(m, QwenEmbedRope):
                pos_index = torch.arange(4096)
                neg_index = torch.arange(4096).flip(0) * -1 - 1
                m.pos_freqs = torch.cat(
                    [
                        m.rope_params(pos_index, m.axes_dim[0], m.theta),
                        m.rope_params(pos_index, m.axes_dim[1], m.theta),
                        m.rope_params(pos_index, m.axes_dim[2], m.theta),
                    ],
                    dim=1,
                )
                m.neg_freqs = torch.cat(
                    [
                        m.rope_params(neg_index, m.axes_dim[0], m.theta),
                        m.rope_params(neg_index, m.axes_dim[1], m.theta),
                        m.rope_params(neg_index, m.axes_dim[2], m.theta),
                    ],
                    dim=1,
                )
                m.rope_cache = {}
                logger.debug("Reinitialized QwenEmbedRope pos_freqs/neg_freqs from meta device")

        super().reset_parameters()

        logger.debug("Reinitialized QwenImage parameters")

    def fully_shard(self, **kwargs):
        """Fully shard the QwenImage network for FSDP.

        QwenImage has a single list of 60 dual-stream QwenImageTransformerBlock instances.

        We shard `self.transformer` instead of `self` because the network wrapper
        class may have complex multiple inheritance with ABC, which causes Python's
        __class__ assignment to fail due to incompatible memory layouts.
        """
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(
                self.transformer,
                check_fn=lambda block: isinstance(block, QwenImageTransformerBlock),
            )
            logger.info("Applied FSDP activation checkpointing to QwenImage transformer blocks")

        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)

        fully_shard(self.transformer, **kwargs)

    def init_preprocessors(self):
        """Initialize text and image encoders."""
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self, torch_dtype: Optional[dtype] = torch.bfloat16):
        """Initialize the text encoder for QwenImage."""
        self.text_encoder = QwenImageTextEncoder(model_id=self.model_id, torch_dtype=torch_dtype)

    def init_vae(self):
        """Initialize only the VAE for visualization."""
        self.vae = QwenImageImageEncoder(model_id=self.model_id)

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        return self

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack latents from [B, C, H, W] to [B, (H//2)*(W//2), C*4] for transformer.

        Uses 2x2 patch packing where each 2x2 spatial block is flattened into channels.

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

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of QwenImage diffusion model.

        Args:
            x_t: The diffused data sample [B, C, H, W].
            t: The current timestep in [0, 1] range.
            condition: Tuple of (encoder_hidden_states, encoder_hidden_states_mask).
            r: Another timestep (for mean flow methods), unused.
            return_features_early: If True, return features once collected.
            feature_indices: Set of block indices for feature extraction.
            return_logvar: If True, return the logvar.
            fwd_pred_type: Override network prediction type.

        Returns:
            Model output tensor or tuple with logvar/features.
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
        height, width = x_t.shape[2], x_t.shape[3]

        # Unpack condition: (encoder_hidden_states, encoder_hidden_states_mask)
        encoder_hidden_states, encoder_hidden_states_mask = condition

        # Pack latents for transformer: [B, C, H, W] -> [B, (H//2)*(W//2), C*4]
        hidden_states = self._pack_latents(x_t)

        # Prepare img_shapes: [[(1, latent_H_packed, latent_W_packed)]] per batch item
        img_shapes = [[(1, height // 2, width // 2)]] * batch_size

        # Prepare txt_seq_lens from attention mask (must be ints for slice indices in pos_embed)
        txt_seq_lens = encoder_hidden_states_mask.sum(dim=1).int().tolist()

        model_outputs = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=t,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
        )

        if return_features_early:
            return model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        # Unpack output: [B, seq_len, C] -> [B, C, H, W]
        if isinstance(out, torch.Tensor):
            out = self._unpack_latents(out, height, width)
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            out[0] = self._unpack_latents(out[0], height, width)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )

        if return_logvar:
            return out, logvar
        return out

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        neg_condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 4.0,
        num_steps: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using Euler flow matching with classifier-free guidance.

        Args:
            noise: Initial noise tensor [B, C, H, W].
            condition: Tuple of (encoder_hidden_states, encoder_hidden_states_mask).
            neg_condition: Optional negative condition tuple for CFG.
            guidance_scale: CFG scale. When > 1.0 and neg_condition is provided,
                enables classifier-free guidance with double forward passes.
            num_steps: Number of sampling steps (default 50).

        Returns:
            Generated latent samples.
        """
        batch_size, _, height, width = noise.shape

        if not hasattr(self, "scheduler"):
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.model_id,
                subfolder="scheduler",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
        scheduler = self.scheduler

        # Calculate resolution-dependent shift (mu) from scheduler config
        image_seq_len = (height // 2) * (width // 2)
        cfg = scheduler.config
        mu = cfg.base_shift + (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len - cfg.base_image_seq_len) * (
            image_seq_len - cfg.base_image_seq_len
        )

        # Set timesteps with sigmas (matching pipeline pattern)
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        scheduler.set_timesteps(sigmas=sigmas, device=noise.device, mu=mu)
        timesteps = scheduler.timesteps

        # Initialize latents with proper scaling based on the initial timestep
        t_init = self.noise_scheduler.safe_clamp(
            timesteps[0] / 1000.0, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t
        )
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        do_cfg = guidance_scale is not None and guidance_scale > 1.0 and neg_condition is not None

        # Sampling loop
        scheduler.set_begin_index(0)
        for timestep in timesteps:
            # Pipeline passes timestep/1000 directly to transformer — no clamping
            t = (timestep / 1000.0).expand(batch_size).to(latents.dtype)

            # Conditional prediction
            pred = self(latents, t, condition, fwd_pred_type="flow")

            # CFG: separate unconditional prediction + combination
            if do_cfg:
                neg_pred = self(latents, t, neg_condition, fwd_pred_type="flow")
                comb_pred = neg_pred + guidance_scale * (pred - neg_pred)
                # Norm-preserving rescaling in packed token space (matching pipeline behavior).
                # Pipeline operates on [B, seq, 64] with dim=-1 (per-token norm).
                # We have [B, C, H, W], so pack to [B, seq, C*4], rescale, then unpack.
                pred_packed = self._pack_latents(pred)
                comb_packed = self._pack_latents(comb_pred)
                cond_norm = torch.norm(pred_packed, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb_packed, dim=-1, keepdim=True)
                pred = self._unpack_latents(comb_packed * (cond_norm / (comb_norm + 1e-8)), height, width)

            # Euler step
            latents = scheduler.step(pred, timestep, latents, return_dict=False)[0]

        return latents
