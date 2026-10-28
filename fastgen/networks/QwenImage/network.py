# SPDX-FileCopyrightText: Copyright (c) 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import types

import numpy as np
import torch
import torch.utils.checkpoint
from torch.distributed.fsdp import fully_shard

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKLQwenImage
from diffusers.models import QwenImageTransformer2DModel
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope
from diffusers.models.embeddings import TimestepEmbedding
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from fastgen.networks.network import FastGenNetwork, FastGenModule
from fastgen.networks.lora_utils import LoRAMixin

from fastgen.utils.basic_utils import attr_override
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
from fastgen.utils.guidance import (
    GuidanceConfig,
    apply_cfg,
    apply_mix,
    apply_rescale,
    GuidanceContext,
    predict_with_guidance,
)
from fastgen.utils.env import hf_load
from fastgen.utils.tree_utils import OutputDtype, assert_same
import fastgen.utils.logging_utils as logger


class QwenImageTextEncoder(FastGenModule):
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

    def __init__(self, model_id_or_local_path: str, dtype: Optional[torch.dtype] = torch.bfloat16):
        """
        Args:
            model_id_or_local_path: The HuggingFace model ID or local path to load.
            dtype: Data type used to load the text encoder weights. Defaults to
                bfloat16; loading them in float32 produces embeddings with large
                differences from the bfloat16 path, leading to poor performance.
        """
        super().__init__()

        self.tokenizer = hf_load(
            Qwen2Tokenizer.from_pretrained,
            model_id_or_local_path,
            subfolder="tokenizer",
        )
        self.text_encoder = hf_load(
            Qwen2_5_VLForConditionalGeneration.from_pretrained,
            model_id_or_local_path,
            subfolder="text_encoder",
            dtype=dtype,
        )

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
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts to embeddings.

        Args:
            conditioning: Text prompt(s) to encode.
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
        )
        return self(
            txt_tokens.input_ids.to(self.device),
            txt_tokens.attention_mask.to(self.device),
            max_sequence_length=max_sequence_length,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        drop_idx = self.PROMPT_TEMPLATE_DROP_IDX
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            logits_to_keep=1,  # only hidden_states is used; skips the full-vocabulary lm_head projection
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
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

        return prompt_embeds, encoder_attention_mask


class QwenImageImageEncoder(FastGenModule):
    """VAE encoder/decoder for Qwen-Image (Wan 2.1 based).

    Uses per-channel latents_mean/latents_std normalization instead of
    Flux's scalar shift_factor/scaling_factor. The VAE operates on 5D
    tensors [B, C, T, H, W] so we add/remove the temporal dimension.

    The official Qwen-Image repository does not contain a separate native VAE
    implementation: its executable reference is the Diffusers pipeline, loaded as a
    whole in BF16 on CUDA by the official examples. That pipeline applies these
    statistics outside ``AutoencoderKLQwenImage`` in the VAE/result dtype and casts
    denoised state to ``vae.dtype`` before the inverse. This wrapper retains that
    boundary ordering and returns results in the VAE dtype.
    """

    def __init__(self, model_id_or_local_path: str):
        super().__init__()
        self.vae: AutoencoderKLQwenImage = hf_load(
            AutoencoderKLQwenImage.from_pretrained, model_id_or_local_path, subfolder="vae"
        )

        # Keep auxiliary constants as Python values so ambient meta construction
        # cannot strand undeclared tensors that preparation cannot materialize.
        self.latents_mean = tuple(self.vae.config.latents_mean)
        self.latents_std = tuple(self.vae.config.latents_std)

    @property
    def latent_compression_factors(self) -> tuple[int, int]:
        """Pixel-to-latent ``(H, W)`` downsampling of the VAE: every downsampling stage halves the resolution."""
        factor = 2 ** len(self.vae.config.temperal_downsample)
        return (factor, factor)

    def encode(self, real_images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space.

        Args:
            real_images: Input images in [-1, 1] range, shape [B, C, H, W].

        Returns:
            torch.Tensor: Normalized latent representations [B, 16, H//8, W//8].
        """
        return self(real_images, operation="encode")

    def forward(self, data: torch.Tensor, operation: str) -> torch.Tensor:
        if operation == "encode":
            # VAE expects 5D: [B, C, T, H, W] with T=1 for images
            latents = self.vae.encode(
                data.unsqueeze(2).to(dtype=self.precision_model),
                return_dict=False,
            )[0].sample()
            # Remove temporal dim: [B, 16, 1, H', W'] -> [B, 16, H', W']
            latents = latents.squeeze(2)
            # Normalize per-channel: (latents - mean) / std
            mean = torch.as_tensor(self.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
            std = torch.as_tensor(self.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
            return (latents - mean) / std
        if operation == "decode":
            # De-normalize per-channel: latents * std + mean
            model_latents = data.to(dtype=self.precision_model)
            mean = torch.as_tensor(self.latents_mean, device=model_latents.device, dtype=model_latents.dtype).view(
                1, -1, 1, 1
            )
            std = torch.as_tensor(self.latents_std, device=model_latents.device, dtype=model_latents.dtype).view(
                1, -1, 1, 1
            )
            latents = (model_latents * std + mean).unsqueeze(2)
            images = self.vae.decode(latents, return_dict=False)[0]
            # Remove temporal dim: [B, C, T, H, W] -> [B, C, H, W]
            return images[:, :, 0].clip(-1.0, 1.0)
        raise ValueError(f"Unsupported preprocessor operation: {operation!r}")

    def decode(self, latent_images: torch.Tensor) -> torch.Tensor:
        """Decode latents to images.

        Args:
            latent_images: Normalized latent representations [B, 16, H', W'].

        Returns:
            torch.Tensor: Decoded images in [-1, 1] range, shape [B, C, H, W].
        """
        return self(latent_images, operation="decode")


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    img_shapes: List,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    r: Optional[torch.Tensor] = None,
    guidance: Optional[torch.Tensor] = None,
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
        img_shapes: Packed image shapes, e.g. [[(1, H, W)]], for RoPE and the ``[B, C, H, W]`` feature reshape.
        encoder_hidden_states: Text encoder hidden states [B, text_seq_len, 3584].
        encoder_hidden_states_mask: Text attention mask [B, text_seq_len].
        timestep: Current timestep in [0, 1] range.
        attention_kwargs: Additional attention kwargs.
        return_features_early: If True, return features as soon as collected.
        feature_indices: Set of block indices to extract features from.
        return_logvar: If True, return log variance estimate.

    Returns:
        Model output, optionally with features or logvar.
    """
    features = []

    # diffusers >= 0.39 concatenates this mask with a *bool* image mask inside
    # QwenDoubleStreamAttnProcessor2_0; a long mask promotes that to int64, which SDPA rejects.
    # Upstream normalizes the mask inside its own forward (compute_text_seq_len_from_mask), but
    # classify_forward replaces that forward entirely -- so we redo the coercion here.
    if encoder_hidden_states_mask is not None and encoder_hidden_states_mask.dtype != torch.bool:
        encoder_hidden_states_mask = encoder_hidden_states_mask.bool()

    # Store spatial size for feature reshaping
    # hidden_states: [B, seq_len, C*4] where seq_len = (H//2) * (W//2)
    spatial_size = (img_shapes[0][0][1], img_shapes[0][0][2])

    # 1. Patch embedding
    hidden_states = self.img_in(hidden_states)

    # Qwen's official repo delegates to Diffusers; QwenImageTransformer2DModel
    # casts raw time to the hidden dtype before computing the sinusoid.
    timestep = timestep.to(hidden_states.dtype)
    r = None if r is None else r.to(hidden_states.dtype)
    guidance = None if guidance is None else guidance.to(hidden_states.dtype)

    # 2. Text processing
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    # 3. Time embedding (QwenTimestepProjEmbeddings takes timestep and hidden_states for dtype)
    temb = self.time_text_embed(timestep, hidden_states)

    # 3b. Second-timestep (``r``) embedding for flow-map methods: a fresh additive
    # ``r_embedder`` (zero-init -> no-op at start). ``time_proj`` bakes in the *1000
    # scale, so ``r`` is passed in [0, 1].
    if getattr(self, "r_embedder", None) is not None and r is not None:
        r_in = (timestep - r) if self.time_cond_type == "diff" else r
        r_proj = self.time_text_embed.time_proj(r_in)
        temb = temb + self.r_embedder(r_proj.to(hidden_states.dtype))
    elif getattr(self, "r_embedder", None) is None and r is not None:
        raise ValueError("`r` was provided but the network has no `r_embedder` (set net.r_timestep=True).")

    # 3c. Guidance (CFG-scale) embedding for guidance distillation: the raw scale
    # feeds the shared ``time_proj`` (which already bakes the *1000 phase scale);
    # zero-init -> no-op at start.
    if getattr(self, "guidance_embedder", None) is not None and guidance is not None:
        guidance_proj = self.time_text_embed.time_proj(guidance)
        temb = temb + self.guidance_embedder(guidance_proj.to(hidden_states.dtype))
    elif getattr(self, "guidance_embedder", None) is None and guidance is not None:
        raise ValueError(
            "`guidance` was provided but the network has no `guidance_embedder` (set net.guidance_embeds=True)."
        )

    # 4. RoPE (handled internally by QwenEmbedRope). The text table must span the padded text length, as upstream
    # does since https://github.com/huggingface/diffusers/pull/12702: a mask-derived length is shorter whenever
    # every row is padded, and then fails to broadcast against the text tokens.
    image_rotary_emb = self.pos_embed(
        img_shapes, max_txt_seq_len=encoder_hidden_states.shape[1], device=hidden_states.device
    )

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
        if feature_indices is not None and idx in feature_indices:
            feat = hidden_states.clone()
            B, S, C = feat.shape
            feat = feat.permute(0, 2, 1).reshape(B, C, spatial_size[0], spatial_size[1])
            features.append(feat)

        if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
            return features

    # 6. Final projection
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


DEFAULT_QWEN_LORA_TARGETS: List[str] = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]


class QwenImage(LoRAMixin, FastGenNetwork):
    """Qwen-Image network for text-to-image generation.

    A ~20B parameter MMDiT model using flow matching with true classifier-free guidance.
    Uses Qwen2.5-VL-7B as text encoder and a Wan2.1-based VAE.

    Reference: https://huggingface.co/Qwen/Qwen-Image
    """

    MODEL_ID = "Qwen/Qwen-Image"
    latent_compression_factors = (8, 8)

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        disable_grad_ckpt: bool = False,
        load_pretrained: bool = True,
        r_timestep: bool = False,
        r_embedder_init: str = "zero",
        time_cond_type: str = "diff",
        guidance_embeds: bool = False,
        guidance_embedder_init: str = "zero",
        guidance_scale: Optional[float] = None,
        **schedule_kwargs,
    ):
        """QwenImage constructor.

        Args:
            model_id_or_local_path: The HuggingFace model ID or local path to load. Defaults to "Qwen/Qwen-Image".
            net_pred_type: Prediction type. Defaults to "flow" for flow matching.
            schedule_type: Schedule type. Defaults to "rf" (rectified flow), which the released weights require.
            disable_grad_ckpt: Disable activation checkpointing. FSDP needs no override: ``fully_shard``
                replaces the Diffusers checkpointing with FSDP-compatible wrappers.
            load_pretrained: Whether to load pretrained weights.
            r_timestep: Whether to support a second timestep (flow-map / mean-flow /
                AYF methods) via an additive ``r_embedder``.
            r_embedder_init: ``"zero"`` (no-op at init), ``"warm"`` (pretrained
                linear_1 features, zero linear_2 -- still a no-op), or ``"pretrained"``
                (initialize from the pretrained time embedder).
            time_cond_type: ``"diff"`` embeds ``t - r``; ``"abs"`` embeds ``r``.
            guidance_embeds: Attach an additive zero-init ``guidance_embedder`` so a
                distilled student can condition on a CFG-scale value (Qwen-Image
                itself is not guidance-distilled).
            guidance_embedder_init: ``"zero"`` (fresh linear_1) or ``"pretrained"``
                (linear_1 from the pretrained time embedder). linear_2 is always
                zeroed, so the embedder is a no-op at init either way.
            guidance_scale: Default embedded guidance value, auto-filled by
                ``forward`` (FLUX contract); requires ``guidance_embeds=True``.
        """
        # Only the released weights fix the time convention; weights trained from scratch learn their schedule's.
        if load_pretrained and schedule_type != "rf":
            raise ValueError(
                f"Qwen-Image's released weights read the raw time t as the noise level, which holds only for "
                f"rectified flow; `schedule_type` must be 'rf', got {schedule_type!r}."
            )
        # The native sampler grids start at sigma 1.0, so the raw-time bound defaults to 1.0 (configs may lower it).
        schedule_kwargs.setdefault("max_t", 1.0)
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)

        self.model_id_or_local_path = model_id_or_local_path
        self.guidance_scale = guidance_scale

        self._initialize_network(model_id_or_local_path, load_pretrained)

        # Optional second-timestep (``r``) embedder for flow-map / mean-flow / AYF methods.
        self._init_r_embedder(r_timestep, r_embedder_init, time_cond_type)

        # Optional additive guidance embedder (CFG-scale conditioning for distillation).
        self._init_guidance_embedder(guidance_embeds, guidance_embedder_init)
        if guidance_scale is not None and not guidance_embeds:
            raise ValueError("guidance_scale requires guidance_embeds=True; set guidance_scale=None to sample with CFG")

        # Override forward with classify_forward for feature extraction support
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Gradient checkpointing configuration
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    def _initialize_network(self, model_id_or_local_path: str, load_pretrained: bool) -> None:
        """Initialize the transformer network.

        Args:
            model_id_or_local_path: The HuggingFace model ID or local path.
            load_pretrained: Whether to load pretrained weights.
        """
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading QwenImage transformer from pretrained")
            self.transformer: QwenImageTransformer2DModel = hf_load(
                QwenImageTransformer2DModel.from_pretrained,
                model_id_or_local_path,
                subfolder="transformer",
            )
        else:
            config = hf_load(
                QwenImageTransformer2DModel.load_config,
                model_id_or_local_path,
                subfolder="transformer",
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

        # Add logvar linear layer for variance estimation on the time embedding
        inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.transformer.logvar_linear = torch.nn.Linear(inner_dim, 1)

    def default_lora_target_modules(self) -> List[str]:
        return DEFAULT_QWEN_LORA_TARGETS

    def _init_r_embedder(self, r_timestep: bool, r_embedder_init: str, time_cond_type: str) -> None:
        """Attach an additive second-timestep (``r``) embedder to the transformer,
        added to the time embedding inside ``classify_forward`` (Wan/DiT pattern)."""
        if time_cond_type not in ("abs", "diff"):
            raise ValueError(f"Invalid time condition: {time_cond_type}")
        self.transformer.time_cond_type = time_cond_type
        if not r_timestep:
            self.transformer.r_embedder = None
            return

        if r_embedder_init not in ("zero", "pretrained", "warm"):
            raise ValueError(f"Unknown r_embedder_init '{r_embedder_init}' ('zero', 'pretrained', or 'warm').")
        self._r_embedder_init = r_embedder_init

        inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.transformer.r_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
        # Under FSDP meta-init the params are meta tensors ('pretrained' cannot even be
        # copied); skip -- rank 0 initializes for real and FSDP broadcasts its weights.
        if self._is_in_meta_context():
            logger.info("Skipping r_embedder weight init on meta device (weights will arrive via FSDP sync)")
            return
        self._reset_r_embedder_parameters()
        logger.info(f"Initialized QwenImage r_embedder (init={r_embedder_init}, time_cond_type={time_cond_type}).")

    def _reset_r_embedder_parameters(self) -> None:
        """Apply the configured r_embedder init; the zero-init keeps the r_embedder a no-op at start."""
        r_embedder = getattr(self.transformer, "r_embedder", None)
        if r_embedder is None:
            return
        if self._r_embedder_init == "pretrained":
            r_embedder.load_state_dict(self.transformer.time_text_embed.timestep_embedder.state_dict())
            return
        if self._r_embedder_init == "warm":
            # Feature warm start: linear_1 from the pretrained time embedder;
            # linear_2 stays zero so the embedder remains a no-op at init.
            r_embedder.linear_1.load_state_dict(
                self.transformer.time_text_embed.timestep_embedder.linear_1.state_dict()
            )
        torch.nn.init.zeros_(r_embedder.linear_2.weight)
        if r_embedder.linear_2.bias is not None:
            torch.nn.init.zeros_(r_embedder.linear_2.bias)

    def _init_guidance_embedder(self, guidance_embeds: bool, guidance_embedder_init: str = "zero") -> None:
        """Attach an additive guidance embedder to the transformer (same pattern as
        ``_init_r_embedder``); active whenever ``guidance`` is passed to forward."""
        if not guidance_embeds:
            self.transformer.guidance_embedder = None
            return

        if guidance_embedder_init not in ("zero", "pretrained"):
            raise ValueError(f"Unknown guidance_embedder_init '{guidance_embedder_init}' ('zero' or 'pretrained').")
        self._guidance_embedder_init = guidance_embedder_init

        inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.transformer.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
        if self._is_in_meta_context():
            logger.info("Skipping guidance_embedder weight init on meta device (weights will arrive via FSDP sync)")
            return
        self._reset_guidance_embedder_parameters()
        logger.info(f"Initialized QwenImage guidance_embedder (init={guidance_embedder_init}).")

    def _reset_guidance_embedder_parameters(self) -> None:
        """Apply the guidance_embedder init (see ``_reset_r_embedder_parameters``)."""
        guidance_embedder = getattr(self.transformer, "guidance_embedder", None)
        if guidance_embedder is None:
            return
        if self._guidance_embedder_init == "pretrained":
            # Feature warm start: linear_1 from the pretrained time embedder;
            # linear_2 stays zero so the embedder remains a no-op at init.
            guidance_embedder.linear_1.load_state_dict(
                self.transformer.time_text_embed.timestep_embedder.linear_1.state_dict()
            )
        torch.nn.init.zeros_(guidance_embedder.linear_2.weight)
        if guidance_embedder.linear_2.bias is not None:
            torch.nn.init.zeros_(guidance_embedder.linear_2.bias)

    def reset_parameters(self):
        """Rebuild the QwenEmbedRope frequency tables after meta materialization."""
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
                logger.debug("Reinitialized QwenEmbedRope pos_freqs/neg_freqs from meta device")

        super().reset_parameters()

        logger.debug("Reinitialized QwenImage parameters")

    def fully_shard(self, **kwargs):
        """Shard ``self.transformer``; see ``FastGenNetwork.fully_shard``.

        QwenImage has a single list of 60 dual-stream QwenImageTransformerBlock instances.
        """
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(self.transformer, QwenImageTransformerBlock)

        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)

        fully_shard(self.transformer, **kwargs)

    def init_preprocessors(self):
        """Initialize text and image encoders."""
        if "text_encoder" not in self.preprocessors:
            self.init_text_encoder()
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_text_encoder(self, dtype: Optional[torch.dtype] = torch.bfloat16):
        """Initialize the text encoder for QwenImage."""
        self.preprocessors["text_encoder"] = QwenImageTextEncoder(
            model_id_or_local_path=self.model_id_or_local_path, dtype=dtype
        )

    def init_vae(self):
        """Initialize only the VAE for visualization."""
        self.preprocessors["vae"] = QwenImageImageEncoder(model_id_or_local_path=self.model_id_or_local_path)

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
        guidance: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of QwenImage diffusion model.

        Args:
            x_t: The diffused data sample [B, C, H, W].
            t: The current timestep in [0, 1] range.
            condition: Tuple of (encoder_hidden_states, encoder_hidden_states_mask).
            r: Second timestep for flow-map / mean-flow / AYF (None -> ignored).
            guidance: Per-sample CFG-scale conditioning. ``None`` is auto-filled from
                ``self.guidance_scale`` when a ``guidance_embedder`` exists (FLUX contract).
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
            r=r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            supports_r_timestep=getattr(self.transformer, "r_embedder", None) is not None,
            num_features=len(self.transformer.transformer_blocks),
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        x_t_model = self.cast_to_model_precision(x_t)
        encoder_hidden_states, encoder_hidden_states_mask = condition
        assert_same(dtype=self.precision_model, device=x_t_model.device, encoder_hidden_states=encoder_hidden_states)
        assert_same(
            shape=True,
            dtype=self.noise_scheduler.t_precision,
            device=True,
            t=t,
            r=r,
        )
        batch_size = x_t_model.shape[0]
        height, width = x_t_model.shape[2], x_t_model.shape[3]

        # Guidance-distilled nets fill the embedded default (FLUX contract); ``__init__`` ties the scale to an embedder.
        if guidance is None and self.guidance_scale is not None:
            guidance = torch.full((batch_size,), self.guidance_scale, device=x_t_model.device, dtype=x_t_model.dtype)
        assert_same(shape=True, device=True, t=t, guidance=guidance)
        # Pack latents for transformer: [B, C, H, W] -> [B, (H//2)*(W//2), C*4]
        hidden_states = self._pack_latents(x_t_model)

        # Prepare img_shapes: [[(1, latent_H_packed, latent_W_packed)]] per batch item
        img_shapes = [[(1, height // 2, width // 2)]] * batch_size

        model_outputs = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=t,
            r=r,
            guidance=guidance,
            img_shapes=img_shapes,
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
            out = self.transform_output(out, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning)
        else:
            out[0] = self._unpack_latents(out[0], height, width)
            out[0] = self.transform_output(
                out[0], x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
            )

        if return_logvar:
            return out, logvar.to(dtype=x_t.dtype)
        return out

    @staticmethod
    def default_guidance_cfg() -> GuidanceConfig:
        """Default guidance recipe: CFG at 4.0 + per-token L2 rescale (matches the upstream pipeline)."""
        g = GuidanceConfig()
        g.cfg.scale = 4.0
        g.rescale = 1.0
        g.rescale_type = "norm"
        g.rescale_dims = (-1,)
        return g

    def predict_with_guidance(self, ctx: GuidanceContext) -> torch.Tensor:
        """Compose Qwen-Image's packed-token rescaling on the provider's native predictions (flow, as upstream)."""
        g = ctx.guidance_cfg
        g.validate(supported_terms=("cfg", "mix", "rescale"), network_name=type(self).__name__)
        cond = ctx.predict("pos", condition=ctx.condition)
        # Linear CFG/mix commute with packing; keep these operations in BCHW.
        guided = apply_cfg(ctx, cond)
        guided = apply_mix(ctx, guided)
        if g.rescale is not None:
            # Per-token norms require [B, tokens, 4*C], not BCHW's spatial last axis.
            height, width = guided.shape[-2:]
            guided = self._unpack_latents(
                apply_rescale(ctx, self._pack_latents(guided), condition_prediction=self._pack_latents(cond)),
                height,
                width,
            )
        return guided

    def sample_scheduler_and_grid(
        self, sample_steps: int, image_seq_len: int, device: Optional[torch.device] = None
    ) -> Tuple[FlowMatchEulerDiscreteScheduler, torch.Tensor]:
        """Lazily load the pipeline's flow-matching Euler scheduler and set it to the official pipeline's grid.

        That is dynamic shifting on ``linspace(1, 1/steps, steps)``, with ``mu`` interpolated from the scheduler config
        in ``image_seq_len`` (the token count after 2x2 packing). Needs ``sample_steps >= 2``.

        Returns:
            The scheduler, set to begin at the first step, and the times to query the network at, one per step: its
            sigmas in ``t_precision``.
        """
        # At 1 step the scheduler's shift_terminal stretch divides by zero -> NaN timesteps.
        assert sample_steps >= 2, f"QwenImage sampling needs sample_steps >= 2, got {sample_steps}"
        if not hasattr(self, "scheduler"):
            self.scheduler = hf_load(
                FlowMatchEulerDiscreteScheduler.from_pretrained,
                self.model_id_or_local_path,
                subfolder="scheduler",
            )
        scheduler = self.scheduler

        cfg = scheduler.config
        mu = cfg.base_shift + (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len - cfg.base_image_seq_len) * (
            image_seq_len - cfg.base_image_seq_len
        )
        scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / sample_steps, sample_steps), device=device, mu=mu)
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
        sample_steps: int = 50,
        sample_type: str = "ode",
        guidance_cfg: Optional[GuidanceConfig] = None,
        guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples using Euler flow matching.

        Embedded-guidance mode (a ``guidance_embedder`` net) runs one conditioned
        forward per step, ignoring ``neg_condition``/``guidance_cfg``; CFG mode
        (undistilled base) applies true CFG via ``predict_with_guidance``.

        Args:
            noise: Initial noise tensor [B, C, H, W].
            condition: Tuple of (encoder_hidden_states, encoder_hidden_states_mask).
            neg_condition: Optional negative condition tuple; CFG mode only.
            sample_steps: Number of sampling steps (default 50).
            sample_type: Integration policy; only ``"ode"`` is supported.
            guidance_cfg: Guidance config; CFG mode only. ``None`` →
                ``default_guidance_cfg()`` (CFG at 4.0 + per-token L2 rescale).
            guidance: Per-sample guidance value for a ``guidance_embedder`` net; passing it
                selects embedded-guidance mode. ``None`` uses ``self.guidance_scale`` when the
                net has a ``guidance_embedder`` and runs CFG otherwise.

        Returns:
            Generated latent samples.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )

        batch_size, _, height, width = noise.shape

        # Embedded mode iff an explicit guidance kwarg or a configured default with
        # an embedder present.
        embedded_cfg = guidance is not None or (
            self.guidance_scale is not None and getattr(self.transformer, "guidance_embedder", None) is not None
        )
        if not embedded_cfg and guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()

        image_seq_len = (height // 2) * (width // 2)
        scheduler, t_grid = self.sample_scheduler_and_grid(sample_steps, image_seq_len, device=noise.device)

        # Initialize latents with proper scaling based on the initial timestep
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])

        # Sampling loop
        for sched_step, t in zip(scheduler.timesteps, t_grid, strict=True):
            t = t.expand(batch_size)

            if embedded_cfg:
                # Embedded guidance
                pred = self(latents, t, condition=condition, guidance=guidance, fwd_pred_type="flow")
            else:
                # Manual guidance (true CFG; block the embedded auto-fill).
                with attr_override(self, "guidance_scale", None):
                    pred = predict_with_guidance(
                        self,
                        latents,
                        t,
                        guidance_cfg=guidance_cfg,
                        condition=condition,
                        neg_condition=neg_condition,
                        fwd_pred_type="flow",
                    )

            # Euler step
            latents = scheduler.step(pred, sched_step, latents, return_dict=False)[0]

        return latents
