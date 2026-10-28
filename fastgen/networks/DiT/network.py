# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Transformer-based diffusion model architecture for FastGen.
This implementation provides a clean, modular DiT-style architecture with
adaptive normalization and patch-based processing for latent diffusion.
"""

from typing import List, Set, Optional, Union, Tuple, Mapping, Any
from functools import partial

import numpy as np
import math
import timm
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from diffusers import DDPMScheduler
from diffusers.models import AutoencoderKL

from fastgen.networks.network import FastGenNetwork, FastGenModule
from fastgen.networks.lora_utils import LoRAMixin
from fastgen.utils.tree_utils import OutputDtype, assert_same
from fastgen.utils.guidance import (
    GuidanceConfig,
    predict_with_guidance,
    apply_cfg,
    apply_mix,
    apply_rescale,
    GuidanceContext,
)
from fastgen.utils.env import hf_load
from fastgen.utils.tensor import to_high_precision
import fastgen.utils.logging_utils as logger


def apply_adaptive_modulation(features, shift, scale):
    """
    Apply adaptive modulation to features using shift and scale parameters.

    Args:
        features: Input features of shape (B, N, D)
        shift: Shift parameters of shape (B, D)
        scale: Scale parameters of shape (B, D)

    Returns:
        Modulated features of shape (B, N, D)
    """
    return features * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#          Time and Condition Embedding Modules for Adaptive Conditioning        #
#################################################################################


class FourierTimeEmbedding(nn.Module):
    """
    Fourier-based time embedding module that projects scalar timesteps
    to high-dimensional vectors using sinusoidal frequency encoding.
    """

    def __init__(self, embed_dim, frequency_dim=256, max_freq=10000):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_freq = max_freq

        # Projection network with residual connection
        self.proj_net = nn.Sequential(
            nn.Linear(frequency_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

    def encode_timesteps(self, timesteps, dim):
        """
        Generate Fourier features from timesteps using sinusoidal encoding.

        Args:
            timesteps: (B, ) tensor of timestep values
            dim: Dimension of the frequency encoding

        Returns:
            (B, dim) tensor of Fourier features
        """
        half_dim = dim // 2

        # Compute frequency bands
        freq_bands = torch.exp(
            -math.log(self.max_freq)
            * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
            / half_dim
        )

        # Compute sinusoidal encodings
        angles = timesteps[:, None].float() * freq_bands[None, :]
        fourier_features = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

        # Handle odd dimensions
        if dim % 2 == 1:
            fourier_features = torch.cat([fourier_features, torch.zeros_like(fourier_features[:, :1])], dim=-1)

        return fourier_features

    def forward(
        self,
        t: torch.Tensor,
        proj_input_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        fourier_feats = self.encode_timesteps(t, self.frequency_dim)
        # SiT computes the sinusoidal features in FP32 and feeds them directly to this
        # MLP. Cast at that boundary for mixed precision, using the configured compute
        # dtype rather than a parameter dtype that may describe FP32 master storage.
        embeddings = self.proj_net(fourier_feats.to(dtype=proj_input_dtype))
        return embeddings


class ConditionalEmbedding(nn.Module):
    """
    Conditional embedding module for class labels with built-in classifier-free guidance support.
    Provides learnable embeddings for discrete class labels and handles stochastic masking
    for classifier-free guidance during training.
    """

    def __init__(self, num_classes, embed_dim, cfg_enabled, cfg_dropout_rate):
        super().__init__()
        self.num_classes = num_classes
        self.cfg_dropout_rate = cfg_dropout_rate if cfg_enabled else 0.0

        # Add one extra embedding for null/unconditional token when CFG is enabled
        num_embeddings = num_classes + (1 if cfg_dropout_rate > 0 else 0)
        self.class_embeddings = nn.Embedding(num_embeddings, embed_dim)

    def apply_cfg_mask(self, class_ids, force_mask=None):
        """
        Apply classifier-free guidance masking to class labels.

        Args:
            class_ids: (B, ) tensor of class indices
            force_mask: Optional (B, ) tensor to force specific samples to be masked

        Returns:
            (B, ) tensor with some labels replaced by unconditional token
        """
        if force_mask is not None:
            mask = force_mask == 1
        else:
            mask = torch.rand(class_ids.shape[0], device=class_ids.device) < self.cfg_dropout_rate

        # Replace masked labels with unconditional token index
        masked_ids = torch.where(mask, self.num_classes, class_ids)
        return masked_ids

    def forward(self, class_labels, is_training, force_mask=None):
        # Apply masking if in training mode or forced
        if (is_training and self.cfg_dropout_rate > 0) or (force_mask is not None):
            class_labels = self.apply_cfg_mask(class_labels, force_mask)

        return self.class_embeddings(class_labels)


#################################################################################
#           Transformer Block with Adaptive Normalization                       #
#################################################################################


class DiTBlock(nn.Module):
    """
    Transformer block with adaptive layer normalization and gated residuals.

    This block implements a two-stage architecture:
    1. Multi-head self-attention with adaptive normalization
    2. Feed-forward MLP with adaptive normalization

    Both stages use gated residual connections conditioned on the input conditioning signal.
    """

    def __init__(self, model_dim, num_attention_heads, ffn_expansion=4.0, **kwargs):
        super().__init__()
        # Attention branch with parameter-free normalization
        self.attn_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(model_dim, num_heads=num_attention_heads, qkv_bias=True, **kwargs)

        # FFN branch with parameter-free normalization
        self.ffn_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        ffn_hidden_dim = int(model_dim * ffn_expansion)
        self.feed_forward = Mlp(
            in_features=model_dim,
            hidden_features=ffn_hidden_dim,
            act_layer=partial(nn.GELU, approximate="tanh"),
            drop=0,
        )

        # Adaptive modulation network: generates 6 conditioning parameters
        # (scale/shift/gate for attention, scale/shift/gate for FFN)
        self.conditioning_net = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, 6 * model_dim, bias=True))

    def forward(self, hidden_states, conditioning):
        # Generate adaptive parameters from conditioning
        params = self.conditioning_net(conditioning).chunk(6, dim=1)
        attn_shift, attn_scale, attn_gate, ffn_shift, ffn_scale, ffn_gate = params

        # Attention branch with adaptive modulation and gated residual
        normed_hidden = self.attn_norm(hidden_states)
        modulated_hidden = apply_adaptive_modulation(normed_hidden, attn_shift, attn_scale)
        attn_output = self.attention(modulated_hidden)
        hidden_states = hidden_states + attn_gate.unsqueeze(1) * attn_output

        # FFN branch with adaptive modulation and gated residual
        normed_hidden = self.ffn_norm(hidden_states)
        modulated_hidden = apply_adaptive_modulation(normed_hidden, ffn_shift, ffn_scale)
        ffn_output = self.feed_forward(modulated_hidden)
        hidden_states = hidden_states + ffn_gate.unsqueeze(1) * ffn_output

        return hidden_states


class OutputProjection(nn.Module):
    """
    Final output projection layer with adaptive normalization.
    Projects transformer hidden states back to patch-space predictions.
    """

    def __init__(self, hidden_dim, patch_dim, output_channels):
        super().__init__()
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.projection = nn.Linear(hidden_dim, patch_dim * patch_dim * output_channels, bias=True)

        # Conditioning network for adaptive normalization
        self.adaptive_params = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True))

    def forward(self, hidden_states, conditioning):
        # Apply adaptive normalization
        shift, scale = self.adaptive_params(conditioning).chunk(2, dim=1)
        normalized = apply_adaptive_modulation(self.output_norm(hidden_states), shift, scale)

        # Project to output space
        output = self.projection(normalized)
        return output


class DiT(LoRAMixin, FastGenNetwork):
    """Diffusion Transformer.

    Runs in the SD-VAE's latent space for ImageNet-256/512 setups. To run the
    same transformer backbone on raw pixels, set ``input_size`` / ``in_channels``
    accordingly, disable preprocessors, and override ``encode_real`` to return
    ``real_raw``.
    """

    # The SD-VAE that ``init_vae`` loads.
    latent_compression_factors = (8, 8)

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        enable_class_dropout=False,
        label_dim=1000,
        learn_sigma=False,
        r_timestep=False,
        scale_t=True,  # whether to rescale the timesteps
        time_cond_type="abs",  # "abs" or "diff"
        net_pred_type="flow",  # Prediction type for FastGenNetwork
        schedule_type="rf",  # Schedule type for FastGenNetwork
        enable_fused_attn=True,
        use_sit_convention=False,  # Use SiT convention: t -> 1-t and v -> -v
        cfg_channels: Optional[int] = None,  # Leading output channels CFG guides (None: all); the DiT/SiT releases: 3
        vae_variant="mse",  # VAE variant: "mse" or "ema"
        **schedule_kwargs,
    ):
        timm.layers.set_fused_attn(enable_fused_attn)
        self.use_sit_convention = use_sit_convention
        if use_sit_convention and scale_t:
            raise ValueError(
                "use_sit_convention=True requires scale_t=False: SiT conditions on the continuous time 1 - t in [0, 1]."
            )
        if cfg_channels is not None and not 0 < cfg_channels <= in_channels:
            raise ValueError(f"cfg_channels must be in [1, {in_channels}] or None, got {cfg_channels}.")
        if class_dropout_prob <= 0:
            raise ValueError(
                "DiT needs class_dropout_prob > 0: the extra embedding row is its null class for unconditional and "
                "CFG-negative forwards."
            )
        self.cfg_channels = cfg_channels
        self.vae_variant = vae_variant

        # Initialize FastGenNetwork with DiT-specific defaults
        super().__init__(
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            **schedule_kwargs,
        )
        self.input_size = input_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.label_dim = label_dim

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = FourierTimeEmbedding(hidden_size)

        self.scale_t = scale_t
        self.r_timestep = r_timestep
        self.time_cond_type = time_cond_type
        if self.r_timestep:
            self.r_embedder = FourierTimeEmbedding(hidden_size)
        else:
            self.r_embedder = None

        self.y_embedder = ConditionalEmbedding(label_dim, hidden_size, enable_class_dropout, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Positional embeddings (persistent=True to save in state_dict)
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size), persistent=True)

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, ffn_expansion=mlp_ratio) for _ in range(depth)])
        self.final_layer = OutputProjection(hidden_size, patch_size, self.out_channels)
        self.logvar_linear = nn.Linear(hidden_size, 1)
        self.initialize_weights()

    def initialize_weights(self):
        # Xavier initialization for linear layers
        def _xavier_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_xavier_init)

        # Initialize positional embeddings with sinusoidal encoding
        grid_size = int(self.x_embedder.num_patches**0.5)
        pos_encoding = compute_sinusoidal_2d_embeddings(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_encoding).float().unsqueeze(0))

        # Initialize patch embedding projection
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize class embeddings
        nn.init.normal_(self.y_embedder.class_embeddings.weight, std=0.02)

        # Initialize time embedding networks
        nn.init.normal_(self.t_embedder.proj_net[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.proj_net[2].weight, std=0.02)

        if self.r_timestep:
            nn.init.normal_(self.r_embedder.proj_net[0].weight, std=0.02)
            nn.init.normal_(self.r_embedder.proj_net[2].weight, std=0.02)

        # Zero-initialize adaptive conditioning layers for residual path
        for block in self.blocks:
            nn.init.constant_(block.conditioning_net[-1].weight, 0)
            nn.init.constant_(block.conditioning_net[-1].bias, 0)

        # Zero-initialize output projection layers
        nn.init.constant_(self.final_layer.adaptive_params[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaptive_params[-1].bias, 0)
        nn.init.constant_(self.final_layer.projection.weight, 0)
        nn.init.constant_(self.final_layer.projection.bias, 0)

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        """Load state dict with automatic conversion for Facebook DiT checkpoints.

        Converts key names from Facebook's DiT checkpoint format to our naming convention:
            - t_embedder.mlp -> t_embedder.proj_net
            - blocks.X.attn -> blocks.X.attention
            - blocks.X.mlp -> blocks.X.feed_forward
            - blocks.X.adaLN_modulation -> blocks.X.conditioning_net
            - y_embedder.embedding_table -> y_embedder.class_embeddings
            - final_layer.linear -> final_layer.projection
            - final_layer.adaLN_modulation -> final_layer.adaptive_params
        """
        # Check if this looks like a Facebook DiT checkpoint
        is_facebook_format = any(
            k.startswith("t_embedder.mlp") or ".attn." in k or ".adaLN_modulation" in k for k in state_dict.keys()
        )
        if is_facebook_format:
            logger.info("Detected Facebook DiT checkpoint format, converting keys...")
            new_state_dict = {}

            for key, value in state_dict.items():
                new_key = key

                # Apply key conversions (order matters - more specific rules first)
                new_key = new_key.replace("t_embedder.mlp", "t_embedder.proj_net")
                new_key = new_key.replace(".attn.", ".attention.")
                new_key = new_key.replace(".mlp.", ".feed_forward.")
                new_key = new_key.replace("y_embedder.embedding_table", "y_embedder.class_embeddings")
                new_key = new_key.replace("final_layer.linear", "final_layer.projection")
                # Handle final_layer.adaLN_modulation BEFORE the generic .adaLN_modulation. rule
                new_key = new_key.replace("final_layer.adaLN_modulation", "final_layer.adaptive_params")
                new_key = new_key.replace(".adaLN_modulation.", ".conditioning_net.")

                new_state_dict[new_key] = value

            logger.info(f"Converted {len(new_state_dict)} keys from Facebook DiT format")
            state_dict = new_state_dict

        return super().load_state_dict(state_dict, **kwargs)

    def fully_shard(self, **kwargs):
        """
        Apply FSDP sharding to DiT network components.

        Shards each transformer block and embedding module independently
        for optimal memory efficiency in distributed training.
        """
        # ``self`` is never sharded (see ``FastGenNetwork.fully_shard``), so every group here, blocks included, is an
        # FSDP root. Roots keep their parameters gathered after forward by default, which elsewhere only affects the
        # parameters outside the blocks but here would keep the whole model gathered, so reshard like nested groups.
        kwargs = {**kwargs, "reshard_after_forward": True}

        # Shard each transformer block
        for block in self.blocks:
            fully_shard(block, **kwargs)

        # Shard embedding and projection modules
        fully_shard(self.x_embedder, **kwargs)
        fully_shard(self.t_embedder, **kwargs)
        fully_shard(self.y_embedder, **kwargs)
        fully_shard(self.final_layer, **kwargs)
        fully_shard(self.logvar_linear, **kwargs)

        if self.r_embedder is not None:
            fully_shard(self.r_embedder, **kwargs)

    def init_preprocessors(self):
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_vae(self):
        """Initialize the VAE for visualization/decoding."""
        self.preprocessors["vae"] = SDVAE(variant=self.vae_variant)

    def unpatchify(self, x):
        """
        Reconstruct spatial features from patch tokens.

        Args:
            x: (N, T, patch_size^2 * C) tensor of patch tokens

        Returns:
            imgs: (N, C, H, W) tensor of spatial features
        """
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, -1)
        x = torch.einsum("bhwpqc->bchpwq", x)
        imgs = x.reshape(x.shape[0], -1, h * p, w * p)
        return imgs

    def prepare_t(self, t: Optional[torch.Tensor] = None) -> torch.Tensor | None:
        if t is None:
            return t
        if self.scale_t:
            t = self.noise_scheduler.rescale_t(t)
        if self.use_sit_convention:
            t = 1 - t
        return t

    def _convert_output(self, prediction, x_t, t, condition, fwd_pred_type, return_variance=False, **context):
        """Convert the prediction channels; appended variance channels (``return_variance``) are not a prediction."""
        if not return_variance:
            return super()._convert_output(prediction, x_t, t, condition, fwd_pred_type)
        prediction, variance = prediction.chunk(2, dim=1)
        prediction = super()._convert_output(prediction, x_t, t, condition, fwd_pred_type)
        return torch.cat([prediction, variance.to(dtype=prediction.dtype)], dim=1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        return_variance: bool = False,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of DiT.
        x_t: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N, ) tensor of diffusion timesteps
        condition: (N, C) tensor of class labels
        r: (N, ) tensor of another diffusion timesteps
        return_variance: append a ``learn_sigma`` model's variance channels to the prediction, the reference DiT's
            output layout
        apply_output_conditioning: apply the output constraint after the conversion
        output_dtype: ``"same"`` rounds the prediction to the state dtype; ``None`` keeps it in high precision
        """
        if return_variance and not self.learn_sigma:
            raise ValueError("return_variance requires a learn_sigma=True DiT.")
        self._validate_forward_kwargs(
            fwd_kwargs,
            r=r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            supports_r_timestep=self.r_embedder is not None,
            num_features=len(self.blocks),
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        # Match EDM's conditioning contract: None denotes unconditional
        # inference. DiT represents that state with its learned null-class token.
        if condition is None:
            condition = torch.full(
                (x_t.shape[0],),
                self.label_dim,
                device=x_t.device,
                dtype=torch.long,
            )
        # Non-None conditions are passed in as class indices or one-hot encodings.
        elif condition.ndim == 2:
            # this mimics the behavior of LabelEmbedder.token_drop, mapping a zero-vector to `self.label_dim`
            # (`__init__` requires `class_dropout_prob > 0`, which allocates that extra embedding row)
            mask = torch.any(condition, dim=1)
            condition = torch.where(~mask, self.label_dim, condition.argmax(dim=1))

        assert_same(
            shape=True,
            dtype=self.noise_scheduler.t_precision,
            device=True,
            t=t,
            r=r,
        )
        x_t_model = self.cast_to_model_precision(x_t)
        t_model = self.prepare_t(t)
        r_model = self.prepare_t(r)
        assert condition.dtype in (
            torch.int32,
            torch.int64,
        ), f"condition must contain integer class indices, got {condition.dtype}"
        assert_same(shape=True, device=True, t=t, condition=condition)

        # Embed spatial patches and add positional information
        x = self.x_embedder(x_t_model)
        x = x + self.pos_embed.to(dtype=x.dtype)

        # Embed time conditioning
        t_emb = self.t_embedder(t_model, proj_input_dtype=x_t_model.dtype)

        # Handle secondary time conditioning (for two-stage models)
        if self.r_embedder is not None and r_model is not None:
            if self.time_cond_type == "diff":
                r_emb = self.r_embedder(t_model - r_model, proj_input_dtype=x_t_model.dtype)
            elif self.time_cond_type == "abs":
                r_emb = self.r_embedder(r_model, proj_input_dtype=x_t_model.dtype)
            else:
                raise ValueError(f"Invalid time_cond_type: {self.time_cond_type}")
        else:
            r_emb = torch.zeros_like(t_emb)

        # Embed class conditioning
        y = self.y_embedder(condition, self.training).to(dtype=t_emb.dtype)

        # Combine all conditioning signals
        assert_same(dtype=True, device=True, x=x, t_emb=t_emb, class_embedding=y, r_emb=r_emb)
        assert_same(shape=True, t_emb=t_emb, class_embedding=y, r_emb=r_emb)
        c = t_emb + y + r_emb

        # Process through transformer blocks
        features = []
        for idx, block in enumerate(self.blocks):
            x = block(x, c)
            if feature_indices is not None and idx in feature_indices:
                # Match the image-DiT discriminator contract used by Flux and
                # Qwen: [B, tokens, hidden] -> [B, hidden, patch_h, patch_w].
                batch_size, num_tokens, hidden_size = x.shape
                patch_h, patch_w = self.x_embedder.grid_size
                if num_tokens != patch_h * patch_w:
                    raise ValueError(
                        f"Cannot reshape {num_tokens} DiT feature tokens into " f"the patch grid {patch_h}x{patch_w}."
                    )
                features.append(x.permute(0, 2, 1).reshape(batch_size, hidden_size, patch_h, patch_w))

            if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
                return features

        # Project to output space
        x = self.final_layer(x, c)

        # Reconstruct spatial output
        x = self.unpatchify(x)

        # The second half of a learn_sigma model's channels is its learned variance, which only the reference DiT's
        # DDPM sampler uses; SiT leaves it unused.
        variance = None
        if self.learn_sigma:
            x, variance = x.chunk(2, dim=1)

        # SiT convention: v_sit = x_0 - eps, but RF expects v_rf = eps - x_0
        # Note: "eps" and "x0" predictions don't need negation
        if self.use_sit_convention and self.net_pred_type == "flow":
            x = -x

        if return_variance:
            x = torch.cat([x, variance], dim=1)
        x = self.transform_output(
            x,
            x_t,
            t,
            condition,
            fwd_pred_type,
            output_dtype,
            apply_output_conditioning,
            return_variance=return_variance,
        )

        out = [x, features] if feature_indices is not None else x

        # Optionally compute log variance
        if return_logvar:
            logvar = self.logvar_linear(t_emb)
            return out, logvar.to(dtype=x_t.dtype)

        return out

    @staticmethod
    def default_guidance_cfg() -> GuidanceConfig:
        """Default guidance recipe: CFG only at scale 1.5 (SiT default)."""
        g = GuidanceConfig()
        g.cfg.scale = 1.5
        return g

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        sample_steps: int = 250,
        sample_type: Optional[str] = None,
        guidance_cfg: Optional[GuidanceConfig] = None,
        last_step_size: float = 0.04,
    ) -> torch.Tensor:
        """Generate samples using appropriate sampler based on schedule_type.

        Args:
            noise: Initial noise tensor [B, C, H, W].
            condition: Class label conditioning (class indices or one-hot).
            neg_condition: Negative conditioning for CFG.
            sample_steps: Number of model evaluations (250 in the DiT and SiT references).
            sample_type: Optional integration-policy override. RF defaults to
                SiT's stochastic ``"sde"``; DDPM schedules default to ``"ddpm"``,
                the reference DiT's ancestral sampler, and take DDIM steps with
                ``"ode"``.
            guidance_cfg: Composite ``GuidanceConfig``; ``None`` →
                ``default_guidance_cfg()``. Supported terms flow through
                ``predict_with_guidance``; unsupported perturbation modifiers raise.
                With ``self.cfg_channels``, CFG guides only that many leading
                channels and takes the rest from the conditional prediction, as
                the DiT and SiT releases do with 3.
            last_step_size: Size of the final deterministic step for RF SDE sampling.

        Returns:
            Generated samples of shape [B, C, H, W].
        """
        if sample_type is None:
            sample_type = "sde" if self.schedule_type == "rf" else "ddpm"
        supported_sample_types = ("ode", "sde") if self.schedule_type == "rf" else ("ode", "ddpm")
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("ddpm", "sd", "sdxl", "alphas", "rf"),
            supported_sample_types=supported_sample_types,
        )
        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()

        if self.schedule_type != "rf":
            return self._sample_ddpm(
                noise,
                condition=condition,
                neg_condition=neg_condition,
                guidance_cfg=guidance_cfg,
                sample_steps=sample_steps,
                sample_type=sample_type,
            )
        if sample_type == "sde":
            return self._sample_flow_sde(
                noise,
                condition=condition,
                neg_condition=neg_condition,
                guidance_cfg=guidance_cfg,
                sample_steps=sample_steps,
                last_step_size=last_step_size,
            )
        return self._sample_flow(
            noise,
            condition=condition,
            neg_condition=neg_condition,
            guidance_cfg=guidance_cfg,
            sample_steps=sample_steps,
        )

    def predict_with_guidance(self, ctx: GuidanceContext) -> torch.Tensor:
        """Compose DiT/SiT's channel convention on the provider's native predictions."""
        ctx.guidance_cfg.validate(supported_terms=("cfg", "mix", "rescale"), network_name=type(self).__name__)
        cond = ctx.predict("pos", condition=ctx.condition)
        guided = apply_cfg(ctx, cond)
        if self.cfg_channels is not None and guided is not cond:
            guided = torch.cat([guided[:, : self.cfg_channels], cond[:, self.cfg_channels :]], dim=1)
        guided = apply_mix(ctx, guided)
        # A learn_sigma model's variance channels ride along (return_variance=True); rescale the prediction only.
        if ctx.fwd_kwargs.get("return_variance"):
            c = guided.shape[1] // 2
            guided = torch.cat([apply_rescale(ctx, guided[:, :c], cond[:, :c]), cond[:, c:]], dim=1)
        else:
            guided = apply_rescale(ctx, guided)
        return guided

    def _sample_flow(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        guidance_cfg: Optional[GuidanceConfig] = None,
        sample_steps: int = 250,
    ) -> torch.Tensor:
        """Generate samples using Euler method for flow matching (ODE).

        Uses the RF noise schedule via the noise_scheduler:
        x_t = (1-t)*x_0 + t*eps, integrated over the descending grid t: max_t → 0
        with the DDIM step on the flow, which on unshifted RF is the Euler step x = x + (t_next - t) * v.

        Args:
            noise: Initial noise tensor.
            condition: Class conditioning (one-hot or indices).
            neg_condition: Negative/null conditioning for CFG.
            guidance_cfg: Composite ``GuidanceConfig`` forwarded to
                ``predict_with_guidance``.
            sample_steps: Number of sampling steps.
        """
        grid = self.noise_scheduler.grid(sample_steps, device=noise.device)

        x = self.noise_scheduler.latents(noise=noise, t_init=grid[0])
        for t, t_next in zip(grid[:-1], grid[1:]):
            t_batch = t.expand(x.shape[0])
            v = predict_with_guidance(
                self,
                x,
                t_batch,
                condition,
                guidance_cfg=guidance_cfg,
                neg_condition=neg_condition,
                batched=True,
                output_dtype=None,
                fwd_pred_type="flow",
            )
            x = self.noise_scheduler.step(x, v, t_batch, t_next, pred_type="flow")

        return x

    def _sample_flow_sde(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        guidance_cfg: Optional[GuidanceConfig] = None,
        sample_steps: int = 250,
        last_step_size: float = 0.04,
    ) -> torch.Tensor:
        """Generate samples using SDE sampling.

        Implements SiT's default SDE sampler (``Sampler.sample_sde``): Euler-Maruyama steps with diffusion
        ``sigma = t`` from ``max_t`` over ``linspace(max_t, last_step_size, sample_steps)``, then the deterministic
        "Mean" step to t=0 (if last_step_size > 0): ``sample_steps`` model evaluations, one fewer when
        ``last_step_size == 0``. SiT starts from pure noise at ``max_t = 1``.

        Reference: https://github.com/willisma/SiT

        Args:
            noise: Initial noise tensor.
            condition: Class conditioning (one-hot or indices).
            neg_condition: Negative/null conditioning for CFG.
            guidance_cfg: Composite ``GuidanceConfig`` forwarded to
                ``predict_with_guidance``.
            sample_steps: Number of model evaluations, at least 2.
            last_step_size: Size of the final deterministic step (0 to skip).
        """
        if sample_steps < 2:
            raise ValueError(f"The SiT SDE sampler needs at least 2 sample steps, got {sample_steps}.")
        times = torch.linspace(
            self.noise_scheduler.max_t, last_step_size, sample_steps, dtype=self.noise_scheduler.t_precision
        )
        times = self.noise_scheduler.safe_clamp(times, max=self.noise_scheduler.max_t)
        dt = (times[0] - times[1]).item()

        x = self.noise_scheduler.latents(noise=noise, t_init=times[0].to(device=noise.device))
        state_dtype = x.dtype

        for t in times[:-1].tolist():
            t_batch = torch.full([x.shape[0]], t, device=x.device, dtype=self.noise_scheduler.t_precision)

            v = predict_with_guidance(
                self,
                x,
                t_batch,
                condition,
                guidance_cfg=guidance_cfg,
                neg_condition=neg_condition,
                batched=True,
                output_dtype=None,
                fwd_pred_type="flow",
            )

            # SiT integrates dx = (v_sit + w * score) dτ with τ = 1 - t, v_sit = -v, w = sigma = t and
            # score = -eps / t = -(x + (1 - t) v) / t, so per unit τ the drift is -(v + eps) = -((2 - t) v + x).
            # Each update runs in the prediction's high precision and rounds once to the state dtype, like ``step``.
            # Fresh noise is drawn like the state, so the draw does not depend on the prediction's precision.
            fresh_noise = torch.randn_like(x)
            x, v, fresh_noise = to_high_precision(x, v, fresh_noise)
            mean_x = x - ((2.0 - t) * v + x) * dt
            x = (mean_x + math.sqrt(2.0 * t * dt) * fresh_noise).to(state_dtype)

        if last_step_size > 0:
            # Final deterministic step
            t = last_step_size
            t_batch = torch.full([x.shape[0]], t, device=x.device, dtype=self.noise_scheduler.t_precision)
            v = predict_with_guidance(
                self,
                x,
                t_batch,
                condition,
                guidance_cfg=guidance_cfg,
                neg_condition=neg_condition,
                batched=True,
                output_dtype=None,
                fwd_pred_type="flow",
            )
            x, v = to_high_precision(x, v)
            x = (x - ((2.0 - t) * v + x) * last_step_size).to(state_dtype)

        return x

    def _sample_ddpm(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        guidance_cfg: Optional[GuidanceConfig] = None,
        sample_steps: int = 250,
        sample_type: str = "ddpm",
    ) -> torch.Tensor:
        """Sample the DiT release's respaced DDPM chain from pure noise: ``sample_steps`` evenly strided timesteps
        of this network's schedule table, from the last one down to 0.

        ``"ddpm"`` takes ancestral steps, whose variance a ``learn_sigma`` model interpolates between the
        posterior's and ``beta`` and which is ``beta`` otherwise; ``"ode"`` takes DDIM steps. Both end on the x0
        prediction at timestep 0. The variance comes from the conditional prediction, which CFG leaves unguided.

        The ancestral steps are diffusers' ``DDPMScheduler`` over the kept timesteps (float32 tables). No diffusers
        scheduler takes the DDIM step on that grid (``DDIMScheduler`` strides by a fixed
        ``num_train_timesteps // steps``), so it is the noise schedule's ``step``: into timestep 0, the table's
        first, noisy entry (``clean_end=False``), and from it to x0.
        """
        schedule = self.noise_scheduler
        alphas_cumprod = to_high_precision(schedule.alphas_cumprod)
        num_timesteps = len(alphas_cumprod)
        if schedule.min_t > 0 or schedule.max_t < (num_timesteps - 1) / num_timesteps:
            raise ValueError(
                f"{type(self).__name__}'s DDPM sampler runs the schedule's whole table, timesteps 0 to "
                f"{num_timesteps - 1}, so it needs min_t = 0 and max_t >= {(num_timesteps - 1) / num_timesteps}; "
                f"got [{schedule.min_t}, {schedule.max_t}]."
            )
        timesteps = _respaced_timesteps(num_timesteps, sample_steps)
        channels = noise.shape[1]
        if sample_type == "ddpm":
            betas = 1 - alphas_cumprod / torch.cat([alphas_cumprod.new_ones(1), alphas_cumprod[:-1]])
            ddpm_scheduler = DDPMScheduler(
                num_train_timesteps=num_timesteps,
                trained_betas=betas.cpu().numpy(),
                variance_type="learned_range" if self.learn_sigma else "fixed_large",
                clip_sample=False,
                prediction_type="epsilon",
            )
            ddpm_scheduler.set_timesteps(timesteps=timesteps[:-1])

        x = noise
        for step, next_step in zip(timesteps, timesteps[1:]):
            t = torch.full([x.shape[0]], step / num_timesteps, device=x.device, dtype=schedule.t_precision)
            guided = predict_with_guidance(
                self,
                x,
                t,
                condition,
                guidance_cfg=guidance_cfg,
                neg_condition=neg_condition,
                batched=True,
                # Diffusers' step returns its inputs' dtype; the schedule's step rounds to the state dtype itself.
                output_dtype="same" if sample_type == "ddpm" else None,
                fwd_pred_type="eps",
                return_variance=self.learn_sigma,
            )
            if sample_type == "ddpm":
                x = ddpm_scheduler.step(guided, step, x).prev_sample
            else:
                # The chain always keeps timestep 0; the step from it, to ``alpha_bar_prev = 1``, is the x0 prediction.
                t_next = torch.full_like(t, next_step / num_timesteps)
                x = schedule.step(x, guided[:, :channels], t, t_next, pred_type="eps", clean_end=step == 0)

        return x


def _respaced_timesteps(num_timesteps: int, sample_steps: int) -> List[int]:
    """The DiT release's respaced chain in sampling order: ``sample_steps`` rounded, evenly strided timesteps from
    ``num_timesteps - 1`` down to 0, followed by the 0 that its last step, the x0 prediction, goes to."""
    if not 1 <= sample_steps <= num_timesteps:
        raise ValueError(f"sample_steps must be in [1, {num_timesteps}], got {sample_steps}.")
    stride = 1 if sample_steps == 1 else (num_timesteps - 1) / (sample_steps - 1)
    timesteps, position = [], 0.0
    for _ in range(sample_steps):
        timesteps.append(round(position))
        position += stride
    return sorted(set(timesteps), reverse=True) + [0]


#################################################################################
#              2D Sinusoidal Positional Encoding Generation                     #
#################################################################################


def compute_sinusoidal_2d_embeddings(
    embedding_dim: int,
    spatial_size: int,
    include_cls_token: bool = False,
    num_extra_tokens: int = 0,
    max_wavelength: float = 10000.0,
) -> np.ndarray:
    """
    Generate 2D sinusoidal positional embeddings for a square spatial grid.

    Args:
        embedding_dim: Dimension of the output embeddings (must be even)
        spatial_size: Height and width of the spatial grid
        include_cls_token: Whether to prepend zeros for classification token
        num_extra_tokens: Number of extra token positions to prepend
        max_wavelength: Maximum wavelength for frequency bands

    Returns:
        pos_embeddings: Array of shape (num_positions, embedding_dim)
    """
    assert embedding_dim % 2 == 0, "Embedding dimension must be even"

    # Generate coordinate grids
    y_coords = np.arange(spatial_size, dtype=np.float32)
    x_coords = np.arange(spatial_size, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")

    # Flatten spatial coordinates
    y_flat = y_grid.reshape(-1)  # (H*W,)
    x_flat = x_grid.reshape(-1)  # (H*W,)

    # Split embedding dimension between x and y coordinates
    dim_per_axis = embedding_dim // 2

    # Encode x and y coordinates separately
    x_embeddings = _encode_1d_positions(x_flat, dim_per_axis, max_wavelength)
    y_embeddings = _encode_1d_positions(y_flat, dim_per_axis, max_wavelength)

    # Concatenate x and y embeddings
    pos_embeddings = np.concatenate([x_embeddings, y_embeddings], axis=1)

    # Optionally prepend zeros for extra tokens
    if include_cls_token and num_extra_tokens > 0:
        extra_token_embeddings = np.zeros((num_extra_tokens, embedding_dim), dtype=np.float32)
        pos_embeddings = np.concatenate([extra_token_embeddings, pos_embeddings], axis=0)

    return pos_embeddings


def _encode_1d_positions(positions: np.ndarray, output_dim: int, max_wavelength: float = 10000.0) -> np.ndarray:
    """
    Encode 1D positions using sinusoidal functions with varying frequencies.

    Args:
        positions: Array of position values, shape (num_positions,)
        output_dim: Output dimension (must be even)
        max_wavelength: Maximum wavelength for frequency scaling

    Returns:
        embeddings: Array of shape (num_positions, output_dim)
    """
    assert output_dim % 2 == 0, "Output dimension must be even"

    # Compute frequency bands
    half_dim = output_dim // 2
    freq_bands = np.arange(half_dim, dtype=np.float64)
    freq_bands = freq_bands / float(half_dim)
    freq_bands = 1.0 / (max_wavelength**freq_bands)

    # Compute position-frequency products
    angles = np.outer(positions, freq_bands)  # (num_positions, half_dim)

    # Apply sinusoidal functions
    sin_embeddings = np.sin(angles)
    cos_embeddings = np.cos(angles)

    # Interleave sin and cos embeddings
    embeddings = np.concatenate([sin_embeddings, cos_embeddings], axis=1)

    return embeddings


class SDVAE(FastGenModule):
    """
    The Stable Diffusion VAE for DiT.

    Handles both:
    - Raw RGB images (3 channels) -> encodes to latents
    - Pre-encoded latents (8 channels = mean + std) -> samples and scales

    The official DiT runtime and Stable Diffusion VAE are FP32 without autocast, and
    DiT applies ``0.18215`` outside the VAE immediately to its sampled posterior. The
    raw-image path retains that FP32 boundary; cached posterior moments bypass the VAE
    and are consequently sampled and scaled in their incoming dtype.
    """

    def __init__(self, variant="mse"):
        super().__init__()
        self._vae = hf_load(AutoencoderKL.from_pretrained, f"stabilityai/sd-vae-ft-{variant}")
        self.scaling_factor = self._vae.config.scaling_factor

    @property
    def precision_model(self) -> Optional[torch.dtype]:
        """Keep the released FP32 VAE fixed while allowing custom replacements to opt out."""
        if getattr(self._vae.config, "force_upcast", False):
            return torch.float32
        return super().precision_model

    @property
    def latent_compression_factors(self) -> tuple[int, int]:
        """Pixel-to-latent ``(H, W)`` downsampling of the VAE; every down block but the last halves resolution."""
        factor = 2 ** (len(self._vae.config.block_out_channels) - 1)
        return (factor, factor)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latents.

        Args:
            x: Either RGB images [B, 3, H, W] or pre-encoded [B, 8, H, W] (mean + std).

        Returns:
            Scaled latents [B, 4, H//8, W//8].
        """
        return self(x, operation="encode")

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latents to images."""
        return self(x, operation="decode")

    def forward(self, x: torch.Tensor, operation: str) -> torch.Tensor:
        if operation == "encode":
            if x.shape[1] == 3:
                # Raw RGB images - encode with VAE
                d = self._vae.encode(x.to(dtype=self.precision_model))["latent_dist"]
                latents = d.mean + torch.randn_like(d.mean) * d.std
            elif x.shape[1] == 8:
                # Pre-encoded latents (mean + std) - sample from distribution
                mean, std = x.chunk(2, dim=1)
                latents = mean + torch.randn_like(mean) * std
            else:
                raise ValueError(f"Expected 3 (RGB) or 8 (mean+std) channels, got {x.shape[1]}")
            return latents * self.scaling_factor
        if operation == "decode":
            latents = x.to(dtype=self.precision_model) / self.scaling_factor
            images = self._vae.decode(latents, return_dict=False)[0]
            return images.clamp(-1.0, 1.0)
        raise ValueError(f"Unsupported preprocessor operation: {operation!r}")
