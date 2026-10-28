# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Main network classes for Cosmos Predict2.

This module contains the main network classes:
- CosmosPredict2DiT: The core DiT transformer model
- CosmosPredict2TextEncoder: Text encoder wrapper
- CosmosPredict2: Main network class inheriting from FastGenNetwork
"""

import contextlib
import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union
from tqdm.auto import tqdm
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from fastgen.networks.Wan import WanVideoEncoder
from diffusers import UniPCMultistepScheduler

from fastgen.networks.network import FastGenNetwork, FastGenModule, validate_conditioning
from fastgen.networks.time_distributions import polynomial_spacing, time_shift

# Import all modules from modules.py
from fastgen.networks.cosmos_predict2.modules import (
    # Checkpointing
    CheckpointMode,
    SACConfig,
    HAS_CHECKPOINT_WRAPPER,
    ptd_checkpoint_wrapper,
    # Basic layers
    RMSNorm,
    # DiT components
    VideoRopePosition3DEmb,
    LearnablePosEmbAxis,
    Timesteps,
    TimestepEmbedding,
    PatchEmbed,
    FinalLayer,
    Block,
)
import fastgen.utils.logging_utils as logger
from fastgen.utils.tree_utils import OutputDtype, assert_same
from fastgen.utils.tensor import assert_tensor, expand_like
from fastgen.utils.module import module_state_names
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.env import hf_load


# ---------------------- DiT Network -----------------------


class CosmosPredict2DiT(nn.Module):
    """
    Cosmos Predict2 DiT (Diffusion Transformer) for video generation.

    A minimal implementation of the DiT architecture used in Cosmos Predict2,
    featuring 3D positional embeddings, AdaLN modulation, and cross-attention
    for text conditioning.

    Args:
        max_img_h: Maximum height of input images
        max_img_w: Maximum width of input images
        max_frames: Maximum number of frames
        in_channels: Number of input channels
        out_channels: Number of output channels
        patch_spatial: Spatial patch size
        patch_temporal: Temporal patch size
        concat_padding_mask: Whether to concatenate padding mask
        add_video_condition_mask: Whether to add video condition mask channel (for video2world/LVG models)
        model_channels: Base number of channels in the model
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Expansion ratio for MLP
        crossattn_emb_channels: Dimension of cross-attention embeddings
        pos_emb_cls: Type of positional embedding ('rope3d')
        use_adaln_lora: Whether to use AdaLN LoRA
        adaln_lora_dim: Dimension of AdaLN LoRA
        extra_per_block_abs_pos_emb: Whether to use extra per-block positional embeddings
        sac_config: Configuration for selective activation checkpointing
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        add_video_condition_mask: bool = False,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "rope3d",
        pos_emb_interpolation: str = "crop",
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        rope_enable_fps_modulation: bool = True,
        sac_config: Optional[SACConfig] = None,
        use_crossattn_projection: bool = True,  # Whether to use crossattn_proj for text embeddings
        crossattn_proj_in_channels: int = 100352,  # Input dim: full_concat = 28 * 3584 = 100352
        use_wan_fp32_strategy: bool = True,  # FP32 timestep, modulation and final projection
        skip_custom_init: bool = False,  # Skip the custom weight init; only safe when a checkpoint overwrites all params
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.add_video_condition_mask = add_video_condition_mask
        self.crossattn_emb_channels = crossattn_emb_channels
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = crossattn_proj_in_channels
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.skip_custom_init = skip_custom_init

        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_interpolation = pos_emb_interpolation
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.rope_enable_fps_modulation = rope_enable_fps_modulation

        self.build_patch_embed()
        self.build_pos_embed()

        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim

        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(
                model_channels,
                model_channels,
                use_adaln_lora=use_adaln_lora,
                use_wan_fp32_strategy=use_wan_fp32_strategy,
            ),
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)

        # Cross-attention projection for Cosmos-Reason1 embeddings
        # Projects from 100,352 dims (full_concat: 28 layers * 3584) to crossattn_emb_channels (1024)
        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        # Skip the custom trunc-normal init when skip_custom_init is set: each submodule's
        # construction already applies PyTorch's default init, and these weights are
        # overwritten by the checkpoint anyway. Buffers (e.g. RoPE tables) are built in the
        # submodule __init__ regardless of this flag, so they stay initialized.
        if not self.skip_custom_init:
            self.init_weights()

        # Apply selective activation checkpointing if configured
        if sac_config is None:
            sac_config = SACConfig()
        self.enable_selective_checkpoint(sac_config)

    def init_weights(self):
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()
        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()
        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

    def enable_selective_checkpoint(self, sac_config: SACConfig) -> "CosmosPredict2DiT":
        """
        Enable selective activation checkpointing to reduce memory usage during training.

        This method wraps transformer blocks and optionally the final layer with
        checkpoint wrappers that selectively save or recompute activations during
        the backward pass.

        Args:
            sac_config: Configuration for selective activation checkpointing.
                - mode: The checkpoint mode (none, block_wise, aggressive, etc.)
                - every_n_blocks: Apply checkpointing to every N blocks
                - checkpoint_final_layer: Whether to checkpoint the final layer

        Returns:
            Self for method chaining.

        Example:
            >>> model = CosmosPredict2DiT(...)
            >>> model.enable_selective_checkpoint(SACConfig(
            ...     mode=CheckpointMode.AGGRESSIVE,
            ...     every_n_blocks=2,
            ... ))
        """
        if sac_config.mode == CheckpointMode.NONE:
            return self

        if not HAS_CHECKPOINT_WRAPPER:
            logger.warning(
                "torch.distributed checkpoint_wrapper not available. Selective activation checkpointing disabled."
            )
            return self

        context_fn = sac_config.get_context_fn()
        if context_fn is None:
            # Fall back to basic checkpointing without selective policy
            logger.warning(
                f"Selective checkpoint contexts not available for mode {sac_config.mode}. "
                "Using basic block-wise checkpointing."
            )

        # Wrap blocks with checkpoint wrapper
        for block_id, block in self.blocks.named_children():
            if int(block_id) % sac_config.every_n_blocks == 0:
                if context_fn is not None:
                    wrapped_block = ptd_checkpoint_wrapper(
                        block,
                        context_fn=context_fn,
                        preserve_rng_state=False,
                    )
                else:
                    # Basic checkpointing without selective policy
                    wrapped_block = ptd_checkpoint_wrapper(
                        block,
                        preserve_rng_state=False,
                    )
                self.blocks.register_module(block_id, wrapped_block)

        # Optionally wrap final layer
        if sac_config.checkpoint_final_layer:
            if context_fn is not None:
                self.final_layer = ptd_checkpoint_wrapper(
                    self.final_layer,
                    context_fn=context_fn,
                    preserve_rng_state=False,
                )
            else:
                self.final_layer = ptd_checkpoint_wrapper(
                    self.final_layer,
                    preserve_rng_state=False,
                )

        return self

    def build_patch_embed(self):
        in_channels = self.in_channels
        if self.add_video_condition_mask:
            in_channels += 1  # Add channel for video condition mask (video2world/LVG models)
        if self.concat_padding_mask:
            in_channels += 1  # Add channel for padding mask
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_channels,
            out_channels=self.model_channels,
        )

    def build_pos_embed(self):
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        self.pos_embedder = cls_type(**kwargs)

        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = LearnablePosEmbAxis(**kwargs)

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        video_condition_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare embedded sequence with positional embeddings.

        Args:
            x_B_C_T_H_W: Input tensor of shape (B, C, T, H, W)
            fps: Optional FPS tensor for temporal conditioning
            padding_mask: Optional padding mask tensor
            video_condition_mask: Optional video condition mask for video2world models.
                If None and add_video_condition_mask=True, zeros are used (text2world mode).
        """
        # Add video condition mask channel for video2world/LVG models
        if self.add_video_condition_mask:
            if video_condition_mask is not None:
                # Video2world mode: use provided condition mask
                x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, video_condition_mask.type_as(x_B_C_T_H_W)], dim=1)
            else:
                # Text2world mode: use zeros as condition mask
                B, C, T, H, W = x_B_C_T_H_W.shape
                zeros_mask = torch.zeros(B, 1, T, H, W, dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
                x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, zeros_mask], dim=1)

        # Add padding mask channel
        if self.concat_padding_mask:
            if padding_mask is not None:
                # Expected input shape: (B, 1, H, W) matching official Cosmos format
                # If (B, H, W), add channel dim
                if padding_mask.ndim == 3:
                    padding_mask = padding_mask.unsqueeze(1)
                # Interpolate to match latent spatial dimensions
                padding_mask = F.interpolate(
                    padding_mask.float(),
                    size=x_B_C_T_H_W.shape[-2:],
                    mode="nearest",
                ).to(dtype=x_B_C_T_H_W.dtype)
                # Repeat over time and concat as channel
                # unsqueeze(2) adds time dimension before repeat to preserve batch dimension
                x_B_C_T_H_W = torch.cat(
                    [x_B_C_T_H_W, padding_mask.unsqueeze(2).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
                )
            else:
                # Create default padding mask of ZEROS (official Cosmos convention)
                # Shape: (B, 1, 1, H, W) -> repeat over T -> concat as channel
                B, C, T, H, W = x_B_C_T_H_W.shape
                padding_mask = torch.zeros(B, 1, 1, H, W, device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype)
                x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask.repeat(1, 1, T, 1, 1)], dim=1)

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)
        return x_B_T_H_W_D, None, extra_pos_emb

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        # Original Cosmos ordering: (p1 p2 t C)
        # This matches the original minimal_v4_dit.py unpatchify
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        video_condition_mask: Optional[torch.Tensor] = None,
        skip_layers: Optional[List[int]] = None,
        feature_indices: Optional[Set[int]] = None,
        return_features_early: bool = False,
        return_logvar: bool = False,
        adaln_lora_scale: float = 1.0,
        crossattn_gate_scale: float = 1.0,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the DiT model.

        Args:
            x_B_C_T_H_W: Input tensor of shape (B, C, T, H, W)
            timesteps_B_T: Timesteps of shape (B,) or (B, T)
            crossattn_emb: Cross-attention embeddings of shape (B, N, D)
            fps: Frames per second tensor
            padding_mask: Padding mask tensor
            video_condition_mask: Optional video condition mask for video2world models
            skip_layers: List of block indices to skip (for skip-layer guidance)
            feature_indices: Set of block indices to extract features from
            return_features_early: If True, return features once collected without final output
            return_logvar: If True, return one log variance per sample, (B, 1), for uncertainty estimation
            adaln_lora_scale: Scale factor for AdaLN LoRA (numerical stability)
            crossattn_gate_scale: Scale factor for cross-attention gating

        Returns:
            Depending on arguments:
            - torch.Tensor: Model output if feature_indices is None and logvar is not requested
            - List[torch.Tensor]: Features if return_features_early=True
            - [output, features]: If feature_indices is not None
            - (output, logvar): If return_logvar=True
        """
        if return_features_early and feature_indices is None:
            raise ValueError("CosmosPredict2DiT.forward requires feature_indices when return_features_early=True")
        if return_features_early and not feature_indices:
            return []

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            video_condition_mask=video_condition_mask,
        )

        # The reference computes timestep features in FP32 even with BF16 stored weights.
        # The selected linear layers cast their operands explicitly, without relying on FP32 autocast.
        timestep_context = (
            torch.amp.autocast(device_type=x_B_T_H_W_D.device.type, enabled=False)
            if self.use_wan_fp32_strategy
            else contextlib.nullcontext()
        )
        with timestep_context:
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)

            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # Scale adaln_lora for numerical stability
        if adaln_lora_B_T_3D is not None and adaln_lora_scale != 1.0:
            adaln_lora_B_T_3D = adaln_lora_B_T_3D * adaln_lora_scale

        # Apply cross-attention projection for Cosmos-Reason1 embeddings
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if extra_pos_emb is not None:
            assert_same(shape=True, device=True, x_B_T_H_W_D=x_B_T_H_W_D, extra_pos_emb=extra_pos_emb)

        # Block forward with feature extraction
        features = []
        for idx, block in enumerate(self.blocks):
            # A skipped block acts as the identity; still record its feature so each requested index yields one.
            if skip_layers is None or idx not in skip_layers:
                x_B_T_H_W_D = block(
                    x_B_T_H_W_D,
                    t_embedding_B_T_D,
                    crossattn_emb,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                    extra_per_block_pos_emb=extra_pos_emb,
                    crossattn_gate_scale=crossattn_gate_scale,
                )

            if feature_indices and idx in feature_indices:
                features.append(x_B_T_H_W_D)

            # Early exit if we have all requested features
            if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
                return features

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        output = self.unpatchify(x_B_T_H_W_O)

        out = [output, features] if feature_indices is not None else output

        if return_logvar:
            if not hasattr(self, "logvar_linear"):
                raise RuntimeError(
                    "logvar_linear layer is required when return_logvar=True. "
                    "Set enable_logvar_linear=True in model config."
                )
            # One value per sample: average the time embeddings of the generated frames. Video2world
            # conditioning frames are excluded, as their timestep is not the sample's.
            if video_condition_mask is None:
                logvar_emb_B_D = t_embedding_B_T_D.mean(dim=1)
            else:
                generated_B_T_1 = video_condition_mask[:, 0, :, 0, 0, None].logical_not()
                logvar_emb_B_D = (t_embedding_B_T_D * generated_B_T_1).sum(dim=1) / generated_B_T_1.sum(dim=1)
            logvar = self.logvar_linear(logvar_emb_B_D.to(x_B_C_T_H_W.dtype))
            return out, logvar

        return out


# ---------------------- Text Encoder -----------------------


class CosmosPredict2TextEncoder(FastGenModule):
    """
    Text Encoder for Cosmos Predict2 using Cosmos-Reason1 (Qwen2.5-VL based).

    Cosmos-Predict2.5 uses Cosmos-Reason1, which is based on Qwen2.5-VL-7B-Instruct.
    The text embeddings are computed using FULL_CONCAT of all 28 hidden layers,
    resulting in 100,352-dim embeddings (28 layers × 3584 hidden_size).

    The DiT model uses a crossattn_proj layer to project these to 1024 dims.

    Configuration:
        - Model: Qwen/Qwen2.5-VL-7B-Instruct
        - Hidden size: 3584
        - Num layers: 28
        - Embedding strategy: full_concat (concatenate all layers) -> 100,352 dims
        - Max sequence length: 512
    """

    # Embedding concat strategies
    FULL_CONCAT = "full_concat"  # Concatenate all layer outputs: 28 * 3584 = 100,352
    MEAN_POOLING = "mean_pooling"  # Mean of all layer outputs: 3584
    POOL_EVERY_N_LAYERS_AND_CONCAT = "pool_every_n_layers_and_concat"

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason1-7B",
        max_length: int = 512,
        embedding_concat_strategy: str = "full_concat",  # Checkpoint uses full_concat -> 100352 dims
        n_layers_per_group: int = 5,
    ):
        """
        Initialize Cosmos-Reason1 text encoder using Qwen2.5-VL.

        Args:
            model_name: HuggingFace model name or local path to Cosmos-Reason1.
                - HuggingFace: "nvidia/Cosmos-Reason1-7B"
                - Local path: "/path/to/Cosmos-Reason1-7B"
            max_length: Maximum sequence length for text tokenization.
            embedding_concat_strategy: How to combine layer embeddings:
                - "full_concat": Concatenate all 28 layers -> 100,352 dims
                - "mean_pooling": Mean across layers -> 3584 dims
                - "pool_every_n_layers_and_concat": Pool groups then concat
            n_layers_per_group: Number of layers per group for pool_every_n strategy.
        """

        super().__init__()
        self.max_length = max_length
        self.embedding_concat_strategy = embedding_concat_strategy
        self.n_layers_per_group = n_layers_per_group

        # Load processor and tokenizer
        self.processor = hf_load(
            AutoProcessor.from_pretrained,
            model_name,
        )
        self.tokenizer = self.processor.tokenizer

        # Load model (requires transformers >= 4.49.0)
        logger.info(f"Loading Cosmos-Reason1 model from {model_name}")
        self.model = hf_load(
            Qwen2_5_VLForConditionalGeneration.from_pretrained,
            model_name,
            dtype=torch.bfloat16,
            # The reference extracts Reason1 features with FA2; SDPA materially changes the conditioning.
            attn_implementation="flash_attention_2",
            output_hidden_states=True,
        )

        # transformers>=5 dropped the flattened `hidden_size`/`num_hidden_layers` aliases on
        # `Qwen2_5_VLConfig`; the nested `text_config` works on both 4.x and 5.x.
        text_config = getattr(self.model.config, "text_config", self.model.config)
        self.hidden_size = text_config.hidden_size  # 3584 for Qwen2.5-7B
        self.num_layers = text_config.num_hidden_layers  # 28 for Qwen2.5-7B

    def _fixed_precision_overrides(self) -> Dict[str, torch.dtype]:
        # Qwen constructs these in FP32. Casting them to BF16 and back permanently rounds
        # the frequencies, so pin them before preparation casts the rest of the encoder.
        return {
            **super()._fixed_precision_overrides(),
            **{
                name: torch.float32
                for name, _ in self.named_buffers(remove_duplicate=False)
                if name.rsplit(".", 1)[-1] in ("inv_freq", "original_inv_freq")
            },
        }

    @staticmethod
    def mean_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Mean normalize tensor by subtracting mean and dividing by std."""
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (tensor.std(dim=-1, keepdim=True) + 1e-8)

    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text prompts to embeddings using Cosmos-Reason1 style.

        Args:
            text: Single text prompt or list of prompts
        Returns:
            Text embeddings of shape (B, max_length, embedding_dim)
        """
        if isinstance(text, str):
            text = [text]

        input_ids_batch = []
        for prompt in text:
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ]

            formatted = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                return_tensors="pt",
            )
            # transformers 5 returns a BatchEncoding here for multimodal
            # tokenizers, while transformers 4 returned the input-id tensor
            # directly. Normalize both APIs before padding.
            if not isinstance(formatted, torch.Tensor):
                formatted = formatted["input_ids"]

            # Pad or truncate to max_length
            if formatted.shape[1] < self.max_length:
                pad_len = self.max_length - formatted.shape[1]
                padding = torch.full((1, pad_len), self.tokenizer.pad_token_id or 0, dtype=formatted.dtype)
                formatted = torch.cat([formatted, padding], dim=1)
            else:
                formatted = formatted[:, : self.max_length]
            input_ids_batch.append(formatted)

        # Keep the compiled/FSDP-visible module boundary accelerator-only: tokenization
        # happens on CPU, then the completed batch is placed before ``self(...)`` runs.
        input_ids = torch.cat(input_ids_batch, dim=0).to(self.device)
        return self(input_ids)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Forward pass with hidden states
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,  # only hidden_states is used; skips the full-vocabulary lm_head projection
        )
        hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors

        # Skip embedding layer (index 0), normalize and combine layers
        normalized_hidden_states = [self.mean_normalize(hidden_states[i]) for i in range(1, len(hidden_states))]

        if self.embedding_concat_strategy == self.FULL_CONCAT:
            text_embeddings = torch.cat(normalized_hidden_states, dim=-1)
        elif self.embedding_concat_strategy == self.MEAN_POOLING:
            text_embeddings = torch.stack(normalized_hidden_states).mean(dim=0)
        elif self.embedding_concat_strategy == self.POOL_EVERY_N_LAYERS_AND_CONCAT:
            pooled = []
            for i in range(0, len(normalized_hidden_states), self.n_layers_per_group):
                group = normalized_hidden_states[i : i + self.n_layers_per_group]
                pooled.append(torch.stack(group).mean(dim=0))
            text_embeddings = torch.cat(pooled, dim=-1)
        else:
            raise ValueError(f"Invalid embedding_concat_strategy: {self.embedding_concat_strategy}")

        return text_embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension based on concat strategy."""
        if self.embedding_concat_strategy == self.FULL_CONCAT:
            return self.num_layers * self.hidden_size
        elif self.embedding_concat_strategy == self.MEAN_POOLING:
            return self.hidden_size
        elif self.embedding_concat_strategy == self.POOL_EVERY_N_LAYERS_AND_CONCAT:
            n_groups = (self.num_layers + self.n_layers_per_group - 1) // self.n_layers_per_group
            return n_groups * self.hidden_size
        return self.hidden_size


# ---------------------- Main Network Class -----------------------


class CosmosPredict2(FastGenNetwork):
    """
    Cosmos Predict2 main network for text-to-video diffusion.

    This class wraps the DiT model and provides the FastGenNetwork interface
    for training and inference.

    Args:
        max_img_h: Maximum image height
        max_img_w: Maximum image width
        max_frames: Maximum number of frames
        in_channels: Number of input channels (latent channels)
        out_channels: Number of output channels
        patch_spatial: Spatial patch size
        patch_temporal: Temporal patch size
        model_channels: Base model dimension
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        crossattn_emb_channels: Cross-attention embedding dimension
        sac_config: Configuration for selective activation checkpointing
        enable_logvar_linear: Whether to add logvar_linear layer for uncertainty estimation
        net_pred_type: Network prediction type ('flow', 'eps', 'v', 'x0')
        schedule_type: Noise schedule type
    """

    # The Wan2.1 VAE that ``init_vae`` loads.
    latent_compression_factors = (4, 8, 8)

    def __init__(
        self,
        # Model architecture (must be specified per model size)
        model_channels: int = 2048,  # 2B: 2048, 14B: 5120
        num_blocks: int = 28,  # 2B: 28, 14B: 36
        num_heads: int = 16,  # 2B: 16, 14B: 40
        # Resolution and frame settings (latent space dimensions)
        # These are MAXIMUM supported sizes for RoPE positional embeddings.
        # Set higher than training resolution to allow extrapolation at inference time.
        max_img_h: int = 240,  # Supports up to 1920 pixel height (240*8)
        max_img_w: int = 240,  # Supports up to 1920 pixel width (240*8)
        max_frames: int = 128,  # Supports many video frames
        # Input/output channels (latent space)
        in_channels: int = 16,
        out_channels: int = 16,
        # Patch sizes
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        concat_padding_mask: bool = True,
        # Video condition mask for video2world/LVG models (adds 1 channel)
        add_video_condition_mask: bool = True,
        mlp_ratio: float = 4.0,
        # Cross-attention (after projection from Cosmos-Reason1 full_concat)
        crossattn_emb_channels: int = 1024,
        # Positional embeddings
        pos_emb_cls: str = "rope3d",
        # RoPE extrapolation ratios - 3.0 for spatial dimensions for better generalization
        rope_h_extrapolation_ratio: float = 3.0,
        rope_w_extrapolation_ratio: float = 3.0,
        rope_t_extrapolation_ratio: float = 1.0,
        # AdaLN LoRA
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        adaln_lora_scale: float = 1.0,  # Scale factor for adaln_lora values (1.0 = no scaling)
        crossattn_gate_scale: float = 1.0,  # Scale factor for cross-attention gate
        extra_per_block_abs_pos_emb: bool = False,
        # Selective activation checkpointing
        sac_config: Optional[SACConfig] = None,
        # Enable logvar for distillation
        enable_logvar_linear: bool = False,
        # Noise schedule
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        # Cross-attention projection for Cosmos-Reason1 (Qwen2.5-7B: 28 layers * 3584 hidden = 100352)
        use_crossattn_projection: bool = True,
        crossattn_proj_in_channels: int = 100352,  # full_concat: 28 layers * 3584 = 100352
        # Text encoder: HuggingFace model name or local path to Cosmos-Reason1-7B
        text_encoder_model_name: str = "nvidia/Cosmos-Reason1-7B",
        # The native default; the released 2B post-trained checkpoint overrides it with Karras.
        sample_sigma_schedule: str = "linear",
        # WAN fp32 strategy: use fp32 for modulation computation
        use_wan_fp32_strategy: bool = True,
        # FPS for temporal position embeddings
        fps: float = 24.0,
        rope_enable_fps_modulation: bool = True,
        # Video2world (image-to-video) mode settings
        is_video2world: bool = False,
        num_conditioning_frames: int = 1,
        # Skip the transformer's custom weight init (safe only when a checkpoint overwrites all params)
        skip_custom_init: bool = False,
        **schedule_kwargs,
    ):
        # The native linear preset's shifted grid starts above the RF default of 0.999, so the raw-time bound defaults
        # to 1.0 (configs may lower it).
        schedule_kwargs.setdefault("max_t", 1.0)
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)

        # Scale factor for adaln_lora values
        self.adaln_lora_scale = adaln_lora_scale

        # Scale factor for cross-attention gate
        self.crossattn_gate_scale = crossattn_gate_scale

        self.transformer = CosmosPredict2DiT(
            max_img_h=max_img_h,
            max_img_w=max_img_w,
            max_frames=max_frames,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            concat_padding_mask=concat_padding_mask,
            add_video_condition_mask=add_video_condition_mask,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            crossattn_emb_channels=crossattn_emb_channels,
            pos_emb_cls=pos_emb_cls,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            extra_per_block_abs_pos_emb=extra_per_block_abs_pos_emb,
            sac_config=sac_config,
            use_crossattn_projection=use_crossattn_projection,
            crossattn_proj_in_channels=crossattn_proj_in_channels,
            rope_h_extrapolation_ratio=rope_h_extrapolation_ratio,
            rope_w_extrapolation_ratio=rope_w_extrapolation_ratio,
            rope_t_extrapolation_ratio=rope_t_extrapolation_ratio,
            rope_enable_fps_modulation=rope_enable_fps_modulation,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
            skip_custom_init=skip_custom_init,
        )

        # Add logvar_linear layer for uncertainty estimation (used by some loss functions)
        if enable_logvar_linear:
            self.transformer.logvar_linear = nn.Linear(model_channels, 1)

        # Store text encoder model name/path for init_preprocessors
        self.text_encoder_model_name = text_encoder_model_name
        if sample_sigma_schedule not in ("karras", "linear"):
            raise ValueError("sample_sigma_schedule must be 'karras' or 'linear'")
        self.sample_sigma_schedule = sample_sigma_schedule

        # Video2world attributes
        self.is_video2world = is_video2world
        self.num_conditioning_frames = num_conditioning_frames

        # Default FPS for temporal position embeddings
        self.fps = fps

        # Initialize the sample scheduler for inference later (otherwise it causes issues with Meta init)
        self.sample_scheduler = None

    def init_preprocessors(self):
        """Initialize text encoder and video encoder.

        The text encoder uses Cosmos-Reason1 (based on Qwen2.5-VL-7B).
        You can specify either:
        - A HuggingFace model name: "nvidia/Cosmos-Reason1-7B"
        - A local path: "/path/to/Cosmos-Reason1-7B"

        Skips initialization if preprocessors already exist.
        """
        if "text_encoder" not in self.preprocessors:
            self.init_text_encoder()
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_text_encoder(self):
        """Initialize text encoder (Cosmos-Reason1)."""
        self.preprocessors["text_encoder"] = CosmosPredict2TextEncoder(
            model_name=self.text_encoder_model_name,
        )

    def init_vae(self):
        """Initialize video encoder (Wan VAE)."""
        # Unlike native Wan's default FP32 path, the native Cosmos tokenizer constructs this VAE
        # with dtype=BF16 and is_amp=False. Its weights, inputs, and internal mean/std normalization
        # therefore stay in BF16 without autocast.
        self.preprocessors["vae"] = WanVideoEncoder(
            model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            fixed_precision_model=torch.bfloat16,
        )

    def build_conditioning(self, data, input_shape=None, precision=None):
        """Build the conditioning fields for CosmosPredict2.

        Two regimes, gated by ``self.is_video2world``:

        * **T2V (default, video2world=False)** — return the ordinary text
          branches resolved by the inherited builder.
        * **V2W (video2world=True)** — slice the first
          ``self.num_conditioning_frames`` pixel frames, VAE-encode them
          deterministically (``mode="argmax"``), and build a binary
          condition mask in latent space marking those frames, both in the
          state ``precision``.  Latent dims come from ``input_shape``.

        V2W requires both ``data["real_raw"]`` (pixel) and ``input_shape``;
        raises ``ValueError`` otherwise.  When the caller supplies fewer
        pixel frames than ``num_conditioning_frames`` (typical at
        inference: a single seed image), the input is replicated along T
        to fill — the inference script can therefore always load 1 frame.

        V2W enriches the resolved positive branch and a non-``None`` resolved
        negative branch without mutating ``data``.
        """
        out = super().build_conditioning(data, input_shape=input_shape, precision=precision)
        # Complete conditions are builder outputs and are never prepared again, in T2V mode too.
        validate_conditioning(out, required_fields=("condition",) if self.is_video2world else ())
        if not self.is_video2world:
            return out

        text_embeds = out.get("condition")

        real_raw = data.get("real_raw")
        if real_raw is None:
            raise ValueError(
                f"{type(self).__name__} V2W requires data['real_raw'] (pixel-space). " "Use a pixel-space dataloader."
            )
        if input_shape is None:
            raise ValueError(
                "CosmosPredict2 V2W requires input_shape to derive latent "
                "spatial / temporal dims for the condition mask."
            )

        n_cond = self.num_conditioning_frames
        n_missing = n_cond - real_raw.shape[2]
        if n_missing > 0:
            # Prepend repeats of the first frame so the user-supplied frames
            # land at the *end* of the conditioning window — the model sees
            # a held initial state evolving into the user clip rather than a
            # backward jump.
            pad = real_raw[:, :, :1].expand(-1, -1, n_missing, -1, -1)
            real_raw = torch.cat([pad, real_raw], dim=2)
        cond_frames = real_raw[:, :, :n_cond]
        vae = self.get_preprocessor("vae", "encode the V2W conditioning frames")
        cond_frames = vae.encode(cond_frames, mode="argmax").to(dtype=precision)

        _, t_latent, h_latent, w_latent = input_shape
        bsz, _, t_cond_latent, _, _ = cond_frames.shape
        condition_mask = torch.zeros(bsz, 1, t_latent, h_latent, w_latent, device=cond_frames.device, dtype=torch.bool)
        condition_mask[:, :, :t_cond_latent] = True

        extras = {
            "conditioning_latents": cond_frames,
            "condition_mask": condition_mask,
        }
        out["condition"] = {"text_embeds": text_embeds, **extras}
        neg_text_embeds = out.get("neg_condition")
        if neg_text_embeds is not None:
            out["neg_condition"] = {"text_embeds": neg_text_embeds, **extras}
        return out

    @staticmethod
    def remap_checkpoint_keys(state_dict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Remap checkpoint keys from original Cosmos format to our format.

        Handles the following transformations:
        1. `net.xxx` -> `transformer.xxx` (from Cosmos DCP training checkpoints)
        2. Raw keys like `blocks.xxx` -> `transformer.blocks.xxx` (from HuggingFace release)
        3. `._checkpoint_wrapped_module` -> removed (activation checkpointing wrapper)
        4. `_extra_state` keys -> skipped (Transformer Engine FP8 state)
        5. `accum_*` keys -> skipped (training accumulators)

        Args:
            state_dict: The original state dict from the checkpoint.

        Returns:
            A new state dict with remapped keys.
        """
        remapped = {}
        skipped_keys = []

        # Detect checkpoint format by checking key prefixes
        has_net_prefix = any(k.startswith("net.") for k in state_dict.keys())
        has_transformer_prefix = any(k.startswith("transformer.") for k in state_dict.keys())

        # Keys that indicate raw Cosmos checkpoint (no net. or transformer. prefix)
        cosmos_raw_keys = {
            "blocks.",
            "x_embedder.",
            "t_embedder.",
            "final_layer.",
            "pos_embedder.",
            "t_embedding_norm.",
        }
        is_raw_cosmos = any(any(k.startswith(prefix) for prefix in cosmos_raw_keys) for k in state_dict.keys())

        for key, value in state_dict.items():
            # Skip Transformer Engine extra state (FP8)
            if "_extra_state" in key:
                skipped_keys.append(key)
                continue

            # Skip training accumulators
            if ".accum_" in key or key.startswith("accum_"):
                skipped_keys.append(key)
                continue

            new_key = key

            # Remove _checkpoint_wrapped_module from block layers
            new_key = new_key.replace("._checkpoint_wrapped_module", "")

            # Remap based on checkpoint format
            if new_key.startswith("net."):
                # From Cosmos DCP/training checkpoint: net.xxx -> transformer.xxx
                new_key = "transformer." + new_key[4:]
            elif not new_key.startswith("transformer.") and is_raw_cosmos:
                # Raw Cosmos checkpoint from HuggingFace: xxx -> transformer.xxx
                new_key = "transformer." + new_key

            remapped[new_key] = value

        if skipped_keys:
            logger.debug(f"Skipped {len(skipped_keys)} keys during checkpoint loading")

        logger.info(
            f"Checkpoint key remapping: has_net_prefix={has_net_prefix}, "
            f"has_transformer_prefix={has_transformer_prefix}, "
            f"is_raw_cosmos={is_raw_cosmos}"
        )

        return remapped

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        """Load state dict with automatic key remapping for Cosmos checkpoints.

        This method automatically detects and remaps keys from the original
        Cosmos checkpoint format to our model format.

        Handles:
        - `net.xxx` -> `transformer.xxx` prefix remapping (DCP checkpoints)
        - Raw keys -> `transformer.xxx` (HuggingFace release checkpoints)
        - `_checkpoint_wrapped_module` removal
        - Skipping `_extra_state` (Transformer Engine) and `accum_*` (training) keys

        Args:
            state_dict: The state dict to load.
            strict: Whether to strictly enforce that the keys in state_dict match.
            assign: Whether to assign instead of copy (PyTorch 2.1+).

        Returns:
            NamedTuple with missing_keys and unexpected_keys.
        """
        # Always try remapping - it will detect the checkpoint format
        state_dict = self.remap_checkpoint_keys(state_dict)

        return super().load_state_dict(state_dict, **kwargs)

    def fully_shard(self, **kwargs):
        """Shard ``self.transformer``; see ``FastGenNetwork.fully_shard``.

        Cosmos already applies selective activation checkpointing through ``ptd_checkpoint_wrapper``,
        so no checkpoint conversion happens here.
        """

        # First, shard each transformer block for fine-grained parallelism
        for block in self.transformer.blocks:
            fully_shard(block, **kwargs)

        # Then shard the entire transformer
        fully_shard(self.transformer, **kwargs)

    def _fixed_precision_overrides(self) -> Dict[str, torch.dtype]:
        """Pin the RoPE tables to FP32 before preparation casts the rest of the DiT.

        The reference computes the exponent tables in FP32 and FSDP mixed precision never casts buffers.
        """
        prefix = "transformer.pos_embedder."
        return {
            **super()._fixed_precision_overrides(),
            **{name: torch.float32 for name in module_state_names(self) if name.startswith(prefix)},
        }

    def _get_conditioning_tensors(
        self, x: torch.Tensor, condition: Optional[Any]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Extract and prepare conditioning tensors for video2world mode.

        Args:
            x: Reference tensor for shape and device
            condition: The condition dict containing conditioning_latents and condition_mask

        Returns:
            Tuple of (conditioning_latents_full, condition_mask, condition_mask_C) or None
        """
        if not isinstance(condition, Mapping):
            return None

        conditioning_latents = condition.get("conditioning_latents")
        condition_mask = condition.get("condition_mask")

        if (conditioning_latents is None) != (condition_mask is None):
            raise ValueError("conditioning_latents and condition_mask must be provided together")
        if conditioning_latents is None:
            return None

        assert_same(device=True, x=x, conditioning_latents=conditioning_latents)
        assert_same(dtype=torch.bool, device=x.device, condition_mask=condition_mask)

        # Pad conditioning_latents to full temporal dimension
        T_cond = conditioning_latents.shape[2]
        conditioning_latents_full = torch.zeros_like(x)
        conditioning_latents_full[:, :, :T_cond] = conditioning_latents

        # Expand mask to channel dimension [B, 1, T, H, W] -> [B, C, T, H, W]
        condition_mask_C = condition_mask.expand(-1, x.shape[1], -1, -1, -1)

        return conditioning_latents_full, condition_mask, condition_mask_C

    def preserve_conditioning(self, x: torch.Tensor, condition: Optional[Any]) -> torch.Tensor:
        """Replace the video2world conditioning frames of ``x`` with the clean latents.

        ``forward()`` applies this to its input, so the DiT sees clean conditioning
        frames while samplers keep evolving the raw latent state.

        Args:
            x: Noisy latent state, shape (B, C, T, H, W)
            condition: The condition dict containing conditioning_latents and condition_mask

        Returns:
            ``x`` with its conditioning frames replaced
        """
        tensors = self._get_conditioning_tensors(x, condition)
        if tensors is None:
            return x

        conditioning_latents_full, _, condition_mask_C = tensors
        return torch.where(condition_mask_C, conditioning_latents_full, x)

    def _conditioned_inputs(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: Any, conditional_frame_timestep: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The transformer's input and timesteps: Video2World frames clean, with one timestep per frame.

        Conditioning frames take ``conditional_frame_timestep`` unless it is negative. Without a
        Video2World condition, these are the caller's ``x_t`` and ``t``.
        """
        conditioning_latents = condition.get("conditioning_latents") if isinstance(condition, Mapping) else None
        condition_mask = condition.get("condition_mask") if isinstance(condition, Mapping) else None
        if (conditioning_latents is None) != (condition_mask is None):
            raise ValueError("conditioning_latents and condition_mask must be provided together")
        if condition_mask is None:
            return x_t, t
        model_input = self.preserve_conditioning(x_t, condition)
        B, _, T = x_t.shape[:3]
        if t.ndim == 1:
            t_expanded = t.unsqueeze(1).expand(B, T)
        elif t.ndim == 2:
            t_expanded = t.expand(B, T)
        else:
            raise ValueError(f"Expected t with rank 1 or 2, got shape {tuple(t.shape)}")
        if conditional_frame_timestep >= 0:
            mask_B_T = condition_mask[:, 0, :, 0, 0]  # (B, T)
            t_expanded = torch.where(mask_B_T, torch.full_like(t_expanded, conditional_frame_timestep), t_expanded)
        return model_input, t_expanded

    def _convert_output(self, prediction, x_t, t, condition, fwd_pred_type, conditional_frame_timestep=1e-4, **context):
        """Convert against the input and per-frame timesteps the transformer saw.

        These equal the caller's ``x_t`` and ``t`` on the generated frames; ``_condition_output`` then re-pins the
        Video2World frames against the caller's ``x_t`` and ``t`` in every prediction type. A ``(B, T)`` time is
        reshaped to ``(B, 1, T, 1, 1)`` because the schedule's ``expand_like`` only pads trailing dims.
        """
        model_input, transformer_t = self._conditioned_inputs(x_t, t, condition, conditional_frame_timestep)
        if transformer_t.ndim == 2:
            transformer_t = transformer_t.reshape(x_t.shape[0], 1, -1, 1, 1)
        return super()._convert_output(prediction, model_input, transformer_t, condition, fwd_pred_type)

    def _condition_output(self, prediction, x_t, t, condition, fwd_pred_type, **context):
        """Pin clean Video2World frames in the requested prediction space after guidance.

        The reference's ``denoise_replace_gt_frames`` flow is ``noise - gt_frames``.
        Converting the clean target against the caller's raw state gives that same
        constraint in every prediction domain. At sigma zero a non-x0 prediction
        cannot carry it, which raises at the Video2World frames.
        """
        tensors = self._get_conditioning_tensors(x_t, condition)
        if tensors is None:
            return prediction
        clean, _, mask = tensors
        if fwd_pred_type != "x0":
            B, _, T = x_t.shape[:3]
            t_out = t if t.ndim == 1 else t.expand(B, T).reshape(B, 1, T, 1, 1)
            noised = self.noise_scheduler.sigma(t_out) > 0
            assert_tensor(
                (expand_like(noised, prediction) | ~mask).all(),
                "A non-x0 prediction cannot carry the output constraint at sigma = 0; request fwd_pred_type='x0' "
                "or apply_output_conditioning=False there",
                error_type=ValueError,
            )
            # Other frames keep the prediction, so their target only needs to be finite at sigma = 0.
            t_safe = torch.where(noised, t_out, torch.full_like(t_out, self.noise_scheduler.max_t))
            clean = self.noise_scheduler.convert_model_output(
                x_t, clean, t_safe, src_pred_type="x0", target_pred_type=fwd_pred_type, output_dtype=prediction.dtype
            )
        return torch.where(mask, clean.to(dtype=prediction.dtype), prediction)

    @staticmethod
    def default_guidance_cfg() -> GuidanceConfig:
        """Default guidance recipe: CFG only at scale 5.0.

        Experiments may override via ``config.model.guidance_cfg``.
        """
        g = GuidanceConfig()
        g.cfg.scale = 5.0
        return g

    def sample_scheduler_and_grid(
        self, sample_steps: int, flow_shift: float = 5.0, device: Optional[torch.device] = None
    ) -> Tuple[UniPCMultistepScheduler, torch.Tensor]:
        """Set UniPC to the native preset and return the times to query the network at.

        Karras evaluates ``sample_steps + 1`` times; shifted linear evaluates ``sample_steps``
        times, shifted by ``flow_shift``, which the Karras preset ignores like the reference. The
        network is queried at the FP32 sigmas the solver integrates, in ``t_precision``: the time
        the reference trains it on. The reference sampler instead truncates them to integer
        timesteps, which offsets the network's time from the solver's by up to 0.001.
        """
        if not isinstance(sample_steps, int) or isinstance(sample_steps, bool) or sample_steps <= 0:
            raise ValueError("sample_steps must be a positive integer")
        if not np.isfinite(flow_shift) or flow_shift <= 0:
            raise ValueError("flow_shift must be finite and positive")
        if self.sample_sigma_schedule == "karras":
            # The native scheduler interpolates down from sigma_max, so the bounds run high to low.
            sigmas = polynomial_spacing(np.arange(sample_steps + 1) / sample_steps, 7, 200.0, 0.01)
            sigmas = sigmas / (1 + sigmas)
        else:
            # The native scheduler stores its training endpoints in FP32 before interpolating in NumPy.
            sigmas = np.linspace(float(np.float32(1 - 1 / 1000)), 0.0, sample_steps + 1)[:-1]
            sigmas = time_shift(sigmas, flow_shift)
        if self.sample_scheduler is None:
            self.sample_scheduler = UniPCMultistepScheduler(
                num_train_timesteps=1000,
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                flow_shift=1.0,  # The supplied sigmas already include the native shift.
            )
        self.sample_scheduler.set_timesteps(num_inference_steps=len(sigmas), sigmas=sigmas, device=device)
        # From 1058 steps the leading int64 timestep repeats; UniPC would resolve the first step to its second
        # occurrence, skip a sigma, and step past the grid's end.
        self.sample_scheduler.set_begin_index(0)
        t_grid = self.noise_scheduler.safe_clamp(
            self.sample_scheduler.sigmas[:-1].to(device=device, dtype=self.noise_scheduler.t_precision),
            min=self.noise_scheduler.min_t,
            max=self.noise_scheduler.max_t,
            warn=True,
        )
        return self.sample_scheduler, t_grid

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        sample_steps: int = 50,
        sample_type: str = "ode",
        shift: float = 5.0,
        fps: Optional[torch.Tensor] = None,
        conditional_frame_timestep: float = 1e-4,
        denoise_replace_gt_frames: bool = True,
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Multistep sampling using the FlowUniPC method.

        This implementation follows the official Cosmos inference, except that, as in training,
        it starts from noise scaled to the first sigma rather than from unscaled noise, and queries
        the model at the solver's sigmas rather than at their truncated integer timesteps. It
        supports both text2world and video2world modes:
        - NO preconditioning - model receives the raw input and the solver's sigma as its time
        - Model outputs velocity directly
        - CFG is applied on velocity predictions
        - Scheduler handles the integration

        For video2world mode (when ``condition`` carries conditioning latents and their mask):
        - The frames selected by ``condition["condition_mask"]`` are replaced with conditioning latents
        - Conditioning frames are presented to the transformer with
          `conditional_frame_timestep` (1e-4 by default, as in forward())
        - Predicted velocity for conditioning frames is replaced with the constant velocity
          that carries the initial latents onto the conditioning frames

        Args:
            noise: The noisy latents to start from, shape (B, C, T, H, W).
            condition: The conditioning embeddings, or a dict with 'text_embeds' and, for video2world
                mode, both 'conditioning_latents' of shape (B, C, T_cond, H, W) and a boolean
                'condition_mask' of shape (B, 1, T, H, W) that selects only frames before T_cond.
            neg_condition: The negative/unconditional embeddings for CFG.
            sample_steps: Native preset step count. Karras evaluates ``sample_steps + 1``
                timesteps; shifted linear evaluates ``sample_steps``.
            sample_type: Integration policy; only ``"ode"`` is supported.
            shift: Shift of the linear sigma preset (the reference's 5.0 by default); the Karras
                preset ignores it.
            fps: Frames per second tensor for temporal conditioning.
            conditional_frame_timestep: Normalized conditioning-frame timestep
                forwarded to ``forward()`` (default 1e-4).
            denoise_replace_gt_frames: Whether to replace velocity for conditioning frames
                with the constant velocity ``(initial_latents - conditioning_latents) / t_init``.
                False retains their guided model predictions; input conditioning still applies. Default True.
            guidance_cfg: Composite guidance config; ``None`` →
                ``default_guidance_cfg()`` (CFG at 5.0).

        Returns:
            The denoised sample tensor.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )

        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()

        scheduler, t_grid = self.sample_scheduler_and_grid(sample_steps, flow_shift=shift, device=noise.device)

        # Unlike native FlowUniPC, which starts from unscaled noise, start at the first query time's noise level.
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])
        # The conditioning-frame velocity must span the solver's first sigma; t_grid[0] matches it unless clamped.
        t_init = scheduler.sigmas[0].to(device=noise.device, dtype=self.noise_scheduler.t_precision)

        # Default fps for video generation (use self.fps which can be set via model.net.fps)
        if fps is None:
            fps = torch.full((latents.shape[0],), self.fps, device=latents.device)

        # Video2world conditions are dicts that carry the conditioning latents and mask with the text embeddings
        conditioning_latents = None
        condition_mask = None
        if isinstance(condition, Mapping):
            conditioning_latents = condition.get("conditioning_latents")
            condition_mask = condition.get("condition_mask")
            # Extract text embeddings for the actual forward pass
            condition = condition["text_embeds"]
        if isinstance(neg_condition, Mapping):
            neg_condition = neg_condition["text_embeds"]
        if (conditioning_latents is None) != (condition_mask is None):
            raise ValueError("conditioning_latents and condition_mask must be provided together")

        # Video2world setup: prepare the conditioning mask and latents
        video2world_mode = conditioning_latents is not None
        conditioning_latents_full = None
        condition_mask_C = None
        initial_latents = None

        if video2world_mode:
            T = latents.shape[2]
            T_cond = conditioning_latents.shape[2]

            # Validate shapes
            assert T_cond <= T, f"Conditioning frames ({T_cond}) cannot exceed total frames ({T})"
            assert_tensor(
                ~condition_mask[:, :, T_cond:].any(),
                f"condition_mask selects frames beyond the {T_cond} conditioning_latents frames",
            )

            # Build condition dict for _get_conditioning_tensors
            v2w_condition = {"conditioning_latents": conditioning_latents, "condition_mask": condition_mask}
            conditioning_latents_full, condition_mask, condition_mask_C = self._get_conditioning_tensors(
                latents, v2w_condition
            )

            # Store the initial latents for velocity replacement
            initial_latents = latents.clone()

        for sched_step, t in tqdm(zip(scheduler.timesteps, t_grid, strict=True), total=len(t_grid), desc="Sampling"):
            t = t.expand(latents.shape[0])

            if video2world_mode:
                # forward() replaces the conditioning frames with the clean latents and presents
                # them with `conditional_frame_timestep`; guidance pins their prediction against
                # the raw latent state that UniPC evolves below. No t-masking is done here.
                cond_with_mask = {
                    "text_embeds": condition,
                    "conditioning_latents": conditioning_latents,
                    "condition_mask": condition_mask,
                }
                neg_cond_with_mask = {
                    "text_embeds": neg_condition,
                    "conditioning_latents": conditioning_latents,
                    "condition_mask": condition_mask,
                }
            else:
                cond_with_mask = condition
                neg_cond_with_mask = neg_condition

            # Forward pass with guidance. Pass the scalar timestep `t`; forward() handles
            # the conditioning-frame timestep masking internally (using
            # `conditional_frame_timestep` for the clean frames).
            velocity_pred = predict_with_guidance(
                self,
                latents,
                t,
                guidance_cfg=guidance_cfg,
                condition=cond_with_mask,
                neg_condition=neg_cond_with_mask,
                fwd_pred_type="flow",
                fps=fps,
                conditional_frame_timestep=conditional_frame_timestep,
                apply_output_conditioning=denoise_replace_gt_frames,
            )

            # Replace velocity for conditioning frames with the constant velocity that carries
            # the initial latents onto the conditioning frames over the interval [0, t_init].
            if video2world_mode and denoise_replace_gt_frames:
                gt_velocity = (initial_latents - conditioning_latents_full) / t_init
                velocity_pred = torch.where(condition_mask_C, gt_velocity, velocity_pred)

            # UniPC evolves the raw latents; forward() feeds the DiT clean conditioning frames.
            latents = scheduler.step(velocity_pred, sched_step, latents, return_dict=False)[0]

        return latents

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Union[torch.Tensor, Dict[str, Any]] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        unpatchify_features: bool = True,
        fwd_pred_type: Optional[str] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        skip_layers: Optional[List[int]] = None,
        conditional_frame_timestep: float = 1e-4,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the Cosmos Predict2 diffusion model.

        Supports feature extraction and log variance output for training
        with auxiliary losses (e.g., discriminator, distillation).

        Args:
            x_t: Noisy input tensor of shape (B, C, T, H, W)
            t: Timestep tensor of shape (B,) or (B, T)
            condition: Cross-attention conditioning. Can be:
                - torch.Tensor of shape (B, L, D) for text embeddings
                - Dict with keys: 'text_embeds' (required), and optionally
                  'conditioning_latents', 'condition_mask' for video2world mode
            return_features_early: If True, return features once all requested
                indices are collected without completing the full forward pass
            feature_indices: Set of block indices to extract features from.
                Any provided set returns features along with the output, including
                an empty feature list for an empty set. None returns only the output.
            return_logvar: If True, return log variance along with the output, one (B, 1) value per sample
            unpatchify_features: If True, reshape features to (B, C, T, H, W) format
            fwd_pred_type: Override prediction type ('x0', 'eps', 'v', 'flow')
            fps: Frames per second tensor
            padding_mask: Padding mask tensor
            skip_layers: List of block indices to skip during forward pass
            conditional_frame_timestep: Normalized timestep for conditioning frames (1e-4 by default,
                the reference post-training's raw timestep 0.1 after its 0.001 network scale).
                A negative value disables the special conditioning-frame timestep.
            apply_output_conditioning: Re-pin the Video2World frames after the conversion.
            output_dtype: ``"same"`` rounds the prediction to the state dtype; ``None`` keeps it in high precision.
        Returns:
            Depending on the arguments:
            - torch.Tensor: Model output if feature_indices is None and logvar is not requested
            - List[torch.Tensor]: Features if return_features_early=True
            - [output, features]: If feature_indices is not None
            - (output, logvar): If return_logvar=True
            - ([output, features], logvar): If both features and logvar requested
        """
        self._validate_forward_kwargs(
            fwd_kwargs,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            skip_layers=skip_layers,
            supports_skip_layers=True,
            num_features=len(self.transformer.blocks),
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        if fps is None:
            fps = torch.full((x_t.shape[0],), self.fps, device=x_t.device)

        # Handle dict-style condition for video2world mode
        # Extract: text_embeds and condition_mask (used as video_condition_mask)
        condition_mask = None
        if isinstance(condition, Mapping):
            text_embeds = condition["text_embeds"]
            condition_mask = condition.get("condition_mask")  # Used as video_condition_mask for transformer
        else:
            text_embeds = condition
        model_input, transformer_t = self._conditioned_inputs(x_t, t, condition, conditional_frame_timestep)

        model_input_core = self.cast_to_model_precision(model_input)
        assert_same(dtype=self.precision_model, device=model_input_core.device, text_embeds=text_embeds)
        # NVIDIA Predict2.5 casts floating batch inputs, including fps, to the model's
        # execution dtype before the DiT. Mirror that boundary here for every fps source dtype.
        fps = fps.to(dtype=model_input_core.dtype)
        # Keep normalized timesteps in FP32 through the sinusoidal projection.
        transformer_t_core = transformer_t.to(dtype=torch.float32)
        model_outputs = self.transformer(
            x_B_C_T_H_W=model_input_core,
            timesteps_B_T=transformer_t_core,
            crossattn_emb=text_embeds,
            fps=fps,
            padding_mask=padding_mask,
            video_condition_mask=condition_mask,  # From condition dict for video2world
            skip_layers=skip_layers,
            feature_indices=feature_indices,
            return_features_early=return_features_early,
            return_logvar=return_logvar,
            adaln_lora_scale=self.adaln_lora_scale,
            crossattn_gate_scale=self.crossattn_gate_scale,
        )

        # Handle early feature return (no output conversion needed)
        if return_features_early:
            assert len(model_outputs) == len(feature_indices)
            if unpatchify_features:
                model_outputs = [rearrange(f, "B T H W D -> B D T H W") for f in model_outputs]
            else:
                model_outputs = [rearrange(f, "B T H W D -> B (T H W) D") for f in model_outputs]
            return model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        output_args = (x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning)
        if feature_indices is None:
            assert isinstance(out, torch.Tensor)
            out = self.transform_output(out, *output_args, conditional_frame_timestep=conditional_frame_timestep)
        else:
            assert isinstance(out, list)
            out[0] = self.transform_output(out[0], *output_args, conditional_frame_timestep=conditional_frame_timestep)
            # Unpatchify features to match expected output format
            if unpatchify_features:
                out[1] = [rearrange(f, "B T H W D -> B D T H W") for f in out[1]]
            else:
                out[1] = [rearrange(f, "B T H W D -> B (T H W) D") for f in out[1]]

        if return_logvar:
            return out, logvar.to(dtype=x_t.dtype)
        return out
