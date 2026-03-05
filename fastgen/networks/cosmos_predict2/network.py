# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Main network classes for Cosmos Predict2.

This module contains the main network classes:
- CosmosPredict2DiT: The core DiT transformer model
- CosmosPredict2VideoEncoder: Video VAE for encode/decode
- CosmosPredict2TextEncoder: Text encoder wrapper
- CosmosPredict2: Main network class inheriting from FastGenNetwork
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union, Mapping
import os
from tqdm.auto import tqdm
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from fastgen.networks.Wan import WanVideoEncoder
from diffusers import UniPCMultistepScheduler

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES

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
from fastgen.utils.basic_utils import str2bool


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
        use_wan_fp32_strategy: bool = True,  # Use fp32 for modulation computation and RoPE (numerical stability)
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
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
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
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)

        # Cross-attention projection for Cosmos-Reason1 embeddings
        # Projects from 100,352 dims (full_concat: 28 layers * 3584) to crossattn_emb_channels (1024)
        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

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
            print(
                "Warning: torch.distributed checkpoint_wrapper not available. "
                "Selective activation checkpointing disabled."
            )
            return self

        context_fn = sac_config.get_context_fn()
        if context_fn is None and sac_config.mode != CheckpointMode.NONE:
            # Fall back to basic checkpointing without selective policy
            print(
                f"Warning: Selective checkpoint contexts not available for mode {sac_config.mode}. "
                "Using basic block-wise checkpointing."
            )
            context_fn = None

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
                ).type_as(x_B_C_T_H_W)
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
            return_logvar: If True, return log variance for uncertainty estimation
            adaln_lora_scale: Scale factor for AdaLN LoRA (numerical stability)
            crossattn_gate_scale: Scale factor for cross-attention gating

        Returns:
            Depending on arguments:
            - torch.Tensor: Model output if no features or logvar requested
            - List[torch.Tensor]: Features if return_features_early=True
            - [output, features]: If feature_indices is non-empty
            - (output, logvar): If return_logvar=True
        """
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            video_condition_mask=video_condition_mask,
        )

        # Timestep embedding (in fp32 if use_wan_fp32_strategy for numerical stability)
        with torch.amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)

            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # Scale adaln_lora for numerical stability
        if adaln_lora_B_T_3D is not None and adaln_lora_scale != 1.0:
            adaln_lora_B_T_3D = adaln_lora_B_T_3D * adaln_lora_scale

        # Apply cross-attention projection for Cosmos-Reason1 embeddings
        if self.use_crossattn_projection and hasattr(self, "crossattn_proj"):
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if extra_pos_emb is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb.shape

        # Block forward with feature extraction
        features = []
        for idx, block in enumerate(self.blocks):
            if skip_layers is not None and idx in skip_layers:
                continue

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
                features.append(x_B_T_H_W_D.clone())

            # Early exit if we have all requested features
            if return_features_early and len(features) == len(feature_indices):
                return features

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        output = self.unpatchify(x_B_T_H_W_O)

        # Prepare output based on requested returns
        if len(feature_indices) == 0:
            out = output
        else:
            out = [output, features]

        if return_logvar:
            if not hasattr(self, "logvar_linear"):
                raise RuntimeError(
                    "logvar_linear layer is required when return_logvar=True. "
                    "Set enable_logvar_linear=True in model config."
                )
            logvar = self.logvar_linear(t_embedding_B_T_D)
            return out, logvar

        return out


# ---------------------- Text Encoder -----------------------


class CosmosPredict2TextEncoder:
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
        device: str = "cuda",
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
            device: Device to load the model on.
            embedding_concat_strategy: How to combine layer embeddings:
                - "full_concat": Concatenate all 28 layers -> 100,352 dims
                - "mean_pooling": Mean across layers -> 3584 dims
                - "pool_every_n_layers_and_concat": Pool groups then concat
            n_layers_per_group: Number of layers per group for pool_every_n strategy.
        """

        self.max_length = max_length
        self.device = device
        self.embedding_concat_strategy = embedding_concat_strategy
        self.n_layers_per_group = n_layers_per_group

        # Check if using local path
        local_files_only = os.path.isdir(model_name) or str2bool(os.getenv("LOCAL_FILES_ONLY", "false"))

        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=os.getenv("HF_HOME", None),
            local_files_only=local_files_only,
        )
        self.tokenizer = self.processor.tokenizer

        # Load model (requires transformers >= 4.49.0)
        logger.info(f"Loading Cosmos-Reason1 model from {model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=os.getenv("HF_HOME", None),
            local_files_only=local_files_only,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
        )

        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        # Store model config
        self.hidden_size = self.model.config.hidden_size  # 3584 for Qwen2.5-7B
        self.num_layers = self.model.config.num_hidden_layers  # 28 for Qwen2.5-7B

    @staticmethod
    def mean_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Mean normalize tensor by subtracting mean and dividing by std."""
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (tensor.std(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def encode(self, text: Union[str, List[str]], precision: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """
        Encode text prompts to embeddings using Cosmos-Reason1 style.

        Args:
            text: Single text prompt or list of prompts
            precision: Output dtype

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

            # Pad or truncate to max_length
            if formatted.shape[1] < self.max_length:
                pad_len = self.max_length - formatted.shape[1]
                padding = torch.full((1, pad_len), self.tokenizer.pad_token_id or 0, dtype=formatted.dtype)
                formatted = torch.cat([formatted, padding], dim=1)
            else:
                formatted = formatted[:, : self.max_length]
            input_ids_batch.append(formatted)

        input_ids = torch.cat(input_ids_batch, dim=0).to(self.device)

        # Forward pass with hidden states
        outputs = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
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

        return text_embeddings.to(dtype=precision)

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

    def to(self, *args, **kwargs):
        """Move model to device."""
        self.model.to(*args, **kwargs)
        if args and isinstance(args[0], (str, torch.device)):
            self.device = args[0]
        if "device" in kwargs:
            self.device = kwargs["device"]
        return self


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
        enable_logvar_linear: bool = True,
        # Noise schedule
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        # Cross-attention projection for Cosmos-Reason1 (Qwen2.5-7B: 28 layers * 3584 hidden = 100352)
        use_crossattn_projection: bool = True,
        crossattn_proj_in_channels: int = 100352,  # full_concat: 28 layers * 3584 = 100352
        # Text encoder: HuggingFace model name or local path to Cosmos-Reason1-7B
        text_encoder_model_name: str = "nvidia/Cosmos-Reason1-7B",
        # WAN fp32 strategy: use fp32 for RoPE and modulation computation
        use_wan_fp32_strategy: bool = True,
        # FPS for temporal position embeddings
        fps: float = 24.0,
        # Video2world (image-to-video) mode settings
        is_video2world: bool = False,
        num_conditioning_frames: int = 1,
        **model_kwargs,
    ):
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        # Scale factor for adaln_lora values
        self.adaln_lora_scale = adaln_lora_scale

        # Scale factor for cross-attention gate
        # The checkpoint has small gate values (~0.09) which weakens text conditioning
        # Setting this to 10.0 can improve text conditioning from ~1% to ~15%
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
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        # Add logvar_linear layer for uncertainty estimation (used by some loss functions)
        if enable_logvar_linear:
            self.transformer.logvar_linear = nn.Linear(model_channels, 1)

        # Store text encoder model name/path for init_preprocessors
        self.text_encoder_model_name = text_encoder_model_name

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
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self):
        """Initialize text encoder (Cosmos-Reason1)."""
        self.text_encoder = CosmosPredict2TextEncoder(
            model_name=self.text_encoder_model_name,
        )

    def init_vae(self):
        """Initialize video encoder (Wan VAE)."""
        self.vae = WanVideoEncoder(model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

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

    def to(self, *args, **kwargs):
        """Move model to device."""
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae") and self.vae is not None:
            self.vae.to(*args, **kwargs)
        return self

    def reset_parameters(self):
        """Reinitialize parameters and non-persistent buffers for FSDP meta device initialization.

        This method is called after to_empty() when using FSDP2 with meta device initialization.
        It reinitializes:
        1. All Linear layer weights and biases
        2. Embedding layers (patch embedder, timestep embedder)
        3. Output layers (final_layer)
        4. Non-persistent buffers (RoPE, etc.)

        Example usage in FSDP initialization:
            model.to_empty(device=torch.cuda.current_device())
            model.reset_parameters()  # Reinitialize weights and buffers
            set_model_state_dict(model, state_dict, options=options)
        """
        # Initialize all Linear layers with Xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize patch embedder
        if hasattr(self.transformer, "x_embedder"):
            self.transformer.x_embedder.init_weights()

        # Initialize timestep embedder
        if hasattr(self.transformer, "t_embedder"):
            # t_embedder is a Sequential: [nn.Linear, TimestepEmbedder, nn.Linear]
            # TimestepEmbedder has its own init_weights method
            for submodule in self.transformer.t_embedder.modules():
                if hasattr(submodule, "init_weights"):
                    submodule.init_weights()

        # Initialize final layer
        if hasattr(self.transformer, "final_layer"):
            self.transformer.final_layer.init_weights()

        # Initialize cross-attention projection if it exists
        if hasattr(self.transformer, "crossattn_proj"):
            for m in self.transformer.crossattn_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Reinitialize RoPE buffers (non-persistent buffers must be recomputed)
        if hasattr(self.transformer, "rope_embedder") and self.transformer.rope_embedder is not None:
            rope = self.transformer.rope_embedder
            device = next(rope.buffers()).device

            # Recompute the sequence buffer
            max_len = max(rope.max_h, rope.max_w, rope.max_t)
            rope.seq.copy_(torch.arange(max_len, dtype=torch.float, device=device))

            logger.debug("Reinitialized Cosmos RoPE buffers")

        super().reset_parameters()

        logger.info("Reset parameters for Cosmos Predict2 network")

    def fully_shard(self, **kwargs):
        """Fully shard the Cosmos Predict2 network for FSDP.

        Note: We shard `self.transformer` instead of `self` because the network wrapper
        class may have complex multiple inheritance with ABC, which causes Python's
        __class__ assignment to fail due to incompatible memory layouts.

        FSDP2's fully_shard works by dynamically modifying __class__, so we apply it only
        to the transformer submodule which is a standard torch.nn.Module.
        """
        # Note: Cosmos already has SAC (selective activation checkpointing) via ptd_checkpoint_wrapper
        # We don't need to re-apply checkpointing here

        # First, shard each transformer block for fine-grained parallelism
        for block in self.transformer.blocks:
            fully_shard(block, **kwargs)

        # Then shard the entire transformer
        fully_shard(self.transformer, **kwargs)

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
        if not isinstance(condition, dict):
            return None

        conditioning_latents = condition.get("conditioning_latents")
        condition_mask = condition.get("condition_mask")

        if conditioning_latents is None or condition_mask is None:
            return None

        # Pad conditioning_latents to full temporal dimension
        T_cond = conditioning_latents.shape[2]
        conditioning_latents_full = torch.zeros_like(x)
        conditioning_latents_full[:, :, :T_cond] = conditioning_latents

        # Expand mask to channel dimension [B, 1, T, H, W] -> [B, C, T, H, W]
        condition_mask_C = condition_mask.expand(-1, x.shape[1], -1, -1, -1)

        return conditioning_latents_full, condition_mask, condition_mask_C

    def preserve_conditioning(self, x: torch.Tensor, condition: Optional[Any]) -> torch.Tensor:
        """Preserve conditioning frames for video2world during student sampling.

        This hook is called by _student_sample_loop to ensure conditioning frames
        are preserved after each denoising step.

        Args:
            x: The tensor to modify (either x0 prediction or noisy latents)
            condition: The condition dict containing conditioning_latents and condition_mask

        Returns:
            The tensor with conditioning frames preserved
        """
        tensors = self._get_conditioning_tensors(x, condition)
        if tensors is None:
            return x

        conditioning_latents_full, _, condition_mask_C = tensors
        return conditioning_latents_full * condition_mask_C + x * (1 - condition_mask_C)

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 50,
        shift: float = 5.0,  # Cosmos default
        skip_layers: Optional[List[int]] = None,
        skip_layers_start_percent: float = 0.0,
        fps: Optional[torch.Tensor] = None,
        conditioning_latents: Optional[torch.Tensor] = None,
        num_conditioning_frames: int = 1,
        conditional_frame_timestep: float = 0.0,
        denoise_replace_gt_frames: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Multistep sampling using the FlowUniPC method.

        This implementation matches the official Cosmos inference and supports both
        text2world and video2world modes:
        - NO preconditioning - model receives raw input and timesteps
        - Model outputs velocity directly
        - CFG is applied on velocity predictions
        - Scheduler handles the integration

        For video2world mode (when conditioning_latents is provided):
        - First N frames are replaced with conditioning latents
        - Conditioning frames use a special timestep (conditional_frame_timestep)
        - Predicted velocity for conditioning frames is replaced with analytical velocity

        Args:
            noise: The noisy latents to start from, shape (B, C, T, H, W).
            condition: The conditioning embeddings.
            neg_condition: The negative/unconditional embeddings for CFG.
            guidance_scale: Classifier-free guidance scale. Values > 1.0 enable CFG.
            num_steps: The number of sampling steps.
            shift: Noise schedule shift parameter.
            skip_layers: List of transformer layers to skip (for skip-layer guidance).
            skip_layers_start_percent: Percentage of steps before starting to skip layers.
            fps: Frames per second tensor for temporal conditioning.
            conditioning_latents: Latent frames to condition on for video2world mode,
                shape (B, C, T, H, W). If provided, enables video2world mode.
            num_conditioning_frames: Number of frames to use for conditioning (default 1).
            conditional_frame_timestep: Timestep value for conditioning frames (default 0.0).
                Use 0.0 to indicate clean frames with no noise.
            denoise_replace_gt_frames: Whether to replace velocity for conditioning frames
                with analytical velocity (noise - gt_frames). Default True.

        Returns:
            The denoised sample tensor.
        """
        assert self.schedule_type == "rf", f"{self.schedule_type} is not supported"

        # Set timesteps with shift parameter (matching official Cosmos inference)
        if self.sample_scheduler is None:
            self.sample_scheduler = UniPCMultistepScheduler(
                num_train_timesteps=1000,
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                flow_shift=shift,
            )
        else:
            self.sample_scheduler.config.flow_shift = shift
        self.sample_scheduler.set_timesteps(num_inference_steps=num_steps, device=noise.device)
        timesteps = self.sample_scheduler.timesteps

        # Initialize latents with proper scaling based on the initial timestep
        t_init = timesteps[0] / self.sample_scheduler.config.num_train_timesteps
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        # Default fps for video generation (use self.fps which can be set via model.net.fps)
        if fps is None:
            fps = torch.full((latents.shape[0],), self.fps, device=latents.device)

        # Handle dict-style condition (extract conditioning info if provided this way)
        # This allows both direct kwargs and dict-style condition to work
        if isinstance(condition, dict):
            if conditioning_latents is None and "conditioning_latents" in condition:
                conditioning_latents = condition["conditioning_latents"]
            if "condition_mask" in condition:
                # Extract num_conditioning_frames from mask if not explicitly provided
                mask = condition["condition_mask"]
                if conditioning_latents is not None:
                    num_conditioning_frames = int(mask[:, :, :, 0, 0].sum(dim=2).max().item())
            # Extract text embeddings for the actual forward pass
            condition = condition["text_embeds"]
        if isinstance(neg_condition, dict):
            neg_condition = neg_condition["text_embeds"]

        # Video2world setup: create conditioning mask and prepare conditioning latents
        video2world_mode = conditioning_latents is not None
        conditioning_latents_full = None
        condition_mask = None
        condition_mask_C = None
        initial_noise = None

        if video2world_mode:
            B, C, T, H, W = latents.shape
            T_cond = conditioning_latents.shape[2]

            # Validate shapes
            assert T_cond <= T, f"Conditioning frames ({T_cond}) cannot exceed total frames ({T})"
            assert (
                num_conditioning_frames <= T_cond
            ), f"num_conditioning_frames ({num_conditioning_frames}) exceeds conditioning_latents frames ({T_cond})"

            # Create condition mask: 1 for conditioning frames, 0 for generated
            condition_mask = torch.zeros(B, 1, T, H, W, device=latents.device, dtype=latents.dtype)
            condition_mask[:, :, :num_conditioning_frames, :, :] = 1.0

            # Build condition dict for _get_conditioning_tensors
            v2w_condition = {"conditioning_latents": conditioning_latents, "condition_mask": condition_mask}
            conditioning_latents_full, condition_mask, condition_mask_C = self._get_conditioning_tensors(
                latents, v2w_condition
            )

            # Store initial noise for velocity replacement
            initial_noise = latents.clone()

        for idx, timestep in tqdm(enumerate(timesteps), total=num_steps, desc="Sampling"):
            # Normalize timestep to [0, 1] range
            t = (timestep / self.sample_scheduler.config.num_train_timesteps).expand(latents.shape[0])
            t = self.noise_scheduler.safe_clamp(t, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t).to(
                latents.dtype
            )

            if video2world_mode:
                # Replace conditioning frames with clean latents using preserve_conditioning
                v2w_condition = {"conditioning_latents": conditioning_latents, "condition_mask": condition_mask}
                model_input = self.preserve_conditioning(latents, v2w_condition)

                # Per-frame timesteps: conditioning frames get special timestep if enabled
                B, _, T, _, _ = latents.shape
                if conditional_frame_timestep >= 0:
                    t_expanded = t.unsqueeze(1).expand(B, T)
                    mask_B_T = condition_mask[:, 0, :, 0, 0]
                    t_per_frame = conditional_frame_timestep * mask_B_T + t_expanded * (1 - mask_B_T)
                else:
                    t_per_frame = t

                # Wrap condition with mask for forward() to use
                cond_with_mask = {"text_embeds": condition, "condition_mask": condition_mask}
                neg_cond_with_mask = {"text_embeds": neg_condition, "condition_mask": condition_mask}
            else:
                model_input = latents
                t_per_frame = t
                cond_with_mask = condition
                neg_cond_with_mask = neg_condition

            # Forward pass
            velocity_pred = self(model_input, t_per_frame, cond_with_mask, fps=fps)

            # Classifier-free guidance
            if guidance_scale > 1.0:
                velocity_uncond = self(model_input, t_per_frame, neg_cond_with_mask, fps=fps)
                velocity_pred = velocity_uncond + guidance_scale * (velocity_pred - velocity_uncond)

            # Replace velocity for conditioning frames with analytical velocity: v = noise - x0
            if video2world_mode and denoise_replace_gt_frames:
                gt_velocity = initial_noise - conditioning_latents_full
                velocity_pred = gt_velocity * condition_mask_C + velocity_pred * (1 - condition_mask_C)

            # Scheduler step (use model_input for video2world since that's what velocity was computed on)
            sample_input = model_input if video2world_mode else latents
            latents = self.sample_scheduler.step(velocity_pred, timestep, sample_input, return_dict=False)[0]

            # Preserve conditioning frames after each step to prevent numerical drift
            if video2world_mode:
                v2w_condition = {"conditioning_latents": conditioning_latents, "condition_mask": condition_mask}
                latents = self.preserve_conditioning(latents, v2w_condition)

        return latents

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Union[torch.Tensor, Dict[str, Any]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        unpatchify_features: bool = True,
        fwd_pred_type: Optional[str] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        skip_layers: Optional[List[int]] = None,
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
            r: Optional second timestep for flow matching (reserved for future use)
            return_features_early: If True, return features once all requested
                indices are collected without completing the full forward pass
            feature_indices: Set of block indices to extract features from.
                If non-empty, features will be returned along with the output.
            return_logvar: If True, return log variance along with the output
            unpatchify_features: If True, reshape features to (B, C, T, H, W) format
            fwd_pred_type: Override prediction type ('x0', 'eps', 'v', 'flow')
            fps: Frames per second tensor
            padding_mask: Padding mask tensor
            skip_layers: List of block indices to skip during forward pass

        Returns:
            Depending on the arguments:
            - torch.Tensor: Model output if no features or logvar requested
            - List[torch.Tensor]: Features if return_features_early=True
            - [output, features]: If feature_indices is non-empty
            - (output, logvar): If return_logvar=True
            - ([output, features], logvar): If both features and logvar requested
        """
        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            # Exit immediately if user requested this.
            return []

        if fps is None:
            fps = torch.full((x_t.shape[0],), self.fps, device=x_t.device)

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"

        # Handle dict-style condition for video2world mode
        # Extract: text_embeds, conditioning_latents, condition_mask (used as video_condition_mask)
        conditioning_latents = None
        condition_mask = None
        if isinstance(condition, dict):
            text_embeds = condition["text_embeds"]
            conditioning_latents = condition.get("conditioning_latents")
            condition_mask = condition.get("condition_mask")  # Used as video_condition_mask for transformer
        else:
            text_embeds = condition

        # Video2world training: replace input frames with conditioning latents
        model_input = x_t
        if conditioning_latents is not None and condition_mask is not None:
            B, C, T, H, W = x_t.shape
            # Expand condition_mask to channel dimension
            condition_mask_C = condition_mask.expand(-1, C, -1, -1, -1)
            # Pad conditioning_latents to full temporal dimension if needed
            if conditioning_latents.shape[2] < T:
                conditioning_latents_full = torch.zeros_like(x_t)
                conditioning_latents_full[:, :, : conditioning_latents.shape[2]] = conditioning_latents
            else:
                conditioning_latents_full = conditioning_latents
            # Replace conditioning frames
            model_input = conditioning_latents_full * condition_mask_C + x_t * (1 - condition_mask_C)

        model_outputs = self.transformer(
            x_B_C_T_H_W=model_input,
            timesteps_B_T=t,
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

        if len(feature_indices) == 0:
            assert isinstance(out, torch.Tensor)
            out = self.noise_scheduler.convert_model_output(
                model_input, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            # Video2world training: replace x0 prediction for conditioning frames
            if conditioning_latents is not None and condition_mask is not None and fwd_pred_type == "x0":
                out = conditioning_latents_full * condition_mask_C + out * (1 - condition_mask_C)
        else:
            assert isinstance(out, list)
            out[0] = self.noise_scheduler.convert_model_output(
                model_input, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            # Video2world training: replace x0 prediction for conditioning frames
            if conditioning_latents is not None and condition_mask is not None and fwd_pred_type == "x0":
                out[0] = conditioning_latents_full * condition_mask_C + out[0] * (1 - condition_mask_C)
            # Unpatchify features to match expected output format
            if unpatchify_features:
                out[1] = [rearrange(f, "B T H W D -> B D T H W") for f in out[1]]
            else:
                out[1] = [rearrange(f, "B T H W D -> B (T H W) D") for f in out[1]]

        if return_logvar:
            return out, logvar
        return out
