# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

from fastgen.configs.instantiate import LazyCall as L
from fastgen.networks.discriminators import (
    Discriminator_EDM,
    Discriminator_SD15,
    Discriminator_SDXL,
    Discriminator_ImageDiT,
    Discriminator_VideoDiT,
    Discriminator_Audio2D,
    CompositeDiscriminator,
)

Discriminator_EDM_CIFAR10_Config: DictConfig = L(Discriminator_EDM)(
    feature_indices={0, 1, 2},
    all_res=[32, 16, 8],
    in_channels=256,  # every tap: EDM_CIFAR10_Config's DDPM++ backbone has channel_mult=[2, 2, 2]
)

Discriminator_EDM_ImageNet64_Config: DictConfig = L(Discriminator_EDM)(
    feature_indices=None,
    all_res=[64, 32, 16, 8],
    in_channels=[192, 384, 576, 768],  # per-tap widths: the last block of each DhariwalUNet resolution
)

Discriminator_SD15_Res512_Config: DictConfig = L(Discriminator_SD15)(
    feature_indices=None,
    all_res=[32, 16, 8, 8, 8],
    in_channels=[320, 640, 1280, 1280, 1280],  # per-tap widths: the four down blocks, then mid
)

Discriminator_SDXL_Res512_Config: DictConfig = L(Discriminator_SDXL)(
    feature_indices=None,
    all_res=[32, 16, 16, 16],
    in_channels=[320, 640, 1280, 1280],  # per-tap widths: the three down blocks, then mid
)

Discriminator_SDXL_Res1024_Config: DictConfig = L(Discriminator_SDXL)(
    feature_indices=None,
    all_res=[64, 32, 32, 32],
    in_channels=[320, 640, 1280, 1280],  # per-tap widths: the three down blocks, then mid
)

# Flux: hidden_dim=3072, 19 joint blocks + 38 single blocks = 57 total
Discriminator_Flux_Config: DictConfig = L(Discriminator_ImageDiT)(
    feature_indices=None,
    num_blocks=57,  # 19 joint + 38 single blocks
    inner_dim=3072,  # Flux hidden dimension
)

# Flux.2-dev: hidden_dim=6144, 8 joint blocks + 48 single blocks = 56 total
Discriminator_Flux2_Config: DictConfig = L(Discriminator_ImageDiT)(
    feature_indices=None,
    num_blocks=56,  # 8 joint + 48 single blocks
    inner_dim=6144,  # Flux.2-dev hidden dimension (48 heads x 128 head_dim)
)

# Flux.2-klein-base-4B: hidden_dim=3072, 5 joint + 20 single = 25 total
Discriminator_Flux2KleinBase4B_Config: DictConfig = L(Discriminator_ImageDiT)(
    feature_indices=None,
    num_blocks=25,  # 5 joint + 20 single blocks
    inner_dim=3072,  # Flux.2-klein-base-4B hidden dimension (24 heads x 128 head_dim)
)

# Flux.2-klein-base-9B: hidden_dim=4096, 8 joint + 24 single = 32 total
Discriminator_Flux2KleinBase9B_Config: DictConfig = L(Discriminator_ImageDiT)(
    feature_indices=None,
    num_blocks=32,  # 8 joint + 24 single blocks
    inner_dim=4096,  # Flux.2-klein-base-9B hidden dimension (32 heads x 128 head_dim)
)

# QwenImage: hidden_dim=3072, 60 dual-stream DiT blocks
Discriminator_QwenImage_Config: DictConfig = L(Discriminator_ImageDiT)(
    feature_indices=None,
    num_blocks=60,  # 60 dual-stream blocks
    inner_dim=3072,  # 24 heads x 128 head_dim
)

# 2B patchify: spatial-2, temporal-1; inner_dim=1920; layer=30
Discriminator_CogVideoX2B_Config: DictConfig = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=30,
    disc_type="dit_simple_conv3d",
    inner_dim=1920 // 4,
)

# 5B patchify: spatial-2, temporal-1; inner_dim=3072; layer=42
Discriminator_CogVideoX5B_Config: DictConfig = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=42,
    disc_type="dit_simple_conv3d",
    inner_dim=3072 // 4,
)

# 1.3B patchify: spatial-2, temporal-1; inner_dim=1536; layer=30
Discriminator_Wan_1_3B_Config: DictConfig = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=30,
    disc_type="dit_simple_conv3d",
    inner_dim=1536 // 4,
)

# 14B patchify: spatial-2, temporal-1; inner_dim=5120; layer=40
Discriminator_Wan_14B_Config: DictConfig = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=40,
    disc_type="dit_simple_conv3d",
    inner_dim=5120 // 4,
)

# 5B patchify: spatial-2, temporal-1; inner_dim=3072; layer=30
Discriminator_Wan22_5B_Config: DictConfig = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=30,
    disc_type="dit_simple_conv3d",
    inner_dim=3072 // 4,
)

# LTX-2: patchify spatial-1, temporal-1; inner_dim=4096; layer=48
Discriminator_LTX2_Config: DictConfig = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=48,
    disc_type="dit_simple_conv3d",
    inner_dim=4096,
)

# LTX-2 multimodal: video 3D disc (inner_dim=4096), audio 2D disc (inner_dim=128); layer=48
# Audio inner_dim=128 because token dim 2048 unpacks to [B, 128, L, 16] (C=2048/M, M=mel_bins=16)
Discriminator_LTX2_Multimodal_Config: DictConfig = L(CompositeDiscriminator)(
    modality_discs={
        "video": L(Discriminator_VideoDiT)(
            num_blocks=48,
            disc_type="dit_simple_conv3d",
            inner_dim=4096,
        ),
        "audio": L(Discriminator_Audio2D)(
            inner_dim=128,
            hidden_channels=256,
        ),
    },
)

# Cosmos Predict2.5-2B: patchify spatial-2, temporal-1; inner_dim=2048; layer=28
Discriminator_CosmosPredict2_2B_Config: DictConfig = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=28,
    disc_type="dit_simple_conv3d",
    inner_dim=2048,  # Must match model's inner_dim for Cosmos
)

# Cosmos Predict2.5-14B: patchify spatial-2, temporal-1; inner_dim=5120; layer=36
Discriminator_CosmosPredict2_14B_Config: DictConfig = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=36,
    disc_type="dit_simple_conv3d",
    inner_dim=5120,  # Must match model's inner_dim for Cosmos
)

# Cosmos3-Nano (8B-class generator): 36 MoT layers, hidden 4096. Cosmos3.forward reshapes the
# gen-stream features to [B, 4096, T, ph, pw] for this discriminator.
Discriminator_Cosmos3_Config: DictConfig = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=36,
    disc_type="dit_simple_conv3d",
    inner_dim=4096,  # Must match Cosmos3 hidden_size
)
