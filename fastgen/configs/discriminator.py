# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from fastgen.networks.discriminators import (
    Discriminator_EDM,
    Discriminator_SD15,
    Discriminator_SDXL,
    Discriminator_ImageDiT,
    Discriminator_VideoDiT,
)

Discriminator_EDM_CIFAR10_Config: DictConfig = L(Discriminator_EDM)(
    feature_indices={0, 1, 2},
    all_res=[32, 16, 8],
    in_channels=256,
)

Discriminator_EDM_ImageNet64_Config: DictConfig = L(Discriminator_EDM)(
    feature_indices=None,
    all_res=[64, 32, 16, 8],
    in_channels=768,
)

Discriminator_SD15_Res512_Config: DictConfig = L(Discriminator_SD15)(
    feature_indices=None,
    all_res=[32, 16, 8, 8, 8],
    in_channels=1280,
)

Discriminator_SDXL_Res512_Config: DictConfig = L(Discriminator_SDXL)(
    feature_indices=None,
    all_res=[32, 16, 16, 16],
    in_channels=1280,
)

Discriminator_SDXL_Res1024_Config: DictConfig = L(Discriminator_SDXL)(
    feature_indices=None,
    all_res=[64, 32, 32, 32],
    in_channels=1280,
)

# Flux: hidden_dim=3072, 19 joint blocks + 38 single blocks = 57 total
Discriminator_Flux_Config: DictConfig = L(Discriminator_ImageDiT)(
    feature_indices=None,
    num_blocks=57,  # 19 joint + 38 single blocks
    inner_dim=3072,  # Flux hidden dimension
)

# QwenImage: hidden_dim=3072, 60 dual-stream DiT blocks
Discriminator_QwenImage_Config: DictConfig = L(Discriminator_ImageDiT)(
    feature_indices=None,
    num_blocks=60,  # 60 dual-stream blocks
    inner_dim=3072,  # 24 heads x 128 head_dim
)

# 2B patchify: spatial-2, temporal-1; inner_dim=1920; layer=30
Discriminator_CogVideoX2B_Config = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=30,
    disc_type="dit_simple_conv3d",
    inner_dim=1920 // 4,
)

# 5B patchify: spatial-2, temporal-1; inner_dim=3072; layer=42
Discriminator_CogVideoX5B_Config = L(Discriminator_VideoDiT)(
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
