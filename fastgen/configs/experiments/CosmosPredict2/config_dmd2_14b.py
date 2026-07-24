# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DMD2 config for Cosmos-Predict2.5-14B model."""

import copy

import fastgen.configs.experiments.CosmosPredict2.config_dmd2 as config_dmd2_base
from fastgen.configs.discriminator import Discriminator_CosmosPredict2_14B_Config
from fastgen.configs.net import CosmosPredict2_14B_Config, CosmosPredict2_14B_Aggressive_Config, CKPT_ROOT_DIR


def create_config():
    config = config_dmd2_base.create_config()
    config.trainer.fsdp_cpu_offload = True

    # Latent shape: [C, T_latent, H_latent, W_latent]
    # Cosmos VAE uses 4x8x8 compression (time, height, width)
    # Small resolution for testing: 320x176 video -> 40x22 latent, 93 frames -> 24 latent
    # Full 720p: [16, 24, 88, 160] (1280x704 @ 93 frames)
    # config.model.input_shape = [16, 24, 22, 40]  # cthw - 256p (320x176 video)
    config.model.input_shape = [16, 24, 60, 104]  # cthw - 480p, 93 frames
    # config.model.input_shape = [16, 24, 88, 160]  # cthw - full 720p, 93 frames

    # Network and discriminator for 14B
    config.model.net = CosmosPredict2_14B_Config
    config.model.discriminator = Discriminator_CosmosPredict2_14B_Config
    # Teacher uses AGGRESSIVE SAC for memory savings
    config.model.teacher = CosmosPredict2_14B_Aggressive_Config

    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cosmos_predict2/Cosmos-Predict2.5-14B/base/post-trained/e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt"

    # Dataloader img_size = (W, H) = (latent_W * 8, latent_H * 8)
    config.dataloader_train = copy.deepcopy(config.dataloader_train)
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.log_config.group = "cosmos_predict2_14b_dmd2"

    return config
