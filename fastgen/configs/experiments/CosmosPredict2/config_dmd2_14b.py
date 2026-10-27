# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DMD2 config for Cosmos-Predict2.5-14B model."""

import copy

import fastgen.configs.experiments.CosmosPredict2.config_dmd2 as config_dmd2_base
from fastgen.configs.discriminator import Discriminator_CosmosPredict2_14B_Config
from fastgen.configs.net import CosmosPredict2_14B_Aggressive_Config, CKPT_ROOT_DIR
from fastgen.configs.config_utils import model_pixel_shape


def create_config():
    config = config_dmd2_base.create_config()
    config.trainer.fsdp = True
    config.trainer.fsdp_cpu_offload = True
    config.model.meta_init = True
    config.model.precision_opt = "float32"

    # Latent shape: [C, T_latent, H_latent, W_latent]
    # Cosmos VAE uses 4x8x8 compression (time, height, width)
    # config.model.input_shape = [16, 24, 22, 40]  # cthw - 256p (320x176 video @ 93 frames)
    config.model.input_shape = [16, 24, 60, 104]  # cthw - 480p (832x480 video @ 93 frames)
    # config.model.input_shape = [16, 24, 88, 160]  # cthw - full 720p (1280x704 video @ 93 frames)

    # Network and discriminator for 14B
    config.model.net = copy.deepcopy(CosmosPredict2_14B_Aggressive_Config)
    config.model.discriminator = copy.deepcopy(Discriminator_CosmosPredict2_14B_Config)
    # Three heads at the 2B recipe's relative depths (~50% / ~75% / 100% of the 36 blocks).
    config.model.discriminator.feature_indices = [17, 26, 35]
    # Teacher uses AGGRESSIVE SAC for memory savings
    config.model.teacher = copy.deepcopy(CosmosPredict2_14B_Aggressive_Config)

    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cosmos_predict2/Cosmos-Predict2.5-14B/base/post-trained/e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt"

    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames

    config.log_config.group = "cosmos_predict2_14b_dmd2"

    return config
