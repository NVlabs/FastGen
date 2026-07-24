# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for SFT on Cosmos-Predict2.5-2B model."""

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import CosmosPredict2_2B_Config, CKPT_ROOT_DIR


def create_config():
    config = config_sft_default.create_config()
    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.model.net_optimizer.lr = 1e-5

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.precision = "bfloat16"

    # Latent shape: [C, T_latent, H_latent, W_latent]
    # Cosmos VAE uses 4x8x8 compression (time, height, width)
    # Small resolution for testing: 320x176 video -> 40x22 latent, 21 frames -> 6 latent
    # Full 720p: [16, 24, 88, 160] (1280x704 @ 93 frames)
    # config.model.input_shape = [16, 24, 24, 40]  # cthw - 256p
    # config.model.input_shape = [16, 24, 60, 104]  # cthw - 480p
    config.model.input_shape = [16, 24, 88, 160]  # cthw - full 720p

    config.model.net = CosmosPredict2_2B_Config
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cosmos_predict2/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
    # Official post-trained 2B/14B inference uses guidance scale 7
    config.model.guidance_scale = 5.0
    config.model.student_sample_steps = 35

    config.dataloader_train = VideoLoaderConfig
    config.dataloader_train.batch_size = 1

    # Dataloader img_size = (W, H) = (latent_W * 8, latent_H * 8)
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.log_config.group = "cosmos_predict2_sft"

    return config
