# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DMD2 config for Cosmos-Predict2.5-2B model."""

import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.discriminator import Discriminator_CosmosPredict2_2B_Config
from fastgen.configs.net import CosmosPredict2_2B_Config, CosmosPredict2_2B_Aggressive_Config, CKPT_ROOT_DIR


def create_config():
    config = config_dmd2_default.create_config()

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    # Optimizer settings
    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    config.model.precision = "bfloat16"

    # Latent shape: [C, T_latent, H_latent, W_latent]
    # Cosmos VAE uses 4x8x8 compression (time, height, width)
    # Small resolution for testing: 320x176 video -> 40x22 latent, 93 frames -> 24 latent
    # Full 720p: [16, 24, 88, 160] (1280x704 @ 93 frames)
    # config.model.input_shape = [16, 24, 22, 40]  # cthw - 256p (320x176 video)
    # config.model.input_shape = [16, 24, 60, 104]  # cthw - 480p, 93 frames
    config.model.input_shape = [16, 24, 88, 160]  # cthw - full 720p, 93 frames

    # Network and discriminator
    config.model.net = CosmosPredict2_2B_Config
    config.model.discriminator = Discriminator_CosmosPredict2_2B_Config
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [13, 20, 27]
    # Teacher uses AGGRESSIVE SAC for memory savings
    config.model.teacher = CosmosPredict2_2B_Aggressive_Config

    # DMD2 settings
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"
    config.model.student_sample_type = "ode"
    # Official 2B DMD2 teacher guidance scale
    config.model.guidance_scale = 4.0
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cosmos_predict2/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"

    # Timestep sampling
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # setting for 4-step training
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Dataloader settings
    config.dataloader_train = VideoLoaderConfig
    config.dataloader_train.batch_size = 1

    # Dataloader img_size = (W, H) = (latent_W * 8, latent_H * 8)
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.log_config.group = "cosmos_predict2_dmd2"

    return config
