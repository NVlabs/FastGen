# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.discriminator import Discriminator_Wan_1_3B_Config
import fastgen.configs.methods.config_ladd as config_ladd_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan_1_3B_Config

""" Configs for the LADD model on Wan-1.3B model. """


def create_config():
    config = config_ladd_default.create_config()
    config.model.net_optimizer.lr = 2e-6
    config.model.discriminator_optimizer.lr = 4e-6

    config.model.precision = "bfloat16"
    # VAE compress ratio: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]
    config.model.discriminator = Discriminator_Wan_1_3B_Config
    config.model.discriminator.feature_indices = [15, 22, 29]
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.net = Wan_1_3B_Config

    config.model.gan_r1_reg_weight = 0.1
    config.model.gan_r1_reg_alpha = 0.1
    config.model.gan_use_same_t_noise = True
    config.model.student_update_freq = 2

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # setting for 4-step training
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    config.dataloader_train = VideoLoaderConfig

    # 480p (832x480) resolution
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.log_config.group = "wan_ladd"
    return config
