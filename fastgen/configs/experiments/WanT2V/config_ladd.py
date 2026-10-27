# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for the LADD model on Wan-1.3B."""

import copy

from fastgen.configs.discriminator import Discriminator_Wan_1_3B_Config
import fastgen.configs.methods.config_ladd as config_ladd_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan_1_3B_Config
from fastgen.configs.config_utils import model_pixel_shape


def create_config():
    config = config_ladd_default.create_config()
    config.model.net_optimizer.lr = 1e-7
    config.model.discriminator_optimizer.lr = 4e-6

    config.trainer.fsdp = True
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"
    # VAE compress ratio: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]
    config.model.discriminator = copy.deepcopy(Discriminator_Wan_1_3B_Config)
    config.model.discriminator.feature_indices = [5, 11, 17, 23, 29]
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.net = copy.deepcopy(Wan_1_3B_Config)

    config.model.gan_r1_reg_weight = 1.0
    config.model.gan_r1_reg_alpha = 0.1
    config.model.gan_use_same_t_noise = True
    config.model.student_update_freq = 2

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # setting for 4-step training
    config.model.student_grid_cfg.steps = 4
    config.model.student_grid_cfg.grid_type = "explicit"
    config.model.student_grid_cfg.grid_kwargs = {"timesteps": [0.999, 0.937, 0.833, 0.624, 0.0]}

    config.dataloader_train = copy.deepcopy(VideoLoaderConfig)
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps

    # 480p (832x480) resolution
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames

    config.trainer.max_iter = 3000
    config.trainer.logging_iter = 50
    config.trainer.save_ckpt_iter = 250

    config.log_config.group = "wan_ladd"
    return config
