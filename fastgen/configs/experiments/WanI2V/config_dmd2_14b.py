# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.discriminator import Discriminator_Wan_14B_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan21_I2V_14B_480P_Config
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.WanI2V.network import WanI2V

"""Configs for the DMD2 model on the Wan2.1 I2V 14B model."""


def create_config():
    config = config_dmd2_default.create_config()
    config.trainer.fsdp = True
    config.model.meta_init = True

    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.model.net_optimizer.lr = 3e-6
    config.model.discriminator_optimizer.lr = 3e-6
    config.model.fake_score_optimizer.lr = 3e-6

    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"
    # VAE compress ratio: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]
    config.model.discriminator = copy.deepcopy(Discriminator_Wan_14B_Config)
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [21, 30, 39]
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"
    config.model.net = copy.deepcopy(Wan21_I2V_14B_480P_Config)
    config.model.guidance_cfg = WanI2V.default_guidance_cfg()

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.student_sample_type = "ode"

    # setting for 2-step training
    config.model.student_grid_cfg.steps = 2
    config.model.student_grid_cfg.grid_type = "explicit"
    config.model.student_grid_cfg.grid_kwargs = {"timesteps": [0.999, 0.833, 0.0]}

    config.dataloader_train = copy.deepcopy(VideoLoaderConfig)
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps

    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames

    config.dataloader_train.batch_size = 1

    config.log_config.group = "wan21_14b_i2v_dmd2"
    return config
