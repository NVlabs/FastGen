# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan22_I2V_5B_Config
from fastgen.networks.WanI2V.network import WanI2V
from fastgen.configs.config_utils import model_pixel_shape

""" Configs for SFT on Wan-2.2-5B model. """


def create_config():
    config = config_sft_default.create_config()
    config.model.meta_init = True
    config.trainer.fsdp = True

    config.trainer.logging_iter = 100
    config.model.net_optimizer.lr = 1e-5
    config.model.student_grid_cfg.steps = 50

    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"

    # 121 video frames at 1280x704: 4x temporal and 16x spatial VAE compression.
    config.model.input_shape = [48, 31, 44, 80]  # cthw
    config.model.net = copy.deepcopy(Wan22_I2V_5B_Config)
    config.model.guidance_cfg = WanI2V.default_guidance_cfg()

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.dataloader_train = copy.deepcopy(VideoLoaderConfig)
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps
    config.dataloader_train.batch_size = 1

    # Native 1280x704 resolution.
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames

    config.log_config.group = "wan22_5b_i2v_sft"
    return config
