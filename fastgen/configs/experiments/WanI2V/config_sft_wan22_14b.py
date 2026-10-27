# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan22_I2V_14B_Config
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.WanI2V.wan_moe import WanMoEI2V

"""Configs for SFT on WAN 2.2 I2V A14B (MoE) model."""


def create_config():
    config = config_sft_default.create_config()
    config.trainer.fsdp = True
    config.model.meta_init = True

    config.trainer.logging_iter = 500
    config.model.net_optimizer.lr = 1e-5
    config.model.student_grid_cfg.steps = 40

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"

    # VAE compress ratio for WAN 14B: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]  # cthw
    config.model.net = copy.deepcopy(Wan22_I2V_14B_Config)
    config.model.guidance_cfg = WanMoEI2V.default_guidance_cfg()

    # Note that we need access to the raw data to encode the first frame, padded with zeros
    config.dataloader_train = copy.deepcopy(VideoLoaderConfig)
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps
    config.dataloader_train.batch_size = 1

    # 480p (832x480) resolution
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames

    config.log_config.group = "wan22_14b_i2v_sft"
    return config
