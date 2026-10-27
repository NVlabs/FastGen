# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.discriminator import Discriminator_Wan_1_3B_Config
import fastgen.configs.methods.config_f_distill as config_f_distill_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan_1_3B_Config
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.Wan.network import Wan

""" Configs for the f-distill model on Wan-1.3B model. """


def create_config():
    config = config_f_distill_default.create_config()
    config.model.net_optimizer.lr = 5e-5
    config.model.discriminator_optimizer.lr = 5e-5
    config.model.fake_score_optimizer.lr = 5e-5

    config.model.precision = "bfloat16"
    # VAE compress ratio: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]
    config.model.discriminator = copy.deepcopy(Discriminator_Wan_1_3B_Config)
    config.model.gan_loss_weight_gen = 0.001
    config.model.net = copy.deepcopy(Wan_1_3B_Config)
    config.model.guidance_cfg = Wan.default_guidance_cfg()

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.f_distill.f_div = "js"

    config.dataloader_train = copy.deepcopy(VideoLoaderConfig)
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps

    # 480p (832x480) resolution
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames

    config.log_config.group = "wan_fdistill"
    return config
