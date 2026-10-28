# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for the DMD2 model on CogVideoX-5B model."""

import copy

from fastgen.configs.discriminator import Discriminator_CogVideoX5B_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.net import CogVideoX5BConfig
from fastgen.networks.CogVideoX.network import CogVideoX


def create_config():
    config = config_dmd2_default.create_config()

    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    config.model.input_shape = [16, 13, 60, 90]
    config.model.discriminator = copy.deepcopy(Discriminator_CogVideoX5B_Config)
    config.model.discriminator.feature_indices = [21, 31, 41]  # 42-layer 5B: ~50%, ~74%, 100%
    config.model.gan_loss_weight_gen = 0.03
    config.model.net = copy.deepcopy(CogVideoX5BConfig)
    config.model.guidance_cfg = CogVideoX.default_guidance_cfg()
    config.model.enable_preprocessors = False

    # SAT's CogVideoX-5B recipe
    config.model.precision = "float32"
    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"
    config.trainer.fsdp = True
    config.model.meta_init = True

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"
    config.model.student_sample_type = "ode"

    # setting for 4-step training
    config.model.student_grid_cfg.steps = 4
    config.model.student_grid_cfg.grid_kwargs = {"shift": 5}

    config.dataloader_train = copy.deepcopy(VideoLatentLoaderConfig)
    config.dataloader_train.batch_size = 2

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "CogVideoX5B_dmd2"
    return config
