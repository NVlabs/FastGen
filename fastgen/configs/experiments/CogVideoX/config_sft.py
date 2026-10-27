# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for the SFT model on CogVideoX-5B model."""

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.net import CogVideoX5BConfig
from fastgen.networks.CogVideoX.network import CogVideoX


def create_config():
    config = config_sft_default.create_config()

    config.trainer.logging_iter = 500

    config.model.net_optimizer.lr = 2e-5

    config.model.guidance_cfg = CogVideoX.default_guidance_cfg()
    # CogVideoX's DPM-Solver++ sampler is stochastic; ``net.sample`` accepts only ``"sde"``.
    config.model.student_sample_type = "sde"
    config.model.student_grid_cfg.steps = 50

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # SAT's CogVideoX-5B recipe
    config.model.precision = "float32"
    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"
    config.trainer.fsdp = True
    config.model.meta_init = True

    # CogVideoX latent shape: [C, T, H, W] = [16, 13, 60, 90]
    # Corresponds to 49 frames at 480x720 resolution after VAE encoding
    config.model.input_shape = [16, 13, 60, 90]
    config.model.net = copy.deepcopy(CogVideoX5BConfig)
    config.model.enable_preprocessors = False  # Using precomputed latents

    config.dataloader_train = copy.deepcopy(VideoLatentLoaderConfig)
    config.dataloader_train.batch_size = 1

    config.log_config.group = "CogVideoX5B_sft"
    return config
