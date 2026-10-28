# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import SDXLConfig
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.SDXL.network import StableDiffusionXL

"""Configs for SFT (Supervised Fine-Tuning) on SDXL."""


def create_config():
    config = config_sft_default.create_config()

    # SDXL latent shape: [C, H, W] = [4, 128, 128] for 1024x1024 images
    config.model.input_shape = [4, 128, 128]

    # Network config
    config.model.net = copy.deepcopy(SDXLConfig)

    # Optimizer config
    config.model.net_optimizer.lr = 1e-5

    # Timestep sampling config
    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Sampling config for visualization
    config.model.guidance_cfg = StableDiffusionXL.default_guidance_cfg()
    config.model.student_grid_cfg.steps = 50

    # Dataloader (raw images - SDXL VAE encodes to latents)
    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    config.dataloader_train.batch_size = 8
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    # Trainer
    config.trainer.batch_size_global = 256
    config.trainer.max_iter = 100000
    config.trainer.logging_iter = 500

    config.log_config.group = "sdxl_sft"

    return config
