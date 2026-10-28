# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import QwenImageConfig
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.QwenImage.network import QwenImage

"""Configs for SFT (Supervised Fine-Tuning) on Qwen-Image."""


def create_config():
    config = config_sft_default.create_config()

    # Enable meta init for FSDP - only rank 0 loads weights, others use meta device
    # Critical for QwenImage (~20B params) to avoid OOM during loading
    config.model.meta_init = True
    config.trainer.fsdp = True

    # Model precision
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # QwenImage latent shape: [C, H, W] = [16, H//8, W//8]
    # For 512x512 images: [16, 64, 64]
    # For 1024x1024 images: [16, 128, 128]
    config.model.input_shape = [16, 64, 64]  # 512x512 images

    # Network config
    config.model.net = copy.deepcopy(QwenImageConfig)
    config.model.guidance_cfg = QwenImage.default_guidance_cfg()

    # Optimizer config
    config.model.net_optimizer.lr = 1e-5

    # Timestep sampling config
    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Sampling config for visualization
    config.model.student_grid_cfg.steps = 50

    # Dataloader (raw images - QwenImage VAE encodes to 16ch latents)
    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    config.dataloader_train.batch_size = 4
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    # Trainer
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100

    config.log_config.group = "qwen_image_sft"

    return config
