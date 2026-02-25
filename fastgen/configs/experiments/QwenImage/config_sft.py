# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import HiDreamG5_JourneyDB_Loader_Config
from fastgen.configs.net import QwenImageConfig

"""Configs for SFT (Supervised Fine-Tuning) on Qwen-Image."""


def create_config():
    config = config_sft_default.create_config()

    # Enable meta init for FSDP - only rank 0 loads weights, others use meta device
    # Critical for QwenImage (~20B params) to avoid OOM during loading
    config.model.fsdp_meta_init = True

    # Model precision
    config.model.precision = "bfloat16"

    # QwenImage latent shape: [C, H, W] = [16, H//8, W//8]
    # For 512x512 images: [16, 64, 64]
    # For 1024x1024 images: [16, 128, 128]
    config.model.input_shape = [16, 64, 64]  # 512x512 images

    # Network config
    config.model.net = QwenImageConfig

    # Optimizer config
    config.model.net_optimizer.lr = 1e-5

    # Timestep sampling config
    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Sampling config for visualization
    config.model.guidance_scale = 4.0
    config.model.student_sample_steps = 50

    # Dataloader (raw images - QwenImage VAE encodes to 16ch latents)
    config.dataloader_train = HiDreamG5_JourneyDB_Loader_Config
    config.dataloader_train.batch_size = 4
    config.dataloader_train.input_res = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)

    # Trainer
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100

    config.log_config.group = "qwen_image_sft"

    return config
