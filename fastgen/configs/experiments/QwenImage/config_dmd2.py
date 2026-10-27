# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.discriminator import Discriminator_QwenImage_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import QwenImageConfig
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.QwenImage.network import QwenImage

"""Configs for DMD2 distillation on Qwen-Image model."""


def create_config():
    config = config_dmd2_default.create_config()

    # Optimizer settings
    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    # QwenImage latent shape: [C, H, W] = [16, H//8, W//8]
    # For 512x512 images: [16, 64, 64]
    # For 1024x1024 images: [16, 128, 128]
    config.model.input_shape = [16, 64, 64]

    # Discriminator config - ImageDiT with simple conv2d architecture
    config.model.discriminator = copy.deepcopy(Discriminator_QwenImage_Config)

    # GAN settings
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"

    # We can further set GAN discriminator feature indices to control which
    # intermediate features to use for discrimination. By default, we use the middle block
    # config.model.discriminator.feature_indices=[30, 45] #use the 30th and 45th blocks from the DiT

    # Network config
    config.model.net = copy.deepcopy(QwenImageConfig)
    config.model.guidance_cfg = QwenImage.default_guidance_cfg()

    # Meta init required for QwenImage (~20B params) — only rank 0 loads weights,
    # others use meta device to avoid OOM from 3 copies (student+teacher+fake_score)
    config.trainer.fsdp = True
    config.model.meta_init = True

    # Precision
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # Student sampling steps
    config.model.student_grid_cfg.steps = 4

    # Time sampling config
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 2.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    # Qwen's sampler maps each native grid point u to
    # exp(mu) / (exp(mu) + (1 / u - 1)), where
    # mu = 0.5 + (0.9 - 0.5) / (8192 - 256) * ((W / 16) * (H / 16) - 256).
    # At the configured 512px resolution, exp(mu) ~= 1.714.
    config.model.student_grid_cfg.grid_kwargs = {"shift": 1.714}

    # Dataloader
    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    config.dataloader_train.batch_size = 2
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    # Training iterations
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    # FSDP CPU offload
    config.trainer.fsdp_cpu_offload = True

    # Logging
    config.log_config.group = "qwen_image_dmd2"

    return config
