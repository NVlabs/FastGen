# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.discriminator import Discriminator_Flux2_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import Flux2Config
from fastgen.configs.config_utils import model_pixel_shape

"""Configs for DMD2 distillation on Flux.2-dev model."""


def create_config():
    config = config_dmd2_default.create_config()

    # Optimizer settings
    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    # Flux.2 latent shape: [C, H, W] = [32, H//8, W//8]
    # For 1024x1024 images: [32, 128, 128]
    config.model.input_shape = [32, 128, 128]

    # Discriminator config - ImageDiT with Flux.2-dev dimensions
    # 8 joint + 48 single = 56 blocks, hidden_dim = 6144
    config.model.discriminator = copy.deepcopy(Discriminator_Flux2_Config)

    # Late-block feature taps for the GAN: blocks at ~50% / ~75% / 100% depth;
    # alternative `[28, 42]` drops the final block if memory is tight.
    config.model.discriminator.feature_indices = [28, 42, 55]

    # GAN settings
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"

    # Network config. Flux.2-dev uses embedded distillation guidance, pinned
    # to 4.0 in `Flux2Config` itself. `model.guidance_cfg` is left at its empty
    # `GuidanceConfig()` default so DMD2's teacher rollout (via
    # `predict_with_guidance`) skips the CFG term — stacking CFG on
    # top of embedded distillation guidance would double-count.
    config.model.net = copy.deepcopy(Flux2Config)
    config.trainer.fsdp = True
    config.model.meta_init = True

    # Precision: store FSDP params/grads in float32 for stable distillation
    # (compute still happens in bfloat16 via `precision`).
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # Student sampling: 4-step ODE rollout (DMD2 distillation target).
    config.model.student_grid_cfg.steps = 4
    config.model.student_sample_type = "ode"

    # Time sampling config: rectified-flow `shifted` distribution biases t
    # toward higher noise levels (matches the FLUX.2 inference schedule
    # better than plain uniform). shift=3 is the conservative default;
    # try 4 or 5 if the student under-trains the high-noise regime.
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 3
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Approximate the FLUX.2 scheduler's empirical-mu grid at 1024px / 4 steps.
    config.model.student_grid_cfg.grid_kwargs = {"shift": 9.9}

    # Dataloader: raw-image dataset (point datatags at your shards).
    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    config.dataloader_train.batch_size = 1
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    # Training iterations
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    # Logging
    config.log_config.group = "flux2_dmd2"

    return config
