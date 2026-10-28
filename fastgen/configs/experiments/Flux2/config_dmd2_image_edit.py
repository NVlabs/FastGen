# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.discriminator import Discriminator_Flux2_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import ImageEditLoaderConfig
from fastgen.configs.net import Flux2ImageEditConfig
from fastgen.configs.config_utils import model_pixel_shape

"""DMD2 image-edit distillation: FLUX.2-dev (32B, 28 steps) -> 4-step student."""


def create_config():
    config = config_dmd2_default.create_config()

    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    # Raw VAE-output latent shape for 1024x1024 inputs.
    config.model.input_shape = [32, 128, 128]

    # ImageDiT discriminator sized for FLUX.2-dev (56 blocks, hidden=6144).
    config.model.discriminator = copy.deepcopy(Discriminator_Flux2_Config)
    config.model.discriminator.feature_indices = [28, 42, 55]

    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"

    # Dev (32B) with image-editing reference path enabled. The same
    # ``Flux2ImageEditConfig`` is reused for the student, the teacher, and
    # the fake_score network -- DMD2 instantiates a copy of `model.net` for
    # each, and the image-editing branch is consistent across all three so
    # the score distillation operates entirely in the joint
    # (target + reference) sequence space.
    # Flux.2-dev image-edit teacher uses embedded distillation guidance
    # (constructor default `guidance_scale=4.0`); `model.guidance_cfg` is
    # left at its empty `GuidanceConfig()` default so DMD2's teacher rollout
    # (via `predict_with_guidance`) skips the CFG term.
    config.model.net = copy.deepcopy(Flux2ImageEditConfig)
    config.trainer.fsdp = True
    config.model.meta_init = True

    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # 4-step student rollout (the distillation target).
    config.model.student_grid_cfg.steps = 4
    config.model.student_sample_type = "ode"

    # Rectified-flow timestep sampler.
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 3
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Approximate BFL's empirical-mu grid at 1024px / 4 steps. Reference-image
    # tokens are conditioning only: the sampler computes
    # image_seq_len from the noisy target, so image editing uses the same
    # timestep grid as text-to-image at the same target resolution.
    config.model.student_grid_cfg.grid_kwargs = {"shift": 9.9}

    # Image-editing dataset.
    config.dataloader_train = copy.deepcopy(ImageEditLoaderConfig)
    config.dataloader_train.batch_size = 1
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "flux2_dev_image_edit_dmd2"

    return config
