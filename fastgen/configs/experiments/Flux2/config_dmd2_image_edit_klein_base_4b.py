# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.discriminator import Discriminator_Flux2KleinBase4B_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import ImageEditLoaderConfig
from fastgen.configs.net import Flux2KleinBase4BImageEditConfig
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.Flux2.network import Flux2

"""DMD2 image-edit distillation: FLUX.2-klein-base-4B (50 steps) -> 4-step student."""


def create_config():
    config = config_dmd2_default.create_config()

    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    config.model.input_shape = [32, 128, 128]

    # Discriminator sized for klein-base-4B's 25-block transformer.
    config.model.discriminator = copy.deepcopy(Discriminator_Flux2KleinBase4B_Config)
    config.model.discriminator.feature_indices = [12, 19, 24]

    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"

    config.model.net = copy.deepcopy(Flux2KleinBase4BImageEditConfig)
    config.trainer.fsdp = True
    config.model.meta_init = True

    # Teacher CFG for the DMD2 distribution-matching target (also used for teacher samples).
    config.model.guidance_cfg = Flux2.default_guidance_cfg()

    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    config.model.student_grid_cfg.steps = 4
    config.model.student_sample_type = "ode"

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 3
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.student_grid_cfg.grid_kwargs = {"shift": 9.9}

    # Image-editing dataset.
    config.dataloader_train = copy.deepcopy(ImageEditLoaderConfig)
    config.dataloader_train.batch_size = 1
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "flux2_klein_base_4b_image_edit_dmd2"

    return config
