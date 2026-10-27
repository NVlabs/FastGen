# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageEditLoaderConfig
from fastgen.configs.net import Flux2ImageEditConfig
from fastgen.configs.config_utils import model_pixel_shape

"""SFT (instruction-tuned image editing) config for FLUX.2-dev (32B)."""


def create_config():
    config = config_sft_default.create_config()

    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # Raw VAE-output latent shape for 1024x1024 inputs.
    config.model.input_shape = [32, 128, 128]

    # FLUX.2-dev with image-editing reference path enabled.
    config.model.net = copy.deepcopy(Flux2ImageEditConfig)
    config.trainer.fsdp = True
    config.model.meta_init = True

    config.model.net_optimizer.lr = 1e-5

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 3
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.student_grid_cfg.steps = 28

    # Image-editing dataset.
    config.dataloader_train = copy.deepcopy(ImageEditLoaderConfig)
    config.dataloader_train.batch_size = 1
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    config.model.cond_dropout_prob = 0.1
    config.model.cond_keys_no_dropout = ["condition_image"]

    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "flux2_dev_image_edit_sft"

    return config
