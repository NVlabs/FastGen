# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import Flux2Config
from fastgen.configs.config_utils import model_pixel_shape

"""Configs for SFT (Supervised Fine-Tuning) on Flux.2-dev."""


def create_config():
    config = config_sft_default.create_config()

    # Model precision: keep FSDP-sharded params/grads in float32 so the
    # bfloat16 compute path doesn't accumulate weight drift over long SFT runs.
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # Flux.2 latent shape: [C, H, W] = [32, H//8, W//8]
    # AutoencoderKLFlux2 has 32 latent channels and 8x spatial compression.
    # The 2x2 spatial-to-channel patchification happens later in
    # `Flux2._pack_latents`, so `input_shape` here MUST be the raw VAE
    # output (i.e. before packing) to match the VAE-encoded `real_data` shape.
    #   For 512x512 images : [32, 64, 64]
    #   For 1024x1024 images: [32, 128, 128]
    config.model.input_shape = [32, 128, 128]

    # Network config
    config.model.net = copy.deepcopy(Flux2Config)
    config.trainer.fsdp = True
    config.model.meta_init = True

    # Optimizer config
    config.model.net_optimizer.lr = 1e-5

    # Timestep sampling config. `shifted` biases t toward higher noise
    # to be closer to the FLUX.2 inference schedule.
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 3
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Sampling config for visualization
    config.model.student_grid_cfg.steps = 28

    # Dataloader
    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    # A micro-batch of one limits activation memory for the 32B model.
    # Size the FSDP world and CPU offload for the available device memory;
    # the trainer adjusts accumulation to reach `trainer.batch_size_global`.
    config.dataloader_train.batch_size = 1
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    # Trainer
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500
    config.trainer.batch_size_global = 64

    config.log_config.group = "flux2_sft"

    return config
