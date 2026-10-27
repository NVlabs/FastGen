# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import Flux2KleinBase9BConfig
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.Flux2.network import Flux2

"""SFT config for FLUX.2-klein-base-9B (FLUX NC v2.1, undistilled, 9B params)."""


def create_config():
    config = config_sft_default.create_config()

    # Precision: bf16 compute, fp32 FSDP storage for numerical stability.
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # Raw VAE-output latent shape for 1024x1024 inputs.
    # For 512x512 inputs, switch to [32, 64, 64].
    config.model.input_shape = [32, 128, 128]

    # FLUX.2-klein-base-9B (FLUX Non-Commercial v2.1, undistilled). The
    # prebuilt config already pins `model_id_or_local_path` and `guidance_scale=None`
    # (no embedded guidance vector).
    config.model.net = copy.deepcopy(Flux2KleinBase9BConfig)
    config.trainer.fsdp = True
    config.model.meta_init = True

    # Sample-time CFG for the visualization sampler. Matches the BFL
    # reference call for klein-base-9B:
    #   `Flux2KleinPipeline(prompt, guidance_scale=4.0, num_inference_steps=50)`.
    config.model.guidance_cfg = Flux2.default_guidance_cfg()

    # Optimizer
    config.model.net_optimizer.lr = 1e-5

    # Timestep sampling: rectified-flow `shifted` distribution biases t
    # toward higher noise levels (closer to the FLUX.2 inference schedule
    # than plain uniform). shift=3 is conservative; alternatives 4 or 5
    # push the high-noise regime harder.
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 3
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Visualization sampler steps: BFL defaults to 50 steps for dev and klein-base;
    # 28 is BFL's suggested trade-off for dev. See:
    # https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B
    config.model.student_grid_cfg.steps = 50

    # Dataloader: raw-image dataset (point datatags at your shards).
    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    config.dataloader_train.batch_size = 1
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    # Trainer
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "flux2_klein_base_9b_sft"

    return config
