# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageEditLoaderConfig
from fastgen.configs.net import Flux2KleinBase4BImageEditConfig
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.Flux2.network import Flux2

"""SFT (instruction-tuned image editing) config for FLUX.2-klein-base-4B."""


def create_config():
    config = config_sft_default.create_config()

    # Compute / storage precision: bf16 compute, fp32 FSDP storage to avoid
    # weight drift over long SFT runs.
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # Raw VAE-output latent shape for 1024x1024 inputs
    config.model.input_shape = [32, 128, 128]

    # Instruction-tuned image editing on klein-base-4B.
    config.model.net = copy.deepcopy(Flux2KleinBase4BImageEditConfig)
    config.trainer.fsdp = True
    config.model.meta_init = True

    # Optimizer
    config.model.net_optimizer.lr = 1e-5

    # Rectified-flow timestep sampler (biased toward higher noise to match
    # the FLUX.2 inference schedule).
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 3
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Sample-time CFG used by the visualization sampler.
    config.model.guidance_cfg = Flux2.default_guidance_cfg()
    config.model.student_grid_cfg.steps = 50  # klein-base default

    # Image-editing dataset.
    config.dataloader_train = copy.deepcopy(ImageEditLoaderConfig)
    config.dataloader_train.batch_size = 2
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    # Conditional dropout: with non-zero probability we replace the text
    # embedding with the negative (empty-string) embedding during training,
    # which lets the trained student support classifier-free guidance at
    # sample time. We do NOT drop the reference image (``cond_keys_no_dropout``)
    # because the editing task is fundamentally conditioned on it.
    config.model.cond_dropout_prob = 0.1
    config.model.cond_keys_no_dropout = ["condition_image"]

    # Trainer
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "flux2_klein_base_4b_image_edit_sft"

    return config
