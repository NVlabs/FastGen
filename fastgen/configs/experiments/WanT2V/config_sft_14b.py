# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.net import Wan21_T2V_14B_Config
from fastgen.networks.Wan.network import Wan

"""Configs for SFT on WAN 14B model."""


def create_config():
    config = config_sft_default.create_config()
    config.trainer.fsdp = True
    config.model.meta_init = True

    config.trainer.logging_iter = 500
    config.model.net_optimizer.lr = 1e-5
    config.model.student_grid_cfg.steps = 50

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"

    # VAE compress ratio for WAN: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]  # cthw
    config.model.net = copy.deepcopy(Wan21_T2V_14B_Config)
    config.model.guidance_cfg = Wan.default_guidance_cfg()
    # Precomputed latents + text-embedding conditioning: the VAE/text-encoder are not
    # needed at train time (sample logging builds a VAE on demand). Matches the other
    # latent T2V configs (e.g. config_sft_causal).
    config.model.enable_preprocessors = False

    config.dataloader_train = copy.deepcopy(VideoLatentLoaderConfig)
    config.dataloader_train.batch_size = 1

    config.log_config.group = "wan21_14b_sft"
    return config
