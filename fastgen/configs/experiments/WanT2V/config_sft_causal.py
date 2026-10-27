# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft_causal as config_sft_causal_default
from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.net import CausalWan_1_3B_Config
from fastgen.networks.Wan.network_causal import CausalWan

""" Configs for causal (autoregressive) SFT on Wan-1.3B model. """


def create_config():
    config = config_sft_causal_default.create_config()
    config.trainer.logging_iter = 100
    config.model.net_optimizer.lr = 1e-5
    config.model.student_grid_cfg.steps = 50

    config.model.precision_model = "bfloat16"

    # VAE compress ratio for WAN: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]  # cthw
    config.model.net = copy.deepcopy(CausalWan_1_3B_Config)
    config.model.guidance_cfg = CausalWan.default_guidance_cfg()
    config.model.enable_preprocessors = False

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.dataloader_train = copy.deepcopy(VideoLatentLoaderConfig)
    config.dataloader_train.batch_size = 1

    config.trainer.max_iter = 5000
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "wan_sft_ar_df"
    return config
