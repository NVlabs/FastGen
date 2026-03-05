# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan21_I2V_14B_480P_Config
from fastgen.utils import LazyCall as L
from fastgen.methods import SFTModel

""" Configs for SFT on Wan-14B model. """


def create_config():
    config = config_sft_default.create_config()
    config.model_class = L(SFTModel)(config=None)
    config.model.fsdp_meta_init = True

    config.trainer.logging_iter = 100
    config.model.net_optimizer.lr = 5e-5
    config.model.guidance_scale = 5.0
    config.model.student_sample_steps = 50

    config.model.precision = "bfloat16"

    # VAE compress ratio for WAN: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]  # cthw
    config.model.net = Wan21_I2V_14B_480P_Config

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # The loader needs to include the raw video since we need the first frame in pixel space
    config.dataloader_train = VideoLoaderConfig
    config.dataloader_train.batch_size = 1

    # 480p (832x480) resolution
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.log_config.group = "wan21_14b_i2v_sft"
    return config
