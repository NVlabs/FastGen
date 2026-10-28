# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_kd as config_kd_default
from fastgen.configs.data import PairLoaderConfig
from fastgen.configs.net import Wan_1_3B_Config

""" Configs for the KD model on Wan-1.3B model. """


def create_config():
    config = config_kd_default.create_config()
    config.trainer.max_iter = 6000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 1000

    config.model.net_optimizer.lr = 7e-5

    config.model.input_shape = [16, 21, 60, 104]
    config.model.net = copy.deepcopy(Wan_1_3B_Config)
    config.model.enable_preprocessors = False
    config.model.precision = "bfloat16"

    config.dataloader_train = copy.deepcopy(PairLoaderConfig)
    config.dataloader_train.batch_size = 2

    config.log_config.group = "wan_kd"
    return config
