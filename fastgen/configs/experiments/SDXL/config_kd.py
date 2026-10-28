# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_kd as config_kd_default
from fastgen.configs.data import PairLoaderConfig
from fastgen.configs.net import SDXLConfig

""" Configs for the KD model on SDXL. """


def create_config():
    config = config_kd_default.create_config()

    config.trainer.max_iter = 6000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 1000

    config.model.net_optimizer.lr = 1e-5

    config.model.input_shape = [4, 128, 128]
    config.model.net = copy.deepcopy(SDXLConfig)
    config.model.enable_preprocessors = False

    config.dataloader_train = copy.deepcopy(PairLoaderConfig)
    config.dataloader_train.batch_size = 16

    config.log_config.group = "sdxl_kd"
    return config
