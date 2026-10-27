# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for chunkwise causal teacher-forcing SFT (with EMA) on the Wan-14B model,
initialized from the bidirectional pretrained weights."""

import copy

import fastgen.configs.experiments.WanT2V.config_sft_teacher_forcing as config_sft_teacher_forcing_default

from fastgen.configs.net import CausalWan_14B_Config


def create_config():
    config = config_sft_teacher_forcing_default.create_config()

    config.model.meta_init = True
    config.model.net = copy.deepcopy(CausalWan_14B_Config)

    config.dataloader_train.batch_size = 1

    config.trainer.callbacks.ema.start_iter = 200
    config.log_config.group = "wan21_14b_sft_teacher_forcing"
    return config
