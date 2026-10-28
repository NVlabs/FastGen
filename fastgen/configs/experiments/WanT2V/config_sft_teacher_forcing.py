# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for chunkwise causal teacher-forcing SFT (with EMA) on the Wan-1.3B model."""

import copy

from fastgen.configs.callbacks import EMA_CALLBACK
import fastgen.configs.methods.config_sft_teacher_forcing as config_sft_teacher_forcing_default
from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.net import CausalWan_1_3B_Config


def create_config():
    config = config_sft_teacher_forcing_default.create_config()
    config.trainer.max_iter = 50000
    config.trainer.logging_iter = 500
    config.trainer.save_ckpt_iter = 500

    config.model.use_ema = ["ema"]
    config.trainer.callbacks.update(EMA_CALLBACK)
    config.trainer.callbacks.ema.beta = 0.999
    config.trainer.callbacks.ema.start_iter = 1000

    config.model.net_optimizer.lr = 2e-6
    config.model.net_optimizer.betas = (0.0, 0.999)
    config.model.guidance_cfg.cfg.scale = 3.0
    config.model.student_grid_cfg.steps = 50

    config.trainer.fsdp = True
    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"
    config.model.input_shape = [16, 21, 60, 104]
    config.model.net = copy.deepcopy(CausalWan_1_3B_Config)
    config.model.enable_preprocessors = False

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.0
    config.model.sample_t_cfg.max_t = 0.999

    config.dataloader_train = copy.deepcopy(VideoLatentLoaderConfig)
    config.dataloader_train.batch_size = 2
    config.dataloader_train.num_workers = 1

    config.log_config.group = "wan_sft_teacher_forcing"
    return config
