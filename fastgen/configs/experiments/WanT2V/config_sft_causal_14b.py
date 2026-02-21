# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fastgen.configs.methods.config_sft_causal as config_sft_causal_default
from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.net import CausalWan_14B_Config
from fastgen.utils import LazyCall as L
from fastgen.methods import CausalSFTModel


def create_config():
    config = config_sft_causal_default.create_config()
    config.model_class = L(CausalSFTModel)(config=None)
    config.model.fsdp_meta_init = True

    config.trainer.logging_iter = 100
    config.model.net_optimizer.lr = 5e-5
    config.model.guidance_scale = 5.0
    config.model.student_sample_steps = 50

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"

    # VAE compress ratio for WAN: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]  # cthw
    config.model.net = CausalWan_14B_Config
    config.model.enable_preprocessors = False

    config.dataloader_train = VideoLatentLoaderConfig
    config.dataloader_train.batch_size = 1

    # 480p (832x480) resolution
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.trainer.max_iter = 5000
    config.trainer.save_ckpt_iter = 500
    config.trainer.validation_iter = 500

    config.log_config.group = "wan_14b_sft_ar_df"
    return config
