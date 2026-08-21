# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

import fastgen.configs.methods.config_mean_flow as config_mf_default
from fastgen.configs.data import ImageNet256_Loader_Config
from fastgen.configs.net import DiT_IN256_B_Config
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS

""" Configs for the MeanFlow model, on DiT-XL and ImageNet-256 dataset. """


def create_config():
    config = config_mf_default.create_config()

    # model
    config.model.input_shape = [4, 32, 32]
    config.model.precision_amp = "bfloat16"
    config.model.precision_amp_jvp = "float32"
    config.model.cond_dropout_prob = 0.1
    config.model.guidance_mixture_ratio = 0.5

    config.model.sample_t_cfg.time_dist_type = "logitnormal"
    config.model.sample_t_cfg.train_p_mean = -0.4
    config.model.sample_t_cfg.train_p_std = 1.0
    config.model.sample_t_cfg.min_t = 0.0
    config.model.sample_t_cfg.max_t = 0.999
    config.model.sample_t_cfg.flow_matching_ratio = 0.75

    config.model.loss_config.norm_method = "poly_1.0"
    config.model.loss_config.norm_const = 1.0
    config.model.loss_config.tangent_warmup_steps = 0
    config.model.loss_config.loss_type = "l2"

    config.model.net = DiT_IN256_B_Config
    # remove the additional 1000 factor in JVP
    config.model.net.scale_t = False
    config.model.net.r_timestep = True
    config.model.net.time_cond_type = "diff"

    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 1e-4
    config.model.net_optimizer.betas = (0.9, 0.95)
    config.model.net_optimizer.weight_decay = 0.0

    # ema
    config.model.use_ema = ["ema_9999", "ema_99995", "ema_9996"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    # Recommended setting for 2-step:
    # config.model.sample_t_cfg.t_list = [0.999, 0.5, 0.0]
    # config.model.student_sample_steps = 2

    # dataloader
    config.dataloader_train = ImageNet256_Loader_Config
    config.dataloader_train.batch_size = 32

    # trainer
    config.trainer.batch_size_global = 1024
    config.trainer.max_iter = 1200000
    config.trainer.save_ckpt_iter = 50000
    config.trainer.logging_iter = 10000

    config.log_config.group = "imagenet256"

    return config
