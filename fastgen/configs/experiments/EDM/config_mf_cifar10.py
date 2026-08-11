# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig
from fastgen.configs.methods.config_mean_flow import create_config as mf_create_config
from fastgen.utils import LazyCall as L
from fastgen.datasets.augment import AugmentPipe
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS

""" Configs for the MeanFlow model, on CIFAR-10 dataset, imitating the v1 version of https://github.com/Gsunshine/py-meanflow"""


def create_config():
    config = mf_create_config()

    # Recommended setting for 2-step:
    # config.model.sample_t_cfg.t_list = [0.999, 0.5, 0.0]
    # config.model.student_sample_steps = 2

    config.model.sample_t_cfg.train_p_mean = -0.6
    config.model.sample_t_cfg.train_p_std = 1.6
    config.model.sample_t_cfg.flow_matching_ratio = 0.25
    config.model.sample_t_cfg.time_dist_type = "logitnormal"
    config.model.sample_t_cfg.min_t = 0.0
    config.model.sample_t_cfg.max_t = 0.999

    config.model.sample_r_cfg.enabled = True
    config.model.sample_r_cfg.time_dist_type = "logitnormal"
    config.model.sample_r_cfg.train_p_mean = -4.0
    config.model.sample_r_cfg.train_p_std = 1.6
    config.model.sample_r_cfg.min_t = 0.0
    config.model.sample_r_cfg.max_t = 0.999

    config.model.loss_config.norm_method = "poly_0.75"

    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 6e-4
    config.model.net_optimizer.betas = (0.9, 0.999)
    config.model.net.drop_precond = "both"
    config.model.net_scheduler.warm_up_steps = [(200 * 50000) // 1024]  # 200 epochs for CIFAR-10 with batch size 1024
    config.model.net_scheduler.f_start = [1e-8 / config.model.net_optimizer.lr]

    config.model.net.dropout = 0.2
    config.model.net.schedule_type = "rf"
    config.model.net.net_pred_type = "flow"
    config.model.loss_config.norm_const = 1e-3
    config.model.loss_config.loss_type = "l2"
    config.model.cond_dropout_prob = 0

    # ema
    config.model.use_ema = ["ema_9999", "ema_99995", "ema_9996"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    # data
    config.trainer.augment_pipe = L(AugmentPipe)(
        p=0.12, xflip=1e8, yflip=0, scale=1, rotate_frac=0, aniso=1, translate_frac=1
    )
    config.model.net.augment_dim = 6
    config.dataloader_train.xflip = True

    # trainer
    config.trainer.batch_size_global = 1024
    config.trainer.max_iter = 1000000
    config.trainer.save_ckpt_iter = 50000
    config.trainer.logging_iter = 10000

    return config
