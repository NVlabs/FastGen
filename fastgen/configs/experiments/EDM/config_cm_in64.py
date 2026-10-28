# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from omegaconf import DictConfig

import fastgen.configs.methods.config_cm as config_cm_default
from fastgen.configs.data import ImageNet64_Loader_Config
from fastgen.configs.net import EDM_ImageNet64_Config, CKPT_ROOT_DIR
from fastgen.configs.instantiate import LazyCall as L
from fastgen.utils.lr_scheduler import LambdaInverseSquareRootScheduler
from fastgen.configs.callbacks import EMA_POWER_CALLBACKS

""" Configs for the CM model, on EDM (ADM U-Net) and ImageNet-64 dataset. """


def create_config():
    config = config_cm_default.create_config()

    # trainer
    # recommended setting for ImageNet-64 is max_iter * batch_size // (4 * 1000)
    config.trainer.callbacks.ct_schedule.kimg_per_stage = 3200
    config.trainer.callbacks.ct_schedule.q = 4
    config.trainer.callbacks.ct_schedule.ratio_limit = 0.9961

    # ema
    config.model.use_ema = ["ema_1", "ema_5", "ema_10"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_POWER_CALLBACKS)

    # model
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm-imagenet-64x64-cond-adm.pth"
    config.model.input_shape = [3, 64, 64]
    config.model.sample_t_cfg.train_p_mean = -0.8
    config.model.sample_t_cfg.train_p_std = 1.6
    config.model.loss_config.huber_const = 0.06
    config.model.loss_config.weighting_ct_loss = "c_out_sq"

    # (optional) EDM's typical FP16 training: EDM's FP16 mode casts the network input and weights to FP16
    # without autocast, and keeps FP32 master weights.
    # config.model.precision_model = "float16"
    # config.model.precision_opt = "float32"
    # config.model.grad_scaler_enabled = True
    # config.model.grad_scaler_growth_interval = 20000

    # Recommended setting for 2-step:
    # config.model.student_grid_cfg.grid_type = "explicit"
    # config.model.student_grid_cfg.grid_kwargs = {"timesteps": [80.0, 1.526, 0.0]}
    # config.model.student_grid_cfg.steps = 2

    config.model.net = copy.deepcopy(EDM_ImageNet64_Config)
    config.model.net.backbone.dropout = 0.2
    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 1e-3
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler = L(LambdaInverseSquareRootScheduler)(
        warm_up_steps=0,
        decay_steps=2000,
    )
    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003

    # dataloader
    config.dataloader_train = copy.deepcopy(ImageNet64_Loader_Config)

    config.log_config.group = "edm_imagenet64_cm"
    return config
