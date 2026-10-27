# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from omegaconf import DictConfig

import fastgen.configs.methods.config_tcm as config_tcm_default
from fastgen.configs.data import ImageNet64_EDMV2_Loader_Config
from fastgen.configs.net import EDM2_IN64_S_Config, CKPT_ROOT_DIR
from fastgen.configs.instantiate import LazyCall as L
from fastgen.utils.lr_scheduler import LambdaInverseSquareRootScheduler
from fastgen.configs.callbacks import ForcedWeightNorm_CALLBACK, EMA_POWER_CALLBACKS

""" Configs for the TCM model, on EDM2 and ImageNet-64 dataset. """


def create_config():
    config = config_tcm_default.create_config()

    # trainer
    config.trainer.callbacks.ct_schedule.q = 4
    config.trainer.callbacks.ct_schedule.ratio_limit = 0.9961

    # ema
    config.model.use_ema = ["ema_1", "ema_5", "ema_10"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_POWER_CALLBACKS)

    # model
    # (optional) The reference's FP16 training: EDM2's FP16 mode casts the network input and weights to FP16
    # without autocast, keeps FP32 master weights, and scales the loss dynamically.
    # config.model.precision_model = "float16"
    # config.model.precision_opt = "float32"
    # config.model.grad_scaler_enabled = True
    # config.model.grad_scaler_init_scale = 16
    # config.model.grad_scaler_growth_interval = 20000
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm2-img64-s-fid.pth"
    config.model.input_shape = [3, 64, 64]
    config.model.sample_t_cfg.train_p_mean = 0.0
    config.model.sample_t_cfg.train_p_std = 0.2
    config.model.loss_config.huber_const = 0.06
    config.model.loss_config.weighting_ct_loss = "c_out_sq"
    # Checkpoint from the first-stage CM training
    # config.trainer.checkpointer.pretrained_ckpt_path = ""
    # Choose keys from checkpoint for teacher, student, and EMA models
    config.trainer.checkpointer.pretrained_ckpt_key_map = {
        "cm_teacher": "net",
        "net": "net",
        "ema_1": "net",
        "ema_5": "net",
        "ema_10": "net",
    }

    # Recommended setting for 2-step:
    # config.model.student_grid_cfg.grid_type = "explicit"
    # config.model.student_grid_cfg.grid_kwargs = {"timesteps": [80.0, 1.526, 0.0]}
    # config.model.student_grid_cfg.steps = 2

    config.model.net = copy.deepcopy(EDM2_IN64_S_Config)
    # Important!!! EDM2 needs the ForcedWeightNorm callback
    config.trainer.callbacks.update(ForcedWeightNorm_CALLBACK)
    config.model.net.dropout = 0.4
    config.model.net.dropout_resolutions = [16, 8]
    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 5e-4
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler = L(LambdaInverseSquareRootScheduler)(
        warm_up_steps=0,
        decay_steps=8000,
    )
    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003

    # dataloader
    config.dataloader_train = copy.deepcopy(ImageNet64_EDMV2_Loader_Config)

    config.trainer.batch_size_global = 1024
    config.trainer.max_iter = 400000
    config.log_config.group = "edm2_imagenet64_tcm"
    return config
