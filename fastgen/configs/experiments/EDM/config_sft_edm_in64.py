# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from omegaconf import DictConfig

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageNet64_Loader_Config
from fastgen.configs.net import EDM_ImageNet64_Config, CKPT_ROOT_DIR
from fastgen.configs.callbacks import EMA_POWER_CALLBACKS
from fastgen.networks.EDM.network import EDMPrecond

"""Configs for SFT (Supervised Fine-Tuning) on EDM and ImageNet-64 dataset."""


def create_config():
    config = config_sft_default.create_config()

    # model
    # EDM works on raw pixel space: [C, H, W] = [3, 64, 64]
    config.model.input_shape = [3, 64, 64]
    # EDM's FP16 mode: the network casts its input and weights to FP16, without autocast.
    config.model.precision_model = "float16"
    config.model.precision_opt = "float32"
    config.model.grad_scaler_enabled = True
    config.model.grad_scaler_growth_interval = 20000

    # EDM uses sigma-based noise schedule (log-normal for training)
    config.model.sample_t_cfg.time_dist_type = "lognormal"
    config.model.sample_t_cfg.train_p_mean = -1.2
    config.model.sample_t_cfg.train_p_std = 1.2

    # Pretrained model path
    # EDM-ImageNet64 checkpoint from Karras et al. (converted to .pth state dict)
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm-imagenet-64x64-cond-adm.pth"

    # Network config
    config.model.net = copy.deepcopy(EDM_ImageNet64_Config)
    config.model.net.backbone.dropout = 0.1  # Light dropout for regularization
    config.model.guidance_cfg = EDMPrecond.default_guidance_cfg()

    # Optimizer config (lower LR for fine-tuning; EDM trains ImageNet-64 from scratch at 1e-4)
    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 2e-5  # 5x lower for fine-tuning
    config.model.net_optimizer.betas = (0.9, 0.999)
    config.model.net_optimizer.weight_decay = 0.0

    # Power-function EMA for this fine-tuning recipe.
    config.model.use_ema = ["ema_1", "ema_5", "ema_10"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_POWER_CALLBACKS)

    # Dataloader
    config.dataloader_train = copy.deepcopy(ImageNet64_Loader_Config)

    # Sampling config for visualization
    config.model.student_grid_cfg.steps = 18  # EDM typically uses 18-50 steps

    # Trainer
    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False
    config.trainer.batch_size_global = 512
    config.trainer.max_iter = 200000
    config.trainer.save_ckpt_iter = 10000
    config.trainer.logging_iter = 1000

    config.log_config.group = "edm_imagenet64_sft"

    return config
