# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from omegaconf import DictConfig

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import CIFAR10_Loader_Config
from fastgen.configs.net import EDM_CIFAR10_Config, CKPT_ROOT_DIR
from fastgen.networks.EDM.network import EDMPrecond
from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.callbacks import EMA_POWER_CALLBACKS
from fastgen.datasets.augment import AugmentPipe

"""Configs for SFT (Supervised Fine-Tuning) on EDM and CIFAR-10 dataset."""


def create_config():
    config = config_sft_default.create_config()

    # model
    # EDM works on raw pixel space: [C, H, W] = [3, 32, 32]
    config.model.input_shape = [3, 32, 32]

    # EDM uses sigma-based noise schedule (log-normal for training)
    config.model.sample_t_cfg.time_dist_type = "lognormal"
    config.model.sample_t_cfg.train_p_mean = -1.2
    config.model.sample_t_cfg.train_p_std = 1.2

    # Pretrained model path
    # EDM-CIFAR10 checkpoint from Karras et al. (converted to .pth state dict)
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cifar10/edm-cifar10-32x32-cond-vp.pth"

    # Network config
    config.model.net = copy.deepcopy(EDM_CIFAR10_Config)
    config.model.net.backbone.dropout = 0.13
    config.model.guidance_cfg = EDMPrecond.default_guidance_cfg()

    # Optimizer config (lower LR for fine-tuning; EDM trains CIFAR-10 from scratch at 1e-3)
    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 2e-5
    config.model.net_optimizer.betas = (0.9, 0.999)
    config.model.net_optimizer.weight_decay = 0.0

    # EMA config
    config.model.use_ema = ["ema_1", "ema_5", "ema_10"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_POWER_CALLBACKS)

    # Data augmentation of EDM's CIFAR-10 recipe, which the pretrained augment-label embedding expects:
    # xflip(1) + yflip(1) + scale(1) + rotate_frac(2) + aniso(2) + translate_frac(2) = augment_dim 9
    config.trainer.augment_pipe = L(AugmentPipe)(
        p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
    )
    config.model.net.backbone.augment_dim = 9

    # Dataloader (x-flips come labeled from the augment pipe, so no dataset x-flips)
    config.dataloader_train = copy.deepcopy(CIFAR10_Loader_Config)

    # Sampling config for visualization
    config.model.student_grid_cfg.steps = 18  # EDM typically uses 18-50 steps

    # Trainer
    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False
    config.trainer.batch_size_global = 512
    config.trainer.max_iter = 100000
    config.trainer.save_ckpt_iter = 10000

    config.log_config.group = "edm_cifar10_sft"

    return config
