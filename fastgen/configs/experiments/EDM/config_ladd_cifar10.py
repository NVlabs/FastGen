# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-step LADD on class-conditional CIFAR-10 (best FID-50k: 2.08 at 50k iterations).

This config builds on FastGen's LADD defaults with the EDM CIFAR-10 setup, discriminator
Adam beta1 = 0, and the existing polynomial distribution for noising inputs.
"""

from omegaconf import DictConfig

from fastgen.configs.callbacks import EMA_CONST_CALLBACKS
from fastgen.configs.methods.config_ladd import create_config as ladd_create_config
from fastgen.configs.net import CKPT_ROOT_DIR


def create_config():
    config = ladd_create_config()

    # Match the EDM CIFAR-10 teacher, noise range, and EMA setup used by DMD2.
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cifar10/edm-cifar10-32x32-cond-vp.pth"
    config.model.sample_t_cfg.min_t, config.model.sample_t_cfg.max_t = 0.00210147, 79.5638
    config.model.use_ema = ["ema_9999", "ema_99995", "ema_9996"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    # Tune the discriminator optimizer and its input noise sampling.
    config.model.discriminator_optimizer.betas = [0.0, 0.999]
    config.model.sample_t_cfg.time_dist_type = "polynomial"

    config.trainer.seed = 1000
    config.trainer.ddp = True
    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False
    config.trainer.max_iter = 100000
    config.trainer.save_ckpt_iter = 5000
    config.log_config.group = "cifar10_ladd"
    return config
