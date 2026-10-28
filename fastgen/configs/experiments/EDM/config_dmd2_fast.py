# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.methods.config_dmd2 import create_config as dmd2_create_config
from fastgen.configs.net import CKPT_ROOT_DIR


def create_config():
    config = dmd2_create_config()

    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cifar10/edm-cifar10-32x32-cond-vp.pth"
    # Polynomial draws within entries 2 and 998 of EDM's 1000-entry polynomial table (rho = 7).
    config.model.sample_t_cfg.min_t, config.model.sample_t_cfg.max_t = 0.00210147, 79.5638

    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 50
    config.trainer.save_ckpt_iter = 1000
    config.trainer.batch_size_global = 64

    config.dataloader_train.batch_size = 64

    return config
