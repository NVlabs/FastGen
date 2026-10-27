# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.methods.config_cm import create_config as cm_create_config
from fastgen.configs.net import CKPT_ROOT_DIR


def create_config():
    config = cm_create_config()

    # Recommended setting for 2-step:
    # config.model.student_grid_cfg.grid_type = "explicit"
    # config.model.student_grid_cfg.grid_kwargs = {"timesteps": [80.0, 0.821, 0.0]}
    # config.model.student_grid_cfg.steps = 2

    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cifar10/edm-cifar10-32x32-uncond-vp.pth"

    config.trainer.callbacks.ct_schedule.kimg_per_stage = 512000
    config.dataloader_train.batch_size = 128

    config.trainer.max_iter = 8000
    config.trainer.callbacks.ct_schedule.q = 256.0
    config.trainer.callbacks.ema.beta = 0.9993
    config.trainer.save_ckpt_iter = 500
    config.trainer.logging_iter = 100
    return config
