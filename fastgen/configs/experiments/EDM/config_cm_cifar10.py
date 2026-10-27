# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig
from fastgen.configs.methods.config_cm import create_config as cm_create_config
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS
from fastgen.configs.net import CKPT_ROOT_DIR


def create_config():
    config = cm_create_config()

    # Recommended setting for 2-step:
    # config.model.student_grid_cfg.grid_type = "explicit"
    # config.model.student_grid_cfg.grid_kwargs = {"timesteps": [80.0, 0.821, 0.0]}
    # config.model.student_grid_cfg.steps = 2

    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cifar10/edm-cifar10-32x32-uncond-vp.pth"

    config.model.use_ema = ["ema_9999", "ema_99995", "ema_9996"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    config.trainer.max_iter = 350000
    config.trainer.batch_size_global = 512

    return config
