# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDD recipe for SiT-XL+REPA on ImageNet-256.

For the Euler variant, set ``GRID_SIZE = 128`` and ``TYPE_RK = "euler"``
below. The block bounds and number of supervised heads are derived from those
values and therefore need no separate edit.
"""

import copy

from omegaconf import DictConfig

import fastgen.configs.methods.config_pdd as config_pdd_default
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS
from fastgen.configs.data import ImageNet256_Loader_Config
from fastgen.configs.net import SIT_XL_2_REPA_256_URL, SiT_XL_2_REPA_256_Config
from fastgen.networks.DiT.network_pdd import DiTPDD

NFES = (1, 2, 4, 8)
GRID_SIZE = 64
TYPE_RK = "midpoint"


def create_config():
    config = config_pdd_default.create_config()

    config.model.input_shape = [4, 32, 32]
    config.model.precision_amp = "bfloat16"
    config.model.precision_amp_infer = "bfloat16"

    config.model.teacher = copy.deepcopy(SiT_XL_2_REPA_256_Config)

    config.model.net = copy.deepcopy(config.model.teacher)
    config.model.net._target_ = DiTPDD
    config.model.net.pdd_steps = GRID_SIZE
    config.model.student_grid_cfg.steps = 4
    config.model.student_grid_cfg.grid_type = "native"
    config.model.student_grid_cfg.grid_kwargs = {"base_steps": GRID_SIZE}

    # For best performance at NFE=1 pick 3.2 and at NFE=8 pick scale=2.7 instead of scale=2.9
    config.model.guidance_cfg.cfg.scale = [2.9, 1.0]
    config.model.guidance_cfg.cfg.boundary = 0.7

    config.model.loss_config.type_rk = TYPE_RK
    config.model.loss_config.num_supervised_heads = 2 if TYPE_RK == "euler" else 1
    config.model.loss_config.block_size_min = GRID_SIZE // max(NFES)
    config.model.loss_config.block_size_max = GRID_SIZE // min(NFES)

    config.model.pretrained_model_path = SIT_XL_2_REPA_256_URL

    config.model.net_optimizer.lr = 5e-5
    config.model.net_optimizer.betas = (0.9, 0.95)
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler.warm_up_steps = [0]

    config.model.use_ema = ["ema_99995"]
    config.trainer.callbacks = DictConfig(
        {key: value for key, value in config.trainer.callbacks.items() if not key.startswith("ema")}
    )
    config.trainer.callbacks.update({k: EMA_CONST_CALLBACKS[k] for k in config.model.use_ema})
    config.trainer.callbacks.grad_clip.grad_norm = 1

    config.dataloader_train = copy.deepcopy(ImageNet256_Loader_Config)

    config.trainer.batch_size_global = 2048
    config.trainer.max_iter = 300000
    config.trainer.save_ckpt_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.callbacks.wandb.sample_logging_iter = 1000

    config.log_config.group = "pdd_repa_xl"
    return config
