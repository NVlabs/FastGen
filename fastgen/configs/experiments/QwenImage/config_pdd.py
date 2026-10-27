# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel Decoding Distillation (PDD) recipe for Qwen-Image.

For the Euler variant, set ``GRID_SIZE = 256`` and ``TYPE_RK = "euler"``
below. The block bounds and number of supervised heads are derived from those
values and therefore need no separate edit.
"""

import copy

import fastgen.configs.methods.config_pdd as config_pdd_default
from fastgen.callbacks.carry import CarryCallback
from fastgen.configs.data import PromptLoaderConfig
from fastgen.configs.net import QwenImageConfig
from fastgen.networks.QwenImage.network import QwenImage
from fastgen.networks.QwenImage.network_pdd import QwenImagePDD
from fastgen.configs.instantiate import LazyCall as L

NFES = (2, 4, 8)
GRID_SIZE = 128
TYPE_RK = "midpoint"


def create_config():
    config = config_pdd_default.create_config()

    config.trainer.fsdp = True
    config.model.meta_init = True
    config.model.input_shape = [16, 128, 128]
    config.model.precision = "float32"
    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"

    config.model.teacher = copy.deepcopy(QwenImageConfig)
    config.model.net = copy.deepcopy(config.model.teacher)
    config.model.net._target_ = QwenImagePDD
    config.model.net.pdd_steps = GRID_SIZE
    config.model.student_grid_cfg.steps = 4
    config.model.student_grid_cfg.grid_type = "native"
    config.model.student_grid_cfg.grid_kwargs = {"base_steps": GRID_SIZE, "shift": 5.0}
    config.model.guidance_cfg = QwenImage.default_guidance_cfg()

    config.model.loss_config.type_rk = TYPE_RK
    config.model.loss_config.num_supervised_heads = 2 if TYPE_RK == "euler" else 1
    config.model.loss_config.block_size_min = GRID_SIZE // max(NFES)
    config.model.loss_config.block_size_max = GRID_SIZE // min(NFES)
    # Paper data-free training keeps all ranks on one shared trajectory phase.
    config.model.backward_simulation = "synced"
    config.trainer.callbacks.carry = L(CarryCallback)()

    config.model.net_optimizer.lr = 1e-5
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler.warm_up_steps = [0]

    # Generic prompt-only loader.
    config.dataloader_train = copy.deepcopy(PromptLoaderConfig)

    config.trainer.batch_size_global = 2048
    config.trainer.max_iter = 3000
    config.trainer.save_ckpt_iter = 250
    config.trainer.logging_iter = 100
    config.trainer.callbacks.wandb.sample_logging_iter = 250
    config.trainer.callbacks.grad_clip.grad_norm = 1

    config.log_config.group = "pdd_qwen_image"
    return config
