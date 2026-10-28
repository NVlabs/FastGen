# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDD recipe for Wan2.1 1.3B.

For the Euler variant, set ``GRID_SIZE = 256`` and ``TYPE_RK = "euler"``
below. The block bounds and number of supervised heads are derived from those
values and therefore need no separate edit.
"""

import copy

import fastgen.configs.methods.config_pdd as config_pdd_default
from fastgen.callbacks.carry import CarryCallback
from fastgen.configs.data import PromptLoaderConfig
from fastgen.configs.net import Wan_1_3B_Config
from fastgen.networks.Wan.network import Wan
from fastgen.networks.Wan.network_pdd import WanPDD
from fastgen.configs.instantiate import LazyCall as L

NFES = (2, 4, 8)
GRID_SIZE = 128
TYPE_RK = "midpoint"


def create_config():
    config = config_pdd_default.create_config()

    config.trainer.fsdp = True
    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"
    config.model.precision_fsdp_reduce = "bfloat16"
    config.model.input_shape = [16, 21, 60, 104]

    config.model.teacher = copy.deepcopy(Wan_1_3B_Config)
    # Schedule/grid endpoint.
    config.model.teacher.max_t = 0.9999
    config.model.net = copy.deepcopy(config.model.teacher)
    config.model.net._target_ = WanPDD
    config.model.net.max_t = 0.9999
    config.model.net.pdd_steps = GRID_SIZE
    config.model.student_grid_cfg.steps = 4
    config.model.student_grid_cfg.grid_type = "native"
    config.model.student_grid_cfg.grid_kwargs = {"base_steps": GRID_SIZE, "shift": 6.0}

    config.model.guidance_cfg = Wan.default_guidance_cfg()
    config.model.guidance_cfg.cfg.skip_layers = [10]

    config.model.loss_config.type_rk = TYPE_RK
    config.model.loss_config.num_supervised_heads = 2 if TYPE_RK == "euler" else 1
    config.model.loss_config.block_size_min = GRID_SIZE // max(NFES)
    config.model.loss_config.block_size_max = GRID_SIZE // min(NFES)
    # Paper data-free training keeps all ranks on one shared trajectory phase.
    config.model.backward_simulation = "synced"
    config.trainer.callbacks.carry = L(CarryCallback)()

    config.model.net_optimizer.lr = 1e-5
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler.warm_up_steps = [0]
    config.trainer.callbacks.grad_clip.grad_norm = 1

    # Generic prompt-only training loader.
    config.dataloader_train = copy.deepcopy(PromptLoaderConfig)
    config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_wan"}

    config.trainer.batch_size_global = 256
    config.trainer.max_iter = 250
    config.trainer.logging_iter = 25
    config.trainer.save_ckpt_iter = 25

    config.log_config.group = "wan_pdd"
    return config
