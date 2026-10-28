# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-run PDD recipe for Wan2.1 14B.

This variant uses the Midpoint solver with the 14B backbone and CFG skip layer
12, trained for 3.5k iterations. For the Euler variant, set ``GRID_SIZE = 256``
and ``TYPE_RK = "euler"`` below; the block bounds and number of supervised
heads are derived automatically.
"""

import copy

import fastgen.configs.experiments.WanT2V.config_pdd as base
from fastgen.configs.net import Wan21_T2V_14B_Config
from fastgen.networks.Wan.network_pdd import WanPDD

NFES = (2, 4, 8)
GRID_SIZE = 128
TYPE_RK = "midpoint"


def create_config():
    config = base.create_config()
    config.model.meta_init = True

    config.model.teacher = copy.deepcopy(Wan21_T2V_14B_Config)
    # Schedule/grid endpoint.
    config.model.teacher.max_t = 0.9999
    config.model.net = copy.deepcopy(config.model.teacher)
    config.model.net._target_ = WanPDD
    config.model.net.max_t = 0.9999
    config.model.net.pdd_steps = GRID_SIZE
    config.model.student_grid_cfg.grid_kwargs = {"base_steps": GRID_SIZE, "shift": 6.0}

    config.model.loss_config.type_rk = TYPE_RK
    config.model.loss_config.num_supervised_heads = 2 if TYPE_RK == "euler" else 1
    config.model.loss_config.block_size_min = GRID_SIZE // max(NFES)
    config.model.loss_config.block_size_max = GRID_SIZE // min(NFES)
    config.model.guidance_cfg.cfg.skip_layers = [12]

    # EMA can lead to better performance on long runs, at the memory cost of
    # another sharded FP32 copy of the 14B student.
    # config.model.use_ema = ["ema"]
    # config.trainer.callbacks.ema.type = "constant"
    # config.trainer.callbacks.ema.beta = 0.99

    config.trainer.max_iter = 3500
    config.trainer.save_ckpt_iter = 25
    config.trainer.callbacks.wandb.sample_logging_iter = 50

    config.log_config.group = "wan14b_pdd"
    return config
