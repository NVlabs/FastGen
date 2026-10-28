# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for the TCM model, on CIFAR-10 dataset."""

import attrs

from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.methods.config_cm import (
    ModelConfig as CMModelConfig,
    create_config as cm_create_config,
)

from fastgen.methods import TCMModel


@attrs.define
class ModelConfig(CMModelConfig):
    # TCM keeps a frozen copy of the student alongside the regular teacher.
    extra_fsdp_modules: list[str] = attrs.field(factory=lambda: ["teacher", "cm_teacher"])

    # probability of sampling the boundary time step in TCM
    boundary_prob: float = 0.25
    # weighting coefficient for the boundary loss
    w_boundary: float = 0.1
    # critical time step at boundary
    transition_t: float = 1.0


def create_config():
    config = cm_create_config()
    config.model_class = L(TCMModel)(
        config=None,
    )
    config.model = ModelConfig()

    # specify the path to the stage-1 consistency model in trainer.checkpointer.pretrained_ckpt_path
    # here we use the EMA model as initialization for the teacher and student models and we continue the EMA
    config.trainer.checkpointer.pretrained_ckpt_key_map = {"cm_teacher": "ema", "net": "ema", "ema": "ema"}

    # model (from CM)
    config.model.net.backbone.dropout = 0.2
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.net_optimizer.weight_decay = 0.0
    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003
    config.model.net.backbone.label_dim = 0

    # model (TCM-specific)
    config.model.net_optimizer.lr = 5e-5
    config.model.sample_t_cfg.train_p_mean = 0.0
    config.model.sample_t_cfg.train_p_std = 0.2
    config.model.sample_t_cfg.log_t_df = 0.01
    config.model.sample_t_cfg.time_dist_type = "log_t"
    # lowest value in truncated range (there is no gradient for t < transition_t)
    config.model.sample_t_cfg.min_t = config.model.transition_t
    # midpoint sampling (recommended setting for 2-step: [1.0])
    # config.model.student_grid_cfg.grid_type = "explicit"
    # config.model.student_grid_cfg.grid_kwargs = {"timesteps": [80.0, 1.0, 0.0]}
    # config.model.student_grid_cfg.steps = 2

    # trainer (recommended: low value such that ratio equals ct_schedule.ratio_limit)
    config.trainer.callbacks.ct_schedule.kimg_per_stage = 1000

    return config
