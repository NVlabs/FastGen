# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import attrs
from omegaconf import DictConfig

from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.methods.config_causvid import ModelConfig as CausVidModelConfig
from fastgen.configs.methods.config_dmd2 import (
    Config as DMD2Config,
)
from fastgen.methods import SelfForcingModel
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
    EMA_CALLBACK,
)


@attrs.define
class ModelConfig(CausVidModelConfig):
    # Self-Forcing starts from fresh noise and completes its causal rollout in-iteration, so it
    # requires the carry-free "default" mode. Real data is optional, although inherited GAN
    # terms still use it when both present and weighted.
    backward_simulation: str | None = "default"
    start_gradient_frame: int = 0
    same_step_across_blocks: bool = True
    last_step_only: bool = False
    grad_through_kv_cache: bool = False


@attrs.define
class Config(DMD2Config):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(SelfForcingModel)(config=None))


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **EMA_CALLBACK,
            **BASE_CALLBACKS,
        }
    )

    config.dataloader_train.batch_size = 256
    config.model.student_grid_cfg.steps = 4
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]

    return config
