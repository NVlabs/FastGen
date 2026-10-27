# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import attrs
from omegaconf import DictConfig

from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.methods.config_dmd2 import (
    Config as DMD2Config,
    ModelConfig as DMD2ModelConfig,
)
from fastgen.methods import CausVidModel
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    EMA_CALLBACK,
    GradClip_CALLBACK,
)


@attrs.define
class ModelConfig(DMD2ModelConfig):
    # Time to which chunk-by-chunk generation re-noises each finished chunk before writing it to the
    # KV cache; None caches the clean chunk.
    context_noise: float | None = None


@attrs.define
class Config(DMD2Config):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(CausVidModel)(config=None))


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
