# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import attrs
from omegaconf import DictConfig

from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.methods.config_dmd2 import (
    Config as DMD2Config,
    ModelConfig as DMD2ModelConfig,
)
from fastgen.methods import FdistillModel
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
)


@attrs.define
class FdistillConfig:
    # f-div type
    f_div: str = "js"

    # ratio clipping range
    ratio_lower: float = 0.1
    ratio_upper: float = 20.0

    # ratio normalization: normalize the density ratio with EMA history
    ratio_normalization: bool = True
    bin_num: int = 10
    ratio_ema_rate: float = 0.0


@attrs.define
class ModelConfig(DMD2ModelConfig):
    # hyperparameters specific to the f-distill
    f_distill: FdistillConfig = attrs.field(factory=FdistillConfig)


@attrs.define
class Config(DMD2Config):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(FdistillModel)(config=None))


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **BASE_CALLBACKS,
        }
    )

    config.dataloader_train.batch_size = 256
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.sample_t_cfg.time_dist_type = "polynomial"

    return config
