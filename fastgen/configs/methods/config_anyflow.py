# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config schema for the AnyFlow on-policy method (paper Stage 3).

AnyFlow's on-policy stage is DMD2 with a flow-map student that generates via
a multi-step rollout-with-gradient, so the config is the DMD2 model config
unchanged. The flow-map pretrain stage (paper Stage 2) is MeanFlow with
AnyFlow's hyperparameters — see ``configs/experiments/WanT2V/config_anyflow.py``.
"""

import attrs
from omegaconf import DictConfig

from fastgen.configs.callbacks import (
    EMA_CALLBACK,
    GPUStats_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    TrainProfiler_CALLBACK,
    WANDB_CALLBACK,
)
from fastgen.configs.config import BaseConfig
from fastgen.configs.methods.config_dmd2 import ModelConfig as DMD2ModelConfig
from fastgen.methods import AnyFlowModel
from fastgen.utils import LazyCall as L


@attrs.define(slots=False)
class ModelConfig(DMD2ModelConfig):
    """AnyFlow on-policy model config — identical to DMD2's."""


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(AnyFlowModel)(
        config=None,
    )


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **EMA_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **WANDB_CALLBACK,
        }
    )

    # The student is a flow-map network with a dual-timestep input.
    config.model.use_ema = True
    config.model.net.r_timestep = True
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.discriminator_scheduler.warm_up_steps = [0]

    return config
