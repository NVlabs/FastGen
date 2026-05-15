# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config schema for the AnyFlow method.

AnyFlow inherits the DMD2 model config (so the on-policy stage gets fake_score
/ discriminator / alternating-step machinery for free) and adds a
``LossConfig`` describing the flow-map pretrain hyperparameters.
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
class LossConfig:
    """Hyperparameters for the AnyFlow flow-map loss and on-policy switch."""

    # Which stage to train. "pretrain" runs the central-difference flow-map
    # objective; "onpolicy" inherits DMD2's alternating distillation with
    # dual-timestep r=0 conditioning.
    training_stage: str = "pretrain"

    # Central-difference step size for estimating dF/dt. Lives in the same
    # units as the noise scheduler's timesteps. The default (5e-3) matches the
    # AnyFlow paper's choice of epsilon=5 with num_train_timesteps=1000.
    jvp_finite_diff_eps: float = 5e-3

    # Per-batch fraction with r = t (recovers pure flow matching).
    diffusion_ratio: float = 0.5
    # Per-batch fraction with r = min_t (forces consistency to clean data).
    consistency_ratio: float = 0.25

    # Per-timestep loss weighting scheme — passed through to the flow-map
    # scheduler. One of "gaussian", "beta08", "uniform".
    weight_type: str = "beta08"
    # Flow-matching schedule shift for the weighting / sampling scheduler.
    # Wan video defaults use 5.0; image use 1.0.
    shift: float = 1.0
    # Resolution of the discrete weighting grid; matches the AnyFlow reference.
    num_train_timesteps: int = 1000


@attrs.define(slots=False)
class ModelConfig(DMD2ModelConfig):
    """AnyFlow model config — inherits DMD2 fields, adds the flow-map loss config."""

    loss_config: LossConfig = attrs.field(factory=LossConfig)


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

    # Pretrain stage relies on a flow-matching net_pred_type and dual-timestep input.
    config.model.use_ema = True
    config.model.net.r_timestep = True
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.discriminator_scheduler.warm_up_steps = [0]

    return config
