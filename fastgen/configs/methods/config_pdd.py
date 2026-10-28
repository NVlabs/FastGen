# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import attrs
from omegaconf import DictConfig

from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
)
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
    EMA_CALLBACK,
)
from fastgen.methods import PDDModel


@attrs.define
class LossConfig:
    """Config for the losses in PDD"""

    # type of Runge-Kutta method to approximate the teacher mean velocity
    type_rk: str = "euler"
    # minimum block size for sampling initial states
    block_size_min: int = 32
    # maximum block size for evaluating the loss
    block_size_max: int = 128
    # number of heads supervised per sample within the block, drawn without
    # replacement; must not exceed the smallest sampled block
    num_supervised_heads: int = 1


@attrs.define
class ModelConfig(BaseModelConfig):
    # Fused PDD intervals are deterministic ODE transitions.
    student_sample_type: str = "ode"
    # Student input source, all on the student grid at the largest NFE PDD trains for,
    # ceil(pdd_steps / block_size_min): None noises real data to a uniformly drawn node;
    # "staggered"/"synced" carry one trajectory across iterations with CarryCallback
    # (rank-offset vs. common phases); "default" carries nothing and starts each iteration
    # at a random step.
    backward_simulation: str | None = None
    # Precision for PDD trajectory integration, which AMP does not lower. None inherits method-state precision.
    precision_decoding: str | None = None
    # config for losses
    loss_config: LossConfig = attrs.field(factory=LossConfig)


@attrs.define
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(PDDModel)(config=None))


def create_config():
    config = Config()

    # trainer
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **EMA_CALLBACK,
            **BASE_CALLBACKS,
        }
    )
    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False

    return config
