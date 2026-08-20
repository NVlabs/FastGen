# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
import copy

import attrs
from omegaconf import DictConfig

from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    EMA_CALLBACK,
    GPUStats_CALLBACK,
    TrainProfiler_CALLBACK,
    ParamCount_CALLBACK,
)
from fastgen.configs.opt import RAdamOptimizerConfig
from fastgen.utils import LazyCall as L
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
    SampleTConfig as BaseSampleTConfig,
)
from fastgen.methods import SCMModel

""" Configs for the sCM model, on CIFAR-10 dataset. """


@attrs.define(slots=False)
class SampleTConfig(BaseSampleTConfig):
    """Config for sampling t from a time distribution for SCM."""

    time_dist_type: str = "lognormal"
    train_p_mean: float = -1.0
    train_p_std: float = 1.4  # update from 1.4 with the TrigFlow checkpoint to 1.8 for sCT with the EDM checkpoint

    # sigma in data distribution
    sigma_data: float = 0.5
    # quantize t
    quantize: bool = False


@attrs.define(slots=False)
class LossConfig:
    """Config for the losses in sCM"""

    # use consistency distillation
    use_cd: bool = False
    # warm-up steps for tangent
    tangent_warmup_steps: int = 10000
    # tangent normalization constant
    tangent_warmup_const: float = 0.1
    # enable prior weighting
    prior_weighting_enabled: bool = True
    # enable g_norm_spatial_invariance
    g_norm_spatial_invariance: bool = True
    # enable divide_x_0_spatial_dim
    divide_x_0_spatial_dim: bool = True
    # enable finite difference estimate of JVP
    use_jvp_finite_diff: bool = False
    # episilon t in finite difference estimation of JVP
    jvp_finite_diff_eps: float = 1e-3


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    # config for sampling t
    sample_t_cfg: SampleTConfig = attrs.field(factory=SampleTConfig)

    # config for losses
    loss_config: LossConfig = attrs.field(factory=LossConfig)

    # EMA for the main net (requires EMACallback)
    use_ema: Any = True

    # optimizer
    net_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(RAdamOptimizerConfig))

    # precision for autocast in JVP (none disables autocast in the JVP region)
    precision_amp_jvp: str | None = None


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(SCMModel)(
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
    config.trainer.callbacks.grad_clip.grad_norm = 1000000
    config.trainer.tf32_enabled = False

    config.trainer.callbacks.ema.type = "halflife"

    config.model.net.dropout = 0.2  # set dropout=0.2 for cifar10, by default
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.weight_decay = 0.0

    config.model.net_scheduler.warm_up_steps = [0]
    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003
    config.model.net.label_dim = 0

    # dataloader (unconditional CIFAR-10)
    config.dataloader_train.use_labels = False

    return config
