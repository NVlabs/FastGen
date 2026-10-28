# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import attrs
from omegaconf import DictConfig
from typing import Any

from fastgen.configs.opt import RAdamOptimizerConfig
from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
    SampleTConfig as BaseSampleTConfig,
)
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    CTSchedule_CALLBACK,
    GradClip_CALLBACK,
    EMA_CALLBACK,
)
from fastgen.methods import CMModel


@attrs.define
class SampleTConfig(BaseSampleTConfig):
    """Config for sampling t from a time distribution for CM."""

    time_dist_type: str = "lognormal"

    # lowest value for end point, raised to the network's min_t if below it; defaults to EDM's min_t
    # (min_r > 0 requires the network to satisfy net(x_t, t) ~ x_t for small t).
    min_r: float = 0.002


@attrs.define
class LossConfig:
    """Config for the losses in CM"""

    # use consistency distillation
    use_cd: bool = False
    # the constant value in the pseudo-huber loss
    huber_const: float = 1e-8
    # use the squared l2 loss or the l2 loss
    use_squared_l2: bool = False
    # weighting of the CT loss, choices: ['default', 'c_out', 'c_out_sq', 'sigma_sq', 'sqrt', 'one']
    weighting_ct_loss: str = "default"


@attrs.define
class ModelConfig(BaseModelConfig):
    # optimizer
    net_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(RAdamOptimizerConfig))

    # config for sampling t
    sample_t_cfg: SampleTConfig = attrs.field(factory=SampleTConfig)

    # EMA for the main net (requires EMACallback)
    use_ema: Any = True

    # config for losses
    loss_config: LossConfig = attrs.field(factory=LossConfig)

    # prediction type of the teacher's DDIM steps for consistency distillation (None means its net_pred_type)
    teacher_pred_type: str | None = None


@attrs.define
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(CMModel)(config=None))


def create_config():
    config = Config()

    # trainer
    config.trainer.callbacks = DictConfig(
        {
            **CTSchedule_CALLBACK,
            **GradClip_CALLBACK,
            **EMA_CALLBACK,
            **BASE_CALLBACKS,
        }
    )  # CM training must have the ct_schedule callback
    # 9 stages of 6.4M images (q=2), then the ratio stays at ratio_limit
    config.trainer.callbacks.ct_schedule.kimg_per_stage = 6400
    config.trainer.callbacks.grad_clip.grad_norm = 1000000
    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False

    # model
    config.model.net.backbone.dropout = 0.2
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.net_optimizer.weight_decay = 0.0
    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003
    config.model.net.backbone.label_dim = 0

    # dataloader (unconditional CIFAR-10)
    config.dataloader_train.use_labels = False

    return config
