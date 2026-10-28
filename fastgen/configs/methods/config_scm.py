# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for the sCM model, on CIFAR-10 dataset."""

from typing import Any
import copy

import attrs
from omegaconf import DictConfig

from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
    EMA_CALLBACK,
)
from fastgen.configs.opt import RAdamOptimizerConfig
from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
    SampleTConfig as BaseSampleTConfig,
)
from fastgen.methods import SCMModel


@attrs.define
class SampleTConfig(BaseSampleTConfig):
    """Config for sampling t from a time distribution for SCM."""

    time_dist_type: str = "lognormal"
    train_p_mean: float = -1.0
    train_p_std: float = 1.4  # update from 1.4 with the TrigFlow checkpoint to 1.8 for sCT with the EDM checkpoint

    # sigma in data distribution
    sigma_data: float = 0.5


@attrs.define
class LossConfig:
    """Config for the losses in sCM"""

    # use consistency distillation
    use_cd: bool = False
    # warm-up steps for tangent
    tangent_warmup_steps: int = 10000
    # tangent normalization constant
    tangent_warmup_const: float = 0.1
    # Extra 1 / (sigma_d tan t) loss weight on top of the paper's prior weighting, which is already the cos(t) in g.
    # From github.com/xandergos/sCM-mnist (also in Sana-Sprint), whose author found it unstable and disabled it.
    prior_weighting_enabled: bool = False
    # enable g_norm_spatial_invariance
    g_norm_spatial_invariance: bool = True
    # enable finite difference estimate of JVP
    use_jvp_finite_diff: bool = False
    # step in TrigFlow time of the finite-difference JVP estimate
    jvp_finite_diff_eps: float = 5e-4


@attrs.define
class ModelConfig(BaseModelConfig):
    # The inherited guidance_cfg drives the TrigFlow-wrapped teacher: piecewise ``boundary`` values are in TrigFlow
    # time t_hat = atan2(sigma, alpha * sigma_data), not the wrapped network's time.

    # config for sampling t
    sample_t_cfg: SampleTConfig = attrs.field(factory=SampleTConfig)

    # config for losses
    loss_config: LossConfig = attrs.field(factory=LossConfig)

    # EMA for the main net (requires EMACallback)
    use_ema: Any = True

    # optimizer
    net_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(RAdamOptimizerConfig))

    # Autocast dtype for the JVP (None disables autocast there). The JVP state and its network input are kept in at
    # least this dtype; the conditioning stays in precision_model. Unlike FastGen's own dropout, native dropout layers,
    # such as LoRA's, replay the forward's masks only if this dtype equals precision_amp.
    precision_amp_jvp: str | None = None


@attrs.define
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(SCMModel)(config=None))


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **EMA_CALLBACK,
            **BASE_CALLBACKS,
        }
    )
    config.trainer.callbacks.grad_clip.grad_norm = 1000000
    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False

    config.trainer.callbacks.ema.type = "halflife"

    config.model.net.backbone.dropout = 0.2  # set dropout=0.2 for cifar10, by default
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.weight_decay = 0.0

    config.model.net_scheduler.warm_up_steps = [0]
    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003
    config.model.net.backbone.label_dim = 0

    # dataloader (unconditional CIFAR-10)
    config.dataloader_train.use_labels = False

    return config
