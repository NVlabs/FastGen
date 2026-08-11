# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import attrs
from omegaconf import DictConfig
from typing import Any, List, Optional

from fastgen.configs.opt import RAdamOptimizerConfig
from fastgen.utils import LazyCall as L
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
    SampleTConfig as BaseSampleTConfig,
)
from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    EMA_CALLBACK,
    GPUStats_CALLBACK,
    TrainProfiler_CALLBACK,
    ParamCount_CALLBACK,
)
from fastgen.methods import MeanFlowModel


@attrs.define(slots=False)
class SampleTConfig(BaseSampleTConfig):
    """Config for sampling t from a time distribution for MeanFlow."""

    time_dist_type: str = "lognormal"
    train_p_mean: float = -1.1
    train_p_std: float = 2.0

    # fraction of the batch with r = t (pure flow matching); the rest keeps
    # the randomly sampled r, minus the consistency fraction below
    flow_matching_ratio: float = 0.25

    # fraction of the batch forced to r = 0 (consistency-to-clean, used by
    # AnyFlow; 0.0 keeps the original MeanFlow behavior)
    consistency_ratio: float = 0.0

    # how samples are assigned to the two buckets above: False draws each
    # sample's bucket independently, True partitions the global batch by rank
    # index so the bucket sizes are exact every iteration (used by AnyFlow)
    deterministic_buckets: bool = False


@attrs.define(slots=False)
class SampleRConfig(BaseSampleTConfig):
    """Config for sampling r from a time distribution for MeanFlow."""

    # whether to use a different distribution for r than t
    enabled: bool = False

    time_dist_type: str = "lognormal"
    train_p_mean: float = -1.1
    train_p_std: float = 2.0


@attrs.define(slots=False)
class LossConfig:
    """Config for the losses in Mean Flow"""

    # use consistency distillation
    use_cd: bool = False
    # use the squared l2 loss or the l2 loss
    use_squared_l2: bool = False
    # enable finite difference estimate of JVP
    use_jvp_finite_diff: bool = False
    # epsilon for finite difference estimate of JVP
    jvp_finite_diff_eps: float = 1e-4
    # adaptive loss normalization as a function of the per-sample loss
    # (None disables it; the l2 loss then reduces with a per-element mean)
    norm_method: Optional[str] = "poly_1.0"
    # tangent warmup constant
    norm_const: float = 1e-1
    # tangent warmup steps
    tangent_warmup_steps: int = 0
    # make tangent dimension invariant
    tangent_spatial_invariance: bool = False
    # loss type (choice between l2 and opt_grad)
    loss_type: str = "opt_grad"
    # optional fixed per-timestep loss weighting evaluated as a function of t
    # ("beta08", "gaussian", "uniform"). Multiplies the adaptive norm_method 
    # weight above; None disables it.
    weight_type: Optional[str] = None
    # prediction-side guidance fusion scale (AnyFlow guidance distillation):
    # the conditional output is trained to be the guided flow directly via
    # (u_cond + (g-1) * u_uncond) / g against the raw data velocity. None 
    # keeps MeanFlow's target-side guidance (guidance_scale).
    guidance_fuse_scale: Optional[float] = None
    # rebalance the flow-map / consistency (r < t) sample losses to the global
    # flow-matching (r = t) loss mean via a detached per-sample factor
    rebalance_to_flow_matching: bool = False


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    # config for sampling t
    sample_t_cfg: SampleTConfig = attrs.field(factory=SampleTConfig)

    # config for sampling r (if None uses sample_t_cfg)
    sample_r_cfg: SampleRConfig = attrs.field(factory=SampleRConfig)

    # config for losses
    loss_config: LossConfig = attrs.field(factory=LossConfig)

    # EMA for the main net (requires EMACallback)
    use_ema: Any = True

    # guidance mixture ratio for cfg in teacher diffusion model, when doing consistency distillation. None means no mixture guidance.
    guidance_mixture_ratio: Optional[float] = None

    # optimizer
    net_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(RAdamOptimizerConfig))

    # condition dropout probability
    cond_dropout_prob: Optional[float] = None

    # list of condition keys that do not drop
    cond_keys_no_dropout: List[str] = attrs.field(factory=list)

    # guidance t start
    guidance_t_start: float = 0.0

    # guidance t end
    guidance_t_end: float = 1.0

    # precision for autocast in JVP (none defaults to training precision)
    precision_amp_jvp: str | None = None


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(MeanFlowModel)(
        config=None,
    )


def create_config():
    config = Config()

    # trainer
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **EMA_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **WANDB_CALLBACK,
        }
    )  # CM training must have the ct_schedule callback
    # recommended setting for CIFAR-10 is max_iter * batch_size // (8 * 1000)
    config.trainer.callbacks.grad_clip.grad_norm = 1000000
    config.trainer.tf32_enabled = False

    # model
    config.model.student_sample_type = "ode"
    config.model.load_student_weights = False
    config.model.net.dropout = 0.2
    config.model.net.r_timestep = True
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.net_optimizer.weight_decay = 0.0
    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003
    config.model.net.label_dim = 0

    # dataloader (unconditional CIFAR-10)
    config.dataloader_train.use_labels = False

    return config
