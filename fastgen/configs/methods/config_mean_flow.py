# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import attrs
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional

from fastgen.configs.opt import RAdamOptimizerConfig
from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
    SampleTConfig as BaseSampleTConfig,
)
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
    EMA_CALLBACK,
)
from fastgen.methods import MeanFlowModel


@attrs.define
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


@attrs.define
class LossConfig:
    """Config for the losses in Mean Flow"""

    # use consistency distillation
    use_cd: bool = False
    # enable finite difference estimate of JVP
    use_jvp_finite_diff: bool = False
    # epsilon for finite difference estimate of JVP
    jvp_finite_diff_eps: float = 1e-4
    # adaptive loss normalization as a function of a per-sample scale x (see norm_const)
    # (None disables it; the l2 loss then reduces with a per-element mean)
    norm_method: Optional[str] = "poly_1.0"
    # additive constant c of the adaptive normalization (poly: 1/(x + c)^p, exp: const * exp(scale * x + c)),
    # where x is the summed squared residual under loss_type="l2" (the paper's 1/(||Δ||^2 + c)^p) and the
    # tangent's L2 norm under "opt_grad" (1/(||Δ|| + c)^p), so norm_const is in squared or linear units accordingly
    norm_const: float = 1e-1
    # tangent warmup steps
    tangent_warmup_steps: int = 0
    # make tangent dimension invariant
    tangent_spatial_invariance: bool = False
    # loss type (choice between l2 and opt_grad)
    loss_type: str = "opt_grad"
    # optional fixed per-timestep loss weighting evaluated as a function of t
    # ("beta08", "gaussian") of the noise fraction sigma / (alpha + sigma), which is t
    # on rectified flow, normalized to mean one over a draw of the flow-map t density.
    # Multiplies the adaptive norm_method weight above; None disables it.
    weight_type: Optional[str] = None
    # rebalance the flow-map / consistency (r < t) sample losses to the global
    # flow-matching (r = t) loss mean via a detached per-sample factor
    rebalance_to_flow_matching: bool = False


@attrs.define
class ModelConfig(BaseModelConfig):
    # config for sampling t
    sample_t_cfg: SampleTConfig = attrs.field(factory=SampleTConfig)

    # changes to sample_t_cfg for sampling r, e.g. {"train_p_mean": -4.0} (None samples r like t)
    sample_r_overrides: Optional[Dict[str, Any]] = None

    # config for losses
    loss_config: LossConfig = attrs.field(factory=LossConfig)

    # EMA for the main net (requires EMACallback)
    use_ema: Any = True

    # optimizer
    net_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(RAdamOptimizerConfig))

    # prediction-side guidance fusion scale (AnyFlow guidance distillation):
    # the conditional output is trained to be the guided flow directly via
    # (u_cond + (g-1) * u_uncond) / g against the raw data velocity. None
    # keeps MeanFlow's target-side guidance (guidance_cfg).
    guidance_fuse_scale: Optional[float] = None

    # condition dropout probability
    cond_dropout_prob: Optional[float] = None

    # list of condition keys that do not drop
    cond_keys_no_dropout: List[str] = attrs.field(factory=list)

    # Autocast dtype for the JVP (None disables autocast there). The JVP state and its network input are kept in at
    # least this dtype; the conditioning stays in precision_model. Unlike FastGen's own dropout, native dropout layers,
    # such as LoRA's, replay the forward's masks only if this dtype equals precision_amp.
    precision_amp_jvp: str | None = None


@attrs.define
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(MeanFlowModel)(config=None))


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
    config.trainer.callbacks.grad_clip.grad_norm = 1000000
    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False

    # model
    config.model.student_sample_type = "ode"
    config.model.load_student_weights = False
    config.model.net.backbone.dropout = 0.2
    config.model.net.backbone.r_timestep = True
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.net_optimizer.weight_decay = 0.0
    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003
    config.model.net.backbone.label_dim = 0

    # dataloader (unconditional CIFAR-10)
    config.dataloader_train.use_labels = False

    return config
