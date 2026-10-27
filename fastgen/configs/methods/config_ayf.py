# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for Align Your Flow (AYF, EMD) distillation.

AYF distills a flow-map student ``u(x_t, t, r)`` from a base flow-matching model
(the paper calls the second timestep ``s`` and the tangent scale ``r``). The
experiment configs pick the network and the student realization (the FLUX/Qwen
recipe sets ``model.net_adapter.type = "lora"``).
"""

import attrs
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple

from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.config import (
    BaseConfig,
    BaseModelConfig,
)
from fastgen.configs.methods.config_mean_flow import (
    LossConfig as MeanFlowLossConfig,
    SampleTConfig as MeanFlowSampleTConfig,
)
from fastgen.configs.callbacks import BASE_CALLBACKS, GradClip_CALLBACK
from fastgen.methods import AYFModel


@attrs.define
class SampleTConfig(MeanFlowSampleTConfig):
    """Distribution for the first timestep ``t``."""

    time_dist_type: str = "uniform"
    min_t: float = 0.001
    max_t: float = 0.999
    # AYF trains the flow map on r < t everywhere (no flow-matching anchor bucket).
    flow_matching_ratio: float = 0.0


@attrs.define
class LossConfig(MeanFlowLossConfig):
    """AYF (EMD) loss config on top of the MeanFlow one.

    AYF keeps its own EMD weighting (``rms_clamp_min``), so the MeanFlow
    ``norm_method``/``weight_type`` adaptive weights stay disabled.
    """

    # AYF always distills from a frozen teacher (CM.build_model builds it via this switch).
    use_cd: bool = True
    norm_method: Optional[str] = None
    # The paper's ``r``: scales the JVP term in the EMD target.
    tangent_scale: float = 0.95
    # EMD normalization floor: weight = 1 / clamp(RMS(target), min=rms_clamp_min).
    rms_clamp_min: float = 1.0


@attrs.define
class ModelConfig(BaseModelConfig):
    sample_t_cfg: SampleTConfig = attrs.field(factory=SampleTConfig)
    # Changes to sample_t_cfg for the second timestep r (the AYF paper's s); None samples r like t.
    sample_r_overrides: Optional[Dict[str, Any]] = None
    loss_config: LossConfig = attrs.field(factory=LossConfig)

    # Optional per-sample teacher guidance interval, e.g. (1.5, 5.0). Only active
    # when the net exposes a ``guidance_scale`` attribute.
    teacher_guidance_range: Optional[Tuple[float, float]] = None

    # How the sampled guidance drives the teacher: "embedded" feeds it to the
    # teacher's embedded guidance input (guidance-distilled bases, e.g. FLUX.1-dev);
    # "cfg" runs true classifier-free guidance with it as the scale (composed via
    # ``guidance_cfg``), while the student/JVP see it through the guidance_embedder.
    teacher_guidance_type: str = "embedded"

    # Guidance-distillation warmup: for ``iteration <= warmup``, force ``r = t`` and
    # skip the JVP — the EMD loss degenerates to guided-velocity distillation (the
    # warmup stage for non-guidance-distilled teachers).
    guidance_distill_warmup_steps: int = 0

    # Autocast dtype for the JVP (None disables autocast there). The JVP state and its network input are kept in at
    # least this dtype; the conditioning stays in precision_model. Unlike FastGen's own dropout, native dropout layers,
    # such as LoRA's, replay the forward's masks only if this dtype equals precision_amp.
    precision_amp_jvp: Optional[str] = None


@attrs.define
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(AYFModel)(config=None))


def create_config():
    config = Config()

    # Callbacks: no EMA callback (use_ema=False for AYF/LoRA).
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **BASE_CALLBACKS,
        }
    )
    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False

    # The student is a flow-map network with a dual-timestep input.
    config.model.net.backbone.r_timestep = True
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.student_sample_type = "ode"

    return config
