# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config schema for the AnyFlow on-policy method (paper Stage 3).

AnyFlow's on-policy stage is DMD2 with a flow-map student that generates via
a multi-step rollout-with-gradient, so the config is the DMD2 model config
unchanged. The flow-map pretrain stage (paper Stage 2) is MeanFlow with
AnyFlow's hyperparameters — see ``configs/experiments/WanT2V/config_anyflow.py``.
"""

from typing import List, Optional

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
from fastgen.configs.methods.config_mean_flow import (
    LossConfig as MeanFlowLossConfig,
    SampleRConfig,
    SampleTConfig as MeanFlowSampleTConfig,
)
from fastgen.methods import AnyFlowModel
from fastgen.utils import LazyCall as L


@attrs.define(slots=False)
class ModelConfig(DMD2ModelConfig):
    """AnyFlow on-policy model config — DMD2 plus the rollout / cotrain knobs.

    The MeanFlow loss / sampling configs drive the co-trained Stage-2
    flow-map loss inside the student update (the reference's
    ``cotrain_forward_kl``).
    """

    # MeanFlow-style (t, r) sampling for the co-trained flow-map loss; the
    # extra fields are ignored by the DMD2 noising-time sampling.
    sample_t_cfg: MeanFlowSampleTConfig = attrs.field(factory=MeanFlowSampleTConfig)
    sample_r_cfg: SampleRConfig = attrs.field(factory=SampleRConfig)
    loss_config: MeanFlowLossConfig = attrs.field(factory=MeanFlowLossConfig)

    # Weight of the co-trained Stage-2 flow-map loss in the student update.
    # The reference runs it at weight 1 (cotrain_forward_kl: True); 0 disables.
    cotrain_pretrain_weight: float = 1.0

    # Rollout NFE list, sampled uniformly per iteration with rank-0 broadcast
    # (reference rollout_cfg.num_inference_steps_list). None falls back to
    # the fixed student_sample_steps.
    student_sample_steps_list: Optional[List[int]] = None

    # Text dropout for the co-trained flow-map loss (reference drop_text_ratio).
    cond_dropout_prob: Optional[float] = None
    cond_keys_no_dropout: List[str] = []

    # Precision for autocast in the co-trained loss JVP (None = training precision).
    precision_amp_jvp: str | None = None


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
