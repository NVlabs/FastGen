# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Method config for causal SFT with teacher forcing (clean-context conditioning)."""

from typing import Optional

import attrs
from omegaconf import DictConfig

from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
)
from fastgen.configs.config import SampleTConfig as BaseSampleTConfig
from fastgen.configs.methods.config_sft_causal import (
    Config as CausalSFTConfig,
    ModelConfig as CausalSFTModelConfig,
)
from fastgen.methods import CausalTeacherForcingSFTModel


@attrs.define
class SampleTContextConfig(BaseSampleTConfig):
    """Config for sampling the timesteps used to noise-augment the clean context stream."""

    # whether to noise-augment the clean context frames (otherwise they stay clean at t=0)
    enabled: bool = False

    time_dist_type: str = "uniform"
    # None defers to the noise scheduler's own t range
    min_t: Optional[float] = None
    max_t: Optional[float] = None


@attrs.define
class ModelConfig(CausalSFTModelConfig):
    # condition the context stream on the same noisy frames the denoising stream is trained on
    teacher_forcing_context_match_train_t: bool = False
    # config for sampling the clean-context timesteps (if not enabled, the context stays clean)
    sample_t_context_cfg: SampleTContextConfig = attrs.field(factory=SampleTContextConfig)


@attrs.define
class Config(CausalSFTConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(
        factory=lambda: L(CausalTeacherForcingSFTModel)(
            config=None,
        )
    )


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **BASE_CALLBACKS,
        }
    )
    config.dataloader_train.batch_size = 256
    config.model.student_grid_cfg.steps = 4
    config.model.net_scheduler.warm_up_steps = [0]
    return config
