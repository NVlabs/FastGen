# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Method config for LADD, the GAN-only DMD2 specialization; experiments set the network and discriminator."""

import copy

import attrs
from omegaconf import DictConfig

from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.config import BaseConfig, BaseModelConfig
from fastgen.configs.opt import BaseOptimizerConfig, BaseSchedulerConfig
from fastgen.methods import LADDModel
from fastgen.configs.discriminator import Discriminator_EDM_CIFAR10_Config
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
)


@attrs.define
class ModelConfig(BaseModelConfig):
    """Configuration for Latent Adversarial Diffusion Distillation (LADD)."""

    discriminator: DictConfig = attrs.field(factory=lambda: copy.deepcopy(Discriminator_EDM_CIFAR10_Config))
    # optimizer and scheduler for the discriminator
    discriminator_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseOptimizerConfig))
    discriminator_scheduler: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseSchedulerConfig))

    # student update frequency
    student_update_freq: int = 5
    gan_loss_weight_gen: float = 1.0

    # use the same t and noise to perturb the real data and fake data
    gan_use_same_t_noise: bool = False

    # Student input source (modes as in DMD2).
    backward_simulation: str | None = None

    # What the student generates in training, "x0" or its sampler step, "step" (as in DMD2).
    student_generation: str = "x0"

    # Draw noising times within each sample's student-grid bucket [t_{i+1}, t_i] (as in DMD2).
    sample_t_in_student_bucket: bool = False

    # APT-style R1 regularization weight (arXiv:2501.08316); 0 means no R1 reg.
    # Unlike APT, which perturbs the clean latent, the penalty here is computed on the noised
    # sample x_t and only the discriminator head is optimized (whereas the teacher is frozen).
    # APT states its sigma as a variance whereas gan_r1_reg_alpha is a standard
    # deviation, so matching the strength gan_r1_reg_weight * gan_r1_reg_alpha**2 (see below) gives
    #     gan_r1_reg_weight = lambda_apt * sigma_apt / gan_r1_reg_alpha**2
    # where APT uses lambda_apt=100 with sigma_apt=0.01 for images and sigma_apt=0.1 for videos.
    gan_r1_reg_weight: float = 0.0
    # Perturbation scale (standard deviation) for approximate R1, applied to the noised network
    # input x_t. The loss is unnormalized, so the effective strength scales as
    # gan_r1_reg_weight * gan_r1_reg_alpha**2: scaling alpha by c needs the weight scaled by 1/c**2.
    gan_r1_reg_alpha: float = 0.1


@attrs.define
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(LADDModel)(config=None))


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **BASE_CALLBACKS,
        }
    )

    config.dataloader_train.batch_size = 256
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]

    return config
