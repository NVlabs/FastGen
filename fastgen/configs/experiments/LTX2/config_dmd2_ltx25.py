# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 DMD2 with backward simulation and the canonical discriminator.

This keeps the LTX-2.3 DMD2 recipe's four-step schedule, precision, and
1280x704 / 241-frame shape while switching to the full LTX-2.5 transformer.
Backward simulation supplies the student's trajectory; the real audio/video
loader, GAN loss, and discriminator callback are inherited unchanged.
"""

import copy

import fastgen.configs.experiments.LTX2.config_dmd2_ltx23 as base_config
from fastgen.callbacks.carry import CarryCallback
from fastgen.configs.net import LTX2_5_Config
from fastgen.networks.LTX2.network import LTX2
from fastgen.configs.instantiate import LazyCall as L


def create_config():
    # Intentionally retain LTX-2.3's [128, 31, 22, 40] latent shape and
    # four-step, shift-5 grid.
    config = base_config.create_config()

    config.model.net = copy.deepcopy(LTX2_5_Config)
    config.model.guidance_cfg = LTX2.default_guidance_cfg(version="2.5")

    config.model.backward_simulation = "staggered"
    config.trainer.callbacks.carry = L(CarryCallback)()

    # For data-free training (without GAN loss), uncomment the following
    # from fastgen.configs.data import PromptLoaderConfig
    # config.model.gan_loss_weight_gen = 0.0
    # config.trainer.callbacks.pop("grad_clip_discriminator", None)
    # config.dataloader_train = copy.deepcopy(PromptLoaderConfig)
    # config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_ltx2"}

    config.trainer.logging_iter = 50
    config.trainer.callbacks.wandb.max_ranks = 8
    config.log_config.group = "ltx2_5_dmd2_multimodal_backward_sim"
    return config
