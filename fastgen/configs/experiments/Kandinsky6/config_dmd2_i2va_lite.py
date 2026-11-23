# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kandinsky 6 Lite I2VA DMD2: the Pro recipe on the public 3B checkpoint, its GAN heads sized for Lite."""

import copy

import fastgen.configs.experiments.Kandinsky6.config_dmd2_i2va as config_pro
from fastgen.configs.discriminator import Discriminator_Kandinsky6Lite_Multimodal_Config
from fastgen.configs.net import Kandinsky6LiteConfig


def create_config():
    config = config_pro.create_config()
    config.model.net = copy.deepcopy(Kandinsky6LiteConfig)
    config.model.net.image_conditioned = True

    discriminator = copy.deepcopy(Discriminator_Kandinsky6Lite_Multimodal_Config)
    discriminator.modality_discs.video.disc_type = config.model.discriminator.modality_discs.video.disc_type
    # Halfway, three quarters and the last of the 32 fused blocks, like Pro's 30/45/59 of 60.
    discriminator.feature_indices = [16, 24, 31]
    for modality_disc in discriminator.modality_discs.values():
        modality_disc.feature_indices = list(discriminator.feature_indices)
    config.model.discriminator = discriminator

    config.log_config.group = "kandinsky6_lite_i2va_dmd2"
    return config
