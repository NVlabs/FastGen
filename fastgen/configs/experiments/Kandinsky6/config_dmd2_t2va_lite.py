# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kandinsky 6 Lite T2VA DMD2: the GAN-free Pro recipe on the public 3B checkpoint."""

import copy

import fastgen.configs.experiments.Kandinsky6.config_dmd2_t2va as config_pro
from fastgen.configs.net import Kandinsky6LiteConfig


def create_config():
    config = config_pro.create_config()
    config.model.net = copy.deepcopy(Kandinsky6LiteConfig)
    config.log_config.group = "kandinsky6_lite_t2va_dmd2"
    return config
