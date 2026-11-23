# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kandinsky 6 Pro I2VA SFT: the T2VA recipe conditioned on each clip's first frame."""

import fastgen.configs.experiments.Kandinsky6.config_sft_t2va as config_t2va


def create_config():
    config = config_t2va.create_config()
    config.model.net.image_conditioned = True
    config.log_config.group = "kandinsky6_i2va_sft"
    return config
