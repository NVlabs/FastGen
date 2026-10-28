# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DMD2 config for LTX-2.3 Image-to-Video (video + audio)."""

import copy

import fastgen.configs.experiments.LTX2.config_dmd2_ltx23 as config_t2v
from fastgen.configs.net import LTX2I2V_3_Config
from fastgen.configs.config_utils import model_pixel_shape


def create_config():
    config = config_t2v.create_config()

    config.model.net = copy.deepcopy(LTX2I2V_3_Config)
    # Re-apply conditional tiling after swapping the net config.
    _, height, _ = model_pixel_shape(config.model)
    if height > 720:
        config.model.net.enable_vae_tiling = True

    config.log_config.group = "ltx2_3_i2v_dmd2"

    return config
