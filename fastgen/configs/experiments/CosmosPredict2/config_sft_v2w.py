# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for SFT on Cosmos-Predict2.5-2B video2world model."""

import fastgen.configs.experiments.CosmosPredict2.config_sft as config_sft_base


def create_config():
    config = config_sft_base.create_config()

    # Network for v2w
    config.model.net.is_video2world = True
    # Official post-trained Video2World inference uses guidance 7, applied as cond + 7 * (cond - uncond), which is
    # FastGen scale 8; our default of 5 is more conservative
    # config.model.guidance_cfg.cfg.scale = 8.0

    config.log_config.group = "cosmos_predict2_sft_v2w"

    return config
