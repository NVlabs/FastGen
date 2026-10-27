# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DMD2 config for Cosmos-Predict2.5-2B video2world model."""

import fastgen.configs.experiments.CosmosPredict2.config_dmd2 as config_dmd2_base


def create_config():
    config = config_dmd2_base.create_config()

    config.model.net.is_video2world = True
    config.model.teacher.is_video2world = True

    config.log_config.group = "cosmos_predict2_dmd2_v2w"

    return config
