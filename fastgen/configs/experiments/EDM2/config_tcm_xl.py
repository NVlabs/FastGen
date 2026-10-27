# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.experiments.EDM2.config_tcm_s as config_tcm_default
from fastgen.configs.net import EDM2_IN64_XL_Config, CKPT_ROOT_DIR

""" Configs for the TCM model, on EDM2 and ImageNet-64 dataset. """


def create_config():
    config = config_tcm_default.create_config()
    # Reference native FP16 training settings.
    config.model.precision_model = "float16"
    config.model.precision_opt = "float32"
    config.model.grad_scaler_enabled = True
    config.model.grad_scaler_init_scale = 16
    config.model.grad_scaler_growth_interval = 20000
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm2-img64-xl-fid.pth"
    config.model.net = copy.deepcopy(EDM2_IN64_XL_Config)
    config.model.net_optimizer.lr = 1e-4
    config.model.net.dropout = 0.5
    config.model.net.dropout_resolutions = [16, 8]
    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003
    return config
