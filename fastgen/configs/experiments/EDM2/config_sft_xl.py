# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy

import fastgen.configs.experiments.EDM2.config_sft_s as config_sft_edm2_s
from fastgen.configs.net import EDM2_IN64_XL_Config, CKPT_ROOT_DIR

"""Configs for SFT (Supervised Fine-Tuning) on EDM2-XL and ImageNet-64 dataset."""


def create_config():
    config = config_sft_edm2_s.create_config()

    # Override network config for XL model
    config.model.net = copy.deepcopy(EDM2_IN64_XL_Config)

    # Pretrained model path
    # EDM2-XL ImageNet64 checkpoint from https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm2-img64-xl-fid.pth"

    # Adjust batch size for larger model
    config.dataloader_train.batch_size = 32
    config.trainer.batch_size_global = 512

    config.log_config.group = "edm2_xl_imagenet64_sft"

    return config
