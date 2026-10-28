# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for SFT on Cosmos-Predict2.5-2B model."""

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import CosmosPredict2_2B_Config, CKPT_ROOT_DIR
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.cosmos_predict2.network import CosmosPredict2


def create_config():
    config = config_sft_default.create_config()
    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.model.net_optimizer.lr = 5e-6

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.precision = "float32"
    config.model.precision_model = "bfloat16"

    # Latent shape: [C, T_latent, H_latent, W_latent]
    # Cosmos VAE uses 4x8x8 compression (time, height, width)
    # config.model.input_shape = [16, 24, 22, 40]  # cthw - 256p (320x176 video @ 93 frames)
    # config.model.input_shape = [16, 24, 60, 104]  # cthw - 480p (832x480 video @ 93 frames)
    config.model.input_shape = [16, 24, 88, 160]  # cthw - full 720p (1280x704 video @ 93 frames)

    config.model.net = copy.deepcopy(CosmosPredict2_2B_Config)
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cosmos_predict2/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
    # Official post-trained 2B/14B inference uses guidance 7, applied as cond + 7 * (cond - uncond), which is FastGen
    # scale 8; our default of 5 is more conservative
    config.model.guidance_cfg = CosmosPredict2.default_guidance_cfg()
    config.model.student_grid_cfg.steps = 35

    config.dataloader_train = copy.deepcopy(VideoLoaderConfig)
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps
    config.dataloader_train.batch_size = 1
    # Cosmos's own negative prompt, as used at inference.
    config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_cosmos"}

    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames

    config.log_config.group = "cosmos_predict2_sft"

    return config
