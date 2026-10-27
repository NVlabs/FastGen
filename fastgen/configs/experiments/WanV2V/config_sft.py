# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import VACE_Wan_1_3B_Config
from fastgen.networks.VaceWan.network import VACEWan
from fastgen.configs.config_utils import model_pixel_shape

""" Configs for SFT on WAN V2V model. """


def create_config():
    config = config_sft_default.create_config()

    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 100

    config.model.net_optimizer.lr = 1e-5
    config.model.student_grid_cfg.steps = 50

    config.model.precision_model = "bfloat16"

    # VAE compress ratio for WAN: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]  # cthw
    config.model.net = copy.deepcopy(VACE_Wan_1_3B_Config)
    config.model.guidance_cfg = VACEWan.default_guidance_cfg()

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Raw-video loader; VACE derives the depth condition from the video via the
    # configured depth model (depth_model_path), so no precomputed depth latent is needed.
    config.dataloader_train = copy.deepcopy(VideoLoaderConfig)
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps
    # 480p (832x480) resolution
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames

    config.dataloader_train.batch_size = 1

    config.log_config.group = "vacewan_sft"
    return config
