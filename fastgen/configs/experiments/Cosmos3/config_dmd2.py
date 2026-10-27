# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DMD2 config for the Cosmos3-Nano video-generation tower.

Uses the generic ``VideoLoaderConfig``: point ``dataloader_train.datatags`` at WebDataset
shards with ``mp4`` and ``txt`` members (81-frame 1280x720 clips at 24 FPS by default).
The pinned ``nvidia/Cosmos3-Nano`` snapshot is loaded through the shared Hugging Face cache.
"""

from copy import deepcopy

import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.discriminator import Discriminator_Cosmos3_Config
from fastgen.configs.net import Cosmos3_Nano_Config
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.cosmos3.network import Cosmos3


def create_config():
    config = config_dmd2_default.create_config()
    config.trainer.fsdp = True
    config.model.meta_init = True

    config.trainer.max_iter = 6000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500
    config.trainer.batch_size_global = 64

    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # Wan2.2 VAE compression is 4x temporal and 16x spatial:
    # 1280x720 @ 81 frames -> [48, 21, 45, 80].
    config.model.input_shape = [48, 21, 45, 80]

    cosmos3_config = deepcopy(Cosmos3_Nano_Config)
    config.model.net = cosmos3_config
    config.model.teacher = deepcopy(cosmos3_config)
    config.model.discriminator = deepcopy(Discriminator_Cosmos3_Config)
    config.model.discriminator.disc_type = "multiscale_down_mlp_medium"
    # Taps at ~50% / ~75% / 100% of the 36 decoder layers.
    config.model.discriminator.feature_indices = [17, 26, 35]

    # Keep text conditioning fixed while distilling the video-generation tower.
    config.model.net.freeze_understanding_tower = True
    config.model.teacher.freeze_understanding_tower = True

    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"
    config.model.student_sample_type = "ode"
    config.model.guidance_cfg = Cosmos3.default_guidance_cfg()

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.student_grid_cfg.steps = 4
    config.model.student_grid_cfg.grid_kwargs = {"shift": 5}

    config.dataloader_train = deepcopy(VideoLoaderConfig)
    config.dataloader_train.batch_size = 1
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames
    config.dataloader_train.target_fps = 24
    config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_cosmos3"}

    config.log_config.group = "cosmos3_dmd2"
    return config
