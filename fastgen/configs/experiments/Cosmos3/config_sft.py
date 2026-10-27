# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT config for the Cosmos3-Nano video-generation tower.

Uses the generic ``VideoLoaderConfig``: point ``dataloader_train.datatags`` at WebDataset
shards with ``mp4`` and ``txt`` members (81-frame 1280x720 clips at 24 FPS by default).
The pinned ``nvidia/Cosmos3-Nano`` snapshot is loaded through the shared Hugging Face cache.
"""

from copy import deepcopy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Cosmos3_Nano_Config
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.cosmos3.network import Cosmos3


def create_config():
    config = config_sft_default.create_config()
    config.trainer.fsdp = True
    config.model.meta_init = True

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.model.net_optimizer.lr = 1e-5
    config.model.precision = "float32"
    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"

    # Wan2.2 VAE compression is 4x temporal and 16x spatial:
    # 1280x720 @ 81 frames -> [48, 21, 45, 80].
    config.model.input_shape = [48, 21, 45, 80]
    config.model.net = deepcopy(Cosmos3_Nano_Config)
    config.model.guidance_cfg = Cosmos3.default_guidance_cfg()
    config.model.student_grid_cfg.steps = 35

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # CFG dropout swaps in the explicit negative-prompt encoding (Cosmos3 has no null
    # condition): ``Cosmos3.mix_condition`` replaces the prompt token IDs with the
    # negative's per sample, keeping the negative branch used by CFG at inference
    # in-distribution during fine-tuning.
    config.model.cond_dropout_prob = 0.1

    config.dataloader_train = deepcopy(VideoLoaderConfig)
    config.dataloader_train.batch_size = 1
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames
    config.dataloader_train.target_fps = 24
    config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_cosmos3"}

    config.log_config.group = "cosmos3_sft"
    return config
