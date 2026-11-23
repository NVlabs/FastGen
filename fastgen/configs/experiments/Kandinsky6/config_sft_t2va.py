# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kandinsky 6 Pro T2VA SFT at the native 5-second canvas: 121 frames at 24 fps, 480x864, 44.1 kHz audio.

Uses the generic ``VideoAudioLoaderConfig``: point ``dataloader_train.datatags`` at WebDataset shards of
captioned clips with audio. Video and audio share one flow time and shift; the native precision contract
is in the network module.
"""

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.configs.data import VideoAudioLoaderConfig
from fastgen.configs.net import Kandinsky6Config
from fastgen.networks.Kandinsky6.network import Kandinsky6


def create_config():
    config = config_sft_default.create_config()
    config.trainer.fsdp = True
    config.model.meta_init = True
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"
    config.model.extra_fsdp_modules += ["text_encoder", "vae"]
    # Match the native factory's FP32 matrix products and validated codec execution.
    config.trainer.tf32_matmul = True
    config.trainer.cudnn.deterministic = True
    config.trainer.cudnn.benchmark = False

    config.model.net = copy.deepcopy(Kandinsky6Config)
    config.model.input_shape = {"video": [16, 31, 60, 108], "audio": None}
    config.model.guidance_cfg = Kandinsky6.default_guidance_cfg()
    config.model.net_optimizer.lr = 5e-6
    config.model.net_optimizer.fused = True

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.student_sample_type = "ode"
    config.model.student_grid_cfg.steps = 50
    config.model.student_grid_cfg.grid_kwargs = {"shift": 5.0}

    config.dataloader_train = copy.deepcopy(VideoAudioLoaderConfig)
    config.dataloader_train.batch_size = 1
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.sequence_length = num_frames
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.target_fps = config.model.net.fps
    config.dataloader_train.target_sample_rate = Kandinsky6.audio_sampling_rate
    config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_kandinsky6"}

    config.trainer.batch_size_global = 16
    config.trainer.max_iter = 1000
    config.trainer.logging_iter = 20
    config.trainer.save_ckpt_iter = 100
    config.trainer.callbacks.wandb.max_ranks = 8
    config.log_config.group = "kandinsky6_t2va_sft"
    return config
