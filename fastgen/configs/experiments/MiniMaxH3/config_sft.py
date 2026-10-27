# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT config for MiniMax-H3 (joint video + audio).

MiniMax-H3 packs one request into one sequence and the text length sets the origin of the
media rotary clock, so there is no padding to batch different prompts with: one sample per
GPU, scaled through ``grad_accum_rounds``.
"""

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoAudioLoaderConfig
from fastgen.configs.net import MiniMaxH3_Config
from fastgen.networks.MiniMaxH3.network import MINIMAX_H3_FLOW_SHIFT, MiniMaxH3
from fastgen.configs.config_utils import model_pixel_shape


def create_config():
    config = config_sft_default.create_config()

    # ---- precision / FSDP ----
    config.trainer.fsdp = True
    config.trainer.fsdp_cpu_offload = True
    config.model.meta_init = True
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    config.model.net = copy.deepcopy(MiniMaxH3_Config)

    # MiniMax-H3 is guidance-distilled: guidance is baked into the weights, so there is no
    # negative prompt and no CFG, and sampling runs one forward per step.
    config.model.guidance_cfg = MiniMaxH3.default_guidance_cfg()
    config.model.student_grid_cfg.steps = 49  # the release's num_inference_steps=50 counts grid points

    # ---- input shape (video + audio; audio is inferred from the video shape) ----
    # Video latent ``[C, F, H, W]``; pixels = (H*16) x (W*16), frames = 17*n + 5 for
    # F = 5*n + 2. Audio is ``[32, 2, L]`` with L on the shared 40 Hz clock.
    # Resolution presets (video latent only; audio is derived). Full length is the largest
    # ``17*n + 5`` within MiniMax-H3's 15 s ceiling: 345 frames (14.375 s).
    #   [24,  37, 34, 60]  ->  544 x 960,  124 frames ( 5.17 s)
    #   [24,  37, 48, 84]  ->  768 x 1344, 124 frames ( 5.17 s,  native res)
    #   [24, 102, 34, 60]  ->  544 x 960,  345 frames (14.375 s, full length)
    #   [24, 102, 48, 84]  ->  768 x 1344, 345 frames (14.375 s, native res, full length)
    config.model.input_shape = {"video": [24, 37, 34, 60], "audio": None}

    # ---- optimizer ----
    config.model.net_optimizer.lr = 2e-5
    config.trainer.callbacks.grad_clip.grad_norm = 1

    # Video and audio noise levels follow the released sampler.
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = MINIMAX_H3_FLOW_SHIFT
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # ---- dataloader (placeholder audio+video loader) ----
    # One request per sequence, so batch 1/GPU (the loader's own default); raise the global batch
    # with grad_accum.
    config.dataloader_train = copy.deepcopy(VideoAudioLoaderConfig)
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames
    config.dataloader_train.target_sample_rate = 32000
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps
    # Guidance-distilled: no negative prompt, so don't encode one.
    config.dataloader_train.presets_map = {}

    # ---- trainer schedule ----
    config.trainer.max_iter = 2000
    config.trainer.logging_iter = 10
    config.trainer.save_ckpt_iter = 500
    config.trainer.callbacks.wandb.sample_logging_iter = 100

    config.log_config.group = "minimax_h3_sft"
    return config
