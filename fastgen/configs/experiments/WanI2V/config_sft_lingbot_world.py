# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import CameraVideoLoaderConfig
from fastgen.configs.net import LingBot_World_Config
from fastgen.networks.WanI2V.wan_moe import LingBotWorld
from fastgen.configs.callbacks import VIPE_CALLBACK
from fastgen.configs.config_utils import model_pixel_shape

"""SFT config for LingBot-world on a camera-annotated video dataset at ~10s clips."""


def create_config():
    config = config_sft_default.create_config()
    config.trainer.fsdp = True
    config.model.meta_init = True
    config.trainer.fsdp_cpu_offload = True

    # Camera diagnostics: attach the VipeLogger as a W&B log module so each
    # logged step ships the GT camera-trajectory plot + WASD/IJKL action
    # overlays (derived from the GT camera poses) on the saved clips. Set
    # ``callbacks.wandb.log_modules["vipe"].run_vipe_on_generated = True`` to
    # also report scale-aligned pose/intrinsics error (needs the vipe package).
    config.trainer.callbacks.update(VIPE_CALLBACK)

    config.trainer.logging_iter = 100
    config.model.net_optimizer.lr = 1e-6
    config.model.student_grid_cfg.steps = 70

    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"
    config.model.precision_fsdp_reduce = "bfloat16"

    # VAE compress ratio for WAN 14B: (1+T/4) * H / 8 * W / 8
    # 480p (832x480) resolution, 10s clips (sequence_length=161): 41 x 60 x 104 latent
    config.model.input_shape = [16, 41, 60, 104]  # cthw

    config.model.net = copy.deepcopy(LingBot_World_Config)
    config.model.guidance_cfg = LingBotWorld.default_guidance_cfg()

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Camera-annotated video shards at 480p, 10s clips.
    num_frames, height, width = model_pixel_shape(config.model)
    img_size = (width, height)
    sequence_length = num_frames  # 161

    config.dataloader_train = copy.deepcopy(CameraVideoLoaderConfig)
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps
    config.dataloader_train.img_size = img_size
    config.dataloader_train.sequence_length = sequence_length
    config.dataloader_train.batch_size = 1
    # LingBot-World's own negative prompt, as used at inference.
    config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_lingbot_world"}

    config.log_config.group = "wancami2v_lingbot_sft"
    return config
