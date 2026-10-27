# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DMD2 config for LTX-2 (video + audio)."""

import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import VideoAudioLoaderConfig
import copy

from fastgen.configs.discriminator import Discriminator_LTX2_Multimodal_Config
from fastgen.configs.net import LTX2Config
from fastgen.networks.LTX2.network import LTX2
from fastgen.configs.config_utils import model_pixel_shape


def create_config():
    config = config_dmd2_default.create_config()

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    # Enable cpu offloading for higher resolution training
    config.trainer.fsdp_cpu_offload = True

    # Optimizer settings
    config.model.net_optimizer.lr = 5e-6
    config.model.discriminator_optimizer.lr = 5e-6
    config.model.fake_score_optimizer.lr = 5e-6

    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"
    config.trainer.fsdp = True
    config.model.meta_init = True

    config.model.net = copy.deepcopy(LTX2Config)
    config.model.guidance_cfg = LTX2.default_guidance_cfg(version="2.0")

    # 720p resolution (1280x704)
    config.model.input_shape = {"video": [128, 31, 22, 40], "audio": None}
    # 1080p resolution (1920x1088)
    # config.model.input_shape = {"video": [128, 31, 34, 60], "audio": None}

    # Enable VAE tiling for high resolutions to avoid 32-bit index overflow.
    num_frames, height, width = model_pixel_shape(config.model)
    if height > 720:
        config.model.net.enable_vae_tiling = True

    config.model.discriminator = copy.deepcopy(Discriminator_LTX2_Multimodal_Config)
    config.model.discriminator.modality_discs.video.disc_type = "multiscale_down_mlp_medium"
    config.model.discriminator.feature_indices = [24, 36, 47]
    # Every modality head taps the same layers; stating them per modality avoids the override warning.
    for modality_disc in config.model.discriminator.modality_discs.values():
        modality_disc.feature_indices = list(config.model.discriminator.feature_indices)

    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"
    config.model.student_sample_type = "ode"
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.student_grid_cfg.steps = 4
    config.model.student_grid_cfg.grid_kwargs = {"shift": 5}

    config.dataloader_train = copy.deepcopy(VideoAudioLoaderConfig)
    config.dataloader_train.batch_size = 1
    # LTX-2's own negative prompt, as used at inference.
    config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_ltx2"}
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps

    config.log_config.group = "ltx2_dmd2"

    return config
