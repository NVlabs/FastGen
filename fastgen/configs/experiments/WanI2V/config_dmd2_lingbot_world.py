# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.discriminator import Discriminator_Wan_14B_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import CameraVideoLoaderConfig
from fastgen.configs.net import LingBot_World_Config
from fastgen.networks.WanI2V.wan_moe import LingBotWorld
from fastgen.configs.callbacks import VIPE_CALLBACK
from fastgen.configs.config_utils import model_pixel_shape

"""DMD2 config for LingBot-world on a camera-annotated video dataset at ~10s clips."""


def create_config():
    config = config_dmd2_default.create_config()
    config.trainer.fsdp = True
    config.model.meta_init = True
    config.trainer.fsdp_cpu_offload = True

    # Camera diagnostics: attach the VipeLogger as a W&B log module so each
    # logged step ships the GT camera-trajectory plot + WASD/IJKL action
    # overlays (derived from the GT camera poses) on the saved student / real /
    # teacher clips. Set ``callbacks.wandb.log_modules["vipe"].run_vipe_on_generated
    # = True`` to also report scale-aligned pose/intrinsics error (needs vipe).
    config.trainer.callbacks.update(VIPE_CALLBACK)

    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.model.net_optimizer.lr = 1e-6
    config.model.discriminator_optimizer.lr = 1e-6
    config.model.fake_score_optimizer.lr = 1e-6

    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"
    config.model.precision_fsdp_reduce = "bfloat16"

    # VAE compress ratio for WAN 14B: (1+T/4) * H / 8 * W / 8
    # 480p (832x480) resolution, 10s clips (sequence_length=161): 41 x 60 x 104 latent
    config.model.input_shape = [16, 41, 60, 104]  # cthw

    # LingBot-world is a Wan2.2-A14B-based MoE, so reuse the 14B discriminator.
    config.model.discriminator = copy.deepcopy(Discriminator_Wan_14B_Config)
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [21, 30, 39]
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"

    config.model.net = copy.deepcopy(LingBot_World_Config)
    config.model.guidance_cfg = LingBotWorld.default_guidance_cfg()

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.student_sample_type = "ode"

    # 4-step student schedule. With LingBot boundary 0.947, steps 1-2
    # use the high-noise expert and steps 3-4 use the low-noise expert.
    config.model.student_grid_cfg.steps = 4
    config.model.student_grid_cfg.grid_type = "explicit"
    config.model.student_grid_cfg.grid_kwargs = {"timesteps": [0.999, 0.950, 0.833, 0.624, 0.0]}

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

    config.log_config.group = "wancami2v_lingbot_dmd2"
    return config
