# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.discriminator import Discriminator_Wan_1_3B_Config
import fastgen.configs.methods.config_self_forcing as config_self_forcing_default

from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import SELF_FORCING_ODE_INIT_URL, Wan_1_3B_Config, CausalWan_1_3B_Config
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.Wan.network import Wan

""" Configs for the Self-Forcing model, on WAN 1.3B model. """


def create_config():
    config = config_self_forcing_default.create_config()
    config.model.net_optimizer.lr = 2e-6
    config.model.discriminator_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.lr = 2e-6

    config.trainer.fsdp = True
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"
    # VAE compress ratio: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]
    config.model.fake_score_pred_type = "x0"
    config.model.net = copy.deepcopy(CausalWan_1_3B_Config)
    config.model.net.total_num_frames = config.model.input_shape[1]
    config.model.teacher = copy.deepcopy(Wan_1_3B_Config)
    config.model.guidance_cfg = Wan.default_guidance_cfg()

    # GAN settings
    config.model.gan_loss_weight_gen = 0.003
    config.model.discriminator = copy.deepcopy(Discriminator_Wan_1_3B_Config)
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [15, 22, 29]
    config.model.gan_use_same_t_noise = True

    # Pinned Self-Forcing ODE initialization; rank 0 mirrors it into the shared model cache.
    config.model.pretrained_student_net_path = SELF_FORCING_ODE_INIT_URL

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.student_grid_cfg.grid_type = "explicit"
    config.model.student_grid_cfg.grid_kwargs = {"timesteps": [0.999, 0.937, 0.833, 0.624, 0.0]}

    # EMA can lead to better performance when running longer
    # config.model.use_ema = True
    # config.trainer.callbacks.ema.start_iter = 200
    # config.trainer.callbacks.ema.beta = 0.99

    config.dataloader_train = copy.deepcopy(VideoLoaderConfig)
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps
    config.dataloader_train.batch_size = 1

    # 480p (832x480) resolution
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames

    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "wan_sf"
    return config
