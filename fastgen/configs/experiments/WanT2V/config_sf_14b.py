# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.discriminator import Discriminator_Wan_14B_Config
import fastgen.configs.methods.config_self_forcing as config_self_forcing_default

from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.net import Wan21_T2V_14B_Config, CausalWan_14B_Config
from fastgen.networks.Wan.network import Wan

""" Configs for the Self-Forcing model, on WAN 14B model. """


def create_config():
    config = config_self_forcing_default.create_config()
    # Pretrain the causal model using SFT or KD and start from the checkpoint
    # config.trainer.checkpointer.pretrained_ckpt_path = /path/to/pretrained/ckpt.pth

    config.model.net_optimizer.lr = 2e-6
    config.model.discriminator_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.lr = 2e-6

    config.trainer.fsdp = True
    config.model.meta_init = True
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"
    # VAE compress ratio: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]
    config.model.fake_score_pred_type = "x0"
    config.model.net = copy.deepcopy(CausalWan_14B_Config)
    config.model.net.total_num_frames = config.model.input_shape[1]
    config.model.teacher = copy.deepcopy(Wan21_T2V_14B_Config)
    config.model.guidance_cfg = Wan.default_guidance_cfg()

    # GAN settings
    config.model.gan_loss_weight_gen = 0.003
    config.model.discriminator = copy.deepcopy(Discriminator_Wan_14B_Config)
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [21, 30, 39]
    config.model.gan_use_same_t_noise = True
    config.model.enable_preprocessors = False

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.student_grid_cfg.grid_type = "explicit"
    config.model.student_grid_cfg.grid_kwargs = {"timesteps": [0.999, 0.937, 0.833, 0.624, 0.0]}

    # EMA can lead to better performance when running longer
    # config.model.use_ema = True
    # config.trainer.callbacks.ema.start_iter = 200
    # config.trainer.callbacks.ema.beta = 0.99

    config.dataloader_train = copy.deepcopy(VideoLatentLoaderConfig)
    config.dataloader_train.batch_size = 1

    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "wan_14b_sf"
    return config
