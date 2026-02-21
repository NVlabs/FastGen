# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from omegaconf import DictConfig

from fastgen.configs.callbacks import EMA_CALLBACK
from fastgen.configs.discriminator import Discriminator_Wan_1_3B_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import VideoLatentLoaderConfig

from fastgen.configs.net import VACE_Wan_1_3B_Config, CKPT_ROOT_DIR

""" Configs for the DMD2 model on VACE WAN 1.3B model. """


def create_config():
    config = config_dmd2_default.create_config()
    config.model.fsdp_meta_init = True

    # ema
    config.model.use_ema = ["ema"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CALLBACK)

    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    config.model.precision = "bfloat16"
    # VAE compress ratio: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]
    config.model.discriminator = Discriminator_Wan_1_3B_Config
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [15, 22, 29]
    config.model.gan_loss_weight_gen = 0.03
    config.model.guidance_scale = 4.0
    config.model.net = VACE_Wan_1_3B_Config
    config.model.net.depth_model_path = f"{CKPT_ROOT_DIR}/annotators/depth_anything_v2_vitl.pth"
    config.model.net.total_num_frames = config.model.input_shape[1]

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"
    config.model.student_sample_type = "ode"

    # setting for 2-step training
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    config.dataloader_train = VideoLatentLoaderConfig

    # 480p (832x480) resolution
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.log_config.group = "wan21_dmd2"
    return config
