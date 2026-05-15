# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference AnyFlow experiment config on Wan-1.3B T2V.

Mirrors the AnyFlow paper's pretrain configuration: 1.3B student initialised
from a Wan2.1-T2V checkpoint, flow-matching shift=5, beta08 loss weighting,
6k iterations with batch_size_global=32 and lr=5e-5.

Switching to the on-policy stage:

    config.model.loss_config.training_stage = "onpolicy"
    config.model.pretrained_student_net_path = "<path-to-pretrain-ckpt>"

and adjust ``student_update_freq`` / ``gan_loss_weight_gen`` to taste.
"""

import fastgen.configs.methods.config_anyflow as config_anyflow_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.discriminator import Discriminator_Wan_1_3B_Config
from fastgen.configs.net import Wan_1_3B_Config


def create_config():
    config = config_anyflow_default.create_config()

    # Default to the pretrain stage; flip the switch to "onpolicy" once the
    # flow-map pretrain checkpoint is available.
    config.model.loss_config.training_stage = "pretrain"
    config.model.loss_config.jvp_finite_diff_eps = 5e-3
    config.model.loss_config.diffusion_ratio = 0.5
    config.model.loss_config.consistency_ratio = 0.25
    config.model.loss_config.weight_type = "beta08"
    config.model.loss_config.shift = 5.0

    config.model.net = Wan_1_3B_Config
    config.model.net.r_timestep = True

    # The on-policy stage uses these too, but they are harmless in pretrain.
    config.model.discriminator = Discriminator_Wan_1_3B_Config
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [15, 22, 29]
    config.model.gan_loss_weight_gen = 0.0  # disabled by default in pretrain
    config.model.guidance_scale = 5.0

    config.model.precision = "bfloat16"
    # VAE compress ratio: (1 + T/4) * H/8 * W/8. 81-frame, 480p clips.
    config.model.input_shape = [16, 21, 60, 104]

    config.model.net_optimizer.lr = 5e-5
    config.model.fake_score_optimizer.lr = 5e-5
    config.model.discriminator_optimizer.lr = 5e-5

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.student_sample_type = "ode"
    # Any-step model — multiple NFEs validated at inference time.
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    config.dataloader_train = VideoLoaderConfig
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1
    config.dataloader_train.batch_size = 1

    config.trainer.max_iter = 6000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500
    config.trainer.batch_size_global = 32

    config.log_config.group = "wan_anyflow"
    return config
