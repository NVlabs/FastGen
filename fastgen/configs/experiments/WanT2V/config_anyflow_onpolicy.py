# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AnyFlow on-policy distillation config on Wan-1.3B T2V (paper Stage 3).

DMD2's alternating fake_score / discriminator updates with a flow-map student
that generates via a multi-step rollout-with-gradient (see
:class:`~fastgen.methods.distribution_matching.anyflow.AnyFlowModel`).
Mirrors the paper's Stage 3 hyperparameters: 1.2k iterations at lr=2e-6 on
top of a Stage 2 flow-map pretrain checkpoint (``config_anyflow.py``).

Note: the AnyFlow paper trains this stage with a rank-256 LoRA adapter, but
FastGen does not ship a PEFT/LoRA training path today, so this config does
full-rank fine-tuning. Set ``config.model.pretrained_student_net_path`` to a
checkpoint produced by :mod:`config_anyflow` before launching.
"""

import copy

import fastgen.configs.methods.config_anyflow as config_anyflow_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.discriminator import Discriminator_Wan_1_3B_Config
from fastgen.configs.net import Wan_1_3B_Config


def create_config():
    config = config_anyflow_default.create_config()

    # ------ network: gated dual-timestep Wan, same as the pretrain stage ------
    config.model.net = copy.deepcopy(Wan_1_3B_Config)
    config.model.net.r_timestep = True
    config.model.net.encoder_depth = None
    config.model.net.time_cond_type = "abs"
    config.model.net.r_embedder_fusion = "gated"
    config.model.net.r_embedder_gate_value = 0.25

    config.model.pretrained_student_net_path = "<path-to-stage2-pretrain-ckpt>"

    config.model.precision = "bfloat16"
    # VAE compress ratio: (1 + T/4) * H/8 * W/8. 81-frame, 480p clips.
    config.model.input_shape = [16, 21, 60, 104]

    # ------ DMD2 machinery ------
    config.model.discriminator = Discriminator_Wan_1_3B_Config
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [15, 22, 29]
    config.model.gan_loss_weight_gen = 0.03
    config.model.student_update_freq = 5
    config.model.guidance_scale = 3.0

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # ------ student rollout (AnyFlow's hand-tuned 4-step schedule) ------
    config.model.student_sample_type = "ode"
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # ------ Stage 3 learning rates from the AnyFlow paper ------
    config.model.net_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.lr = 2e-6
    config.model.discriminator_optimizer.lr = 2e-6

    # ------ data / trainer ------
    config.dataloader_train = VideoLoaderConfig
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1
    config.dataloader_train.batch_size = 1

    config.trainer.max_iter = 1200
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 200
    config.trainer.batch_size_global = 32

    config.log_config.group = "wan_anyflow_onpolicy"
    return config
