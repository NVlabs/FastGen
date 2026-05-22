# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference AnyFlow on-policy distillation config on Wan-1.3B T2V (Stage 3).

Inherits the pretrain config and flips the loss into the on-policy stage,
turning on DMD2's alternating fake_score / discriminator updates with the
``r=0`` dual-timestep conditioning that AnyFlow keeps from MeanFlow. Mirrors
the paper's Stage 3 hyperparameters: 1.2k iterations at lr=2e-6 on top of a
Stage 2 flow-map pretrain checkpoint.

Note: the AnyFlow paper trains this stage with a rank-256 LoRA adapter, but
FastGen does not ship a PEFT/LoRA training path today, so this config does
full-rank fine-tuning. Set ``config.model.pretrained_student_net_path`` to a
checkpoint produced by :mod:`config_anyflow` before launching.
"""

import fastgen.configs.experiments.WanT2V.config_anyflow as config_anyflow_pretrain


def create_config():
    config = config_anyflow_pretrain.create_config()

    config.model.loss_config.training_stage = "onpolicy"
    config.model.pretrained_student_net_path = "<path-to-stage2-pretrain-ckpt>"

    # Re-enable the DMD2 alternating-update machinery.
    config.model.gan_loss_weight_gen = 0.03
    config.model.student_update_freq = 5

    # Stage 3 learning rates from the AnyFlow paper.
    config.model.net_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.lr = 2e-6
    config.model.discriminator_optimizer.lr = 2e-6

    config.trainer.max_iter = 1200
    config.trainer.save_ckpt_iter = 200

    config.log_config.group = "wan_anyflow_onpolicy"
    return config
