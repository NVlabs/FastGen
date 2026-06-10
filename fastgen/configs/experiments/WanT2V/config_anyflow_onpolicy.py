# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AnyFlow on-policy distillation config on Wan-1.3B T2V (paper Stage 3).

DMD2-style distribution matching with a flow-map student that generates via
a compressed rollout (jump -> fine step -> jump, gradient through all
segments, NFE sampled per iteration from [2, 4, 8, 16, 50]) and co-trains the
Stage-2 flow-map loss at every student update — see
:class:`~fastgen.methods.distribution_matching.anyflow.AnyFlowModel`.

Mirrors the reference recipe
(``train_wan1b_onpolicy_81f_480p_lr2e-6_1k_b32.yml``): 1.2k iterations at
lr=2e-6, AdamW betas=(0.0, 0.999), wd=0, grad clip 1.0, EMA 0.99,
real-score CFG strength 3, no adversarial loss (the reference
"discriminator" is the fake score network). Set
``config.model.pretrained_student_net_path`` to a Stage 2 checkpoint from
``config_anyflow.py`` before launching, and point the teacher
(``config.model.pretrained_model_path``) at a flow-map teacher checkpoint —
the reference initializes both real and fake score from a separately
fine-tuned flow-map teacher, not stock Wan2.1.

Known deviations from the reference, both documented:
* full-rank fine-tuning instead of the paper's rank-256 LoRA (FastGen has no
  PEFT path today);
* the fake-score noising time is shifted-uniform (FastGen's sampler) instead
  of shifted-logit-normal(0, 1).
"""

import copy

import fastgen.configs.methods.config_anyflow as config_anyflow_default
from fastgen.configs.data import VideoLoaderConfig
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

    # ------ DMD machinery ------
    # The reference Stage 3 has no adversarial loss: its "discriminator" is
    # the fake score network, trained with denoising score matching only.
    config.model.gan_loss_weight_gen = 0.0
    # The reference updates generator and fake score 1:1 (both every global
    # step); in DMD2's alternating scheme that is student_update_freq=2.
    config.model.student_update_freq = 2
    # Reference real_guidance_scale=3.0 applies cond + 3*(cond - uncond);
    # FastGen's CFG formula is cond + (g-1)*(cond - uncond), so g=4.
    config.model.guidance_scale = 4.0

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # ------ student rollout (reference rollout_cfg) ------
    config.model.student_sample_type = "ode"
    config.model.student_sample_steps = 4  # used for validation sampling
    config.model.student_sample_steps_list = [2, 4, 8, 16, 50]
    config.model.sample_t_cfg.t_list = None  # rollout schedules are computed per NFE

    # ------ co-trained Stage-2 flow-map loss (reference cotrain_forward_kl) ------
    # FastGen's VSD loss carries a 0.5 factor the reference's DMD loss does
    # not; 0.5 here keeps the DMD : flow-map gradient ratio at the
    # reference's 1 : 1 (the common 0.5 scale is absorbed by Adam).
    config.model.cotrain_pretrain_weight = 0.5
    config.model.loss_config.use_cd = False
    config.model.loss_config.loss_type = "l2"
    config.model.loss_config.weight_type = "beta08"
    config.model.loss_config.use_jvp_finite_diff = True
    config.model.loss_config.jvp_finite_diff_eps = 5e-3
    config.model.loss_config.rebalance_to_diffusion = True
    config.model.loss_config.guidance_fuse_scale = 3.0
    config.model.cond_dropout_prob = 0.1
    config.model.precision_amp_jvp = "float32"
    config.model.sample_t_cfg.r_sample_ratio = 0.5
    config.model.sample_t_cfg.consistency_ratio = 0.25

    # ------ optimization (reference: AdamW lr=2e-6, betas=(0.0, 0.999), wd=0,
    # grad clip 1.0, EMA 0.99) ------
    config.model.net_optimizer.lr = 2e-6
    config.model.net_optimizer.betas = (0.0, 0.999)
    config.model.net_optimizer.weight_decay = 0.0
    config.model.fake_score_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.betas = (0.0, 0.999)
    config.model.fake_score_optimizer.weight_decay = 0.0
    config.trainer.callbacks.grad_clip.grad_norm = 1.0
    config.trainer.callbacks.ema.beta = 0.99

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
