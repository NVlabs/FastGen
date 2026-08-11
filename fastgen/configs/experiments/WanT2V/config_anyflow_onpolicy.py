# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AnyFlow on-policy distillation config on Wan-1.3B T2V (paper Stage 2).

DMD2-style distribution matching with a flow-map student that generates via a
compressed rollout (jump -> fine step -> jump, gradient through all segments,
NFE sampled per iteration) and co-trains the Stage-1 flow-map loss at every
student update — see ``AnyFlowModel``. The values below mirror the reference
recipe ``train_wan1b_onpolicy_81f_480p_lr2e-6_1k_b32.yml``, which runs 1200
generator updates.

Two checkpoints must be supplied. The student is seeded from a Stage 1 trainer
checkpoint, read directly with no conversion; ``pretrained_ckpt_key_map`` below
 takes it from the Stage 1 ``ema`` weights, matching the reference's
``pretrained_weight: ema``::

    trainer.checkpointer.pretrained_ckpt_path=<stage1>/checkpoints/0006000.pth

Known deviation from the reference: full-rank fine-tuning instead of the paper's
rank-256 LoRA.
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
    # Full [0, 1] noise schedule, as in the pretrain stage
    config.model.net.min_t = 0.0
    config.model.net.max_t = 1.0

    # ------ teacher / fake score: PLAIN single-timestep Wan ------
    # Neither consumes r (DMD2 queries both at the instantaneous velocity r = t)
    # and the reference builds them without the flow-map pathway at all.
    config.model.teacher = copy.deepcopy(Wan_1_3B_Config)
    config.model.teacher.r_timestep = False

    # Student init comes from the Stage 1 trainer checkpoint via
    # trainer.checkpointer.pretrained_ckpt_path (see the module docstring);
    # "ema" mirrors the reference's `pretrained_weight: ema`.
    config.trainer.checkpointer.pretrained_ckpt_key_map = {"net": "ema", "ema": "ema"}

    config.model.precision = "bfloat16"
    # FSDP2 parameter storage and gradient reduction in fp32 while compute stays
    # bfloat16 -- the same split as the reference. Takes
    # effect only under FSDP (`trainer.ddp=False`); it is ignored under DDP, where
    # params, grads and compute are all `precision`.
    config.model.precision_fsdp = "float32"

    # VAE compress ratio: (1 + T/4) * H/8 * W/8. 81-frame, 480p clips.
    config.model.input_shape = [16, 21, 60, 104]

    # ------ DMD machinery ------
    # The reference Stage 2 has no adversarial loss: its "discriminator" is
    # the fake score network, trained with denoising score matching only.
    config.model.gan_loss_weight_gen = 0.0
    # The reference updates generator and fake score 1:1 (both every global
    # step); in DMD2's alternating scheme that is student_update_freq=2.
    config.model.student_update_freq = 2
    # Reference real_guidance_scale=3.0 applies cond + 3*(cond - uncond);
    # FastGen's CFG formula is cond + (g-1)*(cond - uncond), so g=4.
    config.model.guidance_scale = 4.0

    # DMD gradient noising time: reference `generator_loss` draws torch.rand
    # then applies the shift -> shifted-uniform.
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.0
    config.model.sample_t_cfg.max_t = 1.0

    # Fake-score noising time: reference `discriminator_loss` draws
    # logit_normal(0, 1) then applies the same shift. This is a DIFFERENT
    # density from the DMD path above, so it needs its own config.
    config.model.fake_score_sample_t_cfg = copy.deepcopy(config.model.sample_t_cfg)
    config.model.fake_score_sample_t_cfg.time_dist_type = "shifted_logitnormal"
    config.model.fake_score_sample_t_cfg.train_p_mean = 0.0
    config.model.fake_score_sample_t_cfg.train_p_std = 1.0

    # ------ student rollout (reference rollout_cfg) ------
    config.model.student_sample_type = "ode"
    config.model.student_sample_steps_list = [2, 4, 8, 16, 50]
    config.model.student_sample_steps = 4
    # Validation schedule for `student_sample_steps = 4`: `shift * s / (1 + (shift
    # - 1) * s)` on `s = linspace(1, 0, 5)` -- the same grid the per-NFE rollout
    # builds. Recompute if `shift`, `max_t` or `student_sample_steps` changes.
    config.model.sample_t_cfg.t_list = [1.0, 0.9375, 0.8333333333333334, 0.625, 0.0]

    # ------ co-trained Stage-1 flow-map loss (reference cotrain_forward_kl) ------
    # FastGen's VSD loss carries a 0.5 factor the reference's DMD loss does
    # not; 0.5 here keeps the DMD : flow-map gradient ratio at the
    # reference's 1 : 1.
    config.model.cotrain_pretrain_weight = 0.5
    config.model.loss_config.use_cd = False
    config.model.loss_config.loss_type = "l2"
    config.model.loss_config.weight_type = "beta08"
    config.model.loss_config.norm_method = None
    config.model.loss_config.use_jvp_finite_diff = True
    config.model.loss_config.jvp_finite_diff_eps = 5e-3
    config.model.loss_config.rebalance_to_flow_matching = True
    config.model.loss_config.guidance_fuse_scale = 3.0
    config.model.cond_dropout_prob = 0.1
    config.model.precision_amp_jvp = "float32"
    config.model.sample_t_cfg.flow_matching_ratio = 0.5
    config.model.sample_t_cfg.consistency_ratio = 0.25
    config.model.sample_t_cfg.deterministic_buckets = True

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
    # EMA start = 6399 = 6000 + 400 - 1: a 400-iteration warmup (the reference's
    # `ema_warmup_step: 200` x `student_update_freq = 2`) offset by the Stage-1
    # checkpoint iteration this stage seeds from -- `EMACallback` adds
    # `model.resume_iter`, so a bare 399 is already past and never warms up.
    # Seeding from another checkpoint: use <its iteration> + 399.
    config.trainer.callbacks.ema.start_iter = 6399

    # ------ data / trainer ------
    config.dataloader_train = VideoLoaderConfig
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1
    config.dataloader_train.batch_size = 1

    # The reference runs 1200 global steps, each performing BOTH a generator
    # and a fake-score update. FastGen alternates them across iterations
    # (student_update_freq=2), so 1200 generator updates need 2 * 1200 = 2400
    # iterations here.
    config.trainer.max_iter = 2400
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 400
    config.trainer.batch_size_global = 32

    config.log_config.group = "wan_anyflow_onpolicy"
    return config
