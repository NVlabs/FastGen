# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AnyFlow flow-map pretrain config on Wan-1.3B T2V (paper Stage 2).

AnyFlow's pretrain objective is the MeanFlow objective with a fixed
``beta08`` per-timestep loss weighting, a finite-difference JVP, shifted
timestep sampling, and a ``consistency_ratio`` fraction of the batch pinned
to ``r = min_t`` — so this config runs :class:`MeanFlowModel` directly.

Mirrors the AnyFlow paper's pretrain recipe (``shift=5``, ``beta08``
weighting, ``epsilon=5e-3``, ``diffusion_ratio=0.5``,
``consistency_ratio=0.25``, lr=5e-5, 6k iterations, batch_size_global=32,
text dropout 0.1 with velocity-fused guidance 3.0) on the gated dual-timestep
Wan architecture (``gate=0.25``, absolute-r conditioning, matching
``deltatime_type: r`` in the reference).

The on-policy stage (paper Stage 3) lives in ``config_anyflow_onpolicy.py``.
"""

import copy

import fastgen.configs.methods.config_mean_flow as config_mean_flow
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan_1_3B_Config


def create_config():
    config = config_mean_flow.create_config()

    # ------ network: gated dual-timestep Wan (AnyFlow architecture) ------
    config.model.net = copy.deepcopy(Wan_1_3B_Config)
    config.model.net.r_timestep = True
    config.model.net.encoder_depth = None
    # AnyFlow conditions the r-pathway on the absolute r (deltatime_type "r")
    # and fuses the two time embeddings with a fixed convex gate of 0.25.
    config.model.net.time_cond_type = "abs"
    config.model.net.r_embedder_fusion = "gated"
    config.model.net.r_embedder_gate_value = 0.25

    config.model.precision = "bfloat16"
    # VAE compress ratio: (1 + T/4) * H/8 * W/8. 81-frame, 480p clips.
    config.model.input_shape = [16, 21, 60, 104]

    # ------ AnyFlow loss: MeanFlow l2 with fixed beta08 weighting ------
    config.model.loss_config.use_cd = False
    config.model.loss_config.loss_type = "l2"
    config.model.loss_config.weight_type = "beta08"
    config.model.loss_config.use_jvp_finite_diff = True
    config.model.loss_config.jvp_finite_diff_eps = 5e-3
    # Rebalance the non-diffusion (flow-map / consistency) sample losses to
    # the global diffusion-loss mean (reference scale_weight).
    config.model.loss_config.rebalance_to_diffusion = True
    config.model.precision_amp_jvp = "float32"

    # Prediction-side guidance fusion with text dropout (reference:
    # drop_text_ratio=0.1, fuse_guidance_scale=3.0): the conditional output
    # learns the guided flow directly. guidance_scale stays None — MeanFlow's
    # target-side eq. 19 fusion is a different mechanism.
    config.model.guidance_scale = None
    config.model.loss_config.guidance_fuse_scale = 3.0
    config.model.cond_dropout_prob = 0.1

    # ------ (t, r) sampling: shifted uniform pairs + AnyFlow buckets ------
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    # diffusion_ratio=0.5 of the batch keeps r = t (pure flow matching);
    # r_sample_ratio is the complementary fraction that keeps the sampled r.
    config.model.sample_t_cfg.r_sample_ratio = 0.5
    # consistency_ratio=0.25 of the batch is pinned to r = min_t.
    config.model.sample_t_cfg.consistency_ratio = 0.25

    # ------ optimization (reference: AdamW lr=5e-5, wd=0, betas=(0.9, 0.95),
    # max_grad_norm=1.0, 1000-step LR warmup, EMA decay 0.999) ------
    config.model.net_optimizer.optim_type = "adamw"
    config.model.net_optimizer.lr = 5e-5
    config.model.net_optimizer.betas = (0.9, 0.95)
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler.warm_up_steps = [1000]
    config.trainer.callbacks.grad_clip.grad_norm = 1.0
    config.trainer.callbacks.ema.beta = 0.999

    # ------ inference / validation ------
    config.model.student_sample_type = "ode"
    config.model.student_sample_steps = 4
    # 4-step shifted schedule shift*s/(1+(shift-1)*s) on s=linspace(1,0,5),
    # first entry clamped to max_t (the reference samples from t=1.0).
    config.model.sample_t_cfg.t_list = [0.999, 0.9375, 0.83333333, 0.625, 0.0]

    # ------ data / trainer ------
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
