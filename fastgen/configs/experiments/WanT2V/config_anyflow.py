# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AnyFlow flow-map pretrain config on Wan-1.3B T2V (paper Stage 1).

AnyFlow's pretrain objective is MeanFlow's with a fixed ``beta08`` per-timestep
weighting, a finite-difference JVP, shifted timestep sampling, and a
``consistency_ratio`` fraction of the batch pinned to ``r = 0`` — so this config
runs ``MeanFlowModel`` directly. The values below mirror the reference recipe
``train_wan1b_student_shift5_81f_480p_lr5e-5_6k_b32.yml``.

Known deviations from the reference, both of which also apply to the on-policy stage:
full-rank fine-tuning instead of the paper's rank-256 LoRA; and under
``guidance_fuse_scale`` the 1/g rescaling of dF/dt is applied only to the samples that
kept their condition, where the reference's ``compute_central_difference`` rescales the
whole batch. A dropped sample's fused prediction is plain ``u_uncond``, so an ungated
1/g would regress it onto a different fixed point -- ours is gated on ``keep``.

The on-policy stage (paper Stage 2) lives in ``config_anyflow_onpolicy.py``.
"""

import copy

import fastgen.configs.methods.config_mean_flow as config_mean_flow
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan_1_3B_Config
from fastgen.configs.config_utils import model_pixel_shape


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
    # Noise-schedule bounds, forwarded to RFNoiseSchedule.
    config.model.net.min_t = 0.0
    config.model.net.max_t = 1.0

    config.trainer.fsdp = True
    config.model.precision_model = "bfloat16"
    # FSDP2 parameter storage and gradient reduction in fp32 while compute stays
    # bfloat16 -- the same split as the reference. With eager execution or DDP,
    # precision_opt still controls storage; use precision_amp for autocast.
    config.model.precision_opt = "float32"

    # VAE compress ratio: (1 + T/4) * H/8 * W/8. 81-frame, 480p clips.
    config.model.input_shape = [16, 21, 60, 104]

    # ------ AnyFlow loss: MeanFlow l2 with fixed beta08 weighting ------
    config.model.loss_config.use_cd = False
    config.model.loss_config.loss_type = "l2"
    # Fixed beta08 per-timestep weighting on the per-element mean loss, no
    # adaptive normalization — matching the reference train_bidirection.
    config.model.loss_config.weight_type = "beta08"
    config.model.loss_config.norm_method = None
    config.model.loss_config.use_jvp_finite_diff = True
    # Reference epsilon=5 in 1000-step units = 5e-3 in FastGen's continuous time.
    config.model.loss_config.jvp_finite_diff_eps = 5e-3
    # Rebalance the flow-map / consistency (r < t) sample losses to the global
    # flow-matching (r = t) loss mean (reference scale_weight).
    config.model.loss_config.rebalance_to_flow_matching = True
    config.model.precision_amp_jvp = "float32"

    # Prediction-side guidance fusion with text dropout (reference:
    # drop_text_ratio=0.1, fuse_guidance_scale=3.0): the conditional output
    # learns the guided flow directly. `guidance_cfg` stays at its empty default
    # — MeanFlow's target-side eq. 19/21 fusion is a different mechanism.
    config.model.guidance_fuse_scale = 3.0
    config.model.cond_dropout_prob = 0.1

    # ------ (t, r) sampling: shifted uniform pairs + AnyFlow buckets ------
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.0
    config.model.sample_t_cfg.max_t = 1.0
    # diffusion_ratio=0.5 of the batch keeps r = t (pure flow matching).
    config.model.sample_t_cfg.flow_matching_ratio = 0.5
    # consistency_ratio=0.25 of the batch is pinned to r = 0 (the reference
    # sets r = 0 pre-shift, and the shift maps 0 to 0).
    config.model.sample_t_cfg.consistency_ratio = 0.25
    # The reference assigns both buckets by rank-indexed partition of the
    # global batch, not by an independent per-sample draw.
    config.model.sample_t_cfg.deterministic_buckets = True

    # ------ optimization (reference: AdamW lr=5e-5, wd=0, betas=(0.9, 0.95),
    # max_grad_norm=1.0, 1000-step LR warmup, EMA decay 0.999) ------
    config.model.net_optimizer.optim_type = "adamw"
    config.model.net_optimizer.lr = 5e-5
    config.model.net_optimizer.betas = (0.9, 0.95)
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler.warm_up_steps = [1000]
    config.trainer.callbacks.grad_clip.grad_norm = 1.0
    config.trainer.callbacks.ema.beta = 0.999
    # Reference `ema_warmup_step: 1000` (as `start_iter = warmup_steps - 1`).
    config.trainer.callbacks.ema.start_iter = 999

    # ------ inference ------
    config.model.student_sample_type = "ode"
    config.model.student_grid_cfg.steps = 4
    # Independent flow-map grid policy; this recipe happens to use the same shift as t/r sampling.
    config.model.student_grid_cfg.grid_kwargs["shift"] = 5.0

    # ------ data / trainer ------
    config.dataloader_train = copy.deepcopy(VideoLoaderConfig)
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames
    config.dataloader_train.batch_size = 1

    config.trainer.max_iter = 6000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500
    config.trainer.batch_size_global = 32

    config.log_config.group = "wan_anyflow"
    return config
