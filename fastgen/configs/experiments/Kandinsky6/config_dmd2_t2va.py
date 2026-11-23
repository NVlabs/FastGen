# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kandinsky 6 Pro T2VA DMD2: GAN-free four-step student at the native 5-second canvas.

Without the GAN no real data is needed: the generic ``PromptLoaderConfig`` supplies captions, and
``backward_simulation`` takes the student's inputs from its own trajectory. Video and audio share one
flow time and shift; the native precision contract is in the network module.
"""

import copy

import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import PromptLoaderConfig
from fastgen.configs.net import Kandinsky6Config
from fastgen.networks.Kandinsky6.network import Kandinsky6


def create_config():
    config = config_dmd2_default.create_config()
    config.trainer.fsdp = True
    config.model.meta_init = True
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"
    config.model.extra_fsdp_modules += ["text_encoder", "vae"]
    # Match the native factory's FP32 matrix products and validated codec execution.
    config.trainer.tf32_matmul = True
    config.trainer.cudnn.deterministic = True
    config.trainer.cudnn.benchmark = False

    config.model.net = copy.deepcopy(Kandinsky6Config)
    config.model.input_shape = {"video": [16, 31, 60, 108], "audio": None}
    config.model.guidance_cfg = Kandinsky6.default_guidance_cfg()
    config.model.net_optimizer.lr = 5e-6
    config.model.net_optimizer.fused = True
    config.model.fake_score_optimizer.lr = 5e-6
    config.model.fake_score_optimizer.fused = True
    config.model.fake_score_pred_type = "x0"

    # GAN-free: without a discriminator, drop its grad-clip callback too.
    config.model.gan_loss_weight_gen = 0.0
    config.model.discriminator = None
    config.trainer.callbacks.pop("grad_clip_discriminator", None)
    config.model.backward_simulation = "default"

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.student_sample_type = "ode"
    config.model.student_grid_cfg.steps = 4
    config.model.student_grid_cfg.grid_kwargs = {"shift": 5.0}

    config.dataloader_train = copy.deepcopy(PromptLoaderConfig)
    config.dataloader_train.batch_size = 2
    config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_kandinsky6"}

    config.trainer.batch_size_global = 16
    config.trainer.max_iter = 1000
    config.trainer.logging_iter = 20
    config.trainer.save_ckpt_iter = 100
    config.trainer.callbacks.wandb.max_ranks = 8
    config.log_config.group = "kandinsky6_t2va_dmd2"
    return config
