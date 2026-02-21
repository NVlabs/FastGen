# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_mean_flow as config_mf_default
from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.net import Wan_1_3B_Config
from fastgen.utils import LazyCall as L
from fastgen.utils.lr_scheduler import LambdaInverseSquareRootScheduler
from fastgen.callbacks.ema import EMACallback

""" Configs for the MeanFlow model on Wan-1.3B model. """


def create_config():
    config = config_mf_default.create_config()

    # ema
    config.model.use_ema = ["ema_1"]
    config.trainer.callbacks.pop("ema")
    config.trainer.callbacks.update(dict(ema_1=L(EMACallback)(type="power", gamma=96.99, ema_name="ema_1")))
    config.trainer.callbacks.pop("grad_clip")

    # model
    config.model.precision = "bfloat16"
    # VAE compress ratio: (1+T/4) * H / 8 * W / 8
    # config.model.input_shape = [16, 21, 60, 104]
    config.model.input_shape = [16, 21, 60, 104]
    config.model.net_optimizer.optim_type = "adamw"
    config.model.net_optimizer.lr = 1e-5
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.eps = 1e-8
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler = L(LambdaInverseSquareRootScheduler)(
        warm_up_steps=0,
        decay_steps=2000,
    )

    # CFG guidance
    config.model.guidance_scale = 3.0
    config.model.cond_dropout_prob = 0.1
    config.model.precision_amp_jvp = "float32"
    config.model.loss_config.use_cd = True
    config.model.loss_config.pred_type = "flow"
    config.model.loss_config.norm_method = "poly_1.0"
    config.model.loss_config.norm_const = 10.0

    config.model.loss_config.use_jvp_finite_diff = True
    config.model.loss_config.jvp_finite_diff_eps = 5e-3
    config.model.loss_config.tangent_warmup_steps = 2000
    config.model.loss_config.tangent_spatial_invariance = True

    config.model.net = copy.deepcopy(Wan_1_3B_Config)
    config.model.net.r_timestep = True
    config.model.net.encoder_depth = None
    config.model.net.time_cond_type = "diff"
    config.model.net.r_embedder_init = "zero"
    config.model.net.norm_temb = False

    # we use simple diffusion version: 0.73 = 0.5 * log((21 * 60 * 104)/(64 * 64)) - 0.1
    config.model.enable_preprocessors = False
    config.model.sample_t_cfg.time_dist_type = "logitnormal"
    config.model.sample_t_cfg.r_sample_ratio = 1.0
    config.model.sample_t_cfg.train_p_mean = -0.8
    config.model.sample_t_cfg.train_p_std = 1.6
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.dataloader_train = VideoLatentLoaderConfig
    config.dataloader_train.batch_size = 1

    # 480p (832x480) resolution
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.log_config.group = "wan_mf"
    return config
