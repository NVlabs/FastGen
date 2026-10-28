# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.net import DiT_IN256_XL_Config
import fastgen.configs.experiments.DiT.config_mf_b as config_mf_default
from fastgen.networks.DiT.network import DiT


""" Configs for the MeanFlow model, on DiT-XL and ImageNet-256 dataset. """


def create_config():
    config = config_mf_default.create_config()

    # 3-way teacher mixture (MeanFlow paper eq. 21):
    #   guided = ω·dxt_dt + κ·pos + (1-ω-κ)·neg
    # where dxt_dt is the cond_velocity, ``pos``/``neg`` are the
    # network cond/uncond passes, ω is the coefficient on ``dxt_dt`` and κ
    # is the coefficient on ``pos``.  Here ω=0.2, κ=0.92, gated to t in
    # [0, 0.75) (so 1-ω-κ = -0.12 inside the window).  Composed via
    # ``cfg.scale=ω`` and ``mix.scale=κ+1`` (see ``apply_guidance`` identity
    # in mean_flow._get_velocity).  Outside the window both terms reduce
    # to scale=1 (no-op), so dxt_dt passes through.
    config.model.guidance_cfg = DiT.default_guidance_cfg()
    config.model.guidance_cfg.cfg.scale = [0.2, 1.0]
    config.model.guidance_cfg.cfg.boundary = [0.75]
    config.model.guidance_cfg.mix.scale = [1.92, 1.0]
    config.model.guidance_cfg.mix.boundary = [0.75]

    config.model.net = copy.deepcopy(DiT_IN256_XL_Config)
    # MeanFlow's JVP is incompatible with fused attention.
    config.model.net.enable_fused_attn = False
    # remove the additional 1000 factor in JVP
    config.model.net.scale_t = False
    config.model.net.r_timestep = True
    config.model.net.time_cond_type = "diff"

    config.dataloader_train.batch_size = 8

    return config
