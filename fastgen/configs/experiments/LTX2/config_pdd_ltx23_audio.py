# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-free LTX-2.3 audio-only PDD for four- and eight-step generation.

Train on text prompts with staggered backward simulation and generate roughly
ten seconds of stereo audio. One student covers both step counts; validation
samples each separately.

Qwen-Image PDD supplies the 128-head midpoint targets and 3,000-step training
schedule. The LTX PDD base supplies shift 10, AdamW at 1e-5 with betas (0.9, 0.99),
zero weight decay/warmup, global batch 2048, gradient clipping at 1, trainable
connectors, and no EMA.
"""

import fastgen.configs.experiments.LTX2.config_pdd_ltx23 as base
from fastgen.networks.LTX2.network import LTX2


NFES = (4, 8)
GRID_SIZE = 128


def create_config():
    config = base.create_config()

    config.model.net.modalities = ("audio",)
    config.model.teacher.modalities = ("audio",)
    # Ten seconds at 16 kHz -> 1001 centered mel frames -> 251 causal VAE frames.
    config.model.input_shape = [8, 251, 16]
    # PDD calls the guided teacher directly, so supply the scalar audio recipe
    # explicitly: CFG 7, STG 2 at layer 28, rescale 0.7, no cross-modal guidance.
    config.model.guidance_cfg = LTX2.default_guidance_cfg(modalities=("audio",), version="2.3")
    config.model.extra_fsdp_modules = ["teacher", "text_encoder"]

    config.model.net.pdd_steps = GRID_SIZE
    config.model.student_grid_cfg.steps = min(NFES)
    config.model.student_grid_cfg.grid_kwargs["base_steps"] = GRID_SIZE
    config.model.loss_config.type_rk = "midpoint"
    config.model.loss_config.num_supervised_heads = 1
    config.model.loss_config.block_size_min = GRID_SIZE // max(NFES)
    config.model.loss_config.block_size_max = GRID_SIZE // min(NFES)
    # CarryCallback is inherited; its eight-step trajectories have rank-offset phases.
    config.model.backward_simulation = "staggered"

    config.trainer.max_iter = 3000
    config.trainer.save_ckpt_iter = 250
    config.log_config.group = "ltx2_3_audio_pdd"
    return config
