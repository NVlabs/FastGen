# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-simulation PDD recipe for MiniMax-H3 audio+video.

The full-resolution run requires
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` to avoid allocator fragmentation.
"""

import copy

import fastgen.configs.methods.config_pdd as config_pdd_default
from fastgen.callbacks.carry import CarryCallback
from fastgen.configs.data import PromptLoaderConfig
from fastgen.configs.net import MiniMaxH3_Config
from fastgen.networks.MiniMaxH3.network import MINIMAX_H3_FLOW_SHIFT
from fastgen.networks.MiniMaxH3.network_pdd import MiniMaxH3PDD
from fastgen.configs.instantiate import LazyCall as L

NFES = (4, 8)
GRID_SIZE = 32
TYPE_RK = "euler"


def create_config():
    config = config_pdd_default.create_config()

    # ---- precision ----
    config.trainer.fsdp = True
    config.trainer.fsdp_cpu_offload = True
    config.model.meta_init = True
    config.model.precision = "float32"
    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"

    # ---- network: student (MiniMaxH3PDD, widened) + teacher (vanilla MiniMax-H3) ----
    # pdd_steps is the parallel-decoding *grid resolution* (velocity fields predicted per
    # forward), NOT the number of inference steps -- that is ``student_grid_cfg.steps`` below.
    # ``shift`` is the clock: the shared time axis is video's own noise level, so this places the
    # grid at video's released 12 and audio reaches its own noise level through the network's
    # ``relative_audio_shift``.
    config.model.teacher = copy.deepcopy(MiniMaxH3_Config)
    config.model.net = copy.deepcopy(MiniMaxH3_Config)
    config.model.net._target_ = MiniMaxH3PDD
    config.model.net.pdd_steps = GRID_SIZE
    config.model.student_grid_cfg.steps = min(NFES)
    config.model.student_grid_cfg.grid_type = "native"
    config.model.student_grid_cfg.grid_kwargs = {
        "base_steps": GRID_SIZE,
        "shift": MINIMAX_H3_FLOW_SHIFT,
    }

    # ---- input shape (video + audio; audio is inferred from the video shape) ----
    # Video latent ``[C, F, H, W]``; pixels = (H*16) x (W*16), frames = 17*n + 5 for
    # F = 5*n + 2. Audio is ``[32, 2, L]`` on the shared 40 Hz clock.
    # Resolution presets (video latent only; audio is derived). The rightmost column is the
    # packed sequence length, which is what attention and the activation checkpoints scale
    # with -- and, multiplied by pdd_steps, what the widened output projection scales with:
    #   [24, 37, 34, 60]  ->  544 x 960,  124 frames ( 5.17 s)  ~19.3k rows
    #   [24, 72, 34, 60]  ->  544 x 960,  243 frames (10.12 s)  ~37.5k rows
    #   [24, 37, 48, 84]  ->  768 x 1344, 124 frames ( 5.17 s)  ~37.7k rows (trained canvas)
    #   [24, 72, 48, 84]  ->  768 x 1344, 243 frames (10.12 s)  ~73.5k rows (trained canvas)
    # Memory scales with pdd_steps and packed sequence length.
    config.model.input_shape = {"video": [24, 37, 48, 84], "audio": None}

    # ---- optimizer ----
    config.model.net_optimizer.lr = 5e-6
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler.warm_up_steps = [0]
    config.trainer.callbacks.grad_clip.grad_norm = 1

    # ---- PDD blocks ----
    config.model.loss_config.type_rk = TYPE_RK
    config.model.loss_config.num_supervised_heads = 2 if TYPE_RK == "euler" else 1
    config.model.loss_config.block_size_min = GRID_SIZE // max(NFES)
    config.model.loss_config.block_size_max = GRID_SIZE // min(NFES)

    # ---- backward simulation + cross-iteration carry ----
    config.model.backward_simulation = "staggered"
    config.trainer.callbacks.carry = L(CarryCallback)()

    # ---- EMA disabled (saves memory: no EMA copy of the widened net) ----
    config.trainer.callbacks.pop("ema", None)

    # ---- generic prompt loaders (backward-simulation PDD needs only the prompts) ----
    config.dataloader_train = copy.deepcopy(PromptLoaderConfig)
    # MiniMax-H3 is guidance-distilled and does not use a negative prompt.
    config.dataloader_train.presets_map = {}
    config.trainer.batch_size_global = 512

    # ---- validation ----
    # MAX_VAL_STEPS / STUDENT_STEPS are global vars honored by Trainer.validate() and
    # FastGenModel._student_sample_loop; each dict in global_vars_val triggers a separate validation
    # pass. Validation carry is disabled by default, so each batch independently samples media at
    # 4 or 8 NFE and starts backward simulation from the first coarse interval.
    config.dataloader_val = copy.deepcopy(PromptLoaderConfig)
    config.dataloader_val.presets_map = {}
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 1, "STUDENT_STEPS": nfe} for nfe in NFES]
    config.trainer.validation_iter = 25
    config.trainer.val_seed = 42

    # ---- trainer schedule ----
    config.trainer.max_iter = 300
    config.trainer.logging_iter = 25
    config.trainer.save_ckpt_iter = 50
    # Write media/samples from 8 ranks (capped by world size) instead of just rank 0.
    config.trainer.callbacks.wandb.max_ranks = 8

    config.log_config.group = "minimax_h3_pdd"

    return config
