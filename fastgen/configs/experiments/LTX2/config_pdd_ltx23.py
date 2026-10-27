# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel Decoding Distillation (PDD) config for LTX-2.3 audio+video.

Multimodal, backward-simulation, prompt-driven distillation of LTX-2.3 at 704p x ~10s. The
student is an ``LTX2PDD`` whose two output projections are widened by
``pdd_steps`` so it predicts the whole velocity grid in one pass; the teacher is
the unmodified LTX-2.3. ``input_shape`` is a modality dict, so every PDD
state/velocity flows through the per-modality path; the audio shape is inferred
from the video shape (``"audio": None``), matching the LTX-2.3 DMD2 / SFT configs.
"""

import copy

import fastgen.configs.methods.config_pdd as config_pdd_default
from fastgen.callbacks.carry import CarryCallback
from fastgen.configs.data import PromptLoaderConfig
from fastgen.configs.net import LTX2_3_Config
from fastgen.networks.LTX2.network import LTX2
from fastgen.networks.LTX2.network_pdd import LTX2PDD
from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.config_utils import model_pixel_shape


NFES = (4, 8)
GRID_SIZE = 256


def create_config():
    config = config_pdd_default.create_config()

    # ---- precision ----
    config.trainer.fsdp = True
    config.model.meta_init = True
    config.model.precision = "float32"
    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"

    # ---- network: student (LTX2PDD, widened) + teacher (vanilla LTX-2.3) ----
    # pdd_steps is the parallel-decoding *grid resolution* (velocity fields predicted
    # per forward), NOT the number of inference steps -- that is
    # ``student_grid_cfg.steps`` below. ``shift`` warps the grid the same way LTX-2's
    # FlowMatchEuler schedule does (sigma = a*t / (1 + (a-1)*t))
    config.model.teacher = copy.deepcopy(LTX2_3_Config)
    config.model.net = copy.deepcopy(LTX2_3_Config)
    config.model.net._target_ = LTX2PDD
    # Reference runs train the connectors; keep them unfrozen in the student.
    config.model.net.train_connectors = True
    config.model.net.pdd_steps = GRID_SIZE
    config.model.student_grid_cfg.steps = 4
    config.model.student_grid_cfg.grid_type = "native"
    config.model.student_grid_cfg.grid_kwargs = {"base_steps": GRID_SIZE, "shift": 10.0}

    # ---- input shape (video + audio; audio is inferred from the video shape) ----
    # Latent ``[C, F, H, W]``; pixels = (H*32) x (W*32), frames = (F-1)*8 + 1.
    # Resolution presets (video latent only; audio is derived):
    #   [128,  7, 17, 30]  -> 544 x 960,   49 frames  (lightest)
    #   [128, 13, 17, 30]  -> 544 x 960,   97 frames
    #   [128, 31, 22, 40]  -> 704 x 1280, 241 frames  (active; matches LTX-2.3 DMD2/SFT)
    #   [128, 31, 34, 60]  -> 1088 x 1920, 241 frames (~1080p, auto-enables VAE tiling)
    config.model.input_shape = {"video": [128, 31, 22, 40], "audio": None}

    _, height, _ = model_pixel_shape(config.model)
    if height > 720:
        config.model.net.enable_vae_tiling = True
        config.model.teacher.enable_vae_tiling = True

    # ---- optimizer ----
    config.model.net_optimizer.lr = 1e-5
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler.warm_up_steps = [0]
    config.trainer.callbacks.grad_clip.grad_norm = 1

    # ---- teacher guidance (LTX2's own recipe, through predict_with_guidance) ----
    # Canonical LTX-2.3 recipe (per-modality CFG, STG 2.0 @ self-attn layer 28,
    # cross-modal guidance 3.0, final rescale 0.7) with the video CFG scale raised
    # from the inference default 3.0 to 4.5; audio stays at 7.0.
    config.model.guidance_cfg = LTX2.default_guidance_cfg(version="2.3")
    config.model.guidance_cfg.cfg.scale["video"] = 4.5

    # ---- PDD blocks ----
    # The two validation passes below sample exactly the trained NFE endpoints.
    config.model.loss_config.type_rk = "euler"
    config.model.loss_config.block_size_min = GRID_SIZE // max(NFES)
    config.model.loss_config.block_size_max = GRID_SIZE // min(NFES)

    # ---- backward simulation + cross-iteration carry ----
    # Paper data-free training keeps all ranks on one shared trajectory phase.
    config.model.backward_simulation = "synced"
    config.trainer.callbacks.carry = L(CarryCallback)()

    # ---- EMA disabled (saves memory: no EMA copy of the widened net) ----
    config.trainer.callbacks.pop("ema", None)

    # ---- generic prompt loaders (backward-simulation PDD needs only the prompts) ----
    # Keep the requested batch independent of launch world size.
    config.dataloader_train = copy.deepcopy(PromptLoaderConfig)
    config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_ltx2"}
    config.trainer.batch_size_global = 2048

    # ---- validation ----
    # MAX_VAL_STEPS / STUDENT_STEPS are global vars honored by
    # Trainer.validate() and FastGenModel._student_sample_loop; each dict in global_vars_val
    # triggers a separate validation pass. Validation carry is disabled by default, so
    # each batch independently samples media at 4 or 8 NFE and starts backward simulation
    # from the first coarse interval.
    config.dataloader_val = copy.deepcopy(PromptLoaderConfig)
    config.dataloader_val.presets_map = {"neg_condition_raw": "neg_prompt_ltx2"}
    config.trainer.global_vars_val = [
        {"MAX_VAL_STEPS": 8, "STUDENT_STEPS": 4},
        {"MAX_VAL_STEPS": 8, "STUDENT_STEPS": 8},
    ]
    config.trainer.validation_iter = 10
    config.trainer.val_seed = 42

    # ---- trainer schedule ----
    config.trainer.max_iter = 250
    config.trainer.logging_iter = 10
    config.trainer.save_ckpt_iter = 25
    config.trainer.callbacks.wandb.sample_logging_iter = 10
    # Write media/samples from 8 ranks (capped by world size) instead of just rank 0.
    config.trainer.callbacks.wandb.max_ranks = 8

    config.log_config.group = "ltx2_3_pdd"

    return config
