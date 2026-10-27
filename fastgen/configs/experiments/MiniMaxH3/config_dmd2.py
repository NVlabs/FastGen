# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DMD2 config for MiniMax-H3 audio+video: backward simulation, no GAN loss, four-step student.

Backward simulation means the student walks its own sampling grid one step per iteration,
``CarryCallback`` handing the state on. With the GAN off (below) no real data is used at all, so
this recipe is also data-free -- but that is the GAN's absence, not the backward-simulation flag.
"""

import copy

import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.callbacks.carry import CarryCallback
from fastgen.configs.data import PromptLoaderConfig
from fastgen.configs.net import MiniMaxH3_Config
from fastgen.networks.MiniMaxH3.network import MINIMAX_H3_FLOW_SHIFT, MiniMaxH3
from fastgen.configs.instantiate import LazyCall as L

MIN_T, MAX_T = 0.001, 0.999


def create_config():
    config = config_dmd2_default.create_config()

    # ---- precision / parallelism ----
    config.trainer.fsdp_cpu_offload = True
    config.trainer.fsdp = True
    config.model.meta_init = True
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # ---- networks: student, teacher, and the fake score built from the teacher config ----
    config.model.teacher = copy.deepcopy(MiniMaxH3_Config)
    config.model.net = copy.deepcopy(MiniMaxH3_Config)
    # Hold the per-block AdaLN projections fixed (13.0B of the transformer's 33.1B). The
    # fake score is instantiated from ``teacher``, so setting it there covers both.
    config.model.net.freeze_adaln = True
    config.model.teacher.freeze_adaln = True
    # Shard the conditioner root across ranks.
    config.model.extra_fsdp_modules += ["text_encoder"]

    # ---- input shape (video + audio; audio is inferred from the video shape) ----
    # Video latent ``[C, F, H, W]``; pixels = (H*16) x (W*16), frames = 17*n + 5 for
    # F = 5*n + 2. Audio is ``[32, 2, L]`` on the shared 40 Hz clock. Full length is the largest
    # ``17*n + 5`` whose duration stays within MiniMax-H3's 15 s ceiling: 345 frames (14.375 s)
    #   [24,  37, 34, 60]  ->  544 x 960,  124 frames ( 5.17 s)
    #   [24, 102, 34, 60]  ->  544 x 960,  345 frames (14.375 s, full length)
    #   [24,  37, 48, 84]  ->  768 x 1344, 124 frames ( 5.17 s,  native res)
    #   [24, 102, 48, 84]  ->  768 x 1344, 345 frames (14.375 s, native res, full length)
    config.model.input_shape = {"video": [24, 102, 48, 84], "audio": None}

    # ---- student: four steps on the release's shifted grid ----
    config.model.student_grid_cfg.steps = 4
    config.model.student_sample_type = "ode"
    # The student's grid, warped to the release's clock (shift 12) so its four steps sit at the
    # noise levels the sampler walks.
    config.model.student_grid_cfg.grid_kwargs = {"shift": MINIMAX_H3_FLOW_SHIFT}

    # The teacher/fake-score comparison time, drawn under a ``shifted`` warp. Shift 8 is a compromise between
    # the generic 5 and the release's sampling shift of 12; audio's noise level follows at 8 * 3/12 = 2.
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 8.0
    config.model.sample_t_cfg.min_t = MIN_T
    config.model.sample_t_cfg.max_t = MAX_T

    # ---- backward simulation + cross-iteration carry ----
    # ``backward_simulation`` makes the loader's prompts the whole batch, and with them the student's
    # inputs have to come from its own trajectory. The GAN loss is the one term that needs real data,
    # so it drops out on its own here; zeroing the weight also stops the discriminator being built.
    config.model.backward_simulation = "staggered"
    config.model.gan_loss_weight_gen = 0.0
    config.model.discriminator = None
    # With the discriminator gone, so is its optimizer -- drop the grad-clip callback keyed to it,
    # which would otherwise warn about the missing ``discriminator`` optimizer.
    config.trainer.callbacks.pop("grad_clip_discriminator", None)
    config.trainer.callbacks.carry = L(CarryCallback)()

    # ---- optimizers ----
    config.model.net_optimizer.lr = 3e-6
    config.model.fake_score_optimizer.lr = 3e-6
    config.trainer.callbacks.grad_clip.grad_norm = 1
    config.model.fake_score_pred_type = "x0"

    # ---- teacher guidance: none (guidance-distilled checkpoint) ----
    config.model.guidance_cfg = MiniMaxH3.default_guidance_cfg()

    # ---- dataloader (prompt loader; backward-simulation DMD2 needs only the prompts) ----
    # The loader is batch 1 per GPU already; raise the global batch with grad_accum.
    config.dataloader_train = copy.deepcopy(PromptLoaderConfig)
    # MiniMax-H3 is guidance-distilled, so a negative prompt would be encoded and then never used.
    config.dataloader_train.presets_map = {}

    # ---- trainer schedule ----
    config.trainer.max_iter = 1500
    config.trainer.logging_iter = 10
    config.trainer.save_ckpt_iter = 25
    config.trainer.callbacks.wandb.sample_logging_iter = 50
    config.trainer.callbacks.wandb.max_ranks = 8

    config.log_config.group = "minimax_h3_dmd2"

    return config
