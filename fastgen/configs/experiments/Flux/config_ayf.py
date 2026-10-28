# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_ayf as config_ayf_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import FluxConfig
from fastgen.configs.config_utils import model_pixel_shape

"""Align Your Flow (AYF) LoRA distillation on Flux.1-dev (DDP)."""


def create_config():
    config = config_ayf_default.create_config()

    # Network: Flux.1-dev with a second-timestep (r) embedder (zero-init, "diff" time conditioning by default).
    # Flux.1-dev is guidance-distilled, so the teacher uses the embedded scale
    # (model.guidance_cfg stays at its empty default).
    config.model.net = copy.deepcopy(FluxConfig)
    # Replacing the net above wipes the method-config default; re-apply.
    config.model.net.r_timestep = True
    # T5 text length 100: Flux attends to pad tokens, so a longer padded length
    # would change the conditioning as well as the compute.
    config.model.net.max_sequence_length = 100

    # LoRA student (default rank 64) on the frozen base (the teacher is the same network with
    # adapters disabled). Adapt every Linear except the new r_embedder.
    config.model.net_adapter.type = "lora"
    config.model.net_adapter.alpha = 512
    config.model.net_adapter.target_modules = "all-linear"
    config.model.net_adapter.extra_trainable_modules = ["r_embedder"]

    # LoRA (run with DDP): bf16 frozen base, fp32 LoRA masters (bf16 masters
    # stall at lr ~1e-6), bf16 autocast over the mixed dtypes.
    config.trainer.ddp = True
    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"
    config.model.precision_amp = "bfloat16"
    config.model.precision_amp_infer = "bfloat16"
    config.model.precision_amp_jvp = "float32"

    # Activations fit at this size; recompute would double the backward cost.
    config.model.net.disable_grad_ckpt = True

    # Guidance-interval distillation: one embedded guidance scale per sample in
    # [1.5, 5.0], shared by the teacher, student, and JVP forwards.
    config.model.teacher_guidance_range = (1.5, 5.0)

    # Flux latent shape: 256x256 -> [16, 32, 32] (512px: [16, 64, 64], shift 2.0).
    config.model.input_shape = [16, 32, 32]

    # Few-step flow-map: t, r drawn i.i.d. shifted-uniform (r from the same distribution as t),
    # ordered t >= r, no flow-matching anchor. shift = sqrt(packed_tokens / 256) -> 1.0 at 256px.
    config.model.student_grid_cfg.steps = 4
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 1.0
    # Keep the 0.999 student-grid top of the existing AYF checkpoints; Flux's default 1.0 bound serves the base
    # sampler grid. The adapters-off teacher shares this schedule.
    config.model.net.max_t = 0.999

    # Optimizer: AdamW, constant lr (no warmup).
    config.model.net_optimizer.lr = 1e-6
    config.model.net_optimizer.weight_decay = 1e-4

    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    config.dataloader_train.batch_size = 2
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)
    config.trainer.batch_size_global = 16

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 1
    config.trainer.callbacks.wandb.sample_logging_iter = 100
    config.trainer.save_ckpt_iter = 500
    # LoRA-recipe checkpoints: save only the adapter tensors (LoRA + embedders).
    config.trainer.checkpointer.save_adapter_only = True
    # The base net loads its own pretrained weights (no teacher-checkpoint init).
    config.model.load_student_weights = False

    config.log_config.group = "flux_ayf"

    return config
