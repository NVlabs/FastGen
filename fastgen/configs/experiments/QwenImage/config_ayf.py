# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_ayf as config_ayf_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import QwenImageConfig
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.QwenImage.network import QwenImage

"""Align Your Flow (AYF) LoRA distillation on Qwen-Image (DDP)."""


def create_config():
    config = config_ayf_default.create_config()

    # Network: Qwen-Image with a second-timestep (r) embedder (zero-init, "diff" time conditioning by default) and a guidance
    # embedder (Qwen-Image is not guidance-distilled so the student learns
    # CFG-scale conditioning from scratch).
    config.model.net = copy.deepcopy(QwenImageConfig)
    # Replacing the net above wipes the method-config default; re-apply.
    config.model.net.r_timestep = True
    config.model.net.guidance_embeds = True
    # Default embedded guidance for sampling (training passes per-sample values).
    config.model.net.guidance_scale = 4.0

    # LoRA student (default rank 64, default Qwen attention/MLP targets) on the frozen base.
    config.model.net_adapter.type = "lora"
    config.model.net_adapter.alpha = 512
    config.model.net_adapter.extra_trainable_modules = ["r_embedder", "guidance_embedder"]
    # The embedded-guidance default is student-only; the base model samples with true CFG.
    config.model.net_adapter.base_model_overrides = {"guidance_scale": None}

    # The teacher runs true CFG with the per-sample scale from
    # ``teacher_guidance_range``, composed via this recipe (keeps the per-token
    # norm rescale); the batch must carry neg_condition.
    config.model.guidance_cfg = QwenImage.default_guidance_cfg()
    config.model.teacher_guidance_type = "cfg"
    config.model.teacher_guidance_range = (3.0, 5.0)

    # Warmup stage: guided-velocity distillation only (r = t, no JVP) so the fresh
    # guidance/r embedders settle before flow-map training starts.
    config.model.guidance_distill_warmup_steps = 5000

    # LoRA (run with DDP; teacher = same net, adapters off): bf16 frozen base, fp32
    # LoRA masters (bf16 masters stall at lr ~1e-6), bf16 autocast over the mixed dtypes.
    config.trainer.ddp = True
    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"
    config.model.precision_amp = "bfloat16"
    config.model.precision_amp_infer = "bfloat16"
    config.model.precision_amp_jvp = "float32"

    # Qwen-Image latent shape: 256x256 -> [16, 32, 32] (512px: [16, 64, 64]).
    config.model.input_shape = [16, 32, 32]

    # Few-step flow-map.
    config.model.student_grid_cfg.steps = 4
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 1.25
    # Keep the 0.999 student-grid top of the existing AYF checkpoints; Qwen-Image's default 1.0 bound serves the
    # base sampler grid. The adapters-off teacher shares this schedule.
    config.model.net.max_t = 0.999

    # Optimizer.
    config.model.net_optimizer.lr = 1e-6
    config.model.net_optimizer.weight_decay = 1e-4

    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    config.dataloader_train.batch_size = 2
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    # Training iterations.
    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 1
    config.trainer.callbacks.wandb.sample_logging_iter = 100
    config.trainer.save_ckpt_iter = 500
    # LoRA-recipe checkpoints: save only the adapter tensors (LoRA + embedders).
    config.trainer.checkpointer.save_adapter_only = True
    # The base net loads its own pretrained weights (no teacher-checkpoint init).
    config.model.load_student_weights = False

    config.log_config.group = "qwen_image_ayf"

    return config
