# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from omegaconf import DictConfig

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageNet256_Loader_Config
from fastgen.configs.net import SIT_XL_2_256_URL, SiT_XL_2_256_Config
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS
from fastgen.networks.DiT.network import DiT

"""Configs for SFT (Supervised Fine-Tuning) on SiT-XL and ImageNet-256 dataset.

SiT (Scalable Interpolant Transformers) uses the same architecture as DiT but
is trained with flow matching (rectified flow) instead of DDPM.

Reference: https://github.com/willisma/SiT
"""


def create_config():
    config = config_sft_default.create_config()

    # model
    # SiT latent shape: [C, H, W] = [4, 32, 32] for 256x256 images (256/8 = 32)
    config.model.input_shape = [4, 32, 32]
    config.model.precision_amp = "bfloat16"
    config.model.precision_amp_infer = "bfloat16"
    config.model.cond_dropout_prob = 0.1  # 10% dropout for CFG training

    # Timestep sampling config for flow matching
    # SiT uses uniform time sampling during training
    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Pretrained model path
    # SiT-XL/2 ImageNet-256 checkpoint from https://github.com/willisma/SiT
    config.model.pretrained_model_path = SIT_XL_2_256_URL

    # Network config: the SiT-XL/2 release (DiT architecture, flow matching, SiT time convention, 3-channel CFG)
    config.model.net = copy.deepcopy(SiT_XL_2_256_Config)
    config.model.guidance_cfg = DiT.default_guidance_cfg()
    config.model.student_grid_cfg.steps = 250  # SiT uses 250 SDE steps
    config.model.student_sample_type = "sde"  # SiT's own Euler-Maruyama sampler

    # Optimizer config (lower LR for fine-tuning; 1e-4 is for training from scratch)
    config.model.net_optimizer.optim_type = "adamw"
    config.model.net_optimizer.lr = 1e-5  # 10x lower for fine-tuning
    config.model.net_optimizer.betas = (0.9, 0.95)
    config.model.net_optimizer.weight_decay = 0.0

    # EMA config
    config.model.use_ema = ["ema_9999", "ema_99995"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update({k: EMA_CONST_CALLBACKS[k] for k in config.model.use_ema})

    # Dataloader
    config.dataloader_train = copy.deepcopy(ImageNet256_Loader_Config)

    # Trainer
    config.trainer.batch_size_global = 256
    config.trainer.max_iter = 400000
    config.trainer.save_ckpt_iter = 10000

    config.log_config.group = "sit_xl_imagenet256_sft"

    return config
