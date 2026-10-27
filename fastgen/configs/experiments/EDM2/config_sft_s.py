# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from omegaconf import DictConfig

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageNet64_EDMV2_Loader_Config
from fastgen.configs.net import EDM2_IN64_S_Config, CKPT_ROOT_DIR
from fastgen.configs.callbacks import EMA_POWER_CALLBACKS, ForcedWeightNorm_CALLBACK
from fastgen.networks.EDM2.network import EDM2Precond

"""Configs for SFT (Supervised Fine-Tuning) on EDM2-S and ImageNet-64 dataset."""


def create_config():
    config = config_sft_default.create_config()

    # model
    # EDM2 works on raw pixel space: [C, H, W] = [3, 64, 64]
    config.model.input_shape = [3, 64, 64]
    # EDM2's FP16 mode: the network casts its input and weights to FP16, without autocast.
    config.model.precision_model = "float16"
    config.model.precision_opt = "float32"
    config.model.grad_scaler_enabled = True
    config.model.grad_scaler_growth_interval = 20000

    # EDM2 uses sigma-based noise schedule (log-normal for training)
    config.model.sample_t_cfg.time_dist_type = "lognormal"
    config.model.sample_t_cfg.train_p_mean = -0.8
    config.model.sample_t_cfg.train_p_std = 1.6
    # Note: EDM2 uses sigma, not t. The sampler handles the conversion.

    # Pretrained model path
    # EDM2-S ImageNet64 checkpoint from https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm2-img64-s-fid.pth"

    # Network config
    config.model.net = copy.deepcopy(EDM2_IN64_S_Config)
    config.model.guidance_cfg = EDM2Precond.default_guidance_cfg()

    # Optimizer config (lower LR for fine-tuning)
    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 1e-4  # Fine-tuning; the official ImageNet64-S scratch preset uses 1e-2.
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.weight_decay = 0.0

    # EMA config (EDM2 uses power-function EMA)
    config.model.use_ema = ["ema_1", "ema_5", "ema_10"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_POWER_CALLBACKS)

    # Important!!! EDM2 needs the ForcedWeightNorm callback - must be added AFTER recreating callbacks
    config.trainer.callbacks.update(ForcedWeightNorm_CALLBACK)

    # Dataloader (using EDMv2 preprocessed data)
    config.dataloader_train = copy.deepcopy(ImageNet64_EDMV2_Loader_Config)
    config.dataloader_train.batch_size = 64

    # Sampling config for visualization
    config.model.student_grid_cfg.steps = 32  # EDM2 typically uses 32-64 steps

    # Trainer
    # EDM2 disables TF32 and reduced-precision reductions.
    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False
    config.trainer.batch_size_global = 1024
    config.trainer.max_iter = 150000
    config.trainer.save_ckpt_iter = 10000
    config.trainer.logging_iter = 1000

    config.log_config.group = "edm2_s_imagenet64_sft"

    return config
