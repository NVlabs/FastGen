# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig
import copy

from fastgen.configs.discriminator import Discriminator_EDM_ImageNet64_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import ImageNet64_Loader_Config
from fastgen.configs.net import EDM_ImageNet64_Config, CKPT_ROOT_DIR
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS

""" Configs for the DMD2 model, on EDM and ImageNet-64 dataset. """


def create_config():
    config = config_dmd2_default.create_config()

    config.model.net_optimizer.lr = 2e-6
    config.model.discriminator_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.lr = 2e-6

    config.model.input_shape = [3, 64, 64]
    # EDM's FP16 mode: the network casts its input and weights to FP16, without autocast.
    config.model.precision_model = "float16"
    config.model.precision_opt = "float32"
    config.model.grad_scaler_enabled = True
    config.model.grad_scaler_init_scale = 32768
    config.model.grad_scaler_growth_interval = 20000
    config.model.discriminator = copy.deepcopy(Discriminator_EDM_ImageNet64_Config)
    config.model.gan_loss_weight_gen = 3e-3
    # path to the pretrained diffusion model ckpt
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm-imagenet-64x64-cond-adm.pth"
    config.model.net = copy.deepcopy(EDM_ImageNet64_Config)
    # Polynomial draws within entries 2 and 998 of EDM's 1000-entry polynomial table (rho = 7).
    config.model.sample_t_cfg.min_t, config.model.sample_t_cfg.max_t = 0.00210147, 79.5638

    # ema
    config.model.use_ema = ["ema_9999", "ema_99995", "ema_9996"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    config.dataloader_train = copy.deepcopy(ImageNet64_Loader_Config)

    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False
    config.trainer.batch_size_global = 512
    config.trainer.max_iter = 600000
    config.log_config.group = "edm_imagenet64_dmd2"
    return config
