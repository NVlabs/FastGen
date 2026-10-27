# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from omegaconf import DictConfig
import fastgen.configs.methods.config_scm as config_scm_default
from fastgen.configs.data import ImageNet64_EDMV2_Loader_Config
from fastgen.configs.net import EDM2_IN64_XL_Config, CKPT_ROOT_DIR
from fastgen.configs.callbacks import ForcedWeightNorm_CALLBACK, EMA_POWER_CALLBACKS
from fastgen.configs.instantiate import LazyCall as L
from fastgen.utils.lr_scheduler import LambdaInverseSquareRootScheduler

""" Configs for the sCM model, on EDM2-XL and ImageNet-64 dataset. """


def create_config():
    config = config_scm_default.create_config()

    # ema
    config.model.use_ema = ["ema_1", "ema_5", "ema_10"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_POWER_CALLBACKS)

    # EDM2's FP16 mode: the network casts its input and weights to FP16, without autocast.
    config.model.precision_model = "float16"
    config.model.precision_opt = "float32"
    config.model.precision_amp_jvp = "float32"
    config.model.grad_scaler_enabled = True
    config.model.grad_scaler_init_scale = 4096
    config.model.grad_scaler_growth_interval = 20000
    config.model.input_shape = [3, 64, 64]

    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 7e-5
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.eps = 1e-11
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler = L(LambdaInverseSquareRootScheduler)(
        warm_up_steps=0,
        decay_steps=35000,
    )

    # path to the pretrained diffusion model ckpt
    # Note: The Fourier-based embedding of EDM2 leads to instabilities in training!
    # While one can try to distill the pretrained model with a smaller bandwidth,
    # we recommend to follow the paper and pretrain a new model with TrigFlow noise schedule.
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm2-img64-xl-fid.pth"
    config.model.net = copy.deepcopy(EDM2_IN64_XL_Config)
    config.model.net.dropout = 0.45  # set dropout=0.45 for resolutions <= 16 only
    config.model.net.dropout_resolutions = [16, 8]
    config.model.net.embedding_type = "mp_fourier"
    config.model.net.mp_fourier_bandwidth = 1.0

    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003

    # Important!!! EDM2 needs the ForcedWeightNorm callback
    config.trainer.callbacks.update(ForcedWeightNorm_CALLBACK)

    config.model.sample_t_cfg.train_p_mean = -1.0
    config.model.sample_t_cfg.train_p_std = 1.6

    config.dataloader_train = copy.deepcopy(ImageNet64_EDMV2_Loader_Config)

    config.log_config.group = "edm2_imagenet64_scm"
    return config
