# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.discriminator import Discriminator_QwenImage_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import HiDreamG5_JourneyDB_Loader_Config
from fastgen.configs.net import QwenImageConfig

"""Configs for DMD2 distillation on Qwen-Image model."""


def create_config():
    config = config_dmd2_default.create_config()

    # Optimizer settings
    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    # QwenImage latent shape: [C, H, W] = [16, H//8, W//8]
    # For 512x512 images: [16, 64, 64]
    # For 1024x1024 images: [16, 128, 128]
    config.model.input_shape = [16, 64, 64]

    # Discriminator config - ImageDiT with simple conv2d architecture
    config.model.discriminator = Discriminator_QwenImage_Config

    # GAN settings
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"

    # We can further set GAN discriminator feature indices to control which
    # intermediate features to use for discrimination. By default, we use the middle block
    # config.model.discriminator.feature_indices=[30, 45] #use the 30th and 45th blocks from the DiT

    # Network config
    config.model.net = QwenImageConfig

    # CFG guidance for the teacher during distillation
    config.model.guidance_scale = 4.0

    # Meta init required for QwenImage (~20B params) — only rank 0 loads weights,
    # others use meta device to avoid OOM from 3 copies (student+teacher+fake_score)
    config.model.fsdp_meta_init = True

    # Precision
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"

    # Student sampling steps
    config.model.student_sample_steps = 4

    # Time sampling config
    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    # we can further set the time list to control the sampled timesteps for the student, to focus more on
    # specific noise levels
    # config.model.sample_t_cfg.t_list = [0.999,t1,t2,0]

    # Dataloader
    config.dataloader_train = HiDreamG5_JourneyDB_Loader_Config
    config.dataloader_train.batch_size = 2
    config.dataloader_train.input_res = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)

    # Training iterations
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    # FSDP CPU offload
    config.trainer.fsdp_cpu_offload = True

    # Logging
    config.log_config.group = "qwen_image_dmd2"

    return config
