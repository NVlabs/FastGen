# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.discriminator import Discriminator_Flux2KleinBase9B_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import Flux2KleinBase9BConfig
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.Flux2.network import Flux2

"""DMD2 distillation config for FLUX.2-klein-base-9B (FLUX NC v2.1)."""


def create_config():
    config = config_dmd2_default.create_config()

    # Optimizer settings (same LR for all three nets for stability).
    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    # Raw VAE-output latent shape for 1024x1024 inputs.
    # For 512x512 inputs use [32, 64, 64].
    config.model.input_shape = [32, 128, 128]

    # ImageDiT discriminator sized for klein-base-9B's transformer:
    # 8 joint + 24 single = 32 blocks total, hidden_dim = 4096.
    config.model.discriminator = copy.deepcopy(Discriminator_Flux2KleinBase9B_Config)

    # Late-block feature taps for the GAN, proportional to dev's
    # [28, 42, 55] (which targets blocks at ~50% / ~75% / ~100% of the
    # 56-block dev transformer). For klein-9B (32 blocks, indices 0..31)
    # the same proportions land on [16, 24, 31]. Drop the final block
    # to [16, 24] if you hit memory pressure.
    config.model.discriminator.feature_indices = [16, 24, 31]

    # GAN settings (same as dev recipe -- not architecture-dependent).
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"

    # FLUX.2-klein-base-9B (undistilled, FLUX NC v2.1). The prebuilt config
    # already pins `model_id_or_local_path` and `guidance_scale=None` (no embedded
    # guidance vector -- klein-base was trained without guidance
    # distillation, so feeding it an embedded scale would corrupt the
    # time embedding).
    config.model.net = copy.deepcopy(Flux2KleinBase9BConfig)
    config.trainer.fsdp = True
    config.model.meta_init = True

    # Teacher CFG for the DMD2 distribution-matching target (also used for teacher
    # samples). Matches BFL's reference call for klein-base-9B:
    #   `Flux2KleinPipeline(prompt, guidance_scale=4.0, num_inference_steps=50)`
    # Note: this is distinct from `model.net.guidance_scale` (which is
    # the embedded guidance and stays None for klein-base).
    config.model.guidance_cfg = Flux2.default_guidance_cfg()

    # Precision: bf16 compute, fp32 FSDP storage for stable distillation.
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # Student sampling: 4-step ODE rollout (DMD2 distillation target).
    config.model.student_grid_cfg.steps = 4
    config.model.student_sample_type = "ode"

    # Time sampling: rectified-flow `shifted` distribution biases t toward
    # higher noise (matches FLUX.2 inference better than plain uniform).
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 3
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Approximate BFL's empirical-mu grid at 1024px / 4 steps. The same grid
    # works for klein-base-9B because `compute_empirical_mu`
    # depends on `image_seq_len` (= 4096 at 1024px), not on the model
    # variant.
    config.model.student_grid_cfg.grid_kwargs = {"shift": 9.9}

    # Dataloader: raw-image dataset (point datatags at your shards).
    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    config.dataloader_train.batch_size = 1
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    # Training iterations
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    # Logging
    config.log_config.group = "flux2_klein_base_9b_dmd2"

    return config
