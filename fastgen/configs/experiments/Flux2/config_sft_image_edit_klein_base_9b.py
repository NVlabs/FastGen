# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageEditLoaderConfig
from fastgen.configs.net import Flux2KleinBase9BImageEditConfig
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.Flux2.network import Flux2

"""SFT (instruction-tuned image editing) config for FLUX.2-klein-base-9B (FLUX NC v2.1)."""


def create_config():
    config = config_sft_default.create_config()

    # bf16 compute, fp32 FSDP storage prevents drift over long SFT runs.
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    # Raw VAE-output latent shape for 1024x1024 inputs (pre-pack). The
    # source-image reference latent is also at this shape and uses the
    # same packing path -- doubles the transformer token count vs. plain
    # T2I (4096 -> 8192 tokens at 1024 px).
    config.model.input_shape = [32, 128, 128]

    # Image-edit klein-base-9B (FLUX NC v2.1, undistilled). The prebuilt
    # config pins ``model_id_or_local_path``, ``guidance_scale=None`` (klein-base was
    # not trained with embedded guidance distillation), and
    # ``enable_image_edit=True`` (turn on the reference-image path in
    # ``Flux2.forward``).
    config.model.net = copy.deepcopy(Flux2KleinBase9BImageEditConfig)
    config.trainer.fsdp = True
    config.model.meta_init = True

    # Optimizer
    config.model.net_optimizer.lr = 1e-5

    # Rectified-flow timestep sampler biased toward higher noise to match
    # the FLUX.2 inference schedule.
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 3
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Sample-time CFG used by the visualization sampler.
    config.model.guidance_cfg = Flux2.default_guidance_cfg()
    config.model.student_grid_cfg.steps = 50  # klein-base default

    # Image-editing dataset.
    config.dataloader_train = copy.deepcopy(ImageEditLoaderConfig)
    config.dataloader_train.batch_size = 1
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)

    # Conditional dropout: replace the text embedding with the negative
    # (empty-string) embedding for `cond_dropout_prob` of training steps,
    # so the trained student supports classifier-free guidance at sample
    # time. We do NOT drop the reference image (`cond_keys_no_dropout`)
    # because the editing task is fundamentally conditioned on it.
    config.model.cond_dropout_prob = 0.1
    config.model.cond_keys_no_dropout = ["condition_image"]

    # Trainer
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "flux2_klein_base_9b_image_edit_sft"

    return config
