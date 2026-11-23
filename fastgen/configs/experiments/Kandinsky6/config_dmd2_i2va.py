# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kandinsky 6 Pro I2VA DMD2: the T2VA recipe conditioned on each clip's first frame, plus a GAN on the clips.

Uses the generic ``VideoAudioLoaderConfig``: point ``dataloader_train.datatags`` at WebDataset shards of
captioned clips with audio. The clips supply the image condition and the real video and audio the
discriminator sees; its heads, GAN weight and learning rate follow LTX-2's DMD2 recipe.
"""

import copy

import fastgen.configs.experiments.Kandinsky6.config_dmd2_t2va as config_t2va
from fastgen.callbacks.grad_clip import GradClipCallback
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.configs.data import VideoAudioLoaderConfig
from fastgen.configs.discriminator import Discriminator_Kandinsky6_Multimodal_Config
from fastgen.configs.instantiate import LazyCall as L
from fastgen.networks.Kandinsky6.network import Kandinsky6


def create_config():
    config = config_t2va.create_config()
    config.model.net.image_conditioned = True

    config.model.discriminator = copy.deepcopy(Discriminator_Kandinsky6_Multimodal_Config)
    config.model.discriminator.modality_discs.video.disc_type = "multiscale_down_mlp_medium"
    # Halfway, three quarters and the last of the 60 fused blocks, like LTX-2's 24/36/47 of 48.
    config.model.discriminator.feature_indices = [30, 45, 59]
    for modality_disc in config.model.discriminator.modality_discs.values():
        modality_disc.feature_indices = list(config.model.discriminator.feature_indices)
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.discriminator_optimizer.lr = 5e-6
    config.trainer.callbacks.grad_clip_discriminator = L(GradClipCallback)(grad_norm=None, model_key="discriminator")

    config.dataloader_train = copy.deepcopy(VideoAudioLoaderConfig)
    config.dataloader_train.batch_size = 1
    num_frames, height, width = model_pixel_shape(config.model)
    config.dataloader_train.sequence_length = num_frames
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.target_fps = config.model.net.fps
    config.dataloader_train.target_sample_rate = Kandinsky6.audio_sampling_rate
    config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_kandinsky6"}

    # One clip per GPU on eight GPUs: under FSDP, gradient accumulation keeps the 30B gradients unsharded.
    config.trainer.batch_size_global = 8
    config.log_config.group = "kandinsky6_i2va_dmd2"
    return config
