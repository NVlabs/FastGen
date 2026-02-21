# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from fastgen.datasets.class_cond_dataloader import ImageLoader
from fastgen.datasets.wds_dataloaders import (
    WDSLoader,
    ImageWDSLoader,
    VideoWDSLoader,
)

from fastgen.utils import LazyCall as L

OUTPUT_ROOT = os.environ.get("FASTGEN_OUTPUT_ROOT", "FASTGEN_OUTPUT")
DATA_ROOT_DIR = os.getenv("DATA_ROOT_DIR", f"{OUTPUT_ROOT}/DATA")
S3_DATA_ROOT_DIR = os.getenv("DATA_ROOT_DIR", "s3://data")

# ################################################################################
# Generic Loaders (for config templates - override datatags for actual use)
# ################################################################################
# See fastgen/datasets/README.md for more details.

ImageLoaderConfig = L(ImageWDSLoader)(
    datatags=["WDS:/path/to/images"],
    batch_size=32,
    key_map={"real": "jpg", "condition": "txt"},
    presets_map={"neg_condition": "empty_string"},
    input_res=512,
)

ImageLatentLoaderConfig = L(WDSLoader)(
    datatags=["WDS:/path/to/image_latents"],
    batch_size=32,
    key_map={"real": "latent.pth", "condition": "txt_emb.pth"},
    # Negative condition embedding loaded from a shared file (same for all samples)
    files_map={"neg_condition": "/path/to/neg_prompt_emb.npy"},
)

VideoLoaderConfig = L(VideoWDSLoader)(
    datatags=["WDS:/path/to/videos"],
    batch_size=2,
    key_map={"real": "mp4", "condition": "txt"},
    presets_map={"neg_condition": "neg_prompt_wan"},
    sequence_length=81,
    img_size=(832, 480),
    num_workers=2,
)

VideoLatentLoaderConfig = L(WDSLoader)(
    datatags=["WDS:/path/to/video_latents"],
    batch_size=2,
    key_map={"real": "latent.pth", "condition": "txt_emb.pth"},
    # Negative condition embedding loaded from a shared file (same for all samples)
    files_map={"neg_condition": "/path/to/neg_prompt_emb.npy"},
    num_workers=2,
    # NOTE: For v2v tasks, add condition latent (e.g., depth) to key_map:
    #   key_map={"real": "latent.pth", "condition": "txt_emb.pth", "depth_latent": "depth_latent.pth"}
)

# ################################################################################
# Generic KD Loaders (for paired/path data)
# ################################################################################
# See fastgen/methods/knowledge_distillation/README.md for more details.

# For single-step KD: provides (real, noise, condition) pairs
# Data requirements: {"real": clean, "noise": noise, "condition": cond}
PairLoaderConfig = L(WDSLoader)(
    datatags=["WDS:/path/to/pairs"],
    batch_size=2,
    key_map={"real": "latent.pth", "noise": "noise.pth", "condition": "txt_emb.pth"},
    shuffle_size=100,
)

# For multi-step KD: provides (real, path, condition) with denoising trajectory
# Data requirements: {"real": clean, "path": [B, steps, C, ...], "condition": cond}
# path contains intermediate denoising steps (typically 4 steps)
PathLoaderConfig = L(WDSLoader)(
    datatags=["WDS:/path/to/paths"],
    batch_size=2,
    key_map={"real": "latent.pth", "path": "path.pth", "condition": "txt_emb.pth"},
    shuffle_size=100,
)

# ################################################################################
# Specific Datasets
# ################################################################################

CIFAR10_Loader_Config = L(ImageLoader)(
    dataset_path=f"{DATA_ROOT_DIR}/cifar10/cifar10-32x32.zip",
    s3_path=f"{S3_DATA_ROOT_DIR}/cifar10/cifar10-32x32.zip",
    use_labels=True,
    cache=True,
    batch_size=128,
    shuffle=True,
    sampler_start_idx=None,
)

ImageNet64_Loader_Config = L(ImageLoader)(
    dataset_path=f"{DATA_ROOT_DIR}/imagenet-64/imagenet-64x64.zip",
    s3_path=f"{S3_DATA_ROOT_DIR}/imagenet-64/imagenet-64x64.zip",
    use_labels=True,
    cache=True,
    batch_size=32,
    shuffle=True,
    sampler_start_idx=None,
)

ImageNet256_Loader_Config = L(ImageLoader)(
    dataset_path=f"{DATA_ROOT_DIR}/imagenet-256/imagenet_256_sd.zip",
    s3_path=f"{S3_DATA_ROOT_DIR}/imagenet-256/imagenet_256_sd.zip",
    use_labels=True,
    cache=True,
    batch_size=32,
    shuffle=True,
    sampler_start_idx=None,
)

ImageNet64_EDMV2_Loader_Config = L(ImageLoader)(
    dataset_path=f"{DATA_ROOT_DIR}/imagenet-64/imagenet-64x64-edmv2.zip",
    s3_path=f"{S3_DATA_ROOT_DIR}/imagenet-64/imagenet-64x64-edmv2.zip",
    use_labels=True,
    cache=True,
    batch_size=32,
    shuffle=True,
    sampler_start_idx=None,
)
