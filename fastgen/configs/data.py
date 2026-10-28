# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataloader configs.

The CIFAR-10 and ImageNet loaders read the ZIPs that ``scripts/download_data.py`` prepares under ``DATA_ROOT_DIR``. The
generic WebDataset loaders are templates: a run overrides their placeholder ``datatags`` and, for the latent loaders,
``files_map["neg_condition"]``, the precomputed negative-prompt embedding.
"""

import os

from fastgen.datasets.class_cond_dataloader import ImageLoader
from fastgen.datasets.wds_dataloaders import (
    WDSLoader,
    ImageWDSLoader,
    VideoWDSLoader,
    AudioVideoWDSLoader,
    CameraVideoWDSLoader,
)

from fastgen.configs.instantiate import LazyCall as L
from fastgen.utils.env import output_root

DATA_ROOT_DIR = os.getenv("DATA_ROOT_DIR", f"{output_root()}/DATA")
# Optional S3 or HTTP(S) mirror of DATA_ROOT_DIR; a local dataset ZIP that is missing is downloaded from it.
REMOTE_DATA_ROOT_DIR = os.getenv("REMOTE_DATA_ROOT_DIR")


def _remote_dataset_path(relative_path: str) -> str | None:
    """The mirrored copy of a dataset under ``REMOTE_DATA_ROOT_DIR``, or None without a mirror."""
    return f"{REMOTE_DATA_ROOT_DIR}/{relative_path}" if REMOTE_DATA_ROOT_DIR else None


# ################################################################################
# Generic Loaders (for config templates - override datatags for actual use)
# ################################################################################
# See fastgen/datasets/README.md for more details.
# sampler_start_idx=None lets the trainer resume each training loader at the next unseen sample.

ImageLoaderConfig = L(ImageWDSLoader)(
    datatags=["WDS:/path/to/images"],
    batch_size=32,
    key_map={"real_raw": "jpg", "condition_raw": "txt"},
    presets_map={"neg_condition_raw": "empty_string"},
    input_res=512,
    sampler_start_idx=None,
)

ImageLatentLoaderConfig = L(WDSLoader)(
    datatags=["WDS:/path/to/image_latents"],
    batch_size=32,
    key_map={"real": "latent.pth", "condition": "txt_emb.pth"},
    # Negative condition embedding loaded from a shared file (same for all samples)
    files_map={"neg_condition": "/path/to/neg_prompt_emb.npy"},
    sampler_start_idx=None,
)

PromptLoaderConfig = L(WDSLoader)(
    datatags=["WDS:/path/to/prompts"],
    batch_size=1,
    key_map={"condition_raw": "txt"},
    presets_map={"neg_condition_raw": "empty_string"},
    num_workers=2,
    sampler_start_idx=None,
)

# Default key_map for image-editing datasets: `(edited image, source image,
# instruction)` triplets. Shared by the generic ImageEditLoaderConfig below.
IMAGE_EDIT_KEY_MAP = {
    "real_raw": "edited.jpg",
    "condition_image_raw": "source.jpg",
    "condition_raw": "txt",
}

# Generic image-editing loader. Drop-in compatible with the FLUX.2 image-edit
# configs; point `datatags` at your own image-edit WebDataset shards.
ImageEditLoaderConfig = L(ImageWDSLoader)(
    datatags=["WDS:/path/to/image_edit_shards"],
    batch_size=8,
    key_map=IMAGE_EDIT_KEY_MAP,
    presets_map={"neg_condition_raw": "empty_string"},
    input_res=1024,
    sampler_start_idx=None,
)

# Raw-video templates: set target_fps to the network's frame rate (config.model.net.fps); see datasets/README.md.
VideoLoaderConfig = L(VideoWDSLoader)(
    datatags=["WDS:/path/to/videos"],
    batch_size=2,
    key_map={"real_raw": "mp4", "condition_raw": "txt"},
    presets_map={"neg_condition_raw": "neg_prompt_wan"},
    sequence_length=81,
    img_size=(832, 480),
    num_workers=2,
    sampler_start_idx=None,
)

VideoAudioLoaderConfig = L(AudioVideoWDSLoader)(
    datatags=["WDS:/path/to/videos_with_audio"],
    batch_size=1,
    key_map={"real_raw": "mp4", "condition_raw": "txt"},
    presets_map={"neg_condition_raw": "empty_string"},
    sequence_length=81,
    img_size=(960, 544),
    num_workers=2,
    target_sample_rate=16000,
    sampler_start_idx=None,
)

# Video loader with pre-computed camera poses/intrinsics (e.g. from VIPE),
# for camera-conditioned models like WanCamI2V.
CameraVideoLoaderConfig = L(CameraVideoWDSLoader)(
    datatags=["WDS:/path/to/videos_with_vipe"],
    batch_size=2,
    key_map={
        "real_raw": "mp4",
        "condition_raw": "txt",
        "camera_poses": "poses.npy",
        "camera_intrinsics": "intrinsics.npy",
    },
    presets_map={"neg_condition_raw": "neg_prompt_wan"},
    sequence_length=81,
    img_size=(832, 480),
    num_workers=2,
    sampler_start_idx=None,
)

VideoLatentLoaderConfig = L(WDSLoader)(
    datatags=["WDS:/path/to/video_latents"],
    batch_size=2,
    key_map={"real": "latent.pth", "condition": "txt_emb.pth"},
    # Negative condition embedding loaded from a shared file (same for all samples)
    files_map={"neg_condition": "/path/to/neg_prompt_emb.npy"},
    num_workers=2,
    sampler_start_idx=None,
)

# ################################################################################
# Generic KD Loaders (for paired/path data)
# ################################################################################
# See fastgen/methods/trajectory_matching/README.md#kd-knowledge-distillation for more details.

# For single-step KD: provides (real, noise, condition) pairs
# Data requirements: {"real": clean, "noise": noise, "condition": cond}
PairLoaderConfig = L(WDSLoader)(
    datatags=["WDS:/path/to/pairs"],
    batch_size=2,
    key_map={"real": "latent.pth", "noise": "noise.pth", "condition": "txt_emb.pth"},
    shuffle_size=100,
    sampler_start_idx=None,
)

# For multi-step KD: provides (real, path, condition) with denoising trajectory
# Data requirements: {"real": clean, "path": [B, steps, C, ...], "condition": cond}
# path contains one non-clean state per base-grid interval, so KD needs an anchored student grid: a native grid
# with grid_kwargs.base_steps equal to the path length, or explicit timesteps
PathLoaderConfig = L(WDSLoader)(
    datatags=["WDS:/path/to/paths"],
    batch_size=2,
    key_map={"real": "latent.pth", "path": "path.pth", "condition": "txt_emb.pth"},
    shuffle_size=100,
    sampler_start_idx=None,
)

# ################################################################################
# Specific Datasets
# ################################################################################

CIFAR10_Loader_Config = L(ImageLoader)(
    dataset_path=f"{DATA_ROOT_DIR}/cifar10/cifar10-32x32.zip",
    remote_path=_remote_dataset_path("cifar10/cifar10-32x32.zip"),
    use_labels=True,
    cache=True,
    batch_size=128,
    shuffle=True,
    sampler_start_idx=None,
)

ImageNet64_Loader_Config = L(ImageLoader)(
    dataset_path=f"{DATA_ROOT_DIR}/imagenet-64/imagenet-64x64.zip",
    remote_path=_remote_dataset_path("imagenet-64/imagenet-64x64.zip"),
    use_labels=True,
    cache=True,
    batch_size=32,
    shuffle=True,
    sampler_start_idx=None,
)

ImageNet256_Loader_Config = L(ImageLoader)(
    dataset_path=f"{DATA_ROOT_DIR}/imagenet-256/imagenet_256_sd.zip",
    remote_path=_remote_dataset_path("imagenet-256/imagenet_256_sd.zip"),
    use_labels=True,
    cache=True,
    batch_size=32,
    shuffle=True,
    sampler_start_idx=None,
)

ImageNet64_EDMV2_Loader_Config = L(ImageLoader)(
    dataset_path=f"{DATA_ROOT_DIR}/imagenet-64/imagenet-64x64-edmv2.zip",
    remote_path=_remote_dataset_path("imagenet-64/imagenet-64x64-edmv2.zip"),
    use_labels=True,
    cache=True,
    batch_size=32,
    shuffle=True,
    sampler_start_idx=None,
)
