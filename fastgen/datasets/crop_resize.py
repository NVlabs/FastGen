# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as transforms_F


# ------------------------------------------------------------------------
# normalize


def normalize_pixels(
    data: Any,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Pixels -> float tensor via per-channel ``(x - mean) / std`` (default ``[-1, 1]``).

    Input is first mapped to ``[0, 1]`` (``PIL`` via ``to_tensor``; ``ndarray`` /
    ``Tensor`` in ``[0, 255]`` via ``/255``). ``mean``/``std`` broadcast over the
    channel dim (``-3``), so it handles image ``[C,H,W]`` / ``[B,C,H,W]`` and video
    ``[T,C,H,W]``. Default ``0.5`` == ``x/127.5 - 1``; shared by the loaders + inference.
    """
    if isinstance(data, Image.Image):
        tensor = transforms_F.to_tensor(data)
    else:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        tensor = data.float() / 255.0
    mean_t = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_t = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


def obtain_image_size(data: Any) -> tuple[int, int]:
    """Function for obtaining the image size from the data.

    Args:
        data: Input data
    Returns:
        width (int): Width of the input image
        height (int): Height of the input image
    """

    if isinstance(data, Image.Image):
        width, height = data.size
    elif isinstance(data, torch.Tensor):
        height, width = data.size()[-2:]
    else:
        raise ValueError("data to crop or resize should be PIL Image or tensor")

    return width, height


# ------------------------------------------------------------------------
# crop


def center_crop_offsets(outer_w: int, outer_h: int, img_size: tuple[int, int]) -> tuple[int, int]:
    """Effective top-left ``(crop_x0, crop_y0)`` of a torchvision-style center crop.

    Half-pixel crop ties use Python's round-to-even rule, matching
    ``torchvision.transforms.functional.center_crop``. Negative offsets encode
    left/top padding when the requested crop is larger than the input.

    Args:
        outer_w: Width of the image being cropped.
        outer_h: Height of the image being cropped.
        img_size: Target crop size as ``(width, height)``.
    """
    img_w, img_h = img_size

    def offset(outer: int, crop: int) -> int:
        if crop > outer:
            return -((crop - outer) // 2)
        return int(round((outer - crop) / 2.0))

    return offset(outer_w, img_w), offset(outer_h, img_h)


def center_crop(data: Any, img_size: tuple[int, int], return_cropping_params: bool = False) -> Any:
    """Performs center crop."""
    img_w, img_h = img_size

    orig_w, orig_h = obtain_image_size(data)
    # Apply the recorded offsets explicitly so pixels and camera intrinsics use identical geometry.
    crop_x0, crop_y0 = center_crop_offsets(orig_w, orig_h, img_size)
    data = transforms_F.crop(data, crop_y0, crop_x0, img_h, img_w)
    cropping_params = {
        "resize_w": orig_w,
        "resize_h": orig_h,
        "crop_x0": crop_x0,
        "crop_y0": crop_y0,
        "crop_w": img_w,
        "crop_h": img_h,
    }

    if return_cropping_params:
        return data, cropping_params
    return data


# ------------------------------------------------------------------------
# resize


def aspect_preserving_resize_size(orig_w: int, orig_h: int, img_size: tuple[int, int]) -> tuple[int, int]:
    """Aspect-preserving resize target as ``(resize_w, resize_h)``.

    Scales ``(orig_w, orig_h)`` by the larger of the two target ratios, so the result
    covers ``img_size = (width, height)`` (the small-side fit used before a center crop).

    Args:
        orig_w: Native source width.
        orig_h: Native source height.
        img_size: Target size as ``(width, height)``.
    """
    img_w, img_h = img_size
    scaling_ratio = max(img_w / orig_w, img_h / orig_h)
    resize_w = int(scaling_ratio * orig_w + 0.5)
    resize_h = int(scaling_ratio * orig_h + 0.5)
    return resize_w, resize_h


def resize_small_side_aspect_preserving(data: Any, img_size: tuple[int, int]) -> Any:
    """
    Performs aspect-ratio preserving resizing.
    Image is resized to the dimension which has the larger ratio of (size / target_size).
    """
    img_w, img_h = img_size

    orig_w, orig_h = obtain_image_size(data)
    resize_w, resize_h = aspect_preserving_resize_size(orig_w, orig_h, img_size)

    assert (
        resize_h >= img_h and resize_w >= img_w
    ), f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {(resize_w, resize_h)}"

    data = transforms_F.resize(
        data,
        size=(resize_h, resize_w),  # type: ignore
        interpolation=transforms_F.InterpolationMode.BICUBIC,
        antialias=True,
    )
    return data


def transform_image(
    image: Image.Image,
    img_size: tuple[int, int],
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Resize, center-crop, RGB-convert, and normalize one decoded PIL image.

    RGB conversion deliberately follows the geometric transforms to preserve
    the historical ``ImageWDSLoader`` pipeline. Callers that require conversion
    before resizing can pass an image already converted to RGB.
    """
    image = resize_small_side_aspect_preserving(image, img_size)
    image = center_crop(image, img_size)
    return normalize_pixels(image.convert("RGB"), mean, std)
