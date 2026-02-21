# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import json
from typing import Any, Dict, List, Callable, Tuple, Optional, Iterable
import os
import subprocess
from io import BytesIO

import torch
import webdataset as wds
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, Lambda

from fastgen.datasets.wds_utils import (
    BaseWDSLoader,
    init_output_dict,
    get_prefixes,
)
from fastgen.datasets.decoders import (
    decode_video_full,
    decode_video_segment,
    decode_image,
    decode_txt,
    decode_npy_torch,
    decode_npz_torch,
)
from fastgen.datasets.crop_resize import center_crop, resize_small_side_aspect_preserving
import fastgen.utils.logging_utils as logger


def transform_video(
    video: torch.Tensor,
    sequence_length: int,
    img_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Transform a video tensor for inference, matching the training preprocessing.

    This is a standalone function for use in inference scripts. For training,
    use the VideoWDSLoader class which applies this automatically.

    Args:
        video: Input video tensor of shape (T, H, W, C) with uint8 RGB values.
        sequence_length: Number of frames to extract.
        img_size: Target size as (width, height).

    Returns:
        Dict with:
            - "real": Transformed video tensor of shape (C, T, H, W) normalized to [-1, 1]
            - "cropping_params": Dict of cropping parameters

    Raises:
        ValueError: If video has fewer frames than sequence_length.
    """
    video = video.permute(0, 3, 1, 2)  # t, h, w, c -> t, c, h, w

    if video.shape[0] < sequence_length:
        raise ValueError(f"video too short: {video.shape[0]} < {sequence_length}")

    video_sub = video[0:sequence_length]

    # resize & crop
    video_sub = resize_small_side_aspect_preserving(video_sub, img_size)
    video_sub, cropping_params = center_crop(video_sub, img_size, return_cropping_params=True)
    # normalize the video
    video_sub = video_sub.float() / 127.5 - 1
    video_sub = video_sub.permute(1, 0, 2, 3)  # t, c, h, w -> c, t, h, w

    return {"real": video_sub, "cropping_params": cropping_params}


# =============================================================================
# Constant Presets
# =============================================================================

# Dictionary of predefined constant values that can be added to outputs.
# Use presets_map in loader configs to map output keys to preset names.
#
# Example usage in config:
#   presets_map={"neg_condition": "neg_prompt_wan"}
#   # This adds output["neg_condition"] = PRESET_CONSTANTS["neg_prompt_wan"]
#
# Available presets:
#   - "neg_prompt_wan": Standard negative prompt for WAN video generation model.
#       Used to guide the model away from common video artifacts.
#   - "neg_prompt_cosmos": Negative prompt for Cosmos video generation model.
#       Used to guide the model away from common video artifacts.
#   - "empty_string": Empty string, useful for unconditional generation or
#       as a default fallback for missing text conditions.
PRESET_CONSTANTS: Dict[str, str] = {
    "neg_prompt_wan": (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
        "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
        "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
        "fused fingers, still picture, messy background, three legs, many people in the background, "
        "walking backwards"
    ),
    "neg_prompt_cosmos": (
        "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
        "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
        "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
        "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, "
        "outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, "
        "visual noise, and flickering. Overall, the video is of poor quality."
    ),
    "empty_string": "",
}


# =============================================================================
# Filter Presets
# =============================================================================

# Filter functions for filtering WebDataset items during loading.
# Use presets_filter in loader configs to apply these filters before decoding.
# Each filter receives the raw item dict (with byte values) and returns
# True to keep the item or False to discard it.
#
# Example usage in config:
#   presets_filter={
#       "score": {"threshold": 5.5, "score_key": "aesthetic_score_laion_v2"},
#       "empty_keys": {"keys": ["txt"]}
#   }
#   # This keeps only items with aesthetic_score_laion_v2 >= 5.5 and non-empty txt


def _filter_score(
    item: dict,
    threshold: float = 5.0,
    score_key: str = "score",
) -> bool:
    """Filter items based on a numeric score from JSON metadata.

    Keeps items where the score is >= threshold, or if no JSON/score exists.

    Args:
        item: WebDataset item containing raw bytes
        threshold: Minimum score to keep the item (default: 5.0)
        score_key: Key in the JSON metadata containing the score

    Returns:
        True if item should be kept, False if it should be filtered out
    """
    if "json" not in item:
        return True

    json_file = item["json"]
    if isinstance(json_file, bytes):
        json_string = json_file.decode("utf-8")
    else:
        json_string = json_file
    data = json.loads(json_string)

    if score_key in data and data[score_key] < threshold:
        return False

    return True


def _filter_empty_keys(
    item: dict,
    keys: List[str],
) -> bool:
    """Filter out items where specified keys have empty values.

    Args:
        item: WebDataset item containing raw bytes
        keys: List of keys that must be non-empty

    Returns:
        True if all specified keys are non-empty, False otherwise
    """
    for key in keys:
        if key in item and len(item[key]) == 0:
            return False
    return True


# Registry of available filter functions.
# Each filter takes an item dict and returns True to keep, False to discard.
# Filters receive additional kwargs from the presets_filter config.
PRESET_FILTERS: Dict[str, Callable[[dict], bool]] = {
    "score": _filter_score,
    "empty_keys": _filter_empty_keys,
}


# =============================================================================
# General WDS Loader
# =============================================================================


class WDSLoader(BaseWDSLoader):
    """Generic WebDataset loader for various data formats.

    Supports .npy/.npz/.pth tensors and text payloads, with configurable key mappings,
    preset constants, and filters. This is the base loader class that can be configured
    to handle most WebDataset formats.

    Args:
        datatags: List of WDS tags in format 'WDS:<PATH>'. Each tag points to a
            directory containing .tar shards.
        batch_size: Batch size for the dataloader.

        key_map: Dict mapping output keys to item keys (file extensions in the shard).
            Example: {"real": "latent.pth", "condition": "txt_emb.pth"}

        files_map: Optional dict mapping output keys to file paths. These files are
            loaded once at initialization and added as constants to every output.
            Supports both local paths and S3 paths. Relative paths are resolved
            relative to the first shard directory.
            Example: {"neg_condition": "neg_prompt_emb.npy"}

        presets_map: Optional dict mapping output keys to preset constant names.
            Available presets are defined in PRESET_CONSTANTS.
            Example: {"neg_condition": "neg_prompt_wan"}

        presets_filter: Optional dict of filter configurations. Keys are filter names
            (from PRESET_FILTERS), values are kwargs for that filter.
            Example: {"score": {"threshold": 5.5, "score_key": "aesthetic_score"}}

        txt_extensions: Iterable of file extensions to decode as text. Default: ("txt",)

        **kwargs: Additional arguments passed to BaseWDSLoader, including:
            - num_workers: Number of dataloader workers (auto-detected if None)
            - train: Whether this is a training dataset (default: True)
            - cache_path: Path to cache directory for downloaded shards
            - deterministic: Use deterministic iteration for resumability
            - sampler_start_idx: Start index for resumable iteration
            - shard_count_file: Path to JSON with shard sample counts
            - ignore_index_paths: List of JSON files specifying samples to skip
            - shard_start/shard_end: Range of shards to use
            - shards_per_worker: Forces a specific number of total shards per worker by truncating/repeating shards.
            - shuffle_size: Shuffle buffer size (default: 1000)
    """

    def __init__(
        self,
        datatags: List[str],
        batch_size: int,
        # Key configuration (from output keys to file extension keys)
        key_map: Dict[str, str] | None = None,
        # File key configuration (from output keys to file paths; these are constants that are not loaded from the item)
        files_map: Dict[str, str] | None = None,
        # Presets key configuration (from output keys to preset names)
        presets_map: Dict[str, str] | None = None,
        # Filter options
        presets_filter: Dict[str, Dict[str, Any]] | None = None,
        txt_extensions: Iterable[str] = ("txt",),
        # Common options passed to BaseWDSLoader
        **kwargs,
    ):
        # key_map: output keys -> item keys (file extensions in the shard)
        if key_map is None:
            key_map = {"real": "latent.pth", "condition": "txt_emb.pth"}
        self._key_map = key_map

        # Store text extensions
        self._txt_extensions = txt_extensions

        # Preload constants from file paths
        self._constants = {}
        if files_map is not None:
            self._constants.update(
                {output_key: self._load_file(file, datatags) for output_key, file in files_map.items()}
            )

        # Preload constants from preset names
        if presets_map is not None:
            self._constants.update({output_key: PRESET_CONSTANTS[preset] for output_key, preset in presets_map.items()})

        # Initialize filter functions
        self._filter_fn = []
        if presets_filter is not None:
            self._filter_fn.extend(
                [
                    functools.partial(PRESET_FILTERS[filter_name], **filter_kwargs)
                    for filter_name, filter_kwargs in presets_filter.items()
                ]
            )

        super().__init__(datatags=datatags, batch_size=batch_size, **kwargs)
        logger.debug(f"{self.__class__.__name__} ({datatags}) initialized, batch_size={self.batch_size}.")

    def _preprocess(self, item: dict) -> Dict[str, Any]:
        """Preprocess data from the WebDataset item.

        First populates the output dictionary with preset constants, then overrides
        with values from the item according to key_map. Constants serve as defaults
        when an item key is missing.

        Args:
            item: Decoded WebDataset item dict containing file data keyed by extension.

        Returns:
            Dict with 'fname', 'shard', and all mapped output keys.

        Raises:
            AssertionError: If a required key is missing from both item and constants.
        """
        output = {
            **init_output_dict(item),
            **self._constants,
        }

        # Extract keys from item, falling back to constants if present
        for output_key, item_key in self._key_map.items():
            if item_key in item:
                output[output_key] = item[item_key]
            else:
                assert (
                    output_key in self._constants
                ), f"Missing required key '{item_key}' for output '{output_key}', without a default preset value."
                logger.warning(f"Missing key '{item_key}' for output '{output_key}', using default preset value.")
        return output

    def filter_items(
        self,
        item: dict,
    ) -> bool:
        """Filter function for WebDataset items before decoding.

        Checks that all required keys from key_map are present (unless a constant
        fallback exists) and that present keys contain raw byte data.

        Args:
            item: Raw WebDataset item dict (pre-decoding, values are bytes).

        Returns:
            True if item passes all filters and should be processed.
        """
        for output_key, item_key in self._key_map.items():
            if item_key not in item:
                # Missing key is only OK if we have a constant fallback
                if output_key not in self._constants:
                    return False
            elif not isinstance(item[item_key], (bytes, bytearray)):
                # Present keys must be raw bytes (not yet decoded)
                return False
        if not all(f(item) for f in self._filter_fn):
            return False
        return True

    @staticmethod
    def _load_file(path: str, wds_tags: List[str]) -> torch.Tensor:
        """Load a numpy file as a torch tensor.

        Supports both local paths and S3 paths (via s5cmd). Relative paths
        are resolved relative to the directory of the first shard.

        Args:
            path: Path to the .npy file (local, S3, or relative)
            wds_tags: List of WDS tags used to resolve relative paths

        Returns:
            Loaded tensor, or None if file not found (for local paths)

        Raises:
            RuntimeError: If S3 download fails
        """
        if not path.startswith("s3://") and not os.path.isabs(path):
            prefixes = get_prefixes(wds_tags)
            path = os.path.join(os.path.dirname(prefixes[0]), path)

        if path.startswith("s3://"):
            proc = subprocess.run(["s5cmd", "cat", path], capture_output=True)
            if proc.returncode == 0:
                neg_condition = np.load(BytesIO(proc.stdout))
                return torch.from_numpy(neg_condition)
            else:
                raise RuntimeError(f"Failed to load file from {path}: {proc.stderr.decode()}")

        elif os.path.isfile(path):
            neg_condition = np.load(path)
            return torch.from_numpy(neg_condition)

        logger.info(f"No file found at {path}, using zero embedding.")
        return None

    def _pipeline(self, dataset: wds.WebDataset) -> wds.WebDataset:
        """Process the WebDataset through filtering, decoding, and preprocessing.

        Pipeline stages:
            1. Filter: Remove items missing required keys (via filter_items)
            2. Decode: Convert raw bytes to typed values using self.decoders
            3. Preprocess: Map decoded values to output keys via _preprocess

        Args:
            dataset: Raw WebDataset with byte values.

        Returns:
            Processed WebDataset ready for batching.
        """
        dataset = dataset.select(self.filter_items)
        # Decode only the keys we need from the key_map
        decode_keys = list(self._key_map.values())
        decoder = wds.autodecode.Decoder(self.decoders, only=decode_keys)
        dataset = dataset.map(decoder, handler=wds.warn_and_continue)
        dataset = dataset.map(self._preprocess, handler=wds.handlers.warn_and_continue)

        return dataset

    @property
    def decoders(self) -> list[Callable[[str, bytes], Any]]:
        """Decoders for converting raw bytes to typed values.

        Returns a list of decoder functions that are tried in order.
        Each decoder returns None if it doesn't handle the file type.
        Note: .pth and .json are handled by webdataset's autodecode.
        """
        return [
            decode_npy_torch,
            decode_npz_torch,
            functools.partial(decode_txt, txt_extensions=self._txt_extensions, strip=True),
        ]


class ImageWDSLoader(WDSLoader):
    """WebDataset loader for raw images with automatic transforms.

    Extends WDSLoader to add image decoding and transformation. Images are
    automatically decoded from common formats (jpg, png, etc.) and transformed
    to a consistent size and normalization.

    Args:
        datatags: List of WDS tags in format 'WDS:<PATH>'
        batch_size: Batch size for the dataloader
        input_res: Target image resolution (square). Images are resized and
            center-cropped to this size. Default: 512

        key_map: Dict mapping output keys to item keys. Image keys should
            reference image file extensions (e.g., "jpg", "png").
            Example: {"real": "jpg", "condition": "txt"}

        **kwargs: Additional arguments passed to WDSLoader and BaseWDSLoader.

    The default image transform pipeline:
        1. Resize to input_res (bicubic interpolation)
        2. Center crop to input_res x input_res
        3. Convert to RGB
        4. Convert to tensor
        5. Normalize with norm_mean and norm_std
    """

    def __init__(
        self,
        *args,
        input_res: int = 512,
        norm_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        norm_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        **kwargs,
    ):
        # Store pipeline configuration
        self._input_res = input_res
        self._norm_mean = norm_mean
        self._norm_std = norm_std
        super().__init__(*args, **kwargs)

    @property
    def decoders(self) -> list[Callable[[str, bytes], Any]]:
        return super().decoders + [functools.partial(decode_image, image_transform=self._image_transform)]

    @property
    def _image_transform(self) -> Compose:
        """Default image transform pipeline for resizing, cropping, and normalizing.

        Returns:
            Composed transform pipeline that:
            - Resizes to input_res using bicubic interpolation
            - Center crops to input_res
            - Converts to RGB
            - Converts to tensor
            - Normalizes the tensor with norm_mean and norm_std
        """
        return Compose(
            [
                Resize(self._input_res, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(self._input_res),
                Lambda(lambda img: img.convert("RGB")),
                ToTensor(),
                Normalize(self._norm_mean, self._norm_std),
            ]
        )


class VideoWDSLoader(WDSLoader):
    """WebDataset loader for raw videos with automatic transforms.

    Extends WDSLoader to add video decoding and transformation. Videos are
    decoded using PyAV with random segment sampling, then resized and cropped.

    Args:
        datatags: List of WDS tags in format 'WDS:<PATH>'
        batch_size: Batch size for the dataloader
        sequence_length: Number of frames to extract from each video. Default: 81
        img_size: Target (width, height) for video frames. Default: (832, 480)
        video_keys: Iterable of output keys that contain video data and should be
            transformed. Default: ("real",)
        decoder_type: Type of video decoder to use. Default: "segment"
            - "segment": Random segment sampling (if video is longer than sequence_length)
            - "full": Full video decoding (no random segment sampling)

        key_map: Dict mapping output keys to item keys. Video keys should
            reference video file extensions (e.g., "mp4").
            Example: {"real": "mp4", "condition": "txt"}

        **kwargs: Additional arguments passed to WDSLoader and BaseWDSLoader.

    The video transform pipeline:
        1. Random segment sampling (if video is longer than sequence_length)
        2. Resize preserving aspect ratio (small side to target size)
        3. Center crop to exact img_size
        4. Normalize to [-1, 1] range
        5. Permute to (C, T, H, W) format
    """

    def __init__(
        self,
        *args,
        sequence_length: int = 81,
        img_size: Tuple[int, int] = (832, 480),
        video_keys: Optional[Iterable[str]] = ("real",),
        decoder_type: str = "segment",
        **kwargs,
    ):
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.video_keys = [] if video_keys is None else list(video_keys)
        self.decoder_type = decoder_type
        super().__init__(*args, **kwargs)
        if not all(key in self._key_map for key in self.video_keys):
            raise ValueError(f"`{self.video_keys}` keys are required but got {list(self._key_map.keys())}")

    @property
    def decoders(self) -> list[Callable[[str, bytes], Any]]:
        if self.decoder_type == "segment":
            video_decoder = functools.partial(
                decode_video_segment,
                num_frames=self.sequence_length,
                output_format="torch",
            )
        elif self.decoder_type == "full":
            video_decoder = functools.partial(
                decode_video_full,
                output_format="torch",
            )
        else:
            raise ValueError(f"Invalid decoder type: {self.decoder_type}")
        return super().decoders + [video_decoder]

    def _transform_video(self, video: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Transform a video to the configured sequence length and frame size.

        Args:
            video: Input video tensor of shape (T, H, W, C) with uint8 RGB values.

        Returns:
            Tuple of:
                - Transformed video tensor of shape (C, T, H, W) normalized to [-1, 1]
                - Dict of cropping parameters (resize_w, resize_h, crop_x0, crop_y0, crop_w, crop_h)

        Raises:
            ValueError: If video has fewer frames than sequence_length.
        """
        result = transform_video(video, self.sequence_length, self.img_size)
        return result["real"], result["cropping_params"]

    def _preprocess(self, item: dict) -> Dict[str, Any]:
        output = super()._preprocess(item)
        for video_key in self.video_keys:
            video, cropping_params = self._transform_video(output[video_key])
            output[video_key] = video
            if len(self.video_keys) == 1:
                output["cropping_params"] = cropping_params
            else:
                output[f"{video_key}_cropping_params"] = cropping_params
        return output
