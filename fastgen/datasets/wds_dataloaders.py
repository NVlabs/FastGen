# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import json
import math
from typing import Any, Dict, List, Callable, Tuple, Optional, Iterable, Union
import os
import subprocess
from io import BytesIO

import torch
import webdataset as wds
import numpy as np
import fastgen.utils.s3 as s3
from fastgen.datasets.wds_utils import (
    BaseWDSLoader,
    init_output_dict,
    get_prefixes,
    validate_wds_config,
)
from fastgen.datasets.decoders import (
    decode_video_full,
    decode_video_segment,
    decode_video_audio_segment,
    decode_video_audio_full,
    decode_image,
    decode_txt,
    decode_npy_torch,
    decode_npz_torch,
)
from fastgen.datasets.crop_resize import (
    center_crop,
    normalize_pixels,
    resize_small_side_aspect_preserving,
    transform_image,
)
from fastgen.utils.audio import resample_audio
from fastgen.utils.cache_utils import file_extension
from fastgen.utils.camera_utils import transform_intrinsics
import fastgen.utils.logging_utils as logger


def transform_video(
    video: torch.Tensor,
    sequence_length: int,
    img_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Per-sample video preprocessing: truncate -> aspect-preserving resize -> center-crop -> normalize.

    Args:
        video: Input video tensor of shape (T, H, W, C) with uint8 RGB values.
        sequence_length: Number of frames to extract (from the start).
        img_size: Target size as (width, height).

    Returns:
        Dict with:
            - "real_raw": Transformed video tensor of shape (C, T, H, W) normalized to [-1, 1]
            - "cropping_params": resize/crop geometry (resize_w/h, orig_w/h, crop_x0/y0) consumed by
              ``transform_intrinsics`` to rescale camera intrinsics into the cropped frame

    Raises:
        ValueError: If video has fewer frames than sequence_length.
    """
    video = video.permute(0, 3, 1, 2)  # t, h, w, c -> t, c, h, w

    if video.shape[0] < sequence_length:
        raise ValueError(f"video too short: {video.shape[0]} < {sequence_length}")

    video_sub = video[0:sequence_length]

    # resize & crop
    orig_h, orig_w = video_sub.shape[2], video_sub.shape[3]
    video_sub = resize_small_side_aspect_preserving(video_sub, img_size)
    video_sub, cropping_params = center_crop(video_sub, img_size, return_cropping_params=True)
    cropping_params["orig_w"], cropping_params["orig_h"] = orig_w, orig_h
    # normalize the video to [-1, 1]
    video_sub = normalize_pixels(video_sub)
    video_sub = video_sub.permute(1, 0, 2, 3)  # t, c, h, w -> c, t, h, w

    return {"real_raw": video_sub, "cropping_params": cropping_params}


# =============================================================================
# Constant Presets
# =============================================================================

# Dictionary of predefined constant values that can be added to outputs.
# Use presets_map in loader configs to map output keys to preset names.
#
# Example usage in config:
#   presets_map={"neg_condition_raw": "neg_prompt_wan"}
#   # This adds output["neg_condition_raw"] = PRESET_CONSTANTS["neg_prompt_wan"]
#
# The per-model negative prompts are shared with the inference scripts: the single
# canonical copy of each lives in scripts/inference/prompts/ (usable directly via
# ``--neg_prompt_file``) and is loaded here for the dataloader presets.
#
# Available presets:
#   - "neg_prompt_wan": Standard negative prompt for WAN video generation model.
#       Used to guide the model away from common video artifacts.
#   - "neg_prompt_lingbot_world": Negative prompt for the LingBot-World model.
#   - "neg_prompt_cosmos": Negative prompt for Cosmos video generation model.
#       Used to guide the model away from common video artifacts.
#   - "neg_prompt_cosmos3": Official Cosmos3 structured-caption negative prompt
#       (the nvidia/Cosmos3-Nano release's negative_prompt.json, serialized with
#       ``json.dumps`` exactly like the model-card examples).
#   - "neg_prompt_ltx2": Full negative prompt from the LTX-2 pipeline defaults.
#   - "neg_prompt_ltx2_short": Short negative prompt from LTX-2 training configs.
#   - "empty_string": Empty string, useful for unconditional generation or
#       as a default fallback for missing text conditions.

_NEG_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "scripts", "inference", "prompts")


def _load_negative_prompt(filename: str) -> str:
    with open(os.path.join(_NEG_PROMPTS_DIR, filename), encoding="utf-8") as f:
        return f.read().strip()


PRESET_CONSTANTS: Dict[str, str] = {
    "neg_prompt_wan": _load_negative_prompt("negative_prompt.txt"),
    "neg_prompt_lingbot_world": _load_negative_prompt("negative_prompt_lingbot_world.txt"),
    "neg_prompt_cosmos": _load_negative_prompt("negative_prompt_cosmos.txt"),
    "neg_prompt_cosmos3": _load_negative_prompt("negative_prompt_cosmos3.txt"),
    "neg_prompt_ltx2": _load_negative_prompt("negative_prompt_ltx.txt"),
    "neg_prompt_ltx2_short": _load_negative_prompt("negative_prompt_ltx_short.txt"),
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

    Keeps items where the score is >= threshold, or if no JSON/score exists. Items whose
    metadata cannot be parsed or whose score is not a finite number (e.g. ``null``, ``true``,
    ``NaN``, or ``Infinity``) are dropped with a warning, like samples that fail to decode,
    since raising here would stop the loader. A non-numeric ``threshold`` is a configuration
    error and raises.

    Args:
        item: WebDataset item containing raw bytes
        threshold: Minimum score to keep the item (default: 5.0)
        score_key: Key in the JSON metadata containing the score

    Returns:
        True if item should be kept, False if it should be filtered out
    """
    if "json" not in item:
        return True

    try:
        data = json.loads(item["json"])
    except (ValueError, RecursionError) as error:
        logger.warning(f"Dropping sample {item.get('__key__')}: cannot parse its JSON metadata: {error}")
        return False
    if not isinstance(data, dict) or score_key not in data:
        return True

    score = data[score_key]
    # Ints are exact and always finite; math.isfinite raises OverflowError on ones beyond the float range.
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or (isinstance(score, float) and not math.isfinite(score))
    ):
        logger.warning(
            f"Dropping sample {item.get('__key__')}: its {score_key!r} score {score!r} is not a finite number"
        )
        return False
    return score >= threshold


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
            Supports both local paths and S3 paths. Relative paths are resolved against
            the parent directory of the first datatag path (the directory itself when the
            path ends with ``/``). ``.npy`` loads a single tensor;
            ``.pth``/``.pt`` loads a tensor or a list of tensors (e.g. SDXL's dual
            ``[text_embedding, pooled_text_embedding]`` negative).
            Example: {"neg_condition": "neg_prompt_emb.npy"}

        presets_map: Optional dict mapping output keys to preset constant names.
            Available presets are defined in PRESET_CONSTANTS.
            Text presets are raw prompts and therefore use raw condition keys.
            Example: {"neg_condition_raw": "neg_prompt_wan"}

        presets_filter: Optional dict of filter configurations. Keys are filter names
            (from PRESET_FILTERS), values are kwargs for that filter.
            Example: {"score": {"threshold": 5.5, "score_key": "aesthetic_score"}}

        txt_extensions: Iterable of file extensions to decode as text. Default: ("txt",)

        **kwargs: Additional arguments passed to BaseWDSLoader, including:
            - num_workers: Number of dataloader workers (auto-detected if None)
            - train: Whether this is a training dataset (default: True)
            - deterministic: Use deterministic iteration for resumability
            - strict: Raise on samples or shards that fail instead of skipping them (default: False)
            - drop_last: With deterministic and train=False, give every rank the same number of samples (default: True)
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
        validate_wds_config(deterministic=kwargs.get("deterministic", False), presets_filter=presets_filter)
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
                # ``filter_items`` already dropped items that lack both the key and a constant fallback.
                assert (
                    output_key in self._constants
                ), f"Missing required key '{item_key}' for output '{output_key}', without a default preset value."
                logger.debug(f"Missing key '{item_key}' for output '{output_key}', using default preset value.")
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
            True if the item has the data to be processed.
        """
        for output_key, item_key in self._key_map.items():
            if item_key not in item:
                # Missing key is only OK if we have a constant fallback
                if output_key not in self._constants:
                    return False
            elif not isinstance(item[item_key], (bytes, bytearray)):
                # Present keys must be raw bytes (not yet decoded)
                return False
        return True

    def _prefilter(self, dataset: wds.WebDataset) -> wds.WebDataset:
        # presets_filter excludes samples on purpose, so it must not count toward failures; deterministic loading
        # rejects it (see validate_wds_config).
        for filter_fn in self._filter_fn:
            dataset = dataset.select(filter_fn)
        return dataset

    @staticmethod
    def _load_file(path: str, wds_tags: List[str]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Load a constant embedding file as a torch tensor (or list of tensors).

        Supports both local paths and S3 paths (via s5cmd). Relative paths are
        resolved against ``os.path.dirname`` of the first datatag path: its parent
        directory, or the directory itself when the path ends with ``/``.

        Dispatch is by extension so a ``files_map`` constant can mirror whatever
        structure its per-sample counterpart produces:

        - ``.npy``: a single array -> ``torch.Tensor`` (unchanged default).
        - ``.pth`` / ``.pt``: whatever was ``torch.save``-d, mirroring the
          per-sample ``.pth`` autodecode path. A ``list`` / ``tuple`` of tensors
          loads as ``list[torch.Tensor]`` (e.g. SDXL's dual
          ``[text_embedding, pooled_text_embedding]`` negative); a bare tensor
          stays a ``torch.Tensor``.

        Args:
            path: Path to the file (local, S3, or relative)
            wds_tags: List of WDS tags used to resolve relative paths

        Returns:
            Loaded tensor, or list of tensors for a list/tuple ``.pth`` payload.

        Raises:
            RuntimeError: If an S3 download fails.
            FileNotFoundError: If a local file does not exist.
        """
        if not s3.is_uri(path) and not os.path.isabs(path):
            prefixes = get_prefixes(wds_tags)
            path = os.path.join(os.path.dirname(prefixes[0]), path)

        if s3.is_uri(path):
            proc = subprocess.run(["s5cmd", "cat", path], capture_output=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to load file from {path}: {proc.stderr.decode()}")
            source: Union[str, BytesIO] = BytesIO(proc.stdout)
        elif os.path.isfile(path):
            source = path
        else:
            raise FileNotFoundError(
                f"files_map file not found: '{path}'. Every files_map entry must reference an "
                f"existing file (e.g. a precomputed embedding); provide the file or remove the entry."
            )

        # ``.pth``/``.pt`` mirror the per-sample autodecode path (torch.load), so a
        # list payload (e.g. SDXL's dual [text, pooled] negative) survives intact.
        if file_extension(path) in {".pth", ".pt"}:
            obj = torch.load(source, map_location="cpu")
            if isinstance(obj, (list, tuple)):
                return [t if torch.is_tensor(t) else torch.as_tensor(t) for t in obj]
            return obj if torch.is_tensor(obj) else torch.as_tensor(obj)

        return torch.from_numpy(np.load(source))

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
        # Errors propagate: process_samples runs this per sample and skips or, with strict=True, raises.
        dataset = dataset.map(decoder)
        dataset = dataset.map(self._preprocess)

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
        input_res: Target image resolution as an int (square) or a (W, H) tuple
            for non-square crops. Images are resized and center-cropped to this
            size. Default: 512

        key_map: Required dict mapping output keys to item keys. Image keys should
            reference image file extensions (e.g., "jpg", "png").
            Example: {"real_raw": "jpg", "condition_raw": "txt"}

        **kwargs: Additional arguments passed to WDSLoader and BaseWDSLoader.

    The default image transform pipeline:
        1. Resize preserving aspect ratio (small side to target size, bicubic interpolation)
        2. Center crop to input_res (W x H)
        3. Convert to RGB
        4. Convert to tensor
        5. Normalize with norm_mean and norm_std
    """

    def __init__(
        self,
        datatags: List[str],
        batch_size: int,
        key_map: Dict[str, str],
        input_res: Union[int, Tuple[int, int]] = 512,  # int for square, or (W, H) for non-square
        norm_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        norm_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        **kwargs,
    ):
        # Store pipeline configuration
        if isinstance(input_res, int):
            self._input_res = (input_res, input_res)
        else:
            self._input_res = input_res
        assert len(self._input_res) == 2, f"input_res must be a tuple of two integers, but got {input_res}"
        self._norm_mean = norm_mean
        self._norm_std = norm_std
        super().__init__(datatags=datatags, batch_size=batch_size, key_map=key_map, **kwargs)

    @property
    def decoders(self) -> list[Callable[[str, bytes], Any]]:
        return super().decoders + [functools.partial(decode_image, image_transform=self._image_transform)]

    def _image_transform(self, img):
        """Aspect-preserving resize (bicubic) + center-crop to ``input_res``, then
        normalize to [-1, 1] via the shared ``normalize_pixels`` (== ``ToTensor()`` +
        ``Normalize(norm_mean, norm_std)``). Shared with the inference path.
        """
        return transform_image(img, self._input_res, self._norm_mean, self._norm_std)


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
            transformed. Default: ("real_raw",)
        decoder_type: Type of video decoder to use. Default: "segment"
            - "segment": Random segment sampling (if video is longer than sequence_length)
            - "full": Full video decoding (no random segment sampling); the whole stream is read, without
              the decoder's default wall-clock budget.
        target_fps: Optional target FPS for timestamp-based resampling. When set
            with decoder_type="segment", the random segment spans
            sequence_length / target_fps seconds and is resampled by timestamp.
            When set with decoder_type="full", the full decoded video is
            resampled by timestamp before transforms.

        key_map: Required dict mapping output keys to item keys. Video keys should
            reference video file extensions (e.g., "mp4").
            Example: {"real_raw": "mp4", "condition_raw": "txt"}

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
        datatags: List[str],
        batch_size: int,
        key_map: Dict[str, str],
        sequence_length: int = 81,
        img_size: Tuple[int, int] = (832, 480),
        video_keys: Optional[Iterable[str]] = ("real_raw",),
        decoder_type: str = "segment",
        target_fps: Optional[float] = None,
        **kwargs,
    ):
        if key_map is None:
            raise TypeError("`key_map` must not be None")
        if target_fps is not None and (not math.isfinite(target_fps) or target_fps <= 0):
            raise ValueError("target_fps must be a finite positive number")
        if decoder_type not in ("segment", "full"):
            raise ValueError(f"decoder_type must be 'segment' or 'full', got {decoder_type!r}")
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.video_keys = [] if video_keys is None else list(video_keys)
        self.decoder_type = decoder_type
        self.target_fps = target_fps
        super().__init__(datatags=datatags, batch_size=batch_size, key_map=key_map, **kwargs)
        if not all(key in self._key_map for key in self.video_keys):
            raise ValueError(f"`{self.video_keys}` keys are required but got {list(self._key_map.keys())}")

    @property
    def decoders(self) -> list[Callable[[str, bytes], Any]]:
        if self.decoder_type == "segment":
            video_decoder = functools.partial(
                decode_video_segment,
                num_frames=self.sequence_length,
                output_format="torch",
                target_fps=self.target_fps,
                return_segment_start_frame=True,
            )
        else:  # "full", validated in __init__
            # Training reads the whole stream; the decoder's default wall-clock budget serves the VLM judge.
            video_decoder = functools.partial(
                decode_video_full,
                output_format="torch",
                target_fps=self.target_fps,
                timeout_s=None,
                return_source_frame_indices=True,
            )
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
        return result["real_raw"], result["cropping_params"]

    def _preprocess(self, item: dict) -> Dict[str, Any]:
        output = super()._preprocess(item)
        for video_key in self.video_keys:
            decoded = output[video_key]
            if isinstance(decoded, (bytes, bytearray)):
                # Video decoders return None on failure, and WebDataset then passes the raw bytes on.
                raise ValueError(f"Cannot decode '{self._key_map[video_key]}' of sample {item['__key__']}")
            if not isinstance(decoded, dict):
                decoded = {"frames": decoded}
            video, cropping_params = self._transform_video(decoded["frames"])
            output[video_key] = video
            # Per-video-key outputs keep bare names for the usual single video key; several keys each get their own.
            prefix = "" if len(self.video_keys) == 1 else f"{video_key}_"
            output[f"{prefix}cropping_params"] = cropping_params
            output[f"{prefix}segment_start_frame"] = decoded.get("segment_start_frame", 0)
        return output


class CameraVideoWDSLoader(VideoWDSLoader):
    """VideoWDSLoader with pre-computed camera poses and intrinsics (e.g. from VIPE).

    Gathers the pose/intrinsics rows of the frames the decoder kept and rescales
    intrinsics to the final resolution after resize + center-crop. Pose /
    intrinsics arrays carry one row per *source* frame (e.g. VIPE output); when
    ``target_fps`` drops or duplicates frames, each kept frame's source row is
    taken, so no pose interpolation is needed. Full mode places frames by decode
    order; segment mode places the first kept frame by its timestamp at the
    source frame rate, so it needs constant-frame-rate video. Samples whose
    frames cannot be placed, or whose sidecar lacks rows for them, raise: the
    loader skips them with a warning, or stops with ``strict=True``.

    Required key_map entries::

        key_map={
            "real_raw": "mp4",
            "condition_raw": "txt",
            "camera_poses": "poses.npy",           # (T, 4, 4) OpenCV c2w
            "camera_intrinsics": "intrinsics.npy", # (T, 4) [fx, fy, cx, cy]
        }
    """

    def __init__(
        self,
        *args,
        camera_pose_key: str = "camera_poses",
        camera_intrinsics_key: str = "camera_intrinsics",
        **kwargs,
    ):
        self.camera_pose_key = camera_pose_key
        self.camera_intrinsics_key = camera_intrinsics_key
        super().__init__(*args, **kwargs)
        # `_preprocess` reads one decoded video directly and the bare `output["cropping_params"]`,
        # which `VideoWDSLoader._preprocess` only writes for single-video-key configs.
        assert len(self.video_keys) == 1, (
            f"{type(self).__name__} requires exactly one video key (got {self.video_keys}); "
            "camera alignment relies on one decoded video and the bare 'cropping_params' entry."
        )
        camera_keys = (self.camera_pose_key, self.camera_intrinsics_key)
        if not all(key in self._key_map for key in camera_keys):
            raise ValueError(f"`{list(camera_keys)}` keys are required but got {list(self._key_map.keys())}")

    def _camera_sidecars(self, output: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-source-frame ``(poses (T, 4, 4), intrinsics (T, 4))`` of the sample; subclasses whose shards
        store camera data in another format convert it here before the kept frames' rows are gathered."""
        return output[self.camera_pose_key], output[self.camera_intrinsics_key]

    @staticmethod
    def _select_camera(arr: torch.Tensor, indices: Optional[list]) -> torch.Tensor:
        """Rows of ``arr`` for the kept frames, gathered by the source frame indices the decoder reports.

        Raises ``ValueError``, like ``transform_video`` on a short clip, when the decoder could not place the
        frames or the sidecar lacks a row for one of them; the loader then skips the sample, or stops with
        ``strict=True``. Slicing at the decoded cadence or clamping would silently train on misaligned poses.
        """
        if indices is None:
            raise ValueError(
                "decoder could not place the kept frames in the source stream (unknown or uneven frame rate)"
            )
        index = torch.as_tensor(indices, dtype=torch.long)
        if index.min() < 0 or index.max() >= arr.shape[0]:
            raise ValueError(
                f"camera sidecar has {arr.shape[0]} rows but the kept source frames span "
                f"{int(index.min())}..{int(index.max())}"
            )
        return arr[index]

    def _preprocess(self, item: dict) -> Dict[str, Any]:
        decoded = item[self._key_map[self.video_keys[0]]]
        indices = decoded.get("source_frame_indices") if isinstance(decoded, dict) else None
        if indices is not None:
            decoded_frame_count = decoded["frames"].shape[0]
            if len(indices) != decoded_frame_count:
                raise ValueError(
                    f"decoder returned {decoded_frame_count} frames but {len(indices)} source frame indices"
                )
            # Full decoding may return more frames than ``transform_video`` keeps.
            indices = indices[: self.sequence_length]

        # VideoWDSLoader handles unpacking, transformation, segment_start_frame, and cropping_params.
        output = super()._preprocess(item)
        poses, intrinsics = self._camera_sidecars(output)
        output[self.camera_pose_key] = self._select_camera(poses, indices)
        intrinsics = self._select_camera(intrinsics, indices)
        output[self.camera_intrinsics_key] = transform_intrinsics(intrinsics, output["cropping_params"])
        return output


class AudioVideoWDSLoader(VideoWDSLoader):
    """WebDataset loader that extracts both video frames and audio from MP4.

    Extends :class:`VideoWDSLoader` to additionally decode the audio track
    embedded in the video container.  Per-sample (pre-collate), the
    audio waveform is resampled to ``target_sample_rate`` (when provided),
    then trimmed/padded to ``num_video_frames / fps`` seconds at the
    target rate, then packed together with the video tensor into a
    modality dict under each ``video_key``::

        output[video_key] = {
            "video": Tensor[C, T, H, W],
            "audio": Tensor[A_C, target_duration * target_sample_rate],
        }

    When ``target_sample_rate`` is ``None`` no resampling is performed
    and the audio is packed at its native rate; in that mode mixed-SR
    batches will fail collate (different per-sample lengths).

    The ``"audio"`` key is **omitted** for clips whose mp4 container
    carries no audio stream — in that case the dict contains only
    ``"video"``.  Such clips will not collate with audio-bearing clips
    in the same batch (the loader rejects mismatched nested key sets
    before collation, stopping the loader regardless of ``strict``), so
    **mixed audio presence requires ``batch_size=1``** or excluding the
    audio-less clips via ``ignore_index_paths``.

    A clip with audio but no frame rate (no ``target_fps`` and none in
    the container) raises ``ValueError``, since its audio cannot be
    sized to the kept frames; the loader skips it, or stops with
    ``strict=True``.

    Args:
        target_sample_rate: Target sample rate (Hz) to resample every
            audio clip to before collate.  If ``None``, audio is kept
            at its native rate.  Defaults to ``None``.
        **kwargs: Forwarded to :class:`VideoWDSLoader`.
    """

    def __init__(self, *args, target_sample_rate: Optional[int] = None, **kwargs):
        if target_sample_rate is not None and target_sample_rate <= 0:
            raise ValueError(f"target_sample_rate must be positive, got {target_sample_rate}")
        self.target_sample_rate = target_sample_rate
        super().__init__(*args, **kwargs)

    @property
    def decoders(self) -> list[Callable[[str, bytes], Any]]:
        if self.decoder_type == "segment":
            va_decoder = functools.partial(
                decode_video_audio_segment,
                num_frames=self.sequence_length,
                output_format="torch",
                target_fps=self.target_fps,
            )
        else:  # "full", validated in __init__
            va_decoder = functools.partial(
                decode_video_audio_full,
                output_format="torch",
                target_fps=self.target_fps,
            )
        # The video-audio decoder replaces the plain video decoder but we
        # keep the parent's non-video decoders (npy, npz, txt).
        return WDSLoader.decoders.fget(self) + [va_decoder]  # type: ignore[attr-defined]

    def _preprocess(self, item: dict) -> Dict[str, Any]:
        # The video-audio decoder returns a dict for each video key; hand
        # VideoWDSLoader._preprocess the frames and the segment start (0 for full decoding).
        # Audio data is stashed separately since it's not in the key_map.
        audio_data = {}
        for video_key in self.video_keys:
            item_key = self._key_map[video_key]
            va = item[item_key]
            if isinstance(va, (bytes, bytearray)):
                raise ValueError(f"Cannot decode '{item_key}' of sample {item['__key__']}")
            if va.get("audio") is not None and va.get("video_fps", 0.0) <= 0:
                raise ValueError(
                    f"Cannot size the audio of '{item_key}' in sample {item['__key__']}: unknown video frame rate"
                )
            item[item_key] = {"frames": va["video"], "segment_start_frame": va.get("segment_start_frame", 0)}
            audio_data[video_key] = (
                va.get("audio"),
                va.get("audio_sr"),
                va.get("video_fps", 0.0),
            )

        output = super()._preprocess(item)

        # Resample-then-trim-then-pack. Order matters: doing the trim against
        # target_sample_rate (rather than native audio_sr) gives a uniform
        # output length across mixed-SR batches and avoids the boundary
        # discontinuity from trimming the signal before sinc resampling.
        for video_key, (audio, audio_sr, video_fps) in audio_data.items():
            video_tensor = output[video_key]
            modality_dict: Dict[str, torch.Tensor] = {"video": video_tensor}

            if audio is not None and audio_sr is not None:
                if self.target_sample_rate is not None and audio_sr != self.target_sample_rate:
                    audio = resample_audio(audio, audio_sr, self.target_sample_rate)
                    audio_sr = self.target_sample_rate

                # Decoder always emits stereo [2, samples] (reference LTX-2 channel
                # handling, see ``decoders._downmix_to_stereo``).
                assert audio.ndim == 2, f"expected audio [channels, samples], got shape {tuple(audio.shape)}"
                num_video_frames = video_tensor.shape[1]  # (C, T, H, W)
                # Multiply before dividing (exact for integral counts); mirrors the decoder
                # and ``LTX2._compute_audio_shape``.
                target_samples = int(num_video_frames * audio_sr / video_fps)
                if audio.shape[-1] > target_samples:
                    audio = audio[..., :target_samples]
                elif audio.shape[-1] < target_samples:
                    pad_size = target_samples - audio.shape[-1]
                    audio = torch.nn.functional.pad(audio, (0, pad_size))

                modality_dict["audio"] = audio

            output[video_key] = modality_dict

        return output
