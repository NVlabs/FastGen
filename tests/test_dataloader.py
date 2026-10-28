# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for fastgen dataloaders: the class-conditional image loaders, ``WDSLoader`` and its image/video
subclasses, and the ``AudioVideoWDSLoader`` / ``CameraVideoWDSLoader`` modality loaders.

This module tests:
1. Dummy dataset creation with correct specifications
2. Loader initialization and batch validation (keys, types, shapes)
3. Edge cases with missing files in WebDataset shards
4. Deterministic loading with resuming capabilities
5. Audio sizing and alignment in ``AudioVideoWDSLoader`` and the camera sidecars of ``CameraVideoWDSLoader``
"""

import io
import json
import os
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import shutil
import importlib

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image
from pydantic import TypeAdapter, ValidationError, ConfigDict

from fastgen.configs.instantiate import instantiate
from fastgen.datasets.crop_resize import (
    center_crop,
    center_crop_offsets,
    normalize_pixels,
    resize_small_side_aspect_preserving,
)


@pytest.mark.parametrize(
    ("outer_size", "img_size", "expected_offsets"),
    [
        ((11, 13), (8, 8), (2, 2)),  # round(1.5) -> 2; round(2.5) -> 2
        ((15, 11), (8, 8), (4, 2)),  # round(3.5) -> 4
        ((5, 3), (8, 4), (-1, 0)),  # torchvision pads left/top by 1/0
    ],
)
def test_center_crop_matches_torchvision_pixels_and_reported_offsets(outer_size, img_size, expected_offsets):
    outer_w, outer_h = outer_size
    img_w, img_h = img_size
    data = torch.arange(outer_h * outer_w, dtype=torch.int32).reshape(1, outer_h, outer_w)

    actual, params = center_crop(data, img_size, return_cropping_params=True)
    expected = transforms_F.center_crop(data, [img_h, img_w])

    assert center_crop_offsets(outer_w, outer_h, img_size) == expected_offsets
    assert (params["crop_x0"], params["crop_y0"]) == expected_offsets
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_shared_image_transform_is_bitwise_equivalent_to_legacy_loader_pipeline(channels):
    from fastgen.datasets.wds_dataloaders import ImageWDSLoader

    height, width = 7, 11
    pixels = (np.arange(height * width * channels, dtype=np.uint32) * 37 % 256).astype(np.uint8)
    pixels = pixels.reshape(height, width) if channels == 1 else pixels.reshape(height, width, channels)
    image = Image.fromarray(pixels)
    img_size = (8, 6)
    mean = (0.2, 0.4, 0.6)
    std = (0.7, 0.8, 0.9)

    legacy = resize_small_side_aspect_preserving(image, img_size)
    legacy = center_crop(legacy, img_size)
    legacy = normalize_pixels(legacy.convert("RGB"), mean, std)

    loader = object.__new__(ImageWDSLoader)
    loader._input_res = img_size
    loader._norm_mean = mean
    loader._norm_std = std
    assert torch.equal(loader._image_transform(image), legacy)


# =============================================================================
# Dummy Dataset Generation Utilities
# =============================================================================


class DummyImageDatasetBuilder:
    """Builder for creating dummy CIFAR10/ImageNet-style ZIP datasets.

    Creates ZIP files containing images and optional labels (dataset.json).
    Supports both pixel images (.png) and latent representations (.npy).

    Example:
        >>> builder = DummyImageDatasetBuilder("/tmp/test", "cifar10")
        >>> builder.add_image("00000.png", np.random.rand(32, 32, 3), label=0)
        >>> builder.add_image("00001.png", np.random.rand(32, 32, 3), label=1)
        >>> zip_path = builder.build()
    """

    def __init__(self, tmp_dir: str, dataset_name: str = "dummy_dataset"):
        """Initialize the builder.

        Args:
            tmp_dir: Directory to store the zip file
            dataset_name: Name for the zip file (without extension)
        """
        self.tmp_dir = tmp_dir
        self.dataset_name = dataset_name
        self.zip_path = os.path.join(tmp_dir, f"{dataset_name}.zip")
        self.images: List[Tuple[str, np.ndarray]] = []
        self.labels: Dict[str, int] = {}

    def add_image(self, filename: str, image: np.ndarray, label: int | None = None) -> "DummyImageDatasetBuilder":
        """Add an image to the dataset.

        Args:
            filename: Image filename (e.g., "00000.png" or "00000.npy")
            image: Image array. For .png: (H, W, C) uint8, for .npy: (C, H, W) float32
            label: Optional class label (integer)

        Returns:
            self for method chaining
        """
        self.images.append((filename, image))
        if label is not None:
            self.labels[filename] = label
        return self

    def build(self) -> str:
        os.makedirs(self.tmp_dir, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, "w") as zf:
            for filename, image in self.images:
                if filename.endswith(".npy"):
                    buf = io.BytesIO()
                    np.save(buf, image)
                    zf.writestr(filename, buf.getvalue())
                else:
                    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                        image = image.transpose(1, 2, 0)
                    if image.shape[-1] == 1:
                        image = image.squeeze(-1)
                    img = Image.fromarray(image.astype(np.uint8))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    zf.writestr(filename, buf.getvalue())

            if self.labels:
                labels_data = {"labels": [[fname, label] for fname, label in self.labels.items()]}
                zf.writestr("dataset.json", json.dumps(labels_data))
        return self.zip_path

    @classmethod
    def create_cifar10_style(
        cls, tmp_dir: str, num_samples: int = 100, resolution: int = 32, num_classes: int = 10
    ) -> str:
        """Create a CIFAR10-style dataset with random images.

        Args:
            tmp_dir: Directory for the dataset
            num_samples: Number of images to generate
            resolution: Image resolution (square)
            num_classes: Number of classes for labels

        Returns:
            Path to the created ZIP file
        """
        builder = cls(tmp_dir, f"cifar10-{resolution}x{resolution}")
        for i in range(num_samples):
            image = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
            builder.add_image(f"{i:05d}.png", image, i % num_classes)
        return builder.build()

    @classmethod
    def create_imagenet_style(
        cls,
        tmp_dir: str,
        num_samples: int = 100,
        resolution: int = 64,
        num_classes: int = 1000,
        latent: bool = False,
        latent_channels: int = 8,
    ) -> str:
        """Create an ImageNet-style dataset with random images or latents.

        Args:
            tmp_dir: Directory for the dataset
            num_samples: Number of images to generate
            resolution: Image resolution (square)
            num_classes: Number of classes for labels
            latent: If True, create latent representations instead of images
            latent_channels: Number of channels for latent representations

        Returns:
            Path to the created ZIP file
        """
        suffix = "_sd" if latent else ""
        builder = cls(tmp_dir, f"imagenet_{resolution}{suffix}")
        for i in range(num_samples):
            label = num_classes - 1 if i == num_samples - 1 else (i * num_classes) // num_samples
            if latent:
                latent_res = resolution // 8
                image = np.random.randn(latent_channels, latent_res, latent_res).astype(np.float32)
                builder.add_image(f"{i:05d}.npy", image, label)
            else:
                image = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
                builder.add_image(f"{i:05d}.png", image, label)
        return builder.build()


class DummyWebDatasetBuilder:
    """Builder for creating dummy WebDataset tar files for testing.

    Creates tar files containing samples with specified file types and shapes.
    Supports: .npy, .npz, .pth, .jpg, .png, .txt, .json, .mp4, .caption

    Example:
        >>> builder = DummyWebDatasetBuilder("/tmp/test", "00000.tar")
        >>> builder.add_sample("sample_0", {"latents.npy": np.random.randn(4, 64, 64), "txt": "caption"})
        >>> builder.add_sample("sample_1", {"latents.npy": np.random.randn(4, 64, 64), "txt": "caption"})
        >>> tar_path = builder.build()
    """

    NPY_EXTENSIONS = frozenset(
        [
            "npy",
            "latents.npy",
            "sample.npy",
            "noise.npy",
            "text_embedding.npy",
            "pooled_text_embedding.npy",
            "neg_text_embedding.npy",
            "sample_path.npy",
            "text_emb.npy",
            "path.npy",
        ]
    )
    PTH_EXTENSIONS = frozenset(["pth", "latent.pth", "txt_emb.pth", "depth_latent.pth", "noise.pth", "path.pth"])

    def __init__(self, tmp_dir: str, shard_name: str = "00000.tar"):
        self.tmp_dir = tmp_dir
        self.shard_name = shard_name
        self.shard_path = os.path.join(tmp_dir, shard_name)
        self.samples: List[Dict[str, bytes]] = []

    def add_sample(self, key: str, files: Dict[str, Any]) -> "DummyWebDatasetBuilder":
        sample_files = {f"{key}.{ext}": self._encode_data(ext, data) for ext, data in files.items()}
        self.samples.append(sample_files)
        return self

    def _encode_data(self, ext: str, data: Any) -> bytes:
        if ext == "npy" or ext.endswith(".npy"):
            buf = io.BytesIO()
            np.save(buf, data)
            return buf.getvalue()
        elif ext == "npz":
            buf = io.BytesIO()
            np.savez(buf, **data)
            return buf.getvalue()
        elif ext == "pth" or ext.endswith(".pth"):
            buf = io.BytesIO()
            torch.save(data, buf)
            return buf.getvalue()
        elif ext == "json":
            return json.dumps(data).encode("utf-8")
        elif ext in ("txt", "caption"):
            return data.encode("utf-8") if isinstance(data, str) else data
        elif ext in ("jpg", "jpeg"):
            buf = io.BytesIO()
            img = data if isinstance(data, Image.Image) else Image.fromarray(data.astype(np.uint8))
            img.save(buf, format="JPEG")
            return buf.getvalue()
        elif ext == "png":
            buf = io.BytesIO()
            img = data if isinstance(data, Image.Image) else Image.fromarray(data.astype(np.uint8))
            img.save(buf, format="PNG")
            return buf.getvalue()
        elif ext == "mp4":
            if isinstance(data, dict):
                return self._create_dummy_mp4(data["frames"], fps=data.get("fps", 24))
            return data if isinstance(data, bytes) else self._create_dummy_mp4(data)
        else:
            return data if isinstance(data, bytes) else str(data).encode("utf-8")

    def _create_dummy_mp4(self, frames: np.ndarray, fps: int = 24) -> bytes:
        import av

        buf = io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for frame_data in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame_data, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return buf.getvalue()

    @staticmethod
    def create_dummy_av_mp4(
        frames: np.ndarray,
        audio: np.ndarray,
        audio_sr: int,
        video_fps: int = 24,
    ) -> bytes:
        """Build an in-memory mp4 with both video (h264) and audio (aac) streams.

        Args:
            frames: ``(T, H, W, 3)`` uint8 RGB video frames.
            audio: ``(2, num_samples)`` float32 stereo waveform in ``[-1, 1]``.
            audio_sr: Audio sample rate in Hz (must be one of AAC's supported
                rates, e.g. 16000, 22050, 32000, 44100, 48000).
            video_fps: Video frame rate.

        Returns:
            The encoded mp4 as raw bytes.
        """
        import av

        if audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError(f"audio must be (2, samples) stereo float32, got shape {audio.shape}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        buf = io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")

        v_stream = container.add_stream("h264", rate=video_fps)
        v_stream.width, v_stream.height = frames.shape[2], frames.shape[1]
        v_stream.pix_fmt = "yuv420p"

        a_stream = container.add_stream("aac", rate=audio_sr)
        a_stream.layout = "stereo"
        a_stream.format = "fltp"

        # Encode video.
        for frame_data in frames:
            for packet in v_stream.encode(av.VideoFrame.from_ndarray(frame_data, format="rgb24")):
                container.mux(packet)
        for packet in v_stream.encode():
            container.mux(packet)

        # Encode audio in AAC-sized chunks via a FIFO. AAC needs a fixed
        # frame_size (typically 1024) per encoded frame; the FIFO handles
        # the chunking. We feed a single large planar stereo frame in.
        big_frame = av.AudioFrame.from_ndarray(audio, format="fltp", layout="stereo")
        big_frame.sample_rate = audio_sr
        fifo = av.AudioFifo()
        fifo.write(big_frame)
        while True:
            chunk = fifo.read(a_stream.frame_size)
            if chunk is None:
                break
            chunk.sample_rate = audio_sr
            for packet in a_stream.encode(chunk):
                container.mux(packet)
        # Flush remaining samples and the encoder.
        tail = fifo.read()
        if tail is not None and tail.samples > 0:
            tail.sample_rate = audio_sr
            for packet in a_stream.encode(tail):
                container.mux(packet)
        for packet in a_stream.encode():
            container.mux(packet)

        container.close()
        return buf.getvalue()

    def build(self) -> str:
        os.makedirs(os.path.dirname(self.shard_path) or ".", exist_ok=True)
        with tarfile.open(self.shard_path, "w") as tar:
            for sample_files in self.samples:
                for filename, data in sample_files.items():
                    info = tarfile.TarInfo(name=filename)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        return self.shard_path

    @classmethod
    def create_dataset_dir(
        cls, tmp_dir: str, num_shards: int, samples_per_shard: int, file_specs: Dict[str, Any]
    ) -> str:
        """Create a complete WebDataset directory with multiple shards.

        Args:
            tmp_dir: Base directory for the dataset
            num_shards: Number of tar shards to create
            samples_per_shard: Number of samples per shard
            file_specs: Dict mapping extension -> shape/value specification
                        For arrays: tuple of shape (e.g., (4, 64, 64))
                        For json: dict template or callable(shard_idx, sample_idx)
                        For txt/caption: str template or callable(shard_idx, sample_idx)
                        For mp4: (T, H, W) tuple for frame dimensions

        Returns:
            Path to the dataset directory

        Example:
            >>> dataset_dir = DummyWebDatasetBuilder.create_dataset_dir(
            ...     "/tmp", num_shards=2, samples_per_shard=4,
            ...     file_specs={"latents.npy": (4, 64, 64), "txt": lambda s, i: f"Sample {s}_{i}"}
            ... )
        """
        dataset_dir = os.path.join(tmp_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        for shard_idx in range(num_shards):
            builder = cls(dataset_dir, f"{shard_idx:05d}.tar")
            for sample_idx in range(samples_per_shard):
                key = f"{shard_idx:05d}_{sample_idx:05d}"
                files = {}
                for ext, spec in file_specs.items():
                    if ext in cls.NPY_EXTENSIONS:
                        files[ext] = np.random.randn(*spec).astype(np.float32)
                    elif ext in cls.PTH_EXTENSIONS:
                        files[ext] = torch.randn(*spec)
                    elif ext == "npz":
                        files[ext] = {k: np.random.randn(*v).astype(np.float32) for k, v in spec.items()}
                    elif ext == "json":
                        files[ext] = (
                            spec(shard_idx, sample_idx)
                            if callable(spec)
                            else (spec.copy() if isinstance(spec, dict) else spec)
                        )
                    elif ext in ("txt", "caption"):
                        files[ext] = spec(shard_idx, sample_idx) if callable(spec) else spec
                    elif ext in ("jpg", "jpeg", "png"):
                        h, w = spec[:2]
                        files[ext] = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                    elif ext == "mp4":
                        if isinstance(spec, (bytes, bytearray)):
                            files[ext] = spec
                        elif isinstance(spec, dict):
                            files[ext] = spec
                        else:
                            t, h, w = spec
                            files[ext] = np.random.randint(0, 255, (t, h, w, 3), dtype=np.uint8)
                    else:
                        files[ext] = spec
                builder.add_sample(key, files)
            builder.build()
        return dataset_dir


def create_shard_count_file(dataset_dir: str, samples_per_shard: int) -> str:
    """Create a shard count JSON file for deterministic loading."""
    shard_count = {f: samples_per_shard for f in os.listdir(dataset_dir) if f.endswith(".tar")}
    count_file = os.path.join(dataset_dir, "shard_count.json")
    with open(count_file, "w", encoding="utf-8") as f:
        json.dump(shard_count, f)
    return count_file


def create_ignore_index_file(dataset_dir: str, ignore_spec: Dict[str, List[str]]) -> str:
    """Create an ignore index JSON file."""
    ignore_file = os.path.join(dataset_dir, "ignore_index.json")
    with open(ignore_file, "w", encoding="utf-8") as f:
        json.dump(ignore_spec, f)
    return ignore_file


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tmp_dataset_dir():
    """Create a temporary directory for test datasets."""
    tmp_dir = tempfile.mkdtemp(prefix="fastgen_test_")
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_batch_keys(batch: Dict, expected_keys: List[str], allow_extra: bool = False):
    """Assert that batch contains expected keys."""
    for key in expected_keys:
        assert key in batch, f"Missing expected key: {key}"
    if not allow_extra:
        for key in batch:
            if not key.startswith("__"):
                assert key in expected_keys, f"Unexpected key in batch: {key}"


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str = "tensor"):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"


def assert_batch_type(batch: Dict, key: str, expected_type: Type):
    """Assert that batch[key] has expected type using pydantic TypeAdapter.

    This function validates types including nested types like List[str] and Dict[str, Any]; with
    ``arbitrary_types_allowed`` pydantic checks classes it has no schema for, such as torch.Tensor, by isinstance.

    Args:
        batch: The batch dictionary
        key: The key to check
        expected_type: The expected type (can be a generic type like List[str])
    """
    value = batch[key]

    try:
        adapter = TypeAdapter(expected_type, config=ConfigDict(arbitrary_types_allowed=True))
        adapter.validate_python(value)
    except ValidationError as e:
        raise AssertionError(f"batch['{key}'] type validation failed for {expected_type}: {e}") from e


def make_constant_video_frames(values: np.ndarray, height: int = 16, width: int = 16) -> np.ndarray:
    """Create RGB video frames where each frame has a single scalar value."""
    frames = np.repeat(values[:, None, None, None], height, axis=1)
    frames = np.repeat(frames, width, axis=2)
    return np.repeat(frames, 3, axis=3)


def _assert_zip_handle_is_owned_by_worker(_worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info is not None
    dataset = worker_info.dataset
    inherited_zipfile = dataset._zipfile
    inherited_owner_pid = dataset._zipfile_owner_pid

    worker_zipfile = dataset._get_zipfile()

    assert dataset._zipfile_owner_pid == os.getpid()
    if inherited_zipfile is not None and inherited_owner_pid != os.getpid():
        assert worker_zipfile is not inherited_zipfile


# =============================================================================
# Test Specification Classes
# =============================================================================


@dataclass
class LoaderTestSpec:
    """Specification for a loader test.

    The expected keys are derived from expected_types.keys(), so you don't need
    to specify them separately. Multiple configs with the same specs can be
    tested together by providing a list of config_paths.

    The test will
    run twice: once with dummy data (always), once with real data (skipped by default).
    """

    name: str
    config_paths: List[str]  # e.g., ["fastgen.configs.data.VideoLatentLoaderConfig"]
    file_specs: Dict[str, Any]
    expected_types: Dict[str, type]  # key -> expected type (keys define expected batch keys)
    expected_shapes: Dict[str, Tuple] = field(default_factory=dict)  # key -> expected shape
    num_shards: int = 1
    samples_per_shard: int = 4
    batch_size: int = 2
    extra_config: Dict[str, Any] = field(default_factory=dict)
    extra_files: Optional[Callable[[str], Dict[str, str]]] = None  # Creates extra files, returns config updates
    # Real data integration testing (optional - skipped by default)
    credentials_dir: str = "./credentials"  # Folder with s3.json for S3


# =============================================================================
# Generic Test Runners
# =============================================================================


def run_loader_test(tmp_dir: str, spec: LoaderTestSpec, use_real_data: bool = False):
    """Run a generic loader test from specification.

    Tests all configs in spec.config_paths with either dummy or real data.

    Args:
        tmp_dir: Temporary directory for dummy dataset
        spec: Test specification
        use_real_data: If True, use real_datatags from spec instead of dummy data
    """
    if use_real_data:
        # Real data integration test
        from fastgen.utils.env import set_env_vars

        set_env_vars(spec.credentials_dir)
        # ``extra_config`` is plain loader configuration and applies to both branches; specs
        # sharing one config template rely on it to select their own resolution. Only
        # ``extra_files`` is dummy-data specific.
        extra_config_updates = dict(spec.extra_config)
        datatags = None
    else:
        # Dummy data test
        dataset_dir = DummyWebDatasetBuilder.create_dataset_dir(
            tmp_dir, spec.num_shards, spec.samples_per_shard, spec.file_specs
        )
        datatags = [f"WDS:{dataset_dir}"]
        extra_config_updates = spec.extra_files(dataset_dir) if spec.extra_files else {}
        extra_config_updates.update(spec.extra_config)

    # Test each config path
    for config_path in spec.config_paths:
        # Import config dynamically
        module_path, config_name = config_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        config_class = getattr(module, config_name)

        # Configure loader
        config = config_class.copy()
        if datatags is not None:
            config.datatags = datatags
        config.batch_size = spec.batch_size
        config.num_workers = 0
        for k, v in extra_config_updates.items():
            setattr(config, k, v)

        # Get batch
        loader = instantiate(config)
        batch = next(iter(loader))

        # Assertions - expected keys derived from expected_types
        assert_batch_keys(batch, list(spec.expected_types.keys())), f"Failed for {config_name}"
        for key, expected_type in spec.expected_types.items():
            assert_batch_type(batch, key, expected_type)
        for key, expected_shape in spec.expected_shapes.items():
            assert_tensor_shape(batch[key], expected_shape, key)


# =============================================================================
# ImageLoader Tests
# =============================================================================


@dataclass
class ImageLoaderTestSpec:
    """Specification for an ImageLoader test.

    Similar to LoaderTestSpec but for class-conditional image datasets (CIFAR10, ImageNet).
    The expected keys are derived from expected_types.keys().
    """

    name: str
    config_path: str  # e.g., "fastgen.configs.data.CIFAR10_Loader_Config"
    expected_types: Dict[str, type]  # key -> expected type
    expected_shapes: Dict[str, Tuple]  # key -> expected shape (with batch_size)
    num_samples: int = 64
    batch_size: int = 8
    latent_channels: Optional[int] = None  # If not None, used for latent-based dummy data generation
    resolution: int = 64  # Used for dummy data generation
    num_classes: int = 1000  # Used for dummy data generation
    credentials_dir: str = "./credentials"


# Test specifications for ImageLoaders
IMAGE_LOADER_SPECS = [
    ImageLoaderTestSpec(
        name="cifar10",
        config_path="fastgen.configs.data.CIFAR10_Loader_Config",
        expected_types={
            "real_raw": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "idx": torch.Tensor,
        },
        expected_shapes={
            "real_raw": (8, 3, 32, 32),
            "condition": (8, 10),
            "neg_condition": (8, 10),
            "idx": (8,),
        },
        resolution=32,
        num_classes=10,
    ),
    ImageLoaderTestSpec(
        name="imagenet64",
        config_path="fastgen.configs.data.ImageNet64_Loader_Config",
        expected_types={
            "real_raw": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "idx": torch.Tensor,
        },
        expected_shapes={
            "real_raw": (8, 3, 64, 64),
            "condition": (8, 1000),
            "neg_condition": (8, 1000),
            "idx": (8,),
        },
        resolution=64,
        num_classes=1000,
    ),
    ImageLoaderTestSpec(
        name="imagenet64_edmv2",
        config_path="fastgen.configs.data.ImageNet64_EDMV2_Loader_Config",
        expected_types={
            "real_raw": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "idx": torch.Tensor,
        },
        expected_shapes={
            "real_raw": (8, 3, 64, 64),
            "condition": (8, 1000),
            "neg_condition": (8, 1000),
            "idx": (8,),
        },
        resolution=64,
        num_classes=1000,
    ),
    ImageLoaderTestSpec(
        name="imagenet256_latent",
        config_path="fastgen.configs.data.ImageNet256_Loader_Config",
        expected_types={
            "real_raw": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "idx": torch.Tensor,
        },
        expected_shapes={
            "real_raw": (8, 8, 32, 32),
            "condition": (8, 1000),
            "neg_condition": (8, 1000),
            "idx": (8,),
        },
        resolution=256,
        num_classes=1000,
        latent_channels=8,
    ),
]


def run_image_loader_test(tmp_dir: str, spec: ImageLoaderTestSpec, use_real_data: bool = False):
    """Run an ImageLoader test from specification.

    Args:
        tmp_dir: Temporary directory for dummy dataset
        spec: Test specification
        use_real_data: If True, use real data paths from config instead of dummy data
    """
    # Import config dynamically
    module_path, config_name = spec.config_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    config_class = getattr(module, config_name)

    config = config_class.copy()
    config.batch_size = spec.batch_size
    config.sampler_start_idx = 0

    if use_real_data:
        # Real data integration test - use paths from config
        from fastgen.utils.env import set_env_vars

        set_env_vars(spec.credentials_dir)
    else:
        # Dummy data test - create synthetic dataset
        if spec.latent_channels is not None:
            zip_path = DummyImageDatasetBuilder.create_imagenet_style(
                tmp_dir,
                spec.num_samples,
                spec.resolution,
                spec.num_classes,
                latent=True,
                latent_channels=spec.latent_channels,
            )
        elif spec.num_classes == 10:  # CIFAR10-style
            zip_path = DummyImageDatasetBuilder.create_cifar10_style(
                tmp_dir, spec.num_samples, spec.resolution, spec.num_classes
            )
        else:  # ImageNet-style
            zip_path = DummyImageDatasetBuilder.create_imagenet_style(
                tmp_dir, spec.num_samples, spec.resolution, spec.num_classes
            )
        config.dataset_path = zip_path
        config.remote_path = None

    # Get batch
    loader = instantiate(config)
    batch = next(iter(loader))

    # Assert keys (derived from expected_types)
    expected_keys = list(spec.expected_types.keys())
    assert_batch_keys(batch, expected_keys)

    # Assert types
    for key, expected_type in spec.expected_types.items():
        assert_batch_type(batch, key, expected_type)

    # Shape checks (only for dummy data - real data shapes depend on config batch_size)
    if not use_real_data:
        for key, expected_shape in spec.expected_shapes.items():
            assert_tensor_shape(batch[key], expected_shape, key)
        if spec.latent_channels is None:
            assert batch["real_raw"].min() >= -1.0 and batch["real_raw"].max() <= 1.0
    else:
        # For real data, just check that shapes are consistent
        assert batch["condition"].shape[1] == spec.num_classes, f"Expected {spec.num_classes} classes"


class TestImageLoader:
    """Tests for ImageLoader with CIFAR10 and ImageNet-style datasets."""

    def test_missing_dataset_without_a_mirror_names_the_path_and_how_to_prepare_it(self, tmp_path):
        """Without a remote mirror (``remote_path=None``), a missing ZIP points at the data preparation."""
        from fastgen.datasets.class_cond_dataloader import ImageLoader

        missing = tmp_path / "cifar10" / "cifar10-32x32.zip"
        with pytest.raises(IOError) as error:
            ImageLoader(dataset_path=str(missing), remote_path=None, batch_size=1)

        assert str(missing) in str(error.value)
        assert "DATA_ROOT_DIR" in str(error.value) and "scripts/download_data.py" in str(error.value)

    def test_missing_local_zip_uses_shared_remote_cache(self, tmp_path, monkeypatch):
        import fastgen.datasets.class_cond_dataset as dataset_module

        source_path = DummyImageDatasetBuilder.create_imagenet_style(
            str(tmp_path / "source"), num_samples=4, resolution=32, num_classes=10, latent=True
        )
        with open(source_path, "rb") as source_file:
            payload = source_file.read()

        remote_path = "s3://datasets/unit/remote.zip"
        downloads = []

        def fake_s3_download(path, output):
            downloads.append(path)
            output.write(payload)
            return 0

        cache_calls = []
        real_cache = dataset_module.cache

        def recording_cache(path, strict):
            cache_calls.append((path, strict))
            return real_cache(path, strict=strict)

        monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", str(tmp_path / "output"))
        monkeypatch.setattr("fastgen.utils.cache_utils.s3.resumable_download", fake_s3_download)
        monkeypatch.setattr(dataset_module, "cache", recording_cache)

        requested_path = tmp_path / "configured" / "local.zip"
        expected_path = tmp_path / "output" / "MODEL" / "datasets" / "unit" / "remote.zip"
        dataset = dataset_module.ImageFolderDataset(
            path=str(requested_path),
            remote_path=remote_path,
            use_labels=True,
        )
        try:
            assert dataset._path == str(expected_path)
            assert expected_path.is_file()
            assert not requested_path.exists()
            assert "real_raw" in dataset[0]
        finally:
            dataset.close()

        assert cache_calls == [(remote_path, True)]
        assert downloads == [remote_path]

    def test_missing_local_zip_downloads_http_remote_with_query(self, tmp_path, monkeypatch):
        import fastgen.datasets.class_cond_dataset as dataset_module
        from fastgen.utils.cache_utils import mirror_path

        source_path = DummyImageDatasetBuilder.create_imagenet_style(
            str(tmp_path / "source"), num_samples=4, resolution=32, num_classes=10, latent=True
        )
        with open(source_path, "rb") as source_file:
            payload = source_file.read()

        remote_path = "https://example.test/datasets/remote.zip?download=1"
        downloads = []

        def fake_http_download(path, output):
            downloads.append(path)
            output.write(payload)

        monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", str(tmp_path / "output"))
        monkeypatch.setattr("fastgen.utils.cache_utils._download_http", fake_http_download)

        requested_path = tmp_path / "configured" / "local.zip"
        cache_root = tmp_path / "output" / "MODEL"
        expected_path = mirror_path(remote_path, str(cache_root))
        dataset = dataset_module.ImageFolderDataset(
            path=str(requested_path),
            remote_path=remote_path,
            use_labels=True,
        )
        try:
            assert dataset._path == expected_path
            assert os.path.isfile(expected_path)
            assert not requested_path.exists()
            assert "real_raw" in dataset[0]
        finally:
            dataset.close()

        assert downloads == [remote_path]

    def test_zip_handle_reopens_when_process_owner_changes(self, tmp_dataset_dir):
        from fastgen.datasets.class_cond_dataset import ImageFolderDataset

        zip_path = DummyImageDatasetBuilder.create_imagenet_style(
            tmp_dataset_dir, num_samples=4, resolution=32, num_classes=10, latent=True
        )
        dataset = ImageFolderDataset(path=zip_path, use_labels=True)
        inherited_zipfile = dataset._get_zipfile()
        owner_pid = dataset._zipfile_owner_pid

        dataset._zipfile_owner_pid = owner_pid + 1
        replacement_zipfile = dataset._get_zipfile()

        assert replacement_zipfile is not inherited_zipfile
        assert inherited_zipfile.fp is None
        assert dataset._zipfile_owner_pid == os.getpid()
        assert "real_raw" in dataset[0]

    def test_state_key_can_emit_materialized_model_data(self, tmp_dataset_dir):
        from fastgen.datasets.class_cond_dataset import ImageFolderDataset

        zip_path = DummyImageDatasetBuilder.create_imagenet_style(
            tmp_dataset_dir, num_samples=4, resolution=32, num_classes=10, latent=True
        )
        dataset = ImageFolderDataset(path=zip_path, use_labels=True, state_key="real")

        item = dataset[0]

        assert "real" in item
        assert "real_raw" not in item

    def test_zip_handle_is_process_local_across_dataloader_iterators(self, tmp_dataset_dir):
        from fastgen.datasets.class_cond_dataset import ImageFolderDataset

        num_samples = 4
        zip_path = DummyImageDatasetBuilder.create_imagenet_style(
            tmp_dataset_dir, num_samples=num_samples, resolution=32, num_classes=10, latent=True
        )
        dataset = ImageFolderDataset(path=zip_path, use_labels=True, cache=True)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            worker_init_fn=_assert_zip_handle_is_owned_by_worker,
        )

        for _ in range(2):
            batches = list(loader)
            assert sum(batch["real_raw"].shape[0] for batch in batches) == num_samples

    def test_image_loader_supports_multiple_workers(self, tmp_dataset_dir):
        from fastgen.datasets.class_cond_dataloader import ImageLoader

        num_samples = 8
        batch_size = 2
        zip_path = DummyImageDatasetBuilder.create_imagenet_style(
            tmp_dataset_dir, num_samples=num_samples, resolution=32, num_classes=10, latent=True
        )
        loader = ImageLoader(
            dataset_path=zip_path,
            remote_path=None,
            batch_size=batch_size,
            use_labels=True,
            cache=True,
            shuffle=False,
            sampler_start_idx=0,
            num_workers=2,
        )

        iterator = iter(loader)
        batches = [next(iterator) for _ in range(num_samples // batch_size)]
        del iterator

        assert loader.loader.num_workers == 2
        assert torch.cat([batch["idx"] for batch in batches]).tolist() == list(range(num_samples))

    @pytest.mark.parametrize("spec", IMAGE_LOADER_SPECS, ids=lambda s: s.name)
    def test_image_loader(self, tmp_dataset_dir, spec):
        """Test image loaders with dummy datasets."""
        run_image_loader_test(tmp_dataset_dir, spec, use_real_data=False)

    @pytest.mark.parametrize("spec", IMAGE_LOADER_SPECS, ids=lambda s: s.name)
    @pytest.mark.integration
    def test_image_loader_real_data(self, tmp_dataset_dir, spec):
        """Test image loaders with real data (integration test)."""
        run_image_loader_test(tmp_dataset_dir, spec, use_real_data=True)

    def test_sampler_start_idx_resume(self, tmp_dataset_dir):
        """Test that sampler_start_idx correctly resumes from a given position."""
        from fastgen.configs.data import CIFAR10_Loader_Config

        zip_path = DummyImageDatasetBuilder.create_cifar10_style(tmp_dataset_dir, 32, 32, 10)

        def create_loader(start_idx):
            config = CIFAR10_Loader_Config.copy()
            config.dataset_path = zip_path
            config.remote_path = None
            config.batch_size = 4
            config.sampler_start_idx = start_idx
            config.shuffle = True
            return instantiate(config)

        # Get first 3 batches from start
        loader_start = iter(create_loader(0))
        batches = [next(loader_start)["idx"].tolist() for _ in range(3)]

        # Resume from index 8 should match 3rd batch
        loader_resume = iter(create_loader(8))
        resumed_batch = next(loader_resume)["idx"].tolist()

        assert resumed_batch == batches[2], f"Resume mismatch: expected {batches[2]}, got {resumed_batch}"

    def test_samples_unique_within_epoch(self, tmp_dataset_dir):
        """Test that all samples within one epoch are unique."""
        from fastgen.configs.data import CIFAR10_Loader_Config

        num_samples, batch_size = 32, 4
        zip_path = DummyImageDatasetBuilder.create_cifar10_style(tmp_dataset_dir, num_samples, 32, 10)

        config = CIFAR10_Loader_Config.copy()
        config.dataset_path = zip_path
        config.remote_path = None
        config.batch_size = batch_size
        config.sampler_start_idx = 0
        config.shuffle = True

        loader = instantiate(config)
        all_indices = []
        for i, batch in enumerate(loader):
            if i >= num_samples // batch_size:
                break
            all_indices.extend(batch["idx"].tolist())

        assert len(all_indices) == num_samples
        assert len(set(all_indices)) == num_samples, "Found duplicate samples in epoch"

    def test_samples_unique_across_resumed_training(self, tmp_dataset_dir):
        """Test that resumed training doesn't repeat samples."""
        from fastgen.configs.data import CIFAR10_Loader_Config

        num_samples, batch_size = 32, 4
        zip_path = DummyImageDatasetBuilder.create_cifar10_style(tmp_dataset_dir, num_samples, 32, 10)

        def create_loader(start_idx):
            config = CIFAR10_Loader_Config.copy()
            config.dataset_path = zip_path
            config.remote_path = None
            config.batch_size = batch_size
            config.sampler_start_idx = start_idx
            config.shuffle = False
            return instantiate(config)

        # First half
        loader1 = create_loader(0)
        first_half = []
        for i, batch in enumerate(loader1):
            if i >= num_samples // (2 * batch_size):
                break
            first_half.extend(batch["idx"].tolist())

        # Second half (resumed)
        loader2 = create_loader(len(first_half))
        second_half = []
        for i, batch in enumerate(loader2):
            if i >= num_samples // (2 * batch_size):
                break
            second_half.extend(batch["idx"].tolist())

        assert set(first_half).isdisjoint(set(second_half)), "Found overlapping samples"


# =============================================================================
# WDSLoader Tests
# =============================================================================


def _create_neg_prompt_file(dataset_dir: str, shape: Tuple[int, ...]) -> Dict[str, str]:
    """Create a neg_prompt_emb.npy file for testing files_map loading.

    Args:
        dataset_dir: Directory to create the file in
        shape: Shape of the tensor to create

    Returns:
        Dict with files_map config update pointing to the created file
    """
    neg_prompt_path = os.path.join(dataset_dir, "neg_prompt_emb.npy")
    np.save(neg_prompt_path, np.random.randn(*shape).astype(np.float32))
    return {"files_map": {"neg_condition": neg_prompt_path}}


# Test specifications for WDS loaders (generic template configs)
WDS_LOADER_SPECS = [
    # ImageWDSLoader - generic image loader with jpg + txt
    LoaderTestSpec(
        name="image_wds",
        config_paths=["fastgen.configs.data.ImageLoaderConfig"],
        file_specs={
            "jpg": (512, 512),
            "txt": lambda s, i: f"A sample image {s}_{i}",
        },
        expected_types={
            "real_raw": torch.Tensor,
            "condition_raw": List[str],
            "neg_condition_raw": List[str],
            "idx": List[str],
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={"real_raw": (2, 3, 512, 512)},
        extra_config={"input_res": 512},
    ),
    # ImageWDSLoader - non-square source (16:9) with different non-square target (4:3)
    LoaderTestSpec(
        name="image_wds_nonsquare_different_ar",
        config_paths=["fastgen.configs.data.ImageLoaderConfig"],
        file_specs={
            "jpg": (480, 832),
            "txt": lambda s, i: f"A sample image {s}_{i}",
        },
        expected_types={
            "real_raw": torch.Tensor,
            "condition_raw": List[str],
            "neg_condition_raw": List[str],
            "idx": List[str],
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={"real_raw": (2, 3, 384, 512)},
        extra_config={"input_res": (512, 384)},
    ),
    # VideoWDSLoader - generic video loader with mp4 + txt
    LoaderTestSpec(
        name="video_wds",
        config_paths=["fastgen.configs.data.VideoLoaderConfig"],
        file_specs={
            "mp4": (90, 480, 832),
            "txt": lambda s, i: f"A sample video {s}_{i}",
        },
        expected_types={
            "real_raw": torch.Tensor,
            "condition_raw": List[str],
            "neg_condition_raw": List[str],
            "cropping_params": Dict[str, Any],
            "segment_start_frame": torch.Tensor,
            "idx": List[str],
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={"real_raw": (1, 3, 81, 480, 832)},  # (B, C, T, H, W)
        batch_size=1,
        samples_per_shard=2,
        extra_config={"sequence_length": 81, "img_size": (832, 480)},
    ),
    # WDSLoader - generic image latent loader with latent.pth + txt_emb.pth + neg_condition from files_map
    LoaderTestSpec(
        name="image_latent_wds",
        config_paths=["fastgen.configs.data.ImageLatentLoaderConfig"],
        file_specs={
            "latent.pth": (4, 128, 128),
            "txt_emb.pth": (77, 2048),
        },
        expected_types={
            "real": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "idx": List[str],
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={
            "real": (2, 4, 128, 128),
            "condition": (2, 77, 2048),
            "neg_condition": (2, 77, 2048),
        },
        extra_files=lambda d: _create_neg_prompt_file(d, (77, 2048)),
        batch_size=2,
    ),
    # WDSLoader - generic video latent loader with latent.pth + txt_emb.pth + neg_condition from files_map
    LoaderTestSpec(
        name="video_latent_wds",
        config_paths=["fastgen.configs.data.VideoLatentLoaderConfig"],
        file_specs={
            "latent.pth": (16, 21, 60, 104),
            "txt_emb.pth": (512, 4096),
        },
        expected_types={
            "real": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "idx": List[str],
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={
            "real": (2, 16, 21, 60, 104),
            "condition": (2, 512, 4096),
            "neg_condition": (2, 512, 4096),
        },
        extra_files=lambda d: _create_neg_prompt_file(d, (512, 4096)),
    ),
    # PairLoaderConfig - for single-step KD with (real, noise, condition)
    # Data requirements from KD.py: {"real": clean, "noise": noise, "condition": cond}
    LoaderTestSpec(
        name="pair_wds",
        config_paths=["fastgen.configs.data.PairLoaderConfig"],
        file_specs={
            "latent.pth": (16, 21, 60, 104),
            "noise.pth": (16, 21, 60, 104),
            "txt_emb.pth": (512, 4096),
        },
        expected_types={
            "real": torch.Tensor,
            "noise": torch.Tensor,
            "condition": torch.Tensor,
            "idx": List[str],
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={
            "real": (2, 16, 21, 60, 104),
            "noise": (2, 16, 21, 60, 104),
            "condition": (2, 512, 4096),
        },
    ),
    # PathLoaderConfig - for multi-step KD with (real, path, condition)
    # Data requirements from KD.py: {"real": clean, "path": [B, steps, C, ...], "condition": cond}
    # path shape: [B, num_inf_steps=4, C, ...]
    LoaderTestSpec(
        name="path_wds",
        config_paths=["fastgen.configs.data.PathLoaderConfig"],
        file_specs={
            "latent.pth": (16, 21, 60, 104),
            "path.pth": (4, 16, 21, 60, 104),  # [num_inf_steps=4, C, T, H, W]
            "txt_emb.pth": (512, 4096),
        },
        expected_types={
            "real": torch.Tensor,
            "path": torch.Tensor,
            "condition": torch.Tensor,
            "idx": List[str],
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={
            "real": (2, 16, 21, 60, 104),
            "path": (2, 4, 16, 21, 60, 104),  # [B, num_inf_steps=4, C, T, H, W]
            "condition": (2, 512, 4096),
        },
    ),
]


class TestWDSLoader:
    """Tests for WDSLoader.

    Each test runs with dummy data by default. A second test variant runs with real data
    (skipped by default, enable with pytest --run-integration).
    """

    @pytest.mark.parametrize("spec", WDS_LOADER_SPECS, ids=lambda s: s.name)
    def test_wds_loader(self, tmp_dataset_dir, spec):
        """Test WDS loaders with dummy datasets."""
        run_loader_test(tmp_dataset_dir, spec, use_real_data=False)

    @pytest.mark.parametrize("spec", WDS_LOADER_SPECS, ids=lambda s: s.name)
    @pytest.mark.integration
    def test_wds_loader_real_data(self, tmp_dataset_dir, spec):
        """Test WDS loaders with real data (integration test)."""
        run_loader_test(tmp_dataset_dir, spec, use_real_data=True)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_optional_files(self, tmp_dataset_dir):
        """Test handling of missing optional files."""
        from fastgen.datasets.wds_dataloaders import WDSLoader

        dataset_dir = DummyWebDatasetBuilder.create_dataset_dir(
            tmp_dataset_dir, 1, 4, {"latents.npy": (16, 64, 64), "text_embedding.npy": (512, 4096)}
        )

        loader = WDSLoader(
            datatags=[f"WDS:{dataset_dir}"],
            batch_size=2,
            key_map={"real": "latents.npy", "condition": "text_embedding.npy"},
            presets_map={"neg_condition_raw": "empty_string"},
            num_workers=0,
        )

        batch = next(iter(loader))
        assert_batch_keys(batch, ["real", "condition", "neg_condition_raw", "idx", "fname", "shard"])
        assert_batch_type(batch, "neg_condition_raw", List[str])
        assert all(c == "" for c in batch["neg_condition_raw"])

    def test_partial_files_filtering(self, tmp_dataset_dir):
        """Test that samples with missing required files are filtered out."""
        from fastgen.datasets.wds_dataloaders import WDSLoader

        partial_dir = os.path.join(tmp_dataset_dir, "partial")
        os.makedirs(partial_dir, exist_ok=True)
        builder = DummyWebDatasetBuilder(partial_dir, "00000.tar")

        # Add complete and incomplete samples (indices 0,2 have text, 1,3 don't)
        for i, has_text in enumerate([True, False, True, False]):
            key = f"00000_{i:05d}"
            sample = {f"{key}.latents.npy": builder._encode_data("npy", np.random.randn(16, 64, 64).astype(np.float32))}
            if has_text:
                sample[f"{key}.text_embedding.npy"] = builder._encode_data(
                    "npy", np.random.randn(512, 4096).astype(np.float32)
                )
            builder.samples.append(sample)
        builder.build()

        loader = WDSLoader(
            datatags=[f"WDS:{partial_dir}"],
            batch_size=2,
            key_map={"real": "latents.npy", "condition": "text_embedding.npy"},
            num_workers=0,
            train=False,
        )

        batch = next(iter(loader))
        # check that only the two complete samples are loaded based on fname
        assert batch["fname"] == ["00000_00000", "00000_00002"]

    def test_ignore_index_filtering(self, tmp_dataset_dir):
        """Test ignore index filtering."""
        from fastgen.datasets.wds_dataloaders import WDSLoader

        dataset_dir = DummyWebDatasetBuilder.create_dataset_dir(tmp_dataset_dir, 1, 4, {"latents.npy": (16, 64, 64)})

        ignore_file = create_ignore_index_file(dataset_dir, {"00000.tar": ["00000_00000", "00000_00001"]})

        loader = WDSLoader(
            datatags=[f"WDS:{dataset_dir}"],
            batch_size=2,
            key_map={"real": "latents.npy"},
            ignore_index_paths=[ignore_file],
            num_workers=0,
        )

        batch = next(iter(loader))
        assert_batch_type(batch, "fname", List[str])
        assert "00000_00000" not in batch["fname"]
        assert "00000_00001" not in batch["fname"]

    def test_score_filter_drops_unscorable_samples_instead_of_stopping_the_loader(self, tmp_dataset_dir, monkeypatch):
        """Unparseable metadata or a score that is not a finite number drops a sample with a warning, like an
        undecodable one; a score below the threshold drops it silently, and a sample without a score is kept."""
        from fastgen.datasets.wds_dataloaders import WDSLoader

        warnings = []
        monkeypatch.setattr("fastgen.datasets.wds_dataloaders.logger.warning", warnings.append)
        dataset_dir = os.path.join(tmp_dataset_dir, "scored")
        os.makedirs(dataset_dir, exist_ok=True)
        builder = DummyWebDatasetBuilder(dataset_dir, "00000.tar")
        metadata = [
            '{"score": 7}',
            '{"score": null}',
            '{"score": 2}',
            None,
            '{"other": 1}',
            "{broken",
            '{"score": "x"}',
            '{"score": NaN}',
            '{"score": 1' + "0" * 400 + "}",
            '{"score": true}',
            '{"score": Infinity}',
        ]
        for i, meta in enumerate(metadata):
            key = f"00000_{i:05d}"
            sample = {f"{key}.latents.npy": builder._encode_data("npy", np.zeros((2, 2), dtype=np.float32))}
            if meta is not None:
                sample[f"{key}.json"] = meta.encode()
            builder.samples.append(sample)
        builder.build()

        loader = WDSLoader(
            datatags=[f"WDS:{dataset_dir}"],
            batch_size=1,
            key_map={"real": "latents.npy"},
            presets_filter={"score": {"threshold": 5.0}},
            num_workers=0,
            train=False,
        )

        assert [batch["fname"][0] for batch in loader] == ["00000_00000", "00000_00003", "00000_00004", "00000_00008"]
        keys = [f"00000_{i:05d}" for i in range(len(metadata))]
        warned = [key for key in keys if any(key in warning for warning in warnings)]
        assert warned == ["00000_00001", "00000_00005", "00000_00006", "00000_00007", "00000_00009", "00000_00010"]

    @pytest.mark.parametrize("threshold", [None, "5.0"])
    def test_score_filter_raises_on_a_non_numeric_threshold(self, tmp_dataset_dir, threshold):
        """A non-numeric threshold is a config error: it stops the loader instead of dropping all scored samples."""
        from fastgen.datasets.wds_dataloaders import WDSLoader

        dataset_dir = DummyWebDatasetBuilder.create_dataset_dir(
            tmp_dataset_dir, 1, 2, {"latents.npy": (2, 2), "json": {"score": 7}}
        )
        loader = WDSLoader(
            datatags=[f"WDS:{dataset_dir}"],
            batch_size=1,
            key_map={"real": "latents.npy"},
            presets_filter={"score": {"threshold": threshold}},
            num_workers=0,
            train=False,
        )

        with pytest.raises(TypeError):
            next(iter(loader))

    def test_preset_filter_exclusions_do_not_count_as_failed_samples(self, tmp_dataset_dir):
        """More presets_filter exclusions in a row than MAX_CONSECUTIVE_SKIPS do not stop the loader."""
        from fastgen.datasets.wds_dataloaders import WDSLoader
        from fastgen.datasets.wds_utils import MAX_CONSECUTIVE_SKIPS

        builder = DummyWebDatasetBuilder(tmp_dataset_dir)
        for i in range(MAX_CONSECUTIVE_SKIPS + 1):
            builder.add_sample(f"{i:05d}", {"txt": "kept" if i == MAX_CONSECUTIVE_SKIPS else ""})
        builder.build()
        loader = WDSLoader(
            datatags=[f"WDS:{tmp_dataset_dir}"],
            batch_size=1,
            key_map={"condition_raw": "txt"},
            presets_filter={"empty_keys": {"keys": ["txt"]}},
            num_workers=0,
            train=False,
        )

        assert [batch["fname"] for batch in loader] == [[f"{MAX_CONSECUTIVE_SKIPS:05d}"]]

    @pytest.mark.parametrize("bad_target_fps", [0, -1, float("nan"), float("inf")])
    def test_video_wds_loader_rejects_nonfinite_target_fps(self, bad_target_fps):
        """``VideoWDSLoader`` validates ``target_fps`` up-front. Decoder-level validation
        is covered in ``test_decoders.py``."""
        from fastgen.datasets.wds_dataloaders import VideoWDSLoader

        with pytest.raises(ValueError, match="finite positive"):
            VideoWDSLoader(
                datatags=["WDS:/unused"],
                batch_size=1,
                key_map={"real_raw": "mp4", "condition_raw": "txt"},
                target_fps=bad_target_fps,
            )

    def test_image_wds_loader_requires_key_map(self):
        from fastgen.datasets.wds_dataloaders import ImageWDSLoader

        with pytest.raises(TypeError, match="key_map"):
            ImageWDSLoader(datatags=["WDS:/unused"], batch_size=1)

    def test_video_wds_loader_requires_key_map(self):
        from fastgen.datasets.wds_dataloaders import VideoWDSLoader

        with pytest.raises(TypeError, match="key_map"):
            VideoWDSLoader(datatags=["WDS:/unused"], batch_size=1)

        with pytest.raises(TypeError, match="must not be None"):
            VideoWDSLoader(datatags=["WDS:/unused"], batch_size=1, key_map=None)

    @pytest.mark.parametrize("loader_name", ["VideoWDSLoader", "AudioVideoWDSLoader", "CameraVideoWDSLoader"])
    def test_video_loaders_name_a_sample_whose_video_cannot_be_decoded(self, tmp_dataset_dir, loader_name):
        """A failed video decode leaves the raw bytes, which strict deterministic loading reports by sample."""
        import fastgen.datasets.wds_dataloaders as wds_dataloaders

        sample = {
            "mp4": b"not a video",
            "txt": "caption",
            "poses.npy": np.zeros((4, 4, 4)),
            "intrinsics.npy": np.ones((4, 4)),
        }
        DummyWebDatasetBuilder(tmp_dataset_dir).add_sample("clip", sample).build()
        key_map = {"real_raw": "mp4", "condition_raw": "txt"}
        if loader_name == "CameraVideoWDSLoader":
            key_map.update(camera_poses="poses.npy", camera_intrinsics="intrinsics.npy")
        loader = getattr(wds_dataloaders, loader_name)(
            datatags=[f"WDS:{tmp_dataset_dir}"],
            batch_size=1,
            key_map=key_map,
            sequence_length=4,
            img_size=(64, 64),
            num_workers=0,
            train=False,
            deterministic=True,
            strict=True,
        )

        with pytest.raises(ValueError, match="Cannot process sample .*clip") as raised:
            next(iter(loader))
        assert "Cannot decode 'mp4' of sample clip" in str(raised.value.__cause__)

    def test_image_edit_loader_maps_source_pixels_to_raw_condition_image(self):
        from fastgen.configs.data import IMAGE_EDIT_KEY_MAP

        assert IMAGE_EDIT_KEY_MAP == {
            "real_raw": "edited.jpg",
            "condition_image_raw": "source.jpg",
            "condition_raw": "txt",
        }

    def test_video_wds_loader_target_fps(self, tmp_dataset_dir, monkeypatch):
        """Test VideoWDSLoader forwards target_fps to the segment decoder."""
        from fastgen.datasets.decoders import decode_video_segment
        from fastgen.datasets.wds_dataloaders import transform_video
        from fastgen.datasets.wds_dataloaders import VideoWDSLoader

        monkeypatch.setattr("fastgen.datasets.decoders.random.uniform", lambda _start, _end: 0.0)
        source_fps = 12
        target_fps = 2
        sequence_length = 6
        num_source_frames = source_fps * 3
        values = np.linspace(0, 240, num_source_frames, dtype=np.uint8)
        frames = make_constant_video_frames(values, height=32, width=48)
        mp4 = DummyWebDatasetBuilder(tmp_dataset_dir, "unused.tar")._create_dummy_mp4(frames, fps=source_fps)
        dataset_dir = DummyWebDatasetBuilder.create_dataset_dir(
            tmp_dataset_dir,
            1,
            1,
            {
                "mp4": mp4,
                "txt": "fps-aware video",
            },
        )

        loader = VideoWDSLoader(
            datatags=[f"WDS:{dataset_dir}"],
            batch_size=1,
            key_map={"real_raw": "mp4", "condition_raw": "txt"},
            presets_map={"neg_condition_raw": "empty_string"},
            sequence_length=sequence_length,
            target_fps=target_fps,
            img_size=(48, 32),
            num_workers=0,
            train=False,
        )
        batch = next(iter(loader))
        decoded = decode_video_segment(
            "sample.mp4",
            mp4,
            num_frames=sequence_length,
            target_fps=target_fps,
            output_format="torch",
        )
        expected_real = transform_video(decoded, sequence_length, (48, 32))["real_raw"].unsqueeze(0)

        assert loader.decoder_type == "segment"
        assert loader.target_fps == target_fps
        torch.testing.assert_close(batch["real_raw"], expected_real, rtol=0, atol=0)

    def test_video_wds_loader_full_target_fps(self, tmp_dataset_dir):
        """Test VideoWDSLoader forwards target_fps to the full decoder."""
        from fastgen.datasets.decoders import decode_video_full
        from fastgen.datasets.wds_dataloaders import transform_video
        from fastgen.datasets.wds_dataloaders import VideoWDSLoader

        source_fps = 12
        target_fps = 2
        sequence_length = 6
        num_source_frames = source_fps * 4
        values = np.linspace(0, 240, num_source_frames, dtype=np.uint8)
        frames = make_constant_video_frames(values, height=32, width=48)
        mp4 = DummyWebDatasetBuilder(tmp_dataset_dir, "unused.tar")._create_dummy_mp4(frames, fps=source_fps)
        dataset_dir = DummyWebDatasetBuilder.create_dataset_dir(
            tmp_dataset_dir,
            1,
            1,
            {
                "mp4": mp4,
                "txt": "fps-aware full video",
            },
        )

        loader = VideoWDSLoader(
            datatags=[f"WDS:{dataset_dir}"],
            batch_size=1,
            key_map={"real_raw": "mp4", "condition_raw": "txt"},
            presets_map={"neg_condition_raw": "empty_string"},
            sequence_length=sequence_length,
            target_fps=target_fps,
            decoder_type="full",
            img_size=(48, 32),
            num_workers=0,
            train=False,
        )
        batch = next(iter(loader))
        decoded = decode_video_full("sample.mp4", mp4, target_fps=target_fps, output_format="torch")
        expected_real = transform_video(decoded, sequence_length, (48, 32))["real_raw"].unsqueeze(0)

        assert loader.decoder_type == "full"
        assert loader.target_fps == target_fps
        torch.testing.assert_close(batch["real_raw"], expected_real, rtol=0, atol=0)


# =============================================================================
# Deterministic WDSLoader Tests
# =============================================================================


class TestDeterministicWDSLoader:
    """Tests for deterministic WDSLoader with resuming capabilities."""

    @pytest.fixture
    def deterministic_dataset_dir(self, tmp_dataset_dir):
        """Create a deterministic test dataset with multiple shards."""
        dataset_dir = os.path.join(tmp_dataset_dir, "deterministic")
        os.makedirs(dataset_dir, exist_ok=True)

        for shard_idx in range(3):
            builder = DummyWebDatasetBuilder(dataset_dir, f"{shard_idx:05d}.tar")
            for sample_idx in range(8):
                # Deterministic data for verification
                latent_data = np.full((16, 64, 64), float(shard_idx * 8 + sample_idx), dtype=np.float32)
                builder.samples.append(
                    {
                        f"{shard_idx:05d}_{sample_idx:05d}.latents.npy": builder._encode_data("npy", latent_data),
                        f"{shard_idx:05d}_{sample_idx:05d}.text_embedding.npy": builder._encode_data(
                            "npy", np.random.randn(512, 4096).astype(np.float32)
                        ),
                    }
                )
            builder.build()

        shard_count_file = create_shard_count_file(dataset_dir, 8)
        return dataset_dir, shard_count_file

    def _create_det_loader(self, dataset_dir, shard_count_file, start_idx=0, **kwargs):
        """Helper to create a deterministic WDSLoader."""
        from fastgen.datasets.wds_dataloaders import WDSLoader

        defaults = {
            "datatags": [f"WDS:{dataset_dir}"],
            "batch_size": 4,
            "key_map": {"real": "latents.npy", "condition": "text_embedding.npy"},
            "num_workers": 1,
            "deterministic": True,
            "sampler_start_idx": start_idx,
            "shard_count_file": shard_count_file,
        }
        return WDSLoader(**{**defaults, **kwargs})

    def _collect_fnames(self, loader, max_samples=None, max_batches=None):
        """Helper to collect fnames from loader."""
        fnames = []
        for i, batch in enumerate(loader):
            fnames.extend(batch["fname"])
            if max_samples and len(fnames) >= max_samples:
                break
            if max_batches and i >= max_batches - 1:
                break
        return fnames[:max_samples] if max_samples else fnames

    def test_deterministic_same_order(self, deterministic_dataset_dir):
        """Test that deterministic loader produces same order every run."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        fnames1 = self._collect_fnames(self._create_det_loader(dataset_dir, shard_count_file), max_batches=6)
        fnames2 = self._collect_fnames(self._create_det_loader(dataset_dir, shard_count_file), max_batches=6)

        assert fnames1 == fnames2, "Deterministic loader should produce same order"

    def test_deterministic_resume_from_index(self, deterministic_dataset_dir):
        """Test resuming from a specific sampler_start_idx."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        full_fnames = self._collect_fnames(self._create_det_loader(dataset_dir, shard_count_file), max_batches=6)
        resumed_fnames = self._collect_fnames(
            self._create_det_loader(dataset_dir, shard_count_file, start_idx=12), max_batches=3
        )

        assert resumed_fnames == full_fnames[12 : 12 + len(resumed_fnames)]

    def test_deterministic_no_overlap_on_resume(self, deterministic_dataset_dir):
        """Test that resumed samples don't overlap with pre-resume samples."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        start_fnames = self._collect_fnames(self._create_det_loader(dataset_dir, shard_count_file), max_batches=2)
        resumed_fnames = self._collect_fnames(
            self._create_det_loader(dataset_dir, shard_count_file, start_idx=8), max_batches=2
        )

        assert set(start_fnames).isdisjoint(set(resumed_fnames))

    def test_deterministic_unique_samples_per_epoch(self, deterministic_dataset_dir):
        """Test that samples are unique within one epoch."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        all_fnames = self._collect_fnames(self._create_det_loader(dataset_dir, shard_count_file), max_samples=24)

        assert len(all_fnames) == 24
        assert len(set(all_fnames)) == 24, "Found duplicate samples"

    def test_deterministic_data_integrity(self, deterministic_dataset_dir):
        """Test that data values are correctly loaded."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        loader = self._create_det_loader(dataset_dir, shard_count_file, batch_size=1)

        for i, batch in enumerate(loader):
            fname = batch["fname"][0]
            shard_idx, sample_idx = int(fname.split("_")[0]), int(fname.split("_")[1])
            expected_value = float(shard_idx * 8 + sample_idx)
            actual_value = batch["real"][0, 0, 0, 0].item()

            assert abs(actual_value - expected_value) < 1e-5, f"Data mismatch for {fname}"
            assert_batch_type(batch, "real", torch.Tensor)

            if i >= 5:
                break

    def test_deterministic_with_ignore_index(self, deterministic_dataset_dir):
        """Test deterministic loader with ignore index filtering."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        ignore_file = create_ignore_index_file(
            dataset_dir,
            {
                "00000.tar": ["00000_00000", "00000_00001"],
                "00001.tar": ["00001_00000"],
            },
        )

        loader = self._create_det_loader(dataset_dir, shard_count_file, ignore_index_paths=[ignore_file], partial=True)
        all_fnames = self._collect_fnames(loader, max_samples=21)

        assert "00000_00000" not in all_fnames
        assert "00000_00001" not in all_fnames
        assert "00001_00000" not in all_fnames
        assert len(all_fnames) == 21

    def test_deterministic_resume_with_ignore_index(self, deterministic_dataset_dir):
        """Test resuming deterministic loader with ignore index."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        ignore_file = create_ignore_index_file(dataset_dir, {"00000.tar": ["00000_00000", "00000_00001"]})

        full_fnames = self._collect_fnames(
            self._create_det_loader(dataset_dir, shard_count_file, ignore_index_paths=[ignore_file], partial=True),
            max_samples=22,
        )
        resumed_fnames = self._collect_fnames(
            self._create_det_loader(
                dataset_dir, shard_count_file, start_idx=8, ignore_index_paths=[ignore_file], partial=True
            ),
            max_samples=14,
        )

        assert resumed_fnames == full_fnames[8:]

    def test_deterministic_batch_types(self, deterministic_dataset_dir):
        """Test batch types are correct in deterministic mode."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        batch = next(iter(self._create_det_loader(dataset_dir, shard_count_file)))

        assert_batch_type(batch, "real", torch.Tensor)
        assert_batch_type(batch, "condition", torch.Tensor)
        assert_batch_type(batch, "fname", List[str])
        assert_batch_type(batch, "shard", List[str])
        assert_tensor_shape(batch["real"], (4, 16, 64, 64))
        assert_tensor_shape(batch["condition"], (4, 512, 4096))

    @staticmethod
    def _epoch():
        return [f"{shard:05d}_{sample:05d}" for shard in range(3) for sample in range(8)]

    @pytest.mark.parametrize("start_idx", [10, 30])
    def test_deterministic_resume_continues_the_stream_across_epochs(self, deterministic_dataset_dir, start_idx):
        """Epochs form one stream: batches span epoch ends and a resume skips its offset only once."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        fresh = self._collect_fnames(self._create_det_loader(dataset_dir, shard_count_file, batch_size=5), 60)
        resumed = self._collect_fnames(
            self._create_det_loader(dataset_dir, shard_count_file, start_idx=start_idx, batch_size=5), 40
        )

        assert fresh == (self._epoch() * 3)[:60]
        assert resumed == (self._epoch() * 4)[start_idx : start_idx + 40]

    def test_deterministic_training_without_shard_counts_repeats_and_resumes(self, deterministic_dataset_dir):
        """Without a shard count file the stream still repeats, and a resume reads up to its offset."""
        dataset_dir, _ = deterministic_dataset_dir

        fresh = self._collect_fnames(self._create_det_loader(dataset_dir, None), 56)
        resumed = self._collect_fnames(self._create_det_loader(dataset_dir, None, start_idx=30), 24)

        assert fresh == (self._epoch() * 3)[:56]
        assert resumed == (self._epoch() * 3)[30:54]


# =============================================================================
# AudioVideoWDSLoader Tests
# =============================================================================


def _build_av_dataset(
    tmp_dir: str,
    num_samples: int,
    video_shape: Tuple[int, int, int] = (40, 64, 96),  # (T, H, W)
    audio_sr: int = 16000,
    audio_freq_hz: float = 440.0,
    video_fps: int = 24,
    sr_per_sample: Optional[List[int]] = None,
    audio_per_sample: Optional[List[bool]] = None,
) -> str:
    """Build a one-shard WDS dataset of AV mp4s plus matching txt captions.

    Each sample's audio is a deterministic stereo sine wave at
    ``audio_freq_hz`` so tests can sanity-check the decoded waveform.

    Args:
        sr_per_sample: If provided, overrides ``audio_sr`` per-sample
            (used for the mixed-SR test). Length must equal ``num_samples``.
        audio_per_sample: If provided, sample ``i`` is a video-only mp4 when
            ``audio_per_sample[i]`` is False. Length must equal ``num_samples``.

    Returns:
        Path to the dataset directory containing a single ``00000.tar``.
    """
    dataset_dir = os.path.join(tmp_dir, "av_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    builder = DummyWebDatasetBuilder(dataset_dir, "00000.tar")

    t = video_shape[0]
    duration_sec = t / video_fps

    rng = np.random.default_rng(0)
    for i in range(num_samples):
        sr_i = sr_per_sample[i] if sr_per_sample is not None else audio_sr
        n = int(round(duration_sec * sr_i))
        # Deterministic stereo sine: left = sin(2pi*f*t), right = -sin (anti-phase).
        time_axis = np.arange(n, dtype=np.float32) / float(sr_i)
        wave = np.sin(2.0 * np.pi * audio_freq_hz * time_axis).astype(np.float32)
        audio = np.stack([wave, -wave], axis=0)  # (2, n)
        # Random RGB frames so the h264 encoder has something to work with.
        frames = rng.integers(0, 256, (t,) + video_shape[1:] + (3,), dtype=np.uint8)
        if audio_per_sample is None or audio_per_sample[i]:
            mp4 = DummyWebDatasetBuilder.create_dummy_av_mp4(frames, audio, sr_i, video_fps)
        else:
            mp4 = {"frames": frames, "fps": video_fps}
        builder.add_sample(f"sample_{i:04d}", {"mp4": mp4, "txt": f"caption {i}"})
    builder.build()
    return dataset_dir


class TestAudioVideoWDSLoader:
    """Tests for :class:`AudioVideoWDSLoader`.

    Covers the contract that the loader emits, per-sample pre-collate:
      - ``data[video_key] = {"video": Tensor, "audio": Tensor}``
        as a multimodal dict (no separate top-level audio keys);
      - audio resampled to ``target_sample_rate`` (when set);
      - audio trimmed/padded to ``num_video_frames / fps`` seconds at
        the target rate, giving a uniform per-sample length even when
        source clips have different native sample rates;
      - only ``video`` for clips without an audio track, which cannot
        share a batch with clips that have one.
    """

    def _make_loader(
        self,
        dataset_dir: str,
        target_sample_rate: Optional[int],
        batch_size: int,
        sequence_length: int,
        img_size: Tuple[int, int],
        **kwargs,
    ):
        from fastgen.datasets.wds_dataloaders import AudioVideoWDSLoader

        return AudioVideoWDSLoader(
            datatags=[f"WDS:{dataset_dir}"],
            batch_size=batch_size,
            num_workers=0,
            key_map={"real_raw": "mp4", "condition_raw": "txt"},
            sequence_length=sequence_length,
            img_size=img_size,
            target_sample_rate=target_sample_rate,
            **kwargs,
        )

    def test_multimodal_dict_output_shape_and_keys(self, tmp_dataset_dir):
        """Loader emits ``data["real_raw"] = {"video": ..., "audio": ...}`` with the right shapes."""
        target_sr = 16000
        sequence_length = 24
        img_size = (96, 64)  # (W, H)
        video_fps = 24
        dataset_dir = _build_av_dataset(
            tmp_dataset_dir,
            num_samples=4,
            video_shape=(40, 64, 96),
            audio_sr=44100,
            video_fps=video_fps,
        )
        loader = self._make_loader(
            dataset_dir,
            target_sample_rate=target_sr,
            batch_size=2,
            sequence_length=sequence_length,
            img_size=img_size,
        )
        batch = next(iter(loader))

        assert isinstance(batch["real_raw"], dict), f"expected real_raw to be a dict, got {type(batch['real_raw'])!r}"
        assert set(batch["real_raw"].keys()) == {"video", "audio"}, batch["real_raw"].keys()

        # Video: (B, C, T, H, W) per the existing VideoWDSLoader contract.
        assert_tensor_shape(
            batch["real_raw"]["video"],
            (2, 3, sequence_length, img_size[1], img_size[0]),
            "real_raw.video",
        )

        # Audio: (B, A_C, T_a) where T_a = int(sequence_length * target_sr / fps) (multiplied first).
        expected_audio_len = int(sequence_length * target_sr / video_fps)
        audio = batch["real_raw"]["audio"]
        assert audio.dtype == torch.float32, audio.dtype
        assert audio.ndim == 3, f"expected (B, C, T_a), got shape {tuple(audio.shape)}"
        assert audio.shape[0] == 2, audio.shape
        assert audio.shape[1] == 2, audio.shape
        assert audio.shape[2] == expected_audio_len, (
            f"expected audio T_a={expected_audio_len}, got {audio.shape[2]} "
            f"(target_sr={target_sr}, fps={video_fps}, sequence_length={sequence_length})"
        )

    def test_mixed_source_sr_batch(self, tmp_dataset_dir):
        """A batch containing two clips at different native SRs collates cleanly.

        Two failure modes guarded against:

          - ``RuntimeError: Boolean value of Tensor with more than one
            element is ambiguous`` — would fire if the per-sample SR
            were carried through the batch as a tensor and compared to
            a scalar.
          - ``RuntimeError: stack expects each tensor to be equal size``
            — would fire if per-sample audio length depended on
            per-sample native SR (it does not, because the loader trims
            at the target SR after resample, giving a uniform length).
        """
        target_sr = 16000
        sequence_length = 24
        img_size = (96, 64)
        video_fps = 24
        # Two samples at deliberately different source rates.
        dataset_dir = _build_av_dataset(
            tmp_dataset_dir,
            num_samples=2,
            video_shape=(40, 64, 96),
            video_fps=video_fps,
            sr_per_sample=[44100, 48000],
        )
        loader = self._make_loader(
            dataset_dir,
            target_sample_rate=target_sr,
            batch_size=2,
            sequence_length=sequence_length,
            img_size=img_size,
        )
        batch = next(iter(loader))

        expected_audio_len = int(sequence_length / video_fps * target_sr)
        audio = batch["real_raw"]["audio"]
        assert audio.shape[0] == 2, audio.shape
        assert audio.shape[-1] == expected_audio_len, (
            f"mixed-SR batch did not produce uniform output length: shape={tuple(audio.shape)}, "
            f"expected last dim {expected_audio_len}"
        )

    @pytest.mark.parametrize("audio_first", [True, False])
    def test_mixed_audio_presence_requires_batch_size_one(self, tmp_dataset_dir, audio_first):
        """Decoded clips with and without an audio track reach batching with different modality keys. A shared
        batch is rejected in either order (``default_collate`` alone would drop the audio or raise ``KeyError``,
        depending on which clip comes first); ``batch_size=1`` keeps each clip's own modalities."""
        has_audio = [audio_first, not audio_first]
        dataset_dir = _build_av_dataset(tmp_dataset_dir, num_samples=2, audio_per_sample=has_audio)

        def batch_modalities(batch_size):
            loader = self._make_loader(
                dataset_dir, 16000, batch_size, sequence_length=24, img_size=(96, 64), train=False
            )
            return [set(batch["real_raw"]) for batch in loader]

        with pytest.raises(ValueError, match="different mapping keys"):
            batch_modalities(batch_size=2)
        assert batch_modalities(batch_size=1) == [{"video", "audio"} if audio else {"video"} for audio in has_audio]

    def test_segment_start_frame_reports_the_decoded_segment(self, tmp_dataset_dir, monkeypatch):
        """The audio-video segment start reaches the batch, as it does for video-only segments."""
        monkeypatch.setattr("fastgen.datasets.decoders.random.uniform", lambda _start, _end: 0.5)
        dataset_dir = _build_av_dataset(tmp_dataset_dir, num_samples=1, video_shape=(40, 64, 96), video_fps=24)
        loader = self._make_loader(
            dataset_dir, target_sample_rate=16000, batch_size=1, sequence_length=12, img_size=(96, 64)
        )

        batch = next(iter(loader))

        assert batch["segment_start_frame"].tolist() == [12]  # 0.5 s at 24 fps

    def test_target_sample_rate_none_passes_audio_through(self, tmp_dataset_dir):
        """``target_sample_rate=None`` skips resample and packs audio at its native rate.

        Verified by checking that the audio's last-dim length equals
        the native-rate length, not a (different) target-rate length.
        """
        native_sr = 22050
        sequence_length = 24
        img_size = (96, 64)
        video_fps = 24
        dataset_dir = _build_av_dataset(
            tmp_dataset_dir,
            num_samples=2,
            video_shape=(40, 64, 96),
            audio_sr=native_sr,
            video_fps=video_fps,
        )
        loader = self._make_loader(
            dataset_dir,
            target_sample_rate=None,
            batch_size=2,
            sequence_length=sequence_length,
            img_size=img_size,
        )
        batch = next(iter(loader))

        native_len = int(sequence_length / video_fps * native_sr)
        # If the loader had silently resampled to 16 kHz the lengths would differ.
        # Length should match the native rate.
        audio = batch["real_raw"]["audio"]
        assert audio.dtype == torch.float32, audio.dtype
        assert audio.shape[-1] == native_len, (
            f"target_sample_rate=None should preserve native-rate length {native_len}, " f"got {audio.shape[-1]}"
        )

    def test_unknown_frame_rate_rejects_clips_with_audio(self, tmp_dataset_dir):
        """``_preprocess`` of a hand-built decoded item: without a frame rate the audio cannot be sized to the kept
        frames, so a clip with audio raises; a video-only clip needs no frame rate, and a known one trims to
        ``num_video_frames / fps``."""
        sequence_length, audio_sr = 8, 16000
        dataset_dir = _build_av_dataset(tmp_dataset_dir, num_samples=1, video_shape=(sequence_length, 64, 96))
        loader = self._make_loader(
            dataset_dir, audio_sr, batch_size=1, sequence_length=sequence_length, img_size=(96, 64)
        )

        def decoded_item(video_fps, audio):
            frames = torch.zeros(sequence_length, 64, 96, 3, dtype=torch.uint8)
            video = {"video": frames, "audio": audio, "audio_sr": audio_sr, "video_fps": video_fps}
            return {"__key__": "sample", "mp4": video, "txt": "caption"}

        # Two seconds of audio for eight frames: not a 24 fps clip.
        audio = torch.zeros(2, 2 * audio_sr)
        with pytest.raises(ValueError, match="unknown video frame rate"):
            loader._preprocess(decoded_item(0.0, audio))
        assert set(loader._preprocess(decoded_item(0.0, None))["real_raw"]) == {"video"}
        expected = int(sequence_length * audio_sr / 24)
        assert loader._preprocess(decoded_item(24.0, audio))["real_raw"]["audio"].shape == (2, expected)

    @pytest.mark.parametrize(
        ("source_fps", "target_fps", "decoder_type"),
        [
            (24, 12, "segment"),  # downsample, segment decoder
            (24, 48, "segment"),  # upsample, segment decoder
            (24, 12, "full"),  # downsample, full decoder
            (24, 48, "full"),  # upsample, full decoder
        ],
    )
    def test_target_fps_resamples_video_and_audio_length(
        self, tmp_dataset_dir, monkeypatch, source_fps, target_fps, decoder_type
    ):
        """``target_fps`` must drive both video resampling and the audio window length.

        The AV decoder reports ``video_fps = target_fps`` whenever ``target_fps``
        is set; the loader's ``_preprocess`` then trims/pads the (already
        resampled) audio to ``sequence_length / video_fps * target_sample_rate``
        samples. We therefore expect the per-sample audio length to track
        ``target_fps``, *not* ``source_fps``. To make this a real regression
        guard we additionally check that the source-FPS-derived length would
        be different, so a future bug that silently used ``source_fps`` (e.g.
        decoder returning ``avg_fps`` instead of ``target_fps``) would fail
        this assertion.
        """
        from fastgen.datasets.wds_dataloaders import AudioVideoWDSLoader

        monkeypatch.setattr("fastgen.datasets.decoders.random.uniform", lambda _start, _end: 0.0)
        target_sr = 16000
        sequence_length = 24
        img_size = (96, 64)  # (W, H)
        # 4s clip leaves plenty of headroom for both decoder modes and for
        # both downsampling (target<source) and upsampling (target>source).
        num_source_frames = source_fps * 4
        dataset_dir = _build_av_dataset(
            tmp_dataset_dir,
            num_samples=2,
            video_shape=(num_source_frames, 64, 96),
            audio_sr=44100,
            video_fps=source_fps,
        )

        loader = AudioVideoWDSLoader(
            datatags=[f"WDS:{dataset_dir}"],
            batch_size=1,
            num_workers=0,
            key_map={"real_raw": "mp4", "condition_raw": "txt"},
            sequence_length=sequence_length,
            img_size=img_size,
            target_sample_rate=target_sr,
            target_fps=target_fps,
            decoder_type=decoder_type,
        )
        batch = next(iter(loader))

        assert loader.target_fps == target_fps
        assert loader.decoder_type == decoder_type

        assert_tensor_shape(
            batch["real_raw"]["video"],
            (1, 3, sequence_length, img_size[1], img_size[0]),
            "real_raw.video",
        )

        expected_audio_len = int(sequence_length / target_fps * target_sr)
        source_audio_len = int(sequence_length / source_fps * target_sr)
        audio = batch["real_raw"]["audio"]
        assert audio.dtype == torch.float32, audio.dtype
        assert audio.shape == (1, 2, expected_audio_len), (
            f"target_fps={target_fps} (source_fps={source_fps}, decoder_type={decoder_type}) "
            f"should produce audio shape (1, 2, {expected_audio_len}), got {tuple(audio.shape)}"
        )
        assert expected_audio_len != source_audio_len, (
            "test is ill-posed: target_fps and source_fps yield the same audio length, "
            "so it cannot detect a regression where source_fps is used instead of target_fps"
        )


def _build_camera_dataset(tmp_dir: str, num_frames: int = 32, video_fps: int = 16) -> str:
    """One-shard dataset of a video plus per-source-frame camera sidecars whose rows encode the frame index."""
    dataset_dir = os.path.join(tmp_dir, "camera_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    builder = DummyWebDatasetBuilder(dataset_dir, "00000.tar")
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (num_frames, 32, 48, 3), dtype=np.uint8)
    silence = np.zeros((2, int(num_frames * 16000 / video_fps)), dtype=np.float32)
    poses = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    poses[:, 0, 3] = np.arange(num_frames)  # x-translation = source frame index
    intrinsics = np.tile(np.array([40.0, 40.0, 24.0, 16.0], dtype=np.float32), (num_frames, 1))
    builder.add_sample(
        "sample_0000",
        {
            "mp4": DummyWebDatasetBuilder.create_dummy_av_mp4(frames, silence, 16000, video_fps),
            "poses.npy": poses,
            "intrinsics.npy": intrinsics,
            "txt": "caption",
        },
    )
    builder.build()
    return dataset_dir


class TestCameraVideoWDSLoader:
    def test_poses_follow_the_kept_source_frames_under_target_fps(self, tmp_dataset_dir, monkeypatch):
        """Resampling a 16 fps source onto 8 fps keeps every other source frame; the pose rows are
        gathered by those source indices rather than sliced at the decoded cadence."""
        import fastgen.datasets.decoders as decoders_module
        from fastgen.datasets.wds_dataloaders import CameraVideoWDSLoader

        dataset_dir = _build_camera_dataset(tmp_dataset_dir)
        monkeypatch.setattr(decoders_module, "_random_start_sec", lambda total, needed: 0.5)
        loader = CameraVideoWDSLoader(
            datatags=[f"WDS:{dataset_dir}"],
            batch_size=1,
            num_workers=0,
            key_map={
                "real_raw": "mp4",
                "camera_poses": "poses.npy",
                "camera_intrinsics": "intrinsics.npy",
                "condition_raw": "txt",
            },
            sequence_length=6,
            img_size=(48, 32),
            target_fps=8.0,
        )
        batch = next(iter(loader))

        assert batch["real_raw"].shape[2] == 6
        torch.testing.assert_close(batch["camera_poses"][0, :, 0, 3], torch.tensor([8.0, 10.0, 12.0, 14.0, 16.0, 18.0]))
        assert batch["camera_intrinsics"].shape[:2] == (1, 6)

    @pytest.mark.parametrize(
        ("target_fps", "expected_x"),
        [(8.0, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]), (None, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])],
    )
    def test_full_mode_keeps_only_the_rows_of_the_leading_frames(self, tmp_dataset_dir, target_fps, expected_x):
        """Full decoding keeps the whole 32-frame clip and ``transform_video`` truncates it to
        ``sequence_length``; the pose / intrinsics rows must follow the truncated video, not the whole
        decoded stream."""
        from fastgen.datasets.wds_dataloaders import CameraVideoWDSLoader

        loader = CameraVideoWDSLoader(
            datatags=[f"WDS:{_build_camera_dataset(tmp_dataset_dir)}"],
            batch_size=1,
            num_workers=0,
            key_map={
                "real_raw": "mp4",
                "camera_poses": "poses.npy",
                "camera_intrinsics": "intrinsics.npy",
                "condition_raw": "txt",
            },
            sequence_length=6,
            img_size=(48, 32),
            target_fps=target_fps,
            decoder_type="full",
        )
        batch = next(iter(loader))

        assert batch["real_raw"].shape[2] == 6
        torch.testing.assert_close(batch["camera_poses"][0, :, 0, 3], torch.tensor(expected_x))
        assert batch["camera_intrinsics"].shape[:2] == (1, 6)

    def test_sidecar_without_rows_for_the_kept_frames_is_rejected(self, tmp_dataset_dir):
        """``_select_camera`` raises for a sidecar shorter than the kept source frames, which disagrees with the
        video, and for frames the decoder could not place, which have no rows at all, instead of repeating the last
        pose or slicing at the decoded cadence."""
        from fastgen.datasets.wds_dataloaders import CameraVideoWDSLoader

        loader = CameraVideoWDSLoader(
            datatags=[f"WDS:{_build_camera_dataset(tmp_dataset_dir)}"],
            batch_size=1,
            num_workers=0,
            key_map={
                "real_raw": "mp4",
                "camera_poses": "poses.npy",
                "camera_intrinsics": "intrinsics.npy",
                "condition_raw": "txt",
            },
            sequence_length=6,
            img_size=(48, 32),
        )
        poses = torch.arange(20.0).view(20, 1, 1).expand(20, 4, 4)

        assert loader._select_camera(poses, [0, 19]).shape[0] == 2
        with pytest.raises(ValueError, match="20 rows"):
            loader._select_camera(poses, [18, 20])
        with pytest.raises(ValueError, match="could not place"):
            loader._select_camera(poses, None)
