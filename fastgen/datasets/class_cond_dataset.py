# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import zipfile

import numpy as np
import PIL.Image

try:
    import pyspng
except ImportError:
    pyspng = None


import fastgen.utils.logging_utils as logger
from fastgen.utils.cache_utils import cache, file_extension, resolve_cached_path
from fastgen.utils.distributed import get_rank
from fastgen.utils.distributed.collectives import raise_across_ranks
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        name,  # Name of the dataset.
        raw_shape,  # Shape of the raw image data (NCHW).
        max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
        xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed=0,  # Random seed to use when applying max_size.
        cache=False,  # Cache images in CPU memory?
        state_key="real_raw",  # Output key; use "real" for already-materialized model-space data.
    ):
        if state_key not in ("real", "real_raw"):
            raise ValueError(f"state_key must be 'real' or 'real_raw', got {state_key!r}")

        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._state_key = state_key
        self._cached_images = dict()  # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                if self._use_labels:
                    logger.warning(
                        f"use_labels=True but no labels found for dataset '{self._name}' "
                        "(missing dataset.json or null labels); treating as unconditional "
                        "(empty condition, no neg_condition)."
                    )
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except Exception as e:  # Catch 'Exception', not everything
            logger.warning(f"Error closing in __del__: {e}")

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self._raw_shape[1:]
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]

        is_pixel_data = image.dtype == np.uint8
        if is_pixel_data:
            # this pathway is for pixel data
            image = image.astype(np.float32) / 127.5 - 1
        else:
            # this pathway is for latent data
            image = image.astype(np.float32)

        item = {self._state_key: image.copy(), "condition": self.get_label(idx), "idx": idx}
        if self.has_labels:
            item["neg_condition"] = np.zeros(self.label_shape, dtype=np.float32)
        return item

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)


# ----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.


class ImageFolderDataset(ImageDataset):
    def __init__(
        self,
        path,  # Path to directory or zip.
        remote_path=None,  # S3 or HTTP(S) path from which to download the zip file.
        resolution=None,  # Ensure specific resolution, None = highest available.
        use_pyspng=True,  # Use pyspng if available?
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None
        self._zipfile_owner_pid = None

        if file_extension(self._path) == ".zip" and os.path.isfile(self._path):
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())
        elif os.path.isdir(self._path):
            self._type = "dir"
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }
        elif remote_path is not None and file_extension(remote_path) == ".zip":
            # Only rank 0 populates the shared mirror; leaving the block publishes it to the other ranks
            # and raises a cache failure on every rank instead of stalling a barrier.
            with raise_across_ranks(f"cache dataset {remote_path}"):
                if get_rank() == 0:
                    cache(remote_path, strict=True)

            self._path = resolve_cached_path(remote_path)
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())

        else:
            raise IOError(
                f"Dataset not found at {self._path}: expected a ZIP file or a directory there, or a remote_path ZIP "
                f"to download (got {remote_path!r}). The dataset configs look under $DATA_ROOT_DIR; prepare the "
                "datasets there with scripts/download_data.py."
            )

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {".npy"}
        self._image_fnames = sorted(fname for fname in self._all_fnames if file_extension(fname) in supported_ext)
        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def _get_zipfile(self):
        assert self._type == "zip"
        current_pid = os.getpid()
        if self._zipfile is not None and self._zipfile_owner_pid != current_pid:
            self.close()
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
            self._zipfile_owner_pid = current_pid
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None
            self._zipfile_owner_pid = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None, _zipfile_owner_pid=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        ext = file_extension(fname)
        with self._open_file(fname) as f:
            if ext == ".npy":
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == ".png" and self._use_pyspng and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        return image

    def _load_raw_labels(self):
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
