# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.utils.data import DataLoader

from fastgen.datasets.class_cond_dataset import ImageFolderDataset
from fastgen.datasets.samplers import InfiniteSampler, normalize_sampler_start_idx


class ImageLoader:
    def __init__(
        self,
        dataset_path: str,
        remote_path: str | None,
        batch_size: int,
        use_labels: bool = True,
        cache: bool = True,
        shuffle: bool = True,
        sampler_start_idx: int | None = None,
        num_workers: int = 1,
        **kwargs,
    ):
        """
        ImageLoader for class conditional datasets
        Args:
            dataset_path (str): Path to the dataset
            remote_path (str | None): S3 or HTTP(S) path to a prepared dataset ZIP
            batch_size (int): Batch size
            use_labels (bool): Whether to use labels
            cache (bool): Whether to cache the dataset
            shuffle (bool): Whether to shuffle the dataset
            sampler_start_idx (int | None): Global start index for the sampler. A ``None`` set in a config lets
                the trainer choose it: the resume offset for training, 0 for validation. An unresolved ``None``
                defaults to 0 with a warning.
            num_workers (int): Number of data loading worker processes
        """
        sampler_start_idx = normalize_sampler_start_idx(sampler_start_idx)

        self.dataset = ImageFolderDataset(
            path=dataset_path, remote_path=remote_path, use_labels=use_labels, cache=cache, **kwargs
        )

        dataset_sampler = InfiniteSampler(dataset=self.dataset, shuffle=shuffle, start_idx=sampler_start_idx)
        # The image cache is process-local, so its memory use can grow with the number of workers.
        data_loader_kwargs = dict(
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if num_workers > 0:
            data_loader_kwargs["prefetch_factor"] = 2

        self.loader = DataLoader(
            dataset=self.dataset, sampler=dataset_sampler, batch_size=batch_size, **data_loader_kwargs
        )

    def __iter__(self):
        return iter(self.loader)
