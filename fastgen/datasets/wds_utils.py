# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for WebDataset loaders.

This module provides shared functionality for all WDS loaders including:
- Shard discovery (local and S3)
- Ignore index management
- Shard counting for deterministic iteration
- Common preprocessing functions
- Prompt embedding loading
"""

import functools
import json
import os
import sys
import itertools
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

import torch
import webdataset as wds
from torch.utils.data import IterableDataset, default_collate

import fastgen.utils.logging_utils as logger
from fastgen.utils.basic_utils import ensure_trailing_slash
from fastgen.utils.distributed import get_rank, world_size


# =============================================================================
# Shard Utilities
# =============================================================================


def shard_to_name(shard: str) -> str:
    """Convert a shard path to a name by extracting the filename.

    Example:
        '<path>/00012.tar' -> '00012.tar'
    """
    filename = shard.split("/")[-1]
    return filename


def shard_to_idx(shard: str) -> int:
    """Convert a shard path to an index by converting the filename to a number.

    Example:
        '<path>/00012.tar' -> 12

    Raises:
        ValueError: If the filename is not a number
    """
    filename = shard_to_name(shard)
    stem = filename.split(".")[0]
    if not stem.isdigit():
        raise ValueError(f"Shard name {stem} is not a number")
    return int(stem)


def get_prefixes(wds_tags: List[str]) -> List[str]:
    """Extract path prefixes from WDS tags.

    Args:
        wds_tags: List of WDS tags in format 'WDS:<PATH>'

    Returns:
        List of path prefixes
    """
    prefixes = []
    for name in wds_tags:
        wds_start_idx = name.find(":") + 1
        if wds_start_idx == 0:
            raise ValueError("WDS datatag format must match 'WDS:<PATH>'.")
        prefixes.append(name[wds_start_idx:])
    return prefixes


def get_shards(
    wds_tags: List[str],
    shard_start: int = 0,
    shard_end: int | None = None,
    num_shards: int | None = None,
) -> List[str]:
    """Get the shards from the WDS tags.

    Supports both local file paths and S3 paths (using s5cmd).

    Args:
        wds_tags: List of WDS tags in format 'WDS:<PATH>'
        shard_start: Start index for the list of shards
        shard_end: End index for the list of shards (inclusive)
        num_shards: Target number of shards (will truncate or repeat)

    Returns:
        List of shard URLs
    """
    # Validate the dataset tags
    for d in wds_tags:
        if not d.startswith("WDS"):
            raise ValueError("Cannot mix and match WDS datasets with other types")

    prefixes = get_prefixes(wds_tags)
    all_urls = []

    for prefix in prefixes:
        if prefix.startswith("s3://"):
            prefix = ensure_trailing_slash(prefix)
            with os.popen(f"s5cmd ls {prefix}", "r") as f:
                urls = [
                    f"pipe:s5cmd cat {prefix}{path.strip().split(' ')[-1]}"
                    for path in f.readlines()
                    if path.strip().endswith(".tar")
                ]
            all_urls.extend(urls)
        elif Path(prefix).is_dir():
            # Local directory with tar files
            for f in Path(prefix).iterdir():
                if f.is_file() and f.suffix.lower() == ".tar":
                    all_urls.append(str(f))
        else:
            raise ValueError(f"Invalid prefix: {prefix}")

    # Sort by shard name
    try:
        all_urls.sort(key=shard_to_idx)
    except ValueError as e:
        logger.error(f"Error sorting urls: {e}")
        all_urls.sort(key=shard_to_name)

    # Truncate or repeat shards
    all_urls = all_urls[max(shard_start, 0) :]
    if shard_end is not None:
        all_urls = all_urls[: int(shard_end + 1)]
    if num_shards is not None:
        if len(all_urls) > num_shards:
            logger.warning(
                f"Number of shards {len(all_urls)} is greater than the number of shards to use {num_shards}. Truncating shards"
            )
            all_urls = all_urls[:num_shards]
        elif len(all_urls) < num_shards:
            logger.warning(
                f"Number of shards {len(all_urls)} is less than the number of shards to use {num_shards}. Repeating shards"
            )
            all_urls = all_urls * (num_shards // len(all_urls)) + all_urls[: num_shards % len(all_urls)]
        assert len(all_urls) == num_shards

    if len(all_urls) == 0:
        raise ValueError(
            f"No shards remaining in {wds_tags} with shard_start={shard_start}, shard_end={shard_end}, and num_shards={num_shards}."
        )

    logger.info(f"Using a total of {len(all_urls)} shards")
    return all_urls


# =============================================================================
# Ignore Index Utilities
# =============================================================================


def get_ignore_index(ignore_index_paths: List[str] | None, shard_list: List[str] | None = None) -> Dict[str, Set[str]]:
    """Get the ignore index from the ignore index paths.

    Args:
        ignore_index_paths: List of JSON files specifying the files to skip.
            JSON files are formatted as {shard: [fname1, fname2, ...]}
            Supports both local file paths and S3 paths (starting with "s3://")
        shard_list: List of shard paths to check for consistency

    Returns:
        Dictionary mapping shard names to sets of filenames to ignore
    """
    ignore_index: Dict[str, Set[str]] = {}

    if ignore_index_paths is not None:
        for p in ignore_index_paths:
            logger.info(f"Loading ignore index from {p}")

            if p.startswith("s3://"):
                try:
                    with os.popen(f"s5cmd cat {p}", "r") as f:
                        json_content = f.read()
                        if not json_content.strip():
                            logger.warning(f"Empty or failed to read S3 file: {p}")
                            continue
                        data = json.loads(json_content)
                except (json.JSONDecodeError, OSError) as e:
                    logger.error(f"Failed to load ignore index from S3 path {p}: {e}")
                    continue
            else:
                try:
                    with open(p, mode="r") as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                    logger.error(f"Failed to load ignore index from local path {p}: {e}")
                    continue

            for shard, files in data.items():
                if shard not in ignore_index:
                    ignore_index[shard] = set()
                ignore_index[shard].update(files)

        logger.info(f"Ignoring {sum(len(files) for files in ignore_index.values())} files.")

    # Check index file shards
    if shard_list is not None:
        shard_names = [shard_to_name(shard) for shard in shard_list]
        shards_to_remove = []
        for shard in list(ignore_index.keys()):
            if shard not in shard_names:
                logger.warning(f"Shard {shard} of ignore index cannot be found in dataset shards.")
                shards_to_remove.append(shard)

        for shard in shards_to_remove:
            ignore_index.pop(shard)

    return ignore_index


def get_shard_count(
    shard_count_file: str,
    ignore_index: Dict[str, Set[str]] | None = None,
    shard_list: List[str] | None = None,
) -> Dict[str, int]:
    """Load shard count from a JSON file.

    Args:
        shard_count_file: Path to JSON file with shard counts
        ignore_index: Optional ignore index to adjust counts
        shard_list: Optional list of shards to check

    Returns:
        Dictionary mapping shard names to sample counts
    """
    if shard_count_file.startswith("s3://"):
        try:
            with os.popen(f"s5cmd cat {shard_count_file}", "r") as f:
                json_content = f.read()
                if not json_content.strip():
                    raise ValueError(f"Empty or failed to read S3 file: {shard_count_file}")
                shard_count = json.loads(json_content)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load shard count from S3 path {shard_count_file}: {e}")
            raise
    else:
        with open(shard_count_file, "r") as f:
            shard_count = json.load(f)

    if ignore_index is not None:
        for shard_name in shard_count.keys():
            if shard_name in ignore_index:
                shard_count[shard_name] -= len(ignore_index[shard_name])

    if shard_list is not None:
        for shard in shard_list:
            shard_name = shard_to_name(shard)
            if shard_name not in shard_count:
                logger.warning(f"Shard {shard_name} cannot be found in shard count file.")

    logger.info(
        f"Shard count file loaded from {shard_count_file}. Using {sum(shard_count.values())} samples from the dataset."
    )

    return shard_count


def filter_by_index(item: dict, ignore_index: Dict[str, Set[str]]) -> bool:
    """Filter function to ignore files in the ignore index."""
    shard_name = shard_to_name(item["__url__"])
    fname = item["__key__"]
    return fname not in ignore_index.get(shard_name, set())


# =============================================================================
# Common Loader Utilities
# =============================================================================


def _nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
    else:
        yield from src


def _collate_fn(batch):
    # Filter out None samples and collate
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return {}
    return default_collate(valid_batch)


def batch_wds_dataset(dataset: wds.WebDataset, batch_size: int, partial: bool = False) -> wds.WebDataset:
    """Batch a webdataset.

    Args:
        dataset (wds.WebDataset): Input webdataset.
        batch_size (int): Batchsize of the loader.
        partial (bool, optional): Whether to return partial batches. Defaults to False.

    Returns:
        wds.WebDataset: Batched webdataset.
    """
    return dataset.batched(batch_size, partial=partial, collation_fn=_collate_fn)


def wds_to_dataloader(
    dataset: wds.WebDataset,
    batch_size: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    partial: bool = False,
) -> wds.WebLoader:
    """Wrap a webdataset with the DataLoader interface.

    Args:
        dataset (wds.WebDataset): Input webdataset.
        batch_size (int): Batchsize of the loader. If None, assumes that the dataset is already batched.
        num_workers (int): Number of workers for parallel dataprocessing. Defaults to 0.
        pin_memory (bool, optional): Whether to pin memory in loader. Defaults to True.
        partial (bool, optional): Whether to return partial batches. Defaults to False.

    Returns:
        wds.WebLoader: WebLoader wrapping the input webdataset.
    """
    if batch_size is not None:
        dataset = batch_wds_dataset(dataset, batch_size, partial)

    data_loader = wds.WebLoader(
        dataset,
        batch_size=None,
        collate_fn=None,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return data_loader


def init_output_dict(item: dict) -> Dict[str, Any]:
    """Initialize the output dictionary with common fields.

    Args:
        item: WebDataset item

    Returns:
        Dictionary with fname and shard fields
    """
    return {
        "fname": item["__key__"],
        "shard": item.get("__url__", ""),
    }


class BaseWDSLoader:
    """Base class for WebDataset loaders.

    This class handles common functionality:
    - num_workers auto-calculation
    - Deterministic behavior warnings
    - Common attribute initialization
    - The __iter__ method

    Subclasses should call super().__init__() with the common parameters,
    then create their dataset and call _create_loader().
    """

    def __init__(
        self,
        datatags: List[str],
        batch_size: int,
        num_workers: int | None = None,
        train: bool = True,
        cache_path: str | None = None,
        deterministic: bool = False,
        sampler_start_idx: int = 0,
        shard_count_file: str | None = None,
        ignore_index_paths: List[str] | None = None,
        shard_start: int = 0,
        shard_end: int | None = None,
        shards_per_worker: int | None = None,
        partial: bool = False,
        empty_check: bool = True,
        shuffle_size: int = 1000,
    ):
        """Initialize common loader attributes.

        Args:
            datatags: List of WDS tags in format 'WDS:<PATH>'
            batch_size: Batch size for the dataloader
            num_workers: Number of workers (None for auto)
            train: Whether this is a training dataset
            cache_path: Path to cache directory
            deterministic: Whether to use deterministic iteration
            sampler_start_idx: Start index for resumable iteration
            shard_count_file: Path to shard count file
            ignore_index_paths: List of JSON files specifying files to skip
            shard_start: Start index for shard selection
            shard_end: End index for shard selection
            shards_per_worker: If None, forces a specific number of total shards per worker by truncating/repeating shards.
            partial: Whether to allow partial batches
            empty_check: Whether to check for empty URL list
            shuffle_size: Size of shuffle buffer (0 for no shuffle, only used when train=True)
        """
        self.datatags = datatags
        self.batch_size = batch_size
        self.train = train
        self.cache_path = cache_path
        self.deterministic = deterministic
        self.sampler_start_idx = sampler_start_idx or 0
        self.shard_count_file = shard_count_file
        self.ignore_index_paths = ignore_index_paths
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.shards_per_worker = shards_per_worker
        self.partial = partial
        self.empty_check = empty_check
        self.shuffle_size = shuffle_size

        # Compute num_workers
        self.num_workers = num_workers
        if self.num_workers is None:
            if deterministic:
                self.num_workers = 1
            else:
                self.num_workers = min(os.cpu_count() // world_size(), 8)
            logger.info(f"Automatically setting num_workers to {self.num_workers}")

        # Log deterministic warnings
        if self.deterministic:
            if self.num_workers > 1:
                logger.warning(
                    "Due to concurrency of workers, deterministic behavior cannot be guaranteed with more than one worker."
                )
            if self.train:
                logger.warning("Deterministic webdataset does not shuffle the shards or files within shards.")
        elif self.sampler_start_idx > 0:
            logger.warning(
                f"Sampler start index is set to {self.sampler_start_idx} but deterministic is False. "
                "This will result in a non-deterministic dataset."
            )

        self.loader = self._create_loader()

    def _pipeline(self, dataset: "wds.WebDataset") -> "wds.WebDataset":
        """Process the WebDataset through the loader's pipeline.

        Subclasses should override this method to define their data processing.
        The base implementation is an identity function (no processing).

        Args:
            dataset: The raw WebDataset

        Returns:
            Processed WebDataset
        """
        return dataset

    def _create_loader(self) -> "wds.WebLoader":
        """Create the dataset and dataloader using this loader's pipeline.

        This handles both deterministic and non-deterministic modes internally.
        Calls self._pipeline to process the data.

        Returns:
            WebLoader wrapping the dataset
        """
        # force single shard per worker for validation
        num_shards = None if self.shards_per_worker is None else world_size() * (self.num_workers or 1)

        if self.deterministic:
            dataset = DeterministicWDS(
                wds_tags=self.datatags,
                pipeline_fn=self._pipeline,
                batch_size=self.batch_size,
                repeat=self.train,
                ignore_index_paths=self.ignore_index_paths,
                sampler_start_idx=self.sampler_start_idx,
                shard_count_file=self.shard_count_file,
                shard_start=self.shard_start,
                shard_end=self.shard_end,
                num_shards=num_shards,
                empty_check=self.empty_check,
                partial=self.partial,
                cache_path=self.cache_path,
            )
        else:
            dataset = create_base_webdataset(
                wds_tags=self.datatags,
                train=self.train,
                cache_path=self.cache_path,
                ignore_index_paths=self.ignore_index_paths,
                shard_start=self.shard_start,
                shard_end=self.shard_end,
                num_shards=num_shards,
                empty_check=self.empty_check,
                shuffle_size=self.shuffle_size if self.train else 0,
            )
            dataset = self._pipeline(dataset)

        loader = wds_to_dataloader(
            dataset,
            batch_size=None if self.deterministic else self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            partial=self.partial,
        )

        return loader

    def __iter__(self):
        """Iterate over the dataloader."""
        return iter(self.loader)


# =============================================================================
# WebDataset creation
# =============================================================================


class DeterministicWDS(IterableDataset):
    """Base class for deterministic WebDataset loaders that support resumable iteration.

    The full dataset is read in order (i.e., every shard and every file in that shard
    that is not ignored) and all workers across ranks read their corresponding slice.
    """

    def __init__(
        self,
        wds_tags: List[str],
        batch_size: int = 1,
        repeat: bool = True,
        ignore_index_paths: List[str] | None = None,
        sampler_start_idx: int = 0,
        shard_count_file: str | None = None,
        shard_start: int = 0,
        shard_end: int | None = None,
        num_shards: int | None = None,
        empty_check: bool = True,
        partial: bool = False,
        cache_path: str | None = None,
        pipeline_fn: Callable[[wds.WebDataset], wds.WebDataset] | None = None,
    ):
        """Initialize the deterministic WebDataset.

        Args:
            wds_tags: List of WDS tags in format 'WDS:<PATH>'
            batch_size: Batch size for the loader
            repeat: Whether to repeat the dataset indefinitely
            ignore_index_paths: List of JSON files specifying files to skip
            sampler_start_idx: Start index for resuming iteration
            shard_count_file: Path to JSON file with shard sample counts
            shard_start: Start index for shard selection
            shard_end: End index for shard selection (inclusive)
            num_shards: Target number of shards
            empty_check: Whether to check for empty URL list
            partial: Whether to allow partial batches
            cache_path: Path to cache directory
            pipeline_fn: Function that processes the WebDataset
        """
        super().__init__()
        self.cache_path = cache_path
        self.empty_check = empty_check
        self.sampler_start_idx = sampler_start_idx
        self.partial = partial
        self.batch_size = batch_size
        self.repeat = repeat
        self.pipeline_fn = pipeline_fn

        # Get URLs and ignore index
        self.all_urls = get_shards(wds_tags, shard_start, shard_end=shard_end, num_shards=num_shards)
        self.ignore_index = get_ignore_index(ignore_index_paths, shard_list=self.all_urls)

        if shard_count_file is None:
            logger.warning("Shard count file is not provided, will not be able to fast-forward the urls.")
            self.shard_count = None
        else:
            self.shard_count = get_shard_count(
                shard_count_file, ignore_index=self.ignore_index, shard_list=self.all_urls
            )

        # Create cache directory
        if self.cache_path is not None:
            logger.info("Creating WebDataset cache directory")
            os.makedirs(self.cache_path, exist_ok=True)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        start_idx = self.sampler_start_idx
        if self.shard_count is not None:
            # Fast-forward URLs until offset is reached
            shard_idx = 0
            for idx, url in enumerate(itertools.cycle(self.all_urls)):
                new_start_idx = start_idx - self.shard_count[shard_to_name(url)]
                if new_start_idx >= 0:
                    start_idx = new_start_idx
                else:
                    shard_idx = idx % len(self.all_urls)
                    break

            assert (
                sum(self.shard_count[shard_to_name(url)] for url in self.all_urls[:shard_idx]) <= self.sampler_start_idx
            )
            logger.debug(
                f"Fast-forwarding to shard {shard_idx} (with start index {start_idx}) "
                f"for worker {worker_id} of {num_workers} worker(s)"
            )
            self.all_urls = self.all_urls[shard_idx:] + self.all_urls[:shard_idx]

        # Calculate offset and splitsize for this specific worker
        offset = start_idx + worker_id + num_workers * get_rank()
        splitsize = num_workers * world_size()

        # Build the webdataset
        dataset = wds.WebDataset(
            self.all_urls,
            cache_dir=self.cache_path,
            nodesplitter=None,
            workersplitter=None,
            shardshuffle=False,
            handler=wds.handlers.warn_and_continue,
            empty_check=self.empty_check,
        )

        if self.repeat:
            dataset = dataset.repeat()

        # Filter ignored files
        if len(self.ignore_index) > 0:
            filter_fn = functools.partial(filter_by_index, ignore_index=self.ignore_index)
            dataset = dataset.select(filter_fn)

        # Jump to offset and skip splitsize samples for this worker
        dataset = dataset.slice(offset, sys.maxsize, splitsize)

        # Process and batch the dataset
        if self.pipeline_fn is not None:
            dataset = self.pipeline_fn(dataset)
        dataset = batch_wds_dataset(dataset, self.batch_size, partial=self.partial)

        yield from dataset


def create_base_webdataset(
    wds_tags: List[str],
    train: bool = True,
    cache_path: str | None = None,
    ignore_index_paths: List[str] | None = None,
    shard_start: int = 0,
    shard_end: int | None = None,
    num_shards: int | None = None,
    empty_check: bool = True,
    shuffle_size: int = 1000,
) -> wds.WebDataset:
    """Create a base WebDataset with common configuration.

    Args:
        wds_tags: List of WDS tags
        train: Whether this is a training dataset
        cache_path: Path to cache directory
        ignore_index_paths: List of JSON files specifying files to skip
        shard_start: Start index for shard selection
        shard_end: End index for shard selection
        num_shards: Target number of shards
        empty_check: Whether to check for empty URL list
        shuffle_size: Size of shuffle buffer (0 for no shuffle)

    Returns:
        Configured WebDataset ready for processing
    """
    all_urls = get_shards(wds_tags, shard_start, shard_end=shard_end, num_shards=num_shards)
    ignore_index = get_ignore_index(ignore_index_paths, shard_list=all_urls)

    # Create cache directory
    if cache_path is not None:
        logger.info("Creating WebDataset cache directory")
        os.makedirs(cache_path, exist_ok=True)

    dataset = wds.WebDataset(
        all_urls,
        resampled=train,
        nodesplitter=None if train else _nodesplitter,
        cache_dir=cache_path,
        shardshuffle=False,
        handler=wds.handlers.warn_and_continue,
        empty_check=empty_check,
    ).shuffle(shuffle_size if train else 0)

    # Apply ignore filter
    if len(ignore_index) > 0:
        filter_fn = functools.partial(filter_by_index, ignore_index=ignore_index)
        dataset = dataset.select(filter_fn)

    return dataset
