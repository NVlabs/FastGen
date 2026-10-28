# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for WebDataset loaders.

This module provides shared functionality for all WDS loaders including:
- Shard discovery (local and S3) and selection
- Ignore indices and per-shard sample counts
- ``BaseWDSLoader`` with its shuffled training dataset and its deterministic, resumable ``DeterministicWDS``
- Batching and ``DataLoader`` wrapping

Deterministic loading
---------------------
``sampler_start_idx`` counts samples before decoding, after removing the union of all ``ignore_index_paths``.
The trainer's resume offset matches the batches it received only if each remaining sample produces exactly one
output. A sample that fails to decode or preprocess, lacks a required field, or is dropped by the pipeline is
skipped with a warning naming it. Each skip pulls that worker's next sample forward, so the stream runs ahead of the
trainer's offset and a resume repeats one sample per skip: with one worker (the deterministic default) the last
samples before the checkpoint, with more the samples just after it that the skipping worker had already delivered.
``MAX_CONSECUTIVE_SKIPS`` skips in a row on one worker raise, since they point to a systematic problem.
``strict=True`` raises on the first such sample instead. ``presets_filter`` is rejected either way, because it drops
samples systematically; precompute its exclusions into ``ignore_index_paths``.

``DeterministicWDS`` documents the sample order. It is independent of the worker count only for a fixed shard
selection (setting ``shards_per_worker`` makes the selection depend on the worker count) and while no sample is
skipped, since a skip shifts only its worker's later batches. It requires the DataLoader's default in-order delivery.
Only ``train=True`` repeats the stream; otherwise a start index at or beyond the end yields no batches.

A ``shard_count_file`` must hold accurate raw counts (before ignore indices), and every ignored key must exist in
its shard, because the number of ignored keys is subtracted from these counts. Counts are checked for shards that
are read but not for shards skipped on resume; rebuild stale counts before resuming, or omit the file to read
through the offset.

Finite and distributed iteration
--------------------------------
Finite workers or ranks without shards finish normally. Ranks split the data only under FastGen's CUDA process
group (see ``fastgen.utils.distributed.world_size``). The trainer advances training and validation loaders through
``synced_iter``: an iterator error or unequal exhaustion raises on every rank before the next model step. When all
ranks run out together, iteration ends normally and the next training batch request raises ``StopIteration``, so
training loaders should repeat (``train=True``) to supply ``max_iter`` iterations. ``synced_iter`` does not compare
batch sizes. With ``drop_last`` (the default), a finite deterministic stream ends at its last complete group of
``world_size`` samples, so every rank reads the same number and yields the same batches. Skipped samples (unless
``strict=True``) and shuffled loading, which splits whole shards across ranks, can still leave ranks unequal: balance
such finite validation data or cap it with ``MAX_VAL_STEPS``; ``shards_per_worker`` balances shard URLs, not the
samples that remain after filtering.
"""

import functools
import itertools
import json
import os
import shlex
import signal
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

import torch
import webdataset as wds
from torch.utils.data import IterableDataset, default_collate
from webdataset.gopen import Pipe, gopen_schemes

import fastgen.utils.logging_utils as logger
import fastgen.utils.s3 as s3
from fastgen.datasets.samplers import normalize_sampler_start_idx
from fastgen.utils.basic_utils import ensure_trailing_slash
from fastgen.utils.distributed import get_rank, local_world_size, world_size


# =============================================================================
# Validation
# =============================================================================


def validate_wds_config(sampler_start_idx=0, num_shards=None, deterministic=False, presets_filter=None):
    """Reject settings that could silently change the selected sample stream."""
    if sampler_start_idx < 0:
        raise ValueError("sampler_start_idx must be nonnegative")
    if num_shards is not None and num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if deterministic and presets_filter:
        raise ValueError(
            "deterministic=True does not support presets_filter: precompute these exclusions in "
            "ignore_index_paths so shard counts and sampler_start_idx describe the same sample stream."
        )


def validate_ignore_index(data, source):
    """A string or non-string keys would silently exclude the wrong samples."""
    for shard, files in data.items():
        if not isinstance(files, list) or not all(isinstance(key, str) for key in files):
            raise ValueError(f"Ignore index {source}: {shard} must contain a list of sample keys")


def validate_shard_counts(counts):
    """Negative effective counts corrupt shard skipping and epoch lengths."""
    for shard, count in counts.items():
        if count < 0:
            raise ValueError(f"Negative sample count for {shard}; check shard counts and ignore_index_paths")


def check_repeat_samples(has_samples, repeat):
    """Finite streams may be empty; repeating an empty pass cannot make progress."""
    if repeat and not has_samples:
        raise ValueError("No samples remain in the selected shards after applying ignore_index_paths")


# Skipping this many samples, or failing this many shard reads, in a row points to a systematic problem (e.g. a wrong
# key_map or expired storage credentials) rather than bad data, and a repeating stream whose samples or shards all
# fail would otherwise search forever without yielding a batch.
MAX_CONSECUTIVE_SKIPS = 100


def process_samples(samples, pipeline_fn, strict=False):
    """Run ``pipeline_fn`` on each sample separately, which must yield exactly one output per sample.

    Both loading modes process samples this way. A sample that fails or yields nothing is skipped with a warning
    (in deterministic loading this shifts the resume offset by one), and ``MAX_CONSECUTIVE_SKIPS`` skips in a row
    raise; ``strict=True`` raises on the first one instead. Only the selected rank/worker's samples are decoded.
    """
    skipped = 0
    for sample in samples:
        name = f"{sample.get('__url__', '')}:{sample.get('__key__', '')}"
        reason = "the pipeline dropped it"
        try:
            outputs = [sample] if pipeline_fn is None else list(pipeline_fn(wds.FluidWrapper([sample])))
        except Exception as error:
            # Name the sample: WebDataset's DecodingError has an empty message (its cause holds the reason).
            if strict:
                raise ValueError(f"Cannot process sample {name}") from error
            outputs, reason = [], repr(error.__cause__ or error)
        if len(outputs) > 1:
            raise ValueError(f"The pipeline produced {len(outputs)} outputs for sample {name}")
        if outputs and outputs[0] is not None:
            skipped = 0
            yield outputs[0]
            continue
        if strict:
            raise ValueError(
                f"The pipeline dropped sample {name}. "
                "Use ignore_index_paths to exclude samples without changing resume offsets."
            )
        skipped += 1
        logger.warning(f"Skipping sample {name}: {reason}")
        if skipped == MAX_CONSECUTIVE_SKIPS:
            raise ValueError(
                f"{skipped} samples in a row failed or were dropped, the last one {name} ({reason}); "
                "check the data and key_map, or set strict=True to raise on the first failure"
            )


def check_shard_counts(samples, urls, counts):
    """Check visited shards against counts keyed by URL, including adjacent repeated URLs.

    Zero-count shards have no nonignored samples, so exclude them from the expected
    sequence. Consecutive visits to the same URL form one group in both sequences.
    """
    nonempty_urls = (url for url in urls if counts[url] > 0)
    actual_groups = iter(itertools.groupby(samples, key=lambda sample: sample["__url__"]))
    for url, visits in itertools.groupby(nonempty_urls):
        expected = counts[url] * sum(1 for _ in visits)
        actual_url, shard_samples = next(actual_groups, (None, ()))
        if actual_url != url:
            raise ValueError(f"Shard count mismatch: expected samples from {url}, next sampled shard was {actual_url}")
        count = 0
        for sample in shard_samples:
            count += 1
            if count > expected:
                raise ValueError(f"Shard count mismatch for {url}: more than {expected} nonignored samples")
            yield sample
        if count != expected:
            raise ValueError(f"Shard count mismatch for {url}: expected {expected} nonignored samples, read {count}")
    extra = next(actual_groups, None)
    if extra is not None:
        raise ValueError(f"Shard count mismatch: unexpected nonignored samples from {extra[0]}")


# =============================================================================
# Shard Utilities
# =============================================================================


def shard_to_name(shard: str) -> str:
    """Convert a shard path or URI to a name by extracting the filename.

    Example:
        '<path>/00012.tar' -> '00012.tar'
        's3://<bucket>/<prefix>/00012.tar' -> '00012.tar'
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


def _open_s3_shard(url: str, mode: str = "rb", bufsize: int = 8192) -> Pipe:
    """Open an ``s3://`` shard for WebDataset by streaming it with retryable bounded ranged GETs."""
    if mode != "rb":
        raise ValueError(f"S3 shards are read-only: {url}")
    # Popen reports a child killed by SIGPIPE (the reader stopped mid-shard) as -SIGPIPE, which is not a failure.
    command = [sys.executable, "-m", "scripts.resumable_s3_download", url]
    return Pipe(command, mode=mode, bufsize=bufsize, ignore_status=[-signal.SIGPIPE])


# Registered at import: DataLoader workers import this module when they unpickle the dataset, before opening a shard.
gopen_schemes["s3"] = _open_s3_shard


def get_shards(
    wds_tags: List[str],
    shard_start: int = 0,
    shard_end: int | None = None,
    num_shards: int | None = None,
) -> List[str]:
    """Get the shards from the WDS tags.

    Supports local file paths and S3 paths. S3 shard discovery uses ``s5cmd``, and S3 shards are returned as
    ``s3://`` URIs, which WebDataset reads through ``_open_s3_shard`` with retryable bounded ranged GETs.

    Args:
        wds_tags: List of WDS tags in format 'WDS:<PATH>'
        shard_start: Start index for the list of shards
        shard_end: End index for the list of shards (inclusive)
        num_shards: Target number of shards (will truncate or repeat)

    Returns:
        List of shard URLs
    """
    validate_wds_config(num_shards=num_shards)

    # Validate the dataset tags
    for d in wds_tags:
        if not d.startswith("WDS"):
            raise ValueError("Cannot mix and match WDS datasets with other types")

    prefixes = get_prefixes(wds_tags)
    all_urls = []

    for prefix in prefixes:
        if s3.is_uri(prefix):
            prefix = ensure_trailing_slash(prefix)
            urls = []
            with os.popen(f"s5cmd ls {shlex.quote(prefix)}", "r") as proc:
                for line in proc:
                    # Date, time and size precede the object name, which may itself contain spaces.
                    listed_path = line.strip().split(maxsplit=3)[-1] if line.strip() else ""
                    if not listed_path.endswith(".tar"):
                        continue
                    # s5cmd versions may print either the full URI or a path
                    # relative to the listed prefix. Accept both forms.
                    s3_path = listed_path if s3.is_uri(listed_path) else prefix + listed_path.lstrip("/")
                    urls.append(s3_path)
                # Exiting the block discards close()'s exit status, so read it here; closing again is a no-op.
                status = proc.close()
            if status:
                raise RuntimeError(
                    f"s5cmd ls {prefix} failed with status {status}; install s5cmd and check the S3 credentials"
                )
            all_urls.extend(urls)
        elif Path(prefix).is_dir():
            # Local directory with tar files
            for f in Path(prefix).iterdir():
                if f.is_file() and f.suffix.lower() == ".tar":
                    all_urls.append(str(f))
        else:
            raise ValueError(f"Invalid prefix: {prefix}")

    # Sort by shard index, falling back to lexicographic shard names
    try:
        all_urls.sort(key=shard_to_idx)
    except ValueError as e:
        logger.warning(f"Shard names are not all numeric ({e}); selecting shards in lexicographic order.")
        all_urls.sort(key=shard_to_name)

    # Truncate or repeat shards
    stop = int(shard_end + 1) if shard_end is not None else None
    all_urls = all_urls[max(shard_start, 0) : stop]
    if not all_urls:
        raise ValueError(
            f"No shards remaining in {wds_tags} with shard_start={shard_start}, shard_end={shard_end}, and num_shards={num_shards}."
        )

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

    Raises:
        OSError, ValueError: If an ignore index cannot be read or parsed; skipping it would train on the
            files it excludes.
    """
    ignore_index: Dict[str, Set[str]] = {}

    if ignore_index_paths is not None:
        for p in ignore_index_paths:
            logger.info(f"Loading ignore index from {p}")

            if s3.is_uri(p):
                with os.popen(f"s5cmd cat {shlex.quote(p)}", "r") as f:
                    json_content = f.read()
                if not json_content.strip():
                    raise ValueError(f"Empty or failed to read S3 ignore index: {p}")
                data = json.loads(json_content)
            else:
                with open(p, mode="r") as f:
                    data = json.load(f)

            validate_ignore_index(data, p)
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
) -> Dict[str, int]:
    """Load shard count from a JSON file.

    Args:
        shard_count_file: Path to JSON file with shard counts
        ignore_index: Optional ignore index to adjust counts

    Returns:
        Dictionary mapping shard names to sample counts
    """
    if s3.is_uri(shard_count_file):
        try:
            with os.popen(f"s5cmd cat {shlex.quote(shard_count_file)}", "r") as f:
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
    validate_shard_counts(shard_count)

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


def _nodesplitter(src, rank: int, size: int):
    yield from itertools.islice(src, rank, None, size)


def _collate_samples(samples):
    """Reject unequal nested mapping keys before ``default_collate`` can silently discard them."""

    def validate(values):
        first = values[0]
        if isinstance(first, Mapping):
            if any(not isinstance(value, Mapping) or value.keys() != first.keys() for value in values):
                raise ValueError(
                    "Cannot collate samples with different mapping keys (e.g. clips with and without audio); "
                    "use batch_size=1 or exclude them via ignore_index_paths"
                )
            for key in first:
                validate([value[key] for value in values])
        elif isinstance(first, (list, tuple)):
            for items in zip(*values):
                validate(items)

    validate(samples)
    return default_collate(samples)


def batch_wds_dataset(dataset: wds.WebDataset, batch_size: int, partial: bool = False) -> wds.WebDataset:
    """Batch a webdataset.

    Args:
        dataset (wds.WebDataset): Input webdataset.
        batch_size (int): Batchsize of the loader.
        partial (bool, optional): Whether to return partial batches. Defaults to False.

    Returns:
        wds.WebDataset: Batched webdataset.
    """
    # ``process_samples`` yields only non-``None`` samples, so nothing needs filtering before collation.
    return dataset.batched(batch_size, partial=partial, collation_fn=_collate_samples)


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
        Dictionary with idx (the generic per-sample id shared by every loader), fname and shard fields
    """
    return {
        "idx": item["__key__"],
        "fname": item["__key__"],
        "shard": item.get("__url__", ""),
    }


class BaseWDSLoader:
    """Base class for WebDataset loaders.

    It picks the number of workers, builds the dataset (shuffled and resampled, or read in order with
    ``deterministic=True``, which supports resuming), and iterates batches. Subclasses set their own attributes
    before calling ``super().__init__()``, which builds the loader, and override ``_pipeline`` to filter,
    decode, and preprocess samples; ``process_samples`` runs it on each sample. Samples excluded on purpose
    belong in ``_prefilter``, so they never count as failures.
    """

    def __init__(
        self,
        datatags: List[str],
        batch_size: int,
        num_workers: int | None = None,
        train: bool = True,
        deterministic: bool = False,
        strict: bool = False,
        sampler_start_idx: int | None = None,
        shard_count_file: str | None = None,
        ignore_index_paths: List[str] | None = None,
        shard_start: int = 0,
        shard_end: int | None = None,
        shards_per_worker: int | None = None,
        partial: bool = False,
        drop_last: bool = True,
        shuffle_size: int = 1000,
    ):
        """Initialize common loader attributes.

        Args:
            datatags: List of WDS tags in format 'WDS:<PATH>'
            batch_size: Batch size for the dataloader
            num_workers: Number of workers (None for auto)
            train: Whether this is a training dataset
            deterministic: Whether to use deterministic iteration
            strict: Raise on a sample that fails to decode or preprocess, or on an unreadable shard, instead of
                skipping it with a warning (deterministic loading always raises on unreadable shards). With
                ``deterministic`` this keeps resume offsets exact: each skip makes a resume repeat one sample (see
                the module docstring). ``MAX_CONSECUTIVE_SKIPS`` skips in a row raise either way.
            sampler_start_idx: Global start sample of deterministic iteration. A ``None`` set in a config lets the
                trainer choose it: the resume offset for training, 0 for validation. An unresolved ``None`` defaults
                to 0 with a warning.
            shard_count_file: Path to shard count file, which lets a deterministic resume skip whole shards
            ignore_index_paths: List of JSON files specifying files to skip
            shard_start: Start index for shard selection
            shard_end: End index for shard selection (inclusive)
            shards_per_worker: If not None, force this many shards per worker
                (``world_size * num_workers * shards_per_worker`` shards in total) by truncating/repeating shards.
            partial: Whether to allow partial batches
            drop_last: With ``deterministic`` and ``train=False``, end the stream at its last complete group of
                ``world_size`` samples so every rank reads the same number of samples and, unless some are skipped,
                yields the same batches; ``False`` reads every sample, which gives the first ranks one extra each.
            shuffle_size: Size of shuffle buffer (0 for no shuffle, only used when train=True)
        """
        self.datatags = datatags
        self.batch_size = batch_size
        self.train = train
        self.deterministic = deterministic
        self.strict = strict
        self.sampler_start_idx = normalize_sampler_start_idx(sampler_start_idx)
        self.shard_count_file = shard_count_file
        self.ignore_index_paths = ignore_index_paths
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.shards_per_worker = shards_per_worker
        self.partial = partial
        self.drop_last = drop_last
        self.shuffle_size = shuffle_size

        # Compute num_workers
        self.num_workers = num_workers
        if self.num_workers is None:
            if deterministic:
                self.num_workers = 1
            else:
                # The workers share the node's CPUs with the ranks on that node, not with the whole world.
                self.num_workers = max(1, min((os.cpu_count() or 1) // local_world_size(), 8))
            logger.info(f"Automatically setting num_workers to {self.num_workers}")

        # Log deterministic warnings
        if self.deterministic:
            if self.train:
                logger.warning("Deterministic webdataset does not shuffle the shards or files within shards.")
        elif self.sampler_start_idx > 0:
            # Shuffled, resampled iteration has no position to resume.
            logger.info(f"Ignoring sampler_start_idx={self.sampler_start_idx} since deterministic is False.")

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

    def _prefilter(self, dataset: "wds.WebDataset") -> "wds.WebDataset":
        """Drop raw samples on purpose before ``_pipeline`` runs; these never count as failures.

        Shuffled loading only: deterministic loading counts its resume offset in raw samples, so it must not drop
        any outside ``ignore_index_paths``.
        """
        return dataset

    def _create_loader(self) -> "wds.WebLoader":
        """Create the dataset and dataloader using this loader's pipeline.

        This handles both deterministic and non-deterministic modes internally.
        Calls self._pipeline to process the data.

        Returns:
            WebLoader wrapping the dataset
        """
        # Force a fixed number of shards per worker (e.g. one for validation) by truncating/repeating shards.
        num_shards = (
            None if self.shards_per_worker is None else world_size() * (self.num_workers or 1) * self.shards_per_worker
        )

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
                partial=self.partial,
                drop_last=self.drop_last,
                strict=self.strict,
            )
        else:
            dataset = create_base_webdataset(
                wds_tags=self.datatags,
                train=self.train,
                ignore_index_paths=self.ignore_index_paths,
                shard_start=self.shard_start,
                shard_end=self.shard_end,
                num_shards=num_shards,
                shuffle_size=self.shuffle_size if self.train else 0,
                strict=self.strict,
            )
            dataset = self._prefilter(dataset)
            dataset = wds.FluidWrapper(functools.partial(process_samples, dataset, self._pipeline, self.strict))

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


def _resume_shards(urls: List[str], counts: Dict[str, int] | None, start: int, repeat: bool):
    """Return the remaining shard order and the offset within its first shard."""
    if counts is None:
        return urls, start
    total = sum(counts[shard_to_name(url)] for url in urls)
    if total == 0 or (not repeat and start >= total):
        return [], 0
    if repeat:
        start %= total
    for index, url in enumerate(urls):
        count = counts[shard_to_name(url)]
        if start < count:
            remaining = urls[index:] + urls[:index] if repeat else urls[index:]
            return remaining, start
        start -= count
    raise AssertionError("Validated shard counts did not contain the resume offset")


def _complete_group_samples(samples, rank: int, num_ranks: int):
    """Yield sample ``rank`` of each complete group of ``num_ranks`` consecutive samples.

    Dropping the final incomplete group gives every rank the same number of samples. Raw samples can be large, so
    only the selected sample of a group is held.
    """
    samples = iter(samples)
    while True:
        selected, count = None, 0
        for count, sample in enumerate(itertools.islice(samples, num_ranks), start=1):
            if count == rank + 1:
                selected = sample
        if count < num_ranks:
            return
        yield selected


class DeterministicWDS(IterableDataset):
    """Base class for deterministic WebDataset loaders that support resumable iteration.

    The full dataset is read in order (i.e., every shard and every file in that shard
    that is not ignored). Rank r reads every world_size-th sample starting at
    sampler_start_idx + r; with ``drop_last``, a finite stream stops at its last complete
    group of world_size samples, so every rank reads the same number. Workers take whole
    batches round-robin, so the combined order is independent of their number, including
    when resuming mid-epoch, as long as no sample is skipped.

    Repeated epochs form one continuous stream; batches can cross epoch boundaries.
    An empty pass raises when repeating; finite streams may finish empty.
    Count metadata can skip whole shards and must match the data and ignore indices.
    Processing should yield one output per input: samples that fail or are dropped are
    skipped with a warning (``strict=True`` raises); use explicit ignore indices for
    exclusions. Determinism covers sample order, not random augmentations.
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
        partial: bool = False,
        drop_last: bool = True,
        pipeline_fn: Callable[[wds.WebDataset], wds.WebDataset] | None = None,
        strict: bool = False,
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
            partial: Whether to allow partial batches
            drop_last: Without ``repeat``, end at the last complete group of world-size samples so every rank reads
                the same number; ``False`` reads every sample, which gives the first ranks one extra sample each
            pipeline_fn: Function that processes the WebDataset
            strict: Raise when ``pipeline_fn`` fails on or drops a sample instead of skipping it with a warning
        """
        super().__init__()
        self.sampler_start_idx = normalize_sampler_start_idx(sampler_start_idx)
        self.partial = partial
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.repeat = repeat
        self.pipeline_fn = pipeline_fn
        self.strict = strict
        # Spawned workers have no process group; capture the rank in the parent.
        self.rank, self.num_ranks = get_rank(), world_size()
        validate_wds_config(sampler_start_idx=self.sampler_start_idx)

        # Get URLs and ignore index
        self.all_urls = get_shards(wds_tags, shard_start, shard_end=shard_end, num_shards=num_shards)
        self.ignore_index = get_ignore_index(ignore_index_paths, shard_list=self.all_urls)

        if shard_count_file is None:
            if self.sampler_start_idx > 0:
                logger.warning(
                    f"Without a shard count file, every worker reads the {self.sampler_start_idx} skipped samples to "
                    "reach sampler_start_idx; set shard_count_file to skip whole shards."
                )
            self.shard_count = None
        else:
            self.shard_count = get_shard_count(shard_count_file, ignore_index=self.ignore_index)
            check_repeat_samples(any(self.shard_count[shard_to_name(url)] for url in self.all_urls), self.repeat)

    def _sample_stream(self, urls):
        dataset = wds.WebDataset(
            urls,
            nodesplitter=None,
            workersplitter=None,
            shardshuffle=False,
            handler=wds.handlers.reraise_exception,
            empty_check=False,
        )
        dataset = dataset.select(functools.partial(filter_by_index, ignore_index=self.ignore_index))
        counts = None if self.shard_count is None else {url: self.shard_count[shard_to_name(url)] for url in urls}
        while True:
            seen = False
            samples = dataset if counts is None else check_shard_counts(dataset, urls, counts)
            for sample in samples:
                seen = True
                yield sample
            check_repeat_samples(seen, self.repeat)
            if not self.repeat:
                return

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        urls, start = _resume_shards(self.all_urls, self.shard_count, self.sampler_start_idx, self.repeat)
        if not urls:
            return

        stream = self._sample_stream(urls)
        if self.drop_last and not self.repeat:
            rank_samples = _complete_group_samples(itertools.islice(stream, start, None), self.rank, self.num_ranks)
        else:
            rank_samples = itertools.islice(stream, start + self.rank, None, self.num_ranks)
        # DataLoader fetches worker batches round-robin. Partition *batches*, not
        # individual samples, so restarting worker 0 continues the global stream.
        samples = (sample for i, sample in enumerate(rank_samples) if (i // self.batch_size) % num_workers == worker_id)
        processed = functools.partial(process_samples, samples, self.pipeline_fn, self.strict)
        dataset = wds.FluidWrapper(processed)
        yield from batch_wds_dataset(dataset, self.batch_size, partial=self.partial)


class _ShardReadHandler:
    """WebDataset handler that skips unreadable shard data with a warning, or re-raises the error with ``strict``.

    ``MAX_CONSECUTIVE_SKIPS`` failures in a row raise; ``reset`` marks each sample read successfully.
    """

    def __init__(self, strict: bool):
        self.strict = strict
        self.failures = 0

    def __call__(self, error: Exception) -> bool:
        if self.strict:
            raise error
        self.failures += 1
        # WebDataset appends the shard URL to the error's arguments.
        logger.warning(f"Skipping unreadable shard data: {error!r}")
        if self.failures == MAX_CONSECUTIVE_SKIPS:
            raise ValueError(
                f"{self.failures} shard reads in a row failed, the last with {error!r}; "
                "check the shard paths and storage credentials"
            ) from error
        return True

    def reset(self, sample: dict) -> dict:
        self.failures = 0
        return sample


def create_base_webdataset(
    wds_tags: List[str],
    train: bool = True,
    ignore_index_paths: List[str] | None = None,
    shard_start: int = 0,
    shard_end: int | None = None,
    num_shards: int | None = None,
    shuffle_size: int = 1000,
    strict: bool = False,
) -> wds.WebDataset:
    """Create a base WebDataset with common configuration.

    Unreadable shards are skipped with a warning, and ``MAX_CONSECUTIVE_SKIPS`` failed shard reads in a row raise;
    ``strict=True`` raises on the first one.

    Args:
        wds_tags: List of WDS tags
        train: Whether this is a training dataset
        ignore_index_paths: List of JSON files specifying files to skip
        shard_start: Start index for shard selection
        shard_end: End index for shard selection (inclusive)
        num_shards: Target number of shards
        shuffle_size: Size of shuffle buffer (0 for no shuffle)
        strict: Raise on the first unreadable shard instead of skipping it

    Returns:
        Configured WebDataset ready for processing
    """
    all_urls = get_shards(wds_tags, shard_start, shard_end=shard_end, num_shards=num_shards)
    ignore_index = get_ignore_index(ignore_index_paths, shard_list=all_urls)
    # Bind rank information before DataLoader creates isolated worker processes.
    rank, size = get_rank(), world_size()

    shard_errors = _ShardReadHandler(strict)
    dataset = wds.WebDataset(
        all_urls,
        resampled=train,
        nodesplitter=None if train else functools.partial(_nodesplitter, rank=rank, size=size),
        shardshuffle=False,
        handler=shard_errors,
        # Finite workers may have no assigned shards or only empty shards.
        empty_check=train,
    )
    # A resampled stream whose shards all fail would otherwise retry them forever without yielding a sample.
    dataset = dataset.map(shard_errors.reset)
    dataset = dataset.shuffle(shuffle_size if train else 0)

    # Apply ignore filter
    if len(ignore_index) > 0:
        filter_fn = functools.partial(filter_by_index, ignore_index=ignore_index)
        dataset = dataset.select(filter_fn)

    return dataset
