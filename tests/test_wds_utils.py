# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebDataset shard discovery, plus sample-order oracles and real worker/rank regressions for resumable streams."""

import datetime
import io
import itertools
import json
import multiprocessing as mp
import signal
import sys
import tarfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from webdataset.gopen import gopen, gopen_schemes

from fastgen.datasets import wds_utils
from fastgen.datasets.wds_dataloaders import WDSLoader
from fastgen.datasets.wds_utils import DeterministicWDS, get_ignore_index, get_shard_count, get_shards


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("nested_sequence", [False, True])
def test_batching_rejects_mixed_modalities_in_either_order(reverse, nested_sequence):
    samples = [{"video": torch.ones(2)}, {"video": torch.ones(2), "audio": torch.ones(3)}]
    if reverse:
        samples.reverse()
    samples = [{"real_raw": [value] if nested_sequence else value} for value in samples]
    dataset = wds_utils.wds.FluidWrapper(samples)
    with pytest.raises(ValueError, match="different mapping keys"):
        next(iter(wds_utils.batch_wds_dataset(dataset, batch_size=2)))


def test_batching_preserves_every_homogeneous_modality():
    samples = [{"real_raw": {"video": torch.full((2,), i), "audio": torch.full((3,), i)}} for i in range(2)]
    dataset = wds_utils.wds.FluidWrapper(samples)
    batch = next(iter(wds_utils.batch_wds_dataset(dataset, batch_size=2)))
    for modality in ("video", "audio"):
        assert torch.equal(
            batch["real_raw"][modality], torch.stack([sample["real_raw"][modality] for sample in samples])
        )


@pytest.mark.parametrize(
    ("start", "end", "count", "expected"),
    [(2, 3, None, [2, 3]), (0, 0, None, [0]), (2, None, None, [2, 3, 4, 5]), (2, 3, 5, [2, 3, 2, 3, 2])],
)
def test_shard_selection_uses_inclusive_absolute_end(tmp_path, start, end, count, expected):
    for index in range(6):
        (tmp_path / f"{index:05d}.tar").touch()
    urls = get_shards([f"WDS:{tmp_path}"], shard_start=start, shard_end=end, num_shards=count)
    assert [int(Path(url).stem) for url in urls] == expected


@pytest.mark.parametrize("count", [None, 3])
@pytest.mark.parametrize("empty_directory", [False, True])
def test_empty_shard_selection_has_a_useful_error(tmp_path, count, empty_directory):
    if not empty_directory:
        (tmp_path / "00000.tar").touch()
    with pytest.raises(ValueError, match="No shards remaining"):
        get_shards([f"WDS:{tmp_path}"], shard_start=1, num_shards=count)


@pytest.mark.parametrize("count", [0, -1])
def test_shard_count_must_be_positive(tmp_path, count):
    with pytest.raises(ValueError, match="num_shards"):
        get_shards([f"WDS:{tmp_path}"], num_shards=count)


def test_ignore_index_merges_files_and_keeps_only_listed_shards(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps({"00000.tar": ["a"], "00009.tar": ["z"]}))
    second.write_text(json.dumps({"00000.tar": ["b"]}))

    ignore_index = get_ignore_index([str(first), str(second)], shard_list=["/data/00000.tar"])

    assert ignore_index == {"00000.tar": {"a", "b"}}


@pytest.mark.parametrize(("content", "error"), [(None, FileNotFoundError), ("{not json", ValueError)])
def test_unreadable_ignore_index_is_an_error(tmp_path, content, error):
    """Skipping an unreadable ignore index would silently train on the samples it excludes."""
    path = tmp_path / "ignore.json"
    if content is not None:
        path.write_text(content)

    with pytest.raises(error):
        get_ignore_index([str(path)])


def test_empty_s3_ignore_index_is_an_error(monkeypatch):
    monkeypatch.setattr("fastgen.datasets.wds_utils.os.popen", lambda command, mode: io.StringIO(""))

    with pytest.raises(ValueError, match="s3://bucket/ignore.json"):
        get_ignore_index(["s3://bucket/ignore.json"])


@pytest.mark.parametrize("content", [{"00000.tar": "sample"}, {"00000.tar": [1]}])
def test_invalid_ignore_metadata_is_rejected(tmp_path, content):
    path = tmp_path / "ignore.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="Ignore index"):
        get_ignore_index([str(path)])


@pytest.mark.parametrize("shard_prefix", ["", "shard name "])
def test_s3_shards_are_uris_opened_through_the_resumable_stream(monkeypatch, shard_prefix):
    commands = []
    listing = f"""\
2026/09/07 00:00:00 20 s3://bucket/data/{shard_prefix}00002.tar
2026/09/07 00:00:00 10 {shard_prefix}00001.tar
2026/09/07 00:00:00 30 s3://bucket/data/not-a-shard.txt
"""

    def popen(command, mode):
        commands.append((command, mode))
        return io.StringIO(listing)

    monkeypatch.setattr("fastgen.datasets.wds_utils.os.popen", popen)

    urls = get_shards(["WDS:s3://bucket/data"])

    assert commands == [("s5cmd ls s3://bucket/data/", "r")]
    assert urls == [f"s3://bucket/data/{shard_prefix}00001.tar", f"s3://bucket/data/{shard_prefix}00002.tar"]

    opened = []
    monkeypatch.setattr(wds_utils, "Pipe", lambda command, **kwargs: opened.append((command, kwargs)))
    gopen(urls[0])
    # A download that dies from SIGPIPE only means the reader stopped mid-shard (see test_s3).
    assert opened == [
        (
            [sys.executable, "-m", "scripts.resumable_s3_download", urls[0]],
            {"mode": "rb", "bufsize": 8192, "ignore_status": [-signal.SIGPIPE]},
        )
    ]
    with pytest.raises(ValueError, match="read-only"):
        gopen(urls[0], "wb")


def _fixture(root, sizes=(2, 0, 3), ignore=False):
    root.mkdir(exist_ok=True)
    keys, excluded = [], {}
    for shard, size in enumerate(sizes):
        name = f"{shard:05d}.tar"
        shard_keys = [f"{shard:05d}_{i:05d}" for i in range(size)]
        with tarfile.open(root / name, "w") as archive:
            for key in shard_keys:
                content = key.encode()
                info = tarfile.TarInfo(key + ".txt")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        excluded[name] = shard_keys[1::2] if ignore else []
        keys.append([key for key in shard_keys if key not in excluded[name]])
    (root / "counts.json").write_text(json.dumps({f"{i:05d}.tar": n for i, n in enumerate(sizes)}))
    # Duplicate exclusions in separate files must only be subtracted once.
    for name in ["ignore-a.json", "ignore-b.json"]:
        (root / name).write_text(json.dumps(excluded))
    return keys


def _key(sample):
    return {"key": sample["__key__"]}


def _pipeline(dataset):
    return dataset.map(_key)


def _batches(keys, size, partial):
    return [keys[i : i + size] for i in range(0, len(keys), size) if partial or i + size <= len(keys)]


class _Timeout(BaseException):
    pass


@contextmanager
def _deadline(seconds=5):
    # Exception handlers in WebDataset must not swallow the timeout itself.
    def expired(*_):
        raise _Timeout("WebDataset did not make progress")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("counted,repeat,partial,drop_last", itertools.product([False, True], repeat=4))
@pytest.mark.parametrize("sizes,ignore", [((1,), False), ((2, 0, 3), True), ((2, 5, 1), False)])
def test_stream_matches_global_sample_oracle(tmp_path, monkeypatch, counted, repeat, partial, drop_last, sizes, ignore):
    shards = _fixture(tmp_path, sizes, ignore)
    for start_shard, end_shard, forced in [(0, None, None), (0, 0, None), (0, None, 5)]:
        selected = shards[start_shard : None if end_shard is None else end_shard + 1]
        if forced:
            selected = (selected * forced)[:forced]
        epoch = list(itertools.chain.from_iterable(selected))
        for ranks, workers, batch_size in [(1, 1, 1), (1, 4, 3), (2, 4, 2), (4, 2, 5), (4, 1, 3)]:
            for start in sorted({0, 1, len(epoch) - 1, len(epoch), len(epoch) + 1, 3 * len(epoch) + 2}):
                for rank, worker in {(0, 0), (ranks - 1, workers - 1)}:
                    monkeypatch.setattr(wds_utils, "get_rank", lambda: rank)
                    monkeypatch.setattr(wds_utils, "world_size", lambda: ranks)
                    dataset = DeterministicWDS(
                        [f"WDS:{tmp_path}"],
                        batch_size=batch_size,
                        repeat=repeat,
                        partial=partial,
                        drop_last=drop_last,
                        sampler_start_idx=start,
                        pipeline_fn=_pipeline,
                        shard_start=start_shard,
                        shard_end=end_shard,
                        num_shards=forced,
                        shard_count_file=str(tmp_path / "counts.json") if counted else None,
                        ignore_index_paths=[str(tmp_path / name) for name in ["ignore-a.json", "ignore-b.json"]],
                    )
                    # Workers do not inherit an initialized process group under spawn/forkserver.
                    monkeypatch.setattr(wds_utils, "get_rank", lambda: 0)
                    monkeypatch.setattr(wds_utils, "world_size", lambda: 1)
                    monkeypatch.setattr(
                        torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=worker, num_workers=workers)
                    )
                    if repeat:
                        rank_keys = [
                            epoch[(start + rank + i * ranks) % len(epoch)] for i in range(4 * workers * batch_size)
                        ]
                    else:
                        remaining = epoch[start:]
                        if drop_last:
                            remaining = remaining[: len(remaining) // ranks * ranks]
                        rank_keys = remaining[rank::ranks]
                    expected = _batches(rank_keys, batch_size, partial)[worker::workers][:4]
                    actual = [batch["key"] for batch in itertools.islice(dataset, 4)]
                    assert actual == expected, (ranks, workers, batch_size, start, rank, worker)


@pytest.mark.parametrize("workers", [0, 1, 2, 4])
@pytest.mark.parametrize("counted", [False, True])
def test_real_workers_resume_at_any_batch_boundary(tmp_path, workers, counted):
    epoch = _fixture(tmp_path, (13,))[0]

    def collect(start, n, worker_count):
        dataset = DeterministicWDS(
            [f"WDS:{tmp_path}"],
            batch_size=3,
            sampler_start_idx=start,
            pipeline_fn=_pipeline,
            shard_count_file=str(tmp_path / "counts.json") if counted else None,
        )
        loader = DataLoader(dataset, batch_size=None, num_workers=worker_count)
        return [batch["key"] for batch in itertools.islice(loader, n)]

    fresh = collect(0, 7, workers)
    assert fresh == _batches((epoch * 2)[:21], 3, False)
    # Resume between worker rounds, then change worker count at the next resume.
    assert collect(3, 4, workers) == fresh[1:5]
    assert collect(9, 4, 0 if workers else 2) == fresh[3:7]


def _distributed_loader_probe(rank, directory, rendezvous, method, workers):
    mp.set_start_method(method, force=True)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=60)
    )
    try:

        def collect(start=0, repeat=True, n=7):
            dataset = DeterministicWDS(
                [f"WDS:{directory}"],
                batch_size=3,
                repeat=repeat,
                partial=True,
                sampler_start_idx=start,
                pipeline_fn=_pipeline,
                shard_count_file=str(Path(directory) / "counts.json"),
            )
            loader = DataLoader(dataset, batch_size=None, num_workers=workers)
            return [batch["key"] for batch in itertools.islice(loader, n)]

        fresh = collect()
        resumed = collect(start=6, n=4)
        finite = collect(repeat=False, n=100)
        base = wds_utils.create_base_webdataset([f"WDS:{directory}"], train=False)
        loader = wds_utils.wds_to_dataloader(
            _pipeline(base), batch_size=3, num_workers=workers, pin_memory=False, partial=True
        )
        sparse = [batch["key"] for batch in loader]
        forced = wds_utils.create_base_webdataset([f"WDS:{directory}"], train=False, num_shards=2 * max(workers, 1))
        loader = wds_utils.wds_to_dataloader(
            _pipeline(forced), batch_size=3, num_workers=workers, pin_memory=False, partial=True
        )
        repeated = [key for batch in loader for key in batch["key"]]
        (Path(directory) / f"rank-{rank}.json").write_text(
            json.dumps({"fresh": fresh, "resumed": resumed, "finite": finite, "sparse": sparse, "repeated": repeated})
        )
        # Keep a fast rank alive until its peer has finished initializing and
        # consuming the loaders; otherwise Gloo can close during connectFullMesh.
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "method,workers", [("spawn", 0)] + list(itertools.product(["fork", "spawn", "forkserver"], [1, 4]))
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FastGen's shared world_size requires CUDA")
def test_real_ranks_with_isolated_workers_and_one_shard(tmp_path, method, workers):
    epoch = _fixture(tmp_path, (13,))[0]
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_distributed_loader_probe, args=(rank, str(tmp_path), str(tmp_path / "rendezvous"), method, workers)
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(120)
            assert not process.is_alive(), "Distributed WebDataset workers did not terminate"
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
    for rank in range(2):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        expected = [epoch[(rank + 2 * i) % len(epoch)] for i in range(21)]
        assert result["fresh"] == _batches(expected, 3, True)
        assert result["resumed"] == result["fresh"][1:5]
        assert result["finite"] == _batches(epoch[: len(epoch) // 2 * 2][rank::2], 3, True)
        assert result["sparse"] == (_batches(epoch, 3, True) if rank == 0 else [])
        assert Counter(result["repeated"]) == Counter(epoch * max(workers, 1))


@pytest.mark.parametrize("counted,repeat,start", itertools.product([False, True], [False, True], [0, 9]))
@pytest.mark.parametrize("sizes", [(0, 0), (2, 0, 3)])
def test_empty_sample_stream_terminates(tmp_path, counted, repeat, start, sizes):
    shards = _fixture(tmp_path, sizes)
    (tmp_path / "ignore-a.json").write_text(json.dumps({f"{i:05d}.tar": keys for i, keys in enumerate(shards)}))
    with _deadline():
        kwargs = dict(
            ignore_index_paths=[str(tmp_path / "ignore-a.json")],
            repeat=repeat,
            sampler_start_idx=start,
            shard_count_file=str(tmp_path / "counts.json") if counted else None,
        )
        if repeat:
            with pytest.raises(ValueError, match="No samples remain"):
                list(DeterministicWDS([f"WDS:{tmp_path}"], **kwargs))
        else:
            assert list(DeterministicWDS([f"WDS:{tmp_path}"], **kwargs)) == []


@pytest.mark.parametrize("deterministic,workers", itertools.product([False, True], [0, 4]))
@pytest.mark.parametrize("sizes", [(0, 0), (2, 0, 3)])
def test_finite_workers_allow_empty_streams(tmp_path, deterministic, workers, sizes):
    shards = _fixture(tmp_path, sizes)
    (tmp_path / "ignore-a.json").write_text(json.dumps({f"{i:05d}.tar": keys for i, keys in enumerate(shards)}))
    loader = WDSLoader(
        datatags=[f"WDS:{tmp_path}"],
        batch_size=2,
        num_workers=workers,
        train=False,
        deterministic=deterministic,
        key_map={"condition_raw": "txt"},
        ignore_index_paths=[str(tmp_path / "ignore-a.json")],
        partial=True,
    )
    with _deadline(30):
        assert list(loader) == []


def test_repeating_workers_reject_an_empty_stream(tmp_path):
    epoch = _fixture(tmp_path, (2,))[0]
    (tmp_path / "ignore-a.json").write_text(json.dumps({"00000.tar": epoch}))
    loader = WDSLoader(
        datatags=[f"WDS:{tmp_path}"],
        batch_size=2,
        num_workers=4,
        train=True,
        deterministic=True,
        key_map={"condition_raw": "txt"},
        ignore_index_paths=[str(tmp_path / "ignore-a.json")],
    )
    with _deadline(30), pytest.raises(ValueError, match="No samples remain"):
        next(iter(loader))


@pytest.mark.parametrize("workers", [0, 2])
def test_shuffled_training_fails_instead_of_spinning_when_no_shard_is_readable(tmp_path, workers):
    for index in range(2):
        (tmp_path / f"{index:05d}.tar").write_bytes(b"not a tar archive")
    loader = WDSLoader(
        datatags=[f"WDS:{tmp_path}"],
        batch_size=2,
        num_workers=workers,
        train=True,
        shuffle_size=0,
        key_map={"condition_raw": "txt"},
    )
    with _deadline(60), pytest.raises(ValueError, match="shard reads in a row failed"):
        next(iter(loader))


def test_shuffled_loading_skips_sporadic_unreadable_shards(tmp_path):
    epoch = _fixture(tmp_path, (3,))[0]
    (tmp_path / "00001.tar").write_bytes(b"not a tar archive")
    dataset = wds_utils.create_base_webdataset([f"WDS:{tmp_path}"], train=True, shuffle_size=0)
    with _deadline(30):
        keys = [sample["__key__"] for sample in itertools.islice(dataset, 300)]
    assert set(keys) == set(epoch)


def test_strict_shuffled_loading_raises_on_an_unreadable_shard(tmp_path):
    epoch = _fixture(tmp_path, (3,))[0]
    (tmp_path / "00001.tar").write_bytes(b"not a tar archive")

    def keys(strict):
        loader = WDSLoader(
            datatags=[f"WDS:{tmp_path}"],
            batch_size=1,
            num_workers=0,
            train=False,
            strict=strict,
            key_map={"condition_raw": "txt"},
        )
        return [key for batch in loader for key in batch["fname"]]

    with _deadline(30):
        assert keys(strict=False) == epoch
        with pytest.raises(tarfile.ReadError, match=r"00001\.tar"):
            keys(strict=True)


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("rejected", ["all", "first", "last"])
def test_downstream_filtering_fails_without_hanging(tmp_path, repeat, rejected):
    epoch = _fixture(tmp_path, (3,))[0]
    excluded = epoch if rejected == "all" else [epoch[0 if rejected == "first" else -1]]

    def filtered(dataset):
        return _pipeline(dataset.select(lambda sample: sample["__key__"] not in excluded))

    with _deadline(), pytest.raises(ValueError, match="pipeline dropped.*ignore_index_paths"):
        list(
            itertools.islice(DeterministicWDS([f"WDS:{tmp_path}"], repeat=repeat, pipeline_fn=filtered, strict=True), 6)
        )


@pytest.mark.parametrize("rejected", ["first", "last"])
def test_lenient_pipelines_skip_dropped_samples(tmp_path, rejected):
    epoch = _fixture(tmp_path, (3,))[0]
    excluded = epoch[0 if rejected == "first" else -1]

    def filtered(dataset):
        return _pipeline(dataset.select(lambda sample: sample["__key__"] != excluded))

    dataset = DeterministicWDS([f"WDS:{tmp_path}"], repeat=False, pipeline_fn=filtered)
    with _deadline():
        assert [key for batch in dataset for key in batch["key"]] == [key for key in epoch if key != excluded]


@pytest.mark.parametrize("rejected", ["all", "first"])
def test_repeating_lenient_streams_stop_only_after_consecutive_skips(tmp_path, rejected):
    epoch = _fixture(tmp_path, (3,))[0]
    excluded = epoch if rejected == "all" else epoch[:1]

    def filtered(dataset):
        return _pipeline(dataset.select(lambda sample: sample["__key__"] not in excluded))

    dataset = DeterministicWDS([f"WDS:{tmp_path}"], repeat=True, pipeline_fn=filtered)
    with _deadline(30):
        if rejected == "all":
            with pytest.raises(ValueError, match=f"{wds_utils.MAX_CONSECUTIVE_SKIPS} samples in a row"):
                next(iter(dataset))
        else:
            # One skip per pass never accumulates.
            keys = [key for batch in itertools.islice(dataset, 6) for key in batch["key"]]
            assert keys == epoch[1:] * 3


@pytest.mark.parametrize("deterministic", [False, True])
def test_lenient_loading_skips_missing_keys_and_corrupt_decoding(tmp_path, deterministic):
    with tarfile.open(tmp_path / "00000.tar", "w") as archive:
        for name, content in [("a.json", b"{}"), ("broken.json", b"{invalid"), ("bare.txt", b""), ("d.json", b"{}")]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    loader = WDSLoader(
        datatags=[f"WDS:{tmp_path}"],
        batch_size=1,
        num_workers=0,
        deterministic=deterministic,
        train=False,
        key_map={"real": "json"},
    )
    with _deadline(30):
        assert [batch["fname"] for batch in loader] == [["a"], ["d"]]


@pytest.mark.parametrize("deterministic", [False, True])
def test_loaders_whose_samples_all_fail_raise_instead_of_hanging(tmp_path, deterministic):
    _fixture(tmp_path, (3,))
    loader = WDSLoader(
        datatags=[f"WDS:{tmp_path}"],
        batch_size=1,
        num_workers=0,
        deterministic=deterministic,
        key_map={"real": "missing.npy"},
    )
    with _deadline(30), pytest.raises(ValueError, match="samples in a row"):
        next(iter(loader))


@pytest.mark.parametrize("deterministic,repeat", itertools.product([False, True], repeat=2))
def test_strict_loading_rejects_missing_keys_and_corrupt_decoding(tmp_path, deterministic, repeat):
    _fixture(tmp_path, (3,))
    with _deadline(), pytest.raises(ValueError, match="pipeline dropped"):
        list(
            WDSLoader(
                datatags=[f"WDS:{tmp_path}"],
                batch_size=2,
                num_workers=0,
                deterministic=deterministic,
                strict=True,
                train=repeat,
                key_map={"real": "missing.npy"},
            )
        )
    with tarfile.open(tmp_path / "00000.tar", "w") as archive:
        content = b"{invalid json"
        info = tarfile.TarInfo("broken.json")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    with _deadline(), pytest.raises(ValueError, match="Cannot process sample .*broken") as raised:
        list(
            WDSLoader(
                datatags=[f"WDS:{tmp_path}"],
                batch_size=2,
                num_workers=0,
                deterministic=deterministic,
                strict=True,
                train=repeat,
                key_map={"real": "json"},
            )
        )
    assert isinstance(raised.value.__cause__.__cause__, json.JSONDecodeError)


def test_deterministic_preset_filters_require_explicit_ignore_indices(tmp_path):
    with pytest.raises(ValueError, match="presets_filter.*ignore_index_paths"):
        WDSLoader(
            datatags=[f"WDS:{tmp_path}"],
            batch_size=2,
            deterministic=True,
            presets_filter={"empty_keys": {"keys": ["txt"]}},
        )


def test_negative_count_metadata_is_rejected(tmp_path):
    path = tmp_path / "counts.json"
    path.write_text(json.dumps({"00000.tar": -1}))
    with pytest.raises(ValueError, match="Negative sample count"):
        get_shard_count(str(path))


def test_ignore_indices_cannot_exceed_the_shard_count(tmp_path):
    _fixture(tmp_path, (1,))
    with pytest.raises(ValueError, match="Negative sample count"):
        get_shard_count(str(tmp_path / "counts.json"), {"00000.tar": {"a", "b"}})


@pytest.mark.parametrize("claimed", [1, 4])
def test_read_shards_are_checked_against_count_metadata(tmp_path, claimed):
    _fixture(tmp_path, (3,))
    (tmp_path / "counts.json").write_text(json.dumps({"00000.tar": claimed}))
    with pytest.raises(ValueError, match="Shard count mismatch"):
        list(
            DeterministicWDS(
                [f"WDS:{tmp_path}"], repeat=False, pipeline_fn=_pipeline, shard_count_file=str(tmp_path / "counts.json")
            )
        )


@pytest.mark.parametrize("claimed", [1, 3])
def test_bad_counts_fail_before_delivering_the_next_shard(tmp_path, claimed):
    shards = _fixture(tmp_path, (2, 0, 3))
    (tmp_path / "counts.json").write_text(json.dumps({"00000.tar": claimed, "00001.tar": 0, "00002.tar": 3}))
    iterator = iter(
        DeterministicWDS(
            [f"WDS:{tmp_path}"], repeat=False, pipeline_fn=_pipeline, shard_count_file=str(tmp_path / "counts.json")
        )
    )
    delivered = []
    with pytest.raises(ValueError, match="Shard count mismatch for .*00000.tar"):
        for batch in iterator:
            delivered.extend(batch["key"])
    assert set(delivered) <= set(shards[0])


@pytest.mark.parametrize("counted", [False, True])
def test_s3_shards_keep_sample_urls_and_resume(tmp_path, monkeypatch, counted):
    """Samples keep their shard's ``s3://`` URI, which count and ignore lookups use."""
    epoch = _fixture(tmp_path, (7,), ignore=True)[0]
    url = "s3://bucket/data/00000.tar"
    monkeypatch.setattr(wds_utils, "get_shards", lambda *args, **kwargs: [url])
    # Forkserver workers (this suite's default) and spawn workers would register the real opener instead of this
    # patch, so the dataset is read in this process.
    monkeypatch.setitem(gopen_schemes, "s3", lambda uri, mode, bufsize: open(tmp_path / Path(uri).name, mode))
    dataset = DeterministicWDS(
        [f"WDS:{tmp_path}"],
        batch_size=3,
        sampler_start_idx=5,
        shard_count_file=str(tmp_path / "counts.json") if counted else None,
        ignore_index_paths=[str(tmp_path / "ignore-a.json"), str(tmp_path / "ignore-b.json")],
    )
    batches = list(itertools.islice(dataset, 3))
    assert [key for batch in batches for key in batch["__key__"]] == [epoch[i % len(epoch)] for i in range(5, 14)]
    assert all(batch["__url__"] == [url] * 3 for batch in batches)


@pytest.mark.parametrize("method", ["forkserver", "spawn"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_fresh_workers_open_s3_shards(tmp_path, monkeypatch, method, deterministic):
    """Workers that do not inherit the parent's modules get the ``s3`` opener by unpickling the dataset."""
    epoch = _fixture(tmp_path, (3,))[0]
    # ``python -m`` searches the working directory first, so ``_open_s3_shard`` runs this stub, not the S3 reader.
    # Without ``__init__.py``, the repository's ``scripts`` package would still win if it is also on ``sys.path``.
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "__init__.py").touch()
    (tmp_path / "scripts" / "resumable_s3_download.py").write_text(
        "import shutil, sys\n"
        "from pathlib import Path\n"
        "with open(Path(sys.argv[1]).name, 'rb') as shard:\n"
        "    shutil.copyfileobj(shard, sys.stdout.buffer)\n"
    )
    monkeypatch.chdir(tmp_path)
    url = "s3://bucket/data/00000.tar"
    monkeypatch.setattr(wds_utils, "get_shards", lambda *args, **kwargs: [url])
    if deterministic:
        dataset = DeterministicWDS([f"WDS:{tmp_path}"], batch_size=3, repeat=False)
    else:
        dataset = wds_utils.batch_wds_dataset(wds_utils.create_base_webdataset([f"WDS:{tmp_path}"], train=False), 3)
    loader = DataLoader(dataset, batch_size=None, num_workers=1, multiprocessing_context=method, timeout=120)
    batches = list(loader)
    assert [key for batch in batches for key in batch["__key__"]] == epoch
    assert all(batch["__url__"] == [url] * 3 for batch in batches)


@pytest.mark.parametrize("counted,repeat", itertools.product([False, True], repeat=2))
def test_negative_resume_offsets_cannot_wrap_to_another_epoch(tmp_path, counted, repeat):
    _fixture(tmp_path, (3,))
    with pytest.raises(ValueError, match="sampler_start_idx must be nonnegative"):
        DeterministicWDS(
            [f"WDS:{tmp_path}"],
            sampler_start_idx=-1,
            repeat=repeat,
            shard_count_file=str(tmp_path / "counts.json") if counted else None,
        )
