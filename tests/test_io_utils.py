# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for checkpoint discovery/loading."""

from __future__ import annotations

import argparse
import io
import json

import pytest
import torch
from safetensors.torch import save_file

import fastgen.utils.cache_utils as cache_utils
from fastgen.utils.cache_utils import cache
import fastgen.utils.io_utils as io_utils
from fastgen.utils.io_utils import (
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint_state_dict,
    load_training_checkpoint,
    parse_checkpoint_iteration,
    path_exists,
)


@pytest.fixture(autouse=True)
def _isolate_checkpoint_mirror(tmp_path_factory, monkeypatch):
    """Point the checkpoint mirror at a scratch dir for every test.

    ``load_checkpoint_state_dict`` resolves an s3:// path against the mirror, so without
    this a stray file under the real ``FASTGEN_OUTPUT/MODEL`` could satisfy a test that
    means to exercise the S3 path.
    """
    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", str(tmp_path_factory.mktemp("fastgen_output")))


@pytest.mark.parametrize("path", [None, ""])
def test_path_exists_unset_paths_do_not_access_storage(monkeypatch, path):
    def unexpected_access(*args, **kwargs):
        pytest.fail("An unset path must not access storage")

    monkeypatch.setattr(io_utils.os.path, "exists", unexpected_access)
    monkeypatch.setattr(io_utils.s3, "exists", unexpected_access)
    assert path_exists(path) is False


@pytest.mark.parametrize("name", ["checkpoint.pth", "shards/.metadata", "data.txt"])
@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_path_exists_local_paths_are_not_checkpoint_specific(tmp_path, monkeypatch, name, relative):
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(io_utils.s3, "exists", lambda *_: pytest.fail("Local paths must not query S3"))
    path = name if relative else str(target)
    assert path_exists(path) is False
    target.touch()
    assert path_exists(path) is True


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("key", ["checkpoint.pth", "shards/.metadata", "data.txt"])
def test_path_exists_routes_s3_objects_without_local_lookup(monkeypatch, key, exists):
    path = f"s3://bucket/{key}"
    calls = []
    monkeypatch.setattr(io_utils.s3, "exists", lambda value: calls.append(value) or exists)
    monkeypatch.setattr(io_utils.os.path, "exists", lambda *_: pytest.fail("S3 paths must not query local storage"))
    assert path_exists(path) is exists
    assert calls == [path]


def test_path_exists_preserves_s3_errors(monkeypatch):
    error = PermissionError("Access denied")

    def inaccessible(_path):
        raise error

    monkeypatch.setattr(io_utils.s3, "exists", inaccessible)
    with pytest.raises(PermissionError) as caught:
        path_exists("s3://bucket/checkpoint.pth")
    assert caught.value is error


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(3, 2)
        self.second = torch.nn.Linear(2, 1)


def assert_state_dict_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])


@pytest.mark.parametrize("checkpoint_format", ["pytorch", "safetensors"])
@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_load_checkpoint_state_dict_supports_single_file_checkpoints(tmp_path, checkpoint_format, device):
    expected = TinyModel().state_dict()
    checkpoint_path = tmp_path / f"model.{'pt' if checkpoint_format == 'pytorch' else 'safetensors'}"
    if checkpoint_format == "pytorch":
        torch.save(expected, checkpoint_path)
    else:
        save_file(expected, checkpoint_path)

    assert_state_dict_equal(load_checkpoint_state_dict(str(checkpoint_path), device=device), expected)


def test_load_checkpoint_state_dict_defaults_to_cpu_for_torch_checkpoints(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    torch.save({"w": torch.ones(2)}, path)
    seen = {}
    real_load = io_utils.torch.load

    def fake_load(*args, **kwargs):
        seen["map_location"] = kwargs.get("map_location")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(io_utils.torch, "load", fake_load)
    assert torch.equal(load_checkpoint_state_dict(str(path))["w"], torch.ones(2))
    assert seen["map_location"] == "cpu"


def test_load_checkpoint_state_dict_forwards_device_to_torch_load(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    torch.save({"w": torch.ones(2)}, path)
    seen = {}
    real_load = io_utils.torch.load

    def fake_load(*args, **kwargs):
        seen["map_location"] = kwargs.get("map_location")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(io_utils.torch, "load", fake_load)
    loaded = load_checkpoint_state_dict(str(path), device=None)  # type: ignore[arg-type]
    assert torch.equal(loaded["w"], torch.ones(2))
    assert seen["map_location"] is None


def test_load_checkpoint_state_dict_supports_s3(monkeypatch):
    expected = TinyModel().state_dict()
    checkpoint = io.BytesIO()
    torch.save(expected, checkpoint)
    checkpoint.seek(0)

    def fake_load(path):
        assert path == "s3://bucket/model.pt"
        return checkpoint

    monkeypatch.setattr("fastgen.utils.s3.load", fake_load)
    actual = load_checkpoint_state_dict("s3://bucket/model.pt")
    assert_state_dict_equal(actual, expected)


def test_load_checkpoint_state_dict_supports_sharded_safetensors_index(tmp_path):
    expected = TinyModel().state_dict()
    first_keys = [key for key in expected if key.startswith("first.")]
    second_keys = [key for key in expected if key.startswith("second.")]
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    save_file({key: expected[key] for key in first_keys}, tmp_path / shard_names[0])
    save_file({key: expected[key] for key in second_keys}, tmp_path / shard_names[1])

    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(tensor.nbytes for tensor in expected.values())},
                "weight_map": {
                    **{key: shard_names[0] for key in first_keys},
                    **{key: shard_names[1] for key in second_keys},
                },
            }
        ),
        encoding="utf-8",
    )

    assert_state_dict_equal(load_checkpoint_state_dict(str(index_path)), expected)


def test_load_checkpoint_state_dict_reports_missing_safetensors_shard(tmp_path):
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {"first.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        load_checkpoint_state_dict(str(index_path))


def _stub_s3(monkeypatch, objects):
    """Serve ``objects`` (S3 URI -> bytes) from a fake bucket, recording every fetch."""
    fetched = []

    def fetch(path):
        fetched.append(path)
        if path not in objects:
            raise AssertionError(f"unexpected fetch of absent object {path}")
        return objects[path]

    def fake_download(path, output):
        output.write(fetch(path))
        return 0

    monkeypatch.setattr("fastgen.utils.s3.load", lambda path: io.BytesIO(fetch(path)))
    monkeypatch.setattr("fastgen.utils.s3.resumable_download", fake_download)
    monkeypatch.setattr("fastgen.utils.s3.exists", lambda path: path in objects)
    return fetched


def _torch_bytes(state_dict):
    payload = io.BytesIO()
    torch.save(state_dict, payload)
    return payload.getvalue()


@pytest.mark.parametrize("checkpoint_format", ["pytorch", "safetensors"])
def test_load_checkpoint_state_dict_supports_single_file_http(tmp_path, monkeypatch, checkpoint_format):
    """An uncached HTTP checkpoint loads from memory without writing a mirror."""
    expected = TinyModel().state_dict()
    if checkpoint_format == "pytorch":
        suffix = ".pt"
        payload = _torch_bytes(expected)
    else:
        suffix = ".safetensors"
        source = tmp_path / f"source{suffix}"
        save_file(expected, source)
        payload = source.read_bytes()

    url = f"https://example.com/model{suffix}?revision=immutable"
    fetched = []

    def fake_download(path, output):
        fetched.append(path)
        output.write(payload)

    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setattr(cache_utils, "_download_http", fake_download)

    assert_state_dict_equal(load_checkpoint_state_dict(url), expected)
    assert fetched == [url]
    assert not (tmp_path / "output").exists()


def test_load_checkpoint_state_dict_rejects_sharded_safetensors_over_http(monkeypatch):
    url = "https://example.com/model.safetensors.index.json"
    monkeypatch.setattr(
        cache_utils,
        "_download_http",
        lambda *_args, **_kwargs: pytest.fail("an unsupported HTTP index must not be downloaded"),
    )

    with pytest.raises(ValueError, match=r"not supported over HTTP\(S\)"):
        load_checkpoint_state_dict(url)


def _sharded_safetensors_objects(tmp_path, expected, prefix):
    """Build a two-shard SafeTensors checkpoint as an ``{S3 URI: bytes}`` mapping."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    first_keys = [key for key in expected if key.startswith("first.")]
    second_keys = [key for key in expected if key.startswith("second.")]
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

    objects = {}
    for shard_name, keys in zip(shard_names, (first_keys, second_keys)):
        shard_path = tmp_path / shard_name
        save_file({key: expected[key] for key in keys}, shard_path)
        objects[f"{prefix}/{shard_name}"] = shard_path.read_bytes()

    index = {
        "weight_map": {
            **{key: shard_names[0] for key in first_keys},
            **{key: shard_names[1] for key in second_keys},
        }
    }
    objects[f"{prefix}/model.safetensors.index.json"] = json.dumps(index).encode("utf-8")
    return objects, shard_names


def test_load_checkpoint_state_dict_supports_single_file_safetensors_on_s3(monkeypatch, tmp_path):
    """A single-file SafeTensors object loads straight from S3, with no local mirror."""
    expected = TinyModel().state_dict()
    blob_path = tmp_path / "model.safetensors"
    save_file(expected, blob_path)
    _stub_s3(monkeypatch, {"s3://bucket/model.safetensors": blob_path.read_bytes()})

    assert_state_dict_equal(load_checkpoint_state_dict("s3://bucket/model.safetensors"), expected)


def test_load_checkpoint_state_dict_supports_sharded_safetensors_on_s3(monkeypatch, tmp_path):
    """A shard index on S3 resolves its shards against the index's own S3 prefix."""
    expected = TinyModel().state_dict()
    objects, _ = _sharded_safetensors_objects(tmp_path, expected, "s3://bucket/run")
    fetched = _stub_s3(monkeypatch, objects)

    actual = load_checkpoint_state_dict("s3://bucket/run/model.safetensors.index.json")

    assert_state_dict_equal(actual, expected)
    assert fetched[0] == "s3://bucket/run/model.safetensors.index.json"
    assert sorted(fetched[1:]) == sorted(key for key in objects if key.endswith(".safetensors"))


def test_load_checkpoint_state_dict_reports_missing_s3_shard(monkeypatch, tmp_path):
    """A shard the index names but S3 does not have fails with the same clear error."""
    index = {"weight_map": {"first.weight": "missing.safetensors"}}
    _stub_s3(monkeypatch, {"s3://bucket/run/model.safetensors.index.json": json.dumps(index).encode("utf-8")})

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        load_checkpoint_state_dict("s3://bucket/run/model.safetensors.index.json")


def _stub_s3_download(monkeypatch, state_dict):
    """Serve ``state_dict`` as a single PyTorch object from a fake S3."""
    return _stub_s3(monkeypatch, {"s3://bucket/model.pt": _torch_bytes(state_dict)})


def test_load_checkpoint_state_dict_never_writes_to_disk(tmp_path, monkeypatch):
    """Loading is read-only: mirroring is ``cache``'s job alone.

    Each load therefore re-reads from S3, which is exactly why a distributed caller
    primes the mirror once up front rather than relying on a side effect here.
    """
    expected = TinyModel().state_dict()
    fetched = _stub_s3_download(monkeypatch, expected)
    monkeypatch.chdir(tmp_path)

    assert_state_dict_equal(load_checkpoint_state_dict("s3://bucket/model.pt"), expected)
    assert_state_dict_equal(load_checkpoint_state_dict("s3://bucket/model.pt"), expected)

    assert fetched == ["s3://bucket/model.pt"] * 2
    assert list(tmp_path.iterdir()) == [], "loading must not write anything"


def _stub_s3_client(monkeypatch, **methods):
    """Install a stub boto3 S3 client exposing ``methods``, e.g. ``head_object=fn``.

    Stubbing at the client rather than at ``load``/``save`` keeps the bucket/key
    derivation and the boto3 call shape under test; only the endpoint is faked.
    """

    client = type("_StubS3Client", (), {name: staticmethod(fn) for name, fn in methods.items()})()
    monkeypatch.setattr("fastgen.utils.s3.boto3.client", lambda *args, **kwargs: client)


@pytest.mark.parametrize(
    "path, iteration",
    [
        ("/run/checkpoints/0000500.pth", 500),
        ("43.pth", 43),
        ("s3://bucket/run/0000002.pth", 2),
        ("latest_ar.pth", None),
        ("notes2.pth", None),
        ("-1.pth", None),
        ("1_000.pth", None),
        ("0000500", None),
        ("0000500.net_model", None),
        ("0000500.pth.11111111111111111111111111111111.tmp", None),
    ],
)
def test_parse_checkpoint_iteration_reads_only_interval_checkpoint_names(path, iteration):
    assert parse_checkpoint_iteration(path) == iteration


def test_list_checkpoints_orders_interval_checkpoints_on_both_backends(tmp_path, monkeypatch):
    names = ("100.pth", "latest_ar.pth", "0000002.pth", "metadata.json", "notes2.pth")
    for name in names:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "0000050.pth").mkdir()
    monkeypatch.setattr(io_utils.s3, "iter_objects", lambda path: iter((f"run/{name}", 0.0) for name in names))

    assert list_checkpoints(str(tmp_path)) == [
        (2, str(tmp_path / "0000002.pth")),
        (100, str(tmp_path / "100.pth")),
    ]
    assert list_checkpoints("s3://bucket/run/") == [
        (2, "s3://bucket/run/0000002.pth"),
        (100, "s3://bucket/run/100.pth"),
    ]


@pytest.mark.parametrize("latest_name", ["43.pth", "0000043.pth", "000000043.pth", "latest_ar.pth"])
def test_latest_checkpoint_returns_an_explicit_local_marker(tmp_path, latest_name):
    torch.save({"iteration": 42}, tmp_path / "0000042.pth")
    latest = tmp_path / latest_name
    torch.save({"iteration": 43}, latest)

    assert latest_checkpoint(str(tmp_path)) == str(latest)


def test_latest_checkpoint_ignores_directories_with_checkpoint_names(tmp_path):
    latest = tmp_path / "0000042.pth"
    torch.save({"iteration": 42}, latest)
    (tmp_path / "0000100.pth").mkdir()

    assert latest_checkpoint(str(tmp_path)) == str(latest)


def test_latest_checkpoint_keeps_the_s3_object_name(monkeypatch):
    monkeypatch.setattr(io_utils.s3, "iter_objects", lambda path: iter((("run/checkpoints/42.pth", 1.0),)))

    assert latest_checkpoint("s3://bucket/run/checkpoints") == "s3://bucket/run/checkpoints/42.pth"


def test_latest_checkpoint_warns_when_it_skips_an_unreadable_latest_ar(tmp_path, monkeypatch):
    torch.save({"iteration": 42}, tmp_path / "0000042.pth")
    (tmp_path / "latest_ar.pth").write_bytes(b"not a checkpoint")
    warnings = []
    monkeypatch.setattr(io_utils.logger, "warning", warnings.append)

    assert latest_checkpoint(str(tmp_path)) == str(tmp_path / "0000042.pth")
    assert len(warnings) == 1 and str(tmp_path / "latest_ar.pth") in warnings[0]


@pytest.mark.parametrize("source, expect_mmap", [("zipfile", True), ("legacy", False), ("s3", False)])
def test_load_training_checkpoint_maps_only_local_zipfile_checkpoints(tmp_path, monkeypatch, source, expect_mmap):
    payload = io.BytesIO()
    state = {"weight": torch.arange(4.0), "iteration": 3}
    torch.save(state, payload, _use_new_zipfile_serialization=source != "legacy")
    path = tmp_path / "0000003.pth"
    path.write_bytes(payload.getvalue())
    if source == "s3":
        monkeypatch.setattr("fastgen.utils.s3.exists", lambda uri: True)
        monkeypatch.setattr("fastgen.utils.s3.load", lambda uri: io.BytesIO(payload.getvalue()))
        path = "s3://bucket/0000003.pth"
    mmap_flags = []
    real_load = torch.load

    def recording_load(*args, **kwargs):
        mmap_flags.append(kwargs["mmap"])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    loaded = load_training_checkpoint(str(path), device="cpu", mmap=True)

    assert mmap_flags == [expect_mmap]
    assert loaded["iteration"] == 3
    assert torch.equal(loaded["weight"], state["weight"])


def test_load_training_checkpoint_reads_non_tensor_state_from_both_backends(tmp_path, monkeypatch):
    """Both branches of ``load_training_checkpoint`` must pass ``weights_only=False``.

    ``torch.load`` resolves the default to True, so a branch that omitted it would
    reject callback/optimizer payloads that the other branch loads fine. ``Namespace``
    stands in for any non-tensor object a callback may persist.
    """
    state = {"callbacks": {"marker": argparse.Namespace(step=7)}, "iteration": 3}
    payload = io.BytesIO()
    torch.save(state, payload)

    local_path = tmp_path / "0000003.pth"
    local_path.write_bytes(payload.getvalue())
    local_state = load_training_checkpoint(str(local_path), device="cpu")

    monkeypatch.setattr("fastgen.utils.s3.exists", lambda path: True)
    monkeypatch.setattr("fastgen.utils.s3.load", lambda path: io.BytesIO(payload.getvalue()))
    s3_state = load_training_checkpoint("s3://bucket/0000003.pth", device="cpu")

    assert local_state["callbacks"]["marker"].step == 7
    assert s3_state["callbacks"]["marker"].step == 7
    assert local_state["iteration"] == s3_state["iteration"] == 3


def test_latest_checkpoint_paginates_s3_listing(monkeypatch):
    """A checkpoint manifest after S3's 1,000-key page boundary must be visible."""
    prefix = "run/checkpoints"
    first_page = {
        "Contents": [{"Key": f"{prefix}/metadata-{index:04d}.json"} for index in range(996)]
        + [{"Key": f"{prefix}/{iteration:07d}.pth"} for iteration in (25, 50, 75, 100)]
    }
    second_page = {"Contents": [{"Key": f"{prefix}/0000425.pth"}]}
    seen = {}

    class _Paginator:
        def paginate(self, **kwargs):
            seen.update(kwargs)
            return iter((first_page, second_page))

    def get_paginator(name):
        assert name == "list_objects_v2"
        return _Paginator()

    _stub_s3_client(monkeypatch, get_paginator=get_paginator)

    checkpoint_root = "s3://bucket/run/checkpoints"
    assert len(first_page["Contents"]) == 1000
    assert latest_checkpoint(checkpoint_root) == f"{checkpoint_root}/0000425.pth"
    assert seen == {"Bucket": "bucket", "Prefix": f"{prefix}/", "Delimiter": "/"}


def test_latest_checkpoint_accepts_an_s3_bucket_root(monkeypatch):
    """Checkpoint discovery accepts a bucket root even though object operations do not."""
    seen = {}

    class _Paginator:
        def paginate(self, **kwargs):
            seen.update(kwargs)
            return iter(({"Contents": [{"Key": "0000042.pth"}]},))

    _stub_s3_client(monkeypatch, get_paginator=lambda name: _Paginator())

    assert latest_checkpoint("s3://bucket") == "s3://bucket/0000042.pth"
    assert seen == {"Bucket": "bucket", "Prefix": "", "Delimiter": "/"}


def test_latest_checkpoint_only_considers_direct_children(monkeypatch):
    """Nested keys and sibling prefixes are not candidates, mirroring the local ``os.listdir`` branch."""
    prefix = "run/checkpoints"
    page = {
        "Contents": [{"Key": f"{prefix}/0000100.pth"}],
        "CommonPrefixes": [{"Prefix": f"{prefix}/archive/"}],
    }
    seen = {}

    class _Paginator:
        def paginate(self, **kwargs):
            seen.update(kwargs)
            return iter((page,))

    _stub_s3_client(monkeypatch, get_paginator=lambda name: _Paginator())

    assert latest_checkpoint(f"s3://bucket/{prefix}") == f"s3://bucket/{prefix}/0000100.pth"
    assert seen == {"Bucket": "bucket", "Prefix": f"{prefix}/", "Delimiter": "/"}


def test_load_checkpoint_state_dict_reads_the_mirror_when_present(monkeypatch, tmp_path):
    """An s3:// path resolves to its mirror inside the loader, costing no further traffic.

    Also covers the default cache dir: neither call passes ``cache_dir``, so both go
    through ``checkpoint_cache_dir()``.
    """
    expected = TinyModel().state_dict()
    fetched = _stub_s3(monkeypatch, {"s3://bucket/model.pt": _torch_bytes(expected)})
    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", str(tmp_path))

    cache("s3://bucket/model.pt")
    assert fetched == ["s3://bucket/model.pt"]
    assert (tmp_path / "MODEL" / "bucket" / "model.pt").is_file()

    assert_state_dict_equal(load_checkpoint_state_dict("s3://bucket/model.pt"), expected)
    assert fetched == ["s3://bucket/model.pt"], "the load must read the mirror, not S3"
