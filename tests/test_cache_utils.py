# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for atomic publication and remote-to-local caching."""

from __future__ import annotations

import email.message
import io
import json
import os
import urllib.error
from types import SimpleNamespace

import botocore.exceptions
import pytest
import torch
from safetensors.torch import save_file

import fastgen.utils.cache_utils as cache_utils
from fastgen.utils.cache_utils import (
    atomic_write,
    cache,
    checkpoint_cache_dir,
    file_extension,
    mirror_path,
    remove_atomic_write_orphans,
    resolve_cached_path,
)
from fastgen.utils.io_utils import load_checkpoint_state_dict


@pytest.fixture(autouse=True)
def _isolate_checkpoint_mirror(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", str(tmp_path_factory.mktemp("fastgen_output")))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(3, 2)
        self.second = torch.nn.Linear(2, 1)


def assert_state_dict_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])


class _FakeHttpResponse(io.BytesIO):
    """A ``urlopen`` response stand-in: a byte body with HTTP headers."""

    def __init__(self, body: bytes, content_type: str = "application/octet-stream", content_length=None):
        super().__init__(body)
        self.headers = email.message.Message()
        self.headers["Content-Type"] = content_type
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)


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


def _stub_s3_download(monkeypatch, state_dict):
    return _stub_s3(monkeypatch, {"s3://bucket/model.pt": _torch_bytes(state_dict)})


@pytest.mark.parametrize(
    ("path", "is_http", "is_file", "is_index"),
    [
        ("https://example.com/model.safetensors?download=1", True, True, False),
        ("https://example.com/model.safetensors.index.json?download=1", True, False, True),
        ("s3://bucket/model.safetensors", False, True, False),
        ("/local/model.pt", False, False, False),
    ],
)
def test_path_classifiers(path, is_http, is_file, is_index):
    assert cache_utils.is_http_url(path) is is_http
    assert cache_utils.is_safetensors_file(path) is is_file
    assert cache_utils.is_safetensors_index(path) is is_index


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("https://example.com/datasets/images.ZIP?download=1#archive", ".zip"),
        ("s3://bucket/images/sample.NPY", ".npy"),
        ("relative/archive.tar.gz", ".gz"),
        ("relative/no_extension", ""),
    ],
)
def test_file_extension(path, expected):
    assert file_extension(path) == expected


def test_file_extension_accepts_pathlike(tmp_path):
    assert file_extension(tmp_path / "sample.PNG") == ".png"


@pytest.mark.parametrize(
    "classifier",
    [
        cache_utils.is_http_url,
        cache_utils.is_safetensors_file,
        cache_utils.is_safetensors_index,
    ],
)
def test_path_classifiers_require_a_string(classifier):
    with pytest.raises(AssertionError, match="path must be a string"):
        classifier(None)


@pytest.mark.parametrize(
    ("url", "suffix"),
    [
        ("https://example.com/model.pt?revision=1", ".pt"),
        ("https://example.com/model.pth", ".pth"),
        ("https://example.com/model.ckpt", ".ckpt"),
        ("https://example.com/model.bin", ".bin"),
        ("https://example.com/model.safetensors", ".safetensors"),
        ("https://example.com/model.safetensors.index.json", ".safetensors.index.json"),
        ("https://example.com/model.zip", ""),
    ],
)
def test_http_mirror_path_preserves_recognized_suffixes(url, suffix, tmp_path):
    digest = cache_utils.hashlib.sha256(url.encode("utf-8")).hexdigest()

    assert mirror_path(url, str(tmp_path)) == str(tmp_path / "url" / f"{digest}{suffix}")


def test_atomic_write_replaces_only_after_complete_write(tmp_path, monkeypatch):
    destination = tmp_path / "state.bin"
    destination.write_bytes(b"old")
    original_replace = os.replace
    replacements = []
    fsyncs = []

    def inspect_then_replace(source, target):
        assert target == str(destination)
        assert destination.read_bytes() == b"old"
        with open(source, "rb") as source_file:
            assert source_file.read() == b"new"
        replacements.append((source, target))
        original_replace(source, target)

    monkeypatch.setattr(cache_utils.os, "replace", inspect_then_replace)
    monkeypatch.setattr(cache_utils.os, "fsync", fsyncs.append)

    atomic_write(destination, lambda output: output.write(b"new"), fsync=True)

    assert len(replacements) == 1
    assert len(fsyncs) == 1
    assert destination.read_bytes() == b"new"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("sole_writer", [False, True], ids=["shared", "sole-writer"])
def test_atomic_write_failure_preserves_destination_and_cleans_up(tmp_path, sole_writer):
    destination = tmp_path / "state.bin"
    destination.write_bytes(b"old")

    def fail_after_partial_write(output):
        output.write(b"partial")
        raise OSError("injected write failure")

    with pytest.raises(OSError, match="injected write failure"):
        atomic_write(destination, fail_after_partial_write, sole_writer=sole_writer)

    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("leftover_is_symlink", [False, True], ids=["file", "symlink"])
def test_atomic_write_sole_writer_replaces_an_interrupted_write(tmp_path, leftover_is_symlink):
    destination = tmp_path / "state.bin"
    destination.write_bytes(b"old")
    unrelated = tmp_path / "unrelated.bin"
    unrelated.write_bytes(b"unrelated")
    leftover = tmp_path / "state.bin.tmp"
    if leftover_is_symlink:
        leftover.symlink_to(unrelated)
    else:
        leftover.write_bytes(b"interrupted")

    atomic_write(destination, lambda output: output.write(b"new"), sole_writer=True)

    assert destination.read_bytes() == b"new"
    assert unrelated.read_bytes() == b"unrelated"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_does_not_remove_a_colliding_temporary_file(tmp_path, monkeypatch):
    destination = tmp_path / "state.bin"
    temporary = tmp_path / "state.bin.fixed.tmp"
    temporary.write_bytes(b"someone else's file")
    monkeypatch.setattr(cache_utils.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed"))

    with pytest.raises(FileExistsError):
        atomic_write(destination, lambda output: output.write(b"new"))

    assert temporary.read_bytes() == b"someone else's file"
    assert not destination.exists()


def test_remove_atomic_write_orphans_only_removes_matching_atomic_temporaries(tmp_path):
    directory = tmp_path / "cache[0]"
    directory.mkdir()
    removable = [
        directory / f"state.bin.{'1' * 32}.tmp",
        directory / f"weights.bin.{'2' * 32}.tmp",
    ]
    retained = [
        directory / f"state.pt.{'3' * 32}.tmp",
        directory / "state.bin.not-a-uuid.tmp",
    ]
    for path in removable + retained:
        path.write_bytes(b"temporary")

    removed = remove_atomic_write_orphans(directory, "*.bin")

    assert removed == [str(path) for path in removable]
    assert all(not path.exists() for path in removable)
    assert all(path.is_file() for path in retained)


def test_cache_raises_when_file_mirror_cannot_be_written(tmp_path, monkeypatch):
    """A strict caller reports an unwritable mirror and removes its temporary file."""
    expected = TinyModel().state_dict()
    _stub_s3_download(monkeypatch, expected)
    cache_dir = tmp_path / "cache"

    def exploding_replace(*args, **kwargs):
        raise OSError("disk full")

    # Fail at the rename, so the temporary file genuinely exists when cleanup runs.
    monkeypatch.setattr(cache_utils.os, "replace", exploding_replace)

    with pytest.raises(OSError, match="disk full"):
        cache("s3://bucket/model.pt", cache_dir=str(cache_dir))
    assert list(cache_dir.rglob("*.tmp")) == [], "the temporary file must not survive a failed write"

    # Direct S3 loading remains independently available when a caller explicitly wants it.
    assert_state_dict_equal(load_checkpoint_state_dict("s3://bucket/model.pt"), expected)


def _raise_mirror_failure(*args, **kwargs):
    raise OSError("mirror unavailable")


def test_cache_without_strict_degrades_to_the_original_uri(tmp_path, monkeypatch):
    """A non-strict caller keeps the S3 URI it can still read, and says so once."""
    expected = TinyModel().state_dict()
    _stub_s3_download(monkeypatch, expected)
    monkeypatch.setattr(cache_utils.os, "replace", _raise_mirror_failure)
    warnings = []
    monkeypatch.setattr(cache_utils.logger, "warning", warnings.append)
    cache_dir = tmp_path / "cache"

    assert cache("s3://bucket/model.pt", cache_dir=str(cache_dir), strict=False) == "s3://bucket/model.pt"
    assert len(warnings) == 1
    assert "Could not mirror s3://bucket/model.pt" in warnings[0]

    # The returned URI is exactly what the streaming loader accepts.
    assert_state_dict_equal(load_checkpoint_state_dict("s3://bucket/model.pt"), expected)


def test_cache_without_strict_downgrades_an_unsupported_http_index(tmp_path, monkeypatch):
    """The HTTP shard-index rejection follows ``strict`` like every other mirror failure."""
    warnings = []
    monkeypatch.setattr(cache_utils.logger, "warning", warnings.append)
    url = "https://example.com/model.safetensors.index.json"

    with pytest.raises(ValueError, match="only supported for S3"):
        cache(url, cache_dir=str(tmp_path))
    assert cache(url, cache_dir=str(tmp_path), strict=False) == url
    assert len(warnings) == 1 and "only supported for S3" in warnings[0]


def test_index_shard_names_treats_a_malformed_index_as_shardless(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text("{not json", encoding="utf-8")
    assert cache_utils._index_shard_names(str(index)) == []
    index.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
    assert cache_utils._index_shard_names(str(index)) == []


def test_cache_without_strict_still_returns_a_usable_mirror(tmp_path, monkeypatch):
    """Relaxing strictness must not change the result when mirroring succeeds."""
    expected = TinyModel().state_dict()
    _stub_s3_download(monkeypatch, expected)
    cache_dir = tmp_path / "cache"

    cached_path = cache("s3://bucket/model.pt", cache_dir=str(cache_dir), strict=False)

    assert cached_path == str(cache_dir / "bucket" / "model.pt")
    assert_state_dict_equal(load_checkpoint_state_dict(cached_path), expected)


def test_cache_rejects_a_remote_cache_dir(monkeypatch, tmp_path):
    """A remote ``cache_dir`` raises instead of creating a literal ``s3:`` directory."""
    _stub_s3_download(monkeypatch, TinyModel().state_dict())
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="cache_dir must be a local directory"):
        cache("s3://bucket/model.pt", cache_dir="s3://checkpoints")
    assert not list(tmp_path.glob("s3:*")), "must not create a literal 's3:' directory"


def test_cache_mirrors_s3_file_without_deserializing(monkeypatch, tmp_path):
    """Priming streams into the mirror; the later load is then a pure cache hit."""
    expected = TinyModel().state_dict()
    fetched = _stub_s3(monkeypatch, {"s3://bucket/model.pt": _torch_bytes(expected)})
    monkeypatch.setattr(
        cache_utils.s3,
        "load",
        lambda _path: pytest.fail("caching must not stage S3 data in memory"),
    )
    cache_dir = tmp_path / "cache"

    cached_path = cache("s3://bucket/model.pt", cache_dir=str(cache_dir))

    assert cached_path == str(cache_dir / "bucket" / "model.pt")
    assert (cache_dir / "bucket" / "model.pt").is_file()
    assert fetched == ["s3://bucket/model.pt"]

    # The load a rank would do next reads the mirror, so it never touches S3 again.
    assert_state_dict_equal(load_checkpoint_state_dict(cached_path), expected)
    assert fetched == ["s3://bucket/model.pt"]


def test_cache_does_not_publish_a_failed_s3_download(monkeypatch, tmp_path):
    def fail_after_partial_write(_path, output):
        output.write(b"partial")
        return 1

    monkeypatch.setattr(cache_utils.s3, "resumable_download", fail_after_partial_write)
    cache_dir = tmp_path / "cache"

    with pytest.raises(IOError, match="Failed to download"):
        cache("s3://bucket/model.pt", cache_dir=str(cache_dir))

    assert resolve_cached_path("s3://bucket/model.pt", cache_dir=str(cache_dir)) == "s3://bucket/model.pt"
    assert list(cache_dir.rglob("*.tmp")) == []


def _s3_client_error(status: int) -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError(
        {"Error": {"Code": str(status)}, "ResponseMetadata": {"HTTPStatusCode": status}}, "HeadObject"
    )


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(botocore.exceptions.NoCredentialsError(), id="no-credentials"),
        pytest.param(_s3_client_error(403), id="403"),
        pytest.param(_s3_client_error(404), id="404"),
    ],
)
def test_cache_without_strict_raises_failures_the_original_uri_would_hit_too(monkeypatch, tmp_path, error):
    """Reading the original URI fails the same way, so a non-strict caller has nothing to fall back to."""

    def fail(_path, output):
        raise error

    monkeypatch.setattr(cache_utils.s3, "resumable_download", fail)
    warnings = []
    monkeypatch.setattr(cache_utils.logger, "warning", warnings.append)
    cache_dir = tmp_path / "cache"

    with pytest.raises(type(error)) as raised:
        cache("s3://bucket/data.zip", cache_dir=str(cache_dir), strict=False)

    assert raised.value is error
    assert warnings == []
    assert list(cache_dir.rglob("*.tmp")) == []


def test_cache_without_strict_still_downgrades_an_s3_server_error(monkeypatch, tmp_path):
    def fail(_path, output):
        raise _s3_client_error(500)

    monkeypatch.setattr(cache_utils.s3, "resumable_download", fail)
    warnings = []
    monkeypatch.setattr(cache_utils.logger, "warning", warnings.append)

    assert cache("s3://bucket/data.zip", cache_dir=str(tmp_path / "cache"), strict=False) == "s3://bucket/data.zip"
    assert len(warnings) == 1


def test_cache_mirrors_http_file_once(monkeypatch, tmp_path):
    """An HTTP cache hit resolves locally without contacting the server again."""
    url = "https://example.com/inception.pt?version=1"
    downloads = []

    def fake_urlopen(request, timeout):
        downloads.append((request.full_url, timeout))
        return _FakeHttpResponse(b"inception", content_length=len(b"inception"))

    monkeypatch.setattr(cache_utils, "urlopen", fake_urlopen)
    cache_dir = tmp_path / "cache"

    assert resolve_cached_path(url, cache_dir=str(cache_dir)) == url
    cached_path = cache(url, cache_dir=str(cache_dir))

    assert cached_path == mirror_path(url, str(cache_dir))
    assert cached_path.endswith(".pt")
    with open(cached_path, "rb") as cached_file:
        assert cached_file.read() == b"inception"
    assert resolve_cached_path(url, cache_dir=str(cache_dir)) == cached_path
    assert cache(url, cache_dir=str(cache_dir)) == cached_path
    assert downloads == [(url, 60)]


def test_cached_http_safetensors_retains_format_for_state_dict_loading(monkeypatch, tmp_path):
    expected = TinyModel().state_dict()
    source = tmp_path / "source.safetensors"
    save_file(expected, source)
    url = "https://example.com/model.safetensors?revision=1"
    downloads = []

    def fake_urlopen(request, timeout):
        downloads.append((request.full_url, timeout))
        return _FakeHttpResponse(source.read_bytes())

    monkeypatch.setattr(cache_utils, "urlopen", fake_urlopen)

    cached_path = cache(url)

    assert cached_path.endswith(".safetensors")
    assert_state_dict_equal(load_checkpoint_state_dict(url), expected)
    assert downloads == [(url, 60)]


def test_cache_does_not_publish_an_empty_http_download(monkeypatch, tmp_path):
    url = "https://example.com/empty.pt"
    monkeypatch.setattr(cache_utils, "urlopen", lambda request, timeout: _FakeHttpResponse(b""))
    monkeypatch.setattr(cache_utils.logger, "warning", lambda message: None)
    cache_dir = tmp_path / "cache"

    with pytest.raises(IOError, match="No data received"):
        cache(url, cache_dir=str(cache_dir))

    assert resolve_cached_path(url, cache_dir=str(cache_dir)) == url
    assert list(cache_dir.rglob("*.tmp")) == []


def test_cache_does_not_publish_a_truncated_http_download(monkeypatch, tmp_path):
    """A body shorter than ``Content-Length`` is a dropped connection, not a complete file."""
    url = "https://example.com/truncated.pt"
    monkeypatch.setattr(
        cache_utils, "urlopen", lambda request, timeout: _FakeHttpResponse(b"0123456789", content_length=100)
    )
    warnings = []
    monkeypatch.setattr(cache_utils.logger, "warning", warnings.append)
    cache_dir = tmp_path / "cache"

    with pytest.raises(IOError, match="Truncated download"):
        cache(url, cache_dir=str(cache_dir))

    assert len(warnings) == cache_utils._HTTP_DOWNLOAD_ATTEMPTS - 1
    assert resolve_cached_path(url, cache_dir=str(cache_dir)) == url
    assert list(cache_dir.rglob("*.tmp")) == []


def test_cache_rejects_an_html_interstitial(monkeypatch, tmp_path):
    url = "https://example.com/gated.pt"
    monkeypatch.setattr(
        cache_utils, "urlopen", lambda request, timeout: _FakeHttpResponse(b"<html>sign in</html>", "text/html")
    )
    monkeypatch.setattr(cache_utils.logger, "warning", lambda message: None)

    with pytest.raises(IOError, match="HTML page"):
        cache(url, cache_dir=str(tmp_path / "cache"))


def test_cache_retries_a_transient_http_failure(monkeypatch, tmp_path):
    url = "https://example.com/flaky.pt"
    attempts = []

    def flaky_urlopen(request, timeout):
        attempts.append(request.full_url)
        if len(attempts) == 1:
            raise urllib.error.URLError("connection reset")
        return _FakeHttpResponse(b"payload", content_length=len(b"payload"))

    monkeypatch.setattr(cache_utils, "urlopen", flaky_urlopen)
    monkeypatch.setattr(cache_utils.logger, "warning", lambda message: None)
    cache_dir = tmp_path / "cache"

    cached_path = cache(url, cache_dir=str(cache_dir))

    assert attempts == [url, url]
    with open(cached_path, "rb") as cached_file:
        assert cached_file.read() == b"payload"


def test_cache_does_not_retry_a_client_error(monkeypatch, tmp_path):
    url = "https://example.com/missing.pt"
    attempts = []

    def forbidden_urlopen(request, timeout):
        attempts.append(request.full_url)
        raise urllib.error.HTTPError(url, 403, "Forbidden", email.message.Message(), None)

    monkeypatch.setattr(cache_utils, "urlopen", forbidden_urlopen)

    with pytest.raises(urllib.error.HTTPError):
        cache(url, cache_dir=str(tmp_path / "cache"))

    assert attempts == [url]


def test_cache_rejects_an_http_safetensors_index_before_downloading(monkeypatch, tmp_path):
    url = "https://example.com/model.safetensors.index.json?version=1"

    def unexpected_download(*args, **kwargs):
        raise AssertionError("unsupported HTTP index must not be downloaded")

    monkeypatch.setattr(cache_utils, "urlopen", unexpected_download)

    with pytest.raises(ValueError, match="only supported for S3 URIs"):
        cache(url, cache_dir=str(tmp_path / "cache"))


def test_cache_file_mirrors_every_shard(monkeypatch, tmp_path):
    """A shard index primes the index *and* its shards, so no rank refetches either."""
    expected = TinyModel().state_dict()
    objects, shard_names = _sharded_safetensors_objects(tmp_path / "src", expected, "s3://bucket/run")
    fetched = _stub_s3(monkeypatch, objects)
    cache_dir = tmp_path / "cache"

    cache("s3://bucket/run/model.safetensors.index.json", cache_dir=str(cache_dir))

    assert (cache_dir / "bucket" / "run" / "model.safetensors.index.json").is_file()
    for shard_name in shard_names:
        assert (cache_dir / "bucket" / "run" / shard_name).is_file()
    assert len(fetched) == 3


def test_cache_file_reresolve_makes_no_s3_calls(monkeypatch, tmp_path):
    """Every rank re-resolves an already-mirrored checkpoint without S3 traffic."""
    expected = TinyModel().state_dict()
    objects, _ = _sharded_safetensors_objects(tmp_path / "src", expected, "s3://bucket/run")
    fetched = _stub_s3(monkeypatch, objects)
    index_uri = "s3://bucket/run/model.safetensors.index.json"
    cache_dir = tmp_path / "cache"

    cache(index_uri, cache_dir=str(cache_dir))
    assert len(fetched) == 3

    heads = []
    monkeypatch.setattr("fastgen.utils.s3.exists", lambda path: heads.append(path) or True)

    assert cache(index_uri, cache_dir=str(cache_dir)) == str(
        cache_dir / "bucket" / "run" / "model.safetensors.index.json"
    )
    assert len(fetched) == 3, "a cache hit must not re-download"
    assert heads == [], "a cache hit must not issue HEAD requests either"


def test_cache_passes_non_s3_paths_through(tmp_path):
    """A local checkpoint is already readable by every rank, so nothing is mirrored."""
    checkpoint_path = tmp_path / "model.pt"
    torch.save(TinyModel().state_dict(), checkpoint_path)
    cache_dir = tmp_path / "cache"

    assert cache(str(checkpoint_path), str(cache_dir)) == str(checkpoint_path)
    assert resolve_cached_path(str(checkpoint_path), str(cache_dir)) == str(checkpoint_path)
    assert not cache_dir.exists()


def test_resolve_cached_path_resolves_file_without_touching_s3(monkeypatch, tmp_path):
    """The resolver is a pure lookup: no download, no GET, no HEAD."""
    expected = TinyModel().state_dict()
    fetched = _stub_s3(monkeypatch, {"s3://bucket/model.pt": _torch_bytes(expected)})
    cache_dir = tmp_path / "cache"

    assert resolve_cached_path("s3://bucket/model.pt", cache_dir=str(cache_dir)) == "s3://bucket/model.pt"
    assert fetched == [], "resolving must not download"

    cache("s3://bucket/model.pt", cache_dir=str(cache_dir))
    assert resolve_cached_path("s3://bucket/model.pt", cache_dir=str(cache_dir)) == str(
        cache_dir / "bucket" / "model.pt"
    )
    assert len(fetched) == 1, "resolving after the mirror exists must not re-fetch"


def test_resolve_cached_path_rejects_an_incomplete_shard_mirror(monkeypatch, tmp_path):
    """An index without its shards is unusable, so a half-populated mirror must not win."""
    expected = TinyModel().state_dict()
    objects, shard_names = _sharded_safetensors_objects(tmp_path / "src", expected, "s3://bucket/run")
    _stub_s3(monkeypatch, objects)
    index_uri = "s3://bucket/run/model.safetensors.index.json"
    cache_dir = tmp_path / "cache"

    cache(index_uri, cache_dir=str(cache_dir))
    assert resolve_cached_path(index_uri, cache_dir=str(cache_dir)) == str(
        cache_dir / "bucket" / "run" / "model.safetensors.index.json"
    )

    (cache_dir / "bucket" / "run" / shard_names[0]).unlink()
    assert resolve_cached_path(index_uri, cache_dir=str(cache_dir)) == index_uri


def test_checkpoint_cache_dir_follows_fastgen_output_root(monkeypatch, tmp_path):
    """The mirror location lives in one place and never follows a remote CKPT_ROOT_DIR."""
    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("CKPT_ROOT_DIR", "s3://checkpoints")

    assert checkpoint_cache_dir() == str(tmp_path / "MODEL")


def test_mirror_path_preserves_safe_encoded_key_characters(tmp_path):
    assert mirror_path("s3://bucket/path%20with%20spaces/model%2Ept", str(tmp_path)) == str(
        tmp_path / "bucket" / "path with spaces" / "model.pt"
    )


@pytest.mark.parametrize("key", ["../outside", "%2e%2e/outside", "run/../outside"])
def test_mirror_path_rejects_relative_key_components(tmp_path, key):
    with pytest.raises(ValueError, match="relative path components"):
        mirror_path(f"s3://bucket/{key}", str(tmp_path))


@pytest.mark.parametrize("uri", ["s3://../outside", "s3://bucket/%2Foutside"])
def test_mirror_path_rejects_a_uri_that_escapes_the_cache_root(tmp_path, uri):
    with pytest.raises(ValueError, match="escapes cache directory"):
        mirror_path(uri, str(tmp_path))


def test_mirror_keys_on_bucket_and_key_not_basename(monkeypatch, tmp_path):
    """Two checkpoints sharing a file name must not share a cache entry."""
    first = {"w": torch.ones(2)}
    second = {"w": torch.zeros(2)}
    _stub_s3(
        monkeypatch,
        {
            "s3://bucket/runA/model.pth": _torch_bytes(first),
            "s3://bucket/runB/model.pth": _torch_bytes(second),
        },
    )
    cache_dir = tmp_path / "cache"

    a = cache("s3://bucket/runA/model.pth", cache_dir=str(cache_dir))
    b = cache("s3://bucket/runB/model.pth", cache_dir=str(cache_dir))

    assert a != b, "same base name must not collapse to one entry"
    assert_state_dict_equal(load_checkpoint_state_dict(a), first)
    assert_state_dict_equal(load_checkpoint_state_dict(b), second)
