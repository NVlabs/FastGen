# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for S3 transport and the distributed S3 filesystem adapter."""

from __future__ import annotations

import io
import signal
import subprocess
import sys
from pathlib import Path

import botocore.exceptions
import pytest

import fastgen.utils.s3 as s3
from fastgen.utils.distributed.s3_filesystem import S3FileSystem


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("s3://bucket/key", True),
        ("https://example.com/key", False),
        ("/local/key", False),
    ],
)
def test_is_uri(path, expected):
    assert s3.is_uri(path) is expected


def test_is_uri_requires_a_string():
    with pytest.raises(AssertionError, match="path must be a string"):
        s3.is_uri(Path("/local/key"))


class _FakeS3Client:
    def __init__(self):
        self.upload = None
        self.copies = []
        self.deletes = []
        self.copy_error = None
        self.content_length = 0
        self.managed_copies = []

    def upload_fileobj(self, stream, bucket, key):
        self.upload = (bucket, key, stream.read())

    def copy_object(self, **kwargs):
        self.copies.append(kwargs)
        if self.copy_error is not None:
            raise self.copy_error

    def head_object(self, **kwargs):
        return {"ContentLength": self.content_length}

    def copy(self, copy_source, bucket, key):
        self.managed_copies.append((copy_source, bucket, key))

    def delete_object(self, **kwargs):
        self.deletes.append(kwargs)


def test_s3_write_stream_survives_consumer_close(monkeypatch):
    """DCP closes its stream before the S3 context manager uploads it."""
    client = _FakeS3Client()
    monkeypatch.setattr(
        "fastgen.utils.distributed.s3_filesystem.boto3.client",
        lambda *args, **kwargs: client,
    )

    filesystem = S3FileSystem()
    with filesystem.create_stream("s3://bucket/checkpoint/__0_0.distcp", "wb") as stream:
        stream.write(b"checkpoint-bytes")
        stream.close()
        assert not stream.closed

    assert stream.closed
    assert client.upload == ("bucket", "checkpoint/__0_0.distcp", b"checkpoint-bytes")


def test_dcp_and_marker_writes_resolve_the_same_encoded_prefix(monkeypatch):
    client = _FakeS3Client()
    monkeypatch.setattr(s3.boto3, "client", lambda *args, **kwargs: client)
    prefix = "s3://bucket/path%20with%20spaces/0000001"

    s3.save(prefix + ".pth", b"marker")
    marker_key = client.upload[1]
    with S3FileSystem().create_stream(prefix + ".net_model/.metadata", "wb") as stream:
        stream.write(b"metadata")

    assert marker_key == "path with spaces/0000001.pth"
    assert client.upload == ("bucket", marker_key.removesuffix(".pth") + ".net_model/.metadata", b"metadata")


@pytest.mark.parametrize("uri", ["S3://bucket/key", " s3://bucket/key", "s3://bucket", "s3:///key"])
def test_dcp_checkpoint_validation_rejects_invalid_paths(uri):
    assert not S3FileSystem.validate_checkpoint_id(uri)


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_dcp_filesystem_recognizes_missing_objects(monkeypatch, code):
    def missing(**kwargs):
        raise botocore.exceptions.ClientError({"Error": {"Code": code}}, "HeadObject")

    _stub_s3_client(monkeypatch, head_object=missing)
    assert not S3FileSystem().exists("s3://bucket/missing")


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda filesystem: filesystem.create_stream("s3://bucket", "rb").__enter__(), id="create_stream"),
        pytest.param(lambda filesystem: filesystem.rename("s3://bucket", "s3://bucket/key"), id="rename_source"),
        pytest.param(lambda filesystem: filesystem.rename("s3://bucket/key", "s3://bucket"), id="rename_destination"),
        pytest.param(lambda filesystem: filesystem.rm_file("s3://bucket"), id="rm_file"),
    ],
)
def test_dcp_object_operations_reject_uri_without_an_object_key(monkeypatch, operation):
    _stub_s3_client(monkeypatch)

    with pytest.raises(ValueError, match="expected s3://bucket/key"):
        operation(S3FileSystem())


def test_dcp_mkdir_at_a_bucket_root_is_a_no_op(monkeypatch):
    """The stub client has no methods, so any S3 request would raise."""
    _stub_s3_client(monkeypatch)

    S3FileSystem().mkdir("s3://bucket")


def test_s3_filesystem_uses_the_default_credential_chain(monkeypatch):
    """``setup`` loads ``credentials_dir/s3.json`` into the environment, where boto3 finds it."""
    calls = []
    monkeypatch.setattr(
        "fastgen.utils.distributed.s3_filesystem.boto3.client",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    S3FileSystem()

    assert calls == [(("s3",), {})]


def test_s3_rename_uses_copy_object_then_deletes_source(monkeypatch):
    """Avoid managed-copy ETag validation on S3-compatible endpoints.

    The decoded key is passed as a dict ``CopySource``, which botocore percent-encodes exactly once;
    a pre-quoted string would be encoded twice and miss keys with reserved characters.
    """
    client = _FakeS3Client()
    monkeypatch.setattr(
        "fastgen.utils.distributed.s3_filesystem.boto3.client",
        lambda *args, **kwargs: client,
    )

    filesystem = S3FileSystem()
    filesystem.rename("s3://source/path%20with%20spaces/.metadata.tmp", "s3://dest/path%20with%20spaces/.metadata")

    assert client.copies == [
        {
            "Bucket": "dest",
            "Key": "path with spaces/.metadata",
            "CopySource": {"Bucket": "source", "Key": "path with spaces/.metadata.tmp"},
        }
    ]
    assert client.deletes == [{"Bucket": "source", "Key": "path with spaces/.metadata.tmp"}]


def test_s3_rename_uses_managed_copy_for_large_objects(monkeypatch):
    client = _FakeS3Client()
    client.content_length = 5 * 1024**3 + 1
    monkeypatch.setattr(
        "fastgen.utils.distributed.s3_filesystem.boto3.client",
        lambda *args, **kwargs: client,
    )

    filesystem = S3FileSystem()
    filesystem.rename("s3://source/large.bin", "s3://dest/large.bin")

    assert client.copies == []
    assert client.managed_copies == [({"Bucket": "source", "Key": "large.bin"}, "dest", "large.bin")]
    assert client.deletes == [{"Bucket": "source", "Key": "large.bin"}]


def test_s3_rename_does_not_mask_small_object_copy_errors(monkeypatch):
    client = _FakeS3Client()
    client.copy_error = botocore.exceptions.ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}},
        "CopyObject",
    )
    client.content_length = 1024
    monkeypatch.setattr(
        "fastgen.utils.distributed.s3_filesystem.boto3.client",
        lambda *args, **kwargs: client,
    )

    filesystem = S3FileSystem()
    with pytest.raises(botocore.exceptions.ClientError, match="AccessDenied"):
        filesystem.rename("s3://source/small.bin", "s3://dest/small.bin")

    assert client.managed_copies == []
    assert client.deletes == []


def _stub_s3_client(monkeypatch, **methods):
    """Install a stub boto3 S3 client exposing ``methods``."""
    client = type("_StubS3Client", (), {name: staticmethod(fn) for name, fn in methods.items()})()
    monkeypatch.setattr("fastgen.utils.s3.boto3.client", lambda *args, **kwargs: client)


def test_iter_objects_lists_direct_children(monkeypatch):
    seen = {}

    class _Paginator:
        def paginate(self, **kwargs):
            seen.update(kwargs)
            return iter(({"Contents": [{"Key": "run/checkpoints/0000042.pth"}]},))

    def get_paginator(name):
        assert name == "list_objects_v2"
        return _Paginator()

    _stub_s3_client(monkeypatch, get_paginator=get_paginator)

    assert list(s3.iter_objects("s3://bucket/run/checkpoints")) == [("run/checkpoints/0000042.pth", 0.0)]
    assert seen == {"Bucket": "bucket", "Prefix": "run/checkpoints/", "Delimiter": "/"}


def test_create_client_uses_inherited_provider_chain(monkeypatch):
    captured = {}
    sentinel = object()

    def make_client(service_name, **kwargs):
        captured["service_name"] = service_name
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr("fastgen.utils.s3.boto3.client", make_client)

    assert s3.create_client() is sentinel
    assert captured["service_name"] == "s3"
    assert set(captured["kwargs"]) == {"config"}


def test_resumable_download_retries_whole_ranges_without_duplicate_bytes(monkeypatch):
    """A partial response is discarded before retrying, keeping the tar stream exact."""
    payload = b"0123456789"
    requests = []
    bodies = []
    sleeps = []
    attempts = {}

    class Body:
        def __init__(self, data, fail_after_first_read=False):
            self.data = data
            self.fail_after_first_read = fail_after_first_read
            self.offset = 0
            self.closed = False
            bodies.append(self)

        def read(self, size):
            if self.fail_after_first_read and self.offset:
                raise OSError("interrupted response")
            chunk = self.data[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

        def close(self):
            self.closed = True

    class Client:
        @staticmethod
        def head_object(**kwargs):
            assert kwargs == {"Bucket": "bucket", "Key": "data/shard.tar"}
            return {"ContentLength": len(payload), "ETag": '"fixed-etag"'}

        @staticmethod
        def get_object(**kwargs):
            requests.append(kwargs)
            byte_range = kwargs["Range"]
            attempts[byte_range] = attempts.get(byte_range, 0) + 1
            start, end = (int(value) for value in byte_range.removeprefix("bytes=").split("-"))
            data = payload[start : end + 1]
            fail = byte_range == "bytes=0-3" and attempts[byte_range] == 1
            return {"Body": Body(data, fail_after_first_read=fail), "ContentLength": len(data)}

    def make_client(service_name, **kwargs):
        assert service_name == "s3"
        assert "config" in kwargs
        assert set(kwargs) == {"config"}
        return Client()

    monkeypatch.setattr("fastgen.utils.s3.boto3.client", make_client)
    monkeypatch.setattr("fastgen.utils.s3.time.sleep", sleeps.append)
    output = io.BytesIO()

    assert (
        s3.resumable_download(
            "s3://bucket/data/shard.tar",
            range_bytes=4,
            chunk_bytes=2,
            max_retries=1,
            output=output,
        )
        == 0
    )
    assert output.getvalue() == payload
    assert [request["Range"] for request in requests] == [
        "bytes=0-3",
        "bytes=0-3",
        "bytes=4-7",
        "bytes=8-9",
    ]
    assert all(request["IfMatch"] == '"fixed-etag"' for request in requests)
    assert sleeps == [0.25]
    assert all(body.closed for body in bodies)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"range_bytes": 0}, "sizes must be positive"),
        ({"chunk_bytes": 0}, "sizes must be positive"),
        ({"max_retries": -1}, "max retries non-negative"),
        ({"max_bytes": -1}, "max bytes must be non-negative"),
    ],
)
def test_resumable_download_rejects_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        s3.resumable_download("s3://bucket/shard.tar", **kwargs)


def test_resumable_s3_download_module_help():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.resumable_s3_download", "--help"],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "S3 object URI" in result.stdout


def test_resumable_s3_download_module_exits_quietly_when_its_reader_stops():
    """A reader that stops mid-object, like WebDataset leaving a shard, ends the download with SIGPIPE."""
    # Run the module as ``__main__`` with an endless download in place of S3.
    code = (
        "import runpy, sys\n"
        "import fastgen.utils.s3 as s3\n"
        "def endless(uri, **kwargs):\n"
        "    while True:\n"
        "        sys.stdout.buffer.write(bytes(65536))\n"
        "s3.resumable_download = endless\n"
        "sys.argv = ['resumable_s3_download', 's3://bucket/shard.tar']\n"
        "runpy.run_module('scripts.resumable_s3_download', run_name='__main__')\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code], cwd=_REPOSITORY_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    process.stdout.read(1024)
    process.stdout.close()
    _, stderr = process.communicate(timeout=30)

    assert process.returncode == -signal.SIGPIPE, stderr.decode()
    assert b"BrokenPipeError" not in stderr


def _raise_client_error(code: str):
    def head_object(**kwargs):
        raise botocore.exceptions.ClientError({"Error": {"Code": code}}, "HeadObject")

    return head_object


def test_exists_parses_bucket_and_key(monkeypatch):
    seen = {}

    def head_object(**kwargs):
        seen.update(kwargs)
        return {"ContentLength": 1}

    _stub_s3_client(monkeypatch, head_object=head_object)

    assert s3.exists("s3://bucket/prefix/model.pth") is True
    assert seen == {"Bucket": "bucket", "Key": "prefix/model.pth"}


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_exists_is_false_for_missing_object(monkeypatch, code):
    _stub_s3_client(monkeypatch, head_object=_raise_client_error(code))

    assert s3.exists("s3://bucket/prefix/model.pth") is False


def test_exists_propagates_errors_other_than_missing(monkeypatch):
    """Bad credentials, denied access, and outages must not read as "no object"."""
    _stub_s3_client(monkeypatch, head_object=_raise_client_error("AccessDenied"))

    with pytest.raises(botocore.exceptions.ClientError):
        s3.exists("s3://bucket/prefix/model.pth")


def test_load_reads_the_object_at_bucket_and_key(monkeypatch):
    """``load`` uses ranged reads and returns a rewound in-memory buffer."""
    seen = {"head": [], "get": []}
    payload = b"payload-bytes"

    def head_object(**kwargs):
        seen["head"].append(kwargs)
        return {"ContentLength": len(payload)}

    def get_object(**kwargs):
        seen["get"].append(kwargs)
        return {"Body": io.BytesIO(payload), "ContentLength": len(payload)}

    _stub_s3_client(monkeypatch, head_object=head_object, get_object=get_object)

    assert s3.load("s3://bucket/prefix/model.pt").read() == payload
    assert seen["head"] == [{"Bucket": "bucket", "Key": "prefix/model.pt"}]
    assert seen["get"] == [{"Bucket": "bucket", "Key": "prefix/model.pt", "Range": f"bytes=0-{len(payload) - 1}"}]


def test_save_writes_to_bucket_and_key(monkeypatch):
    """``save`` is the mirror image of ``load`` and must split the URI the same way."""
    seen = {}

    def upload_fileobj(stream, bucket, key):
        seen.update({"Bucket": bucket, "Key": key, "Body": stream.read()})

    _stub_s3_client(monkeypatch, upload_fileobj=upload_fileobj)
    s3.save("s3://bucket/prefix/model.pt", b"payload-bytes")

    assert seen == {"Bucket": "bucket", "Key": "prefix/model.pt", "Body": b"payload-bytes"}


def test_delete_removes_object_at_bucket_and_key(monkeypatch):
    """``delete`` uses the same URI parsing contract as the other object operations."""
    seen = {}

    def delete_object(**kwargs):
        seen.update(kwargs)

    _stub_s3_client(monkeypatch, delete_object=delete_object)
    s3.delete("s3://bucket/prefix/model.pt")

    assert seen == {"Bucket": "bucket", "Key": "prefix/model.pt"}


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(s3.load, id="load"),
        pytest.param(s3.exists, id="exists"),
        pytest.param(lambda path: s3.save(path, b"payload"), id="save"),
        pytest.param(s3.delete, id="delete"),
    ],
)
def test_s3_object_operations_reject_uri_without_an_object_key(operation):
    with pytest.raises(ValueError, match="expected s3://bucket/key"):
        operation("s3://bucket")


@pytest.mark.parametrize(
    "uri",
    [
        "https://bucket/key",
        "s3:///key",
        "s3://bucket/key?versionId=123",
        "s3://bucket/key#fragment",
    ],
)
def test_parse_uri_rejects_malformed_or_ambiguous_uris(uri):
    with pytest.raises(ValueError, match="expected s3://bucket/key"):
        s3.parse_uri(uri)


def test_parse_uri_decodes_an_encoded_object_key():
    assert s3.parse_uri("s3://bucket/path%20with%20spaces/model%2Ept") == (
        "bucket",
        "path with spaces/model.pt",
    )


def test_parse_uri_allows_a_bucket_root_when_requested():
    assert s3.parse_uri("s3://bucket", allow_bucket_root=True) == ("bucket", "")
