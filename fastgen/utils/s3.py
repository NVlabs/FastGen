# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 transport operations."""

from __future__ import annotations

import io
import sys
import time
from collections.abc import Iterator
from typing import Any, BinaryIO
from urllib.parse import unquote, urlparse

import boto3
import botocore.exceptions
from botocore.config import Config


S3_PREFIX = "s3://"


def is_uri(path: str) -> bool:
    """Return whether ``path`` uses the S3 URI scheme."""
    assert isinstance(path, str), f"path must be a string, got {type(path).__name__}"
    return path.startswith(S3_PREFIX)


def create_client():
    """Create an S3 client through boto3's inherited credential provider chain."""
    return boto3.client(
        "s3",
        config=Config(
            connect_timeout=15,
            read_timeout=60,
            retries={"max_attempts": 10, "mode": "adaptive"},
            max_pool_connections=4,
        ),
    )


def parse_uri(uri: str, allow_bucket_root: bool = False) -> tuple[str, str]:
    """Parse an S3 URI into its bucket and decoded key.

    Bucket-root URIs return an empty key when ``allow_bucket_root`` is true.
    Otherwise, the URI must identify a single object.
    """
    parsed = urlparse(uri)
    key = unquote(parsed.path.lstrip("/"))
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or (not allow_bucket_root and not key)
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        expected = "s3://bucket[/prefix]" if allow_bucket_root else "s3://bucket/key"
        raise ValueError(f"expected {expected}, got {uri!r}")
    return parsed.netloc, key


def iter_objects(s3_prefix: str) -> Iterator[tuple[str, float]]:
    """Yield direct-child object keys and modification times under an S3 URI prefix.

    A bucket-root URI is valid because a listing prefix may be empty. Pagination
    stays behind this boundary so callers do not depend on boto3's response shape.
    """
    bucket, prefix = parse_uri(s3_prefix, allow_bucket_root=True)
    prefix = f"{prefix.rstrip('/')}/" if prefix else ""

    paginator = create_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for obj in page.get("Contents") or []:
            last_modified = obj.get("LastModified")
            mtime = float(last_modified.timestamp()) if last_modified is not None else 0.0
            yield obj["Key"], mtime


def load(s3_path: str) -> io.BytesIO:
    """Load an S3 object into memory using independently retryable byte ranges."""

    output = io.BytesIO()
    status = resumable_download(s3_path, output=output)
    if status != 0:
        raise IOError(f"Failed to download {s3_path}")
    output.seek(0)
    return output


def exists(s3_path: str) -> bool:
    """Return whether an object exists at ``s3_path`` (``head_object``, metadata only).

    Errors other than "not found" are re-raised, so bad credentials or an outage is never
    mistaken for a missing checkpoint and used to restart training from iteration 0.
    """

    bucket, key = parse_uri(s3_path)
    try:
        create_client().head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as error:
        if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    return True


def save(s3_path: str, data: bytes) -> None:
    """Save ``data`` as the object at ``s3_path``.

    Uses boto3's managed upload, which switches to multipart above its 8 MiB threshold, so monolithic
    checkpoints beyond the 5 GiB single-``put_object`` limit upload as well.
    """

    bucket, key = parse_uri(s3_path)
    create_client().upload_fileobj(io.BytesIO(data), bucket, key)


def delete(s3_path: str) -> None:
    """Delete the object at ``s3_path``.

    S3 deletion is idempotent, so this also succeeds when the object no longer
    exists. This is useful for invalidating a checkpoint's completion marker
    before overwriting its shards.
    """

    bucket, key = parse_uri(s3_path)
    create_client().delete_object(Bucket=bucket, Key=key)


def resumable_download(
    uri: str,
    range_bytes: int = 64 * 1024 * 1024,
    chunk_bytes: int = 8 * 1024 * 1024,
    max_retries: int = 50,
    max_bytes: int | None = None,
    output: BinaryIO | None = None,
) -> int:
    """Write an S3 object to ``output`` using independently retryable ranges.

    Each range is buffered completely and its response body is closed before
    any bytes are emitted. Consequently a failed GET can restart at the same
    offset without duplicating bytes, and WebDataset backpressure never leaves
    an HTTP response idle between training steps.

    Returns zero on success (including a downstream pipe closing early) and
    one after exhausting retries. ``max_bytes`` is intended for bounded soak
    tests.
    """
    if range_bytes <= 0 or chunk_bytes <= 0 or max_retries < 0:
        raise ValueError("range/chunk sizes must be positive and max retries non-negative")
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max bytes must be non-negative")

    bucket, key = parse_uri(uri)
    client = create_client()
    head = client.head_object(Bucket=bucket, Key=key)
    object_size = int(head["ContentLength"])
    target_size = object_size if max_bytes is None else min(object_size, max_bytes)
    etag = head.get("ETag")
    if output is None:
        output = sys.stdout.buffer

    offset = 0
    consecutive_failures = 0
    while offset < target_size:
        body = None
        range_end = min(offset + range_bytes, target_size) - 1
        expected_size = range_end - offset + 1
        request: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Range": f"bytes={offset}-{range_end}",
        }
        if etag:
            request["IfMatch"] = etag

        try:
            response = client.get_object(**request)
            body = response["Body"]
            response_size = response.get("ContentLength")
            if response_size is not None and int(response_size) != expected_size:
                raise IOError(f"range returned {response_size} bytes; expected {expected_size}")
            buffered = bytearray()
            while len(buffered) < expected_size:
                chunk = body.read(min(chunk_bytes, expected_size - len(buffered)))
                if not chunk:
                    raise EOFError(f"short range at byte {offset + len(buffered)} of {range_end + 1}")
                buffered.extend(chunk)
            body.close()
            body = None
        except Exception as error:  # The next ranged GET is the recovery boundary.
            consecutive_failures += 1
            if consecutive_failures > max_retries:
                print(
                    f"resumable_download: exhausted retries at byte {offset}/{target_size}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
            delay = min(0.25 * (2 ** min(consecutive_failures - 1, 6)), 10.0)
            print(
                f"resumable_download: reopening at byte {offset}/{target_size} after "
                f"{type(error).__name__}: {error}; retry={consecutive_failures}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            continue
        finally:
            if body is not None:
                try:
                    body.close()
                except Exception:  # Preserve the read error that selected the retry path.
                    pass

        try:
            output.write(buffered)
            output.flush()
        except BrokenPipeError:
            return 0
        offset += len(buffered)
        consecutive_failures = 0

    return 0
