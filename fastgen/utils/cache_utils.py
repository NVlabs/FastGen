# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic local publication and remote-to-local cache helpers."""

from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import shutil
import uuid
from collections.abc import Callable
from typing import BinaryIO, Optional
import http.client
import urllib.error
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import botocore.exceptions

import fastgen.utils.logging_utils as logger
import fastgen.utils.s3 as s3
from fastgen.utils.env import output_root


_HTTP_PREFIXES = ("http://", "https://")
SAFETENSORS_SUFFIX = ".safetensors"
SAFETENSORS_INDEX_SUFFIX = ".safetensors.index.json"
_RECOGNIZED_HTTP_SUFFIXES = (
    SAFETENSORS_INDEX_SUFFIX,
    SAFETENSORS_SUFFIX,
    ".ckpt",
    ".pth",
    ".pt",
    ".bin",
)
_HTTP_USER_AGENT = "fastgen"
_HTTP_TIMEOUT_S = 60
_HTTP_DOWNLOAD_ATTEMPTS = 3


def is_http_url(path: str) -> bool:
    """Return whether ``path`` uses HTTP or HTTPS."""
    assert isinstance(path, str), f"path must be a string, got {type(path).__name__}"
    return path.startswith(_HTTP_PREFIXES)


def _path_component(path: str | os.PathLike[str]) -> str:
    """Return the filesystem or URI path without query and fragment components."""
    path = os.fspath(path)
    return urlsplit(path).path if "://" in path else path


def file_extension(path: str | os.PathLike[str]) -> str:
    """Return the lowercase final extension of a local path or URI."""
    return os.path.splitext(_path_component(path))[1].lower()


def is_safetensors_file(path: str) -> bool:
    """Return whether ``path`` names a single-file SafeTensors payload."""
    assert isinstance(path, str), f"path must be a string, got {type(path).__name__}"
    return _path_component(path).endswith(SAFETENSORS_SUFFIX)


def is_safetensors_index(path: str) -> bool:
    """Return whether ``path`` names a sharded SafeTensors index."""
    assert isinstance(path, str), f"path must be a string, got {type(path).__name__}"
    return _path_component(path).endswith(SAFETENSORS_INDEX_SUFFIX)


def _http_suffix(path: str) -> str:
    """Return the longest recognized suffix on an HTTP URL path."""
    url_path = _path_component(path)
    return max((suffix for suffix in _RECOGNIZED_HTTP_SUFFIXES if url_path.endswith(suffix)), key=len, default="")


def atomic_write(
    path: str | os.PathLike[str],
    writer: Callable[[BinaryIO], object],
    fsync: bool = False,
    sole_writer: bool = False,
) -> None:
    """Write ``path`` through a sibling temporary file and atomically replace it.

    ``writer`` receives an exclusively created binary file. The temporary file
    lives beside ``path`` so ``os.replace`` does not cross filesystem boundaries.
    Its name is unique per write, so concurrent writers never share one, and a
    write killed outside Python leaves it for ``remove_atomic_write_orphans``.
    A caller that is the only writer of ``path`` may pass ``sole_writer=True``:
    the temporary is then always ``<path>.tmp``, and each write first removes one
    left by an interrupted write, so no sweep is needed.
    If ``fsync`` is true, file contents are flushed to storage before the name is
    published. Any failure leaves an existing destination untouched and removes
    the temporary file. The parent directory is not fsynced, so a system crash
    may still lose the rename itself.
    """
    destination = os.fspath(path)
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    if sole_writer:
        temporary_path = f"{destination}.tmp"
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
    else:
        temporary_path = f"{destination}.{uuid.uuid4().hex}.tmp"
    temporary_created = False

    try:
        with open(temporary_path, "xb") as temporary_file:
            temporary_created = True
            writer(temporary_file)
            if fsync:
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
        temporary_created = False
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def remove_atomic_write_orphans(
    directory: str | os.PathLike[str],
    destination_pattern: str = "*",
) -> list[str]:
    """Remove abandoned ``atomic_write`` temporaries for matching destinations.

    ``destination_pattern`` matches destination basenames, for example
    ``"*.pth"``. Callers must have exclusive ownership of matching writes:
    an orphan is indistinguishable from a temporary file still being written.
    Returns the paths that were removed.
    """
    if os.path.basename(destination_pattern) != destination_pattern:
        raise ValueError(f"destination_pattern must be a basename pattern, got {destination_pattern!r}")

    uuid_pattern = "[0-9a-f]" * 32
    temporary_pattern = os.path.join(
        glob.escape(os.fspath(directory)),
        f"{destination_pattern}.{uuid_pattern}.tmp",
    )
    removed = []
    for temporary_path in sorted(glob.iglob(temporary_pattern)):
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            continue
        removed.append(temporary_path)
    return removed


def mirror_path(path: str, cache_dir: str) -> str:
    """Return the deterministic local mirror path for a remote object.

    S3 paths preserve ``<bucket>/<key>`` so a shard index lands beside its shards.
    HTTP URLs use a digest of the complete URL, including any query string, and
    retain recognized format suffixes.
    """
    if is_http_url(path):
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, "url", f"{digest}{_http_suffix(path)}")
    else:
        bucket, key = s3.parse_uri(path)
        if any(component in (".", "..") for component in key.split("/")):
            raise ValueError(f"cannot mirror S3 key with relative path components: {path!r}")
        cache_path = os.path.join(cache_dir, bucket, key)
    cache_root = os.path.abspath(cache_dir)
    if os.path.commonpath((cache_root, os.path.abspath(cache_path))) != cache_root:
        raise ValueError(f"mirror path escapes cache directory: {path!r}")
    return cache_path


class _NotAFileError(IOError):
    """The URL answered with an HTML page, which no retry turns into the requested file."""


def _download_http(url: str, output: BinaryIO) -> None:
    """Stream ``url`` into ``output``, rejecting HTML bodies and truncated transfers.

    Hosting services answer sign-in walls and download interstitials with ``200 text/html``, and a
    connection that drops mid-body returns short without an error, so the body is checked against
    ``Content-Length`` before the caller publishes it. Transient failures are retried, including the
    408 (timeout) and 429 (rate limit) answers; other client errors (403 for a gated repository, 404) are not.
    """
    for attempt in range(1, _HTTP_DOWNLOAD_ATTEMPTS + 1):
        output.seek(0)
        output.truncate()
        try:
            request = Request(url, headers={"User-Agent": _HTTP_USER_AGENT})
            with urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
                if response.headers.get_content_type() == "text/html":
                    raise _NotAFileError(
                        f"{url} returned an HTML page instead of a file (sign-in or download interstitial?)"
                    )
                shutil.copyfileobj(response, output)
                content_length = response.headers.get("Content-Length")
            received = output.tell()
            if received == 0:
                raise IOError(f"No data received from {url}")
            if content_length is not None and received != int(content_length):
                raise IOError(f"Truncated download from {url}: received {received} of {content_length} bytes")
            return
        except (OSError, http.client.HTTPException, ValueError) as error:
            if isinstance(error, _NotAFileError) or (
                isinstance(error, urllib.error.HTTPError) and 400 <= error.code < 500 and error.code not in (408, 429)
            ):
                raise
            if attempt == _HTTP_DOWNLOAD_ATTEMPTS:
                raise
            logger.warning(f"Download attempt {attempt}/{_HTTP_DOWNLOAD_ATTEMPTS} for {url} failed: {error}; retrying")


def load_http(url: str) -> io.BytesIO:
    """Load a single HTTP(S) object into memory."""
    output = io.BytesIO()
    _download_http(url, output)
    output.seek(0)
    return output


def cache_object(path: str, cache_dir: str) -> str:
    """Mirror one remote object into ``cache_dir`` verbatim, returning its local path.

    The download lands in a unique temp file and is renamed in. ``os.replace`` is atomic,
    which matters because the cache lookup is only ``os.path.isfile``: without it a
    concurrent or preempted write is read back as a complete object.
    """
    if "://" in cache_dir:
        # A mirror must be a local path, not another remote URI.
        raise ValueError(f"cache_dir must be a local directory, got {cache_dir}")

    cache_path = mirror_path(path, cache_dir)
    if os.path.isfile(cache_path):
        return cache_path

    def download(cache_file: BinaryIO) -> None:
        logger.info(f"Downloading {path} to {cache_path}")
        if is_http_url(path):
            _download_http(path, cache_file)
        else:
            status = s3.resumable_download(path, output=cache_file)
            if status != 0:
                raise IOError(f"Failed to download {path}")

    atomic_write(cache_path, download)
    return cache_path


def checkpoint_cache_dir() -> str:
    """Local directory used for checkpoint and model-asset mirrors.

    Not ``CKPT_ROOT_DIR``: that may itself be an ``s3://`` prefix, and a mirror must be local.
    """
    return os.path.join(output_root(), "MODEL")


def _index_shard_names(index_path: str) -> list[str]:
    """Shard names a SafeTensors index references, or ``[]`` for any other file.

    A malformed index yields no shards; the eventual load reports the ``weight_map`` contract.
    """
    if not is_safetensors_index(index_path):
        return []
    with open(index_path, encoding="utf-8") as index_file:
        try:
            weight_map = json.load(index_file).get("weight_map")
        except (ValueError, AttributeError):  # not JSON, or JSON that is not a mapping
            return []
    return sorted(set(weight_map.values())) if isinstance(weight_map, dict) else []


def _s3_shard_uri(checkpoint_path: str, shard_name: str) -> str:
    """Resolve a shard named by an index against the index's own S3 prefix."""
    shard_uri = os.path.join(os.path.dirname(checkpoint_path), shard_name)
    if not s3.exists(shard_uri):
        raise FileNotFoundError(f"Checkpoint shard {shard_name} referenced by {checkpoint_path} does not exist")
    return shard_uri


def resolve_cached_path(path: str, cache_dir: Optional[str] = None) -> str:
    """Resolve a remote path to a ready cached local mirror when available.

    Local inputs and uncached remote paths pass through unchanged. This is a pure lookup:
    it never contacts the remote source. An S3 SafeTensors shard index counts as cached
    only when the index and every referenced shard are present.
    """
    if not (s3.is_uri(path) or is_http_url(path)):
        return path

    cache_dir = cache_dir or checkpoint_cache_dir()
    cache_path = mirror_path(path, cache_dir)
    if is_http_url(path):
        return cache_path if os.path.isfile(cache_path) else path

    shard_dir = os.path.dirname(cache_path)
    complete = os.path.isfile(cache_path) and all(
        os.path.isfile(os.path.join(shard_dir, shard_name)) for shard_name in _index_shard_names(cache_path)
    )
    return cache_path if complete else path


def _fails_at_the_source(error: Exception) -> bool:
    """Whether reading the original URI would fail the same way, leaving a non-strict caller no fallback."""
    if isinstance(error, botocore.exceptions.ClientError):
        return error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") in (403, 404)
    return isinstance(error, botocore.exceptions.NoCredentialsError)


def cache(path: str, cache_dir: Optional[str] = None, strict: bool = True) -> str:
    """Mirror an S3 or HTTP(S) file locally, returning the path to read from.

    An S3 SafeTensors index brings along all referenced shards; HTTP(S) supports
    single-file objects only. Local inputs pass through unchanged.

    ``strict`` states what a failed mirror means to the caller. A caller that can read the
    original URI directly passes ``strict=False`` to downgrade the failure to a warning.
    Missing S3 credentials and S3 403/404 responses raise either way, because reading the original
    URI fails the same way.
    """
    if not (s3.is_uri(path) or is_http_url(path)):
        return path
    cache_dir = cache_dir or checkpoint_cache_dir()

    try:
        if is_safetensors_index(path) and not s3.is_uri(path):
            raise ValueError(f"Caching sharded SafeTensors indexes is only supported for S3 URIs, got {path!r}")
        cached_path = cache_object(path, cache_dir)
        if is_safetensors_index(path):
            # A shard index is unusable without its shards, so pull those too. Skip mirrored
            # shards before ``_s3_shard_uri`` to avoid a HEAD request on every cache hit.
            for shard_name in _index_shard_names(cached_path):
                if not os.path.isfile(os.path.join(os.path.dirname(cached_path), shard_name)):
                    cache_object(_s3_shard_uri(path, shard_name), cache_dir)

        resolved = resolve_cached_path(path, cache_dir)
        if resolved == path:
            raise RuntimeError(f"Cache is incomplete after mirroring {path}")
        return resolved
    except Exception as error:
        if strict or _fails_at_the_source(error):
            raise
        logger.warning(f"Could not mirror {path} into {cache_dir}: {error}. Using the original URI.")
        return path
