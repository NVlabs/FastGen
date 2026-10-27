#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream an S3 object to stdout using independently retryable byte ranges."""

from __future__ import annotations

import argparse
import signal

import fastgen.utils.s3 as s3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uri", help="S3 object URI (s3://bucket/key)")
    parser.add_argument(
        "--range-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Maximum size of each bounded ranged GET",
    )
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-retries", type=int, default=50)
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Stop after this many bytes; intended for bounded soak tests",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return s3.resumable_download(
        args.uri,
        range_bytes=args.range_bytes,
        chunk_bytes=args.chunk_bytes,
        max_retries=args.max_retries,
        max_bytes=args.max_bytes,
    )


if __name__ == "__main__":
    # A reader that stops mid-object closes the pipe; exit quietly like other Unix producers instead of printing a
    # BrokenPipeError traceback.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(main())
