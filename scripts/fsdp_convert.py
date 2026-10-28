#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export one network of a DCP checkpoint as a plain ``torch.save`` state dict.

The output ``<iter>.<model-type>.pth`` holds the bare state dict of one network for external tools such as
``scripts/inference/flashdreams_inference.py``. FastGen's own loaders do not read it; they take the DCP marker
``<iter>.pth`` directly. ``ckpt_dir`` must be a local directory: download checkpoints written to an ``s3://``
``trainer.checkpointer.save_dir`` first.
"""

import argparse
import os
import tempfile

from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def parse_iter_list(s: str) -> list[int]:
    """Parse a comma-separated list of iteration numbers, or a ``start:stop:step`` range that excludes ``stop``."""
    if ":" in s:
        parts = [int(x) for x in s.split(":")]
        if len(parts) == 2:
            parts.append(1)
        return list(range(parts[0], parts[1], parts[2]))
    return [int(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Export one network of a DCP checkpoint as a plain torch.save state dict for external tools "
        "(e.g. scripts/inference/flashdreams_inference.py). FastGen's own loaders read the DCP marker <iter>.pth "
        "directly instead."
    )
    parser.add_argument("ckpt_dir", type=str, help="Local checkpoint directory containing numbered sub-folders")
    parser.add_argument(
        "--iters",
        type=str,
        required=True,
        help="Iterations to convert: comma-separated (e.g. 10000,12000) or range start:stop:step (stop excluded)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="net_model",
        help="Model type suffix appended to the iteration folder (default: net_model)",
    )
    args = parser.parse_args()

    iters = parse_iter_list(args.iters)
    print(f"Converting iters {iters} from {args.ckpt_dir} (model type: {args.model_type})")

    for it in iters:
        dcp_dir = os.path.join(args.ckpt_dir, f"{it:07d}.{args.model_type}")
        pth_path = os.path.join(args.ckpt_dir, f"{it:07d}.{args.model_type}.pth")

        if not os.path.exists(dcp_dir):
            print(f"[skip] {dcp_dir} does not exist")
            continue

        if os.path.exists(pth_path):
            print(f"[skip] {pth_path} already exists")
            continue

        # A failed conversion must not leave a final file that reruns mistake for a complete checkpoint.
        with tempfile.TemporaryDirectory(dir=args.ckpt_dir, prefix=f".{it:07d}.convert-") as temporary_dir:
            temporary_path = os.path.join(temporary_dir, os.path.basename(pth_path))
            dcp_to_torch_save(dcp_dir, temporary_path)
            with open(temporary_path, "rb") as temporary_file:
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, pth_path)
        print(f"[done] iter {it} -> {pth_path}")


if __name__ == "__main__":
    main()
