# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import os
from fastgen.utils import logging_utils as logger
from fastgen.utils.env import output_root
from fastgen.callbacks.callback import Callback
import atexit
import pickle
from typing import Callable, Optional, TYPE_CHECKING
import base64
import json

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


def create_dump(dump_path):
    logger.critical(f"Creating {dump_path}")
    if not dump_path.endswith("html"):
        logger.warning(f"create_dump produces an HTML file but was called with {dump_path=}")
    torch.cuda.memory._dump_snapshot(dump_path + ".pickle")
    with open(dump_path + ".pickle", "rb") as f:
        data = pickle.load(f)
    _memory_viz_template = r"""
    <!DOCTYPE html>
    <html>
    <head>
    </head>
    <body>
    <script type="module">
    import {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"
    const local_files = $SNAPSHOT
    add_local_files(local_files, $VIZ_KIND)
    </script>
    </body>
    """

    # find which GPU was active
    idx_device = next((i for i, trace in enumerate(data["device_traces"]) if trace), None)
    if idx_device is None:
        logger.warning(f"No GPU recorded allocation events; skipping {dump_path}")
        return

    traces = data["device_traces"][idx_device]  # create an aliasing variable for convenience
    traces = [
        d for d in traces if d["action"] == "alloc" or d["action"] == "free_completed"
    ]  # only the `alloc` and `free_completed` events matter for our visualization

    for d in traces:
        d["fastgen_frames"] = [
            f for f in d["frames"] if "fastgen" in f["filename"]
        ]  # get the callstack frames from fastgen code (e.g. ignore frames in pytorch/other libraries)
        if not d["fastgen_frames"]:
            d["fastgen_frames"] = d["frames"]

    # run through the trace and find allocations that were allocated but never freed
    set_alloced_addrs: dict = {}
    for d in traces:
        if d["action"] == "alloc":
            set_alloced_addrs[d["addr"]] = d
        elif d["action"] == "free_completed":
            if d["addr"] in set_alloced_addrs:
                del set_alloced_addrs[d["addr"]]
        else:
            raise NotImplementedError(f"{d['action']}")

    never_freed_traces = list(set_alloced_addrs.values())
    KB = 1 << 10
    never_freed_traces = [t for t in never_freed_traces if t["size"] > KB]  # get rid of allocations below 1 KB

    # now proceed through the trace (guarenteed to be all `alloc` events as we removed all free events).
    # for each pair of alloc events, merge them iff they share a common fastgen ancestor.
    # Merging events is useful as it both speeds up the visualization rendering and also makes it more understandable.
    i = 0
    while i < len(never_freed_traces) - 1:
        curr_frames = never_freed_traces[i]["fastgen_frames"]
        next_frames = never_freed_traces[i + 1]["fastgen_frames"]
        if (
            curr_frames and next_frames and curr_frames[0] == next_frames[0]
        ):  # TODO: probably should compare the full callstack
            # same ancestor, delete next event and add its size to current event
            never_freed_traces[i]["size"] += never_freed_traces[i + 1]["size"]
            never_freed_traces.pop(i + 1)
        else:
            i += 1  # different ancestor, do not combine and move on

    data["device_traces"][idx_device] = never_freed_traces  # update the trace to only be the merged-alloc events
    data["segments"] = []  # shrink the trace, unused in memory timeline
    data["external_annotations"] = []  # shrink the trace, unused in memory timeline
    buffer = pickle.dumps(data)
    buffer += b"\x00" * (3 - len(buffer) % 3)
    encoded_buffer = base64.b64encode(buffer).decode("utf-8")
    json_format = json.dumps([{"name": "snapshot.pickle", "base64": encoded_buffer}])
    html_src = _memory_viz_template.replace("$VIZ_KIND", repr("Active Memory Timeline")).replace(
        "$SNAPSHOT", json_format
    )
    with open(dump_path, "w") as f:
        f.write(html_src)
    os.remove(dump_path + ".pickle")


class MemTrackerCallback(Callback):
    def __init__(self, save_every_n_iters: Optional[int] = None, deactivate_after_n_iters: int = 100):
        def close_and_save():
            create_dump(f"{self._dump_dir()}/crash_rank{os.environ.get('RANK', '0')}.html")

        self.deactivate_after_n_iters = deactivate_after_n_iters  # Deactivate eventually to prevent leaking host memory
        self.save_every_n_iters = save_every_n_iters
        self.atexit_fn = close_and_save
        atexit.register(self.atexit_fn)

    def _dump_dir(self) -> str:
        """Run directory of the attached config, else the shared output root."""
        return self.config.log_config.save_path if hasattr(self, "config") else output_root()

    def on_app_begin(self):
        logger.info("[MemTrackerCallback] Tracking peak memory usage")
        torch.cuda.memory._record_memory_history(stacks="python")

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        if iteration > self.deactivate_after_n_iters:
            torch.cuda.memory._record_memory_history(enabled=None)  # frees pytorch tracking datastructures
        if self.save_every_n_iters is not None and (iteration % self.save_every_n_iters) == 0:
            create_dump(f"{self._dump_dir()}/step{iteration}_rank{os.environ.get('RANK', '0')}.html")
