# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared timestamp cadence for video-frame resampling."""

from __future__ import annotations

import math


def validate_frame_rate(value: float, name: str = "frame_rate") -> float:
    """Return ``value`` as a finite positive float."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


class FrameCaptureCadence:
    """Track fixed-interval capture points on a stream timestamp axis.

    ``emissions_at`` returns how many capture points the current source frame
    has reached: zero drops it, one keeps it, and larger values duplicate it.
    """

    _TOLERANCE_RATIO = 1e-3

    def __init__(self, capture_interval: float, anchor: float | None = None):
        self.capture_interval = validate_frame_rate(capture_interval, "capture_interval")
        self._anchor = anchor
        self._captured = 0
        self._tolerance = self.capture_interval * self._TOLERANCE_RATIO

    def emissions_at(self, timestamp: float, max_emissions: int) -> int:
        """Return the capture points emitted by ``timestamp``, at most ``max_emissions``.

        Withheld points carry over to the next call. The cap is required because it is the loop's only bound
        when a timestamp lies far, or infinitely far, past the anchor.
        """
        if self._anchor is None:
            self._anchor = timestamp

        emitted = 0
        while timestamp >= self._anchor + self._captured * self.capture_interval - self._tolerance:
            if emitted >= max_emissions:
                break
            self._captured += 1
            emitted += 1
        return emitted


def constant_rate_frame_indices(frame_count: int, source_fps: float, target_fps: float) -> list[int]:
    """Select CFR source indices with the shared timestamp-capture policy.

    ComfyUI VIDEO components retain a nominal frame rate but not per-frame
    timestamps, so source frame ``i`` is placed at ``i / source_fps``.
    """
    source_fps = validate_frame_rate(source_fps, "source_fps")
    target_fps = validate_frame_rate(target_fps, "target_fps")
    cadence = FrameCaptureCadence(1.0 / target_fps)
    # A source interval holds at most ceil(target_fps / source_fps) + 1 capture points, so this cap never binds.
    max_emissions = math.ceil(target_fps / source_fps) + 1

    indices: list[int] = []
    for source_index in range(frame_count):
        emissions = cadence.emissions_at(source_index / source_fps, max_emissions)
        indices.extend([source_index] * emissions)
    return indices
