# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import av
import numpy as np
import io
import math
import re
import time
import torch
from typing import Callable, Optional, Union, Tuple
import random
from PIL import Image

import fastgen.utils.logging_utils as logger
from fastgen.datasets.video_resampling import FrameCaptureCadence, validate_frame_rate

# List of common video extensions PyAV might handle
VIDEO_EXTENSIONS = "mp4 ogv mjpeg avi mov h264 mpg webm wmv mkv".split()
# List of common image extensions PIL might handle
IMAGE_EXTENSIONS = "jpg jpeg png bmp tif tiff webp gif".split()


# ============================================================================
# Internal helpers (validation, stream metadata, frame/audio assembly)
# ============================================================================


def get_extension(key: str) -> str:
    return re.sub(r".*[.]", "", key).lower()


def _validate_video_decode_args(
    output_format: str,
    target_fps: Optional[float],
    num_frames: Optional[int] = None,
    min_frames: Optional[int] = None,
) -> None:
    """Raise ``ValueError`` on bad ``output_format``, ``target_fps``, ``num_frames``, or ``min_frames``.

    These are configuration errors, so callers check them before the key's extension and fail fast
    on every sample, not only on video keys.
    """
    if output_format not in ("numpy", "torch"):
        raise ValueError("output_format must be either 'numpy' or 'torch'")
    if target_fps is not None:
        validate_frame_rate(target_fps, "target_fps")
    if num_frames is not None and num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if min_frames is not None and min_frames <= 0:
        raise ValueError("min_frames must be a positive integer")


def _effective_fps(stream, target_fps: Optional[float] = None) -> float:
    """Effective output FPS — ``target_fps`` when set, else the stream's source rate.

    With ``target_fps=None`` (the default) this returns the source rate via
    ``stream.average_rate`` with ``guessed_rate`` fallback (``0.0`` if unknown);
    when ``target_fps`` is set it short-circuits without touching the stream.
    Call the source-rate branch while the parent ``av.open`` context is live.
    """
    if target_fps is not None:
        return float(target_fps)
    if stream.average_rate:
        return float(stream.average_rate)
    if stream.guessed_rate:
        return float(stream.guessed_rate)
    return 0.0


def _segment_source_frame_indices(
    frame_times: list, frame_ordinals: list, stream_start_sec: float, source_fps: float, key: str
) -> Optional[list]:
    """Source index of every kept segment frame, or ``None`` when the frames cannot be placed.

    The seek hides the frames before the segment, so the first kept frame is placed by its timestamp,
    ``round((t - stream_start) * source_fps)``, and the rest follow in decode order (``frame_ordinals``).
    That is exact only for constant-frame-rate video: when a kept frame's own timestamp disagrees with the
    placement (variable frame rate, or rate metadata such as a field-rate ``guessed_rate``) the result is
    ``None`` rather than wrong indices. A rate change entirely before the segment cannot be detected.
    """
    if source_fps <= 0:
        return None
    from_timestamps = [int(round((float(t) - stream_start_sec) * source_fps)) for t in frame_times]
    shift = from_timestamps[0] - frame_ordinals[0]
    if from_timestamps != [ordinal + shift for ordinal in frame_ordinals]:
        logger.warning(
            f"Key '{key}': frame timestamps do not follow the {source_fps:g} fps grid (variable frame rate?); "
            "cannot place the segment's frames in the source stream."
        )
        return None
    return from_timestamps


def _stream_duration_sec(stream) -> float:
    """Return the stream's duration in seconds, or ``0.0`` if duration/time_base are missing."""
    if stream.duration is None or stream.time_base is None:
        return 0.0
    return float(stream.duration * stream.time_base)


def _stream_start_sec(stream) -> float:
    """Stream start in seconds, or ``0.0`` if ``start_time``/``time_base`` are missing. Frame timestamps are
    absolute stream time, and streams often start after zero (mp4 edit lists, MPEG-TS)."""
    if stream.start_time is None or stream.time_base is None:
        return 0.0
    return float(stream.start_time * stream.time_base)


def _random_start_sec(total_duration_sec: float, estimated_needed_sec: float) -> float:
    """Random start offset that leaves room for a clip of ``estimated_needed_sec``.

    Returns ``0.0`` when the source isn't long enough for a random offset.
    """
    if total_duration_sec > estimated_needed_sec:
        return random.uniform(0.0, total_duration_sec - estimated_needed_sec)
    return 0.0


def _to_output(
    data: Union[list, np.ndarray], output_format: str, dtype: Optional[np.dtype] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Stack/forward an array and convert to the requested output format.

    Accepts a list (e.g. per-frame RGB ``uint8`` arrays — gets stacked) or an
    already-stacked ndarray (e.g. ``[channels, samples]`` audio). ``dtype`` (when
    set) enforces an explicit numeric contract before the optional torch wrap;
    leaving it ``None`` preserves the source dtype (uint8 video frames, etc.).
    """
    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return torch.from_numpy(arr) if output_format == "torch" else arr


def _downmix_to_stereo(audio: np.ndarray) -> np.ndarray:
    """Convert ``[C, samples]`` to stereo like LTX-2's ``process_videos.py``: mono is
    duplicated at unit gain, 5.1/7.1 use its normalized weights, other layouts are
    mean-downmixed. (swresample's own upmix would attenuate mono by 3 dB.)"""
    channels = audio.shape[0]
    if channels == 2:
        return audio
    if channels == 1:
        return np.repeat(audio, 2, axis=0)
    if channels == 6:  # [L, R, C, LFE, Ls, Rs]
        w_main, w_other = 1.0 / 2.414, 0.707 / 2.414
        left = w_main * audio[0] + w_other * audio[2] + w_other * audio[4]
        right = w_main * audio[1] + w_other * audio[2] + w_other * audio[5]
        return np.stack([left, right])
    if channels == 8:  # [L, R, C, LFE, Ls, Rs, Lb, Rb]
        w_main, w_other = 1.0 / 3.121, 0.707 / 3.121
        center = audio[2]
        left = w_main * audio[0] + w_other * (center + audio[4] + audio[6])
        right = w_main * audio[1] + w_other * (center + audio[5] + audio[7])
        return np.stack([left, right])
    logger.warning(f"Unknown audio channel layout ({channels} channels), using mean downmix")
    return np.repeat(audio.mean(axis=0, keepdims=True), 2, axis=0)


def _decode_aligned_audio(
    data: bytes,
    output_format: str,
    effective_fps: float,
    num_video_frames: int,
    frame_times: list,
    start_sec: Optional[float] = None,
) -> Tuple[Optional[Union[np.ndarray, torch.Tensor]], Optional[int]]:
    """Decode the container's audio track and slice/pad it to the video timeline, in ``output_format``.

    The slice starts at ``start_sec`` (stream time of the first kept video frame; ``None`` = the audio's
    first sample), measured from the audio's own origin (its first decoded frame's timestamp, else the
    stream start), with silence led in when the audio begins later. Its length is ``num_video_frames /
    effective_fps``, else the span of ``frame_times``, else the rest of the audio; short sources are
    zero-padded.

    Returns:
        ``(audio_out, audio_sr)``: both ``None`` without an audio track or on decode failure, ``audio_out``
        alone ``None`` when the target length is zero.
    """
    arrays: list = []
    sr: Optional[int] = None
    audio_start_sec = 0.0
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.audio:
                return None, None
            stream = container.streams.audio[0]
            sr = stream.rate or stream.codec_context.sample_rate
            audio_start_sec = _stream_start_sec(stream)
            # Keep the source layout; ``_downmix_to_stereo`` applies the reference channel handling.
            resampler = av.AudioResampler(format="fltp", layout=None, rate=sr)
            for index, frame in enumerate(container.decode(stream)):
                if index == 0 and frame.time is not None:
                    audio_start_sec = float(frame.time)  # sample 0 of the decoded audio sits here
                for resampled in resampler.resample(frame):
                    arrays.append(resampled.to_ndarray())
            for resampled in resampler.resample(None):
                arrays.append(resampled.to_ndarray())
    except Exception:
        return None, None
    if not arrays or sr is None:
        return None, None
    full_audio = _downmix_to_stereo(np.concatenate(arrays, axis=1))  # [2, total_samples]

    # Offset of the first kept video frame into the decoded audio; a negative offset means the audio
    # starts later than the video, so lead in with silence.
    start_sample = 0 if start_sec is None else int(round((start_sec - audio_start_sec) * sr))
    if start_sample < 0:
        full_audio = np.pad(full_audio, ((0, 0), (-start_sample, 0)))
        start_sample = 0

    # Target length in samples: prefer FPS-derived, fall back to the captured frame-timestamp span, then
    # to the full audio past the start.
    if effective_fps > 0:
        # Multiply before dividing so integral sample counts survive float rounding
        # (201 frames at 25 fps: 128640 samples, not ``int(8.04 * 16000)`` = 128639).
        target_samples = int(num_video_frames * sr / effective_fps)
    elif len(frame_times) >= 2:
        target_samples = int(max(0.0, float(frame_times[-1] - frame_times[0])) * sr)
    else:
        target_samples = full_audio.shape[-1] - start_sample
    if target_samples <= 0:
        return None, sr

    # Slice at the start for ``target_samples``; zero-pad when the source can't supply enough samples.
    sliced = full_audio[..., start_sample : start_sample + target_samples]
    if sliced.shape[-1] < target_samples:
        sliced = np.pad(sliced, ((0, 0), (0, target_samples - sliced.shape[-1])))

    return _to_output(sliced, output_format, dtype=np.float32), sr


def _collect_video_frames(
    container,
    stream,
    target_interval: Optional[float],
    start_sec: Optional[float] = None,
    next_capture: Optional[float] = None,
    max_total_frames: Optional[int] = None,
    max_dup_per_frame: Optional[int] = None,
    timeout_s: Optional[float] = None,
    key: str = "",
) -> Tuple[list, list, list]:
    """Iterate ``stream``'s frames, returning RGB arrays with optional cadence densification.

    Two modes, parameterised so the four ``decode_video_*`` entry points share
    one inner loop:

    - ``target_interval is None``: keep every decoded source frame as-is.
    - ``target_interval`` set: emit the current frame each time its timestamp reaches the next capture
      point ``anchor + k * target_interval`` (duplicating it when the source is slower), with a
      ``1e-3 * target_interval`` tolerance for float rounding. The anchor is ``next_capture`` (segment
      callers pass their seek target) or, when ``None``, the first frame's timestamp.

    ``frame_times`` is always returned; callers that don't need it discard via
    ``frames, _, _ = _collect_video_frames(...)``. Entries are skipped (not
    ``None``-padded) when the source frame has no timestamp, so the list may be
    shorter than ``frames`` only when *both* ``target_interval`` and
    ``start_sec`` are ``None`` (the full-video, no-target-fps path) — every
    other path requires ``t`` and the lists stay aligned 1:1.

    Args:
        start_sec: When set, skip frames with ``t < start_sec`` and require
            ``t`` to be non-``None`` everywhere (segment seek alignment). When
            ``None`` (full-video behaviour) frames with missing timestamps are
            kept in the no-target-interval branch and skipped in the
            target-interval branch (cadence needs ``t``).
        max_total_frames: Stop once ``len(frames)`` reaches this cap. Used by
            segment callers to collect exactly ``num_frames``.
        max_dup_per_frame: Per-source-frame duplication cap, only consulted
            when ``target_interval`` is set. Used by full-video callers to
            bound densification when ``target_fps > source_fps``.
        timeout_s: Wall-clock budget for the whole loop; checked after each
            source frame. Used by ``decode_video_full``.
        key: For logging context.

    ``frame.to_ndarray`` failures are always logged and skipped so a single
    corrupted source frame can't drop the whole sample; container-level errors
    still resurface from ``container.decode(stream)`` and are caught by the
    caller's outer ``except``.

    Returns:
        ``(frames, frame_times, frame_ordinals)``. ``frame_times`` never contains ``None``. ``frame_ordinals``
        is each kept frame's position in decode order since open/seek (duplicates repeat it, frames without
        timestamps count), which is the source frame index when decoding starts at the stream's first frame.
    """
    frames: list = []
    frame_times: list = []
    frame_ordinals: list = []
    loop_start = time.time() if timeout_s is not None else None
    cadence = FrameCaptureCadence(target_interval, anchor=next_capture) if target_interval is not None else None

    ordinal = -1  # decode-order position of ``frame``, skipped frames included
    for frame in container.decode(stream):
        if frame is None:
            continue
        ordinal += 1
        # Frame timestamp (seconds): prefer ``frame.time``, fall back to
        # ``frame.pts * time_base`` (some PyAV builds leave ``frame.time`` unset
        # on otherwise-valid frames). ``t`` is required for cadence emission
        # and for the ``start_sec`` filter; only the full-video no-target-fps
        # path tolerates missing timestamps.
        if frame.time is not None:
            t: Optional[float] = float(frame.time)
        elif frame.pts is not None and stream.time_base is not None:
            t = float(frame.pts * stream.time_base)
        else:
            t = None
        if t is None and (target_interval is not None or start_sec is not None):
            continue
        if start_sec is not None and t is not None and t < start_sec:
            continue

        try:
            frame_array = frame.to_ndarray(format="rgb24")
        except Exception as decode_err:  # pylint: disable=broad-except
            at = f" at time {t:.2f}s" if t is not None else ""
            logger.warning(f"Key '{key}': Error decoding frame{at}: {decode_err}. Skipping frame.")
            continue

        if target_interval is None:
            frames.append(frame_array)
            frame_ordinals.append(ordinal)
            if t is not None:
                frame_times.append(t)
        else:
            assert cadence is not None and t is not None
            max_emissions = max_dup_per_frame
            if max_total_frames is not None:
                remaining = max_total_frames - len(frames)
                max_emissions = remaining if max_emissions is None else min(max_emissions, remaining)
            assert max_emissions is not None, "target_interval needs max_dup_per_frame or max_total_frames"
            emissions = cadence.emissions_at(t, max_emissions)
            frames.extend([frame_array] * emissions)
            frame_times.extend([t] * emissions)
            frame_ordinals.extend([ordinal] * emissions)

        if max_total_frames is not None and len(frames) >= max_total_frames:
            break
        if loop_start is not None and (time.time() - loop_start) > timeout_s:
            logger.warning(
                f"Key '{key}': stopped decoding after {timeout_s}s with {len(frames)} frames collected; the "
                "remaining frames are dropped (raise timeout_s to decode the whole stream)."
            )
            break

    return frames, frame_times, frame_ordinals


def _decode_video_segment_frames(
    key: str,
    data: bytes,
    num_frames: int,
    target_fps: Optional[float],
    output_format: str,
) -> Optional[Tuple[Union[np.ndarray, torch.Tensor], list, float, dict]]:
    """Open the container, seek to a random segment start, collect up to
    ``num_frames`` frames, and convert them to ``output_format``.

    Shared preamble for the segment decoders (``decode_video_segment`` and
    ``decode_video_audio_segment``): both need a backward-seekable container,
    a defined ``time_base``, the same effective-FPS / duration / start-sec
    derivation, the same ``_collect_video_frames`` call, and the same
    output-format conversion.

    Returns:
        ``(video_out, frame_times, effective_fps, provenance)`` with exactly
        ``num_frames`` frames stacked as ``output_format``, or ``None`` on
        non-video extension, container-level failure, empty collection, short
        collection, or output-conversion failure. Raises ``ValueError`` on bad
        configuration (``output_format``, ``target_fps``, ``num_frames``); all
        other failure cases are logged so callers can stay slim. Segments are
        treated as all-or-nothing (no partial-frame fallback) since downstream
        consumers expect a fixed ``num_frames`` shape. ``provenance`` is the
        ``{"segment_start_frame", "source_frame_indices"}`` dict described in ``decode_video_segment``.
    """
    _validate_video_decode_args(output_format, target_fps, num_frames=num_frames)
    if get_extension(key) not in VIDEO_EXTENSIONS:
        return None

    target_interval = 1.0 / float(target_fps) if target_fps is not None else None

    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.video:
                logger.warning(f"Key '{key}': No video stream found.")
                return None
            video = container.streams.video[0]
            if video.time_base is None:
                logger.warning(f"Key '{key}': Video stream has no time_base. Cannot process.")
                return None

            effective_fps = _effective_fps(video, target_fps)
            source_fps = _effective_fps(video)
            stream_start_sec = _stream_start_sec(video)
            total_duration_sec = _stream_duration_sec(video)

            # Random offset into the clip that leaves room for the segment, placed on the stream's absolute
            # time axis (``seek`` and frame timestamps include the stream start). With ``target_fps`` the last
            # capture at ``start + (num_frames - 1) / target_fps`` needs a source frame at or after it, and the
            # last source frame sits ``1 / source_fps`` before the end; else source-rate with a +1 frame buffer
            # for avg-FPS noise, else 0.
            if target_fps is not None:
                margin_sec = 1.0 / source_fps if source_fps > 0 else 1.0 / float(target_fps)
                needed_sec = (num_frames - 1) / float(target_fps) + margin_sec
                start_offset_sec = _random_start_sec(total_duration_sec, needed_sec)
            elif effective_fps > 0:
                start_offset_sec = _random_start_sec(total_duration_sec, (num_frames + 1) / effective_fps)
            else:
                logger.warning(f"Key '{key}': Cannot determine FPS. Reading from start.")
                start_offset_sec = 0.0
            start_sec = stream_start_sec + start_offset_sec

            # Best-effort backward seek to ``start_sec``. On failure (unseekable
            # stream, bad pts, version-specific PyAV exceptions) reset to the
            # stream start; unrecoverable container-level errors will
            # resurface from ``container.decode(...)`` below and be caught by
            # the outer ``except``.
            try:
                container.seek(int(start_sec / video.time_base), stream=video, backward=True, any_frame=False)
                next_capture: Optional[float] = float(start_sec) if target_interval is not None else None
            except Exception as seek_err:  # pylint: disable=broad-except
                logger.warning(
                    f"Key '{key}': Seek to {start_sec:.2f}s failed ({seek_err}). Attempting decode from beginning."
                )
                start_sec = stream_start_sec
                next_capture = stream_start_sec if target_interval is not None else None

            frames, frame_times, frame_ordinals = _collect_video_frames(
                container,
                video,
                target_interval,
                start_sec=start_sec,
                next_capture=next_capture,
                max_total_frames=num_frames,
                key=key,
            )
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Key '{key}': Error processing video with PyAV: {e}")
        return None

    if len(frames) == 0:
        logger.warning(f"Key '{key}': No frames collected after start time {start_sec:.2f}s.")
        return None
    if len(frames) < num_frames:
        logger.warning(f"Key '{key}': Collected only {len(frames)} frames, less than requested {num_frames}.")
        return None

    try:
        video_out = _to_output(frames, output_format)
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Key '{key}': Error converting collected frames to array: {e}")
        return None

    # Both indices count from the stream's first frame. ``frame_times`` is 1:1 with ``frames`` here because
    # the ``start_sec`` filter requires a timestamp.
    first_kept_sec = float(frame_times[0]) - stream_start_sec
    provenance = {
        "segment_start_frame": int(round(first_kept_sec * effective_fps)) if effective_fps > 0 else 0,
        "source_frame_indices": _segment_source_frame_indices(
            frame_times, frame_ordinals, stream_start_sec, source_fps, key
        ),
    }
    return video_out, frame_times, effective_fps, provenance


def _decode_video_full_frames(
    key: str,
    data: bytes,
    output_format: str,
    target_fps: Optional[float] = None,
    probesize: Optional[str] = None,
    analyzeduration_us: Optional[int] = None,
    timeout_s: Optional[float] = None,
    min_frames: Optional[int] = None,
) -> Optional[Tuple[Union[np.ndarray, torch.Tensor], list, float, list]]:
    """Open the container, collect every decodable frame within ``timeout_s``,
    and convert them to ``output_format``.

    Shared preamble for the full-stream decoders (``decode_video_full`` and
    ``decode_video_audio_full``): same ``av.open``, same ``target_interval`` /
    ``max_dup_per_frame`` derivation, same ``_collect_video_frames`` call, and
    same output-format conversion. The ``probesize`` / ``analyzeduration_us``
    / ``timeout_s`` / ``min_frames`` options are only consumed by
    ``decode_video_full``; passing ``None`` (default) leaves the corresponding
    behaviour unset.

    Returns:
        ``(video_out, frame_times, effective_fps, source_frame_indices)`` on success, or ``None`` on
        non-video extension, container-level failure, empty collection
        (no decodable frames, or ``timeout_s`` elapsed before a frame was kept), or
        output-conversion failure. Raises ``ValueError`` on bad configuration
        (``output_format``, ``target_fps``, ``min_frames``); all other failure
        cases are logged so callers can stay slim. ``source_frame_indices`` is each kept frame's
        position in decode order, i.e. its source index since decoding starts at the stream's first frame.
    """
    _validate_video_decode_args(output_format, target_fps, min_frames=min_frames)
    if get_extension(key) not in VIDEO_EXTENSIONS:
        return None

    options: dict = {}
    if probesize is not None:
        options["probesize"] = probesize
    if analyzeduration_us is not None:
        options["analyzeduration"] = str(analyzeduration_us)

    target_interval: Optional[float] = None
    max_dup_per_frame: Optional[int] = None
    if target_fps is not None:
        target_interval = 1.0 / float(target_fps)
        # Cap per-source-frame duplications to ``ceil(target_fps) + 1``. When
        # ``target_fps > source_fps`` this fills in the cadence gap; keeping
        # the cap tied to raw ``target_fps`` (not the min_frames-clamped
        # interval) keeps the densification bounded to what playback at
        # ``target_fps`` would imply.
        max_dup_per_frame = math.ceil(float(target_fps)) + 1

    try:
        with av.open(io.BytesIO(data), mode="r", options=options) as container:
            if not container.streams.video:
                return None
            video = container.streams.video[0]
            effective_fps = _effective_fps(video, target_fps)

            if target_interval is not None and min_frames is not None:
                # Lower-bound the kept count over the clip's duration —
                # i.e., effective_fps = max(target_fps, min_frames / duration).
                # Skipped when duration metadata is missing (clip falls through
                # with raw target_fps).
                duration_s = _stream_duration_sec(video)
                if duration_s > 0:
                    target_interval = min(target_interval, duration_s / min_frames)

            frames, frame_times, frame_ordinals = _collect_video_frames(
                container,
                video,
                target_interval,
                max_dup_per_frame=max_dup_per_frame,
                timeout_s=timeout_s,
                key=key,
            )
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Key '{key}': Error processing video with PyAV: {e}")
        return None

    if len(frames) == 0:
        suffix = f" within {timeout_s}s" if timeout_s is not None else ""
        logger.warning(f"Key '{key}': No frames collected{suffix}.")
        return None

    try:
        video_out = _to_output(frames, output_format)
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Key '{key}': Error converting collected frames to array: {e}")
        return None

    return video_out, frame_times, effective_fps, frame_ordinals


# ============================================================================
# Video decoders
# ============================================================================


def decode_video_full(
    key: str,
    data: bytes,
    output_format: str = "torch",
    target_fps: Optional[float] = None,
    probesize: str = "1M",
    analyzeduration_us: int = 2_000_000,
    timeout_s: Optional[float] = 3.0,
    min_frames: Optional[int] = None,
    return_source_frame_indices: bool = False,
) -> Optional[Union[np.ndarray, torch.Tensor, dict]]:
    """Decode video bytes to ``(T, H, W, 3)`` RGB frames, or ``None`` on failure.

    ``target_fps`` resamples via timestamp-based selection: each source frame
    whose timestamp (``frame.time``, or ``frame.pts * time_base`` when
    ``frame.time`` is unset) has crossed ``next_capture`` is emitted, and
    (when ``target_fps > source_fps``) duplicated up to ``ceil(target_fps) + 1``
    times so the kept stream maintains a constant cadence. This preserves
    wall-clock duration for consumers that re-encode at ``target_fps``.

    ``min_frames`` clamps the interval to ``duration / min_frames`` so
    an aggressive ``target_fps`` doesn't starve fixed-cardinality
    consumers; ignored without ``target_fps`` or duration metadata. The
    per-frame duplication cap stays tied to the raw ``target_fps`` so a
    ``min_frames`` request larger than the source can physically support
    is best-effort, not unbounded.

    Args:
        key (str): The key (filename/extension) associated with the data.
        data (bytes): The video data as bytes.
        output_format (str): output format of the video frames.
        target_fps (float | None): Optional target FPS for timestamp-based resampling.
        probesize (str): probesize for ffmpeg.
        analyzeduration_us (int): analyzeduration for ffmpeg.
        timeout_s (float | None): Wall-clock budget in seconds for decoding, checked after each source frame; the
            frames decoded by then are returned and the rest of the clip is dropped. ``None`` decodes the whole
            stream.
        min_frames (int | None): Floor on kept frames over the clip duration.
        return_source_frame_indices (bool): When True, return ``{"frames": ..., "source_frame_indices":
            list[int]}`` with each kept frame's index in the source stream (exact, taken from decode order).
            Defaults to False (bare frames tensor).

    Returns:
        np.ndarray | torch.Tensor | dict | None: video frames array (T, H, W, 3) RGB (or the dict above), or
            None on failure.
    """
    result = _decode_video_full_frames(
        key,
        data,
        output_format,
        target_fps,
        probesize=probesize,
        analyzeduration_us=analyzeduration_us,
        timeout_s=timeout_s,
        min_frames=min_frames,
    )
    if result is None:
        return None
    video_out, _, _, source_frame_indices = result
    if return_source_frame_indices:
        return {"frames": video_out, "source_frame_indices": source_frame_indices}
    return video_out


def decode_video_segment(
    key: str,
    data: bytes,
    num_frames: int,
    output_format: str = "torch",
    target_fps: Optional[float] = None,
    return_segment_start_frame: bool = False,
) -> Optional[Union[np.ndarray, torch.Tensor, dict]]:
    """Decode a randomly selected segment of exactly ``num_frames`` from a video.

    Selects a random start time such that a clip of ``num_frames`` can be extracted.
    If ``target_fps`` is set, the segment duration is ``num_frames / target_fps`` and
    frames are sampled by timestamp. Returns ``None`` (not a partial sample) when the
    source can't supply ``num_frames`` frames past the chosen start — segments are
    all-or-nothing so consumers can rely on a fixed ``(num_frames, H, W, 3)`` shape.

    Args:
        key (str): The key (filename/extension) associated with the data.
        data (bytes): The video data as bytes.
        num_frames (int): The target number of frames to decode.
        output_format (str): The desired output format ('numpy' or 'torch').
                             Defaults to 'torch'.
        target_fps (float | None): Optional target FPS for timestamp-based
                                   resampling.
        return_segment_start_frame (bool): When True, return
            ``{"frames": ..., "segment_start_frame": int, "source_frame_indices": list[int] | None}``
            instead of a bare frames tensor. Both count from the stream's first frame:
            ``segment_start_frame`` is the first kept frame's index at the output rate, and
            ``source_frame_indices`` each kept frame's index in the source stream, exact even when
            ``target_fps`` drops or duplicates frames, or ``None`` when the source rate is unknown or the
            timestamps do not follow it (variable frame rate; placement after a seek needs a constant
            rate). Defaults to False (bare frames tensor).

    Returns:
        np.ndarray | torch.Tensor | dict | None: ``(num_frames, H, W, 3)`` RGB frames (or the dict above),
            or ``None`` on container failure, empty collection, or short collection.
    """
    result = _decode_video_segment_frames(key, data, num_frames, target_fps, output_format)
    if result is None:
        return None
    video_out, _, _, provenance = result
    if return_segment_start_frame:
        return {"frames": video_out, **provenance}
    return video_out


# ============================================================================
# Video + audio decoders
# ============================================================================


def decode_video_audio_segment(
    key: str,
    data: bytes,
    num_frames: int,
    output_format: str = "torch",
    target_fps: Optional[float] = None,
) -> Optional[dict]:
    """Decode a video+audio segment from the same container.

    Video: random-segment of exactly ``num_frames`` frames (same all-or-nothing
    contract as ``decode_video_segment`` — returns ``None`` when the source
    can't supply ``num_frames``). If ``target_fps`` is set, frames are sampled
    by timestamp at ``1 / target_fps`` intervals and the segment duration is
    ``num_frames / target_fps``.
    Audio: the full track is decoded once, then sliced from the first kept
    frame's stream time to the exact video segment duration — mirroring the LTX-2 reference
    ``_extract_audio(video_path, target_duration)`` pattern. ``audio`` is
    ``None`` when the container has no audio track or audio decode fails;
    ``video`` is never ``None`` in a returned dict.

    Args:
        key (str): The key (filename/extension) associated with the data.
        data (bytes): The container bytes.
        num_frames (int): The target number of video frames to decode.
        output_format (str): ``"numpy"`` or ``"torch"``. Defaults to ``"torch"``.
        target_fps (float | None): Optional target FPS for timestamp-based
            resampling. When set, the audio slice length and the returned
            ``video_fps`` use ``target_fps`` rather than the source FPS,
            keeping audio/video aligned at the resampled rate.

    Returns:
        dict with ``video`` [T,H,W,C], ``audio`` [2, samples] or None,
        ``audio_sr``, ``video_fps``, and ``segment_start_frame`` as in
        ``decode_video_segment``.
    """
    result = _decode_video_segment_frames(key, data, num_frames, target_fps, output_format)
    if result is None:
        return None
    video_out, frame_times, effective_fps, provenance = result

    # Anchor the audio at the first kept frame, which can sit up to one source interval after the seek target.
    audio_out, audio_sr = _decode_aligned_audio(
        data, output_format, effective_fps, len(video_out), frame_times, start_sec=frame_times[0]
    )
    return {
        "video": video_out,
        "audio": audio_out,
        "audio_sr": audio_sr,
        "video_fps": effective_fps,
        "segment_start_frame": provenance["segment_start_frame"],
    }


def decode_video_audio_full(
    key: str,
    data: bytes,
    output_format: str = "torch",
    target_fps: Optional[float] = None,
) -> Optional[dict]:
    """Decode a full video+audio from the same container.

    If ``target_fps`` is set, frames are sampled by timestamp at
    ``1 / target_fps`` intervals (mirroring ``decode_video_full``),
    and audio is trimmed/padded to ``len(output_frames) / target_fps``
    so the returned video+audio remain length-consistent at the
    resampled rate.

    Args:
        key (str): The key (filename/extension) associated with the data.
        data (bytes): The container bytes.
        output_format (str): ``"numpy"`` or ``"torch"``. Defaults to ``"torch"``.
        target_fps (float | None): Optional target FPS for timestamp-based
            resampling. When set, the audio slice length and the returned
            ``video_fps`` use ``target_fps`` rather than the source FPS.

    Returns:
        dict with ``video`` [T,H,W,C], ``audio`` [2, samples] or None,
        ``audio_sr``, ``video_fps``.
    """
    result = _decode_video_full_frames(key, data, output_format, target_fps)
    if result is None:
        return None
    video_out, frame_times, effective_fps, _ = result

    # Anchor the audio at the first frame's time when every kept frame has one (without ``target_fps``, frames
    # lacking timestamps are kept but absent from ``frame_times``); otherwise use the audio's own origin.
    first_frame_sec = frame_times[0] if len(frame_times) == len(video_out) else None
    audio_out, audio_sr = _decode_aligned_audio(
        data, output_format, effective_fps, len(video_out), frame_times, start_sec=first_frame_sec
    )
    return {"video": video_out, "audio": audio_out, "audio_sr": audio_sr, "video_fps": effective_fps}


# ============================================================================
# Image / text / array decoders
# ============================================================================


def decode_image(
    key: str, data: bytes, image_transform: Callable[[Image.Image], torch.Tensor] | None = None
) -> Optional[Union[Image.Image, torch.Tensor]]:
    """Decode image bytes (any extension in ``IMAGE_EXTENSIONS``) to a PIL image,
    or to the ``image_transform`` output (e.g. a normalized ``torch.Tensor``) when
    a transform is provided.

    Returns ``None`` for non-image extensions so the webdataset pipeline can
    fall through to the next decoder.
    """
    if get_extension(key) in IMAGE_EXTENSIONS:
        image = Image.open(io.BytesIO(data))
        if image_transform is not None:
            image = image_transform(image)
        return image
    return None


def decode_txt(key: str, data: bytes, txt_extensions: Tuple[str, ...] = ("txt",), strip: bool = True) -> Optional[str]:
    extension = get_extension(key)
    if extension in txt_extensions:
        txt = data.decode("utf-8")
        if strip:
            txt = txt.strip()
        return txt
    return None


def decode_npy_torch(key: str, data: bytes) -> Optional[torch.Tensor]:
    """Decode .npy bytes into a torch tensor."""
    if get_extension(key) == "npy":
        array = np.load(io.BytesIO(data))
        return torch.from_numpy(array)
    return None


def decode_npz_torch(key: str, data: bytes) -> Optional[dict]:
    """Decode .npz bytes into a dict of torch tensors."""
    if get_extension(key) == "npz":
        arrays = dict(np.load(io.BytesIO(data)))
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in arrays.items()}
    return None
