# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``fastgen.datasets.decoders`` and ``fastgen.datasets.video_resampling``.

Bulk of the suite focuses on ``decode_video_full`` (the ``target_fps``
timestamp-based subsample + ``min_frames`` interval clamp wired up by
``vlm_judge`` / ``VLMLogger``). Smaller blocks cover the shared
``FrameCaptureCadence`` behind that subsample, ``decode_video_segment``
and the lighter-weight image / text / npy / npz decoders for completeness.

Mp4 fixtures come from ``save_video`` (libx264, the encoder behind the
mp4s of the training callbacks) or are muxed directly with PyAV, wav
fixtures with ``wave``; the fake-container tests need no file.
"""

import io
import math
from fractions import Fraction
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize

from fastgen.datasets.decoders import (
    decode_image,
    decode_npy_torch,
    decode_npz_torch,
    decode_txt,
    decode_video_audio_full,
    decode_video_full,
    decode_video_segment,
    get_extension,
)
from fastgen.datasets.video_resampling import FrameCaptureCadence, constant_rate_frame_indices
from fastgen.utils.media_utils import save_video


def _write_mp4(tmp_path: Path, n_frames: int, fps: int, hw: int = 32) -> Path:
    """Build a deterministic ``[3, T, hw, hw]`` clip in ``[-1, 1]`` float
    and save it via the project's standard ``save_video`` (libx264)."""
    v = torch.linspace(-1.0, 1.0, n_frames).view(1, n_frames, 1, 1).expand(3, n_frames, hw, hw).contiguous()
    out = tmp_path / f"clip_{n_frames}f_{fps}fps.mp4"
    save_video(v, str(out), vid_format="mp4", fps=fps)
    return out


# ============================================================================
# Constant-frame mp4 fixtures used by ``target_fps`` resampling tests.
#
# Each frame is filled with a single scalar so that the per-frame mean uniquely
# identifies which source frame ended up at each output position. That lets us
# verify the exact subsample chosen by the timestamp-based target_fps loop in
# the decoders, not just the output shape.
# ============================================================================


def _make_constant_video_frames(values: np.ndarray, height: int = 16, width: int = 16) -> np.ndarray:
    """Build ``[T, H, W, 3]`` uint8 RGB frames where frame ``t`` is filled with ``values[t]``."""
    frames = np.repeat(values[:, None, None, None], height, axis=1)
    frames = np.repeat(frames, width, axis=2)
    return np.repeat(frames, 3, axis=3)


def _pcm16_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Write ``[C, N]`` floats in [-1, 1] as a 16-bit PCM WAV (lossless, layout kept as stored)."""
    import wave

    pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(pcm.shape[0])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.ascontiguousarray(pcm.T).tobytes())
    return buf.getvalue()


def test_decode_aligned_audio_mirrors_reference_channel_handling():
    """Mono is duplicated at unit gain and 5.1 uses LTX-2's normalized downmix, not swresample's."""
    from fastgen.datasets.decoders import _decode_aligned_audio

    sr = 16000
    mono = np.full((1, sr), 0.5, dtype=np.float32)
    audio, out_sr = _decode_aligned_audio(_pcm16_wav_bytes(mono, sr), "torch", 1.0, 1, [])
    assert out_sr == sr and tuple(audio.shape) == (2, sr)
    torch.testing.assert_close(audio, torch.full((2, sr), 0.5), atol=1e-4, rtol=0)

    surround = np.zeros((6, sr), dtype=np.float32)  # [L, R, C, LFE, Ls, Rs] impulses of 0.5
    for channel in range(6):
        surround[channel, 100 + channel] = 0.5
    audio, _ = _decode_aligned_audio(_pcm16_wav_bytes(surround, sr), "torch", 1.0, 1, [])
    w_main, w_other = 0.5 / 2.414, 0.5 * 0.707 / 2.414
    expected_left = [w_main, 0.0, w_other, 0.0, w_other, 0.0]
    expected_right = [0.0, w_main, w_other, 0.0, 0.0, w_other]
    for channel in range(6):
        assert abs(audio[0, 100 + channel].item() - expected_left[channel]) < 1e-3
        assert abs(audio[1, 100 + channel].item() - expected_right[channel]) < 1e-3


def test_downmix_to_stereo_follows_the_reference_for_every_layout():
    """7.1 uses LTX-2's normalized weights (sides and backs alike, LFE dropped), layouts the reference does
    not know (quad here) are mean-downmixed into both channels, and stereo passes through untouched."""
    from fastgen.datasets.decoders import _downmix_to_stereo

    impulses = np.eye(8, dtype=np.float32)  # channel c = [L, R, C, LFE, Ls, Rs, Lb, Rb][c] pulses at sample c
    out = _downmix_to_stereo(impulses)
    w_main, w_other = 1.0 / 3.121, 0.707 / 3.121
    np.testing.assert_allclose(out[0], [w_main, 0.0, w_other, 0.0, w_other, 0.0, w_other, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[1], [0.0, w_main, w_other, 0.0, 0.0, w_other, 0.0, w_other], atol=1e-6)

    quad = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(_downmix_to_stereo(quad), np.repeat(quad.mean(axis=0, keepdims=True), 2, axis=0))

    stereo = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert _downmix_to_stereo(stereo) is stereo


def _encode_mov(
    frames: Optional[np.ndarray] = None,
    fps: int = 16,
    video_pts_offset: int = 0,
    audio: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    audio_pts_offset: int = 0,
) -> bytes:
    """PyAV-mux constant-value RGB frames (h264) and/or ``[1, N]`` mono floats (16-bit PCM) into one mov.

    A ``*_pts_offset`` delays that track by as many frames / samples (an edit-list delay), so tracks can
    start after zero and at different times.
    """
    import av

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mov") as container:
        video = sound = None  # all streams must exist before the first packet is muxed
        if frames is not None:
            video = container.add_stream("h264", rate=fps)
            video.width, video.height = frames.shape[2], frames.shape[1]
            video.pix_fmt = "yuv420p"
            video.codec_context.time_base = Fraction(1, fps)
        if audio is not None:
            sound = container.add_stream("pcm_s16le", rate=sample_rate)
            sound.layout = "mono"
            sound.codec_context.time_base = Fraction(1, sample_rate)
        if video is not None:
            for index, frame_data in enumerate(frames):
                frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                frame.pts = index + video_pts_offset
                frame.time_base = video.codec_context.time_base
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
        if sound is not None:
            pcm = np.clip(np.round(audio * 32768.0), -32768, 32767).astype("<i2")
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm), format="s16", layout="mono")
            frame.sample_rate = sample_rate
            frame.pts = audio_pts_offset
            frame.time_base = sound.codec_context.time_base
            for packet in sound.encode(frame):
                container.mux(packet)
            for packet in sound.encode():
                container.mux(packet)
    return buf.getvalue()


def test_decode_aligned_audio_subtracts_the_audio_streams_origin():
    """``start_sec`` is the first kept video frame's stream time. Audio that starts at 0.5 s must be sliced
    relative to that origin, and audio starting after the video frame is led in with silence."""
    from fastgen.datasets.decoders import _decode_aligned_audio

    sr = 16000
    ramp = np.linspace(0.0, 1.0, 2 * sr, endpoint=False, dtype=np.float32)[None]  # value = elapsed / 2 s
    mov = _encode_mov(audio=ramp, sample_rate=sr, audio_pts_offset=sr // 2)

    audio, out_sr = _decode_aligned_audio(mov, "torch", 4.0, 4, [], start_sec=0.75)  # 0.25 s into the audio
    assert out_sr == sr and tuple(audio.shape) == (2, sr)
    assert abs(audio[0, 0].item() - 0.125) < 2e-3 and abs(audio[0, -1].item() - 0.625) < 2e-3

    audio, _ = _decode_aligned_audio(mov, "torch", 4.0, 4, [], start_sec=0.25)  # 0.25 s before the audio
    assert torch.all(audio[:, : sr // 4] == 0)
    assert abs(audio[0, sr // 4].item()) < 2e-3 and abs(audio[0, -1].item() - 0.375) < 2e-3


def test_decode_aligned_audio_uses_an_exact_sample_count_for_integral_durations():
    """201 frames at 25 fps is exactly 128640 samples at 16 kHz; ``int(8.04 * 16000)`` would drop one."""
    from fastgen.datasets.decoders import _decode_aligned_audio

    sr = 16000
    silence = np.zeros((1, 9 * sr), dtype=np.float32)
    audio, _ = _decode_aligned_audio(_pcm16_wav_bytes(silence, sr), "torch", 25.0, 201, [])
    assert audio.shape[-1] == 128640


def _encode_constant_mp4(frames: np.ndarray, fps: int, pts_offset: int = 0) -> bytes:
    """PyAV-encode constant-value RGB frames to an in-memory mp4 (h264, yuv420p).

    ``pts_offset`` delays every timestamp by that many frames, so the stream's first frame sits at
    ``pts_offset / fps`` seconds (an mp4 edit-list delay) instead of at zero.
    """
    import av

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=fps)
    stream.width, stream.height = frames.shape[2], frames.shape[1]
    stream.pix_fmt = "yuv420p"
    stream.codec_context.time_base = Fraction(1, fps)
    for index, frame_data in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame.pts = index + pts_offset
        frame.time_base = stream.codec_context.time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buf.getvalue()


def _expected_target_fps_indices(
    source_fps: int,
    target_fps: int,
    num_frames: int,
    num_source_frames: int,
) -> List[int]:
    """Source index of each frame a ``target_fps`` decoder keeps from a constant-rate source starting at zero:
    the first source frame at or after every capture point ``k / target_fps``, repeated while the source is
    slower than the target. Exact rational arithmetic, so the expectation is independent of the float
    rounding the decoders have to tolerate (an accumulating float grid put frame 13 where 12 belongs)."""
    expected_indices: List[int] = []
    for capture in range(num_frames):
        source_index = math.ceil(Fraction(capture, target_fps) * source_fps)
        if source_index >= num_source_frames:
            break
        expected_indices.append(source_index)
    return expected_indices


def _assert_frame_means_match_values(
    decoded: torch.Tensor,
    values: np.ndarray,
    expected_indices: List[int],
    atol: float = 3.0,
) -> None:
    """Check decoded frame identities via per-frame scalar means. libx264 shifts a constant frame's mean by
    about 1 while adjacent fixture values differ by at least ``240 / 59``, so ``atol`` stays below that step
    and an off-by-one frame is caught."""
    frame_means = decoded.float().mean(dim=(1, 2, 3))
    expected_means = torch.tensor(values[expected_indices], dtype=frame_means.dtype)
    torch.testing.assert_close(frame_means, expected_means, rtol=0, atol=atol)


def test_no_target_fps_keeps_all_frames(tmp_path: Path):
    src = _write_mp4(tmp_path, n_frames=16, fps=16)
    arr = decode_video_full(src.name, src.read_bytes())
    assert arr is not None
    assert isinstance(arr, torch.Tensor)
    assert arr.shape == (16, 32, 32, 3)


def test_mkv_full_and_segment_decoding_preserve_remuxed_frames(tmp_path, monkeypatch):
    """MKV accepted by the VLM/media entry points must reach the shared video decoders."""
    import av
    import fastgen.datasets.decoders as decoders_module

    source = _write_mp4(tmp_path, n_frames=16, fps=16)
    remuxed = source.with_suffix(".mkv")
    with av.open(str(source)) as original, av.open(str(remuxed), "w") as output:
        stream = original.streams.video[0]
        target = output.add_stream_from_template(stream)
        for packet in original.demux(stream):
            if packet.dts is not None:
                packet.stream = target
                output.mux(packet)

    expected = decode_video_full(source.name, source.read_bytes())
    payload = remuxed.read_bytes()
    assert torch.equal(decode_video_full(remuxed.name, payload), expected)
    monkeypatch.setattr(decoders_module, "_random_start_sec", lambda total, needed: 0.0)
    assert torch.equal(decode_video_segment(remuxed.name, payload, 4), expected[:4])


def test_target_fps_equal_to_source_fps_keeps_every_frame(tmp_path: Path, monkeypatch):
    """Resampling onto the source rate is the identity. Accumulated float capture points one ulp above a
    source timestamp used to drop frame 2 of every 24 fps clip and shift the rest against the audio."""
    import fastgen.datasets.decoders as decoders_module

    src = _write_mp4(tmp_path, n_frames=72, fps=24)
    data = src.read_bytes()
    assert torch.equal(decode_video_full(src.name, data), decode_video_full(src.name, data, target_fps=24.0))

    # The segment decoder anchors its cadence at the seek offset; pin it so both decodes read one span.
    monkeypatch.setattr(decoders_module, "_random_start_sec", lambda total, needed: 1.0)
    plain = decoders_module.decode_video_audio_segment(src.name, data, 24, "torch", None)
    resampled = decoders_module.decode_video_audio_segment(src.name, data, 24, "torch", 24.0)
    assert torch.equal(plain["video"], resampled["video"])
    assert resampled["video_fps"] == 24.0


def test_target_fps_strides_at_decode_time(tmp_path: Path):
    """16-frame, 16 fps source, target_fps=4 → ~4 kept frames at ~0.25s spacing."""
    src = _write_mp4(tmp_path, n_frames=16, fps=16)
    arr = decode_video_full(src.name, src.read_bytes(), target_fps=4.0)
    assert arr is not None
    assert arr.shape[0] == 4
    assert arr.shape[1:] == (32, 32, 3)


def test_min_frames_floor_protects_short_clip(tmp_path: Path):
    """1-second source + target_fps=1 would normally yield 1 frame; the
    ``min_frames=4`` floor clamps the interval to ``duration / 4`` so
    we keep enough frames for a 4-panel grid."""
    src = _write_mp4(tmp_path, n_frames=16, fps=16)
    raw = decode_video_full(src.name, src.read_bytes(), target_fps=1.0)
    assert raw.shape[0] == 1, "sanity: target_fps=1 alone yields 1 frame"

    floored = decode_video_full(
        src.name,
        src.read_bytes(),
        target_fps=1.0,
        min_frames=4,
    )
    assert floored.shape[0] >= 4


def test_min_frames_inactive_when_target_fps_already_yields_enough(tmp_path: Path):
    """Long clip (5s) + target_fps=1 keeps ~5 frames; the floor for
    min_frames=4 is ``D/4 = 1.25s`` which is *larger* than the
    target_fps interval (1.0s), so the floor doesn't kick in and
    target_fps wins."""
    src = _write_mp4(tmp_path, n_frames=80, fps=16)
    arr = decode_video_full(
        src.name,
        src.read_bytes(),
        target_fps=1.0,
        min_frames=4,
    )
    assert arr is not None
    # 5 captures at t = 0, 1, 2, 3, 4 (interval = 1.0s, last frame ≈ 4.94s).
    assert arr.shape[0] == 5


def test_min_frames_floor_duplicates_short_clip(tmp_path: Path):
    """Source has fewer frames than min_frames → we duplicate (bounded
    by ``ceil(target_fps) + 1`` per source frame) to densify the
    cadence. Keeps consumers re-encoding at ``target_fps`` from
    speeding up the clip."""
    src = _write_mp4(tmp_path, n_frames=3, fps=4)
    arr = decode_video_full(
        src.name,
        src.read_bytes(),
        target_fps=1.0,
        min_frames=10,
    )
    assert arr is not None
    # Duplication kicked in (above raw source count of 3) but is
    # bounded by ``source_count * (ceil(target_fps) + 1) = 3 * 2``.
    assert arr.shape[0] > 3
    assert arr.shape[0] <= 3 * (math.ceil(1.0) + 1)


def test_min_frames_ignored_when_target_fps_is_none(tmp_path: Path):
    """``min_frames`` is gated on ``target_fps`` being set; without
    target_fps we keep every frame regardless of min_frames."""
    src = _write_mp4(tmp_path, n_frames=16, fps=16)
    arr = decode_video_full(src.name, src.read_bytes(), min_frames=4)
    assert arr.shape[0] == 16


def test_fractional_target_fps(tmp_path: Path):
    """Sub-1 fps is supported (e.g., ``0.5`` keeps every 2 seconds)."""
    src = _write_mp4(tmp_path, n_frames=80, fps=16)  # 5s clip
    arr = decode_video_full(src.name, src.read_bytes(), target_fps=0.5)
    assert arr is not None
    # 1 capture every 2s over a 5s clip → 3 captures (t=0, 2, 4).
    assert arr.shape[0] == 3


def test_output_format_numpy_vs_torch(tmp_path: Path):
    src = _write_mp4(tmp_path, n_frames=4, fps=4)
    np_arr = decode_video_full(
        src.name,
        src.read_bytes(),
        output_format="numpy",
    )
    torch_arr = decode_video_full(
        src.name,
        src.read_bytes(),
        output_format="torch",
    )
    assert isinstance(np_arr, np.ndarray)
    assert isinstance(torch_arr, torch.Tensor)
    assert np_arr.shape == torch_arr.shape


def test_invalid_output_format_raises(tmp_path: Path):
    src = _write_mp4(tmp_path, n_frames=4, fps=4)
    with pytest.raises(ValueError):
        decode_video_full(src.name, src.read_bytes(), output_format="nope")


def test_non_video_extension_returns_none():
    assert decode_video_full("fake.txt", b"hello") is None


def test_corrupted_bytes_returns_none(tmp_path: Path):
    """Garbage in → ``None`` out (logged), no exception escapes."""
    assert decode_video_full("clip.mp4", b"\x00" * 64) is None


def test_decode_video_full_target_fps_resamples_source_fps():
    """``decode_video_full`` resamples by timestamp and picks the expected source indices."""
    source_fps = 20
    target_fps = 5
    num_source_frames = source_fps * 3
    values = np.linspace(0, 240, num_source_frames, dtype=np.uint8)
    frames = _make_constant_video_frames(values)
    mp4 = _encode_constant_mp4(frames, fps=source_fps)

    decoded = decode_video_full(
        "sample.mp4", mp4, target_fps=target_fps, output_format="torch", return_source_frame_indices=True
    )

    assert decoded is not None
    assert decoded["frames"].shape == (15, 16, 16, 3)
    expected_indices = _expected_target_fps_indices(source_fps, target_fps, 15, num_source_frames)
    assert decoded["source_frame_indices"] == expected_indices  # [0, 4, 8, 12, ...]: no accumulated drift
    _assert_frame_means_match_values(decoded["frames"], values, expected_indices)


def test_resampled_decoders_report_each_kept_frames_source_index(monkeypatch):
    """With ``target_fps`` every kept frame comes from a known source frame, and both decoders report its index."""
    import fastgen.datasets.decoders as decoders_module

    source_fps, target_fps = 16, 8
    values = np.linspace(0, 240, 32, dtype=np.uint8)
    mp4 = _encode_constant_mp4(_make_constant_video_frames(values), fps=source_fps)

    full = decode_video_full(
        "sample.mp4", mp4, target_fps=target_fps, output_format="torch", return_source_frame_indices=True
    )
    assert full["source_frame_indices"] == list(range(0, 32, 2))
    _assert_frame_means_match_values(full["frames"], values, full["source_frame_indices"])

    monkeypatch.setattr(decoders_module, "_random_start_sec", lambda total, needed: 0.5)
    segment = decoders_module.decode_video_segment(
        "sample.mp4", mp4, 6, "torch", target_fps, return_segment_start_frame=True
    )
    assert segment["source_frame_indices"] == [8, 10, 12, 14, 16, 18]
    assert segment["segment_start_frame"] == 4  # the first kept frame (0.5 s) on the 8 fps grid
    _assert_frame_means_match_values(segment["frames"], values, segment["source_frame_indices"])


def test_source_frame_indices_are_relative_to_the_stream_start(monkeypatch):
    """Frame timestamps are absolute stream time. When the first frame sits at 0.5 s (mp4 edit-list delay,
    MPEG-TS) the full decoder must still report source frame 0 first and the segment anchor must subtract
    the same origin; ``round(t * fps)`` alone placed every frame 8 indices late."""
    import fastgen.datasets.decoders as decoders_module

    source_fps, target_fps = 16, 8
    values = np.linspace(0, 240, 32, dtype=np.uint8)
    mp4 = _encode_constant_mp4(_make_constant_video_frames(values), fps=source_fps, pts_offset=8)

    full = decode_video_full("sample.mp4", mp4, output_format="torch", return_source_frame_indices=True)
    assert full["source_frame_indices"] == list(range(32))
    _assert_frame_means_match_values(full["frames"], values, full["source_frame_indices"])
    resampled = decode_video_full(
        "sample.mp4", mp4, target_fps=target_fps, output_format="torch", return_source_frame_indices=True
    )
    assert resampled["source_frame_indices"] == list(range(0, 32, 2))
    _assert_frame_means_match_values(resampled["frames"], values, resampled["source_frame_indices"])

    # An offset of 1.0 s into the clip is stream time 1.5 s: source frame 16, index 8 on the 8 fps grid.
    monkeypatch.setattr(decoders_module, "_random_start_sec", lambda total, needed: 1.0)
    segment = decoders_module.decode_video_segment(
        "sample.mp4", mp4, 6, "torch", target_fps, return_segment_start_frame=True
    )
    assert segment["source_frame_indices"] == [16, 18, 20, 22, 24, 26]
    assert segment["segment_start_frame"] == 8
    _assert_frame_means_match_values(segment["frames"], values, segment["source_frame_indices"])


def test_segment_start_is_placed_on_the_stream_time_axis(monkeypatch):
    """``_random_start_sec`` draws an offset into the clip while the seek and the capture grid work in stream
    time. On a stream whose first frame sits at 0.5 s, offset 0 must start at that frame instead of
    duplicating it while a grid anchored at 0 s catches up, and a late offset must reach the clip's tail."""
    import fastgen.datasets.decoders as decoders_module

    values = np.linspace(0, 240, 32, dtype=np.uint8)
    mp4 = _encode_constant_mp4(_make_constant_video_frames(values), fps=16, pts_offset=8)

    monkeypatch.setattr(decoders_module, "_random_start_sec", lambda total, needed: 0.0)
    head = decoders_module.decode_video_segment("sample.mp4", mp4, 6, "torch", 8.0, return_segment_start_frame=True)
    assert head["source_frame_indices"] == [0, 2, 4, 6, 8, 10]
    assert head["segment_start_frame"] == 0
    _assert_frame_means_match_values(head["frames"], values, head["source_frame_indices"])

    monkeypatch.setattr(decoders_module, "_random_start_sec", lambda total, needed: 1.5)
    tail = decoders_module.decode_video_segment("sample.mp4", mp4, 6, "torch", None, return_segment_start_frame=True)
    assert tail["source_frame_indices"] == [24, 25, 26, 27, 28, 29]
    assert tail["segment_start_frame"] == 24
    _assert_frame_means_match_values(tail["frames"], values, tail["source_frame_indices"])


def test_segment_indices_are_withheld_when_timestamps_leave_the_frame_rate_grid(monkeypatch):
    """After a seek the first kept frame is placed by ``timestamp * source_fps``, which only holds for
    constant-frame-rate video. When the kept frames' timestamps disagree with decode order (variable frame
    rate, or rate metadata such as a field-rate ``guessed_rate``), the decoder reports ``None`` rather than
    wrong indices."""
    from fastgen.datasets import decoders

    class _FakeFrame:
        def __init__(self, pts):
            self.pts = pts
            self.time = None

        def to_ndarray(self, **_kwargs):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    class _FakeVideoStream:
        time_base = Fraction(1, 24)
        average_rate = 24
        guessed_rate = 24
        duration = 12
        start_time = 0

    class _FakeStreams:
        video = (_FakeVideoStream,)
        audio = ()

    class _FakeContainer:
        streams = _FakeStreams()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def seek(self, *args, **kwargs):
            pass

        def decode(self, stream):
            for pts in (0, 1, 2, 3, 5, 6, 7, 8):  # frame 4 is missing: variable frame rate
                yield _FakeFrame(pts)

    monkeypatch.setattr(decoders.av, "open", lambda *a, **kw: _FakeContainer())
    monkeypatch.setattr(decoders, "_random_start_sec", lambda total, needed: 0.0)
    vfr = decoders.decode_video_segment("sample.mp4", b"x", 6, "numpy", None, return_segment_start_frame=True)
    assert vfr["frames"].shape[0] == 6
    assert vfr["source_frame_indices"] is None


def test_full_decoder_indices_stay_aligned_when_a_frame_has_no_timestamp(monkeypatch):
    """Without ``target_fps`` the full decoder keeps frames that carry neither ``time`` nor ``pts``. Their
    source index comes from decode order, so the list stays one entry per frame; a timestamp-derived list
    would skip the frame and shift every later index."""
    from fastgen.datasets import decoders

    class _FakeFrame:
        def __init__(self, idx, timestamped):
            self.pts = idx if timestamped else None
            self.time = None

        def to_ndarray(self, **_kwargs):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    class _FakeVideoStream:
        time_base = Fraction(1, 24)
        average_rate = 24
        guessed_rate = 24
        duration = 5
        start_time = 0

    class _FakeStreams:
        video = (_FakeVideoStream,)
        audio = ()

    class _FakeContainer:
        streams = _FakeStreams()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def decode(self, stream):
            for i in range(5):
                yield _FakeFrame(i, timestamped=i != 2)

    monkeypatch.setattr(decoders.av, "open", lambda *a, **kw: _FakeContainer())
    result = decoders.decode_video_full("sample.mp4", b"x", output_format="numpy", return_source_frame_indices=True)
    assert result["frames"].shape[0] == 5
    assert result["source_frame_indices"] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("bad_target_fps", [0, -1, float("nan"), float("inf")])
def test_decode_video_full_rejects_nonfinite_target_fps(bad_target_fps):
    """Non-finite or non-positive ``target_fps`` is a configuration error."""
    with pytest.raises(ValueError, match="finite positive"):
        decode_video_full("sample.mp4", b"", target_fps=bad_target_fps)


@pytest.mark.parametrize("bad_min_frames", [0, -1, -42])
def test_decode_video_full_rejects_nonpositive_min_frames(bad_min_frames):
    """``min_frames`` divides duration when clamping the capture interval.
    Non-positive values would ``ZeroDivisionError`` or yield a negative interval."""
    with pytest.raises(ValueError, match="min_frames must be a positive integer"):
        decode_video_full("sample.mp4", b"", target_fps=4.0, min_frames=bad_min_frames)


# ============================================================================
# FrameCaptureCadence / constant_rate_frame_indices
# ============================================================================


@pytest.mark.parametrize(
    ("frame_count", "source_fps", "target_fps", "expected"),
    [
        (8, 16, 8, [0, 2, 4, 6]),
        (4, 8, 16, [0, 1, 1, 2, 2, 3, 3]),
        (72, 24, 24, list(range(72))),
        (3, 0.5, 2, [0, 1, 1, 1, 1, 2, 2, 2, 2]),
    ],
)
def test_constant_rate_frame_indices(frame_count, source_fps, target_fps, expected):
    assert constant_rate_frame_indices(frame_count, source_fps, target_fps) == expected


def test_capture_cadence_defers_emissions_limited_on_one_frame():
    cadence = FrameCaptureCadence(0.1, anchor=0.0)
    timestamp = 0.3 - 1e-12

    assert cadence.emissions_at(timestamp, max_emissions=2) == 2
    assert cadence.emissions_at(timestamp, max_emissions=2) == 2
    assert cadence.emissions_at(timestamp, max_emissions=2) == 0


@pytest.mark.parametrize(("timestamp", "anchor"), [(math.inf, 0.0), (0.0, -math.inf)])
def test_capture_cadence_cap_bounds_infinite_gaps(timestamp, anchor):
    cadence = FrameCaptureCadence(0.1, anchor=anchor)

    assert cadence.emissions_at(timestamp, max_emissions=3) == 3


# ============================================================================
# get_extension
# ============================================================================


@pytest.mark.parametrize(
    "key, expected",
    [
        ("clip.mp4", "mp4"),
        ("IMAGE.JPG", "jpg"),
        ("path/to/file.WebP", "webp"),
        ("a.b.c.mp4", "mp4"),
        ("no_dot", "no_dot"),
    ],
)
def test_get_extension(key: str, expected: str):
    assert get_extension(key) == expected


# ============================================================================
# decode_video_segment
# ============================================================================


def test_video_segment_returns_exact_num_frames(tmp_path: Path):
    """Source is long enough → returns exactly ``num_frames``."""
    src = _write_mp4(tmp_path, n_frames=32, fps=16)
    arr = decode_video_segment(src.name, src.read_bytes(), num_frames=8)
    assert isinstance(arr, torch.Tensor)
    assert arr.shape == (8, 32, 32, 3)


def test_video_segment_short_source_returns_none(tmp_path: Path):
    """Source has fewer than ``num_frames`` → returns ``None``.

    Segments are all-or-nothing: a fixed ``(num_frames, H, W, 3)`` is the contract,
    so a source that can't fill the request fails rather than returning a partial
    sample of unpredictable shape."""
    src = _write_mp4(tmp_path, n_frames=4, fps=4)
    arr = decode_video_segment(src.name, src.read_bytes(), num_frames=16)
    assert arr is None


def test_video_segment_random_start_in_bounds(tmp_path: Path, monkeypatch):
    """Force ``random.uniform`` to a known offset and verify the segment
    actually starts there by comparing the first segment frame to the
    full-decode frame at the matching source index. Pinning the RNG and
    comparing pixels (not just the shape) makes the seek path observable."""
    fps = 16
    src = _write_mp4(tmp_path, n_frames=80, fps=fps)  # 5s clip
    forced_start_sec = 2.0
    monkeypatch.setattr(
        "fastgen.datasets.decoders.random.uniform",
        lambda a, b: forced_start_sec,
    )
    arr = decode_video_segment(src.name, src.read_bytes(), num_frames=16)
    assert arr.shape == (16, 32, 32, 3)

    full = decode_video_full(src.name, src.read_bytes())
    expected_start = int(forced_start_sec * fps)
    # libx264 round-trip is lossy and a mid-stream seek can shift decoder
    # state slightly versus a from-start decode, so allow a small atol on
    # the uint8 pixel comparison (wider than the 3.0 of the per-frame mean
    # checks, since single pixels vary more than frame means).
    torch.testing.assert_close(arr[0].float(), full[expected_start].float(), rtol=0, atol=8.0)


def test_video_segment_invalid_num_frames_raises(tmp_path: Path):
    src = _write_mp4(tmp_path, n_frames=8, fps=8)
    with pytest.raises(ValueError):
        decode_video_segment(src.name, src.read_bytes(), num_frames=0)
    with pytest.raises(ValueError):
        decode_video_segment(src.name, src.read_bytes(), num_frames=-3)


def test_video_segment_invalid_output_format_raises(tmp_path: Path):
    src = _write_mp4(tmp_path, n_frames=8, fps=8)
    with pytest.raises(ValueError):
        decode_video_segment(src.name, src.read_bytes(), num_frames=4, output_format="nope")


def test_video_segment_non_video_extension_returns_none():
    assert decode_video_segment("file.txt", b"hello", num_frames=4) is None


def test_video_segment_corrupted_bytes_returns_none():
    assert decode_video_segment("clip.mp4", b"\x00" * 64, num_frames=4) is None


@pytest.mark.parametrize(("source_fps", "target_fps"), [(8, 2), (20, 2), (8, 16)])
def test_decode_video_segment_target_fps_resamples_source_fps(monkeypatch, source_fps, target_fps):
    """Timestamp-based segment decoding picks the expected source frames at multiple FPS pairs."""
    monkeypatch.setattr("fastgen.datasets.decoders.random.uniform", lambda _start, _end: 0.0)
    num_source_frames = source_fps * 3
    values = np.linspace(0, 240, num_source_frames, dtype=np.uint8)
    frames = _make_constant_video_frames(values)
    mp4 = _encode_constant_mp4(frames, fps=source_fps)

    decoded = decode_video_segment(
        "sample.mp4",
        mp4,
        num_frames=6,
        target_fps=target_fps,
        output_format="torch",
        return_segment_start_frame=True,
    )

    assert decoded is not None
    assert decoded["frames"].shape == (6, 16, 16, 3)
    expected_indices = _expected_target_fps_indices(source_fps, target_fps, 6, num_source_frames)
    assert decoded["source_frame_indices"] == expected_indices
    _assert_frame_means_match_values(decoded["frames"], values, expected_indices)


@pytest.mark.parametrize(
    ("source_fps", "target_fps", "expected_indices"),
    [(8, 16, [21, 21, 22, 22, 23, 23]), (20, 8, [47, 49, 52, 54, 57, 59])],
)
def test_decode_video_segment_target_fps_latest_start_ends_on_the_last_frame(
    monkeypatch, source_fps, target_fps, expected_indices
):
    """The random start reserves exactly the span a ``target_fps`` segment needs: at the largest offset the last
    capture lands on the last source frame, ``1 / source_fps`` before the stream end. A window of
    ``num_frames / target_fps`` puts that capture past the last frame when upsampling (a short segment, so the
    sample is dropped) and keeps segments off the clip's last frames when downsampling."""
    monkeypatch.setattr("fastgen.datasets.decoders.random.uniform", lambda _start, end: end)
    num_source_frames = source_fps * 3
    values = np.linspace(0, 240, num_source_frames, dtype=np.uint8)
    mp4 = _encode_constant_mp4(_make_constant_video_frames(values), fps=source_fps)

    decoded = decode_video_segment(
        "sample.mp4",
        mp4,
        num_frames=6,
        target_fps=target_fps,
        output_format="torch",
        return_segment_start_frame=True,
    )

    assert decoded is not None
    assert decoded["source_frame_indices"] == expected_indices
    _assert_frame_means_match_values(decoded["frames"], values, expected_indices)


def test_decode_video_segment_target_fps_short_source_returns_none():
    """Target-FPS segment decoding returns ``None`` when the source can't supply
    ``num_frames`` frames at the requested cadence — no static tail padding,
    no partial sample."""
    source_fps = 4
    target_fps = 8
    values = np.linspace(0, 240, 2, dtype=np.uint8)
    frames = _make_constant_video_frames(values)
    mp4 = _encode_constant_mp4(frames, fps=source_fps)

    decoded = decode_video_segment("sample.mp4", mp4, num_frames=6, target_fps=target_fps, output_format="torch")

    assert decoded is None


@pytest.mark.parametrize("target_fps", [None, 12])
def test_decode_video_segment_falls_back_to_pts_when_frame_time_is_none(monkeypatch, target_fps):
    """Frames with ``frame.time is None`` but valid ``frame.pts`` must be kept.

    Some PyAV-decoded streams produce frames where ``frame.time`` is None even
    though ``frame.pts`` is set; in that case the timestamp is recovered via
    ``frame.pts * stream.time_base``. Regression for both the ``target_fps`` and
    no-``target_fps`` branches of ``decode_video_segment`` and
    ``decode_video_audio_segment``.
    """
    from fastgen.datasets import decoders

    num_emitted = 12
    source_fps = 24
    tb = Fraction(1, source_fps)

    class _FakeFrame:
        def __init__(self, idx):
            self.pts = idx
            self.time = None

        def to_ndarray(self, **_kwargs):
            return np.full((8, 8, 3), self.pts, dtype=np.uint8)

    class _FakeVideoStream:
        time_base = tb
        average_rate = source_fps
        guessed_rate = source_fps
        duration = num_emitted
        start_time = 0

    class _FakeStreams:
        video = (_FakeVideoStream,)
        audio = ()

    class _FakeContainer:
        streams = _FakeStreams()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def seek(self, *args, **kwargs):
            pass

        def decode(self, stream):
            for i in range(num_emitted):
                yield _FakeFrame(i)

    monkeypatch.setattr(decoders.av, "open", lambda *a, **kw: _FakeContainer())
    monkeypatch.setattr("fastgen.datasets.decoders.random.uniform", lambda _start, _end: 0.0)

    decoded = decoders.decode_video_segment(
        "sample.mp4", b"x", num_frames=6, target_fps=target_fps, output_format="numpy"
    )
    assert decoded is not None
    assert decoded.shape == (6, 8, 8, 3)

    result = decoders.decode_video_audio_segment(
        "sample.mp4", b"x", num_frames=6, target_fps=target_fps, output_format="numpy"
    )
    assert result is not None
    assert result["video"].shape == (6, 8, 8, 3)
    assert result["audio"] is None


@pytest.mark.parametrize("target_fps, expected_frames", [(None, 12), (12, 6)])
def test_decode_video_full_falls_back_to_pts_when_frame_time_is_none(monkeypatch, target_fps, expected_frames):
    """Full-decode counterpart to the segment pts-fallback test.

    ``decode_video_full`` / ``decode_video_audio_full`` go through the same
    ``_collect_video_frames`` helper as the segment path, so this exercises the
    ``frame.pts * time_base`` recovery branch through the full-stream entry
    points as well. With ``target_fps=None`` every emitted source frame is
    kept; with ``target_fps=12`` over a 24-fps source the cadence subsamples
    to half — both branches produce non-empty arrays whose shape we can pin.
    """
    from fastgen.datasets import decoders

    num_emitted = 12
    source_fps = 24
    tb = Fraction(1, source_fps)

    class _FakeFrame:
        def __init__(self, idx):
            self.pts = idx
            self.time = None

        def to_ndarray(self, **_kwargs):
            return np.full((8, 8, 3), self.pts, dtype=np.uint8)

    class _FakeVideoStream:
        time_base = tb
        average_rate = source_fps
        guessed_rate = source_fps
        duration = num_emitted
        start_time = 0

    class _FakeStreams:
        video = (_FakeVideoStream,)
        audio = ()

    class _FakeContainer:
        streams = _FakeStreams()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def seek(self, *args, **kwargs):
            pass

        def decode(self, stream):
            for i in range(num_emitted):
                yield _FakeFrame(i)

    monkeypatch.setattr(decoders.av, "open", lambda *a, **kw: _FakeContainer())
    monkeypatch.setattr("fastgen.datasets.decoders.random.uniform", lambda _start, _end: 0.0)

    decoded = decoders.decode_video_full("sample.mp4", b"x", target_fps=target_fps, output_format="numpy")
    assert decoded is not None
    assert decoded.shape == (expected_frames, 8, 8, 3)

    result = decoders.decode_video_audio_full("sample.mp4", b"x", target_fps=target_fps, output_format="numpy")
    assert result is not None
    assert result["video"].shape == (expected_frames, 8, 8, 3)
    assert result["audio"] is None


@pytest.mark.parametrize("bad_target_fps", [0, -1, float("nan"), float("inf")])
def test_decode_video_segment_rejects_nonfinite_target_fps(bad_target_fps):
    """Non-finite or non-positive ``target_fps`` is a configuration error."""
    with pytest.raises(ValueError, match="finite positive"):
        decode_video_segment("sample.mp4", b"", num_frames=6, target_fps=bad_target_fps)


@pytest.mark.parametrize("bad_num_frames", [0, -1, -10])
def test_decode_video_audio_segment_rejects_nonpositive_num_frames(bad_num_frames):
    """``decode_video_audio_segment`` rejects ``num_frames <= 0`` up-front,
    matching ``decode_video_segment``."""
    from fastgen.datasets.decoders import decode_video_audio_segment

    with pytest.raises(ValueError, match="num_frames must be positive"):
        decode_video_audio_segment("sample.mp4", b"", num_frames=bad_num_frames)


# ============================================================================
# decode_video_audio_full
# ============================================================================


def test_decode_video_audio_full_video_only_source(tmp_path: Path):
    """``decode_video_audio_full`` on a video-only mp4 returns the full video
    track with ``audio`` / ``audio_sr`` set to ``None`` (no audio stream in the
    container) and ``video_fps`` matching the source rate."""
    src = _write_mp4(tmp_path, n_frames=16, fps=16)
    result = decode_video_audio_full(src.name, src.read_bytes())

    assert isinstance(result, dict)
    assert set(result) == {"video", "audio", "audio_sr", "video_fps"}
    assert isinstance(result["video"], torch.Tensor)
    assert result["video"].shape == (16, 32, 32, 3)
    assert result["audio"] is None
    assert result["audio_sr"] is None
    assert result["video_fps"] == pytest.approx(16.0)


def test_decode_video_audio_full_target_fps_subsamples(tmp_path: Path):
    """``target_fps`` resamples the video track and is reported via ``video_fps``."""
    src = _write_mp4(tmp_path, n_frames=16, fps=16)
    result = decode_video_audio_full(src.name, src.read_bytes(), target_fps=4.0)

    assert result is not None
    assert result["video"].shape[0] == 4
    assert result["video_fps"] == pytest.approx(4.0)
    assert result["audio"] is None


def test_segment_audio_is_anchored_at_the_first_kept_frame(monkeypatch):
    """A random start between two source frames keeps the first frame at or after it; the audio must start at
    that frame's time, not at the requested start, or the clip carries up to one source frame of A/V offset."""
    import fastgen.datasets.decoders as decoders_module

    sr, fps = 16000, 16
    values = np.linspace(0, 240, 32, dtype=np.uint8)
    ramp = np.linspace(0.0, 1.0, 2 * sr, endpoint=False, dtype=np.float32)[None]  # value = stream time / 2 s
    mov = _encode_mov(frames=_make_constant_video_frames(values), fps=fps, audio=ramp, sample_rate=sr)

    monkeypatch.setattr(decoders_module, "_random_start_sec", lambda total, needed: 0.03)  # frame 1 is at 0.0625 s
    result = decoders_module.decode_video_audio_segment("sample.mov", mov, 6, "torch", None)
    _assert_frame_means_match_values(result["video"], values, [1, 2, 3, 4, 5, 6])
    assert result["audio"].shape[-1] == 6 * sr // fps
    assert abs(result["audio"][0, 0].item() - 0.0625 / 2) < 2e-3


def test_full_audio_is_anchored_at_the_first_video_frame():
    """Tracks may start at different times; the full decoder anchors the audio at the first video frame's
    stream time (0.5 s here, while the audio starts at zero)."""
    sr, fps = 16000, 16
    values = np.linspace(0, 240, 32, dtype=np.uint8)
    ramp = np.linspace(0.0, 1.0, 2 * sr, endpoint=False, dtype=np.float32)[None]  # value = stream time / 2 s
    mov = _encode_mov(
        frames=_make_constant_video_frames(values), fps=fps, video_pts_offset=8, audio=ramp, sample_rate=sr
    )

    result = decode_video_audio_full("sample.mov", mov, "torch", None)
    assert result["video"].shape[0] == 32
    assert result["audio"].shape[-1] == 32 * sr // fps
    assert abs(result["audio"][0, 0].item() - 0.5 / 2) < 2e-3


# ============================================================================
# decode_image
# ============================================================================


def _png_bytes(size=(8, 8), color=(120, 30, 200)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_image_png_returns_pil():
    img = decode_image("a.png", _png_bytes())
    assert isinstance(img, Image.Image)
    assert img.size == (8, 8)


def test_decode_image_applies_transform():
    img = decode_image(
        "a.png",
        _png_bytes(size=(8, 8)),
        image_transform=Compose([Resize((4, 4))]),
    )
    assert isinstance(img, Image.Image)
    assert img.size == (4, 4)


def test_decode_image_non_image_extension_returns_none():
    assert decode_image("a.mp4", b"hi") is None
    assert decode_image("a.txt", b"hi") is None


# ============================================================================
# decode_txt
# ============================================================================


def test_decode_txt_strips_by_default():
    assert decode_txt("a.txt", b"  hello world  \n") == "hello world"


def test_decode_txt_strip_false_preserves_whitespace():
    assert decode_txt("a.txt", b"  hello\n", strip=False) == "  hello\n"


def test_decode_txt_custom_extensions():
    assert decode_txt("a.json", b"{}", txt_extensions=("json",)) == "{}"


def test_decode_txt_wrong_extension_returns_none():
    assert decode_txt("a.png", b"hi") is None


# ============================================================================
# decode_npy_torch / decode_npz_torch
# ============================================================================


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_decode_npy_torch_round_trip():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = decode_npy_torch("emb.npy", _npy_bytes(src))
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    np.testing.assert_array_equal(out.numpy(), src)


def test_decode_npy_torch_wrong_extension_returns_none():
    assert decode_npy_torch("emb.bin", b"\x00") is None


def test_decode_npz_torch_round_trip():
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    b = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
    out = decode_npz_torch("bundle.npz", _npz_bytes(a=a, b=b))
    assert isinstance(out, dict)
    assert set(out) == {"a", "b"}
    assert isinstance(out["a"], torch.Tensor) and isinstance(out["b"], torch.Tensor)
    np.testing.assert_array_equal(out["a"].numpy(), a)
    np.testing.assert_array_equal(out["b"].numpy(), b)


def test_decode_npz_torch_wrong_extension_returns_none():
    assert decode_npz_torch("bundle.bin", b"\x00") is None
