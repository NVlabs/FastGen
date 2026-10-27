# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for media encoder option routing."""

import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import fastgen.utils.media_utils as media_utils


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("as_dict", [False, True])
def test_save_media_audio_preserves_channels_rate_and_batch(channels, as_dict, tmp_path):
    waveform = torch.linspace(-1.5, 1.5, 240).reshape(1, 1, -1).repeat(2, channels, 1)
    if channels == 2:
        waveform[:, 1].neg_()
    output = {"audio": waveform} if as_dict else waveform
    media_utils.save_media(output, str(tmp_path / "audio.wav"), audio_sample_rate=48000, vid_format="mp4")
    for index in range(2):
        with wave.open(str(tmp_path / f"audio_{index:03d}.wav")) as stream:
            assert stream.getframerate() == 48000
            assert stream.getnchannels() == channels
            assert stream.getnframes() == 240
            actual = np.frombuffer(stream.readframes(240), dtype="<i2").reshape(-1, channels).T
        expected = waveform[index].clamp(-1, 1).mul(32767).round().to(torch.int16).numpy()
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(3, 64), (64, 2), (1, 2, 64)])
def test_save_audio_rejects_non_channel_first_waveforms(shape, tmp_path):
    with pytest.raises(ValueError, match="channel-first mono or stereo"):
        media_utils.save_audio(torch.zeros(shape), tmp_path / "audio.wav")


@pytest.mark.parametrize("layout", ["samples", "samples_channels", "channels_samples", "batch_channels_samples"])
def test_save_audio_video_mono_layouts_preserve_samples_and_duration(layout, tmp_path):
    import av

    sample_rate = 16000
    mono = 0.25 * torch.sin(torch.arange(sample_rate) * (2 * torch.pi * 440 / sample_rate))
    audio = {
        "samples": mono,
        "samples_channels": mono[:, None],
        "channels_samples": mono[None, :],
        "batch_channels_samples": mono[None, None, :],
    }[layout]
    reference = mono[:, None].expand(-1, 2)
    video = torch.zeros(3, 8, 4, 4)

    def save_and_decode(waveform, filename):
        path = str(tmp_path / filename)
        media_utils.save_audio_video(video, waveform, path, fps=16, audio_sample_rate=sample_rate)
        with av.open(path) as container:
            frames = [torch.from_numpy(frame.to_ndarray()) for frame in container.decode(audio=0)]
        return torch.cat(frames, dim=-1)

    expected = save_and_decode(reference, "reference.mp4")
    actual = save_and_decode(audio, "mono.mp4")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(16, 6), (6, 16), (3, 2, 16)])
def test_save_audio_video_rejects_unsupported_audio_shapes(shape, tmp_path):
    with pytest.raises(ValueError, match="mono or stereo waveform"):
        media_utils.save_audio_video(torch.zeros(3, 1, 4, 4), torch.zeros(shape), str(tmp_path / "sample.mp4"))


def _decoded_audio_seconds(path: str) -> float:
    import av

    with av.open(path) as container:
        stream = container.streams.audio[0]
        samples = sum(frame.samples for frame in container.decode(stream))
    return samples / stream.rate


@pytest.mark.parametrize("audio_seconds", [3.0, 1.0])
def test_save_audio_video_trims_audio_to_the_video_duration(tmp_path, audio_seconds):
    """A 2 s clip keeps at most 2 s of audio (LTX-2 audio latents can run one frame long);
    shorter audio is muxed as is."""
    fps, sample_rate = 4, 16000
    video = torch.zeros(3, 8, 16, 16)  # 8 frames at 4 fps = 2 s
    audio = torch.zeros(2, int(audio_seconds * sample_rate))

    path = media_utils.save_audio_video(
        video,
        audio,
        str(tmp_path / "clip.mp4"),
        fps=fps,
        audio_sample_rate=sample_rate,
    )

    # AAC priming and end padding add well under 0.3 s at 16 kHz.
    assert abs(_decoded_audio_seconds(path) - min(audio_seconds, 2.0)) < 0.3


@pytest.mark.parametrize("fps", [7.5, 29.97, 30000 / 1001])
def test_save_audio_video_preserves_fractional_frame_rate_and_duration(tmp_path, fps):
    import av
    from fractions import Fraction

    frames, sample_rate = 30, 16000
    expected_seconds = frames / fps
    path = media_utils.save_audio_video(
        torch.zeros(3, frames, 16, 16),
        torch.zeros(2, round((expected_seconds + 1) * sample_rate)),
        str(tmp_path / "fractional.mp4"),
        fps=fps,
        audio_sample_rate=sample_rate,
    )
    with av.open(path) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == Fraction(str(fps)).limit_denominator()
        assert float(stream.duration * stream.time_base) == pytest.approx(expected_seconds, abs=1e-4)
        assert sum(1 for _ in container.decode(stream)) == frames
    # AAC may add at most a little encoder padding, independent of video cadence.
    assert _decoded_audio_seconds(path) == pytest.approx(expected_seconds, abs=0.15)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_save_audio_video_rejects_invalid_rates_before_creating_output(tmp_path, fps):
    path = tmp_path / "invalid.mp4"
    with pytest.raises(ValueError, match="fps must be a finite positive number"):
        media_utils.save_audio_video(torch.zeros(3, 1, 4, 4), None, str(path), fps=fps)
    assert not path.exists()


@pytest.mark.parametrize("fps", [1e-7, 7e-7])
def test_save_audio_video_rejects_rates_it_cannot_store_accurately(tmp_path, fps):
    """1e-7 fps rounds to a zero rate, which PyAV replaces with 24 fps; 7e-7 rounds 43% high."""
    path = tmp_path / "tiny_rate.mp4"
    with pytest.raises(ValueError, match="cannot be stored accurately"):
        media_utils.save_audio_video(torch.zeros(3, 1, 4, 4), None, str(path), fps=fps)
    assert not path.exists()


def _audio_video_batch() -> dict[str, torch.Tensor]:
    return {
        "video": torch.zeros(1, 3, 2, 4, 4),
        "audio": torch.zeros(1, 16, 2),
    }


def test_save_media_defaults_audio_video_to_mp4_and_forwards_encoder_options(monkeypatch, tmp_path):
    captured = {}

    def fake_save_audio_video(video, audio, path, **kwargs):
        captured.update(video=video, audio=audio, path=path, kwargs=kwargs)

    monkeypatch.setattr(media_utils, "save_audio_video", fake_save_audio_video)

    media_utils.save_media(
        _audio_video_batch(),
        str(tmp_path / "sample.mp4"),
        quality=18,
        preset="slow",
        x264_options={"tune": "film"},
    )

    assert captured["kwargs"] == {
        "quality": 18,
        "preset": "slow",
        "x264_options": {"tune": "film"},
    }


def test_save_media_falls_back_to_mp4_for_audio_video_gif(monkeypatch, tmp_path):
    paths, messages = [], []
    monkeypatch.setattr(media_utils, "save_audio_video", lambda video, audio, path, **kwargs: paths.append(path))
    monkeypatch.setattr(media_utils.logger, "warning", messages.append)

    media_utils.save_media(_audio_video_batch(), str(tmp_path / "sample.gif"), vid_format="gif")

    assert paths == [str(tmp_path / "sample.mp4")]
    assert len(messages) == 1 and "cannot carry audio" in messages[0]


def test_save_media_names_waveforms_as_wav(tmp_path):
    media_utils.save_media(
        torch.zeros(1, 2, 16), str(tmp_path / "sample.mp4"), audio_sample_rate=16000, vid_format="mp4"
    )

    assert [path.name for path in tmp_path.iterdir()] == ["sample.wav"]
    with wave.open(str(tmp_path / "sample.wav")) as stream:
        assert stream.getnchannels() == 2
        assert stream.getnframes() == 16


def test_save_audio_video_rejects_unknown_encoder_options(tmp_path):
    with pytest.raises(TypeError, match="unexpected keyword argument 'qualty'"):
        media_utils.save_audio_video(
            torch.zeros(3, 1, 4, 4),
            None,
            str(tmp_path / "sample.mp4"),
            qualty=18,
        )


def test_save_media_treats_single_temporal_frame_as_video(monkeypatch, tmp_path):
    captured = {}

    def fake_save_video(frames, path, **kwargs):
        captured.update(frames=frames, path=path, kwargs=kwargs)

    monkeypatch.setattr(media_utils, "save_video", fake_save_video)
    monkeypatch.setattr(media_utils, "save_image", lambda *_args, **_kwargs: pytest.fail("5D media is not an image"))

    media_utils.save_media(
        torch.zeros(1, 3, 1, 4, 4),
        str(tmp_path / "sample.mp4"),
        vid_format="mp4",
    )

    assert captured["frames"].shape == (3, 1, 4, 4)
    assert captured["path"] == tmp_path / "sample.mp4"
    assert captured["kwargs"]["vid_format"] == "mp4"


def test_save_audio_video_owns_x264_option_resolution(monkeypatch, tmp_path):
    class FakeStream:
        options = None

        def encode(self, frame=None):
            return []

    class FakeContainer:
        def __init__(self):
            self.video_stream = FakeStream()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def add_stream(self, codec, rate):
            assert codec == "libx264"
            return self.video_stream

        def mux(self, packet):
            raise AssertionError("The fake encoder does not emit packets")

    container = FakeContainer()
    fake_av = SimpleNamespace(
        open=lambda path, mode: container,
        VideoFrame=SimpleNamespace(from_ndarray=lambda array, format: object()),
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)

    media_utils.save_audio_video(
        torch.zeros(3, 1, 4, 4),
        None,
        str(tmp_path / "sample.mp4"),
        quality=18,
        preset="slow",
        x264_options={"preset": "veryslow", "tune": "film"},
    )

    assert container.video_stream.options == {
        "crf": "18",
        "preset": "veryslow",
        "tune": "film",
    }
