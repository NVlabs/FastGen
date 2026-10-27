# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logger-side unit tests.

Surfaces exercised:

* ``fastgen.utils.logging_utils`` facade routing and configuration.
* ``WandbCallback._log_panels()`` media-suffix dispatch and validation.
* ``VLMLogger.__init__`` validation (rejected knob combinations).
* ``VLMLogger.process()`` — rank-0-only ``LogModule`` that filters
  ``MediaEntry`` records by ``kind == "student"``, scores each via
  the judge (parallelized through a thread pool when
  ``n_concurrent > 1``), logs a single per-step ``wandb.Table``, and
  returns the cross-sample mean ``vlm_score`` for cross-rank scalar
  aggregation.
* ``_LossDictRecord.get_stat()`` — aggregation that averages each key
  over the summed per-rank iteration counts, so sparse keys (e.g.
  per-rank VLM scores with ``n_samples < world_size``) are not
  under-counted.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

from loguru import logger as loguru_logger
import pytest
import torch

from fastgen.callbacks.vlm_logger import VLMLogger
from fastgen.callbacks.wandb import MediaEntry, WandbCallback, _LossDictRecord
import fastgen.utils.logging_utils as log
from fastgen.utils.vlm_judge import JudgeResult, JudgeSample, parse_score


@pytest.fixture(autouse=True)
def _reset_wandb_run(monkeypatch):
    """Force ``wandb.run = None`` at the start of every test, whatever run another test file left behind; tests that
    want a live run re-set ``wandb.run`` via ``monkeypatch.setattr`` themselves."""
    monkeypatch.setattr("fastgen.callbacks.vlm_logger.wandb.run", None)


@pytest.mark.parametrize("value", ["-1", "100", "10.1", "1e1000", "NaN", "Infinity"])
@pytest.mark.parametrize("label", ["FINAL_SCORE: ", "Rating = ", ""])
def test_judge_rejects_invalid_scores(value, label):
    assert parse_score(label + value) is None
    if label:
        assert parse_score(label + value + "\nBreakdown: motion=2\n8") is None


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("FINAL_SCORE: **7**\nJustification: good", 7),
        ("Rating = 8/10", 8),
        ("**9.5**", 9.5),
        ("FINAL SCORE: 0", 0),
        ("10/10", 10),
        ("FINAL_SCORE: 1e1", 10),
        ("FINAL_SCORE: +7", 7),
        ("FINAL_SCORE: 7.5.", 7.5),
        ("Rating: 6.5. The motion is smooth.", 6.5),
        ("Breakdown: motion=2", None),
        ("FINAL_SCORE: -1/10", None),
    ],
)
def test_judge_score_formats(reply, expected):
    assert parse_score(reply) == expected


# ===========================================================================
# fastgen.utils.logging_utils
# ===========================================================================


_FACADE_LEVELS = ("trace", "info", "debug", "success", "critical", "warning", "error")


class _RecordingLogger:
    def __init__(self):
        self.options = []
        self.calls = []

    def opt(self, **kwargs):
        self.options.append(kwargs)
        return self

    def __getattr__(self, name):
        if name not in _FACADE_LEVELS:
            raise AttributeError(name)

        def record(message):
            self.calls.append((name, message))

        return record


class _ConfiguringLogger:
    def __init__(self):
        self.remove_calls = []
        self.add_calls = []

    def remove(self, *args, **kwargs):
        self.remove_calls.append((args, kwargs))

    def add(self, *args, **kwargs):
        self.add_calls.append((args, kwargs))


def _mock_rank(monkeypatch, worker_rank):
    monkeypatch.setattr(log, "get_rank", lambda: worker_rank)
    monkeypatch.setattr(log, "is_rank0", lambda: worker_rank == 0)


def _call_every_facade_level():
    for level in _FACADE_LEVELS:
        getattr(log, level)(f"{level} message")


def test_rank_zero_emits_every_facade_level_outside_debug(monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr(log, "logger", recorder)
    monkeypatch.setattr(log, "LOG_LEVEL", "INFO")
    _mock_rank(monkeypatch, worker_rank=0)

    _call_every_facade_level()

    assert recorder.calls == [(level, f"{level} message") for level in _FACADE_LEVELS]
    assert recorder.options == [{"depth": 2}] * len(_FACADE_LEVELS)


def test_nonzero_rank_is_suppressed_outside_debug(monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr(log, "logger", recorder)
    monkeypatch.setattr(log, "LOG_LEVEL", "INFO")
    _mock_rank(monkeypatch, worker_rank=3)

    _call_every_facade_level()

    assert recorder.calls == []
    assert recorder.options == []


def test_debug_allows_every_facade_level_and_formatter_prefixes_rank(monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr(log, "logger", recorder)
    monkeypatch.setattr(log, "LOG_LEVEL", "DEBUG")
    _mock_rank(monkeypatch, worker_rank=7)

    _call_every_facade_level()

    assert recorder.calls == [(level, f"{level} message") for level in _FACADE_LEVELS]
    assert recorder.options == [{"depth": 2}] * len(_FACADE_LEVELS)
    assert log.formatter({}).startswith("[<magenta>RANK: 7</magenta>] ")
    # A callable format must end its template with the newline and the exception, which loguru adds only to strings.
    assert log.formatter({}).endswith("{message}\n{exception}")


def test_raw_logger_log_level_and_setter_remain_available(monkeypatch):
    assert log.logger is loguru_logger
    assert isinstance(log.LOG_LEVEL, str)
    assert callable(log.set_log_level)

    recorder = _ConfiguringLogger()
    monkeypatch.setattr(log, "logger", recorder)
    monkeypatch.setattr(log, "LOG_LEVEL", "INFO")

    log.set_log_level("warning")

    assert log.LOG_LEVEL == "WARNING"
    assert recorder.remove_calls == [((), {})]
    assert len(recorder.add_calls) == 1
    args, kwargs = recorder.add_calls[0]
    assert args == (sys.stdout,)
    assert kwargs == {"format": log.formatter, "level": "WARNING"}


# ===========================================================================
# _LossDictRecord.get_stat()
# ===========================================================================


def _populate(record: _LossDictRecord, per_rank_loss_dicts: list[dict[str, float]]):
    """Accumulate every rank's contribution into one record; the single-process gather is the identity."""
    for loss_dict in per_rank_loss_dicts:
        record.add({name: torch.tensor(value) for name, value in loss_dict.items()})


def test_get_stat_dense_all_ranks_contribute():
    """Every rank contributes the same key once: the mean is the plain value mean."""
    rec = _LossDictRecord()
    _populate(rec, [{"loss": float(r)} for r in range(4)])
    out = rec.get_stat()
    # mean(0,1,2,3) = 1.5
    assert out == {"loss": 1.5}


def test_get_stat_sparse_only_two_ranks_contribute():
    """Per-rank VLM scoring: ranks 0 and 1 score, ranks 2-3 don't; the mean runs over the two reported
    iterations (0.5), not over all four ranks (0.25)."""
    rec = _LossDictRecord()
    _populate(rec, [{"vlm_score": 0.0}, {"vlm_score": 1.0}, {}, {}])
    out = rec.get_stat()
    assert out == {"vlm_score": 0.5}


def test_get_stat_skips_zero_count_keys():
    """Empty contribution -> key absent (was: 0.0 sentinel)."""
    rec = _LossDictRecord()
    _populate(rec, [{}, {}])
    out = rec.get_stat()
    assert out == {}


def test_get_stat_multi_count_per_rank():
    """Each rank adds the same key multiple times; mean still correct."""
    rec = _LossDictRecord()
    rec.add({"loss": torch.tensor(2.0)})
    rec.add({"loss": torch.tensor(4.0)})  # rank-0 added twice
    # Simulate a second rank: gather merges both rank dicts.
    rec.loss_dict["loss"] = rec.loss_dict["loss"] + 3.0 + 5.0
    rec.iter_count_dict["loss"] = rec.iter_count_dict["loss"] + 2
    out = rec.get_stat()
    # Sum=2+4+3+5=14; count=4; mean=3.5.
    assert out == {"loss": 3.5}


# ===========================================================================
# VLMLogger.__init__ validation
# ===========================================================================


_STUB_KW = dict(model="stub-model", base_url="http://stub")


def test_init_rejects_invalid_media_mode():
    with pytest.raises(ValueError, match="unsupported media_mode"):
        VLMLogger(media_mode="bogus", **_STUB_KW)


def test_init_rejects_zero_max_samples():
    with pytest.raises(ValueError, match="max_samples"):
        VLMLogger(max_samples=0, **_STUB_KW)


def test_init_rejects_zero_n_concurrent():
    with pytest.raises(ValueError, match="n_concurrent"):
        VLMLogger(n_concurrent=0, **_STUB_KW)


def test_init_rejects_non_positive_target_fps():
    with pytest.raises(ValueError, match="target_fps"):
        VLMLogger(target_fps=0.0, **_STUB_KW)
    with pytest.raises(ValueError, match="target_fps"):
        VLMLogger(target_fps=-1.5, **_STUB_KW)


def test_init_accepts_none_target_fps():
    """``None`` keeps every frame — must remain a valid value."""
    m = VLMLogger(target_fps=None, **_STUB_KW)
    assert m.target_fps is None


def test_init_requires_model():
    with pytest.raises(ValueError, match="model is required"):
        VLMLogger(model="", base_url="http://stub")


def test_init_requires_base_url(monkeypatch):
    monkeypatch.delenv("VLM_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="base_url not set"):
        VLMLogger(model="stub-model")


def test_init_picks_up_base_url_env(monkeypatch):
    monkeypatch.setenv("VLM_BASE_URL", "http://from-env")
    m = VLMLogger(model="stub-model")
    assert m.base_url == "http://from-env"


def test_init_prefers_the_base_url_argument_over_env(monkeypatch):
    monkeypatch.setenv("VLM_BASE_URL", "http://from-env")
    m = VLMLogger(model="stub-model", base_url="http://from-config")
    assert m.base_url == "http://from-config"


# ===========================================================================
# load_vlm_client — OPENAI_API_KEY env contract
# ===========================================================================


def test_load_vlm_client_requires_openai_api_key(monkeypatch):
    """Without ``OPENAI_API_KEY`` in the env, ``load_vlm_client``
    raises a clear, actionable error pointing at the credential file."""
    from fastgen.utils.vlm_judge import load_vlm_client

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        load_vlm_client("http://stub")


def test_load_vlm_client_requires_base_url(monkeypatch):
    from fastgen.utils.vlm_judge import load_vlm_client

    monkeypatch.setenv("OPENAI_API_KEY", "stub-key")
    with pytest.raises(ValueError, match="base_url"):
        load_vlm_client("")


def test_load_vlm_client_builds_client_from_env(monkeypatch):
    """Happy path: with both env var + base URL, the SDK gets called
    and returns a client object. Stubs out ``OpenAI`` to keep the test
    hermetic (no network, no openai-package import side effects).

    No ``importorskip`` here: ``openai`` is a required runtime dep for
    VLM scoring, so a missing import is a real CI failure — not a skip."""
    monkeypatch.setenv("OPENAI_API_KEY", "stub-key")

    captured = {}

    class _StubOpenAI:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", _StubOpenAI)

    from fastgen.utils.vlm_judge import load_vlm_client

    client = load_vlm_client("http://stub-gateway")
    assert isinstance(client, _StubOpenAI)
    assert captured["base_url"] == "http://stub-gateway"


# ===========================================================================
# WandbCallback._log_panels()
# ===========================================================================


@pytest.mark.parametrize(
    ("extension", "expected_type"),
    [(".png", "image"), (".JPG", "image"), (".mp4", "video"), (".GIF", "video")],
)
def test_wandb_panel_dispatches_supported_media_suffixes(tmp_path, monkeypatch, extension, expected_type):
    import fastgen.callbacks.wandb as wandb_callback_module

    logged = []
    monkeypatch.setattr(wandb_callback_module.wandb, "run", object())
    monkeypatch.setattr(
        wandb_callback_module.wandb,
        "Image",
        lambda path, **kwargs: ("image", path, kwargs),
    )
    monkeypatch.setattr(
        wandb_callback_module.wandb,
        "Video",
        lambda path, **kwargs: ("video", path, kwargs),
    )
    monkeypatch.setattr(
        wandb_callback_module.wandb,
        "log",
        lambda payload, **kwargs: logged.append((payload, kwargs)),
    )

    path = tmp_path / f"sample{extension}"
    entry = MediaEntry(kind="student", rank=0, batch_idx=0, path=path, prompt="caption")
    WandbCallback()._log_panels([entry], iteration=5, group="train", suffix="")

    payload, log_kwargs = logged.pop()
    tile = payload["train_media/student/generation"][0]
    assert tile[0] == expected_type
    assert tile[1] == str(path)
    assert tile[2]["caption"] == "caption"
    assert log_kwargs == {"step": 5}
    if expected_type == "video":
        assert tile[2]["format"] == extension.lstrip(".").lower()


def test_wandb_panel_rejects_unsupported_media_suffix(tmp_path, monkeypatch):
    import fastgen.callbacks.wandb as wandb_callback_module

    monkeypatch.setattr(wandb_callback_module.wandb, "run", object())
    path = tmp_path / "sample.webp"
    entry = MediaEntry(kind="student", rank=0, batch_idx=0, path=path, prompt="")

    with pytest.raises(ValueError, match="Unsupported media suffix") as exc_info:
        WandbCallback()._log_panels([entry], iteration=5, group="train", suffix="")

    assert str(path) in str(exc_info.value)


# ===========================================================================
# VLMLogger.process()
# ===========================================================================


def _make_module(
    media_mode: str = "image",
    n_concurrent: int = 1,
) -> VLMLogger:
    module = VLMLogger(
        media_mode=media_mode,
        master_prompt="STUB MASTER PROMPT: reply FINAL_SCORE: <0-10>",
        model="stub-model",
        base_url="http://stub",
        n_concurrent=n_concurrent,
    )
    module.callback = SimpleNamespace()
    module._client = object()
    return module


def _seed_student_entry(
    tmp_path: Path,
    rank: int,
    batch_idx: int = 0,
    ext: str = ".png",
    prompt: str = "a test prompt",
) -> MediaEntry:
    """Write a stub media file on disk and return a matching ``MediaEntry``."""
    student_dir = tmp_path / "iter5_train" / "student" / "generation"
    student_dir.mkdir(parents=True, exist_ok=True)
    p = student_dir / f"frame_{rank:03d}_{batch_idx:03d}{ext}"
    p.write_bytes(b"\x00stub\x00")
    return MediaEntry(
        kind="student",
        rank=rank,
        batch_idx=batch_idx,
        path=p,
        prompt=prompt,
    )


def _mock_score_sample(score, latency: float = 0.42) -> JudgeResult:
    return JudgeResult(
        score=score,
        raw=f"FINAL_SCORE: {score}" if score is not None else "no score",
        latency_s=latency,
        error=None,
    )


def test_skip_when_no_student_entries(tmp_path: Path):
    """Empty ``entries`` (or no ``student`` kind) → score_sample never
    called and the module returns ``None``."""
    module = _make_module()
    with patch("fastgen.callbacks.vlm_logger.score_sample") as m:
        out = module.process(
            tmp_path,
            entries=[],
            iteration=5,
            group="train",
            step=0,
            data_batch=None,
            output_batch=None,
        )
    m.assert_not_called()
    assert out is None


def test_logs_row_directly_to_wandb(tmp_path: Path, monkeypatch):
    """Media present → score_sample called once per entry, all rows shipped
    via a single ``wandb.log`` of a ``wandb.Table``, and the cross-sample
    mean is returned for cross-rank scalar aggregation."""
    module = _make_module()
    entry = _seed_student_entry(tmp_path, rank=0, prompt="a dog")

    fake_run = SimpleNamespace()
    monkeypatch.setattr("fastgen.callbacks.vlm_logger.wandb.run", fake_run)

    logged: Dict[str, Any] = {}

    def _capture_log(payload, step=None):
        logged.update(payload)
        logged["__step__"] = step

    class _FakeTable:
        def __init__(self, columns, log_mode=None):
            self.columns = columns
            self.rows: List[Any] = []

        def add_data(self, *vals):
            self.rows.append(vals)

    monkeypatch.setattr("fastgen.callbacks.vlm_logger.wandb.log", _capture_log)
    monkeypatch.setattr("fastgen.callbacks.vlm_logger.wandb.Table", _FakeTable)
    # Stub ``_judge_media_cell`` directly — its real path decodes the
    # ``JudgeSample.media_b64`` and would choke on ``"x"`` since the
    # try/except fallback was removed; this test isn't covering cell
    # rendering anyway, just the row plumbing.
    monkeypatch.setattr(
        "fastgen.callbacks.vlm_logger._judge_media_cell",
        lambda sample: "<media>",
    )

    with patch(
        "fastgen.callbacks.vlm_logger.score_sample",
        return_value=_mock_score_sample(7.5),
    ) as m, patch(
        "fastgen.callbacks.vlm_logger.JudgeSample.from_path",
        return_value=JudgeSample(prompt="a dog", media_b64="x", image_mime="image/png"),
    ):
        losses = module.process(
            tmp_path,
            entries=[entry],
            iteration=5,
            group="train",
            step=0,
            data_batch=None,
            output_batch=None,
        )

    m.assert_called_once()
    assert losses == {"vlm_score": 7.5}
    table = logged["tables/vlm_scores_train_0"]
    assert isinstance(table, _FakeTable)
    assert len(table.rows) == 1
    row = dict(zip(table.columns, table.rows[0]))
    assert row["score"] == 7.5
    assert row["parse_fail"] is False
    assert row["latency_s"] == pytest.approx(0.42)
    assert row["prompt"] == "a dog"
    assert row["media"] == "<media>"
    assert logged["__step__"] == 5


def test_parse_fail_returns_empty_loss_dict(tmp_path: Path):
    """Parse fail → ``vlm_score`` omitted from the returned loss_dict so
    the running mean isn't polluted by a 0 sentinel."""
    module = _make_module()
    entry = _seed_student_entry(tmp_path, rank=0)

    with patch(
        "fastgen.callbacks.vlm_logger.score_sample",
        return_value=_mock_score_sample(None),
    ), patch(
        "fastgen.callbacks.vlm_logger.JudgeSample.from_path",
        return_value=JudgeSample(prompt="p", media_b64="x", image_mime="image/png"),
    ):
        losses = module.process(
            tmp_path,
            entries=[entry],
            iteration=10,
            group="val0",
            step=0,
            data_batch=None,
            output_batch=None,
        )

    assert losses == {}


def test_oversize_mp4_skipped_with_parse_fail(tmp_path: Path):
    """Encoded mp4 payload > max_video_bytes → empty loss_dict, no judge
    call. The check runs on the post-``target_fps`` payload (not the
    source file size) so a large source that re-encodes small still
    gets scored — and a payload that's still oversize is rejected here.
    """
    module = _make_module(media_mode="video")
    module.max_video_bytes = 4
    entry = _seed_student_entry(tmp_path, rank=0, ext=".mp4")

    # 16 chars of base64 → 12 decoded bytes, comfortably above the 4-byte cap.
    oversize_sample = JudgeSample(
        prompt="p",
        media_b64="A" * 16,
        image_mime="video/mp4",
        media_type="video",
    )

    with patch(
        "fastgen.callbacks.vlm_logger.JudgeSample.from_path",
        return_value=oversize_sample,
    ), patch("fastgen.callbacks.vlm_logger.score_sample") as m:
        out = module.process(
            tmp_path,
            entries=[entry],
            iteration=5,
            group="train",
            step=0,
            data_batch=None,
            output_batch=None,
        )

    m.assert_not_called()
    assert out == {}


def test_frame_grid_mode_passes_video_path_to_judge_sample(tmp_path: Path):
    """frame_grid mode → JudgeSample.from_path called with the .mp4 path."""
    module = _make_module(media_mode="frame_grid")
    module.max_video_bytes = 1024
    entry = _seed_student_entry(tmp_path, rank=0, ext=".mp4")

    with patch(
        "fastgen.callbacks.vlm_logger.JudgeSample.from_path",
        return_value=JudgeSample(prompt="p", media_b64="x", image_mime="image/png"),
    ) as from_path, patch(
        "fastgen.callbacks.vlm_logger.score_sample",
        return_value=_mock_score_sample(6.0),
    ):
        module.process(
            tmp_path,
            entries=[entry],
            iteration=5,
            group="train",
            step=0,
            data_batch=None,
            output_batch=None,
        )

    kwargs = from_path.call_args.kwargs
    assert kwargs["media_type"] == "frame_grid"
    assert str(from_path.call_args.args[0]).endswith("frame_000_000.mp4")


def test_caption_taken_from_entry_prompt(tmp_path: Path):
    """Each entry's ``prompt`` field flows directly into the judge call —
    no filename parsing or cross-rank lookup."""
    module = _make_module()
    entry = _seed_student_entry(tmp_path, rank=2, prompt="rank-2 prompt")

    captured: Dict[str, Any] = {}

    def _capture_from_path(*args, **kwargs):
        captured.update(kwargs)
        return JudgeSample(prompt=kwargs.get("prompt", "x"), media_b64="x", image_mime="image/png")

    with patch(
        "fastgen.callbacks.vlm_logger.score_sample",
        return_value=_mock_score_sample(5.0),
    ), patch(
        "fastgen.callbacks.vlm_logger.JudgeSample.from_path",
        side_effect=_capture_from_path,
    ):
        module.process(
            tmp_path,
            entries=[entry],
            iteration=5,
            group="train",
            step=0,
            data_batch=None,
            output_batch=None,
        )

    assert captured["prompt"] == "rank-2 prompt"


def test_non_student_entries_filtered_out(tmp_path: Path):
    """Entries with ``kind != "student"`` are skipped (e.g. teacher /
    real / reward panels are panel-visible but not judged)."""
    module = _make_module()
    student = _seed_student_entry(tmp_path, rank=0, prompt="student prompt")
    # Construct a "teacher" entry pointing at a different file (not
    # actually written — VLMLogger should skip without touching it).
    teacher_entry = MediaEntry(
        kind="teacher",
        rank=0,
        batch_idx=0,
        path=tmp_path / "teacher_does_not_exist.png",
        prompt="teacher prompt",
    )

    with patch(
        "fastgen.callbacks.vlm_logger.score_sample",
        return_value=_mock_score_sample(7.0),
    ) as m, patch(
        "fastgen.callbacks.vlm_logger.JudgeSample.from_path",
        return_value=JudgeSample(prompt="x", media_b64="x", image_mime="image/png"),
    ):
        losses = module.process(
            tmp_path,
            entries=[teacher_entry, student],
            iteration=5,
            group="train",
            step=0,
            data_batch=None,
            output_batch=None,
        )

    # Only the student entry was scored.
    m.assert_called_once()
    assert losses == {"vlm_score": 7.0}


def test_max_samples_caps_scoring(tmp_path: Path):
    """``max_samples`` truncates the student-entries list before scoring,
    so only the first N are sent to the judge regardless of how many
    files the parent callback wrote."""
    module = _make_module()
    module.max_samples = 2
    entries = [
        _seed_student_entry(tmp_path, rank=0),
        _seed_student_entry(tmp_path, rank=1),
        _seed_student_entry(tmp_path, rank=2),
        _seed_student_entry(tmp_path, rank=3),
    ]

    score_iter = iter([7.0, 8.0])

    def _next_score(*args, **kwargs):
        return _mock_score_sample(next(score_iter))

    with patch(
        "fastgen.callbacks.vlm_logger.score_sample",
        side_effect=_next_score,
    ) as m, patch(
        "fastgen.callbacks.vlm_logger.JudgeSample.from_path",
        return_value=JudgeSample(prompt="p", media_b64="x", image_mime="image/png"),
    ):
        losses = module.process(
            tmp_path,
            entries=entries,
            iteration=5,
            group="train",
            step=0,
            data_batch=None,
            output_batch=None,
        )

    assert m.call_count == 2  # only 2 of the 4 entries scored
    assert losses == {"vlm_score": 7.5}  # mean(7, 8)


def test_multi_sample_average_across_rows(tmp_path: Path, monkeypatch):
    """Multiple student entries → every successful score gets averaged
    into one ``vlm_score`` mean and every row appears in the same
    shipped table."""
    module = _make_module()
    entries = [
        _seed_student_entry(tmp_path, rank=0),
        _seed_student_entry(tmp_path, rank=1),
        _seed_student_entry(tmp_path, rank=2),
    ]

    fake_run = SimpleNamespace()
    monkeypatch.setattr("fastgen.callbacks.vlm_logger.wandb.run", fake_run)

    logged: Dict[str, Any] = {}

    def _capture_log(payload, step=None):
        logged.update(payload)

    class _FakeTable:
        def __init__(self, columns, log_mode=None):
            self.columns = columns
            self.rows: List[Any] = []

        def add_data(self, *vals):
            self.rows.append(vals)

    monkeypatch.setattr("fastgen.callbacks.vlm_logger.wandb.log", _capture_log)
    monkeypatch.setattr("fastgen.callbacks.vlm_logger.wandb.Table", _FakeTable)
    monkeypatch.setattr(
        "fastgen.callbacks.vlm_logger._judge_media_cell",
        lambda sample: "<media>",
    )

    score_iter = iter([7.0, 8.0, 9.0])

    def _next_score(*args, **kwargs):
        return _mock_score_sample(next(score_iter))

    with patch(
        "fastgen.callbacks.vlm_logger.score_sample",
        side_effect=_next_score,
    ), patch(
        "fastgen.callbacks.vlm_logger.JudgeSample.from_path",
        return_value=JudgeSample(prompt="p", media_b64="x", image_mime="image/png"),
    ):
        losses = module.process(
            tmp_path,
            entries=entries,
            iteration=5,
            group="train",
            step=0,
            data_batch=None,
            output_batch=None,
        )

    table = logged["tables/vlm_scores_train_0"]
    assert len(table.rows) == 3
    assert losses == {"vlm_score": 8.0}  # mean(7, 8, 9)


def test_rows_without_a_score_or_media_fit_a_real_wandb_table(tmp_path: Path, monkeypatch):
    """A parse failure and an oversize clip leave ``None`` cells, which a real ``wandb.Table`` accepts after a
    scored row; a string there makes ``add_data`` raise on rank 0 while the other ranks wait."""
    import wandb

    module = _make_module(media_mode="video")
    module.max_video_bytes = 64
    entries = [_seed_student_entry(tmp_path, rank=rank, ext=".mp4") for rank in range(3)]
    # 16 base64 chars decode to 12 bytes (under the cap); 256 decode to 192 bytes (oversize, no judge call).
    payloads = {entries[0].path: "A" * 16, entries[1].path: "A" * 16, entries[2].path: "A" * 256}
    scores = iter([7.5, None])

    def _from_path(path, **kwargs):
        return JudgeSample(prompt="p", media_b64=payloads[path], image_mime="video/mp4", media_type="video")

    logged: Dict[str, Any] = {}
    monkeypatch.setattr("fastgen.callbacks.vlm_logger.wandb.run", SimpleNamespace())
    monkeypatch.setattr("fastgen.callbacks.vlm_logger.wandb.log", lambda payload, step=None: logged.update(payload))

    with patch("fastgen.callbacks.vlm_logger.JudgeSample.from_path", side_effect=_from_path), patch(
        "fastgen.callbacks.vlm_logger.score_sample",
        side_effect=lambda *args, **kwargs: _mock_score_sample(next(scores)),
    ):
        losses = module.process(
            tmp_path,
            entries=entries,
            iteration=5,
            group="train",
            step=0,
            data_batch=None,
            output_batch=None,
        )

    table = logged["tables/vlm_scores_train_0"]
    assert isinstance(table, wandb.Table)
    rows = [dict(zip(table.columns, row)) for row in table.data]
    assert [row["score"] for row in rows] == [7.5, None, None]
    assert [type(row["media"]) for row in rows] == [wandb.Video, wandb.Video, type(None)]
    assert [row["parse_fail"] for row in rows] == [False, True, True]
    assert losses == {"vlm_score": 7.5}


# ===========================================================================
# Package-level logging filters (fastgen/__init__.py)
# ===========================================================================


def test_torchao_optional_extension_noise_is_filtered(caplog):
    torchao_logger = logging.getLogger("torchao")
    with caplog.at_level(logging.WARNING, logger="torchao"):
        torchao_logger.warning(
            "Failed to load /site-packages/torchao/_C_mxfp8.cpython-310-x86_64-linux-gnu.so: incompatible ABI"
        )
        torchao_logger.warning("Failed to load /site-packages/torchao/_C_cutlass_90a.abi3.so: incompatible ABI")
        torchao_logger.warning("A genuine TorchAO warning")

    assert [record.getMessage() for record in caplog.records] == ["A genuine TorchAO warning"]
