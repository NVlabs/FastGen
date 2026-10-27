# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``VLMLogger`` — pluggable VLM-as-judge ``LogModule`` for ``WandbCallback``.

Runs **only on rank 0** (the parent ``WandbCallback`` only invokes log
modules on rank 0). Each call to ``process`` receives a flat
``List[MediaEntry]`` covering every rank's writes — already gathered
cross-rank by the parent — filters it down to the ``"student"`` kind,
scores each entry via the judge (in parallel via a thread pool when
``n_concurrent > 1``), ships a single per-step ``wandb.Table``, and
returns the cross-sample mean ``vlm_score`` for cross-rank scalar
aggregation through the parent's ``_LossDictRecord``.

The judge is any OpenAI-compatible endpoint (``base_url`` or the
``VLM_BASE_URL`` env var, bearer token in ``OPENAI_API_KEY``), and
``model`` must be an id that endpoint serves — override the Gemini ids
of the ``VLM_IMG_CALLBACK`` / ``VLM_VID_CALLBACK`` presets for gateways
that name models differently. ``media_mode`` selects what the judge sees
per sample: ``"image"`` (a single still), ``"frame_grid"`` (one image
tiling ``video_frames`` evenly spaced frames), or ``"video"`` (an inline
mp4 subsampled to ``target_fps``).
"""

from __future__ import annotations

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import wandb
from PIL import Image

from fastgen.callbacks.wandb import LogModule, MediaEntry
from fastgen.utils.vlm_judge import (
    JudgeResult,
    JudgeSample,
    load_vlm_client,
    read_master_prompt,
    score_sample,
)
from fastgen.utils import logging_utils as logger

if TYPE_CHECKING:
    from fastgen.callbacks.wandb import WandbCallback


_VLM_MEDIA_MODES = ("image", "frame_grid", "video")


def _judge_media_cell(sample: Optional[JudgeSample]) -> Any:
    """Build a ``wandb.Table`` cell showing exactly what the judge scored.

    Image-MIME samples (``image`` / ``frame_grid``) decode the base64
    payload as a PIL image wrapped in ``wandb.Image``; video-MIME
    samples wrap the mp4 bytes in ``wandb.Video`` (``BytesIO``,
    ``format="mp4"``) so the table shows the post-``target_fps`` clip
    the judge actually saw, not the source file. Returns ``None`` (an
    empty cell; ``wandb.Table`` rejects a string in a media column) for
    the oversize-mp4 short-circuit, whose media the judge never saw.
    """
    if sample is None:
        return None
    data = base64.b64decode(sample.media_b64)
    if sample.image_mime.startswith("image/"):
        return wandb.Image(Image.open(io.BytesIO(data)))
    if sample.image_mime.startswith("video/"):
        return wandb.Video(io.BytesIO(data), format="mp4")
    return None


class VLMLogger(LogModule):
    """``LogModule`` that scores the per-step student media via a VLM judge.

    Constructor knobs (all optional):

    * ``media_mode`` — ``"image"`` | ``"frame_grid"`` | ``"video"``.
    * ``master_prompt`` / ``master_prompt_file`` — judge instruction
      override (inline string or path to a text file; mutually exclusive).
      Leave both ``None`` (the default) to let ``score_sample`` fall back
      to ``DEFAULT_MASTER_PROMPTS[media_mode]`` per call; set one to
      override the rubric.
    * ``video_frames`` — frames per ``frame_grid`` tile (default 4).
    * ``target_fps`` — pre-subsamples mp4s before sending to the judge.
      Defaults to ``8.0``: training-time clips are short with
      fast-changing motion, and giving the judge 8 frames/sec offers a
      richer temporal signal than Gemini's 1 FPS server-side default
      while staying well under the 20 MB inline-data cap for typical
      sub-10-second clips. Set to ``None`` to disable subsampling
      (``video`` ships source bytes verbatim; ``frame_grid`` decodes
      every frame). For ``video`` it bakes the fps into the mp4 (the
      only way to control inline-clip sampling on an OpenAI-compatible
      endpoint). For ``frame_grid`` it bounds decode memory. Both modes
      pass ``min_frames=video_frames`` to ``decode_video_full``, so
      even a very low ``target_fps`` (e.g., ``1`` on a 1-second clip)
      still keeps at least ``video_frames`` frames — the grid gets a
      full tile and the inline mp4 still looks like a video, not a
      still.
    * ``model`` — model id as the endpoint expects it (e.g.
      ``"gemini-3.1-pro-preview"``). Required.
    * ``base_url`` — OpenAI-compatible endpoint URL. Falls back to the
      ``VLM_BASE_URL`` env var when unset; raises if both are unset.
      The bearer token is read from the ``OPENAI_API_KEY`` environment
      variable (populated by ``set_env_vars`` from
      ``credentials/vlm_api_token.txt`` at app startup).
    * ``max_tokens`` / ``temperature`` / ``image_longest_side`` — judge
      call parameters.
    * ``max_video_bytes`` — cap on the *encoded* inline-video payload;
      if the mp4 that the judge would consume after subsampling exceeds
      this, short-circuit to ``parse_fail`` without a judge call.
      Defaults to ``20 MB``.
    * ``max_samples`` — cap student entries scored per call (``None``
      = score all). Independent of ``WandbCallback.max_ranks``.
    * ``n_concurrent`` — judge-call thread pool size (default 4).
    """

    def __init__(
        self,
        model: str,
        media_mode: str = "image",
        master_prompt: Optional[str] = None,
        master_prompt_file: Optional[str] = None,
        video_frames: int = 4,
        target_fps: Optional[float] = 8.0,
        base_url: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        image_longest_side: int = 768,
        max_video_bytes: int = 20 * 1024 * 1024,
        max_samples: Optional[int] = None,
        n_concurrent: int = 4,
    ) -> None:
        super().__init__()
        if not model:
            raise ValueError("VLMLogger: model is required")
        if media_mode not in _VLM_MEDIA_MODES:
            raise ValueError(f"VLMLogger: unsupported media_mode={media_mode!r}")
        if max_samples is not None and max_samples < 1:
            raise ValueError(f"VLMLogger.max_samples must be None or >= 1, got {max_samples}")
        if n_concurrent < 1:
            raise ValueError(f"VLMLogger.n_concurrent must be >= 1, got {n_concurrent}")
        if target_fps is not None and target_fps <= 0:
            raise ValueError(f"VLMLogger.target_fps must be None or > 0, got {target_fps}")
        resolved_base_url = base_url or os.environ.get("VLM_BASE_URL")
        if not resolved_base_url:
            raise ValueError(
                "VLMLogger: base_url not set. Pass base_url=... or set the VLM_BASE_URL environment variable."
            )

        self.model = model
        self.media_mode = media_mode
        self.master_prompt = read_master_prompt(master_prompt, master_prompt_file)
        self.video_frames = video_frames
        self.target_fps = target_fps
        self.base_url = resolved_base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.image_longest_side = image_longest_side
        self.max_video_bytes = max_video_bytes
        self.max_samples = max_samples
        self.n_concurrent = n_concurrent

        self._client: Optional[Any] = None
        self._init_error: Optional[str] = None

    # ----- LogModule lifecycle (rank 0 only) ---------------------------------

    def setup(self, callback: "WandbCallback") -> None:
        super().setup(callback)
        logger.info(
            f"vlm: VLMLogger enabled "
            f"(media_mode={self.media_mode}, max_samples={self.max_samples}, "
            f"n_concurrent={self.n_concurrent})"
        )

    def teardown(self) -> None:
        """Close the OpenAI / httpx client and drop our reference.

        Called from ``WandbCallback.on_validation_end`` and
        ``on_app_end`` so persistent httpx connection-pool state
        (sockets, internal locks, transport threads) does not survive
        into the next DataLoader worker fork. Sticky ``_init_error``
        survives teardown — a permanent init failure stays sticky.
        """
        client = self._client
        if client is None:
            return
        try:
            client.close()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"vlm: client.close() raised — {exc}")
        self._client = None

    # ----- Per-step processing (rank 0 only) ---------------------------------

    def process(
        self,
        save_dir: Path,
        entries: List[MediaEntry],
        iteration: int,
        group: str,
        step: int,
        data_batch: Optional[Dict[str, Any]],
        output_batch: Optional[Dict[str, Any]],
        gathered: Optional[List[Any]] = None,
    ) -> Optional[Dict[str, float]]:
        student_entries = [e for e in entries if e.kind == "student"]
        if self.max_samples is not None:
            student_entries = student_entries[: self.max_samples]
        if not student_entries:
            return None
        if not self._init_client():
            return None

        # Score every student entry — each call is blocking on a network
        # round-trip, so a thread pool gives us roughly N× speedup at no
        # GIL cost. ``_score_entry`` returns ``(entry, result, sample)``
        # (``sample`` is ``None`` on the oversize-mp4 short-circuit, which
        # skips the judge) or ``None`` when encoding or scoring fails.
        results: List[tuple[MediaEntry, JudgeResult, Optional[JudgeSample]]] = []
        if self.n_concurrent > 1 and len(student_entries) > 1:
            with ThreadPoolExecutor(max_workers=self.n_concurrent) as pool:
                for r in pool.map(self._score_entry, student_entries):
                    if r is not None:
                        results.append(r)
        else:
            for e in student_entries:
                r = self._score_entry(e)
                if r is not None:
                    results.append(r)

        # Build the table + collect parsed scores. Only construct row
        # cells when ``wandb.run`` is alive — ``_judge_media_cell`` calls
        # ``wandb.Image`` which opens the file and we don't want to pay
        # that cost (or its PIL probe failure on test stubs) when there's
        # nothing to ship.
        log_rows = wandb.run is not None
        rows: List[List[Any]] = []
        scores: List[float] = []
        for entry, result, sample in results:
            if log_rows:
                rows.append(
                    [
                        entry.rank,
                        entry.batch_idx,
                        entry.prompt,
                        # What the judge actually saw — tiled grid PNG for
                        # ``frame_grid``, resized PNG for ``image``,
                        # subsampled mp4 for ``video``. ``None`` for the
                        # oversize-mp4 short-circuit (no judge call).
                        _judge_media_cell(sample),
                        # ``None`` for a parse failure: the table rejects mixed column types.
                        result.score,
                        result.parse_fail,
                        result.latency_s,
                        (result.raw or "")[:4000],
                    ]
                )
            if result.score is not None:
                scores.append(float(result.score))

        if rows:
            columns = [
                "rank",
                "batch_idx",
                "prompt",
                "media",
                "score",
                "parse_fail",
                "latency_s",
                "raw",
            ]
            table = wandb.Table(columns=columns, log_mode="INCREMENTAL")
            for row in rows:
                table.add_data(*row)
            wandb.log(
                {f"tables/vlm_scores_{group}_{step}": table},
                step=iteration,
            )

        if not scores:
            return {}
        return {"vlm_score": sum(scores) / len(scores)}

    # ----- Internals ---------------------------------------------------------

    def _score_entry(
        self,
        entry: MediaEntry,
    ) -> Optional[tuple[MediaEntry, JudgeResult, Optional[JudgeSample]]]:
        """Score one ``MediaEntry`` via the judge.

        Returns ``(entry, result, sample)`` — ``sample`` is the encoded
        ``JudgeSample`` the judge actually consumed (used by
        ``_judge_media_cell`` to render the table cell with what the
        judge saw) or ``None`` on the oversize-mp4 short-circuit (no
        judge call was made, so the media cell is empty). Returns the
        outer ``None`` when encoding or the judge call fails (the
        warning is already logged)."""
        prompt = entry.prompt or "(no prompt)"

        try:
            sample = JudgeSample.from_path(
                entry.path,
                prompt=prompt,
                media_type=self.media_mode,
                n_frames=self.video_frames,
                longest_side=self.image_longest_side,
                target_fps=self.target_fps,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"vlm: rank {entry.rank} encode failed — {exc}")
            return None

        # Check the *encoded* payload size, not the source file: with
        # ``target_fps`` set, a large source mp4 can re-encode well
        # under the cap and would otherwise be falsely rejected.
        if self.media_mode == "video":
            payload_bytes = len(sample.media_b64) * 3 // 4
            if payload_bytes > self.max_video_bytes:
                logger.warning(
                    f"vlm: rank {entry.rank}: encoded mp4 {entry.path.name} is "
                    f"{payload_bytes / 1e6:.1f} MB "
                    f"(> {self.max_video_bytes / 1e6:.1f} MB cap) "
                    f"— skip with parse_fail"
                )
                return (
                    entry,
                    JudgeResult(
                        score=None,
                        raw="",
                        latency_s=0.0,
                        error=f"oversize_mp4:{payload_bytes}",
                    ),
                    None,
                )

        try:
            result = score_sample(
                self._client,
                self.model,
                sample,
                master_prompt=self.master_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"vlm: rank {entry.rank} score failed — {exc}")
            return None

        if result.score is not None:
            snippet = (result.raw or "")[-240:].replace("\n", " | ")
            logger.debug(f"vlm: rank {entry.rank} parsed={result.score} tail={snippet!r}")
        else:
            snippet = (result.raw or "<empty>")[:400].replace("\n", " | ")
            logger.warning(f"vlm: rank {entry.rank} parse fail — raw={snippet!r}")
        return entry, result, sample

    def _init_client(self) -> bool:
        """Lazy-init the OpenAI client on rank 0's first scoring call."""
        if self._client is not None:
            return True
        if self._init_error is not None:
            return False
        try:
            self._client = load_vlm_client(self.base_url)
        except Exception as exc:  # pylint: disable=broad-except
            self._init_error = repr(exc)
            logger.warning(f"vlm: client init failed — scoring disabled. {exc}")
            return False
        logger.debug(f"vlm: client init ok (model={self.model}, base_url={self.base_url})")
        return True
