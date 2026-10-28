# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone VLM-as-judge core.

Pure judge utilities: OpenAI-compatible client construction from ``base_url``
and the ``OPENAI_API_KEY`` environment variable, media encoding (PIL from
tensor, frame-grid tile, base64 PNG), request payload assembly, score parsing,
and the blocking ``call_judge`` / ``score_sample`` helpers.

Intentionally independent from ``fastgen.methods`` / ``fastgen.utils.
distributed`` / any training-time state so it can be imported from:

* ``fastgen/callbacks/vlm_logger.py`` (``VLMLogger`` per-rank training-
  time scoring as a ``WandbCallback`` log module)
* ``scripts/inference/vlm_infer.py`` (standalone CLI over a JSONL of
  prompt/media pairs)
"""

from __future__ import annotations

import base64
import io
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v3 as iio
import torch
import torchvision
from PIL import Image

from fastgen.utils.tensor import pixels_to_uint8

DEFAULT_MASTER_PROMPT_IMG = (
    "You are an expert image grader. You will receive a text prompt and a "
    "generated image. Score how well the image matches the prompt and its "
    "overall aesthetic quality on an integer scale from 0 (poor) to 10 "
    "(excellent). Consider prompt following first, then visual quality.\n"
    "Reply in EXACTLY this format. The FIRST line MUST be the final score, "
    "so the score survives any downstream truncation. Write the score first, "
    "then a short justification.\n"
    "FINAL_SCORE: <integer 0-10>\n"
    "Justification: <one sentence, <=40 words>"
)


DEFAULT_MASTER_PROMPT_VID_TILE = (
    "You are an expert video grader. You will receive a text prompt and a "
    "tiled grid of evenly-spaced frames sampled from a generated short "
    "video clip. Read the grid left-to-right, top-to-bottom as a temporal "
    "sequence. Grade the clip on an integer scale from 0 (poor) to 10 "
    "(excellent) using four axes that sum to 10:\n"
    "  * prompt adherence (0-3): does the clip depict what the prompt asks?\n"
    "  * motion coherence (0-3): is the motion across frames believable / "
    "continuous rather than stuttery or duplicated?\n"
    "  * temporal consistency (0-2): do subjects / colours / geometry stay "
    "stable across frames (no flicker, no morphing artifacts)?\n"
    "  * aesthetic quality (0-2): framing, lighting, sharpness of each frame.\n"
    "Reply in EXACTLY this format. The FIRST line MUST be the final score, "
    "so the score survives any downstream truncation. Write the score first, "
    "then a one-line per-axis justification.\n"
    "FINAL_SCORE: <integer 0-10>\n"
    "Breakdown: prompt=<0-3>, motion=<0-3>, consistency=<0-2>, "
    "aesthetic=<0-2> — <=60 words total justification>"
)


# Companion rubric for inline-video mode (``media_mode="video"``), where the
# judge receives an mp4 inline-data part rather than a tiled frame grid. Same
# axes, scoring scale, and reply contract as ``DEFAULT_MASTER_PROMPT_VID_TILE`` so
# downstream parsing is identical; only the "tiled grid of evenly-spaced
# frames" framing is dropped to match what the judge actually sees.
DEFAULT_MASTER_PROMPT_VID = (
    "You are an expert video grader. You will receive a text prompt and a "
    "generated short video clip. Watch the clip as a temporal sequence and "
    "grade it on an integer scale from 0 (poor) to 10 (excellent) using "
    "four axes that sum to 10:\n"
    "  * prompt adherence (0-3): does the clip depict what the prompt asks?\n"
    "  * motion coherence (0-3): is the motion believable / continuous "
    "rather than stuttery or duplicated?\n"
    "  * temporal consistency (0-2): do subjects / colours / geometry stay "
    "stable over time (no flicker, no morphing artifacts)?\n"
    "  * aesthetic quality (0-2): framing, lighting, sharpness.\n"
    "Reply in EXACTLY this format. The FIRST line MUST be the final score, "
    "so the score survives any downstream truncation. Write the score first, "
    "then a one-line per-axis justification.\n"
    "FINAL_SCORE: <integer 0-10>\n"
    "Breakdown: prompt=<0-3>, motion=<0-3>, consistency=<0-2>, "
    "aesthetic=<0-2> — <=60 words total justification>"
)


# Single source of truth for the default rubric per ``JudgeSample.from_path``
# ``media_type``. Keyed by the same three concrete strings the loader accepts.
DEFAULT_MASTER_PROMPTS: Dict[str, str] = {
    "image": DEFAULT_MASTER_PROMPT_IMG,
    "frame_grid": DEFAULT_MASTER_PROMPT_VID_TILE,
    "video": DEFAULT_MASTER_PROMPT_VID,
}


def read_master_prompt(
    master_prompt: Optional[str],
    master_prompt_file: Optional[Path | str],
) -> Optional[str]:
    """Resolve an explicit master-prompt override from one of two sources.

    Returns the inline ``master_prompt`` if given, else the contents of
    ``master_prompt_file`` (stripped), else ``None`` so callers can fall
    back to ``DEFAULT_MASTER_PROMPTS[sample.media_type]`` via
    ``score_sample``. Raises ``ValueError`` if both are supplied.
    """
    if master_prompt and master_prompt_file:
        raise ValueError("master_prompt and master_prompt_file are mutually exclusive")
    if master_prompt:
        return master_prompt
    if master_prompt_file:
        return Path(master_prompt_file).read_text(encoding="utf-8").strip()
    return None


# Primary format the master prompt instructs the judge to emit. Tolerant of
# the judge writing "FINAL SCORE", "FinalScore", or wrapping the number in
# markdown bold (`**7**`).
# A score may end a sentence ("7.5."), but may not run into a word or further digits ("7abc", "7.5.3").
_SCORE_NUMBER = r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[+-]?(?:inf(?:inity)?|nan))(?!\w|\.[0-9])"
SCORE_STRICT_RE = re.compile(r"FINAL[\s_]*SCORE\s*:?\s*\**\s*" + _SCORE_NUMBER, re.IGNORECASE)
# Labelled fallback: "Score: 7", "Rating = 8/10".
SCORE_LABELED_RE = re.compile(
    r"(?:overall\s+)?(?:score|rating|grade)\s*[:=]\s*\**\s*" + _SCORE_NUMBER,
    re.IGNORECASE,
)
# Trailing-line fallback: a number (optionally wrapped in markdown or followed
# by "/10") as the final non-empty content on a line.
SCORE_TRAILING_RE = re.compile(r"^\**\s*" + _SCORE_NUMBER + r"\s*\**\s*(?:/\s*10)?\.?\s*$", re.IGNORECASE)

# Model families for which some OpenAI-compatible gateways/providers reject the
# `temperature` kwarg (provider-native APIs and some newer reasoning models refuse
# it). Matched case-insensitively against substrings of the resolved model string.
TEMPERATURE_UNSUPPORTED_MODEL_SUBSTRINGS: Tuple[str, ...] = ("claude", "anthropic")


def model_accepts_temperature(model: str) -> bool:
    m = (model or "").lower()
    return not any(s in m for s in TEMPERATURE_UNSUPPORTED_MODEL_SUBSTRINGS)


def parse_score(text: str) -> Optional[float]:
    """Read a finite score in [0, 10], tolerating fractional replies to the integer rubric.

    An out-of-range or nonfinite labelled score is final: do not reinterpret a breakdown as the score.
    """
    if not text:
        return None
    for pattern in (SCORE_STRICT_RE, SCORE_LABELED_RE):
        m = pattern.search(text)
        if m:
            score = float(m.group(1))
            return score if math.isfinite(score) and 0 <= score <= 10 else None
    for line in reversed(text.strip().splitlines()):
        m = SCORE_TRAILING_RE.search(line.strip())
        if m:
            score = float(m.group(1))
            return score if math.isfinite(score) and 0 <= score <= 10 else None
    return None


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _downscale_to_longest_side(img: Image.Image, longest_side: int) -> Image.Image:
    """Aspect-preserving LANCZOS downscale so ``max(img.size) <= longest_side``.
    No-op when the image already fits (never upscales)."""
    if max(img.size) <= longest_side:
        return img
    ratio = longest_side / max(img.size)
    return img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)


def video_to_frame_grid(
    video: torch.Tensor,
    n_frames: int,
    longest_side: int = 768,
) -> Image.Image:
    """Tile ``n_frames`` evenly-spaced frames from a ``[C,T,H,W]`` video
    in ``[-1, 1]`` into a square-ish PIL grid."""
    if video.ndim != 4:
        raise ValueError(f"vlm_score: expected [C,T,H,W] video, got {tuple(video.shape)}")
    C, T, H, W = video.shape
    if C == 1:
        video = video.repeat(3, 1, 1, 1)
        C = 3
    if C != 3:
        raise ValueError(f"vlm_score: frame_grid expects 3 channels, got {C}")
    n = max(1, min(int(n_frames), T))
    idxs = torch.linspace(0, T - 1, n).round().long().tolist()
    frames = video[:, idxs, :, :]  # [C, n, H, W]
    frames = pixels_to_uint8(frames.detach()).cpu()
    frames = frames.permute(1, 0, 2, 3).contiguous()  # [n, C, H, W]
    nrow = max(1, int(math.ceil(math.sqrt(n))))
    grid = torchvision.utils.make_grid(frames, nrow=nrow, padding=2, pad_value=255)
    arr = grid.permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    return _downscale_to_longest_side(img, longest_side)


def build_judge_messages(
    master_prompt: str,
    user_prompt: str,
    media_b64: str,
    image_mime: str = "image/png",
) -> List[Dict[str, Any]]:
    """Assemble the ``messages`` array for an OpenAI-compatible chat call."""
    return [
        {"role": "system", "content": master_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt or "(no prompt)"},
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{media_b64}"}},
            ],
        },
    ]


def load_vlm_client(base_url: str) -> Any:
    """Build an OpenAI-compatible client pointing at ``base_url``.

    The bearer token is read from the ``OPENAI_API_KEY`` environment
    variable (the OpenAI SDK's native convention; populated by
    ``set_env_vars`` from ``credentials/vlm_api_token.txt`` at app
    startup). Raises if the env var is unset.
    """
    if not base_url:
        raise ValueError("vlm: base_url is required")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "vlm: OPENAI_API_KEY env var not set. Place a token at "
            "credentials/vlm_api_token.txt (loaded by set_env_vars) "
            "or export OPENAI_API_KEY."
        )
    from openai import OpenAI  # local import — kept off the module import path.

    return OpenAI(base_url=base_url)


def call_judge(
    client: Any,
    model: str,
    master_prompt: str,
    user_prompt: str,
    media_b64: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    image_mime: str = "image/png",
) -> str:
    """Blocking judge call. Returns the stripped reply text (possibly empty)."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": build_judge_messages(master_prompt, user_prompt, media_b64, image_mime=image_mime),
        "max_tokens": max_tokens,
    }
    if model_accepts_temperature(model):
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Shared sample / result types and helpers consumed by both the
# training-time callback path (per-rank scoring in
# `vlm_logger.VLMLogger`) and the CLI path
# (`scripts/inference/vlm_infer.py`).

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

_VIDEO_MIME_BY_EXT = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "mkv": "video/x-matroska",
    "avi": "video/x-msvideo",
}


def infer_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "frame_grid"
    raise ValueError(f"cannot infer media_type from extension: {path}")


def load_image_b64(path: Path, longest_side: int = 768) -> str:
    with Image.open(path) as raw:
        img = _downscale_to_longest_side(raw.convert("RGB"), longest_side)
        return pil_to_base64_png(img)


def load_frame_grid_b64(
    path: Path,
    n_frames: int = 4,
    longest_side: int = 768,
    target_fps: Optional[float] = None,
) -> str:
    """Sample ``n_frames`` evenly-spaced frames and return a base64 PNG tile.

    ``target_fps`` bounds peak decode memory on long clips;
    ``min_frames=n_frames`` is passed so the floor in
    ``decode_video_full`` keeps the grid full on short clips.
    """
    from fastgen.datasets.decoders import decode_video_full

    arr = decode_video_full(
        path.name,
        path.read_bytes(),
        output_format="torch",
        target_fps=target_fps,
        min_frames=n_frames,
    )
    if arr is None or arr.shape[0] == 0:
        raise ValueError(f"empty video: {path}")
    # ``[T,H,W,C]`` uint8 → ``[C,T,H,W]`` float in ``[-1, 1]``.
    tensor = arr.permute(3, 0, 1, 2).contiguous().to(torch.float32)
    tensor = tensor.div_(127.5).sub_(1.0)
    grid = video_to_frame_grid(tensor, n_frames=n_frames, longest_side=longest_side)
    return pil_to_base64_png(grid)


def load_video_b64(
    path: Path,
    target_fps: Optional[float] = None,
    min_frames: Optional[int] = None,
) -> Tuple[str, str]:
    """Return ``(b64, mime)`` for an mp4 inline-data part.

    ``target_fps=None`` ships the source bytes verbatim. With
    ``target_fps`` set we decode at that cadence and re-encode to mp4
    bytes (via ``imageio`` to a ``BytesIO``) — the only way to control
    inline-clip sampling on OpenAI-compatible endpoints, and useful to keep
    payloads under Gemini's 20 MB inline-data cap (paired with
    ``VLMLogger.max_video_bytes``). ``min_frames`` mirrors
    ``decode_video_full``'s floor and stops a low ``target_fps`` from
    starving short clips down to a single frame (e.g., a 1-second
    source at ``target_fps=1`` would otherwise ship a still-image mp4
    to the judge).
    """
    if target_fps is None:
        data = path.read_bytes()
        ext = path.suffix.lower().lstrip(".")
        mime = _VIDEO_MIME_BY_EXT.get(ext, "video/mp4")
        return base64.b64encode(data).decode("ascii"), mime

    from fastgen.datasets.decoders import decode_video_full

    arr = decode_video_full(
        path.name,
        path.read_bytes(),
        output_format="numpy",
        target_fps=target_fps,
        min_frames=min_frames,
    )
    if arr is None or arr.shape[0] == 0:
        raise ValueError(f"empty video: {path}")
    # Tag the mp4 at the same cadence as the decode pass to preserve
    # wall-clock duration (Gemini's sampler is time-based).
    buf = io.BytesIO()
    iio.imwrite(
        buf,
        arr,
        extension=".mp4",
        fps=target_fps,
        codec="libx264",
        output_params=["-loglevel", "quiet", "-hide_banner", "-nostats"],
    )
    return base64.b64encode(buf.getvalue()).decode("ascii"), "video/mp4"


@dataclass
class JudgeSample:
    """One sample to score: an OpenAI-compatible ``(prompt, base64 media,
    mime)`` triple plus the resolved ``media_type`` (one of the keys in
    ``DEFAULT_MASTER_PROMPTS``) used by ``score_sample`` to pick a default
    rubric when no ``master_prompt`` override is given."""

    prompt: str
    media_b64: str
    image_mime: str = "image/png"
    media_type: str = "image"

    @classmethod
    def from_path(
        cls,
        media_path: Path | str,
        prompt: str,
        media_type: str = "auto",
        n_frames: int = 4,
        longest_side: int = 768,
        target_fps: Optional[float] = None,
        base_dir: Optional[Path] = None,
    ) -> "JudgeSample":
        path = Path(media_path)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        if media_type == "auto":
            media_type = infer_media_type(path)
        if media_type == "image":
            return cls(
                prompt=prompt,
                media_b64=load_image_b64(path, longest_side=longest_side),
                image_mime="image/png",
                media_type=media_type,
            )
        if media_type == "frame_grid":
            return cls(
                prompt=prompt,
                media_b64=load_frame_grid_b64(
                    path,
                    n_frames=n_frames,
                    longest_side=longest_side,
                    target_fps=target_fps,
                ),
                image_mime="image/png",
                media_type=media_type,
            )
        if media_type == "video":
            b64, mime = load_video_b64(
                path,
                target_fps=target_fps,
                min_frames=n_frames,
            )
            return cls(prompt=prompt, media_b64=b64, image_mime=mime, media_type=media_type)
        raise ValueError(f"unsupported media_type: {media_type}")


@dataclass
class JudgeResult:
    """Outcome of one ``score_sample`` call."""

    score: Optional[float]
    raw: str
    latency_s: float
    error: Optional[str] = None

    @property
    def parse_fail(self) -> bool:
        return self.score is None


def score_sample(
    client: Any,
    model: str,
    sample: JudgeSample,
    master_prompt: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> JudgeResult:
    """Blocking, thread-safe judge scoring. Captures latency, parse status,
    and any transport-layer error so callers can log without re-raising.

    ``master_prompt=None`` falls back to ``DEFAULT_MASTER_PROMPTS[sample.
    media_type]`` so callers (e.g. ``scripts/inference/vlm_infer.py``) that
    dispatch per-record on ``media_type`` don't have to repeat the lookup.
    A ``KeyError`` is intentional if ``sample.media_type`` isn't a known
    key — better to fail loudly than silently pick a mismatched rubric.
    """
    if master_prompt is None:
        master_prompt = DEFAULT_MASTER_PROMPTS[sample.media_type]
    start = time.monotonic()
    try:
        raw = call_judge(
            client,
            model,
            master_prompt=master_prompt,
            user_prompt=sample.prompt,
            media_b64=sample.media_b64,
            max_tokens=max_tokens,
            temperature=temperature,
            image_mime=sample.image_mime,
        )
    except Exception as e:
        latency = time.monotonic() - start
        return JudgeResult(
            score=None,
            raw="",
            latency_s=round(latency, 3),
            error=f"judge call failed: {type(e).__name__}: {e}",
        )
    latency = time.monotonic() - start
    return JudgeResult(
        score=parse_score(raw),
        raw=raw,
        latency_s=round(latency, 3),
        error=None,
    )
