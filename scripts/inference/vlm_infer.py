# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone VLM-as-judge inference over a JSONL of prompt/media pairs.

Reads ``{prompt, media_path, [media_type], [id]}`` records from a JSONL file,
scores each via an OpenAI-compatible VLM endpoint, and writes each input plus
``{score, raw_response, latency_s, parse_fail}`` to an output JSONL. No
training loop, no fastgen configs, no torch-distributed — reuses the pure
helpers in ``fastgen.utils.vlm_judge``.

Run from the fastgen checkout root so ``fastgen.utils.vlm_judge`` imports:

    VLM_BASE_URL=https://<your-openai-compatible-endpoint>/v1 \
    PYTHONPATH=$(pwd) python scripts/inference/vlm_infer.py \
        --jsonl prompts.jsonl --output scores.jsonl \
        --model gemini-3.1-pro-preview \
        --n-concurrent 4

The bearer token is loaded by ``set_env_vars`` from
``credentials/vlm_api_token.txt`` into ``OPENAI_API_KEY``.

``media_type`` per record may be ``image`` | ``frame_grid`` | ``video`` |
``auto`` (default) which infers from the media extension. When neither
``--master-prompt`` nor ``--master-prompt-file`` is supplied, ``score_sample``
falls back to ``DEFAULT_MASTER_PROMPTS[sample.media_type]`` so each record
gets the rubric whose framing matches the payload the judge actually sees.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from fastgen.utils.env import set_env_vars
from fastgen.utils.vlm_judge import (
    JudgeSample,
    load_vlm_client,
    read_master_prompt,
    score_sample,
)


def load_existing_ids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    done: Set[str] = set()
    # Decode records independently: interruption can split a UTF-8 character in the final record.
    with output_path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            rid = obj.get("id")
            if rid is not None:
                done.add(str(rid))
    return done


def iter_input_records(jsonl_path: Path) -> Iterable[Dict[str, Any]]:
    with jsonl_path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{jsonl_path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{jsonl_path}:{lineno}: top-level JSON must be an object")
            obj.setdefault("id", str(lineno))
            yield obj


def score_one(
    record: Dict[str, Any],
    client: Any,
    model: str,
    master_prompt: Optional[str],
    n_frames: int,
    longest_side: int,
    max_tokens: int,
    temperature: float,
    jsonl_dir: Path,
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(record)
    media_path_raw = record.get("media_path")
    if not media_path_raw:
        return {
            **out,
            "score": None,
            "raw_response": "",
            "latency_s": 0.0,
            "parse_fail": True,
            "error": "missing media_path",
        }

    media_type = record.get("media_type", "auto")
    user_prompt = record.get("prompt", "") or ""
    try:
        sample = JudgeSample.from_path(
            media_path_raw,
            prompt=user_prompt,
            media_type=media_type,
            n_frames=n_frames,
            longest_side=longest_side,
            base_dir=jsonl_dir,
        )
    except Exception as e:
        return {
            **out,
            "score": None,
            "raw_response": "",
            "latency_s": 0.0,
            "parse_fail": True,
            "error": f"media load failed: {type(e).__name__}: {e}",
        }

    result = score_sample(
        client,
        model,
        sample,
        master_prompt=master_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        **out,
        "score": result.score,
        "raw_response": result.raw,
        "latency_s": result.latency_s,
        "parse_fail": result.parse_fail,
        **({"error": result.error} if result.error else {}),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score a JSONL of prompt/media pairs with a VLM-as-judge.",
    )
    p.add_argument(
        "--jsonl", required=True, type=Path, help="Input JSONL: {prompt, media_path, [media_type], [id]} per line."
    )
    p.add_argument("--output", required=True, type=Path, help="Output JSONL (one line per scored record).")
    p.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("VLM_BASE_URL"),
        help="Gateway base URL (default: $VLM_BASE_URL).",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model id as the OpenAI-compatible endpoint (VLM_BASE_URL) expects it, e.g. gemini-3.1-pro-preview.",
    )
    p.add_argument(
        "--master-prompt", default=None, help="Inline master prompt; overrides the media-type default for all records."
    )
    p.add_argument(
        "--master-prompt-file",
        default=None,
        help="Read master prompt from file; mutually exclusive with --master-prompt.",
    )
    p.add_argument(
        "--n-concurrent", type=int, default=1, help="Number of in-flight judge requests (thread pool). Default 1."
    )
    p.add_argument("--max-tokens", type=int, default=2048, help="Judge response token cap. Default 2048.")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge sampling temperature. Default 0.0. Ignored for Claude/Anthropic.",
    )
    p.add_argument("--video-frames", type=int, default=4, help="Frames per frame_grid sample. Default 4.")
    p.add_argument("--image-longest-side", type=int, default=768, help="Downscale cap (longest side, px). Default 768.")
    p.add_argument("--resume", action="store_true", help="Skip records whose id already appears in --output.")
    p.add_argument("--limit", type=int, default=0, help="Score at most N records (after --resume skips); 0 = all.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    master_prompt = read_master_prompt(args.master_prompt, args.master_prompt_file)
    if not args.base_url:
        raise SystemExit("vlm_infer: --base-url is required (or set VLM_BASE_URL env var)")
    set_env_vars()  # populates OPENAI_API_KEY from credentials/vlm_api_token.txt
    client = load_vlm_client(args.base_url)
    print(
        f"[vlm_infer] endpoint={args.base_url} model={args.model} " f"concurrency={args.n_concurrent}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    already_done: Set[str] = load_existing_ids(args.output) if args.resume else set()
    if already_done:
        print(f"[vlm_infer] --resume: skipping {len(already_done)} previously scored ids", flush=True)

    records: List[Dict[str, Any]] = []
    for rec in iter_input_records(args.jsonl):
        if args.resume and str(rec.get("id")) in already_done:
            continue
        records.append(rec)
        if args.limit and len(records) >= args.limit:
            break
    print(f"[vlm_infer] {len(records)} records to score", flush=True)
    if not records:
        return 0

    jsonl_dir = args.jsonl.resolve().parent
    out_mode = "a" if args.resume and args.output.exists() else "w"
    if out_mode == "a" and args.output.stat().st_size:
        # Terminate a record that an interrupted run left without its newline before appending to it.
        with args.output.open("rb+") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
    scores: List[float] = []
    parse_fails = 0
    errors = 0
    latencies: List[float] = []
    processed = 0

    def _submit(record: Dict[str, Any]) -> Dict[str, Any]:
        return score_one(
            record,
            client=client,
            model=args.model,
            master_prompt=master_prompt,
            n_frames=args.video_frames,
            longest_side=args.image_longest_side,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            jsonl_dir=jsonl_dir,
        )

    with args.output.open(out_mode) as out_fh:
        with ThreadPoolExecutor(max_workers=max(1, args.n_concurrent)) as pool:
            futures = {pool.submit(_submit, r): r for r in records}
            for fut in as_completed(futures):
                result = fut.result()
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_fh.flush()
                processed += 1
                if result.get("error"):
                    errors += 1
                if result.get("parse_fail"):
                    parse_fails += 1
                elif isinstance(result.get("score"), (int, float)):
                    scores.append(float(result["score"]))
                latencies.append(float(result.get("latency_s") or 0.0))
                if processed % 10 == 0 or processed == len(records):
                    print(
                        f"[vlm_infer] {processed}/{len(records)} " f"parse_fail={parse_fails} errors={errors}",
                        flush=True,
                    )

    if scores:
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(var)
    else:
        mean = float("nan")
        std = float("nan")
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    print(
        "[vlm_infer] done: "
        f"n={processed} mean={mean:.3f} std={std:.3f} "
        f"parse_fail_rate={parse_fails / processed:.3f} "
        f"errors={errors} mean_latency_s={mean_latency:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
