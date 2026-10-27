# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared config override helpers for ComfyUI FastGen nodes."""

from __future__ import annotations

import ast
import contextlib
import shlex

from fastgen.utils.basic_utils import attr_override
import fastgen.utils.logging_utils as logger


def parse_overrides(text: str) -> dict[str, str]:
    """Parse a multiline or space-separated ``key=value`` string.

    Tokenized with ``shlex.split``, so a value may contain spaces when
    quoted (``key="hello world"``) and ``#`` begins a comment that runs to the
    end of the line.  Supports dotted keys (``model.guidance_cfg.cfg.scale=5.0``).
    Returns ``{dotted_key: raw_value_string}``.
    """
    result = {}
    for token in shlex.split(text, comments=True):
        if "=" not in token:
            logger.warning(f"[FastGen] Skipping override (no '='): {token}")
            continue
        key, _, raw = token.partition("=")
        result[key.strip()] = raw
    return result


def _eval_value(raw: str):
    """Best-effort parse a raw string into a Python value."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


@contextlib.contextmanager
def temporary_overrides(config, text: str):
    """Apply ``key=value`` overrides to *config* for the duration of the block.

    Keys are dotted attribute paths relative to *config* (e.g. ``model.guidance_cfg.cfg.scale``);
    keys that do not resolve are skipped with a warning.
    """
    applied = []
    with contextlib.ExitStack() as stack:
        for dotted_key, raw_val in parse_overrides(text).items():
            try:
                stack.enter_context(attr_override(config, dotted_key, _eval_value(raw_val)))
            except (AttributeError, KeyError, TypeError):
                logger.warning(f"[FastGen] Config has no attribute '{dotted_key}', skipping.")
            else:
                applied.append(dotted_key)
        if applied:
            logger.info(f"[FastGen] Applied config overrides: {applied}")
        yield
