# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Global variables for the fastgen project.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

_GLOBAL_VAR_NAMES = ("MAX_VAL_STEPS", "STUDENT_STEPS")
__all__ = list(_GLOBAL_VAR_NAMES)

MAX_VAL_STEPS = None

# Optional process-wide override of the student's sampling step count (NFE).
# ``None`` falls back to the scheduler's configured default. Validation uses
# ``trainer.global_vars_val`` to set it temporarily, but any caller may do so.
STUDENT_STEPS = None


def set_global_vars(config: Mapping[str, object] | None = None) -> None:
    """Update recognized process-wide settings from ``config``."""
    import fastgen.utils.logging_utils as logger

    config = config or {}
    update_config = {key: value for key, value in config.items() if key in _GLOBAL_VAR_NAMES}
    logger.debug(f"Setting global variables {update_config}")
    globals().update(update_config)

    ignore_keys = [key for key in config if key not in _GLOBAL_VAR_NAMES]
    if ignore_keys:
        logger.warning(f"Ignoring keys {ignore_keys} since they are not found in global_vars.")


@contextmanager
def set_temp_global_vars(config: Mapping[str, object] | str | None) -> Iterator[None]:
    """Temporarily update process-wide settings and restore them on exit."""
    import fastgen.utils.logging_utils as logger

    # Handle string "None" from command line parsing.
    if config == "None":
        config = None

    original_global_vars = {key: globals()[key] for key in _GLOBAL_VAR_NAMES}
    try:
        set_global_vars(config)
        yield
    finally:
        logger.debug(f"Resetting global variables to {original_global_vars}")
        globals().update(original_global_vars)
