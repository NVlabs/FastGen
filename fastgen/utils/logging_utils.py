# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import sys
from typing import Callable

from loguru import logger

from fastgen.utils.distributed import get_rank, is_rank0

# Default logging level — can be changed later
LOG_LEVEL = "INFO"

logger.remove(0)
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="[<green>{time:MMM D, YYYY - HH:mm:ss}</green> | "
    "<level>{level}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> ] {message}",
)


def formatter(record):
    """
    Custom formatter function to conditionally add the MPI rank.
    """
    # Base format string for all levels; a callable format must append the newline and {exception} itself
    msg = (
        "[<green>{time:MMM D, YYYY - HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> ] {message}\n{exception}"
    )

    # For DEBUG messages, add the rank
    if LOG_LEVEL == "DEBUG":
        msg = f"[<magenta>RANK: {get_rank()}</magenta>] {msg}"

    # For all other levels, use the base format
    return msg


def set_log_level(level: str):
    """Change log level dynamically."""
    global LOG_LEVEL
    LOG_LEVEL = level.upper()
    logger.remove()  # Remove all existing handlers
    logger.add(
        sys.stdout,
        format=formatter,
        level=LOG_LEVEL,
    )


def rank0_if_not_debug(func: Callable) -> Callable:
    """
    A decorator factory that applies the rank0_only decorator if LOG_LEVEL is not DEBUG.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # At runtime, check the global flag to decide which version to run.
        if LOG_LEVEL == "DEBUG" or is_rank0():
            return func(*args, **kwargs)

    return wrapper


@rank0_if_not_debug
def trace(msg: str):
    logger.opt(depth=2).trace(msg)


@rank0_if_not_debug
def info(msg: str):
    logger.opt(depth=2).info(msg)


@rank0_if_not_debug
def debug(msg: str):
    logger.opt(depth=2).debug(msg)


@rank0_if_not_debug
def success(msg: str):
    logger.opt(depth=2).success(msg)


@rank0_if_not_debug
def critical(msg: str):
    logger.opt(depth=2).critical(msg)


@rank0_if_not_debug
def warning(msg: str):
    logger.opt(depth=2).warning(msg)


@rank0_if_not_debug
def error(msg: str):
    logger.opt(depth=2).error(msg)
