# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for image and video model inference scripts."""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List, Any

import torch

from fastgen.configs.config import BaseConfig
from fastgen.utils import instantiate
from fastgen.utils.checkpointer import FSDPCheckpointer, Checkpointer
import fastgen.utils.logging_utils as logger


def expand_path(path: str | Path, relative_to: str = "cwd") -> Path:
    """Resolve path - absolute paths stay as-is, relative paths are resolved.

    Args:
        path: Path to resolve
        relative_to: How to resolve relative paths:
            - "cwd": relative to current working directory
            - "script": relative to the calling script's directory (for prompt files)

    Returns:
        Resolved absolute path
    """
    path = Path(path)
    if path.is_absolute():
        return path

    if relative_to == "cwd":
        return Path.cwd() / path
    elif relative_to == "script":
        # Get the caller's directory - useful for default prompt files
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_file = frame.f_back.f_globals.get("__file__")
            if caller_file:
                return Path(caller_file).parent / path
        return Path.cwd() / path
    else:
        return Path.cwd() / path


def load_prompts(prompt_file: str | Path, relative_to: str = "cwd") -> List[str]:
    """Load prompts from a file.

    Args:
        prompt_file: Path to prompt file (one prompt per line)
        relative_to: How to resolve relative paths ("cwd" or "script")

    Returns:
        List of prompts

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_path = expand_path(prompt_file, relative_to)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with prompt_path.open("r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    logger.info(f"Loaded {len(prompts)} prompts from {prompt_path}")
    return prompts


def init_model(config: BaseConfig) -> Any:
    """Initialize the model from config.

    Args:
        config: Base configuration object

    Returns:
        Instantiated model
    """
    config.model_class.config = config.model
    model = instantiate(config.model_class)
    config.model_class.config = None
    return model


def load_checkpoint(
    checkpointer: Checkpointer | FSDPCheckpointer,
    model: Any,
    ckpt_path: str,
    config: BaseConfig,
) -> Tuple[Optional[int], str]:
    """Load checkpoint if valid path provided.

    Args:
        checkpointer: Checkpointer instance
        model: Model to load weights into
        ckpt_path: Path to checkpoint
        config: Base configuration object

    Returns:
        Tuple of (checkpoint_iteration or None, save_directory)
    """
    ckpt_iter = None

    if ckpt_path and (os.path.isdir(ckpt_path + ".net_model") or os.path.isfile(ckpt_path)):
        # Construct save directory from checkpoint path
        save_dir = f"{config.log_config.save_path}/{ckpt_path.split('/')[-3]}/{ckpt_path.split('/')[-1].split('.')[0]}"
        logger.info(f"ckpt_path: {ckpt_path}, save_dir: {save_dir}")

        # Build model dict for loading
        model_dict_infer = torch.nn.ModuleDict({"net": model.net, **model.ema_dict})

        ckpt_iter = checkpointer.load(model_dict_infer, path=ckpt_path)
        logger.success(f"Loading successfully checkpoint {ckpt_iter}")
    else:
        save_dir = f"{config.log_config.save_path}/inference_validation"
        logger.warning(f"No valid ckpt path, save_dir: {save_dir}")

    return ckpt_iter, save_dir


def cleanup_unused_modules(model: Any, do_teacher_sampling: bool) -> None:
    """Remove unused modules to free memory.

    Args:
        model: Model to clean up
        do_teacher_sampling: Whether teacher sampling will be performed
    """
    if hasattr(model, "fake_score"):
        del model.fake_score
    if hasattr(model, "discriminator"):
        del model.discriminator
    if (not do_teacher_sampling) and hasattr(model, "teacher"):
        del model.teacher


def setup_inference_modules(
    model: Any,
    config: BaseConfig,
    do_teacher_sampling: bool,
    do_student_sampling: bool,
    precision: torch.dtype,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Set up model modules for inference.

    Args:
        model: The model instance
        config: Base configuration object
        do_teacher_sampling: Whether to set up teacher for sampling
        do_student_sampling: Whether to set up student for sampling
        precision: Inference precision dtype

    Returns:
        Tuple of (teacher, student, vae) - any may be None
    """
    teacher, student, vae = None, None, None

    if do_teacher_sampling:
        # Use model.teacher if available, otherwise use model.net for teacher-style sampling
        if getattr(model, "teacher", None) is not None:
            teacher = model.teacher
        else:
            teacher = model.net
        teacher.eval().to(dtype=precision, device=model.device)

    if do_student_sampling:
        student = getattr(model, model.use_ema[0]) if model.use_ema else model.net
        student.eval().to(dtype=precision, device=model.device)
        logger.info(f"Evaluating student sample steps: {model.config.student_sample_steps}")

    if hasattr(model.net, "init_preprocessors") and config.model.enable_preprocessors:
        model.net.init_preprocessors()
        vae = model.net.vae
        vae.to(device=model.device, dtype=precision)

    return teacher, student, vae


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared between image and video inference.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--ckpt_path",
        default="",
        type=str,
        help="Path to the checkpoint (optional, uses pretrained if not provided)",
    )
    parser.add_argument(
        "--do_student_sampling",
        default=True,
        type=lambda x: x.lower() in ("true", "1", "yes"),
        help="Whether to perform student sampling",
    )
    parser.add_argument(
        "--do_teacher_sampling",
        default=True,
        type=lambda x: x.lower() in ("true", "1", "yes"),
        help="Whether to perform teacher sampling",
    )
