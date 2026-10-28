# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastGen config loader node for ComfyUI."""

from __future__ import annotations

import glob
import hashlib
import os

from fastgen.configs.config import BaseConfig
import fastgen.utils.logging_utils as logger
from fastgen.utils.env import set_env_vars
from fastgen.utils.scripts import load_config as load_fastgen_config, resolve_save_dir, set_cuda_backend

from ._config_helpers import parse_overrides


def _discover_configs() -> list[str]:
    """Scan fastgen experiment configs and return sorted relative paths."""
    repo = os.environ.get("FASTGEN_REPO", "")
    if not repo:
        return []
    pattern = os.path.join(repo, "fastgen", "configs", "experiments", "**", "config*.py")
    configs = sorted(glob.glob(pattern, recursive=True))
    prefix = repo + os.sep
    return [c.removeprefix(prefix) for c in configs]


class FastGenConfigLoader:
    """Load a FastGen experiment config and apply optional overrides.

    Outputs an opaque FASTGEN_CONFIG object consumed by downstream nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        choices = _discover_configs() or ["fastgen/configs/experiments/WanT2V/config_dmd2.py"]
        return {
            "required": {
                "config_path": (choices, {"tooltip": "Experiment config file"}),
            },
            "optional": {
                "config_overrides": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Config overrides as key=value lines (e.g. model.guidance_cfg.cfg.scale=5.0).",
                    },
                ),
                "env_vars": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Environment variables as KEY=VALUE lines (e.g. LOCAL_FILES_ONLY=True).",
                    },
                ),
                "log_level": (["INFO", "DEBUG", "WARNING", "ERROR"], {"default": "INFO"}),
            },
        }

    RETURN_TYPES = ("FASTGEN_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "load_config"
    CATEGORY = "FastGen"
    DESCRIPTION = "Load a FastGen experiment config with optional overrides."

    def load_config(
        self,
        config_path: str,
        config_overrides: str = "",
        env_vars: str = "",
        log_level: str = "INFO",
    ):
        if env_vars and env_vars.strip():
            for key, value in parse_overrides(env_vars).items():
                os.environ[key] = value
                logger.info(f"[FastGen] Set env: {key}")

        logger.set_log_level(log_level)

        opts = None
        if config_overrides and config_overrides.strip():
            overrides = parse_overrides(config_overrides)
            if overrides:
                tokens = [f"{k}={v}" for k, v in overrides.items()]
                opts = ["-"] + tokens

        # config_path comes from the node dropdown as a path relative to
        # FASTGEN_REPO, so chdir into it. Fail fast with a clear error if the
        # repo is unset/missing rather than letting a relative path resolve
        # against an unexpected CWD.
        repo = os.environ.get("FASTGEN_REPO", "")
        if not repo or not os.path.isdir(repo):
            raise FileNotFoundError(
                f"FASTGEN_REPO is not a valid directory (got {repo!r}). "
                "Set the FASTGEN_REPO environment variable to your fastgen checkout."
            )

        old_cwd = os.getcwd()
        os.chdir(repo)
        try:
            config: BaseConfig = load_fastgen_config(config_path, opts, evaluation=True)
        finally:
            os.chdir(old_cwd)

        config.trainer.ddp = False
        config.trainer.fsdp = False

        config.trainer.checkpointer.save_dir = resolve_save_dir(config)

        credentials_dir = os.path.join(repo, config.credentials_dir) if config.credentials_dir else None
        set_env_vars(credentials_dir)

        set_cuda_backend(
            config.trainer.cudnn.deterministic,
            config.trainer.cudnn.benchmark,
            config.trainer.tf32_matmul,
            config.trainer.tf32_conv,
            config.trainer.fp16_reduced_precision_reduction,
            torch_compile_enabled=config.model.torch_compile_mode is not None,
        )

        logger.info(f"[FastGen] Loaded config: {config_path}")
        return (config,)

    @classmethod
    def IS_CHANGED(cls, config_path: str, config_overrides: str = "", env_vars: str = "", log_level: str = "INFO"):
        h = hashlib.md5(f"{config_path}|{config_overrides}|{env_vars}|{log_level}".encode()).hexdigest()
        return h
