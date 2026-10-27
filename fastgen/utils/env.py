# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment parsing and credential/cache configuration."""

import json
import os
from typing import Any, Callable

import fastgen.utils.logging_utils as logger


def str2bool(v):
    """Parse a boolean or one of FastGen's accepted boolean strings."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError("Boolean value expected.")


def output_root() -> str:
    """Return the absolute root for FastGen checkpoints, logs, caches, and datasets."""
    return os.path.abspath(os.getenv("FASTGEN_OUTPUT_ROOT", "").strip() or "FASTGEN_OUTPUT")


def hf_load(loader: Callable[..., Any], model_id_or_local_path: str, *args: Any, **kwargs: Any) -> Any:
    """Call a Hugging Face ``loader`` such as ``from_pretrained`` or ``hf_hub_download`` with FastGen's defaults.

    ``HF_HOME`` and ``LOCAL_FILES_ONLY`` supply ``cache_dir`` and ``local_files_only`` unless passed explicitly.
    Hub access failures (gated, private, or misspelled repos) are re-raised as a ``RuntimeError`` listing the
    license and authentication steps; diffusers otherwise reports them as a misleading "is not a local folder".
    """
    if "HF_HOME" in os.environ:
        kwargs.setdefault("cache_dir", os.environ["HF_HOME"])
    kwargs.setdefault("local_files_only", str2bool(os.getenv("LOCAL_FILES_ONLY", "false")))
    try:
        return loader(model_id_or_local_path, *args, **kwargs)
    except OSError as e:
        # diffusers and transformers re-raise Hub errors as a plain OSError but keep the original in the chain.
        if not any(s in str(e).lower() for s in ("gated", "401", "is not a local folder")):
            raise
        # Lazy: importing huggingface_hub fixes its HF_HOME-derived paths, which set_env_vars may still change.
        from huggingface_hub.errors import GatedRepoError

        gated_error = e
        while gated_error is not None and not isinstance(gated_error, GatedRepoError):
            gated_error = gated_error.__cause__ or gated_error.__context__
        license_step = f"accept the license at https://huggingface.co/{model_id_or_local_path}, " if gated_error else ""
        raise RuntimeError(
            f"Could not load {model_id_or_local_path!r} from Hugging Face. To fix: {license_step}"
            "authenticate with `hf auth login` or `export HF_TOKEN=...` (visible to all torchrun ranks), "
            f"check the spelling of the model ID or local path, and relaunch.\nOriginal error: {e}"
        ) from e


def hf_download(model_id_or_local_path: str, filename: str | None = None, **kwargs: Any) -> str:
    """Download ``filename`` or, without a ``filename``, the whole repo and return its local path.

    A local directory ``model_id_or_local_path`` is used in place and never reaches the Hub. Otherwise ``hf_load``
    calls ``hf_hub_download`` or ``snapshot_download``, which receive ``kwargs`` such as ``subfolder``, ``revision``,
    or ``allow_patterns``.
    """
    if os.path.isdir(model_id_or_local_path):
        if filename is None:
            return model_id_or_local_path
        return os.path.join(model_id_or_local_path, kwargs.get("subfolder") or "", filename)
    # Lazy: importing huggingface_hub fixes its HF_HOME-derived paths, which set_env_vars may still change.
    import huggingface_hub

    if filename is None:
        return hf_load(huggingface_hub.snapshot_download, model_id_or_local_path, **kwargs)
    return hf_load(huggingface_hub.hf_hub_download, model_id_or_local_path, filename, **kwargs)


def _load_aws_credentials(path: str, override: bool = True) -> None:
    """Set the AWS/S3 variables from the JSON credentials file ``path``; ``override=False`` keeps set variables."""
    key_map = {
        "AWS_ACCESS_KEY_ID": "aws_access_key_id",
        "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
        "AWS_SESSION_TOKEN": "aws_session_token",
        "AWS_DEFAULT_REGION": "region_name",
        "AWS_ENDPOINT_URL": "endpoint_url",
        "S3_ENDPOINT_URL": "endpoint_url",
    }
    try:
        with open(path, "r", encoding="utf-8") as credential_file:
            config = json.load(credential_file)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {path}, skip loading AWS credentials.")
        return
    for env_key, config_key in key_map.items():
        if config.get(config_key) is not None:
            if override or not os.environ.get(env_key):
                os.environ[env_key] = config[config_key]
        elif config_key != "aws_session_token":
            logger.warning(f"Missing key {config_key} in {path}, skipping env variable {env_key}.")
    logger.success(f"AWS credentials loaded from {path} and set as environment variables.")


def _load_token(path: str, env_key: str, override: bool = True) -> None:
    """Set ``env_key`` from the single-line token file ``path`` unless the file is empty.

    With ``override=False`` an already set, non-empty ``env_key`` is kept; the file is still validated.
    """
    with open(path, encoding="utf-8") as credential_file:
        token = credential_file.read().strip()
    if not token:
        return
    if "\n" in token or "\r" in token:
        raise ValueError(f"{path} must hold {env_key} on a single line.")
    if not override and os.environ.get(env_key):
        logger.info(f"Keeping the preset {env_key}; ignoring {path}")
        return
    os.environ[env_key] = token
    logger.info(f"Loading {env_key} from {path}")


def set_env_vars(credentials_dir: str | None = "./credentials", override: bool = True) -> None:
    """Load optional credentials from ``credentials_dir`` into the environment and configure the HF cache.

    ``wandb_api.txt``, ``hf_token.txt``, and ``vlm_api_token.txt`` hold ``WANDB_API_KEY``, ``HF_TOKEN``, and
    ``OPENAI_API_KEY``; ``s3.json`` holds the AWS/S3 credentials. A non-empty file overrides the existing value,
    and a missing or empty file leaves the environment alone, so this function stays safe in CI and standalone
    scripts. ``override=False`` inverts the precedence: a variable that is already set and non-empty (e.g. a
    secret injected by the scheduler) is kept, and files only fill in the missing ones.

    Raises:
        ValueError: If a token file spans several lines.
    """
    if credentials_dir:
        s3_path = os.path.join(credentials_dir, "s3.json")
        if os.path.isfile(s3_path):
            _load_aws_credentials(s3_path, override=override)
        for filename, env_key in (
            ("wandb_api.txt", "WANDB_API_KEY"),
            ("hf_token.txt", "HF_TOKEN"),
            ("vlm_api_token.txt", "OPENAI_API_KEY"),
        ):
            path = os.path.join(credentials_dir, filename)
            if os.path.isfile(path):
                _load_token(path, env_key, override=override)

    # A launcher may supply ``HF_HOME`` relative; it is inherited by subprocesses and
    # passed to Hugging Face as ``cache_dir``, so pin it rather than forking the cache.
    os.environ["HF_HOME"] = os.path.abspath(os.getenv("HF_HOME", os.path.join(output_root(), ".cache")))
