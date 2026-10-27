# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import zipfile
from typing import Any, BinaryIO, Callable, Dict, Optional, Tuple

import torch
from safetensors.torch import load as load_safetensors_bytes, load_file as load_safetensors_file

import fastgen.utils.cache_utils as cache_utils
import fastgen.utils.s3 as s3
import fastgen.utils.logging_utils as logger


def path_exists(path: str | None) -> bool:
    """Check a local path or S3 object; unset paths are absent and storage errors propagate."""
    if not path:
        return False
    return s3.exists(path) if s3.is_uri(path) else os.path.exists(path)


def _load_in_memory_state_dict(
    checkpoint_path: str,
    checkpoint: BinaryIO,
    device: torch.device | str,
) -> Dict[str, torch.Tensor]:
    """Deserialize one checkpoint object already held in memory."""
    if not cache_utils.is_safetensors_file(checkpoint_path):
        return torch.load(checkpoint, weights_only=True, map_location=device)

    # ``load_file`` needs a path it can mmap. ``load`` accepts the remote object's
    # bytes but always returns CPU tensors, hence the explicit move.
    state_dict = load_safetensors_bytes(checkpoint.read())
    return {key: tensor.to(device) for key, tensor in state_dict.items()}


def _load_s3_state_dict(checkpoint_path: str, device: torch.device | str) -> Dict[str, torch.Tensor]:
    """Load an S3 checkpoint straight into memory, without touching the filesystem."""
    if not cache_utils.is_safetensors_index(checkpoint_path):
        return _load_in_memory_state_dict(checkpoint_path, s3.load(checkpoint_path), device)

    def load_blob(uri: str) -> Dict[str, torch.Tensor]:
        return _load_in_memory_state_dict(uri, s3.load(uri), device)

    return _merge_safetensors_shards(
        checkpoint_path,
        s3.load(checkpoint_path).read().decode("utf-8"),
        lambda shard_name: load_blob(cache_utils._s3_shard_uri(checkpoint_path, shard_name)),
    )


def _load_http_state_dict(checkpoint_path: str, device: torch.device | str) -> Dict[str, torch.Tensor]:
    """Load a single-file HTTP(S) checkpoint straight into memory."""
    if cache_utils.is_safetensors_index(checkpoint_path):
        raise ValueError(f"Sharded SafeTensors indexes are not supported over HTTP(S): {checkpoint_path}")
    return _load_in_memory_state_dict(checkpoint_path, cache_utils.load_http(checkpoint_path), device)


# =============================================================================
# SafeTensors Utilities
# =============================================================================


def _merge_safetensors_shards(
    checkpoint_path: str,
    index_contents: str,
    load_shard: Callable[[str], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Merge the shards named by a SafeTensors index, rejecting duplicate or missing tensors."""
    weight_map = json.loads(index_contents).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{checkpoint_path} does not contain a non-empty weight_map")

    state_dict: Dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_dict = load_shard(shard_name)
        duplicate_keys = state_dict.keys() & shard_dict.keys()
        if duplicate_keys:
            duplicate_key = min(duplicate_keys)
            raise ValueError(f"Duplicate tensor {duplicate_key} found while loading {checkpoint_path}")
        state_dict.update(shard_dict)

    missing_keys = weight_map.keys() - state_dict.keys()
    if missing_keys:
        missing_key = min(missing_keys)
        raise ValueError(f"Tensor {missing_key} from {checkpoint_path} was not found in its shards")
    return state_dict


# =============================================================================
# Checkpoint Discovery and Loading
# =============================================================================


def load_training_checkpoint(path: str, device: torch.device | str = "cpu", mmap: bool = False) -> Dict[str, Any]:
    """Load a ``.pth`` training checkpoint or DCP marker from a local path or S3.

    This unpickles arbitrary objects, so load pretrained or third-party weights with ``load_checkpoint_state_dict``.
    """
    assert path.endswith(".pth"), f"{path} does not end with .pth"
    if not path_exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")
    source = s3.load(path) if s3.is_uri(path) else path
    # torch.load maps only local zipfile-format files; S3 buffers and legacy serialization are read normally.
    mmap = mmap and not s3.is_uri(path) and zipfile.is_zipfile(path)
    # These are full training checkpoints carrying optimizer/scheduler/callback state,
    # not just tensors, so the restricted unpickler cannot read them. Both branches must
    # pass weights_only=False explicitly: torch.load resolves the default to True, so
    # omitting it on one branch alone makes S3 reject payloads that load fine locally.
    state = torch.load(source, map_location=device, weights_only=False, mmap=mmap)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} holds a {type(state).__name__}, not a state dict")
    return state


def parse_checkpoint_iteration(path: str) -> int | None:
    """Return the iteration in an interval checkpoint name, e.g. ``.../0000500.pth`` -> 500.

    ``Checkpointer`` names interval checkpoints ``<iteration:07d>.pth``; any zero padding is accepted. Every
    other name returns None, including the ``latest_ar.pth`` autoresume marker, whose iteration is recorded
    only inside the checkpoint.
    """
    name = os.path.basename(path)
    stem = name.removesuffix(".pth")
    return int(stem) if name.endswith(".pth") and stem.isdecimal() else None


def _checkpoint_files(path: str) -> list[Tuple[str, float]]:
    """Return the name and modification time of each ``.pth`` file directly under a local directory or S3 prefix."""
    if s3.is_uri(path):
        # Prefix listings intentionally accept bucket roots; single-object APIs do not.
        model_files: list[Tuple[str, float]] = []
        for key, mtime in s3.iter_objects(path):
            if key.endswith(".pth"):
                model_files.append((key.rsplit("/", 1)[-1], mtime))
    elif os.path.exists(path):
        # Get the list of files and modification times in the local directory
        model_files = []
        for f in os.listdir(path):
            if not f.endswith(".pth"):
                continue
            fp = os.path.join(path, f)
            if not os.path.isfile(fp):
                continue
            mtime = float(os.path.getmtime(fp))
            model_files.append((f, mtime))
    else:
        # no model files found
        model_files = []
    return model_files


def list_checkpoints(save_dir: str) -> list[Tuple[int, str]]:
    """Return ``(iteration, path)`` for each interval checkpoint directly under ``save_dir``, ordered by iteration.

    ``save_dir`` may be a local directory or an ``s3://`` prefix. Only names accepted by
    ``parse_checkpoint_iteration`` are listed, so the ``latest_ar.pth`` autoresume marker is not.
    """
    checkpoints = []
    for name, _ in _checkpoint_files(save_dir):
        iteration = parse_checkpoint_iteration(name)
        if iteration is not None:
            checkpoints.append((iteration, os.path.join(save_dir, name)))
    return sorted(checkpoints)


def latest_checkpoint(path: str) -> str:
    """Get the latest checkpoint with the largest iteration number under ``path``.

    Scans ``*.pth`` under ``path``: interval saves named ``<iter>.pth`` and the
    autoresume marker ``latest_ar.pth`` (SIGUSR1). For each candidate we take an
    iteration (filename integer for numeric stems, checkpoint metadata for
    ``latest_ar``). Returns the ``.pth`` path for the best candidate:
    maximize ``(iteration, mtime)`` so higher iteration wins, and when iterations
    tie the file with the newer modification time wins (local ``mtime``, S3
    ``LastModified``).
    """

    # Keep the actual filename: numeric checkpoint names need not use seven-digit padding.
    # Order by (iteration, mtime) only.
    best: Optional[Tuple[int, float, str]] = None

    for file, mtime in _checkpoint_files(path):
        stem = os.path.basename(file)[: -len(".pth")]
        is_latest_ar = stem == "latest_ar"
        iteration: Optional[int] = None
        if is_latest_ar:
            ckpt_path = os.path.join(path, file)
            try:
                # Only the iteration is needed; mmap skips reading storage bytes (~3x faster on large .pth).
                iteration = int(load_training_checkpoint(ckpt_path, device="meta", mmap=True)["iteration"])
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(f"Could not read iteration from {ckpt_path}; resume ignores it: {e}")
        else:
            iteration = parse_checkpoint_iteration(file)

        if iteration is None:
            continue

        if best is None or (iteration, mtime) > (best[0], best[1]):
            best = (iteration, mtime, file)

    if best is None:
        logger.warning(f"No model files found in {path}")
        return ""

    return os.path.join(path, best[2])


def load_checkpoint_state_dict(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load a model state dict from a checkpoint file.

    Supports single-file SafeTensors checkpoints (``.safetensors``), sharded SafeTensors
    checkpoints referenced by an index file (``.safetensors.index.json``), and PyTorch
    checkpoints (any other extension, loaded via ``torch.load``). Every format may live on
    a local path or behind an ``s3://`` URI; single-file checkpoints may also use HTTP(S).
    A sharded index resolves its shards against its own local directory or S3 prefix.

    Remote paths first resolve through ``fastgen.utils.cache_utils.resolve_cached_path``.
    An existing mirror is read from disk, and the log names both paths to make this redirection visible.

    Training checkpoints saved by ``Checkpointer`` also hold optimizer and callback state;
    load them with ``load_training_checkpoint``.

    Args:
        checkpoint_path (str): The local path, S3 URI, or HTTP(S) URL of the checkpoint file,
            or the local path or S3 URI of a SafeTensors index file
        device (torch.device | str): The device to load the tensors on. Defaults to ``"cpu"``.

    Returns:
        The state dict loaded from the checkpoint
    """
    resolved = cache_utils.resolve_cached_path(checkpoint_path)
    if resolved != checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path} (mirrored at {resolved})")
    else:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_path = resolved

    if s3.is_uri(checkpoint_path):
        return _load_s3_state_dict(checkpoint_path, device)

    if cache_utils.is_http_url(checkpoint_path):
        return _load_http_state_dict(checkpoint_path, device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} is not a valid file")

    # SafeTensors' ``load_file`` accepts only ``str`` or ``int`` devices and rejects ``torch.device``.
    safetensors_device = str(device)
    if cache_utils.is_safetensors_index(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as index_file:
            index_contents = index_file.read()
        checkpoint_dir = os.path.dirname(checkpoint_path)

        def load_shard(shard_name: str) -> Dict[str, torch.Tensor]:
            shard_path = os.path.join(checkpoint_dir, shard_name)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Checkpoint shard {shard_name} referenced by {checkpoint_path} does not exist")
            return load_safetensors_file(shard_path, device=safetensors_device)

        return _merge_safetensors_shards(checkpoint_path, index_contents, load_shard)

    if cache_utils.is_safetensors_file(checkpoint_path):
        return load_safetensors_file(checkpoint_path, device=safetensors_device)

    return torch.load(checkpoint_path, weights_only=True, map_location=device)
