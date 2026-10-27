# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monolithic and DCP training checkpoints, both addressed by a ``.pth`` path.

Without FSDP, the trainer saves one monolithic ``.pth`` file, which loads only into unsharded models and optimizers.
Under FSDP, it saves PyTorch Distributed Checkpoint (DCP) shards in directories such as ``0000200.net_model``
beside their ``0000200.pth`` completion marker. DCP checkpoints load into sharded and unsharded models alike,
e.g. for inference. Loading detects the format from the file's contents.

Under FSDP, the trainer therefore loads monolithic ``pretrained_ckpt_path`` weights on CPU before wrapping, as it
does without FSDP, and every rank holds the full model until wrapping; DCP weights load after wrapping, directly
into the shards. Resuming FSDP training requires a DCP checkpoint; to start an FSDP run from monolithic weights,
load them through ``pretrained_ckpt_path`` in a new run directory, and the optimizers start fresh.

The checkpointer uses ``save_dir`` as given, so saving and checkpoint discovery require an absolute or ``s3://``
``save_dir``. ``fastgen.utils.scripts.setup`` resolves a relative one with ``resolve_save_dir``, which code that
skips ``setup`` must call itself. Explicit load paths such as ``pretrained_ckpt_path`` are read from S3 only if
they are ``s3://`` URIs, regardless of ``use_s3``.
"""

import io
import os.path
import torch.nn
from typing import Dict, Union, Any, Iterable, Mapping

from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    _get_fqns,
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.tensor import DTensor

from fastgen.configs.config import BaseCheckpointerConfig
import fastgen.utils.s3 as s3
from fastgen.utils.cache_utils import atomic_write, remove_atomic_write_orphans
from fastgen.utils.distributed.s3_filesystem import S3StorageWriter, S3StorageReader
from fastgen.utils.io_utils import latest_checkpoint, load_training_checkpoint, path_exists
import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed import is_rank0
from fastgen.utils.distributed.collectives import (
    raise_across_ranks,
    sync_any,
    validate_object_across_ranks,
)
from fastgen.utils.distributed.module import (
    broadcast_module,
    validate_module_state_across_ranks,
    validate_module_sync_source,
)
from fastgen.utils.module import (
    ModuleStateSnapshot,
    has_meta_state,
    load_module_state_dict,
    materialize_meta_module,
    module_state_names,
    validate_state_dict_export,
    validate_module_roots,
    validate_module_owners_preserved,
    validate_module_state_preserved,
    reject_extra_state,
    reject_inference_state,
    reject_unsafe_placement,
    reject_unsupported_state_aliases,
    validate_torch_state_settings,
)
from fastgen.callbacks.callback import CallbackDict


def _remove_checkpoint_artifact(path: str) -> None:
    """Remove a local file or S3 object, tolerating an absent local file."""
    if s3.is_uri(path):
        s3.delete(path)
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def validate_checkpoint_path(path: str | None) -> str | None:
    """Validate a user-facing ``.pth`` checkpoint path; an unset option means no checkpoint."""
    if not path:
        return None
    if not path.endswith(".pth"):
        raise ValueError(f"Checkpoint path must end with .pth, got {path!r}")
    return path


def _require_storage_location(path: str | None, name: str) -> str:
    """Return ``path`` if it names its storage independently of the working directory."""
    if not path or not (s3.is_uri(path) or os.path.isabs(path)):
        raise ValueError(
            f"{name} must be an absolute path or an s3:// URI, got {path!r}; "
            "fastgen.utils.scripts.setup resolves a relative save_dir below the run directory"
        )
    return path


def _drop_unheld(entries: Mapping[str, Any] | None, kind: str, operation: str) -> dict[str, Any]:
    """Drop ``None`` entries, which name objects the caller does not hold."""
    entries = entries or {}
    unheld = [name for name, entry in entries.items() if entry is None]
    if unheld:
        logger.info(f"{operation}: skipping {kind} {unheld}, which are None.")
    return {name: entry for name, entry in entries.items() if entry is not None}


def _held_targets(
    model_dict: torch.nn.ModuleDict,
    optimizer_dict: Dict[str, torch.optim.Optimizer] | None,
    scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None,
    operation: str,
) -> tuple[torch.nn.ModuleDict, Dict[str, torch.optim.Optimizer], Dict[str, torch.optim.lr_scheduler.LambdaLR]]:
    """Return the roots, optimizers, and schedulers the caller holds, without ``None`` entries.

    Only ``None`` roots would save or restore no weights without an error, so they are rejected;
    an empty ``model_dict`` requests scalar state alone.
    """
    if len(model_dict) and all(module is None for module in model_dict.values()):
        raise ValueError(
            f"{operation}: every model root in {list(model_dict)} is None; "
            "pass an empty ModuleDict to save or restore scalar state alone."
        )
    held_models = _drop_unheld(model_dict, "model roots", operation)
    if len(held_models) < len(model_dict):
        # Only a snapshot of the caller's own dict detects hooks that replace its roots, so copy only when needed.
        model_dict = torch.nn.ModuleDict(held_models)
    return (
        model_dict,
        _drop_unheld(optimizer_dict, "optimizers", operation),
        _drop_unheld(scheduler_dict, "schedulers", operation),
    )


def _checkpoint_plan(path, model_dict, optimizer_dict, **options) -> dict:
    """Describe the ordered work before any format-specific collective or target mutation."""
    return {
        "path": path if path is None or s3.is_uri(path) else os.path.realpath(path),
        "models": tuple(model_dict),
        "optimizers": tuple(optimizer_dict),
        **options,
    }


def is_dcp_marker_state(state: Dict[str, Any]) -> bool:
    """Return whether loaded ``.pth`` contents are a DCP completion marker rather than a monolithic checkpoint.

    Monolithic files keep their weights under ``"model"``; a DCP marker holds scalar state and, unless a legacy
    version saved it, a shard manifest.
    """
    return "model" not in state


def is_dcp_checkpoint(path: str | None, model_keys: Iterable[str]) -> bool:
    """Identify DCP checkpoints with shards for the requested model keys.

    Check shard metadata first so ordinary monolithic files are never read by the probe.
    When shards exist, inspect the .pth contents too: old shards can survive a monolithic
    overwrite. This fallback uses meta/mmap, but S3 and legacy files still require a full read.
    Callers deciding distributed load order must broadcast the result or error from rank 0.
    """
    path = validate_checkpoint_path(path)
    if path is None:
        return False
    prefix = path.removesuffix(".pth")
    if not any(path_exists(os.path.join(f"{prefix}.{key}_model", ".metadata")) for key in model_keys):
        return False
    return is_dcp_marker_state(load_training_checkpoint(path, device="meta", mmap=True))


def _validate_load_targets(
    model_dict: torch.nn.ModuleDict,
    needs_meta_sync_by_name: Dict[str, bool],
    optimizer_dict: Dict[str, torch.optim.Optimizer],
    operation: str,
) -> None:
    """Locally validate the collection and every selected target before any rank materializes."""
    reject_unsupported_state_aliases(model_dict, operation, allow_buffer_aliases=True)
    validate_module_roots(model_dict, operation)
    validate_torch_state_settings()
    for name, needs_meta_sync in needs_meta_sync_by_name.items():
        module = model_dict[name]
        reject_inference_state(module, operation)
        if not needs_meta_sync:
            continue  # Fully real loads only copy values, so runtime buffer ties remain valid there.
        reject_unsafe_placement(module, operation)
        validate_module_sync_source(module, src=0, operation=operation)
        if optimizer_dict and has_meta_state(module):
            raise RuntimeError(
                f"Model {name!r} still holds meta tensors at checkpoint load time, after its optimizers were created. "
                "Materializing now would replace its parameter objects behind the optimizer's and any DDP wrapper's "
                "back. Materialize it through `prepare_module` (a device or an FSDP context) before `init_optimizers`."
            )


def _prepare_load_targets(
    model_dict: torch.nn.ModuleDict,
    target_names: Iterable[str],
    optimizer_dict: Dict[str, torch.optim.Optimizer],
    device: torch.device | str,
    operation: str,
) -> tuple[ModuleStateSnapshot, dict[str, bool]]:
    """Validate and materialize selected targets, then return the baseline for copy loading.

    The checkpoint plan has already agreed on the ordered targets. Resolve placement first: a real
    rank can also need placement because of a meta peer. Every target must pass local validation and
    the full collection must agree on topology before any target is materialized. Finish all identity
    changes before loading values so the copy phase can reuse one baseline for the entire collection.
    """
    needs_meta_sync_by_name = {
        name: sync_any(has_meta_state(model_dict[name]), device=torch.device(device)) for name in target_names
    }
    with raise_across_ranks(f"{operation} target validation"):
        _validate_load_targets(model_dict, needs_meta_sync_by_name, optimizer_dict, operation)
    validate_module_state_across_ranks(
        model_dict, operation=f"{operation} topology", nonpersistent_buffers=False, include_device=False
    )
    for name, needs_meta_sync in needs_meta_sync_by_name.items():
        if not needs_meta_sync:
            continue
        module = model_dict[name]
        materialization = f"{operation} materialization"
        with raise_across_ranks(materialization):
            with validate_module_owners_preserved(model_dict, materialization):
                # Only this root may replace tensors or change placement. Keep every other root's
                # registrations intact, including roots still awaiting materialization. In-place
                # value changes violate the hook contract but are not detected by these snapshots.
                other_roots = torch.nn.ModuleDict({key: root for key, root in model_dict.items() if root is not module})
                with validate_module_state_preserved(other_roots, materialization):
                    # Materialization validates its own registrations; ``to`` cannot create aliases.
                    if has_meta_state(module):
                        materialize_meta_module(module, device=device)
                    else:
                        module.to(device=device)
                    validate_module_roots(model_dict, materialization)
        validate_module_state_across_ranks(module, operation=materialization)
    # The copy phase needs only the final collection baseline, after all allowed identity changes.
    with raise_across_ranks(f"{operation} baseline"):
        snapshot = ModuleStateSnapshot.capture(model_dict, operation)
    return snapshot, needs_meta_sync_by_name


def _registration_names_for_dcp_keys(module: torch.nn.Module, keys: Iterable[str]) -> set[str]:
    """Translate canonical DCP (or exported ``state_dict``) keys into raw registration names, including aliases."""
    keys = set(keys)
    if not keys:
        return set()
    names: set[str] = set()
    unmatched = keys.copy()
    for name in module_state_names(module, nonpersistent_buffers=False):
        # Use DCP's own private mapping so checkpoint/compile/DDP wrappers follow exactly the
        # same naming rules as get_model_state_dict, rather than maintaining a parallel strip list.
        matched = keys.intersection(_get_fqns(module, name))
        if matched:
            names.add(name)
            unmatched.difference_update(matched)
    # Callers pass the module's own export or DCP keys, so every key names a registered tensor.
    assert not unmatched, f"{type(module).__name__}: DCP keys without a registered tensor: {sorted(unmatched)[:8]}"
    return names


def _validate_adapter_state_dict(
    module: torch.nn.Module, state_with_vars: Dict[str, Any], adapter_state: Dict[str, Any]
) -> None:
    """Require the captured adapter export to contain exactly the structurally selected keys."""
    expected = {name for name in state_with_vars if module._is_adapter_tensor(name)}
    actual = set(adapter_state)
    missing, unexpected = expected - actual, actual - expected
    if missing or unexpected:
        raise RuntimeError(
            f"{type(module).__name__}.adapter_state_dict() must export exactly the keys selected by "
            f"_is_adapter_tensor; missing keys (first 8): {sorted(missing)[:8]}, "
            f"unexpected keys (first 8): {sorted(unexpected)[:8]}."
        )


def _log_optimizer_coverage(name: str, current_params: set[str], checkpoint_keys: Iterable[str]) -> None:
    """Log which optimizer parameters a DCP checkpoint covers; its keys flatten to ``state.<param>.<field>``."""
    saved = {key.removeprefix("state.").rpartition(".")[0] for key in checkpoint_keys if key.startswith("state.")}
    saved.discard("")
    missing, extra = current_params - saved, saved - current_params
    if missing:
        logger.warning(
            f"Optimizer {name}: {len(missing)} params in current model "
            f"but missing from checkpoint (will keep initialized values)"
        )
        logger.debug(f"Missing params: {sorted(missing)}")
    if extra:
        logger.warning(
            f"Optimizer {name}: {len(extra)} params in checkpoint but not in current model (will be ignored)"
        )
        logger.debug(f"Extra params: {sorted(extra)}")
    if not missing and not extra:
        logger.info(f"Optimizer {name}: all {len(current_params)} params match checkpoint")


def _validate_dcp_key_coverage(
    module: torch.nn.Module, model_keys: set[str], checkpoint_keys: set[str], strict: bool, name: str
) -> list[str]:
    """Log and check DCP key coverage before copying; return the model keys the checkpoint lacks.

    Strict loads reject unexpected keys. As with ``require_any``, a permissive load must match a
    parameter; strict reads already matched every model key.
    """
    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)
    log = logger.warning if missing_keys or unexpected_keys else logger.success
    log(f"Model {name}, loading info: missing_keys={missing_keys}, unexpected_keys={unexpected_keys}")
    if strict and unexpected_keys:
        raise RuntimeError(f"Unexpected key(s) in checkpoint state_dict: {unexpected_keys}")
    if not strict:
        parameters = module_state_names(module, persistent_buffers=False, nonpersistent_buffers=False)
        parameter_keys = {key for parameter in parameters for key in _get_fqns(module, parameter)}
        if not parameter_keys.intersection(model_keys, checkpoint_keys):
            raise RuntimeError(f"State-dict load matched no target parameters for Model {name}.")
    return missing_keys


def _uncovered_names(module: torch.nn.Module, missing_names: Iterable[str]) -> set[str]:
    """Registration names a load did not supply: its missing keys plus every non-persistent buffer."""
    return {*missing_names, *module_state_names(module, parameters=False, persistent_buffers=False)}


def _validate_absent_roots(model_dict: torch.nn.ModuleDict, absent: list[str]) -> None:
    """Require a monolithic file to match some root, and rank 0 to hold real state for absent roots.

    Absent roots keep their state; their meta ranks receive rank 0's values when the root is prepared.
    """
    if len(absent) == len(model_dict):
        raise RuntimeError(f"Monolithic checkpoint matched no target model roots: {list(model_dict)}.")
    for k in absent:
        validate_module_sync_source(
            model_dict[k], src=0, operation=f"monolithic load: Model {k!r} is absent from the checkpoint"
        )


def _reshard_fsdp_roots(model_dict: torch.nn.ModuleDict) -> None:
    """Release unsharded FSDP2 parameters, e.g. after a forward with ``reshard_after_forward=False``.

    FSDP's state-dict hooks would otherwise swap them for their shards mid-export, which state
    validation reports as a changed registration. Resharding is local, and all ranks save or load.
    """
    from fastgen.utils.distributed.fsdp import reshard_fsdp_modules  # fsdp imports the network modules

    reshard_fsdp_modules(model_dict.values())


def _reject_dtensors(model_dict: torch.nn.ModuleDict, optimizer_dict: Dict[str, torch.optim.Optimizer]) -> None:
    """Require unsharded model and optimizer state for monolithic checkpoint operations."""
    if any(
        isinstance(tensor, DTensor)
        for module in model_dict.values()
        for tensor in (*module.parameters(), *module.buffers())
    ) or any(
        isinstance(parameter, DTensor)
        for optimizer in optimizer_dict.values()
        for group in optimizer.param_groups
        for parameter in group["params"]
    ):
        raise ValueError(
            "Monolithic checkpoints require unsharded model and optimizer state. Load pretrained weights "
            "before FSDP wrapping (trainer.checkpointer.pretrained_ckpt_path); use DCP (save_sharded=True) "
            "to save or resume FSDP training."
        )


class Checkpointer:
    """Save and load model checkpoints through canonical ``.pth`` paths.

    ``save_sharded=True`` saves with DCP: shards go to sibling ``<prefix>.<key>_model`` and ``_optim``
    directories, and the ``.pth`` file holds the scalar state as their completion marker. False saves one
    monolithic file from rank 0, so every root must be unsharded. Loading recognizes both formats,
    regardless of this option. Monolithic files load only into unsharded targets, before FSDP wrapping;
    DCP loads into either unsharded or sharded targets. Every rank reads the .pth file. Modules
    with ``extra_state`` are unsupported, including generic PyTorch modules that have not gone through
    FastGen preparation.
    """

    def __init__(self, config: BaseCheckpointerConfig, save_sharded: bool = False):
        self.config = config
        self.save_sharded = save_sharded

    @staticmethod
    def _scalar_state_dict(
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR],
        grad_scaler: torch.amp.GradScaler | None,
        callbacks: CallbackDict | None,
        iteration: int,
    ) -> Dict[str, Any]:
        """Build checkpoint state shared by monolithic files and DCP markers."""
        return {
            "scheduler": {k: v.state_dict() for k, v in scheduler_dict.items()},
            "grad_scaler": None if grad_scaler is None else grad_scaler.state_dict(),
            "callbacks": None if callbacks is None else callbacks.state_dict(),
            "iteration": iteration,
        }

    @staticmethod
    def _load_scalar_state(
        state: Dict[str, Any],
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR],
        grad_scaler: torch.amp.GradScaler | None,
        callbacks: CallbackDict | None,
    ) -> int:
        """Restore shared scalar state and return the saved iteration."""
        if scheduler_dict:
            logger.info("Loading the scheduler_dict...")
            scheduler_state = state.get("scheduler") or {}
            for k, v in scheduler_dict.items():
                if k in scheduler_state:
                    v.load_state_dict(scheduler_state[k])
                else:
                    logger.warning(f"Scheduler {k} not found in checkpoint.")

        if grad_scaler is not None:
            logger.info("Loading the gradient scaler...")
            grad_scaler_state = state.get("grad_scaler")
            if grad_scaler_state:
                grad_scaler.load_state_dict(grad_scaler_state)
            else:
                logger.warning("Gradient scaler state is empty (likely saved from disabled scaler), skipping load.")

        if callbacks is not None:
            logger.info("Loading the callbacks...")
            callbacks_state = state.get("callbacks")
            if callbacks_state is not None:
                callbacks.load_state_dict(callbacks_state)
            else:
                logger.warning("Callbacks not found in checkpoint.")

        if "iteration" not in state:
            logger.warning("Iteration not found in checkpoint.")
            return 0
        return state["iteration"]

    def resolve_save_path(self, path: str | None, iteration: int) -> str:
        """Return the canonical .pth path, defaulting below ``save_dir`` for this iteration.

        Reads and writes dispatch on the path's storage location, so the path must be absolute
        or an ``s3://`` URI; ``use_s3`` and ``s3_container`` are not consulted here.
        """
        if path is None:
            path = os.path.join(self.config.save_dir, f"{iteration:07d}.pth")
        return _require_storage_location(validate_checkpoint_path(path), "Checkpoint save path")

    def resolve_load_path(self, path: str | None) -> str | None:
        """Resolve a canonical marker path, or discover the latest completed checkpoint in ``save_dir``.

        Discovery requires an absolute or ``s3://`` ``save_dir``. Explicit paths are used as given,
        so a relative one reads from the working directory.
        """
        if path is not None:
            return validate_checkpoint_path(path)
        save_dir = _require_storage_location(self.config.save_dir, "save_dir")
        return validate_checkpoint_path(latest_checkpoint(save_dir))

    def _save_checkpoint(self, save_dict: Dict[str, Any], path: str):
        if s3.is_uri(path):
            # convert the save_dict to bytes
            buffer = io.BytesIO()
            torch.save(save_dict, buffer)
            s3.save(path, buffer.getvalue())
        else:
            # ``*.pth`` files are both scalar state and the completion markers
            # discovered by ``latest_checkpoint``. Publish through a temporary
            # sibling so readers see either the previous complete file or the new
            # complete file, never a partially serialized checkpoint.
            # Iteration checkpoints normally have unique names, so sweeping only
            # ``path`` would never revisit an orphan left by an earlier iteration.
            # One rank and one trainer own this save directory, making it safe to
            # clear every abandoned atomic checkpoint write before publishing.
            parent = os.path.dirname(path) or "."
            for stale_path in remove_atomic_write_orphans(parent, "*.pth"):
                logger.warning(f"Removed incomplete checkpoint temporary file {stale_path}")
            atomic_write(path, lambda temporary_file: torch.save(save_dict, temporary_file), fsync=True)

    def save(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        iteration: int = 0,
    ) -> str:
        """Save a checkpoint of the model (and optionally optimizer, scheduler, and grad scaler to resume training)

        Roots must own independent parameter and buffer objects. Export hooks must preserve
        registrations, tensor identities, and metadata; validation does not roll back hook mutations.
        ``None`` entries in the model, optimizer, and scheduler dicts name objects the caller does not
        hold and are not saved. A model dict whose roots are all ``None`` is rejected; an empty one
        saves scalar state alone.

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to save
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to save
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to save
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to save
            path (str): The path to save the checkpoint file
            iteration (int): The iteration number

        Returns:
            str: The path to the saved checkpoint file
        """
        with raise_across_ranks("checkpoint save target validation"):
            model_dict, optimizer_dict, scheduler_dict = _held_targets(
                model_dict, optimizer_dict, scheduler_dict, "checkpoint save"
            )
            reject_extra_state(model_dict, "checkpoint save")
            _reshard_fsdp_roots(model_dict)
            path = self.resolve_save_path(path, iteration)
            # Each save path's state snapshot also rejects unsupported aliases before any export.
            validate_module_roots(model_dict, "checkpoint save")
            if not self.save_sharded:
                # Rank 0 writes monolithic files alone and would save only its own shards.
                _reject_dtensors(model_dict, optimizer_dict)
            plan = _checkpoint_plan(
                path,
                model_dict,
                optimizer_dict,
                format="dcp" if self.save_sharded else "monolithic",
                adapter_only=self.config.save_adapter_only,
                iteration=iteration,
            )
        validate_object_across_ranks(plan, operation="checkpoint save plan")
        save = self._save_dcp if self.save_sharded else self._save_monolithic
        return save(model_dict, optimizer_dict, scheduler_dict, grad_scaler, callbacks, path, iteration)

    def _save_monolithic(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR],
        grad_scaler: torch.amp.GradScaler | None,
        callbacks: CallbackDict | None,
        path: str,
        iteration: int,
    ) -> str:
        """Write every model and optimizer state with the scalar state into one ``.pth`` file from rank 0."""
        # Rank 0 publishes while the other ranks wait in the poll, which also replaces a final barrier.
        with raise_across_ranks(f"publish checkpoint {path}"):
            if is_rank0():
                # All exports are local. Validate the complete phase before publishing any bytes.
                with validate_module_state_preserved(model_dict, "checkpoint state export"):
                    model_state = {}
                    for k, v in model_dict.items():
                        if self.config.save_adapter_only and getattr(v, "lora_enabled", False):
                            state_with_vars = v.state_dict(keep_vars=True)
                            validate_state_dict_export(v, state_with_vars)
                            model_state[k] = v.adapter_state_dict()
                            _validate_adapter_state_dict(v, state_with_vars, model_state[k])
                            logger.info(f"Checkpointing adapter-only for '{k}': {len(model_state[k])} tensors")
                        else:
                            # No adapter (e.g. an EMA copy or a plain net): keep the full save.
                            model_state[k] = v.state_dict()
                            state_with_vars = v.state_dict(keep_vars=True)
                            validate_state_dict_export(v, state_with_vars, model_state[k])
                    save_dict = {
                        "model": model_state,
                        "optimizer": {k: v.state_dict() for k, v in optimizer_dict.items()},
                        **self._scalar_state_dict(scheduler_dict, grad_scaler, callbacks, iteration),
                    }

                logger.info(f"Saving the model to {path}")
                self._save_checkpoint(save_dict, path)

        logger.success(f"Model saved at iteration {iteration}")
        return path

    def load(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        device: torch.device | str = "cpu",
        strict: bool = True,
    ) -> int:
        """Load the model checkpoint.

        Loading is not transactional: a failure may leave earlier tensors, models, optimizers,
        or scalar state changed. Discard or restore the entire target collection before reuse.

        Every rank reads the file. Monolithic checkpoints require unsharded model and optimizer
        targets; load pretrained weights before FSDP wrapping and use DCP for full FSDP resume.
        ``None`` entries in the model, optimizer, and scheduler dicts name objects the caller does not
        hold and receive nothing; the checkpoint's state for them is ignored. A model dict whose roots
        are all ``None`` is rejected; an empty one restores scalar state alone.

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to load
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to load
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to load
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to load
            path (str): A monolithic ``.pth`` checkpoint file or DCP completion marker. None loads the latest
                completed checkpoint in ``save_dir``, if any.
            device (torch.device | str): Where the ``.pth`` file's tensors are placed before copying; DCP
                shards load straight into the targets. Targets keep their placement and dtypes and receive values by
                copy, except loaded roots that hold meta tensors on any rank: every rank first places those
                on this device, materializing the meta tensors. Defaults to "cpu".
            strict (bool): Whether state-dict keys must match exactly. In strict mode, an adapter requires
                all adapter tensors, and loading any optional base parameter requires the complete optional
                base state. Optional buffers alone may load without the base parameters.

        Returns:
            int: The saved iteration, or 0 when no checkpoint is found.

        Raises:
            ValueError: If ``path`` does not name a ``.pth`` file, or discovery needs an absolute or
                ``s3://`` ``save_dir``, or monolithic loading is requested for FSDP/DTensor targets, or
                every root of a nonempty ``model_dict`` is ``None``.
            FileNotFoundError: If the checkpoint file is missing, or a DCP checkpoint lacks the shards of a
                requested model.
            RuntimeError: If the checkpoint matches no target parameter, or its keys mismatch under
                ``strict``. FastGen's distributed validation raises ``RuntimeError`` on every rank,
                chained to the original error on the failing rank; DCP read failures raise DCP's
                ``CheckpointException``.
        """
        with raise_across_ranks("checkpoint load setup"):
            model_dict, optimizer_dict, scheduler_dict = _held_targets(
                model_dict, optimizer_dict, scheduler_dict, "checkpoint load"
            )
            reject_extra_state(model_dict, "checkpoint load")
            _reshard_fsdp_roots(model_dict)
            path = self.resolve_load_path(path)
            state = None if path is None else load_training_checkpoint(path, device=device)
            checkpoint_format, source_plan = None, None
            if state is not None:
                if is_dcp_marker_state(state):
                    if "dcp_manifest" not in state:
                        state["dcp_manifest"] = self._legacy_dcp_manifest(path, model_dict, optimizer_dict)
                    checkpoint_format, source_plan = "dcp", state["dcp_manifest"]
                else:
                    _reject_dtensors(model_dict, optimizer_dict)
                    checkpoint_format = "monolithic"
                    source_plan = (tuple(state["model"]), tuple(state.get("optimizer") or {}))
            plan = _checkpoint_plan(
                path,
                model_dict,
                optimizer_dict,
                format=checkpoint_format,
                strict=strict,
                device=torch.device(device).type,
                iteration=None if state is None else state.get("iteration"),
                source=source_plan,
            )
        validate_object_across_ranks(plan, operation="checkpoint load plan")
        if path is None:
            # no checkpoint found, starting from iteration 0
            return 0
        logger.info(f"Loading model from {path}")
        restore = self._load_dcp_state if checkpoint_format == "dcp" else self._load_monolithic_state
        return restore(state, model_dict, optimizer_dict, scheduler_dict, grad_scaler, callbacks, path, device, strict)

    def _load_monolithic_state(
        self,
        state: Dict[str, Any],
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR],
        grad_scaler: torch.amp.GradScaler | None,
        callbacks: CallbackDict | None,
        path: str,
        device: torch.device | str,
        strict: bool,
    ) -> int:
        """Restore an already-read monolithic file.

        At least one requested root must match unless no model roots are requested. Absent roots keep
        their state. Their meta ranks receive rank 0's values when the root is prepared, so rank 0 must
        hold real state for every absent root.
        """
        logger.info(f"Loading monolithic checkpoint from {path}")
        present = [k for k in model_dict if k in state["model"]]
        absent = [k for k in model_dict if k not in state["model"]]
        if absent:
            logger.warning(f"Model(s) {absent} not found in checkpoint.")
            with raise_across_ranks("monolithic load absent roots"):
                _validate_absent_roots(model_dict, absent)

        snapshot, needs_meta_sync_by_name = _prepare_load_targets(
            model_dict, target_names=present, optimizer_dict=optimizer_dict, device=device, operation="monolithic load"
        )

        logger.info("Loading the model_dict...")
        uncovered_by_name = {}
        with raise_across_ranks("monolithic model load"), snapshot.validate_after():
            for k in present:
                v = model_dict[k]
                required_keys = None
                if getattr(v, "lora_enabled", False):
                    state_with_vars = v.state_dict(keep_vars=True)
                    validate_state_dict_export(v, state_with_vars)
                    adapter_state = v.adapter_state_dict()
                    _validate_adapter_state_dict(v, state_with_vars, adapter_state)
                    required_keys = _registration_names_for_dcp_keys(v, adapter_state)
                model_load_info = load_module_state_dict(
                    v,
                    state["model"][k],
                    strict=strict,
                    require_any=True,
                    description=f"Model {k}",
                    # The adapter is required; an omitted frozen base keeps its pretrained values.
                    required_keys=required_keys,
                    check_state_preservation=False,  # The collection snapshot guards the complete load phase.
                )
                if needs_meta_sync_by_name[k]:
                    uncovered_by_name[k] = _uncovered_names(v, model_load_info.missing_keys)

        # Finish all local copies and checks before broadcasting the values the checkpoint did
        # not supply on materialized ranks.
        for k, uncovered in uncovered_by_name.items():
            logger.info(f"Model {k}: broadcasting {len(uncovered)} uncovered tensors from rank 0.")
            broadcast_module(model_dict[k], names=uncovered)

        with raise_across_ranks("monolithic optimizer and scalar load"), snapshot.validate_after():
            if optimizer_dict:
                logger.info("Loading the optimizer_dict...")
                optimizer_state = state.get("optimizer") or {}
                for k, v in optimizer_dict.items():
                    if k in optimizer_state:
                        v.load_state_dict(optimizer_state[k])
                    else:
                        logger.warning(f"Optimizer {k} not found in checkpoint.")

            return self._load_scalar_state(state, scheduler_dict, grad_scaler, callbacks)

    def get_storage_writer(self, checkpoint_path: str) -> Union[S3StorageWriter, FileSystemWriter]:
        if s3.is_uri(checkpoint_path):
            return S3StorageWriter(path=checkpoint_path)
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(self, checkpoint_path: str) -> Union[S3StorageReader, FileSystemReader]:
        if s3.is_uri(checkpoint_path):
            return S3StorageReader(path=checkpoint_path)
        return FileSystemReader(checkpoint_path)

    def _save_dcp(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR],
        grad_scaler: torch.amp.GradScaler | None,
        callbacks: CallbackDict | None,
        path: str,
        iteration: int,
    ) -> str:
        """Save model and optimizer shards with DCP, then publish the scalar ``.pth`` marker from rank 0."""
        # The trainer also rejects this combination at startup.
        if self.config.save_adapter_only:
            raise NotImplementedError("save_adapter_only is not supported with save_sharded=True (DCP checkpointing)")
        prefix = path.removesuffix(".pth")
        logger.info(f"Saving DCP checkpoint to prefix {prefix}")

        # Export every root before the first write. An export hook on a later root can mutate an
        # earlier one, so validate the whole collection. These dictionaries retain tensor views;
        # validation never clones model or optimizer payloads. Default sharded FSDP2 export is local.
        with raise_across_ranks("DCP model state export"):
            snapshot = ModuleStateSnapshot.capture(model_dict, "checkpoint state export")
            with snapshot.validate_after():
                model_exports = {}
                for k, v in model_dict.items():
                    model_exports[k] = get_model_state_dict(v)
                    state_with_vars = v.state_dict(keep_vars=True)
                    validate_state_dict_export(
                        v, state_with_vars, model_exports[k], key_mapping=lambda name: _get_fqns(v, name)
                    )

        # Agree on optimizer participation before the local export phase. Conversion can initialize
        # an unstepped optimizer, so vote on raw state first, including rank-local MoE optimizers.
        saved_optimizers = {}
        for k, v in optimizer_dict.items():
            if sync_any(bool(v.state), device=torch.device("cpu")):
                saved_optimizers[k] = v
            else:
                logger.warning(f"Optimizer '{k}' has no state yet (never stepped); skipping its checkpoint.")

        with raise_across_ranks("DCP save setup"), snapshot.validate_after():
            optimizer_exports = {k: get_optimizer_state_dict(model_dict[k], v) for k, v in saved_optimizers.items()}
            if is_rank0():
                save_dict = self._scalar_state_dict(scheduler_dict, grad_scaler, callbacks, iteration)
                save_dict["dcp_manifest"] = {
                    "models": tuple(model_exports),
                    "optimizers": tuple(optimizer_exports),
                }

            if not s3.is_uri(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            model_writers = {k: self.get_storage_writer(f"{prefix}.{k}_model") for k in model_exports}
            optimizer_writers = {k: self.get_storage_writer(f"{prefix}.{k}_optim") for k in optimizer_exports}

        # Setup failures leave the previous checkpoint discoverable. Once any shard changes,
        # the marker must stay absent until every write succeeds. Its manifest excludes old shards.
        if model_exports or optimizer_exports:
            with raise_across_ranks(f"invalidate DCP checkpoint marker {path}"):
                if is_rank0():
                    _remove_checkpoint_artifact(path)

        # A failed shard write raises DCP's CheckpointException and leaves the marker absent.
        for k, model_state in model_exports.items():
            dcp.save(model_state, storage_writer=model_writers[k])
        for k, optim_state in optimizer_exports.items():
            dcp.save(optim_state, storage_writer=optimizer_writers[k])

        with raise_across_ranks(f"publish DCP checkpoint marker {path}"):
            if is_rank0():
                self._save_checkpoint(save_dict, path)

        logger.success(f"Model saved at iteration {iteration}")
        return path

    @staticmethod
    def _legacy_dcp_manifest(
        marker_path: str, model_keys: Iterable[str], optimizer_keys: Iterable[str]
    ) -> dict[str, tuple[str, ...]]:
        """LEGACY: rebuild the shard manifest of a DCP marker saved before markers recorded one.

        Older versions, including ``main``, wrote no manifest. Like them, every requested model must
        have shards, and an optimizer loads exactly when its ``.metadata`` exists. Unlike a recorded
        manifest, this cannot exclude stale optimizer shards that an earlier save left under the same
        prefix. The result joins the cross-rank load plan, so ranks with different shard views fail
        before loading. Remove once pre-manifest checkpoints no longer need to load.
        """
        prefix = marker_path.removesuffix(".pth")
        logger.warning(
            f"{marker_path}: legacy DCP marker without a shard manifest; inferring optimizer shards from their metadata."
        )
        return {
            "models": tuple(model_keys),
            "optimizers": tuple(
                k for k in optimizer_keys if path_exists(os.path.join(f"{prefix}.{k}_optim", ".metadata"))
            ),
        }

    def _load_dcp_state(
        self,
        state: Dict[str, Any],
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR],
        grad_scaler: torch.amp.GradScaler | None,
        callbacks: CallbackDict | None,
        path: str,
        device: torch.device | str,
        strict: bool,
    ) -> int:
        """Restore a DCP checkpoint's model and optimizer shards from its already-read marker."""
        prefix = path.removesuffix(".pth")
        manifest = state["dcp_manifest"]  # Written by ``_save_dcp`` or rebuilt by ``_legacy_dcp_manifest``.
        with raise_across_ranks("DCP shard availability"):
            absent = [k for k in model_dict if k not in manifest["models"]]
            if absent:
                raise FileNotFoundError(f"{path}: model key(s) {absent} are absent from the DCP shard manifest.")
            missing_shards = [
                k for k in model_dict if not path_exists(os.path.join(f"{prefix}.{k}_model", ".metadata"))
            ]
            if missing_shards:
                raise FileNotFoundError(
                    f"{path} is neither a monolithic checkpoint (it has no 'model' entry) nor a complete DCP "
                    f"checkpoint: {prefix}.<key>_model/.metadata is missing for model key(s) {missing_shards}. Either "
                    "the checkpoint was saved without these models, or their shard directories were not copied or "
                    "symlinked along with the .pth file."
                )
            missing_optimizers = [
                k
                for k in optimizer_dict
                if k in manifest["optimizers"] and not path_exists(os.path.join(f"{prefix}.{k}_optim", ".metadata"))
            ]
            if missing_optimizers:
                raise FileNotFoundError(
                    f"{path}: optimizer shard(s) {missing_optimizers} listed in the DCP manifest are missing."
                )
        logger.info(f"Loading DCP checkpoint from prefix {prefix}")

        snapshot, needs_meta_sync_by_name = _prepare_load_targets(
            model_dict, target_names=model_dict, optimizer_dict=optimizer_dict, device=device, operation="DCP load"
        )

        for k, v in model_dict.items():
            needs_meta_sync = needs_meta_sync_by_name[k]

            logger.info(f"Loading the DCP model dict for key {k}...")
            with raise_across_ranks(f"DCP model load setup {k}"), snapshot.validate_after():
                model_state_dict = get_model_state_dict(v)
                state_with_vars = v.state_dict(keep_vars=True)
                validate_state_dict_export(
                    v, state_with_vars, model_state_dict, key_mapping=lambda name: _get_fqns(v, name)
                )
                storage_reader = self.get_storage_reader(checkpoint_path=f"{prefix}.{k}_model")
                load_planner = DefaultLoadPlanner(allow_partial_load=not strict)
            dcp.load(
                state_dict=model_state_dict,
                storage_reader=storage_reader,
                planner=load_planner,
            )
            # The default sharded FSDP2 setter copies locally. Validate the complete local phase
            # before the next broadcast or DCP read; a failed read still skips copy loading.
            with raise_across_ranks(f"DCP model load {k}"), snapshot.validate_after():
                missing_keys = _validate_dcp_key_coverage(
                    v, set(load_planner.state_dict), set(load_planner.metadata.state_dict_metadata), strict, k
                )
                set_model_state_dict(v, model_state_dict=model_state_dict)
                if needs_meta_sync:
                    uncovered = _uncovered_names(v, _registration_names_for_dcp_keys(v, missing_keys))

            if needs_meta_sync:
                # Synchronize missing keys and nonpersistent buffers from rank 0 after loading;
                # covered tensors already have their final values.
                logger.info(f"Model {k}: broadcasting {len(uncovered)} uncovered tensors from rank 0.")
                broadcast_module(v, names=uncovered)

        for k, v in optimizer_dict.items():
            # Only the marker determines membership; an older save may have left shards here.
            if k not in manifest["optimizers"]:
                logger.warning(
                    f"Optimizer {k} is absent from the DCP shard manifest (not saved or never stepped); "
                    "skipping its load."
                )
                continue
            logger.info(f"Loading the DCP optimizer dict for key {k}...")
            with raise_across_ranks(f"DCP optimizer load setup {k}"), snapshot.validate_after():
                optim_state_dict = get_optimizer_state_dict(model_dict[k], v)
                current_param_names = set(optim_state_dict.get("state", {}).keys())
                storage_reader = self.get_storage_reader(checkpoint_path=f"{prefix}.{k}_optim")
                load_planner = DefaultLoadPlanner(allow_partial_load=True)

            dcp.load(
                state_dict=optim_state_dict,
                storage_reader=storage_reader,
                planner=load_planner,
            )
            with raise_across_ranks(f"DCP optimizer load {k}"), snapshot.validate_after():
                _log_optimizer_coverage(k, current_param_names, load_planner.metadata.state_dict_metadata)
                set_optimizer_state_dict(model_dict[k], v, optim_state_dict=optim_state_dict)
            logger.success(f"Successfully loaded optimizer state for {k}")

        with raise_across_ranks("DCP scalar load"), snapshot.validate_after():
            return self._load_scalar_state(state, scheduler_dict, grad_scaler, callbacks)
