# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``WandbCallback`` — single callback that owns media writes + W&B logging.

Each media step (every ``sample_logging_iter`` training iteration, which
defaults to ``logging_iter``, and every ``validation_logging_step``
validation step) the callback:

1. allocates a fresh per-step shared dir under ``media_path``
   (``iter{iteration}_{group}{suffix}/``);
2. every rank decodes the latents through the model VAE
   (``vae_decode_context``) and writes the per-rank slice of every panel
   (student / real / source / teacher / reward / student_train) to disk as
   ``<key>/frame_<rank:03d>_<batch_idx:03d>.{png,mp4,gif,wav}`` — at most
   ``log_batchsize`` items per rank (capped by the local batch size);
3. all-gathers each rank's ``MediaEntry`` records (kind, rank, batch_idx,
   path, prompt) and each registered ``LogModule.collect(...)`` result so
   rank 0 sees every rank's writes;
4. on rank 0 only:
   - invokes each registered ``LogModule.process(...)`` once with the
     gathered entries (and the per-rank ``collect`` outputs). Each
     module may return a per-rank ``loss_dict`` (aggregated cross-rank
     into the ``_LossDictRecord`` and surfaced as scalar W&B logs) and
     /or call ``wandb.log`` directly to ship tables / extra panels;
   - ships every kind's media files as ``wandb.Image`` / ``wandb.Video`` / ``wandb.Audio``
     panels keyed by ``{group}_media/{prefix}{suffix}`` (see
     ``_KEY_PREFIXES``), captioned with each entry's own ``prompt``.

Every ``logging_iter`` training iteration and at the end of each
validation run, it logs the aggregated loss + log-module-loss means via
the standard ``_LossDictRecord`` collective (rank 0 logs the result).
Training iteration 1 logs both media and scalars.

Only rank 0 has a live ``wandb.run`` — wandb is initialized rank-0-only
(``init_wandb`` is decorated with ``@rank0_only``). Every cross-rank
piece of information rank 0 needs (media entries, log-module rows) is
gathered explicitly via ``dist.all_gather_object``; media files are read
from the shared FS the per-rank writes already populated.

When a participating VAE or teacher is FSDP-sharded, callers must provide
the same ordered media/decode plan on every rank. In particular, ranks must
agree on whether their logged slice is empty and on the presence and media
structure of fields that trigger student, real, source, teacher, reward, or
debug decode calls. Tensor values and nonempty local batch sizes may differ.
``max_ranks`` only filters entries after decode and serialization; it does
not relax this rank-symmetric call-order contract.

The disk artifacts persist across runs (``delete_previous=False`` by
default) so post-hoc inspection / rescoring tools can find them.
"""

from __future__ import annotations

import gc
import os
import secrets
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_lowercase, digits
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING

import torch
import torch.distributed as dist

import wandb

from collections import abc

from fastgen.callbacks.callback import Callback
from fastgen.configs.config_utils import serialize_config
from fastgen.configs.instantiate import instantiate as _fg_instantiate
from fastgen.utils import basic_utils
from fastgen.utils.env import str2bool
from fastgen.utils.media_utils import save_media
from fastgen.utils.tree_utils import (
    TensorOrDict,
    get_batch_size,
    slice_dim,
)
from fastgen.utils.distributed import (
    get_rank,
    is_rank0,
    rank0_only,
    world_size,
)
from fastgen.utils.distributed.collectives import raise_across_ranks
from fastgen.utils.module import reject_unsafe_placement
from fastgen.utils import logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.config import BaseConfig
    from fastgen.methods import FastGenModel


# ---------------------------------------------------------------------------
# On-disk layout — internal to the callback. Log modules consume already-
# resolved ``MediaEntry`` records (see below) and don't need to know the
# disk prefix or filename convention.

_KEY_PREFIXES: Dict[str, str] = {
    "student": "student/generation",
    "real": "data/real",
    "source": "data/source",
    "teacher": "teacher/generation",
    "reward": "student/reward_frames",
    "student_train": "student/generation_train",
}

# Media extensions recognized by this callback's W&B panel logger.
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
_VIDEO_SUFFIXES = frozenset({".mp4", ".gif"})


def is_savable(x: Any) -> bool:
    """Whether *x* can be handed to ``_save_batch`` as audio, image, or video media."""
    if isinstance(x, torch.Tensor):
        return x.ndim in (2, 3, 4, 5)
    return isinstance(x, dict) and bool({"video", "image", "audio"} & x.keys())


class MediaEntry(NamedTuple):
    """One ``(kind, rank, batch_idx)`` media artifact + its caption.

    Built per-rank inside ``WandbCallback._do_save`` and all-gathered to
    rank 0 in ``_step`` so log modules and the panel logger receive a
    single flat ``List[MediaEntry]`` covering every rank's writes — no
    filename parsing, no parallel prompts/files data structures.

    * ``kind`` — one of ``"student"`` / ``"real"`` / ``"source"`` /
      ``"teacher"`` / ``"reward"`` / ``"student_train"``.
    * ``rank`` / ``batch_idx`` — the ``(rank, batch_idx)`` index this
      entry was written under.
    * ``path`` — absolute on-disk path. Stable across the per-step
      window; safe to wrap in ``wandb.Image`` / ``wandb.Video`` / ``wandb.Audio``.
    * ``prompt`` — the corresponding caption (empty string when the
      rank's local ``data_batch`` had nothing for this index).
    """

    kind: str
    rank: int
    batch_idx: int
    path: Path
    prompt: str


# ---------------------------------------------------------------------------
# Helpers.


def extract_prompt(data_batch: Optional[Dict[str, Any]], idx: int = 0) -> str:
    """Look up the raw prompt at ``idx`` in the rank's local batch."""
    if not data_batch:
        return ""
    val = data_batch.get("condition_raw")
    if isinstance(val, (list, tuple)) and len(val) > idx and isinstance(val[idx], str):
        return val[idx]
    return ""


def _all_gather_object(local: Any) -> List[Any]:
    """Cross-rank ``all_gather_object`` returning ``[per-rank value, ...]``.

    Falls back to ``[local]`` when distributed is not initialized
    (single-process runs).
    """
    ws = world_size()
    if ws <= 1:
        return [local]
    gathered: List[Any] = [None] * ws
    dist.all_gather_object(gathered, local)
    return gathered


# ---------------------------------------------------------------------------
# vae_decode_context — convenience for downstream pixel-space callsites.


@contextmanager
def vae_decode_context(
    model: "FastGenModel",
    offload_modules: bool = False,
) -> Iterator[Optional[Callable]]:
    """Yield the VAE preprocessor (when the model has a preprocessing pipeline)
    so the caller can decode latents → pixels inside a single ``with`` block.

    Yields ``None`` immediately when the net is already pixel-space, so the
    caller doesn't have to branch (the pixel-space path skips the
    ``inference_mode`` wrap). Optionally offloads registered components not
    needed for decoding to CPU (frees GPU memory at the cost of two H↔D
    copies).
    """
    net = model.net_preprocessors
    if not hasattr(net, "init_vae"):
        yield None
        return

    device = model.device
    offloaded: List[torch.nn.Module] = []
    if offload_modules:
        # Preprocessors live outside their host network's registered tree, so moving the host
        # does not move its VAE. Keep the VAE, the only registered root the decode path calls.
        offloaded = [module for name, module in model.module_dict.items() if name != "vae"]

    def validate(operation: str) -> None:
        with raise_across_ranks(f"{operation} validation"):
            for module in offloaded:
                reject_unsafe_placement(module, operation)

    def move(target: torch.device | str, operation: str) -> None:
        before = torch.cuda.memory_allocated(device)
        with raise_across_ranks(operation):
            for module in offloaded:
                module.to(target)
        logger.info(
            f"GPU Memory BEFORE/AFTER moving components to {target}: "
            f"{before / 1024 ** 2:.2f} / {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB"
        )

    if offloaded:
        # Validate every component before entering the cleanup scope: a rejected request must
        # neither offload an earlier component nor attempt to restore an invalid one in finally.
        validate("VAE decode offload")
    try:
        if offloaded:
            move("cpu", "VAE decode offload")

        # Acquire after offloading to avoid overlapping a temporary VAE's peak
        # allocation with the generator components being moved away.
        with net.vae_context() as vae:
            with basic_utils.inference_mode(
                precision_amp=model.precision_amp_enc,
                device_type=device.type,
            ):
                yield vae
    finally:
        if offloaded:
            validate("VAE decode restore")
            move(device, "VAE decode restore")


# ---------------------------------------------------------------------------
# wandb init — rank-0-only.


# A wandb run id is just a random base-36 string; nothing on the server side parses it.
# `wandb.util.generate_id` was removed in wandb 0.28 and the implementation moved to the private
# `wandb.sdk.lib.runid`, so copy it instead (verbatim from wandb 0.28.2 `wandb/sdk/lib/runid.py`).
_WANDB_ID_CHARS = f"{ascii_lowercase}{digits}"


def _generate_wandb_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits, as wandb does.

    ~2.8T 8-digit ids; ~1% chance of a collision after 210k of them.
    """
    return "".join(secrets.choice(_WANDB_ID_CHARS) for _ in range(length))


@rank0_only
def init_wandb(config: "BaseConfig") -> None:
    """Initialize the wandb run on rank 0.

    No-op on non-rank-0 ranks (via ``@rank0_only``), so ``wandb.run``
    only ever exists on rank 0. Every chartable user metric flows
    through rank 0; cross-rank artifacts (prompts, log-module rows)
    are gathered explicitly by callers.
    """
    wandb_config = config.log_config

    logger.info(f"wandb_config.save_path: {wandb_config.save_path}")
    os.makedirs(wandb_config.save_path, exist_ok=True)
    wandb_id_path = f"{wandb_config.save_path}/wandb_id.txt"
    if os.path.isfile(wandb_id_path):
        with open(wandb_id_path, encoding="utf-8") as f:
            wandb_id = f.read().strip()
        logger.info(f"Resuming with an existing wandb id: {wandb_id}")
    else:
        wandb_id = _generate_wandb_id()
        with open(wandb_id_path, "w", encoding="utf-8") as f:
            f.write(f"{wandb_id}\n")
        logger.info(f"Generating a wandb id: {wandb_id}")

    config_resolved = serialize_config(config, return_type="dict")

    wandb.init(
        id=wandb_id,
        project=wandb_config.project,
        group=wandb_config.group,
        name=wandb_config.name,
        config=config_resolved,
        dir=wandb_config.save_path,
        resume="allow",
        mode=wandb_config.wandb_mode,
    )

    upload_code = str2bool(os.getenv("WANDB_UPLOAD_CODE", "false"))
    if upload_code:
        logger.info("Uploading code to wandb (this may take a few minutes)...")
        wandb.run.log_code(".")
        logger.info("Code upload to wandb completed")
    else:
        logger.info("Wandb code upload disabled (set WANDB_UPLOAD_CODE=true to enable)")


# ---------------------------------------------------------------------------
# Loss aggregator — averages each metric over the ranks that reported it.
# Used both for in-memory training losses and for ``losses`` dicts returned
# by ``LogModule.process(...)``.


@dataclass
class _LossDictRecord:
    loss_dict: dict = field(default_factory=dict)
    iter_count_dict: dict = field(default_factory=dict)

    def add(self, loss_dict: Optional[Dict[str, Any]]) -> None:
        if loss_dict is None:
            return
        for name, val in loss_dict.items():
            if isinstance(val, torch.Tensor):
                val = val.float().item()
            else:
                val = float(val)
            self.loss_dict[name] = self.loss_dict.get(name, 0.0) + val
            self.iter_count_dict[name] = self.iter_count_dict.get(name, 0) + 1

    def reset(self) -> None:
        self.loss_dict = {}
        self.iter_count_dict = {}

    def gather_dict(
        self,
        dictionary: Dict[str, float | int],
    ) -> Dict[str, float | int]:
        n_ranks = world_size()
        if n_ranks > 1:
            dict_list: List[Optional[Dict[str, float | int]]] = [None] * n_ranks
            dist.all_gather_object(dict_list, dictionary)
            dictionary = {}
            for d in dict_list:
                if d is None:
                    continue
                for key, value in d.items():
                    dictionary[key] = dictionary.get(key, 0.0) + value
        return dictionary

    def get_stat(self) -> Dict[str, float]:
        # Average each key over the ranks that actually reported it.
        # Sparse-key safe: a key reported by only a subset of ranks is
        # averaged over the reporting ranks rather than zero-filled.
        rank_dict = self.gather_dict({k: 1 for k in self.loss_dict.keys()})
        count_dict = self.gather_dict(self.iter_count_dict)
        loss_dict = self.gather_dict(self.loss_dict)

        avg_loss_dict = {}
        for name, total in loss_dict.items():
            count = count_dict.get(name, 0)
            if count <= 0:
                continue
            avg = total / count
            ranks = rank_dict.get(name, 1)
            iter_count = count / ranks
            logger.info(f"avg_{name}: {avg:.4f}".ljust(30) + f"iter count: {iter_count}")
            avg_loss_dict[name] = avg
        self.reset()
        return avg_loss_dict


# ---------------------------------------------------------------------------
# LogModule plug-in surface.


class LogModule:
    """Plug-in consumer of the per-step disk artifacts written by
    ``WandbCallback``.

    All ``LogModule`` lifecycle methods (``setup`` / ``process`` /
    ``teardown``) run **only on rank 0**. Each call to ``process``
    receives a single flat ``List[MediaEntry]`` covering every rank's
    writes, all-gathered cross-rank by the parent callback before the
    invocation. Each entry carries its own ``(kind, rank, batch_idx,
    path, prompt)`` so modules don't have to parse filenames or join
    parallel prompt / file structures.

    Modules may return a ``loss_dict`` from ``process(...)`` — the
    parent aggregates it cross-rank via ``_LossDictRecord`` and
    surfaces the means as scalar W&B logs alongside the training
    losses. They may also call ``wandb.log`` directly (rank 0 has the
    only live ``wandb.run``) to ship tables / extra panels.
    """

    def setup(self, callback: "WandbCallback") -> None:
        """Called once on rank 0 from ``WandbCallback.on_app_begin``."""
        self.callback = callback

    def teardown(self) -> None:
        """Called rank-0-only from ``on_validation_end`` and ``on_app_end``.

        Use to release resources (network clients, file handles) that
        should not survive a fork.
        """

    def collect(self, data_batch: Optional[Dict[str, Any]]) -> Any:
        """Called on **every** rank (collective-safe) before ``process``.

        Return this rank's per-rank inputs to be all-gathered to rank 0 and
        handed to ``process`` as ``gathered`` (a list indexed by rank). Use it
        for diagnostics that need data living only in non-rank-0 ranks' local
        ``data_batch`` — e.g. the per-rank camera poses the ``VipeLogger`` needs
        to overlay *every* rank's clip, not just rank 0's. Must be
        collective-uniform: every rank returns a pickleable object (or ``None``)
        and participates. Default: nothing to gather.
        """
        return None

    def process(
        self,
        save_dir: Path,
        entries: List[MediaEntry],
        iteration: int,
        group: str,
        step: int,
        data_batch: Optional[Dict[str, Any]],
        output_batch: Optional[Dict[str, Any]],
        gathered: Optional[List[Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """Per-step processing hook (rank 0 only). Override in subclasses.

        ``gathered`` is the per-rank list of this module's ``collect`` outputs
        (indexed by rank), all-gathered cross-rank by the parent callback.

        Return an optional ``loss_dict`` for cross-rank aggregation or
        ``None`` to opt out of scalar W&B logging.
        """
        return None


# ---------------------------------------------------------------------------
# WandbCallback.


class WandbCallback(Callback):
    """One-stop W&B callback: every rank writes per-rank media to a shared
    per-step dir; rank 0 reads it back, runs registered ``LogModule``
    plug-ins, and logs panels + scalar means to W&B.

    Constructor knobs (all optional):

    * ``media_path`` — root dir for per-step media (default
      ``<log_config.save_path>/media``).
    * ``logging_iter`` — cadence at which scalar losses (training +
      log-module + LR) get logged (default ``trainer.logging_iter``).
    * ``sample_logging_iter`` — cadence at which media gets written +
      logged (default ``logging_iter``). Set larger than ``logging_iter``
      to throttle expensive decode + W&B uploads without losing
      high-frequency scalar telemetry.
    * ``validation_logging_step`` — log validation panels every Nth val
      step (everything below the gate is skipped on unrecorded steps).
    * ``log_validation_media`` — whether validation steps decode and upload
      media panels (default ``True``). Validation scalar losses are still
      logged from ``on_validation_end`` when this is disabled.
    * ``vid_format`` — container for 5D tensors, ``"mp4"`` (default) or
      ``"gif"``. Audio-video outputs are always saved as ``"mp4"``.
    * ``max_ranks`` — number of ranks whose media is sent to W&B and log
      modules each step. ``None`` means every rank's media is included.
      Every rank still decodes and writes its uniquely named local files;
      ranks ``r >= max_ranks`` discard only their ``MediaEntry`` records.
      It does not permit ranks to skip decode work or use different media
      schemas when a participating module is FSDP-sharded.
    * ``log_batchsize`` — per-rank batch slice written each step
      (default 1). Capped by the rank's local batch size.
    * ``delete_previous`` — rank-0 ``rmtree`` of the prior step's dir on
      each new write (off by default so post-hoc inspection works).
    * ``log_modules`` — name-keyed mapping of ``LogModule`` instances to
      run on each per-step write (rank 0 only). The keys are stable
      handles configs can use to override knobs without positional
      indexing — e.g.,
      ``config.trainer.callbacks.wandb.log_modules["vlm"].max_samples
      = 8``. Iteration order is dict insertion order (Python 3.7+).
    """

    def __init__(
        self,
        media_path: Optional[str] = None,
        logging_iter: Optional[int] = None,
        sample_logging_iter: Optional[int] = None,
        validation_logging_step: int = 1,
        log_validation_media: bool = True,
        vid_format: str = "mp4",
        max_ranks: Optional[int] = 1,
        log_batchsize: int = 1,
        delete_previous: bool = False,
        log_modules: Optional[Dict[str, LogModule]] = None,
    ):
        super().__init__()
        if max_ranks is not None and max_ranks < 1:
            raise ValueError(f"WandbCallback.max_ranks must be None or >= 1, got {max_ranks}")
        if log_batchsize < 1:
            raise ValueError(f"WandbCallback.log_batchsize must be >= 1, got {log_batchsize}")
        self.media_path: Optional[Path] = Path(media_path) if media_path is not None else None
        self.logging_iter: Optional[int] = logging_iter
        self.sample_logging_iter: Optional[int] = sample_logging_iter
        self.validation_logging_step = validation_logging_step
        self.log_validation_media = log_validation_media
        if vid_format not in ("mp4", "gif"):
            raise ValueError(f"WandbCallback.vid_format must be 'mp4' or 'gif', got {vid_format!r}")
        self.vid_format = vid_format
        self.max_ranks: Optional[int] = max_ranks
        self.log_batchsize = log_batchsize
        self.delete_previous = delete_previous
        # log_modules values may arrive as already-instantiated LogModule
        # instances (e.g. when callers pre-build the dict) or as LazyCall
        # DictConfigs with a ``_target_`` (the canonical config-time form
        # produced by ``VLM_IMG_CALLBACK`` / ``VLM_VID_CALLBACK``). Resolve
        # the LazyCall form locally so the global ``instantiate`` does not
        # need to recurse into every name-keyed dict — that broader recursion
        # would also pre-instantiate ``config.net_optimizer`` (with model=None)
        # when ``train.py`` calls ``instantiate(config.model_class)``.
        if log_modules:
            log_modules = {
                k: (_fg_instantiate(v) if isinstance(v, abc.Mapping) and "_target_" in v else v)
                for k, v in dict(log_modules).items()
            }
        self.log_modules: Dict[str, LogModule] = dict(log_modules) if log_modules else {}
        self.offload_module_in_decoding: bool = False
        self._current_dir: Optional[Path] = None
        self.loss_dict_record = _LossDictRecord()
        self.val_loss_dict_record = _LossDictRecord()

    # ----- Lifecycle ---------------------------------------------------------

    def on_app_begin(self) -> None:
        config = getattr(self, "config", None)
        if config is not None:
            if self.media_path is None:
                self.media_path = Path(f"{config.log_config.save_path}/media")
            if self.logging_iter is None:
                self.logging_iter = int(config.trainer.logging_iter)
            # FSDP owns DTensor shards that cannot be relocated with ``Module.to``. DDP keeps
            # ordinary parameters, which are restored before the next wrapped forward.
            self.offload_module_in_decoding = config.trainer.offload_module_in_decoding and not config.trainer.fsdp
        if self.sample_logging_iter is None:
            self.sample_logging_iter = self.logging_iter
        # The wandb run and the log modules live on rank 0 only; leaving the block replaces the barrier and
        # raises a setup failure everywhere.
        with raise_across_ranks("wandb setup"):
            if is_rank0():
                if config is not None:
                    init_wandb(config)
                for module in self.log_modules.values():
                    module.setup(self)

    def on_app_end(self, model: "FastGenModel", iteration: int = 0) -> None:
        if is_rank0():
            for name, module in self.log_modules.items():
                try:
                    module.teardown()
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(f"wandb: log_module {name!r} teardown raised — {exc}")
        if self.delete_previous and self._current_dir is not None and is_rank0():
            shutil.rmtree(self._current_dir, ignore_errors=True)
        self._current_dir = None
        if wandb.run:
            wandb.finish()

    @rank0_only
    def on_optimizer_step_begin(self, model: "FastGenModel", iteration: int = 0) -> None:
        if wandb.run and iteration % self.logging_iter == 0:
            for name, scheduler in model.scheduler_dict.items():
                wandb.log(
                    {f"optimizer/lr_{name}": scheduler.get_last_lr()[0]},
                    step=iteration,
                )

    # ----- Hooks -------------------------------------------------------------

    def on_training_step_end(
        self,
        model: "FastGenModel",
        data_batch: Dict[str, torch.Tensor],
        output_batch: Dict[str, Any],
        loss_dict: Dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        self.loss_dict_record.add(loss_dict)
        # Scalars fire on ``logging_iter``; media fires on the (typically
        # slower) ``sample_logging_iter`` so decode + W&B uploads can be
        # throttled without losing scalar telemetry. Iter 1 fires both
        # for an early sanity-check log.
        log_scalars = iteration == 1 or iteration % self.logging_iter == 0
        log_media = iteration == 1 or iteration % self.sample_logging_iter == 0
        if not (log_scalars or log_media):
            return
        time_start = time.perf_counter()
        if log_media:
            self._step(
                model,
                data_batch,
                output_batch,
                iteration,
                "train",
                "",
                0,
                self.loss_dict_record,
            )
        if log_scalars:
            self._log_stats(self.loss_dict_record, iteration, "train")
        logger.info(f"WandB logging complete after {time.perf_counter() - time_start:.2f} seconds")

    def on_validation_step_end(
        self,
        model: "FastGenModel",
        data_batch: Dict[str, torch.Tensor],
        output_batch: Dict[str, Any],
        loss_dict: Dict[str, torch.Tensor],
        step: int = 0,
        iteration: int = 0,
        idx: int = 0,
    ) -> None:
        self.val_loss_dict_record.add(loss_dict)
        if not self.log_validation_media:
            return
        # Symmetric with ``on_training_step_end``: ``validation_logging_step``
        # gates writes + log modules + panel logging together.
        if step % self.validation_logging_step != 0:
            return
        self._step(
            model,
            data_batch,
            output_batch,
            iteration,
            f"val{idx}",
            f"_{step}",
            step,
            self.val_loss_dict_record,
        )

    def on_validation_end(
        self,
        model: "FastGenModel",
        iteration: int = 0,
        idx: int = 0,
    ) -> None:
        self._log_stats(self.val_loss_dict_record, iteration, f"val{idx}")
        # Tear down per-validation resources (e.g. a log module's HTTP
        # client) so they don't survive into the next DataLoader worker
        # fork. Rank 0 only — log modules only ever existed there.
        if is_rank0():
            for name, module in self.log_modules.items():
                try:
                    module.teardown()
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(f"wandb: log_module {name!r} teardown raised — {exc}")

    # ----- Per-step ---------------------------------------------------------

    def _step(
        self,
        model: "FastGenModel",
        data_batch: Dict[str, Any],
        output_batch: Dict[str, Any],
        iteration: int,
        group: str,
        suffix: str,
        step: int,
        record: _LossDictRecord,
    ) -> None:
        """Per-step write + log:

        1. every rank decodes and writes its slice, then ranks at or above
           ``max_ranks`` discard their local ``MediaEntry`` records;
        2. every rank ``all_gather_object``s its entries → rank 0 has the
           full flattened list of every rank's ``(kind, rank, batch_idx,
           path, prompt)`` records;
        3. every rank runs each log module's ``collect`` and all-gathers
           the result, iterating modules sorted by name so the collective
           sequence is identical on every rank regardless of how each
           rank's ``log_modules`` dict was built;
        4. rank 0 invokes log modules with the flat entries and ships
           media panels; a failure there is raised on every rank.
        """
        save_dir, local_entries = self._write_media(
            model,
            data_batch,
            output_batch,
            iteration,
            group,
            suffix,
        )

        # Gathering entries also ensures every rank has finished writing its media before rank 0 reads it.
        # Ranks excluded by ``max_ranks`` contributed ``local_entries=[]`` already.
        gathered: List[List[MediaEntry]] = _all_gather_object(local_entries)
        entries: List[MediaEntry] = [e for chunk in gathered for e in chunk]

        # Each log module may need per-rank inputs from the local ``data_batch``
        # on rank 0. ``collect`` runs on EVERY rank and its result is
        # all-gathered (collective) before the rank-0 gate so every rank joins;
        # ``process`` then receives the per-rank list (indexed by rank) as
        # ``gathered``. Iterating in sorted-by-name order gives every rank an
        # identical collective sequence derived solely from the key set, so
        # correctness does not rest on every rank's ``log_modules`` dict
        # sharing the same insertion order.
        module_inputs: Dict[str, List[Any]] = {
            name: _all_gather_object(module.collect(data_batch)) for name, module in sorted(self.log_modules.items())
        }

        # Rank-0-only processing; leaving the block raises its failure on every rank instead of hanging the peers.
        with raise_across_ranks("wandb log modules"):
            if is_rank0():
                for name, module in self.log_modules.items():
                    losses = module.process(
                        save_dir,
                        entries,
                        iteration,
                        group,
                        step,
                        data_batch,
                        output_batch,
                        gathered=module_inputs[name],
                    )
                    if losses is not None:
                        record.add(losses)
                self._log_panels(entries, iteration, group, suffix)

    def _write_media(
        self,
        model: "FastGenModel",
        data_batch: Dict[str, Any],
        output_batch: Dict[str, Any],
        iteration: int,
        group: str,
        suffix: str,
    ) -> Tuple[Path, List[MediaEntry]]:
        """Allocate the per-step dir and fan out media writes across ranks.

        Returns ``(save_dir, local_entries)`` where ``local_entries`` is
        this rank's list of written ``MediaEntry`` records (empty for
        ranks ``r >= max_ranks``). Every rank still decodes and writes its
        unique files and participates in every collective (sampler resolve,
        FSDP forward). If the VAE or teacher is sharded,
        callers must supply the same zero/nonzero logged-slice state and the
        same ordered set of fields that trigger their forwards on every rank;
        only the values and nonempty local batch sizes may differ.
        """
        # Allocate a fresh per-step dir; optional rank-0 cleanup of prior.
        save_dir = self.media_path / f"iter{iteration}_{group}{suffix}"
        prior = self._current_dir
        if self.delete_previous and is_rank0() and prior is not None and prior != save_dir:
            shutil.rmtree(prior, ignore_errors=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        self._current_dir = save_dir

        # Resolve deferred sampler on every rank (rank-symmetric).
        if callable(output_batch.get("gen_rand")):
            output_batch["gen_rand"] = output_batch["gen_rand"]()

        gen_rand = output_batch.get("gen_rand")
        # Accept multimodal dicts (e.g. ``{"video": ..., "audio": ...}``) in
        # addition to plain tensors — ``get_batch_size`` reads dim 0 off the
        # first tensor leaf assuming leaf-uniform batch size.
        if is_savable(gen_rand):
            local_bs = get_batch_size(gen_rand)
        else:
            local_bs = 0
        n_per_rank = min(self.log_batchsize, local_bs)
        rank_is_logged = self.max_ranks is None or get_rank() < self.max_ranks

        # Optional teacher generation runs before vae_decode_context so
        # teacher params are still on GPU when offload_module_in_decoding
        # is on. Every rank participates — partial-rank gating would
        # deadlock FSDP's per-layer all_gather.
        if self.config.trainer.visualize_teacher and "input_rand" in output_batch and data_batch is not None:
            # Match the student's full-local-batch generation. ``log_batchsize`` is
            # applied to generated media below, avoiding condition-tree slicing heuristics.
            sample_kwargs = {"condition": data_batch.get("condition")}
            if data_batch.get("neg_condition") is not None:
                sample_kwargs["neg_condition"] = data_batch["neg_condition"]
            # The later teacher-to-decoder reshard cannot prevent this peak: release the
            # retained student/EMA root before the teacher starts its all-gather.
            model.reshard_fsdp_modules()
            output_batch["gen_teacher"] = model.sample(
                model.teacher,
                output_batch["input_rand"],
                **sample_kwargs,
            )

        # This is the important generator-to-decoder handoff. Keep the VAE
        # materialized across the adjacent media decodes below.
        model.reshard_fsdp_modules()
        with vae_decode_context(
            model,
            offload_modules=self.offload_module_in_decoding,
        ):
            local_entries = self._do_save(save_dir, model, data_batch, output_batch, n_per_rank)
        if not rank_is_logged:
            local_entries = []
        # Reshard only after the context has restored any offloaded modules. A VAE created
        # temporarily above is deliberately not FSDP-wrapped, so removing it first is safe.
        model.reshard_fsdp_modules()

        gc.collect()
        torch.cuda.empty_cache()
        return save_dir, local_entries

    def _do_save(
        self,
        save_dir: Path,
        model: "FastGenModel",
        data_batch: Optional[Dict[str, Any]],
        output_batch: Dict[str, Any],
        n_per_rank: int,
    ) -> List[MediaEntry]:
        """Per-rank fan-out for every media kind. Rank ``r`` writes up to
        ``n_per_rank`` items per kind and returns a flat ``MediaEntry``
        list covering every item written — kind, rank, batch_idx, path,
        and the matching prompt from ``data_batch``.

        Must be called inside ``vae_decode_context(model)`` — the VAE (when
        the model has one) is read from ``model.net_preprocessors.preprocessors`` and the active
        ``inference_mode`` autocast wraps the per-kind ``save_media``
        decode calls.
        """
        entries: List[MediaEntry] = []
        if n_per_rank <= 0:
            return entries

        rank = get_rank()
        precision_amp_enc = getattr(model, "precision_amp_enc", None)
        prompts = [extract_prompt(data_batch, i) for i in range(n_per_rank)]

        # Encoding kwargs forwarded to ``save_media`` for every kind:
        # ``vid_format`` (callback-level), plus ``fps`` / ``audio_sample_rate``
        # so audio+video models mux mp4s at the right sample rate / fps. The
        # image path of ``save_media`` ignores these extras via ``**kwargs``.
        model_media_kwargs = {"vid_format": self.vid_format}
        if hasattr(model.net, "fps"):
            model_media_kwargs["fps"] = model.net.fps
        vae = model.net_preprocessors.preprocessors.get("vae")
        if vae is not None and hasattr(vae, "vocoder_sample_rate"):
            model_media_kwargs["audio_sample_rate"] = vae.vocoder_sample_rate

        gen_rand = output_batch.get("gen_rand")
        gen_teacher = output_batch.get("gen_teacher")
        gen_rand_train = output_batch.get("gen_rand_train")
        real = data_batch.get("real") if data_batch is not None else None

        if is_savable(gen_rand):
            entries.extend(
                self._save_batch(
                    save_dir,
                    "student",
                    slice_dim(gen_rand, n_per_rank),
                    rank,
                    prompts,
                    vae,
                    precision_amp_enc,
                    model_media_kwargs,
                )
            )

        if is_savable(real):
            entries.extend(
                self._save_batch(
                    save_dir,
                    "real",
                    slice_dim(real, n_per_rank),
                    rank,
                    prompts,
                    vae,
                    precision_amp_enc,
                    model_media_kwargs,
                )
            )

        # Prefer the original pixel source for display; decode an offline latent when
        # no raw mirror is available.
        source = None
        source_vae: Optional[Callable] = None
        if data_batch is not None:
            source = data_batch.get("condition_image_raw")
            if source is None:
                source = data_batch.get("condition_image")
                if source is not None:
                    source_vae = vae

        if is_savable(source):
            entries.extend(
                self._save_batch(
                    save_dir,
                    "source",
                    slice_dim(source, n_per_rank),
                    rank,
                    prompts,
                    source_vae,
                    precision_amp_enc,
                    model_media_kwargs,
                )
            )

        # Reward decode is video-frame-only — skip on multimodal A/V outputs.
        if (
            hasattr(model, "reward")
            and getattr(model.reward, "decode_input", False)
            and isinstance(gen_rand, torch.Tensor)
        ):
            rf = model.reward.decode(gen_rand[:n_per_rank]).mul(2.0).sub(1.0)
            if rf.ndim == 5:
                rf = rf.permute(0, 2, 1, 3, 4)
            entries.extend(
                self._save_batch(
                    save_dir,
                    "reward",
                    rf,
                    rank,
                    prompts,
                    None,
                    precision_amp_enc,
                    model_media_kwargs,
                )
            )

        if is_savable(gen_teacher):
            entries.extend(
                self._save_batch(
                    save_dir,
                    "teacher",
                    slice_dim(gen_teacher, n_per_rank),
                    rank,
                    prompts,
                    vae,
                    precision_amp_enc,
                    model_media_kwargs,
                )
            )

        if logger.LOG_LEVEL == "DEBUG" and is_savable(gen_rand_train):
            entries.extend(
                self._save_batch(
                    save_dir,
                    "student_train",
                    slice_dim(gen_rand_train, n_per_rank),
                    rank,
                    prompts,
                    vae,
                    precision_amp_enc,
                    model_media_kwargs,
                )
            )

        return entries

    def _save_batch(
        self,
        save_dir: Path,
        kind: str,
        tensor: TensorOrDict,
        rank: int,
        prompts: List[str],
        vae: Optional[Callable],
        precision_amp_enc: Any,
        model_media_kwargs: Dict[str, Any],
    ) -> List[MediaEntry]:
        """Save the per-rank batch under
        ``<save_dir>/<key_prefix>/frame_<rank:03d>_<batch_idx:03d>.{png,mp4,gif,wav}`` and
        return one ``MediaEntry`` per item (``kind``, ``rank``,
        ``batch_idx``, resolved ``path``, matching ``prompt``).

        ``tensor`` follows ``save_media``'s contract — ``[-1, 1]`` range
        (pre-decode latents when ``vae`` is provided), and one of:

        * ``[B, C, H, W]`` image tensor → ``.png``;
        * ``[B, C, T, H, W]`` video tensor (including ``T == 1``) →
          ``.gif`` / ``.mp4`` per ``vid_format``;
        * ``{"image": Tensor}`` → ``.png`` (save_media unwraps);
        * ``{"audio": Tensor}``, audio waveforms, or a tensor with an audio-only
          VAE → ``.wav``;
        * ``{"video": Tensor}`` → ``.gif`` / ``.mp4`` per ``vid_format``
          (save_media unwraps);
        * ``{"video": Tensor, "audio": Tensor}`` → ``.mp4`` (muxed via
          ``save_audio_video`` whatever ``vid_format`` says, because GIF
          cannot carry an audio track).

        ``model_media_kwargs`` carries the encoder knobs forwarded to
        ``save_media`` (``vid_format`` for video kinds, plus ``fps`` /
        ``audio_sample_rate`` for audio+video models).
        """
        vid_ext = f".{model_media_kwargs.get('vid_format', 'mp4')}"
        if isinstance(tensor, dict) and "video" in tensor:
            # A/V dict → muxed mp4 (``save_media`` falls back from GIF, which
            # has no audio track). Video-only dict → ``save_media``
            # unwraps to the bare 5D tensor and ``save_video`` picks ext
            # per ``vid_format``.
            ext = ".mp4" if "audio" in tensor else vid_ext
        elif isinstance(tensor, dict) and "image" in tensor:
            # Image-only dict → ``save_media`` unwraps to a 4D tensor and
            # ``save_image`` writes png.
            ext = ".png"
        elif isinstance(tensor, dict) and "audio" in tensor:
            ext = ".wav"
        elif isinstance(tensor, torch.Tensor):
            if tensor.ndim not in (2, 3, 4, 5):
                raise ValueError(f"_save_batch: expected 2D–5D tensor, got {tensor.ndim}D")
            # Audio latents are 4D, just like images; the decoder identifies their modality.
            if tensor.ndim in (2, 3) or tuple(getattr(vae, "modalities", ())) == ("audio",):
                ext = ".wav"
            else:
                ext = vid_ext if tensor.ndim == 5 else ".png"
        else:
            raise TypeError(f"_save_batch: expected Tensor or media dict, got {type(tensor).__name__}")
        bs = get_batch_size(tensor)
        if bs == 0:
            return []
        save_path = save_dir / _KEY_PREFIXES[kind] / f"frame_{rank:03d}{ext}"
        save_media(
            tensor,
            str(save_path),
            vae=vae,
            precision_amp=precision_amp_enc,
            always_index=True,
            **model_media_kwargs,
        )
        # ``save_media`` with ``always_index=True`` writes
        # ``<stem>_<i:03d><suffix>`` for every batch element; mirror that
        # rule here so we don't need a glob round-trip.
        return [
            MediaEntry(
                kind=kind,
                rank=rank,
                batch_idx=i,
                path=save_path.parent / (save_path.stem + f"_{i:03d}" + save_path.suffix),
                prompt=prompts[i] if i < len(prompts) else "",
            )
            for i in range(bs)
        ]

    # ----- W&B logging -------------------------------------------------------

    def _log_stats(
        self,
        record: _LossDictRecord,
        iteration: int,
        group: str,
    ) -> None:
        logger.info(f"logging {group} stats at iteration {iteration}" + "-" * 20)
        # ``get_stat`` is a collective (cross-rank ``all_gather_object``);
        # every rank participates. The averaged result is logged on rank 0.
        avg = record.get_stat()
        if not is_rank0() or not wandb.run:
            return
        wandb.log({f"{group}/{name}": val for name, val in avg.items()}, step=iteration)

    def _log_panels(
        self,
        entries: List[MediaEntry],
        iteration: int,
        group: str,
        suffix: str,
    ) -> None:
        """Group ``entries`` by kind and ship each kind's tiles to W&B as
        a single image, video, or audio panel (rank 0 only).
        Each tile's caption is the entry's own ``prompt`` field — no
        filename parsing or cross-rank lookup needed.
        """
        if not wandb.run:
            return
        sample_map: Dict[str, List[Any]] = {}
        for e in entries:
            prefix = _KEY_PREFIXES.get(e.kind)
            if prefix is None:
                continue
            cap = e.prompt or None
            media_suffix = e.path.suffix.lower()
            if media_suffix in _IMAGE_SUFFIXES:
                tile: Any = wandb.Image(str(e.path), caption=cap)
            elif media_suffix in _VIDEO_SUFFIXES:
                tile: Any = wandb.Video(
                    str(e.path),
                    caption=cap,
                    format=media_suffix.lstrip("."),
                )
            elif media_suffix == ".wav":
                tile = wandb.Audio(str(e.path), caption=cap)
            else:
                raise ValueError(f"Unsupported media suffix {media_suffix!r}: {e.path}")
            sample_map.setdefault(f"{group}_media/{prefix}{suffix}", []).append(tile)
        if sample_map:
            wandb.log(sample_map, step=iteration)
