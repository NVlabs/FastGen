# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug callback checking replicated module-state fingerprints across ranks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import world_size
from fastgen.utils.distributed.collectives import validate_object_across_ranks
from fastgen.utils.distributed.module import validate_module_values_across_ranks
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


class RankConsistencyCallback(Callback):
    """Check replicated module-state fingerprints every ``every_n`` iterations.

    ``DistributedDataParallel`` corrects buffer drift silently by broadcasting rank 0's buffers before
    each forward; FastGen keeps that default but has no equivalent for FSDP roots, preprocessors, or
    parameters. This callback reports mismatched fingerprints for every buffer (persistent and
    non-persistent) of each selected root and, with ``check_parameters=True``, the replicated
    parameters too. DTensor parameters are checked only when every placement is ``Replicate``;
    parameters with a sharded or partial placement are skipped.

    Checks use device-native 64-bit XOR fingerprints without cloning tensor contents. This is a
    consistency canary: permutations and some repeated-value changes can collide. Each check
    broadcasts registration metadata and scalar fingerprints per root, plus a small root-plan
    preflight. Computing fingerprints still reads the selected tensors, so parameter checks on a
    large DDP model are costly. This is a debugging aid and is a no-op outside distributed execution.
    """

    def __init__(
        self,
        every_n: int = 100,
        check_parameters: bool = False,
        modules: Optional[Sequence[str]] = None,
    ):
        if every_n <= 0:
            raise ValueError(f"every_n must be positive, got {every_n}.")
        self.every_n = every_n
        self.check_parameters = check_parameters
        # ``None`` checks every registered root; otherwise names in ``FastGenModel.module_dict``.
        self.modules = None if modules is None else tuple(modules)

    def _selected(self, model: FastGenModel) -> tuple[dict[str, torch.nn.Module], list[str]]:
        """Return the selected registered roots and the configured names this rank has not registered."""
        if self.modules is None:
            return dict(model.module_dict), []
        selected = {name: model.module_dict[name] for name in self.modules if name in model.module_dict}
        missing = [name for name in self.modules if name not in model.module_dict]
        return selected, missing

    def _selected_root_plan(
        self,
        model: FastGenModel,
        operation: str,
    ) -> tuple[dict[str, torch.nn.Module], tuple[str, ...]]:
        selected, missing = self._selected(model)
        root_names = tuple(sorted(selected))
        # Agree on cadence at startup, before any rank can skip a check, and on the selected
        # roots before entering their collectives. Callback configuration stays fixed after startup.
        validate_object_across_ranks(
            {
                "roots": root_names,
                "missing": tuple(sorted(missing)),
                "every_n": self.every_n,
                "check_parameters": self.check_parameters,
            },
            operation=f"{operation} root-plan validation",
        )
        # Agreement above makes this rejection uniform, so it needs no separate collective guard.
        if missing:
            raise ValueError(f"RankConsistencyCallback: {missing} are not registered in module_dict.")
        return selected, root_names

    def on_train_begin(self, model: FastGenModel, iteration: int = 0) -> None:
        _selected, names = self._selected_root_plan(model, operation="rank consistency callback initialization")
        what = "parameters and buffers" if self.check_parameters else "buffers"
        logger.info(f"Rank consistency check of {what} every {self.every_n} iterations for {list(names)}.")

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        del data_batch, output_batch, loss_dict
        if iteration % self.every_n != 0 or world_size() <= 1:
            return

        operation = f"rank consistency check at iteration {iteration}"
        selected, root_names = self._selected_root_plan(model, operation=operation)
        for name in root_names:
            validate_module_values_across_ranks(
                selected[name],
                parameters=self.check_parameters,
                persistent_buffers=True,
                nonpersistent_buffers=True,
                operation=f"rank consistency check of {name!r} at iteration {iteration}",
            )
        logger.info(f"Rank consistency check passed at iteration {iteration}.")
