# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import torch
import wandb

from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import is_rank0
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


class TrainProfilerCallback(Callback):
    """Callback for profiling training speed and detailed timing breakdowns.

    Logs every ``every_n`` iterations:
    - avg_iteration_time: wall-clock seconds per iteration since the previous log
    - data_loading_time: time spent loading and preparing the iteration's batches
    - avg_forward_pass_time: forward pass time, averaged over accumulation rounds
    - avg_backward_pass_time: backward pass time, averaged over accumulation rounds
    - optimizer_step_time: time from this callback's optimizer-step hook to the end of the optimizer step

    The breakdown excludes the step-end callbacks (e.g. W&B media logging). Logged iterations synchronize CUDA at
    each phase boundary, so every phase includes the kernels it launched.
    """

    def __init__(self, every_n: int | None = None, detailed: bool = True):
        """Initialize the profiler callback.

        Args:
            every_n: Log every N iterations; defaults to ``trainer.logging_iter``
            detailed: If True, log detailed timing breakdown. If False, only log the iteration time.
        """
        # For iter_time tracking
        self.last_log_time = None

        # For detailed profiling
        self.detailed = detailed
        self.train_step_begin_time = None
        self.accum_begin_times = None
        self.backward_begin_times = None
        self.accum_end_times = None
        self.optimizer_step_begin = None
        self.every_n = every_n

    def _now(self, iteration: int) -> float:
        # Kernels run asynchronously; wait for them on logged iterations so each phase ends with its GPU work.
        if iteration % self.every_n == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def on_train_begin(self, model: FastGenModel, iteration: int = 0) -> None:
        if self.every_n is None:
            self.every_n = self.config.trainer.logging_iter
        logger.info(f"every_n to profile trainer: {self.every_n}")

    def on_training_step_begin(
        self,
        model: FastGenModel,
        iteration: int = 0,
    ):
        if self.detailed:
            self.train_step_begin_time = self._now(iteration)
            self.accum_begin_times = []
            self.backward_begin_times = []
            self.accum_end_times = []

    def on_training_accum_step_begin(
        self, model: FastGenModel, data_batch: dict[str, torch.Tensor], iteration: int = 0, accum_iter: int = 0
    ):
        if self.detailed:
            self.accum_begin_times.append(self._now(iteration))

    def on_backward_begin(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
        accum_iter: int = 0,
    ):
        if self.detailed:
            self.backward_begin_times.append(self._now(iteration))

    def on_optimizer_step_begin(self, model: FastGenModel, iteration: int = 0):
        if self.detailed:
            self.optimizer_step_begin = self._now(iteration)

    def on_training_accum_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        iteration: int = 0,
        accum_iter: int = 0,
    ):
        # The last round ends with the optimizer step, before the step-end callbacks run.
        if self.detailed:
            self.accum_end_times.append(self._now(iteration))

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        del data_batch, output_batch, loss_dict

        if hasattr(self, "config"):
            # only wandb log when config exists
            if iteration % self.every_n == 0 and is_rank0():
                metrics = {}

                # Calculate iter_time (wall clock time per iteration)
                cur_time = time.time()
                if self.last_log_time is not None:
                    iter_time = (cur_time - self.last_log_time) / self.every_n
                    logger.info(f"{iteration} : avg iteration time       {iter_time:.2f} seconds")
                    metrics["profiler/avg_iteration_time"] = iter_time
                self.last_log_time = cur_time

                # Calculate detailed timing breakdown
                if self.detailed and self.accum_end_times:
                    # Each round loads its batch after the previous round ends; the last backward ends at the
                    # optimizer step.
                    load_begin_times = [self.train_step_begin_time] + self.accum_end_times[:-1]
                    backward_end_times = self.accum_end_times[:-1] + [self.optimizer_step_begin]
                    rounds = len(self.accum_begin_times)
                    data_load_time = sum(b - a for a, b in zip(load_begin_times, self.accum_begin_times))
                    forward_time = (
                        sum(b - a for a, b in zip(self.accum_begin_times, self.backward_begin_times)) / rounds
                    )
                    backward_time = sum(b - a for a, b in zip(self.backward_begin_times, backward_end_times)) / rounds
                    optim_step_time = self.accum_end_times[-1] - self.optimizer_step_begin

                    logger.info(f"{iteration} : data loading time        {data_load_time:.2f}")
                    logger.info(f"{iteration} : avg forward pass time    {forward_time:.2f}")
                    logger.info(f"{iteration} : avg backward pass time   {backward_time:.2f}")
                    logger.info(f"{iteration} : optimizer step time      {optim_step_time:.2f}")

                    metrics.update(
                        {
                            "profiler/data_loading_time": data_load_time,
                            "profiler/avg_forward_pass_time": forward_time,
                            "profiler/avg_backward_pass_time": backward_time,
                            "profiler/optimizer_step_time": optim_step_time,
                        }
                    )

                if wandb.run and metrics:
                    wandb.log(metrics, step=iteration)
