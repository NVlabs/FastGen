# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Callable, TYPE_CHECKING
import math
import wandb

import torch

from fastgen.callbacks.callback import Callback
import fastgen.utils.logging_utils as logger
from fastgen.utils.basic_utils import get_batch_size_total
from fastgen.utils.distributed import is_rank0

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel
    from fastgen.configs.config import BaseConfig


class CTScheduleCallback(Callback):
    config: "BaseConfig"

    def __init__(
        self,
        q: float = 2.0,
        ratio_limit: float = 0.999,
        kimg_per_stage: int = 12500,
        batch_size: int = 1,
    ):
        if not math.isfinite(q) or q <= 1:
            raise ValueError(f"q must be finite and greater than 1, got {q!r}.")
        self.q = q
        self.ratio_limit = ratio_limit
        self.kimg_per_stage = kimg_per_stage
        self.batch_size = batch_size

        self.stage = 0
        self.ratio = 0.0

    def _get_cur_stage(self, model, iteration):
        # Start from the saved iteration of the first-stage model in TCM
        if hasattr(model, "resume_iter"):
            assert isinstance(model.resume_iter, int)
            iteration = iteration + model.resume_iter

        batch_size = self.batch_size
        if hasattr(self, "config"):
            # override the batch_size using self.config
            batch_size = get_batch_size_total(self.config)

        cur_nimg = iteration * batch_size
        stage = cur_nimg // (self.kimg_per_stage * 1000)
        return stage, cur_nimg

    def _update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - self.q ** (-(stage + 1))
        if self.ratio > self.ratio_limit:
            logger.info(f"Clipping ratio from {self.ratio} -> {self.ratio_limit}")
            self.ratio = self.ratio_limit

    def on_train_begin(self, model: FastGenModel, iteration: int = 0) -> None:
        stage, _ = self._get_cur_stage(model, iteration)
        self._update_schedule(stage)
        setattr(model, "ratio", self.ratio)

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        del data_batch, output_batch, loss_dict
        new_stage, cur_nimg = self._get_cur_stage(model, iteration)
        if new_stage > self.stage:
            self._update_schedule(new_stage)
            setattr(model, "ratio", self.ratio)

        if hasattr(self, "config"):
            # only wandb log when config exists
            if iteration % self.config.trainer.logging_iter == 0 and is_rank0():
                if wandb.run:
                    wandb.log({"ct_schedule/kimg": cur_nimg / 1e3, "ct_schedule/ratio": self.ratio}, step=iteration)
