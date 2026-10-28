# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional
import wandb

import torch
from torch.distributed.tensor import DTensor

from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import is_rank0
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


@contextmanager
def cast_gradients_dtype(model, dtype=torch.float32, enabled=True):
    if enabled:
        try:
            # Cast gradients to the desired dtype
            for param in model.parameters():
                if param.grad is not None and param.grad.dtype != dtype:
                    param.grad.data = param.grad.data.to(dtype)
            yield
        finally:
            # Restore original gradient dtypes
            for param in model.parameters():
                if param.grad is not None and param.grad.dtype != param.dtype:
                    param.grad.data = param.grad.data.to(param.dtype)
    else:
        yield


class GradClipCallback(Callback):
    """Clip a top-level model; configure at most one clipping callback per optimizer."""

    def __init__(
        self,
        grad_norm: float | None = 1.0,
        model_key: str = "net",
        posinf: float | None = None,
        neginf: float | None = None,
        precision_grad_clip: Optional[torch.dtype] = None,
    ) -> None:
        self.grad_norm = grad_norm
        self.model_key = model_key
        self.posinf = posinf
        self.neginf = neginf
        self.precision_grad_clip = precision_grad_clip
        self._warned_missing_optimizer = False

    def nan_to_num(self, module: torch.nn.Module) -> tuple[int, torch.dtype | None]:
        grad_dtype = None
        non_finite_grads_count = 0

        for name, param in module.named_parameters():
            if param.grad is not None:
                grad_dtype = param.grad.dtype

                # Extract local tensor for DTensor (avoids triggering distributed ops on CPU)
                if isinstance(param.grad, DTensor):
                    grad = param.grad._local_tensor
                else:
                    grad = param.grad

                non_finite_grads = grad.numel() - grad.isfinite().sum().item()
                if non_finite_grads:
                    non_finite_grads_count += non_finite_grads
                    logger.debug(
                        f"Gradient of {name} (dtype {grad_dtype}) is not finite: "
                        f"Setting {grad.isnan().sum().item()} NaNs to 0 and {grad.isinf().sum().item()} Infs "
                        f"to {self.posinf} or {self.neginf}."
                    )
                    torch.nan_to_num(grad, nan=0.0, posinf=self.posinf, neginf=self.neginf, out=grad)

        return non_finite_grads_count, grad_dtype

    def _warn_missing_optimizer(self, optimizer_dict: dict[str, torch.optim.Optimizer]) -> None:
        """Warn once that ``model_key`` has no optimizer, so its gradient clipping is skipped."""
        if self._warned_missing_optimizer:
            return
        self._warned_missing_optimizer = True
        logger.warning(
            f"GradClipCallback: no optimizer for model_key {self.model_key!r} "
            f"(optimizers: {list(optimizer_dict)}); skipping its gradient clipping."
        )

    def on_optimizer_step_begin(self, model: FastGenModel, iteration: int = 0) -> None:
        # unscale the optimizer related to the `model_key`
        optimizer_dict = model.optimizer_dict
        if self.model_key not in optimizer_dict:
            # A disabled loss term (e.g. ``gan_loss_weight_gen=0``) builds no model or optimizer to clip.
            self._warn_missing_optimizer(optimizer_dict)
            return

        optimizer = optimizer_dict[self.model_key]
        if model.should_use_grad_scaler(optimizer):
            model.grad_scaler.unscale_(optimizer)

        # select the module to clip (by default, we only perform gradient clips on model.net)
        module = getattr(model, self.model_key)

        # set nan to num for each parameter
        non_finite_grads_count, grad_dtype = self.nan_to_num(module)
        logger.debug(f"Gradient dtype of {self.model_key}: {grad_dtype}")
        if non_finite_grads_count > 0:
            logger.info(
                f"Number of parameters with non-finite gradients (of dtype {grad_dtype}): {non_finite_grads_count}"
            )
        log_dict = {f"optimizer/non_finite_grads_count (model_key {self.model_key})": non_finite_grads_count}

        if self.grad_norm is not None:
            # Cast all gradients to precision_grad_clip for numerical stability during clipping
            cast_grads = (
                self.precision_grad_clip is not None
                and grad_dtype is not None
                and grad_dtype != self.precision_grad_clip
            )

            # log value at first iteration
            if iteration == 1 and cast_grads:
                logger.info(f"Casting gradients from {grad_dtype} to {self.precision_grad_clip} before clipping.")

            with cast_gradients_dtype(module, dtype=self.precision_grad_clip, enabled=cast_grads):
                # Stock clip works for FSDP CPU-offload too because the default PG
                # is mixed-backend (cuda:nccl,cpu:gloo) — DTensor's CPU all_reduce
                # routes through gloo automatically.
                total_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_norm)

            log_dict[f"optimizer/grad_norm (model_key {self.model_key})"] = total_norm.item()

        if hasattr(self, "config"):
            # only wandb log when config exists
            if iteration % self.config.trainer.logging_iter == 0 and is_rank0() and wandb.run:
                wandb.log(log_dict, step=iteration)
