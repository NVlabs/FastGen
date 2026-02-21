# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, TYPE_CHECKING, Optional

import torch
import wandb

from fastgen.callbacks.callback import Callback
from fastgen.utils.basic_utils import get_batch_size_total
from fastgen.utils.distributed import synchronize, is_rank0
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


class EMACallback(Callback):
    def __init__(
        self,
        type: str = "constant",
        # params for type=constant
        beta: float = 0.9999,
        # params for type=power
        gamma: float = 16.97,
        # params for type=halflife
        ema_halflife_kimg: float = 500,
        ema_rampup_ratio: Optional[float] = 0.05,
        start_iter: int = 0,
        ema_name: str = "ema",
        batch_size: int = 1,  # overwritten by self.config if it exists
        fsdp: bool = False,  # overwritten by self.config if it exists
    ):
        self.type = type
        self.beta = beta
        self.gamma = gamma
        self.ema_halflife_kimg = ema_halflife_kimg
        self.ema_rampup_ratio = ema_rampup_ratio
        self.start_iter = start_iter
        self.ema_name = ema_name
        self.batch_size = batch_size
        self._is_fsdp = fsdp
        self._enabled = True

    def on_app_begin(self) -> None:
        if hasattr(self, "config"):
            # override using config
            self._is_fsdp = self.config.trainer.fsdp
            self.batch_size = get_batch_size_total(self.config)

    def on_model_init_end(
        self, model: FastGenModel | torch.nn.parallel.DistributedDataParallel, iteration: int = 0
    ) -> None:
        # Unwrap DDP if needed to access the original model's attributes
        if hasattr(model, "module"):
            model = model.module

        # check ema initialization
        ema = getattr(model, self.ema_name, None)
        if ema is None:
            self._enabled = False
            logger.info(f"EMA {self.ema_name} is not enabled, skipping callback.")
            return

        assert ema.training is False, f"EMA {self.ema_name} should be in eval mode"
        for name, p_net in ema.named_parameters():
            assert not p_net.requires_grad, f"EMA parameter {name} should not require gradients"

    def _total_iteration(self, model: FastGenModel, iteration: int) -> int:
        if hasattr(model, "resume_iter"):
            assert isinstance(model.resume_iter, int)
            iteration = iteration + model.resume_iter
        return iteration

    def _power_function_beta(self, iteration):
        beta = (1 - 1 / iteration) ** (self.gamma + 1)
        return beta

    def _get_cur_nimg(self, iteration):
        cur_nimg = iteration * self.batch_size
        return self.batch_size, cur_nimg

    def _halflife_beta(self, iteration):
        ema_halflife_nimg = self.ema_halflife_kimg * 1000
        batch_size, cur_nimg = self._get_cur_nimg(iteration)
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * self.ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        return ema_beta

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        del data_batch, output_batch, loss_dict

        # Check if EMA is enabled
        if not self._enabled:
            return

        # Get total iteration and skip if before start_iter
        total_iteration = self._total_iteration(model, iteration)
        if total_iteration < self.start_iter:
            return
        elif total_iteration == self.start_iter:
            logger.info(f"Starting to update {self.ema_name} at iteration {total_iteration}.")

        if self.type == "constant":
            beta = self.beta
        elif self.type == "power":
            beta = self._power_function_beta(total_iteration)
        elif self.type == "halflife":
            beta = self._halflife_beta(total_iteration)
        else:
            raise ValueError(f"Invalid {self.ema_name} type: {self.type}")

        with torch.no_grad():
            ema = getattr(model, self.ema_name)
            ema_state_dict = ema.state_dict()

            for name, p_net in model.net.named_parameters():
                if self._is_fsdp and hasattr(p_net, "full_tensor"):
                    # Gather the full tensor from all ranks if using FSDP with DTensor
                    # When CPU offloading is enabled, we need to move to CUDA first because
                    # full_tensor() performs an all_gather which requires a CUDA backend
                    if p_net.device.type == "cpu":
                        # Move local shard to CUDA, gather, then the result stays on CUDA
                        # which is fine since we'll copy to EMA (which handles device placement)
                        full_tensor = p_net.to("cuda").full_tensor()
                    else:
                        full_tensor = p_net.full_tensor()
                else:
                    full_tensor = p_net
                # Strip checkpoint wrapper prefix if present (EMA doesn't have checkpointing)
                ema_name = name.replace("_checkpoint_wrapped_module.", "")
                # Cast to EMA dtype and device for lerp_ compatibility
                if ema_name in ema_state_dict:
                    ema_param = ema_state_dict[ema_name]
                    if total_iteration == self.start_iter:
                        # re-initialize EMA parameter
                        ema_param.copy_(full_tensor.to(device=ema_param.device, dtype=ema_param.dtype))
                    else:
                        # interpolate EMA parameter
                        ema_param.lerp_(full_tensor.to(device=ema_param.device, dtype=ema_param.dtype), 1.0 - beta)
                elif iteration == 1:
                    # only warn on first iteration if parameter is not found
                    logger.warning(f"EMA parameter {ema_name} not found in EMA state dict, skipping update.")

            # FSDP2 doesn't shard buffers, so we can just copy them
            for name, p_net in model.net.named_buffers():
                if name in ema_state_dict:
                    ema_param = ema_state_dict[name]
                    ema_param.copy_(p_net.to(device=ema_param.device, dtype=ema_param.dtype))
                elif iteration == 1:
                    # only warn on first iteration if buffer is not found
                    logger.warning(f"EMA buffer {name} not found in EMA state dict, skipping update.")

            if hasattr(self, "config"):
                # only wandb log when config exists
                if iteration % self.config.trainer.logging_iter == 0 and is_rank0():
                    if wandb.run:
                        wandb.log({f"ema/{self.ema_name}_beta": beta}, step=iteration)
        synchronize()
