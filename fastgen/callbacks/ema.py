# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, TYPE_CHECKING, Optional

import torch
import wandb
from torch.distributed.tensor import DTensor

from fastgen.callbacks.callback import Callback
from fastgen.utils.basic_utils import get_batch_size_total
from fastgen.utils.distributed import is_rank0
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
    ):
        self.type = type
        self.beta = beta
        self.gamma = gamma
        self.ema_halflife_kimg = ema_halflife_kimg
        self.ema_rampup_ratio = ema_rampup_ratio
        self.start_iter = start_iter
        self.ema_name = ema_name
        self.batch_size = batch_size
        self._enabled = True
        self._warned_missing_tensors: set[tuple[str, str]] = set()

    def on_app_begin(self) -> None:
        if hasattr(self, "config"):
            # override using config
            self.batch_size = get_batch_size_total(self.config)

    def on_model_init_end(self, model: FastGenModel | torch.nn.parallel.DistributedDataParallel) -> None:
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

        dtypes = sorted({str(p.dtype) for p in ema.parameters() if p.dtype in (torch.float16, torch.bfloat16)})
        if dtypes:
            logger.warning(
                f"EMA {self.ema_name} parameters are stored in {', '.join(dtypes)}, where updates with beta close "
                'to 1 round away. Set `model.precision_opt="float32"` to keep the EMA moving.'
            )

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

    @staticmethod
    def _collect_named_tensors(named_tensors, owner: str = "module", kind: str = "tensor") -> dict[str, torch.Tensor]:
        """Index parameters or buffers by their wrapper-independent name."""
        tensors = {}
        for raw_name, tensor in named_tensors:
            name = raw_name.replace("_checkpoint_wrapped_module.", "")
            if name in tensors:
                raise RuntimeError(f"Duplicate canonical {kind} name {name!r} in {owner}.")
            tensors[name] = tensor
        return tensors

    @staticmethod
    def _validate_tensor_pair(
        name: str,
        ema_tensor: torch.Tensor,
        net_tensor: torch.Tensor,
        kind: str = "tensor",
        shape_kind: str | None = None,
    ) -> None:
        """Validate the shape shared by an EMA tensor and its student source.

        Device and dtype may differ intentionally: the update casts the student tensor to
        the EMA tensor, and CPU offload may place otherwise compatible tensors differently.
        """
        if tuple(ema_tensor.shape) != tuple(net_tensor.shape):
            shape = "shape" if shape_kind is None else f"{shape_kind} shape"
            raise RuntimeError(
                f"EMA {kind} {name!r} has {shape} {tuple(ema_tensor.shape)}, "
                f"but the student {kind} has {tuple(net_tensor.shape)}."
            )

    @classmethod
    def _validate_dtensor_pair(
        cls,
        name: str,
        ema_tensor: DTensor,
        net_tensor: torch.Tensor,
        kind: str = "tensor",
    ) -> None:
        """Validate matching global and local layouts for a pair of sharded tensors."""
        if not isinstance(net_tensor, DTensor):
            raise RuntimeError(f"EMA {kind} {name!r} is sharded but the corresponding student {kind} is not.")
        if ema_tensor.device_mesh != net_tensor.device_mesh:
            raise RuntimeError(f"EMA {kind} {name!r} has a different device mesh from the student {kind}.")
        if ema_tensor.placements != net_tensor.placements:
            raise RuntimeError(
                f"EMA {kind} {name!r} has placements {ema_tensor.placements}, "
                f"but the student {kind} has {net_tensor.placements}."
            )
        cls._validate_tensor_pair(
            name,
            ema_tensor,
            net_tensor,
            kind=kind,
            shape_kind="global",
        )
        cls._validate_tensor_pair(
            name,
            ema_tensor.to_local(),
            net_tensor.to_local(),
            kind=kind,
            shape_kind="local",
        )

    def _warn_missing_tensor(self, name: str, kind: str = "tensor") -> None:
        """Warn once per student tensor without an EMA counterpart, so its update is skipped."""
        if (kind, name) in self._warned_missing_tensors:
            return
        self._warned_missing_tensors.add((kind, name))
        logger.warning(f"EMA {kind} {name} not found, skipping update.")

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

        ema = getattr(model, self.ema_name)

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
            # Trainer establishes sharded state before the step-end callback chain. Any
            # earlier callback that materializes these roots owns that intra-chain handoff.
            net_params = self._collect_named_tensors(model.net.named_parameters(), owner="student", kind="parameter")
            ema_params = self._collect_named_tensors(
                ema.named_parameters(),
                owner=self.ema_name,
                kind="parameter",
            )
            for name, net_param in net_params.items():
                ema_param = ema_params.get(name)
                if ema_param is None:
                    self._warn_missing_tensor(name, kind="parameter")
                    continue

                if isinstance(ema_param, DTensor):
                    self._validate_dtensor_pair(name, ema_param, net_param, kind="parameter")
                    ema_param = ema_param.to_local()
                    net_param = net_param.to_local()
                else:
                    if hasattr(net_param, "full_tensor"):
                        # Compatibility path for a sharded student with a replicated EMA. A live
                        # sharded EMA takes the branch above and never materializes a full parameter.
                        if isinstance(net_param, DTensor) and net_param.to_local().device.type == "cpu":
                            net_param = net_param.to("cuda")
                        net_param = net_param.full_tensor()
                    self._validate_tensor_pair(name, ema_param, net_param, kind="parameter")

                net_param = net_param.to(device=ema_param.device, dtype=ema_param.dtype)
                if total_iteration == self.start_iter:
                    ema_param.copy_(net_param)
                else:
                    ema_param.lerp_(net_param, 1.0 - beta)

            # FSDP2 doesn't shard buffers, so we can just copy them
            net_buffers = self._collect_named_tensors(model.net.named_buffers(), owner="student", kind="buffer")
            ema_buffers = self._collect_named_tensors(ema.named_buffers(), owner=self.ema_name, kind="buffer")
            for name, p_net in net_buffers.items():
                if name not in ema_buffers:
                    self._warn_missing_tensor(name, kind="buffer")
                    continue
                ema_buffer = ema_buffers[name]
                self._validate_tensor_pair(name, ema_buffer, p_net, kind="buffer")
                ema_buffer.copy_(p_net.to(device=ema_buffer.device, dtype=ema_buffer.dtype))

            if hasattr(self, "config"):
                # only wandb log when config exists
                if iteration % self.config.trainer.logging_iter == 0 and is_rank0():
                    if wandb.run:
                        wandb.log({f"ema/{self.ema_name}_beta": beta}, step=iteration)
