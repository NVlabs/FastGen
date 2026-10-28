# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TYPE_CHECKING

from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import world_size
import fastgen.utils.logging_utils as logger
import torch
import wandb
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


def _get_local_numel(param: torch.Tensor) -> int:
    """Get the local (sharded) number of elements for a parameter.

    For DTensor (FSDP2), returns the local shard size.
    For regular tensors, returns the full size.
    """
    if isinstance(param, DTensor):
        return param._local_tensor.numel()
    return param.numel()


class ParamCountCallback(Callback):
    def on_train_begin(self, model: FastGenModel, **kwargs) -> None:
        # get modules
        modules = {"model": model, **model.model_dict}

        # iterate over modules
        output = {}
        for name, module in modules.items():
            # Logical (full model) param counts
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in module.parameters())

            # Local (sharded) param counts - what's actually in memory on this rank
            local_trainable_params = sum(_get_local_numel(p) for p in module.parameters() if p.requires_grad)
            local_total_params = sum(_get_local_numel(p) for p in module.parameters())

            # check if parameter counts are different across ranks
            if world_size() > 1:
                trainable_params = self.gather_param_counts(trainable_params)
                total_params = self.gather_param_counts(total_params)
                local_trainable_params = self.gather_param_counts(local_trainable_params)
                local_total_params = self.gather_param_counts(local_total_params)
                if len(set(total_params)) == 1 and len(set(trainable_params)) == 1:
                    trainable_params = trainable_params[0]
                    total_params = total_params[0]
                if len(set(local_total_params)) == 1 and len(set(local_trainable_params)) == 1:
                    local_trainable_params = local_trainable_params[0]
                    local_total_params = local_total_params[0]

            # logging
            module_name = module.__class__.__name__
            output.update(
                {
                    f"{name}/trainable_params": trainable_params,
                    f"{name}/total_params": total_params,
                    f"{name}/local_trainable_params": local_trainable_params,
                    f"{name}/local_total_params": local_total_params,
                }
            )
            if isinstance(trainable_params, list):
                logger.warning(f"Parameter counts differ across ranks for {module_name}.")
                for rank, (p_train, p) in enumerate(zip(trainable_params, total_params)):
                    logger.info(
                        f"{name} ({module_name}) has {p_train * 1.e-6:.2f} M trainable and {p * 1.e-6:.2f} M total params on rank {rank}."
                    )
            else:
                logger.info(
                    f"{name} ({module_name}) has {trainable_params * 1.e-6:.2f} M trainable and {total_params * 1.e-6:.2f} M total params (logical)."
                )

            # Report local/sharded counts
            if isinstance(local_trainable_params, list):
                for rank, (p_train, p) in enumerate(zip(local_trainable_params, local_total_params)):
                    logger.info(
                        f"{name} ({module_name}) has {p_train * 1.e-6:.2f} M trainable and {p * 1.e-6:.2f} M total params LOCAL on rank {rank}."
                    )
            else:
                is_sharded = local_total_params < total_params if not isinstance(total_params, list) else True
                if is_sharded:
                    logger.info(
                        f"{name} ({module_name}) has {local_trainable_params * 1.e-6:.2f} M trainable and {local_total_params * 1.e-6:.2f} M total params LOCAL per rank (sharding ratio: {world_size()}x)."
                    )
                else:
                    logger.info(f"{name} ({module_name}) is NOT sharded (local == logical params).")

        if wandb.run:
            wandb.run.summary.update(output)

    def gather_param_counts(self, param_count):
        """
        Gather parameter counts across all ranks.

        Args:
            param_count: Parameter count to gather.

        Returns:
            List of parameter counts across all ranks.
        """
        param_count = torch.tensor(
            [param_count], dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        param_count_list = [torch.zeros_like(param_count) for _ in range(world_size())]
        torch.distributed.all_gather(param_count_list, param_count)
        return [p.item() for p in param_count_list]
