# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, TYPE_CHECKING, Union
import os
import torch
import torch.distributed as dist
from datetime import timedelta
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


def init(required: bool = False) -> bool:
    """Initialize the default distributed process group when launched for it.

    Uses the torchtitan-style mixed-backend init: a single default process
    group with both ``cuda:nccl`` (for CUDA tensors) and ``cpu:gloo`` (for
    CPU tensors) registered. Collectives dispatch to the right backend by
    the tensor's device, so DTensor reductions on CPU (FSDP CPU-offload)
    resolve to gloo without callers having to thread an explicit group.

    The process-group timeout is one hour unless the FastGen-specific ``NCCL_TIMEOUT``
    environment variable (seconds) overrides it.

    Args:
        required: Raise when no compatible launcher environment is present.

    Returns:
        Whether a distributed process group is initialized.
    """
    if dist.is_available() and dist.is_initialized():
        return True

    launcher_vars = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in launcher_vars if name not in os.environ]
    if missing:
        if required:
            raise RuntimeError(
                "Distributed execution requires torchrun or another launcher that sets "
                f"RANK, WORLD_SIZE, and LOCAL_RANK; missing {', '.join(missing)}."
            )
        return False

    if not dist.is_available() or not torch.cuda.is_available():
        raise RuntimeError("Distributed CUDA execution is not available, please check your PyTorch installation.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Large multi-model configs (e.g. 3x14B DMD2) can spend >10 min on rank-0
    # weight load + meta-init sync before the first step.
    timeout_seconds = int(os.environ.get("NCCL_TIMEOUT", "3600"))  # Default 1 hour
    timeout = timedelta(seconds=timeout_seconds)

    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )
    torch.cuda.set_device(local_rank)

    logger.info(
        f"[{os.getpid()}] rank = {dist.get_rank()} ({local_rank}), world_size = {world_size}, timeout = {timeout_seconds}s, backend=cuda:nccl,cpu:gloo"
    )
    return True


def model_to_ddp(model: FastGenModel) -> Union[FastGenModel, torch.nn.parallel.DistributedDataParallel]:
    """Convert model to distributed data parallel."""
    if torch.distributed.is_available() and torch.cuda.is_available():
        model = DDPWrapper(
            model,
            device_ids=[int(os.environ.get("LOCAL_RANK", "0"))],
            output_device=int(os.environ.get("LOCAL_RANK", "0")),
            find_unused_parameters=model.config.ddp_find_unused_parameters,
        )
    else:
        raise RuntimeError("Distributed data parallel is not available")
    return model


class DDPWrapper(torch.nn.parallel.DistributedDataParallel):
    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.show_sync_grad_static_graph_warning = True

    def single_train_step(self, *args, **kwargs) -> Any:
        """Run the model's ``single_train_step`` through DDP's forward.

        With ``find_unused_parameters=True``, DDP joins every returned tensor
        in one ``_DDPSink`` autograd node. A backward over one output then
        traverses all other attached outputs' graphs with undefined gradients,
        which cuDNN SDPA backward turns into garbage on some stacks. Returned
        tensors must therefore be detached or have their graph covered by the
        jointly backwarded losses.
        """

        def wrapped_training_step(*_args, **_kwargs):  # noqa: ANN202
            # The actual .single_train_step.
            return self.module.single_train_step(*_args, **_kwargs)

        # Patch the original_module's forward so we can redirect the arguments back to the real method.
        self.module.forward = wrapped_training_step
        # Call self, which implicitly calls self.forward() --> model.forward(), which is now model.training_step().
        # Without calling self.forward() or model.forward() explicitly, implicit hooks are also executed.
        return self(*args, **kwargs)


@contextmanager
def ddp_sync_grad(model: FastGenModel, enabled: bool):
    r"""
    Context manager to enable/disable gradient synchronizations across DDP processes for DDP model.
    Modified from:
    https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/parallel/distributed.py#L1442
    Note that this is incompatible with static_graph=True and will be an no-op if static_graph=True.

    Within this context, gradients will be accumulated on module
    variables, which will later be synchronized in the first
    forward-backward pass exiting the context.

    .. warning::
        The forward pass should be included inside the context manager, or
        else gradients will still be synchronized.
    """
    assert isinstance(model, torch.nn.Module)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        old_require_backward_grad_sync = model.require_backward_grad_sync
        if model.static_graph and model.require_backward_grad_sync != enabled:
            if model.show_sync_grad_static_graph_warning:
                logger.warning("DDP static_graph=True is incompatible with sync_grad(). Performance will be reduced.")
                model.show_sync_grad_static_graph_warning = False
        else:
            model.require_backward_grad_sync = enabled
    try:
        yield
    finally:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.require_backward_grad_sync = old_require_backward_grad_sync
