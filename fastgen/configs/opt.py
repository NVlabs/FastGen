# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig
import torch
from torch.optim import Optimizer, Adam, AdamW, RAdam, lr_scheduler

from fastgen.configs.instantiate import LazyCall as L, instantiate
from fastgen.utils.lr_scheduler import LambdaLinearScheduler


def get_optimizer(
    model: torch.nn.Module, optim_type: str = "adam", lr: float = 1e-4, weight_decay: float = 0.01, **kwargs
) -> Optimizer:
    if optim_type == "adam":
        opt_cls = Adam
    elif optim_type == "adamw":
        opt_cls = AdamW
    elif optim_type == "radam":
        opt_cls = RAdam
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type!r}; expected 'adam', 'adamw', or 'radam'.")
    optimizer = opt_cls(
        params=[p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay, **kwargs
    )
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict | DictConfig,
) -> lr_scheduler.LambdaLR:
    net_scheduler = instantiate(scheduler_config)
    # One callable applies to every parameter group.
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=net_scheduler.schedule)


BaseOptimizerConfig: DictConfig = L(get_optimizer)(
    model=None,
    optim_type="adamw",
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    fused=False,
)


RAdamOptimizerConfig: DictConfig = L(get_optimizer)(
    model=None,
    optim_type="radam",
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
)


BaseSchedulerConfig: DictConfig = L(LambdaLinearScheduler)(
    warm_up_steps=[1000],  # warm up in the first 1000 iterations
    cycle_lengths=[10000000000],  # it means there is no lr decay
    f_start=[1.0e-6],
    f_max=[1.0],
    f_min=[1.0],
)
