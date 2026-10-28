# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.opt import get_scheduler
from fastgen.utils.lr_scheduler import LambdaLinearScheduler


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.0), (2, 1.0), (6, 0.6), (10, 0.5), (11, 0.75), (20, 0.1), (21, 0.1), (100, 0.1)],
)
def test_linear_schedule_boundaries_and_completed_cycles(step, expected):
    scheduler = LambdaLinearScheduler(
        warm_up_steps=[2, 2],
        f_min=[0.2, 0.1],
        f_max=[1.0, 1.0],
        f_start=[0.0, 0.5],
        cycle_lengths=[10, 10],
    )
    assert scheduler(step) == pytest.approx(expected)


def test_linear_schedule_requires_a_cycle():
    with pytest.raises(ValueError, match="at least one cycle"):
        LambdaLinearScheduler(warm_up_steps=[], f_min=[], f_max=[], f_start=[], cycle_lengths=[])


def test_linear_schedule_boundary_step_starts_the_next_cycle():
    """A cycle boundary belongs to the new cycle: it emits that cycle's f_start while the step before it is still
    decaying; after the last cycle the schedule stays clamped at f_min."""
    scheduler = LambdaLinearScheduler(
        warm_up_steps=[5, 5], f_min=[0.1, 0.1], f_max=[1.0, 1.0], f_start=[0.0, 0.0], cycle_lengths=[10, 10]
    )
    assert scheduler(9) == pytest.approx(1.0 - 0.9 * 4 / 5)
    assert scheduler(10) == 0.0
    assert scheduler(20) == pytest.approx(0.1)
    assert scheduler(25) == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("warm_up_steps", "cycle_lengths", "match"),
    [([2], [0], "positive"), ([10], [10], "shorter than their cycle"), ([11], [10], "shorter than their cycle")],
)
def test_linear_schedule_rejects_impossible_cycles(warm_up_steps, cycle_lengths, match):
    with pytest.raises(ValueError, match=match):
        LambdaLinearScheduler(
            warm_up_steps=warm_up_steps, f_min=[0.1], f_max=[1.0], f_start=[0.0], cycle_lengths=cycle_lengths
        )


def test_get_scheduler_covers_every_parameter_group():
    first, second = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([{"params": [first], "lr": 1.0}, {"params": [second], "lr": 0.5}])
    scheduler_config = L(LambdaLinearScheduler)(
        warm_up_steps=[2], cycle_lengths=[10], f_start=[0.0], f_max=[1.0], f_min=[0.5]
    )
    scheduler = get_scheduler(optimizer, scheduler_config)
    optimizer.step()
    scheduler.step()
    # Step 1 of the warm-up multiplies every group's base learning rate by 0.5.
    assert scheduler.get_last_lr() == pytest.approx([0.5, 0.25])
