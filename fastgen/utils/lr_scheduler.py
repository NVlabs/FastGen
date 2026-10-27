# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import bisect
import itertools

import numpy as np

import fastgen.utils.logging_utils as logger


class LambdaLinearScheduler:
    """
    Piecewise-linear learning-rate multiplier made of consecutive cycles.

    Cycle ``i`` covers ``cycle_lengths[i]`` steps. The multiplier rises linearly from ``f_start[i]`` to ``f_max[i]``
    over the first ``warm_up_steps[i]`` of them, then falls linearly and reaches ``f_min[i]`` at the end of the cycle,
    where the next cycle begins at its own ``f_start``. After the last cycle, the multiplier stays at that cycle's
    ``f_min``.
    """

    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        if not len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths):
            raise ValueError("warm_up_steps, f_min, f_max, f_start, and cycle_lengths need one entry per cycle")
        if len(cycle_lengths) == 0:
            raise ValueError("LambdaLinearScheduler requires at least one cycle")
        for warm_up, cycle_length in zip(warm_up_steps, cycle_lengths):
            if cycle_length <= 0:
                raise ValueError(f"cycle_lengths must be positive, got {cycle_length}")
            if warm_up >= cycle_length:
                raise ValueError(f"warm_up_steps must be shorter than their cycle, got {warm_up} >= {cycle_length}")

        cycle_starts = itertools.accumulate(cycle_lengths, initial=0)
        self.cycles = list(zip(cycle_starts, cycle_lengths, warm_up_steps, f_start, f_max, f_min))
        self.cycle_ends = list(itertools.accumulate(cycle_lengths))
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        index = min(bisect.bisect_right(self.cycle_ends, n), len(self.cycles) - 1)
        cycle_start, cycle_length, warm_up, f_start, f_max, f_min = self.cycles[index]
        step = n - cycle_start
        if step < warm_up:
            f = f_start + (f_max - f_start) / warm_up * step
        else:
            f = f_max - (f_max - f_min) * min((step - warm_up) / (cycle_length - warm_up), 1.0)
        if self.verbosity_interval > 0 and n % self.verbosity_interval == 0:
            logger.info(f"Step {n} (cycle {index}): lr multiplier {f}")
        return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaInverseSquareRootScheduler:
    """
    Learning rate decay schedule used in the paper "Analyzing and Improving
    the Training Dynamics of Diffusion Models".
    """

    def __init__(self, warm_up_steps, decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.decay_steps = decay_steps
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        f = 1.0
        if n > self.decay_steps and self.decay_steps > 0:
            f /= np.sqrt(n / self.decay_steps)

        if n < self.lr_warm_up_steps:
            f *= n / self.lr_warm_up_steps

        if self.verbosity_interval > 0 and n % self.verbosity_interval == 0:
            logger.info(f"Step {n}: lr multiplier {f}")
        return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)
