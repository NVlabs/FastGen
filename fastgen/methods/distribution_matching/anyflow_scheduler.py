# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight flow-map scheduler for any-step inference.

Ported from the AnyFlow reference implementation
(``far/schedulers/scheduling_flowmap_euler_discrete.py``) with the
``diffusers.ConfigMixin`` dependency removed so it can be used standalone.

The scheduler operates on timesteps in ``[0, num_train_timesteps]`` (matching
the Wan T2V/I2V conventions). For pair-step sampling, ``step`` takes both the
current timestep ``t`` and the target ``r`` (with ``r < t``) and integrates the
flow map prediction in one shot:

    x_r = x_t - (t - r) * u_theta(x_t, t, r)

This is the flow-map analogue of an Euler step where the integration interval
``t - r`` is chosen freely at inference time, enabling any-step sampling.
"""

from __future__ import annotations

from typing import Union

import torch


class FlowMapDiscreteScheduler:
    """Any-step flow-map scheduler.

    Args:
        num_train_timesteps: Maximum timestep value used during training.
        shift: Flow-matching schedule shift (Wan default: 5.0 for video, 1.0 for image).
        weight_type: Per-timestep loss weighting scheme — ``gaussian``, ``beta08``,
            or ``uniform``. ``beta08`` matches AnyFlow's default.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        weight_type: str = "beta08",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.weight_type = weight_type

        # Initialise with train-time uniform spacing; overridden by set_timesteps()
        self.set_timesteps(num_train_timesteps, device="cpu")
        self._build_train_weights()

    def _build_train_weights(self) -> None:
        if self.weight_type == "gaussian":
            x = self.timesteps
            y = torch.exp(-2 * ((x - self.num_train_timesteps / 2) / self.num_train_timesteps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (self.num_train_timesteps / y_shifted.sum())
        elif self.weight_type == "beta08":
            t = self.timesteps / self.num_train_timesteps
            y = (t**1.0) * ((1 - t) ** 0.5)
            self.linear_timesteps_weights = y * (self.num_train_timesteps / y.sum())
        elif self.weight_type == "uniform":
            self.linear_timesteps_weights = torch.ones_like(self.timesteps)
        else:
            raise ValueError(f"Invalid weight_type: {self.weight_type!r}")

    @torch.no_grad()
    def get_train_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Look up the per-timestep training loss weight via nearest neighbour."""
        device_weights = self.linear_timesteps_weights.to(timesteps.device)
        device_timesteps = self.timesteps.to(timesteps.device)
        diffs = (device_timesteps.unsqueeze(1) - timesteps.flatten().unsqueeze(0)).abs()
        timestep_id = torch.argmin(diffs, dim=0).reshape(timesteps.shape)
        return device_weights[timestep_id]

    def apply_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Apply the flow-matching schedule shift to normalized sigmas in [0, 1]."""
        if self.shift == 1.0:
            return sigmas
        return self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float64, device=device)
        timesteps = self.apply_shift(timesteps)
        self.timesteps = timesteps * self.num_train_timesteps

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward-noise ``sample`` to ``timestep`` along a linear flow-matching path."""
        timestep = torch.as_tensor(timestep, device=sample.device, dtype=sample.dtype)
        timestep = timestep / self.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (noise.ndim - timestep.ndim)))
        return timestep * noise + (1.0 - timestep) * sample

    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        r_timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Pair-step Euler integration ``x_r = x_t - (t - r) * model_output``."""
        timestep = torch.as_tensor(timestep, device=sample.device, dtype=sample.dtype)
        r_timestep = torch.as_tensor(r_timestep, device=sample.device, dtype=sample.dtype)
        timestep = timestep / self.num_train_timesteps
        r_timestep = r_timestep / self.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (model_output.ndim - timestep.ndim)))
        r_timestep = r_timestep.view(*r_timestep.shape, *([1] * (model_output.ndim - r_timestep.ndim)))
        prev_sample = sample - (timestep - r_timestep) * model_output
        return prev_sample.to(model_output.dtype)
