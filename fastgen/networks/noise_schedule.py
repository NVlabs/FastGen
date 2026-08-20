# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""factory for different noise schedules."""

from abc import abstractmethod
from typing import Optional, Tuple

import torch
import numpy as np
from scipy import stats
from torch.distributions import Normal
from diffusers import DDIMScheduler, CogVideoXDPMScheduler

import fastgen.utils.logging_utils as logger
from fastgen.utils import expand_like
from fastgen.utils.basic_utils import PRECISION_MAP


NET_PRED_TYPES = {"x0", "eps", "v", "flow"}


class BaseNoiseSchedule(torch.nn.Module):
    """Abstract base noise schedule class.

    This class defines a noise schedule used for training/sampling from
    diffusion models.

    We follow the general format to define the forward process:

        x_t = alpha(t) x_0 + sigma(t) eps

    """

    def __init__(
        self,
        min_t: float,
        max_t: float,
        num_steps: int,
        clamp_min: float = 1e-6,
        t_precision: str = "float64",
        **kwargs,
    ):
        super().__init__()
        self._min_t = min_t
        self._max_t = max_t
        self.num_steps = num_steps
        self._supported_time_dist_types = ("lognormal", "logitnormal", "uniform", "polynomial", "shifted", "log_t")
        self.clamp_min = clamp_min
        self.t_precision = PRECISION_MAP[t_precision]

    @property
    def min_t(self) -> float:
        """The minimum time step supported by this schedule.

        Returns:
            float: Minimum time step.
        """
        assert self._min_t >= 0, "min_t must be non-negative"
        return self._min_t

    @property
    def max_t(self) -> float:
        """The maximum time step supported by this schedule.

        Returns:
            float: Maximum time step.
        """
        assert self._max_t > self._min_t, "max_t must be greater than min_t"
        return self._max_t

    def latents(self, noise: torch.Tensor, t_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scale the noise by the sigma at the initial time step.

        Args:
            noise: Noise tensor.
            t_init: Initial time step. If None, use the maximum time step.

        Returns:
            torch.Tensor: Scaled noise tensor.
        """
        if t_init is None:
            t_init = torch.as_tensor(self.max_t, dtype=self.t_precision, device=noise.device)

        # we cast to double to be consistent with the `forward_process` method
        assert self.is_t_valid(t_init), f"t_init must be in [{self.min_t}, {self.max_t}], but got {t_init}"
        sigma = expand_like(self._sigma(t_init.to(torch.float64)), noise)
        return (noise.to(torch.float64) * sigma).to(noise.dtype)

    @staticmethod
    def safe_clamp(t: torch.Tensor, min: float | None = None, max: float | None = None) -> torch.Tensor:
        """
        Clamp the timestep t such that min <= t <= max holds in floating point precision.

        Args:
            t (torch.Tensor): The timestep.
            min (float | None): The minimum time step.
            max (float | None): The maximum time step.

        Returns:
            torch.Tensor: The clamped timestep.
        """
        if min is not None:
            min_tensor = torch.as_tensor(min, dtype=t.dtype, device=t.device)
            if min_tensor.item() < min:
                # round up to the next representable value
                new_min = torch.nextafter(min_tensor, torch.tensor(float("inf"), dtype=t.dtype, device=t.device)).item()
                assert new_min >= min
                min = new_min

        if max is not None:
            max_tensor = torch.as_tensor(max, dtype=t.dtype, device=t.device)
            if max_tensor.item() > max:
                # round down to the next representable value
                new_max = torch.nextafter(
                    max_tensor, torch.tensor(float("-inf"), dtype=t.dtype, device=t.device)
                ).item()
                assert new_max <= max
                max = new_max

        return torch.clamp(t, min=min, max=max)

    def non_zero_clamp(self, tensor: torch.Tensor) -> torch.Tensor:
        """Clamp the tensor such that it is non-zero."""
        return torch.where(
            tensor >= 0,
            tensor.clamp(min=self.clamp_min),
            tensor.clamp(max=-self.clamp_min),
        )

    @property
    @abstractmethod
    def max_sigma(self) -> float:
        """The maximum noise scale, i.e., sigma(max_t).

        Returns:
            float: Maximum noise scale.
        """

    def rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        """Rescale the timestep t to the range that is actually used by the model.

        Returns:
            torch.Tensor: rescaled timestep.
        """
        assert self.is_t_valid(t), f"t must be in range [{self.min_t}, {self.max_t}], but got {t}"
        return self._rescale_t(t)

    @abstractmethod
    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """The data scale alpha(t) at time `t`, as defined within the framework:

            x_t = alpha(t) x_0 + sigma(t) eps

        Args:
            t (torch.Tensor): The timestep.

        Returns:
            torch.Tensor: The scaling value at time `t`.
        """
        assert self.is_t_valid(t), f"t must be in range [{self.min_t}, {self.max_t}], but got {t}"
        return self._alpha(t)

    @abstractmethod
    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the noise scale sigma(t) at time `t`, as defined within the framework:

            x_t = alpha(t) x_0 + sigma(t) eps

        Args:
            t (torch.Tensor): The timestep.

        Returns:
            torch.Tensor: The noise at time `t`.
        """
        assert self.is_t_valid(t), f"t must be in range [{self.min_t}, {self.max_t}], but got {t}"
        return self._sigma(t)

    @abstractmethod
    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the time derivative of alpha(t) at time `t`, as defined within the framework:

            x_t = alpha(t) x_0 + sigma(t) eps

        Args:
            t (torch.Tensor): The timestep.

        Returns:
            torch.Tensor: The time derivative of alpha(t) at time `t`.
        """
        assert self.is_t_valid(t), f"t must be in range [{self.min_t}, {self.max_t}], but got {t}"
        return self._alpha_prime(t)

    @abstractmethod
    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the time derivative of sigma(t) at time `t`, as defined within the framework:

            x_t = alpha(t) x_0 + sigma(t) eps

        Args:
            t (torch.Tensor): The timestep.

        Returns:
            torch.Tensor: The time derivative of sigma(t) at time `t`.
        """
        assert self.is_t_valid(t), f"t must be in range [{self.min_t}, {self.max_t}], but got {t}"
        return self._sigma_prime(t)

    @abstractmethod
    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigmas(self) -> torch.Tensor:
        """The noise scales sigma(t) at all time steps, as defined within the framework:

            x_t = alpha(t) x_0 + sigma(t) eps

        Returns:
            torch.Tensor: The noise scales.
        """

    @abstractmethod
    def sample_t(
        self,
        n: int,
        time_dist_type: str = "uniform",
        min_t: Optional[float] = None,
        max_t: Optional[float] = None,
        device: Optional[torch.device] = None,
        **sample_kwargs,
    ) -> torch.Tensor:
        """Sample random time steps for sampling the teacher model for student model distillation

        Args:
            n (int): Batch size.
            time_dist_type (str): The type of time t distribution to sample from.
            min_t: Minimum time step. Defaults to None.
            max_t: Maximum time step. Defaults to None.
            device: Device for the output tensor. Defaults to None.

        Returns:
            torch.Tensor: Tensor containing time steps, of shape [n].
        """

    def get_t_list(self, sample_steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample a list of timesteps [max_t, ..., 0] for multistep student model distillation

        Args:
            sample_steps: Number of timesteps to return
            device: Target device for the output tensor

        Returns:
            torch.Tensor: List of `num_steps + 1` timesteps uniformly sampled in decreasing order
        """
        device = device or self.sigmas.device
        t_list = torch.linspace(self.max_t, 0, sample_steps + 1, device=device, dtype=self.t_precision)
        return self.safe_clamp(t_list, max=self.max_t)

    def sample_from_t_list(
        self,
        n: int,
        sample_steps: int,
        t_list: Optional[list] = None,
        return_ids: Optional[bool] = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Sample timestep from the t_list tensor. If t_list is not given, it uses self.get_t_list() to
            get the default t_list tensor.

        Args:
            n (int): Batch size.
            sample_steps: (int) Number of timesteps to return
            t_list (torch.Tensor): list of finite time steps
            return_ids (bool): If True, return timestep indices instead of time step indices.
            device: torch device

        Return:
            torch.Tensor: Tensor containing time steps, of shape [n].
        """
        if t_list is None:
            device = device or self.sigmas.device
            t_list = self.get_t_list(sample_steps=sample_steps, device=device)
        else:
            t_list = torch.tensor(t_list, device=device, dtype=self.t_precision)

        ids = torch.randint(0, len(t_list) - 1, (n,)).to(device=device)  # Do not train on clean data (t = 0)
        if return_ids:
            return t_list[ids], ids
        return t_list[ids]

    def next_in_t_list(
        self,
        ids: torch.Tensor,
        sample_steps: int,
        t_list: torch.Tensor | None,
        device: Optional[torch.device] = None,
        stride: int = 1,
    ) -> torch.Tensor:
        """
        Get the next time step in the t_list based on the ids.

        Args:
            ids: Tensor of shape [batch_size] containing the ids of the time steps to get the next time step for.
            sample_steps: Number of timesteps in the t_list.
            t_list: Tensor of shape [sample_steps + 1] containing the time steps.
            device: Device to store the output tensor. Defaults to None.
            stride: Stride to use to get the next time step. Defaults to 1.

        Returns:
            Tensor of shape [batch_size] containing the next time steps.
        """
        if t_list is None:
            t_list = self.get_t_list(sample_steps=sample_steps, device=device)
        else:
            t_list = torch.as_tensor(t_list, device=device, dtype=self.t_precision)
            assert t_list.shape == (
                sample_steps + 1,
            ), f"t_list must be of shape (sample_steps + 1,), but got {t_list.shape}"

        next_ids = ids + stride
        if next_ids.max() > sample_steps:
            raise ValueError(
                f"Clamping next ids to sample steps. ids: {ids}, next_ids: {next_ids}, sample_steps: {sample_steps}"
            )
        return t_list[next_ids]

    def sample_t_inhom(
        self,
        n: int,
        seq_len: int,
        chunk_size: int,
        sample_steps: int,
        t_list: Optional[list] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample block-wise independent t variables for diffusion forcing in distillation.

        Args:
            n: batch size
            seq_len: sequence length
            chunk_size: chunk size
            t_list: list of finite time steps
            device: torch device

        Return:
            Tuple[torch.Tensor, torch.Tensor]: The first tensor contains time steps and
                the second tensor contains the indices, both of shape [n, seq_len].
        """
        if t_list is None:
            target_device = device or self.sigmas.device
            t_list = self.get_t_list(sample_steps, device=target_device)
        else:
            t_list = torch.tensor(t_list, device=device, dtype=self.t_precision)

        num_chunks = seq_len // chunk_size
        remaining_size = seq_len % chunk_size
        unique_id = torch.randint(0, len(t_list), (n, num_chunks)).to(device=device)

        # First chunk’s t_id is repeated for the first (chunk_size + remaining_size) elements
        first_col = unique_id[:, :1].repeat(1, chunk_size + remaining_size)

        # Remaining chunks: each t_id repeated chunk_size times
        rest_cols = unique_id[:, 1:].repeat_interleave(chunk_size, dim=1)

        # Concatenate: [first chunk | remaining chunks] → shape (n, seq_len)
        ids = torch.cat([first_col, rest_cols], dim=1)
        return t_list[ids], ids

    def sample_t_inhom_sft(self, n: int, seq_len: int, chunk_size: int, **sample_t_kwargs) -> torch.Tensor:
        """
        Sample block-wise independent t variables for diffusion forcing in SFT.

        Args:
            n: batch size
            seq_len: sequence length
            chunk_size: chunk size

        Return:
            torch.Tensor: time steps of shape [n, seq_len].
        """
        num_chunks = seq_len // chunk_size
        remaining_size = seq_len % chunk_size
        flat_t = self.sample_t(n * num_chunks, **sample_t_kwargs)  # shape (n*num_chunks,)

        t = flat_t.view(n, num_chunks)  # shape (n, num_chunks)

        first_col = t[:, :1].repeat(1, chunk_size + remaining_size)
        rest_cols = t[:, 1:].repeat_interleave(chunk_size, dim=1)

        t_inhom = torch.cat([first_col, rest_cols], dim=1)
        return t_inhom

    def is_t_valid(self, t: torch.Tensor) -> torch.bool:
        if t.dtype in (
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):  # nextafter only supports these dtypes
            dtype = t.dtype
        else:
            dtype = torch.float32
        min_t = torch.as_tensor(self.min_t, dtype=dtype, device=t.device)
        max_t = torch.as_tensor(self.max_t, dtype=dtype, device=t.device)
        # deal with dtype issue
        lower = torch.nextafter(min_t, torch.tensor(float("-inf"), dtype=dtype, device=t.device))
        upper = torch.nextafter(max_t, torch.tensor(float("inf"), dtype=dtype, device=t.device))
        return torch.all((lower <= t) & (t <= upper))

    def forward_process(self, x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The diffusion forward process for this noise schedule.

        Args:
            x (torch.Tensor): Input sample from the data distribution.
            eps (torch.Tensor): Noise sample, should have the same shape as `x`.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: Sample from the forward process at time `t` given `x`.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = x.dtype

        # Convert input tensors to double precision
        t = t.to(torch.float64)
        x = x.to(torch.float64)
        eps = eps.to(torch.float64)

        # Perform calculation in double precision
        alpha_t = expand_like(self._alpha(t), x)
        sigma_t = expand_like(self._sigma(t), eps)
        forward_process = x.mul(alpha_t).add(eps.mul(sigma_t))
        return forward_process.to(original_dtype)

    def cond_velocity(self, x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The instantaneous velocity for this noise schedule: dx/dt := alpha'(t) x_0 + sigma'(t) eps.

        Args:
            x (torch.Tensor): Input sample from the data distribution.
            eps (torch.Tensor): Noise sample, should have the same shape as `x`.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: Sample from the forward process at time `t` given `x`.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = x.dtype

        # Convert input tensors to double precision
        t = t.to(torch.float64)
        x = x.to(torch.float64)
        eps = eps.to(torch.float64)

        # Perform calculation in double precision
        alpha_prime_t = expand_like(self._alpha_prime(t), x)
        sigma_prime_t = expand_like(self._sigma_prime(t), eps)
        cond_vel = x.mul(alpha_prime_t).add(eps.mul(sigma_prime_t))

        return cond_vel.to(original_dtype)

    def closest_sigma_idx(self, sigma_t: torch.Tensor) -> torch.Tensor:
        """
        Find the closest `sigma` in `self.sigmas` that matches the target `sigma` value

        Args:
            sigma_t (torch.Tensor): The target `sigma`.

        Returns:
            torch.Tensor: Index of the closest `sigma` in `self.sigmas` that matches the target `sigma`.
        """
        shape = sigma_t.shape
        if sigma_t.ndim != 0:
            sigma_t = sigma_t.squeeze(list(range(1, sigma_t.ndim)))

        sigmas = self.sigmas.to(sigma_t)
        assert sigmas.ndim == 1

        # Find the right and left indices
        indices_right = torch.searchsorted(sigmas, sigma_t, side="right")
        indices_left = (indices_right - 1).clamp(min=0)
        indices_right = indices_right.clamp(max=sigmas.numel() - 1)

        # Compute the differences and return the closest index
        right_diff = torch.abs(sigmas[indices_right] - sigma_t)
        left_diff = torch.abs(sigmas[indices_left] - sigma_t)
        indices = torch.where(right_diff < left_diff, indices_right, indices_left)
        return indices.view(shape)

    @abstractmethod
    def sigma_idx_to_t(self, sigma_idx: torch.Tensor) -> torch.Tensor:
        """
        Find `t` that matches the target `sigma` value based on `sigma_idx`.

        Args:
            sigma_idx (torch.Tensor): The index of the tensor `self.sigmas`.

        Returns:
            torch.Tensor: The timestep
        """

    def sqrt_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the square root of the Signal-to-Noise Ratio (SNR) at time `t`.
        sqrt_snr(t) = alpha(t) / sigma(t)
        """
        assert self.is_t_valid(t)
        # Calculate in double precision
        t = t.to(torch.float64)
        alpha = self._alpha(t)
        sigma = self._sigma(t)

        sqrt_snr_val = alpha / self.non_zero_clamp(sigma)
        return sqrt_snr_val.to(t.dtype)

    @abstractmethod
    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """
        Find `t` that matches the target square root of SNR value, where sqrt_snr_t := alpha_t / sigma_t.

        Args:
            sqrt_snr_t (torch.Tensor): The square root of the SNR at time `t`.

        Returns:
            torch.Tensor: The timestep
        """

    def x0_to_eps(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the eps-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The eps-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t = t.to(torch.float64)
        xt = xt.to(torch.float64)
        x0 = x0.to(torch.float64)
        alpha_t = expand_like(self._alpha(t), xt)
        sigma_t = expand_like(self._sigma(t), xt)

        # Perform calculation in double precision
        eps = (xt - x0.mul(alpha_t)).div(self.non_zero_clamp(sigma_t))

        # Convert back to original dtype
        return eps.to(original_dtype)

    def eps_to_x0(self, xt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the x0-prediction from the eps-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            eps (torch.Tensor): The eps-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t = t.to(torch.float64)
        xt = xt.to(torch.float64)
        eps = eps.to(torch.float64)

        # Perform calculation in double precision
        alpha_t = expand_like(self._alpha(t), xt)
        sigma_t = expand_like(self._sigma(t), xt)

        # Perform calculation in double precision
        x0 = (xt - eps.mul(sigma_t)).div(self.non_zero_clamp(alpha_t))

        # Convert back to original dtype
        return x0.to(original_dtype)

    def flow_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the x0-prediction from the flow-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The flow-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t = t.to(torch.float64)
        xt = xt.to(torch.float64)
        v = v.to(torch.float64)

        alpha_t = expand_like(self._alpha(t), xt)
        sigma_t = expand_like(self._sigma(t), xt)
        alpha_prime_t = expand_like(self._alpha_prime(t), xt)
        sigma_prime_t = expand_like(self._sigma_prime(t), xt)

        # Perform calculation in double precision
        xt_coeff = sigma_prime_t / self.non_zero_clamp(sigma_t)
        x0_coeff = xt_coeff * alpha_t - alpha_prime_t
        x0 = (xt * xt_coeff - v) / self.non_zero_clamp(x0_coeff)

        # Convert back to original dtype
        return x0.to(original_dtype)

    @abstractmethod
    def x0_to_flow(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the flow-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The flow-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        return self.cond_velocity(x0, self.x0_to_eps(xt, x0, t), t)

    def convert_model_output(
        self,
        xt: torch.Tensor,
        model_output: torch.Tensor,
        t: torch.Tensor,
        src_pred_type: str = "x0",
        target_pred_type: str = "eps",
    ) -> torch.Tensor:
        """
        Convert the model output from src_pred_type to target_pred_type.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            model_output (`torch.Tensor`): The direct output from the learned diffusion model.
            t (torch.Tensor): The time step in the forward process.
            src_pred_type (`str`): The type of src prediction, choose from 'x0', 'eps', 'v', 'flow'.
            target_pred_type (`str`): The type of target prediction, choose from 'x0', 'eps', 'v', 'flow'.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        # If source and target are the same, return as-is
        if src_pred_type == target_pred_type:
            return model_output

        # Supported prediction types
        if src_pred_type not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported src_pred_type '{src_pred_type}'. Supported types: {NET_PRED_TYPES}")
        if target_pred_type not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported target_pred_type '{target_pred_type}'. Supported types: {NET_PRED_TYPES}")

        if src_pred_type == "v" or target_pred_type == "v":
            assert torch.allclose(
                self.alpha(t) ** 2 + self.sigma(t) ** 2, torch.ones_like(t), rtol=1e-2, atol=1e-2
            ), "Only AlphaNoiseSchedule supports v-prediction!"

        # Convert through intermediate representations
        # First convert src to x0 (if needed)
        if src_pred_type == "x0":
            x0 = model_output
        elif src_pred_type == "eps":
            x0 = self.eps_to_x0(xt, model_output, t)
        elif src_pred_type == "v":
            x0 = self.v_to_x0(xt, model_output, t)
        elif src_pred_type == "flow":
            x0 = self.flow_to_x0(xt, model_output, t)
        else:
            raise ValueError(f"Conversion from '{src_pred_type}' not implemented")

        # Then convert x0 to target (if needed)
        if target_pred_type == "x0":
            return x0
        elif target_pred_type == "eps":
            return self.x0_to_eps(xt, x0, t)
        elif target_pred_type == "v":
            return self.x0_to_v(xt, x0, t)
        elif target_pred_type == "flow":
            return self.x0_to_flow(xt, x0, t)
        else:
            raise ValueError(f"Conversion to '{target_pred_type}' not implemented")


class EDMNoiseSchedule(BaseNoiseSchedule):
    """
    The noise schedule defined in the EDM paper [Karras et al. 2022].

    x_t = x_0 + sigma(t) eps,

    where t in [0.002, 80]

    """

    def __init__(
        self,
        min_t: float = 0.002,
        max_t: float = 80.0,
        rho: float = 7.0,
        min_step_percent: float = 0.002,
        max_step_percent: float = 0.998,
        num_steps: int = 1000,
        **kwargs,
    ):
        super().__init__(min_t, max_t, num_steps, **kwargs)
        if not 0.002 <= min_t < max_t <= 80.0:
            logger.warning(f"EDM min_t and max_t should be between 0.002 and 80.0, but got {min_t} and {max_t}.")
        ramp = torch.linspace(0, 1, num_steps, dtype=self.t_precision)
        min_inv_rho = min_t ** (1 / rho)
        max_inv_rho = max_t ** (1 / rho)
        sigmas_flipped = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        self._sigmas = torch.flip(sigmas_flipped, [0])
        # Map time values to step indices for polynomial sampling
        self._min_step = int(min_step_percent * num_steps)
        self._max_step = int(max_step_percent * num_steps)

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    @property
    def max_sigma(self) -> float:
        """In EDM, sigma and t are the same, so as _max_t and max_sigma"""
        return self._max_t

    @property
    def sigmas(self) -> torch.Tensor:
        return self._sigmas

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def sigma_idx_to_t(self, sigma_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert sigma index back to timestep t.

        For EDM, sigma(t) = t, so this directly returns the sigma values.
        Note: No roundtrip assertion needed since sigma(t) = t is identity.

        Args:
            sigma_idx (torch.Tensor): Indices into the sigma array (long tensor)

        Returns:
            torch.Tensor: Corresponding timestep values (which equal sigma values for EDM)
        """
        assert sigma_idx.dtype == torch.long
        return self.sigmas.to(device=sigma_idx.device)[sigma_idx]

    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """
        Find timestep that matches the target square root of SNR value, where sqrt_snr_t := alpha_t / sigma_t.

        Args:
            sqrt_snr_t: The square root of the SNR at time t

        Returns:
            torch.Tensor: The timestep

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        original_dtype = sqrt_snr_t.dtype
        sqrt_snr_t = sqrt_snr_t.to(torch.float64)
        t = 1 / self.non_zero_clamp(sqrt_snr_t)
        return t.to(original_dtype)

    def _truncated_lognormal_sample(
        self, n: int, mean: float, std, min_t: float, max_t: float, device: Optional[torch.device] = None
    ):
        # Do all calculations on CPU in t_precision for numerical stability, then move to device
        device = device or torch.device("cpu")

        # Transform bounds to log space (on CPU)
        min_t_val = max(min_t, self.clamp_min)
        log_min_t = torch.tensor(min_t_val, dtype=self.t_precision).log()
        log_max_t = torch.tensor(max_t, dtype=self.t_precision).log()

        # Compute CDF values for truncation bounds
        normal = Normal(
            torch.tensor(mean, dtype=self.t_precision),
            torch.tensor(std, dtype=self.t_precision),
        )
        cdf_min = normal.cdf(log_min_t)
        cdf_max = normal.cdf(log_max_t)

        # Sample uniformly between cdf_min and cdf_max, apply icdf, then exp
        u = torch.rand(n, dtype=self.t_precision) * (cdf_max - cdf_min) + cdf_min
        t = normal.icdf(u).exp()

        # Move to device at the end
        return t.to(device=device)

    def _truncated_log_t_sample(
        self,
        n: int,
        mean: float,
        std,
        min_t: float,
        max_t: float,
        df: float = 2.0,
        device: Optional[torch.device] = None,
    ):
        # transform bounds to log space
        min_t = max(min_t, self.clamp_min)
        log_min_t = np.log(min_t)
        log_max_t = np.log(max_t)

        # Sample from truncated, log-transformed student_t
        lower_adjusted = (log_min_t - mean) / std
        upper_adjusted = (log_max_t - mean) / std

        # Get the CDF values for the adjusted bounds
        a = stats.t.cdf(lower_adjusted, df)
        b = stats.t.cdf(upper_adjusted, df)

        # Sample from the uniform distribution within the adjusted CDF bounds
        u = np.random.uniform(a, b, n)

        # Get the PPF and then adjust it to match the desired mean and scale
        log_t = stats.t.ppf(u, df) * std + mean

        # Exponentiate and move to device
        t = torch.from_numpy(np.exp(log_t)).to(device=device, dtype=self.t_precision)
        return t

    def sample_t(
        self,
        n: int,
        time_dist_type: str = "polynomial",
        train_p_mean: float = -1.2,
        train_p_std: float = 1.2,
        min_t: Optional[float] = 0.002,
        max_t: Optional[float] = 80.0,
        log_t_df: float = 0.01,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample random timesteps based on the EDM noise sampling schedule
        (at inference) for training.

        EDM adopts a parameterized scheme where the time steps are defined
        according to a sequence of noise levels (i.e., sigmas)
        """
        assert time_dist_type in self._supported_time_dist_types

        if min_t is not None and min_t < self.min_t:
            logger.warning(f"expected min_t >= {self.min_t}, got {min_t}")
        if max_t is not None and max_t > self.max_t:
            logger.warning(f"expected max_t <= {self.max_t}, got {max_t}")
        min_t = max(min_t, self.min_t) if min_t is not None else self.min_t
        max_t = min(max_t, self.max_t) if max_t is not None else self.max_t

        # Use provided device or default to _sigmas properties
        target_device = device or self._sigmas.device

        if time_dist_type == "lognormal":
            t = self._truncated_lognormal_sample(
                n,
                train_p_mean,
                train_p_std,
                min_t=min_t,
                max_t=max_t,
                device=target_device,
            )
        elif time_dist_type == "log_t":
            t = self._truncated_log_t_sample(
                n,
                train_p_mean,
                train_p_std,
                min_t=min_t,
                max_t=max_t,
                df=log_t_df,
                device=target_device,
            )
        elif time_dist_type == "uniform":
            t = torch.rand(n, device=target_device, dtype=self.t_precision) * (max_t - min_t) + min_t
        elif time_dist_type == "polynomial":
            indices = torch.randint(self._min_step, self._max_step + 1, (n,), device=self._sigmas.device)
            t = self._sigmas[indices].to(device=target_device, dtype=self.t_precision)
        else:
            raise ValueError(
                f"Unsupported time distribution type: {time_dist_type} in EDMNoiseSchedule."
                f"Currently only supports polynomial, uniform, and lognormal distributions."
            )
        return self.safe_clamp(t, min_t, max_t)

    def get_t_list(self, sample_steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get a uniformly spaced list of timesteps from the EDM sigma schedule.

        Timesteps are returned in decreasing order (high noise to low noise) for typical
        diffusion sampling workflows.

        Args:
            sample_steps: Number of timesteps to return
            device: Target device for the output tensor

        Returns:
            torch.Tensor: List of `num_steps+1` timesteps uniformly sampled from sigmas in decreasing order
        """
        device = device or self._sigmas.device

        # Sample num_steps points from sigmas in reverse order (largest to smallest)
        indices = torch.linspace(
            self._max_step,  # Start from max (largest sigma)
            self._min_step,  # End at min (smallest sigma)
            sample_steps + 1,
            device=self._sigmas.device,
        ).long()

        # Get timesteps from sigma indices
        t_list = self._sigmas[indices]

        # Replace the smallest timestep with exact zero for clean final step
        t_list = t_list.clone()  # Make a copy to avoid modifying the original
        t_list[-1] = 0.0

        # Move to target device if needed
        t_list = t_list.to(device=device, dtype=self.t_precision)
        return self.safe_clamp(t_list, max=self.max_t)

    def flow_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the x0-prediction from the flow-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The flow-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"

        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t_expanded = expand_like(t.to(torch.float64), xt)
        xt = xt.to(torch.float64)
        v = v.to(torch.float64)

        # Perform calculation in double precision
        x0 = xt - v.mul(t_expanded)

        # Convert back to original dtype
        return x0.to(original_dtype)

    def x0_to_flow(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the flow-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The flow-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"

        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t_expanded = expand_like(t.to(torch.float64), xt)
        xt = xt.to(torch.float64)
        x0 = x0.to(torch.float64)

        # Perform calculation in double precision
        flow = (xt - x0).div(self.non_zero_clamp(t_expanded))

        # Convert back to original dtype
        return flow.to(original_dtype)


class AlphasNoiseSchedule(BaseNoiseSchedule):
    """
    The SD noise schedule.

    x_t = alpha_cumprod(t).sqrt() x_0 + (1 - alpha_cumprod(t)).sqrt() eps,

    where t is in [0, 1], and alpha_cumprod(t) is from a predefined noise scheduler.

    """

    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        min_t: float = 0.0,
        max_t: float = 0.999,
        num_steps: int = 1000,
        **kwargs,
    ):
        super().__init__(min_t, max_t, num_steps, **kwargs)
        assert 0 <= min_t < max_t <= 0.999, "Alphas min_t and max_t must be between 0 and 0.999"
        self._alphas_cumprod = alphas_cumprod.to(self.t_precision)
        assert (
            len(self._alphas_cumprod) == num_steps
        ), f"alphas_cumprod's length {len(self._alphas_cumprod)} is not equal to num_steps {num_steps}"

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.num_steps * t

    @property
    def max_sigma(self) -> float:
        t_max_scale = int(self.num_steps * self.max_t)
        assert 0 <= t_max_scale < len(self._alphas_cumprod)
        return (1 - self._alphas_cumprod[t_max_scale]).sqrt().item()

    @property
    def sigmas(self) -> torch.Tensor:
        return (1 - self._alphas_cumprod).sqrt()

    def _t_to_idx(self, t: torch.Tensor) -> torch.Tensor:
        t_rescale = self._rescale_t(t).to(torch.long)
        # Clamp to prevent floating point precision issues at boundaries
        t_rescale = torch.clamp(t_rescale, 0, len(self._alphas_cumprod) - 1)
        return t_rescale

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        t_rescale = self._t_to_idx(t)
        return self._alphas_cumprod.to(t)[t_rescale].sqrt()

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        t_rescale = self._t_to_idx(t)
        return (1 - self._alphas_cumprod.to(t)[t_rescale]).sqrt()

    def sigma_idx_to_t(self, sigma_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert sigma index back to timestep t.

        This is the inverse of rescale_t: t = sigma_idx / num_steps

        Args:
            sigma_idx (torch.Tensor): Indices into the sigma array (long tensor)

        Returns:
            torch.Tensor: Corresponding timestep values
        """
        assert sigma_idx.dtype == torch.long
        return sigma_idx.to(self.t_precision) / self.num_steps

    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """Find timestep that matches the target square root of SNR value, where sqrt_snr_t := alpha_t / sigma_t.

        Args:
            sqrt_snr_t: The square root of the SNR at time t

        Returns:
            torch.Tensor: The timestep

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        original_dtype = sqrt_snr_t.dtype
        sqrt_snr_t = sqrt_snr_t.to(torch.float64)
        sigma_t = 1 / (1 + sqrt_snr_t**2)
        sigma_idx = self.closest_sigma_idx(sigma_t)
        return self.sigma_idx_to_t(sigma_idx).to(original_dtype)

    def sample_t(
        self,
        n: int,
        time_dist_type: str = "logitnormal",
        train_p_mean: float = 0,
        train_p_std: float = 1.0,
        min_t: Optional[float] = 0.001,
        max_t: Optional[float] = 0.999,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert time_dist_type in self._supported_time_dist_types

        if min_t is not None and min_t < self.min_t:
            logger.warning(f"expected min_t >= {self.min_t}, got {min_t}")
        if max_t is not None and max_t > self.max_t:
            logger.warning(f"_expected max_t <= {self.max_t}, got {max_t}")
        min_t = max(min_t, self.min_t) if min_t is not None else self.min_t
        max_t = min(max_t, self.max_t) if max_t is not None else self.max_t

        # Use provided device or default to _alphas_cumprod properties
        target_device = device if device is not None else self._alphas_cumprod.device

        if time_dist_type == "logitnormal":
            t = (
                torch.sigmoid(torch.randn(n, device=target_device, dtype=self.t_precision) * train_p_std + train_p_mean)
                * (max_t - min_t)
                + min_t
            )
        elif time_dist_type == "uniform":
            t = torch.rand(n, device=target_device, dtype=self.t_precision) * (max_t - min_t) + min_t
        else:
            raise ValueError(
                f"Unsupported time distribution type: {time_dist_type} in AlphasNoiseSchedule."
                f"Currently only supports logitnormal and uniform."
            )
        return self.safe_clamp(t, min_t, max_t)

    def v_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the x0-prediction from the v-prediction (x0 = alpha(t) xt - sigma(t) v).
        Only AlphaNoise has v-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The v-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert input tensors to double precision
        xt = xt.to(torch.float64)
        v = v.to(torch.float64)
        t = t.to(torch.float64)

        alpha_t = expand_like(self._alpha(t), xt)
        sigma_t = expand_like(self._sigma(t), xt)

        # Use higher tolerance for bfloat16 due to lower precision
        rtol = 1e-2 if original_dtype == torch.bfloat16 else 1e-3
        atol = 1e-2 if original_dtype == torch.bfloat16 else 1e-3

        # Perform constraint check in double precision
        assert torch.allclose(
            alpha_t**2 + sigma_t**2, torch.ones_like(alpha_t), rtol=rtol, atol=atol
        ), "v-prediction only supports the case of alpha_t^2+sigma_t^2 = 1."

        # Perform calculation in double precision
        x0 = xt.mul(alpha_t) - v.mul(sigma_t)

        # Convert back to original dtype
        return x0.to(original_dtype)

    def x0_to_v(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the v-prediction from the x0-prediction (v = (alpha(t) xt - x0) / sigma(t)).
        Only AlphaNoise has v-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The v-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert input tensors to double precision
        xt = xt.to(torch.float64)
        x0 = x0.to(torch.float64)
        t = t.to(torch.float64)

        alpha_t = expand_like(self._alpha(t), xt)
        sigma_t = expand_like(self._sigma(t), xt)

        # Use higher tolerance for bfloat16 due to lower precision
        rtol = 1e-2 if original_dtype == torch.bfloat16 else 1e-3
        atol = 1e-2 if original_dtype == torch.bfloat16 else 1e-3

        # Perform constraint check in double precision
        assert torch.allclose(
            alpha_t**2 + sigma_t**2, torch.ones_like(alpha_t), rtol=rtol, atol=atol
        ), "v-prediction only supports the case of alpha_t^2+sigma_t^2 = 1."

        # Perform calculation in double precision
        v = (xt.mul(alpha_t) - x0).div(self.non_zero_clamp(sigma_t))

        # Convert back to original dtype
        return v.to(original_dtype)


class SDNoiseSchedule(AlphasNoiseSchedule):
    def __init__(self, *args, **kwargs):
        scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        super().__init__(scheduler.alphas_cumprod, *args, **kwargs)
        del scheduler


class SDXLNoiseSchedule(AlphasNoiseSchedule):
    def __init__(self, *args, **kwargs):
        scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        super().__init__(scheduler.alphas_cumprod, *args, **kwargs)
        del scheduler


class CogVideoXNoiseSchedule(AlphasNoiseSchedule):
    def __init__(self, *args, model_id="THUDM/CogVideoX-5b", **kwargs):
        """Initialize the CogVideoX noise schedule."""

        # Store config for later access (e.g., config["num_train_timesteps"])
        if model_id == "THUDM/CogVideoX-5b":
            self.config = {
                "num_train_timesteps": 1000,
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "trained_betas": None,
                "clip_sample": False,
                "set_alpha_to_one": True,
                "steps_offset": 0,
                "prediction_type": "v_prediction",
                "clip_sample_range": 1.0,
                "sample_max_value": 1.0,
                "timestep_spacing": "trailing",
                "rescale_betas_zero_snr": True,
                "snr_shift_scale": 1.0,
            }
        else:
            self.config = {
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "beta_start": 0.00085,
                "trained_betas": None,
                "clip_sample": False,
                "clip_sample_range": 1.0,
                "num_train_timesteps": 1000,
                "prediction_type": "v_prediction",
                "rescale_betas_zero_snr": True,
                "sample_max_value": 1.0,
                "set_alpha_to_one": True,
                "snr_shift_scale": 3.0,
                "steps_offset": 0,
                "timestep_spacing": "trailing",
            }
        scheduler = CogVideoXDPMScheduler.from_config(self.config)
        super().__init__(scheduler.alphas_cumprod, *args, **kwargs)
        del scheduler


# Time distributions whose draws are mapped through shift * t / (1 + (shift - 1) * t).
# Consumers that rebuild that grid outside the sampler -- the flow-map loss weight
# normalization and the AnyFlow rollout schedule -- key the shift off this tuple, so
# adding a shifted variant below stays in sync with them.
SHIFTED_TIME_DIST_TYPES = ("shifted", "shifted_logitnormal")


class RFNoiseSchedule(BaseNoiseSchedule):
    """Rectified Flow noise schedule: x_t = (1-t)*x_0 + t*noise.

    Convention: t=0 is data, t=1 is noise.
    """

    def __init__(
        self,
        min_t: float = 0.0,
        max_t: float = 0.999,
        num_steps: int = 1000,
        **kwargs,
    ):
        super().__init__(min_t, max_t, num_steps, **kwargs)
        self._supported_time_dist_types = self._supported_time_dist_types + SHIFTED_TIME_DIST_TYPES
        assert 0 <= min_t < max_t <= 1.0, "RF min_t and max_t must be between 0 and 1"
        self._sigmas = torch.linspace(min_t, max_t, num_steps, dtype=self.t_precision)

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.num_steps * t

    @property
    def max_sigma(self) -> float:
        return self._max_t

    @property
    def sigmas(self) -> torch.Tensor:
        return self._sigmas

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def sigma_idx_to_t(self, sigma_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert sigma index back to timestep t.

        For Rectified Flow, sigma(t) = t, so this maps indices back to timesteps.

        Args:
            sigma_idx (torch.Tensor): Indices into the sigma array (long tensor)

        Returns:
            torch.Tensor: Corresponding timestep values
        """
        assert sigma_idx.dtype == torch.long
        return sigma_idx.to(self.t_precision) / self.num_steps

    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """Find timestep that matches the target sqrt(SNR) = alpha/sigma = (1-t)/t.

        Solving: sqrt_snr = (1-t)/t
        => sqrt_snr * t = 1 - t
        => t * (sqrt_snr + 1) = 1
        => t = 1 / (sqrt_snr + 1)

        Args:
            sqrt_snr_t: The square root of the SNR at time `t`, i.e., alpha_t / sigma_t.

        Returns:
            torch.Tensor: The timestep
        """
        original_dtype = sqrt_snr_t.dtype
        sqrt_snr_t = sqrt_snr_t.to(torch.float64)
        t = 1 / (sqrt_snr_t + 1)
        return t.to(original_dtype)

    def sample_t(
        self,
        n: int,
        time_dist_type: str = "logitnormal",
        train_p_mean: float = 0,
        train_p_std: float = 1.0,
        min_t: Optional[float] = 0.001,
        max_t: Optional[float] = 0.999,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert time_dist_type in self._supported_time_dist_types

        if min_t is not None and min_t < self.min_t:
            logger.warning(f"expected min_t >= {self.min_t}, got {min_t}")
        if max_t is not None and max_t > self.max_t:
            logger.warning(f"_expected max_t <= {self.max_t}, got {max_t}")
        min_t = max(min_t, self.min_t) if min_t is not None else self.min_t
        max_t = min(max_t, self.max_t) if max_t is not None else self.max_t

        # Use provided device or default to _sigmas properties
        target_device = device or self._sigmas.device

        if time_dist_type == "logitnormal":
            t = (
                torch.sigmoid(torch.randn(n, device=target_device, dtype=self.t_precision) * train_p_std + train_p_mean)
                * (max_t - min_t)
                + min_t
            )
        elif time_dist_type == "uniform":
            t = torch.rand(n, device=target_device, dtype=self.t_precision) * (max_t - min_t) + min_t
        elif time_dist_type == "shifted":
            shift = kwargs.get("shift", 5.0)
            assert shift >= 1, f"shift must be >= 1, got {shift}"
            t = torch.rand(n, device=target_device, dtype=self.t_precision) * (max_t - min_t) + min_t
            t = t * shift / (t * (shift - 1) + 1)
        elif time_dist_type == "shifted_logitnormal":
            # Logit-normal base density under the same shift map as "shifted"
            shift = kwargs.get("shift", 5.0)
            assert shift >= 1, f"shift must be >= 1, got {shift}"
            t = (
                torch.sigmoid(torch.randn(n, device=target_device, dtype=self.t_precision) * train_p_std + train_p_mean)
                * (max_t - min_t)
                + min_t
            )
            t = t * shift / (t * (shift - 1) + 1)
        else:
            raise ValueError(
                f"Unsupported time distribution type: {time_dist_type} in RFNoiseSchedule."
                f"Currently only supports logitnormal, uniform, shifted, and shifted_logitnormal."
            )
        return self.safe_clamp(t, min_t, max_t)

    def flow_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the x0-prediction from the flow-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The flow-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"

        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t_expanded = expand_like(t.to(torch.float64), xt)
        xt = xt.to(torch.float64)
        v = v.to(torch.float64)

        # Perform calculation in double precision
        x0 = xt - v.mul(t_expanded)

        # Convert back to original dtype
        return x0.to(original_dtype)

    def x0_to_flow(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the flow-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The flow-prediction.

        Note:
            Calculation is performed in double precision for better numerical accuracy.
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"

        # Store original dtype for final conversion
        original_dtype = xt.dtype

        # Convert to double precision for calculation
        t_expanded = expand_like(t.to(torch.float64), xt)
        xt = xt.to(torch.float64)
        x0 = x0.to(torch.float64)

        # Perform calculation in double precision
        flow = (xt - x0).div(self.non_zero_clamp(t_expanded))

        # Convert back to original dtype
        return flow.to(original_dtype)


class TrigNoiseSchedule(BaseNoiseSchedule):
    """
    The Trigonometric Flow noise schedule, typically used in Geometric/Consistency flows.
    x_t = cos(t) * x_0 + sin(t) * eps

    where t is in [0, pi/2].
    """

    def __init__(
        self,
        min_t: float = 0.0,
        max_t: float = torch.pi / 2,
        num_steps: int = 1000,
        **kwargs,
    ):
        super().__init__(min_t, max_t, num_steps, **kwargs)
        assert 0 <= min_t < max_t, "Trig min_t must be non-negative and less than max_t"
        self._sigmas = torch.sin(torch.linspace(min_t, max_t, num_steps, dtype=self.t_precision))

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        # In this parameterization, t acts directly as the angle.
        return t

    @property
    def max_sigma(self) -> float:
        return torch.sin(torch.tensor(self.max_t)).item()

    @property
    def sigmas(self) -> torch.Tensor:
        return self._sigmas

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(t)

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.sin(t)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t)

    def sigma_idx_to_t(self, sigma_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert sigma index back to timestep t.
        Since we linspace t to create sigmas, we can map indices linearly back to t.

        Args:
            sigma_idx (torch.Tensor): Indices into the sigma array (long tensor)

        Returns:
            torch.Tensor: Corresponding timestep values
        """
        assert sigma_idx.dtype == torch.long
        t = sigma_idx.to(self.t_precision) / (self.num_steps - 1) * (self.max_t - self.min_t) + self.min_t
        return t

    def sqrt_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the square root of the Signal-to-Noise Ratio (SNR) at time `t`.
        sqrt_snr(t) = cos(t) / sin(t) = cot(t)
        """
        assert self.is_t_valid(t)
        t = t.to(torch.float64)
        # We clamp tan(t) to avoid dividing by zero at t=0
        tan_t = self.non_zero_clamp(torch.tan(t))
        return 1.0 / tan_t

    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """
        Find timestep that matches the target square root of SNR value.
        sqrt_snr = alpha/sigma = cos(t)/sin(t) = cot(t).
        t = arccot(SNR) = arctan(1/SNR).

        Args:
            sqrt_snr_t: The square root of the SNR at time `t`.

        Returns:
            torch.Tensor: The timestep
        """
        original_dtype = sqrt_snr_t.dtype
        sqrt_snr_t = sqrt_snr_t.to(torch.float64)
        # Use atan2(1, sqrt_snr_t) which is a robust equivalent of arccot(sqrt_snr_t)
        t = torch.atan2(torch.ones_like(sqrt_snr_t), sqrt_snr_t)
        return t.to(original_dtype)

    def sample_t(
        self,
        n: int,
        time_dist_type: str = "uniform",
        train_p_mean: float = 0,
        train_p_std: float = 1.0,
        min_t: Optional[float] = 0.0,
        max_t: Optional[float] = torch.pi / 2,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert time_dist_type in self._supported_time_dist_types

        if min_t is not None and min_t < self.min_t:
            logger.warning(f"expected min_t >= {self.min_t}, got {min_t}")
        if max_t is not None and max_t > self.max_t:
            logger.warning(f"_expected max_t <= {self.max_t}, got {max_t}")
        min_t = max(min_t, self.min_t) if min_t is not None else self.min_t
        max_t = min(max_t, self.max_t) if max_t is not None else self.max_t

        target_device = device or self._sigmas.device

        if time_dist_type == "logitnormal":
            # Map sigmoid output [0, 1] to [min_t, max_t]
            norm_t = torch.sigmoid(
                torch.randn(n, device=target_device, dtype=self.t_precision) * train_p_std + train_p_mean
            )
            t = norm_t * (max_t - min_t) + min_t
        elif time_dist_type == "uniform":
            t = torch.rand(n, device=target_device, dtype=self.t_precision) * (max_t - min_t) + min_t
        else:
            raise ValueError(
                f"Unsupported time distribution type: {time_dist_type} in TrigNoiseSchedule."
                f"Currently only supports logitnormal and uniform."
            )
        return self.safe_clamp(t, min_t, max_t)

    def flow_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute x0 from flow prediction v.
        v_t = -sin(t)x0 + cos(t)eps
        x_t = cos(t)x0 + sin(t)eps

        Algebraic rearrangement yields:
        x0 = cos(t)x_t - sin(t)v
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        original_dtype = xt.dtype

        t = t.to(torch.float64)
        xt = xt.to(torch.float64)
        v = v.to(torch.float64)

        cos_t = expand_like(torch.cos(t), xt)
        sin_t = expand_like(torch.sin(t), xt)

        x0 = xt * cos_t - v * sin_t
        return x0.to(original_dtype)

    def x0_to_flow(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute flow prediction v from x0.
        v = (cos(t)x_t - x_0) / sin(t)
        """
        assert self.is_t_valid(t), f"t must be in [{self.min_t}, {self.max_t}], but got {t}"
        original_dtype = xt.dtype

        t = t.to(torch.float64)
        xt = xt.to(torch.float64)
        x0 = x0.to(torch.float64)

        cos_t = expand_like(torch.cos(t), xt)
        sin_t = expand_like(torch.sin(t), xt)

        flow = (xt * cos_t - x0) / self.non_zero_clamp(sin_t)
        return flow.to(original_dtype)


# Global registry of noise schedules for easy access by string name
NOISE_SCHEDULES = {
    "edm": EDMNoiseSchedule,
    "alphas": AlphasNoiseSchedule,
    "sd": SDNoiseSchedule,
    "sdxl": SDXLNoiseSchedule,
    "cogvideox": CogVideoXNoiseSchedule,
    "rf": RFNoiseSchedule,
    "rectified_flow": RFNoiseSchedule,  # Alias for RF
    "trig": TrigNoiseSchedule,
}


def get_noise_schedule(name: str, **kwargs):
    """
    Get a noise schedule class by name.

    Args:
        name (str): Name of the noise schedule. Available options:
            - "edm": EDMNoiseSchedule (Karras et al. 2022)
            - "alphas": AlphasNoiseSchedule (base class)
            - "sd": SDNoiseSchedule (Stable Diffusion v1.5)
            - "sdxl": SDXLNoiseSchedule (Stable Diffusion XL)
            - "cogvideox": CogVideoXNoiseSchedule (CogVideoX)
            - "rf" or "rectified_flow": RFNoiseSchedule (Rectified Flow)
        **kwargs: Arguments to pass to the noise schedule constructor.

    Returns:
        BaseNoiseSchedule: Instantiated noise schedule.

    Raises:
        KeyError: If the noise schedule name is not found.

    Example:
        >>> schedule = get_noise_schedule("edm", min_t=0.002, max_t=80.0)
        >>> schedule = get_noise_schedule("sd")
    """
    if name not in NOISE_SCHEDULES:
        available = ", ".join(sorted(NOISE_SCHEDULES.keys()))
        raise KeyError(f"Unknown noise schedule '{name}'. Available schedules: {available}")

    schedule_class = NOISE_SCHEDULES[name]
    return schedule_class(**kwargs)


def list_noise_schedules():
    """
    List all available noise schedule names.

    Returns:
        list: List of available noise schedule names.
    """
    return sorted(NOISE_SCHEDULES.keys())
