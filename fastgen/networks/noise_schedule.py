# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""factory for different noise schedules."""

from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import attrs
import torch
import numpy as np
from diffusers import DDIMScheduler, CogVideoXDPMScheduler

from fastgen.networks.module import FastGenModule
from fastgen.networks.time_distributions import (
    TimeDistribution,
    checked_bounds,
    get_time_distribution,
    larger_bound,
    polynomial_spacing,
    smaller_bound,
    time_shift,
)
import fastgen.utils.logging_utils as logger
import fastgen.utils.global_vars as global_vars
from fastgen.utils.tensor import (
    assert_tensor,
    expand_like,
    high_precision_dtype,
    precision_to_dtype,
    to_high_precision,
)
from fastgen.utils.env import hf_load
from fastgen.utils.tree_utils import OutputDtype, assert_same, cast_like, index, per_modality


NET_PRED_TYPES = {"x0", "eps", "v", "flow"}
# The samplers of ``BaseNoiseSchedule.step``, where ``float`` stands for an eta.
SAMPLE_TYPES = ("ode", "sde", "ddpm", float)


def check_sample_type(sample_type: Any, supported: Tuple[Any, ...] = SAMPLE_TYPES, name: str = "sample_type") -> None:
    """Raise unless *sample_type* is a *supported* sampler name or, where ``float`` is supported, an eta ``>= 0``."""
    if isinstance(sample_type, str):
        is_supported = sample_type in supported
    else:
        is_eta = isinstance(sample_type, (int, float)) and not isinstance(sample_type, bool)
        is_supported = float in supported and is_eta and sample_type >= 0
    if not is_supported:
        choices = ", ".join("a float eta >= 0" if choice is float else repr(choice) for choice in supported)
        raise ValueError(f"{name} must be {'one of ' if len(supported) > 1 else ''}{choices}; got {sample_type!r}")


@attrs.define
class StudentGridConfig:
    """Declarative configuration for a distilled student's runtime grid.

    ``grid_type="native"`` asks the noise schedule to construct its own grid.
    Without ``grid_kwargs["base_steps"]``, it constructs the requested number of
    intervals directly. With ``base_steps``, it first constructs a persistent
    base grid with that many intervals and selects runtime student-grid timesteps
    from it. The native grid is the schedule's ``grid()``, polynomially spaced
    on EDM and uniform elsewhere; ``grid_kwargs["shift"]`` applies that
    exponential time shift to it.

    ``grid_type="explicit"`` treats ``grid_kwargs["timesteps"]`` as the
    persistent base grid and likewise selects runtime timesteps from it. Optional
    ``grid_kwargs["indices"]`` can override the balanced base-grid indices for
    selected interval counts.

    Only anchored configurations give
    ``BaseNoiseSchedule.student_base_steps`` a non-``None`` value and support
    mapping requested student-grid step indices back to persistent base-grid indices.

    ``steps`` is the default runtime student NFE. ``grid_type`` and
    ``grid_kwargs`` define how the scheduler constructs its student grid.
    """

    steps: int = 1
    grid_type: str = "native"
    grid_kwargs: dict[str, Any] = attrs.field(factory=dict)


def causal_chunk_frames(seq_len: int, chunk_size: int) -> list[int]:
    """Frames per causal chunk; the first chunk takes any remainder frames, as in the causal attention masks."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    num_chunks, remainder = divmod(seq_len, chunk_size)
    if num_chunks == 0:
        return [seq_len]
    return [chunk_size + remainder] + [chunk_size] * (num_chunks - 1)


def sqrt_snr_to_alpha_sigma(sqrt_snr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized coefficients with ``alpha/sigma = sqrt_snr``, in the input's dtype and device.

    The input must be nonnegative; zero and infinity give the exact pure-noise and clean limits.
    Using min(r, 1/r) avoids overflow and cancellation in values and forward/reverse derivatives.
    Callers own precision promotion, validation and any schedule-specific SNR bounds.
    """
    reciprocal = torch.where(sqrt_snr > 1, sqrt_snr, 1).reciprocal()
    small = torch.where(sqrt_snr <= 1, sqrt_snr, reciprocal)
    norm = (1 + small.square()).sqrt()
    return torch.where(sqrt_snr <= 1, small, 1) / norm, torch.where(sqrt_snr <= 1, 1, small) / norm


def _assert_unit_alpha_sigma(
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    tolerance: float = 1e-3,
) -> None:
    """Assert the normalization required by v-prediction."""
    assert_tensor(
        torch.isclose(
            alpha**2 + sigma**2,
            torch.ones_like(alpha),
            rtol=tolerance,
            atol=tolerance,
        ).all(),
        "v-prediction only supports the case of alpha_t^2+sigma_t^2 = 1.",
    )


def _nextafter(value: float, toward: float, dtype: torch.dtype) -> float:
    """*value* rounded to *dtype* and moved one ulp of *dtype* toward *toward*, as a Python float.

    A *dtype* that ``torch.nextafter`` does not handle, such as an integer one, is replaced by float32.
    Computed on the host, also under a default-device context such as meta initialization: a CUDA scalar
    built from a Python number is a blocking copy, so a time check that compares against it would
    synchronize the device, while comparing against a float does not.
    """
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        dtype = torch.float32
    value, toward = (torch.tensor(x, dtype=dtype, device="cpu") for x in (value, toward))
    return torch.nextafter(value, toward).item()


class BaseNoiseSchedule(FastGenModule):
    """Abstract base noise schedule class.

    This class defines a noise schedule used for training/sampling from
    diffusion models. Its unchecked ``_`` implementations compute and return
    at least ``HIGH_PRECISION``, and user-facing methods round results once to the
    dtype of their first tensor argument unless an ``output_dtype``
    (predictions and targets) or ``dtype`` says otherwise; times they draw or
    space are in ``t_precision`` by default. Implementations on per-sample
    values (times, square-root SNRs) expect their caller to cast them; those
    on states cast their own inputs.

    We follow the general format to define the forward process:

        x_t = alpha(t) x_0 + sigma(t) eps

    """

    def __init__(
        self,
        min_t: float,
        max_t: float,
        clamp_min: float = 1e-6,
        t_precision: str | torch.dtype | None = None,
        student_grid_cfg: Optional[StudentGridConfig] = None,
    ):
        super().__init__()
        if min_t < 0:
            raise ValueError(f"min_t must be non-negative, got {min_t}")
        if max_t <= min_t:
            raise ValueError(f"max_t must be greater than min_t, got min_t={min_t} and max_t={max_t}")
        if clamp_min <= 0:
            raise ValueError(f"clamp_min must be positive, got {clamp_min}")
        self._min_t = min_t
        self._max_t = max_t
        self.clamp_min = clamp_min
        self.t_precision = precision_to_dtype(t_precision, default=high_precision_dtype())
        if student_grid_cfg is not None and not isinstance(student_grid_cfg, StudentGridConfig):
            raise TypeError(f"student_grid_cfg must be a StudentGridConfig, got {type(student_grid_cfg)!r}")
        self.student_grid_cfg = StudentGridConfig() if student_grid_cfg is None else student_grid_cfg

    def reset_parameters(self) -> None:
        """Rebuild schedule tables on their current device and dtype after materialization; schedules without
        tables have nothing to rebuild."""

    @property
    def min_t(self) -> float:
        """The minimum time step supported by this schedule.

        Returns:
            float: Minimum time step.
        """
        return self._min_t

    @property
    def max_t(self) -> float:
        """The maximum time step supported by this schedule.

        Returns:
            float: Maximum time step.
        """
        return self._max_t

    @property
    def schedules(self) -> "BaseNoiseSchedule":
        """Per-modality dispatch view (used as ``ns.schedules`` arg to
        ``tree_utils.per_modality``).

        - Base returns ``self``: ``per_modality`` treats it as a non-dict
          scalar and broadcasts it to every per-key call (i.e., the
          unimodal scheduler is applied to every modality, which is the
          correct semantics when data is unimodal Tensor — the dispatch
          short-circuits on the Tensor primary).
        - ``CompositeNoiseSchedule`` overrides this to return its
          inner ``ModuleDict`` (``self.subschedules``), which is in
          ``tree_utils.DICT_TYPES`` and gets indexed per-key.

        This lets callers write ``per_modality(fn, ..., ns.schedules,
        ...)`` uniformly without an ``isinstance`` check.
        """
        return self

    def latents(self, noise: torch.Tensor, t_init: torch.Tensor) -> torch.Tensor:
        """Scale the noise by the sigma at the initial time step.

        Args:
            noise: Noise tensor.
            t_init: Initial time step, i.e. the first time of the caller's grid. It is required because sampler
                grids do not start at ``max_t`` in general (a shifted grid's first sigma or a student grid's
                first node), so no default is assumed.

        Returns:
            torch.Tensor: Scaled noise tensor.
        """
        # we cast to high precision to be consistent with the `forward_process` method
        self._assert_t_valid(t_init, "t_init")
        original_dtype = noise.dtype
        noise = to_high_precision(noise)
        sigma = expand_like(self._sigma(to_high_precision(t_init)).to(noise.dtype), noise)
        return (noise * sigma).to(original_dtype)

    @staticmethod
    def safe_clamp(
        t: torch.Tensor,
        min: float | torch.Tensor | None = None,
        max: float | torch.Tensor | None = None,
        warn: bool = False,
        keep_grad: bool = False,
    ) -> torch.Tensor:
        """
        Clamp the timestep t such that min <= t <= max holds in floating point precision.

        Args:
            t (torch.Tensor): The timestep.
            min (float | torch.Tensor | None): The minimum time step, a number or a tensor broadcastable to `t`.
            max (float | torch.Tensor | None): The maximum time step, a number or a tensor broadcastable to `t`.
            warn (bool): Log a warning when the clamp actually moves values. Sampling grids are meant to
                lie inside the schedule range already; a clamp there silently changes what the network sees.
            keep_grad (bool): Keep the derivative of `t` where the clamp moves it, for clamps that only absorb
                rounding. For finite `t`, the clamped value is unchanged; gradients to the bounds are stopped.

        Returns:
            torch.Tensor: The clamped timestep.
        """
        if t.dtype == torch.float64:
            # One torch.clamp call rejects a tensor bound next to a number bound, so apply them in turn.
            clamped = t if min is None else torch.clamp(t, min=min)
            clamped = clamped if max is None else torch.clamp(clamped, max=max)
        else:
            min_bound = None
            if min is not None:
                min_bound = torch.as_tensor(min, dtype=t.dtype, device=t.device)
                min_bound = torch.where(
                    min_bound.to(torch.float64) < min,
                    torch.nextafter(min_bound, torch.full_like(min_bound, torch.inf)),
                    min_bound,
                )

            max_bound = None
            if max is not None:
                max_bound = torch.as_tensor(max, dtype=t.dtype, device=t.device)
                max_bound = torch.where(
                    max_bound.to(torch.float64) > max,
                    torch.nextafter(max_bound, torch.full_like(max_bound, -torch.inf)),
                    max_bound,
                )
            clamped = torch.clamp(t, min=min_bound, max=max_bound)
        if keep_grad:
            clamped = clamped.detach() + (t - t.detach())

        if warn and not torch.equal(clamped, t):
            moved = clamped != t
            logger.warning(
                f"safe_clamp moved {int(moved.sum())} of {t.numel()} timesteps into [{min}, {max}] "
                f"(e.g. {t[moved][0].item():.6g} -> {clamped[moved][0].item():.6g})."
            )
        return clamped

    def non_zero_clamp(self, tensor: torch.Tensor) -> torch.Tensor:
        """Clamp the tensor such that it is non-zero."""
        return torch.where(
            tensor >= 0,
            tensor.clamp(min=self.clamp_min),
            tensor.clamp(max=-self.clamp_min),
        )

    def rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        """Rescale the timestep t to the range that is actually used by the model.

        Returns:
            torch.Tensor: rescaled timestep.
        """
        self._assert_t_valid(t)
        return self._rescale_t(to_high_precision(t)).to(t.dtype)

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
        self._assert_t_valid(t)
        return self._alpha(to_high_precision(t)).to(t.dtype)

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
        self._assert_t_valid(t)
        return self._sigma(to_high_precision(t)).to(t.dtype)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both ``alpha(t)`` and ``sigma(t)``, with one range check and, on table schedules, one table lookup.

        Args:
            t (torch.Tensor): The timestep.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data and noise scales at time `t`, in the dtype of `t`.
        """
        self._assert_t_valid(t)
        alpha, sigma = self._alpha_sigma(to_high_precision(t))
        return alpha.to(t.dtype), sigma.to(t.dtype)

    def _alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unchecked coefficient pair; table schedules override this to share the lookup and interpolation."""
        return self._alpha(t), self._sigma(t)

    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the time derivative of alpha(t) at time `t`, as defined within the framework:

            x_t = alpha(t) x_0 + sigma(t) eps

        Args:
            t (torch.Tensor): The timestep.

        Returns:
            torch.Tensor: The time derivative of alpha(t) at time `t`.
        """
        self._assert_t_valid(t)
        return self._alpha_prime(to_high_precision(t)).to(t.dtype)

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
        self._assert_t_valid(t)
        return self._sigma_prime(to_high_precision(t)).to(t.dtype)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_t(
        self,
        n: int,
        time_dist_type: str = "uniform",
        train_p_mean: float = 0.0,
        train_p_std: float = 1.0,
        min_t: Optional[float | torch.Tensor] = None,
        max_t: Optional[float | torch.Tensor] = None,
        shift: float = 5.0,
        log_t_df: float = 0.01,
        rho: float = 7.0,
        device: Optional[torch.device] = None,
        window: Optional[Tuple[float | torch.Tensor, float | torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Draw *n* training times from *time_dist_type* on the bounds.

        The bounds apply to the sampled time itself: draws on ``[0, 1]`` (uniform, logit-normal, and their shifted
        forms) are mapped onto ``[min_t, max_t]``, and distributions over time (lognormal, log-t and polynomial) are
        truncated to it. Clipped to the schedule range, every interval must keep ``min_t < max_t``. A ``window``
        conditions the distribution the bounds define on a part of their range, alike for every distribution (see
        ``fastgen.networks.time_distributions``).

        Args:
            n (int): Batch size.
            time_dist_type (str): The type of time t distribution to sample from, any of ``get_time_distribution``'s;
                each needs only the bounds, so every schedule supports them all.
            train_p_mean: The mu of ``logitnormal``, ``shifted_logitnormal``, ``lognormal`` and ``log_t``.
            train_p_std: Their sigma.
            min_t: Minimum time step, a float or one per sample (a tensor of shape [n]). Defaults to None, the
                schedule's own ``min_t``.
            max_t: Maximum time step, a float or one per sample (a tensor of shape [n]). Defaults to None, the
                schedule's own ``max_t``.
            shift: The exponential time shift of ``shifted`` and ``shifted_logitnormal``.
            log_t_df: The degrees of freedom of ``log_t``.
            rho: The exponent of ``polynomial``, whose draws' ``1/rho`` powers are uniform between those of the bounds.
            device: Device for the output tensor. Defaults to None, the schedule's ``device``, or, before
                ``prepare_module`` records one, that of tensor bounds and window ends, else torch's default device.
            window: The range ``(start, end)`` to condition the draws on, each end a float or one per sample,
                intersected with the bounds. Defaults to None, the whole distribution.

        Returns:
            torch.Tensor: Tensor containing time steps, of shape [n], drawn in float64, the precision of the bounds,
            then rounded once to ``t_precision`` and clamped inward to the bounds.
        """
        distribution = self._time_distribution(
            time_dist_type,
            train_p_mean=train_p_mean,
            train_p_std=train_p_std,
            shift=shift,
            log_t_df=log_t_df,
            rho=rho,
        )
        min_t, max_t, window = self._resolve_sample_t_bounds((n,), min_t, max_t, window)
        device = self.device if device is None else device
        t = distribution.sample(n, min_t, max_t, window=window, device=device, dtype=self.t_precision)
        t = self.safe_clamp(t, *((min_t, max_t) if window is None else window))
        assert_tensor(torch.isfinite(t).all(), "sample_t drew non-finite times")
        return t

    def sample_t_icdf(
        self,
        p: torch.Tensor,
        time_dist_type: str = "uniform",
        train_p_mean: float = 0.0,
        train_p_std: float = 1.0,
        min_t: Optional[float | torch.Tensor] = None,
        max_t: Optional[float | torch.Tensor] = None,
        shift: float = 5.0,
        log_t_df: float = 0.01,
        rho: float = 7.0,
    ) -> torch.Tensor:
        """The times at which the distribution that ``sample_t`` draws from, with the same arguments and the same
        bounds clipped to the schedule range, reaches the probabilities *p*; in ``t_precision`` on the device of *p*,
        rounded and clamped like the draws.
        """
        distribution = self._time_distribution(
            time_dist_type, train_p_mean=train_p_mean, train_p_std=train_p_std, shift=shift, log_t_df=log_t_df, rho=rho
        )
        min_t, max_t, _ = self._resolve_sample_t_bounds(p.shape, min_t, max_t, None)
        t = distribution.icdf(p, min_t, max_t, dtype=self.t_precision)
        return self.safe_clamp(t, min_t, max_t)

    def _resolve_sample_t_bounds(
        self,
        shape: Tuple[int, ...],
        min_t: Optional[float | torch.Tensor],
        max_t: Optional[float | torch.Tensor],
        window: Optional[Tuple[float | torch.Tensor, float | torch.Tensor]],
    ) -> Tuple[float | torch.Tensor, float | torch.Tensor, Optional[Tuple[float | torch.Tensor, float | torch.Tensor]]]:
        """Clip the ``sample_t`` bounds to the schedule range, defaulting omitted ones to it, and intersect a window with
        the clipped bounds. The time distribution rejects an interval this leaves empty.

        Scalar bounds outside the range are clipped with a warning, per-sample bounds (tensors of *shape*, on the
        sampling device) silently, since detecting a clip would need a host sync. A window, whose ends may be
        per-sample as well, is intersected silently. Tensor bounds are returned in float64, like number bounds.
        """
        # In float64, tensor bounds clip inside the schedule range (float32 0.999 exceeds 0.999) and keep their values
        # in a coarse time precision, so that the clamp of the draws rounds them inward, as it does number bounds.
        min_t, max_t, start, end = checked_bounds(shape, min_t, max_t, window or (None, None), dtype=torch.float64)
        if min_t is not None and not torch.is_tensor(min_t) and min_t < self.min_t:
            logger.warning(f"expected min_t >= {self.min_t}, got {min_t}")
        if max_t is not None and not torch.is_tensor(max_t) and max_t > self.max_t:
            logger.warning(f"expected max_t <= {self.max_t}, got {max_t}")

        min_t = self.min_t if min_t is None else larger_bound(min_t, self.min_t)
        max_t = self.max_t if max_t is None else smaller_bound(max_t, self.max_t)
        if window is not None:
            window = larger_bound(start, min_t), smaller_bound(end, max_t)
        return min_t, max_t, window

    def _time_distribution(
        self, time_dist_type: str, train_p_mean: float, train_p_std: float, shift: float, log_t_df: float, rho: float
    ) -> TimeDistribution:
        """The ``sample_t`` distribution named *time_dist_type*, which may be any of ``get_time_distribution``'s."""
        return get_time_distribution(
            time_dist_type,
            train_p_mean=train_p_mean,
            train_p_std=train_p_std,
            shift=shift,
            log_t_df=log_t_df,
            clamp_min=self.clamp_min,
            rho=rho,
        )

    @staticmethod
    def _validate_num_steps(num_steps: Any, name: str = "num_steps") -> int:
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
            raise ValueError(f"{name} must be a positive int, got {num_steps!r}")
        return num_steps

    def _student_grid_config(self) -> tuple[str, Mapping[str, Any], Optional[int]]:
        """Validate the student-grid policy and return its base-grid interval count."""
        grid_type = self.student_grid_cfg.grid_type
        grid_kwargs = self.student_grid_cfg.grid_kwargs
        if grid_type not in {"native", "explicit"}:
            raise ValueError(f"Unsupported student grid_type {grid_type!r}; expected 'native' or 'explicit'")
        if not isinstance(grid_kwargs, Mapping):
            raise TypeError(f"student_grid_cfg.grid_kwargs must be a mapping, got {type(grid_kwargs)!r}")

        supported = {
            "native": {"base_steps", "shift", "indices"},
            "explicit": {"timesteps", "indices"},
        }[grid_type]
        unknown = set(grid_kwargs) - supported
        if unknown:
            raise ValueError(f"Unsupported {grid_type} student grid kwargs: {sorted(unknown)}")

        if grid_type == "native":
            base_steps = grid_kwargs.get("base_steps")
            if base_steps is None:
                if "indices" in grid_kwargs:
                    raise ValueError("grid_kwargs.indices requires grid_kwargs.base_steps for a native student grid")
            else:
                base_steps = self._validate_num_steps(base_steps, "grid_kwargs.base_steps")
        else:
            if "timesteps" not in grid_kwargs:
                raise ValueError("An explicit student grid requires grid_kwargs['timesteps']")
            timesteps = grid_kwargs["timesteps"]
            try:
                base_steps = len(timesteps) - 1
            except TypeError as exc:
                raise TypeError("grid_kwargs.timesteps must be a one-dimensional sequence of numbers") from exc
            base_steps = self._validate_num_steps(base_steps, "explicit base-grid interval count")

        configured_indices = grid_kwargs.get("indices")
        if configured_indices is not None and not isinstance(configured_indices, Mapping):
            raise TypeError("grid_kwargs.indices must map a target step count to base-grid indices")
        return grid_type, grid_kwargs, base_steps

    @property
    def default_student_steps(self) -> int:
        """Default student NFE, honoring the process-wide override."""
        override = getattr(global_vars, "STUDENT_STEPS", None)
        configured = self.student_grid_cfg.steps if override is None else override
        return self._validate_num_steps(configured, "student_grid_cfg.steps")

    @property
    def student_base_steps(self) -> Optional[int]:
        """Number of intervals in the configured base grid, or ``None`` when unanchored."""
        return self._student_grid_config()[2]

    def _validate_grid(self, grid: torch.Tensor, num_steps: int) -> None:
        """Validate one descending time grid; a pure check that never converts.

        The grid ends at the clean end 0, which may lie below ``min_t``; every
        earlier time is a network evaluation and must lie in
        ``[min_t, max_t]``, each bound tolerated to one ulp of the grid's own
        (or nearest checkable) dtype so input-precision endpoints are accepted.
        """
        if grid.shape != (num_steps + 1,):
            raise ValueError(f"Time grid must have shape ({num_steps + 1},), got {tuple(grid.shape)}")
        assert_tensor(grid[-1] == 0, "Time grid must end at zero", error_type=ValueError)
        assert_tensor(
            torch.isfinite(grid).all() & (grid[:-1] > grid[1:]).all(),
            "Time grid must be finite and strictly decreasing",
            error_type=ValueError,
        )
        lower = _nextafter(self.min_t, -torch.inf, grid.dtype)
        upper = _nextafter(self.max_t, torch.inf, grid.dtype)
        assert_tensor(
            (grid[0] <= upper) & (grid[-2] >= lower),
            f"Time grid must lie in [{self.min_t}, {self.max_t}] before its clean end 0",
            error_type=ValueError,
        )

    def clamp_grid(self, grid: torch.Tensor, warn: bool = False) -> torch.Tensor:
        """A copy of *grid* with its network times, all but the clean end 0, clamped to ``[min_t, max_t]``.

        *warn* logs when the clamp moves a time, as in ``safe_clamp``.
        """
        return torch.cat([self.safe_clamp(grid[:-1], min=self.min_t, max=self.max_t, warn=warn), grid[-1:]])

    def _grid(self, num_steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Unchecked high-precision implementation of ``grid``: uniform from ``max_t`` to 0 unless a schedule spaces
        its times, as EDM does. Native student grids shift it before rounding."""
        device = self.device if device is None else device
        dtype = high_precision_dtype(self.t_precision)
        return torch.linspace(self.max_t, 0, num_steps + 1, device=device, dtype=dtype)

    def _canonicalize_grid(
        self, grid: torch.Tensor, num_steps: int, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """*grid* clamped in its own precision and converted to *dtype*, with ``clamp_grid`` after each cast.
        Validated before the first clamp, so that the clamps only absorb rounding, and after the last; the first
        clamp keeps the times in range, so a finer *dtype* converts them exactly."""
        self._validate_grid(grid, num_steps)
        grid = self.clamp_grid(grid)
        if dtype is not None and dtype != grid.dtype:
            grid = self.clamp_grid(grid.to(dtype))
        self._validate_grid(grid, num_steps)
        return grid

    def grid(
        self,
        num_steps: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return this noise schedule's grid from ``max_t`` to 0, independent of the student grid policy: uniform
        unless the schedule spaces its times, as EDM does. Its times are computed in high precision and rounded once
        to the time precision; *dtype* converts those times."""
        num_steps = self._validate_num_steps(num_steps)
        return self._canonicalize_grid(self._grid(num_steps, device).to(self.t_precision), num_steps, dtype)

    def _student_grid_indices(
        self,
        base_steps: int,
        target_steps: int,
        configured_indices: Optional[Mapping[Any, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Resolve base-grid indices for all student-grid step indices, including both endpoints."""
        if target_steps > base_steps:
            raise ValueError(f"Cannot select {target_steps} steps from a {base_steps}-step student base grid")

        base_indices = None
        if configured_indices is not None:
            base_indices = configured_indices.get(target_steps, configured_indices.get(str(target_steps)))

        if base_indices is None:
            return torch.arange(target_steps + 1, device=device) * base_steps // target_steps

        if not torch.is_tensor(base_indices) and not isinstance(base_indices, np.ndarray):
            if isinstance(base_indices, (str, bytes)):
                raise TypeError("Student-grid base indices must be a one-dimensional sequence of integers")
            base_indices = list(base_indices)
        base_indices = torch.as_tensor(base_indices, device=device)
        if base_indices.dtype == torch.bool or base_indices.is_floating_point() or base_indices.is_complex():
            raise TypeError(f"Student-grid base indices must have an integer dtype, got {base_indices.dtype}")
        base_indices = base_indices.to(torch.long)
        if base_indices.shape != (target_steps + 1,):
            raise ValueError(
                f"Student-grid base indices for {target_steps} target steps must have shape ({target_steps + 1},), "
                f"got {tuple(base_indices.shape)}"
            )
        assert_tensor(
            (base_indices[0] == 0) & (base_indices[-1] == base_steps),
            f"Student-grid base indices must start at 0 and end at {base_steps}",
            error_type=ValueError,
        )
        assert_tensor(
            (base_indices[:-1] < base_indices[1:]).all(),
            "Student-grid base indices must be strictly increasing",
            error_type=ValueError,
        )
        return base_indices

    def student_grid_indices(
        self,
        target_steps: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return base-grid indices for all student-grid step indices, including both endpoints.

        The base-grid interval count comes from native ``grid_kwargs.base_steps``
        or the length of an explicit timestep grid. Unanchored native grids have
        no persistent index coordinate and therefore cannot use this API.
        ``target_steps`` is explicit and never resolved from the configured
        default or its global override.
        """
        target_steps = self._validate_num_steps(target_steps, "target_steps")

        _, grid_kwargs, base_steps = self._student_grid_config()
        if base_steps is None:
            raise ValueError(
                "student_grid_indices requires an anchored student grid; "
                "set grid_kwargs.base_steps for a native grid or configure an explicit grid"
            )
        configured_indices = grid_kwargs.get("indices")
        return self._student_grid_indices(
            base_steps=base_steps,
            target_steps=target_steps,
            configured_indices=configured_indices,
            device=device,
        )

    def student_grid(
        self,
        num_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return the student grid at ``num_steps`` intervals, defaulting to the configured NFE."""
        if num_steps is None:
            num_steps = self.default_student_steps
        num_steps = self._validate_num_steps(num_steps)
        grid_type, grid_kwargs, base_steps = self._student_grid_config()
        grid_steps = num_steps if base_steps is None else base_steps

        if grid_type == "native":
            student_grid = self._grid(grid_steps, device)
            shift = grid_kwargs.get("shift")
            if shift is not None:
                student_grid = self.shift_t(student_grid, shift=float(shift))
            student_grid = self._canonicalize_grid(student_grid.to(self.t_precision), grid_steps, dtype)
        else:
            timesteps = grid_kwargs["timesteps"]
            # Explicit grids land on the schedule's device by default, like native grids.
            device = self.device if device is None else device
            # A floating tensor or array input keeps its own precision so its endpoints
            # validate and clamp within their native ulp (float32 ``0.999`` is
            # a little larger than the FP64 value 0.999); any other input is
            # materialized in scheduler precision (Python floats are FP64).
            if torch.is_tensor(timesteps) or isinstance(timesteps, np.ndarray):
                student_grid = torch.as_tensor(timesteps, device=device)
                if not student_grid.is_floating_point():
                    student_grid = student_grid.to(self.t_precision)
            else:
                student_grid = torch.as_tensor(list(timesteps), device=device, dtype=self.t_precision)
            # Canonicalize in input precision before any requested promotion.
            # This keeps the physical grid identical between callers that
            # preserve an explicit tensor's dtype (for example generator
            # initialization) and PDD callers that request scheduler precision.
            student_grid = self._canonicalize_grid(student_grid, grid_steps, dtype)

        if base_steps is not None and num_steps != base_steps:
            base_indices = self._student_grid_indices(
                base_steps=base_steps,
                target_steps=num_steps,
                configured_indices=grid_kwargs.get("indices"),
                device=student_grid.device,
            )
            student_grid = student_grid[base_indices]
        return student_grid

    def shift_t(self, t: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
        """Place *t* under the exponential shift over ``[0, max_t]``, this schedule's grid range."""
        self._assert_t_valid(t, min_t=0.0)
        shifted = time_shift(to_high_precision(t), shift, max_t=self.max_t).to(t.dtype)
        return self.safe_clamp(shifted, max=self.max_t, keep_grad=True)

    def sample_student_timesteps(
        self,
        n: int,
        num_steps: Optional[int] = None,
        return_indices: bool = False,
        include_last_step: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Draw timesteps by student-grid step index, defaulting to the configured NFE.

        ``include_last_step`` also draws the clean terminal step index (``t = 0``);
        by default only noisy step indices are sampled. ``return_indices`` also
        returns the sampled step indices in this runtime student grid.
        """
        student_grid = self.student_grid(num_steps, device=device, dtype=dtype)
        device = student_grid.device

        num_step_indices = len(student_grid) if include_last_step else len(student_grid) - 1
        step_indices = torch.randint(0, num_step_indices, (n,), device=device)
        sampled_timesteps = student_grid[step_indices]
        if return_indices:
            return sampled_timesteps, step_indices
        return sampled_timesteps

    def sample_student_timesteps_inhom(
        self,
        n: int,
        seq_len: int,
        chunk_size: int,
        num_steps: Optional[int] = None,
        include_last_step: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample block-wise independent t variables for diffusion forcing in distillation.

        Each causal chunk draws one step index; the first chunk also takes any remainder frames.
        When ``num_steps`` is omitted, use ``default_student_steps``.
        ``include_last_step`` also draws the clean terminal step index (``t = 0``),
        the default here so chunks can act as clean context.

        Args:
            n: batch size
            seq_len: sequence length
            chunk_size: chunk size
            device: torch device

        Return:
            Tuple[torch.Tensor, torch.Tensor]: The first tensor contains time steps and
                the second contains their student-grid step indices, both of shape
                [n, seq_len].
        """
        student_grid = self.student_grid(num_steps, device=device, dtype=dtype)
        device = student_grid.device

        chunk_frames = causal_chunk_frames(seq_len, chunk_size)
        num_step_indices = len(student_grid) if include_last_step else len(student_grid) - 1
        chunk_step_indices = torch.randint(0, num_step_indices, (n, len(chunk_frames)), device=device)

        # A non-blocking copy and output_size keep the expansion free of host syncs.
        repeats = torch.tensor(chunk_frames).to(device, non_blocking=True)
        step_indices = chunk_step_indices.repeat_interleave(repeats, dim=1, output_size=seq_len)
        return student_grid[step_indices], step_indices

    def step(
        self,
        xt: torch.Tensor,
        model_output: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        pred_type: str = "x0",
        sample_type: str | float = "ode",
        clean_end: bool = True,
    ) -> torch.Tensor:
        """Sampler step from ``t`` to ``t_next`` for a prediction made at ``(xt, t)``.

        The step holds the x0 and eps the prediction implies and renoises x0 to ``t_next`` with the forward
        process, using the unit-variance noise ``sqrt(1 - rho^2) eps + rho z`` with fresh ``z ~ N(0, I)``.
        The fresh-noise scale ``rho = eta sqrt(v) / sigma(t)`` is ``eta`` times that of the DDPM posterior,
        where ``v`` is the variance of the forward transition from ``t_next`` to ``t``.

        Args:
            xt: The state at time *t*
            model_output: A prediction of type *pred_type* made at ``(xt, t)``
            t: Current time, a scalar or one per sample
            t_next: Time to step to, a scalar or one per sample; with ``clean_end``, ``0``, the clean end of a
                student grid, gives x0 even below ``min_t``
            pred_type: One of ``"x0"``, ``"eps"``, ``"v"``, ``"flow"``; ``"eps"`` raises where ``alpha(t)``
                vanishes, since it determines no x0 there
            sample_type: ``"ode"`` (``eta = 0``, DDIM), ``"ddpm"`` (``eta = 1``, ancestral sampling),
                ``"sde"`` (``rho = 1``, fresh noise: the largest ``eta``), or a float ``eta >= 0``
            clean_end: Whether ``t_next = 0`` returns x0 instead of the schedule's time-0 state

        Returns:
            The state at *t_next*
        """
        # TODO: Specialize the step per schedule and prediction type for stability. The conversion of the
        # prediction to x0 and eps divides by alpha(t) or sigma(t); RFNoiseSchedule steps flow directly.
        check_sample_type(sample_type)
        if pred_type not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported pred_type '{pred_type}'. Supported types: {NET_PRED_TYPES}")

        t = t.expand(xt.shape[0])
        # Samples at the clean end skip the forward process and return x0.
        is_clean = t_next.expand(xt.shape[0]) == 0 if clean_end else torch.zeros_like(t, dtype=torch.bool)
        t_next = torch.where(is_clean, t, t_next)
        # Check each time once, in its own precision; everything below is unchecked, in high precision.
        self._assert_t_valid(t)
        self._assert_t_valid(t_next, "t_next")

        original_dtype = xt.dtype
        dtype = high_precision_dtype(xt, model_output)
        xt = xt.to(dtype)
        t, t_next = to_high_precision(t, t_next)
        if pred_type == "eps":
            self._assert_eps_determines_x0(t)

        x0 = self._convert_model_output(xt, model_output, t, pred_type, "x0")
        # Fresh noise is drawn like the state, so the draw does not depend on the prediction's precision or layout.
        if sample_type == "sde":
            eps = torch.randn_like(xt, dtype=original_dtype)
        else:
            eps = model_output.to(dtype) if pred_type == "eps" else self._x0_to_eps(xt, x0, t)
            eta = {"ode": 0.0, "ddpm": 1.0}.get(sample_type, sample_type)
            if eta != 0:
                assert_tensor((t_next <= t).all(), "Stochastic steps need t_next <= t.", error_type=ValueError)
                fresh_noise_scale = (
                    eta * self._transition_std(t, t_next, clean_end=clean_end) / self.non_zero_clamp(self._sigma(t))
                )
                # Clean samples draw no noise; their zero-length transition is 0 / 0 where alpha(t) = 0.
                fresh_noise_scale = torch.where(is_clean, 0.0, fresh_noise_scale)
                # The transition variance may round above sigma(t)^2, which the clamp below removes.
                assert_tensor(
                    (fresh_noise_scale <= 1 + self.clamp_min).all(),
                    f"eta={eta} exceeds the fresh-noise step (sample_type='sde').",
                    error_type=ValueError,
                )
                fresh_noise_scale = expand_like(fresh_noise_scale.clamp(max=1), eps)
                fresh_noise = torch.randn_like(xt, dtype=original_dtype).to(dtype)
                eps = (1 - fresh_noise_scale**2).sqrt().to(dtype) * eps + fresh_noise_scale.to(dtype) * fresh_noise
        x_next = self._forward_process(x0, eps, t_next)
        assert_same(dtype=dtype, x0=x0, x_next=x_next)
        return torch.where(expand_like(is_clean, x_next), x0, x_next).to(original_dtype)

    def sample_t_inhom(self, n: int, seq_len: int, chunk_size: int, **sample_t_kwargs) -> torch.Tensor:
        """
        Sample block-wise independent t variables for diffusion forcing in SFT.

        Each causal chunk draws one time; the first chunk also takes any remainder frames.

        Args:
            n: batch size
            seq_len: sequence length
            chunk_size: chunk size

        Return:
            torch.Tensor: time steps of shape [n, seq_len].
        """
        chunk_frames = causal_chunk_frames(seq_len, chunk_size)
        flat_t = self.sample_t(n * len(chunk_frames), **sample_t_kwargs)

        t = flat_t.view(n, len(chunk_frames))

        # A non-blocking copy and output_size keep the expansion free of host syncs.
        repeats = torch.tensor(chunk_frames).to(t.device, non_blocking=True)
        return t.repeat_interleave(repeats, dim=1, output_size=seq_len)

    def is_t_valid(
        self,
        t: torch.Tensor,
        min_t: Optional[float] = None,
        max_t: Optional[float] = None,
    ) -> torch.Tensor:
        """Check *t* against the schedule range, optionally overridden locally."""
        min_t = self.min_t if min_t is None else min_t
        max_t = self.max_t if max_t is None else max_t
        # The bounds rounded to t's precision, widened by one ulp of it.
        lower = _nextafter(min_t, -torch.inf, t.dtype)
        upper = _nextafter(max_t, torch.inf, t.dtype)
        return torch.all((lower <= t) & (t <= upper))

    def _assert_t_valid(
        self,
        t: torch.Tensor,
        name: str = "t",
        min_t: Optional[float] = None,
        max_t: Optional[float] = None,
    ) -> None:
        assert_tensor(
            self.is_t_valid(t, min_t=min_t, max_t=max_t),
            f"{name} must be in "
            f"[{self.min_t if min_t is None else min_t}, {self.max_t if max_t is None else max_t}]",
            error_type=ValueError,
        )

    def _assert_t_valid_or_clean(self, t: torch.Tensor, name: str, clean_end: bool) -> None:
        """``_assert_t_valid``, where ``clean_end`` also admits the exactly clean end ``t = 0`` of a student grid,
        even below ``min_t``, but no other time below it (as ``step`` treats ``t_next``)."""
        self._assert_t_valid(torch.where(t == 0, self.max_t, t) if clean_end else t, name)

    def _assert_eps_determines_x0(self, t: torch.Tensor) -> None:
        """An eps prediction determines x0 only where ``alpha(t)`` survives ``non_zero_clamp``; at zero SNR,
        x_t is all noise."""
        assert_tensor(
            (self._alpha(to_high_precision(t)).abs() >= self.clamp_min).all(),
            f"An eps prediction does not determine x0 where alpha(t) < clamp_min={self.clamp_min}; "
            "use an x0, v or flow prediction there.",
            error_type=ValueError,
        )

    def _assert_transition_valid(
        self,
        t: torch.Tensor,
        t_start: torch.Tensor,
        clean_end: bool = True,
    ) -> None:
        self._assert_t_valid(t)
        self._assert_t_valid_or_clean(t_start, "t_start", clean_end)
        assert_tensor((t >= t_start).all(), "The forward transition needs t >= t_start.", error_type=ValueError)

    def forward_process(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        t_start: Optional[torch.Tensor] = None,
        clean_end: bool = True,
    ) -> torch.Tensor:
        """The diffusion forward process, or with `t_start` its Markov transition from time `t_start` to `t`.

        The transition ``x_t = a x + sqrt(v) eps`` with ``a = alpha(t) / alpha(t_start)`` and
        ``v = sigma(t)^2 - a^2 sigma(t_start)^2`` is the linear-Gaussian one that keeps this schedule's marginals.

        Args:
            x (torch.Tensor): Input sample from the data distribution, or the state at time `t_start`.
            eps (torch.Tensor): Noise sample, should have the same shape as `x`.
            t (torch.Tensor): The time step in the forward process.
            t_start (torch.Tensor, optional): Time of `x`, noised further to `t >= t_start`. Defaults to None,
                i.e., `x` is data.
            clean_end: Whether ``t_start = 0`` denotes data instead of the schedule's time-0 state.

        Returns:
            torch.Tensor: Sample from the forward process at time `t` given `x`.
        """
        if t_start is None:
            self._assert_t_valid(t)
        else:
            self._assert_transition_valid(t, t_start, clean_end=clean_end)
        return self._forward_process(x, eps, t, t_start=t_start, clean_end=clean_end).to(x.dtype)

    def _forward_process(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        t_start: Optional[torch.Tensor] = None,
        clean_end: bool = True,
    ) -> torch.Tensor:
        """Unchecked high-precision implementation of ``forward_process``."""
        x, eps = to_high_precision(x, eps)
        t, t_start = to_high_precision(t, t_start)
        if t_start is None:
            alpha, sigma = self._alpha_sigma(t)
        else:
            alpha = self._alpha(t) / self._alpha_start(t_start, clean_end=clean_end)
            # A zero-length transition from the schedule's own state keeps it, also where alpha(t_start) vanishes and
            # the ratio is 0 / 0 (rectified flow at t = 1); from the clean end it is the forward process at t.
            keeps_state = (t == t_start) & (t_start != 0) if clean_end else t == t_start
            alpha = torch.where(keeps_state, 1.0, alpha)
            sigma = self._transition_std(t, t_start, clean_end=clean_end)
        alpha, sigma = expand_like(alpha.to(x.dtype), x), expand_like(sigma.to(x.dtype), eps)
        return x.mul(alpha).add(eps.mul(sigma))

    def alpha_start(self, t_start: torch.Tensor, clean_end: bool = True) -> torch.Tensor:
        """The data scale of the state a forward transition starts from at time `t_start`.

        Args:
            t_start (torch.Tensor): The time step the transition starts from.
            clean_end: Whether ``t_start = 0`` denotes data instead of the schedule's time-0 state.

        Returns:
            torch.Tensor: ``alpha(t_start)``, or ``1`` at the clean end.
        """
        self._assert_t_valid_or_clean(t_start, "t_start", clean_end)
        return self._alpha_start(to_high_precision(t_start), clean_end=clean_end).to(t_start.dtype)

    def _alpha_start(self, t_start: torch.Tensor, clean_end: bool = True) -> torch.Tensor:
        """Unchecked implementation of ``alpha_start``."""
        # The clean end holds data even where this schedule's own time 0 is noisy (AlphasNoiseSchedule); keep this in
        # step with its noise scale in ``_transition_std``.
        alpha_start = self._alpha(t_start)
        return torch.where(t_start == 0, 1.0, alpha_start) if clean_end else alpha_start

    def transition_std(
        self,
        t: torch.Tensor,
        t_start: torch.Tensor,
        clean_end: bool = True,
    ) -> torch.Tensor:
        """The noise scale ``sqrt(v)`` of the forward transition from time `t_start` to `t`, see ``forward_process``.

        Args:
            t (torch.Tensor): The time step the transition noises to.
            t_start (torch.Tensor): The time step it starts from, with ``t_start <= t``.
            clean_end: Whether ``t_start = 0`` denotes data instead of the schedule's time-0 state.

        Returns:
            torch.Tensor: ``sqrt(sigma(t)^2 - a^2 sigma(t_start)^2)`` with ``a = alpha(t) / alpha(t_start)``,
                or ``sigma(t)`` from the clean end, in the dtype of `t`.
        """
        self._assert_transition_valid(t, t_start, clean_end=clean_end)
        return self._transition_std(*to_high_precision(t, t_start), clean_end=clean_end).to(t.dtype)

    def _transition_std(
        self,
        t: torch.Tensor,
        t_start: torch.Tensor,
        clean_end: bool = True,
    ) -> torch.Tensor:
        """Unchecked implementation of ``transition_std``."""
        alpha_start, sigma_start = self._alpha_sigma(t_start)
        alpha, sigma = self._alpha_sigma(t)
        # v = det (alpha_start sigma_t + alpha_t sigma_start) / alpha_start^2, where only the determinant cancels as
        # t_start -> t.
        variance = self._determinant(t_start, t) * (alpha_start * sigma + alpha * sigma_start) / alpha_start**2
        # A zero-length transition adds no noise, also where alpha(t_start) vanishes and the form above is 0 / 0
        # (rectified flow at t = 1).
        transition_std = torch.where(t == t_start, 0.0, variance.clamp(min=0).sqrt())
        # As in ``_alpha_start``, the clean end holds data even where this schedule's own time 0 is noisy
        # (AlphasNoiseSchedule), which the determinant form above cannot express.
        return torch.where(t_start == 0, sigma, transition_std) if clean_end else transition_std

    def determinant(self, t_start: torch.Tensor, t_end: torch.Tensor, clean_end: bool = True) -> torch.Tensor:
        """The determinant ``alpha(t_start) sigma(t_end) - alpha(t_end) sigma(t_start)`` of two schedule points.

        On rectified flow it is the sigma increment ``sigma(t_end) - sigma(t_start)``. Unlike that difference, it
        keeps its precision as the times meet.

        Args:
            t_start (torch.Tensor): The first time step.
            t_end (torch.Tensor): The second time step.
            clean_end: Whether a time ``0`` denotes data (``alpha = 1``, ``sigma = 0``), even below ``min_t``,
                instead of the schedule's time-0 state; the two differ only where that state is noisy
                (AlphasNoiseSchedule).

        Returns:
            torch.Tensor: The determinant, in the dtype of `t_start`.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        self._assert_t_valid_or_clean(t_start, "t_start", clean_end)
        self._assert_t_valid_or_clean(t_end, "t_end", clean_end)
        dtype = t_start.dtype
        t_start, t_end = to_high_precision(t_start, t_end)
        determinant = self._determinant(t_start, t_end)
        if clean_end:
            # Where this schedule's own time 0 is clean, it is the clean end already.
            noisy_zero = self._sigma(t_start.new_zeros(())) != 0
            at_clean_end = (t_start == 0) | (t_end == 0)
            sigma_other = self._sigma(torch.where(t_end == 0, t_start, t_end))
            clean = torch.where(t_end == 0, -sigma_other, sigma_other)
            clean = torch.where((t_start == 0) & (t_end == 0), 0.0, clean)
            determinant = torch.where(noisy_zero & at_clean_end, clean, determinant)
        return determinant.to(dtype)

    def _determinant(self, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        """``alpha(t_start) sigma(t_end) - alpha(t_end) sigma(t_start)``, the determinant of two schedule points.

        It vanishes as the times meet, where this difference cancels; schedules override it with a form that does not.
        Unchecked: callers own time validation.
        """
        alpha_start, sigma_start = self._alpha_sigma(t_start)
        alpha_end, sigma_end = self._alpha_sigma(t_end)
        return alpha_start * sigma_end - alpha_end * sigma_start

    def cond_velocity(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        output_dtype: OutputDtype = "same",
    ) -> torch.Tensor:
        """The instantaneous velocity for this noise schedule: dx/dt := alpha'(t) x_0 + sigma'(t) eps.

        Args:
            x (torch.Tensor): Input sample from the data distribution.
            eps (torch.Tensor): Noise sample, should have the same shape as `x`.
            t (torch.Tensor): The time step in the forward process.
            output_dtype: Dtype of the returned velocity. ``"same"`` is ``x``'s; ``None`` keeps it uncast.

        Returns:
            torch.Tensor: The velocity ``alpha'(t) x + sigma'(t) eps`` of the forward process at time `t`.
        """
        self._assert_t_valid(t)
        return cast_like(self._cond_velocity(x, eps, t), x, output_dtype)

    def _cond_velocity(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Unchecked high-precision implementation of ``cond_velocity``."""

        # Convert to high precision for calculation
        x, eps = to_high_precision(x, eps)
        t = to_high_precision(t)

        # Perform calculation
        alpha_prime_t = expand_like(self._alpha_prime(t).to(x.dtype), x)
        sigma_prime_t = expand_like(self._sigma_prime(t).to(x.dtype), eps)
        cond_vel = x.mul(alpha_prime_t).add(eps.mul(sigma_prime_t))

        return cond_vel

    def cond_v(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        output_dtype: OutputDtype = "same",
    ) -> torch.Tensor:
        """The v-prediction target for this noise schedule: v := alpha(t) eps - sigma(t) x_0.

        Unlike ``x0_to_v``, it takes the noise instead of ``xt``, so it does not divide by ``sigma(t)`` and also
        holds at clean times. Only ``AlphasNoiseSchedule`` has v-prediction; other schedules raise ``ValueError``.

        Args:
            x (torch.Tensor): Input sample from the data distribution.
            eps (torch.Tensor): Noise sample, should have the same shape as `x`.
            t (torch.Tensor): The time step in the forward process.
            output_dtype: Dtype of the returned target. ``"same"`` is ``x``'s; ``None`` keeps it uncast.

        Returns:
            torch.Tensor: The v-prediction target ``alpha(t) eps - sigma(t) x`` at time `t`.
        """
        self._assert_t_valid(t)
        return cast_like(self._cond_v(x, eps, t), x, output_dtype)

    def _cond_v(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        raise ValueError(f"{type(self).__name__} does not support v-prediction; only AlphasNoiseSchedule does")

    def sqrt_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the square root of the Signal-to-Noise Ratio (SNR) at time `t`.
        sqrt_snr(t) = alpha(t) / sigma(t)
        """
        self._assert_t_valid(t)
        return self._sqrt_snr(to_high_precision(t)).to(t.dtype)

    def _sqrt_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Unchecked implementation of ``sqrt_snr``."""
        alpha, sigma = self._alpha_sigma(t)
        return alpha / self.non_zero_clamp(sigma)

    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """
        Find `t` that matches the target square root of SNR value, where sqrt_snr_t := alpha_t / sigma_t.

        The caller passes the square-root SNR of a time in range, up to rounding. Inverting can carry that rounding
        past the range, so the time is clamped to it after the cast, keeping its derivative.

        Args:
            sqrt_snr_t (torch.Tensor): The square root of the SNR at time `t`.

        Returns:
            torch.Tensor: The timestep, in the dtype of *sqrt_snr_t*.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        t = self._sqrt_snr_to_t(to_high_precision(sqrt_snr_t)).to(sqrt_snr_t.dtype)
        return self.safe_clamp(t, min=self.min_t, max=self.max_t, keep_grad=True)

    def _sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """Unchecked implementation of ``sqrt_snr_to_t``."""
        raise NotImplementedError

    def x0_to_eps(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
        output_dtype: OutputDtype = "same",
    ) -> torch.Tensor:
        """
        Compute the eps-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.
            output_dtype: Dtype of the returned prediction. ``"same"`` is ``xt``'s; ``None`` keeps it uncast.

        Returns:
            torch.Tensor: The eps-prediction.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        self._assert_t_valid(t)
        return cast_like(self._x0_to_eps(xt, x0, t), xt, output_dtype)

    def _x0_to_eps(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Unchecked high-precision implementation of ``x0_to_eps``."""

        # Convert to high precision for calculation
        xt, x0 = to_high_precision(xt, x0)
        t = to_high_precision(t)

        alpha_t, sigma_t = self._alpha_sigma(t)
        alpha_t = expand_like(alpha_t.to(xt.dtype), xt)
        sigma_t = expand_like(self.non_zero_clamp(sigma_t).to(xt.dtype), xt)

        # Perform calculation
        eps = (xt - x0.mul(alpha_t)).div(sigma_t)

        return eps

    def eps_to_x0(
        self,
        xt: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        output_dtype: OutputDtype = "same",
    ) -> torch.Tensor:
        """
        Compute the x0-prediction from the eps-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            eps (torch.Tensor): The eps-prediction.
            t (torch.Tensor): The time step in the forward process.
            output_dtype: Dtype of the returned prediction. ``"same"`` is ``xt``'s; ``None`` keeps it uncast.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        self._assert_t_valid(t)
        self._assert_eps_determines_x0(t)
        return cast_like(self._eps_to_x0(xt, eps, t), xt, output_dtype)

    def _eps_to_x0(
        self,
        xt: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Unchecked high-precision implementation of ``eps_to_x0``."""

        # Convert to high precision for calculation
        xt, eps = to_high_precision(xt, eps)
        t = to_high_precision(t)

        alpha_t, sigma_t = self._alpha_sigma(t)
        alpha_t = expand_like(self.non_zero_clamp(alpha_t).to(xt.dtype), xt)
        sigma_t = expand_like(sigma_t.to(xt.dtype), xt)

        # Perform calculation
        x0 = (xt - eps.mul(sigma_t)).div(alpha_t)

        return x0

    def flow_to_x0(
        self,
        xt: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        output_dtype: OutputDtype = "same",
    ) -> torch.Tensor:
        """
        Compute the x0-prediction from the flow-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The flow-prediction.
            t (torch.Tensor): The time step in the forward process.
            output_dtype: Dtype of the returned prediction. ``"same"`` is ``xt``'s; ``None`` keeps it uncast.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        self._assert_t_valid(t)
        return cast_like(self._flow_to_x0(xt, v, t), xt, output_dtype)

    def _flow_to_x0(
        self,
        xt: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Unchecked high-precision implementation of ``flow_to_x0``."""

        # Convert to high precision for calculation
        xt, v = to_high_precision(xt, v)
        t = to_high_precision(t)

        alpha_t, sigma_t = self._alpha_sigma(t)
        alpha_t, sigma_t = expand_like(alpha_t, xt), expand_like(sigma_t, xt)
        alpha_prime_t = expand_like(self._alpha_prime(t), xt)
        sigma_prime_t = expand_like(self._sigma_prime(t), xt)

        # Perform calculation
        xt_coeff = sigma_prime_t / self.non_zero_clamp(sigma_t)
        x0_coeff = self.non_zero_clamp(xt_coeff * alpha_t - alpha_prime_t)
        xt_coeff, x0_coeff = xt_coeff.to(xt.dtype), x0_coeff.to(xt.dtype)
        x0 = (xt * xt_coeff - v) / x0_coeff

        return x0

    def points_to_x0(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        clean_end: bool = True,
        output_dtype: OutputDtype = "same",
    ) -> torch.Tensor:
        """Recover ``x0`` from two points on one conditional noise path.

        The endpoints must share the same latent ``x0`` and noise ``eps``:

        ``x(t) = alpha(t) * x0 + sigma(t) * eps``.

        Unlike a local model-output conversion, this operation needs no
        prediction-type convention. It is therefore suitable for converting
        an explicitly integrated start/end pair into an ``x0`` estimate.

        The pair runs like a sampler step, ``t_start > t_end``. A ``t_end`` of
        ``0``, the clean end of a student grid, marks ``x_end`` as data, which
        is then ``x0``, even below ``min_t`` or where this schedule's own time
        ``0`` is noisy (see ``forward_process``). Set ``clean_end=False`` to
        interpret it as the schedule's time-0 state instead.

        ``output_dtype`` is the dtype of the returned ``x0``: ``"same"`` is
        ``x_start``'s; ``None`` keeps it uncast.
        """
        self._assert_t_valid(t_start, "t_start")
        self._assert_t_valid_or_clean(t_end, "t_end", clean_end)
        assert_tensor((t_start > t_end).all(), "The two-point inversion needs t_start > t_end.", error_type=ValueError)
        determinant = self._determinant(*to_high_precision(t_start, t_end))
        # A clean x_end is x0 itself, so its pair need not be invertible.
        is_clean = t_end == 0 if clean_end else torch.zeros_like(t_end, dtype=torch.bool)
        assert_tensor(
            (is_clean | (torch.isfinite(determinant) & (determinant != 0))).all(),
            "The two schedule points are singular and do not uniquely determine x0",
            error_type=ValueError,
        )
        x0 = self._points_to_x0(x_start, x_end, t_start, t_end, clean_end=clean_end)
        return cast_like(x0, x_start, output_dtype)

    def _points_to_x0(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        clean_end: bool = True,
    ) -> torch.Tensor:
        """Unchecked high-precision two-point inversion behind ``points_to_x0``, which applies the clean-end rule."""
        # Convert to high precision for calculation
        x_start, x_end = to_high_precision(x_start, x_end)
        t_start, t_end = to_high_precision(t_start, t_end)

        # A clean pair divides by 1: the result ``torch.where`` below backpropagates zeros into its discarded
        # inversion, which a zero determinant would turn into NaN gradients.
        is_clean = t_end == 0 if clean_end else torch.zeros_like(t_end, dtype=torch.bool)
        determinant = self._determinant(t_start, t_end)
        determinant = expand_like(torch.where(is_clean, 1.0, determinant).to(x_start.dtype), x_start)
        sigma_start = expand_like(self._sigma(t_start).to(x_start.dtype), x_start)
        sigma_end = expand_like(self._sigma(t_end).to(x_start.dtype), x_start)
        x0 = (sigma_end * x_start - sigma_start * x_end) / determinant

        # The clean end holds data, but the inversion reads time 0 as this schedule's own: noisy for
        # AlphasNoiseSchedule, and exact only up to roundoff elsewhere.
        return torch.where(expand_like(is_clean, x0), x_end, x0)

    def x0_to_flow(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
        output_dtype: OutputDtype = "same",
    ) -> torch.Tensor:
        """
        Compute the flow-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.
            output_dtype: Dtype of the returned prediction. ``"same"`` is ``xt``'s; ``None`` keeps it uncast.

        Returns:
            torch.Tensor: The flow-prediction.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        self._assert_t_valid(t)
        return cast_like(self._x0_to_flow(xt, x0, t), xt, output_dtype)

    def _x0_to_flow(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Unchecked high-precision implementation of ``x0_to_flow``."""
        eps = self._x0_to_eps(xt, x0, t)
        return self._cond_velocity(x0, eps, t)

    def _v_to_x0(
        self,
        xt: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        raise ValueError(f"{type(self).__name__} does not support v-prediction; only AlphasNoiseSchedule does")

    def _x0_to_v(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        raise ValueError(f"{type(self).__name__} does not support v-prediction; only AlphasNoiseSchedule does")

    def convert_model_output(
        self,
        xt: torch.Tensor,
        model_output: torch.Tensor,
        t: torch.Tensor,
        src_pred_type: str = "x0",
        target_pred_type: str = "eps",
        output_dtype: OutputDtype = "same",
    ) -> torch.Tensor:
        """
        Convert the model output from src_pred_type to target_pred_type.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            model_output (`torch.Tensor`): The direct output from the learned diffusion model.
            t (torch.Tensor): The time step in the forward process.
            src_pred_type (`str`): The type of src prediction, choose from 'x0', 'eps', 'v', 'flow'.
            target_pred_type (`str`): The type of target prediction, choose from 'x0', 'eps', 'v', 'flow'.
            output_dtype: Dtype of the returned prediction. ``"same"`` is ``xt``'s; ``None`` keeps it uncast.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        assert_same(dtype=self.t_precision, device=xt.device, t=t)

        # Supported prediction types
        if src_pred_type not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported src_pred_type '{src_pred_type}'. Supported types: {NET_PRED_TYPES}")
        if target_pred_type not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported target_pred_type '{target_pred_type}'. Supported types: {NET_PRED_TYPES}")

        # Matching prediction types need no arithmetic, so only a conversion checks the times.
        if src_pred_type != target_pred_type:
            self._assert_t_valid(t)
            if src_pred_type == "eps":
                self._assert_eps_determines_x0(t)
            model_output = self._convert_model_output(xt, model_output, t, src_pred_type, target_pred_type)
        # ``None`` keeps the high precision a conversion returns; a matching prediction is cast to it.
        dtype = high_precision_dtype(xt, model_output) if output_dtype is None else output_dtype
        return cast_like(model_output, xt, dtype)

    def _convert_model_output(
        self,
        xt: torch.Tensor,
        model_output: torch.Tensor,
        t: torch.Tensor,
        src_pred_type: str,
        target_pred_type: str,
    ) -> torch.Tensor:
        """Unchecked high-precision implementation of ``convert_model_output``, for any timestep precision."""
        dtype = high_precision_dtype(xt, model_output)
        if src_pred_type == target_pred_type:
            return model_output.to(dtype)

        # Cast once; each stage would otherwise make its own high-precision copy.
        xt = xt.to(dtype)

        # Convert through intermediate representations
        # First convert src to x0 (if needed)
        if src_pred_type == "x0":
            x0 = model_output
        elif src_pred_type == "eps":
            x0 = self._eps_to_x0(xt, model_output, t)
        elif src_pred_type == "v":
            x0 = self._v_to_x0(xt, model_output, t)
        elif src_pred_type == "flow":
            x0 = self._flow_to_x0(xt, model_output, t)
        else:
            raise ValueError(f"Conversion from '{src_pred_type}' not implemented")

        # Then convert x0 to target (if needed)
        if target_pred_type == "x0":
            converted = x0
        elif target_pred_type == "eps":
            converted = self._x0_to_eps(xt, x0, t)
        elif target_pred_type == "v":
            converted = self._x0_to_v(xt, x0, t)
        elif target_pred_type == "flow":
            converted = self._x0_to_flow(xt, x0, t)
        else:
            raise ValueError(f"Conversion to '{target_pred_type}' not implemented")
        return converted


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
        **kwargs,
    ):
        """
        Args:
            rho: The exponent of ``grid``'s polynomial spacing: ``rho = 1`` spaces the times evenly, and a larger
                ``rho`` puts more of them at low noise. ``sample_t`` takes its own ``rho`` for ``polynomial``.
        """
        super().__init__(min_t, max_t, **kwargs)
        if not 0.002 <= min_t < max_t <= 80.0:
            logger.warning(f"EDM min_t and max_t should be between 0.002 and 80.0, but got {min_t} and {max_t}.")
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}")
        self._rho = float(rho)

    def _grid(self, num_steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """The uniform grid mapped through ``polynomial_spacing`` with exponent ``rho``: the times' ``1/rho`` powers
        run evenly between those of ``min_t`` and ``max_t``, and the clean end stays 0."""
        uniform = super()._grid(num_steps, device)
        low, high = self.min_t, self.max_t
        grid = torch.where(uniform == 0, 0.0, polynomial_spacing(uniform / high, self._rho, low, high))
        # The power can round the top several ulps past max_t, more than the check in _canonicalize_grid tolerates.
        return self.clamp_grid(grid)

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def _determinant(self, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        """``alpha = 1`` and ``sigma = t`` reduce the determinant to the time increment."""
        return t_end - t_start

    def _sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        return 1 / self.non_zero_clamp(sqrt_snr_t)

    def _flow_to_x0(
        self,
        xt: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the x0-prediction from the flow-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The flow-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        # Convert to high precision for calculation
        xt, v = to_high_precision(xt, v)
        t = to_high_precision(t)
        t_expanded = expand_like(t.to(xt.dtype), xt)

        # Perform calculation
        x0 = xt - v.mul(t_expanded)

        return x0

    def _x0_to_flow(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the flow-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The flow-prediction.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        # Convert to high precision for calculation
        xt, x0 = to_high_precision(xt, x0)
        t = to_high_precision(t)
        t_expanded = expand_like(self.non_zero_clamp(t).to(xt.dtype), xt)

        # Perform calculation
        flow = (xt - x0).div(t_expanded)

        return flow


class AlphasNoiseSchedule(BaseNoiseSchedule):
    """
    The noise schedule of a discrete DDPM table ``alphas_cumprod``, as in Stable Diffusion:

        x_t = alpha(t) x_0 + sigma(t) eps,

    where t is in [min_t, max_t] within [0, (N - 1) / N], the range up to the last knot of a table of
    ``N = num_train_timesteps`` entries (0.999 for 1000), and the network sees the continuous table index ``N * t``.
    Between table entries, ``alpha(t)^2`` is linear in time and ``sigma(t)^2 = 1 - alpha(t)^2``,
    including next to an exactly clean entry. Into the first pure-noise entry ``alpha(t)`` itself is linear, so it
    vanishes with a finite slope: with ``alpha^2`` linear there, the time would approach that entry quadratically in
    the square-root SNR, which a rounded time cannot resolve. The coefficients are continuous and monotone in t.
    Knots are the representable times ``k / N`` and give the table's own square roots; interpolation uses
    differences of these times, avoiding the loss of distinct times when first multiplying them by ``N``.
    A table must have at least two finite entries in [0, 1] and be nonincreasing. Plateaus are allowed. At an
    interior knot, autograd uses the right-hand derivative, except at the first pure-noise knot, which keeps the
    segment into it; at a clean endpoint a square-root derivative can be singular.
    """

    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        min_t: float = 0.0,
        max_t: Optional[float] = None,
        **kwargs,
    ):
        if alphas_cumprod.ndim != 1 or alphas_cumprod.numel() < 2:
            raise ValueError("alphas_cumprod must be a vector of at least two entries")
        num_train_timesteps = alphas_cumprod.numel()
        # Times above the last knot share its coefficients while the network sees distinct indices, so the knot bounds
        # the range (max_t defaults to it).
        last_knot = (num_train_timesteps - 1) / num_train_timesteps
        if max_t is None:
            max_t = last_knot
        elif max_t > last_knot:
            raise ValueError(f"Alphas max_t must be at most the table's last knot {last_knot}, got {max_t}")
        super().__init__(min_t, max_t, **kwargs)
        self.num_train_timesteps = num_train_timesteps
        # Keep the reconstruction data as configuration, independent of buffer placement and
        # to_empty(). Standard schedule factories build this small source table on CPU even
        # during meta initialization; a meta source itself cannot supply reconstruction values.
        if alphas_cumprod.is_meta:
            raise ValueError("alphas_cumprod must contain real values to reconstruct the non-persistent buffer")
        self._alphas_cumprod_values = tuple(alphas_cumprod.tolist())
        values = self._alphas_cumprod_values
        if not all(0 <= p <= 1 for p in values) or any(a < b for a, b in zip(values, values[1:])):
            raise ValueError("alphas_cumprod must be finite, nonincreasing and in [0, 1]")
        # Only a table that descends from a positive entry to an exact zero has a segment into pure noise.
        self._reaches_pure_noise = values[0] > 0 and values[-1] == 0
        self.register_buffer("alphas_cumprod", torch.empty(num_train_timesteps, dtype=torch.float64), persistent=False)
        self.reset_parameters()

    def _fixed_precision_overrides(self) -> Dict[str, torch.dtype]:
        """The table stays float64 in every precision: bfloat16 would round DDPM's first entry 0.9999 to 1, which
        would make time 0 clean."""
        return {"alphas_cumprod": torch.float64}

    def reset_parameters(self) -> None:
        self.alphas_cumprod.copy_(self.alphas_cumprod.new_tensor(self._alphas_cumprod_values))

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.num_train_timesteps * t

    def _time_at_index(self, index: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        index = index.to(dtype)
        # CUDA scalar division multiplies by a rounded reciprocal. A tensor divisor gives the same correctly
        # rounded k/N knots as CPU division, including native timesteps constructed on CPU and then transferred.
        return index / torch.full_like(index, self.num_train_timesteps)

    def _interpolate(self, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return (time, left, right, width, complement, alpha, sigma).

        Exact interior knots use the segment on their right, except the first pure-noise knot, which keeps the
        segment into it.
        """
        t = t.clamp(0, (self.num_train_timesteps - 1) / self.num_train_timesteps)
        knots = self._time_at_index(torch.arange(self.num_train_timesteps, device=t.device, dtype=t.dtype), t.dtype)
        table = self.alphas_cumprod.to(t)
        left = torch.searchsorted(knots, t.contiguous(), side="right").sub(1).clamp(0, self.num_train_timesteps - 1)
        if self._reaches_pure_noise:
            previous = (left - 1).clamp(min=0)
            left = torch.where((t == knots[left]) & (table[left] == 0) & (table[previous] > 0), previous, left)
        right = (left + 1).clamp(max=self.num_train_timesteps - 1)
        low, high = knots[left], knots[right]
        width = torch.where(right > left, high - low, 1)
        # Compute both weights directly: 1 - weight would lose distances very near an endpoint.
        weight = torch.where(right > left, (t - low) / width, 0)
        complement = torch.where(right > left, (high - t) / width, 1)
        p, q = table[left], table[right]
        change = p - q
        # Scale tiny probabilities before their square root, and use the nearer endpoint's weight to avoid
        # cancellation. At weight=0 the roots are exactly the table's roots, with their time derivatives intact.
        scale = torch.where(p > 0, p, 1)
        alpha_factor = torch.where(
            weight <= 0.5, 1 - (change / scale) * weight, q / scale + (change / scale) * complement
        )
        alpha_squared = p * alpha_factor
        sigma_squared = (1 - p) + change * weight
        # Only an exactly clean left entry can have an arbitrarily tiny noise probability. Keep unused
        # square-root branches positive so flat pure-state segments have finite, zero derivatives.
        clean_root = torch.where((p == 1) & (q < 1), weight, 1).sqrt()
        if self._reaches_pure_noise:
            # Into pure noise alpha = sqrt(p) * complement and sigma^2 = (1 - p) + p * (1 + complement) * weight.
            into_noise = (q == 0) & (p > 0)
            alpha_root = torch.where(into_noise, complement, torch.where((p > 0) & ~into_noise, alpha_factor, 1).sqrt())
            alpha_squared = torch.where(into_noise, p * complement.square(), alpha_squared)
            sigma_squared = torch.where(into_noise, (1 - p) + p * (1 + complement) * weight, sigma_squared)
            # Rooted separately so that backward gives sigma's slope at an exactly clean entry as inf, not NaN.
            clean_root = clean_root * torch.where(into_noise, 1 + complement, 1).sqrt()
        else:
            alpha_root = torch.where(p > 0, alpha_factor, 1).sqrt()
        alpha = p.sqrt() * alpha_root
        sigma = torch.where(p == 1, (1 - q).sqrt() * clean_root, torch.where(p < 1, sigma_squared, 1).sqrt())
        # The larger root is well conditioned as sqrt(1 - smaller_probability). This avoids a one-ulp
        # normalization overshoot at knots, especially in float32, while retaining tiny smaller roots.
        alpha_larger = alpha_squared >= sigma_squared
        larger = (1 - torch.where(alpha_larger, sigma_squared, alpha_squared)).sqrt()
        alpha, sigma = torch.where(alpha_larger, larger, alpha), torch.where(alpha_larger, sigma, larger)
        return t, left, right, width, complement, alpha, sigma

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        alpha, _ = self._alpha_sigma(t)
        return alpha

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        _, sigma = self._alpha_sigma(t)
        return sigma

    def _alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        *_, alpha, sigma = self._interpolate(t)
        return alpha, sigma

    def _determinant(self, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        """sin(theta_end - theta_start), where (alpha, sigma) = (cos(theta), sin(theta)).

        Rationalize the difference as (p_start - p_end) / (alpha_start*sigma_end + alpha_end*sigma_start).
        Compute the numerator from time differences and table differences, splitting across knots so nearby
        probabilities never cancel. Into pure noise the probability ``p * complement^2`` is quadratic in time, so
        a piece there carries the sum of its endpoint complements as a factor. Divide before multiplying tiny time
        differences to avoid underflow.
        """
        s, left_s, right_s, width_s, complement_s, a_s, b_s = self._interpolate(t_start)
        e, left_e, right_e, width_e, complement_e, a_e, b_e = self._interpolate(t_end)
        table = self.alphas_cumprod.to(s)
        denominator = a_s * b_e + a_e * b_s
        denominator = torch.where(denominator > 0, denominator, 1)
        slope_s = (table[right_s] - table[left_s]) / denominator
        slope_e = (table[right_e] - table[left_e]) / denominator
        same = left_s == left_e
        forward = s <= e
        single = slope_s * ((s - e) / width_s)
        entry_s = torch.where(forward, right_s, left_s)
        entry_e = torch.where(forward, left_e, right_e)
        knot_s, knot_e = self._time_at_index(entry_s, s.dtype), self._time_at_index(entry_e, s.dtype)
        first = slope_s * ((s - knot_s) / width_s)
        middle = (table[entry_s] - table[entry_e]) / denominator
        last = slope_e * ((knot_e - e) / width_e)
        if self._reaches_pure_noise:
            # A segment's complement is 0 at its right knot and 1 at its left one.
            into_noise_s = (table[right_s] == 0) & (table[left_s] > 0)
            into_noise_e = (table[right_e] == 0) & (table[left_e] > 0)
            single = single * torch.where(into_noise_s, complement_s + complement_e, 1)
            first = first * torch.where(into_noise_s, complement_s + (~forward).to(s.dtype), 1)
            last = last * torch.where(into_noise_e, complement_e + forward.to(s.dtype), 1)
        # The partial segments are summed first, so that swapping the times negates the determinant exactly.
        split = (first + last) + middle
        determinant = torch.where(same, single, split)
        # At pure noise the rationalized form is 0/0; the direct form keeps the finite time derivatives.
        return torch.where((a_s == 0) & (a_e == 0), a_s * b_e - a_e * b_s, determinant)

    def _sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """Invert alpha/sigma, saturating outside the table's SNR range.

        Plateaus select their earliest knot. It inverts ``sqrt_snr`` exactly up to rounding where the latter's sigma
        denominator is not affected by ``clamp_min``.
        Into the first pure-noise entry the time is linear in alpha, so dt/dr stays finite at pure noise (r=0).
        """
        r = sqrt_snr_t
        table = self.alphas_cumprod.to(r)
        # Search in the same rounded alpha/sigma values returned at knots to select the first knot of a plateau.
        rising, target = -table.sqrt() / (1 - table).sqrt(), -r
        right = torch.searchsorted(rising, target.contiguous(), side="left").clamp(1, self.num_train_timesteps - 1)
        left = right - 1
        p, q = table[left], table[right]
        alpha, sigma = sqrt_snr_to_alpha_sigma(r)
        # Solve p(t) = r^2/(1+r^2), using signal probability on noisy segments and noise probability on
        # cleaner ones. Normalize by a table probability before dividing by the segment's change; this avoids
        # cancellation near p=1 and overflow of 1/(p-q) for subnormal tables, including in reverse gradients.
        use_alpha = p <= 0.5
        scale = torch.where(use_alpha, p, 1 - q)
        scale = torch.where(scale > 0, scale, 1)
        # Inside the table's SNR range these clamps only absorb rounding; the saturated times are set last.
        root = self.safe_clamp(torch.where(use_alpha, alpha, sigma) / scale.sqrt(), max=1.0, keep_grad=True)
        change = (p - q) / scale
        numerator = torch.where(use_alpha, 1 - root.square(), root.square() - (1 - p) / scale)
        divisor = torch.where(change > 0, change, 1)
        low, high = self._time_at_index(left, r.dtype), self._time_at_index(right, r.dtype)
        if self._reaches_pure_noise:
            # Into pure noise alpha = sqrt(p) * complement: a noisy segment's root is that complement, and a cleaner
            # one's weight is (sigma^2 - (1 - p)) / (sqrt(p) * (sqrt(p) + alpha)).
            into_noise = (q == 0) & (p > 0)
            weight = self.safe_clamp(
                numerator / torch.where(into_noise & ~use_alpha, p + alpha * p.sqrt(), divisor),
                min=0.0,
                max=1.0,
                keep_grad=True,
            )
            time = torch.where(into_noise & use_alpha, high - (high - low) * root, torch.lerp(low, high, weight))
        else:
            time = torch.lerp(low, high, self.safe_clamp(numerator / divisor, min=0.0, max=1.0, keep_grad=True))
        time = torch.where(target <= rising[0], 0, time)
        time = torch.where(target > rising[-1], (self.num_train_timesteps - 1) / self.num_train_timesteps, time)
        return time

    def v_to_x0(
        self,
        xt: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        output_dtype: OutputDtype = "same",
    ) -> torch.Tensor:
        """
        Compute the x0-prediction from the v-prediction (x0 = alpha(t) xt - sigma(t) v).
        Only AlphasNoiseSchedule has v-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The v-prediction.
            t (torch.Tensor): The time step in the forward process.
            output_dtype: Dtype of the returned prediction. ``"same"`` is ``xt``'s; ``None`` keeps it uncast.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        self._assert_t_valid(t)
        return cast_like(self._v_to_x0(xt, v, t), xt, output_dtype)

    def _v_to_x0(
        self,
        xt: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Unchecked high-precision implementation of ``v_to_x0``."""

        # Convert to high precision for calculation
        xt, v = to_high_precision(xt, v)
        t = to_high_precision(t)

        alpha_t, sigma_t = self._alpha_sigma(t)

        # Perform constraint check
        _assert_unit_alpha_sigma(alpha_t, sigma_t)

        alpha_t = expand_like(alpha_t.to(xt.dtype), xt)
        sigma_t = expand_like(sigma_t.to(xt.dtype), xt)

        # Perform calculation
        x0 = xt.mul(alpha_t) - v.mul(sigma_t)

        return x0

    def x0_to_v(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
        output_dtype: OutputDtype = "same",
    ) -> torch.Tensor:
        """
        Compute the v-prediction from the x0-prediction (v = (alpha(t) xt - x0) / sigma(t)).
        Only AlphasNoiseSchedule has v-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.
            output_dtype: Dtype of the returned prediction. ``"same"`` is ``xt``'s; ``None`` keeps it uncast.

        Returns:
            torch.Tensor: The v-prediction.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        self._assert_t_valid(t)
        return cast_like(self._x0_to_v(xt, x0, t), xt, output_dtype)

    def _x0_to_v(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Unchecked high-precision implementation of ``x0_to_v``."""

        # Convert to high precision for calculation
        xt, x0 = to_high_precision(xt, x0)
        t = to_high_precision(t)

        alpha_t, sigma_t = self._alpha_sigma(t)

        # Perform constraint check
        _assert_unit_alpha_sigma(alpha_t, sigma_t)

        alpha_t = expand_like(alpha_t.to(xt.dtype), xt)
        sigma_t = expand_like(self.non_zero_clamp(sigma_t).to(xt.dtype), xt)

        # Perform calculation
        v = (xt.mul(alpha_t) - x0).div(sigma_t)

        return v

    def _cond_v(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Unchecked high-precision implementation of ``cond_v``."""
        x, eps = to_high_precision(x, eps)
        alpha_t, sigma_t = self._alpha_sigma(to_high_precision(t))
        _assert_unit_alpha_sigma(alpha_t, sigma_t)

        alpha_t = expand_like(alpha_t.to(x.dtype), x)
        sigma_t = expand_like(sigma_t.to(x.dtype), x)
        return eps.mul(alpha_t) - x.mul(sigma_t)


class DDPMNoiseSchedule(AlphasNoiseSchedule):
    """The linear DDPM schedule (Ho et al., 2020): betas from ``1e-4`` to ``0.02`` over 1000 steps, as in ADM and DiT."""

    def __init__(self, *args, **kwargs):
        betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
        super().__init__(torch.from_numpy(np.cumprod(1.0 - betas)), *args, **kwargs)


class SDNoiseSchedule(AlphasNoiseSchedule):
    def __init__(self, *args, **kwargs):
        with torch.device("cpu"):
            scheduler = hf_load(
                DDIMScheduler.from_pretrained,
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                subfolder="scheduler",
            )
        super().__init__(scheduler.alphas_cumprod, *args, **kwargs)
        del scheduler


class SDXLNoiseSchedule(AlphasNoiseSchedule):
    def __init__(self, *args, **kwargs):
        with torch.device("cpu"):
            scheduler = hf_load(
                DDIMScheduler.from_pretrained,
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="scheduler",
            )
        super().__init__(scheduler.alphas_cumprod, *args, **kwargs)
        del scheduler


class CogVideoXNoiseSchedule(AlphasNoiseSchedule):
    def __init__(self, *args, snr_shift_scale: float, **kwargs):
        """Initialize the CogVideoX noise schedule.

        Args:
            snr_shift_scale: The scheduler config's SNR shift, the only field in which the released
                schedules differ (CogVideoX-2b: 3.0, CogVideoX-5b: 1.0).
        """

        config = {
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
            "snr_shift_scale": snr_shift_scale,
        }
        with torch.device("cpu"):
            scheduler = CogVideoXDPMScheduler.from_config(config)
        super().__init__(scheduler.alphas_cumprod, *args, **kwargs)
        del scheduler


class ShiftedRFNoiseSchedule(BaseNoiseSchedule):
    """Shifted Rectified Flow noise schedule: x_t = (1 - sigma(t)) x_0 + sigma(t) noise.

    ``sigma(t)`` is the exponential ``time_shift`` of ``t`` over ``[0, max_t]``, which is ``t`` itself at
    ``shift = 1``, and the network sees ``num_train_timesteps * sigma(t)``. Convention: t=0 is data, t=1 is noise.
    """

    def __init__(
        self,
        min_t: float = 0.0,
        max_t: float = 0.999,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            num_train_timesteps: Scale of the time the network sees, as in the diffusers flow-matching schedulers.
            shift: Exponential shift of this schedule's own time against the shared one, so that
                ``sigma(t) = time_shift(t, shift, max_t=max_t)`` rather than ``t``.  ``1.0`` is ``RFNoiseSchedule``.
        """
        super().__init__(min_t, max_t, **kwargs)
        self.num_train_timesteps = self._validate_num_steps(num_train_timesteps, "num_train_timesteps")
        if max_t > 1.0:
            raise ValueError(f"RF max_t must be at most 1.0, got {max_t}")
        if shift <= 0:
            raise ValueError(f"schedule shift must be positive, got {shift}")
        self.shift = float(shift)

    @property
    def is_unwarped(self) -> bool:
        """Whether ``shift = 1``, the identity warp, so this schedule's own time is the shared time."""
        return self.shift == 1.0

    def _warp(self, t: torch.Tensor) -> torch.Tensor:
        """This schedule's own time for the shared time *t*; the identity at ``shift = 1``."""
        return time_shift(t, self.shift, max_t=self.max_t)

    def _warp_prime(self, t: torch.Tensor) -> torch.Tensor:
        """``d/dt`` of `_warp` -- ``s M^2 / (s t + (M - t))^2``, and ``1`` at ``shift = 1``."""
        if self.is_unwarped:
            return torch.ones_like(t)
        return self.shift * self.max_t**2 / (self.shift * t + (self.max_t - t)) ** 2

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.num_train_timesteps * self._warp(t)

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - self._warp(t)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self._warp(t)

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return -self._warp_prime(t)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return self._warp_prime(t)

    def _unwarp(self, value: torch.Tensor) -> torch.Tensor:
        """The shared time whose own time is *value*; the warps form a group, so ``f_(1/shift)``."""
        return time_shift(value, 1.0 / self.shift, max_t=self.max_t)

    def step(
        self,
        xt: torch.Tensor,
        model_output: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        pred_type: str = "x0",
        sample_type: str | float = "ode",
        clean_end: bool = True,
    ) -> torch.Tensor:
        """The sampler step, where the DDIM step on a flow prediction is ``xt + (sigma(t_next) - sigma(t)) (eps - x0)``.

        That is the Euler step in sigma (and in t at ``shift = 1``), taken without converting to x0. As in the
        base step, with ``clean_end`` ``t_next = 0`` is the clean end and is accepted below ``min_t``; RF's own
        time 0 is clean (``sigma(0) = 0``), so the step lands on the x0 the flow implies either way.
        """
        check_sample_type(sample_type)
        if pred_type != "flow" or sample_type not in ("ode", 0):
            return super().step(
                xt, model_output, t, t_next, pred_type=pred_type, sample_type=sample_type, clean_end=clean_end
            )
        self._assert_t_valid(t)
        self._assert_t_valid(torch.where(t_next == 0, t, t_next) if clean_end else t_next, "t_next")
        original_dtype = xt.dtype
        xt, model_output = to_high_precision(xt, model_output)
        t, t_next = to_high_precision(t, t_next)
        # ``flow = dx/dt = sigma'(t) (eps - x0)``
        delta = expand_like((self._determinant(t, t_next) / self._warp_prime(t)).to(xt.dtype), xt)
        x_next = xt + delta * model_output
        return x_next.to(original_dtype)

    def _determinant(self, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        """``alpha = 1 - sigma`` reduces the determinant to the sigma increment ``sigma(t_end) - sigma(t_start)``.

        For the exponential warp, that rational difference simplifies to
        ``shift * max_t**2 * (t_end - t_start) / (D_start * D_end)`` with
        ``D(t) = shift * t + (max_t - t)``, `time_shift`'s denominator, free of subtractive cancellation;
        the identity warp keeps the direct time difference bit-exact.
        """
        if self.is_unwarped:
            return t_end - t_start
        denominator_start = self.shift * t_start + (self.max_t - t_start)
        denominator_end = self.shift * t_end + (self.max_t - t_end)
        return self.shift * self.max_t**2 * (t_end - t_start) / (denominator_start * denominator_end)

    def _points_to_x0(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        clean_end: bool = True,
    ) -> torch.Tensor:
        """Cancellation-resistant RF specialization of two-point inversion."""
        # Convert to high precision for calculation
        x_start, x_end = to_high_precision(x_start, x_end)
        t_start, t_end = to_high_precision(t_start, t_end)

        # A clean pair divides by 1, as in the base inversion.
        is_clean = t_end == 0 if clean_end else torch.zeros_like(t_end, dtype=torch.bool)
        delta_sigma = self._determinant(t_start, t_end)
        delta_sigma = expand_like(torch.where(is_clean, 1.0, delta_sigma).to(x_start.dtype), x_start)
        sigma_start = expand_like(self._sigma(t_start).to(x_start.dtype), x_start)
        x0 = x_start - sigma_start * (x_end - x_start) / delta_sigma

        return torch.where(expand_like(is_clean, x0), x_end, x0)

    def _sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """Solving ``sqrt_snr = (1 - sigma) / sigma`` gives this schedule's own time ``sigma = 1 / (sqrt_snr + 1)``,
        which the warp then takes back to the shared time. At ``shift = 1`` the two coincide; next to ``max_t`` the
        inverse warp multiplies rounding by the shift."""
        return self._unwarp(1 / (sqrt_snr_t + 1))

    def sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Find the timestep whose noise scale is *sigma*, the inverse of ``sigma``.

        The caller passes a noise scale of a time in range, up to rounding; the time is clamped to the range as in
        ``sqrt_snr_to_t``.

        Args:
            sigma: The noise scale at time `t`.

        Returns:
            torch.Tensor: The timestep, in the dtype of *sigma*.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        t = self._unwarp(to_high_precision(sigma)).to(sigma.dtype)
        return self.safe_clamp(t, min=self.min_t, max=self.max_t, keep_grad=True)

    def _flow_to_x0(
        self,
        xt: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the x0-prediction from the flow-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The flow-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        # Convert to high precision for calculation
        xt, v = to_high_precision(xt, v)
        t = to_high_precision(t)

        # ``sigma / sigma'`` rather than ``t``.  ``flow`` is ``dx/dt``, which for rectified flow is
        # ``sigma'(t) (eps - x0)``, and ``x_t - x0`` is ``sigma (eps - x0)`` -- so the two differ by
        # exactly this ratio.  Both factors reduce to the plain ``sigma`` at ``shift = 1``.
        scale = expand_like((self._sigma(t) / self.non_zero_clamp(self._sigma_prime(t))).to(xt.dtype), xt)

        # Perform calculation
        x0 = xt - v.mul(scale)

        return x0

    def _cond_velocity(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """``dx/dt = sigma'(t) (eps - x0)``, the base formula in closed form.

        Kept as an override for the same reasons the other conversions are: the base builds this
        from ``alpha'(t) x0 + sigma'(t) eps``, two multiplies and an add, whereas rectified flow's
        ``alpha' = -sigma'`` collapses it to one multiply of a difference.  It also short-circuits
        exactly at ``shift = 1``, where ``sigma'`` is identically one and the target is ``eps - x0``
        untouched -- which is what keeps ``RFNoiseSchedule`` bit-identical.
        """
        # Convert to high precision for calculation
        x, eps = to_high_precision(x, eps)
        t = to_high_precision(t)

        # Perform calculation
        velocity = eps - x
        if not self.is_unwarped:
            velocity = velocity * expand_like(self._sigma_prime(t).to(x.dtype), velocity)

        return velocity

    def _x0_to_flow(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the flow-prediction from the x0-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            x0 (torch.Tensor): The x0-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The flow-prediction.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        # Convert to high precision for calculation
        xt, x0 = to_high_precision(xt, x0)
        t = to_high_precision(t)

        # ``sigma' / sigma`` rather than ``1 / t``, the inverse of ``flow_to_x0``'s ratio above.
        scale = expand_like((self._sigma_prime(t) / self.non_zero_clamp(self._sigma(t))).to(xt.dtype), xt)

        # Perform calculation
        flow = (xt - x0).mul(scale)

        return flow


class RFNoiseSchedule(ShiftedRFNoiseSchedule):
    """Rectified Flow noise schedule: x_t = (1 - t) x_0 + t noise, and the network sees ``num_train_timesteps * t``.

    The unshifted ``ShiftedRFNoiseSchedule``, without a ``shift`` parameter, so networks that embed the raw time t
    can rely on ``sigma(t) = t``. Convention: t=0 is data, t=1 is noise.
    """

    def __init__(self, min_t: float = 0.0, max_t: float = 0.999, num_train_timesteps: int = 1000, **kwargs):
        super().__init__(min_t, max_t, num_train_timesteps, shift=1.0, **kwargs)


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
        **kwargs,
    ):
        super().__init__(min_t, max_t, **kwargs)
        if max_t > torch.pi / 2:
            raise ValueError(f"Trig max_t must be at most pi/2, got {max_t}")

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        # In this parameterization, t acts directly as the angle.
        return t

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(t)

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.sin(t)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t)

    def _determinant(self, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        """``cos(t_start) sin(t_end) - cos(t_end) sin(t_start)`` is ``sin(t_end - t_start)``."""
        return torch.sin(t_end - t_start)

    def _sqrt_snr(self, t: torch.Tensor) -> torch.Tensor:
        """``sqrt_snr(t) = cos(t) / sin(t) = cot(t)``."""
        # We clamp tan(t) to avoid dividing by zero at t=0
        return 1.0 / self.non_zero_clamp(torch.tan(t))

    def _sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        """``t = arccot(sqrt_snr)``, as its robust equivalent ``atan2(1, sqrt_snr)``."""
        return torch.atan2(torch.ones_like(sqrt_snr_t), sqrt_snr_t)

    def _flow_to_x0(
        self,
        xt: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute x0 from flow prediction v.
        v_t = -sin(t)x0 + cos(t)eps
        x_t = cos(t)x0 + sin(t)eps

        Algebraic rearrangement yields:
        x0 = cos(t)x_t - sin(t)v
        """
        # Convert to high precision for calculation
        xt, v = to_high_precision(xt, v)
        t = to_high_precision(t)

        cos_t = expand_like(torch.cos(t).to(xt.dtype), xt)
        sin_t = expand_like(torch.sin(t).to(xt.dtype), xt)

        # Perform calculation
        x0 = xt * cos_t - v * sin_t

        return x0

    def _x0_to_flow(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow prediction v from x0.
        v = (cos(t)x_t - x_0) / sin(t)
        """
        # Convert to high precision for calculation
        xt, x0 = to_high_precision(xt, x0)
        t = to_high_precision(t)

        cos_t = expand_like(torch.cos(t).to(xt.dtype), xt)
        sin_t = expand_like(self.non_zero_clamp(torch.sin(t)).to(xt.dtype), xt)

        # Perform calculation
        flow = (xt * cos_t - x0) / sin_t

        return flow


class CompositeNoiseSchedule(BaseNoiseSchedule):
    """Applies independent per-modality noise schedules to dict-valued inputs.

    Each entry in *schedules* maps a modality key (e.g. ``"video"``,
    ``"audio"``) to a dict ``{"name": <schedule_name>, **schedule_kwargs}``.
    Methods that accept data tensors (``forward_process``, ``latents``,
    ``x0_to_eps``, …) require ``Dict[str, Tensor]`` inputs and dispatch
    each key to the corresponding sub-schedule.  All modalities share the
    same timestep ``t`` and time range.

    Per-modality queries (``alpha``, ``sigma``, ``rescale_t``, …) return
    dicts keyed by modality.  Time-only queries (``sample_t``, ``grid``)
    delegate to the **primary** sub-schedule (configurable via
    *primary_key*; defaults to the first key), and the composite uses that
    schedule's ``student_grid_cfg``.  Individual sub-schedules are accessible
    via ``self.subschedules[key]``.


    Example::

        composite = CompositeNoiseSchedule(
            schedules={
                "video": {"name": "rf", "min_t": 0.0, "max_t": 0.999},
                "audio": {"name": "shifted_rf", "shift": 3.0, "min_t": 0.0, "max_t": 0.999},
            },
            primary_key="video",
        )
        t = composite.sample_t(batch_size)   # one Tensor
        x_t = composite.forward_process({"video": x_v, "audio": x_a}, {"video": e_v, "audio": e_a}, t)
    """

    def __init__(
        self,
        schedules: dict[str, dict],
        primary_key: Optional[str] = None,
    ):
        instantiated = {
            k: get_noise_schedule(v["name"], **{kk: vv for kk, vv in v.items() if kk != "name"})
            for k, v in schedules.items()
        }

        if primary_key is not None:
            if primary_key not in instantiated:
                raise KeyError(
                    f"primary_key '{primary_key}' not in schedules. " f"Available keys: {list(instantiated.keys())}"
                )
        else:
            primary_key = next(iter(instantiated))

        primary = instantiated[primary_key]

        super().__init__(
            min_t=primary._min_t,
            max_t=primary._max_t,
            clamp_min=primary.clamp_min,
            student_grid_cfg=primary.student_grid_cfg,
        )
        self.t_precision = primary.t_precision

        # Underlying ModuleDict is exposed via the ``schedules`` property
        # below; instance attribute uses a different name so the property
        # on ``BaseNoiseSchedule`` (returning ``self``) can be cleanly
        # overridden here without a property/instance-attr collision.
        self.subschedules = torch.nn.ModuleDict(instantiated)
        self.primary = self.subschedules[primary_key]

        for k, sched in self.subschedules.items():
            if sched.t_precision != primary.t_precision:
                raise ValueError(
                    f"Sub-schedule '{k}' has t_precision={sched.t_precision}, but primary "
                    f"'{primary_key}' has t_precision={primary.t_precision}. Composite schedules "
                    "share one timestep tensor, so all sub-schedules must use the same t_precision."
                )
            if sched._min_t != primary._min_t or sched._max_t != primary._max_t:
                raise ValueError(
                    f"Sub-schedule '{k}' has time range [{sched._min_t}, {sched._max_t}], "
                    f"but primary has [{primary._min_t}, {primary._max_t}]. "
                    "All sub-schedules must share the same time range."
                )

    def reset_parameters(self) -> None:
        for schedule in self.subschedules.values():
            schedule.reset_parameters()

    @property
    def schedules(self) -> torch.nn.ModuleDict:
        """Per-modality dispatch view (used as ``ns.schedules`` arg to
        ``tree_utils.per_modality``).  Overrides ``BaseNoiseSchedule``'s
        property (which returns ``self``) so callers can dispatch
        uniformly without an ``isinstance`` check."""
        return self.subschedules

    def _per_schedule(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call *method* on every sub-schedule, with the structure coming from the **schedules**.

        The counterpart to `_per_modality`, which is driven by the *data*.  These take the
        shared time, which is one Tensor rather than a per-modality dict, so the modalities have to
        supply the structure instead.  Dict arguments are still indexed per key.
        """
        return per_modality(
            lambda sched, *a, **kw: getattr(sched, method)(*a, **kw), self.subschedules, *args, **kwargs
        )

    def alpha(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._per_schedule("alpha", t)

    def sigma(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._per_schedule("sigma", t)

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        pairs = self._per_schedule("alpha_sigma", t)
        return index(pairs, 0), index(pairs, 1)

    def alpha_prime(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._per_schedule("alpha_prime", t)

    def sigma_prime(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._per_schedule("sigma_prime", t)

    def alpha_start(self, t_start: torch.Tensor, clean_end: bool = True) -> Dict[str, torch.Tensor]:
        return self._per_schedule("alpha_start", t_start, clean_end=clean_end)

    def transition_std(
        self,
        t: torch.Tensor,
        t_start: torch.Tensor,
        clean_end: bool = True,
    ) -> Dict[str, torch.Tensor]:
        return self._per_schedule("transition_std", t, t_start, clean_end=clean_end)

    def determinant(
        self,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        clean_end: bool = True,
    ) -> Dict[str, torch.Tensor]:
        return self._per_schedule("determinant", t_start, t_end, clean_end=clean_end)

    def rescale_t(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """The scale each modality's model input is conditioned on, from the one shared time.

        A fan-out like the rest, since each sub-schedule may scale differently and carries its own
        shift; delegating to the primary would put every modality on video's noise level.
        """
        return self._per_schedule("rescale_t", t)

    def sqrt_snr(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._per_schedule("sqrt_snr", t)

    # ------------------------------------------------------------------
    # Draws. The time is shared, so these are the primary's; the schedule grid
    # likewise delegates to the primary clock.
    # ------------------------------------------------------------------

    def sample_t(self, *args, **kwargs) -> torch.Tensor:
        """Draw the shared training time; each modality's noise level follows from its own shift."""
        return self.primary.sample_t(*args, **kwargs)

    def sample_t_icdf(self, *args, **kwargs) -> torch.Tensor:
        """The quantiles of the shared training time, as drawn by ``sample_t``."""
        return self.primary.sample_t_icdf(*args, **kwargs)

    def grid(
        self,
        num_steps: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return the shared grid on the primary schedule's clock."""
        return self.primary.grid(num_steps=num_steps, device=device, dtype=dtype)

    def _grid(self, num_steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """The primary schedule's grid before rounding, which native student grids shift."""
        return self.primary._grid(num_steps, device)

    # Ambiguous with several sub-schedules: each reaches a given sqrt-SNR at a different shared
    # time, because each warps that time by its own shift.
    def sqrt_snr_to_t(self, sqrt_snr_t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Ambiguous with multiple schedules; use subschedules[key].sqrt_snr_to_t()")

    # ------------------------------------------------------------------
    # Per-modality dispatch helper (delegates to canonical ``per_modality``)
    # ------------------------------------------------------------------

    def _per_modality(self, method: str, first_arg: Any, *args: Any, **kwargs: Any) -> Any:
        """Call *method* per-modality on each sub-schedule.

        Drives ``per_modality`` by *first_arg* (the data) and passes
        ``self.subschedules`` (a ``ModuleDict``) as a regular arg that
        gets indexed per-key by ``per_modality``'s standard rule.
        ``args`` / ``kwargs`` that are dicts are indexed per-key;
        everything else is broadcast.

        Bare-Tensor *first_arg* is rejected with ``TypeError``: a
        ``CompositeNoiseSchedule`` is only meaningful for multimodal data,
        and single-modality slices should call
        ``self.subschedules[key].{method}(tensor, ...)`` directly rather
        than route through the dispatcher.
        """
        if isinstance(first_arg, torch.Tensor):
            raise TypeError(
                f"CompositeNoiseSchedule.{method}() requires dict-shaped data input. "
                f"For a single-modality slice, call "
                f"``self.subschedules[key].{method}(tensor, ...)`` directly "
                f"with `key` from {sorted(self.subschedules.keys())}."
            )
        return per_modality(
            lambda x, sched, *a, **kw: getattr(sched, method)(x, *a, **kw),
            first_arg,
            self.subschedules,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Dict-aware overrides for data-accepting methods
    # ------------------------------------------------------------------

    def forward_process(self, x, eps, t, t_start=None, clean_end=True):
        return self._per_modality("forward_process", x, eps, t, t_start=t_start, clean_end=clean_end)

    def step(self, xt, model_output, t, t_next, pred_type="x0", sample_type="ode", clean_end=True):
        return self._per_modality(
            "step", xt, model_output, t, t_next, pred_type=pred_type, sample_type=sample_type, clean_end=clean_end
        )

    def latents(self, noise, t_init):
        return self._per_modality("latents", noise, t_init=t_init)

    def cond_velocity(self, x, eps, t, output_dtype="same"):
        return self._per_modality("cond_velocity", x, eps, t, output_dtype=output_dtype)

    def cond_v(self, x, eps, t, output_dtype="same"):
        return self._per_modality("cond_v", x, eps, t, output_dtype=output_dtype)

    def x0_to_eps(self, xt, x0, t, output_dtype="same"):
        return self._per_modality("x0_to_eps", xt, x0, t, output_dtype=output_dtype)

    def eps_to_x0(self, xt, eps, t, output_dtype="same"):
        return self._per_modality("eps_to_x0", xt, eps, t, output_dtype=output_dtype)

    def flow_to_x0(self, xt, v, t, output_dtype="same"):
        return self._per_modality("flow_to_x0", xt, v, t, output_dtype=output_dtype)

    def points_to_x0(self, x_start, x_end, t_start, t_end, clean_end=True, output_dtype="same"):
        return self._per_modality(
            "points_to_x0", x_start, x_end, t_start, t_end, clean_end=clean_end, output_dtype=output_dtype
        )

    def x0_to_flow(self, xt, x0, t, output_dtype="same"):
        return self._per_modality("x0_to_flow", xt, x0, t, output_dtype=output_dtype)

    def convert_model_output(
        self,
        xt,
        model_output,
        t,
        src_pred_type="x0",
        target_pred_type="eps",
        output_dtype="same",
    ):
        return self._per_modality(
            "convert_model_output",
            xt,
            model_output,
            t,
            src_pred_type=src_pred_type,
            target_pred_type=target_pred_type,
            output_dtype=output_dtype,
        )


# Global registry of noise schedules for easy access by string name
NOISE_SCHEDULES = {
    "edm": EDMNoiseSchedule,
    "alphas": AlphasNoiseSchedule,
    "ddpm": DDPMNoiseSchedule,
    "sd": SDNoiseSchedule,
    "sdxl": SDXLNoiseSchedule,
    "cogvideox": CogVideoXNoiseSchedule,
    "rf": RFNoiseSchedule,
    "shifted_rf": ShiftedRFNoiseSchedule,
    "trig": TrigNoiseSchedule,
}


def get_noise_schedule(name: str, **kwargs):
    """
    Instantiate the noise schedule registered under *name*.

    Args:
        name (str): Name of the noise schedule. Available options:
            - "edm": EDMNoiseSchedule (Karras et al. 2022)
            - "alphas": AlphasNoiseSchedule (a given ``alphas_cumprod`` table)
            - "ddpm": DDPMNoiseSchedule (linear betas 1e-4..0.02; ADM, DiT)
            - "sd": SDNoiseSchedule (Stable Diffusion v1.5)
            - "sdxl": SDXLNoiseSchedule (Stable Diffusion XL)
            - "cogvideox": CogVideoXNoiseSchedule (CogVideoX)
            - "rf": RFNoiseSchedule (Rectified Flow)
            - "shifted_rf": ShiftedRFNoiseSchedule (Rectified Flow under an exponential time shift)
            - "trig": TrigNoiseSchedule
            Multimodal networks build a ``CompositeNoiseSchedule`` of these directly.
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
