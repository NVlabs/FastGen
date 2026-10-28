# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-time distributions behind ``BaseNoiseSchedule.sample_t``.

The bounds ``[min_t, max_t]``, numbers or per-sample tensors, define each distribution: ``uniform``, ``logitnormal``,
``shifted`` and ``shifted_logitnormal`` are drawn on ``[0, 1]`` and mapped onto the bounds, and ``lognormal``, ``log_t``
and ``polynomial`` are distributions over time truncated to them. Each has an inverse CDF on its bounds, ``icdf``,
through which it draws, except the logit-normal and shifted ones, which sample directly. Every draw comes from a torch
generator, so ``torch.manual_seed`` and ``torch.random.fork_rng`` govern all of them.

Every distribution computes in its ``precision``, float64: the user-facing methods ``sample``, ``icdf`` and ``cdf``
cast their tensor operands to it, assert that the private hook they call returns in it, and cast the result once to
their ``dtype``, by default the precision itself. The private hooks compute and return in the precision (``LogT``'s
compute in SciPy's float64 and round to it), while ``time_shift`` and ``polynomial_spacing`` compute in the dtype of
their inputs, so that their callers choose it. ``BaseNoiseSchedule.sample_t`` asks for ``t_precision``, so a coarse
time precision rounds the times but not the distribution.

A window ``[lo, hi]``, intersected with the bounds, conditions the distribution they define, the same way for every
distribution: one mapped from ``[0, 1]`` draws between its CDF's values at the window's ends and inverts the CDF,
and a truncated one is truncated to the window, as narrower bounds would truncate it. For the shift, the
conditioned draw is again a shifted draw, over the window, with the shift ``D(lo) / D(hi)``, where
``D(y) = shift - (shift - 1) y`` and ``y`` is the time normalized to the bounds. The logit-normal, lognormal and log-t
CDFs round to 1 short of the upper bound, so where a window or truncation starts in that tail (``_upper_tail``), they
draw between survival probabilities instead, which keep their precision there.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TypeVar

import numpy as np
import torch
from scipy import stats

from fastgen.utils.tensor import as_array, as_tensor, assert_tensor
from fastgen.utils.tree_utils import assert_same, to

Bound = float | torch.Tensor
Values = TypeVar("Values", float, np.ndarray, torch.Tensor)


def check_bounds(
    values: float | np.ndarray | torch.Tensor | Tuple[int, ...],
    min_t: Optional[Bound | np.ndarray],
    max_t: Optional[Bound | np.ndarray],
    window: Optional[Tuple[Optional[Bound], Optional[Bound]]] = None,
) -> None:
    """Raise unless *min_t*, *max_t* and the ends of *window* are each a number or an array of the kind of *values*, a
    NumPy array or a tensor, that is 0-d or has their shape, so that no bound changes the kind or the shape of the
    result. A shape stands for tensor values still to be drawn, and number *values* take number bounds only. Any
    ``None`` among them, a bound not yet defaulted or an end of no window, passes."""
    if isinstance(values, tuple) or torch.is_tensor(values):
        kind = torch.Tensor
    elif isinstance(values, np.ndarray):
        kind = np.ndarray
    else:
        kind = None
    shape = values if isinstance(values, tuple) else getattr(values, "shape", None)
    start, end = (None, None) if window is None else window
    for name, bound in (("min_t", min_t), ("max_t", max_t), ("window start", start), ("window end", end)):
        if not isinstance(bound, (np.ndarray, torch.Tensor)):
            continue
        if kind is None:
            raise ValueError(f"{name} must be a number for number values, got {type(bound).__name__}")
        if not isinstance(bound, kind):
            raise ValueError(f"{name} must be a number or a {kind.__name__}, got {type(bound).__name__}")
        if bound.ndim > 0 and bound.shape != shape:
            expected = ", ".join(map(str, shape))
            raise ValueError(f"Per-sample {name} must have shape [{expected}], got {tuple(bound.shape)}")


def checked_bounds(
    shape: Tuple[int, ...],
    min_t: Optional[Bound],
    max_t: Optional[Bound],
    window: Optional[Tuple[Optional[Bound], Optional[Bound]]] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[Optional[Bound], ...]:
    """``(min_t, max_t, *window)``, checked for tensor values of *shape* (``check_bounds``), with the tensors among
    them in *dtype*, float64 by default; numbers and ``None`` pass through."""
    check_bounds(shape, min_t, max_t, window)
    return to((min_t, max_t, *(() if window is None else window)), dtype=dtype)


def larger_bound(a: Bound, b: Bound) -> Bound:
    """The larger of two bounds, numbers or tensors, without building a device tensor from a number."""
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.maximum(a, b)
    if torch.is_tensor(a) or torch.is_tensor(b):
        tensor, number = (a, b) if torch.is_tensor(a) else (b, a)
        return tensor.clamp(min=number)
    return max(a, b)


def smaller_bound(a: Bound, b: Bound) -> Bound:
    """The smaller of two bounds, numbers or tensors, without building a device tensor from a number."""
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.minimum(a, b)
    if torch.is_tensor(a) or torch.is_tensor(b):
        tensor, number = (a, b) if torch.is_tensor(a) else (b, a)
        return tensor.clamp(max=number)
    return min(a, b)


def time_shift(
    t: Values,
    shift: float = 1.0,
    max_t: float | Values = 1.0,
    min_t: float | Values = 0.0,
) -> Values:
    """Exponential time shift ``f_s(t) = s*t / (1 + (s-1)*t)``, over ``[min_t, max_t]``.

    The warp is a one-parameter multiplicative group (``f_a . f_b = f_{a*b}``, ``f_1``
    the identity) and it fixes both endpoints, ``f_s(min_t) = min_t`` and ``f_s(max_t) = max_t``,
    also in floating point, where it maps ``[min_t, max_t]`` into itself.
    *t* may be a number, a NumPy array or a tensor, and the result is the same kind. Each bound is a number or, for an
    array *t*, an array of its kind that is 0-d or has its shape (``check_bounds``). Callers ensure ``min_t < max_t``.
    """
    if shift <= 0:
        raise ValueError(f"time shift must be positive, got {shift}")
    check_bounds(t, min_t, max_t)
    if shift == 1.0:
        # The algebra already reduces to ``(t * max_t) / max_t`` here, which is the identity
        # numerically but not bit-exactly for every value.  Every unimodal rectified-flow network
        # takes this path on every step, so hand ``t`` straight back rather than move it by an ulp.
        return t
    # ``min_t + span * f_s((t - min_t) / span)``, taken from the end nearer the result: its distance above ``min_t``
    # or below ``max_t`` is a product of nonnegative terms, so that end is kept exactly and never crossed. The
    # denominator adds nonnegative terms; the expanded ``span + (shift - 1) * offset`` cancels for a shift below 1, as
    # in every inverse warp, which multiplies that error by the shift next to ``max_t``.
    span = max_t - min_t
    offset = t - min_t
    denominator = shift * offset + (span - offset)
    above_min = offset * shift * span / denominator
    below_max = span * (span - offset) / denominator
    nearer_min = above_min <= below_max
    if torch.is_tensor(t):
        return torch.where(nearer_min, min_t + above_min, max_t - below_max)
    if isinstance(t, np.ndarray):
        return np.where(nearer_min, min_t + above_min, max_t - below_max)
    return min_t + above_min if nearer_min else max_t - below_max


def polynomial_spacing(u: Values, rho: float, min_t: float | Values, max_t: float | Values) -> Values:
    """The polynomial ``(min_t**(1/rho) + u * (max_t**(1/rho) - min_t**(1/rho)))**rho``: the ``rho``-th power of the
    line between the bounds' ``1/rho`` powers. The EDM paper (Karras et al. 2022) spaces its sampling steps this way,
    with ``rho = 7``.

    It maps positions ``u`` in ``[0, 1]``, not times, onto ``[min_t, max_t]``: ``u = 0`` lands on ``min_t``, not 0.
    ``rho = 1`` keeps evenly spaced positions evenly spaced, and a larger ``rho`` moves every time toward ``min_t``,
    approaching the geometric spacing ``min_t**(1-u) * max_t**u``. *u* may be a number, a NumPy array or a tensor,
    and the result is the same kind. Each bound is a number or, for an array *u*, an array of its kind that is 0-d
    or has its shape (``check_bounds``).
    """
    check_bounds(u, min_t, max_t)
    low_root, high_root = min_t ** (1 / rho), max_t ** (1 / rho)
    return (low_root + u * (high_root - low_root)) ** rho


def _normal_cdf(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """The CDF of ``N(mean, std^2)`` at *x*. ``torch.special.ndtr`` cancels in the lower tail (2% off at z = -8 and 0
    from z = -9), where erfc keeps its relative precision, so a window there still inverts the CDF accurately."""
    return 0.5 * torch.erfc((mean - x) / (std * math.sqrt(2)))


def _normal_sf(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """The survival function ``1 - _normal_cdf`` of ``N(mean, std^2)`` at *x*, which keeps its relative precision in the
    upper tail, where the CDF rounds to 1."""
    return 0.5 * torch.erfc((x - mean) / (std * math.sqrt(2)))


def _upper_tail(dtype: torch.dtype) -> float:
    """The survival probability below which a CDF value in *dtype*, near 1, keeps less than half the bits of the mass
    above it: windows and truncations that start there are drawn between survival probabilities."""
    return math.sqrt(torch.finfo(dtype).eps)


def _device_of(*operands: Optional[Bound], default: torch.device | str) -> torch.device | str:
    """The device of the first tensor among *operands*, or *default* if there is none."""
    return next((operand.device for operand in operands if torch.is_tensor(operand)), default)


class TimeDistribution(ABC):
    """A distribution of training times on ``[min_t, max_t]``, which a window can condition, computed in
    ``precision``."""

    precision: torch.dtype = torch.float64

    def sample(
        self,
        n: int,
        min_t: Bound,
        max_t: Bound,
        window: Optional[Tuple[Bound, Bound]] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Draw *n* times on ``[min_t, max_t]``, conditioned on *window*, if one is given, where it overlaps those
        bounds, and return them on *device* in *dtype*, by default the precision.

        Every bound and window end is a number, a 0-d tensor or one value per sample, a tensor of shape [n]
        (``check_bounds``). *device* defaults to that of the tensor bounds and window ends, and to torch's default
        device when they are all numbers.
        """
        min_t, max_t, *ends = self._sampling_bounds((n,), min_t, max_t, window)
        device = _device_of(min_t, max_t, *ends, default=torch.get_default_device()) if device is None else device
        t = self._sample_window(n, min_t, max_t, *ends, device) if ends else self._sample(n, min_t, max_t, device)
        return self._output(t, dtype)

    def icdf(self, p: torch.Tensor, min_t: Bound, max_t: Bound, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """The time on ``[min_t, max_t]`` at which the CDF of the distribution these bounds define reaches *p*, in
        *dtype*, by default the precision. Each bound is a number, a 0-d tensor or a tensor of *p*'s shape."""
        min_t, max_t = self._sampling_bounds(p.shape, min_t, max_t)
        return self._output(self._icdf(p.to(self.precision), min_t, max_t), dtype)

    def _sample(self, n: int, min_t: Bound, max_t: Bound, device: torch.device | str) -> torch.Tensor:
        """The draw on ``[min_t, max_t]``, in the precision: by default the inverse CDF of uniform probabilities."""
        return self._icdf(torch.rand(n, device=device, dtype=self.precision), min_t, max_t)

    @abstractmethod
    def _sample_window(
        self, n: int, min_t: Bound, max_t: Bound, lo: Bound, hi: Bound, device: torch.device | str
    ) -> torch.Tensor:
        """The draw on ``[min_t, max_t]`` conditioned on the window ``[lo, hi]``, in the precision."""

    @abstractmethod
    def _icdf(self, p: torch.Tensor, min_t: Bound, max_t: Bound) -> torch.Tensor:
        """``icdf`` of *p* and tensor bounds in the precision, computed and returned in it."""

    def _sampling_bounds(
        self, shape: Tuple[int, ...], min_t: Bound, max_t: Bound, window: Optional[Tuple[Bound, Bound]] = None
    ) -> Tuple[Bound, ...]:
        """``checked_bounds`` in the precision, with a window intersected with the bounds, raising unless they are in
        order: ``min_t < max_t`` and the window starts below its end. An empty interval would draw NaN under a shift
        (zero span) or times outside it (inverted ends), and a NaN end is in order with nothing. Tensor ends are
        checked asynchronously on CUDA (``assert_tensor``)."""
        min_t, max_t, *ends = checked_bounds(shape, min_t, max_t, window, dtype=self.precision)
        intervals = [(min_t, max_t, "min_t < max_t")]
        if ends:
            start, end = ends
            ends = [larger_bound(start, min_t), smaller_bound(end, max_t)]
            intervals.append((*ends, "a window with start < end that overlaps the bounds"))
        for lo, hi, requirement in intervals:
            if torch.is_tensor(lo) or torch.is_tensor(hi):
                assert_tensor((lo < hi).all(), f"A time distribution needs {requirement}", error_type=ValueError)
            elif not lo < hi:
                raise ValueError(f"A time distribution needs {requirement}, got [{lo}, {hi}]")
        return (min_t, max_t, *ends)

    def _output(self, t: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
        """*t*, which the hooks return in the precision, in *dtype*, by default the precision."""
        assert_same(dtype=self.precision, result=t)
        return t.to(self.precision if dtype is None else dtype)


class UnitIntervalDistribution(TimeDistribution):
    """A distribution drawn on ``[0, 1]`` and mapped onto the bounds; a window inverts its CDF between its ends."""

    def _sample_window(
        self, n: int, min_t: Bound, max_t: Bound, lo: Bound, hi: Bound, device: torch.device | str
    ) -> torch.Tensor:
        lo, hi = (as_tensor(end, device, self.precision) for end in (lo, hi))
        return self._conditioned(torch.rand(n, device=device, dtype=self.precision), lo, hi, min_t, max_t)

    def _conditioned(
        self, u: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, min_t: Bound, max_t: Bound
    ) -> torch.Tensor:
        """The times at the fractions *u* of the mass between the window ends *lo* and *hi*: the CDF inverted between
        their CDF values."""
        p_lo, p_hi = self._cdf(lo, min_t, max_t), self._cdf(hi, min_t, max_t)
        return self._icdf(p_lo + (p_hi - p_lo) * u, min_t, max_t)

    def cdf(self, t: torch.Tensor, min_t: Bound, max_t: Bound, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """The probability that a draw on ``[min_t, max_t]`` is at most *t*, in *dtype*, by default the precision.
        Each bound is a number, a 0-d tensor or a tensor of *t*'s shape."""
        min_t, max_t = self._sampling_bounds(t.shape, min_t, max_t)
        return self._output(self._cdf(t.to(self.precision), min_t, max_t), dtype)

    def _cdf(self, t: torch.Tensor, min_t: Bound, max_t: Bound) -> torch.Tensor:
        """``cdf`` of *t* and tensor bounds in the precision, in it."""
        return self._unit_cdf((t - min_t) / (max_t - min_t))

    def _icdf(self, p: torch.Tensor, min_t: Bound, max_t: Bound) -> torch.Tensor:
        return min_t + (max_t - min_t) * self._unit_icdf(p)

    @abstractmethod
    def _unit_cdf(self, y: torch.Tensor) -> torch.Tensor:
        """The CDF on ``[0, 1]``."""

    @abstractmethod
    def _unit_icdf(self, p: torch.Tensor) -> torch.Tensor:
        """The inverse of ``_unit_cdf``."""


class Uniform(UnitIntervalDistribution):
    """The uniform distribution on the bounds."""

    def _unit_cdf(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def _unit_icdf(self, p: torch.Tensor) -> torch.Tensor:
        return p


class LogitNormal(UnitIntervalDistribution):
    """``sigmoid(z)`` for ``z ~ N(mean, std^2)``, mapped onto the bounds."""

    def __init__(self, mean: float, std: float):
        self.mean, self.std = mean, std

    def _sample(self, n: int, min_t: Bound, max_t: Bound, device: torch.device | str) -> torch.Tensor:
        unit = torch.sigmoid(torch.randn(n, device=device, dtype=self.precision) * self.std + self.mean)
        return unit * (max_t - min_t) + min_t

    def _unit_cdf(self, y: torch.Tensor) -> torch.Tensor:
        # The clamp keeps logit finite where rounding puts y just past [0, 1]: a time in a coarser precision than the
        # bounds.
        return _normal_cdf(torch.logit(y.clamp(0.0, 1.0)), self.mean, self.std)

    def _unit_icdf(self, p: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.special.ndtri(p) * self.std + self.mean)

    def _conditioned(
        self, u: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, min_t: Bound, max_t: Bound
    ) -> torch.Tensor:
        """Windows that start in the upper tail (``_upper_tail``) are drawn between survival probabilities; the
        others invert the CDF, as for every unit-interval distribution."""
        t = super()._conditioned(u, lo, hi, min_t, max_t)
        span = max_t - min_t
        s_lo, s_hi = (self._unit_sf((end - min_t) / span) for end in (lo, hi))
        tail = min_t + span * self._unit_isf(s_lo - (s_lo - s_hi) * u)
        return torch.where(s_lo < _upper_tail(self.precision), tail, t)

    def _unit_sf(self, y: torch.Tensor) -> torch.Tensor:
        """``1 - _unit_cdf``, which keeps its precision where the CDF rounds to 1."""
        return _normal_sf(torch.logit(y.clamp(0.0, 1.0)), self.mean, self.std)

    def _unit_isf(self, s: torch.Tensor) -> torch.Tensor:
        """The inverse of ``_unit_sf``."""
        return torch.sigmoid(self.mean - torch.special.ndtri(s) * self.std)


class Shifted(UnitIntervalDistribution):
    """A uniform draw under the exponential shift ``time_shift`` over the bounds, whose ends the shift fixes."""

    def __init__(self, shift: float):
        self.shift = shift

    def _sample(self, n: int, min_t: Bound, max_t: Bound, device: torch.device | str) -> torch.Tensor:
        t = torch.rand(n, device=device, dtype=self.precision) * (max_t - min_t) + min_t
        return time_shift(t, self.shift, min_t=min_t, max_t=max_t)

    def _unit_cdf(self, y: torch.Tensor) -> torch.Tensor:
        return time_shift(y, 1.0 / self.shift)

    def _unit_icdf(self, p: torch.Tensor) -> torch.Tensor:
        return time_shift(p, self.shift)


class ShiftedLogitNormal(LogitNormal):
    """A logit-normal draw under the exponential shift over the bounds, as for ``Shifted``."""

    def __init__(self, mean: float, std: float, shift: float):
        super().__init__(mean, std)
        self.shift = shift

    def _sample(self, n: int, min_t: Bound, max_t: Bound, device: torch.device | str) -> torch.Tensor:
        return time_shift(super()._sample(n, min_t, max_t, device), self.shift, min_t=min_t, max_t=max_t)

    def _unit_cdf(self, y: torch.Tensor) -> torch.Tensor:
        return super()._unit_cdf(time_shift(y, 1.0 / self.shift))

    def _unit_icdf(self, p: torch.Tensor) -> torch.Tensor:
        return time_shift(super()._unit_icdf(p), self.shift)

    def _unit_sf(self, y: torch.Tensor) -> torch.Tensor:
        return super()._unit_sf(time_shift(y, 1.0 / self.shift))

    def _unit_isf(self, s: torch.Tensor) -> torch.Tensor:
        return time_shift(super()._unit_isf(s), self.shift)


class TruncatedTimeDistribution(TimeDistribution):
    """A distribution over time truncated to the bounds; a window truncates it further, as narrower bounds would."""

    def _sample_window(
        self, n: int, min_t: Bound, max_t: Bound, lo: Bound, hi: Bound, device: torch.device | str
    ) -> torch.Tensor:
        return self._sample(n, lo, hi, device)


class LogNormal(TruncatedTimeDistribution):
    """``exp(z)`` for ``z ~ N(mean, std^2)``, truncated to the bounds, the lower one at least ``clamp_min``."""

    def __init__(self, mean: float, std: float, clamp_min: float):
        self.mean, self.std, self.clamp_min = mean, std, clamp_min

    def _sample(self, n: int, min_t: Bound, max_t: Bound, device: torch.device | str) -> torch.Tensor:
        # Scalar bounds are computed on CPU for numerical stability, then moved to device.
        # Per-sample bounds are computed on their own device: copying them to the host would synchronize it, while
        # filling a tensor from a number does not.
        compute_device = _device_of(min_t, max_t, default="cpu")
        t = self._icdf(torch.rand(n, device=compute_device, dtype=self.precision), min_t, max_t)
        return t.to(device=device)

    def _icdf(self, p: torch.Tensor, min_t: Bound, max_t: Bound) -> torch.Tensor:
        """The normal's quantile, exponentiated, at the fraction *p* between its CDF values at the log-bounds, or
        between its survival probabilities there where the lower one lies in the upper tail (``_upper_tail``); on the
        device of *p*."""
        log_min_t = as_tensor(min_t, p.device, self.precision).clamp(min=self.clamp_min).log()
        log_max_t = as_tensor(max_t, p.device, self.precision).log()
        cdf_min, cdf_max = _normal_cdf(log_min_t, self.mean, self.std), _normal_cdf(log_max_t, self.mean, self.std)
        sf_min, sf_max = _normal_sf(log_min_t, self.mean, self.std), _normal_sf(log_max_t, self.mean, self.std)
        z = torch.where(
            sf_min < _upper_tail(self.precision),
            -torch.special.ndtri(sf_min - p * (sf_min - sf_max)),
            torch.special.ndtri(p * (cdf_max - cdf_min) + cdf_min),
        )
        return (z * self.std + self.mean).exp()


class LogT(TruncatedTimeDistribution):
    """``exp(mean + std * x)`` for Student-t ``x`` with ``df`` degrees of freedom, truncated to the bounds, the lower
    one at least ``clamp_min``. SciPy computes it on the host in float64, which the hooks round to the precision."""

    def __init__(self, mean: float, std: float, df: float, clamp_min: float):
        self.mean, self.std, self.df, self.clamp_min = mean, std, df, clamp_min

    def _sample(self, n: int, min_t: Bound, max_t: Bound, device: torch.device | str) -> torch.Tensor:
        # SciPy works on the host, so the probabilities are drawn there, from torch's CPU generator: seeding and
        # ``torch.random.fork_rng`` cover it, as they do the other distributions' draws, but not NumPy's.
        p = torch.rand(n, dtype=torch.float64, device="cpu")
        return self._icdf(p, min_t, max_t).to(device=device)

    def _icdf(self, p: torch.Tensor, min_t: Bound, max_t: Bound) -> torch.Tensor:
        """The Student-t quantile, rescaled and exponentiated, at the fraction *p* between its CDF values at the
        standardized log-bounds, or between its survival probabilities there where the lower one lies in the upper
        tail (``_upper_tail``). SciPy computes it on the host in float64, copying tensors there, and the times are on
        the device of *p*, rounded to the precision."""
        host_p, min_t, max_t = (as_array(value, torch.float64) for value in (p, min_t, max_t))
        lower = (np.log(np.maximum(min_t, self.clamp_min)) - self.mean) / self.std
        upper = (np.log(max_t) - self.mean) / self.std
        a, b = stats.t.cdf(lower, self.df), stats.t.cdf(upper, self.df)
        x = stats.t.ppf(a + (b - a) * host_p, self.df)
        sf_a = stats.t.sf(lower, self.df)
        tail = sf_a < _upper_tail(torch.float64)
        if np.any(tail):
            sf_b = stats.t.sf(upper, self.df)
            x = np.where(tail, stats.t.isf(sf_a - (sf_a - sf_b) * host_p, self.df), x)
        return torch.as_tensor(np.exp(x * self.std + self.mean), dtype=self.precision, device=p.device)


class Polynomial(TruncatedTimeDistribution):
    """A uniform draw under ``polynomial_spacing`` over the bounds: ``t**(1/rho)`` is uniform between the bounds'
    ``1/rho`` powers, the spacing of ``EDMNoiseSchedule``'s grids."""

    def __init__(self, rho: float):
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}")
        self.rho = rho

    def _icdf(self, p: torch.Tensor, min_t: Bound, max_t: Bound) -> torch.Tensor:
        """``polynomial_spacing`` of *p* over the bounds."""
        min_t, max_t = (as_tensor(bound, p.device, self.precision) for bound in (min_t, max_t))
        return polynomial_spacing(p, self.rho, min_t, max_t)


def get_time_distribution(
    time_dist_type: str,
    train_p_mean: float = 0.0,
    train_p_std: float = 1.0,
    shift: float = 5.0,
    log_t_df: float = 0.01,
    clamp_min: float = 1e-6,
    rho: float = 7.0,
) -> TimeDistribution:
    """The ``sample_t`` distribution named *time_dist_type*."""
    if time_dist_type == "uniform":
        return Uniform()
    if time_dist_type == "logitnormal":
        return LogitNormal(train_p_mean, train_p_std)
    if time_dist_type == "shifted":
        return Shifted(shift)
    if time_dist_type == "shifted_logitnormal":
        return ShiftedLogitNormal(train_p_mean, train_p_std, shift)
    if time_dist_type == "lognormal":
        return LogNormal(train_p_mean, train_p_std, clamp_min)
    if time_dist_type == "log_t":
        return LogT(train_p_mean, train_p_std, log_t_df, clamp_min)
    if time_dist_type == "polynomial":
        return Polynomial(rho)
    raise ValueError(f"Unknown time distribution type {time_dist_type!r}")
