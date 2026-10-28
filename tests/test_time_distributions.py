# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time-distribution tests: CDF round trips, windows that condition every distribution, and ``sample_t``."""

import math

import numpy as np
import pytest
import torch
from scipy import stats

from fastgen.networks.noise_schedule import (
    AlphasNoiseSchedule,
    BaseNoiseSchedule,
    CompositeNoiseSchedule,
    EDMNoiseSchedule,
    RFNoiseSchedule,
    TrigNoiseSchedule,
)
from fastgen.networks.time_distributions import (
    LogitNormal,
    LogNormal,
    LogT,
    Polynomial,
    Shifted,
    ShiftedLogitNormal,
    Uniform,
    UnitIntervalDistribution,
    get_time_distribution,
    polynomial_spacing,
    time_shift,
)
import fastgen.utils.tensor as tensor_module
from tests.utils import RunIf

MIN_T, MAX_T = 0.001, 0.999

UNIT_INTERVAL_DISTRIBUTIONS = [
    pytest.param(Uniform(), id="uniform"),
    pytest.param(LogitNormal(0.0, 1.0), id="logitnormal"),
    pytest.param(LogitNormal(-0.6, 1.6), id="logitnormal-skewed"),
    pytest.param(Shifted(5.0), id="shifted"),
    pytest.param(Shifted(0.25), id="shifted-below-one"),
    pytest.param(ShiftedLogitNormal(0.0, 1.0, 3.0), id="shifted-logitnormal"),
]

TRUNCATED_DISTRIBUTIONS = [
    pytest.param(LogNormal(-1.2, 1.2, 1e-6), id="lognormal"),
    pytest.param(LogT(-1.2, 1.2, 2.0, 1e-6), id="log-t"),
    pytest.param(Polynomial(7.0), id="polynomial"),
]


def _in_time(y):
    """Positions *y* in [0, 1], as times on the bounds."""
    return MIN_T + (MAX_T - MIN_T) * y


def _alphas_schedule() -> AlphasNoiseSchedule:
    return AlphasNoiseSchedule(alphas_cumprod=torch.linspace(0.9999, 0.0001, 1000, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Round trips and windows of the distributions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS)
def test_cdf_and_icdf_round_trip(distribution):
    """``icdf(cdf(t)) = t`` and ``cdf(icdf(p)) = p``, on number bounds and on per-sample bounds."""
    p = torch.linspace(0, 1, 1001, dtype=torch.float64)
    t = _in_time(p)
    round_trip_t = distribution.icdf(distribution.cdf(t, MIN_T, MAX_T), MIN_T, MAX_T)
    round_trip_p = distribution.cdf(distribution.icdf(p, MIN_T, MAX_T), MIN_T, MAX_T)
    torch.testing.assert_close(round_trip_t, t, rtol=0, atol=1e-8)
    torch.testing.assert_close(round_trip_p, p, rtol=0, atol=1e-10)

    lower = torch.linspace(0.0, 0.5, 1001, dtype=torch.float64)
    upper = lower + 0.4
    t = lower + (upper - lower) * p
    torch.testing.assert_close(distribution.icdf(distribution.cdf(t, lower, upper), lower, upper), t, rtol=0, atol=1e-8)
    torch.testing.assert_close(distribution.cdf(lower, lower, upper), torch.zeros_like(p), rtol=0, atol=1e-15)
    torch.testing.assert_close(distribution.cdf(upper, lower, upper), torch.ones_like(p), rtol=0, atol=1e-15)


def test_logit_normal_cdf_round_trips_deep_in_its_lower_tail():
    """The CDF keeps its relative precision where ``torch.special.ndtr`` cancels, at ``z = -8`` and below."""
    distribution = LogitNormal(0.0, 1.0)
    t = _in_time(torch.sigmoid(torch.tensor([-8.0, -10.0, -14.0], dtype=torch.float64)))
    round_trip = distribution.icdf(distribution.cdf(t, MIN_T, MAX_T), MIN_T, MAX_T)
    torch.testing.assert_close(round_trip - MIN_T, t - MIN_T, rtol=1e-9, atol=0)


def test_lognormal_windows_deep_in_the_lower_tail_still_condition_the_distribution():
    """As for the logit-normal, the CDF and its inverse keep their relative precision at ``z = -8.6`` here, where
    ``1 + erf`` and ``erfinv(2p - 1)`` cancel, so the draws keep the window's conditional median."""
    torch.manual_seed(0)
    lo, hi = 1e-5, 2e-5
    draws = LogNormal(-1.2, 1.2, 1e-6).sample(10_000, 1e-6, 80.0, window=(lo, hi))
    assert ((draws >= lo * (1 - 1e-12)) & (draws <= hi * (1 + 1e-12))).all()
    cdf_lo, cdf_hi = (stats.norm.cdf((math.log(end) + 1.2) / 1.2) for end in (lo, hi))
    median = math.exp(stats.norm.ppf((cdf_lo + cdf_hi) / 2) * 1.2 - 1.2)
    torch.testing.assert_close(draws.median().item(), median, rtol=1e-2, atol=0)


@pytest.mark.parametrize("shift", [None, 3.0])
def test_logit_normal_windows_deep_in_the_upper_tail_still_condition_the_distribution(shift):
    """The CDF rounds to 1 from ``z = 8.3``, so a window at logits 10 to 12, with or without a shift, draws between
    survival probabilities instead: the draws stay in the window and keep its conditional median."""
    torch.manual_seed(0)
    distribution = LogitNormal(0.0, 1.0) if shift is None else ShiftedLogitNormal(0.0, 1.0, shift)
    lo, hi = (1 / (1 + math.exp(-z)) for z in (10.0, 12.0))
    draws = distribution.sample(10_000, 0.0, 1.0, window=(lo, hi))
    tail = 1 - draws
    assert ((tail >= (1 - hi) * (1 - 1e-9)) & (tail <= (1 - lo) * (1 + 1e-9))).all()

    def unshifted(y):
        return y if shift is None else time_shift(y, 1 / shift)

    s_lo, s_hi = (stats.norm.sf(math.log(unshifted(end) / (1 - unshifted(end)))) for end in (lo, hi))
    median = 1 / (1 + math.exp(-stats.norm.isf((s_lo + s_hi) / 2)))
    median = median if shift is None else float(time_shift(np.float64(median), shift))
    torch.testing.assert_close(tail.median().item(), 1 - median, rtol=1e-2, atol=0)


@pytest.mark.parametrize(
    ("distribution", "law"),
    [
        pytest.param(LogNormal(-1.2, 1.2, 1e-6), stats.norm, id="lognormal"),
        pytest.param(LogT(-1.2, 1.2, 1e4, 1e-6), stats.t(1e4), id="log-t"),
    ],
)
def test_windows_deep_in_the_upper_tail_still_truncate_the_distribution(distribution, law):
    """As for the logit-normal, a window at ``z = 9`` to ``10``, where the CDF rounds to 1, is drawn between survival
    probabilities: the draws stay in it and keep its conditional median."""
    torch.manual_seed(0)
    lo, hi = (math.exp(1.2 * z - 1.2) for z in (9.0, 10.0))
    draws = distribution.sample(10_000, 1e-6, 1e6, window=(lo, hi))
    assert ((draws >= lo * (1 - 1e-12)) & (draws <= hi * (1 + 1e-12))).all()
    median = math.exp(law.isf((law.sf(9.0) + law.sf(10.0)) / 2) * 1.2 - 1.2)
    torch.testing.assert_close(draws.median().item(), median, rtol=1e-2, atol=0)


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS)
def test_windows_short_of_the_upper_tail_invert_the_cdf_between_their_ends(distribution):
    """Only windows that start where the CDF has lost half its bits of the mass above use survival probabilities;
    one just short of that, at 0.99 to 0.995 of the bounds, inverts the CDF between its ends, draw for draw."""
    lo, hi = (torch.tensor(_in_time(y), dtype=torch.float64) for y in (0.99, 0.995))
    torch.manual_seed(0)
    draws = distribution.sample(1000, MIN_T, MAX_T, window=(lo, hi))
    torch.manual_seed(0)
    u = torch.rand(1000, dtype=torch.float64)
    p_lo, p_hi = distribution.cdf(lo, MIN_T, MAX_T), distribution.cdf(hi, MIN_T, MAX_T)
    assert torch.equal(draws, distribution.icdf(p_lo + (p_hi - p_lo) * u, MIN_T, MAX_T))


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS)
def test_cdf_is_the_cdf_of_the_unconditioned_draw(distribution):
    """Pushed through the CDF, draws without a window are uniform on [0, 1] (the probability integral transform),
    which ties the CDF that a window inverts to the unconditioned sampler."""
    torch.manual_seed(0)
    n = 200_000
    u = distribution.cdf(distribution.sample(n, MIN_T, MAX_T), MIN_T, MAX_T).sort().values
    # Dvoretzky-Kiefer-Wolfowitz: the ECDF strays 0.005 from the uniform CDF with probability below 1e-4.
    assert (torch.arange(1, n + 1, dtype=torch.float64) / n - u).abs().max() < 0.005


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS)
def test_a_window_conditions_the_distribution(distribution):
    """A window draws what the unconditioned sampler draws inside the window, for number and per-sample windows."""
    torch.manual_seed(0)
    reference = distribution.sample(4_000_000, MIN_T, MAX_T)
    windows = [(_in_time(0.3), _in_time(0.5)), (_in_time(0.7), _in_time(0.9))]
    n = 100_000
    lower = torch.tensor([start for start, _ in windows], dtype=torch.float64).repeat(n // 2)
    upper = torch.tensor([end for _, end in windows], dtype=torch.float64).repeat(n // 2)
    per_sample = distribution.sample(n, MIN_T, MAX_T, window=(lower, upper))
    assert ((per_sample >= lower - 1e-12) & (per_sample <= upper + 1e-12)).all()

    for i, (start, end) in enumerate(windows):
        kept = reference[(reference >= start) & (reference <= end)]
        for draws in (distribution.sample(n, MIN_T, MAX_T, window=(start, end)), per_sample[i::2]):
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
                gap = torch.quantile(draws, quantile) - torch.quantile(kept, quantile)
                assert abs(gap.item()) < 0.015 * (end - start), (start, end, quantile)


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS)
def test_a_window_on_the_bounds_is_the_whole_distribution(distribution):
    torch.manual_seed(0)
    whole = distribution.sample(1_000_000, MIN_T, MAX_T)
    windowed = distribution.sample(1_000_000, MIN_T, MAX_T, window=(MIN_T, MAX_T))
    for quantile in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert abs((torch.quantile(windowed, quantile) - torch.quantile(whole, quantile)).item()) < 0.005


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS)
def test_window_ends_rounded_past_the_bounds_draw_finite_times(distribution):
    """A float32 window end at a float64 bound rounds past it (float32 0.999 exceeds 0.999), which must not turn
    the draws into NaN."""
    torch.manual_seed(0)
    lower = torch.full((1000,), 0.5, dtype=torch.float32)
    upper = torch.full((1000,), MAX_T, dtype=torch.float32)
    assert upper[0].item() > MAX_T
    t = distribution.sample(1000, MIN_T, MAX_T, window=(lower, upper))
    assert torch.isfinite(t).all() and (t >= 0.5).all() and (t <= upper.double()).all()


@pytest.mark.parametrize("shift", [5.0, 3.0, 0.25])
@pytest.mark.parametrize(("start", "end"), [(0.938, 0.999), (0.834, 0.937), (0.625, 0.833), (0.002, 0.624), (0.3, 0.6)])
def test_a_shifted_window_is_a_shifted_draw_over_the_window(shift, start, end):
    """The shift on the bounds, conditioned on ``[start, end]``, is the shift over the window with
    ``D(start) / D(end)``, where ``D(y) = shift - (shift - 1) y`` for the time ``y`` normalized to the bounds."""
    distribution = Shifted(shift)
    ends = torch.tensor([start, end], dtype=torch.float64)
    cdf_start, cdf_end = distribution.cdf(ends, MIN_T, MAX_T)
    y_start, y_end = (ends - MIN_T) / (MAX_T - MIN_T)
    window_shift = float((shift - (shift - 1) * y_start) / (shift - (shift - 1) * y_end))

    t = torch.linspace(start, end, 257, dtype=torch.float64)
    conditioned = (distribution.cdf(t, MIN_T, MAX_T) - cdf_start) / (cdf_end - cdf_start)
    torch.testing.assert_close(conditioned, Shifted(window_shift).cdf(t, start, end), rtol=0, atol=1e-12)


@pytest.mark.parametrize("distribution", TRUNCATED_DISTRIBUTIONS)
def test_a_window_truncates_like_narrower_bounds(distribution):
    """A distribution over time, conditioned on a window, is that distribution truncated to it, draw for draw."""
    lower = torch.tensor([0.05, 1.0], dtype=torch.float64).repeat(4)
    upper = torch.tensor([0.3, 5.0], dtype=torch.float64).repeat(4)
    for window in ((0.05, 0.3), (lower, upper)):
        torch.manual_seed(0)
        windowed = distribution.sample(8, 0.002, 80.0, window=window)
        torch.manual_seed(0)
        assert torch.equal(windowed, distribution.sample(8, *window))


@pytest.mark.parametrize("rho", [7.0, 3.0])
def test_polynomial_is_uniform_in_the_rho_root(rho):
    """``t**(1/rho)`` is uniform between the bounds' roots, for number and per-sample bounds."""
    torch.manual_seed(0)
    n = 200_000
    lower = torch.tensor([0.002, 1.0], dtype=torch.float64).repeat(n // 2)
    upper = torch.tensor([80.0, 5.0], dtype=torch.float64).repeat(n // 2)
    per_sample = Polynomial(rho).sample(n, lower, upper)
    for i, (lo, hi) in enumerate(((0.002, 80.0), (1.0, 5.0))):
        for draws in (Polynomial(rho).sample(n // 2, lo, hi), per_sample[i::2]):
            u = ((draws ** (1 / rho) - lo ** (1 / rho)) / (hi ** (1 / rho) - lo ** (1 / rho))).sort().values
            # Dvoretzky-Kiefer-Wolfowitz: the ECDF strays 0.0075 from the uniform CDF with probability below 1e-4.
            assert (torch.arange(1, len(u) + 1, dtype=torch.float64) / len(u) - u).abs().max() < 0.0075


def test_polynomial_spacing_maps_the_unit_interval_onto_the_bounds():
    """Positions 0 and 1 land on the bounds, 0 on ``min_t`` rather than the clean end, and the ``1/rho`` powers run
    linearly in between, for number and per-sample bounds."""
    rho = 7.0
    u = torch.linspace(0, 1, 9, dtype=torch.float64)[:, None].expand(-1, 2)
    lower = torch.tensor([0.002, 1.0], dtype=torch.float64)
    upper = torch.tensor([80.0, 5.0], dtype=torch.float64)
    t = polynomial_spacing(u, rho, lower.expand_as(u), upper.expand_as(u))
    torch.testing.assert_close(t[0], lower, rtol=1e-12, atol=0)
    torch.testing.assert_close(t[-1], upper, rtol=1e-12, atol=0)
    roots = lower ** (1 / rho) + u * (upper ** (1 / rho) - lower ** (1 / rho))
    torch.testing.assert_close(t ** (1 / rho), roots, rtol=1e-12, atol=0)
    for i, (lo, hi) in enumerate(((0.002, 80.0), (1.0, 5.0))):
        torch.testing.assert_close(polynomial_spacing(u[:, 0], rho, lo, hi), t[:, i], rtol=1e-12, atol=0)


def test_get_time_distribution_builds_every_named_distribution():
    expected = {
        "uniform": Uniform,
        "logitnormal": LogitNormal,
        "shifted": Shifted,
        "shifted_logitnormal": ShiftedLogitNormal,
        "lognormal": LogNormal,
        "log_t": LogT,
        "polynomial": Polynomial,
    }
    for name, cls in expected.items():
        assert type(get_time_distribution(name)) is cls
    with pytest.raises(ValueError, match="Unknown time distribution"):
        get_time_distribution("cosine")


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS + TRUNCATED_DISTRIBUTIONS)
def test_every_draw_comes_from_a_torch_generator(distribution):
    """``torch.manual_seed`` alone fixes the draws, whatever NumPy's generator holds, and ``torch.random.fork_rng``,
    with which methods and callbacks replay a step's randomness, replays them."""
    for window in (None, (0.2, 0.4)):
        draws = []
        for numpy_seed in (1, 2):
            torch.manual_seed(0)
            np.random.seed(numpy_seed)
            draws.append(distribution.sample(100, MIN_T, MAX_T, window=window))
        assert torch.equal(draws[0], draws[1])
        with torch.random.fork_rng(devices=[]):
            forked = distribution.sample(100, MIN_T, MAX_T, window=window)
        assert torch.equal(distribution.sample(100, MIN_T, MAX_T, window=window), forked)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_a_bound_of_another_shape_raises_and_a_0d_tensor_acts_as_a_number():
    """A bound is a number, a 0-d tensor or a tensor of the values' shape: every entry point raises for one that would
    broadcast the values to another shape, and for a tensor bound on values that are not a tensor."""
    t = torch.linspace(0.1, 0.9, 4, dtype=torch.float64)
    column = torch.full((4, 1), 0.95, dtype=torch.float64)
    distribution = Shifted(3.0)
    for call in (
        lambda: time_shift(t, 3.0, max_t=column),
        lambda: polynomial_spacing(t, 7.0, 0.002, column),
        lambda: distribution.sample(4, 0.0, column),
        lambda: distribution.sample(4, 0.0, 1.0, window=(0.1, column)),
        lambda: distribution.icdf(t, 0.0, column),
        lambda: distribution.cdf(t, 0.0, column),
    ):
        with pytest.raises(ValueError, match=r"must have shape \[4\], got \(4, 1\)"):
            call()
    with pytest.raises(ValueError, match="must be a number"):
        time_shift(0.5, 3.0, max_t=torch.tensor(0.95))

    scalar = torch.tensor(0.95, dtype=torch.float64)
    assert torch.equal(time_shift(t, 3.0, max_t=scalar), time_shift(t, 3.0, max_t=0.95))
    torch.manual_seed(0)
    draws = distribution.sample(4, 0.0, scalar)
    torch.manual_seed(0)
    assert torch.equal(draws, distribution.sample(4, 0.0, 0.95))


def test_numpy_bounds_are_checked_like_tensor_bounds():
    """NumPy values take NumPy bounds of their shape, a NumPy bound of another shape raises, and a bound of the other
    kind raises, also where the values are tensors: in the distributions and ``sample_t``."""
    t = np.linspace(0.1, 0.9, 4)
    upper = np.full(4, 0.95)
    np.testing.assert_array_equal(time_shift(t, 3.0, max_t=upper), time_shift(t, 3.0, max_t=0.95))
    np.testing.assert_allclose(polynomial_spacing(t, 7.0, 0.002, upper), polynomial_spacing(t, 7.0, 0.002, 0.95))
    for call in (
        lambda: time_shift(t, 3.0, max_t=upper[:, None]),
        lambda: polynomial_spacing(t, 7.0, 0.002, upper[:, None]),
    ):
        with pytest.raises(ValueError, match=r"must have shape \[4\], got \(4, 1\)"):
            call()
    for call in (
        lambda: time_shift(t, 3.0, max_t=torch.from_numpy(upper)),
        lambda: time_shift(torch.from_numpy(t), 3.0, max_t=upper),
        lambda: Shifted(3.0).sample(4, 0.0, upper),
        lambda: RFNoiseSchedule().sample_t(4, max_t=upper),
    ):
        with pytest.raises(ValueError, match="must be a number or a"):
            call()


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS + TRUNCATED_DISTRIBUTIONS)
def test_every_distribution_rejects_empty_intervals(distribution):
    """An inverted or zero-width interval or window, or one with a NaN end, raises, for number and per-sample ends,
    instead of drawing NaN under a shift (zero span) or times outside it; ``sample_t`` relies on this for the
    intervals it clips."""
    p = torch.linspace(0.1, 0.9, 4, dtype=torch.float64)
    lower = torch.full((4,), 0.5, dtype=torch.float64)
    for min_t, max_t in ((0.5, 0.5), (0.9, 0.5), (math.nan, 0.5), (lower, lower), (lower, 0.2)):
        with pytest.raises(ValueError, match="min_t < max_t"):
            distribution.sample(4, min_t, max_t)
        with pytest.raises(ValueError, match="min_t < max_t"):
            distribution.icdf(p, min_t, max_t)
        if isinstance(distribution, UnitIntervalDistribution):
            with pytest.raises(ValueError, match="min_t < max_t"):
                distribution.cdf(p, min_t, max_t)
    for window in ((0.5, 0.5), (0.6, 0.4), (math.nan, 0.5), (lower, lower)):
        with pytest.raises(ValueError, match="window with start < end"):
            distribution.sample(4, MIN_T, MAX_T, window=window)


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS + TRUNCATED_DISTRIBUTIONS)
def test_a_window_past_the_bounds_draws_on_its_part_inside_them(distribution):
    """A window that reaches past the bounds draws, draw for draw, what its part inside them draws, for number and
    per-sample ends, and one that misses the bounds raises."""
    lower = torch.tensor([0.1, 0.3, 0.6, 0.1], dtype=torch.float64)
    upper = torch.tensor([0.5, 0.9, 0.95, 0.7], dtype=torch.float64)
    cases = [((0.1, 0.5), (0.2, 0.5)), ((lower, upper), (lower.clamp(min=0.2), upper.clamp(max=0.8)))]
    for window, inside in cases:
        torch.manual_seed(0)
        draws = distribution.sample(4, 0.2, 0.8, window=window)
        torch.manual_seed(0)
        assert torch.equal(draws, distribution.sample(4, 0.2, 0.8, window=inside))
    with pytest.raises(ValueError, match="window with start < end"):
        distribution.sample(4, 0.2, 0.8, window=(0.85, 0.95))


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS + TRUNCATED_DISTRIBUTIONS)
def test_user_facing_methods_compute_and_return_float64_by_default(monkeypatch, distribution):
    """Without a dtype, every user-facing method computes and returns float64, whatever ``HIGH_PRECISION`` and the
    dtypes of its operands are."""
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)
    p = torch.linspace(0.01, 0.99, 99, dtype=torch.bfloat16)
    results = [
        distribution.sample(4, MIN_T, MAX_T),
        distribution.sample(4, torch.tensor(MIN_T), torch.tensor(MAX_T), window=(0.2, 0.4)),
        distribution.icdf(p, MIN_T, MAX_T),
    ]
    if isinstance(distribution, UnitIntervalDistribution):
        results.append(distribution.cdf(p, MIN_T, MAX_T))
    assert all(result.dtype == torch.float64 for result in results)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("window", [None, (0.1, 0.4)])
@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS + TRUNCATED_DISTRIBUTIONS)
def test_a_dtype_only_rounds_the_float64_result(distribution, window, dtype):
    """The arithmetic stays in the float64 precision, and a dtype casts the result once: draws, quantiles and CDF
    values in a coarser dtype, half precision too, are the float64 ones rounded."""
    draws = []
    for draw_dtype in (dtype, None):
        torch.manual_seed(0)
        draws.append(distribution.sample(1000, 0.002, 0.999, window=window, dtype=draw_dtype))
    assert draws[0].dtype == dtype and torch.equal(draws[0], draws[1].to(dtype))
    p = torch.linspace(0.01, 0.99, 99, dtype=torch.float64)
    assert torch.equal(distribution.icdf(p, MIN_T, MAX_T, dtype=dtype), distribution.icdf(p, MIN_T, MAX_T).to(dtype))
    if isinstance(distribution, UnitIntervalDistribution):
        t = _in_time(p)
        assert torch.equal(distribution.cdf(t, MIN_T, MAX_T, dtype=dtype), distribution.cdf(t, MIN_T, MAX_T).to(dtype))


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS + TRUNCATED_DISTRIBUTIONS)
def test_float32_operands_act_as_their_float64_values(distribution):
    """Every tensor operand is cast to float64 before any arithmetic, so float32 bounds, window ends and probabilities
    give, draw for draw, what their values give in float64: the span between float32 bounds, for one, is not rounded
    to float32."""
    lower = torch.tensor([0.002, 0.3], dtype=torch.float32).repeat(500)
    upper = torch.tensor([0.6, 0.999], dtype=torch.float32).repeat(500)
    window = (lower + 0.1 * (upper - lower), lower + 0.7 * (upper - lower))
    for conditioned in (False, True):
        draws = []
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(0)
            ends = tuple(end.to(dtype) for end in window) if conditioned else None
            draws.append(distribution.sample(1000, lower.to(dtype), upper.to(dtype), window=ends))
        assert torch.equal(draws[0], draws[1])

    p = torch.linspace(0.001, 0.999, 1000, dtype=torch.float32)
    t = distribution.icdf(p, lower, upper)
    assert torch.equal(t, distribution.icdf(p.double(), lower.double(), upper.double()))
    if isinstance(distribution, UnitIntervalDistribution):
        coarse = t.float()
        p = distribution.cdf(coarse, lower, upper)
        assert torch.equal(p, distribution.cdf(coarse.double(), lower.double(), upper.double()))


@pytest.mark.parametrize("distribution", UNIT_INTERVAL_DISTRIBUTIONS + TRUNCATED_DISTRIBUTIONS)
def test_integer_operands_act_as_their_float64_values(distribution):
    """Integer tensors are cast to float64 too, rather than promoted to float32 or, as the dtype of the probabilities,
    truncating the bounds: integer probabilities, times and bounds give what their float64 values give."""
    lower, upper = torch.tensor([1, 2]), torch.tensor([3, 5])
    as_float64 = lower.double(), upper.double()
    draws = []
    for bounds in ((lower, upper), as_float64):
        torch.manual_seed(0)
        draws.append(distribution.sample(2, *bounds))
    assert torch.equal(draws[0], draws[1])
    p = torch.tensor([0, 1])
    assert torch.equal(distribution.icdf(p, lower, upper), distribution.icdf(p.double(), *as_float64))
    if isinstance(distribution, UnitIntervalDistribution):
        t = torch.tensor([1, 4])
        assert torch.equal(distribution.cdf(t, lower, upper), distribution.cdf(t.double(), *as_float64))


# ---------------------------------------------------------------------------
# sample_t
# ---------------------------------------------------------------------------


def _draw_as_before(time_dist_type, n, min_t, max_t, mean, std, shift):
    """The unconditioned draws ``sample_t`` made before its distributions moved, written out as they were."""
    if time_dist_type == "uniform":
        return torch.rand(n, dtype=torch.float64) * (max_t - min_t) + min_t
    if time_dist_type in ("logitnormal", "shifted_logitnormal"):
        t = torch.sigmoid(torch.randn(n, dtype=torch.float64) * std + mean) * (max_t - min_t) + min_t
    else:
        t = torch.rand(n, dtype=torch.float64) * (max_t - min_t) + min_t
    return t if time_dist_type == "logitnormal" else time_shift(t, shift, min_t=min_t, max_t=max_t)


@pytest.mark.parametrize(
    ("make_schedule", "time_dist_type", "min_t", "max_t"),
    [
        pytest.param(RFNoiseSchedule, "uniform", 0.01, 0.9, id="rf-uniform"),
        pytest.param(RFNoiseSchedule, "logitnormal", 0.01, 0.9, id="rf-logitnormal"),
        pytest.param(RFNoiseSchedule, "shifted", 0.01, 0.9, id="rf-shifted"),
        pytest.param(RFNoiseSchedule, "shifted_logitnormal", 0.01, 0.9, id="rf-shifted-logitnormal"),
        pytest.param(_alphas_schedule, "logitnormal", 0.01, 0.9, id="alphas-logitnormal"),
        pytest.param(TrigNoiseSchedule, "logitnormal", 0.1, 1.4, id="trig-logitnormal"),
        pytest.param(EDMNoiseSchedule, "uniform", 0.01, 5.0, id="edm-uniform"),
    ],
)
def test_unconditioned_draws_are_the_ones_sample_t_made_before(make_schedule, time_dist_type, min_t, max_t):
    schedule = make_schedule()
    kwargs = {"train_p_mean": -0.3, "train_p_std": 1.4, "shift": 3.0}
    torch.manual_seed(0)
    actual = schedule.sample_t(1000, time_dist_type=time_dist_type, min_t=min_t, max_t=max_t, **kwargs)
    torch.manual_seed(0)
    expected = _draw_as_before(time_dist_type, 1000, min_t, max_t, -0.3, 1.4, 3.0)
    assert torch.equal(actual, BaseNoiseSchedule.safe_clamp(expected, min_t, max_t))


SCHEDULES = [
    pytest.param(RFNoiseSchedule, id="rf"),
    pytest.param(_alphas_schedule, id="alphas"),
    pytest.param(TrigNoiseSchedule, id="trig"),
    pytest.param(EDMNoiseSchedule, id="edm"),
]


@pytest.mark.parametrize("make_schedule", SCHEDULES)
@pytest.mark.parametrize(
    "time_dist_type", ["uniform", "logitnormal", "shifted", "shifted_logitnormal", "lognormal", "log_t", "polynomial"]
)
def test_every_schedule_draws_every_named_distribution(make_schedule, time_dist_type):
    """The named distributions need only the bounds, so every schedule draws each one, with its parameters, as the
    distribution itself does on the same bounds and window."""
    schedule = make_schedule()
    span = schedule.max_t - schedule.min_t
    min_t, max_t = schedule.min_t + 0.1 * span, schedule.min_t + 0.9 * span
    params = {"train_p_mean": -0.5, "train_p_std": 1.3, "shift": 3.0, "log_t_df": 2.0, "rho": 5.0}
    distribution = get_time_distribution(time_dist_type, clamp_min=schedule.clamp_min, **params)
    for window in (None, (schedule.min_t + 0.3 * span, schedule.min_t + 0.6 * span)):
        torch.manual_seed(0)
        t = schedule.sample_t(1000, time_dist_type=time_dist_type, min_t=min_t, max_t=max_t, window=window, **params)
        torch.manual_seed(0)
        expected = distribution.sample(1000, min_t, max_t, window=window)
        lo, hi = (min_t, max_t) if window is None else window
        assert torch.equal(t, BaseNoiseSchedule.safe_clamp(expected, lo, hi))
        assert torch.isfinite(t).all() and (t >= lo).all() and (t <= hi).all()


@pytest.mark.parametrize("make_schedule", SCHEDULES)
def test_sample_t_rejects_unknown_distributions(make_schedule):
    with pytest.raises(ValueError, match="Unknown time distribution"):
        make_schedule().sample_t(2, time_dist_type="cosine")


@pytest.mark.parametrize("make_schedule", SCHEDULES)
def test_sample_t_defaults_to_uniform_over_the_schedule_range(make_schedule):
    schedule = make_schedule()
    torch.manual_seed(0)
    t = schedule.sample_t(1000)
    torch.manual_seed(0)
    assert torch.equal(t, schedule.sample_t(1000, "uniform", min_t=schedule.min_t, max_t=schedule.max_t))


def test_sample_t_intersects_windows_with_the_bounds_and_rejects_empty_ones():
    schedule = RFNoiseSchedule(max_t=0.999)
    kwargs = {"time_dist_type": "shifted", "shift": 5.0, "min_t": 0.1, "max_t": 0.8}
    torch.manual_seed(0)
    t = schedule.sample_t(1000, window=(0.0, 0.5), **kwargs)
    assert t.min() >= 0.1 and t.max() <= 0.5
    lower = torch.tensor([0.2, 0.6], dtype=torch.float64)
    t = schedule.sample_t(2, window=(lower, torch.tensor([0.4, 2.0], dtype=torch.float64)), **kwargs)
    assert (t >= lower).all() and t[0] <= 0.4 and t[1] <= 0.8

    for window in ((0.5, 0.5), (0.6, 0.4), (0.85, 0.95), (lower, lower), (torch.tensor([0.2, 0.9]).double(), 0.95)):
        with pytest.raises(ValueError, match="window with start < end"):
            schedule.sample_t(2, window=window, **kwargs)
    with pytest.raises(ValueError, match="Per-sample window start must have shape"):
        schedule.sample_t(3, window=(lower, 0.8), **kwargs)


def test_sample_t_rejects_nan_draws(monkeypatch):
    monkeypatch.setattr(Uniform, "_sample", lambda self, n, *args: torch.full((n,), math.nan, dtype=torch.float64))
    with pytest.raises(AssertionError, match="sample_t drew non-finite times"):
        RFNoiseSchedule(min_t=MIN_T, max_t=MAX_T).sample_t(4)


@pytest.mark.parametrize(
    "time_dist_type", ["uniform", "logitnormal", "shifted", "shifted_logitnormal", "lognormal", "log_t", "polynomial"]
)
def test_sample_t_icdf_is_the_quantile_function_of_sample_t(time_dist_type):
    """``sample_t_icdf`` inverts the CDF of what ``sample_t`` draws, on the same bounds clipped to the schedule range
    (``max_t = 80`` clips to the rectified-flow ``max_t``)."""
    schedule = RFNoiseSchedule(max_t=0.999)
    params = {"train_p_mean": -0.5, "train_p_std": 1.3, "shift": 3.0, "log_t_df": 2.0, "rho": 5.0}
    kwargs = {"time_dist_type": time_dist_type, "min_t": 0.0, "max_t": 80.0, **params}
    torch.manual_seed(0)
    draws = schedule.sample_t(100_000, **kwargs)
    p = torch.linspace(0, 1, 11, dtype=torch.float64)
    t = schedule.sample_t_icdf(p, **kwargs)
    assert (t.diff() > 0).all() and t[-1] == pytest.approx(0.999, rel=1e-15)
    # Dvoretzky-Kiefer-Wolfowitz: the ECDF strays 0.01 from the CDF with probability below 1e-8.
    torch.testing.assert_close((draws[:, None] <= t).double().mean(dim=0), p, rtol=0, atol=0.01)


@pytest.mark.parametrize("t_precision", ["bfloat16", "float32"])
@pytest.mark.parametrize("window", [None, (0.1, 0.4)])
@pytest.mark.parametrize(
    "time_dist_type", ["uniform", "logitnormal", "shifted", "shifted_logitnormal", "lognormal", "log_t", "polynomial"]
)
def test_sample_t_in_a_coarser_time_precision_rounds_the_float64_draws_once(time_dist_type, window, t_precision):
    """The distributions draw in float64, and ``sample_t`` rounds the draws once to its time precision and clamps them
    inward to the bounds, so a coarse time precision rounds the times but not the distribution."""
    params = {"train_p_mean": -0.5, "train_p_std": 1.3, "log_t_df": 2.0}
    kwargs = {"time_dist_type": time_dist_type, "min_t": 0.002, "max_t": 0.999, "window": window, **params}
    draws = []
    for precision in (t_precision, "float64"):
        torch.manual_seed(0)
        draws.append(RFNoiseSchedule(t_precision=precision).sample_t(1000, **kwargs))
    lo, hi = (0.002, 0.999) if window is None else window
    assert draws[0].dtype == getattr(torch, t_precision)
    assert torch.equal(draws[0], BaseNoiseSchedule.safe_clamp(draws[1].to(draws[0].dtype), lo, hi))


def test_sample_t_icdf_rounds_only_the_times_to_the_time_precision():
    p = torch.linspace(0, 1, 101, dtype=torch.float64)
    coarse = RFNoiseSchedule(t_precision="bfloat16").sample_t_icdf(p, time_dist_type="logitnormal")
    fine = RFNoiseSchedule().sample_t_icdf(p, time_dist_type="logitnormal")
    assert torch.equal(coarse, BaseNoiseSchedule.safe_clamp(fine.to(torch.bfloat16), min=0.0, max=0.999))


@pytest.mark.parametrize(
    "time_dist_type", ["uniform", "logitnormal", "shifted", "shifted_logitnormal", "lognormal", "log_t", "polynomial"]
)
def test_sample_t_computes_in_float64_whatever_high_precision(monkeypatch, time_dist_type):
    """Bounds are float64, numbers and tensors alike, so draws and quantiles are computed in float64 even under a
    float32 ``HIGH_PRECISION``: a float32 time precision rounds the float64 times once, and scalar and per-sample
    bounds of one value draw alike."""
    params = {"train_p_mean": -0.5, "train_p_std": 1.3, "shift": 3.0, "log_t_df": 2.0, "rho": 5.0}
    lower, upper = (torch.full((1000,), bound, dtype=torch.float64) for bound in (0.2, 0.7))
    calls = [{"min_t": 0.1, "max_t": 0.9}, {"min_t": torch.full((1000,), 0.1, dtype=torch.float64), "max_t": 0.9}]
    calls.append({"window": (lower, upper)})
    clamps = [(0.1, 0.9), (0.1, 0.9), (lower, upper), (0.0, 0.999)]
    p = torch.linspace(0, 1, 1000, dtype=torch.float32)

    def times(schedule):
        results = []
        for kwargs in calls:
            torch.manual_seed(0)
            results.append(schedule.sample_t(1000, time_dist_type=time_dist_type, **kwargs, **params))
        return results + [schedule.sample_t_icdf(p, time_dist_type=time_dist_type, **params)]

    fine = times(RFNoiseSchedule(t_precision="float64"))
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)
    coarse = times(RFNoiseSchedule(t_precision="float32"))
    for t, reference, (lo, hi) in zip(coarse, fine, clamps):
        assert t.dtype == torch.float32
        assert torch.equal(t, BaseNoiseSchedule.safe_clamp(reference.float(), lo, hi))
    assert torch.equal(coarse[0], coarse[1])


@pytest.mark.parametrize("shift", [0.25, 1.0, 5.0])
@pytest.mark.parametrize("max_t", [1.0, 0.999])
def test_shifted_quantiles_are_the_shifted_schedule_grid(max_t, shift):
    """A shifted draw on the schedule range has the shifted schedule grid as its quantiles at probabilities
    1, 1 - 1/N, ..., 1/N: bit for bit for ``max_t = 1``, the reference's table, and to rounding below it."""
    schedule = RFNoiseSchedule(min_t=0.0, max_t=max_t)
    p = torch.linspace(1.0, 0.0, 1001, dtype=torch.float64)[:-1]
    quantiles = schedule.sample_t_icdf(p, time_dist_type="shifted", shift=shift, min_t=0.0, max_t=max_t)
    grid = schedule.shift_t(schedule.grid(1000, device=torch.device("cpu"))[:-1], shift=shift)
    if max_t == 1.0:
        assert torch.equal(quantiles, grid)
    else:
        torch.testing.assert_close(quantiles, grid, rtol=0, atol=4 * torch.finfo(torch.float64).eps)


@pytest.mark.parametrize("time_dist_type", ["uniform", "logitnormal", "shifted", "shifted_logitnormal"])
def test_sample_t_keeps_float32_windows_inside_the_schedule_range(time_dist_type):
    """A float32 window end at ``max_t`` rounds past it (float32 0.999 exceeds 0.999); the draws still lie in the
    schedule range, here from a window one float32 step wide, a fifth of which lies past ``max_t``."""
    torch.manual_seed(0)
    schedule = RFNoiseSchedule(min_t=MIN_T, max_t=MAX_T)
    upper = torch.full((1000,), MAX_T, dtype=torch.float32)
    lower = torch.nextafter(upper, torch.zeros_like(upper))
    t = schedule.sample_t(1000, time_dist_type=time_dist_type, window=(lower, upper))
    assert t.dtype == schedule.t_precision and torch.isfinite(t).all()
    assert (t >= lower.double()).all() and (t <= MAX_T).all()


def test_per_sample_bounds_and_windows_act_as_scalar_ones_in_a_coarse_time_precision():
    """Tensor bounds and windows are kept in float64, like number bounds, so in a coarse time precision they draw as
    the scalar ones do, and the clamp of the draws rounds them inward: bfloat16 0.999 is 1, pure noise for RF."""
    schedule = RFNoiseSchedule(t_precision="bfloat16")
    n = 100_000
    lower, upper = (torch.full((n,), bound, dtype=torch.float64) for bound in (0.9, 0.999))
    cases = [
        ({"min_t": 0.9}, {"min_t": lower, "max_t": upper}),
        ({"window": (0.9, 0.999)}, {"window": (lower, upper)}),
    ]
    for scalar, per_sample in cases:
        draws = []
        for kwargs in (scalar, per_sample):
            torch.manual_seed(0)
            draws.append(schedule.sample_t(n, **kwargs))
        assert torch.equal(draws[1], draws[0]) and draws[1].double().max() < 0.999


def test_logit_normal_windows_next_to_max_t_do_not_collapse_onto_it():
    """The logit-normal is mapped onto the bounds, so its upper tail ends at ``max_t``: on RF's ``[0, 0.999]`` its CDF
    rounds to 1 from ``t = 0.99875``, and a window above that still draws its conditional distribution."""
    torch.manual_seed(0)
    lo, hi = 0.9988, 0.999
    t = RFNoiseSchedule(t_precision="float64").sample_t(10_000, time_dist_type="logitnormal", window=(lo, hi))
    assert ((t >= lo) & (t <= hi)).all()
    median = hi / (1 + math.exp(-stats.norm.isf(stats.norm.sf(math.log(lo / (hi - lo))) / 2)))
    torch.testing.assert_close(hi - t.median().item(), hi - median, rtol=1e-2, atol=0)


def test_composite_sample_t_passes_windows_to_its_primary():
    composite = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "shifted_rf", "shift": 12.0, "min_t": 0.0, "max_t": 0.999},
            "audio": {"name": "shifted_rf", "shift": 3.0, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )
    window = (torch.tensor([0.1, 0.6], dtype=torch.float64), torch.tensor([0.4, 0.9], dtype=torch.float64))
    kwargs = {"time_dist_type": "shifted", "shift": 5.0, "window": window}
    torch.manual_seed(0)
    t = composite.sample_t(2, **kwargs)
    torch.manual_seed(0)
    assert torch.equal(t, composite.primary.sample_t(2, **kwargs))
    assert ((t >= window[0]) & (t <= window[1])).all()

    p = torch.linspace(0, 1, 11, dtype=torch.float64)
    icdf_kwargs = {"time_dist_type": "shifted", "shift": 5.0}
    assert torch.equal(composite.sample_t_icdf(p, **icdf_kwargs), composite.primary.sample_t_icdf(p, **icdf_kwargs))


@RunIf(min_gpus=1)
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature")
@pytest.mark.parametrize(
    ("make_schedule", "time_dist_type"),
    [
        pytest.param(RFNoiseSchedule, "shifted", id="rf-shifted"),
        pytest.param(RFNoiseSchedule, "shifted_logitnormal", id="rf-shifted-logitnormal"),
        pytest.param(EDMNoiseSchedule, "lognormal", id="edm-lognormal"),
        pytest.param(_alphas_schedule, "logitnormal", id="alphas-logitnormal"),
        pytest.param(lambda: TrigNoiseSchedule(max_t=math.pi / 2), "uniform", id="trig-uniform"),
    ],
)
def test_windowed_sample_t_does_not_synchronize_the_device(make_schedule, time_dist_type):
    """Per-sample windows, which DMD2 passes on every step, are intersected, checked and drawn on their device."""
    schedule = make_schedule()
    span = schedule.max_t - schedule.min_t
    start = schedule.min_t + span * torch.tensor([0.1, 0.5], dtype=torch.float64, device="cuda")
    end = start + 0.3 * span
    torch.cuda.synchronize()

    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        t = schedule.sample_t(2, time_dist_type=time_dist_type, window=(start, end), device="cuda")
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert t.device.type == "cuda" and ((t >= start) & (t <= end)).all()
