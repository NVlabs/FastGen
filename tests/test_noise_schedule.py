# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Noise-schedule tests: conversions, path inversion, and module lifecycle."""

import math
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule, fully_shard

from fastgen.networks.module import FastGenModule
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import (
    AlphasNoiseSchedule,
    BaseNoiseSchedule,
    CompositeNoiseSchedule,
    EDMNoiseSchedule,
    RFNoiseSchedule,
    ShiftedRFNoiseSchedule,
    StudentGridConfig,
    TrigNoiseSchedule,
    check_sample_type,
    get_noise_schedule,
    sqrt_snr_to_alpha_sigma,
    time_shift,
)
import fastgen.utils.tensor as tensor_module
from fastgen.utils.tensor import high_precision_dtype
from fastgen.utils.distributed.ddp import model_to_ddp
from fastgen.utils.distributed.fsdp import create_fsdp_context
from fastgen.utils.module import materialize_meta_module
from fastgen.utils.tree_utils import randn_like
from tests.utils import RunIf, run_distributed_test


def _cosine_alphas_schedule() -> AlphasNoiseSchedule:
    """Standalone DDPM-style schedule so v-prediction is testable without files."""
    steps = 1000
    grid = torch.linspace(0.0, 1.0, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos((grid + 0.008) / 1.008 * math.pi / 2) ** 2
    return AlphasNoiseSchedule(
        alphas_cumprod=alphas_cumprod / alphas_cumprod[0],
        min_t=0.0,
        max_t=0.999,
    )


def _scaled_linear_alphas_schedule() -> AlphasNoiseSchedule:
    """SD's scaled-linear DDPM schedule, whose first discrete time ``0`` already carries noise."""
    betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=torch.float64) ** 2
    return AlphasNoiseSchedule(alphas_cumprod=torch.cumprod(1 - betas, dim=0), min_t=0.0, max_t=0.999)


def _interior_times(schedule: BaseNoiseSchedule, n: int = 7) -> torch.Tensor:
    """Times strictly inside the schedule range, away from clamped endpoints."""
    span = schedule.max_t - schedule.min_t
    return torch.linspace(
        schedule.min_t + 0.05 * span,
        schedule.max_t - 0.05 * span,
        n,
        dtype=torch.float64,
    )


_SMOOTH_SCHEDULES = [
    pytest.param(lambda: RFNoiseSchedule(min_t=0.0, max_t=0.999), id="rf"),
    pytest.param(lambda: ShiftedRFNoiseSchedule(shift=0.25, min_t=0.0, max_t=0.999), id="rf-shift0.25"),
    pytest.param(lambda: ShiftedRFNoiseSchedule(shift=6.0, min_t=0.0, max_t=0.999), id="rf-shift6"),
    pytest.param(lambda: EDMNoiseSchedule(min_t=0.002, max_t=5.0), id="edm"),
    pytest.param(lambda: TrigNoiseSchedule(min_t=0.0, max_t=math.pi / 2), id="trig"),
]


def test_rf_rejects_a_shift_and_is_shifted_rf_at_unit_shift():
    """``"rf"`` is plain rectified flow and takes no ``shift``; ``"shifted_rf"`` applies one and is ``"rf"`` at 1."""
    rf = get_noise_schedule("rf", min_t=0.0, max_t=0.999)
    unit = get_noise_schedule("shifted_rf", shift=1.0, min_t=0.0, max_t=0.999)
    shifted = get_noise_schedule("shifted_rf", shift=3.0, min_t=0.0, max_t=0.999)
    assert type(rf) is RFNoiseSchedule and isinstance(rf, ShiftedRFNoiseSchedule)
    assert type(unit) is type(shifted) is ShiftedRFNoiseSchedule and shifted.shift == 3.0
    for make_shifted_rf in (lambda: get_noise_schedule("rf", shift=3.0), lambda: RFNoiseSchedule(shift=1.0)):
        with pytest.raises(TypeError, match="shift"):
            make_shifted_rf()

    torch.manual_seed(0)
    x0, eps = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    t = _interior_times(rf, n=5)
    t_next = 0.5 * t
    xt, x_next, flow = rf.forward_process(x0, eps, t), rf.forward_process(x0, eps, t_next), rf.cond_velocity(x0, eps, t)
    calls = {
        "sigma": (t,),
        "sigma_prime": (t,),
        "rescale_t": (t,),
        "sqrt_snr_to_t": (rf.sqrt_snr(t),),
        "cond_velocity": (x0, eps, t),
        "flow_to_x0": (xt, flow, t),
        "x0_to_flow": (xt, x0, t),
        "points_to_x0": (xt, x_next, t, t_next),
    }
    for method, args in calls.items():
        assert torch.equal(getattr(unit, method)(*args), getattr(rf, method)(*args)), method
    assert torch.equal(unit.step(xt, flow, t, t_next, pred_type="flow"), rf.step(xt, flow, t, t_next, pred_type="flow"))
    assert not torch.allclose(shifted.sigma(t), rf.sigma(t))


def test_rf_network_time_is_sigma_scaled_by_num_train_timesteps():
    """RF networks see ``num_train_timesteps * sigma(t)``, by default the diffusers flow-matching scale 1000."""
    t = torch.tensor([0.0, 0.25, 0.999], dtype=torch.float64)
    assert torch.equal(get_noise_schedule("rf").rescale_t(t), 1000 * t)
    shifted = get_noise_schedule("shifted_rf", shift=3.0, num_train_timesteps=32)
    assert torch.equal(shifted.rescale_t(t), 32 * shifted.sigma(t))
    with pytest.raises(ValueError, match="num_train_timesteps"):
        get_noise_schedule("rf", num_train_timesteps=0)


def test_rf_flow_is_the_time_derivative_at_any_shift():
    """``flow`` means ``dx/dt`` for rectified flow too, which is ``sigma'(t) (eps - x0)``.

    The overrides exist to compute that in one multiply instead of the base's two-plus-an-add, not
    to hold a different convention -- so they have to agree with the base formula, and be the plain
    ``eps - x0`` exactly where ``sigma'`` is one.
    """
    x0, eps = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
    t = torch.linspace(0.05, 0.95, 4, dtype=torch.float64)[:, None]

    for shift in (1.0, 3.0 / 12.0, 3.0, 12.0):
        s = get_noise_schedule("shifted_rf", shift=shift, min_t=0.0, max_t=0.999)
        xt = s.alpha(t) * x0 + s.sigma(t) * eps
        flow = s.cond_velocity(x0, eps, t)

        torch.testing.assert_close(flow, BaseNoiseSchedule._cond_velocity(s, x0, eps, t), msg=f"shift {shift}")
        torch.testing.assert_close(flow, s.sigma_prime(t) * (eps - x0), msg=f"shift {shift}")
        # The conversions have to invert that same convention.
        torch.testing.assert_close(s.flow_to_x0(xt, flow, t), x0, msg=f"flow_to_x0 at shift {shift}")
        torch.testing.assert_close(s.x0_to_flow(xt, x0, t), flow, msg=f"x0_to_flow at shift {shift}")

    # Unshifted, the scaling is skipped rather than multiplied by a rounded one.
    unshifted = get_noise_schedule("rf", min_t=0.0, max_t=0.999)
    assert torch.equal(unshifted.cond_velocity(x0, eps, t), eps - x0)


@pytest.mark.parametrize(
    "schedule",
    [
        EDMNoiseSchedule(min_t=0.002, max_t=5.0),
        TrigNoiseSchedule(min_t=0.0, max_t=math.pi / 2),
    ],
    ids=["edm", "trig"],
)
def test_base_points_to_x0_recovers_batched_conditional_path(schedule):
    x0 = torch.tensor(
        [[[-1.25, 0.5], [0.75, 2.0]], [[0.4, -0.3], [1.1, -0.8]]],
        dtype=torch.float64,
    )
    eps = torch.tensor(
        [[[0.2, -0.7], [1.5, -0.4]], [[-0.6, 0.9], [0.3, 1.7]]],
        dtype=torch.float64,
    )
    if isinstance(schedule, EDMNoiseSchedule):
        t_start = torch.tensor([4.0, 2.5], dtype=torch.float64)
        t_end = torch.tensor([1.25, 0.4], dtype=torch.float64)
    else:
        t_start = torch.tensor([1.3, 1.0], dtype=torch.float64)
        t_end = torch.tensor([0.55, 0.2], dtype=torch.float64)

    x_start = schedule.forward_process(x0, eps, t_start)
    x_end = schedule.forward_process(x0, eps, t_end)

    actual = schedule.points_to_x0(x_start, x_end, t_start, t_end)

    torch.testing.assert_close(actual, x0, rtol=1e-12, atol=1e-12)


def test_base_points_to_x0_accepts_clean_endpoint_below_edm_min_t():
    schedule = EDMNoiseSchedule(min_t=0.002, max_t=5.0)
    x0 = torch.tensor([[0.25, -1.5]], dtype=torch.float64)
    eps = torch.tensor([[-0.4, 0.8]], dtype=torch.float64)
    t_start = torch.tensor([1.25], dtype=torch.float64)
    t_end = torch.zeros_like(t_start)
    x_start = schedule.forward_process(x0, eps, t_start)

    actual = schedule.points_to_x0(x_start, x0, t_start, t_end)

    torch.testing.assert_close(actual, x0, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "make_schedule",
    _SMOOTH_SCHEDULES + [pytest.param(_scaled_linear_alphas_schedule, id="alphas-scaled-linear")],
)
def test_points_to_x0_is_the_clean_grid_end_exactly(make_schedule):
    """Student grids end at 0, below EDM's ``min_t`` and at SD's noisy first discrete time; that endpoint is x0."""
    torch.manual_seed(0)
    schedule = make_schedule()
    x0, eps = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    t = _interior_times(schedule, n=5)
    x_t, t_clean = schedule.forward_process(x0, eps, t), torch.zeros_like(t)

    assert torch.equal(schedule.points_to_x0(x_t, x0, t, t_clean), x0)
    assert torch.equal(schedule._points_to_x0(x_t, x0, t, t_clean), x0)


def test_points_to_x0_returns_the_clean_grid_end_where_the_schedules_time_zero_is_noisy():
    """SD's first discrete time 0 carries noise, but an endpoint at the clean end 0 of a student grid is the data."""
    torch.manual_seed(0)
    schedule = _scaled_linear_alphas_schedule()
    assert schedule.sigma(torch.zeros(1, dtype=torch.float64)) > 0
    x0, eps = torch.randn(3, 4, dtype=torch.float64), torch.randn(3, 4, dtype=torch.float64)
    t_start = torch.tensor([0.9, 0.5, 0.3], dtype=torch.float64)
    t_end = torch.tensor([0.4, 0.0, 0.0], dtype=torch.float64)
    x_start = schedule.forward_process(x0, eps, t_start)
    x_end = schedule.forward_process(x0, eps, t_end)
    x_end[1:] = x0[1:]

    actual = schedule.points_to_x0(x_start, x_end, t_start, t_end)

    torch.testing.assert_close(actual, x0, rtol=1e-12, atol=1e-12)
    assert torch.equal(actual[1:], x0[1:])


def test_points_to_x0_can_use_the_schedules_noisy_time_zero():
    torch.manual_seed(0)
    schedule = _scaled_linear_alphas_schedule()
    x0, eps = torch.randn(3, 4, dtype=torch.float64), torch.randn(3, 4, dtype=torch.float64)
    t_start = torch.tensor([0.9, 0.5, 0.3], dtype=torch.float64)
    t_end = torch.zeros_like(t_start)
    x_start = schedule.forward_process(x0, eps, t_start)
    x_end = schedule.forward_process(x0, eps, t_end)

    torch.testing.assert_close(
        schedule.points_to_x0(x_start, x_end, t_start, t_end, clean_end=False), x0, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        schedule._points_to_x0(x_start, x_end, t_start, t_end, clean_end=False), x0, rtol=1e-12, atol=1e-12
    )


def test_points_to_x0_takes_the_clean_grid_end_where_the_schedule_points_are_singular():
    """Times of equal SNR are singular schedule points, here those up to the second entry of a table whose first two
    entries agree. The clean end 0 is still x0, with finite gradients; pairs without it are still checked."""
    torch.manual_seed(0)
    table = _scaled_linear_alphas_schedule().alphas_cumprod
    schedule = AlphasNoiseSchedule(alphas_cumprod=torch.cat([table[:1], table[:-1]]))
    x_start = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    x0 = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    t_start = torch.full((2,), 1 / schedule.num_train_timesteps, dtype=torch.float64)
    t_clean = torch.zeros_like(t_start)
    assert schedule._determinant(t_start, t_clean).eq(0).all()

    actual = schedule.points_to_x0(x_start, x0, t_start, t_clean)
    actual.sum().backward()

    assert torch.equal(actual, x0)
    assert torch.equal(x_start.grad, torch.zeros_like(x_start)) and torch.equal(x0.grad, torch.ones_like(x0))
    with pytest.raises(ValueError, match="singular"):
        schedule.points_to_x0(x_start, x0, 0.75 * t_start, 0.5 * t_start)


@pytest.mark.parametrize("shift", [0.25, 1.0, 12.0])
def test_rf_points_to_x0_recovers_shifted_paths_and_matches_local_flow(shift):
    schedule = ShiftedRFNoiseSchedule(shift=shift, min_t=0.0, max_t=0.999)
    x0 = torch.tensor([[0.25, -1.5], [2.0, 0.75]], dtype=torch.float64)
    eps = torch.tensor([[-0.4, 0.8], [1.25, -0.2]], dtype=torch.float64)
    t_start = torch.tensor([0.93, 0.71], dtype=torch.float64)
    t_end = torch.tensor([0.37, 0.0], dtype=torch.float64)
    x_start = schedule.forward_process(x0, eps, t_start)
    x_end = schedule.forward_process(x0, eps, t_end)

    from_points = schedule.points_to_x0(x_start, x_end, t_start, t_end)
    flow = schedule.cond_velocity(x0, eps, t_start)
    from_local_flow = schedule.flow_to_x0(x_start, flow, t_start)

    torch.testing.assert_close(from_points, x0, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(from_points, from_local_flow, rtol=1e-12, atol=1e-12)
    assert torch.equal(from_points[1], x_end[1])


def test_points_to_x0_preserves_endpoint_gradients_and_start_dtype():
    schedule = ShiftedRFNoiseSchedule(shift=4.0, min_t=0.0, max_t=0.999)
    x_start = torch.tensor([[0.5, -0.25]], dtype=torch.float32, requires_grad=True)
    x_end = torch.tensor([[0.1, 0.75]], dtype=torch.float64, requires_grad=True)

    x0 = schedule.points_to_x0(
        x_start,
        x_end,
        torch.tensor(0.8),
        torch.tensor(0.3),
    )

    assert x0.dtype == x_start.dtype
    x0.sum().backward()
    assert x_start.grad is not None
    assert x_end.grad is not None


def test_points_to_x0_rejects_unordered_and_out_of_range_times():
    schedule = ShiftedRFNoiseSchedule(shift=0.25, min_t=0.1, max_t=0.999)
    x_start = torch.zeros(2, 3)

    for t_end in (0.5, 0.7):
        with pytest.raises(ValueError, match="t_start > t_end"):
            schedule.points_to_x0(x_start, x_start + 1.0, torch.tensor(0.5), torch.tensor(t_end))
    # The exactly clean endpoint is allowed below min_t, but no other time below it, nor one above max_t; t_start is
    # never clean.
    schedule.points_to_x0(x_start, x_start + 1.0, torch.tensor(0.5), torch.tensor(0.0))
    for t_end in (0.05, 1.5):
        with pytest.raises(ValueError, match=r"t_end must be in \[0.1, 0.999\]"):
            schedule.points_to_x0(x_start, x_start + 1.0, torch.tensor(0.5), torch.tensor(t_end))
    with pytest.raises(ValueError, match=r"t_start must be in \[0.1, 0.999\]"):
        schedule.points_to_x0(x_start, x_start + 1.0, torch.tensor(0.05), torch.tensor(0.0))


@pytest.mark.parametrize(
    ("min_t", "max_t"),
    [(0.1, 0.999), (0.0, 0.95)],
    ids=["min_t", "max_t"],
)
def test_composite_requires_matching_subschedule_time_ranges(min_t, max_t):
    with pytest.raises(ValueError, match="time range"):
        CompositeNoiseSchedule(
            schedules={
                "video": {"name": "rf", "min_t": 0.0, "max_t": 0.999},
                "audio": {"name": "rf", "min_t": min_t, "max_t": max_t},
            },
            primary_key="video",
        )


def test_composite_requires_shared_timestep_precision():
    with pytest.raises(ValueError, match="all sub-schedules must use the same t_precision"):
        CompositeNoiseSchedule(
            schedules={
                "video": {"name": "rf", "t_precision": "float64"},
                "audio": {"name": "rf", "t_precision": "float32"},
            },
            primary_key="video",
        )


def test_composite_points_to_x0_dispatches_three_distinct_rf_shifts():
    schedule = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "shifted_rf", "shift": 0.25, "min_t": 0.0, "max_t": 0.999},
            "audio": {"name": "rf", "min_t": 0.0, "max_t": 0.999},
            "depth": {"name": "shifted_rf", "shift": 12.0, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )
    x0 = {
        "video": torch.tensor([[0.25, -0.5], [1.0, 0.75]], dtype=torch.float64),
        "audio": torch.tensor([[-0.2], [0.4]], dtype=torch.float64),
        "depth": torch.tensor([[0.9, -0.3, 0.1], [-0.6, 0.8, 0.2]], dtype=torch.float64),
    }
    eps = {key: torch.flip(value, dims=(-1,)) + 0.3 for key, value in x0.items()}
    t_start = torch.tensor([0.91, 0.68], dtype=torch.float64)
    t_end = torch.tensor([0.42, 0.0], dtype=torch.float64)
    x_start = schedule.forward_process(x0, eps, t_start)
    x_end = schedule.forward_process(x0, eps, t_end)

    actual = schedule.points_to_x0(x_start, x_end, t_start, t_end)

    assert set(actual) == set(x0)
    for key in x0:
        torch.testing.assert_close(actual[key], x0[key], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Prediction-type conversions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_schedule",
    _SMOOTH_SCHEDULES + [pytest.param(_cosine_alphas_schedule, id="alphas-cosine")],
)
def test_pred_type_conversions_roundtrip_through_every_pair(make_schedule):
    """Every src -> target conversion recovers the analytic ground truth.

    ``flow`` only exists on schedules with a differentiable clock;
    ``v``-prediction is only defined for ``AlphasNoiseSchedule``.
    """
    schedule = make_schedule()
    torch.manual_seed(0)
    x0 = torch.randn(7, 3, dtype=torch.float64)
    eps = torch.randn(7, 3, dtype=torch.float64)
    t = _interior_times(schedule)[:, None]
    xt = schedule.forward_process(x0, eps, t)

    truths = {"x0": x0, "eps": eps}
    if isinstance(schedule, AlphasNoiseSchedule):
        truths["v"] = schedule.cond_v(x0, eps, t)
    else:
        truths["flow"] = schedule.cond_velocity(x0, eps, t)

    for src_pred_type, source in truths.items():
        for target_pred_type, expected in truths.items():
            converted = schedule.convert_model_output(
                xt,
                source,
                t,
                src_pred_type=src_pred_type,
                target_pred_type=target_pred_type,
            )
            torch.testing.assert_close(
                converted,
                expected,
                rtol=1e-7,
                atol=1e-7,
                msg=f"{src_pred_type} -> {target_pred_type}",
            )


def test_cond_v_is_the_noise_at_a_clean_time():
    """Where ``sigma = 0``, ``xt = x0`` carries no noise, so the v target cannot be recovered from ``xt``."""
    schedule = _cosine_alphas_schedule()
    x0, eps = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
    t = torch.zeros(4, 1, dtype=torch.float64)

    assert torch.equal(schedule.sigma(t), torch.zeros_like(t))
    assert torch.equal(schedule.cond_v(x0, eps, t), eps)


@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_cond_v_is_only_defined_for_alphas_schedules(make_schedule):
    schedule = make_schedule()
    x0 = torch.randn(7, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="does not support v-prediction"):
        schedule.cond_v(x0, torch.randn_like(x0), _interior_times(schedule)[:, None])


def test_convert_model_output_accepts_native_prediction_dtype_and_controls_output_dtype():
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    xt = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    flow = torch.tensor([[0.75, -0.125]], dtype=torch.bfloat16)
    t = torch.tensor([[0.4]], dtype=torch.float64)

    x0 = schedule.convert_model_output(xt, flow, t, src_pred_type="flow", target_pred_type="x0")
    passthrough = schedule.convert_model_output(xt, flow, t, "flow", "flow", torch.float64)
    private_x0 = schedule._flow_to_x0(xt, flow, t)
    api_x0 = schedule.flow_to_x0(xt, flow, t, torch.float64)

    assert x0.dtype == xt.dtype
    assert passthrough.dtype == torch.float64
    assert private_x0.dtype == high_precision_dtype(xt, flow)
    assert api_x0.dtype == torch.float64
    torch.testing.assert_close(passthrough, flow.to(torch.float64))
    torch.testing.assert_close(api_x0, private_x0.to(api_x0.dtype))


def test_points_to_x0_controls_its_output_dtype():
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    x_start = torch.tensor([[0.25, -0.5]], dtype=torch.bfloat16)
    x_end = torch.tensor([[0.5, -0.125]], dtype=torch.float32)
    t_start, t_end = torch.tensor([0.75], dtype=torch.float64), torch.tensor([0.25], dtype=torch.float64)

    same = schedule.points_to_x0(x_start, x_end, t_start, t_end)
    uncast = schedule.points_to_x0(x_start, x_end, t_start, t_end, output_dtype=None)
    float32 = schedule.points_to_x0(x_start, x_end, t_start, t_end, output_dtype=torch.float32)

    assert uncast.dtype == high_precision_dtype(x_start, x_end)
    assert torch.equal(same, uncast.to(x_start.dtype))
    assert torch.equal(float32, uncast.float())


@pytest.mark.parametrize("time_dtype", [torch.float32, torch.bfloat16])
def test_convert_model_output_requires_scheduler_time_precision(time_dtype):
    schedule = RFNoiseSchedule()
    state = torch.randn(2, 3, dtype=torch.float32)
    prediction = torch.randn_like(state).to(torch.bfloat16)

    with pytest.raises(AssertionError, match=r"t\.dtype .*must be torch\.float64"):
        schedule.convert_model_output(
            state,
            prediction,
            torch.full((2,), 0.5, dtype=time_dtype),
            src_pred_type="flow",
            target_pred_type="x0",
        )


def test_convert_model_output_keeps_multistage_x0_in_high_precision():
    class _IntermediateDtypeProbe(RFNoiseSchedule):
        intermediate_x0_dtype = None

        def _x0_to_eps(self, xt, x0, t):
            self.intermediate_x0_dtype = x0.dtype
            return super()._x0_to_eps(xt, x0, t)

    schedule = _IntermediateDtypeProbe(min_t=0.0, max_t=0.999)
    xt = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    flow = torch.tensor([[0.75, -0.125]], dtype=torch.bfloat16)
    t = torch.tensor([[0.4]], dtype=torch.float64)

    eps = schedule.convert_model_output(xt, flow, t, src_pred_type="flow", target_pred_type="eps")

    assert schedule.intermediate_x0_dtype == high_precision_dtype(xt, flow)
    assert eps.dtype == xt.dtype


def test_conversions_keep_wider_states_and_ignore_the_time_precision(monkeypatch):
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)
    schedule = ShiftedRFNoiseSchedule(shift=3.0, min_t=0.0, max_t=0.999)
    xt = torch.tensor([[0.25, -0.5]], dtype=torch.float64)
    flow = torch.tensor([[0.75, -0.125]], dtype=torch.float64)
    t = torch.tensor([0.4], dtype=torch.float64)

    assert schedule._flow_to_x0(xt, flow, t).dtype == torch.float64
    assert schedule._flow_to_x0(xt.bfloat16(), flow.bfloat16(), t).dtype == torch.float32


@pytest.mark.parametrize(
    "make_schedule",
    _SMOOTH_SCHEDULES + [pytest.param(_cosine_alphas_schedule, id="alphas-cosine")],
)
@pytest.mark.parametrize(
    ("high_precision", "state_dtype"),
    [
        pytest.param(torch.float64, torch.float32, id="float64"),
        pytest.param(torch.float32, torch.bfloat16, id="float32-below-time-precision"),
    ],
)
def test_private_conversion_returns_the_high_precision_on_every_path(
    monkeypatch, make_schedule, high_precision, state_dtype
):
    """Every conversion path lands on the joint high precision of its states: it neither keeps a low-precision
    prediction nor widens to a time precision above it."""
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", high_precision)
    schedule = make_schedule()
    t = _interior_times(schedule)[:, None]
    xt = torch.randn(t.shape[0], 3).to(state_dtype)
    prediction = torch.randn_like(xt).to(torch.bfloat16)
    pred_types = ["x0", "eps", "v" if isinstance(schedule, AlphasNoiseSchedule) else "flow"]

    for src_pred_type in pred_types:
        for target_pred_type in pred_types:
            converted = schedule._convert_model_output(xt, prediction, t, src_pred_type, target_pred_type)
            assert converted.dtype == high_precision, f"{src_pred_type} -> {target_pred_type}"


def test_step_keeps_a_time_precision_wider_than_its_states(monkeypatch):
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    t = torch.tensor([0.5], dtype=torch.float64)

    # A step of 1e-9 rounds away in float32 times but not in float64 ones.
    x_next = schedule.step(torch.zeros(1, 2), torch.ones(1, 2), t, t - 1e-9, pred_type="flow")

    assert x_next.dtype == torch.float32
    torch.testing.assert_close(x_next, torch.full((1, 2), -1e-9), rtol=1e-6, atol=0)


def test_time_precision_takes_a_dtype_and_defaults_to_the_high_precision_at_construction(monkeypatch):
    assert get_noise_schedule("rf").t_precision is torch.float64
    assert get_noise_schedule("rf", t_precision=torch.float32).t_precision is torch.float32
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)
    assert get_noise_schedule("rf", t_precision=None).t_precision is torch.float32


def test_composite_conversion_returns_each_state_dtype_without_prediction_precasts():
    schedule = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "shifted_rf", "shift": 3.0, "min_t": 0.0, "max_t": 0.999},
            "audio": {"name": "rf", "min_t": 0.0, "max_t": 0.999},
        }
    )
    xt = {
        "video": torch.tensor([[0.25, -0.5]], dtype=torch.float32),
        "audio": torch.tensor([[0.75]], dtype=torch.float64),
    }
    flow = {
        "video": torch.tensor([[0.5, -0.25]], dtype=torch.bfloat16),
        "audio": torch.tensor([[-0.125]], dtype=torch.float32),
    }
    t = torch.tensor([[0.4]], dtype=torch.float64)

    x0 = schedule.convert_model_output(xt, flow, t, src_pred_type="flow", target_pred_type="x0")

    assert {key: value.dtype for key, value in x0.items()} == {
        "video": torch.float32,
        "audio": torch.float64,
    }


# ---------------------------------------------------------------------------
# Signal-to-noise ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_sqrt_snr_matches_alpha_over_sigma_decreases_and_round_trips(make_schedule):
    schedule = make_schedule()
    t = _interior_times(schedule)

    sqrt_snr = schedule.sqrt_snr(t)
    torch.testing.assert_close(sqrt_snr, schedule.alpha(t) / schedule.sigma(t), rtol=1e-10, atol=0.0)
    assert torch.all(sqrt_snr[1:] < sqrt_snr[:-1]), "SNR must strictly decrease in t"
    torch.testing.assert_close(schedule.sqrt_snr_to_t(sqrt_snr), t, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sqrt_snr_coefficients_preserve_limits_values_and_derivatives(device_type, dtype):
    from decimal import Decimal, localcontext

    values = [0.0, 1e-20, 0.5, 1.0, 2.0, 1e10, 1e20, math.inf]
    if dtype == torch.float64:
        values += [math.ulp(0.0), 1e-160, 1e160, 1e308]
    r = torch.tensor(values, dtype=dtype, device=device_type, requires_grad=True)
    with localcontext() as ctx:
        ctx.prec = 100
        expected_alpha, expected_sigma, expected_da, expected_ds = [], [], [], []
        for value in r.detach().cpu().tolist():
            if math.isinf(value):
                a, s, da, ds = 1.0, 0.0, 0.0, 0.0
            else:
                x = Decimal.from_float(value)
                norm = (1 + x * x).sqrt()
                a, s, da, ds = float(x / norm), float(1 / norm), float(1 / norm**3), float(-x / norm**3)
            expected_alpha.append(a)
            expected_sigma.append(s)
            expected_da.append(da)
            expected_ds.append(ds)
    alpha, sigma = sqrt_snr_to_alpha_sigma(r)
    tolerance = 5e-7 if dtype == torch.float32 else 1e-14
    for index, actual, values, derivatives in (
        (0, alpha, expected_alpha, expected_da),
        (1, sigma, expected_sigma, expected_ds),
    ):
        expected = torch.tensor(values, dtype=dtype, device=device_type)
        derivative = torch.tensor(derivatives, dtype=dtype, device=device_type)
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), r, retain_graph=True)[0], derivative, rtol=tolerance, atol=0
        )
        tangent = torch.func.jvp(sqrt_snr_to_alpha_sigma, (r,), (torch.ones_like(r),))[1][index]
        torch.testing.assert_close(tangent, derivative, rtol=tolerance, atol=0)


def _time_and_coefficients(schedule, sqrt_snr):
    """sCM's SNR matching: the inverse time and the schedule's coefficients at that time."""
    t = schedule.sqrt_snr_to_t(sqrt_snr)
    return t, *schedule.alpha_sigma(t)


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("schedule_name", ["rf", "edm", "trig"])
def test_sqrt_snr_to_t_preserves_each_schedules_normalization(device_type, dtype, schedule_name):
    schedule = get_noise_schedule(schedule_name).to(device_type)
    sqrt_snr = torch.tensor([0.5, 1.0, 2.0], device=device_type, dtype=dtype, requires_grad=True)

    def reference(sqrt_snr):
        if schedule_name == "rf":
            sigma = 1 / (1 + sqrt_snr)
            return sigma, 1 - sigma, sigma
        if schedule_name == "edm":
            sigma = 1 / sqrt_snr
            return sigma, torch.ones_like(sigma), sigma
        t = torch.atan(1 / sqrt_snr)
        return t, t.cos(), t.sin()

    actual, tangent = torch.func.jvp(
        lambda r: _time_and_coefficients(schedule, r), (sqrt_snr,), (torch.ones_like(sqrt_snr),)
    )
    expected, expected_tangent = torch.func.jvp(reference, (sqrt_snr,), (torch.ones_like(sqrt_snr),))
    tolerance = 3e-7 if dtype == torch.float32 else 1e-14
    for value, truth, derivative, derivative_truth in zip(actual, expected, tangent, expected_tangent):
        assert value.dtype == dtype and value.device == sqrt_snr.device
        torch.testing.assert_close(value, truth, rtol=tolerance, atol=0)
        torch.testing.assert_close(derivative, derivative_truth, rtol=tolerance, atol=0)
    gradient = torch.autograd.grad(sum(value.sum() for value in actual), sqrt_snr)[0]
    torch.testing.assert_close(gradient, sum(expected_tangent), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
def test_alphas_sqrt_snr_to_t_resolves_pure_noise(device_type):
    """Into the pure-noise entry ``alpha = 1 - 2t`` is linear, so the inverse time resolves a tiny SNR, the
    coefficients at that time are the SNR's, and all three keep finite derivatives at ``sqrt_snr = 0``."""
    schedule = AlphasNoiseSchedule(torch.tensor([1.0, 0.0], dtype=torch.float64)).to(device_type)
    sqrt_snr = torch.tensor([0.0, 1e-10], device=device_type, dtype=torch.float64, requires_grad=True)
    (t, alpha, sigma), tangents = torch.func.jvp(
        lambda r: _time_and_coefficients(schedule, r), (sqrt_snr,), (torch.ones_like(sqrt_snr),)
    )
    r = sqrt_snr.detach()
    torch.testing.assert_close(t, (1 - r) / 2, rtol=0, atol=4e-16)
    torch.testing.assert_close(alpha, r, rtol=0, atol=4e-16)
    torch.testing.assert_close(sigma, torch.ones_like(sigma), rtol=0, atol=0)
    ones = torch.ones_like(r)
    for value, tangent, expected in zip((t, alpha, sigma), tangents, (-ones / 2, ones, -r)):
        torch.testing.assert_close(tangent, expected, rtol=1e-5, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(value.sum(), sqrt_snr, retain_graph=True)[0], expected, rtol=1e-5, atol=0
        )


@pytest.mark.parametrize("probability, times", [(0.0, [0, 0, 0]), (0.5, [0.5, 0, 0]), (1.0, [0.5, 0.5, 0])])
def test_alphas_sqrt_snr_to_t_keeps_constant_tables_constant(probability, times):
    """A constant table is one plateau: an SNR below the table's saturates at its last knot, the table's own and
    higher ones select its first knot, and the coefficients stay constant."""
    schedule = AlphasNoiseSchedule(torch.tensor([probability, probability], dtype=torch.float64))
    sqrt_snr = torch.tensor([0.0, 1.0, math.inf], dtype=torch.float64, requires_grad=True)
    values, tangents = torch.func.jvp(
        lambda r: _time_and_coefficients(schedule, r), (sqrt_snr,), (torch.ones_like(sqrt_snr),)
    )
    p = schedule.alphas_cumprod[0]
    expected_values = (torch.tensor(times, dtype=torch.float64), p.sqrt(), (1 - p).sqrt())
    for value, tangent, expected in zip(values, tangents, expected_values):
        torch.testing.assert_close(value, expected.expand_as(value), rtol=0, atol=0)
        torch.testing.assert_close(tangent, torch.zeros_like(value), rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(value.sum(), sqrt_snr, retain_graph=True)[0], torch.zeros_like(value)
        )


def test_alphas_sqrt_snr_to_t_saturates_at_the_schedule_range_and_inverts_plateaus_at_their_first_knot():
    schedule = AlphasNoiseSchedule(torch.tensor([0.8, 0.5, 0.5, 0.2], dtype=torch.float64), min_t=0.2, max_t=0.4)
    sqrt_snr = torch.tensor([0.0, 1.0, math.inf], dtype=torch.float64, requires_grad=True)
    t = schedule.sqrt_snr_to_t(sqrt_snr)
    # The table saturates at its own ends, which the clamp takes to the narrower range.
    torch.testing.assert_close(t, torch.tensor([0.4, 0.25, 0.2], dtype=t.dtype), rtol=0, atol=1e-15)
    gradient = torch.autograd.grad(t.sum(), sqrt_snr)[0]
    torch.testing.assert_close(gradient[[0, 2]], torch.zeros(2, dtype=t.dtype), rtol=0, atol=0)


def test_alphas_schedule_bounds_max_t_by_the_tables_last_knot():
    """The admissible range ends at the last knot ``(N - 1) / N`` of an ``N``-entry table, the default ``max_t`` (0.999
    for 1000 entries): above it every time would share the knot's coefficients while the network sees distinct
    indices."""
    table = torch.linspace(0.99, 0.01, 50, dtype=torch.float64)
    assert AlphasNoiseSchedule(alphas_cumprod=table).max_t == 49 / 50
    assert _scaled_linear_alphas_schedule().max_t == 0.999 == get_noise_schedule("ddpm").max_t
    with pytest.raises(ValueError, match="last knot"):
        AlphasNoiseSchedule(alphas_cumprod=table, max_t=0.999)
    long_table = torch.linspace(0.99, 0.01, 1024, dtype=torch.float64)
    assert AlphasNoiseSchedule(alphas_cumprod=long_table, max_t=1023 / 1024).max_t == 1023 / 1024
    with pytest.raises(ValueError, match="last knot"):
        AlphasNoiseSchedule(alphas_cumprod=long_table, max_t=1.0)


def test_ddpm_schedule_is_the_linear_ddpm_table():
    """ADM and DiT's ``get_named_beta_schedule("linear", 1000)``: the float64 cumulative product of betas from 1e-4 to
    0.02 over 1000 steps, far from SD1.5's scaled-linear schedule (0.0047 at the end)."""
    schedule = get_noise_schedule("ddpm")
    expected = torch.from_numpy(np.cumprod(1.0 - np.linspace(0.0001, 0.02, 1000, dtype=np.float64)))

    assert schedule.num_train_timesteps == 1000 and schedule.alphas_cumprod.dtype == torch.float64
    assert torch.equal(schedule.alphas_cumprod, expected)
    assert schedule.alphas_cumprod[-1].item() < 5e-5
    t = torch.tensor([0.0, 0.5, 0.999], dtype=torch.float64)
    torch.testing.assert_close(schedule.alpha(t), expected[[0, 500, 999]].sqrt(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_train_timesteps", [1000, 1024])
@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
def test_alphas_schedule_is_its_table_at_integer_indices(num_train_timesteps, device_type):
    """At ``t = k / num_train_timesteps`` the network sees the integer index ``k``, and alpha and sigma are exactly the
    table's ``sqrt(alphas_cumprod[k])`` and ``sqrt(1 - alphas_cumprod[k])``."""
    betas = torch.linspace(0.00085**0.5, 0.012**0.5, num_train_timesteps, dtype=torch.float64) ** 2
    table = torch.cumprod(1 - betas, dim=0)
    schedule = AlphasNoiseSchedule(alphas_cumprod=table).to(device_type)
    assert schedule.num_train_timesteps == num_train_timesteps
    k = torch.arange(int(0.999 * num_train_timesteps) + 1)
    t = (k.to(torch.float64) / num_train_timesteps).to(device_type)
    k, table = k.to(device_type), table.to(device_type)

    assert torch.equal(schedule.rescale_t(t), k.to(torch.float64))
    assert torch.equal(schedule.alpha(t), table[k].sqrt())
    assert torch.equal(schedule.sigma(t), (1 - table[k]).sqrt())


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
def test_alphas_schedule_interpolates_alpha_squared_linearly_between_entries(device_type):
    """On a table without a pure-noise entry, signal and noise probabilities are affine in time on every segment."""
    schedule = _scaled_linear_alphas_schedule()
    schedule = schedule.to(device_type)
    table = schedule.alphas_cumprod
    k = torch.arange(0, 999, 37, device=device_type)

    for weight in (1e-9, 0.25, 0.5, 0.75, 1 - 1e-9):
        t = ((k.cpu().to(torch.float64) + weight) / schedule.num_train_timesteps).to(device_type)
        alpha, sigma = schedule.alpha(t), schedule.sigma(t)
        expected = (1 - weight) * table[k] + weight * table[k + 1]
        torch.testing.assert_close(alpha.square(), expected, rtol=0, atol=1e-15)
        torch.testing.assert_close(sigma.square(), 1 - expected, rtol=0, atol=1e-15)
        torch.testing.assert_close(alpha.square() + sigma.square(), torch.ones_like(t), rtol=0, atol=1e-15)
    # Continuous at both ends of every interval.
    left, right = (k.to(torch.float64) + 1e-9) / 1000, (k.to(torch.float64) + 1 - 1e-9) / 1000
    torch.testing.assert_close(schedule.alpha(left), table[k].sqrt(), rtol=1e-9, atol=0)
    torch.testing.assert_close(schedule.sigma(right), (1 - table[k + 1]).sqrt(), rtol=1e-9, atol=0)
    t = torch.linspace(0.0, 0.999, 99_901, dtype=torch.float64, device=device_type)
    assert (schedule.alpha(t).diff() < 0).all() and (schedule.sigma(t).diff() > 0).all()


def test_alphas_sqrt_snr_to_t_inverts_sqrt_snr_between_entries():
    """``sqrt_snr_to_t`` inverts the linear-alpha-squared interpolation, on and between entries, and maps values beyond
    the table's to its ends."""
    schedule = _scaled_linear_alphas_schedule()
    t = torch.linspace(0.0, 0.999, 9_991, dtype=torch.float64)
    torch.testing.assert_close(schedule.sqrt_snr_to_t(schedule.sqrt_snr(t)), t, rtol=0, atol=1e-12)
    torch.testing.assert_close(schedule.sqrt_snr_to_t(schedule.sqrt_snr(t[1234])), t[1234], rtol=0, atol=1e-12)

    ends = schedule.sqrt_snr(torch.tensor([0.0, 0.999], dtype=torch.float64))
    beyond = torch.tensor([math.inf, 2 * ends[0], 0.5 * ends[1], 0.0], dtype=torch.float64)
    assert schedule.sqrt_snr_to_t(beyond).tolist() == [0.0, 0.0, 0.999, 0.999]


@pytest.mark.parametrize("entry", [999, 0], ids=["pure-noise-last", "clean-first"])
def test_alphas_schedule_interpolates_next_to_an_exactly_clean_or_pure_noise_entry(entry):
    """An exactly clean entry keeps ``alpha^2`` linear like every other entry; into an exactly pure-noise entry
    ``alpha`` itself is linear, so it vanishes with a finite slope. Both stay continuous and monotone."""
    table = _scaled_linear_alphas_schedule().alphas_cumprod.clone()
    table[entry] = 0.0 if entry else 1.0
    schedule = AlphasNoiseSchedule(alphas_cumprod=table)
    k = min(entry, 998)
    weight = torch.linspace(0, 1, 101, dtype=torch.float64)
    t = (k + weight) / schedule.num_train_timesteps
    alpha, sigma = schedule.alpha(t), schedule.sigma(t)
    if entry:
        torch.testing.assert_close(alpha, (1 - weight) * table[k].sqrt(), rtol=1e-9, atol=1e-16)
        torch.testing.assert_close(sigma.square(), 1 - alpha.square(), rtol=0, atol=1e-15)
    else:
        torch.testing.assert_close(alpha.square(), torch.lerp(table[k], table[k + 1], weight), rtol=1e-9, atol=1e-15)
        torch.testing.assert_close(
            sigma.square(), torch.lerp(1 - table[k], 1 - table[k + 1], weight), rtol=1e-9, atol=1e-15
        )
    assert alpha[0] == table[k].sqrt() and alpha[-1] == table[k + 1].sqrt()
    assert sigma[0] == (1 - table[k]).sqrt() and sigma[-1] == (1 - table[k + 1]).sqrt()
    torch.testing.assert_close(schedule.sqrt_snr_to_t(schedule.sqrt_snr(t)), t, rtol=0, atol=1e-12)

    # Continuous at both entries of the segment and monotone through the segments around it, without NaN or inf.
    for index in (k, k + 1):
        near = (
            torch.tensor([index - 1e-12, index + 1e-12], dtype=torch.float64).clamp(0, 999)
            / schedule.num_train_timesteps
        )
        exact = torch.full_like(near, index / schedule.num_train_timesteps)
        torch.testing.assert_close(schedule.alpha(near), schedule.alpha(exact), rtol=0, atol=1e-6)
        torch.testing.assert_close(schedule.sigma(near), schedule.sigma(exact), rtol=0, atol=1e-6)
    t = torch.linspace(max(k - 2, 0), min(k + 3, 999), 5_001, dtype=torch.float64) / schedule.num_train_timesteps
    alpha, sigma = schedule.alpha(t), schedule.sigma(t)
    assert torch.isfinite(alpha).all() and torch.isfinite(sigma).all()
    assert (alpha.diff() < 0).all() and (sigma.diff() > 0).all()
    sqrt_snr = torch.tensor([0.0, math.inf], dtype=torch.float64)
    assert schedule.sqrt_snr_to_t(sqrt_snr).tolist() == [0.999, 0.0]


def test_alphas_schedule_matches_diffusers_at_the_official_timesteps():
    """On the integer timesteps of diffusers' SD samplers, the network time, alpha and sigma are the schedulers' own:
    DDIM's ``alphas_cumprod`` and Euler's sigmas, from the same scaled-linear betas."""
    from diffusers import DDIMScheduler, EulerDiscreteScheduler

    config = {"beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear", "steps_offset": 1}
    ddim = DDIMScheduler(**config, clip_sample=False, set_alpha_to_one=False)
    ddim.set_timesteps(50)
    euler = EulerDiscreteScheduler(**config, timestep_spacing="leading")
    euler.set_timesteps(50)
    assert ddim.timesteps[0] == 981 and ddim.timesteps[-1] == 1
    assert torch.equal(euler.timesteps, ddim.timesteps.to(euler.timesteps.dtype))
    schedule = AlphasNoiseSchedule(alphas_cumprod=ddim.alphas_cumprod)

    timesteps = torch.cat([torch.tensor([999]), ddim.timesteps])
    t = timesteps.to(torch.float64) / 1000
    alphas_cumprod = ddim.alphas_cumprod[timesteps].to(torch.float64)
    assert torch.equal(schedule.rescale_t(t), timesteps.to(torch.float64))
    assert torch.equal(schedule.alpha(t), alphas_cumprod.sqrt())
    assert torch.equal(schedule.sigma(t), (1 - alphas_cumprod).sqrt())
    # Euler's sigmas are the variance-exploding sigma / alpha, in float32.
    ve_sigma = schedule.sigma(t[1:]) / schedule.alpha(t[1:])
    torch.testing.assert_close(ve_sigma.float(), euler.sigmas[:-1], rtol=1e-6, atol=0)


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("endpoint_segment", [False, True])
def test_alphas_autograd_and_jvp_preserve_the_time_derivative(device_type, endpoint_segment):
    if endpoint_segment:
        # Into the pure-noise entry alpha = 1 - 2t, so alpha = 1/2 and sigma = sqrt(3)/2 at t = 1/4.
        values, time = [1.0, 0.0], 0.25
        alpha, sigma, alpha_prime = 0.5, math.sqrt(3) / 2, -2.0
    else:
        # The two linear-probability segments meeting at p=.5 have the SAME slope, so this knot is differentiable.
        values, time = [0.9, 0.75, 0.5, 0.25], 0.5
        alpha = sigma = 1 / math.sqrt(2)
        alpha_prime = -alpha
    sigma_prime = -alpha * alpha_prime / sigma
    schedule = AlphasNoiseSchedule(torch.tensor(values, dtype=torch.float64)).to(device_type)
    t = torch.tensor([time], dtype=torch.float64, device=device_type, requires_grad=True)

    def forward_process(t):
        # Including when both coefficients share a graph.
        return schedule.forward_process(torch.full_like(t, 2), torch.full_like(t, 3), t)

    torch.testing.assert_close(forward_process(t), torch.full_like(t, 2 * alpha + 3 * sigma), rtol=1e-12, atol=0)
    for getter, derivative in (
        (schedule.alpha, alpha_prime),
        (schedule.sigma, sigma_prime),
        (forward_process, 2 * alpha_prime + 3 * sigma_prime),
    ):
        expected = torch.full_like(t, derivative)
        torch.testing.assert_close(torch.autograd.grad(getter(t).sum(), t)[0], expected, rtol=1e-12, atol=0)
        torch.testing.assert_close(torch.func.jvp(getter, (t,), (torch.ones_like(t),))[1], expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("values", [[1.0, 0.0], [1.0, 0.8, 0.0]], ids=["clean-to-pure-noise", "clean-to-noisy"])
def test_alphas_sigma_slope_at_an_exactly_clean_entry_is_infinite_not_nan(device_type, values):
    """At an exactly clean entry sigma rises like a square root, so its slope is inf in both AD modes, not NaN."""
    schedule = AlphasNoiseSchedule(torch.tensor(values, dtype=torch.float64)).to(device_type)
    t = torch.zeros(1, dtype=torch.float64, device=device_type, requires_grad=True)
    assert torch.isposinf(torch.autograd.grad(schedule.sigma(t).sum(), t)[0]).all()
    assert torch.isposinf(torch.func.jvp(schedule.sigma, (t,), (torch.ones_like(t),))[1]).all()


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize(
    "values, target, expected",
    [
        ([0.9, 0.5, 0.5], 1.0, 1 / 3),
        ([0.5, 0.5, 0.25], 1.0, 0.0),
        ([0.5, 0.5], 1.0, 0.0),
        ([0.0, 0.0], 0.0, 0.0),
        ([1.0, 1.0], math.inf, 0.0),
    ],
)
def test_alphas_inverse_selects_the_first_preimage_on_plateaus(device_type, values, target, expected):
    schedule = AlphasNoiseSchedule(torch.tensor(values, dtype=torch.float64)).to(device_type)
    r = torch.tensor([target], dtype=torch.float64, device=device_type, requires_grad=True)
    t = schedule.sqrt_snr_to_t(r)
    # The inverse reconstructs through square roots and interpolation. Allow float64 rounding
    # at nonzero knots while still requiring zero endpoints exactly and the first plateau knot.
    tolerance = 4 * math.ulp(expected) if expected else 0
    torch.testing.assert_close(t, torch.full_like(r, expected), rtol=0, atol=tolerance)
    assert torch.isfinite(torch.autograd.grad(t.sum(), r)[0]).all()


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("probability", [0.4, 0.99])
def test_alphas_inverse_does_not_skip_plateaus_due_to_rounding(device_type, probability):
    table = torch.tensor([1.0, probability, probability, 0.0], dtype=torch.float64)
    schedule = AlphasNoiseSchedule(table).to(device_type)
    first = torch.tensor([0.25], dtype=torch.float64, device=device_type)
    # Recovering a probability from rounded alpha/sigma can cross a search boundary, skipping from .25 to .5.
    torch.testing.assert_close(schedule.sqrt_snr_to_t(schedule.sqrt_snr(first)), first, rtol=0, atol=1e-15)


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize(
    "make_schedule",
    [_scaled_linear_alphas_schedule, lambda: get_noise_schedule("cogvideox", snr_shift_scale=1)],
    ids=["finite-snr", "zero-terminal-snr"],
)
def test_alphas_inverse_keeps_its_slope_at_the_entries_own_sqrt_snr(device_type, make_schedule):
    """At an entry's own sqrt SNR the inverse lands on its knot, where rounding can cross the clamps that keep the time
    in its segment. The slope survives as one of the two one-sided slopes there."""
    schedule = make_schedule().to(device_type)
    table = schedule.alphas_cumprod
    k = torch.arange(1, schedule.num_train_timesteps - 1, device=device_type)
    r = (table.sqrt() / (1 - table).sqrt())[k]
    width = 1 / schedule.num_train_timesteps

    def one_sided(p, q):
        # alpha^2 runs linearly from p to q, or into pure noise alpha = sqrt(p) * complement.
        linear = width / (q - p) * 2 * r / (1 + r.square()).square()
        return torch.where(q == 0, -width / (p.sqrt() * (1 + r.square()).pow(1.5)), linear)

    left, right = one_sided(table[k - 1], table[k]), one_sided(table[k], table[k + 1])
    grad_input = r.clone().requires_grad_(True)
    for slope in (
        torch.autograd.grad(schedule.sqrt_snr_to_t(grad_input).sum(), grad_input)[0],
        torch.func.jvp(schedule.sqrt_snr_to_t, (r,), (torch.ones_like(r),))[1],
    ):
        mismatch = torch.minimum((slope - left).abs() / left.abs(), (slope - right).abs() / right.abs())
        assert (mismatch <= 1e-12).all()


@pytest.mark.parametrize(
    "values, check",
    [
        ([], "length"),
        ([0.5], "length"),
        ([[0.5, 0.4]], "length"),
        ([0.5, 0.6], "values"),
        ([1.01, 0.5], "values"),
        ([0.5, -0.01], "values"),
        ([math.nan, 0.5], "values"),
        ([math.inf, 0.5], "values"),
    ],
)
def test_alphas_rejects_invalid_tables(values, check):
    """Each table fails the check it violates; tables of two or more entries reach the value check."""
    match = {"length": "vector of at least two entries", "values": r"finite, nonincreasing and in \[0, 1\]"}[check]
    with pytest.raises(ValueError, match=match):
        AlphasNoiseSchedule(torch.tensor(values, dtype=torch.float64))


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
def test_alphas_zero_snr_inverse_has_a_finite_derivative(device_type):
    """Into the pure-noise entry alpha is linear, so at r = 0 the inverse has the slope ``-width / sqrt(p)``."""
    schedule = get_noise_schedule("cogvideox", snr_shift_scale=1).to(device_type)
    r = torch.tensor([0.0], dtype=torch.float64, device=device_type, requires_grad=True)
    torch.testing.assert_close(schedule.sqrt_snr_to_t(r), torch.full_like(r, 0.999), rtol=0, atol=0)
    expected = torch.full_like(r, -(0.999 - 0.998) / schedule.alphas_cumprod[-2].sqrt().item())
    torch.testing.assert_close(torch.autograd.grad(schedule.sqrt_snr_to_t(r).sum(), r)[0], expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(
        torch.func.jvp(schedule.sqrt_snr_to_t, (r,), (torch.ones_like(r),))[1], expected, rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
def test_alphas_inverse_preserves_finite_gradients_with_subnormal_endpoint_probabilities(device_type):
    # Into pure noise t(r) = .5 - .5*alpha(r)/sqrt(p). Its derivative is finite even though 1/p overflows.
    p = math.ulp(0.0)
    schedule = AlphasNoiseSchedule(torch.tensor([p, 0.0], dtype=torch.float64)).to(device_type)
    r = torch.tensor([0.0, 1e-320, 1e-170, 1e-163], dtype=torch.float64, device=device_type, requires_grad=True)
    expected = torch.full_like(r.detach(), -0.5 / math.sqrt(p))
    torch.testing.assert_close(torch.autograd.grad(schedule.sqrt_snr_to_t(r).sum(), r)[0], expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(
        torch.func.jvp(schedule.sqrt_snr_to_t, (r,), (torch.ones_like(r),))[1], expected, rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize(
    "values",
    [
        [1.0, 0.0],
        [0.95, 0.8],
        [0.9, 0.1],
        [0.4, 0.1],
        [1.0, math.nextafter(1.0, 0.0)],
        [1e-310, 1e-320],
        [math.ulp(0.0), 0.0],
    ],
)
def test_alphas_inverse_matches_high_precision_values_and_derivatives(device_type, values):
    from decimal import Decimal, localcontext

    schedule = AlphasNoiseSchedule(torch.tensor(values, dtype=torch.float64)).to(device_type)
    with localcontext() as ctx:
        ctx.prec = 100
        p, q = map(Decimal.from_float, values)
        probabilities = [p + w * (q - p) for w in map(Decimal, ["0.1", "0.5", "0.9"])]
        snrs = [float((a / (1 - a)).sqrt()) for a in probabilities]
        # Evaluate the inverse at the actual rounded SNR input, independently of schedule evaluation.
        targets = list(map(Decimal.from_float, snrs))
        if q == 0:
            # Into the pure-noise entry alpha = sqrt(p) (1 - 2t) is linear in time.
            times = [float((1 - r / ((1 + r * r).sqrt() * p.sqrt())) / 2) for r in targets]
            derivatives = [float(-1 / (2 * p.sqrt() * (1 + r * r).sqrt() ** 3)) for r in targets]
        else:
            times = [float((p - r * r / (1 + r * r)) / (2 * (p - q))) for r in targets]
            derivatives = [float(-r / ((p - q) * (1 + r * r) ** 2)) for r in targets]
    r = torch.tensor(snrs, dtype=torch.float64, device=device_type, requires_grad=True)
    expected = torch.tensor(derivatives, dtype=r.dtype, device=r.device)
    torch.testing.assert_close(
        schedule.sqrt_snr_to_t(r), torch.tensor(times, dtype=r.dtype, device=r.device), rtol=0, atol=2e-15
    )
    torch.testing.assert_close(torch.autograd.grad(schedule.sqrt_snr_to_t(r).sum(), r)[0], expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(
        torch.func.jvp(schedule.sqrt_snr_to_t, (r,), (torch.ones_like(r),))[1], expected, rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
def test_alphas_roots_survive_when_interpolated_probabilities_underflow(device_type):
    from decimal import Decimal, localcontext

    for values, time, coefficient in (
        ([math.ulp(0.0), 0.0], 0.25, "alpha"),
        ([1.0, math.nextafter(1.0, 0.0)], math.ulp(0.0), "sigma"),
        ([1.0, 0.0], math.ulp(0.0), "sigma"),
    ):
        schedule = AlphasNoiseSchedule(torch.tensor(values, dtype=torch.float64)).to(device_type)
        t = torch.tensor([time], dtype=torch.float64, device=device_type, requires_grad=True)
        with localcontext() as ctx:
            ctx.prec = 400
            p, q = map(Decimal.from_float, values)
            if q == 0:
                # Into the pure-noise entry alpha = sqrt(p) (1 - 2t) is linear in time.
                alpha = p.sqrt() * (1 - 2 * Decimal.from_float(time))
                root, derivative = alpha, -2 * p.sqrt()
                if coefficient == "sigma":
                    root = (1 - alpha * alpha).sqrt()
                    derivative = -alpha * derivative / root
            else:
                probability = p + 2 * Decimal.from_float(time) * (q - p)
                if coefficient == "sigma":
                    probability = 1 - probability
                root = probability.sqrt()
                derivative = (q - p if coefficient == "alpha" else p - q) / root
        getter = getattr(schedule, coefficient)
        torch.testing.assert_close(getter(t), torch.full_like(t, float(root)), rtol=1e-12, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(getter(t).sum(), t)[0], torch.full_like(t, float(derivative)), rtol=1e-12, atol=0
        )


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
def test_alphas_tiny_probabilities_keep_representable_coefficients_and_gradients(device_type):
    from decimal import Decimal, localcontext

    table = torch.tensor([1e-310, 1e-320], dtype=torch.float64)
    schedule = AlphasNoiseSchedule(table).to(device_type)
    t = torch.tensor([0.25], dtype=torch.float64, device=device_type, requires_grad=True)
    with localcontext() as ctx:
        ctx.prec = 100
        p, q = map(Decimal.from_float, table.tolist())
        alpha = ((p + q) / 2).sqrt()
        derivative = (q - p) / alpha
    torch.testing.assert_close(schedule.alpha(t), torch.full_like(t, float(alpha)), rtol=1e-12, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(schedule.alpha(t).sum(), t)[0], torch.full_like(t, float(derivative)), rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize(
    "schedule_name, kwargs",
    [
        ("ddpm", {}),
        ("cogvideox", {"snr_shift_scale": 1}),
        ("cogvideox", {"snr_shift_scale": 3}),
    ],
)
def test_alphas_float32_interpolation_stays_monotone_across_knots(monkeypatch, device_type, schedule_name, kwargs):
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)
    schedule = get_noise_schedule(schedule_name, **kwargs).to(device_type)
    t = torch.linspace(0, 0.999, 100_001, dtype=torch.float32, device=device_type)
    alpha, sigma = schedule.alpha(t), schedule.sigma(t)
    assert (alpha.diff() <= 0).all() and (sigma.diff() >= 0).all()
    torch.testing.assert_close(alpha.square() + sigma.square(), torch.ones_like(t), rtol=0, atol=3e-7)


@pytest.mark.parametrize(
    "make_other",
    [
        pytest.param(lambda: ShiftedRFNoiseSchedule(shift=0.25, min_t=0.0, max_t=0.999), id="rf-shift0.25"),
        pytest.param(lambda: ShiftedRFNoiseSchedule(shift=6.0, min_t=0.0, max_t=0.999), id="rf-shift6"),
        pytest.param(lambda: EDMNoiseSchedule(min_t=0.002, max_t=5.0), id="edm"),
        pytest.param(lambda: TrigNoiseSchedule(min_t=0.0, max_t=math.pi / 2), id="trig"),
        pytest.param(_scaled_linear_alphas_schedule, id="alphas-scaled-linear"),
    ],
)
def test_matched_snr_times_agree_on_the_alpha_normalized_state(make_other):
    """Schedules are equivalent in SNR space: ``x_t / alpha = x0 + eps / sqrt_snr``.

    Placing two different schedules at times with equal SNR must therefore
    produce the identical alpha-normalized state, and both must recover the
    same ``x0`` from it.
    """
    reference = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    other = make_other()
    torch.manual_seed(1)
    x0 = torch.randn(3, 4, dtype=torch.float64)
    eps = torch.randn(3, 4, dtype=torch.float64)
    sqrt_snr = torch.tensor([0.5, 1.0, 3.0], dtype=torch.float64)

    t_reference = reference.sqrt_snr_to_t(sqrt_snr)
    t_other = other.sqrt_snr_to_t(sqrt_snr)
    torch.testing.assert_close(other.sqrt_snr(t_other), sqrt_snr, rtol=1e-8, atol=1e-10)

    normalized_reference = (
        reference.forward_process(x0, eps, t_reference[:, None]) / reference.alpha(t_reference)[:, None]
    )
    normalized_other = other.forward_process(x0, eps, t_other[:, None]) / other.alpha(t_other)[:, None]
    torch.testing.assert_close(normalized_reference, normalized_other, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(x0 + eps / sqrt_snr[:, None], normalized_other, rtol=1e-7, atol=1e-9)

    recovered = other.eps_to_x0(other.forward_process(x0, eps, t_other[:, None]), eps, t_other[:, None])
    torch.testing.assert_close(recovered, x0, rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# Time-sampling distributions
# ---------------------------------------------------------------------------


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


_QUARTILE_Z = 0.6744897501960817  # standard-normal 75% quantile
_EDM_LOG_BOUNDS = tuple((math.log(end) + 1.2) / 1.2 for end in (0.002, 80.0))  # EDM's bounds in N(-1.2, 1.2) units


@pytest.mark.parametrize(
    ("make_schedule", "time_dist_type", "sample_kwargs", "expected_quantiles"),
    [
        pytest.param(
            lambda: RFNoiseSchedule(min_t=0.0, max_t=0.999),
            "uniform",
            {"min_t": 0.0, "max_t": 0.999},
            {0.25: 0.25 * 0.999, 0.5: 0.5 * 0.999, 0.75: 0.75 * 0.999},
            id="rf-uniform",
        ),
        pytest.param(
            lambda: RFNoiseSchedule(min_t=0.0, max_t=0.999),
            "logitnormal",
            {"train_p_mean": 0.3, "train_p_std": 0.8, "min_t": 0.0, "max_t": 0.999},
            # Quantiles pass exactly through the monotone sigmoid transform.
            {
                0.25: _sigmoid(0.3 - _QUARTILE_Z * 0.8) * 0.999,
                0.5: _sigmoid(0.3) * 0.999,
                0.75: _sigmoid(0.3 + _QUARTILE_Z * 0.8) * 0.999,
            },
            id="rf-logitnormal",
        ),
        pytest.param(
            lambda: RFNoiseSchedule(min_t=0.0, max_t=0.999),
            "shifted",
            {"shift": 5.0, "min_t": 0.0, "max_t": 0.999},
            # The shift map is monotone, so it transports the uniform quantiles.
            {
                q: float(time_shift(torch.tensor(q * 0.999, dtype=torch.float64), 5.0, max_t=0.999))
                for q in (0.25, 0.5, 0.75)
            },
            id="rf-shifted-uniform",
        ),
        pytest.param(
            lambda: EDMNoiseSchedule(min_t=0.002, max_t=80.0),
            "lognormal",
            {"train_p_mean": -1.2, "train_p_std": 1.2, "min_t": 0.002, "max_t": 80.0},
            # The draw is truncated to the bounds, so the quantiles are the truncated normal's in log space.
            {q: math.exp(stats.truncnorm(*_EDM_LOG_BOUNDS, loc=-1.2, scale=1.2).ppf(q)) for q in (0.25, 0.5, 0.75)},
            id="edm-lognormal",
        ),
    ],
)
def test_sample_t_respects_bounds_and_analytic_quantiles(
    make_schedule, time_dist_type, sample_kwargs, expected_quantiles
):
    schedule = make_schedule()
    torch.manual_seed(0)
    t = schedule.sample_t(200_000, time_dist_type=time_dist_type, **sample_kwargs)

    assert t.shape == (200_000,)
    assert torch.isfinite(t).all()
    assert t.min() >= schedule.min_t and t.max() <= schedule.max_t
    for quantile, expected in expected_quantiles.items():
        actual = torch.quantile(t.double(), quantile).item()
        assert math.isclose(actual, expected, rel_tol=1e-2), f"quantile {quantile}: expected {expected}, got {actual}"


def test_uniform_sample_t_matches_analytic_mean_and_variance():
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    torch.manual_seed(0)
    t = schedule.sample_t(200_000, time_dist_type="uniform", min_t=0.0, max_t=0.999).double()

    assert abs(t.mean().item() - 0.999 / 2) < 2e-3
    assert abs(t.var().item() - 0.999**2 / 12) < 2e-3


@pytest.mark.parametrize(
    ("make_schedule", "time_dist_type", "sample_kwargs", "intervals"),
    [
        pytest.param(
            lambda: RFNoiseSchedule(min_t=0.0, max_t=0.999),
            "uniform",
            {},
            [(0.1, 0.3), (0.624, 0.833)],
            id="rf-uniform",
        ),
        pytest.param(
            lambda: RFNoiseSchedule(min_t=0.0, max_t=0.999),
            "logitnormal",
            {"train_p_mean": 0.3, "train_p_std": 0.8},
            [(0.1, 0.3), (0.833, 0.937)],
            id="rf-logitnormal",
        ),
        pytest.param(
            lambda: RFNoiseSchedule(min_t=0.0, max_t=0.999),
            "shifted",
            {"shift": 5.0},
            # The Wan 4-step buckets.
            [(0.001, 0.624), (0.624, 0.833), (0.833, 0.937), (0.937, 0.999)],
            id="rf-shifted",
        ),
        pytest.param(
            lambda: RFNoiseSchedule(min_t=0.0, max_t=0.999),
            "shifted_logitnormal",
            {"shift": 3.0, "train_p_mean": -0.2, "train_p_std": 1.1},
            [(0.2, 0.5), (0.9, 0.99)],
            id="rf-shifted-logitnormal",
        ),
        pytest.param(
            lambda: EDMNoiseSchedule(min_t=0.002, max_t=80.0),
            "uniform",
            {},
            [(0.05, 0.3), (1.0, 5.0)],
            id="edm-uniform",
        ),
        pytest.param(
            lambda: EDMNoiseSchedule(min_t=0.002, max_t=80.0),
            "lognormal",
            {"train_p_mean": -1.2, "train_p_std": 1.2},
            [(0.05, 0.3), (1.0, 5.0)],
            id="edm-lognormal",
        ),
        pytest.param(
            lambda: EDMNoiseSchedule(min_t=0.002, max_t=80.0),
            "log_t",
            {"train_p_mean": -1.2, "train_p_std": 1.2, "log_t_df": 2.0},
            [(0.05, 0.3), (1.0, 5.0)],
            id="edm-log-t",
        ),
        pytest.param(
            lambda: EDMNoiseSchedule(min_t=0.002, max_t=80.0),
            "polynomial",
            {"rho": 7.0},
            [(0.05, 0.3), (1.0, 5.0)],
            id="edm-polynomial",
        ),
        pytest.param(
            _cosine_alphas_schedule,
            "logitnormal",
            {"train_p_mean": 0.0, "train_p_std": 1.0},
            [(0.2, 0.4), (0.7, 0.95)],
            id="alphas-logitnormal",
        ),
        pytest.param(
            lambda: TrigNoiseSchedule(min_t=0.0, max_t=math.pi / 2),
            "uniform",
            {},
            [(0.1, 0.5), (1.0, 1.5)],
            id="trig-uniform",
        ),
    ],
)
def test_per_sample_bounds_mean_what_scalar_bounds_mean(make_schedule, time_dist_type, sample_kwargs, intervals):
    """A per-sample bound, also next to a scalar one, does for its sample exactly what the same scalar bound
    does for all."""
    schedule = make_schedule()
    n = 20_000
    scalar_draws = []
    for lo, hi in intervals:
        torch.manual_seed(0)
        scalar = schedule.sample_t(n, time_dist_type=time_dist_type, min_t=lo, max_t=hi, **sample_kwargs)
        lo_per_sample, hi_per_sample = (torch.full((n,), bound, dtype=torch.float64) for bound in (lo, hi))
        for min_t, max_t in ((lo_per_sample, hi_per_sample), (lo_per_sample, hi), (lo, hi_per_sample)):
            torch.manual_seed(0)
            per_sample = schedule.sample_t(n, time_dist_type=time_dist_type, min_t=min_t, max_t=max_t, **sample_kwargs)
            torch.testing.assert_close(per_sample, scalar, rtol=0, atol=0)
        assert (scalar >= lo).all() and (scalar <= hi).all()
        scalar_draws.append(scalar.double())

    # Interleaved bounds: every sample follows its own interval's distribution.
    lower = torch.tensor([lo for lo, _ in intervals], dtype=torch.float64).repeat(n // len(intervals))
    upper = torch.tensor([hi for _, hi in intervals], dtype=torch.float64).repeat(n // len(intervals))
    t = schedule.sample_t(len(lower), time_dist_type=time_dist_type, min_t=lower, max_t=upper, **sample_kwargs)
    assert t.shape == lower.shape and t.dtype == schedule.t_precision
    assert (t >= lower).all() and (t <= upper).all()
    for i, ((lo, hi), scalar) in enumerate(zip(intervals, scalar_draws)):
        mixed = t[i :: len(intervals)].double()
        for quantile in (0.1, 0.5, 0.9):
            expected = torch.quantile(scalar, quantile).item()
            assert abs(torch.quantile(mixed, quantile).item() - expected) < 0.03 * (hi - lo)


def test_shifted_bounds_hold_for_the_shifted_time():
    """The shift acts on [0, 1], whose ends it fixes, before the draw is mapped onto [min_t, max_t]."""
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    torch.manual_seed(0)
    t = schedule.sample_t(200_000, time_dist_type="shifted", shift=5.0, min_t=0.624, max_t=0.833).double()

    assert t.min() >= 0.624 and t.max() <= 0.833
    assert t.min() < 0.625, "shifting after bounding would start at shift(0.624) ~ 0.89 and clamp everything"
    for quantile in (0.25, 0.5, 0.75):
        unit = float(time_shift(torch.tensor(quantile, dtype=torch.float64), 5.0))
        expected = 0.624 + (0.833 - 0.624) * unit
        assert abs(torch.quantile(t, quantile).item() - expected) < 1e-3

    torch.manual_seed(0)
    t = schedule.sample_t(200_000, time_dist_type="shifted", shift=5.0, min_t=0.001, max_t=0.999)
    assert 0.001 <= t.min() < 0.002 and t.max() <= 0.999


@pytest.mark.parametrize(
    "make_schedule",
    _SMOOTH_SCHEDULES + [pytest.param(_cosine_alphas_schedule, id="alphas-cosine")],
)
def test_forward_transition_keeps_the_marginals(make_schedule):
    schedule = make_schedule()
    times = _interior_times(schedule, n=5)
    t_start, t = times[:-1], times[1:]
    ones, zeros = torch.ones_like(t_start), torch.zeros_like(t_start)

    scale = schedule.forward_process(ones, zeros, t, t_start=t_start)
    variance = schedule.forward_process(zeros, ones, t, t_start=t_start) ** 2

    torch.testing.assert_close(scale * schedule.alpha(t_start), schedule.alpha(t))
    torch.testing.assert_close(scale**2 * schedule.sigma(t_start) ** 2 + variance, schedule.sigma(t) ** 2)


def test_rf_forward_transition_variance_has_a_closed_form_at_unit_shift():
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    t_start = torch.tensor([0.0, 0.3, 0.624, 0.937], dtype=torch.float64)
    t = torch.tensor([0.2, 0.5, 0.8, 0.999], dtype=torch.float64)

    std = schedule.transition_std(t, t_start)

    torch.testing.assert_close(std**2, (t - t_start) * (t + t_start - 2 * t * t_start) / (1 - t_start) ** 2)
    torch.testing.assert_close(
        schedule.forward_process(torch.zeros_like(t_start), torch.ones_like(t_start), t, t_start=t_start), std
    )


@pytest.mark.parametrize(
    "make_schedule",
    _SMOOTH_SCHEDULES
    + [
        pytest.param(_scaled_linear_alphas_schedule, id="alphas-scaled-linear"),
        pytest.param(_cosine_alphas_schedule, id="alphas-cosine"),
    ],
)
def test_determinant_overrides_are_the_generic_determinant(make_schedule):
    schedule = make_schedule()
    times = _interior_times(schedule, n=5)
    t_start, t_end = times[:-1], times[1:]

    generic = BaseNoiseSchedule._determinant(schedule, t_start, t_end)
    torch.testing.assert_close(schedule._determinant(t_start, t_end), generic)


@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_forward_transition_variance_keeps_its_precision_as_the_times_meet(make_schedule):
    """``v / (t - t_start)`` tends to ``2 sigma (alpha sigma' - alpha' sigma) / alpha``, where direct differences
    cancel."""
    schedule = make_schedule()
    t_start = _interior_times(schedule, n=5)
    t = t_start + 1e-12
    alpha, sigma = schedule.alpha(t_start), schedule.sigma(t_start)
    slope = 2 * sigma * (alpha * schedule.sigma_prime(t_start) - schedule.alpha_prime(t_start) * sigma) / alpha

    torch.testing.assert_close(schedule.transition_std(t, t_start) ** 2 / (t - t_start), slope, rtol=1e-8, atol=0)


@pytest.mark.parametrize(
    "index",
    [500.25, 500.0, 0.5, 1.0, 998.5, 998.0],
    ids=["interior", "interior-entry", "clean-end", "clean-end-entry", "pure-noise-end", "pure-noise-end-entry"],
)
def test_alphas_determinant_keeps_its_precision_as_the_times_meet(index):
    """With ``alpha = cos(theta)`` and ``sigma = sin(theta)``, the determinant is ``sin`` of the angle between two
    times, so splitting them at a third adds the angles. That keeps holding as the times meet, within a segment
    or across an entry, including next to exactly clean or pure-noise entries, where the direct
    difference ``alpha_start sigma_end - alpha_end sigma_start`` cancels; where it does not, the two agree.
    Swapping the times negates it exactly."""
    table = _scaled_linear_alphas_schedule().alphas_cumprod.clone()
    table[0], table[-1] = 1.0, 0.0
    schedule = AlphasNoiseSchedule(alphas_cumprod=table)
    n = schedule.num_train_timesteps
    gap = torch.tensor([0.5, 1e-6, 1e-9, 1e-12], dtype=torch.float64)
    t_start, t_mid, t_end = (index - gap / 2) / n, torch.full_like(gap, index / n), (index + gap / 2) / n

    def cos(t_a, t_b):
        return schedule.alpha(t_a) * schedule.alpha(t_b) + schedule.sigma(t_a) * schedule.sigma(t_b)

    determinant = schedule._determinant
    torch.testing.assert_close(
        determinant(t_start, t_end),
        determinant(t_start, t_mid) * cos(t_mid, t_end) + cos(t_start, t_mid) * determinant(t_mid, t_end),
        rtol=1e-10,
        atol=0,
    )
    generic = BaseNoiseSchedule._determinant(schedule, t_start[:1], t_end[:1])
    torch.testing.assert_close(determinant(t_start[:1], t_end[:1]), generic, rtol=1e-11, atol=0)
    t_far = torch.full_like(gap, 0.25)
    for a, b in ((t_start, t_end), (t_start, t_far)):
        assert torch.equal(determinant(b, a), -determinant(a, b))


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize(
    "values, times, derivatives",
    [
        pytest.param([1.0, 0.0], (0.5, 0.5), (-2.0, 2.0), id="pure-noise-knot"),
        pytest.param(
            [0.9, 0.6, 0.0, 0.0],
            (0.5, 0.5),
            (-4 * math.sqrt(0.6), 4 * math.sqrt(0.6)),
            id="first-zero-of-plateau",
        ),
        pytest.param([0.9, 0.6, 0.0, 0.0], (0.5, 0.75), (-4 * math.sqrt(0.6), 0.0), id="knot-and-plateau"),
        pytest.param([0.9, 0.6, 0.0, 0.0], (0.6, 0.7), (0.0, 0.0), id="within-zero-plateau"),
        pytest.param([0.0, 0.0], (0.0, 0.5), (0.0, 0.0), id="constant-pure-noise"),
    ],
)
def test_alphas_determinant_preserves_time_derivatives_at_pure_noise(device_type, values, times, derivatives):
    """At two pure-noise points d(det) = d(alpha_start) - d(alpha_end). The first zero knot keeps
    alpha's incoming slope -sqrt(p)/width, while later points on a zero plateau have zero slope."""
    schedule = AlphasNoiseSchedule(torch.tensor(values, dtype=torch.float64)).to(device_type)
    s, e = (torch.tensor([t], dtype=torch.float64, device=device_type, requires_grad=True) for t in times)
    ds, de = derivatives
    for start, end, slopes in ((s, e, (ds, de)), (e, s, (-de, -ds))):
        value = schedule._determinant(start, end)
        assert value.eq(0).all()
        reverse = torch.autograd.grad(value.sum(), (start, end))
        for index, slope in enumerate(slopes):
            expected = torch.full_like(start, slope)
            torch.testing.assert_close(reverse[index], expected, rtol=1e-12, atol=0)
            tangents = tuple(torch.full_like(start, float(i == index)) for i in range(2))
            _, forward = torch.func.jvp(schedule._determinant, (start.detach(), end.detach()), tangents)
            torch.testing.assert_close(forward, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("case", ["within-segment", "knot", "pure-noise", "nearly-flat"])
def test_alphas_determinant_matches_an_independent_high_precision_reference(device_type, case):
    from bisect import bisect_right
    from decimal import Decimal, localcontext

    table = get_noise_schedule("ddpm").alphas_cumprod.clone()
    if case == "within-segment":
        start, end = 0.5120335103351034, 0.5120335103351035  # N*t rounds to the same number.
        assert start * 1000 == end * 1000
    elif case == "knot":
        start, end = math.nextafter(0.512, -math.inf), math.nextafter(0.512, math.inf)
    elif case == "pure-noise":
        table[-1] = 0
        start, end = math.nextafter(0.999, -math.inf), 0.999
    else:
        table = 0.5 - torch.arange(5, dtype=torch.float64) * math.ulp(0.5)
        start, end = 0.01, 0.79  # Many segments apart, but a direct determinant still cancels.
    schedule = AlphasNoiseSchedule(table).to(device_type)
    knots = [i / len(table) for i in range(len(table))]

    def coefficients(time):
        i = min(bisect_right(knots, time) - 1, len(table) - 2)
        p, q = map(Decimal.from_float, table[i : i + 2].tolist())
        w = (Decimal.from_float(time) - Decimal.from_float(knots[i])) / (
            Decimal.from_float(knots[i + 1]) - Decimal.from_float(knots[i])
        )
        if q == 0:
            # Into the pure-noise entry alpha itself is linear in time.
            alpha = (1 - w) * p.sqrt()
            return alpha, (1 - alpha * alpha).sqrt()
        probability = (1 - w) * p + w * q
        return probability.sqrt(), (1 - probability).sqrt()

    with localcontext() as ctx:
        ctx.prec = 100
        a, b = coefficients(start)
        c, d = coefficients(end)
        expected = float(a * d - c * b)
    s, e = (torch.tensor(v, dtype=torch.float64, device=device_type) for v in (start, end))
    actual = schedule._determinant(s, e)
    assert expected > 0 and actual > 0
    torch.testing.assert_close(actual, torch.full_like(s, expected), rtol=1e-12, atol=0)
    torch.testing.assert_close(schedule._determinant(e, s), -actual, rtol=1e-12, atol=0)
    torch.testing.assert_close(schedule._determinant(s, s), torch.zeros_like(s), rtol=0, atol=0)


@pytest.mark.parametrize(
    "make_schedule",
    _SMOOTH_SCHEDULES + [pytest.param(_scaled_linear_alphas_schedule, id="alphas-scaled-linear")],
)
def test_forward_transition_from_the_clean_grid_end_is_the_forward_process(make_schedule):
    """Student grids end at 0, below EDM's ``min_t`` and at SD's noisy first discrete time; it marks data."""
    torch.manual_seed(0)
    schedule = make_schedule()
    x, eps = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    t = _interior_times(schedule, n=5)
    t_start = torch.zeros_like(t)

    torch.testing.assert_close(
        schedule.forward_process(x, eps, t, t_start=t_start), schedule.forward_process(x, eps, t)
    )
    torch.testing.assert_close(schedule.transition_std(t, t_start), schedule.sigma(t))


def test_forward_transition_can_start_from_the_schedules_noisy_time_zero():
    torch.manual_seed(0)
    schedule = _scaled_linear_alphas_schedule()
    x, eps = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
    t = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)
    t_start = torch.zeros_like(t)
    alpha_start = schedule.alpha(t_start)
    scale = schedule.alpha(t) / alpha_start
    variance = schedule.sigma(t).square() - scale.square() * schedule.sigma(t_start).square()

    torch.testing.assert_close(schedule.alpha_start(t_start, clean_end=False), alpha_start)
    torch.testing.assert_close(schedule.transition_std(t, t_start, clean_end=False), variance.sqrt())
    torch.testing.assert_close(
        schedule.forward_process(x, eps, t, t_start=t_start, clean_end=False),
        scale[:, None] * x + variance.sqrt()[:, None] * eps,
    )


def test_zero_length_transition_keeps_the_state_also_where_alpha_vanishes():
    """From ``t_start = t`` the transition keeps its state, also at rectified flow's pure-noise end ``t = 1`` where
    ``alpha = 0``; from the clean end, it is still the forward process at ``t``."""
    torch.manual_seed(0)
    x, eps = torch.randn(3, 2, dtype=torch.float64), torch.randn(3, 2, dtype=torch.float64)
    schedule = RFNoiseSchedule(min_t=0.0, max_t=1.0)
    t = torch.tensor([0.3, 0.999, 1.0], dtype=torch.float64)

    assert torch.equal(schedule.forward_process(x, eps, t, t_start=t), x)
    assert torch.equal(schedule.transition_std(t, t), torch.zeros_like(t))

    schedule = _scaled_linear_alphas_schedule()
    zero = torch.zeros(3, dtype=torch.float64)
    torch.testing.assert_close(
        schedule.forward_process(x, eps, zero, t_start=zero), schedule.forward_process(x, eps, zero)
    )
    assert torch.equal(schedule.forward_process(x, eps, zero, t_start=zero, clean_end=False), x)


@pytest.mark.parametrize(
    "make_schedule",
    _SMOOTH_SCHEDULES + [pytest.param(_scaled_linear_alphas_schedule, id="alphas-scaled-linear")],
)
def test_alpha_start_is_alpha_but_one_at_the_clean_grid_end(make_schedule):
    schedule = make_schedule()
    t = _interior_times(schedule, n=5)

    torch.testing.assert_close(schedule.alpha_start(t), schedule.alpha(t))
    torch.testing.assert_close(schedule.alpha_start(torch.zeros_like(t)), torch.ones_like(t))
    with pytest.raises(ValueError, match=rf"t_start must be in \[{schedule.min_t}, "):
        schedule.alpha_start(torch.full_like(t, -0.1))


def test_composite_forward_transition_and_step_run_per_modality():
    torch.manual_seed(0)
    composite = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "shifted_rf", "shift": 12.0, "min_t": 0.0, "max_t": 0.999},
            "audio": {"name": "shifted_rf", "shift": 3.0, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )
    x = {"video": torch.randn(2, 3), "audio": torch.randn(2, 4)}
    eps = randn_like(x)
    t, t_start = torch.tensor([0.5, 0.9], dtype=torch.float64), torch.tensor([0.3, 0.6], dtype=torch.float64)

    transitioned = composite.forward_process(x, eps, t, t_start=t_start, clean_end=False)
    alpha_starts = composite.alpha_start(t_start, clean_end=False)
    transition_stds = composite.transition_std(t, t_start, clean_end=False)
    stepped = composite.step(x, eps, t, t_start, pred_type="flow", clean_end=False)
    inverted = composite.points_to_x0(x, eps, t, t_start, clean_end=False, output_dtype=None)

    for key, schedule in composite.subschedules.items():
        torch.testing.assert_close(
            transitioned[key], schedule.forward_process(x[key], eps[key], t, t_start=t_start, clean_end=False)
        )
        torch.testing.assert_close(alpha_starts[key], schedule.alpha_start(t_start, clean_end=False))
        torch.testing.assert_close(transition_stds[key], schedule.transition_std(t, t_start, clean_end=False))
        torch.testing.assert_close(
            stepped[key], schedule.step(x[key], eps[key], t, t_start, pred_type="flow", clean_end=False)
        )
        torch.testing.assert_close(
            inverted[key], schedule.points_to_x0(x[key], eps[key], t, t_start, clean_end=False, output_dtype=None)
        )


def _exact_predictions(schedule, n=3, flow=True):
    """A state at interior times with its exact predictions, and earlier times to step to."""
    times = _interior_times(schedule, n=n + 1)
    t, t_next = times[1:].flip(0), times[:-1].flip(0)
    x0, eps = torch.randn(n, 5, dtype=torch.float64), torch.randn(n, 5, dtype=torch.float64)
    predictions = {"x0": x0, "eps": eps}
    if flow:  # ``flow`` needs a differentiable clock
        predictions["flow"] = schedule.cond_velocity(x0, eps, t)
    return schedule.forward_process(x0, eps, t), predictions, t, t_next


@pytest.mark.parametrize("pred_type", ["x0", "eps", "flow"])
@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_ode_step_with_an_exact_prediction_lands_on_the_forward_process(make_schedule, pred_type):
    torch.manual_seed(0)
    schedule = make_schedule()
    xt, predictions, t, t_next = _exact_predictions(schedule)

    x_next = schedule.step(xt, predictions[pred_type], t, t_next, pred_type=pred_type)

    expected = schedule.forward_process(predictions["x0"], predictions["eps"], t_next)
    torch.testing.assert_close(x_next, expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    "make_schedule",
    _SMOOTH_SCHEDULES + [pytest.param(_cosine_alphas_schedule, id="alphas-cosine")],
)
def test_ddpm_step_samples_the_posterior_given_the_prediction(make_schedule):
    torch.manual_seed(0)
    schedule = make_schedule()
    xt, predictions, t, t_next = _exact_predictions(schedule, flow=False)
    x0 = predictions["x0"]

    torch.manual_seed(1)
    x_next = schedule.step(xt, x0, t, t_next, sample_type="ddpm")
    torch.manual_seed(1)
    noise = torch.randn_like(x0)

    alpha_t, sigma_t = schedule.alpha(t)[:, None], schedule.sigma(t)[:, None]
    alpha_next, sigma_next = schedule.alpha(t_next)[:, None], schedule.sigma(t_next)[:, None]
    a = alpha_t / alpha_next
    v = sigma_t**2 - a**2 * sigma_next**2
    posterior_mean = a * sigma_next**2 / sigma_t**2 * xt + v / sigma_t**2 * alpha_next * x0
    torch.testing.assert_close(x_next, posterior_mean + sigma_next * v.sqrt() / sigma_t * noise)


@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_sde_step_draws_fresh_noise_around_the_prediction(make_schedule):
    torch.manual_seed(0)
    schedule = make_schedule()
    xt, predictions, t, t_next = _exact_predictions(schedule)
    x0 = predictions["x0"]

    torch.manual_seed(1)
    x_next = schedule.step(xt, x0, t, t_next, sample_type="sde")
    torch.manual_seed(1)
    noise = torch.randn_like(x0)

    torch.testing.assert_close(x_next, schedule.forward_process(x0, noise, t_next))


@pytest.mark.parametrize("sample_type", ["ode", "ddpm", "sde", 0.5])
@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_steps_do_not_depend_on_the_prediction_type(make_schedule, sample_type):
    torch.manual_seed(0)
    schedule = make_schedule()
    xt, predictions, t, t_next = _exact_predictions(schedule)

    steps = {}
    for pred_type, prediction in predictions.items():
        torch.manual_seed(1)
        steps[pred_type] = schedule.step(xt, prediction, t, t_next, pred_type=pred_type, sample_type=sample_type)

    for pred_type in ("eps", "flow"):
        torch.testing.assert_close(steps[pred_type], steps["x0"], rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("sample_type", ["ddpm", "sde", 0.5])
@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_stochastic_steps_draw_their_noise_like_the_state(make_schedule, sample_type):
    """A prediction kept in high precision or in another memory layout draws the same fresh noise as one in the
    state's dtype and layout."""
    torch.manual_seed(0)
    schedule = make_schedule()
    xt, predictions, t, t_next = _exact_predictions(schedule, flow=False)
    xt, x0 = xt.float(), predictions["x0"].float()

    steps = []
    for prediction in (x0, x0.double(), x0.mT.contiguous().mT):
        torch.manual_seed(1)
        steps.append(schedule.step(xt, prediction, t, t_next, sample_type=sample_type))

    assert all(step.dtype == torch.float32 for step in steps)
    for step in steps[1:]:
        torch.testing.assert_close(step, steps[0], rtol=0, atol=0)


def test_eta_interpolates_from_ddim_to_the_fresh_noise_step():
    torch.manual_seed(0)
    schedule = ShiftedRFNoiseSchedule(shift=3.0, min_t=0.0, max_t=0.999)
    xt, predictions, t, t_next = _exact_predictions(schedule)
    x0 = predictions["x0"]

    torch.testing.assert_close(schedule.step(xt, x0, t, t_next, sample_type=0.0), schedule.step(xt, x0, t, t_next))
    torch.manual_seed(1)
    ddpm = schedule.step(xt, x0, t, t_next, sample_type="ddpm")
    torch.manual_seed(1)
    torch.testing.assert_close(schedule.step(xt, x0, t, t_next, sample_type=1.0), ddpm)
    with pytest.raises(ValueError, match="exceeds the fresh-noise step"):
        schedule.step(xt, x0, t, t_next, sample_type=100.0)
    with pytest.raises(ValueError, match="sample_type must be"):
        schedule.step(xt, x0, t, t_next, sample_type="euler")


def test_sample_types_are_sampler_names_or_nonnegative_etas():
    for sample_type in ("ode", "sde", "ddpm", 0, 0.5, 2):
        check_sample_type(sample_type)
    for sample_type in ("euler", "0.5", -0.5, float("nan"), True, None):
        with pytest.raises(ValueError, match="sample_type must be one of"):
            check_sample_type(sample_type)
    with pytest.raises(ValueError, match="must be one of 'ode', 'sde'; got 0.5"):
        check_sample_type(0.5, ("ode", "sde"))
    with pytest.raises(ValueError, match="must be 'ode'; got 'sde'"):
        check_sample_type("sde", ("ode",))


def test_rf_ode_step_on_a_flow_is_the_euler_step_at_unit_shift():
    torch.manual_seed(0)
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    xt, flow = torch.randn(3, 5), torch.randn(3, 5)
    t = torch.tensor([0.999, 0.833, 0.624], dtype=torch.float64)
    t_next = torch.tensor([0.937, 0.624, 0.0], dtype=torch.float64)

    x_next = schedule.step(xt, flow, t, t_next, pred_type="flow")

    dtype = high_precision_dtype(xt, flow)
    expected = (xt.to(dtype) + (t_next - t).to(dtype)[:, None] * flow.to(dtype)).float()
    torch.testing.assert_close(x_next, expected, rtol=0, atol=0)


@pytest.mark.parametrize("pred_type", ["x0", "eps"])
def test_step_without_clean_end_lands_on_the_tables_noisy_time_zero(pred_type):
    """``clean_end=False`` renoises x0 to the schedule's own time 0 (``alpha(0) x0 + sigma(0) eps``), as the DiT
    chain's step into timestep 0 and SD1.5's release sampler do; RF's time 0 is clean, so both agree there."""
    torch.manual_seed(0)
    schedule = _scaled_linear_alphas_schedule()
    xt, predictions, t, _ = _exact_predictions(schedule, flow=False)
    t_next = torch.zeros((), dtype=torch.float64)

    clean = schedule.step(xt, predictions[pred_type], t, t_next, pred_type=pred_type)
    noisy = schedule.step(xt, predictions[pred_type], t, t_next, pred_type=pred_type, clean_end=False)

    torch.testing.assert_close(clean, predictions["x0"], rtol=1e-7, atol=1e-7)
    expected = schedule.forward_process(predictions["x0"], predictions["eps"], t_next.expand(xt.shape[0]))
    torch.testing.assert_close(noisy, expected, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(noisy, clean)

    rf = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    xt, predictions, t, _ = _exact_predictions(rf)
    for pred in ("x0", "flow"):
        torch.testing.assert_close(
            rf.step(xt, predictions[pred], t, t_next, pred_type=pred, clean_end=False),
            rf.step(xt, predictions[pred], t, t_next, pred_type=pred),
        )


def test_ddpm_step_without_clean_end_uses_the_schedules_time_zero_transition():
    torch.manual_seed(0)
    schedule = _scaled_linear_alphas_schedule()
    xt, predictions, t, _ = _exact_predictions(schedule, flow=False)
    x0 = predictions["x0"]
    t_next = torch.zeros_like(t)

    torch.manual_seed(1)
    actual = schedule.step(xt, x0, t, t_next, sample_type="ddpm", clean_end=False)
    torch.manual_seed(1)
    noise = torch.randn_like(x0)

    alpha_t, sigma_t = schedule.alpha(t)[:, None], schedule.sigma(t)[:, None]
    alpha_next, sigma_next = schedule.alpha(t_next)[:, None], schedule.sigma(t_next)[:, None]
    a = alpha_t / alpha_next
    variance = sigma_t**2 - a**2 * sigma_next**2
    posterior_mean = a * sigma_next**2 / sigma_t**2 * xt + variance / sigma_t**2 * alpha_next * x0
    torch.testing.assert_close(actual, posterior_mean + sigma_next * variance.sqrt() / sigma_t * noise)


def test_rf_flow_step_checks_the_sample_type_and_times_like_the_base_step():
    """The flow shortcut rejects bool sample types and, below ``min_t``, every ``t_next`` but the clean end 0."""
    torch.manual_seed(0)
    schedule = RFNoiseSchedule(min_t=0.01, max_t=0.999)
    xt, flow = torch.randn(2, 3), torch.randn(2, 3)
    t = torch.full((2,), 0.5, dtype=torch.float64)

    with pytest.raises(ValueError, match="sample_type must be"):
        schedule.step(xt, flow, t, torch.full((2,), 0.25, dtype=torch.float64), pred_type="flow", sample_type=False)
    clean = schedule.step(xt, flow, t, torch.zeros(2, dtype=torch.float64), pred_type="flow")
    torch.testing.assert_close(clean, (xt.double() - 0.5 * flow.double()).float(), rtol=0, atol=0)
    with pytest.raises(ValueError, match="t_next must be in"):
        schedule.step(xt, flow, t, torch.full((2,), 0.005, dtype=torch.float64), pred_type="flow")


@pytest.mark.parametrize("sample_type", ["ode", "ddpm", "sde", 0.5])
@pytest.mark.parametrize("pred_type", ["x0", "eps", "flow"])
@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_step_to_the_clean_grid_end_lands_on_x0(make_schedule, pred_type, sample_type):
    """Student grids end at 0, below EDM's ``min_t``."""
    torch.manual_seed(0)
    schedule = make_schedule()
    xt, predictions, t, _ = _exact_predictions(schedule)
    t_next = torch.zeros((), dtype=torch.float64)

    x_next = schedule.step(xt, predictions[pred_type], t, t_next, pred_type=pred_type, sample_type=sample_type)

    torch.testing.assert_close(x_next, predictions["x0"], rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("sample_type", ["ode", "ddpm", "sde", 0.5])
@pytest.mark.parametrize("pred_type", ["x0", "eps", "v"])
def test_step_to_the_clean_grid_end_lands_on_x0_where_the_schedules_time_zero_is_noisy(pred_type, sample_type):
    """SD's first discrete time 0 carries noise, but a step to the grid's 0 returns x0 instead of renoising to it."""
    torch.manual_seed(0)
    schedule = _scaled_linear_alphas_schedule()
    assert schedule.sigma(torch.zeros(1, dtype=torch.float64)) > 0
    xt, predictions, t, _ = _exact_predictions(schedule, flow=False)
    predictions["v"] = schedule.convert_model_output(xt, predictions["x0"], t, src_pred_type="x0", target_pred_type="v")
    t_next = torch.zeros((), dtype=torch.float64)

    x_next = schedule.step(xt, predictions[pred_type], t, t_next, pred_type=pred_type, sample_type=sample_type)

    torch.testing.assert_close(x_next, predictions["x0"], rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("sample_type", ["ddpm", 0.5])
def test_stochastic_step_to_the_clean_grid_end_lands_on_x0_where_alpha_vanishes(sample_type):
    """A clean-end sample's zero-length transition is ``0 / 0`` at ``alpha(t) = 0``; it draws no noise."""
    torch.manual_seed(0)
    schedule = RFNoiseSchedule(min_t=0.0, max_t=1.0)
    xt, x0 = torch.randn(2, 5, dtype=torch.float64), torch.randn(2, 5, dtype=torch.float64)
    t, t_next = torch.ones((), dtype=torch.float64), torch.tensor([0.0, 0.5], dtype=torch.float64)

    x_next = schedule.step(xt, x0, t, t_next, sample_type=sample_type)

    torch.testing.assert_close(x_next[0], x0[0])
    assert torch.isfinite(x_next).all()


@pytest.mark.parametrize("pred_type", ["x0", "flow"])
def test_x0_and_flow_steps_from_pure_noise_take_x_t_as_the_noise(pred_type):
    """At ``t = max_t = 1``, where ``alpha(t) = 0``, x_t is all noise and the prediction only sets x0."""
    torch.manual_seed(0)
    schedule = ShiftedRFNoiseSchedule(shift=3.0, min_t=0.0, max_t=1.0)
    x0, eps = torch.randn(3, 5, dtype=torch.float64), torch.randn(3, 5, dtype=torch.float64)
    t, t_next = torch.ones(3, dtype=torch.float64), torch.tensor([0.9, 0.5, 0.1], dtype=torch.float64)
    xt = schedule.forward_process(x0, eps, t)
    prediction = x0 if pred_type == "x0" else schedule.cond_velocity(x0, eps, t)

    x_next = schedule.step(xt, prediction, t, t_next, pred_type=pred_type)
    transition_std = schedule.forward_process(torch.zeros_like(t), torch.ones_like(t), t, t_start=t_next)

    torch.testing.assert_close(x_next, schedule.forward_process(x0, eps, t_next))
    # The DDPM posterior draws all of its noise fresh here, so "ddpm" is also the largest eta.
    torch.testing.assert_close(transition_std, schedule.sigma(t))
    assert torch.isfinite(schedule.step(xt, prediction, t, t_next, pred_type=pred_type, sample_type="ddpm")).all()
    with pytest.raises(ValueError, match="exceeds the fresh-noise step"):
        schedule.step(xt, prediction, t, t_next, pred_type=pred_type, sample_type=1.5)


@pytest.mark.parametrize(
    ("make_schedule", "t_pure_noise"),
    [
        pytest.param(lambda: ShiftedRFNoiseSchedule(shift=3.0, min_t=0.0, max_t=1.0), 1.0, id="rf-max1"),
        pytest.param(lambda: TrigNoiseSchedule(min_t=0.0, max_t=math.pi / 2), math.pi / 2, id="trig"),
        pytest.param(_cosine_alphas_schedule, 0.999, id="alphas-cosine"),
    ],
)
def test_eps_predictions_raise_where_they_determine_no_x0(make_schedule, t_pure_noise):
    """Where ``alpha(t)`` is below ``clamp_min``, the conversion would divide by the clamp instead."""
    torch.manual_seed(0)
    schedule = make_schedule()
    xt, eps = torch.randn(2, 5, dtype=torch.float64), torch.randn(2, 5, dtype=torch.float64)
    t = torch.tensor([0.5, t_pure_noise], dtype=torch.float64)
    t_next = torch.full((2,), 0.25, dtype=torch.float64)

    for sample_type in ("ode", "ddpm"):
        with pytest.raises(ValueError, match="eps prediction does not determine x0"):
            schedule.step(xt, eps, t, t_next, pred_type="eps", sample_type=sample_type)
    with pytest.raises(ValueError, match="eps prediction does not determine x0"):
        schedule.eps_to_x0(xt, eps, t)
    with pytest.raises(ValueError, match="eps prediction does not determine x0"):
        schedule.convert_model_output(xt, eps, t, src_pred_type="eps", target_pred_type="x0")


@pytest.mark.parametrize("pred_type", ["x0", "eps", "flow"])
@pytest.mark.parametrize("name", ["edm", "rf"])
def test_step_takes_timesteps_in_the_schedule_precision(name, pred_type):
    """A float32-time schedule steps its own grid exactly as a float64 one steps the same times."""
    torch.manual_seed(0)
    schedule = get_noise_schedule(name, t_precision="float32")
    reference = get_noise_schedule(name)
    grid = schedule.grid(4)
    assert grid.dtype == torch.float32
    xt, prediction = torch.randn(2, 5), torch.randn(2, 5)

    for t, t_next in zip(grid[:-1], grid[1:]):
        x_next = schedule.step(xt, prediction, t, t_next, pred_type=pred_type)
        expected = reference.step(xt, prediction, t.double(), t_next.double(), pred_type=pred_type)
        assert torch.equal(x_next, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "make_schedule",
    _SMOOTH_SCHEDULES + [pytest.param(_scaled_linear_alphas_schedule, id="alphas-scaled-linear")],
)
def test_coefficient_getters_compute_lower_precision_times_in_high_precision(make_schedule, dtype):
    """A float32 or bfloat16 time is computed with in ``HIGH_PRECISION``, so each getter returns the coefficient of
    the same time in float64, rounded to the time's dtype."""
    schedule = make_schedule()
    t = _interior_times(schedule).to(dtype)
    getters = ["alpha", "sigma", "alpha_start", "rescale_t"]
    if not isinstance(schedule, AlphasNoiseSchedule):
        getters += ["alpha_prime", "sigma_prime"]

    for name in getters:
        getter = getattr(schedule, name)
        assert torch.equal(getter(t), getter(t.double()).to(dtype)), name


@pytest.mark.parametrize(
    "make_schedule",
    _SMOOTH_SCHEDULES
    + [
        pytest.param(_scaled_linear_alphas_schedule, id="alphas-scaled-linear"),
        pytest.param(lambda: get_noise_schedule("cogvideox", snr_shift_scale=1), id="alphas-zero-terminal-snr"),
    ],
)
def test_alpha_sigma_is_the_pair_of_coefficient_getters(make_schedule):
    """One range check and, on table schedules, one table lookup give exactly both getters' coefficients, in the
    time's dtype."""
    schedule = make_schedule()
    t = torch.cat([_interior_times(schedule), torch.tensor([schedule.min_t, schedule.max_t], dtype=torch.float64)])
    for times in (t, t.float(), t.bfloat16()):
        alpha, sigma = schedule.alpha_sigma(times)
        assert torch.equal(alpha, schedule.alpha(times)) and torch.equal(sigma, schedule.sigma(times))
    with pytest.raises(ValueError, match="t must be in"):
        schedule.alpha_sigma(t + schedule.max_t)


@pytest.mark.parametrize(
    "make_schedule",
    _SMOOTH_SCHEDULES + [pytest.param(_scaled_linear_alphas_schedule, id="alphas-scaled-linear")],
)
def test_determinant_checks_its_times_and_computes_in_high_precision(make_schedule):
    """``determinant`` is the unchecked ``_determinant`` of float64 times, rounded to the dtype of lower-precision
    ones, and rejects times outside the schedule range."""
    schedule = make_schedule()
    times = _interior_times(schedule, n=5)
    t_start, t_end = times[:-1], times[1:]

    assert torch.equal(schedule.determinant(t_start, t_end), schedule._determinant(t_start, t_end))
    for dtype in (torch.float32, torch.bfloat16):
        low_start, low_end = t_start.to(dtype), t_end.to(dtype)
        expected = schedule._determinant(low_start.double(), low_end.double()).to(dtype)
        assert torch.equal(schedule.determinant(low_start, low_end), expected)
    with pytest.raises(ValueError, match="t_end must be in"):
        schedule.determinant(t_start, t_end + schedule.max_t)


def test_two_time_getters_round_to_the_dtype_of_their_first_time():
    """Mixed time dtypes compute in high precision; only the result takes the first time's dtype."""
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    t = torch.tensor([0.5, 0.75], dtype=torch.float64)
    t_start = torch.tensor([0.25, 0.5], dtype=torch.float64)

    for low, high in ((torch.float32, torch.float64), (torch.bfloat16, torch.float32)):
        for t_dtype, t_start_dtype in ((low, high), (high, low)):
            times, starts = t.to(t_dtype), t_start.to(t_start_dtype)
            std = schedule.transition_std(times, starts)
            determinant = schedule.determinant(starts, times)
            assert std.dtype == t_dtype and determinant.dtype == t_start_dtype
            assert torch.equal(std, schedule._transition_std(times.double(), starts.double()).to(t_dtype))
            assert torch.equal(determinant, schedule._determinant(starts.double(), times.double()).to(t_start_dtype))


@pytest.mark.parametrize(
    "make_schedule",
    [
        pytest.param(lambda: EDMNoiseSchedule(min_t=0.002, max_t=5.0), id="edm"),
        pytest.param(lambda: ShiftedRFNoiseSchedule(shift=6.0, min_t=0.125, max_t=0.875), id="rf-shift6"),
        pytest.param(_scaled_linear_alphas_schedule, id="alphas-scaled-linear"),
    ],
)
def test_determinant_takes_data_at_the_clean_end(make_schedule):
    """A time 0 is the clean end, ``alpha = 1`` and ``sigma = 0``, also below ``min_t``: the determinant from or to it
    is the other time's sigma. Without ``clean_end`` it is the schedule's own time 0, noisy only for the Alphas
    table, and below ``min_t`` it is rejected."""
    schedule = make_schedule()
    t = _interior_times(schedule, n=3)
    zero = torch.zeros_like(t)

    torch.testing.assert_close(schedule.determinant(t, zero), -schedule.sigma(t), rtol=1e-14, atol=0)
    torch.testing.assert_close(schedule.determinant(zero, t), schedule.sigma(t), rtol=1e-14, atol=0)
    assert schedule.determinant(zero, zero).eq(0).all()
    if schedule.min_t > 0:
        with pytest.raises(ValueError, match="t_end must be in"):
            schedule.determinant(t, zero, clean_end=False)
    else:
        own = schedule.determinant(t, zero, clean_end=False)
        assert torch.equal(own, schedule._determinant(t, zero))
        assert not torch.allclose(own, -schedule.sigma(t))


@pytest.mark.parametrize("shift", [1.0, 0.25, 6.0])
def test_rf_sigma_to_t_inverts_sigma_in_high_precision(shift):
    schedule = ShiftedRFNoiseSchedule(shift=shift, min_t=0.0, max_t=0.999)
    t = _interior_times(schedule)
    sigma = schedule.sigma(t)

    torch.testing.assert_close(schedule.sigma_to_t(sigma), t, rtol=1e-14, atol=0)
    assert torch.equal(schedule.sigma_to_t(sigma.float()), schedule.sigma_to_t(sigma.float().double()).float())


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("max_t", [0.999, 1.0])
@pytest.mark.parametrize("shift", [1.0, 3.16, 7.0, 12.0])
def test_rf_inverses_return_valid_times_next_to_max_t(device_type, dtype, max_t, shift):
    """Next to max_t the inverse warp multiplies rounding by the shift. Both inverses of the top 200 valid times still
    return valid times no later than max_t, and sqrt_snr_to_t keeps the slope of the exact inverse there."""
    schedule = ShiftedRFNoiseSchedule(shift=shift, min_t=0.0, max_t=max_t)
    up, down = torch.tensor(math.inf, dtype=dtype), torch.tensor(-math.inf, dtype=dtype)
    times = [torch.nextafter(torch.tensor(max_t, dtype=dtype), up)]  # the one ulp past max_t that is_t_valid accepts
    for _ in range(199):
        times.append(torch.nextafter(times[-1], down))
    t = torch.stack(times).to(device_type)
    top = BaseNoiseSchedule.safe_clamp(torch.tensor(max_t, dtype=dtype), max=max_t).item()

    for inverse, value in ((schedule.sigma_to_t, schedule.sigma(t)), (schedule.sqrt_snr_to_t, schedule.sqrt_snr(t))):
        back = inverse(value)
        assert back.dtype == dtype and bool(schedule.is_t_valid(back)) and (back.double() <= max_t).all()
        torch.testing.assert_close(back, t.clamp(max=top), rtol=(shift + 1) * torch.finfo(dtype).eps, atol=0)

    r = schedule.sqrt_snr(t)
    # dt/dr = -sigma^2 / sigma'(t) for sigma = 1 / (1 + r), at the exact time of r.
    sigma = 1 / (1 + r.double())
    exact = time_shift(sigma, 1 / shift, max_t=max_t)
    expected = (-(sigma**2) * (shift * exact + (max_t - exact)) ** 2 / (shift * max_t**2)).to(dtype)
    tolerance = 1e-10 if dtype == torch.float64 else 1e-6
    gradient = torch.autograd.grad(schedule.sqrt_snr_to_t(r.requires_grad_()).sum(), r)[0]
    torch.testing.assert_close(gradient, expected, rtol=tolerance, atol=0)
    _, tangent = torch.func.jvp(schedule.sqrt_snr_to_t, (r.detach(),), (torch.ones_like(r),))
    torch.testing.assert_close(tangent, expected, rtol=tolerance, atol=0)


def test_rf_inverses_map_the_ends_of_the_schedule_back_exactly():
    schedule = ShiftedRFNoiseSchedule(shift=7.0, min_t=0.0, max_t=0.999)
    top = torch.tensor([0.999], dtype=torch.float64)
    assert schedule.sigma_to_t(schedule.sigma(top)).item() == 0.999
    assert schedule.sigma_to_t(torch.zeros(1, dtype=torch.float64)).item() == 0
    assert schedule.sqrt_snr_to_t(torch.tensor([math.inf], dtype=torch.float64)).item() == 0


@pytest.mark.parametrize(
    ("min_t", "max_t", "shift"),
    [(0.02, 0.9729831748260129, 3.16), (0.02, 0.9398823542224268, 3.16), (0.0, 0.999, 7.0), (1e-4, 0.995, 24.0)],
)
def test_rf_inverses_return_times_in_range_for_valid_times_next_to_each_end(min_t, max_t, shift):
    """For the 300 valid times next to each end, the inverses give times in range for the schedule's own square-root
    SNRs, those of its noise levels, sCM's TrigFlow round trip of them, and the noise levels themselves."""
    schedule = ShiftedRFNoiseSchedule(shift=shift, min_t=min_t, max_t=max_t)
    times = []
    bottom = math.nextafter(min_t, -math.inf) if min_t > 0 else 0.0
    for start, toward in ((math.nextafter(max_t, math.inf), -math.inf), (bottom, math.inf)):
        times.append(start)
        for _ in range(299):
            times.append(math.nextafter(times[-1], toward))
    t = torch.tensor(times, dtype=torch.float64)
    alpha, sigma = schedule.alpha_sigma(t)
    trigflow = TrigNoiseSchedule().sqrt_snr(torch.atan2(sigma, alpha * 0.5)) / 0.5
    for sqrt_snr in (schedule.sqrt_snr(t), (1 - sigma) / sigma, trigflow):
        assert bool(schedule.is_t_valid(schedule.sqrt_snr_to_t(sqrt_snr)))
    assert bool(schedule.is_t_valid(schedule.sigma_to_t(sigma)))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize(
    "make_schedule",
    [
        pytest.param(EDMNoiseSchedule, id="edm"),
        pytest.param(lambda: get_noise_schedule("ddpm"), id="ddpm"),
        pytest.param(lambda: get_noise_schedule("ddpm", min_t=0.1, max_t=0.9), id="ddpm-narrowed"),
        pytest.param(RFNoiseSchedule, id="rf"),
        pytest.param(lambda: ShiftedRFNoiseSchedule(shift=5.0), id="shifted-rf"),
        pytest.param(TrigNoiseSchedule, id="trig"),
    ],
)
def test_every_sqrt_snr_to_t_returns_the_ends_inside_the_range_in_their_precision(make_schedule, dtype):
    """Every inverse clamps after its cast, so the ends come back inside the range, also where their dtype rounds them
    past it (float32 0.999 and pi/2)."""
    schedule = make_schedule()
    back = schedule.sqrt_snr_to_t(schedule.sqrt_snr(torch.tensor([schedule.min_t, schedule.max_t], dtype=dtype)))
    assert back.dtype == dtype
    assert schedule.min_t <= back[0].item() <= back[1].item() <= schedule.max_t


def test_trig_rejects_max_t_past_pi_over_2():
    with pytest.raises(ValueError, match="at most pi/2"):
        TrigNoiseSchedule(max_t=2.0)


def test_composite_determinant_runs_per_modality():
    composite = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "shifted_rf", "shift": 12.0, "min_t": 0.0, "max_t": 0.999},
            "audio": {"name": "shifted_rf", "shift": 3.0, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )
    t_start = torch.tensor([0.3, 0.6], dtype=torch.float64)
    t_end = torch.tensor([0.5, 0.0], dtype=torch.float64)

    determinant = composite.determinant(t_start, t_end)
    for key, schedule in composite.subschedules.items():
        assert torch.equal(determinant[key], schedule.determinant(t_start, t_end))


def test_composite_alpha_sigma_runs_per_modality():
    composite = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "shifted_rf", "shift": 12.0, "min_t": 0.0, "max_t": 0.999},
            "audio": {"name": "shifted_rf", "shift": 3.0, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )
    t = torch.tensor([0.3, 0.6], dtype=torch.float64)

    alpha, sigma = composite.alpha_sigma(t)
    assert set(alpha) == set(sigma) == set(composite.subschedules)
    for key, schedule in composite.subschedules.items():
        assert torch.equal(alpha[key], schedule.alpha(t)) and torch.equal(sigma[key], schedule.sigma(t))


def test_composite_cond_v_runs_per_modality():
    composite = CompositeNoiseSchedule(
        schedules={"video": {"name": "ddpm"}, "audio": {"name": "cogvideox", "snr_shift_scale": 3.0}},
        primary_key="video",
    )
    x = {"video": torch.randn(2, 3), "audio": torch.randn(2, 4)}
    eps = randn_like(x)
    t = torch.tensor([0.3, 0.6], dtype=torch.float64)

    v = composite.cond_v(x, eps, t)
    assert set(v) == set(composite.subschedules)
    for key, schedule in composite.subschedules.items():
        assert torch.equal(v[key], schedule.cond_v(x[key], eps[key], t))


@pytest.mark.parametrize("sample_type", ["ode", "ddpm", "sde", 0.5])
@pytest.mark.parametrize("pred_type", ["x0", "eps", "flow"])
@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_step_checks_each_time_once(monkeypatch, make_schedule, pred_type, sample_type):
    torch.manual_seed(0)
    schedule = make_schedule()
    xt, predictions, t, t_next = _exact_predictions(schedule)
    original_assert_t_valid = schedule._assert_t_valid
    names = []

    def record_assert_t_valid(actual_t, name="t", **kwargs):
        names.append(name)
        original_assert_t_valid(actual_t, name, **kwargs)

    monkeypatch.setattr(schedule, "_assert_t_valid", record_assert_t_valid)
    schedule.step(xt, predictions[pred_type], t, t_next, pred_type=pred_type, sample_type=sample_type)

    assert names == ["t", "t_next"]


def _step_without_synchronizing(schedule, pred_type):
    """Take every kind of step on the device, with synchronizing CUDA calls raising."""
    t_next, t = _interior_times(schedule, n=2).cuda()
    xt = torch.randn(3, 5, device="cuda")
    prediction = torch.randn_like(xt)
    torch.cuda.synchronize()

    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for sample_type in ("ode", "ddpm", "sde"):
            schedule.step(xt, prediction, t, t_next, pred_type=pred_type, sample_type=sample_type)
        schedule.step(xt, prediction, t, torch.zeros_like(t_next), pred_type=pred_type)
    finally:
        torch.cuda.set_sync_debug_mode(previous)


@RunIf(min_gpus=1)
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature")
@pytest.mark.parametrize("pred_type", ["x0", "eps", "flow"])
@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_step_does_not_synchronize_the_device(make_schedule, pred_type):
    _step_without_synchronizing(make_schedule(), pred_type)


@RunIf(min_gpus=1)
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature")
@pytest.mark.parametrize("pred_type", ["x0", "eps", "v"])
def test_alphas_step_does_not_synchronize_once_the_table_is_on_the_device(pred_type):
    """Preparation places the registered table once, so sampling steps need no host-to-device copy."""
    schedule = _cosine_alphas_schedule()
    schedule.prepare_module(device="cuda", precision_model=torch.bfloat16, precision_opt=torch.float32)

    _step_without_synchronizing(schedule, pred_type)

    assert schedule.device.type == "cuda"


def test_forward_transition_needs_increasing_time():
    x = torch.zeros(2, dtype=torch.float64)
    t = torch.tensor([0.5, 0.6], dtype=torch.float64)

    for schedule in (EDMNoiseSchedule(min_t=0.002, max_t=80.0), RFNoiseSchedule(min_t=0.0, max_t=0.999)):
        with pytest.raises(ValueError, match="t >= t_start"):
            schedule.forward_process(x, x, t, t_start=t + 0.1)
        with pytest.raises(ValueError, match="t >= t_start"):
            schedule.transition_std(t, t + 0.1)


def test_transition_std_checks_its_time_range():
    """``t_start`` may be the clean end ``0`` of a student grid below ``min_t``, but no other time below it; ``t``
    may not."""
    schedule = EDMNoiseSchedule(min_t=0.002, max_t=80.0)
    t = torch.tensor([0.5, 0.6], dtype=torch.float64)

    torch.testing.assert_close(schedule.transition_std(t, torch.zeros_like(t)), schedule.sigma(t))
    for t_start in (torch.full_like(t, -0.1), torch.full_like(t, 0.001)):
        with pytest.raises(ValueError, match=r"t_start must be in \[0.002, 80.0\]"):
            schedule.transition_std(t, t_start)
        with pytest.raises(ValueError, match=r"t_start must be in \[0.002, 80.0\]"):
            schedule.forward_process(t, t, t, t_start=t_start)
        with pytest.raises(ValueError, match=r"t_start must be in \[0.002, 80.0\]"):
            schedule.alpha_start(t_start)
    with pytest.raises(ValueError, match=r"t must be in \[0.002, 80.0\]"):
        schedule.transition_std(t + 80.0, t)


_FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def _representable_steps_past(bound: float, dtype: torch.dtype, toward: float, steps: int) -> torch.Tensor:
    """*bound* rounded to *dtype*, then moved *steps* representable values of *dtype* toward *toward*."""
    t = torch.tensor(bound, dtype=dtype)
    for _ in range(steps):
        t = torch.nextafter(t, torch.tensor(toward, dtype=dtype))
    return t


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
@pytest.mark.parametrize("batched", [False, True], ids=["0-dim", "batched"])
@pytest.mark.parametrize(("min_t", "max_t"), [(0.002, 80.0), (0.1, 0.7), (0.1, 0.999)])
def test_is_t_valid_allows_one_ulp_past_each_bound_in_the_time_precision(dtype, batched, min_t, max_t):
    """A bound rounded to t's precision passes, as does the next value outward, but not the one after that."""
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)

    for bound, toward in ((min_t, -math.inf), (max_t, math.inf)):
        for steps, valid in ((0, True), (1, True), (2, False)):
            t = _representable_steps_past(bound, dtype, toward, steps)
            t = t.expand(3) if batched else t
            assert bool(schedule.is_t_valid(t, min_t=min_t, max_t=max_t)) is valid, (bound, steps)


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
@pytest.mark.parametrize("max_t", [0.7, 0.999])
def test_explicit_student_grid_may_start_one_ulp_past_max_t_in_its_precision(dtype, max_t):
    """An explicit grid is validated in its own precision, before it is clamped into the schedule range."""
    for steps, valid in ((0, True), (1, True), (2, False)):
        first = _representable_steps_past(max_t, dtype, math.inf, steps)
        timesteps = torch.stack([first, first / 2, torch.zeros_like(first)])
        schedule = get_noise_schedule(
            "rf",
            max_t=max_t,
            student_grid_cfg=StudentGridConfig(steps=2, grid_type="explicit", grid_kwargs={"timesteps": timesteps}),
        )
        if valid:
            assert schedule.student_grid(2).shape == (3,)
        else:
            with pytest.raises(ValueError, match="must lie in"):
                schedule.student_grid(2)


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_grid_times_before_the_clean_end_must_reach_min_t(dtype):
    """Every grid time before the clean end 0 is a network evaluation, so it must reach ``min_t``, to one ulp of the
    grid's precision, which the clamp after validation removes; the clean end itself lies below it."""
    for steps, valid in ((0, True), (1, True), (2, False)):
        last = _representable_steps_past(0.1, dtype, -math.inf, steps)
        timesteps = torch.stack([torch.full_like(last, 0.9), last, torch.zeros_like(last)])
        schedule = get_noise_schedule(
            "rf",
            min_t=0.1,
            student_grid_cfg=StudentGridConfig(steps=2, grid_type="explicit", grid_kwargs={"timesteps": timesteps}),
        )
        if valid:
            grid = schedule.student_grid(2)
            assert grid[-1] == 0 and grid[-2].item() >= 0.1
        else:
            with pytest.raises(ValueError, match=r"must lie in \[0.1, 0.999\]"):
                schedule.student_grid(2)
    # The uniform native grid's last step, 0.999 / 10, falls below min_t. It is checked in the time precision, before
    # the clamp, so a bfloat16 grid of it raises too, although bfloat16 rounds the time up past min_t.
    schedule = get_noise_schedule("rf", min_t=0.1)
    assert schedule.grid(9)[-2] > 0.1
    for dtype in (None, torch.bfloat16):
        with pytest.raises(ValueError, match="must lie in"):
            schedule.grid(10, dtype=dtype)
    # A shift below one moves the last time, 0.111, to 0.059, past rounding, so the clamp leaves it to raise.
    shifted = get_noise_schedule(
        "rf", min_t=0.1, student_grid_cfg=StudentGridConfig(steps=9, grid_kwargs={"shift": 0.5})
    )
    with pytest.raises(ValueError, match="must lie in"):
        shifted.student_grid()


def test_lower_precision_grids_reach_min_t_rounded_inward():
    """Grids are clamped at ``min_t`` after their cast as well: RF's last time over 3 steps, 0.999 / 3, is a
    ``min_t`` of 0.333, which bfloat16 rounds down, so the clamp moves it to the next bfloat16 time."""
    bottom = BaseNoiseSchedule.safe_clamp(torch.tensor(0.333, dtype=torch.bfloat16), min=0.333)
    grid = RFNoiseSchedule(min_t=0.333, t_precision="bfloat16").grid(3)
    assert bottom.item() > 0.333 and grid[-2] == bottom and grid[-1] == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_lower_precision_grids_start_at_max_t_rounded_inward(dtype):
    """Grids are clamped after their cast, which may round max_t up: bfloat16 0.999 is 1, pure noise for RF."""
    top = BaseNoiseSchedule.safe_clamp(torch.tensor(0.999, dtype=dtype), max=0.999)
    rf = get_noise_schedule("rf", student_grid_cfg=StudentGridConfig(steps=4, grid_kwargs={"shift": 5}))
    edm = EDMNoiseSchedule(max_t=0.999)
    for grid in (rf.grid(4, dtype=dtype), rf.student_grid(dtype=dtype), edm.grid(4, dtype=dtype)):
        assert grid.dtype == dtype
        assert grid[0] == top and grid[-1] == 0


def test_time_shift_over_an_interval_fixes_both_ends():
    lower = torch.tensor([0.0, 0.624, 0.937], dtype=torch.float64)
    upper = torch.tensor([0.624, 0.833, 0.999], dtype=torch.float64)
    t = torch.stack(
        [torch.linspace(lo, hi, 9, dtype=torch.float64) for lo, hi in zip(lower.tolist(), upper.tolist())], 1
    )

    shifted = time_shift(t, 5.0, max_t=upper.expand_as(t), min_t=lower.expand_as(t))

    torch.testing.assert_close(shifted[0], lower)
    torch.testing.assert_close(shifted[-1], upper)
    assert (shifted[1:] > shifted[:-1]).all()
    torch.testing.assert_close(shifted, lower + (upper - lower) * time_shift((t - lower) / (upper - lower), 5.0))


def test_time_shift_below_one_is_the_inverse_bias():
    values = np.array([0.0, 0.25, 0.75, 1.0])
    np.testing.assert_allclose(time_shift(values, 0.5), values * 0.5 / (values * -0.5 + 1.0))
    with pytest.raises(ValueError, match="positive"):
        time_shift(values, 0.0)


@pytest.mark.parametrize("shift", [0.25, 1 / 3, 3.0, 3.16, 7.0, 12.0])
@pytest.mark.parametrize(("min_t", "max_t"), [(0.0, 0.999), (0.0, 1.0), (0.002, 80.0), (0.624, 0.833)])
def test_time_shift_keeps_its_ends_exactly_and_never_leaves_the_interval(shift, min_t, max_t):
    """Both ends map to themselves bit for bit, no time leaves the interval, and each is within 3 ulps of the exact
    warp; checked on the 200 floats next to each end and a uniform grid."""
    top, bottom = [max_t], [min_t]
    for _ in range(200):
        top.append(math.nextafter(top[-1], -math.inf))
        bottom.append(math.nextafter(bottom[-1], math.inf))
    uniform = [min(max(min_t + (max_t - min_t) * k / 1000, min_t), max_t) for k in range(1001)]
    t = torch.tensor(top + bottom + uniform, dtype=torch.float64)

    shifted = time_shift(t, shift, max_t=max_t, min_t=min_t)
    assert shifted[0] == max_t and shifted[len(top)] == min_t
    assert ((shifted >= min_t) & (shifted <= max_t)).all()
    low, high, s = Fraction(min_t), Fraction(max_t), Fraction(shift)
    for value, time in zip(shifted.tolist(), t.tolist()):
        y = (Fraction(time) - low) / (high - low)
        exact = low + (high - low) * s * y / (1 + (s - 1) * y)
        assert abs(Fraction(value) - exact) <= 3 * Fraction(math.ulp(float(exact))), (time, value)


def test_time_shift_keeps_numpy_arrays():
    shifted = time_shift(np.array([0.0, 0.5, 0.999]), 7.0, max_t=0.999)
    assert isinstance(shifted, np.ndarray) and shifted[0] == 0.0 and shifted[-1] == 0.999
    zero_dim = time_shift(np.array(0.5), 7.0, max_t=0.999)
    assert isinstance(zero_dim, np.ndarray) and zero_dim.shape == () and zero_dim == shifted[1]


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("max_t", [0.999, 1.0, 80.0])
@pytest.mark.parametrize("shift", [1 / 3, 1.0, 3.0, 12.0])
def test_shift_t_preserves_the_warps_derivative_and_representable_bounds(device_type, dtype, max_t, shift):
    """The endpoint slope is 1/shift even if the warp rounds above max_t; lower-precision bounds still round inward."""
    schedule = EDMNoiseSchedule(max_t=max_t)
    t = torch.tensor(
        [0.0, max_t / 4, max_t / 2, 0.9 * max_t, max_t], dtype=dtype, device=device_type, requires_grad=True
    )
    shifted = schedule.shift_t(t, shift=shift)
    assert shifted.dtype == dtype
    assert ((shifted.double() >= 0) & (shifted.double() <= max_t)).all()

    # d/dt [shift * max_t * t / (max_t + (shift - 1) * t)].
    time = t.detach().double()
    denominator = max_t + (shift - 1) * time
    expected_value = (shift * max_t * time / denominator).clamp(max=max_t).to(dtype)
    expected = (shift * max_t**2 / denominator.square()).to(dtype)
    tolerance = max(1e-12, 2 * torch.finfo(dtype).eps)
    torch.testing.assert_close(shifted, expected_value, rtol=tolerance, atol=0)
    torch.testing.assert_close(torch.autograd.grad(shifted.sum(), t)[0], expected, rtol=tolerance, atol=0)
    _, tangent = torch.func.jvp(lambda t: schedule.shift_t(t, shift=shift), (t.detach(),), (torch.ones_like(t),))
    torch.testing.assert_close(tangent, expected, rtol=tolerance, atol=0)


def test_per_sample_bounds_are_validated_and_clipped_to_the_schedule():
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)

    with pytest.raises(ValueError, match=r"Per-sample min_t must have shape \[3\]"):
        schedule.sample_t(3, time_dist_type="uniform", min_t=torch.zeros(2, dtype=torch.float64), max_t=0.5)

    t = schedule.sample_t(
        2,
        time_dist_type="uniform",
        min_t=torch.tensor([-1.0, 0.2], dtype=torch.float64),
        max_t=torch.tensor([0.5, 2.0], dtype=torch.float64),
    )
    assert 0.0 <= t[0] <= 0.5 and 0.2 <= t[1] <= 0.999


def test_composite_sample_t_passes_per_sample_bounds_to_its_primary():
    composite = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "shifted_rf", "shift": 12.0, "min_t": 0.0, "max_t": 0.999},
            "audio": {"name": "shifted_rf", "shift": 3.0, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )
    lower = torch.tensor([0.001, 0.6], dtype=torch.float64)
    upper = torch.tensor([0.6, 0.999], dtype=torch.float64)
    sample_kwargs = {"time_dist_type": "shifted", "shift": 5.0, "min_t": lower, "max_t": upper}

    torch.manual_seed(0)
    t = composite.sample_t(2, **sample_kwargs)
    torch.manual_seed(0)
    expected = composite.primary.sample_t(2, **sample_kwargs)

    torch.testing.assert_close(t, expected, rtol=0, atol=0)
    assert (t >= lower).all() and (t <= upper).all()


@pytest.mark.parametrize("time_dist_type", ["uniform", "shifted"])
def test_sample_t_rejects_intervals_that_are_empty_after_clipping(time_dist_type):
    """An inverted or zero-width interval raises instead of drawing NaN (a shift over a zero span) or collapsing onto
    max_t, also where it is clipping to the schedule range that empties it."""
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    lower = torch.tensor([0.2, 0.5], dtype=torch.float64)
    cases = [
        (0.9, 0.5),
        (0.5, 0.5),
        (0.9995, 1.0),
        (lower, lower),
        (lower, 0.2),
        (torch.tensor([0.2, 0.9995], dtype=torch.float64), torch.tensor([0.5, 1.0], dtype=torch.float64)),
    ]
    for min_t, max_t in cases:
        with pytest.raises(ValueError, match="min_t < max_t"):
            schedule.sample_t(2, time_dist_type=time_dist_type, shift=5.0, min_t=min_t, max_t=max_t)


@RunIf(min_gpus=1)
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature")
@pytest.mark.parametrize(
    ("make_schedule", "time_dist_type"),
    [
        pytest.param(lambda: RFNoiseSchedule(min_t=0.0, max_t=0.999), "shifted", id="rf-shifted"),
        pytest.param(lambda: EDMNoiseSchedule(min_t=0.002, max_t=80.0), "lognormal", id="edm-lognormal"),
        pytest.param(_cosine_alphas_schedule, "logitnormal", id="alphas-logitnormal"),
    ],
)
def test_per_sample_sample_t_does_not_synchronize_the_device(make_schedule, time_dist_type):
    """Per-sample bounds, which DMD2 passes on every step, are clipped, checked and drawn from on their device."""
    schedule = make_schedule()
    span = schedule.max_t - schedule.min_t
    lower = schedule.min_t + span * torch.tensor([0.1, 0.5], dtype=torch.float64, device="cuda")
    upper = lower + 0.3 * span
    torch.cuda.synchronize()

    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        t = schedule.sample_t(2, time_dist_type=time_dist_type, min_t=lower, max_t=upper, device="cuda")
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert t.device.type == "cuda" and ((t >= lower) & (t <= upper)).all()


@pytest.mark.parametrize(
    ("seq_len", "chunk_size", "chunk_starts"),
    [(8, 3, [5]), (2, 3, []), (6, 3, [3])],
    ids=["remainder", "shorter-than-a-chunk", "divisible"],
)
def test_per_chunk_samplers_put_remainder_frames_in_the_first_chunk(seq_len, chunk_size, chunk_starts):
    """Causal training draws one time and one student step per chunk, chunked like the causal attention masks:
    8 frames in chunks of 3 form chunks [0, 5) and [5, 8)."""
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    torch.manual_seed(0)
    t = schedule.sample_t_inhom(64, seq_len, chunk_size)
    t_student, step_indices = schedule.sample_student_timesteps_inhom(64, seq_len, chunk_size, num_steps=4)

    torch.testing.assert_close(t_student, schedule.student_grid(4)[step_indices])
    for frames in (t, step_indices):
        assert frames.shape == (64, seq_len)
        # Somewhere in the batch, the draws change exactly where a chunk starts.
        redrawn = (frames[:, 1:] != frames[:, :-1]).any(dim=0).nonzero().flatten() + 1
        assert redrawn.tolist() == chunk_starts


@RunIf(min_gpus=1)
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature")
def test_per_chunk_samplers_do_not_synchronize_the_device():
    """Causal training samples every step, so the per-chunk expansion stays on the device."""
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    torch.cuda.synchronize()

    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        t = schedule.sample_t_inhom(2, seq_len=8, chunk_size=3, device="cuda")
        _, step_indices = schedule.sample_student_timesteps_inhom(2, seq_len=8, chunk_size=3, device="cuda")
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert t.device.type == step_indices.device.type == "cuda"


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_alpha_and_sigma_primes_match_central_finite_differences(make_schedule):
    schedule = make_schedule()
    t = _interior_times(schedule, n=9)
    step = 1e-6 * (schedule.max_t - schedule.min_t)

    fd_alpha = (schedule.alpha(t + step) - schedule.alpha(t - step)) / (2 * step)
    fd_sigma = (schedule.sigma(t + step) - schedule.sigma(t - step)) / (2 * step)
    torch.testing.assert_close(schedule.alpha_prime(t), fd_alpha, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(schedule.sigma_prime(t), fd_sigma, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_cond_velocity_is_the_time_derivative_of_the_forward_process(make_schedule):
    schedule = make_schedule()
    torch.manual_seed(2)
    x0 = torch.randn(9, 3, dtype=torch.float64)
    eps = torch.randn(9, 3, dtype=torch.float64)
    t = _interior_times(schedule, n=9)[:, None]
    step = 1e-6 * (schedule.max_t - schedule.min_t)

    finite_difference = (schedule.forward_process(x0, eps, t + step) - schedule.forward_process(x0, eps, t - step)) / (
        2 * step
    )
    torch.testing.assert_close(schedule.cond_velocity(x0, eps, t), finite_difference, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


def test_edm_grid_matches_the_previous_table_and_the_closed_form():
    """FastGen's previous EDM schedule took its grids from entries 2 to 998 of a 1000-entry table, polynomially
    spaced with ``rho = 7`` between 0.002 and 80. With those entries as its range, the schedule's grid matches them
    exactly where they fall on the uniform positions and the closed form elsewhere, as does the 18-step grid of the
    EDM and EDM2 recipes over the default range. Grids may have any length, such as the 1000 steps MeanFlow
    normalizes its weights over."""

    def closed_form(schedule, num_steps):
        low_root, high_root = schedule.min_t ** (1 / 7), schedule.max_t ** (1 / 7)
        grid = (high_root + torch.linspace(0, 1, num_steps + 1, dtype=torch.float64) * (low_root - high_root)) ** 7
        grid[-1] = 0.0
        return grid

    ramp = torch.linspace(0, 1, 1000, dtype=torch.float64)
    table = ((80.0 ** (1 / 7) + ramp * (0.002 ** (1 / 7) - 80.0 ** (1 / 7))) ** 7).flip(0)
    schedule = EDMNoiseSchedule(min_t=table[2].item(), max_t=table[998].item())
    for num_steps in (1, 2, 4):
        table_grid = table[torch.linspace(998, 2, num_steps + 1).long()]
        table_grid[-1] = 0.0
        torch.testing.assert_close(schedule.grid(num_steps), table_grid, rtol=1e-12, atol=0)
    for num_steps in (18, 50):
        torch.testing.assert_close(schedule.grid(num_steps), closed_form(schedule, num_steps), rtol=1e-12, atol=0)
    default = EDMNoiseSchedule()
    torch.testing.assert_close(default.grid(18), closed_form(default, 18), rtol=1e-12, atol=0)
    long_grid = schedule.grid(1000)
    assert long_grid.shape == (1001,) and (long_grid[:-1] > long_grid[1:]).all()


def test_edm_grid_is_polynomially_spaced_by_its_rho():
    """For any ``rho``, the times of EDM's grid have ``1/rho`` powers running evenly between those of ``min_t`` and
    ``max_t``, with the clean end at 0. Native student grids shift that grid, and ``rho`` is no student-grid kwarg."""
    rho = 5.0
    schedule = EDMNoiseSchedule(rho=rho, student_grid_cfg=StudentGridConfig(steps=4, grid_kwargs={"shift": 2.0}))
    low, high = schedule.min_t, schedule.max_t
    grid = schedule.grid(4)
    assert grid[-1] == 0
    roots = torch.linspace(high ** (1 / rho), low ** (1 / rho), 5, dtype=grid.dtype)[:-1]
    torch.testing.assert_close(grid[:-1] ** (1 / rho), roots, rtol=1e-12, atol=0)
    torch.testing.assert_close(schedule.student_grid(), schedule.shift_t(grid, shift=2.0))
    with pytest.raises(ValueError, match="rho must be positive"):
        EDMNoiseSchedule(rho=0.0)

    schedule.student_grid_cfg = StudentGridConfig(steps=4, grid_kwargs={"rho": 7.0})
    with pytest.raises(ValueError, match="Unsupported native student grid kwargs"):
        schedule.student_grid()


@pytest.mark.parametrize("num_steps", [1, 16, 18])
@pytest.mark.parametrize("t_precision", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "name, kwargs", [("edm", {}), ("edm", {"max_t": 0.999}), ("rf", {}), ("trig", {}), ("ddpm", {})]
)
def test_grid_in_a_coarse_time_precision_rounds_only_its_times(name, kwargs, t_precision, num_steps):
    """Grids are computed in high precision, so the grid of a coarse time precision is the float64 grid rounded once
    and clamped, also where its times or ``max_t`` are not exact in it, as 0.999 is not. A finer dtype converts those
    coarse times exactly, including a ``max_t`` the clamp rounded inward."""
    coarse = get_noise_schedule(name, t_precision=t_precision, **kwargs)
    fine = get_noise_schedule(name, t_precision="float64", **kwargs)
    grid = coarse.grid(num_steps)
    assert grid.dtype == coarse.t_precision
    assert torch.equal(grid, coarse.clamp_grid(fine.grid(num_steps).to(coarse.t_precision)))
    assert torch.equal(coarse.grid(num_steps, dtype=torch.float64), grid.double())


@pytest.mark.parametrize("t_precision", ["float32", "bfloat16"])
@pytest.mark.parametrize("name", ["edm", "rf"])
def test_shifted_student_grid_in_a_coarse_time_precision_rounds_only_its_times(name, t_precision):
    """Native student grids shift the high-precision grid before it is rounded, so a coarse time precision rounds
    the shifted times once, before the anchored grid is subsampled; a finer dtype converts them exactly."""
    student_grid_cfg = StudentGridConfig(steps=4, grid_kwargs={"base_steps": 18, "shift": 3.0})
    coarse = get_noise_schedule(name, t_precision=t_precision, student_grid_cfg=student_grid_cfg)
    fine = get_noise_schedule(name, t_precision="float64", student_grid_cfg=student_grid_cfg)
    grid = coarse.student_grid()
    assert grid.dtype == coarse.t_precision
    assert torch.equal(grid, coarse.clamp_grid(fine.student_grid().to(coarse.t_precision)))
    assert torch.equal(coarse.student_grid(dtype=torch.float64), grid.double())


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
def test_explicit_student_grid_defaults_to_the_schedule_device(device_type):
    """Explicit grids follow the schedule's device like native grids, including through a composite."""
    explicit = StudentGridConfig(steps=2, grid_type="explicit", grid_kwargs={"timesteps": [0.999, 0.5, 0.0]})
    single = RFNoiseSchedule(min_t=0.0, max_t=0.999, student_grid_cfg=explicit)
    composite = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "shifted_rf", "shift": 3.0, "min_t": 0.0, "max_t": 0.999, "student_grid_cfg": explicit},
            "audio": {"name": "rf", "min_t": 0.0, "max_t": 0.999},
        }
    )

    device = torch.device("cuda", torch.cuda.current_device()) if device_type == "cuda" else torch.device("cpu")
    for schedule in (single, composite):
        assert schedule.device is None
        assert schedule.student_grid().device.type == "cpu"
        schedule.prepare_module(device)
        grid = schedule.student_grid()
        assert grid.device == schedule.device == schedule.grid(2).device == device
        torch.testing.assert_close(grid.cpu(), torch.tensor([0.999, 0.5, 0.0], dtype=torch.float64))
    assert composite.device == composite.primary.device


def test_safe_clamp_warn_flag_reports_moved_timesteps(monkeypatch):
    """``warn=True`` logs once when the clamp actually moves values; in-range grids and the default stay silent."""
    import fastgen.networks.noise_schedule as noise_schedule_module

    messages = []
    monkeypatch.setattr(noise_schedule_module.logger, "warning", messages.append)

    grid = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
    clamped = RFNoiseSchedule.safe_clamp(grid, min=0.0, max=0.999, warn=True)
    torch.testing.assert_close(clamped, torch.tensor([0.999, 0.5, 0.0], dtype=torch.float64))
    assert len(messages) == 1 and "1 of 3" in messages[0]

    RFNoiseSchedule.safe_clamp(grid, min=0.0, max=1.0, warn=True)  # in range: silent
    RFNoiseSchedule.safe_clamp(grid, min=0.0, max=0.999)  # default: silent
    assert len(messages) == 1

    RFNoiseSchedule.safe_clamp(grid.float(), min=0.0, max=0.999, warn=True)  # float32 path (nextafter bounds)
    assert len(messages) == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_safe_clamp_keep_grad_keeps_the_derivative_where_it_clamps(dtype):
    """``keep_grad=True`` returns the same values, with the derivative of t also where the clamp moves it."""
    above = torch.nextafter(torch.tensor(0.999, dtype=dtype), torch.tensor(math.inf, dtype=dtype))
    t = torch.stack([torch.tensor(0.5, dtype=dtype), above, torch.nextafter(above, above + 1)]).requires_grad_()
    hard = BaseNoiseSchedule.safe_clamp(t, max=0.999)
    kept = BaseNoiseSchedule.safe_clamp(t, max=0.999, keep_grad=True)
    assert torch.equal(kept, hard) and (kept.double() <= 0.999).all()
    assert torch.equal(torch.autograd.grad(hard.sum(), t)[0], torch.tensor([1.0, 0.0, 0.0], dtype=dtype))
    assert torch.equal(torch.autograd.grad(kept.sum(), t)[0], torch.ones(3, dtype=dtype))


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_safe_clamp_keep_grad_preserves_values_for_large_corrections(device_type, dtype):
    """The STE keeps the clamped values even when a correction would round away the bound."""
    large = torch.finfo(dtype).max
    t = torch.tensor([-large, 0.5, large], dtype=dtype, device=device_type, requires_grad=True)

    def clamp(t):
        return BaseNoiseSchedule.safe_clamp(t, min=0.25, max=0.75, keep_grad=True)

    clamped = clamp(t)
    expected = torch.tensor([0.25, 0.5, 0.75], dtype=dtype, device=device_type)
    assert torch.equal(clamped, expected)
    assert torch.equal(torch.autograd.grad(clamped.sum(), t)[0], torch.ones_like(t))
    tangent = torch.tensor([2.0, -1.0, 0.5], dtype=dtype, device=device_type)
    _, jvp = torch.func.jvp(clamp, (t.detach(),), (tangent,))
    assert torch.equal(jvp, tangent)


def test_latents_requires_the_initial_time():
    """Sampler grids do not start at ``max_t`` in general, so ``latents`` takes the grid's first time explicitly."""
    schedule = RFNoiseSchedule(min_t=0.0, max_t=0.999)
    noise = torch.randn(2, 3, 4, 4)
    t_init = torch.tensor(0.75, dtype=schedule.t_precision)

    scaled = schedule.latents(noise, t_init=t_init)
    torch.testing.assert_close(scaled, (noise.double() * schedule.sigma(t_init).double()).to(noise.dtype))
    with pytest.raises(TypeError):
        schedule.latents(noise)


_SCHEDULES = ["edm", "rf", "trig", "ddpm", "cogvideox", "composite"]


def _schedule(name, t_precision):
    kwargs = {"t_precision": t_precision}
    if name == "cogvideox":
        kwargs["snr_shift_scale"] = 3.0
    elif name == "composite":
        return CompositeNoiseSchedule(
            schedules={
                "video": {"name": "rf", "max_t": 0.9, **kwargs},
                "audio": {"name": "ddpm", "max_t": 0.9, **kwargs},
            }
        )
    return get_noise_schedule(name, **kwargs)


@pytest.mark.parametrize("name", _SCHEDULES)
@pytest.mark.parametrize("t_precision", ["float32", "float64"])
def test_nested_schedule_pins_buffers_and_excludes_them_from_checkpoints(name, t_precision):
    root = FastGenModule()
    root.proj = torch.nn.Linear(2, 2)
    schedule = _schedule(name, t_precision)
    root.container = torch.nn.ModuleDict({"schedule": schedule})
    original = {name: tensor.clone() for name, tensor in schedule.named_buffers()}

    root.prepare_module("cpu", precision_model=torch.bfloat16, precision_opt=torch.float32)

    assert set(root.state_dict()) == {"proj.weight", "proj.bias"}
    # The Alphas schedules have tables, and the composite through its DDPM child; EDM, RF and Trig have none.
    assert bool(original) == (name in ("ddpm", "cogvideox", "composite"))
    for module in schedule.modules():
        if isinstance(module, FastGenModule):
            assert module.is_prepared
            assert module.device == torch.device("cpu")
            assert module.precision_model == torch.bfloat16
            assert module.precision_opt == torch.float32
    for name, tensor in schedule.named_buffers():
        assert tensor.dtype == torch.float64 and schedule.t_precision == getattr(torch, t_precision)
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
    assert schedule.grid(3).device == schedule.device == torch.device("cpu")
    assert schedule.sample_t(2, time_dist_type="uniform").dtype == schedule.t_precision


@pytest.mark.parametrize("t_precision", ["bfloat16", "float32"])
def test_alphas_tables_keep_float64_entries_in_coarser_time_precisions(t_precision):
    """In bfloat16 DDPM's first entry 0.9999 would round to 1, which would make time 0 clean."""
    schedule = get_noise_schedule("ddpm", t_precision=t_precision)
    assert torch.equal(schedule.alphas_cumprod, get_noise_schedule("ddpm").alphas_cumprod)
    sigma = schedule.sigma(torch.zeros(1, dtype=schedule.t_precision))
    torch.testing.assert_close(sigma.double(), torch.tensor([0.01], dtype=torch.float64), rtol=1e-2, atol=0)


@pytest.mark.parametrize("name", _SCHEDULES)
@pytest.mark.parametrize("t_precision", ["float32", "float64"])
def test_meta_schedule_reset_preserves_materialized_device_dtype_and_values(name, t_precision):
    reference = _schedule(name, t_precision)
    with torch.device("meta"):
        schedule = _schedule(name, t_precision)
    assert all(tensor.is_meta for tensor in schedule.buffers())
    # CPU offload materializes here while the recorded execution target remains CUDA.
    schedule._set_device("cuda")
    schedule._set_precision(precision_model=torch.bfloat16, precision_opt=torch.float32)
    materialize_meta_module(schedule, "cpu")

    original_slots = dict(schedule.named_buffers())
    for tensor in schedule.buffers():
        tensor.fill_(float("nan"))
    schedule.reset_parameters()

    expected = dict(reference.named_buffers())
    for name, tensor in schedule.named_buffers():
        assert tensor is original_slots[name]
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["sd", "sdxl"])
def test_diffusers_schedule_keeps_reconstruction_values_during_meta_init(monkeypatch, name):
    import fastgen.networks.noise_schedule as schedules

    def load_scheduler(*args, **kwargs):
        # This mimics diffusers' factory operations under the active device context.
        return SimpleNamespace(alphas_cumprod=torch.linspace(0.999, 0.001, 1000))

    monkeypatch.setattr(schedules.DDIMScheduler, "from_pretrained", load_scheduler)
    reference = get_noise_schedule(name)
    with torch.device("meta"):
        schedule = get_noise_schedule(name)
    assert schedule.alphas_cumprod.is_meta
    materialize_meta_module(schedule, "cpu")
    torch.testing.assert_close(schedule.alphas_cumprod, reference.alphas_cumprod, rtol=0, atol=0)


def test_alphas_schedule_owns_its_table_and_retains_the_original_values_for_reset():
    source = torch.linspace(0.999, 0.001, 1000, dtype=torch.float64)
    expected = source.clone()
    schedule = AlphasNoiseSchedule(source)
    source.zero_()
    schedule.alphas_cumprod.zero_()
    schedule.reset_parameters()
    torch.testing.assert_close(schedule.alphas_cumprod, expected, rtol=0, atol=0)


class _ScheduleNetwork(FastGenNetwork):
    """A small network using the production network reset and explicit FSDP boundary."""

    def __init__(self, model_id_or_local_path=None, transformer_subfolder=None, **kwargs):
        super().__init__(**kwargs)
        self.transformer = torch.nn.Linear(2, 2)
        self.transformer.blocks = torch.nn.ModuleList()

    def reset_parameters(self):
        super().reset_parameters()
        self.transformer.reset_parameters()

    def fully_shard(self, **kwargs):
        fully_shard(self.transformer, **kwargs)

    def forward(self, x_t, t, **kwargs):
        output = self.transformer(x_t.to(self.precision_model)).to(x_t.dtype)
        alpha = self.noise_scheduler.alpha(t).reshape(-1, *([1] * (output.ndim - 1)))
        return output * alpha.to(output.dtype)


def _shared_network(schedule_type="rf", t_precision="float64"):
    from fastgen.networks.Wan.wan_moe import WanMoE

    return WanMoE(_ScheduleNetwork, "unused", schedule_type=schedule_type, t_precision=t_precision)


def test_wan_moe_preparation_and_reset_keep_the_shared_scheduler():
    network = _shared_network()
    schedule = network.noise_scheduler
    network.prepare_module("cpu", precision_model=torch.bfloat16, precision_opt=torch.float32)
    network.reset_parameters()

    assert network.noise_scheduler is network.network_high.noise_scheduler is schedule
    assert schedule.is_prepared and schedule.device == torch.device("cpu")
    assert schedule.t_precision == torch.float64
    assert all("noise_scheduler" not in key for key in network.state_dict())


@pytest.mark.parametrize("name", ["rf", "ddpm", "composite"])
@pytest.mark.parametrize("min_num_params", [0, 1, 10**9])
def test_generic_fsdp_does_not_wrap_schedule_buffers_as_precision_islands(monkeypatch, name, min_num_params):
    import fastgen.utils.distributed.fsdp as fsdp_utils

    root = FastGenModule()
    root.projection = torch.nn.Linear(2, 2)
    root.noise_scheduler = _schedule(name, "float64")
    root.shared = torch.nn.ModuleDict({"schedule": root.noise_scheduler})
    root.prepare_module("cpu", precision_model=torch.bfloat16, precision_opt=torch.float32)
    wrapped = []
    monkeypatch.setattr(fsdp_utils, "fully_shard", lambda module, **kwargs: wrapped.append(module))

    fsdp_utils._fully_shard_module(root, min_num_params=min_num_params, mesh="mesh")

    assert wrapped == ([root.projection, root] if min_num_params <= 6 else [root])
    assert all(tensor.dtype == torch.float64 for tensor in root.noise_scheduler.buffers())


class _GenericScheduleRoot(FastGenModule):
    def __init__(self, schedule_type):
        super().__init__()
        self.net = _ScheduleNetwork(schedule_type=schedule_type)
        self.noise_scheduler = self.net.noise_scheduler

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, x, t):
        return self.net(x, t)


def _distributed_schedule_impl(rank, world_size, mode, cpu_offload=False):
    device = torch.device("cuda", rank)
    # Each run includes a schedule without tables and an alphas table reconstructed from CPU values.
    for schedule_type in ("rf", "ddpm"):
        reference = get_noise_schedule(schedule_type)
        expected = {name: tensor.to(device) for name, tensor in reference.named_buffers()}
        generic = mode == "generic_fsdp"
        with torch.device("cpu" if rank == 0 else "meta"):
            network = _GenericScheduleRoot(schedule_type) if generic else _shared_network(schedule_type)
        shared = network.noise_scheduler
        context = None
        if mode != "ddp":
            context = create_fsdp_context(
                reduce_dtype=torch.float32,
                min_num_params=0 if generic else 1,
                apply_cpu_offload=cpu_offload,
            )
        network.prepare_module(
            device,
            precision_model=torch.bfloat16,
            precision_opt=torch.float32,
            trainable=True,
            fsdp_context=context,
        )
        experts = (network.net,) if generic else (network.network_high, network.network_low)
        assert network.noise_scheduler is experts[0].noise_scheduler is shared
        for expert in experts:
            assert expert.noise_scheduler.is_prepared
            assert expert.noise_scheduler.device == device
            assert expert.noise_scheduler.grid(3).device == device
            assert expert.noise_scheduler.sample_t(2, time_dist_type="uniform").device == device
            for name, tensor in expert.noise_scheduler.named_buffers():
                torch.testing.assert_close(tensor, expected[name])
                assert tensor.dtype == torch.float64
            assert not any(isinstance(module, FSDPModule) for module in expert.noise_scheduler.modules())
        if generic:
            assert {id(module) for module in network.modules() if isinstance(module, FSDPModule)} == {
                id(network),
                id(network.net.transformer),
            }
        assert all("noise_scheduler" not in key for key in network.state_dict())

        wrapped = network
        if mode == "ddp":
            network.config = SimpleNamespace(ddp_find_unused_parameters=False)
            # DDP construction must also synchronize the non-persistent buffers.
            if rank != 0:
                for tensor in network.buffers():
                    tensor.zero_()
            wrapped = model_to_ddp(network)
            for name, tensor in shared.named_buffers():
                torch.testing.assert_close(tensor, expected[name])

        for _ in range(2):
            if mode == "ddp" and rank != 0:
                for tensor in network.buffers():
                    tensor.zero_()
            # Both experts execute, so the shared scheduler must survive DDP's repeated broadcasts
            # and FSDP's parameter offload without moving its table or lowering its precision.
            x = torch.ones(2, 1, 1, 1, 2, device=device)
            t = torch.tensor([0.25, 0.95], device=device, dtype=shared.t_precision)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = wrapped(x, t)
            output.sum().backward()
            assert torch.isfinite(output).all()
            for expert in experts:
                for name, tensor in expert.noise_scheduler.named_buffers():
                    torch.testing.assert_close(tensor, expected[name])
            for parameter in network.parameters():
                grad = parameter.grad
                assert grad is not None
                local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
                assert torch.isfinite(local_grad).all()
            network.zero_grad(set_to_none=True)
        dist.barrier()
    return {"checked": True}


@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    "mode,cpu_offload",
    [("ddp", False), ("fsdp", False), ("fsdp", True), ("generic_fsdp", False), ("generic_fsdp", True)],
)
def test_distributed_meta_initialization_preserves_shared_schedule_buffers(mode, cpu_offload):
    result = run_distributed_test(
        _distributed_schedule_impl, world_size=2, timeout=180, mode=mode, cpu_offload=cpu_offload
    )
    assert result == {"checked": True}
