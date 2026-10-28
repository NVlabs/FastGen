# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU tests for Parallel Decoding Distillation.

The PDD arithmetic assertions use ``torch.equal`` where operation order is part of the supported behavior.
MiniMax-H3 additionally exercises the PDD core with raw projections that are
already native sigma-velocity heads and therefore bypass output conversion.
"""

import copy
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from einops import rearrange

import fastgen.methods.trajectory_matching.pdd as pdd_module
from fastgen.callbacks.carry import CARRY_OUTPUT_KEY, CARRY_OVERLAY_KEYS, CarryCallback
from fastgen.configs.methods.config_pdd import ModelConfig as PDDModelConfig
from fastgen.methods.model import FastGenModel
from fastgen.methods.trajectory_matching.pdd import PDDGridPlan, PDDModel, PDDModelMixin
from fastgen.networks.MiniMaxH3.network_pdd import MiniMaxH3PDD
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import (
    CompositeNoiseSchedule,
    EDMNoiseSchedule,
    ShiftedRFNoiseSchedule,
    StudentGridConfig,
    get_noise_schedule,
    time_shift,
)
from fastgen.networks.utils.pdd_mixin import (
    PDDLinear,
    PDDMixin,
    PDDSpec,
    make_pdd,
)
from fastgen.utils.global_vars import set_temp_global_vars
from fastgen.utils.guidance import GuidanceConfig
from fastgen.utils.tensor import expand_like, high_precision_dtype
from fastgen.utils.tree_utils import get_device, per_modality, to
from tests.model_utils import (
    FunctionNet,
    _FakePDDNet,
    _ToyPDD,
    _ToyPredictionBase,
    _explicit_grid,
    _native_grid,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pdd_grid_plan_preserves_source_tensors_and_derives_differentiable_student_grid(dtype):
    device = torch.device("cpu")
    base_grid = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0], dtype=dtype, device=device, requires_grad=True)
    student_grid_indices = torch.tensor([0, 2, 4], device=device)
    calls = []

    def get_base_grid(device):
        calls.append(("base", device))
        return base_grid

    def get_student_grid_indices(target_steps, device):
        calls.append(("indices", target_steps, device))
        return student_grid_indices

    net = SimpleNamespace(
        pdd_base_grid=get_base_grid,
        noise_scheduler=SimpleNamespace(student_grid_indices=get_student_grid_indices),
    )
    model = object.__new__(PDDModelMixin)

    plan = model._pdd_student_grid_plan(net, num_steps=2, device=device)

    assert isinstance(plan, PDDGridPlan)
    assert calls == [("base", device), ("indices", 2, device)]
    assert plan.base_grid is base_grid
    assert plan.student_grid_indices is student_grid_indices
    assert plan.student_grid_indices.dtype == torch.long
    assert plan.student_grid_indices.device == device
    student_grid = plan.student_grid
    assert student_grid.dtype == dtype and student_grid.device == device
    assert student_grid.requires_grad
    torch.testing.assert_close(student_grid, torch.tensor([1.0, 0.5, 0.0], dtype=dtype, device=device))
    (gradient,) = torch.autograd.grad(student_grid.sum(), base_grid)
    torch.testing.assert_close(gradient, torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], dtype=dtype, device=device))


def test_pdd_method_config_owns_its_deterministic_sampling_mode():
    config = PDDModelConfig()
    assert config.student_sample_type == "ode"
    assert config.backward_simulation is None
    assert not hasattr(config.loss_config, "data_free")
    assert config.loss_config.num_supervised_heads == 1
    assert config.precision_decoding is None

    config.student_sample_type = "sde"
    with pytest.raises(ValueError, match="PDDModel student_sample_type must be 'ode'; got 'sde'"):
        PDDModel(config)


@pytest.mark.parametrize(
    ("block_size_min", "block_size_max", "message"),
    [
        (0, 1, "block_size_min must be positive"),
        (1, 0, "block_size_max must be positive"),
        (2, 1, "block_size_min=2 must not exceed block_size_max=1"),
    ],
)
def test_pdd_model_validates_block_size_bounds_at_construction(block_size_min, block_size_max, message):
    config = PDDModelConfig()
    config.loss_config.block_size_min = block_size_min
    config.loss_config.block_size_max = block_size_max

    with pytest.raises(ValueError, match=message):
        PDDModel(config)


def test_pdd_model_rejects_nonpositive_supervised_head_count():
    config = PDDModelConfig()
    config.loss_config.num_supervised_heads = 0

    with pytest.raises(ValueError, match="num_supervised_heads must be positive"):
        PDDModel(config)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, torch.bfloat16), ("float64", torch.float64)],
)
def test_pdd_model_owns_decoding_precision_and_defaults_it_to_method_precision(monkeypatch, configured, expected):
    config = PDDModelConfig()
    config.precision = "bfloat16"
    config.precision_decoding = configured

    def init_fastgen_model(self, config):
        torch.nn.Module.__init__(self)
        self.precision = {"bfloat16": torch.bfloat16}[config.precision]

    monkeypatch.setattr(FastGenModel, "__init__", init_fastgen_model)

    model = PDDModel(config)

    assert model.precision_decoding == expected


class _SamplerHarness(PDDModel):
    pass


class _RKFlowHarness(PDDModel):
    pass


class _BackwardSimulationCarryHarness(PDDModel):
    def _compute_loss(self, input_student, step_idx, base_grid, condition, neg_condition, next_idx=None):
        del base_grid
        self.seen.append((input_student, step_idx, next_idx, condition, neg_condition))
        loss = (input_student + self.anchor).float().mean()
        with torch.no_grad():
            next_state = input_student + self.anchor
            x0_approx = next_state + self.anchor
        return {"total_loss": loss}, x0_approx, next_state

    def _get_outputs(self, gen_data, **kwargs):
        return {"gen_rand_train": gen_data}


def _assert_tree_equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert torch.equal(actual[key], expected[key]), key
    else:
        assert torch.equal(actual, expected)


def _build_teacher_pdd_target(model, x_t, step_idx, base_grid, condition=None, neg_condition=None):
    """Exercise the same scheduler-owned teacher-target boundary as PDD training."""
    with torch.no_grad():
        return model.net.pdd_teacher_target(
            x_t,
            step_idx,
            base_grid,
            solver=model.config.loss_config.type_rk,
            predict_teacher=partial(
                model._guided_teacher,
                condition=condition,
                neg_condition=neg_condition,
            ),
        )


def _advance_materialized_pdd_block(net, state, pred, start, end, base_grid, allow_empty=False):
    return net.pdd_advance_blocks(
        state,
        pred,
        start,
        end,
        base_grid,
        allow_empty=allow_empty,
    )


def _predict_heads(model, net, state, step_index, condition=None):
    """Student training forward; PDD carries states in method precision, so tests cast here."""
    return net(to(state, dtype=model.precision), step_index=step_index, condition=condition)


def _sigma_delta(schedule, t_start, t_end):
    """Expected native sigma increment, computed directly from the warp."""
    return schedule.sigma(t_end) - schedule.sigma(t_start)


def _interval_weights(net, base_grid):
    """Every base-grid interval's sigma increment, through the ``determinant`` the PDD weights use."""
    return per_modality(
        lambda schedule: schedule.determinant(base_grid[:-1], base_grid[1:]),
        net.noise_scheduler.schedules,
    )


def _sigma_to_t(schedule, sigma):
    """Expected inverse warp for RK stage placement (exponential shifts form a group)."""
    return time_shift(sigma, 1.0 / schedule.shift, max_t=schedule.max_t)


def test_backward_simulation_pdd_uses_atomic_carry():
    """PDD keeps state and batch metadata paired, then drops both at the trajectory boundary."""
    base_grid = torch.tensor([0.999, 0.7, 0.3, 0.0], dtype=torch.float64)
    model = object.__new__(_BackwardSimulationCarryHarness)
    torch.nn.Module.__init__(model)
    model.net = _FakePDDNet(get_noise_schedule("rf"), pdd_steps=3, base_grid=base_grid)
    model.config = SimpleNamespace(
        backward_simulation="staggered",
        loss_config=SimpleNamespace(block_size_min=2, block_size_max=2),
        student_sample_type="ode",
    )
    model.device = torch.device("cpu")
    model.precision = torch.bfloat16
    model.precision_decoding = torch.float32
    model.input_shape = [1]
    model.anchor = torch.nn.Parameter(torch.tensor(1.0))
    model.net.predictions = lambda x, start, end, condition: torch.ones_like(x)
    model.seen = []
    callback = CarryCallback()

    first = {
        "condition": torch.tensor([[1.0], [2.0]]),
        "neg_condition": torch.zeros(2, 1),
        "condition_raw": ["old-a", "old-b"],
        "idx": [10, 11],
    }
    _, outputs = model.single_train_step(first, iteration=1)
    carry = outputs[CARRY_OUTPUT_KEY]
    assert carry["step_carried"] == 1
    assert set(carry) == {*(key for key in CARRY_OVERLAY_KEYS if key in first), "state_carried", "step_carried"}
    assert not carry["state_carried"].requires_grad and carry["state_carried"].grad_fn is None
    torch.testing.assert_close(outputs["gen_rand_train"], carry["state_carried"] + model.anchor)
    assert not torch.equal(outputs["gen_rand_train"], carry["state_carried"])

    callback.on_training_accum_step_end(model, first, outputs, iteration=1, accum_iter=0)
    fresh_second = {
        "condition": torch.tensor([[8.0], [9.0]]),
        "neg_condition": torch.ones(2, 1),
        "condition_raw": ["discard-a", "discard-b"],
        "idx": [20, 21],
    }
    second = dict(fresh_second)
    callback.on_training_accum_step_begin(model, second, iteration=2, accum_iter=0)
    assert second["condition"] is first["condition"]
    assert second["condition_raw"] == first["condition_raw"]
    assert second["idx"] == first["idx"]
    assert second["state_carried"] is carry["state_carried"]

    _, outputs = model.single_train_step(second, iteration=2)
    assert model.seen[-1][0] is carry["state_carried"]
    student_grid_indices = model.net.noise_scheduler.student_grid_indices(
        target_steps=model._max_student_steps(),
    )
    expected_base_step = int(student_grid_indices[1].item())
    torch.testing.assert_close(model.seen[-1][1], torch.full((2,), expected_base_step, dtype=torch.long))
    expected_base_end = int(student_grid_indices[2].item())
    torch.testing.assert_close(model.seen[-1][2], torch.full((2,), expected_base_end, dtype=torch.long))
    assert outputs[CARRY_OUTPUT_KEY] == {"step_carried": model._max_student_steps()}
    torch.testing.assert_close(outputs["gen_rand_train"], model.seen[-1][0] + 2 * model.anchor)

    callback.on_training_accum_step_end(model, second, outputs, iteration=2, accum_iter=0)
    fresh_third = {
        "condition": torch.tensor([[30.0], [31.0]]),
        "neg_condition": torch.full((2, 1), -1.0),
        "condition_raw": ["new-a", "new-b"],
        "idx": [30, 31],
    }
    third = dict(fresh_third)
    callback.on_training_accum_step_begin(model, third, iteration=3, accum_iter=0)
    assert "state_carried" not in third
    assert third["condition_raw"] == ["new-a", "new-b"]
    _, outputs = model.single_train_step(third, iteration=3)
    assert outputs[CARRY_OUTPUT_KEY]["condition"] is third["condition"]
    assert outputs[CARRY_OUTPUT_KEY]["condition_raw"] == ["new-a", "new-b"]


def test_backward_simulation_pdd_staggers_only_with_gradients_enabled(monkeypatch):
    import fastgen.methods.backward_simulation as backward_simulation

    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = _FakePDDNet(get_noise_schedule("rf"), pdd_steps=4)
    model.config = SimpleNamespace(
        backward_simulation="staggered",
        loss_config=SimpleNamespace(block_size_min=1, block_size_max=4),
    )
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    model.input_shape = [1]
    calls = []
    loss_inputs = []
    fresh_noise_dtypes = []

    monkeypatch.setattr(backward_simulation, "get_rank", lambda: 2)
    monkeypatch.setattr(backward_simulation, "world_size", lambda: 4)

    def capture_noise(input_shape, batch_size, device, dtype):
        fresh_noise_dtypes.append(dtype)
        return torch.zeros(batch_size, *input_shape, device=device, dtype=dtype)

    monkeypatch.setattr(backward_simulation, "noise_from_shape", capture_noise)

    def advance(net, state, step_index, student_grid, condition):
        calls.append((step_index, condition, torch.is_grad_enabled()))
        return state + 1

    def compute_loss(input_student, step_idx, base_grid, condition, neg_condition, next_idx=None):
        assert base_grid.shape == (model.net.pdd_steps + 1,)
        loss_inputs.append((input_student, step_idx, next_idx, condition, neg_condition))
        with torch.no_grad():
            next_state = input_student + 1
            x0_approx = input_student + 2
        return {"total_loss": input_student.mean()}, x0_approx, next_state

    monkeypatch.setattr(model, "_advance_student_step", advance)
    monkeypatch.setattr(model, "_compute_loss", compute_loss)
    monkeypatch.setattr(model, "_get_outputs", lambda gen_data, **kwargs: {"gen_rand_train": gen_data})
    _, outputs = model.single_train_step(
        {"idx": [0, 1], "condition": "prompt"},
        iteration=1,
    )
    x0_approx = outputs["gen_rand_train"]
    carry = outputs[CARRY_OUTPUT_KEY]

    # Every rank completes the three-forward collective-safe stagger; rank two
    # retains its step-two snapshot. The loss forward also supplies its detached carry,
    # so there is no fourth, carry-only student forward with gradients enabled.
    assert calls == [
        (0, "prompt", False),
        (1, "prompt", False),
        (2, "prompt", False),
    ]
    torch.testing.assert_close(loss_inputs[0][0], torch.full_like(x0_approx, 2.0))
    torch.testing.assert_close(loss_inputs[0][2], torch.full((2,), 3, dtype=torch.long))
    torch.testing.assert_close(x0_approx, torch.full_like(x0_approx, 4.0))
    assert not x0_approx.requires_grad
    assert carry["step_carried"] == 3
    torch.testing.assert_close(carry["state_carried"], torch.full_like(x0_approx, 3.0))
    assert not torch.equal(carry["state_carried"], x0_approx)
    assert fresh_noise_dtypes == [model.precision]

    calls.clear()
    loss_inputs.clear()
    with torch.no_grad():
        _, outputs = model.single_train_step(
            {"idx": [0, 1], "condition": "prompt"},
            iteration=0,
        )
        x0_approx = outputs["gen_rand_train"]
        carry = outputs[CARRY_OUTPUT_KEY]

    # Validation starts at the first trajectory interval without a rank-dependent stagger.
    assert calls == []
    torch.testing.assert_close(loss_inputs[0][0], torch.zeros_like(x0_approx))
    torch.testing.assert_close(loss_inputs[0][2], torch.full((2,), 1, dtype=torch.long))
    torch.testing.assert_close(x0_approx, torch.full_like(x0_approx, 2.0))
    torch.testing.assert_close(carry["state_carried"], torch.ones_like(x0_approx))
    assert carry["step_carried"] == 1
    assert fresh_noise_dtypes == [model.precision, model.precision]

    # The opt-in ablation retains normal training autograd but removes the rank
    # offset. At the first iteration every rank therefore optimizes step zero
    # without any initialization rollout.
    model.config.backward_simulation = "synced"
    calls.clear()
    loss_inputs.clear()
    _, outputs = model.single_train_step(
        {"idx": [0, 1], "condition": "prompt"},
        iteration=1,
    )
    x0_approx = outputs["gen_rand_train"]
    carry = outputs[CARRY_OUTPUT_KEY]
    assert calls == []
    torch.testing.assert_close(loss_inputs[0][0], torch.zeros_like(x0_approx))
    torch.testing.assert_close(loss_inputs[0][1], torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(loss_inputs[0][2], torch.ones(2, dtype=torch.long))
    assert carry["step_carried"] == 1


@pytest.mark.parametrize("backward_simulation", [None, "staggered"], ids=["real-data", "backward-simulation"])
def test_pdd_training_reuses_one_grid_plan_and_preserves_branch_endpoints(monkeypatch, backward_simulation):
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        backward_simulation=backward_simulation,
        loss_config=SimpleNamespace(block_size_min=3, block_size_max=4),
    )
    # A distinct placement token checks dispatch without actually allocating meta state.
    model.device = torch.device("meta")
    model.precision = torch.float32
    model.input_shape = [1]
    real = torch.zeros(2, 1)
    noise = torch.ones_like(real)
    state = torch.full_like(real, 2.0)
    plan = PDDGridPlan(
        base_grid=torch.tensor([0.99, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.0], dtype=torch.float64),
        student_grid_indices=torch.tensor([0, 2, 5, 7]),
    )
    events = []
    plan_calls = []

    def grid_plan(net, num_steps, device):
        assert net is model.net and num_steps == 3
        plan_calls.append(device)
        return plan

    def sample_noise(value):
        assert value is real
        events.append("noise")
        return noise

    def sample_starts(high, size, device, dtype):
        assert high == 3 and size == (2,) and device == model.device and dtype == torch.long
        events.append("starts")
        return torch.tensor([1, 2])

    def forward_process(value, eps, t):
        assert value is real and eps is noise
        torch.testing.assert_close(t, plan.base_grid[torch.tensor([2, 5])])
        return state

    def resolve(data, student_grid, **kwargs):
        assert "real" not in data
        torch.testing.assert_close(student_grid, plan.student_grid)
        events.append("resolve")
        return state, 1

    def compute_loss(input_student, step_idx, base_grid, condition, neg_condition, next_idx):
        assert input_student is state and base_grid is plan.base_grid
        expected_starts, expected_ends = ([2, 5], [6, 7]) if backward_simulation is None else ([2, 2], [5, 5])
        torch.testing.assert_close(step_idx, torch.tensor(expected_starts))
        torch.testing.assert_close(next_idx, torch.tensor(expected_ends))
        return {"total_loss": state.mean()}, state, state

    model.net = SimpleNamespace(pdd_steps=7, noise_scheduler=SimpleNamespace(forward_process=forward_process))
    monkeypatch.setattr(model, "_pdd_student_grid_plan", grid_plan)
    monkeypatch.setattr(pdd_module, "randn_like", sample_noise)
    monkeypatch.setattr(pdd_module.torch, "randint", sample_starts)
    monkeypatch.setattr(pdd_module, "resolve_trajectory_point", resolve)
    monkeypatch.setattr(model, "_compute_loss", compute_loss)
    monkeypatch.setattr(model, "_get_outputs", lambda gen_data, **kwargs: {"gen_rand_train": gen_data})
    data = {"idx": [0, 1], **({"real": real} if backward_simulation is None else {})}

    _, outputs = model.single_train_step(data, iteration=1)

    assert plan_calls == [real.device if backward_simulation is None else model.device]
    assert events == (["noise", "starts"] if backward_simulation is None else ["resolve"])
    assert outputs["gen_rand_train"] is state


def test_backward_simulation_pdd_dispatch_does_not_depend_on_grad_mode(monkeypatch):
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = _FakePDDNet(get_noise_schedule("rf"), pdd_steps=1)
    model.config = SimpleNamespace(
        backward_simulation="staggered",
        loss_config=SimpleNamespace(block_size_min=1, block_size_max=1),
    )
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    model.input_shape = [1]
    seen_grad_modes = []

    def resolve(*args, batch_size, **kwargs):
        del args, kwargs
        seen_grad_modes.append(torch.is_grad_enabled())
        return torch.zeros(batch_size, 1), 0

    def compute_loss(input_student, *args, **kwargs):
        del args, kwargs
        return {"total_loss": input_student.mean()}, input_student, None

    monkeypatch.setattr(pdd_module, "resolve_trajectory_point", resolve)
    monkeypatch.setattr(model, "_compute_loss", compute_loss)
    monkeypatch.setattr(model, "_get_outputs", lambda gen_data, **kwargs: {"gen_rand_train": gen_data})
    data = {"condition": None, "idx": [10, 11, 12]}

    model.single_train_step(data, iteration=1)

    with torch.no_grad():
        model.single_train_step(data, iteration=1)
    assert seen_grad_modes == [True, False]


def test_data_dependent_pdd_requires_real_data():
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(backward_simulation=None)

    with pytest.raises(KeyError, match="real"):
        model.single_train_step({"idx": [0]}, iteration=0)


def test_single_train_step_applies_a_configured_reward_to_the_clean_estimate(monkeypatch):
    """PDD reports rewards on ``x0_approx`` like the other trajectory-matching methods."""
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = _FakePDDNet(get_noise_schedule("rf"), pdd_steps=1)
    model.use_ema = []
    model.config = SimpleNamespace(
        backward_simulation=None,
        loss_config=SimpleNamespace(block_size_min=1, block_size_max=1),
        optimize_reward=False,
    )
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    rewarded = []

    def reward(gen_data, data, condition_dropout_mask=None):
        rewarded.append((gen_data, data))
        return {"total_loss": torch.tensor(0.25), "aesthetic": torch.tensor(0.5)}

    model.reward = reward
    real = torch.ones(2, 3)
    data = {"real": real, "idx": [0, 1]}

    def compute_loss(*args, **kwargs):
        del args, kwargs
        return {"total_loss": real.mean()}, real, None

    monkeypatch.setattr(model, "_compute_loss", compute_loss)

    loss_map, outputs = model.single_train_step(data, iteration=0)

    assert len(rewarded) == 1 and rewarded[0][1] is data
    torch.testing.assert_close(rewarded[0][0], real)
    torch.testing.assert_close(loss_map["reward_loss"], torch.tensor(0.25))
    torch.testing.assert_close(loss_map["aesthetic"], torch.tensor(0.5))
    # Metrics-only rewards leave the training objective untouched.
    torch.testing.assert_close(loss_map["total_loss"], torch.tensor(1.0))
    assert torch.equal(outputs["gen_rand_train"], real)


def test_data_dependent_noising_preserves_native_inputs_and_noised_state(
    monkeypatch,
):
    schedule = get_noise_schedule("rf", min_t=0.0, max_t=0.999)
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = _FakePDDNet(schedule, pdd_steps=4)
    model.config = SimpleNamespace(
        backward_simulation=None,
        loss_config=SimpleNamespace(block_size_min=1, block_size_max=1),
    )
    model.device = torch.device("cpu")
    model.precision = torch.bfloat16
    model.precision_decoding = torch.float32

    real_data = torch.randn(3, 2, dtype=torch.bfloat16)
    sampled_noise = torch.randn_like(real_data)
    observed = {}
    original_forward_process = schedule.forward_process

    def capture_forward_process(real, noise, t):
        observed["real"] = real
        observed["noise"] = noise
        observed["t_dtype"] = t.dtype
        observed["noised_state"] = original_forward_process(real, noise, t)
        return observed["noised_state"]

    def capture_loss(input_student, step_idx, base_grid, condition, neg_condition, next_idx):
        assert base_grid.shape == (model.net.pdd_steps + 1,)
        del step_idx, condition, neg_condition, next_idx
        observed["loss_state"] = input_student
        return {"total_loss": input_student.float().mean()}, input_student, None

    monkeypatch.setattr(schedule, "forward_process", capture_forward_process)
    monkeypatch.setattr(model, "_compute_loss", capture_loss)
    monkeypatch.setattr(model, "_get_outputs", lambda gen_data, **kwargs: {"gen_rand_train": gen_data})

    def fixed_noise(value, dtype=None):
        assert value is real_data and dtype is None
        return sampled_noise

    monkeypatch.setattr(pdd_module, "randn_like", fixed_noise)

    model.single_train_step({"real": real_data, "idx": [0, 1, 2]}, iteration=0)

    assert observed["real"] is real_data
    assert observed["real"].dtype == real_data.dtype
    assert observed["noise"] is sampled_noise
    assert observed["noise"].dtype == real_data.dtype
    assert observed["t_dtype"] == schedule.t_precision
    assert observed["loss_state"] is observed["noised_state"]
    assert observed["loss_state"].dtype == real_data.dtype


@pytest.mark.parametrize(
    ("pdd_steps", "stride", "sampled_candidates", "expected_starts"),
    [
        # The largest trained NFE, ceil(5 / 2) = 3, balances five base steps as [0, 1, 3, 5].
        (5, 2, [0, 1, 2], [0, 1, 3]),
        (3, 8, [0, 0], [0, 0]),
    ],
)
def test_data_dependent_starts_are_the_max_student_steps_grid_nodes(
    monkeypatch,
    pdd_steps,
    stride,
    sampled_candidates,
    expected_starts,
):
    schedule = get_noise_schedule("rf", min_t=0.0, max_t=0.999)
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = _FakePDDNet(schedule, pdd_steps=pdd_steps)
    model.config = SimpleNamespace(
        backward_simulation=None,
        loss_config=SimpleNamespace(block_size_min=stride, block_size_max=stride),
    )
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    observed = {}

    def deterministic_randint(high, size, device, dtype):
        assert high == model._max_student_steps()
        assert size == (len(sampled_candidates),)
        return torch.tensor(sampled_candidates, device=device, dtype=dtype)

    def capture_loss(input_student, step_idx, base_grid, condition, neg_condition, next_idx):
        assert base_grid.shape == (model.net.pdd_steps + 1,)
        # The x0 preview endpoint is the supervised block end on the base grid.
        assert torch.equal(
            next_idx,
            torch.clamp(step_idx + model.config.loss_config.block_size_max, max=model.net.pdd_steps),
        )
        del condition, neg_condition
        observed["step_idx"] = step_idx
        return {"total_loss": input_student.float().mean()}, input_student, None

    monkeypatch.setattr(pdd_module.torch, "randint", deterministic_randint)
    monkeypatch.setattr(pdd_module, "randn_like", lambda value, dtype=None: torch.ones_like(value))
    monkeypatch.setattr(model, "_compute_loss", capture_loss)
    monkeypatch.setattr(model, "_get_outputs", lambda gen_data, **kwargs: {"gen_rand_train": gen_data})
    real = torch.zeros(len(sampled_candidates), 1)

    model.single_train_step(
        {"real": real, "idx": list(range(len(sampled_candidates)))},
        iteration=0,
    )

    assert torch.equal(observed["step_idx"], torch.tensor(expected_starts))


@pytest.mark.parametrize("name", ["block_end_idx", "next_idx"])
def test_pdd_end_idx_validation_is_shared_by_sampling_and_output_advances(name):
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = SimpleNamespace(pdd_steps=4)
    step_idx = torch.tensor([0, 2])

    def run(end_idx):
        return model._validate_pdd_end_idx(step_idx, end_idx, name)

    run(torch.tensor([1, 4]))

    with pytest.raises(ValueError, match=rf"{name} must have shape"):
        run(torch.tensor([1]))
    with pytest.raises(ValueError, match=rf"step_idx < {name} <= pdd_steps"):
        run(torch.tensor([0, 3]))
    with pytest.raises(ValueError, match=rf"step_idx < {name} <= pdd_steps"):
        run(torch.tensor([1, 5]))


@pytest.mark.parametrize("multimodal", [False, True])
def test_pdd_loss_uses_one_detached_advance_for_the_sampled_state(multimodal, monkeypatch):
    schedule = get_noise_schedule("shifted_rf", min_t=0.0, max_t=0.999, shift=0.25)
    base_grid = torch.tensor([0.999, 0.87, 0.61, 0.29, 0.08, 0.0], dtype=torch.float32)
    batch, grid = 3, base_grid.numel() - 1
    net = _FakePDDNet(schedule, pdd_steps=grid)
    step_idx = torch.tensor([0, 1, 3])

    torch.manual_seed(4)
    x = torch.randn(batch, 2, 3, dtype=torch.float32)
    pred = torch.randn(batch, grid, 2, 3, dtype=torch.bfloat16, requires_grad=True)
    if multimodal:
        x = {"video": x, "audio": torch.randn(batch, 4, dtype=torch.float32)}
        pred = {
            "video": pred,
            "audio": torch.randn(batch, grid, 4, dtype=torch.bfloat16, requires_grad=True),
        }

    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.net = net
    model.config = SimpleNamespace(
        backward_simulation=True,
        loss_config=SimpleNamespace(
            block_size_min=1,
            block_size_max=3,
            type_rk="euler",
            num_supervised_heads=1,
        ),
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    sampled = torch.tensor([0, 2, 4])
    monkeypatch.setattr(model, "_sample_pdd_supervision_idx", lambda *args, **kwargs: sampled[:, None])

    advance_calls = []
    advance_block = net.pdd_advance_blocks

    def tracked_advance(*args, **kwargs):
        advance_calls.append((args[2], args[3], kwargs))
        return advance_block(*args, **kwargs)

    monkeypatch.setattr(net, "pdd_advance_blocks", tracked_advance)
    observed = {}

    def teacher_target(x_t, step_idx, base_grid, solver, predict_teacher):
        del step_idx, base_grid, solver, predict_teacher
        observed["local_state"] = x_t
        return per_modality(torch.zeros_like, x_t)

    monkeypatch.setattr(net, "pdd_teacher_target", teacher_target)

    loss_map, _ = model._compute_pdd_loss_from_prediction(
        input_student=x,
        step_idx=step_idx,
        pred=pred,
        base_grid=base_grid,
        condition=None,
        neg_condition=None,
    )

    grid_idx = torch.arange(grid)[None]
    head_mask = ((grid_idx >= step_idx[:, None]) & (grid_idx < sampled[:, None])).to(torch.float32)
    step_weights = _interval_weights(net, base_grid).to(torch.float32)
    weights = head_mask * step_weights[None]

    def reference(xi, pi):
        integration_x = xi.to(model.precision_decoding)
        integration_pred = pi.to(model.precision_decoding)
        prefix_total = weights.sum(dim=-1)
        prefix_alpha = (
            weights
            / torch.where(
                prefix_total != 0,
                prefix_total,
                torch.ones_like(prefix_total),
            )[:, None]
        )
        prefix_mean = torch.einsum("bn,bn...->b...", prefix_alpha, integration_pred)
        return integration_x + prefix_mean * expand_like(prefix_total, prefix_mean)

    if multimodal:
        expected_local_state = {key: reference(x[key], pred[key]) for key in x}
    else:
        expected_local_state = reference(x, pred)

    # The batched multi-endpoint einsum may reduce heads in a different order
    # than the single-block reference, so compare up to fp32 rounding.
    torch.testing.assert_close(observed["local_state"], expected_local_state, rtol=1e-4, atol=1e-6)
    assert len(advance_calls) == 1
    assert torch.equal(advance_calls[0][1], sampled[:, None])
    assert advance_calls[0][2]["allow_empty"] is True
    for value in observed["local_state"].values() if multimodal else (observed["local_state"],):
        assert value.dtype == model.precision
        assert not value.requires_grad
    loss_map["total_loss"].backward()
    for value in pred.values() if multimodal else (pred,):
        assert value.grad is not None


def test_pdd_supervision_sampler_draws_multiple_heads_within_each_block():
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = SimpleNamespace(pdd_steps=6)
    model.config = SimpleNamespace(loss_config=SimpleNamespace(block_size_max=4, num_supervised_heads=3))

    torch.manual_seed(0)
    step_idx = torch.tensor([0, 1])
    for _ in range(8):
        sampled = model._sample_pdd_supervision_idx(step_idx)
        assert sampled.shape == (2, 3)
        # Supervision targets are deterministic, so draws are unique per row.
        for row in range(2):
            assert sampled[row].unique().numel() == 3
            assert torch.all((sampled[row] >= step_idx[row]) & (sampled[row] <= step_idx[row] + 3))

    # Row 1 would hold only heads 4..5: too few for three unique draws.
    with pytest.raises(ValueError, match="num_supervised_heads=3 exceeds"):
        model._sample_pdd_supervision_idx(torch.tensor([0, 4]))

    model.config.loss_config.num_supervised_heads = 1
    assert model._sample_pdd_supervision_idx(step_idx).shape == (2, 1)


def test_single_head_supervision_preserves_reference_samples_and_rng_state():
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = SimpleNamespace(pdd_steps=6)
    model.config = SimpleNamespace(loss_config=SimpleNamespace(block_size_max=4, num_supervised_heads=1))
    step_idx = torch.tensor([0, 2, 4])
    head_idx = torch.arange(model.net.pdd_steps)[None]
    max_idx = torch.clamp(step_idx + model.config.loss_config.block_size_max - 1, max=model.net.pdd_steps - 1)
    block_mask = (head_idx >= step_idx[:, None]) & (head_idx <= max_idx[:, None])

    torch.manual_seed(123)
    expected = torch.multinomial(block_mask.float(), 1, replacement=False)
    expected_rng_state = torch.random.get_rng_state()

    torch.manual_seed(123)
    actual = model._sample_pdd_supervision_idx(step_idx)

    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), expected_rng_state)


def test_pdd_loss_averages_over_multiple_supervised_heads(monkeypatch):
    schedule = get_noise_schedule("shifted_rf", min_t=0.0, max_t=0.999, shift=0.25)
    base_grid = torch.tensor([0.999, 0.87, 0.61, 0.29, 0.08, 0.0], dtype=torch.float32)
    batch, grid = 2, base_grid.numel() - 1
    net = _FakePDDNet(schedule, pdd_steps=grid)
    step_idx = torch.tensor([0, 1])

    torch.manual_seed(7)
    x = torch.randn(batch, 2, 3, dtype=torch.float32)
    pred = torch.randn(batch, grid, 2, 3, dtype=torch.float32, requires_grad=True)

    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.net = net
    model.config = SimpleNamespace(
        backward_simulation=True,
        loss_config=SimpleNamespace(block_size_min=1, block_size_max=4, type_rk="euler", num_supervised_heads=2),
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    sampled = torch.tensor([[0, 2], [1, 3]])
    monkeypatch.setattr(model, "_sample_pdd_supervision_idx", lambda *args, **kwargs: sampled)
    monkeypatch.setattr(
        net,
        "pdd_teacher_target",
        lambda x_t, *args, **kwargs: per_modality(torch.zeros_like, x_t),
    )

    loss_map, next_state = model._compute_pdd_loss_from_prediction(
        input_student=x,
        step_idx=step_idx,
        pred=pred,
        base_grid=base_grid,
        condition=None,
        neg_condition=None,
    )
    assert next_state is None

    batch_idx = torch.arange(batch)
    loss_dtype = high_precision_dtype(pred)
    expected = (
        sum(
            torch.nn.functional.mse_loss(
                pred[batch_idx, sampled[:, head]].to(loss_dtype), torch.zeros(batch, 2, 3, dtype=loss_dtype)
            )
            for head in range(2)
        )
        / 2
    )
    torch.testing.assert_close(loss_map["total_loss"], expected)

    loss_map["total_loss"].backward()
    grad_heads = pred.grad.abs().flatten(2).sum(dim=-1) > 0
    expected_heads = torch.zeros(batch, grid, dtype=torch.bool)
    expected_heads[batch_idx[:, None], sampled] = True
    assert torch.equal(grad_heads, expected_heads)

    pred.grad = None
    loss_scale = torch.tensor([0.0, 1.0])
    masked_loss_map, _ = model._compute_pdd_loss_from_prediction(
        input_student=x,
        step_idx=step_idx,
        pred=pred,
        base_grid=base_grid,
        condition=None,
        neg_condition=None,
        loss_scale=loss_scale,
    )
    per_sample_head_losses = [
        pred[batch_idx, sampled[:, head]].to(loss_dtype).square().flatten(1).mean(dim=1) for head in range(2)
    ]
    expected_masked = sum((head_loss * loss_scale).mean() for head_loss in per_sample_head_losses) / 2
    torch.testing.assert_close(masked_loss_map["total_loss"], expected_masked)

    masked_loss_map["total_loss"].backward()
    assert torch.count_nonzero(pred.grad[0]) == 0
    assert torch.count_nonzero(pred.grad[1]) > 0


def test_pdd_multihead_loss_extracts_the_appended_next_state(monkeypatch):
    schedule = get_noise_schedule("shifted_rf", min_t=0.0, max_t=0.999, shift=0.25)
    base_grid = torch.tensor([0.999, 0.87, 0.61, 0.29, 0.08, 0.0], dtype=torch.float32)
    batch, grid = 2, base_grid.numel() - 1
    net = _FakePDDNet(schedule, pdd_steps=grid)
    step_idx = torch.tensor([0, 1])
    sampled = torch.tensor([[0, 2], [1, 3]])
    next_idx = torch.tensor([4, 5])
    x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    pred = torch.arange(batch * grid * 2, dtype=torch.float32).reshape(batch, grid, 2)

    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.net = net
    model.config = SimpleNamespace(
        backward_simulation=True,
        loss_config=SimpleNamespace(
            block_size_min=1,
            block_size_max=4,
            type_rk="euler",
            num_supervised_heads=2,
        ),
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    monkeypatch.setattr(model, "_sample_pdd_supervision_idx", lambda *args, **kwargs: sampled)
    monkeypatch.setattr(
        net,
        "pdd_teacher_target",
        lambda x_t, *args, **kwargs: per_modality(torch.zeros_like, x_t),
    )

    expected_next_state = net.pdd_advance_blocks(
        x,
        pred,
        step_idx,
        next_idx,
        base_grid,
        precision_decoding=model.precision_decoding,
    )
    advance = net.pdd_advance_blocks
    observed = {}

    def record_states(*args, **kwargs):
        observed["states"] = advance(*args, **kwargs)
        return observed["states"]

    monkeypatch.setattr(net, "pdd_advance_blocks", record_states)
    _, next_state = model._compute_pdd_loss_from_prediction(
        input_student=x,
        step_idx=step_idx,
        pred=pred,
        base_grid=base_grid,
        condition=None,
        neg_condition=None,
        next_idx=next_idx,
    )

    torch.testing.assert_close(next_state, expected_next_state, rtol=0, atol=0)
    assert next_state.untyped_storage().data_ptr() != observed["states"].untyped_storage().data_ptr()


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize(
    ("dtype", "precision_decoding"),
    [
        (torch.float32, None),
        (torch.float64, None),
        (torch.bfloat16, torch.float32),
    ],
)
def test_pdd_advance_blocks_matches_per_endpoint_single_advance(multimodal, dtype, precision_decoding):
    schedule = get_noise_schedule("shifted_rf", min_t=0.0, max_t=0.999, shift=0.25)
    grid = 8
    base_grid = torch.linspace(0.999, 0.0, grid + 1, dtype=torch.float32)
    net = _FakePDDNet(schedule, pdd_steps=grid)
    batch = 3

    torch.manual_seed(11)
    x = torch.randn(batch, 2, 3, dtype=dtype)
    pred = torch.randn(batch, grid, 2, 3, dtype=dtype)
    if multimodal:
        x = {"video": x, "audio": torch.randn(batch, 4, dtype=dtype)}
        pred = {"video": pred, "audio": torch.randn(batch, grid, 4, dtype=dtype)}

    for num_endpoints in (1, 2, 5):
        for allow_empty in (False, True):
            step_idx = torch.randint(0, grid, (batch,))
            low = step_idx if allow_empty else step_idx + 1
            columns = [
                (low + torch.floor(torch.rand(batch) * (grid + 1 - low)).long()).clamp(max=grid)
                for _ in range(num_endpoints)
            ]
            if allow_empty:
                # Force identity endpoints; random draws also cover duplicates
                # and unsorted columns.
                columns[0] = step_idx.clone()
            end_indices = torch.stack(columns, dim=1)

            batched = net.pdd_advance_blocks(
                x,
                pred,
                step_idx,
                end_indices,
                base_grid,
                allow_empty=allow_empty,
                precision_decoding=precision_decoding,
            )
            for value in batched.values() if multimodal else (batched,):
                assert value.dtype == dtype
                assert value.shape[:2] == (batch, num_endpoints)
            for column in range(num_endpoints):
                single = net.pdd_advance_blocks(
                    x,
                    pred,
                    step_idx,
                    end_indices[:, column],
                    base_grid,
                    allow_empty=allow_empty,
                    precision_decoding=precision_decoding,
                )
                torch.testing.assert_close(
                    per_modality(lambda value: value[:, column], batched),
                    single,
                    rtol=0,
                    atol=0,
                )


def test_pdd_advance_blocks_validates_endpoint_shape_and_ordering():
    schedule = get_noise_schedule("shifted_rf", min_t=0.0, max_t=0.999, shift=0.25)
    base_grid = torch.tensor([0.999, 0.61, 0.29, 0.0], dtype=torch.float32)
    grid = base_grid.numel() - 1
    net = _FakePDDNet(schedule, pdd_steps=grid)
    x = torch.randn(2, 2)
    pred = torch.randn(2, grid, 2)
    step_idx = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="num_endpoints"):
        net.pdd_advance_blocks(x, pred, step_idx, torch.tensor([[1], [2], [3]]), base_grid)
    with pytest.raises(ValueError, match="materialized heads"):
        net.pdd_advance_blocks(x, pred, step_idx, (step_idx + 1)[:, None], base_grid, fused=True)
    with pytest.raises(ValueError, match="0 <= start < end"):
        net.pdd_advance_blocks(x, pred, step_idx, step_idx[:, None], base_grid)
    with pytest.raises(ValueError, match="0 <= start <= end"):
        net.pdd_advance_blocks(x, pred, step_idx, torch.tensor([[1], [0]]), base_grid, allow_empty=True)


@pytest.mark.parametrize("multimodal", [False, True])
def test_pdd_loss_map_uses_the_same_named_pattern_for_tensor_and_dict(multimodal, monkeypatch):
    prediction = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    if multimodal:
        prediction = {"video": prediction, "audio": torch.tensor([[[2.0]], [[4.0]]])}
    target = per_modality(lambda value: torch.zeros_like(value[:, 0]), prediction)
    state = per_modality(torch.zeros_like, target)

    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.net = SimpleNamespace(
        pdd_steps=1,
        pdd_advance_blocks=lambda x_t, *args, **kwargs: per_modality(lambda value: value[:, None], x_t),
    )
    model.config = SimpleNamespace(
        loss_config=SimpleNamespace(block_size_max=1, type_rk="euler", num_supervised_heads=1)
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    monkeypatch.setattr(model.net, "pdd_teacher_target", lambda *args, **kwargs: target, raising=False)

    loss_map, _ = model._compute_pdd_loss_from_prediction(
        input_student=state,
        step_idx=torch.zeros(2, dtype=torch.long),
        pred=prediction,
        base_grid=torch.tensor([1.0, 0.0]),
        condition=None,
        neg_condition=None,
    )

    if multimodal:
        assert set(loss_map) == {"video_pdd_loss", "audio_pdd_loss", "total_loss"}
        expected_total = loss_map["video_pdd_loss"] + loss_map["audio_pdd_loss"]
    else:
        assert set(loss_map) == {"pdd_loss", "total_loss"}
        expected_total = loss_map["pdd_loss"]
    torch.testing.assert_close(loss_map["total_loss"], expected_total)


@pytest.mark.parametrize("multimodal", [False, True])
def test_data_dependent_single_interval_x0_preview_inverts_the_block_endpoint(multimodal, monkeypatch):
    base_grid = torch.tensor([0.999, 0.71, 0.32, 0.0], dtype=torch.float64)
    if multimodal:
        schedule = CompositeNoiseSchedule(
            schedules={
                "video": {"name": "shifted_rf", "shift": 4.0, "min_t": 0.0, "max_t": 0.999},
                "audio": {"name": "shifted_rf", "shift": 0.25, "min_t": 0.0, "max_t": 0.999},
            },
            primary_key="video",
        )
    else:
        schedule = get_noise_schedule("shifted_rf", shift=0.25, min_t=0.0, max_t=0.999)

    batch, steps = 3, len(base_grid) - 1
    state = torch.randn(batch, 2, 3, dtype=torch.float32)
    prediction = torch.randn(batch, steps, 2, 3, dtype=torch.bfloat16, requires_grad=True)
    if multimodal:
        state = {"video": state, "audio": torch.randn(batch, 4, dtype=torch.float32)}
        prediction = {
            "video": prediction,
            "audio": torch.randn(batch, steps, 4, dtype=torch.bfloat16, requires_grad=True),
        }

    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.net = _FakePDDNet(schedule, pdd_steps=steps)
    model.config = SimpleNamespace(
        backward_simulation=False,
        loss_config=SimpleNamespace(block_size_min=1, block_size_max=1, type_rk="euler", num_supervised_heads=1),
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    step_idx = torch.tensor([0, 1, 2])
    model.net.predictions = prediction
    model.net.student_grid_cfg_override = None

    monkeypatch.setattr(
        model.net,
        "pdd_teacher_target",
        lambda x_t, *args, **kwargs: per_modality(torch.zeros_like, x_t),
    )

    next_idx = step_idx + 1
    loss_map, preview, next_state = model._compute_loss(
        state,
        step_idx,
        base_grid,
        condition=None,
        neg_condition=None,
        next_idx=next_idx,
    )

    # The preview inverts the endpoint pair (state at step_idx, state at
    # next_idx) through the scheduler, exactly as x0_approx does.
    expected = schedule.points_to_x0(state, next_state, base_grid[step_idx], base_grid[next_idx])

    if multimodal:
        assert set(preview) == set(expected)
        for key in expected:
            torch.testing.assert_close(preview[key], expected[key], rtol=0, atol=0)
            assert not preview[key].requires_grad and preview[key].grad_fn is None
            assert not next_state[key].requires_grad
    else:
        torch.testing.assert_close(preview, expected, rtol=0, atol=0)
        assert not preview.requires_grad and preview.grad_fn is None
        assert not next_state.requires_grad
    loss_map["total_loss"].backward()
    for value in prediction.values() if multimodal else (prediction,):
        assert value.grad is not None


def test_pdd_compute_loss_uses_endpoint_pair_when_next_state_is_requested(monkeypatch):
    base_grid = torch.tensor([0.999, 0.72, 0.31, 0.0], dtype=torch.float64)
    net = _FakePDDNet(get_noise_schedule("shifted_rf", shift=0.25, min_t=0.0, max_t=0.999), base_grid=base_grid)
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = net
    model.config = SimpleNamespace(
        backward_simulation=False,
        loss_config=SimpleNamespace(block_size_min=1, block_size_max=2),
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float32

    step_idx = torch.tensor([0, 1])
    input_student = torch.tensor([[0.25, -0.5], [1.0, 0.75]], dtype=model.precision)
    next_idx = torch.tensor([2, 3])
    next_state = torch.tensor([[0.1, 0.7], [-0.6, 0.2]], dtype=model.precision)
    prediction = torch.randn(2, net.pdd_steps, 2)
    prediction.requires_grad_()

    net.predictions = prediction
    loss_call_kwargs = {}

    def fake_loss(**kwargs):
        loss_call_kwargs.update(kwargs)
        return {"total_loss": prediction.square().mean()}, next_state

    monkeypatch.setattr(model, "_compute_pdd_loss_from_prediction", fake_loss)
    schedule = net.noise_scheduler
    inversion_operands = []
    invert = schedule.points_to_x0

    def track_inversion(x_start, x_end, t_start, t_end):
        inversion_operands.append((x_start, x_end, t_start, t_end))
        return invert(x_start, x_end, t_start, t_end)

    monkeypatch.setattr(schedule, "points_to_x0", track_inversion)

    loss_map, x0_approx, actual_next = model._compute_loss(
        input_student,
        step_idx,
        base_grid,
        condition=None,
        neg_condition=None,
        next_idx=next_idx,
    )

    expected = invert(input_student, next_state, base_grid[step_idx], base_grid[next_idx])
    torch.testing.assert_close(x0_approx, expected)
    torch.testing.assert_close(actual_next, next_state)
    assert torch.equal(loss_call_kwargs["next_idx"], next_idx)
    # Both endpoints anchor the inversion in the state's own (model) precision;
    # the two-point inversion itself upcasts internally.
    assert len(inversion_operands) == 1
    x_start, x_end, t_start, t_end = inversion_operands[0]
    assert x_start.dtype == input_student.dtype
    assert x_end.dtype == input_student.dtype
    assert not x_start.requires_grad
    assert not x_end.requires_grad
    assert torch.equal(t_start, base_grid[step_idx])
    assert torch.equal(t_end, base_grid[next_idx])
    assert torch.equal(
        x0_approx[1],
        next_state[1],
    ), "a zero-sigma endpoint must be returned exactly"
    loss_map["total_loss"].backward()
    assert prediction.grad is not None


@pytest.mark.parametrize("multimodal", [False, True])
def test_rf_pdd_sampler_uses_summed_block_weight(multimodal):
    """Fused sampling advances each modality with its summed sigma-increment weight."""
    base_grid = torch.tensor([0.999, 0.77, 0.46, 0.19, 0.0], dtype=torch.float64)
    blocks = [1, 3]
    latents = torch.randn(2, 3)
    predictions = {0: torch.randn(2, 3), 1: torch.randn(2, 3)}
    if multimodal:
        latents = {"video": latents, "audio": torch.randn(2, 4)}
        predictions = {step: {"video": pred, "audio": torch.randn(2, 4)} for step, pred in predictions.items()}
        schedule = CompositeNoiseSchedule(
            schedules={
                "video": {"name": "rf", "min_t": 0.0, "max_t": 0.999},
                "audio": {"name": "shifted_rf", "shift": 0.25, "min_t": 0.0, "max_t": 0.999},
            },
            primary_key="video",
        )
    else:
        schedule = get_noise_schedule("shifted_rf", shift=3.0)

    net = _FakePDDNet(schedule, base_grid=base_grid, blocks=blocks)
    net.predictions = predictions
    model = object.__new__(_SamplerHarness)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        guidance_cfg=None,
        student_sample_type="ode",
        loss_config=SimpleNamespace(block_size_max=net.pdd_steps),
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    student_grid = net.noise_scheduler.student_grid(device=get_device(latents))
    actual = model._student_sample_loop(net, latents, student_grid=student_grid)

    expected = latents
    weights = _interval_weights(net, base_grid)
    start = 0
    for block in blocks:
        end = start + block
        if multimodal:
            expected = {
                key: expected[key] + weights[key][start:end].sum().to(expected[key].dtype) * predictions[start][key]
                for key in expected
            }
        else:
            expected = expected + weights[start:end].sum().to(expected.dtype) * predictions[start]
        start = end
    if multimodal:
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
    else:
        torch.testing.assert_close(actual, expected)


def test_pdd_student_sampler_does_not_reapply_teacher_guidance():
    """A PDD student already models the guided teacher target and gets one positive pass per block."""
    base_grid = torch.tensor([0.999, 0.7, 0.3, 0.0], dtype=torch.float64)
    blocks = [1, 2]
    latents = torch.randn(2, 3)
    predictions = {0: torch.randn_like(latents), 1: torch.randn_like(latents)}
    guidance_cfg = GuidanceConfig()
    guidance_cfg.cfg.scale = 7.0

    net = _FakePDDNet(get_noise_schedule("shifted_rf", shift=0.25), base_grid=base_grid, blocks=blocks)
    model = object.__new__(_SamplerHarness)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        guidance_cfg=guidance_cfg,
        student_sample_type="ode",
        loss_config=SimpleNamespace(block_size_max=net.pdd_steps),
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    net.predictions = predictions
    net.conditions = []
    student_grid = net.noise_scheduler.student_grid(device=latents.device)

    actual = model._student_sample_loop(
        net,
        latents,
        student_grid=student_grid,
        condition="positive",
        neg_condition="negative",
    )

    expected = latents
    weights = _interval_weights(net, base_grid)
    start = 0
    for block in blocks:
        end = start + block
        expected = expected + weights[start:end].sum().to(expected.dtype) * predictions[start]
        start = end
    torch.testing.assert_close(actual, expected)
    assert net.conditions == ["positive", "positive"]


def test_pdd_sampling_helpers_inherit_the_callers_autograd_mode():
    base_grid = torch.tensor([0.999, 0.5, 0.0], dtype=torch.float64)
    net = _FakePDDNet(get_noise_schedule("rf"), base_grid=base_grid, blocks=[1, 1])
    model = object.__new__(_SamplerHarness)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        student_sample_type="ode",
        loss_config=SimpleNamespace(block_size_max=net.pdd_steps),
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    scale = torch.nn.Parameter(torch.tensor(0.5))
    grad_modes = []

    def prediction(x, start, end, condition):
        del start, end, condition
        grad_modes.append(torch.is_grad_enabled())
        return scale.expand_as(x)

    net.predictions = prediction
    latents = torch.randn(2, 3)
    student_grid = net.noise_scheduler.student_grid(device=latents.device)
    output = model._student_sample_loop(net, latents, student_grid=student_grid)

    assert grad_modes == [True, True]
    assert output.requires_grad
    output.sum().backward()
    assert scale.grad is not None

    grad_modes.clear()
    with torch.no_grad():
        output = model._student_sample_loop(net, latents, student_grid=student_grid)
    assert grad_modes == [False, False]
    assert not output.requires_grad


@pytest.mark.parametrize("shift", [1.0, 5.0, 10.0])
def test_pdd_grid_applies_configured_shift(shift):
    """The PDD grid applies its exponential shift to the base schedule."""
    schedule = get_noise_schedule(
        "rf",
        min_t=0.0,
        max_t=0.999,
        student_grid_cfg=_native_grid(8, shift=shift),
    )
    x = torch.zeros(1, 2)

    actual = schedule.student_grid(device=x.device)
    base_grid = get_noise_schedule(
        "rf",
        min_t=0.0,
        max_t=0.999,
        student_grid_cfg=_native_grid(8),
    ).student_grid(device=x.device)
    expected = schedule.shift_t(base_grid, shift=shift)
    assert torch.equal(actual, expected)


def test_pdd_schedule_coordinate_helpers_use_the_configured_primary_and_leaf_schedules():
    composite = CompositeNoiseSchedule(
        schedules={
            "audio": {"name": "shifted_rf", "shift": 4.0, "min_t": 0.0, "max_t": 0.999},
            "video": {"name": "shifted_rf", "shift": 0.25, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )
    net = _FakePDDNet(composite, pdd_steps=4)
    primary = composite.subschedules["video"]
    assert net.pdd_primary_schedule is primary

    base_grid = torch.tensor([0.82, 0.53, 0.31, 0.17, 0.0], dtype=torch.float64)
    weights = _interval_weights(net, base_grid)
    for key, leaf in composite.subschedules.items():
        torch.testing.assert_close(
            weights[key],
            _sigma_delta(leaf, base_grid[:-1], base_grid[1:]),
            rtol=1e-12,
            atol=1e-15,
        )
    assert not torch.allclose(weights["audio"], weights["video"])


@pytest.mark.parametrize("shift", [1.0, 0.25])
@pytest.mark.parametrize("rk_type", ["euler", "midpoint", "heun", "kutta4"])
def test_teacher_rk_converts_flow_to_native_sigma_velocity_on_the_shared_clock(monkeypatch, rk_type, shift):
    """Stages sit at shared primary-clock times with native per-modality scaling.

    Samples whose interval ends on the clean endpoint blend to the solver's
    terminal fallback stage instead of supervising from a ``t = 0`` teacher
    evaluation.
    """
    model = object.__new__(_RKFlowHarness)
    torch.nn.Module.__init__(model)
    base_grid = torch.tensor([0.999, 0.72, 0.41, 0.18, 0.0], dtype=torch.float64)
    schedule = get_noise_schedule("shifted_rf", shift=shift, min_t=0.0, max_t=0.999)
    model.net = _FakePDDNet(schedule, pdd_steps=4, base_grid=base_grid)
    model.config = SimpleNamespace(loss_config=SimpleNamespace(type_rk=rk_type), guidance_cfg=None)
    model.teacher = FunctionNet(lambda xi, ti, **kwargs: xi * 0.125 + expand_like(ti, xi))
    model.precision = torch.float64
    model.precision_decoding = torch.float64
    x = torch.randn(2, 3, dtype=torch.float64)
    step_idx = torch.tensor([1, 3])
    next_step_idx = step_idx + 1
    t = base_grid[step_idx]
    t_next = base_grid[next_step_idx]
    delta_sigma = _sigma_delta(schedule, t, t_next)
    requested_grid_dtypes = []
    original_student_grid = model.net.noise_scheduler.student_grid

    def capture_student_grid(*args, **kwargs):
        requested_grid_dtypes.append(kwargs.get("dtype"))
        return original_student_grid(*args, **kwargs)

    monkeypatch.setattr(model.net.noise_scheduler, "student_grid", capture_student_grid)

    def teacher(xi, ti):
        return xi * 0.125 + expand_like(ti, xi)

    def sigma_velocity(xi, ti):
        flow = teacher(xi, ti)
        return flow / expand_like(schedule.sigma_prime(ti), flow)

    def stage(k, fraction):
        stage_delta_sigma = fraction * delta_sigma
        if fraction == 1.0:
            stage_time = t_next
        else:
            stage_time = _sigma_to_t(schedule, schedule.sigma(t) + stage_delta_sigma)
        stage_state = x + k * expand_like(stage_delta_sigma, k)
        return sigma_velocity(stage_state, stage_time)

    # ``step_idx[1]`` addresses the terminal interval, whose trailing stage
    # would evaluate the teacher at ``t = 0``.
    terminal = expand_like(t_next <= base_grid[-1], x).expand_as(x)
    if rk_type == "euler":
        expected = sigma_velocity(x, t)
    elif rk_type == "midpoint":
        k1 = sigma_velocity(x, t)
        expected = stage(k1, 0.5)
    elif rk_type == "heun":
        k1 = sigma_velocity(x, t)
        k2 = stage(k1, 1.0)
        expected = sum(c * value for c, value in zip([1 / 2, 1 / 2], [k1, k2]))
        expected = torch.where(terminal, k1, expected)
    else:
        k1 = sigma_velocity(x, t)
        k2 = stage(k1, 0.5)
        k3 = stage(k2, 0.5)
        k4 = stage(k3, 1.0)
        expected = sum(c * value for c, value in zip([1 / 6, 2 / 6, 2 / 6, 1 / 6], [k1, k2, k3, k4]))
        expected = torch.where(terminal, k2, expected)

    actual = _build_teacher_pdd_target(
        model,
        x,
        step_idx,
        base_grid,
        condition=None,
        neg_condition=None,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    assert requested_grid_dtypes == []


def test_teacher_rk_keeps_time_native_and_casts_only_state_update_operands(monkeypatch):
    model = object.__new__(_RKFlowHarness)
    torch.nn.Module.__init__(model)
    schedule = get_noise_schedule("shifted_rf", shift=0.25, min_t=0.0, max_t=0.999)
    model.net = _FakePDDNet(schedule, pdd_steps=4)
    model.config = SimpleNamespace(loss_config=SimpleNamespace(type_rk="midpoint"), guidance_cfg=None)
    model.precision = torch.bfloat16
    model.precision_decoding = torch.float32
    teacher_inputs = []
    update_dtypes = []
    unwarp_inputs = []
    original_advance = model.net.pdd_advance_prediction
    original_unwarp = schedule._unwarp

    def teacher(x_t, t, **kwargs):
        del kwargs
        teacher_inputs.append((x_t.dtype, t.dtype, t.detach().clone()))
        return torch.ones_like(x_t)

    def capture_advance(x_t, pred, weight):
        update_dtypes.append((x_t.dtype, pred.dtype, weight.dtype))
        return original_advance(x_t, pred, weight)

    def capture_unwarp(sigma):
        unwarp_inputs.append(sigma.detach().clone())
        return original_unwarp(sigma)

    model.teacher = FunctionNet(teacher)
    monkeypatch.setattr(model.net, "pdd_advance_prediction", capture_advance)
    monkeypatch.setattr(schedule, "_unwarp", capture_unwarp)
    x_t = torch.randn(2, 3, dtype=model.precision_decoding)
    t = torch.tensor([0.812345678901, 0.412345678901], dtype=model.net.noise_scheduler.t_precision)
    delta_t = torch.tensor([-0.223456789012, -0.123456789012], dtype=model.net.noise_scheduler.t_precision)
    t_next = t + delta_t
    base_grid = torch.stack([t[0], t_next[0], t[1], t_next[1], torch.zeros_like(t[0])])
    step_idx = torch.tensor([0, 2])
    delta_sigma = _sigma_delta(schedule, t, t_next)

    target = _build_teacher_pdd_target(
        model,
        x_t,
        step_idx,
        base_grid,
        condition=None,
        neg_condition=None,
    )

    assert [(state_dtype, time_dtype) for state_dtype, time_dtype, _ in teacher_inputs] == [
        (model.precision, t.dtype)
    ] * 2
    assert torch.equal(teacher_inputs[0][2], t)
    expected_midpoint_sigma = schedule.sigma(t) + schedule.determinant(t, t_next) * 0.5
    expected_midpoint_time = _sigma_to_t(schedule, schedule.sigma(t) + delta_sigma / 2)
    assert len(unwarp_inputs) == 1
    assert torch.equal(unwarp_inputs[0], expected_midpoint_sigma)
    assert torch.equal(teacher_inputs[1][2], expected_midpoint_time)
    assert not torch.equal(expected_midpoint_time, t + delta_t / 2)
    # The stage weight stays in scheduler precision; the advance casts it at
    # the state's arithmetic boundary.
    assert update_dtypes == [
        (model.precision_decoding, model.precision_decoding, model.net.noise_scheduler.t_precision)
    ]
    assert target.dtype == model.precision_decoding
    # The flow-to-sigma-velocity conversion divides in high precision and returns the decoding dtype.
    midpoint_sigma_prime = expand_like(schedule.sigma_prime(expected_midpoint_time), target)
    expected = (torch.ones_like(target, dtype=torch.float64) / midpoint_sigma_prime).to(target.dtype)
    assert torch.equal(target, expected)


def test_teacher_target_converts_flow_to_sigma_velocity_in_high_precision():
    """The flow-to-``dx/dsigma`` division runs in high precision and returns the decoding dtype."""
    model = object.__new__(_RKFlowHarness)
    torch.nn.Module.__init__(model)
    schedule = get_noise_schedule("shifted_rf", shift=0.25, min_t=0.0, max_t=0.999)
    base_grid = torch.tensor([0.91, 0.63, 0.22, 0.0], dtype=torch.float64)
    model.net = _FakePDDNet(schedule, pdd_steps=3, base_grid=base_grid)
    model.config = SimpleNamespace(loss_config=SimpleNamespace(type_rk="euler"), guidance_cfg=None)
    flow = torch.randn(2, 256, generator=torch.Generator().manual_seed(0))
    model.teacher = FunctionNet(lambda xi, ti, **kwargs: flow)
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    step_idx = torch.tensor([0, 1])

    target = _build_teacher_pdd_target(model, torch.zeros_like(flow), step_idx, base_grid)

    sigma_prime = expand_like(schedule.sigma_prime(base_grid[step_idx]), flow)
    assert target.dtype == model.precision_decoding
    assert torch.equal(target, (flow.double() / sigma_prime).float())
    # A float32 division by the float32-rounded sigma' rounds some of these elements differently.
    assert not torch.equal(target, flow / sigma_prime.float())


def test_warped_rf_euler_target_is_the_schedule_native_teacher_sigma_velocity():
    """One-call Euler freezes the native RF direction ``dx/dsigma``."""
    model = object.__new__(_RKFlowHarness)
    torch.nn.Module.__init__(model)
    schedule = get_noise_schedule("shifted_rf", shift=0.25, min_t=0.0, max_t=0.999)
    base_grid = torch.tensor([0.91, 0.63, 0.22, 0.0], dtype=torch.float64)
    model.net = _FakePDDNet(
        schedule,
        pdd_steps=3,
        base_grid=base_grid,
    )
    model.config = SimpleNamespace(loss_config=SimpleNamespace(type_rk="euler"), guidance_cfg=None)
    model.teacher = FunctionNet(lambda xi, ti, **kwargs: xi * 0.125 + expand_like(ti, xi))
    model.precision = torch.float64
    model.precision_decoding = torch.float64
    x = torch.randn(2, 3, dtype=torch.float64)
    step_idx = torch.tensor([0, 1])
    t = base_grid[step_idx]
    weight = _interval_weights(model.net, base_grid)[:2]

    teacher_sigma_velocity = model.teacher(x, t) / expand_like(schedule.sigma_prime(t), x)
    target = _build_teacher_pdd_target(
        model,
        x,
        step_idx,
        base_grid,
        condition=None,
        neg_condition=None,
    )
    torch.testing.assert_close(target, teacher_sigma_velocity, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        model.net.pdd_advance_prediction(x, target, weight),
        x + expand_like(weight, x) * teacher_sigma_velocity,
        rtol=1e-12,
        atol=1e-12,
    )


def test_composite_uses_primary_student_grid_cfg():
    student_grid_cfg = _explicit_grid([0.999, 0.75, 0.0], blocks=[1, 1])
    composite = CompositeNoiseSchedule(
        schedules={
            "video": {
                "name": "rf",
                "min_t": 0.0,
                "max_t": 0.999,
                "student_grid_cfg": student_grid_cfg,
            },
            "audio": {"name": "shifted_rf", "shift": 0.25, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )

    assert composite.student_grid_cfg is student_grid_cfg
    assert composite.student_grid_cfg is composite.primary.student_grid_cfg


def test_pdd_block_weights_are_per_modality_and_other_contracts_are_rejected():
    composite = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "rf", "min_t": 0.0, "max_t": 0.999},
            "audio": {"name": "shifted_rf", "shift": 0.25, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )
    base_grid = torch.tensor([0.999, 0.75, 0.31, 0.0])
    net = _FakePDDNet(composite, base_grid=base_grid)
    intervals = torch.arange(net.pdd_steps)
    _, weights = net.pdd_block_weights(intervals, intervals + 1, base_grid)
    assert set(weights) == {"video", "audio"}
    assert torch.equal(weights["video"], torch.diff(base_grid))
    audio = composite.subschedules["audio"]
    expected_audio = audio.sigma(base_grid[1:]) - audio.sigma(base_grid[:-1])
    torch.testing.assert_close(weights["audio"], expected_audio)
    assert not torch.allclose(weights["audio"], weights["video"])

    net._validate_pdd_schedule_contract()
    with pytest.raises(ValueError, match="net_pred_type='flow'"):
        schedule = get_noise_schedule("rf")
        _FakePDDNet(schedule, "x0", pdd_steps=4)._validate_pdd_schedule_contract()
    with pytest.raises(ValueError, match="requires an 'rf' or 'shifted_rf' schedule"):
        schedule = get_noise_schedule("trig")
        _FakePDDNet(schedule, pdd_steps=4)._validate_pdd_schedule_contract()


@pytest.mark.parametrize("min_t", [1e-6, 0.1])
def test_pdd_schedule_contract_requires_zero_min_t(min_t):
    schedule = get_noise_schedule("rf", min_t=min_t, max_t=0.999)

    with pytest.raises(ValueError, match="min_t=0"):
        _FakePDDNet(schedule, pdd_steps=4)._validate_pdd_schedule_contract()


def _masked_block_weights(net, step_index, next_step_index, base_grid):
    """Each row's unnormalized interval weights, zero outside its block ``[step_index, next_step_index)``."""
    grid_index = torch.arange(net.pdd_steps)
    active = (grid_index >= step_index[:, None]) & (grid_index < next_step_index[:, None])
    return per_modality(
        lambda weights: torch.where(active, weights, torch.zeros_like(weights)),
        _interval_weights(net, base_grid),
    )


def test_pdd_block_weights_normalize_by_the_telescoped_total_for_shared_batched_and_multimodal_blocks():
    base_grid = torch.tensor([0.999, 0.75, 0.31, 0.0], dtype=torch.float64)
    composite = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "rf", "min_t": 0.0, "max_t": 0.999},
            "audio": {"name": "shifted_rf", "shift": 0.25, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )
    net = _FakePDDNet(composite, base_grid=base_grid)
    weights = _interval_weights(net, base_grid)

    shared_weights, shared = net.pdd_block_weights(1, 3, base_grid)
    starts = torch.tensor([0, 1])
    ends = torch.tensor([2, 3])
    batched_weights, batched = net.pdd_block_weights(starts, ends, base_grid)

    for key, modality_weights in weights.items():
        schedule = composite.subschedules[key]
        assert torch.equal(shared[key], schedule.determinant(base_grid[1], base_grid[3]))
        torch.testing.assert_close(shared[key], modality_weights[1:3].sum())
        expected_batched = torch.stack([modality_weights[0:2].sum(), modality_weights[1:3].sum()])
        torch.testing.assert_close(batched[key], expected_batched)

        expected_shared_weights = torch.cat([torch.zeros(1, dtype=torch.float64), modality_weights[1:3] / shared[key]])
        assert torch.equal(shared_weights[key], expected_shared_weights)
        torch.testing.assert_close(shared_weights[key].sum(), torch.ones((), dtype=torch.float64))
        torch.testing.assert_close(batched_weights[key].sum(dim=-1), torch.ones(2, dtype=torch.float64))

    endpoint_delta = base_grid[3] - base_grid[1]
    assert torch.equal(shared["video"], endpoint_delta)
    assert not torch.isclose(shared["audio"], endpoint_delta)


def test_pdd_advance_blocks_matches_fused_multimodal_head_reduction():
    base_grid = torch.tensor([0.999, 0.75, 0.31, 0.0], dtype=torch.float64)
    schedule = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "shifted_rf", "shift": 4.0, "min_t": 0.0, "max_t": 0.999},
            "audio": {"name": "shifted_rf", "shift": 0.25, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )
    net = _FakePDDNet(schedule, base_grid=base_grid)
    starts = torch.tensor([0, 1])
    ends = torch.tensor([2, 3])
    pred = {
        "video": torch.tensor(
            [
                [[0.4, -0.2], [0.1, 0.3], [9.0, 9.0]],
                [[9.0, 9.0], [-0.3, 0.6], [0.2, -0.1]],
            ]
        ),
        "audio": torch.tensor(
            [
                [[0.7], [0.1], [9.0]],
                [[9.0], [-0.5], [0.2]],
            ]
        ),
    }

    state = per_modality(lambda value: torch.zeros_like(value[:, 0]), pred)
    actual_state = net.pdd_advance_blocks(state, pred, starts, ends, base_grid)
    block_weights = _masked_block_weights(net, starts, ends, base_grid)

    for key in pred:
        normalized_weights = block_weights[key] / block_weights[key].sum(dim=-1, keepdim=True)
        expected_mean = torch.einsum("bn,bn...->b...", normalized_weights.float(), pred[key])
        total_weight = block_weights[key].sum(dim=-1).to(state[key])
        expected_state = net.pdd_advance_prediction(state[key], expected_mean, total_weight)
        torch.testing.assert_close(actual_state[key], expected_state)


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 1), (2, 1), (0, 4)])
def test_pdd_block_indices_reject_invalid_shared_half_open_blocks(start, end):
    base_grid = torch.tensor([0.999, 0.6, 0.2, 0.0], dtype=torch.float64)
    net = _FakePDDNet(get_noise_schedule("rf"), pdd_steps=3, base_grid=base_grid)

    with pytest.raises(ValueError, match="0 <= step_index < next_step_index <= 3"):
        net._validate_pdd_indices(start, end)


def test_pdd_block_indices_reject_malformed_batched_indices_with_explicit_errors():
    """Structural checks on batched indices raise real exceptions rather than bare asserts."""
    base_grid = torch.tensor([0.999, 0.6, 0.2, 0.0], dtype=torch.float64)
    net = _FakePDDNet(get_noise_schedule("rf"), pdd_steps=3, base_grid=base_grid)
    starts = torch.tensor([0, 1])

    with pytest.raises(TypeError, match="must be long tensors"):
        net._validate_pdd_indices(starts, torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match=r"must have shape \(2,\)"):
        net._validate_pdd_indices(starts, torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError, match=r"must have shape \(4,\)"):
        net._validate_pdd_indices(starts, batch_size=4)
    # Well-formed batched indices pass the structural checks.
    net._validate_pdd_indices(starts, torch.tensor([1, 2]), batch_size=2, device=starts.device)


@pytest.mark.parametrize(
    "starts,ends",
    [
        (torch.tensor([-1, 0]), torch.tensor([1, 2])),
        (torch.tensor([0, 1]), torch.tensor([0, 2])),
        (torch.tensor([2, 1]), torch.tensor([1, 2])),
        (torch.tensor([0, 1]), torch.tensor([2, 4])),
    ],
)
def test_pdd_block_weights_reject_invalid_batched_half_open_blocks(starts, ends):
    base_grid = torch.tensor([0.999, 0.6, 0.2, 0.0], dtype=torch.float64)
    net = _FakePDDNet(get_noise_schedule("rf"), pdd_steps=3, base_grid=base_grid)

    with pytest.raises(ValueError, match="0 <= start < end <= 3"):
        net.pdd_block_weights(starts, ends, base_grid)


def test_pdd_block_weights_reject_mixed_int_and_tensor_endpoints():
    """A mixed pair fails the shared type check instead of tripping the multi-endpoint shape check."""
    base_grid = torch.tensor([0.999, 0.6, 0.2, 0.0], dtype=torch.float64)
    net = _FakePDDNet(get_noise_schedule("rf"), pdd_steps=3, base_grid=base_grid)
    endpoints = torch.tensor([[1, 2], [2, 3]])

    with pytest.raises(TypeError, match="both be integers or both be batched tensors"):
        net.pdd_block_weights(0, endpoints, base_grid)
    with pytest.raises(TypeError, match="both be integers or both be batched tensors"):
        net.pdd_block_weights(torch.tensor([0, 1]), 3, base_grid)


def test_pdd_block_weights_reject_zero_length_intervals():
    net = _FakePDDNet(get_noise_schedule("shifted_rf", shift=0.25), pdd_steps=1)

    with pytest.raises(ValueError, match="integration weight must be finite and non-zero"):
        net.pdd_block_weights(0, 1, torch.tensor([0.5, 0.5], dtype=torch.float64))


@pytest.mark.parametrize("shift", [0.25, 4.0])
def test_rf_determinant_preserves_adjacent_float32_intervals(shift):
    schedule = get_noise_schedule("shifted_rf", shift=shift, min_t=0.0, max_t=0.999)
    t_end = torch.tensor([0.5], dtype=torch.float32)
    t_start = torch.nextafter(t_end, torch.ones_like(t_end))

    actual = schedule.determinant(t_start, t_end)
    expected = (schedule.sigma(t_end.double()) - schedule.sigma(t_start.double())).float()

    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) == actual.numel()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=0)


def test_pdd_advance_blocks_normalizes_in_scheduler_precision():
    base_grid = torch.tensor([0.999, 0.7, 0.2, 0.0], dtype=torch.float64)
    net = _FakePDDNet(get_noise_schedule("shifted_rf", shift=0.25), base_grid=base_grid)
    step_index = torch.tensor([0, 1])
    next_step_index = torch.tensor([2, 3])
    pred = torch.randn(2, 3, 1, dtype=torch.float32)

    weights = _masked_block_weights(net, step_index, next_step_index, base_grid)
    normalized_weights = weights / weights.sum(dim=-1, keepdim=True)
    expected = torch.einsum("bn,bn...->b...", normalized_weights.float(), pred)
    state = torch.zeros_like(pred[:, 0])
    actual_state = net.pdd_advance_blocks(state, pred, step_index, next_step_index, base_grid)

    block_weights, block_total = net.pdd_block_weights(step_index, next_step_index, base_grid)
    assert block_weights.dtype == block_total.dtype == base_grid.dtype
    assert actual_state.dtype == state.dtype
    torch.testing.assert_close(actual_state, expected * weights.sum(dim=-1, keepdim=True).float())

    with pytest.raises(AssertionError, match="state and prediction must have the same dtype"):
        net.pdd_advance_blocks(state.double(), pred, step_index, next_step_index, base_grid)


def test_pdd_block_weights_are_bitwise_the_same_for_shared_batched_and_multi_endpoint_blocks():
    """Sums of a slice and of a masked row can round differently; the telescoped total cannot."""
    base_grid = torch.tensor([0.999, 0.91, 0.64, 0.37, 0.12, 0.0], dtype=torch.float64)
    composite = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "rf", "min_t": 0.0, "max_t": 0.999},
            "audio": {"name": "shifted_rf", "shift": 0.25, "min_t": 0.0, "max_t": 0.999},
        },
        primary_key="video",
    )
    net = _FakePDDNet(composite, base_grid=base_grid)
    blocks = [(start, end) for start in range(net.pdd_steps) for end in range(start, net.pdd_steps + 1)]
    starts = torch.tensor([start for start, _ in blocks])
    ends = torch.tensor([end for _, end in blocks])

    batched_weights, batched = net.pdd_block_weights(starts, ends, base_grid, allow_empty=True)
    endpoints = torch.stack([ends, starts], dim=1)
    multi_endpoint_weights, multi_endpoint = net.pdd_block_weights(starts, endpoints, base_grid, allow_empty=True)

    for row, (start, end) in enumerate(blocks):
        shared_weights, shared = net.pdd_block_weights(start, end, base_grid, allow_empty=True)
        for key in shared:
            assert torch.equal(batched[key][row], shared[key])
            assert torch.equal(multi_endpoint[key][row, 0], shared[key])
            assert multi_endpoint[key][row, 1] == 0
            assert torch.equal(batched_weights[key][row], shared_weights[key])
            assert torch.equal(multi_endpoint_weights[key][row, 0], shared_weights[key])
            assert not multi_endpoint_weights[key][row, 1].any()


@pytest.mark.parametrize("schedule_shift", [1.0, 0.25])
def test_fused_and_materialized_pdd_paths_share_the_block_weights_and_total(monkeypatch, schedule_shift):
    """Unit heads make the materialized mean the weights themselves, since adding zeros is exact."""
    net = _toy_pdd(pdd_steps=8, schedule_shift=schedule_shift)
    base_grid = net.pdd_base_grid(device=torch.device("cpu"))
    fused_weights, advances = [], []
    original_fuse, original_advance = net.pdd_linear.fuse, net.pdd_advance_prediction

    def capture_fuse(step_index, next_step_index, weights, precision_decoding):
        fused_weights.append(weights)
        return original_fuse(step_index, next_step_index, weights, precision_decoding)

    def capture_advance(x_t, prediction, weight):
        advances.append((prediction, weight))
        return original_advance(x_t, prediction, weight)

    monkeypatch.setattr(net.pdd_linear, "fuse", capture_fuse)
    monkeypatch.setattr(net, "pdd_advance_prediction", capture_advance)

    shared_blocks = [(start, end) for start in range(net.pdd_steps) for end in range(start + 1, net.pdd_steps + 1)]
    batched_blocks = [(torch.tensor([0, 2, 5]), torch.tensor([3, 8, 6]))]
    for start, end in shared_blocks + batched_blocks:
        batch = 1 if isinstance(start, int) else len(start)
        unit_heads = torch.eye(net.pdd_steps, dtype=torch.float64).expand(batch, -1, -1)
        state = torch.zeros(batch, net.pdd_steps, dtype=torch.float64)

        net(torch.zeros(batch, 1), start, fused=True, next_step_index=end)
        net.pdd_advance_blocks(state, torch.zeros_like(state), start, end, base_grid, fused=True)
        net.pdd_advance_blocks(state, unit_heads, start, end, base_grid)
        (_, fused_total), (materialized_weights, materialized_total) = advances[-2:]

        weights = fused_weights[-1]
        if isinstance(start, int):
            weights, materialized_weights = weights[start:end], materialized_weights[0, start:end]
        assert torch.equal(weights, materialized_weights)
        assert torch.equal(fused_total, materialized_total)
        assert torch.equal(fused_total, net.pdd_block_weights(start, end, base_grid)[1])


def test_fused_and_materialized_pdd_advances_are_bitwise_equal_in_exact_arithmetic():
    """With every intermediate exactly representable, any difference is a different formula, not rounding.

    ``max_t = 0.5`` on four steps puts the grid on multiples of 1/8, so blocks of 1, 2 or 4 steps have
    normalized weights 1, 1/2 or 1/4; the heads' parameters and the inputs are small dyadic numbers.
    """
    net = _ToyPDD(
        pdd_steps=4,
        net_pred_type="flow",
        noise_scheduler=get_noise_schedule("rf", min_t=0.0, max_t=0.5),
        student_grid_cfg=_native_grid(4),
    ).double()
    with torch.no_grad():
        net.pdd_linear.weight.copy_(torch.tensor([[1.0], [-2.0], [3.0], [0.5]]))
        net.pdd_linear.bias.copy_(torch.tensor([0.25, -1.0, 2.0, -0.5]))
    base_grid = net.pdd_base_grid(device=torch.device("cpu"))
    assert torch.equal(base_grid, torch.tensor([0.5, 0.375, 0.25, 0.125, 0.0], dtype=torch.float64))
    x_t = torch.tensor([[-2.0], [1.0], [3.0]], dtype=torch.float64)

    shared_blocks = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4), (0, 4)]
    batched_blocks = [
        (torch.tensor([0, 1, 2]), torch.tensor([2, 3, 4])),
        (torch.tensor([0, 3, 1]), torch.tensor([4, 4, 2])),
    ]
    for start, end in shared_blocks + batched_blocks:
        heads = net(x_t, start)
        materialized = net.pdd_advance_blocks(x_t, heads, start, end, base_grid)
        mean_sigma_velocity = net(x_t, start, fused=True, next_step_index=end, precision_decoding=torch.float64)
        fused = net.pdd_advance_blocks(x_t, mean_sigma_velocity, start, end, base_grid, fused=True)

        assert not torch.equal(fused, x_t)
        assert torch.equal(fused, materialized)


def test_pdd_shared_materialized_advance_ignores_heads_outside_the_direct_slice():
    base_grid = torch.tensor([0.999, 0.8, 0.5, 0.2, 0.0], dtype=torch.float64)
    net = _FakePDDNet(get_noise_schedule("shifted_rf", shift=0.25), base_grid=base_grid)
    start, end = 1, 3
    pred = torch.tensor([[[torch.nan], [2.0], [4.0], [torch.nan]]])
    state = torch.zeros(1, 1)

    weights = _interval_weights(net, base_grid)[start:end]
    expected_state = state + torch.einsum(
        "n,bn...->b...",
        weights.to(dtype=pred.dtype),
        pred[:, start:end],
    )
    actual_state = net.pdd_advance_blocks(state, pred, start, end, base_grid)

    assert torch.isfinite(actual_state).all()
    torch.testing.assert_close(actual_state, expected_state)


@pytest.mark.parametrize("weight", [2.0**-30, 2.0**20])
@pytest.mark.parametrize("batched", [False, True])
def test_pdd_advance_rejects_nonempty_total_lost_when_cast_to_state_dtype(
    monkeypatch,
    weight,
    batched,
):
    """A finite scheduler total must not silently underflow or overflow in FP16 integration."""
    base_grid = torch.tensor([0.999, 0.0], dtype=torch.float64)
    net = _FakePDDNet(get_noise_schedule("rf"), pdd_steps=1, base_grid=base_grid)

    original_block_weights = net.pdd_block_weights

    def block_weights(step_index, next_step_index, grid, allow_empty=False):
        weights, total = original_block_weights(step_index, next_step_index, grid, allow_empty=allow_empty)
        return weights, torch.where(total != 0, -weight, 0.0).to(torch.float64)

    monkeypatch.setattr(net, "pdd_block_weights", block_weights)

    if batched:
        step_index = torch.tensor([0, 0])
        next_step_index = torch.tensor([0, 1])
        state = torch.zeros(2, 1, dtype=torch.float16)
    else:
        step_index, next_step_index = 0, 1
        state = torch.zeros(1, 1, dtype=torch.float16)
    block_prediction = torch.ones_like(state)

    with pytest.raises(ValueError, match="block scale in state dtype must be finite and non-zero"):
        net.pdd_advance_blocks(
            state,
            block_prediction,
            step_index,
            next_step_index,
            base_grid,
            fused=True,
            allow_empty=batched,
        )


def test_pdd_advance_blocks_uses_one_sided_sigma_velocity_for_mixed_empty_prefixes():
    base_grid = torch.tensor([0.999, 0.7, 0.2, 0.0], dtype=torch.float64)
    net = _FakePDDNet(get_noise_schedule("shifted_rf", shift=0.25), base_grid=base_grid)
    step_index = torch.tensor([0, 1])
    next_step_index = torch.tensor([0, 3])
    state = torch.randn(2, 1, dtype=torch.float32)
    pred = torch.randn(2, 3, 1, dtype=torch.float32)
    block_weights = _masked_block_weights(net, step_index, next_step_index, base_grid)

    actual_state = net.pdd_advance_blocks(
        state,
        pred,
        step_index,
        next_step_index,
        base_grid,
        allow_empty=True,
    )

    assert torch.equal(actual_state[0], state[0])
    torch.testing.assert_close(
        actual_state[1],
        state[1] + torch.einsum("n,n...->...", block_weights[1].float(), pred[1]),
    )


@pytest.mark.parametrize("layout,patch", [("channel_major", 1), ("patch_major", 4)])
def test_pdd_linear_sigma_velocity_fusion_uses_normalized_interval_weights(layout, patch):
    """Final-layer mixing uses the caller's warped-RF interval weights, normalized by the block total."""
    torch.manual_seed(2)
    grid, channels, in_features = 5, 3, 7
    linear = PDDLinear(
        in_features,
        channels * patch,
        patch_size=patch,
        grid_size=grid,
        channel_layout=layout,
    ).float()
    base_grid = torch.tensor([0.999, 0.91, 0.64, 0.37, 0.12, 0.0], dtype=torch.float64)
    start, end = 1, 5
    net = _FakePDDNet(get_noise_schedule("shifted_rf", shift=0.25), base_grid=base_grid)
    weights, _ = net.pdd_block_weights(start, end, base_grid)

    actual_weight, actual_bias = linear._fused_params(start, end, weights, torch.float32)
    if layout == "patch_major":
        wide_weight = rearrange(linear.weight, "(p n c) ... -> n (p c) ...", p=patch, n=grid)[start:end]
        wide_bias = rearrange(linear.bias, "(p n c) -> n (p c)", p=patch, n=grid)[start:end]
    else:
        wide_weight = rearrange(linear.weight, "(n rest) ... -> n rest ...", n=grid)[start:end]
        wide_bias = rearrange(linear.bias, "(n rest) -> n rest", n=grid)[start:end]
    block_weights = weights[start:end].float()
    expected_weight = torch.einsum("n,n... -> ...", block_weights, wide_weight)
    expected_bias = torch.einsum("n,n... -> ...", block_weights, wide_bias)
    assert torch.equal(actual_weight, expected_weight)
    assert torch.equal(actual_bias, expected_bias)


@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
def test_pdd_head_reductions_keep_decoding_precision_under_autocast(amp_dtype):
    """Autocast must not lower the FP32 head reductions of block advances or fused parameters."""
    net = _FakePDDNet(get_noise_schedule("shifted_rf", shift=5.0, min_t=0.0, max_t=0.999), pdd_steps=16)
    base_grid = net.pdd_base_grid()
    torch.manual_seed(0)
    state = torch.randn(2, 3, 4)
    heads = torch.randn(2, 16, 3, 4)
    starts, ends = torch.tensor([0, 4]), torch.tensor([8, 16])
    for start, end in [(4, 12), (starts, ends), (starts, torch.stack([ends, starts + 4], dim=1))]:
        expected = net.pdd_advance_blocks(
            state.double(), heads.double(), start, end, base_grid, precision_decoding=torch.float64
        )
        with torch.autocast("cpu", dtype=amp_dtype):
            actual = net.pdd_advance_blocks(state, heads, start, end, base_grid, precision_decoding=torch.float32)
        torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)

    linear = PDDLinear(5, 3, patch_size=1, grid_size=16, channel_layout="channel_major")
    reference = copy.deepcopy(linear).double()
    for start, end in [(4, 12), (starts, ends)]:
        weights, _ = net.pdd_block_weights(start, end, base_grid)
        with torch.no_grad():
            expected = reference._fused_params(start, end, weights, torch.float64)
            with torch.autocast("cpu", dtype=amp_dtype):
                actual = linear._fused_params(start, end, weights, torch.float32)
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value.double(), expected_value, rtol=1e-6, atol=1e-6)


def test_fused_pdd_block_constructs_coefficients_in_scheduler_precision(monkeypatch):
    """The base grid, integration weights and their normalization stay FP64 until final-layer fusion."""
    net = _toy_pdd(pdd_steps=64, steps=4)
    captured = {}
    original_fused_params = net.pdd_linear._fused_params

    def capture_fused_params(step_index, next_step_index, weights, compute_dtype):
        captured["weights"] = weights.detach().clone()
        captured["compute_dtype"] = compute_dtype
        return original_fused_params(step_index, next_step_index, weights, compute_dtype)

    monkeypatch.setattr(net.pdd_linear, "_fused_params", capture_fused_params)
    state = torch.zeros(1, 1, dtype=torch.float32)
    net(state, 0, fused=True, next_step_index=16)

    weights = captured["weights"]
    native_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=state.device,
        dtype=net.noise_scheduler.t_precision,
    )
    expected, _ = net.pdd_block_weights(0, 16, native_grid)
    assert weights.dtype == torch.float64
    assert torch.equal(weights, expected)
    assert captured["compute_dtype"] == state.dtype

    alpha64 = weights[:16].float()
    alpha32 = net.pdd_block_weights(0, 16, native_grid.float())[0][:16]
    assert not torch.equal(alpha64, alpha32)


@pytest.mark.parametrize("layout,patch", [("channel_major", 1), ("patch_major", 3)])
def test_pdd_linear_repeats_teacher_parameters_for_both_layouts(layout, patch):
    grid, channels, in_features = 3, 2, 4
    teacher = torch.nn.Linear(in_features, channels * patch)
    with torch.no_grad():
        teacher.weight.copy_(torch.arange(teacher.weight.numel()).reshape_as(teacher.weight))
        teacher.bias.copy_(torch.arange(teacher.bias.numel()) + 1)

    linear = PDDLinear(
        in_features,
        channels * patch,
        patch_size=patch,
        grid_size=grid,
        channel_layout=layout,
    )
    linear.load_state_dict(teacher.state_dict())

    if layout == "patch_major":
        expected_weight = rearrange(teacher.weight, "(p c) i -> p c i", p=patch)
        expected_weight = rearrange(expected_weight[:, None].expand(-1, grid, -1, -1), "p n c i -> (p n c) i")
        expected_bias = rearrange(teacher.bias, "(p c) -> p c", p=patch)
        expected_bias = rearrange(expected_bias[:, None].expand(-1, grid, -1), "p n c -> (p n c)")
    else:
        expected_weight = teacher.weight.repeat(grid, 1)
        expected_bias = teacher.bias.repeat(grid)
    assert torch.equal(linear.weight, expected_weight)
    assert torch.equal(linear.bias, expected_bias)
    assert set(linear.state_dict()) == {"weight", "bias"}

    # A widened checkpoint is already laid out per head and must not be repeated again.
    reloaded = PDDLinear(
        in_features,
        channels * patch,
        patch_size=patch,
        grid_size=grid,
        channel_layout=layout,
    )
    reloaded.load_state_dict(linear.state_dict())
    assert torch.equal(reloaded.weight, linear.weight)
    assert torch.equal(reloaded.bias, linear.bias)


def test_pdd_linear_nested_fusion_restores_one_complete_transient_state(monkeypatch):
    linear = PDDLinear(3, 2, patch_size=1, grid_size=4, channel_layout="channel_major").float()
    x = torch.randn(2, 3)
    outer_weights = torch.tensor([-0.2, -0.3, -0.1, -0.4], dtype=torch.float64)
    inner_weights = torch.tensor(
        [[-0.2, -0.3, 0.0, 0.0], [0.0, 0.0, -0.1, -0.4]],
        dtype=torch.float64,
    )
    calls = []
    original_fused_params = linear._fused_params

    def capture_fused_params(step_index, next_step_index, weights, compute_dtype):
        calls.append((step_index, next_step_index, weights, compute_dtype))
        return original_fused_params(step_index, next_step_index, weights, compute_dtype)

    monkeypatch.setattr(linear, "_fused_params", capture_fused_params)
    unfused = linear(x)
    with linear.fuse(1, 4, outer_weights, torch.float64):
        outer_first = linear(x)
        with linear.fuse(None, None, inner_weights, torch.float32):
            linear(x)
        outer_second = linear(x)

    torch.testing.assert_close(outer_second, outer_first)
    torch.testing.assert_close(linear(x), unfused)
    assert len(calls) == 3
    assert calls[0][:2] == calls[2][:2] == (1, 4)
    assert calls[0][2] is outer_weights and calls[2][2] is outer_weights
    assert calls[0][3] == calls[2][3] == torch.float64
    assert calls[1][:2] == (None, None)
    assert calls[1][2] is inner_weights
    assert calls[1][3] == torch.float32
    assert set(linear.state_dict()) == {"weight", "bias"}


class _H3LikeBase(FastGenNetwork):
    """Small two-head stand-in that uses MiniMax-H3's actual output-conversion methods."""

    def __init__(
        self,
        load_pretrained=True,
        net_pred_type="flow",
        student_grid_cfg=None,
        modalities=("video", "audio"),
    ):
        super().__init__(
            net_pred_type=net_pred_type,
            schedule_type="rf",
            min_t=0.0,
            max_t=0.999,
            student_grid_cfg=student_grid_cfg,
        )
        self.modalities = modalities
        self.noise_scheduler = CompositeNoiseSchedule(
            schedules={
                "video": {
                    "name": "rf",
                    "min_t": 0.0,
                    "max_t": 0.999,
                    "student_grid_cfg": self.noise_scheduler.student_grid_cfg,
                },
                "audio": {"name": "shifted_rf", "shift": 0.25, "min_t": 0.0, "max_t": 0.999},
            },
            primary_key="video",
        )
        self.transformer = torch.nn.Module()
        self.transformer.proj_out = torch.nn.Linear(3, 4)
        self.transformer.audio_proj_out = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.transformer.proj_out.weight.copy_(torch.arange(12).reshape(4, 3))
            self.transformer.proj_out.bias.copy_(torch.arange(4))
            self.transformer.audio_proj_out.weight.copy_(torch.arange(6).reshape(2, 3) + 20)
            self.transformer.audio_proj_out.bias.copy_(torch.arange(2) + 10)

    def _is_in_meta_context(self):
        return self.transformer.proj_out.weight.device.type == "meta"

    def flow_convention_scale(self, t):
        from fastgen.networks.MiniMaxH3.network import MiniMaxH3

        return MiniMaxH3.flow_convention_scale(self, t)

    def _to_flow_convention(self, model_output, t):
        from fastgen.networks.MiniMaxH3.network import MiniMaxH3

        return MiniMaxH3._to_flow_convention(self, model_output, t)

    def forward(self, x_t, t, condition=None, fwd_pred_type=None, **kwargs):
        del condition, fwd_pred_type, kwargs
        output = {
            "video": self.transformer.proj_out(x_t["video"]),
            "audio": self.transformer.audio_proj_out(x_t["audio"]),
        }
        return self._to_flow_convention(output, t)


_H3_LIKE_SPEC = PDDSpec(
    final_linear={"video": "transformer.proj_out", "audio": "transformer.audio_proj_out"},
    channel_layout="channel_major",
)

_H3LikeGenericPDD = make_pdd(
    _H3LikeBase,
    _H3_LIKE_SPEC,
)


class _H3LikePDD(PDDMixin, _H3LikeBase):
    """Lightweight stand-in exercising MiniMaxH3PDD's architecture-specific methods."""

    PDD_SPEC = _H3_LIKE_SPEC
    pdd_teacher_head_scale = MiniMaxH3PDD.pdd_teacher_head_scale
    _to_flow_convention = MiniMaxH3PDD._to_flow_convention


def test_h3_heads_fold_their_fixed_node_scales_into_initial_parameters():
    grid = 4
    net = _H3LikePDD(pdd_steps=grid, student_grid_cfg=_native_grid(grid))
    base_grid = net.noise_scheduler.student_grid(net.pdd_steps, device=torch.device("cpu"))

    # H3-style raw heads already emit ``eps - x0 = dx/dsigma``: copied
    # teacher parameters are repeated without any fold.
    expected_video_weight = torch.arange(12).reshape(4, 3).float().repeat(grid, 1)
    expected_video_bias = torch.arange(4).float().repeat(grid)
    base_audio_weight = (torch.arange(6).reshape(2, 3) + 20).float()
    base_audio_bias = (torch.arange(2) + 10).float()
    assert torch.equal(net.pdd_linear["video"].weight, expected_video_weight)
    assert torch.equal(net.pdd_linear["video"].bias, expected_video_bias)
    assert torch.equal(net.pdd_linear["audio"].weight, base_audio_weight.repeat(grid, 1))
    assert torch.equal(net.pdd_linear["audio"].bias, base_audio_bias.repeat(grid))
    assert net.pdd_linear["audio"]._teacher_scale is None
    assert set(net.pdd_linear["audio"].state_dict()) == {"weight", "bias"}

    # A widened H3-PDD checkpoint is already native and must not be scaled again.
    reloaded = _H3LikePDD(pdd_steps=grid, student_grid_cfg=_native_grid(grid))
    reloaded.load_state_dict(net.state_dict())
    assert torch.equal(reloaded.pdd_linear["audio"].weight, base_audio_weight.repeat(grid, 1))
    assert torch.equal(reloaded.pdd_linear["audio"].bias, base_audio_bias.repeat(grid))

    # The generic mixin folds ``1/sigma'_m(t_j)`` into a flow-emitting
    # teacher's copied parameters; the unwarped video modality needs no fold.
    raw = {"video": torch.randn(2, 3), "audio": torch.randn(2, 3)}
    generic = _H3LikeGenericPDD(pdd_steps=grid, student_grid_cfg=_native_grid(grid))
    audio_sigma_prime = net.noise_scheduler.subschedules["audio"].sigma_prime(base_grid[:-1])
    audio_fold = 1.0 / audio_sigma_prime
    expected_audio_weight = rearrange(
        (base_audio_weight.double()[None] * audio_fold[:, None, None]).float(),
        "n c i -> (n c) i",
    )
    expected_audio_bias = rearrange(
        (base_audio_bias.double()[None] * audio_fold[:, None]).float(),
        "n c -> (n c)",
    )
    assert generic.pdd_linear["video"]._teacher_scale is None
    assert generic.pdd_linear["audio"]._teacher_scale == tuple((1.0 / audio_sigma_prime).tolist())
    assert torch.equal(generic.pdd_linear["video"].weight, expected_video_weight)
    assert torch.equal(generic.pdd_linear["audio"].weight, expected_audio_weight)
    assert torch.equal(generic.pdd_linear["audio"].bias, expected_audio_bias)
    base_converted = generic._to_flow_convention(raw, base_grid[:2])
    assert torch.equal(base_converted["video"], raw["video"])
    assert torch.equal(base_converted["audio"], raw["audio"].double() * audio_sigma_prime[:2, None])

    # Only H3-PDD suppresses H3's single-time conversion because its raw heads are already native.
    assert "_to_flow_convention" not in PDDMixin.__dict__
    assert "_to_flow_convention" in MiniMaxH3PDD.__dict__
    pdd_passthrough = net._to_flow_convention(raw, base_grid[:2])
    assert all(torch.equal(pdd_passthrough[key], raw[key]) for key in raw)

    x = torch.randn(2, 3)
    audio = net.pdd_linear["audio"]
    weights = net.pdd_block_weights(0, grid, base_grid)[0]["audio"]
    unfused = audio(x).unflatten(-1, (grid, 2))
    with audio.fuse(0, grid, weights, unfused.dtype):
        fused = audio(x)
    expected_fused = torch.einsum("n,bnc->bc", weights.to(unfused.dtype), unfused)
    torch.testing.assert_close(fused, expected_fused)


def test_teacher_head_scale_is_available_during_meta_initialization():
    with torch.device("meta"):
        net = _H3LikeGenericPDD(pdd_steps=4, student_grid_cfg=_native_grid(4))

    audio = net.pdd_linear["audio"]
    assert audio.weight.device.type == "meta"
    base_grid = net.noise_scheduler.student_grid(net.pdd_steps, device=torch.device("cpu"))
    expected = 1.0 / net.noise_scheduler.subschedules["audio"].sigma_prime(base_grid[:-1])
    assert audio._teacher_scale == tuple(expected.tolist())
    assert net.pdd_linear["video"]._teacher_scale is None


def test_teacher_head_scale_uses_runtime_scheduler_precision_for_float32_explicit_grid():
    explicit_grid = torch.tensor([0.999, 0.73, 0.38, 0.11, 0.0], dtype=torch.float32)
    net = _H3LikeGenericPDD(pdd_steps=4, student_grid_cfg=_explicit_grid(explicit_grid))

    runtime_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=torch.device("cpu"),
        dtype=net.noise_scheduler.t_precision,
    )
    generator_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=torch.device("cpu"),
    )
    expected = 1.0 / net.noise_scheduler.subschedules["audio"].sigma_prime(runtime_grid[:-1])

    assert runtime_grid.dtype == net.noise_scheduler.t_precision
    assert torch.equal(runtime_grid, generator_grid.to(net.noise_scheduler.t_precision))
    assert net.pdd_linear["audio"]._teacher_scale == tuple(expected.tolist())


def test_multimodal_fused_pdd_block_returns_its_mean_sigma_velocity():
    grid = 4
    net = _H3LikePDD(pdd_steps=grid, student_grid_cfg=_native_grid(grid))
    x_t = {"video": torch.randn(2, 3), "audio": torch.randn(2, 3)}
    state = {"video": torch.randn(2, 4), "audio": torch.randn(2, 2)}

    heads = net(x_t, step_index=0)
    native_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=torch.device("cpu"),
        dtype=net.noise_scheduler.t_precision,
    )
    weights = _interval_weights(net, native_grid)
    mean_sigma_velocity = net(
        x_t,
        step_index=0,
        fused=True,
        next_step_index=grid,
    )
    _, total_weight = net.pdd_block_weights(0, grid, native_grid)
    state_weight = per_modality(lambda x, w: w.to(dtype=x.dtype), state, total_weight)
    advanced = net.pdd_advance_prediction(state, mean_sigma_velocity, state_weight)

    for key in state:
        normalized_weights = (weights[key] / total_weight[key]).to(heads[key].dtype)
        expected_sigma_velocity = torch.einsum("n,bnc->bc", normalized_weights, heads[key]).to(x_t[key].dtype)
        expected_state = state[key] + total_weight[key].to(state[key].dtype) * expected_sigma_velocity.to(
            state[key].dtype
        )
        torch.testing.assert_close(mean_sigma_velocity[key], expected_sigma_velocity)
        torch.testing.assert_close(advanced[key], expected_state)


def test_fused_pdd_block_defaults_fusion_to_input_precision_and_preserves_output_dtype(monkeypatch):
    net = _toy_pdd(schedule_shift=0.25)
    x_t = torch.tensor([[0.25], [-0.4]], dtype=torch.float32)
    base_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=x_t.device,
        dtype=net.noise_scheduler.t_precision,
    )
    weights = _interval_weights(net, base_grid)
    fusion_compute_dtypes = []
    original_fused_params = net.pdd_linear._fused_params

    def capture_fused_params(step_index, next_step_index, fusion_weights, compute_dtype):
        fusion_compute_dtypes.append(compute_dtype)
        return original_fused_params(step_index, next_step_index, fusion_weights, compute_dtype)

    monkeypatch.setattr(net.pdd_linear, "_fused_params", capture_fused_params)

    heads = net(x_t, step_index=0)
    normalized_weights = (weights / weights.sum()).to(heads.dtype)
    expected = torch.einsum("n,bn...->b...", normalized_weights, heads)
    actual = net(
        x_t,
        step_index=0,
        fused=True,
        next_step_index=net.pdd_steps,
    )

    assert actual.dtype == x_t.dtype
    assert fusion_compute_dtypes == [x_t.dtype]
    torch.testing.assert_close(actual, expected)


def test_fused_forward_output_follows_input_dtype_not_fusion_precision(monkeypatch):
    net = _toy_pdd(schedule_shift=0.25).to(dtype=torch.bfloat16)
    x_t = torch.tensor([[0.25], [-0.4]], dtype=torch.bfloat16)
    fusion_compute_dtypes = []
    original_fused_params = net.pdd_linear._fused_params

    def capture_fused_params(step_index, next_step_index, weights, precision_decoding):
        fusion_compute_dtypes.append(precision_decoding)
        return original_fused_params(step_index, next_step_index, weights, precision_decoding)

    monkeypatch.setattr(net.pdd_linear, "_fused_params", capture_fused_params)

    output = net(
        x_t,
        step_index=0,
        fused=True,
        next_step_index=net.pdd_steps,
        precision_decoding=torch.float32,
    )
    assert output.dtype == x_t.dtype

    assert fusion_compute_dtypes == [torch.float32]
    for pred_type in ("flow", "x0", "eps"):
        with pytest.raises(TypeError, match="does not accept fwd_pred_type"):
            net(
                x_t,
                step_index=0,
                fused=True,
                next_step_index=net.pdd_steps,
                precision_decoding=torch.float32,
                fwd_pred_type=pred_type,
            )


def test_fused_prediction_rejects_conversion_and_restores_native_output_dtype(monkeypatch):
    net = _toy_pdd(schedule_shift=0.25)
    x_t = torch.tensor([[0.25], [-0.4]], dtype=torch.bfloat16)
    mean_sigma_velocity = torch.tensor([[0.1234567], [-0.7654321]], dtype=torch.float32)

    monkeypatch.setattr(_ToyPredictionBase, "forward", lambda *args, **kwargs: mean_sigma_velocity)
    monkeypatch.setattr(
        net.noise_scheduler,
        "convert_model_output",
        lambda *args, **kwargs: pytest.fail("fused PDD output must remain in its native representation"),
    )
    output = net(
        x_t,
        step_index=0,
        fused=True,
        next_step_index=net.pdd_steps,
        precision_decoding=torch.float32,
    )
    assert output.dtype == x_t.dtype
    torch.testing.assert_close(output, mean_sigma_velocity.to(dtype=x_t.dtype))
    with pytest.raises(TypeError, match="does not accept fwd_pred_type"):
        net(
            x_t,
            step_index=0,
            fused=True,
            next_step_index=net.pdd_steps,
            precision_decoding=torch.float32,
            fwd_pred_type="x0",
        )


@pytest.mark.parametrize(("start", "block_size"), [(0, 1), (0, 4), (1, 2)])
def test_fused_mean_sigma_velocity_advance_matches_materialized_training_integration(monkeypatch, start, block_size):
    net = _toy_pdd(schedule_shift=0.25).to(dtype=torch.bfloat16)
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.precision = torch.bfloat16
    model.precision_decoding = torch.float32
    with torch.no_grad():
        net.pdd_linear.weight.copy_(torch.tensor([[0.25], [-0.5], [1.0], [-1.5]]))

    final_input_dtypes = []
    fusion_compute_dtypes = []
    original_fused_params = net.pdd_linear._fused_params

    def capture_fused_params(step_index, next_step_index, weights, precision_decoding):
        fusion_compute_dtypes.append(precision_decoding)
        return original_fused_params(step_index, next_step_index, weights, precision_decoding)

    monkeypatch.setattr(net.pdd_linear, "_fused_params", capture_fused_params)
    handle = net.pdd_linear.register_forward_pre_hook(lambda _module, args: final_input_dtypes.append(args[0].dtype))

    x_t = torch.tensor([[-0.75], [0.125], [1.5]], dtype=torch.float64)
    base_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=x_t.device,
        dtype=net.noise_scheduler.t_precision,
    )
    step_index = torch.full((x_t.shape[0],), start, dtype=torch.long)
    next_step_index = step_index + block_size

    try:
        heads = _predict_heads(model, net, x_t, step_index=step_index, condition=None)
        training_state = _advance_materialized_pdd_block(
            net,
            x_t.to(model.precision_decoding),
            heads.to(model.precision_decoding),
            step_index,
            next_step_index,
            base_grid,
        )
        mean_sigma_velocity = net(
            x_t.to(model.precision),
            start,
            fused=True,
            next_step_index=start + block_size,
            precision_decoding=model.precision_decoding,
        )
        _, total_weight = net.pdd_block_weights(start, start + block_size, base_grid)
        inference_state = net.pdd_advance_prediction(
            x_t.to(dtype=model.precision_decoding),
            mean_sigma_velocity.to(dtype=model.precision_decoding),
            total_weight.to(dtype=model.precision_decoding),
        )
    finally:
        handle.remove()

    assert x_t.dtype == torch.float64
    assert heads.dtype == model.precision
    assert mean_sigma_velocity.dtype == model.precision
    assert training_state.dtype == model.precision_decoding
    assert inference_state.dtype == model.precision_decoding
    assert final_input_dtypes == [model.precision, model.precision]
    assert fusion_compute_dtypes == [model.precision_decoding]
    torch.testing.assert_close(inference_state, training_state, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize(("start", "block_size"), [(0, 1), (0, 4), (1, 2)])
def test_materialized_next_state_matches_fused_mean_sigma_velocity_advance(start, block_size):
    net = _toy_pdd(schedule_shift=0.25)
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = net
    model.config = SimpleNamespace(
        backward_simulation=True,
        loss_config=SimpleNamespace(
            block_size_min=block_size,
            block_size_max=block_size,
        ),
        student_sample_type="ode",
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float64

    state = torch.randn(3, 1, dtype=model.precision_decoding)
    base_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=state.device,
        dtype=net.noise_scheduler.t_precision,
    )
    step_index = torch.full((state.shape[0],), start, dtype=torch.long)
    next_step_index = step_index + block_size
    heads = _predict_heads(model, net, state, step_index=step_index, condition=None)

    next_state = net.pdd_advance_blocks(
        state,
        heads.to(dtype=model.precision_decoding),
        step_index,
        next_step_index,
        base_grid,
    ).to(model.precision)
    mean_sigma_velocity = net(
        state.to(model.precision),
        step_index,
        fused=True,
        next_step_index=next_step_index,
        precision_decoding=model.precision_decoding,
    )
    _, total_weight = net.pdd_block_weights(step_index, next_step_index, base_grid)
    forward_next_state = net.pdd_advance_prediction(
        state,
        mean_sigma_velocity.to(dtype=model.precision_decoding),
        total_weight.to(dtype=model.precision_decoding),
    ).to(model.precision)

    assert next_state.dtype == model.precision
    torch.testing.assert_close(next_state, forward_next_state, rtol=2e-6, atol=2e-6)


def test_pdd_step_returns_method_dtype_and_integrates_in_decoding_precision(monkeypatch):
    """The fused update uses FP32 locally and returns to model precision."""
    net = _toy_pdd(steps=2, schedule_shift=0.25).to(dtype=torch.bfloat16)
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        student_sample_type="ode",
        loss_config=SimpleNamespace(block_size_max=net.pdd_steps),
    )
    model.precision = torch.bfloat16
    model.precision_decoding = torch.float32

    student_grid = net.noise_scheduler.student_grid(
        net.noise_scheduler.default_student_steps,
        device=torch.device("cpu"),
        dtype=net.noise_scheduler.t_precision,
    )
    base_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=torch.device("cpu"),
        dtype=net.noise_scheduler.t_precision,
    )
    start, end = net.noise_scheduler.student_grid_indices(
        target_steps=len(student_grid) - 1,
    )[:2].tolist()
    # The sampler carries states in method precision (trainer/generator contract).
    state = torch.tensor([[-0.75], [0.125], [1.5]], dtype=model.precision)
    step_index = torch.full((state.shape[0],), start, dtype=torch.long)
    next_step_index = torch.full_like(step_index, end)
    heads = _predict_heads(model, net, state, step_index=step_index, condition=None)
    expected = _advance_materialized_pdd_block(
        net,
        state.to(model.precision_decoding),
        heads.to(model.precision_decoding),
        step_index,
        next_step_index,
        base_grid,
    ).to(model.precision)

    network_input_dtypes = []
    network_output_dtypes = []
    fusion_compute_dtypes = []
    update_dtypes = []
    original_fused_params = net.pdd_linear._fused_params
    original_advance = net.pdd_advance_prediction

    def capture_fused_params(step_index, next_step_index, weights, precision_decoding):
        fusion_compute_dtypes.append(precision_decoding)
        return original_fused_params(step_index, next_step_index, weights, precision_decoding)

    def capture_advance(x_t, pred, weight):
        update_dtypes.append((x_t.dtype, pred.dtype, weight.dtype))
        return original_advance(x_t, pred, weight)

    monkeypatch.setattr(net.pdd_linear, "_fused_params", capture_fused_params)
    monkeypatch.setattr(net, "pdd_advance_prediction", capture_advance)
    pre_handle = net.register_forward_pre_hook(lambda _module, args: network_input_dtypes.append(args[0].dtype))
    post_handle = net.register_forward_hook(lambda _module, _args, output: network_output_dtypes.append(output.dtype))
    try:
        actual = model._advance_student_step(net, state, 0, student_grid, condition=None)
    finally:
        pre_handle.remove()
        post_handle.remove()

    # The method layer passes the method-precision state straight to the network;
    # the decoding promotion lives inside pdd_advance_blocks, as update_dtypes proves.
    assert network_input_dtypes == [model.precision]
    assert network_output_dtypes == [model.precision]
    assert fusion_compute_dtypes == [model.precision_decoding]
    assert update_dtypes == [(model.precision_decoding, model.precision_decoding, model.precision_decoding)]
    # The advanced state returns in the incoming (method-precision) dtype.
    assert actual.dtype == model.precision
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)


def test_pdd_step_uses_the_explicit_runtime_partition_instead_of_the_default_partition(monkeypatch):
    """A runtime NFE must choose its own base-grid head blocks, even when the default NFE differs."""
    net = _toy_pdd(steps=4, schedule_shift=0.25)

    def unexpected_step(*args, **kwargs):
        raise AssertionError("PDD advances its fused mean sigma-velocity without a scheduler step")

    monkeypatch.setattr(net.noise_scheduler, "step", unexpected_step)
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        student_sample_type="ode",
        loss_config=SimpleNamespace(block_size_max=net.pdd_steps),
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float64

    runtime_steps = 2
    student_grid = net.noise_scheduler.student_grid(
        runtime_steps,
        device=torch.device("cpu"),
        dtype=net.noise_scheduler.t_precision,
    )
    base_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=torch.device("cpu"),
        dtype=net.noise_scheduler.t_precision,
    )
    student_grid_indices = net.noise_scheduler.student_grid_indices(target_steps=runtime_steps)
    state = torch.randn(3, 1, dtype=model.precision)

    for trajectory_step in range(runtime_steps):
        start = int(student_grid_indices[trajectory_step])
        end = int(student_grid_indices[trajectory_step + 1])
        step_index = torch.full((state.shape[0],), start, dtype=torch.long)
        next_step_index = torch.full_like(step_index, end)
        heads = _predict_heads(model, net, state, step_index=step_index, condition=None)
        expected = _advance_materialized_pdd_block(
            net,
            state.to(model.precision_decoding),
            heads.to(model.precision_decoding),
            step_index,
            next_step_index,
            base_grid,
        ).to(model.precision)

        actual = model._advance_student_step(
            net,
            state,
            trajectory_step,
            student_grid,
            condition=None,
        )

        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        state = actual


def test_h3_pdd_rejects_non_flow_prediction_types():
    with pytest.raises(ValueError, match="requires net_pred_type='flow'"):
        _H3LikePDD(pdd_steps=4, student_grid_cfg=_native_grid(4), net_pred_type="x0")


def test_make_pdd_sets_an_explicit_defining_module():
    wrapped = make_pdd(
        _ToyPredictionBase,
        PDDSpec(final_linear="proj", channel_layout="channel_major"),
        module=__name__,
    )

    assert wrapped.__module__ == __name__


def test_pdd_constructor_requires_flow_prediction_and_rf_scheduler():
    sampling = _native_grid(4)
    net = _ToyPDD(
        pdd_steps=4,
        student_grid_cfg=sampling,
        net_pred_type="flow",
    )
    assert net.net_pred_type == "flow"
    assert isinstance(net.noise_scheduler, ShiftedRFNoiseSchedule)
    assert not hasattr(net, "precision_decoding")

    with pytest.raises(ValueError, match="requires net_pred_type='flow'"):
        _ToyPDD(pdd_steps=4, student_grid_cfg=sampling, net_pred_type="x0")
    with pytest.raises(ValueError, match="requires an 'rf' or 'shifted_rf' schedule"):
        _ToyPDD(
            pdd_steps=4,
            student_grid_cfg=sampling,
            net_pred_type="flow",
            noise_scheduler=get_noise_schedule("trig"),
        )


def test_pdd_owns_architectural_steps_independently_of_the_sampling_default():
    sampling = StudentGridConfig(
        steps=4,
        grid_type="native",
        grid_kwargs={"base_steps": 8, "shift": 5.0},
    )

    net = _ToyPDD(pdd_steps=8, student_grid_cfg=sampling)

    assert net.pdd_steps == 8
    assert net.pdd_linear.grid_size == 8
    assert torch.equal(
        net.noise_scheduler.student_grid_indices(target_steps=net.noise_scheduler.default_student_steps),
        torch.tensor([0, 2, 4, 6, 8]),
    )
    with set_temp_global_vars({"STUDENT_STEPS": 2}):
        assert torch.equal(
            net.noise_scheduler.student_grid_indices(target_steps=net.noise_scheduler.default_student_steps),
            torch.tensor([0, 4, 8]),
        )
    base_grid = net.pdd_base_grid()
    assert base_grid.shape == (9,)
    assert base_grid.dtype == net.noise_scheduler.t_precision


@pytest.mark.parametrize(
    ("student_grid_cfg", "match"),
    [
        (_native_grid(4), r"pdd_steps=8 .* student_base_steps=4"),
        (StudentGridConfig(steps=4), "requires an anchored student grid"),
    ],
)
def test_pdd_constructor_rejects_a_base_grid_that_does_not_match_its_heads(student_grid_cfg, match):
    with pytest.raises(ValueError, match=match):
        _ToyPDD(pdd_steps=8, student_grid_cfg=student_grid_cfg)


def test_pdd_rollout_contract_validates_backward_blocks_at_build_time():
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = _FakePDDNet(
        get_noise_schedule("rf"),
        pdd_steps=8,
        blocks=[1, 1, 1, 5],
    )
    model.config = SimpleNamespace(
        backward_simulation=None,
        loss_config=SimpleNamespace(block_size_min=2, block_size_max=4),
    )

    model._validate_pdd_rollout_contract()

    model.config.backward_simulation = "staggered"
    with pytest.raises(ValueError, match="Backward-simulation PDD student-grid block 5 exceeds"):
        model._validate_pdd_rollout_contract()


def test_pdd_rollout_balances_a_non_divisible_base_grid():
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = _FakePDDNet(get_noise_schedule("rf"), pdd_steps=128)
    model.config = SimpleNamespace(
        backward_simulation="staggered",
        loss_config=SimpleNamespace(block_size_min=48, block_size_max=48),
    )

    assert model._max_student_steps() == 3
    grid_plan = model._pdd_student_grid_plan(
        model.net,
        num_steps=model._max_student_steps(),
        device=torch.device("cpu"),
    )
    assert grid_plan.student_grid_indices.tolist() == [0, 42, 85, 128]
    torch.testing.assert_close(grid_plan.student_grid, grid_plan.base_grid[grid_plan.student_grid_indices])
    model._validate_pdd_rollout_contract()


def test_pdd_validates_the_default_sampling_mapping_when_requested():
    sampling = StudentGridConfig(
        steps=4,
        grid_type="native",
        grid_kwargs={"base_steps": 8, "indices": {4: [0, 2, 8]}},
    )

    net = _ToyPDD(pdd_steps=8, student_grid_cfg=sampling)

    with pytest.raises(ValueError, match=r"must have shape \(5,\)"):
        net.noise_scheduler.student_grid_indices(target_steps=net.noise_scheduler.default_student_steps)


@pytest.mark.parametrize("pdd_steps", [0, 1, -1, True, 1.5])
def test_pdd_rejects_invalid_architectural_steps(pdd_steps):
    with pytest.raises(ValueError, match="pdd_steps must be an int >= 2"):
        _ToyPDD(pdd_steps=pdd_steps, student_grid_cfg=_native_grid(4))


def test_pdd_linear_preserves_the_spec_modality_structure():
    sampling = _native_grid(4)

    unimodal = _ToyPDD(pdd_steps=4, student_grid_cfg=sampling)
    assert unimodal.pdd_linear is unimodal.proj

    audio_only = _H3LikeGenericPDD(pdd_steps=4, student_grid_cfg=sampling, modalities=("audio",))
    assert set(audio_only.pdd_linear) == {"audio"}
    assert audio_only.pdd_linear["audio"] is audio_only.transformer.audio_proj_out
    assert not isinstance(audio_only.transformer.proj_out, PDDLinear)


def _toy_pdd(
    pdd_steps=4,
    steps=2,
    shift=1.0,
    schedule_shift=1.0,
    blocks=None,
):
    net = _ToyPDD(
        pdd_steps=pdd_steps,
        net_pred_type="flow",
        schedule_shift=schedule_shift,
        student_grid_cfg=_native_grid(
            pdd_steps,
            steps=steps,
            shift=shift,
            blocks=blocks,
        ),
    )
    with torch.no_grad():
        net.pdd_linear.weight.zero_()
        net.pdd_linear.bias.copy_(torch.arange(1, pdd_steps + 1, dtype=net.pdd_linear.bias.dtype))
    return net


def _expected_block_mean_sigma_velocity(net, heads, step_index, next_step_index, base_grid):
    """Direct reference for one PDD block's weighted mean native sigma-velocity."""
    weights = _interval_weights(net, base_grid)[step_index:next_step_index]
    normalized_weights = (weights / weights.sum()).to(heads.dtype)
    return torch.einsum("n,n...->...", normalized_weights, heads[step_index:next_step_index])


def test_unfused_pdd_heads_are_native_sigma_velocity_and_reject_prediction_type_override():
    net = _toy_pdd(schedule_shift=0.25)
    x_t = torch.tensor([[0.25], [-0.4]])

    default = net(x_t, step_index=0)

    assert default.shape == (2, net.pdd_steps, 1)
    for pred_type in ("flow", "x0", "eps"):
        with pytest.raises(TypeError, match="does not accept fwd_pred_type"):
            net(x_t, step_index=0, fwd_pred_type=pred_type)


def test_pdd_forward_never_applies_the_base_output_constraint(monkeypatch):
    """The base network constrains single-time predictions, so neither PDD heads nor fused blocks request it."""
    net = _toy_pdd(schedule_shift=0.25)
    x_t = torch.tensor([[0.25], [-0.4]])
    forward = _ToyPredictionBase.forward
    requested = []

    def record(self, *args, apply_output_conditioning=True, **kwargs):
        requested.append(apply_output_conditioning)
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(_ToyPredictionBase, "forward", record)
    net(x_t, step_index=0)
    net(x_t, step_index=0, fused=True, next_step_index=net.pdd_steps, apply_output_conditioning=False)

    assert requested == [False, False]
    with pytest.raises(NotImplementedError, match="does not apply output conditioning"):
        net(x_t, step_index=0, apply_output_conditioning=True)


def test_batched_pdd_linear_fusion_matches_independent_shared_fusions():
    """Mixed timestep rows get distinct fused parameters without producing widened activations."""
    linear = PDDLinear(3, 2, patch_size=1, grid_size=4, channel_layout="channel_major").float()
    weights = torch.tensor(
        [
            [-0.2, -0.3, -0.1, -0.4],
            [0.0, 0.0, -0.25, -0.75],
        ],
        dtype=torch.float64,
    )
    x = torch.randn(2, 5, 3)

    with linear.fuse(None, None, weights, x.dtype):
        actual = linear(x)

    expected = []
    for row, row_weights in zip(x, weights):
        active = torch.nonzero(row_weights).flatten()
        start, end = int(active[0]), int(active[-1]) + 1
        with linear.fuse(start, end, row_weights, row.dtype):
            expected.append(linear(row[None])[0])
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 5, 2), "the head-sized output axis was fused inside PDDLinear"


@pytest.mark.parametrize(
    ("step_index", "next_step_index"),
    [(0, 2), (torch.tensor([0, 1]), torch.tensor([2, 3]))],
)
def test_fused_and_materialized_pdd_paths_reject_a_zero_block_total(monkeypatch, step_index, next_step_index):
    """Every interval of this non-monotone grid has a weight, but the block from 0.9 back to 0.9 totals zero."""
    net = _toy_pdd()
    base_grid = torch.tensor([0.9, 0.5, 0.9, 0.3, 0.0], dtype=torch.float64)
    monkeypatch.setattr(net, "pdd_base_grid", lambda device=None: base_grid.to(device))
    state = torch.zeros(2, 1)

    with pytest.raises(ValueError, match="PDD block total weight must be finite and non-zero"):
        net(state, step_index, fused=True, next_step_index=next_step_index)
    with pytest.raises(ValueError, match="PDD block total weight must be finite and non-zero"):
        net.pdd_advance_blocks(state, torch.zeros(2, net.pdd_steps, 1), step_index, next_step_index, base_grid)


def test_pdd_forward_fuses_only_the_selected_block_as_native_mean_sigma_velocity():
    net = _toy_pdd(schedule_shift=0.25)
    base_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=torch.device("cpu"),
        dtype=net.noise_scheduler.t_precision,
    )
    student_steps = torch.tensor([0, 1])
    student_grid_indices = net.noise_scheduler.student_grid_indices(
        target_steps=net.noise_scheduler.default_student_steps
    )
    step_index = student_grid_indices[student_steps]
    next_step_index = student_grid_indices[student_steps + 1]
    x_t = torch.tensor([[0.25], [-0.4]], requires_grad=True)
    head_values = torch.arange(1, net.pdd_steps + 1, dtype=x_t.dtype)
    expected_mean_sigma_velocity = torch.stack(
        [
            _expected_block_mean_sigma_velocity(
                net,
                head_values[:, None],
                int(start),
                int(end),
                base_grid,
            )
            for start, end in zip(step_index, next_step_index)
        ]
    )

    projection_shapes = []
    hook = net.pdd_linear.register_forward_hook(lambda _module, _args, output: projection_shapes.append(output.shape))
    actual_mean_sigma_velocity = net(
        x_t,
        step_index,
        fused=True,
        next_step_index=next_step_index,
    )
    hook.remove()
    torch.testing.assert_close(actual_mean_sigma_velocity, expected_mean_sigma_velocity)
    assert projection_shapes == [torch.Size([2, 1])]

    actual_mean_sigma_velocity.sum().backward()
    assert net.pdd_linear.bias.grad is not None
    assert torch.isfinite(net.pdd_linear.bias.grad).all()


def test_fused_mean_sigma_velocity_gradient_is_confined_to_each_configured_block():
    """Every student start updates exactly its next block, never any other PDD head."""
    blocks = [1, 3, 2, 2]
    net = _toy_pdd(
        pdd_steps=8,
        steps=4,
        shift=5.0,
        schedule_shift=0.25,
        blocks=blocks,
    )
    base_grid = net.noise_scheduler.student_grid(net.pdd_steps, device=torch.device("cpu"))
    student_grid_indices = net.noise_scheduler.student_grid_indices(
        target_steps=net.noise_scheduler.default_student_steps
    ).tolist()

    for start, end in zip(student_grid_indices, student_grid_indices[1:]):
        net.zero_grad(set_to_none=True)
        x_t = torch.tensor([[0.25]])
        net(x_t, start, fused=True, next_step_index=end).sum().backward()

        block_support = torch.zeros(net.pdd_steps, dtype=torch.bool)
        block_support[start:end] = True
        expected_support = (_interval_weights(net, base_grid) != 0) & block_support
        for grad in (net.pdd_linear.weight.grad[:, 0], net.pdd_linear.bias.grad):
            assert grad is not None
            assert torch.equal(grad != 0, expected_support), (start, end, grad)
            assert not torch.any((grad != 0) & ~block_support)


def test_block_prediction_stays_native_and_advances_through_block_contract():
    """The opaque fused prediction advances through the scheduler-specific hook."""
    blocks = [1, 3, 2, 2]
    net = _toy_pdd(
        pdd_steps=8,
        steps=4,
        shift=5.0,
        schedule_shift=3.0,
        blocks=blocks,
    )
    base_grid = net.noise_scheduler.student_grid(net.pdd_steps, device=torch.device("cpu"))
    student_grid_indices = net.noise_scheduler.student_grid_indices(
        target_steps=net.noise_scheduler.default_student_steps
    ).tolist()

    for start, end in zip(student_grid_indices, student_grid_indices[1:]):
        x_t = torch.tensor([[0.25 + start]])
        heads = net(x_t, start)
        weights = _interval_weights(net, base_grid)[start:end]
        expected_mean_sigma_velocity = torch.einsum(
            "n,bn...->b...",
            (weights / weights.sum()).to(heads.dtype),
            heads[:, start:end],
        )

        mean_sigma_velocity = net(x_t, start, fused=True, next_step_index=end)
        advanced = net.pdd_advance_blocks(
            x_t,
            mean_sigma_velocity,
            start,
            end,
            base_grid,
            fused=True,
        )
        expected_state = x_t + torch.einsum(
            "n,bn...->b...",
            weights.to(heads.dtype),
            heads[:, start:end],
        )

        torch.testing.assert_close(mean_sigma_velocity, expected_mean_sigma_velocity)
        torch.testing.assert_close(advanced, expected_state)


def test_fused_forward_matches_explicit_mean_sigma_velocity_at_mixed_indices():
    """Each row fuses exactly the weighted mean of its selected PDD heads."""
    net = _toy_pdd(schedule_shift=0.25)
    base_grid = net.noise_scheduler.student_grid(net.pdd_steps, device=torch.device("cpu"))
    student_steps = torch.tensor([0, 1])
    student_grid_indices = net.noise_scheduler.student_grid_indices(
        target_steps=net.noise_scheduler.default_student_steps
    )
    pdd_indices = student_grid_indices[student_steps]
    next_pdd_indices = student_grid_indices[student_steps + 1]
    x_t = torch.tensor([[0.25], [-0.4]])
    heads = net(x_t, pdd_indices)

    expected = [
        _expected_block_mean_sigma_velocity(net, heads[row], int(start), int(end), base_grid)
        for row, (start, end) in enumerate(zip(pdd_indices, next_pdd_indices))
    ]

    actual = net(
        x_t,
        pdd_indices,
        fused=True,
        next_step_index=next_pdd_indices,
    )
    torch.testing.assert_close(actual, torch.stack(expected))


def test_pdd_student_grid_is_exact_subset_of_the_base_grid():
    net = _ToyPDD(
        pdd_steps=8,
        student_grid_cfg=_native_grid(8, shift=5.0),
        net_pred_type="flow",
    )
    full = net.noise_scheduler.student_grid(net.pdd_steps, device=torch.device("cpu"))
    selected = net.noise_scheduler.student_grid(4, device=torch.device("cpu"))
    assert torch.equal(selected, full[::2])

    balanced = net.noise_scheduler.student_grid(3, device=torch.device("cpu"))
    assert torch.equal(balanced, full[torch.tensor([0, 2, 5, 8])])


def test_dynamic_native_grid_is_constructed_at_the_requested_resolution():
    schedule = get_noise_schedule("rf", student_grid_cfg=StudentGridConfig(steps=5))

    assert schedule.student_grid().shape == (6,)
    assert schedule.student_grid(12).shape == (13,)


def test_student_grid_applies_explicit_indices_to_an_explicit_base_grid():
    base_grid = [0.999, 0.8, 0.4, 0.0]
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(
            steps=2,
            grid_type="explicit",
            grid_kwargs={"timesteps": base_grid, "indices": {2: [0, 1, 3]}},
        ),
    )
    assert torch.equal(
        schedule.student_grid(),
        torch.tensor(base_grid, dtype=schedule.t_precision)[torch.tensor([0, 1, 3])],
    )


def test_student_grid_rejects_unsupported_or_invalid_shift():
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(
            steps=1,
            grid_type="explicit",
            grid_kwargs={"timesteps": [0.999, 0.0], "shift": 2.0},
        ),
    )
    with pytest.raises(ValueError, match="Unsupported explicit student grid kwargs"):
        schedule.student_grid(device=torch.device("cpu"))

    schedule.student_grid_cfg = StudentGridConfig(steps=1, grid_kwargs={"shift": 0.0})
    with pytest.raises(ValueError, match="shift must be positive"):
        schedule.student_grid(device=torch.device("cpu"))


def test_pdd_sampling_uses_the_configured_partition():
    net = _ToyPDD(
        pdd_steps=8,
        student_grid_cfg=_native_grid(8, steps=4),
        net_pred_type="flow",
    )
    assert torch.equal(
        net.noise_scheduler.student_grid_indices(target_steps=net.noise_scheduler.default_student_steps),
        torch.tensor([0, 2, 4, 6, 8]),
    )


def test_pdd_advance_validates_only_the_current_block_and_inherits_module_mode():
    net = _toy_pdd(pdd_steps=8, steps=2, blocks=[1, 7])
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(loss_config=SimpleNamespace(block_size_max=3))
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    latents = torch.zeros(1, 1)
    student_grid = net.noise_scheduler.student_grid(device=latents.device)

    forward_modes = []
    handle = net.register_forward_hook(lambda module, _args, _output: forward_modes.append(module.training))
    try:
        net.train()
        state = model._advance_student_step(net, latents, 0, student_grid, condition=None)
        assert forward_modes == [True]
        assert net.training

        with pytest.raises(ValueError, match="exceeds the trained block_size_max"):
            model._advance_student_step(net, state, 1, student_grid, condition=None)
        assert forward_modes == [True]

        model.config.loss_config.block_size_max = 7
        net.eval()
        state = model._advance_student_step(net, state, 1, student_grid, condition=None)
    finally:
        handle.remove()
    assert forward_modes == [True, False]
    assert not net.training
    assert state.shape == latents.shape


def test_pdd_advance_validates_step_index_bounds():
    net = _toy_pdd(pdd_steps=8, steps=2)
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(loss_config=SimpleNamespace(block_size_max=4))
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    state = torch.tensor([[0.25], [-0.75]])
    student_grid = net.noise_scheduler.student_grid(device=state.device)

    model._advance_student_step(net, state, 0, student_grid, condition=None)

    for invalid in (-1, len(student_grid) - 1):
        with pytest.raises(ValueError, match="step_index must be"):
            model._advance_student_step(net, state, invalid, student_grid, condition=None)


def test_pdd_generator_uses_instance_precision(monkeypatch):
    """PDD inherits the native-input generator and caches its grid plan for sampling."""
    net = _toy_pdd(schedule_shift=0.25, blocks=[1, 3]).to(dtype=torch.bfloat16)
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        guidance_cfg=None,
        student_sample_type="ode",
        loss_config=SimpleNamespace(block_size_max=net.pdd_steps),
    )
    model.precision = torch.bfloat16
    model.precision_decoding = torch.float32
    model.precision_amp_infer = None

    # Every caller (trainer outputs, inference scripts, ComfyUI) hands the generator
    # method-precision noise, so the sampler never casts its network input.
    noise = torch.tensor([[0.5], [-1.25]], dtype=model.precision)
    observed_t_init = []
    observed_noise_dtype = []
    fusion_compute_dtypes = []
    state_dtypes = []
    original_latents = net.noise_scheduler.latents
    original_fused_params = net.pdd_linear._fused_params

    def capture_latents(noise, t_init):
        observed_t_init.append(t_init.detach().clone())
        observed_noise_dtype.append(noise.dtype)
        return original_latents(noise, t_init=t_init)

    def capture_fused_params(step_index, next_step_index, weights, precision_decoding):
        fusion_compute_dtypes.append(precision_decoding)
        return original_fused_params(step_index, next_step_index, weights, precision_decoding)

    monkeypatch.setattr(net.noise_scheduler, "latents", capture_latents)
    monkeypatch.setattr(net.pdd_linear, "_fused_params", capture_fused_params)
    handle = net.register_forward_pre_hook(lambda _module, args: state_dtypes.append(args[0].dtype))
    try:
        actual = model.generator_fn(net, noise, condition=None)
    finally:
        handle.remove()

    base_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=noise.device,
        dtype=net.noise_scheduler.t_precision,
    )
    float32_base_grid = base_grid.float()
    # The sampler keeps the caller's latent dtype; each block advance promotes
    # to decoding precision internally and restores the incoming dtype.
    expected = original_latents(noise, t_init=base_grid[0])
    heads = torch.arange(1, net.pdd_steps + 1, dtype=model.precision)
    fuse_weights = _interval_weights(net, base_grid)
    student_grid_indices = net.noise_scheduler.student_grid_indices(
        target_steps=net.noise_scheduler.default_student_steps
    ).tolist()
    for start_idx, end_idx in zip(student_grid_indices, student_grid_indices[1:]):
        block_fuse_weights = fuse_weights[start_idx:end_idx]
        normalized_weights = block_fuse_weights / block_fuse_weights.sum()
        fused = (
            normalized_weights.to(model.precision_decoding) * heads[start_idx:end_idx].to(model.precision_decoding)
        ).sum()
        fused = fused.to(model.precision).to(model.precision_decoding)
        expected = (
            expected.to(model.precision_decoding) + block_fuse_weights.sum().to(model.precision_decoding) * fused
        ).to(noise.dtype)
    torch.testing.assert_close(actual, expected)
    assert PDDModel.generator_fn is FastGenModel.generator_fn
    assert PDDModel._student_sample_loop is FastGenModel._student_sample_loop
    assert actual.dtype == noise.dtype
    assert len(observed_t_init) == 1
    assert observed_noise_dtype == [noise.dtype]
    assert state_dtypes == [model.precision] * net.noise_scheduler.default_student_steps
    assert fusion_compute_dtypes == [model.precision_decoding] * net.noise_scheduler.default_student_steps
    assert observed_t_init[0].dtype == torch.float64
    assert torch.equal(observed_t_init[0], base_grid[0])
    assert observed_t_init[0].item() != float32_base_grid[0].double().item()


def test_pdd_validation_noise_uses_the_same_native_precision_as_inference():
    net = _toy_pdd()
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = net
    model.use_ema = []
    model.precision = torch.bfloat16
    model.precision_decoding = model.precision

    outputs = model._get_outputs(torch.zeros(2, 1, dtype=torch.float32))

    assert outputs["input_rand"].dtype == model.precision
    assert outputs["gen_rand"].keywords["noise"] is outputs["input_rand"]


def test_one_step_generation_matches_integer_fused_mean_sigma_velocity_advance():
    """A one-step PDD generator applies the full-grid fused mean sigma-velocity once."""
    net = _toy_pdd(steps=1, schedule_shift=0.25)
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        guidance_cfg=None,
        student_sample_type="ode",
        loss_config=SimpleNamespace(block_size_max=net.pdd_steps),
    )
    model.precision = torch.float32
    model.precision_decoding = torch.float32
    model.precision_amp_infer = None
    noise = torch.tensor([[0.5], [-1.25]], dtype=torch.float32)

    via_generator = model.generator_fn(net, noise)
    base_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=noise.device,
        dtype=net.noise_scheduler.t_precision,
    )
    latents = net.noise_scheduler.latents(noise, t_init=base_grid[0])
    block_prediction = net(
        latents,
        0,
        fused=True,
        next_step_index=net.pdd_steps,
    )
    via_fused_advance = net.pdd_advance_blocks(
        latents,
        block_prediction,
        0,
        net.pdd_steps,
        base_grid,
        fused=True,
    )

    torch.testing.assert_close(via_generator, via_fused_advance)


# ---------------------------------------------------------------------------
# Scheduler-neutral method boundary: the affine mixin below intentionally uses
# an EDM scheduler and a transition whose current-state scale is not one. It
# is the smallest counterexample to the RF-only identity state update and
# exercises the scheduler-specific-mixin protocol that ``PDDMixin`` documents.
# ---------------------------------------------------------------------------

_AFFINE_STATE_SCALES = (0.5, 0.75, 0.8)
_AFFINE_PREDICTION_SCALES = (1.0, 2.0, -0.25)


class _AffineBase(FastGenNetwork):
    """Minimal network base used only to exercise ``make_pdd(mixin_cls=...)``."""

    def __init__(self, student_grid_cfg: StudentGridConfig):
        super().__init__(
            net_pred_type="eps",
            schedule_type="edm",
            student_grid_cfg=student_grid_cfg,
        )
        # The custom mixin does not widen this projection.  It remains a real
        # final-linear target so the factory receives an ordinary valid spec.
        self.proj = torch.nn.Linear(1, 1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        fwd_pred_type: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del t, condition, fwd_pred_type, kwargs
        return self.proj(x_t)


class _AffinePDDMixin:
    """Synthetic scheduler adapter implementing the documented neutral API.

    Each base-grid interval follows ``x[j+1] = a[j] * x[j] + b[j] * head[j]``.
    The block reduction composes those affine rules, so earlier head
    coefficients include the state scales of all later intervals.
    """

    pdd_base_grid = PDDMixin.pdd_base_grid

    def __init__(self, pdd_steps: int, *args, **kwargs):
        if pdd_steps != len(_AFFINE_STATE_SCALES):
            raise ValueError(f"the affine fixture requires {len(_AFFINE_STATE_SCALES)} PDD steps")
        super().__init__(*args, **kwargs)
        self.pdd_steps = pdd_steps
        self.register_buffer("_affine_a", torch.tensor(_AFFINE_STATE_SCALES, dtype=torch.float64))
        self.register_buffer("_affine_b", torch.tensor(_AFFINE_PREDICTION_SCALES, dtype=torch.float64))
        self.forward_calls: list[dict[str, Any]] = []
        self.advance_calls: list[dict[str, Any]] = []

    def _materialized_heads(self, state: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(1, self.pdd_steps + 1, device=state.device, dtype=state.dtype)
        offsets = offsets.view(1, self.pdd_steps, *((1,) * (state.ndim - 1)))
        return state[:, None] * 0.25 + offsets

    def _affine_coefficients(
        self,
        step_index: int,
        next_step_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            type(step_index) is not int
            or type(next_step_index) is not int
            or not 0 <= step_index < next_step_index <= self.pdd_steps
        ):
            raise ValueError(f"expected an integer nonempty block, got [{step_index}, {next_step_index})")

        state_scales = self._affine_a.to(device=device, dtype=dtype)
        prediction_scales = self._affine_b.to(device=device, dtype=dtype)
        coefficients = torch.zeros(self.pdd_steps, device=device, dtype=dtype)
        suffix_state_scale = torch.ones((), device=device, dtype=dtype)
        for index in range(next_step_index - 1, step_index - 1, -1):
            coefficients[index] = prediction_scales[index] * suffix_state_scale
            suffix_state_scale = state_scales[index] * suffix_state_scale

        prediction_scale = coefficients.sum()
        if prediction_scale == 0:
            raise ValueError("the affine fixture requires a nonzero normalized block coefficient")
        return suffix_state_scale, coefficients / prediction_scale, prediction_scale

    @staticmethod
    def _reduce_materialized(prediction: torch.Tensor, normalized_weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("n,bn...->b...", normalized_weights, prediction)

    def forward(
        self,
        x_t: torch.Tensor,
        step_index: int | torch.Tensor,
        condition: Any = None,
        fused: bool = False,
        next_step_index: int | torch.Tensor | None = None,
        precision_decoding: torch.dtype | None = None,
    ) -> torch.Tensor:
        del condition, precision_decoding
        self.forward_calls.append(
            {
                "step_index": step_index,
                "next_step_index": next_step_index,
                "fused": fused,
            }
        )
        heads = self._materialized_heads(x_t)
        if not fused:
            return heads

        if not isinstance(step_index, int) or not isinstance(next_step_index, int):
            raise TypeError("the focused affine fixture only needs shared integer fused blocks")
        _, normalized_weights, _ = self._affine_coefficients(
            step_index,
            next_step_index,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        return self._reduce_materialized(heads, normalized_weights)

    def pdd_advance_blocks(
        self,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        step_index: int,
        next_step_index: int,
        base_grid: torch.Tensor,
        fused: bool = False,
        allow_empty: bool = False,
        precision_decoding: torch.dtype | None = None,
    ) -> torch.Tensor:
        del base_grid
        if allow_empty:
            raise NotImplementedError("empty blocks are outside this focused affine fixture")
        input_dtype = x_t.dtype
        if precision_decoding is not None:
            x_t = x_t.to(precision_decoding)
            pred = pred.to(precision_decoding)
        state_scale, normalized_weights, prediction_scale = self._affine_coefficients(
            step_index,
            next_step_index,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        block_prediction = pred if fused else self._reduce_materialized(pred, normalized_weights)
        self.advance_calls.append(
            {
                "step_index": step_index,
                "next_step_index": next_step_index,
                "fused": fused,
                "state_scale": state_scale.detach().clone(),
            }
        )
        next_state = state_scale * x_t + prediction_scale * block_prediction
        return next_state.to(input_dtype)

    # This teacher hook mirrors the live method-facing signature. The focused
    # affine tests exercise sampling; dedicated teacher numerics remain covered
    # by the RF tests above.
    def pdd_teacher_target(self, state, step_index, base_grid, solver, predict_teacher):
        del solver
        return predict_teacher(state, base_grid[step_index], fwd_pred_type=self.net_pred_type)

    # These legacy RF seams must not be needed by scheduler-neutral PDD code.
    @staticmethod
    def pdd_block_weights(*args, **kwargs):
        raise AssertionError("PDD called the legacy RF block-weight seam")

    @staticmethod
    def pdd_advance_prediction(*args, **kwargs):
        raise AssertionError("PDD called the legacy additive RF advance seam")

    @staticmethod
    def pdd_flow_to_sigma_velocity(*args, **kwargs):
        raise AssertionError("PDD called an RF/flow residual seam")


class _MethodHarness(PDDModelMixin):
    """Uninitialized method host for exercising one scheduler-neutral sample step."""


def _make_affine_net():
    sampling = StudentGridConfig(
        steps=1,
        grid_type="native",
        grid_kwargs={"base_steps": len(_AFFINE_STATE_SCALES)},
    )
    spec = PDDSpec(final_linear="proj", channel_layout="channel_major")
    affine_cls = make_pdd(
        _AffineBase,
        spec,
        name="SyntheticAffinePDD",
        mixin_cls=_AffinePDDMixin,
    )
    return affine_cls(pdd_steps=len(_AFFINE_STATE_SCALES), student_grid_cfg=sampling), affine_cls, spec


def test_make_pdd_accepts_scheduler_specific_mixin_class():
    net, affine_cls, spec = _make_affine_net()

    assert isinstance(net, _AffinePDDMixin)
    assert isinstance(net, _AffineBase)
    assert affine_cls.PDD_SPEC is spec
    assert affine_cls.__mro__[:3] == (affine_cls, _AffinePDDMixin, _AffineBase)
    assert isinstance(net.noise_scheduler, EDMNoiseSchedule)
    assert net.pdd_base_grid().dtype == net.noise_scheduler.t_precision


def test_affine_materialized_and_fused_block_advances_match_with_state_scaling():
    net, _, _ = _make_affine_net()
    state = torch.tensor([[-1.25], [0.75]], dtype=torch.float64)
    base_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=state.device,
        dtype=net.noise_scheduler.t_precision,
    )
    step_index, next_step_index = 0, net.pdd_steps
    heads = net(state, step_index)

    materialized_state = net.pdd_advance_blocks(
        state,
        heads,
        step_index,
        next_step_index,
        base_grid,
    )
    fused_prediction = net(
        state,
        step_index,
        fused=True,
        next_step_index=next_step_index,
    )
    fused_state = net.pdd_advance_blocks(
        state,
        fused_prediction,
        step_index,
        next_step_index,
        base_grid,
        fused=True,
    )

    reference = state
    for index, (state_scale, prediction_scale) in enumerate(zip(_AFFINE_STATE_SCALES, _AFFINE_PREDICTION_SCALES)):
        reference = state_scale * reference + prediction_scale * heads[:, index]

    assert net.advance_calls[0]["state_scale"].item() != 1.0
    torch.testing.assert_close(materialized_state, reference)
    torch.testing.assert_close(fused_state, reference)


def test_pdd_sampling_method_uses_only_neutral_fused_transition(monkeypatch):
    net, _, _ = _make_affine_net()
    model = object.__new__(_MethodHarness)
    model.config = SimpleNamespace(loss_config=SimpleNamespace(block_size_max=net.pdd_steps))
    model.precision = torch.float64
    model.precision_decoding = torch.float64

    def forbidden_schedule_operation(*args, **kwargs):
        del args, kwargs
        raise AssertionError("PDD sampling called an RF/flow/sigma scheduler operation")

    # Grid construction and index selection are generic scheduler operations;
    # none of these semantic conversion/coordinate methods should be consulted.
    for name in ("flow_to_x0", "sigma", "sigma_prime"):
        monkeypatch.setattr(net.noise_scheduler, name, forbidden_schedule_operation)

    state = torch.tensor([[-0.5], [1.5]], dtype=torch.float64)
    base_grid = net.noise_scheduler.student_grid(
        net.pdd_steps,
        device=state.device,
        dtype=net.noise_scheduler.t_precision,
    )
    runtime_grid = base_grid[[0, -1]]
    expected_prediction = net._reduce_materialized(
        net._materialized_heads(state),
        net._affine_coefficients(0, net.pdd_steps, device=state.device, dtype=state.dtype)[1],
    )
    expected_state = net.pdd_advance_blocks(
        state,
        expected_prediction,
        0,
        net.pdd_steps,
        base_grid,
        fused=True,
    )
    net.advance_calls.clear()
    net.forward_calls.clear()

    actual = model._advance_student_step(
        net,
        state,
        step_index=0,
        student_grid=runtime_grid,
        condition="condition",
    )

    torch.testing.assert_close(actual, expected_state)
    assert len(net.forward_calls) == 1
    assert net.forward_calls[0]["fused"] is True
    assert len(net.advance_calls) == 1
    assert net.advance_calls[0]["fused"] is True
    assert net.advance_calls[0]["state_scale"].item() != 1.0
