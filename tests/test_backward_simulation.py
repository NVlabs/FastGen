# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import fastgen.methods.backward_simulation as backward_simulation
from fastgen.callbacks.carry import CARRY_OUTPUT_KEY
from fastgen.methods.backward_simulation import (
    attach_trajectory_carry,
    resolve_trajectory_point,
)


class _Schedule:
    def __init__(self, num_steps=4, events=None):
        self.num_steps = num_steps
        self.events = [] if events is None else events

    def student_grid(self, num_steps=None, device=None, dtype=None):
        steps = self.num_steps if num_steps is None else num_steps
        return torch.linspace(0.999, 0.0, steps + 1, device=device, dtype=dtype or torch.float32)

    def latents(self, noise, t_init):
        del t_init
        self.events.append(("initialize", torch.is_grad_enabled()))
        return torch.zeros_like(noise)


class _Net(torch.nn.Module):
    def __init__(self, schedule):
        super().__init__()
        self.noise_scheduler = schedule


def _identity_step(net, state, step, student_grid, condition):
    del net, step, student_grid, condition
    return state


def _resolver_kwargs(num_steps=4, batch_size=2, events=None, advance_step=_identity_step):
    schedule = _Schedule(num_steps=num_steps, events=events)
    net = _Net(schedule)
    return schedule, {
        "net": net,
        "student_grid": schedule.student_grid(),
        "input_shape": [1],
        "batch_size": batch_size,
        "dtype": torch.float32,
        "condition": None,
        "advance_step": advance_step,
        "iteration": 1,
        "update_period": 1,
        "owner": "test",
    }


def test_resolve_trajectory_point_returns_active_carry_without_initializing():
    state = {"video": torch.ones(2, 1), "audio": torch.zeros(2, 1)}

    def unexpected(*args, **kwargs):
        raise AssertionError("an active carry must not advance")

    schedule, kwargs = _resolver_kwargs(advance_step=unexpected)
    resolved_state, step = resolve_trajectory_point(
        {"step_carried": 2, "state_carried": state},
        **kwargs,
    )

    assert step == 2 and resolved_state is state
    assert schedule.events == []


def test_resolve_trajectory_point_normalizes_boundary_and_unstaggered_empty_slot():
    def unexpected(*args, **kwargs):
        raise AssertionError("a boundary or unstaggered slot must not advance")

    events = []
    _, kwargs = _resolver_kwargs(events=events, advance_step=unexpected)
    boundary_state, boundary_step = resolve_trajectory_point(
        {"step_carried": 4},
        **{**kwargs, "iteration": 0},
    )
    empty_state, empty_step = resolve_trajectory_point(
        {},
        **{**kwargs, "iteration": 0, "advance_empty": False},
    )

    assert boundary_step == empty_step == 0
    assert events == [("initialize", True), ("initialize", True)]
    assert not boundary_state.requires_grad and not empty_state.requires_grad


@pytest.mark.parametrize(
    ("carry", "error"),
    [
        ({"state_carried": torch.zeros(1)}, ValueError),
        ({"step_carried": True}, TypeError),
        ({"step_carried": 0}, ValueError),
        ({"step_carried": 2}, ValueError),
        ({"step_carried": 4, "state_carried": torch.zeros(1)}, ValueError),
    ],
)
def test_resolve_trajectory_point_rejects_malformed_slots(carry, error):
    _, kwargs = _resolver_kwargs()
    with pytest.raises(error):
        resolve_trajectory_point(carry, **kwargs)


def test_resolve_trajectory_point_runs_collective_safe_graph_free_eval_stagger(monkeypatch):
    events = []
    monkeypatch.setattr(backward_simulation, "get_rank", lambda: 2)
    monkeypatch.setattr(backward_simulation, "world_size", lambda: 4)

    def advance(net, state, step, student_grid, condition):
        del student_grid, condition
        events.append(("advance", torch.is_grad_enabled(), net.training))
        next_state = state + step + 1
        return next_state

    _, kwargs = _resolver_kwargs(events=events, advance_step=advance)
    net = kwargs["net"]
    net.train()
    state, step = resolve_trajectory_point({}, **kwargs)

    assert events == [("initialize", True), *(("advance", False, False) for _ in range(3))]
    assert net.training
    assert step == 2
    torch.testing.assert_close(state, torch.full_like(state, 3.0))
    assert not state.requires_grad and state.grad_fn is None


def test_staggered_start_retains_the_original_complete_warmup(monkeypatch):
    events = []
    monkeypatch.setattr(backward_simulation, "get_rank", lambda: 0)
    monkeypatch.setattr(backward_simulation, "world_size", lambda: 2)

    def advance(net, state, step, student_grid, condition):
        del student_grid, condition
        events.append((step, torch.is_grad_enabled(), net.training))
        return state + step + 1

    _, kwargs = _resolver_kwargs(events=events, advance_step=advance)
    net = kwargs["net"]
    net.train()
    state, step = resolve_trajectory_point({}, **kwargs)

    # Rank zero selects the initial state, but every rank still executes the
    # old num_steps - 1 warmup forwards to preserve baseline behavior exactly.
    assert step == 0
    assert events == [
        ("initialize", True),
        (0, False, False),
        (1, False, False),
        (2, False, False),
    ]
    torch.testing.assert_close(state, torch.zeros_like(state))
    assert net.training


def test_resolve_trajectory_point_can_disable_rank_stagger_and_preserve_common_resume_phase(monkeypatch):
    active_rank = 0
    monkeypatch.setattr(backward_simulation, "get_rank", lambda: active_rank)
    monkeypatch.setattr(backward_simulation, "world_size", lambda: 4)

    for rank in range(4):
        active_rank = rank
        events = []

        def advance(net, state, step, student_grid, condition):
            del student_grid, condition
            events.append(("advance", torch.is_grad_enabled(), net.training))
            return state + step + 1

        _, kwargs = _resolver_kwargs(events=events, advance_step=advance)
        net = kwargs["net"]
        net.train()
        state, step = resolve_trajectory_point(
            {},
            **{**kwargs, "iteration": 3},
            advance_type="synced",
        )

        # With two iterations already completed, every rank reconstructs the
        # common rank-zero phase two. All ranks execute the same minimal prefix,
        # keeping FSDP collectives aligned without the rank-stagger warmup.
        assert step == 2
        assert events == [("initialize", True), *(("advance", False, False) for _ in range(2))]
        torch.testing.assert_close(state, torch.full_like(state, 3.0))
        assert net.training
        assert not state.requires_grad and state.grad_fn is None


def test_unstaggered_initial_iteration_starts_at_zero_without_warmup(monkeypatch):
    monkeypatch.setattr(backward_simulation, "get_rank", lambda: 3)
    monkeypatch.setattr(backward_simulation, "world_size", lambda: 8)

    def unexpected(*args, **kwargs):
        raise AssertionError("the common initial phase must not warm up")

    events = []
    _, kwargs = _resolver_kwargs(events=events, advance_step=unexpected)
    state, step = resolve_trajectory_point({}, **kwargs, advance_type="synced")

    assert step == 0
    assert events == [("initialize", True)]
    torch.testing.assert_close(state, torch.zeros_like(state))


@pytest.mark.parametrize("value", [None, True, False, 0, "stagger", "none"])
def test_resolve_trajectory_point_rejects_unknown_advance_types(value):
    _, kwargs = _resolver_kwargs()
    with pytest.raises(ValueError, match="advance_type must be one of"):
        resolve_trajectory_point({}, **kwargs, advance_type=value)


def test_default_mode_starts_fresh_at_a_random_step_with_rank_uniform_warmup(monkeypatch):
    monkeypatch.setattr(backward_simulation, "get_rank", lambda: 1)
    monkeypatch.setattr(backward_simulation, "world_size", lambda: 4)
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([2]))
    # Another rank drew step 3, so the MAX all-reduce bounds every rank's walk.
    monkeypatch.setattr(backward_simulation, "sync_max", lambda value, device: max(value, 3))

    events = []

    def advance(net, state, step, student_grid, condition):
        del student_grid, condition
        events.append(("advance", torch.is_grad_enabled(), net.training))
        return state + step + 1

    _, kwargs = _resolver_kwargs(events=events, advance_step=advance)
    net = kwargs["net"]
    net.train()
    state, step = resolve_trajectory_point({}, **kwargs, advance_type="default")

    # Multi-rank runs execute the same synced-maximum warmup so per-rank
    # random starting steps cannot desynchronize FSDP collectives.
    assert step == 2
    assert events == [("initialize", True), *(("advance", False, False) for _ in range(3))]
    torch.testing.assert_close(state, torch.full_like(state, 3.0))
    assert net.training and not state.requires_grad


def test_default_mode_rejects_a_carried_trajectory():
    _, kwargs = _resolver_kwargs()
    with pytest.raises(ValueError, match="does not consume a carried trajectory"):
        resolve_trajectory_point(
            {"step_carried": 2, "state_carried": torch.zeros(1)},
            **kwargs,
            advance_type="default",
        )


def test_default_mode_validation_starts_at_zero_without_advancing():
    def unexpected(*args, **kwargs):
        raise AssertionError("validation must not advance")

    _, kwargs = _resolver_kwargs(advance_step=unexpected)
    state, step = resolve_trajectory_point(
        {},
        **{**kwargs, "advance_empty": False},
        advance_type="default",
    )
    assert step == 0 and not state.requires_grad


def test_resolve_trajectory_point_validates_phase_coverage_and_keeps_single_rank_fast_path(monkeypatch):
    monkeypatch.setattr(backward_simulation, "get_rank", lambda: 0)
    monkeypatch.setattr(backward_simulation, "world_size", lambda: 1)

    _, kwargs = _resolver_kwargs()
    with pytest.raises(ValueError, match="cannot cover every student timestep"):
        resolve_trajectory_point({}, **{**kwargs, "update_period": 2})

    # One shared synced phase cannot cover multiple residue classes.
    with pytest.raises(ValueError, match="synced.*cannot cover every student timestep"):
        resolve_trajectory_point(
            {},
            **{**kwargs, "update_period": 2},
            advance_type="synced",
        )

    # A coprime update period still visits every phase and keeps the common
    # initial position-zero fast path.
    state, step = resolve_trajectory_point(
        {},
        **{**kwargs, "update_period": 3},
        advance_type="synced",
    )
    assert step == 0 and not state.requires_grad

    def unexpected(*args, **kwargs):
        raise AssertionError("must not advance")

    _, kwargs = _resolver_kwargs(advance_step=unexpected)
    state, step = resolve_trajectory_point({}, **kwargs)
    assert step == 0 and not state.requires_grad


def test_staggered_requires_equal_rank_coverage_of_update_phase_classes(monkeypatch):
    monkeypatch.setattr(backward_simulation, "get_rank", lambda: 0)
    active_world_size = 3
    monkeypatch.setattr(backward_simulation, "world_size", lambda: active_world_size)
    _, kwargs = _resolver_kwargs(num_steps=4)

    # Two update-phase residue classes cannot be weighted equally by three ranks.
    with pytest.raises(ValueError, match="world_size must be divisible by 2"):
        resolve_trajectory_point({}, **{**kwargs, "update_period": 2})

    # One rank per residue class is sufficient even though the update period is not coprime.
    active_world_size = 2
    state, step = resolve_trajectory_point({}, **{**kwargs, "update_period": 2})
    assert step == 0 and not state.requires_grad


def test_attach_trajectory_carry_owns_position_advancement_and_output_attachment():
    state = {"video": torch.ones(2, 1, requires_grad=True)}
    condition = object()
    data = {"condition": condition, "condition_raw": ["a", "b"], "unrelated": object()}
    outputs = {}

    attach_trajectory_carry(
        outputs,
        data,
        current_step=0,
        num_steps=2,
        next_state=state,
    )
    active = outputs[CARRY_OUTPUT_KEY]
    assert active["condition"] is condition
    assert active["condition_raw"] == ["a", "b"]
    assert active["step_carried"] == 1
    assert "unrelated" not in active
    assert not active["state_carried"]["video"].requires_grad

    terminal_outputs = {}
    attach_trajectory_carry(
        terminal_outputs,
        data,
        current_step=1,
        num_steps=2,
    )
    terminal = terminal_outputs[CARRY_OUTPUT_KEY]
    assert terminal == {"step_carried": 2}

    disabled_outputs = {}
    attach_trajectory_carry(
        disabled_outputs,
        data,
        current_step=0,
        num_steps=2,
        next_state=state,
        advance_type="default",
    )
    assert CARRY_OUTPUT_KEY not in disabled_outputs
    for current_step, num_steps in [(-1, 2), (2, 2), (0, 0)]:
        with pytest.raises(ValueError, match="current_step must be"):
            attach_trajectory_carry(
                {},
                data,
                current_step=current_step,
                num_steps=num_steps,
                next_state=state,
                advance_type="default",
            )
    with pytest.raises(ValueError, match="advance_type must be one of"):
        attach_trajectory_carry(
            {},
            data,
            current_step=0,
            num_steps=2,
            next_state=state,
            advance_type=True,
        )


def test_trajectory_carry_preserves_paired_camera_metadata():
    from fastgen.callbacks.carry import CarryCallback

    poses = torch.eye(4).reshape(1, 1, 4, 4)
    intrinsics = torch.ones(1, 1, 4)
    data = {"camera_poses": poses, "camera_intrinsics": intrinsics}
    outputs = {}
    attach_trajectory_carry(outputs, data, current_step=0, num_steps=2, next_state=torch.ones(1, 1))
    callback = CarryCallback()
    callback.on_training_accum_step_end(None, data, outputs, iteration=1)

    fresh_batch = {"camera_poses": torch.zeros_like(poses), "camera_intrinsics": torch.zeros_like(intrinsics)}
    callback.on_training_accum_step_begin(None, fresh_batch, iteration=2)
    assert fresh_batch["camera_poses"] is poses
    assert fresh_batch["camera_intrinsics"] is intrinsics

    terminal = {}
    attach_trajectory_carry(terminal, fresh_batch, current_step=1, num_steps=2)
    callback.on_training_accum_step_end(None, fresh_batch, terminal, iteration=2)
    new_poses = torch.zeros_like(poses)
    next_batch = {"camera_poses": new_poses, "camera_intrinsics": torch.zeros_like(intrinsics)}
    callback.on_training_accum_step_begin(None, next_batch, iteration=3)
    assert next_batch["camera_poses"] is new_poses
