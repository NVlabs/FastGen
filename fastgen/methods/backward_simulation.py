# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared state machine for trajectories carried between iterations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
import math
from typing import Any, TYPE_CHECKING

import torch

from fastgen.callbacks.carry import CARRY_OUTPUT_KEY, CARRY_OVERLAY_KEYS
from fastgen.utils import basic_utils
from fastgen.utils.distributed import get_rank, world_size
from fastgen.utils.distributed.collectives import sync_max
from fastgen.utils.tree_utils import TensorOrDict, detach, noise_from_shape


if TYPE_CHECKING:
    from fastgen.networks.network import FastGenNetwork


ADVANCE_TYPES = ("default", "staggered", "synced")


def resolve_trajectory_point(
    data: Mapping[str, Any],
    net: FastGenNetwork,
    student_grid: torch.Tensor,
    input_shape: Any,
    batch_size: int,
    dtype: torch.dtype,
    condition: Any,
    advance_step: Callable[..., TensorOrDict],
    iteration: int,
    update_period: int,
    owner: str = "backward simulation",
    advance_empty: bool = True,
    advance_type: str = "staggered",
) -> tuple[TensorOrDict, int]:
    """Return the student state and ordinal step for this iteration.

    ``advance_type`` selects how a fresh trajectory picks its starting
    step. ``"staggered"`` and ``"synced"`` carry one trajectory across
    iterations: ``step_carried`` always names an ordinal trajectory node,
    interior nodes resume their state directly, a terminal boundary starts a
    fresh trajectory at step zero, and a completely empty slot also starts
    fresh and, when ``advance_empty``, walks the new state to its iteration
    phase — offset by distributed rank under ``"staggered"``, on the common
    rank-zero phase under ``"synced"``. ``"default"`` carries nothing: every
    call starts fresh and, when ``advance_empty``, walks to a per-rank random
    starting step.

    Fresh noise remains lazy: an active carry neither allocates it nor perturbs
    the random-number stream. Only the fresh-state walk runs under
    ``no_grad``; other work inherits the caller's autograd mode.
    """
    if advance_type not in ADVANCE_TYPES:
        raise ValueError(f"advance_type must be one of {ADVANCE_TYPES}, got {advance_type!r}.")

    num_steps = len(student_grid) - 1
    state = data.get("state_carried")
    position = data.get("step_carried")

    if advance_type == "default":
        if state is not None or position is not None:
            raise ValueError(
                f"{owner} with advance_type='default' does not consume a carried trajectory; "
                "remove CarryCallback or use 'staggered'/'synced'."
            )
    # No position denotes an empty carry slot, which cannot already contain a state.
    elif position is None:
        if state is not None:
            raise ValueError(f"{owner} carry has a latent but no step.")
    # A present position must be an integer trajectory-node index.
    elif not isinstance(position, int) or isinstance(position, bool):
        raise TypeError(f"{owner} carry step must be an int, got {type(position).__name__}.")
    # Carried positions name an interior node or the terminal boundary.
    elif not 1 <= position <= num_steps:
        raise ValueError(f"Invalid carried {owner} step {position}; expected 1..{num_steps}.")
    # The terminal marker starts a fresh trajectory and therefore carries no state.
    elif position == num_steps:
        if state is not None:
            raise ValueError(f"A completed {owner} trajectory still carries a latent.")
    # Every remaining valid position is an active trajectory that resumes directly.
    else:
        if state is None:
            raise ValueError(f"{owner} step {position} has no carried latent.")
        return state, position

    noise = noise_from_shape(input_shape, batch_size, student_grid.device, dtype)
    state = net.noise_scheduler.latents(noise=noise, t_init=student_grid[0])
    # Only a genuinely fresh start reconstructs its phase; terminal and validation starts stay at step zero.
    if position is None and advance_empty:
        return _trajectory_start(
            state,
            net=net,
            student_grid=student_grid,
            condition=condition,
            num_steps=num_steps,
            iteration=iteration,
            update_period=update_period,
            rank=get_rank(),
            num_ranks=world_size(),
            advance_type=advance_type,
            advance_step=advance_step,
            owner=owner,
        )
    return state, 0


@torch.no_grad()
def _trajectory_start(
    initial_state: TensorOrDict,
    net: FastGenNetwork,
    student_grid: torch.Tensor,
    condition: Any,
    num_steps: int,
    iteration: int,
    update_period: int,
    rank: int,
    num_ranks: int,
    advance_type: str,
    advance_step: Callable[..., TensorOrDict],
    owner: str = "backward simulation",
) -> tuple[TensorOrDict, int]:
    """Walk a fresh trajectory to its retained starting step.

    ``"staggered"`` selects a rank-offset iteration phase and retains the
    original complete ``num_steps - 1`` eval warmup on every rank. ``"default"``
    selects a per-rank random step and synchronizes the largest selected prefix.
    ``"synced"`` keeps all ranks on the common rank-zero iteration phase and
    executes exactly that prefix. The network's prior mode is restored afterward.
    """
    if update_period <= 0:
        raise ValueError(f"update_period must be positive, got {update_period!r}.")
    if iteration < 1:
        raise ValueError(f"iteration must be positive, got {iteration!r}.")
    if not 0 <= rank < num_ranks:
        raise ValueError(f"rank must be in [0, {num_ranks - 1}], got {rank!r}.")

    if advance_type == "staggered":
        phase_classes = math.gcd(num_steps, update_period)
        if num_ranks % phase_classes != 0:
            raise ValueError(
                f"{owner} cannot cover every student timestep uniformly with num_steps={num_steps}, "
                f"update_period={update_period}, and world_size={num_ranks}. Student updates preserve "
                f"the trajectory step modulo gcd(num_steps, update_period)={phase_classes}, so "
                f"world_size must be divisible by {phase_classes} to weight every phase class equally. "
                "Use a divisible world size or choose a coprime update period."
            )
        offset = (rank + iteration - 1) % num_steps
    elif advance_type == "synced":
        phase_classes = math.gcd(num_steps, update_period)
        if phase_classes > 1:
            raise ValueError(
                f"{owner} with advance_type='synced' cannot cover every student timestep with "
                f"num_steps={num_steps} and update_period={update_period}. Student updates preserve "
                f"the trajectory step modulo gcd(num_steps, update_period)={phase_classes}, but "
                "synced mode has only one shared phase. Choose a coprime update period or use "
                "advance_type='staggered'."
            )
        offset = (iteration - 1) % num_steps
    elif advance_type == "default":
        offset = int(torch.randint(num_steps, (1,)).item())
    else:
        raise ValueError(f"advance_type must be one of {ADVANCE_TYPES}, got {advance_type!r}.")

    # Keep FSDP collectives aligned by walking the same prefix on every rank.
    # Staggered mode deliberately retains its original complete warmup; synced
    # ranks share one phase and need only its exact prefix; independent random
    # draws need a MAX all-reduce (aligned, since "default" starts fresh on
    # every rank each call).
    if advance_type == "synced":
        rollout_steps = offset
    elif advance_type == "staggered":
        rollout_steps = 0 if num_ranks == 1 and offset == 0 else num_steps - 1
    elif num_ranks == 1:
        rollout_steps = offset
    else:
        rollout_steps = sync_max(offset, device=student_grid.device)

    selected_state = initial_state
    if rollout_steps == 0:
        return selected_state, offset

    state = initial_state
    with basic_utils.train_mode(net, mode=False):
        for step in range(rollout_steps):
            state = advance_step(
                net,
                state,
                step,
                student_grid,
                condition=condition,
            )
            if step + 1 == offset:
                selected_state = state
    # The no_grad walk may cache detached autocast casts of trainable weights; the caller's gradient
    # forward of the same network must not reuse them.
    torch.clear_autocast_cache()
    return selected_state, offset


def attach_trajectory_carry(
    outputs: MutableMapping[str, Any],
    data: Mapping[str, Any],
    current_step: int,
    num_steps: int,
    next_state: TensorOrDict | None = None,
    advance_type: str = "staggered",
) -> None:
    """Advance, package, and attach one trajectory carry atomically.

    The final step emits only a state-free boundary marker and therefore does
    not require ``next_state``. ``"default"`` carries nothing.
    """
    if advance_type not in ADVANCE_TYPES:
        raise ValueError(f"advance_type must be one of {ADVANCE_TYPES}, got {advance_type!r}.")
    if not 0 <= current_step < num_steps:
        raise ValueError(f"current_step must be in [0, {num_steps - 1}], got {current_step}.")
    if advance_type == "default":
        return

    next_position = current_step + 1
    if next_position == num_steps:
        outputs[CARRY_OUTPUT_KEY] = {"step_carried": num_steps}
        return
    if next_state is None:
        raise ValueError(f"Active trajectory position {next_position} has no next state.")

    carry = {key: data[key] for key in CARRY_OVERLAY_KEYS if key in data}
    carry.update({"state_carried": detach(next_state), "step_carried": next_position})
    outputs[CARRY_OUTPUT_KEY] = carry
