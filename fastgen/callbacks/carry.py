# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from fastgen.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


# Output dictionary key consumed by ``CarryCallback``.
CARRY_OUTPUT_KEY = "carry"

# Canonical batch data that a method can include with carried state when the two
# must stay paired. The method decides when an overlay is appropriate; the
# callback transports the resulting dictionary atomically.
CARRY_OVERLAY_KEYS = (
    "condition",
    "neg_condition",
    "real",
    "condition_raw",
    "neg_condition_raw",
    "condition_image",
    "condition_image_raw",
    "camera_poses",
    "camera_intrinsics",
    "idx",
)


class CarryCallback(Callback):
    """Carry one atomic output dictionary along training trajectories.

    One independent slot per gradient-accumulation step: the callback captures
    ``output_batch[CARRY_OUTPUT_KEY]`` after each accum step and atomically
    replaces the next preprocessed batch in the same slot with it (a
    state-free terminal marker is overlaid instead). Methods include
    ``CARRY_OVERLAY_KEYS`` fields when they must stay paired with the carried
    state. Validation carry is opt-in with ``enable_validation=True``; by
    default, every validation batch remains an independent sample. The optional
    validation slot is reset around every pass, so trajectories never leak
    across validation configurations or checkpoints.

    Every rank still loads and preprocesses a fresh batch each step -- even
    when an active carry then discards it -- keeping sampler resume accounting
    and sharded preprocessing rank-synchronous. Carry state is deliberately
    not checkpointed: resumed slots start empty and methods rebuild the phase
    encoded by the resumed iteration from fresh data, preserving the per-rank
    phase schedule but not the exact in-flight trajectory.
    """

    def __init__(self, enable_validation: bool = False):
        self.enable_validation = enable_validation
        self._slots_dict: dict[int, dict[str, Any]] = {}
        self._validation_slot: dict[str, Any] | None = None

    @staticmethod
    def _overlay_carry(data_batch: dict[str, Any], slot: dict[str, Any] | None) -> None:
        if slot is None:
            return
        # An active slot is the complete effective trajectory batch. Drop all
        # fields from the newly preprocessed batch before restoring it so new
        # prompt/sample metadata cannot leak into an in-flight trajectory.
        if "state_carried" in slot:
            data_batch.clear()
        data_batch.update(slot)

    @staticmethod
    def _capture_carry(output_batch: dict[str, Any]) -> dict[str, Any] | None:
        carry = output_batch.get(CARRY_OUTPUT_KEY)
        if carry is None:
            return None
        if not isinstance(carry, dict):
            raise TypeError(f"output_batch[{CARRY_OUTPUT_KEY!r}] must be a dict or None, got {type(carry).__name__}.")
        return dict(carry)

    def on_training_accum_step_begin(
        self,
        model: FastGenModel,
        data_batch: dict[str, Any],
        iteration: int = 0,
        accum_iter: int = 0,
    ) -> None:
        self._overlay_carry(data_batch, self._slots_dict.get(accum_iter))

    def on_training_accum_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        iteration: int = 0,
        accum_iter: int = 0,
    ) -> None:
        carry = self._capture_carry(output_batch)
        if carry is None:
            self._slots_dict.pop(accum_iter, None)
            return

        # Replace the complete slot rather than updating it. An active slot is
        # the complete trajectory batch; a state-free terminal marker
        # deliberately accompanies the next fresh preprocessed batch.
        self._slots_dict[accum_iter] = carry

    def on_validation_begin(self, model: FastGenModel, iteration: int = 0, idx: int = 0) -> None:
        self._validation_slot = None

    def on_validation_step_begin(
        self,
        model: FastGenModel,
        data_batch: dict[str, Any],
        step: int = 0,
        iteration: int = 0,
        idx: int = 0,
    ) -> None:
        if not self.enable_validation:
            return
        self._overlay_carry(data_batch, self._validation_slot)

    def on_validation_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss_dict: dict[str, Any],
        step: int = 0,
        iteration: int = 0,
        idx: int = 0,
    ) -> None:
        if not self.enable_validation:
            return
        self._validation_slot = self._capture_carry(output_batch)

    def on_validation_end(self, model: FastGenModel, iteration: int = 0, idx: int = 0) -> None:
        self._validation_slot = None

    def state_dict(self) -> dict[str, Any]:
        # Carry state is transient in-flight sampling state and is intentionally
        # not persisted; the method reconstructs it as needed on resume.
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._slots_dict = {}
        self._validation_slot = None
