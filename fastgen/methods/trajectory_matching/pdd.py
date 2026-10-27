# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel Decoding Distillation (PDD).

Handles both unimodal (plain tensor) and multimodal (dict-of-tensors) inputs.
Every data tensor passes through the per-modality helpers in
``fastgen.utils.tree_utils``, so a single code path serves both
``input_student = Tensor`` and
``input_student = {"video": Tensor, "audio": Tensor}``. The base and runtime
student grids, base-grid indices, and block masks are shared across modalities;
only states and predictions are dispatched per modality. Multimodal losses are
reported per modality and summed into ``total_loss``.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Any, TYPE_CHECKING, Callable

import attrs
import torch

from fastgen.methods import FastGenModel
from fastgen.methods.backward_simulation import (
    attach_trajectory_carry,
    resolve_trajectory_point,
)
from fastgen.methods.common_loss import mse_loss
from fastgen.utils.guidance import predict_with_guidance
from fastgen.utils.tensor import assert_tensor, precision_to_dtype
from fastgen.utils.tree_utils import (
    TensorOrDict,
    assert_same,
    detach,
    get_batch_size,
    get_device,
    index,
    per_modality,
    randn_like,
    to,
    to_dict,
)


if TYPE_CHECKING:
    from fastgen.configs.methods.config_pdd import ModelConfig
    from fastgen.networks.network import FastGenNetwork


@attrs.frozen(eq=False)
class PDDGridPlan:
    """A PDD base timestep grid and the base-grid indices of one runtime student grid."""

    base_grid: torch.Tensor
    student_grid_indices: torch.Tensor

    @property
    def student_grid(self) -> torch.Tensor:
        """Select runtime timesteps without changing the base grid's dtype, device, or autograd path."""
        return self.base_grid[self.student_grid_indices]


class PDDModelMixin:
    """PDD training and inference primitives shared by PDD-based methods."""

    # Fused intervals use deterministic integration.
    student_sample_types = ("ode",)

    def build_model(self):
        super().build_model()
        self._validate_pdd_rollout_contract()

    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): The configuration for the PDD model
        """
        block_size_min = config.loss_config.block_size_min
        if block_size_min <= 0:
            raise ValueError(f"block_size_min must be positive, got {block_size_min}.")
        block_size_max = config.loss_config.block_size_max
        if block_size_max <= 0:
            raise ValueError(f"block_size_max must be positive, got {block_size_max}.")
        if block_size_min > block_size_max:
            raise ValueError(f"block_size_min={block_size_min} must not exceed block_size_max={block_size_max}.")
        num_supervised_heads = config.loss_config.num_supervised_heads
        if num_supervised_heads <= 0:
            raise ValueError(f"num_supervised_heads must be positive, got {num_supervised_heads}.")
        super().__init__(config)
        self.precision_decoding = precision_to_dtype(
            config.precision_decoding,
            default=self.precision,
        )

    def _validate_pdd_rollout_contract(self) -> None:
        """Validate the fixed backward-simulation rollout against trained block sizes."""
        if not hasattr(self.net, "pdd_steps"):
            raise TypeError(f"{type(self).__name__} requires a PDD network with a pdd_steps attribute.")
        if self.config.backward_simulation is None:
            return

        num_steps = self._max_student_steps()
        student_grid_indices = self.net.noise_scheduler.student_grid_indices(
            target_steps=num_steps,
            device=torch.device("cpu"),
        )
        block_sizes = student_grid_indices[1:] - student_grid_indices[:-1]
        max_block_size = int(block_sizes.max().item())
        if max_block_size > self.config.loss_config.block_size_max:
            raise ValueError(
                f"Backward-simulation PDD student-grid block {max_block_size} exceeds the trained "
                f"block_size_max={self.config.loss_config.block_size_max}."
            )

    def _validate_pdd_end_idx(
        self,
        step_idx: torch.Tensor,
        end_idx: torch.Tensor,
        name: str = "end_idx",
    ) -> None:
        """Validate one batched, exclusive PDD block endpoint."""
        if end_idx.shape != step_idx.shape:
            raise ValueError(f"{name} must have shape {tuple(step_idx.shape)}, got {tuple(end_idx.shape)}")
        assert_tensor(
            torch.all((end_idx > step_idx) & (end_idx <= self.net.pdd_steps)),
            f"Each PDD block must satisfy step_idx < {name} <= pdd_steps",
            error_type=ValueError,
        )

    def _sample_pdd_supervision_idx(
        self,
        step_idx: torch.Tensor,
        block_end_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``[batch, num_supervised_heads]`` heads, unique per block.

        Supervision targets are deterministic, so a repeated head would only
        duplicate its loss term; a block holding fewer heads raises instead.
        Every valid block contains its start head, so K=1 never raises.
        """
        if block_end_idx is None:
            max_idx = torch.clamp(
                step_idx + self.config.loss_config.block_size_max - 1,
                max=self.net.pdd_steps - 1,
            )
        else:
            self._validate_pdd_end_idx(step_idx, block_end_idx, "block_end_idx")
            max_idx = block_end_idx - 1

        head_idx = torch.arange(self.net.pdd_steps, device=step_idx.device)[None]
        block_mask = (head_idx >= step_idx[:, None]) & (head_idx <= max_idx[:, None])
        num_heads = self.config.loss_config.num_supervised_heads
        assert_tensor(
            torch.all(block_mask.sum(dim=1) >= num_heads),
            f"num_supervised_heads={num_heads} exceeds a sampled block's head count",
            error_type=ValueError,
        )
        return torch.multinomial(block_mask.float(), num_heads, replacement=False)

    def _compute_pdd_loss_from_prediction(
        self,
        input_student: TensorOrDict,
        step_idx: torch.Tensor,
        pred: TensorOrDict,
        base_grid: torch.Tensor,
        condition: Any,
        neg_condition: Any,
        block_end_idx: torch.Tensor | None = None,
        next_idx: torch.Tensor | None = None,
        loss_scale: torch.Tensor | None = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        TensorOrDict | None,
    ]:
        """Supervise sampled heads of ``pred``, the forward from ``base_grid[step_idx]``.

        Each sampled head (an absolute base-grid index, optionally confined to
        the exclusive ``block_end_idx``) is matched against the teacher target
        at its grid node, reconstructed by the prefix heads; the per-head MSEs
        are averaged. One multi-endpoint ``pdd_advance_blocks`` call supplies
        all those states plus, when ``next_idx`` is given, the returned
        ``next_state`` for the caller's endpoint-pair x0 approximation. An
        optional per-sample ``loss_scale`` gates the loss without skipping any
        teacher forwards, which keeps FSDP collectives aligned across ranks.
        """
        supervision_idx = self._sample_pdd_supervision_idx(step_idx, block_end_idx)
        num_heads = supervision_idx.shape[1]
        advance_indices = supervision_idx
        if next_idx is not None:
            self._validate_pdd_end_idx(step_idx, next_idx, "next_idx")
            advance_indices = torch.cat([supervision_idx, next_idx[:, None]], dim=1)

        with torch.no_grad():
            states = self.net.pdd_advance_blocks(
                input_student,
                pred,
                step_idx,
                advance_indices,
                base_grid,
                allow_empty=True,
                precision_decoding=self.precision_decoding,
            )
        assert all(
            value.shape[1] == advance_indices.shape[1]
            for value in (states.values() if isinstance(states, dict) else (states,))
        ), "pdd_advance_blocks must stack one state per endpoint on dim 1"

        batch_idx = torch.arange(get_batch_size(input_student), device=get_device(input_student))
        head_losses = []
        for head in range(num_heads):
            head_step_idx = supervision_idx[:, head]
            head_pred = index(pred, (batch_idx, head_step_idx))
            head_state = index(states, (slice(None), head))

            with torch.no_grad():
                teacher_target = self.net.pdd_teacher_target(
                    head_state,
                    head_step_idx,
                    base_grid,
                    solver=self.config.loss_config.type_rk,
                    predict_teacher=partial(
                        self._guided_teacher,
                        condition=condition,
                        neg_condition=neg_condition,
                    ),
                )

            # PDD may deliberately decode the target in a wider dtype than the
            # prediction, but its per-modality shape and device must still agree.
            assert_same(
                shape=True,
                device=self.device,
                head_pred=head_pred,
                teacher_target=teacher_target,
            )
            head_losses.append(
                per_modality(
                    mse_loss,
                    head_pred,
                    teacher_target,
                    loss_scale=loss_scale,
                )
            )
        mse = per_modality(lambda *losses: torch.stack(losses).mean(), *head_losses)

        loss_map = to_dict(mse, suffix="pdd_loss")
        loss_map["total_loss"] = sum(loss_map.values())

        next_state = None
        if next_idx is not None:
            # Clone the selected column so a cross-iteration carry does not
            # keep the complete multi-endpoint allocation alive.
            next_state = per_modality(torch.clone, index(states, (slice(None), num_heads)))
        return loss_map, next_state

    def _guided_teacher(
        self,
        x_t: TensorOrDict,
        t: torch.Tensor,
        condition: Any,
        neg_condition: Any,
        fwd_pred_type: str,
        **fwd_kwargs,
    ) -> TensorOrDict:
        """Return one guided teacher prediction in the requested representation."""

        # ``skip_inactive=False`` keeps the set of forwards identical on every
        # rank so FSDP collectives cannot diverge between ranks.
        return predict_with_guidance(
            self.teacher,
            to(x_t, dtype=self.precision),
            t,
            guidance_cfg=self.config.guidance_cfg,
            condition=condition,
            neg_condition=neg_condition,
            skip_inactive=False,
            output_dtype=self.precision_decoding,
            fwd_pred_type=fwd_pred_type,
            **fwd_kwargs,
        )

    def _compute_loss(
        self,
        input_student: TensorOrDict,
        step_idx: torch.Tensor,
        base_grid: torch.Tensor,
        condition: Any,
        neg_condition: Any,
        next_idx: torch.Tensor,
    ):
        """Run the PDD heads once and return loss, x0 approximation, and the state at ``next_idx``."""
        pred = self.net(
            input_student,
            step_index=step_idx,
            condition=condition,
        )

        loss_map, next_state = self._compute_pdd_loss_from_prediction(
            input_student=input_student,
            step_idx=step_idx,
            pred=pred,
            base_grid=base_grid,
            condition=condition,
            neg_condition=neg_condition,
            next_idx=next_idx,
        )

        # The x0 preview inverts the endpoint pair through the noise scheduler.
        x0_approx = self.net.noise_scheduler.points_to_x0(
            detach(input_student),
            next_state,
            base_grid[step_idx],
            base_grid[next_idx],
        )
        return loss_map, x0_approx, next_state

    def _max_student_steps(self) -> int:
        """Return the largest student NFE PDD trains for, ``ceil(pdd_steps / block_size_min)``.

        Backward simulation carries trajectories on the student grid at this NFE,
        data-dependent training draws its starts from its nodes, and sampling at
        this NFE visits the same nodes. Sampling with more steps still runs while
        each block fits ``block_size_max``, but starts blocks from states training
        never draws. Divisibility is not required: the scheduler balances a
        non-divisible base grid across this many intervals.
        """
        block_size_min = self.config.loss_config.block_size_min
        return (self.net.pdd_steps + block_size_min - 1) // block_size_min

    def _pdd_student_grid_plan(
        self,
        net: FastGenNetwork,
        num_steps: int,
        device: torch.device,
    ) -> PDDGridPlan:
        """Return the anchored base grid and base-grid indices of one runtime student grid."""
        base_grid = net.pdd_base_grid(device=device)
        student_grid_indices = net.noise_scheduler.student_grid_indices(
            target_steps=num_steps,
            device=device,
        )
        return PDDGridPlan(base_grid=base_grid, student_grid_indices=student_grid_indices)

    def _advance_student_step(
        self,
        net: FastGenNetwork,
        state: TensorOrDict,
        step_index: int,
        student_grid: torch.Tensor,
        condition: Any = None,
    ) -> TensorOrDict:
        """Advance one explicit fused PDD block.

        PDD owns a fixed base grid, while ``step_index`` selects one interval
        of the supplied runtime ``student_grid``. Resolve that interval's two
        base-grid indices, request the native fused block prediction, then
        delegate the scheduler-specific state transition to the network.
        Autograd and module mode are inherited from the caller; staggered
        initialization and inference own those contexts at their call sites.
        """
        num_steps = len(student_grid) - 1
        state_device = get_device(state)
        block_size_max = self.config.loss_config.block_size_max
        if type(step_index) is not int or not 0 <= step_index < num_steps:
            raise ValueError(f"step_index must be an int in [0, {num_steps - 1}], got {step_index!r}.")

        grid_plan = self._pdd_student_grid_plan(
            net,
            num_steps=num_steps,
            device=state_device,
        )
        start_idx, end_idx = grid_plan.student_grid_indices[step_index : step_index + 2].tolist()
        block_size = end_idx - start_idx
        if block_size > block_size_max:
            raise ValueError(
                f"PDD sampling block {block_size} at step_index={step_index} exceeds the trained "
                f"block_size_max={block_size_max}."
            )

        block_prediction = net(
            state,
            step_index=start_idx,
            condition=condition,
            fused=True,
            next_step_index=end_idx,
            precision_decoding=self.precision_decoding,
        )
        return net.pdd_advance_blocks(
            state,
            block_prediction,
            start_idx,
            end_idx,
            grid_plan.base_grid,
            fused=True,
            precision_decoding=self.precision_decoding,
        )


class PDDModel(PDDModelMixin, FastGenModel):
    """Standalone PDD training method."""

    def build_model(self):
        super().build_model()
        self.load_student_weights_and_ema()
        self.build_teacher()

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, TensorOrDict | Callable]]:
        """
        Single training step for PDD model

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """
        condition = data.get("condition")
        neg_condition = data.get("neg_condition")
        batch_size = self._batch_size(data)
        grid_device = get_device(data["real"]) if self.config.backward_simulation is None else self.device
        num_steps = self._max_student_steps()
        grid_plan = self._pdd_student_grid_plan(self.net, num_steps, device=grid_device)

        if self.config.backward_simulation is None:
            real_data = data["real"]
            noise = randn_like(real_data)

            # Draw starts uniformly from the non-terminal student-grid nodes.
            student_step = torch.randint(num_steps, (batch_size,), device=self.device, dtype=torch.long)
            step_idx = grid_plan.student_grid_indices[student_step]
            input_student = self.net.noise_scheduler.forward_process(real_data, noise, grid_plan.base_grid[step_idx])

            # The x0 preview inverts from the supervised block's endpoint, which the
            # loss materializes as one extra column of its multi-endpoint advance.
            next_idx = torch.clamp(
                step_idx + self.config.loss_config.block_size_max,
                max=self.net.pdd_steps,
            )
        else:
            input_student, trajectory_step = resolve_trajectory_point(
                data,
                net=self.net,
                student_grid=grid_plan.student_grid,
                input_shape=self.input_shape,
                batch_size=batch_size,
                dtype=self.precision,
                condition=condition,
                advance_step=self._advance_student_step,
                iteration=iteration,
                update_period=1,
                owner="backward-simulation PDD",
                advance_empty=torch.is_grad_enabled(),
                advance_type=self.config.backward_simulation,
            )

            # The carried trajectory uses one runtime student-grid interval. PDD's
            # loss separately addresses a base-grid architectural head from this state.
            step_idx = grid_plan.student_grid_indices[trajectory_step].expand(batch_size)
            next_idx = grid_plan.student_grid_indices[trajectory_step + 1].expand(batch_size)

        # Materialize PDD heads once. For backward simulation the exact runtime
        # interval is also integrated under no_grad to supply the carry.
        loss_map, x0_approx, next_state = self._compute_loss(
            input_student,
            step_idx,
            grid_plan.base_grid,
            condition,
            neg_condition,
            next_idx=next_idx,
        )

        outputs = self._get_outputs(
            x0_approx,
            condition=condition,
            neg_condition=neg_condition,
        )
        if self.config.backward_simulation is not None:
            attach_trajectory_carry(
                outputs,
                data,
                current_step=trajectory_step,
                num_steps=num_steps,
                next_state=next_state,
                advance_type=self.config.backward_simulation,
            )

        loss_map = self.apply_reward(x0_approx, data, loss_map)
        return loss_map, outputs
