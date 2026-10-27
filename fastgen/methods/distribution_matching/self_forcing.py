# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TYPE_CHECKING, List, Optional

import torch

from fastgen.methods import CausVidModel
from fastgen.methods.distribution_matching.dmd2 import Generation, TrainingInputs

import fastgen.utils.logging_utils as logger
from fastgen.networks.network_causal import (
    CausalContext,
    CausalFastGenNetwork,
    causal_chunk_ranges,
    check_chunk_boundary,
)
from fastgen.utils.distributed.collectives import broadcast_choice
from fastgen.utils.tree_utils import noise_from_shape

if TYPE_CHECKING:
    from fastgen.configs.methods.config_self_forcing import ModelConfig


class SelfForcingModel(CausVidModel):
    """Causal distribution matching with an in-iteration self-forcing rollout.

    ``SelfForcingModel`` inherits DMD2's losses but generates ``gen_data`` with
    an in-iteration causal rollout. Each causal block samples an exit denoising
    step, with gradients enabled there according to the rollout configuration.

    Self-Forcing requires ``backward_simulation="default"`` to select fresh-noise,
    real-data-optional input construction. Unlike base DMD2 backward simulation,
    it completes the rollout within the current iteration and returns no carried
    trajectory state. Real data is used only by inherited optional objectives,
    such as a positively weighted GAN loss.
    """

    def __init__(self, config: ModelConfig):
        if config.backward_simulation != "default":
            raise ValueError(
                "SelfForcingModel requires backward_simulation='default': it always starts from fresh noise "
                "and performs its complete causal rollout within the current iteration, carrying nothing."
            )
        super().__init__(config)

    def _training_inputs(self, data: dict[str, Any], iteration: int = 1) -> TrainingInputs:
        """Construct fresh-noise inputs for Self-Forcing's in-iteration rollout.

        Shape and dtype come from the model configuration, so this path does
        not depend on real data. No trajectory is carried between iterations.
        The shared VSD/GAN noising pair always uses DMD2's main distribution;
        an optional fake-score-only pair is sampled inside the auxiliary step.
        """
        del iteration
        batch_size = self._batch_size(data)
        noise = noise_from_shape(self.input_shape, batch_size, self.device, self.precision)
        student_grid = self.net.noise_scheduler.student_grid(device=self.device)
        input_student = self.net.noise_scheduler.latents(noise=noise, t_init=student_grid[0])
        t, eps = self._sample_t_and_eps(batch_size)
        return TrainingInputs(input_student=input_student, student_step=None, t=t, eps=eps)

    def _sample_exit_steps(self, num_blocks: int) -> List[int]:
        """Sample one exit-step index per block."""
        sample_steps = self.net.noise_scheduler.default_student_steps
        if self.config.last_step_only:
            return [sample_steps - 1] * num_blocks

        if self.config.same_step_across_blocks:
            return [broadcast_choice(sample_steps, device=self.device)] * num_blocks

        exit_steps = broadcast_choice(sample_steps, num_blocks, device=self.device)
        return [exit_steps] if num_blocks == 1 else exit_steps

    def gen_data_from_net(
        self,
        input_student: torch.Tensor,
        student_step: Optional[torch.Tensor],
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> Generation:
        """Generate data with Self-Forcing's in-iteration causal rollout.

        No external KV cache is used. Instead, ``_store_causal_chunk`` writes every
        completed block but the last to the model's internal caches, under no_grad
        unless ``grad_through_kv_cache``.

        ``student_step`` is intentionally absent: Self-Forcing starts from fresh
        noise and builds a complete per-block student trajectory here.
        """
        del student_step, neg_condition

        assert isinstance(self.net, CausalFastGenNetwork), f"{self.net} must be a CausalFastGenNetwork"
        self.net.clear_caches()

        _, _, num_frames, _, _ = input_student.shape
        # Training layout: the first block holds any remainder frames; fewer frames than a chunk form one block.
        blocks = causal_chunk_ranges(num_frames, self.net.chunk_size)
        check_chunk_boundary(self.config.start_gradient_frame, blocks, "start_gradient_frame")
        main_engine = self.net.transformer.kv_cache_engines["main"]

        ctx = CausalContext(
            cache_tag="pos",
            is_ar=True,
            chunk_size=self.net.chunk_size,
            total_num_frames=self.net.total_num_frames,
            cache_engines={"main": main_engine},
            grad_through_kv_cache=self.config.grad_through_kv_cache,
        )

        exit_steps = self._sample_exit_steps(len(blocks))
        logger.debug(f"exit_steps: {exit_steps}")

        # student_grid, for however many steps this rollout was asked for
        student_grid = self.net.noise_scheduler.student_grid(device=self.device)

        # Collect denoised blocks and concatenate to preserve autograd graph
        denoised_blocks = []
        for block_idx, (cur_start_frame, cur_end_frame) in enumerate(blocks):
            ctx.cur_start_frame = cur_start_frame
            enable_exit_grad = torch.is_grad_enabled() and (cur_start_frame >= self.config.start_gradient_frame)
            with torch.set_grad_enabled(enable_exit_grad):
                x0_pred_chunk = self._denoise_causal_chunk(
                    self.net,
                    input_student[:, :, cur_start_frame:cur_end_frame],
                    student_grid,
                    condition,
                    ctx,
                    cur_start_frame,
                    stop_step=exit_steps[block_idx],
                    grad_only_at_stop=True,
                )

            # Save denoised block; keep autograd path by collecting and concatenating later
            denoised_blocks.append(x0_pred_chunk)

            # Update internal KV cache for this finished block. The final
            # block's cache is never read: nothing follows it in this rollout
            # and the next call starts from clear_caches(). Skipping its store
            # saves one chunk forward per rollout and, with
            # grad_through_kv_cache, avoids retaining an unused graph.
            if block_idx == len(blocks) - 1:
                continue
            enable_cache_grad = ctx.grad_through_kv_cache and torch.is_grad_enabled()
            with torch.set_grad_enabled(enable_cache_grad):
                self._store_causal_chunk(
                    self.net,
                    x0_pred_chunk,
                    student_grid,
                    condition,
                    ctx,
                    cur_start_frame,
                )

        # Concatenate blocks along the temporal dimension to form full output with gradients
        output = torch.cat(denoised_blocks, dim=2)

        generation_metrics = {
            "avg_denoising_steps": torch.tensor(
                sum(step + 1 for step in exit_steps) / len(exit_steps),
                device=self.device,
            ),
        }
        return Generation(data=output, x0=output, metrics=generation_metrics)
