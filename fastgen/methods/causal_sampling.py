# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable autoregressive student sampling for causal FastGen methods."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from fastgen.networks.network_causal import CausalContext, causal_chunk_ranges
from fastgen.utils.tree_utils import cast_like
import fastgen.utils.logging_utils as logger


if TYPE_CHECKING:
    from fastgen.networks.network_causal import CausalFastGenNetwork


class CausalStudentSamplingMixin:
    """Instance-level chunked sampling shared by causal distillation methods."""

    def _denoise_causal_chunk(
        self,
        net: CausalFastGenNetwork,
        x_chunk: torch.Tensor,
        student_grid: torch.Tensor,
        condition: Any,
        ctx: CausalContext,
        start_frame: int,
        stop_step: int | None = None,
        grad_only_at_stop: bool = False,
    ) -> torch.Tensor:
        """Denoise one temporal chunk through an optional inclusive stop step.

        The full ``student_grid`` determines the denoising intervals. By default the
        entire grid runs with the caller's grad mode.
        ``grad_only_at_stop`` detaches the prefix and preserves gradients only
        for the returned x0 prediction at ``stop_step``.
        """
        x_next = x_chunk
        num_steps = len(student_grid) - 1
        if stop_step is None:
            stop_step = num_steps - 1
        if not 0 <= stop_step < num_steps:
            raise ValueError(f"stop_step must be in [0, {num_steps}), got {stop_step}.")

        caller_grad_enabled = torch.is_grad_enabled()
        for step_index, (t_cur, t_next) in enumerate(zip(student_grid[:-1], student_grid[1:])):
            x_cur = x_next
            ctx.cur_start_frame = start_frame
            ctx.store_kv = False
            step_grad_enabled = caller_grad_enabled and (not grad_only_at_stop or step_index == stop_step)
            if step_grad_enabled and grad_only_at_stop:
                # Prefix no_grad steps may cache detached AMP casts (notably LoRA).
                # Clear them before the gradient step.
                torch.clear_autocast_cache()
            with torch.set_grad_enabled(step_grad_enabled):
                x_next = net(
                    x_cur,
                    t_cur.expand(x_cur.shape[0]),
                    condition=condition,
                    fwd_pred_type="x0",
                    causal_ctx=ctx,
                    output_dtype=None,
                )

                if step_index == stop_step:
                    return cast_like(x_next, x_cur)

                x_next = net.noise_scheduler.step(
                    x_cur,
                    x_next,
                    t_cur,
                    t_next,
                    sample_type=self.config.student_sample_type,
                )

        raise RuntimeError(f"Denoising did not reach stop_step={stop_step}.")

    def _store_causal_chunk(
        self,
        net: CausalFastGenNetwork,
        x_clean: torch.Tensor,
        student_grid: torch.Tensor,
        condition: Any,
        ctx: CausalContext,
        start_frame: int,
        t_cache: torch.Tensor | None = None,
    ) -> None:
        """Store one completed chunk in the KV cache, with configured context noise."""
        batch_size = x_clean.shape[0]
        x_cache = x_clean
        context_noise = self.config.context_noise
        if context_noise is not None:
            t_cache = torch.full(
                (batch_size,),
                context_noise,
                device=student_grid.device,
                dtype=student_grid.dtype,
            )
            x_cache = net.noise_scheduler.forward_process(x_clean, torch.randn_like(x_clean), t_cache)
        elif t_cache is None:
            t_cache = student_grid[-1].expand(batch_size)

        ctx.cur_start_frame = start_frame
        with ctx.set_store_kv(True):
            _ = net(
                x_cache,
                t_cache,
                condition=condition,
                fwd_pred_type="x0",
                causal_ctx=ctx,
                apply_output_conditioning=False,
            )

    def _student_sample_loop(
        self,
        net: CausalFastGenNetwork,
        x: torch.Tensor,
        student_grid: torch.Tensor,
        condition: Any = None,
        neg_condition: Any = None,
    ) -> torch.Tensor:
        """Sample a causal student chunk by chunk while maintaining its KV cache."""
        del neg_condition
        logger.debug(f"Using causal student sampling for {type(self).__name__}")
        net.clear_caches()

        num_frames = x.shape[2]
        chunk_size = net.chunk_size

        ctx = CausalContext(
            cache_tag="pos",
            is_ar=True,
            chunk_size=chunk_size,
            total_num_frames=net.total_num_frames,
        )

        for start, end in causal_chunk_ranges(num_frames, chunk_size):
            x_next = self._denoise_causal_chunk(
                net,
                x[:, :, start:end, ...],
                student_grid,
                condition,
                ctx,
                start,
            )
            x[:, :, start:end, ...] = x_next
            self._store_causal_chunk(net, x_next, student_grid, condition, ctx, start)

        net.clear_caches()
        return x
