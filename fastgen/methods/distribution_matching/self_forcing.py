# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TYPE_CHECKING, List, Optional

import torch
import torch.distributed as dist

from fastgen.methods import CausVidModel

import fastgen.utils.logging_utils as logger
from fastgen.networks.network import CausalFastGenNetwork
from fastgen.utils.basic_utils import convert_cfg_to_dict
from fastgen.utils.distributed import is_rank0, world_size

if TYPE_CHECKING:
    from fastgen.configs.methods.config_self_forcing import ModelConfig


class SelfForcingModel(CausVidModel):
    """Self-Forcing model for distribution matching distillation
    Inheritance hierarchy:
    SelfForcingModel -> CausVidModel -> DMD2Model -> FastGenModel

    The major difference between SelfForcingModel and DMD2Model is how we get
    the gen_data in the single_train_step() function.  In SelfForcingModel, we
    use self.rollout_with_gradient() to get the gen_data, which
    does the rollout with gradient tracking at the last denoising step.  The
    number of denoising steps is stochastic, and is sampled from the
    denoising_step_list.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

    def _generate_noise_and_time(
        self, real_data: torch.Tensor, iteration: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random noises and time step

        Args:
            real_data: Real data tensor for dtype/device reference
            iteration: Current training iteration, used only to tell a generator
                update from a fake-score one so `fake_score_sample_t_cfg` can
                apply on the latter. `None` always uses `sample_t_cfg`.

        Returns:
            input_student: Random noise used by the student
            t_max: Time step used by the student
            t: Time step for distribution matching
            eps: Random noise used by a forward process
        """
        batch_size = real_data.shape[0]

        eps_student = torch.randn(batch_size, *self.input_shape, device=self.device, dtype=real_data.dtype)
        t_student = torch.full(
            (batch_size,),
            self.net.noise_scheduler.max_t,
            device=self.device,
            dtype=self.net.noise_scheduler.t_precision,
        )
        input_student = self.net.noise_scheduler.latents(noise=eps_student)

        t = self._sample_noising_time(batch_size, iteration)

        eps = torch.randn_like(real_data, device=self.device, dtype=real_data.dtype)

        return input_student, t_student, t, eps

    def _sample_denoising_end_steps(self, num_blocks: int) -> List[int]:
        """Sample a list of denoising end indices for each block"""
        sample_steps = self.config.student_sample_steps

        if is_rank0():
            if self.config.last_step_only:
                indices = torch.full((num_blocks,), sample_steps - 1, dtype=torch.long, device=self.device)
            else:
                indices = torch.randint(low=0, high=sample_steps, size=(num_blocks,), device=self.device)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=self.device)

        # Broadcast the random indices to all ranks
        if world_size() > 1:
            dist.broadcast(indices, src=0)

        return indices.tolist()

    def rollout_with_gradient(
        self,
        noise: torch.Tensor,
        condition: Optional[Any] = None,
        enable_gradient: bool = True,
        start_gradient_frame: int = 0,
    ) -> torch.Tensor:
        """
        Perform self-forcing rollout with gradient tracking at the last step of each block.

        No external KV cache is used. Instead, we update the model's internal caches
        once per completed block using `store_kv=True` under no_grad.

        Args:
            noise: Initial noise tensor [B, C, T, H, W]
            condition: Conditioning (dict with 'text_embeds'/'prompt_embeds' or a tensor)
            enable_gradient: Whether to enable gradients at the exit step
            start_gradient_frame: Frame index to start gradient tracking

        Returns:
            generated_frames: Generated video frames, same shape as noise [B, C, T, H, W]
        """
        assert isinstance(self.net, CausalFastGenNetwork), f"{self.net} must be a CausalFastGenNetwork"
        self.net.clear_caches()

        # Reset peak memory stats for per-rollout VRAM monitoring
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=self.device)

        batch_size, C, num_frames, H, W = noise.shape
        chunk_size = self.net.chunk_size
        num_blocks = num_frames // chunk_size
        remaining_size = num_frames % chunk_size
        sample_steps = self.config.student_sample_steps
        dtype = noise.dtype

        # Sample denoising end steps
        denoising_end_steps = self._sample_denoising_end_steps(num_blocks)
        logger.debug(f"denoising_end_steps: {denoising_end_steps}")

        # t_list
        t_list = self.config.sample_t_cfg.t_list
        if t_list is None:
            t_list = self.net.noise_scheduler.get_t_list(sample_steps, device=self.device)
        else:
            assert (
                len(t_list) - 1 == sample_steps
            ), f"t_list length (excluding zero) != student_sample_steps: {len(t_list) - 1} != {sample_steps}"
            t_list = torch.tensor(t_list, device=self.device, dtype=self.net.noise_scheduler.t_precision)

        # Collect denoised blocks and concatenate to preserve autograd graph
        denoised_blocks = []
        for block_idx in range(num_blocks):
            if num_blocks == 0:
                # Handle case where num_frames < chunk_size
                cur_start_frame, cur_end_frame = 0, remaining_size
            else:
                # Normal chunking logic
                cur_start_frame = 0 if block_idx == 0 else chunk_size * block_idx + remaining_size
                cur_end_frame = chunk_size * (block_idx + 1) + remaining_size

            noisy_input = noise[:, :, cur_start_frame:cur_end_frame]

            # Denoising steps for current block
            for step, t_cur in enumerate(t_list):
                if self.config.same_step_across_blocks:
                    exit_flag = step == denoising_end_steps[0]
                else:
                    exit_flag = step == denoising_end_steps[block_idx]

                t_chunk_cur = t_cur.expand(batch_size)

                if not exit_flag:
                    # Non-exit steps: no grads, no cache updates
                    with torch.no_grad():
                        x0_pred_chunk = self.net(
                            noisy_input,
                            t_chunk_cur,
                            condition=condition,
                            cache_tag="pos",
                            store_kv=False,
                            cur_start_frame=cur_start_frame,
                            fwd_pred_type="x0",
                            is_ar=True,
                        )

                    # update to the next timestep for forward process
                    t_next = t_list[step + 1]
                    t_chunk_next = t_next.expand(batch_size)
                    if self.config.student_sample_type == "sde":
                        eps_infer = torch.randn_like(x0_pred_chunk)
                    elif self.config.student_sample_type == "ode":
                        eps_infer = self.net.noise_scheduler.x0_to_eps(xt=noisy_input, x0=x0_pred_chunk, t=t_chunk_cur)
                    else:
                        raise NotImplementedError(
                            f"student_sample_type must be one of 'sde', 'ode' but got {self.config.student_sample_type}"
                        )
                    noisy_input = self.net.noise_scheduler.forward_process(x0_pred_chunk, eps_infer, t_chunk_next)
                else:
                    # Exit step: allow gradient if enabled
                    enable_grad = (
                        enable_gradient and torch.is_grad_enabled() and (cur_start_frame >= start_gradient_frame)
                    )
                    with torch.set_grad_enabled(enable_grad):
                        x0_pred_chunk = self.net(
                            noisy_input,
                            t_chunk_cur,
                            condition=condition,
                            cache_tag="pos",
                            store_kv=False,
                            cur_start_frame=cur_start_frame,
                            fwd_pred_type="x0",
                            is_ar=True,
                        )
                    break

            # Save denoised block; keep autograd path by collecting and concatenating later
            denoised_blocks.append(x0_pred_chunk)

            # Update internal KV cache for this finished block using t=0 or context noise (no grads)
            with torch.no_grad():
                if self.config.context_noise > 0:
                    # Add context noise to denoised frames before caching
                    t_cache = torch.full((batch_size,), self.config.context_noise, device=self.device, dtype=dtype)
                    x0_pred_cache = self.net.noise_scheduler.forward_process(
                        x0_pred_chunk,
                        torch.randn_like(x0_pred_chunk),
                        t_cache,
                    )
                else:
                    x0_pred_cache = x0_pred_chunk
                    t_cache = torch.zeros(batch_size, device=self.device, dtype=dtype)

                # update kv-cache with generated frames
                _ = self.net(
                    x0_pred_cache,
                    t_cache,
                    condition=condition,
                    cache_tag="pos",
                    store_kv=True,
                    cur_start_frame=cur_start_frame,
                    fwd_pred_type="x0",
                    is_ar=True,
                )

        # Concatenate blocks along the temporal dimension to form full output with gradients
        output = torch.cat(denoised_blocks, dim=2) if len(denoised_blocks) > 0 else torch.empty_like(noise)

        self.net.clear_caches()
        return output

    def gen_data_from_net(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> torch.Tensor:
        del t_student
        gen_data = self.rollout_with_gradient(
            noise=input_student,
            condition=condition,
            enable_gradient=self.config.enable_gradient_in_rollout,
            start_gradient_frame=self.config.start_gradient_frame,
        )
        return gen_data
