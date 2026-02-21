# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Any, TYPE_CHECKING, Optional, Dict, Callable

import torch

from fastgen.methods import DMD2Model
from fastgen.utils import basic_utils
import fastgen.utils.logging_utils as logger


if TYPE_CHECKING:
    from fastgen.networks.network import CausalFastGenNetwork


class CausVidModel(DMD2Model):
    """CausVid implementation"""

    def _generate_noise_and_time(
        self, real_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random noises and time step

        Args:
            real_data: Real data tensor  of shape [B, C, T, H, W]

        Returns:
            noisy_real_data: Random noise used by the student
            t_inhom: Inhomogeneous time steps used by the student [B, T] for causal networks
            t: Homogeneous time step [B] for teacher network
            eps: Random noise used by a forward process
        """
        assert real_data.ndim == 5, "CausVid only works for video data"
        batch_size, num_frames = real_data.shape[0], real_data.shape[2]
        assert hasattr(self.net, "chunk_size"), "net does not have the chunk_size attribute"
        chunk_size = self.net.chunk_size

        # Add noise to real image data (for multistep generation)
        eps_inhom = torch.randn(batch_size, *self.input_shape, device=self.device, dtype=real_data.dtype)
        assert hasattr(
            self.net.noise_scheduler, "sample_t_inhom"
        ), "net.noise_scheduler does not have the sample_t_inhom() method"
        t_inhom, _ = self.net.noise_scheduler.sample_t_inhom(
            batch_size,
            num_frames,
            chunk_size,
            sample_steps=self.config.student_sample_steps,
            t_list=self.config.sample_t_cfg.t_list,
            device=self.device,
        )  # shape [B, T]
        t_inhom_expanded = t_inhom[:, None, :, None, None]  # shape [B, 1, T, 1, 1]
        noisy_real_data = self.net.noise_scheduler.forward_process(real_data, eps_inhom, t_inhom_expanded)

        t = self.net.noise_scheduler.sample_t(
            batch_size,
            **basic_utils.convert_cfg_to_dict(self.config.sample_t_cfg),
            device=self.device,
        )
        eps = torch.randn_like(eps_inhom, device=self.device, dtype=real_data.dtype)

        return noisy_real_data, t_inhom, t, eps

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        noise = torch.randn_like(gen_data, dtype=self.precision)
        gen_rand_func = partial(
            self.generator_fn,
            net=self.net_inference,
            noise=noise,
            condition=condition,
            student_sample_steps=self.config.student_sample_steps,
            student_sample_type=self.config.student_sample_type,
            t_list=self.config.sample_t_cfg.t_list,
            precision_amp=self.precision_amp_infer,
            context_noise=getattr(self.config, "context_noise", 0),  # Optional context noise
        )
        return {"gen_rand": gen_rand_func, "input_rand": noise, "gen_rand_train": gen_data}

    @classmethod
    def _student_sample_loop(
        cls,
        net: CausalFastGenNetwork,
        x: torch.Tensor,
        t_list: torch.Tensor,
        condition: Any = None,
        student_sample_type: str = "sde",
        context_noise: Optional[float] = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample loop for the student network.

        Args:
            net: The FastGenNetwork network
            x: The latents to start from
            t_list: Timesteps to sample
            condition: Optional conditioning information
            student_sample_type: Type of student multistep sampling

        Returns:
            The sampled data
        """
        logger.debug("Using generator_fn in CausVidModel")

        # cleanup caches before sampling
        net.clear_caches()

        batch_size, num_frames = x.shape[0], x.shape[2]
        chunk_size = net.chunk_size
        num_chunks = num_frames // chunk_size
        remaining_size = num_frames % chunk_size

        # initialize all noise using the first timestep
        for i in range(max(1, num_chunks)):
            if num_chunks == 0:
                # Handle case where num_frames < chunk_size
                start, end = 0, remaining_size
            else:
                # Normal chunking logic
                start = 0 if i == 0 else chunk_size * i + remaining_size
                end = chunk_size * (i + 1) + remaining_size

            x_next = x[:, :, start:end, ...]
            for step in range(len(t_list) - 1):
                # denoise
                t_cur = t_list[step].expand(batch_size)
                x_cur = x_next
                x_next = net(
                    x_cur,
                    t_cur,
                    condition=condition,
                    fwd_pred_type="x0",
                    cache_tag="pos",
                    cur_start_frame=start,
                    store_kv=False,
                    is_ar=True,
                    **kwargs,
                )

                # update to the next timestep for forward process
                t_next = t_list[step + 1]
                if t_next > 0:
                    t_chunk_next = t_next.expand(batch_size)
                    if student_sample_type == "sde":
                        eps_infer = torch.randn_like(x_next)
                    elif student_sample_type == "ode":
                        eps_infer = net.noise_scheduler.x0_to_eps(xt=x_cur, x0=x_next, t=t_cur)
                    else:
                        raise NotImplementedError(
                            f"student_sample_type must be one of 'sde', 'ode' but got {student_sample_type}"
                        )
                    x_next = net.noise_scheduler.forward_process(x_next, eps_infer, t_chunk_next)
            x[:, :, start:end, ...] = x_next

            # compute and update the KV cache
            x_cache = x_next
            t_cache = t_list[-1].expand(batch_size)
            if context_noise > 0:
                # Add context noise to denoised frames before caching
                t_cache = torch.full((batch_size,), context_noise, device=x.device, dtype=x.dtype)
                x_cache = net.noise_scheduler.forward_process(x_next, torch.randn_like(x_next), t_cache)

            _ = net(
                x_cache,
                t_cache,
                condition=condition,
                fwd_pred_type="x0",
                cache_tag="pos",
                cur_start_frame=start,
                store_kv=True,
                is_ar=True,
                **kwargs,
            )

        # cleanup caches after full sampling
        net.clear_caches()
        return x

    @classmethod
    def generator_fn_extrapolation(
        cls,
        net: CausalFastGenNetwork,
        noise: torch.Tensor,
        condition: Any = None,
        *,
        num_segments: int,
        overlap_frames: int,
        student_sample_steps: int = 1,
        student_sample_type: str = "sde",
        t_list: Optional[torch.Tensor] = None,
        precision_amp: Optional[torch.dtype] = None,
        context_noise: Optional[float] = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Autoregressively generate multiple segments using the student generator_fn stepping,
        with optional frame-overlap bridging via a VAE.

        Args:
            net: The student causal network.
            noise: Initial latents for a single segment [B, C, T, H, W].
            condition: Optional conditioning tensor.
            num_segments: Number of segments to autoregressively generate (>= 1).
            overlap_frames: Number of frames to overlap/bridge across segments. Must be divisible by chunk_size.
            student_sample_steps: Number of denoising steps used by generator_fn.
            student_sample_type: One of {"sde", "ode"}.
            t_list: Optional custom t_list; if None, derived from scheduler and student_sample_steps.
            precision_amp (Optional[torch.dtype]): If not None, uses precision_amp with this dtype for inference.
            context_noise: Optional context noise scale in [0, 1] for cache prefill.
            **kwargs: Passed through to network forward calls.

        Returns:
            The concatenated video latents across all segments [B, C, num_segments*T - (num_segments-1)*overlap_frames, H, W].
        """
        logger.debug("Using generator_fn_extrapolation in CausVidModel")
        with basic_utils.inference_mode(net, precision_amp=precision_amp, device_type=noise.device.type):
            if num_segments < 1:
                raise ValueError("num_segments must be >= 1")
            if overlap_frames > 0 and net.vae is None:
                raise ValueError("generator_fn_extrapolation requires a VAE instance via `vae` when overlap_frames > 0")

            batch_size, channels, segment_frames, height, width = noise.shape
            dtype = noise.dtype
            device = noise.device
            chunk_size = net.chunk_size

            if segment_frames % chunk_size != 0:
                raise ValueError(f"Segment length {segment_frames} must be divisible by chunk_size {chunk_size}")
            if overlap_frames < 0 or overlap_frames >= segment_frames:
                raise ValueError("overlap_frames must be in [0, segment_frames)")
            if overlap_frames % chunk_size != 0:
                raise ValueError("overlap_frames must be divisible by chunk_size")

            # Prepare t_list consistent with generator_fn
            if t_list is None:
                t_list = net.noise_scheduler.get_t_list(student_sample_steps, device=device, dtype=torch.float32)
            else:
                assert (
                    len(t_list) - 1 == student_sample_steps
                ), f"t_list length (excluding zero) != student_sample_steps: {len(t_list) - 1} != {student_sample_steps}"
                t_list = torch.tensor(t_list, device=device, dtype=torch.float32)
            assert t_list[-1].item() == 0, "t_list[-1] must be zero"

            def _prefill_caches(segment_latents: torch.Tensor, frames: int, frame_offset: int = 0) -> None:
                """Prefill KV caches with overlapping frames from the previous segment.

                Args:
                    segment_latents: Latents to prefill caches with [B, C, T, H, W]
                    frames: Number of frames to prefill
                    frame_offset: Segment-level offset for extrapolation.
                        For segment N, this is N * (segment_frames - overlap_frames).
                """
                if frames == 0:
                    return
                start_frame = 0
                t_zero = t_list[-1].expand(batch_size)  # zero timestep
                while start_frame < frames:
                    end_frame = min(start_frame + chunk_size, frames)
                    slice_latents = segment_latents[:, :, start_frame:end_frame, ...]
                    _ = net(
                        slice_latents,
                        t_zero,
                        condition=condition,
                        fwd_pred_type="x0",
                        cache_tag="pos",
                        cur_start_frame=start_frame,
                        frame_offset=frame_offset,
                        store_kv=True,
                        is_ar=True,
                        **kwargs,
                    )
                    start_frame = end_frame

            def _run_segment(segment_latents: torch.Tensor, prefill_frames: int, frame_offset: int = 0) -> torch.Tensor:
                """Run a single segment of autoregressive generation.

                Args:
                    segment_latents: Input latents for this segment [B, C, T, H, W]
                    prefill_frames: Number of frames to prefill from previous segment (for overlap)
                    frame_offset: Segment-level offset for extrapolation.
                        The network computes global frame index as: frame_offset + cur_start_frame.
                        For segment N, this is N * (segment_frames - overlap_frames).
                """
                # Clone to avoid in-place modifications on the input tensor
                x = segment_latents.clone()

                # Clear caches before processing a new segment
                net.clear_caches()

                # If we have overlapping frames from the previous segment, prefill caches for them
                # The prefill frames use the same frame_offset since they are the OVERLAP
                # from the previous segment (their global indices are frame_offset + 0, 1, ...)
                if prefill_frames > 0:
                    _prefill_caches(x, prefill_frames, frame_offset)

                # Initialize only the frames we are about to generate using the first timestep sigma
                if prefill_frames == 0:
                    x = net.noise_scheduler.latents(x, t_init=t_list[0])
                else:
                    x[:, :, prefill_frames:, ...] = net.noise_scheduler.latents(
                        x[:, :, prefill_frames:, ...], t_init=t_list[0]
                    )

                start_frame = prefill_frames
                while start_frame < segment_frames:
                    end_frame = min(start_frame + chunk_size, segment_frames)
                    x_next = x[:, :, start_frame:end_frame, ...]

                    for step in range(len(t_list) - 1):
                        # Denoise to x0 using the student network
                        t_cur = t_list[step].expand(batch_size)
                        x_cur = x_next
                        x_next = net(
                            x_cur,
                            t_cur,
                            condition=condition,
                            fwd_pred_type="x0",
                            cache_tag="pos",
                            cur_start_frame=start_frame,
                            frame_offset=frame_offset,
                            store_kv=False,
                            is_ar=True,
                            **kwargs,
                        )

                        # Move forward in the forward process if not at the final step
                        t_next = t_list[step + 1]
                        if t_next > 0:
                            t_chunk_next = t_next.expand(batch_size)
                            if student_sample_type == "sde":
                                eps_infer = torch.randn_like(x_next)
                            elif student_sample_type == "ode":
                                eps_infer = net.noise_scheduler.x0_to_eps(xt=x_cur, x0=x_next, t=t_cur)
                            else:
                                raise NotImplementedError(
                                    f"student_sample_type must be one of 'sde', 'ode' but got {student_sample_type}"
                                )
                            x_next = net.noise_scheduler.forward_process(x_next, eps_infer, t_chunk_next)

                    # Write the generated slice back
                    x[:, :, start_frame:end_frame, ...] = x_next

                    # Update KV caches with the denoised slice (optionally with context noise)
                    x_cache = x_next
                    t_cache = t_list[-1].expand(batch_size)
                    if context_noise and context_noise > 0:
                        t_cache = torch.full((batch_size,), context_noise, device=device, dtype=dtype)
                        x_cache = net.noise_scheduler.forward_process(x_next, torch.randn_like(x_next), t_cache)

                    _ = net(
                        x_cache,
                        t_cache,
                        condition=condition,
                        fwd_pred_type="x0",
                        cache_tag="pos",
                        cur_start_frame=start_frame,
                        frame_offset=frame_offset,
                        store_kv=True,
                        is_ar=True,
                        **kwargs,
                    )

                    start_frame = end_frame

                # Clean up caches after finishing the segment
                net.clear_caches()
                return x

            segments = []
            current_latents = noise
            prefill_frames = 0

            for segment_idx in range(num_segments):
                # Compute the global frame offset for depth/control conditioning
                # For segment N, the depth conditioning starts at frame N * (segment_frames - overlap_frames)
                # This ensures:
                # - Segment 0: local frame i → depth frame i
                # - Segment 1 with overlap: local frame 0 (overlap from seg 0's tail) → depth frame (seg_frames - overlap)
                #                           local frame overlap → depth frame seg_frames
                # - And so on...
                frame_offset = segment_idx * (segment_frames - overlap_frames)

                segment_latents = _run_segment(current_latents, prefill_frames, frame_offset)

                if segment_idx == 0:
                    segments.append(segment_latents)
                else:
                    if overlap_frames > 0:
                        segments.append(segment_latents[:, :, overlap_frames:, :, :])
                    else:
                        segments.append(segment_latents)

                if segment_idx == num_segments - 1:
                    break

                # Prepare latents for the next segment
                if overlap_frames == 0:
                    current_latents = torch.randn_like(noise)
                    prefill_frames = 0
                    continue

                # Bridge with VAE: take the last overlap frames from current segment (pixels), encode back to latents
                decoded_video = net.vae.decode(segment_latents)
                tail_pixels = decoded_video[:, :, -overlap_frames:, :, :]
                encoded_tail = net.vae.encode(tail_pixels).to(dtype=dtype, device=device)

                # Reuse all but the first overlapped latent directly to avoid unnecessary encode/decode
                if overlap_frames > 1:
                    reused_tail = segment_latents[:, :, -(overlap_frames - 1) :, :, :]
                    encoded_tail = torch.cat([encoded_tail[:, :, :1, :, :], reused_tail], dim=2)

                # Compose the next segment latents with bridged head and random remainder
                next_latents = torch.randn_like(segment_latents)
                next_latents[:, :, :overlap_frames, :, :] = encoded_tail
                current_latents = next_latents
                prefill_frames = overlap_frames

            # Final cleanup and concatenate along the temporal dimension
            net.clear_caches()
            return torch.cat(segments, dim=2).to(dtype=noise.dtype)
