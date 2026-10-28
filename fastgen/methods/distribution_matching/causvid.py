# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Optional

import torch

from fastgen.methods.distribution_matching.dmd2 import DMD2Model, Generation, TrainingInputs
from fastgen.methods.causal_sampling import CausalStudentSamplingMixin
from fastgen.networks.network_causal import CausalContext, causal_chunk_ranges, check_chunk_boundary
from fastgen.utils import basic_utils
import fastgen.utils.logging_utils as logger


if TYPE_CHECKING:
    from fastgen.networks.network_causal import CausalFastGenNetwork


class CausVidModel(CausalStudentSamplingMixin, DMD2Model):
    """CausVid implementation"""

    # Causal generations are clean estimates.
    student_generations = ("x0",)

    def init_optimizers(self):
        super().init_optimizers()
        # Runs after the trainer loads trainer.checkpointer.pretrained_ckpt_path, which sets resume_iter.
        if self.config.pretrained_student_net_path is None and not hasattr(self, "resume_iter"):
            logger.warning(
                f"{type(self).__name__} starts its causal student without a causal initialization; set "
                "model.pretrained_student_net_path or trainer.checkpointer.pretrained_ckpt_path to a checkpoint "
                "pretrained with causal SFT or causal KD."
            )

    def _training_inputs(self, data: dict[str, Any], iteration: int = 1) -> TrainingInputs:
        """Construct CausVid's data-driven, inhomogeneously noised inputs.

        Args:
            data: Training batch containing real video and optional conditions.
            iteration: Current training iteration.

        Returns:
            ``TrainingInputs`` with ``input_student`` = the real video noised per causal chunk at student-grid
            times, ``student_step`` [B, T] grid indices, and the homogeneous ``t`` [B] / ``eps`` for the losses.
        """
        del iteration
        if self.config.backward_simulation is not None:
            raise NotImplementedError(
                "CausVid does not support backward_simulation training; it distills from noised real video. "
                "Set backward_simulation=None and provide a data loader."
            )

        real_data = data["real"]
        assert real_data.ndim == 5, "CausVid only works for video data"
        batch_size, num_frames = real_data.shape[0], real_data.shape[2]
        assert hasattr(self.net, "chunk_size"), "net does not have the chunk_size attribute"
        chunk_size = self.net.chunk_size

        # Add noise to real image data (for multistep generation)
        eps_inhom = torch.randn(batch_size, *self.input_shape, device=self.device, dtype=real_data.dtype)
        t_inhom, student_step = self.net.noise_scheduler.sample_student_timesteps_inhom(
            batch_size,
            num_frames,
            chunk_size,
            device=self.device,
        )  # shape [B, T]
        t_inhom_expanded = t_inhom[:, None, :, None, None]  # shape [B, 1, T, 1, 1]
        noisy_real_data = self.net.noise_scheduler.forward_process(real_data, eps_inhom, t_inhom_expanded)

        t, eps = self._sample_t_and_eps(batch_size)

        return TrainingInputs(input_student=noisy_real_data, student_step=student_step, t=t, eps=eps)

    def gen_data_from_net(
        self,
        input_student: torch.Tensor,
        student_step: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> Generation:
        del neg_condition
        t_student = self._student_grid(student_step.device)[student_step]
        causal_ctx = CausalContext(
            cache_tag="pos",
            chunk_size=self.net.chunk_size,
            total_num_frames=input_student.shape[2],
            store_kv=False,
            is_ar=False,
        )
        gen_data = self.net(
            input_student,
            t_student,
            condition=condition,
            fwd_pred_type="x0",
            causal_ctx=causal_ctx,
        )
        return Generation(data=gen_data, x0=gen_data)

    def generator_fn_extrapolation(
        self,
        net: CausalFastGenNetwork,
        noise: torch.Tensor,
        num_segments: int,
        overlap_frames: int,
        condition: Any = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate multiple segments using the student generator_fn stepping,
        with optional frame-overlap bridging via a VAE.

        Args:
            net: The student causal network.
            noise: Initial latents for a single segment [B, C, T, H, W].
            condition: Optional conditioning tensor.
            num_segments: Number of segments to autoregressively generate (>= 1).
            overlap_frames: Number of frames to overlap/bridge across segments. Must end on a chunk boundary of the
                segment, whose first chunk holds any remainder frames as in training. Bridging uses the VAE's
                ``reencode_overlap`` round trip (the Wan VAE).
        Returns:
            The concatenated video latents across all segments [B, C, num_segments*T - (num_segments-1)*overlap_frames, H, W].
        """
        logger.debug("Using generator_fn_extrapolation in CausVidModel")
        with basic_utils.inference_mode(net, precision_amp=self.precision_amp_infer, device_type=noise.device.type):
            if num_segments < 1:
                raise ValueError("num_segments must be >= 1")
            if overlap_frames > 0:
                # Checked before the first segment rather than at its bridge.
                vae = self.net_preprocessors.get_preprocessor("vae", purpose="bridge extrapolation segments")

            segment_frames = noise.shape[2]
            dtype = noise.dtype
            device = noise.device
            chunk_size = net.chunk_size

            # Every segment uses the training layout: the first chunk holds any remainder frames, so a continuation
            # segment's prefilled overlap, re-encoded as a clip start, is its leading chunk(s).
            chunks = causal_chunk_ranges(segment_frames, chunk_size)
            check_chunk_boundary(overlap_frames, chunks, "overlap_frames")

            # Prepare the student grid as generator_fn does; the network forward asserts the schedule precision.
            student_grid = net.noise_scheduler.student_grid(device=device)

            ctx = CausalContext(
                cache_tag="pos",
                is_ar=True,
                chunk_size=chunk_size,
                total_num_frames=net.total_num_frames,
            )

            def _prefill_caches(segment_latents: torch.Tensor, frames: int) -> None:
                """Prefill KV caches with overlapping frames from the previous segment, stored like finished chunks.

                Args:
                    segment_latents: Latents to prefill caches with [B, C, T, H, W]
                    frames: Number of frames to prefill
                """
                for start_frame, end_frame in chunks:
                    if end_frame > frames:
                        break
                    slice_latents = segment_latents[:, :, start_frame:end_frame, ...]
                    self._store_causal_chunk(net, slice_latents, student_grid, condition, ctx, start_frame)

            def _run_segment(segment_latents: torch.Tensor, prefill_frames: int) -> torch.Tensor:
                """Run a single segment of autoregressive generation.

                Args:
                    segment_latents: Input latents for this segment [B, C, T, H, W]
                    prefill_frames: Number of frames to prefill from previous segment (for overlap)
                """
                # Clone to avoid in-place modifications on the input tensor
                x = segment_latents.clone()

                # Clear caches before processing a new segment
                net.clear_caches()

                # If we have overlapping frames from the previous segment, prefill caches for them.
                if prefill_frames > 0:
                    _prefill_caches(x, prefill_frames)

                # Initialize only the frames we are about to generate using the first timestep sigma
                if prefill_frames == 0:
                    x = net.noise_scheduler.latents(x, t_init=student_grid[0])
                else:
                    x[:, :, prefill_frames:, ...] = net.noise_scheduler.latents(
                        x[:, :, prefill_frames:, ...], t_init=student_grid[0]
                    )

                for start_frame, end_frame in chunks:
                    if start_frame < prefill_frames:
                        continue
                    x_next = self._denoise_causal_chunk(
                        net,
                        x[:, :, start_frame:end_frame, ...],
                        student_grid,
                        condition,
                        ctx,
                        start_frame,
                    )

                    # Write the generated slice back
                    x[:, :, start_frame:end_frame, ...] = x_next
                    self._store_causal_chunk(net, x_next, student_grid, condition, ctx, start_frame)

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
                # Carry the segment's global frame offset on the shared causal context.
                # Only control-conditioned causal nets (e.g. VaceWan) read it.
                ctx.frame_offset = frame_offset

                segment_latents = _run_segment(current_latents, prefill_frames)

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

                # The VAE round trip re-encodes the overlap as the start of a new clip, under the encoder/decoder AMP
                # policy rather than the surrounding sampler policy.
                with basic_utils.inference_mode(vae, precision_amp=self.precision_amp_enc, device_type=device.type):
                    overlap_latents = vae.reencode_overlap(segment_latents, overlap_frames)

                # Compose the next segment latents with the re-encoded overlap and fresh noise
                next_latents = torch.randn_like(segment_latents)
                next_latents[:, :, :overlap_frames, :, :] = overlap_latents.to(dtype=dtype, device=device)
                current_latents = next_latents
                prefill_frames = overlap_frames

            # Final cleanup and concatenate along the temporal dimension
            net.clear_caches()
            return torch.cat(segments, dim=2).to(dtype=noise.dtype)
