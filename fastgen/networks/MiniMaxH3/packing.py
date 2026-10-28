# SPDX-FileCopyrightText: Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Packed-sequence and conditioning geometry of the MiniMax-H3 blocks -- the checkpoint's constants and row
placement, kept in one place so no block reimplements it.

MiniMax-H3 runs its transformer over a single packed 1-D sequence holding every modality at once. For the
text/keyframe tasks the row order is ``[ text (L) | keyframe conditions (C) | target audio (A) | target video
(V) ]``, and each helper here places a row and gives it its ``(t, h, w)`` rotary coordinate. Coordinates are
built in float64 because video and audio share one 40-units-per-second rotary clock (video ``5/3`` units per
pixel frame at 24 fps, audio one unit per latent at 40 latents/s) -- that shared clock *is* the alignment.

The reference pads the sequence to a multiple of 64 as a separate attention document; we build it without
padding, so ``MiniMaxH3Transformer3DModel`` needs no attention mask (keeping the unmasked attention backends
available).

Vendored from the diffusers MiniMax-H3 modular pipeline, where the "modular only" refactor keeps these helpers
inside the pipeline-block classes rather than a standalone module; fastgen imports just the H3 models and
``randn_tensor`` from diffusers.
"""

from dataclasses import dataclass

import numpy as np
import torch

from diffusers.utils.torch_utils import randn_tensor


# Per-row modality tags. They index the transformer's AdaLN table, so the values are a checkpoint contract.
MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

# MiniMax-H3 generates at a fixed 24 fps.
MINIMAX_H3_FPS = 24

# The video VAE encodes each chunk of 17 pixel frames to 5 latent frames, filling the last chunk by repetition, and
# drops the 3 trailing latent frames of the result, so `17 * n + 5` pixel frames map to `5 * n + 2` latent frames.
MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_LATENTS_PER_CHUNK = 5

# The pixel convention of the video VAE: ImageNet-normalized RGB over a `[0, 1]` base range.
MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)

# MiniMax-H3 conditions on the *unnormalized* hidden state its Qwen3-VL conditioner produces after the 50th of its 64
# decoder layers, i.e. `hidden_states[50]` (`hidden_states[0]` being the embedding output).
MINIMAX_H3_TEXT_ENCODER_LAYER = 50

# The audio VAE hops 800 samples at 32 kHz, i.e. 40 latents per second. Stereo is carried as two channel-major
# blocks of audio rows (and as two batch items at the audio VAE boundary, which is mono).
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2

# MiniMax-H3 reverses FastGen's RF time: `t_h3 = 1` is data and `t_h3 = 0` is noise. The released
# model uses `t_h3 = 0.999` for keyframe rows, i.e. a noise coefficient of `1 - t_h3 = 0.001`.
MINIMAX_H3_KEYFRAME_TIMESTEP = 0.999

# Rotary-time constants. One latent frame spans `5/3 * frames_per_latent` rotary units, where the pattern
# `(1, 4, 4, 4, 4)` mirrors the VAE's 17-pixel-frames-to-5-latent-frames grouping; the spatial axes are normalized
# by the square root of the latent area and scaled by 32.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@dataclass
class MiniMaxH3PackedSequence:
    r"""
    The structural description of one packed MiniMax-H3 sequence.

    Attributes:
        sequence_length (`int`):
            Total number of rows, `L + C + A + V`.
        position_ids (`torch.Tensor` of shape `(sequence_length, 3)`, float64):
            The `(t, h, w)` rotary coordinate of every row.
        token_tags (`torch.Tensor` of shape `(sequence_length,)`):
            The modality tag of every row.
        video_indices (`torch.Tensor`):
            Sequence positions of the video rows: the keyframe conditioning rows first, then the target rows.
        audio_indices (`torch.Tensor`):
            Sequence positions of the target audio rows.
        text_indices (`torch.Tensor`):
            Sequence positions of the text rows.
        num_condition_video_rows (`int`):
            How many leading entries of `video_indices` are conditioning rows rather than generated rows.
    """

    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int


def video_latent_num_frames(num_frames: int) -> int:
    r"""
    The number of latent frames the video VAE produces for a `17 * n + 5` frame count.

    Args:
        num_frames (`int`): An aligned number of frames.

    Returns:
        `int`: The number of latent frames, `5 * n + 2`.
    """
    if num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(f"`num_frames` must be of the form 17 * n + 5, got {num_frames}.")
    return (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK * MINIMAX_H3_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int, fps: float = MINIMAX_H3_FPS) -> int:
    r"""
    The number of audio latents that covers a video of `num_frames` frames.

    Args:
        num_frames (`int`): The number of video frames.
        fps (`float`, defaults to 24): The rate those frames run at.

    Returns:
        `int`: The number of audio latents, rounded at the 40 Hz latent grid.
    """
    return int(round(num_frames / fps * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    r"""
    Pack video latents into transformer rows.

    Args:
        latents (`torch.Tensor` of shape `(batch_size, channels, num_frames, height, width)`):
            The latents to pack.
        patch_size (`tuple[int, int, int]`): The `(t, h, w)` patch.

    Returns:
        `torch.Tensor` of shape `(batch_size * num_patches, channels * prod(patch_size))`: The packed rows, ordered
        frame-major then row-major.
    """
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.")

    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    r"""
    Unpack transformer rows back into video latents. The inverse of [`patchify_video_latents`].

    Args:
        rows (`torch.Tensor` of shape `(num_patches, channels * prod(patch_size))`): The packed rows.
        num_latent_frames (`int`): Number of latent frames.
        latent_height (`int`): Latent height.
        latent_width (`int`): Latent width.
        channels (`int`): Number of latent channels.
        patch_size (`tuple[int, int, int]`): The `(t, h, w)` patch.

    Returns:
        `torch.Tensor` of shape `(batch_size, channels, num_latent_frames, latent_height, latent_width)`.
    """
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    r"""
    One aspect-normalized spatial rotary axis: `dim // patch` coordinates centred on the unit interval, scaled up by
    32. The right endpoint is excluded, so a square canvas spans `[0, 32)`.
    """
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # Built with numpy: `np.linspace(..., endpoint=False)` is `start + arange(num) * (stop - start) / num`, which is
    # not what `torch.linspace` computes, and the float64 grid has to be reproduced exactly.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    r"""The rotary time of every latent frame, starting at `origin`. Spacing is non-uniform: `5/3 * (1, 4, 4, 4, 4)`."""
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _temporal_position_span(num_latent_frames: int) -> float:
    r"""
    The rotary time spanned by `num_latent_frames` latent frames.

    Summed by numpy (pairwise summation) rather than sequentially: the reference computes the keyframe anchor this
    way and the two summation orders differ in the last ulp from 16 latent frames onwards.
    """
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for index in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[index :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[index]
    return float(spans.sum())


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3PackedSequence:
    r"""
    Build the `[text | keyframe conditions | target audio | target video]` layout used by the `t2va` and `fl2va`
    tasks.

    Args:
        text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
            The modality tag of every text row. Text is tagged `1`, except for the rows of a keyframe's vision block,
            which MiniMax-H3 tags `0` (video).
        num_latent_frames (`int`): Number of target latent frames.
        latent_height (`int`): Target latent height.
        latent_width (`int`): Target latent width.
        num_audio_latents (`int`): Number of target audio latents per channel.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
        keyframe_anchors (`tuple[str, ...]`):
            One entry per keyframe conditioning block, in packed order: `"first"` anchors the block at the first
            latent frame, `"last"` at the last one.

    Returns:
        [`MiniMaxH3PackedSequence`]
    """
    invalid_tags = set(text_token_tags.tolist()) - {MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG}
    if invalid_tags:
        raise ValueError(
            f"`text_token_tags` must tag every text row {MINIMAX_H3_TEXT_TAG} (text) or {MINIMAX_H3_VIDEO_TAG} "
            f"(a keyframe's vision block), got {sorted(invalid_tags)}. The tags index the transformer's AdaLN "
            "table, and the packed sequence has no padding rows, so any other value modulates a row with the "
            "wrong parameters."
        )

    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    # 1. The (t, h, w) grid. Text rows sit on the time axis at their row index, and the media rows continue the time
    # axis from there, so text length shifts the whole media clock.
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    frame_grid = torch.stack([grid.reshape(-1) for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")], -1)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            anchor_time = float(num_text_tokens) + _temporal_position_span(num_latent_frames) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio rows are channel-major and share the video's rotary clock: one unit per latent at 40 latents/s equals
    # 24 fps * 5/3. They carry no height coordinate and are pinned to the two extremes of the width grid.
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    # 2. Row indices and modality tags.
    video_indices = torch.cat([torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
    )


def build_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Assign a timestep to every row of the packed sequence and reduce it to the transformer's `(timestep,
    timestep_indices)` pair.

    One forward serves rows at different noise levels: the generated video and audio rows step down their own
    schedules while the conditioning rows stay pinned at their MiniMax-H3 conditioning timestep. Text rows never
    reach an output head and inherit the video timestep.

    Args:
        layout ([`MiniMaxH3PackedSequence`]): The packed layout.
        video_timestep (`float`): Timestep of the generated video rows.
        audio_timestep (`float`): Timestep of the generated audio rows.
        condition_video_timestep (`float`): Timestep of the video conditioning rows.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: the distinct timesteps, sorted, and the index of every row into them.
    """
    row_timesteps = torch.full(
        (layout.sequence_length,),
        video_timestep,
        dtype=torch.float32,
        device=layout.video_indices.device,
    )
    row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = condition_video_timestep
    row_timesteps[layout.audio_indices] = audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


def keyframe_condition_noise(
    condition_latent_shapes: tuple[tuple[int, int, int], ...],
    patch_size: tuple[int, int, int],
    latent_channels: int,
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Draw the noise that the keyframe (or reference) conditioning rows are mixed with.

    One draw per condition, in packed order, off the request's generator. The conditioning rows are prepared before
    the target rows, so these are the *first* draws of a request, ahead of the video and audio noise of
    [`~MiniMaxH3PrepareLatentsStep.prepare_latents`] — the order is part of what a generator reproduces.

    Args:
        condition_latent_shapes (`tuple[tuple[int, int, int], ...]`):
            The `(num_latent_frames, latent_height, latent_width)` of every condition, in packed order.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
        latent_channels (`int`): Number of video latent channels.
        generator (`torch.Generator`, *optional*): The generator of the request.
        device (`torch.device`, *optional*): The device the noise is drawn on.
        dtype (`torch.dtype`, defaults to `torch.float32`): The dtype of the noise.

    Returns:
        `torch.Tensor` of shape `(num_condition_rows, latent_channels * prod(patch_size))`: the noise rows,
        concatenated in packed order.
    """
    rows = []
    for num_latent_frames, latent_height, latent_width in condition_latent_shapes:
        noise = randn_tensor(
            (1, latent_channels, num_latent_frames, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        rows.append(patchify_video_latents(noise, patch_size))
    return torch.cat(rows)
