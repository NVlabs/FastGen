# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDD wrapper for Kandinsky 6's channel-major video and audio projections.

``Kandinsky6OutLayer`` views the video projection's features as
``(C, p_t, p_h, p_w)`` and the audio head emits per-token channels, so both
unpackers infer the widened channel counts; a video-only network widens only
its video head. Under the default unwarped RF schedule the flow already is each
head's native sigma-velocity, so the teacher projections are copied unfolded.
"""

from fastgen.networks.Kandinsky6.network import Kandinsky6
from fastgen.networks.utils import PDDSpec, make_pdd

Kandinsky6PDD = make_pdd(
    Kandinsky6,
    PDDSpec(
        final_linear={"video": "transformer.out_layer.out_layer", "audio": "transformer.audio_out_layer.out_layer"},
        channel_layout="channel_major",
    ),
    module=__name__,
)
