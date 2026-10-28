# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDD wrapper for LTX2's channel-major audio and video projections.

The unpackers infer widened channel counts, and only configured modalities
are widened. PDD sampling consumes each modality's native sigma-velocity heads.
"""

from fastgen.networks.LTX2.network import LTX2
from fastgen.networks.utils import PDDSpec, make_pdd

LTX2PDD = make_pdd(
    LTX2,
    PDDSpec(
        final_linear={"video": "transformer.proj_out", "audio": "transformer.audio_proj_out"},
        channel_layout="channel_major",
    ),
    module=__name__,
)
