# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDD wrapper for MiniMax-H3's channel-major audio and video heads.

H3's output projections emit ``x0 - eps``, which ``forward`` negates into
``eps - x0``: for rectified flow, already the native sigma velocity that PDD
heads are parameterized in. The wrapper therefore needs no teacher-parameter
fold and bypasses H3's ordinary single-time output scaling.
"""

from typing import Optional

import torch

from fastgen.networks.MiniMaxH3.network import MiniMaxH3
from fastgen.networks.utils import PDDMixin, PDDSpec


class MiniMaxH3PDD(PDDMixin, MiniMaxH3):
    """MiniMax-H3 with grid-widened output projections for PDD."""

    PDD_SPEC = PDDSpec(
        final_linear={"video": "transformer.proj_out", "audio": "transformer.audio_proj_out"},
        channel_layout="channel_major",
    )

    def pdd_teacher_head_scale(self, key: Optional[str]) -> Optional[torch.Tensor]:
        """Negated by ``forward``, H3's projections already give ``eps - x0 = dx/dsigma``; no fold."""
        return None

    def _to_flow_convention(self, model_output, t):
        """Keep PDD heads in their native sigma-velocity space."""
        return model_output
