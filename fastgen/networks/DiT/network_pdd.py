# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDD wrapper for DiT's patch-major final projection.

``unpatchify`` infers the widened channel count. ``learn_sigma`` and
``r_timestep`` are unsupported.
"""

from fastgen.networks.DiT.network import DiT
from fastgen.networks.utils import PDDMixin, PDDSpec


class DiTPDD(PDDMixin, DiT):
    """DiT with a grid-widened final linear for parallel decoding distillation."""

    PDD_SPEC = PDDSpec(
        final_linear="final_layer.projection",
        channel_layout="patch_major",
        out_channels_path="out_channels",  # self.out_channels (== in_channels when learn_sigma=False)
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.learn_sigma:
            raise ValueError(
                "DiTPDD does not support learn_sigma=True: the post-unpatchify "
                "`chunk(2, dim=1)` splits the grid-widened channels incorrectly. "
                "Set learn_sigma=False."
            )
        if self.r_timestep:
            raise ValueError(
                "DiTPDD does not support MeanFlow r_timestep=True: the PDD forward "
                "does not thread `r` through the base network."
            )
