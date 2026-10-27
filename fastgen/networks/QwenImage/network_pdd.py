# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDD wrapper for QwenImage's channel-major output projection.

QwenImage packs 2x2 patches outside the transformer, and its unpacker infers
the widened channel count, so the generic wrapper needs no overrides.
"""

from fastgen.networks.QwenImage.network import QwenImage
from fastgen.networks.utils import PDDSpec, make_pdd

QwenImagePDD = make_pdd(
    QwenImage,
    PDDSpec(
        final_linear="transformer.proj_out",
        channel_layout="channel_major",
    ),
    module=__name__,
)
