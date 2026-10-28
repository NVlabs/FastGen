# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.networks.utils import make_pdd, PDDSpec
from fastgen.networks.Wan.network import Wan

# Define the PDD specification for the Wan model
PDD_SPEC_WAN = PDDSpec(
    final_linear="transformer.proj_out",
    channel_layout="patch_major",
    out_channels_path="transformer.config.out_channels",
)
# Create the PDD variant of the Wan model
WanPDD = make_pdd(Wan, PDD_SPEC_WAN, module=__name__)
