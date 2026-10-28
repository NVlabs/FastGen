# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

# Suppress FutureWarnings attributed to bundled third-party code without hiding
# warnings from FastGen or unrelated dependencies.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"^fastgen\.third_party(?:\.|$)",
)
