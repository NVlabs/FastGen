#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to automatically format code and files

# Run from the repository root, wherever the script is called from.
fastgen_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$fastgen_root" || exit 1
echo "Formatting $fastgen_root"

set -e

make install-linters
make format
