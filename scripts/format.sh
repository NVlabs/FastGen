#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to automatically format code and files

fastgen_root=$(git rev-parse --show-toplevel)
echo "Formatting $fastgen_root"

dependencies=($(pip3 freeze | grep -E 'black==23.10.0|ruff==0.6.9|mypy==1.9.0|types-psutil'))
if [ "${#dependencies[@]}" -ne 6 ]; then
    python3 -m pip install --upgrade pip
    python3 -m pip install black==23.10.0
    python3 -m pip install ruff==0.6.9
    python3 -m pip install mypy==1.9.0
    python3 -m pip install types-psutil
fi
set -e

make format
