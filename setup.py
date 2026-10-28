# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_namespace_packages
from pathlib import Path


def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with open(requirements_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="fastgen",
    version="0.2.0",
    description="FastGen is a PyTorch-based framework for building fast generative models using various distillation and acceleration techniques.",
    license="Apache-2.0",
    license_files=("LICENSE", "licenses/**/LICENSE"),
    # Experiment configs and some vendored modules use implicit namespace packages.
    packages=find_namespace_packages(
        include=("fastgen", "fastgen.*", "scripts", "scripts.*", "comfyui", "comfyui.*"),
        exclude=("*.__pycache__", "*.__pycache__.*"),
    ),
    package_data={
        "": ["LICENSE", "*.json"],
        "scripts.inference": ["prompts/*.txt", "examples/*.png", "examples/*.mp4", "examples/camera/*/*.npy"],
    },
    # NumPy 1.x (required below) supports Python only through 3.12.
    python_requires=">=3.10,<3.13",
    install_requires=read_requirements(),
)
