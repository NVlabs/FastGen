# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latent Adversarial Diffusion Distillation (LADD)."""

from fastgen.methods.distribution_matching.dmd2 import DMD2Model


class LADDModel(DMD2Model):
    """GAN-only specialization of DMD2."""

    @property
    def use_vsd(self) -> bool:
        """LADD never builds or trains a fake-score model."""
        return False
