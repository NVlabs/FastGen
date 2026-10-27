# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from fastgen.utils.cache_utils import cache


INCEPTION_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"


class InceptionV3(nn.Module):
    """Wrapper around the cached StyleGAN Inception V3 TorchScript model."""

    def __init__(self, resize_inside=True):
        super(InceptionV3, self).__init__()
        self.base = torch.jit.load(cache(INCEPTION_URL)).eval()
        self.layers = self.base.layers
        self.resize_inside = resize_inside

    def forward(self, x):
        """
        Get the inception features.

        Args:
            x: Image tensor with values in range [-1, 1] (NCHW format).
               Internally converted to [0, 255] before passing to base model.

        Returns:
            Inception features of shape (N, 2048).
        """
        bs = x.shape[0]
        x = (x + 1) * 127.5
        if self.resize_inside:
            features = self.base(x, return_features=True).view((bs, 2048))
        else:
            # make sure it is resized already
            assert (x.shape[2] == 299) and (x.shape[3] == 299)
            # same normalization self.base applies to [0, 255] pixels before self.layers
            x1 = x - 128
            x2 = x1 / 128
            features = self.layers.forward(
                x2,
            ).view((bs, 2048))
        return features
