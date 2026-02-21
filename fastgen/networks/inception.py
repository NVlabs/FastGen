# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import contextlib
from fastgen.utils.io_utils import open_url


@contextlib.contextmanager
def disable_gpu_fuser_on_pt19():
    # On PyTorch 1.9 a CUDA fuser bug prevents the Inception JIT model to run. See
    #   https://github.com/GaParmar/clean-fid/issues/5
    #   https://github.com/pytorch/pytorch/issues/64062
    if torch.__version__.startswith("1.9."):
        old_val = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_gpu(False)
    yield
    if torch.__version__.startswith("1.9."):
        torch._C._jit_override_can_fuse_on_gpu(old_val)


class InceptionV3(nn.Module):
    """
    Wrapper around Inception V3 torchscript model provided here
    https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt

    path: locally saved inception weights
    """

    def __init__(self, resize_inside=True):
        super(InceptionV3, self).__init__()
        # download the network if it is not present at the given directory
        # use the current directory by default

        with open_url(
            "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
        ) as model:
            self.base = torch.jit.load(model).eval()
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
        with disable_gpu_fuser_on_pt19():
            bs = x.shape[0]
            if self.resize_inside:
                x = (x + 1) * 127.5
                features = self.base(x, return_features=True).view((bs, 2048))
            else:
                # make sure it is resized already
                assert (x.shape[2] == 299) and (x.shape[3] == 299)
                # apply normalization
                x1 = x - 128
                x2 = x1 / 128
                features = self.layers.forward(
                    x2,
                ).view((bs, 2048))
            return features
