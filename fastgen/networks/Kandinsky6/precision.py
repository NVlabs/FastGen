# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Kandinsky 6 arithmetic with BF16 execution and FP32 optimizer storage.

The reference rounds every transformer weight to BF16, then evaluates time and
modulation projections in FP32 and rounds their outputs back to BF16. Diffusers'
FP32 parameter exceptions alone do not reproduce this policy.
"""

import math

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_kandinsky6 import (
    Kandinsky6Modulation,
    Kandinsky6RoPE1D,
    Kandinsky6RoPE3D,
    Kandinsky6TimeEmbeddings,
)

from fastgen.networks.module import FastGenModule


def frequencies(dim, maximum=10000.0):
    return torch.exp(-math.log(maximum) * torch.arange(dim, dtype=torch.float32, device="cpu") / dim)


def linear_fp32(layer, value, execution_dtype):
    weight = layer.weight.to(execution_dtype).float()
    bias = None if layer.bias is None else layer.bias.to(execution_dtype).float()
    return F.linear(value.float(), weight, bias)


class NativeTimeEmbeddings(FastGenModule, Kandinsky6TimeEmbeddings):
    def _fixed_precision_overrides(self):
        return {"native_frequencies": torch.float32}

    def forward(self, timestep):
        mlp = self.timestep_embedder
        dtype = self.precision_model or mlp.linear_1.weight.dtype
        with torch.autocast(timestep.device.type, enabled=False):
            angles = torch.outer(timestep.float(), self.native_frequencies)
            embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
            hidden = linear_fp32(mlp.linear_1, embedding, dtype)
            return linear_fp32(mlp.linear_2, mlp.act(hidden), dtype).to(dtype)


class NativeModulation(FastGenModule, Kandinsky6Modulation):
    def forward(self, value):
        dtype = self.precision_model or self.out_layer.weight.dtype
        with torch.autocast(value.device.type, enabled=False):
            result = linear_fp32(self.out_layer, self.activation(value.float()), dtype)
        return result.to(value.dtype)


class NativeLayerNorm(torch.nn.LayerNorm):
    def forward(self, value):
        weight = None if self.weight is None else self.weight.to(value.dtype)
        bias = None if self.bias is None else self.bias.to(value.dtype)
        # CUDA autocast otherwise promotes native BF16 text/time states to FP32.
        with torch.autocast(value.device.type, enabled=False):
            return F.layer_norm(value, self.normalized_shape, weight, bias, self.eps)


class NativeRMSNorm(torch.nn.RMSNorm):
    def forward(self, value):
        weight = None if self.weight is None else self.weight.to(value.dtype)
        return F.rms_norm(value, self.normalized_shape, weight, self.eps)


class NativeEmbedding(FastGenModule, torch.nn.Embedding):
    def forward(self, indices):
        dtype = self.precision_model or self.weight.dtype
        return F.embedding(indices, self.weight.to(dtype))


def install_native_precision(transformer):
    """Attach policies without renaming pretrained state or replacing parameters."""
    for module in transformer.modules():
        if isinstance(module, Kandinsky6TimeEmbeddings) and not isinstance(module, NativeTimeEmbeddings):
            module.__class__ = NativeTimeEmbeddings
            device = module.timestep_embedder.linear_1.weight.device
            module.register_buffer(
                "native_frequencies", frequencies(module.time_proj.num_channels // 2).to(device), persistent=False
            )
        elif isinstance(module, Kandinsky6Modulation):
            module.__class__ = NativeModulation
        elif isinstance(module, torch.nn.LayerNorm):
            module.__class__ = NativeLayerNorm
        elif isinstance(module, torch.nn.RMSNorm):
            module.__class__ = NativeRMSNorm
        elif isinstance(module, torch.nn.Embedding):
            module.__class__ = NativeEmbedding


def rope_precision_overrides(transformer):
    return {
        f"transformer.{name}.{buffer_name}": torch.float32
        for name, module in transformer.named_modules()
        if isinstance(module, (Kandinsky6RoPE1D, Kandinsky6RoPE3D))
        for buffer_name, _ in module.named_buffers(recurse=False)
    }


def reset_deterministic_buffers(transformer):
    """Rebuild every nonpersistent transformer buffer in place, without random draws."""
    with torch.no_grad():
        for module in transformer.modules():
            if isinstance(module, NativeTimeEmbeddings):
                module.native_frequencies.copy_(frequencies(module.time_proj.num_channels // 2))
            elif isinstance(module, Kandinsky6RoPE1D):
                freq = frequencies(module.dim // 2, module.max_period) * module.freqs_scaling
                module.angles.copy_(torch.outer(torch.arange(module.max_pos, dtype=torch.float32, device="cpu"), freq))
            elif isinstance(module, Kandinsky6RoPE3D):
                for index, (dim, length) in enumerate(zip(module.axes_dims, module.max_pos)):
                    freq = frequencies(dim // 2, module.max_period)
                    getattr(module, f"angles_{index}").copy_(
                        torch.outer(torch.arange(length, dtype=torch.float32, device="cpu"), freq)
                    )
