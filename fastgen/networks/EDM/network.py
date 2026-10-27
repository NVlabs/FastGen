# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The implementation is based on the EDM repo."""

from typing import Callable, Dict, Set, Optional, Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import silu
from torch.distributed.fsdp import fully_shard
from fastgen.networks.network import FastGenNetwork, validate_conditioning
from fastgen.networks.noise_schedule import BaseNoiseSchedule
from fastgen.utils.distributed.fsdp import make_fully_shard_kwargs
from fastgen.utils.module import module_state_names
from fastgen.utils.tensor import dtype_invariant_dropout, einsum_without_autocast, expand_like, promote_dtype
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.tree_utils import OutputDtype, assert_same


# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode="kaiming_normal", init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
        stride=1,
        padding=None,
    ):
        assert not (up and down)
        assert not ((up or down) and (stride != 1 or padding is not None)), "stride and padding exclude resampling"
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        self.stride = stride
        self.padding = padding
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel, fan_out=out_channels * kernel * kernel)
        self.weight = (
            torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight)
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(dtype=x.dtype) if self.resample_filter is not None else None
        w_pad = (w.shape[-1] // 2 if self.padding is None else self.padding) if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad
                )
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# Group normalization.


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x.contiguous(),
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, also under autocast, but uses the
# original datatype for inputs/outputs/gradients to conserve memory.
#
# Add the jvp() implementation in AttentionOp to support the forward-mode AD


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(q, k):
        dtype = promote_dtype(q, torch.float32)
        logits = einsum_without_autocast("ncq,nck->nqk", q.to(dtype), (k / np.sqrt(k.shape[1])).to(dtype))
        w = logits.softmax(dim=2).to(q.dtype)
        return w

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        q, k = inputs
        w = outputs
        ctx.save_for_backward(q, k, w)
        ctx.save_for_forward(q, k, w)

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        dtype = promote_dtype(q, torch.float32)
        db = torch._softmax_backward_data(grad_output=dw.to(dtype), output=w.to(dtype), dim=2, input_dtype=dtype)
        dq = einsum_without_autocast("nck,nqk->ncq", k.to(dtype), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = einsum_without_autocast("ncq,nqk->nck", q.to(dtype), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

    @staticmethod
    def jvp(ctx, v0, v1):
        q, k, w = ctx.saved_tensors
        dtype = promote_dtype(q, torch.float32)
        jvp_q = einsum_without_autocast("ncq,nck->nqk", v0.to(dtype), k.to(dtype)) / np.sqrt(k.shape[1])
        jvp_k = einsum_without_autocast("ncq,nck->nqk", q.to(dtype), v1.to(dtype)) / np.sqrt(k.shape[1])
        jvp_qk = jvp_q + jvp_k
        result = torch._softmax_backward_data(
            grad_output=jvp_qk.to(dtype), output=w.to(dtype), dim=2, input_dtype=dtype
        ).to(q.dtype)

        return result


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(in_features=emb_channels, out_features=out_channels * (2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(dtype_invariant_dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1)
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        freqs = self.freqs.to(dtype=x.dtype)
        x = x.ger(2 * np.pi * freqs)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch


class SongUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 2, 2],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type="positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type="standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type="standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[1, 1],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        r_timestep=False,  # Flag for taking target time r
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        super().__init__()
        self.label_dim = label_dim
        self.label_dropout = label_dropout
        self.num_blocks = num_blocks
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        if r_timestep:
            self.r_timestep = (
                PositionalEmbedding(num_channels=noise_channels, endpoint=True)
                if embedding_type == "positional"
                else FourierEmbedding(num_channels=noise_channels)
            )
        else:
            self.r_timestep = None

        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = (
            Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        )
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer0_r = (
            Linear(in_features=noise_channels, out_features=emb_channels, bias=False, **init) if r_timestep else None
        )
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(
                        in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [block.out_channels for name, block in self.enc.items() if "aux" not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )
        self.logvar_linear = Linear(noise_channels, 1)

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        r_noise_labels=None,
        augment_labels=None,
        return_features_early=False,
        feature_indices=None,
        return_logvar=False,
    ):
        # Mapping.
        # NVLabs EDM keeps the noise label and mapping path in FP32 even when
        # the image path runs in FP16; each UNet block casts the affine result.
        emb_timestep = self.map_noise(noise_labels)
        emb = emb_timestep
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos

        emb_r = None
        if self.r_timestep is not None and r_noise_labels is not None:
            emb_r = self.r_timestep(r_noise_labels)
            emb_r = emb_r.reshape(emb_r.shape[0], 2, -1).flip(1).reshape(*emb_r.shape)  # swap sin/cos
        elif r_noise_labels is not None:
            raise ValueError("r_noise_labels provided, but r_timestep is not set")

        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = self.map_layer0(emb)
        if emb_r is not None:
            emb = emb + self.map_layer0_r(emb_r)
        emb = silu(emb)
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        idx, features = 0, []
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)
                if name.endswith(f"_block{self.num_blocks - 1}"):  # last residual block of each resolution
                    if feature_indices is not None and idx in feature_indices:
                        features.append(x)
                    idx += 1

        # If we have all the features, we can exit early
        if return_features_early:
            assert feature_indices is not None
            assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
            return features

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)

        out = [aux, features] if feature_indices is not None else aux

        if return_logvar:
            logvar = self.logvar_linear(emb_timestep)
            return out, logvar
        else:
            return out


# ----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion


class DhariwalUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 3, 4],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[32, 16, 8],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        r_timestep=False,  # Flag for taking target time r
    ):
        super().__init__()
        self.label_dim = label_dim
        self.label_dropout = label_dropout
        self.num_blocks = num_blocks
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode="kaiming_uniform", init_weight=np.sqrt(1 / 3), init_bias=np.sqrt(1 / 3))
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        if r_timestep:
            self.r_timestep = PositionalEmbedding(num_channels=model_channels)
        else:
            self.r_timestep = None
        self.map_augment = (
            Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero)
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer0_r = (
            Linear(in_features=model_channels, out_features=emb_channels, bias=False, **init) if r_timestep else None
        )
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = (
            Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
            if label_dim
            else None
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs
                )
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs
                )
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

        self.logvar_linear = Linear(model_channels, 1)

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        r_noise_labels=None,
        augment_labels=None,
        return_features_early=False,
        feature_indices=None,
        return_logvar=False,
    ):
        # Mapping.
        # NVLabs EDM keeps the noise label and mapping path in FP32 even when
        # the image path runs in FP16; each UNet block casts the affine result.
        emb_timestep = self.map_noise(noise_labels)
        emb = emb_timestep

        emb_r = None
        if self.r_timestep is not None and r_noise_labels is not None:
            emb_r = self.r_timestep(r_noise_labels)
        elif r_noise_labels is not None:
            raise ValueError("r_noise_labels provided, but r_timestep is not set")

        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = self.map_layer0(emb)
        if emb_r is not None:
            emb = emb + self.map_layer0_r(emb_r)
        emb = silu(emb)
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        idx, features = 0, []
        for name, block in self.enc.items():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            if name.endswith(f"_block{self.num_blocks - 1}"):  # last residual block of each resolution
                if feature_indices is not None and idx in feature_indices:
                    features.append(x)
                idx += 1
            skips.append(x)

        # If we have all the features, we can exit early
        if return_features_early:
            assert feature_indices is not None
            assert len(features) == len(feature_indices)
            return features

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))

        out = [x, features] if feature_indices is not None else x

        if return_logvar:
            logvar = self.logvar_linear(emb_timestep)
            return out, logvar
        else:
            return out


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


class EDMPreconditioningMixin:
    """Shared EDM equations and their precision/policy boundary."""

    # Annotation-only requirements supplied by FastGenNetwork and the concrete wrappers.
    drop_precond: Optional[str]
    cast_to_model_precision: Callable[[torch.Tensor], torch.Tensor]
    sigma_data: float
    net_pred_type: str
    noise_scheduler: BaseNoiseSchedule
    _resolve_conditions: Callable[..., dict]
    transform_output: Callable[..., torch.Tensor]

    def build_conditioning(self, data, input_shape=None, precision=None):
        """Resolve the labels directly in FP32: they enter NVLabs' FP32 embedding path alongside the noise label.

        The base builder's ``precision_model`` cast would round the labels first.
        """
        return self._resolve_conditions(data, dtype=torch.float32)

    def precond_input(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply the EDM input and noise-label preconditioning equations."""
        c_in = 1 / (self.sigma_data**2 + t**2).sqrt()
        c_in = expand_like(c_in.to(x_t.dtype), x_t)
        x_t = c_in * x_t

        t = t.clamp(min=self.noise_scheduler.clamp_min).log() / 4
        if r is not None:
            r = r.clamp(min=self.noise_scheduler.clamp_min).log() / 4
        return x_t, t, r

    def precond_output(
        self,
        prediction: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        sigma_shift: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply the EDM skip and output preconditioning equations."""
        if sigma_shift is not None:
            t = t - sigma_shift
        c_skip = self.sigma_data**2 / (t**2 + self.sigma_data**2)
        c_out = t * self.sigma_data / (t**2 + self.sigma_data**2).sqrt()

        c_skip = expand_like(c_skip.to(x_t.dtype), x_t)
        c_out = expand_like(c_out.to(x_t.dtype), x_t)
        return c_skip * x_t + c_out * prediction

    def _prepare_edm_inputs(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Prepare the image and time inputs while keeping EDM arithmetic in FP32."""
        assert_same(
            shape=True,
            dtype=self.noise_scheduler.t_precision,
            device=True,
            t=t,
            r=r,
        )
        t = t.float()
        r = None if r is None else r.float()

        if self.drop_precond not in ["input", "both"]:
            x_t, t, r = self.precond_input(x_t.float(), t, r=r)

        return self.cast_to_model_precision(x_t), t, r

    def _finalize_edm_prediction(
        self,
        prediction: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        sigma_shift: Optional[float],
        fwd_pred_type: Optional[str],
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
    ) -> torch.Tensor:
        """Apply the FP32 EDM output path, then end the forward.

        Like the reference's ``D_x``, skip/output preconditioning runs in FP32 unless dropped.
        ``transform_output`` then converts it to ``fwd_pred_type`` and rounds once to ``output_dtype``,
        by default the incoming state dtype.
        """
        if self.drop_precond not in ["output", "both"]:
            prediction = self.precond_output(prediction.float(), x_t.float(), t.float(), sigma_shift=sigma_shift)
        return self.transform_output(prediction, x_t, t, None, fwd_pred_type, output_dtype, apply_output_conditioning)


class EDMPrecond(EDMPreconditioningMixin, FastGenNetwork):
    """EDM preconditioning wrapper (Karras et al. 2022).

    Wraps a ``SongUNet`` or ``DhariwalUNet`` ``backbone``, configured as its own nested
    ``L(...)`` node and stored as ``self.model``. Runs on pixels: ``real_raw`` is used as the
    model-space state. To apply the same preconditioning / σ schedule / Euler sampler to VAE
    latents, supply precomputed latents as ``real``.
    """

    latent_compression_factors = (1, 1)

    def encode_real(self, real_raw):
        return real_raw

    def __init__(
        self,
        backbone: nn.Module,  # SongUNet or DhariwalUNet, built from its own config node.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        sigma_shift=0.0,  # Shift sigma during inference as done in ECT.
        drop_precond=None,  # Can be set to 'input'/'output'/'both' to drop the preconditioning of the input/output/both.
        net_pred_type="x0",  # Prediction type for FastGenNetwork
        schedule_type="edm",  # Schedule type for FastGenNetwork
        **schedule_kwargs,  # Keyword arguments for the noise scheduler.
    ):
        # Initialize FastGenNetwork with EDM-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)

        self.sigma_data = sigma_data
        self.sigma_shift = sigma_shift
        if drop_precond is not None and drop_precond not in ["input", "output", "both"]:
            raise ValueError(f"drop_precond must be one of 'input', 'output', 'both', or None, got {drop_precond}")
        self.drop_precond = drop_precond
        if not isinstance(backbone, nn.Module):
            raise TypeError(
                f"backbone must be an instantiated module such as L(SongUNet)(...), got {type(backbone).__name__}"
            )
        self.model = backbone
        self.label_dim = backbone.label_dim

    def _fixed_precision_overrides(self) -> Dict[str, torch.dtype]:
        """Keep the mapping path in FP32, the precision NVLabs EDM computes it in from the stored weights.

        The image path casts every weight to its input dtype, so with these pinned a lower-precision copy computes
        exactly what FP32 master weights compute.
        """
        prefixes = ("model.map_", "model.r_timestep.", "model.logvar_linear.")
        return {
            **super()._fixed_precision_overrides(),
            **{
                name: torch.float32
                for name in module_state_names(self)
                if name.startswith(prefixes) or ".affine." in name
            },
        }

    def fully_shard(self, **kwargs):
        """Fully shard the EDM network for FSDP.

        Note: EDM wraps a U-Net architecture (SongUNet or DhariwalUNet) with preconditioning.
        The underlying model has encoder (enc) and decoder (dec) blocks stored as ModuleDicts.
        We shard each block for optimal memory efficiency. The FP32 mapping layers
        (``_fixed_precision_overrides``) get their own groups when the enclosing policy uses another dtype.
        """
        fp32_kwargs = make_fully_shard_kwargs(
            kwargs, param_dtype=torch.float32, storage_dtype=self.precision_opt, enabled=True
        )

        def shard_block(block):
            if fp32_kwargs is not None:
                fully_shard(block.affine, **fp32_kwargs)
            fully_shard(block, **kwargs)

        # Shard encoder blocks
        for name, block in self.model.enc.items():
            if isinstance(block, UNetBlock):
                shard_block(block)

        # Shard decoder blocks
        for name, block in self.model.dec.items():
            if isinstance(block, UNetBlock):
                shard_block(block)

        # Shard the FP32 mapping layers, then the entire model
        if fp32_kwargs is not None:
            for name in ("map_label", "map_augment", "map_layer0", "map_layer0_r", "map_layer1", "logvar_linear"):
                layer = getattr(self.model, name)
                if layer is not None:
                    fully_shard(layer, **fp32_kwargs)
        fully_shard(self.model, **kwargs)

    def build_conditioning(self, data, input_shape=None, precision=None):
        """Resolve ordinary conditioning, then attach preprocessing-owned augmentation labels."""
        out = super().build_conditioning(data, input_shape=input_shape, precision=precision)
        validate_conditioning(out, require_tensor=True)
        if data.get("aug_condition") is None:
            return out

        aug_condition = data["aug_condition"].to(dtype=torch.float32)
        out["condition"] = {
            "orig_condition": out.get("condition"),
            "aug_condition": aug_condition,
        }
        if out.get("neg_condition") is not None:
            out["neg_condition"] = {
                "orig_condition": out["neg_condition"],
                "aug_condition": aug_condition,
            }
        return out

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        self._validate_forward_kwargs(
            fwd_kwargs,
            r=r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            supports_r_timestep=getattr(self.model, "r_timestep", None) is not None,
            num_features=sum(name.endswith(f"_block{self.model.num_blocks - 1}") for name in self.model.enc),
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        # Extract augmentation labels if condition is a dict
        if isinstance(condition, dict) and "aug_condition" in condition:
            augment_labels = condition["aug_condition"]
            condition = condition["orig_condition"]
        else:
            augment_labels = None

        # NVLabs EDM preconditions in FP32: the state and sigma are upcast, only the network input
        # is cast to the model dtype, and F_x is cast back to FP32 for the skip/output arithmetic.
        model_input, model_t, model_r = self._prepare_edm_inputs(x_t, t, r)
        assert_same(
            dtype=torch.float32,
            device=model_input.device,
            model_t=model_t,
            condition=condition,
            augment_labels=augment_labels,
        )
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=model_t.device, dtype=model_t.dtype)
            if condition is None
            else condition.reshape(-1, self.label_dim)
        )
        model_outputs = self.model(
            model_input,
            model_t,
            class_labels=class_labels,
            r_noise_labels=model_r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            augment_labels=augment_labels,
        )

        if return_features_early:
            return model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        sigma_shift = None if self.training else self.sigma_shift

        output_args = (sigma_shift, fwd_pred_type, apply_output_conditioning, output_dtype)
        if feature_indices is None:
            assert isinstance(out, torch.Tensor)
            out = self._finalize_edm_prediction(out, x_t, t, *output_args)
        else:
            assert isinstance(out, list)
            out[0] = self._finalize_edm_prediction(out[0], x_t, t, *output_args)

        if return_logvar:
            return out, logvar.to(dtype=x_t.dtype)
        return out

    @staticmethod
    def default_guidance_cfg() -> GuidanceConfig:
        """Default guidance recipe: no CFG (guidance disabled)."""
        return GuidanceConfig()

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        sample_steps: int = 50,
        sample_type: str = "ode",
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Generate samples using a deterministic EDM/RF Euler sampler.

        Args:
            noise: Initial standard-normal noise tensor [B, C, H, W].
            condition: Dense floating-point class-label vectors.
            neg_condition: Negative conditioning for CFG.
            sample_steps: Number of sampling steps.
            sample_type: Integration policy; only ``"ode"`` is supported.
            guidance_cfg: Composite guidance config; ``None`` →
                ``default_guidance_cfg()`` (no CFG by default).

        Returns:
            Generated samples of shape [B, C, H, W] in the space of the training
            state (pixels, or latents when trained on precomputed latents).
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("edm", "rf"),
            supported_sample_types=("ode",),
        )

        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()

        grid = self.noise_scheduler.grid(sample_steps, device=noise.device)

        x = self.noise_scheduler.latents(noise=noise, t_init=grid[0])
        for t_scalar, t_next in zip(grid[:-1], grid[1:]):
            t = t_scalar.expand(x.shape[0])
            fwd_pred_type = "x0" if self.schedule_type == "edm" else "flow"

            prediction = predict_with_guidance(
                self,
                x,
                t,
                batched=True,
                guidance_cfg=guidance_cfg,
                condition=condition,
                neg_condition=neg_condition,
                output_dtype=None,
                fwd_pred_type=fwd_pred_type,
            )

            # The DDIM step, which is the Euler step on EDM (alpha = 1, sigma = t) and on unshifted RF.
            x = self.noise_scheduler.step(x, prediction, t, t_next, pred_type=fwd_pred_type)

        return x


# ----------------------------------------------------------------------------
