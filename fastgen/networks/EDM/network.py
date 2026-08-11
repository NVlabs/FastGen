# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The implementation is based on the EDM repo."""

from typing import Set, Optional, Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import silu
from torch.distributed.fsdp import fully_shard
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils import expand_like
import fastgen.utils.logging_utils as logger


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
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
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
        # Move buffer to input device and dtype
        f = self.resample_filter.to(device=x.device, dtype=x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
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
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
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
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.
#
# Add the jvp() implementation in AttentionOp to support the forward-mode AD


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(q, k):
        w = (
            torch.einsum("ncq,nck->nqk", q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32))
            .softmax(dim=2)
            .to(q.dtype)
        )
        return w

    def setup_context(ctx, inputs, outputs):
        q, k = inputs
        w = outputs
        ctx.save_for_backward(q, k, w)
        ctx.save_for_forward(q, k, w)

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

    @staticmethod
    def jvp(ctx, v0, v1):
        q, k, w = ctx.saved_tensors
        jvp_k = torch.einsum("ncq,nck->nqk", v0.to(torch.float32), k.to(torch.float32)) / np.sqrt(k.shape[1])
        jvp_q = torch.einsum("ncq,nck->nqk", q.to(torch.float32), v1.to(torch.float32)) / np.sqrt(k.shape[1])
        jvp_kq = jvp_k + jvp_q
        result = torch._softmax_backward_data(
            grad_output=jvp_kq.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32
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

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
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
        # Move buffer to input device and dtype (needed for FSDP CPU offloading)
        freqs = self.freqs.to(device=x.device, dtype=x.dtype)
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
        **kwargs,
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        cond_channels = noise_channels * (1 + r_timestep)
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

        self.map_label = Linear(in_features=label_dim, out_features=cond_channels, **init) if label_dim else None
        self.map_augment = (
            Linear(in_features=augment_dim, out_features=cond_channels, bias=False, **init) if augment_dim else None
        )
        self.map_layer0 = Linear(in_features=cond_channels, out_features=emb_channels, **init)
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
        emb_timestep = self.map_noise(noise_labels)
        emb = emb_timestep
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos

        if r_noise_labels is not None:
            if self.r_timestep is not None:
                emb_r = self.r_timestep(r_noise_labels)
                emb_r = emb_r.reshape(emb_r.shape[0], 2, -1).flip(1).reshape(*emb_r.shape)  # swap sin/cos
                emb = torch.cat([emb, emb_r], dim=-1)
            else:
                raise ValueError("r_noise_labels provided, but r_timestep is not set")

        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
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
                if "block3" in name:
                    if idx in feature_indices:
                        features.append(x)
                    idx += 1

        # If we have all the features, we can exit early
        if return_features_early:
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

        if len(feature_indices) == 0:
            # no features requested, return only the model output
            out = aux
        else:
            # score and features； score, features
            out = [aux, features]

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
        dropout=0.10,  # List of resolutions with self-attention.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        r_timestep=False,  # Flag for taking target time r
        **kwargs,
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        cond_channels = model_channels * (1 + r_timestep)
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
            Linear(in_features=augment_dim, out_features=cond_channels, bias=False, **init_zero)
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(in_features=cond_channels, out_features=emb_channels, **init)
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
        emb_timestep = self.map_noise(noise_labels)
        emb = emb_timestep

        if self.r_timestep is not None:
            if r_noise_labels is not None:
                emb_r = self.r_timestep(r_noise_labels)
                emb = torch.cat([emb, emb_r], dim=-1)
            else:
                raise ValueError("r_noise_labels provided, but r_timestep is not set")

        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
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
            if "block2" in name:
                if idx in feature_indices:
                    features.append(x)
                idx += 1
            skips.append(x)

        # If we have all the features, we can exit early
        if return_features_early:
            assert len(features) == len(feature_indices)
            return features

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))

        if len(feature_indices) == 0:
            # no features requested, return only the model output
            out = x
        else:
            # score and features； score, features
            out = [x, features]

        if return_logvar:
            logvar = self.logvar_linear(emb_timestep)
            return out, logvar
        else:
            return out


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


def precond_input(
    x_t: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, sigma_data: float = 0.5, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precondition the input of the model.

    Args:
        x_t: The input tensor.
        t: The timestep tensor.
        r: The target time tensor.
        sigma_data: The standard deviation of the data.
        eps: The epsilon value.

    Returns:
        The preconditioned input tensors (x_t, t, r).
    """
    c_in = 1 / (sigma_data**2 + t**2).sqrt()
    c_in = expand_like(c_in.to(x_t.dtype), x_t)
    x_t = c_in * x_t

    t = t.clamp(min=eps).log() / 4
    if r is not None:
        r = r.clamp(min=eps).log() / 4
    return x_t, t, r


def precond_output(
    out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, sigma_shift: Optional[float] = None, sigma_data: float = 0.5
) -> torch.Tensor:
    """
    Precondition the output of the model.

    Args:
        out: The output tensor.
        x_t: The input tensor.
        t: The timestep tensor.
        sigma_shift: The shift in the timestep.
        sigma_data: The standard deviation of the data.

    Returns:
        The preconditioned output tensor.
    """
    # Preconditioning weights for output
    if sigma_shift is not None:
        t = t - sigma_shift
    c_skip = sigma_data**2 / (t**2 + sigma_data**2)
    c_out = t * sigma_data / (t**2 + sigma_data**2).sqrt()

    c_skip = expand_like(c_skip.to(x_t.dtype), x_t)
    c_out = expand_like(c_out.to(x_t.dtype), x_t)
    return c_skip * x_t + c_out * out


class EDMPrecond(FastGenNetwork):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        sigma_shift=0.0,  # Shift sigma during inference as done in ECT.
        model_type="DhariwalUNet",  # Class name of the underlying model.
        drop_precond=None,  # Can be set to 'input'/'output'/'both' to drop the preconditioning of the input/output/both.
        net_pred_type="x0",  # Prediction type for FastGenNetwork
        schedule_type="edm",  # Schedule type for FastGenNetwork
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        # Initialize FastGenNetwork with EDM-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.sigma_data = sigma_data
        self.sigma_shift = sigma_shift
        if drop_precond is not None and drop_precond not in ["input", "output", "both"]:
            raise ValueError(f"drop_precond must be one of 'input', 'output', 'both', or None, got {drop_precond}")
        self.drop_precond = drop_precond
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def reset_parameters(self):
        """Reinitialize parameters for FSDP meta device initialization.

        This is required when using meta device initialization for FSDP2.
        Reinitializes all linear and convolutional layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        super().reset_parameters()

    def fully_shard(self, **kwargs):
        """Fully shard the EDM network for FSDP.

        Note: EDM wraps a U-Net architecture (SongUNet or DhariwalUNet) with preconditioning.
        The underlying model has encoder (enc) and decoder (dec) blocks stored as ModuleDicts.
        We shard each block for optimal memory efficiency.
        """
        # Shard encoder blocks
        for name, block in self.model.enc.items():
            if isinstance(block, UNetBlock):
                fully_shard(block, **kwargs)

        # Shard decoder blocks
        for name, block in self.model.dec.items():
            if isinstance(block, UNetBlock):
                fully_shard(block, **kwargs)

        # Shard the entire model
        fully_shard(self.model, **kwargs)

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
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            # Exit immediately if user requested this.
            return []
        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"

        # Extract augmentation labels if condition is a dict
        if isinstance(condition, dict) and "aug_condition" in condition:
            augment_labels = condition["aug_condition"]
            condition = condition["orig_condition"]
            # Check if augment dimension matches the model's expected dimension
            if hasattr(self.model, "map_augment") and self.model.map_augment is not None:
                expected_dim = self.model.map_augment.in_features
                if augment_labels.shape[-1] != expected_dim:
                    logger.warning(
                        f"Augment labels dimension mismatch: got {augment_labels.shape[-1]}, "
                        f"expected {expected_dim}. Ignoring augment_labels."
                    )
                    augment_labels = None  # Dimension mismatch, don't use augment_labels
        else:
            augment_labels = None

        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x_t.device, dtype=x_t.dtype)
            if condition is None
            else condition.reshape(-1, self.label_dim)
        )

        # Preconditioning weights for input
        x_t_in, t_in = x_t, t
        if self.drop_precond not in ["input", "both"]:
            x_t, t, r = precond_input(x_t, t, r=r, sigma_data=self.sigma_data, eps=self.noise_scheduler.clamp_min)

        t = t.to(x_t.dtype)
        if r is not None:
            r = r.to(x_t.dtype)

        model_outputs = self.model(
            x_t,
            t,
            class_labels=class_labels,
            r_noise_labels=r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            augment_labels=augment_labels,
            **fwd_kwargs,
        )

        if return_features_early:
            return model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        sigma_shift = None if self.training else self.sigma_shift
        if len(feature_indices) == 0:
            assert isinstance(out, torch.Tensor)
            if self.drop_precond not in ["output", "both"]:
                out = precond_output(out, x_t_in, t_in, sigma_shift=sigma_shift, sigma_data=self.sigma_data)
            out = self.noise_scheduler.convert_model_output(
                x_t_in, out, t_in, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            assert isinstance(out, list)
            if self.drop_precond not in ["output", "both"]:
                out[0] = precond_output(out[0], x_t_in, t_in, sigma_shift=sigma_shift, sigma_data=self.sigma_data)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t_in, out[0], t_in, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )

        if return_logvar:
            return out, logvar
        return out

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using EDM's deterministic Euler sampler.

        Args:
            noise: Initial noise tensor [B, C, H, W] (should be scaled by max sigma).
            condition: Class label conditioning (one-hot or class indices).
            neg_condition: Negative conditioning for CFG.
            guidance_scale: CFG guidance scale. None disables guidance.
            num_steps: Number of sampling steps.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated samples in latent space.
        """
        assert self.schedule_type == "edm", f"{self.schedule_type} is not supported"

        # Get sigma schedule from noise scheduler (in EDM, t == sigma)
        sigmas = self.noise_scheduler.get_t_list(num_steps, device=noise.device)

        x = self.noise_scheduler.latents(noise=noise, t_init=sigmas[0])
        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            # Expand sigma for batch
            t = sigma.expand(x.shape[0])

            # Get x0 prediction with optional CFG
            if guidance_scale is not None and guidance_scale > 1.0 and neg_condition is not None:
                # CFG: predict with both conditions
                x_input = torch.cat([x, x], dim=0)
                t_input = torch.cat([t, t], dim=0)
                cond_input = torch.cat([neg_condition, condition], dim=0)

                x0_pred = self(x_input, t_input, condition=cond_input, fwd_pred_type="x0")
                x0_uncond, x0_cond = x0_pred.chunk(2)
                x0_pred = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            else:
                x0_pred = self(x, t, condition=condition, fwd_pred_type="x0")

            # EDM Euler step: x_next = x0 + sigma_next * (x - x0) / sigma
            # Equivalent to: x_next = x + (sigma_next - sigma) * (x - x0) / sigma
            d = (x - x0_pred) / expand_like(t, x)  # derivative estimate
            x = x + (sigma_next - sigma).to(x.dtype) * d

        return x


# ----------------------------------------------------------------------------
