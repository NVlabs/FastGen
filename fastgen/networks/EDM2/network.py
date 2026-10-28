# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The implementation is based on the EDM2 repo."""

from typing import Dict, Set, Optional, Union, List, Tuple
import numpy as np
import torch
from torch.distributed.fsdp import fully_shard

from fastgen.networks.network import FastGenNetwork

from fastgen.networks.EDM.network import EDMPreconditioningMixin
from fastgen.utils.distributed.fsdp import make_fully_shard_kwargs
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.module import module_state_names
from fastgen.utils.tensor import dtype_invariant_dropout, promote_dtype
from fastgen.utils.tree_utils import OutputDtype, assert_same

# ----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


# ----------------------------------------------------------------------------
# Variant of constant() that inherits dtype and device from the given
# reference tensor by default.


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    dtype = promote_dtype(x, torch.float32)
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=dtype)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


# ----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.


def resample(x, f=[1, 1], mode="keep"):
    if mode == "keep":
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = const_like(x, f)
    c = x.shape[1]
    if mode == "down":
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == "up"
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))


# ----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


# ----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t**2)


# ----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


# ----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1.0):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        dtype = promote_dtype(x, torch.float32)
        y = x.to(dtype)
        y = y.ger(self.freqs.to(dtype=dtype))
        y = y + self.phases.to(dtype=dtype)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


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
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).


@torch.no_grad()
def normalize_weights(m):
    if isinstance(m, MPConv):
        m.weight.copy_(normalize(m.weight))


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        dtype = promote_dtype(x, torch.float32)
        w = self.weight.to(dtype)
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


# ----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).


class Block(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        emb_channels,  # Number of embedding channels.
        flavor="enc",  # Flavor: 'enc' or 'dec'.
        resample_mode="keep",  # Resampling: 'keep', 'up', or 'down'.
        resample_filter=[1, 1],  # Resampling filter.
        attention=False,  # Include self-attention?
        channels_per_head=64,  # Number of channels per attention head.
        dropout=0,  # Dropout probability.
        res_balance=0.3,  # Balance between main branch (0) and residual branch (1).
        attn_balance=0.3,  # Balance between main branch (0) and self-attention (1).
        clip_act=256,  # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([1]))
        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels, out_channels, kernel=[3, 3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1, 1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1, 1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1, 1]) if self.num_heads != 0 else None

    def forward(self, x, emb):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = dtype_invariant_dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3)  # pixel norm & split
            w = torch.einsum("nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum("nhqk,nhck->nhcq", w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


# ----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).


class EDM2UNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Image channels.
        label_dim,  # Class label dimensionality. 0 = unconditional.
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 3, 4],  # Per-resolution multipliers for the number of channels.
        channel_mult_noise=None,
        # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb=None,
        # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[16, 8],  # List of resolutions with self-attention.
        label_balance=0.5,  # Balance between noise embedding (0) and class embedding (1).
        concat_balance=0.5,  # Balance between skip connections (0) and main path (1).
        dropout=0,  # Dropout probability.
        dropout_resolutions=None,  # List of resolutions at which to apply dropout. None = all resolutions.
        embedding_type="mp_fourier",  # time embedding types: ["mp_fourier", "positional"]
        mp_fourier_bandwidth=1.0,  # bandwidth for the mp_fourier embedding
        r_timestep=False,  # Flag for taking target time r
        **block_kwargs,  # Arguments for Block.
    ):
        assert embedding_type in ["mp_fourier", "positional"]

        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.dropout = dropout
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.num_blocks = num_blocks
        self.out_gain = torch.nn.Parameter(torch.zeros([1]))

        # Embedding.
        self.emb_fourier = (
            PositionalEmbedding(cnoise, endpoint=True)
            if embedding_type == "positional"
            else MPFourier(cnoise, bandwidth=mp_fourier_bandwidth)
        )
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])

        if r_timestep:
            self.emb_fourier_r = (
                PositionalEmbedding(cnoise, endpoint=True)
                if embedding_type == "positional"
                else MPFourier(cnoise, bandwidth=mp_fourier_bandwidth)
            )
            self.emb_noise_r = MPConv(cnoise, cemb, kernel=[])
        else:
            self.emb_fourier_r = None
            self.emb_noise_r = None

        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            dout = self.dropout if (dropout_resolutions is None or res in dropout_resolutions) else 0

            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=[3, 3])
            else:
                self.enc[f"{res}x{res}_down"] = Block(
                    cout, cout, cemb, flavor="enc", resample_mode="down", dropout=dout, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_block{idx}"] = Block(
                    cin, cout, cemb, flavor="enc", attention=(res in attn_resolutions), dropout=dout, **block_kwargs
                )

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            dout = self.dropout if (dropout_resolutions is None or res in dropout_resolutions) else 0

            if level == len(cblock) - 1:
                self.dec[f"{res}x{res}_in0"] = Block(
                    cout, cout, cemb, flavor="dec", attention=True, dropout=dout, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = Block(cout, cout, cemb, flavor="dec", dropout=dout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = Block(
                    cout, cout, cemb, flavor="dec", resample_mode="up", dropout=dout, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"{res}x{res}_block{idx}"] = Block(
                    cin, cout, cemb, flavor="dec", attention=(res in attn_resolutions), dropout=dout, **block_kwargs
                )
        self.out_conv = MPConv(cout, img_channels, kernel=[3, 3])

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        r_noise_labels=None,
        return_features_early=False,
        feature_indices=None,
    ):
        # Embedding.
        # NVLabs EDM2 keeps c_noise and its Fourier/mapping path in FP32 even
        # when the image UNet runs in FP16.
        emb = self.emb_noise(self.emb_fourier(noise_labels))

        if r_noise_labels is not None:
            if self.emb_noise_r is not None:
                emb_r = self.emb_noise_r(self.emb_fourier_r(r_noise_labels))
                emb = mp_sum(emb, emb_r, t=0.5)
            else:
                raise ValueError("r_noise_labels provided, but r_timestep is not set")

        if self.emb_label is not None:
            emb = mp_sum(
                emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])).to(emb.dtype), t=self.label_balance
            )
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        idx, features = 0, []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)
            if name.endswith(f"_block{self.num_blocks - 1}"):  # last residual block of each resolution
                if feature_indices is not None and idx in feature_indices:
                    features.append(x)
                idx += 1

        # If we have all the features, we can exit early
        if return_features_early:
            assert feature_indices is not None
            assert len(features) == len(feature_indices)
            return features

        # Decoder.
        for name, block in self.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)

        out = [x, features] if feature_indices is not None else x
        return out


# ----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.


class EDM2Precond(EDMPreconditioningMixin, FastGenNetwork):
    """EDM2 preconditioning wrapper (Karras et al. 2024).

    Runs on pixels to match this repo's ImageNet-64 configs: ``real_raw`` is used
    as the model-space state. For latent EDM2 (e.g. SD-VAE ImageNet-512, as in the
    original paper), supply precomputed latents as ``real``.
    """

    latent_compression_factors = (1, 1)

    def encode_real(self, real_raw):
        return real_raw

    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Image channels.
        label_dim,  # Class label dimensionality. 0 = unconditional.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        sigma_shift=0.0,  # Shift sigma during inference as done in ECT.
        logvar_channels=128,  # Intermediate dimensionality for uncertainty estimation.
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 3, 4],  # Per-resolution multipliers for the number of channels.
        channel_mult_noise=None,  # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb=None,  # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[16, 8],  # List of resolutions with self-attention.
        label_balance=0.5,  # Balance between noise embedding (0) and class embedding (1).
        concat_balance=0.5,  # Balance between skip connections (0) and main path (1).
        dropout=0,  # Dropout probability.
        dropout_resolutions=None,  # List of resolutions at which to apply dropout. None = all resolutions.
        embedding_type="mp_fourier",  # Time embedding type: "mp_fourier" or "positional".
        mp_fourier_bandwidth=1.0,  # Bandwidth of the mp_fourier embedding.
        r_timestep=False,  # Flag for taking target time r
        drop_precond=None,  # Can be set to 'input'/'output'/'both' to drop the preconditioning of the input/output/both.
        net_pred_type="x0",  # Prediction type for FastGenNetwork
        schedule_type="edm",  # Schedule type for FastGenNetwork
        **schedule_kwargs,  # Keyword arguments for the noise scheduler.
    ):
        # Initialize FastGenNetwork with EDM2-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)

        self.label_dim = label_dim
        self.sigma_data = sigma_data
        self.sigma_shift = sigma_shift
        if drop_precond is not None and drop_precond not in ["input", "output", "both"]:
            raise ValueError(f"drop_precond must be one of 'input', 'output', 'both', or None, got {drop_precond}")
        self.drop_precond = drop_precond
        self.unet = EDM2UNet(
            img_resolution=img_resolution,
            img_channels=img_channels,
            label_dim=label_dim,
            model_channels=model_channels,
            channel_mult=channel_mult,
            channel_mult_noise=channel_mult_noise,
            channel_mult_emb=channel_mult_emb,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            label_balance=label_balance,
            concat_balance=concat_balance,
            dropout=dropout,
            dropout_resolutions=dropout_resolutions,
            embedding_type=embedding_type,
            mp_fourier_bandwidth=mp_fourier_bandwidth,
            r_timestep=r_timestep,
        )
        self.logvar_fourier = (
            PositionalEmbedding(logvar_channels, endpoint=True)
            if embedding_type == "positional"
            else MPFourier(logvar_channels, bandwidth=mp_fourier_bandwidth)
        )
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forced_weight_normalization(self):
        self.apply(normalize_weights)

    def _state_dtype(self) -> torch.dtype:
        """The storage dtype of all EDM2 state: the requested storage, but at least FP32."""
        return promote_dtype(self.precision_opt, torch.float32)

    def _fixed_precision_overrides(self) -> Dict[str, torch.dtype]:
        """Keep all state in at least FP32, as NVLabs EDM2 stores it in every precision.

        ``MPConv`` normalizes and scales each stored weight in at least FP32 before casting it to its input dtype,
        and the embedding path runs in FP32, so only this storage makes a frozen or inference copy compute exactly
        what FP32 master weights compute.
        """
        owned = module_state_names(self, exclude=self._nearest_nested_modules())
        return {**super()._fixed_precision_overrides(), **{name: self._state_dtype() for name in owned}}

    def fully_shard(self, **kwargs):
        """Fully shard the EDM2 network for FSDP.

        Note: EDM2 wraps the EDM2UNet with preconditioning.
        The underlying unet has encoder (enc) and decoder (dec) blocks stored as ModuleDicts.
        We shard each Block for optimal memory efficiency. Every group gathers the storage dtype that
        ``_fixed_precision_overrides`` pins, which ``MPConv`` casts to its input after normalizing.
        """
        kwargs = (
            make_fully_shard_kwargs(
                kwargs, param_dtype=self._state_dtype(), storage_dtype=self.precision_opt, enabled=True
            )
            or kwargs
        )

        # Shard encoder blocks
        for name, block in self.unet.enc.items():
            if isinstance(block, Block):
                fully_shard(block, **kwargs)

        # Shard decoder blocks
        for name, block in self.unet.dec.items():
            if isinstance(block, Block):
                fully_shard(block, **kwargs)

        # Shard the entire unet
        fully_shard(self.unet, **kwargs)
        fully_shard(self.logvar_linear, **kwargs)

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
            supports_r_timestep=self.unet.emb_noise_r is not None,
            num_features=sum(name.endswith(f"_block{self.unet.num_blocks - 1}") for name in self.unet.enc),
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        # NVLabs EDM2 preconditions in FP32: the state and sigma are upcast, only the network input
        # is cast to the model dtype, and F_x is cast back to FP32 for the skip/output arithmetic.
        model_input, model_t, model_r = self._prepare_edm_inputs(x_t, t, r)
        assert_same(dtype=torch.float32, device=model_input.device, model_t=model_t, condition=condition)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=model_input.device, dtype=model_t.dtype)
            if condition is None
            else condition.reshape(-1, self.label_dim)
        )
        out = self.unet(
            model_input,
            model_t,
            class_labels=class_labels,
            r_noise_labels=model_r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
        )

        if return_features_early:
            return out

        sigma_shift = None if self.training else self.sigma_shift

        output_args = (sigma_shift, fwd_pred_type, apply_output_conditioning, output_dtype)
        if feature_indices is None:
            assert isinstance(out, torch.Tensor)
            out = self._finalize_edm_prediction(out, x_t, t, *output_args)
        else:
            assert isinstance(out, list)
            out[0] = self._finalize_edm_prediction(out[0], x_t, t, *output_args)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(model_t)).reshape(-1, 1)
            return out, logvar.to(dtype=x_t.dtype)  # u(sigma) in Equation 21
        return out

    @staticmethod
    def default_guidance_cfg() -> GuidanceConfig:
        """Default guidance recipe: no CFG (guidance disabled).

        Note that ImageNet-512 models benefit from CFG (~1.4-2.2); such a config should set
        ``config.model.guidance_cfg.cfg.scale`` explicitly.
        """
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
        """Generate samples with deterministic first-order Euler steps on the schedule's grid.

        EDM2's reference ``edm_sampler`` adds Heun's second-order correction to every step but the last.

        Args:
            noise: Initial standard-normal noise tensor [B, C, H, W]; the sampler scales it to the first sigma.
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
            supported_schedule_types=("edm",),
            supported_sample_types=("ode",),
        )

        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()

        # Get sigma schedule from noise scheduler (in EDM, t == sigma)
        sigmas = self.noise_scheduler.grid(sample_steps, device=noise.device)

        x = self.noise_scheduler.latents(noise=noise, t_init=sigmas[0])
        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            # Expand sigma for batch
            t = sigma.expand(x.shape[0])

            x0_pred = predict_with_guidance(
                self,
                x,
                t,
                batched=True,
                guidance_cfg=guidance_cfg,
                condition=condition,
                neg_condition=neg_condition,
                output_dtype=None,
                fwd_pred_type="x0",
            )

            # The DDIM step, x_next = x0 + sigma_next * (x - x0) / sigma: the Euler step on EDM.
            x = self.noise_scheduler.step(x, x0_pred, t, sigma_next)

        return x


# ----------------------------------------------------------------------------
