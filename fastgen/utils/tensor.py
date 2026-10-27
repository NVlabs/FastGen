# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


PRECISION_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

# Minimum dtype of numerically sensitive method arithmetic on full-size states, such as finite-difference JVPs,
# prediction conversions, and loss kernels. Resolve it with ``high_precision_dtype``, which never demotes a wider
# operand. Setting it to the method precision (``config.model.precision``) skips these up-casts, which can speed up
# training and save memory at some cost in accuracy. Schedules built without a ``t_precision`` also take it as their
# time precision. Times keep their ``t_precision``, as does per-sample arithmetic with them, such as time-dependent
# weights and loss scales and the per-sample losses they scale; these values take the states' dtype only where they
# meet the states.
HIGH_PRECISION = torch.float64


def assert_tensor(
    condition: torch.Tensor,
    message: str,
    error_type: type[Exception] = AssertionError,
) -> None:
    """Check a scalar tensor without synchronizing a successful CUDA path."""
    assert condition.numel() == 1, f"Tensor assertion requires one element, got shape {tuple(condition.shape)}"
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise error_type(message)


def as_tensor(value: float | torch.Tensor, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """*value* as a tensor on *device*: a number fills one, where copying it there would synchronize the device."""
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.full((), value, device=device, dtype=dtype)


def as_array(value: float | torch.Tensor, dtype: torch.dtype) -> np.ndarray:
    """*value* as a NumPy array in *dtype*, which NumPy must have (it has no ``bfloat16``): ``as_tensor`` on the host,
    detached, so the array shares memory with a host tensor already in *dtype*."""
    return as_tensor(value, "cpu", dtype).detach().numpy()


def expand_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand ``x`` with trailing singleton dimensions to match ``target``'s rank.

    Args:
        x: The input tensor to expand.
        target: The tensor whose rank to match.

    Returns:
        ``x`` with trailing singleton dimensions added as needed.
    """
    x = torch.atleast_1d(x)
    while len(x.shape) < len(target.shape):
        x = x[..., None]
    return x


def precision_to_dtype(precision: str | torch.dtype | None, default: torch.dtype | None = None) -> torch.dtype | None:
    """Map a precision string to a torch dtype, returning a dtype unchanged and ``default`` for ``None``.

    Raises ``KeyError`` on an unknown string, so a typo in a config fails loudly rather
    than silently disabling the cast it was meant to configure.
    """
    if isinstance(precision, torch.dtype):
        return precision
    return PRECISION_MAP[precision] if precision is not None else default


def resolve_device(device: torch.device | str | None) -> torch.device | None:
    """Return the concrete device a request names: bare ``cuda`` is the current CUDA device and CPU has no index."""
    if device is None:
        return None
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.type == "cpu":
        return torch.device("cpu")
    return device


def promote_dtype(*operands: torch.Tensor | torch.dtype | None) -> torch.dtype:
    """Resolve the common dtype of tensors/dtypes using PyTorch's promotion rules.

    ``None`` is skipped. There is no implicit precision floor; include a dtype
    operand such as ``torch.float32`` to set one. At least one non-None operand
    is required. Resolution uses dtypes, independently of tensor values or ranks.
    """
    dtype = None
    for operand in operands:
        if operand is not None:
            operand_dtype = operand if isinstance(operand, torch.dtype) else operand.dtype
            dtype = operand_dtype if dtype is None else torch.promote_types(dtype, operand_dtype)
    if dtype is None:
        raise ValueError("promote_dtype requires at least one tensor or dtype")
    return dtype


def to_promoted_dtype(
    *tensors: torch.Tensor | None, min_dtype: torch.dtype | None = None
) -> torch.Tensor | None | tuple[torch.Tensor | None, ...]:
    """Cast tensors to their common dtype, optionally with a ``min_dtype`` floor.

    Preserve devices, gradients, and ``None`` entries. Return a tensor/None for
    one argument and a tuple otherwise, including an empty tuple for no arguments.
    Tensors already in the resolved dtype are returned without a copy.
    """
    if not any(tensor is not None for tensor in tensors):
        return tensors[0] if len(tensors) == 1 else tensors
    dtype = promote_dtype(min_dtype, *tensors)
    cast = tuple(None if tensor is None else tensor.to(dtype=dtype) for tensor in tensors)
    return cast[0] if len(cast) == 1 else cast


def high_precision_dtype(*operands: torch.Tensor | torch.dtype | None) -> torch.dtype:
    """Return one dtype for numerically sensitive arithmetic on *operands*.

    This is ``HIGH_PRECISION`` unless an operand is wider, so casting every operand to it promotes but never
    demotes. Operands are tensors or dtypes; ``None`` operands are skipped.

    The typical contract is to resolve full-size states without the times, which carry their schedule's
    ``t_precision``, and to cast coefficients derived from the times to the states' dtype where the two meet.
    Arithmetic on per-sample values alone (times, weights, per-sample losses) is cheap, so it resolves over all
    of them.
    """
    return promote_dtype(HIGH_PRECISION, *operands)


def to_high_precision(*tensors: torch.Tensor | None) -> torch.Tensor | None | tuple[torch.Tensor | None, ...]:
    """Cast *tensors* to their joint ``high_precision_dtype``; ``None`` passes through.

    Returns a tensor for one argument and a tuple otherwise, like ``torch.atleast_1d``. A tensor that already has
    that dtype is returned without a copy. Group operands as for ``high_precision_dtype``.
    """
    return to_promoted_dtype(*tensors, min_dtype=HIGH_PRECISION)


def einsum_without_autocast(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """``torch.einsum`` that keeps its operands' dtype inside an enclosing autocast region.

    Autocast lowers einsum contractions, which run as batched matmuls, to its low-precision dtype. Use this
    where the caller owns the contraction's precision, such as method arithmetic or a strict FP32 region.
    FP32 contractions still follow the global TF32 setting.
    """
    with torch.autocast(device_type=operands[0].device.type, enabled=False):
        return torch.einsum(equation, *operands)


def dtype_invariant_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """``torch.nn.functional.dropout``'s distribution with a mask drawn as a bool tensor, so the mask depends on the
    RNG state and ``x``'s shape and layout but not on its dtype.

    CUDA's fused dropout assigns random numbers to elements by their size, so forwards that replay one RNG state in
    different autocast dtypes, such as a training forward and its JVP, would drop different activations. The masks
    differ from ``F.dropout``'s; each entry is still kept with probability 1 - p and scaled by 1 / (1 - p).
    """
    if not 0 <= p <= 1:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    if not training or p == 0:
        return x
    if p == 1:
        return x * 0
    keep = torch.empty_like(x, dtype=torch.bool).bernoulli_(1 - p)
    # Selection also zeros dropped gradients when the preceding backward scaling overflows.
    return torch.where(keep, x, 0) * (1 / (1 - p))


@contextmanager
def temp_disable_efficient_attn(device_type: str = "cuda"):
    """Use the math SDPA backend temporarily, which supports forward-mode AD."""
    if device_type != "cuda":
        yield
        return

    backend = torch.backends.cuda
    enabled = {
        "flash": backend.flash_sdp_enabled(),
        "mem_efficient": backend.mem_efficient_sdp_enabled(),
        "cudnn": backend.cudnn_sdp_enabled(),
        "math": backend.math_sdp_enabled(),
    }
    backend.enable_flash_sdp(False)
    backend.enable_mem_efficient_sdp(False)
    backend.enable_cudnn_sdp(False)
    backend.enable_math_sdp(True)
    try:
        yield
    finally:
        backend.enable_flash_sdp(enabled["flash"])
        backend.enable_mem_efficient_sdp(enabled["mem_efficient"])
        backend.enable_cudnn_sdp(enabled["cudnn"])
        backend.enable_math_sdp(enabled["math"])


def unit_pixels_to_uint8(pixels: torch.Tensor) -> torch.Tensor:
    """``[0, 1]`` pixels -> ``uint8`` levels in the same layout, quantized in FP32.

    Out-of-range values saturate, and rounding to the nearest level makes this the exact inverse of ``uint8 / 255``;
    truncation would map some levels one below themselves.
    """
    return pixels.float().clamp(0, 1).mul_(255).round_().to(torch.uint8)


def pixels_to_uint8(pixels: torch.Tensor) -> torch.Tensor:
    """FastGen's ``[-1, 1]`` pixels -> ``uint8`` levels; the exact inverse of ``uint8 / 127.5 - 1``."""
    return unit_pixels_to_uint8(pixels.float().mul(0.5).add_(0.5))


def waveform_to_pcm16(waveform: torch.Tensor) -> torch.Tensor:
    """``[-1, 1]`` waveform -> 16-bit PCM samples in the same layout, saturated and rounded to the nearest level."""
    return waveform.float().clamp(-1, 1).mul_(32767).round_().to(torch.int16)
