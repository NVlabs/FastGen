# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import fastgen.utils.tensor as tensor_module
from fastgen.utils.tensor import (
    HIGH_PRECISION,
    as_array,
    assert_tensor,
    dtype_invariant_dropout,
    einsum_without_autocast,
    expand_like,
    high_precision_dtype,
    pixels_to_uint8,
    precision_to_dtype,
    promote_dtype,
    resolve_device,
    temp_disable_efficient_attn,
    to_high_precision,
    to_promoted_dtype,
    unit_pixels_to_uint8,
    waveform_to_pcm16,
)
from tests.utils import RunIf

DROPOUT_DEVICES = ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))]


def test_assert_tensor_accepts_true_scalar():
    assert_tensor(torch.tensor(True), "must be true")


def test_assert_tensor_raises_requested_error_type():
    with pytest.raises(ValueError, match="must be true"):
        assert_tensor(torch.tensor(False), "must be true", error_type=ValueError)


def test_assert_tensor_requires_one_element():
    with pytest.raises(AssertionError, match="requires one element"):
        assert_tensor(torch.tensor([True, True]), "must be true")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
def test_as_array_copies_a_tensor_that_requires_grad_to_the_host_in_the_dtype(device):
    source = torch.tensor([0.1, -2.5], device=device, dtype=torch.bfloat16, requires_grad=True)

    array = as_array(source, torch.float32)

    assert array.dtype == np.float32
    np.testing.assert_array_equal(array, source.detach().cpu().double().numpy())


def test_as_array_fills_a_zero_dimensional_array_from_a_number():
    array = as_array(0.1, torch.float64)

    assert array.shape == () and array.dtype == np.float64 and array == 0.1


def test_as_array_shares_memory_only_with_a_host_tensor_already_in_the_dtype():
    source = torch.arange(3, dtype=torch.float64)

    assert np.shares_memory(as_array(source, torch.float64), source.numpy())
    assert not np.shares_memory(as_array(source, torch.float32), source.numpy())


def test_as_array_rejects_bfloat16_which_numpy_lacks():
    with pytest.raises(TypeError, match="BFloat16"):
        as_array(torch.ones(2), torch.bfloat16)


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32),
        ("float64", torch.float64),
    ],
)
def test_precision_to_dtype(precision, expected):
    assert precision_to_dtype(precision) is expected


def test_precision_to_dtype_uses_default_for_none():
    assert precision_to_dtype(None, default=torch.float64) is torch.float64


def test_precision_to_dtype_returns_a_dtype_unchanged():
    assert precision_to_dtype(torch.bfloat16, default=torch.float64) is torch.bfloat16


def test_precision_to_dtype_rejects_unknown_precision():
    with pytest.raises(KeyError):
        precision_to_dtype("float128")


def test_resolve_device_names_the_current_cuda_device_and_an_unindexed_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    assert resolve_device(None) is None
    assert resolve_device("cuda") == torch.device("cuda", 3)
    assert resolve_device(torch.device("cuda", 1)) == torch.device("cuda", 1)
    assert resolve_device("cpu:0") == resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("meta") == torch.device("meta")


@pytest.mark.parametrize(
    "operands,expected",
    [
        ((torch.float16,), torch.float16),
        ((None, torch.bfloat16, None), torch.bfloat16),
        ((torch.float16, torch.bfloat16), torch.float32),
        ((torch.float32, torch.float16, torch.float64), torch.float64),
        ((torch.bool, torch.int32), torch.int32),
    ],
)
def test_promote_dtype_resolves_dtypes_without_an_implicit_floor(operands, expected):
    assert promote_dtype(*operands) is expected


def test_promote_dtype_accepts_tensors_dtypes_and_none_independently_of_rank():
    narrow = torch.zeros(2, 3, dtype=torch.float16)
    wide_scalar = torch.tensor(1.0, dtype=torch.float64)
    assert promote_dtype(narrow, torch.bfloat16, None) is torch.float32
    assert promote_dtype(narrow, wide_scalar) is torch.float64
    assert promote_dtype(torch.float32, narrow) is torch.float32


@pytest.mark.parametrize("operands", [(), (None,), (None, None)])
def test_promote_dtype_requires_a_dtype_to_resolve(operands):
    with pytest.raises(ValueError, match="at least one tensor or dtype"):
        promote_dtype(*operands)


@pytest.mark.parametrize("cast", [to_promoted_dtype, to_high_precision])
@pytest.mark.parametrize("tensors", [(), (None,), (None, None)])
def test_promotion_casts_preserve_empty_and_missing_inputs(cast, tensors):
    expected = tensors[0] if len(tensors) == 1 else tensors
    assert cast(*tensors) == expected


@pytest.mark.parametrize(
    "source_dtype,min_dtype,expected_dtype",
    [
        (torch.float16, None, torch.float16),
        (torch.bfloat16, torch.float16, torch.float32),
        (torch.float16, torch.float32, torch.float32),
        (torch.float64, torch.float32, torch.float64),
    ],
)
def test_to_promoted_dtype_applies_only_the_requested_floor(source_dtype, min_dtype, expected_dtype):
    source = torch.tensor([1.25, -2.5], dtype=source_dtype)
    actual = to_promoted_dtype(source, min_dtype=min_dtype)
    assert actual.dtype is expected_dtype
    torch.testing.assert_close(actual.double(), source.double(), rtol=0, atol=0)
    if expected_dtype == source_dtype:
        assert actual is source
    assert source.dtype is source_dtype


def test_to_promoted_dtype_preserves_devices_optional_entries_and_existing_storage():
    local = torch.ones(2, dtype=torch.bfloat16)
    remote = torch.empty(2, dtype=torch.float64, device="meta")
    promoted_local, promoted_remote, missing = to_promoted_dtype(local, remote, None)
    assert promoted_local.dtype is promoted_remote.dtype is torch.float64
    assert promoted_local.device == local.device
    assert promoted_remote is remote and missing is None
    assert local.dtype is torch.bfloat16


def test_to_promoted_dtype_preserves_gradients_to_original_operands():
    narrow = torch.tensor([1.25, -2.5], dtype=torch.float16, requires_grad=True)
    wide = torch.tensor([2.0, -3.0], dtype=torch.float64, requires_grad=True)
    promoted_narrow, promoted_wide = to_promoted_dtype(narrow, wide)
    assert promoted_wide is wide
    (promoted_narrow * promoted_wide).sum().backward()
    torch.testing.assert_close(narrow.grad, wide.detach().half(), rtol=0, atol=0)
    torch.testing.assert_close(wide.grad, narrow.detach().double(), rtol=0, atol=0)


def test_to_promoted_dtype_supports_fullgraph_compilation():
    def multiply(x, y):
        x, y, _ = to_promoted_dtype(x, y, None, min_dtype=torch.float32)
        return x * y

    compiled = torch.compile(multiply, backend="eager", fullgraph=True)
    x = torch.tensor([1.25, -2.5], dtype=torch.float16)
    y = torch.tensor([2.0, -3.0], dtype=torch.bfloat16)
    torch.testing.assert_close(compiled(x, y), torch.tensor([2.5, 7.5]), rtol=0, atol=0)


def test_high_precision_dtype_promotes_operands_to_high_precision():
    assert high_precision_dtype() is HIGH_PRECISION
    assert high_precision_dtype(torch.bfloat16, torch.zeros(1, dtype=torch.float16), None) is HIGH_PRECISION


def test_high_precision_dtype_keeps_operands_wider_than_high_precision(monkeypatch):
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)

    assert high_precision_dtype(torch.bfloat16) is torch.float32
    assert high_precision_dtype(torch.zeros(1, dtype=torch.bfloat16), torch.float64) is torch.float64


def test_to_high_precision_casts_one_tensor_and_keeps_one_already_there():
    assert to_high_precision(torch.zeros(2, dtype=torch.bfloat16)).dtype is HIGH_PRECISION

    high = torch.zeros(2, dtype=HIGH_PRECISION)
    assert to_high_precision(high) is high


def test_to_high_precision_casts_tensors_to_one_joint_dtype(monkeypatch):
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)

    low, wide, missing = to_high_precision(
        torch.zeros(2, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.float64), None
    )

    assert low.dtype is wide.dtype is torch.float64
    assert missing is None


@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
def test_einsum_without_autocast_keeps_operand_dtype_inside_autocast(amp_dtype):
    generator = torch.Generator().manual_seed(0)
    weights, values = torch.randn(3, 8, generator=generator), torch.randn(8, 5, 4, generator=generator)
    expected = torch.einsum("bn,n...->b...", weights, values)

    with torch.autocast("cpu", dtype=amp_dtype):
        lowered = torch.einsum("bn,n...->b...", weights, values)
        actual = einsum_without_autocast("bn,n...->b...", weights, values)
        assert torch.is_autocast_enabled("cpu")

    assert lowered.dtype is amp_dtype
    assert actual.dtype is torch.float32
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", DROPOUT_DEVICES)
@pytest.mark.parametrize("memory_format", [torch.contiguous_format, torch.channels_last])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("p", "training"), [(0.5, False), (1.0, False), (0.0, True), (1.0, True)])
def test_dtype_invariant_dropout_is_native_dropout_when_it_keeps_or_drops_everything(
    device, memory_format, dtype, p, training
):
    x = torch.randn(4, 3, 17, 19, device=device).to(dtype).contiguous(memory_format=memory_format)

    out = dtype_invariant_dropout(x, p, training)

    assert torch.equal(out, torch.nn.functional.dropout(x, p, training))
    if not training or p == 0:
        assert out is x


@pytest.mark.parametrize("device", DROPOUT_DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_dtype_invariant_dropout_follows_the_distribution_of_native_dropout(device, dtype, p):
    """Like ``F.dropout``, each entry is kept with probability 1 - p and then scaled by 1 / (1 - p)."""
    torch.manual_seed(0)
    # Nonzero activations, so the zeros of an output are exactly its dropped entries.
    x = (torch.rand(512, 1024, device=device) + 1).to(dtype)
    n = x.numel()

    for out in (dtype_invariant_dropout(x, p), torch.nn.functional.dropout(x, p)):
        keep = out != 0
        # The kept count is binomial: five standard deviations bound it.
        assert abs(keep.sum().item() - n * (1 - p)) < 5 * (n * p * (1 - p)) ** 0.5
        assert out.dtype is dtype
        torch.testing.assert_close(out[keep], x[keep] / (1 - p))


@pytest.mark.parametrize("device", DROPOUT_DEVICES)
def test_dtype_invariant_dropout_draws_one_mask_for_every_dtype(device):
    # Nonzero activations, so the zeros of an output are exactly its dropped entries.
    x = torch.rand(4, 3, 17, 19, device=device) + 1
    masks = []
    for dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
        torch.manual_seed(0)
        masks.append(dtype_invariant_dropout(x.to(dtype), p=0.3) != 0)

    assert all(torch.equal(mask, masks[0]) for mask in masks[1:])
    assert 0.6 < masks[0].float().mean().item() < 0.8


@pytest.mark.parametrize("training", [True, False])
def test_dtype_invariant_dropout_rejects_probabilities_outside_the_unit_interval(training):
    for p in (-0.1, 1.1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            dtype_invariant_dropout(torch.randn(5), p, training)


@pytest.mark.parametrize("device", DROPOUT_DEVICES)
@pytest.mark.parametrize("dropped_gradient", [10000, 40000])
def test_dtype_invariant_dropout_keeps_dropped_gradients_zero_when_scaling_overflows(device, dropped_gradient):
    torch.manual_seed(0)
    x = torch.ones(256, device=device, dtype=torch.float16, requires_grad=True)
    out = dtype_invariant_dropout(x, p=0.4)
    keep = out.detach() != 0
    assert keep.any() and (~keep).any()
    grad = torch.full_like(x, dropped_gradient)
    grad[keep] = 1

    out.backward(grad)

    expected = (grad.double() * keep / 0.6).to(x.dtype)
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, expected)


def test_dtype_invariant_dropout_tangent_reuses_the_primal_mask():
    x, tangent = torch.rand(64) + 1, torch.randn(64)

    torch.manual_seed(0)
    out, out_tangent = torch.func.jvp(lambda x: dtype_invariant_dropout(x, p=0.5), (x,), (tangent,))
    torch.manual_seed(0)
    assert torch.equal(out, dtype_invariant_dropout(x, p=0.5))
    assert torch.equal(out_tangent, tangent * (out != 0) * 2.0)


def test_expand_like_appends_singleton_dimensions():
    source = torch.tensor([1.0, 2.0])
    target = torch.empty(2, 3, 4)

    expanded = expand_like(source, target)

    assert expanded.shape == (2, 1, 1)
    assert torch.equal(expanded[:, 0, 0], source)


def test_temp_disable_efficient_attn_restores_backend_state_on_error(monkeypatch):
    original = {
        "flash": True,
        "mem_efficient": False,
        "cudnn": True,
        "math": False,
    }
    state = original.copy()
    backend = SimpleNamespace(
        flash_sdp_enabled=lambda: state["flash"],
        mem_efficient_sdp_enabled=lambda: state["mem_efficient"],
        cudnn_sdp_enabled=lambda: state["cudnn"],
        math_sdp_enabled=lambda: state["math"],
        enable_flash_sdp=lambda enabled: state.__setitem__("flash", enabled),
        enable_mem_efficient_sdp=lambda enabled: state.__setitem__("mem_efficient", enabled),
        enable_cudnn_sdp=lambda enabled: state.__setitem__("cudnn", enabled),
        enable_math_sdp=lambda enabled: state.__setitem__("math", enabled),
    )
    monkeypatch.setattr(torch.backends, "cuda", backend)

    with pytest.raises(RuntimeError, match="jvp failed"):
        with temp_disable_efficient_attn():
            assert state == {
                "flash": False,
                "mem_efficient": False,
                "cudnn": False,
                "math": True,
            }
            raise RuntimeError("jvp failed")

    assert state == original


def test_pixel_quantization_inverts_uint8_normalization_and_saturates():
    levels = torch.arange(256).to(torch.uint8)
    assert torch.equal(unit_pixels_to_uint8(levels / 255), levels)
    assert torch.equal(pixels_to_uint8(levels / 127.5 - 1), levels)
    assert pixels_to_uint8(torch.tensor([-1.5, -1.0, 0.0, 1.0, 1.5])).tolist() == [0, 0, 128, 255, 255]


def test_pixel_quantization_rounds_low_precision_pixels_at_their_exact_values():
    pixels = torch.linspace(-1, 1, 4097).to(torch.bfloat16)
    assert torch.equal(pixels_to_uint8(pixels), pixels_to_uint8(pixels.double()))
    assert torch.equal(unit_pixels_to_uint8(pixels.abs()), unit_pixels_to_uint8(pixels.abs().double()))


def test_waveform_to_pcm16_inverts_pcm16_normalization_and_saturates():
    samples = torch.arange(-32767, 32768).to(torch.int16)
    assert torch.equal(waveform_to_pcm16(samples / 32767), samples)
    assert waveform_to_pcm16(torch.tensor([-2.0, -0.4 / 32767, 0.6 / 32767, 2.0])).tolist() == [-32767, 0, 1, 32767]
