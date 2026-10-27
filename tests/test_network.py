# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import importlib
import inspect
import itertools
import math
import os
import warnings
from itertools import chain
from types import MethodType, SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import pytest
from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding

from fastgen.networks.network_causal import CausalContext
from fastgen.networks.cosmos3.modules import (
    Cosmos3VLTextRotaryEmbedding,
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)
from fastgen.networks.cosmos3.network import Cosmos3
from fastgen.networks.cosmos_predict2.network import CosmosPredict2, CosmosPredict2TextEncoder
from fastgen.networks.DiT.network import FourierTimeEmbedding
from fastgen.networks.EDM.network import (
    DhariwalUNet,
    EDMPreconditioningMixin,
    FourierEmbedding,
    PositionalEmbedding,
    SongUNet,
)
from fastgen.networks.EDM2.network import MPFourier
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.configs.instantiate import instantiate
from fastgen.configs.net import (
    EDM_CIFAR10_Config,
    EDM_ImageNet64_Config,
    EDM2_IN64_S_Config,
    DiT_IN256_XL_Config,
    SD15Config,
    SDXLConfig,
    FluxConfig,
    Flux2Config,
    Flux2KleinBase4BConfig,
    Flux2KleinBase4BImageEditConfig,
    QwenImageConfig,
    CogVideoXConfig,
    Wan_1_3B_Config,
    CausalWan_1_3B_Config,
    VACE_Wan_1_3B_Config,
    CausalVACE_Wan_1_3B_Config,
    Wan21_I2V_14B_480P_Config,
    Wan22_I2V_5B_Config,
    CausalWan22_I2V_5B_Config,
    CausalWan21_I2V_14B_480P_Config,
    CausalWan21_I2V_14B_720P_Config,
    LTX2Config,
    LTX2I2VConfig,
    Cosmos3_Nano_Config,
)
from fastgen.networks.noise_schedule import (
    AlphasNoiseSchedule,
    CompositeNoiseSchedule,
    EDMNoiseSchedule,
    RFNoiseSchedule,
    StudentGridConfig,
    time_shift,
)
from fastgen.networks.discriminators import Discriminator, Discriminator_ImageDiT
from fastgen.networks.network import FastGenModule, FastGenNetwork
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.utils.basic_utils import clear_gpu_memory
from fastgen.utils.tensor import high_precision_dtype, resolve_device
from fastgen.utils.tree_utils import assert_same, cast_like, get_device, to
from tests.utils import RunIf, network_schedule_kwargs, run_distributed_test
from fastgen.utils.env import hf_download, set_env_vars
from unittest.mock import MagicMock


def _make_causal_ctx(net, cache_tag="pos", store_kv=False, cur_start_frame=0, is_ar=False, total_num_frames=None):
    return CausalContext(
        cache_tag=cache_tag,
        store_kv=store_kv,
        cur_start_frame=cur_start_frame,
        is_ar=is_ar,
        total_num_frames=net.total_num_frames if total_num_frames is None else total_num_frames,
        chunk_size=net.chunk_size,
        cache_engines=dict(net.transformer.kv_cache_engines),
    )


# ---------------------------------------------------------------------------
# Preprocessor placement helpers
# ---------------------------------------------------------------------------
#
# Two modes, selected by the ``--no-offload-preprocessors`` pytest CLI flag
# (see ``tests/conftest.py``):
#   * offload mode (default) — VAE / text-encoder / image-encoder are prepared
#     one at a time on the test device, then kept on CPU; ``_borrow_on_device``
#     moves each module to GPU only for the duration of an encode/decode call
#     and returns it to CPU afterwards. Needed on memory-tight runners that
#     can't hold the transformer **and** all preprocessors on device simultaneously.
#   * ``--no-offload-preprocessors`` — preprocessors are moved to the test
#     device once at startup and stay there.
#
# The module-scoped fixture below configures this toggle only when these tests run.

_OFFLOAD_PREPROCESSORS = True
_STATE_DTYPE = torch.float32


@pytest.fixture(autouse=True, scope="module")
def _configure_offload_preprocessors(pytestconfig):
    """Configure network placement helpers without importing this suite for unrelated tests."""
    global _OFFLOAD_PREPROCESSORS
    previous = _OFFLOAD_PREPROCESSORS
    _OFFLOAD_PREPROCESSORS = not pytestconfig.getoption("--no-offload-preprocessors")
    yield
    _OFFLOAD_PREPROCESSORS = previous


def _set_split_component_precision(module, model_dtype):
    """Configure neural module input precision without moving heavyweight fixtures."""
    for child in module.modules():
        if isinstance(child, FastGenNetwork):
            child._record_precision(precision_model=model_dtype)
        if isinstance(child, Discriminator):
            child._record_precision(precision_model=model_dtype)


def _preprocessor_modules(preprocessor):
    """Return every FastGen precision island in a preprocessor tree."""
    return {name: module for name, module in preprocessor.named_modules() if isinstance(module, FastGenModule)}


def _assert_preprocessor_precision(name, preprocessor, dtype):
    """Check each precision island's owned floating state after placement."""
    mismatches = []
    for module_name, module in _preprocessor_modules(preprocessor).items():
        nested_prefixes = tuple(
            f"{child_name}."
            for child_name, child in module.named_modules()
            if child_name and isinstance(child, FastGenModule)
        )
        expected_dtype = module.precision_opt if module.precision_opt is not None else dtype
        overrides = module._fixed_precision_overrides()
        for tensor_name, tensor in chain(module.named_parameters(), module.named_buffers()):
            if tensor_name.startswith(nested_prefixes):
                continue
            tensor_dtype = overrides.get(tensor_name, expected_dtype)
            if tensor.is_floating_point() and tensor.dtype != tensor_dtype:
                path = ".".join(part for part in (name, module_name, tensor_name) if part)
                mismatches.append(f"{path}={tensor.dtype} (expected {tensor_dtype})")
    assert not mismatches, f"Unexpected preprocessor dtype(s): {', '.join(mismatches[:8])}"


def _assert_tree_dtype_and_device(value, floating_dtype, device, path="output"):
    """Assert device for every tensor leaf and dtype for floating leaves."""
    expected_device = resolve_device(device)
    if torch.is_tensor(value):
        assert value.device == expected_device, f"Expected {path} device {expected_device}, got {value.device}"
        if value.is_floating_point():
            assert value.dtype == floating_dtype, f"Expected {path} dtype {floating_dtype}, got {value.dtype}"
        return value
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_tree_dtype_and_device(child, floating_dtype, expected_device, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_tree_dtype_and_device(child, floating_dtype, expected_device, f"{path}[{index}]")
    return value


def _assert_gpu_tree_dtype_and_device(value, floating_dtype, path="output"):
    """Assert the output boundary used by the GPU preprocessor tests."""
    device = torch.device("cuda", torch.cuda.current_device())
    return _assert_tree_dtype_and_device(value, floating_dtype, device, path)


def _load_preprocessors(teacher, device, dtype):
    """Prepare every preprocessor frozen, offloading each after setup when requested."""
    _set_split_component_precision(teacher, dtype)
    for name, preprocessor in teacher.preprocessors.items():
        preprocessor.prepare_module(
            device=device,
            precision_model=dtype,
            trainable=False,
        )
        _assert_preprocessor_precision(name, preprocessor, preprocessor.precision_model)
        if _OFFLOAD_PREPROCESSORS:
            preprocessor.to(device="cpu")
            clear_gpu_memory()


@contextlib.contextmanager
def _borrow_on_device(module, device):
    """Briefly move a prepared ``module`` to ``device`` for an encode/decode call.

    Return it to CPU on exit. No-op outside offload mode (the module is already
    on device) or when ``module`` is ``None``.
    """
    if not _OFFLOAD_PREPROCESSORS or module is None:
        yield module
        return
    module.to(device=device)
    _assert_preprocessor_precision(type(module).__name__, module, module.precision_model)
    try:
        yield module
    finally:
        module.to(device="cpu")
        clear_gpu_memory()


@contextlib.contextmanager
def _autocast_fp32_weights(device, dtype):
    """Autocast while hiding PyTorch's expected unfused-RMSNorm performance warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Mismatch dtype between input and weight:.*Cannot dispatch to fused implementation.*",
            category=UserWarning,
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            yield


def _decode_smoke(teacher, latents):
    """Decode sampled latents to pixel / waveform space and assert basic
    non-degeneracy.  Exercises the ``sample()`` → ``vae.decode()`` contract
    (scaling, packing/unpacking, device propagation) which the rest of these
    tests don't otherwise cover; the decoded dtype is each VAE's own.

    Uses ``teacher.preprocessors["vae"].decode`` (the wrapper) so per-network scale/shift,
    clamping and LTX-2's multimodal dict-dispatch are all in scope.  In
    offload mode the VAE is moved to the latents' device for the call and
    returned to CPU afterwards.
    """
    device = get_device(latents)
    assert teacher.precision_model is not None
    with _borrow_on_device(teacher.preprocessors["vae"], device), torch.no_grad():
        out = teacher.preprocessors["vae"].decode(latents)

    def _check(t, src):
        assert t is not None, f"vae.decode returned None for {src}"
        assert torch.is_tensor(t), f"vae.decode returned non-tensor for {src}: {type(t)}"
        assert t.device == device, f"vae.decode returned {src} on {t.device}, expected {device}"
        assert torch.isfinite(t).all(), f"vae.decode produced non-finite values for {src}"

    if isinstance(out, dict):
        assert out, "vae.decode returned an empty dict"
        for k, v in out.items():
            _check(v, k)
    else:
        _check(out, "tensor")
    return out


def _assert_class_guidance(teacher, x, t, labels, sample_steps=1):
    """One-hot labels plus the zero null negative through every class-conditional guidance path."""
    guidance_cfg = GuidanceConfig()
    guidance_cfg.cfg.scale = 1.5
    null_labels = torch.zeros_like(labels)
    with torch.no_grad():
        pos = teacher(x, t, condition=labels)
        neg = teacher(x, t, condition=null_labels)
        # The forward maps ``None`` to the same null class the zero vector spells out.
        torch.testing.assert_close(teacher(x, t, condition=None), neg)
        # ``sample()`` concatenates the pair for batched CFG.
        sample = teacher.sample(
            x, condition=labels, neg_condition=null_labels, guidance_cfg=guidance_cfg, sample_steps=sample_steps
        )
        assert sample.shape == x.shape
        # A missing negative under active CFG fails loudly instead of sampling against the wrong thing.
        with pytest.raises(ValueError, match="neg_condition"):
            teacher.sample(
                x, condition=labels, neg_condition=None, guidance_cfg=guidance_cfg, sample_steps=sample_steps
            )
        # Sequential CFG (the training-time teacher path).
        guided = predict_with_guidance(
            teacher, x, t, guidance_cfg=guidance_cfg, condition=labels, neg_condition=null_labels
        )
    torch.testing.assert_close(guided, pos + 0.5 * (pos - neg))


def _smoke_sample(teacher, noise, condition, neg_condition, decode_after=False, **sample_kwargs):
    """One-step ``sample()`` for shape/dtype/device regression coverage of the
    full guidance path.  The caller names the negative: the network's genuine
    null (class nets pass the zero vector) or, when only the plumbing is under
    test, ``condition`` itself — never a silent default that could mask a
    missing negative.  Pass ``sample_steps`` via *sample_kwargs* if a network
    refuses the default of 1.

    When ``decode_after=True``, additionally pipes the sampled latents
    through ``teacher.preprocessors["vae"].decode`` to cover the sample-↔-decode contract
    (see ``_decode_smoke``).  One decode pass per test is sufficient.
    """
    sample_kwargs.setdefault("sample_steps", 1)
    with torch.no_grad():
        out = teacher.sample(
            noise=noise,
            condition=condition,
            neg_condition=neg_condition,
            **sample_kwargs,
        )
    assert_same(shape=True, dtype=True, device=True, noise=noise, out=out)

    if decode_after:
        _decode_smoke(teacher, out)
    return out


def _validate_basic_scheduler_properties(scheduler):
    """Test basic scheduler properties and structure."""
    # Test basic attributes
    assert hasattr(scheduler, "min_t"), "Scheduler should have min_t attribute"
    assert hasattr(scheduler, "max_t"), "Scheduler should have max_t attribute"
    assert hasattr(scheduler, "alpha"), "Scheduler should have alpha method"
    assert hasattr(scheduler, "sigma"), "Scheduler should have sigma method"
    assert hasattr(scheduler, "sample_t"), "Scheduler should have sample_t method"

    # Validate time range
    assert scheduler.min_t < scheduler.max_t, f"min_t ({scheduler.min_t}) should be < max_t ({scheduler.max_t})"
    assert scheduler.min_t >= 0, f"min_t ({scheduler.min_t}) should be non-negative"
    # Allow very large max_t values (some schedules use very large values)
    assert scheduler.max_t <= 10000, f"max_t ({scheduler.max_t}) should be reasonable (<=10000)"


def _validate_time_sampling_and_functions(scheduler):
    """Test time sampling and alpha/sigma function behavior."""
    # Test time sampling and validation
    batch_size = 2
    t = scheduler.sample_t(batch_size, time_dist_type="uniform")
    assert t.shape == (batch_size,), f"Expected shape ({batch_size},), got {t.shape}"

    # Only validate if scheduler has proper time bounds
    assert scheduler.is_t_valid(t), f"Sampled times {t} should be valid"

    # Test alpha and sigma functions with sampled times
    alpha_t = scheduler.alpha(t)
    sigma_t = scheduler.sigma(t)

    assert alpha_t.shape == t.shape, f"alpha(t) shape {alpha_t.shape} should match t shape {t.shape}"
    assert sigma_t.shape == t.shape, f"sigma(t) shape {sigma_t.shape} should match t shape {t.shape}"
    assert torch.all(alpha_t >= 0), f"alpha(t) should be non-negative, got {alpha_t}"
    assert torch.all(sigma_t >= 0), f"sigma(t) should be non-negative, got {sigma_t}"

    return t, alpha_t, sigma_t


def _validate_boundary_values(scheduler, device):
    """Test alpha and sigma at boundary time values."""

    # Test boundary values
    min_t_tensor = torch.tensor([scheduler.min_t]).to(device)
    max_t_tensor = torch.tensor([scheduler.max_t]).to(device)

    alpha_min = scheduler.alpha(min_t_tensor)
    alpha_max = scheduler.alpha(max_t_tensor)
    sigma_min = scheduler.sigma(min_t_tensor)
    sigma_max = scheduler.sigma(max_t_tensor)

    boundary = torch.cat([alpha_min, alpha_max, sigma_min, sigma_max])
    assert torch.isfinite(boundary).all(), f"alpha and sigma at min_t and max_t should be finite, got {boundary}"
    assert (sigma_max > 0).all(), f"sigma(max_t) should be positive, got {sigma_max.item()}"

    return alpha_min, alpha_max, sigma_min, sigma_max


def _validate_edm_schedule(scheduler, t, alpha_t, sigma_t, sigma_min, sigma_max):
    """Validate EDM-specific schedule properties."""
    # EDM: alpha(t) = 1, sigma(t) = t
    assert torch.allclose(alpha_t, torch.ones_like(alpha_t), rtol=1e-4), f"EDM: alpha(t) should be 1, got {alpha_t}"
    assert torch.allclose(sigma_t, t, rtol=1e-4), f"EDM: sigma(t) should equal t, got σ(t)={sigma_t}, t={t}"

    # SNR should be monotonically decreasing (sigma increases)
    assert sigma_min <= sigma_max, (
        f"EDM: sigma should increase with t, got "
        f"σ({scheduler.min_t})={sigma_min.item():.4f} "
        f"> σ({scheduler.max_t})={sigma_max.item():.4f}"
    )

    # sigma(max_t) is max_t itself
    assert torch.allclose(
        sigma_max.float().cpu(), torch.tensor([scheduler.max_t]), rtol=1e-3
    ), f"EDM: sigma(max_t) ({sigma_max.item()}) should equal max_t ({scheduler.max_t})"

    assert scheduler.max_t <= 100.0, f"EDM: max_t should be reasonable (≤100), got {scheduler.max_t}"


def _validate_rf_schedule(scheduler, device, t, alpha_t, sigma_t):
    """Validate Rectified Flow-specific mathematical properties.

    Stated in terms of sigma rather than t, so they hold at any schedule ``shift``: an unshifted
    schedule has ``sigma(t) = t`` and these reduce to the familiar ``alpha = 1 - t`` form, while a
    shifted one (MiniMax-H3's audio sub-schedule at the relative shift 3/12; the video clock's 12
    lives in the sampling grid) puts the same linear interpolation on a warped noise level.
    """
    # 1. Core RF property: α(t) = 1-σ(t), so α(t) + σ(t) = 1
    assert torch.allclose(
        alpha_t + sigma_t, torch.ones_like(alpha_t), rtol=1e-4, atol=1e-5
    ), f"RF: α(t) + σ(t) should equal 1, got α+σ = {(alpha_t + sigma_t).tolist()}"

    # 2. Linear relationships for RF, on this schedule's own noise level
    expected_sigma = time_shift(t, scheduler.shift, max_t=scheduler.max_t)
    assert torch.allclose(
        sigma_t, expected_sigma, rtol=1e-4, atol=1e-5
    ), f"RF: σ(t) should equal f_shift(t) at shift {scheduler.shift}, got σ(t)={sigma_t.tolist()}"
    assert torch.allclose(
        alpha_t, 1.0 - sigma_t, rtol=1e-4, atol=1e-5
    ), f"RF: α(t) should equal 1-σ(t), got α(t)={alpha_t.tolist()}, expected={(1.0 - sigma_t).tolist()}"

    # 3. Boundary conditions (with relaxed tolerance): the warp fixes max_t, so sigma(max_t) = max_t at any shift
    max_t_tensor = torch.tensor([scheduler.max_t], device=device)
    sigma_max = scheduler.sigma(max_t_tensor)
    assert torch.allclose(
        sigma_max, max_t_tensor, rtol=1e-3, atol=1e-4
    ), f"RF: σ(max_t) should equal max_t, got σ(max_t)={sigma_max.item()}, max_t={scheduler.max_t}"

    # 4. Test that at t≈0, α≈1 and σ≈0 (if min_t is close to 0).  The warp fixes t=0, so a shifted
    # schedule is pinned there too, only reaching it along a different curve.
    if scheduler.min_t <= 0.01:
        near_zero_t = torch.tensor([scheduler.min_t]).to(device)
        alpha_zero = scheduler.alpha(near_zero_t)
        sigma_zero = scheduler.sigma(near_zero_t)
        expected_sigma_zero = time_shift(near_zero_t, scheduler.shift, max_t=scheduler.max_t)
        assert torch.allclose(alpha_zero, 1.0 - expected_sigma_zero, rtol=1e-2), (
            f"RF: α(t≈0) should ≈ 1-σ(min_t), got "
            f"α({scheduler.min_t})={alpha_zero.item():.4f}, "
            f"expected {(1.0 - expected_sigma_zero).item():.4f}"
        )
        assert torch.allclose(sigma_zero, expected_sigma_zero, rtol=1e-2), (
            f"RF: σ(t≈0) should ≈ f_shift(min_t), got σ({scheduler.min_t})={sigma_zero.item():.4f}, "
            f"expected {expected_sigma_zero.item():.4f}"
        )


def _validate_ddpm_based_schedule(scheduler, alpha_t, sigma_t, alpha_min, alpha_max, sigma_min, sigma_max):
    """Validate DDPM-based schedule properties (shared by SD, CogVideoX, Alphas)."""
    # DDPM constraint: alpha²(t) + sigma²(t) = 1
    alpha_squared_plus_sigma_squared = alpha_t**2 + sigma_t**2
    expected_ones = torch.ones_like(alpha_squared_plus_sigma_squared)
    assert torch.allclose(
        alpha_squared_plus_sigma_squared, expected_ones, rtol=1e-2, atol=1e-3
    ), f"DDPM: α²(t) + σ²(t) should equal 1, got {alpha_squared_plus_sigma_squared}"

    # Monotonicity checks
    assert alpha_min >= alpha_max, (
        f"DDPM: alpha should decrease with t, "
        f"got α({scheduler.min_t})={alpha_min.item():.4f} < "
        f"α({scheduler.max_t})={alpha_max.item():.4f}"
    )
    assert sigma_min <= sigma_max, (
        f"DDPM: sigma should increase with t, "
        f"got σ({scheduler.min_t})={sigma_min.item():.4f} "
        f"> σ({scheduler.max_t})={sigma_max.item():.4f}"
    )


def validate_noise_scheduler_properties(teacher, device, expected_schedule_type):
    """
    Comprehensive and consistent noise scheduler testing helper.

    Args:
        teacher: The instantiated network model
        device: Device to run tests on
        expected_schedule_type: Expected schedule type string (e.g., "edm", "rf", "sd")
    """
    # Basic sanity checks
    assert hasattr(teacher, "noise_scheduler"), "Model should have noise_scheduler attribute"
    assert teacher.schedule_type == expected_schedule_type

    if isinstance(teacher.noise_scheduler, CompositeNoiseSchedule):
        schedulers = teacher.noise_scheduler.schedules.values()
    else:
        schedulers = [teacher.noise_scheduler]

    for scheduler in schedulers:
        assert scheduler.max_t is not None and scheduler.min_t is not None

        # Step 1: Validate basic properties
        _validate_basic_scheduler_properties(scheduler)

        # Step 2: Test time sampling and alpha/sigma functions
        t, alpha_t, sigma_t = _validate_time_sampling_and_functions(scheduler)

        # Step 3: Test boundary values
        alpha_min, alpha_max, sigma_min, sigma_max = _validate_boundary_values(scheduler, device)

        # Step 4: Schedule-specific validations
        if expected_schedule_type in ["edm"]:
            _validate_edm_schedule(scheduler, t, alpha_t, sigma_t, sigma_min, sigma_max)

        elif expected_schedule_type in ("rf", "shifted_rf"):
            _validate_rf_schedule(scheduler, device, t, alpha_t, sigma_t)

        elif expected_schedule_type in ("sd", "sdxl", "cogvideox", "alphas"):
            _validate_ddpm_based_schedule(scheduler, alpha_t, sigma_t, alpha_min, alpha_max, sigma_min, sigma_max)

        else:
            raise ValueError(f"Unrecognized schedule type: {expected_schedule_type}")


def test_network_edm_cifar10():
    teacher_config = EDM_CIFAR10_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Use valid parameters that exist in the config
    teacher_config = override_config_with_opts(
        teacher_config,
        [
            "-",
            "backbone.img_resolution=2",
            "backbone.model_channels=32",
            "backbone.channel_mult=[1]",
            "backbone.channel_mult_noise=1",
            "backbone.r_timestep=False",
        ],
    )

    student_grid_cfg = StudentGridConfig(steps=1)
    teacher = instantiate(teacher_config, student_grid_cfg=student_grid_cfg)
    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    _set_split_component_precision(teacher, dtype)

    # Test noise scheduler properties
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="edm")
    assert teacher.noise_scheduler.student_grid_cfg is student_grid_cfg

    batch_size = 1
    x = torch.randn(batch_size, 3, 2, 2, device=device, dtype=_STATE_DTYPE)
    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="polynomial").to(device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    # to one-hot
    labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
    output = teacher(x, t, labels)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    assert teacher(x, t, labels, return_features_early=True, feature_indices=set()) == []

    _assert_class_guidance(teacher, x, t, labels)

    # One-step sample() smoke (full guidance path).
    _smoke_sample(teacher, x, condition=labels, neg_condition=torch.zeros_like(labels))


def test_network_edm_imagenet64():
    teacher_config = EDM_ImageNet64_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Use valid parameters that exist in the config
    teacher_config = override_config_with_opts(
        teacher_config,
        [
            "-",
            "backbone.img_resolution=2",
            "backbone.model_channels=32",
            "backbone.channel_mult=[1]",
            "backbone.num_blocks=1",
            "backbone.r_timestep=False",
        ],
    )

    teacher = instantiate(teacher_config)
    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    _set_split_component_precision(teacher, dtype)

    # Test noise scheduler properties
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="edm")

    batch_size = 1
    x = torch.randn(batch_size, 3, 2, 2, device=device, dtype=_STATE_DTYPE)
    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="lognormal").to(device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    # to one-hot
    labels = torch.nn.functional.one_hot(labels, num_classes=1000).float()

    output = teacher(x, t, labels)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    assert teacher(x, t, labels, return_features_early=True, feature_indices=set()) == []

    _assert_class_guidance(teacher, x, t, labels)

    # One-step sample() smoke (full guidance path).
    _smoke_sample(teacher, x, condition=labels, neg_condition=torch.zeros_like(labels))


def test_network_edm2_in64():
    teacher_config = EDM2_IN64_S_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Use valid parameters that exist in the config
    teacher_config = override_config_with_opts(
        teacher_config, ["-", "img_resolution=2", "model_channels=32", "channel_mult=[1]", "num_blocks=1"]
    )

    student_grid_cfg = StudentGridConfig(steps=1)
    teacher = instantiate(teacher_config, student_grid_cfg=student_grid_cfg)
    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    _set_split_component_precision(teacher, dtype)

    # Test noise scheduler properties
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="edm")
    assert teacher.noise_scheduler.student_grid_cfg is student_grid_cfg

    batch_size = 1
    x = torch.randn(batch_size, 3, 2, 2, device=device, dtype=_STATE_DTYPE)
    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    # to one-hot
    labels = torch.nn.functional.one_hot(labels, num_classes=1000).float()
    output = teacher(x, t, labels)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    # An explicitly empty set still requests the feature-bearing return shape.
    output_empty = teacher(x, t, labels, feature_indices=set())
    assert isinstance(output_empty, list) and len(output_empty) == 2
    assert output_empty[0].shape == torch.Size([batch_size, 3, 2, 2])
    assert output_empty[1] == []

    assert teacher(x, t, labels, return_features_early=True, feature_indices=set()) == []

    _assert_class_guidance(teacher, x, t, labels)

    # One-step sample() smoke (full guidance path).
    _smoke_sample(teacher, x, condition=labels, neg_condition=torch.zeros_like(labels))


def _tiny_reference_dit(learn_sigma=True, **kwargs):
    """A one-block DiT with the released checkpoints' 3-channel CFG and random weights (the default init zeros the
    output head)."""
    from fastgen.networks.DiT.network import DiT

    torch.manual_seed(0)
    net = DiT(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=1,
        num_heads=2,
        label_dim=10,
        learn_sigma=learn_sigma,
        cfg_channels=3,
        **kwargs,
    )
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    return net.eval()


@pytest.mark.parametrize(
    "network_name", ["dit", "dit-lora", "cosmos3", "cosmos-predict2", "qwenimage", "wan", "wan-moe", "minimax-h3"]
)
def test_reset_parameters_leaves_synchronized_state_and_rng_untouched(network_name):
    """Rank 0's values replace parameters and persistent buffers after meta materialization, so reset
    hooks only rebuild the remaining state and never draw random values."""

    def shell(cls, **children):
        # Skip constructors that read released configs; the reset hooks only need the modules.
        network = object.__new__(cls)
        torch.nn.Module.__init__(network)
        for name, child in children.items():
            setattr(network, name, child)
        return network

    if network_name == "cosmos3":
        network = _tiny_cosmos3()
    elif network_name == "cosmos-predict2":
        network = CosmosPredict2(
            model_channels=48,
            num_blocks=1,
            num_heads=4,
            max_img_h=4,
            max_img_w=4,
            max_frames=4,
            in_channels=2,
            out_channels=2,
            patch_spatial=1,
            patch_temporal=1,
            enable_logvar_linear=False,
            use_crossattn_projection=False,
        )
    elif network_name == "qwenimage":
        from diffusers.models import QwenImageTransformer2DModel

        from fastgen.networks.QwenImage.network import QwenImage

        transformer = QwenImageTransformer2DModel(
            patch_size=2,
            in_channels=8,
            out_channels=2,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=8,
            axes_dims_rope=(2, 2, 4),
        )
        network = shell(QwenImage, transformer=transformer, noise_scheduler=RFNoiseSchedule())
    elif network_name in ("wan", "wan-moe"):
        from fastgen.networks.Wan.network import Wan
        from fastgen.networks.Wan.wan_moe import WanMoE

        network = shell(Wan, transformer=_tiny_wan_transformer())
        if network_name == "wan-moe":
            low = shell(Wan, transformer=_tiny_wan_transformer())
            network = shell(WanMoE, network_high=network, network_low=low)
    elif network_name == "minimax-h3":
        from diffusers.models.transformers import MiniMaxH3Transformer3DModel

        transformer = MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=6,
            hidden_size=12,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=24,
            in_channels=2,
            audio_in_channels=2,
            patch_size=(1, 1, 1),
            text_dim=8,
            freq_dim=8,
            time_embed_hidden_dim=12,
            time_embed_dim=6,
            rope_freq_dim=1,
        )
        network = _h3_sampling_net(transformer)
    else:
        network = _tiny_reference_dit()
        if network_name == "dit-lora":
            network.lora_base_name = ""
            network.init_lora(rank=2, target_modules=["conditioning_net.1", "projection"])

    # Shift the state off its constructed values, so a hook that rebuilds them in place fails too.
    with torch.no_grad():
        for tensor in network.state_dict().values():
            tensor.copy_(~tensor if tensor.dtype == torch.bool else tensor + 1)
    state = {name: tensor.clone() for name, tensor in network.state_dict().items()}
    rng_state = torch.get_rng_state()

    with torch.no_grad():
        network.reset_parameters()

    assert all(torch.equal(state[name], tensor) for name, tensor in network.state_dict().items())
    assert torch.equal(rng_state, torch.get_rng_state())


def _reference_respaced_chain(alphas_cumprod, steps):
    """The DiT reference's ``create_diffusion(str(steps))`` tables (``space_timesteps`` and ``SpacedDiffusion``)."""
    stride = (len(alphas_cumprod) - 1) / (steps - 1)
    kept, position = set(), 0.0
    for _ in range(steps):
        kept.add(round(position))
        position += stride
    betas, last = [], 1.0
    for timestep, alpha_cumprod in enumerate(alphas_cumprod.tolist()):
        if timestep in kept:
            betas.append(1 - alpha_cumprod / last)
            last = alpha_cumprod
    betas = np.array(betas)
    alphas_cumprod = np.cumprod(1.0 - betas)
    return sorted(kept), betas, alphas_cumprod, np.append(1.0, alphas_cumprod[:-1])


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize(("sample_type", "learn_sigma"), [("ddpm", True), ("ddpm", False), ("ode", True)])
@pytest.mark.parametrize("schedule_type", ["ddpm", "sd"])
def test_dit_ddpm_sampling_matches_the_reference_respaced_chain(sample_type, learn_sigma, schedule_type, device_type):
    """The DiT reference's ``p_sample_loop`` (learned or fixed-large variance) and ``ddim_sample_loop`` on
    ``create_diffusion("25")``: pure noise in, CFG on 3 channels, and a last step that returns the x0 prediction."""
    net = _tiny_reference_dit(learn_sigma=learn_sigma, net_pred_type="eps", schedule_type=schedule_type)
    alphas_cumprod = net.noise_scheduler.alphas_cumprod.numpy()
    net.prepare_module(device=device_type, precision_model=torch.float32)
    device = net.noise_scheduler.alphas_cumprod.device
    assert device.type == device_type
    timesteps, betas, alphas_cumprod, alphas_cumprod_prev = _reference_respaced_chain(alphas_cumprod, 25)
    posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    labels = torch.nn.functional.one_hot(torch.tensor([3, 7]), 10).float().to(device)
    null = torch.zeros_like(labels)
    noise = torch.randn(2, 4, 8, 8).to(device)
    guidance_cfg = GuidanceConfig()
    guidance_cfg.cfg.scale = 1.5

    with torch.no_grad():
        torch.manual_seed(1)
        output = net.sample(
            noise,
            condition=labels,
            neg_condition=null,
            sample_steps=25,
            sample_type=sample_type,
            guidance_cfg=guidance_cfg,
        )

        torch.manual_seed(1)
        x = noise
        for i in reversed(range(len(timesteps))):
            t = torch.full((2,), timesteps[i] / 1000, device=device, dtype=torch.float64)
            cond, uncond = (
                net(x, t, condition=c, fwd_pred_type="eps", return_variance=learn_sigma) for c in (labels, null)
            )
            eps = torch.cat([uncond[:, :3] + 1.5 * (cond[:, :3] - uncond[:, :3]), cond[:, 3:4]], dim=1).double()
            x0 = x.double() / math.sqrt(alphas_cumprod[i]) - math.sqrt(1 / alphas_cumprod[i] - 1) * eps
            if sample_type == "ode":
                x_next = math.sqrt(alphas_cumprod_prev[i]) * x0 + math.sqrt(1 - alphas_cumprod_prev[i]) * eps
            else:
                x_next = (
                    betas[i] * math.sqrt(alphas_cumprod_prev[i]) * x0
                    + (1 - alphas_cumprod_prev[i]) * math.sqrt(1 - betas[i]) * x.double()
                ) / (1 - alphas_cumprod[i])
                step_noise = torch.randn_like(x)
                if i > 0:
                    if learn_sigma:
                        frac = (cond[:, 4:].double() + 1) / 2
                        log_variance = frac * math.log(betas[i]) + (1 - frac) * math.log(posterior_variance[i])
                    else:
                        log_variance = torch.full_like(x_next, math.log(betas[i]))
                    x_next = x_next + torch.exp(0.5 * log_variance) * step_noise.double()
            x = x_next.float()

    assert timesteps[0] == 0 and timesteps[-1] == 999
    # On CUDA, matmuls may run in TF32 (NVIDIA's container default), which magnifies the float32 scheduler's
    # small deviations from this float64 reference.
    tolerance = 1e-4 if device_type == "cpu" else 1e-3
    torch.testing.assert_close(output, x, rtol=tolerance, atol=tolerance)


def test_sit_sde_sampling_follows_the_reference_grid():
    """SiT's ``sample_sde`` defaults: ``sample_steps`` forwards from pure noise at t = 1, Euler-Maruyama over
    ``linspace(1, 0.04, sample_steps)``, and the noise-free "Mean" step from 0.04 to 0."""
    net = _tiny_reference_dit(
        net_pred_type="flow", schedule_type="rf", use_sit_convention=True, scale_t=False, max_t=1.0
    )
    labels = torch.nn.functional.one_hot(torch.tensor([3, 7]), 10).float()
    noise = torch.randn(2, 4, 8, 8)
    inputs, times = [], []

    def capture(module, args, kwargs):
        inputs.append(args[0].clone())
        times.append(args[1][0].item())

    handle = net.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            output = net.sample(
                noise, condition=labels, sample_steps=6, sample_type="sde", guidance_cfg=GuidanceConfig()
            )
    finally:
        handle.remove()

    assert times == pytest.approx(np.linspace(1.0, 0.04, 6).tolist(), abs=1e-12)
    assert torch.equal(inputs[0], noise)
    with torch.no_grad():
        velocity = net(inputs[-1], torch.full((2,), 0.04, dtype=torch.float64), condition=labels, fwd_pred_type="flow")
    torch.testing.assert_close(output, inputs[-1] - ((2.0 - 0.04) * velocity + inputs[-1]) * 0.04)
    with pytest.raises(ValueError, match="at least 2 sample steps"):
        net.sample(noise, condition=labels, sample_steps=1, sample_type="sde", guidance_cfg=GuidanceConfig())


def test_sit_sde_sampling_clamps_its_times_after_the_cast_to_the_time_precision():
    """Float32 rounds ``max_t = 0.999`` up; the SDE sampler's times are built in the time precision and then
    clamped, so its first query is float32's largest time not above 0.999."""
    net = _tiny_reference_dit(
        net_pred_type="flow",
        schedule_type="rf",
        use_sit_convention=True,
        scale_t=False,
        max_t=0.999,
        t_precision="float32",
    )
    labels = torch.nn.functional.one_hot(torch.tensor([3, 7]), 10).float()
    times = []
    handle = net.register_forward_pre_hook(lambda module, args, kwargs: times.append(args[1][0]), with_kwargs=True)
    try:
        with torch.no_grad():
            net.sample(
                torch.randn(2, 4, 8, 8),
                condition=labels,
                sample_steps=3,
                sample_type="sde",
                guidance_cfg=GuidanceConfig(),
            )
    finally:
        handle.remove()

    assert all(t.dtype == torch.float32 for t in times)
    assert times[0] == torch.nextafter(torch.tensor(0.999, dtype=torch.float32), torch.tensor(0.0))
    assert max(t.item() for t in times) <= 0.999


def test_network_dit_in256_xl():
    """Tiny DiT forward and sampling smoke; no preprocessors are initialized, so nothing is downloaded."""
    teacher_config = DiT_IN256_XL_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    # Override input_size to match our test input dimensions
    teacher_config = override_config_with_opts(
        teacher_config, ["-", "input_size=8", "hidden_size=32", "depth=1", "num_heads=1"]
    )

    teacher = instantiate(teacher_config)

    # Ensure the model is on the correct device and dtype
    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    _set_split_component_precision(teacher, dtype)

    batch_size = 1
    # Use input size that matches the overridden config
    x = torch.randn(batch_size, 4, 8, 8, device=device, dtype=_STATE_DTYPE)
    # Use scheduler to sample valid time steps with correct dtype
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform")
    t = t.to(device=device)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    labels = torch.randint(0, 1000, (batch_size,), device=device)
    labels = torch.nn.functional.one_hot(labels, num_classes=1000).to(device=device, dtype=dtype)

    # Test basic forward pass
    output = teacher(x, t, labels)
    assert output.shape == x.shape
    assert output.device == x.device

    # As with EDM, None is the supported unconditional representation. DiT
    # translates it to its learned null-class token internally.
    seen_class_ids = []

    def capture_class_ids(_module, inputs):
        seen_class_ids.append(inputs[0].detach().clone())

    hook = teacher.y_embedder.register_forward_pre_hook(capture_class_ids)
    teacher(x, t, condition=None)
    hook.remove()
    torch.testing.assert_close(
        seen_class_ids[-1],
        torch.full((batch_size,), teacher.label_dim, device=device, dtype=torch.long),
    )

    # Test with return_logvar
    output, logvar = teacher(x, t, labels, return_logvar=True)
    assert output.shape == x.shape
    assert logvar.shape == torch.Size([batch_size, 1])

    # Image-DiT GAN features are spatial maps, not raw token sequences.
    features = teacher(x, t, labels, return_features_early=True, feature_indices={0})
    assert len(features) == 1
    assert features[0].shape == (batch_size, 32, 4, 4)
    discriminator = Discriminator_ImageDiT(feature_indices={0}, num_blocks=1, inner_dim=32).to(
        device=device, dtype=dtype
    )
    logits = discriminator(features)
    assert logits.shape == (batch_size, 1)
    assert torch.isfinite(logits).all()

    # The RF default sampler, SiT's SDE, needs at least two steps.
    _assert_class_guidance(teacher, x, t, labels, sample_steps=2)

    # sample() smoke (full guidance path).
    _smoke_sample(teacher, x, condition=labels, neg_condition=torch.zeros_like(labels), sample_steps=2)


# ---------------------------------------------------------------------------
# EDM / EDM2 / DiT — CPU precision-boundary unit tests
# ---------------------------------------------------------------------------


class _PreconditioningProbe(EDMPreconditioningMixin, FastGenNetwork):
    def __init__(self, drop_precond, precision_model):
        super().__init__(net_pred_type="x0", schedule_type="edm")
        self._record_precision(precision_model=precision_model)
        self.drop_precond = drop_precond
        self.sigma_data = 0.5
        self.noise_scheduler = EDMNoiseSchedule()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("the probe exercises the EDM input and output boundaries only")


def test_edm_feature_taps_select_the_last_block_of_each_resolution():
    """Feature taps follow ``num_blocks``; at the default depth they coincide with the former literal names."""
    from fastgen.networks.EDM.network import DhariwalUNet, SongUNet
    from fastgen.networks.EDM2.network import EDM2UNet

    def song(**kw):
        return SongUNet(
            img_resolution=16,
            in_channels=1,
            out_channels=1,
            model_channels=8,
            channel_mult=[1, 2],
            attn_resolutions=[],
            **kw,
        )

    def dhariwal(**kw):
        return DhariwalUNet(
            img_resolution=16,
            in_channels=1,
            out_channels=1,
            model_channels=8,
            channel_mult=[1, 2],
            attn_resolutions=[],
            **kw,
        )

    def edm2(**kw):
        return EDM2UNet(
            img_resolution=16,
            img_channels=1,
            label_dim=0,
            model_channels=8,
            channel_mult=[1, 2],
            attn_resolutions=[],
            **kw,
        )

    for factory, default_depth, legacy_literal in ((song, 4, "block3"), (dhariwal, 3, "block2"), (edm2, 3, "block2")):
        unet = factory()
        assert unet.num_blocks == default_depth
        tapped = [name for name in unet.enc if name.endswith(f"_block{unet.num_blocks - 1}")]
        assert tapped and tapped == [name for name in unet.enc if legacy_literal in name]
        shallow = factory(num_blocks=2)
        assert [name for name in shallow.enc if name.endswith(f"_block{shallow.num_blocks - 1}")] == [
            "16x16_block1",
            "8x8_block1",
        ]
        assert not [name for name in shallow.enc if legacy_literal in name]


def _make_edm_unet(unet_cls, r_timestep):
    kwargs = dict(
        img_resolution=2,
        in_channels=1,
        out_channels=1,
        label_dim=2,
        augment_dim=3,
        model_channels=8,
        channel_mult=[1],
        channel_mult_emb=1,
        num_blocks=1,
        attn_resolutions=[],
        dropout=0,
        r_timestep=r_timestep,
    )
    return unet_cls(**kwargs)


@pytest.mark.parametrize("unet_cls", [SongUNet, DhariwalUNet])
def test_edm_r_forward_uses_the_optional_embedder_convention(unet_cls):
    r_network = _make_edm_unet(unet_cls, r_timestep=True).eval()
    base_network = _make_edm_unet(unet_cls, r_timestep=False).eval()
    base_state = {
        key: value
        for key, value in r_network.state_dict().items()
        if key != "map_layer0_r.weight" and not key.startswith("r_timestep.")
    }
    base_network.load_state_dict(base_state, strict=True)

    x = torch.randn(2, 1, 2, 2)
    t = torch.rand(2)
    class_labels = torch.randn(2, 2)
    augment_labels = torch.randn(2, 3)

    def run_and_capture_mapping(network, r):
        mapped = []
        handle = network.map_layer1.register_forward_pre_hook(lambda _module, args: mapped.append(args[0]))
        try:
            output = network(
                x,
                t,
                class_labels=class_labels,
                r_noise_labels=r,
                augment_labels=augment_labels,
                feature_indices=None,
            )
        finally:
            handle.remove()
        assert len(mapped) == 1
        return output, mapped[0]

    r_network_output, r_network_mapping = run_and_capture_mapping(r_network, r=None)
    base_output, base_mapping = run_and_capture_mapping(base_network, r=None)
    torch.testing.assert_close(r_network_mapping, base_mapping)
    torch.testing.assert_close(r_network_output, base_output)

    r_network_mapping.sum().backward()
    assert r_network.map_layer0_r.weight.grad is None

    r_network.zero_grad(set_to_none=True)
    _, dual_timestep_mapping = run_and_capture_mapping(r_network, r=t / 2)
    dual_timestep_mapping.sum().backward()
    assert r_network.map_layer0_r.weight.grad is not None
    assert torch.count_nonzero(r_network.map_layer0_r.weight.grad)

    with pytest.raises(ValueError, match="r_noise_labels provided, but r_timestep is not set"):
        run_and_capture_mapping(base_network, r=t / 2)


@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
def test_edm_attention_weights_compute_in_fp32_under_autocast(amp_dtype):
    """EDM's FP32 attention weights, their gradients, and their JVP do not follow an enclosing autocast."""
    from fastgen.networks.EDM.network import AttentionOp

    generator = torch.Generator().manual_seed(0)
    q, k, q_tangent, k_tangent = (torch.randn(2, 16, 12, generator=generator) for _ in range(4))
    weights_grad = torch.randn(2, 12, 12, generator=generator)

    def attention_outputs():
        q_leaf, k_leaf = q.clone().requires_grad_(), k.clone().requires_grad_()
        weights = AttentionOp.apply(q_leaf, k_leaf)
        q_grad, k_grad = torch.autograd.grad(weights, (q_leaf, k_leaf), weights_grad)
        _, weights_jvp = torch.func.jvp(AttentionOp.apply, (q, k), (q_tangent, k_tangent))
        return weights, q_grad, k_grad, weights_jvp

    expected = attention_outputs()
    with torch.autocast("cpu", dtype=amp_dtype):
        actual = attention_outputs()

    for expected_tensor, actual_tensor in zip(expected, actual):
        assert actual_tensor.dtype is torch.float32
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.parametrize("unet_cls", [SongUNet, DhariwalUNet])
def test_edm_low_precision_storage_computes_like_fp32_masters(unet_cls):
    """The image path casts every weight to its input dtype, so with the FP32 mapping path pinned a BF16-stored copy
    computes exactly what FP32 master weights compute in BF16, while an unpinned copy does not."""
    from fastgen.networks.EDM.network import EDMPrecond

    embedding = {"embedding_type": "fourier", "channel_mult_noise": 2} if unet_cls is SongUNet else {}
    torch.manual_seed(0)
    backbone = unet_cls(
        img_resolution=4,
        in_channels=1,
        out_channels=1,
        label_dim=2,
        augment_dim=3,
        model_channels=8,
        channel_mult=[1],
        channel_mult_emb=1,
        num_blocks=1,
        attn_resolutions=[4],
        dropout=0,
        r_timestep=True,
        **embedding,
    )
    with torch.no_grad():
        for parameter in backbone.parameters():
            parameter.normal_(std=0.3)
    masters = EDMPrecond(backbone=backbone)
    pinned, unpinned = copy.deepcopy(masters), copy.deepcopy(masters)
    unpinned._fixed_precision_overrides = lambda: {}
    masters.prepare_module(device="cpu", precision_model=torch.bfloat16, precision_opt=torch.float32)
    for network in (pinned, unpinned):
        network.prepare_module(device="cpu", precision_model=torch.bfloat16)

    overrides = pinned._fixed_precision_overrides()
    assert overrides and set(overrides.values()) == {torch.float32}
    assert all(
        tensor.dtype == overrides.get(name, torch.bfloat16)
        for name, tensor in chain(pinned.named_parameters(), pinned.named_buffers())
        if tensor.is_floating_point() and not name.startswith("noise_scheduler.")
    )

    generator = torch.Generator().manual_seed(1)
    x_t = torch.randn(2, 1, 4, 4, generator=generator)
    t = torch.tensor([0.3, 2.0], dtype=masters.noise_scheduler.t_precision)
    condition = {"orig_condition": torch.eye(2), "aug_condition": torch.randn(2, 3, generator=generator)}
    reference, actual, unpinned_output = (
        network(x_t, t, condition=condition, r=t / 2) for network in (masters, pinned, unpinned)
    )
    assert torch.equal(actual, reference)
    assert not torch.equal(unpinned_output, reference)


@pytest.mark.parametrize(
    ("precision_model", "storage"), [(torch.bfloat16, torch.float32), (torch.float64, torch.float64)]
)
def test_edm2_frozen_copy_stores_all_state_in_at_least_fp32(precision_model, storage):
    """``MPConv`` normalizes every stored weight in at least FP32 before casting it, so EDM2 keeps all of its state in
    at least FP32 even where frozen state otherwise follows the model precision."""
    from fastgen.networks.EDM2.network import EDM2Precond

    network = EDM2Precond(
        img_resolution=4,
        img_channels=1,
        label_dim=2,
        logvar_channels=8,
        model_channels=8,
        channel_mult=[1],
        num_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
    )
    network.prepare_module(device="cpu", precision_model=precision_model)

    assert network.precision_model == precision_model
    assert {
        tensor.dtype
        for name, tensor in chain(network.named_parameters(), network.named_buffers())
        if tensor.is_floating_point() and not name.startswith("noise_scheduler.")
    } == {storage}


def test_prepare_edm_inputs_owns_fp32_preconditioning_and_model_cast():
    probe = _PreconditioningProbe(drop_precond=None, precision_model=torch.bfloat16)
    x_t = torch.arange(8, dtype=torch.float64).reshape(2, 1, 2, 2)
    t = torch.tensor([0.5, 1.0], dtype=torch.float64)
    r = torch.tensor([0.25, 0.75], dtype=torch.float64)

    model_input, model_t, model_r = probe._prepare_edm_inputs(x_t, t, r)
    expected_input, expected_t, expected_r = probe.precond_input(
        x_t.float(),
        t.float(),
        r=r.float(),
    )

    assert model_input.dtype == torch.bfloat16
    assert model_t.dtype == model_r.dtype == torch.float32
    torch.testing.assert_close(model_input, expected_input.to(torch.bfloat16))
    torch.testing.assert_close(model_t, expected_t)
    torch.testing.assert_close(model_r, expected_r)


def test_prepare_edm_inputs_requires_scheduler_timestep_precision():
    probe = _PreconditioningProbe(drop_precond=None, precision_model=torch.bfloat16)
    x_t = torch.zeros(2, 1, 2, 2)
    t = torch.tensor([0.5, 1.0], dtype=torch.float32)

    with pytest.raises(AssertionError, match=r"t\.dtype .*must be torch\.float64"):
        probe._prepare_edm_inputs(x_t, t, None)


@pytest.mark.parametrize("drop_precond", ["input", "both"])
def test_prepare_edm_inputs_preserves_no_cast_contract_when_input_preconditioning_is_dropped(drop_precond):
    probe = _PreconditioningProbe(drop_precond=drop_precond, precision_model=None)
    x_t = torch.randn(2, 1, 2, 2, dtype=torch.float64)
    t = torch.tensor([0.5, 1.0], dtype=torch.float64)
    r = torch.tensor([0.25, 0.75], dtype=torch.float64)

    model_input, model_t, model_r = probe._prepare_edm_inputs(x_t, t, r)

    assert model_input is x_t
    assert model_input.dtype == torch.float64
    torch.testing.assert_close(model_t, t.float())
    torch.testing.assert_close(model_r, r.float())


def test_finalize_edm_prediction_owns_fp32_output_path_and_restores_state_dtype():
    probe = _PreconditioningProbe(drop_precond=None, precision_model=torch.bfloat16)
    x_t = torch.arange(8, dtype=torch.float64).reshape(2, 1, 2, 2)
    prediction = torch.full_like(x_t, 0.25, dtype=torch.bfloat16)
    t = torch.tensor([0.5, 1.0], dtype=torch.float64)

    result = probe._finalize_edm_prediction(
        prediction,
        x_t,
        t,
        sigma_shift=None,
        fwd_pred_type="eps",
    )
    expected = probe.precond_output(prediction.float(), x_t.float(), t.float())
    expected = probe.noise_scheduler.convert_model_output(
        x_t,
        expected,
        t,
        src_pred_type=probe.net_pred_type,
        target_pred_type="eps",
    )

    assert result.dtype == x_t.dtype
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("drop_precond", ["output", "both"])
def test_finalize_edm_prediction_skips_only_output_preconditioning(drop_precond):
    probe = _PreconditioningProbe(drop_precond=drop_precond, precision_model=torch.bfloat16)
    x_t = torch.randn(2, 1, 2, 2, dtype=torch.float64)
    prediction = torch.randn(2, 1, 2, 2, dtype=torch.bfloat16)
    t = torch.tensor([0.5, 1.0], dtype=torch.float64)

    result = probe._finalize_edm_prediction(
        prediction,
        x_t,
        t,
        sigma_shift=None,
        fwd_pred_type="x0",
    )

    assert result.dtype == x_t.dtype
    torch.testing.assert_close(result, prediction.to(x_t.dtype))


def test_finalize_edm_prediction_preserves_state_precision_when_output_preconditioning_is_dropped():
    probe = _PreconditioningProbe(drop_precond="output", precision_model=torch.bfloat16)
    probe.net_pred_type = "flow"
    x_t = torch.tensor([1.00000001], dtype=torch.float64)
    prediction = torch.zeros_like(x_t, dtype=torch.bfloat16)
    t = torch.tensor([0.5], dtype=torch.float64)

    result = probe._finalize_edm_prediction(
        prediction,
        x_t,
        t,
        sigma_shift=None,
        fwd_pred_type="x0",
    )

    assert torch.equal(result, x_t)


def test_finalize_edm_prediction_uses_original_state_for_conversion_after_preconditioning():
    probe = _PreconditioningProbe(drop_precond=None, precision_model=torch.bfloat16)
    x_t = torch.tensor([1.00000001], dtype=torch.float64)
    prediction = torch.zeros_like(x_t, dtype=torch.bfloat16)
    t = torch.tensor([0.5], dtype=torch.float64)

    result = probe._finalize_edm_prediction(
        prediction,
        x_t,
        t,
        sigma_shift=None,
        fwd_pred_type="eps",
    )
    preconditioned = probe.precond_output(prediction.float(), x_t.float(), t.float())
    expected = probe.noise_scheduler.convert_model_output(
        x_t,
        preconditioned,
        t,
        src_pred_type=probe.net_pred_type,
        target_pred_type="eps",
    )
    fp32_state_result = probe.noise_scheduler.convert_model_output(
        x_t.float(),
        preconditioned,
        t,
        src_pred_type=probe.net_pred_type,
        target_pred_type="eps",
        output_dtype=x_t.dtype,
    )

    assert torch.equal(result, expected)
    assert not torch.equal(result, fp32_state_result)


def test_full_precision_reference_projectors_preserve_phase_before_boundary():
    # EDM, EDM2, and DiT intentionally project time in FP32 or better. These
    # values show why those families must not inherit the BF16 time policy used
    # by architectures such as Flux.
    timesteps = torch.tensor([0.5, 0.5005], dtype=torch.float64)
    quantized_timesteps = timesteps.to(torch.bfloat16)
    assert quantized_timesteps[0] == quantized_timesteps[1]

    torch.manual_seed(0)
    projectors = [
        PositionalEmbedding(8),
        FourierEmbedding(8),
        MPFourier(8),
    ]
    for projector in projectors:
        full_precision_phase = projector(timesteps)
        quantized_phase = projector(quantized_timesteps)
        assert not torch.equal(full_precision_phase[0], full_precision_phase[1])
        assert torch.equal(quantized_phase[0], quantized_phase[1])

    dit_embedder = FourierTimeEmbedding(embed_dim=8, frequency_dim=8).float()
    full_precision_phase = dit_embedder.encode_timesteps(timesteps, 8)
    quantized_phase = dit_embedder.encode_timesteps(quantized_timesteps, 8)
    assert not torch.equal(full_precision_phase[0], full_precision_phase[1])
    assert torch.equal(quantized_phase[0], quantized_phase[1])

    projection_inputs = []
    handle = dit_embedder.proj_net[0].register_forward_pre_hook(
        lambda module, args: projection_inputs.append(args[0].dtype)
    )
    try:
        output = dit_embedder(timesteps, proj_input_dtype=torch.float32)
    finally:
        handle.remove()
    assert projection_inputs == [torch.float32]
    assert output.dtype == torch.float32


def test_dit_fourier_projection_uses_model_dtype_with_fp32_master_params():
    embedder = FourierTimeEmbedding(embed_dim=8, frequency_dim=8).float()
    projection_inputs = []
    handle = embedder.proj_net[0].register_forward_pre_hook(
        lambda module, args: projection_inputs.append(args[0].dtype)
    )
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = embedder(torch.tensor([0.5], dtype=torch.float64), proj_input_dtype=torch.bfloat16)
    finally:
        handle.remove()

    assert embedder.proj_net[0].weight.dtype == torch.float32
    assert projection_inputs == [torch.bfloat16]
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("time_cond_type", ["abs", "diff"])
def test_dit_sit_convention_negates_dit_output_for_inverted_timesteps(time_cond_type):
    from fastgen.networks.DiT.network import DiT

    kwargs = dict(
        input_size=2,
        patch_size=1,
        in_channels=1,
        hidden_size=8,
        depth=0,
        num_heads=1,
        label_dim=2,
        scale_t=False,
        r_timestep=True,
        time_cond_type=time_cond_type,
        net_pred_type="flow",
    )
    torch.manual_seed(0)
    dit = DiT(**kwargs)
    sit = DiT(**kwargs, use_sit_convention=True)

    # DiT zero-initializes its output layers; make the shared test weights
    # nonzero so timestep inversion and flow-output negation are observable.
    with torch.no_grad():
        for layer in (dit.final_layer.adaptive_params[-1], dit.final_layer.projection):
            torch.nn.init.normal_(layer.weight, std=0.2)
            torch.nn.init.normal_(layer.bias, std=0.2)
    sit.load_state_dict(dit.state_dict())
    dit.eval()
    sit.eval()

    state = torch.randn(2, 1, 2, 2)
    t_dtype = dit.noise_scheduler.t_precision
    t = torch.tensor([0.2, 0.8], dtype=t_dtype)
    r = torch.tensor([0.7, 0.1], dtype=t_dtype)
    condition = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        dit_output = dit(state, t, condition=condition, r=r, fwd_pred_type="flow")
        sit_output = sit(state, 1 - t, condition=condition, r=1 - r, fwd_pred_type="flow")
        sit_output_without_input_inversion = sit(state, t, condition=condition, r=r, fwd_pred_type="flow")

    assert torch.count_nonzero(dit_output)
    torch.testing.assert_close(sit_output, -dit_output)
    assert not torch.allclose(sit_output_without_input_inversion, sit_output)


def test_dit_unconfigured_forward_casts_positional_operand_to_patch_dtype():
    from fastgen.networks.DiT.network import DiT

    network = DiT(
        input_size=2,
        patch_size=1,
        in_channels=1,
        hidden_size=8,
        depth=0,
        num_heads=1,
        label_dim=2,
        scale_t=False,
        r_timestep=True,
    )
    network.pos_embed = network.pos_embed.to(dtype=torch.float64)
    assert network.precision_model is None

    state = torch.randn(1, 1, 2, 2, dtype=torch.float32)
    t = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
    r = torch.tensor([0.25], dtype=network.noise_scheduler.t_precision)
    condition = torch.zeros(1, dtype=torch.long)
    output = network(
        state,
        t,
        condition=condition,
        r=r,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_output = network(state, t, condition=condition, r=r)

    assert output.dtype == state.dtype
    assert autocast_output.dtype == state.dtype


def test_dit_fsdp_groups_reshard_after_forward(monkeypatch):
    """DiT computes on ``self``, so its groups share no FSDP root; each must reshard after forward explicitly."""
    from fastgen.networks.DiT import network as dit_network
    from fastgen.networks.DiT.network import DiT

    network = DiT(input_size=2, patch_size=1, in_channels=1, hidden_size=8, depth=2, num_heads=1, r_timestep=True)
    shard_calls = []
    monkeypatch.setattr(dit_network, "fully_shard", lambda module, **kwargs: shard_calls.append((module, kwargs)))
    network.fully_shard(mesh="mesh")

    sharded = [module for module, _ in shard_calls]
    assert sharded == [
        *network.blocks,
        network.x_embedder,
        network.t_embedder,
        network.y_embedder,
        network.final_layer,
        network.logvar_linear,
        network.r_embedder,
    ]
    assert all(kwargs == {"mesh": "mesh", "reshard_after_forward": True} for _, kwargs in shard_calls)
    assert {id(p) for module in sharded for p in module.parameters()} == {id(p) for p in network.parameters()}


def test_sd15_sample_final_step_follows_set_alpha_to_one():
    """``set_alpha_to_one=True`` ends on the last x0 prediction; ``False`` renoises it to the SD table's first
    entry, ``alpha(0) x0 + sigma(0) eps``, as the SD1.5 release's scheduler config does."""
    from fastgen.networks.SD15.network import StableDiffusion15
    from fastgen.networks.noise_schedule import get_noise_schedule

    seen = []

    class UNet(torch.nn.Module):
        down_blocks = ()

        def forward(self, x, t, encoder_hidden_states, encoder_attention_mask, **kwargs):
            del t, encoder_hidden_states, encoder_attention_mask, kwargs
            seen.append(x.clone())
            return torch.full_like(x, 0.1)

    network = object.__new__(StableDiffusion15)
    torch.nn.Module.__init__(network)
    network.unet = UNet()
    network.noise_scheduler = get_noise_schedule("sd")
    network.schedule_type = "sd"
    network.net_pred_type = "eps"
    network._record_precision(precision_model=None)

    torch.manual_seed(0)
    noise = torch.randn(1, 4, 2, 2)
    condition = [torch.zeros(1, 3, 8), torch.ones(1, 3, dtype=torch.bool)]
    kwargs = dict(condition=condition, sample_steps=3, guidance_cfg=GuidanceConfig())
    with torch.no_grad():
        exact = network.sample(noise, set_alpha_to_one=True, **kwargs)
        seen.clear()
        residual = network.sample(noise, set_alpha_to_one=False, **kwargs)

    schedule = network.noise_scheduler
    grid = schedule.grid(3)
    eps = torch.full_like(noise, 0.1)
    x0 = schedule.eps_to_x0(seen[-1], eps, grid[-2].expand(1))
    torch.testing.assert_close(exact, x0)
    clean = torch.zeros(1, dtype=grid.dtype)
    expected = schedule.alpha(clean).reshape(-1, 1, 1, 1) * x0 + schedule.sigma(clean).reshape(-1, 1, 1, 1) * eps
    torch.testing.assert_close(residual, expected.to(residual.dtype))
    assert not torch.allclose(residual, exact)


def test_sd_unet_classify_forward_matches_the_diffusers_forward():
    """The feature-tapping U-Net forward shared by SD1.5 and SDXL reproduces ``UNet2DConditionModel.forward``."""
    from diffusers import UNet2DConditionModel

    from fastgen.networks.SD15.network import classify_forward

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=8,
        attention_head_dim=2,
        norm_num_groups=4,
    )
    unet.forward = MethodType(classify_forward, unet)
    state = torch.randn(2, 4, 8, 8)
    timestep = torch.tensor([10.0, 500.0])
    text = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    with torch.no_grad():
        expected = UNet2DConditionModel.forward(unet, state, timestep, text, encoder_attention_mask=mask)[0]
        output = unet(state, timestep, text, encoder_attention_mask=mask)
        sample, features = unet(state, timestep, text, encoder_attention_mask=mask, feature_indices={0, 2})

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(sample, expected, rtol=0, atol=0)
    assert [tuple(feature.shape) for feature in features] == [(2, 8, 4, 4), (2, 16, 4, 4)]


def test_sdxl_micro_conditioning_follows_the_input_resolution():
    """``time_ids`` give the pixel (height, width) as original and target size with a (0, 0) crop, the SDXL
    pipeline's defaults at that resolution."""
    from fastgen.networks.SDXL.network import StableDiffusionXL

    captured = {}

    class UNet(torch.nn.Module):
        down_blocks = ()

        def forward(self, x, t, text_embedding, added_cond_kwargs, **kwargs):
            del t, text_embedding, kwargs
            captured.update(added_cond_kwargs)
            return torch.zeros_like(x)

    network = object.__new__(StableDiffusionXL)
    torch.nn.Module.__init__(network)
    network.unet = UNet()
    network.noise_scheduler = RFNoiseSchedule()
    network.net_pred_type = "eps"
    network._record_precision(precision_model=None)

    state = torch.randn(2, 4, 4, 6)
    condition = [torch.randn(2, 3, 8), torch.randn(2, 8)]
    network(state, torch.full((2,), 0.5, dtype=network.noise_scheduler.t_precision), condition=condition)

    assert torch.equal(captured["time_ids"], torch.tensor([[32.0, 48.0, 0.0, 0.0, 32.0, 48.0]] * 2))


@pytest.mark.parametrize(("num_blocks", "spatial_factor"), [(4, 8), (3, 4)])
def test_sdxl_vae_latent_compression_factors_match_its_downsampling(num_blocks, spatial_factor):
    """Every VAE down block but the last halves the resolution, as in the reference pipeline's ``vae_scale_factor``."""
    from diffusers import AutoencoderKL
    from fastgen.networks.SDXL.network import SDXLImageEncoder

    wrapper = SDXLImageEncoder.__new__(SDXLImageEncoder)
    torch.nn.Module.__init__(wrapper)
    wrapper.vae = AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * num_blocks,
        up_block_types=("UpDecoderBlock2D",) * num_blocks,
        block_out_channels=(4,) * num_blocks,
        norm_num_groups=4,
    )
    with torch.no_grad():
        latent = wrapper.vae.encode(torch.zeros(1, 3, 32, 32)).latent_dist.mode()

    assert wrapper.latent_compression_factors == (32 // latent.shape[-2], 32 // latent.shape[-1])
    assert wrapper.latent_compression_factors == (spatial_factor, spatial_factor)


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_sd15():
    teacher_config = SD15Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # SD15 doesn't support these overrides, so we'll test with default config
    # but use smaller inputs for testing
    teacher = instantiate(teacher_config)
    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="sd")

    batch_size = 1
    x = torch.randn(batch_size, 4, 8, 8, device=device, dtype=_STATE_DTYPE)  # Smaller than original but reasonable
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device)

    captions = ["a caption"]
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        condition = teacher.preprocessors["text_encoder"].encode(captions)
    _assert_gpu_tree_dtype_and_device(
        condition, teacher.preprocessors["text_encoder"].precision_model, "sd15.text_encoder.encode"
    )

    # SD15 text encoder returns (embeddings, attention_mask) tuple
    assert isinstance(condition, tuple) and len(condition) == 2
    embeddings, attention_mask = condition
    assert attention_mask.dtype == torch.long
    embeddings = embeddings.to(device=device, dtype=dtype)
    attention_mask = attention_mask.to(device=device)
    condition = (embeddings, attention_mask)

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    # Forward pass without autocast
    output = teacher(x, t, condition=condition)

    output, logvar = teacher(x, t, condition=condition, return_logvar=True)
    assert output.shape == x.shape
    assert logvar.shape == (batch_size, 1)

    assert teacher(x, t, condition=condition, return_features_early=True, feature_indices=set()) == []

    # One-step sample() smoke (full guidance path; SD15 batched-CFG).
    _smoke_sample(teacher, x, condition=condition, neg_condition=condition, decode_after=True)


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_sdxl():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    teacher = instantiate(SDXLConfig)
    teacher.init_preprocessors()
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="sdxl")

    state = torch.randn(1, 4, 8, 8, device=device, dtype=_STATE_DTYPE)
    timestep = teacher.noise_scheduler.sample_t(1, time_dist_type="uniform").to(device=device)
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        condition = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        condition, teacher.preprocessors["text_encoder"].precision_model, "sdxl.text_encoder.encode"
    )
    condition = to(condition, device=device, dtype=dtype)

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    output, logvar = teacher(state, timestep, condition=condition, return_logvar=True)

    assert output.shape == state.shape and output.dtype == state.dtype
    assert logvar.shape == (state.shape[0], 1) and logvar.dtype == state.dtype
    _smoke_sample(teacher, state, condition=condition, neg_condition=condition, decode_after=True)

    del teacher
    clear_gpu_memory()


@pytest.mark.parametrize(
    "module_path, class_name",
    [
        ("fastgen.networks.SD15.network", "StableDiffusion15"),
        ("fastgen.networks.SDXL.network", "StableDiffusionXL"),
    ],
)
def test_unet_fully_shard_converts_block_level_checkpointing(monkeypatch, module_path, class_name):
    """Diffusers keeps ``gradient_checkpointing`` on the U-Net blocks, never on the root."""
    from diffusers import UNet2DConditionModel
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    network_module = importlib.import_module(module_path)
    network = object.__new__(getattr(network_module, class_name))
    torch.nn.Module.__init__(network)
    network.unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=8,
        attention_head_dim=2,
        norm_num_groups=8,
    )
    network.unet.enable_gradient_checkpointing()
    assert network.unet.is_gradient_checkpointing

    sharded = []
    monkeypatch.setattr(network_module, "fully_shard", lambda module, **kwargs: sharded.append(module))

    network.fully_shard(mesh=object())

    unet = network.unet
    blocks = [*unet.down_blocks, unet.mid_block, *unet.up_blocks]
    assert not unet.is_gradient_checkpointing
    assert all(isinstance(block, CheckpointWrapper) for block in blocks)
    # Blocks are sharded through their wrappers (FSDP outside the checkpoint), then the U-Net root.
    assert [id(module) for module in sharded] == [id(module) for module in [*blocks, unet]]


@RunIf(min_gpus=1, min_gpu_mem=60)
@pytest.mark.large_model
def test_network_flux():
    """Test Flux network for text-to-image generation."""
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    teacher_config = FluxConfig

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Try to instantiate Flux model - skip if not accessible (gated model)
    try:
        teacher = instantiate(teacher_config)
    except Exception as e:
        if "not a valid model identifier" in str(e) or "token" in str(e).lower() or "gated" in str(e).lower():
            pytest.skip(f"Test skipped: Flux model not accessible (requires HuggingFace authentication): {e}")
        raise
    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # Test noise scheduler properties - Flux uses rectified flow
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    # Flux operates on latent space: [B, C, H, W] where C=16 for Flux VAE
    x = torch.randn(batch_size, 16, 8, 8, device=device, dtype=_STATE_DTYPE)
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    captions = ["a caption"]
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        condition = teacher.preprocessors["text_encoder"].encode(captions)
    _assert_gpu_tree_dtype_and_device(
        condition, teacher.preprocessors["text_encoder"].precision_model, "flux.text_encoder.encode"
    )

    guidance_scale = 3.5
    guidance_tensor = torch.full((batch_size,), guidance_scale, device=x.device, dtype=x.dtype)

    # Flux text encoder returns (pooled_prompt_embeds, prompt_embeds) tuple
    assert isinstance(condition, tuple) and len(condition) == 2
    pooled_prompt_embeds, prompt_embeds = condition
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    condition = (pooled_prompt_embeds, prompt_embeds)

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition, guidance=guidance_tensor)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    # Test with return_logvar
    output, logvar = teacher(x, t, condition, guidance=guidance_tensor, return_logvar=True)
    assert output.shape == x.shape
    assert logvar.shape == torch.Size([batch_size, 1])

    assert teacher(x, t, condition, guidance=guidance_tensor, return_features_early=True, feature_indices=set()) == []

    # Test feature extraction with non-empty set (extract from first transformer block)
    output = teacher(x, t, condition, guidance=guidance_tensor, return_features_early=False, feature_indices={0})
    assert isinstance(output, list) and len(output) == 2  # [model_output, features]
    assert output[0].shape == x.shape  # model output shape
    assert isinstance(output[1], list) and len(output[1]) == 1  # one feature extracted

    # Test feature extraction with early return
    features = teacher(x, t, condition, guidance=guidance_tensor, return_features_early=True, feature_indices={0})
    assert isinstance(features, list)
    assert len(features) == 1  # one feature extracted

    # One-step sample() smoke (full guidance path).  Exercise both Flux.1
    # sample-time modes — embedded-guidance distillation (Flux.1-dev) and
    # batched CFG via ``predict_with_guidance(..., batched=True)`` — regardless of
    # which mode the instantiated config picks. The mode is selected by
    # ``self.guidance_scale`` (None -> CFG, else distillation), so we
    # toggle that attribute around the second call.  Decode-after-sample
    # is exercised on the first call only (one decode pass per test is
    # sufficient — see ``_decode_smoke``).
    _smoke_sample(teacher, x, condition=condition, neg_condition=condition, decode_after=True)
    original_guidance_scale = teacher.guidance_scale
    try:
        teacher.guidance_scale = 3.5 if original_guidance_scale is None else None
        _smoke_sample(teacher, x, condition=condition, neg_condition=condition)
    finally:
        teacher.guidance_scale = original_guidance_scale

    # Clear memory after testing
    clear_gpu_memory()


def test_flux_text_encoder_keeps_both_outputs_in_their_native_dtype():
    from fastgen.networks.Flux.network import FluxTextEncoder

    class Tokenizer:
        model_max_length = 4

        def __call__(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(input_ids=torch.ones(1, 4, dtype=torch.long))

    class TextEncoder:
        device = torch.device("cpu")

        def __init__(self, output):
            self.output = output

        def __call__(self, *args, **kwargs):
            del args, kwargs
            return self.output

    pooled_raw = torch.randn(1, 3, dtype=torch.float64)
    sequence_raw = torch.randn(1, 4, 3, dtype=torch.float64)
    encoder = FluxTextEncoder.__new__(FluxTextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.tokenizer = Tokenizer()
    encoder.tokenizer_2 = Tokenizer()
    encoder.text_encoder = TextEncoder(SimpleNamespace(pooler_output=pooled_raw))
    encoder.text_encoder_2 = TextEncoder((sequence_raw,))
    encoder.max_sequence_length = 4

    pooled, sequence = encoder.encode(["prompt"])

    assert pooled.dtype == pooled_raw.dtype == torch.float64
    assert sequence.dtype == sequence_raw.dtype == torch.float64


def _flux2_forward_smoke(teacher, device, dtype, image_edit=False):
    """Shared forward-pass exercise for the Flux.2 family.

    Runs the same set of assertions used by ``test_network_flux``:
    autocast forward, fp-cast forward, ``return_logvar``, and the
    ``feature_indices`` extraction paths. When ``image_edit=True`` the
    condition is bundled as a dict with an extra reference latent.
    """
    batch_size = 1
    # Flux.2 VAE: 32 latent channels, raw pre-pack latent shape.
    # Keep H, W small to fit on memory-tight test runners.
    latent_h, latent_w = 8, 8
    x = torch.randn(batch_size, 32, latent_h, latent_w, device=device, dtype=_STATE_DTYPE)
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    # Run a real prompt through the text encoder so the embedding shape /
    # dtype lines up with whatever variant we actually instantiated.
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        pooled_prompt_embeds, prompt_embeds = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        (pooled_prompt_embeds, prompt_embeds),
        teacher.preprocessors["text_encoder"].precision_model,
        "flux2.text_encoder.encode",
    )
    assert pooled_prompt_embeds is None, "Flux.2 has no pooled embedding; expected None."
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    text_condition = (None, prompt_embeds)
    # The 32B transformer stays on CPU until the text encoder has been offloaded.
    teacher.to(device=device)
    teacher._set_device(device)

    if image_edit:
        ref_latent = torch.randn(batch_size, 32, latent_h, latent_w, device=device, dtype=dtype)
        condition = {"text_embeds": text_condition, "condition_image": ref_latent}
    else:
        condition = text_condition

    # Pass a guidance tensor only for guidance-distilled variants (dev).
    guidance = None
    if teacher.guidance_scale is not None:
        guidance = torch.full((batch_size,), float(teacher.guidance_scale), device=device, dtype=dtype)

    # Smaller variants retain float32 weights for this autocast check. The 32B dev
    # model uses bfloat16 storage to fit on an 80 GB GPU.
    with _autocast_fp32_weights(device, dtype):
        output = teacher(x, t, condition, guidance=guidance)
    assert output.shape == x.shape
    assert output.device == x.device

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    # return_logvar.
    output, logvar = teacher(x, t, condition, guidance=guidance, return_logvar=True)
    assert output.shape == x.shape
    assert logvar.shape == torch.Size([batch_size, 1])

    assert teacher(x, t, condition, guidance=guidance, return_features_early=True, feature_indices=set()) == []

    # feature_indices = {0} (first joint block) without early return ->
    # [model_output, [feature_tensor]].
    output = teacher(x, t, condition, guidance=guidance, return_features_early=False, feature_indices={0})
    assert isinstance(output, list) and len(output) == 2
    assert output[0].shape == x.shape
    assert isinstance(output[1], list) and len(output[1]) == 1

    # Early return collects exactly one feature.
    features = teacher(x, t, condition, guidance=guidance, return_features_early=True, feature_indices={0})
    assert isinstance(features, list) and len(features) == 1

    # One-step sample() smoke (full guidance path).  Exercise both Flux.2
    # sample-time modes — embedded-guidance distillation (Flux.2-dev) and
    # batched CFG via ``predict_with_guidance(..., batched=True)`` + the default
    # ``tmap``-based ``cat_fn`` (klein-base, dict-bundled on the image-edit
    # path) — regardless of which mode this variant's config picks. The
    # mode is selected by ``self.guidance_scale`` (None -> CFG, else
    # distillation), so we toggle that attribute around the second call.
    # Decode-after-sample is exercised on the first call only.
    _smoke_sample(teacher, x, condition=condition, neg_condition=condition, decode_after=True)
    original_guidance_scale = teacher.guidance_scale
    try:
        teacher.guidance_scale = 4.0 if original_guidance_scale is None else None
        _smoke_sample(teacher, x, condition=condition, neg_condition=condition)
    finally:
        teacher.guidance_scale = original_guidance_scale


def _instantiate_flux2_or_skip(teacher_config, variant_label: str):
    """Instantiate a Flux.2 config or pytest.skip on gating / network errors.

    All Flux.2 variants live under the BFL HF org. ``dev`` and
    ``klein-base-9B`` are gated; ``klein-base-4B`` is ungated (Apache-2.0)
    but a 401 there typically still means "no HF token configured".
    """
    try:
        return instantiate(teacher_config)
    except (OSError, RuntimeError) as e:
        msg = str(e).lower()
        if any(s in msg for s in ("gated", "401", "token", "not a valid model identifier", "is not a local folder")):
            pytest.skip(f"Test skipped: Flux.2 {variant_label} not accessible (HF auth / gating): {e}")
        raise


@RunIf(min_gpus=1, min_gpu_mem=80)
@pytest.mark.large_model
@torch.no_grad()
def test_network_flux2():
    """Test FLUX.2-dev (32B) end-to-end forward pass + feature extraction."""
    clear_gpu_memory()
    set_env_vars()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    teacher = _instantiate_flux2_or_skip(Flux2Config, variant_label="dev")
    # Cast before placement: float32 storage alone exceeds an 80 GB GPU.
    teacher.to(dtype=dtype)
    teacher.init_preprocessors()
    _load_preprocessors(teacher, device, dtype)

    _flux2_forward_smoke(teacher, device, dtype, image_edit=False)

    # Flux.2 uses rectified flow.
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    del teacher
    clear_gpu_memory()


@RunIf(min_gpus=1, min_gpu_mem=40)
@pytest.mark.large_model
def test_network_flux2_klein_base_4b():
    """Test FLUX.2-klein-base-4B (Apache-2.0, undistilled).

    Smaller counterpart of ``test_network_flux2`` so the dev test can skip
    on memory-tight runners while we still exercise the same code path on
    one variant of the Flux.2 family.
    """
    clear_gpu_memory()
    set_env_vars()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    teacher = _instantiate_flux2_or_skip(Flux2KleinBase4BConfig, variant_label="klein-base-4B")
    teacher.init_preprocessors()
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # klein-base-4B has `guidance_scale=None` (no embedded guidance vector).
    assert teacher.guidance_scale is None, "klein-base-4B should not use embedded guidance."

    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    _flux2_forward_smoke(teacher, device, dtype, image_edit=False)

    del teacher
    clear_gpu_memory()


def test_flux2_text_encoder_bypasses_auto_tokenizer_config_lookup(monkeypatch):
    """Offline Flux.2-dev snapshots load their declared tokenizer without config.json."""
    import transformers

    from fastgen.networks.Flux2.network import Flux2TextEncoder

    tokenizer = SimpleNamespace(chat_template="{{ messages }}")
    tokenizer_loader = MagicMock(return_value=tokenizer)
    auto_tokenizer_loader = MagicMock(side_effect=AssertionError("AutoTokenizer must not be used"))
    model_loader = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(transformers.LlamaTokenizerFast, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", auto_tokenizer_loader)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", model_loader)
    monkeypatch.setenv("HF_HOME", "/tmp/fastgen-hf-cache")
    monkeypatch.setenv("LOCAL_FILES_ONLY", "true")

    encoder = Flux2TextEncoder("black-forest-labs/FLUX.2-dev")

    assert encoder.processor is tokenizer
    tokenizer_loader.assert_called_once_with(
        "black-forest-labs/FLUX.2-dev",
        subfolder="tokenizer",
        cache_dir="/tmp/fastgen-hf-cache",
        local_files_only=True,
    )
    auto_tokenizer_loader.assert_not_called()
    model_loader.assert_called_once_with(
        "black-forest-labs/FLUX.2-dev",
        subfolder="text_encoder",
        dtype=torch.bfloat16,
        cache_dir="/tmp/fastgen-hf-cache",
        local_files_only=True,
    )


def test_flux2_klein_text_encoder_bypasses_auto_tokenizer_config_lookup(monkeypatch):
    """Offline Klein snapshots load their declared tokenizer without a root config.json."""
    import transformers

    from fastgen.networks.Flux2.network import Flux2KleinTextEncoder

    tokenizer = SimpleNamespace(chat_template="{{ messages }}")
    tokenizer_loader = MagicMock(return_value=tokenizer)
    auto_tokenizer_loader = MagicMock(side_effect=AssertionError("AutoTokenizer must not be used"))
    model_loader = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(transformers.Qwen2Tokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", auto_tokenizer_loader)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_loader)
    monkeypatch.setenv("HF_HOME", "/tmp/fastgen-hf-cache")
    monkeypatch.setenv("LOCAL_FILES_ONLY", "true")

    encoder = Flux2KleinTextEncoder("black-forest-labs/FLUX.2-klein-base-4B")

    assert encoder.tokenizer is tokenizer
    tokenizer_loader.assert_called_once_with(
        "black-forest-labs/FLUX.2-klein-base-4B",
        subfolder="tokenizer",
        cache_dir="/tmp/fastgen-hf-cache",
        local_files_only=True,
    )
    auto_tokenizer_loader.assert_not_called()
    model_loader.assert_called_once_with(
        "black-forest-labs/FLUX.2-klein-base-4B",
        subfolder="text_encoder",
        dtype=torch.bfloat16,
        cache_dir="/tmp/fastgen-hf-cache",
        local_files_only=True,
    )


def test_flux2_klein_text_encoder_rejects_missing_chat_template(monkeypatch):
    import transformers

    from fastgen.networks.Flux2.network import Flux2KleinTextEncoder

    monkeypatch.setattr(
        transformers.Qwen2Tokenizer,
        "from_pretrained",
        MagicMock(return_value=SimpleNamespace(chat_template=None)),
    )
    model_loader = MagicMock()
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_loader)

    with pytest.raises(RuntimeError, match="has no chat_template"):
        Flux2KleinTextEncoder("offline-klein-snapshot")

    model_loader.assert_not_called()


@pytest.mark.parametrize(
    ("model_type", "encoder_name"), [("qwen3", "Flux2KleinTextEncoder"), ("mistral3", "Flux2TextEncoder")]
)
def test_flux2_loads_the_text_encoder_its_checkpoint_ships(monkeypatch, model_type, encoder_name):
    """The checkpoint's ``text_encoder`` config picks the encoder: Qwen3 for the klein variants, Mistral-3 for dev."""
    from fastgen.networks.Flux2 import network as flux2_network

    monkeypatch.setattr(
        flux2_network.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(model_type=model_type)
    )
    for name in ("Flux2TextEncoder", "Flux2KleinTextEncoder"):
        monkeypatch.setattr(
            flux2_network, name, lambda model_id_or_local_path, name=name: (name, model_id_or_local_path)
        )
    network = object.__new__(flux2_network.Flux2)
    torch.nn.Module.__init__(network)
    network.preprocessors = {}
    # A fine-tuned checkpoint whose path names neither variant.
    network.model_id_or_local_path = "/checkpoints/fine-tuned-flux2"

    network.init_text_encoder()
    assert network.preprocessors["text_encoder"] == (encoder_name, "/checkpoints/fine-tuned-flux2")


@RunIf(min_gpus=1, min_gpu_mem=40)
@pytest.mark.large_model
def test_network_flux2_image_edit():
    """Test FLUX.2-klein-base-4B image-edit branch.

    Exercises the ``enable_image_edit=True`` code path that concatenates a
    VAE-encoded reference image with the noisy target along the sequence
    dimension. Uses klein-base-4B so the test stays runnable on a single
    GPU.
    """
    clear_gpu_memory()
    set_env_vars()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    teacher = _instantiate_flux2_or_skip(Flux2KleinBase4BImageEditConfig, variant_label="klein-base-4B image-edit")
    assert teacher.is_image_edit, "Flux2KleinBase4BImageEditConfig should set is_image_edit=True."

    teacher.init_preprocessors()
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    _flux2_forward_smoke(teacher, device, dtype, image_edit=True)

    del teacher
    clear_gpu_memory()


def test_flux2_image_edit_packs_reference_latent_in_target_token_dtype():
    from fastgen.networks.Flux2.network import Flux2

    captured = {}

    class Transformer(torch.nn.Module):
        transformer_blocks = ()
        single_transformer_blocks = ()

        def forward(self, **kwargs):
            captured.update(kwargs)
            return kwargs["hidden_states"]

    network = Flux2.__new__(Flux2)
    torch.nn.Module.__init__(network)
    network.transformer = Transformer()
    network.noise_scheduler = RFNoiseSchedule()
    network.net_pred_type = "flow"
    network.guidance_scale = None
    network.ref_image_t_scale = 10
    network._record_precision(precision_model=None)

    state = torch.randn(1, 2, 4, 4, dtype=torch.bfloat16)
    reference = torch.randn(1, 2, 4, 4, dtype=state.dtype)
    condition = {
        "text_embeds": (None, torch.randn(1, 3, 4, dtype=state.dtype)),
        "condition_image": reference,
    }
    timestep = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
    output = network(
        state,
        timestep,
        condition=condition,
        guidance=torch.ones(1, dtype=torch.float32),
    )

    assert output.dtype == state.dtype
    for name in ("hidden_states", "encoder_hidden_states"):
        assert captured[name].dtype == state.dtype
    # Position ids stay exact (bf16 cannot represent odd integers above 256).
    for name in ("img_ids", "txt_ids"):
        assert captured[name].dtype == torch.float32
    # Guidance is normalized at the embedding site against the actual activation dtype.
    assert captured["guidance"].dtype == torch.float32

    # The reference latent joins the target tokens in the state's dtype. It must arrive in ``precision_model``, which
    # a network without one does not check.
    mixed_dtype = {**condition, "condition_image": reference.to(torch.float32)}
    network(state, timestep, condition=mixed_dtype, guidance=torch.ones(1, dtype=torch.float32))
    assert captured["hidden_states"].dtype == state.dtype
    network._record_precision(precision_model=state.dtype)
    with pytest.raises(AssertionError, match=r"condition_image\.dtype"):
        network(state, timestep, condition=mixed_dtype, guidance=torch.ones(1, dtype=torch.float32))


def _tiny_flux_network(class_name):
    """A Flux.1 or Flux.2 network around a tiny random transformer, without pretrained weights."""
    if class_name == "Flux":
        from diffusers.models import FluxTransformer2DModel

        from fastgen.networks.Flux.network import Flux as network_type, classify_forward

        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=8,
            pooled_projection_dim=8,
            guidance_embeds=True,
            axes_dims_rope=(2, 2, 4),
        )
        transformer.r_embedder = None
    else:
        from diffusers.models import Flux2Transformer2DModel

        from fastgen.networks.Flux2.network import Flux2 as network_type, classify_forward

        transformer = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=8,
            axes_dims_rope=(2, 2, 2, 2),
        )
    transformer.forward = MethodType(classify_forward, transformer)
    network = object.__new__(network_type)
    torch.nn.Module.__init__(network)
    network.transformer = transformer
    network.noise_scheduler = RFNoiseSchedule()
    network.net_pred_type = "flow"
    network.guidance_scale = 3.5
    network.ref_image_t_scale = 10
    network._record_precision(precision_model=None)
    return network


def test_flux_time_embedding_follows_the_state_not_the_pooled_text():
    """Diffusers casts Flux's time and guidance projections to the pooled text's dtype, so the network hands the pooled
    text over in the state's: its dtype then changes nothing, as in a JVP whose autocast promotes the state above the
    text's dtype."""
    torch.manual_seed(0)
    network = _tiny_flux_network("Flux")
    state = torch.randn(1, 2, 4, 8)
    pooled = torch.randn(1, 8).to(torch.bfloat16)
    text = torch.randn(1, 3, 8)
    t = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)

    with torch.no_grad():
        expected = network(state, t, condition=(pooled.float(), text))
        actual = network(state, t, condition=(pooled, text))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("class_name", "image_edit"), [("Flux", False), ("Flux2", False), ("Flux2", True)])
def test_flux_feature_maps_follow_non_square_latent_grids(class_name, image_edit):
    """Feature maps keep a non-square latent's packed (rows, cols) grid and hold only the target image's tokens:
    no text, and on Flux.2's image-edit path no reference-image tokens."""
    torch.manual_seed(0)
    network = _tiny_flux_network(class_name)
    transformer = network.transformer
    block_outputs = []
    for block in (transformer.transformer_blocks[0], transformer.single_transformer_blocks[0]):
        block.register_forward_hook(lambda module, args, output: block_outputs.append(output))

    state = torch.randn(1, 2, 4, 8)  # packed grid 2 x 4
    text = torch.randn(1, 3, 8)
    condition = (torch.randn(1, 8), text) if class_name == "Flux" else (None, text)
    if image_edit:
        condition = {"text_embeds": condition, "condition_image": torch.randn(1, 2, 8, 4)}
    t = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
    with torch.no_grad():
        output, features = network(state, t, condition=condition, feature_indices={0, 1})

    assert output.shape == state.shape
    joint_tokens = block_outputs[0][1]
    # Flux.1's single blocks keep the streams apart; Flux.2's return them joined, text first.
    single_tokens = block_outputs[1][1] if class_name == "Flux" else block_outputs[1][:, text.shape[1] :]
    for feature, tokens in zip(features, (joint_tokens, single_tokens), strict=True):
        assert feature.shape == (1, 16, 2, 4)
        torch.testing.assert_close(feature, tokens[:, :8].permute(0, 2, 1).reshape(1, 16, 2, 4), rtol=0, atol=0)


def test_flux_position_ids_stay_exact_float32_past_bf16_range():
    """BFL's ``prepare`` builds float32 position ids; bf16 would round packed positions above 256."""
    from fastgen.networks.Flux.network import Flux

    captured = {}

    class Transformer(torch.nn.Module):
        transformer_blocks = ()
        single_transformer_blocks = ()
        r_embedder = None

        def forward(self, **kwargs):
            captured.update(kwargs)
            return kwargs["hidden_states"]

    network = object.__new__(Flux)
    torch.nn.Module.__init__(network)
    network.transformer = Transformer()
    network.noise_scheduler = RFNoiseSchedule()
    network.net_pred_type = "flow"
    network.guidance_scale = 3.5
    network._record_precision(precision_model=None)

    state = torch.randn(1, 1, 2, 600, dtype=torch.bfloat16)  # packed grid 1 x 300
    condition = (torch.randn(1, 4, dtype=state.dtype), torch.randn(1, 3, 4, dtype=state.dtype))
    network(state, torch.tensor([0.5], dtype=network.noise_scheduler.t_precision), condition=condition)

    assert captured["img_ids"].dtype == captured["txt_ids"].dtype == torch.float32
    torch.testing.assert_close(captured["img_ids"][:, 2], torch.arange(300, dtype=torch.float32), rtol=0, atol=0)
    assert captured["img_grid"] == (1, 300)


@pytest.mark.parametrize(
    ("module_name", "class_name", "latent_channels", "pooled_dim"),
    [
        ("fastgen.networks.Flux.network", "Flux", 16, 8),
        ("fastgen.networks.Flux2.network", "Flux2", 32, None),
    ],
)
def test_flux_sample_evaluates_on_bfl_shifted_grid(module_name, class_name, latent_channels, pooled_dim, monkeypatch):
    """``sample()`` queries the transformer on Black Forest Labs' grid:
    ``linspace(1, 0, steps + 1)[:-1]`` under the exponential time shift
    ``exp(mu) / (exp(mu) + 1/t - 1)`` (``flux.sampling.get_schedule``,
    ``flux2.sampling.get_schedule``), obtained from the diffusers scheduler's dynamic shifting
    on an explicit ``linspace(1, 1/steps, steps)`` grid -- the same map up to float32 rounding.
    A scheduler built with ``shift=mu`` would instead apply ``mu`` (not ``exp(mu)``) as a
    multiplicative sigma shift, and its default grid ends at ``sigma_min``. The last step goes to
    sigma 0, on clean data.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    network_type = getattr(importlib.import_module(module_name), class_name)
    captured_t = []
    schedulers = []
    set_timesteps = FlowMatchEulerDiscreteScheduler.set_timesteps

    def record_set_timesteps(self, *args, **kwargs):
        schedulers.append(self)
        return set_timesteps(self, *args, **kwargs)

    monkeypatch.setattr(FlowMatchEulerDiscreteScheduler, "set_timesteps", record_set_timesteps)

    class Transformer(torch.nn.Module):
        transformer_blocks = ()
        single_transformer_blocks = ()

        def forward(self, **kwargs):
            captured_t.append(kwargs["timestep"].detach().clone())
            return torch.zeros_like(kwargs["hidden_states"])

    network = object.__new__(network_type)
    torch.nn.Module.__init__(network)
    network.transformer = Transformer()
    network.schedule_type = "rf"
    network.noise_scheduler = RFNoiseSchedule(max_t=1.0)  # as FluxConfig / Flux2Config
    network.net_pred_type = "flow"
    network.guidance_scale = 3.5  # embedded-guidance branch: exactly one forward per step
    network._record_precision(precision_model=None)

    steps = 6
    noise = torch.randn(1, latent_channels, 4, 4)  # 2x2 packing -> 4 image tokens
    pooled = None if pooled_dim is None else torch.randn(1, pooled_dim)
    with torch.no_grad():
        network.sample(noise, condition=(pooled, torch.randn(1, 3, 8)), sample_steps=steps)

    image_seq_len = (4 // 2) * (4 // 2)
    if class_name == "Flux2":
        mu = network._calculate_shift(image_seq_len, num_steps=steps)
    else:
        mu = network._calculate_shift(image_seq_len)
    # BFL ``get_schedule``, verbatim: float32 grid under the exponential time shift.
    grid = torch.linspace(1, 0, steps + 1)
    expected = (math.exp(mu) / (math.exp(mu) + (1 / grid - 1) ** 1.0))[:-1]
    ns = network.noise_scheduler
    expected = ns.safe_clamp(expected.to(ns.t_precision), min=ns.min_t, max=ns.max_t)

    assert len(captured_t) == steps
    got = torch.stack([t.reshape(-1)[0] for t in captured_t])
    assert got.dtype == ns.t_precision
    # Algebraically identical forms; allow a few float32 ulps (the old ``shift=mu`` schedule is off by ~0.2).
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)
    # BFL queries the network at exactly t=1 on pure noise; the [0, 1] schedule must not clamp it.
    assert got[0].item() == 1.0
    [scheduler] = schedulers
    assert scheduler.sigmas[-1].item() == 0.0


@pytest.mark.parametrize(
    ("module_name", "class_name", "pixels", "steps"),
    [
        ("fastgen.networks.Flux.network", "Flux", 4096, 258),
        ("fastgen.networks.Flux2.network", "Flux2", 4096, 258),
        ("fastgen.networks.QwenImage.network", "QwenImage", 8192, 50),
    ],
)
def test_flow_match_sample_scheduler_starts_at_step_zero_on_repeated_leading_timesteps(
    module_name, class_name, pixels, steps
):
    """High shifts round the leading sigmas to 1.0 and repeat the first timestep (Flux and Flux.2 from 258 steps at
    4096x4096, Qwen-Image at its default 50 steps at 8192x8192); the scheduler still starts at the first sigma."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    network = object.__new__(getattr(importlib.import_module(module_name), class_name))
    torch.nn.Module.__init__(network)
    network.noise_scheduler = RFNoiseSchedule(max_t=1.0)  # as FluxConfig / Flux2Config / QwenImageConfig
    if class_name == "QwenImage":
        # Qwen/Qwen-Image scheduler_config.json.
        network.scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=0.9,
            base_image_seq_len=256,
            max_image_seq_len=8192,
            shift_terminal=0.02,
        )
    scheduler, _ = network.sample_scheduler_and_grid(steps, (pixels // 16) ** 2)  # 8x VAE, 2x2 packing
    assert scheduler.timesteps[0] == scheduler.timesteps[1]

    sample = torch.randn(1, 4, 2, 2)
    for index, timestep in enumerate(scheduler.timesteps):
        sample = scheduler.step(torch.zeros_like(sample), timestep, sample, return_dict=False)[0]
        assert scheduler.step_index == index + 1
    assert torch.isfinite(sample).all()


def test_flux_time_guidance_casts_anchor_on_model_input_dtype():
    """BFL creates t/guidance at the unprojected model-input dtype (``img.dtype``). Under bf16
    autocast the projected activations turn bf16, but the time/guidance quantization must keep
    the input dtype (diffusers' own forward casts after ``x_embedder`` instead)."""
    from diffusers.models import FluxTransformer2DModel

    from fastgen.networks.Flux.network import classify_forward

    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        pooled_projection_dim=8,
        guidance_embeds=True,
        axes_dims_rope=(2, 2, 4),
    )
    transformer.forward = MethodType(classify_forward, transformer)
    transformer.r_embedder = None
    seen = {}
    embed = transformer.time_text_embed.forward

    def spy(timestep, guidance, pooled_projection):
        seen.update(timestep=timestep.dtype, guidance=guidance.dtype, pooled=pooled_projection.dtype)
        return embed(timestep, guidance, pooled_projection)

    transformer.time_text_embed.forward = spy

    def run(input_dtype, autocast_dtype):
        seen.clear()
        hidden_states = torch.randn(1, 4, 8, dtype=input_dtype)  # packed model input, 4 tokens
        autocast = torch.autocast("cpu", dtype=autocast_dtype, enabled=autocast_dtype is not None)
        with torch.no_grad(), autocast:
            out = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=torch.randn(1, 3, 8, dtype=input_dtype),
                pooled_projections=torch.randn(1, 8, dtype=input_dtype),
                timestep=torch.tensor([0.5], dtype=torch.float64),
                img_ids=torch.zeros(4, 3),
                txt_ids=torch.zeros(3, 3),
                guidance=torch.full((1,), 3.5, dtype=input_dtype),
                img_grid=(2, 2),
            )
        assert out.shape == hidden_states.shape
        return dict(seen)

    # fp32 input under bf16 autocast: projected activations are bf16, t/guidance stay fp32 like BFL.
    fp32 = {"timestep": torch.float32, "guidance": torch.float32, "pooled": torch.float32}
    assert run(torch.float32, torch.bfloat16) == fp32
    # bf16 input without autocast: the shipped configs; everything bf16.
    transformer.to(torch.bfloat16)
    bf16 = {"timestep": torch.bfloat16, "guidance": torch.bfloat16, "pooled": torch.bfloat16}
    assert run(torch.bfloat16, None) == bf16


def test_reference_parity_net_configs_sample_on_the_full_schedule():
    """The BFL flux/flux2, Qwen-Image and LTX-2 reference samplers query the network at exactly t=1 on
    pure noise, so these net configs run the noise schedule up to ``max_t = 1.0`` (``min_t`` keeps the RF
    default 0.0). Training-time bounds stay explicit in every experiment config via ``sample_t_cfg``."""
    from fastgen.configs import net as net_configs

    names = [
        "FluxConfig",
        "Flux2Config",
        "Flux2KleinBase4BConfig",
        "Flux2KleinBase9BConfig",
        "Flux2ImageEditConfig",
        "Flux2KleinBase4BImageEditConfig",
        "Flux2KleinBase9BImageEditConfig",
        "QwenImageConfig",
        "LTX2Config",
        "LTX2_3_Config",
        "LTX2_5_Config",
        "LTX2I2VConfig",
        "LTX2I2V_3_Config",
        "LTX2I2V_5_Config",
    ]
    for name in names:
        assert network_schedule_kwargs(getattr(net_configs, name))["max_t"] == 1.0, name


@pytest.mark.parametrize("scale_rope", [False, True])
def test_qwenimage_reset_rebuilds_unregistered_rope_after_meta_init(scale_rope):
    from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope

    from fastgen.networks.QwenImage.network import QwenImage
    from fastgen.utils.module import materialize_meta_module

    rope_config = {"theta": 10000, "axes_dim": [4, 6, 8], "scale_rope": scale_rope}
    reference = QwenEmbedRope(**rope_config)
    with torch.device("meta"):
        # Exercise the production reset without loading the full transformer or any weights.
        network = object.__new__(QwenImage)
        FastGenModule.__init__(network)
        network.transformer = torch.nn.Module()
        network.transformer.proj = torch.nn.Linear(2, 2)
        network.transformer.pos_embed = QwenEmbedRope(**rope_config)

    rope = network.transformer.pos_embed
    assert rope.pos_freqs.is_meta and rope.neg_freqs.is_meta
    assert not dict(rope.named_buffers())

    materialize_meta_module(network, "cpu")

    assert network.transformer.pos_embed is rope
    assert network.transformer.proj.weight.device.type == "cpu"

    def check_rope(video_shape, device):
        for name in ("pos_freqs", "neg_freqs"):
            torch.testing.assert_close(getattr(rope, name), getattr(reference, name), rtol=0, atol=0)
        kwargs = {"device": device, "max_txt_seq_len": 5}
        torch.testing.assert_close(rope(video_shape, **kwargs), reference(video_shape, **kwargs), rtol=0, atol=0)

    cpu = torch.device("cpu")
    check_rope((2, 3, 4), cpu)
    # Reset after use. diffusers caches the tables per instance and device, so CPU forwards keep reading
    # the pre-reset copies (identical values); device=None has no cache entry yet and reads the rebuilt ones.
    with torch.no_grad():
        network.reset_parameters()
    check_rope((2, 3, 4), cpu)
    check_rope((1, 4, 3), cpu)
    check_rope((1, 4, 3), None)


def test_qwenimage_text_rope_spans_the_padded_sequence_like_diffusers():
    """Text RoPE covers the padded text sequence, as diffusers' forward does, rather than the mask's longest row,
    so embeddings padded past every prompt run (no host sync) and match the diffusers forward."""
    from diffusers.models import QwenImageTransformer2DModel

    from fastgen.networks.QwenImage.network import QwenImage, classify_forward

    torch.manual_seed(0)
    transformer = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=8,
        out_channels=2,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    transformer.forward = MethodType(classify_forward, transformer)
    transformer.r_embedder = transformer.guidance_embedder = None
    network = object.__new__(QwenImage)
    torch.nn.Module.__init__(network)
    network.transformer = transformer
    network.noise_scheduler = RFNoiseSchedule()
    network.net_pred_type = "flow"
    network.guidance_scale = None
    network._record_precision(precision_model=None)

    state = torch.randn(2, 2, 4, 6)
    text = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])  # padded past every prompt
    t = torch.tensor([0.3, 0.7], dtype=network.noise_scheduler.t_precision)
    with torch.no_grad():
        output = network(state, t, condition=(text, mask))
        expected = QwenImageTransformer2DModel.forward(
            transformer,
            hidden_states=network._pack_latents(state),
            encoder_hidden_states=text,
            encoder_hidden_states_mask=mask,
            timestep=t,
            img_shapes=[[(1, 2, 3)]] * 2,
            return_dict=False,
        )[0]

    torch.testing.assert_close(output, network._unpack_latents(expected, 4, 6), rtol=0, atol=0)


def test_qwenimage_released_weights_require_the_rf_schedule(monkeypatch):
    """The released weights read the raw time t as the noise level, which only ``"rf"`` guarantees; a network
    trained from scratch may use a shifted schedule."""
    from diffusers.models import QwenImageTransformer2DModel

    from fastgen.networks.QwenImage import network as qwen_network
    from fastgen.networks.QwenImage.network import QwenImage

    def load_tiny_config(loader, model_id, **kwargs):
        del model_id, kwargs
        assert loader == QwenImageTransformer2DModel.load_config, "only the config may load"
        return {
            "patch_size": 2,
            "in_channels": 8,
            "out_channels": 2,
            "num_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 2,
            "joint_attention_dim": 8,
            "axes_dims_rope": (2, 2, 4),
        }

    monkeypatch.setattr(qwen_network, "hf_load", load_tiny_config)
    with pytest.raises(ValueError, match="must be 'rf'"):
        QwenImage(schedule_type="shifted_rf", shift=3.0)
    network = QwenImage(schedule_type="shifted_rf", shift=3.0, load_pretrained=False)
    assert network.noise_scheduler.shift == 3.0


def test_qwenimage_sample_evaluates_on_the_pipeline_grid():
    """``QwenImage.sample()`` queries the transformer on the official pipeline's grid (dynamic shifting on
    ``linspace(1, 1/steps, steps)`` with the released scheduler config), first step at exactly t=1, and its last step
    goes to sigma 0, on clean data."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    from fastgen.networks.QwenImage.network import QwenImage

    captured = []

    class Transformer(torch.nn.Module):
        guidance_embedder = torch.nn.Identity()  # selects the embedded-guidance branch: one forward per step
        transformer_blocks = ()

        def forward(self, **kwargs):
            captured.append(kwargs["timestep"].detach().clone())
            return torch.randn_like(kwargs["hidden_states"]) * 0.01

    network = object.__new__(QwenImage)
    torch.nn.Module.__init__(network)
    network.transformer = Transformer()
    network.schedule_type = "rf"
    # Qwen/Qwen-Image scheduler_config.json.
    network.scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=0.9,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        shift_terminal=0.02,
    )
    network.noise_scheduler = RFNoiseSchedule(max_t=1.0)  # as QwenImageConfig
    network.net_pred_type = "flow"
    network.guidance_scale = 4.0
    network._record_precision(precision_model=None)

    steps = 5
    noise = torch.randn(1, 16, 4, 4)
    condition = (torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.long))
    with torch.no_grad():
        network.sample(noise, condition=condition, sample_steps=steps)

    # The pipeline's recipe on a fresh scheduler, independent of ``sample_scheduler_and_grid``: ``calculate_shift``
    # on the 2x2-packed token count, then ``linspace(1, 1/steps, steps)`` under dynamic shifting.
    ns = network.noise_scheduler
    image_seq_len = (noise.shape[-2] // 2) * (noise.shape[-1] // 2)
    mu = 0.5 + (0.9 - 0.5) / (8192 - 256) * (image_seq_len - 256)
    reference = FlowMatchEulerDiscreteScheduler.from_config(network.scheduler.config)
    reference.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), mu=mu)
    expected = reference.sigmas[:steps].to(ns.t_precision)
    got = torch.stack([t.reshape(-1)[0] for t in captured])
    assert len(captured) == steps and got.dtype == ns.t_precision
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert got[0].item() == 1.0
    assert network.scheduler.sigmas[-1].item() == 0.0  # the last ``step`` lands on clean data


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("fastgen.networks.Flux2.network", "Flux2"),
        ("fastgen.networks.cosmos3.network", "Cosmos3"),
        ("fastgen.networks.cosmos_predict2.network", "CosmosPredict2"),
    ],
)
def test_networks_without_secondary_timestep_support_reject_r(module_name, class_name):
    network_type = getattr(importlib.import_module(module_name), class_name)
    network = object.__new__(network_type)
    torch.nn.Module.__init__(network)
    if class_name == "Flux2":
        network.transformer = SimpleNamespace(transformer_blocks=(), single_transformer_blocks=())
    elif class_name == "Cosmos3":
        network.transformer = SimpleNamespace(layers=())
    else:
        network.transformer = SimpleNamespace(blocks=())

    with pytest.raises(NotImplementedError, match=rf"{class_name}\.forward does not support secondary r timesteps"):
        network(torch.zeros(1), torch.zeros(1), r=torch.zeros(1))


@RunIf(min_gpus=1, min_gpu_mem=40)
@pytest.mark.large_model
def test_network_qwen_image():
    """Test QwenImage network for text-to-image generation (~20B param MMDiT)."""
    clear_gpu_memory()

    set_env_vars()
    teacher_config = QwenImageConfig

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    try:
        teacher = instantiate(teacher_config)
    except OSError as e:
        if "not a valid model identifier" in str(e) or "token" in str(e) or "does not appear" in str(e):
            pytest.skip(f"Test skipped: QwenImage model not accessible: {e}")
        raise

    teacher.transformer.transformer_blocks = teacher.transformer.transformer_blocks[:1]

    teacher.init_preprocessors()
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    x = torch.randn(batch_size, 16, 8, 8, device=device, dtype=_STATE_DTYPE)
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    captions = ["a caption"]
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        condition = teacher.preprocessors["text_encoder"].encode(captions)
    _assert_gpu_tree_dtype_and_device(
        condition, teacher.preprocessors["text_encoder"].precision_model, "qwen_image.text_encoder.encode"
    )

    assert isinstance(condition, tuple) and len(condition) == 2
    prompt_embeds, prompt_embeds_mask = condition
    assert prompt_embeds_mask.dtype == torch.long
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    prompt_embeds_mask = prompt_embeds_mask.to(device=device)
    condition = (prompt_embeds, prompt_embeds_mask)

    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition)

    assert output.shape == x.shape
    assert output.device == x.device

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    output, logvar = teacher(x, t, condition, return_logvar=True)
    assert output.shape == x.shape
    assert logvar.shape == torch.Size([batch_size, 1])

    assert teacher(x, t, condition, return_features_early=True, feature_indices=set()) == []

    output = teacher(x, t, condition, return_features_early=False, feature_indices={0})
    assert isinstance(output, list) and len(output) == 2
    assert output[0].shape == x.shape
    assert isinstance(output[1], list) and len(output[1]) == 1

    features = teacher(x, t, condition, return_features_early=True, feature_indices={0})
    assert isinstance(features, list)
    assert len(features) == 1
    assert features[0].shape == (batch_size, 3072, 4, 4)

    # Two-step sample() smoke (full guidance path; QwenImage applies a custom
    # norm-preserving rescale via guidance_cfg.rescale_type="norm" inside
    # predict_with_guidance).  ``sample_steps=1`` would feed a single-element
    # ``sigmas`` array to the upstream ``FlowMatchEulerDiscreteScheduler``,
    # whose dynamic-shift stretch divides by zero and yields NaN timesteps
    # (``stretched_t = 1 - (one_minus_z / scale_factor)``).
    _smoke_sample(teacher, x, condition=condition, neg_condition=condition, sample_steps=2, decode_after=True)

    del teacher
    clear_gpu_memory()


class _StubCogVideoXTransformer(torch.nn.Module):
    """Records the rotary tables the network passes and returns zeros of the input's shape."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_rotary_positional_embeddings=False)
        self.transformer_blocks = torch.nn.ModuleList()
        self.rotary = []

    def forward(self, x, t, condition, image_rotary_emb=None, **kwargs):
        self.rotary.append(image_rotary_emb)
        return torch.zeros_like(x)

    def enable_gradient_checkpointing(self):
        pass


def _cogvideox_scheduler_config(snr_shift_scale):
    """The released ``CogVideoXDPMScheduler`` config, in which the CogVideoX checkpoints differ only by SNR shift."""
    return {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "trained_betas": None,
        "clip_sample": False,
        "set_alpha_to_one": True,
        "steps_offset": 0,
        "prediction_type": "v_prediction",
        "clip_sample_range": 1.0,
        "sample_max_value": 1.0,
        "timestep_spacing": "trailing",
        "rescale_betas_zero_snr": True,
        "snr_shift_scale": snr_shift_scale,
    }


def _assert_released_cogvideox_schedule(schedule, snr_shift_scale):
    """The schedule holds ``CogVideoXDPMScheduler``'s table under the released config with this SNR shift, one entry
    per training timestep, so its ``num_train_timesteps`` is the config's."""
    from diffusers import CogVideoXDPMScheduler

    config = _cogvideox_scheduler_config(snr_shift_scale)
    table = CogVideoXDPMScheduler.from_config(config).alphas_cumprod
    assert schedule.alphas_cumprod.numel() == schedule.num_train_timesteps == config["num_train_timesteps"]
    assert torch.equal(schedule.alphas_cumprod, table.to(schedule.alphas_cumprod.dtype))


def test_cogvideox_schedule_and_rotary_follow_the_checkpoint_not_its_path(monkeypatch):
    """The SNR shift comes from the released configs or an explicit value, rotary tables from the transformer."""
    from fastgen.networks.CogVideoX.network import CogVideoX

    monkeypatch.setattr(
        CogVideoX, "_initialize_network", lambda self, *args: setattr(self, "transformer", _StubCogVideoXTransformer())
    )
    monkeypatch.setattr(CogVideoX, "_prepare_rotary_positional_embeddings", lambda self, *args, **kwargs: "rotary")
    for model_id, snr_shift_scale in ((CogVideoX.MODEL_PATH_2B, 3.0), (CogVideoX.MODEL_PATH_5B, 1.0)):
        _assert_released_cogvideox_schedule(CogVideoX(model_id).noise_scheduler, snr_shift_scale)
    # A local copy has no released ID to look up, so its shift must be given.
    with pytest.raises(ValueError, match="snr_shift_scale"):
        CogVideoX("/checkpoints/CogVideoX-5b")
    network = CogVideoX("/checkpoints/CogVideoX-5b", snr_shift_scale=1.0)
    _assert_released_cogvideox_schedule(network.noise_scheduler, 1.0)

    del network.transformer.forward  # drop the feature-extraction forward installed at construction
    x_t, t, condition = torch.zeros(1, 2, 1, 2, 2), torch.full((1,), 0.5, dtype=torch.float64), torch.zeros(1, 3, 4)
    for use_rotary in (False, True):
        network.transformer.config.use_rotary_positional_embeddings = use_rotary
        network(x_t, t, condition)
    assert network.transformer.rotary == [None, "rotary"]


def test_cogvideox_to_places_the_noise_schedule(monkeypatch):
    """A plain ``to`` places every registered child, including the schedule's non-persistent table."""
    from fastgen.networks.CogVideoX.network import CogVideoX

    monkeypatch.setattr(
        CogVideoX, "_initialize_network", lambda self, *args: setattr(self, "transformer", _StubCogVideoXTransformer())
    )
    network = CogVideoX(CogVideoX.MODEL_PATH_2B)
    network.transformer.proj = torch.nn.Linear(2, 2)

    assert network.to("meta") is network
    assert network.transformer.proj.weight.is_meta
    assert network.noise_scheduler.alphas_cumprod.is_meta


class _StubCogVideoXVelocity(_StubCogVideoXTransformer):
    """Predicts a velocity that depends on the state and the time."""

    def forward(self, x, t, condition, image_rotary_emb=None, **kwargs):
        return 0.5 * torch.tanh(x) * (t / 1000).to(x.dtype).view(-1, 1, 1, 1, 1) + 0.1 * x


def _stub_cogvideox(monkeypatch, transformer_type=_StubCogVideoXTransformer):
    """A CogVideoX-2B network around a stub transformer, which its forward calls directly."""
    from fastgen.networks.CogVideoX.network import CogVideoX

    monkeypatch.setattr(
        CogVideoX, "_initialize_network", lambda self, *args: setattr(self, "transformer", transformer_type())
    )
    network = CogVideoX(CogVideoX.MODEL_PATH_2B)
    del network.transformer.forward  # drop the feature-extraction forward installed at construction
    return network


def _cogvideox_query_times(network, sample_steps, **sample_kwargs):
    """The times at which ``sample`` queries the network, and the sample."""
    times = []
    handle = network.register_forward_pre_hook(
        lambda module, args, kwargs: times.append(args[1][0].item()), with_kwargs=True
    )
    try:
        with torch.no_grad():
            sample = network.sample(
                torch.randn(1, 2, 1, 2, 2),
                condition=torch.zeros(1, 3, 4),
                sample_steps=sample_steps,
                guidance_cfg=GuidanceConfig(),
                **sample_kwargs,
            )
    finally:
        handle.remove()
    return times, sample


@pytest.mark.parametrize("sample_steps", [1, 7, 50, 1001])
def test_cogvideox_sampler_queries_the_schedule_grid_by_default(monkeypatch, sample_steps):
    """By default the network is queried on the noise schedule's uniform ``grid`` before its clean end, at any step
    count."""
    network = _stub_cogvideox(monkeypatch)
    times, sample = _cogvideox_query_times(network, sample_steps)

    assert times == network.noise_scheduler.grid(sample_steps)[:-1].tolist()
    assert torch.isfinite(sample).all()


@pytest.mark.parametrize("sample_steps", [3, 6, 7, 50, 61])
def test_cogvideox_sampler_queries_the_diffusers_trailing_timesteps(monkeypatch, sample_steps):
    """With ``timestep_spacing="trailing"`` the network is queried at ``CogVideoXDPMScheduler``'s "trailing" spacing
    before its rounding to integer timesteps: exactly its timesteps when the step count divides 1000, and within half
    a table step otherwise (where diffusers' float ``arange`` appends a spurious near-zero entry for some, e.g. 61)."""
    from diffusers import CogVideoXDPMScheduler

    network = _stub_cogvideox(monkeypatch)
    reference = CogVideoXDPMScheduler.from_config(_cogvideox_scheduler_config(snr_shift_scale=3.0))
    reference.set_timesteps(sample_steps)
    times, sample = _cogvideox_query_times(network, sample_steps, timestep_spacing="trailing")

    indices = torch.tensor(times, dtype=torch.float64) * 1000
    timesteps = reference.timesteps[:sample_steps].to(torch.float64)
    if 1000 % sample_steps == 0:
        assert torch.equal(indices, timesteps)
    else:
        assert (indices - timesteps).abs().max() <= 0.5
    assert torch.isfinite(sample).all()


def test_cogvideox_trailing_sampler_rejects_more_steps_than_timesteps(monkeypatch):
    """Above 1000 steps the trailing spacing would leave the schedule."""
    network = _stub_cogvideox(monkeypatch)
    with pytest.raises(ValueError, match="sample_steps"):
        _cogvideox_query_times(network, 1001, timestep_spacing="trailing")


@pytest.mark.parametrize("sample_steps", [10, 50])
def test_cogvideox_trailing_sampler_takes_the_diffusers_scheduler_steps(monkeypatch, sample_steps):
    """When the step count divides 1000, the trailing sampler takes ``CogVideoXDPMScheduler.step``'s steps given the
    same injected noise, which it draws once per step."""
    from diffusers import CogVideoXDPMScheduler
    from diffusers.schedulers import scheduling_dpm_cogvideox

    network = _stub_cogvideox(monkeypatch, _StubCogVideoXVelocity)
    noise, condition = torch.randn(2, 4, 2, 4, 4), torch.zeros(2, 3, 4)
    draws = []
    randn_like = torch.randn_like

    def recorded_randn_like(value):
        draws.append(randn_like(value))
        return draws[-1]

    monkeypatch.setattr(torch, "randn_like", recorded_randn_like)
    with torch.no_grad():
        sample = network.sample(
            noise,
            condition=condition,
            sample_steps=sample_steps,
            timestep_spacing="trailing",
            guidance_cfg=GuidanceConfig(),
        )
    assert len(draws) == sample_steps

    scheduler = CogVideoXDPMScheduler.from_config(_cogvideox_scheduler_config(snr_shift_scale=3.0))
    scheduler.set_timesteps(sample_steps)
    latents, x0 = noise, None
    with torch.no_grad():
        for i, timestep in enumerate(scheduler.timesteps):
            # diffusers' second-order steps draw twice and inject the second draw.
            monkeypatch.setattr(scheduling_dpm_cogvideox, "randn_tensor", lambda *args, draw=draws[i], **kwargs: draw)
            t = torch.full((latents.shape[0],), float(timestep), dtype=torch.float64)
            velocity = network.transformer(latents.transpose(1, 2), t, condition).transpose(1, 2)
            back = scheduler.timesteps[i - 1] if i > 0 else None
            latents, x0 = scheduler.step(velocity, x0, timestep, back, latents, return_dict=False)
    torch.testing.assert_close(sample, latents, rtol=1e-5, atol=1e-5)


def test_cogvideox_rejects_temporally_patched_checkpoints(tmp_path):
    """CogVideoX 1.5 checkpoints patch time (``patch_size_t``), which the network does not support."""
    import json

    from fastgen.networks.CogVideoX.network import CogVideoX

    tiny_config = {
        "_class_name": "CogVideoXTransformer3DModel",
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "in_channels": 4,
        "out_channels": 4,
        "time_embed_dim": 16,
        "text_embed_dim": 16,
        "num_layers": 1,
        "sample_width": 8,
        "sample_height": 8,
        "sample_frames": 9,
        "patch_size": 2,
        "patch_size_t": 2,
        "max_text_seq_length": 4,
        "use_rotary_positional_embeddings": True,
    }
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(tiny_config))

    with pytest.raises(NotImplementedError, match="patch_size_t"):
        CogVideoX(str(tmp_path), load_pretrained=False, snr_shift_scale=1.0)


@pytest.mark.parametrize("sdpa_no_cudnn", [True, False])
def test_cogvideox_attention_runs_off_cudnn_sdpa_within_the_enclosing_backends(tmp_path, monkeypatch, sdpa_no_cudnn):
    """cuDNN's reduced-precision SDPA backward is non-finite for CogVideoX-2B, so ``sdpa_no_cudnn`` disables cuDNN for
    every block's attention while the other backends keep their enclosing state, such as a math-only JVP region; a
    deep copy's attention stays bound to the copy. Without it, attention keeps every enclosing backend."""
    import copy
    import json

    from torch.nn.attention import SDPBackend, sdpa_kernel

    from fastgen.networks.CogVideoX.network import CogVideoX
    from fastgen.utils.tensor import temp_disable_efficient_attn

    tiny_config = {
        "_class_name": "CogVideoXTransformer3DModel",
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "in_channels": 4,
        "out_channels": 4,
        "time_embed_dim": 16,
        "text_embed_dim": 16,
        "num_layers": 2,
        "sample_width": 8,
        "sample_height": 8,
        "sample_frames": 9,
        "patch_size": 2,
        "max_text_seq_length": 4,
    }
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(tiny_config))
    network = CogVideoX(
        str(tmp_path), load_pretrained=False, snr_shift_scale=1.0, disable_grad_ckpt=True, sdpa_no_cudnn=sdpa_no_cudnn
    )
    network.prepare_module("cpu", precision_model=torch.float32)

    backends = torch.backends.cuda
    seen = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def recording_sdpa(*args, **kwargs):
        seen.append((backends.cudnn_sdp_enabled(), backends.flash_sdp_enabled(), backends.math_sdp_enabled()))
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording_sdpa)
    x_t = torch.randn(1, 4, 3, 8, 8)
    t = network.noise_scheduler.sample_t(1, time_dist_type="uniform")
    condition = torch.randn(1, 4, 16)

    all_backends = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    with sdpa_kernel(all_backends):
        network(x_t, t, condition=condition)
        with temp_disable_efficient_attn("cuda"):
            network(x_t, t, condition=condition)
        assert backends.cudnn_sdp_enabled()

    assert seen == [(not sdpa_no_cudnn, True, True)] * 2 + [(False, False, True)] * 2
    copied = copy.deepcopy(network).transformer.transformer_blocks[0].attn1
    assert ("forward" in vars(copied)) is sdpa_no_cudnn
    assert copied.forward.__self__ is copied


def test_only_the_cogvideox_2b_config_runs_attention_off_cudnn():
    """cuDNN's reduced-precision SDPA backward is non-finite for CogVideoX-2B; 5B trains finitely on faster cuDNN."""
    from fastgen.configs.net import CogVideoX5BConfig, CogVideoXConfig

    assert CogVideoXConfig.sdpa_no_cudnn is True
    assert CogVideoX5BConfig.get("sdpa_no_cudnn", False) is False


@pytest.mark.parametrize("model_id", ["THUDM/CogVideoX-2b", "THUDM/CogVideoX-5b"])
def test_cogvideox_text_encoder_tokenizes_like_the_reference_pipeline(monkeypatch, model_id):
    """The released tokenizers' ``model_max_length`` is the 226 tokens to which ``CogVideoXPipeline`` explicitly pads
    and truncates, so empty, short, and overlong prompts reach T5 as the reference's token IDs."""
    from diffusers import CogVideoXPipeline

    from fastgen.networks.CogVideoX import network as cogvideox_network

    class StubT5(torch.nn.Module):
        def forward(self, input_ids):
            # Each token ID embeds as itself, so equal embeddings are equal token IDs.
            return (input_ids.unsqueeze(-1).float(),)

    core = StubT5()
    monkeypatch.setattr(cogvideox_network.T5EncoderModel, "from_pretrained", MagicMock(return_value=core))
    encoder = cogvideox_network.CogVideoXTextEncoder(model_id).prepare_module(device="cpu")
    pipeline = object.__new__(CogVideoXPipeline)
    pipeline.tokenizer, pipeline.text_encoder = encoder.tokenizer, core

    for prompt in ("", "a cat", "a cat " * 300):
        reference, _ = pipeline.encode_prompt(
            [prompt], do_classifier_free_guidance=False, device=torch.device("cpu"), dtype=torch.float32
        )
        embeddings = encoder.encode([prompt])
        assert embeddings.shape == (1, 226, 1)
        assert torch.equal(embeddings, reference)


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_cogvideox():
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    set_env_vars()
    teacher_config = CogVideoXConfig

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # CogVideoX doesn't support model_channels override, use default config but test with smaller inputs
    teacher = instantiate(teacher_config)

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="cogvideox")

    batch_size = 1
    C, T, H, W = 16, 2, 4, 4  # Reduced from 16, 4, 8, 16

    # B, C, T, H, W
    x = torch.randn(batch_size, C, T, H, W, device=device, dtype=_STATE_DTYPE)

    # Use scheduler to sample valid time steps (CogVideoX uses integer timesteps)
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    captions = ["a caption"]
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        condition = teacher.preprocessors["text_encoder"].encode(captions)
    _assert_gpu_tree_dtype_and_device(
        condition, teacher.preprocessors["text_encoder"].precision_model, "cogvideox.text_encoder.encode"
    )
    condition = condition.to(device=device, dtype=dtype)

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    # Test with return_logvar
    output, logvar = teacher(x, t, condition=condition, return_logvar=True)
    assert output.shape == x.shape
    assert logvar.shape == torch.Size([batch_size, 1])

    # return features, without early return
    output = teacher(x, t, condition=condition, return_features_early=False, feature_indices={0})
    assert output[0].shape == x.shape
    assert isinstance(output[1], list) and len(output[1]) == 1
    for feature in output[1]:
        assert feature.shape == (batch_size, 480, T, H, W)

    # return features, with early return
    output = teacher(x, t, condition=condition, return_features_early=True, feature_indices={0})
    assert isinstance(output, list) and len(output) == 1
    for feature in output:
        # Feature shape should match the new tensor dimensions (original model channels)
        expected_channels = 480  # Original CogVideoX channels
        assert feature.shape == (batch_size, expected_channels, T, H, W)

    # One-step sample() smoke (full guidance path; CogVideoX's per-step
    # ``attrs.evolve`` of guidance_cfg.cfg.scale runs through here; the
    # DPM++ multistep solver falls back to a first-order step at i=0
    # via the ``prev_pred_xhat is None`` branch).
    _smoke_sample(teacher, x, condition=condition, neg_condition=condition, decode_after=True)

    # Clear memory after testing
    clear_gpu_memory()


# ---------------------------------------------------------------------------
# Image / video preprocessor wrappers — split-precision boundaries (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name,class_name,module_attr",
    [
        ("fastgen.networks.Flux.network", "FluxImageEncoder", "vae"),
        ("fastgen.networks.Flux2.network", "Flux2ImageEncoder", "vae"),
        ("fastgen.networks.QwenImage.network", "QwenImageImageEncoder", "vae"),
        ("fastgen.networks.DiT.network", "SDVAE", "_vae"),
        ("fastgen.networks.CogVideoX.network", "CogVideoXVideoEncoder", "vae"),
        ("fastgen.networks.SD15.network", "StableDiffusionImageEncoder", "vae"),
        ("fastgen.networks.SDXL.network", "SDXLImageEncoder", "vae"),
    ],
)
def test_standard_vae_wrappers_enforce_split_precision_on_encode_decode(module_name, class_name, module_attr):
    """Every ordinary VAE preserves the caller's device and computes and returns in model dtype."""

    class _Distribution:
        def __init__(self, value):
            self.mean = value
            self.std = torch.zeros_like(value)

        def sample(self):
            return self.mean

        def mode(self):
            return self.mean

    class _EncodeResult:
        def __init__(self, distribution):
            self.distribution = distribution

        def __getitem__(self, key):
            assert key in (0, "latent_dist")
            return self.distribution

    class _DecodeResult:
        def __init__(self, value):
            self.sample = value

        def __getitem__(self, key):
            assert key == 0
            return self.sample

    class _TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(
                scaling_factor=1.0,
                latents_mean=(0.0, 0.0, 0.0),
                latents_std=(1.0, 1.0, 1.0),
                z_dim=3,
                force_upcast=False,
            )
            self.bn = torch.nn.BatchNorm2d(3)
            self.encode_input_dtype = None
            self.decode_input_dtype = None
            self.encode_input_device = None
            self.decode_input_device = None

        @property
        def dtype(self):
            return self.anchor.dtype

        @property
        def device(self):
            return self.anchor.device

        def encode(self, value, **kwargs):
            del kwargs
            self.encode_input_dtype = value.dtype
            self.encode_input_device = value.device
            return _EncodeResult(_Distribution(torch.zeros_like(value)))

        def decode(self, value, **kwargs):
            del kwargs
            self.decode_input_dtype = value.dtype
            self.decode_input_device = value.device
            return _DecodeResult(torch.zeros_like(value))

    wrapper_type = getattr(importlib.import_module(module_name), class_name)
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    core = _TinyVAE()
    setattr(wrapper, module_attr, core)
    wrapper.scaling_factor = 1.0
    wrapper.shift_factor = 0.0
    wrapper.latents_mean = torch.zeros(1, 3, 1, 1)
    wrapper.latents_std = torch.ones(1, 3, 1, 1)
    wrapper._patch_h = wrapper._patch_w = 1
    wrapper._latent_channels = 3
    wrapper._bn_eps = 1e-5
    wrapper._record_precision(precision_model=torch.float64)
    wrapper._set_device("meta")

    wrapper.to(dtype=torch.float64)
    assert core.anchor.dtype == wrapper.precision_model == torch.float64

    shape = (1, 3, 1, 2, 2) if class_name == "CogVideoXVideoEncoder" else (1, 3, 2, 2)
    state = torch.randn(shape, dtype=torch.float32)
    encoded = wrapper.encode(state)
    decoded = wrapper.decode(encoded)

    assert core.encode_input_dtype == torch.float64
    assert core.decode_input_dtype == torch.float64
    assert core.encode_input_device == state.device
    assert core.decode_input_device == state.device
    assert encoded.dtype == decoded.dtype == wrapper.precision_model

    wrapper.to(device="cpu")
    assert wrapper.precision_model == torch.float64


@pytest.mark.parametrize(
    ("module_name", "class_name", "module_attr"),
    [
        ("fastgen.networks.SD15.network", "StableDiffusionImageEncoder", "vae"),
        ("fastgen.networks.SDXL.network", "SDXLImageEncoder", "vae"),
        ("fastgen.networks.Flux.network", "FluxImageEncoder", "vae"),
        ("fastgen.networks.Flux2.network", "Flux2ImageEncoder", "vae"),
        ("fastgen.networks.DiT.network", "SDVAE", "_vae"),
    ],
)
@pytest.mark.parametrize(
    ("force_upcast", "expected_dtype"),
    [(False, torch.bfloat16), (True, torch.float32)],
)
def test_vae_force_upcast_controls_precision_boundary(
    module_name, class_name, module_attr, force_upcast, expected_dtype
):
    """VAEs configured for FP32 stay fixed while reduced-precision replacements remain configurable."""

    class _TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(force_upcast=force_upcast)

    wrapper_type = getattr(importlib.import_module(module_name), class_name)
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    setattr(wrapper, module_attr, _TinyVAE())
    wrapper._set_precision(precision_model=torch.bfloat16)

    assert wrapper.precision_model == expected_dtype
    assert wrapper.precision_opt == expected_dtype
    assert getattr(wrapper, module_attr).anchor.dtype == expected_dtype


def test_wan_encoder_precision_flag_defaults_to_true():
    """Wan encoders default to original Wan's precision."""
    from fastgen.networks.Wan.network import Wan

    assert inspect.signature(Wan.__init__).parameters["keep_enc_original_prec"].default is True


@pytest.mark.parametrize(
    ("module_name", "class_name", "model_id_attr", "expected_model_id"),
    [
        (
            "fastgen.networks.cosmos_predict2.network",
            "CosmosPredict2",
            None,
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ),
        ("fastgen.networks.cosmos3.network", "Cosmos3", "model_id_or_local_path", "/tmp/cosmos3"),
    ],
)
def test_cosmos_vae_explicitly_uses_native_bf16_boundary(
    monkeypatch, module_name, class_name, model_id_attr, expected_model_id
):
    """Cosmos forks intentionally opt out of standalone Wan's FP32 VAE policy."""
    module = importlib.import_module(module_name)
    captured = {}

    def make_vae(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(module, "WanVideoEncoder", make_vae)
    network = getattr(module, class_name).__new__(getattr(module, class_name))
    network.preprocessors = {}
    if model_id_attr is not None:
        setattr(network, model_id_attr, expected_model_id)

    network.init_vae()

    assert captured == {
        "model_id_or_local_path": expected_model_id,
        "fixed_precision_model": torch.bfloat16,
    }
    assert network.preprocessors["vae"] is not None


def test_cosmos3_loads_pinned_hf_snapshot_into_fastgen_backbone(monkeypatch, tmp_path):
    """The Hub supplies raw flat keys; Cosmos3 maps them onto its custom backbone exactly."""
    import huggingface_hub

    from fastgen.networks.cosmos3 import network as cosmos3_network

    source_network = _tiny_cosmos3(enable_logvar_linear=False)
    source_state = {name: tensor.detach().clone() for name, tensor in source_network.transformer.state_dict().items()}

    calls = {}

    def download(repo_id, **kwargs):
        calls["snapshot"] = {"repo_id": repo_id, **kwargs}
        return str(tmp_path)

    def load(path, device):
        calls["checkpoint"] = (path, device)
        return source_state

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(cosmos3_network, "load_checkpoint_state_dict", load)
    monkeypatch.setenv("HF_HOME", "/shared/hf")
    monkeypatch.setenv("LOCAL_FILES_ONLY", "true")

    network = _tiny_cosmos3(enable_logvar_linear=False, load_pretrained=True)

    assert network.model_id_or_local_path == str(tmp_path)
    assert calls == {
        "snapshot": {
            "repo_id": "nvidia/Cosmos3-Nano",
            "revision": Cosmos3.MODEL_REVISION,
            "allow_patterns": ["transformer/*", "text_tokenizer/*", "vae/*"],
            "cache_dir": "/shared/hf",
            "local_files_only": True,
        },
        "checkpoint": (str(tmp_path / Cosmos3.TRANSFORMER_INDEX), "cpu"),
    }
    for name, tensor in network.transformer.state_dict().items():
        assert torch.equal(tensor, source_state[name])


def test_cosmos3_nano_config_pins_immutable_hf_revision():
    assert Cosmos3.MODEL_REVISION == "03c14e74a6ddb51985d614b75d70f2443efc6a05"
    assert Cosmos3_Nano_Config.revision == Cosmos3.MODEL_REVISION


def test_cosmos3_meta_init_resolves_snapshot_but_skips_transformer_load(monkeypatch, tmp_path):
    """Meta ranks share the pinned component root while rank 0 supplies transformer tensors."""
    import huggingface_hub

    from fastgen.networks.cosmos3 import network as cosmos3_network

    snapshot_calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, **kwargs: snapshot_calls.append({"repo_id": repo_id, **kwargs}) or str(tmp_path),
    )
    monkeypatch.setenv("HF_HOME", "/shared/hf")
    monkeypatch.setenv("LOCAL_FILES_ONLY", "true")
    monkeypatch.setattr(
        cosmos3_network.Cosmos3,
        "_load_pretrained_transformer",
        lambda self: pytest.fail("a meta-initialized rank must not load the Hugging Face snapshot"),
    )
    with torch.device("meta"):
        network = _tiny_cosmos3(load_pretrained=True)

    assert all(parameter.is_meta for parameter in network.parameters())
    assert network.model_id_or_local_path == str(tmp_path)
    assert snapshot_calls == [
        {
            "repo_id": "nvidia/Cosmos3-Nano",
            "revision": Cosmos3.MODEL_REVISION,
            "allow_patterns": ["transformer/*", "text_tokenizer/*", "vae/*"],
            "cache_dir": "/shared/hf",
            "local_files_only": True,
        }
    ]


def test_cosmos3_preprocessors_share_hf_model_id_without_path_resolution(monkeypatch):
    """The VAE and tokenizer receive the same Hub ID directly."""
    from fastgen.networks.cosmos3 import network as cosmos3_network

    captured = {}
    monkeypatch.setattr(
        cosmos3_network, "Cosmos3TextTokenizer", lambda path: captured.setdefault("tokenizer", path) or object()
    )
    monkeypatch.setattr(
        cosmos3_network, "WanVideoEncoder", lambda **kwargs: captured.setdefault("vae", kwargs) or object()
    )

    network = Cosmos3.__new__(Cosmos3)
    network.preprocessors = {}
    network.model_id_or_local_path = "nvidia/Cosmos3-Nano"
    network.init_vae()
    network.init_text_tokenizer()

    assert captured == {
        "tokenizer": "nvidia/Cosmos3-Nano",
        "vae": {
            "model_id_or_local_path": "nvidia/Cosmos3-Nano",
            "fixed_precision_model": torch.bfloat16,
        },
    }


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("fastgen.networks.Flux2.network", "Flux2ImageEncoder"),
        ("fastgen.networks.QwenImage.network", "QwenImageImageEncoder"),
        ("fastgen.networks.Wan.network", "WanVideoEncoder"),
    ],
)
def test_checkpoint_latent_normalization_runs_in_vae_precision(module_name, class_name):
    """Checkpoint statistics use VAE precision even when the caller uses a lower precision."""

    # Keep the checkpoint statistic exactly representable in FP32 (as it is in
    # the reference Wan pipeline), while choosing a posterior value whose cast
    # to FP32 changes the subtraction result.
    mean_value = 1.0
    encoded_value = 1.00000008

    class _Distribution:
        def __init__(self, value):
            self.value = value
            self.mean = value
            self.std = torch.zeros_like(value)

        def sample(self):
            return self.value

        def mode(self):
            return self.value

    class _TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(1, dtype=torch.float64)
            self.bn.running_mean.fill_(mean_value)
            self.bn.running_var.fill_(1.0)
            self.config = SimpleNamespace(
                latents_mean=(mean_value,),
                latents_std=(1.0,),
                z_dim=1,
                force_upcast=False,
            )
            self.decode_input = None

        def encode(self, value, **kwargs):
            del kwargs
            return (_Distribution(torch.full_like(value, encoded_value)),)

        def decode(self, value, **kwargs):
            del kwargs
            self.decode_input = value.detach().clone()
            return (value,)

    wrapper_type = getattr(importlib.import_module(module_name), class_name)
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    wrapper.vae = _TinyVAE()
    wrapper.latents_mean = (mean_value,)
    wrapper.latents_std = (1.0,)
    wrapper._patch_h = wrapper._patch_w = wrapper._latent_channels = 1
    wrapper._bn_eps = 0.0
    if class_name == "WanVideoEncoder":
        wrapper.fixed_precision_model = None
    else:
        wrapper.keep_in_fp32 = False
    wrapper._record_precision(precision_model=torch.float64)

    shape = (1, 1, 1, 2, 2) if class_name == "WanVideoEncoder" else (1, 1, 2, 2)
    caller_value = torch.zeros(shape, dtype=torch.float32)
    encoded = wrapper.encode(caller_value)
    precise_delta = torch.tensor(encoded_value, dtype=torch.float64) - mean_value
    premature_delta = torch.tensor(encoded_value, dtype=torch.float32) - torch.tensor(mean_value, dtype=torch.float32)

    assert precise_delta.to(torch.float32) != premature_delta
    assert encoded.dtype == torch.float64
    assert torch.equal(encoded, torch.full_like(encoded, precise_delta))

    decode_state = torch.full_like(caller_value, encoded_value - mean_value)
    wrapper.decode(decode_state)
    expected_decode_value = decode_state.flatten()[0].to(torch.float64) + mean_value
    expected_decode_input = torch.full_like(wrapper.vae.decode_input, expected_decode_value, dtype=torch.float64)
    premature_decode_value = (decode_state.flatten()[0] + mean_value).to(torch.float64)
    premature_decode_input = torch.full_like(wrapper.vae.decode_input, premature_decode_value, dtype=torch.float64)

    assert not torch.equal(expected_decode_input, premature_decode_input)
    assert wrapper.vae.decode_input.dtype == torch.float64
    assert torch.equal(wrapper.vae.decode_input, expected_decode_input)


def test_flux2_bn_standard_deviation_is_cast_after_sqrt():
    """Match Flux2Pipeline's buffer-dtype sqrt before crossing to latent precision."""
    from fastgen.networks.Flux2.network import Flux2ImageEncoder

    wrapper = Flux2ImageEncoder.__new__(Flux2ImageEncoder)
    torch.nn.Module.__init__(wrapper)
    wrapper.vae = SimpleNamespace(bn=torch.nn.BatchNorm2d(1, dtype=torch.float64))
    wrapper.vae.bn.running_var.fill_(1.00000012)
    wrapper._patch_h = wrapper._patch_w = wrapper._latent_channels = 1
    wrapper._bn_eps = 0.0

    _, actual = wrapper._bn_stats_per_position(torch.float32)
    expected = torch.sqrt(torch.tensor(1.00000012, dtype=torch.float64)).to(torch.float32)
    cast_before_sqrt = torch.sqrt(torch.tensor(1.00000012, dtype=torch.float32))

    assert expected != cast_before_sqrt
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected.reshape_as(actual))


def test_flux2_vae_encodes_the_posterior_mean_by_default():
    """BFL's ``AutoEncoder.encode`` and diffusers' Flux.2 pipelines and training script take the posterior mean."""
    from fastgen.networks.Flux2.network import Flux2ImageEncoder

    mean = torch.randn(1, 1, 2, 2)

    class Posterior:
        def mode(self):
            return mean

        def sample(self):
            return mean + 1

    wrapper = Flux2ImageEncoder.__new__(Flux2ImageEncoder)
    torch.nn.Module.__init__(wrapper)
    wrapper.vae = SimpleNamespace(
        bn=torch.nn.BatchNorm2d(4),  # identity statistics for one channel in 2 x 2 patches
        config=SimpleNamespace(force_upcast=True),
        encode=lambda data, return_dict: (Posterior(),),
    )
    wrapper._patch_h = wrapper._patch_w = 2
    wrapper._latent_channels = 1
    wrapper._bn_eps = 0.0
    wrapper._record_precision(precision_model=torch.float32)

    images = torch.randn(1, 3, 16, 16)
    torch.testing.assert_close(wrapper.encode(images), mean, rtol=0, atol=0)
    torch.testing.assert_close(wrapper.encode(images, mode="sample"), mean + 1, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("module_name", "class_name", "decode_autocast_dtype"),
    [
        ("fastgen.networks.Flux.network", "FluxImageEncoder", torch.bfloat16),
        ("fastgen.networks.SD15.network", "StableDiffusionImageEncoder", torch.float16),
    ],
)
@pytest.mark.parametrize(
    ("force_upcast", "outer_autocast"), [(False, False), (False, True), (True, False), (True, True)]
)
def test_image_vae_decode_autocast_contract(
    module_name, class_name, decode_autocast_dtype, force_upcast, outer_autocast
):
    expected_autocast_dtype = decode_autocast_dtype if force_upcast else torch.bfloat16 if outer_autocast else None

    class _TinyVAE:
        config = SimpleNamespace(force_upcast=force_upcast, scaling_factor=2.0)

        def decode(self, latents, return_dict=False):
            assert not return_dict
            assert latents.dtype == torch.float32
            assert torch.is_autocast_enabled("cpu") is (expected_autocast_dtype is not None)
            if expected_autocast_dtype is not None:
                assert torch.get_autocast_dtype("cpu") == expected_autocast_dtype
            return (latents,)

    wrapper_type = getattr(importlib.import_module(module_name), class_name)
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    wrapper.vae = _TinyVAE()
    wrapper.scaling_factor = 2.0
    wrapper.shift_factor = 0.25
    wrapper._record_precision(precision_model=torch.float32)

    state = torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16)
    outer_context = (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16) if outer_autocast else contextlib.nullcontext()
    )
    with outer_context:
        decoded = wrapper.decode(state)

    assert decoded.dtype == wrapper.precision_model


@pytest.mark.parametrize(
    ("module_name", "class_name", "module_attr", "uses_shift", "shape"),
    [
        ("fastgen.networks.Flux.network", "FluxImageEncoder", "vae", True, (1, 3, 2, 2)),
        ("fastgen.networks.SD15.network", "StableDiffusionImageEncoder", "vae", False, (1, 3, 2, 2)),
        ("fastgen.networks.SDXL.network", "SDXLImageEncoder", "vae", False, (1, 3, 2, 2)),
        ("fastgen.networks.CogVideoX.network", "CogVideoXVideoEncoder", "vae", False, (1, 3, 1, 2, 2)),
        ("fastgen.networks.DiT.network", "SDVAE", "_vae", False, (1, 3, 2, 2)),
    ],
)
def test_scalar_latent_normalization_runs_in_vae_precision(module_name, class_name, module_attr, uses_shift, shape):
    """Scalar checkpoint transforms run beside the VAE, in its precision."""
    encoded_value = 16_777_217.0
    scaling_factor = 1.5
    shift_factor = 0.25

    class _Distribution:
        def __init__(self, value):
            self.value = value
            self.mean = value
            self.std = torch.zeros_like(value)

        def sample(self):
            return self.value

    class _EncodeResult:
        def __init__(self, distribution):
            self.latent_dist = distribution

        def __getitem__(self, key):
            assert key in (0, "latent_dist")
            return self.latent_dist

    class _DecodeResult:
        def __init__(self, value):
            self.sample = value

        def __getitem__(self, key):
            assert key == 0
            return self.sample

    class _TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(scaling_factor=scaling_factor, force_upcast=False)
            self.decode_input = None

        def encode(self, value, return_dict=False):
            assert not return_dict
            encoded = torch.full_like(value, encoded_value, dtype=torch.float64)
            return _EncodeResult(_Distribution(encoded))

        def decode(self, value, return_dict=False):
            assert not return_dict
            self.decode_input = value.detach().clone()
            return _DecodeResult(value)

    wrapper_type = getattr(importlib.import_module(module_name), class_name)
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    core = _TinyVAE()
    setattr(wrapper, module_attr, core)
    wrapper.scaling_factor = scaling_factor
    wrapper.shift_factor = shift_factor
    wrapper._record_precision(precision_model=torch.float64)

    caller_value = torch.zeros(shape, dtype=torch.float32)
    encoded = wrapper.encode(caller_value)
    produced = torch.full(shape, encoded_value, dtype=torch.float64)
    expected_encoded = produced - shift_factor if uses_shift else produced
    expected_encoded = expected_encoded * scaling_factor
    premature_encoded = produced.to(torch.float32) - shift_factor if uses_shift else produced.to(torch.float32)
    premature_encoded = premature_encoded * scaling_factor

    assert not torch.equal(expected_encoded.to(torch.float32), premature_encoded)
    assert encoded.dtype == torch.float64
    assert torch.equal(encoded, expected_encoded)

    caller_latents = torch.ones(shape, dtype=torch.float32)
    wrapper.decode(caller_latents)
    expected_decode_input = caller_latents.to(torch.float64) / scaling_factor
    premature_decode_input = (caller_latents / scaling_factor).to(torch.float64)
    if uses_shift:
        expected_decode_input = expected_decode_input + shift_factor
        premature_decode_input = premature_decode_input + shift_factor

    assert not torch.equal(expected_decode_input, premature_decode_input)
    assert core.decode_input.dtype == torch.float64
    assert torch.equal(core.decode_input, expected_decode_input)


def test_dit_sdvae_preencoded_latents_preserve_input_device_and_precision():
    """The pre-encoded mean/std path keeps the caller-owned device and dtype."""
    from fastgen.networks.DiT.network import SDVAE

    wrapper = SDVAE.__new__(SDVAE)
    torch.nn.Module.__init__(wrapper)
    wrapper.scaling_factor = 0.5
    wrapper._set_device("meta")

    preencoded = torch.randn(1, 8, 2, 2, dtype=torch.float64)
    encoded = wrapper.encode(preencoded)

    assert encoded.dtype == preencoded.dtype
    assert encoded.device == preencoded.device


def test_diffusers_vae_dtype_policy_declarations_are_audited():
    """Trip if an ordinary upstream VAE gains a precision carve-out we must honor."""
    from diffusers import AutoencoderKL, AutoencoderKLCogVideoX, AutoencoderKLQwenImage, AutoencoderKLWan
    from diffusers.models.autoencoders import (
        AutoencoderKLFlux2,
        AutoencoderKLLTX2Audio,
        AutoencoderKLLTX2Video,
    )

    ordinary_vaes = (
        AutoencoderKL,
        AutoencoderKLFlux2,
        AutoencoderKLQwenImage,
        AutoencoderKLCogVideoX,
        AutoencoderKLWan,
        AutoencoderKLLTX2Video,
        AutoencoderKLLTX2Audio,
    )
    assert all(getattr(vae, "_keep_in_fp32_modules", None) is None for vae in ordinary_vaes)


@pytest.mark.parametrize(
    ("module_name", "class_name", "model_loader_name", "tokenizer_loader_name", "with_mask"),
    [
        (
            "fastgen.networks.SD15.network",
            "StableDiffusionTextEncoder",
            "CLIPTextModel",
            "CLIPTokenizer",
            True,
        ),
        (
            "fastgen.networks.CogVideoX.network",
            "CogVideoXTextEncoder",
            "T5EncoderModel",
            "T5Tokenizer",
            False,
        ),
    ],
)
def test_text_encoder_encodes_empty_prompts_like_any_other_prompt(
    monkeypatch,
    module_name,
    class_name,
    model_loader_name,
    tokenizer_loader_name,
    with_mask,
):
    """No unconditional cache: empty prompts take the live path, so their dtype follows the encoder."""

    class StubTextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
            self.calls = []

        def forward(self, input_ids):
            self.calls.append((input_ids.shape[0], self.training))
            return (input_ids.to(dtype=self.scale.dtype).unsqueeze(-1) * self.scale,)

    class StubTokenizer:
        model_max_length = 4

        def __call__(self, prompts, **kwargs):
            del kwargs
            input_ids = torch.arange(self.model_max_length, dtype=torch.long).repeat(len(prompts), 1)
            input_ids[:, 0] = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long)
            return SimpleNamespace(
                input_ids=input_ids,
                attention_mask=torch.ones(len(prompts), self.model_max_length, dtype=torch.long),
            )

    module = importlib.import_module(module_name)
    core = StubTextEncoder()
    initial_requires_grad = core.scale.requires_grad
    monkeypatch.setattr(getattr(module, model_loader_name), "from_pretrained", MagicMock(return_value=core))
    monkeypatch.setattr(
        getattr(module, tokenizer_loader_name), "from_pretrained", MagicMock(return_value=StubTokenizer())
    )

    encoder = getattr(module, class_name)("unused")

    # Construction runs no encoder pass and preserves the loader's policy until FastGen assigns the role.
    assert core.calls == []
    assert encoder.training and core.training
    assert core.scale.requires_grad is initial_requires_grad

    encoder.prepare_module(device="cpu", precision_model=torch.float32)
    assert all(not child.training for child in encoder.modules())
    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    def split(result):
        return result if with_mask else (result, None)

    # An all-empty batch is encoded row by row like any other batch, in the encoder's storage dtype.
    embeddings, mask = split(encoder.encode(["", ""]))
    assert core.calls == [(2, False)]
    assert embeddings.shape[0] == 2 and embeddings.dtype == core.scale.dtype
    if mask is not None:
        assert mask.shape[0] == 2 and mask.dtype == torch.long

    # Mixed batches keep their row order and take the same path.
    embeddings, _ = split(encoder.encode(["a prompt", ""]))
    assert core.calls[-1] == (2, False) and embeddings.shape[0] == 2
    assert not torch.equal(embeddings[0], embeddings[1])

    # The empty prompt follows the encoder's live dtype rather than a construction-time cache.
    core.to(torch.bfloat16)
    embeddings, _ = split(encoder.encode(["", "", ""]))
    assert embeddings.dtype == torch.bfloat16 and embeddings.shape[0] == 3

    # Under one execution policy, empty and live prompts always share a dtype.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        empty, _ = split(encoder.encode(["", ""]))
        live, _ = split(encoder.encode(["a prompt"]))
    assert empty.dtype == live.dtype


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_wan():
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    set_env_vars()
    teacher_config = Wan_1_3B_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Wan config doesn't support model_channels/num_blocks override, use default
    teacher = instantiate(teacher_config)

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    T, H, W = 2, 4, 4  # Reduced from 4, 8, 16

    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=_STATE_DTYPE)  # [B, C, T, H, W]
    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        condition = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        condition, teacher.preprocessors["text_encoder"].precision_model, "wan.text_encoder.encode"
    )
    condition = condition.to(device=device, dtype=dtype)

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    # Forward pass without autocast
    output = teacher(x, t, condition=condition)

    output = teacher(x, t, condition=condition, return_features_early=True, feature_indices={0})

    assert isinstance(output, list) and len(output) == 1

    for feature in output:
        # Use original model channels (not overridden)
        expected_channels = 384  # Original Wan channels
        assert feature.shape == (batch_size, expected_channels, T, H, W)

    # One-step sample() smoke (full guidance path; sequential CFG with
    # condition broadcast via ``predict_with_guidance``).
    _smoke_sample(teacher, x, condition=condition, neg_condition=condition, decode_after=True)

    # Clear memory after testing
    clear_gpu_memory()


@RunIf(min_gpus=1, min_gpu_mem=24)
@pytest.mark.large_model
def test_network_vace_wan(tmp_path):
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    set_env_vars()
    # The logvar head is off by default; opt in so the `return_logvar` path below is actually reachable.
    teacher_config = copy.deepcopy(VACE_Wan_1_3B_Config)
    teacher_config.enable_logvar_linear = True

    # Supply the depth checkpoint before preprocessors are constructed. This test exercises
    # depth plumbing, so random weights avoid downloading the official depth checkpoint.
    from fastgen.third_party.annotators.depth_anything_v2.dpt import DepthAnythingV2

    depth_ckpt = tmp_path / "depth_anything_v2_vitl.pth"
    torch.save(
        DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]).state_dict(),
        depth_ckpt,
    )
    teacher_config.depth_model_path = str(depth_ckpt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    teacher = instantiate(teacher_config)

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    # B, C, T, H, W
    x = torch.randn(batch_size, 16, 2, 4, 4, device=device, dtype=_STATE_DTYPE)  # Reduced from 21, 60, 104
    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="logitnormal").to(device=device)

    # Prepare text embeddings
    captions = ["a caption"]
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        text_embeds = teacher.preprocessors["text_encoder"].encode(captions)
    _assert_gpu_tree_dtype_and_device(
        text_embeds, teacher.preprocessors["text_encoder"].precision_model, "vace.text_encoder.encode"
    )
    text_embeds = text_embeds.to(device=device, dtype=dtype)

    # Create a dummy video for depth extraction (B, C, T, H, W) in [-1, 1] that encodes to x's latent shape
    context_video = torch.randn(batch_size, 3, *teacher.pixel_shape(x.shape[1:]), device=device, dtype=_STATE_DTYPE)
    context_video = torch.clamp(context_video, -1, 1)  # Ensure it's in [-1, 1] range

    # Prepare VACE conditioning (``_build_vid_context`` uses both preprocessors internally,
    # so they must be on device for the call).
    with _borrow_on_device(teacher.preprocessors["depth_annotator"], device), _borrow_on_device(
        teacher.preprocessors["vae"], device
    ):
        vid_context = teacher._build_vid_context(context_video).to(dtype=dtype)

    # Create condition dict with both text_embeds and vid_context
    condition = {"text_embeds": text_embeds, "vid_context": vid_context}

    # The checkpoint is still float32 here, so PyTorch correctly uses its unfused
    # RMSNorm path for mixed float32/bfloat16 operands.
    with _autocast_fp32_weights(device, dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    # Forward pass without autocast
    output = teacher(x, t, condition=condition)

    # Test with return_logvar (requires the `enable_logvar_linear=True` override above).
    output_logvar, logvar = teacher(x, t, condition=condition, return_logvar=True)
    assert output_logvar.shape == x.shape
    assert logvar.shape == torch.Size([batch_size, 1])

    output = teacher(x, t, condition=condition, return_features_early=True, feature_indices={0})

    assert isinstance(output, list) and len(output) == 1

    expected_channels = 384  # Original VACE channels (not overridden)
    for feature in output:
        assert feature.shape == (batch_size, expected_channels, 2, 4, 4)  # Adjusted dimensions

    # One-step sample() smoke (full guidance path; dict condition with
    # text_embeds + vid_context).
    _smoke_sample(teacher, x, condition=condition, neg_condition=condition, decode_after=True)

    # Clear memory after testing
    clear_gpu_memory()


def test_wan_video_encoder_preserves_float32_when_moved():
    """A fixed FP32 precision boundary keeps the VAE in float32."""
    from fastgen.networks.Wan.network import WanVideoEncoder

    encoder = WanVideoEncoder.__new__(WanVideoEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.vae = torch.nn.Linear(2, 2)
    encoder.fixed_precision_model = torch.float32
    with torch.no_grad():
        encoder.vae.weight.flatten()[0] = 1.0001
    expected = encoder.vae.weight.detach().clone()

    encoder._set_precision(precision_model=torch.bfloat16)
    encoder.to(device="cpu")

    assert encoder.vae.weight.dtype == torch.float32
    assert encoder.precision_model == torch.float32
    assert torch.equal(encoder.vae.weight, expected)


def test_wan_video_encoder_honors_consumer_fixed_bf16_precision():
    """Cosmos' BF16 VAE remains BF16 even if the surrounding model requests FP32."""
    from fastgen.networks.Wan.network import WanVideoEncoder

    encoder = WanVideoEncoder.__new__(WanVideoEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.vae = torch.nn.Linear(2, 2)
    encoder.fixed_precision_model = torch.bfloat16

    encoder._set_precision(precision_model=torch.float32)

    assert encoder.precision_model == torch.bfloat16
    assert encoder.precision_opt == torch.bfloat16
    assert encoder.vae.weight.dtype == torch.bfloat16


def test_wan_text_encoder_uses_reference_bf16_eager_attention(monkeypatch):
    from fastgen.networks.Wan import network as wan_network

    model_loader = MagicMock(return_value=MagicMock())
    tokenizer_loader = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(wan_network.UMT5EncoderModel, "from_pretrained", model_loader)
    monkeypatch.setattr(wan_network.T5TokenizerFast, "from_pretrained", tokenizer_loader)
    monkeypatch.setenv("HF_HOME", "/tmp/hf")

    wan_network.WanTextEncoder("local-wan")

    kwargs = model_loader.call_args.kwargs
    assert kwargs["dtype"] is torch.bfloat16
    assert kwargs["attn_implementation"] == "eager"


class _WanBoundaryOp(torch.nn.Module):
    def __init__(self, name, records, value=None):
        super().__init__()
        self.name = name
        self.records = records
        self.value = value

    def forward(self, hidden_states, **kwargs):
        self.records.setdefault(self.name, []).append(
            (hidden_states.dtype, getattr(kwargs.get("encoder_hidden_states"), "dtype", None))
        )
        if self.value is None:
            return hidden_states
        return torch.full_like(hidden_states, self.value)


def _make_wan_boundary_block(records, vace=False):
    block = torch.nn.Module()
    block.scale_shift_table = torch.nn.Parameter(torch.zeros(1, 6, 4))
    block.norm1 = _WanBoundaryOp("norm1", records)
    block.norm2 = _WanBoundaryOp("norm2", records)
    block.norm3 = _WanBoundaryOp("norm3", records)
    block.attn1 = _WanBoundaryOp("attn1", records, 1.0)
    block.attn2 = _WanBoundaryOp("attn2", records, 1e-3)
    block.ffn = _WanBoundaryOp("ffn", records, 0.5)
    if vace:
        block.proj_in = _WanBoundaryOp("proj_in", records)
        block.proj_out = _WanBoundaryOp("proj_out", records)
    return block


@pytest.mark.parametrize("core_dtype", [torch.bfloat16, torch.float32])
def test_wan_block_uses_unconditional_fp32_carrier_and_context_core_dtype(monkeypatch, core_dtype):
    from fastgen.networks.Wan.network import block_forward

    records = {}
    block = _make_wan_boundary_block(records)
    hidden_states = torch.ones(1, 1, 4, dtype=core_dtype)
    encoder_hidden_states = torch.zeros_like(hidden_states)
    temb = torch.zeros(1, 6, 4)
    rotary_emb = torch.zeros_like(hidden_states)

    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: pytest.fail("unexpected autocast"))
    first = block_forward(block, hidden_states, temb, encoder_hidden_states, rotary_emb, False)
    second = block_forward(block, first, temb, encoder_hidden_states, rotary_emb, False)

    assert [dtype for dtype, _ in records["norm1"]] == [core_dtype, torch.float32]
    for name in ("attn1", "attn2", "ffn"):
        assert [dtype for dtype, _ in records[name]] == [core_dtype, core_dtype]
    assert [context_dtype for _, context_dtype in records["attn2"]] == [core_dtype, core_dtype]
    assert first.dtype == second.dtype == torch.float32
    if core_dtype is torch.bfloat16:
        assert not torch.equal(first, first.to(torch.bfloat16).float())
        assert not torch.equal(second, second.to(torch.bfloat16).float())


@pytest.mark.parametrize("core_dtype", [torch.bfloat16, torch.float32])
def test_wan_block_fp32_residual_flag_rounds_carrier_at_block_boundary(core_dtype):
    """``fp32_residual=False`` rounds the carrier once per block; blocks without the flag keep fp32."""
    from fastgen.networks.Wan.network import block_forward

    block = _make_wan_boundary_block({})
    hidden_states = torch.ones(1, 1, 4, dtype=core_dtype)
    encoder_hidden_states = torch.zeros_like(hidden_states)
    temb = torch.zeros(1, 6, 4)
    rotary_emb = torch.zeros_like(hidden_states)

    assert block_forward(block, hidden_states, temb, encoder_hidden_states, rotary_emb, False).dtype == torch.float32
    block.fp32_residual = False
    assert block_forward(block, hidden_states, temb, encoder_hidden_states, rotary_emb, False).dtype == core_dtype
    block.fp32_residual = True
    assert block_forward(block, hidden_states, temb, encoder_hidden_states, rotary_emb, False).dtype == torch.float32


@pytest.mark.parametrize("core_dtype", [torch.bfloat16, torch.float32])
def test_vace_control_block_uses_shared_carrier_boundaries_and_override_binding(core_dtype):
    from diffusers.models.embeddings import Timesteps

    from fastgen.networks.VaceWan import network as vace_network
    from fastgen.networks.Wan import network as wan_network

    records = {}
    main_block = torch.nn.Module()
    control_block = _make_wan_boundary_block(records, vace=True)
    transformer = torch.nn.Module()
    transformer.blocks = torch.nn.ModuleList([main_block])
    transformer.vace_blocks = torch.nn.ModuleList([control_block])
    transformer.condition_embedder = torch.nn.Module()
    transformer.condition_embedder.timesteps_proj = Timesteps(8, flip_sin_to_cos=True, downscale_freq_shift=0)
    transformer.r_embedder = None

    network = object.__new__(vace_network.VACEWan)
    torch.nn.Module.__init__(network)
    network.transformer = transformer
    network._keep_original_prec = False
    network._sdpa_no_cudnn_blocks = None
    network.transformer.encoder_depth = None
    network.override_transformer_forward()

    assert main_block.forward.__func__ is wan_network.block_forward
    assert control_block.forward.__func__.__module__ == vace_network.__name__
    assert transformer.forward.__func__ is vace_network.vace_classify_forward
    t = torch.tensor([999.0], dtype=torch.float64)
    assert torch.equal(transformer.condition_embedder.timesteps_proj(t), wan_network.sinusoidal_embedding_1d_wan(8, t))

    hidden_states = torch.ones(1, 1, 4, dtype=core_dtype)
    encoder_hidden_states = torch.zeros_like(hidden_states)
    temb = torch.zeros(1, 6, 4)
    rotary_emb = torch.zeros_like(hidden_states)
    hint, carrier = control_block(
        hidden_states,
        encoder_hidden_states,
        torch.ones_like(hidden_states),
        temb,
        rotary_emb,
        False,
    )
    control_block.proj_in = None
    second_hint, second_carrier = control_block(
        hidden_states,
        encoder_hidden_states,
        carrier,
        temb,
        rotary_emb,
        False,
    )

    assert [dtype for dtype, _ in records["proj_in"]] == [core_dtype]
    assert [dtype for dtype, _ in records["norm1"]] == [core_dtype, torch.float32]
    for name in ("attn1", "attn2", "ffn", "proj_out"):
        assert [dtype for dtype, _ in records[name]] == [core_dtype, core_dtype]
    assert carrier.dtype == second_carrier.dtype == torch.float32
    assert hint.dtype == second_hint.dtype == core_dtype


def test_wan_video_encoder_to_does_not_redefine_precision_by_default():
    """Default placement follows the requested storage dtype without changing policy."""
    from fastgen.networks.Wan.network import WanVideoEncoder

    encoder = WanVideoEncoder.__new__(WanVideoEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.vae = torch.nn.Linear(2, 2)
    encoder.fixed_precision_model = None
    encoder._record_precision(precision_model=torch.float32)

    encoder.to(device="cpu", dtype=torch.bfloat16)

    assert encoder.vae.weight.dtype == torch.bfloat16
    assert encoder.precision_model == torch.float32


def test_wan_video_encoder_keeps_retained_fp32_vae_inputs_and_outputs():
    """The retained FP32 VAE receives FP32 tensors and returns its FP32 results."""
    from fastgen.networks.Wan.network import WanVideoEncoder

    class Distribution:
        def __init__(self, value):
            self.value = value

        def sample(self):
            return self.value

        def mode(self):
            return self.value

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(latents_mean=(0.0, 0.0), latents_std=(1.0, 1.0), z_dim=2)
            self.encode_input_dtype = None
            self.decode_input_dtype = None
            self.encode_input_device = None
            self.decode_input_device = None
            self.encode_autocast_enabled = None
            self.decode_autocast_enabled = None

        @property
        def device(self):
            return self.anchor.device

        @property
        def dtype(self):
            return self.anchor.dtype

        def encode(self, value, return_dict):
            assert not return_dict
            self.encode_input_dtype = value.dtype
            self.encode_input_device = value.device
            self.encode_autocast_enabled = torch.is_autocast_enabled("cpu")
            return (Distribution(value[:, :2] * self.anchor),)

        def decode(self, value, return_dict):
            assert not return_dict
            self.decode_input_dtype = value.dtype
            self.decode_input_device = value.device
            self.decode_autocast_enabled = torch.is_autocast_enabled("cpu")
            return (value * self.anchor,)

    encoder = WanVideoEncoder.__new__(WanVideoEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.vae = TinyVAE()
    encoder.fixed_precision_model = torch.float32
    encoder._set_precision(precision_model=torch.bfloat16)
    encoder._set_device("meta")

    pixels = torch.randn(1, 2, 1, 2, 2, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        latents = encoder.encode(pixels, mode="argmax")
        decoded = encoder.decode(latents)

    assert encoder.vae.dtype == torch.float32
    assert encoder.precision_model == torch.float32
    assert encoder.vae.encode_input_dtype == torch.float32
    assert encoder.vae.decode_input_dtype == torch.float32
    assert encoder.vae.encode_input_device == pixels.device
    assert encoder.vae.decode_input_device == pixels.device
    assert latents.dtype == decoded.dtype == torch.float32


def test_wan_image_encoder_fixed_precision_ignores_the_enclosing_autocast():
    """Original Wan stores and computes CLIP in FP16: the fixed precision sets the weights, the input and the output
    dtype, and the encoder disables the enclosing autocast."""
    from fastgen.networks.WanI2V.network import WanImageEncoder

    class TinyCLIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 3)
            self.config = SimpleNamespace(image_size=4)
            self.input_dtype = None
            self.autocast_enabled = None

        def forward(self, pixel_values, output_hidden_states):
            assert output_hidden_states
            self.input_dtype = pixel_values.dtype
            self.autocast_enabled = torch.is_autocast_enabled("cpu")
            hidden = self.proj(pixel_values.mean(dim=(-2, -1))).unsqueeze(1)
            return SimpleNamespace(hidden_states=(hidden, hidden))

    encoder = WanImageEncoder.__new__(WanImageEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.image_encoder = TinyCLIP()
    encoder.image_processor = SimpleNamespace(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    encoder.fixed_precision_model = torch.float16
    with torch.no_grad():
        # BF16 rounds this to 1.0, FP16 keeps 1.0009766: a cast through BF16 would show.
        encoder.image_encoder.proj.weight.flatten()[0] = 1.001
    expected = encoder.image_encoder.proj.weight.detach().to(torch.float16)

    encoder._set_precision(precision_model=torch.bfloat16)
    encoder._set_device("cpu")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        image_embeds = encoder.encode(torch.randn(1, 3, 4, 4, dtype=torch.bfloat16))

    assert encoder.precision_model == torch.float16
    assert encoder.image_encoder.proj.weight.dtype == torch.float16
    assert torch.equal(encoder.image_encoder.proj.weight, expected)
    assert encoder.image_encoder.input_dtype == torch.float16
    assert encoder.image_encoder.autocast_enabled is False
    assert image_embeds.dtype == torch.float16


def test_wan_image_encoder_to_does_not_redefine_precision_by_default():
    """Default placement follows the requested storage dtype without changing policy."""
    from fastgen.networks.WanI2V.network import WanImageEncoder

    encoder = WanImageEncoder.__new__(WanImageEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.image_encoder = torch.nn.Linear(3, 3)
    encoder.fixed_precision_model = None
    encoder._record_precision(precision_model=torch.float32)

    encoder.to(device="cpu", dtype=torch.bfloat16)

    assert encoder.image_encoder.weight.dtype == torch.bfloat16
    assert encoder.precision_model == torch.float32


@pytest.mark.parametrize("keep_enc_original_prec", [False, True])
def test_wan_encoder_precision_flag(monkeypatch, keep_enc_original_prec):
    """Wan applies the encoder precision flag to both the VAE and I2V CLIP encoder: original Wan's FP32 VAE and
    FP16 CLIP."""
    from fastgen.networks.WanI2V import network as wan_i2v_network

    calls = []

    class StubImageEncoder:
        def __init__(self, model_id_or_local_path, fixed_precision_model):
            calls.append(("image", model_id_or_local_path, fixed_precision_model))

    class StubVideoEncoder:
        def __init__(self, model_id_or_local_path, fixed_precision_model):
            calls.append(("vae", model_id_or_local_path, fixed_precision_model))

    monkeypatch.setattr(wan_i2v_network, "WanImageEncoder", StubImageEncoder)
    monkeypatch.setattr(wan_i2v_network, "WanVideoEncoder", StubVideoEncoder)

    network = object.__new__(wan_i2v_network.WanI2V)
    torch.nn.Module.__init__(network)
    network.preprocessors = {}
    network.model_id_or_local_path = "/tmp/wan"
    network._keep_enc_original_prec = keep_enc_original_prec
    network.init_image_encoder()
    network.init_vae()

    assert calls == [
        ("image", "/tmp/wan", torch.float16 if keep_enc_original_prec else None),
        ("vae", "/tmp/wan", torch.float32 if keep_enc_original_prec else None),
    ]


@pytest.mark.parametrize("keep_original_prec", [True, False])
def test_wan_set_precision_honors_keep_original_prec(keep_original_prec):
    """Every Wan parameter, Diffusers' declared FP32 modules included, follows the requested precision;
    ``keep_original_prec`` keeps only the rotary tables FP32."""
    from fastgen.networks.Wan.network import Wan

    class TinyTransformer(torch.nn.Module):
        _keep_in_fp32_modules = ["time_embedder", "scale_shift_table"]

        def __init__(self):
            super().__init__()
            self.time_embedder = torch.nn.Linear(2, 2)
            self.proj = torch.nn.Linear(2, 2)
            self.scale_shift_table = torch.nn.Parameter(torch.tensor([1.0001]))
            self.rope = torch.nn.Module()
            self.rope.register_buffer("freqs_cos", torch.tensor([1.0001]))
            self.condition_embedder = torch.nn.Module()
            self.condition_embedder.time_proj = torch.nn.Linear(2, 2)
            self.r_embedder = torch.nn.Module()
            self.r_embedder.time_proj = torch.nn.Linear(2, 2)

    network = object.__new__(Wan)
    torch.nn.Module.__init__(network)
    network.transformer = TinyTransformer()
    network._keep_original_prec = keep_original_prec
    retained = network.transformer.rope.freqs_cos.clone()

    network._set_precision(precision_model=torch.bfloat16)

    assert all(parameter.dtype == torch.bfloat16 for parameter in network.transformer.parameters())
    rope_dtype = torch.float32 if keep_original_prec else torch.bfloat16
    assert network.transformer.rope.freqs_cos.dtype == rope_dtype
    if keep_original_prec:
        assert torch.equal(network.transformer.rope.freqs_cos, retained)


def test_wan_lora_split_precision_composes_with_architecture_overrides():
    """The LoRA frozen-base policy and Wan's FP32 rotary tables are both retained."""
    from fastgen.networks.Wan.network import Wan

    class TinyTransformer(torch.nn.Module):
        _keep_in_fp32_modules = ["time_embedder"]

        def __init__(self):
            super().__init__()
            self.time_embedder = torch.nn.Linear(2, 2)
            self.proj = torch.nn.Linear(2, 2)
            self.lora_A = torch.nn.Parameter(torch.ones(2, 2))
            self.rope = torch.nn.Module()
            self.rope.register_buffer("freqs_cos", torch.ones(2))

    network = object.__new__(Wan)
    torch.nn.Module.__init__(network)
    network.transformer = TinyTransformer()
    network._keep_original_prec = True
    network._lora_enabled = True

    network._set_precision(precision_model=torch.bfloat16, precision_opt=torch.float32)

    assert network.transformer.proj.weight.dtype == torch.bfloat16
    assert network.transformer.lora_A.dtype == torch.float32
    assert network.transformer.time_embedder.weight.dtype == torch.bfloat16
    assert network.transformer.rope.freqs_cos.dtype == torch.float32


def test_causal_wan_rotary_offset_uses_diffusers_split_dimensions_and_rope_dtype():
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
    from fastgen.networks.VaceWan.network_causal import (
        compute_rotary_emb_with_time_offset as compute_vace_rotary,
    )
    from fastgen.networks.Wan.network_causal import compute_rotary_emb_with_time_offset as compute_wan_rotary

    rope = WanRotaryPosEmbed(attention_head_dim=64, patch_size=(1, 1, 1), max_seq_len=4).to(dtype=torch.float64)
    assert (rope.t_dim, rope.h_dim, rope.w_dim) == (24, 20, 20)

    hidden_states = torch.zeros(1, 1, 2, 2, 2, dtype=torch.bfloat16)
    wan_cos, wan_sin = compute_wan_rotary(rope, hidden_states, start_frame=1)
    vace_cos, vace_sin = compute_vace_rotary(rope, hidden_states, start_frame=1)

    assert wan_cos.shape == wan_sin.shape == (1, 8, 1, 64)
    assert wan_cos.dtype == wan_sin.dtype == rope.freqs_cos.dtype == torch.float64
    assert torch.equal(vace_cos, wan_cos)
    assert torch.equal(vace_sin, wan_sin)


def test_noncausal_wan_and_vace_prepare_preserve_rope_dtype():
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
    from fastgen.networks.VaceWan.network import vace_classify_forward_prepare
    from fastgen.networks.Wan.network import classify_forward_prepare

    class TinyConditionEmbedder:
        def __call__(
            self,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image=None,
            timestep_seq_len=None,
        ):
            del timestep_seq_len
            batch_size = timestep.shape[0]
            return (
                encoder_hidden_states.new_zeros(batch_size, 1),
                encoder_hidden_states.new_zeros(batch_size, 6),
                encoder_hidden_states,
                encoder_hidden_states_image,
            )

    rope = WanRotaryPosEmbed(attention_head_dim=64, patch_size=(1, 1, 1), max_seq_len=4).to(dtype=torch.float64)
    transformer = SimpleNamespace(
        rope=rope,
        patch_embedding=torch.nn.Identity(),
        condition_embedder=TinyConditionEmbedder(),
        r_embedder=None,
    )
    hidden_states = torch.zeros(1, 1, 2, 2, 2, dtype=torch.bfloat16)
    timestep = torch.zeros(1)
    encoder_hidden_states = torch.zeros(1, 1, 1, dtype=torch.bfloat16)

    wan_out = classify_forward_prepare(transformer, hidden_states, timestep, encoder_hidden_states)
    vace_out = vace_classify_forward_prepare(transformer, hidden_states, timestep, encoder_hidden_states)
    wan_rotary = wan_out[5]
    vace_rotary = vace_out[5]
    assert torch.equal(wan_out[1], vace_out[1])
    assert torch.equal(wan_out[4], vace_out[4])

    for rotary_emb in (wan_rotary, vace_rotary):
        assert isinstance(rotary_emb, tuple)
        assert all(emb.dtype == rope.freqs_cos.dtype == torch.float64 for emb in rotary_emb)
        assert all(emb.device == hidden_states.device for emb in rotary_emb)


def _tiny_wan_transformer(seed: int = 0, attention_head_dim: int = 8):
    """Two-block Wan transformer small enough to run on CPU."""
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    torch.manual_seed(seed)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=attention_head_dim,
        in_channels=4,
        out_channels=4,
        text_dim=8,
        freq_dim=16,
        ffn_dim=16,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=32,
    ).eval()


def test_wan_feature_output_contract_distinguishes_none_from_empty_set():
    from fastgen.networks.Wan.network import Wan

    network = object.__new__(Wan)
    torch.nn.Module.__init__(network)
    network.expand_timesteps = False
    network.net_pred_type = "flow"
    network._record_precision(precision_model=torch.float32)
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer = _tiny_wan_transformer(seed=0)
    network.transformer.r_embedder = None
    network.transformer.encoder_depth = None
    network.transformer.time_cond_type = "abs"
    network.transformer.norm_temb = False
    network._keep_original_prec = False
    network._sdpa_no_cudnn_blocks = None
    network.override_transformer_forward()
    network.eval()

    x_t = torch.randn(1, 4, 2, 4, 4)
    timestep = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
    condition = torch.randn(1, 3, 8)

    with torch.no_grad():
        score = network(x_t, timestep, condition=condition, feature_indices=None)
        score_and_features = network(x_t, timestep, condition=condition, feature_indices=set())

    assert isinstance(score, torch.Tensor)
    assert isinstance(score_and_features, list)
    assert torch.equal(score_and_features[0], score)
    assert score_and_features[1] == []

    with torch.no_grad():
        for return_features_early in (False, True):
            output = network(
                x_t,
                timestep,
                condition=condition,
                feature_indices={0, 1},
                skip_layers=[1],
                return_features_early=return_features_early,
            )
            features = output if return_features_early else output[1]
            assert len(features) == 2
            assert torch.equal(features[0], features[1])


@pytest.mark.parametrize("per_token", [False, True])
def test_wan_condition_embedding_forward_matches_diffusers_in_a_uniform_dtype(per_token):
    """Without a dtype boundary, the condition embedder reproduces the pinned Diffusers forward exactly."""
    from fastgen.networks.Wan.network import wan_condition_embedding_forward, wan_timestep_embedding_forward

    embedder = _wan_embedder_stub(seed=1).transformer.condition_embedder
    embedder.time_embedder.forward = MethodType(wan_timestep_embedding_forward, embedder.time_embedder)
    embedder.keep_original_prec = False
    timestep, seq_len = torch.tensor([400.0, 300.0]), None
    if per_token:
        timestep, seq_len = timestep.repeat_interleave(3), 3
    text = torch.randn(2, 5, 8)

    with torch.no_grad():
        expected = WanTimeTextImageEmbedding.forward(embedder, timestep, text, timestep_seq_len=seq_len)
        actual = wan_condition_embedding_forward(embedder, timestep, text, timestep_seq_len=seq_len)

    assert actual[3] is None and expected[3] is None
    for actual_tensor, expected_tensor in zip(actual[:3], expected[:3]):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_wan_rms_norm_forward_rounds_like_original_wan(weight_dtype):
    """Q/K norms reproduce ``WanRMSNorm`` bit for bit: normalize in FP32, round to the input dtype, then scale, so a
    BF16 weight rounds the product again and an FP32 weight returns FP32 into RoPE."""
    from fastgen.networks.Wan.network import wan_rms_norm_forward

    torch.manual_seed(0)
    norm = torch.nn.RMSNorm(16, eps=1e-6, dtype=weight_dtype)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.3 * torch.randn(16))
    norm.forward = MethodType(wan_rms_norm_forward, norm)
    x = (4 * torch.randn(3, 5, 16)).to(torch.bfloat16)

    with torch.no_grad():
        # Original Wan's ``WanRMSNorm.forward``.
        x_fp32 = x.float()
        expected = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).type_as(x) * norm.weight
        actual = norm(x)

    assert actual.dtype == torch.promote_types(torch.bfloat16, weight_dtype)
    assert torch.equal(actual, expected)


def test_wan_rms_norm_forward_keeps_fp64_inputs_in_fp64():
    from fastgen.networks.Wan.network import wan_rms_norm_forward

    norm = torch.nn.RMSNorm(16, eps=1e-6, dtype=torch.float64)
    reference = copy.deepcopy(norm)
    norm.forward = MethodType(wan_rms_norm_forward, norm)
    x = torch.randn(3, 5, 16, dtype=torch.float64)

    with torch.no_grad():
        actual = norm(x)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, reference(x), rtol=1e-12, atol=0)


@pytest.mark.parametrize("keep_original_prec", [True, False])
@pytest.mark.parametrize("kind", ["wan", "vace"])
def test_wan_family_rounds_every_qk_norm_like_original_wan(kind, keep_original_prec):
    """``keep_original_prec`` gives the Q/K norms of every attention, VACE control blocks included, ``WanRMSNorm``'s
    rounding, and deep copies rebind it to the copied norms."""
    from fastgen.networks.VaceWan.network import VACEWan
    from fastgen.networks.Wan.network import Wan, wan_rms_norm_forward

    network_cls, factory = (Wan, _tiny_wan_transformer) if kind == "wan" else (VACEWan, _tiny_vace_transformer)
    network = object.__new__(network_cls)
    torch.nn.Module.__init__(network)
    network.transformer = factory(seed=0)
    network.transformer.r_embedder = None
    network._keep_original_prec = keep_original_prec
    network._sdpa_no_cudnn_blocks = None
    network.override_transformer_forward()

    num_blocks = 3 if kind == "vace" else 2
    for copied in (network, copy.deepcopy(network)):
        blocks = copied._transformer_blocks()
        norms = [module for block in blocks for module in block.modules() if isinstance(module, torch.nn.RMSNorm)]
        assert len(blocks) == num_blocks and len(norms) == 4 * num_blocks
        for norm in norms:
            installed = norm.__dict__.get("forward")
            if keep_original_prec:
                assert installed.__func__ is wan_rms_norm_forward and installed.__self__ is norm
            else:
                assert installed is None


@RunIf(min_gpus=1)
@pytest.mark.parametrize("bf16_autocast", [False, True], ids=["no_autocast", "bf16_autocast"])
def test_wan_time_path_stays_fp32_like_original_wan(bf16_autocast):
    """Original Wan computes the time embedding, its projection and the output projection in FP32 under a BF16 model
    and BF16 autocast, where Diffusers rounds ``temb`` to the text's dtype before ``time_proj``. With
    ``keep_original_prec`` these BF16 modules compute under FP32 autocast, so ``temb``, every block's modulation and
    the output projection are FP32 while the text, the blocks and the logvar head compute in BF16."""
    from fastgen.networks.Wan.network import Wan

    network = object.__new__(Wan)
    torch.nn.Module.__init__(network)
    network.expand_timesteps = False
    network.net_pred_type = "flow"
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer = _attach_wan_r_embedder(_tiny_wan_transformer(seed=0))
    config = network.transformer.config
    network.transformer.logvar_linear = torch.nn.Linear(config.num_attention_heads * config.attention_head_dim, 1)
    network._keep_original_prec = True
    network._sdpa_no_cudnn_blocks = None
    network.override_transformer_forward()
    network._set_precision(precision_model=torch.bfloat16)
    network.to("cuda").eval()

    transformer = network.transformer
    embedder = transformer.condition_embedder
    time_path = (embedder.time_embedder, embedder.time_proj, transformer.r_embedder.time_proj, transformer.proj_out)
    for module in time_path:
        assert all(parameter.dtype == torch.bfloat16 for parameter in module.parameters())

    observed = {}
    embedder.register_forward_pre_hook(lambda module, args: observed.update(timestep=args[0]))
    embedder.register_forward_hook(lambda module, args, output: observed.update(condition=output))
    transformer.blocks[0].register_forward_pre_hook(lambda module, args: observed.update(block_temb=args[1]))
    transformer.proj_out.register_forward_hook(lambda module, args, output: observed.update(head=output))

    x_t = torch.randn(1, 4, 2, 4, 4, device="cuda")
    t = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision, device="cuda")
    r = torch.tensor([0.2], dtype=network.noise_scheduler.t_precision, device="cuda")
    condition = torch.randn(1, 3, 8, device="cuda").to(torch.bfloat16)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if bf16_autocast else contextlib.nullcontext()
    with torch.no_grad():
        with autocast:
            _, logvar = network(x_t, t, condition=condition, r=r, return_logvar=True)
        temb, timestep_proj, text, _ = observed["condition"]
        with torch.autocast("cuda", dtype=torch.float32):
            expected_temb = embedder.time_embedder(embedder.timesteps_proj(observed["timestep"]))
            expected_proj = embedder.time_proj(embedder.act_fn(expected_temb))
            text_rounded_proj = embedder.time_proj(embedder.act_fn(expected_temb.to(text.dtype)).float())

    assert temb.dtype == timestep_proj.dtype == observed["block_temb"].dtype == observed["head"].dtype == torch.float32
    assert text.dtype == torch.bfloat16
    assert torch.equal(temb, expected_temb)
    assert torch.equal(timestep_proj, expected_proj)
    assert not torch.equal(timestep_proj, text_rounded_proj)
    assert logvar.shape == (1, 1) and torch.isfinite(logvar).all()


@pytest.mark.parametrize(
    "keep_original_prec,precision_model,bf16_autocast,expected",
    [
        (True, torch.float32, True, torch.float32),
        (False, torch.float32, True, torch.bfloat16),
        (False, torch.bfloat16, False, torch.bfloat16),
        (False, torch.bfloat16, True, torch.bfloat16),
    ],
)
def test_wan_time_path_and_head_follow_the_network_without_keep_original_prec(
    keep_original_prec, precision_model, bf16_autocast, expected
):
    """Without ``keep_original_prec`` the time path, its r branch included, and the output projection compute like the
    rest of the network, in the model dtype or the active autocast's. With it they stay FP32 even under BF16 autocast
    of an FP32-stored model."""
    from fastgen.networks.Wan.network import Wan

    network = object.__new__(Wan)
    torch.nn.Module.__init__(network)
    network.expand_timesteps = False
    network.net_pred_type = "flow"
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer = _attach_wan_r_embedder(_tiny_wan_transformer(seed=0))
    network._keep_original_prec = keep_original_prec
    network._sdpa_no_cudnn_blocks = None
    network.override_transformer_forward()
    network._set_precision(precision_model=precision_model)
    network.eval()

    transformer = network.transformer
    observed = {}
    transformer.condition_embedder.register_forward_hook(lambda module, args, output: observed.update(condition=output))
    transformer.blocks[0].register_forward_pre_hook(lambda module, args: observed.update(block_temb=args[1]))
    transformer.proj_out.register_forward_hook(lambda module, args, output: observed.update(head=output))

    t = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
    r = torch.tensor([0.2], dtype=network.noise_scheduler.t_precision)
    autocast = torch.autocast("cpu", dtype=torch.bfloat16) if bf16_autocast else contextlib.nullcontext()
    with torch.no_grad(), autocast:
        network(torch.randn(1, 4, 2, 4, 4), t, condition=torch.randn(1, 3, 8).to(precision_model), r=r)

    temb, timestep_proj = observed["condition"][:2]
    assert temb.dtype == timestep_proj.dtype == observed["block_temb"].dtype == observed["head"].dtype == expected


def _attach_wan_r_embedder(
    transformer, fusion="additive", encoder_depth=None, time_cond_type="abs", gate_value=0.25, keep_original_prec=False
):
    r_embedder = copy.deepcopy(transformer.condition_embedder)
    del r_embedder.text_embedder
    del r_embedder.image_embedder
    r_embedder.fusion_mode = fusion
    r_embedder.gate_value = float(gate_value)
    if fusion == "gated":
        del r_embedder.time_proj
        del r_embedder.act_fn
    transformer.r_embedder = r_embedder
    transformer.encoder_depth = encoder_depth
    transformer.time_cond_type = time_cond_type
    transformer.norm_temb = False
    from fastgen.networks.Wan.network import wan_condition_embedding_forward, wan_timestep_embedding_forward

    for embedder in (transformer.condition_embedder, transformer.r_embedder):
        embedder.time_embedder.forward = MethodType(wan_timestep_embedding_forward, embedder.time_embedder)
    transformer.condition_embedder.forward = MethodType(wan_condition_embedding_forward, transformer.condition_embedder)
    for module in (transformer, transformer.condition_embedder, transformer.r_embedder):
        module.keep_original_prec = keep_original_prec
    return transformer


def _wan_family_prepare_fns():
    from fastgen.networks.VaceWan.network import vace_classify_forward_prepare
    from fastgen.networks.VaceWan.network_causal import vace_causal_classify_forward_prepare
    from fastgen.networks.Wan.network import classify_forward_prepare
    from fastgen.networks.Wan.network_causal import classify_forward_prepare as causal_classify_forward_prepare
    from fastgen.networks.WanI2V.network_cam import _wancami2v_classify_forward_prepare

    return (
        ("wan", classify_forward_prepare),
        ("causal_wan", causal_classify_forward_prepare),
        ("vace", vace_classify_forward_prepare),
        ("causal_vace", vace_causal_classify_forward_prepare),
        ("cam", _wancami2v_classify_forward_prepare),
    )


@pytest.mark.parametrize("fusion", ["additive", "gated"])
@pytest.mark.parametrize("encoder_depth", [None, 1])
def test_wan_family_prepares_are_bitwise_identical_for_shared_r_fusion(fusion, encoder_depth):
    """Every Wan-family prepare must fold t/r the same way on a shared transformer."""
    transformer = _attach_wan_r_embedder(_tiny_vace_transformer(seed=0), fusion=fusion, encoder_depth=encoder_depth)
    hidden_states = torch.randn(1, 4, 2, 4, 4)
    timestep = torch.tensor([0.7])
    encoder_hidden_states = torch.randn(1, 3, 8)
    r_timestep = torch.tensor([0.2])

    reference = None
    for name, prepare in _wan_family_prepare_fns():
        out = prepare(transformer, hidden_states, timestep, encoder_hidden_states, r_timestep)
        temb, timestep_proj, r_timestep_proj = out[4], out[1], out[2]
        if reference is None:
            reference = (temb, timestep_proj, r_timestep_proj)
            continue
        assert torch.equal(temb, reference[0]), name
        assert torch.equal(timestep_proj, reference[1]), name
        if reference[2] is None:
            assert r_timestep_proj is None, name
        else:
            assert torch.equal(r_timestep_proj, reference[2]), name


@RunIf(min_gpus=1)
@pytest.mark.parametrize("fusion", ["additive", "gated"])
@pytest.mark.parametrize("kind", ["wan", "causal_wan", "vace", "causal_vace", "cam"])
def test_wan_family_prepares_keep_the_fp32_time_path(kind, fusion):
    """Every Wan-family prepare hands the blocks original Wan's FP32 ``temb`` and modulation from a BF16 model."""
    from fastgen.networks.Wan.network import Wan

    network = object.__new__(Wan)
    torch.nn.Module.__init__(network)
    network.transformer = _attach_wan_r_embedder(
        _tiny_vace_transformer(seed=0), fusion=fusion, encoder_depth=1, keep_original_prec=True
    )
    network._keep_original_prec = True
    network._set_precision(precision_model=torch.bfloat16)
    network.to("cuda")

    prepare = dict(_wan_family_prepare_fns())[kind]
    hidden_states = torch.randn(1, 4, 2, 4, 4, device="cuda").to(torch.bfloat16)
    text = torch.randn(1, 3, 8, device="cuda").to(torch.bfloat16)
    t, r = torch.tensor([0.7], device="cuda"), torch.tensor([0.2], device="cuda")
    with torch.no_grad():
        out = prepare(network.transformer, hidden_states, t, text, r)

    timestep_proj, r_timestep_proj, encoder_hidden_states, temb = out[1], out[2], out[3], out[4]
    assert temb.dtype == timestep_proj.dtype == r_timestep_proj.dtype == torch.float32
    assert encoder_hidden_states.dtype == torch.bfloat16


@pytest.mark.parametrize("encoder_depth", [None, 1])
def test_wan_and_vace_additive_forwards_match_legacy_inline_r_mix(encoder_depth):
    """Transformer outputs must stay bitwise identical to the pre-helper additive mix."""
    import fastgen.networks.VaceWan.network as vace_network
    import fastgen.networks.Wan.network as wan_network
    from fastgen.networks.Wan.network import flatten_timestep, unflatten_timestep_proj

    def legacy_additive_r_mix(self, timestep, r_timestep, temb, timestep_proj):
        if self.time_cond_type == "abs":
            pass
        elif self.time_cond_type == "diff":
            r_timestep = timestep - r_timestep
        else:
            raise ValueError(f"Invalid time condition: {self.time_cond_type}")
        r_timestep, rs_seq_len = flatten_timestep(r_timestep)
        r_timestep = self.r_embedder.timesteps_proj(r_timestep)
        remb = self.r_embedder.time_embedder(r_timestep)
        r_timestep_proj = self.r_embedder.time_proj(self.r_embedder.act_fn(remb))
        r_timestep_proj = unflatten_timestep_proj(r_timestep_proj, rs_seq_len)
        if self.encoder_depth is None:
            return temb + remb, timestep_proj + r_timestep_proj, None
        return remb, timestep_proj, r_timestep_proj

    def run_wan(transformer):
        transformer.classify_forward_prepare = MethodType(wan_network.classify_forward_prepare, transformer)
        transformer.classify_forward_block_forward = MethodType(wan_network.classify_forward_block_forward, transformer)
        transformer.forward = MethodType(wan_network.classify_forward, transformer)
        for block in transformer.blocks:
            block.forward = MethodType(wan_network.block_forward, block)
        return transformer

    def run_vace(transformer):
        transformer.classify_forward_prepare = MethodType(vace_network.vace_classify_forward_prepare, transformer)
        transformer.classify_forward_block_forward = MethodType(
            vace_network.vace_classify_forward_block_forward, transformer
        )
        transformer.forward = MethodType(vace_network.vace_classify_forward, transformer)
        for block in transformer.blocks:
            block.forward = MethodType(wan_network.block_forward, block)
        for block in transformer.vace_blocks:
            block.forward = MethodType(vace_network.vace_block_forward, block)
        return transformer

    hidden_states = torch.randn(1, 4, 2, 4, 4)
    timestep = torch.tensor([400.0])
    encoder_hidden_states = torch.randn(1, 3, 8)
    r_timestep = torch.tensor([50.0])
    control_hidden_states = torch.randn(1, 8, 2, 4, 4)

    for kind, factory, bind, extra in (
        ("wan", _tiny_wan_transformer, run_wan, {}),
        (
            "vace",
            _tiny_vace_transformer,
            run_vace,
            {"control_hidden_states": control_hidden_states, "control_hidden_states_scale": torch.ones(1)},
        ),
    ):
        torch.manual_seed(3)
        transformer = bind(_attach_wan_r_embedder(factory(seed=3), fusion="additive", encoder_depth=encoder_depth))
        if hasattr(transformer, "disable_gradient_checkpointing"):
            transformer.disable_gradient_checkpointing()

        def _inputs():
            cloned = {key: value.clone() if torch.is_tensor(value) else value for key, value in extra.items()}
            return dict(
                hidden_states=hidden_states.clone(),
                timestep=timestep.clone(),
                encoder_hidden_states=encoder_hidden_states.clone(),
                r_timestep=r_timestep.clone(),
                feature_indices=None,
                **cloned,
            )

        with torch.no_grad():
            new_out = transformer(**_inputs())

        module = wan_network if kind == "wan" else vace_network
        original = module.apply_r_embedding

        def legacy_apply(self, timestep, r_timestep, temb, timestep_proj):
            return legacy_additive_r_mix(self, timestep, r_timestep, temb, timestep_proj)

        module.apply_r_embedding = legacy_apply
        try:
            with torch.no_grad():
                legacy_out = transformer(**_inputs())
        finally:
            module.apply_r_embedding = original
        assert torch.equal(new_out, legacy_out), kind


def _main_branch_additive_apply_r_embedding(self, timestep, r_timestep, temb, timestep_proj):
    """Main-branch additive r-mix: explicit dtype cast and leftover ``r_proj`` when ``encoder_depth`` is None."""
    from fastgen.networks.Wan.network import flatten_timestep, unflatten_timestep_proj

    if self.r_embedder is not None and r_timestep is not None:
        if self.time_cond_type == "abs":
            pass
        elif self.time_cond_type == "diff":
            r_timestep = timestep - r_timestep
        else:
            raise ValueError(f"Invalid time condition: {self.time_cond_type}")

        r_timestep, rs_seq_len = flatten_timestep(r_timestep)
        r_timestep = self.r_embedder.timesteps_proj(r_timestep)
        time_embedder_dtype = next(iter(self.r_embedder.time_embedder.parameters())).dtype
        if r_timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            r_timestep = r_timestep.to(time_embedder_dtype)
        remb = self.r_embedder.time_embedder(r_timestep)
        r_timestep_proj = self.r_embedder.time_proj(self.r_embedder.act_fn(remb))
        r_timestep_proj = unflatten_timestep_proj(r_timestep_proj, rs_seq_len)
        if self.encoder_depth is None:
            return temb + remb, timestep_proj + r_timestep_proj, r_timestep_proj
        return remb, timestep_proj, r_timestep_proj
    if r_timestep is not None:
        raise ValueError("r_timestep provided but no r_embedder is present")
    return temb, timestep_proj, None


def _bind_wan_family_forwards(kind, transformer):
    import fastgen.networks.VaceWan.network as vace_network
    import fastgen.networks.VaceWan.network_causal as vace_causal
    import fastgen.networks.Wan.network as wan_network
    import fastgen.networks.Wan.network_causal as causal_wan
    from fastgen.networks.Wan.network_causal import CausalWanAttnProcessor

    if kind == "wan":
        transformer.classify_forward_prepare = MethodType(wan_network.classify_forward_prepare, transformer)
        transformer.classify_forward_block_forward = MethodType(wan_network.classify_forward_block_forward, transformer)
        transformer.forward = MethodType(wan_network.classify_forward, transformer)
        for block in transformer.blocks:
            block.forward = MethodType(wan_network.block_forward, block)
        apply_module = wan_network
    elif kind == "vace":
        transformer.classify_forward_prepare = MethodType(vace_network.vace_classify_forward_prepare, transformer)
        transformer.classify_forward_block_forward = MethodType(
            vace_network.vace_classify_forward_block_forward, transformer
        )
        transformer.forward = MethodType(vace_network.vace_classify_forward, transformer)
        for block in transformer.blocks:
            block.forward = MethodType(wan_network.block_forward, block)
        for block in transformer.vace_blocks:
            block.forward = MethodType(vace_network.vace_block_forward, block)
        apply_module = vace_network
    elif kind == "causal_wan":
        transformer.classify_forward_prepare = MethodType(causal_wan.classify_forward_prepare, transformer)
        transformer.classify_forward_block_forward = MethodType(causal_wan.classify_forward_block_forward, transformer)
        transformer.forward = MethodType(causal_wan.classify_forward, transformer)
        for block in transformer.blocks:
            block.forward = MethodType(causal_wan.block_forward, block)
        transformer.set_attn_processor(CausalWanAttnProcessor())
        apply_module = causal_wan
    elif kind == "causal_vace":
        transformer.classify_forward_prepare = MethodType(vace_causal.vace_causal_classify_forward_prepare, transformer)
        transformer.classify_forward_block_forward = MethodType(
            vace_causal.vace_causal_classify_forward_block_forward, transformer
        )
        transformer.forward = MethodType(vace_causal.vace_causal_classify_forward, transformer)
        for block in transformer.blocks:
            block.forward = MethodType(causal_wan.block_forward, block)
        for block in transformer.vace_blocks:
            block.forward = MethodType(vace_causal._wan_vace_block_forward, block)
        transformer.set_attn_processor(CausalWanAttnProcessor())
        apply_module = vace_causal
    else:
        raise ValueError(kind)
    if hasattr(transformer, "disable_gradient_checkpointing"):
        transformer.disable_gradient_checkpointing()
    return transformer, apply_module


# The smallest head dimension CUDA flex attention supports.
_FLEX_ATTENTION_HEAD_DIM = 16


def _flex_attention_device() -> torch.device:
    """CUDA when available; otherwise the CPU, where Inductor lowers flex attention only with AVX2."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not torch.cpu._is_avx2_supported():
        pytest.skip("compiled flex attention needs CUDA or an AVX2 CPU")
    return torch.device("cpu")


def _clone_r_fusion_inputs(hidden_states, timestep, encoder_hidden_states, r_timestep, extra):
    return dict(
        hidden_states=hidden_states.clone(),
        timestep=timestep.clone(),
        encoder_hidden_states=encoder_hidden_states.clone(),
        r_timestep=r_timestep.clone(),
        feature_indices=None,
        **{key: value.clone() if torch.is_tensor(value) else value for key, value in extra.items()},
    )


@pytest.mark.parametrize("kind", ["wan", "vace", "causal_wan", "causal_vace"])
@pytest.mark.parametrize("encoder_depth", [None, 1])
@pytest.mark.parametrize("time_cond_type", ["abs", "diff"])
@pytest.mark.parametrize("t_layout", ["1d", "per_frame"])
def test_wan_family_additive_r_fusion_matches_main_branch(kind, encoder_depth, time_cond_type, t_layout):
    """Current prepares/forwards must match main's additive mix under the t/r layouts main supported.

    Non-causal VACE on main did not flatten ``t`` before ``condition_embedder``, so 2D t is not
    compared. Per-frame ``t``/``r`` (causal ``_compute_timestep_inputs``) raised on main because
    ``r`` was never unflattened like ``t``; ``r`` now follows ``t``, so each frame of the per-frame
    prepare must equal the 1-D prepare for that frame's ``(t, r)``. Leftover ``r_proj`` on main
    VACE/causal when ``encoder_depth is None`` is unused by the block loop; temb/timestep_proj and
    forwards must still match.
    """
    if kind == "vace" and t_layout == "per_frame":
        pytest.skip("main VACE did not flatten t before condition_embedder")

    factory = _tiny_vace_transformer if kind in ("vace", "causal_vace") else _tiny_wan_transformer
    torch.manual_seed(7)
    transformer, apply_module = _bind_wan_family_forwards(
        kind,
        _attach_wan_r_embedder(
            factory(seed=7, attention_head_dim=_FLEX_ATTENTION_HEAD_DIM),
            fusion="additive",
            encoder_depth=encoder_depth,
            time_cond_type=time_cond_type,
        ),
    )

    hidden_states = torch.randn(1, 4, 2, 4, 4)
    encoder_hidden_states = torch.randn(1, 3, 8)
    if t_layout == "1d":
        timestep = torch.tensor([400.0])
        r_timestep = torch.tensor([50.0])
    else:
        timestep = torch.tensor([[400.0, 300.0]])
        r_timestep = torch.tensor([[50.0, 40.0]])
    extra = {}
    if kind in ("vace", "causal_vace"):
        extra = {
            "control_hidden_states": torch.randn(1, 8, 2, 4, 4),
            "control_hidden_states_scale": torch.ones(1),
        }

    def _prepare_kwargs():
        return dict(
            hidden_states=hidden_states.clone(),
            timestep=timestep.clone(),
            encoder_hidden_states=encoder_hidden_states.clone(),
            r_timestep=r_timestep.clone(),
            **{key: value.clone() if torch.is_tensor(value) else value for key, value in extra.items()},
        )

    if t_layout == "per_frame":
        # Causal training expands both t and r to [B, F]; r is flattened before the ``diff`` and its
        # embedding unflattened like ``temb``, so frame i of the fused outputs (timestep_proj at index 1,
        # r_timestep_proj at 2, temb at 4) equals the 1-D prepare for (t_i, r_i).
        with torch.no_grad():
            per_frame = transformer.classify_forward_prepare(**_prepare_kwargs())
            for frame in range(timestep.shape[1]):
                frame_kwargs = _prepare_kwargs()
                frame_kwargs["timestep"] = timestep[:, frame].clone()
                frame_kwargs["r_timestep"] = r_timestep[:, frame].clone()
                single = transformer.classify_forward_prepare(**frame_kwargs)
                torch.testing.assert_close(per_frame[4][:, frame], single[4])
                torch.testing.assert_close(per_frame[1][:, frame], single[1])
                if single[2] is None:
                    assert per_frame[2] is None
                else:
                    torch.testing.assert_close(per_frame[2][:, frame], single[2])
        return

    if kind in ("causal_wan", "causal_vace"):
        device = _flex_attention_device()
        transformer.to(device)
        hidden_states, timestep, encoder_hidden_states, r_timestep = (
            tensor.to(device) for tensor in (hidden_states, timestep, encoder_hidden_states, r_timestep)
        )
        extra = to(extra, device=device)

    with torch.no_grad():
        new_prepare = transformer.classify_forward_prepare(**_prepare_kwargs())
        new_out = transformer(
            **_clone_r_fusion_inputs(hidden_states, timestep, encoder_hidden_states, r_timestep, extra)
        )

    original = apply_module.apply_r_embedding
    apply_module.apply_r_embedding = _main_branch_additive_apply_r_embedding
    try:
        with torch.no_grad():
            main_prepare = transformer.classify_forward_prepare(**_prepare_kwargs())
            main_out = transformer(
                **_clone_r_fusion_inputs(hidden_states, timestep, encoder_hidden_states, r_timestep, extra)
            )
    finally:
        apply_module.apply_r_embedding = original

    # prepare: hidden, timestep_proj, r_timestep_proj, encoder, temb, rotary, ...
    assert torch.equal(new_prepare[4], main_prepare[4]), "temb"
    assert torch.equal(new_prepare[1], main_prepare[1]), "timestep_proj"
    if encoder_depth is None:
        assert new_prepare[2] is None
        assert main_prepare[2] is not None
    else:
        assert torch.equal(new_prepare[2], main_prepare[2]), "r_timestep_proj"
    assert torch.equal(new_out, main_out)


@pytest.mark.parametrize("kind", ["wan", "vace", "cam", "causal_wan", "causal_vace"])
def test_wan_family_logvar_is_one_value_per_sample(kind):
    """Per-token (Wan2.2 TI2V) and per-frame (causal) timesteps give ``temb`` a sequence axis; ``return_logvar``
    still yields one ``[B, 1]`` value per sample, the per-sample result for the ``t`` its noised tokens share:
    tokens at timestep zero, like Wan2.2 TI2V's image-to-video first frame, are left out."""
    if kind == "cam":
        from fastgen.networks.WanI2V.network_cam import WanCamI2V

        network = object.__new__(WanCamI2V)
        torch.nn.Module.__init__(network)
        network.transformer = _attach_wan_r_embedder(_tiny_wan_transformer(seed=3))
        network._keep_original_prec = False
        network._sdpa_no_cudnn_blocks = None
        network.override_transformer_forward()
        network.latent_compression_factors = (4, 8, 8)
        network._add_camera_layers()
        transformer = network.transformer
    else:
        factory = _tiny_vace_transformer if kind.endswith("vace") else _tiny_wan_transformer
        transformer, _ = _bind_wan_family_forwards(kind, _attach_wan_r_embedder(factory(seed=3)))
    config = transformer.config
    transformer.logvar_linear = torch.nn.Linear(config.num_attention_heads * config.attention_head_dim, 1)
    inputs = dict(hidden_states=torch.randn(2, 4, 2, 4, 4), encoder_hidden_states=torch.randn(2, 3, 8))
    if kind.endswith("vace"):
        inputs.update(control_hidden_states=torch.randn(2, 8, 2, 4, 4), control_hidden_states_scale=torch.ones(1))
    t = torch.tensor([400.0, 300.0])
    # Two frames of four tokens: causal networks take t per frame, the others per token like Wan2.2 TI2V.
    expanded = t[:, None].expand(-1, 2 if kind.startswith("causal") else 8)
    first_frame_clean = expanded.clone()
    first_frame_clean[:, : expanded.shape[1] // 2] = 0
    embedder = transformer.condition_embedder

    def per_sample_logvar(times):
        return transformer.logvar_linear(embedder.time_embedder(embedder.timesteps_proj(times)))

    with torch.no_grad():
        _, logvar = transformer(timestep=expanded, return_logvar=True, **inputs)
        _, conditioned_logvar = transformer(timestep=first_frame_clean, return_logvar=True, **inputs)
        _, clean_logvar = transformer(timestep=torch.zeros_like(expanded), return_logvar=True, **inputs)
        expected = per_sample_logvar(t)
        expected_clean = per_sample_logvar(torch.zeros_like(t))

    for value in (logvar, conditioned_logvar, clean_logvar):
        assert value.shape == (2, 1)
    torch.testing.assert_close(logvar, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(conditioned_logvar, expected, rtol=1e-5, atol=1e-5)
    # A sample with every timestep at zero averages all of its tokens.
    torch.testing.assert_close(clean_logvar, expected_clean, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["wan", "vace"])
def test_norm_temb_modulation_agrees_across_timestep_layouts(kind):
    """``norm_temb`` normalizes each modulation vector on its own, in the main and the VACE control blocks alike: a
    per-sample ``t``, the per-token ``t`` of Wan2.2 TI2V and the causal per-frame ``t`` (one unmasked pass) give one
    output, so a causal student and its teacher agree."""
    factory = _tiny_vace_transformer if kind == "vace" else _tiny_wan_transformer
    reference, _ = _bind_wan_family_forwards(kind, _attach_wan_r_embedder(factory(seed=6)))
    causal, _ = _bind_wan_family_forwards(f"causal_{kind}", copy.deepcopy(reference))
    inputs = dict(hidden_states=torch.randn(1, 4, 2, 4, 4), encoder_hidden_states=torch.randn(1, 3, 8))
    if kind == "vace":
        inputs.update(control_hidden_states=torch.randn(1, 8, 2, 4, 4), control_hidden_states_scale=torch.ones(1))
    t = torch.tensor([400.0])

    outputs = {}
    for norm_temb in (False, True):
        reference.norm_temb = causal.norm_temb = norm_temb
        with torch.no_grad():
            outputs[norm_temb] = (
                reference(timestep=t, **inputs),
                reference(timestep=t[:, None].expand(-1, 8), **inputs),
                causal(timestep=t[:, None].expand(-1, 2), **inputs),
            )

    for per_sample, per_token, per_frame in outputs.values():
        torch.testing.assert_close(per_token, per_sample, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(per_frame, per_sample, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(outputs[True][0], outputs[False][0])


def test_vace_wan_fsdp_checkpointing_includes_control_blocks(monkeypatch):
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from fastgen.networks.VaceWan import network as vace_network
    from fastgen.networks.Wan import network as wan_network

    class MainBlock(torch.nn.Module):
        pass

    class VaceBlock(torch.nn.Module):
        pass

    class OtherBlock(torch.nn.Module):
        pass

    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gradient_checkpointing = True
            self.blocks = torch.nn.ModuleList([MainBlock()])
            self.vace_blocks = torch.nn.ModuleList([VaceBlock()])

        def disable_gradient_checkpointing(self):
            self.gradient_checkpointing = False

    network = object.__new__(vace_network.VACEWan)
    torch.nn.Module.__init__(network)
    network.transformer = TinyTransformer()
    network._use_fsdp_checkpoint = True

    checkpoint_block_types = []
    monkeypatch.setattr(
        wan_network,
        "apply_fsdp_checkpointing",
        lambda _module, block_types: checkpoint_block_types.append(block_types),
    )
    monkeypatch.setattr(wan_network, "fully_shard", lambda *_args, **_kwargs: None)

    network.fully_shard(mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))

    assert len(checkpoint_block_types) == 1
    block_types = checkpoint_block_types[0]
    assert set(block_types) == {MainBlock, VaceBlock}
    assert isinstance(network.transformer.blocks[0], block_types)
    assert isinstance(network.transformer.vace_blocks[0], block_types)
    assert not isinstance(OtherBlock(), block_types)
    assert not network.transformer.gradient_checkpointing


def test_vace_wan_fsdp_shards_each_control_block(monkeypatch):
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from fastgen.networks.VaceWan import network as vace_network
    from fastgen.networks.Wan import network as wan_network

    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gradient_checkpointing = False
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.vace_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    network = object.__new__(vace_network.VACEWan)
    torch.nn.Module.__init__(network)
    network.transformer = TinyTransformer()
    network._use_fsdp_checkpoint = True

    sharded = []
    monkeypatch.setattr(wan_network, "fully_shard", lambda module, **_kwargs: sharded.append(module))

    network.fully_shard(mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))

    transformer = network.transformer
    assert sharded == [transformer.blocks[0], transformer.vace_blocks[0], transformer]


def test_vace_wan_transformer_blocks_include_the_control_blocks():
    from fastgen.networks.VaceWan.network import VACEWan
    from fastgen.networks.Wan.network import Wan

    def bare_network(network_type, with_vace=False):
        network = object.__new__(network_type)
        torch.nn.Module.__init__(network)
        transformer = torch.nn.Module()
        transformer.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        if with_vace:
            transformer.vace_blocks = torch.nn.ModuleList([torch.nn.Identity()])
        network.transformer = transformer
        return network

    wan = bare_network(Wan)
    vace = bare_network(VACEWan, with_vace=True)

    assert wan._transformer_blocks() == tuple(wan.transformer.blocks)
    assert vace._transformer_blocks() == (*vace.transformer.blocks, *vace.transformer.vace_blocks)


def test_wan_sdpa_no_cudnn_hook_scopes_backends_and_survives_deepcopy():
    """The ``sdpa_no_cudnn_blocks`` hook disables cuDNN SDPA, leaving the other backends as enclosing, inside the
    configured self-attention blocks only, lets batch-0 inputs through untouched, ignores out-of-range indices, and
    stays bound to the copied module after ``copy.deepcopy``."""
    from fastgen.networks.Wan import network as wan_network

    def backend_flags():
        return {
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        }

    class RecordingAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)
            self.calls = []

        def forward(self, hidden_states, rotary_emb=None):
            self.calls.append((hidden_states.shape[0], backend_flags()))
            return self.proj(hidden_states)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = RecordingAttention()

    class TinyTransformer(torch.nn.Module):
        def __init__(self, num_blocks):
            super().__init__()
            self.blocks = torch.nn.ModuleList(Block() for _ in range(num_blocks))

    def make(no_cudnn_blocks):
        network = object.__new__(wan_network.Wan)
        torch.nn.Module.__init__(network)
        network.transformer = TinyTransformer(3)
        network._sdpa_no_cudnn_blocks = no_cudnn_blocks
        network._install_sdpa_no_cudnn_hooks()
        return network

    ambient = backend_flags()
    network = make([1, 3, -1])  # only index 1 exists
    assert ["forward" in block.attn1.__dict__ for block in network.transformer.blocks] == [False, True, False]
    x = torch.randn(2, 5, 4)
    for block in network.transformer.blocks:
        block.attn1(hidden_states=x)
    assert network.transformer.blocks[0].attn1.calls[0][1] == ambient
    inside = network.transformer.blocks[1].attn1.calls[0][1]
    assert inside == {**ambient, "cudnn": False}, inside
    assert backend_flags() == ambient, "SDPA backend flags were not restored after the hooked forward"
    network.transformer.blocks[1].attn1(torch.randn(0, 5, 4))  # positional batch-0 input bypasses the restriction
    assert network.transformer.blocks[1].attn1.calls[1] == (0, ambient)

    for disabled in (None, []):
        assert all("forward" not in block.attn1.__dict__ for block in make(disabled).transformer.blocks)

    copied = copy.deepcopy(network)
    copied_attn = copied.transformer.blocks[1].attn1
    assert copied_attn.forward.__self__ is copied_attn
    copied_attn(hidden_states=x)
    assert len(copied_attn.calls) == 3 and len(network.transformer.blocks[1].attn1.calls) == 2

    with pytest.raises(RuntimeError, match="already instance-bound"):
        network._install_sdpa_no_cudnn_hooks()


def test_wan_moe_set_precision_dispatches_to_expert_policies():
    """The parent MoE delegates its precision boundary to both Wan experts."""
    from fastgen.networks.Wan.network import Wan
    from fastgen.networks.Wan.wan_moe import WanMoE

    class TinyTransformer(torch.nn.Module):
        _keep_in_fp32_modules = ["time_embedder", "scale_shift_table"]

        def __init__(self):
            super().__init__()
            self.time_embedder = torch.nn.Linear(2, 2)
            self.proj = torch.nn.Linear(2, 2)
            self.scale_shift_table = torch.nn.Parameter(torch.tensor([1.0001]))
            self.rope = torch.nn.Module()
            self.rope.register_buffer("freqs_cos", torch.tensor([1.0001]))

    def tiny_expert():
        expert = object.__new__(Wan)
        torch.nn.Module.__init__(expert)
        expert.transformer = TinyTransformer()
        expert._keep_original_prec = True
        return expert

    network = object.__new__(WanMoE)
    torch.nn.Module.__init__(network)
    network.network_high = tiny_expert()
    network.network_low = tiny_expert()
    retained = {
        name: expert.transformer.rope.freqs_cos.clone()
        for name, expert in (("high", network.network_high), ("low", network.network_low))
    }

    network._set_precision(precision_model=torch.bfloat16)

    assert network.precision_model == torch.bfloat16
    for name, expert in (("high", network.network_high), ("low", network.network_low)):
        assert expert.precision_model == torch.bfloat16
        assert all(parameter.dtype == torch.bfloat16 for parameter in expert.transformer.parameters())
        assert expert.transformer.rope.freqs_cos.dtype == torch.float32
        assert torch.equal(expert.transformer.rope.freqs_cos, retained[name])


def test_wan_moe_fps_delegates_to_high_expert():
    from fastgen.networks.Wan.wan_moe import WanMoE

    network = object.__new__(WanMoE)
    object.__setattr__(network, "network_high", SimpleNamespace(fps=12.0))
    object.__setattr__(network, "network_low", SimpleNamespace(fps=12.0))

    assert network.fps == 12.0
    network.network_high.fps = 16.0
    with pytest.raises(AssertionError, match="experts must use the same FPS"):
        _ = network.fps
    network.network_low.fps = 16.0
    assert network.fps == 16.0


def test_wan_moe_generic_sample_uses_wrapped_default_guidance():
    from fastgen.networks.Wan.wan_moe import WanMoE

    class TinyWanSampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.guidance_cfg = GuidanceConfig()
            self.guidance_cfg.cfg.scale = 2.5

        def default_guidance_cfg(self):
            return self.guidance_cfg

        def sample(self, noise, guidance_cfg=None):
            del noise
            return self.default_guidance_cfg() if guidance_cfg is None else guidance_cfg

    network = object.__new__(WanMoE)
    torch.nn.Module.__init__(network)
    network.network_high = TinyWanSampler()

    assert network.sample(torch.zeros(1)) is network.network_high.guidance_cfg
    explicit = GuidanceConfig()
    assert network.sample(torch.zeros(1), guidance_cfg=explicit) is explicit


def test_wan_moe_expert_overrides_and_schedulers_share_configuration():
    from fastgen.networks.Wan.wan_moe import WanMoE

    class TinyWan(FastGenNetwork):
        def __init__(
            self,
            model_id_or_local_path,
            transformer_subfolder,
            net_pred_type,
            schedule_type,
            expert_name=None,
            sdpa_no_cudnn_blocks=None,
            **kwargs,
        ):
            del model_id_or_local_path
            super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **kwargs)
            self.transformer_subfolder = transformer_subfolder
            self.expert_name = expert_name
            self.sdpa_no_cudnn_blocks = sdpa_no_cudnn_blocks
            self.transformer = torch.nn.Identity()

        def forward(self, x_t, t, **kwargs):
            del t, kwargs
            return x_t

    default_network = WanMoE(TinyWan, "unused")
    assert default_network.network_high.transformer_subfolder == "transformer"
    assert default_network.network_low.transformer_subfolder == "transformer_2"

    for role in ("high", "low"):
        with pytest.raises(ValueError, match="must keep net_pred_type='flow'"):
            WanMoE(TinyWan, "unused", expert_overrides={role: {"net_pred_type": "x0"}})

    expert_overrides = copy.deepcopy(WanMoE.DEFAULT_EXPERT_OVERRIDES)
    expert_overrides["high"].update(expert_name="high", net_pred_type="flow")
    expert_overrides["low"].update(expert_name="low", net_pred_type="flow", sdpa_no_cudnn_blocks=[33])

    student_grid_cfg = StudentGridConfig(steps=3)
    network = WanMoE(
        TinyWan,
        "unused",
        min_t=0.125,
        max_t=0.875,
        t_precision="float32",
        student_grid_cfg=student_grid_cfg,
        expert_overrides=expert_overrides,
    )

    assert network.network_high.transformer_subfolder == "transformer"
    assert network.network_high.expert_name == "high"
    assert network.network_high.sdpa_no_cudnn_blocks is None
    assert network.network_low.transformer_subfolder == "transformer_2"
    assert network.network_low.expert_name == "low"
    assert network.network_low.sdpa_no_cudnn_blocks == [33]

    def assert_scheduler_contract():
        high = network.network_high.noise_scheduler
        low = network.network_low.noise_scheduler
        assert network.noise_scheduler is high
        assert low is not high
        for scheduler in (high, low):
            assert isinstance(scheduler, RFNoiseSchedule)
            assert scheduler.min_t == 0.125
            assert scheduler.max_t == 0.875
            assert scheduler.t_precision == torch.float32
            assert scheduler.student_grid_cfg is student_grid_cfg

    assert_scheduler_contract()
    network.reset_parameters()
    assert_scheduler_contract()
    with pytest.raises(NotImplementedError, match="shared schedule"):
        network._set_noise_schedule(min_t=0.25)
    assert_scheduler_contract()


def test_wan_moe_empty_batch_attention_stays_connected():
    """Batch-0 native attention returns a query-shaped result whose autograd graph reaches Q, K and V.

    FSDP2 fires a parameter group's reduce-scatter from a node on the group's module inputs, so an
    expert that received no samples on a rank must still connect its attention output to the Q/K/V
    projections; a detached zeros tensor would reorder that rank's collectives against the others.
    On CUDA the bf16 batch-0 call is the case where cuDNN SDPA returns ``None``. The test checks the
    contract of the patched path (output connected to Q, K and V on every backend); it can only tell the
    old detached substitution apart on a stack whose default batch-0 SDPA call returns ``None`` (H100 and
    GB200 with the pinned PyTorch), which is exactly the condition under which that bug exists.
    """
    from diffusers.models import attention_dispatch

    from fastgen.networks.Wan.wan_moe import _patch_diffusers_native_attention_for_empty_batch

    _patch_diffusers_native_attention_for_empty_batch()
    _patch_diffusers_native_attention_for_empty_batch()  # idempotent
    native_backend = attention_dispatch.AttentionBackendName.NATIVE
    native = attention_dispatch._AttentionBackendRegistry._backends[native_backend]
    assert native is attention_dispatch._native_attention
    assert getattr(native, "_fastgen_empty_batch_safe", False)

    cases = [(torch.device("cpu"), torch.float32)]
    if torch.cuda.is_available():
        cases.append((torch.device("cuda"), torch.bfloat16))
    seq_q, seq_kv, heads, head_dim = 6, 9, 2, 8
    for device, dtype in cases:
        query = torch.randn(0, seq_q, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
        key = torch.randn(0, seq_kv, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
        value = torch.randn(0, seq_kv, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
        out = attention_dispatch.dispatch_attention_fn(query, key, value, backend=native_backend)
        assert out.shape == query.shape and out.dtype == query.dtype
        assert out.grad_fn is not None, f"{device}: empty attention output is detached from the autograd graph"
        out.sum().backward()
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            assert tensor.grad is not None and tensor.grad.shape == tensor.shape, f"{device}: {name} got no gradient"

        # Populated inputs are untouched: identical to PyTorch's SDPA in the [B, S, heads, head_dim] layout.
        query = torch.randn(2, seq_q, heads, head_dim, device=device, dtype=dtype)
        key = torch.randn(2, seq_kv, heads, head_dim, device=device, dtype=dtype)
        value = torch.randn(2, seq_kv, heads, head_dim, device=device, dtype=dtype)
        out = attention_dispatch.dispatch_attention_fn(query, key, value, backend=native_backend)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
        torch.testing.assert_close(out, expected)


def test_wan_moe_routing_feeds_each_expert_only_its_own_samples():
    """Batch-split routing: each expert sees only the samples (or frames) it owns, the merge is a plain
    per-position selection with gradients confined to the selected expert, and a non-finite value from
    the responsible expert surfaces in the output instead of being masked."""
    from fastgen.networks.Wan.wan_moe import WanMoE

    class RecordingExpert(torch.nn.Module):
        def __init__(self, scale: float):
            super().__init__()
            self.transformer = torch.nn.Identity()
            self.transformer.blocks = torch.nn.ModuleList()
            self.scale = torch.nn.Parameter(torch.tensor(scale))
            self.seen_t = []
            self.seen_condition_batch = []
            self.poison = None  # (sample, frame) of the returned output to overwrite with NaN

        def forward(self, x_t, t, condition=None, **kwargs):
            self.seen_t.append(t.detach().clone())
            self.seen_condition_batch.append(None if condition is None else condition.shape[0])
            out = x_t * self.scale
            if self.poison is not None and out.shape[0] > 0:
                out = out.clone()
                out[self.poison[0], :, self.poison[1]] = float("nan")
            return out

    def make_moe(boundary=0.9):
        network = object.__new__(WanMoE)
        torch.nn.Module.__init__(network)
        network.net_pred_type = "flow"
        network.guidance_boundary = boundary
        network.enable_single_expert_shortcut = False
        network.cond_keys_no_routing = set()
        network.network_high = RecordingExpert(2.0)
        network.network_low = RecordingExpert(3.0)
        network.noise_scheduler = RFNoiseSchedule()
        return network

    batch, frames = 4, 2
    x_t = torch.randn(batch, 16, frames, 4, 4)
    condition = torch.randn(batch, 5, 8)

    # Per-element timesteps: samples 0 and 2 are at or above the boundary, 1 and 3 below.
    network = make_moe()
    t = torch.tensor([0.95, 0.3, 0.9, 0.1], dtype=network.noise_scheduler.t_precision)
    out = network(x_t, t, condition=condition)
    (t_high,), (t_low,) = network.network_high.seen_t, network.network_low.seen_t
    assert torch.equal(t_high, t[[0, 2]]) and torch.equal(t_low, t[[1, 3]])
    assert network.network_high.seen_condition_batch == [2] and network.network_low.seen_condition_batch == [2]
    high_mask = (t >= 0.9).view(-1, 1, 1, 1, 1)
    torch.testing.assert_close(out, torch.where(high_mask, x_t * 2.0, x_t * 3.0))
    out.sum().backward()
    torch.testing.assert_close(network.network_high.scale.grad, x_t[[0, 2]].sum())
    torch.testing.assert_close(network.network_low.scale.grad, x_t[[1, 3]].sum())

    # Every sample below the boundary: the high expert runs on an empty batch and contributes nothing.
    network = make_moe()
    t_low_only = torch.full((batch,), 0.5, dtype=network.noise_scheduler.t_precision)
    out = network(x_t, t_low_only, condition=condition)
    assert network.network_high.seen_t[0].numel() == 0 and network.network_high.seen_condition_batch == [0]
    torch.testing.assert_close(out, x_t * 3.0)

    # Per-frame timesteps: sample 0 straddles the boundary and is sent to both experts (all frames), sample 1
    # only to low. Frames are merged from the responsible expert and the masked-out frames get zero gradient.
    network = make_moe()
    t_frames = torch.tensor([[0.95, 0.3], [0.1, 0.2]], dtype=network.noise_scheduler.t_precision)
    x_two = x_t[:2]
    out = network(x_two, t_frames, condition=condition[:2])
    assert torch.equal(network.network_high.seen_t[0], t_frames[[0]])
    assert torch.equal(network.network_low.seen_t[0], t_frames[[0, 1]])
    frame_mask = (t_frames >= 0.9).view(2, 1, frames, 1, 1)
    torch.testing.assert_close(out, torch.where(frame_mask, x_two * 2.0, x_two * 3.0))
    out.sum().backward()
    torch.testing.assert_close(network.network_high.scale.grad, x_two[0, :, 0].sum())
    torch.testing.assert_close(network.network_low.scale.grad, x_two[0, :, 1].sum() + x_two[1].sum())

    # A non-finite value on a frame the expert does not own never reaches the merged output.
    network = make_moe()
    network.network_high.poison = (0, 1)  # high evaluates sample 0 on both frames but owns only frame 0
    out = network(x_two, t_frames, condition=condition[:2])
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.where(frame_mask, x_two * 2.0, x_two * 3.0))

    # A non-finite value produced by the responsible expert is not masked.
    network = make_moe()
    network.network_low.poison = (0, 1)  # low owns samples 1 and 3; local index 0 is sample 1, frame 1
    out = network(x_t, t, condition=condition)
    expected_nan = torch.zeros_like(out, dtype=torch.bool)
    expected_nan[1, :, 1] = True
    assert torch.equal(torch.isnan(out), expected_nan)


def test_wan_moe_uses_shallower_expert_for_shared_layer_index_range():
    from fastgen.networks.Wan.wan_moe import WanMoE

    class Expert(torch.nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(depth))

        def forward(self, x_t, _t, **_kwargs):
            return x_t

    network = object.__new__(WanMoE)
    torch.nn.Module.__init__(network)
    network.net_pred_type = "flow"
    network.guidance_boundary = 0.5
    network.enable_single_expert_shortcut = False
    network.cond_keys_no_routing = set()
    network.network_high = Expert(depth=3)
    network.network_low = Expert(depth=2)
    network.noise_scheduler = RFNoiseSchedule()

    x_t = torch.randn(2, 1, 1, 1, 1)
    t = torch.tensor([0.75, 0.25], dtype=network.noise_scheduler.t_precision)
    torch.testing.assert_close(network(x_t, t, feature_indices={1}), x_t)

    with pytest.raises(ValueError, match=r"out-of-range indices \[2\].*valid range is \[0, 2\)"):
        network(x_t, t, feature_indices={2})


@pytest.mark.parametrize("shortcut_source", ["explicit", "inference"])
@pytest.mark.parametrize(
    ("sharded_expert", "timestep", "expected_high_batch", "expected_low_batch"),
    [
        ("low", 1.0, 2, 0),
        ("high", 0.0, 0, 2),
    ],
)
def test_wan_moe_disables_single_expert_shortcut_when_either_transformer_is_fsdp_sharded(
    monkeypatch,
    shortcut_source,
    sharded_expert,
    timestep,
    expected_high_batch,
    expected_low_batch,
):
    """A live FSDP transformer root forces both expert forwards for either shortcut source."""
    import fastgen.networks.Wan.wan_moe as wan_moe_module
    from fastgen.networks.Wan.wan_moe import WanMoE

    class FakeFSDPModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()

    class RecordingExpert(torch.nn.Module):
        def __init__(self, offset: float):
            super().__init__()
            self.transformer = torch.nn.Identity()
            self.transformer.blocks = torch.nn.ModuleList()
            self.offset = offset
            self.batch_sizes = []

        def forward(self, x_t, _t, **_kwargs):
            self.batch_sizes.append(x_t.shape[0])
            return x_t + self.offset

    def make_moe():
        network = object.__new__(WanMoE)
        torch.nn.Module.__init__(network)
        network.net_pred_type = "flow"
        network.guidance_boundary = 0.5
        network.enable_single_expert_shortcut = shortcut_source == "explicit"
        network.cond_keys_no_routing = set()
        network.network_high = RecordingExpert(1.0)
        network.network_low = RecordingExpert(2.0)
        network.noise_scheduler = RFNoiseSchedule()
        return network

    monkeypatch.setattr(wan_moe_module, "FSDPModule", FakeFSDPModule)
    x_t = torch.zeros(2, 1, 1, 1, 1)
    shortcut_context = torch.inference_mode() if shortcut_source == "inference" else contextlib.nullcontext()

    # Preserve the fast path for two unsharded expert roots.
    network = make_moe()
    t = torch.full((2,), timestep, dtype=network.noise_scheduler.t_precision)
    with shortcut_context:
        output = network(x_t, t)
    selected = network.network_high if expected_high_batch else network.network_low
    skipped = network.network_low if expected_high_batch else network.network_high
    assert selected.batch_sizes == [2]
    assert skipped.batch_sizes == []
    torch.testing.assert_close(output, x_t + selected.offset)

    # Sharding either root, including the locally skipped expert, forces the
    # collective-safe path where both expert forwards execute.
    network = make_moe()
    getattr(network, f"network_{sharded_expert}").transformer = FakeFSDPModule()
    with shortcut_context:
        output = network(x_t, t)
    assert network.network_high.batch_sizes == [expected_high_batch]
    assert network.network_low.batch_sizes == [expected_low_batch]
    selected = network.network_high if expected_high_batch else network.network_low
    torch.testing.assert_close(output, x_t + selected.offset)


def _wan_moe_fsdp_rank_dependent_route_impl(rank: int, world_size: int, checkpoint_dir: str) -> dict:
    """Run unequal routes through real FSDP2 hooks and checkpoint their optimizer state."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, fully_shard

    from fastgen.configs.config import BaseCheckpointerConfig
    from fastgen.networks.Wan.wan_moe import WanMoE
    from fastgen.utils.checkpointer import Checkpointer

    device = torch.device("cuda", rank)

    class RecordingTransformer(torch.nn.Module):
        def __init__(self, scale: float):
            super().__init__()
            self.blocks = torch.nn.ModuleList()
            self.proj = torch.nn.Linear(2, 2, bias=False, device=device)
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(2, device=device) * scale)
            self.batch_sizes = []

        def forward(self, x_t):
            self.batch_sizes.append(x_t.shape[0])
            return self.proj(x_t)

    class TinyExpert(torch.nn.Module):
        def __init__(self, scale: float):
            super().__init__()
            self.transformer = RecordingTransformer(scale)

        def forward(self, x_t, _t, **_kwargs):
            return self.transformer(x_t)

    network = object.__new__(WanMoE)
    torch.nn.Module.__init__(network)
    network.net_pred_type = "flow"
    network.guidance_boundary = 0.5
    network.enable_single_expert_shortcut = True
    network.cond_keys_no_routing = set()
    network.network_high = TinyExpert(2.0)
    network.network_low = TinyExpert(3.0)
    network.noise_scheduler = RFNoiseSchedule()
    t_precision = network.noise_scheduler.t_precision

    mesh = init_device_mesh(
        "cuda",
        (1, world_size),
        mesh_dim_names=("replicate", "shard"),
    )
    fully_shard(network.network_high.transformer, mesh=mesh, reshard_after_forward=True)
    fully_shard(network.network_low.transformer, mesh=mesh, reshard_after_forward=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0)

    x_t = torch.ones(1, 1, 1, 1, 2, device=device)
    rank_local_t = torch.tensor([1.0 if rank == 0 else 0.0], device=device, dtype=t_precision)
    rank_local_output = network(x_t, rank_local_t)
    rank_local_output.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    rank_local_calls = (
        network.network_high.transformer.batch_sizes[-1],
        network.network_low.transformer.batch_sizes[-1],
    )

    optimizer_state_parameter_ids = {id(parameter) for parameter in optimizer.state}
    optimizer_state_keys = frozenset(
        name for name, parameter in network.named_parameters() if id(parameter) in optimizer_state_parameter_ids
    )
    gathered_optimizer_state_keys = [None] * world_size
    dist.all_gather_object(gathered_optimizer_state_keys, optimizer_state_keys)
    expected_optimizer_state_keys = frozenset(
        {
            "network_high.transformer.proj.weight",
            "network_low.transformer.proj.weight",
        }
    )
    if gathered_optimizer_state_keys != [expected_optimizer_state_keys] * world_size:
        raise AssertionError(f"optimizer state coverage diverged: {gathered_optimizer_state_keys}")

    checkpoint_config = BaseCheckpointerConfig(save_dir=checkpoint_dir)
    checkpoint_path = Checkpointer(checkpoint_config, save_sharded=True).save(
        model_dict=torch.nn.ModuleDict({"net": network}),
        optimizer_dict={"net": optimizer},
        iteration=1,
    )

    # Identical high-noise routing is safe and still enters both roots so their
    # FSDP hook sequence remains symmetric (the low expert receives batch 0).
    with torch.no_grad():
        high_output = network(x_t, torch.ones(1, device=device, dtype=t_precision))
    high_calls = (
        network.network_high.transformer.batch_sizes[-1],
        network.network_low.transformer.batch_sizes[-1],
    )

    # Strict inference mode cannot re-enable the one-expert shortcut for a
    # sharded root. Exercise the symmetric low-noise route after resharding.
    network.enable_single_expert_shortcut = False
    with torch.inference_mode():
        low_output = network(x_t, torch.zeros(1, device=device, dtype=t_precision))
    low_calls = (
        network.network_high.transformer.batch_sizes[-1],
        network.network_low.transformer.batch_sizes[-1],
    )

    local_result = {
        "rank": rank,
        "roots_are_fsdp": isinstance(network.network_high.transformer, FSDPModule)
        and isinstance(network.network_low.transformer, FSDPModule),
        "rank_local_calls": rank_local_calls,
        "high_calls": high_calls,
        "low_calls": low_calls,
        "rank_local_output_correct": torch.equal(rank_local_output, x_t * (2.0 if rank == 0 else 3.0)),
        "high_output_correct": torch.equal(high_output, x_t * 2.0),
        "low_output_correct": torch.equal(low_output, x_t * 3.0),
        "optimizer_state_keys": sorted(optimizer_state_keys),
        "checkpoint_path": checkpoint_path,
        "checkpoint_marker_exists": os.path.isfile(checkpoint_path),
        "optimizer_metadata_exists": os.path.isfile(
            os.path.join(f"{checkpoint_path.removesuffix('.pth')}.net_optim", ".metadata")
        ),
    }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_result)
    return {"per_rank": gathered}


@RunIf(min_gpus=2)
def test_wan_moe_fsdp_rank_dependent_routes_persist_complete_optimizer_state(tmp_path):
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_wan_moe_fsdp_rank_dependent_route_impl,
        world_size=2,
        timeout=120,
        setup_fn=set_env_vars,
        checkpoint_dir=str(tmp_path),
    )
    expected_optimizer_state_keys = [
        "network_high.transformer.proj.weight",
        "network_low.transformer.proj.weight",
    ]
    assert [entry["rank"] for entry in result["per_rank"]] == [0, 1]
    for entry in result["per_rank"]:
        assert entry["roots_are_fsdp"] is True
        assert entry["rank_local_calls"] == ((1, 0) if entry["rank"] == 0 else (0, 1))
        assert entry["high_calls"] == (1, 0)
        assert entry["low_calls"] == (0, 1)
        assert entry["rank_local_output_correct"] is True
        assert entry["high_output_correct"] is True
        assert entry["low_output_correct"] is True
        assert entry["optimizer_state_keys"] == expected_optimizer_state_keys
        assert entry["checkpoint_marker_exists"] is True
        assert entry["optimizer_metadata_exists"] is True
    assert len({entry["checkpoint_path"] for entry in result["per_rank"]}) == 1
    clear_gpu_memory()


def test_wan_set_precision_keeps_only_the_rotary_tables_fp32():
    """Storage follows ``precision_opt``, Diffusers' declared FP32 tensors included, while the RoPE tables stay FP32."""
    from fastgen.networks.Wan.network import Wan

    class TinyTransformer(torch.nn.Module):
        _keep_in_fp32_modules = ["time_embedder", "scale_shift_table"]

        def __init__(self):
            super().__init__()
            self.time_embedder = torch.nn.Linear(2, 2)
            self.proj = torch.nn.Linear(2, 2)
            self.scale_shift_table = torch.nn.Parameter(torch.ones(2))
            self.rope = torch.nn.Module()
            self.rope.register_buffer("freqs_cos", torch.ones(2))

    network = object.__new__(Wan)
    torch.nn.Module.__init__(network)
    network.transformer = TinyTransformer()
    network._keep_original_prec = True

    network._set_precision(precision_model=torch.bfloat16, precision_opt=torch.float64)

    assert network.transformer.proj.weight.dtype == torch.float64
    assert network.transformer.time_embedder.weight.dtype == torch.float64
    assert network.transformer.scale_shift_table.dtype == torch.float64
    assert network.transformer.rope.freqs_cos.dtype == torch.float32

    # Plain ``to`` remains a native uniform cast that does not redefine the policy.
    network.to(dtype=torch.bfloat16)
    assert network.transformer.proj.weight.dtype == torch.bfloat16
    assert network.transformer.time_embedder.weight.dtype == torch.bfloat16
    assert network.transformer.scale_shift_table.dtype == torch.bfloat16
    assert network.transformer.rope.freqs_cos.dtype == torch.bfloat16

    network._keep_original_prec = False
    network._set_precision(precision_model=torch.bfloat16, precision_opt=torch.float64)
    assert network.transformer.rope.freqs_cos.dtype == torch.float64


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("fastgen.networks.Wan.network", "Wan"),
        ("fastgen.networks.Wan.network_causal", "CausalWan"),
    ],
    ids=["base", "causal"],
)
def test_wan_t2v_condition_boundary_requires_a_tensor(module_name, class_name):
    network_cls = getattr(importlib.import_module(module_name), class_name)
    captured = {}

    class Transformer(torch.nn.Module):
        blocks = ()

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()
            self.config = SimpleNamespace(patch_size=(1, 1, 1))

        def forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
            del timestep, kwargs
            captured["encoder_hidden_states"] = encoder_hidden_states
            return torch.zeros_like(hidden_states)

    network = network_cls.__new__(network_cls)
    torch.nn.Module.__init__(network)
    network.expand_timesteps = False
    network.net_pred_type = "flow"
    network._record_precision(precision_model=torch.bfloat16)
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer = Transformer()

    state = torch.randn(1, 2, 2, 2, 2, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
    condition = torch.randn(1, 2, 3, dtype=network.precision_model)
    output = network(state, timestep, condition=condition)

    assert output.dtype == state.dtype
    assert captured["encoder_hidden_states"].dtype == state.dtype
    with pytest.raises(AssertionError, match="condition must be a tensor"):
        network(state, timestep, condition=[condition])
    with pytest.raises(AssertionError, match=r"condition\.dtype"):
        network(state, timestep, condition=condition.float())


def _tiny_vace_transformer(seed: int = 0, attention_head_dim: int = 8):
    """Two-block VACE transformer (one VACE block) small enough to run on CPU."""
    from diffusers.models.transformers.transformer_wan_vace import WanVACETransformer3DModel

    torch.manual_seed(seed)
    return WanVACETransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=attention_head_dim,
        in_channels=4,
        out_channels=4,
        text_dim=8,
        freq_dim=16,
        ffn_dim=16,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=32,
        vace_layers=[0],
        vace_in_channels=8,
    ).eval()


def _reference_recursive_set_attn_processor(module: torch.nn.Module, processor) -> None:
    """Recursive assignment FastGen used to pin itself before relying on diffusers' ``AttentionMixin``."""

    def visit(name: str, child: torch.nn.Module) -> None:
        if hasattr(child, "set_processor"):
            child.set_processor(processor)
        for sub_name, grandchild in child.named_children():
            visit(f"{name}.{sub_name}", grandchild)

    for name, child in module.named_children():
        visit(name, child)


def _attention_processors(module: torch.nn.Module) -> dict:
    return {name: child.processor for name, child in module.named_modules() if hasattr(child, "set_processor")}


def test_vace_transformer_attention_mixin_matches_reference_recursive_assignment():
    """The causal VACE network installs its processor through diffusers' ``AttentionMixin``; that walk must
    reach exactly the modules of the reference recursion (main and VACE blocks, self- and cross-attention),
    share one instance, and leave the forward bitwise identical."""
    from diffusers.models.attention import AttentionMixin

    from fastgen.networks.Wan.network_causal import CausalWanAttnProcessor

    via_mixin = _tiny_vace_transformer(attention_head_dim=_FLEX_ATTENTION_HEAD_DIM)
    via_reference = copy.deepcopy(via_mixin)
    assert isinstance(via_mixin, AttentionMixin)

    mixin_processor, reference_processor = CausalWanAttnProcessor(), CausalWanAttnProcessor()
    via_mixin.set_attn_processor(mixin_processor)
    _reference_recursive_set_attn_processor(via_reference, reference_processor)

    installed, expected = _attention_processors(via_mixin), _attention_processors(via_reference)
    assert set(installed) == set(expected)
    assert {name.split(".")[0] for name in installed} == {"blocks", "vace_blocks"}
    assert {name.rsplit(".", 1)[1] for name in installed} == {"attn1", "attn2"}
    assert all(processor is mixin_processor for processor in installed.values())
    assert all(processor is reference_processor for processor in expected.values())

    device = _flex_attention_device()
    via_mixin.to(device)
    via_reference.to(device)
    torch.manual_seed(1)
    inputs = dict(
        hidden_states=torch.randn(1, 4, 2, 4, 4, device=device),
        timestep=torch.tensor([500], device=device),
        encoder_hidden_states=torch.randn(1, 3, 8, device=device),
        control_hidden_states=torch.randn(1, 8, 2, 4, 4, device=device),
        control_hidden_states_scale=torch.ones(1, device=device),
        return_dict=False,
    )
    with torch.no_grad():
        out_mixin = via_mixin(**inputs)[0]
        out_reference = via_reference(**inputs)[0]
    assert torch.isfinite(out_mixin).all()
    assert torch.equal(out_mixin, out_reference)


def test_causal_vace_override_installs_one_shared_causal_processor_on_every_attention():
    """The production override path leaves no default processor behind in the main or VACE blocks."""
    from fastgen.networks.VaceWan.network_causal import CausalVACEWan
    from fastgen.networks.Wan.network_causal import CausalWanAttnProcessor

    network = CausalVACEWan.__new__(CausalVACEWan)
    torch.nn.Module.__init__(network)
    network.transformer = _tiny_vace_transformer()
    network._keep_original_prec = False
    network._sdpa_no_cudnn_blocks = None
    network.transformer.r_embedder = None
    network.transformer.encoder_depth = None

    network.override_transformer_forward()

    processors = _attention_processors(network.transformer)
    assert {name.split(".")[0] for name in processors} == {"blocks", "vace_blocks"}
    assert {type(processor) for processor in processors.values()} == {CausalWanAttnProcessor}
    assert len({id(processor) for processor in processors.values()}) == 1


@pytest.mark.parametrize(("frame_offset", "control_frames"), [(1, 3), (0, 1), (1, 2), (2, 2)])
def test_causal_vace_embeds_the_offset_control_window_and_raises_past_its_end(frame_offset, control_frames):
    """A causal pass embeds control frames ``[frame_offset, frame_offset + T)`` and raises if the control ends sooner."""
    from fastgen.networks.kv_cache import KVCacheEngine
    from fastgen.networks.VaceWan.network_causal import vace_causal_classify_forward_prepare

    transformer = _tiny_vace_transformer()
    transformer.r_embedder = None
    hidden_states = torch.randn(1, 4, 2, 4, 4)
    control = torch.randn(1, 8, control_frames, 4, 4)
    window = control[:, :, frame_offset : frame_offset + hidden_states.shape[2]]

    def prepare():
        ctx = CausalContext(
            frame_offset=frame_offset, is_ar=True, cache_engines={"main": KVCacheEngine(), "vace": KVCacheEngine()}
        )
        return vace_causal_classify_forward_prepare(
            transformer,
            hidden_states,
            torch.tensor([400.0]),
            torch.randn(1, 3, 8),
            control_hidden_states=control,
            causal_ctx=ctx,
        )

    with torch.no_grad():
        if window.shape[2] < hidden_states.shape[2]:
            with pytest.raises(ValueError, match="VACE control has"):
                prepare()
            return
        torch.testing.assert_close(prepare()[6], transformer.vace_patch_embedding(window).flatten(2).transpose(1, 2))


@pytest.mark.parametrize("control_frames", [1, 3])
def test_vace_prepare_raises_when_control_frames_differ_from_the_video(control_frames):
    from fastgen.networks.VaceWan.network import vace_classify_forward_prepare

    with pytest.raises(ValueError, match="VACE control has"):
        vace_classify_forward_prepare(
            _tiny_vace_transformer(),
            torch.randn(1, 4, 2, 4, 4),
            torch.tensor([400.0]),
            torch.randn(1, 3, 8),
            control_hidden_states=torch.randn(1, 8, control_frames, 4, 4),
        )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("precision_model", "state_dtype"),
    [(torch.bfloat16, torch.float32), (None, torch.float64)],
)
def test_vace_control_context_uses_actual_model_input_dtype(causal, precision_model, state_dtype):
    from fastgen.networks.VaceWan.network import VACEWan
    from fastgen.networks.VaceWan.network_causal import CausalVACEWan

    captured = {}

    class Transformer(torch.nn.Module):
        config = SimpleNamespace(vace_layers=(0,), patch_size=(1, 1, 1))
        blocks = ()

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()

        def forward(self, *args, **kwargs):
            if args:
                captured.update(hidden_states=args[0], encoder_hidden_states=args[2], **kwargs)
                return args[0]
            captured.update(kwargs)
            return kwargs["hidden_states"]

    network_cls = CausalVACEWan if causal else VACEWan
    network = network_cls.__new__(network_cls)
    torch.nn.Module.__init__(network)
    network.transformer = Transformer()
    network.noise_scheduler = RFNoiseSchedule()
    network.net_pred_type = "flow"
    network.context_scale = 1.0
    network._keep_original_prec = False
    network._record_precision(precision_model=precision_model)

    state = torch.randn(1, 2, 2, 2, 2, dtype=state_dtype)
    condition = {
        "text_embeds": torch.randn(1, 3, 4, dtype=precision_model or state_dtype),
        "vid_context": torch.randn(1, 6, 2, 2, 2, dtype=precision_model or state_dtype),
    }
    timestep = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
    output = network(state, timestep, condition=condition)

    assert output.dtype == state.dtype
    model_input_dtype = captured["hidden_states"].dtype
    assert model_input_dtype == (precision_model or state_dtype)
    assert captured["encoder_hidden_states"].dtype == model_input_dtype
    assert captured["control_hidden_states"].dtype == model_input_dtype
    assert captured["control_hidden_states_scale"].dtype == model_input_dtype

    # The control context only reaches the backbone, so it must already be in ``precision_model``; a network without
    # one casts and checks nothing.
    context_dtype = torch.float16 if state_dtype != torch.float16 else torch.float32
    mixed_dtype = {**condition, "vid_context": condition["vid_context"].to(context_dtype)}
    if precision_model is None:
        network(state, timestep, condition=mixed_dtype)
    else:
        with pytest.raises(AssertionError, match=r"vid_context\.dtype"):
            network(state, timestep, condition=mixed_dtype)


@pytest.mark.parametrize("causal", [False, True])
def test_vace_skipped_block_keeps_its_control_hint_and_feature(causal):
    """A skipped VACE layer still adds its hint and records its feature, so later hints and features stay aligned."""
    from fastgen.networks.VaceWan.network import vace_classify_forward_block_forward
    from fastgen.networks.VaceWan.network_causal import vace_causal_classify_forward_block_forward

    calls = []

    class Block(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, hidden_states, *args, **kwargs):
            del args, kwargs
            calls.append((self.index, hidden_states.clone()))
            return hidden_states

    class VaceBlock(torch.nn.Module):
        def __init__(self, hint):
            super().__init__()
            self.hint = hint

        def forward(self, hidden_states, encoder_hidden_states, control_hidden_states, *args, **kwargs):
            del encoder_hidden_states, args, kwargs
            return torch.full_like(hidden_states, self.hint), control_hidden_states

    transformer = SimpleNamespace(
        blocks=torch.nn.ModuleList([Block(index) for index in range(3)]),
        vace_blocks=torch.nn.ModuleList([VaceBlock(10.0**index) for index in range(3)]),
        config=SimpleNamespace(vace_layers=(0, 1, 2)),
        gradient_checkpointing=False,
        encoder_depth=None,
        norm_temb=False,
    )
    loop = vace_causal_classify_forward_block_forward if causal else vace_classify_forward_block_forward
    hidden = torch.zeros(1, 4, 2)
    inputs = dict(
        timestep_proj=torch.zeros(1, 6, 2),
        r_timestep_proj=None,
        encoder_hidden_states=torch.zeros(1, 3, 2),
        rotary_emb=torch.zeros(1, 4, 2),
        control_hidden_states=torch.zeros(1, 4, 2),
        control_hidden_states_scale=[torch.ones(()) for _ in range(3)],
        skip_layers=[1],
    )
    output, features = loop(transformer, hidden, **inputs, feature_indices={1, 2})

    # Block 1 is skipped but its hint (10) is still applied, so block 2 sees 1 + 10 and the output is 1 + 10 + 100.
    assert [index for index, _ in calls] == [0, 2]
    assert torch.equal(calls[1][1], torch.full_like(hidden, 11.0))
    assert torch.equal(output, torch.full_like(hidden, 111.0))
    assert len(features) == 2
    assert torch.equal(features[0], torch.full_like(hidden, 11.0))
    assert torch.equal(features[1], torch.full_like(hidden, 111.0))

    # The skipped block's feature completes the request, so the early return stops before block 2.
    calls.clear()
    _, features = loop(transformer, hidden, **inputs, feature_indices={1}, return_features_early=True)
    assert [index for index, _ in calls] == [0]
    assert len(features) == 1
    assert torch.equal(features[0], torch.full_like(hidden, 11.0))


@pytest.mark.parametrize("variant", ["wan", "causal-wan", "wan-camera"])
def test_wan_skipped_block_still_records_its_feature(variant):
    """A skipped block acts as the identity, but every requested index, skipped or not, yields one feature."""
    from fastgen.networks.Wan import network as wan_network
    from fastgen.networks.Wan import network_causal as causal_wan_network
    from fastgen.networks.WanI2V import network_cam

    loop = {
        "wan": wan_network.classify_forward_block_forward,
        "causal-wan": causal_wan_network.classify_forward_block_forward,
        "wan-camera": network_cam._wancami2v_classify_forward_block_forward,
    }[variant]
    calls = []

    class Block(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, hidden_states, *args, **kwargs):
            del args, kwargs
            calls.append(self.index)
            return hidden_states + 10.0**self.index

    transformer = SimpleNamespace(
        blocks=torch.nn.ModuleList([Block(index) for index in range(3)]),
        gradient_checkpointing=False,
        encoder_depth=None,
        norm_temb=False,
    )
    hidden = torch.zeros(1, 4, 2)
    inputs = dict(
        timestep_proj=torch.zeros(1, 6, 2),
        encoder_hidden_states=torch.zeros(1, 3, 2),
        rotary_emb=torch.zeros(1, 4, 2),
        skip_layers=[1],
    )
    output, features = loop(transformer, hidden, **inputs, feature_indices={1, 2})

    # Block 1 is skipped, so its feature is block 0's output (1), and block 2 then adds 100.
    assert calls == [0, 2]
    assert torch.equal(output, torch.full_like(hidden, 101.0))
    assert len(features) == 2
    assert torch.equal(features[0], torch.full_like(hidden, 1.0))
    assert torch.equal(features[1], torch.full_like(hidden, 101.0))

    # The skipped block's feature completes the request, so the early return stops before block 2.
    calls.clear()
    _, features = loop(transformer, hidden, **inputs, feature_indices={1}, return_features_early=True)
    assert calls == [0]
    assert len(features) == 1
    assert torch.equal(features[0], torch.full_like(hidden, 1.0))


def test_wan_i2v_concat_uses_pre_aligned_first_frame_condition():
    from fastgen.networks.WanI2V.network import WanI2V

    state = torch.randn(1, 2, 2, 2, 2, dtype=torch.bfloat16)
    clean = torch.randn_like(state, dtype=torch.float32).to(dtype=state.dtype)

    network = WanI2V.__new__(WanI2V)
    torch.nn.Module.__init__(network)
    network.concat_mask = True
    network.latent_compression_factors = (4, 8, 8)
    network.expand_timesteps = False
    network.noise_scheduler = RFNoiseSchedule()
    model_input, _, _ = network._compute_i2v_inputs(
        state,
        first_frame_cond=clean,
        timestep=torch.tensor([0.5], dtype=network.noise_scheduler.t_precision),
    )
    assert model_input.dtype == state.dtype


def test_wan_i2v_non_concat_uses_pre_aligned_first_frame_condition():
    from fastgen.networks.WanI2V.network import WanI2V

    network = WanI2V.__new__(WanI2V)
    torch.nn.Module.__init__(network)
    network.concat_mask = False
    network.expand_timesteps = False
    network.noise_scheduler = RFNoiseSchedule()

    state = torch.randn(1, 2, 2, 2, 2, dtype=torch.bfloat16)
    clean = torch.randn(1, 2, 1, 2, 2, dtype=torch.float32).to(dtype=state.dtype)
    original_replace = network._replace_first_frame
    replacement_inputs = []

    def replace_first_frame(first_frame_cond, latents, return_mask=False):
        replacement_inputs.append(first_frame_cond)
        return original_replace(first_frame_cond, latents, return_mask=return_mask)

    network._replace_first_frame = replace_first_frame
    model_input, _, _ = network._compute_i2v_inputs(
        state,
        first_frame_cond=clean,
        timestep=torch.tensor([0.5], dtype=network.noise_scheduler.t_precision),
    )

    assert len(replacement_inputs) == 1
    assert replacement_inputs[0] is clean
    assert model_input.dtype == state.dtype
    assert torch.equal(model_input[:, :, :1], clean)
    assert torch.equal(model_input[:, :, 1:], state[:, :, 1:])


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("fastgen.networks.WanI2V.network", "WanI2V"),
        ("fastgen.networks.WanI2V.network_cam", "WanCamI2V"),
    ],
    ids=["base", "camera"],
)
@pytest.mark.parametrize("concat_mask", [False, True], ids=["replace", "concat"])
def test_wan_i2v_forward_variants_check_conditioning_dtypes(module_name, class_name, concat_mask):
    network_cls = getattr(importlib.import_module(module_name), class_name)
    captured = {}

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()

        def forward(self, hidden_states, encoder_hidden_states, **kwargs):
            captured["hidden_states"] = hidden_states
            captured["encoder_hidden_states"] = encoder_hidden_states
            captured["encoder_hidden_states_image"] = kwargs["encoder_hidden_states_image"]
            return torch.zeros_like(hidden_states[:, :2])

    network = network_cls.__new__(network_cls)
    torch.nn.Module.__init__(network)
    network.concat_mask = concat_mask
    network.latent_compression_factors = (4, 8, 8)
    network.expand_timesteps = False
    network.net_pred_type = "flow"
    network._record_precision(precision_model=torch.bfloat16)
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer = Transformer()

    state = torch.randn(1, 2, 2, 2, 2, dtype=torch.float32)
    cond_frames = state.shape[2] if concat_mask else 1
    raw_clean = torch.randn(1, 2, cond_frames, 2, 2, dtype=state.dtype)
    aligned_conditions = []
    replacement_inputs = []
    original_compute = network._compute_i2v_inputs
    original_replace = network._replace_first_frame

    def compute_i2v_inputs(*args, **kwargs):
        aligned_conditions.append(kwargs["first_frame_cond"])
        return original_compute(*args, **kwargs)

    def replace_first_frame(first_frame_cond, latents, return_mask=False):
        replacement_inputs.append((first_frame_cond, latents))
        return original_replace(first_frame_cond, latents, return_mask=return_mask)

    network._compute_i2v_inputs = compute_i2v_inputs
    network._replace_first_frame = replace_first_frame
    timestep = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
    condition = {
        "text_embeds": torch.randn(1, 2, 3, dtype=network.precision_model),
        "first_frame_cond": raw_clean,
        "encoder_hidden_states_image": torch.randn(1, 2, 3, dtype=network.precision_model),
    }
    output = network(state, timestep, condition=condition)

    assert output.dtype == state.dtype
    assert len(aligned_conditions) == 1
    assert aligned_conditions[0] is raw_clean
    assert aligned_conditions[0].dtype == state.dtype
    if concat_mask:
        assert replacement_inputs == []
    else:
        # The input replacement uses the clean latent itself; the output pin is that latent expressed in
        # the flow domain against the raw input, so converting the output back to x0 recovers it.
        # The input is replaced in the state dtype, the output in the output transform's high-precision dtype.
        assert len(replacement_inputs) == 2
        assert replacement_inputs[0][0] is aligned_conditions[0]
        assert replacement_inputs[0][1].dtype == state.dtype
        assert replacement_inputs[1][1].dtype == high_precision_dtype(state)
        expected_pin = network.noise_scheduler.convert_model_output(
            state[:, :, :1],
            raw_clean[:, :, :1],
            timestep,
            src_pred_type="x0",
            target_pred_type="flow",
            output_dtype=high_precision_dtype(state),
        )
        assert torch.equal(replacement_inputs[1][0], expected_pin)
        x0_pred = network.noise_scheduler.convert_model_output(
            state, output, timestep, src_pred_type="flow", target_pred_type="x0"
        )
        torch.testing.assert_close(x0_pred[:, :, :1], raw_clean[:, :, :1], rtol=1e-5, atol=1e-5)
    assert captured["hidden_states"].dtype == network.precision_model
    assert captured["encoder_hidden_states"].dtype == network.precision_model
    assert captured["encoder_hidden_states_image"].dtype == network.precision_model

    # The first-frame latent joins the state, so a narrower one is cast to the state's dtype, which holds it exactly.
    narrow_clean = raw_clean.to(torch.bfloat16)
    narrow_output = network(state, timestep, condition={**condition, "first_frame_cond": narrow_clean})
    narrow_input = captured["hidden_states"]
    upcast_output = network(state, timestep, condition={**condition, "first_frame_cond": narrow_clean.to(state.dtype)})
    assert torch.equal(narrow_input, captured["hidden_states"])
    assert torch.equal(narrow_output, upcast_output)

    # The text, image and camera embeddings only reach the backbone, so another dtype is rejected rather than cast.
    mismatched = {
        "text_embeds": torch.randn(1, 2, 3),
        "encoder_hidden_states_image": torch.randn(1, 2, 3),
    }
    if class_name == "WanCamI2V":
        mismatched["c2ws_plucker_emb"] = torch.randn(1, 2, 3)
    for name, value in mismatched.items():
        with pytest.raises(AssertionError, match=rf"{name}\.dtype"):
            network(state, timestep, condition={**condition, name: value})


def test_wan_i2v_sample_runs_on_the_schedule_grid_and_queries_the_network_at_its_sigmas():
    """The sampler builds the schedule's shifted grid (top = ``max_t``), hands it to UniPC pre-shifted, and queries the
    network at exactly the sigmas UniPC steps on: nothing is clamped or truncated."""
    from diffusers import UniPCMultistepScheduler
    from fastgen.networks.WanI2V.network import WanI2V

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()

        def forward(self, hidden_states, *args, **kwargs):
            del args, kwargs
            return torch.zeros_like(hidden_states[:, :2])

    network = WanI2V.__new__(WanI2V)
    torch.nn.Module.__init__(network)
    network.concat_mask = False
    network.expand_timesteps = False
    network.net_pred_type = "flow"
    network.schedule_type = "rf"
    network._record_precision(precision_model=torch.float32)
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer = Transformer()
    network._unipc_scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000, prediction_type="flow_prediction", use_flow_sigmas=True
    )

    seen_times = []
    handle = network.register_forward_pre_hook(
        lambda module, args, kwargs: seen_times.append(args[1][0].item()), with_kwargs=True
    )
    noise = torch.randn(1, 2, 2, 2, 2)
    condition = {"text_embeds": torch.randn(1, 2, 3), "first_frame_cond": torch.randn(1, 2, 1, 2, 2)}
    try:
        with torch.no_grad():
            network.sample(noise, condition=condition, neg_condition=condition, sample_steps=5, shift=3.0)
    finally:
        handle.remove()

    ns, sched = network.noise_scheduler, network._unipc_scheduler
    grid = ns.shift_t(ns.grid(5, dtype=torch.float32), shift=3.0)
    expected = grid[:-1].to(ns.t_precision).tolist()
    assert seen_times == pytest.approx([t for t in expected for _ in range(2)], abs=0)  # two CFG passes per step
    assert torch.equal(sched.sigmas, grid.cpu())  # UniPC steps on exactly this grid, terminal 0 included
    assert 0.998 < expected[0] <= ns.max_t == 0.999  # the top is max_t in float32; nothing clamped
    assert expected[1] != sched.timesteps[1].item() / 1000  # the exact sigma, not the truncated grid value


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("fastgen.networks.WanI2V.network", "WanI2V"),
        ("fastgen.networks.WanI2V.network_cam", "WanCamI2V"),
        ("fastgen.networks.WanI2V.network_causal", "CausalWanI2V"),
    ],
    ids=["base", "camera", "causal"],
)
@pytest.mark.parametrize(
    ("schedule", "net_pred_type", "fwd_pred_type"),
    [
        ("rf", "flow", "x0"),
        ("rf", "flow", "eps"),
        ("rf", "flow", "flow"),
        ("alphas", "eps", "x0"),
        ("alphas", "eps", "eps"),
        ("alphas", "eps", "v"),
    ],
    ids=["rf-x0", "rf-eps", "rf-flow", "alphas-x0", "alphas-eps", "alphas-v"],
)
def test_wan_i2v_non_concat_output_pin_round_trips_for_every_prediction_type(
    module_name, class_name, schedule, net_pred_type, fwd_pred_type
):
    """Frame 0 converts back to the clean frame for either schedule and any domain; other frames are the plain conversion."""
    network_cls = getattr(importlib.import_module(module_name), class_name)

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()
            self.config = SimpleNamespace(patch_size=(1, 1, 1))

        def forward(self, hidden_states, *args, **kwargs):
            del args, kwargs
            return 0.5 * hidden_states[:, :2] - 0.25

    network = network_cls.__new__(network_cls)
    torch.nn.Module.__init__(network)
    network.concat_mask = False
    network.expand_timesteps = False
    network.net_pred_type = net_pred_type
    network._record_precision(precision_model=torch.float32)
    if schedule == "rf":
        network.noise_scheduler = RFNoiseSchedule()
    else:
        betas = torch.linspace(1e-4, 2e-2, 1000, dtype=torch.float64)
        network.noise_scheduler = AlphasNoiseSchedule(torch.cumprod(1 - betas, dim=0))
    network.transformer = Transformer()

    torch.manual_seed(0)
    state = torch.randn(2, 2, 3, 2, 2)
    clean = torch.randn(2, 2, 1, 2, 2)
    timestep = torch.tensor([0.3, 0.8], dtype=network.noise_scheduler.t_precision)
    condition = {"text_embeds": torch.randn(2, 2, 3), "first_frame_cond": clean}
    output = network(state, timestep, condition=condition, fwd_pred_type=fwd_pred_type)

    ns = network.noise_scheduler
    x0_pred = ns.convert_model_output(state, output, timestep, src_pred_type=fwd_pred_type, target_pred_type="x0")
    torch.testing.assert_close(x0_pred[:, :, :1], clean, rtol=1e-5, atol=1e-5)
    if fwd_pred_type == "x0":
        assert torch.equal(output[:, :, :1], clean)
    # Frames 1+ are the standard conversion of the raw network output; the pin touches frame 0 only.
    raw = network.transformer(network._compute_i2v_inputs(state, first_frame_cond=clean, timestep=timestep)[0], None)
    expected = ns.convert_model_output(
        state, raw, timestep, src_pred_type=net_pred_type, target_pred_type=fwd_pred_type
    )
    torch.testing.assert_close(output[:, :, 1:], expected[:, :, 1:])


class _ConditioningTextEncoder:
    def __init__(self, dtype=torch.float64):
        self.dtype = dtype
        self.calls = []

    def encode(self, prompts):
        self.calls.append(prompts)
        return torch.full((len(prompts), 2, 3), len(self.calls), dtype=self.dtype)


class _ConditioningVAE:
    latent_compression_factors = (4, 8, 8)

    def __init__(self):
        self.calls = []

    def encode(self, value, mode):
        self.calls.append((value, mode))
        return value[:, :2, :1].clone()


def test_wan_i2v_build_conditioning_composes_raw_text_once_without_mutation():
    from fastgen.networks.WanI2V.network import WanI2V

    text_encoder = _ConditioningTextEncoder()
    vae = _ConditioningVAE()
    network = WanI2V.__new__(WanI2V)
    torch.nn.Module.__init__(network)
    network.preprocessors = {"text_encoder": text_encoder, "vae": vae}
    network.concat_mask = False
    network.latent_compression_factors = (4, 8, 8)

    real_raw = torch.randn(1, 3, 2, 4, 4, dtype=torch.float32)
    condition_raw = ["positive"]
    neg_condition_raw = ["negative"]
    data = {
        "condition_raw": condition_raw,
        "neg_condition_raw": neg_condition_raw,
        "real_raw": real_raw,
    }
    original_keys = set(data)

    out = network.build_conditioning(data, input_shape=[2, 2, 4, 4])

    assert text_encoder.calls == [condition_raw, neg_condition_raw]
    assert torch.all(out["condition"]["text_embeds"] == 1)
    assert torch.all(out["neg_condition"]["text_embeds"] == 2)
    assert out["condition"]["text_embeds"].dtype == torch.float64
    assert out["neg_condition"]["text_embeds"].dtype == torch.float64
    assert out["condition"]["first_frame_cond"].dtype == real_raw.dtype
    assert out["condition"]["first_frame_cond"] is out["neg_condition"]["first_frame_cond"]
    assert len(vae.calls) == 1
    assert vae.calls[0][1] == "argmax"
    assert torch.equal(vae.calls[0][0], real_raw[:, :, :1])
    assert set(data) == original_keys
    assert data["condition_raw"] is condition_raw
    assert data["neg_condition_raw"] is neg_condition_raw
    assert data["real_raw"] is real_raw


def test_wan_cam_build_conditioning_composes_wan_once_and_preserves_context():
    from fastgen.networks.WanI2V.network_cam import WanCamI2V

    text_encoder = _ConditioningTextEncoder()
    vae = _ConditioningVAE()
    network = WanCamI2V.__new__(WanCamI2V)
    torch.nn.Module.__init__(network)
    network.preprocessors = {"text_encoder": text_encoder, "vae": vae}
    network.concat_mask = False
    network.latent_compression_factors = (4, 8, 8)
    plucker = torch.randn(1, 6, 2, 1, 1)
    network.compute_camera_conditioning = lambda working: plucker

    real_raw = torch.randn(1, 3, 5, 8, 8)
    poses = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 5, 1, 1)
    intrinsics = torch.ones(1, 5, 4)
    data = {
        "condition_raw": ["positive"],
        "neg_condition_raw": ["negative"],
        "real_raw": real_raw,
        "camera_poses": poses,
        "camera_intrinsics": intrinsics,
    }
    original_keys = set(data)

    out = network.build_conditioning(data, input_shape=[2, 2, 1, 1], precision=torch.bfloat16)

    # Exactly one inherited WanI2V pass encodes each requested text branch and
    # the shared first-frame context once.
    assert text_encoder.calls == [["positive"], ["negative"]]
    assert len(vae.calls) == 1
    assert out["condition"]["first_frame_cond"].dtype == torch.bfloat16
    assert torch.all(out["condition"]["text_embeds"] == 1)
    assert torch.all(out["neg_condition"]["text_embeds"] == 2)
    assert out["condition"]["c2ws_plucker_emb"] is plucker
    assert out["neg_condition"]["c2ws_plucker_emb"] is plucker
    assert out["condition"]["first_frame_cond"] is out["neg_condition"]["first_frame_cond"]
    assert set(data) == original_keys
    assert data["real_raw"] is real_raw
    assert data["camera_poses"] is poses


def test_vace_build_conditioning_preserves_encoded_primary_and_encodes_raw_negative():
    from fastgen.networks.VaceWan.network import VACEWan

    text_encoder = _ConditioningTextEncoder()
    network = VACEWan.__new__(VACEWan)
    torch.nn.Module.__init__(network)
    network.preprocessors = {"text_encoder": text_encoder}
    vid_context = torch.randn(1, 6, 2, 2, 2, dtype=torch.bfloat16)
    network._build_vid_context = lambda video, condition_latents: vid_context

    encoded = torch.randn(1, 2, 3, dtype=torch.float64)
    depth_latent = torch.randn(1, 2, 2, 2, 2, dtype=torch.bfloat16)
    condition_raw = ["retained mirror"]
    neg_condition_raw = ["negative"]
    data = {
        "condition": encoded,
        "condition_raw": condition_raw,
        "neg_condition": None,
        "neg_condition_raw": neg_condition_raw,
        "depth_latent": depth_latent,
    }
    original_keys = set(data)

    out = network.build_conditioning(data)

    assert text_encoder.calls == [neg_condition_raw]
    assert out["condition"]["text_embeds"] is encoded
    assert out["condition"]["text_embeds"].dtype == torch.float64
    assert out["neg_condition"]["text_embeds"].dtype == torch.float64
    assert out["condition"]["vid_context"] is out["neg_condition"]["vid_context"] is vid_context
    assert set(data) == original_keys
    assert data["condition"] is encoded
    assert data["neg_condition"] is None


def test_cosmos_predict2_build_conditioning_composes_text_in_t2v_and_v2w():
    text_encoder = _ConditioningTextEncoder()
    t2v = CosmosPredict2.__new__(CosmosPredict2)
    torch.nn.Module.__init__(t2v)
    t2v.preprocessors = {"text_encoder": text_encoder}
    t2v.is_video2world = False
    raw = ["positive"]
    neg_raw = ["negative"]
    t2v_data = {"condition_raw": raw, "neg_condition_raw": neg_raw}

    t2v_out = t2v.build_conditioning(t2v_data)

    assert text_encoder.calls == [raw, neg_raw]
    assert set(t2v_out) == {"condition", "neg_condition"}
    assert torch.all(t2v_out["condition"] == 1)
    assert torch.all(t2v_out["neg_condition"] == 2)
    assert t2v_out["condition"].dtype == torch.float64
    assert t2v_out["neg_condition"].dtype == torch.float64
    assert t2v_data == {"condition_raw": raw, "neg_condition_raw": neg_raw}

    text_encoder.calls.clear()
    vae = _ConditioningVAE()
    v2w = CosmosPredict2.__new__(CosmosPredict2)
    torch.nn.Module.__init__(v2w)
    v2w.preprocessors = {"text_encoder": text_encoder, "vae": vae}
    v2w.is_video2world = True
    v2w.num_conditioning_frames = 2
    encoded = torch.randn(1, 2, 3, dtype=torch.float64)
    real_raw = torch.randn(1, 3, 1, 2, 2, dtype=torch.bfloat16)
    v2w_data = {
        "condition": encoded,
        "condition_raw": ["retained mirror"],
        "neg_condition_raw": ["negative"],
        "real_raw": real_raw,
    }
    original_keys = set(v2w_data)

    v2w_out = v2w.build_conditioning(v2w_data, input_shape=[2, 3, 2, 2])

    assert text_encoder.calls == [["negative"]]
    assert v2w_out["condition"]["text_embeds"] is encoded
    assert v2w_out["neg_condition"]["text_embeds"].dtype == torch.float64
    assert v2w_out["condition"]["conditioning_latents"].dtype == real_raw.dtype
    assert v2w_out["condition"]["condition_mask"].dtype == torch.bool
    assert v2w_out["condition"]["condition_mask"] is v2w_out["neg_condition"]["condition_mask"]
    assert set(v2w_data) == original_keys
    assert v2w_data["condition"] is encoded
    assert v2w_data["real_raw"] is real_raw


def test_wan_moe_build_conditioning_delegates_complete_request_once():
    from fastgen.networks.Wan.wan_moe import WanMoE

    class Expert(torch.nn.Module):
        _resolve_conditions = FastGenNetwork._resolve_conditions

        def __init__(self):
            super().__init__()
            self.calls = []
            self.text_encoder = _ConditioningTextEncoder()
            self.preprocessors = {"text_encoder": self.text_encoder}
            self.precision_model = None  # unprepared, like FastGenModule's class default

        def build_conditioning(self, data, input_shape=None, precision=None):
            self.calls.append((data, input_shape))
            return FastGenNetwork.build_conditioning(self, data, input_shape=input_shape, precision=precision)

    network = WanMoE.__new__(WanMoE)
    torch.nn.Module.__init__(network)
    network.network_high = Expert()
    data = {"condition_raw": ["positive"], "neg_condition_raw": ["negative"]}

    out = network.build_conditioning(data, input_shape=[2, 3, 4, 5])

    assert network.network_high.calls == [(data, [2, 3, 4, 5])]
    assert network.network_high.text_encoder.calls == [["positive"], ["negative"]]
    assert torch.all(out["condition"] == 1)
    assert torch.all(out["neg_condition"] == 2)
    assert out["condition"].dtype == out["neg_condition"].dtype == torch.float64
    assert data == {"condition_raw": ["positive"], "neg_condition_raw": ["negative"]}


@pytest.mark.parametrize(
    ("module_name", "class_name", "attributes"),
    [
        ("fastgen.networks.Flux2.network", "Flux2", {"is_image_edit": True}),
        ("fastgen.networks.LTX2.network", "LTX2", {"is_i2v": True, "train_connectors": True}),
        ("fastgen.networks.WanI2V.network", "WanI2V", {}),
        ("fastgen.networks.WanI2V.network_cam", "WanCamI2V", {}),
        ("fastgen.networks.VaceWan.network", "VACEWan", {}),
        ("fastgen.networks.cosmos_predict2.network", "CosmosPredict2", {"is_video2world": True}),
    ],
)
def test_rich_conditioning_builders_require_a_resolved_positive(module_name, class_name, attributes):
    network_cls = getattr(importlib.import_module(module_name), class_name)
    network = network_cls.__new__(network_cls)
    torch.nn.Module.__init__(network)
    network.preprocessors = {}
    for name, value in attributes.items():
        setattr(network, name, value)

    with pytest.raises(ValueError, match=r"requires non-None data\['condition'\]"):
        network.build_conditioning({})


def test_wan_cam_i2v_rejects_invalid_forward_prediction_type():
    from fastgen.networks.WanI2V.network_cam import WanCamI2V

    network = WanCamI2V.__new__(WanCamI2V)
    torch.nn.Module.__init__(network)
    network.net_pred_type = "flow"
    network.transformer = SimpleNamespace(r_embedder=None, blocks=[])
    state = torch.zeros(1)
    timestep = torch.zeros(1)
    condition = {"text_embeds": None, "first_frame_cond": None}

    with pytest.raises(ValueError, match="requires feature_indices"):
        network(state, timestep, condition=condition, return_features_early=True)

    with pytest.raises(ValueError, match="fwd_pred_type"):
        network.transform_output(state, state, timestep, condition, fwd_pred_type="unsupported")


def test_causal_wan_i2v_forward_reuses_condition_only_for_initial_chunk():
    from fastgen.networks.WanI2V.network_causal import CausalWanI2V

    captured_text_embeds = []

    class Transformer(torch.nn.Module):
        blocks = ()

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()
            self.config = SimpleNamespace(patch_size=(1, 1, 1))

        def forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
            del timestep, kwargs
            captured_text_embeds.append(encoder_hidden_states)
            return torch.zeros_like(hidden_states[:, :2])

    network = CausalWanI2V.__new__(CausalWanI2V)
    torch.nn.Module.__init__(network)
    network.concat_mask = False
    network.net_pred_type = "flow"
    network._record_precision(precision_model=torch.bfloat16)
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer = Transformer()

    state = torch.randn(1, 2, 2, 2, 2, dtype=torch.float32)
    raw_clean = torch.randn(1, 2, 1, 2, 2, dtype=state.dtype)
    replacement_inputs = []
    original_replace = network._replace_first_frame

    def replace_first_frame(first_frame_cond, latents, return_mask=False):
        replacement_inputs.append(first_frame_cond)
        return original_replace(first_frame_cond, latents, return_mask=return_mask)

    network._replace_first_frame = replace_first_frame
    timestep = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
    text_embeds = torch.randn(1, 2, 3, dtype=network.precision_model)
    output = network(
        state,
        timestep,
        condition={"text_embeds": text_embeds, "first_frame_cond": raw_clean},
        causal_ctx=CausalContext(cur_start_frame=0),
    )

    assert output.dtype == state.dtype
    assert len(replacement_inputs) == 2
    assert replacement_inputs[0] is raw_clean
    # The output pin is computed in the output transform's high-precision dtype.
    assert replacement_inputs[1].dtype == high_precision_dtype(state)
    # Frame 0 of the flow output is the clean frame expressed in the flow domain, not the latent itself.
    x0_pred = network.noise_scheduler.convert_model_output(
        state, output, timestep, src_pred_type="flow", target_pred_type="x0"
    )
    torch.testing.assert_close(x0_pred[:, :, :1], raw_clean, rtol=1e-5, atol=1e-5)
    assert captured_text_embeds[-1].dtype == network.precision_model

    later_output = network(
        state,
        timestep,
        condition={"text_embeds": text_embeds, "first_frame_cond": raw_clean},
        causal_ctx=CausalContext(cur_start_frame=1),
    )

    assert later_output.dtype == state.dtype
    assert len(replacement_inputs) == 2
    assert all(text_embeds.dtype == network.precision_model for text_embeds in captured_text_embeds)

    # A narrower first-frame condition conditions like its exact upcast to the state's dtype.
    narrow_clean = raw_clean.to(torch.bfloat16)
    narrow_output, upcast_output = [
        network(
            state,
            timestep,
            condition={"text_embeds": text_embeds, "first_frame_cond": clean},
            causal_ctx=CausalContext(cur_start_frame=0),
        )
        for clean in (narrow_clean, narrow_clean.to(state.dtype))
    ]
    assert torch.equal(narrow_output, upcast_output)


@pytest.mark.parametrize("with_image_branch", [True, False], ids=["i2v", "t2v"])
def test_causal_wan_cross_attention_matches_diffusers_processor(with_image_branch):
    """Uncached, the causal processor's cross-attention matches diffusers' Wan processor, which attends the Wan2.1
    I2V image tokens and the text tokens with separate softmaxes."""
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    from fastgen.networks.Wan.network_causal import TEXT_ENCODER_CONTEXT_LENGTH, CausalWanAttnProcessor

    torch.manual_seed(0)
    attn = WanAttention(
        dim=16,
        heads=2,
        dim_head=8,
        added_kv_proj_dim=16 if with_image_branch else None,
        cross_attention_dim_head=8,
    )
    hidden_states = torch.randn(2, 6, 16)
    context = torch.randn(2, (257 if with_image_branch else 0) + TEXT_ENCODER_CONTEXT_LENGTH, 16)
    with torch.no_grad():
        expected = WanAttnProcessor()(attn, hidden_states, context)
        actual = CausalWanAttnProcessor()(attn, hidden_states, context)
    torch.testing.assert_close(actual, expected)


def test_causal_wan_self_attention_without_a_block_mask_runs_unpadded():
    """Without a block mask or a KV cache, the causal processor's self-attention matches diffusers' Wan processor:
    the zero keys that pad flex attention's sequence to 128 tokens never enter the softmax."""
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    from fastgen.networks.Wan.network_causal import CausalWanAttnProcessor

    torch.manual_seed(0)
    attn = WanAttention(dim=32, heads=2, dim_head=_FLEX_ATTENTION_HEAD_DIM)
    hidden_states = torch.randn(2, 12, 32)
    angles = torch.randn(1, 12, 1, _FLEX_ATTENTION_HEAD_DIM)
    rotary_emb = (angles.cos(), angles.sin())
    with torch.no_grad():
        expected = WanAttnProcessor()(attn, hidden_states, rotary_emb=rotary_emb)
        actual = CausalWanAttnProcessor()(attn, hidden_states, rotary_emb=rotary_emb)
    torch.testing.assert_close(actual, expected)


def _tiny_causal_wan_family_network(kind, tmp_path):
    """A causal Wan-family network built by its constructor from a tiny local config, with flex-sized heads."""
    import json

    from fastgen.networks.VaceWan.network_causal import CausalVACEWan
    from fastgen.networks.Wan.network_causal import CausalWan
    from fastgen.networks.WanI2V.network_causal import CausalWanI2V

    config = {
        "_class_name": "WanTransformer3DModel",
        "attention_head_dim": _FLEX_ATTENTION_HEAD_DIM,
        "cross_attn_norm": True,
        "eps": 1e-06,
        "ffn_dim": 32,
        "freq_dim": 16,
        "in_channels": 4,
        "num_attention_heads": 2,
        "num_layers": 2,
        "out_channels": 4,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 32,
        "text_dim": 8,
    }
    network_cls, model_name = {
        "causal_wan": (CausalWan, "Wan2.1-T2V-1.3B-Diffusers"),
        "causal_wan_i2v": (CausalWanI2V, "Wan2.2-TI2V-5B-Diffusers"),
        "causal_vace": (CausalVACEWan, "Wan2.1-VACE-1.3B-diffusers"),
    }[kind]
    if kind == "causal_vace":
        config.update(_class_name="WanVACETransformer3DModel", vace_layers=[0], vace_in_channels=8)
    model_dir = tmp_path / model_name
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "transformer" / "config.json").write_text(json.dumps(config))
    return network_cls(
        model_id_or_local_path=str(model_dir),
        load_pretrained=False,
        disable_grad_ckpt=True,
        enable_logvar_linear=False,
        chunk_size=1,
        total_num_frames=3,
    ).eval()


@RunIf(min_gpus=1)
@pytest.mark.parametrize("kind", ["causal_wan", "causal_wan_i2v", "causal_vace"])
def test_causal_wan_family_attends_block_causally_only_with_a_context(kind, tmp_path):
    """A causal context runs causal training's full-sequence block-causal pass: a chunk never sees later frames,
    and flex attention's zero padding never enters the softmax, so the output equals a chunk-by-chunk KV-cache
    rollout of the same inputs. Without a context, the network attends to the whole sequence."""
    device = torch.device("cuda")
    net = _tiny_causal_wan_family_network(kind, tmp_path).to(device)
    torch.manual_seed(0)
    x = torch.randn(1, 4, 3, 4, 4, device=device)
    t = torch.full((1,), 0.5, dtype=net.noise_scheduler.t_precision, device=device)
    condition = torch.randn(1, 6, 8, device=device)
    if kind == "causal_wan_i2v":
        condition = {"text_embeds": condition, "first_frame_cond": torch.randn(1, 4, 1, 4, 4, device=device)}
    elif kind == "causal_vace":
        condition = {"text_embeds": condition, "vid_context": torch.randn(1, 8, 3, 4, 4, device=device)}
    later_frames_changed = torch.cat([x[:, :, :1], torch.randn_like(x[:, :, 1:])], dim=2)

    def forward(state, **kwargs):
        # The unconstrained network output: the TI2V frame-0 pin would hide what frame 0 attends to.
        return net(state, t, condition=condition, apply_output_conditioning=False, **kwargs)

    with torch.no_grad():
        out = forward(x, causal_ctx=CausalContext(chunk_size=1, total_num_frames=3))
        out_later_frames_changed = forward(
            later_frames_changed, causal_ctx=CausalContext(chunk_size=1, total_num_frames=3)
        )
        unmasked = forward(x)
        unmasked_later_frames_changed = forward(later_frames_changed)
        ctx = _make_causal_ctx(net, is_ar=True)
        rollout = []
        for start in range(3):
            ctx.cur_start_frame = start
            with ctx.set_store_kv(True):
                rollout.append(forward(x[:, :, start : start + 1], causal_ctx=ctx))
        net.clear_caches()

    assert torch.equal(out_later_frames_changed[:, :, :1], out[:, :, :1])
    assert not torch.allclose(out_later_frames_changed[:, :, 1:], out[:, :, 1:])
    assert not torch.allclose(unmasked_later_frames_changed[:, :, :1], unmasked[:, :, :1])
    # Flex and SDPA kernels differ by TF32 rounding (about 6e-4 here); padding keys or a bidirectional pass move
    # the output by O(1).
    torch.testing.assert_close(torch.cat(rollout, dim=2), out, rtol=5e-3, atol=5e-3)


def test_causal_wan_i2v_cross_cache_is_reused_across_chunks(tmp_path):
    """End to end on a tiny Wan2.1-I2V-shaped transformer: the cross cache holds the image and text K/V, is
    allocated once, and a cache hit skips every K/V projection without changing the output."""
    import json

    from fastgen.networks.WanI2V.network_causal import CausalWanI2V

    model_dir = tmp_path / "Wan2.1-I2V-14B-480P-Diffusers"
    (model_dir / "transformer").mkdir(parents=True)
    tiny_config = {
        "_class_name": "WanTransformer3DModel",
        "added_kv_proj_dim": 16,
        "attention_head_dim": 8,
        "cross_attn_norm": True,
        "eps": 1e-06,
        "ffn_dim": 32,
        "freq_dim": 32,
        "image_dim": 16,
        "in_channels": 36,
        "num_attention_heads": 2,
        "num_layers": 1,
        "out_channels": 16,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 16,
    }
    (model_dir / "transformer" / "config.json").write_text(json.dumps(tiny_config))
    net = CausalWanI2V(
        model_id_or_local_path=str(model_dir),
        load_pretrained=False,
        disable_efficient_attn=True,
        disable_grad_ckpt=True,
        enable_logvar_linear=False,
        chunk_size=1,
        total_num_frames=2,
    ).eval()
    attn2 = net.transformer.blocks[0].attn2
    assert net.concat_mask and attn2.add_k_proj is not None

    kv_projections = ["add_k_proj", "add_v_proj", "to_k", "to_v"]
    projection_calls = []
    for name in kv_projections:
        getattr(attn2, name).register_forward_hook(lambda *_, name=name: projection_calls.append(name))

    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, 4, 4)
    t = torch.full((1,), 0.5, dtype=net.noise_scheduler.t_precision)
    condition = {
        "text_embeds": torch.randn(1, 512, 16),
        "first_frame_cond": torch.randn(1, 16, 2, 4, 4),
        "encoder_hidden_states_image": torch.randn(1, 257, 16),
    }
    engine = net.transformer.kv_cache_engines["main"]

    def forward(cur_start_frame, store_kv):
        causal_ctx = _make_causal_ctx(net, store_kv=store_kv, is_ar=True, cur_start_frame=cur_start_frame)
        return net(x, t, condition=condition, causal_ctx=causal_ctx)

    with torch.no_grad():
        forward(cur_start_frame=0, store_kv=True)
    key_first, _, initialized_first = engine.get_cross_kv(0, "pos")
    assert initialized_first and key_first.shape[1] == 257 + 512  # image then text tokens, as in the context
    assert sorted(projection_calls) == kv_projections

    projection_calls.clear()
    with torch.no_grad():
        out_cached = forward(cur_start_frame=1, store_kv=False)
    key_second, _, initialized_second = engine.get_cross_kv(0, "pos")
    assert projection_calls == []
    assert initialized_second and key_second.data_ptr() == key_first.data_ptr()  # no per-chunk reallocation

    with torch.enable_grad():  # the grad path recomputes K/V instead of reading the cache
        out_fresh = forward(cur_start_frame=1, store_kv=False)
    assert sorted(projection_calls) == kv_projections
    torch.testing.assert_close(out_cached, out_fresh.detach())


def test_causal_wan_i2v_concat_slices_before_validating_condition():
    from fastgen.networks.WanI2V.network_causal import CausalWanI2V

    events = []
    captured = {}

    class Transformer(torch.nn.Module):
        blocks = ()

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()
            self.config = SimpleNamespace(patch_size=(1, 1, 1))

        def forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
            del timestep, kwargs
            captured["hidden_states"] = hidden_states
            captured["encoder_hidden_states"] = encoder_hidden_states
            return torch.zeros_like(hidden_states[:, :2])

    source = torch.randn(1, 2, 4, 2, 2, dtype=torch.float32)

    class FirstFrameCondition:
        def __getitem__(self, key):
            events.append("slice")
            return source[key]

        def to(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("causal concat must slice before validation")

    network = CausalWanI2V.__new__(CausalWanI2V)
    torch.nn.Module.__init__(network)
    network.concat_mask = True
    network.latent_compression_factors = (4, 8, 8)
    network.net_pred_type = "flow"
    network._record_precision(precision_model=torch.bfloat16)
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer = Transformer()

    aligned_conditions = []
    original_compute = network._compute_i2v_inputs

    def compute_i2v_inputs(*args, **kwargs):
        aligned_conditions.append(kwargs["first_frame_cond"])
        return original_compute(*args, **kwargs)

    network._compute_i2v_inputs = compute_i2v_inputs
    state = torch.randn(1, 2, 2, 2, 2, dtype=torch.float32)
    output = network(
        state,
        torch.tensor([0.5], dtype=network.noise_scheduler.t_precision),
        condition={
            "text_embeds": torch.randn(1, 2, 3, dtype=network.precision_model),
            "first_frame_cond": FirstFrameCondition(),
        },
        causal_ctx=CausalContext(cur_start_frame=1),
    )

    assert events == ["slice"]
    assert len(aligned_conditions) == 1
    assert aligned_conditions[0].shape == state.shape
    assert aligned_conditions[0].dtype == state.dtype
    assert captured["hidden_states"].dtype == network.precision_model
    assert captured["encoder_hidden_states"].dtype == network.precision_model
    assert output.dtype == state.dtype


def test_causal_wan_i2v_non_concat_keeps_per_frame_timesteps_when_masking_all_frames():
    """``mask_all_frames`` (KV-store passes with ``mask_when_storing_cache``) must keep the ``[B, T]`` layout."""
    from fastgen.networks.WanI2V.network_causal import CausalWanI2V

    network = CausalWanI2V.__new__(CausalWanI2V)
    torch.nn.Module.__init__(network)
    network.concat_mask = False
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 1, 1)))

    state = torch.randn(2, 2, 3, 2, 2)
    clean = torch.randn(2, 2, 1, 2, 2)
    timestep = torch.full((2,), 0.5, dtype=network.noise_scheduler.t_precision)
    _, all_frames, _ = network._compute_i2v_inputs(
        state, first_frame_cond=clean, timestep=timestep, mask_all_frames=True
    )
    _, first_frame_pinned, _ = network._compute_i2v_inputs(state, first_frame_cond=clean, timestep=timestep)

    assert all_frames.shape == first_frame_pinned.shape == (2, 3)
    assert torch.equal(first_frame_pinned[:, 0], torch.zeros(2, dtype=first_frame_pinned.dtype))
    assert torch.equal(all_frames, torch.full_like(all_frames, first_frame_pinned[0, 1].item()))


def test_causal_wan_i2v_output_pin_rejects_a_noise_free_first_frame():
    """Per-frame timesteps with ``t = 0`` at frame 0 (clean context) cannot carry the flow pin, so constraining them
    raises; without output conditioning, as KV-cache stores run, that frame is the model's own finite output."""
    from fastgen.networks.WanI2V.network_causal import CausalWanI2V

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()
            self.config = SimpleNamespace(patch_size=(1, 1, 1))

        def forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
            del timestep, encoder_hidden_states, kwargs
            return 0.5 * hidden_states[:, :2] - 0.25

    network = CausalWanI2V.__new__(CausalWanI2V)
    torch.nn.Module.__init__(network)
    network.concat_mask = False
    network.net_pred_type = "flow"
    network._record_precision(precision_model=torch.float32)
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer = Transformer()

    torch.manual_seed(0)
    state = torch.randn(1, 2, 3, 2, 2)
    clean = torch.randn(1, 2, 1, 2, 2)
    condition = {"text_embeds": torch.randn(1, 2, 3), "first_frame_cond": clean}
    t_precision = network.noise_scheduler.t_precision
    noised = torch.tensor([[0.4, 0.4, 0.4]], dtype=t_precision)
    clean_first = torch.tensor([[0.0, 0.4, 0.4]], dtype=t_precision)
    out_noised = network(state, noised, condition=condition, causal_ctx=CausalContext(cur_start_frame=0))
    with pytest.raises(ValueError, match="sigma = 0"):
        network(state, clean_first, condition=condition, causal_ctx=CausalContext(cur_start_frame=0))
    out_clean_first = network(
        state,
        clean_first,
        condition=condition,
        causal_ctx=CausalContext(cur_start_frame=0),
        apply_output_conditioning=False,
    )
    assert torch.isfinite(out_clean_first).all()

    ns = network.noise_scheduler
    x0_pred = ns.convert_model_output(
        state, out_noised, noised[:, None, :, None, None], src_pred_type="flow", target_pred_type="x0"
    )
    torch.testing.assert_close(x0_pred[:, :, :1], clean, rtol=1e-5, atol=1e-5)
    raw = network.transformer(
        network._compute_i2v_inputs(state, first_frame_cond=clean, timestep=clean_first)[0], None, None
    )
    assert torch.equal(out_clean_first[:, :, :1], raw[:, :, :1])
    torch.testing.assert_close(out_clean_first[:, :, 1:], out_noised[:, :, 1:])

    # A KV-cache store runs every frame at the clean end, without output conditioning.
    network._mask_when_storing_cache = False
    ctx = CausalContext(cur_start_frame=0)
    with ctx.set_store_kv(True):
        stored = network(
            state,
            torch.zeros(1, dtype=t_precision),
            condition=condition,
            causal_ctx=ctx,
            apply_output_conditioning=False,
        )
    assert torch.isfinite(stored).all()


def test_causal_wan_i2v_non_concat_only_uses_first_frame_in_initial_chunk():
    from fastgen.networks.WanI2V.network_causal import CausalWanI2V

    network = CausalWanI2V.__new__(CausalWanI2V)
    torch.nn.Module.__init__(network)
    network.concat_mask = False
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 1, 1)))

    state = torch.randn(1, 2, 2, 2, 2, dtype=torch.bfloat16)
    clean = torch.randn(1, 2, 1, 2, 2, dtype=torch.float32).to(dtype=state.dtype)
    original_replace = network._replace_first_frame
    replacement_inputs = []

    def replace_first_frame(first_frame_cond, latents, return_mask=False):
        replacement_inputs.append(first_frame_cond)
        return original_replace(first_frame_cond, latents, return_mask=return_mask)

    network._replace_first_frame = replace_first_frame
    initial_input, _, _ = network._compute_i2v_inputs(
        state,
        first_frame_cond=clean,
        timestep=torch.tensor([0.5], dtype=network.noise_scheduler.t_precision),
        cur_start_frame=0,
    )

    assert len(replacement_inputs) == 1
    assert replacement_inputs[0] is clean
    assert initial_input.dtype == state.dtype
    assert torch.equal(initial_input[:, :, :1], clean)
    assert torch.equal(initial_input[:, :, 1:], state[:, :, 1:])

    class UnusedFirstFrame:
        def to(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("later causal chunks must not align the first-frame condition")

    later_input, _, _ = network._compute_i2v_inputs(
        state,
        first_frame_cond=UnusedFirstFrame(),
        timestep=torch.tensor([0.5], dtype=network.noise_scheduler.t_precision),
        cur_start_frame=1,
    )

    assert later_input is state
    assert len(replacement_inputs) == 1
    assert replacement_inputs[0] is clean


# Wan r-embedder: ``Wan.init_embedder`` only needs the transformer's condition embedder, so the
# real diffusers module is used without instantiating a Wan transformer.
_WAN_EMBEDDER_DIM, _WAN_EMBEDDER_FREQ = 16, 32


def _wan_embedder_stub(seed: int = 0):
    torch.manual_seed(seed)
    condition_embedder = WanTimeTextImageEmbedding(
        dim=_WAN_EMBEDDER_DIM,
        time_freq_dim=_WAN_EMBEDDER_FREQ,
        time_proj_dim=6 * _WAN_EMBEDDER_DIM,
        text_embed_dim=8,
    )
    return SimpleNamespace(
        transformer=SimpleNamespace(condition_embedder=condition_embedder),
        _is_in_meta_context=lambda: False,
    )


def _wan_r_contribution(r_embedder, r):
    """The additive r contribution as ``classify_forward_prepare`` builds it: ``(remb, r_ts_proj)``."""
    remb = r_embedder.time_embedder(r_embedder.timesteps_proj(r))
    return remb, r_embedder.time_proj(r_embedder.act_fn(remb))


def test_wan_time_embedder_tracks_live_projection_dtype_across_to_calls():
    from fastgen.networks.Wan.network import wan_timestep_embedding_forward

    condition_embedder = WanTimeTextImageEmbedding(
        dim=_WAN_EMBEDDER_DIM,
        time_freq_dim=_WAN_EMBEDDER_FREQ,
        time_proj_dim=6 * _WAN_EMBEDDER_DIM,
        text_embed_dim=8,
    ).float()
    condition_embedder.time_embedder.forward = MethodType(
        wan_timestep_embedding_forward, condition_embedder.time_embedder
    )

    proj_input_dtypes = []
    time_embedder_handle = condition_embedder.time_embedder.linear_1.register_forward_pre_hook(
        lambda module, args: proj_input_dtypes.append(args[0].dtype)
    )
    try:
        output_fp32 = condition_embedder.time_embedder(torch.randn(2, _WAN_EMBEDDER_FREQ, dtype=torch.bfloat16))

        # An ordinary dtype conversion must take effect immediately; there is no
        # constructor-time precision flag for the input path to become stale.
        condition_embedder.time_embedder.to(dtype=torch.bfloat16)
        output_bf16 = condition_embedder.time_embedder(torch.randn(2, _WAN_EMBEDDER_FREQ, dtype=torch.float32))
    finally:
        time_embedder_handle.remove()

    assert proj_input_dtypes == [torch.float32, torch.bfloat16]
    assert output_fp32.dtype == torch.float32
    assert output_bf16.dtype == torch.bfloat16


def test_wan_loads_official_keys():
    from fastgen.networks.Wan.network import Wan

    network = object.__new__(Wan)
    torch.nn.Module.__init__(network)
    network.transformer = torch.nn.Module()
    network.transformer.condition_embedder = torch.nn.Module()
    network.transformer.condition_embedder.time_embedder = torch.nn.Module()
    network.transformer.condition_embedder.time_embedder.linear_1 = torch.nn.Linear(2, 2, bias=False)

    weight = torch.randn(2, 2)
    load_info = network.load_state_dict({"time_embedding.0.weight": weight})

    assert load_info.missing_keys == []
    assert load_info.unexpected_keys == []
    torch.testing.assert_close(network.transformer.condition_embedder.time_embedder.linear_1.weight, weight)


@pytest.mark.parametrize("wrapper", [None, "generator", "state_dict", "model", "module", "net"])
@pytest.mark.parametrize("prefixed", [False, True], ids=["diffusers", "fastgen"])
def test_wan_loads_bare_diffusers_keys(wrapper, prefixed):
    """Bare and nested diffusers/FastGen layouts preserve norm names and unwrap before adding a prefix."""
    from fastgen.networks.Wan.network import Wan

    network = object.__new__(Wan)
    torch.nn.Module.__init__(network)
    network.transformer = torch.nn.Module()
    network.transformer.blocks = torch.nn.ModuleList([torch.nn.Module()])
    network.transformer.blocks[0].norm2 = torch.nn.LayerNorm(2)
    network.transformer.blocks[0].norm3 = torch.nn.LayerNorm(2)

    expected = {name: torch.randn_like(tensor) for name, tensor in network.transformer.state_dict().items()}
    checkpoint = {f"transformer.{name}" if prefixed else name: tensor for name, tensor in expected.items()}
    if wrapper is not None:
        checkpoint = {wrapper: checkpoint}
    load_info = network.load_state_dict(checkpoint)

    assert load_info.missing_keys == []
    assert load_info.unexpected_keys == []
    state = network.transformer.state_dict()
    for name, tensor in expected.items():
        torch.testing.assert_close(state[name], tensor)


def test_wan_loads_anyflow_release_keys():
    """AnyFlow's diffusers releases store the r-pathway as ``condition_embedder.delta_embedder``."""
    from fastgen.networks.Wan.network import Wan

    network = object.__new__(Wan)
    torch.nn.Module.__init__(network)
    network.transformer = torch.nn.Module()
    network.transformer.r_embedder = torch.nn.Module()
    network.transformer.r_embedder.time_embedder = torch.nn.Module()
    network.transformer.r_embedder.time_embedder.linear_1 = torch.nn.Linear(2, 2, bias=False)

    weight = torch.randn(2, 2)
    load_info = network.load_state_dict({"condition_embedder.delta_embedder.linear_1.weight": weight})

    assert load_info.missing_keys == []
    assert load_info.unexpected_keys == []
    torch.testing.assert_close(network.transformer.r_embedder.time_embedder.linear_1.weight, weight)


@pytest.mark.parametrize("initialize_adapter", [False, True])
def test_wan_loads_nested_adapter_state_requires_an_adapter_target(initialize_adapter):
    from fastgen.networks.Wan.network import Wan
    from fastgen.utils.module import load_module_state_dict

    target = object.__new__(Wan)
    torch.nn.Module.__init__(target)
    target.transformer = torch.nn.Module()
    target.transformer.proj_in = torch.nn.Linear(2, 2)
    target.transformer.proj_out = torch.nn.Linear(2, 2)
    source = copy.deepcopy(target)
    source.init_lora(rank=1, target_modules=["proj_in"])
    with torch.no_grad():
        source.transformer.proj_out.weight.fill_(7)
    if initialize_adapter:
        target.init_lora(rank=1, target_modules=["proj_in"])
    before = copy.deepcopy(target.state_dict())
    checkpoint = {"model": source.state_dict()}

    if initialize_adapter:
        load_info = load_module_state_dict(target, checkpoint, strict=False, require_any=True)
        assert load_info.missing_keys == []
        assert load_info.unexpected_keys == []
        torch.testing.assert_close(target.state_dict(), source.state_dict())
        return

    # A permissive pretrained load must not silently ignore the adapter while loading other weights.
    with pytest.raises(RuntimeError, match="no initialized adapter"):
        load_module_state_dict(target, checkpoint, strict=False, require_any=True)
    torch.testing.assert_close(target.state_dict(), before)


def test_vace_loads_official_control_projection_keys():
    """Original VACE checkpoints name the control-block projections ``before_proj`` / ``after_proj``; diffusers'
    conversion renames them to ``proj_in`` / ``proj_out``."""
    from fastgen.networks.VaceWan.network import VACEWan

    network = object.__new__(VACEWan)
    torch.nn.Module.__init__(network)
    network.transformer = _tiny_vace_transformer()
    expected = {
        name: torch.randn_like(tensor)
        for name, tensor in network.transformer.state_dict().items()
        if name.startswith("vace_blocks.0.proj_")
    }
    official = {
        name.replace("proj_in", "before_proj").replace("proj_out", "after_proj"): tensor
        for name, tensor in expected.items()
    }
    assert len(official) == 4  # input and output projection, weight and bias

    load_info = network.load_state_dict(official, strict=False)

    assert load_info.unexpected_keys == []
    state = network.transformer.state_dict()
    for name, tensor in expected.items():
        torch.testing.assert_close(state[name], tensor)


def test_wan_condition_and_r_time_embedders_infer_their_dtypes_independently():
    from fastgen.networks.Wan.network import Wan, wan_timestep_embedding_forward

    stub = _wan_embedder_stub()
    condition_embedder = stub.transformer.condition_embedder.float()
    r_embedder = Wan.init_embedder(stub, "pretrained").to(dtype=torch.bfloat16)
    condition_embedder.time_embedder.forward = MethodType(
        wan_timestep_embedding_forward, condition_embedder.time_embedder
    )
    r_embedder.time_embedder.forward = MethodType(wan_timestep_embedding_forward, r_embedder.time_embedder)

    seen = {"condition": [], "r": []}
    handles = [
        condition_embedder.time_embedder.linear_1.register_forward_pre_hook(
            lambda module, args: seen["condition"].append(args[0].dtype)
        ),
        r_embedder.time_embedder.linear_1.register_forward_pre_hook(
            lambda module, args: seen["r"].append(args[0].dtype)
        ),
    ]
    try:
        condition_embedder.time_embedder(torch.randn(2, _WAN_EMBEDDER_FREQ, dtype=torch.bfloat16))
        r_embedder.time_embedder(torch.randn(2, _WAN_EMBEDDER_FREQ, dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()

    assert seen == {"condition": [torch.float32], "r": [torch.bfloat16]}


@pytest.mark.parametrize("embedder_init", ["zero", "random"])
def test_wan_r_embedder_zero_output_inits_are_a_no_op_that_still_trains_the_r_dependence(embedder_init):
    """The r path contributes nothing at init, yet its weights receive gradients and learn r."""
    from fastgen.networks.Wan.network import Wan

    stub = _wan_embedder_stub()
    r_embedder = Wan.init_embedder(stub, embedder_init)
    r_a, r_b = torch.full((3,), 300.0), torch.full((3,), 100.0)
    head = torch.nn.Linear(_WAN_EMBEDDER_DIM, 1).requires_grad_(False)

    with torch.no_grad():
        for r in (r_a, r_b):
            remb, r_ts_proj = _wan_r_contribution(r_embedder, r)
            assert not remb.abs().any() and not r_ts_proj.abs().any()

    def loss_fn():
        out_a = head(_wan_r_contribution(r_embedder, r_a)[0])
        out_b = head(_wan_r_contribution(r_embedder, r_b)[0])
        return (out_a - 1.0).square().mean() + (out_b + 1.0).square().mean()

    # The first backward pass must reach a weight of the hidden-to-output map, not just biases.
    loss_fn().backward()
    assert r_embedder.time_embedder.linear_2.weight.grad.abs().any()

    optimizer = torch.optim.Adam(r_embedder.parameters(), lr=1e-2)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss_fn().backward()
        optimizer.step()
    with torch.no_grad():
        out_a = head(_wan_r_contribution(r_embedder, r_a)[0])
        out_b = head(_wan_r_contribution(r_embedder, r_b)[0])
    assert (out_a - out_b).abs().min() > 1.0, "the r path never became r-dependent"
    assert not torch.equal(
        r_embedder.time_embedder.linear_1.weight, torch.zeros_like(r_embedder.time_embedder.linear_1.weight)
    )


def test_wan_r_embedder_zero_init_keeps_the_copied_hidden_layer_and_zeroes_only_the_output_layers():
    from fastgen.networks.Wan.network import Wan

    stub = _wan_embedder_stub()
    source = stub.transformer.condition_embedder.time_embedder.linear_1
    r_embedder = Wan.init_embedder(stub, "zero")

    torch.testing.assert_close(r_embedder.time_embedder.linear_1.weight, source.weight)
    torch.testing.assert_close(r_embedder.time_embedder.linear_1.bias, source.bias)
    for linear in (r_embedder.time_embedder.linear_2, r_embedder.time_proj):
        assert not linear.weight.abs().any()
        assert not linear.bias.abs().any()
    # The copied text and image branches are dropped from the r path.
    assert not hasattr(r_embedder, "text_embedder")


def test_wan_r_embedder_pretrained_init_is_a_full_copy_with_a_non_zero_contribution():
    from fastgen.networks.Wan.network import Wan

    stub = _wan_embedder_stub()
    r_embedder = Wan.init_embedder(stub, "pretrained")
    source = stub.transformer.condition_embedder
    torch.testing.assert_close(r_embedder.time_embedder.linear_2.weight, source.time_embedder.linear_2.weight)
    with torch.no_grad():
        remb, r_ts_proj = _wan_r_contribution(r_embedder, torch.full((2,), 300.0))
    assert remb.abs().any() and r_ts_proj.abs().any()


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_causal_vace_wan():
    clear_gpu_memory()
    set_env_vars()
    config = copy.deepcopy(CausalVACE_Wan_1_3B_Config)
    config.enable_logvar_linear = True
    teacher = instantiate(config)
    teacher.init_preprocessors()
    teacher = teacher.to(device="cuda")
    teacher._set_device("cuda")
    _load_preprocessors(teacher, torch.device("cuda"), torch.bfloat16)

    state = torch.randn(1, 16, 3, 4, 4, device="cuda", dtype=_STATE_DTYPE)
    timestep = teacher.noise_scheduler.sample_t(1, time_dist_type="uniform").to(device="cuda")
    with _borrow_on_device(teacher.preprocessors["text_encoder"], torch.device("cuda")):
        text_embeds = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        text_embeds, teacher.preprocessors["text_encoder"].precision_model, "causal_vace.text_encoder.encode"
    )
    pixels = torch.randn(1, 3, *teacher.pixel_shape(state.shape[1:]), device="cuda", dtype=_STATE_DTYPE).clamp(-1, 1)
    with _borrow_on_device(teacher.preprocessors["depth_annotator"], torch.device("cuda")), _borrow_on_device(
        teacher.preprocessors["vae"], torch.device("cuda")
    ):
        vid_context = teacher._build_vid_context(pixels).to(dtype=torch.bfloat16)
    condition = {"text_embeds": text_embeds, "vid_context": vid_context}

    teacher = teacher.to(device="cuda", dtype=torch.bfloat16)
    teacher._set_device("cuda")
    output, logvar = teacher(
        state,
        timestep,
        condition=condition,
        causal_ctx=_make_causal_ctx(teacher, total_num_frames=state.shape[2]),
        return_logvar=True,
    )

    assert output.shape == state.shape and output.dtype == state.dtype
    assert logvar.shape == (state.shape[0], 1) and logvar.dtype == state.dtype
    del teacher
    clear_gpu_memory()


@RunIf(min_gpus=1, min_gpu_mem=24)
@pytest.mark.large_model
def test_network_causal_wan():
    """
    Test CausalWan network, specifically the sample method.
    """
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    set_env_vars()
    teacher_config = CausalWan_1_3B_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Instantiate CausalWan and prepare
    teacher = instantiate(teacher_config)
    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # RF schedule for WAN
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    C, T, H, W = 16, 3, 4, 4  # Use T divisible by chunk_size=3

    # B, C, T, H, W
    x = torch.randn(batch_size, C, T, H, W, device=device, dtype=_STATE_DTYPE)

    # CausalWan supports 2D timesteps with shape (batch_size, num_frames)
    t_1d = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)
    t = t_1d.unsqueeze(1).expand(batch_size, T)  # Shape: (batch_size, T)

    captions = ["a test caption for causal WAN"]
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        condition = teacher.preprocessors["text_encoder"].encode(captions)
    _assert_gpu_tree_dtype_and_device(
        condition, teacher.preprocessors["text_encoder"].precision_model, "causal_wan.text_encoder.encode"
    )
    if isinstance(condition, tuple):
        condition = condition[0]
    condition = condition.to(device=device, dtype=dtype)

    # Negative condition for classifier-free guidance path
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        neg_condition = teacher.preprocessors["text_encoder"].encode([""])
    _assert_gpu_tree_dtype_and_device(
        neg_condition, teacher.preprocessors["text_encoder"].precision_model, "causal_wan.text_encoder.encode_negative"
    )
    if isinstance(neg_condition, tuple):
        neg_condition = neg_condition[0]
    neg_condition = neg_condition.to(device=device, dtype=dtype)

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    assert output.device == x.device, f"Expected device {x.device}, got {output.device}"

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    # Standard forward pass without autocast
    output = teacher(x, t, condition=condition)

    # Forward with causal_ctx.store_kv=True (needed for autoregressive sampling caches)
    output_with_kv = teacher(
        x,
        t,
        condition=condition,
        causal_ctx=_make_causal_ctx(teacher, store_kv=True, is_ar=True),
    )
    assert output_with_kv.shape == x.shape, f"Expected shape {x.shape}, got {output_with_kv.shape}"

    # Test the sample method
    original_noise = torch.randn(batch_size, C, T, H, W, device=device, dtype=_STATE_DTYPE)
    with torch.no_grad():
        ar_output = teacher.sample(
            noise=original_noise,
            condition=condition,
            neg_condition=neg_condition,
        )

    assert_same(shape=True, dtype=True, device=True, original_noise=original_noise, ar_output=ar_output)

    # Decode-after-sample: exercise the sample()→vae.decode() contract once.
    _decode_smoke(teacher, ar_output)

    # Inhomogeneous timestep sampling and forward process
    t_inhom, step_indices = teacher.noise_scheduler.sample_student_timesteps_inhom(
        batch_size,
        T,
        teacher.chunk_size,
        num_steps=4,
    )
    assert step_indices.shape == (batch_size, T)
    t_inhom = t_inhom.to(device=device)
    t_inhom_reshaped = t_inhom[:, None, :, None, None]  # shape: (batch_size, 1, T, 1, 1) corresponds to (B,C,T,H,W)

    eps_inhom = torch.randn_like(x)
    noisy = teacher.noise_scheduler.forward_process(x, eps_inhom, t_inhom_reshaped)
    assert noisy.shape == x.shape and noisy.device == x.device and noisy.dtype == x.dtype

    # Network forward using inhomogeneous timesteps
    output_inhom = teacher(x, t_inhom, condition=condition)
    assert output_inhom.shape == x.shape

    # Test feature extraction path
    output_features = teacher(x, t, condition=condition, return_features_early=True, feature_indices={0})
    assert isinstance(output_features, list), "Feature extraction should return a list"

    # Verify chunk_size property
    assert hasattr(teacher, "chunk_size"), "CausalWan should have chunk_size attribute"
    assert teacher.chunk_size == 3, f"Expected chunk_size=3, got {teacher.chunk_size}"

    # Edge case: single frame
    single_frame_latents = torch.randn(batch_size, C, 1, H, W, device=device, dtype=_STATE_DTYPE)
    with torch.no_grad():
        single_frame_output = teacher.sample(
            noise=single_frame_latents,
            condition=condition,
            neg_condition=neg_condition,
        )
    assert single_frame_output.shape == single_frame_latents.shape

    # Frames with remainder when divided by chunk_size (5 frames, 3-per-chunk => remainder 2)
    odd_frames_latents = torch.randn(batch_size, C, 5, H, W, device=device, dtype=_STATE_DTYPE)
    with torch.no_grad():
        odd_output = teacher.sample(
            noise=odd_frames_latents,
            condition=condition,
            neg_condition=neg_condition,
        )
    assert odd_output.shape == odd_frames_latents.shape

    # Clear memory after memory-intensive test
    clear_gpu_memory()


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_wan22_5b_i2v():
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    set_env_vars()
    # Exercise the mixed-precision path: the VAE stays float32 while the transformer is
    # cast to bfloat16. ``WanVideoEncoder.encode`` must hand latents back in the caller's
    # dtype, or blending the first-frame latent promotes the input back to float32 and the
    # patch embedding rejects it.
    teacher_config = copy.deepcopy(Wan22_I2V_5B_Config)
    teacher_config.keep_enc_original_prec = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Wan config doesn't support model_channels/num_blocks override, use default
    teacher = instantiate(teacher_config)

    # only use a single block for testing
    teacher.transformer.blocks = teacher.transformer.blocks[:1]

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    num_frames, height, width = 5, 32, 32
    (
        T,
        H,
        W,
    ) = (
        (num_frames + 3) // 4,
        height // 16,
        width // 16,
    )

    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=_STATE_DTYPE)  # [B, C, T, H, W]

    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    # compute text encoder hidden states
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        text_embeds = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        text_embeds, teacher.preprocessors["text_encoder"].precision_model, "wan_i2v.text_encoder.encode"
    )

    # compute input for I2V models
    image = torch.zeros(batch_size, 3, height, width, device=device, dtype=_STATE_DTYPE)  # [B, C, H, W]
    image = image.unsqueeze(2)
    first_frame_cond = image
    first_frame_cond = first_frame_cond.to(device=device, dtype=_STATE_DTYPE)
    with _borrow_on_device(teacher.preprocessors["vae"], device):
        first_frame_cond = teacher.preprocessors["vae"].encode(first_frame_cond)
    _assert_gpu_tree_dtype_and_device(
        first_frame_cond, teacher.preprocessors["vae"].precision_model, "wan_i2v.vae.encode"
    )
    first_frame_cond = first_frame_cond.to(dtype=_STATE_DTYPE)

    # Handle case where text encoder returns tuple or needs device placement
    if isinstance(text_embeds, tuple):
        text_embeds = text_embeds[0]  # Take the first element if it's a tuple
    text_embeds = text_embeds.to(device=device, dtype=dtype)
    condition = dict(
        text_embeds=text_embeds,
        first_frame_cond=first_frame_cond,
    )

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    # Forward pass without autocast
    output = teacher(x, t, condition=condition)

    output = teacher(
        x,
        t,
        condition=condition,
        return_features_early=True,
        feature_indices={0},
    )

    assert isinstance(output, list)  # confirm output is a list

    for feature in output:
        # Use original model channels (not overridden)
        expected_channels = 768  # Original Wan channels
        assert feature.shape == (batch_size, expected_channels, T, H, W)

    # One-step sample() smoke (full guidance path; WanI2V first-frame
    # blending applied after each step).
    _smoke_sample(teacher, x, condition=condition, neg_condition=condition, decode_after=True)

    # Explicitly delete the 5B model to free GPU memory immediately
    del teacher

    # Clear memory after memory-intensive test
    clear_gpu_memory()


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_wancami2v_5b():
    """WanCamI2V (Wan2.2-TI2V-5B + camera Plücker conditioning).

    Covers the new ``cam`` network: the camera-aware ``build_conditioning``
    (per-frame poses/intrinsics -> ``c2ws_plucker_emb``), an autocast + plain
    forward with camera conditioning injected, and the ``guidance_cfg``
    ``sample()`` path. Camera layers are zero-init residuals, so a single
    block is enough to exercise the wiring.
    """
    from fastgen.networks.WanI2V.network_cam import WanCamI2V

    clear_gpu_memory()
    set_env_vars()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    teacher = WanCamI2V(
        model_id_or_local_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    )
    # Only use a single block for testing (camera layers were added per block).
    teacher.transformer.blocks = teacher.transformer.blocks[:1]
    teacher.init_preprocessors()
    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    assert hasattr(teacher.transformer, "camera_patch_embedding"), "camera layers must be added"

    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 16, width // 16
    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=_STATE_DTYPE)  # [B, C, T, H, W]
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        text_embeds = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        text_embeds, teacher.preprocessors["text_encoder"].precision_model, "wancam.text_encoder.encode"
    )
    if isinstance(text_embeds, tuple):
        text_embeds = text_embeds[0]
    text_embeds = text_embeds.to(device=device, dtype=dtype)

    # Network-owned conditioning: pixel ``real_raw`` + per-frame camera
    # poses/intrinsics -> structured condition (first_frame_cond + Plücker emb).
    real_raw = torch.zeros(batch_size, 3, num_frames, height, width, device=device, dtype=_STATE_DTYPE)
    camera_poses = torch.eye(4, device=device, dtype=dtype).reshape(1, 1, 4, 4).repeat(batch_size, num_frames, 1, 1)
    intrinsics = torch.tensor([width, width, width / 2, height / 2], device=device, dtype=dtype)
    camera_intrinsics = intrinsics.reshape(1, 1, 4).repeat(batch_size, num_frames, 1)
    data = {
        "condition": text_embeds,
        "neg_condition": text_embeds,
        "real_raw": real_raw,
        "camera_poses": camera_poses,
        "camera_intrinsics": camera_intrinsics,
    }
    with _borrow_on_device(teacher.preprocessors["vae"], device):
        built = teacher.build_conditioning(data, input_shape=[48, T, H, W])
    assert "c2ws_plucker_emb" in built["condition"], "build_conditioning must add camera Plücker emb"
    assert "c2ws_plucker_emb" in built["neg_condition"]
    condition = built["condition"]
    neg_condition = built["neg_condition"]

    # Autocasted forward with camera conditioning.
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)
    assert output.shape == x.shape
    assert output.device == x.device

    # Plain forward.
    output = teacher(x, t, condition=condition)
    assert output.shape == x.shape

    # One-step sample() smoke (guidance_cfg path + I2V first-frame pinning).
    _smoke_sample(teacher, x, condition=condition, neg_condition=neg_condition, decode_after=True)

    del teacher
    clear_gpu_memory()


@RunIf(min_gpus=1, min_gpu_mem=80)
@pytest.mark.large_model
def test_network_wan22_t2v_a14b_moe():
    """Wan2.2 T2V A14B MoE (``WanMoET2V``): routed forward + ``sample()`` smoke.

    Loads the two-expert A14B MoE, truncates each expert to a single block so it
    fits, then:
      * runs a ``batch_size=2`` forward with timesteps straddling
        ``guidance_boundary`` so both experts run via the batch-split path (one
        element each), and
      * runs a one-step ``sample()`` smoke. During sampling every batch element
        shares the step's timestep, so we enable the single-expert shortcut (the
        realistic inference path) which routes each step to one expert; the
        batch-split path is already covered by the forward above and by
        ``tests/test_network_fsdp.py``.
    """
    clear_gpu_memory()
    set_env_vars()

    from fastgen.configs.net import Wan22_T2V_14B_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    teacher = instantiate(Wan22_T2V_14B_Config)
    # Truncate each expert to a single block to keep the (2x ~14B) MoE testable.
    teacher.network_high.transformer.blocks = teacher.network_high.transformer.blocks[:1]
    teacher.network_low.transformer.blocks = teacher.network_low.transformer.blocks[:1]
    teacher.init_preprocessors()
    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 2
    T, H, W = 2, 4, 4
    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=_STATE_DTYPE)

    # Timesteps straddle guidance_boundary: element 0 -> high expert, element 1
    # -> low expert, so the batch-split path runs both experts.
    boundary = float(teacher.guidance_boundary)
    t = torch.tensor(
        [min(boundary + 0.05, 0.99), max(boundary - 0.2, 0.05)],
        device=device,
        dtype=teacher.noise_scheduler.t_precision,
    )

    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        condition = teacher.preprocessors["text_encoder"].encode(["a caption", "another caption"])
    _assert_gpu_tree_dtype_and_device(
        condition, teacher.preprocessors["text_encoder"].precision_model, "wan_moe.text_encoder.encode"
    )
    if isinstance(condition, tuple):
        condition = condition[0]
    condition = condition.to(device=device, dtype=dtype)

    # Routed forward with both experts active.
    output = teacher(x, t, condition=condition)
    assert output.shape == x.shape
    assert output.device == x.device

    # One-step sample() smoke via the single-expert shortcut (inference path).
    teacher.enable_single_expert_shortcut = True
    noise = torch.randn(batch_size, 16, T, H, W, device=device, dtype=_STATE_DTYPE)
    _smoke_sample(teacher, noise, condition=condition, neg_condition=condition)

    del teacher
    clear_gpu_memory()


def test_lingbot_world_remap_keys():
    """LingBot-world checkpoint key remapping (CPU-only unit test, no download).

    The official LingBot-world checkpoint names its camera layers
    ``patch_embedding_wancamctrl`` / ``c2ws_hidden_states_layer{1,2}``; these
    must be renamed to FastGen's ``camera_patch_embedding`` /
    ``camera_hidden_layer{1,2}`` so the weights land on the ``WanCamI2V``
    experts. Exercises ``LingBotWorld._remap_lingbot_keys`` directly -- it only
    reads the class-level ``LINGBOT_CAMERA_KEY_MAPPING``, so no (multi-GB) model
    construction or HF download is required.
    """
    from fastgen.networks.WanI2V.wan_moe import LingBotWorld

    # Stub carrying just the class-level mapping the method relies on.
    stub = SimpleNamespace(LINGBOT_CAMERA_KEY_MAPPING=LingBotWorld.LINGBOT_CAMERA_KEY_MAPPING)

    w_patch = torch.randn(2, 3)
    w_l1 = torch.randn(2, 2)
    b_l2 = torch.randn(2)
    w_attn = torch.randn(2, 2)
    state_dict = {
        "transformer.blocks.0.patch_embedding_wancamctrl.weight": w_patch,
        "transformer.blocks.0.c2ws_hidden_states_layer1.weight": w_l1,
        "transformer.blocks.0.c2ws_hidden_states_layer2.bias": b_l2,
        # Unrelated key: must pass through untouched.
        "transformer.blocks.0.attn1.to_q.weight": w_attn,
    }

    remapped = LingBotWorld._remap_lingbot_keys(stub, state_dict)

    # Camera keys are renamed to the WanCamI2V layer names.
    assert "transformer.blocks.0.camera_patch_embedding.weight" in remapped
    assert "transformer.blocks.0.camera_hidden_layer1.weight" in remapped
    assert "transformer.blocks.0.camera_hidden_layer2.bias" in remapped
    # Old LingBot names are gone.
    assert "transformer.blocks.0.patch_embedding_wancamctrl.weight" not in remapped
    assert "transformer.blocks.0.c2ws_hidden_states_layer1.weight" not in remapped
    assert "transformer.blocks.0.c2ws_hidden_states_layer2.bias" not in remapped
    # Unrelated keys are preserved, and no keys are dropped or aliased.
    assert "transformer.blocks.0.attn1.to_q.weight" in remapped
    assert len(remapped) == len(state_dict)
    # Tensor values are carried over unchanged (renaming only touches keys).
    assert torch.equal(remapped["transformer.blocks.0.camera_patch_embedding.weight"], w_patch)
    assert torch.equal(remapped["transformer.blocks.0.camera_hidden_layer1.weight"], w_l1)
    assert torch.equal(remapped["transformer.blocks.0.attn1.to_q.weight"], w_attn)


def test_lingbot_world_loads_expert_weights_from_a_local_directory(monkeypatch, tmp_path):
    """A local LingBot-world checkout is read from disk (index and shards of each expert subfolder), never the Hub."""
    import json

    import huggingface_hub
    from safetensors.torch import save_file

    from fastgen.networks.WanI2V import wan_moe

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *args, **kwargs: pytest.fail("local loads use no Hub")
    )
    shards = {
        "diffusion_pytorch_model-00001-of-00002.safetensors": {
            "patch_embedding_wancamctrl.weight": torch.randn(2, 2),
            "patch_embedding_wancamctrl.bias": torch.randn(2),
        },
        "diffusion_pytorch_model-00002-of-00002.safetensors": {
            "blocks.0.weight": torch.randn(2, 2),
            "blocks.0.bias": torch.randn(2),
        },
    }
    subfolder = tmp_path / "high_noise_model"
    subfolder.mkdir()
    for shard, tensors in shards.items():
        save_file(tensors, str(subfolder / shard))
    weight_map = {name: shard for shard, tensors in shards.items() for name in tensors}
    (subfolder / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    expert = torch.nn.Module()
    expert.camera_patch_embedding = torch.nn.Linear(2, 2)
    expert.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    network = wan_moe.LingBotWorld.__new__(wan_moe.LingBotWorld)
    torch.nn.Module.__init__(network)
    load_info = network._load_expert_weights(expert, str(tmp_path), checkpoint_subfolder="high_noise_model")

    assert load_info.missing_keys == [] and load_info.unexpected_keys == []
    camera, block = shards.values()
    torch.testing.assert_close(expert.camera_patch_embedding.weight, camera["patch_embedding_wancamctrl.weight"])
    torch.testing.assert_close(expert.camera_patch_embedding.bias, camera["patch_embedding_wancamctrl.bias"])
    torch.testing.assert_close(expert.blocks[0].weight, block["blocks.0.weight"])
    torch.testing.assert_close(expert.blocks[0].bias, block["blocks.0.bias"])


@RunIf(min_gpus=1, min_gpu_mem=80)
@pytest.mark.large_model
def test_network_wan21_14b_i2v():
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    set_env_vars()
    teacher_config = Wan21_I2V_14B_480P_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Wan config doesn't support model_channels/num_blocks override, use default
    teacher = instantiate(teacher_config)

    # only use a single block for testing
    teacher.transformer.blocks = teacher.transformer.blocks[:1]

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    num_frames, height, width = 5, 32, 32
    (
        T,
        H,
        W,
    ) = (
        (num_frames + 3) // 4,
        height // 8,
        width // 8,
    )

    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=_STATE_DTYPE)  # [B, C, T, H, W]

    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    # compute text encoder hidden states
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        text_embeds = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        text_embeds, teacher.preprocessors["text_encoder"].precision_model, "wan_i2v.text_encoder.encode"
    )

    # compute image encoder hidden states
    image = torch.zeros(batch_size, 3, height, width, device=device, dtype=_STATE_DTYPE)  # [B, C, H, W]
    with _borrow_on_device(teacher.preprocessors["image_encoder"], device):
        encoder_hidden_states_image = teacher.preprocessors["image_encoder"].encode(image)
    _assert_gpu_tree_dtype_and_device(
        encoder_hidden_states_image,
        teacher.preprocessors["image_encoder"].precision_model,
        "wan_i2v.image_encoder.encode",
    )
    encoder_hidden_states_image = encoder_hidden_states_image.to(dtype=dtype)

    # compute input for I2V models
    image = image.unsqueeze(2)
    first_frame_cond = image
    # wan 2.1 pads the zero tensor after the first frame
    first_frame_cond = torch.cat(
        [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
    )
    first_frame_cond = first_frame_cond.to(device=device, dtype=_STATE_DTYPE)
    with _borrow_on_device(teacher.preprocessors["vae"], device):
        first_frame_cond = teacher.preprocessors["vae"].encode(first_frame_cond)
    _assert_gpu_tree_dtype_and_device(
        first_frame_cond, teacher.preprocessors["vae"].precision_model, "wan_i2v.vae.encode"
    )
    first_frame_cond = first_frame_cond.to(dtype=_STATE_DTYPE)

    # Handle case where text encoder returns tuple or needs device placement
    if isinstance(text_embeds, tuple):
        text_embeds = text_embeds[0]  # Take the first element if it's a tuple
    text_embeds = text_embeds.to(device=device, dtype=dtype)
    condition = dict(
        text_embeds=text_embeds,
        first_frame_cond=first_frame_cond,
        encoder_hidden_states_image=encoder_hidden_states_image,
    )

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    # Forward pass without autocast
    output = teacher(x, t, condition)

    output = teacher(
        x,
        t,
        condition=condition,
        return_features_early=True,
        feature_indices={0},
    )

    assert isinstance(output, list)  # confirm output is a list

    for feature in output:
        # Use original model channels (not overridden)
        expected_channels = 1280  # Original Wan channels
        assert feature.shape == (batch_size, expected_channels, T, H, W)

    # One-step sample() smoke (full guidance path; WanI2V first-frame
    # blending applied after each step).
    _smoke_sample(teacher, x, condition=condition, neg_condition=condition, decode_after=True)

    # Explicitly delete the large 14B model to free GPU memory immediately
    del teacher

    # Clear memory after memory-intensive test
    clear_gpu_memory()


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_causal_wan22_5b_i2v():
    """
    Test CausalWanI2V network with Wan 2.2 TI2V 5B model.
    Tests forward pass, sample, and causal-specific features.
    """
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    set_env_vars()
    teacher_config = CausalWan22_I2V_5B_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    teacher = instantiate(teacher_config)

    # Only use a single block for testing
    teacher.transformer.blocks = teacher.transformer.blocks[:1]

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # Test noise scheduler properties
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 16, width // 16

    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=_STATE_DTYPE)  # [B, C, T, H, W]

    # CausalWanI2V supports 2D timesteps with shape (batch_size, num_frames)
    t_1d = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)
    t = t_1d.unsqueeze(1).expand(batch_size, T)  # Shape: (batch_size, T)

    # Compute text encoder hidden states
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        text_embeds = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        text_embeds, teacher.preprocessors["text_encoder"].precision_model, "causal_wan_i2v.text_encoder.encode"
    )

    # Compute input for I2V models (Wan 2.2 uses first frame replacement, not mask concat)
    image = torch.zeros(batch_size, 3, height, width, device=device, dtype=_STATE_DTYPE)  # [B, C, H, W]
    image = image.unsqueeze(2)
    first_frame_cond = image.to(device=device, dtype=_STATE_DTYPE)
    with _borrow_on_device(teacher.preprocessors["vae"], device):
        first_frame_cond = teacher.preprocessors["vae"].encode(first_frame_cond)
    _assert_gpu_tree_dtype_and_device(
        first_frame_cond, teacher.preprocessors["vae"].precision_model, "causal_wan_i2v.vae.encode"
    )
    first_frame_cond = first_frame_cond.to(dtype=_STATE_DTYPE)

    if isinstance(text_embeds, tuple):
        text_embeds = text_embeds[0]
    text_embeds = text_embeds.to(device=device, dtype=dtype)
    condition = dict(
        text_embeds=text_embeds,
        first_frame_cond=first_frame_cond,
    )

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    assert output.device == x.device, f"Expected device {x.device}, got {output.device}"

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    # Forward with a causal context that stores autoregressive caches.
    output_with_kv = teacher(x, t, condition, causal_ctx=_make_causal_ctx(teacher, store_kv=True, is_ar=True))
    assert output_with_kv.shape == x.shape, f"Expected shape {x.shape}, got {output_with_kv.shape}"

    # Clear caches before sample
    teacher.clear_caches()

    # Prepare negative condition for classifier-free guidance
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        neg_text_embeds = teacher.preprocessors["text_encoder"].encode([""])
    _assert_gpu_tree_dtype_and_device(
        neg_text_embeds,
        teacher.preprocessors["text_encoder"].precision_model,
        "causal_wan_i2v.text_encoder.encode_negative",
    )
    if isinstance(neg_text_embeds, tuple):
        neg_text_embeds = neg_text_embeds[0]
    neg_text_embeds = neg_text_embeds.to(device=device, dtype=dtype)
    neg_condition = dict(
        text_embeds=neg_text_embeds,
        first_frame_cond=first_frame_cond,
    )

    # Test sample method
    original_noise = torch.randn(batch_size, 48, T, H, W, device=device, dtype=_STATE_DTYPE)
    with torch.no_grad():
        ar_output = teacher.sample(
            noise=original_noise,
            condition=condition,
            neg_condition=neg_condition,
            sample_steps=2,  # Use fewer steps for testing
        )

    assert_same(shape=True, dtype=True, device=True, original_noise=original_noise, ar_output=ar_output)

    # Decode-after-sample: exercise the sample()→vae.decode() contract once.
    _decode_smoke(teacher, ar_output)

    # Test feature extraction
    output_features = teacher(x, t, condition, return_features_early=True, feature_indices={0})
    assert isinstance(output_features, list), "Feature extraction should return a list"

    for feature in output_features:
        expected_channels = 768  # Wan 2.2 5B model channels
        assert feature.shape == (batch_size, expected_channels, T, H, W)

    # Verify chunk_size property
    assert hasattr(teacher, "chunk_size"), "CausalWanI2V should have chunk_size attribute"
    assert teacher.chunk_size == 3, f"Expected chunk_size=3, got {teacher.chunk_size}"

    # Explicitly delete to free GPU memory
    del teacher

    # Clear memory after testing
    clear_gpu_memory()


@RunIf(min_gpus=1, min_gpu_mem=80)
@pytest.mark.large_model
@pytest.mark.parametrize(
    "teacher_config", [CausalWan21_I2V_14B_480P_Config, CausalWan21_I2V_14B_720P_Config], ids=["480p", "720p"]
)
def test_network_causal_wan21_14b_i2v(teacher_config):
    """
    Test CausalWanI2V network with the Wan 2.1 I2V 14B 480P and 720P models.
    Tests forward pass, sample, and causal-specific features.
    """
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    set_env_vars()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    teacher = instantiate(teacher_config)

    # Only use a single block for testing
    teacher.transformer.blocks = teacher.transformer.blocks[:1]

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # Test noise scheduler properties
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 8, width // 8

    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=_STATE_DTYPE)  # [B, C, T, H, W]

    # CausalWanI2V supports 2D timesteps with shape (batch_size, num_frames)
    t_1d = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)
    t = t_1d.unsqueeze(1).expand(batch_size, T)  # Shape: (batch_size, T)

    # Compute text encoder hidden states
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        text_embeds = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        text_embeds, teacher.preprocessors["text_encoder"].precision_model, "causal_wan_i2v.text_encoder.encode"
    )

    # Compute image encoder hidden states (required for Wan 2.1 14B models)
    image = torch.zeros(batch_size, 3, height, width, device=device, dtype=_STATE_DTYPE)  # [B, C, H, W]
    with _borrow_on_device(teacher.preprocessors["image_encoder"], device):
        encoder_hidden_states_image = teacher.preprocessors["image_encoder"].encode(image)
    _assert_gpu_tree_dtype_and_device(
        encoder_hidden_states_image,
        teacher.preprocessors["image_encoder"].precision_model,
        "causal_wan_i2v.image_encoder.encode",
    )
    encoder_hidden_states_image = encoder_hidden_states_image.to(dtype=dtype)

    # Compute input for I2V models (Wan 2.1 uses mask concatenation)
    image = image.unsqueeze(2)
    # Wan 2.1 pads zero tensor after the first frame
    first_frame_cond = torch.cat(
        [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
    )
    first_frame_cond = first_frame_cond.to(device=device, dtype=_STATE_DTYPE)
    with _borrow_on_device(teacher.preprocessors["vae"], device):
        first_frame_cond = teacher.preprocessors["vae"].encode(first_frame_cond)
    _assert_gpu_tree_dtype_and_device(
        first_frame_cond, teacher.preprocessors["vae"].precision_model, "causal_wan_i2v.vae.encode"
    )
    first_frame_cond = first_frame_cond.to(dtype=_STATE_DTYPE)

    if isinstance(text_embeds, tuple):
        text_embeds = text_embeds[0]
    text_embeds = text_embeds.to(device=device, dtype=dtype)
    condition = dict(
        text_embeds=text_embeds,
        first_frame_cond=first_frame_cond,
        encoder_hidden_states_image=encoder_hidden_states_image,
    )

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    assert output.device == x.device, f"Expected device {x.device}, got {output.device}"

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    output = teacher(x, t, condition)

    # Forward with a causal context that stores autoregressive caches.
    output_with_kv = teacher(x, t, condition, causal_ctx=_make_causal_ctx(teacher, store_kv=True, is_ar=True))
    assert output_with_kv.shape == x.shape, f"Expected shape {x.shape}, got {output_with_kv.shape}"

    # Clear caches before sample
    teacher.clear_caches()

    # Prepare negative condition for classifier-free guidance
    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        neg_text_embeds = teacher.preprocessors["text_encoder"].encode([""])
    _assert_gpu_tree_dtype_and_device(
        neg_text_embeds,
        teacher.preprocessors["text_encoder"].precision_model,
        "causal_wan_i2v.text_encoder.encode_negative",
    )
    if isinstance(neg_text_embeds, tuple):
        neg_text_embeds = neg_text_embeds[0]
    neg_text_embeds = neg_text_embeds.to(device=device, dtype=dtype)
    neg_condition = dict(
        text_embeds=neg_text_embeds,
        first_frame_cond=first_frame_cond,
        encoder_hidden_states_image=encoder_hidden_states_image,
    )

    # Test sample method
    original_noise = torch.randn(batch_size, 16, T, H, W, device=device, dtype=_STATE_DTYPE)
    with torch.no_grad():
        ar_output = teacher.sample(
            noise=original_noise,
            condition=condition,
            neg_condition=neg_condition,
            sample_steps=2,  # Use fewer steps for testing
        )

    assert_same(shape=True, dtype=True, device=True, original_noise=original_noise, ar_output=ar_output)

    # Decode-after-sample: exercise the sample()→vae.decode() contract once.
    _decode_smoke(teacher, ar_output)

    # Test feature extraction
    output_features = teacher(x, t, condition, return_features_early=True, feature_indices={0})
    assert isinstance(output_features, list), "Feature extraction should return a list"

    for feature in output_features:
        expected_channels = 1280  # Wan 2.1 14B model channels
        assert feature.shape == (batch_size, expected_channels, T, H, W)

    # Verify chunk_size property
    assert hasattr(teacher, "chunk_size"), "CausalWanI2V should have chunk_size attribute"
    assert teacher.chunk_size == 3, f"Expected chunk_size=3, got {teacher.chunk_size}"

    # Explicitly delete to free GPU memory
    del teacher

    # Clear memory after testing
    clear_gpu_memory()


def test_ltx2_text_encoder_returns_raw_flattened_hidden_states():
    from fastgen.networks.LTX2.network import LTX2TextEncoder

    class _TextEncoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask, output_hidden_states, return_dict, logits_to_keep=0):
            assert attention_mask.shape == input_ids.shape
            assert output_hidden_states and return_dict
            assert logits_to_keep == 1  # the vocabulary logits are never read
            hidden_states = input_ids.to(torch.float32).unsqueeze(-1)
            return SimpleNamespace(hidden_states=(hidden_states, hidden_states + 10))

    encoder = object.__new__(LTX2TextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.text_encoder = _TextEncoder()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 0]])

    prompt_embeds, returned_mask = encoder(input_ids, attention_mask)

    hidden_states = input_ids.to(torch.float32).unsqueeze(-1)
    expected = torch.stack((hidden_states, hidden_states + 10), dim=-1).flatten(2, 3)
    assert torch.equal(prompt_embeds, expected)
    assert returned_mask is attention_mask


def test_ltx2_sigma_schedule_matches_ltx_core_bit_for_bit():
    """The sampling grid is ltx_core's ``LTX2Scheduler`` sigmas exactly, on the network's own noise schedule."""
    from fastgen.networks.LTX2.network import _ltx2_sigma_schedule

    def execute(steps, max_shift=2.05, base_shift=0.95, stretch=True, terminal=0.1, default_number_of_tokens=4096):
        # Verbatim from ltx_core's ``LTX2Scheduler.execute``, minus its ``latent`` argument.
        tokens = default_number_of_tokens
        sigmas = torch.linspace(1.0, 0.0, steps + 1)

        x1 = 1024  # BASE_SHIFT_ANCHOR
        x2 = 4096  # MAX_SHIFT_ANCHOR
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = (tokens) * mm + b

        power = 1
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        # Stretch sigmas so that its final value matches the given terminal value.
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        return sigmas.to(torch.float32)

    for steps in range(2, 201):
        assert torch.equal(_ltx2_sigma_schedule(steps), execute(steps)), steps
    assert torch.equal(
        _ltx2_sigma_schedule(30, max_shift=1.5, base_shift=0.5, image_seq_len=2048),
        execute(30, max_shift=1.5, base_shift=0.5, default_number_of_tokens=2048),
    )
    assert torch.equal(_ltx2_sigma_schedule(30, stretch=False), execute(30, stretch=False))
    with pytest.raises(ValueError, match="sample_steps >= 2"):
        _ltx2_sigma_schedule(1)

    network = _ltx2_i2v_sample_stub()
    network.noise_scheduler = RFNoiseSchedule(max_t=1.0)
    ns, grid = network.sample_scheduler_and_grid(30)
    assert ns is network.noise_scheduler
    assert grid.dtype == ns.t_precision and torch.equal(grid, execute(30).to(ns.t_precision))

    # Only the query times are clamped to the schedule; the clean end stays at 0.
    network.noise_scheduler = RFNoiseSchedule(min_t=0.2, max_t=0.9)
    _, grid = network.sample_scheduler_and_grid(30)
    assert grid[0] == 0.9 and grid[-2] == 0.2 and grid[-1] == 0


@pytest.mark.parametrize(
    "model_id, override, expected",
    [
        ("Lightricks/LTX-2.5-Diffusers", None, "transformer_full"),
        ("diffusers/LTX-2.3-Diffusers", None, "transformer"),
        ("Lightricks/LTX-2.5-Diffusers", "transformer", "transformer"),
    ],
)
def test_ltx2_transformer_subfolder_is_version_aware(model_id, override, expected):
    from fastgen.networks.LTX2.network import _ltx2_transformer_subfolder

    assert _ltx2_transformer_subfolder(model_id, override) == expected


def test_ltx2_infer_shape_preserves_the_unimodal_tensor_contract():
    from fastgen.networks.LTX2 import network as ltx2_network

    network = object.__new__(ltx2_network.LTX2)
    torch.nn.Module.__init__(network)
    network.modalities = ("video",)
    shape = [128, 5, 8, 12]

    assert network.infer_shape(shape) is shape

    network.modalities = ("video", "audio")
    with pytest.raises(AssertionError, match="exactly one"):
        network.infer_shape(shape)


def test_ltx2_config_freezes_connectors_by_default():
    assert LTX2Config.train_connectors is False


def test_ltx2_fully_shard_wraps_connector_blocks_before_their_parents(monkeypatch):
    """Connector collectives stay bounded by leaf-first FSDP2 wrapping."""
    from fastgen.networks.LTX2 import network as ltx2_network

    class Connector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
            self.learnable_registers = torch.nn.Parameter(torch.zeros(2))

    transformer = torch.nn.Module()
    transformer.gradient_checkpointing = False
    transformer.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    connectors = torch.nn.ModuleDict(
        {
            "video_text_proj_in": torch.nn.Linear(2, 2),
            "video_connector": Connector(),
            "audio_connector": Connector(),
        }
    )
    network = object.__new__(ltx2_network.LTX2)
    torch.nn.Module.__init__(network)
    network.train_connectors = True
    network.transformer = transformer
    network.connectors = connectors

    wrapped = []
    monkeypatch.setattr(ltx2_network, "fully_shard", lambda module, **kwargs: wrapped.append((module, kwargs)))
    network.fully_shard(mesh="mesh", mp_policy="policy")

    expected = [
        transformer.transformer_blocks[0],
        transformer,
        connectors["video_text_proj_in"],
        *connectors["video_connector"].transformer_blocks,
        connectors["video_connector"],
        *connectors["audio_connector"].transformer_blocks,
        connectors["audio_connector"],
        connectors,
    ]
    assert [module for module, _ in wrapped] == expected
    assert all(kwargs == {"mesh": "mesh", "mp_policy": "policy"} for _, kwargs in wrapped)


def test_ltx2_frozen_connectors_preserve_dtype_and_use_bf16_fsdp(monkeypatch):
    from fastgen.networks.LTX2 import network as ltx2_network
    import fastgen.networks.network as network_module
    from torch.distributed.fsdp import MixedPrecisionPolicy

    transformer = torch.nn.Module()
    transformer.gradient_checkpointing = False
    transformer.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2, dtype=torch.float32)])
    connectors = torch.nn.ModuleDict({"video_connector": torch.nn.Linear(2, 2, dtype=torch.bfloat16)})
    network = object.__new__(ltx2_network.LTX2)
    torch.nn.Module.__init__(network)
    network.train_connectors = False
    network.transformer = transformer
    network.connectors = connectors
    network._freeze_connectors_()

    # Generic student setup must neither warn about nor re-enable protected connector gradients.
    # The explicit master-weight precision boundary must also preserve checkpoint dtype.
    warnings = []
    monkeypatch.setattr(network_module.logger, "warning", warnings.append)
    network._set_precision(
        precision_model=torch.bfloat16,
        precision_opt=torch.float64,
    )
    network._set_train_mode(True)
    assert warnings == []
    assert network.transformer.transformer_blocks[0].weight.dtype == torch.float64
    assert network.connectors["video_connector"].weight.dtype == torch.bfloat16
    assert not network.connectors.training
    assert not any(parameter.requires_grad for parameter in network.connectors.parameters())

    wrapped = []
    monkeypatch.setattr(ltx2_network, "fully_shard", lambda module, **kwargs: wrapped.append((module, kwargs)))
    policy = MixedPrecisionPolicy(param_dtype=torch.float64, reduce_dtype=torch.float32)
    network.fully_shard(mesh="mesh", mp_policy=policy)

    assert [module for module, _ in wrapped] == [
        network.transformer.transformer_blocks[0],
        network.transformer,
        network.connectors["video_connector"],
        network.connectors,
    ]
    assert wrapped[0][1]["mp_policy"] is policy
    assert wrapped[1][1]["mp_policy"] is policy
    for _, connector_kwargs in wrapped[2:]:
        assert connector_kwargs["mp_policy"].param_dtype == torch.bfloat16
        assert connector_kwargs["mp_policy"].reduce_dtype == torch.float32
    assert not any(parameter.requires_grad for parameter in network.connectors.parameters())


def test_ltx2_set_precision_keeps_frozen_connectors_bfloat16():
    """Frozen connectors keep the model precision; trainable connectors follow ``precision_opt``."""
    from fastgen.networks.LTX2.network import LTX2

    network = object.__new__(LTX2)
    torch.nn.Module.__init__(network)
    network.transformer = torch.nn.Linear(2, 2)
    network.connectors = torch.nn.ModuleDict({"video": torch.nn.Linear(2, 2)})
    network.connectors.requires_grad_(False)
    network.train_connectors = False

    network._set_precision(precision_model=torch.bfloat16, precision_opt=torch.float64)

    assert network.transformer.weight.dtype == torch.float64
    assert network.connectors["video"].weight.dtype == torch.bfloat16

    network.train_connectors = True
    network._set_precision(precision_model=torch.bfloat16, precision_opt=torch.float32)
    assert network.connectors["video"].weight.dtype == torch.float32


def test_ltx2_classify_forward_preserves_conditioning_dtype_under_autocast():
    from fastgen.networks.LTX2 import network as ltx2_network

    class CaptureBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtypes = {}

        def forward(self, **kwargs):
            for name, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    self.dtypes[name] = value.dtype
                elif isinstance(value, tuple) and value and all(isinstance(item, torch.Tensor) for item in value):
                    self.dtypes[name] = tuple(item.dtype for item in value)
            return kwargs["hidden_states"], kwargs["audio_hidden_states"]

    transformer = ltx2_network.LTX2VideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        num_layers=0,
        qk_norm="rms_norm_across_heads",
        caption_channels=16,
        rope_double_precision=False,
        cross_attn_mod=True,
        audio_cross_attn_mod=True,
        use_prompt_embeddings=False,
        use_prompt_adaln_single=True,
    ).eval()
    block = CaptureBlock()
    transformer.transformer_blocks = torch.nn.ModuleList([block])
    transformer._fastgen_ltx2_version = "2.3"
    assert {parameter.dtype for parameter in transformer.parameters()} == {torch.float32}

    module_output_dtypes = {}

    def capture_output_dtypes(name):
        def hook(_module, _args, output):
            module_output_dtypes[name] = tuple(value.dtype for value in output)

        return hook

    autocast_modules = (
        "time_embed",
        "audio_time_embed",
        "prompt_adaln",
        "audio_prompt_adaln",
        "av_cross_attn_video_scale_shift",
        "av_cross_attn_video_a2v_gate",
        "av_cross_attn_audio_scale_shift",
        "av_cross_attn_audio_v2a_gate",
    )
    for name in autocast_modules:
        getattr(transformer, name).register_forward_hook(capture_output_dtypes(name))

    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        video, audio, logvar, features = ltx2_network.classify_forward(
            transformer,
            hidden_states=torch.randn(1, 1, 4),
            audio_hidden_states=torch.randn(1, 1, 4),
            encoder_hidden_states=torch.randn(1, 2, 16),
            audio_encoder_hidden_states=torch.randn(1, 2, 8),
            timestep=torch.tensor([[500.0]]),
            audio_timestep=torch.tensor([[400.0]]),
            sigma=torch.tensor([500.0]),
            audio_sigma=torch.tensor([400.0]),
            encoder_attention_mask=torch.tensor([[1, 0]], dtype=torch.long),
            audio_encoder_attention_mask=torch.tensor([[1, 0]], dtype=torch.long),
            num_frames=1,
            height=1,
            width=1,
            audio_num_frames=1,
        )

    tensor_inputs = (
        "hidden_states",
        "audio_hidden_states",
        "encoder_attention_mask",
        "audio_encoder_attention_mask",
        "temb",
        "temb_audio",
        "temb_ca_scale_shift",
        "temb_ca_audio_scale_shift",
        "temb_ca_gate",
        "temb_ca_audio_gate",
        "temb_prompt",
        "temb_prompt_audio",
    )
    assert all(block.dtypes[name] == torch.bfloat16 for name in tensor_inputs)
    assert block.dtypes["encoder_hidden_states"] == torch.float32
    assert block.dtypes["audio_encoder_hidden_states"] == torch.float32
    rotary_inputs = (
        "video_rotary_emb",
        "audio_rotary_emb",
        "ca_video_rotary_emb",
        "ca_audio_rotary_emb",
    )
    assert all(block.dtypes[name] == (torch.bfloat16, torch.bfloat16) for name in rotary_inputs)
    assert set(module_output_dtypes) == set(autocast_modules)
    assert all(dtypes == (torch.bfloat16, torch.bfloat16) for dtypes in module_output_dtypes.values())
    assert video.dtype == audio.dtype == torch.bfloat16
    assert logvar is None and features is None
    assert {parameter.dtype for parameter in transformer.parameters()} == {torch.float32}


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_ltx2_i2v_condition_unpacking_requires_float32_strength(state_dtype):
    from fastgen.networks.LTX2.network import LTX2

    state = torch.randn(1, 2, 5, 2, 2, dtype=state_dtype)
    raw_clean = torch.randn_like(state)
    strength = torch.tensor([0.0, 0.4, 0.75, 1.0, 1.0 - 2**-24], dtype=torch.float32)
    saved_strength = strength.clone()
    condition = {
        "text_embeds": None,
        "frame_cond": raw_clean,
        "frame_idx": torch.arange(5),
        "strength": strength,
    }
    frame_idx, clean, weight = LTX2._unpack_i2v_condition(condition)
    assert frame_idx.device == raw_clean.device
    assert clean is raw_clean and clean.dtype == state.dtype
    assert weight.dtype == torch.float32
    torch.testing.assert_close(weight.flatten(), 1.0 - strength, rtol=0, atol=0)
    assert torch.equal(strength, saved_strength)
    blend_dtype = torch.promote_types(state_dtype, torch.float32)
    promoted_weight = weight.to(blend_dtype)
    expected = (state.to(blend_dtype) * promoted_weight + clean.to(blend_dtype) * (1 - promoted_weight)).to(state_dtype)
    actual = LTX2._blend_with_clean(state, clean, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    default_idx, _, default_weight = LTX2._unpack_i2v_condition(
        {"text_embeds": None, "frame_cond": raw_clean[:, :, :1]}
    )
    assert default_idx.dtype == torch.long
    assert default_idx.device == raw_clean.device
    assert default_weight.dtype == torch.float32
    assert default_weight.device == raw_clean.device

    for bad_strength in (strength.half(), strength.bfloat16(), strength.double(), strength.long(), strength.bool()):
        with pytest.raises(AssertionError, match=r"strength\.dtype .*must be torch\.float32"):
            LTX2._unpack_i2v_condition({**condition, "strength": bad_strength})
    for bad_frame_idx in (torch.arange(5.0), torch.arange(5, dtype=torch.int16)):
        with pytest.raises(AssertionError, match=r"frame_idx\.dtype .*must be torch\.int64"):
            LTX2._unpack_i2v_condition({**condition, "frame_idx": bad_frame_idx})


def test_ltx2_i2v_condition_unpacking_accepts_a_per_sample_strength():
    """A ``[B, N]`` strength keeps its batch axis; a 1-D strength is always one value per conditioned frame."""
    from fastgen.networks.LTX2.network import LTX2

    strength = torch.tensor([[0.0, 0.4, 1.0], [0.75, 1.0, 0.5]])
    condition = {
        "text_embeds": None,
        "frame_cond": torch.randn(2, 2, 3, 2, 2),
        "frame_idx": torch.arange(3),
        "strength": strength,
    }
    _, _, weight = LTX2._unpack_i2v_condition(condition)
    assert weight.shape == (2, 1, 3, 1, 1)
    torch.testing.assert_close(weight.flatten(1), 1.0 - strength, rtol=0, atol=0)
    for bad_strength in (strength.T.contiguous(), strength[:, :2], strength[None], torch.ones(2), torch.ones(3, 3)):
        with pytest.raises(AssertionError, match="is not one of"):
            LTX2._unpack_i2v_condition({**condition, "strength": bad_strength})


@pytest.mark.parametrize("fwd_pred_type", ["x0", "eps", "flow"])
def test_ltx2_i2v_output_conditioning_handles_zero_sigma(fwd_pred_type):
    from fastgen.networks.LTX2.network import LTX2

    network = LTX2.__new__(LTX2)
    torch.nn.Module.__init__(network)
    network.net_pred_type = "flow"
    network.noise_scheduler = RFNoiseSchedule()

    torch.manual_seed(0)
    video_latents = torch.randn(2, 2, 3, 2, 2)
    model_output = torch.randn_like(video_latents)
    frame_idx = torch.tensor([0])
    cond_latent = torch.randn(2, 2, 1, 2, 2)
    t = torch.tensor([0.0, 0.5], dtype=network.noise_scheduler.t_precision)
    condition = {"text_embeds": None, "frame_cond": cond_latent, "frame_idx": frame_idx}

    def constrain(condition):
        return network._condition_output(
            model_output, x_t=video_latents, t=t, condition=condition, fwd_pred_type=fwd_pred_type
        )

    if fwd_pred_type != "x0":
        # No non-x0 prediction carries the clean frame at sigma = 0; a zero strength constrains nothing.
        with pytest.raises(ValueError, match="sigma = 0"):
            constrain(condition)
        assert torch.equal(constrain({**condition, "strength": torch.zeros(1)}), model_output)
        # Per sample, only the clean sample's strengths must be zero.
        assert torch.equal(constrain({**condition, "strength": torch.tensor([[0.0], [1.0]])})[0], model_output[0])
        with pytest.raises(ValueError, match="sigma = 0"):
            constrain({**condition, "strength": torch.tensor([[1.0], [0.0]])})
        return

    actual = constrain(condition)
    expected = model_output.clone()
    expected[:, :, frame_idx] = cond_latent
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[0, :, frame_idx], expected[0, :, frame_idx])
    assert torch.isfinite(actual).all()


def test_ltx2_forward_rejects_invalid_inputs_before_feature_early_exit():
    from fastgen.networks.LTX2.network import LTX2

    network = object.__new__(LTX2)
    torch.nn.Module.__init__(network)
    network.modalities = ("video", "audio")
    network.ltx2_version = "2.3"
    network.fps = 24.0
    network.net_pred_type = "flow"
    network.transformer = SimpleNamespace(transformer_blocks=[None])
    state = torch.zeros(1, 2, 1, 2, 2)
    timestep = torch.tensor([0.5])
    early_features = {"return_features_early": True, "feature_indices": {0}}

    with pytest.raises(ValueError, match="bare Tensor input requires exactly one modality"):
        network(state, timestep, **early_features)
    with pytest.raises(ValueError, match="non-empty subset"):
        network({}, timestep, **early_features)
    with pytest.raises(ValueError, match="non-empty subset"):
        network({"text": state}, timestep, **early_features)
    with pytest.raises(TypeError, match="must be a Tensor or dict"):
        network([state], timestep, **early_features)


def test_ltx2_sample_invalid_inputs_fail_in_downstream_contracts():
    from fastgen.networks.LTX2.network import LTX2

    network = object.__new__(LTX2)
    torch.nn.Module.__init__(network)
    network.modalities = ("video", "audio")
    network.ltx2_version = "2.3"
    network.fps = 24.0
    network.schedule_type = "rf"
    network.noise_scheduler = CompositeNoiseSchedule(
        schedules={"video": {"name": "rf"}, "audio": {"name": "rf"}},
        primary_key="video",
    )
    state = torch.zeros(1, 2, 1, 2, 2)

    with pytest.raises(ValueError, match="no Tensor leaf"):
        network.sample({}, sample_steps=2)
    with pytest.raises(ValueError, match="modalities must be a non-empty selection"):
        network.sample({"text": state}, sample_steps=2)
    with pytest.raises(TypeError, match="CompositeNoiseSchedule.latents.*requires dict-shaped"):
        network.sample(state, sample_steps=2)


def test_ltx2_joint_forward_preserves_input_dtypes_without_model_precision():
    from fastgen.networks.LTX2.network import LTX2

    class Connectors(torch.nn.Module):
        def forward(self, prompt_embeds, attention_mask):
            return prompt_embeds, prompt_embeds, attention_mask

    class Rope:
        @staticmethod
        def prepare_video_coords(*args, **kwargs):
            del args, kwargs
            return None

        @staticmethod
        def prepare_audio_coords(*args, **kwargs):
            del args, kwargs
            return None

    class Transformer(torch.nn.Module):
        transformer_blocks = ()

        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList()
            self.rope = Rope()
            self.audio_rope = Rope()
            self.seen_dtypes = None

        def forward(self, **kwargs):
            self.seen_dtypes = (
                kwargs["hidden_states"].dtype,
                kwargs["audio_hidden_states"].dtype,
                kwargs["encoder_hidden_states"].dtype,
                kwargs["audio_encoder_hidden_states"].dtype,
            )
            return kwargs["hidden_states"], kwargs["audio_hidden_states"], None, None

    class Schedule:
        @staticmethod
        def rescale_t(timestep):
            return {"video": timestep, "audio": timestep}

        @staticmethod
        def convert_model_output(state, model_output, timestep, **kwargs):
            del timestep, kwargs  # called with output_dtype=None: uncast, in at least HIGH_PRECISION
            return {
                name: value.to(dtype=high_precision_dtype(state[name], value)) for name, value in model_output.items()
            }

    network = object.__new__(LTX2)
    torch.nn.Module.__init__(network)
    network.modalities = ("video", "audio")
    network.fps = 24.0
    network.net_pred_type = "flow"
    network.train_connectors = True
    network.transformer_spatial_patch_size = 1
    network.transformer_temporal_patch_size = 1
    network.audio_vae_mel_bins = 4
    network.audio_vae_mel_compression_ratio = 1
    network.connectors = Connectors()
    network.transformer = Transformer()
    network.noise_scheduler = Schedule()
    network._record_precision(precision_model=None)

    state = {
        "video": torch.randn(1, 2, 1, 2, 2, dtype=torch.float32),
        "audio": torch.randn(1, 2, 3, 4, dtype=torch.float64),
    }
    prompt_embeds = torch.randn(1, 2, 4, dtype=torch.float64)
    attention_mask = torch.ones(1, 2, dtype=torch.long)
    output = network(
        state,
        torch.tensor([0.5]),
        condition=(prompt_embeds, attention_mask),
    )

    assert network.transformer.seen_dtypes == (torch.float32, torch.float64, torch.float64, torch.float64)
    assert output["video"].dtype == state["video"].dtype
    assert output["audio"].dtype == state["audio"].dtype


def test_ltx2_forward_promotes_blends_without_mutating_prepared_condition():
    from fastgen.networks.LTX2.network import LTX2

    class Connectors(torch.nn.Module):
        def forward(self, prompt_embeds, attention_mask):
            return prompt_embeds, prompt_embeds, attention_mask

    class Rope:
        @staticmethod
        def prepare_video_coords(batch_size, num_frames, height, width, device, fps):
            del batch_size, num_frames, height, width, device, fps
            return None

    class Transformer(torch.nn.Module):
        transformer_blocks = ()

        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList()
            self.rope = Rope()

        def forward(self, **kwargs):
            return kwargs["hidden_states"], None, None, None

    network = LTX2.__new__(LTX2)
    torch.nn.Module.__init__(network)
    network.modalities = ("video",)
    network.fps = 24.0
    network.net_pred_type = "flow"
    network._record_precision(precision_model=torch.bfloat16)
    network.train_connectors = True
    network.noise_scheduler = RFNoiseSchedule()
    network.transformer_spatial_patch_size = 1
    network.transformer_temporal_patch_size = 1
    network.connectors = Connectors()
    network.transformer = Transformer()

    unpacked_conditions = []
    blend_inputs = []
    original_unpack = network._unpack_i2v_condition
    original_blend = network._blend_with_clean

    def unpack_i2v_condition(condition):
        result = original_unpack(condition)
        unpacked_conditions.append(result)
        return result

    def blend_with_clean(x, clean, weight):
        result = original_blend(x, clean, weight)
        blend_inputs.append((x, clean, weight, result))
        return result

    network._unpack_i2v_condition = unpack_i2v_condition
    network._blend_with_clean = blend_with_clean

    state = torch.randn(1, 2, 2, 2, 2, dtype=torch.bfloat16)
    raw_clean = torch.randn(1, 2, 1, 2, 2, dtype=state.dtype)
    raw_strength = torch.tensor([0.4], dtype=torch.float32)
    condition = {
        "text_embeds": (
            torch.randn(1, 2, 3, dtype=network.connector_precision),
            torch.ones(1, 2, dtype=torch.long),
        ),
        "frame_cond": raw_clean,
        "frame_idx": torch.tensor([0]),
        "strength": raw_strength,
    }
    seen_timesteps = []
    handle = network.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: seen_timesteps.append(kwargs["timestep"].clone()), with_kwargs=True
    )
    output = network(
        state,
        torch.tensor([0.5], dtype=network.noise_scheduler.t_precision),
        condition=condition,
        fwd_pred_type="x0",
    )
    handle.remove()

    assert output.dtype == state.dtype
    _, aligned_clean, aligned_inv_strength = unpacked_conditions[0]
    assert aligned_clean is raw_clean
    assert aligned_clean.dtype == state.dtype
    assert aligned_inv_strength.dtype == raw_strength.dtype
    reference_mask = 1.0 - raw_strength
    expected_time = network.noise_scheduler.rescale_t(
        torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
    ) * reference_mask.to(network.noise_scheduler.t_precision)
    # Four tokens per frame; the prepared mask enters the scheduler's timestep dtype.
    torch.testing.assert_close(seen_timesteps[0][:, :4], expected_time.expand(1, 4), rtol=0, atol=0)
    # The input blend, its re-derivation for the per-token output conversion, and the output constraint.
    assert len(blend_inputs) == 3
    assert all(clean is aligned_clean for _, clean, _, _ in blend_inputs)
    assert all(torch.equal(weight, aligned_inv_strength) for _, _, weight, _ in blend_inputs)
    for blend_state, clean, weight, actual in blend_inputs:
        blend_dtype = torch.promote_types(blend_state.dtype, weight.dtype)
        expected = (blend_state.to(blend_dtype) * weight + clean.to(blend_dtype) * (1.0 - weight)).to(blend_state.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(clean is raw_clean for _, clean, _ in unpacked_conditions)

    # Frozen connectors read bf16 text whatever the transformer precision, so other text is rejected.
    network.train_connectors = False
    float_text = (condition["text_embeds"][0].float(), condition["text_embeds"][1])
    with pytest.raises(AssertionError, match=r"prompt_embeds\.dtype .*must be torch\.bfloat16"):
        network(
            state,
            torch.tensor([0.5], dtype=network.noise_scheduler.t_precision),
            condition={**condition, "text_embeds": float_text},
        )


def _ltx2_i2v_sample_stub(modalities=("video",)):
    """``LTX2`` whose stub video velocity depends on the prompt, so CFG and its rescale take effect."""
    from fastgen.networks.LTX2.network import LTX2

    class Connectors(torch.nn.Module):
        def forward(self, prompt_embeds, attention_mask):
            return prompt_embeds, prompt_embeds, attention_mask

    class Rope:
        @staticmethod
        def prepare_video_coords(batch_size, num_frames, height, width, device, fps):
            del batch_size, num_frames, height, width, device, fps
            return None

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList()
            self.rope = Rope()

        def forward(self, **kwargs):
            prompt_shift = kwargs["encoder_hidden_states"].mean(dim=(1, 2)).view(-1, 1, 1)
            return 0.5 * kwargs["hidden_states"] + prompt_shift, None, None, None

    network = LTX2.__new__(LTX2)
    torch.nn.Module.__init__(network)
    network.modalities = modalities
    network.fps = 24.0
    network.ltx2_version = "2.3"
    network.net_pred_type = "flow"
    network.schedule_type = "rf"
    network._record_precision(precision_model=torch.float32)
    network.train_connectors = True
    network.noise_scheduler = (
        RFNoiseSchedule()
        if len(modalities) == 1
        else CompositeNoiseSchedule(schedules={m: {"name": "rf"} for m in modalities}, primary_key="video")
    )
    network.transformer_spatial_patch_size = 1
    network.transformer_temporal_patch_size = 1
    network.connectors = Connectors()
    network.transformer = Transformer()
    return network


def test_ltx2_forward_validates_its_stg_layers_before_the_transformer():
    """STG layers are LTX-2's own forward control: blocks in range reach the transformer, others fail first."""
    network = _ltx2_i2v_sample_stub()
    network.transformer.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    perturbations = []
    network.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: perturbations.append(
            (kwargs["skip_self_attn_layers"], kwargs["skip_cross_modal_attn"])
        ),
        with_kwargs=True,
    )
    state = torch.randn(1, 2, 3, 2, 2)
    time = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
    condition = (torch.randn(1, 2, 3), torch.ones(1, 2, dtype=torch.long))
    network(state, time, condition, skip_self_attn_layers=[1], skip_cross_modal_attn=True)
    with pytest.raises(ValueError, match=r"skip_self_attn_layers contains out-of-range indices \[2\]"):
        network(state, time, condition, skip_self_attn_layers=[2])
    assert perturbations == [([1], True)]


@pytest.mark.parametrize("fwd_pred_type", ["x0", "flow"])
def test_ltx2_i2v_per_sample_strength_conditions_each_sample_as_its_row_alone(fwd_pred_type):
    """A ``[B, N]`` strength conditions every sample as its row would alone, from the transformer's per-token
    timesteps to the output blend."""
    network = _ltx2_i2v_sample_stub()
    timesteps = []
    network.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: timesteps.append(kwargs["timestep"]), with_kwargs=True
    )
    torch.manual_seed(0)
    state = torch.randn(2, 2, 3, 2, 2)
    t = torch.tensor([0.3, 0.7], dtype=network.noise_scheduler.t_precision)
    text = (torch.randn(2, 2, 3, dtype=network.connector_precision), torch.ones(2, 2, dtype=torch.long))
    frame_idx = torch.tensor([0, 2])
    clean = torch.randn(2, 2, 2, 2, 2)
    strength = torch.tensor([[0.4, 1.0], [0.0, 0.6]])
    condition = {"text_embeds": text, "frame_cond": clean, "frame_idx": frame_idx, "strength": strength}

    batched = network(state, t, condition, fwd_pred_type=fwd_pred_type)
    for b in range(2):
        alone = {
            "text_embeds": (text[0][b : b + 1], text[1][b : b + 1]),
            "frame_cond": clean[b : b + 1],
            "frame_idx": frame_idx,
            "strength": strength[b],
        }
        expected = network(state[b : b + 1], t[b : b + 1], alone, fwd_pred_type=fwd_pred_type)
        torch.testing.assert_close(batched[b : b + 1], expected)
        torch.testing.assert_close(timesteps[0][b : b + 1], timesteps[b + 1])


@pytest.mark.parametrize("time_precision", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("fwd_pred_type", ["x0", "flow"])
@pytest.mark.parametrize("modalities", [("video",), ("video", "audio")], ids=["tensor", "dict"])
def test_ltx2_i2v_forward_keeps_masked_times_in_scheduler_precision(
    time_precision, fwd_pred_type, modalities, monkeypatch
):
    network = _ltx2_i2v_sample_stub(modalities)
    network.noise_scheduler = (
        RFNoiseSchedule(t_precision=time_precision)
        if len(modalities) == 1
        else CompositeNoiseSchedule(
            schedules={m: {"name": "rf", "t_precision": time_precision} for m in modalities}, primary_key="video"
        )
    )
    ns = network.noise_scheduler
    video_ns = ns.subschedules["video"] if len(modalities) > 1 else ns
    state = torch.randn(2, 2, 3, 2, 2)
    time = torch.tensor([0.317, 0.731], dtype=ns.t_precision)
    condition = {
        "text_embeds": (torch.randn(2, 2, 3), torch.ones(2, 2, dtype=torch.long)),
        "frame_cond": torch.randn_like(state[:, :, :2]),
        "frame_idx": torch.tensor([0, 2]),
        "strength": torch.tensor([0.4, 1.0]),
    }
    converted_times = []
    convert = video_ns.convert_model_output

    def record_conversion(x_t, prediction, t, **kwargs):
        converted_times.append(t.clone())
        return convert(x_t, prediction, t, **kwargs)

    monkeypatch.setattr(video_ns, "convert_model_output", record_conversion)
    transformer_times = []
    handle = network.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: transformer_times.append(kwargs["timestep"].clone()), with_kwargs=True
    )
    prediction = network(
        state if len(modalities) == 1 else {"video": state},
        time,
        condition,
        fwd_pred_type=fwd_pred_type,
    )
    handle.remove()
    video = prediction if len(modalities) == 1 else prediction["video"]
    assert video.dtype == state.dtype and torch.isfinite(video).all()
    assert all(t.dtype == ns.t_precision for t in converted_times + transformer_times)
    masked_times = [t for t in converted_times if t.ndim == state.ndim]
    assert len(masked_times) == 1
    weight = (1 - condition["strength"]).to(time.dtype)
    expected_times = time.view(-1, 1, 1, 1, 1) * weight.view(1, 1, -1, 1, 1)
    torch.testing.assert_close(masked_times[0], expected_times, rtol=0, atol=0)
    token_times = transformer_times[0].reshape(2, 3, 4)
    expected_tokens = ns.rescale_t(time)
    if len(modalities) > 1:
        expected_tokens = expected_tokens["video"]
    expected_tokens = expected_tokens[:, None, None] * weight[None, :, None]
    torch.testing.assert_close(token_times[:, condition["frame_idx"]], expected_tokens.expand(2, 2, 4), rtol=0, atol=0)


@pytest.mark.parametrize("fwd_pred_type", ["x0", "flow"])
def test_ltx2_i2v_forward_rounds_a_bf16_transformer_output_once(fwd_pred_type, monkeypatch):
    """The conditioned frames are re-expressed, converted and blended in float64, then rounded once to the state."""
    network = _ltx2_i2v_sample_stub()
    network._record_precision(precision_model=torch.bfloat16)
    ns = network.noise_scheduler
    state = torch.randn(2, 2, 3, 2, 2)
    time = torch.tensor([0.317, 0.731], dtype=ns.t_precision)
    clean = torch.randn_like(state[:, :, :2])
    condition = {
        "text_embeds": (torch.randn(2, 2, 3, dtype=network.connector_precision), torch.ones(2, 2, dtype=torch.long)),
        "frame_cond": clean,
        "frame_idx": torch.tensor([0, 2]),
        "strength": torch.tensor([0.4, 1.0]),
    }
    captured = {}
    convert_i2v_output = network._convert_i2v_output

    def record(model_output, model_input_video, *args):
        captured.update(native=model_output, model_input=model_input_video)
        return convert_i2v_output(model_output, model_input_video, *args)

    monkeypatch.setattr(network, "_convert_i2v_output", record)
    actual = network(state, time, condition, fwd_pred_type=fwd_pred_type)
    assert captured["native"].dtype == torch.bfloat16 and actual.dtype == state.dtype

    idx = condition["frame_idx"]
    weight = (1.0 - condition["strength"]).view(1, 1, -1, 1, 1)

    def convert(x_t, prediction, t, src, target):
        return ns.convert_model_output(
            x_t, prediction, t, src_pred_type=src, target_pred_type=target, output_dtype=None
        )

    def finish(velocity):
        # transform_output in float64: the scalar-t conversion and the clean-frame blend, then one rounding.
        prediction = convert(state, velocity, time, "flow", fwd_pred_type).clone()
        target = convert(state[:, :, idx], clean, time, "x0", fwd_pred_type)
        blend = weight.double()
        prediction[:, :, idx] = prediction[:, :, idx] * blend + target * (1.0 - blend)
        return prediction.to(state.dtype)

    # REF to_denoised at the effective timesteps, then back to a velocity against the raw state and scalar t.
    native = captured["native"].double()
    effective_time = time.view(-1, 1, 1, 1, 1) * weight.to(time.dtype)
    x0_cond = convert(captured["model_input"][:, :, idx], native[:, :, idx], effective_time, "flow", "x0")
    velocity = native.clone()
    velocity[:, :, idx] = convert(state[:, :, idx], x0_cond, time, "x0", "flow")
    torch.testing.assert_close(actual, finish(velocity), rtol=0, atol=0)

    # Rounding the re-expressed frames to the transformer dtype in between would change the result.
    rounded = velocity.clone()
    rounded[:, :, idx] = velocity[:, :, idx].to(torch.bfloat16).double()
    assert not torch.equal(finish(rounded), finish(velocity))


@pytest.mark.parametrize("modalities", [("video",), ("video", "audio")], ids=["tensor", "dict"])
def test_ltx2_i2v_sample_keeps_conditioned_frame_clean_under_cfg_rescale(modalities):
    """The x0-space conditioning blend follows guidance, as in the reference, so the rescale cannot scale the frame."""
    network = _ltx2_i2v_sample_stub(modalities)
    torch.manual_seed(0)
    clean = torch.randn(1, 2, 1, 2, 2)
    mask = torch.ones(1, 2, dtype=torch.long)
    condition = {"text_embeds": (torch.randn(1, 2, 3), mask), "frame_cond": clean}
    neg_condition = {"text_embeds": (torch.randn(1, 2, 3), mask), "frame_cond": clean}
    guidance_cfg = GuidanceConfig()
    guidance_cfg.cfg.scale = 3.0
    guidance_cfg.rescale = 0.7
    noise = torch.randn(1, 2, 3, 2, 2)

    with torch.no_grad():
        sample = network.sample(
            noise if len(modalities) == 1 else {"video": noise},
            condition=condition,
            neg_condition=neg_condition,
            sample_steps=4,
            guidance_cfg=guidance_cfg,
        )
    video = sample if len(modalities) == 1 else sample["video"]

    assert torch.isfinite(video).all()
    assert torch.equal(video[:, :, :1], clean)


def test_ltx2_i2v_sample_blends_partial_conditioning_once():
    """At strength < 1 the post-guidance blend matches ``forward``'s per-pass blend under linear guidance."""
    network = _ltx2_i2v_sample_stub()
    torch.manual_seed(0)
    mask = torch.ones(1, 2, dtype=torch.long)
    condition = {
        "text_embeds": (torch.randn(1, 2, 3), mask),
        "frame_cond": torch.randn(1, 2, 1, 2, 2),
        "strength": torch.tensor([0.5]),
    }
    neg_condition = {**condition, "text_embeds": (torch.randn(1, 2, 3), mask)}
    guidance_cfg = GuidanceConfig()
    guidance_cfg.cfg.scale = 3.0
    noise = torch.randn(1, 2, 3, 2, 2)
    sample_steps = 4

    with torch.no_grad():
        sample = network.sample(
            noise,
            condition=condition,
            neg_condition=neg_condition,
            sample_steps=sample_steps,
            guidance_cfg=guidance_cfg,
        )

        # Blending in every pass commutes with linear guidance, so it reproduces the single post-guidance blend.
        ns = network.noise_scheduler
        _, grid = network.sample_scheduler_and_grid(sample_steps)
        latents = ns.latents(noise=noise, t_init=grid[0])
        frame_idx, clean, inv_strength = network._unpack_i2v_condition(condition)
        latents[:, :, frame_idx] = network._blend_with_clean(latents[:, :, frame_idx], clean, inv_strength)
        pos = {**condition, "preserve_cond_input": False}
        neg = {**neg_condition, "preserve_cond_input": False}
        for t, t_next in zip(grid[:-1], grid[1:], strict=True):
            t = t.expand(1)
            x0 = predict_with_guidance(
                network,
                latents,
                t,
                guidance_cfg=guidance_cfg,
                condition=pos,
                neg_condition=neg,
                output_dtype=torch.float32,
                fwd_pred_type="x0",
            )
            latents = ns.step(latents, ns.x0_to_flow(latents, x0, t), t, t_next, pred_type="flow")

    torch.testing.assert_close(sample, latents)


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_ltx2_sample_guides_and_steps_in_high_precision(state_dtype, monkeypatch):
    """Guidance returns x0 in high precision, which the velocity and Euler step keep until it rounds once to the
    state dtype."""
    from fastgen.networks.LTX2 import network as ltx2_network

    network = _ltx2_i2v_sample_stub()
    network.noise_scheduler = RFNoiseSchedule(max_t=1.0)
    torch.manual_seed(0)
    noise = torch.randn(1, 2, 3, 16, 16, dtype=state_dtype)
    x0 = torch.randn_like(noise)
    output_dtypes = []

    def fake_predict_with_guidance(net, latents, t, **kwargs):
        output_dtypes.append(kwargs["output_dtype"])
        return x0.to(high_precision_dtype(latents))

    monkeypatch.setattr(ltx2_network, "predict_with_guidance", fake_predict_with_guidance)
    sample_steps = 4
    sample = network.sample(noise, sample_steps=sample_steps, guidance_cfg=GuidanceConfig())

    ns = network.noise_scheduler
    _, grid = network.sample_scheduler_and_grid(sample_steps)
    expected = ns.latents(noise, t_init=grid[0])
    for t, t_next in zip(grid[:-1], grid[1:], strict=True):
        flow = ns.x0_to_flow(expected, x0.double(), t.expand(1), output_dtype=torch.float64)
        expected = (expected.double() + (t_next - t).double() * flow).to(state_dtype)

    assert sample.dtype == state_dtype
    torch.testing.assert_close(sample, expected, rtol=0, atol=0)
    assert output_dtypes == [None] * sample_steps


def _ltx2_single_block_guidance(teacher, modalities):
    """Keep the native guidance recipe valid after large-model fixtures retain one block."""
    guidance_cfg = teacher.default_guidance_cfg(modalities=modalities, version=teacher.ltx2_version)
    guidance_cfg.stg.skip_self_attn_layers = [0]
    return guidance_cfg


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_ltx2():
    """Test LTX-2 text-to-video network (transformer + connectors + scheduler)."""
    clear_gpu_memory()
    set_env_vars()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    try:
        teacher = instantiate(LTX2Config, enable_logvar_linear=True)
    except OSError as e:
        if "not a valid model identifier" in str(e) or "token" in str(e) or "does not appear" in str(e):
            pytest.skip(f"Test skipped: LTX-2 model not accessible: {e}")
        raise

    # only use a single block for testing
    teacher.transformer.transformer_blocks = teacher.transformer.transformer_blocks[:1]

    teacher.init_preprocessors()
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    F, H, W = 2, 4, 4
    C = teacher.transformer.config.in_channels

    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        condition = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        condition, teacher.preprocessors["text_encoder"].precision_model, "ltx2.text_encoder.encode"
    )
    assert isinstance(condition, tuple) and len(condition) == 2
    prompt_embeds, attention_mask = condition
    assert attention_mask.dtype == torch.long
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    attention_mask = attention_mask.to(device=device)
    condition = (prompt_embeds, attention_mask)

    # --- Input-shape cases for the multimodal-capable teacher ---
    # The teacher is configured with ``modalities=("video", "audio")``, so its
    # ``noise_scheduler`` is a ``CompositeNoiseSchedule`` and ``forward`` requires
    # dict-shaped ``x_t``.
    x_video = torch.randn(batch_size, C, F, H, W, device=device, dtype=_STATE_DTYPE)
    audio_num_frames, latent_mel_bins, num_audio_ch = teacher._compute_audio_shape((C, F, H, W), teacher.fps)
    x_audio = torch.randn(
        batch_size, num_audio_ch, audio_num_frames, latent_mel_bins, device=device, dtype=_STATE_DTYPE
    )
    x_video_dict = {"video": x_video}
    x_audio_dict = {"audio": x_audio}
    x_multi = {"video": x_video, "audio": x_audio}

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    # ---- Case 1a: full multimodal dict ----
    output = teacher(x_multi, t, condition=condition)
    assert isinstance(output, dict)
    assert output["video"].shape == x_video.shape
    assert output["audio"].shape == x_audio.shape

    # Per-block feature capture covers both modalities under 1a.
    features = teacher(x_multi, t, condition=condition, return_features_early=True, feature_indices={0})
    assert isinstance(features, list) and len(features) == 1
    assert isinstance(features[0], dict)
    assert "video" in features[0] and "audio" in features[0]

    # ---- Case 1b-V: single-key ``{"video": ...}`` dict ----
    # Includes the ``return_logvar`` and ``return_features_early`` variants;
    # only the video arm should run, and the dict output should have a single
    # ``"video"`` key.
    output = teacher(x_video_dict, t, condition=condition)
    assert isinstance(output, dict)
    assert set(output.keys()) == {"video"}
    assert output["video"].shape == x_video.shape

    output, logvar = teacher(x_video_dict, t, condition=condition, return_logvar=True)
    assert output["video"].shape == x_video.shape
    assert logvar.shape == torch.Size([batch_size, 1])

    assert teacher(x_video_dict, t, condition=condition, return_features_early=True, feature_indices=set()) == []

    output = teacher(x_video_dict, t, condition=condition, return_features_early=False, feature_indices={0})
    assert isinstance(output, list) and len(output) == 2
    assert isinstance(output[0], dict) and output[0]["video"].shape == x_video.shape

    # ---- Case 1b-A: single-key ``{"audio": ...}`` dict ----
    # Audio-only counterpart to 1b-V: ``run_video=False, run_audio=True``, so
    # no video tokens are instantiated and the dict output has a single
    # ``"audio"`` key.
    output = teacher(x_audio_dict, t, condition=condition)
    assert isinstance(output, dict)
    assert set(output.keys()) == {"audio"}
    assert output["audio"].shape == x_audio.shape

    output, logvar = teacher(x_audio_dict, t, condition=condition, return_logvar=True)
    assert output["audio"].shape == x_audio.shape
    assert logvar.shape == torch.Size([batch_size, 1])

    # ---- Two-step sample() smoke for each input-shape case ----
    # ``_ltx2_sigma_schedule`` divides by zero at ``sample_steps=1`` due to the
    # stretch step (``one_minus_z[-1] == 0``); ``2`` is the minimum safe value.
    # Decode-after-sample on the full multimodal case (1a) covers both VAE
    # branches in one pass.
    _smoke_sample(
        teacher,
        x_video_dict,
        condition=condition,
        neg_condition=condition,
        fps=teacher.fps,
        sample_steps=2,
        guidance_cfg=_ltx2_single_block_guidance(teacher, ("video",)),
    )  # 1b-V
    _smoke_sample(
        teacher,
        x_audio_dict,
        condition=condition,
        neg_condition=condition,
        fps=teacher.fps,
        sample_steps=2,
        guidance_cfg=_ltx2_single_block_guidance(teacher, ("audio",)),
    )  # 1b-A
    _smoke_sample(
        teacher,
        x_multi,
        condition=condition,
        neg_condition=condition,
        fps=teacher.fps,
        sample_steps=2,
        decode_after=True,
        guidance_cfg=_ltx2_single_block_guidance(teacher, ("video", "audio")),
    )  # 1a

    del teacher
    clear_gpu_memory()


@RunIf(min_gpus=1)
@pytest.mark.large_model
@pytest.mark.parametrize(
    "modality, shape_fn",
    [
        (
            "video",
            lambda teacher, B, F, H, W, dtype, device: (
                torch.randn(B, teacher.transformer.config.in_channels, F, H, W, device=device, dtype=_STATE_DTYPE),
            ),
        ),
        (
            "audio",
            lambda teacher, B, F, H, W, dtype, device: (
                (
                    lambda audio_num_frames, latent_mel_bins, num_audio_ch: torch.randn(
                        B,
                        num_audio_ch,
                        audio_num_frames,
                        latent_mel_bins,
                        device=device,
                        dtype=_STATE_DTYPE,
                    )
                )(*teacher._compute_audio_shape((teacher.transformer.config.in_channels, F, H, W), teacher.fps)),
            ),
        ),
    ],
)
def test_network_ltx2_unimodal(modality, shape_fn):
    """**Case 2** for LTX-2: a unimodal teacher (``modalities=("video",)`` or
    ``modalities=("audio",)``) uses ``BaseNoiseSchedule`` directly (no
    ``CompositeNoiseSchedule``), so ``forward`` accepts a bare ``Tensor`` ``x_t``
    and returns a bare ``Tensor``.  Complements ``test_network_ltx2`` which
    covers Cases 1a / 1b-V / 1b-A (dict inputs on a multimodal teacher)."""
    clear_gpu_memory()
    set_env_vars()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    try:
        teacher = instantiate(LTX2Config, modalities=(modality,))
    except OSError as e:
        if "not a valid model identifier" in str(e) or "token" in str(e) or "does not appear" in str(e):
            pytest.skip(f"Test skipped: LTX-2 model not accessible: {e}")
        raise

    teacher.transformer.transformer_blocks = teacher.transformer.transformer_blocks[:1]
    teacher.init_preprocessors()
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    # Unimodal-only network: noise_scheduler is BaseNoiseSchedule (not Composite).
    from fastgen.networks.noise_schedule import BaseNoiseSchedule

    assert isinstance(teacher.noise_scheduler, BaseNoiseSchedule)
    assert not isinstance(teacher.noise_scheduler, CompositeNoiseSchedule)

    batch_size = 1
    F, H, W = 2, 4, 4
    (x,) = shape_fn(teacher, batch_size, F, H, W, dtype, device)
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        prompt_embeds, attention_mask = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        (prompt_embeds, attention_mask),
        teacher.preprocessors["text_encoder"].precision_model,
        "ltx2.text_encoder.encode",
    )
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    attention_mask = attention_mask.to(device=device)
    condition = (prompt_embeds, attention_mask)

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    # forward() with bare Tensor → bare Tensor output.
    output = teacher(x, t, condition=condition)
    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape

    # sample() with bare Tensor → bare Tensor result.
    _smoke_sample(
        teacher,
        x,
        condition=condition,
        neg_condition=condition,
        fps=teacher.fps,
        sample_steps=2,
        decode_after=True,
        guidance_cfg=_ltx2_single_block_guidance(teacher, (modality,)),
    )

    del teacher
    clear_gpu_memory()


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_ltx2_i2v():
    """Test LTX-2 image-to-video network with first-frame conditioning."""
    clear_gpu_memory()
    set_env_vars()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    try:
        teacher = instantiate(LTX2I2VConfig, enable_logvar_linear=True)
    except OSError as e:
        if "not a valid model identifier" in str(e) or "token" in str(e) or "does not appear" in str(e):
            pytest.skip(f"Test skipped: LTX-2 model not accessible: {e}")
        raise

    # only use a single block for testing
    teacher.transformer.transformer_blocks = teacher.transformer.transformer_blocks[:1]

    teacher.init_preprocessors()
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    F, H, W = 2, 4, 4
    C = teacher.transformer.config.in_channels

    x_video = torch.randn(batch_size, C, F, H, W, device=device, dtype=_STATE_DTYPE)
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        text_embeds = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        text_embeds, teacher.preprocessors["text_encoder"].precision_model, "ltx2.text_encoder.encode"
    )
    prompt_embeds, attention_mask = text_embeds
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    attention_mask = attention_mask.to(device=device)
    text_embeds = (prompt_embeds, attention_mask)

    frame_cond = torch.randn(batch_size, C, 1, H, W, device=device, dtype=_STATE_DTYPE)
    condition = {"text_embeds": text_embeds, "frame_cond": frame_cond}

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    audio_num_frames, latent_mel_bins, num_audio_ch = teacher._compute_audio_shape((C, F, H, W), teacher.fps)
    x_audio = torch.randn(
        batch_size, num_audio_ch, audio_num_frames, latent_mel_bins, device=device, dtype=_STATE_DTYPE
    )
    x_video_dict = {"video": x_video}
    x_multi = {"video": x_video, "audio": x_audio}

    # --- Input-shape cases for the multimodal-capable I2V teacher ---
    # Same case taxonomy as ``test_network_ltx2`` (Cases 1a / 1b-V), but with
    # ``is_i2v=True`` and a dict ``condition`` carrying ``"frame_cond"``.
    # **Case 1b-A** (``{"audio": ...}``-only dict) is intentionally omitted —
    # I2V conditioning requires a video input, so the audio-only dispatch is
    # a misuse and ``forward`` would assert.

    # ---- Case 1b-V: single-key ``{"video": ...}`` dict ----
    # Includes ``return_logvar`` and ``return_features_early`` variants.
    output = teacher(x_video_dict, t, condition=condition)
    assert isinstance(output, dict) and set(output.keys()) == {"video"}
    assert output["video"].shape == x_video.shape

    output, logvar = teacher(x_video_dict, t, condition=condition, return_logvar=True)
    assert output["video"].shape == x_video.shape
    assert logvar.shape == torch.Size([batch_size, 1])

    assert teacher(x_video_dict, t, condition=condition, return_features_early=True, feature_indices=set()) == []

    output = teacher(x_video_dict, t, condition=condition, return_features_early=False, feature_indices={0})
    assert isinstance(output, list) and len(output) == 2
    assert isinstance(output[0], dict) and output[0]["video"].shape == x_video.shape

    # ---- Case 1a: full multimodal dict ----
    output = teacher(x_multi, t, condition=condition)
    assert isinstance(output, dict)
    assert output["video"].shape == x_video.shape
    assert output["audio"].shape == x_audio.shape

    # ---- Two-step sample() smoke for each input-shape case ----
    # See ``test_network_ltx2`` note: ``sample_steps>=2`` is required because
    # ``_ltx2_sigma_schedule`` divides by zero at ``sample_steps=1`` (stretch
    # step's ``one_minus_z[-1] == 0``).  Decode-after-sample on the full
    # multimodal case (1a) covers both VAE branches in one pass.
    _smoke_sample(
        teacher,
        x_video_dict,
        condition=condition,
        neg_condition=condition,
        fps=teacher.fps,
        sample_steps=2,
        guidance_cfg=_ltx2_single_block_guidance(teacher, ("video",)),
    )  # 1b-V
    _smoke_sample(
        teacher,
        x_multi,
        condition=condition,
        neg_condition=condition,
        fps=teacher.fps,
        sample_steps=2,
        decode_after=True,
        guidance_cfg=_ltx2_single_block_guidance(teacher, ("video", "audio")),
    )  # 1a

    del teacher
    clear_gpu_memory()


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_ltx2_i2v_video_only():
    """**Case 2** for I2V: a unimodal video-only I2V teacher
    (``modalities=("video",)`` + ``is_i2v=True``) uses ``BaseNoiseSchedule``
    directly, so ``forward`` accepts a bare ``Tensor`` ``x_t`` and returns a
    bare ``Tensor``.  Complements ``test_network_ltx2_i2v`` (Cases 1a / 1b-V
    on a multimodal I2V teacher); audio-only Case 1b-A and the audio
    counterpart of Case 2 are not applicable to I2V (which requires video)."""
    clear_gpu_memory()
    set_env_vars()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    try:
        teacher = instantiate(LTX2I2VConfig, modalities=("video",))
    except OSError as e:
        if "not a valid model identifier" in str(e) or "token" in str(e) or "does not appear" in str(e):
            pytest.skip(f"Test skipped: LTX-2 model not accessible: {e}")
        raise

    teacher.transformer.transformer_blocks = teacher.transformer.transformer_blocks[:1]
    teacher.init_preprocessors()
    teacher = teacher.to(device=device)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    from fastgen.networks.noise_schedule import BaseNoiseSchedule

    assert isinstance(teacher.noise_scheduler, BaseNoiseSchedule)
    assert not isinstance(teacher.noise_scheduler, CompositeNoiseSchedule)

    batch_size = 1
    F, H, W = 2, 4, 4
    C = teacher.transformer.config.in_channels
    x_video = torch.randn(batch_size, C, F, H, W, device=device, dtype=_STATE_DTYPE)
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)

    with _borrow_on_device(teacher.preprocessors["text_encoder"], device):
        prompt_embeds, attention_mask = teacher.preprocessors["text_encoder"].encode(["a caption"])
    _assert_gpu_tree_dtype_and_device(
        (prompt_embeds, attention_mask),
        teacher.preprocessors["text_encoder"].precision_model,
        "ltx2.text_encoder.encode",
    )
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    attention_mask = attention_mask.to(device=device)
    text_embeds = (prompt_embeds, attention_mask)
    frame_cond = torch.randn(batch_size, C, 1, H, W, device=device, dtype=_STATE_DTYPE)
    condition = {"text_embeds": text_embeds, "frame_cond": frame_cond}

    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)

    # forward() with bare Tensor → bare Tensor output, with first-frame preserved
    # at conditioned tokens (in x0 space).
    output = teacher(x_video, t, condition=condition)
    assert isinstance(output, torch.Tensor)
    assert output.shape == x_video.shape

    # sample() with bare Tensor → bare Tensor result.
    _smoke_sample(
        teacher,
        x_video,
        condition=condition,
        neg_condition=condition,
        fps=teacher.fps,
        sample_steps=2,
        decode_after=True,
        guidance_cfg=_ltx2_single_block_guidance(teacher, ("video",)),
    )

    del teacher
    clear_gpu_memory()


# ---------------------------------------------------------------------------
# LTX-2 VAE — input-shape dispatch coverage
# ---------------------------------------------------------------------------
#
# The LTX-2 ``forward``/``sample`` tests above cover the network's input-shape
# taxonomy (Case 1a / 1b-V / 1b-A on a multimodal teacher, Case 2 on a
# unimodal teacher).  ``LTX2VAE.encode``/``decode`` follow the same contract
# (any non-empty subset of the configured ``{"video", "audio"}`` keys;
# bare ``Tensor`` only when exactly one modality is loaded), so the cases
# below mirror the network tests one-for-one for the VAE wrapper.


@pytest.mark.parametrize("modalities", [(), ("video", "video"), ("text",)])
def test_ltx2_rejects_invalid_modality_configurations_before_loading(modalities):
    from fastgen.networks.LTX2.network import LTX2VAE

    with pytest.raises(ValueError, match="modalities must be a non-empty selection"):
        LTX2VAE(model_id_or_local_path="unused", modalities=modalities)


def test_ltx2_vae_rejects_invalid_dispatch_inputs_before_running_a_branch():
    from fastgen.networks.LTX2.network import LTX2VAE

    vae = object.__new__(LTX2VAE)
    torch.nn.Module.__init__(vae)
    vae.modalities = ("video",)
    vae._video_vae = None
    vae._audio_vae = None
    vae._vocoder = None
    vae._mel_transform = None
    data = torch.zeros(1, 3, 1, 2, 2)

    with pytest.raises(AssertionError, match="Invalid mode"):
        vae.encode(data, mode="mean")
    with pytest.raises(ValueError, match="non-empty subset"):
        vae.encode({})
    with pytest.raises(ValueError, match="non-empty subset"):
        vae.decode({"audio": data})
    with pytest.raises(TypeError, match="must be a Tensor or dict"):
        vae.decode([data])

    vae.modalities = ("audio",)
    with pytest.raises(AssertionError, match="Invalid mode"):
        vae.encode(data, mode="mean")

    vae.modalities = ("video", "audio")
    with pytest.raises(AssertionError, match="Invalid mode"):
        vae.encode_video(data, mode="mean")
    with pytest.raises(ValueError, match="bare Tensor input requires exactly one modality"):
        vae.encode(data)
    with pytest.raises(ValueError, match="bare Tensor input requires exactly one modality"):
        vae.decode(data)


def test_ltx2_vae_precision_boundary_keeps_fixed_precision_sub_parts_float32():
    """Owned VAEs follow the model precision; the mel transform and the BWE vocoder resolve fp32 themselves."""
    from fastgen.networks.LTX2.network import LTX2VAE, MelSpectrogramFP32

    class _FP32Vocoder(FastGenModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        @property
        def precision_model(self):
            return torch.float32

    vae = object.__new__(LTX2VAE)
    torch.nn.Module.__init__(vae)
    vae._video_vae = torch.nn.Linear(1, 1)
    vae._audio_vae = torch.nn.Linear(1, 1)
    vae._vocoder = _FP32Vocoder()
    vae._mel_transform = MelSpectrogramFP32(sample_rate=16000, n_fft=16, hop_length=4, n_mels=4)
    target = torch.empty((), dtype=torch.bfloat16)
    vae._set_precision(precision_model=target.dtype)
    result = vae.to(device=target.device)

    assert result is vae
    assert vae._video_vae.weight.dtype == torch.bfloat16
    assert vae._audio_vae.weight.dtype == torch.bfloat16
    assert vae._vocoder.weight.dtype == torch.float32
    assert vae._vocoder.precision_model == torch.float32
    assert vae._mel_transform.spectrogram.window.device == target.device
    assert vae._mel_transform.spectrogram.window.dtype == torch.float32
    assert vae._mel_transform.mel_scale.fb.dtype == torch.float32
    assert vae.precision_model == torch.bfloat16


def test_ltx2_vae_audio_decode_feeds_the_bwe_vocoder_in_fp32():
    """The BWE vocoder casts its own input to fp32; the wrapper casts only for the owned audio VAE."""
    from fastgen.networks.LTX2.network import LTX2VAE

    channels, frames, mel_bins = 2, 3, 4

    class _Statistic:
        def __init__(self, value):
            self.tensor = torch.full((channels * mel_bins,), value)
            self.seen_dtypes = []
            self.seen_devices = []

        def to(self, dtype):
            result = self.tensor.to(dtype=dtype)
            self.seen_dtypes.append(result.dtype)
            self.seen_devices.append(result.device)
            return result

    class _TinyAudioVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(mel_bins, mel_bins)
            self.latents_mean = _Statistic(0.0)
            self.latents_std = _Statistic(1.0)

        def decode(self, latents, return_dict=True):
            del return_dict
            return (self.proj(latents),)

    class _BWEVocoder(FastGenModule):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(channels * mel_bins, 1, kernel_size=1)
            self.seen_dtypes = []

        @property
        def precision_model(self):
            return torch.float32

        def forward(self, mel):
            mel = mel.to(dtype=self.precision_model)
            self.seen_dtypes.append(mel.dtype)
            return self.conv(mel.transpose(2, 3).flatten(1, 2))

    vae = object.__new__(LTX2VAE)
    torch.nn.Module.__init__(vae)
    vae.modalities = ("audio",)
    vae._video_vae = None
    vae._audio_vae = _TinyAudioVAE()
    vae._vocoder = _BWEVocoder()
    vae._mel_transform = None
    vae._set_precision(precision_model=torch.bfloat16)
    vae._set_device("cpu")

    latents = torch.randn(1, channels, frames, mel_bins)
    waveform = vae.decode(latents)

    assert vae._audio_vae.proj.weight.dtype == torch.bfloat16
    assert vae._audio_vae.latents_mean.seen_dtypes == [torch.bfloat16]
    assert vae._audio_vae.latents_std.seen_dtypes == [torch.bfloat16]
    assert vae._audio_vae.latents_mean.seen_devices == [latents.device]
    assert vae._audio_vae.latents_std.seen_devices == [latents.device]
    assert vae._vocoder.conv.weight.dtype == torch.float32
    assert vae._vocoder.seen_dtypes == [torch.float32]
    assert waveform.dtype == torch.float32


def test_ltx2_vae_device_only_to_does_not_redefine_boundary():
    """Storage inspection is not part of a device-only placement operation."""
    from fastgen.networks.LTX2.network import LTX2VAE

    vae = object.__new__(LTX2VAE)
    torch.nn.Module.__init__(vae)
    vae._video_vae = torch.nn.Linear(1, 1)
    vae._audio_vae = torch.nn.Linear(1, 1).to(dtype=torch.bfloat16)
    vae._vocoder = None
    vae._mel_transform = None
    vae._record_precision(precision_model=torch.float64)

    vae.to(device="cpu")

    assert vae.precision_model == torch.float64


def test_ltx2_vae_video_branch_enforces_split_precision_without_checkpoint():
    from fastgen.networks.LTX2.network import LTX2VAE

    class Statistic:
        def __init__(self, value):
            self.tensor = torch.full((3,), value)
            self.seen_dtypes = []
            self.seen_devices = []
            self.shape = None

        def view(self, *shape):
            self.shape = shape
            return self

        def to(self, dtype):
            result = self.tensor.view(*self.shape).to(dtype=dtype)
            self.seen_dtypes.append(result.dtype)
            self.seen_devices.append(result.device)
            return result

    class Distribution:
        def __init__(self, value):
            self.value = value

        def sample(self):
            return self.value

        def mode(self):
            return self.value

    class TinyVideoVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(()))
            self.latents_mean = Statistic(0.0)
            self.latents_std = Statistic(1.0)
            self.config = SimpleNamespace(scaling_factor=1.0, timestep_conditioning=False)
            self.encode_input_dtype = None
            self.decode_input_dtype = None
            self.encode_input_device = None
            self.decode_input_device = None

        @property
        def device(self):
            return self.anchor.device

        @property
        def dtype(self):
            return self.anchor.dtype

        def encode(self, value, return_dict=False):
            assert not return_dict
            self.encode_input_dtype = value.dtype
            self.encode_input_device = value.device
            return (Distribution(value * self.anchor),)

        def decode(self, value, temb, return_dict=False):
            assert temb is None and not return_dict
            self.decode_input_dtype = value.dtype
            self.decode_input_device = value.device
            return (value * self.anchor,)

    vae = LTX2VAE.__new__(LTX2VAE)
    torch.nn.Module.__init__(vae)
    vae.modalities = ("video",)
    vae._video_vae = TinyVideoVAE()
    vae._audio_vae = None
    vae._vocoder = None
    vae._mel_transform = None
    vae._record_precision(precision_model=torch.float64)
    vae._set_device("meta")

    vae.to(dtype=torch.float64)
    state = torch.randn(1, 3, 1, 2, 2, dtype=torch.float32)
    latents = vae.encode(state, mode="argmax")
    pixels = vae.decode(latents)

    assert vae.precision_model == torch.float64
    assert vae._video_vae.encode_input_dtype == torch.float64
    assert vae._video_vae.decode_input_dtype == torch.float64
    assert vae._video_vae.encode_input_device == state.device
    assert vae._video_vae.decode_input_device == state.device
    # Both encode normalization and decode denormalization stay in the VAE's model dtype.
    assert vae._video_vae.latents_mean.seen_dtypes == [torch.float64, torch.float64]
    assert vae._video_vae.latents_std.seen_dtypes == [torch.float64, torch.float64]
    assert vae._video_vae.latents_mean.seen_devices == [state.device, state.device]
    assert vae._video_vae.latents_std.seen_devices == [state.device, state.device]
    assert latents.dtype == pixels.dtype == torch.float64


@pytest.mark.parametrize(
    "pixel_frames,fps,expected",
    [(241, 24, 252), (121, 24, 127), (241, 25, 242), (121, 30, 101), (201, 25, 202)],
)
def test_ltx2_audio_latent_frames_follow_the_encoder_not_the_inference_round(pixel_frames, fps, expected):
    """``ceil`` on the mel grid like the reference trainer; ``round(duration * 25)`` gives 251/126/241/101/201."""
    from fastgen.networks.LTX2.network import _audio_latent_frames

    waveform_samples = int(pixel_frames * 16000 / fps)  # multiplied first, like the loader
    assert _audio_latent_frames(waveform_samples, hop_length=160, temporal_compression=4) == expected


@pytest.mark.parametrize("latent_f,fps,expected", [(31, 24.0, 252), (26, 25.0, 202), (16, 30.0, 101)])
def test_ltx2_compute_audio_shape_matches_the_loader_waveform_length(latent_f, fps, expected):
    """``int(frames * sr / fps)`` like the loader: 26 latents at 25 fps (8.04 s) is exactly 128640 samples."""
    from fastgen.networks.LTX2.network import LTX2

    net = LTX2.__new__(LTX2)
    torch.nn.Module.__init__(net)
    net.audio_sampling_rate, net.audio_hop_length, net.audio_vae_temporal_compression = 16000, 160, 4
    net.audio_vae_mel_bins, net.audio_vae_mel_compression_ratio, net.audio_vae_latent_channels = 64, 4, 8

    assert net._compute_audio_shape((128, latent_f, 4, 4), fps) == (expected, 16, 8)


@pytest.mark.parametrize(
    "clip_fps,accepted", [(24.0, True), (24000 / 1001, True), (25.0, False), (30.0, False)], ids=str
)
def test_ltx2_encode_real_rejects_audio_video_at_another_frame_rate(clip_fps, accepted):
    """The loader trims each waveform to the clip's frames at the clip's own rate, so the rate is recoverable; at
    another rate than the network's, audio and video RoPE positions would disagree. NTSC 23.976 passes for 24."""
    from fastgen.networks.LTX2.network import LTX2

    net = LTX2.__new__(LTX2)
    torch.nn.Module.__init__(net)
    net.fps, net.audio_sampling_rate, net.audio_hop_length, net.audio_vae_temporal_compression = 24.0, 16000, 160, 4
    net.preprocessors = {"vae": SimpleNamespace(encode=lambda data: data)}
    frames = 121
    real_raw = {"video": torch.zeros(1, 3, frames, 2, 2), "audio": torch.zeros(1, 2, int(frames * 16000 / clip_fps))}

    if accepted:
        assert net.encode_real(real_raw) is real_raw
    else:
        with pytest.raises(ValueError, match="target_fps=24.0"):
            net.encode_real(real_raw)


def test_ltx2_vae_encode_defaults_to_the_posterior_mean_like_the_reference():
    """The reference trainer caches means and its pipelines condition on means; sampling is opt-in."""
    from fastgen.networks.LTX2.network import LTX2VAE

    for fn in (LTX2VAE.forward, LTX2VAE.encode, LTX2VAE.encode_video, LTX2VAE._encode_video, LTX2VAE._encode_audio):
        assert inspect.signature(fn).parameters["mode"].default == "argmax", fn.__qualname__


def test_ltx2_audio_vae_stays_fp32_under_bf16_model_precision():
    """The reference encodes training audio in FP32; the wrapper keeps state and inputs there."""
    from fastgen.networks.LTX2.network import AutoencoderKLLTX2AudioFP32

    audio_vae = AutoencoderKLLTX2AudioFP32(base_channels=8, ch_mult=(1, 2, 4), num_res_blocks=1, mel_bins=64)
    audio_vae._set_precision(precision_model=torch.bfloat16)

    assert {p.dtype for p in audio_vae.parameters()} == {torch.float32}
    dist = audio_vae.encode(torch.zeros(1, 2, 9, 64, dtype=torch.bfloat16), return_dict=False)[0]
    assert dist.mode().dtype == torch.float32 and tuple(dist.mode().shape) == (1, 8, 3, 16)
    mel = audio_vae.decode(dist.mode().to(torch.bfloat16), return_dict=False)[0]
    assert mel.dtype == torch.float32


@pytest.mark.parametrize("encoded_frames", [252, 251])
def test_ltx2_vae_audio_encode_keeps_the_encoder_native_duration(encoded_frames):
    """A 241-frame/24-fps waveform must encode to 252 latents; any other count means the
    loader's waveform and the network's audio shape disagree."""
    from fastgen.networks.LTX2.network import LTX2VAE

    waveform_samples = int(241 * 16000 / 24)
    encoded = torch.arange(8 * encoded_frames * 16, dtype=torch.float32).reshape(1, 8, encoded_frames, 16)

    class Distribution:
        def mode(self):
            return encoded

    class TinyMel(torch.nn.Module):
        def forward(self, waveform):
            assert waveform.shape == (1, 2, waveform_samples)
            return torch.zeros(1, 2, 64, 1005)

    class TinyAudioVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(mel_hop_length=160, ch_mult=(1, 2, 4))
            self.register_buffer("latents_mean", torch.zeros(8 * 16))
            self.register_buffer("latents_std", torch.ones(8 * 16))

        def encode(self, mel, return_dict=False):
            assert not return_dict
            assert mel.shape == (1, 2, 1005, 64)
            return (Distribution(),)

    vae = LTX2VAE.__new__(LTX2VAE)
    torch.nn.Module.__init__(vae)
    vae.modalities = ("audio",)
    vae._video_vae = None
    vae._audio_vae = TinyAudioVAE()
    vae._vocoder = None
    vae._mel_transform = TinyMel()
    vae._record_precision(precision_model=torch.float32)

    waveform = torch.zeros(1, 2, waveform_samples)
    if encoded_frames != 252:
        with pytest.raises(ValueError, match="got 251 frames .* expected 252"):
            vae.encode(waveform, mode="argmax")
        return

    actual = vae.encode(waveform, mode="argmax")
    assert actual.shape == (1, 8, 252, 16)
    torch.testing.assert_close(actual, encoded)


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_ltx2_vae_audio_normalization_matches_the_packed_reference_bitwise(device_type, dtype):
    """Broadcasting the ``[C*F]`` stats over ``[B, C, T, F]`` reproduces normalizing the packed ``[B, T, C*F]``
    latents bit for bit, in encode and in decode, for contiguous and permuted decode inputs."""
    from fastgen.networks.LTX2.network import LTX2VAE

    device = torch.device(device_type)
    batch, channels, frames, mel_bins = 2, 8, 5, 16
    generator = torch.Generator().manual_seed(0)
    raw = torch.randn(batch, channels, frames, mel_bins, generator=generator).to(device=device, dtype=dtype)
    latents_mean = torch.randn(channels * mel_bins, generator=generator).to(device)
    latents_std = (torch.rand(channels * mel_bins, generator=generator) + 0.5).to(device)

    def packed(latents):
        return latents.permute(0, 2, 1, 3).reshape(batch, frames, channels * mel_bins)

    def unpacked(patched):
        return patched.view(batch, frames, channels, mel_bins).permute(0, 2, 1, 3)

    class Distribution:
        def mode(self):
            return raw

    class TinyMel(torch.nn.Module):
        def forward(self, waveform):
            return torch.ones(waveform.shape[0], 2, 1, 1, device=waveform.device)

    class TinyAudioVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # A hop of one sample without temporal compression: ``frames - 1`` samples encode to ``frames`` latents.
            self.config = SimpleNamespace(mel_hop_length=1, ch_mult=(1,))
            self.register_buffer("latents_mean", latents_mean)
            self.register_buffer("latents_std", latents_std)

        def encode(self, mel, return_dict=False):
            assert not return_dict
            return (Distribution(),)

        def decode(self, latents, return_dict=False):
            assert not return_dict
            return (latents,)

    vae = LTX2VAE.__new__(LTX2VAE)
    torch.nn.Module.__init__(vae)
    vae.modalities = ("audio",)
    vae._video_vae = None
    vae._audio_vae = TinyAudioVAE()
    vae._vocoder = torch.nn.Identity()
    vae._mel_transform = TinyMel()
    vae._record_precision(precision_model=dtype)

    mean, std = latents_mean.to(dtype=dtype), latents_std.to(dtype=dtype)
    normed = vae._encode_audio(torch.zeros(batch, 2, frames - 1, device=device))
    assert normed.dtype == dtype
    assert torch.equal(normed, unpacked((packed(raw) - mean) / std))

    restored = vae._decode_audio(normed)
    assert restored.dtype == dtype
    assert torch.equal(restored, unpacked(packed(normed) * std + mean))
    assert torch.equal(vae._decode_audio(unpacked(packed(normed))), restored)


@pytest.mark.parametrize(
    "precision_model,latent_dtype,expected_dtype",
    [
        (torch.float64, torch.float32, torch.float64),
        (None, torch.float64, torch.float64),
    ],
)
def test_ltx2_vae_video_decode_enters_model_precision_before_noise(
    monkeypatch, precision_model, latent_dtype, expected_dtype
):
    from fastgen.networks.LTX2.network import LTX2VAE

    class TinyVideoVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("latents_mean", torch.zeros(3))
            self.register_buffer("latents_std", torch.ones(3))
            self.config = SimpleNamespace(scaling_factor=1.0, timestep_conditioning=True)
            self.decode_inputs = []

        def decode(self, value, temb, return_dict=False):
            assert not return_dict
            self.decode_inputs.append((value.dtype, temb.dtype))
            return (value,)

    noise_input_dtypes = []

    def zero_noise_like(value):
        noise_input_dtypes.append(value.dtype)
        return torch.zeros_like(value)

    monkeypatch.setattr(torch, "randn_like", zero_noise_like)
    vae = LTX2VAE.__new__(LTX2VAE)
    torch.nn.Module.__init__(vae)
    vae.modalities = ("video",)
    vae._video_vae = TinyVideoVAE()
    vae._audio_vae = None
    vae._vocoder = None
    vae._mel_transform = None
    vae._record_precision(precision_model=precision_model)
    vae._set_device("cpu")

    latents = torch.randn(1, 3, 1, 2, 2, dtype=latent_dtype)
    decoded = vae.decode(latents)

    assert noise_input_dtypes == [expected_dtype]
    assert vae._video_vae.decode_inputs == [(expected_dtype, expected_dtype)]
    assert decoded.dtype == expected_dtype


def test_ltx2_fixed_precision_sub_parts_are_nested_fastgen_modules():
    from diffusers.models.autoencoders import AutoencoderKLLTX2Audio
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
    from fastgen.networks.LTX2.network import (
        AutoencoderKLLTX2AudioFP32,
        LTX2VAE,
        LTX2VocoderWithBWEFP32,
        MelSpectrogramFP32,
    )
    from fastgen.utils.audio import MelSpectrogram

    assert "_fixed_precision_overrides" not in LTX2VAE.__dict__
    assert issubclass(LTX2VocoderWithBWEFP32, LTX2VocoderWithBWE)
    assert issubclass(AutoencoderKLLTX2AudioFP32, AutoencoderKLLTX2Audio)
    assert issubclass(MelSpectrogramFP32, MelSpectrogram)
    # ``FastGenModule`` precedes the Diffusers bases so its ``device`` property is not shadowed.
    assert LTX2VocoderWithBWEFP32.device is FastGenModule.__dict__["device"]
    assert AutoencoderKLLTX2AudioFP32.device is FastGenModule.__dict__["device"]

    mel = MelSpectrogramFP32(sample_rate=16000, n_fft=16, hop_length=4, n_mels=4)
    vae = object.__new__(LTX2VAE)
    torch.nn.Module.__init__(vae)
    vae._video_vae = None
    vae._audio_vae = torch.nn.Linear(1, 1)
    vae._vocoder = torch.nn.Linear(1, 1)  # the LTX-2.0 vocoder stays an owned third-party module
    vae._mel_transform = mel

    vae._set_precision(precision_model=torch.bfloat16)

    assert vae._audio_vae.weight.dtype == torch.bfloat16 and vae._vocoder.weight.dtype == torch.bfloat16
    assert mel.precision_model == torch.float32
    assert mel.spectrogram.window.dtype == torch.float32 and mel.mel_scale.fb.dtype == torch.float32
    features = mel(torch.randn(1, 1, 64, dtype=torch.bfloat16))
    assert features.dtype == torch.float32


def test_ltx2_fp32_wrappers_leave_autocast_to_the_caller(monkeypatch):
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
    from fastgen.networks.LTX2.network import LTX2VocoderWithBWEFP32, MelSpectrogramFP32
    from fastgen.utils.audio import MelSpectrogram

    seen = {}

    def mel_forward(self, waveform):
        seen["mel"] = (waveform.dtype, waveform.device, torch.is_autocast_enabled("cpu"))
        return waveform

    def vocoder_forward(self, mel_spec):
        seen["vocoder"] = (mel_spec.dtype, mel_spec.device, torch.is_autocast_enabled("cpu"))
        return mel_spec

    monkeypatch.setattr(MelSpectrogram, "forward", mel_forward)
    monkeypatch.setattr(LTX2VocoderWithBWE, "forward", vocoder_forward)

    mel = MelSpectrogramFP32(sample_rate=16000, n_fft=16, hop_length=4, n_mels=4)
    vocoder = object.__new__(LTX2VocoderWithBWEFP32)
    torch.nn.Module.__init__(vocoder)
    mel._set_device("meta")
    vocoder._set_device("meta")

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        mel(torch.randn(1, 1, 64, dtype=torch.bfloat16))
        vocoder(torch.randn(1, 2, 4, 8, dtype=torch.bfloat16))

    assert seen == {
        "mel": (torch.float32, torch.device("cpu"), True),
        "vocoder": (torch.float32, torch.device("cpu"), True),
    }


# LTX-2.0 ships the plain ``LTX2Vocoder`` (24 kHz); LTX-2.3+ ships ``LTX2VocoderWithBWE`` (48 kHz), which
# FastGen loads as the nested fp32 ``LTX2VocoderWithBWEFP32``. The ``diffusers`` mirror of 2.3 is ungated;
# ``Lightricks/LTX-2.5-Diffusers`` (same vocoder class) is gated and needs a Hub token.
_LTX2_VAE_MODEL_IDS = ("Lightricks/LTX-2", "diffusers/LTX-2.3-Diffusers")


def _assert_ltx2_vae_storage(vae, dtype):
    """Video state stores ``dtype``; the audio VAE, mel transform, and BWE vocoder stay in fp32."""
    from fastgen.networks.LTX2.network import LTX2VocoderWithBWEFP32, MelSpectrogramFP32

    if vae._video_vae is not None:
        assert {p.dtype for p in vae._video_vae.parameters()} == {dtype}
    if vae._audio_vae is not None:
        assert vae._audio_vae.precision_model == torch.float32
        assert {p.dtype for p in vae._audio_vae.parameters()} == {torch.float32}
        assert isinstance(vae._mel_transform, MelSpectrogramFP32)
        assert vae._mel_transform.precision_model == torch.float32
        assert {b.dtype for b in vae._mel_transform.buffers()} == {torch.float32}
        vocoder_dtypes = {p.dtype for p in vae._vocoder.parameters()}
        if isinstance(vae._vocoder, LTX2VocoderWithBWEFP32):
            assert vae._vocoder.precision_model == torch.float32 and vocoder_dtypes == {torch.float32}
            assert vae._vocoder.device == vae.device
        else:
            assert vocoder_dtypes == {dtype}


def _assert_ltx2_vae_outputs(vae, outputs, label, decoded):
    """Each branch returns its own precision: video the VAE's, audio latents the FP32 audio VAE's, waveforms the
    vocoder's."""
    for key, value in outputs.items():
        if key == "video":
            expected = vae.precision_model
        elif decoded:
            expected = next(vae._vocoder.parameters()).dtype
        else:
            expected = vae._audio_vae.precision_model
        _assert_gpu_tree_dtype_and_device(value, expected, f"{label}.{key}")


@RunIf(min_gpus=1)
@pytest.mark.large_model
@pytest.mark.parametrize("model_id", _LTX2_VAE_MODEL_IDS)
def test_vae_ltx2(model_id):
    """LTX-2 VAE input-shape dispatch on a **multimodal-init** wrapper.

    Mirrors ``test_network_ltx2``'s Case 1a / 1b-V / 1b-A: a full
    ``{"video", "audio"}`` dict, a single-key ``{"video": ...}`` dict, and a
    single-key ``{"audio": ...}`` dict each dispatch only to the matching
    branch(es) and return a dict whose key set equals the input's.
    """
    from fastgen.networks.LTX2.network import LTX2VAE

    clear_gpu_memory()
    set_env_vars()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    try:
        vae = LTX2VAE(model_id_or_local_path=model_id, modalities=("video", "audio"))
    except OSError as e:
        if "not a valid model identifier" in str(e) or "token" in str(e) or "does not appear" in str(e):
            pytest.skip(f"Test skipped: LTX-2 VAE not accessible: {e}")
        raise

    vae.prepare_module(device=device, precision_model=dtype)
    assert vae.precision_model == dtype
    _assert_ltx2_vae_storage(vae, dtype)

    batch_size = 1
    fps, sampling_rate = 24.0, 16000
    # Video spatial compression = 32, temporal = 8 → 9×64×64 pixels give a
    # 2×2×2 latent (small but non-degenerate along every axis).
    pixel_F, pixel_H, pixel_W = 9, 64, 64
    audio_samples = int(round(pixel_F / fps * sampling_rate))

    pixel_video = torch.randn(batch_size, 3, pixel_F, pixel_H, pixel_W, device=device, dtype=_STATE_DTYPE)
    pixel_audio = torch.randn(batch_size, 2, audio_samples, device=device, dtype=_STATE_DTYPE)

    # ---- Case 1a: full multimodal dict ----
    enc_full = vae.encode({"video": pixel_video, "audio": pixel_audio}, mode="argmax")
    _assert_ltx2_vae_outputs(vae, enc_full, "ltx2.vae.encode", decoded=False)
    assert isinstance(enc_full, dict) and set(enc_full.keys()) == {"video", "audio"}
    assert enc_full["video"].ndim == 5 and enc_full["video"].shape[0] == batch_size
    assert enc_full["audio"].ndim == 4 and enc_full["audio"].shape[0] == batch_size

    dec_full = vae.decode(enc_full)
    _assert_ltx2_vae_outputs(vae, dec_full, "ltx2.vae.decode", decoded=True)
    assert isinstance(dec_full, dict) and set(dec_full.keys()) == {"video", "audio"}

    # ---- Case 1b-V: single-key {"video": ...} dict on a multimodal-init VAE ----
    enc_v = vae.encode({"video": pixel_video}, mode="argmax")
    _assert_ltx2_vae_outputs(vae, enc_v, "ltx2.vae.encode_video", decoded=False)
    assert isinstance(enc_v, dict) and set(enc_v.keys()) == {"video"}
    assert enc_v["video"].shape == enc_full["video"].shape

    dec_v = vae.decode({"video": enc_v["video"]})
    _assert_ltx2_vae_outputs(vae, dec_v, "ltx2.vae.decode_video", decoded=True)
    assert isinstance(dec_v, dict) and set(dec_v.keys()) == {"video"}

    # ---- Case 1b-A: single-key {"audio": ...} dict on a multimodal-init VAE ----
    enc_a = vae.encode({"audio": pixel_audio}, mode="argmax")
    _assert_ltx2_vae_outputs(vae, enc_a, "ltx2.vae.encode_audio", decoded=False)
    assert isinstance(enc_a, dict) and set(enc_a.keys()) == {"audio"}
    assert enc_a["audio"].shape == enc_full["audio"].shape

    dec_a = vae.decode({"audio": enc_a["audio"]})
    _assert_ltx2_vae_outputs(vae, dec_a, "ltx2.vae.decode_audio", decoded=True)
    assert isinstance(dec_a, dict) and set(dec_a.keys()) == {"audio"}

    del vae
    clear_gpu_memory()


@RunIf(min_gpus=1)
@pytest.mark.large_model
@pytest.mark.parametrize("model_id", _LTX2_VAE_MODEL_IDS)
@pytest.mark.parametrize("modality", ["video", "audio"])
def test_vae_ltx2_unimodal(model_id, modality):
    """**Case 2** for LTX-2 VAE: a unimodal-init wrapper
    (``modalities=("video",)`` or ``modalities=("audio",)``) accepts a bare
    ``Tensor`` for ``encode``/``decode`` and returns a bare ``Tensor``.

    Complements ``test_vae_ltx2`` (Cases 1a / 1b-V / 1b-A on a multimodal
    wrapper) and mirrors ``test_network_ltx2_unimodal``.
    """
    from fastgen.networks.LTX2.network import LTX2VAE

    clear_gpu_memory()
    set_env_vars()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    try:
        vae = LTX2VAE(model_id_or_local_path=model_id, modalities=(modality,))
    except OSError as e:
        if "not a valid model identifier" in str(e) or "token" in str(e) or "does not appear" in str(e):
            pytest.skip(f"Test skipped: LTX-2 VAE not accessible: {e}")
        raise

    vae.prepare_module(device=device, precision_model=dtype)
    assert vae.precision_model == dtype
    _assert_ltx2_vae_storage(vae, dtype)

    batch_size = 1
    fps, sampling_rate = 24.0, 16000
    pixel_F = 9
    if modality == "video":
        pixel = torch.randn(batch_size, 3, pixel_F, 64, 64, device=device, dtype=_STATE_DTYPE)
    else:
        audio_samples = int(round(pixel_F / fps * sampling_rate))
        pixel = torch.randn(batch_size, 2, audio_samples, device=device, dtype=_STATE_DTYPE)

    # encode(): bare Tensor → bare Tensor (latent space).
    enc = vae.encode(pixel, mode="argmax")
    _assert_ltx2_vae_outputs(vae, {modality: enc}, "ltx2.vae.encode", decoded=False)
    assert isinstance(enc, torch.Tensor)

    # decode(): bare Tensor → bare Tensor (pixel / waveform space).
    dec = vae.decode(enc)
    _assert_ltx2_vae_outputs(vae, {modality: dec}, "ltx2.vae.decode", decoded=True)
    assert isinstance(dec, torch.Tensor)

    del vae
    clear_gpu_memory()


# ---------------------------------------------------------------------------
# Cosmos preprocessors and networks
# ---------------------------------------------------------------------------


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_cosmos_predict2_text_encoder_precision():
    """Exercise BF16 Qwen2.5-VL compute with its native BF16 output."""
    clear_gpu_memory()
    set_env_vars()
    try:
        encoder = CosmosPredict2TextEncoder()
    except OSError as exc:
        message = str(exc).lower()
        if any(token in message for token in ("not a valid model identifier", "token", "gated", "not found")):
            pytest.skip(f"Cosmos-Reason1 text encoder is unavailable: {exc}")
        raise

    encoder.prepare_module(device="cuda", precision_model=torch.bfloat16)
    _assert_preprocessor_precision("cosmos_predict2.text_encoder", encoder, torch.bfloat16)
    embeddings = encoder.encode(["a caption"])

    assert encoder.model.dtype == torch.bfloat16
    text_config = getattr(encoder.model.config, "text_config", encoder.model.config)
    assert text_config._attn_implementation == "flash_attention_2"
    _assert_gpu_tree_dtype_and_device(embeddings, encoder.precision_model, "cosmos_predict2.text_encoder.encode")
    del encoder
    clear_gpu_memory()


def test_cosmos_predict2_text_encoder_keeps_native_dtype_and_ids_integer():
    class TinyTokenizer:
        pad_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            del args, kwargs
            return torch.tensor([[1, 2]], dtype=torch.long)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones((), dtype=torch.bfloat16))
            self.input_dtype = None

        def forward(self, input_ids, **kwargs):
            del kwargs
            self.input_dtype = input_ids.dtype
            values = torch.arange(input_ids.numel() * 3, dtype=self.anchor.dtype).reshape(*input_ids.shape, 3)
            return SimpleNamespace(hidden_states=(values, values + self.anchor, values + 2 * self.anchor))

        @property
        def device(self):
            return self.anchor.device

    encoder = CosmosPredict2TextEncoder.__new__(CosmosPredict2TextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.max_length = 4
    encoder.embedding_concat_strategy = encoder.FULL_CONCAT
    encoder.n_layers_per_group = 1
    encoder.tokenizer = TinyTokenizer()
    encoder.model = TinyModel()
    encoder._record_precision(precision_model=torch.bfloat16)

    embeddings = encoder.encode(["a caption"])

    assert encoder.model.input_dtype == torch.long
    assert embeddings.dtype == encoder.model.anchor.dtype == torch.bfloat16
    assert embeddings.shape == (1, encoder.max_length, 6)


@pytest.mark.parametrize("storage", [torch.float16, torch.bfloat16, torch.float32])
def test_cosmos_predict2_text_encoder_preserves_unrounded_rotary_frequency_buffers(storage):
    encoder = CosmosPredict2TextEncoder.__new__(CosmosPredict2TextEncoder)
    FastGenModule.__init__(encoder)
    encoder.model = torch.nn.Module()
    encoder.model.weight = torch.nn.Parameter(torch.ones(4))
    encoder.model.rotary = torch.nn.Module()
    frequencies = 1 / (10000 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
    encoder.model.rotary.register_buffer("inv_freq", frequencies.clone(), persistent=False)
    encoder.model.rotary.register_buffer("original_inv_freq", frequencies.clone(), persistent=False)
    encoder.prepare_module("cpu", precision_model=storage)
    assert encoder.model.weight.dtype == storage
    for name, value in encoder.named_buffers():
        assert value.dtype == torch.float32, name
        assert torch.equal(value, frequencies), name


def test_cosmos3_tokenizer_keeps_integer_ids_under_split_precision():
    """Tokenizer outputs stay integer when model-precision metadata is attached."""
    from fastgen.networks.cosmos3.network import Cosmos3TextTokenizer

    tokenizer = Cosmos3TextTokenizer.__new__(Cosmos3TextTokenizer)
    torch.nn.Module.__init__(tokenizer)
    tokenizer._record_precision(precision_model=torch.bfloat16)
    tokenizer.tokenize_prompt = lambda prompt, **kwargs: [len(prompt), int(kwargs["is_negative"])]

    output = tokenizer.encode(["cat", "dog"], is_negative=True)

    assert set(output) == {"input_ids"}
    assert all(ids.dtype == torch.long for ids in output["input_ids"])
    assert all(ids.device.type == "cpu" for ids in output["input_ids"])
    assert [ids.tolist() for ids in output["input_ids"]] == [[3, 1], [3, 1]]


def test_cosmos3_rejects_an_s3_model_path():
    """Pretrained loading delegates source validation to the Hugging Face resolver."""
    with pytest.raises(ValueError, match="s3://bucket/cosmos3"):
        _tiny_cosmos3(model_id_or_local_path="s3://bucket/cosmos3", load_pretrained=True)


def test_cosmos3_without_pretrained_weights_does_not_resolve_the_model_path(monkeypatch):
    import huggingface_hub

    from fastgen.networks.cosmos3 import network as cosmos3_network

    def unexpected_load(*args, **kwargs):
        pytest.fail("load_pretrained=False must not resolve or load checkpoint assets")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected_load)
    monkeypatch.setattr(cosmos3_network, "load_checkpoint_state_dict", unexpected_load)

    network = _tiny_cosmos3(model_id_or_local_path="s3://bucket/cosmos3", load_pretrained=False)

    assert network.model_id_or_local_path == "s3://bucket/cosmos3"


def _tiny_cosmos3(**overrides):
    kwargs = dict(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        latent_channel=8,
        latent_patch_size=2,
        patch_latent_dim=32,
        vocab_size=100,
        rope_axes_dim=(4, 2, 2),
        rope_theta=1e4,
        temporal_modality_margin=8,
        enable_logvar_linear=True,
        load_pretrained=False,
    )
    kwargs.update(overrides)
    return Cosmos3(**kwargs).eval()


def test_cosmos3_set_precision_keeps_time_embedder_fp32_for_ddp_precast(monkeypatch):
    """The DDP pre-cast retains FP32 time-embedding compute."""
    network = _tiny_cosmos3()
    time_weight = network.transformer.time_embedder.linear_1.weight
    with torch.no_grad():
        time_weight.flatten()[0] = 1.0001  # Not exactly representable in bfloat16.
    expected_time_weight = time_weight.detach().clone()

    network._set_precision(precision_model=torch.bfloat16)

    assert network.transformer.proj_in.weight.dtype == torch.bfloat16
    assert network.transformer.time_embedder.linear_1.weight.dtype == torch.float32
    assert torch.equal(network.transformer.time_embedder.linear_1.weight, expected_time_weight)
    assert network.transformer.compute_t_embed(torch.tensor([0.5])).dtype == torch.float32

    observed = {}

    def capture_time_embedder(module, inputs, output):
        observed["time_input"] = inputs[0].dtype
        observed["time_output"] = output.dtype

    network.transformer.time_embedder.register_forward_hook(capture_time_embedder)

    def capture_logvar(module, inputs, output):
        observed["logvar_input"] = inputs[0].dtype
        observed["logvar_output"] = output.dtype

    network.transformer.logvar_linear.register_forward_hook(capture_logvar)
    apply_timestep_embeds = network.transformer._apply_timestep_embeds_to_noisy_tokens

    def capture_token_boundary(packed_tokens, packed_timestep_embeds, *args, **kwargs):
        observed["packed_tokens"] = packed_tokens.dtype
        observed["packed_timestep_embeds"] = packed_timestep_embeds.dtype
        return apply_timestep_embeds(packed_tokens, packed_timestep_embeds, *args, **kwargs)

    monkeypatch.setattr(network.transformer, "_apply_timestep_embeds_to_noisy_tokens", capture_token_boundary)
    x_t = torch.randn(1, 8, 2, 4, 4, dtype=torch.bfloat16)
    condition = {"input_ids": [torch.randint(0, 100, (5,))]}
    with torch.no_grad():
        timestep = torch.tensor([0.5], dtype=network.noise_scheduler.t_precision)
        output, logvar = network(x_t, timestep, condition=condition, return_logvar=True)

    assert observed == {
        "time_input": torch.float32,
        "time_output": torch.float32,
        "logvar_input": torch.bfloat16,
        "logvar_output": torch.bfloat16,
        "packed_tokens": torch.bfloat16,
        "packed_timestep_embeds": torch.bfloat16,
    }
    assert output.dtype == x_t.dtype
    assert logvar.dtype == x_t.dtype


def test_cosmos3_timestep_scatter_handles_split_amp_dtype():
    """Vision AMP and model embeddings may use different low-precision dtypes."""
    transformer = _tiny_cosmos3().transformer
    packed_tokens = torch.zeros((2, transformer.hidden_size), dtype=torch.float16)
    timestep_embeds = torch.ones((1, transformer.hidden_size), dtype=torch.bfloat16)

    output = transformer._apply_timestep_embeds_to_noisy_tokens(
        packed_tokens=packed_tokens,
        packed_timestep_embeds=timestep_embeds,
        noisy_frame_indexes=[torch.tensor([1])],
        token_shapes=[(2, 1, 1)],
    )

    assert output.dtype == packed_tokens.dtype
    assert torch.equal(output[0], torch.zeros_like(output[0]))
    assert torch.equal(output[1], torch.ones_like(output[1]))


@RunIf(min_gpus=1)
def test_cosmos3_forward_handles_split_model_and_inference_amp():
    """A BF16 model can run end to end under FP16 inference autocast."""
    network = _tiny_cosmos3()
    network._set_precision(precision_model=torch.bfloat16)
    network = network.to(device="cuda")
    x_t = torch.randn((1, 8, 2, 4, 4), device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], device="cuda", dtype=network.noise_scheduler.t_precision)
    condition = {"input_ids": [torch.randint(0, 100, (5,))]}

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = network(x_t, timestep, condition=condition)

    assert output.shape == x_t.shape
    assert torch.isfinite(output).all()


def test_cosmos3_fully_shard_keeps_time_embedder_fp32(monkeypatch):
    """The reference computes timestep embeddings in FP32 under BF16 training."""
    from fastgen.networks.cosmos3 import network as cosmos3_network
    from torch.distributed.fsdp import MixedPrecisionPolicy

    network = _tiny_cosmos3()
    # Exercise the pre-FSDP master-parameter cast as well as the child group's compute policy.
    network._set_precision(precision_model=torch.bfloat16)
    assert network.transformer.proj_in.weight.dtype == torch.bfloat16
    assert network.transformer.time_embedder.linear_1.weight.dtype == torch.float32
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    wrapped = []
    monkeypatch.setattr(cosmos3_network, "fully_shard", lambda module, **kwargs: wrapped.append((module, kwargs)))

    network.fully_shard(mesh="mesh", mp_policy=policy)

    assert [module for module, _ in wrapped] == [
        *network.transformer.layers,
        network.transformer.time_embedder,
        network.transformer,
    ]
    for _, layer_kwargs in wrapped[: len(network.transformer.layers)]:
        assert layer_kwargs["mp_policy"] is policy
    time_embedder_policy = wrapped[-2][1]["mp_policy"]
    assert time_embedder_policy.param_dtype == torch.float32
    assert time_embedder_policy.reduce_dtype == torch.float32
    assert wrapped[-1][1]["mp_policy"] is policy


def test_cosmos3_time_embedder_fp32_is_configurable(monkeypatch):
    from fastgen.networks.cosmos3 import network as cosmos3_network
    from torch.distributed.fsdp import MixedPrecisionPolicy

    assert Cosmos3_Nano_Config.keep_time_embedder_fp32 is True
    network = _tiny_cosmos3(keep_time_embedder_fp32=False)
    network._set_precision(precision_model=torch.bfloat16)
    assert network.transformer.time_embedder.linear_1.weight.dtype == torch.bfloat16
    # The rotary frequencies stay FP32 regardless, as the reference asserts.
    assert network.transformer.rotary_emb.inv_freq.dtype == torch.float32
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    wrapped = []
    monkeypatch.setattr(cosmos3_network, "fully_shard", lambda module, **kwargs: wrapped.append((module, kwargs)))

    network.fully_shard(mesh="mesh", mp_policy=policy)

    assert [module for module, _ in wrapped] == [*network.transformer.layers, network.transformer]
    assert all(wrap_kwargs["mp_policy"] is policy for _, wrap_kwargs in wrapped)


def test_cosmos3_set_precision_keeps_only_its_rotary_and_time_embedder_fp32():
    """Cosmos3 pins its rotary frequencies and time embedder by name; declared tensors follow the storage precision."""

    class TinyTransformer(torch.nn.Module):
        _keep_in_fp32_modules = ["time_embedder", "scale_shift_table"]

        def __init__(self):
            super().__init__()
            self.time_embedder = torch.nn.Linear(2, 2)
            self.proj = torch.nn.Linear(2, 2)
            self.scale_shift_table = torch.nn.Parameter(torch.ones(2))
            self.rotary_emb = torch.nn.Module()
            self.rotary_emb.register_buffer("inv_freq", torch.ones(2), persistent=False)
            self.rope = torch.nn.Module()
            self.rope.register_buffer("freqs_cos", torch.ones(2))

    network = object.__new__(Cosmos3)
    torch.nn.Module.__init__(network)
    network.transformer = TinyTransformer()
    network.keep_time_embedder_fp32 = True

    network._set_precision(precision_model=torch.bfloat16, precision_opt=torch.float64)

    assert network.transformer.proj.weight.dtype == torch.float64
    assert network.transformer.time_embedder.weight.dtype == torch.float32
    assert network.transformer.rotary_emb.inv_freq.dtype == torch.float32
    # Cosmos does not read the Diffusers-style declaration; other rotary-like tables are ordinary state.
    assert network.transformer.scale_shift_table.dtype == torch.float64
    assert network.transformer.rope.freqs_cos.dtype == torch.float64


def test_cosmos3_forward_shapes_logvar_features(monkeypatch):
    net = _tiny_cosmos3().double()
    net._record_precision(precision_model=torch.float64)
    assert net.net_pred_type == "flow" and net.schedule_type == "rf"
    batch_size, channels, frames, height, width = 2, 8, 2, 4, 4
    x_t = torch.randn(batch_size, channels, frames, height, width)
    timestep = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    condition = {"input_ids": [torch.randint(0, 100, (5,)), torch.randint(0, 100, (7,))]}
    packed_fps = []
    build_sample_inputs = net._build_sample_inputs

    def capture_fps(*args, **kwargs):
        packed_fps.append(kwargs["fps"])
        return build_sample_inputs(*args, **kwargs)

    monkeypatch.setattr(net, "_build_sample_inputs", capture_fps)

    with torch.no_grad():
        output = net(x_t, timestep, condition=condition)
    assert output.shape == x_t.shape and output.dtype == x_t.dtype and torch.isfinite(output).all()

    with torch.no_grad():
        output_x0 = net(x_t, timestep, condition=condition, fwd_pred_type="x0")
    assert output_x0.shape == x_t.shape and torch.isfinite(output_x0).all()

    with torch.no_grad():
        output_with_logvar, logvar = net(x_t, timestep, condition=condition, return_logvar=True)
    assert output_with_logvar.shape == x_t.shape and logvar.shape == (batch_size, 1)

    with torch.no_grad():
        result = net(x_t, timestep, condition=condition, feature_indices={0, 1})
    assert isinstance(result, list) and len(result) == 2
    output_with_features, features = result
    assert output_with_features.shape == x_t.shape and len(features) == 2
    # Feature maps: [B, hidden, T, ph, pw].
    assert features[0].shape == (batch_size, 64, frames, height // 2, width // 2), features[0].shape
    # Rotary positions use the network's configured clock; conditions carry no FPS of their own.
    assert packed_fps and set(packed_fps) == {net.fps}

    # Each packed sample runs on its own clock; a single value covers the whole batch.
    packed_fps.clear()
    with torch.no_grad():
        net(x_t, timestep, condition=condition, fps=torch.tensor([24.0, 30.0]))
        net(x_t, timestep, condition=condition, fps=torch.tensor([30.0]))
    assert packed_fps == [24.0, 30.0, 30.0, 30.0]
    with pytest.raises(AssertionError, match="one per sample"):
        net(x_t, timestep, condition=condition, fps=torch.tensor([24.0, 30.0, 12.0]))


def _cosmos3_i2v_condition(batch_size, channels, frames, height, width, dtype):
    mask = torch.zeros(batch_size, 1, frames, 1, 1, dtype=torch.bool)
    mask[:, :, 0] = True
    return {
        "input_ids": [torch.randint(0, 100, (5,)) for _ in range(batch_size)],
        "condition_mask": mask,
        "conditioning_latents": torch.randn(batch_size, channels, 1, height, width, dtype=dtype),
    }


@pytest.mark.parametrize("fwd_pred_type", ["x0", "eps", "flow"])
def test_cosmos3_forward_feeds_clean_conditioning_frames_and_pins_them(fwd_pred_type):
    """I2V: the backbone sees the clean anchor (reference: sigma 0, no noise) and the output round-trips to it."""
    net = _tiny_cosmos3(is_i2v=True).double()
    net._record_precision(precision_model=torch.float64)
    batch_size, channels, frames, height, width = 2, 8, 2, 4, 4
    x_t = torch.randn(batch_size, channels, frames, height, width, dtype=torch.float64)
    timestep = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    condition = _cosmos3_i2v_condition(batch_size, channels, frames, height, width, torch.float64)
    clean = condition["conditioning_latents"]

    seen_inputs, raw_velocities = [], []
    transformer_forward = net.transformer.forward

    def capture(*args, **kwargs):
        seen_inputs.append(kwargs["vision_tokens"][0].clone())
        result = transformer_forward(*args, **kwargs)
        raw_velocities.append(result[0][0].clone())
        return result

    net.transformer.forward = capture
    with torch.no_grad():
        output = net(x_t, timestep, condition=condition, fwd_pred_type=fwd_pred_type)

    model_input = torch.cat(seen_inputs, dim=0)
    assert torch.equal(model_input[:, :, :1], clean)
    assert torch.equal(model_input[:, :, 1:], x_t[:, :, 1:])
    ns = net.noise_scheduler
    x0_pred = ns.convert_model_output(x_t, output, timestep, src_pred_type=fwd_pred_type, target_pred_type="x0")
    torch.testing.assert_close(x0_pred[:, :, :1], clean, rtol=1e-9, atol=1e-9)
    if fwd_pred_type == "x0":
        assert torch.equal(output[:, :, :1], clean)
    raw = torch.cat(raw_velocities, dim=0)
    expected = ns.convert_model_output(x_t, raw, timestep, src_pred_type="flow", target_pred_type=fwd_pred_type)
    torch.testing.assert_close(output[:, :, 1:], expected[:, :, 1:])


def test_cosmos3_sample_keeps_the_conditioning_frames_clean():
    """``forward`` feeds the clean anchor; the sampler re-injects it after each ``step`` like the reference's blend,
    which keeps the anchor exact under bf16 rounding."""

    net = _tiny_cosmos3(is_i2v=True).double()
    net._record_precision(precision_model=torch.float64)
    batch_size, channels, frames, height, width = 1, 8, 2, 4, 4
    noise = torch.randn(batch_size, channels, frames, height, width, dtype=torch.float64)
    condition = _cosmos3_i2v_condition(batch_size, channels, frames, height, width, torch.float64)
    clean = condition["conditioning_latents"]
    injections = []
    preserve = net.preserve_conditioning

    def spy(x, cond):
        injections.append(tuple(x.shape))
        return preserve(x, cond)

    net.preserve_conditioning = spy
    with torch.no_grad():
        sample = net.sample(noise, condition=condition, sample_steps=4, shift=10.0, guidance_cfg=GuidanceConfig())

    assert torch.equal(sample[:, :, :1], clean)
    assert torch.isfinite(sample).all()
    # One injection per network evaluation (``forward``) plus one per scheduler step.
    assert len(injections) == 8


def test_cosmos3_sample_uses_full_precision_timestep_vector():
    from diffusers import UniPCMultistepScheduler

    net = _tiny_cosmos3()
    net.sample_scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
    )
    timestep_dtypes, timestep_values, fps_values = [], [], []

    def capture_timestep(module, args, kwargs):
        del module
        timestep_dtypes.append(args[1].dtype)
        timestep_values.append(args[1][0].item())
        fps_values.append(kwargs["fps"])

    handle = net.register_forward_pre_hook(capture_timestep, with_kwargs=True)
    noise = torch.randn(1, 8, 2, 4, 4)
    condition = {"input_ids": [torch.randint(0, 100, (5,))]}
    try:
        with torch.no_grad():
            output = net.sample(
                noise,
                condition=condition,
                sample_steps=2,
                shift=float(net.sample_scheduler.config.flow_shift),
                guidance_cfg=GuidanceConfig(),
            )
    finally:
        handle.remove()

    assert output.shape == noise.shape and output.dtype == noise.dtype
    assert timestep_dtypes == [net.noise_scheduler.t_precision] * 2
    assert fps_values == [None, None]  # the sampler leaves the clock to the network's configured fps
    # The model is queried at the sigmas the solver integrates, not at their truncated integer timesteps.
    ns, sched = net.noise_scheduler, net.sample_scheduler
    expected_t = ns.safe_clamp(sched.sigmas[:-1].to(ns.t_precision), min=ns.min_t, max=ns.max_t)
    assert timestep_values == expected_t.tolist()


def _native_cosmos3_unipc_grid(shift: float, steps: int):
    """``cosmos_framework`` ``FlowUniPCMultistepScheduler.set_timesteps``, replicated independently.

    The native scheduler derives ``sigma_max``/``sigma_min`` from its float32 training sigmas, spaces the
    inference sigmas linearly, applies the shift in float64, truncates ``sigma * 1000`` to int64 timesteps,
    and appends the final sigma 0 as float32.
    """
    train_sigmas = torch.from_numpy((1.0 - np.linspace(1, 1 / 1000, 1000)[::-1]).copy()).to(torch.float32)
    sigma_max, sigma_min = train_sigmas[0].item(), train_sigmas[-1].item()
    sigmas = np.linspace(sigma_max, sigma_min, steps + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = torch.from_numpy(sigmas * 1000).to(torch.int64)
    return timesteps, torch.from_numpy(np.concatenate([sigmas, [0.0]]).astype(np.float32))


@pytest.mark.parametrize(
    ("shift", "steps", "first", "last"),
    [(10.0, 35, 999, 227), (5.0, 35, 999, 128), (3.0, 20, 999, 136), (10.0, 4, 999, 768)],
)
def test_cosmos3_sample_scheduler_matches_native_flow_unipc_grid(shift, steps, first, last):
    """The in-code scheduler reproduces the native shifted-linear grid bit for bit (timesteps and sigmas), and the
    model is queried at its sigmas."""
    net = object.__new__(Cosmos3)
    torch.nn.Module.__init__(net)
    net.sample_scheduler = None
    net.noise_scheduler = RFNoiseSchedule(max_t=1.0)  # as Cosmos3_Nano_Config
    scheduler, t_grid = net.sample_scheduler_and_grid(steps, height=256, width=256, flow_shift=shift)

    ref_timesteps, ref_sigmas = _native_cosmos3_unipc_grid(shift, steps)
    assert torch.equal(scheduler.timesteps.cpu(), ref_timesteps)
    assert scheduler.sigmas.dtype == torch.float32 and torch.equal(scheduler.sigmas, ref_sigmas)
    assert (scheduler.timesteps[0].item(), scheduler.timesteps[-1].item()) == (first, last)
    assert torch.equal(t_grid, ref_sigmas[:-1].to(net.noise_scheduler.t_precision))
    # The released scheduler_config.json selects a Karras grid whose diffusers branch ignores the shift.
    config = scheduler.config
    assert not config.use_karras_sigmas and config.use_flow_sigmas and config.flow_shift == shift
    assert config.prediction_type == "flow_prediction" and config.predict_x0
    assert (config.solver_order, config.solver_type, config.final_sigmas_type) == (2, "bh2", "zero")
    assert config.lower_order_final and not config.thresholding


@pytest.mark.parametrize(
    ("height", "width", "shift"),
    [(480, 832, 5.0), (832, 480, 5.0), (704, 1280, 10.0), (720, 1280, 10.0), (256, 256, 3.0), (320, 576, 3.0)],
)
def test_cosmos3_sample_scheduler_uses_native_resolution_shift_tiers(height, width, shift):
    net = object.__new__(Cosmos3)
    torch.nn.Module.__init__(net)
    net.sample_scheduler = None
    net.noise_scheduler = RFNoiseSchedule(max_t=1.0)  # as Cosmos3_Nano_Config
    assert net.sample_scheduler_and_grid(2, height, width)[0].config.flow_shift == shift


def test_cosmos3_sample_scheduler_starts_at_step_zero_on_repeated_leading_timesteps():
    """From 113 steps at 720p (shift 10) the leading int64 timestep repeats; UniPC still starts at the first sigma."""
    net = object.__new__(Cosmos3)
    torch.nn.Module.__init__(net)
    net.sample_scheduler = None
    net.noise_scheduler = RFNoiseSchedule(max_t=1.0)  # as Cosmos3_Nano_Config
    scheduler, _ = net.sample_scheduler_and_grid(113, height=720, width=1280)
    assert scheduler.config.flow_shift == 10.0 and scheduler.timesteps[0] == scheduler.timesteps[1]

    sample = torch.randn(1, 4, 2, 2)
    for index, timestep in enumerate(scheduler.timesteps):
        sample = scheduler.step(torch.zeros_like(sample), timestep, sample, return_dict=False)[0]
        assert scheduler.step_index == index + 1
    assert torch.isfinite(sample).all()


def test_cosmos3_sample_starts_at_the_first_sigma_of_the_native_grid():
    """The first model input is noise at the native grid's first sigma, and the model is queried at its sigmas."""

    net = _tiny_cosmos3(max_t=1.0)
    seen_inputs, seen_times = [], []

    def capture(module, args, kwargs):
        del module, kwargs
        seen_inputs.append(args[0].clone())
        seen_times.append(args[1][0].item())

    handle = net.register_forward_pre_hook(capture, with_kwargs=True)
    noise = torch.randn(1, 8, 2, 4, 4)
    condition = {"input_ids": [torch.randint(0, 100, (5,))]}
    try:
        with torch.no_grad():
            net.sample(noise, condition=condition, sample_steps=4, shift=10.0, guidance_cfg=GuidanceConfig())
    finally:
        handle.remove()

    _, ref_sigmas = _native_cosmos3_unipc_grid(10.0, 4)
    ns = net.noise_scheduler
    ref_times = ns.safe_clamp(ref_sigmas[:-1].to(ns.t_precision), min=ns.min_t, max=ns.max_t)
    expected_initial = ns.latents(noise, t_init=ref_times[0])
    assert torch.equal(seen_inputs[0], expected_initial)
    assert seen_times == ref_times.tolist()
    # Without an explicit shift the tier of the 64x64 pixel clip (nearest: 256) selects the native default.
    net.sample_scheduler = None
    with torch.no_grad():
        net.sample(noise, condition=condition, sample_steps=2, guidance_cfg=GuidanceConfig())
    assert net.sample_scheduler.config.flow_shift == 3.0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("t_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("max_t", [0.5, 1.0])
def test_cosmos3_sample_keeps_selected_state_and_time_dtypes(dtype, t_dtype, max_t, monkeypatch):
    net = _tiny_cosmos3(max_t=max_t)
    net.noise_scheduler.t_precision = t_dtype
    states, times = [], []

    def zero_velocity(x, t, output_dtype="same", **kwargs):
        states.append(x.clone())
        times.append(t.clone())
        return cast_like(torch.zeros_like(x), x, output_dtype)

    monkeypatch.setattr(net, "forward", zero_velocity)
    noise = torch.randn(2, 8, 2, 4, 4).to(dtype)
    initial = noise.clone()
    result = net.sample(noise, sample_steps=3, guidance_cfg=GuidanceConfig())
    assert torch.equal(states[0], net.noise_scheduler.latents(initial, t_init=times[0][0]))
    assert torch.equal(noise, initial)
    assert result.dtype == dtype and all(x.dtype == dtype for x in states)
    assert all(t.dtype == t_dtype for t in times)
    expected = net.sample_scheduler.sigmas[:-1].to(t_dtype)
    expected = net.noise_scheduler.safe_clamp(expected, min=net.noise_scheduler.min_t, max=net.noise_scheduler.max_t)
    assert torch.equal(torch.stack(times)[:, 0], expected)
    assert torch.isfinite(result).all()


def _stub_cosmos3_tokenizer():
    """``Cosmos3TextTokenizer`` with a recording chat-template stub instead of the Qwen tokenizer files."""
    from fastgen.networks.cosmos3.network import Cosmos3TextTokenizer

    tokenizer = Cosmos3TextTokenizer.__new__(Cosmos3TextTokenizer)
    torch.nn.Module.__init__(tokenizer)
    conversations = []

    class _ChatTemplate:
        def apply_chat_template(self, conversation, **kwargs):
            conversations.append(conversation)
            return {"input_ids": [1, 2]}

    tokenizer.tokenizer = _ChatTemplate()
    tokenizer.eos_token_id = 7
    tokenizer.start_of_generation_id = 9
    tokenizer.duration_template = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
    tokenizer.image_resolution_template = "This image is of {height}x{width} resolution."
    tokenizer.video_resolution_template = "This video is of {height}x{width} resolution."
    tokenizer.inverse_duration_template = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
    tokenizer.inverse_image_resolution_template = "This image is not of {height}x{width} resolution."
    tokenizer.inverse_video_resolution_template = "This video is not of {height}x{width} resolution."
    return tokenizer, conversations


def test_cosmos3_tokenizer_prompt_defaults_match_the_native_framework():
    """No system prompt by default, and the negative prompt carries the positive metadata templates."""
    tokenizer, conversations = _stub_cosmos3_tokenizer()
    geometry = dict(num_frames=13, height=128, width=128, fps=24.0)
    metadata = "The video is 0.5 seconds long and is of 24 FPS. This video is of 128x128 resolution."

    ids = tokenizer.tokenize_prompt("a cat", **geometry)
    assert ids == [1, 2, 7, 9]
    assert conversations[-1] == [{"role": "user", "content": f"a cat. {metadata}"}]

    tokenizer.tokenize_prompt("not a cat", is_negative=True, **geometry)
    assert conversations[-1] == [{"role": "user", "content": f"not a cat. {metadata}"}]

    tokenizer.tokenize_prompt("", is_negative=True, **geometry)
    assert conversations[-1] == [{"role": "user", "content": metadata}]

    tokenizer.tokenize_prompt("a cat", use_system_prompt=True, **geometry)
    assert conversations[-1][0] == {
        "role": "system",
        "content": "You are a helpful assistant who will generate videos from a give prompt.",
    }


def test_cosmos3_tokenizer_negative_metadata_modes():
    tokenizer, conversations = _stub_cosmos3_tokenizer()
    geometry = dict(num_frames=13, height=128, width=128, fps=24.0)

    tokenizer.tokenize_prompt("not a cat", is_negative=True, negative_metadata_mode="inverse", **geometry)
    assert conversations[-1][-1]["content"] == (
        "not a cat. The video is not 0.5 seconds long and is not of 24 FPS. This video is not of 128x128 resolution."
    )
    tokenizer.tokenize_prompt("not a cat", is_negative=True, negative_metadata_mode="none", **geometry)
    assert conversations[-1][-1]["content"] == "not a cat"
    # The mode only concerns the negative prompt.
    tokenizer.tokenize_prompt("a cat", negative_metadata_mode="none", **geometry)
    assert conversations[-1][-1]["content"].startswith("a cat. The video is 0.5 seconds long")
    with pytest.raises(ValueError, match="negative_metadata_mode"):
        tokenizer.tokenize_prompt("a cat", negative_metadata_mode="reverse", **geometry)


def test_cosmos_predict2_sample_uses_full_precision_timestep_vector():
    net = CosmosPredict2(
        model_channels=48,
        num_blocks=1,
        num_heads=4,
        max_img_h=4,
        max_img_w=4,
        max_frames=4,
        in_channels=2,
        out_channels=2,
        patch_spatial=1,
        patch_temporal=1,
        concat_padding_mask=False,
        add_video_condition_mask=False,
        mlp_ratio=2.0,
        crossattn_emb_channels=8,
        use_adaln_lora=False,
        enable_logvar_linear=False,
        use_crossattn_projection=False,
        use_wan_fp32_strategy=False,
        sample_sigma_schedule="karras",
    ).eval()
    timestep_dtypes, timestep_values = [], []

    def capture_timestep(module, args, kwargs):
        del module, kwargs
        timestep_dtypes.append(args[1].dtype)
        timestep_values.append(args[1][0].item())

    handle = net.register_forward_pre_hook(capture_timestep, with_kwargs=True)
    noise = torch.randn(1, 2, 2, 2, 2)
    condition = torch.randn(1, 3, 8)
    try:
        with torch.no_grad():
            output = net.sample(
                noise,
                condition=condition,
                sample_steps=1,
                guidance_cfg=GuidanceConfig(),
            )
    finally:
        handle.remove()

    assert output.shape == noise.shape and output.dtype == noise.dtype
    assert timestep_dtypes == [net.noise_scheduler.t_precision] * 2
    # The model is queried at the sigmas the solver integrates, not at their truncated integer timesteps.
    ns, sched = net.noise_scheduler, net.sample_scheduler
    expected_t = ns.safe_clamp(sched.sigmas[:-1].to(ns.t_precision), min=ns.min_t, max=ns.max_t)
    assert timestep_values == expected_t.tolist()
    # ``sigma_min`` bounds the grid's noise levels; the last step still goes to sigma 0, on clean data.
    assert sched.sigmas[-1].item() == 0.0


def test_cosmos_predict2_sample_scheduler_starts_at_step_zero_on_repeated_leading_timesteps():
    """From 1058 steps the Karras grid's leading int64 timestep repeats; UniPC still starts at the first sigma."""
    net = object.__new__(CosmosPredict2)
    torch.nn.Module.__init__(net)
    net.sample_scheduler = None
    net.sample_sigma_schedule = "karras"
    net.noise_scheduler = RFNoiseSchedule()
    scheduler, _ = net.sample_scheduler_and_grid(1058)
    assert scheduler.timesteps[0] == scheduler.timesteps[1]

    sample = torch.randn(1, 4, 2, 2)
    for index, timestep in enumerate(scheduler.timesteps):
        sample = scheduler.step(torch.zeros_like(sample), timestep, sample, return_dict=False)[0]
        assert scheduler.step_index == index + 1
    assert torch.isfinite(sample).all()


def _tiny_cosmos_predict2(**overrides):
    kwargs = dict(
        model_channels=48,
        num_blocks=1,
        num_heads=4,
        max_img_h=4,
        max_img_w=4,
        max_frames=4,
        in_channels=2,
        out_channels=2,
        patch_spatial=1,
        patch_temporal=1,
        concat_padding_mask=False,
        add_video_condition_mask=False,
        mlp_ratio=2.0,
        crossattn_emb_channels=8,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        enable_logvar_linear=False,
        use_crossattn_projection=False,
    )
    kwargs.update(overrides)
    return CosmosPredict2(**kwargs)


@pytest.mark.parametrize(
    ("config_name", "count", "first", "second", "last", "first_time", "last_time"),
    [
        ("CosmosPredict2_2B_Config", 36, 0.9950248599052429, 0.9942067265510559, 0.009900989942252636, 995, 9),
        ("CosmosPredict2_14B_Config", 35, 0.9997998476028442, 0.9939484000205994, 0.12809006869792938, 999, 128),
    ],
)
def test_cosmos_predict2_released_configs_reproduce_native_sigma_grids(
    config_name, count, first, second, last, first_time, last_time
):
    """35-step grids captured from native FlowUniPC: 2B uses the Karras preset and 14B shifted linear sigmas.

    Both released configs query the model at the solver's own sigmas, which their full [0, 1] RF clock leaves
    unclamped, rather than at the truncated integer timesteps.
    """
    from fastgen.configs import net as net_configs

    config = getattr(net_configs, config_name)
    default_schedule = inspect.signature(CosmosPredict2.__init__).parameters["sample_sigma_schedule"].default
    net = _tiny_cosmos_predict2(
        sample_sigma_schedule=config.get("sample_sigma_schedule", default_schedule), **network_schedule_kwargs(config)
    )
    scheduler, times = net.sample_scheduler_and_grid(35)
    assert len(scheduler.timesteps) == count
    assert scheduler.sigmas[[0, 1, -2]].tolist() == [first, second, last]
    assert scheduler.sigmas[-1] == 0
    assert scheduler.timesteps[[0, -1]].tolist() == [first_time, last_time]
    assert times.dtype == net.noise_scheduler.t_precision
    assert torch.equal(times.float(), scheduler.sigmas[:-1])
    assert not torch.equal(times.float(), scheduler.timesteps.float() * 0.001)


def test_cosmos_predict2_defaults_to_the_native_sigma_preset():
    # Upstream's default preset; only the released 2B config selects Karras.
    assert _tiny_cosmos_predict2().sample_sigma_schedule == "linear"


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_cosmos_predict2_sampler_rejects_invalid_step_counts(steps):
    with pytest.raises(ValueError, match="positive integer"):
        _tiny_cosmos_predict2().sample_scheduler_and_grid(steps)


@pytest.mark.parametrize(
    ("preset", "count", "first", "second", "last", "first_time", "last_time"),
    [
        # The native Karras preset ignores the shift.
        ("karras", 36, 0.9950248599052429, 0.9942067265510559, 0.009900989942252636, 995, 9),
        ("linear", 35, 0.9996664524078369, 0.9899545311927795, 0.08100438117980957, 999, 81),
    ],
)
def test_cosmos_predict2_presets_reproduce_native_sigma_grids_at_a_custom_shift(
    preset, count, first, second, last, first_time, last_time
):
    """35-step grids captured from native FlowUniPC at shift 3."""
    net = _tiny_cosmos_predict2(sample_sigma_schedule=preset, max_t=1.0)
    scheduler, _ = net.sample_scheduler_and_grid(35, flow_shift=3.0)
    assert len(scheduler.timesteps) == count
    assert scheduler.sigmas[[0, 1, -2]].tolist() == [first, second, last]
    assert scheduler.timesteps[[0, -1]].tolist() == [first_time, last_time]


@pytest.mark.parametrize("shift", [0.0, -1.0, float("inf"), float("nan")])
def test_cosmos_predict2_sampler_rejects_invalid_shifts(shift):
    with pytest.raises(ValueError, match="finite and positive"):
        _tiny_cosmos_predict2().sample_scheduler_and_grid(2, flow_shift=shift)


def test_cosmos_predict2_sample_queries_the_network_at_the_solver_sigmas(monkeypatch):
    net = _tiny_cosmos_predict2(sample_sigma_schedule="linear", max_t=1.0)
    times = []

    def zero_velocity(x_t, t, output_dtype="same", **kwargs):
        times.append(t.clone())
        return cast_like(torch.zeros_like(x_t), x_t, output_dtype)

    monkeypatch.setattr(net, "forward", zero_velocity)
    net.sample(torch.randn(1, 2, 3, 2, 2), sample_steps=2, shift=3.0, guidance_cfg=GuidanceConfig())
    sigmas = net.sample_scheduler.sigmas[:-1].clone()
    assert torch.equal(torch.stack(times)[:, 0].float(), sigmas)
    assert not torch.equal(torch.stack(times)[:, 0].float(), net.sample_scheduler.timesteps.float() * 0.001)
    assert not torch.equal(sigmas, net.sample_scheduler_and_grid(2)[0].sigmas[:-1])


@pytest.mark.parametrize("preset", ["karras", "linear"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_cosmos_predict2_sample_starts_at_the_first_sigma_and_keeps_state_dtype(preset, dtype, monkeypatch):
    net = _tiny_cosmos_predict2(sample_sigma_schedule=preset)
    noise = torch.randn(2, 2, 3, 2, 2).to(dtype)
    initial = noise.clone()
    states, times = [], []

    def zero_velocity(x_t, t, output_dtype="same", **kwargs):
        states.append(x_t.clone())
        times.append(t.clone())
        return cast_like(torch.zeros_like(x_t), x_t, output_dtype)

    monkeypatch.setattr(net, "forward", zero_velocity)
    result = net.sample(noise, sample_steps=2, guidance_cfg=GuidanceConfig())
    assert torch.equal(states[0], net.noise_scheduler.latents(initial, t_init=times[0][0]))
    assert torch.equal(noise, initial)
    assert result.dtype == dtype
    assert all(x.dtype == dtype for x in states)
    assert torch.isfinite(result).all()


def test_cosmos_predict2_per_frame_time_converts_per_frame_and_logvar_matches_consumers():
    """A per-frame t converts each frame at its own time; logvar is one [B, 1] value per sample, as consumers
    (sCM) expect, from the mean time embedding of the generated frames."""
    net = CosmosPredict2(
        model_channels=48,
        num_blocks=1,
        num_heads=4,
        max_img_h=4,
        max_img_w=4,
        max_frames=4,
        in_channels=2,
        out_channels=2,
        patch_spatial=1,
        patch_temporal=1,
        concat_padding_mask=False,
        add_video_condition_mask=True,
        mlp_ratio=2.0,
        crossattn_emb_channels=8,
        use_adaln_lora=False,
        enable_logvar_linear=True,
        use_crossattn_projection=False,
        use_wan_fp32_strategy=False,
    ).eval()
    torch.manual_seed(0)
    x_t = torch.randn(2, 2, 3, 2, 2)  # three frames against two channels: a misaligned time cannot broadcast
    text = torch.randn(2, 3, 8)
    t_dtype = net.noise_scheduler.t_precision
    timestep = torch.tensor([0.3, 0.7], dtype=t_dtype)
    per_frame = torch.tensor([[0.2, 0.5, 0.8], [0.4, 0.6, 0.9]], dtype=t_dtype)
    mask = torch.zeros(2, 1, 3, 2, 2, dtype=torch.bool)
    mask[:, :, :1] = True
    v2w = {"text_embeds": text, "conditioning_latents": torch.randn(2, 2, 1, 2, 2), "condition_mask": mask}

    def logvar_of_mean_embedding(times):
        dit = net.transformer
        return dit.logvar_linear(dit.t_embedding_norm(dit.t_embedder(times.float())[0]).mean(dim=1))

    with torch.no_grad():
        flow, logvar = net(x_t, timestep, condition=text, return_logvar=True)
        repeated_flow, repeated_logvar = net(x_t, timestep[:, None].expand(2, 3), condition=text, return_logvar=True)
        frame_flow = net(x_t, per_frame, condition=text)
        frame_x0, frame_logvar = net(x_t, per_frame, condition=text, fwd_pred_type="x0", return_logvar=True)
        _, v2w_logvar = net(x_t, timestep, condition=v2w, return_logvar=True)
        _, v2w_frame_logvar = net(x_t, per_frame, condition=v2w, return_logvar=True)
        expected_frame_logvar = logvar_of_mean_embedding(per_frame)
        expected_v2w_frame_logvar = logvar_of_mean_embedding(per_frame[:, 1:])

    for value in (logvar, repeated_logvar, frame_logvar, v2w_logvar, v2w_frame_logvar):
        assert value.shape == (2, 1)
    torch.testing.assert_close(repeated_flow, flow)
    torch.testing.assert_close(repeated_logvar, logvar)
    # Video2world's generated frames carry the caller's t, so the per-sample logvar is unchanged.
    torch.testing.assert_close(v2w_logvar, logvar)
    # A per-frame t averages the time embeddings of the generated frames; the conditioning frame is excluded.
    torch.testing.assert_close(frame_logvar, expected_frame_logvar)
    torch.testing.assert_close(v2w_frame_logvar, expected_v2w_frame_logvar)
    # Rectified flow: x0 = x_t - t * v, frame by frame.
    torch.testing.assert_close(frame_x0, x_t - per_frame.reshape(2, 1, 3, 1, 1).to(x_t.dtype) * frame_flow)


def test_cosmos_predict2_split_boundary_matches_reference_input_casts(monkeypatch):
    network = CosmosPredict2(
        model_channels=48,
        num_blocks=1,
        num_heads=4,
        max_img_h=4,
        max_img_w=4,
        max_frames=4,
        in_channels=2,
        out_channels=2,
        patch_spatial=1,
        patch_temporal=1,
        concat_padding_mask=False,
        add_video_condition_mask=False,
        mlp_ratio=2.0,
        crossattn_emb_channels=8,
        use_adaln_lora=False,
        enable_logvar_linear=False,
        use_crossattn_projection=False,
        use_wan_fp32_strategy=False,
    ).float()
    network._record_precision(precision_model=torch.bfloat16)

    captured = {}

    def transformer_forward(**kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["x_B_C_T_H_W"])

    monkeypatch.setattr(network.transformer, "forward", transformer_forward)
    state = torch.randn(1, 2, 2, 2, 2, dtype=torch.float64)
    timestep = torch.tensor([0.5], dtype=torch.float64)
    condition = torch.randn(1, 3, 8, dtype=network.precision_model)
    padding_mask = torch.ones(1, 2, 2, dtype=torch.bool)
    output = network(
        state,
        timestep,
        condition=condition,
        fps=torch.tensor([24.0], dtype=torch.float64),
        padding_mask=padding_mask,
    )

    assert output.dtype == state.dtype
    assert captured["x_B_C_T_H_W"].dtype == torch.bfloat16
    assert captured["crossattn_emb"].dtype == torch.bfloat16
    assert captured["timesteps_B_T"].dtype == torch.float32
    assert torch.equal(captured["timesteps_B_T"], timestep.float())
    assert captured["fps"].dtype == torch.bfloat16
    assert captured["padding_mask"].dtype == torch.bool

    # Text in another dtype is rejected rather than cast to the model input's dtype.
    with pytest.raises(AssertionError, match=r"text_embeds\.dtype"):
        network(
            state,
            timestep,
            condition=condition.double(),
            fps=torch.tensor([24.0], dtype=torch.float64),
            padding_mask=padding_mask,
        )

    for incomplete_condition in (
        {"text_embeds": condition, "conditioning_latents": torch.zeros_like(state[:, :, :1])},
        {"text_embeds": condition, "condition_mask": torch.zeros_like(state[:, :1])},
    ):
        with pytest.raises(ValueError, match="conditioning_latents and condition_mask"):
            network(state, timestep, condition=incomplete_condition)

    conditioning_latents = torch.randn(1, 2, 1, 2, 2, dtype=state.dtype)
    condition_mask = torch.zeros(1, 1, 2, 2, 2, dtype=torch.bool)
    condition_mask[:, :, :1] = 1
    structured_condition = {
        "text_embeds": condition,
        "conditioning_latents": conditioning_latents,
        "condition_mask": condition_mask,
    }
    structured_output = network(
        state,
        timestep,
        condition=structured_condition,
        fps=torch.tensor([24.0], dtype=torch.float64),
        padding_mask=padding_mask,
    )
    assert structured_output.dtype == state.dtype

    # Narrower clean latents are cast exactly to the state's dtype, so they condition like their upcast copy.
    narrow_latents = conditioning_latents.to(torch.float32)
    narrow_output, upcast_output = [
        network(
            state,
            timestep,
            condition={**structured_condition, "conditioning_latents": latents},
            fps=torch.tensor([24.0], dtype=torch.float64),
            padding_mask=padding_mask,
        )
        for latents in (narrow_latents, narrow_latents.to(state.dtype))
    ]
    assert torch.equal(narrow_output, upcast_output)

    # A condition mask in another dtype is rejected rather than cast.
    with pytest.raises(AssertionError, match=r"condition_mask\.dtype"):
        network(
            state,
            timestep,
            condition={**structured_condition, "condition_mask": condition_mask.to(torch.float32)},
            fps=torch.tensor([24.0], dtype=torch.float64),
            padding_mask=padding_mask,
        )


def test_cosmos_predict2_preserve_conditioning_requires_a_boolean_mask_and_the_state_device():
    network = CosmosPredict2.__new__(CosmosPredict2)
    state = torch.randn(1, 2, 2, 2, 2, dtype=torch.bfloat16)
    conditioning_latents = torch.randn(1, 2, 1, 2, 2, dtype=state.dtype)
    condition_mask = torch.zeros(1, 1, 2, 2, 2, dtype=torch.bool)
    condition_mask[:, :, :1] = 1
    condition = {
        "conditioning_latents": conditioning_latents,
        "condition_mask": condition_mask,
    }

    actual = network.preserve_conditioning(state, condition)
    expected = state.clone()
    expected[:, :, :1] = conditioning_latents
    torch.testing.assert_close(actual, expected)

    # Copying the clean latents into a wider state is exact.
    wide_actual = network.preserve_conditioning(state.float(), condition)
    assert wide_actual.dtype == torch.float32
    assert torch.equal(wide_actual, expected.float())

    with pytest.raises(AssertionError, match=r"condition_mask\.dtype"):
        network.preserve_conditioning(state, {**condition, "condition_mask": condition_mask.to(torch.float32)})

    for name, value in (
        ("conditioning_latents", conditioning_latents.to("meta")),
        ("condition_mask", condition_mask.to("meta")),
    ):
        with pytest.raises(AssertionError, match=name):
            network.preserve_conditioning(state, {**condition, name: value})

    for incomplete_condition in (
        {"conditioning_latents": conditioning_latents},
        {"condition_mask": condition_mask},
    ):
        with pytest.raises(ValueError, match="conditioning_latents and condition_mask"):
            network.preserve_conditioning(state, incomplete_condition)


@pytest.mark.parametrize("fwd_pred_type", ["x0", "eps", "flow"])
def test_cosmos_predict2_v2w_forward_feeds_clean_conditioning_frames_and_pins_them(fwd_pred_type):
    """V2W: the DiT sees the clean conditioning frame and the output round-trips to it in every prediction type.

    In the flow domain the pin is the reference's ``denoise_replace_gt_frames`` velocity ``noise - gt_frames``.
    """
    net = CosmosPredict2(
        model_channels=48,
        num_blocks=1,
        num_heads=4,
        max_img_h=4,
        max_img_w=4,
        max_frames=4,
        in_channels=2,
        out_channels=2,
        patch_spatial=1,
        patch_temporal=1,
        concat_padding_mask=False,
        add_video_condition_mask=True,
        mlp_ratio=2.0,
        crossattn_emb_channels=8,
        use_adaln_lora=False,
        enable_logvar_linear=False,
        use_crossattn_projection=False,
        use_wan_fp32_strategy=False,
        is_video2world=True,
    ).eval()
    torch.manual_seed(0)
    ns = net.noise_scheduler
    x0 = torch.randn(2, 2, 3, 2, 2)
    eps = torch.randn_like(x0)
    timestep = torch.tensor([0.3, 0.7], dtype=ns.t_precision)
    x_t = ns.forward_process(x0, eps, timestep)
    clean = x0[:, :, :1].clone()
    condition_mask = torch.zeros(2, 1, 3, 2, 2, dtype=torch.bool)
    condition_mask[:, :, :1] = 1
    condition = {"text_embeds": torch.randn(2, 3, 8), "conditioning_latents": clean, "condition_mask": condition_mask}

    seen_inputs, raw_velocities = [], []
    transformer_forward = net.transformer.forward

    def capture(**kwargs):
        seen_inputs.append(kwargs["x_B_C_T_H_W"].clone())
        result = transformer_forward(**kwargs)
        raw_velocities.append(result.clone())
        return result

    net.transformer.forward = capture
    with torch.no_grad():
        output = net(x_t, timestep, condition=condition, fwd_pred_type=fwd_pred_type)

    assert torch.equal(seen_inputs[0][:, :, :1], clean)
    assert torch.equal(seen_inputs[0][:, :, 1:], x_t[:, :, 1:])
    x0_pred = ns.convert_model_output(x_t, output, timestep, src_pred_type=fwd_pred_type, target_pred_type="x0")
    torch.testing.assert_close(x0_pred[:, :, :1], clean, rtol=1e-5, atol=1e-5)
    if fwd_pred_type == "x0":
        assert torch.equal(output[:, :, :1], clean)
    if fwd_pred_type == "flow":
        torch.testing.assert_close(output[:, :, :1], (eps - x0)[:, :, :1], rtol=1e-5, atol=1e-5)
    expected = ns.convert_model_output(
        x_t, raw_velocities[0], timestep, src_pred_type="flow", target_pred_type=fwd_pred_type
    )
    torch.testing.assert_close(output[:, :, 1:], expected[:, :, 1:])


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("use_wan_fp32_strategy", [False, True])
def test_cosmos_predict2_fp32_strategy_controls_local_amp(device_type, use_wan_fp32_strategy):
    network = CosmosPredict2(
        model_channels=48,
        num_blocks=1,
        num_heads=4,
        max_img_h=4,
        max_img_w=4,
        max_frames=4,
        in_channels=2,
        out_channels=2,
        patch_spatial=1,
        patch_temporal=1,
        concat_padding_mask=False,
        add_video_condition_mask=False,
        mlp_ratio=2.0,
        crossattn_emb_channels=8,
        use_adaln_lora=False,
        enable_logvar_linear=False,
        use_crossattn_projection=False,
        use_wan_fp32_strategy=use_wan_fp32_strategy,
    ).to(device=device_type, dtype=torch.float32)
    network._record_precision(precision_model=torch.float32)
    operator_states = {}

    def capture_operator(name):
        def hook(module, args, output):
            del module
            operator_states[name] = (
                torch.is_autocast_enabled(device_type),
                args[0].dtype,
                output.dtype,
            )

        return hook

    timestep_linear = network.transformer.t_embedder[1].linear_1
    modulation_linear = network.transformer.blocks[0].adaln_modulation_self_attn[1]
    handles = [
        timestep_linear.register_forward_hook(capture_operator("timestep")),
        modulation_linear.register_forward_hook(capture_operator("modulation")),
    ]
    try:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            network(
                torch.randn(1, 2, 2, 2, 2, device=device_type),
                torch.tensor([0.5], dtype=network.noise_scheduler.t_precision, device=device_type),
                condition=torch.randn(1, 3, 8, device=device_type),
                fps=torch.tensor([24.0], device=device_type),
                padding_mask=torch.ones(1, 2, 2, dtype=torch.bool, device=device_type),
            )
    finally:
        for handle in handles:
            handle.remove()

    expected_amp_enabled = not use_wan_fp32_strategy
    expected_output_dtype = torch.float32 if use_wan_fp32_strategy else torch.bfloat16
    assert operator_states == {
        "timestep": (expected_amp_enabled, torch.float32, expected_output_dtype),
        "modulation": (expected_amp_enabled, torch.float32, expected_output_dtype),
    }


@RunIf(min_gpus=1)
def test_cosmos_predict2_fp32_strategy_uses_fp32_operations_over_bfloat16_storage():
    network = CosmosPredict2(
        model_channels=48,
        num_blocks=1,
        num_heads=4,
        max_img_h=4,
        max_img_w=4,
        max_frames=4,
        in_channels=2,
        out_channels=2,
        patch_spatial=1,
        patch_temporal=1,
        concat_padding_mask=False,
        add_video_condition_mask=False,
        mlp_ratio=2.0,
        crossattn_emb_channels=8,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        enable_logvar_linear=False,
        use_crossattn_projection=False,
        use_wan_fp32_strategy=True,
    ).to(device="cuda", dtype=torch.bfloat16)
    network._record_precision(precision_model=torch.bfloat16)
    operator_states = {}

    def capture_operator(name):
        def hook(module, args, output):
            del module
            operator_states[name] = (
                torch.is_autocast_enabled("cuda"),
                args[0].dtype,
                output.dtype,
            )

        return hook

    timestep_linear = network.transformer.t_embedder[1].linear_1
    modulation_linear = network.transformer.blocks[0].adaln_modulation_self_attn[2]
    handles = [
        timestep_linear.register_forward_hook(capture_operator("timestep")),
        modulation_linear.register_forward_hook(capture_operator("modulation")),
    ]
    try:
        with torch.no_grad():
            output = network(
                torch.randn(1, 2, 2, 2, 2, device="cuda", dtype=torch.bfloat16),
                torch.tensor([0.5], device="cuda", dtype=network.noise_scheduler.t_precision),
                condition=torch.randn(1, 3, 8, device="cuda", dtype=torch.bfloat16),
                fps=torch.tensor([24.0], device="cuda", dtype=torch.bfloat16),
                padding_mask=torch.ones(1, 2, 2, device="cuda", dtype=torch.bool),
            )
    finally:
        for handle in handles:
            handle.remove()

    assert operator_states == {
        "timestep": (False, torch.float32, torch.float32),
        "modulation": (False, torch.float32, torch.float32),
    }
    assert all(p.dtype == torch.bfloat16 for p in network.parameters())
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("storage", [torch.bfloat16, torch.float32, torch.float64])
def test_cosmos_predict2_fp32_linear_matches_explicit_reference_operation_and_backward(storage):
    """The operation runs in at least FP32 under any autocast; gradients return in the storage dtype."""
    from fastgen.networks.cosmos_predict2.modules import FP32Linear

    torch.manual_seed(17)
    layer = FP32Linear(8, 5).to(storage)
    compute_dtype = torch.promote_types(storage, torch.float32)
    x = torch.randn(3, 8, dtype=storage, requires_grad=True)
    weight = layer.weight.detach().to(compute_dtype).requires_grad_()
    bias = layer.bias.detach().to(compute_dtype).requires_grad_()
    reference_x = x.detach().to(compute_dtype).requires_grad_()
    expected = F.linear(reference_x, weight, bias)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = layer(x)
    assert actual.dtype == compute_dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(layer.weight.grad, weight.grad.to(storage), rtol=0, atol=0)
    torch.testing.assert_close(x.grad, reference_x.grad.to(storage), rtol=0, atol=0)
    assert layer.weight.dtype == storage


def test_cosmos_predict2_final_layer_keeps_modulation_and_output_in_fp32():
    from fastgen.networks.cosmos_predict2.modules import FinalLayer

    layer = FinalLayer(8, 1, 1, 2, use_wan_fp32_strategy=True).bfloat16()
    x = torch.randn(1, 2, 2, 2, 8, dtype=torch.bfloat16, requires_grad=True)
    embedding = torch.randn(1, 2, 8, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = layer(x, embedding)
    mod = F.linear(F.silu(embedding), layer.adaln_modulation[1].weight.float())
    shift, scale = (v[:, :, None, None] for v in mod.chunk(2, dim=-1))
    expected = F.linear(F.layer_norm(x.float(), (8,), eps=1e-6) * (1 + scale) + shift, layer.linear.weight.float())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert actual.dtype == torch.float32
    assert layer.linear.weight.dtype == layer.linear.weight.grad.dtype == torch.bfloat16
    assert torch.isfinite(x.grad).all() and torch.isfinite(embedding.grad).all()


@pytest.mark.parametrize("use_wan_fp32_strategy", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_cosmos_predict2_bf16_training_with_features_and_logvar_preserves_storage_and_output_contract(
    use_wan_fp32_strategy, state_dtype
):
    net = _tiny_cosmos_predict2(use_wan_fp32_strategy=use_wan_fp32_strategy, enable_logvar_linear=True)
    net.prepare_module("cpu", precision_model=torch.bfloat16, trainable=True)
    x = torch.randn(2, 2, 3, 2, 2).to(state_dtype).requires_grad_()
    t = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    text = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    (out, features), logvar = net(x, t, condition=text, feature_indices={0}, return_logvar=True)
    assert out.dtype == logvar.dtype == state_dtype
    assert out.shape == x.shape and logvar.shape == (2, 1) and len(features) == 1
    (out.float().square().mean() + logvar.float().square().mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in net.named_parameters():
        assert parameter.dtype == torch.bfloat16, name
        if parameter.grad is not None:
            assert parameter.grad.dtype == parameter.dtype and torch.isfinite(parameter.grad).all(), name
    assert net.transformer.logvar_linear.weight.grad is not None


def test_cosmos3_eval_grad_enabled_forward_uses_checkpoint(monkeypatch):
    net = _tiny_cosmos3(gradient_checkpointing=True)
    assert not net.training
    calls = []

    def fake_checkpoint(fn, *args, **kwargs):
        calls.append(kwargs)
        return fn(*args)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", fake_checkpoint)

    batch_size, channels, frames, height, width = 1, 8, 2, 4, 4
    x_t = torch.randn(batch_size, channels, frames, height, width, requires_grad=True)
    timestep = torch.tensor([0.3], dtype=net.noise_scheduler.t_precision)
    condition = {"input_ids": [torch.randint(0, 100, (5,))]}

    output = net(x_t, timestep, condition=condition)
    assert output.requires_grad
    assert len(calls) == net.transformer.num_hidden_layers
    assert all(call.get("use_reentrant") is False for call in calls)

    calls.clear()
    with torch.no_grad():
        net(x_t.detach(), timestep, condition=condition)
    assert calls == []


def test_cosmos3_build_conditioning_t2v_reuses_encoded_condition():
    """An already encoded branch is returned as is without tokenization."""
    net = Cosmos3.__new__(Cosmos3)
    net.preprocessors = {}
    net.is_i2v = False
    input_ids = [torch.zeros(3, dtype=torch.long)]
    data = {"condition": {"input_ids": input_ids}}

    out = net.build_conditioning(data, input_shape=[48, 4, 8, 8])

    assert set(out) == {"condition"}
    assert out["condition"] is data["condition"]
    assert out["condition"]["input_ids"] is input_ids


def test_cosmos3_build_conditioning_requires_encoded_mapping():
    net = Cosmos3.__new__(Cosmos3)
    net.preprocessors = {}
    net.is_i2v = False

    with pytest.raises(TypeError, match=r"condition.*mapping"):
        net.build_conditioning(
            {"condition": [torch.zeros(3, dtype=torch.long)]},
            input_shape=[48, 4, 8, 8],
        )


def test_cosmos3_build_conditioning_requires_positive_condition():
    net = Cosmos3.__new__(Cosmos3)
    net.preprocessors = {}
    net.is_i2v = False

    with pytest.raises(ValueError, match=r"requires non-None data\['condition'\]"):
        net.build_conditioning({}, input_shape=[48, 4, 8, 8])


def test_cosmos3_single_frame_geometry_uses_image_templates():
    """One frame is text2image in the native framework and diffusers: image template, no duration."""
    net = Cosmos3.__new__(Cosmos3)
    net.fps = 24.0
    tokenizer, conversations = _stub_cosmos3_tokenizer()

    kwargs = net.text_encode_kwargs("condition", input_shape=[48, 1, 8, 8])
    assert kwargs["num_frames"] == 1
    assert kwargs["is_image"] is True
    tokenizer.tokenize_prompt("a cat", **kwargs)
    assert conversations[-1] == [{"role": "user", "content": "a cat. This image is of 128x128 resolution."}]

    tokenizer.tokenize_prompt("", **net.text_encode_kwargs("neg_condition", input_shape=[48, 1, 8, 8]))
    assert conversations[-1] == [{"role": "user", "content": "This image is of 128x128 resolution."}]


def test_cosmos3_build_conditioning_encodes_raw_strings():
    """Cosmos3 tokenizes only explicit raw keys with the real clip geometry."""
    net = Cosmos3.__new__(Cosmos3)
    net.preprocessors = {}
    net.is_i2v = False
    net.fps = 24.0

    encode_calls = []
    encoded = []

    class _StubTokenizer:
        def encode(self, strings, **kwargs):
            encode_calls.append((strings, kwargs))
            result = {"input_ids": [torch.zeros(3, dtype=torch.long)] * len(strings)}
            encoded.append(result)
            return result

    net.preprocessors["text_tokenizer"] = _StubTokenizer()
    data = {"condition_raw": ["a cat"], "neg_condition_raw": ["not a cat"]}
    out = net.build_conditioning(data, input_shape=[48, 4, 8, 8])

    assert set(out) == {"condition", "neg_condition"}
    assert set(data) == {"condition_raw", "neg_condition_raw"}
    assert [c[0] for c in encode_calls] == [["a cat"], ["not a cat"]]
    assert out["condition"] is encoded[0]
    assert out["neg_condition"] is encoded[1]
    # Real geometry (13f = (4-1)*4+1, 128x128 = 8*16) + the native prompt defaults for both branches.
    assert encode_calls[0][1] == {
        "num_frames": 13,
        "height": 128,
        "width": 128,
        "fps": 24.0,
        "is_image": False,
        "is_negative": False,
        "use_system_prompt": False,
        "negative_metadata_mode": "same",
    }
    assert encode_calls[1][1]["is_negative"] is True


def test_cosmos3_build_conditioning_reuses_encoded_primary_over_raw_mirror():
    """A cached branch is never re-tokenized from its raw mirror; raw peers use the configured clock."""
    net = Cosmos3.__new__(Cosmos3)
    net.preprocessors = {}
    net.is_i2v = False
    net.fps = 24.0

    encode_calls = []

    class _StubTokenizer:
        def encode(self, strings, **kwargs):
            encode_calls.append((strings, kwargs))
            return {"input_ids": [torch.zeros(3, dtype=torch.long)] * len(strings)}

    net.preprocessors["text_tokenizer"] = _StubTokenizer()
    cached_negative = {"input_ids": [torch.ones(3, dtype=torch.long)]}
    data = {
        "condition_raw": ["a cat"],
        "neg_condition": cached_negative,
        "neg_condition_raw": ["not a cat"],
    }
    out = net.build_conditioning(data, input_shape=[48, 4, 8, 8])

    assert [call[0] for call in encode_calls] == [["a cat"]]
    assert encode_calls[0][1]["fps"] == 24.0
    assert set(out) == {"condition", "neg_condition"}
    assert out["neg_condition"] is cached_negative


def test_cosmos3_build_conditioning_i2v_mask():
    net = Cosmos3.__new__(Cosmos3)
    net.preprocessors = {}
    net.is_i2v = True

    class _StubVAE:
        def encode(self, pixels, mode="sample"):
            assert mode == "argmax"
            b, _, t, h, w = pixels.shape
            return torch.zeros(b, 48, t, h // 16, w // 16)

    net.preprocessors["vae"] = _StubVAE()
    batch_size, latent_frames = 2, 4
    data = {
        "condition": {"input_ids": [torch.zeros(3, dtype=torch.long)] * batch_size},
        "neg_condition": {"input_ids": [torch.zeros(2, dtype=torch.long)] * batch_size},
        "real_raw": torch.randn(batch_size, 3, 13, 128, 128),
    }
    output = net.build_conditioning(data, input_shape=[48, latent_frames, 8, 8])
    assert set(output) == {"condition", "neg_condition"}
    condition_mask = output["condition"]["condition_mask"]
    assert condition_mask.shape == (batch_size, 1, latent_frames, 1, 1)
    assert condition_mask.dtype == torch.bool
    assert condition_mask[:, :, 0].all() and condition_mask[:, :, 1:].sum() == 0
    assert output["condition"]["input_ids"] is data["condition"]["input_ids"]
    # The anchor frame is VAE-encoded once and shared by both branches so
    # ``forward`` can feed it as the clean input and pin it in the output.
    conditioning_latents = output["condition"]["conditioning_latents"]
    assert conditioning_latents.shape == (batch_size, 48, 1, 8, 8)
    assert output["neg_condition"]["conditioning_latents"] is conditioning_latents


@pytest.mark.parametrize(
    "negative_field",
    [None, "neg_condition", "neg_condition_raw"],
    ids=["missing", "encoded_none", "raw_none"],
)
def test_cosmos3_build_conditioning_i2v_preserves_absent_negative(negative_field):
    net = Cosmos3.__new__(Cosmos3)
    net.preprocessors = {}
    net.is_i2v = True

    class _StubVAE:
        def encode(self, pixels, mode="sample"):
            assert mode == "argmax"
            return torch.zeros(pixels.shape[0], 48, 1, 1, 1)

    net.preprocessors["vae"] = _StubVAE()
    data = {
        "condition": {"input_ids": [torch.zeros(3, dtype=torch.long)]},
        "real_raw": torch.randn(1, 3, 1, 16, 16),
    }
    if negative_field is not None:
        data[negative_field] = None
    output = net.build_conditioning(data, input_shape=[48, 1, 1, 1])

    assert set(output) == {"condition"}


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16])
def test_cosmos3_preserve_conditioning_requires_boolean_mask_and_clean_latent_on_the_state_device(state_dtype):
    net = Cosmos3.__new__(Cosmos3)
    state = torch.arange(2 * 3 * 4 * 2 * 2, dtype=torch.float32).reshape(2, 3, 4, 2, 2).to(state_dtype)
    clean = torch.full((2, 3, 1, 2, 2), -3.5, dtype=state_dtype)
    condition_mask = torch.zeros(2, 1, 4, 1, 1, dtype=torch.bool)
    condition_mask[:, :, 0] = True

    actual = net.preserve_conditioning(
        state,
        {"condition_mask": condition_mask, "conditioning_latents": clean},
    )

    expected = state.clone()
    expected[:, :, :1] = clean
    assert actual.dtype == state_dtype
    assert torch.equal(actual, expected)
    assert torch.equal(actual[:, :, 1:], state[:, :, 1:])

    with pytest.raises(AssertionError, match=r"condition_mask\.dtype .*must be torch\.bool"):
        net.preserve_conditioning(
            state,
            {"condition_mask": condition_mask.to(torch.float32), "conditioning_latents": clean},
        )

    # Copying the clean latent into a wider state is exact.
    wide_actual = net.preserve_conditioning(
        state.float(),
        {"condition_mask": condition_mask, "conditioning_latents": clean},
    )
    assert wide_actual.dtype == torch.float32
    assert torch.equal(wide_actual, expected.float())

    with pytest.raises(AssertionError, match=r"conditioning_latents\.device"):
        net.preserve_conditioning(
            state,
            {"condition_mask": condition_mask, "conditioning_latents": clean.to("meta")},
        )

    for incomplete_condition in (
        {"conditioning_latents": clean},
        {"condition_mask": condition_mask},
    ):
        with pytest.raises(ValueError, match="conditioning_latents and condition_mask"):
            net.preserve_conditioning(state, incomplete_condition)


def test_cosmos3_build_conditioning_i2v_errors():
    net = Cosmos3.__new__(Cosmos3)
    net.preprocessors = {}
    net.is_i2v = True
    with pytest.raises(ValueError, match="requires input_shape"):
        net.build_conditioning(
            {"condition": {"input_ids": []}, "real_raw": torch.randn(1, 3, 2, 8, 8)},
            input_shape=None,
        )
    with pytest.raises(ValueError, match=r"requires data\['real_raw'\]"):
        net.build_conditioning({"condition": {"input_ids": []}}, input_shape=[48, 4, 8, 8])


def test_cosmos3_build_conditioning_rejects_reentrant_i2v_condition():
    net = Cosmos3.__new__(Cosmos3)
    net.preprocessors = {}
    net.is_i2v = True
    condition = {
        "input_ids": [torch.zeros(3, dtype=torch.long)],
        "condition_mask": torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
        "conditioning_latents": torch.zeros(1, 48, 1, 1, 1),
    }

    with pytest.raises(ValueError, match="cannot be prepared again"):
        net.build_conditioning(
            {"condition": condition, "real_raw": torch.randn(1, 3, 1, 16, 16)},
            input_shape=[48, 1, 1, 1],
        )


@pytest.mark.parametrize("is_causal,nq,nkv", [(True, 7, 7), (False, 5, 9), (True, 3, 7)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_cosmos3_attention_preserves_grouped_heads_and_backward(is_causal, nq, nkv, dtype):
    from fastgen.networks.cosmos3.modules import Cosmos3PackedMoTAttention

    q, k, v = [torch.randn(n, h, 16, dtype=dtype, requires_grad=True) for n, h in [(nq, 4), (nkv, 2), (nkv, 2)]]
    actual = Cosmos3PackedMoTAttention._sdpa(q, k, v, is_causal)
    reference = (
        F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.repeat_interleave(2, 1).transpose(0, 1)[None],
            v.repeat_interleave(2, 1).transpose(0, 1)[None],
            is_causal=is_causal,
        )[0]
        .transpose(0, 1)
        .flatten(-2, -1)
    )
    torch.testing.assert_close(
        actual,
        reference,
        rtol=0.01 if dtype == torch.bfloat16 else 1e-5,
        atol=0.01 if dtype == torch.bfloat16 else 1e-6,
    )
    actual.float().square().sum().backward()
    assert actual.dtype == dtype and all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))


@RunIf(min_gpus=1)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cosmos3_cuda_attention_supports_backward_and_jvp(causal, dtype):
    from fastgen.networks.cosmos3 import modules
    from fastgen.utils.tensor import temp_disable_efficient_attn

    q, k, v = [
        torch.randn(n, h, 16, device="cuda", dtype=dtype, requires_grad=True)
        for n, h in [(7, 4), (7 if causal else 9, 2), (7 if causal else 9, 2)]
    ]
    actual = modules.Cosmos3PackedMoTAttention._sdpa(q, k, v, causal)
    assert actual.shape == (7, 64) and actual.dtype == dtype
    actual.float().square().sum().backward()
    assert all(torch.isfinite(x.grad).all() for x in (q, k, v))

    with temp_disable_efficient_attn():
        _, tangent = torch.func.jvp(
            lambda value: modules.Cosmos3PackedMoTAttention._sdpa(value, k.detach(), v.detach(), causal),
            (q.detach(),),
            (torch.ones_like(q),),
        )
    assert torch.isfinite(tangent).all()
    # Double precision remains supported outside the JVP context.
    assert modules.Cosmos3PackedMoTAttention._sdpa(q.double(), k.double(), v.double(), causal).dtype == torch.float64


def _assert_cosmos3_rope_prefix_invariant(device: torch.device, dtype: torch.dtype):
    understanding_length = 179
    patch_height, patch_width = 23, 40
    fps = 16.0
    text_ids, next_offset = get_3d_mrope_ids_text_tokens(understanding_length, 0.0, use_float_positions=True)
    vision_offset = next_offset + 100000
    vision_ids_21, _ = get_3d_mrope_ids_vae_tokens(21, patch_height, patch_width, vision_offset, True, fps, 24.0, 4)
    vision_ids_3, _ = get_3d_mrope_ids_vae_tokens(3, patch_height, patch_width, vision_offset, True, fps, 24.0, 4)
    position_ids_21 = torch.cat([text_ids, vision_ids_21], dim=1).to(device)
    position_ids_3 = torch.cat([text_ids, vision_ids_3], dim=1).to(device)
    vision_length_3 = vision_ids_3.shape[1]

    rope = Cosmos3VLTextRotaryEmbedding(128, 5000000.0, (24, 20, 20)).to(device=device, dtype=dtype)
    cos_21, sin_21 = rope(position_ids_21, device=device, dtype=dtype)
    cos_3, sin_3 = rope(position_ids_3, device=device, dtype=dtype)

    torch.testing.assert_close(
        cos_21[understanding_length : understanding_length + vision_length_3],
        cos_3[understanding_length:],
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        sin_21[understanding_length : understanding_length + vision_length_3],
        sin_3[understanding_length:],
        rtol=0,
        atol=1e-6,
    )


def test_cosmos3_rope_prefix_invariant_cpu():
    _assert_cosmos3_rope_prefix_invariant(torch.device("cpu"), torch.float32)


def test_cosmos3_reset_inv_freq_preserves_prepared_storage():
    rope = Cosmos3VLTextRotaryEmbedding(128, 5000000.0, (24, 20, 20)).to(dtype=torch.bfloat16)
    storage = rope.inv_freq
    expected = storage.clone()
    storage.zero_()

    rope.reset_inv_freq()

    assert rope.inv_freq is storage
    assert rope.inv_freq.dtype is torch.bfloat16
    assert torch.equal(rope.inv_freq, expected)


def test_cosmos_predict2_rope_reset_preserves_prepared_storage():
    from fastgen.networks.cosmos_predict2.modules import VideoRopePosition3DEmb

    rope = VideoRopePosition3DEmb(head_dim=12, len_h=8, len_w=6, len_t=4).to(dtype=torch.bfloat16)
    storage = {name: buffer for name, buffer in rope.named_buffers()}
    expected = {name: buffer.clone() for name, buffer in storage.items()}
    for buffer in storage.values():
        buffer.zero_()

    rope.reset_parameters()

    for name, buffer in rope.named_buffers():
        assert buffer is storage[name]
        assert buffer.dtype is torch.bfloat16
        assert torch.equal(buffer, expected[name])


def test_cosmos_predict2_meta_materialization_takes_persistent_rope_tables_from_the_state():
    """The RoPE tables are persistent, so rank 0's state restores them after meta materialization."""
    from fastgen.utils.module import materialize_meta_module

    kwargs = dict(
        model_channels=48,
        num_blocks=1,
        num_heads=4,
        max_img_h=4,
        max_img_w=4,
        max_frames=4,
        in_channels=2,
        out_channels=2,
        patch_spatial=1,
        patch_temporal=1,
        enable_logvar_linear=False,
        use_crossattn_projection=False,
    )
    source = CosmosPredict2(**kwargs)
    with torch.device("meta"):
        network = CosmosPredict2(**kwargs)

    materialize_meta_module(network, "cpu")
    network.load_state_dict(source.state_dict(), strict=True)

    for name, table in source.transformer.pos_embedder.named_buffers():
        assert torch.equal(network.transformer.pos_embedder.get_buffer(name), table)


def test_cosmos_predict2_low_precision_rope_rotates_in_fp32_and_rounds_once():
    """Like Transformer Engine's fused kernel, RoPE rotates in FP32 and rounds once, with or without the Wan cast."""
    from fastgen.networks.cosmos_predict2.modules import VideoRopePosition3DEmb, apply_rotary_pos_emb_simple

    rope = VideoRopePosition3DEmb(head_dim=12, len_h=64, len_w=64, len_t=4)
    rope_emb = rope.generate_embeddings(torch.Size([1, 1, 64, 64, 12]))  # FP32 angles up to 63 rad
    x = torch.randn(1, rope_emb.shape[0], 2, 12, dtype=torch.bfloat16)

    angles = rope_emb[:, 0, 0, :6][None, :, None]
    cos, sin = angles.cos().repeat(1, 1, 1, 2), angles.sin().repeat(1, 1, 1, 2)
    x_fp32 = x.float()
    x_rotated = torch.cat([-x_fp32[..., 6:], x_fp32[..., :6]], dim=-1)
    expected = (x_fp32 * cos + x_rotated * sin).bfloat16()
    actual = apply_rotary_pos_emb_simple(x, rope_emb)
    assert torch.equal(actual, expected)
    assert torch.equal(apply_rotary_pos_emb_simple(x.float(), rope_emb).bfloat16(), actual)


@RunIf(min_gpus=1)
def test_cosmos_predict2_normalization_and_rotary_support_forward_ad():
    """JVP-based methods such as sCM and MeanFlow differentiate through these operations on CUDA."""
    from fastgen.networks.cosmos_predict2.modules import RMSNorm, apply_rotary_pos_emb_simple

    norm = RMSNorm(128).cuda()
    x = torch.randn(2, 7, 3, 128, device="cuda")
    tangent = torch.randn_like(x)
    angles = torch.randn(7, 1, 1, 64, device="cuda").repeat(1, 1, 1, 2)

    def fn(value):
        return apply_rotary_pos_emb_simple(norm(value), angles)

    primal, derivative = torch.func.jvp(fn, (x,), (tangent,))
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, tangent)
        expected, expected_derivative = torch.autograd.forward_ad.unpack_dual(fn(dual))
    torch.testing.assert_close(primal, expected)
    torch.testing.assert_close(derivative, expected_derivative)
    assert torch.isfinite(derivative).all()


@pytest.mark.parametrize("trainable", [False, True])
def test_cosmos3_bf16_preparation_keeps_rotary_frequencies_fp32(trainable):
    """The reference asserts FP32 rotary frequencies; BF16 frozen and trainable networks keep them exact."""
    network = _tiny_cosmos3()
    reference = Cosmos3VLTextRotaryEmbedding(16, 1e4, (4, 2, 2))

    network.prepare_module(device="cpu", precision_model=torch.bfloat16, trainable=trainable)

    rope = network.transformer.rotary_emb
    assert network.transformer.proj_in.weight.dtype == torch.bfloat16
    assert rope.inv_freq.dtype == torch.float32 and torch.equal(rope.inv_freq, reference.inv_freq)
    # Positions as far out as the released model's (text, then the vision segment past a 15000 margin).
    position_ids = torch.arange(0.0, 20000.0, 997.0).expand(3, -1)
    actual = rope(position_ids, "cpu", torch.bfloat16)
    expected = reference(position_ids, "cpu", torch.bfloat16)
    assert all(torch.equal(cos_or_sin, ref) for cos_or_sin, ref in zip(actual, expected))


def test_cosmos3_meta_materialization_rebuilds_fp32_rotary_frequencies():
    from fastgen.utils.module import materialize_meta_module

    with torch.device("meta"):
        network = _tiny_cosmos3()
    network._set_precision(precision_model=torch.bfloat16)

    materialize_meta_module(network, "cpu")

    inv_freq = network.transformer.rotary_emb.inv_freq
    assert inv_freq.dtype == torch.float32
    assert torch.equal(inv_freq, Cosmos3VLTextRotaryEmbedding(16, 1e4, (4, 2, 2)).inv_freq)


@RunIf(min_gpus=1)
def test_cosmos3_rope_prefix_invariant_with_cuda_tf32():
    matmul_backend = torch.backends.cuda.matmul
    previous = matmul_backend.allow_tf32
    try:
        # Keep the test process on the same API family used by torch.compile;
        # the maintained Inductor build still reads this flag internally.
        matmul_backend.allow_tf32 = True
        _assert_cosmos3_rope_prefix_invariant(torch.device("cuda"), torch.bfloat16)
    finally:
        matmul_backend.allow_tf32 = previous


# =============================================================================
# MiniMax-H3
# =============================================================================
#
# The checkpoint is mixed precision: the two patch projections, the timestep MLP, the two
# output heads and the rotary buffer are float32 while the block stack is bfloat16, and both
# VAEs are float32 throughout.  Diffusers applies that only when ``from_pretrained`` builds a
# model -- ``ModelMixin.to`` warns that casting afterwards "can lead to inconsistent results"
# and then casts everything anyway.  ``MiniMaxH3`` therefore owns the carve-out itself: the
# network records exact state overrides through ``_fixed_precision_overrides`` and applies them at
# its pre-DDP/FSDP precision boundary. The tests below pin that policy down along with the checkpoint
# properties it rests on.
#
# These tests build on the meta device from default configs, so they need no weights or GPU.


MINIMAX_H3_TRANSFORMER_FP32 = ["proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "rope"]
MINIMAX_H3_VIDEO_VAE_FP32 = ["encoder", "decoder", "quant_conv", "post_quant_conv"]
MINIMAX_H3_AUDIO_VAE_FP32 = ["encoder", "decoder", "pre_block", "dec_in_proj", "mean_proj", "logs_proj"]


def _h3_dtype_map(module):
    """``{name: dtype}`` over every floating-point parameter and buffer of *module*."""
    return {
        name: tensor.dtype
        for name, tensor in itertools.chain(module.named_parameters(), module.named_buffers())
        if tensor is not None and tensor.is_floating_point()
    }


def _h3_is_pinned(name, patterns=MINIMAX_H3_TRANSFORMER_FP32):
    return any(pattern in name.split(".") for pattern in patterns)


def _h3_meta_net(**kwargs):
    """A ``MiniMaxH3`` on the meta device: real module graph, no weights read, no memory."""
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    with torch.device("meta"):
        return MiniMaxH3(load_pretrained=False, **kwargs)


def _h3_meta_vae():
    """``MiniMaxH3VAE`` with both VAEs built on meta from the checkpoint's own configs.

    ``__init__`` is bypassed because it reads weights. Building from the deployed configs rather
    than from each class's argument defaults is the point: a config flag that adds a module is
    exactly the kind of change that could escape the pin lists.
    """
    from fastgen.networks.MiniMaxH3 import network as h3

    root = hf_download(h3.MiniMaxH3.MODEL_ID, allow_patterns=list(h3.MINIMAX_H3_SUBFOLDERS))
    wrapper = object.__new__(h3.MiniMaxH3VAE)
    torch.nn.Module.__init__(wrapper)
    for attr, cls, subfolder in (
        ("_video_vae", h3.AutoencoderKLMiniMaxH3, "vae"),
        ("_audio_vae", h3.AutoencoderKLMiniMaxH3Audio, "audio_vae"),
    ):
        with torch.device("meta"):
            setattr(wrapper, attr, cls.from_config(cls.load_config(os.path.join(root, subfolder))))
    return wrapper


def _h3_meta_text_encoder():
    """``MiniMaxH3TextEncoder`` with Qwen3-VL built on meta from the checkpoint's own config.

    ``__init__`` is bypassed because it reads a 62 GB conditioner. The real module graph is the
    point: the conditioner must own only the backbone it calls, without the unused language-model
    head from the released conditional-generation checkpoint.
    """
    from fastgen.networks.MiniMaxH3 import network as h3

    root = hf_download(h3.MiniMaxH3.MODEL_ID, allow_patterns=list(h3.MINIMAX_H3_SUBFOLDERS))
    config = h3.Qwen3VLModel.config_class.from_pretrained(os.path.join(root, "text_encoder"))
    wrapper = object.__new__(h3.MiniMaxH3TextEncoder)
    torch.nn.Module.__init__(wrapper)
    with torch.device("meta"):
        wrapper.text_encoder = h3.Qwen3VLModel._from_config(config)
    return wrapper


@pytest.mark.parametrize("rank0", [True, False])
def test_minimax_h3_text_encoder_defers_distributed_state_to_preparation(monkeypatch, rank0):
    """The constructor loads rank 0 and leaves peer state meta for outer preparation."""
    from fastgen.networks.MiniMaxH3 import network as h3

    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=h3.MINIMAX_H3_TEXT_ENCODER_LAYER + 1))
    load_calls = []

    class StubLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            del args, kwargs
            return object()

    class StubConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            del args, kwargs
            load_calls.append("config")
            return config

    class StubTextEncoder(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, dtype=dtype))
            self.config = config

    class StubQwen3VLModel:
        config_class = StubConfig

        @staticmethod
        def _from_config(_config, dtype):
            assert _config is config
            load_calls.append("from_config")
            return StubTextEncoder(dtype)

        @staticmethod
        def from_pretrained(*args, **kwargs):
            del args
            load_calls.append("from_pretrained")
            return StubTextEncoder(kwargs["dtype"])

    monkeypatch.setattr(h3, "Qwen2TokenizerFast", StubLoader)
    monkeypatch.setattr(h3, "Qwen3VLProcessor", StubLoader)
    monkeypatch.setattr(h3, "Qwen3VLModel", StubQwen3VLModel)
    monkeypatch.setattr(h3, "world_size", lambda: 2)
    monkeypatch.setattr(h3, "is_rank0", lambda: rank0)

    wrapper = h3.MiniMaxH3TextEncoder("/unused", dtype=torch.bfloat16)

    assert wrapper.text_encoder.weight.is_meta == (not rank0)
    assert wrapper.text_encoder.weight.dtype == torch.bfloat16
    assert not wrapper.is_prepared
    assert load_calls == (["from_pretrained"] if rank0 else ["config", "from_config"])


def _h3_condition(net, batch_size, seq_len, device, dtype):
    """A synthetic conditioning dict.

    MiniMax-H3's conditioner is a 62 GB Qwen3-VL, and what the transformer consumes from it is
    a hidden state plus the per-row modality tags -- the rows of a keyframe's vision block are
    tagged as video rather than text, which is what the AdaLN modulation keys off.
    """
    from fastgen.networks.MiniMaxH3.packing import MINIMAX_H3_TEXT_TAG

    return {
        "prompt_embeds": torch.randn(batch_size, seq_len, net.transformer.config.text_dim, device=device, dtype=dtype),
        "text_token_tags": torch.full((seq_len,), MINIMAX_H3_TEXT_TAG, dtype=torch.long, device=device),
    }


@pytest.mark.parametrize("operation", ["encode", "decode"])
def test_minimax_h3_vae_rejects_input_without_a_supported_modality(operation):
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3VAE

    vae = object.__new__(MiniMaxH3VAE)
    torch.nn.Module.__init__(vae)
    data = torch.zeros(1, 3, 1, 2, 2)
    run = getattr(vae, operation)

    # Both directions read the audio key once video is absent, so a dict with neither
    # modality fails on that lookup rather than on a branch-specific guard.
    with pytest.raises(KeyError, match="audio"):
        run({})
    with pytest.raises(KeyError, match="audio"):
        run({"text": data})


def test_minimax_h3_forward_requires_complete_condition():
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    network = object.__new__(MiniMaxH3)
    torch.nn.Module.__init__(network)
    network.fps = 25.0
    network.net_pred_type = "flow"
    network.transformer = SimpleNamespace(transformer_blocks=[])

    video = torch.zeros(1, 2, 1, 1, 1)
    audio = torch.zeros(1, 2, 2, 1)
    t = torch.zeros(1)
    with pytest.raises(ValueError, match="text_token_tags"):
        network({"video": video, "audio": audio}, t, condition={"prompt_embeds": torch.zeros(1, 1, 1)})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bool, torch.int16, torch.uint8])
def test_minimax_h3_forward_rejects_text_token_tags_outside_the_index_dtypes(dtype):
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    network = object.__new__(MiniMaxH3)
    torch.nn.Module.__init__(network)
    network.fps = 25.0
    network.net_pred_type = "flow"
    network.transformer = SimpleNamespace(transformer_blocks=[])

    with pytest.raises(TypeError, match="int32 or int64"):
        network(
            {"video": torch.zeros(1, 2, 1, 1, 1), "audio": torch.zeros(1, 2, 2, 1)},
            torch.zeros(1),
            condition={
                "prompt_embeds": torch.zeros(1, 1, 1),
                "text_token_tags": torch.zeros(1, dtype=dtype),
            },
        )


def test_minimax_h3_forward_rejects_batch_sliced_text_token_tags():
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    network = object.__new__(MiniMaxH3)
    torch.nn.Module.__init__(network)
    network.fps = 25.0
    network.net_pred_type = "flow"
    network.transformer = SimpleNamespace(transformer_blocks=[])

    # The row count lines up, so this pins the batched shape itself rather than a length mismatch,
    # which ``index_copy_`` already reports where the rows are packed.
    with pytest.raises(ValueError, match="one shared sequence of per-row tags"):
        network(
            {"video": torch.zeros(1, 2, 1, 1, 1), "audio": torch.zeros(1, 2, 2, 1)},
            torch.zeros(1),
            condition={
                "prompt_embeds": torch.zeros(1, 4, 4),
                "text_token_tags": torch.zeros(1, 4, dtype=torch.long),
            },
        )


def test_minimax_h3_rejects_per_sample_timesteps():
    """Every batch item shares one packed layout and one row-to-timestep plan, so the timestep is one scalar."""
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    assert MiniMaxH3._scalar_timestep(torch.full((3,), 0.25)) == 0.75
    with pytest.raises(ValueError, match="uniform across the batch"):
        MiniMaxH3._scalar_timestep(torch.tensor([0.25, 0.5]))


def test_minimax_h3_forward_restores_logvar_to_primary_state_dtype():
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList()

        def forward(self, hidden_states, audio_hidden_states, **kwargs):
            return (
                hidden_states.to(torch.bfloat16),
                audio_hidden_states.to(torch.bfloat16),
                torch.ones(1, 1, dtype=torch.bfloat16),
                None,
            )

    class NoiseScheduler(torch.nn.Module):
        @staticmethod
        def sigma(t):
            return {"video": t, "audio": t}

        @staticmethod
        def convert_model_output(x_t, model_output, *args, **kwargs):
            del args, kwargs  # called with output_dtype=None: uncast, in at least HIGH_PRECISION
            return {
                name: value.to(dtype=high_precision_dtype(x_t[name], value)) for name, value in model_output.items()
            }

    network = object.__new__(MiniMaxH3)
    torch.nn.Module.__init__(network)
    network.fps = 25.0
    network.net_pred_type = "flow"
    network.patch_size = (1, 1, 1)
    network.transformer = Transformer()
    network.noise_scheduler = NoiseScheduler()
    network._to_flow_convention = lambda model_output, t: model_output
    network._layout = lambda *args: SimpleNamespace(
        sequence_length=4,
        num_condition_video_rows=0,
        video_indices=torch.tensor([3]),
        audio_indices=torch.tensor([1, 2]),
        text_indices=torch.tensor([0]),
        token_tags=torch.tensor([1, 2, 2, 0]),
        position_ids=torch.zeros(4, 3),
    )

    video = torch.zeros(1, 2, 1, 1, 1, dtype=torch.float64)
    audio = torch.zeros(1, 2, 2, 1, dtype=torch.float32)
    output, logvar = network(
        {"video": video, "audio": audio},
        torch.full((1,), 0.5),
        condition={
            "prompt_embeds": torch.zeros(1, 1, 1),
            "text_token_tags": torch.ones(1, dtype=torch.long),
        },
        return_logvar=True,
    )

    assert output["video"].dtype == video.dtype
    assert output["audio"].dtype == audio.dtype
    assert logvar.dtype == video.dtype


def test_minimax_h3_layout_cache_builds_normal_tensors_without_gradients(monkeypatch):
    from fastgen.networks.MiniMaxH3 import network as h3_network

    observed = {}

    def fake_layout(*args, **kwargs):
        observed["inference"] = torch.is_inference_mode_enabled()
        observed["grad"] = torch.is_grad_enabled()
        tensor = torch.zeros(1)
        return SimpleNamespace(
            position_ids=tensor,
            token_tags=tensor,
            video_indices=tensor,
            audio_indices=tensor,
            text_indices=tensor,
        )

    monkeypatch.setattr(h3_network, "build_packed_sequence", fake_layout)
    network = object.__new__(h3_network.MiniMaxH3)
    torch.nn.Module.__init__(network)
    network._layout_cache = {}
    network.patch_size = (1, 1, 1)

    for outer_context in (torch.enable_grad(), torch.no_grad(), torch.inference_mode()):
        network._layout_cache.clear()
        observed.clear()
        with outer_context:
            layout = network._layout(
                torch.ones(2, dtype=torch.long),
                num_latent_frames=2,
                latent_height=2,
                latent_width=2,
                num_audio_latents=2,
                keyframe_anchors=(),
                device=torch.device("cpu"),
            )

        assert observed == {"inference": False, "grad": False}
        assert not layout.position_ids.is_inference()


class _H3ConditioningVAE:
    UNSET = object()

    def encode_reference_video(self, video, mode="sample", seed=UNSET):
        self.frames = video
        self.mode = mode
        self.seed = seed
        return torch.ones(video.shape[0], 2, 1, 2, 3)


class _H3ConditioningTextEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, prompt, keyframes=None):
        self.calls.append((prompt, keyframes))
        return {
            "prompt_embeds": torch.full((1, 3, 4), float(len(self.calls))),
            "text_token_tags": torch.ones(3, dtype=torch.long),
        }


def _make_h3_conditioning_net(is_fl2va):
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    net = MiniMaxH3.__new__(MiniMaxH3)
    torch.nn.Module.__init__(net)
    net.preprocessors = {"text_encoder": _H3ConditioningTextEncoder()}
    net.is_fl2va = is_fl2va
    if is_fl2va:
        net.preprocessors["vae"] = _H3ConditioningVAE()
        net.patch_size = (1, 1, 1)
        net.video_latent_channels = 2
    return net


@pytest.mark.parametrize(
    "num_frames, requested_anchors, expected_anchors, endpoint_indices",
    [
        (1, None, ("first",), (0,)),
        (4, None, ("first", "last"), (0, -1)),
        (4, ("first",), ("first",), (0,)),
        (4, ("last",), ("last",), (-1,)),
        (4, ("first", "last"), ("first", "last"), (0, -1)),
    ],
    ids=["single_default", "both_default", "first", "last", "first_and_last"],
)
def test_minimax_h3_build_conditioning_uses_endpoint_keyframes(
    num_frames, requested_anchors, expected_anchors, endpoint_indices
):
    """Pixel conditioning keeps both endpoint anchors without entering a second packing path."""
    net = _make_h3_conditioning_net(is_fl2va=True)

    video = torch.linspace(-1, 1, 3 * num_frames * 2 * 3).reshape(1, 3, num_frames, 2, 3)
    state_video = torch.zeros(1, 2, 1, 2, 3, dtype=torch.bfloat16)
    negative_raw = ["negative prompt"]
    data = {
        "real": {"video": state_video},
        "real_raw": {"video": video},
        "condition_raw": ["a prompt"],
        "neg_condition_raw": negative_raw,
    }
    if requested_anchors is not None:
        data["keyframe_anchors"] = requested_anchors
    output = net.build_conditioning(data)
    condition = output["condition"]

    assert condition["keyframe_anchors"] == expected_anchors
    assert "ref_geometry" not in condition
    negative = output["neg_condition"]
    assert negative["keyframe_anchors"] == expected_anchors
    assert negative["cond_video_rows"] is condition["cond_video_rows"]
    assert torch.equal(condition["prompt_embeds"], torch.ones_like(condition["prompt_embeds"]))
    assert torch.equal(negative["prompt_embeds"], torch.full_like(negative["prompt_embeds"], 2))
    assert "neg_condition_raw" not in output
    merged = {**data, **output}
    assert merged["neg_condition"] is negative
    assert merged["neg_condition_raw"] is negative_raw
    assert net.preprocessors["vae"].mode == "sample"
    assert net.preprocessors["vae"].seed is _H3ConditioningVAE.UNSET
    uint8_frames = net.preprocessors["vae"].frames
    assert uint8_frames.shape == (len(expected_anchors), 3, 1, 2, 3)
    assert uint8_frames.dtype == torch.uint8
    for actual, index in zip(uint8_frames, endpoint_indices, strict=True):
        endpoint = video[0, :, index : index + 1] if index >= 0 else video[0, :, -1:]
        assert torch.equal(actual, endpoint.mul(0.5).add(0.5).mul(255).round().to(torch.uint8))
    assert [call[0] for call in net.preprocessors["text_encoder"].calls] == [["a prompt"], negative_raw]
    for _, keyframes in net.preprocessors["text_encoder"].calls:
        # The text encoder is shown the very uint8 frames the VAE encodes.
        assert isinstance(keyframes, list)
        torch.testing.assert_close(torch.stack(keyframes), uint8_frames[:, :, 0])
    assert condition["cond_video_rows"].shape == (1, len(expected_anchors) * 2 * 3, 2)
    assert condition["cond_video_rows"].dtype == torch.float32
    assert condition["cond_video_rows"].device == state_video.device


def test_minimax_h3_t2va_does_not_treat_target_pixels_as_keyframes():
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    net = MiniMaxH3.__new__(MiniMaxH3)
    net.is_fl2va = False
    condition = {"prompt_embeds": torch.ones(1, 3, 4), "text_token_tags": torch.ones(3)}
    data = {"condition": condition, "real_raw": {"video": torch.zeros(1, 3, 4, 2, 3)}}

    assert net.build_conditioning(data) == {"condition": condition}
    assert data["condition"] is condition


@pytest.mark.parametrize(
    ("is_fl2va", "required_field"),
    [(False, "condition"), (True, "condition_raw")],
    ids=["t2va", "fl2va"],
)
def test_minimax_h3_build_conditioning_requires_positive_condition(is_fl2va, required_field):
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    net = MiniMaxH3.__new__(MiniMaxH3)
    net.is_fl2va = is_fl2va
    data = {"real_raw": {"video": torch.zeros(1, 3, 1, 1, 1)}} if is_fl2va else {}

    with pytest.raises(ValueError, match=rf"requires non-None data\['{required_field}'\]"):
        net.build_conditioning(data)


def test_minimax_h3_fl2va_requires_reference_pixels():
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    net = MiniMaxH3.__new__(MiniMaxH3)
    net.is_fl2va = True
    with pytest.raises(ValueError, match=r"requires non-None data\['real_raw'\]"):
        net.build_conditioning({"condition_raw": ["a prompt"]})


@pytest.mark.parametrize(
    ("key", "raw_key"),
    [("condition", "condition_raw"), ("neg_condition", "neg_condition_raw")],
)
@pytest.mark.parametrize("value_kind", ["none", "encoded"])
def test_minimax_h3_fl2va_requires_plain_condition_fields_to_be_absent(key, raw_key, value_kind):
    net = _make_h3_conditioning_net(is_fl2va=True)
    encoded = {
        "prompt_embeds": torch.ones(1, 3, 4),
        "text_token_tags": torch.ones(3, dtype=torch.long),
    }
    data = {
        "real_raw": {"video": torch.zeros(1, 3, 1, 2, 3)},
        "condition_raw": ["positive prompt"],
        key: None if value_kind == "none" else encoded,
    }
    if key == "condition":
        data.pop("condition_raw")
    else:
        data[raw_key] = ["ignored raw mirror"]

    with pytest.raises(ValueError, match=rf"requires absent data\['{key}'\]"):
        net.build_conditioning(data)

    assert net.preprocessors["text_encoder"].calls == []


@pytest.mark.parametrize("is_fl2va", [False, True], ids=["t2va", "fl2va"])
def test_minimax_h3_rejects_reentrant_fl2va_condition(is_fl2va):
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    net = MiniMaxH3.__new__(MiniMaxH3)
    net.is_fl2va = is_fl2va
    condition = {
        "prompt_embeds": torch.ones(1, 3, 4),
        "text_token_tags": torch.ones(3, dtype=torch.long),
        "cond_video_rows": torch.ones(1, 1, 2),
    }
    data = {"condition": condition}
    if is_fl2va:
        data["real_raw"] = {"video": torch.zeros(1, 3, 1, 1, 1)}

    with pytest.raises(ValueError, match="cannot be prepared again|requires absent"):
        net.build_conditioning(data)


def test_minimax_h3_fl2va_encodes_raw_primary_prompts():
    """FL2VA reads raw prompts only from the explicit raw namespace."""
    net = _make_h3_conditioning_net(is_fl2va=True)

    data = {
        "real_raw": {"video": torch.zeros(1, 3, 2, 2, 3)},
        "condition_raw": ["a prompt"],
        "neg_condition_raw": ["negative prompt"],
    }
    out = net.build_conditioning(data)

    assert [call[0] for call in net.preprocessors["text_encoder"].calls] == [["a prompt"], ["negative prompt"]]
    assert set(out) == {"condition", "neg_condition"}
    assert torch.all(out["condition"]["prompt_embeds"] == 1)
    assert torch.all(out["neg_condition"]["prompt_embeds"] == 2)


@pytest.mark.parametrize(
    "negative_field",
    [None, "neg_condition_raw"],
    ids=["missing", "raw_none"],
)
def test_minimax_h3_fl2va_preserves_absent_negative(negative_field):
    net = _make_h3_conditioning_net(is_fl2va=True)
    data = {
        "real_raw": {"video": torch.zeros(1, 3, 1, 2, 3)},
        "condition_raw": ["a prompt"],
    }
    if negative_field is not None:
        data[negative_field] = None
    out = net.build_conditioning(data)

    assert set(out) == {"condition"}


def test_minimax_h3_fl2va_casts_text_to_compute_dtype():
    """The text follows the compute dtype; the keyframe rows stay float32 under a bfloat16 state."""
    net = _make_h3_conditioning_net(is_fl2va=True)
    net._precision_model = torch.bfloat16

    out = net.build_conditioning(
        {
            "real_raw": {"video": torch.zeros(1, 3, 1, 2, 3)},
            "condition_raw": ["a prompt"],
        },
        precision=torch.bfloat16,
    )

    assert out["condition"]["prompt_embeds"].dtype == torch.bfloat16
    assert out["condition"]["text_token_tags"].dtype == torch.long
    assert out["condition"]["cond_video_rows"].dtype == torch.float32


def test_minimax_h3_t2va_encodes_explicit_raw_branch_once():
    net = _make_h3_conditioning_net(is_fl2va=False)

    out = net.build_conditioning({"condition_raw": ["prompt"]})

    assert set(out) == {"condition"}
    assert [call[0] for call in net.preprocessors["text_encoder"].calls] == [["prompt"]]


@pytest.mark.parametrize("is_fl2va", [False, True], ids=["t2va", "fl2va"])
def test_minimax_h3_encodes_explicit_empty_negative_prompt(is_fl2va):
    """An empty negative prompt is a request, not a placeholder to be dropped: callers that
    want no negative branch omit ``neg_condition_raw`` entirely."""
    net = _make_h3_conditioning_net(is_fl2va=is_fl2va)
    data = {"condition_raw": ["prompt"], "neg_condition_raw": [""]}
    if is_fl2va:
        data["real_raw"] = {"video": torch.zeros(1, 3, 2, 2, 3)}

    out = net.build_conditioning(data)

    assert set(out) == {"condition", "neg_condition"}
    assert [call[0] for call in net.preprocessors["text_encoder"].calls] == [["prompt"], [""]]
    assert data["neg_condition_raw"] == [""]


def test_minimax_h3_rejects_non_reference_keyframe_anchor_orders():
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    video = torch.zeros(1, 3, 4, 2, 3)
    with pytest.raises(ValueError, match="keyframe_anchors"):
        MiniMaxH3._endpoint_keyframes(video, ("last", "first"))


def test_minimax_h3_reference_vae_encode_matches_diffusers_recipe():
    """Each ``uint8`` keyframe gets Diffusers' exact ``/ 255``, fresh seed-42 draw and float16 round-trip."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from fastgen.networks.MiniMaxH3.network import (
        MINIMAX_H3_PIXEL_MEAN,
        MINIMAX_H3_PIXEL_STD,
        MiniMaxH3VAE,
    )

    class Config:
        latents_mean = (0.125,)
        latents_std = (0.75,)

    class FakeVAE:
        config = Config()
        device = torch.device("cpu")

        def encode(self, pixels, return_dict=False):
            mean = pixels[:, :1].mul(0.137).add(0.019)
            logvar = torch.full_like(mean, -0.7)
            return (DiagonalGaussianDistribution(torch.cat([mean, logvar], dim=1)),)

    vae = FakeVAE()
    wrapper = MiniMaxH3VAE.__new__(MiniMaxH3VAE)
    torch.nn.Module.__init__(wrapper)
    wrapper._video_vae = vae

    # Identical conditions are intentional: a fresh seed per keyframe must give identical
    # samples. A single generator over the batch would give the second one a different draw.
    pixels = torch.tensor(
        [
            [
                [[[0, 17, 63], [127, 191, 255]]],
                [[[3, 31, 95], [129, 193, 251]]],
                [[[7, 47, 111], [131, 223, 247]]],
            ]
        ],
        dtype=torch.uint8,
    ).repeat(2, 1, 1, 1, 1)

    actual = wrapper.encode_reference_video(pixels, mode="sample", seed=42)
    repeated = wrapper.encode_reference_video(pixels, mode="sample", seed=42)
    expected = torch.cat(
        [
            encode_vae_condition(vae, keyframe, MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD)
            for keyframe in pixels.split(1)
        ]
    )

    assert torch.equal(actual, expected)
    assert torch.equal(actual, repeated), "a supplied parity seed must reproduce the same keyframe encode"
    assert torch.equal(actual[0], actual[1]), "each keyframe must start from a fresh seed-42 generator"

    # Dropping the float16 round-trip is numerically observable, so it is required for parity even
    # though the encoder and normalization otherwise run in float32.
    distr = vae.encode(wrapper._pixels_to_vae_space(pixels[:1]), return_dict=False)[0]
    unrounded = distr.sample(generator=torch.Generator().manual_seed(42)).float()
    unrounded = wrapper._normalize_video_latents(unrounded)
    assert not torch.equal(actual[:1], unrounded)


def test_minimax_h3_vae_reads_uint8_pixels_exactly():
    """``uint8`` keyframes reach the VAE as the official processor's exact ``/ 255`` under ``Normalize``."""
    from torchvision.transforms import Normalize

    from fastgen.networks.MiniMaxH3.network import MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD, MiniMaxH3VAE

    wrapper = MiniMaxH3VAE.__new__(MiniMaxH3VAE)
    torch.nn.Module.__init__(wrapper)
    pixels = torch.arange(256, dtype=torch.uint8).expand(3, 256).reshape(1, 3, 1, 1, 256)
    expected = Normalize(MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD)(pixels[:, :, 0].float() / 255.0)

    assert torch.equal(wrapper._pixels_to_vae_space(pixels)[:, :, 0], expected)


def test_minimax_h3_reference_vae_encode_modes():
    """Sample mode uses the active RNG while argmax is deterministic and invalid modes fail."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3VAE

    class Config:
        latents_mean = (0.0,)
        latents_std = (1.0,)

    class FakeVAE:
        config = Config()
        device = torch.device("cpu")

        def encode(self, pixels, return_dict=False):
            mean = torch.zeros_like(pixels[:, :1])
            logvar = torch.zeros_like(mean)
            return (DiagonalGaussianDistribution(torch.cat([mean, logvar], dim=1)),)

    wrapper = MiniMaxH3VAE.__new__(MiniMaxH3VAE)
    torch.nn.Module.__init__(wrapper)
    wrapper._video_vae = FakeVAE()
    video = torch.zeros(1, 3, 1, 4, 4)

    torch.manual_seed(7)
    first = wrapper.encode_reference_video(video)
    second = wrapper.encode_reference_video(video)
    torch.manual_seed(7)
    repeated = wrapper.encode_reference_video(video)

    assert not torch.equal(first, second), "successive training samples should advance the active RNG"
    assert torch.equal(first, repeated), "the active RNG still controls reproducibility"

    torch.manual_seed(1)
    argmax = wrapper.encode_reference_video(video, mode="argmax")
    torch.manual_seed(999)
    assert torch.equal(argmax, wrapper.encode_reference_video(video, mode="argmax"))
    with pytest.raises(ValueError, match="Invalid mode"):
        wrapper.encode_reference_video(video, mode="median")


def _h3_tiny_vae(audio_latents_of):
    """``MiniMaxH3VAE`` over stub autoencoders; ``audio_latents_of(samples)`` sets the audio latent count."""
    from fastgen.networks.MiniMaxH3.network import MINIMAX_H3_FPS, MiniMaxH3VAE

    class Stats:
        latents_mean = (0.0, 0.0)
        latents_std = (1.0, 1.0)

    class Moments:
        def __init__(self, value):
            self.value = value

        def mode(self):
            return self.value

    class FakeVideoVAE:
        config = Stats()

        def encode(self, pixels, return_dict=False):
            return (Moments(torch.zeros(pixels.shape[0], 2, 1, 1, 1)),)

    class FakeAudioVAE:
        config = Stats()
        hop_length = 800

        def __init__(self):
            self.seen_samples = []

        def encode(self, mono, return_dict=False):
            self.seen_samples.append(mono.shape[-1])
            return (Moments(torch.zeros(mono.shape[0], 2, audio_latents_of(mono.shape[-1]))),)

    vae = MiniMaxH3VAE.__new__(MiniMaxH3VAE)
    torch.nn.Module.__init__(vae)
    vae._fps = float(MINIMAX_H3_FPS)
    vae._video_vae = FakeVideoVAE()
    vae._audio_vae = FakeAudioVAE()
    return vae


@pytest.mark.parametrize("num_frames,loader_samples,target", [(107, 142666, 178), (124, 165333, 207)])
def test_minimax_h3_vae_encode_aligns_the_waveform_to_the_audio_contract(num_frames, loader_samples, target):
    """``round(frames / 24 * 40)`` latents of 800 samples: the loader's ``int(frames * 32000 / 24)``
    waveform is trimmed (107 frames) or zero-padded (124 frames) to that span before encoding."""
    vae = _h3_tiny_vae(lambda samples: -(-samples // 800))  # ceil, like the right-padding encoder

    video = torch.zeros(1, 3, num_frames, 4, 4)
    out = vae.encode({"video": video, "audio": torch.zeros(1, 2, loader_samples)}, mode="argmax")
    assert vae._audio_vae.seen_samples == [target * 800]
    assert tuple(out["audio"].shape) == (1, 2, 2, target)

    silent = vae.encode({"video": video}, mode="argmax")
    assert vae._audio_vae.seen_samples == [target * 800, target * 800]
    assert tuple(silent["audio"].shape) == (1, 2, 2, target)


def test_minimax_h3_vae_encode_rejects_an_unexpected_audio_latent_count():
    vae = _h3_tiny_vae(lambda samples: samples // 800 - 1)
    with pytest.raises(ValueError, match="got 177 frames for 142400 waveform samples, expected 178"):
        vae.encode({"video": torch.zeros(1, 3, 107, 4, 4), "audio": torch.zeros(1, 2, 142666)}, mode="argmax")


def test_minimax_h3_reference_encoding_defaults_to_sample_mode():
    from fastgen.configs.net import MiniMaxH3_Config
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3, MiniMaxH3VAE

    assert "keyframe_encode_seed" not in MiniMaxH3_Config
    network_parameters = inspect.signature(MiniMaxH3.__init__).parameters
    assert "keyframe_encode_seed" not in network_parameters
    assert network_parameters["is_fl2va"].default is False
    assert MiniMaxH3_Config.is_fl2va is False
    parameters = inspect.signature(MiniMaxH3VAE.encode_reference_video).parameters
    assert parameters["mode"].default == "sample"
    assert parameters["seed"].default is None


def _sglang_h3_packed_sequence(*, text_len, latent_t, latent_h, latent_w, audio_t, keyframe_frame_indices, frame_count):
    """SGLang's ``minimax_h3_packed_sequence`` (``minimax_h3/packed_sequence.py``) for t2va/fl2va, verbatim but for its
    argument checks and the attention metadata FastGen does not build."""
    _INTERP, _T_GROUP, _FRAME_PER_TOKEN, _FRAME_RESCALE, _PATCH_H, _PATCH_W = 32, 5, (1, 4, 4, 4, 4), 5.0 / 3.0, 2, 2
    audio_channel = 2

    def _temporal_position_span(temporal_length):
        spans = np.ones(int(temporal_length), dtype=np.float64) * _FRAME_RESCALE
        for token_index in range(_T_GROUP):
            spans[token_index::_T_GROUP] *= _FRAME_PER_TOKEN[token_index]
        return float(spans.sum())

    def _axis_from_sqrt_area(dim, patch, sqrt_area):
        ratio = dim / sqrt_area
        left = (1.0 - ratio) * 1.0 / 2.0
        right = left + ratio * 1.0
        grid = np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
        return torch.from_numpy(grid).to(torch.float64)

    def _video_t_grid(n, origin):
        spans = torch.tensor([_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n)], dtype=torch.float64)
        return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

    ph, pw = latent_h // _PATCH_H, latent_w // _PATCH_W
    frame_rows = ph * pw
    resolved_cond_frame_indices = [frame_count - 1 if index == -1 else index for index in keyframe_frame_indices]
    cond_rows = len(resolved_cond_frame_indices) * frame_rows
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel
    used = text_len + cond_rows + audio_rows + video_rows
    seq_len = (used + 64 - 1) // 64 * 64

    text_sl = slice(0, text_len)
    cond_sl = slice(text_len, text_len + cond_rows)
    audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)
    target_img_pos = torch.arange(video_sl.start, video_sl.stop)
    img_pos = torch.cat([torch.arange(cond_sl.start, cond_sl.stop), target_img_pos]) if cond_rows else target_img_pos
    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[cond_rows:] = True
    audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
    text_pos = torch.arange(0, text_len)

    g = torch.zeros(seq_len, 3, dtype=torch.float64)
    g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

    t_grid = _video_t_grid(latent_t, float(text_len))
    sqrt_area = np.sqrt(latent_h * latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, _PATCH_H, sqrt_area)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
    video_g = g[video_sl].view(latent_t, frame_rows, 3)
    video_g[:, :, 0] = t_grid[:, None]
    video_g[:, :, 1:] = frame[None]
    for block_index, pixel_index in enumerate(resolved_cond_frame_indices):
        sl = slice(cond_sl.start + block_index * frame_rows, cond_sl.start + (block_index + 1) * frame_rows)
        if pixel_index == 0:
            cond_t = float(text_len)
        elif frame_count is not None and pixel_index == frame_count - 1:
            cond_t = float(text_len) + _temporal_position_span(latent_t) - _FRAME_RESCALE
        g[sl, 0] = cond_t
        g[sl, 1:] = frame
    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    g[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    g[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    g[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_sl] = 1
    token_tags[audio_sl] = 2
    token_tags[img_pos] = 0
    return {
        "seq_len": seq_len,
        "used": used,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "img_position_ids": g,
        "token_tags": token_tags,
    }


def _sglang_h3_row_timesteps(packed, t_video, t_audio):
    """SGLang's ``MiniMaxH3DenoiseBranch`` timestep plan (``minimax_h3/denoise_loop.py``) for one t2va/fl2va step.

    Text, padding and generated video rows take the video timestep, keyframe rows ``max(t_video, 0.999)``.
    """
    candidates = [float(t_video)]
    fill_groups = []
    for seq_idx, value in (
        (packed["img_pos"][~packed["update_mask"]], max(t_video, 0.999)),
        (packed["audio_pos"], t_audio),
    ):
        if seq_idx.numel() > 0:
            fill_groups.append((seq_idx, len(candidates)))
            candidates.append(float(value))
    unique, slot_to_unique = torch.unique(
        torch.tensor(candidates, dtype=torch.float32), sorted=True, return_inverse=True
    )
    inverse = torch.full((packed["seq_len"],), int(slot_to_unique[0]), dtype=torch.long)
    for seq_idx, slot in fill_groups:
        inverse.index_fill_(0, seq_idx, int(slot_to_unique[slot]))
    return unique, inverse


@pytest.mark.parametrize("anchors", [(), ("first",), ("last",), ("first", "last")])
def test_minimax_h3_packing_matches_sglang_reference(anchors):
    """Layout, rotary positions, tags and row timesteps match SGLang's for every keyframe mode.

    SGLang pads the sequence to a multiple of 64 as its own attention document; FastGen builds no padding, so the
    comparison covers the used rows.
    """
    from fastgen.networks.MiniMaxH3.packing import (
        MINIMAX_H3_KEYFRAME_TIMESTEP,
        build_packed_sequence,
        build_row_timesteps,
    )

    text_tags = torch.tensor([1, 0, 0, 1, 1], dtype=torch.long)
    actual = build_packed_sequence(text_tags, 7, 4, 6, 11, (1, 2, 2), anchors)
    expected = _sglang_h3_packed_sequence(
        text_len=5,
        latent_t=7,
        latent_h=4,
        latent_w=6,
        audio_t=11,
        keyframe_frame_indices=[{"first": 0, "last": -1}[anchor] for anchor in anchors],
        frame_count=22,  # 17 * n + 5 pixel frames for 5 * n + 2 = 7 latent frames
    )
    # SGLang writes the presentation's tags over its text rows before building the branch.
    expected["token_tags"][expected["text_pos"]] = text_tags

    used = expected["used"]
    assert actual.sequence_length == used
    assert torch.equal(actual.position_ids, expected["img_position_ids"][:used])
    assert torch.equal(actual.token_tags, expected["token_tags"][:used])
    assert torch.equal(actual.video_indices, expected["img_pos"])
    assert torch.equal(actual.audio_indices, expected["audio_pos"])
    assert torch.equal(actual.text_indices, expected["text_pos"])
    assert actual.num_condition_video_rows == int((~expected["update_mask"]).sum())

    for t_video, t_audio in ((0.4, 0.2), (0.9995, 0.5)):
        unique, inverse = build_row_timesteps(actual, t_video, t_audio, max(t_video, MINIMAX_H3_KEYFRAME_TIMESTEP))
        expected_unique, expected_inverse = _sglang_h3_row_timesteps(expected, t_video, t_audio)
        assert torch.equal(unique, expected_unique)
        assert torch.equal(inverse, expected_inverse[:used])


@RunIf(min_gpus=1)
def test_minimax_h3_row_timesteps_follow_layout_index_device():
    from fastgen.networks.MiniMaxH3.packing import build_packed_sequence, build_row_timesteps

    layout = build_packed_sequence(
        torch.tensor([1, 0, 1], dtype=torch.long),
        7,
        4,
        6,
        11,
        (1, 2, 2),
        (),
    )
    layout.video_indices = layout.video_indices.cuda()
    layout.audio_indices = layout.audio_indices.cuda()

    timesteps, timestep_indices = build_row_timesteps(layout, 0.4, 0.2, 0.999)

    assert timesteps.device.type == "cuda"
    assert timestep_indices.device.type == "cuda"


@pytest.mark.parametrize("anchors", [(), ("first",), ("last",), ("first", "last")])
@pytest.mark.parametrize("geometry", [(5, 7, 4, 6, 11), (1, 2, 2, 2, 1), (37, 12, 6, 10, 23)])
def test_minimax_h3_packed_rows_tile_the_sequence(anchors, geometry):
    """Every packed position holds exactly one text, video or audio row, and every tag addresses the AdaLN table.

    That is why ``classify_forward`` attends without a mask: no position of its zero-filled buffer stays
    unwritten, so there is no padding to hide.
    """
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

    from fastgen.networks.MiniMaxH3.packing import (
        MINIMAX_H3_AUDIO_CHANNELS,
        MINIMAX_H3_AUDIO_TAG,
        MINIMAX_H3_TEXT_TAG,
        MINIMAX_H3_VIDEO_TAG,
        build_packed_sequence,
    )

    num_text_tokens, num_latent_frames, latent_height, latent_width, num_audio_latents = geometry
    text_tags = torch.full((num_text_tokens,), MINIMAX_H3_TEXT_TAG)
    text_tags[1::3] = MINIMAX_H3_VIDEO_TAG
    layout = build_packed_sequence(
        text_tags, num_latent_frames, latent_height, latent_width, num_audio_latents, (1, 2, 2), anchors
    )

    rows_per_frame = (latent_height // 2) * (latent_width // 2)
    assert layout.text_indices.numel() == num_text_tokens
    assert layout.video_indices.numel() == (len(anchors) + num_latent_frames) * rows_per_frame
    assert layout.audio_indices.numel() == MINIMAX_H3_AUDIO_CHANNELS * num_audio_latents
    rows = torch.cat([layout.text_indices, layout.video_indices, layout.audio_indices])
    assert torch.equal(rows.sort().values, torch.arange(layout.sequence_length))
    assert layout.position_ids.shape == (layout.sequence_length, 3)

    assert torch.equal(layout.token_tags[layout.text_indices], text_tags)
    assert bool((layout.token_tags[layout.video_indices] == MINIMAX_H3_VIDEO_TAG).all())
    assert bool((layout.token_tags[layout.audio_indices] == MINIMAX_H3_AUDIO_TAG).all())
    assert 0 <= int(layout.token_tags.min()) <= int(layout.token_tags.max()) < MINIMAX_H3_MODALITY_NUM


@pytest.mark.parametrize("tag", [-1, 2, 3])
def test_minimax_h3_layout_rejects_text_tags_outside_the_presentation(tag):
    """A text row is text or a keyframe's vision block; any other tag would select the wrong AdaLN row.

    ``-1`` is the reference's padding tag, which has no meaning in a sequence without padding rows.
    """
    from fastgen.networks.MiniMaxH3.packing import MINIMAX_H3_TEXT_TAG, build_packed_sequence

    text_tags = torch.tensor([MINIMAX_H3_TEXT_TAG, tag, MINIMAX_H3_TEXT_TAG])
    with pytest.raises(ValueError, match=rf"text_token_tags.*\[{tag}\]"):
        build_packed_sequence(text_tags, 2, 2, 2, 1, (1, 2, 2))


@pytest.mark.parametrize("anchors", [(), ("first", "last")])
def test_minimax_h3_unmasked_attention_matches_the_reference_padding(anchors):
    """Dropping the reference's padding leaves every real row's prediction unchanged.

    The reference pads the packed sequence to a multiple of 64, tags the pad rows ``-1`` (clamped to 0 for the
    AdaLN lookup) and attends with ``cu_seqlens = [0, used, padded]``, which makes the pad rows a second attention
    document. Rebuilding that pass around the same weights reproduces ``classify_forward``'s unmasked one on the
    real rows, whatever the pad rows hold; attending across the document boundary does not.
    """
    from diffusers.models.transformers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

    from fastgen.networks.MiniMaxH3.network import classify_forward
    from fastgen.networks.MiniMaxH3.packing import build_packed_sequence, build_row_timesteps

    torch.manual_seed(0)
    transformer = (
        MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            hidden_size=16,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=3,
            patch_size=(1, 2, 2),
            text_dim=8,
            freq_dim=8,
            time_embed_hidden_dim=16,
            time_embed_dim=8,
            rope_freq_dim=1,
        )
        .double()
        .eval()
    )
    layout = build_packed_sequence(torch.tensor([1, 0, 0, 1, 1]), 2, 4, 6, 3, (1, 2, 2), anchors)
    timestep, timestep_indices = build_row_timesteps(layout, 0.4, 0.6, 0.999)
    video_rows = torch.randn(1, layout.video_indices.numel(), 8, dtype=torch.float64)
    audio_rows = torch.randn(1, layout.audio_indices.numel(), 3, dtype=torch.float64)
    text = torch.randn(1, layout.text_indices.numel(), 8, dtype=torch.float64)

    used = layout.sequence_length
    padded = -(-used // 64) * 64
    is_pad = torch.arange(padded) >= used
    position_ids = torch.cat([layout.position_ids, layout.position_ids.new_zeros(padded - used, 3)])
    row_timesteps = torch.cat([timestep_indices, timestep_indices.new_zeros(padded - used)])
    tags = torch.cat([layout.token_tags, layout.token_tags.new_full((padded - used,), -1)])
    adaln_indices = row_timesteps * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)

    def padded_reference(attention_mask):
        hidden = torch.randn(1, padded, transformer.config.hidden_size, dtype=torch.float64)
        text_embeds = transformer.token_refiner(transformer.context_embedder(text))
        hidden = hidden.index_copy(1, layout.text_indices, text_embeds)
        hidden = hidden.index_copy(1, layout.video_indices, transformer.proj_in(video_rows))
        hidden = hidden.index_copy(1, layout.audio_indices, transformer.audio_proj_in(audio_rows))
        temb = transformer.time_embedder(transformer.time_proj(timestep).to(torch.float64))
        rotary_emb = transformer.rope(position_ids)
        for block in transformer.transformer_blocks:
            hidden = block(hidden, temb, adaln_indices, rotary_emb, attention_mask=attention_mask)
        hidden = transformer.norm_out(hidden, temb, row_timesteps)
        return (
            transformer.proj_out(hidden).index_select(1, layout.video_indices),
            transformer.audio_proj_out(hidden).index_select(1, layout.audio_indices),
        )

    with torch.no_grad():
        video, audio, _, _ = classify_forward(
            transformer,
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=text,
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
        )
        reference_video, reference_audio = padded_reference((is_pad[:, None] == is_pad[None, :])[None, None])
        leaky_video, _ = padded_reference(None)

    torch.testing.assert_close(reference_video, video)
    torch.testing.assert_close(reference_audio, audio)
    assert not torch.allclose(leaky_video, video)


def test_minimax_h3_condition_noise_matches_diffusers_reference():
    """The reversed H3 keyframe timestep uses Diffusers' draw order and arithmetic."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents as ref_patchify
    from diffusers.schedulers import MiniMaxH3Scheduler
    from diffusers.utils.torch_utils import randn_tensor
    from fastgen.networks.MiniMaxH3.packing import keyframe_condition_noise, patchify_video_latents

    conditions = [torch.randn(1, 2, 1, 4, 6), torch.randn(1, 2, 1, 4, 6)]
    patch_size = (1, 2, 2)
    reference_generator = torch.Generator().manual_seed(7)
    scheduler = MiniMaxH3Scheduler()
    expected = []
    for condition in conditions:
        noise = randn_tensor(condition.shape, generator=reference_generator, device="cpu", dtype=torch.float32)
        expected.append(ref_patchify(scheduler.scale_noise(condition, 0.999, noise), patch_size))
    expected = torch.cat(expected)

    noise_rows = keyframe_condition_noise(
        tuple(tuple(condition.shape[2:]) for condition in conditions),
        patch_size,
        latent_channels=2,
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
    )
    clean_rows = torch.cat([patchify_video_latents(condition, patch_size) for condition in conditions])
    h3_timestep = clean_rows.new_tensor(0.999)
    actual = h3_timestep * clean_rows + (1.0 - h3_timestep) * noise_rows

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "cls_name, expected",
    [
        ("MiniMaxH3Transformer3DModel", MINIMAX_H3_TRANSFORMER_FP32),
        ("AutoencoderKLMiniMaxH3", MINIMAX_H3_VIDEO_VAE_FP32),
        ("AutoencoderKLMiniMaxH3Audio", MINIMAX_H3_AUDIO_VAE_FP32),
    ],
)
def test_minimax_h3_keep_in_fp32_modules(cls_name, expected):
    """The float32 carve-out of each component, asserted name by name.

    The transformer policy follows its declaration. The VAE wrapper instead pins the complete
    component to float32 at its precision boundary; its declarations remain upstream parity checks.
    """
    import diffusers.models.autoencoders as autoencoders
    import diffusers.models.transformers as transformers

    cls = getattr(transformers, cls_name, None) or getattr(autoencoders, cls_name)
    assert list(cls._keep_in_fp32_modules) == expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_minimax_h3_vae_precision_boundary_keeps_all_state_float32(dtype):
    """The VAE component resolves any requested model precision to float32."""
    wrapper = _h3_meta_vae()
    vaes = {"video": wrapper._video_vae, "audio": wrapper._audio_vae}
    before = {name: _h3_dtype_map(sub) for name, sub in vaes.items()}
    assert all(dtypes and set(dtypes.values()) == {torch.float32} for dtypes in before.values())

    wrapper._set_precision(precision_model=dtype)

    assert {name: _h3_dtype_map(sub) for name, sub in vaes.items()} == before
    assert wrapper.precision_model == torch.float32


def test_minimax_h3_vae_buffers_follow_the_runtime_device_without_staging_parameters():
    """Generic FSDP placement covers persistent and non-persistent MiniMax-H3 VAE buffers."""
    from fastgen.utils.distributed.fsdp import _move_registered_buffers_

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = torch.nn.Module()
            self.rope.register_buffer("inv_freq", torch.ones(4, dtype=torch.float64), persistent=False)
            self.upsample = torch.nn.Module()
            self.upsample.register_buffer("filter", torch.ones(1, 1, 3))
            self.proj = torch.nn.Linear(1, 1)

    vae = TinyVAE()
    _move_registered_buffers_(vae, torch.device("meta"))

    assert vae.rope.inv_freq.device == torch.device("meta")
    assert vae.rope.inv_freq.dtype == torch.float64
    assert "inv_freq" in vae.rope._non_persistent_buffers_set
    assert vae.upsample.filter.device == torch.device("meta")
    assert "filter" not in vae.upsample._non_persistent_buffers_set
    assert vae.proj.weight.device == torch.device("cpu")


def test_minimax_h3_vae_shards_as_one_group_behind_its_invoked_forward(monkeypatch):
    """The owned layout must not isolate modules whose forward hooks Diffusers bypasses."""
    from torch.distributed.fsdp import FSDPModule

    from fastgen.networks.MiniMaxH3 import network as minimax_h3_network
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3VAE
    import fastgen.utils.distributed.fsdp as fsdp_utils

    vae = MiniMaxH3VAE.__new__(MiniMaxH3VAE)
    torch.nn.Module.__init__(vae)
    vae._audio_vae = torch.nn.Module()
    vae._audio_vae.pre_block = torch.nn.Module()
    vae._audio_vae.pre_block.attn = torch.nn.Module()
    vae._audio_vae.pre_block.attn.qkv = torch.nn.Linear(4, 12, bias=False)
    vae._video_vae = torch.nn.Module()
    vae._video_vae.direct_child = torch.nn.Linear(4, 12, bias=False)

    shard_calls = []

    def record_shard(module, **kwargs):
        shard_calls.append((module, kwargs))
        module_type = type(module)
        module.__class__ = type(f"FSDP{module_type.__name__}", (FSDPModule, module_type), {})

    # Patching both namespaces proves the generic size traversal never runs for this wrapper.
    monkeypatch.setattr(minimax_h3_network, "fully_shard", record_shard)
    monkeypatch.setattr(fsdp_utils, "fully_shard", record_shard)
    fsdp_utils._fully_shard_module(vae, min_num_params=1, mesh=object())

    assert [module for module, _ in shard_calls] == [vae]
    assert isinstance(vae, FSDPModule)
    # The root keeps PyTorch's own reshard_after_forward default, as generic traversal does.
    assert "reshard_after_forward" not in shard_calls[0][1]


def test_minimax_h3_text_encoder_loads_only_the_backbone_it_forwards_through():
    """The conditioner reads hidden_states[50]; its 778M LM head must never be built or sharded."""
    encoder = _h3_meta_text_encoder()
    assert not hasattr(encoder.text_encoder, "lm_head")
    assert [name for name, _ in encoder.text_encoder.named_children()] == ["visual", "language_model"]


@pytest.mark.parametrize(
    ("device_type", "height", "width", "grid_hw"),
    [
        pytest.param("cpu", 544, 960, (34, 60), id="cpu-544x960"),
        pytest.param("cpu", 768, 1344, (48, 84), id="cpu-768x1344"),
        pytest.param("cpu", 552, 984, (34, 62), id="cpu-552x984_resampled"),
        pytest.param("cuda", 544, 960, (34, 60), id="cuda-544x960", marks=RunIf(min_gpus=1)),
        pytest.param("cuda", 768, 1344, (48, 84), id="cuda-768x1344", marks=RunIf(min_gpus=1)),
    ],
)
def test_minimax_h3_keyframe_processor_sees_uint8_frames_as_pil_images(device_type, height, width, grid_hw):
    """The released Qwen3-VL image processor sees uint8 keyframes exactly as it sees the reference's PIL keyframes.

    ``Qwen2VLImageProcessorFast`` resolves to the torchvision backend, which turns each PIL image into its uint8
    tensor and leaves the tensors it is given untouched, so both share the CPU's uint8 resize. Canvases that are
    multiples of 32 keep their resolution, so CUDA frames match there too; elsewhere torchvision resizes CUDA
    frames in float32, without the uint8 intermediate, and they differ.
    """
    from PIL import Image

    from fastgen.networks.MiniMaxH3 import network as h3

    root = hf_download(h3.MiniMaxH3.MODEL_ID, allow_patterns=["processor/*"])
    image_processor = h3.Qwen3VLProcessor.from_pretrained(os.path.join(root, "processor")).image_processor
    assert image_processor.backend == "torchvision"

    generator = torch.Generator().manual_seed(0)
    keyframes = torch.randint(0, 256, (2, 3, height, width), dtype=torch.uint8, generator=generator)
    untouched = keyframes.clone()
    pil_keyframes = [Image.fromarray(frame) for frame in keyframes.permute(0, 2, 3, 1).contiguous().numpy()]
    from_pil = image_processor(images=pil_keyframes, return_tensors="pt")
    frames = keyframes.to(device_type)
    from_tensors = image_processor(images=list(frames), return_tensors="pt")

    assert torch.equal(frames.cpu(), untouched)
    assert from_pil["image_grid_thw"].tolist() == [[1, *grid_hw]] * len(keyframes)
    assert torch.equal(from_tensors["image_grid_thw"], from_pil["image_grid_thw"])
    assert torch.equal(from_tensors["pixel_values"].cpu(), from_pil["pixel_values"])


def test_ltx2_vae_statistics_follow_generic_fsdp_buffer_placement():
    """LTX2 latent statistics move with registered buffers while parameters stay offloaded."""
    from fastgen.networks.LTX2.network import LTX2VAE
    from fastgen.utils.distributed.fsdp import _move_registered_buffers_

    class TinyVAE(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.register_buffer("latents_mean", torch.zeros(channels, dtype=torch.float64))
            self.register_buffer("latents_std", torch.ones(channels, dtype=torch.float64))
            self.proj = torch.nn.Linear(1, 1)

    vae = object.__new__(LTX2VAE)
    torch.nn.Module.__init__(vae)
    vae._video_vae = TinyVAE(3)
    vae._audio_vae = TinyVAE(8)

    _move_registered_buffers_(vae, torch.device("meta"))

    for component in (vae._video_vae, vae._audio_vae):
        assert component.latents_mean.device == torch.device("meta")
        assert component.latents_std.device == torch.device("meta")
        assert component.latents_mean.dtype == torch.float64
        assert component.latents_std.dtype == torch.float64
        assert component.proj.weight.device == torch.device("cpu")


def test_minimax_h3_vae_uses_precision_property_without_relocating_inputs():
    from fastgen.networks.MiniMaxH3.network import MINIMAX_H3_FPS, MiniMaxH3VAE

    class FlexiblePrecisionVAE(MiniMaxH3VAE):
        @property
        def precision_model(self):
            return torch.float64

    class Distribution:
        def __init__(self, value):
            self.value = value

        def sample(self, generator=None):
            del generator
            return self.value

        def mode(self):
            return self.value

    class TinyVideoVAE:
        config = SimpleNamespace(latents_mean=(0.0, 0.0, 0.0), latents_std=(1.0, 1.0, 1.0))

        def __init__(self):
            self.encode_inputs = []
            self.decode_inputs = []

        def encode(self, value, return_dict=False):
            assert not return_dict
            self.encode_inputs.append((value.dtype, value.device))
            return (Distribution(value),)

        def decode(self, value, return_dict=False):
            assert not return_dict
            self.decode_inputs.append((value.dtype, value.device))
            return (value,)

    class TinyAudioVAE:
        config = SimpleNamespace(latents_mean=(0.0,), latents_std=(1.0,), latent_channels=1, sampling_rate=32000)
        hop_length = 800

        def __init__(self):
            self.encode_inputs = []
            self.decode_inputs = []

        def encode(self, value, return_dict=False):
            assert not return_dict
            self.encode_inputs.append((value.dtype, value.device))
            # One latent per hop, so the wrapper's latent-count contract holds on this stub too.
            return (Distribution(value[..., :: self.hop_length]),)

        def decode(self, value, return_dict=False):
            assert not return_dict
            self.decode_inputs.append((value.dtype, value.device))
            return (value,)

    vae = FlexiblePrecisionVAE.__new__(FlexiblePrecisionVAE)
    torch.nn.Module.__init__(vae)
    vae._fps = float(MINIMAX_H3_FPS)
    vae._video_vae = TinyVideoVAE()
    vae._audio_vae = TinyAudioVAE()
    vae._set_device("meta")

    # 5 frames is the smallest count the video VAE can encode (``17 * 0 + 5``).
    video = torch.zeros(1, 3, 5, 2, 2, dtype=torch.float32)
    audio = torch.zeros(1, 2, 8, dtype=torch.float32)
    video_latents = vae._encode_video(video, mode="argmax")
    audio_latents = vae._encode_audio(audio)
    decoded_video = vae._decode_video(video_latents)
    decoded_audio = vae._decode_audio(audio_latents)

    expected_boundary = [(torch.float64, torch.device("cpu"))]
    assert vae._video_vae.encode_inputs == expected_boundary
    assert vae._video_vae.decode_inputs == expected_boundary
    assert vae._audio_vae.encode_inputs == expected_boundary
    assert vae._audio_vae.decode_inputs == expected_boundary
    assert video_latents.dtype == decoded_video.dtype == torch.float64
    assert audio_latents.dtype == decoded_audio.dtype == torch.float64

    joint = vae.encode({"video": video.to(torch.bfloat16)}, mode="argmax")
    assert joint["video"].dtype == joint["audio"].dtype == torch.float64


def test_minimax_h3_preprocessors_keep_native_text_and_vae_precision():
    from fastgen.networks.MiniMaxH3.network import (
        MINIMAX_H3_TEXT_ENCODER_LAYER,
        MINIMAX_H3_TEXT_TAG,
        MiniMaxH3TextEncoder,
        MiniMaxH3VAE,
    )

    class TinyTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones((), dtype=torch.bfloat16))

        def forward(self, input_ids, **kwargs):
            del kwargs
            hidden = torch.ones(*input_ids.shape, 3, dtype=self.anchor.dtype)
            return SimpleNamespace(hidden_states=[hidden] * (MINIMAX_H3_TEXT_ENCODER_LAYER + 1))

    text_encoder = MiniMaxH3TextEncoder.__new__(MiniMaxH3TextEncoder)
    torch.nn.Module.__init__(text_encoder)
    text_encoder.text_encoder = TinyTower()
    text_encoder.processor = SimpleNamespace(
        create_mm_token_type_ids=lambda token_ids: [[MINIMAX_H3_TEXT_TAG] * len(token_ids[0])]
    )
    text_encoder._keyframe_presentation = lambda prompt, keyframes: (
        [1, 2],
        [MINIMAX_H3_TEXT_TAG, MINIMAX_H3_TEXT_TAG],
        {},
    )
    text_encoder._record_precision(precision_model=torch.bfloat16)

    condition = text_encoder.encode("prompt")

    assert next(text_encoder.text_encoder.parameters()).dtype == text_encoder.precision_model
    assert condition["prompt_embeds"].dtype == torch.bfloat16
    assert condition["text_token_tags"].dtype == torch.long

    class Distribution:
        def __init__(self, value):
            self.value = value

        def sample(self, generator=None):
            del generator
            return self.value

        def mode(self):
            return self.value

    class TinyVAE:
        device = torch.device("cpu")
        config = SimpleNamespace(latents_mean=(0.0,), latents_std=(1.0,))
        encode_input_dtype = None
        encode_input_device = None

        def encode(self, pixels, return_dict=False):
            assert not return_dict
            self.encode_input_dtype = pixels.dtype
            self.encode_input_device = pixels.device
            return (Distribution(pixels[:, :1]),)

    vae = MiniMaxH3VAE.__new__(MiniMaxH3VAE)
    torch.nn.Module.__init__(vae)
    vae._video_vae = TinyVAE()
    vae._record_precision(precision_model=torch.bfloat16)
    vae._set_device("meta")

    latents = vae.encode(torch.zeros(1, 3, 1, 2, 2, dtype=torch.bfloat16), mode="argmax")

    assert vae.precision_model == torch.float32
    assert vae._video_vae.encode_input_dtype == vae.precision_model
    assert vae._video_vae.encode_input_device == torch.device("cpu")
    assert latents.dtype == vae.precision_model


class _H3Toy(torch.nn.Module):
    _keep_in_fp32_modules = None

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.b = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        self.register_buffer("steps", torch.zeros(3, dtype=torch.long))
        self.register_buffer("scale", torch.ones(3))
        self.register_buffer("scratch", torch.ones(3), persistent=False)


def test_minimax_h3_conditioner_honours_dtype():
    """The conditioner obeys a requested dtype, unlike the transformer and the VAEs.

    Nothing is pinned in it -- Qwen3-VL declares no ``_keep_in_fp32_modules`` and its checkpoint
    is uniformly bfloat16 -- so a cast cannot round away a precision the checkpoint depends on,
    which is the only thing the carve-out policy exists to prevent.  A caller asking for float32
    is choosing to turn 62 GB into 124 GB, and that is its choice.

    Built without ``__init__`` so that no 62 GB conditioner is read.
    """
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3TextEncoder

    wrapper = object.__new__(MiniMaxH3TextEncoder)
    torch.nn.Module.__init__(wrapper)
    wrapper.text_encoder = _H3Toy().to(dtype=torch.bfloat16)

    wrapper.to(dtype=torch.float32)
    assert wrapper.text_encoder.a.weight.dtype == torch.float32
    assert wrapper.text_encoder.scale.dtype == torch.float32
    assert wrapper.text_encoder.steps.dtype == torch.long

    # And a device-only move still leaves the dtype alone.
    wrapper.to("cpu")
    assert wrapper.text_encoder.a.weight.dtype == torch.float32


def test_minimax_h3_unpatchify_features_round_trips_the_packed_sequence():
    """The split is the exact inverse of the packing, for both modalities.

    Built without ``__init__`` so no 62 GB conditioner or 50-block transformer is read.
    """
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3, _unpack_audio_latents
    from fastgen.networks.MiniMaxH3.packing import MiniMaxH3PackedSequence, patchify_video_latents

    batch, patch_size = 2, (1, 2, 2)
    frames, height, width = 2, 4, 4
    inner_dim = 8
    video_channels = inner_dim // (patch_size[0] * patch_size[1] * patch_size[2])
    num_audio_latents, num_condition_video_rows = 3, 2

    net = object.__new__(MiniMaxH3)
    net.patch_size = patch_size

    torch.manual_seed(0)
    video = torch.randn(batch, video_channels, frames, height, width)
    audio = torch.randn(batch, inner_dim, 2, num_audio_latents)

    # Pack each modality into rows exactly as ``forward`` does on the way in.
    video_rows = patchify_video_latents(video, patch_size).unflatten(0, (batch, -1))
    audio_rows = audio.permute(0, 2, 3, 1).flatten(1, 2)
    condition_rows = torch.randn(batch, num_condition_video_rows, inner_dim)
    text_rows = torch.randn(batch, 5, inner_dim)

    # Scatter them into one packed buffer, laid out text | condition-video | audio | video.
    parts = [text_rows, condition_rows, audio_rows, video_rows]
    packed = torch.cat(parts, dim=1)
    offsets, cursor = [], 0
    for part in parts:
        offsets.append(cursor)
        cursor += part.shape[1]
    text_at, cond_at, audio_at, video_at = offsets

    layout = MiniMaxH3PackedSequence(
        sequence_length=packed.shape[1],
        position_ids=torch.zeros(packed.shape[1], 3, dtype=torch.float64),
        token_tags=torch.zeros(packed.shape[1], dtype=torch.long),
        video_indices=torch.cat(
            [
                torch.arange(cond_at, cond_at + num_condition_video_rows),
                torch.arange(video_at, video_at + video_rows.shape[1]),
            ]
        ),
        audio_indices=torch.arange(audio_at, audio_at + audio_rows.shape[1]),
        text_indices=torch.arange(text_at, text_at + text_rows.shape[1]),
        num_condition_video_rows=num_condition_video_rows,
    )

    unpatchified = net._unpatchify_features([packed], layout, frames, height, width, num_audio_latents)

    assert len(unpatchified) == 1
    assert set(unpatchified[0]) == {"video", "audio"}
    # Video comes back on its latent grid, with the patch folded into the channel count.
    torch.testing.assert_close(unpatchified[0]["video"], video)
    assert unpatchified[0]["video"].shape == (batch, video_channels, frames, height, width)
    # Audio comes back the way the output head is unpacked.
    torch.testing.assert_close(unpatchified[0]["audio"], audio)
    assert unpatchified[0]["audio"].shape == (batch, inner_dim, 2, num_audio_latents)
    # The conditioning rows are inputs and must not reach a discriminator.
    torch.testing.assert_close(_unpack_audio_latents(audio_rows, num_audio_latents), unpatchified[0]["audio"])


def test_wan_reuses_caller_model_path_for_all_lazy_components(monkeypatch, tmp_path):
    """A local Wan directory must never be normalized back to a Hub ID for lazy loads."""
    from fastgen.networks.Wan import network as wan_network

    monkeypatch.setenv("HF_HOME", "/tmp/fastgen-hf-cache")
    monkeypatch.setenv("LOCAL_FILES_ONLY", "true")
    local_model = tmp_path / "Wan2.1-T2V-1.3B-Diffusers"
    transformer = MagicMock()

    def initialize_network(self, model_source, load_pretrained):
        assert model_source == local_model
        assert load_pretrained
        self.transformer = transformer
        return 4

    monkeypatch.setattr(wan_network.Wan, "_initialize_network", initialize_network)
    monkeypatch.setattr(wan_network.Wan, "override_transformer_forward", lambda self: None)

    model = wan_network.Wan(
        model_id_or_local_path=local_model,
        enable_logvar_linear=False,
    )
    assert model.model_id_or_local_path == local_model

    text_encoder = MagicMock()
    text_encoder_ctor = MagicMock(return_value=text_encoder)
    video_encoder_ctor = MagicMock(return_value=MagicMock())
    scheduler = wan_network.UniPCMultistepScheduler(
        num_train_timesteps=1000, prediction_type="flow_prediction", use_flow_sigmas=True
    )
    scheduler_loader = MagicMock(return_value=scheduler)
    monkeypatch.setattr(wan_network, "WanTextEncoder", text_encoder_ctor)
    monkeypatch.setattr(wan_network, "WanVideoEncoder", video_encoder_ctor)
    monkeypatch.setattr(wan_network.UniPCMultistepScheduler, "from_pretrained", scheduler_loader)

    model.init_preprocessors()
    first, _ = model.sample_scheduler_and_grid(4, flow_shift=5.0)
    second, _ = model.sample_scheduler_and_grid(4, flow_shift=5.0)
    assert first is second is scheduler

    text_encoder_ctor.assert_called_once_with(model_id_or_local_path=local_model)
    video_encoder_ctor.assert_called_once_with(
        model_id_or_local_path=local_model,
        fixed_precision_model=torch.float32,
    )
    scheduler_loader.assert_called_once_with(
        local_model,
        subfolder="scheduler",
        flow_shift=1.0,
        cache_dir="/tmp/fastgen-hf-cache",
        local_files_only=True,
    )


def test_wan_sample_runs_the_released_unipc_config_down_to_sigma_zero(tmp_path):
    """Wan loads the released UniPC config without its shift, applies UniPC's rational shift once over the RF
    schedule's range, and queries one time per step. The released ``final_sigmas_type="zero"`` takes the last step
    to sigma 0, on clean data.
    """
    from diffusers import UniPCMultistepScheduler

    import fastgen.networks.Wan.network as wan_network

    network = object.__new__(wan_network.Wan)
    torch.nn.Module.__init__(network)
    network.schedule_type = "rf"
    network.net_pred_type = "flow"
    network.noise_scheduler = RFNoiseSchedule()
    # scheduler/scheduler_config.json of the Wan2.1-T2V-1.3B/14B and Wan2.2-T2V-A14B Diffusers releases.
    UniPCMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=3.0,
        solver_order=2,
        solver_type="bh2",
        predict_x0=True,
        lower_order_final=True,
        final_sigmas_type="zero",
        timestep_spacing="linspace",
    ).save_config(tmp_path / "scheduler")
    network.model_id_or_local_path = tmp_path
    network._unipc_scheduler = None
    captured = []

    def forward(x_t, t, condition=None, output_dtype="same", **kwargs):
        captured.append(t.clone())
        return cast_like(torch.zeros_like(x_t), x_t, output_dtype)

    network.forward = forward
    steps = 4
    with torch.no_grad():
        network.sample(
            torch.randn(1, 4, 2, 4, 4),
            condition=torch.randn(1, 3, 8),
            sample_steps=steps,
            shift=5.0,
            guidance_cfg=GuidanceConfig(),
        )

    scheduler = network._unipc_scheduler
    assert len(captured) == steps
    # Independent UniPC flow-shift formula, mapped from [0, 1] onto the RF schedule's own range.
    # Wan supplies pre-shifted sigmas, so the scheduler must not apply the shift a second time.
    normalized_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    expected_sigmas = (network.noise_scheduler.max_t * 5.0 * normalized_grid / (1.0 + 4.0 * normalized_grid)).float()
    assert scheduler.config.flow_shift == 1.0
    torch.testing.assert_close(scheduler.sigmas, expected_sigmas, rtol=0, atol=1e-7)
    torch.testing.assert_close(
        torch.cat(captured),
        expected_sigmas[:-1].to(network.noise_scheduler.t_precision),
        rtol=0,
        atol=1e-7,
    )
    assert scheduler.sigmas[-1].item() == 0.0


def test_wan_sample_runs_every_step_once_when_the_first_timesteps_repeat():
    """UniPC truncates ``1000 * sigma`` to integer timesteps, so a fine, strongly shifted grid repeats its first
    timestep, and UniPC's timestep lookup would start at the second step and run past the last sigma. The sampler
    starts at step 0, so every step runs once and in order."""
    from diffusers import UniPCMultistepScheduler

    import fastgen.networks.Wan.network as wan_network

    network = object.__new__(wan_network.Wan)
    torch.nn.Module.__init__(network)
    network.schedule_type = "rf"
    network.net_pred_type = "flow"
    network.noise_scheduler = RFNoiseSchedule()
    network._unipc_scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000, prediction_type="flow_prediction", use_flow_sigmas=True
    )
    captured = []

    def forward(x_t, t, condition=None, output_dtype="same", **kwargs):
        captured.append(t[0].item())
        return cast_like(torch.zeros_like(x_t), x_t, output_dtype)

    network.forward = forward
    steps, shift = 100, 16.0  # VACE's official shift
    scheduler, t_grid = network.sample_scheduler_and_grid(steps, flow_shift=shift)
    assert scheduler.timesteps[0] == scheduler.timesteps[1]
    with torch.no_grad():
        network.sample(
            torch.randn(1, 4, 2, 4, 4),
            condition=torch.randn(1, 3, 8),
            sample_steps=steps,
            shift=shift,
            guidance_cfg=GuidanceConfig(),
        )

    assert captured == t_grid.tolist()
    assert scheduler.step_index == steps


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_minimax_h3_set_precision_keeps_the_carve_out(dtype):
    """The explicit precision boundary retains the declared state in float32."""
    net = _h3_meta_net()
    net._set_precision(precision_model=dtype)
    expected = {name: torch.float32 if _h3_is_pinned(name) else dtype for name in _h3_dtype_map(net.transformer)}
    assert _h3_dtype_map(net.transformer) == expected
    # Idempotent: the boundary already brought the module to the requested map.
    before = _h3_dtype_map(net.transformer)
    net._set_precision(precision_model=dtype)
    assert _h3_dtype_map(net.transformer) == before


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda net: net.to(dtype=torch.bfloat16), id="dtype-kwarg"),
        pytest.param(lambda net: net.to(torch.bfloat16), id="dtype-arg"),
        pytest.param(lambda net: net.to(torch.zeros(1, dtype=torch.bfloat16, device="meta")), id="tensor"),
        pytest.param(lambda net: net.to(device="meta", dtype=torch.bfloat16), id="device-and-dtype"),
    ],
)
def test_minimax_h3_every_to_form_is_a_native_uniform_cast(call):
    """Ordinary ``to`` retains native semantics, including the ``to(other_tensor)`` overload."""
    net = _h3_meta_net()
    call(net)
    assert all(dtype == torch.bfloat16 for dtype in _h3_dtype_map(net.transformer).values())


def test_minimax_h3_device_only_move_leaves_dtypes_alone():
    net = _h3_meta_net()
    net._set_precision(precision_model=torch.bfloat16)
    before = _h3_dtype_map(net.transformer)
    net.to("meta")
    assert _h3_dtype_map(net.transformer) == before


def test_minimax_h3_keep_in_fp32_false_casts_uniformly():
    """The escape hatch: stock behaviour, for a caller that wants one dtype everywhere."""
    net = _h3_meta_net(keep_in_fp32=False)
    assert tuple(net.transformer._keep_in_fp32_modules) == tuple(MINIMAX_H3_TRANSFORMER_FP32)

    net._set_precision(precision_model=torch.bfloat16)
    assert all(dtype == torch.bfloat16 for dtype in _h3_dtype_map(net.transformer).values())


def test_minimax_h3_rope_reset_preserves_prepared_storage():
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3RotaryPosEmbed

    net = _h3_meta_net(keep_in_fp32=False)
    net._set_precision(precision_model=torch.bfloat16)
    rope_modules = [module for module in net.transformer.modules() if isinstance(module, MiniMaxH3RotaryPosEmbed)]
    storage = [module.inv_freq for module in rope_modules]
    assert rope_modules

    net._reset_rope_buffers()

    assert all(module.inv_freq is buffer for module, buffer in zip(rope_modules, storage))
    assert all(module.inv_freq.dtype is torch.bfloat16 for module in rope_modules)


def test_minimax_h3_fully_shard_uses_fp32_policy_for_declared_modules(monkeypatch):
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from fastgen.networks.MiniMaxH3 import network as minimax_h3_network

    net = _h3_meta_net(disable_grad_ckpt=True)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    wrapped = []
    monkeypatch.setattr(minimax_h3_network, "fully_shard", lambda module, **kwargs: wrapped.append((module, kwargs)))

    net.fully_shard(mp_policy=policy)

    wrapped_kwargs = {module: wrap_kwargs for module, wrap_kwargs in wrapped}
    for module in (
        net.transformer.proj_in,
        net.transformer.audio_proj_in,
        net.transformer.time_embedder,
        net.transformer.proj_out,
        net.transformer.audio_proj_out,
    ):
        assert wrapped_kwargs[module]["mp_policy"].param_dtype == torch.float32
        assert wrapped_kwargs[module]["mp_policy"].reduce_dtype == policy.reduce_dtype
    assert all(wrapped_kwargs[block]["mp_policy"] is policy for block in net.transformer.transformer_blocks)
    assert wrapped_kwargs[net.transformer]["mp_policy"] is policy


def test_minimax_h3_keep_in_fp32_false_uses_one_fsdp_policy(monkeypatch):
    """Disabling the carve-out applies to FSDP compute as well as the pre-DDP cast."""
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from fastgen.networks.MiniMaxH3 import network as minimax_h3_network

    net = _h3_meta_net(keep_in_fp32=False, disable_grad_ckpt=True)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    wrapped = []
    monkeypatch.setattr(minimax_h3_network, "fully_shard", lambda module, **kwargs: wrapped.append((module, kwargs)))

    net.fully_shard(mp_policy=policy)

    assert wrapped
    assert all(wrap_kwargs["mp_policy"] is policy for _, wrap_kwargs in wrapped)


def test_minimax_h3_mixed_input_output_precision_matches_diffusers():
    """Every FP32 island receives FP32 and reconnects to the BF16 block stack explicitly."""
    from diffusers.models.transformers import MiniMaxH3Transformer3DModel

    from fastgen.networks.MiniMaxH3.network import classify_forward
    from fastgen.utils.module import (
        cast_to_storage_dtypes,
        matches_fp32_declaration,
        module_state_names,
        resolve_storage_dtypes,
    )

    transformer = MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=6,
        hidden_size=12,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=24,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=12,
        time_embed_dim=6,
        rope_freq_dim=1,
    ).eval()
    dtype_overrides = {
        name: torch.float32 for name in module_state_names(transformer) if matches_fp32_declaration(name, transformer)
    }
    cast_to_storage_dtypes(transformer, resolve_storage_dtypes(transformer, torch.bfloat16, dtype_overrides))

    observed = {}

    def capture(name):
        def hook(module, inputs, output):
            observed[name] = (inputs[0].dtype, output.dtype)

        return hook

    def capture_rope(module, inputs, output):
        observed["rope"] = (inputs[0].dtype, output[0].dtype, output[1].dtype)

    def capture_block(module, inputs, output):
        observed["block"] = (inputs[0].dtype, inputs[1].dtype, output.dtype)

    for name, module in {
        "video_in": transformer.proj_in,
        "audio_in": transformer.audio_proj_in,
        "text_in": transformer.context_embedder,
        "time": transformer.time_embedder,
        "video_out": transformer.proj_out,
        "audio_out": transformer.audio_proj_out,
    }.items():
        module.register_forward_hook(capture(name))
    transformer.rope.register_forward_hook(capture_rope)
    transformer.transformer_blocks[0].register_forward_hook(capture_block)

    with torch.no_grad():
        video, audio, _, _ = classify_forward(
            transformer,
            hidden_states=torch.randn(1, 1, 2, dtype=torch.bfloat16),
            audio_hidden_states=torch.randn(1, 1, 2, dtype=torch.bfloat16),
            encoder_hidden_states=torch.randn(1, 1, 8, dtype=torch.bfloat16),
            timestep=torch.tensor([0.5], dtype=torch.bfloat16),
            timestep_indices=torch.zeros(3, dtype=torch.long),
            token_tags=torch.tensor([0, 2, 1]),
            position_ids=torch.zeros(3, 3, dtype=torch.long),
            video_indices=torch.tensor([0]),
            audio_indices=torch.tensor([1]),
            text_indices=torch.tensor([2]),
        )

    assert observed == {
        "video_in": (torch.float32, torch.float32),
        "audio_in": (torch.float32, torch.float32),
        "text_in": (torch.bfloat16, torch.bfloat16),
        "time": (torch.float32, torch.float32),
        "rope": (torch.int64, torch.float32, torch.float32),
        "block": (torch.bfloat16, torch.float32, torch.bfloat16),
        "video_out": (torch.float32, torch.float32),
        "audio_out": (torch.float32, torch.float32),
    }
    assert video.dtype == torch.float32
    assert audio.dtype == torch.float32


def test_minimax_h3_runs_its_two_modalities_on_different_clocks():
    """Video is the clock: whoever grids carries its 12, the schedule only audio's ``3/12``.

    So a grid point *is* video's noise level, and audio's is that point under the ratio -- which
    composes back to audio's shift of 3, since the warps multiply.  Splitting the placement this way
    is what keeps ``sigma'(t) = 1`` for video; the failure it invites instead is video's
    sub-schedule warping by 12 on top of the grid's 12, landing it on ``f_144``.
    """
    from fastgen.networks.MiniMaxH3.network import MINIMAX_H3_FLOW_SHIFT

    expected_shifts = {"video": 12.0, "audio": 3.0}
    net = _h3_meta_net()
    assert not hasattr(net, "flow_shift"), "the network holds no clock; whoever grids does"
    assert type(net.noise_scheduler.subschedules["video"]) is RFNoiseSchedule, "video is the clock"
    assert net.noise_scheduler.subschedules["audio"].shift == net.relative_audio_shift
    assert net.relative_audio_shift == 3.0 / 12.0, "audio's 3 against video's 12"
    # The schedule built inside the meta context holds meta sigmas, which a grid cannot be read
    # off; rebuilding it here is the cheapest way to keep the rest of the network weightless.
    net._set_noise_schedule()

    ns = net.noise_scheduler
    uniform = ns.student_grid(8)
    max_t = ns.subschedules["video"].max_t
    shifted_grid = ns.primary.shift_t(uniform, shift=MINIMAX_H3_FLOW_SHIFT)

    # Each modality's noise level lands at its expected shift, measured off the uniform list so
    # the test states those shifts directly rather than restating the implementation.
    sigmas = ns.sigma(shifted_grid)
    for modality, expected_shift in expected_shifts.items():
        expected = time_shift(uniform, expected_shift, max_t=ns.subschedules[modality].max_t)
        assert torch.allclose(sigmas[modality], expected), modality
        assert (sigmas[modality] <= max_t).all(), f"{modality}: above max_t"
        assert (sigmas[modality].diff() < 0).all(), f"{modality}: not strictly decreasing"

    assert torch.equal(sigmas["video"], shifted_grid), "video is the clock, so its sigma is the grid"
    assert not torch.allclose(sigmas["video"], sigmas["audio"]), "the two noise levels must differ"
    # The failure this guards against: video warped by the grid *and* by its sub-schedule.
    assert not torch.allclose(sigmas["video"], time_shift(uniform, 12.0 * 12.0, max_t=max_t))
    # Video's "rf" sub-schedule takes no ``shift``, so one is refused rather than applied.
    with pytest.raises(TypeError, match="shift"):
        net._set_noise_schedule(shift=12.0)
    assert net.noise_scheduler is ns


def test_minimax_h3_sample_forwards_negative_condition(monkeypatch):
    from fastgen.networks.MiniMaxH3 import network as h3_network
    from fastgen.utils.guidance import CFGTerm, GuidanceConfig

    net = object.__new__(h3_network.MiniMaxH3)
    torch.nn.Module.__init__(net)
    net.modalities = ("video", "audio")
    net.relative_audio_shift = h3_network.MINIMAX_H3_RELATIVE_AUDIO_SHIFT
    net.schedule_type = "rf"
    net._set_noise_schedule()

    negative_condition = object()
    guidance_cfg = GuidanceConfig(cfg=CFGTerm(scale=2.0))
    observed = []

    def fake_predict_with_guidance(network, latents, timestep, **kwargs):
        observed.append(
            (
                network,
                kwargs["neg_condition"],
                kwargs["guidance_cfg"],
                timestep.dtype,
                kwargs["output_dtype"],
            )
        )
        return {modality: torch.zeros_like(value) for modality, value in latents.items()}

    monkeypatch.setattr(h3_network, "predict_with_guidance", fake_predict_with_guidance)
    noise = {"video": torch.ones(1, 1), "audio": torch.ones(1, 1)}
    net.sample(
        noise,
        condition={"prompt_embeds": torch.ones(1, 1)},
        neg_condition=negative_condition,
        guidance_cfg=guidance_cfg,
        sample_steps=1,
    )

    assert observed == [(net, negative_condition, guidance_cfg, net.noise_scheduler.t_precision, None)]


def _h3_sampling_net(transformer=None, **schedule_kwargs):
    """A weightless ``MiniMaxH3``: its schedule, layout and forward plumbing around ``transformer``."""
    from fastgen.networks.MiniMaxH3 import network as h3_network

    net = object.__new__(h3_network.MiniMaxH3)
    torch.nn.Module.__init__(net)
    net.modalities = ("video", "audio")
    net.relative_audio_shift = h3_network.MINIMAX_H3_RELATIVE_AUDIO_SHIFT
    net.schedule_type = "rf"
    net.net_pred_type = "flow"
    net.fps = float(h3_network.MINIMAX_H3_FPS)
    net.patch_size = (1, 1, 1)
    net._layout_cache = {}
    net.transformer = transformer
    net._set_noise_schedule(**schedule_kwargs)
    return net


def test_minimax_h3_sample_walks_the_release_grid(monkeypatch):
    """The default 49 forwards query SGLang's 50-point grid from sigma 1.

    MiniMax's README serves the release with SGLang, whose default ``num_inference_steps=50`` counts grid points.
    SGLang builds that grid in float32 and FastGen in the scheduler's precision, and audio reaches its shift of 3
    through the composed relative warp, so the timesteps agree to float32 rounding rather than bit for bit.
    """
    from fastgen.configs.net import MiniMaxH3_Config
    from fastgen.networks.MiniMaxH3 import network as h3_network

    net = _h3_sampling_net(**network_schedule_kwargs(MiniMaxH3_Config))
    observed = []

    def fake_predict_with_guidance(network, latents, timestep, **kwargs):
        sigmas = network.noise_scheduler.sigma(timestep)
        observed.append({name: 1.0 - float(sigma[0]) for name, sigma in sigmas.items()})
        return {name: torch.zeros_like(value) for name, value in latents.items()}

    monkeypatch.setattr(h3_network, "predict_with_guidance", fake_predict_with_guidance)
    net.sample({"video": torch.ones(1, 1), "audio": torch.ones(1, 1)}, condition={})

    def minimax_h3_time_shift_sigmas(*, num_steps, shift_scale):
        # Verbatim from SGLang's ``minimax_h3/time_request.py``, minus its argument checks.
        base = torch.linspace(1.0, 0.0, int(num_steps), device="cpu", dtype=torch.float32)
        shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
        shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
        if num_steps > 1 and shifted[-1].item() > 0.0:
            shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
        return [float(value) for value in shifted.tolist()]

    # The release's ``sigma_shift_scales``. SGLang's denoise loop runs one forward per sigma but the terminal
    # one, at ``t = 1 - sigma``.
    for name, shift in (("video", 12.0), ("audio", 3.0)):
        sigmas = minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=shift)
        reference = torch.tensor([1.0 - sigma for sigma in sigmas[:-1]], dtype=torch.float32)
        actual = torch.tensor([step[name] for step in observed], dtype=torch.float32)
        assert actual.shape == reference.shape == (49,)
        torch.testing.assert_close(actual, reference, rtol=0, atol=1e-6)


def test_minimax_h3_keyframe_rows_reach_the_transformer_in_float32():
    """Float32 keyframe rows lead the video rows unrounded, in the FP32 sampling loop and on a bfloat16 state.

    The reference feeds float32 keyframe rows to its float32 input projection, so ``forward`` rejects rows
    already rounded to another precision.
    """
    from fastgen.networks.MiniMaxH3.packing import MINIMAX_H3_TEXT_TAG, patchify_video_latents

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList()
            self.video_rows = []

        def forward(self, hidden_states, audio_hidden_states, **kwargs):
            self.video_rows.append(hidden_states)
            return torch.zeros_like(hidden_states), torch.zeros_like(audio_hidden_states), None, None

    net = _h3_sampling_net(Transformer(), max_t=1.0)
    state_dtype = torch.bfloat16
    # One first-frame keyframe of a 1 x 2 latent frame with 2 channels: two conditioning rows.
    keyframe_rows = torch.randn(1, 2, 2)
    assert not torch.equal(keyframe_rows.to(state_dtype).float(), keyframe_rows)
    condition = {
        "prompt_embeds": torch.zeros(1, 3, 4, dtype=state_dtype),
        "text_token_tags": torch.full((3,), MINIMAX_H3_TEXT_TAG),
        "cond_video_rows": keyframe_rows,
        "keyframe_anchors": ("first",),
    }
    noise = {
        "video": torch.randn(1, 2, 2, 1, 2, dtype=state_dtype),
        "audio": torch.randn(1, 2, 2, 3, dtype=state_dtype),
    }

    out = net.sample(noise, condition=condition, sample_steps=2)

    assert {name: value.dtype for name, value in out.items()} == {"video": torch.float32, "audio": torch.float32}
    assert len(net.transformer.video_rows) == 2
    for rows in net.transformer.video_rows:
        assert rows.dtype == torch.float32
        assert torch.equal(rows[:, :2], keyframe_rows), "the keyframe rows lead the video rows, unrounded"

    net.transformer.video_rows.clear()
    t = torch.full((1,), 0.5, dtype=net.noise_scheduler.t_precision)
    net(noise, t, condition=condition)
    [rows] = net.transformer.video_rows
    generated = patchify_video_latents(noise["video"], net.patch_size).unflatten(0, (1, -1))
    assert torch.equal(rows, torch.cat([keyframe_rows, generated.float()], dim=1))

    with pytest.raises(AssertionError, match="cond_video_rows"):
        net(noise, t, condition={**condition, "cond_video_rows": keyframe_rows.to(state_dtype)})


@pytest.mark.parametrize("missing", ["cond_video_rows", "keyframe_anchors"])
def test_minimax_h3_forward_requires_keyframe_rows_and_anchors_together(missing):
    """The anchors place the keyframe rows, so rows without anchors are rejected rather than dropped."""
    net = _h3_sampling_net(SimpleNamespace(transformer_blocks=[]))
    condition = {
        "prompt_embeds": torch.zeros(1, 1, 4),
        "text_token_tags": torch.ones(1, dtype=torch.long),
        "cond_video_rows": torch.zeros(1, 2, 2),
        "keyframe_anchors": ("first",),
    }
    del condition[missing]
    x_t = {"video": torch.zeros(1, 2, 1, 1, 2), "audio": torch.zeros(1, 2, 2, 1)}

    with pytest.raises(ValueError, match="needs both"):
        net(x_t, torch.zeros(1), condition=condition)


def test_minimax_h3_frame_counts_follow_the_vae_chunking():
    """The compression factors are the VAE's own; ``17 * n + 5`` frames encode to ``5 * n + 2`` latent frames."""
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3
    from fastgen.networks.MiniMaxH3.packing import video_latent_num_frames

    assert _h3_meta_vae().latent_compression_factors == MiniMaxH3.latent_compression_factors == (4, 16, 16)
    net = object.__new__(MiniMaxH3)
    for num_latent_frames, num_frames in [(2, 5), (7, 22), (37, 124), (102, 345)]:
        assert MiniMaxH3.pixel_shape_for([24, num_latent_frames, 2, 3]) == (num_frames, 32, 48)
        assert net.pixel_shape([24, num_latent_frames, 2, 3]) == (num_frames, 32, 48)
        assert video_latent_num_frames(num_frames) == num_latent_frames
    with pytest.raises(ValueError, match=r"`5 \* n \+ 2` latent frames, got 13"):
        MiniMaxH3.pixel_shape_for([24, 13, 2, 3])


def test_minimax_h3_text_encoder_presents_prompts_verbatim():
    """Diffusers and SGLang tokenize the prompt as given, surrounding whitespace included."""
    from fastgen.networks.MiniMaxH3.network import (
        MINIMAX_H3_TEXT_ENCODER_LAYER,
        MINIMAX_H3_TEXT_TAG,
        MiniMaxH3TextEncoder,
    )

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones((), dtype=torch.bfloat16))

        def forward(self, input_ids, **kwargs):
            hidden = torch.ones(*input_ids.shape, 3, dtype=self.anchor.dtype)
            return SimpleNamespace(hidden_states=[hidden] * (MINIMAX_H3_TEXT_ENCODER_LAYER + 1))

    presented = []

    def present(prompt, keyframes):
        presented.append(prompt)
        return [1, 2], [MINIMAX_H3_TEXT_TAG] * 2, {}

    text_encoder = MiniMaxH3TextEncoder.__new__(MiniMaxH3TextEncoder)
    torch.nn.Module.__init__(text_encoder)
    text_encoder.text_encoder = Tower()
    text_encoder.processor = SimpleNamespace(create_mm_token_type_ids=lambda token_ids: [[0] * len(token_ids[0])])
    text_encoder._keyframe_presentation = present
    text_encoder._record_precision(precision_model=torch.bfloat16)

    text_encoder.encode(["  a prompt\n"])

    assert presented == ["  a prompt\n"]


def test_minimax_h3_text_encoder_rejects_prompts_that_would_need_padding():
    """A batch shares one packed layout, so prompts of different lengths are refused rather than padded."""
    from fastgen.networks.MiniMaxH3.network import MINIMAX_H3_TEXT_TAG, MiniMaxH3TextEncoder

    text_encoder = MiniMaxH3TextEncoder.__new__(MiniMaxH3TextEncoder)
    torch.nn.Module.__init__(text_encoder)
    text_encoder._keyframe_presentation = lambda prompt, keyframes: (
        [1] * len(prompt),
        [MINIMAX_H3_TEXT_TAG] * len(prompt),
        {},
    )

    with pytest.raises(ValueError, match="no padding row"):
        text_encoder.encode(["two", "three"])


def test_minimax_h3_scales_only_the_audio_branch_into_dx_dt():
    """FastGen's ``flow`` is ``dx/dt``, and only audio needs converting into it.

    The heads emit ``eps - x0``; the two differ by ``sigma'``.  Video is the clock, so its
    ``sigma'`` is one and its head already speaks ``dx/dt``; audio, on the relative warp, is
    scaled.  Getting this wrong would put the audio branch's training target a factor of
    ``sigma' in [0.25, 4]`` away from what the head emits.
    """
    net = _h3_meta_net()
    net._set_noise_schedule()

    raw = {"video": torch.randn(2, 4), "audio": torch.randn(2, 4)}
    t = torch.tensor([0.3, 0.6])

    converted = net._to_flow_convention(raw, t)
    assert torch.equal(converted["video"], raw["video"]), "video's sigma' is one, so it is untouched"
    sigma_prime = net.noise_scheduler.subschedules["audio"].sigma_prime(t)
    torch.testing.assert_close(converted["audio"], raw["audio"].double() * sigma_prime[:, None].double())


def test_minimax_h3_logvar_head_follows_the_block_dtype():
    """It is not in the checkpoint, so the precision boundary carries it with the block stack.

    If it were accidentally included in the override map, it would remain float32 and
    ``set_model_state_dict(broadcast_from_rank0=True)`` would have to reconcile the difference
    against a rank that correctly used the block dtype.
    """
    net = _h3_meta_net(enable_logvar_linear=True)
    net._set_precision(precision_model=torch.bfloat16)
    assert net.transformer.logvar_linear.weight.dtype == torch.bfloat16
    before = _h3_dtype_map(net.transformer)
    net._set_precision(precision_model=torch.bfloat16)
    assert _h3_dtype_map(net.transformer) == before


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_minimax_h3():
    """MiniMax-H3 joint video + audio: packed forward, feature capture, sample, and both decodes.

    Video and audio are denoised in one packed sequence, so ``x_t`` is always a dict with both
    keys and both output heads always run -- there is no unimodal case to parametrize, unlike
    LTX-2.  The 62 GB conditioner is not built: ``_h3_condition`` synthesizes what the
    transformer reads from it, and the VAE is constructed directly so that
    ``init_preprocessors`` does not pull the conditioner in behind it.
    """
    from fastgen.configs.net import MiniMaxH3_Config
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3VAE

    clear_gpu_memory()
    set_env_vars()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    try:
        teacher = instantiate(MiniMaxH3_Config)
    except OSError as e:
        if "not a valid model identifier" in str(e) or "token" in str(e) or "does not appear" in str(e):
            pytest.skip(f"Test skipped: MiniMax-H3 model not accessible: {e}")
        raise

    # One block is enough for the packing, guidance and sampling paths under test.
    teacher.transformer.transformer_blocks = teacher.transformer.transformer_blocks[:1]
    teacher.preprocessors["vae"] = MiniMaxH3VAE(teacher.model_root)
    teacher = teacher.to(device=device, dtype=dtype)
    teacher._set_device(device)
    _load_preprocessors(teacher, device, dtype)

    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    # Latent frames follow ``F = 5n + 2``, the shape of MiniMax-H3's 17-frames-to-5-latents
    # chunking; the audio shape is derived from it on the shared 40 Hz clock.
    shape = teacher.infer_shape({"video": [teacher.video_latent_channels, 7, 4, 4], "audio": None})
    x_t = {name: torch.randn(batch_size, *dims, device=device, dtype=_STATE_DTYPE) for name, dims in shape.items()}
    assert set(x_t) == {"video", "audio"}

    # One shared time is already video's noise level: video's sub-schedule is the identity and
    # audio's applies the relative 3/12 warp. ``forward`` asks both for their conditioning times.
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device)
    sigmas = teacher.noise_scheduler.sigma(t)
    assert not torch.allclose(sigmas["video"], sigmas["audio"]), "the two noise levels must differ"
    condition = _h3_condition(teacher, batch_size, seq_len=16, device=device, dtype=_STATE_DTYPE)

    output = teacher(x_t, t, condition=condition)
    assert isinstance(output, dict) and set(output) == {"video", "audio"}
    for name, value in output.items():
        assert value.shape == x_t[name].shape
        assert value.dtype == x_t[name].dtype, "predictions come back in the dtype of the latents"

    # Per-block feature capture starts from one packed sequence, then the forward path splits
    # each block activation back onto the modality grids consumed by CompositeDiscriminator.
    features = teacher(x_t, t, condition=condition, return_features_early=True, feature_indices={0})
    assert isinstance(features, list) and len(features) == 1
    assert isinstance(features[0], dict) and set(features[0]) == {"video", "audio"}
    for value in features[0].values():
        assert torch.is_tensor(value) and torch.isfinite(value).all()

    # Audio and video sample on separate sigma grids (shift 3 and 12), which ``sample`` steps in
    # lockstep: two forwards over three grid points.
    _smoke_sample(teacher, x_t, condition=condition, neg_condition=condition, sample_steps=2, decode_after=True)

    del teacher
    clear_gpu_memory()


def test_network_constructors_reject_keys_unknown_to_the_network_and_its_schedule():
    """Kwargs a network does not name go to its noise schedule; a stale or misspelt key must not be swallowed."""

    class _Net(FastGenNetwork):
        def __init__(self, **kwargs):
            super().__init__(net_pred_type="flow", schedule_type="rf", **kwargs)

        def forward(self, x_t, t, **kwargs):
            return x_t

    assert _Net(max_t=0.9).noise_scheduler.max_t == 0.9
    with pytest.raises(TypeError, match="bogus_key"):
        _Net(bogus_key=1)


def test_edm_family_routes_schedule_kwargs_to_the_schedule():
    """EDM builds its backbone from a nested ``L(...)`` node, schedule kwargs sit at the top; stray keys raise."""
    from fastgen.configs.net import EDM2_IN64_S_Config, EDM_CIFAR10_Config
    from fastgen.networks.EDM.network import SongUNet

    edm = copy.deepcopy(EDM_CIFAR10_Config)
    edm.max_t = 40.0
    edm.backbone.img_resolution, edm.backbone.channel_mult, edm.backbone.channel_mult_noise = 8, [1], 1
    edm2 = copy.deepcopy(EDM2_IN64_S_Config)
    edm2.img_resolution, edm2.model_channels, edm2.channel_mult, edm2.num_blocks = 8, 8, [1], 1
    edm2.attn_resolutions, edm2.max_t = [], 40.0
    for config in (edm, edm2):
        network = instantiate(config)
        assert network.noise_scheduler.max_t == 40.0, type(network).__name__
        if config is edm:
            assert isinstance(network.model, SongUNet) and network.label_dim == network.model.label_dim == 10

        misspelt = copy.deepcopy(config)
        misspelt.bogus_key = 1
        with pytest.raises(TypeError, match="bogus_key"):
            instantiate(misspelt)

    flat = copy.deepcopy(edm)
    flat.dropout = 0.1
    with pytest.raises(TypeError, match="dropout"):
        instantiate(flat)
    nested = copy.deepcopy(edm)
    nested.backbone.bogus_key = 1
    with pytest.raises(TypeError, match="bogus_key"):
        instantiate(nested)
    # A plain kwargs dict is not a backbone: the node needs its ``_target_``.
    untargeted = copy.deepcopy(edm)
    untargeted.backbone = {key: value for key, value in edm.backbone.items() if key != "_target_"}
    with pytest.raises(TypeError, match="backbone must be an instantiated module"):
        instantiate(untargeted)


@pytest.mark.parametrize(("latent_channels", "latent_compression_factors"), [(16, (4, 8, 8)), (48, (4, 16, 16))])
def test_wan_compression_follows_the_latent_channels(monkeypatch, latent_channels, latent_compression_factors):
    """Wan's two VAEs differ in latent channels: the Wan2.1 VAE's 16 downsample 8x, Wan2.2's 48 (TI2V-5B) 16x."""
    from fastgen.networks.VaceWan.network import VACEWan
    from fastgen.networks.Wan import network as wan_network
    from fastgen.networks.Wan.network_pdd import WanPDD
    from fastgen.networks.WanI2V.network import WanI2V

    t_factor, h_factor, w_factor = latent_compression_factors
    expected = (4 * t_factor + 1, 4 * h_factor, 6 * w_factor)
    for network_cls in (wan_network.Wan, WanI2V, VACEWan, WanPDD):
        assert network_cls.pixel_shape_for([latent_channels, 5, 4, 6]) == expected

    # An instance reads the channels its transformer outputs.
    def initialize_network(self, model_source, load_pretrained):
        self.transformer = MagicMock()
        self.transformer.config.out_channels = latent_channels
        return 4

    monkeypatch.setattr(wan_network.Wan, "_initialize_network", initialize_network)
    monkeypatch.setattr(wan_network.Wan, "override_transformer_forward", lambda self: None)
    network = wan_network.Wan(enable_logvar_linear=False)
    assert network.latent_compression_factors == latent_compression_factors
    assert network.pixel_shape([latent_channels, 5, 4, 6]) == expected
    # TI2V-5B, the one Wan on 48-channel latents, also conditions each latent token on its own timestep.
    assert network.expand_timesteps == (latent_channels == 48)


# Fields of the released Wan I2V transformer configs that ``WanI2V._initialize_network`` reads, as on the Hub.
_WAN_I2V_TRANSFORMER_CONFIGS = {
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": dict(
        in_channels=36, out_channels=16, num_attention_heads=40, attention_head_dim=128, image_dim=1280
    ),
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": dict(
        in_channels=36, out_channels=16, num_attention_heads=40, attention_head_dim=128, image_dim=1280
    ),
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": dict(
        in_channels=36, out_channels=16, num_attention_heads=40, attention_head_dim=128, image_dim=None
    ),
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": dict(
        in_channels=48, out_channels=48, num_attention_heads=24, attention_head_dim=128, image_dim=None
    ),
}


def _stub_wan_i2v_transformer(monkeypatch, wan_i2v_network, model_id):
    """Build ``WanI2V``'s transformer from the checkpoint's config fields, without its weights."""
    config = _WAN_I2V_TRANSFORMER_CONFIGS[model_id]

    def from_config(config):
        transformer = torch.nn.Module()
        transformer.config = SimpleNamespace(**config)
        return transformer

    monkeypatch.setattr(wan_i2v_network.WanTransformer3DModel, "load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(wan_i2v_network.WanTransformer3DModel, "from_config", from_config)


@pytest.mark.parametrize(
    ("model_id", "inner_dim", "concat_mask", "use_image_encoder"),
    [
        ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", 5120, True, True),
        ("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", 5120, True, True),
        ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", 5120, True, False),
        ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", 3072, False, False),
    ],
)
def test_wan_i2v_reads_its_conditioning_from_the_transformer_config(
    monkeypatch, model_id, inner_dim, concat_mask, use_image_encoder
):
    """Mask concatenation, the CLIP image encoder and the width follow the transformer config, not the model id."""
    from fastgen.networks.WanI2V import network as wan_i2v_network

    _stub_wan_i2v_transformer(monkeypatch, wan_i2v_network, model_id)
    network = wan_i2v_network.WanI2V.__new__(wan_i2v_network.WanI2V)
    torch.nn.Module.__init__(network)
    network.transformer_subfolder = "transformer"

    assert network._initialize_network(model_id, load_pretrained=False) == inner_dim
    assert network.concat_mask == concat_mask
    assert network.use_image_encoder == use_image_encoder


@pytest.mark.parametrize(
    ("model_id", "pixel_hw", "sample_steps", "sample_shift"),
    [
        # Wan2.1 ``generate.py``: I2V samples 40 steps, at shift 3.0 within the 480P pixel budget and 5.0 above it,
        # whichever checkpoint is loaded. The budget bounds the area: upstream fits a 4:3 image into it at (544, 720).
        ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", (480, 832), 40, 3.0),
        ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", (544, 720), 40, 3.0),
        ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", (720, 1280), 40, 5.0),
        ("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", (720, 1280), 40, 5.0),
        ("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", (832, 480), 40, 3.0),
        # The Wan2.2 task configs ``i2v_A14B`` and ``ti2v_5B`` use one setting at every resolution.
        ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", (480, 832), 40, 5.0),
        ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", (720, 1280), 40, 5.0),
        ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", (480, 832), 50, 5.0),
        ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", (704, 1280), 50, 5.0),
    ],
)
@pytest.mark.parametrize("causal", [False, True], ids=["base", "causal"])
def test_wan_i2v_sampler_defaults_follow_the_checkpoint_and_resolution(
    monkeypatch, model_id, pixel_hw, sample_steps, sample_shift, causal
):
    from fastgen.networks.Wan.wan_moe import WanMoE
    from fastgen.networks.WanI2V import network as wan_i2v_network
    from fastgen.networks.WanI2V.network_causal import CausalWanI2V
    from fastgen.networks.WanI2V.wan_moe import WanMoEI2V

    _stub_wan_i2v_transformer(monkeypatch, wan_i2v_network, model_id)
    network_cls = CausalWanI2V if causal else wan_i2v_network.WanI2V
    network = network_cls.__new__(network_cls)
    torch.nn.Module.__init__(network)
    network.transformer_subfolder = "transformer"
    network.schedule_type = "rf"
    network._initialize_network(model_id, load_pretrained=False)
    # ``Wan.__init__``, which this stand-in skips, sets these from the 48 channels of TI2V-5B.
    wan22_vae = _WAN_I2V_TRANSFORMER_CONFIGS[model_id]["out_channels"] == 48
    network.latent_compression_factors = (4, 16, 16) if wan22_vae else (4, 8, 8)
    network.expand_timesteps = wan22_vae
    network.transformer.kv_cache_engines = {"main": None}
    network.chunk_size = network.total_num_frames = 1
    network.clear_caches = lambda: None

    class GridRequested(Exception):
        pass

    def sample_scheduler_and_grid(steps, flow_shift, device=None):
        raise GridRequested(steps, flow_shift)

    network.sample_scheduler_and_grid = sample_scheduler_and_grid
    _, s_h, s_w = network.latent_compression_factors
    input_shape = [4, 1, pixel_hw[0] // s_h, pixel_hw[1] // s_w]
    noise = torch.zeros(1, *input_shape)
    with pytest.raises(GridRequested) as defaults:
        network.sample(noise)
    assert defaults.value.args == (sample_steps, sample_shift)
    with pytest.raises(GridRequested) as overridden:
        network.sample(noise, sample_steps=7, shift=2.0)
    assert overridden.value.args == (7, 2.0)

    # A config's ``input_shape`` resolves the same defaults.
    assert network.resolve_sample_settings(input_shape) == (sample_steps, sample_shift)

    # Both MoE samplers resolve them through the high-noise expert; the generic ``WanMoE.sample`` runs the expert's
    # sampler with the wrapper as ``self``.
    if not causal:
        for moe_cls in (WanMoE, WanMoEI2V):
            moe = object.__new__(moe_cls)
            torch.nn.Module.__init__(moe)
            moe.network_high, moe.schedule_type, moe.guidance_boundary = network, "rf", 0.9
            with pytest.raises(GridRequested) as wrapped:
                moe.sample(noise)
            assert wrapped.value.args == (sample_steps, sample_shift)


def test_causal_vace_sampler_defaults_match_vace():
    from fastgen.networks.VaceWan.network import VACEWan
    from fastgen.networks.VaceWan.network_causal import CausalVACEWan

    def defaults(network_cls):
        parameters = inspect.signature(network_cls.sample).parameters
        return parameters["sample_steps"].default, parameters["shift"].default

    # 50 steps at the Diffusers VACE checkpoints' ``flow_shift`` of 3.0; the official scripts use shift 16.
    assert defaults(CausalVACEWan) == defaults(VACEWan) == (50, 3.0)


@pytest.mark.parametrize(
    ("latent_compression_factors", "channels", "in_features"),
    [
        # LingBot-world's pretrained camera_patch_embedding takes 1536 inputs: 6 rays * 8 * 8 pixels * (1, 2, 2) patch.
        ((4, 8, 8), 384, 1536),
        ((4, 16, 16), 1536, 6144),
    ],
)
def test_wan_cam_patch_embedding_width_follows_the_latent_compression(
    latent_compression_factors, channels, in_features
):
    from fastgen.networks.WanI2V.network_cam import WanCamI2V

    network = object.__new__(WanCamI2V)
    torch.nn.Module.__init__(network)
    network.transformer = _tiny_wan_transformer()
    network.latent_compression_factors = latent_compression_factors
    network._add_camera_layers()

    assert network.plucker_channels == channels
    assert network.transformer.camera_patch_embedding.in_features == in_features


def test_wan_cam_init_is_noop_without_deadening_the_injector_mlp():
    from fastgen.networks.WanI2V.network_cam import WanCamI2V

    torch.manual_seed(7)
    network = object.__new__(WanCamI2V)
    torch.nn.Module.__init__(network)
    network.transformer = _tiny_wan_transformer()
    network.latent_compression_factors = (4, 8, 8)
    network._add_camera_layers()

    transformer = network.transformer
    block = transformer.blocks[0]
    embedding_layers = [
        transformer.camera_patch_embedding,
        transformer.camera_hidden_layer1,
        transformer.camera_hidden_layer2,
    ]
    injector_layers = [block.cam_injector_layer1, block.cam_injector_layer2]
    output_layers = [block.cam_scale_layer, block.cam_shift_layer]

    for layer in [*embedding_layers, *injector_layers]:
        assert torch.count_nonzero(layer.weight) > 0
        assert torch.count_nonzero(layer.bias) == 0
    for layer in output_layers:
        assert torch.count_nonzero(layer.weight) == 0
        assert torch.count_nonzero(layer.bias) == 0

    camera_input = torch.randn(2, 3, transformer.camera_patch_embedding.in_features)
    hidden_states = torch.randn(2, 3, transformer.camera_patch_embedding.out_features)

    def inject_camera():
        camera_states = transformer.camera_patch_embedding(camera_input)
        camera_states = (
            transformer.camera_hidden_layer2(torch.nn.functional.silu(transformer.camera_hidden_layer1(camera_states)))
            + camera_states
        )
        cam_h = block.cam_injector_layer2(torch.nn.functional.silu(block.cam_injector_layer1(camera_states)))
        cam_h = cam_h + camera_states
        return (1.0 + block.cam_scale_layer(cam_h)) * hidden_states + block.cam_shift_layer(cam_h)

    output = inject_camera()
    assert torch.equal(output, hidden_states)
    output.square().mean().backward()
    for layer in output_layers:
        assert layer.weight.grad is not None
        assert torch.count_nonzero(layer.weight.grad) > 0

    with torch.no_grad():
        for layer in output_layers:
            for parameter in layer.parameters():
                parameter.add_(parameter.grad, alpha=-0.1)
    network.zero_grad(set_to_none=True)

    inject_camera().square().mean().backward()
    for layer in [*embedding_layers, *injector_layers]:
        assert layer.weight.grad is not None
        assert torch.count_nonzero(layer.weight.grad) > 0


def test_wan_cam_plucker_grid_has_one_cell_per_latent_position():
    from fastgen.networks.WanI2V.network_cam import _compute_plucker_cond

    poses = torch.eye(4).repeat(5, 1, 1)
    intrinsics = torch.tensor([64.0, 64.0, 32.0, 32.0]).repeat(5, 1)
    for stride, cells in (((4, 8, 8), 8), ((4, 16, 16), 4)):
        emb = _compute_plucker_cond(
            poses, intrinsics, height=64, width=64, latent_compression_factors=stride, patch_size=(1, 2, 2)
        )
        assert emb.shape == (1, 6 * stride[1] * stride[2], 2, cells, cells), stride


def test_wan_cam_forward_rejects_camera_tokens_that_do_not_match_the_latent():
    """A Plücker grid built for another VAE compression has 4x the tokens; it must fail instead of being trimmed."""
    from fastgen.networks.WanI2V.network_cam import WanCamI2V

    network = object.__new__(WanCamI2V)
    torch.nn.Module.__init__(network)
    network.transformer = _tiny_wan_transformer(seed=1)
    network.transformer.r_embedder = None
    network.transformer.encoder_depth = None
    network.transformer.time_cond_type = "abs"
    network.transformer.norm_temb = False
    network._keep_original_prec = False
    network._sdpa_no_cudnn_blocks = None
    network.override_transformer_forward()
    network.latent_compression_factors = (4, 8, 8)
    network._add_camera_layers()

    def run(camera):
        return network.transformer(
            hidden_states=torch.randn(1, 4, 2, 4, 4),
            timestep=torch.tensor([400.0]),
            encoder_hidden_states=torch.randn(1, 3, 8),
            attention_kwargs={"camera_hidden_states": camera},
        )

    with torch.no_grad():
        run(torch.randn(1, network.plucker_channels, 2, 4, 4))  # 8 camera tokens for 8 video tokens
        with pytest.raises(ValueError, match="camera conditioning has 32 tokens but the video latent has 8"):
            run(torch.randn(1, network.plucker_channels, 2, 8, 8))


def test_wan_cam_injection_promotes_into_the_fp32_residual(monkeypatch):
    """LingBot injects the camera as ``x = (1 + cam_scale) * x + cam_shift`` on the fp32 residual: the bf16 camera
    terms promote, so the residual is never rounded to the camera dtype."""
    from fastgen.networks.WanI2V import network_cam

    torch.manual_seed(0)
    dim = 8
    block = torch.nn.Module()
    block.scale_shift_table = torch.nn.Parameter(torch.zeros(1, 6, dim))
    for name in ("cam_injector_layer1", "cam_injector_layer2", "cam_scale_layer", "cam_shift_layer"):
        setattr(block, name, torch.nn.Linear(dim, dim).to(torch.bfloat16))
    camera = torch.randn(1, 4, dim, dtype=torch.bfloat16)
    residual = 100 * torch.randn(1, 4, dim)  # a bf16 rounding of these values is visible

    def run_post_self_attention(self, hidden_states, *args, post_self_attn_fn):
        return post_self_attn_fn(hidden_states)

    monkeypatch.setattr(network_cam, "wan_block_forward_impl", run_post_self_attention)
    with torch.no_grad():
        out = network_cam._wancami2v_block_forward(block, residual, torch.zeros(1, 6, dim), None, None, False, camera)
        cam_h = block.cam_injector_layer2(torch.nn.functional.silu(block.cam_injector_layer1(camera))) + camera
        expected = (1.0 + block.cam_scale_layer(cam_h)) * residual + block.cam_shift_layer(cam_h)

    assert out.dtype == expected.dtype == torch.float32
    assert torch.equal(out, expected)


@pytest.mark.parametrize("time_cond_type", ["abs", "diff"])
def test_wan_expanded_r_timestep_matches_the_per_sample_path(time_cond_type):
    """TI2V forwards expand ``t`` and ``r`` alike to ``[B, S]``; ``r`` must then flatten like ``t`` (``diff``
    used to subtract the flattened ``t`` from the unflattened ``r``) and give the per-sample result."""
    from fastgen.networks.Wan.network import Wan

    network = object.__new__(Wan)
    torch.nn.Module.__init__(network)
    network.transformer = _attach_wan_r_embedder(_tiny_wan_transformer(seed=5), time_cond_type=time_cond_type)
    network._keep_original_prec = False
    network._sdpa_no_cudnn_blocks = None
    network.override_transformer_forward()
    network.eval()

    hidden = torch.randn(2, 4, 2, 4, 4)  # 8 tokens per sample with patch size (1, 2, 2)
    text = torch.randn(2, 3, 8)
    t, r = torch.tensor([400.0, 300.0]), torch.tensor([50.0, 20.0])

    def run(timestep, r_timestep):
        return network.transformer(
            hidden_states=hidden.clone(), timestep=timestep, encoder_hidden_states=text.clone(), r_timestep=r_timestep
        )

    with torch.no_grad():
        per_sample = run(t, r)
        expanded = run(t[:, None].expand(-1, 8), r[:, None].expand(-1, 8))
    torch.testing.assert_close(expanded, per_sample, rtol=1e-5, atol=1e-5)
