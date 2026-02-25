# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import pytest

from fastgen.utils import instantiate

# Set CUDA memory configuration for better memory management
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")


from fastgen.configs.net import (
    EDM_CIFAR10_Config,
    EDM_ImageNet64_Config,
    EDM2_IN64_S_Config,
    DiT_IN256_XL_Config,
    SD15Config,
    FluxConfig,
    QwenImageConfig,
    CogVideoXConfig,
    Wan_1_3B_Config,
    CausalWan_1_3B_Config,
    VACE_Wan_1_3B_Config,
    Wan21_I2V_14B_480P_Config,
    Wan22_I2V_5B_Config,
    CausalWan22_I2V_5B_Config,
    CausalWan21_I2V_14B_480P_Config,
    CausalWan21_I2V_14B_720P_Config,
)
from fastgen.configs.discriminator import (
    Discriminator_Wan_1_3B_Config,
    Discriminator_EDM_CIFAR10_Config,
    Discriminator_EDM_ImageNet64_Config,
)
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.utils.basic_utils import clear_gpu_memory
from fastgen.utils.test_utils import RunIf
from fastgen.utils.io_utils import set_env_vars
from unittest.mock import patch, MagicMock


def _validate_basic_scheduler_properties(scheduler, device):
    """Test basic scheduler properties and structure."""
    # Test basic attributes
    assert hasattr(scheduler, "max_sigma"), "Scheduler should have max_sigma attribute"
    assert hasattr(scheduler, "min_t"), "Scheduler should have min_t attribute"
    assert hasattr(scheduler, "max_t"), "Scheduler should have max_t attribute"
    assert hasattr(scheduler, "alpha"), "Scheduler should have alpha method"
    assert hasattr(scheduler, "sigma"), "Scheduler should have sigma method"
    assert hasattr(scheduler, "sample_t"), "Scheduler should have sample_t method"

    # Validate time range
    assert scheduler.min_t < scheduler.max_t, f"min_t ({scheduler.min_t}) should be < max_t ({scheduler.max_t})"

    if scheduler.min_t is not None:
        assert scheduler.min_t >= 0, f"min_t ({scheduler.min_t}) should be non-negative"

    if scheduler.max_t is not None:
        # Allow very large max_t values (some schedules use very large values)
        assert scheduler.max_t <= 10000, f"max_t ({scheduler.max_t}) should be reasonable (<=10000)"


def _validate_time_sampling_and_functions(scheduler, device):
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

    # max_sigma should match sigma at max_t (with relaxed tolerance for floating point precision)
    max_sigma_tensor = torch.tensor(scheduler.max_sigma, device=device, dtype=sigma_max.dtype)

    # Use higher tolerance for bfloat16 due to lower precision
    rtol = 1e-2 if sigma_max.dtype == torch.bfloat16 else 1e-3
    atol = 1e-2 if sigma_max.dtype == torch.bfloat16 else 1e-3

    assert torch.allclose(
        sigma_max, max_sigma_tensor, rtol=rtol, atol=atol
    ), f"max_sigma ({scheduler.max_sigma}) should match sigma(max_t) ({sigma_max.item()}) within tolerance"

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

    # max_sigma should match max_t for EDM
    assert torch.allclose(
        torch.tensor(scheduler.max_sigma), torch.tensor(scheduler.max_t), rtol=1e-3
    ), f"EDM: max_sigma ({scheduler.max_sigma}) should equal max_t ({scheduler.max_t})"

    if scheduler.max_t is not None:
        assert scheduler.max_t <= 100.0, f"EDM: max_t should be reasonable (≤100), got {scheduler.max_t}"


def _validate_rectified_flow_schedule(scheduler, device, t, alpha_t, sigma_t):
    """Validate Rectified Flow-specific mathematical properties."""
    # Rectified Flow Schedule - Linear Interpolation Properties

    # 1. Core RF property: α(t) = 1-t, σ(t) = t, so α(t) + σ(t) = 1
    assert torch.allclose(
        alpha_t + sigma_t, torch.ones_like(alpha_t), rtol=1e-4, atol=1e-5
    ), f"RF: α(t) + σ(t) should equal 1, got α+σ = {(alpha_t + sigma_t).tolist()}"

    # 2. Linear relationships for RF
    assert torch.allclose(
        alpha_t, 1.0 - t, rtol=1e-4, atol=1e-5
    ), f"RF: α(t) should equal 1-t, got α(t)={alpha_t.tolist()}, expected={1.0 - t}"
    assert torch.allclose(
        sigma_t, t, rtol=1e-4, atol=1e-5
    ), f"RF: σ(t) should equal t, got σ(t)={sigma_t.tolist()}, expected={t.tolist()}"

    # 3. Boundary conditions (with relaxed tolerance)
    max_t_tensor = torch.tensor(scheduler.max_t, dtype=torch.float32)
    max_sigma_tensor = torch.tensor(scheduler.max_sigma, dtype=torch.float32)
    assert torch.allclose(
        max_sigma_tensor, max_t_tensor, rtol=1e-3, atol=1e-4
    ), f"RF: max_sigma should equal max_t, got max_σ={scheduler.max_sigma}, max_t={scheduler.max_t}"

    # 4. Test that at t≈0, α≈1 and σ≈0 (if min_t is close to 0)
    if scheduler.min_t <= 0.01:
        near_zero_t = torch.tensor([scheduler.min_t]).to(device)
        alpha_zero = scheduler.alpha(near_zero_t)
        sigma_zero = scheduler.sigma(near_zero_t)
        expected_alpha_zero = 1.0 - scheduler.min_t
        assert torch.allclose(alpha_zero, torch.tensor([expected_alpha_zero]).to(device), rtol=1e-2), (
            f"RF: α(t≈0) should ≈ 1-min_t, got "
            f"α({scheduler.min_t})={alpha_zero.item():.4f}, "
            f"expected {expected_alpha_zero:.4f}"
        )
        assert torch.allclose(sigma_zero, near_zero_t, rtol=1e-2), (
            f"RF: σ(t≈0) should ≈ min_t, "
            f"got σ({scheduler.min_t})={sigma_zero.item():.4f}, "
            f"expected {scheduler.min_t}"
        )

    # 5. Variance preservation: α²(t) + σ²(t) = (1-t)² + t² = 1 - 2t + 2t²
    variance_sum = alpha_t**2 + sigma_t**2
    expected_variance = 1 - 2 * t + 2 * t**2
    assert torch.allclose(variance_sum, expected_variance, rtol=1e-3, atol=1e-4), (
        f"RF: α²(t) + σ²(t) should equal 1-2t+2t², got "
        f"{variance_sum.tolist()}, expected {expected_variance.tolist()}"
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


def validate_noise_scheduler_properties(teacher, device, expected_schedule_type=None):
    """
    Comprehensive and consistent noise scheduler testing helper.

    Args:
        teacher: The instantiated network model
        device: Device to run tests on
        expected_schedule_type: Expected schedule type string (e.g., "edm", "rf", "sd")
    """
    # Basic existence check
    assert hasattr(teacher, "noise_scheduler"), "Model should have noise_scheduler attribute"

    scheduler = teacher.noise_scheduler
    assert teacher.schedule_type == expected_schedule_type
    assert scheduler.max_t is not None and scheduler.min_t is not None

    # Step 1: Validate basic properties
    _validate_basic_scheduler_properties(scheduler, device)

    # Step 2: Test time sampling and alpha/sigma functions
    t, alpha_t, sigma_t = _validate_time_sampling_and_functions(scheduler, device)

    # Step 3: Test boundary values
    alpha_min, alpha_max, sigma_min, sigma_max = _validate_boundary_values(scheduler, device)

    # Step 4: Schedule-specific validations
    if expected_schedule_type in ["edm"]:
        _validate_edm_schedule(scheduler, t, alpha_t, sigma_t, sigma_min, sigma_max)

    elif expected_schedule_type in ["rf", "rectified_flow"]:
        _validate_rectified_flow_schedule(scheduler, device, t, alpha_t, sigma_t)

    elif expected_schedule_type in ["sd", "sdxl"]:
        _validate_ddpm_based_schedule(scheduler, alpha_t, sigma_t, alpha_min, alpha_max, sigma_min, sigma_max)

    elif expected_schedule_type in ["cogvideox"]:
        _validate_ddpm_based_schedule(scheduler, alpha_t, sigma_t, alpha_min, alpha_max, sigma_min, sigma_max)

    elif expected_schedule_type in ["alphas"]:
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
        ["-", "img_resolution=2", "model_channels=32", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=False"],
    )

    teacher = instantiate(teacher_config)
    teacher = teacher.to(device=device, dtype=dtype)

    # Test noise scheduler properties
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="edm")

    batch_size = 1
    x = torch.randn(batch_size, 3, 2, 2, device=device, dtype=dtype)
    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="polynomial").to(device=device, dtype=dtype)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    # to one-hot
    labels = torch.nn.functional.one_hot(labels, num_classes=10).to(dtype=dtype)
    output = teacher(x, t, labels)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    # Test feature extraction with empty set first to avoid index issues
    output = teacher(x, t, labels, return_features_early=True, feature_indices=set())
    assert isinstance(output, list)  # confirm output is a list


def test_network_edm_imagenet64():
    teacher_config = EDM_ImageNet64_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Use valid parameters that exist in the config
    teacher_config = override_config_with_opts(
        teacher_config,
        ["-", "img_resolution=2", "model_channels=32", "channel_mult=[1]", "num_blocks=1", "r_timestep=False"],
    )

    teacher = instantiate(teacher_config)
    teacher = teacher.to(device=device, dtype=dtype)

    # Test noise scheduler properties
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="edm")

    batch_size = 1
    x = torch.randn(batch_size, 3, 2, 2, device=device, dtype=dtype)
    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="lognormal").to(device=device, dtype=dtype)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    # to one-hot
    labels = torch.nn.functional.one_hot(labels, num_classes=1000).to(dtype=dtype)

    output = teacher(x, t, labels)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    # Test feature extraction with empty set first to avoid index issues
    output = teacher(x, t, labels, return_features_early=True, feature_indices=set())
    assert isinstance(output, list)  # confirm output is a list


def test_network_edm2_in64():
    teacher_config = EDM2_IN64_S_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Use valid parameters that exist in the config
    teacher_config = override_config_with_opts(
        teacher_config, ["-", "img_resolution=2", "model_channels=32", "channel_mult=[1]", "num_blocks=1"]
    )

    teacher = instantiate(teacher_config)
    teacher = teacher.to(device=device, dtype=dtype)

    # Test noise scheduler properties
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="edm")

    batch_size = 1
    x = torch.randn(batch_size, 3, 2, 2, device=device, dtype=dtype)
    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device, dtype=dtype)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    # to one-hot
    labels = torch.nn.functional.one_hot(labels, num_classes=1000).to(dtype=dtype)
    output = teacher(x, t, labels)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    # Test with empty feature_indices - returns just the model output
    output_empty = teacher(x, t, labels, feature_indices=set())
    assert output_empty.shape == torch.Size([batch_size, 3, 2, 2])  # confirm score network output shape

    # Test with non-empty feature_indices - returns [model_output, features]
    # But since we have num_blocks=1, there might not be any features, so let's test with return_features_early
    features = teacher(x, t, labels, return_features_early=True, feature_indices=set())
    assert isinstance(features, list)  # confirm output is a list
    assert len(features) == 0  # empty feature_indices should return empty list


def test_network_dit_in256_xl():
    """
    Lightweight test that mocks the VAE to avoid downloading models.
    """
    teacher_config = DiT_IN256_XL_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dtype = torch.float16

    # Override input_size to match our test input dimensions
    teacher_config = override_config_with_opts(
        teacher_config, ["-", "input_size=2", "hidden_size=32", "depth=1", "num_heads=1"]
    )

    # Mock the VAE to avoid downloading
    with patch("diffusers.AutoencoderKL.from_pretrained") as mock_vae_from_pretrained:
        mock_vae = MagicMock()
        mock_vae.decode.return_value = torch.randn(1, 3, 16, 16, device=device, dtype=dtype)
        mock_vae_from_pretrained.return_value = mock_vae

        teacher = instantiate(teacher_config)
        teacher.vae = mock_vae

        # Ensure the model is on the correct device and dtype
        teacher = teacher.to(device=device, dtype=dtype)

        batch_size = 1
        # Use input size that matches the overridden config
        x = torch.randn(batch_size, 4, 2, 2, device=device, dtype=dtype)
        # Use scheduler to sample valid time steps with correct dtype
        t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform")
        t = t.to(device=device, dtype=dtype)  # Ensure timestep has correct dtype

        # Test noise scheduler properties comprehensively
        validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

        labels = torch.randint(0, 1000, (batch_size,), device=device)
        labels = torch.nn.functional.one_hot(labels, num_classes=1000).to(device=device, dtype=dtype)

        # Test basic forward pass
        output = teacher(x, t, labels)
        assert output.shape == x.shape
        assert output.device == x.device

        # Test with return_logvar
        output, logvar = teacher(x, t, labels, return_logvar=True)
        assert output.shape == x.shape
        assert logvar.shape == torch.Size([batch_size, 1])


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
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="sd")

    batch_size = 1
    x = torch.randn(batch_size, 4, 8, 8, device=device, dtype=dtype)  # Smaller than original but reasonable
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device)

    captions = ["a caption"]
    condition = teacher.text_encoder.encode(captions)

    # SD15 text encoder returns (embeddings, attention_mask) tuple
    assert isinstance(condition, tuple) and len(condition) == 2
    embeddings, attention_mask = condition
    embeddings = embeddings.to(device=device, dtype=dtype)
    attention_mask = attention_mask.to(device=device, dtype=dtype)
    condition = (embeddings, attention_mask)

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    teacher = teacher.to(device=device, dtype=dtype)
    # Forward pass without autocast
    output = teacher(x, t, condition=condition)

    output = teacher(x, t, condition=condition, return_features_early=True, feature_indices=set())

    assert isinstance(output, list)  # confirm output is a list


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_flux():
    """Test Flux network for text-to-image generation."""
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    teacher_config = FluxConfig

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Check available GPU memory before attempting to load Flux model
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if total_memory < 40:  # Flux model needs significant GPU memory
            pytest.skip(f"Test skipped: Flux model requires ~40GB GPU memory, but only {total_memory:.1f}GB available")

    # Try to instantiate Flux model - skip if not accessible (gated model)
    try:
        teacher = instantiate(teacher_config)
    except OSError as e:
        if "not a valid model identifier" in str(e) or "token" in str(e):
            pytest.skip(f"Test skipped: Flux model not accessible (requires HuggingFace authentication): {e}")
        raise
    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)

    # Test noise scheduler properties - Flux uses rectified flow
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    # Flux operates on latent space: [B, C, H, W] where C=16 for Flux VAE
    x = torch.randn(batch_size, 16, 8, 8, device=device, dtype=dtype)
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device, dtype=dtype)

    captions = ["a caption"]
    condition = teacher.text_encoder.encode(captions)

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

    # Test with return_logvar
    output, logvar = teacher(x, t, condition, guidance=guidance_tensor, return_logvar=True)
    assert output.shape == x.shape
    assert logvar.shape == torch.Size([batch_size, 1])

    # Test feature extraction with empty set
    output = teacher(x, t, condition, guidance=guidance_tensor, return_features_early=True, feature_indices=set())
    assert isinstance(output, list)  # confirm output is a list
    assert len(output) == 0  # empty feature_indices should return empty list

    # Test feature extraction with non-empty set (extract from first transformer block)
    output = teacher(x, t, condition, guidance=guidance_tensor, return_features_early=False, feature_indices={0})
    assert isinstance(output, list) and len(output) == 2  # [model_output, features]
    assert output[0].shape == x.shape  # model output shape
    assert isinstance(output[1], list) and len(output[1]) == 1  # one feature extracted

    # Test feature extraction with early return
    features = teacher(x, t, condition, guidance=guidance_tensor, return_features_early=True, feature_indices={0})
    assert isinstance(features, list)
    assert len(features) == 1  # one feature extracted

    # Clear memory after testing
    clear_gpu_memory()


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_qwen_image():
    """Test QwenImage network for text-to-image generation (~20B param MMDiT)."""
    clear_gpu_memory()

    set_env_vars()
    teacher_config = QwenImageConfig

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_memory < 40:
            pytest.skip(f"Test skipped: QwenImage requires ~40GB+ GPU memory, but only {total_memory:.1f}GB available")

    try:
        teacher = instantiate(teacher_config)
    except OSError as e:
        if "not a valid model identifier" in str(e) or "token" in str(e) or "does not appear" in str(e):
            pytest.skip(f"Test skipped: QwenImage model not accessible: {e}")
        raise

    teacher.transformer.transformer_blocks = teacher.transformer.transformer_blocks[:1]

    teacher.init_preprocessors()
    teacher = teacher.to(device=device)
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)

    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    x = torch.randn(batch_size, 16, 8, 8, device=device, dtype=dtype)
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device, dtype=dtype)

    captions = ["a caption"]
    condition = teacher.text_encoder.encode(captions)

    assert isinstance(condition, tuple) and len(condition) == 2
    prompt_embeds, prompt_embeds_mask = condition
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    prompt_embeds_mask = prompt_embeds_mask.to(device=device)
    condition = (prompt_embeds, prompt_embeds_mask)

    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition)

    assert output.shape == x.shape
    assert output.device == x.device

    teacher = teacher.to(device=device, dtype=dtype)

    output, logvar = teacher(x, t, condition, return_logvar=True)
    assert output.shape == x.shape
    assert logvar.shape == torch.Size([batch_size, 1])

    output = teacher(x, t, condition, return_features_early=True, feature_indices=set())
    assert isinstance(output, list)
    assert len(output) == 0

    output = teacher(x, t, condition, return_features_early=False, feature_indices={0})
    assert isinstance(output, list) and len(output) == 2
    assert output[0].shape == x.shape
    assert isinstance(output[1], list) and len(output[1]) == 1

    features = teacher(x, t, condition, return_features_early=True, feature_indices={0})
    assert isinstance(features, list)
    assert len(features) == 1
    assert features[0].shape == (batch_size, 3072, 4, 4)

    del teacher
    clear_gpu_memory()


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
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="cogvideox")

    batch_size = 1
    C, T, H, W = 16, 2, 4, 4  # Reduced from 16, 4, 8, 16

    # B, C, T, H, W
    x = torch.randn(batch_size, C, T, H, W, device=device, dtype=dtype)

    # Use scheduler to sample valid time steps (CogVideoX uses integer timesteps)
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device, dtype=dtype)

    captions = ["a caption"]
    condition = teacher.text_encoder.encode(captions)
    # Handle case where text encoder returns tuple or needs device placement
    if isinstance(condition, tuple):
        condition = condition[0]  # Take the first element if it's a tuple
    condition = condition.to(device=device, dtype=dtype)

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    teacher = teacher.to(device=device, dtype=dtype)

    # return features, without early return
    output = teacher(x, t, condition=condition, return_features_early=False, feature_indices={0})
    assert output[0].shape == x.shape
    assert isinstance(output[1], list) and len(output[1]) == 1
    for feature in output[1]:
        assert feature.shape == (batch_size, 480, T, H, W)

    # return features, with early return
    output = teacher(x, t, condition=condition, return_features_early=True, feature_indices={0})
    assert isinstance(output, list)  # confirm output is a list
    for feature in output:
        # Feature shape should match the new tensor dimensions (original model channels)
        expected_channels = 480  # Original CogVideoX channels
        assert feature.shape == (batch_size, expected_channels, T, H, W)

    # Clear memory after testing
    clear_gpu_memory()


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
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    T, H, W = 2, 4, 4  # Reduced from 4, 8, 16

    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)  # [B, C, T, H, W]
    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device, dtype=dtype)

    condition = teacher.text_encoder.encode(["a caption"])
    # Handle case where text encoder returns tuple or needs device placement
    if isinstance(condition, tuple):
        condition = condition[0]  # Take the first element if it's a tuple
    condition = condition.to(device=device, dtype=dtype)

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    teacher = teacher.to(device=device, dtype=dtype)

    # Forward pass without autocast
    output = teacher(x, t, condition=condition)

    output = teacher(x, t, condition=condition, return_features_early=True, feature_indices={0})

    assert isinstance(output, list)  # confirm output is a list

    for feature in output:
        # Use original model channels (not overridden)
        expected_channels = 384  # Original Wan channels
        assert feature.shape == (batch_size, expected_channels, T, H, W)

    # test rf schedule
    teacher_config.schedule_type = "rf"
    teacher = instantiate(teacher_config)
    teacher.init_preprocessors()
    teacher = teacher.to(device=device, dtype=dtype)

    # Test RF schedule properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    # Clear memory after testing
    clear_gpu_memory()


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_vace_wan():
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    # Check available GPU memory before loading model
    if torch.cuda.is_available():
        free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        if free_memory < 20:  # Need ~20GB free for this test
            pytest.skip(f"Test skipped: requires ~20GB free GPU memory, but only {free_memory:.1f}GB available")

    set_env_vars()
    teacher_config = VACE_Wan_1_3B_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    teacher = instantiate(teacher_config)

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    # B, C, T, H, W
    x = torch.randn(batch_size, 16, 2, 4, 4, device=device, dtype=dtype)  # Reduced from 21, 60, 104
    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="logitnormal").to(device=device, dtype=dtype)

    # Prepare text embeddings
    captions = ["a caption"]
    text_embeds = teacher.text_encoder.encode(captions).to(device=device, dtype=dtype)

    # Prepare video context for VACE conditioning with much smaller video
    # Create a dummy video for depth extraction (B, C, T, H, W) in [-1, 1]
    context_video = torch.randn(batch_size, 3, 2, 16, 16, device=device, dtype=dtype)  # Much smaller
    context_video = torch.clamp(context_video, -1, 1)  # Ensure it's in [-1, 1] range

    # Prepare VACE conditioning
    vid_context = teacher.prepare_vid_conditioning(context_video)

    # Create condition dict with both text_embeds and vid_context
    condition = {"text_embeds": text_embeds, "vid_context": vid_context}

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape  # confirm output shape is the same as input shape
    assert output.device == x.device  # confirm output device is the same as input device

    teacher = teacher.to(device=device, dtype=dtype)

    # Forward pass without autocast
    output = teacher(x, t, condition=condition)

    output = teacher(x, t, condition=condition, return_features_early=True, feature_indices={0})

    assert isinstance(output, list)  # confirm output is a list

    expected_channels = 384  # Original VACE channels (not overridden)
    for feature in output:
        assert feature.shape == (batch_size, expected_channels, 2, 4, 4)  # Adjusted dimensions

    # Clear memory after testing
    clear_gpu_memory()


@RunIf(min_gpus=1)
def test_network_discriminator_wan():
    """
    Lightweight unit test for Discriminator_Wan implementation.

    Tests core functionality with minimal memory usage.
    """
    set_env_vars()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    discriminator_config = Discriminator_Wan_1_3B_Config

    # Use larger dimensions to avoid kernel size issues
    batch_size = 1
    inner_dim = 16
    T, H, W = 8, 8, 8  # Increased from 2, 4, 4 to avoid kernel size errors

    # Create dummy features with larger spatial dimensions
    dummy_features = [torch.randn(batch_size, inner_dim, T, H, W, device=device, dtype=dtype)]

    # Test only the most memory-efficient architectures
    efficient_architectures = [
        "conv3d_down_mlp_efficient",
        "multiscale_down_mlp_efficient",
    ]

    # Test single-head discriminator with config-based approach
    for arch_name in efficient_architectures:
        # Configure the discriminator config with smaller parameters
        # Use ++ to force override existing fields
        discriminator_config = override_config_with_opts(
            discriminator_config,
            ["-", f"++disc_type={arch_name}", f"++inner_dim={inner_dim}", "++num_blocks=2", "++feature_indices=[0]"],
        )

        # Instantiate using config
        discriminator = instantiate(discriminator_config)

        # Keep on device with appropriate dtype
        discriminator = discriminator.to(device=device, dtype=dtype)

        # Test forward pass
        with torch.no_grad():
            output = discriminator(dummy_features)

            # Basic shape verification
            expected_shape = torch.Size([batch_size, 1])
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

            # Basic numerical verification
            assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

        # Basic parameter counting (should be reasonable)
        total_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        assert total_params > 10, f"Too few parameters: {total_params}"
        assert total_params < 50_000_000, f"Too many parameters: {total_params}"  # Increased limit for real models

    # Test multi-head discriminator with lightweight config
    discriminator_config = override_config_with_opts(
        discriminator_config,
        [
            "-",
            "++disc_type=factorized_down_mlp_efficient",
            f"++inner_dim={inner_dim}",
            "++num_blocks=2",
            "++feature_indices=[0,1]",
        ],
    )

    multi_head_discriminator = instantiate(discriminator_config)
    multi_head_discriminator = multi_head_discriminator.to(device=device, dtype=dtype)

    # Create features for multiple heads (2 heads) with larger dimensions
    multi_head_features = [torch.randn(batch_size, inner_dim, T, H, W, device=device, dtype=dtype) for _ in range(2)]

    with torch.no_grad():
        output = multi_head_discriminator(multi_head_features)

        # Verify output shape for multi-head
        expected_shape = torch.Size([batch_size, 2])  # Two heads output
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert torch.isfinite(output).all(), "Multi-head output contains NaN or Inf values"


def test_network_discriminator_edm():
    """
    Lightweight unit test for Discriminator_EDM implementation.

    Tests core functionality with minimal memory usage for both CIFAR10 and ImageNet64 configurations.
    """
    set_env_vars()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Test configurations
    test_configs = [
        {
            "name": "CIFAR10",
            "config": Discriminator_EDM_CIFAR10_Config,
            "resolutions": [32, 16, 8],
            "in_channels": 256,
            "feature_indices": {0, 1, 2},
        },
        {
            "name": "ImageNet64",
            "config": Discriminator_EDM_ImageNet64_Config,
            "resolutions": [64, 32, 16, 8],
            "in_channels": 768,
            "feature_indices": None,  # Will use default (last index)
        },
    ]

    batch_size = 1

    for test_case in test_configs:
        print(f"Testing EDM Discriminator {test_case['name']} configuration...")

        config = test_case["config"]
        resolutions = test_case["resolutions"]
        in_channels = test_case["in_channels"]
        feature_indices = test_case["feature_indices"]

        # Use smaller channels for testing to reduce memory usage
        test_in_channels = min(in_channels, 128)  # Reduce channel count for testing

        # Override config for lightweight testing
        config = override_config_with_opts(
            config,
            ["-", f"in_channels={test_in_channels}", f"all_res={resolutions}"],
        )

        # Instantiate discriminator
        discriminator = instantiate(config)
        discriminator = discriminator.to(device=device, dtype=dtype)

        # Determine which feature indices to test
        if feature_indices is None:
            # Default behavior: use last index
            test_feature_indices = [len(resolutions) - 1]
        else:
            # Use provided indices, but limit to valid range
            test_feature_indices = sorted([i for i in feature_indices if i < len(resolutions)])

        # Create dummy features for the expected resolutions
        # EDM discriminator expects 2D features (H, W) not 3D (T, H, W) like WAN
        dummy_features = []
        for idx in test_feature_indices:
            res = resolutions[idx]
            # Create 2D feature maps: [batch_size, channels, height, width]
            feature = torch.randn(batch_size, test_in_channels, res, res, device=device, dtype=dtype)
            dummy_features.append(feature)

        # Test forward pass
        with torch.no_grad():
            output = discriminator(dummy_features)

            # Verify output shape
            expected_num_heads = len(test_feature_indices)
            expected_shape = torch.Size([batch_size, expected_num_heads])
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

            # Basic numerical verification
            assert torch.isfinite(output).all(), f"Output contains NaN or Inf values for {test_case['name']}"

            # Output should be reasonable discriminator logits
            assert output.abs().max() < 100, f"Output values seem too large for {test_case['name']}: {output}"


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_causal_wan():
    """
    Test CausalWan network, specifically the sample method.
    """
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    # Check available GPU memory before loading model
    if torch.cuda.is_available():
        free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        if free_memory < 20:  # Need ~20GB free for this test
            pytest.skip(f"Test skipped: requires ~20GB free GPU memory, but only {free_memory:.1f}GB available")

    set_env_vars()
    teacher_config = CausalWan_1_3B_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Instantiate CausalWan and prepare
    teacher = instantiate(teacher_config)
    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)

    # RF schedule for WAN
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    C, T, H, W = 16, 3, 4, 4  # Use T divisible by chunk_size=3

    # B, C, T, H, W
    x = torch.randn(batch_size, C, T, H, W, device=device, dtype=dtype)

    # CausalWan supports 2D timesteps with shape (batch_size, num_frames)
    t_1d = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device, dtype=dtype)
    t = t_1d.unsqueeze(1).expand(batch_size, T)  # Shape: (batch_size, T)

    captions = ["a test caption for causal WAN"]
    condition = teacher.text_encoder.encode(captions)
    if isinstance(condition, tuple):
        condition = condition[0]
    condition = condition.to(device=device, dtype=dtype)

    # Negative condition for classifier-free guidance path
    neg_condition = teacher.text_encoder.encode([""])
    if isinstance(neg_condition, tuple):
        neg_condition = neg_condition[0]
    neg_condition = neg_condition.to(device=device, dtype=dtype)

    # Do an autocasted forward pass
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = teacher(x, t, condition=condition)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    assert output.device == x.device, f"Expected device {x.device}, got {output.device}"

    teacher = teacher.to(device=device, dtype=dtype)
    # Standard forward pass without autocast
    output = teacher(x, t, condition=condition)

    # Forward with store_kv=True (needed for autoregressive sampling caches)
    output_with_kv = teacher(x, t, condition=condition, store_kv=True)
    assert output_with_kv.shape == x.shape, f"Expected shape {x.shape}, got {output_with_kv.shape}"

    # Test the sample method
    original_noise = torch.randn(batch_size, C, T, H, W, device=device, dtype=dtype)
    with torch.no_grad():
        ar_output = teacher.sample(
            noise=original_noise,
            condition=condition,
            neg_condition=neg_condition,
        )

    assert ar_output.shape == original_noise.shape, f"Expected shape {original_noise.shape}, got {ar_output.shape}"
    assert ar_output.device == original_noise.device, f"Expected device {original_noise.device}, got {ar_output.device}"
    assert ar_output.dtype == original_noise.dtype, f"Expected dtype {original_noise.dtype}, got {ar_output.dtype}"

    # Inhomogeneous timestep sampling and forward process
    t_inhom, idx = teacher.noise_scheduler.sample_t_inhom(batch_size, T, teacher.chunk_size, sample_steps=4)
    t_inhom = t_inhom.to(device=device, dtype=dtype)
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
    single_frame_latents = torch.randn(batch_size, C, 1, H, W, device=device, dtype=dtype)
    with torch.no_grad():
        single_frame_output = teacher.sample(
            noise=single_frame_latents,
            condition=condition,
            neg_condition=neg_condition,
        )
    assert single_frame_output.shape == single_frame_latents.shape

    # Frames with remainder when divided by chunk_size (5 frames, 3-per-chunk => remainder 2)
    odd_frames_latents = torch.randn(batch_size, C, 5, H, W, device=device, dtype=dtype)
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
    teacher_config = Wan22_I2V_5B_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Wan config doesn't support model_channels/num_blocks override, use default
    teacher = instantiate(teacher_config)

    # only use a single block for testing
    teacher.transformer.blocks = teacher.transformer.blocks[:1]

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    num_frames, height, width = 5, 32, 32
    (
        T,
        H,
        W,
    ) = (num_frames + 3) // 4, height // 16, width // 16

    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=dtype)  # [B, C, T, H, W]

    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device, dtype=dtype)

    # compute text encoder hidden states
    text_embeds = teacher.text_encoder.encode(["a caption"])

    # compute input for I2V models
    image = torch.zeros(batch_size, 3, height, width, device=device, dtype=dtype)  # [B, C, H, W]
    image = image.unsqueeze(2)
    first_frame_cond = image
    first_frame_cond = first_frame_cond.to(device=device, dtype=dtype)
    first_frame_cond = teacher.vae.encode(first_frame_cond)

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

    # test rf schedule
    teacher_config.schedule_type = "rf"
    teacher = instantiate(teacher_config)
    teacher.init_preprocessors()
    teacher = teacher.to(device=device, dtype=dtype)

    # Test RF schedule properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    # Explicitly delete the 5B model to free GPU memory immediately
    del teacher

    # Clear memory after memory-intensive test
    clear_gpu_memory()


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_wan21_14b_i2v():
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    set_env_vars()
    teacher_config = Wan21_I2V_14B_480P_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Check available GPU memory before attempting to load 14B model
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if total_memory < 79:  # 14B model needs ~80GB
            pytest.skip(f"Test skipped: 14B model requires ~80GB GPU memory, but only {total_memory:.1f}GB available")

    # Wan config doesn't support model_channels/num_blocks override, use default
    teacher = instantiate(teacher_config)

    # only use a single block for testing
    teacher.transformer.blocks = teacher.transformer.blocks[:1]

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)
    teacher.image_encoder.to(device=device, dtype=dtype)

    # Test noise scheduler properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    num_frames, height, width = 5, 32, 32
    (
        T,
        H,
        W,
    ) = (num_frames + 3) // 4, height // 8, width // 8

    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)  # [B, C, T, H, W]

    # Use scheduler to sample valid time steps
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device, dtype=dtype)

    # compute text encoder hidden states
    text_embeds = teacher.text_encoder.encode(["a caption"])

    # compute image encoder hidden states
    image = torch.zeros(batch_size, 3, height, width, device=device, dtype=dtype)  # [B, C, H, W]
    encoder_hidden_states_image = teacher.image_encoder.encode(image)

    # compute input for I2V models
    image = image.unsqueeze(2)
    first_frame_cond = image
    # wan 2.1 pads the zero tensor after the first frame
    first_frame_cond = torch.cat(
        [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
    )
    first_frame_cond = first_frame_cond.to(device=device, dtype=dtype)
    first_frame_cond = teacher.vae.encode(first_frame_cond)

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

    # test rf schedule
    teacher_config.schedule_type = "rf"
    teacher = instantiate(teacher_config)
    teacher.init_preprocessors()
    teacher = teacher.to(device=device, dtype=dtype)

    # Test RF schedule properties comprehensively
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

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
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)

    # Test noise scheduler properties
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 16, width // 16

    x = torch.randn(batch_size, 48, T, H, W, device=device, dtype=dtype)  # [B, C, T, H, W]

    # CausalWanI2V supports 2D timesteps with shape (batch_size, num_frames)
    t_1d = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device, dtype=dtype)
    t = t_1d.unsqueeze(1).expand(batch_size, T)  # Shape: (batch_size, T)

    # Compute text encoder hidden states
    text_embeds = teacher.text_encoder.encode(["a caption"])

    # Compute input for I2V models (Wan 2.2 uses first frame replacement, not mask concat)
    image = torch.zeros(batch_size, 3, height, width, device=device, dtype=dtype)  # [B, C, H, W]
    image = image.unsqueeze(2)
    first_frame_cond = image.to(device=device, dtype=dtype)
    first_frame_cond = teacher.vae.encode(first_frame_cond)

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

    # Forward with store_kv=True (needed for autoregressive sampling caches)
    output_with_kv = teacher(x, t, condition, store_kv=True)
    assert output_with_kv.shape == x.shape, f"Expected shape {x.shape}, got {output_with_kv.shape}"

    # Clear caches before sample
    teacher.clear_caches()

    # Prepare negative condition for classifier-free guidance
    neg_text_embeds = teacher.text_encoder.encode([""])
    if isinstance(neg_text_embeds, tuple):
        neg_text_embeds = neg_text_embeds[0]
    neg_text_embeds = neg_text_embeds.to(device=device, dtype=dtype)
    neg_condition = dict(
        text_embeds=neg_text_embeds,
        first_frame_cond=first_frame_cond,
    )

    # Test sample method
    original_noise = torch.randn(batch_size, 48, T, H, W, device=device, dtype=dtype)
    with torch.no_grad():
        ar_output = teacher.sample(
            noise=original_noise,
            condition=condition,
            neg_condition=neg_condition,
            sample_steps=2,  # Use fewer steps for testing
        )

    assert ar_output.shape == original_noise.shape, f"Expected shape {original_noise.shape}, got {ar_output.shape}"
    assert ar_output.device == original_noise.device
    assert ar_output.dtype == original_noise.dtype

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


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_causal_wan21_14b_480p_i2v():
    """
    Test CausalWanI2V network with Wan 2.1 I2V 14B 480P model.
    Tests forward pass, sample, and causal-specific features.
    """
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    set_env_vars()
    teacher_config = CausalWan21_I2V_14B_480P_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Check available GPU memory before attempting to load 14B model
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if total_memory < 79:  # 14B model needs ~80GB
            pytest.skip(f"Test skipped: 14B model requires ~80GB GPU memory, but only {total_memory:.1f}GB available")

    teacher = instantiate(teacher_config)

    # Only use a single block for testing
    teacher.transformer.blocks = teacher.transformer.blocks[:1]

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)
    teacher.image_encoder.to(device=device, dtype=dtype)

    # Test noise scheduler properties
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 8, width // 8

    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)  # [B, C, T, H, W]

    # CausalWanI2V supports 2D timesteps with shape (batch_size, num_frames)
    t_1d = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device, dtype=dtype)
    t = t_1d.unsqueeze(1).expand(batch_size, T)  # Shape: (batch_size, T)

    # Compute text encoder hidden states
    text_embeds = teacher.text_encoder.encode(["a caption"])

    # Compute image encoder hidden states (required for Wan 2.1 14B models)
    image = torch.zeros(batch_size, 3, height, width, device=device, dtype=dtype)  # [B, C, H, W]
    encoder_hidden_states_image = teacher.image_encoder.encode(image)

    # Compute input for I2V models (Wan 2.1 uses mask concatenation)
    image = image.unsqueeze(2)
    # Wan 2.1 pads zero tensor after the first frame
    first_frame_cond = torch.cat(
        [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
    )
    first_frame_cond = first_frame_cond.to(device=device, dtype=dtype)
    first_frame_cond = teacher.vae.encode(first_frame_cond)

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
    output = teacher(x, t, condition)

    # Forward with store_kv=True (needed for autoregressive sampling caches)
    output_with_kv = teacher(x, t, condition, store_kv=True)
    assert output_with_kv.shape == x.shape, f"Expected shape {x.shape}, got {output_with_kv.shape}"

    # Clear caches before sample
    teacher.clear_caches()

    # Prepare negative condition for classifier-free guidance
    neg_text_embeds = teacher.text_encoder.encode([""])
    if isinstance(neg_text_embeds, tuple):
        neg_text_embeds = neg_text_embeds[0]
    neg_text_embeds = neg_text_embeds.to(device=device, dtype=dtype)
    neg_condition = dict(
        text_embeds=neg_text_embeds,
        first_frame_cond=first_frame_cond,
        encoder_hidden_states_image=encoder_hidden_states_image,
    )

    # Test sample method
    original_noise = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    with torch.no_grad():
        ar_output = teacher.sample(
            noise=original_noise,
            condition=condition,
            neg_condition=neg_condition,
            sample_steps=2,  # Use fewer steps for testing
        )

    assert ar_output.shape == original_noise.shape, f"Expected shape {original_noise.shape}, got {ar_output.shape}"
    assert ar_output.device == original_noise.device
    assert ar_output.dtype == original_noise.dtype

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


@RunIf(min_gpus=1)
@pytest.mark.large_model
def test_network_causal_wan21_14b_720p_i2v():
    """
    Test CausalWanI2V network with Wan 2.1 I2V 14B 720P model.
    Tests forward pass, sample, and causal-specific features.
    """
    # Clear memory before starting memory-intensive test
    clear_gpu_memory()

    set_env_vars()
    teacher_config = CausalWan21_I2V_14B_720P_Config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Check available GPU memory before attempting to load 14B model
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if total_memory < 79:  # 14B model needs ~80GB
            pytest.skip(f"Test skipped: 14B model requires ~80GB GPU memory, but only {total_memory:.1f}GB available")

    teacher = instantiate(teacher_config)

    # Only use a single block for testing
    teacher.transformer.blocks = teacher.transformer.blocks[:1]

    teacher.init_preprocessors()
    # No dtype cast yet, so we can test autocast first
    teacher = teacher.to(device=device)
    teacher.vae.to(device=device, dtype=dtype)
    teacher.text_encoder.to(device=device, dtype=dtype)
    teacher.image_encoder.to(device=device, dtype=dtype)

    # Test noise scheduler properties
    validate_noise_scheduler_properties(teacher, device, expected_schedule_type="rf")

    batch_size = 1
    num_frames, height, width = 5, 32, 32
    T, H, W = (num_frames + 3) // 4, height // 8, width // 8

    x = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)  # [B, C, T, H, W]

    # CausalWanI2V supports 2D timesteps with shape (batch_size, num_frames)
    t_1d = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform").to(device=device, dtype=dtype)
    t = t_1d.unsqueeze(1).expand(batch_size, T)  # Shape: (batch_size, T)

    # Compute text encoder hidden states
    text_embeds = teacher.text_encoder.encode(["a caption"])

    # Compute image encoder hidden states (required for Wan 2.1 14B models)
    image = torch.zeros(batch_size, 3, height, width, device=device, dtype=dtype)  # [B, C, H, W]
    encoder_hidden_states_image = teacher.image_encoder.encode(image)

    # Compute input for I2V models (Wan 2.1 uses mask concatenation)
    image = image.unsqueeze(2)
    # Wan 2.1 pads zero tensor after the first frame
    first_frame_cond = torch.cat(
        [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
    )
    first_frame_cond = first_frame_cond.to(device=device, dtype=dtype)
    first_frame_cond = teacher.vae.encode(first_frame_cond)

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
    output = teacher(x, t, condition)

    # Forward with store_kv=True (needed for autoregressive sampling caches)
    output_with_kv = teacher(x, t, condition, store_kv=True)
    assert output_with_kv.shape == x.shape, f"Expected shape {x.shape}, got {output_with_kv.shape}"

    # Clear caches before sample
    teacher.clear_caches()

    # Prepare negative condition for classifier-free guidance
    neg_text_embeds = teacher.text_encoder.encode([""])
    if isinstance(neg_text_embeds, tuple):
        neg_text_embeds = neg_text_embeds[0]
    neg_text_embeds = neg_text_embeds.to(device=device, dtype=dtype)
    neg_condition = dict(
        text_embeds=neg_text_embeds,
        first_frame_cond=first_frame_cond,
        encoder_hidden_states_image=encoder_hidden_states_image,
    )

    # Test sample method
    original_noise = torch.randn(batch_size, 16, T, H, W, device=device, dtype=dtype)
    with torch.no_grad():
        ar_output = teacher.sample(
            noise=original_noise,
            condition=condition,
            neg_condition=neg_condition,
            sample_steps=2,  # Use fewer steps for testing
        )

    assert ar_output.shape == original_noise.shape, f"Expected shape {original_noise.shape}, got {ar_output.shape}"
    assert ar_output.device == original_noise.device
    assert ar_output.dtype == original_noise.dtype

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
