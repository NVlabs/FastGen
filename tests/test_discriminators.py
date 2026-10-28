# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Discriminator subclasses and their shared feature-index validation."""

import pytest
import torch

from fastgen.configs.config_utils import override_config_with_opts
from fastgen.configs.discriminator import (
    Discriminator_EDM_CIFAR10_Config,
    Discriminator_EDM_ImageNet64_Config,
    Discriminator_SD15_Res512_Config,
    Discriminator_Wan_1_3B_Config,
)
from fastgen.networks import discriminators as disc_mod
from fastgen.networks.discriminators import (
    CompositeDiscriminator,
    Discriminator,
    Discriminator_Audio2D,
    Discriminator_EDM,
    Discriminator_ImageDiT,
    Discriminator_SD15,
    Discriminator_SDXL,
    Discriminator_VideoDiT,
)
from fastgen.configs.instantiate import LazyCall as L, instantiate
from fastgen.utils.env import set_env_vars
from tests.utils import RunIf


@pytest.fixture
def captured_warnings(monkeypatch):
    """Capture ``logger.warning`` calls emitted from the discriminators module."""
    messages: list[str] = []
    monkeypatch.setattr(disc_mod.logger, "warning", messages.append)
    return messages


# ---------------------------------------------------------------------------
# Helper-level tests: _resolve_feature_indices
# ---------------------------------------------------------------------------


def test_resolve_feature_indices_defaults_when_none(captured_warnings):
    result = Discriminator._resolve_feature_indices(None, max_index=29, default={15})
    assert result == {15}
    assert captured_warnings == []


def test_resolve_feature_indices_valid_passes_through(captured_warnings):
    result = Discriminator._resolve_feature_indices({0, 5, 29}, max_index=29, default={15})
    assert result == {0, 5, 29}
    assert captured_warnings == []


def test_resolve_feature_indices_filters_and_warns_on_invalid(captured_warnings):
    # Covers all the invalid categories at once: too-high, negative, non-int, bool.
    result = Discriminator._resolve_feature_indices({3, 30, -1, "2", True}, max_index=29, default={15})
    assert result == {3}
    assert len(captured_warnings) == 1
    msg = captured_warnings[0]
    # Dropped entries and bounds should both appear in the message.
    for token in ("30", "-1", "[0, 29]"):
        assert token in msg, f"expected {token!r} in warning, got: {msg}"


def test_resolve_feature_indices_all_invalid_raises():
    with pytest.raises(ValueError, match="empty set"):
        Discriminator._resolve_feature_indices({-1, 999}, max_index=29, default={15})


# ---------------------------------------------------------------------------
# Constructor smoke tests: each subclass wires the helper correctly and
# preserves its prior default for feature_indices=None.
# ---------------------------------------------------------------------------


def test_edm_default_uses_last_index(captured_warnings):
    d = Discriminator_EDM(feature_indices=None, all_res=[32, 16, 8], in_channels=16)
    assert d.feature_indices == {2}
    assert d.in_res == [8]
    assert len(d.discriminator_heads) == 1
    assert captured_warnings == []


@pytest.mark.parametrize("channels", [9, 11, 14, 23, 31])
def test_edm_discriminator_supports_small_channel_counts(channels):
    discriminator = Discriminator_EDM(feature_indices={0}, all_res=[8], in_channels=channels)
    discriminator._record_precision(precision_model=torch.float32)
    features = torch.randn(2, channels, 8, 8, requires_grad=True)

    logits = discriminator([features])

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


@pytest.mark.parametrize("channels", [0, -1])
def test_group_norm_rejects_nonpositive_channel_counts(channels):
    with pytest.raises(ValueError, match="num_channels must be positive"):
        disc_mod._get_optimal_groups(channels)


def test_group_norm_groups_divide_every_channel_count():
    """GroupNorm needs a divisor: up to 32 groups, a quarter of the channels below 33 (30 channels give 6, not 7)."""
    for channels in range(1, 2049):
        groups = disc_mod._get_optimal_groups(channels)
        assert 1 <= groups <= (max(1, channels // 4) if channels <= 32 else 32)
        assert channels % groups == 0, channels
    assert [disc_mod._get_optimal_groups(c) for c in (30, 384, 480, 1280)] == [6, 32, 32, 32]


def test_image_dit_default_is_middle_block(captured_warnings):
    d = Discriminator_ImageDiT(feature_indices=None, num_blocks=57, inner_dim=8)
    assert d.feature_indices == {28}
    assert d.num_features == 1
    assert len(d.cls_pred_heads) == 1
    assert captured_warnings == []


def test_video_dit_default_is_middle_block(captured_warnings):
    d = Discriminator_VideoDiT(feature_indices=None, num_blocks=30, inner_dim=16)
    assert d.feature_indices == {15}
    assert d.num_features == 1
    assert captured_warnings == []


def test_audio2d_default_is_middle_block(captured_warnings):
    d = Discriminator_Audio2D(feature_indices=None, inner_dim=8, hidden_channels=16, num_blocks=48)
    assert d.feature_indices == {24}
    assert len(d.heads) == 1
    assert captured_warnings == []


def test_subclass_filters_and_warns_via_helper(captured_warnings):
    # End-to-end check that subclasses invoke the shared helper: a stale
    # config containing both valid and invalid indices should be filtered,
    # produce exactly one warning, and build the matching number of heads.
    d = Discriminator_ImageDiT(feature_indices={0, 5, 999}, num_blocks=57, inner_dim=8)
    assert d.feature_indices == {0, 5}
    assert d.num_features == 2
    assert len(d.cls_pred_heads) == 2
    assert len(captured_warnings) == 1


def test_subclass_normalizes_collection_to_set(captured_warnings):
    d = Discriminator_ImageDiT(feature_indices=[0, 5, 5], num_blocks=57, inner_dim=8)
    assert d.feature_indices == {0, 5}
    assert isinstance(d.feature_indices, set)
    assert captured_warnings == []


# ---------------------------------------------------------------------------
# Config-driven forward-pass tests
# These exercise the discriminators end-to-end via their registered configs.
# ---------------------------------------------------------------------------


@RunIf(min_gpus=1)
def test_discriminator_wan_forward():
    """Lightweight forward-pass test for Discriminator_VideoDiT via the Wan config."""
    set_env_vars()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    discriminator_config = Discriminator_Wan_1_3B_Config

    batch_size = 1
    inner_dim = 16
    T, H, W = 8, 8, 8  # avoid kernel-size issues with the more aggressive archs

    dummy_features = [torch.randn(batch_size, inner_dim, T, H, W, device=device, dtype=dtype)]

    # Only the most memory-efficient single-head architectures.
    efficient_architectures = [
        "conv3d_down_mlp_efficient",
        "multiscale_down_mlp_efficient",
    ]

    for arch_name in efficient_architectures:
        discriminator_config = override_config_with_opts(
            discriminator_config,
            ["-", f"++disc_type={arch_name}", f"++inner_dim={inner_dim}", "++num_blocks=2", "++feature_indices=[0]"],
        )

        discriminator = instantiate(discriminator_config).to(device=device, dtype=dtype)

        with torch.no_grad():
            output = discriminator(dummy_features)
            assert output.shape == torch.Size([batch_size, 1]), f"Got {output.shape}"
            assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

        total_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        assert 10 < total_params < 50_000_000, f"Unexpected param count: {total_params}"

    # Multi-head variant.
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
    multi_head_discriminator = instantiate(discriminator_config).to(device=device, dtype=dtype)
    multi_head_features = [torch.randn(batch_size, inner_dim, T, H, W, device=device, dtype=dtype) for _ in range(2)]

    with torch.no_grad():
        output = multi_head_discriminator(multi_head_features)
        assert output.shape == torch.Size([batch_size, 2]), f"Got {output.shape}"
        assert torch.isfinite(output).all(), "Multi-head output contains NaN or Inf values"


def test_discriminator_edm_forward():
    """Lightweight forward-pass test for Discriminator_EDM (CIFAR10 + ImageNet64 configs)."""
    set_env_vars()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

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
            "feature_indices": None,  # default (last index)
        },
    ]

    batch_size = 1

    for test_case in test_configs:
        config = test_case["config"]
        resolutions = test_case["resolutions"]
        feature_indices = test_case["feature_indices"]

        test_in_channels = min(test_case["in_channels"], 128)

        config = override_config_with_opts(
            config,
            ["-", f"in_channels={test_in_channels}", f"all_res={resolutions}"],
        )

        discriminator = instantiate(config).to(device=device, dtype=dtype)

        if feature_indices is None:
            test_feature_indices = [len(resolutions) - 1]
        else:
            test_feature_indices = sorted(i for i in feature_indices if i < len(resolutions))

        dummy_features = [
            torch.randn(batch_size, test_in_channels, resolutions[idx], resolutions[idx], device=device, dtype=dtype)
            for idx in test_feature_indices
        ]

        with torch.no_grad():
            output = discriminator(dummy_features)
            expected_shape = torch.Size([batch_size, len(test_feature_indices)])
            assert (
                output.shape == expected_shape
            ), f"[{test_case['name']}] expected {expected_shape}, got {output.shape}"
            assert torch.isfinite(output).all(), f"[{test_case['name']}] NaN/Inf in output"
            assert output.abs().max() < 100, f"[{test_case['name']}] outputs too large: {output}"


def test_edm_discriminator_computes_in_precision_model_from_wider_master_weights():
    """As EDM's layers, the heads cast their weights per call, so FP32 master weights serve reduced-precision
    compute without autocast or FSDP, as in EDM's FP16 mode."""
    discriminator = Discriminator_EDM(feature_indices={0}, all_res=[8], in_channels=8)
    discriminator.prepare_module("cpu", precision_model=torch.bfloat16, precision_opt=torch.float32, trainable=True)
    features = torch.randn(2, 8, 8, 8, requires_grad=True)

    logits = discriminator([features])

    assert all(parameter.dtype == torch.float32 for parameter in discriminator.parameters())
    assert logits.dtype == torch.bfloat16
    logits.float().sum().backward()
    assert all(parameter.grad.dtype == torch.float32 for parameter in discriminator.parameters())
    assert features.grad is not None and features.grad.dtype == torch.float32


def test_edm_conv2d_takes_an_explicit_stride_and_padding():
    """The heads use EDM's ``Conv2d`` with ``torch.nn.Conv2d``'s stride and padding, which resampling excludes."""
    from fastgen.networks.EDM.network import Conv2d

    torch.manual_seed(0)
    conv = Conv2d(4, 6, kernel=4, stride=2, padding=1, init_bias=1)
    x = torch.randn(2, 4, 8, 8)

    torch.testing.assert_close(conv(x), torch.nn.functional.conv2d(x, conv.weight, conv.bias, stride=2, padding=1))
    with pytest.raises(AssertionError, match="stride and padding exclude resampling"):
        Conv2d(4, 6, kernel=3, down=True, stride=2)


def test_edm_discriminator_heads_keep_torch_conv2d_init_and_group_counts():
    """The EDM layers keep the heads' ``torch.nn.Conv2d`` initialization bounds and their group counts, which
    EDM's own rule would cap at ``channels // 4``."""
    from fastgen.networks.EDM.network import Conv2d, GroupNorm

    channels = 48
    discriminator = Discriminator_EDM(feature_indices={0}, all_res=[16], in_channels=channels)
    convs = [module for module in discriminator.modules() if isinstance(module, Conv2d)]
    norms = [module for module in discriminator.modules() if isinstance(module, GroupNorm)]

    assert len(convs) == 4 and len(norms) == 3
    for conv in convs:
        bound = conv.weight[0].numel() ** -0.5
        assert conv.weight.abs().max() <= bound and 0 < conv.bias.abs().max() <= bound
    assert disc_mod._get_optimal_groups(channels) == 24 != channels // 4
    assert all(norm.num_groups == 24 for norm in norms)


def test_sd15_discriminator_head_takes_the_selected_tap_width():
    """``Discriminator_SD15_Res512_Config`` lists the SD1.5 tap widths, so a head on the first down block takes 320
    channels while the default bottleneck head keeps 1280."""
    config = override_config_with_opts(Discriminator_SD15_Res512_Config, ["-", "feature_indices=[0]"])
    discriminator = instantiate(config)

    logits = discriminator([torch.randn(2, 320, 32, 32)])

    assert discriminator.in_channels == [320]
    assert logits.shape == (2, 1)
    assert instantiate(Discriminator_SD15_Res512_Config).in_channels == [1280]


# ---------------------------------------------------------------------------
# CompositeDiscriminator: resolver branches in __init__.
# Forward correctness is covered by tests/test_dmd2model.py::test_multimodal_*.
# ---------------------------------------------------------------------------


def _tiny_video_cfg(feature_indices=None):
    return L(Discriminator_VideoDiT)(
        num_blocks=2,
        disc_type="dit_simple_conv3d",
        inner_dim=8,
        feature_indices=feature_indices,
    )


def _tiny_audio_cfg(feature_indices=None):
    return L(Discriminator_Audio2D)(
        num_blocks=2,
        inner_dim=8,
        hidden_channels=16,
        feature_indices=feature_indices,
    )


def test_composite_agreeing_indices_no_override(captured_warnings):
    disc = CompositeDiscriminator(
        modality_discs={
            "video": _tiny_video_cfg(feature_indices={0}),
            "audio": _tiny_audio_cfg(feature_indices={0}),
        },
        feature_indices=None,
    )
    assert set(disc.modality_discs.keys()) == {"video", "audio"}
    assert disc.modality_discs["video"].feature_indices == {0}
    assert disc.modality_discs["audio"].feature_indices == {0}
    assert disc.feature_indices == {0}
    assert captured_warnings == []


def test_composite_adopts_shared_child_defaults(captured_warnings):
    disc = CompositeDiscriminator(
        modality_discs={
            "video": _tiny_video_cfg(),
            "audio": _tiny_audio_cfg(),
        }
    )
    assert disc.feature_indices == {1}
    assert captured_warnings == []


def test_composite_conflicting_indices_raises():
    with pytest.raises(ValueError, match="conflicting feature_indices"):
        CompositeDiscriminator(
            modality_discs={
                "video": _tiny_video_cfg(feature_indices={0}),
                "audio": _tiny_audio_cfg(feature_indices={1}),
            },
            feature_indices=None,
        )


def test_composite_override_warns_and_propagates(captured_warnings):
    disc = CompositeDiscriminator(
        modality_discs={
            "video": _tiny_video_cfg(feature_indices={0}),
            "audio": _tiny_audio_cfg(feature_indices={0}),
        },
        feature_indices={1},
    )
    # Both modalities should end up using the override.
    assert disc.modality_discs["video"].feature_indices == {1}
    assert disc.modality_discs["audio"].feature_indices == {1}
    assert disc.feature_indices == {1}
    # One warning per modality whose cfg differs from the override.
    assert len(captured_warnings) == 2
    assert all("overriding with CompositeDiscriminator" in m for m in captured_warnings)


# ---------------------------------------------------------------------------
# Precision boundaries: features are cast to the model dtype, logits keep their
# native dtype, and input gradients keep the caller's dtype.
# ---------------------------------------------------------------------------


def test_discriminator_casts_features_to_model_dtype_and_returns_native_logits():
    discriminator = Discriminator_Audio2D(feature_indices={0}, inner_dim=2, hidden_channels=4, num_blocks=1).double()
    discriminator._record_precision(precision_model=torch.float64)

    features = torch.randn(2, 2, 16, 8, dtype=torch.float32, requires_grad=True)
    logits = discriminator([features])

    assert logits.dtype == torch.float64
    logits.sum().backward()
    assert features.grad is not None
    assert features.grad.dtype == torch.float32


def test_discriminator_returns_native_autocast_logits():
    discriminator = Discriminator_Audio2D(feature_indices={0}, inner_dim=2, hidden_channels=4, num_blocks=1)
    discriminator._record_precision(precision_model=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = discriminator([torch.randn(2, 2, 16, 8)])

    assert logits.dtype == torch.bfloat16


def test_every_discriminator_returns_native_logits_and_preserves_input_gradients():
    cases = [
        (Discriminator_EDM(feature_indices={0}, all_res=[8], in_channels=4), (2, 4, 8, 8)),
        (Discriminator_SD15(feature_indices={0}, all_res=[8], in_channels=4), (2, 4, 8, 8)),
        (Discriminator_SDXL(feature_indices={0}, all_res=[8], in_channels=4), (2, 4, 8, 8)),
        (Discriminator_ImageDiT(feature_indices={0}, num_blocks=1, inner_dim=4), (2, 4, 8, 8)),
        (
            Discriminator_VideoDiT(feature_indices={0}, num_blocks=1, inner_dim=4, disc_type="dit_simple_conv3d"),
            (2, 4, 2, 8, 8),
        ),
        (
            Discriminator_Audio2D(feature_indices={0}, num_blocks=1, inner_dim=4, hidden_channels=4),
            (2, 4, 16, 8),
        ),
    ]

    for discriminator, shape in cases:
        discriminator = discriminator.double()
        discriminator._record_precision(precision_model=torch.float64)
        features = torch.randn(*shape, dtype=torch.float32, requires_grad=True)

        logits = discriminator([features])

        assert logits.dtype == torch.float64
        logits.sum().backward()
        assert features.grad is not None and features.grad.dtype == torch.float32


def test_composite_discriminator_returns_native_logits_and_preserves_nested_gradients():
    discriminator = CompositeDiscriminator(
        modality_discs={
            "image": L(Discriminator_ImageDiT)(feature_indices={0}, num_blocks=1, inner_dim=4),
            "audio": L(Discriminator_Audio2D)(feature_indices={0}, inner_dim=4, hidden_channels=4, num_blocks=1),
        }
    ).double()
    for module in discriminator.modules():
        if isinstance(module, Discriminator):
            module._record_precision(precision_model=torch.float64)

    image = torch.randn(2, 4, 8, 8, dtype=torch.float32, requires_grad=True)
    audio = torch.randn(2, 4, 16, 8, dtype=torch.float32, requires_grad=True)
    logits = discriminator([{"image": image, "audio": audio}])

    assert logits.shape == (2, 2) and logits.dtype == torch.float64
    logits.sum().backward()
    assert image.grad is not None and image.grad.dtype == torch.float32
    assert audio.grad is not None and audio.grad.dtype == torch.float32


def test_composite_discriminator_delegates_precision_to_its_modalities():
    """The composite owns no tensors; each modality resolves and casts its own state."""

    class _FixedDiscriminator(Discriminator):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
            self.fixed = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))

        def _fixed_precision_overrides(self):
            return {"fixed": torch.float32}

    discriminator = object.__new__(CompositeDiscriminator)
    torch.nn.Module.__init__(discriminator)
    discriminator.modality_discs = torch.nn.ModuleDict({"video": _FixedDiscriminator(), "audio": _FixedDiscriminator()})

    discriminator._set_precision(precision_model=torch.bfloat16, precision_opt=torch.float64)

    assert discriminator.precision_model == torch.bfloat16
    for child in discriminator.modality_discs.values():
        assert child.precision_model == torch.bfloat16
        assert child.weight.dtype == torch.float64
        assert child.fixed.dtype == torch.float32
