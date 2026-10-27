# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for augmentation and the explicit raw/model-space preprocessing boundary."""

from unittest.mock import MagicMock

import pytest
import torch

from fastgen.datasets.augment import AugmentPipe
from fastgen.methods.model import FastGenModel
from fastgen.networks.EDM.network import EDMPrecond
from fastgen.networks.network import FastGenNetwork, validate_conditioning
from tests.utils import RunIf


class _BaseConditioningNet:
    """Small duck-typed network exercising FastGenNetwork's base conditioning builder."""

    precision_model = None  # unprepared, like FastGenModule's class default
    encode_real = FastGenNetwork.encode_real
    _resolve_conditions = FastGenNetwork._resolve_conditions

    def __init__(self, **preprocessors):
        self.preprocessors = preprocessors

    def build_conditioning(self, data, input_shape=None, precision=None):
        return FastGenNetwork.build_conditioning(self, data, input_shape=input_shape, precision=precision)


class _PassthroughNet:
    encode_real = FastGenNetwork.encode_real

    def __init__(self, **preprocessors):
        self.preprocessors = preprocessors

    def build_conditioning(self, data, input_shape=None, precision=None):
        return {}


class _PixelNet(_PassthroughNet):
    def encode_real(self, real_raw):
        return real_raw


def _stub_model(
    net,
    input_shape=(4, 1, 1),
    dtype=torch.float32,
    device="cpu",
    state_fields=frozenset({"real"}),
):
    model = MagicMock()
    model.device = torch.device(device)
    model.precision = dtype
    model.precision_amp_enc = None
    model.net = net
    model.net_preprocessors = net
    model.input_shape = input_shape
    model.state_fields = state_fields
    model.prepare_batch = FastGenModel.prepare_batch.__get__(model)
    return model


def _run(net, data, state_fields=frozenset({"real"}), **kwargs):
    return _stub_model(net, state_fields=state_fields).prepare_batch(data, **kwargs)


def test_preprocess_reshards_preprocessors_after_conditioning():
    events = []

    class Net(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            del data, input_shape
            events.append("conditioning")
            return {"built": True}

    model = _stub_model(Net())
    model.reshard_fsdp_modules.side_effect = lambda preprocessors_only=False: events.append(
        ("reshard", preprocessors_only)
    )

    out = model.prepare_batch({})

    assert out == {"built": True}
    assert events == ["conditioning", ("reshard", True)]
    model.reshard_fsdp_modules.assert_called_once_with(preprocessors_only=True)


def test_preprocess_does_not_publish_a_success_boundary_after_conditioning_failure():
    class Net(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            del data, input_shape
            raise RuntimeError("conditioning failed")

    model = _stub_model(Net())

    with pytest.raises(RuntimeError, match="conditioning failed"):
        model.prepare_batch({})

    model.reshard_fsdp_modules.assert_not_called()


def test_validate_conditioning_allows_mapping_without_absent_keys():
    condition = {"input_ids": [torch.ones(1, dtype=torch.long)]}

    assert (
        validate_conditioning(
            {"condition": condition},
            require_mapping=True,
            absent_mapping_keys=("condition_mask",),
        )
        is None
    )


@pytest.mark.parametrize("branch", ["condition", "neg_condition"])
def test_validate_conditioning_requires_mapping(branch):
    data = {"condition": {"input_ids": [torch.ones(1, dtype=torch.long)]}}
    data[branch] = torch.ones(1)

    with pytest.raises(TypeError, match=rf"{branch}.*mapping"):
        validate_conditioning(data, require_mapping=True)


@pytest.mark.parametrize("branch", ["condition", "neg_condition"])
@pytest.mark.parametrize("value", [None, True], ids=["none", "non_none"])
def test_validate_conditioning_requires_builder_output_keys_to_be_absent(branch, value):
    data = {"condition": {"input_ids": [torch.ones(1, dtype=torch.long)]}}
    data[branch] = {"input_ids": [], "condition_mask": value}

    with pytest.raises(ValueError, match=r"condition_mask.*cannot be prepared again"):
        validate_conditioning(
            data,
            require_mapping=True,
            absent_mapping_keys=("condition_mask",),
        )


def test_validate_conditioning_rejects_conflicting_type_requirements():
    with pytest.raises(ValueError, match="both a tensor and a mapping"):
        validate_conditioning({}, require_tensor=True, require_mapping=True)


def test_validate_conditioning_requires_named_non_none_fields():
    with pytest.raises(ValueError, match=r"data\['condition'\].*data\['real_raw'\]"):
        validate_conditioning(
            {"condition": None},
            required_fields=("condition", "real_raw"),
        )


@pytest.mark.parametrize("field", ["condition", "neg_condition"])
@pytest.mark.parametrize("value", [None, "encoded"], ids=["none", "non_none"])
def test_validate_conditioning_rejects_present_absent_fields(field, value):
    with pytest.raises(ValueError, match=rf"requires absent data\['{field}'\]"):
        validate_conditioning({field: value}, absent_fields=(field,))


@RunIf(min_gpus=1)
def test_augment_pipe_updates_input_and_labels_on_input_device():
    images = torch.randn(2, 3, 8, 8, device="cuda", dtype=torch.float64)
    data = {"real_raw": images}

    out = AugmentPipe(p=1, xflip=1)(data)

    assert out is data
    assert out["real_raw"].device == images.device and out["real_raw"].dtype == images.dtype
    assert out["aug_condition"].device == images.device and out["aug_condition"].dtype == torch.float32


def test_augment_pipe_does_not_own_conditioning():
    images = torch.randn(2, 3, 8, 8)
    condition = torch.randn(2, 4)
    data = {"real_raw": images, "condition": condition}

    out = AugmentPipe(p=0)(data)

    assert out["real_raw"] is images
    assert out["condition"] is condition
    assert "neg_condition" not in out
    assert isinstance(out["aug_condition"], torch.Tensor)
    assert out["aug_condition"].shape == (2, 0)


def test_base_builder_resolves_positive_and_negative_raw_branches_once():
    encoded = [
        torch.full((1, 2), 1, dtype=torch.bfloat16),
        torch.full((1, 2), 2, dtype=torch.bfloat16),
    ]
    text_encoder = MagicMock()
    text_encoder.encode.side_effect = encoded
    net = _BaseConditioningNet(text_encoder=text_encoder)
    data = {"condition_raw": ["positive"], "neg_condition_raw": ["negative"]}

    out = net.build_conditioning(data)

    assert out == {"condition": encoded[0], "neg_condition": encoded[1]}
    assert [entry.args[0] for entry in text_encoder.encode.call_args_list] == [
        data["condition_raw"],
        data["neg_condition_raw"],
    ]
    assert set(data) == {"condition_raw", "neg_condition_raw"}


def test_base_builder_preserves_authoritative_encoded_branches_and_raw_mirrors():
    condition = torch.randn(1, 3, dtype=torch.bfloat16)
    neg_condition = torch.randn(1, 3, dtype=torch.bfloat16)
    text_encoder = MagicMock()
    net = _BaseConditioningNet(text_encoder=text_encoder)

    out = net.build_conditioning(
        {
            "condition": condition,
            "condition_raw": ["positive mirror"],
            "neg_condition": neg_condition,
            "neg_condition_raw": ["negative mirror"],
        }
    )

    assert out["condition"] is condition and out["neg_condition"] is neg_condition
    text_encoder.encode.assert_not_called()


def test_base_builder_rejects_conditioning_dtype_mismatch():
    with pytest.raises(AssertionError, match=r"neg_condition\.dtype"):
        _BaseConditioningNet().build_conditioning(
            {
                "condition": torch.zeros(1, 4, dtype=torch.bfloat16),
                "neg_condition": torch.zeros(1, 4, dtype=torch.float32),
            }
        )


def test_base_builder_rejects_conditioning_device_mismatch():
    with pytest.raises(AssertionError, match=r"neg_condition\.device"):
        _BaseConditioningNet().build_conditioning(
            {
                "condition": torch.zeros(1, 4),
                "neg_condition": torch.zeros(1, 4, device="meta"),
            }
        )


def test_base_builder_resolves_none_and_encoded_hybrid_independently():
    condition = torch.randn(1, 3, dtype=torch.bfloat16)
    encoded_negative = torch.randn(1, 3, dtype=torch.bfloat16)
    text_encoder = MagicMock()
    text_encoder.encode.return_value = encoded_negative
    net = _BaseConditioningNet(text_encoder=text_encoder)

    out = net.build_conditioning(
        {
            "condition": condition,
            "condition_raw": ["ignored"],
            "neg_condition": None,
            "neg_condition_raw": ["negative"],
        }
    )

    assert out == {"condition": condition, "neg_condition": encoded_negative}
    text_encoder.encode.assert_called_once_with(["negative"])


@pytest.mark.parametrize("data", [{}, {"condition": None}, {"neg_condition": None}])
def test_base_builder_omits_unresolved_branches(data):
    assert _BaseConditioningNet().build_conditioning(data) == {}


def test_prepare_batch_preserves_absent_vs_explicit_none_conditioning():
    assert _run(_BaseConditioningNet(), {}) == {}
    assert _run(_BaseConditioningNet(), {"condition": None}) == {"condition": None}


def test_base_builder_passes_raw_values_to_encoder_without_classification():
    raw = [["producer-owned nested value"]]
    text_encoder = MagicMock()
    text_encoder.encode.return_value = torch.ones(1)

    _BaseConditioningNet(text_encoder=text_encoder).build_conditioning({"condition_raw": raw})

    text_encoder.encode.assert_called_once_with(raw)


def test_base_builder_rejects_raw_text_without_an_encoder():
    with pytest.raises(ValueError, match="does not provide a text-conditioning implementation"):
        _BaseConditioningNet().build_conditioning({"condition_raw": ["prompt"]})


def test_prepare_batch_casts_only_declared_state_and_network_owned_conditioning():
    """Method state and conditioning use their owners' precision; unrelated fields remain unchanged."""
    condition = torch.randn(1, 3, dtype=torch.float32)
    neg_condition = torch.randn(1, 3, dtype=torch.float64)
    state = torch.randn(1, 4, 1, 1, dtype=torch.bfloat16)
    task_input = {"float": torch.randn(1, dtype=torch.bfloat16), "mask": torch.ones(1, dtype=torch.bool)}
    net = _BaseConditioningNet()
    net.precision_model = torch.bfloat16

    out = _stub_model(net, dtype=torch.float32).prepare_batch(
        {"real": state, "condition": condition, "neg_condition": neg_condition, "task": task_input},
    )

    assert out["real"].dtype == torch.float32
    assert out["task"]["float"].dtype == torch.bfloat16
    assert out["task"]["mask"].dtype == torch.bool
    assert out["condition"].dtype == torch.bfloat16
    assert out["neg_condition"].dtype == torch.bfloat16


def test_prepare_batch_preserves_fields_not_declared_as_method_state():
    poses = torch.eye(4).reshape(1, 1, 4, 4)

    out = _stub_model(_PassthroughNet(), dtype=torch.bfloat16).prepare_batch(
        {"real": torch.randn(1, 4, 1, 1), "camera_poses": poses},
    )

    assert out["real"].dtype == torch.bfloat16
    assert out["camera_poses"] is poses


def test_prepare_batch_feeds_raw_inputs_uncast_and_casts_encoded_state():
    seen = {}

    class _CaptureVAE:
        def encode(self, value):
            seen["vae_input"] = value.dtype
            return value.mean(dim=(-1, -2), keepdim=True)

    class _CaptureNet(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            seen["builder_raw"] = data["real_raw"].dtype
            seen["precision"] = precision
            return {}

    raw = torch.randn(1, 3, 8, 8)
    out = _stub_model(_CaptureNet(vae=_CaptureVAE()), dtype=torch.bfloat16).prepare_batch({"real_raw": raw})

    assert seen == {"vae_input": torch.float32, "builder_raw": torch.float32, "precision": torch.bfloat16}
    assert "real_raw" not in out
    assert out["real"].dtype == torch.bfloat16


def test_prepare_batch_aligns_cached_and_live_conditioning_dtypes():
    """A cached negative and a freshly encoded positive agree regardless of the encoder's output dtype."""
    text_encoder = MagicMock()
    text_encoder.encode.return_value = torch.zeros(1, 4, dtype=torch.float32)  # e.g. FP32 under no autocast
    net = _BaseConditioningNet(text_encoder=text_encoder)
    net.precision_model = torch.bfloat16
    cached_negative = torch.ones(1, 4, dtype=torch.float64)  # offline embedding in yet another dtype

    out = _stub_model(net, dtype=torch.float32).prepare_batch(
        {"condition_raw": ["a prompt"], "neg_condition": cached_negative},
    )

    text_encoder.encode.assert_called_once_with(["a prompt"])
    assert out["condition"].dtype == out["neg_condition"].dtype == torch.bfloat16


def test_prepare_batch_does_not_mutate_input_mapping():
    condition = torch.randn(1, 3, dtype=torch.bfloat16)
    raw_state = torch.randn(1, 3, 8, 8)
    latent = torch.randn(1, 4, 1, 1)
    vae = MagicMock()
    vae.encode.return_value = latent
    source = {"real_raw": raw_state, "condition": condition, "condition_raw": ["mirror"]}

    out = _run(_BaseConditioningNet(vae=vae), source)

    assert set(source) == {"real_raw", "condition", "condition_raw"}
    assert "real" not in source
    assert source["real_raw"] is raw_state and source["condition"] is condition
    assert out is not source and out["real"] is latent


def test_prepare_batch_materializes_multimodal_real_raw_with_vae():
    raw = {"video": torch.randn(2, 3, 8, 16, 16), "audio": torch.randn(2, 2, 800)}
    latent = {
        "video": torch.randn(2, 16, 1, 1, 1),
        "audio": torch.randn(2, 8, 2, 3),
    }
    vae = MagicMock()
    vae.encode.return_value = latent
    net = _PassthroughNet(vae=vae)

    out = _run(net, {"real_raw": raw})

    vae.encode.assert_called_once()
    assert "real_raw" not in out
    assert out["real"]["video"] is latent["video"] and out["real"]["audio"] is latent["audio"]


def test_prepare_batch_preserves_authoritative_real_with_raw_mirror():
    real = torch.randn(2, 4, 1, 1)
    real_raw = torch.randn(2, 3, 8, 8)
    vae = MagicMock()

    out = _run(_PassthroughNet(vae=vae), {"real": real, "real_raw": real_raw})

    assert out["real"] is real and "real_raw" not in out
    vae.encode.assert_not_called()


def test_prepare_batch_does_not_classify_plain_real_by_shape():
    pixel_shaped_authoritative_real = torch.randn(2, 3, 8, 8)
    vae = MagicMock()

    out = _run(_PassthroughNet(vae=vae), {"real": pixel_shaped_authoritative_real})

    assert out["real"] is pixel_shaped_authoritative_real
    assert "real_raw" not in out
    vae.encode.assert_not_called()


def test_prepare_batch_materialize_false_builds_from_reference_without_primary_encode():
    raw = torch.randn(2, 3, 8, 8)
    vae = MagicMock()
    captured = {}

    class _CaptureNet(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            captured.update(data)
            return {"condition": data["real_raw"].mean()}

    out = _run(_CaptureNet(vae=vae), {"real_raw": raw}, materialize_real=False)

    assert "real" not in out and "real" not in captured
    assert captured["real_raw"] is raw and "real_raw" not in out
    assert "condition" in out
    vae.encode.assert_not_called()


def test_prepare_batch_pixel_network_aliases_missing_real_from_real_raw():
    raw = torch.randn(2, 3, 8, 8)

    out = _run(_PixelNet(), {"real_raw": raw})

    assert out["real"] is raw and "real_raw" not in out


def test_prepare_batch_missing_primary_state_is_left_to_downstream_consumer():
    out = _run(_PassthroughNet(), {})
    assert "real" not in out and "real_raw" not in out


def test_prepare_batch_requires_vae_for_raw_latent_state_materialization():
    with pytest.raises(ValueError, match="requires a VAE"):
        _run(_PassthroughNet(), {"real_raw": torch.randn(1, 3, 8, 8)})


def test_prepare_batch_casts_native_vae_output_to_state_precision():
    vae = MagicMock()
    vae.encode.return_value = torch.ones(2, 4, 1, 1, dtype=torch.float64)

    out = _run(_PassthroughNet(vae=vae), {"real_raw": torch.randn(2, 3, 8, 8)})

    assert out["real"].dtype == torch.float32


def test_prepare_batch_rejects_vae_output_on_another_device():
    vae = MagicMock()
    vae.encode.return_value = torch.empty(2, 4, 1, 1, device="meta")

    with pytest.raises(AssertionError, match=r"vae_output\.device"):
        _run(_PassthroughNet(vae=vae), {"real_raw": torch.randn(2, 3, 8, 8)})


def test_prepare_batch_rejects_state_a_builder_returns_in_another_dtype():
    class _StateWritingNet(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            return {"real": torch.zeros(1, 4, 1, 1, dtype=torch.float64)}

    with pytest.raises(AssertionError, match=r"real\.dtype"):
        _run(_StateWritingNet(), {"real": torch.zeros(1, 4, 1, 1)})


def test_prepare_batch_rejects_state_a_builder_returns_on_another_device():
    class _StateWritingNet(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            return {"real": torch.zeros(1, 4, 1, 1, device="meta")}

    with pytest.raises(AssertionError, match=r"real\.device"):
        _run(_StateWritingNet(), {"real": torch.zeros(1, 4, 1, 1)})


def test_prepare_batch_validates_synthesized_vae_output_modality():
    raw_video = torch.randn(2, 3, 8, 16, 16)
    encoded_video = torch.randn(2, 16, 1, 1, 1)
    encoded_audio = torch.randn(2, 8, 2, 3)
    vae = MagicMock()
    vae.encode.return_value = {"video": encoded_video, "audio": encoded_audio}

    out = _run(_PassthroughNet(vae=vae), {"real_raw": {"video": raw_video}})

    assert out["real"]["video"] is encoded_video
    assert out["real"]["audio"] is encoded_audio


def test_prepare_batch_casts_synthesized_native_vae_output_to_state_precision():
    vae = MagicMock()
    vae.encode.return_value = {
        "video": torch.randn(2, 16, 1, 1, 1),
        "audio": torch.randn(2, 8, 2, 3, dtype=torch.float64),
    }

    out = _run(_PassthroughNet(vae=vae), {"real_raw": {"video": torch.randn(2, 3, 8, 16, 16)}})

    assert out["real"]["video"].dtype == out["real"]["audio"].dtype == torch.float32


def test_prepare_batch_rejects_synthesized_vae_output_on_another_device():
    vae = MagicMock()
    vae.encode.return_value = {
        "video": torch.randn(2, 16, 1, 1, 1),
        "audio": torch.randn(2, 8, 2, 3, device="meta"),
    }

    with pytest.raises(AssertionError, match=r"audio_vae_output\.device"):
        _run(_PassthroughNet(vae=vae), {"real_raw": {"video": torch.randn(2, 3, 8, 16, 16)}})


def test_prepare_batch_casts_method_declared_noise_and_path_without_vae_encoding():
    noise = torch.randn(2, 4, 1, 1, dtype=torch.bfloat16)
    path = torch.randn(2, 5, 4, 1, 1, dtype=torch.bfloat16)
    vae = MagicMock()

    out = _stub_model(
        _PassthroughNet(vae=vae),
        dtype=torch.float32,
        state_fields=frozenset({"real", "noise", "path"}),
    ).prepare_batch(
        {"noise": noise, "path": path},
    )

    assert out["noise"].dtype == torch.float32 and out["path"].dtype == torch.float32
    assert "noise_raw" not in out and "path_raw" not in out
    vae.encode.assert_not_called()


def test_prepare_batch_casts_all_method_state_before_conditioning():
    class _CaptureNet(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            assert data["real"].dtype == data["noise"].dtype == data["path"].dtype == torch.float32
            return {}

    _run(
        _CaptureNet(),
        {
            "real": torch.randn(1, 4, 1, 1, dtype=torch.float64),
            "noise": torch.randn(1, 4, 1, 1, dtype=torch.bfloat16),
            "path": torch.randn(1, 2, 4, 1, 1, dtype=torch.float16),
        },
        state_fields=frozenset({"real", "noise", "path"}),
    )


def test_prepare_batch_runs_builder_in_both_materialization_modes():
    seen = []

    class _CaptureNet(_PixelNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            seen.append(("real" in data, torch.is_inference_mode_enabled()))
            return {}

    raw = torch.randn(1, 3, 8, 8)
    _stub_model(_CaptureNet()).prepare_batch({"real_raw": raw}, materialize_real=True)
    _stub_model(_CaptureNet()).prepare_batch(
        {"real_raw": raw},
        inference_mode=True,
        materialize_real=False,
    )

    assert seen == [(True, False), (False, True)]


def test_prepare_batch_disables_outer_autocast_when_encoder_amp_is_unspecified():
    seen = []

    class _CaptureVAE:
        def encode(self, value):
            seen.append(("vae", torch.is_autocast_enabled("cpu")))
            return value

    class _CaptureNet(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            seen.append(("builder", torch.is_autocast_enabled("cpu")))
            return {}

    net = _CaptureNet(vae=_CaptureVAE())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _stub_model(net).prepare_batch({"real_raw": torch.randn(1, 3, 2, 2)})
        assert torch.is_autocast_enabled("cpu")

    assert seen == [("vae", False), ("builder", False)]


def test_prepare_batch_uses_configured_encoder_autocast_on_cpu():
    seen = []

    class _CaptureNet(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            seen.append((torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")))
            return {}

    model = _stub_model(_CaptureNet())
    model.precision_amp_enc = torch.bfloat16
    with torch.autocast("cpu", dtype=torch.float16):
        model.prepare_batch({})
        assert torch.get_autocast_dtype("cpu") == torch.float16

    assert seen == [(True, torch.bfloat16)]


@pytest.mark.parametrize(
    ("raw_key", "key"),
    [("condition_raw", "condition"), ("neg_condition_raw", "neg_condition")],
)
@pytest.mark.parametrize("return_explicit_none", [False, True], ids=["omitted", "explicit_none"])
def test_prepare_batch_preserves_network_owned_conditioning_policy(raw_key, key, return_explicit_none):
    class _ExplicitNoneNet(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            return {key: None} if return_explicit_none else {}

    out = _run(_ExplicitNoneNet(), {raw_key: ["prompt"]})

    assert out[raw_key] == ["prompt"]
    assert (key in out) is return_explicit_none
    if return_explicit_none:
        assert out[key] is None


def test_prepare_batch_allows_explicit_none_raw_conditioning():
    out = _run(_PassthroughNet(), {"condition_raw": None, "neg_condition_raw": None})

    assert out == {"condition_raw": None, "neg_condition_raw": None}


def test_prepare_batch_preserves_builder_owned_conditioning_dtypes():
    """Shared preprocessing does not override a builder's branch precision policy."""

    condition = torch.zeros(1, 4, dtype=torch.bfloat16)
    neg_condition = torch.zeros(1, 4, dtype=torch.float32)

    class _MismatchNet(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            return {
                "condition": condition,
                "neg_condition": neg_condition,
            }

    out = _run(_MismatchNet(), {"condition_raw": ["a prompt"]})

    assert out["condition"] is condition and out["condition"].dtype == torch.bfloat16
    assert out["neg_condition"] is neg_condition and out["neg_condition"].dtype == torch.float32


def test_prepare_batch_preserves_builder_owned_conditioning_devices():
    condition = torch.zeros(1, 4)
    neg_condition = torch.zeros(1, 4, device="meta")

    class _MismatchNet(_PassthroughNet):
        def build_conditioning(self, data, input_shape=None, precision=None):
            return {
                "condition": condition,
                "neg_condition": neg_condition,
            }

    out = _run(_MismatchNet(), {"condition_raw": ["a prompt"]})

    assert out["condition"] is condition and out["condition"].device.type == "cpu"
    assert out["neg_condition"] is neg_condition and out["neg_condition"].device.type == "meta"


def test_prepare_batch_accepts_different_conditioning_shapes_with_aligned_dtypes():
    positive = {
        "text_embeds": (None, torch.zeros(2, 3, dtype=torch.bfloat16)),
        "reference": {"video": torch.zeros(2, 4), "audio": torch.zeros(2, 5)},
        "attention_mask": torch.zeros(2, 3, dtype=torch.long),
    }
    negative = {
        "text_embeds": (None, torch.ones(2, 5, dtype=torch.bfloat16)),
        "reference": {"video": torch.ones(2, 6), "audio": torch.ones(2, 7)},
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }

    net = _BaseConditioningNet()
    net.precision_model = torch.bfloat16
    out = _stub_model(net).prepare_batch({"condition": positive, "neg_condition": negative})

    assert out["condition"] is not positive
    # Every floating leaf of a pre-encoded condition enters in the compute dtype; masks are untouched.
    assert out["condition"]["text_embeds"][1].dtype == torch.bfloat16
    assert out["condition"]["reference"]["video"].dtype == torch.bfloat16
    assert out["condition"]["attention_mask"].dtype == torch.long


def test_prepare_batch_augments_real_raw_before_vae_and_keeps_labels_separate():
    raw = torch.zeros(2, 3, 8, 8)
    labels = torch.ones(2, 2)
    vae = MagicMock()
    vae.encode.side_effect = lambda value: value.mean(dim=(-1, -2), keepdim=True)
    augment = MagicMock(side_effect=lambda batch: {**batch, "real_raw": batch["real_raw"] + 1, "aug_condition": labels})

    out = _run(_PassthroughNet(vae=vae), {"real_raw": raw}, augment_pipe=augment)

    augment.assert_called_once()
    torch.testing.assert_close(vae.encode.call_args.args[0], raw + 1)
    assert "real_raw" not in out
    assert out["aug_condition"] is labels
    assert "condition" not in out and "neg_condition" not in out


def test_prepare_batch_delegates_source_key_to_augment_pipe():
    real = torch.zeros(2, 3, 8, 8)
    raw_mirror = torch.full_like(real, 3)
    labels = torch.ones(2, 1)
    augment = AugmentPipe(input_key="real")
    augment.process = MagicMock(side_effect=lambda value: (value + 1, labels))

    out = _run(
        _PixelNet(),
        {"real": real, "real_raw": raw_mirror},
        augment_pipe=augment,
    )

    augment.process.assert_called_once_with(real)
    torch.testing.assert_close(out["real"], real + 1)
    assert "real_raw" not in out
    assert out["aug_condition"] is labels


@pytest.mark.parametrize("attribute", ["dtype", "device"])
def test_augment_pipe_rejects_metadata_mismatch_without_mutating_batch(attribute):
    raw = torch.zeros(2, 3, 8, 8)
    data = {"real_raw": raw}

    def process(value):
        output = value.to(dtype=torch.float64) if attribute == "dtype" else value.to(device="meta")
        return output, torch.ones(2, 1)

    augment = AugmentPipe()
    augment.process = process
    with pytest.raises(AssertionError, match=rf"augment_output\.{attribute}"):
        augment(data)

    assert data["real_raw"] is raw
    assert "aug_condition" not in data


def test_augment_pipe_rejects_real_raw_when_authoritative_real_is_present():
    real = torch.randn(2, 4, 1, 1)
    raw = torch.randn(2, 3, 8, 8)
    augment = AugmentPipe()
    augment.process = MagicMock()

    with pytest.raises(ValueError, match=r"cannot augment non-authoritative data\['real_raw'\]"):
        augment({"real": real, "real_raw": raw})

    augment.process.assert_not_called()


def test_augment_pipe_rejects_already_augmented_batch():
    raw = torch.randn(2, 3, 8, 8)
    labels = torch.ones(2, 1)
    augment = AugmentPipe()
    augment.process = MagicMock(return_value=(raw + 1, labels))
    data = {"real_raw": raw}

    assert augment(data) is data
    with pytest.raises(ValueError, match=r"non-None data\['aug_condition'\]"):
        augment(data)

    augment.process.assert_called_once_with(raw)


def test_prepare_batch_rejects_augmentation_without_the_configured_input():
    with pytest.raises(ValueError, match="requires a non-None `real_raw`"):
        _run(_PassthroughNet(), {}, augment_pipe=AugmentPipe())


def _uninitialized_edm(**preprocessors):
    edm = EDMPrecond.__new__(EDMPrecond)
    object.__setattr__(edm, "preprocessors", preprocessors)
    return edm


def test_edm_builder_packages_augmentation_with_both_label_branches():
    positive = torch.randn(2, 4)
    negative = torch.zeros(2, 4)
    labels = torch.randn(2, 3)

    out = _uninitialized_edm().build_conditioning(
        {"condition": positive, "neg_condition": negative, "aug_condition": labels}
    )

    assert out["condition"] == {"orig_condition": positive, "aug_condition": labels}
    assert out["neg_condition"] == {"orig_condition": negative, "aug_condition": labels}
    assert out["condition"]["aug_condition"] is out["neg_condition"]["aug_condition"]


def test_edm_builder_ignores_raw_mirrors_and_rejects_raw_only_branches():
    labels = torch.randn(2, 4)
    out = _uninitialized_edm().build_conditioning({"condition": labels, "condition_raw": ["mirror"]})
    assert out == {"condition": labels}

    with pytest.raises(ValueError, match=r"received raw text for \('neg_condition',\)"):
        _uninitialized_edm().build_conditioning({"condition": labels, "neg_condition_raw": ["negative"]})


@pytest.mark.parametrize("explicit_none", [False, True], ids=["missing", "explicit-none"])
def test_edm_builder_packages_unconditional_positive_without_creating_negative(explicit_none):
    labels = torch.randn(2, 3)
    data = {"aug_condition": labels}
    if explicit_none:
        data.update(condition=None, neg_condition=None)

    out = _uninitialized_edm().build_conditioning(data)

    assert out == {"condition": {"orig_condition": None, "aug_condition": labels}}


def test_edm_builder_without_augmentation_returns_plain_labels():
    condition = torch.randn(2, 4)
    out = _uninitialized_edm().build_conditioning({"condition": condition})
    assert out == {"condition": condition}


def test_edm_builder_keeps_float32_labels_exact_under_reduced_model_precision():
    labels = torch.full((2, 4), 1 / 3)
    edm = _uninitialized_edm()
    edm._precision_model = torch.bfloat16

    out = edm.build_conditioning({"condition": labels, "neg_condition": torch.zeros(2, 4, dtype=torch.float64)})

    assert out["condition"].dtype == out["neg_condition"].dtype == torch.float32
    assert torch.equal(out["condition"], labels)


@pytest.mark.parametrize("key", ["condition", "neg_condition"])
def test_edm_builder_requires_tensor_conditions(key):
    with pytest.raises(TypeError, match=rf"requires `{key}` to be a tensor"):
        _uninitialized_edm().build_conditioning({key: (torch.randn(2, 4),)})


@pytest.mark.parametrize("key", ["condition", "neg_condition"])
def test_edm_builder_rejects_reentrant_augmentation_condition(key):
    labels = torch.randn(2, 3)
    completed = {"orig_condition": torch.randn(2, 4), "aug_condition": labels}

    with pytest.raises(ValueError, match="cannot be prepared again"):
        _uninitialized_edm().build_conditioning({key: completed, "aug_condition": labels})
