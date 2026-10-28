# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-network ``build_conditioning``.

Each network's ``build_conditioning`` is exercised against a synthetic
``data`` dict and a stubbed VAE / image-encoder via ``__new__`` (no
weights downloaded) to verify that it returns the network-specific condition
updates its forward pass expects, including required, omitted, and explicit
``None`` branches.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from fastgen.configs.net import LTX2I2VConfig
from fastgen.networks.LTX2.network import LTX2
from fastgen.networks.VaceWan import network as vace_network_module
from fastgen.networks.VaceWan.modules import vace_depth_annotator as vace_depth_module
from fastgen.networks.VaceWan.network import VACEWan
from fastgen.networks.WanI2V.network import WanI2V
from fastgen.networks.cosmos_predict2.network import CosmosPredict2
from fastgen.networks.network import validate_latent_compression
from fastgen.configs.instantiate import instantiate
from fastgen.utils.tensor import resolve_device
from tests.utils import RunIf


def _assert_tree_dtype_and_device(value, floating_dtype, device, path="output"):
    """Assert device for every tensor leaf and dtype for floating leaves."""
    expected_device = resolve_device(device)
    if torch.is_tensor(value):
        assert value.device == expected_device, f"Expected {path} device {expected_device}, got {value.device}"
        if value.is_floating_point():
            assert value.dtype == floating_dtype, f"Expected {path} dtype {floating_dtype}, got {value.dtype}"
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_tree_dtype_and_device(
                child,
                floating_dtype=floating_dtype,
                device=expected_device,
                path=f"{path}.{key}",
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_tree_dtype_and_device(
                child,
                floating_dtype=floating_dtype,
                device=expected_device,
                path=f"{path}[{index}]",
            )


@pytest.fixture(
    params=(
        pytest.param(("cpu", torch.float64), id="cpu-float64"),
        pytest.param(("cuda", torch.float64), marks=RunIf(min_gpus=1), id="cuda-float64"),
    )
)
def conditioning_placement(request):
    device, dtype = request.param
    return torch.device(device), dtype


# ═════════════════════════════════════════════════════════════════════════
# Network builders (bypass __init__ → no weights, no pretrained downloads)
# ═════════════════════════════════════════════════════════════════════════


def _make_wan_i2v(concat_mask: bool, with_image_encoder: bool, t_factor: int = 4):
    """Build a WanI2V instance bypassing __init__ so no weights are loaded."""
    net = WanI2V.__new__(WanI2V)
    net.preprocessors = {}
    net.concat_mask = concat_mask
    net.latent_compression_factors = (t_factor, 8, 8)

    vae = MagicMock()
    vae.latent_compression_factors = net.latent_compression_factors
    net.preprocessors["vae"] = vae

    if with_image_encoder:
        net.preprocessors["image_encoder"] = MagicMock()

    return net


def _make_ltx2i2v():
    net = LTX2.__new__(LTX2)
    net.preprocessors = {}
    net.is_i2v = True
    # Trainable connectors read text in ``precision_model``, so the unprepared builder leaves its dtype alone.
    net.train_connectors = True
    vae = MagicMock()
    vae.latent_compression_factors = (8, 32, 32)
    net.preprocessors["vae"] = vae
    return net


def _make_vacewan(vid_context: torch.Tensor):
    net = VACEWan.__new__(VACEWan)
    net.preprocessors = {}
    net._build_vid_context = MagicMock(return_value=vid_context)
    return net


def _make_cosmos(is_video2world: bool, num_conditioning_frames: int = 1):
    net = CosmosPredict2.__new__(CosmosPredict2)
    net.preprocessors = {}
    net.is_video2world = is_video2world
    net.num_conditioning_frames = num_conditioning_frames
    vae = MagicMock()
    vae.latent_compression_factors = (4, 8, 8)
    net.preprocessors["vae"] = vae
    return net


# ═════════════════════════════════════════════════════════════════════════
# WanI2V.build_conditioning
# ═════════════════════════════════════════════════════════════════════════


class TestWanI2VBuildConditioning:
    def test_first_frame_encoding_requires_the_vae(self):
        net = _make_wan_i2v(concat_mask=True, with_image_encoder=False, t_factor=4)
        net.preprocessors.pop("vae")
        data = {"condition": torch.randn(2, 77, 4096), "real_raw": torch.randn(2, 3, 8, 16, 16)}
        with pytest.raises(ValueError, match=r"requires the 'vae' preprocessor"):
            net.build_conditioning(data, input_shape=[16, 5, 2, 2])

    def test_vae_of_another_checkpoint_is_rejected(self):
        net = _make_wan_i2v(concat_mask=False, with_image_encoder=False)
        vae = net.preprocessors["vae"]
        validate_latent_compression(net, vae)
        vae.latent_compression_factors = (4, 16, 16)
        with pytest.raises(ValueError, match=r"\(4, 8, 8\), but its VAE compresses by \(4, 16, 16\)"):
            validate_latent_compression(net, vae)

    def test_concat_mask_path_pads_to_pixel_length_and_argmax_encodes(self, conditioning_placement):
        """``concat_mask=True`` (Wan-2.1): pad first pixel frame to pixel-T then VAE-encode."""
        device, dtype = conditioning_placement
        net = _make_wan_i2v(concat_mask=True, with_image_encoder=False, t_factor=4)

        bsz, c, h, w = 2, 3, 16, 16
        # input_shape advertises 5 latent frames → (5-1)*4 + 1 = 17 pixel frames.
        input_shape = [16, 5, 2, 2]
        raw_video = torch.randn(
            bsz, c, 8, h, w, device=device, dtype=dtype
        )  # T=8 (any source T works; only first frame is used)
        encoded_first = torch.randn(bsz, 16, 5, 2, 2, device=device, dtype=dtype)

        captured = {}

        def fake_encode(x, mode=None):
            captured["mode"] = mode
            captured["pixel_T"] = x.shape[2]
            captured["channels"] = x.shape[1]
            return encoded_first

        net.preprocessors["vae"].encode.side_effect = fake_encode

        text = torch.randn(bsz, 77, 768, device=device, dtype=dtype)
        neg_text = torch.randn(bsz, 77, 768, device=device, dtype=dtype)
        data = {"real_raw": raw_video, "condition": text, "neg_condition": neg_text}

        out = net.build_conditioning(data, input_shape=input_shape)
        cond, neg_cond = out["condition"], out["neg_condition"]

        assert captured["mode"] == "argmax"
        assert captured["pixel_T"] == 17, "padded T should be (latent_T - 1) * t_factor + 1"
        assert captured["channels"] == c
        assert cond == {"text_embeds": text, "first_frame_cond": encoded_first}
        assert neg_cond == {"text_embeds": neg_text, "first_frame_cond": encoded_first}
        _assert_tree_dtype_and_device(out, floating_dtype=dtype, device=device)

    def test_no_concat_mask_skips_padding(self):
        """``concat_mask=False`` (Wan-2.2 5B): VAE-encode the bare first frame, no padding."""
        net = _make_wan_i2v(concat_mask=False, with_image_encoder=False)

        bsz, c, h, w = 1, 3, 8, 8
        raw_video = torch.randn(bsz, c, 4, h, w)
        encoded_first = torch.randn(bsz, 16, 1, 1, 1)

        captured_T: list = []

        def fake_encode(x, mode=None):
            captured_T.append(x.shape[2])
            assert mode == "argmax"
            return encoded_first

        net.preprocessors["vae"].encode.side_effect = fake_encode

        text = torch.randn(bsz, 77, 768)
        data = {"real_raw": raw_video, "condition": text, "neg_condition": text}

        cond = net.build_conditioning(data, input_shape=None)["condition"]  # input_shape unused when no concat_mask

        assert captured_T == [1], "no padding → only the first frame is encoded"
        assert torch.equal(cond["first_frame_cond"], encoded_first)

    def test_image_encoder_path_adds_clip_embedding(self):
        """An image-encoder registry entry also CLIP-embeds the raw 4-D first frame."""
        net = _make_wan_i2v(concat_mask=False, with_image_encoder=True)

        bsz, c, h, w = 1, 3, 16, 16
        raw_video = torch.randn(bsz, c, 4, h, w)
        encoded_first = torch.randn(bsz, 16, 1, 2, 2)
        clip_embed = torch.randn(bsz, 257, 1280)

        net.preprocessors["vae"].encode.return_value = encoded_first
        net.preprocessors["image_encoder"].encode.return_value = clip_embed

        text = torch.randn(bsz, 77, 768)
        data = {"real_raw": raw_video, "condition": text, "neg_condition": text}

        out = net.build_conditioning(data, input_shape=None)
        cond, neg_cond = out["condition"], out["neg_condition"]

        called_with = net.preprocessors["image_encoder"].encode.call_args[0][0]
        assert called_with.shape == (bsz, c, h, w)
        assert torch.equal(called_with, raw_video[:, :, 0])

        assert torch.equal(cond["encoder_hidden_states_image"], clip_embed)
        assert torch.equal(neg_cond["encoder_hidden_states_image"], clip_embed)

    def test_missing_real_raw_raises(self):
        """Per the new contract, missing real_raw is a hard error (no silent latent-zeros path)."""
        net = _make_wan_i2v(concat_mask=False, with_image_encoder=False)
        text = torch.randn(1, 77, 768)
        with pytest.raises(ValueError, match=r"real_raw"):
            net.build_conditioning({"condition": text, "neg_condition": text}, input_shape=[16, 1, 1, 1])

    def test_dict_condition_raises(self):
        """Complete rich conditions cannot be passed back through the builder."""
        net = _make_wan_i2v(concat_mask=False, with_image_encoder=False)
        with pytest.raises(ValueError, match=r"cannot be prepared again"):
            net.build_conditioning(
                {"condition": {"video": torch.randn(1, 77, 768)}, "real_raw": torch.randn(1, 3, 4, 8, 8)},
                input_shape=None,
            )

    def test_concat_mask_without_input_shape_raises(self):
        """concat_mask path needs input_shape to derive pixel-T."""
        net = _make_wan_i2v(concat_mask=True, with_image_encoder=False)
        text = torch.randn(1, 77, 768)
        data = {"real_raw": torch.randn(1, 3, 4, 8, 8), "condition": text, "neg_condition": text}
        with pytest.raises(ValueError, match=r"input_shape"):
            net.build_conditioning(data, input_shape=None)

    def test_neg_condition_can_be_missing(self):
        """neg_condition is optional (e.g., when CFG is off) and remains absent."""
        net = _make_wan_i2v(concat_mask=False, with_image_encoder=False)
        net.preprocessors["vae"].encode.return_value = torch.randn(1, 16, 1, 1, 1)

        text = torch.randn(1, 77, 768)
        data = {"real_raw": torch.randn(1, 3, 4, 8, 8), "condition": text}

        out = net.build_conditioning(data, input_shape=None)
        assert out["condition"]["text_embeds"] is text
        assert "neg_condition" not in out


# ═════════════════════════════════════════════════════════════════════════
# LTX2I2V.build_conditioning
# ═════════════════════════════════════════════════════════════════════════


class TestLTX2I2VBuildConditioning:
    @pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    @pytest.mark.parametrize("strength_input", ["scalar", "tensor"])
    def test_strength_is_prepared_once_with_the_frame_latent(self, state_dtype, strength_input, conditioning_placement):
        device, _ = conditioning_placement
        device = resolve_device(device)
        net = _make_ltx2i2v()
        encoded_first = torch.randn(1, 2, 1, 2, 2, device=device, dtype=torch.float64)
        net.preprocessors["vae"].encode_video.return_value = encoded_first
        text = (torch.randn(1, 2, 3, device=device), torch.ones(1, 2, device=device, dtype=torch.long))
        raw_strength = 0.4 if strength_input == "scalar" else torch.tensor([0.4], dtype=torch.float64)
        data = {
            "real_raw": torch.randn(1, 3, 4, 8, 8, device=device),
            "condition": text,
            "neg_condition": text,
            "strength": raw_strength,
        }

        out = net.build_conditioning(data, precision=state_dtype)
        condition, negative = out["condition"], out["neg_condition"]
        clean, strength = condition["frame_cond"], condition["strength"]
        assert clean.dtype == state_dtype and strength.dtype == torch.float32
        assert clean.device == strength.device == device
        assert negative["frame_cond"] is clean and negative["strength"] is strength
        expected_strength = torch.tensor([0.4], dtype=torch.float32, device=device)
        torch.testing.assert_close(strength, expected_strength, rtol=0, atol=0)
        torch.testing.assert_close(clean, encoded_first.to(state_dtype), rtol=0, atol=0)

        _, unpacked_clean, weight = net._unpack_i2v_condition(condition)
        assert unpacked_clean is clean and weight.dtype == torch.float32 and weight.device == device
        torch.testing.assert_close(weight.flatten(), 1 - expected_strength, rtol=0, atol=0)
        assert data["strength"] is raw_strength
        if strength_input == "tensor":
            assert raw_strength.dtype == torch.float64 and raw_strength.device.type == "cpu"
            torch.testing.assert_close(raw_strength, torch.tensor([0.4], dtype=torch.float64), rtol=0, atol=0)
        net.preprocessors["vae"].encode_video.assert_called_once()

    def test_a_strength_per_sample_keeps_its_batch_axis(self):
        net = _make_ltx2i2v()
        net.preprocessors["vae"].encode_video.return_value = torch.randn(2, 2, 1, 2, 2)
        text = (torch.randn(2, 2, 3), torch.ones(2, 2, dtype=torch.long))
        data = {"real_raw": torch.randn(2, 3, 4, 8, 8), "condition": text, "strength": torch.tensor([0.4, 1.0])}
        condition = net.build_conditioning(data, precision=torch.float32)["condition"]
        torch.testing.assert_close(condition["strength"], torch.tensor([[0.4], [1.0]]), rtol=0, atol=0)
        assert net._unpack_i2v_condition(condition)[2].shape == (2, 1, 1, 1, 1)

    def test_raw_text_branches_are_encoded_once_before_enrichment(self):
        """The LTX builder encodes each raw text branch once and shares one frame encode."""
        net = _make_ltx2i2v()
        encoded_first = torch.randn(1, 16, 1, 1, 1, dtype=torch.bfloat16)
        net.preprocessors["vae"].encode_video.return_value = encoded_first

        positive = (torch.randn(1, 3, 4, dtype=torch.float64), torch.ones(1, 3, dtype=torch.long))
        negative = (torch.randn(1, 3, 4, dtype=torch.float64), torch.ones(1, 3, dtype=torch.long))
        text_encoder = MagicMock()
        text_encoder.encode.side_effect = [positive, negative]
        net.preprocessors["text_encoder"] = text_encoder
        data = {
            "real_raw": torch.randn(1, 3, 4, 8, 8),
            "condition_raw": ["positive"],
            "neg_condition_raw": ["negative"],
        }

        out = net.build_conditioning(data)

        assert [call.args[0] for call in text_encoder.encode.call_args_list] == [["positive"], ["negative"]]
        assert out["condition"] == {"text_embeds": positive, "frame_cond": encoded_first}
        assert out["neg_condition"] == {"text_embeds": negative, "frame_cond": encoded_first}
        assert out["condition"]["text_embeds"][0].dtype == torch.float64
        assert out["condition"]["frame_cond"].dtype == torch.bfloat16
        assert set(data) == {"real_raw", "condition_raw", "neg_condition_raw"}

    def test_t2v_encodes_raw_text_once(self):
        net = LTX2.__new__(LTX2)
        net.preprocessors = {"text_encoder": MagicMock()}
        net.is_i2v = False
        net.train_connectors = True
        encoded = (torch.randn(1, 3, 4), torch.ones(1, 3, dtype=torch.long))
        net.preprocessors["text_encoder"].encode.return_value = encoded

        out = net.build_conditioning({"condition_raw": ["prompt"]})

        assert out == {"condition": encoded}
        net.preprocessors["text_encoder"].encode.assert_called_once_with(["prompt"])

    def test_frozen_connectors_take_text_in_bfloat16(self):
        """Frozen connectors keep their bf16 checkpoint weights, so text is cast to bf16 even before preparation."""
        net = LTX2.__new__(LTX2)
        net.preprocessors = {}
        net.is_i2v = False
        net.train_connectors = False
        text = (torch.randn(1, 3, 4, dtype=torch.float64), torch.ones(1, 3, dtype=torch.long))

        prompt_embeds, attention_mask = net.build_conditioning({"condition": text})["condition"]

        assert prompt_embeds.dtype == torch.bfloat16
        assert attention_mask is text[1]

    def test_frozen_connectors_take_text_without_a_model_precision_detour(self):
        """Values beyond FP16's range reach the BF16 connectors intact under an FP16 transformer."""
        net = LTX2.__new__(LTX2)
        net.preprocessors = {}
        net.is_i2v = False
        net.train_connectors = False
        net._precision_model = torch.float16
        text = (torch.full((1, 3, 4), 1e5), torch.ones(1, 3, dtype=torch.long))

        prompt_embeds, _ = net.build_conditioning({"condition": text})["condition"]

        assert torch.equal(prompt_embeds, text[0].to(torch.bfloat16))

    def test_t2v_rejects_reentrant_rich_condition(self):
        net = LTX2.__new__(LTX2)
        net.preprocessors = {}
        net.is_i2v = False
        net.train_connectors = True

        with pytest.raises(ValueError, match="cannot be prepared again"):
            net.build_conditioning(
                {"condition": {"text_embeds": (torch.randn(1, 3, 4), torch.ones(1, 3)), "frame_cond": None}}
            )

    def test_first_frame_is_argmax_encoded_no_padding(self, conditioning_placement):
        """LTX2I2V: extract first pixel frame, VAE-encode in argmax mode (no concat-mask padding)."""
        device, dtype = conditioning_placement
        net = _make_ltx2i2v()

        bsz, c, h, w = 1, 3, 64, 64
        raw_video = torch.randn(bsz, c, 8, h, w, device=device, dtype=dtype)
        encoded_first = torch.randn(bsz, 16, 1, 2, 2, device=device, dtype=dtype)

        captured = {}

        def fake_encode(x, mode=None):
            captured["mode"] = mode
            captured["pixel_T"] = x.shape[2]
            return encoded_first

        net.preprocessors["vae"].encode_video.side_effect = fake_encode

        text = (
            torch.randn(bsz, 77, 768, device=device, dtype=dtype),
            torch.ones(bsz, 77, device=device, dtype=torch.long),
        )
        neg_text = (
            torch.randn(bsz, 77, 768, device=device, dtype=dtype),
            torch.ones(bsz, 77, device=device, dtype=torch.long),
        )
        data = {"real_raw": raw_video, "condition": text, "neg_condition": neg_text}

        out = net.build_conditioning(data, input_shape=None)
        cond, neg_cond = out["condition"], out["neg_condition"]

        assert captured["mode"] == "argmax"
        assert captured["pixel_T"] == 1, "no padding — only the first frame"
        assert cond == {"text_embeds": text, "frame_cond": encoded_first}
        assert neg_cond == {"text_embeds": neg_text, "frame_cond": encoded_first}
        _assert_tree_dtype_and_device(out, floating_dtype=dtype, device=device)

    def test_multimodal_real_raw_picks_video(self):
        """real_raw can be a {'video': ..., 'audio': ...} dict — only video is used."""
        net = _make_ltx2i2v()

        bsz, c, h, w = 1, 3, 32, 32
        video = torch.randn(bsz, c, 4, h, w)
        audio = torch.randn(bsz, 1, 1024)
        encoded_first = torch.randn(bsz, 16, 1, 1, 1)

        captured = {}

        def fake_encode(x, mode=None):
            captured["x"] = x
            return encoded_first

        net.preprocessors["vae"].encode_video.side_effect = fake_encode

        text = (torch.randn(bsz, 77, 768), torch.ones(bsz, 77))
        data = {
            "real_raw": {"video": video, "audio": audio},
            "condition": text,
            "neg_condition": text,
        }

        cond = net.build_conditioning(data, input_shape=None)["condition"]
        assert torch.equal(captured["x"], video[:, :, :1])
        assert torch.equal(cond["frame_cond"], encoded_first)

    def test_missing_real_raw_raises(self):
        net = _make_ltx2i2v()
        text = (torch.randn(1, 77, 768), torch.ones(1, 77))
        with pytest.raises(ValueError, match=r"real_raw"):
            net.build_conditioning({"condition": text, "neg_condition": text}, input_shape=None)

    def test_missing_positive_condition_raises_clearly(self):
        net = _make_ltx2i2v()
        with pytest.raises(ValueError, match=r"data\['condition'\]"):
            net.build_conditioning({"real_raw": torch.randn(1, 3, 4, 8, 8)}, input_shape=None)

    def test_dict_condition_raises(self):
        net = _make_ltx2i2v()
        with pytest.raises(ValueError, match=r"cannot be prepared again"):
            net.build_conditioning(
                {"condition": {"video": torch.randn(1, 77, 768)}, "real_raw": torch.randn(1, 3, 4, 8, 8)},
                input_shape=None,
            )


# ═══════════════════════════════════════════════════════════════
# Flux2.build_conditioning
# ═════════════════════════════════════════════════════════════


class TestFlux2BuildConditioning:
    @staticmethod
    def _make(image_edit):
        from fastgen.networks.Flux2.network import Flux2

        net = Flux2.__new__(Flux2)
        net.preprocessors = {"text_encoder": MagicMock(), "vae": MagicMock()}
        net.is_image_edit = image_edit
        return net

    def test_t2i_delegates_raw_text_resolution_to_base(self):
        net = self._make(image_edit=False)
        encoded = (None, torch.randn(1, 3, 4, dtype=torch.float64))
        net.preprocessors["text_encoder"].encode.return_value = encoded

        out = net.build_conditioning({"condition_raw": ["prompt"]})

        assert out == {"condition": encoded}
        assert out["condition"][1].dtype == torch.float64
        net.preprocessors["text_encoder"].encode.assert_called_once_with(["prompt"])

    def test_t2i_rejects_reentrant_rich_condition(self):
        net = self._make(image_edit=False)
        with pytest.raises(ValueError, match="cannot be prepared again"):
            net.build_conditioning(
                {"condition": {"text_embeds": (None, torch.randn(1, 3, 4)), "condition_image": None}}
            )

    def test_image_edit_encodes_raw_image_then_enriches_both_text_branches(self):
        net = self._make(image_edit=True)
        positive = (None, torch.randn(1, 3, 4, dtype=torch.float64))
        negative = (None, torch.randn(1, 3, 4, dtype=torch.float64))
        net.preprocessors["text_encoder"].encode.side_effect = [positive, negative]
        reference = torch.randn(1, 3, 16, 16)
        encoded_reference = torch.randn(1, 4, 2, 2, dtype=torch.bfloat16)
        net.preprocessors["vae"].encode.return_value = encoded_reference
        data = {
            "condition_raw": ["positive"],
            "neg_condition_raw": ["negative"],
            "condition_image_raw": reference,
        }

        out = net.build_conditioning(data, input_shape=[4, 2, 2])

        assert [call.args[0] for call in net.preprocessors["text_encoder"].encode.call_args_list] == [
            ["positive"],
            ["negative"],
        ]
        net.preprocessors["vae"].encode.assert_called_once_with(reference, mode="argmax")
        assert out["condition"] == {"text_embeds": positive, "condition_image": encoded_reference}
        assert out["neg_condition"] == {"text_embeds": negative, "condition_image": encoded_reference}
        assert out["condition"]["text_embeds"][1].dtype == torch.float64
        assert out["condition"]["condition_image"].dtype == torch.bfloat16
        assert set(data) == {"condition_raw", "neg_condition_raw", "condition_image_raw"}

    def test_image_edit_uses_precomputed_condition_image_without_vae(self):
        net = self._make(image_edit=True)
        text = (None, torch.randn(1, 3, 4))
        reference_latent = torch.randn(1, 4, 2, 2)

        out = net.build_conditioning(
            {"condition": text, "condition_image": reference_latent},
            input_shape=[4, 2, 2],
        )

        assert out == {"condition": {"text_embeds": text, "condition_image": reference_latent}}
        net.preprocessors["vae"].encode.assert_not_called()

    def test_image_edit_precomputed_condition_image_takes_precedence_over_raw(self):
        net = self._make(image_edit=True)
        text = (None, torch.randn(1, 3, 4))
        reference_latent = torch.randn(1, 4, 2, 2)

        out = net.build_conditioning(
            {
                "condition": text,
                "condition_image": reference_latent,
                "condition_image_raw": torch.randn(1, 3, 16, 16),
            },
            input_shape=[4, 2, 2],
        )

        assert out["condition"]["condition_image"] is reference_latent
        net.preprocessors["vae"].encode.assert_not_called()

    def test_image_edit_missing_condition_image_raises_clearly(self):
        net = self._make(image_edit=True)

        with pytest.raises(ValueError) as exc_info:
            net.build_conditioning({"condition": (None, torch.randn(1, 3, 4))}, input_shape=[4, 2, 2])

        message = str(exc_info.value)
        assert "condition_image_raw" in message
        assert "condition_image" in message.replace("condition_image_raw", "")

    def test_image_edit_rejects_reentrant_rich_condition(self):
        net = self._make(image_edit=True)
        with pytest.raises(ValueError, match="cannot be prepared again"):
            net.build_conditioning(
                {
                    "condition": {"text_embeds": (None, torch.randn(1, 3, 4))},
                    "condition_image": torch.randn(1, 4, 2, 2),
                },
                input_shape=[4, 2, 2],
            )

    def test_image_edit_missing_positive_condition_raises_clearly(self):
        net = self._make(image_edit=True)
        with pytest.raises(ValueError, match=r"data\['condition'\]"):
            net.build_conditioning(
                {"condition_image": torch.randn(1, 4, 2, 2)},
                input_shape=[4, 2, 2],
            )


# ═════════════════════════════════════════════════════════════════════════
# LTX2 I2V — end-to-end cond-override stub
# ═════════════════════════════════════════════════════════════════════════
# Instantiates ``LTX2I2VConfig`` with ``load_pretrained=False`` and patches
# the transformer config to ``num_layers=0`` — the "stub transformer" is
# just the input/output projections, cheap enough to fit in CPU memory.
# Per-frame override math is independent of the transformer's specific
# output, so a 0-layer stub is sufficient to verify the per-pred-type wiring
# end-to-end through the real ``forward()``.  Skipped when the LTX-2 config
# is not accessible (weights aren't downloaded).  Full multimodal-dict
# (``{"video": ..., "audio": ...}``) input-shape coverage lives in the
# real-weights ``test_network_ltx2_i2v`` test.


@pytest.mark.parametrize("modalities", [("video",), ("video", "audio")])
def test_ltx2_i2v_multi_frame_per_frame_strength_stub(modalities):
    """CPU stub for **multi-frame conditioning with per-frame strength**.

    Exercises the per-frame ``frame_idx``/``strength`` generalization of the I2V
    output conditioning: two cond frames at non-adjacent latent indices, with mixed
    strength (one fully conditioned at ``s=1``, one partial at ``s=0.5``).

    Asserts:

    1. **Override at s=1 cond frames (to 1e-3)** (Case A and Case B Test 1):
       the network's cond-frame output equals
       ``convert(xt[cond], cond_latent[slot], t, "x0", pred_type)`` for every
       ``pred_type``, proving the per-frame fancy-indexing path
       (``denoise_weight[:, frame_idx, :] = inv_strength.flatten(1).unsqueeze(-1)``,
       ``model_input_video[:, :, frame_idx] = blend(...)``,
       ``video[:, :, frame_idx] = fwd_cond``) handles non-adjacent indices.
    2. **Partial blend at s=0.5** (Case B Test 2): the output equals the
       documented blend formula ``convert(xt, 0.5*x0_intermediate +
       0.5*cond_latent, t, "x0", pred_type)``, where ``x0_intermediate`` is
       reconstructed by replicating ``X0Model.to_denoised`` on the
       transformer's pre-convert ``raw_flow`` (captured via a forward hook
       on ``proj_out``).  Parametrized over ``preserve_cond_input ∈ (True,
       False)`` — the training default (``True``, input blend applied) and
       the sampling setting (``False``, raw ``video_latents`` fed into the
       transformer); the formula's ``x0_intermediate`` and the captured
       ``raw_flow`` both depend on this flag at s<1.  Positive equality,
       not "not-equal" sanity.
    3. **Non-cond frame isolation** (Case B Test 3): I2V output equals T2V
       output exactly at non-cond frames.  Valid only because ``num_layers=0``
       means no attention mixes cond and non-cond tokens (with real attention
       the cond tokens' different per-token timestep and blended input would
       propagate into non-cond outputs).
    """
    from diffusers.models.transformers import LTX2VideoTransformer3DModel

    from fastgen.networks.noise_schedule import BaseNoiseSchedule, CompositeNoiseSchedule

    _orig_load_config = LTX2VideoTransformer3DModel.load_config.__func__

    def _stub_load_config(cls, *args, **kwargs):
        config = _orig_load_config(cls, *args, **kwargs)
        config["num_layers"] = 0
        return config

    try:
        with patch.object(LTX2VideoTransformer3DModel, "load_config", classmethod(_stub_load_config)):
            teacher = instantiate(LTX2I2VConfig, modalities=modalities, load_pretrained=False)
    except OSError as e:
        if "not a valid model identifier" in str(e) or "token" in str(e) or "does not appear" in str(e):
            pytest.skip(f"Test skipped: LTX-2 config not accessible: {e}")
        raise

    # Exercise the real lifecycle contract: frozen connectors retain BF16 storage, while
    # their outputs are converted to the prepared transformer's FP32 input dtype.
    teacher.prepare_module(device="cpu", precision_model=torch.float32)

    is_multimodal = "audio" in modalities
    if is_multimodal:
        assert isinstance(teacher.noise_scheduler, CompositeNoiseSchedule)
    else:
        assert isinstance(teacher.noise_scheduler, BaseNoiseSchedule)

    batch_size = 1
    F, H, W = 4, 4, 4  # F=4 so cond frames at indices [0, 2] are non-adjacent
    C = teacher.transformer.config.in_channels

    torch.manual_seed(0)
    x_video = torch.randn(batch_size, C, F, H, W)
    t = teacher.noise_scheduler.sample_t(batch_size, time_dist_type="uniform")

    text_embed_dim = next(m.in_features for m in teacher.connectors.modules() if isinstance(m, torch.nn.Linear))
    # See note in ``test_ltx2_i2v_conditioning_blend_stub``: ``seq_len`` must
    # be a multiple of the connector's ``num_learnable_registers``.
    seq_len = next(
        int(m.num_learnable_registers)
        for m in teacher.connectors.modules()
        if getattr(m, "num_learnable_registers", None) is not None
    )
    prompt_embeds = torch.randn(batch_size, seq_len, text_embed_dim, dtype=teacher.connector_precision)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    text_embeds = (prompt_embeds, attention_mask)

    # Two cond frames at non-adjacent indices, N=2 matches len(frame_idx).
    frame_idx = torch.tensor([0, 2], dtype=torch.long)
    cond_latent = torch.randn(batch_size, C, frame_idx.numel(), H, W)

    video_ns = teacher.noise_scheduler.subschedules["video"] if is_multimodal else teacher.noise_scheduler

    def _forward(condition, fwd_pred_type):
        """Run forward and return the raw video output (in ``fwd_pred_type`` space)."""
        x_t = {"video": x_video} if is_multimodal else x_video
        out = teacher(x_t, t, condition=condition, fwd_pred_type=fwd_pred_type)
        return out["video"] if isinstance(out, dict) else out

    def _expected_cond(slot, fwd_pred_type):
        """The override result the network must produce at the cond frame
        ``frame_idx[slot]``: ``convert(xt[idx], cond_latent[slot], t, "x0", pred_type)``."""
        idx = frame_idx[slot].item()
        return video_ns.convert_model_output(
            x_video[:, :, idx : idx + 1],
            cond_latent[:, :, slot : slot + 1],
            t,
            src_pred_type="x0",
            target_pred_type=fwd_pred_type,
        )

    # --- Case A: multi-frame, all strength=1 → both cond frames match the
    # override result (direct equality, no round-trip).
    cond_unified = {
        "text_embeds": text_embeds,
        "frame_cond": cond_latent,
        "frame_idx": frame_idx,
        "strength": torch.ones(frame_idx.numel(), dtype=cond_latent.dtype),
    }
    for pred_type in ("x0", "eps", "flow"):
        out_i2v = _forward(cond_unified, pred_type)
        for slot, idx in enumerate(frame_idx.tolist()):
            assert torch.allclose(out_i2v[:, :, idx : idx + 1], _expected_cond(slot, pred_type), atol=1e-3), (
                f"modalities={modalities}, pred_type={pred_type}: "
                f"strength=1 cond at frame {idx} must equal convert(xt, cond_latent[{slot}], t, x0, {pred_type})"
            )

    # --- Case B: per-frame strength [1.0, 0.5] ---
    # The s=1 frame still matches the pure-override result (independent of
    # ``preserve_cond_input``: the ``(1-s)=0`` factor zeroes out
    # ``x0_intermediate``, leaving ``x0_blend = cond_latent``).  The s=0.5
    # frame is positively asserted against the partial-blend formula —
    # parametrized over ``preserve_cond_input`` because at s<1 both the
    # transformer's ``raw_flow`` (different ``model_input_video`` at the cond
    # frame) and the override's ``x0_intermediate`` depend on the flag.
    # Non-cond frames must equal a pure T2V (no override) forward (also
    # independent of ``preserve_cond_input``; only valid here because
    # ``num_layers=0`` means no attention mixes cond/non-cond tokens).
    from fastgen.networks.LTX2.network import _unpack_latents

    cond_mixed_base = {
        "text_embeds": text_embeds,
        "frame_cond": cond_latent,
        "frame_idx": frame_idx,
        "strength": torch.tensor([1.0, 0.5], dtype=cond_latent.dtype),
    }
    # Pre-compute T2V outputs (independent of ``preserve_cond_input``).
    out_t2v_per_pred = {pt: _forward(text_embeds, pt) for pt in ("x0", "eps", "flow")}

    captured: list = [None]

    def _capture_proj_out(_module, _inp, output):
        captured[0] = output.detach().clone()

    for preserve_cond_input in (True, False):
        cond_mixed = {**cond_mixed_base, "preserve_cond_input": preserve_cond_input}

        # Capture the transformer's pre-convert ``raw_flow`` for this
        # ``preserve_cond_input`` setting via a forward hook on ``proj_out``.
        # ``raw_flow`` is the same across all ``fwd_pred_type`` values (the
        # transformer doesn't see ``fwd_pred_type`` — that only affects the
        # post-convert), so one capture per ``preserve_cond_input`` suffices.
        captured[0] = None
        handle = teacher.transformer.proj_out.register_forward_hook(_capture_proj_out)
        try:
            _ = _forward(cond_mixed, "x0")
        finally:
            handle.remove()
        raw_flow = _unpack_latents(
            captured[0],
            F,
            H,
            W,
            teacher.transformer_spatial_patch_size,
            teacher.transformer_temporal_patch_size,
        )

        # Replicate the output-conditioning math at the s=0.5 slot
        # (slot=1, ``frame_idx[1]=2``).  ``inv_strength = 1 - 0.5 = 0.5``;
        # ``mi_cond_s05`` is what the transformer actually saw at the cond
        # frame (input-blend on ``True``, unmodified ``video_latents`` on
        # ``False``); ``t_cond`` is the per-frame timestep
        # ``X0Model.to_denoised`` uses.
        s05_inv = (1.0 - cond_mixed["strength"][1:2]).view(1, 1, 1, 1, 1)
        if preserve_cond_input:
            mi_cond_s05 = 0.5 * x_video[:, :, 2:3] + 0.5 * cond_latent[:, :, 1:2]
        else:
            mi_cond_s05 = x_video[:, :, 2:3]
        t_cond_s05 = video_ns.safe_clamp(t.view(-1, 1, 1, 1, 1) * s05_inv, min=video_ns.min_t, max=video_ns.max_t)
        x0_inter_s05 = video_ns.convert_model_output(
            mi_cond_s05, raw_flow[:, :, 2:3], t_cond_s05, src_pred_type="flow", target_pred_type="x0"
        )
        x0_blend_s05 = s05_inv * x0_inter_s05 + (1.0 - s05_inv) * cond_latent[:, :, 1:2]

        for pred_type in ("x0", "eps", "flow"):
            out_i2v = _forward(cond_mixed, pred_type)
            out_t2v = out_t2v_per_pred[pred_type]

            # 1. s=1 frame: bit-exact (the (1-s)=0 factor zeroes out vel and
            # mixes cond_latent in fully — independent of preserve_cond_input).
            assert torch.allclose(out_i2v[:, :, 0:1], _expected_cond(0, pred_type), atol=1e-3), (
                f"modalities={modalities}, preserve_cond_input={preserve_cond_input}, "
                f"pred_type={pred_type}: strength=1 cond at frame 0 must equal "
                f"convert(xt, cond_latent[0], t, x0, {pred_type})"
            )

            # 2. s=0.5 frame: matches the partial-blend formula
            # ``fwd_cond = convert(xt[cond], 0.5*x0_intermediate + 0.5*cond_latent, t, "x0", pred_type)``
            # where ``x0_intermediate`` comes from ``X0Model.to_denoised``
            # on the transformer's actual ``raw_flow`` at the cond frame
            # (and ``mi_cond_s05`` follows ``preserve_cond_input``).
            expected_s05 = video_ns.convert_model_output(
                x_video[:, :, 2:3], x0_blend_s05, t, src_pred_type="x0", target_pred_type=pred_type
            )
            assert torch.allclose(out_i2v[:, :, 2:3], expected_s05, atol=1e-3), (
                f"modalities={modalities}, preserve_cond_input={preserve_cond_input}, "
                f"pred_type={pred_type}: strength=0.5 cond at frame 2 must equal the partial-blend "
                f"formula (0.5*x0_intermediate + 0.5*cond_latent[1] → convert → {pred_type})"
            )

            # 3. Non-cond frames (1, 3): I2V output equals T2V output exactly,
            # proving the per-frame override scatter does not leak into
            # non-cond slots and that ``preserve_cond_input`` only affects
            # the cond-frame transformer input (with no attention, that
            # doesn't propagate to non-cond positions).
            for non_cond_idx in (1, 3):
                assert torch.allclose(out_i2v[:, :, non_cond_idx], out_t2v[:, :, non_cond_idx], atol=1e-4), (
                    f"modalities={modalities}, preserve_cond_input={preserve_cond_input}, "
                    f"pred_type={pred_type}: non-cond frame {non_cond_idx} must equal T2V output "
                    f"(override must NOT touch it)"
                )


# ═════════════════════════════════════════════════════════════════════════
# VACEWan.build_conditioning
# ═════════════════════════════════════════════════════════════════════════


class TestVACEWanBuildConditioning:
    def test_raw_text_branches_are_resolved_before_context_enrichment(self):
        vid_ctx = torch.randn(1, 96, 4, 8, 8)
        net = _make_vacewan(vid_context=vid_ctx)
        positive = torch.randn(1, 77, 768)
        negative = torch.randn(1, 77, 768)
        text_encoder = MagicMock()
        text_encoder.encode.side_effect = [positive, negative]
        net.preprocessors["text_encoder"] = text_encoder
        data = {
            "depth_latent": torch.randn(1, 16, 4, 8, 8),
            "condition_raw": ["positive"],
            "neg_condition_raw": ["negative"],
        }

        out = net.build_conditioning(data)

        assert [call.args[0] for call in text_encoder.encode.call_args_list] == [["positive"], ["negative"]]
        assert out["condition"]["text_embeds"] is positive
        assert out["neg_condition"]["text_embeds"] is negative
        assert out["condition"]["vid_context"] is out["neg_condition"]["vid_context"] is vid_ctx
        assert set(data) == {"depth_latent", "condition_raw", "neg_condition_raw"}

    def test_delegates_to_build_vid_context(self, conditioning_placement):
        """build_conditioning forwards real_raw (and optional depth_latent) to _build_vid_context."""
        device, dtype = conditioning_placement
        vid_ctx = torch.randn(1, 96, 4, 8, 8, device=device, dtype=dtype)
        net = _make_vacewan(vid_context=vid_ctx)

        raw_video = torch.randn(1, 3, 16, 64, 64, device=device, dtype=dtype)
        text = torch.randn(1, 77, 768, device=device, dtype=dtype)
        neg_text = torch.randn(1, 77, 768, device=device, dtype=dtype)
        data = {"real_raw": raw_video, "condition": text, "neg_condition": neg_text}

        out = net.build_conditioning(data, input_shape=None)
        cond, neg_cond = out["condition"], out["neg_condition"]

        net._build_vid_context.assert_called_once_with(video=raw_video, condition_latents=None)
        assert cond == {"text_embeds": text, "vid_context": vid_ctx}
        assert neg_cond == {"text_embeds": neg_text, "vid_context": vid_ctx}
        _assert_tree_dtype_and_device(out, floating_dtype=dtype, device=device)

    def test_forwards_depth_latent_shortcut(self):
        """Pixels remain available in case another rank needs depth extraction."""
        vid_ctx = torch.randn(1, 96, 4, 8, 8)
        net = _make_vacewan(vid_context=vid_ctx)

        raw_video = torch.randn(1, 3, 16, 64, 64)
        depth_latent = torch.randn(1, 16, 4, 8, 8)
        text = torch.randn(1, 77, 768)
        data = {
            "real_raw": raw_video,
            "depth_latent": depth_latent,
            "condition": text,
            "neg_condition": text,
        }

        net.build_conditioning(data, input_shape=None)
        net._build_vid_context.assert_called_once_with(video=raw_video, condition_latents=depth_latent)

    def test_missing_real_raw_raises(self):
        # The check lives in ``_build_vid_context`` (which decides the path collectively), so run it for real.
        net = VACEWan.__new__(VACEWan)
        torch.nn.Module.__init__(net)
        net.preprocessors = {}
        net._all_zero_latent = None
        text = torch.randn(1, 77, 768)
        with pytest.raises(ValueError, match=r"real_raw"):
            net.build_conditioning({"condition": text, "neg_condition": text}, input_shape=None)

    @pytest.mark.parametrize("with_mask", [False, True])
    def test_prepare_video_latents_uses_deterministic_posterior(self, with_mask):
        net = VACEWan.__new__(VACEWan)
        torch.nn.Module.__init__(net)
        vae = MagicMock()
        vae.encode.side_effect = lambda value, **kwargs: value[:, :1]
        net.preprocessors = {"vae": vae}

        video = torch.randn(1, 3, 2, 4, 4)
        mask = torch.ones_like(video) if with_mask else None
        net._prepare_video_latents(video, mask=mask)

        expected_calls = 2 if with_mask else 1
        assert vae.encode.call_count == expected_calls
        assert all(call.kwargs == {"mode": "argmax"} for call in vae.encode.call_args_list)
        if with_mask:
            with pytest.raises(AssertionError, match="mask"):
                net._prepare_video_latents(video, mask=mask.to("meta"))

    @pytest.fixture
    def depth_checkpoint(self, monkeypatch, tmp_path):
        class DepthAnything(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.inputs = []

            def infer_tensor(self, frame):
                self.inputs.append((frame.dtype, frame.device))
                return frame.mean(dim=0) + self.weight

        model_path = tmp_path / "depth.pth"
        torch.save({"weight": torch.tensor([2.0])}, model_path)
        constructor = MagicMock(side_effect=DepthAnything)
        monkeypatch.setattr(vace_depth_module, "DepthAnythingV2", constructor)
        monkeypatch.setattr(vace_depth_module, "cache", lambda path, strict: path)
        return str(model_path), constructor

    def test_init_preprocessors_loads_depth_annotator_once(self, depth_checkpoint):
        model_path, constructor = depth_checkpoint
        net = VACEWan.__new__(VACEWan)
        torch.nn.Module.__init__(net)
        net.preprocessors = {"text_encoder": MagicMock(), "vae": MagicMock()}
        net.depth_model_path = model_path

        net.init_preprocessors()

        depth_annotator = net.preprocessors["depth_annotator"]
        assert isinstance(depth_annotator, vace_depth_module.VACEDepthExtractor)
        assert depth_annotator.model_path == model_path
        assert torch.equal(depth_annotator.model.weight, torch.tensor([2.0]))
        assert not depth_annotator.is_prepared
        assert "depth_annotator" not in dict(net.named_children())

        net.init_preprocessors()
        assert net.preprocessors["depth_annotator"] is depth_annotator
        assert constructor.call_count == 1

    def test_depth_annotator_defaults_to_pinned_http_checkpoint(self, monkeypatch, depth_checkpoint):
        model_path, _ = depth_checkpoint
        load = vace_depth_module.load_checkpoint_state_dict
        checkpoint_load = MagicMock(side_effect=lambda _path, device: load(model_path, device=device))
        monkeypatch.setattr(vace_depth_module, "load_checkpoint_state_dict", checkpoint_load)

        depth_annotator = vace_depth_module.VACEDepthExtractor()

        assert depth_annotator.model_path == vace_depth_module.DEPTH_ANYTHING_V2_VITL_URL
        assert depth_annotator.model_path.startswith("https://")
        assert "/resolve/main/" not in depth_annotator.model_path
        checkpoint_load.assert_called_once_with(depth_annotator.model_path, device="cpu")

    @pytest.mark.parametrize("prepared_device", ["cpu", None])
    @pytest.mark.parametrize("inference_context", [False, True])
    def test_depth_annotator_loads_before_preparation_in_fp32(
        self, monkeypatch, depth_checkpoint, prepared_device, inference_context
    ):
        model_path, constructor = depth_checkpoint
        checkpoint_load = MagicMock(wraps=vace_depth_module.load_checkpoint_state_dict)
        monkeypatch.setattr(vace_depth_module, "load_checkpoint_state_dict", checkpoint_load)

        with torch.inference_mode(inference_context):
            depth_annotator = vace_depth_module.VACEDepthExtractor(model_path=model_path)
        loaded_model = depth_annotator.model
        assert not depth_annotator.is_prepared
        assert loaded_model.training and loaded_model.weight.requires_grad
        assert loaded_model.weight.device == torch.device("cpu")
        assert loaded_model.weight.dtype == torch.float32
        assert not loaded_model.weight.is_inference()
        assert torch.equal(loaded_model.weight, torch.tensor([2.0]))
        checkpoint_load.assert_called_once_with(model_path, device="cpu")

        depth_annotator.prepare_module(device=prepared_device, precision_model=torch.bfloat16, trainable=False)
        assert depth_annotator.precision_model == torch.float32
        assert depth_annotator._precision_opt is None
        assert not loaded_model.training and not loaded_model.weight.requires_grad
        assert loaded_model.weight.dtype == torch.float32

        monkeypatch.setattr(
            depth_annotator,
            "prepare_module",
            MagicMock(side_effect=AssertionError("forward must not repeat preparation")),
        )
        frames = torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 2, 2)
        with torch.inference_mode():
            output = depth_annotator(frames)
            video_output = depth_annotator(frames.unsqueeze(2))

        assert output.shape == frames.shape
        assert output.dtype == torch.float32
        torch.testing.assert_close(video_output.squeeze(2), output)
        assert loaded_model.inputs == [(torch.float32, torch.device("cpu"))] * 2
        assert depth_annotator.model is loaded_model
        assert constructor.call_count == 1
        assert checkpoint_load.call_count == 1

    def test_depth_annotator_propagates_cache_failure_before_model_construction(self, monkeypatch, depth_checkpoint):
        model_path, constructor = depth_checkpoint
        monkeypatch.setattr(
            vace_depth_module,
            "cache",
            MagicMock(side_effect=RuntimeError("injected cache failure")),
        )

        with pytest.raises(RuntimeError, match="injected cache failure"):
            vace_depth_module.VACEDepthExtractor(model_path=model_path)

        constructor.assert_not_called()

    def test_failed_depth_load_does_not_register_a_preprocessor(self, monkeypatch, depth_checkpoint):
        model_path, _ = depth_checkpoint
        net = VACEWan.__new__(VACEWan)
        torch.nn.Module.__init__(net)
        net.preprocessors = {"text_encoder": MagicMock(), "vae": MagicMock()}
        net.depth_model_path = model_path
        monkeypatch.setattr(
            vace_depth_module,
            "load_checkpoint_state_dict",
            MagicMock(side_effect=RuntimeError("injected checkpoint failure")),
        )

        with pytest.raises(RuntimeError, match="injected checkpoint failure"):
            net.init_preprocessors()
        assert "depth_annotator" not in net.preprocessors

    def test_build_vid_context_uses_registered_depth_annotator(self, monkeypatch):
        net = VACEWan.__new__(VACEWan)
        torch.nn.Module.__init__(net)
        depth_annotator = MagicMock(side_effect=lambda frames: frames)
        net.preprocessors = {"vae": MagicMock(), "depth_annotator": depth_annotator}
        net._all_zero_latent = None
        net._prepare_video_latents = lambda video, mask: video.new_zeros(video.shape[0], 32, *video.shape[2:])

        sync_inputs = []

        def fake_sync(local_value, device):
            sync_inputs.append((local_value, device))
            return local_value

        monkeypatch.setattr(vace_network_module, "sync_any", fake_sync)
        monkeypatch.setattr(vace_network_module, "sync_all", fake_sync)

        video = torch.zeros(1, 3, 1, 2, 2, device="meta")
        out = net._build_vid_context(video=video)

        # The raw path is selected because not every rank has precomputed depth,
        # then pixel availability is confirmed everywhere.
        assert sync_inputs == [(False, video.device), (True, video.device)]
        depth_input = depth_annotator.call_args.args[0]
        assert depth_input.device == video.device
        assert depth_input.dtype == video.dtype
        assert out.device == video.device

    def test_rank_wide_raw_fallback_recomputes_local_precomputed_depth(self, monkeypatch):
        net = VACEWan.__new__(VACEWan)
        torch.nn.Module.__init__(net)
        depth_annotator = MagicMock(side_effect=lambda frames: frames)
        net.preprocessors = {"vae": MagicMock(), "depth_annotator": depth_annotator}
        net._all_zero_latent = None
        net._prepare_video_latents = lambda video, mask: video.new_zeros(video.shape[0], 32, *video.shape[2:])

        sync_all_results = iter([False, True])

        def fake_sync_all(local_value, device):
            assert local_value is True
            return next(sync_all_results)

        monkeypatch.setattr(vace_network_module, "sync_all", fake_sync_all)

        video = torch.zeros(1, 3, 1, 2, 2)
        condition_latents = torch.ones(1, 16, 1, 2, 2)
        out = net._build_vid_context(video=video, condition_latents=condition_latents)

        depth_annotator.assert_called_once()
        assert torch.count_nonzero(out[:, :32]) == 0

    def test_depth_latent_without_real_raw_is_allowed(self):
        """Precomputed-latent shortcut: data['depth_latent'] alone (no real_raw) is accepted."""
        vid_ctx = torch.randn(1, 96, 4, 8, 8)
        net = _make_vacewan(vid_context=vid_ctx)
        depth_latent = torch.randn(1, 16, 4, 8, 8)
        text = torch.randn(1, 77, 768)
        data = {"depth_latent": depth_latent, "condition": text, "neg_condition": text}

        out = net.build_conditioning(data, input_shape=None)

        net._build_vid_context.assert_called_once_with(video=None, condition_latents=depth_latent)
        assert out["condition"] == {"text_embeds": text, "vid_context": vid_ctx}
        assert out["neg_condition"] == {"text_embeds": text, "vid_context": vid_ctx}

    def test_dict_condition_raises(self):
        net = _make_vacewan(vid_context=torch.zeros(1))
        with pytest.raises(ValueError, match=r"cannot be prepared again"):
            net.build_conditioning(
                {"condition": {"video": torch.randn(1, 77, 768)}, "real_raw": torch.randn(1, 3, 4, 8, 8)},
                input_shape=None,
            )

    def test_build_vid_context_precomputed_depth_needs_no_real_raw(self, conditioning_placement, monkeypatch):
        """Precomputed-depth path (``video=None``): the all-zero inactive latent is
        cached independently of batch size and deterministically reconstructed from
        latent geometry, so no pixel ``real_raw`` is required."""
        device, dtype = conditioning_placement
        net = VACEWan.__new__(VACEWan)
        torch.nn.Module.__init__(net)
        net.preprocessors = {}
        net._all_zero_latent = None

        b, t_lat, h_lat, w_lat = 2, 4, 8, 8
        t_factor, h_factor, w_factor = 4, 8, 8
        condition_latents = torch.randn(b, 16, t_lat, h_lat, w_lat, device=device, dtype=dtype)

        encoded_inputs = []
        sync_inputs = []

        def fake_encode(x, mode=None):
            encoded_inputs.append((tuple(x.shape), x.dtype, x.device, mode))
            # VAE encoding of the black clip → 16-channel inactive latent.
            return x.new_zeros(
                (
                    x.shape[0],
                    16,
                    (x.shape[2] - 1) // t_factor + 1,
                    x.shape[3] // h_factor,
                    x.shape[4] // w_factor,
                )
            )

        def fake_sync_any(local_any, device):
            sync_inputs.append((local_any, device))
            return local_any

        monkeypatch.setattr(vace_network_module, "sync_any", fake_sync_any)

        vae = MagicMock()
        vae.latent_compression_factors = (t_factor, h_factor, w_factor)
        vae.encode.side_effect = fake_encode
        net.preprocessors["vae"] = vae

        out = net._build_vid_context(video=None, condition_latents=condition_latents)
        same_shape_out = net._build_vid_context(video=None, condition_latents=condition_latents)

        changed_batch = torch.randn(3, 16, t_lat, h_lat, w_lat, device=device, dtype=dtype)
        changed_batch_out = net._build_vid_context(video=None, condition_latents=changed_batch)
        changed_shape = torch.randn(2, 16, 3, 6, 5, device=device, dtype=dtype)
        changed_shape_out = net._build_vid_context(video=None, condition_latents=changed_shape)
        changed_dtype = changed_shape.to(dtype=torch.float32)
        changed_dtype_out = net._build_vid_context(video=None, condition_latents=changed_dtype)

        # Zeros clip shape is reconstructed from the latent shape (RGB pixels).
        assert encoded_inputs == [
            (
                (1, 3, (t_lat - 1) * t_factor + 1, h_lat * h_factor, w_lat * w_factor),
                dtype,
                condition_latents.device,
                "argmax",
            ),
            ((1, 3, 9, 48, 40), dtype, changed_shape.device, "argmax"),
        ]
        # The only any-rank synchronization here is for the inactive-latent cache.
        assert [local_any for local_any, _ in sync_inputs] == [True, False, False, True, False]
        # Tensor.device resolves an unindexed ``cuda`` request to the concrete device (for example
        # ``cuda:0``); synchronization must follow that concrete input placement.
        assert all(sync_device == condition_latents.device for _, sync_device in sync_inputs)
        # 16 inactive + 16 reactive depth latents + 64 mask = 96 channels.
        assert out.shape == (b, 96, t_lat, h_lat, w_lat)
        assert same_shape_out.shape == out.shape
        assert changed_batch_out.shape == (3, 96, t_lat, h_lat, w_lat)
        assert changed_shape_out.shape == (2, 96, 3, 6, 5)
        assert changed_dtype_out.shape == changed_shape_out.shape
        assert net._all_zero_latent.shape[0] == 1
        _assert_tree_dtype_and_device(out, floating_dtype=dtype, device=device)
        _assert_tree_dtype_and_device(changed_dtype_out, floating_dtype=torch.float32, device=device)
        # Per-batch conditioning must not mutate persistent preprocessor ownership.
        assert net.preprocessors["vae"] is vae
        assert "depth_annotator" not in net.preprocessors

    def test_precomputed_depth_placement_does_not_follow_unused_pixels(self, monkeypatch):
        net = VACEWan.__new__(VACEWan)
        torch.nn.Module.__init__(net)
        net._all_zero_latent = torch.zeros(1, 16, 2, 2, 2, dtype=torch.float32)

        monkeypatch.setattr(vace_network_module, "sync_any", lambda *_args, **_kwargs: False)
        monkeypatch.setattr(vace_network_module, "sync_all", lambda local_value, **_kwargs: local_value)

        condition_latents = torch.ones(2, 16, 2, 2, 2, dtype=torch.bfloat16)
        unused_pixels = torch.empty(2, 3, 5, 16, 16, device="meta")
        out = net._build_vid_context(video=unused_pixels, condition_latents=condition_latents)

        assert out.device == condition_latents.device
        assert out.dtype == condition_latents.dtype
        assert torch.equal(out[:, 16:32], condition_latents)

    def test_symmetric_cache_miss_borrows_a_temporary_vae(self, monkeypatch):
        net = VACEWan.__new__(VACEWan)
        torch.nn.Module.__init__(net)
        net.preprocessors = {}
        net._all_zero_latent = None

        vae = MagicMock()
        vae.latent_compression_factors = (4, 8, 8)
        vae.encode.side_effect = lambda x, mode=None: x.new_ones(1, 16, 4, 8, 8)
        net.vae_context = MagicMock(return_value=contextlib.nullcontext(vae))

        sync_any_inputs = []
        sync_all_inputs = []

        def fake_sync_any(local_any, device):
            sync_any_inputs.append((local_any, device))
            return local_any

        def fake_sync_all(local_all, device):
            sync_all_inputs.append((local_all, device))
            return local_all

        monkeypatch.setattr(vace_network_module, "sync_any", fake_sync_any)
        monkeypatch.setattr(vace_network_module, "sync_all", fake_sync_all)
        condition_latents = torch.zeros(2, 16, 4, 8, 8)

        out = net._build_vid_context(video=None, condition_latents=condition_latents)

        # Every rank has precomputed depth, then every rank agrees that the cache is missing.
        assert [local_any for local_any, _ in sync_any_inputs] == [True]
        assert [local_all for local_all, _ in sync_all_inputs] == [True, True]
        assert all(device == condition_latents.device for _, device in sync_any_inputs + sync_all_inputs)
        net.vae_context.assert_called_once_with()
        encoded_clip = vae.encode.call_args.args[0]
        assert encoded_clip.shape == (1, 3, 13, 64, 64)
        assert vae.encode.call_args.kwargs == {"mode": "argmax"}
        assert torch.all(net._all_zero_latent == 1)
        assert torch.all(out[:, :16] == 1)

    def test_rank_asymmetric_cache_miss_fails_before_vae_acquisition(self, monkeypatch):
        net = VACEWan.__new__(VACEWan)
        torch.nn.Module.__init__(net)
        net.preprocessors = {}
        net._all_zero_latent = torch.zeros(1, 16, 4, 8, 8)
        net.vae_context = MagicMock(side_effect=AssertionError("must not acquire with an asymmetric cache"))

        # Every rank has precomputed depth, but the cache state disagrees.
        monkeypatch.setattr(vace_network_module, "sync_any", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(vace_network_module, "sync_all", lambda local_all, **_kwargs: local_all)

        with pytest.raises(AssertionError, match="cache misses must be consistent"):
            net._build_vid_context(video=None, condition_latents=torch.zeros(2, 16, 4, 8, 8))

        net.vae_context.assert_not_called()

    @pytest.mark.parametrize("precomputed", [False, True], ids=["raw_video", "precomputed_depth"])
    def test_rank_wide_raw_fallback_requires_video_on_every_rank(self, monkeypatch, precomputed):
        net = VACEWan.__new__(VACEWan)
        torch.nn.Module.__init__(net)
        net.preprocessors = {}
        net._all_zero_latent = None
        net.vae_context = MagicMock(side_effect=AssertionError("must not acquire a VAE on a split path"))

        monkeypatch.setattr(vace_network_module, "sync_all", lambda *_args, **_kwargs: False)

        video = None if precomputed else torch.zeros(2, 3, 13, 64, 64)
        condition_latents = torch.zeros(2, 16, 4, 8, 8) if precomputed else None
        with pytest.raises(RuntimeError, match=r"`real_raw` is not available on every rank"):
            net._build_vid_context(video=video, condition_latents=condition_latents)

        net.vae_context.assert_not_called()
        assert net.preprocessors == {}


# ═════════════════════════════════════════════════════════════════════════
# CosmosPredict2.build_conditioning
# ═════════════════════════════════════════════════════════════════════════


class TestCosmosPredict2BuildConditioning:
    def test_t2v_passthrough(self):
        """T2V returns the base builder's authoritative encoded branches."""
        net = _make_cosmos(is_video2world=False)
        text = torch.randn(1, 77, 768)
        neg_text = torch.randn(1, 77, 768)
        data = {"condition": text, "neg_condition": neg_text}
        out = net.build_conditioning(data, input_shape=[16, 4, 8, 8])
        assert out["condition"] is text
        assert out["neg_condition"] is neg_text

    def test_t2v_passthrough_no_real_raw_needed(self):
        """T2V mode does not require real_raw or input_shape."""
        net = _make_cosmos(is_video2world=False)
        text = torch.randn(1, 77, 768)
        out = net.build_conditioning({"condition": text}, input_shape=None)
        assert out == {"condition": text}

    def test_t2v_rejects_rich_condition_mapping(self):
        """Complete conditions are builder outputs and are not prepared again, in T2V mode too."""
        net = _make_cosmos(is_video2world=False)
        with pytest.raises(ValueError, match=r"cannot be prepared again"):
            net.build_conditioning({"condition": {"text_embeds": torch.randn(1, 77, 768)}}, input_shape=None)

    def test_v2w_requires_the_vae(self):
        net = _make_cosmos(is_video2world=True)
        net.preprocessors.pop("vae")
        data = {"condition": torch.randn(1, 77, 768), "real_raw": torch.randn(1, 3, 2, 16, 16)}
        with pytest.raises(ValueError, match=r"requires the 'vae' preprocessor"):
            net.build_conditioning(data, input_shape=[16, 4, 8, 8])

    def test_v2w_builds_cond_latents_and_mask(self, conditioning_placement):
        """V2W: encode first num_conditioning_frames; build mask in latent space from input_shape."""
        device, dtype = conditioning_placement
        net = _make_cosmos(is_video2world=True, num_conditioning_frames=1)

        bsz, c, t, h, w = 2, 3, 8, 64, 64
        raw_video = torch.randn(bsz, c, t, h, w, device=device, dtype=dtype)
        cond_latents = torch.randn(bsz, 16, 1, 8, 8, device=device, dtype=dtype)

        captured = {}

        def fake_encode(x, mode=None):
            captured["mode"] = mode
            captured["pixel_T"] = x.shape[2]
            return cond_latents

        net.preprocessors["vae"].encode.side_effect = fake_encode

        text = torch.randn(bsz, 77, 768, device=device, dtype=dtype)
        input_shape = [16, 4, 8, 8]  # t_latent=4, h_latent=w_latent=8
        out = net.build_conditioning(
            {"real_raw": raw_video, "condition": text, "neg_condition": text},
            input_shape=input_shape,
        )
        cond, neg_cond = out["condition"], out["neg_condition"]

        assert captured["mode"] == "argmax"
        assert captured["pixel_T"] == 1  # only 1 conditioning frame requested
        assert torch.equal(cond["conditioning_latents"], cond_latents)

        mask = cond["condition_mask"]
        assert mask.shape == (bsz, 1, 4, 8, 8)
        assert mask.dtype == torch.bool
        assert torch.all(mask[:, :, :1] == 1.0)
        assert torch.all(mask[:, :, 1:] == 0.0)
        assert torch.equal(neg_cond["conditioning_latents"], cond_latents)
        assert torch.equal(neg_cond["condition_mask"], mask)
        _assert_tree_dtype_and_device(out, floating_dtype=dtype, device=device)

    def test_v2w_replicates_when_caller_provides_too_few_frames(self):
        """V2W with num_conditioning_frames=4 and a 1-frame seed: replicate to fill (inference path)."""
        net = _make_cosmos(is_video2world=True, num_conditioning_frames=4)

        bsz, c, h, w = 1, 3, 32, 32
        raw_video = torch.randn(bsz, c, 1, h, w)  # only 1 frame supplied
        cond_latents = torch.randn(bsz, 16, 4, 4, 4)

        captured = {}

        def fake_encode(x, mode=None):
            captured["pixel_T"] = x.shape[2]
            return cond_latents

        net.preprocessors["vae"].encode.side_effect = fake_encode

        text = torch.randn(bsz, 77, 768)
        net.build_conditioning(
            {"real_raw": raw_video, "condition": text, "neg_condition": text},
            input_shape=[16, 8, 4, 4],
        )
        assert captured["pixel_T"] == 4, "real_raw should have been replicated to num_conditioning_frames"

    def test_v2w_pads_partial_input_with_first_frame(self):
        """V2W with num_conditioning_frames=4 and a T=2 input: prepend two copies of the first frame, then keep both user frames.

        The user-supplied frames land at the END of the conditioning window
        so the model sees a held initial state evolving into the user clip.

        Also guards against a ``torch.Tensor.expand()`` regression — expand
        only works on size-1 dims, so naive replication would crash when
        the caller supplies more than 1 but fewer than
        ``num_conditioning_frames``.
        """
        net = _make_cosmos(is_video2world=True, num_conditioning_frames=4)

        bsz, c, h, w = 1, 3, 32, 32
        zeros = torch.zeros(bsz, c, h, w)
        ones = torch.ones(bsz, c, h, w)
        raw_video = torch.stack([zeros, ones], dim=2)  # T=2: [0s, 1s]
        cond_latents = torch.randn(bsz, 16, 4, 4, 4)

        captured = {}

        def fake_encode(x, mode=None):
            captured["x"] = x
            return cond_latents

        net.preprocessors["vae"].encode.side_effect = fake_encode

        text = torch.randn(bsz, 77, 768)
        net.build_conditioning(
            {"real_raw": raw_video, "condition": text, "neg_condition": text},
            input_shape=[16, 8, 4, 4],
        )
        x = captured["x"]
        assert x.shape == (bsz, c, 4, h, w), "padded to num_conditioning_frames=4"
        # Pad (2 copies of first frame = zeros) ++ user frames [zeros, ones].
        assert torch.equal(x[:, :, 0], zeros)
        assert torch.equal(x[:, :, 1], zeros)
        assert torch.equal(x[:, :, 2], zeros)
        assert torch.equal(x[:, :, 3], ones)

    def test_v2w_missing_real_raw_raises(self):
        net = _make_cosmos(is_video2world=True)
        text = torch.randn(1, 77, 768)
        with pytest.raises(ValueError, match=r"real_raw"):
            net.build_conditioning(
                {"condition": text, "neg_condition": text},
                input_shape=[16, 4, 8, 8],
            )

    def test_v2w_missing_input_shape_raises(self):
        net = _make_cosmos(is_video2world=True)
        text = torch.randn(1, 77, 768)
        with pytest.raises(ValueError, match=r"input_shape"):
            net.build_conditioning(
                {"condition": text, "neg_condition": text, "real_raw": torch.randn(1, 3, 4, 32, 32)},
                input_shape=None,
            )

    def test_v2w_dict_condition_raises(self):
        net = _make_cosmos(is_video2world=True)
        with pytest.raises(ValueError, match=r"cannot be prepared again"):
            net.build_conditioning(
                {
                    "condition": {"video": torch.randn(1, 77, 768)},
                    "real_raw": torch.randn(1, 3, 4, 32, 32),
                },
                input_shape=[16, 4, 4, 4],
            )


# ═════════════════════════════════════════════════════════════════════════
# Explicit ``neg_condition=None`` across rich builders
# ═════════════════════════════════════════════════════════════════════════


def _wan_i2v_none_case():
    net = _make_wan_i2v(concat_mask=False, with_image_encoder=False)
    net.preprocessors["vae"].encode.return_value = torch.randn(1, 16, 1, 1, 1)
    return net, {"real_raw": torch.randn(1, 3, 4, 8, 8), "condition": torch.randn(1, 77, 768)}, None


def _wan_moe_none_case():
    """WanMoE delegates to its high-noise expert, so the inherited rule must hold through it."""
    from fastgen.networks.Wan.wan_moe import WanMoE

    expert, data, input_shape = _wan_i2v_none_case()
    net = WanMoE.__new__(WanMoE)
    torch.nn.Module.__init__(net)
    net.network_high = expert
    return net, data, input_shape


def _ltx2_i2v_none_case():
    net = _make_ltx2i2v()
    net.preprocessors["vae"].encode_video.return_value = torch.randn(1, 16, 1, 1, 1)
    text = (torch.randn(1, 77, 768), torch.ones(1, 77, dtype=torch.long))
    return net, {"real_raw": torch.randn(1, 3, 4, 32, 32), "condition": text}, None


def _vacewan_none_case():
    net = _make_vacewan(vid_context=torch.randn(1, 96, 4, 8, 8))
    return net, {"depth_latent": torch.randn(1, 16, 4, 8, 8), "condition": torch.randn(1, 77, 768)}, None


def _cosmos_v2w_none_case():
    net = _make_cosmos(is_video2world=True)
    net.preprocessors["vae"].encode.return_value = torch.randn(1, 16, 1, 8, 8)
    return net, {"real_raw": torch.randn(1, 3, 4, 64, 64), "condition": torch.randn(1, 77, 768)}, [16, 4, 8, 8]


def _flux2_image_edit_none_case():
    from fastgen.networks.Flux2.network import Flux2

    net = Flux2.__new__(Flux2)
    net.preprocessors = {}
    net.is_image_edit = True
    # The encoded key is authoritative, so no VAE is consulted.
    data = {"condition": (None, torch.randn(1, 3, 8)), "condition_image": torch.zeros(1, 4, 2, 2)}
    return net, data, [4, 2, 2]


@pytest.mark.parametrize(
    "make_case",
    [
        _wan_i2v_none_case,
        _wan_moe_none_case,
        _ltx2_i2v_none_case,
        _vacewan_none_case,
        _cosmos_v2w_none_case,
        _flux2_image_edit_none_case,
    ],
    ids=["wan-i2v", "wan-moe", "ltx2-i2v", "vacewan", "cosmos-v2w", "flux2-image-edit"],
)
def test_explicit_none_negative_matches_missing_negative(make_case):
    """A rich builder omits both ``neg_condition=None`` and an absent negative branch."""
    net, data, input_shape = make_case()

    missing = net.build_conditioning(dict(data), input_shape=input_shape)
    explicit = net.build_conditioning({**data, "neg_condition": None}, input_shape=input_shape)

    assert "neg_condition" not in missing and "neg_condition" not in explicit
    assert set(missing) == set(explicit) == {"condition"}


@pytest.mark.parametrize("explicit_none", [False, True], ids=["missing", "explicit-none"])
def test_wan_camera_enriches_only_resolved_condition_branches(explicit_none):
    """Camera context must not manufacture a negative branch when WanI2V omitted it."""
    from fastgen.networks.WanI2V.network_cam import WanCamI2V

    net = WanCamI2V.__new__(WanCamI2V)
    net.preprocessors = {}
    net.concat_mask = False
    net.latent_compression_factors = (4, 8, 8)
    vae = MagicMock()
    vae.latent_compression_factors = net.latent_compression_factors
    vae.encode.return_value = torch.randn(1, 16, 1, 1, 1)
    net.preprocessors["vae"] = vae
    camera_condition = torch.randn(1, 24, 1, 1, 1)
    net.compute_camera_conditioning = MagicMock(return_value=camera_condition)

    data = {
        "real_raw": torch.randn(1, 3, 1, 8, 8),
        "condition": torch.randn(1, 77, 768),
        "camera_poses": torch.eye(4).reshape(1, 1, 4, 4),
    }
    if explicit_none:
        data["neg_condition"] = None

    out = net.build_conditioning(data, input_shape=[16, 1, 1, 1])

    assert out["condition"]["c2ws_plucker_emb"] is camera_condition
    assert "neg_condition" not in out
    net.compute_camera_conditioning.assert_called_once_with(data)


def test_cosmos3_explicit_none_branches_fall_back_to_raw_prompts():
    """Cosmos3 owns its resolver, so its explicit-None fallback is tested directly."""
    from fastgen.networks.cosmos3.network import Cosmos3

    net = Cosmos3.__new__(Cosmos3)
    net.preprocessors = {}
    net.is_i2v = False
    net.fps = 24.0
    positive = {"input_ids": [torch.zeros(3, dtype=torch.long)]}
    negative = {"input_ids": [torch.ones(3, dtype=torch.long)]}
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = [positive, negative]
    net.preprocessors["text_tokenizer"] = tokenizer
    data = {
        "condition": None,
        "condition_raw": ["positive"],
        "neg_condition": None,
        "neg_condition_raw": ["negative"],
    }

    out = net.build_conditioning(data, input_shape=[48, 4, 8, 8])

    assert out == {"condition": positive, "neg_condition": negative}
    assert [call.args[0] for call in tokenizer.encode.call_args_list] == [["positive"], ["negative"]]
    assert [call.kwargs["is_negative"] for call in tokenizer.encode.call_args_list] == [False, True]


def _make_cosmos3(is_i2v: bool, with_tokenizer: bool = True):
    from fastgen.networks.cosmos3.network import Cosmos3

    net = Cosmos3.__new__(Cosmos3)
    net.preprocessors = {}
    net.is_i2v = is_i2v
    net.fps = 24.0
    if with_tokenizer:
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda prompts, **kwargs: {"input_ids": [torch.zeros(3, dtype=torch.long)]}
        net.preprocessors["text_tokenizer"] = tokenizer
    return net


def test_cosmos3_raw_prompts_require_the_text_tokenizer():
    """Preprocessors are constructed by the model lifecycle, never lazily inside the builder."""
    net = _make_cosmos3(is_i2v=False, with_tokenizer=False)
    with pytest.raises(ValueError, match=r"requires the 'text_tokenizer' preprocessor"):
        net.build_conditioning({"condition_raw": ["a prompt"]}, input_shape=[48, 4, 8, 8])
    assert net.preprocessors == {}


def test_cosmos3_prompt_geometry_requires_input_shape():
    """``input_shape`` is mandatory: the tokenizer's 189-frame 720x1280 template defaults are never used."""
    net = _make_cosmos3(is_i2v=False)
    with pytest.raises(TypeError, match="has no len"):
        net.build_conditioning({"condition_raw": ["a prompt"]}, input_shape=None)
    net.preprocessors["text_tokenizer"].encode.assert_not_called()


def test_cosmos3_i2v_requires_the_vae():
    net = _make_cosmos3(is_i2v=True)
    data = {"condition_raw": ["a prompt"], "real_raw": torch.zeros(1, 3, 5, 16, 16)}
    with pytest.raises(ValueError, match=r"requires the 'vae' preprocessor"):
        net.build_conditioning(data, input_shape=[48, 2, 1, 1])
    assert "vae" not in net.preprocessors
