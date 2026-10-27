# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the FastGen ComfyUI custom-node logic.

The tests exercise the nodes' own decision logic — config-override parsing,
reference-image conditioning, and sampling argument handling — in isolation.
They run on CPU with no models, checkpoints, or real ComfyUI runtime: ComfyUI's
``folder_paths`` module is stubbed so the ``comfyui`` package imports, and the
heavy per-network loaders / ``build_conditioning`` hook are monkeypatched so
only the node branching is exercised.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import wave
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Make the ``comfyui`` package importable outside a real ComfyUI runtime:
#   * ``FASTGEN_REPO`` is read by comfyui/__init__.py to locate the checkout.
#   * ``folder_paths`` is a ComfyUI-provided module; stub the few attributes the
#     package touches at import time (get_input_directory + folder_names_and_paths).
# Must happen before importing anything under ``comfyui``.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Pin the checkout under test: comfyui/__init__.py puts FASTGEN_REPO at sys.path[0], so an inherited value would
# import another checkout's fastgen and scripts.
os.environ["FASTGEN_REPO"] = _REPO_ROOT

if "folder_paths" not in sys.modules:
    _fp = ModuleType("folder_paths")
    _fp_tmp = tempfile.TemporaryDirectory(prefix="fastgen_comfyui_test_")  # held so the directory goes at exit
    _fp_dir = _fp_tmp.name
    _fp.get_input_directory = lambda: _fp_dir
    _fp.get_output_directory = lambda: _fp_dir
    _fp.folder_names_and_paths = {}
    _fp.get_filename_list = lambda *a, **k: []
    _fp.get_full_path = lambda *a, **k: None
    _fp.get_save_image_path = lambda prefix, out: (_fp_dir, "test", 0, "", "test")
    sys.modules["folder_paths"] = _fp

from comfyui import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    nodes_condition,
    nodes_config,
    nodes_decode,
    nodes_model,
    nodes_prompt,
)
from comfyui._config_helpers import parse_overrides, temporary_overrides  # noqa: E402
from comfyui.nodes_condition import (  # noqa: E402
    FastGenCameraLoader,
    FastGenDepthLatentLoader,
    FastGenPreprocess,
)
from comfyui.nodes_config import FastGenConfigLoader, _discover_configs  # noqa: E402
from comfyui.nodes_decode import FastGenDecodeVideo  # noqa: E402
from comfyui.nodes_model import FastGenModelLoader  # noqa: E402
from comfyui.nodes_prompt import FastGenPromptExpand  # noqa: E402
from comfyui.nodes_sampling import (  # noqa: E402
    _CONFIG_OVERRIDES_TOOLTIP,
    FastGenNoiseGenerator,
    FastGenStudentSample,
    FastGenTeacherSample,
)
from fastgen.methods.model import FastGenModel  # noqa: E402
from fastgen.networks.network import FastGenNetwork  # noqa: E402
from fastgen.utils import media_utils  # noqa: E402
from scripts.inference import inference_utils  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════
# _config_helpers.parse_overrides
# ═════════════════════════════════════════════════════════════════════════


class TestParseOverrides:
    def test_dotted_key(self):
        assert parse_overrides("model.guidance_cfg.cfg.scale=5.0") == {"model.guidance_cfg.cfg.scale": "5.0"}

    def test_multiline_and_space_separated(self):
        assert parse_overrides("a.b=1\nc.d=2 e.f=3") == {"a.b": "1", "c.d": "2", "e.f": "3"}

    def test_quoted_value_and_trailing_comment(self):
        assert parse_overrides('k="hello world"  # a comment') == {"k": "hello world"}

    def test_token_without_equals_is_skipped(self):
        assert parse_overrides("valid=1 bogus other=2") == {"valid": "1", "other": "2"}

    def test_list_value_kept_as_raw_string(self):
        # Values stay strings here; ``temporary_overrides`` is what literal-evals them.
        assert parse_overrides("m.student_grid_cfg.grid_kwargs.timesteps=[0.9,0.5,0.0]") == {
            "m.student_grid_cfg.grid_kwargs.timesteps": "[0.9,0.5,0.0]"
        }

    def test_empty_string(self):
        assert parse_overrides("") == {}


# ═════════════════════════════════════════════════════════════════════════
# _config_helpers.temporary_overrides
# ═════════════════════════════════════════════════════════════════════════


class TestTemporaryOverrides:
    def test_apply_then_restore_nested(self):
        cfg = SimpleNamespace(model=SimpleNamespace(scale=1.0, sub=SimpleNamespace(x=2)))
        with temporary_overrides(cfg, "model.scale=5.0\nmodel.sub.x=9"):
            assert cfg.model.scale == 5.0
            assert cfg.model.sub.x == 9

        assert cfg.model.scale == 1.0
        assert cfg.model.sub.x == 2

    def test_restores_when_the_block_raises(self):
        cfg = SimpleNamespace(scale=1.0)
        with pytest.raises(RuntimeError, match="sampling failed"), temporary_overrides(cfg, "scale=5.0"):
            raise RuntimeError("sampling failed")
        assert cfg.scale == 1.0

    def test_failing_override_undoes_the_ones_already_applied(self):
        class Config:
            scale = 1.0

            @property
            def steps(self):
                return 4

            @steps.setter
            def steps(self, value):
                raise ValueError("steps must be positive")

        cfg = Config()
        with pytest.raises(ValueError, match="positive"), temporary_overrides(cfg, "scale=5.0 steps=0"):
            pass
        assert cfg.scale == 1.0

    def test_literal_eval_of_value_types(self):
        cfg = SimpleNamespace(a=0, b=None, c=None)
        with temporary_overrides(cfg, "a=[1,2] b=True c=hello"):
            assert cfg.a == [1, 2]  # list literal
            assert cfg.b is True  # bool literal
            assert cfg.c == "hello"  # unparseable → raw string

    @pytest.mark.parametrize(
        "overrides, model_inside, original_inside",
        [
            ("model=\"{'scale': 2}\" model.scale=3", {"scale": 3}, {"scale": 1}),
            ("model.scale=3 model=\"{'scale': 2}\"", {"scale": 2}, {"scale": 3}),
        ],
    )
    def test_overlapping_overrides_restore_the_original_subtree(self, overrides, model_inside, original_inside):
        original = {"scale": 1}
        cfg = SimpleNamespace(model=original)
        with temporary_overrides(cfg, overrides):
            assert cfg.model is not original
            assert (cfg.model, original) == (model_inside, original_inside)

        assert cfg.model is original
        assert original == {"scale": 1}

    def test_plain_dict_path_can_be_added_and_restored(self):
        cfg = SimpleNamespace(model=SimpleNamespace(student_grid_cfg=SimpleNamespace(grid_kwargs={})))
        with temporary_overrides(cfg, "model.student_grid_cfg.grid_kwargs.shift=5.0"):
            assert cfg.model.student_grid_cfg.grid_kwargs == {"shift": 5.0}

        assert cfg.model.student_grid_cfg.grid_kwargs == {}

    def test_nested_dict_path_can_be_added_and_restored(self):
        cfg = SimpleNamespace(model={"student_grid_cfg": {"grid_kwargs": {}}})
        with temporary_overrides(cfg, "model.student_grid_cfg.grid_kwargs.shift=5.0"):
            assert cfg.model["student_grid_cfg"]["grid_kwargs"] == {"shift": 5.0}

        assert cfg.model["student_grid_cfg"]["grid_kwargs"] == {}

    def test_missing_attribute_is_skipped(self):
        cfg = SimpleNamespace(a=1)
        with temporary_overrides(cfg, "does.not.exist=1 a=2"):
            assert vars(cfg) == {"a": 2}  # the unresolved key does not block the others

        assert vars(cfg) == {"a": 1}

    def test_sampling_tooltip_example_replaces_a_native_grid(self):
        grid = SimpleNamespace(grid_type="native", grid_kwargs={"shift": 5.0})
        guidance = SimpleNamespace(cfg=SimpleNamespace(scale=1.0))
        cfg = SimpleNamespace(model=SimpleNamespace(guidance_cfg=guidance, student_grid_cfg=grid))

        with temporary_overrides(cfg, _CONFIG_OVERRIDES_TOOLTIP.split("e.g.:\n", 1)[1]):
            assert guidance.cfg.scale == 5.0
            assert grid.grid_type == "explicit"
            assert grid.grid_kwargs == {"timesteps": [0.999, 0.937, 0.833, 0.624, 0.0]}

        assert (grid.grid_type, grid.grid_kwargs) == ("native", {"shift": 5.0})


# ═════════════════════════════════════════════════════════════════════════
# FastGenPreprocess — unified raw-input preprocessing
# ═════════════════════════════════════════════════════════════════════════


def _make_bundle(input_shape):
    """Minimal model bundle for tests that stub the preprocessing call."""
    net = SimpleNamespace(preprocessors={}, fps=16)
    model = SimpleNamespace(
        input_shape=input_shape,
        precision=torch.float32,
        precision_amp_enc=None,
        precision_amp_infer=None,
        device=torch.device("cpu"),
        net_preprocessors=net,
        reshard_fsdp_modules=lambda preprocessors_only=False: None,
    )
    model.state_fields = FastGenModel.state_fields
    model.prepare_batch = FastGenModel.prepare_batch.__get__(model)
    return {
        "model": model,
        "net": net,
        "ctx": {"dtype": torch.float32, "device": torch.device("cpu")},
        "fps": 16.0,
    }


def _boom(*_a, **_k):
    raise AssertionError("this loader must not be called for this input_shape rank")


def _conditioning(condition="cond", neg_condition="neg", num_segments=1, overlap_frames=0, fps=16.0):
    return {
        "condition": condition,
        "neg_condition": neg_condition,
        "num_segments": num_segments,
        "overlap_frames": overlap_frames,
        "fps": fps,
    }


class _ParityTextEncoder(torch.nn.Module):
    def __init__(self, tuple_output: bool = False):
        super().__init__()
        self.tuple_output = tuple_output

    def encode(self, prompts):
        embeddings = torch.tensor(
            [[len(prompt), sum(map(ord, prompt)) % 997] for prompt in prompts],
            dtype=torch.float32,
        )
        if self.tuple_output:
            return embeddings, torch.ones(embeddings.shape[0], 2, dtype=torch.int64)
        return embeddings


class _ParityVAE(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def encode(self, value, **_kwargs):
        return torch.zeros(value.shape[0], *self.input_shape, device=value.device, dtype=value.dtype)


class _ParityNet(torch.nn.Module):
    """Expose every task input in the built condition for exact comparisons."""

    encode_real = FastGenNetwork.encode_real

    def __init__(self, input_shape, required_fields=(), tuple_text=False, passthrough=False):
        super().__init__()
        self.required_fields = tuple(required_fields)
        self.passthrough = passthrough
        self.preprocessing_modes = []
        self.preprocessors = {"text_encoder": _ParityTextEncoder(tuple_output=tuple_text)}
        self.fps = 16
        if any(field in self.required_fields for field in ("condition_image_raw", "real_raw", "depth_latent")):
            self.preprocessors["vae"] = _ParityVAE(input_shape)

    def build_conditioning(self, data, input_shape=None, precision=None):
        self.preprocessing_modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        encoded = {}
        for key, raw_key in (("condition", "condition_raw"), ("neg_condition", "neg_condition_raw")):
            value = data.get(key)
            if value is None and data.get(raw_key) is not None:
                value = self.preprocessors["text_encoder"].encode(data[raw_key])
            if value is not None:
                encoded[key] = value
        if self.passthrough:
            return encoded

        required = (*self.required_fields, "condition_raw", "neg_condition_raw")
        missing = [field for field in required if field not in data]
        if missing:
            raise AssertionError(f"Missing task conditioning fields: {missing}")
        shared = {}
        for field in self.required_fields:
            if field == "condition_image_raw":
                shared["condition_image"] = self.preprocessors["vae"].encode(data[field], mode="argmax")
            else:
                shared[field] = data[field]
        return {
            "condition": {
                "text": encoded["condition"],
                "text_raw": data["condition_raw"],
                **shared,
            },
            "neg_condition": {
                "text": encoded["neg_condition"],
                "text_raw": data["neg_condition_raw"],
                **shared,
            },
        }


def _parity_bundle(input_shape, required_fields=(), tuple_text=False, passthrough=False, fps=16.0):
    net = _ParityNet(
        input_shape,
        required_fields,
        tuple_text=tuple_text,
        passthrough=passthrough,
    )
    net.fps = fps
    model = SimpleNamespace(
        device=torch.device("cpu"),
        precision=torch.float32,
        precision_amp_enc=None,
        precision_amp_infer=None,
        input_shape=input_shape,
        net_preprocessors=net,
        sample_calls=[],
        generator_calls=[],
        reshard_fsdp_modules=lambda preprocessors_only=False: None,
    )
    model.state_fields = FastGenModel.state_fields
    model.prepare_batch = FastGenModel.prepare_batch.__get__(model)

    sample_steps_unset = object()

    def sample(
        net,
        noise,
        condition=None,
        neg_condition=None,
        sample_steps=sample_steps_unset,
    ):
        call = {
            "teacher": net,
            "noise": noise,
            "condition": condition,
            "neg_condition": neg_condition,
        }
        if sample_steps is not sample_steps_unset:
            call["sample_steps"] = sample_steps
        model.sample_calls.append(call)
        return noise

    def generator_fn(net, noise, data=None, condition=None, neg_condition=None):
        model.generator_calls.append(
            {
                "student": net,
                "noise": noise,
                "condition": condition,
                "neg_condition": neg_condition,
            }
        )
        return noise

    model.sample = sample
    model.generator_fn = generator_fn
    return {
        "model": model,
        "net": net,
        "teacher": net,
        "student": SimpleNamespace(fps=fps, noise_scheduler=SimpleNamespace(default_student_steps=4)),
        "config": SimpleNamespace(),
        "ctx": {"dtype": model.precision, "device": model.device},
        "has_teacher": True,
        "has_student": True,
        "fps": fps,
    }


def _assert_condition_tree_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, Mapping):
        assert isinstance(expected, Mapping)
        assert set(actual) == set(expected)
        for key in actual:
            _assert_condition_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_condition_tree_equal(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.mark.parametrize("sampler", ["teacher", "student"])
def test_comfyui_text_conditioning_matches_training(sampler):
    bundle = _parity_bundle([4, 2, 2, 2], passthrough=True, fps=24.0)
    prompt = "a paper boat crossing a pond"
    neg_prompt = "blurry"
    (conditioning,) = FastGenPreprocess().preprocess(bundle, prompt, neg_prompt)

    if sampler == "teacher":
        (sampled,) = FastGenTeacherSample().sample(bundle, conditioning, "noise")
        sample_call = bundle["model"].sample_calls[0]
    else:
        (sampled,) = FastGenStudentSample().sample(bundle, conditioning, "noise")
        sample_call = bundle["model"].generator_calls[0]

    training = bundle["model"].prepare_batch(
        {"condition_raw": [prompt], "neg_condition_raw": [neg_prompt]},
    )
    _assert_condition_tree_equal(sample_call["condition"], training["condition"])
    _assert_condition_tree_equal(sample_call["neg_condition"], training["neg_condition"])
    assert "fps" not in sample_call
    assert sampled == {"latent": "noise"}


def test_comfyui_passes_raw_negative_prompt_to_builders_without_text_encoder():
    """Cosmos3-like networks tokenize both prompts themselves, so the negative must not be dropped."""

    class _TokenizerNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocessors = {}

        def build_conditioning(self, data, input_shape=None, precision=None):
            return {
                "condition": {"prompts": data["condition_raw"]},
                "neg_condition": {"prompts": data["neg_condition_raw"]},
            }

    bundle = _parity_bundle([4, 2, 2, 2], passthrough=True)
    net = _TokenizerNet()
    bundle["net"] = bundle["teacher"] = net
    bundle["model"].net_preprocessors = net

    (conditioning,) = FastGenPreprocess().preprocess(bundle, "a paper boat", "blurry")

    assert conditioning["condition"] == {"prompts": ["a paper boat"]}
    assert conditioning["neg_condition"] == {"prompts": ["blurry"]}


@pytest.mark.parametrize(
    ("negative_prompt", "expected"),
    [(None, None), ("", [""]), ("  ", [""]), ("blurry", ["blurry"])],
    ids=["disconnected", "empty", "whitespace", "prompt"],
)
def test_comfyui_negative_prompt_connection_decides_negative_conditioning(negative_prompt, expected):
    """A disconnected negative prompt builds no negative conditioning, like an omitted --neg_prompt_file;
    a connected empty string is the empty negative prompt, like scripts/inference/prompts/empty_prompt.txt."""

    requests = []

    class _RecordingNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocessors = {}

        def build_conditioning(self, data, input_shape=None, precision=None):
            requests.append(data)
            return {"condition": {"prompts": data["condition_raw"]}}

    bundle = _parity_bundle([4, 2, 2, 2], passthrough=True)
    net = _RecordingNet()
    bundle["net"] = bundle["teacher"] = net
    bundle["model"].net_preprocessors = net

    FastGenPreprocess().preprocess(bundle, "a paper boat", negative_prompt)

    assert len(requests) == 1
    assert ("neg_condition_raw" in requests[0]) == (expected is not None)
    assert requests[0].get("neg_condition_raw") == expected


_COMFYUI_CONDITIONING_TASKS = {
    "image-editing": {
        "node": "image",
        "input_shape": [4, 2, 2],
        "required_fields": ("condition_image_raw",),
        "tuple_text": True,
    },
    "image-to-video": {
        "node": "image",
        "input_shape": [4, 2, 2, 2],
        "required_fields": ("real_raw",),
    },
    "video-to-world": {
        "node": "image",
        "input_shape": [4, 2, 2, 2],
        "required_fields": ("real_raw",),
    },
    "camera-image-to-video": {
        "node": "image",
        "input_shape": [4, 2, 2, 2],
        "required_fields": ("real_raw", "camera_poses", "camera_intrinsics"),
        "camera": True,
    },
    "video-to-video": {
        "node": "video",
        "input_shape": [4, 2, 2, 2],
        "required_fields": ("real_raw",),
    },
    "video-to-video-depth": {
        "node": "video",
        "input_shape": [4, 2, 2, 2],
        "required_fields": ("real_raw", "depth_latent"),
        "depth": True,
    },
}


@pytest.mark.parametrize("task", list(_COMFYUI_CONDITIONING_TASKS))
def test_comfyui_task_conditioning_matches_training(task, monkeypatch):
    spec = _COMFYUI_CONDITIONING_TASKS[task]
    bundle = _parity_bundle(
        spec["input_shape"],
        spec["required_fields"],
        tuple_text=spec.get("tuple_text", False),
    )
    prompt = "a paper boat crossing a pond"
    neg_prompt = "blurry"
    condition_image = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
    raw_image = torch.arange(3 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 3, 1, 4, 4)
    raw_video = torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 3, 4, 4)
    depth_latent = torch.arange(2 * 2 * 2 * 2, dtype=torch.float32).reshape(1, 2, 2, 2, 2)
    camera_poses = torch.eye(4).reshape(1, 1, 4, 4)
    camera_intrinsics = torch.tensor([[[1.0, 1.0, 0.5, 0.5]]])

    training_data = {"condition_raw": [prompt], "neg_condition_raw": [neg_prompt]}
    if task == "image-editing":
        training_data["condition_image_raw"] = condition_image
    elif spec["node"] == "image":
        training_data["real_raw"] = raw_image
    else:
        training_data["real_raw"] = raw_video
    if spec.get("depth"):
        training_data["depth_latent"] = depth_latent
    if spec.get("camera"):
        training_data["camera_poses"] = camera_poses
        training_data["camera_intrinsics"] = camera_intrinsics

    if spec["node"] == "image":
        monkeypatch.setattr(nodes_condition, "prepare_source_image_for_net", lambda *_args, **_kwargs: condition_image)
        monkeypatch.setattr(
            nodes_condition,
            "prepare_conditioning_image_for_net",
            lambda *_args, **_kwargs: {"real_raw": raw_image, "cropping_params": {"crop": "metadata"}},
        )
        camera = None
        if spec.get("camera"):
            camera = {"poses": "RAW_POSES", "intrinsics": "RAW_INTRINSICS"}
            monkeypatch.setattr(
                nodes_condition,
                "prepare_camera_poses",
                lambda *_args, **_kwargs: (camera_poses, camera_intrinsics),
            )
        (comfy_conditioning,) = FastGenPreprocess().preprocess(
            bundle,
            prompt,
            neg_prompt,
            image=_IMAGE,
            camera=camera,
        )
    else:
        monkeypatch.setattr(
            nodes_condition,
            "prepare_source_video_for_net",
            lambda *_args, **_kwargs: {
                "real_raw": raw_video,
                "cropping_params": {"crop": "metadata"},
                "source_frame_indices": [0, 1],
            },
        )
        depth_input = None
        if spec.get("depth"):
            depth_input = torch.ones(2, 2, 2, 2)
            monkeypatch.setattr(nodes_condition, "prepare_depth_latent", lambda *_args: depth_latent)
        (comfy_conditioning,) = FastGenPreprocess().preprocess(
            bundle,
            prompt,
            neg_prompt,
            video=torch.zeros(3, 4, 4, 3),
            depth_latent=depth_input,
        )

    training = bundle["model"].prepare_batch(training_data)
    _assert_condition_tree_equal(comfy_conditioning["condition"], training["condition"])
    _assert_condition_tree_equal(comfy_conditioning["neg_condition"], training["neg_condition"])
    if task == "image-editing":
        assert "condition_image" in comfy_conditioning["condition"]
        assert "condition_image_raw" not in comfy_conditioning["condition"]
    assert bundle["net"].preprocessing_modes == [(True, False), (False, False)]


def test_decode_uses_encoder_amp(monkeypatch):
    recorded = {}

    def fake_inference_mode(module, precision_amp, device_type):
        recorded.update(module=module, precision_amp=precision_amp, device_type=device_type)
        return nullcontext()

    vae = SimpleNamespace(decode=lambda latent: latent)
    monkeypatch.setattr(nodes_decode.basic_utils, "inference_mode", fake_inference_mode)
    monkeypatch.setattr(nodes_decode, "save_media", lambda *args, **kwargs: None)
    model = SimpleNamespace(precision_amp_enc=torch.bfloat16, precision_amp_infer=torch.float16)
    bundle = {"model": model, "vae": vae, "net": SimpleNamespace(fps=16), "fps": 16.0}

    result = FastGenDecodeVideo().decode(bundle, {"latent": torch.zeros(1, 3, 2, 2)})

    assert result["result"][0].shape == (1, 2, 2, 3)
    assert recorded == {"module": vae, "precision_amp": torch.bfloat16, "device_type": "cpu"}


def test_decode_treats_single_temporal_frame_as_video(monkeypatch):
    saved = {}

    def fake_save_media(output, path, **kwargs):
        saved.update(output=output, path=path, kwargs=kwargs)

    monkeypatch.setattr(nodes_decode, "save_media", fake_save_media)
    bundle = {"model": SimpleNamespace(), "vae": None, "net": SimpleNamespace(fps=16)}

    result = FastGenDecodeVideo().decode(
        bundle,
        {"latent": torch.zeros(1, 3, 1, 2, 2)},
        vid_format="mp4",
    )

    assert saved["output"].shape == (1, 3, 1, 2, 2)
    assert saved["path"].endswith(".mp4")
    assert saved["kwargs"]["vid_format"] == "mp4"
    assert result["ui"]["animated"] == (True,)
    assert result["result"][0].shape == (1, 2, 2, 3)
    assert result["result"][1].endswith(".mp4")
    assert result["result"][2] is None


@pytest.mark.parametrize("with_audio", [False, True], ids=["video", "audio-video"])
def test_decode_saves_gif_only_without_audio(with_audio, monkeypatch):
    """GIF cannot carry audio, so an audio-video output is saved and reported as MP4."""
    written = []
    monkeypatch.setattr(media_utils, "save_video", lambda video, path, **kwargs: written.append(str(path)))
    monkeypatch.setattr(media_utils, "save_audio_video", lambda video, audio, path, **kwargs: written.append(path))
    latent = torch.zeros(1, 3, 2, 2, 2)
    if with_audio:
        latent = {"video": latent, "audio": torch.zeros(1, 1, 8)}
    bundle = {"model": SimpleNamespace(), "vae": None, "net": SimpleNamespace(fps=16)}

    result = FastGenDecodeVideo().decode(bundle, {"latent": latent}, vid_format="gif")

    suffix = ".mp4" if with_audio else ".gif"
    assert written == [result["result"][1]]
    assert result["result"][1].endswith(suffix)
    assert result["ui"]["images"][0]["filename"].endswith(suffix)


def test_decode_returns_audio_at_the_saved_sample_rate(monkeypatch):
    saved = {}
    monkeypatch.setattr(nodes_decode.basic_utils, "inference_mode", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(nodes_decode, "save_media", lambda output, path, **kwargs: saved.update(kwargs))
    vae = SimpleNamespace(decode=lambda latent: latent, vocoder_sample_rate=48000)
    bundle = {"model": SimpleNamespace(precision_amp_enc=None), "vae": vae, "net": SimpleNamespace(fps=16)}
    latent = {"video": torch.zeros(1, 3, 2, 2, 2), "audio": torch.full((1, 2, 8), 2.0)}

    image, saved_path, audio = FastGenDecodeVideo().decode(bundle, {"latent": latent})["result"]

    assert image.shape == (2, 2, 2, 3)
    assert saved_path.endswith(".mp4")
    assert saved["audio_sample_rate"] == audio["sample_rate"] == 48000
    # Like the saved file, the AUDIO output is clipped to [-1, 1].
    torch.testing.assert_close(audio["waveform"], torch.ones(1, 2, 8))


def test_decode_declares_the_frontend_audio_player():
    # The frontend fills an ``audioUI`` widget from ``audio`` UI results and never sends its value.
    input_types = FastGenDecodeVideo.INPUT_TYPES()
    assert input_types["optional"]["audioUI"] == ("AUDIO_UI", {})
    assert "audioUI" not in input_types["required"]


@pytest.mark.parametrize("shape, channels", [((1, 16), 1), ((1, 2, 16), 2)], ids=["mono", "stereo"])
def test_decode_saves_audio_only_output_as_wav(shape, channels, monkeypatch, tmp_path):
    monkeypatch.setattr(nodes_decode.basic_utils, "inference_mode", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        nodes_decode.folder_paths, "get_save_image_path", lambda prefix, out: (str(tmp_path), "audio", 1, "", prefix)
    )
    vae = SimpleNamespace(decode=lambda latent: torch.zeros(shape), vocoder_sample_rate=16000)
    bundle = {"model": SimpleNamespace(precision_amp_enc=None), "vae": vae, "net": SimpleNamespace(fps=25)}

    result = FastGenDecodeVideo().decode(bundle, {"latent": torch.zeros(1, 8, 4, 16)})

    image, saved_path, audio = result["result"]
    assert image is None
    assert saved_path == str(tmp_path / "audio_00001_.wav")
    with wave.open(saved_path) as stream:
        assert (stream.getnchannels(), stream.getframerate(), stream.getnframes()) == (channels, 16000, 16)
    assert audio["sample_rate"] == 16000
    assert audio["waveform"].shape == (1, channels, 16)
    assert result["ui"] == {"audio": [{"filename": "audio_00001_.wav", "subfolder": "", "type": "output"}]}


@pytest.fixture
def capture_build(monkeypatch):
    """Stub model preprocessing and capture its assembled raw data dict."""
    captured = {}

    def fake_preprocess(model, data, inference_mode=False, materialize_real=True):
        assert inference_mode
        assert not materialize_real
        captured["data"] = dict(data)
        return {**data, "condition": "cond_out", "neg_condition": "neg_out"}

    monkeypatch.setattr(FastGenModel, "prepare_batch", fake_preprocess)
    return captured


_IMAGE = torch.zeros(1, 2, 2, 3)


class TestImageConditionModeSelection:
    def test_image_model_3d_uses_raw_condition_image_key(self, monkeypatch, capture_build):
        monkeypatch.setattr(nodes_condition, "prepare_source_image_for_net", lambda *a, **k: "IMG_REF")
        monkeypatch.setattr(nodes_condition, "prepare_conditioning_image_for_net", _boom)

        out = FastGenPreprocess().preprocess(_make_bundle([32, 128, 128]), "prompt", image=_IMAGE)

        assert out[0]["condition"] == "cond_out"
        assert out[0]["neg_condition"] == "neg_out"
        assert capture_build["data"]["condition_image_raw"] == "IMG_REF"
        assert "condition_image" not in capture_build["data"]
        assert "real_raw" not in capture_build["data"]

    def test_video_model_4d_uses_real_raw(self, monkeypatch, capture_build):
        monkeypatch.setattr(
            nodes_condition,
            "prepare_conditioning_image_for_net",
            lambda *a, **k: {"real_raw": "VID_REF", "cropping_params": {"crop": "metadata"}},
        )
        monkeypatch.setattr(nodes_condition, "prepare_source_image_for_net", _boom)

        out = FastGenPreprocess().preprocess(_make_bundle([16, 21, 60, 104]), "prompt", image=_IMAGE)

        assert out[0]["condition"] == "cond_out"
        assert out[0]["neg_condition"] == "neg_out"
        assert capture_build["data"]["real_raw"] == "VID_REF"
        assert "condition_image_raw" not in capture_build["data"]
        assert "condition_image" not in capture_build["data"]

    def test_multimodal_dict_input_shape_uses_video_path(self, monkeypatch, capture_build):
        # LTX-2: input_shape is a dict keyed by modality; the video stream (4D)
        # must drive the rank check, not len(dict) == 2.
        monkeypatch.setattr(
            nodes_condition,
            "prepare_conditioning_image_for_net",
            lambda *a, **k: {"real_raw": "VID_REF", "cropping_params": {"crop": "metadata"}},
        )
        monkeypatch.setattr(nodes_condition, "prepare_source_image_for_net", _boom)

        shape = {"video": [128, 31, 22, 40], "audio": None}
        out = FastGenPreprocess().preprocess(_make_bundle(shape), "prompt", image=_IMAGE)

        assert out[0]["condition"] == "cond_out"
        assert out[0]["neg_condition"] == "neg_out"
        assert capture_build["data"]["real_raw"] == "VID_REF"

    def test_invalid_image_batch_is_an_error(self, monkeypatch):
        monkeypatch.setattr(FastGenModel, "prepare_batch", _boom)

        with pytest.raises(ValueError, match="Expected one decoded image"):
            FastGenPreprocess().preprocess(
                _make_bundle([16, 21, 60, 104]),
                "prompt",
                image=torch.zeros(2, 2, 2, 3),
            )

    def test_image_and_video_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="either image or video"):
            FastGenPreprocess().preprocess(
                _make_bundle([16, 21, 60, 104]),
                "prompt",
                image=_IMAGE,
                video=torch.zeros(2, 2, 2, 3),
            )


class TestImageConditionCamera:
    def test_camera_data_is_prepared_and_merged(self, monkeypatch, capture_build):
        monkeypatch.setattr(
            nodes_condition,
            "prepare_conditioning_image_for_net",
            lambda *a, **k: {"real_raw": "VID_REF", "cropping_params": {"crop": "metadata"}},
        )
        seen = {}

        def prepare_camera(poses, intrinsics, cropping_params, source_frame_indices):
            seen.update(
                poses=poses,
                intrinsics=intrinsics,
                cropping_params=cropping_params,
                source_frame_indices=source_frame_indices,
            )
            return "POSES", "INTR"

        monkeypatch.setattr(nodes_condition, "prepare_camera_poses", prepare_camera)

        FastGenPreprocess().preprocess(
            _make_bundle([16, 41, 60, 104]),
            "prompt",
            image=_IMAGE,
            camera={"poses": "RAW_POSES", "intrinsics": "RAW_INTR"},
        )

        assert capture_build["data"]["camera_poses"] == "POSES"
        assert capture_build["data"]["camera_intrinsics"] == "INTR"
        assert seen == {
            "poses": "RAW_POSES",
            "intrinsics": "RAW_INTR",
            "cropping_params": {"crop": "metadata"},
            "source_frame_indices": None,
        }

    def test_invalid_camera_object_is_rejected(self):
        with pytest.raises(ValueError, match="Camera Loader"):
            FastGenPreprocess().preprocess(_make_bundle([16, 41, 60, 104]), "prompt", camera={})

    def test_camera_requires_pixel_input_for_crop_metadata(self):
        with pytest.raises(ValueError, match="requires an input image or video"):
            FastGenPreprocess().preprocess(
                _make_bundle([16, 41, 60, 104]),
                "prompt",
                camera={"poses": "POSES", "intrinsics": "INTRINSICS"},
            )


class TestVideoAndDepthPreprocessing:
    @pytest.mark.parametrize("target_fps", [0, -1, float("nan"), float("inf")])
    def test_video_frames_rejects_invalid_target_fps(self, target_fps):
        with pytest.raises(ValueError, match="target_fps must be a finite positive number"):
            nodes_condition._video_frames(torch.zeros(1, 8, 8, 3), target_fps)

    @pytest.mark.parametrize("source_fps", [0, -1, float("nan"), float("inf")])
    def test_video_frames_rejects_invalid_source_fps(self, source_fps):
        frames = torch.zeros(1, 8, 8, 3)
        video = SimpleNamespace(get_components=lambda: SimpleNamespace(images=frames, frame_rate=source_fps))

        with pytest.raises(ValueError, match="source_fps must be a finite positive number"):
            nodes_condition._video_frames(video, target_fps=24)

    @pytest.mark.parametrize(
        ("frame_count", "source_fps", "target_fps", "expected"),
        [(8, 16, 8, [0, 2, 4, 6]), (4, 8, 16, [0, 1, 1, 2, 2, 3, 3])],
    )
    def test_native_video_frames_use_shared_cadence(self, frame_count, source_fps, target_fps, expected):
        frames = torch.arange(frame_count).reshape(frame_count, 1, 1, 1)
        video = SimpleNamespace(get_components=lambda: SimpleNamespace(images=frames, frame_rate=source_fps))

        sampled, indices = nodes_condition._video_frames(video, target_fps)

        assert indices == expected
        assert sampled.flatten().tolist() == expected

    def test_native_video_frames_are_prepared(self, monkeypatch, capture_build):
        frames = torch.zeros(5, 8, 8, 3)
        video = SimpleNamespace(get_components=lambda: SimpleNamespace(images=frames))
        monkeypatch.setattr(
            nodes_condition,
            "prepare_source_video_for_net",
            lambda received, *_args, **_kwargs: {
                "real_raw": "VIDEO",
                "cropping_params": {"crop": "metadata"},
                "source_frame_indices": [0, 1, 2, 3, 4],
            }
            if received is frames
            else _boom(),
        )

        conditioning = FastGenPreprocess().preprocess(
            _make_bundle([16, 5, 8, 8]), "prompt", video=video, num_segments=2, overlap_frames=1
        )[0]

        assert capture_build["data"]["real_raw"] == "VIDEO"
        assert conditioning["num_segments"] == 2
        assert conditioning["overlap_frames"] == 1

    def test_depth_can_be_prepared_without_source_video(self, monkeypatch, capture_build):
        monkeypatch.setattr(nodes_condition, "total_latent_frames", lambda *_args: 7)
        monkeypatch.setattr(nodes_condition, "prepare_depth_latent", lambda depth, count: (depth, count))

        raw_depth = torch.ones(2, 3, 4, 4)
        FastGenPreprocess().preprocess(
            _make_bundle([16, 5, 8, 8]), "prompt", depth_latent=raw_depth, num_segments=2, overlap_frames=3
        )

        depth, count = capture_build["data"]["depth_latent"]
        assert depth is raw_depth
        assert count == 7

    def test_depth_reuses_source_video_latent_frame_count(self, monkeypatch, capture_build):
        counts = {}

        def prepare_video(_frames, _net, _shape, latent_frame_count, _frame_indices=None):
            counts["video"] = latent_frame_count
            return {
                "real_raw": "VIDEO",
                "cropping_params": {"crop": "metadata"},
                "source_frame_indices": list(range(latent_frame_count)),
            }

        def prepare_depth(depth, latent_frame_count):
            counts["depth"] = latent_frame_count
            return depth, latent_frame_count

        monkeypatch.setattr(nodes_condition, "prepare_source_video_for_net", prepare_video)
        monkeypatch.setattr(nodes_condition, "prepare_depth_latent", prepare_depth)

        raw_depth = torch.ones(2, 10, 4, 4)
        FastGenPreprocess().preprocess(
            _make_bundle([16, 5, 8, 8]),
            "prompt",
            video=torch.zeros(5, 8, 8, 3),
            depth_latent=raw_depth,
            num_segments=2,
            overlap_frames=1,
        )

        depth, latent_frame_count = capture_build["data"]["depth_latent"]
        assert depth is raw_depth
        assert latent_frame_count == 9
        assert counts == {"video": 9, "depth": 9}


class TestRawInputLoaders:
    def test_camera_loader_returns_unencoded_camera_data(self, monkeypatch):
        monkeypatch.setattr(nodes_condition, "load_camera_poses", lambda *_args: ("POSES", "INTRINSICS"))

        (camera,) = FastGenCameraLoader().load(camera_poses_path="/poses")

        assert camera == {"poses": "POSES", "intrinsics": "INTRINSICS"}

    @pytest.mark.parametrize("poses_path,video_path", [("", ""), ("/poses", "/video.mp4")])
    def test_camera_loader_requires_exactly_one_source(self, poses_path, video_path):
        with pytest.raises(ValueError, match="exactly one"):
            FastGenCameraLoader().load(poses_path, video_path)

    def test_depth_loader_returns_raw_tensor(self, monkeypatch):
        path = "/depth.pt"
        raw = torch.ones(2, 3, 4, 4)

        def fake_load(received):
            assert received == path
            return raw

        monkeypatch.setattr(nodes_condition, "load_raw_depth_latent", fake_load)

        assert FastGenDepthLatentLoader().load(path) == (raw,)

    def test_depth_loader_rejects_missing_file(self):
        with pytest.raises(FileNotFoundError, match="Depth latent not found"):
            FastGenDepthLatentLoader().load("/missing/depth.pt")


class TestPromptExpandPreprocessFlow:
    def test_preprocess_strips_positive_prompt_like_text_dataloader(self, capture_build):
        FastGenPreprocess().preprocess(_make_bundle([4, 2, 2, 2]), "  a paper boat  \n")

        assert capture_build["data"]["condition_raw"] == ["a paper boat"]

    def test_passthrough_prompt_feeds_preprocessing(self, capture_build):
        (prompt,) = FastGenPromptExpand().expand("a paper boat", "None")

        FastGenPreprocess().preprocess(_make_bundle([4, 2, 2, 2]), prompt)

        assert capture_build["data"]["condition_raw"] == ["a paper boat"]

    def test_expanded_prompt_feeds_preprocessing(self, monkeypatch, capture_build):
        monkeypatch.setattr(nodes_prompt, "expand_prompts_with_qwen", lambda *_args, **_kwargs: ["expanded prompt"])

        (prompt,) = FastGenPromptExpand().expand("short prompt", "Qwen2.5_3B")
        FastGenPreprocess().preprocess(_make_bundle([4, 2, 2, 2]), prompt)

        assert capture_build["data"]["condition_raw"] == ["expanded prompt"]

    def test_vision_language_expansion_receives_the_same_image(self, monkeypatch):
        captured = {}

        def fake_expand(_prompts, _model, _device, _seed, images=None, **_kwargs):
            captured["images"] = images
            return ["expanded prompt"]

        monkeypatch.setattr(
            nodes_prompt, "_image_tensor_to_pil", lambda image: "PIL_IMAGE" if image is _IMAGE else None
        )
        monkeypatch.setattr(nodes_prompt, "expand_prompts_with_qwen", fake_expand)

        assert FastGenPromptExpand().expand("move", "QwenVL2.5_3B", image=_IMAGE) == ("expanded prompt",)
        assert captured["images"] == ["PIL_IMAGE"]


# ═════════════════════════════════════════════════════════════════════════
# FastGenTeacherSample — sample_steps gating (CLI parity)
# ═════════════════════════════════════════════════════════════════════════


def _make_teacher_bundle(fps=24.0):
    """Bundle whose ``model.sample`` records the kwargs it was called with."""
    recorded = {}

    def fake_sample(net, noise, **kwargs):
        recorded["kwargs"] = kwargs
        return "VIDEO_LATENT"

    model = SimpleNamespace(sample=fake_sample, precision_amp_infer=None)
    bundle = {
        "model": model,
        "net": SimpleNamespace(fps=fps),
        "teacher": SimpleNamespace(fps=fps),
        "config": SimpleNamespace(),
        "ctx": {"dtype": torch.float32, "device": torch.device("cpu")},
        "has_teacher": True,
        "fps": fps,
    }
    return bundle, recorded


class TestTeacherSampleSteps:
    def test_zero_uses_model_default(self):
        bundle, recorded = _make_teacher_bundle()
        out = FastGenTeacherSample().sample(bundle, _conditioning(fps=24.0), "noise", sample_steps=0)

        assert "sample_steps" not in recorded["kwargs"], "0 must fall back to the model/scheduler default"
        assert "fps" not in recorded["kwargs"], "the teacher must use its configured default"
        assert out[0] == {"latent": "VIDEO_LATENT"}

    def test_positive_value_is_forwarded(self):
        bundle, recorded = _make_teacher_bundle()
        FastGenTeacherSample().sample(bundle, _conditioning(fps=24.0), "noise", sample_steps=30)

        assert recorded["kwargs"]["sample_steps"] == 30

    def test_fps_input_is_not_exposed(self):
        assert "fps" not in FastGenTeacherSample.INPUT_TYPES()["optional"]

    def test_missing_teacher_raises(self):
        bundle, _ = _make_teacher_bundle()
        bundle["has_teacher"] = False
        with pytest.raises(RuntimeError, match="Teacher network was not loaded"):
            FastGenTeacherSample().sample(bundle, _conditioning(), "noise")


# ═════════════════════════════════════════════════════════════════════════
# FastGenNoiseGenerator.generate
# ═════════════════════════════════════════════════════════════════════════


class TestNoiseGenerator:
    def test_shape_matches_input_shape(self):
        (noise,) = FastGenNoiseGenerator().generate(_make_bundle([4, 8, 8]), seed=1)
        assert tuple(noise.shape) == (1, 4, 8, 8)

    def test_same_seed_is_deterministic(self):
        (a,) = FastGenNoiseGenerator().generate(_make_bundle([4, 8, 8]), seed=7)
        (b,) = FastGenNoiseGenerator().generate(_make_bundle([4, 8, 8]), seed=7)
        assert torch.equal(a, b)

    def test_different_seed_differs(self):
        (a,) = FastGenNoiseGenerator().generate(_make_bundle([4, 8, 8]), seed=1)
        (b,) = FastGenNoiseGenerator().generate(_make_bundle([4, 8, 8]), seed=2)
        assert not torch.equal(a, b)


# ═════════════════════════════════════════════════════════════════════════
# FastGenStudentSample.sample — extrapolation gating
# ═════════════════════════════════════════════════════════════════════════


def _make_student_bundle(with_extrap=True, net_has_vae=True, has_student=True, fps=16):
    """Bundle whose generator fns record which path ran and with what kwargs."""
    recorded = {}

    def gen(net, noise, **kwargs):
        recorded["fn"] = "generator_fn"
        recorded["kwargs"] = kwargs
        return "VIDEO"

    def gen_extrap(net, noise, **kwargs):
        recorded["fn"] = "generator_fn_extrapolation"
        recorded["kwargs"] = kwargs
        return "VIDEO_EXTRAP"

    model = SimpleNamespace(
        config=SimpleNamespace(
            guidance_cfg="GCFG",
            student_sample_type="stype",
        ),
        precision_amp_infer=None,
        generator_fn=gen,
    )
    if with_extrap:
        model.generator_fn_extrapolation = gen_extrap
    net = SimpleNamespace(fps=fps, preprocessors={})
    if net_has_vae:
        net.preprocessors["vae"] = MagicMock()
    bundle = {
        "model": model,
        "net": net,
        "student": SimpleNamespace(fps=fps, noise_scheduler=SimpleNamespace(default_student_steps=4)),
        "config": SimpleNamespace(),
        "ctx": {"dtype": torch.float32, "device": torch.device("cpu")},
        "has_student": has_student,
        "fps": float(fps),
    }
    return bundle, recorded


class TestStudentSample:
    def test_single_pass_uses_generator_fn(self):
        bundle, recorded = _make_student_bundle()
        out = FastGenStudentSample().sample(bundle, _conditioning(), "noise")

        assert recorded["fn"] == "generator_fn"
        assert "num_segments" not in recorded["kwargs"]
        assert recorded["kwargs"] == {"condition": "cond", "neg_condition": "neg"}
        assert out[0] == {"latent": "VIDEO"}

    def test_extrapolation_uses_generator_fn_extrapolation(self):
        bundle, recorded = _make_student_bundle()
        out = FastGenStudentSample().sample(
            bundle,
            _conditioning(num_segments=2, overlap_frames=1),
            "noise",
        )

        assert recorded["fn"] == "generator_fn_extrapolation"
        assert recorded["kwargs"]["num_segments"] == 2
        assert recorded["kwargs"]["overlap_frames"] == 1
        assert recorded["kwargs"]["condition"] == "cond"
        assert "neg_condition" not in recorded["kwargs"]
        assert out[0]["latent"] == "VIDEO_EXTRAP"

    def test_extrapolation_without_method_raises(self):
        bundle, _ = _make_student_bundle(with_extrap=False)
        with pytest.raises(RuntimeError, match="generator_fn_extrapolation.*SimpleNamespace has none"):
            FastGenStudentSample().sample(bundle, _conditioning(num_segments=2), "noise")

    def test_extrapolation_without_vae_raises(self):
        bundle, _ = _make_student_bundle(net_has_vae=False)
        with pytest.raises(RuntimeError, match="VAE is required"):
            FastGenStudentSample().sample(bundle, _conditioning(num_segments=2, overlap_frames=1), "noise")

    def test_missing_student_raises(self):
        bundle, _ = _make_student_bundle(has_student=False)
        with pytest.raises(RuntimeError, match="Student network was not loaded"):
            FastGenStudentSample().sample(bundle, _conditioning(), "noise")


# ═════════════════════════════════════════════════════════════════════════
# IS_CHANGED caching hashes (pure)
# ═════════════════════════════════════════════════════════════════════════


class TestIsChanged:
    def test_config_loader_hash_is_deterministic(self):
        assert FastGenConfigLoader.IS_CHANGED("a/config.py", "o=1", "E=1", "INFO") == FastGenConfigLoader.IS_CHANGED(
            "a/config.py", "o=1", "E=1", "INFO"
        )

    def test_config_loader_hash_changes_with_path(self):
        assert FastGenConfigLoader.IS_CHANGED("a/config.py") != FastGenConfigLoader.IS_CHANGED("b/config.py")

    def test_model_loader_hashes_content_not_key_order(self):
        # sort_keys=True → semantically equal configs hash the same.
        assert FastGenModelLoader.IS_CHANGED({"x": 1, "y": 2}, True, True) == FastGenModelLoader.IS_CHANGED(
            {"y": 2, "x": 1}, True, True
        )

    def test_model_loader_hash_changes_with_flags(self):
        cfg = {"x": 1}
        assert FastGenModelLoader.IS_CHANGED(cfg, True, True) != FastGenModelLoader.IS_CHANGED(cfg, False, True)

    def test_model_loader_ckpt_path_overrides_dropdown(self):
        cfg = {"x": 1}
        # An explicit ckpt_path wins over the dropdown, so ckpt_name is irrelevant.
        h1 = FastGenModelLoader.IS_CHANGED(cfg, True, True, ckpt_name="foo", ckpt_path="/abs/ckpt.pth")
        h2 = FastGenModelLoader.IS_CHANGED(cfg, True, True, ckpt_name="bar", ckpt_path="/abs/ckpt.pth")
        assert h1 == h2

    def test_model_loader_hash_changes_with_config_content(self):
        assert FastGenModelLoader.IS_CHANGED({"x": 1}, True, True) != FastGenModelLoader.IS_CHANGED(
            {"x": 2}, True, True
        )


# ═════════════════════════════════════════════════════════════════════════
# Config loader preparation
# ═════════════════════════════════════════════════════════════════════════


class TestConfigLoader:
    def test_prepares_a_single_process_evaluation_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FASTGEN_REPO", _REPO_ROOT)
        monkeypatch.delenv("FASTGEN_OUTPUT_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)
        credential_paths = []
        monkeypatch.setattr(nodes_config, "set_env_vars", credential_paths.append)
        monkeypatch.setattr(nodes_config, "set_cuda_backend", lambda *args, **kwargs: None)

        (config,) = FastGenConfigLoader().load_config(
            "fastgen/configs/experiments/EDM/config_dmd2_fast.py", config_overrides="trainer.seed=3"
        )

        assert config.log_config.name == "config_dmd2_fast"
        assert config.trainer.seed == 3
        assert config.model.inference and config.model.enable_preprocessors
        assert not config.trainer.ddp and not config.trainer.fsdp
        assert Path(credential_paths[0]).resolve() == (Path(_REPO_ROOT) / "credentials").resolve()
        # The config is imported from the repo, but its run directory resolves in ComfyUI's working directory.
        assert config.trainer.checkpointer.save_dir.startswith(os.path.join(os.getcwd(), "FASTGEN_OUTPUT") + os.sep)


# ═════════════════════════════════════════════════════════════════════════
# Model loader checkpoint selection
# ═════════════════════════════════════════════════════════════════════════


class TestModelLoaderCheckpoint:
    def test_checkpoint_folder_lists_only_pth_files(self):
        _, extensions = nodes_model.folder_paths.folder_names_and_paths["fastgen_checkpoints"]
        assert extensions == {".pth"}

    def test_missing_dropdown_checkpoint_raises(self, monkeypatch):
        monkeypatch.setattr(nodes_model.folder_paths, "get_full_path", lambda folder, name: None)

        with pytest.raises(FileNotFoundError, match="missing.pth"):
            FastGenModelLoader().load_model(None, ckpt_name="missing.pth")

    def test_symlinked_checkpoint_directory_preserves_bundle_path(self, monkeypatch, tmp_path):
        checkpoint_dir = tmp_path / "run" / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        marker = checkpoint_dir / "0001000.pth"
        marker.write_bytes(b"")
        shard_metadata = checkpoint_dir / "0001000.net_model" / ".metadata"
        shard_metadata.parent.mkdir()
        shard_metadata.write_bytes(b"")

        registered_root = tmp_path / "fastgen_checkpoints"
        registered_root.mkdir()
        linked_dir = registered_root / "run"
        linked_dir.symlink_to(checkpoint_dir, target_is_directory=True)
        selected_name = "run/0001000.pth"
        selected_path = registered_root / selected_name
        assert (selected_path.parent / "0001000.net_model" / ".metadata").is_file()
        loaded_paths = []

        class StopAfterCheckpointLoad(Exception):
            pass

        def record_checkpoint_load(checkpointer, model, ckpt_path):
            loaded_paths.append(ckpt_path)
            raise StopAfterCheckpointLoad

        monkeypatch.setattr(nodes_model.folder_paths, "get_full_path", lambda folder, name: str(registered_root / name))
        monkeypatch.setattr(nodes_model, "instantiate", lambda *args, **kwargs: object())
        monkeypatch.setattr(nodes_model, "Checkpointer", lambda config: object())
        monkeypatch.setattr(nodes_model, "load_checkpoint", record_checkpoint_load)
        config = SimpleNamespace(
            model=SimpleNamespace(), model_class=object, trainer=SimpleNamespace(checkpointer=None)
        )

        with pytest.raises(StopAfterCheckpointLoad):
            FastGenModelLoader().load_model(config, ckpt_name=selected_name)

        assert loaded_paths == [str(selected_path)]

    def test_model_is_built_outside_comfyui_inference_mode(self, monkeypatch):
        # ComfyUI runs nodes under torch.inference_mode, where FastGen rejects building and loading modules.
        modes = {}

        def record(name, result):
            def stub(*args, **kwargs):
                modes[name] = torch.is_inference_mode_enabled()
                return result

            return stub

        teacher = SimpleNamespace(sample=lambda *args, **kwargs: None)
        net = SimpleNamespace(preprocessors={})
        model = SimpleNamespace(precision=torch.float32, device=torch.device("cpu"))
        monkeypatch.setattr(nodes_model, "instantiate", record("instantiate", model))
        monkeypatch.setattr(nodes_model, "Checkpointer", lambda config: object())
        monkeypatch.setattr(nodes_model, "load_checkpoint", record("load_checkpoint", 1))
        monkeypatch.setattr(
            nodes_model, "setup_inference_modules", record("setup_inference_modules", (teacher, None, net))
        )
        config = SimpleNamespace(
            model=SimpleNamespace(), model_class=object, trainer=SimpleNamespace(checkpointer=None)
        )

        with torch.inference_mode():
            (bundle,) = FastGenModelLoader().load_model(config, load_teacher=True, load_student=False)

        assert modes == {"instantiate": False, "load_checkpoint": False, "setup_inference_modules": False}
        assert bundle["has_teacher"]


# ═════════════════════════════════════════════════════════════════════════
# _discover_configs (globs the repo; FASTGEN_REPO is set at import time)
# ═════════════════════════════════════════════════════════════════════════


class TestDiscoverConfigs:
    def test_returns_sorted_relative_experiment_configs(self):
        configs = _discover_configs()
        assert configs, "expected to discover experiment configs under the repo"
        assert all(c.startswith("fastgen/configs/experiments/") for c in configs)
        assert all(not os.path.isabs(c) for c in configs)
        assert configs == sorted(configs)


# ═════════════════════════════════════════════════════════════════════════
# Node registration mappings
# ═════════════════════════════════════════════════════════════════════════


class TestNodeMappings:
    def test_class_and_display_keys_match(self):
        assert set(NODE_CLASS_MAPPINGS) == set(NODE_DISPLAY_NAME_MAPPINGS)

    def test_every_node_exposes_required_comfyui_attrs(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            assert callable(getattr(cls, "INPUT_TYPES", None)), f"{name}: INPUT_TYPES not callable"
            assert isinstance(getattr(cls, "RETURN_TYPES", None), tuple), f"{name}: RETURN_TYPES not a tuple"
            assert isinstance(getattr(cls, "CATEGORY", None), str), f"{name}: CATEGORY not a str"
            fn = getattr(cls, "FUNCTION", None)
            assert isinstance(fn, str) and callable(getattr(cls, fn, None)), f"{name}: FUNCTION does not name a method"

    def test_expected_nodes_registered(self):
        expected = {
            "FastGenConfigLoader",
            "FastGenModelLoader",
            "FastGenPromptExpand",
            "FastGenCameraLoader",
            "FastGenDepthLatentLoader",
            "FastGenPreprocess",
            "FastGenNoiseGenerator",
            "FastGenTeacherSample",
            "FastGenStudentSample",
            "FastGenDecodeVideo",
        }
        assert expected <= set(NODE_CLASS_MAPPINGS)
        assert {"FastGenPromptEncode", "FastGenImageCondition", "FastGenVideoCondition"}.isdisjoint(NODE_CLASS_MAPPINGS)

    def test_only_preprocess_produces_conditioning(self):
        assert FastGenPreprocess.RETURN_TYPES == ("FASTGEN_CONDITIONING",)
        assert FastGenCameraLoader.RETURN_TYPES == ("FASTGEN_CAMERA",)
        assert FastGenDepthLatentLoader.RETURN_TYPES == ("FASTGEN_DEPTH_LATENT",)
        input_types = FastGenPreprocess.INPUT_TYPES()
        assert input_types["required"]["model_bundle"][0] == "FASTGEN_MODEL"
        assert input_types["required"]["prompt"][0] == "STRING"
        assert {name: spec[0] for name, spec in input_types["optional"].items()} == {
            "negative_prompt": "STRING",
            "image": "IMAGE",
            "video": "VIDEO",
            "camera": "FASTGEN_CAMERA",
            "depth_latent": "FASTGEN_DEPTH_LATENT",
            "num_segments": "INT",
            "overlap_frames": "INT",
        }
        for cls in (FastGenTeacherSample, FastGenStudentSample):
            required = cls.INPUT_TYPES()["required"]
            assert required["conditioning"][0] == "FASTGEN_CONDITIONING"
            assert "condition" not in required
            assert "neg_condition" not in required

    def test_nodes_use_shared_inference_preparation_helpers(self):
        assert nodes_prompt.expand_prompts_with_qwen is inference_utils.expand_prompts_with_qwen
        assert nodes_condition.prepare_source_image_for_net is inference_utils.prepare_source_image_for_net
        assert nodes_condition.prepare_conditioning_image_for_net is inference_utils.prepare_conditioning_image_for_net
        assert nodes_condition.prepare_source_video_for_net is inference_utils.prepare_source_video_for_net
        assert nodes_condition.load_raw_depth_latent is inference_utils.load_raw_depth_latent
        assert nodes_condition.prepare_depth_latent is inference_utils.prepare_depth_latent
        assert nodes_condition.load_camera_poses is inference_utils.load_camera_poses
        assert nodes_condition.prepare_camera_poses is inference_utils.prepare_camera_poses

    def test_bundled_workflow_matches_single_conditioning_contract(self):
        workflow_path = Path(_REPO_ROOT) / "comfyui" / "workflows" / "QwenImage-LTX2.3.json"
        workflow = json.loads(workflow_path.read_text())
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        relevant_types = {"FastGenPreprocess", "FastGenTeacherSample", "FastGenStudentSample"}

        assert {"FastGenPromptEncode", "FastGenImageCondition", "FastGenVideoCondition"}.isdisjoint(
            node["type"] for node in nodes.values()
        )

        relevant_nodes = [node for node in nodes.values() if node["type"] in relevant_types]
        assert relevant_nodes
        for node in relevant_nodes:
            port_types = [port["type"] for port in (*node.get("inputs", []), *node.get("outputs", []))]
            assert "FASTGEN_CONDITION" not in port_types

            for input_index, input_port in enumerate(node.get("inputs", [])):
                if input_port["link"] is None:
                    continue
                link = links[input_port["link"]]
                assert link[3:5] == [node["id"], input_index]
                assert link[5] == input_port["type"]

            for output_index, output_port in enumerate(node.get("outputs", [])):
                for link_id in output_port.get("links") or []:
                    link = links[link_id]
                    assert link[1:3] == [node["id"], output_index]
                    assert link[5] == output_port["type"]

        for node in relevant_nodes:
            conditioning_inputs = [port for port in node.get("inputs", []) if port["type"] == "FASTGEN_CONDITIONING"]
            conditioning_outputs = [port for port in node.get("outputs", []) if port["type"] == "FASTGEN_CONDITIONING"]
            if node["type"] == "FastGenPreprocess":
                assert conditioning_inputs == []
                assert len(conditioning_outputs) == 1
            else:
                assert len(conditioning_inputs) == 1
                assert conditioning_inputs[0]["name"] == "conditioning"
            if node["type"] == "FastGenPreprocess":
                assert conditioning_outputs[0]["name"] == "conditioning"
                assert any(port["name"] == "prompt" and port["type"] == "STRING" for port in node["inputs"])

        teacher_nodes = [node for node in relevant_nodes if node["type"] == "FastGenTeacherSample"]
        assert teacher_nodes
        for node in teacher_nodes:
            assert [port["name"] for port in node["inputs"]] == [
                "model_bundle",
                "conditioning",
                "noise",
                "config_overrides",
            ]
            assert node["widgets_values"] == [0]

        image_preprocess = next(
            node
            for node in relevant_nodes
            if node["type"] == "FastGenPreprocess"
            and any(port["name"] == "image" and port["link"] is not None for port in node["inputs"])
        )
        vl_expander = next(
            node
            for node in nodes.values()
            if node["type"] == "FastGenPromptExpand"
            and any(port["name"] == "image" and port["link"] is not None for port in node["inputs"])
        )
        preprocess_image_link = next(port["link"] for port in image_preprocess["inputs"] if port["name"] == "image")
        expander_image_link = next(port["link"] for port in vl_expander["inputs"] if port["name"] == "image")
        assert links[preprocess_image_link][1:3] == links[expander_image_link][1:3]

    def test_bundled_workflow_runs_with_default_settings(self):
        workflow_path = Path(_REPO_ROOT) / "comfyui" / "workflows" / "QwenImage-LTX2.3.json"
        workflow = json.loads(workflow_path.read_text())
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}

        def source(node, name):
            link = next(port["link"] for port in node["inputs"] if port["name"] == name)
            return None if link is None else nodes[links[link][1]]

        # Both teachers (QwenImage, LTX-2.3) run CFG, whose unconditional pass needs negative conditioning.
        teachers = [node for node in nodes.values() if node["type"] == "FastGenTeacherSample"]
        assert len(teachers) == 2
        for teacher in teachers:
            assert source(source(teacher, "conditioning"), "negative_prompt") is not None

        # A fresh install must download weights, so LOCAL_FILES_ONLY is left to the user's environment.
        for loader in (node for node in nodes.values() if node["type"] == "FastGenConfigLoader"):
            env_vars = source(loader, "env_vars")
            assert env_vars is None or "LOCAL_FILES_ONLY" not in env_vars["widgets_values"][0]
