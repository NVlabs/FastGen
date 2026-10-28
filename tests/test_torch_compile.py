# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import subprocess
import sys
import textwrap

import pytest
import torch
from fastgen.methods import DMD2Model
from fastgen.configs.methods.config_sft import ModelConfig as SFTModelConfig
from fastgen.configs.methods.config_dmd2 import ModelConfig as DMD2ModelConfig
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.methods.fine_tuning.sft import SFTModel
from fastgen.networks.network import FastGenModule
from tests.utils import RunIf


def _is_compiled(module):
    # nn.Module.compile() stores the compiled callable on _compiled_call_impl (PyTorch internal).
    return getattr(module, "_compiled_call_impl", None) is not None


@pytest.mark.parametrize(("tf32_matmul", "tf32_conv"), [(False, True), (True, False)])
@RunIf(min_gpus=1)
def test_compile_backend_tf32_api_is_inductor_compatible(tf32_matmul, tf32_conv, tmp_path):
    """Exercise the maintained Inductor readers of both legacy TF32 properties."""

    program = textwrap.dedent(
        f"""
        import warnings

        import torch

        from fastgen.utils.scripts import set_cuda_backend

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            set_cuda_backend(tf32_matmul={tf32_matmul!r}, tf32_conv={tf32_conv!r}, torch_compile_enabled=True)
        assert not any(
            str(warning.message).startswith("Please use the new API settings to control TF32 behavior")
            for warning in caught
        )
        assert torch.backends.cudnn.allow_tf32 is {tf32_conv!r}
        assert torch.backends.cuda.matmul.allow_tf32 is {tf32_matmul!r}

        class BmmConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 4, 3, padding=1, device="cuda", dtype=torch.float32)

            def forward(self, left, right, image):
                return torch.bmm(left, right), self.conv(image)

        model = BmmConv().eval()
        # FP32 bmm reaches compile_fx._warn_tf32_disabled(), which reads
        # cuda.matmul.allow_tf32. Max-autotune adds Triton convolution choices,
        # whose lowering reads cudnn.allow_tf32.
        model.compile(mode="max-autotune-no-cudagraphs")
        with torch.no_grad():
            outputs = model(
                torch.randn(2, 8, 16, device="cuda", dtype=torch.float32),
                torch.randn(2, 16, 8, device="cuda", dtype=torch.float32),
                torch.randn(2, 4, 16, 16, device="cuda", dtype=torch.float32),
            )
        assert all(torch.isfinite(output).all() for output in outputs)
        torch.cuda.synchronize()
        """
    )
    env = os.environ.copy()
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path)
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=180, env=env)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def sft_model_compiled():
    gc.collect()
    instance = SFTModelConfig()
    opts = [
        "-",
        "backbone.img_resolution=8",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=False",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.input_shape = [3, 8, 8]
    instance.torch_compile_mode = "default"
    instance.cond_dropout_prob = 0.1
    instance.cond_keys_no_dropout = []
    model = SFTModel(instance)
    # Mirror the trainer order: on_train_begin() moves parameters to device/dtype and
    # initialises preprocessors; apply_torch_compile() must come after so that
    # preprocessors exist and are on the right device when compiled.
    model.on_train_begin()
    model.apply_torch_compile()
    return model


@pytest.fixture
def sft_model_not_compiled():
    gc.collect()
    instance = SFTModelConfig()
    opts = [
        "-",
        "backbone.img_resolution=8",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=False",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.input_shape = [3, 8, 8]
    instance.torch_compile_mode = None
    instance.cond_dropout_prob = 0.1
    instance.cond_keys_no_dropout = []
    model = SFTModel(instance)
    model.on_train_begin()
    model.apply_torch_compile()
    return model


@pytest.fixture
def dmd2_model_compiled():
    gc.collect()
    instance = DMD2ModelConfig()
    opts = ["-", "backbone.img_resolution=8", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.student_update_freq = 2
    instance.input_shape = [3, 8, 8]
    instance.torch_compile_mode = "default"
    model = DMD2Model(instance)
    model.on_train_begin()
    model.apply_torch_compile()
    return model


@pytest.fixture
def dmd2_model_not_compiled():
    gc.collect()
    instance = DMD2ModelConfig()
    opts = ["-", "backbone.img_resolution=8", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.student_update_freq = 2
    instance.input_shape = [3, 8, 8]
    instance.torch_compile_mode = None
    model = DMD2Model(instance)
    model.on_train_begin()
    model.apply_torch_compile()
    return model


def test_default_torch_compile_mode_is_none():
    from fastgen.configs.config import BaseModelConfig

    config = BaseModelConfig()
    assert config.torch_compile_mode is None


def test_apply_torch_compile_is_idempotent(sft_model_not_compiled):
    # Guard in apply_torch_compile relies on _compiled_call_impl (PyTorch internal).
    # This test documents that contract: a second call must not re-compile the module.
    model = sft_model_not_compiled
    model.config.torch_compile_mode = "default"
    model.apply_torch_compile()
    compiled_impl = getattr(model.net, "_compiled_call_impl", None)
    assert compiled_impl is not None

    model.apply_torch_compile()
    assert getattr(model.net, "_compiled_call_impl", None) is compiled_impl


def test_sft_compile_enabled(sft_model_compiled):
    assert _is_compiled(sft_model_compiled.net)


def test_sft_compile_disabled(sft_model_not_compiled):
    assert not _is_compiled(sft_model_not_compiled.net)


def test_dmd2_compile_enabled(dmd2_model_compiled):
    # apply_torch_compile draws from model_dict (net, fake_score, discriminator) plus the teacher.
    assert _is_compiled(dmd2_model_compiled.net)
    assert _is_compiled(dmd2_model_compiled.teacher)
    assert _is_compiled(dmd2_model_compiled.fake_score)
    assert _is_compiled(dmd2_model_compiled.discriminator)


def test_dmd2_compile_disabled(dmd2_model_not_compiled):
    assert not _is_compiled(dmd2_model_not_compiled.net)
    assert not _is_compiled(dmd2_model_not_compiled.teacher)
    assert not _is_compiled(dmd2_model_not_compiled.fake_score)
    assert not _is_compiled(dmd2_model_not_compiled.discriminator)


def test_compile_excludes_ema_during_training(sft_model_not_compiled):
    # EMA networks live in model_dict but are weight-averaged copies that are not run
    # during training, so apply_torch_compile must not compile them.
    model = sft_model_not_compiled
    assert not model.config.inference
    model.use_ema = ["ema"]
    model.ema = torch.nn.Linear(4, 4)  # any nn.Module suffices for ema_dict/model_dict
    assert "ema" in model.ema_dict and "ema" in model.model_dict

    model.config.torch_compile_mode = "default"
    model.apply_torch_compile()
    assert _is_compiled(model.net)
    assert not _is_compiled(model.ema)


def test_compile_includes_ema_student_for_inference(sft_model_not_compiled):
    # In the real inference flow setup_inference_modules nulls unused nets before
    # apply_torch_compile, reducing model_dict to the running student (use_ema[0]).
    # We emulate that pruning by setting net to None; apply_torch_compile then compiles
    # the surviving EMA student.
    model = sft_model_not_compiled
    model.config.inference = True
    model.use_ema = ["ema"]
    model.ema = model.net
    model.net = None  # emulate setup_inference_modules pruning the unused base net

    model.config.torch_compile_mode = "default"
    model.apply_torch_compile()
    assert _is_compiled(model.ema)


def test_compile_targets_preprocessor_roots(sft_model_not_compiled):
    # Preprocessor modules live outside the generator's registered tree, so compilation
    # discovers their roots through the plain registry.
    model = sft_model_not_compiled

    class _DummyVAEWrapper(FastGenModule):
        def __init__(self):
            super().__init__()
            self.vae = torch.nn.Linear(4, 4)
            self.scaling_factor = 0.18  # non-module attributes are ignored

        def forward(self, value):
            return self.vae(value)

    class _DummyTextEncoder(FastGenModule):
        def forward(self, value):
            return value

    model.net.preprocessors["vae"] = _DummyVAEWrapper()
    model.net.preprocessors["text_encoder"] = _DummyTextEncoder()

    model.config.torch_compile_mode = "default"
    model.apply_torch_compile()
    assert _is_compiled(model.net.preprocessors["vae"])
    assert not _is_compiled(model.net.preprocessors["vae"].vae)
    assert _is_compiled(model.net.preprocessors["text_encoder"])


def test_preprocessor_semantic_api_uses_module_call_boundary():
    class _Preprocessor(FastGenModule):
        def encode(self, value):
            return self(value)

        def forward(self, value):
            return value + 1

    preprocessor = _Preprocessor()
    calls = []
    preprocessor.register_forward_pre_hook(lambda *_: calls.append(True))

    assert torch.equal(preprocessor.encode(torch.tensor(1)), torch.tensor(2))
    assert calls == [True]


def test_compile_rejects_missing_preprocessor_host(sft_model_not_compiled):
    # Inference pruning may discard the base net only when another selected network
    # survives to host preprocessors. A model with no surviving network is invalid.
    model = sft_model_not_compiled
    model.net = None

    assert model.model_dict["net"] is None

    model.config.torch_compile_mode = "default"
    with pytest.raises(RuntimeError, match="no network available to own preprocessors"):
        model.apply_torch_compile()


def test_sft_compiled_train_step(sft_model_compiled):
    model = sft_model_compiled
    model.init_optimizers()

    batch_size = 1
    labels = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10).float()
    # EDM takes its conditioning in FP32, as its ``build_conditioning`` returns it.
    data = {
        "idx": list(range(batch_size)),
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, torch.float32),
        "neg_condition": torch.zeros(batch_size, 10).to(model.device, torch.float32),
    }

    loss_map, _ = model.single_train_step(data, 0)
    assert "total_loss" in loss_map
    assert not torch.isnan(loss_map["total_loss"])
    loss_map["total_loss"].backward()


def test_dmd2_compiled_train_step(dmd2_model_compiled):
    model = dmd2_model_compiled
    model.init_optimizers()

    batch_size = 1
    labels = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10)
    # EDM takes its conditioning in FP32, as its ``build_conditioning`` returns it.
    data = {
        "idx": list(range(batch_size)),
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, torch.float32),
        "neg_condition": torch.zeros(batch_size, 10).to(model.device, torch.float32),
    }

    # Student update step
    loss_map, _ = model.single_train_step(data, 0)
    assert "total_loss" in loss_map
    assert not torch.isnan(loss_map["total_loss"])

    # Fake score update step
    model.optimizers_zero_grad(1)
    loss_map, _ = model.single_train_step(data, 1)
    assert "total_loss" in loss_map
    assert not torch.isnan(loss_map["total_loss"])
