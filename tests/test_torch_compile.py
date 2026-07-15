# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import torch
import pytest
from fastgen.methods import DMD2Model
from fastgen.configs.methods.config_sft import ModelConfig as SFTModelConfig
from fastgen.configs.methods.config_dmd2 import ModelConfig as DMD2ModelConfig
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.methods.fine_tuning.sft import SFTModel


def _is_compiled(module):
    # nn.Module.compile() compiles the module in place: it stores the compiled
    # callable on `_compiled_call_impl` rather than replacing the module.
    return getattr(module, "_compiled_call_impl", None) is not None


@pytest.fixture
def sft_model_compiled():
    gc.collect()
    instance = SFTModelConfig()
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=False"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""
    instance.input_shape = [3, 8, 8]
    instance.torch_compile_mode = "default"
    instance.cond_dropout_prob = 0.1
    instance.cond_keys_no_dropout = []
    instance.guidance_scale = None
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
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=False"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""
    instance.input_shape = [3, 8, 8]
    instance.torch_compile_mode = None
    instance.cond_dropout_prob = 0.1
    instance.cond_keys_no_dropout = []
    instance.guidance_scale = None
    model = SFTModel(instance)
    model.on_train_begin()
    model.apply_torch_compile()
    return model


@pytest.fixture
def dmd2_model_compiled():
    gc.collect()
    instance = DMD2ModelConfig()
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""
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
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""
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


def test_compile_excludes_ema(sft_model_not_compiled):
    # EMA networks live in model_dict but are weight-averaged copies that are not run
    # during training, so apply_torch_compile must not compile them.
    model = sft_model_not_compiled
    model.use_ema = ["ema"]
    model.ema = torch.nn.Linear(4, 4)  # any nn.Module suffices for ema_dict/model_dict
    assert "ema" in model.ema_dict and "ema" in model.model_dict

    model.config.torch_compile_mode = "default"
    model.apply_torch_compile()
    assert _is_compiled(model.net)
    assert not _is_compiled(model.ema)


def test_compile_discovers_preprocessor_submodules(sft_model_not_compiled):
    # Preprocessor wrappers (VAE, text/image encoders) are not nn.Modules themselves but
    # hold the actual nn.Module under an attribute; apply_torch_compile must find and
    # compile those submodules.
    model = sft_model_not_compiled

    class _DummyVAEWrapper:  # mimics WanVideoEncoder/SDVAE (not an nn.Module)
        def __init__(self):
            self.vae = torch.nn.Linear(4, 4)
            self.scaling_factor = 0.18  # non-module attributes are ignored

    model.net.vae = _DummyVAEWrapper()
    # An attribute that is itself an nn.Module is compiled directly under its own name.
    model.net.text_encoder = torch.nn.Linear(4, 4)

    model.config.torch_compile_mode = "default"
    model.apply_torch_compile()
    assert _is_compiled(model.net.vae.vae)
    assert _is_compiled(model.net.text_encoder)


def test_sft_compiled_train_step(sft_model_compiled):
    model = sft_model_compiled
    model.init_optimizers()

    batch_size = 1
    labels = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10).float()
    data = {
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": torch.zeros(batch_size, 10).to(model.device, model.precision),
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
    data = {
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": torch.zeros(batch_size, 10).to(model.device, model.precision),
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
