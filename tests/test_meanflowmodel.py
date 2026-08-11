# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
from copy import deepcopy
import gc
import pytest

import torch

from fastgen.methods import MeanFlowModel
from fastgen.configs.experiments.EDM.config_mf_cifar10 import create_config
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.utils.test_utils import check_grad_zero


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    config = create_config()
    instance = config.model
    opts = ["-", "img_resolution=2", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=True"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""  # disable ckpt loading
    instance.input_shape = [3, 2, 2]

    model = MeanFlowModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    neg_labels = torch.zeros(batch_size, 10)  # negative condition (unconditional)

    # Create mock data
    data = {
        "real": torch.randn(batch_size, 3, 2, 2).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_labels.to(model.device, model.precision),
    }
    return model, data


def test_single_train_step_update(get_model_data):
    model, data = get_model_data
    # Run the training step; cifar10 default config
    assert model.config.sample_t_cfg.train_p_mean == -0.6
    assert model.config.sample_t_cfg.train_p_std == 1.6
    assert model.config.sample_t_cfg.flow_matching_ratio == 0.25

    norm_method, *norm_args = model.config.loss_config.norm_method.split("_")
    assert norm_method == "poly"
    assert float(norm_args[0]) == 0.75

    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "mf_loss" in loss_map
    assert "gen_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)


def test_optimizers(get_model_data):
    model, data = get_model_data
    # Test for net optimizer
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        if iteration > 1:
            check_grad_zero(model.net)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)


def test_single_train_step_update_fp32_jvp():
    # Create config and enable fp32 JVP path
    config = create_config()
    instance = config.model
    opts = [
        "-",
        "img_resolution=2",
        "channel_mult=[1]",
        "channel_mult_noise=1",
        "r_timestep=True",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.teacher = deepcopy(instance.net)
    instance.teacher.r_timestep = False
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""
    instance.loss_config.use_cd = True
    instance.precision_amp_jvp = "float32"
    instance.input_shape = [3, 2, 2]

    model = MeanFlowModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    neg_labels = torch.zeros(batch_size, 10)  # negative condition (unconditional)

    data = {
        "real": torch.randn(batch_size, 3, 2, 2).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_labels.to(model.device, model.precision),
    }

    # Run the training step under fp32 JVP setting
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions: same interface and keys as default path
    assert "total_loss" in loss_map
    assert "mf_loss" in loss_map
    assert "gen_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)
