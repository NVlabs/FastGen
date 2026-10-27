# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Callable
import gc
import pytest

import torch

import fastgen.methods.trajectory_matching.CM as cm_module
import fastgen.utils.tensor as tensor_module
from fastgen.methods import CMModel
from fastgen.methods.trajectory_matching.CM import ode_solver
from fastgen.configs.methods.config_cm import ModelConfig
from fastgen.configs.config_utils import override_config_with_opts
from tests.model_utils import StudentAsTeacher
from tests.utils import assert_autocast_weight_cache_neutral, check_grad_zero


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    instance = ModelConfig()
    opts = ["-", "backbone.img_resolution=2", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.input_shape = [3, 2, 2]

    model = CMModel(instance)
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
    # Run the training step
    assert model.config.sample_t_cfg.train_p_mean == -1.1
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "cm_loss" in loss_map
    assert "unweighted_cm_loss" in loss_map
    assert "gen_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)


def test_optimizers(get_model_data):
    model, data = get_model_data
    # Test for net optimizer
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        if iteration > 0:
            check_grad_zero(model.net)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)


def _fp32_cd_model_with_student_teacher():
    gc.collect()
    instance = ModelConfig()
    opts = ["-", "backbone.img_resolution=2", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float32"  # the autocast weight cache only stores fp32 trainable weights
    instance.loss_config.use_cd = True
    instance.pretrained_model_path = None
    instance.input_shape = [3, 2, 2]
    model = CMModel(instance)
    model.on_train_begin()
    model.teacher = StudentAsTeacher(model.net)
    return model


def test_cd_teacher_pass_keeps_student_gradients_under_autocast_cache():
    """The no_grad consistency-distillation teacher step precedes the student's gradient forward. With
    a teacher view over the student's own weights it must not leave detached autocast casts for that
    forward to reuse: the same parameters must receive gradients as in plain fp32, with values independent
    of the autocast weight cache."""
    model = _fp32_cd_model_with_student_teacher()
    batch_size = 1
    real_data = torch.randn(batch_size, 3, 2, 2, device=model.device)
    labels = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10)
    condition = labels.to(model.device, torch.float32)
    neg_condition = torch.zeros(batch_size, 10, device=model.device)
    t = model.net.noise_scheduler.sample_t(batch_size, **model.sample_t_cfg.sample_t_kwargs(), device=model.device)

    def run_loss():
        cm_loss, _, _ = model._compute_cm_loss(
            model.net, real_data=real_data, t=t, condition=condition, neg_condition=neg_condition
        )
        return cm_loss.mean()

    assert_autocast_weight_cache_neutral(model.net, run_loss, model.device.type)


@pytest.mark.parametrize("pred_type", [None, "x0", "eps", "flow"])
def test_ode_solver_is_the_euler_step_on_edm_for_every_teacher_pred_type(get_model_data, pred_type):
    """On EDM, where alpha = 1 and sigma = t, the DDIM step is the Euler step ``x_t - (t - t_next) flow``."""
    model, data = get_model_data
    x_t = torch.randn_like(data["real"])
    t = torch.full((1,), 2.0, device=model.device, dtype=torch.float64)
    t_next = torch.full((1,), 1.5, device=model.device, dtype=torch.float64)
    with torch.no_grad():
        flow = model.net(x_t, t, condition=data["condition"], fwd_pred_type="flow")

    x_next = ode_solver(model.net, x_t, t, t_next, condition=data["condition"], pred_type=pred_type)

    tolerance = 1e-5 if model.device.type == "cpu" else 3e-2
    euler = x_t - (t - t_next).view(-1, 1, 1, 1) * flow
    torch.testing.assert_close(x_next, euler.to(x_next.dtype), rtol=tolerance, atol=tolerance)


def test_cd_teacher_steps_on_the_configured_prediction_type(monkeypatch):
    model = _fp32_cd_model_with_student_teacher()
    model.config.teacher_pred_type = "eps"
    seen_pred_types = []

    def record_ode_solver(*args, **kwargs):
        seen_pred_types.append(kwargs["pred_type"])
        return ode_solver(*args, **kwargs)

    monkeypatch.setattr(cm_module, "ode_solver", record_ode_solver)
    real_data = torch.randn(1, 3, 2, 2, device=model.device)
    condition = torch.nn.functional.one_hot(torch.tensor([3]), num_classes=10).to(model.device, torch.float32)
    t = model.net.noise_scheduler.sample_t(1, **model.sample_t_cfg.sample_t_kwargs(), device=model.device)

    model._compute_cm_loss(model.net, real_data=real_data, t=t, condition=condition)

    assert seen_pred_types == ["eps"]


def test_cm_weighting_is_per_sample_in_the_time_precision(monkeypatch):
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)
    model = object.__new__(CMModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.loss_config = SimpleNamespace(huber_const=0.0, use_squared_l2=False, weighting_ct_loss="default")
    t = torch.tensor([0.5], dtype=torch.float64)

    # The float32 states give the distance; a time step of 1e-9 survives only in float64 times.
    cm_loss, _ = model._pred_to_loss(None, torch.ones(1, 4), torch.zeros(1, 4), t, t - 1e-9)

    assert cm_loss.dtype == torch.float64
    torch.testing.assert_close(cm_loss, torch.tensor([2.0 / 1e-9], dtype=torch.float64), rtol=1e-6, atol=0)


def test_cm_promotes_before_subtracting_low_precision_predictions(monkeypatch):
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)
    model = object.__new__(CMModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.loss_config = SimpleNamespace(huber_const=0.0, use_squared_l2=False, weighting_ct_loss="one")
    prediction = torch.tensor([[60000.0]], dtype=torch.float16, requires_grad=True)
    target = -prediction.detach()
    t = torch.ones(1)

    loss, _ = model._pred_to_loss(None, prediction, target, t, t / 2)

    torch.testing.assert_close(loss, torch.tensor([120000.0]))
    loss.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.ones_like(prediction))
