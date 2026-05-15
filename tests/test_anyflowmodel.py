# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AnyFlowModel.

The tests run on the tiny EDM backbone with ``r_timestep=True`` (same trick
``test_meanflowmodel.py`` uses) so they execute on CPU without downloading
any pretrained weights.
"""

import gc

import pytest
import torch

from fastgen.configs.config_utils import override_config_with_opts
from fastgen.configs.methods.config_anyflow import ModelConfig
from fastgen.methods import AnyFlowModel
from fastgen.methods.distribution_matching.anyflow_scheduler import FlowMapDiscreteScheduler
from fastgen.utils.test_utils import check_grad_zero


def _build_pretrain_model():
    gc.collect()
    instance = ModelConfig()
    instance.loss_config.training_stage = "pretrain"
    # Use a small finite-difference step relative to t in [0, 1].
    instance.loss_config.jvp_finite_diff_eps = 1e-2

    opts = ["-", "img_resolution=2", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=True"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""
    instance.input_shape = [3, 2, 2]

    model = AnyFlowModel(instance)
    model.on_train_begin()
    model.init_optimizers()
    return model


def _build_onpolicy_model():
    """On-policy fixture mirrors test_dmd2model: img_resolution=8 so the
    discriminator's 4x4 conv kernels can operate."""
    gc.collect()
    instance = ModelConfig()
    instance.loss_config.training_stage = "onpolicy"

    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=True"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_disc = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_disc)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""
    instance.student_update_freq = 2
    instance.input_shape = [3, 8, 8]

    model = AnyFlowModel(instance)
    model.on_train_begin()
    model.init_optimizers()
    return model


def _make_data(model, img_resolution: int = 2):
    batch_size = 1
    labels = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10)
    neg_labels = torch.zeros(batch_size, 10)
    return {
        "real": torch.randn(batch_size, 3, img_resolution, img_resolution).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_labels.to(model.device, model.precision),
    }


# ---------------------------------------------------------------------------
# Pretrain stage
# ---------------------------------------------------------------------------


def test_pretrain_single_train_step():
    model = _build_pretrain_model()
    data = _make_data(model)

    loss_map, outputs = model.single_train_step(data, 0)

    assert "total_loss" in loss_map
    assert "anyflow_loss" in loss_map
    assert "dF_dt_target_norm" in loss_map
    assert torch.isfinite(loss_map["total_loss"]).all()
    assert "gen_rand" in outputs
    assert isinstance(outputs["gen_rand"], torch.Tensor)


def test_pretrain_no_fake_score_or_discriminator():
    """Pretrain stage must not instantiate DMD2's fake_score / discriminator."""
    model = _build_pretrain_model()
    assert not hasattr(model, "fake_score") or model.fake_score is None or "fake_score" not in model.model_dict
    assert "fake_score" not in model.model_dict
    assert "discriminator" not in model.model_dict


def test_pretrain_optimizer_step():
    model = _build_pretrain_model()
    data = _make_data(model)
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)
    # After one zero_grad with no backward in between, gradients should be cleared.
    model.optimizers_zero_grad(2)
    check_grad_zero(model.net)


def test_pretrain_central_difference_falls_back_at_boundaries():
    """When (t ± δ) leaves [min_t, max_t], the one-sided fallback should still
    produce a finite target. We synthesise a worst-case t at the boundary."""
    model = _build_pretrain_model()
    ns = model.net.noise_scheduler

    real = torch.randn(2, 3, 2, 2, device=model.device, dtype=model.precision)
    cond = torch.nn.functional.one_hot(torch.tensor([0, 1]), num_classes=10).to(model.device, model.precision)

    t = torch.tensor([float(ns.min_t), float(ns.max_t)], device=model.device, dtype=ns.t_precision)
    r = torch.tensor([float(ns.min_t), float(ns.min_t)], device=model.device, dtype=ns.t_precision)

    eps_noise = torch.randn_like(real)
    x_t = ns.forward_process(real, eps_noise, t)
    v = eps_noise - real

    target = model._compute_central_difference_target(x_t, t, r, v, cond)
    assert torch.isfinite(target).all(), "boundary samples must yield finite targets"


# ---------------------------------------------------------------------------
# On-policy stage — inherits DMD2's alternating updates
# ---------------------------------------------------------------------------


def test_onpolicy_student_update_step():
    model = _build_onpolicy_model()
    data = _make_data(model, img_resolution=8)
    loss_map, outputs = model.single_train_step(data, 0)  # iteration 0 -> student update
    assert "total_loss" in loss_map
    assert "vsd_loss" in loss_map
    assert "gen_rand" in outputs


def test_onpolicy_fake_score_discriminator_update_step():
    model = _build_onpolicy_model()
    model.precision = torch.float32
    model.on_train_begin()
    data = _make_data(model, img_resolution=8)
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(model.precision)
    loss_map, outputs = model.single_train_step(data, 1)  # iteration 1 -> fake_score/disc update
    assert "fake_score_loss" in loss_map
    assert "gan_loss_disc" in loss_map
    assert "gen_rand" in outputs


def test_onpolicy_optimizer_step():
    model = _build_onpolicy_model()
    data = _make_data(model, img_resolution=8)
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_flowmap_scheduler_apply_shift_identity_for_shift1():
    scheduler = FlowMapDiscreteScheduler(num_train_timesteps=1000, shift=1.0, weight_type="beta08")
    sigmas = torch.linspace(0.0, 1.0, 11)
    assert torch.allclose(scheduler.apply_shift(sigmas), sigmas)


def test_flowmap_scheduler_step_zero_interval():
    scheduler = FlowMapDiscreteScheduler(num_train_timesteps=1000, shift=1.0, weight_type="uniform")
    sample = torch.randn(2, 4, 8, 8)
    model_output = torch.randn_like(sample)
    t = torch.tensor([500.0, 500.0])
    # r = t => zero-length integration interval; the sample should be unchanged.
    out = scheduler.step(model_output, sample, timestep=t, r_timestep=t.clone())
    assert torch.allclose(out, sample, atol=1e-5)


@pytest.mark.parametrize("weight_type", ["gaussian", "beta08", "uniform"])
def test_flowmap_scheduler_weights_positive(weight_type):
    scheduler = FlowMapDiscreteScheduler(num_train_timesteps=1000, shift=1.0, weight_type=weight_type)
    t = torch.tensor([100.0, 500.0, 900.0])
    w = scheduler.get_train_weight(t)
    assert torch.all(w >= 0), f"{weight_type} weights must be non-negative"
    assert torch.isfinite(w).all()
