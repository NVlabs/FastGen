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


@pytest.mark.parametrize("cond_dropout_prob, expect_guided", [(None, True), (0.0, True), (1.0, False)])
def test_target_side_guidance_applies_when_no_cond_dropout(cond_dropout_prob, expect_guided):
    """`guidance_scale` must reach the target for every conditional sample.

    No dropout means every sample is conditional, so it must guide exactly like
    `p=0.0`. It used not to: `_drop_condition` returned `keep=None` when
    `cond_dropout_prob is None` and the caller skipped the update entirely --
    computing `guided_dxt_dt` at the cost of an extra forward pass and then
    discarding it, so `guidance_scale` was silently inert on that path.
    (Pre-existing: `_mix_condition` returned early on `cond_dropout_prob is None`
    before the AnyFlow work.) `_drop_condition` now always returns a mask -- all-True
    here -- so the caller has no special case left to forget.

    The net's forward is stubbed to depend only on the condition, so the guided
    velocity is exact rather than initialization-dependent -- EDM's `SongUNet`
    zero-inits `out_conv`, which would otherwise make the cond and uncond passes
    bitwise identical and guidance a provable no-op.
    """
    config = create_config()
    instance = config.model
    opts = ["-", "img_resolution=2", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=True"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.pretrained_model_path = ""
    instance.input_shape = [3, 2, 2]
    instance.cond_dropout_prob = cond_dropout_prob
    instance.guidance_scale = 2.0
    instance.guidance_fuse_scale = None
    model = MeanFlowModel(instance)
    model.on_train_begin()

    batch_size = 4
    real = torch.randn(batch_size, 3, 2, 2, device=model.device, dtype=torch.float32)
    z = torch.randn_like(real)
    t = torch.full((batch_size,), 0.5, device=model.device, dtype=model.net.noise_scheduler.t_precision)
    condition = torch.nn.functional.one_hot(torch.arange(batch_size) % 10, num_classes=10)
    condition = condition.to(model.device, torch.float32)
    neg_condition = torch.zeros(batch_size, 10, device=model.device, dtype=torch.float32)

    # Depends ONLY on the condition: the all-zero neg pass returns exactly 0.
    def condition_only_forward(x_t, t, **kwargs):
        val = kwargs["condition"].sum(dim=1).reshape(-1, 1, 1, 1).to(x_t.dtype)
        return torch.ones_like(x_t) * val

    orig_forward = model.net.forward
    model.net.forward = condition_only_forward
    try:
        cond_out, dxt_dt, _ = model._get_velocity(real, z, t, condition=condition, neg_condition=neg_condition)
    finally:
        model.net.forward = orig_forward

    # neg_dxt_dt == 0, so guided == neg + scale * (plain - neg) == 2 * plain.
    plain = model.net.noise_scheduler.cond_velocity(x=real, eps=z, t=t)
    dropped = (cond_out == neg_condition).all(dim=1)
    assert bool(dropped.all()) is not expect_guided

    expected = 2.0 * plain if expect_guided else plain
    assert torch.allclose(dxt_dt, expected), (dxt_dt - expected).abs().max()


def test_fused_jvp_scaling_is_gated_on_kept_samples():
    """Under `guidance_fuse_scale`, dF/dt must be divided by g only for samples that
    stayed conditional.

    A dropped sample's fused prediction collapses to plain `u_uncond` (the fusion
    self-cancels), so scaling its derivative too would regress it onto
    `v - (t - r) * d(u_uncond)/dt / g` instead of the unconditional MeanFlow identity.
    The AnyFlow reference scales the whole batch
    (`compute_central_difference(..., guidance)` in
    `far/trainers/trainer_wan_anyflow_pretrain.py`); we gate on `keep`.

    The net is stubbed to ignore `condition`, so dropping it changes nothing about the
    raw derivative -- making the 1/g gate the ONLY difference between the two runs.
    """
    g = 3.0
    config = create_config()
    instance = config.model
    opts = ["-", "img_resolution=2", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=True"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.pretrained_model_path = ""
    instance.input_shape = [3, 2, 2]
    instance.guidance_fuse_scale = g
    instance.loss_config.use_jvp_finite_diff = True
    instance.sample_t_cfg.deterministic_buckets = False
    model = MeanFlowModel(instance)
    model.on_train_begin()

    batch_size = 4
    real = torch.randn(batch_size, 3, 2, 2, device=model.device, dtype=torch.float32)
    t_prec = model.net.noise_scheduler.t_precision
    t = torch.full((batch_size,), 0.6, device=model.device, dtype=t_prec)
    r = torch.full((batch_size,), 0.2, device=model.device, dtype=t_prec)
    condition = torch.nn.functional.one_hot(torch.arange(batch_size) % 10, num_classes=10)
    condition = condition.to(model.device, torch.float32)
    neg_condition = torch.zeros(batch_size, 10, device=model.device, dtype=torch.float32)

    # `+ 0.0 * param.sum()` leaves the value untouched but ties the output to the
    # autograd graph: `_mf_pred_to_loss` asserts
    # `u_theta.requires_grad is torch.is_grad_enabled()`. The JVP runs under `_jvp`'s
    # `@torch.no_grad()`, so `u_theta_jvp` stays grad-free as that code also asserts.
    param = next(model.net.parameters())

    def condition_independent_forward(x_t, t, **kwargs):
        tt = t.reshape(-1, *([1] * (x_t.ndim - 1))).to(x_t.dtype)
        return x_t * (1.0 + tt) + 0.0 * param.sum().to(x_t.dtype)

    orig_forward = model.net.forward

    def run(cond_dropout_prob):
        model.config.cond_dropout_prob = cond_dropout_prob
        model.net.forward = condition_independent_forward
        try:
            torch.manual_seed(11)
            return model._compute_mf_loss(
                real_data=real,
                t=t,
                r=r,
                iteration=0,
                condition=condition,
                neg_condition=neg_condition,
            )[2]
        finally:
            model.net.forward = orig_forward

    jvp_kept = run(0.0)  # every sample conditional -> every derivative divided by g
    jvp_dropped = run(1.0)  # every sample unconditional -> none divided

    assert torch.allclose(jvp_dropped, g * jvp_kept, atol=1e-5, rtol=1e-4), (jvp_dropped - g * jvp_kept).abs().max()
    assert not torch.allclose(jvp_dropped, jvp_kept, atol=1e-6)
