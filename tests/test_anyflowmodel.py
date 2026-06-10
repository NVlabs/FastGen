# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AnyFlow.

The pretrain stage is MeanFlow with AnyFlow's hyperparameters (fixed
per-timestep loss weighting, finite-difference JVP, consistency bucket), so
the pretrain tests drive :class:`MeanFlowModel` directly. The on-policy tests
exercise :class:`AnyFlowModel` (DMD2 with a multi-step rollout-with-gradient
student). Both run on the tiny EDM backbone with ``r_timestep=True`` and
``schedule_type=rf`` so they execute on CPU without pretrained weights.
"""

import gc
import types

import pytest
import torch

from fastgen.configs.config_utils import override_config_with_opts
from fastgen.configs.methods.config_anyflow import ModelConfig as AnyFlowModelConfig
from fastgen.configs.methods.config_mean_flow import ModelConfig as MeanFlowModelConfig
from fastgen.methods import AnyFlowModel, MeanFlowModel
from fastgen.utils.test_utils import check_grad_zero


def _build_pretrain_model(weight_type="beta08", consistency_ratio=0.25, r_sample_ratio=0.5):
    """MeanFlow configured the AnyFlow way (paper Stage 2)."""
    gc.collect()
    instance = MeanFlowModelConfig()

    instance.loss_config.loss_type = "l2"
    instance.loss_config.weight_type = weight_type
    instance.loss_config.use_jvp_finite_diff = True
    instance.loss_config.jvp_finite_diff_eps = 1e-2

    instance.sample_t_cfg.time_dist_type = "shifted"
    instance.sample_t_cfg.shift = 5.0
    instance.sample_t_cfg.min_t = 0.001
    instance.sample_t_cfg.max_t = 0.999
    instance.sample_t_cfg.r_sample_ratio = r_sample_ratio
    instance.sample_t_cfg.consistency_ratio = consistency_ratio

    opts = ["-", "img_resolution=2", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=True", "+schedule_type=rf"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""
    instance.input_shape = [3, 2, 2]

    model = MeanFlowModel(instance)
    model.on_train_begin()
    model.init_optimizers()
    return model


def _build_onpolicy_model():
    """On-policy fixture mirrors test_dmd2model: img_resolution=8 so the
    discriminator's 4x4 conv kernels can operate."""
    gc.collect()
    instance = AnyFlowModelConfig()

    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=True", "+schedule_type=rf"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_disc = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_disc)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""
    instance.student_update_freq = 2
    # student_sample_steps=2 is the smallest value that runs the rollout loop
    # more than once.
    instance.student_sample_steps = 2
    instance.input_shape = [3, 8, 8]

    model = AnyFlowModel(instance)
    model.on_train_begin()
    model.init_optimizers()
    return model


def _make_data(model, img_resolution: int = 2, batch_size: int = 1):
    labels = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10)
    neg_labels = torch.zeros(batch_size, 10)
    return {
        "real": torch.randn(batch_size, 3, img_resolution, img_resolution).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_labels.to(model.device, model.precision),
    }


# ---------------------------------------------------------------------------
# Pretrain stage — MeanFlow with AnyFlow options
# ---------------------------------------------------------------------------


def test_pretrain_single_train_step():
    model = _build_pretrain_model()
    data = _make_data(model)

    loss_map, outputs = model.single_train_step(data, 0)

    assert "total_loss" in loss_map
    assert "mf_loss" in loss_map
    assert torch.isfinite(loss_map["total_loss"]).all()
    assert "gen_rand" in outputs


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


def test_pretrain_finite_difference_falls_back_at_boundaries():
    """When (t ± eps) leaves [min_t, max_t], the one-sided fallback should
    still produce a finite JVP estimate. We synthesise worst-case boundary t."""
    model = _build_pretrain_model()
    ns = model.net.noise_scheduler

    real = torch.randn(2, 3, 2, 2, device=model.device, dtype=model.precision)
    cond = torch.nn.functional.one_hot(torch.tensor([0, 1]), num_classes=10).to(model.device, model.precision)

    t = torch.tensor([float(ns.min_t), float(ns.max_t)], device=model.device, dtype=ns.t_precision)
    r = torch.tensor([float(ns.min_t), float(ns.min_t)], device=model.device, dtype=ns.t_precision)

    eps_noise = torch.randn_like(real)
    x_t = ns.forward_process(real, eps_noise, t)
    dxt_dt = eps_noise - real

    u_theta_jvp = model._jvp(x_t, t, r, dxt_dt, condition=cond)
    assert torch.isfinite(u_theta_jvp).all(), "boundary samples must yield finite JVP estimates"


def test_pretrain_consistency_bucket_pins_r_to_min_t():
    """With consistency_ratio=1.0 (and no flow-matching head), every sample's
    r must be pinned to min_t."""
    model = _build_pretrain_model(consistency_ratio=1.0, r_sample_ratio=1.0)
    data = _make_data(model, batch_size=4)

    captured = {}
    orig = model._compute_mf_loss

    def spy(real_data, t, r, **kwargs):
        captured["r"] = r
        return orig(real_data=real_data, t=t, r=r, **kwargs)

    model._compute_mf_loss = spy
    model.single_train_step(data, 0)

    min_t = float(model.net.noise_scheduler.min_t)
    assert torch.allclose(captured["r"].float(), torch.full_like(captured["r"].float(), min_t))


@pytest.mark.parametrize("weight_type", ["beta08", "gaussian", "uniform"])
def test_timestep_weight_function(weight_type):
    """The fixed per-timestep weight is a direct function of t: non-negative,
    finite, and normalized to ~unit mean over the shifted timestep grid."""
    model = _build_pretrain_model(weight_type=weight_type)
    model._timestep_weight_scale = None  # force re-derivation for this weight_type

    t = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    w = model._get_timestep_weight(t)
    assert torch.all(w >= 0), f"{weight_type} weights must be non-negative"
    assert torch.isfinite(w).all()

    # The normalization matches the reference: sum over the shifted grid == 1000.
    shift = model.sample_t_cfg.shift
    grid = torch.linspace(1.0, 0.0, 1001, dtype=torch.float64)
    grid = shift * grid / (1 + (shift - 1) * grid)
    assert abs(model._get_timestep_weight(grid).sum().item() - 1000.0) < 1e-6


# ---------------------------------------------------------------------------
# On-policy stage — DMD2 with the rollout-with-gradient student
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


def test_onpolicy_rollout_propagates_gradient():
    """The multi-step rollout must allow gradient flow on the chosen step.

    Mirrors AnyFlow's ``training_rollout`` (pipeline_wan_anyflow.py): one
    randomly-chosen step in the rollout has gradients enabled; the remaining
    steps are no_grad. The rollout output must keep the autograd graph so the
    DMD generator update has a valid gradient.
    """
    model = _build_onpolicy_model()
    real = torch.randn(1, 3, 8, 8, device=model.device, dtype=model.precision)
    cond = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=10).to(model.device, model.precision)

    input_student, t_student, _, _ = model._generate_noise_and_time(real)
    gen = model.gen_data_from_net(input_student, t_student, condition=cond)

    assert tuple(gen.shape) == (1, 3, 8, 8), f"rollout output shape mismatch: {gen.shape}"
    assert gen.requires_grad, "rollout output must keep autograd graph at the chosen step"
    loss = gen.float().pow(2).mean()
    loss.backward()
    grad_seen = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.net.parameters())
    assert grad_seen, "no gradient reached the student network through the rollout"


def test_onpolicy_rollout_no_grad_under_no_grad():
    """Under torch.no_grad (the fake-score update path), the rollout must run
    fully gradient-free."""
    model = _build_onpolicy_model()
    real = torch.randn(1, 3, 8, 8, device=model.device, dtype=model.precision)
    cond = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=10).to(model.device, model.precision)

    input_student, t_student, _, _ = model._generate_noise_and_time(real)
    with torch.no_grad():
        gen = model.gen_data_from_net(input_student, t_student, condition=cond)
    assert not gen.requires_grad


def test_onpolicy_optimizer_step():
    model = _build_onpolicy_model()
    data = _make_data(model, img_resolution=8)
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)


# ---------------------------------------------------------------------------
# Wan r-embedder fusion + AnyFlow checkpoint remap (pure functions)
# ---------------------------------------------------------------------------


def _make_fake_fusion_self(fusion_mode, gate_value=0.25, encoder_depth=None, dim=4, proj_dim=12):
    r_embedder = torch.nn.Module()
    r_embedder.time_proj = torch.nn.Linear(dim, proj_dim)
    r_embedder.act_fn = torch.nn.SiLU()
    r_embedder.fusion_mode = fusion_mode
    r_embedder.gate_value = gate_value
    return types.SimpleNamespace(r_embedder=r_embedder, encoder_depth=encoder_depth)


def test_fuse_r_embedding_gated():
    from fastgen.networks.Wan.network import _fuse_r_embedding

    torch.manual_seed(0)
    fake = _make_fake_fusion_self("gated")
    temb = torch.randn(2, 4)
    remb = torch.randn(2, 4)
    timestep_proj = torch.randn(2, 6, 2)

    out_temb, out_proj, out_r_proj = _fuse_r_embedding(fake, temb, timestep_proj, remb, None)

    gate = fake.r_embedder.gate_value
    rt_emb = (1 - gate) * temb + gate * remb
    expected = fake.r_embedder.time_proj(fake.r_embedder.act_fn(rt_emb)).unflatten(1, (6, -1))
    assert torch.allclose(out_temb, rt_emb)
    assert torch.allclose(out_proj, expected)
    assert out_r_proj is None


def test_fuse_r_embedding_gated_respects_encoder_depth():
    from fastgen.networks.Wan.network import _fuse_r_embedding

    torch.manual_seed(0)
    fake = _make_fake_fusion_self("gated", encoder_depth=2)
    temb = torch.randn(2, 4)
    remb = torch.randn(2, 4)
    timestep_proj = torch.randn(2, 6, 2)

    out_temb, out_proj, out_r_proj = _fuse_r_embedding(fake, temb, timestep_proj, remb, None)

    # Encoder blocks keep the t-only projection; the gated projection is
    # returned separately so the block loop switches at encoder_depth.
    assert torch.allclose(out_proj, timestep_proj)
    assert out_r_proj is not None and out_r_proj.shape == timestep_proj.shape
    gate = fake.r_embedder.gate_value
    assert torch.allclose(out_temb, (1 - gate) * temb + gate * remb)


def test_fuse_r_embedding_additive_unchanged():
    from fastgen.networks.Wan.network import _fuse_r_embedding

    torch.manual_seed(0)
    fake = _make_fake_fusion_self("additive")
    temb = torch.randn(2, 4)
    remb = torch.randn(2, 4)
    timestep_proj = torch.randn(2, 6, 2)

    out_temb, out_proj, out_r_proj = _fuse_r_embedding(fake, temb, timestep_proj, remb, None)

    r_proj = fake.r_embedder.time_proj(fake.r_embedder.act_fn(remb)).unflatten(1, (6, -1))
    assert torch.allclose(out_temb, temb + remb)
    assert torch.allclose(out_proj, timestep_proj + r_proj)
    assert torch.allclose(out_r_proj, r_proj)


@pytest.mark.parametrize("prefix", ["", "transformer."])
def test_remap_anyflow_keys(prefix):
    from fastgen.networks.Wan.network import remap_anyflow_keys

    sd = {
        f"{prefix}condition_embedder.delta_embedder.linear_1.weight": torch.randn(2, 2),
        f"{prefix}condition_embedder.time_proj.weight": torch.randn(2, 2),
        f"{prefix}condition_embedder.time_proj.bias": torch.randn(2),
        f"{prefix}blocks.0.attn1.to_q.weight": torch.randn(2, 2),
    }
    out = remap_anyflow_keys(sd)

    assert f"{prefix}r_embedder.time_embedder.linear_1.weight" in out
    assert f"{prefix}condition_embedder.delta_embedder.linear_1.weight" not in out
    # The shared time_proj is duplicated into the r_embedder.
    assert torch.equal(out[f"{prefix}r_embedder.time_proj.weight"], sd[f"{prefix}condition_embedder.time_proj.weight"])
    assert torch.equal(out[f"{prefix}r_embedder.time_proj.bias"], sd[f"{prefix}condition_embedder.time_proj.bias"])
    # Unrelated keys untouched.
    assert torch.equal(out[f"{prefix}blocks.0.attn1.to_q.weight"], sd[f"{prefix}blocks.0.attn1.to_q.weight"])


def test_remap_anyflow_keys_noop_without_delta_keys():
    from fastgen.networks.Wan.network import remap_anyflow_keys

    sd = {"transformer.blocks.0.attn1.to_q.weight": torch.randn(2, 2)}
    assert remap_anyflow_keys(sd) is sd
