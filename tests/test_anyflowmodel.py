# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AnyFlow.

The pretrain stage is MeanFlow with AnyFlow's hyperparameters (fixed
per-timestep loss weighting, finite-difference JVP, consistency bucket), so
the pretrain tests drive ``MeanFlowModel`` directly. The on-policy tests
exercise ``AnyFlowModel`` (DMD2 with a multi-step rollout-with-gradient
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
from fastgen.networks.noise_schedule import time_shift
from fastgen.utils.test_utils import check_grad_zero


def _build_pretrain_model(
    weight_type="beta08", consistency_ratio=0.25, flow_matching_ratio=0.5, deterministic_buckets=True
):
    """MeanFlow configured the AnyFlow way (paper Stage 1)."""
    gc.collect()
    instance = MeanFlowModelConfig()

    instance.loss_config.loss_type = "l2"
    instance.loss_config.weight_type = weight_type
    instance.loss_config.norm_method = None
    instance.loss_config.use_jvp_finite_diff = True
    instance.loss_config.jvp_finite_diff_eps = 1e-2

    instance.sample_t_cfg.time_dist_type = "shifted"
    instance.sample_t_cfg.shift = 5.0
    instance.sample_t_cfg.min_t = 0.001
    instance.sample_t_cfg.max_t = 0.999
    instance.sample_t_cfg.flow_matching_ratio = flow_matching_ratio
    instance.sample_t_cfg.consistency_ratio = consistency_ratio
    instance.sample_t_cfg.deterministic_buckets = deterministic_buckets

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


def _build_onpolicy_model(cotrain_time_dist_type="uniform", cotrain_shift=5.0):
    """On-policy fixture mirrors test_dmd2model: img_resolution=8 so the
    discriminator's 4x4 conv kernels can operate.

    ``cotrain_time_dist_type`` defaults to an unshifted density, so ``shift`` stays
    inert; pass a shifted one to exercise the shifted rollout grid.
    """
    gc.collect()
    instance = AnyFlowModelConfig()

    base_opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1", "+schedule_type=rf"]
    # Teacher / fake score are PLAIN single-timestep nets, as in the reference
    # and in config_anyflow_onpolicy.py; only the student carries the r pathway.
    instance.teacher = override_config_with_opts(AnyFlowModelConfig().net, list(base_opts))
    opts = base_opts[:1] + ["r_timestep=True"] + base_opts[1:]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_disc = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_disc)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""
    instance.student_update_freq = 2
    instance.student_sample_steps = 2
    instance.input_shape = [3, 8, 8]

    # Co-trained flow-map loss settings (MeanFlow machinery on the tiny net). These
    # live on their own config: `sample_t_cfg` stays DMD2's noising time.
    instance.cotrain_sample_t_cfg.time_dist_type = cotrain_time_dist_type
    instance.cotrain_sample_t_cfg.shift = cotrain_shift
    instance.cotrain_sample_t_cfg.min_t = 0.001
    instance.cotrain_sample_t_cfg.max_t = 0.999
    instance.cotrain_sample_t_cfg.flow_matching_ratio = 0.5
    instance.cotrain_sample_t_cfg.consistency_ratio = 0.25
    instance.cotrain_sample_t_cfg.deterministic_buckets = True
    instance.loss_config.loss_type = "l2"
    instance.loss_config.weight_type = "uniform"
    instance.loss_config.norm_method = None
    instance.loss_config.use_jvp_finite_diff = True
    instance.loss_config.jvp_finite_diff_eps = 1e-2

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


def test_pretrain_consistency_bucket_pins_r_to_zero():
    """With consistency_ratio=1.0 (and no flow-matching head), every sample's
    r must be pinned to 0 (consistency to clean data, as in the reference)."""
    model = _build_pretrain_model(consistency_ratio=1.0, flow_matching_ratio=0.0)
    _t, r, r_eq_t_mask = model._sample_t_r_buckets(4)
    assert not r_eq_t_mask.any()
    assert torch.allclose(r.float(), torch.zeros_like(r.float()))


def test_pretrain_bucket_partition_is_deterministic():
    """With deterministic_buckets, both buckets follow the reference's global
    partition: flow-matching head, consistency middle, random-pair tail."""
    model = _build_pretrain_model(consistency_ratio=0.25, flow_matching_ratio=0.5, deterministic_buckets=True)
    batch_size = 8
    t, r, r_eq_t_mask = model._sample_t_r_buckets(batch_size)

    n_flow_matching = round(0.5 * batch_size)
    n_consistency = round(0.25 * batch_size)
    assert r_eq_t_mask.tolist() == [True] * n_flow_matching + [False] * (batch_size - n_flow_matching)
    assert torch.equal(r[:n_flow_matching], t[:n_flow_matching])
    assert torch.allclose(
        r[n_flow_matching : n_flow_matching + n_consistency].float(),
        torch.zeros(n_consistency, device=r.device),
    )


def test_pretrain_bucket_partition_is_stochastic():
    """Without deterministic_buckets the same two ratios drive binomial bucket
    SIZES: the layout is still head/middle/tail, but the sizes vary per call
    and average to the configured fractions."""
    model = _build_pretrain_model(consistency_ratio=0.25, flow_matching_ratio=0.5, deterministic_buckets=False)
    torch.manual_seed(0)
    batch_size = 4096
    t, r, r_eq_t_mask = model._sample_t_r_buckets(batch_size)

    # min_t = 0.001, so only the consistency bucket can carry r == 0.
    is_consistency = r == 0
    assert not (r_eq_t_mask & is_consistency).any()
    assert torch.equal(r[r_eq_t_mask], t[r_eq_t_mask])
    assert abs(r_eq_t_mask.float().mean().item() - 0.5) < 0.05
    assert abs(is_consistency.float().mean().item() - 0.25) < 0.05
    # Prefix layout, same as the deterministic policy.
    n_flow_matching = int(r_eq_t_mask.sum())
    assert r_eq_t_mask[:n_flow_matching].all() and not r_eq_t_mask[n_flow_matching:].any()

    # The sizes are what varies, unlike the deterministic policy.
    sizes = {int(model._sample_t_r_buckets(64)[2].sum()) for _ in range(20)}
    assert len(sizes) > 1, f"stochastic bucket sizes should vary, got {sizes}"


def test_pretrain_rebalance_to_flow_matching():
    """With rebalancing on, each r < t loss is rescaled to the flow-matching
    loss mean by a detached per-sample factor."""
    model = _build_pretrain_model()
    model.loss_config.rebalance_to_flow_matching = True

    mf_loss = torch.tensor([2.0, 4.0, 10.0, 100.0], dtype=torch.float64, requires_grad=True)
    r_eq_t_mask = torch.tensor([True, True, False, False])
    loss = model._reduce_mf_loss(mf_loss, r_eq_t_mask)

    # flow-matching mean = 3.0; each r < t sample becomes ~3.0.
    expected = (2.0 + 4.0 + 3.0 * (10.0 / 10.00001) + 3.0 * (100.0 / 100.00001)) / 4.0
    assert abs(loss.item() - expected) < 1e-3
    loss.backward()
    assert mf_loss.grad is not None and torch.isfinite(mf_loss.grad).all()


def test_pretrain_rebalance_all_flow_matching_batch():
    """A rank whose batch is entirely flow-matching (r = t) must still take
    the rebalance branch (the collective inside must run on every rank) and
    reduce to the plain mean."""
    model = _build_pretrain_model()
    model.loss_config.rebalance_to_flow_matching = True

    mf_loss = torch.tensor([2.0, 4.0], dtype=torch.float64)
    r_eq_t_mask = torch.tensor([True, True])
    loss = model._reduce_mf_loss(mf_loss, r_eq_t_mask)
    assert abs(loss.item() - 3.0) < 1e-8


def test_pretrain_prediction_side_guidance_fusion():
    """The AnyFlow guidance-distillation branch (guidance_fuse_scale) must run
    end to end and keep gradients on the fused prediction."""
    model = _build_pretrain_model()
    model.config.guidance_fuse_scale = 3.0
    model.config.cond_dropout_prob = 0.5
    data = _make_data(model, batch_size=2)

    loss_map, _ = model.single_train_step(data, 0)
    assert torch.isfinite(loss_map["total_loss"]).all()
    loss_map["total_loss"].backward()
    grad_seen = any(p.grad is not None for p in model.net.parameters())
    assert grad_seen


def test_pretrain_guidance_fusion_requires_neg_condition():
    """Fusion queries the unconditional branch, so a missing neg_condition must
    stop with a configuration error rather than reach the network as None."""
    model = _build_pretrain_model()
    model.config.guidance_fuse_scale = 3.0
    data = _make_data(model)
    data["neg_condition"] = None

    with pytest.raises(AssertionError, match="requires neg_condition"):
        model.single_train_step(data, 0)


def test_shifted_variants_share_the_shift_map():
    """Both shifted time distributions apply the same shift map when sampling t,
    so the grids rebuilt outside the sampler must pick up `shift` for either one.

    Covers the loss-weight normalization (pretrain) and the rollout schedule
    (on-policy); a variant missing from the lookup degrades silently to shift=1.
    """
    model = _build_pretrain_model()
    shifted_scale = model._timestep_weight_scale
    cfg_t, cfg_r = model.config.sample_t_cfg, model.config.sample_r_cfg

    cfg_t.time_dist_type = "shifted_logitnormal"
    model._init_flow_map_loss(cfg_t, cfg_r)
    assert model._timestep_weight_scale == pytest.approx(shifted_scale)

    # and the shift is what makes it differ from an unshifted grid
    cfg_t.time_dist_type = "uniform"
    model._init_flow_map_loss(cfg_t, cfg_r)
    assert model._timestep_weight_scale != pytest.approx(shifted_scale)

    # Same lookup on the on-policy rollout schedule. The shift is resolved once at
    # init, so each variant needs a re-init rather than a bare config mutation.
    onpolicy = _build_onpolicy_model(cotrain_time_dist_type="shifted", cotrain_shift=5.0)
    cot_t, cot_r = onpolicy.config.cotrain_sample_t_cfg, onpolicy.config.cotrain_sample_r_cfg

    # `rollout_t_list` is static, so read `flow_map_shift` back off the model: that is
    # what checks the co-train density actually reaches the rollout grid.
    def rollout_grid():
        return onpolicy.rollout_t_list(4, onpolicy.flow_map_shift, float(onpolicy.net.noise_scheduler.max_t))

    shifted_grid = rollout_grid()

    cot_t.time_dist_type = "shifted_logitnormal"
    onpolicy._init_flow_map_loss(cot_t, cot_r)
    assert torch.allclose(rollout_grid(), shifted_grid)

    cot_t.time_dist_type = "uniform"
    onpolicy._init_flow_map_loss(cot_t, cot_r)
    assert not torch.allclose(rollout_grid(), shifted_grid)


def test_onpolicy_shifted_rollout_grid_and_step():
    """A shifted co-train density must move the rollout grid, and a student update
    must run on it.

    Every other on-policy test uses the unshifted fixture, so without this the shift
    never reaches `rollout_t_list` or a real rollout -- the shipped
    `config_anyflow_onpolicy.py` runs shift=5.
    """
    model = _build_onpolicy_model(cotrain_time_dist_type="shifted", cotrain_shift=5.0)
    assert model.flow_map_shift == 5.0

    max_t = float(model.net.noise_scheduler.max_t)
    grid = torch.linspace(1.0, 0.0, 5, dtype=torch.float64)
    expected = time_shift(grid, 5.0).clamp(max=max_t)
    t_list = model.rollout_t_list(4, model.flow_map_shift, max_t).double()

    assert torch.allclose(t_list, expected)
    # endpoints are fixed by the map; the interior is pushed towards max_t
    assert t_list[0].item() == pytest.approx(min(1.0, max_t)) and t_list[-1].item() == 0.0
    assert (t_list[1:-1] > grid.clamp(max=max_t)[1:-1]).all()

    # and the whole student update runs on that grid
    data = _make_data(model, img_resolution=8)
    loss_map, _ = model.single_train_step(data, 0)
    assert torch.isfinite(loss_map["total_loss"]).all()
    assert "vsd_loss" in loss_map and "bidirection_loss" in loss_map


def test_onpolicy_dmd_and_cotrain_noising_times_are_separate():
    """DMD2 draws its noising time from `sample_t_cfg`; the co-trained flow-map loss
    draws (t, r) from `cotrain_sample_t_cfg`.

    The reference keeps these on two schedulers (`dmd_scheduler` vs `scheduler`), so
    give the two configs disjoint ranges and check neither path reads the other's.
    """
    model = _build_onpolicy_model()
    dmd = model.config.sample_t_cfg
    dmd.time_dist_type, dmd.min_t, dmd.max_t = "uniform", 0.70, 0.80
    cot = model.config.cotrain_sample_t_cfg
    cot.time_dist_type, cot.min_t, cot.max_t = "uniform", 0.10, 0.20
    model._init_flow_map_loss(cot, model.config.cotrain_sample_r_cfg)

    # the mixin binds the co-train config, never DMD2's
    assert model.flow_map_sample_t_cfg is cot
    assert model.flow_map_sample_t_cfg is not dmd

    t_dmd = model._sample_noising_time(64, iteration=0)
    assert ((t_dmd >= 0.70) & (t_dmd <= 0.80)).all(), t_dmd

    t_mf, r_mf, _ = model._sample_t_r_buckets(64)
    assert ((t_mf >= 0.10) & (t_mf <= 0.20)).all(), t_mf
    # r shares t's density unless sample_r_cfg is enabled; the consistency bucket pins 0
    assert ((r_mf >= 0.0) & (r_mf <= 0.20)).all(), r_mf


@pytest.mark.parametrize("weight_type", ["beta08", "gaussian", "uniform"])
def test_timestep_weight_function(weight_type):
    """The fixed per-timestep weight is a direct function of t: non-negative,
    finite, and normalized like the reference scheduler."""
    model = _build_pretrain_model(weight_type=weight_type)

    # The fixture sets norm_method=None, so _compute_weight on a ones tensor is
    # exactly the fixed per-timestep weight w(t).
    t = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    w = model._compute_weight(torch.ones_like(t), t)
    assert torch.all(w >= 0), f"{weight_type} weights must be non-negative"
    assert torch.isfinite(w).all()

    if weight_type == "uniform":
        # Uniform normalizes to exactly 1.0 over the reference grid.
        assert torch.allclose(w, torch.ones_like(w))

    # The weight has mean one over the network's discrete training timesteps
    # (t=0 excluded), which at the default num_steps=1000 is the reference's
    # set_timesteps grid: sum over the shifted grid == num_steps.
    num_steps = model.net.noise_scheduler.num_steps
    shift = model.flow_map_sample_t_cfg.shift
    grid = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)[:-1]
    grid = shift * grid / (1 + (shift - 1) * grid)
    assert abs(model._compute_weight(torch.ones_like(grid), grid).sum().item() - num_steps) < 1e-6


# ---------------------------------------------------------------------------
# On-policy stage — DMD2 with the rollout-with-gradient student
# ---------------------------------------------------------------------------


def test_onpolicy_student_update_step():
    model = _build_onpolicy_model()
    data = _make_data(model, img_resolution=8)
    loss_map, outputs = model.single_train_step(data, 0)  # iteration 0 -> student update
    assert "total_loss" in loss_map
    assert "vsd_loss" in loss_map
    # The co-trained Stage-1 flow-map loss is part of every student update.
    assert "bidirection_loss" in loss_map
    assert torch.isfinite(loss_map["total_loss"]).all()
    assert "gen_rand" in outputs


def test_onpolicy_cotrain_can_be_disabled():
    model = _build_onpolicy_model()
    model.config.cotrain_pretrain_weight = 0.0
    data = _make_data(model, img_resolution=8)
    loss_map, _ = model.single_train_step(data, 0)
    assert "bidirection_loss" not in loss_map


def test_onpolicy_student_starts_from_pure_noise():
    """The rollout is on-policy: the student starts from pure noise at max_t.

    DMD2's multi-step branch would hand back real data noised to a random entry
    of `t_list`, which `gen_data_from_net` would then roll out as if it sat at
    `t_list[0]` — both off-policy and a latent/timestep mismatch.
    """
    model = _build_onpolicy_model()
    torch.manual_seed(0)
    real = torch.randn(8, 3, 8, 8, device=model.device, dtype=model.precision)
    input_student, t_student, _, _ = model._generate_noise_and_time(real, iteration=0)

    max_t = float(model.net.noise_scheduler.max_t)
    assert torch.allclose(t_student.float(), torch.full_like(t_student.float(), max_t))
    # Pure noise carries no signal from the batch it was drawn alongside; the
    # off-policy failure mode correlates at ~0.7 (1/sqrt(192) ~ 0.07 by chance).
    cos = torch.nn.functional.cosine_similarity(input_student.flatten(1).float(), real.flatten(1).float())
    assert cos.abs().max() < 0.35, f"student input correlates with real data: {cos.tolist()}"


def test_onpolicy_rollout_compresses_to_three_forwards():
    """Regardless of the sampled NFE, the rollout must run at most three
    network forwards (jump -> fine step -> jump), with gradient through all."""
    model = _build_onpolicy_model()
    model.config.student_sample_steps_list = [16]
    real = torch.randn(1, 3, 8, 8, device=model.device, dtype=model.precision)
    cond = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=10).to(model.device, model.precision)
    input_student, t_student, _, _ = model._generate_noise_and_time(real)

    calls = []
    orig_forward = model.net.forward

    def counting_forward(*args, **kwargs):
        calls.append(1)
        return orig_forward(*args, **kwargs)

    model.net.forward = counting_forward
    try:
        gen = model.gen_data_from_net(input_student, t_student, condition=cond)
    finally:
        model.net.forward = orig_forward

    assert len(calls) <= 3, f"rollout must compress to <= 3 forwards, got {len(calls)}"
    assert gen.requires_grad


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
    """The rollout output must keep the autograd graph so the DMD generator
    update has a valid gradient.

    Mirrors AnyFlow's ``training_rollout`` (pipeline_wan_anyflow.py): the
    compressed jump -> fine step -> jump rollout runs with gradient through
    all segments.
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
    condition_embedder = torch.nn.Module()
    condition_embedder.time_proj = torch.nn.Linear(dim, proj_dim)
    condition_embedder.act_fn = torch.nn.SiLU()

    r_embedder = torch.nn.Module()
    r_embedder.fusion_mode = fusion_mode
    r_embedder.gate_value = gate_value
    if fusion_mode == "additive":
        # Gated fusion reuses condition_embedder.time_proj, so Wan.__init__
        # drops these from the r_embedder in that mode.
        r_embedder.time_proj = torch.nn.Linear(dim, proj_dim)
        r_embedder.act_fn = torch.nn.SiLU()
    return types.SimpleNamespace(
        r_embedder=r_embedder, condition_embedder=condition_embedder, encoder_depth=encoder_depth
    )


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
    # The gated projection goes through the SHARED condition_embedder.time_proj.
    expected = fake.condition_embedder.time_proj(fake.condition_embedder.act_fn(rt_emb)).unflatten(1, (6, -1))
    assert torch.allclose(out_temb, rt_emb)
    assert torch.allclose(out_proj, expected)
    assert out_r_proj is None
    assert not hasattr(fake.r_embedder, "time_proj")


def test_validation_t_list_matches_shift():
    """The validation schedule must equal the student's shifted sampling grid.

    `config_anyflow_onpolicy.py` derives `sample_t_cfg.t_list` from
    `cotrain_sample_t_cfg.shift`, the same shift `rollout_t_list` applies, so this pins
    the two together and catches a stale override.

    Left as None, DMD2 hands `generator_fn` the noise scheduler's UNSHIFTED
    `linspace(max_t, 0, N+1)`, which is far off-policy for a shift=5 model --
    asserted below so the schedule cannot silently degrade to that.
    """
    import fastgen.configs.experiments.WanT2V.config_anyflow_onpolicy as mod

    cfg = mod.create_config()
    shift = float(cfg.model.cotrain_sample_t_cfg.shift)
    n = int(cfg.model.student_sample_steps)
    max_t = float(cfg.model.net.max_t)  # the bound `rollout_t_list` clamps to

    grid = [1.0 - i / n for i in range(n + 1)]
    expected = [min(shift * x / (1 + (shift - 1) * x), max_t) for x in grid]

    t_list = list(cfg.model.sample_t_cfg.t_list)
    assert len(t_list) == n + 1, t_list
    assert all(abs(a - b) < 1e-9 for a, b in zip(t_list, expected, strict=True)), (t_list, expected)
    assert t_list[0] == max_t and t_list[-1] == 0.0

    # and it must not be the unshifted fallback DMD2 would use for None
    unshifted = [max_t * (1.0 - i / n) for i in range(n + 1)]
    assert not all(abs(a - b) < 1e-6 for a, b in zip(t_list, unshifted, strict=True))


def test_fuse_r_embedding_gated_trains_the_shared_time_proj():
    """The gated projection must flow gradient into condition_embedder.time_proj.

    A private r_embedder.time_proj would leave the condition_embedder's copy
    without gradient (a dead parameter under DDP/FSDP) while the two silently
    drift apart, breaking round-trips to the AnyFlow checkpoint layout.
    """
    from fastgen.networks.Wan.network import _fuse_r_embedding

    torch.manual_seed(0)
    fake = _make_fake_fusion_self("gated")
    _, out_proj, _ = _fuse_r_embedding(fake, torch.randn(2, 4), torch.randn(2, 6, 2), torch.randn(2, 4), None)
    out_proj.sum().backward()

    grad = fake.condition_embedder.time_proj.weight.grad
    assert grad is not None and torch.any(grad != 0)


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
    # r is already folded into the two above, and without encoder_depth the
    # blocks have nothing to switch to.
    assert out_r_proj is None


@pytest.mark.parametrize("prefix", ["", "transformer."])
def test_remap_anyflow_keys(prefix):
    from fastgen.networks.Wan.utils import remap_anyflow_keys

    sd = {
        f"{prefix}condition_embedder.delta_embedder.linear_1.weight": torch.randn(2, 2),
        f"{prefix}condition_embedder.time_proj.weight": torch.randn(2, 2),
        f"{prefix}condition_embedder.time_proj.bias": torch.randn(2),
        f"{prefix}blocks.0.attn1.to_q.weight": torch.randn(2, 2),
    }
    out = remap_anyflow_keys(sd)

    assert f"{prefix}r_embedder.time_embedder.linear_1.weight" in out
    assert f"{prefix}condition_embedder.delta_embedder.linear_1.weight" not in out
    # time_proj stays shared on the condition_embedder — no r_embedder copy.
    assert f"{prefix}r_embedder.time_proj.weight" not in out
    assert torch.equal(
        out[f"{prefix}condition_embedder.time_proj.weight"], sd[f"{prefix}condition_embedder.time_proj.weight"]
    )
    # Unrelated keys untouched.
    assert torch.equal(out[f"{prefix}blocks.0.attn1.to_q.weight"], sd[f"{prefix}blocks.0.attn1.to_q.weight"])


def test_remap_anyflow_keys_noop_without_delta_keys():
    from fastgen.networks.Wan.utils import remap_anyflow_keys

    sd = {"transformer.blocks.0.attn1.to_q.weight": torch.randn(2, 2)}
    assert remap_anyflow_keys(sd) is sd


def test_rollout_uses_the_flow_map_sample_loop():
    """`gen_data_from_net` must resolve to FlowMapLossMixin's loop, not the base one.

    `AnyFlowModel(FlowMapLossMixin, DMD2Model)` picks it purely by MRO order.
    `FastGenModel._student_sample_loop` is an x0-prediction loop that never
    passes `r`, and it accepts the same arguments -- so swapping the base order
    would silently turn the flow-map rollout (jump / fine step / jump) into a
    plain diffusion sampler instead of raising.
    """
    from fastgen.methods import AnyFlowModel
    from fastgen.methods.consistency_model.mean_flow import FlowMapLossMixin

    assert (
        AnyFlowModel._student_sample_loop.__func__ is FlowMapLossMixin._student_sample_loop.__func__
    ), "AnyFlowModel must inherit the flow-map sample loop; check the base-class order"
