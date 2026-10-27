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

import fastgen.methods.distribution_matching.anyflow as anyflow_module
from fastgen.callbacks.carry import CARRY_OUTPUT_KEY
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.configs.methods.config_anyflow import ModelConfig as AnyFlowModelConfig
from fastgen.configs.methods.config_mean_flow import ModelConfig as MeanFlowModelConfig
from fastgen.methods import AnyFlowModel, MeanFlowModel
from fastgen.networks.noise_schedule import StudentGridConfig, time_shift
from fastgen.utils.global_vars import set_temp_global_vars
from tests.utils import check_grad_zero


def _build_pretrain_model(
    weight_type="beta08",
    consistency_ratio=0.25,
    flow_matching_ratio=0.5,
    deterministic_buckets=True,
    density_shift=5.0,
    grid_shift=5.0,
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
    instance.sample_t_cfg.shift = density_shift
    if grid_shift is None:
        instance.student_grid_cfg.grid_kwargs.pop("shift", None)
    else:
        instance.student_grid_cfg.grid_kwargs["shift"] = grid_shift
    instance.sample_t_cfg.min_t = 0.001
    instance.sample_t_cfg.max_t = 0.999
    instance.sample_t_cfg.flow_matching_ratio = flow_matching_ratio
    instance.sample_t_cfg.consistency_ratio = consistency_ratio
    instance.sample_t_cfg.deterministic_buckets = deterministic_buckets

    opts = [
        "-",
        "backbone.img_resolution=2",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=True",
        "+schedule_type=rf",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None
    instance.input_shape = [3, 2, 2]

    model = MeanFlowModel(instance)
    model.on_train_begin()
    model.init_optimizers()
    return model


def _build_onpolicy_model(
    cotrain_time_dist_type="uniform",
    cotrain_shift=5.0,
    student_grid_shift=None,
    cotrain_pretrain_weight=None,
    gan_loss_weight_gen=None,
    precision_model=None,
):
    """On-policy fixture mirrors test_dmd2model: img_resolution=8 so the
    discriminator's 4x4 conv kernels can operate.

    ``cotrain_time_dist_type`` defaults to an unshifted density, so ``shift`` stays
    inert; pass a shifted one to exercise the shifted rollout grid. ``precision_model``
    defaults to fp64 on CPU and bf16 on GPU; prepared modules keep it, so a test that needs
    another precision builds with it instead of re-resolving afterwards.
    """
    gc.collect()
    instance = AnyFlowModelConfig()

    base_opts = [
        "-",
        "backbone.img_resolution=8",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "+schedule_type=rf",
    ]
    # Teacher / fake score are PLAIN single-timestep nets, as in the reference
    # and in config_anyflow_onpolicy.py; only the student carries the r pathway.
    instance.teacher = override_config_with_opts(AnyFlowModelConfig().net, list(base_opts))
    opts = base_opts[:1] + ["backbone.r_timestep=True"] + base_opts[1:]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_disc = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_disc)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    default_precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.precision_model = default_precision_model if precision_model is None else precision_model
    instance.pretrained_model_path = None
    instance.student_update_freq = 2
    instance.student_grid_cfg.steps = 2
    instance.input_shape = [3, 8, 8]
    if cotrain_pretrain_weight is not None:
        instance.cotrain_pretrain_weight = cotrain_pretrain_weight
    if gan_loss_weight_gen is not None:
        instance.gan_loss_weight_gen = gan_loss_weight_gen

    # Co-trained flow-map loss settings (MeanFlow machinery on the tiny net). These
    # live on their own config: `sample_t_cfg` stays DMD2's noising time.
    instance.cotrain_sample_t_cfg.time_dist_type = cotrain_time_dist_type
    instance.cotrain_sample_t_cfg.shift = cotrain_shift
    instance.cotrain_sample_t_cfg.min_t = 0.001
    instance.cotrain_sample_t_cfg.max_t = 0.999
    instance.cotrain_sample_t_cfg.flow_matching_ratio = 0.5
    instance.cotrain_sample_t_cfg.consistency_ratio = 0.25
    instance.cotrain_sample_t_cfg.deterministic_buckets = True
    if student_grid_shift is None:
        instance.student_grid_cfg.grid_kwargs.pop("shift", None)
    else:
        instance.student_grid_cfg.grid_kwargs["shift"] = student_grid_shift
    instance.loss_config.loss_type = "l2"
    instance.loss_config.weight_type = None
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
        "idx": list(range(batch_size)),
        "real": torch.randn(batch_size, 3, img_resolution, img_resolution).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_labels.to(model.device, model.precision),
    }


# ---------------------------------------------------------------------------
# Pretrain stage — MeanFlow with AnyFlow options
# ---------------------------------------------------------------------------


def _student_inputs(model, real, iteration: int = 1):
    """AnyFlow's live student-input path."""
    data = {"real": real, "idx": list(range(real.shape[0]))}
    _inputs = model._training_inputs(data, iteration=iteration)
    input_student = _inputs.input_student
    student_step = _inputs.student_step
    t = _inputs.t
    eps = _inputs.eps
    trajectory_step = _inputs.trajectory_step
    generation_kwargs = _inputs.generation_kwargs
    loss_scale = _inputs.loss_scale
    assert student_step is None, "AnyFlow constructs its rollout schedule in-iteration"
    assert trajectory_step is None, "AnyFlow completes its rollout in-iteration and carries no step"
    assert generation_kwargs == {}
    assert loss_scale is None
    return input_student, trajectory_step, t, eps


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
    assert abs(loss.detach().item() - expected) < 1e-3
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

    with pytest.raises(ValueError, match="requires neg_condition"):
        model.single_train_step(data, 0)


def test_flow_map_and_time_density_shifts_are_independent(monkeypatch):
    """The student grid and stochastic t/r density retain their own shifts; the weight normalization follows the
    density."""
    model = _build_pretrain_model(density_shift=2.0, grid_shift=5.0)
    cfg_t, r_overrides = model.config.sample_t_cfg, model.config.sample_r_overrides
    scale = model._timestep_weight_scale()

    seen_shifts = []
    sample_t = model.net.noise_scheduler.sample_t

    def record_shift(*args, **kwargs):
        seen_shifts.append(kwargs.get("shift"))
        return sample_t(*args, **kwargs)

    monkeypatch.setattr(model.net.noise_scheduler, "sample_t", record_shift)
    model._sample_t_r_buckets(2)
    assert seen_shifts == [2.0, 2.0]

    # The weight normalization follows the density, its shift and any family.
    cfg_t.shift = 5.0
    model._init_flow_map_loss(cfg_t, r_overrides)
    assert model._timestep_weight_scale() != pytest.approx(scale)
    for time_dist_type in ("shifted_logitnormal", "uniform", "lognormal", "log_t", "polynomial"):
        cfg_t.time_dist_type = time_dist_type
        model._init_flow_map_loss(cfg_t, r_overrides)
        assert 0 < model._timestep_weight_scale() < float("inf"), time_dist_type
        assert model._timestep_weight_scale() != pytest.approx(scale), time_dist_type

    # Different density families and density shifts share a grid when grid policy matches.
    shifted = _build_onpolicy_model(cotrain_time_dist_type="shifted", cotrain_shift=2.0, student_grid_shift=5.0)
    shifted_log = _build_onpolicy_model(
        cotrain_time_dist_type="shifted_logitnormal", cotrain_shift=7.0, student_grid_shift=5.0
    )
    unshifted = _build_onpolicy_model(cotrain_time_dist_type="uniform", student_grid_shift=5.0)
    identity_grid = _build_onpolicy_model(cotrain_time_dist_type="uniform", student_grid_shift=None)
    shifted_grid = shifted.net.noise_scheduler.student_grid(4, device=shifted.device)
    shifted_log_grid = shifted_log.net.noise_scheduler.student_grid(4, device=shifted_log.device)
    unshifted_grid = unshifted.net.noise_scheduler.student_grid(4, device=unshifted.device)
    identity = identity_grid.net.noise_scheduler.student_grid(4, device=identity_grid.device)
    assert torch.allclose(shifted_grid, shifted_log_grid)
    assert torch.allclose(shifted_grid, unshifted_grid)
    assert not torch.allclose(shifted_grid, identity)


def test_timestep_weight_scale_ignores_the_student_grid():
    """The fixed-weight constant comes from the flow-map t density, so neither the student grid's
    shift nor anchoring it to a coarse base grid changes it."""
    model = _build_pretrain_model()
    scale = model._timestep_weight_scale()
    scheduler = model.net.noise_scheduler
    scheduler.student_grid_cfg = StudentGridConfig(steps=4, grid_kwargs={"base_steps": 8, "shift": 2.0})

    model._init_flow_map_loss(model.config.sample_t_cfg, model.config.sample_r_overrides)

    assert model._timestep_weight_scale() == scale


def test_onpolicy_shifted_rollout_grid_and_step():
    """A shifted student grid must drive the rollout and a full student update.

    The co-training density deliberately uses a different shift, proving the rollout
    reads scheduler grid policy. The shipped config happens to set both shifts to 5.
    """
    model = _build_onpolicy_model(cotrain_time_dist_type="shifted", cotrain_shift=2.0, student_grid_shift=5.0)

    max_t = float(model.net.noise_scheduler.max_t)
    grid = torch.linspace(max_t, 0.0, 5, device=model.device, dtype=torch.float64)
    expected = time_shift(grid, 5.0, max_t=max_t)
    student_grid = model.net.noise_scheduler.student_grid(4, device=model.device).double()

    assert torch.allclose(student_grid, expected)
    # endpoints are fixed by the map; the interior is pushed towards max_t
    assert student_grid[0].item() == pytest.approx(max_t) and student_grid[-1].item() == 0.0
    assert (student_grid[1:-1] > grid[1:-1]).all()

    # and the whole student update runs on that grid
    data = _make_data(model, img_resolution=8)
    loss_map, _ = model.single_train_step(data, 0)
    assert torch.isfinite(loss_map["total_loss"]).all()
    assert "vsd_loss" in loss_map and "cotrain_loss" in loss_map


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
    model._init_flow_map_loss(cot, model.config.cotrain_sample_r_overrides)

    # the mixin binds the co-train config, never DMD2's
    assert model.flow_map_sample_t_cfg is cot
    assert model.flow_map_sample_t_cfg is not dmd

    t_dmd, _ = model._sample_t_and_eps(64)
    assert ((t_dmd >= 0.70) & (t_dmd <= 0.80)).all(), t_dmd

    t_mf, r_mf, _ = model._sample_t_r_buckets(64)
    assert ((t_mf >= 0.10) & (t_mf <= 0.20)).all(), t_mf
    # without cotrain_sample_r_overrides r shares t's density; the consistency bucket pins 0
    assert ((r_mf >= 0.0) & (r_mf <= 0.20)).all(), r_mf


@pytest.mark.parametrize("weight_type", ["beta08", "gaussian"])
def test_timestep_weight_function(weight_type):
    """The fixed per-timestep weight is a direct function of t: non-negative,
    finite, and normalized like the reference scheduler."""
    model = _build_pretrain_model(weight_type=weight_type)

    # The fixture sets norm_method=None, so _compute_weight on a ones tensor is
    # exactly the fixed per-timestep weight w(t), over the schedule range.
    max_t = model.net.noise_scheduler.max_t
    t = torch.linspace(0.0, max_t, 101, device=model.device, dtype=torch.float64)
    w = model._compute_weight(torch.ones_like(t), t)
    assert torch.all(w >= 0), f"{weight_type} weights must be non-negative"
    assert torch.isfinite(w).all()

    # The weight has mean one over the quantiles of one draw of the flow-map t
    # density at probabilities 1, 1 - 1/1000, ..., 1/1000: for the shifted
    # density, the shifted 1000-step grid mapped onto the sampling bounds.
    cfg = model.flow_map_sample_t_cfg
    num_steps = 1000
    unit = time_shift(torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)[:-1], cfg.shift)
    table = (cfg.min_t + (cfg.max_t - cfg.min_t) * unit).to(model.device)
    assert abs(model._compute_weight(torch.ones_like(table), table).sum().item() - num_steps) < 1e-6


# ---------------------------------------------------------------------------
# On-policy stage — DMD2 with the rollout-with-gradient student
# ---------------------------------------------------------------------------


def test_onpolicy_requires_backward_simulation():
    config = AnyFlowModelConfig()
    assert config.backward_simulation == "default"
    assert config.max_steps_rollout == 3

    config.backward_simulation = None
    with pytest.raises(ValueError, match="requires backward_simulation='default'"):
        AnyFlowModel(config)


def test_onpolicy_rejects_sample_t_in_student_bucket():
    model = _build_onpolicy_model()
    model.config.sample_t_in_student_bucket = True

    with pytest.raises(ValueError, match="AnyFlowModel provided student_step of shape None"):
        model._training_inputs({"idx": [0, 1]})


def test_onpolicy_rejects_step_generations():
    config = AnyFlowModelConfig()
    config.student_generation = "step"

    with pytest.raises(ValueError, match=r"AnyFlowModel supports student_generation in \('x0',\)"):
        AnyFlowModel(config)


@pytest.mark.parametrize(
    "max_steps_rollout, selected, expected",
    [
        (1, 0, [0, 8]),
        (2, 1, [0, 2, 8]),
        (3, 2, [0, 2, 3, 8]),
        (5, 2, [0, 2, 3, 4, 5, 8]),
        (9, 0, list(range(9))),  # both window starts expose the whole interior: one path, not two
        (10, 0, list(range(9))),
    ],
)
def test_onpolicy_rollout_paths_respect_forward_cap(max_steps_rollout, selected, expected):
    model = object.__new__(AnyFlowModel)
    torch.nn.Module.__init__(model)
    model.config = types.SimpleNamespace(max_steps_rollout=max_steps_rollout)
    model.device = torch.device("cpu")

    paths = AnyFlowModel._rollout_paths(model, num_steps=8)

    assert paths[selected] == expected
    assert len(paths) == len({tuple(path) for path in paths})
    assert all(len(path) - 1 <= max_steps_rollout for path in paths)


@pytest.mark.parametrize("value", [0, -1])
def test_onpolicy_rollout_rejects_invalid_forward_cap(value):
    config = AnyFlowModelConfig()
    config.max_steps_rollout = value

    with pytest.raises(ValueError, match="max_steps_rollout must be positive"):
        AnyFlowModel(config)


def test_onpolicy_student_update_step():
    model = _build_onpolicy_model()
    data = _make_data(model, img_resolution=8)
    loss_map, outputs = model.single_train_step(data, 0)  # iteration 0 -> student update
    assert "total_loss" in loss_map
    assert "vsd_loss" in loss_map
    assert "student_steps" in loss_map
    assert "num_rollout_forwards" in loss_map
    # The co-trained Stage-1 flow-map loss is part of every student update.
    assert "cotrain_loss" in loss_map
    assert {
        "cotrain_mf_loss",
        "cotrain_v_loss",
        "cotrain_jvp_norm",
        "cotrain_tangent_norm",
        "cotrain_v_norm",
        "cotrain_loss_weight",
        "cotrain_tangent_warmup_weight",
    } <= loss_map.keys()
    assert torch.equal(loss_map["cotrain_loss"], loss_map["cotrain_mf_loss"])
    assert torch.isfinite(loss_map["total_loss"]).all()
    assert "gen_rand" in outputs
    assert CARRY_OUTPUT_KEY not in outputs


def test_onpolicy_cotrain_reads_only_the_named_loss_map(monkeypatch):
    model = object.__new__(AnyFlowModel)
    torch.nn.Module.__init__(model)
    model.config = types.SimpleNamespace(student_update_freq=2, cotrain_pretrain_weight=0.25)
    data = {"real": torch.ones(2, 1)}
    base_loss = torch.tensor(3.0, requires_grad=True)
    cotrain_loss = torch.tensor(2.0, requires_grad=True)
    base_losses = {"total_loss": base_loss, "reward_loss": torch.tensor(0.5)}
    outputs = {"gen_rand_train": torch.zeros(2, 1)}
    calls = []

    def dmd2_step(self, step_data, iteration):
        assert self is model
        assert step_data is data
        assert iteration == 0
        calls.append("dmd2")
        return base_losses, outputs

    def flow_map_loss_step(step_data, iteration):
        assert step_data is data
        assert iteration == 0
        calls.append("flow_map")
        # Cotraining must not unpack the result or depend on MeanFlow's output/reward fields.
        return types.SimpleNamespace(
            loss_map={"total_loss": cotrain_loss, "mf_loss": cotrain_loss, "jvp_norm": torch.tensor(7.0)}
        )

    monkeypatch.setattr(anyflow_module.DMD2Model, "single_train_step", dmd2_step)
    monkeypatch.setattr(model, "_flow_map_loss_step", flow_map_loss_step)

    loss_map, actual_outputs = model.single_train_step(data, iteration=0)

    assert calls == ["dmd2", "flow_map"]
    assert loss_map is base_losses
    assert actual_outputs is outputs
    assert loss_map["cotrain_loss"] is cotrain_loss
    assert loss_map["cotrain_mf_loss"] is cotrain_loss
    torch.testing.assert_close(loss_map["cotrain_jvp_norm"], torch.tensor(7.0))
    torch.testing.assert_close(loss_map["reward_loss"], torch.tensor(0.5))
    torch.testing.assert_close(loss_map["total_loss"], base_loss + 0.25 * cotrain_loss)
    loss_map["total_loss"].backward()
    torch.testing.assert_close(base_loss.grad, torch.tensor(1.0))
    torch.testing.assert_close(cotrain_loss.grad, torch.tensor(0.25))


def test_onpolicy_cotrain_can_be_disabled():
    model = _build_onpolicy_model()
    model.config.cotrain_pretrain_weight = 0.0
    data = _make_data(model, img_resolution=8)
    loss_map, _ = model.single_train_step(data, 0)
    assert "cotrain_loss" not in loss_map


@pytest.mark.parametrize("iteration", [0, 1])
def test_onpolicy_data_free_training(iteration):
    """Both alternating updates run without real data when real-data losses are off."""
    model = _build_onpolicy_model(cotrain_pretrain_weight=0.0, gan_loss_weight_gen=0.0)
    data = _make_data(model, img_resolution=8)
    del data["real"]

    loss_map, outputs = model.single_train_step(data, iteration)

    assert torch.isfinite(loss_map["total_loss"]).all()
    assert "cotrain_loss" not in loss_map
    assert "gan_loss_disc" not in loss_map
    assert CARRY_OUTPUT_KEY not in outputs


@pytest.mark.parametrize("iteration", [0, 1])
def test_onpolicy_data_free_training_rejects_enabled_cotrain(iteration):
    model = _build_onpolicy_model()
    data = _make_data(model, img_resolution=8)
    del data["real"]

    with pytest.raises(ValueError, match=r"cotrain_pretrain_weight > 0 requires data\['real'\]"):
        model.single_train_step(data, iteration)


def test_onpolicy_student_starts_from_pure_noise():
    """The rollout is on-policy: the student starts from pure noise at max_t.

    DMD2's multi-step branch would hand back real data noised to a random entry
    of `student_grid`, which `gen_data_from_net` would then roll out as if it sat at
    `student_grid[0]` — both off-policy and a latent/timestep mismatch.
    """
    model = _build_onpolicy_model()
    torch.manual_seed(0)
    real = torch.randn(8, 3, 8, 8, device=model.device, dtype=model.precision)
    input_student, trajectory_step, _, _ = _student_inputs(model, real, iteration=0)

    assert trajectory_step is None
    # Pure noise carries no signal from the batch it was drawn alongside; the
    # off-policy failure mode correlates at ~0.7 (1/sqrt(192) ~ 0.07 by chance).
    cos = torch.nn.functional.cosine_similarity(input_student.flatten(1).float(), real.flatten(1).float())
    assert cos.abs().max() < 0.35, f"student input correlates with real data: {cos.tolist()}"


def test_onpolicy_student_starts_at_the_first_student_grid_node(monkeypatch):
    """The start state sits at ``student_grid[0]`` even when an explicit grid starts below ``max_t``."""
    model = _build_onpolicy_model()
    model.net.noise_scheduler.student_grid_cfg = StudentGridConfig(
        steps=2,
        grid_type="explicit",
        grid_kwargs={"timesteps": [0.9, 0.5, 0.0]},
    )
    noise = torch.randn(2, 3, 8, 8, device=model.device, dtype=model.precision)
    monkeypatch.setattr(anyflow_module, "noise_from_shape", lambda *_args, **_kwargs: noise)
    data = {"idx": [0, 1]}

    _inputs = model._training_inputs(data, iteration=0)
    input_student = _inputs.input_student

    scheduler = model.net.noise_scheduler
    first_node = torch.tensor(0.9, dtype=torch.float64, device=model.device)
    torch.testing.assert_close(input_student, scheduler.latents(noise, t_init=first_node))
    # ... which differs from the schedule's own maximum time.
    max_time = torch.as_tensor(scheduler.max_t, dtype=torch.float64, device=model.device)
    assert not torch.allclose(input_student, scheduler.latents(noise, t_init=max_time))


def test_onpolicy_training_inputs_always_use_main_dmd_noising_time():
    """AnyFlow's shared training inputs always use DMD2's main noising config."""
    model = _build_onpolicy_model()
    model.config.sample_t_cfg.time_dist_type = "uniform"
    model.config.sample_t_cfg.min_t = 0.10
    model.config.sample_t_cfg.max_t = 0.20
    model.config.fake_score_sample_t_overrides = {"min_t": 0.70, "max_t": 0.80}
    real = torch.randn(64, 3, 8, 8, device=model.device, dtype=model.precision)

    _, _, t_student_update, _ = _student_inputs(model, real, iteration=0)
    _, _, t_auxiliary_update, _ = _student_inputs(model, real, iteration=1)

    assert ((t_student_update >= 0.10) & (t_student_update <= 0.20)).all()
    assert ((t_auxiliary_update >= 0.10) & (t_auxiliary_update <= 0.20)).all()


def test_onpolicy_rollout_respects_default_forward_cap():
    """The default preserves the reference's at-most-three-forward rollout."""
    model = _build_onpolicy_model()
    model.config.student_steps_list = [16]
    real = torch.randn(1, 3, 8, 8, device=model.device, dtype=model.precision)
    cond = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=10).to(model.device, model.precision)
    input_student, trajectory_step, _, _ = _student_inputs(model, real, iteration=0)

    calls = []
    orig_forward = model.net.forward

    def counting_forward(*args, **kwargs):
        calls.append(1)
        return orig_forward(*args, **kwargs)

    model.net.forward = counting_forward
    try:
        generation = model.gen_data_from_net(input_student, trajectory_step, condition=cond)
        gen = generation.data
        generation_metrics = generation.metrics
    finally:
        model.net.forward = orig_forward

    assert len(calls) <= model.config.max_steps_rollout
    assert gen.requires_grad
    assert generation.next_state is None
    assert generation.x0 is generation.data
    torch.testing.assert_close(generation_metrics["student_steps"], torch.tensor(16.0, device=model.device))
    torch.testing.assert_close(
        generation_metrics["num_rollout_forwards"],
        torch.tensor(float(len(calls)), device=model.device),
    )


def test_onpolicy_training_rollout_ignores_global_student_steps(monkeypatch):
    """The process-wide override must not change the sampled training rollout."""
    model = _build_onpolicy_model()
    model.config.student_steps_list = [16]
    real = torch.randn(1, 3, 8, 8, device=model.device, dtype=model.precision)
    cond = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=10).to(model.device, model.precision)
    input_student, trajectory_step, _, _ = _student_inputs(model, real, iteration=0)
    choice_bounds = []

    def choose_first(high, **kwargs):
        del kwargs
        choice_bounds.append(high)
        return 0

    monkeypatch.setattr("fastgen.methods.distribution_matching.anyflow.broadcast_choice", choose_first)

    with set_temp_global_vars({"STUDENT_STEPS": 4}):
        generation = model.gen_data_from_net(input_student, trajectory_step, condition=cond)
        generation_metrics = generation.metrics

    assert choice_bounds == [1, len(model._rollout_paths(16))]
    torch.testing.assert_close(generation_metrics["student_steps"], torch.tensor(16.0, device=model.device))
    torch.testing.assert_close(
        generation_metrics["num_rollout_forwards"],
        torch.tensor(float(len(model._rollout_paths(16)[0]) - 1), device=model.device),
    )


def test_onpolicy_auxiliary_update_step():
    model = _build_onpolicy_model(precision_model="float32")
    data = _make_data(model, img_resolution=8)
    loss_map, outputs = model.single_train_step(data, 1)  # iteration 1 -> fake_score/disc update
    assert "fake_score_loss" in loss_map
    assert "gan_loss_disc" in loss_map
    assert "student_steps" in loss_map
    assert "num_rollout_forwards" in loss_map
    assert "gen_rand" in outputs
    assert CARRY_OUTPUT_KEY not in outputs


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

    input_student, trajectory_step, _, _ = _student_inputs(model, real, iteration=0)
    generation = model.gen_data_from_net(input_student, trajectory_step, condition=cond)
    gen = generation.data

    assert tuple(gen.shape) == (1, 3, 8, 8), f"rollout output shape mismatch: {gen.shape}"
    assert generation.next_state is None
    assert generation.x0 is generation.data
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

    input_student, trajectory_step, _, _ = _student_inputs(model, real, iteration=0)
    with torch.no_grad():
        generation = model.gen_data_from_net(input_student, trajectory_step, condition=cond)
        gen = generation.data
    assert not gen.requires_grad
    assert generation.next_state is None
    assert generation.x0 is generation.data


def test_onpolicy_optimizer_step():
    model = _build_onpolicy_model()
    data = _make_data(model, img_resolution=8)
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)


def test_onpolicy_setup_warns_once_without_the_stage_one_checkpoint(monkeypatch):
    """The trainer loads the Stage-1 flow map from pretrained_ckpt_path, which sets resume_iter, before the
    optimizers are built; without it, setup warns once."""
    model = _build_onpolicy_model()
    warnings = []
    monkeypatch.setattr(anyflow_module.logger, "warning", warnings.append)

    model.init_optimizers()
    [message] = warnings
    assert "pretrained_ckpt_path" in message

    warnings.clear()
    model.resume_iter = 6000
    model.init_optimizers()
    assert warnings == []


def test_onpolicy_validation_generator_runs_the_flow_map_step():
    """The `gen_rand` callable in `outputs` is what W&B validation actually
    invokes. `test_rollout_uses_shared_loop_and_flow_map_step` only checks the MRO
    statically, so drive the callable end to end and assert it really passed
    `r` (i.e. dispatched to the flow-map step, not FastGenModel's x0 step).
    """
    model = _build_onpolicy_model(cotrain_time_dist_type="shifted", student_grid_shift=5.0)
    model.config.student_sample_type = "ode"
    data = _make_data(model, img_resolution=8)
    _, outputs = model.single_train_step(data, 0)

    seen_r = []
    orig_forward = model.net_inference.forward

    def recording_forward(*args, **kwargs):
        seen_r.append(kwargs.get("r"))
        return orig_forward(*args, **kwargs)

    model.net_inference.forward = recording_forward
    try:
        gen = outputs["gen_rand"]()
        configured_r = seen_r.copy()
        seen_r.clear()
        with set_temp_global_vars({"STUDENT_STEPS": 4}):
            gen_override = outputs["gen_rand"]()
        overridden_r = seen_r.copy()
    finally:
        model.net_inference.forward = orig_forward

    assert tuple(gen.shape) == (1, 3, 8, 8), gen.shape
    assert tuple(gen_override.shape) == tuple(gen.shape)
    assert torch.isfinite(gen.float()).all()
    assert torch.isfinite(gen_override.float()).all()
    assert len(configured_r) == 2, f"expected one forward per configured step, got {len(configured_r)}"
    assert all(r is not None for r in configured_r), "flow-map step must pass r at every interval"
    # ODE: r is the NEXT timestep and only the last one lands on r = 0. A non-zero first r distinguishes
    # the flow-map loop from the sde branch (which always passes r = 0).
    configured_grid = model.net_inference.noise_scheduler.student_grid(device=model.device)
    for actual_r, expected_r in zip(configured_r, configured_grid[1:], strict=True):
        assert torch.allclose(actual_r.float(), torch.full_like(actual_r.float(), float(expected_r)))

    expected_grid = model.net_inference.noise_scheduler.student_grid(4, device=model.device)
    assert len(overridden_r) == 4, f"expected one forward per overridden step, got {len(overridden_r)}"
    for actual_r, expected_r in zip(overridden_r, expected_grid[1:], strict=True):
        assert torch.allclose(actual_r.float(), torch.full_like(actual_r.float(), float(expected_r)))


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
    r_embedder.keep_original_prec = False
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


def test_apply_r_embedding_additive_abs_matches_fuse_and_rejects_orphan_r():
    from fastgen.networks.Wan.network import apply_r_embedding, _fuse_r_embedding

    torch.manual_seed(0)
    fake = _make_fake_fusion_self("additive")
    fake.time_cond_type = "abs"
    fake.r_embedder.timesteps_proj = torch.nn.Identity()
    fake.r_embedder.time_embedder = lambda x: x.unsqueeze(-1).expand(-1, 4).clone()

    temb = torch.randn(2, 4)
    timestep_proj = torch.randn(2, 6, 2)
    r_timestep = torch.randn(2)
    remb = fake.r_embedder.time_embedder(r_timestep)

    actual = apply_r_embedding(fake, torch.zeros(2), r_timestep, temb, timestep_proj)
    expected = _fuse_r_embedding(fake, temb, timestep_proj, remb, None)
    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])
    assert actual[2] is None and expected[2] is None

    no_r = apply_r_embedding(fake, torch.zeros(2), None, temb, timestep_proj)
    assert no_r[0] is temb and no_r[1] is timestep_proj and no_r[2] is None

    fake.r_embedder = None
    with pytest.raises(ValueError, match="no r_embedder"):
        apply_r_embedding(fake, torch.zeros(2), r_timestep, temb, timestep_proj)


def test_apply_r_embedding_diff_subtracts_flattened_timestep():
    from fastgen.networks.Wan.network import apply_r_embedding, _fuse_r_embedding

    torch.manual_seed(1)
    fake = _make_fake_fusion_self("additive")
    fake.time_cond_type = "diff"
    fake.r_embedder.timesteps_proj = torch.nn.Identity()
    fake.r_embedder.time_embedder = lambda x: x.unsqueeze(-1).expand(-1, 4).clone()

    timestep = torch.tensor([1.0, 2.0])
    r_timestep = torch.tensor([0.25, 0.5])
    temb = torch.randn(2, 4)
    timestep_proj = torch.randn(2, 6, 2)
    remb = fake.r_embedder.time_embedder(timestep - r_timestep)

    actual = apply_r_embedding(fake, timestep, r_timestep, temb, timestep_proj)
    expected = _fuse_r_embedding(fake, temb, timestep_proj, remb, None)
    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])


def _legacy_additive_r_mix(self, timestep, r_timestep, temb, timestep_proj):
    """Pre-helper additive mix copied from the VACE/causal prepares."""
    from fastgen.networks.Wan.network import flatten_timestep, unflatten_timestep_proj

    if self.time_cond_type == "abs":
        pass
    elif self.time_cond_type == "diff":
        r_timestep = timestep - r_timestep
    else:
        raise ValueError(f"Invalid time condition: {self.time_cond_type}")
    r_timestep, rs_seq_len = flatten_timestep(r_timestep)
    r_timestep = self.r_embedder.timesteps_proj(r_timestep)
    remb = self.r_embedder.time_embedder(r_timestep)
    r_timestep_proj = self.r_embedder.time_proj(self.r_embedder.act_fn(remb))
    r_timestep_proj = unflatten_timestep_proj(r_timestep_proj, rs_seq_len)
    if self.encoder_depth is None:
        return temb + remb, timestep_proj + r_timestep_proj, r_timestep_proj
    return remb, timestep_proj, r_timestep_proj


@pytest.mark.parametrize("encoder_depth", [None, 1])
@pytest.mark.parametrize("time_cond_type", ["abs", "diff"])
def test_apply_r_embedding_additive_matches_legacy_inline_mix(encoder_depth, time_cond_type):
    from fastgen.networks.Wan.network import apply_r_embedding

    torch.manual_seed(0)
    fake = _make_fake_fusion_self("additive", encoder_depth=encoder_depth)
    fake.time_cond_type = time_cond_type
    fake.r_embedder.timesteps_proj = torch.nn.Identity()
    fake.r_embedder.time_embedder = lambda x: x.unsqueeze(-1).expand(-1, 4).clone()

    timestep = torch.tensor([1.0, 2.0])
    r_timestep = torch.tensor([0.25, 0.5])
    temb = torch.randn(2, 4)
    timestep_proj = torch.randn(2, 6, 2)

    actual = apply_r_embedding(fake, timestep, r_timestep, temb, timestep_proj)
    legacy = _legacy_additive_r_mix(fake, timestep, r_timestep, temb, timestep_proj)
    assert torch.equal(actual[0], legacy[0])
    assert torch.equal(actual[1], legacy[1])
    if encoder_depth is None:
        assert actual[2] is None
        assert legacy[2] is not None
    else:
        assert torch.equal(actual[2], legacy[2])


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


def test_rollout_uses_shared_loop_and_flow_map_step():
    """MeanFlow and AnyFlow share orchestration but dispatch each interval to the flow-map hook."""
    import inspect

    from fastgen.methods import AnyFlowModel, MeanFlowModel
    from fastgen.methods.model import FastGenModel
    from fastgen.methods.trajectory_matching.mean_flow import FlowMapLossMixin

    for model_cls in (MeanFlowModel, AnyFlowModel):
        assert inspect.getattr_static(model_cls, "_student_sample_loop") is inspect.getattr_static(
            FastGenModel, "_student_sample_loop"
        )
        assert inspect.getattr_static(model_cls, "_advance_student_step") is inspect.getattr_static(
            FlowMapLossMixin, "_advance_student_step"
        )
