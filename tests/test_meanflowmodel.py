# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
from copy import deepcopy
import gc
from types import SimpleNamespace
import pytest

import torch

from fastgen.methods import MeanFlowModel
from fastgen.methods.common_loss import finite_difference_points
from fastgen.configs.experiments.EDM.config_mf_cifar10 import create_config
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.configs.methods.config_mean_flow import SampleTConfig
import fastgen.methods.trajectory_matching.mean_flow as mean_flow_module
from fastgen.methods.trajectory_matching.mean_flow import FlowMapLossMixin, FlowMapStepResult
from fastgen.networks.module import FastGenModule
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import (
    AlphasNoiseSchedule,
    EDMNoiseSchedule,
    RFNoiseSchedule,
    TrigNoiseSchedule,
)
from fastgen.utils.guidance import GuidanceConfig
from fastgen.utils.tensor import expand_like, high_precision_dtype
from fastgen.utils.tree_utils import assert_same
from tests.utils import RunIf, assert_autocast_weight_cache_neutral, check_grad_zero


class _AutocastCacheSchedule:
    min_t = 0.0
    max_t = 1.0

    @staticmethod
    def forward_process(x, eps, t):
        t = t.reshape(-1, *(1 for _ in x.shape[1:]))
        return (1 - t) * x + t * eps

    @staticmethod
    def cond_velocity(x, eps, t, output_dtype="same"):
        del t
        dtype = x.dtype if output_dtype == "same" else output_dtype or high_precision_dtype(x, eps)
        return eps.to(dtype) - x.to(dtype)

    @staticmethod
    def flow_to_x0(x_t, velocity, t):
        del velocity, t
        return x_t


class _AutocastCacheNet(torch.nn.Module):
    """Tiny FP32 network whose autocast weight cache exposes no-grad reuse."""

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)
        self.noise_scheduler = _AutocastCacheSchedule()

    def forward(self, x_t, t, r, condition, fwd_pred_type, **kwargs):
        del t, r, condition, fwd_pred_type, kwargs
        return self.projection(x_t.flatten(1)).reshape_as(x_t)


def _autocast_cache_flow_map_loss(use_jvp_finite_diff):
    loss = FlowMapLossMixin()
    loss.device = torch.device("cpu")
    loss.precision = torch.float32
    loss.net = _AutocastCacheNet()
    loss.precision_amp_jvp = torch.bfloat16
    loss.loss_config = SimpleNamespace(
        use_cd=False,
        use_jvp_finite_diff=use_jvp_finite_diff,
        jvp_finite_diff_eps=1e-2,
        tangent_warmup_steps=0,
        loss_type="l2",
        norm_method=None,
        weight_type=None,
        tangent_spatial_invariance=False,
    )
    loss.config = SimpleNamespace(
        guidance_fuse_scale=None,
        guidance_cfg=GuidanceConfig(),
        loss_config=loss.loss_config,
    )
    return loss


def _record_autocast_cache_clears(monkeypatch):
    clear_calls = []
    clear_autocast_cache = torch.clear_autocast_cache

    def record_clear():
        clear_calls.append(None)
        clear_autocast_cache()

    monkeypatch.setattr(torch, "clear_autocast_cache", record_clear)
    return clear_calls


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    config = create_config()
    instance = config.model
    opts = [
        "-",
        "backbone.img_resolution=2",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=True",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
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


def _assert_student_sample_loop_propagates_gradient(model, data, student_sample_type):
    model.config.student_sample_type = student_sample_type
    model.net.zero_grad(set_to_none=True)

    noise = torch.randn_like(data["real"])
    student_grid = model.net.noise_scheduler.student_grid(2, device=model.device)
    latents = model.net.noise_scheduler.latents(noise, t_init=student_grid[0])
    generated = model._student_sample_loop(model.net, latents, student_grid, condition=data["condition"])

    assert generated.requires_grad, f"{student_sample_type} rollout detached the autograd graph"
    generated.float().square().mean().backward()
    grads = [param.grad for param in model.net.parameters() if param.grad is not None]
    assert grads, f"no gradients reached the student through the {student_sample_type} rollout"
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(torch.count_nonzero(grad) for grad in grads)


def test_ode_student_sample_loop_propagates_gradient(get_model_data):
    model, data = get_model_data
    _assert_student_sample_loop_propagates_gradient(model, data, "ode")


def test_sde_student_sample_loop_propagates_gradient(get_model_data):
    model, data = get_model_data
    _assert_student_sample_loop_propagates_gradient(model, data, "sde")


@pytest.mark.parametrize("student_sample_type", ["ode", "sde"])
def test_the_last_step_to_the_clean_grid_end_is_the_one_step_x0(student_sample_type):
    """Both rules jump to ``x_t - t u(x_t, t, r=0)`` at the grid's 0, and the SDE does not renoise it."""

    def unexpected_renoise(*args, **kwargs):
        raise AssertionError("the step to the clean grid end renoised its sample")

    class FlowMapNet:
        noise_scheduler = SimpleNamespace(forward_process=unexpected_renoise)

        def __call__(self, x, t, condition=None, r=None, fwd_pred_type=None):
            assert fwd_pred_type == "flow"
            return x * t[:, None] + r[:, None]

    model = object.__new__(MeanFlowModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(student_sample_type=student_sample_type)
    net = FlowMapNet()
    state = torch.randn(2, 3, dtype=torch.float64)
    grid = torch.tensor([0.9, 0.4, 0.0], dtype=torch.float64)

    x = model._advance_student_step(net, state, 1, grid)

    t = grid[1].expand(2)
    torch.testing.assert_close(x, state - grid[1] * net(state, t, r=torch.zeros_like(t), fwd_pred_type="flow"))


def test_sample_types_beyond_the_flow_map_rules_are_rejected_before_building():
    config = create_config().model
    config.device = torch.device("cpu")
    config.student_sample_type = 0.5
    with pytest.raises(ValueError, match="MeanFlowModel student_sample_type must be one of 'ode', 'sde'; got 0.5"):
        MeanFlowModel(config)


def test_r_is_sampled_from_the_t_distribution_with_its_overrides():
    """The CIFAR-10 recipe draws r from its t distribution with only the mean changed."""
    config = create_config().model
    draws = []

    def sample_t(n, device=None, **kwargs):
        draws.append(kwargs)
        return torch.rand(n, device=device)

    loss = FlowMapLossMixin()
    loss.device = torch.device("cpu")
    loss.net = SimpleNamespace(noise_scheduler=SimpleNamespace(sample_t=sample_t))
    loss.flow_map_sample_t_cfg = config.sample_t_cfg
    loss.flow_map_sample_r_overrides = config.sample_r_overrides
    loss._sample_t_r_buckets(4)

    t_kwargs, r_kwargs = draws
    assert t_kwargs["train_p_mean"] == -0.6
    assert r_kwargs == {**t_kwargs, "train_p_mean": -4.0}


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


def test_flow_map_step_result_preserves_meanflow_outputs_and_reward_inputs(monkeypatch):
    model = object.__new__(MeanFlowModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    data = {
        "real": torch.zeros(2, 1),
        "condition": {"embedding": torch.ones(2, 1)},
        "neg_condition": {"embedding": torch.zeros(2, 1)},
    }
    mf_loss = torch.tensor([2.0, 4.0], requires_grad=True)
    x0_approx = torch.ones(2, 1, requires_grad=True)
    dropout_mask = torch.tensor([True, False])
    t, r = torch.ones(2), torch.zeros(2)
    bucket_mask = torch.tensor([False, False])
    monkeypatch.setattr(model, "_sample_t_r_buckets", lambda batch_size: (t, r, bucket_mask))

    def compute_mf_loss(real_data, t, r, iteration, condition, neg_condition):
        assert real_data is data["real"]
        assert condition is data["condition"]
        assert neg_condition is data["neg_condition"]
        assert iteration == 7
        monitoring = torch.ones_like(real_data)
        return mf_loss, mf_loss, monitoring, monitoring, monitoring, x0_approx, torch.ones(2), 1.0, dropout_mask

    monkeypatch.setattr(model, "_compute_mf_loss", compute_mf_loss)
    monkeypatch.setattr(model, "_reduce_mf_loss", lambda loss, mask: loss.mean())

    result = model._flow_map_loss_step(data, iteration=7)

    assert isinstance(result, FlowMapStepResult)
    assert result.x0_approx is x0_approx
    assert result.condition is data["condition"]
    assert result.condition_dropout_mask is dropout_mask
    assert result.loss_map["total_loss"] is result.loss_map["mf_loss"]
    torch.testing.assert_close(result.loss_map["total_loss"], mf_loss.mean())
    assert result.loss_map["total_loss"].requires_grad
    outputs = {"gen_rand_train": x0_approx.detach()}
    rewarded_losses = {"total_loss": result.loss_map["total_loss"] + x0_approx.mean()}

    def flow_map_loss_step(step_data, iteration):
        assert step_data is data
        assert iteration == 7
        return result

    def get_outputs(generated, condition, neg_condition):
        assert generated is result.x0_approx
        assert condition is result.condition
        assert neg_condition is data["neg_condition"]
        return outputs

    def apply_reward(generated, reward_data, loss_map, condition_dropout_mask):
        assert generated is result.x0_approx
        assert reward_data is data
        assert loss_map is result.loss_map
        assert condition_dropout_mask is result.condition_dropout_mask
        return rewarded_losses

    monkeypatch.setattr(model, "_flow_map_loss_step", flow_map_loss_step)
    monkeypatch.setattr(model, "_get_outputs", get_outputs)
    monkeypatch.setattr(model, "apply_reward", apply_reward)

    loss_map, actual_outputs = model.single_train_step(data, iteration=7)

    assert loss_map is rewarded_losses
    assert actual_outputs is outputs


@pytest.mark.parametrize("optimize_reward", [False, True])
@pytest.mark.parametrize("dropout_probability", [None, 0.0, 1.0])
def test_single_train_step_forwards_the_actual_dropout_mask_to_rewards(
    get_model_data, optimize_reward, dropout_probability
):
    model, data = get_model_data
    model.config.optimize_reward = optimize_reward
    model.config.cond_dropout_prob = dropout_probability
    model.config.guidance_fuse_scale = 2.0
    rewarded = []

    def reward(gen_data, reward_data, condition_dropout_mask=None):
        rewarded.append((gen_data, reward_data, condition_dropout_mask))
        return {"total_loss": gen_data.square().mean()}

    model.reward = reward

    loss_map, outputs = model.single_train_step(data, 0)

    ((scored, scored_data, dropout_mask),) = rewarded
    assert scored_data is data
    assert scored.requires_grad == optimize_reward
    if dropout_probability is None:
        assert dropout_mask is None
    else:
        assert dropout_mask.dtype == torch.bool
        assert dropout_mask.shape == (data["real"].shape[0],)
        assert dropout_mask.eq(dropout_probability == 1.0).all()
    torch.testing.assert_close(scored.detach(), outputs["gen_rand_train"])
    expected_total = loss_map["mf_loss"] + loss_map["reward_loss"] if optimize_reward else loss_map["mf_loss"]
    torch.testing.assert_close(loss_map["total_loss"], expected_total)


def test_inactive_meanflow_guidance_does_not_report_configured_but_unused_dropout():
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.config.cond_dropout_prob = 1.0
    real = torch.ones(2, 1, 2, 2)
    z, t = torch.zeros_like(real), torch.tensor([0.5, 0.5])
    condition = torch.ones(2, 1)
    mixed, _, dropout_mask = loss._get_velocity(
        real, z, t, loss.net.noise_scheduler.forward_process(real, z, t), condition, torch.zeros_like(condition)
    )
    assert mixed is condition
    assert dropout_mask is None


class _FlowModelNet(FastGenNetwork):
    """The flow model ``v = condition * x_t``, predicting in ``net_pred_type``."""

    def __init__(self, net_pred_type):
        super().__init__(net_pred_type=net_pred_type, schedule_type="rf")

    def forward(
        self, x_t, t, condition, fwd_pred_type=None, apply_output_conditioning=True, output_dtype="same", **kwargs
    ):
        del kwargs
        velocity = expand_like(condition, x_t) * x_t
        native = self.noise_scheduler.convert_model_output(
            x_t, velocity, t, src_pred_type="flow", target_pred_type=self.net_pred_type, output_dtype=None
        )
        return self.transform_output(native, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning)

    def sample(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.parametrize("net_pred_type", ["x0", "eps", "flow"])
def test_meanflow_guided_target_is_independent_of_the_net_parametrization(net_pred_type):
    """``"pos"`` enters guidance as the conditional target in the net's native prediction type, so every
    parametrization of one flow model gives the guided velocity ``v + (s - 1) (v - u_neg)``."""
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.net = _FlowModelNet(net_pred_type)
    loss.config.cond_dropout_prob = None
    loss.config.guidance_cfg.cfg.scale = 3.0
    # Dyadic values keep x_t exact, so the data and noise are exactly the x0 and eps targets of x_t.
    real = torch.tensor([1.0, -0.5, 0.25, 2.0, 0.5, 1.5, -1.0, 0.0]).view(2, 1, 2, 2)
    z = torch.tensor([0.5, 1.0, -2.0, 0.75, -0.25, 0.0, 1.0, 2.5]).view(2, 1, 2, 2)
    t = torch.tensor([0.25, 0.5], dtype=loss.net.noise_scheduler.t_precision)
    condition, neg_condition = torch.tensor([1.0, 2.0]), torch.tensor([0.5, -1.0])
    x_t = loss.net.noise_scheduler.forward_process(real, z, t)

    _, target, _ = loss._get_velocity(real, z, t, x_t, condition, neg_condition)

    velocity = (z - real).double()
    neg_velocity = (expand_like(neg_condition, x_t) * x_t).double()
    torch.testing.assert_close(target, velocity + 2.0 * (velocity - neg_velocity))


@pytest.mark.parametrize("cond_dropout_prob", [None, 0.0, 1.0])
def test_target_side_guidance_applies_when_no_cond_dropout(cond_dropout_prob):
    """Without dropout every sample is conditional, so ``cond_dropout_prob=None`` guides the target exactly like
    ``0.0``; a dropped sample regresses onto the unguided velocity under ``neg_condition``."""
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.net = _FlowModelNet("flow")
    loss.config.cond_dropout_prob = cond_dropout_prob
    loss.config.guidance_cfg.cfg.scale = 3.0
    loss.flow_map_sample_t_cfg = SimpleNamespace(deterministic_buckets=True)
    real = torch.tensor([1.0, -0.5, 0.25, 2.0, 0.5, 1.5, -1.0, 0.0]).view(2, 1, 2, 2)
    z = torch.tensor([0.5, 1.0, -2.0, 0.75, -0.25, 0.0, 1.0, 2.5]).view(2, 1, 2, 2)
    t = torch.tensor([0.25, 0.5], dtype=loss.net.noise_scheduler.t_precision)
    condition, neg_condition = torch.tensor([1.0, 2.0]), torch.tensor([0.5, -1.0])
    x_t = loss.net.noise_scheduler.forward_process(real, z, t)

    mixed, target, dropout_mask = loss._get_velocity(real, z, t, x_t, condition, neg_condition)

    velocity = (z - real).double()
    if cond_dropout_prob == 1.0:
        assert dropout_mask.all()
        torch.testing.assert_close(mixed, neg_condition)
        torch.testing.assert_close(target, velocity)
    else:
        assert dropout_mask is None or not dropout_mask.any()
        torch.testing.assert_close(mixed, condition)
        neg_velocity = (expand_like(neg_condition, x_t) * x_t).double()
        torch.testing.assert_close(target, velocity + 2.0 * (velocity - neg_velocity))


def test_fused_jvp_scaling_is_gated_on_kept_samples():
    """Under ``guidance_fuse_scale``, dF/dt is divided by g only for the samples that stayed conditional: a dropped
    sample's fused prediction collapses to ``u_uncond``, so it keeps the unconditional MeanFlow identity. The toy net
    ignores the condition, which leaves that gate as the only difference between dropping every sample and none."""
    torch.manual_seed(0)
    g = 2.0
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.precision_amp_jvp = None
    loss.config.guidance_fuse_scale = g
    loss.flow_map_sample_t_cfg = SimpleNamespace(deterministic_buckets=True)
    real_data = torch.randn(2, 1, 2, 2)
    t = torch.tensor([0.75, 0.5])
    r = torch.tensor([0.25, 0.1])
    condition = torch.ones(2, 1)

    def fused_jvp(cond_dropout_prob):
        loss.config.cond_dropout_prob = cond_dropout_prob
        torch.manual_seed(1)  # the same noise in both runs
        return loss._compute_mf_loss(
            real_data, t, r, iteration=0, condition=condition, neg_condition=torch.zeros_like(condition)
        )[2]

    kept, dropped = fused_jvp(0.0), fused_jvp(1.0)
    assert torch.count_nonzero(kept)
    torch.testing.assert_close(dropped, g * kept)


def test_meanflow_keeps_fully_protected_conditions_without_a_dropout_mask():
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.config.cond_dropout_prob = 1.0
    loss.config.cond_keys_no_dropout = {"image"}
    loss.flow_map_sample_t_cfg = SimpleNamespace(deterministic_buckets=True)
    condition = {"image": torch.ones(2, 1)}
    mixed, dropout_mask = loss._drop_condition(condition, {"image": torch.zeros(2, 1)}, 2, torch.device("cpu"))
    assert mixed is condition
    assert dropout_mask is None


def test_meanflow_drops_structured_conditions_leaf_wise():
    """Dict values may be ``None`` or tuples/lists of tensors (Flux2's ``text_embeds=(None, tensor)``); protected
    keys pass through."""
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.config.cond_dropout_prob = 1.0
    loss.config.cond_keys_no_dropout = {"image"}
    loss.flow_map_sample_t_cfg = SimpleNamespace(deterministic_buckets=False)
    condition = {"text_embeds": (None, torch.ones(2, 3)), "ids": [torch.ones(2, 1)], "image": torch.ones(2, 1)}
    neg_condition = {"text_embeds": (None, torch.zeros(2, 3)), "ids": [torch.zeros(2, 1)], "image": torch.zeros(2, 1)}
    mixed, dropout_mask = loss._drop_condition(condition, neg_condition, 2, torch.device("cpu"))
    assert dropout_mask.tolist() == [True, True]
    assert mixed["text_embeds"][0] is None
    torch.testing.assert_close(mixed["text_embeds"][1], torch.zeros(2, 3))
    torch.testing.assert_close(mixed["ids"][0], torch.zeros(2, 1))
    assert mixed["image"] is condition["image"]


def test_meanflow_delegates_condition_dropout_to_the_network():
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.config.cond_dropout_prob = 0.5
    calls = []

    def mix_condition(condition, neg_condition, cond_dropout_prob):
        calls.append((condition, neg_condition, cond_dropout_prob))
        return "mixed", torch.tensor([True])

    loss.net.mix_condition = mix_condition
    mixed, dropout_mask = loss._drop_condition("condition", "negative", 1, torch.device("cpu"))
    assert mixed == "mixed" and dropout_mask.tolist() == [True]
    assert calls == [("condition", "negative", 0.5)]


def _raw_timestep_weight(weight_type, x):
    """The fixed per-timestep weights, written out as functions of an argument in [0, 1]."""
    if weight_type == "beta08":
        return x * (1 - x).clamp(min=0).sqrt()
    return (torch.exp(-2 * (x - 0.5) ** 2) - float(torch.exp(torch.tensor(-0.5)))).clamp(min=0)


@pytest.mark.parametrize("weight_type", ["beta08", "gaussian"])
def test_timestep_weights_are_functions_of_the_noise_fraction(weight_type):
    """The fixed per-timestep weights are functions of the noise fraction sigma / (alpha + sigma), so they apply on any
    schedule: it is exactly t on rectified flow, t / (1 + t) on EDM and sin t / (cos t + sin t) on Trig."""
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.config.precision_amp_jvp = None
    loss.loss_config.weight_type = weight_type

    t = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    loss.net.noise_scheduler = RFNoiseSchedule(min_t=0.0, max_t=1.0)
    assert torch.equal(loss._timestep_weight_raw(t), _raw_timestep_weight(weight_type, t))

    t = torch.linspace(0.002, 80.0, 101, dtype=torch.float64)
    loss.net.noise_scheduler = EDMNoiseSchedule()
    expected = _raw_timestep_weight(weight_type, t / (1 + t))
    torch.testing.assert_close(loss._timestep_weight_raw(t), expected, rtol=1e-12, atol=1e-15)

    t = torch.linspace(0.0, torch.pi / 2, 101, dtype=torch.float64)
    loss.net.noise_scheduler = TrigNoiseSchedule()
    expected = _raw_timestep_weight(weight_type, t.sin() / (t.cos() + t.sin()))
    torch.testing.assert_close(loss._timestep_weight_raw(t), expected, rtol=1e-12, atol=1e-15)

    # So the normalization is finite on any schedule, here over EDM's lognormal noise levels.
    loss.net.noise_scheduler = EDMNoiseSchedule()
    loss._init_flow_map_loss(SampleTConfig(time_dist_type="lognormal", min_t=0.002, max_t=80.0), None)
    assert 0 < loss._timestep_weight_scale() < float("inf")


@pytest.mark.parametrize("weight_type", ["beta08", "gaussian"])
@pytest.mark.parametrize("shift", [0.25, 1.0, 5.0])
def test_timestep_weight_scale_reproduces_the_shifted_schedule_grid(weight_type, shift):
    """In the AnyFlow setting, a shifted flow-map density on a rectified-flow schedule over [0, 1], the noise fraction
    is t itself and the density's quantiles are the shifted 1000-step schedule grid, so the constant is the one
    normalized over that grid with the weights of t, bit for bit."""
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.config.precision_amp_jvp = None
    schedule = RFNoiseSchedule(min_t=0.0, max_t=1.0)
    loss.net.noise_scheduler = schedule
    loss.loss_config.weight_type = weight_type
    loss._init_flow_map_loss(SampleTConfig(time_dist_type="shifted", shift=shift, min_t=0.0, max_t=1.0), None)

    grid = schedule.shift_t(schedule.grid(1000, device=torch.device("cpu"))[:-1], shift=shift)
    assert loss._timestep_weight_scale() == float(len(grid) / _raw_timestep_weight(weight_type, grid).sum())


def test_timestep_weight_scale_waits_for_a_meta_initialized_schedule_table():
    """Meta initialization materializes a table schedule only after construction, so the normalization is computed on
    first use, where it equals that of a schedule that was never on the meta device."""
    table = torch.linspace(0.9999, 0.0001, 1000, dtype=torch.float64)
    cfg = SampleTConfig(time_dist_type="uniform", min_t=0.0, max_t=0.999)
    scales = []
    for device in ("cpu", "meta"):
        loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
        loss.config.precision_amp_jvp = None
        loss.loss_config.weight_type = "beta08"
        with torch.device(device):
            loss.net.noise_scheduler = AlphasNoiseSchedule(alphas_cumprod=table)
        loss._init_flow_map_loss(cfg, None)
        loss.net.noise_scheduler.to_empty(device="cpu")
        loss.net.noise_scheduler.reset_parameters()
        scales.append(loss._timestep_weight_scale())
    assert scales[0] == scales[1] and 0 < scales[0] < float("inf")


def test_timestep_weight_scale_rejects_a_density_on_which_the_weight_vanishes():
    """A shift so large that every quantile of the density rounds to t = 1, where ``beta08`` vanishes, leaves no finite
    normalization."""
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.config.precision_amp_jvp = None
    loss.net.noise_scheduler = RFNoiseSchedule(min_t=0.0, max_t=1.0)
    loss.loss_config.weight_type = "beta08"
    loss._init_flow_map_loss(SampleTConfig(time_dist_type="shifted", shift=1e300, min_t=0.0, max_t=1.0), None)
    with pytest.raises(ValueError, match="no finite normalization: its weights sum to 0.0"):
        loss._timestep_weight_scale()


def test_timestep_weight_scale_reports_errors_raised_while_computing_it():
    """``nn.Module.__getattr__`` would mask an ``AttributeError`` raised in a property; a method lets it surface."""

    class _ModuleHost(FlowMapLossMixin, torch.nn.Module):
        pass

    host = _ModuleHost()
    host.loss_config = SimpleNamespace(weight_type="beta08")
    host.net = SimpleNamespace(noise_scheduler=RFNoiseSchedule(min_t=0.0, max_t=1.0))
    host.flow_map_sample_t_cfg = SimpleNamespace()
    host._timestep_weight_scale_value = None
    with pytest.raises(AttributeError, match="sample_t_kwargs"):
        host._timestep_weight_scale()


@pytest.mark.parametrize(
    "knobs, match",
    [
        ({"guidance_fuse_scale": 0.0}, "guidance_fuse_scale must be > 0"),
        ({"cond_dropout_prob": 1.5}, r"cond_dropout_prob must be in \[0, 1\]"),
    ],
)
def test_cond_dropout_and_fusion_check_rejects_invalid_values(knobs, match):
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.config = SimpleNamespace(guidance_cfg=GuidanceConfig(), loss_config=loss.loss_config, **knobs)
    with pytest.raises(ValueError, match=match):
        loss._check_cond_dropout_and_fusion_config()


@pytest.mark.parametrize(
    "knobs, use_cd, cfg_scale, warns",
    [
        ({"cond_dropout_prob": 0.1}, True, 2.0, True),
        ({"cond_dropout_prob": 0.1, "guidance_fuse_scale": 2.0}, True, None, False),
        ({"cond_dropout_prob": 0.1}, False, None, True),
        ({"cond_dropout_prob": 0.1}, False, 2.0, False),
        ({"cond_dropout_prob": 0}, True, None, False),
        ({}, True, None, False),
    ],
)
def test_cond_dropout_and_fusion_check_warns_when_dropout_never_applies(monkeypatch, knobs, use_cd, cfg_scale, warns):
    """The CD target drops the condition only under fusion; without CD, target-side guidance drops it too. A config
    declaring neither knob (AYF's) passes silently."""
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.loss_config.use_cd = use_cd
    guidance_cfg = GuidanceConfig()
    guidance_cfg.cfg.scale = cfg_scale
    loss.config = SimpleNamespace(guidance_cfg=guidance_cfg, loss_config=loss.loss_config, **knobs)
    warnings = []
    monkeypatch.setattr(mean_flow_module.logger, "warning", warnings.append)
    loss._check_cond_dropout_and_fusion_config()
    assert bool(warnings) is warns


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


def test_single_train_step_update_fp32_jvp():
    # Create config and enable fp32 JVP path
    config = create_config()
    instance = config.model
    opts = [
        "-",
        "backbone.img_resolution=2",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=True",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.teacher = deepcopy(instance.net)
    instance.teacher.backbone.r_timestep = False
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None
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


@pytest.mark.parametrize("use_jvp_finite_diff", [False, True])
def test_low_precision_no_grad_jvp_preserves_gradient_forward(monkeypatch, use_jvp_finite_diff):
    """The clear before the gradient forward keeps a no_grad JVP's detached BF16 casts out of it: the toy
    weights get the same gradient with the autocast weight cache on and off."""
    torch.manual_seed(0)
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff)
    clear_calls = _record_autocast_cache_clears(monkeypatch)
    real_data = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    t = torch.tensor([0.75])
    r = torch.tensor([0.25])

    def run_loss():
        return loss._compute_mf_loss(real_data, t, r, iteration=0)[0].sum()

    assert_autocast_weight_cache_neutral(loss.net, run_loss, "cpu")
    # Per run: the finite-difference pre-clear, plus the clear before the gradient forward (forward-mode AD
    # never touches the cache), plus torch's own clear when the outermost autocast region exits.
    assert len(clear_calls) == 3 * ((2 if use_jvp_finite_diff else 1) + 1)
    assert torch.count_nonzero(loss.net.projection.weight.grad)


@pytest.mark.parametrize("use_jvp_finite_diff", [False, True])
def test_jvp_clears_cache_under_disabled_enclosing_autocast(monkeypatch, use_jvp_finite_diff):
    """A disabled enclosing region still nests the JVP autocast, so torch does not drop its casts on exit.

    ``precision_amp=None`` training opens exactly such a region via ``model.autocast()``; without the clear
    before the gradient forward, the detached BF16 weights survive and detach any later BF16 forward.
    """
    torch.manual_seed(0)
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff)
    clear_calls = _record_autocast_cache_clears(monkeypatch)
    real_data = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    t = torch.tensor([0.75])
    r = torch.tensor([0.25])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
        loss._compute_mf_loss(real_data, t, r, iteration=0)
        # The clears do not depend on the enclosing region: the finite-difference pre-clear, plus the clear
        # before the gradient forward (forward-mode AD never touches the cache).
        assert len(clear_calls) == (2 if use_jvp_finite_diff else 1)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            prediction = loss.net(real_data, t, r=r, condition=None, fwd_pred_type="flow")

    assert prediction.requires_grad
    prediction.float().sum().backward()
    gradient = loss.net.projection.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient)


def test_finite_difference_points_go_one_sided_where_a_step_is_invalid():
    """Central where both steps are valid, one-sided where one is, and zero at the time itself where neither is."""
    x_t = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    dxt_dt = torch.arange(1, 9, dtype=torch.float64).reshape(4, 2)
    t = torch.tensor([0.5, 0.005, 0.995, 0.995], dtype=torch.float64)
    step = torch.full_like(t, 1e-2)
    forward_valid = torch.tensor([True, True, False, False])
    backward_valid = torch.tensor([True, False, True, False])

    x_t_plus, t_plus, x_t_minus, t_minus, factor = finite_difference_points(
        x_t, t, dxt_dt, step, forward_valid, backward_valid
    )

    torch.testing.assert_close(t_plus, torch.tensor([0.51, 0.015, 0.995, 0.995], dtype=torch.float64))
    torch.testing.assert_close(t_minus, torch.tensor([0.49, 0.005, 0.985, 0.995], dtype=torch.float64))
    plus_delta = torch.tensor([[0.01], [0.01], [0.0], [0.0]], dtype=x_t.dtype)
    minus_delta = torch.tensor([[-0.01], [0.0], [-0.01], [0.0]], dtype=x_t.dtype)
    torch.testing.assert_close(x_t_plus, x_t + plus_delta * dxt_dt)
    torch.testing.assert_close(x_t_minus, x_t + minus_delta * dxt_dt)
    torch.testing.assert_close(factor, torch.tensor([50.0, 100.0, 100.0, 0.0], dtype=torch.float64))


@pytest.mark.parametrize("step", [0.0, -1e-2, 1e-20])
def test_finite_difference_points_reject_a_valid_step_that_is_not_positive_or_rounds_away(step):
    """Such a step would silently give a zero JVP; a sample without a valid step keeps its zero factor."""
    x_t = torch.zeros(2, 1, dtype=torch.float64)
    dxt_dt = torch.ones(2, 1, dtype=torch.float64)
    t = torch.full((2,), 0.5, dtype=torch.float64)
    steps = torch.full_like(t, step)
    valid = torch.tensor([True, False])
    with pytest.raises(ValueError, match="not positive or rounds away"):
        finite_difference_points(x_t, t, dxt_dt, steps, valid, valid)
    invalid = torch.zeros(2, dtype=torch.bool)
    assert torch.equal(finite_difference_points(x_t, t, dxt_dt, steps, invalid, invalid)[-1], torch.zeros_like(t))


def test_finite_difference_jvp_is_exact_for_a_linear_net_and_zero_without_a_valid_step(monkeypatch):
    """For a net linear in x_t, central and one-sided differences both give the exact JVP. A sample whose steps
    would leave [min_t, max_t] or reach r gets a zero JVP, both evaluations at its own time."""
    torch.manual_seed(0)
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=True)
    loss.precision_amp_jvp = None
    t = torch.tensor([0.5, 0.005, 0.995, 0.995])
    r = torch.tensor([0.1, 0.0, 0.1, 0.995])
    x_t = torch.randn(4, 4)
    dxt_dt = torch.randn(4, 4)
    times = []
    forward = loss.net.forward

    def recording_forward(x_t, t, r, condition, fwd_pred_type, **kwargs):
        times.append(t)
        return forward(x_t, t, r, condition, fwd_pred_type, **kwargs)

    monkeypatch.setattr(loss.net, "forward", recording_forward)
    u_theta_jvp = loss._jvp(x_t, t, r, dxt_dt)

    with torch.no_grad():
        expected = loss.net.projection(dxt_dt).double()
    torch.testing.assert_close(u_theta_jvp[:3], expected[:3], rtol=1e-3, atol=1e-4)
    assert torch.equal(u_theta_jvp[3], torch.zeros_like(u_theta_jvp[3]))
    t_plus, t_minus = times
    assert t_plus[3] == t_minus[3] == t[3]


class _ModelPrecisionFlowNet(FastGenModule):
    """An elementwise flow that casts its state with ``cast_to_model_precision`` and reads its condition in
    ``precision_model``."""

    def __init__(self):
        super().__init__()
        self.noise_scheduler = _AutocastCacheSchedule()

    def forward(self, x_t, t, r, condition, fwd_pred_type, **kwargs):
        del r, fwd_pred_type, kwargs
        if self.precision_model is not None:
            assert_same(dtype=self.precision_model, condition=condition)
        x_t_model = self.cast_to_model_precision(x_t)
        return (torch.sin(x_t_model) * expand_like(t, x_t_model) + condition).to(dtype=x_t.dtype)


@pytest.mark.parametrize("use_jvp_finite_diff", [False, True])
@pytest.mark.parametrize(
    "device_type, precision_model, precision_amp_jvp",
    [
        ("cpu", torch.float16, torch.bfloat16),
        pytest.param("cuda", torch.bfloat16, torch.float32, marks=RunIf(min_gpus=1)),
    ],
)
def test_jvp_reads_network_inputs_promoted_with_the_autocast_dtype(
    device_type, precision_model, precision_amp_jvp, use_jvp_finite_diff
):
    """A ``precision_model`` narrower than ``precision_amp_jvp`` would round the state, the tangent and the
    finite-difference stencil at the network's input. Inside the JVP the network casts them to ``precision_model``
    promoted with the autocast dtype, while the condition stays in ``precision_model``, so this elementwise flow, which
    autocast leaves alone, gives the JVP of the uncast network."""
    torch.manual_seed(0)
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff)
    loss.device = torch.device(device_type)
    loss.net = _ModelPrecisionFlowNet()
    x_t = torch.randn(2, 4, device=device_type)
    dxt_dt = torch.randn_like(x_t)
    t = torch.tensor([0.5, 0.75], dtype=torch.float64, device=device_type)
    r = torch.tensor([0.1, 0.25], dtype=torch.float64, device=device_type)
    condition = torch.randn(2, 4, device=device_type).to(dtype=precision_model)

    loss.precision_amp_jvp = None
    reference = loss._jvp(x_t, t, r, dxt_dt, condition=condition.float())
    loss.net.prepare_module(device=None, precision_model=precision_model)
    loss.precision_amp_jvp = precision_amp_jvp
    jvp = loss._jvp(x_t, t, r, dxt_dt, condition=condition)

    torch.testing.assert_close(jvp, reference, rtol=0, atol=0)


def test_func_jvp_leaves_the_autocast_weight_cache_untouched():
    """``_jvp`` clears only before finite-difference forwards because ``torch.func.jvp`` neither fills nor
    reads the autocast weight cache: functorch wraps the captured parameters before autocast sees them. A
    plain no_grad forward in the same region does poison the cache, which is the control."""
    torch.manual_seed(0)
    x = torch.randn(2, 4)

    def gradient_forward_survives(no_grad_pass):
        linear = torch.nn.Linear(4, 4, bias=False)  # fp32 leaf weight: eligible for the cache
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, cache_enabled=True):
            with torch.no_grad():
                no_grad_pass(linear)
            return linear(x).requires_grad  # no clear in between

    assert gradient_forward_survives(lambda linear: torch.func.jvp(linear, (x,), (torch.ones_like(x),)))
    assert not gradient_forward_survives(lambda linear: linear(x))


class _FlowMapToTarget(torch.nn.Module):
    """A flow map whose average velocity jumps every state at time t onto ``target`` at time 0."""

    def __init__(self, noise_scheduler, target):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.target = target

    def forward(self, x_t, t, r=None, condition=None, fwd_pred_type=None, **kwargs):
        return (x_t - self.target) / expand_like(t.to(x_t.dtype), x_t)


def test_monitoring_uses_the_schedule_velocity_and_the_flow_map_jump():
    """On any schedule, ``v_loss`` measures against the schedule's conditional velocity, and the x0 preview takes the
    flow-map jump ``x_t - t u`` to time 0, exact at ``r = 0``, rather than reading ``u`` as an instantaneous velocity."""
    schedule = TrigNoiseSchedule()
    target = torch.randn(2, 4)
    loss = _autocast_cache_flow_map_loss(use_jvp_finite_diff=False)
    loss.precision_amp_jvp = None
    loss.net = _FlowMapToTarget(schedule, target)
    real_data = torch.randn(2, 4)
    t = torch.tensor([0.4, 1.2], dtype=torch.float64)

    torch.manual_seed(0)
    z = torch.randn_like(real_data)
    torch.manual_seed(0)
    with torch.no_grad():
        _, v_loss, _, _, _, x0_approx, *_ = loss._compute_mf_loss(real_data, t, torch.zeros_like(t), iteration=0)

    torch.testing.assert_close(x0_approx, target)
    u_theta = loss.net(schedule.forward_process(real_data, z, t), t)
    expected_v_loss = (u_theta - schedule.cond_velocity(real_data, z, t)).pow(2).mean(dim=1)
    torch.testing.assert_close(v_loss, expected_v_loss)
