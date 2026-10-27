# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import gc
from types import SimpleNamespace
import pytest

import torch

from fastgen.methods import SCMModel
from fastgen.configs.methods.config_scm import LossConfig, create_config
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.methods.trajectory_matching.sCM import TrigFlowPrecond
from fastgen.networks.module import FastGenModule
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import (
    AlphasNoiseSchedule,
    CogVideoXNoiseSchedule,
    EDMNoiseSchedule,
    RFNoiseSchedule,
)
from fastgen.utils.tensor import expand_like
from fastgen.utils.tree_utils import assert_same
from tests.model_utils import StudentAsTeacher
from tests.utils import RunIf, assert_autocast_weight_cache_neutral, check_grad_zero


@pytest.fixture(params=[True, False], ids=["cd", "ct"])
def get_model_data(request):
    """Consistency distillation (``use_cd=True``) or the config default, consistency training without a teacher."""
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    config = create_config()
    instance = config.model
    opts = [
        "-",
        "backbone.img_resolution=2",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=False",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.loss_config.use_cd = request.param
    instance.input_shape = [3, 2, 2]

    model = SCMModel(instance)
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
    assert model.config.sample_t_cfg.train_p_mean == -1.0
    assert model.config.sample_t_cfg.train_p_std == 1.4
    assert model.config.net.backbone.dropout == 0.2
    # Distillation wraps the teacher in TrigFlow; consistency training has no teacher.
    assert (model.teacher_trigflow is not None) == model.config.loss_config.use_cd
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "scm_loss" in loss_map
    assert "unweighted_scm_loss" in loss_map
    assert "gen_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)


@pytest.mark.parametrize("get_model_data", [True], ids=["cd"], indirect=True)
def test_cd_guides_trigflow_prediction_before_converting_to_velocity(get_model_data, monkeypatch):
    model, data = get_model_data
    calls = []

    def teacher_guidance(ctx):
        assert ctx.guidance_cfg is model.config.guidance_cfg
        assert ctx.skip_inactive is False
        assert ctx.net is model.teacher_trigflow
        prediction = ctx.predict("pos", condition=ctx.condition)
        expected = model.teacher_trigflow(
            ctx.x_t, ctx.t, condition=ctx.condition, apply_output_conditioning=False, output_dtype=None
        )
        torch.testing.assert_close(prediction, expected)
        guided = prediction + 1
        calls.append(guided)
        return guided

    original_jvp = model._jvp
    velocities = []

    def jvp(x_t_hat, t_hat, dxt_dt, **kwargs):
        # The JVP receives velocity after guidance has returned its final F_theta, in high precision.
        torch.testing.assert_close(dxt_dt, model.sigma_data * calls[0].to(dxt_dt.dtype), rtol=0, atol=0)
        velocities.append(dxt_dt)
        return original_jvp(x_t_hat, t_hat, dxt_dt, **kwargs)

    # A forward-only wrapper would hide this method and silently use generic CFG.
    monkeypatch.setattr(model.teacher, "predict_with_guidance", teacher_guidance, raising=False)
    monkeypatch.setattr(model, "_jvp", jvp)
    loss_map, _ = model.single_train_step(data, 0)
    assert len(calls) == len(velocities) == 1
    assert torch.isfinite(loss_map["total_loss"]).all()


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
        "backbone.r_timestep=False",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None
    instance.loss_config.use_cd = True
    instance.precision_amp_jvp = "float32"
    instance.input_shape = [3, 2, 2]

    model = SCMModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    neg_labels = torch.zeros(batch_size, 10)

    data = {
        "real": torch.randn(batch_size, 3, 2, 2).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_labels.to(model.device, model.precision),
    }

    # Run the training step under fp32 JVP setting
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions: same interface and keys as default path
    assert "total_loss" in loss_map
    assert "scm_loss" in loss_map
    assert "unweighted_scm_loss" in loss_map
    assert "gen_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)


@pytest.mark.parametrize(
    ("t_precision", "use_jvp_finite_diff"), [("float32", False), ("float32", True), ("bfloat16", False)]
)
def test_consistency_distillation_trains_in_a_coarser_time_precision(t_precision, use_jvp_finite_diff):
    """TrigFlow times take the wrapped network's time precision, in which guidance converts the teacher's F_theta."""
    gc.collect()
    config = create_config()
    instance = config.model
    opts = [
        "-",
        "backbone.img_resolution=2",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=False",
        f"+t_precision={t_precision}",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cpu")
    instance.precision = "float32"
    instance.precision_model = "float64"
    instance.pretrained_model_path = None
    instance.loss_config.use_cd = True
    instance.loss_config.use_jvp_finite_diff = use_jvp_finite_diff
    instance.input_shape = [3, 2, 2]
    model = SCMModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    labels = torch.nn.functional.one_hot(torch.tensor([3]), num_classes=10).float()
    data = {"real": torch.randn(1, 3, 2, 2), "condition": labels, "neg_condition": torch.zeros(1, 10)}
    loss_map, _ = model.single_train_step(data, 0)

    dtype = getattr(torch, t_precision)
    assert model.net.noise_scheduler.t_precision == model.net_trigflow.noise_scheduler.t_precision == dtype
    assert torch.isfinite(loss_map["total_loss"])


def _fp32_scm_model_with_student_teacher():
    gc.collect()
    config = create_config()
    instance = config.model
    opts = [
        "-",
        "backbone.img_resolution=2",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=False",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float32"  # the autocast weight cache only stores fp32 trainable weights
    instance.pretrained_model_path = None
    instance.loss_config.use_cd = True
    instance.input_shape = [3, 2, 2]
    model = SCMModel(instance)
    model.on_train_begin()
    model.teacher = StudentAsTeacher(model.net)
    model.teacher_trigflow = TrigFlowPrecond(model.teacher, sigma_data=model.sigma_data)
    return model


def test_cd_teacher_pass_keeps_student_gradients_under_autocast_cache():
    """The no_grad consistency-distillation teacher pass precedes the student's gradient forward. With
    a teacher view over the student's own weights it must not leave detached autocast casts for that
    forward to reuse: the same parameters must receive gradients as in plain fp32, with values independent
    of the autocast weight cache."""
    model = _fp32_scm_model_with_student_teacher()
    batch_size = 1
    real_data = torch.randn(batch_size, 3, 2, 2, device=model.device)
    labels = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10)
    condition = labels.to(model.device, torch.float32)
    neg_condition = torch.zeros(batch_size, 10, device=model.device)
    t = model.net.noise_scheduler.sample_t(batch_size, **model.sample_t_cfg.sample_t_kwargs(), device=model.device)

    def run_loss():
        scm_loss, *_ = model._compute_scm_loss(
            real_data, t, condition=condition, neg_condition=neg_condition, iteration=0
        )
        return scm_loss.mean()

    assert_autocast_weight_cache_neutral(model.net, run_loss, model.device.type)


def test_consistency_distillation_rejects_teachers_with_output_conditioning(monkeypatch):
    from fastgen.networks.network import FastGenNetwork

    def hook(self, prediction, *args, **kwargs):
        return prediction

    monkeypatch.setattr(FastGenNetwork, "_condition_output", hook)
    config = create_config()
    instance = config.model
    opts = [
        "-",
        "backbone.img_resolution=2",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=False",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cpu")
    instance.precision = "float32"
    instance.precision_model = "float64"
    instance.pretrained_model_path = None
    instance.loss_config.use_cd = True
    instance.input_shape = [3, 2, 2]
    with pytest.raises(NotImplementedError, match="output conditioning"):
        SCMModel(instance)


def _float64_consistency_training_model():
    """A CPU sCM model in float64 that trains by consistency training, so it needs no teacher."""
    gc.collect()
    config = create_config()
    instance = config.model
    opts = [
        "-",
        "backbone.img_resolution=2",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=False",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cpu")
    instance.precision = "float32"
    instance.precision_model = "float64"
    instance.pretrained_model_path = None
    instance.loss_config.use_cd = False
    instance.input_shape = [3, 2, 2]
    model = SCMModel(instance)
    model.on_train_begin()
    return model


def test_finite_difference_jvp_goes_one_sided_at_the_ends_of_the_wrapped_range(monkeypatch):
    """Next to EDM's ``min_t`` and ``max_t``, a central step in TrigFlow time would map outside EDM's range, whose
    validity check rejects it. The estimate goes one-sided there instead, keeps every evaluation inside the range,
    and still matches the exact JVP."""
    torch.manual_seed(0)
    model = _float64_consistency_training_model()
    model.loss_config.jvp_finite_diff_eps = 1e-6
    schedule = model.net.noise_scheduler
    trig_schedule = model.net_trigflow.noise_scheduler
    t = torch.tensor([schedule.min_t * (1 + 1e-9), 1.0, schedule.max_t * (1 - 1e-9)], dtype=torch.float64)
    t_hat = model.net_trigflow.trigflow_t(t)
    real_data = torch.randn(3, 3, 2, 2, dtype=torch.float64)
    z = torch.randn_like(real_data) * model.sigma_data
    x_t_hat = trig_schedule.forward_process(real_data, z, t_hat)
    dxt_dt = trig_schedule.cond_velocity(real_data, z, t_hat)
    condition = torch.nn.functional.one_hot(torch.arange(3), num_classes=10).float()

    def jvp(use_jvp_finite_diff):
        model.loss_config.use_jvp_finite_diff = use_jvp_finite_diff
        return model._jvp(x_t_hat, t_hat, dxt_dt, condition=condition)

    # The network itself raises for any evaluation outside EDM's range.
    assert torch.isfinite(jvp(True)).all()

    # A smooth stand-in for the network, float64 throughout unlike EDM's float32 preconditioning, compares the
    # estimate with the exact JVP and records where it is evaluated.
    times = []

    def smooth_forward(x_t_hat, t_hat, condition=None, return_logvar=False, return_x0_pred=False):
        times.append(t_hat)
        return x_t_hat * expand_like(torch.cos(t_hat) ** 2, x_t_hat) + expand_like(torch.sin(t_hat), x_t_hat)

    monkeypatch.setattr(model.net_trigflow, "forward", smooth_forward)
    torch.testing.assert_close(jvp(True), jvp(False), rtol=1e-4, atol=1e-7)
    t_hat_plus, t_hat_minus = times[:2]
    low, high = model.net_trigflow.trigflow_t(torch.tensor([schedule.min_t, schedule.max_t], dtype=torch.float64))
    assert ((low <= t_hat_minus) & (t_hat_plus <= high)).all()
    # Forward at the low end, central inside the range, and backward at the high end.
    assert t_hat_minus[0] == t_hat[0] < t_hat_plus[0]
    assert t_hat_minus[1] < t_hat[1] < t_hat_plus[1]
    assert t_hat_minus[2] < t_hat[2] == t_hat_plus[2]


def test_trigflow_pure_noise_maps_into_an_rf_schedule_that_reaches_it():
    """TrigFlow's pure-noise time pi/2 is the image of RF's ``max_t = 1`` and maps back to a valid time, so the JVPs
    need not keep away from it."""

    class _RFNet:
        noise_scheduler = RFNoiseSchedule(min_t=0.0, max_t=1.0)

    wrapper = TrigFlowPrecond(_RFNet(), sigma_data=0.5)
    pure_noise = wrapper.trigflow_t(torch.tensor([1.0], dtype=torch.float64))
    assert pure_noise == torch.pi / 2

    x_t, t = wrapper._convert_trigflow_to_net_input(torch.ones(1, 2, dtype=torch.float64), pure_noise)
    assert torch.isfinite(x_t).all()
    assert 1 - 1e-12 < t <= 1


class _SmoothRFNet(torch.nn.Module):
    """An x0 prediction smooth in the state and in the time of an RF schedule that reaches pure noise."""

    def __init__(self):
        super().__init__()
        self.noise_scheduler = RFNoiseSchedule(min_t=0.0, max_t=1.0)

    def forward(
        self,
        x_t,
        t,
        condition=None,
        return_logvar=False,
        fwd_pred_type=None,
        apply_output_conditioning=True,
        output_dtype="same",
    ):
        del apply_output_conditioning, output_dtype
        assert fwd_pred_type == "x0" and not return_logvar
        return x_t * expand_like(torch.cos(t), x_t) + expand_like(t**2, x_t)


def _bare_scm_model(net_trigflow):
    """An ``SCMModel`` holding only what ``_jvp`` reads for its exact JVP."""
    model = SCMModel.__new__(SCMModel)
    torch.nn.Module.__init__(model)
    model.net_trigflow = net_trigflow
    model.precision_amp_jvp = None
    model.device = torch.device("cpu")
    model.loss_config = SimpleNamespace(use_jvp_finite_diff=False)
    return model


def test_exact_jvp_is_finite_at_pure_noise_and_keeps_the_time_derivative_next_to_it():
    """At TrigFlow's pure-noise time pi/2 of an RF schedule that reaches it, the exact JVP is finite. Just below
    pi/2, within 1e-4 of it, where the JVP used to clamp its times, it matches a float64 central difference along
    the noising path, time derivative included."""
    torch.manual_seed(0)
    wrapper = TrigFlowPrecond(_SmoothRFNet(), sigma_data=0.5)
    model = _bare_scm_model(wrapper)
    trig_schedule = wrapper.noise_scheduler
    t_hat = torch.tensor([torch.pi / 2, torch.pi / 2 - 1e-5], dtype=torch.float64)
    real_data = torch.randn(2, 3, dtype=torch.float64)
    z = torch.randn_like(real_data) * wrapper.sigma_data
    x_t_hat = trig_schedule.forward_process(real_data, z, t_hat)
    dxt_dt = trig_schedule.cond_velocity(real_data, z, t_hat)

    jvp = model._jvp(x_t_hat, t_hat, dxt_dt)
    assert torch.isfinite(jvp).all()

    def along_path(t_hat):
        return wrapper(trig_schedule.forward_process(real_data[1:], z[1:], t_hat), t_hat)

    step = 1e-8
    near = t_hat[1:]
    derivative = (along_path(near + step) - along_path(near - step)) / (2 * step)
    v_t = torch.cos(near) * torch.sin(near)
    torch.testing.assert_close(jvp[1:], expand_like(v_t, derivative) * derivative, rtol=1e-6, atol=1e-12)


def test_finite_difference_jvp_matches_the_exact_jvp_along_an_arbitrary_tangent():
    """The finite-difference estimate steps along whichever tangent it is given (under consistency distillation the
    teacher velocity, not the conditional velocity of the training pair) and matches the exact JVP along it."""
    torch.manual_seed(0)
    wrapper = TrigFlowPrecond(_SmoothRFNet(), sigma_data=0.5)
    model = _bare_scm_model(wrapper)
    model.loss_config = SimpleNamespace(use_jvp_finite_diff=False, jvp_finite_diff_eps=1e-6)
    t_hat = wrapper.trigflow_t(torch.tensor([0.3, 0.7], dtype=torch.float64))
    x_t_hat = torch.randn(2, 3, dtype=torch.float64)
    tangent = torch.randn_like(x_t_hat)

    exact = model._jvp(x_t_hat, t_hat, tangent)
    model.loss_config.use_jvp_finite_diff = True
    torch.testing.assert_close(model._jvp(x_t_hat, t_hat, tangent), exact, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("use_jvp_finite_diff", [False, True])
def test_jvp_autocast_dtype_below_the_state_precision_does_not_round_the_state(use_jvp_finite_diff):
    """A ``precision_amp_jvp`` narrower than the state only sets the autocast dtype. Rounding the state to it would be
    divided by ``sigma_data * sin(t_hat)`` in the x0 -> F_theta conversion, so the float64 state of this
    elementwise network, which autocast leaves alone, gives the JVP of autocast disabled."""
    torch.manual_seed(0)
    wrapper = TrigFlowPrecond(_SmoothRFNet(), sigma_data=0.5)
    model = _bare_scm_model(wrapper)
    model.loss_config = SimpleNamespace(use_jvp_finite_diff=use_jvp_finite_diff, jvp_finite_diff_eps=1e-6)
    t_hat = wrapper.trigflow_t(torch.tensor([1e-3, 0.5], dtype=torch.float64))
    x_t_hat = torch.randn(2, 3, dtype=torch.float64)
    velocity = torch.randn_like(x_t_hat)

    reference = model._jvp(x_t_hat, t_hat, velocity)
    model.precision_amp_jvp = torch.bfloat16
    jvp = model._jvp(x_t_hat, t_hat, velocity)

    assert jvp.dtype is torch.float64
    torch.testing.assert_close(jvp, reference, rtol=0, atol=0)


@pytest.mark.parametrize("use_jvp_finite_diff", [False, True])
def test_jvp_of_a_narrow_tangent_is_that_of_its_float64_upcast(use_jvp_finite_diff):
    """A ``dxt_dt`` narrower than the state is widened before it meets ``v_t``, so neither ``v_t`` nor their product
    is rounded to its dtype: a bf16 tangent gives the JVP of the same values in float64."""
    torch.manual_seed(0)
    wrapper = TrigFlowPrecond(_SmoothRFNet(), sigma_data=0.5)
    model = _bare_scm_model(wrapper)
    model.loss_config = SimpleNamespace(use_jvp_finite_diff=use_jvp_finite_diff, jvp_finite_diff_eps=1e-6)
    t_hat = wrapper.trigflow_t(torch.tensor([0.3, 0.7], dtype=torch.float64))
    x_t_hat = torch.randn(2, 3)
    velocity = torch.randn(2, 3).to(torch.bfloat16)

    torch.testing.assert_close(
        model._jvp(x_t_hat, t_hat, velocity), model._jvp(x_t_hat, t_hat, velocity.double()), rtol=0, atol=0
    )


class _RFFlowNet(FastGenNetwork):
    """A float32 flow prediction smooth in the state and time, converted by ``transform_output``."""

    def __init__(self):
        super().__init__(net_pred_type="flow", schedule_type="rf")

    def forward(
        self,
        x_t,
        t,
        condition=None,
        return_logvar=False,
        fwd_pred_type=None,
        apply_output_conditioning=True,
        output_dtype="same",
    ):
        assert not return_logvar
        flow = 0.2 * x_t.float() + expand_like(0.3 * t.float().square(), x_t)
        return self.transform_output(flow, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning)


def test_trigflow_converts_the_unrounded_x0_and_rounds_its_outputs_once():
    """The x0 -> F_theta conversion divides x0's rounding by ``sigma_data * sin(t_hat)``, so the wrapped network's x0
    is converted unrounded: in the bf16 state dtype, F_theta and the returned x0 are those of ``output_dtype=None``
    rounded once."""
    torch.manual_seed(0)
    wrapper = TrigFlowPrecond(_RFFlowNet(), sigma_data=0.5)
    t_hat = wrapper.trigflow_t(torch.tensor([1e-3, 0.5], dtype=torch.float64))
    x_t_hat = torch.randn(2, 64).to(torch.bfloat16)

    F_theta, x0 = wrapper(x_t_hat, t_hat, return_x0_pred=True)
    unrounded_F_theta, unrounded_x0 = wrapper(x_t_hat, t_hat, return_x0_pred=True, output_dtype=None)

    assert F_theta.dtype is x0.dtype is torch.bfloat16
    torch.testing.assert_close(F_theta, unrounded_F_theta.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(x0, unrounded_x0.to(torch.bfloat16), rtol=0, atol=0)


class _ModelPrecisionX0Net(FastGenModule):
    """``_SmoothRFNet``'s x0 from the state cast by ``cast_to_model_precision``, plus a condition in
    ``precision_model``."""

    def __init__(self):
        super().__init__()
        self.noise_scheduler = RFNoiseSchedule(min_t=0.0, max_t=1.0)

    def forward(self, x_t, t, condition=None, return_logvar=False, fwd_pred_type=None, **kwargs):
        del kwargs
        assert fwd_pred_type == "x0" and not return_logvar
        if self.precision_model is not None:
            assert_same(dtype=self.precision_model, condition=condition)
        x_t_model = self.cast_to_model_precision(x_t)
        x0 = x_t_model * expand_like(torch.cos(t), x_t_model) + expand_like(t**2, x_t_model) + condition
        return x0.to(dtype=x_t.dtype)


@pytest.mark.parametrize("use_jvp_finite_diff", [False, True])
def test_jvp_reads_network_inputs_promoted_with_the_autocast_dtype(use_jvp_finite_diff):
    """Inside the JVP, a wrapped network whose ``precision_model`` is narrower than the autocast dtype casts the state
    and the tangent to ``precision_model`` promoted with it and reads its condition in ``precision_model``, so its FP16
    input cast rounds neither: this elementwise network, which autocast leaves alone, gives the JVP of the uncast
    network."""
    torch.manual_seed(0)
    wrapper = TrigFlowPrecond(_ModelPrecisionX0Net(), sigma_data=0.5)
    model = _bare_scm_model(wrapper)
    model.loss_config = SimpleNamespace(use_jvp_finite_diff=use_jvp_finite_diff, jvp_finite_diff_eps=1e-3)
    t_hat = wrapper.trigflow_t(torch.tensor([0.3, 0.7], dtype=torch.float64))
    x_t_hat = torch.randn(2, 3)
    velocity = torch.randn(2, 3, dtype=torch.float64)
    condition = torch.randn(2, 3).to(dtype=torch.float16)

    reference = model._jvp(x_t_hat, t_hat, velocity, condition=condition.float())
    wrapper.net.prepare_module(device=None, precision_model=torch.float16)
    model.precision_amp_jvp = torch.bfloat16
    jvp = model._jvp(x_t_hat, t_hat, velocity, condition=condition)

    torch.testing.assert_close(jvp, reference, rtol=0, atol=0)


class _ScheduleNet(torch.nn.Module):
    """A wrapped network that only provides its noise schedule."""

    def __init__(self, noise_scheduler):
        super().__init__()
        self.noise_scheduler = noise_scheduler


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("snr_shift_scale", [1, 3])
def test_trigflow_alphas_input_jvp_is_smooth_at_and_near_pure_noise(device_type, snr_shift_scale):
    schedule = CogVideoXNoiseSchedule(snr_shift_scale=snr_shift_scale)
    wrapper = TrigFlowPrecond(_ScheduleNet(schedule), sigma_data=0.5).to(device_type)
    theta = torch.tensor(
        [torch.pi / 2, torch.pi / 2 - 1e-10, torch.pi / 2 - 1e-5],
        dtype=torch.float64,
        device=device_type,
        requires_grad=True,
    )
    x = torch.ones(3, 2, dtype=torch.float64, device=device_type)
    (mapped, time), (input_jvp, time_jvp) = torch.func.jvp(
        wrapper._convert_trigflow_to_net_input, (x, theta), (torch.zeros_like(x), torch.ones_like(theta))
    )
    # Independent normalized-VP formula: c(theta) = (cos^2(theta) + sigma_data^2 sin^2(theta))^(-1/2).
    norm_squared = theta.cos().square() + 0.25 * theta.sin().square()
    coeff = norm_squared.rsqrt()
    derivative = 0.75 * theta.sin() * theta.cos() / norm_squared.pow(1.5)
    torch.testing.assert_close(mapped, coeff[:, None].expand_as(x), rtol=1e-12, atol=0)
    # The scaling uses the coefficients of the rounded matched time, exact up to its resolution near pure noise.
    torch.testing.assert_close(input_jvp, derivative[:, None].expand_as(x), rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(torch.autograd.grad(mapped.sum(), theta)[0], 2 * derivative, rtol=1e-12, atol=1e-15)
    assert torch.isfinite(time_jvp).all()
    # Alpha is linear in time into pure noise, so the matched time resolves the SNR there.
    torch.testing.assert_close(schedule.alpha(time), (theta.cos() * coeff).detach(), rtol=0, atol=1e-16)


def test_trigflow_alphas_scaling_respects_inverse_saturation():
    schedule = AlphasNoiseSchedule(torch.tensor([0.8, 0.5, 0.2], dtype=torch.float64))
    wrapper = TrigFlowPrecond(_ScheduleNet(schedule), sigma_data=0.5)
    theta = torch.tensor([0.0, 0.1, 0.9, 1.5, torch.pi / 2], dtype=torch.float64)
    x = torch.ones(5, 2, dtype=torch.float64)
    mapped, t = wrapper._convert_trigflow_to_net_input(x, theta)
    expected = (schedule.alpha(t).square() + 4 * schedule.sigma(t).square()).sqrt()
    torch.testing.assert_close(mapped, expected[:, None].expand_as(x), rtol=1e-12, atol=0)
    tangent = torch.func.jvp(
        wrapper._convert_trigflow_to_net_input, (x, theta), (torch.zeros_like(x), torch.ones_like(theta))
    )[1][0]
    assert torch.equal(tangent[[0, 1, 3, 4]], torch.zeros_like(tangent[[0, 1, 3, 4]]))


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_trigflow_constant_alphas_tables_have_constant_scaling(probability):
    schedule = AlphasNoiseSchedule(torch.tensor([probability, probability], dtype=torch.float64))
    wrapper = TrigFlowPrecond(_ScheduleNet(schedule), sigma_data=0.5)
    theta = torch.tensor([0.0, 0.3, torch.pi / 2], dtype=torch.float64)
    x = torch.ones(3, 2, dtype=torch.float64)
    (mapped, _), (tangent, _) = torch.func.jvp(
        wrapper._convert_trigflow_to_net_input, (x, theta), (torch.zeros_like(x), torch.ones_like(theta))
    )
    torch.testing.assert_close(mapped, torch.full_like(x, (probability + 4 * (1 - probability)) ** 0.5))
    torch.testing.assert_close(tangent, torch.zeros_like(x), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("max_t", [80.0, 800.0])
def test_trigflow_round_trip_keeps_edm_max_t_in_range_with_its_derivative(max_t, dtype):
    """Next to pure noise the TrigFlow time resolves its distance to pi/2 only absolutely, so EDM's max_t comes back
    past itself. ``sqrt_snr_to_t`` clamps it back after its cast and keeps the derivative of t = sigma_d tan(t_hat)."""
    schedule = EDMNoiseSchedule(max_t=max_t)
    wrapper = TrigFlowPrecond(_ScheduleNet(schedule), sigma_data=0.5)
    t_hat = wrapper.trigflow_t(torch.tensor([schedule.min_t, max_t], dtype=dtype))
    x = torch.ones(2, 3, dtype=dtype)
    (_, t), (_, tangent) = torch.func.jvp(
        wrapper._convert_trigflow_to_net_input, (x, t_hat), (torch.zeros_like(x), torch.ones_like(t_hat))
    )
    assert t.dtype == dtype and bool(schedule.is_t_valid(t)) and t[1].item() == max_t
    torch.testing.assert_close(tangent.double(), 0.5 / t_hat.double().cos().square(), rtol=1e-6, atol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize(
    "make_schedule",
    [
        pytest.param(lambda: AlphasNoiseSchedule(torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)), id="alphas"),
        pytest.param(lambda: RFNoiseSchedule(max_t=1.0), id="rf"),
    ],
)
def test_trigflow_pure_noise_round_trips_into_the_wrapped_range_in_every_precision(make_schedule, dtype):
    """TrigFlow's pure noise is pi/2, which float32 rounds past pi/2, where cos turns negative: ``trigflow_t`` clamps it
    back, so the conversion maps it to the wrapped max_t."""
    schedule = make_schedule()
    wrapper = TrigFlowPrecond(_ScheduleNet(schedule), sigma_data=0.5)
    t_hat = wrapper.trigflow_t(torch.tensor([schedule.max_t], dtype=dtype))
    assert t_hat.dtype == dtype and t_hat.item() <= torch.pi / 2
    _, t = wrapper._convert_trigflow_to_net_input(torch.ones(1, 2, dtype=dtype), t_hat)
    assert t.dtype == dtype and schedule.min_t <= t.item() <= schedule.max_t
    assert t.item() == pytest.approx(schedule.max_t, abs=1e-6)


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))])
@pytest.mark.parametrize("return_logvar", [False, True])
def test_trigflow_native_v_prediction_keeps_the_full_zero_snr_derivative(device_type, return_logvar):
    """A v network converts with the coefficients of its own time, which resolve the SNR into pure noise."""

    class VNet(torch.nn.Module):
        net_pred_type = "v"

        def __init__(self):
            super().__init__()
            self.noise_scheduler = AlphasNoiseSchedule(torch.tensor([1.0, 0.0], dtype=torch.float64))

        def forward(
            self,
            x,
            t,
            condition=None,
            return_logvar=False,
            fwd_pred_type=None,
            apply_output_conditioning=True,
            output_dtype="same",
        ):
            del apply_output_conditioning, output_dtype
            assert fwd_pred_type == "x0"
            v = 0.2 * x + expand_like(0.3 * t.square(), x)
            prediction = self.noise_scheduler.v_to_x0(x, v, t)
            return (prediction, torch.zeros_like(t)) if return_logvar else prediction

    wrapper = TrigFlowPrecond(VNet(), sigma_data=0.5).to(device_type)
    theta = torch.tensor(
        [0.2, 0.7, torch.pi / 2 - 1e-8, torch.pi / 2], dtype=torch.float64, device=device_type, requires_grad=True
    )
    x = torch.arange(1, 9, dtype=torch.float64, device=device_type).reshape(4, 2)

    def reference(x, theta):
        c, s = theta.cos(), theta.sin()
        norm = (c.square() + 0.25 * s.square()).sqrt()
        alpha, sigma = c / norm, 0.5 * s / norm
        time = 0.5 * (1 - alpha)  # Exact inverse for the [1,0] table, where alpha = 1 - 2t.
        teacher_x = x / norm[:, None]
        v = 0.2 * teacher_x + 0.3 * time[:, None].square()
        x0 = alpha[:, None] * teacher_x - sigma[:, None] * v
        return (c[:, None] * x - x0) / (0.5 * s[:, None])

    tangents = (torch.zeros_like(x), torch.ones_like(theta))

    def forward(x, theta):
        result = wrapper(x, theta, return_logvar=return_logvar, return_x0_pred=True)
        assert result[-1].shape == x.shape
        if return_logvar:
            assert torch.equal(result[1], torch.zeros_like(theta))
        return result[0]

    actual, jvp = torch.func.jvp(forward, (x, theta), tangents)
    expected, expected_jvp = torch.func.jvp(reference, (x, theta), tangents)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(jvp, expected_jvp, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), theta)[0], expected_jvp.sum(dim=1), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    ("make_schedule", "times", "sqrt_snr"),
    [
        (lambda: RFNoiseSchedule(min_t=0.0, max_t=1.0), [0.2, 0.5, 0.9], lambda t: (1 - t) / t),
        (EDMNoiseSchedule, [0.1, 1.0, 10.0], lambda t: 1 / t),
    ],
    ids=["rf", "edm"],
)
def test_prior_weighting_is_the_wrapped_schedules_sqrt_snr(make_schedule, times, sqrt_snr):
    """The prior weight 1 / (sigma_d tan(t_hat)) of the TrigFlow time is the wrapped schedule's sqrt SNR
    alpha_t / sigma_t: 1 / sigma_t only where alpha_t = 1, as on EDM."""
    torch.manual_seed(0)
    wrapper = TrigFlowPrecond(_ScheduleNet(make_schedule()), sigma_data=0.5)
    model = _bare_scm_model(wrapper)
    model.sigma_data = wrapper.sigma_data
    model.loss_config = SimpleNamespace(
        tangent_warmup_steps=0,
        tangent_warmup_const=0.1,
        g_norm_spatial_invariance=False,
        prior_weighting_enabled=True,
    )
    t = torch.tensor(times, dtype=torch.float64)
    F_theta, F_theta_jvp, x_t_hat, dxt_dt = torch.randn(4, 3, 5, dtype=torch.float64)
    logvar = torch.zeros(3, 1, dtype=torch.float64)

    def scm_loss(prior_weighting_enabled):
        model.loss_config.prior_weighting_enabled = prior_weighting_enabled
        loss, _, _ = model._scm_pred_to_loss(
            F_theta, F_theta_jvp, x_t_hat, dxt_dt, logvar, wrapper.trigflow_t(t), iteration=1
        )
        return loss

    torch.testing.assert_close(scm_loss(True) / scm_loss(False), sqrt_snr(t))


def test_scm_loss_of_a_narrow_time_is_that_of_its_float64_upcast():
    """A time narrower than the loss arithmetic is widened before the schedule evaluates it, so neither the TrigFlow
    coefficients nor the prior weight is rounded to its dtype: a float32 time gives the loss of the same values in
    float64."""
    torch.manual_seed(0)
    wrapper = TrigFlowPrecond(_ScheduleNet(EDMNoiseSchedule()), sigma_data=0.5)
    model = _bare_scm_model(wrapper)
    model.sigma_data = wrapper.sigma_data
    model.loss_config = LossConfig(tangent_warmup_steps=0, prior_weighting_enabled=True)
    t_hat = wrapper.trigflow_t(torch.tensor([0.1, 1.0, 10.0], dtype=torch.float32))
    F_theta, F_theta_jvp, x_t_hat, dxt_dt = torch.randn(4, 3, 5, dtype=torch.float64)
    logvar = torch.zeros(3, 1, dtype=torch.float64)

    def scm_loss(t_hat):
        loss, unweighted, _ = model._scm_pred_to_loss(F_theta, F_theta_jvp, x_t_hat, dxt_dt, logvar, t_hat, iteration=1)
        return loss, unweighted

    torch.testing.assert_close(scm_loss(t_hat), scm_loss(t_hat.double()), rtol=0, atol=0)


def test_scm_loss_and_gradients_are_invariant_to_spatial_replication():
    """With RMS tangent normalization, repeating identical pixels preserves the per-sample loss and the
    gradient accumulated into their shared prediction. An extra division by the image size breaks this."""
    wrapper = TrigFlowPrecond(_ScheduleNet(EDMNoiseSchedule()), sigma_data=0.5)
    model = _bare_scm_model(wrapper)
    model.sigma_data = wrapper.sigma_data
    model.loss_config = LossConfig(
        tangent_warmup_steps=0,
        g_norm_spatial_invariance=True,
        prior_weighting_enabled=False,
    )
    t_hat = wrapper.trigflow_t(torch.tensor([0.1, 1.0], dtype=torch.float64))

    def loss_and_gradients(side):
        prediction = torch.tensor([[0.2, -0.4, 0.6], [-0.3, 0.1, 0.5]], dtype=torch.float64, requires_grad=True)
        logvar = torch.tensor([[0.25], [-0.5]], dtype=torch.float64, requires_grad=True)

        def image(values):
            return values.reshape(2, 3, 1, 1).expand(-1, -1, side, side)

        loss, unweighted, _ = model._scm_pred_to_loss(
            F_theta=image(prediction),
            F_theta_jvp=image(torch.full_like(prediction, 0.3)),
            x_t_hat=image(torch.full_like(prediction, -0.2)),
            dxt_dt=image(torch.full_like(prediction, 0.4)),
            logvar=logvar,
            t_hat=t_hat,
            iteration=1,
        )
        prediction_grad, logvar_grad = torch.autograd.grad(loss.sum(), (prediction, logvar))
        assert prediction_grad.abs().sum() > 0
        return loss, unweighted, prediction_grad, logvar_grad

    for original, replicated in zip(loss_and_gradients(1), loss_and_gradients(4)):
        torch.testing.assert_close(original, replicated, rtol=1e-12, atol=1e-12)
