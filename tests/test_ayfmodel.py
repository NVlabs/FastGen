# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
from types import SimpleNamespace
import pytest

import torch

from fastgen.methods import AYFModel
from fastgen.configs.methods.config_ayf import create_config
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.networks.noise_schedule import TrigNoiseSchedule
from fastgen.utils.tensor import expand_like, high_precision_dtype
from tests.model_utils import StudentAsTeacher
from tests.utils import assert_autocast_weight_cache_neutral


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
        "backbone.label_dim=0",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.precision_amp_jvp = "float32"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.input_shape = [3, 2, 2]

    model = AYFModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    # Unconditional tiny-EDM data (label_dim=0 -> zero-width condition tensors).
    data = {
        "real": torch.randn(batch_size, 3, 2, 2).to(model.device, model.precision),
        "condition": torch.zeros(batch_size, 0).to(model.device, model.precision),
        "neg_condition": torch.zeros(batch_size, 0).to(model.device, model.precision),
    }
    return model, data


def test_ayf_loss_config_defaults(get_model_data):
    model, _ = get_model_data
    assert model.config.loss_config.use_jvp_finite_diff is False  # analytic torch.func.jvp
    assert model.config.loss_config.tangent_scale == 0.95
    assert model.config.loss_config.rms_clamp_min == 1.0
    assert model.tangent_scale == 0.95
    # use_cd=True: AYF always distills from a teacher (built by CM.build_model).
    assert model.config.loss_config.use_cd is True
    # No net_adapter (default): the teacher is a separate frozen network.
    assert model.config.net_adapter.type is None
    assert model.teacher is not model.net
    assert not any(p.requires_grad for p in model.teacher.parameters())
    assert all(p.requires_grad for p in model.net.parameters())


@pytest.mark.parametrize("iteration", [0, 1])
def test_ayf_edm_single_train_step(get_model_data, iteration):
    model, data = get_model_data

    loss_map, outputs = model.single_train_step(data, iteration)

    assert torch.isfinite(loss_map["total_loss"]).all()
    assert "gen_rand" in outputs
    loss_map["total_loss"].backward()
    gradients = [parameter.grad for parameter in model.net.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_lora_net_adapter_requires_capable_net():
    config = create_config()
    instance = config.model
    opts = [
        "-",
        "backbone.img_resolution=2",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.label_dim=0",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.pretrained_model_path = None
    instance.net_adapter.type = "lora"
    # EDM has no LoRA support; the adapter must fail loudly at build time.
    with pytest.raises(ValueError, match="LoRA-capable"):
        AYFModel(instance)


def test_emd_loss_gradient(get_model_data):
    """The AYF EMD loss yields a backward gradient of exactly 2 * normalize(emd_target) / numel,
    where emd_target = -(v_t - model_pred + (r - t) * jvp_out),
    normalize(x) = x / clamp(RMS(x), min=rms_clamp_min), and numel = C*H*W (the loss is
    mean-reduced over elements to match the reference's `.pow(2).mean()`)."""
    model, _ = get_model_data
    dev = model.device
    b, c, h, w = 2, 3, 2, 2

    model_pred = torch.randn(b, c, h, w, device=dev, requires_grad=True)
    v_t = torch.randn(b, c, h, w, device=dev)
    jvp_out = torch.randn(b, c, h, w, device=dev)
    t = torch.rand(b, device=dev) * 0.5 + 0.4
    r = t - 0.2

    loss, emd_target, weight = model._emd_pred_to_loss(
        model_pred=model_pred, v_t=v_t, jvp_out=jvp_out, t=t, r=r, iteration=10_000
    )

    assert loss.shape == (b,)
    assert torch.isfinite(loss).all()
    assert not emd_target.requires_grad and not weight.requires_grad
    # RMS-clamp weight is in (0, 1] (rms_clamp_min == 1.0).
    assert (weight > 0).all() and (weight <= 1.0 + 1e-6).all()

    loss.sum().backward()
    assert model_pred.grad is not None
    # Expected gradient: 2 * normalize(emd_target) / numel (mean reduction over C*H*W).
    numel = c * h * w
    rms = emd_target.flatten(1).pow(2).mean(dim=-1, keepdim=True).sqrt()
    clipped = emd_target.flatten(1) / rms.clamp(min=model.rms_clamp_min)
    expected = (2.0 * clipped.reshape(b, c, h, w) / numel).to(model_pred.grad.dtype)
    assert torch.allclose(model_pred.grad, expected, atol=1e-4, rtol=1e-3)


def test_emd_target_form(get_model_data):
    """emd_target = -(v_t - model_pred + tangent_scale * (r - t) * jvp_out) at full warmup."""
    model, _ = get_model_data
    dev = model.device
    b, c, h, w = 2, 3, 2, 2
    model_pred = torch.randn(b, c, h, w, device=dev, requires_grad=True)
    v_t = torch.randn(b, c, h, w, device=dev)
    jvp_out = torch.randn(b, c, h, w, device=dev)
    t = torch.full((b,), 0.6, device=dev)
    r = torch.full((b,), 0.4, device=dev)

    _, emd_target, _ = model._emd_pred_to_loss(model_pred, v_t, jvp_out, t, r, iteration=10_000)
    dtype = high_precision_dtype(model_pred, v_t, jvp_out)
    r_minus_t = (r - t).to(dtype).view(b, 1, 1, 1)
    jvp_term = model.tangent_scale * r_minus_t * jvp_out.to(dtype)
    expected = -(v_t.to(dtype) - model_pred.detach().to(dtype) + jvp_term)
    assert torch.allclose(emd_target, expected, atol=1e-5, rtol=1e-4)


def test_monitoring_uses_the_schedule_velocity_and_the_flow_map_jump(monkeypatch):
    """On any schedule, ``v_loss`` measures against the schedule's conditional velocity, and the x0 estimate takes
    the flow-map jump ``x_t - t u`` to time 0, exact at ``r = 0``, rather than reading ``u`` as instantaneous."""
    schedule = TrigNoiseSchedule()
    target = torch.randn(2, 4)

    class FlowMapToTarget(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.noise_scheduler = schedule

        def forward(self, x_t, t, r=None, condition=None, fwd_pred_type=None):
            return (x_t - target) / expand_like(t.to(x_t.dtype), x_t)

    model = object.__new__(AYFModel)
    torch.nn.Module.__init__(model)
    model.net = FlowMapToTarget()
    model.teacher_guidance_range = None
    model.tangent_scale = 1.0
    model.rms_clamp_min = 1.0
    model.loss_config = SimpleNamespace(tangent_warmup_steps=0)
    monkeypatch.setattr(model, "_teacher_velocity", lambda x_t, t, **kwargs: torch.zeros_like(x_t))
    real_data = torch.randn(2, 4)
    t = torch.tensor([0.4, 1.2], dtype=torch.float64)

    torch.manual_seed(0)
    noise = torch.randn_like(real_data)
    torch.manual_seed(0)
    with torch.no_grad():
        _, v_loss, _, _, _, x0_approx, _ = model._compute_emd_loss(real_data, t, t, iteration=0, skip_jvp=True)

    torch.testing.assert_close(x0_approx, target)
    model_pred = model.net(schedule.forward_process(real_data, noise, t), t)
    expected_v_loss = (model_pred - schedule.cond_velocity(real_data, noise, t)).pow(2).mean(dim=1)
    torch.testing.assert_close(v_loss, expected_v_loss)


def _fp32_ayf_model_with_student_teacher():
    gc.collect()
    config = create_config()
    instance = config.model
    opts = [
        "-",
        "backbone.img_resolution=2",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.label_dim=0",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float32"  # the autocast weight cache only stores fp32 trainable weights
    instance.precision_amp_jvp = "float32"
    instance.pretrained_model_path = None
    instance.input_shape = [3, 2, 2]
    model = AYFModel(instance)
    model.on_train_begin()
    model.teacher = StudentAsTeacher(model.net)
    return model


def test_warmup_teacher_pass_keeps_student_gradients_under_autocast_cache():
    """During guidance-distillation warmup ``_jvp`` is skipped, so only the clear before the student forward
    separates the no_grad teacher pass from the gradient forward. A teacher view over the student's own
    weights is the exposed case: the same parameters must receive gradients as in plain fp32, with values
    independent of the autocast weight cache."""
    model = _fp32_ayf_model_with_student_teacher()
    batch_size = 1
    real_data = torch.randn(batch_size, 3, 2, 2, device=model.device)
    condition = torch.zeros(batch_size, 0, device=model.device)
    t, _, _ = model._sample_t_r_buckets(batch_size)

    def run_loss():
        emd_loss, *_ = model._compute_emd_loss(
            real_data, t, t, iteration=0, condition=condition, neg_condition=condition, skip_jvp=True
        )
        return emd_loss.sum()

    assert_autocast_weight_cache_neutral(model.net, run_loss, model.device.type)
