# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Callable
import pytest
import gc

import torch

from fastgen.configs.config_utils import override_config_with_opts
from fastgen.methods import TCMModel
import fastgen.methods.trajectory_matching.TCM as TCM_module
from fastgen.methods.trajectory_matching.TCM import TCMPrecond
from fastgen.utils.tensor import expand_like
from fastgen.utils.tree_utils import per_modality, randn_like
from tests.test_cmmodel import _assert_trains_both_heads, _joint_kandinsky6_config, _joint_kandinsky6_data
from tests.utils import RunIf, check_grad_zero, run_distributed_test
from fastgen.configs.instantiate import instantiate


class MockNetwork(torch.nn.Module):
    """Mock network for testing TCMPrecond"""

    def __init__(self, output_shape, device=torch.device("cpu")):
        super().__init__()
        self.output_shape = output_shape
        self.device = device

        # Mock noise scheduler
        self.noise_scheduler = type("MockScheduler", (), {"min_t": 0.0, "max_t": 10.0})()

        # Mock prediction type
        self.net_pred_type = "x0"

        # Simple linear layer to make it a real network
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x_t, t, condition=None, fwd_pred_type="x0", **kwargs):
        # Return a tensor (or a dict of them) with the shape of the input but with modified values
        # to distinguish between teacher and student outputs
        if self.training:
            # simulate randomness, e.g., from dropout layers
            x_t = per_modality(torch.mul, x_t, randn_like(x_t))

        # Teacher and student return differently scaled inputs
        return per_modality(torch.mul, x_t, 0.5 if hasattr(self, "_is_teacher") else 2.0)


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    config_module = importlib.import_module("fastgen.configs.methods.config_tcm")
    config = config_module.create_config()
    assert config.model.transition_t == 1.0

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
    instance.input_shape = [3, 2, 2]

    model = TCMModel(instance)
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


@pytest.fixture
def get_tcm_precond_setup():
    """Setup for TCMPrecond tests"""
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create mock networks
    net_t = MockNetwork([3, 4, 4], device)
    net_t._is_teacher = True  # Mark as teacher for different behavior
    net_s = MockNetwork([3, 4, 4], device)

    # Create TCMPrecond
    transition_t = 2.0
    tcm_precond = TCMPrecond(net_t, net_s, transition_t)
    tcm_precond.to(device)

    # Test data
    batch_size = 4
    x_t = torch.randn(batch_size, 3, 4, 4, device=device)
    t = torch.tensor([1.0, 1.5, 2.5, 3.0], device=device)  # Mix of first/second stage
    condition = torch.randn(batch_size, 10, device=device)

    return tcm_precond, x_t, t, condition


def test_single_train_step_update(get_model_data):
    model, data = get_model_data
    # Run the training step
    assert model.config.transition_t == 1.0
    assert model.config.sample_t_cfg.train_p_mean == 0.0
    assert model.config.sample_t_cfg.train_p_std == 0.2
    assert model.net_tcm.net_s.training
    assert model.net_tcm.net_t.training

    # check intialization
    net_s_params = list(model.net_tcm.net_s.parameters())
    net_t_params = list(model.net_tcm.net_t.parameters())
    assert len([p for p in net_t_params if p.requires_grad]) == 0
    assert torch.allclose(net_s_params[0], net_t_params[0])
    assert torch.allclose(net_s_params[1], net_t_params[1])
    assert torch.allclose(net_s_params[-1], net_t_params[-1])

    # check single train step output
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "cm_loss" in loss_map
    assert "loss_boundary" in loss_map
    assert "unweighted_cm_loss" in loss_map
    assert all(torch.isfinite(value).all() for value in loss_map.values())
    assert "gen_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)


@pytest.mark.parametrize("boundary_prob", [0.0, 1.0])
def test_single_train_step_empty_loss_subset_is_finite(get_model_data, boundary_prob):
    model, data = get_model_data
    model.config.boundary_prob = boundary_prob

    loss_map, _ = model.single_train_step(data, 0)

    assert all(torch.isfinite(value).all() for value in loss_map.values())
    empty_subset_loss = "loss_boundary" if boundary_prob == 0.0 else "cm_loss"
    assert loss_map[empty_subset_loss].item() == 0.0
    assert not loss_map[empty_subset_loss].requires_grad
    loss_map["total_loss"].backward()


@pytest.mark.parametrize("multimodal", [False, True])
def test_single_train_step_splits_each_modality_loss_at_the_boundary(get_model_data, monkeypatch, multimodal):
    """Each modality's per-sample loss splits into the mean over regular samples and the mean over the boundary
    samples at transition_t, which the total weights by w_boundary; a plain tensor keeps its unprefixed loss keys."""
    model, _ = get_model_data
    model.config.boundary_prob = 0.5
    per_sample = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64, device=model.device)
    losses = {"video": per_sample, "audio": 3 * per_sample} if multimodal else per_sample
    sampled = {}

    def compute_cm_loss(net, real_data, t, condition=None, neg_condition=None):
        sampled["t"] = t
        return losses, per_modality(lambda loss: loss / 10, losses), real_data

    monkeypatch.setattr(model, "_compute_cm_loss", compute_cm_loss)
    torch.manual_seed(0)
    loss_map, _ = model.single_train_step({"real": torch.randn(4, 3, 2, 2, device=model.device)}, 0)

    boundary = sampled["t"] == model.config.transition_t
    assert boundary.sum() == 2
    prefixed = {"video_": losses["video"], "audio_": losses["audio"]} if multimodal else {"": losses}
    names = ("cm_loss", "loss_boundary", "unweighted_cm_loss")
    assert set(loss_map) == {prefix + name for prefix in prefixed for name in names} | {"total_loss"}
    total = 0
    for prefix, loss in prefixed.items():
        cm_loss, loss_boundary = loss[~boundary].mean(), loss[boundary].mean()
        torch.testing.assert_close(loss_map[prefix + "cm_loss"], cm_loss, rtol=0, atol=0)
        torch.testing.assert_close(loss_map[prefix + "loss_boundary"], loss_boundary, rtol=0, atol=0)
        torch.testing.assert_close(
            loss_map[prefix + "unweighted_cm_loss"], (loss / 10)[~boundary].mean(), rtol=0, atol=0
        )
        total = total + cm_loss + model.config.w_boundary * loss_boundary
    # The unimodal total is bit-for-bit the former cm_loss + w_boundary * loss_boundary.
    exact = {} if multimodal else dict(rtol=0, atol=0)
    torch.testing.assert_close(loss_map["total_loss"], total, **exact)


@pytest.mark.parametrize("is_boundary", [False, True])
def test_batch_size_one_loss_has_gradient_for_each_partition(is_boundary):
    per_sample_loss = torch.randn(1, requires_grad=True)
    mask_t = torch.tensor([is_boundary])

    loss_boundary = TCMModel._mean_or_zero(per_sample_loss[mask_t])
    cm_loss = TCMModel._mean_or_zero(per_sample_loss[~mask_t])
    total_loss = cm_loss + 0.1 * loss_boundary

    assert total_loss.requires_grad
    total_loss.backward()
    expected_grad = torch.tensor([0.1 if is_boundary else 1.0])
    torch.testing.assert_close(per_sample_loss.grad, expected_grad)


def test_ct_ratio(get_model_data):
    model, _ = get_model_data

    # instantiate ct_schedule callback
    config_module = importlib.import_module("fastgen.configs.methods.config_tcm")
    config = config_module.create_config()
    ct_schedule_callback = instantiate(config.trainer.callbacks.ct_schedule)
    ct_schedule_callback.config = config

    # check ct ratio (starts at ratio_limit for 2nd stage of TCM)
    ct_schedule_callback.on_train_begin(model, iteration=100_000)
    assert model.ratio == config.trainer.callbacks.ct_schedule.ratio_limit


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


def test_init_optimizers_warns_on_each_init_without_the_stage_one_checkpoint(get_model_data, monkeypatch):
    """The trainer loads the stage-1 consistency model from pretrained_ckpt_path, which sets resume_iter, before
    the optimizers are built; without it, every init_optimizers call warns (the fixture's own call ran unobserved),
    and none does once resume_iter is set."""
    model, _ = get_model_data
    warnings = []
    monkeypatch.setattr(TCM_module.logger, "warning", warnings.append)

    model.init_optimizers()
    [message] = warnings
    assert "pretrained_ckpt_path" in message

    warnings.clear()
    model.resume_iter = 0
    model.init_optimizers()
    assert warnings == []


def test_sampling_below_transition_t_is_rejected_at_construction():
    """Times below transition_t carry no gradient, so the effective lower sampling bound must be at least
    transition_t; the check runs at construction instead of on one rank inside the training step."""
    gc.collect()
    instance = importlib.import_module("fastgen.configs.methods.config_tcm").create_config().model
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
    instance.input_shape = [3, 2, 2]
    instance.sample_t_cfg.min_t = instance.transition_t / 2
    with pytest.raises(ValueError, match="transition_t"):
        TCMModel(instance)


def test_tcm_precond_forward_mixed_stages(get_tcm_precond_setup):
    """Test forward pass with mixed first/second stage samples"""
    tcm_precond, x_t, t, condition = get_tcm_precond_setup

    # first: test in eval mode
    tcm_precond.eval()

    # t = [1.0, 1.5, 2.5, 3.0] with transition_t = 2.0
    # First two samples in first stage, last two in second stage

    with torch.no_grad():
        output = tcm_precond(x_t, t, condition=condition)

    # First two samples should use teacher (x_t * 0.5)
    # Last two samples should use student (x_t * 2.0)
    expected = x_t.clone()
    expected[:2] *= 0.5  # Teacher output for first stage
    expected[2:] *= 2.0  # Student output for second stage

    assert torch.allclose(output, expected, atol=1e-6)

    # second: test in training mode
    tcm_precond.train()

    # get random scale (should be shared across teacher and student) and reset RNG state
    with torch.random.fork_rng(devices=[x_t.device] if x_t.device.type == "cuda" else []):
        scale = torch.randn_like(x_t)

    with torch.no_grad():
        output = tcm_precond(x_t, t, condition=condition)

    # First two samples should use teacher (x_t * scale * 0.5)
    # Last two samples should use student (x_t * scale * 2.0)
    expected = x_t.clone()
    expected[:2] *= scale[:2] * 0.5  # Teacher output for first stage
    expected[2:] *= scale[2:] * 2.0  # Student output for second stage

    assert torch.allclose(output, expected, atol=1e-6)

    # simulate CM loss (expect same output due to shared randomness)
    device = x_t.device
    t_stage1 = torch.tensor([1.0, 1.5, 0.5, 0.25], device=device)  # First stage only
    t_stage2 = torch.tensor([2.5, 3.0, 4.0, 5.0], device=device)  # Second stage only
    for t_test in [t, t_stage1, t_stage2]:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            pred_grad = tcm_precond(x_t, t_test, condition=condition)
        with torch.no_grad():
            pred_no_grad = tcm_precond(x_t, t_test, condition=condition)

        assert torch.allclose(pred_grad, pred_no_grad, atol=1e-6)


def test_tcm_precond_blends_each_modality(get_tcm_precond_setup):
    """A ``{"video", "audio"}`` state takes each modality's second-stage samples from the student."""
    tcm_precond, x_t, t, condition = get_tcm_precond_setup
    tcm_precond.eval()
    state = {"video": x_t, "audio": torch.randn(x_t.shape[0], 6, 5, device=x_t.device)}

    with torch.no_grad():
        output = tcm_precond(state, t, condition=condition)

    # t = [1.0, 1.5, 2.5, 3.0] with transition_t = 2.0: teacher (x 0.5), then student (x 2.0)
    scale = torch.tensor([0.5, 0.5, 2.0, 2.0], device=x_t.device)
    assert set(output) == set(state)
    for key, value in state.items():
        torch.testing.assert_close(output[key], value * expand_like(scale, value))


@pytest.mark.parametrize("use_cd", [False, True])
def test_joint_kandinsky6_step_has_losses_per_modality(tmp_path, use_cd):
    """TCM on a ``{"video", "audio"}`` state reports consistency and boundary losses per modality, trains both
    output heads and leaves the frozen stage-1 model without gradients."""
    config = importlib.import_module("fastgen.configs.methods.config_tcm").create_config()
    instance = _joint_kandinsky6_config(config.model, tmp_path)
    instance.loss_config.use_cd = use_cd
    instance.transition_t = 0.3  # in the network's rectified-flow time
    instance.sample_t_cfg.min_t = instance.transition_t
    instance.sample_t_cfg.max_t = 1.0
    instance.boundary_prob = 0.5
    model = TCMModel(instance)
    model.on_train_begin()
    model.ratio = 0.9
    second_stage = []
    model.net_tcm.register_forward_pre_hook(lambda module, args: second_stage.append(args[1] >= module.transition_t))

    torch.manual_seed(0)
    loss_map, outputs = model.single_train_step(_joint_kandinsky6_data(), 0)

    # The student pass is second stage; the target pass blends the stage-1 model's outputs with the student's.
    student_pass, target_pass = second_stage
    assert student_pass.all() and target_pass.any() and not target_pass.all()
    modalities, names = ("video", "audio"), ("cm_loss", "loss_boundary", "unweighted_cm_loss")
    assert set(loss_map) == {f"{modality}_{name}" for modality in modalities for name in names} | {"total_loss"}
    assert all(torch.isfinite(value) and value > 0 for value in loss_map.values())
    total = sum(
        loss_map[f"{modality}_cm_loss"] + instance.w_boundary * loss_map[f"{modality}_loss_boundary"]
        for modality in modalities
    )
    torch.testing.assert_close(loss_map["total_loss"], total)
    assert set(outputs["gen_rand_train"]) == set(modalities)
    loss_map["total_loss"].backward()
    _assert_trains_both_heads(model.net)
    assert all(parameter.grad is None for parameter in model.cm_teacher.parameters())


def _tiny_edm2_tcm():
    """A synthetic stage-1 state, not a trained consistency checkpoint."""
    instance = importlib.import_module("fastgen.configs.experiments.EDM2.config_tcm_s").create_config().model
    instance.precision_model = "float16"
    instance.precision_opt = "float32"
    instance.pretrained_model_path = None
    instance.device = torch.device("cuda", torch.cuda.current_device())
    instance.use_ema = False
    instance.enable_preprocessors = False
    instance.input_shape = [3, 4, 4]
    instance.net.update(
        img_resolution=4,
        label_dim=0,
        model_channels=16,
        channel_mult=[1],
        num_blocks=1,
        attn_resolutions=[],
        logvar_channels=8,
        dropout_resolutions=[4],
    )
    model = TCMModel(instance)
    # Nonzero output makes the neural path observable; these values also detect FP16 round trips.
    with torch.no_grad():
        model.net.unet.out_gain.fill_(0.1234567)
        model.cm_teacher.unet.out_gain.fill_(0.1234567)
        model.cm_teacher.unet.emb_fourier.freqs[0] = 0.1234567
    return model


def _check_tcm_teacher_storage(fsdp_context=None):
    from torch.distributed.tensor import DTensor

    model = _tiny_edm2_tcm()
    teacher = model.cm_teacher
    pristine = {
        name: value.detach().cpu().clone()
        for name, value in list(teacher.named_parameters()) + list(teacher.named_buffers())
    }
    assert any(not torch.equal(value, value.half().float()) for value in pristine.values())
    model.on_train_begin(fsdp_context=fsdp_context)

    assert teacher.is_prepared
    assert teacher.training and all(module.training for module in teacher.modules())
    assert "cm_teacher" not in model.model_dict
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert teacher.precision_model == torch.float16
    assert teacher._is_fsdp == (fsdp_context is not None)
    for name, value in list(teacher.named_parameters()) + list(teacher.named_buffers()):
        assert value.dtype == torch.float32, name
        if isinstance(value, DTensor):
            value = value.full_tensor()
        torch.testing.assert_close(value.detach().cpu(), pristine[name], rtol=0, atol=0)

    observed_dtypes = []
    handle = teacher.unet.out_conv.register_forward_hook(
        lambda module, args, output: observed_dtypes.append((args[0].dtype, output.dtype))
    )
    try:
        x = torch.randn(2, 3, 4, 4, device=model.device, dtype=torch.float32)
        t = torch.ones(2, device=model.device, dtype=model.net.noise_scheduler.t_precision)
        prediction = teacher(x, t)
    finally:
        handle.remove()
    assert observed_dtypes == [(torch.float16, torch.float16)]
    assert prediction.dtype == torch.float32 and torch.isfinite(prediction).all()
    assert not prediction.requires_grad
    model.net(x, t).square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.net.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    model.reshard_fsdp_modules()
    assert all(parameter.dtype == torch.float32 for parameter in teacher.parameters())


@RunIf(min_gpus=1)
def test_tcm_teacher_preserves_fp32_state_with_fp16_compute():
    _check_tcm_teacher_storage()


def _tcm_teacher_storage_fsdp_worker(rank, world_size):
    from fastgen.utils.distributed.fsdp import create_fsdp_context

    context = create_fsdp_context(reduce_dtype=torch.float32, min_num_params=1)
    # Each rank's pristine state must match the state FSDP synchronizes from rank zero.
    torch.manual_seed(1234)
    _check_tcm_teacher_storage(fsdp_context=context)
    return {"passed": True}


@RunIf(min_gpus=2)
def test_tcm_teacher_preserves_fp32_state_with_fp16_compute_fsdp():
    result = run_distributed_test(_tcm_teacher_storage_fsdp_worker, world_size=2, timeout=120)
    assert result["passed"]
