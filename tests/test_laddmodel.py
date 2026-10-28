# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
from types import SimpleNamespace
from typing import Callable

import pytest
import attrs
import torch
from fastgen.callbacks.carry import CARRY_OUTPUT_KEY, CarryCallback
from fastgen.methods import DMD2Model, LADDModel
from fastgen.configs.methods.config_ladd import ModelConfig, create_config
from fastgen.configs.config_utils import override_config_with_opts
from tests.utils import check_grad_zero, record_generation_renoising


def test_ladd_inherits_the_shared_dmd2_training_engine():
    assert issubclass(LADDModel, DMD2Model)
    assert not issubclass(DMD2Model, LADDModel)
    for name in (
        "build_model",
        "_compute_real_feat",
        "_compute_gan_discriminator_losses",
        "_training_inputs",
        "_sample_t_and_eps",
        "gen_data_from_net",
        "_student_update_step",
        "_auxiliary_update_step",
        "single_train_step",
    ):
        assert getattr(LADDModel, name) is getattr(DMD2Model, name)


def test_ladd_always_disables_vsd():
    model = object.__new__(LADDModel)
    torch.nn.Module.__init__(model)
    model.config = ModelConfig()

    assert not model.use_vsd
    assert not hasattr(model.config, "vsd_loss_weight")


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    instance = ModelConfig()
    opts = ["-", "backbone.img_resolution=8", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.student_update_freq = 2
    instance.input_shape = [3, 8, 8]

    model = LADDModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    neg_condition = torch.zeros(batch_size, 10)

    # Create mock data
    data = {
        "idx": list(range(batch_size)),
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_condition.to(model.device, model.precision),
    }
    return model, data


def _real_feat_harness(gan_use_same_t_noise):
    model = object.__new__(LADDModel)
    torch.nn.Module.__init__(model)
    forward_process_calls = []
    teacher_calls = []

    class Scheduler:
        def forward_process(self, real_data, eps, t):
            forward_process_calls.append((real_data, eps, t))
            return real_data + eps

    def teacher(perturbed_real, t_real, **kwargs):
        teacher_calls.append((perturbed_real, t_real, kwargs))
        return [perturbed_real]

    model.config = SimpleNamespace(gan_use_same_t_noise=gan_use_same_t_noise)
    model.net = SimpleNamespace(noise_scheduler=Scheduler())
    model.teacher = teacher
    model.discriminator = SimpleNamespace(feature_indices={1, 3})
    return model, forward_process_calls, teacher_calls


def test_compute_real_feat_samples_independent_t_and_eps(monkeypatch):
    model, forward_process_calls, teacher_calls = _real_feat_harness(gan_use_same_t_noise=False)
    real_data = torch.zeros(2, 3)
    fake_t = torch.full((2,), 0.1)
    fake_eps = torch.full_like(real_data, 1.0)
    sampled_t = torch.full((2,), 0.7)
    sampled_eps = torch.full_like(real_data, 2.0)
    student_step = torch.full((2,), 1)
    sample_calls = []

    def sample_t_and_eps(batch_size, sample_t_cfg=None, student_step=None):
        sample_calls.append((batch_size, sample_t_cfg, student_step))
        return sampled_t, sampled_eps

    monkeypatch.setattr(model, "_sample_t_and_eps", sample_t_and_eps)
    condition = object()

    real_feat, t_real, perturbed_real = model._compute_real_feat(
        real_data, fake_t, fake_eps, condition, student_step=student_step
    )

    assert len(sample_calls) == 1
    assert sample_calls[0][:2] == (2, None)
    assert sample_calls[0][2] is student_step
    assert len(forward_process_calls) == 1
    forwarded_real, forwarded_eps, forwarded_t = forward_process_calls[0]
    assert forwarded_real is real_data
    assert forwarded_eps is sampled_eps
    assert forwarded_t is sampled_t
    assert t_real is sampled_t
    torch.testing.assert_close(perturbed_real, real_data + sampled_eps)
    assert len(real_feat) == 1 and real_feat[0] is perturbed_real
    assert len(teacher_calls) == 1
    teacher_input, teacher_t, teacher_kwargs = teacher_calls[0]
    assert teacher_input is perturbed_real
    assert teacher_t is sampled_t
    assert teacher_kwargs == {
        "condition": condition,
        "return_features_early": True,
        "feature_indices": {1, 3},
    }


def test_compute_real_feat_reuses_fake_t_and_eps_when_configured(monkeypatch):
    model, forward_process_calls, _ = _real_feat_harness(gan_use_same_t_noise=True)
    real_data = torch.zeros(2, 3)
    fake_t = torch.full((2,), 0.1)
    fake_eps = torch.full_like(real_data, 1.0)

    def unexpected_sample(*args, **kwargs):
        raise AssertionError("shared GAN noising must not resample t or eps")

    monkeypatch.setattr(model, "_sample_t_and_eps", unexpected_sample)

    _, t_real, perturbed_real = model._compute_real_feat(real_data, fake_t, fake_eps)

    assert len(forward_process_calls) == 1
    forwarded_real, forwarded_eps, forwarded_t = forward_process_calls[0]
    assert forwarded_real is real_data
    assert forwarded_eps is fake_eps
    assert forwarded_t is fake_t
    assert t_real is fake_t
    torch.testing.assert_close(perturbed_real, real_data + fake_eps)


def test_single_train_step_student_update(get_model_data, monkeypatch):
    model, data = get_model_data
    student_grid = []
    forward = model.net.forward

    def unexpected_sample(*args, **kwargs):
        raise AssertionError("a one-step student has no timestep choice to sample")

    def record_student_time(input_student, t_student, *args, **kwargs):
        student_grid.append(t_student.clone())
        return forward(input_student, t_student, *args, **kwargs)

    monkeypatch.setattr(model.net.noise_scheduler, "sample_student_timesteps", unexpected_sample)
    monkeypatch.setattr(model.net, "forward", record_student_time)
    # Run the training step
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "gan_loss_gen" in loss_map
    assert "vsd_loss" not in loss_map
    assert "fake_score_loss" not in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert callable(outputs["gen_rand"])
    assert isinstance(outputs["input_rand"], torch.Tensor)
    assert isinstance(outputs["gen_rand_train"], torch.Tensor)
    assert outputs["gen_rand"].keywords["neg_condition"] is data["neg_condition"]
    assert len(student_grid) == 1
    expected = model.net.noise_scheduler.student_grid()[0].to(student_grid[0]).expand_as(student_grid[0])
    torch.testing.assert_close(student_grid[0], expected)


def test_single_train_step_disc_update(get_model_data):
    model, data = get_model_data
    model.config.gan_r1_reg_weight = 0.1
    # Run the training step
    loss_map, outputs = model.single_train_step(data, 1)

    # Assertions
    assert "gan_loss_disc" in loss_map
    assert "gan_loss_ar1" in loss_map
    assert "vsd_loss" not in loss_map
    assert "fake_score_loss" not in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert callable(outputs["gen_rand"])
    assert isinstance(outputs["input_rand"], torch.Tensor)
    assert isinstance(outputs["gen_rand_train"], torch.Tensor)
    assert outputs["gen_rand"].keywords["neg_condition"] is data["neg_condition"]


@pytest.mark.parametrize(
    ("iteration", "update_hook"),
    ((0, "_student_update_step"), (1, "_auxiliary_update_step")),
)
def test_single_train_step_forwards_training_loss_scale(get_model_data, monkeypatch, iteration, update_hook):
    model, data = get_model_data
    loss_scale = torch.ones(data["real"].shape[0], device=model.device)
    training_inputs = model._training_inputs
    seen_loss_scales = []
    seen_data = []

    def training_inputs_with_scale(*args, **kwargs):
        inputs = training_inputs(*args, **kwargs)
        return attrs.evolve(inputs, loss_scale=loss_scale)

    def record_update(inputs, update_data):
        seen_data.append(update_data)
        seen_loss_scales.append(inputs.loss_scale)
        return {"total_loss": inputs.loss_scale.sum()}, {"gen_rand_train": torch.zeros_like(data["real"])}

    monkeypatch.setattr(model, "_training_inputs", training_inputs_with_scale)
    monkeypatch.setattr(model, update_hook, record_update)

    model.single_train_step(data, iteration)

    assert len(seen_data) == 1 and seen_data[0] is data
    assert seen_loss_scales == [loss_scale]


@pytest.mark.parametrize("iteration", [0, 1])
def test_single_train_step_uses_generation_hook_and_merges_results(get_model_data, monkeypatch, iteration):
    model, data = get_model_data
    generation_kwarg_marker = object()
    generation_metric = torch.ones((), device=model.device)
    training_inputs = model._training_inputs
    gen_data_from_net = model.gen_data_from_net
    seen_generation_kwargs = []

    def training_inputs_with_generation_kwargs(*args, **kwargs):
        inputs = training_inputs(*args, **kwargs)
        return attrs.evolve(inputs, generation_kwargs={"generation_kwarg_marker": generation_kwarg_marker})

    def generate_with_results(*args, generation_kwarg_marker=None, **kwargs):
        seen_generation_kwargs.append(generation_kwarg_marker)
        generation = gen_data_from_net(*args, **kwargs)
        return attrs.evolve(generation, metrics={**generation.metrics, "generation_metric": generation_metric})

    monkeypatch.setattr(model, "_training_inputs", training_inputs_with_generation_kwargs)
    monkeypatch.setattr(model, "gen_data_from_net", generate_with_results)

    loss_map, outputs = model.single_train_step(data, iteration)

    assert seen_generation_kwargs == [generation_kwarg_marker]
    assert "avg_t_student" in loss_map
    assert loss_map["generation_metric"] is generation_metric
    assert "generation_metric" not in outputs


def test_ladd_backward_simulation_is_opt_in():
    config = create_config()
    assert not config.model.backward_simulation
    assert "carry" not in config.trainer.callbacks


def test_ladd_backward_simulation_inputs_do_not_require_real_data(get_multistep_model_data):
    model, data = get_multistep_model_data
    model.config.backward_simulation = "default"
    data = {key: value for key, value in data.items() if key != "real"}

    _inputs = model._training_inputs(data, iteration=1)
    input_student = _inputs.input_student
    student_step = _inputs.student_step
    t = _inputs.t
    eps = _inputs.eps

    batch_size = len(data["idx"])
    assert input_student.shape[0] == batch_size
    assert student_step.shape == t.shape == (batch_size,)
    assert eps.shape[0] == batch_size

    with pytest.raises(ValueError, match="GAN-only discriminator updates require data\\['real'\\]"):
        model._auxiliary_update_step(_inputs, data)


def test_ladd_samples_only_the_main_noising_config(get_model_data, monkeypatch):
    model, data = get_model_data
    assert not hasattr(model.config, "fake_score_sample_t_overrides")
    model.config.gan_use_same_t_noise = True
    sample_calls = []
    sample_t_and_eps = model._sample_t_and_eps

    def record_sample(batch_size, sample_t_cfg=None, student_step=None):
        sample_calls.append((batch_size, sample_t_cfg))
        return sample_t_and_eps(batch_size, sample_t_cfg=sample_t_cfg, student_step=student_step)

    monkeypatch.setattr(model, "_sample_t_and_eps", record_sample)

    loss_map, _ = model.single_train_step(data, iteration=1)

    assert sample_calls == [(data["real"].shape[0], None)]
    assert "fake_score_loss" not in loss_map


def test_optimizers(get_model_data):
    model, data = get_model_data
    # Test both the student and discriminator optimizers.
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)

    # Test both optimizer zero-grad paths after at least one backward pass.
    model.optimizers_zero_grad(2)
    check_grad_zero(model.net)
    model.optimizers_zero_grad(3)
    check_grad_zero(model.discriminator)


def test_auxiliary_dicts_include_constructed_discriminator(get_model_data):
    model, _ = get_model_data

    assert model.use_gan
    assert not model.use_vsd
    assert not hasattr(model, "fake_score")
    assert not hasattr(model, "fake_score_optimizer")
    assert not hasattr(model, "fake_score_lr_scheduler")
    assert list(model.model_dict) == ["net", "discriminator"]
    assert list(model.optimizer_dict) == ["net", "discriminator"]
    assert list(model.scheduler_dict) == ["net", "discriminator"]
    assert model.model_dict["discriminator"] is model.discriminator
    assert model.optimizer_dict["discriminator"] is model.discriminator_optimizer
    assert model.scheduler_dict["discriminator"] is model.discriminator_lr_scheduler


def _multistep_model_data(**overrides):
    """Build a LADD model with two default student steps, and a matching batch."""
    instance = ModelConfig()
    opts = ["-", "backbone.img_resolution=8", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.student_update_freq = 2
    instance.input_shape = [3, 8, 8]

    # Enable multi-step distillation
    instance.student_grid_cfg.steps = 2
    for key, value in overrides.items():
        setattr(instance, key, value)

    model = LADDModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    neg_condition = torch.zeros(batch_size, 10)

    # Create mock data
    data = {
        "idx": list(range(batch_size)),
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_condition.to(model.device, model.precision),
    }
    return model, data


@pytest.fixture
def get_multistep_model_data():
    """Fixture for multi-step distillation testing with two default student steps."""
    return _multistep_model_data()


def test_step_generations_train_the_gan_on_transitions_from_the_next_grid_time(monkeypatch):
    model, data = _multistep_model_data(
        student_generation="step", student_sample_type="ode", sample_t_in_student_bucket=True
    )
    grid = model.net.noise_scheduler.student_grid(device=model.device)
    renoisings = record_generation_renoising(model, monkeypatch)

    loss_maps = []
    for iteration in (0, 1):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)
        assert torch.isfinite(loss_map["total_loss"])
        loss_maps.append(loss_map)

    assert "gan_loss_gen" in loss_maps[0] and "gan_loss_disc" in loss_maps[1]
    assert len(renoisings) == 2
    for t, t_gen_data, student_step in renoisings:
        torch.testing.assert_close(t_gen_data, grid[student_step + 1])
        assert (t >= t_gen_data).all() and (t <= grid[student_step]).all()


def test_multistep_student_update(get_multistep_model_data):
    """Test multi-step student distillation with two default student steps."""
    model, data = get_multistep_model_data

    # Verify multi-step configuration
    assert model.net.noise_scheduler.default_student_steps == 2, "Test should use multi-step configuration"

    # Run the training step (student update)
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions for loss_map
    assert "total_loss" in loss_map
    assert "gan_loss_gen" in loss_map

    # Assertions for outputs - should be Callable in multi-step mode
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable), "Multi-step mode should return Callable"
    assert isinstance(outputs["input_rand"], torch.Tensor)
    assert isinstance(outputs["gen_rand_train"], torch.Tensor)

    # Execute the callable and verify shape
    gen_tensor = outputs["gen_rand"]()
    assert isinstance(gen_tensor, torch.Tensor)
    assert gen_tensor.shape == data["real"].shape
    assert outputs["input_rand"].shape == data["real"].shape


def test_multistep_discriminator_update(get_multistep_model_data):
    """Test multi-step discriminator update with two default student steps."""
    model, data = get_multistep_model_data

    # Verify multi-step configuration
    assert model.net.noise_scheduler.default_student_steps == 2, "Test should use multi-step configuration"

    # Run the training step (discriminator update)
    loss_map, outputs = model.single_train_step(data, 1)

    # Assertions for loss_map
    assert "gan_loss_disc" in loss_map

    # Assertions for outputs - should be Callable in multi-step mode
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable), "Multi-step mode should return Callable"
    assert isinstance(outputs["input_rand"], torch.Tensor)
    assert isinstance(outputs["gen_rand_train"], torch.Tensor)

    # Verify callable can be executed (don't need to check shape for discriminator test)
    gen_tensor = outputs["gen_rand"]()
    assert isinstance(gen_tensor, torch.Tensor)


def test_backward_simulation_resumes_the_carried_ladd_state(get_multistep_model_data):
    model, data = get_multistep_model_data
    model.config.backward_simulation = "staggered"
    state = torch.randn_like(data["real"])
    carried_condition = torch.zeros_like(data["condition"])
    carried_neg_condition = torch.ones_like(data["neg_condition"])
    carried_real = torch.randn_like(data["real"])

    batch = {
        **data,
        "state_carried": state,
        "step_carried": 1,
        "condition": carried_condition,
        "neg_condition": carried_neg_condition,
        "real": carried_real,
    }
    inputs = model._training_inputs(batch, iteration=2)
    input_student = inputs.input_student
    student_step = inputs.student_step
    t = inputs.t
    eps = inputs.eps
    step = inputs.trajectory_step
    generation_kwargs = inputs.generation_kwargs
    loss_scale = inputs.loss_scale

    assert input_student is state and step == 1
    assert t.shape == student_step.shape == (state.shape[0],)
    assert eps.shape == carried_real.shape
    assert generation_kwargs == {}
    assert (student_step == 1).all()
    assert loss_scale is None


def test_backward_simulation_carries_ladd_state_and_batch_context(get_multistep_model_data):
    model, data = get_multistep_model_data
    model.config.backward_simulation = "staggered"
    model.config.student_sample_type = "ode"
    model.config.student_update_freq = 1
    callback = CarryCallback()

    _, outputs = model.single_train_step(data, iteration=1)
    carry = outputs[CARRY_OUTPUT_KEY]
    assert carry["step_carried"] == 1
    assert carry["condition"] is data["condition"]
    assert carry["neg_condition"] is data["neg_condition"]
    assert carry["real"] is data["real"]
    assert not carry["state_carried"].requires_grad

    callback.on_training_accum_step_end(model, data, outputs, iteration=1, accum_iter=0)
    fresh_next_batch = {
        "real": torch.randn_like(data["real"]),
        "condition": torch.ones_like(data["condition"]),
        "neg_condition": torch.ones_like(data["neg_condition"]),
    }
    next_batch = dict(fresh_next_batch)
    callback.on_training_accum_step_begin(model, next_batch, iteration=2, accum_iter=0)
    assert next_batch["real"] is data["real"]
    assert next_batch["condition"] is data["condition"]
    assert next_batch["neg_condition"] is data["neg_condition"]

    _, outputs = model.single_train_step(next_batch, iteration=2)
    assert outputs[CARRY_OUTPUT_KEY] == {"step_carried": 2}
