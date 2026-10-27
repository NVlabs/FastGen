# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
from types import SimpleNamespace

import attrs
import torch
import pytest
import fastgen.methods.distribution_matching.f_distill as f_distill_module
from fastgen.methods import DMD2Model, FdistillModel
from fastgen.configs.methods.config_f_distill import ModelConfig
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.utils.tensor import high_precision_dtype
from tests.utils import check_grad_zero, record_generation_renoising


def test_fdistill_reuses_dmd2_student_update():
    assert FdistillModel._student_update_step is DMD2Model._student_update_step
    assert FdistillModel._compute_distribution_matching_loss is not DMD2Model._compute_distribution_matching_loss


def _model_config(student_steps=None, **overrides):
    """Build a tiny f-distill config, optionally with a multi-step student grid."""
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
    if student_steps is not None:
        instance.student_grid_cfg.steps = student_steps
    for key, value in overrides.items():
        setattr(instance, key, value)
    return instance


def _model_data(student_steps=None, **overrides):
    """Build a tiny f-distill model, optionally with a multi-step student grid, and a matching batch."""
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    model = FdistillModel(_model_config(student_steps, **overrides))
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
def get_model_data():
    return _model_data()


def test_single_train_step_student_update(get_model_data, monkeypatch):
    model, data = get_model_data
    model.config.vsd_loss_weight = 0.25
    expected_generation_kwarg = object()
    generation_metric_marker = torch.ones((), device=model.device)
    training_inputs = model._training_inputs
    gen_data_from_net = model.gen_data_from_net

    def training_inputs_with_generation_kwargs(*args, **kwargs):
        inputs = training_inputs(*args, **kwargs)
        return attrs.evolve(inputs, generation_kwargs={"generation_kwarg_marker": expected_generation_kwarg})

    def generate_with_metric(*args, generation_kwarg_marker=None, **kwargs):
        assert generation_kwarg_marker is expected_generation_kwarg
        generation = gen_data_from_net(*args, **kwargs)
        return attrs.evolve(
            generation, metrics={**generation.metrics, "generation_metric_marker": generation_metric_marker}
        )

    monkeypatch.setattr(model, "_training_inputs", training_inputs_with_generation_kwargs)
    monkeypatch.setattr(model, "gen_data_from_net", generate_with_metric)
    # Run the training step
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "f_distill_loss" in loss_map
    assert "gan_loss_gen" in loss_map
    assert "avg_h" in loss_map
    assert "min_h" in loss_map
    assert "max_h" in loss_map
    assert "avg_t_student" in loss_map
    assert "gen_rand" in outputs
    assert callable(outputs["gen_rand"])
    assert isinstance(outputs["gen_rand_train"], torch.Tensor)
    assert outputs["gen_rand"].keywords["neg_condition"] is data["neg_condition"]
    assert loss_map["generation_metric_marker"] is generation_metric_marker
    assert "generation_metric_marker" not in outputs
    torch.testing.assert_close(
        loss_map["total_loss"],
        model.config.vsd_loss_weight * loss_map["f_distill_loss"]
        + model.config.gan_loss_weight_gen * loss_map["gan_loss_gen"],
    )


def test_student_update_requires_real_data_for_the_density_ratio(get_model_data):
    """The discriminator trains only on real data, so the density ratio that weights the VSD needs it too."""
    model, data = get_model_data
    data = {key: value for key, value in data.items() if key != "real"}

    with pytest.raises(ValueError, match=r"trains only on data\['real'\]"):
        model.single_train_step(data, iteration=0)


@pytest.mark.parametrize("input_dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_f_divergence_weighting_computes_in_high_precision(input_dtype):
    computation_dtype = high_precision_dtype(input_dtype)
    model = object.__new__(FdistillModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        f_distill=SimpleNamespace(
            ratio_lower=0.1,
            ratio_upper=20.0,
            ratio_normalization=False,
        )
    )
    seen_dtypes = []

    def record_weighting_dtype(ratio):
        seen_dtypes.append(ratio.dtype)
        return ratio

    model.f_div_weighting_function = record_weighting_dtype
    fake_logits = torch.tensor([[0.125, -0.875], [3.125, -6.75]], dtype=input_dtype)
    t = torch.tensor([0.25, 0.75])

    h = model._get_f_div_weighting_h(fake_logits, t)

    assert seen_dtypes == [computation_dtype]
    assert h.dtype == computation_dtype


def test_distribution_matching_combines_f_divergence_and_external_scales(monkeypatch):
    model = object.__new__(FdistillModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(f_distill=SimpleNamespace(f_div="js"))
    h = torch.tensor([0.5, 1.5], dtype=torch.float32)
    external_scale = torch.tensor([0.25, 0.75], dtype=torch.float64)
    seen_loss_scales = []
    model._get_f_div_weighting_h = lambda fake_logits, t: h

    def record_vsd_loss(gen_data, _teacher_x0, _fake_score_x0, loss_scale=None, gen_x0=None):
        seen_loss_scales.append(loss_scale)
        return gen_data.sum() * 0 + loss_scale.mean()

    monkeypatch.setattr(f_distill_module, "variational_score_distillation_loss", record_vsd_loss)
    gen_data = torch.zeros(2, 1)
    loss_map, loss = model._compute_distribution_matching_loss(
        gen_data,
        torch.zeros_like(gen_data),
        torch.zeros_like(gen_data),
        torch.zeros(2, 1),
        torch.zeros(2),
        loss_scale=external_scale,
    )

    expected_scale = h.to(torch.float64) * external_scale
    assert len(seen_loss_scales) == 1
    torch.testing.assert_close(seen_loss_scales[0], expected_scale)
    torch.testing.assert_close(loss_map["f_distill_loss"], expected_scale.mean())
    torch.testing.assert_close(loss, expected_scale.mean())
    torch.testing.assert_close(loss_map["avg_h"], h.mean())


def test_distribution_matching_normalizes_by_the_clean_estimate(monkeypatch):
    model = object.__new__(FdistillModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(f_distill=SimpleNamespace(f_div="js"))
    model._get_f_div_weighting_h = lambda fake_logits, t: torch.ones(2)
    seen_gen_x0 = []

    def record_vsd_loss(gen_data, _teacher_x0, _fake_score_x0, loss_scale=None, gen_x0=None):
        seen_gen_x0.append(gen_x0)
        return gen_data.sum() * 0

    monkeypatch.setattr(f_distill_module, "variational_score_distillation_loss", record_vsd_loss)
    gen_data, gen_x0 = torch.zeros(2, 1), torch.ones(2, 1)
    model._compute_distribution_matching_loss(
        gen_data,
        torch.zeros_like(gen_data),
        torch.zeros_like(gen_data),
        torch.zeros(2, 1),
        torch.zeros(2),
        gen_x0=gen_x0,
    )

    assert len(seen_gen_x0) == 1
    torch.testing.assert_close(seen_gen_x0[0], gen_x0)


def test_step_generations_weight_the_vsd_of_transitions_by_the_density_ratio(monkeypatch):
    model, data = _model_data(
        student_steps=2,
        student_generation="step",
        student_sample_type="ode",
        sample_t_in_student_bucket=True,
        fake_score_pred_type="eps",
        fake_score_weighting="variance_matched",
    )
    grid = model.net.noise_scheduler.student_grid(device=model.device)
    renoisings = record_generation_renoising(model, monkeypatch)
    vsd_loss = f_distill_module.variational_score_distillation_loss
    seen_gen_x0 = []

    def record_vsd_loss(gen_data, teacher_x0, fake_score_x0, loss_scale=None, gen_x0=None):
        seen_gen_x0.append(gen_x0)
        return vsd_loss(gen_data, teacher_x0, fake_score_x0, loss_scale=loss_scale, gen_x0=gen_x0)

    monkeypatch.setattr(f_distill_module, "variational_score_distillation_loss", record_vsd_loss)

    loss_maps, outputs = [], []
    for iteration in (0, 1):
        model.optimizers_zero_grad(iteration)
        loss_map, output = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)
        assert torch.isfinite(loss_map["total_loss"])
        loss_maps.append(loss_map)
        outputs.append(output)

    assert "f_distill_loss" in loss_maps[0] and "fake_score_loss" in loss_maps[1]
    # The noisy generations' VSD is normalized by the x0 their prediction implies, unrounded; the training-sample
    # output is that x0 in the state dtype.
    assert len(seen_gen_x0) == 1
    output_x0 = outputs[0]["gen_rand_train"]
    torch.testing.assert_close(seen_gen_x0[0].to(output_x0.dtype), output_x0, rtol=0, atol=0)
    assert len(renoisings) == 2
    for t, t_gen_data, student_step in renoisings:
        torch.testing.assert_close(t_gen_data, grid[student_step + 1])
        assert (t >= t_gen_data).all() and (t <= grid[student_step]).all()


def test_single_train_step_fake_score_update(get_model_data):
    model, data = get_model_data
    # Run the training step
    loss_map, outputs = model.single_train_step(data, 1)

    # Assertions
    assert "fake_score_loss" in loss_map
    assert "gan_loss_disc" in loss_map
    assert "avg_t_student" in loss_map
    assert "gen_rand" in outputs
    assert callable(outputs["gen_rand"])
    assert isinstance(outputs["gen_rand_train"], torch.Tensor)
    assert outputs["gen_rand"].keywords["neg_condition"] is data["neg_condition"]


def test_optimizers(get_model_data):
    model, data = get_model_data
    # Test for both student and fake_score optimizer
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)

    # Test for both student and fake_score optimizer zero grad, after at least one backward pass
    model.optimizers_zero_grad(2)
    check_grad_zero(model.net)
    model.optimizers_zero_grad(3)
    check_grad_zero(model.discriminator)
    check_grad_zero(model.fake_score)


def test_ratio_normalization_bins_live_on_the_discriminator():
    model, data = _model_data(use_ema=True)
    assert "bins" in model.discriminator.state_dict()
    assert "bins" not in model.net.state_dict()
    assert model.ema.state_dict().keys() == model.net.state_dict().keys()
    initial_bins = model.discriminator.bins.clone()

    loss_map, _ = model.single_train_step(data, 0)

    assert "f_distill_loss" in loss_map
    assert not torch.equal(model.discriminator.bins, initial_bins)


def test_validation_preserves_density_ratio_history():
    model, data = _model_data()
    model.single_train_step(data, 0)
    bins = model.discriminator.bins.clone()
    with torch.no_grad():
        model.single_train_step(data, 0)
    torch.testing.assert_close(model.discriminator.bins, bins, rtol=0, atol=0)


def test_inference_builds_without_a_discriminator_or_ratio_bins():
    model = FdistillModel(_model_config(inference=True))
    assert not hasattr(model, "discriminator")
    assert "bins" not in model.net.state_dict()
