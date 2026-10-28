# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import attrs
import torch
import pytest
import fastgen.methods.backward_simulation as backward_simulation
import fastgen.methods.distribution_matching.dmd2 as dmd2_module
from fastgen.methods import CausVidModel, DMD2Model, SelfForcingModel, SFTModel
from fastgen.methods.distribution_matching.dmd2 import Generation, TrainingInputs
from fastgen.methods.common_loss import (
    flow_matching_loss,
    gan_loss_discriminator,
    gan_loss_generator,
    mse_loss,
    transition_score_loss,
    transition_score_weight,
    variational_score_distillation_loss,
)
from fastgen.callbacks.carry import CARRY_OUTPUT_KEY, CARRY_OVERLAY_KEYS, CarryCallback
from fastgen.callbacks.wandb import extract_prompt
from fastgen.networks.discriminators import CompositeDiscriminator, Discriminator_VideoDiT, Discriminator_Audio2D
from fastgen.configs.config import SampleTConfig
from fastgen.configs.methods.config_dmd2 import ModelConfig
from fastgen.configs.methods.config_sft import ModelConfig as SFTModelConfig
from fastgen.configs.methods.config_self_forcing import ModelConfig as SelfForcingModelConfig
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.networks.lora_utils import LoRAMixin, SharedBaseTeacher
from fastgen.networks.network import FastGenNetwork
from fastgen.configs.instantiate import LazyCall as L
from fastgen.networks.noise_schedule import AlphasNoiseSchedule, ShiftedRFNoiseSchedule, StudentGridConfig
import fastgen.utils.tensor as tensor_module
from fastgen.utils.tensor import high_precision_dtype
from fastgen.utils.tree_utils import cast_like
from tests.model_utils import MultimodalNet
from tests.utils import check_grad_zero, record_generation_renoising


class _TinyCheckpointLoRANet(LoRAMixin, FastGenNetwork):
    """Explicit-checkpoint net for the shared-teacher fake-score regression."""

    lora_base_name = ""

    def __init__(self, student_grid_cfg=None):
        super().__init__(net_pred_type="x0", schedule_type="rf", student_grid_cfg=student_grid_cfg)
        self.proj = torch.nn.Linear(3, 3)

    def forward(self, x_t, t, condition=None, fwd_pred_type=None, **kwargs):
        return self.proj(x_t.movedim(1, -1)).movedim(-1, 1)


def test_standard_dmd_native_shift_resolves_to_expected_grid():
    config = ModelConfig()
    config.student_grid_cfg.steps = 4
    config.student_grid_cfg.grid_kwargs = {"shift": 5}

    net = _TinyCheckpointLoRANet(student_grid_cfg=config.student_grid_cfg)

    assert net.noise_scheduler.student_grid_cfg is config.student_grid_cfg
    assert config.student_grid_cfg.grid_type == "native"
    torch.testing.assert_close(
        net.noise_scheduler.student_grid(),
        torch.tensor([0.999, 0.9365625, 0.8325, 0.624375, 0.0], dtype=net.noise_scheduler.t_precision),
    )


@pytest.mark.parametrize(
    ("model_cls", "config_cls"),
    (
        (CausVidModel, ModelConfig),
        (SelfForcingModel, SelfForcingModelConfig),
    ),
)
def test_dmd2_descendants_support_gan_only_base_routing(model_cls, config_cls):
    assert issubclass(model_cls, DMD2Model)
    assert model_cls._auxiliary_update_step is DMD2Model._auxiliary_update_step
    assert model_cls.single_train_step is DMD2Model.single_train_step

    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.config = config_cls()
    model.config.vsd_loss_weight = 0.0
    model.config.gan_loss_weight_gen = 1.0
    assert not model.use_vsd
    assert model.use_gan

    assert model_cls._student_update_step is DMD2Model._student_update_step


def assert_single_train_step_preserves_training_inputs_record(
    monkeypatch, model_cls, iteration, update_hook, expected_loss
):
    from types import SimpleNamespace

    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(student_update_freq=2, loss_config=SimpleNamespace(pdd_loss_weight=3.0))
    model._setup_grad_requirements = lambda iteration: None
    generation_kwargs = {"marker": object()}
    input_student = {"video": torch.zeros(2, 1), "audio": torch.ones(2, 1)}
    student_step = torch.tensor([1, 1])
    t = torch.tensor([0.1, 0.2])
    eps = {key: torch.ones_like(value) for key, value in input_student.items()}
    loss_scale = torch.tensor([0.25, 0.75])
    inputs = TrainingInputs(
        input_student,
        student_step,
        t,
        eps,
        trajectory_step=1,
        generation_kwargs=generation_kwargs,
        loss_scale=loss_scale,
    )
    data = {"condition": object()}
    outputs = {"gen_rand_train": input_student}
    seen = []

    def training_inputs(update_data, update_iteration):
        assert update_data is data and update_iteration == iteration
        return inputs

    def record_update(hook):
        def update(self, update_inputs, update_data):
            seen.append(hook)
            assert self is model
            assert update_inputs is inputs and update_data is data
            assert update_inputs.input_student is input_student
            assert update_inputs.student_step is student_step
            assert update_inputs.t is t and update_inputs.eps is eps
            assert update_inputs.generation_kwargs is generation_kwargs
            assert update_inputs.loss_scale is loss_scale
            assert update_inputs.trajectory_step == 1
            return {"total_loss": torch.tensor(5.0), "pdd_loss": torch.tensor(2.0)}, outputs

        return update

    monkeypatch.setattr(model, "_training_inputs", training_inputs)
    for hook in ("_student_update_step", "_auxiliary_update_step"):
        monkeypatch.setattr(DMD2Model, hook, record_update(hook))

    with torch.enable_grad():
        loss_map, actual_outputs = model.single_train_step(data, iteration)

    assert seen == [update_hook]
    assert actual_outputs is outputs
    # The real subclass override must still add its PDD objective after forwarding the same record.
    torch.testing.assert_close(loss_map["total_loss"], torch.tensor(expected_loss))


@pytest.mark.parametrize(("iteration", "update_hook"), [(0, "_student_update_step"), (1, "_auxiliary_update_step")])
def test_single_train_step_preserves_training_inputs_record(monkeypatch, iteration, update_hook):
    assert_single_train_step_preserves_training_inputs_record(monkeypatch, DMD2Model, iteration, update_hook, 5.0)


def test_use_gan_tracks_config_dynamically():
    model = object.__new__(DMD2Model)
    torch.nn.Module.__init__(model)
    model.config = ModelConfig()

    for weight, expected in ((-1.0, False), (0.0, False), (1e-3, True)):
        model.config.gan_loss_weight_gen = weight
        assert model.use_gan is expected


def test_use_vsd_tracks_config_dynamically():
    model = object.__new__(DMD2Model)
    torch.nn.Module.__init__(model)
    model.config = ModelConfig()

    assert model.config.vsd_loss_weight == 1.0
    for weight, expected in ((-1.0, False), (0.0, False), (0.25, True)):
        model.config.vsd_loss_weight = weight
        assert model.use_vsd is expected


def test_auxiliary_dicts_follow_constructed_models(get_model_data):
    model, _ = get_model_data
    model.config.gan_loss_weight_gen = 0.0

    assert model.model_dict["fake_score"] is model.fake_score
    assert model.model_dict["discriminator"] is model.discriminator
    assert model.optimizer_dict["fake_score"] is model.fake_score_optimizer
    assert model.optimizer_dict["discriminator"] is model.discriminator_optimizer
    assert model.scheduler_dict["fake_score"] is model.fake_score_lr_scheduler
    assert model.scheduler_dict["discriminator"] is model.discriminator_lr_scheduler


@pytest.mark.parametrize("use_gan", [True, False])
def test_grad_requirements_cover_student_fake_score_and_gan(get_model_data, use_gan):
    """The student trains on its own updates and generates in eval mode for the auxiliary ones, with or without GAN."""
    model, _ = get_model_data
    if not use_gan:
        model.config.gan_loss_weight_gen = 0.0
    assert model.use_gan is use_gan

    model._setup_grad_requirements(iteration=0)
    assert model.net.training
    assert all(parameter.requires_grad for parameter in model.net.parameters())
    assert not model.fake_score.training
    assert all(not parameter.requires_grad for parameter in model.fake_score.parameters())
    if use_gan:
        assert not model.discriminator.training
        assert all(not parameter.requires_grad for parameter in model.discriminator.parameters())

    model._setup_grad_requirements(iteration=1)
    assert not model.net.training
    assert all(not parameter.requires_grad for parameter in model.net.parameters())
    assert model.fake_score.training
    assert all(parameter.requires_grad for parameter in model.fake_score.parameters())
    if use_gan:
        assert model.discriminator.training
        assert all(parameter.requires_grad for parameter in model.discriminator.parameters())


def test_fake_score_clone_uses_strict_module_loader(tmp_path, monkeypatch):
    import fastgen.methods.model as model_module

    checkpoint_net = _TinyCheckpointLoRANet()
    checkpoint_path = tmp_path / "base.pt"
    torch.save(checkpoint_net.state_dict(), checkpoint_path)
    observed = []
    load_module_state_dict = model_module.load_module_state_dict

    def record_load(module, state_dict, **kwargs):
        observed.append((module, kwargs))
        return load_module_state_dict(module, state_dict, **kwargs)

    monkeypatch.setattr(model_module, "load_module_state_dict", record_load)
    instance = ModelConfig()
    instance.device = torch.device("cpu")
    instance.net = L(_TinyCheckpointLoRANet)()
    instance.pretrained_model_path = str(checkpoint_path)
    instance.use_ema = False
    instance.gan_loss_weight_gen = 0.0

    model = DMD2Model(instance)

    assert [kwargs for module, kwargs in observed if module is model.fake_score] == [
        {
            "strict": True,
            "require_any": True,
            "description": "Fake score initialized from teacher",
        }
    ]


def test_adapter_fake_score_loads_explicit_base_checkpoint(tmp_path):
    checkpoint_net = _TinyCheckpointLoRANet()
    checkpoint_path = tmp_path / "base.pt"
    torch.save(checkpoint_net.state_dict(), checkpoint_path)
    expected_weight = checkpoint_net.proj.weight.detach().clone()

    instance = ModelConfig()
    instance.device = torch.device("cpu")
    instance.net = L(_TinyCheckpointLoRANet)()
    instance.pretrained_model_path = str(checkpoint_path)
    instance.net_adapter.type = "lora"
    instance.net_adapter.base_name = ""
    instance.net_adapter.rank = 2
    instance.net_adapter.alpha = 2
    instance.net_adapter.target_modules = ["proj"]
    instance.use_ema = False
    instance.gan_loss_weight_gen = 0.0

    model = DMD2Model(instance)

    assert isinstance(model.teacher, SharedBaseTeacher)
    assert not model.teacher.state_dict()
    torch.testing.assert_close(model.net.proj.base_layer.weight, expected_weight)
    torch.testing.assert_close(model.fake_score.proj.weight, expected_weight)


def test_adapter_loads_distinct_student_checkpoint_before_injection(tmp_path):
    checkpoint_net = _TinyCheckpointLoRANet()
    checkpoint_path = tmp_path / "student.pt"
    torch.save(checkpoint_net.state_dict(), checkpoint_path)
    expected_weight = checkpoint_net.proj.weight.detach().clone()

    instance = ModelConfig()
    instance.device = torch.device("cpu")
    instance.net = L(_TinyCheckpointLoRANet)()
    instance.teacher = L(_TinyCheckpointLoRANet)()
    instance.pretrained_model_path = None
    instance.pretrained_student_net_path = str(checkpoint_path)
    instance.net_adapter.type = "lora"
    instance.net_adapter.base_name = ""
    instance.net_adapter.rank = 2
    instance.net_adapter.alpha = 2
    instance.net_adapter.target_modules = ["proj"]
    instance.use_ema = False
    instance.vsd_loss_weight = 0.0
    instance.gan_loss_weight_gen = 0.0

    model = DMD2Model(instance)

    assert not isinstance(model.teacher, SharedBaseTeacher)
    torch.testing.assert_close(model.net.proj.base_layer.weight, expected_weight)
    assert not torch.equal(model.teacher.proj.weight, expected_weight)


def test_adapter_rejects_distinct_student_checkpoint_with_shared_teacher(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "student.pt"
    torch.save(_TinyCheckpointLoRANet().state_dict(), checkpoint_path)

    loaded_paths = []
    load_pretrained_model = dmd2_module.FastGenModel._load_pretrained_model

    def record_load(model, pretrained_model_path, *args, **kwargs):
        loaded_paths.append(pretrained_model_path)
        return load_pretrained_model(model, pretrained_model_path, *args, **kwargs)

    monkeypatch.setattr(dmd2_module.FastGenModel, "_load_pretrained_model", record_load)

    instance = ModelConfig()
    instance.device = torch.device("cpu")
    instance.net = L(_TinyCheckpointLoRANet)()
    instance.teacher = None
    # Load a real checkpoint first so build_teacher's shared-base guard is authoritative.
    instance.pretrained_student_net_path = str(checkpoint_path)
    instance.net_adapter.type = "lora"
    instance.net_adapter.base_name = ""
    instance.net_adapter.rank = 2
    instance.net_adapter.target_modules = ["proj"]
    instance.use_ema = False

    with pytest.raises(ValueError, match="requires an explicit separate teacher"):
        DMD2Model(instance)

    assert loaded_paths == [str(checkpoint_path)]


def test_adapter_ignores_distinct_student_checkpoint_when_loading_is_disabled():
    instance = ModelConfig()
    instance.device = torch.device("cpu")
    instance.net = L(_TinyCheckpointLoRANet)()
    instance.teacher = None
    instance.pretrained_student_net_path = "/unused/student.pt"
    instance.load_student_weights = False
    instance.net_adapter.type = "lora"
    instance.net_adapter.base_name = ""
    instance.net_adapter.rank = 2
    instance.net_adapter.target_modules = ["proj"]
    instance.use_ema = False
    instance.vsd_loss_weight = 0.0
    instance.gan_loss_weight_gen = 0.0

    model = DMD2Model(instance)

    assert isinstance(model.teacher, SharedBaseTeacher)


def test_adapter_loads_student_checkpoint_for_a_model_without_a_teacher(tmp_path):
    checkpoint_net = _TinyCheckpointLoRANet()
    checkpoint_path = tmp_path / "student.pt"
    torch.save(checkpoint_net.state_dict(), checkpoint_path)
    expected_weight = checkpoint_net.proj.weight.detach().clone()

    instance = SFTModelConfig()
    instance.device = torch.device("cpu")
    instance.net = L(_TinyCheckpointLoRANet)()
    instance.pretrained_student_net_path = str(checkpoint_path)
    instance.net_adapter.type = "lora"
    instance.net_adapter.base_name = ""
    instance.net_adapter.rank = 2
    instance.net_adapter.alpha = 2
    instance.net_adapter.target_modules = ["proj"]
    instance.use_ema = False

    model = SFTModel(instance)

    assert not hasattr(model, "teacher")
    torch.testing.assert_close(model.net.proj.base_layer.weight, expected_weight)


def test_dmd2_generation_uses_student_timestep():
    class GridScheduler:
        def __init__(self):
            self.conversion = None

        @staticmethod
        def student_grid(device=None):
            return torch.tensor([0.9, 0.4, 0.0], device=device)

        def convert_model_output(self, xt, model_output, t, src_pred_type, target_pred_type, output_dtype="same"):
            self.conversion = (xt, model_output, t, src_pred_type, target_pred_type, output_dtype)
            return cast_like(2 * model_output.to(high_precision_dtype(xt, model_output)), xt, output_dtype)

    class RecordingNet(torch.nn.Module):
        net_pred_type = "flow"

        def __init__(self):
            super().__init__()
            self.call = None
            self.noise_scheduler = GridScheduler()

        def forward(self, x, t, condition=None, **kwargs):
            self.call = (x, t, condition, kwargs)
            return x + 1

    model = object.__new__(DMD2Model)
    torch.nn.Module.__init__(model)
    model.config = type("Config", (), {"student_generation": "x0", "backward_simulation": None})()
    model.net = RecordingNet()
    x = torch.zeros(2, 3)
    student_step = torch.tensor([0, 1])
    t_student = torch.tensor([0.9, 0.4])
    condition = object()

    generation = model.gen_data_from_net(x, student_step, condition=condition)

    torch.testing.assert_close(generation.data, 2 * (x + 1))
    # The clean estimate stays in high precision; the generation is rounded to the state dtype.
    torch.testing.assert_close(generation.x0, 2 * (x + 1).to(high_precision_dtype(x)))
    assert set(generation.metrics) == {"avg_t_student"}
    torch.testing.assert_close(generation.metrics["avg_t_student"], t_student.mean())
    assert generation.next_state is None
    seen_x, seen_t, seen_condition, kwargs = model.net.call
    assert seen_x is x
    torch.testing.assert_close(seen_t, t_student)
    assert seen_condition is condition
    assert kwargs == {"output_dtype": None}
    converted_x, prediction, converted_t, src_pred_type, target_pred_type, output_dtype = (
        model.net.noise_scheduler.conversion
    )
    assert converted_x is x
    torch.testing.assert_close(prediction, x + 1)
    torch.testing.assert_close(converted_t, t_student)
    assert (src_pred_type, target_pred_type, output_dtype) == ("flow", "x0", None)

    with pytest.raises(ValueError, match="requires a student step"):
        model.gen_data_from_net(x, None)


def _auxiliary_noising_harness(fake_score_sample_t_overrides):
    model = object.__new__(DMD2Model)
    torch.nn.Module.__init__(model)
    calls = {"forward_process": [], "events": []}
    gen_data = torch.full((2, 3), 5.0)
    model.precision = gen_data.dtype
    model.device = gen_data.device

    class RecordingScheduler:
        schedules = object()
        t_precision = torch.float32

        def forward_process(self, x, eps, t, t_start=None):
            assert t_start is None, "DMD2 generations are clean, so they are noised from time 0"
            t_broadcast = t.reshape(t.shape[0], *([1] * (x.ndim - 1)))
            output = x + eps + t_broadcast
            calls["forward_process"].append((x, eps, t, output))
            calls["events"].append("forward_process")
            return output

    model.config = type(
        "Config",
        (),
        {
            "fake_score_pred_type": "x0",
            "fake_score_weighting": "native",
            "sample_t_cfg": SampleTConfig(time_dist_type="shifted", min_t=0.001, max_t=0.999),
            "fake_score_sample_t_overrides": fake_score_sample_t_overrides,
            "student_generation": "x0",
            "vsd_loss_weight": 1.0,
            "gan_loss_weight_gen": 1.0,
            "gan_r1_reg_weight": 0.0,
        },
    )()
    model.net = type("Net", (), {"noise_scheduler": RecordingScheduler()})()
    model.teacher = type("Teacher", (), {"net_pred_type": "x0"})()

    def gen_data_from_net(input_student, student_step, **kwargs):
        calls["generation"] = (input_student, student_step, kwargs)
        return Generation(data=gen_data, x0=gen_data)

    def fake_score(x_t, t, **kwargs):
        calls["fake_score"] = (x_t, t, kwargs)
        calls["events"].append("fake_score")
        return torch.zeros_like(x_t)

    def compute_gan_discriminator_losses(fake_data, t, eps, real_data, **kwargs):
        calls["gan"] = (fake_data, t, eps, real_data, kwargs)
        calls["events"].append("gan")
        loss = fake_data.sum() * 0
        return {"gan_loss_disc": loss}, loss

    model.gen_data_from_net = gen_data_from_net
    model.fake_score = fake_score
    model._compute_gan_discriminator_losses = compute_gan_discriminator_losses
    model._get_outputs = lambda gen_data, **kwargs: {"gen_rand_train": gen_data}
    return model, gen_data, calls


def test_configured_fake_score_sampler_only_changes_dsm_noising(monkeypatch):
    model, gen_data, calls = _auxiliary_noising_harness({"time_dist_type": "logitnormal", "train_p_std": 1.0})
    input_student = torch.zeros_like(gen_data)
    main_t = torch.tensor([0.1, 0.2])
    main_eps = torch.full_like(gen_data, 1.0)
    fake_score_t = torch.tensor([0.7, 0.8])
    fake_score_eps = torch.full_like(gen_data, 2.0)
    real_data = torch.full_like(gen_data, 3.0)
    condition = object()
    loss_scale = torch.tensor([0.25, 0.75])
    student_step = torch.tensor([1, 1])
    sample_calls = []

    def sample_t_and_eps(batch_size, sample_t_cfg=None, student_step=None):
        sample_calls.append((batch_size, sample_t_cfg, student_step))
        return fake_score_t, fake_score_eps

    def record_dsm_loss(prediction, pred_type, x0, eps, schedules, t, loss_scale=None):
        calls["dsm"] = (prediction, pred_type, x0, eps, schedules, t, loss_scale)
        return prediction.sum() * 0

    monkeypatch.setattr(model, "_sample_t_and_eps", sample_t_and_eps)
    monkeypatch.setattr(dmd2_module, "flow_matching_loss", record_dsm_loss)

    loss_map, outputs = model._auxiliary_update_step(
        TrainingInputs(input_student, student_step, main_t, main_eps, loss_scale=loss_scale),
        {"real": real_data, "condition": condition},
    )

    assert len(sample_calls) == 1
    # The overrides change only their own fields; the rest, bounds included, come from sample_t_cfg.
    expected_cfg = SampleTConfig(time_dist_type="logitnormal", train_p_std=1.0, min_t=0.001, max_t=0.999)
    assert sample_calls[0][:2] == (2, expected_cfg)
    assert sample_calls[0][2] is student_step
    assert len(calls["forward_process"]) == 2
    _, gan_eps, gan_t, gan_input = calls["forward_process"][0]
    _, dsm_eps, dsm_t, dsm_input = calls["forward_process"][1]
    assert calls["events"] == ["forward_process", "gan", "forward_process", "fake_score"]
    assert dsm_eps is fake_score_eps
    assert dsm_t is fake_score_t
    assert gan_eps is main_eps
    assert gan_t is main_t
    torch.testing.assert_close(dsm_input, gen_data + fake_score_eps + fake_score_t[:, None])
    torch.testing.assert_close(gan_input, gen_data + main_eps + main_t[:, None])

    fake_score_input, seen_fake_score_t, fake_score_kwargs = calls["fake_score"]
    assert fake_score_input is dsm_input
    assert seen_fake_score_t is fake_score_t
    assert fake_score_kwargs == {"condition": condition, "fwd_pred_type": "x0"}
    _, _, seen_x0, seen_dsm_eps, _, seen_dsm_t, seen_dsm_scale = calls["dsm"]
    assert seen_x0 is gen_data
    assert seen_dsm_eps is fake_score_eps
    assert seen_dsm_t is fake_score_t
    assert seen_dsm_scale is loss_scale

    gan_fake, seen_gan_t, seen_gan_eps, seen_real, gan_kwargs = calls["gan"]
    assert gan_fake is gan_input
    assert seen_gan_t is main_t
    assert seen_gan_eps is main_eps
    assert seen_real is real_data
    assert set(gan_kwargs) == {"condition", "loss_scale", "student_step"}
    assert gan_kwargs["condition"] is condition
    assert gan_kwargs["loss_scale"] is loss_scale
    assert gan_kwargs["student_step"] is student_step
    assert set(loss_map) == {"fake_score_loss", "gan_loss_disc", "total_loss"}
    assert set(outputs) == {"gen_rand_train"}
    assert outputs["gen_rand_train"] is gen_data


def test_no_fake_score_sampler_reuses_main_noising_for_dsm_and_gan(monkeypatch):
    model, gen_data, calls = _auxiliary_noising_harness(None)
    input_student = torch.zeros_like(gen_data)
    main_t = torch.tensor([0.1, 0.2])
    main_eps = torch.full_like(gen_data, 1.0)
    real_data = torch.full_like(gen_data, 3.0)

    def unexpected_sample(*args, **kwargs):
        raise AssertionError("the main DMD perturbation should be reused")

    def record_dsm_loss(prediction, pred_type, x0, eps, schedules, t, loss_scale=None):
        calls["dsm"] = (eps, t)
        return prediction.sum() * 0

    monkeypatch.setattr(model, "_sample_t_and_eps", unexpected_sample)
    monkeypatch.setattr(dmd2_module, "flow_matching_loss", record_dsm_loss)

    model._auxiliary_update_step(
        TrainingInputs(input_student, None, main_t, main_eps),
        {"real": real_data},
    )

    assert len(calls["forward_process"]) == 1
    _, forwarded_eps, forwarded_t, shared_input = calls["forward_process"][0]
    assert calls["events"] == ["forward_process", "gan", "fake_score"]
    assert forwarded_eps is main_eps
    assert forwarded_t is main_t
    fake_score_input, seen_fake_score_t, _ = calls["fake_score"]
    assert fake_score_input is shared_input
    assert seen_fake_score_t is main_t
    seen_dsm_eps, seen_dsm_t = calls["dsm"]
    assert seen_dsm_eps is main_eps
    assert seen_dsm_t is main_t
    gan_fake, seen_gan_t, seen_gan_eps, seen_real, _ = calls["gan"]
    assert gan_fake is shared_input
    assert seen_gan_t is main_t
    assert seen_gan_eps is main_eps
    assert seen_real is real_data


def _assert_scaled_gradient_matches_active_reference(
    scaled_loss,
    scaled_input,
    reference_loss,
    reference_input,
):
    torch.testing.assert_close(scaled_loss, reference_loss)
    scaled_grad = torch.autograd.grad(scaled_loss, scaled_input)[0]
    reference_grad = torch.autograd.grad(reference_loss, reference_input)[0]
    torch.testing.assert_close(scaled_grad[0], torch.zeros_like(scaled_grad[0]))
    torch.testing.assert_close(scaled_grad[1], reference_grad[0])


def test_sample_scaled_dsm_matches_the_active_sample_value_and_gradient():
    prediction = torch.tensor([[0.0, 1.0], [2.0, 4.0]], requires_grad=True)
    target = torch.tensor([[1.0, 3.0], [5.0, 8.0]])
    loss_scale = torch.tensor([0.0, 2.0])
    scaled_loss = flow_matching_loss(
        prediction,
        "x0",
        x0=target,
        loss_scale=loss_scale,
    )

    reference_prediction = prediction.detach()[1:].clone().requires_grad_()
    reference_loss = flow_matching_loss(
        reference_prediction,
        "x0",
        x0=target[1:],
    )

    _assert_scaled_gradient_matches_active_reference(
        scaled_loss,
        prediction,
        reference_loss,
        reference_prediction,
    )


def test_mse_promotes_bfloat16_inputs_to_high_precision_with_and_without_scale():
    prediction = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.bfloat16, requires_grad=True)
    target = torch.tensor([[0.0, 0.5], [1.0, 2.0]], dtype=torch.bfloat16)
    loss_scale = torch.tensor([0.0, 2.0])

    scaled_loss = flow_matching_loss(
        prediction,
        "x0",
        x0=target,
        loss_scale=loss_scale,
    )
    unscaled_loss = flow_matching_loss(prediction, "x0", target)
    loss_dtype = high_precision_dtype(torch.bfloat16)
    squared_error = (prediction.to(loss_dtype) - target.to(loss_dtype)).square()

    assert scaled_loss.dtype == loss_dtype
    assert unscaled_loss.dtype == loss_dtype
    torch.testing.assert_close(scaled_loss, (squared_error.mean(dim=1) * loss_scale).mean())
    torch.testing.assert_close(unscaled_loss, squared_error.mean())


def test_dsm_requires_prediction_and_target_metadata_to_match():
    prediction = torch.zeros(2, 3)

    with pytest.raises(AssertionError, match=r"x0\.shape"):
        flow_matching_loss(prediction, "x0", x0=torch.zeros(3, 2))
    with pytest.raises(AssertionError, match=r"x0\.dtype"):
        flow_matching_loss(prediction, "x0", x0=torch.zeros(2, 3, dtype=torch.float64))
    with pytest.raises(AssertionError, match=r"x0\.device"):
        flow_matching_loss(prediction, "x0", x0=torch.empty(2, 3, device="meta"))
    # A time grid without the channel axis broadcasts a time-dependent flow target past the inputs' shape.
    data, schedule = torch.zeros(2, 1, 3), ShiftedRFNoiseSchedule(shift=3.0)
    with pytest.raises(AssertionError, match=r"target\.shape"):
        flow_matching_loss(data, "flow", data, data, schedule, torch.full((2, 3), 0.5))


def test_dsm_v_target_is_alpha_eps_minus_sigma_x0_and_needs_an_alphas_schedule():
    torch.manual_seed(0)
    schedule = AlphasNoiseSchedule(torch.linspace(0.999, 0.001, 1000, dtype=torch.float64))
    prediction, x0, eps = torch.randn(3, 2, 4).bfloat16().unbind()
    t = torch.tensor([0.25, 0.75], dtype=torch.float64)

    loss = flow_matching_loss(prediction, "v", x0, eps, schedule, t)

    dtype = high_precision_dtype(torch.bfloat16)
    alpha, sigma = (coefficient.to(dtype)[:, None] for coefficient in schedule.alpha_sigma(t))
    target = alpha * eps.to(dtype) - sigma * x0.to(dtype)
    torch.testing.assert_close(loss, (prediction.to(dtype) - target).square().mean())
    with pytest.raises(ValueError, match="does not support v-prediction"):
        flow_matching_loss(prediction, "v", x0, eps, ShiftedRFNoiseSchedule(shift=3.0), t)


def test_mse_rejects_shapes_that_would_broadcast():
    # (2,) against (2, 1) broadcasts to (2, 2), which still passes the per-sample loss-scale check.
    prediction, target = torch.zeros(2), torch.zeros(2, 1)
    for loss_scale in (None, torch.ones(2)):
        with pytest.raises(AssertionError, match=r"target\.shape"):
            mse_loss(prediction, target, loss_scale)


def test_transition_score_loss_rejects_shapes_that_would_broadcast():
    schedule = ShiftedRFNoiseSchedule(shift=3.0)
    # A prediction without the channel axis broadcasts against the noise to the noise's own shape.
    with pytest.raises(AssertionError, match=r"eps\.shape"):
        transition_score_loss(
            torch.zeros(2, 1, 3), torch.zeros(2, 4, 3), schedule, torch.zeros(2), torch.full((2,), 0.5), "x0"
        )
    # A time grid without the channel axis broadcasts the time-dependent factors past the inputs' shape.
    data = torch.zeros(2, 1, 3)
    with pytest.raises(AssertionError, match=r"elementwise_loss\.shape"):
        transition_score_loss(data, data, schedule, torch.zeros(2, 3), torch.full((2, 3), 0.5), "x0")


def test_sample_scaled_vsd_is_linear_and_matches_the_active_sample():
    gen_data = torch.tensor([[1.0, 2.0], [3.0, 5.0]], requires_grad=True)
    teacher_x0 = torch.tensor([[0.0, 0.5], [1.0, 2.0]])
    fake_score_x0 = torch.tensor([[0.5, 1.0], [2.0, 4.0]])
    loss_scale = torch.tensor([0.0, 2.0])
    scaled_loss = variational_score_distillation_loss(
        gen_data,
        teacher_x0,
        fake_score_x0,
        loss_scale=loss_scale,
    )

    reference_gen_data = gen_data.detach()[1:].clone().requires_grad_()
    reference_loss = variational_score_distillation_loss(
        reference_gen_data,
        teacher_x0[1:],
        fake_score_x0[1:],
    )

    _assert_scaled_gradient_matches_active_reference(
        scaled_loss,
        gen_data,
        reference_loss,
        reference_gen_data,
    )


@pytest.mark.parametrize("input_dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_vsd_promotes_all_operands_to_high_precision(input_dtype):
    loss_dtype = high_precision_dtype(input_dtype)
    gen_data = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=input_dtype, requires_grad=True)
    teacher_x0 = torch.tensor([[0.0, 0.5], [1.0, 2.0]], dtype=input_dtype)
    fake_score_x0 = torch.tensor([[0.5, 1.0], [2.0, 4.0]], dtype=input_dtype)
    loss_scale = torch.tensor([0.25, 0.75], dtype=input_dtype)

    loss = variational_score_distillation_loss(
        gen_data,
        teacher_x0,
        fake_score_x0,
        loss_scale=loss_scale,
    )

    promoted_gen_data = gen_data.to(loss_dtype)
    promoted_teacher_x0 = teacher_x0.to(loss_dtype)
    promoted_fake_score_x0 = fake_score_x0.to(loss_dtype)
    promoted_loss_scale = loss_scale.to(loss_dtype)
    with torch.no_grad():
        weight = 1 / ((promoted_gen_data - promoted_teacher_x0).abs().mean(dim=1, keepdim=True) + 1e-6)
        pseudo_target = promoted_gen_data - (promoted_fake_score_x0 - promoted_teacher_x0) * weight
    expected = 0.5 * ((promoted_gen_data - pseudo_target).square().mean(dim=1) * promoted_loss_scale).mean()

    assert loss.dtype == loss_dtype
    torch.testing.assert_close(loss, expected)


def test_vsd_requires_matching_operand_shapes_and_devices():
    gen_data = torch.zeros(2, 3)
    prediction = torch.zeros_like(gen_data)

    with pytest.raises(AssertionError, match=r"teacher_x0\.shape"):
        variational_score_distillation_loss(gen_data, torch.zeros(3, 2), prediction)
    with pytest.raises(AssertionError, match=r"teacher_x0\.device"):
        variational_score_distillation_loss(gen_data, torch.empty(2, 3, device="meta"), prediction)
    # Clean estimates arrive unrounded in another dtype; the loss runs in high precision.
    loss = variational_score_distillation_loss(gen_data, gen_data.double(), prediction)
    assert loss.dtype == high_precision_dtype(gen_data)


def test_vsd_normalization_matches_the_analytical_gradient():
    gen_data = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float64, requires_grad=True)
    teacher_x0 = torch.tensor([[0.0, 0.5], [1.0, 2.0]], dtype=torch.float64)
    fake_score_x0 = torch.tensor([[0.5, 1.0], [2.0, 4.0]], dtype=torch.float64)
    loss_scale = torch.tensor([0.25, 1.5], dtype=torch.float64)

    loss = variational_score_distillation_loss(
        gen_data,
        teacher_x0,
        fake_score_x0,
        loss_scale=loss_scale,
    )
    (gradient,) = torch.autograd.grad(loss, gen_data)

    normalization = 1 / ((gen_data.detach() - teacher_x0).abs().mean(dim=1, keepdim=True) + 1e-6)
    expected_gradient = normalization * loss_scale[:, None] * (fake_score_x0 - teacher_x0) / gen_data.numel()
    torch.testing.assert_close(gradient, expected_gradient)


def test_vsd_normalizes_by_the_teacher_residual_of_the_clean_estimate():
    gen_data = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float64, requires_grad=True)
    gen_x0 = torch.tensor([[0.5, 1.5], [2.0, 6.0]], dtype=torch.float64)
    teacher_x0 = torch.tensor([[0.0, 0.5], [1.0, 2.0]], dtype=torch.float64)
    fake_score_x0 = torch.tensor([[0.5, 1.0], [2.0, 4.0]], dtype=torch.float64)

    loss = variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0, gen_x0=gen_x0)
    (gradient,) = torch.autograd.grad(loss, gen_data)

    normalization = 1 / ((gen_x0 - teacher_x0).abs().mean(dim=1, keepdim=True) + 1e-6)
    expected_gradient = normalization * (fake_score_x0 - teacher_x0) / gen_data.numel()
    torch.testing.assert_close(gradient, expected_gradient)
    with pytest.raises(AssertionError, match=r"gen_x0\.shape"):
        variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0, gen_x0=torch.zeros(3, 2))


def test_vsd_normalizes_before_constructing_a_close_fp32_target():
    gen_data = torch.full((1, 32), 1.0, dtype=torch.float32, requires_grad=True)
    teacher_x0 = torch.nextafter(gen_data.detach(), torch.zeros_like(gen_data))
    fake_score_x0 = teacher_x0.clone()
    fake_score_x0[:, ::2] = gen_data.detach()[:, ::2] + torch.finfo(torch.float32).eps

    loss = variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0)
    (gradient,) = torch.autograd.grad(loss, gen_data)

    normalization = 1 / ((gen_data.detach() - teacher_x0).abs().mean(dim=1, keepdim=True) + 1e-6)
    expected = normalization * (fake_score_x0 - teacher_x0) / gen_data.numel()
    torch.testing.assert_close(gradient, expected, rtol=2e-6, atol=0)
    assert torch.count_nonzero(gradient) == torch.count_nonzero(expected)


def test_sample_scaled_gan_losses_match_the_active_sample():
    fake_logits = torch.tensor([[0.5, -0.5], [1.0, 2.0]], requires_grad=True)
    loss_scale = torch.tensor([0.0, 2.0])
    scaled_generator_loss = gan_loss_generator(fake_logits, loss_scale)

    reference_fake_logits = fake_logits.detach()[1:].clone().requires_grad_()
    reference_generator_loss = gan_loss_generator(reference_fake_logits)
    _assert_scaled_gradient_matches_active_reference(
        scaled_generator_loss,
        fake_logits,
        reference_generator_loss,
        reference_fake_logits,
    )

    real_logits = torch.tensor([[-1.0, 0.0], [0.5, 1.5]], requires_grad=True)
    fake_logits = torch.tensor([[0.5, -0.5], [1.0, 2.0]], requires_grad=True)
    scaled_discriminator_loss = gan_loss_discriminator(real_logits, fake_logits, loss_scale)
    reference_real_logits = real_logits.detach()[1:].clone().requires_grad_()
    reference_fake_logits = fake_logits.detach()[1:].clone().requires_grad_()
    reference_discriminator_loss = gan_loss_discriminator(reference_real_logits, reference_fake_logits)

    torch.testing.assert_close(scaled_discriminator_loss, reference_discriminator_loss)
    scaled_real_grad, scaled_fake_grad = torch.autograd.grad(
        scaled_discriminator_loss,
        (real_logits, fake_logits),
    )
    reference_real_grad, reference_fake_grad = torch.autograd.grad(
        reference_discriminator_loss,
        (reference_real_logits, reference_fake_logits),
    )
    torch.testing.assert_close(scaled_real_grad[0], torch.zeros_like(scaled_real_grad[0]))
    torch.testing.assert_close(scaled_fake_grad[0], torch.zeros_like(scaled_fake_grad[0]))
    torch.testing.assert_close(scaled_real_grad[1], reference_real_grad[0])
    torch.testing.assert_close(scaled_fake_grad[1], reference_fake_grad[0])


@pytest.mark.parametrize("input_dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_gan_losses_compute_softplus_in_high_precision(input_dtype):
    loss_dtype = high_precision_dtype(input_dtype)
    fake_logits = torch.tensor([[0.125, -0.875], [3.125, -6.75]], dtype=input_dtype)
    real_logits = torch.tensor([[-1.25, 0.375], [0.625, 1.75]], dtype=input_dtype)
    loss_scale = torch.tensor([0.25, 0.75])

    promoted_fake_logits = fake_logits.to(loss_dtype)
    promoted_real_logits = real_logits.to(loss_dtype)
    promoted_loss_scale = loss_scale.to(loss_dtype)
    expected_generator_loss = (
        torch.nn.functional.softplus(-promoted_fake_logits).mean(dim=1) * promoted_loss_scale
    ).mean()
    expected_discriminator_loss = (
        torch.nn.functional.softplus(promoted_fake_logits).mean(dim=1) * promoted_loss_scale
    ).mean() + (torch.nn.functional.softplus(-promoted_real_logits).mean(dim=1) * promoted_loss_scale).mean()

    generator_loss = gan_loss_generator(fake_logits, loss_scale)
    discriminator_loss = gan_loss_discriminator(real_logits, fake_logits, loss_scale)

    assert generator_loss.dtype == discriminator_loss.dtype == loss_dtype
    torch.testing.assert_close(generator_loss, expected_generator_loss)
    torch.testing.assert_close(discriminator_loss, expected_discriminator_loss)


def test_all_zero_loss_scale_keeps_a_finite_zero_gradient_graph():
    prediction = torch.ones(2, 3, requires_grad=True)
    loss = flow_matching_loss(
        prediction,
        "x0",
        x0=torch.zeros_like(prediction),
        loss_scale=torch.zeros(2),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0, dtype=high_precision_dtype(prediction)))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def _build_model_data(precision_model=None):
    """Build a prepared DMD2 model and a batch; ``precision_model`` defaults to fp64 on CPU, bf16 on GPU."""
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    instance = ModelConfig()
    opts = ["-", "backbone.img_resolution=8", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    default_precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.precision_model = default_precision_model if precision_model is None else precision_model
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.student_update_freq = 2
    instance.input_shape = [3, 8, 8]

    model = DMD2Model(instance)
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
    return _build_model_data()


def test_single_train_step_student_update(get_model_data, monkeypatch):
    model, data = get_model_data
    model.config.vsd_loss_weight = 0.25
    generation_marker = torch.ones((), device=model.device)
    gen_data_from_net = model.gen_data_from_net

    def generate_with_metric(*args, **kwargs):
        assert kwargs["neg_condition"] is data["neg_condition"]
        generation = gen_data_from_net(*args, **kwargs)
        return attrs.evolve(generation, metrics={**generation.metrics, "generation_marker": generation_marker})

    monkeypatch.setattr(model, "gen_data_from_net", generate_with_metric)
    # Run the training step
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "vsd_loss" in loss_map
    assert "gan_loss_gen" in loss_map
    assert "avg_t_student" in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert callable(outputs["gen_rand"])
    assert isinstance(outputs["input_rand"], torch.Tensor)
    assert isinstance(outputs["gen_rand_train"], torch.Tensor)
    assert outputs["gen_rand"].keywords["neg_condition"] is data["neg_condition"]
    assert loss_map["generation_marker"] is generation_marker
    assert "generation_marker" not in outputs
    torch.testing.assert_close(
        loss_map["total_loss"],
        model.config.vsd_loss_weight * loss_map["vsd_loss"]
        + model.config.gan_loss_weight_gen * loss_map["gan_loss_gen"],
    )


def test_single_train_step_fake_score_update(monkeypatch):
    # Prepared modules keep their precision, so an fp32 run is built as fp32 rather than re-resolved.
    model, data = _build_model_data(precision_model="float32")
    generation_marker = torch.full((), 3.0, device=model.device)
    gen_data_from_net = model.gen_data_from_net

    def generate_with_metric(*args, **kwargs):
        generation = gen_data_from_net(*args, **kwargs)
        return attrs.evolve(generation, metrics={**generation.metrics, "generation_marker": generation_marker})

    monkeypatch.setattr(model, "gen_data_from_net", generate_with_metric)

    # Run the training step
    loss_map, outputs = model.single_train_step(data, 1)

    # Assertions
    assert "fake_score_loss" in loss_map
    assert "gan_loss_disc" in loss_map
    assert "avg_t_student" in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert callable(outputs["gen_rand"])
    assert isinstance(outputs["input_rand"], torch.Tensor)
    assert isinstance(outputs["gen_rand_train"], torch.Tensor)
    assert outputs["gen_rand"].keywords["neg_condition"] is data["neg_condition"]

    assert loss_map["generation_marker"] is generation_marker
    assert "generation_marker" not in outputs
    assert loss_map["total_loss"].dtype == high_precision_dtype(model.precision)


@pytest.mark.parametrize("iteration", [0, 1])
def test_generations_can_hand_outputs_and_the_reward_a_clean_estimate(get_model_data, monkeypatch, iteration):
    """Both updates log the explicit clean estimate, and student updates use it for rewards."""
    model, data = get_model_data
    scored = []
    model.reward = lambda gen_data, batch, condition_dropout_mask=None: scored.append(gen_data) or {
        "total_loss": gen_data.mean()
    }
    x0_approx = torch.full((1, 3, 8, 8), 5.0, device=model.device)
    gen_data_from_net = model.gen_data_from_net

    def generate_with_x0_approx(*args, **kwargs):
        return attrs.evolve(gen_data_from_net(*args, **kwargs), x0=x0_approx)

    monkeypatch.setattr(model, "gen_data_from_net", generate_with_x0_approx)

    loss_map, outputs = model.single_train_step(data, iteration)

    assert torch.equal(outputs["gen_rand_train"], x0_approx)
    if iteration % model.config.student_update_freq == 0:
        assert len(scored) == 1 and torch.equal(scored[0], x0_approx)
        assert "reward_loss" in loss_map
    else:
        assert scored == [] and "reward_loss" not in loss_map


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


# ---------------------------------------------------------------------------
# Backward simulation: the multistep student's own trajectory, carried
# ---------------------------------------------------------------------------


def _t_list(model):
    """The scheduler's student timesteps at the configured NFE."""
    return model.net.noise_scheduler.student_grid(device=model.device)


def _assert_student_step(student_step, step):
    """Backward simulation hands every sample the trajectory's student step."""
    assert (student_step == step).all()


def build_backward_simulation_model(steps=4, gan_loss_weight_gen=0.0, student_update_freq=5):
    """A backward-simulation student: ``real`` is absent from the batch unless a test puts it there.

    ``ode`` rather than the default ``sde`` so a step is a deterministic function of the
    generation, which is what lets the tests below name the state they expect.
    """
    gc.collect()
    instance = ModelConfig()
    opts = ["-", "backbone.img_resolution=8", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.student_update_freq = student_update_freq
    instance.input_shape = [3, 8, 8]
    instance.student_grid_cfg.steps = steps
    instance.student_sample_type = "ode"
    instance.backward_simulation = "staggered"
    instance.gan_loss_weight_gen = gan_loss_weight_gen

    model = DMD2Model(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 2
    labels = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10)
    neg_condition = torch.zeros(batch_size, 10)
    # EDM takes its conditioning in FP32, as its ``build_conditioning`` returns it.
    return model, {
        "condition": labels.to(model.device, torch.float32),
        "neg_condition": neg_condition.to(model.device, torch.float32),
        "condition_raw": [f"prompt {i}" for i in range(batch_size)],
        "idx": [f"sample_{i}" for i in range(batch_size)],
        "fname": [f"sample_{i}.mp4" for i in range(batch_size)],
        "shard": ["shard-0"] * batch_size,
    }


@pytest.fixture
def get_backward_sim_model_data():
    return build_backward_simulation_model()


def test_backward_simulation_resumes_where_the_carry_left_off(get_backward_sim_model_data):
    """The student's input at step ``k`` is the state the previous iteration handed on, at the
    grid's ``k``-th time -- the time inference will call it at.

    A trajectory also has to keep the condition it set out under: switching prompt halfway would
    put the student off its own distribution in a way of its own, which is what this is avoiding.
    """
    model, data = get_backward_sim_model_data
    steps = model.net.noise_scheduler.default_student_steps
    batch_size = 2
    template = torch.zeros(batch_size, *model.config.input_shape, device=model.device, dtype=model.precision)

    # An empty carry at the first training iteration starts at rank zero's initial phase.
    _inputs = model._training_inputs(data)
    x = _inputs.input_student
    student_step = _inputs.student_step
    step = _inputs.trajectory_step
    loss_scale = _inputs.loss_scale
    assert step == 0
    _assert_student_step(student_step, step)
    assert loss_scale is None

    # Mid-trajectory it resumes from the carried state. The callback has already overlaid the
    # canonical condition/real fields, so there is no parallel ``*_carried`` namespace.
    state, carried_condition = torch.zeros_like(template), torch.zeros_like(data["condition"])
    carried_neg_condition = torch.ones_like(data["neg_condition"])
    carried_real = torch.ones_like(template)
    carried = {
        **data,
        "state_carried": state,
        "step_carried": 2,
        "condition": carried_condition,
        "neg_condition": carried_neg_condition,
        "real": carried_real,
    }
    _inputs = model._training_inputs(carried)
    x = _inputs.input_student
    student_step = _inputs.student_step
    step = _inputs.trajectory_step
    loss_scale = _inputs.loss_scale
    assert step == 2 and x is state
    _assert_student_step(student_step, step)
    assert loss_scale is None

    # And the step past the last one wraps round to a fresh trajectory under this batch's prompt.
    carried = {**data, "step_carried": steps}
    _inputs = model._training_inputs(carried)
    x = _inputs.input_student
    student_step = _inputs.student_step
    step = _inputs.trajectory_step
    loss_scale = _inputs.loss_scale
    assert step == 0 and x is not state and "real" not in carried
    assert carried["condition"] is data["condition"]
    assert carried["neg_condition"] is data["neg_condition"]
    _assert_student_step(student_step, step)
    assert loss_scale is None


def test_the_carried_state_is_the_re_noised_generation(get_backward_sim_model_data, monkeypatch):
    """One step, taken exactly as the sampler takes it, and detached.

    The state is an input to a later iteration rather than part of this one's graph, so a gradient
    must not be able to reach back through it.  Sharing ``noise_scheduler.step`` with
    ``_student_sample_loop`` is what keeps the simulated states the ones inference produces.
    """
    model, data = get_backward_sim_model_data
    student_grid = _t_list(model)
    step = 1
    x = torch.randn(
        len(data["idx"]),
        *model.config.input_shape,
        device=model.device,
        dtype=model.precision,
    )
    trajectory_data = {**data, "state_carried": x, "step_carried": step}
    generations = []
    gen_data_from_net = model.gen_data_from_net

    def record_generation(*args, **kwargs):
        generation = gen_data_from_net(*args, **kwargs)
        generations.append(generation.data)
        return generation

    monkeypatch.setattr(model, "gen_data_from_net", record_generation)

    _, outputs = model.single_train_step(trajectory_data, iteration=model.config.student_update_freq)
    (gen_data,) = generations
    assert gen_data.requires_grad, "the student's own generation carries a graph"
    assert not outputs["gen_rand_train"].requires_grad and torch.equal(outputs["gen_rand_train"], gen_data)

    carry = outputs[CARRY_OUTPUT_KEY]
    assert carry["step_carried"] == step + 1
    assert set(carry) == {*(key for key in CARRY_OVERLAY_KEYS if key in data), "state_carried", "step_carried"}
    assert carry["condition"] is data["condition"]
    assert carry["neg_condition"] is data["neg_condition"]
    assert carry["condition_raw"] is data["condition_raw"]
    assert "real" not in carry, "fields absent from the trajectory batch should remain absent"
    assert not carry["state_carried"].requires_grad and carry["state_carried"].grad_fn is None

    expected = model.net.noise_scheduler.step(
        x,
        gen_data.detach(),
        student_grid[step],
        student_grid[step + 1],
        sample_type="ode",
    )
    torch.testing.assert_close(carry["state_carried"], expected)


def test_training_backward_simulation_needs_only_conditions(get_backward_sim_model_data):
    """With the GAN loss off nothing discriminates against real data, so prompts are the batch.

    Run as the trainer runs it, with ``CarryCallback``'s job done by hand: every iteration takes
    one step of the walk, both phases advance it, and the fourth closes the trajectory.
    """
    model, data = get_backward_sim_model_data
    steps = model.net.noise_scheduler.default_student_steps
    assert "real" not in data, "the point of the exercise"

    batch = dict(data)
    for iteration in range(1, steps + 2):
        loss_map, outputs = model.single_train_step(batch, iteration)
        assert torch.isfinite(loss_map["total_loss"]), f"iteration {iteration}"
        assert "gan_loss_disc" not in loss_map
        carry = outputs[CARRY_OUTPUT_KEY]
        assert carry["step_carried"] == (iteration - 1) % steps + 1
        # The last step leaves only a boundary marker; the next iteration starts fresh without
        # mistaking that normal boundary for an empty post-resume carry.
        assert ("state_carried" not in carry) == (iteration % steps == 0)
        batch = {**data, **carry}


def test_backward_simulation_follows_an_overridden_student_grid(get_backward_sim_model_data, monkeypatch):
    """With two intervals in place of the scheduler's four, step 1 ends the trajectory and its marker restarts it."""
    model, data = get_backward_sim_model_data
    student_grid = model.net.noise_scheduler.student_grid(2, device=model.device)
    monkeypatch.setattr(model, "_student_grid", lambda device: student_grid.to(device))
    state = torch.randn(len(data["idx"]), *model.config.input_shape, device=model.device, dtype=model.precision)

    _, outputs = model.single_train_step(
        {**data, "state_carried": state, "step_carried": 1}, iteration=model.config.student_update_freq
    )
    carry = outputs[CARRY_OUTPUT_KEY]
    assert carry == {"step_carried": 2}

    _inputs = model._training_inputs({**data, **carry})
    step = _inputs.trajectory_step
    assert step == 0


def test_the_student_grid_is_resolved_in_one_place(get_backward_sim_model_data):
    """All multistep paths use the grid policy installed on the scheduler."""
    model, _ = get_backward_sim_model_data
    net = model.net
    scheduler = net.noise_scheduler
    steps = scheduler.default_student_steps
    t_precision = scheduler.t_precision

    assert scheduler.student_grid_cfg is model.config.student_grid_cfg
    assert scheduler.student_grid_cfg.steps == steps
    assert scheduler.student_grid_cfg.grid_type == "native"
    assert "timesteps" not in scheduler.student_grid_cfg.grid_kwargs
    torch.testing.assert_close(_t_list(model), scheduler.student_grid(device=model.device))

    model.config.student_grid_cfg.grid_type = "explicit"
    model.config.student_grid_cfg.grid_kwargs = {"timesteps": [0.9, 0.6, 0.3, 0.1, 0.0]}
    torch.testing.assert_close(
        _t_list(model),
        torch.tensor([0.9, 0.6, 0.3, 0.1, 0.0], device=model.device, dtype=t_precision),
    )
    with pytest.raises(ValueError, match="Cannot select"):
        scheduler.student_grid(steps + 1, device=model.device)

    for grid_type, grid_kwargs in (
        ("native", {}),
        ("explicit", {"timesteps": [0.9, 0.6, 0.3, 0.1, 0.0]}),
    ):
        model.config.student_grid_cfg.grid_type = grid_type
        model.config.student_grid_cfg.grid_kwargs = grid_kwargs
        assert scheduler.student_grid(dtype=torch.float32).dtype == torch.float32
        assert scheduler.student_grid().dtype == t_precision

    model.config.student_grid_cfg.grid_type = "explicit"
    model.config.student_grid_cfg.grid_kwargs = {"timesteps": [0.9, 0.6, 0.3, 0.1, 0.0]}
    with pytest.raises(ValueError, match="Cannot select"):
        scheduler.sample_student_timesteps(2, num_steps=steps + 1, device=model.device)
    with pytest.raises(ValueError, match="Cannot select"):
        scheduler.sample_student_timesteps_inhom(
            2,
            seq_len=4,
            chunk_size=2,
            num_steps=steps + 1,
            device=model.device,
        )
    model.config.student_grid_cfg.grid_kwargs = {"timesteps": [0.9, 0.5, 0.1]}
    # A CUDA grid is checked by a device-side assert, which no exception reports; check the host grid.
    with pytest.raises(ValueError, match="end at zero"):
        scheduler.student_grid(2, device="cpu")


def test_data_driven_multistep_sampling_uses_the_model_student_grid(get_model_data, monkeypatch):
    model, data = get_model_data
    model.config.student_grid_cfg.steps = 4
    scheduler = model.net.noise_scheduler
    # Without its first node, the grid differs from the configured one at every step index.
    student_grid = scheduler.student_grid(device=model.device)[1:]
    monkeypatch.setattr(model, "_student_grid", lambda device: student_grid.to(device))
    noised_to = []
    forward_process = scheduler.forward_process

    def record_forward_process(x, eps, t, t_start=None):
        noised_to.append(t)
        return forward_process(x, eps, t, t_start=t_start)

    monkeypatch.setattr(scheduler, "forward_process", record_forward_process)
    _inputs = model._training_inputs(data)
    student_step = _inputs.student_step

    [t_student] = noised_to
    assert ((student_step >= 0) & (student_step < len(student_grid) - 1)).all()
    torch.testing.assert_close(t_student, student_grid[student_step])


def test_data_driven_one_step_uses_the_initial_grid_timestep(get_model_data, monkeypatch):
    model, data = get_model_data
    data = {key: value for key, value in data.items() if key != "real"}

    def unexpected_sample(*args, **kwargs):
        raise AssertionError("a one-step student has no timestep choice to sample")

    monkeypatch.setattr(model.net.noise_scheduler, "sample_student_timesteps", unexpected_sample)
    _inputs = model._training_inputs(data)
    student_step = _inputs.student_step

    assert student_step.tolist() == [0] * len(data["idx"])


def test_data_driven_multistep_requires_real_data(get_model_data):
    model, data = get_model_data
    model.config.student_grid_cfg.steps = 4
    data = {key: value for key, value in data.items() if key != "real"}

    with pytest.raises(ValueError, match="Multi-step data-driven training requires data\\['real'\\]"):
        model._training_inputs(data)


# ---------------------------------------------------------------------------
# Student-grid buckets for the noising time
# ---------------------------------------------------------------------------


def _assert_in_student_buckets(model, t, student_step):
    """Every noising time lies in the bucket ``[t_{i+1}, t_i]`` of its generator input's student step ``i``."""
    grid = _t_list(model)
    assert t.dtype == model.net.noise_scheduler.t_precision
    assert (t >= grid[student_step + 1]).all() and (t <= grid[student_step]).all()


def _labelled_batch(model, batch_size):
    labels = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10)
    return {
        "idx": list(range(batch_size)),
        "real": torch.randn(batch_size, 3, 8, 8, device=model.device, dtype=model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": torch.zeros(batch_size, 10, device=model.device, dtype=model.precision),
    }


def test_student_buckets_bound_the_data_driven_noising_time(get_model_data):
    model, _ = get_model_data
    model.config.student_grid_cfg.steps = 4
    model.config.sample_t_in_student_bucket = True

    torch.manual_seed(0)
    _inputs = model._training_inputs(_labelled_batch(model, 512))
    student_step = _inputs.student_step
    t = _inputs.t

    _assert_in_student_buckets(model, t, student_step)
    assert set(student_step.tolist()) == {0, 1, 2, 3}


def test_a_one_step_student_bucket_is_the_whole_noising_range(get_model_data):
    model, data = get_model_data
    model.config.sample_t_in_student_bucket = True

    _inputs = model._training_inputs(data)
    student_step = _inputs.student_step
    t = _inputs.t

    assert (student_step == 0).all()
    _assert_in_student_buckets(model, t, student_step)


def test_student_buckets_follow_the_backward_simulation_step(get_backward_sim_model_data):
    model, data = get_backward_sim_model_data
    model.config.sample_t_in_student_bucket = True
    state = torch.zeros(2, *model.config.input_shape, device=model.device, dtype=model.precision)

    batches = [data] + [{**data, "state_carried": state, "step_carried": step} for step in (1, 2, 3)]
    for expected_step, batch in enumerate(batches):
        _inputs = model._training_inputs(batch)
        student_step = _inputs.student_step
        t = _inputs.t
        step = _inputs.trajectory_step
        assert step == expected_step
        _assert_student_step(student_step, expected_step)
        _assert_in_student_buckets(model, t, student_step)


def test_student_buckets_bound_the_real_and_fake_score_noising(get_model_data, monkeypatch):
    model, _ = get_model_data
    model.config.student_grid_cfg.steps = 4
    model.config.sample_t_in_student_bucket = True
    model.config.gan_use_same_t_noise = False
    model.config.fake_score_sample_t_overrides = {"min_t": 0.01}
    draws = []
    sample_t_and_eps = model._sample_t_and_eps

    def record_sample(batch_size, sample_t_cfg=None, student_step=None):
        t, eps = sample_t_and_eps(batch_size, sample_t_cfg=sample_t_cfg, student_step=student_step)
        draws.append((sample_t_cfg, student_step, t))
        return t, eps

    monkeypatch.setattr(model, "_sample_t_and_eps", record_sample)
    model.single_train_step(_labelled_batch(model, 16), iteration=1)

    # The main (shared VSD/GAN) draw, the discriminator's real draw, then the fake-score draw.
    assert [cfg for cfg, _, _ in draws] == [None, None, attrs.evolve(model.config.sample_t_cfg, min_t=0.01)]
    student_step = draws[0][1]
    for _, seen_student_step, t in draws:
        assert seen_student_step is student_step
        _assert_in_student_buckets(model, t, student_step)


def test_student_buckets_require_one_student_step_per_sample(get_model_data):
    model, _ = get_model_data
    model.config.student_grid_cfg.steps = 4
    model.config.sample_t_in_student_bucket = True

    with pytest.raises(ValueError, match="needs one student step per sample"):
        model._sample_t_and_eps(2)
    with pytest.raises(ValueError, match="needs one student step per sample"):
        model._sample_t_and_eps(2, student_step=torch.zeros(3, dtype=torch.long, device=model.device))


def test_student_buckets_must_overlap_the_sampling_range(get_model_data):
    model, _ = get_model_data
    model.config.student_grid_cfg.steps = 4
    model.config.sample_t_in_student_bucket = True
    grid = _t_list(model)
    # Only the top bucket reaches above this range's minimum.
    model.config.sample_t_cfg.min_t = float(grid[1])

    t, _ = model._sample_t_and_eps(2, student_step=torch.zeros(2, dtype=torch.long, device=model.device))
    assert (t >= grid[1]).all() and (t <= grid[0]).all()
    # CPU indices keep the overlap check synchronous wherever the model lives.
    with pytest.raises(ValueError, match="window with start < end"):
        model._sample_t_and_eps(2, student_step=torch.full((2,), 3))


def test_a_one_step_student_is_the_trivial_trajectory(monkeypatch):
    """One step is the degenerate walk, and it needs no special case: the grid is ``[max_t, 0]``,
    so step 0 is already the last, the carry never fills, and every iteration draws fresh noise --
    which is exactly the one-step recipe. Worth pinning because the alternative to it working is a
    config combination nobody can use.
    """
    model, data = build_backward_simulation_model(steps=1)
    batch = dict(data)
    for iteration in range(1, 4):
        loss_map, outputs = model.single_train_step(batch, iteration)
        assert torch.isfinite(loss_map["total_loss"]), f"iteration {iteration}"
        carry = outputs[CARRY_OUTPUT_KEY]
        assert "state_carried" not in carry, "a one-step walk has nothing to hand on"
        assert carry["step_carried"] == 1
        batch = {**data, **carry}

    # And it starts from the top of the grid every time, rather than resuming anything.
    _inputs = model._training_inputs({**data, "step_carried": 1})
    step = _inputs.trajectory_step
    assert step == 0

    # Initialization and resume are also trivial on every rank: modulo one is
    # zero and the staggered rollout must execute no student forwards.
    monkeypatch.setattr(backward_simulation, "world_size", lambda: 4)
    monkeypatch.setattr(backward_simulation, "get_rank", lambda: 3)

    def unexpected_forward(*args, **kwargs):
        raise AssertionError("one-step staggering must not forward")

    monkeypatch.setattr(model, "_advance_student_step", unexpected_forward)
    _inputs = model._training_inputs(data, iteration=9)
    step = _inputs.trajectory_step
    assert step == 0


def test_backward_simulation_rejects_uncovered_student_phases():
    """A rank must represent each trajectory phase that student updates can revisit."""
    model, data = build_backward_simulation_model(steps=4, student_update_freq=2)
    with pytest.raises(ValueError, match="cannot cover every student timestep"):
        model._training_inputs(data)


def test_backward_simulation_resume_recovers_uninterrupted_rank_phase(monkeypatch):
    """Staggering reconstructs the phase from iteration and rank, including validation at zero.

    The model forward and scheduler transition are replaced with identities so the real staggered
    start remains exercised without paying for its rollout.
    """
    model, data = build_backward_simulation_model(steps=4)
    active_rank = 0
    monkeypatch.setattr(backward_simulation, "get_rank", lambda: active_rank)
    monkeypatch.setattr(backward_simulation, "world_size", lambda: 4)
    monkeypatch.setattr(
        model,
        "_advance_student_step",
        lambda net, state, step_index, student_grid, **kwargs: state,
    )

    iteration_and_completed = [(1, 0)]
    iteration_and_completed.extend((checkpoint + 1, checkpoint) for checkpoint in range(1, 8))
    for iteration, completed_iterations in iteration_and_completed:
        for rank in range(4):
            active_rank = rank
            _inputs = model._training_inputs(data, iteration=iteration)
            step = _inputs.trajectory_step
            expected = (rank + completed_iterations) % model.net.noise_scheduler.default_student_steps
            assert step == expected

    # Training iterations are one-based, and validation never reaches the stagger.
    for invalid in (0, -1):
        with pytest.raises(ValueError, match="iteration must be positive"):
            model._training_inputs(data, iteration=invalid)


def test_gradient_accumulation_slots_preserve_phase_but_restart_trajectory_on_resume(monkeypatch):
    """Resume preserves the phase clock, not transient per-slot state or batch context."""
    model, data = build_backward_simulation_model(steps=4)
    callback = CarryCallback()
    rank = 1
    accum_rounds = 3
    seen_offsets = []
    monkeypatch.setattr(backward_simulation, "get_rank", lambda: rank)
    monkeypatch.setattr(backward_simulation, "world_size", lambda: 4)
    monkeypatch.setattr(
        model,
        "_advance_student_step",
        lambda net, state, step_index, student_grid, **kwargs: state,
    )
    states = []
    conditions = []
    prompts = []

    # All slots start at this rank's staggered phase, but retain distinct states and batch context.
    for accum in range(accum_rounds):
        condition = torch.full_like(data["condition"], accum + 1)
        prompt = [f"trajectory-{accum}-{i}" for i in range(2)]
        fresh_batch = {**data, "condition": condition, "condition_raw": prompt}
        batch = dict(fresh_batch)
        callback.on_training_accum_step_begin(model, batch, iteration=1, accum_iter=accum)
        _inputs = model._training_inputs(batch, iteration=1)
        step = _inputs.trajectory_step
        assert step == rank
        seen_offsets.append(step)

        state = torch.full((2, *model.config.input_shape), accum + 1, device=model.device, dtype=model.precision)
        carry = {key: batch[key] for key in CARRY_OVERLAY_KEYS if key in batch}
        carry.update({"state_carried": state, "step_carried": step + 1})
        callback.on_training_accum_step_end(model, batch, {CARRY_OUTPUT_KEY: carry}, iteration=1, accum_iter=accum)
        states.append(state)
        conditions.append(condition)
        prompts.append(prompt)

    assert seen_offsets == [rank] * accum_rounds

    # The next global iteration retrieves the matching slot rather than consuming another
    # accumulation position's trajectory. Active slots do not restagger.
    for accum in range(accum_rounds):
        fresh_batch = {
            **data,
            "condition": torch.zeros_like(data["condition"]),
            "condition_raw": ["wrong fresh prompt"] * 2,
        }
        batch = dict(fresh_batch)
        callback.on_training_accum_step_begin(model, batch, iteration=2, accum_iter=accum)
        assert batch["state_carried"] is states[accum]
        assert batch["condition"] is conditions[accum]
        assert batch["condition_raw"] is prompts[accum]
        _inputs = model._training_inputs(batch, iteration=2)
        step = _inputs.trajectory_step
        assert step == rank + 1

    assert seen_offsets == [rank] * accum_rounds

    # Checkpoint load clears every transient slot. Each one independently reconstructs the same
    # uninterrupted per-rank phase; accumulation position is deliberately not part of the clock.
    # This checkpoint resumes rank zero at phase two, so this also records that phase continuity
    # does not imply exact trajectory continuity: each slot starts from its newly loaded batch.
    callback.load_state_dict(callback.state_dict())
    checkpoint_iteration = 6
    num_steps = model.net.noise_scheduler.default_student_steps
    assert checkpoint_iteration % num_steps == 2, "rank zero resumes at a nonzero trajectory phase"
    expected_resume_step = (rank + checkpoint_iteration) % num_steps
    for accum in range(accum_rounds):
        resumed_condition = torch.full_like(data["condition"], 100 + accum)
        resumed_prompt = [f"resumed-{accum}-{i}" for i in range(2)]
        batch = {**data, "condition": resumed_condition, "condition_raw": resumed_prompt}
        callback.on_training_accum_step_begin(
            model,
            batch,
            iteration=checkpoint_iteration + 1,
            accum_iter=accum,
        )
        assert "state_carried" not in batch and "step_carried" not in batch
        _inputs = model._training_inputs(batch, iteration=checkpoint_iteration + 1)
        step = _inputs.trajectory_step
        assert step == expected_resume_step
        assert batch["condition"] is resumed_condition
        assert batch["condition_raw"] is resumed_prompt
        seen_offsets.append(step)

    assert seen_offsets == [rank] * accum_rounds + [expected_resume_step] * accum_rounds


def test_backward_simulation_boundary_marker_does_not_restagger(monkeypatch, get_backward_sim_model_data):
    """A normal trajectory boundary is latent-free but is not a first/resume iteration."""
    model, data = get_backward_sim_model_data

    def unexpected_advance(*args, **kwargs):
        raise AssertionError("a normal boundary must not run the staggered rollout")

    monkeypatch.setattr(model, "_advance_student_step", unexpected_advance)
    _inputs = model._training_inputs(
        {**data, "step_carried": model.net.noise_scheduler.default_student_steps}, iteration=7
    )
    student_step = _inputs.student_step
    step = _inputs.trajectory_step
    loss_scale = _inputs.loss_scale
    assert step == 0
    _assert_student_step(student_step, step)
    assert loss_scale is None


def test_backward_simulation_validation_does_not_stagger(monkeypatch, get_backward_sim_model_data):
    """Validation skips the staggered rollout and starts at the head of the grid.

    The stagger exists so that, across ranks, every trajectory step receives a student
    gradient. Validation runs under ``no_grad`` and updates nothing, so its ``steps - 1``
    student forwards would be built and thrown away.
    """
    model, data = get_backward_sim_model_data

    def unexpected_advance(*args, **kwargs):
        raise AssertionError("validation must not run the staggered rollout")

    monkeypatch.setattr(model, "_advance_student_step", unexpected_advance)
    with torch.no_grad():
        _inputs = model._training_inputs(data, iteration=7)
        student_step = _inputs.student_step
        step = _inputs.trajectory_step
        loss_scale = _inputs.loss_scale

    assert step == 0
    _assert_student_step(student_step, step)
    assert loss_scale is None


def test_only_globally_empty_carries_stagger_every_rank(monkeypatch, get_backward_sim_model_data):
    """All ranks stagger on initialization/resume; rank-local boundaries never stagger."""
    model, data = get_backward_sim_model_data
    steps = model.net.noise_scheduler.default_student_steps
    active_rank = 0
    staggered_ranks = []

    monkeypatch.setattr(backward_simulation, "get_rank", lambda: active_rank)
    monkeypatch.setattr(backward_simulation, "world_size", lambda: steps)

    def record_advance(net, state, step_index, student_grid, **kwargs):
        if step_index == 0:
            staggered_ranks.append(active_rank)
        return state

    monkeypatch.setattr(model, "_advance_student_step", record_advance)

    # A process start or resume clears every rank's transient carry, so every
    # rank takes the collective-safe staggered path.
    for rank in range(steps):
        active_rank = rank
        model._training_inputs(data, iteration=2)
    assert staggered_ranks == list(range(steps))

    staggered_ranks.clear()
    state = torch.zeros(2, *model.config.input_shape, device=model.device, dtype=model.precision)
    for rank in range(steps):
        active_rank = rank
        carry = (
            {"step_carried": steps}
            if rank == steps - 1
            else {
                "state_carried": state,
                "step_carried": rank + 1,
            }
        )
        model._training_inputs({**data, **carry}, iteration=2)
    assert staggered_ranks == [], "a rank-local boundary must not add unmatched FSDP forwards"


def test_backward_simulation_uses_the_indexed_carried_batch_size(get_backward_sim_model_data):
    """The atomic carry's sample indices remain the authoritative batch-size source."""
    model, data = get_backward_sim_model_data
    state = torch.zeros(1, *model.config.input_shape, device=model.device, dtype=model.precision)
    carried_condition = data["condition"][:1]
    carried_neg_condition = data["neg_condition"][:1]
    batch = {
        **data,
        "state_carried": state,
        "condition": carried_condition,
        "neg_condition": carried_neg_condition,
        "idx": data["idx"][:1],
        "step_carried": 1,
    }

    _inputs = model._training_inputs(batch)
    input_student = _inputs.input_student
    student_step = _inputs.student_step
    t = _inputs.t
    eps = _inputs.eps
    trajectory_step = _inputs.trajectory_step
    loss_scale = _inputs.loss_scale

    assert input_student is state and trajectory_step == 1 and "real" not in batch
    assert t.shape == (1,)
    assert eps.shape[0] == 1
    assert student_step.shape == (1,)
    _assert_student_step(student_step, trajectory_step)
    assert loss_scale is None


def test_real_data_rides_along_the_trajectory_and_brings_the_gan_loss_with_it():
    """Backward simulation means data is not *required*, not that it is refused.

    When the loader does yield real data, the adversarial terms come back -- and the real sample
    is frozen alongside the prompt for the whole walk, because the discriminator scores it under
    that condition. Pairing a fresh real sample with a carried prompt would hand it a mismatched
    real example, and it would learn the mismatch rather than the distribution.
    """
    model, data = build_backward_simulation_model(gan_loss_weight_gen=0.001)
    callback = CarryCallback()
    real = torch.randn(2, 3, 8, 8).to(model.device, model.precision)
    batch = {**data, "real": real}

    # The critic phase is where the discriminator sees real data, so check that phase.
    loss_map, outputs = model.single_train_step(batch, 1)
    assert "gan_loss_disc" in loss_map, "real data is there, so the discriminator trains on it"
    assert outputs[CARRY_OUTPUT_KEY]["real"] is real
    callback.on_training_accum_step_end(model, batch, outputs, iteration=1, accum_iter=0)

    # DMD2 includes the pair in its active carry, and the callback restores it atomically even as
    # the loader moves on to another batch.
    next_real = torch.randn_like(real)
    fresh_batch = {
        "condition": torch.zeros_like(data["condition"]),
        "condition_raw": ["wrong fresh prompt"] * 2,
        "real": next_real,
        "idx": ["wrong-0", "wrong-1"],
    }
    batch = dict(fresh_batch)
    callback.on_training_accum_step_begin(model, batch, iteration=2, accum_iter=0)
    assert batch["condition"] is data["condition"]
    assert batch["condition_raw"] is data["condition_raw"]
    assert batch["real"] is real
    assert extract_prompt(batch, 0) == data["condition_raw"][0]
    loss_map, outputs = model.single_train_step(batch, 2)
    assert outputs["gen_rand"].keywords["condition"] is data["condition"]
    assert outputs["gen_rand"].keywords["neg_condition"] is data["neg_condition"]
    assert "gan_loss_disc" in loss_map

    # DMD2 again attaches the effective batch while this trajectory remains active.
    assert outputs[CARRY_OUTPUT_KEY]["real"] is real
    callback.on_training_accum_step_end(model, batch, outputs, iteration=2, accum_iter=0)
    fresh_next_batch = {"condition": torch.zeros_like(data["condition"]), "real": torch.randn_like(real)}
    next_batch = dict(fresh_next_batch)
    callback.on_training_accum_step_begin(model, next_batch, iteration=3, accum_iter=0)
    assert next_batch["condition"] is data["condition"]
    assert next_batch["condition_raw"] is data["condition_raw"]
    assert next_batch["real"] is real

    # With no real data anywhere, the same model leaves the adversarial terms out instead.
    loss_map, _ = model.single_train_step(dict(data), 1)
    assert "gan_loss_disc" not in loss_map
    assert torch.isfinite(loss_map["total_loss"])


def test_without_real_data_the_student_update_skips_the_untrained_discriminator(monkeypatch):
    """The discriminator trains only on real data, so without it the student update leaves its GAN term out too."""
    model, data = build_backward_simulation_model(gan_loss_weight_gen=0.001)
    student_iteration = model.config.student_update_freq

    def untrained_discriminator(*args, **kwargs):
        raise AssertionError("the student update must not face a discriminator that never trains")

    monkeypatch.setattr(model.discriminator, "forward", untrained_discriminator)
    loss_map, _ = model.single_train_step(dict(data), student_iteration)

    assert loss_map["gan_loss_gen"] == 0
    torch.testing.assert_close(loss_map["total_loss"], model.config.vsd_loss_weight * loss_map["vsd_loss"])

    # Without VSD, nothing is left to train the student on.
    model.config.vsd_loss_weight = 0.0
    with pytest.raises(ValueError, match=r"GAN training with data\['real'\]"):
        model.single_train_step(dict(data), student_iteration)


# ---------------------------------------------------------------------------
# Multimodal DMD2 tests
# ---------------------------------------------------------------------------


def _build_multimodal_model_data(precision_model=None):
    """Build a prepared multimodal DMD2 model and a batch; ``precision_model`` defaults as in ``_build_model_data``."""
    gc.collect()
    instance = ModelConfig()
    opts = ["-", "backbone.img_resolution=8", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.discriminator = L(CompositeDiscriminator)(
        modality_discs={
            "video": L(Discriminator_VideoDiT)(
                num_blocks=1,
                disc_type="dit_simple_conv3d",
                inner_dim=128,
            ),
            "audio": L(Discriminator_Audio2D)(
                inner_dim=128,
                hidden_channels=64,
            ),
        },
        feature_indices=[0],
    )

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    default_precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.precision_model = default_precision_model if precision_model is None else precision_model
    instance.pretrained_model_path = None
    instance.student_update_freq = 2
    instance.input_shape = {"video": [3, 8, 8], "audio": [3, 4, 4]}

    model = DMD2Model(instance)

    model.net = MultimodalNet(model.net)
    model.teacher = MultimodalNet(model.teacher)
    model.fake_score = MultimodalNet(model.fake_score)

    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    neg_condition = torch.zeros(batch_size, 10)

    video = torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision)
    audio = torch.randn(batch_size, 3, 4, 4).to(model.device, model.precision)
    real = {"video": video, "audio": audio}

    data = {
        "idx": list(range(batch_size)),
        "real": real,
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_condition.to(model.device, model.precision),
    }

    return model, data


@pytest.fixture
def get_multimodal_model_data():
    return _build_multimodal_model_data()


def test_multimodal_student_update_step(get_multimodal_model_data):
    model, data = get_multimodal_model_data
    loss_map, outputs = model.single_train_step(data, 0)

    assert "total_loss" in loss_map
    assert "gan_loss_gen" in loss_map
    assert "video_vsd_loss" in loss_map
    assert "audio_vsd_loss" in loss_map
    assert "gen_rand" in outputs


def test_multimodal_auxiliary_update():
    # Prepared modules keep their precision, so the fp32 dtype check builds the model as fp32.
    model, data = _build_multimodal_model_data(precision_model="float32")
    model.config.gan_r1_reg_weight = 0.1

    loss_map, outputs = model.single_train_step(data, 1)

    assert ("gan_loss_disc" in loss_map) == (model.config.gan_loss_weight_gen > 0)
    assert "gan_loss_ar1" in loss_map
    assert "video_fake_score_loss" in loss_map
    assert "audio_fake_score_loss" in loss_map
    assert loss_map["total_loss"].dtype == high_precision_dtype(model.precision)


def test_multimodal_optimizers(get_multimodal_model_data):
    model, data = get_multimodal_model_data
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)

    model.optimizers_zero_grad(2)
    check_grad_zero(model.net)
    model.optimizers_zero_grad(3)
    check_grad_zero(model.discriminator)
    check_grad_zero(model.fake_score)


# ---------------------------------------------------------------------------
# DMD2 step generations (Phased DMD): sampler-step generation and sampling, transition renoising, the fake-score
# loss, and rewards.
# ---------------------------------------------------------------------------

WAN_GRID = [0.999, 0.937, 0.833, 0.624, 0.0]
FAKE_SCORE_WEIGHTINGS = ("variance_matched", "inverse_variance", "conservative", "x0", "unweighted")


class _TinyFlowNet(FastGenNetwork):
    """A 1x1-convolution flow head, reported in the net's prediction type."""

    def __init__(self, student_grid_cfg=None, schedule_type="rf", net_pred_type="flow", **schedule_kwargs):
        super().__init__(
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            student_grid_cfg=student_grid_cfg,
            **schedule_kwargs,
        )
        self.proj = torch.nn.Conv2d(3, 3, 1)

    def forward(
        self, x_t, t, condition=None, fwd_pred_type=None, apply_output_conditioning=True, output_dtype="same", **kwargs
    ):
        self._validate_forward_kwargs(kwargs)
        prediction = self.noise_scheduler.convert_model_output(
            x_t,
            self.proj(x_t.to(dtype=self.precision_model)).to(dtype=x_t.dtype),
            t,
            src_pred_type="flow",
            target_pred_type=self.net_pred_type,
        )
        return self.transform_output(
            prediction, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
        )


def _build_model(net=None, **overrides):
    gc.collect()
    config = ModelConfig()
    config.net = net if net is not None else L(_TinyFlowNet)()
    config.device = torch.device("cpu")
    config.precision = "float32"
    config.pretrained_model_path = None
    config.gan_loss_weight_gen = 0.0
    config.student_update_freq = 2
    config.input_shape = [3, 8, 8]
    config.student_grid_cfg = StudentGridConfig(steps=4, grid_type="explicit", grid_kwargs={"timesteps": WAN_GRID})
    config.sample_t_cfg = SampleTConfig(time_dist_type="uniform", min_t=0.001, max_t=0.999)
    # Phased DMD's step generations and fake-score loss.
    config.student_generation = "step"
    config.student_sample_type = "ode"
    config.sample_t_in_student_bucket = True
    config.fake_score_pred_type = "eps"
    config.fake_score_weighting = "variance_matched"
    for key, value in overrides.items():
        setattr(config, key, value)
    model = DMD2Model(config)
    model.on_train_begin()
    model.init_optimizers()
    return model


def _expand(values, like):
    return values.reshape(-1, *([1] * (like.ndim - 1)))


def _transition(t_start, t):
    """Mean scale and variance of the unshifted rectified-flow transition from ``t_start`` to ``t >= t_start``."""
    a = (1 - t) / (1 - t_start)
    return a, t**2 - a**2 * t_start**2


# ---------------------------------------------------------------------------
# Fake-score weights
# ---------------------------------------------------------------------------


def test_every_weighting_is_a_dmd2_loss_for_clean_generations():
    """DMD2's x0 loss weights the unit-noise residual by ``t^2 / (1-t)^2``; its eps loss does not weight it."""
    t = torch.linspace(0.01, 0.99, 9, dtype=torch.float64)
    _, v = _transition(torch.zeros_like(t), t)

    for weighting in FAKE_SCORE_WEIGHTINGS:
        weight = transition_score_weight(weighting, 1 - t, t, v, data_variance=0.7)
        expected = torch.ones_like(t) if weighting == "unweighted" else t**2 / (1 - t) ** 2
        torch.testing.assert_close(weight, expected)


def test_clean_generations_take_dmd2s_loss_where_the_schedules_time_zero_is_noisy():
    """SD's first discrete time 0 carries noise, but the transition from the clean end 0 is still the forward process
    DMD2 renoises clean generations with, so every weighting is DMD2's x0 loss (its eps loss for ``unweighted``)."""
    torch.manual_seed(0)
    betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=torch.float64) ** 2
    schedule = AlphasNoiseSchedule(alphas_cumprod=torch.cumprod(1 - betas, dim=0))
    assert schedule.sigma(torch.zeros(1, dtype=torch.float64)) > 0
    t = torch.tensor([0.002, 0.01, 0.05, 0.5], dtype=torch.float64)
    t_start = torch.zeros_like(t)
    x0, eps, eps_pred = (torch.randn(4, 3, 8, 8, dtype=torch.float64) for _ in range(3))

    x_t = schedule.forward_process(x0, eps, t)
    torch.testing.assert_close(schedule.forward_process(x0, eps, t, t_start=t_start), x_t)
    x0_pred = schedule.convert_model_output(x_t, eps_pred, t, src_pred_type="eps", target_pred_type="x0")
    x0_loss = flow_matching_loss(x0_pred, "x0", x0, eps, schedule, t)
    eps_loss = flow_matching_loss(eps_pred, "eps", x0, eps, schedule, t)
    for weighting in FAKE_SCORE_WEIGHTINGS:
        expected = eps_loss if weighting == "unweighted" else x0_loss
        torch.testing.assert_close(transition_score_loss(eps_pred, eps, schedule, t_start, t, weighting), expected)


def test_unit_noise_residual_is_the_x0_loss_against_the_transition_target():
    """``t^4 / ((1-t)^2 v) R^2 = (D - Y*)^2`` with ``D = x - t v_theta``, so ``x0`` is x0 regression."""
    torch.manual_seed(0)
    t_start, t = torch.tensor(0.624, dtype=torch.float64), torch.tensor(0.75, dtype=torch.float64)
    _, v = _transition(t_start, t)
    x_t, eps, flow = (torch.randn(1000, dtype=torch.float64) for _ in range(3))

    residual = eps - v.sqrt() / t * (x_t + (1 - t) * flow)
    target = (x_t - t**2 / v.sqrt() * eps) / (1 - t)
    x0_loss = (x_t - t * flow - target) ** 2

    torch.testing.assert_close(transition_score_weight("x0", 1 - t, t, v) * residual**2, x0_loss)


@pytest.mark.parametrize("data_variance", [1.0, 0.4])
def test_weightings_match_dmd2s_gradient_noise_or_loss_floor_for_gaussian_data(data_variance):
    """At the Bayes denoiser, ``variance_matched`` reproduces DMD2's gradient noise and
    ``inverse_variance`` its loss floor; ``kappa`` is the target-variance inflation."""
    torch.manual_seed(0)
    n, t_start, t = 2_000_000, torch.tensor(0.624, dtype=torch.float64), torch.tensor(0.75, dtype=torch.float64)
    a, v = _transition(t_start, t)
    x0 = torch.randn(n, dtype=torch.float64) * data_variance**0.5
    x_start = (1 - t_start) * x0 + t_start * torch.randn(n, dtype=torch.float64)
    eps = torch.randn(n, dtype=torch.float64)
    x_t = a * x_start + v.sqrt() * eps
    bayes_denoiser = (1 - t) * data_variance / ((1 - t) ** 2 * data_variance + t**2) * x_t
    target = (x_t - t**2 / v.sqrt() * eps) / (1 - t)

    reference_noise = ((bayes_denoiser - x0) ** 2).mean()
    transition_noise = ((bayes_denoiser - target) ** 2).mean()
    kappa = t**2 / v * (1 + t_start**2 / ((1 - t_start) ** 2 * data_variance))
    assert abs(transition_noise / reference_noise / kappa - 1) < 0.01

    def x0_weight(weighting):  # the unit-noise weight W as a weight on (D - Y*)^2
        weight = transition_score_weight(weighting, 1 - t, t, v, data_variance=data_variance)
        return weight * (1 - t) ** 2 * v / t**4

    assert abs(x0_weight("variance_matched") ** 2 * transition_noise / reference_noise - 1) < 0.01
    assert abs(x0_weight("inverse_variance") * transition_noise / reference_noise - 1) < 0.01
    torch.testing.assert_close(x0_weight("conservative"), v / t**2)
    torch.testing.assert_close(x0_weight("x0"), torch.tensor(1.0, dtype=torch.float64))


# ---------------------------------------------------------------------------
# DMD2Model with step generations
# ---------------------------------------------------------------------------


def test_generation_and_sampling_take_the_same_ddim_step():
    torch.manual_seed(0)
    model = _build_model()
    grid = model.net.noise_scheduler.student_grid()
    t_student, t_next = grid[:-1].clone(), grid[1:].clone()
    x = torch.randn(4, 3, 8, 8)

    generation = model.gen_data_from_net(x, torch.arange(4))
    gen_data = generation.data
    metrics = generation.metrics

    flow = model.net(x, t_student, fwd_pred_type="flow").double()
    torch.testing.assert_close(gen_data, (x.double() + _expand(t_next - t_student, x) * flow).float())
    torch.testing.assert_close(metrics["avg_t_student"], t_student.mean())
    assert generation.next_state is generation.data
    torch.testing.assert_close(generation.x0, x.double() - _expand(t_student, x) * flow)
    for step in range(len(t_student)):
        advanced = model._advance_student_step(model.net, x[step : step + 1], step, grid)
        torch.testing.assert_close(advanced, gen_data[step : step + 1], rtol=1e-6, atol=1e-6)


def test_the_student_step_does_not_depend_on_the_nets_prediction_type():
    """A net predicting x0 takes the same DDIM step as one predicting the flow it implies."""
    x = torch.randn(4, 3, 8, 8)
    generations = []
    for net_pred_type in ("flow", "x0"):
        torch.manual_seed(0)
        model = _build_model(net=L(_TinyFlowNet)(net_pred_type=net_pred_type))
        generations.append(model.gen_data_from_net(x, torch.arange(4)).data)

    torch.testing.assert_close(generations[0], generations[1], rtol=1e-5, atol=1e-5)


def test_carried_step_requires_an_explicit_next_state():
    model = _build_model(backward_simulation="staggered")
    generation = model.gen_data_from_net(torch.randn(2, 3, 8, 8), torch.ones(2, dtype=torch.long))
    missing_next_state = attrs.evolve(generation, next_state=None)

    with pytest.raises(ValueError, match="has no next state"):
        model._training_outputs(missing_next_state, {}, trajectory_step=1)

    # The final step needs only a boundary marker, so an absent state is valid there.
    outputs = model._training_outputs(missing_next_state, {}, trajectory_step=len(WAN_GRID) - 2)
    assert outputs[CARRY_OUTPUT_KEY] == {"step_carried": len(WAN_GRID) - 1}


def test_sampling_chains_ddim_steps_down_to_clean_data():
    torch.manual_seed(0)
    model = _build_model()
    schedule = model.net.noise_scheduler
    grid = schedule.student_grid()
    noise = torch.randn(2, 3, 8, 8)

    sample = model.generator_fn(model.net, noise)

    x = schedule.latents(noise, t_init=grid[0])
    with torch.no_grad():
        for t, t_next in zip(grid[:-1], grid[1:]):
            flow = model.net(x, t.expand(2), fwd_pred_type="flow")
            x = (x.double() + (t_next - t) * flow.double()).float()
    torch.testing.assert_close(sample, x)


@pytest.mark.parametrize("sample_type", ["ddpm", 0.5, "sde"])
def test_generation_and_sampling_take_the_configured_stochastic_step(sample_type):
    torch.manual_seed(0)
    model = _build_model(student_sample_type=sample_type)
    schedule = model.net.noise_scheduler
    grid = schedule.student_grid()
    x = torch.randn(4, 3, 8, 8)
    flow = model.net(x, grid[:-1], fwd_pred_type="flow")

    torch.manual_seed(1)
    gen_data = model.gen_data_from_net(x, torch.arange(4)).data
    torch.manual_seed(1)
    expected = schedule.step(x, flow, grid[:-1], grid[1:], pred_type="flow", sample_type=sample_type)
    torch.testing.assert_close(gen_data, expected)

    noise = torch.randn(2, 3, 8, 8)
    torch.manual_seed(2)
    sample = model.generator_fn(model.net, noise)
    torch.manual_seed(2)
    x = schedule.latents(noise, t_init=grid[0])
    with torch.no_grad():
        for t, t_next in zip(grid[:-1], grid[1:]):
            flow = model.net(x, t.expand(2), fwd_pred_type="flow")
            x = schedule.step(x, flow, t.expand(2), t_next, pred_type="flow", sample_type=sample_type)
    torch.testing.assert_close(sample, x)


def test_carried_trajectories_continue_from_the_generation(monkeypatch):
    torch.manual_seed(0)
    model = _build_model(backward_simulation="staggered", student_update_freq=5)
    batch_size = 2
    data = {
        "condition": torch.zeros(batch_size, 10),
        "neg_condition": torch.zeros(batch_size, 10),
        "condition_raw": [f"prompt {i}" for i in range(batch_size)],
        "idx": [f"sample_{i}" for i in range(batch_size)],
        "fname": [f"sample_{i}.mp4" for i in range(batch_size)],
        "shard": ["shard-0"] * batch_size,
        "state_carried": torch.randn(batch_size, 3, 8, 8),
        "step_carried": 1,
    }
    generations = []
    gen_data_from_net = model.gen_data_from_net

    def record_generation(*args, **kwargs):
        generation = gen_data_from_net(*args, **kwargs)
        generations.append((generation.data, cast_like(generation.x0, generation.data)))
        return generation

    monkeypatch.setattr(model, "gen_data_from_net", record_generation)

    _, outputs = model.single_train_step(data, iteration=model.config.student_update_freq)

    [(gen_data, x0_approx)] = generations
    carry = outputs[CARRY_OUTPUT_KEY]
    assert carry["step_carried"] == 2
    assert not carry["state_carried"].requires_grad
    torch.testing.assert_close(carry["state_carried"], gen_data.detach(), rtol=0, atol=0)
    torch.testing.assert_close(outputs["gen_rand_train"], x0_approx.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("shift", [1.0, 3.0])
def test_the_last_step_is_bucketed_dmd2(shift):
    """Clean generations: the transition is the forward process, and the weightings are DMD2's x0 loss
    (its eps loss for ``unweighted``)."""
    torch.manual_seed(0)
    model = _build_model(net=L(_TinyFlowNet)(schedule_type="shifted_rf", shift=shift))
    schedule = model.net.noise_scheduler
    n = 16
    student_step = torch.full((n,), 3)
    t = torch.rand(n, dtype=torch.float64) * 0.6 + 0.01
    gen_data, eps, eps_pred = (torch.randn(n, 3, 8, 8) for _ in range(3))

    t_gen_data = model._generation_time(student_step)
    x_t = schedule.forward_process(gen_data, eps, t, t_start=t_gen_data)
    torch.testing.assert_close(x_t, schedule.forward_process(gen_data, eps, t))

    x0_pred = schedule.convert_model_output(x_t, eps_pred, t, src_pred_type="eps", target_pred_type="x0")
    x0_loss = flow_matching_loss(x0_pred, "x0", gen_data, eps, schedule, t)
    eps_loss = flow_matching_loss(eps_pred, "eps", gen_data, eps, schedule, t)
    for weighting in FAKE_SCORE_WEIGHTINGS:
        model.config.fake_score_weighting = weighting
        loss_map = model._fake_score_loss_map(eps_pred, "eps", gen_data, eps, t, t_gen_data)
        expected = eps_loss if weighting == "unweighted" else x0_loss
        torch.testing.assert_close(loss_map["fake_score_loss"], expected, rtol=1e-4, atol=1e-6)
        # DMD2's clean generations have no generation time and take the same loss.
        clean_loss_map = model._fake_score_loss_map(eps_pred, "eps", gen_data, eps, t, None)
        torch.testing.assert_close(clean_loss_map["fake_score_loss"], loss_map["fake_score_loss"])


def test_generations_are_renoised_by_the_forward_transition_from_their_grid_time():
    torch.manual_seed(0)
    model = _build_model()
    grid = model.net.noise_scheduler.student_grid()
    t = torch.tensor([0.84, 0.93], dtype=torch.float64)
    gen_data, eps = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)

    t_gen_data = model._generation_time(torch.full((2,), 1))
    torch.testing.assert_close(t_gen_data, grid[2].expand(2))
    x_t = model.net.noise_scheduler.forward_process(gen_data, eps, t, t_start=t_gen_data)

    a, v = _transition(grid[2], t)
    torch.testing.assert_close(x_t, (_expand(a, gen_data) * gen_data + _expand(v.sqrt(), eps) * eps).float())


def test_the_vsd_reaching_x0_is_bucketed_dmd2s_under_sde():
    """For gen_data = alpha(t_gen_data) x0 + sigma(t_gen_data) eps_gen, the VSD on gen_data / alpha(t_gen_data)
    normalized by x0 is DMD2's VSD on x0."""
    torch.manual_seed(0)
    model = _build_model(student_sample_type="sde")
    grid = model.net.noise_scheduler.student_grid()
    student_step = torch.arange(4)
    t_gen_data = grid[student_step + 1]
    x0 = torch.randn(4, 3, 8, 8, requires_grad=True)
    eps_gen, teacher_x0, fake_score_x0 = (torch.randn(4, 3, 8, 8) for _ in range(3))
    gen_data = (_expand(1 - t_gen_data, x0) * x0 + _expand(t_gen_data, eps_gen) * eps_gen).float()

    loss_map, loss = model._compute_distribution_matching_loss(
        model._vsd_input(gen_data, model._generation_time(student_step)),
        teacher_x0,
        fake_score_x0,
        None,
        grid[student_step],
        gen_x0=x0,
    )
    (gradient,) = torch.autograd.grad(loss, x0)

    reference_x0 = x0.detach().clone().requires_grad_()
    reference = variational_score_distillation_loss(reference_x0, teacher_x0, fake_score_x0)
    (reference_gradient,) = torch.autograd.grad(reference, reference_x0)
    torch.testing.assert_close(loss_map["vsd_loss"], reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gradient, reference_gradient)


@pytest.mark.parametrize("weighting", FAKE_SCORE_WEIGHTINGS)
def test_student_and_auxiliary_steps_train_on_bucketed_transitions(weighting, monkeypatch):
    torch.manual_seed(0)
    model = _build_model(fake_score_weighting=weighting)
    data = {"idx": list(range(32)), "real": torch.randn(32, 3, 8, 8)}
    grid = model.net.noise_scheduler.student_grid()
    renoisings = record_generation_renoising(model, monkeypatch)

    loss_maps = []
    for iteration in (0, 1):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)
        assert torch.isfinite(loss_map["total_loss"])
        loss_maps.append(loss_map)

    assert "vsd_loss" in loss_maps[0] and "fake_score_loss" in loss_maps[1]
    assert len(renoisings) == 2
    for t, t_gen_data, student_step in renoisings:
        torch.testing.assert_close(t_gen_data, grid[student_step + 1])
        assert (t >= t_gen_data).all() and (t <= grid[student_step]).all()


def test_without_the_bucket_generations_are_renoised_from_their_grid_time_up_to_max_t(monkeypatch):
    torch.manual_seed(0)
    model = _build_model(sample_t_in_student_bucket=False, fake_score_weighting="x0")
    data = {"idx": list(range(32)), "real": torch.randn(32, 3, 8, 8)}
    grid = model.net.noise_scheduler.student_grid()
    min_t, max_t = model.config.sample_t_cfg.min_t, model.config.sample_t_cfg.max_t
    renoisings = record_generation_renoising(model, monkeypatch)

    for iteration in (0, 1):
        loss_map, _ = model.single_train_step(data, iteration)
        assert torch.isfinite(loss_map["total_loss"])

    assert len(renoisings) == 2
    for t, t_gen_data, student_step in renoisings:
        torch.testing.assert_close(t_gen_data, grid[student_step + 1])
        assert (t >= t_gen_data.clamp(min=min_t)).all() and (t <= max_t).all()
        assert (t > grid[student_step]).any()


def test_configuration_requirements_are_enforced():
    with pytest.raises(ValueError, match="only fake_score_weighting='x0'"):
        _build_model(sample_t_in_student_bucket=False)
    _build_model(sample_t_in_student_bucket=False, vsd_loss_weight=0.0)
    model = _build_model(fake_score_weighting="nope")
    with pytest.raises(ValueError, match="Unknown transition score weighting"):
        model.single_train_step({"idx": list(range(8)), "real": torch.randn(8, 3, 8, 8)}, 1)
    with pytest.raises(ValueError, match="fake_score_weighting='native'"):
        _build_model(fake_score_weighting="native")
    with pytest.raises(ValueError, match="fake_score_pred_type='eps'"):
        _build_model(fake_score_pred_type="flow")
    with pytest.raises(ValueError, match="supports student_generation"):
        _build_model(student_generation="noise")


def test_transition_weightings_on_x0_generations_warn_that_they_are_native_losses(monkeypatch):
    warnings = []
    monkeypatch.setattr(dmd2_module.logger, "warning", warnings.append)

    def weighting_warnings(**overrides):
        warnings.clear()
        _build_model(**overrides)
        return [message for message in warnings if "fake_score_weighting" in message]

    [message] = weighting_warnings(student_generation="x0", fake_score_weighting="variance_matched")
    assert "fake_score_pred_type='x0'" in message and "sigma_t^2 / alpha_t^2" in message
    [message] = weighting_warnings(student_generation="x0", fake_score_weighting="unweighted")
    assert "fake_score_pred_type='eps'" in message and "sigma_t^2 / alpha_t^2" not in message
    assert not weighting_warnings()
    assert not weighting_warnings(student_generation="x0", fake_score_weighting="native", fake_score_pred_type="x0")


@pytest.mark.parametrize("sample_type", ["ode", "ddpm"])
@pytest.mark.parametrize(
    "schedule_kwargs, overrides",
    [
        pytest.param({"schedule_type": "shifted_rf", "shift": 3.0}, {}, id="rf-shift3"),
        pytest.param({"schedule_type": "trig"}, {}, id="trig"),
        pytest.param(
            {"schedule_type": "edm"},
            {
                "student_grid_cfg": StudentGridConfig(
                    steps=4, grid_type="explicit", grid_kwargs={"timesteps": [80.0, 10.0, 1.0, 0.1, 0.0]}
                ),
                "sample_t_cfg": SampleTConfig(time_dist_type="uniform", min_t=0.002, max_t=80.0),
            },
            id="edm",
        ),
    ],
)
def test_training_and_sampling_run_on_any_noise_schedule(schedule_kwargs, overrides, sample_type):
    """EDM's grid ends at the clean ``0`` below its ``min_t``."""
    torch.manual_seed(0)
    model = _build_model(net=L(_TinyFlowNet)(**schedule_kwargs), student_sample_type=sample_type, **overrides)
    data = {"idx": list(range(8)), "real": torch.randn(8, 3, 8, 8)}

    for iteration in (0, 1):
        loss_map, _ = model.single_train_step(data, iteration)
        assert torch.isfinite(loss_map["total_loss"])
    assert torch.isfinite(model.generator_fn(model.net, torch.randn(2, 3, 8, 8))).all()


def test_outputs_and_rewards_use_the_x0_the_student_prediction_implies(monkeypatch):
    torch.manual_seed(0)
    model = _build_model(optimize_reward=True)
    scored = []
    model.reward = lambda gen_data, data, condition_dropout_mask=None: scored.append(gen_data) or {
        "total_loss": gen_data.square().mean()
    }
    generations = []
    gen_data_from_net = model.gen_data_from_net

    def record_generation(input_student, student_step, **kwargs):
        generations.append((input_student, student_step))
        return gen_data_from_net(input_student, student_step, **kwargs)

    monkeypatch.setattr(model, "gen_data_from_net", record_generation)
    data = {"idx": list(range(4)), "real": torch.randn(4, 3, 8, 8)}

    loss_map, student_outputs = model.single_train_step(data, 0)
    _, auxiliary_outputs = model.single_train_step(data, 1)

    assert len(scored) == 1 and scored[0].requires_grad and "reward_loss" in loss_map
    torch.testing.assert_close(scored[0].detach(), student_outputs["gen_rand_train"])
    for (input_student, student_step), outputs in zip(generations, (student_outputs, auxiliary_outputs), strict=True):
        t_student = model._student_grid(student_step.device)[student_step]
        torch.testing.assert_close(
            outputs["gen_rand_train"], model.net(input_student, t_student, fwd_pred_type="x0").detach()
        )


def test_transition_weights_stay_finite_at_pure_noise_and_zero_length_transitions():
    """The divisors are floored like the schedule's own: alpha = 0 at RF's pure-noise end (max_t = 1), a clean
    generation there (v = sigma^2), and a zero-length transition (v = 0) all give finite weights and losses."""
    from fastgen.networks.noise_schedule import get_noise_schedule

    torch.manual_seed(0)
    schedule = get_noise_schedule("rf", max_t=1.0)
    t = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64)
    t_start = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64)
    eps_pred, eps = torch.randn(3, 2, 4, 4), torch.randn(3, 2, 4, 4)

    for weighting in ("variance_matched", "inverse_variance", "conservative", "x0", "unweighted"):
        loss = transition_score_loss(eps_pred, eps, schedule, t_start, t, weighting)
        assert torch.isfinite(loss), weighting
        alpha, sigma = schedule.alpha(t), schedule.sigma(t)
        variance = schedule.transition_std(t, t_start).square().clamp(min=schedule.clamp_min)
        weight = transition_score_weight(weighting, alpha, sigma, variance, clamp_min=schedule.clamp_min)
        assert torch.isfinite(weight).all() and (weight > 0).all(), weighting
    # Without the floor, pure noise divides by alpha = 0.
    assert not torch.isfinite(
        transition_score_weight("conservative", schedule.alpha(t), schedule.sigma(t), variance)
    ).all()


def test_transition_score_loss_dtype_ignores_the_time_precision(monkeypatch):
    from fastgen.networks.noise_schedule import get_noise_schedule

    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)
    schedule = get_noise_schedule("rf", max_t=0.999)
    t = torch.tensor([0.75, 0.5], dtype=torch.float64)
    t_start = torch.tensor([0.25, 0.0], dtype=torch.float64)
    eps_pred, eps = torch.zeros(2, 3, dtype=torch.bfloat16), torch.ones(2, 3, dtype=torch.bfloat16)

    assert transition_score_loss(eps_pred, eps, schedule, t_start, t, "x0").dtype == torch.float32


def test_loss_scale_is_applied_per_sample_in_its_wider_precision(monkeypatch):
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)
    prediction, target = torch.ones(2, 3, dtype=torch.bfloat16), torch.zeros(2, 3, dtype=torch.bfloat16)
    # A scale of 1 + 1e-9 rounds to 1 in the float32 high precision but not in a float64 time precision.
    loss_scale = torch.tensor([1 + 1e-9, 1.0], dtype=torch.float64)

    loss = flow_matching_loss(prediction, "x0", x0=target, loss_scale=loss_scale)

    torch.testing.assert_close(loss, torch.tensor(1 + 0.5e-9, dtype=torch.float64), rtol=0, atol=1e-12)


def test_bucket_lower_bounds_keep_min_t_above_the_generation_time():
    """``sample_t_cfg.min_t`` is the sampled time's distance above the generation's grid time in every bucket, as it
    is above the clean data for x0 generations, so no transition has (near) zero length. The upper end is the bucket's
    grid time, or the schedule's ``max_t`` without buckets, which ``sample_t`` clips to ``sample_t_cfg.max_t``."""
    model = _build_model()
    grid = model.net.noise_scheduler.student_grid()
    student_step = torch.arange(4)

    lower, upper = model._sample_t_window(student_step, 4, min_t=0.001)

    torch.testing.assert_close(lower, grid[student_step + 1] + 0.001)
    torch.testing.assert_close(upper, grid[student_step])
    model = _build_model(sample_t_in_student_bucket=False, fake_score_weighting="x0")
    lower, upper = model._sample_t_window(student_step, 4, min_t=0.001)
    torch.testing.assert_close(lower, grid[student_step + 1] + 0.001)
    assert upper == model.net.noise_scheduler.max_t


@pytest.mark.parametrize("max_t", [0.999, 0.98], ids=["max-t-at-grid-top", "max-t-below-grid-top"])
@pytest.mark.parametrize("in_bucket", [True, False], ids=["bucket", "up-to-max-t"])
def test_noising_windows_condition_the_sampling_distribution(in_bucket, max_t):
    """Each window, clipped to ``sample_t_cfg.max_t``, draws ``sample_t_cfg``'s distribution conditioned on it. For the
    shift, that is a shifted draw over the window with the shift ``D(start) / D(end)``,
    ``D(y) = shift - (shift - 1) y``, so a share ``1 / (1 + D(start) / D(end))`` of the draws lands in its lower half,
    where the whole shift over every window would put ``1 / (1 + shift)`` there."""
    model = _build_model(
        sample_t_in_student_bucket=in_bucket,
        fake_score_weighting="variance_matched" if in_bucket else "x0",
        sample_t_cfg=SampleTConfig(time_dist_type="shifted", shift=5.0, min_t=0.001, max_t=max_t),
    )
    grid = model.net.noise_scheduler.student_grid().tolist()
    n = 40_000
    torch.manual_seed(0)
    for step in range(4):
        t, _ = model._sample_t_and_eps(n, student_step=torch.full((n,), step, dtype=torch.long))
        start, end = grid[step + 1] + 0.001, min(grid[step], max_t) if in_bucket else max_t
        y_start, y_end = ((x - 0.001) / (max_t - 0.001) for x in (start, end))
        window_shift = (5 - 4 * y_start) / (5 - 4 * y_end)

        assert ((t >= start) & (t <= end)).all()
        lower_half = (t < (start + end) / 2).double().mean().item()
        assert abs(lower_half - 1 / (1 + window_shift)) < 0.01, step
