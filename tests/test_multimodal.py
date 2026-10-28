# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for independent multimodal method and PDD time-warp behavior."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import fastgen.methods.distribution_matching.dmd2 as dmd2_module
import fastgen.methods.distribution_matching.f_distill as f_distill_module
import fastgen.methods.fine_tuning.sft as sft_module
import fastgen.methods.model as model_module
import fastgen.methods.trajectory_matching.pdd as pdd_module
from fastgen.configs.config import SampleTConfig
from fastgen.configs.methods.config_dmd2 import ModelConfig as DMD2Config
from fastgen.configs.methods.config_f_distill import ModelConfig as FDistillConfig
from fastgen.configs.methods.config_pdd import ModelConfig as PDDConfig
from fastgen.configs.methods.config_sft import ModelConfig as SFTConfig
from fastgen.methods.distribution_matching.dmd2 import DMD2Model
from fastgen.methods.distribution_matching.f_distill import FdistillModel
from fastgen.methods.fine_tuning.sft import SFTModel
from fastgen.methods.trajectory_matching.pdd import PDDModel
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import (
    CompositeNoiseSchedule,
    EDMNoiseSchedule,
    RFNoiseSchedule,
    ShiftedRFNoiseSchedule,
    StudentGridConfig,
    TrigNoiseSchedule,
    get_noise_schedule,
)
from fastgen.networks.utils.pdd_mixin import PDDLinear, PDDMixin, PDDSpec, make_pdd
from fastgen.trainer import Trainer
from fastgen.utils.tensor import expand_like, high_precision_dtype
from fastgen.utils.tree_utils import get_batch_size, to as _tree_to
from fastgen.utils.guidance import CFGTerm, GuidanceConfig, predict_with_guidance


_MODALITIES = ("image", "audio", "depth")
_EXPLICIT_GRID = (0.9, 0.6, 0.25, 0.0)


def _explicit_grid(steps: int = 3) -> StudentGridConfig:
    return StudentGridConfig(
        steps=steps,
        grid_type="explicit",
        grid_kwargs={"timesteps": _EXPLICIT_GRID},
    )


def _native_grid(base_steps: int, steps: int, shift: float) -> StudentGridConfig:
    return StudentGridConfig(
        steps=steps,
        grid_type="native",
        grid_kwargs={"base_steps": base_steps, "shift": shift},
    )


def _sampling_config(grid_type: str, steps: int) -> StudentGridConfig:
    if grid_type == "explicit":
        return _explicit_grid(steps)
    assert grid_type == "native"
    return StudentGridConfig(
        steps=steps,
        grid_type="native",
        grid_kwargs={"base_steps": len(_EXPLICIT_GRID) - 1, "shift": 1.35},
    )


def _diverse_schedule_configs(grid_type: str = "explicit") -> dict[str, dict[str, Any]]:
    sampling = _sampling_config(grid_type, steps=3)
    common = {"min_t": 0.002, "max_t": 0.9, "student_grid_cfg": sampling}
    return {
        "image": {"name": "edm", **common},
        "audio": {"name": "shifted_rf", "shift": 2.0, "num_train_timesteps": 32, **common},
        "depth": {"name": "trig", **common},
    }


def _pdd_schedule_configs(grid_type: str = "explicit") -> dict[str, dict[str, Any]]:
    # Two inference steps over three base-grid PDD intervals forces the second
    # runtime student-grid block to fuse multiple heads (balanced indices [0, 1, 3]).
    sampling = _sampling_config(grid_type, steps=2)
    common = {"min_t": 0.0, "max_t": 0.9, "num_train_timesteps": 32, "student_grid_cfg": sampling}
    return {
        "image": {"name": "shifted_rf", "shift": 0.25, **common},
        "audio": {"name": "rf", **common},
        "depth": {"name": "shifted_rf", "shift": 4.0, **common},
    }


def _make_schedule(configs: dict[str, dict[str, Any]], modality: str | None):
    if modality is None:
        return CompositeNoiseSchedule(deepcopy(configs), primary_key=_MODALITIES[0])
    config = deepcopy(configs[modality])
    return get_noise_schedule(config.pop("name"), **config)


def _deterministic_noise_like(value, dtype=None, **kwargs):
    del kwargs
    if isinstance(value, dict):
        return {key: _deterministic_noise_like(tensor, dtype=dtype) for key, tensor in value.items()}
    result = torch.linspace(-0.35, 0.45, value.numel(), device=value.device, dtype=torch.float64).reshape(value.shape)
    return result.to(dtype=dtype or value.dtype)


def _deterministic_noise_from_shape(shape, batch_size, device, dtype):
    if isinstance(shape, dict):
        return {key: _deterministic_noise_from_shape(value, batch_size, device, dtype) for key, value in shape.items()}
    value = torch.empty((batch_size, *shape), device=device, dtype=dtype)
    return _deterministic_noise_like(value)


def _guidance_config(modality: str | None) -> GuidanceConfig:
    scales = dict(zip(_MODALITIES, (1.5, 2.0, 2.75)))
    return GuidanceConfig(cfg=CFGTerm(scale=scales if modality is None else scales[modality]))


def _condition_term(condition: Any, value: torch.Tensor) -> torch.Tensor:
    if condition is None:
        return torch.zeros_like(value)
    assert isinstance(condition, torch.Tensor)
    condition = condition.to(device=value.device, dtype=value.dtype)
    while condition.ndim < value.ndim:
        condition = condition.unsqueeze(-1)
    return condition.expand_as(value)


class _IndependentX0Network(FastGenNetwork):
    """Tiny network whose modality branches share no parameters or computation."""

    def __init__(
        self,
        schedule_configs: dict[str, dict[str, Any]],
        modality: str | None = None,
        role_offset: float = 0.0,
    ):
        torch.nn.Module.__init__(self)
        self.net_pred_type = "x0"
        self.schedule_type = "composite" if modality is None else schedule_configs[modality]["name"]
        self.modalities = _MODALITIES if modality is None else (modality,)
        self.noise_scheduler = _make_schedule(schedule_configs, modality)
        self.heads = torch.nn.ModuleDict({key: torch.nn.Linear(2, 2, dtype=torch.float64) for key in self.modalities})
        self.time_scale = {key: 0.03 * (index + 1) for index, key in enumerate(_MODALITIES)}

        with torch.no_grad():
            for index, key in enumerate(self.modalities):
                base = index + 1 if modality is None else _MODALITIES.index(key) + 1
                self.heads[key].weight.copy_(
                    torch.tensor(
                        [[0.11 * base + role_offset, -0.07], [0.05, -0.09 * base + role_offset]],
                        dtype=torch.float64,
                    )
                )
                self.heads[key].bias.copy_(torch.tensor([0.02 * base + role_offset, -0.04 * base], dtype=torch.float64))

    def _forward_one(
        self,
        key: str,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any,
        fwd_pred_type: str,
    ) -> torch.Tensor:
        schedule = (
            self.noise_scheduler.subschedules[key]
            if isinstance(self.noise_scheduler, CompositeNoiseSchedule)
            else self.noise_scheduler
        )
        model_t = schedule.rescale_t(t).to(dtype=x_t.dtype)
        x0 = (
            self.heads[key](x_t)
            + self.time_scale[key] * expand_like(model_t, x_t)
            + 0.04 * _condition_term(condition, x_t)
        )
        return schedule.convert_model_output(
            x_t,
            x0,
            t,
            src_pred_type=self.net_pred_type,
            target_pred_type=fwd_pred_type,
        )

    def forward(
        self,
        x_t,
        t,
        condition=None,
        fwd_pred_type=None,
        return_features_early=False,
        feature_indices=None,
        **kwargs,
    ):
        del kwargs
        target = self.net_pred_type if fwd_pred_type is None else fwd_pred_type
        if isinstance(x_t, dict):
            result = {key: self._forward_one(key, value, t, condition, target) for key, value in x_t.items()}
        else:
            key = self.modalities[0]
            result = self._forward_one(key, x_t, t, condition, target)

        if feature_indices is None:
            return result
        features = [] if len(feature_indices) == 0 else [result]
        return features if return_features_early else [result, features]

    def sample(
        self,
        noise,
        condition=None,
        neg_condition=None,
        sample_steps=3,
        sample_type="ode",
        guidance_cfg=None,
    ):
        student_grid = self.noise_scheduler.student_grid(sample_steps, device=torch.device("cpu"))
        state = self.noise_scheduler.latents(noise, t_init=student_grid[0])
        for step_index in range(sample_steps):
            t = student_grid[step_index].expand(get_batch_size(state))
            x0 = predict_with_guidance(
                self,
                state,
                t,
                guidance_cfg=guidance_cfg,
                condition=condition,
                neg_condition=neg_condition,
                fwd_pred_type="x0",
            )
            if step_index + 1 == sample_steps:
                state = x0
            else:
                state = self.noise_scheduler.step(
                    state,
                    x0,
                    student_grid[step_index],
                    student_grid[step_index + 1],
                    sample_type=sample_type,
                )
        return state


class _IndependentPDDBase(FastGenNetwork):
    """Flow-native affine PDD base with one completely separate head per modality."""

    def __init__(
        self,
        schedule_configs: dict[str, dict[str, Any]],
        modality: str | None = None,
        role_offset: float = 0.0,
        load_pretrained: bool = True,
    ):
        del load_pretrained
        torch.nn.Module.__init__(self)
        self.net_pred_type = "flow"
        self.schedule_type = "composite" if modality is None else schedule_configs[modality]["name"]
        self.modalities = _MODALITIES if modality is None else (modality,)
        self.noise_scheduler = _make_schedule(schedule_configs, modality)
        self.heads = torch.nn.ModuleDict({key: torch.nn.Linear(3, 2, dtype=torch.float64) for key in self.modalities})

        with torch.no_grad():
            for key in self.modalities:
                base = _MODALITIES.index(key) + 1
                self.heads[key].weight.copy_(
                    torch.tensor(
                        [
                            [0.08 * base + role_offset, -0.03, 0.015],
                            [-0.04, 0.06 * base + role_offset, -0.02],
                        ],
                        dtype=torch.float64,
                    )
                )
                self.heads[key].bias.copy_(
                    torch.tensor([0.01 * base + role_offset, -0.025 * base], dtype=torch.float64)
                )

    def _forward_one(self, key: str, x_t: torch.Tensor, t: torch.Tensor, condition: Any) -> torch.Tensor:
        schedule = (
            self.noise_scheduler.subschedules[key]
            if isinstance(self.noise_scheduler, CompositeNoiseSchedule)
            else self.noise_scheduler
        )
        model_t = (schedule.rescale_t(t) / schedule.num_train_timesteps).to(dtype=x_t.dtype)
        features = torch.cat([x_t, model_t[:, None]], dim=1)
        output = self.heads[key](features)
        condition_output = _condition_term(condition, x_t)
        if output.shape[1] != condition_output.shape[1]:
            condition_output = condition_output.repeat(1, output.shape[1] // condition_output.shape[1])
        return output + 0.025 * condition_output

    def forward(self, x_t, t, condition=None, fwd_pred_type=None, **kwargs):
        del kwargs
        assert fwd_pred_type in (None, "flow")
        if isinstance(x_t, dict):
            return {key: self._forward_one(key, value, t, condition) for key, value in x_t.items()}
        return self._forward_one(self.modalities[0], x_t, t, condition)


_IndependentPDDNetwork = make_pdd(
    _IndependentPDDBase,
    PDDSpec(
        final_linear={key: f"heads.{key}" for key in _MODALITIES},
        channel_layout="channel_major",
    ),
)


class _IndependentDiscriminator(torch.nn.Module):
    """One scalar logit head per modality, matching CompositeDiscriminator reduction semantics."""

    feature_indices = {0}

    def __init__(self, modality: str | None):
        super().__init__()
        modalities = _MODALITIES if modality is None else (modality,)
        self.modalities = modalities
        self.heads = torch.nn.ModuleDict({key: torch.nn.Linear(2, 1, dtype=torch.float64) for key in modalities})
        with torch.no_grad():
            for key in modalities:
                index = _MODALITIES.index(key) + 1
                self.heads[key].weight.copy_(torch.tensor([[0.08 * index, -0.055]], dtype=torch.float64))
                self.heads[key].bias.copy_(torch.tensor([0.015 * index], dtype=torch.float64))

    def forward(self, features):
        assert isinstance(features, list) and len(features) == 1
        feature = features[0]
        if isinstance(feature, dict):
            return torch.cat([self.heads[key](feature[key]) for key in self.modalities], dim=1)
        return self.heads[self.modalities[0]](feature)


class _IdentityGradScaler:
    @staticmethod
    def scale(loss):
        return loss

    @staticmethod
    def is_enabled():
        return False


class _NoOpTrainerCallbacks:
    @staticmethod
    def on_backward_begin(*args, **kwargs):
        del args, kwargs

    @staticmethod
    def on_optimizer_step_begin(*args, **kwargs):
        del args, kwargs


def _install_training_runtime(model, learning_rate: float = 2e-3) -> None:
    model.precision_amp = None
    model.grad_scaler = _IdentityGradScaler()
    model.autocast = nullcontext

    model.net_optimizer = torch.optim.SGD(model.net.parameters(), lr=learning_rate)
    model.net_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(model.net_optimizer, lambda _: 1.0)
    if hasattr(model, "fake_score"):
        model.fake_score_optimizer = torch.optim.SGD(model.fake_score.parameters(), lr=learning_rate)
        model.fake_score_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            model.fake_score_optimizer,
            lambda _: 1.0,
        )
    if hasattr(model, "discriminator"):
        model.discriminator_optimizer = torch.optim.SGD(model.discriminator.parameters(), lr=learning_rate)
        model.discriminator_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            model.discriminator_optimizer,
            lambda _: 1.0,
        )


def _trainer_harness() -> Trainer:
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(grad_accum_rounds=1, fsdp=False))
    trainer.callbacks = _NoOpTrainerCallbacks()
    return trainer


def _parameter_vector(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten().clone() for parameter in module.parameters()])


def _base_model_state(model, config, net) -> None:
    model.config = config
    model.device = torch.device("cpu")
    model.precision = torch.float64
    model.precision_amp_infer = None
    model.input_shape = {key: [2] for key in _MODALITIES} if len(net.modalities) > 1 else [2]
    model.use_ema = []
    model.net = net


def _build_method(
    method_name: str,
    modality: str | None,
    grid_type: str = "explicit",
    guided: bool = False,
):
    if method_name == "sft":
        config = SFTConfig()
        config.sample_t_cfg = SampleTConfig(time_dist_type="uniform", min_t=0.1, max_t=0.8)
        config.student_sample_type = "ode"
        config.cond_dropout_prob = None
        config.guidance_cfg = _guidance_config(modality) if guided else None
        net = _IndependentX0Network(_diverse_schedule_configs(grid_type), modality=modality)
        model = object.__new__(SFTModel)
        torch.nn.Module.__init__(model)
        _base_model_state(model, config, net)
        return model

    if method_name == "dmd2":
        config = DMD2Config()
        config.sample_t_cfg = SampleTConfig(time_dist_type="uniform", min_t=0.1, max_t=0.8)
        config.student_sample_type = "ode"
        config.student_update_freq = 2
        config.gan_loss_weight_gen = 0.0
        config.backward_simulation = None
        config.guidance_cfg = _guidance_config(modality) if guided else None
        configs = _diverse_schedule_configs(grid_type)
        model = object.__new__(DMD2Model)
        torch.nn.Module.__init__(model)
        _base_model_state(model, config, _IndependentX0Network(configs, modality=modality))
        model.teacher = _IndependentX0Network(configs, modality=modality, role_offset=0.08).requires_grad_(False)
        model.fake_score = _IndependentX0Network(configs, modality=modality, role_offset=-0.05)
        return model

    if method_name == "pdd":
        config = PDDConfig()
        config.backward_simulation = None
        config.loss_config.block_size_min = len(_EXPLICIT_GRID) - 1
        config.loss_config.block_size_max = len(_EXPLICIT_GRID) - 1
        config.loss_config.type_rk = "euler"
        config.guidance_cfg = _guidance_config(modality) if guided else None
        configs = _pdd_schedule_configs(grid_type)
        net = _IndependentPDDNetwork(
            pdd_steps=len(_EXPLICIT_GRID) - 1,
            schedule_configs=configs,
            modality=modality,
        )
        model = object.__new__(PDDModel)
        torch.nn.Module.__init__(model)
        _base_model_state(model, config, net)
        model.precision_decoding = torch.float64
        model.teacher = _IndependentPDDBase(configs, modality=modality, role_offset=0.05).requires_grad_(False)
        return model

    raise AssertionError(f"Unhandled method {method_name}")


def _build_fdistill_method(modality: str | None, grid_type: str):
    """Build F-distill with constant RKL weighting and the independent GAN fixture."""
    config = FDistillConfig()
    config.sample_t_cfg = SampleTConfig(time_dist_type="uniform", min_t=0.1, max_t=0.8)
    config.student_sample_type = "ode"
    config.student_update_freq = 2
    config.backward_simulation = None
    config.fake_score_pred_type = "x0"
    config.guidance_cfg = _guidance_config(modality)
    config.f_distill.f_div = "rkl"
    config.f_distill.ratio_normalization = False

    configs = _diverse_schedule_configs(grid_type)
    model = object.__new__(FdistillModel)
    torch.nn.Module.__init__(model)
    _base_model_state(model, config, _IndependentX0Network(configs, modality=modality))
    model.teacher = _IndependentX0Network(configs, modality=modality, role_offset=0.08).requires_grad_(False)
    model.fake_score = _IndependentX0Network(configs, modality=modality, role_offset=-0.05)
    model.f_div_weighting_function = f_distill_module.all_f_div_weighting_function["rkl"]
    _enable_gan(model, modality)
    return model


def _align_native_reference_clocks(method_name: str, joint, isolated) -> None:
    """Drive standalone diverse schedules with the composite primary's native clock."""
    if method_name == "pdd":
        # Every PDD leaf is RF, so its native shared-time grid is identical
        # despite the different internal RF schedule shifts.
        return
    native_grid = joint.net.noise_scheduler.student_grid(
        len(_EXPLICIT_GRID) - 1,
        device=torch.device("cpu"),
    )
    for reference in isolated.values():
        reference.net.noise_scheduler.student_grid_cfg = StudentGridConfig(
            steps=len(_EXPLICIT_GRID) - 1,
            grid_type="explicit",
            grid_kwargs={"timesteps": native_grid.tolist()},
        )


def _enable_gan(model, modality: str | None) -> None:
    model.config.gan_loss_weight_gen = 0.3
    model.config.gan_use_same_t_noise = True
    model.config.gan_r1_reg_weight = 0.2
    model.config.gan_r1_reg_alpha = 0.07
    model.discriminator = _IndependentDiscriminator(modality)


def _training_iterations(method_name: str) -> tuple[tuple[int, str, str], ...]:
    if method_name == "sft":
        return ((0, "dsm_loss", "net"),)
    if method_name == "dmd2":
        return ((0, "vsd_loss", "net"), (1, "fake_score_loss", "fake_score"))
    return ((0, "pdd_loss", "net"),)


def _real_data() -> dict[str, torch.Tensor]:
    return {
        "image": torch.tensor([[-0.8, 0.2], [0.4, 1.1]], dtype=torch.float64),
        "audio": torch.tensor([[0.3, -1.2], [0.7, -0.1]], dtype=torch.float64),
        "depth": torch.tensor([[1.4, 0.5], [-0.6, 0.9]], dtype=torch.float64),
    }


def _zero_all_grads(model) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _assert_head_parameters_match(joint, isolated, module_name: str) -> None:
    joint_heads = getattr(joint, module_name).heads
    for key, reference in isolated.items():
        reference_head = getattr(reference, module_name).heads[key]
        for name, parameter in joint_heads[key].named_parameters():
            _assert_exact(parameter, dict(reference_head.named_parameters())[name])


def _assert_all_grads_cleared(module: torch.nn.Module) -> None:
    assert all(parameter.grad is None for parameter in module.parameters())


def _condition_pair() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[0.8], [-0.35]], dtype=torch.float64),
        torch.tensor([[-0.45], [0.25]], dtype=torch.float64),
    )


def _method_batch(real):
    condition, neg_condition = _condition_pair()
    return {
        "idx": [17, 29],
        "real": real,
        "condition": condition,
        "neg_condition": neg_condition,
    }


def _assert_schedule_fixture(method_name: str, joint, isolated) -> None:
    schedule = joint.net.noise_scheduler
    assert isinstance(schedule, CompositeNoiseSchedule)
    assert tuple(schedule.subschedules) == _MODALITIES
    assert len({id(leaf) for leaf in schedule.subschedules.values()}) == len(_MODALITIES)

    if method_name == "pdd":
        expected_types = dict(zip(_MODALITIES, (ShiftedRFNoiseSchedule, RFNoiseSchedule, ShiftedRFNoiseSchedule)))
        expected_shifts = dict(zip(_MODALITIES, (0.25, 1.0, 4.0)))
        assert joint.net.noise_scheduler.student_grid_indices(
            target_steps=joint.net.noise_scheduler.default_student_steps
        ).tolist() == [0, 1, 3]
    else:
        expected_types = {
            "image": EDMNoiseSchedule,
            "audio": ShiftedRFNoiseSchedule,
            "depth": TrigNoiseSchedule,
        }
        expected_shifts = {"audio": 2.0}

    for key in _MODALITIES:
        joint_leaf = schedule.subschedules[key]
        isolated_leaf = isolated[key].net.noise_scheduler
        assert type(joint_leaf) is expected_types[key]
        assert type(isolated_leaf) is expected_types[key]
        assert joint_leaf is not isolated_leaf
        if key in expected_shifts:
            assert joint_leaf.shift == expected_shifts[key]
            assert isolated_leaf.shift == expected_shifts[key]


@pytest.mark.parametrize("method_name", ("sft", "dmd2", "pdd"), ids=("SFT", "DMD2", "PDD"))
@pytest.mark.parametrize(
    ("grid_type", "guided"),
    (("explicit", False), ("native", True)),
    ids=("fixed-grid", "native-grid-CFG"),
)
def test_multimodal_training_and_inference_match_isolated_modalities(
    method_name,
    grid_type,
    guided,
    monkeypatch,
):
    """The exposed multimodal methods are exactly separable for an independent network."""
    monkeypatch.setattr(sft_module, "randn_like", _deterministic_noise_like)
    monkeypatch.setattr(dmd2_module, "noise_from_shape", _deterministic_noise_from_shape)
    monkeypatch.setattr(model_module, "randn_like", _deterministic_noise_like)
    monkeypatch.setattr(pdd_module, "randn_like", _deterministic_noise_like)

    real = _real_data()
    supplied_noise = _deterministic_noise_like(real)
    condition = torch.tensor([[0.8], [-0.35]], dtype=torch.float64)
    neg_condition = torch.tensor([[-0.45], [0.25]], dtype=torch.float64)
    joint = _build_method(method_name, modality=None, grid_type=grid_type, guided=guided)
    isolated = {
        key: _build_method(method_name, modality=key, grid_type=grid_type, guided=guided) for key in _MODALITIES
    }
    _assert_schedule_fixture(method_name, joint, isolated)
    assert joint.net.noise_scheduler.student_grid_cfg.grid_type == grid_type
    assert all(reference.net.noise_scheduler.student_grid_cfg.grid_type == grid_type for reference in isolated.values())
    if grid_type == "native":
        _align_native_reference_clocks(method_name, joint, isolated)
    for reference in isolated.values():
        _assert_exact(
            joint.net.noise_scheduler.student_grid(),
            reference.net.noise_scheduler.student_grid(),
        )

    if guided:
        guided_net = joint.net if method_name == "sft" else joint.teacher
        state = joint.net.noise_scheduler.forward_process(
            real,
            supplied_noise,
            torch.tensor([0.4, 0.55], dtype=torch.float64),
        )
        t = torch.tensor([0.4, 0.55], dtype=torch.float64)
        target = "flow" if method_name == "pdd" else "x0"
        positive = guided_net(state, t, condition=condition, fwd_pred_type=target)
        guided_prediction = predict_with_guidance(
            guided_net,
            state,
            t,
            guidance_cfg=joint.config.guidance_cfg,
            condition=condition,
            neg_condition=neg_condition,
            fwd_pred_type=target,
        )
        assert all(torch.count_nonzero(guided_prediction[key] - positive[key]) > 0 for key in _MODALITIES)

    for iteration, loss_name, active_module_name in _training_iterations(method_name):
        _zero_all_grads(joint)
        torch.manual_seed(1200 + iteration)
        joint_losses, joint_outputs = joint.single_train_step(_method_batch(real), iteration)
        joint_losses["total_loss"].backward()

        isolated_total = torch.zeros((), dtype=joint_losses["total_loss"].dtype)
        for key in _MODALITIES:
            reference = isolated[key]
            _zero_all_grads(reference)
            torch.manual_seed(1200 + iteration)
            reference_losses, reference_outputs = reference.single_train_step(_method_batch(real[key]), iteration)
            reference_losses["total_loss"].backward()

            _assert_exact(joint_losses[f"{key}_{loss_name}"], reference_losses[loss_name])
            # Data-dependent PDD deliberately returns a dummy zero next-state;
            # its loss and active-head gradients below are the substantive
            # training behavior. SFT and DMD2 return their actual predictions.
            if method_name != "pdd":
                _assert_exact(joint_outputs["gen_rand_train"][key], reference_outputs["gen_rand_train"])
            isolated_total = isolated_total + reference_losses["total_loss"].detach()

            joint_active = getattr(joint, active_module_name).heads[key]
            reference_active = getattr(reference, active_module_name).heads[key]
            assert joint_active.weight.grad is not None
            assert joint_active.bias.grad is not None
            assert reference_active.weight.grad is not None
            assert reference_active.bias.grad is not None
            joint_grad = torch.cat([joint_active.weight.grad.flatten(), joint_active.bias.grad.flatten()])
            reference_grad = torch.cat([reference_active.weight.grad.flatten(), reference_active.bias.grad.flatten()])
            assert torch.isfinite(joint_grad).all()
            assert torch.isfinite(reference_grad).all()
            assert torch.count_nonzero(joint_grad) > 0
            assert torch.count_nonzero(reference_grad) > 0
            _assert_exact(joint_active.weight.grad, reference_active.weight.grad)
            _assert_exact(joint_active.bias.grad, reference_active.bias.grad)

        _assert_exact(joint_losses["total_loss"].detach(), isolated_total)

    inference_noise = {
        key: torch.tensor([[0.25, -0.75], [1.0, 0.5]], dtype=torch.float64) + index
        for index, key in enumerate(_MODALITIES)
    }
    joint_sample = joint.generator_fn(
        net=joint.net,
        noise=inference_noise,
        condition=condition,
        neg_condition=neg_condition,
    )
    for key in _MODALITIES:
        reference_sample = isolated[key].generator_fn(
            net=isolated[key].net,
            noise=inference_noise[key],
            condition=condition,
            neg_condition=neg_condition,
        )
        _assert_exact(joint_sample[key], reference_sample)


@pytest.mark.parametrize("method_name", ("sft", "dmd2", "pdd"), ids=("SFT", "DMD2", "PDD"))
def test_full_trainer_step_preserves_independent_modality_updates(method_name, monkeypatch):
    """The optimizer/scheduler/zero-grad wrapper preserves the core method contract."""
    monkeypatch.setattr(sft_module, "randn_like", _deterministic_noise_like)
    monkeypatch.setattr(dmd2_module, "noise_from_shape", _deterministic_noise_from_shape)
    monkeypatch.setattr(model_module, "randn_like", _deterministic_noise_like)
    monkeypatch.setattr(pdd_module, "randn_like", _deterministic_noise_like)

    joint = _build_method(method_name, modality=None, grid_type="explicit", guided=True)
    isolated = {key: _build_method(method_name, modality=key, grid_type="explicit", guided=True) for key in _MODALITIES}
    _install_training_runtime(joint)
    for reference in isolated.values():
        _install_training_runtime(reference)
    trainer = _trainer_harness()
    real = _real_data()
    joint_batch = _method_batch(real)

    for iteration, loss_name, active_name in _training_iterations(method_name):
        before = _parameter_vector(getattr(joint, active_name))
        torch.manual_seed(5800 + iteration)
        joint_losses, _ = trainer.train_step(
            joint,
            joint,
            joint_batch,
            iteration=iteration,
            grad_accum_iter=0,
        )
        assert not torch.equal(_parameter_vector(getattr(joint, active_name)), before)
        _assert_all_grads_cleared(getattr(joint, active_name))

        isolated_losses = {}
        for key, reference in isolated.items():
            batch = _method_batch(real[key])
            torch.manual_seed(5800 + iteration)
            losses, _ = trainer.train_step(
                reference,
                reference,
                batch,
                iteration=iteration,
                grad_accum_iter=0,
            )
            isolated_losses[key] = losses
            _assert_exact(joint_losses[f"{key}_{loss_name}"], losses[loss_name])
        _assert_head_parameters_match(joint, isolated, active_name)
        expected_total = sum(isolated_losses[key]["total_loss"] for key in _MODALITIES)
        _assert_exact(joint_losses["total_loss"], expected_total)


def test_dmd2_gan_reductions_and_full_train_step_iteration_routing(monkeypatch):
    """GAN means and modality-summed score losses keep their distinct, intentional reductions."""
    monkeypatch.setattr(sft_module, "randn_like", _deterministic_noise_like)
    monkeypatch.setattr(dmd2_module, "noise_from_shape", _deterministic_noise_from_shape)
    monkeypatch.setattr(model_module, "randn_like", _deterministic_noise_like)
    monkeypatch.setattr(torch, "randn_like", _deterministic_noise_like)

    joint = _build_method("dmd2", modality=None, grid_type="native", guided=True)
    isolated = {key: _build_method("dmd2", modality=key, grid_type="native", guided=True) for key in _MODALITIES}
    _align_native_reference_clocks("dmd2", joint, isolated)
    _enable_gan(joint, modality=None)
    for key, reference in isolated.items():
        _enable_gan(reference, modality=key)
    real = _real_data()
    batch = _method_batch(real)

    # Student/VSD iteration: score losses add over modalities, while the GAN
    # generator loss averages the concatenated per-modality logits.
    _zero_all_grads(joint)
    torch.manual_seed(6100)
    joint_student_losses, _ = joint.single_train_step(batch, iteration=0)
    joint_student_losses["total_loss"].backward()
    isolated_student_losses = {}
    for key, reference in isolated.items():
        _zero_all_grads(reference)
        torch.manual_seed(6100)
        losses, _ = reference.single_train_step(_method_batch(real[key]), iteration=0)
        (
            losses["vsd_loss"] + reference.config.gan_loss_weight_gen / len(_MODALITIES) * losses["gan_loss_gen"]
        ).backward()
        isolated_student_losses[key] = losses
        _assert_exact(joint_student_losses[f"{key}_vsd_loss"], losses["vsd_loss"])
        torch.testing.assert_close(
            joint.net.heads[key].weight.grad,
            reference.net.heads[key].weight.grad,
            rtol=1e-14,
            atol=1e-14,
        )
        torch.testing.assert_close(
            joint.net.heads[key].bias.grad,
            reference.net.heads[key].bias.grad,
            rtol=1e-14,
            atol=1e-14,
        )

    expected_vsd = sum(isolated_student_losses[key]["vsd_loss"] for key in _MODALITIES)
    expected_gan_gen = torch.stack([isolated_student_losses[key]["gan_loss_gen"] for key in _MODALITIES]).mean()
    torch.testing.assert_close(joint_student_losses["gan_loss_gen"], expected_gan_gen, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(
        joint_student_losses["total_loss"],
        expected_vsd + joint.config.gan_loss_weight_gen * expected_gan_gen,
        rtol=1e-15,
        atol=1e-15,
    )

    # Fake-score/discriminator iteration: discriminator and approximate-R1
    # losses use the same cross-modality mean; DSM remains a modality sum.
    _zero_all_grads(joint)
    torch.manual_seed(6101)
    joint_fake_losses, _ = joint.single_train_step(batch, iteration=1)
    joint_fake_losses["total_loss"].backward()
    isolated_fake_losses = {}
    for key, reference in isolated.items():
        _zero_all_grads(reference)
        torch.manual_seed(6101)
        losses, _ = reference.single_train_step(_method_batch(real[key]), iteration=1)
        losses["total_loss"].backward()
        isolated_fake_losses[key] = losses
        _assert_exact(joint_fake_losses[f"{key}_fake_score_loss"], losses["fake_score_loss"])
        _assert_exact(
            joint.fake_score.heads[key].weight.grad,
            reference.fake_score.heads[key].weight.grad,
        )
        _assert_exact(
            joint.fake_score.heads[key].bias.grad,
            reference.fake_score.heads[key].bias.grad,
        )
        torch.testing.assert_close(
            joint.discriminator.heads[key].weight.grad,
            reference.discriminator.heads[key].weight.grad / len(_MODALITIES),
            rtol=1e-14,
            atol=1e-14,
        )
        torch.testing.assert_close(
            joint.discriminator.heads[key].bias.grad,
            reference.discriminator.heads[key].bias.grad / len(_MODALITIES),
            rtol=1e-14,
            atol=1e-14,
        )

    expected_fake_score = sum(isolated_fake_losses[key]["fake_score_loss"] for key in _MODALITIES)
    expected_gan_disc = torch.stack([isolated_fake_losses[key]["gan_loss_disc"] for key in _MODALITIES]).mean()
    expected_r1 = torch.stack([isolated_fake_losses[key]["gan_loss_ar1"] for key in _MODALITIES]).mean()
    torch.testing.assert_close(joint_fake_losses["gan_loss_disc"], expected_gan_disc, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(joint_fake_losses["gan_loss_ar1"], expected_r1, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(
        joint_fake_losses["total_loss"],
        expected_fake_score + expected_gan_disc + joint.config.gan_r1_reg_weight * expected_r1,
        rtol=1e-15,
        atol=1e-15,
    )

    # The standard Trainer path must route four consecutive optimizer steps to
    # net, then fake-score+discriminator, and repeat the same two phases.
    trained = _build_method("dmd2", modality=None, grid_type="native", guided=True)
    _enable_gan(trained, modality=None)
    _install_training_runtime(trained)
    trainer = _trainer_harness()
    teacher_initial = _parameter_vector(trained.teacher)
    for iteration in range(4):
        before = {name: _parameter_vector(getattr(trained, name)) for name in ("net", "fake_score", "discriminator")}
        torch.manual_seed(7200 + iteration)
        losses, _ = trainer.train_step(
            trained,
            trained,
            batch,
            iteration,
            grad_accum_iter=0,
        )
        if iteration % 2 == 0:
            assert "gan_loss_gen" in losses and "gan_loss_disc" not in losses
            changed = {"net"}
        else:
            assert "gan_loss_disc" in losses and "gan_loss_ar1" in losses
            changed = {"fake_score", "discriminator"}
        for name in ("net", "fake_score", "discriminator"):
            after = _parameter_vector(getattr(trained, name))
            assert torch.equal(after, before[name]) is (name not in changed)
            if name in changed:
                _assert_all_grads_cleared(getattr(trained, name))
    _assert_exact(_parameter_vector(trained.teacher), teacher_initial)


@pytest.mark.parametrize("grid_type", ("explicit", "native"), ids=("fixed-grid", "native-grid"))
def test_fdistill_rkl_full_train_steps_and_inference_match_isolated_modalities(grid_type, monkeypatch):
    """Constant-h F-distill is separable apart from its intentional cross-modality GAN mean."""
    monkeypatch.setattr(dmd2_module, "noise_from_shape", _deterministic_noise_from_shape)
    monkeypatch.setattr(model_module, "randn_like", _deterministic_noise_like)
    monkeypatch.setattr(torch, "randn_like", _deterministic_noise_like)

    joint = _build_fdistill_method(modality=None, grid_type=grid_type)
    isolated = {key: _build_fdistill_method(modality=key, grid_type=grid_type) for key in _MODALITIES}
    _assert_schedule_fixture("fdistill", joint, isolated)
    if grid_type == "native":
        _align_native_reference_clocks("fdistill", joint, isolated)
    for reference in isolated.values():
        _assert_exact(
            joint.net.noise_scheduler.student_grid(),
            reference.net.noise_scheduler.student_grid(),
        )

    # RKL is a per-sample constant. This shape assertion guards against
    # accidentally returning a scalar weighting tensor for a batched loss.
    batch_t = torch.tensor([0.25, 0.65], dtype=torch.float64)
    fake_logits = torch.tensor([[0.2, -0.1, 0.4], [-0.5, 0.3, 0.7]], dtype=torch.float64)
    h = joint._get_f_div_weighting_h(fake_logits, batch_t)
    assert h.shape == batch_t.shape
    _assert_exact(h, h[:1].expand_as(h))

    real = _real_data()
    condition, neg_condition = _condition_pair()
    state = joint.net.noise_scheduler.forward_process(
        real,
        _deterministic_noise_like(real),
        torch.tensor([0.4, 0.55], dtype=torch.float64),
    )
    t = torch.tensor([0.4, 0.55], dtype=torch.float64)
    positive = joint.teacher(state, t, condition=condition, fwd_pred_type="x0")
    guided = predict_with_guidance(
        joint.teacher,
        state,
        t,
        guidance_cfg=joint.config.guidance_cfg,
        condition=condition,
        neg_condition=neg_condition,
        fwd_pred_type="x0",
    )
    assert all(torch.count_nonzero(guided[key] - positive[key]) > 0 for key in _MODALITIES)

    learning_rate = 2e-3
    num_modalities = len(_MODALITIES)
    joint_gan_weight = joint.config.gan_loss_weight_gen
    _install_training_runtime(joint, learning_rate=learning_rate)
    for reference in isolated.values():
        # The joint generator objective averages GAN logits across modalities,
        # while its F-distill losses add. Scale only the isolated GAN term to
        # reconstruct that exact joint objective.
        reference.config.gan_loss_weight_gen = joint_gan_weight / num_modalities
        _install_training_runtime(reference, learning_rate=learning_rate)
        # On the fake-score iteration the discriminator loss itself is a
        # cross-modality mean. Its isolated gradients therefore need a 1/M
        # optimizer-step scale; fake-score gradients remain modality sums.
        for group in reference.discriminator_optimizer.param_groups:
            group["lr"] = learning_rate / num_modalities

    trainer = _trainer_harness()
    teacher_initial = _parameter_vector(joint.teacher)
    losses_by_iteration = {}
    for iteration in (0, 1):
        before = {name: _parameter_vector(getattr(joint, name)) for name in ("net", "fake_score", "discriminator")}
        torch.manual_seed(8100 + iteration)
        joint_losses, joint_outputs = trainer.train_step(
            joint,
            joint,
            _method_batch(real),
            iteration,
            grad_accum_iter=0,
        )

        reference_losses = {}
        for key, reference in isolated.items():
            torch.manual_seed(8100 + iteration)
            losses, outputs = trainer.train_step(
                reference,
                reference,
                _method_batch(real[key]),
                iteration,
                grad_accum_iter=0,
            )
            reference_losses[key] = losses
            torch.testing.assert_close(
                joint_outputs["gen_rand_train"][key],
                outputs["gen_rand_train"],
                rtol=1e-13,
                atol=1e-14,
            )

        if iteration == 0:
            changed = {"net"}
            loss_name = "f_distill_loss"
            expected_score = sum(reference_losses[key][loss_name] for key in _MODALITIES)
            expected_gan = torch.stack([reference_losses[key]["gan_loss_gen"] for key in _MODALITIES]).mean()
            torch.testing.assert_close(joint_losses["gan_loss_gen"], expected_gan, rtol=1e-15, atol=1e-15)
            torch.testing.assert_close(
                joint_losses["total_loss"],
                expected_score + joint_gan_weight * expected_gan,
                rtol=1e-15,
                atol=1e-15,
            )
            torch.testing.assert_close(
                joint_losses["total_loss"],
                sum(reference_losses[key]["total_loss"] for key in _MODALITIES),
                rtol=1e-15,
                atol=1e-15,
            )
            for metric in ("min_h", "avg_h", "max_h"):
                _assert_exact(joint_losses[metric], torch.ones_like(joint_losses[metric]))
        else:
            changed = {"fake_score", "discriminator"}
            loss_name = "fake_score_loss"
            expected_score = sum(reference_losses[key][loss_name] for key in _MODALITIES)
            expected_gan = torch.stack([reference_losses[key]["gan_loss_disc"] for key in _MODALITIES]).mean()
            expected_r1 = torch.stack([reference_losses[key]["gan_loss_ar1"] for key in _MODALITIES]).mean()
            torch.testing.assert_close(joint_losses["gan_loss_disc"], expected_gan, rtol=1e-15, atol=1e-15)
            torch.testing.assert_close(joint_losses["gan_loss_ar1"], expected_r1, rtol=1e-15, atol=1e-15)
            torch.testing.assert_close(
                joint_losses["total_loss"],
                expected_score + expected_gan + joint.config.gan_r1_reg_weight * expected_r1,
                rtol=1e-15,
                atol=1e-15,
            )

        for key, reference in isolated.items():
            torch.testing.assert_close(
                joint_losses[f"{key}_{loss_name}"],
                reference_losses[key][loss_name],
                rtol=1e-13,
                atol=1e-14,
            )
        for name in ("net", "fake_score", "discriminator"):
            after = _parameter_vector(getattr(joint, name))
            assert torch.equal(after, before[name]) is (name not in changed)
            if name in changed:
                _assert_all_grads_cleared(getattr(joint, name))
                joint_heads = getattr(joint, name).heads
                for key, reference in isolated.items():
                    reference_head = getattr(reference, name).heads[key]
                    for parameter_name, parameter in joint_heads[key].named_parameters():
                        torch.testing.assert_close(
                            parameter,
                            dict(reference_head.named_parameters())[parameter_name],
                            rtol=1e-13,
                            atol=1e-14,
                        )
        losses_by_iteration[iteration] = joint_losses

    assert "image_f_distill_loss" in losses_by_iteration[0]
    assert "image_fake_score_loss" in losses_by_iteration[1]
    _assert_exact(_parameter_vector(joint.teacher), teacher_initial)

    inference_noise = {
        key: torch.tensor([[0.15, -0.65], [0.85, 0.45]], dtype=torch.float64) + index
        for index, key in enumerate(_MODALITIES)
    }
    joint_sample = joint.generator_fn(
        joint.net,
        inference_noise,
        condition=condition,
        neg_condition=neg_condition,
    )
    for key, reference in isolated.items():
        reference_sample = reference.generator_fn(
            reference.net,
            inference_noise[key],
            condition=condition,
            neg_condition=neg_condition,
        )
        torch.testing.assert_close(joint_sample[key], reference_sample, rtol=1e-13, atol=1e-14)


class _TimeRescaledRFBase(FastGenNetwork):
    """One-head H3 analogue: condition on warped time and emit ``sigma'(t) * (eps-x0)``."""

    def __init__(
        self,
        schedule_shift: float,
        student_grid_cfg: StudentGridConfig,
        load_pretrained: bool = True,
    ):
        del load_pretrained
        super().__init__(
            net_pred_type="flow",
            schedule_type="shifted_rf",
            min_t=0.0,
            max_t=0.999,
            num_train_timesteps=1000,
            shift=schedule_shift,
            student_grid_cfg=student_grid_cfg,
        )
        self.proj = torch.nn.Linear(2, 1, dtype=torch.float64)
        self.last_model_time = None
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[0.65, -0.3]], dtype=torch.float64))
            self.proj.bias.copy_(torch.tensor([0.12], dtype=torch.float64))

    def _to_flow_convention(self, native_output: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dtype = high_precision_dtype(native_output)
        scale = self.noise_scheduler.sigma_prime(t).to(dtype=dtype)
        return native_output.to(dtype=dtype) * expand_like(scale, native_output)

    def forward(self, x_t, t, condition=None, fwd_pred_type=None, **kwargs):
        del kwargs
        model_time = (self.noise_scheduler.rescale_t(t) / self.noise_scheduler.num_train_timesteps).to(dtype=x_t.dtype)
        self.last_model_time = model_time.detach().clone()
        # Real H3 applies conditioning before its final projection.  Keeping the
        # mock's condition dependence on that side of ``proj`` ensures the
        # folded per-head conversion to native sigma-velocity multiplies the
        # complete conditional and unconditional outputs, just as it does in H3.
        conditioned_x = x_t + 0.2 * _condition_term(condition, x_t)
        features = torch.cat([conditioned_x, model_time[:, None]], dim=1)
        native_output = self.proj(features)
        flow = self._to_flow_convention(native_output, t)
        return self.noise_scheduler.convert_model_output(
            x_t,
            flow,
            t,
            src_pred_type=self.net_pred_type,
            target_pred_type=self.net_pred_type if fwd_pred_type is None else fwd_pred_type,
        )


_TimeRescaledPDD = make_pdd(
    _TimeRescaledRFBase,
    PDDSpec(final_linear="proj", channel_layout="channel_major"),
)


class _TimeRescaledNativeHeadPDD(PDDMixin, _TimeRescaledRFBase):
    """The two H3 PDD adaptations, reproduced on the tiny time-conditioned network.

    The base's raw projection already emits ``eps - x0 = dx/dsigma``, so the
    heads need no teacher fold and the single-time output scaling is bypassed.
    """

    PDD_SPEC = PDDSpec(final_linear="proj", channel_layout="channel_major")

    def pdd_teacher_head_scale(self, key):
        return None

    @staticmethod
    def _to_flow_convention(native_output: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return native_output


def _predict_heads(model, net, state, step_index, condition=None):
    """Student training forward exactly as the PDD methods issue it."""
    return net(_tree_to(state, dtype=model.precision), step_index=step_index, condition=condition)


def _pdd_model_harness(
    net,
    teacher=None,
    rk_type: str = "euler",
    guidance_cfg: GuidanceConfig | None = None,
) -> PDDModel:
    model = object.__new__(PDDModel)
    torch.nn.Module.__init__(model)
    model.net = net
    model.config = SimpleNamespace(
        student_sample_type="ode",
        backward_simulation=None,
        guidance_cfg=guidance_cfg,
        loss_config=SimpleNamespace(
            block_size_max=net.pdd_steps,
            type_rk=rk_type,
            num_supervised_heads=1,
        ),
    )
    model.device = torch.device("cpu")
    model.precision = torch.float64
    model.precision_decoding = torch.float64
    model.precision_amp_infer = None
    model.use_ema = []
    if teacher is not None:
        model.teacher = teacher.requires_grad_(False)
    return model


def _build_teacher_pdd_target(model, x_t, step_idx, base_grid, condition=None, neg_condition=None):
    """Exercise the same scheduler-owned teacher-target boundary as PDD training."""
    with torch.no_grad():
        return model.net.pdd_teacher_target(
            x_t,
            step_idx,
            base_grid,
            solver=model.config.loss_config.type_rk,
            predict_teacher=partial(
                model._guided_teacher,
                condition=condition,
                neg_condition=neg_condition,
            ),
        )


def test_native_head_pdd_matches_unshifted_physical_clock_for_unfused_training_and_fused_inference():
    """H3's time warp and native sigma-velocity heads cancel in both PDD execution paths."""
    pdd_steps = 4
    outer_grid_shift = 12.0
    relative_schedule_shift = 0.25
    shifted = _TimeRescaledNativeHeadPDD(
        pdd_steps=pdd_steps,
        schedule_shift=relative_schedule_shift,
        student_grid_cfg=_native_grid(pdd_steps, steps=2, shift=outer_grid_shift),
    )
    shifted_grid = shifted.noise_scheduler.student_grid(pdd_steps, device=torch.device("cpu"))
    physical_grid = shifted.noise_scheduler.sigma(shifted_grid)
    reference = _TimeRescaledPDD(
        pdd_steps=pdd_steps,
        schedule_shift=1.0,
        student_grid_cfg=StudentGridConfig(
            steps=2,
            grid_type="explicit",
            grid_kwargs={"timesteps": physical_grid.tolist()},
        ),
    )
    reference_grid = reference.noise_scheduler.student_grid(pdd_steps, device=torch.device("cpu"))
    _assert_exact(reference_grid, physical_grid)

    clean = torch.tensor([[-0.7], [0.45]], dtype=torch.float64)
    noise = torch.tensor([[0.8], [-1.1]], dtype=torch.float64)
    step_index = torch.tensor([0, 1], dtype=torch.long)
    next_step_index = torch.tensor([2, 4], dtype=torch.long)
    reference_state = reference.noise_scheduler.forward_process(clean, noise, reference_grid[step_index])
    shifted_state = shifted.noise_scheduler.forward_process(clean, noise, shifted_grid[step_index])
    _assert_exact(reference_state, shifted_state)

    reference_model = _pdd_model_harness(reference)
    shifted_model = _pdd_model_harness(shifted)
    reference_state = reference_state.requires_grad_()
    shifted_state = shifted_state.detach().clone().requires_grad_()
    reference_heads = _predict_heads(
        reference_model,
        reference,
        reference_state,
        step_index=step_index,
        condition=None,
    )
    shifted_heads = _predict_heads(
        shifted_model,
        shifted,
        shifted_state,
        step_index=step_index,
        condition=None,
    )

    _assert_exact(reference.last_model_time, shifted.last_model_time)
    # Sigma-velocity heads and sigma-increment weights are physical-coordinate
    # quantities, so the time warp cancels exactly in both.
    torch.testing.assert_close(shifted_heads, reference_heads, rtol=1e-14, atol=1e-14)
    intervals = torch.arange(reference.pdd_steps)
    _, reference_weights = reference.pdd_block_weights(intervals, intervals + 1, reference_grid)
    _, shifted_weights = shifted.pdd_block_weights(intervals, intervals + 1, shifted_grid)
    torch.testing.assert_close(shifted_weights, reference_weights, rtol=1e-14, atol=1e-14)

    reference_endpoint = reference.pdd_advance_blocks(
        reference_state,
        reference_heads,
        step_index,
        next_step_index,
        reference_grid,
    )
    shifted_endpoint = shifted.pdd_advance_blocks(
        shifted_state,
        shifted_heads,
        step_index,
        next_step_index,
        shifted_grid,
    )
    torch.testing.assert_close(shifted_endpoint, reference_endpoint, rtol=1e-13, atol=1e-13)
    reference_x0 = reference.noise_scheduler.points_to_x0(
        reference_state,
        reference_endpoint,
        reference_grid[step_index],
        reference_grid[next_step_index],
    )
    shifted_x0 = shifted.noise_scheduler.points_to_x0(
        shifted_state,
        shifted_endpoint,
        shifted_grid[step_index],
        shifted_grid[next_step_index],
    )
    torch.testing.assert_close(shifted_x0, reference_x0, rtol=1e-13, atol=1e-13)
    assert torch.equal(shifted_x0[1], shifted_endpoint[1])

    # Backpropagate the same endpoint objective through the unfused training
    # representation. Sigma-velocity heads see identical inputs and carry
    # identical (unfolded) parameters, so state and head gradients all match.
    reference_training_loss = reference_endpoint.square().sum()
    shifted_training_loss = shifted_endpoint.square().sum()
    torch.testing.assert_close(shifted_training_loss, reference_training_loss, rtol=1e-13, atol=1e-13)
    reference_training_loss.backward()
    shifted_training_loss.backward()
    torch.testing.assert_close(shifted_state.grad, reference_state.grad, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        shifted.pdd_linear.weight.grad,
        reference.pdd_linear.weight.grad,
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(
        shifted.pdd_linear.bias.grad,
        reference.pdd_linear.bias.grad,
        rtol=1e-13,
        atol=1e-13,
    )

    inference_noise = torch.tensor([[0.35], [-0.9]], dtype=torch.float64)
    reference_sample = reference_model.generator_fn(reference, inference_noise)
    shifted_sample = shifted_model.generator_fn(shifted, inference_noise)
    torch.testing.assert_close(shifted_sample, reference_sample, rtol=1e-13, atol=1e-13)

    assert type(reference.pdd_linear) is PDDLinear
    assert type(shifted.pdd_linear) is PDDLinear


@pytest.mark.parametrize("rk_type", ("euler", "midpoint", "heun", "kutta4"))
def test_native_head_pdd_teacher_rk_and_loss_match_unshifted_physical_clock(rk_type):
    """Every primary-schedule RK stage, target endpoint, and PDD loss is time-warp equivariant."""
    pdd_steps = 4
    outer_grid_shift = 12.0
    relative_schedule_shift = 0.25
    shifted_sampling = _native_grid(pdd_steps, steps=2, shift=outer_grid_shift)
    shifted = _TimeRescaledNativeHeadPDD(
        pdd_steps=pdd_steps,
        schedule_shift=relative_schedule_shift,
        student_grid_cfg=shifted_sampling,
    )
    shifted_grid = shifted.noise_scheduler.student_grid(pdd_steps, device=torch.device("cpu"))
    physical_grid = shifted.noise_scheduler.sigma(shifted_grid)
    reference_sampling = StudentGridConfig(
        steps=2,
        grid_type="explicit",
        grid_kwargs={"timesteps": physical_grid.tolist()},
    )
    reference = _TimeRescaledPDD(
        pdd_steps=pdd_steps,
        schedule_shift=1.0,
        student_grid_cfg=reference_sampling,
    )
    reference_grid = reference.noise_scheduler.student_grid(pdd_steps, device=torch.device("cpu"))
    _assert_exact(reference_grid, physical_grid)

    # These ordinary (non-PDD) teachers represent the same vector field. The
    # shifted teacher consumes t, conditions on sigma(t), and emits
    # sigma'(t) times the unshifted teacher's schedule-time velocity.
    shifted_teacher = _TimeRescaledRFBase(
        schedule_shift=relative_schedule_shift,
        student_grid_cfg=shifted_sampling,
    )
    reference_teacher = _TimeRescaledRFBase(
        schedule_shift=1.0,
        student_grid_cfg=reference_sampling,
    )
    guidance_cfg = GuidanceConfig(cfg=CFGTerm(scale=2.25))
    shifted_model = _pdd_model_harness(
        shifted,
        teacher=shifted_teacher,
        rk_type=rk_type,
        guidance_cfg=guidance_cfg,
    )
    reference_model = _pdd_model_harness(
        reference,
        teacher=reference_teacher,
        rk_type=rk_type,
        guidance_cfg=guidance_cfg,
    )

    clean = torch.tensor([[-0.7], [0.45]], dtype=torch.float64)
    noise = torch.tensor([[0.8], [-1.1]], dtype=torch.float64)
    # Exercise distinct batched intervals, including the clean-endpoint interval.
    step_index = torch.tensor([1, pdd_steps - 1], dtype=torch.long)
    block_end_index = step_index + 1
    condition = torch.tensor([[0.4], [-0.75]], dtype=torch.float64)
    neg_condition = torch.tensor([[-0.2], [0.6]], dtype=torch.float64)
    shifted_t = shifted_grid[step_index]
    reference_t = reference_grid[step_index]
    shifted_state = shifted.noise_scheduler.forward_process(clean, noise, shifted_t)
    reference_state = reference.noise_scheduler.forward_process(clean, noise, reference_t)
    _assert_exact(shifted_state, reference_state)

    _, shifted_weight = shifted.pdd_block_weights(step_index, block_end_index, shifted_grid)
    _, reference_weight = reference.pdd_block_weights(step_index, block_end_index, reference_grid)
    torch.testing.assert_close(shifted_weight, reference_weight, rtol=1e-14, atol=1e-14)

    shifted_target = _build_teacher_pdd_target(
        shifted_model,
        shifted_state,
        step_index,
        shifted_grid,
        condition=condition,
        neg_condition=neg_condition,
    )
    reference_target = _build_teacher_pdd_target(
        reference_model,
        reference_state,
        step_index,
        reference_grid,
        condition=condition,
        neg_condition=neg_condition,
    )
    torch.testing.assert_close(shifted_target, reference_target, rtol=1e-12, atol=1e-12)
    shifted_endpoint = shifted.pdd_advance_prediction(shifted_state, shifted_target, shifted_weight)
    reference_endpoint = reference.pdd_advance_prediction(reference_state, reference_target, reference_weight)
    torch.testing.assert_close(shifted_endpoint, reference_endpoint, rtol=1e-12, atol=1e-12)

    # Exercise the real selected-head training boundary as well as the target
    # helper. Restricting each block to one interval makes head selection
    # deterministic. Heads and targets share the native sigma-velocity space,
    # so the scalar objective is invariant too.
    shifted_training_state = shifted_state.detach().clone().requires_grad_()
    reference_training_state = reference_state.detach().clone().requires_grad_()
    shifted_prediction = _predict_heads(
        shifted_model,
        shifted,
        shifted_training_state,
        step_index=step_index,
        condition=condition,
    )
    reference_prediction = _predict_heads(
        reference_model,
        reference,
        reference_training_state,
        step_index=step_index,
        condition=condition,
    )
    shifted_losses, shifted_next_state = shifted_model._compute_pdd_loss_from_prediction(
        input_student=shifted_training_state,
        step_idx=step_index,
        pred=shifted_prediction,
        base_grid=shifted_grid,
        condition=condition,
        neg_condition=neg_condition,
        block_end_idx=block_end_index,
        next_idx=block_end_index,
    )
    reference_losses, reference_next_state = reference_model._compute_pdd_loss_from_prediction(
        input_student=reference_training_state,
        step_idx=step_index,
        pred=reference_prediction,
        base_grid=reference_grid,
        condition=condition,
        neg_condition=neg_condition,
        block_end_idx=block_end_index,
        next_idx=block_end_index,
    )
    torch.testing.assert_close(shifted_next_state, reference_next_state, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        shifted_losses["pdd_loss"],
        reference_losses["pdd_loss"],
        rtol=2e-6,
        atol=2e-7,
    )
    torch.testing.assert_close(
        shifted_losses["total_loss"],
        reference_losses["total_loss"],
        rtol=2e-6,
        atol=2e-7,
    )
    shifted_losses["total_loss"].backward()
    reference_losses["total_loss"].backward()
    torch.testing.assert_close(
        shifted_training_state.grad,
        reference_training_state.grad,
        rtol=2e-6,
        atol=2e-7,
    )
