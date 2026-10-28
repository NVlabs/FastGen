# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib
import torch
import pytest
from fastgen.methods import CausalKDModel, FastGenModel, KDModel
from fastgen.configs.config import BaseModelConfig as ModelConfig
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.networks.kv_cache.base import _build_block_token_ranges
from fastgen.networks.noise_schedule import StudentGridConfig
from tests.utils import RunIf


def test_kd_declares_paired_trajectory_state_fields():
    assert FastGenModel.state_fields == frozenset({"real"})
    assert KDModel.state_fields == CausalKDModel.state_fields == frozenset({"real", "noise", "path"})


def test_causal_kd_config_uses_the_noncausal_path_grid():
    causal = importlib.import_module("fastgen.configs.methods.config_kd_causal").create_config()
    noncausal = importlib.import_module("fastgen.configs.experiments.WanT2V.config_kd_path").create_config()

    assert causal.model.student_grid_cfg.grid_type == noncausal.model.student_grid_cfg.grid_type
    assert causal.model.student_grid_cfg.grid_kwargs == noncausal.model.student_grid_cfg.grid_kwargs


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    instance = ModelConfig()
    opts = ["-", "backbone.img_resolution=8", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.input_shape = [3, 8, 8]

    model = KDModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)

    # Create mock data
    data = {
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "noise": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": torch.zeros_like(labels).to(model.device, model.precision),
    }

    return model, data


def test_single_train_step_student_update(get_model_data, monkeypatch):
    model, data = get_model_data
    assert model.net.noise_scheduler.student_base_steps is None

    def unexpected_sample(*args, **kwargs):
        raise AssertionError("a one-step student has no timestep choice to sample")

    def unexpected_index_mapping(*args, **kwargs):
        raise AssertionError("noise-only one-step KD must not request anchored grid indices")

    monkeypatch.setattr(model.net.noise_scheduler, "sample_student_timesteps", unexpected_sample)
    monkeypatch.setattr(model.net.noise_scheduler, "student_grid_indices", unexpected_index_mapping)
    forward, captured = model.net.forward, {}

    def record_student_input(input_student, t_student, **kwargs):
        captured["input_student"], captured["t_student"] = input_student, t_student
        return forward(input_student, t_student, **kwargs)

    monkeypatch.setattr(model.net, "forward", record_student_input)
    # Run the training step
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "recon_loss" in loss_map
    assert "gen_rand" in outputs
    assert callable(outputs["gen_rand"])
    assert isinstance(outputs["input_rand"], torch.Tensor)
    assert isinstance(outputs["gen_rand_train"], torch.Tensor)
    # The one-step student is fed the batch noise scaled to the first node of its grid.
    scheduler = model.net.noise_scheduler
    t_init = scheduler.student_grid(device=model.device)[0].expand(data["noise"].shape[0])
    torch.testing.assert_close(captured["t_student"], t_init, rtol=0, atol=0)
    torch.testing.assert_close(
        captured["input_student"], scheduler.latents(noise=data["noise"], t_init=t_init), rtol=0, atol=0
    )
    assert outputs["gen_rand"].keywords["condition"] is data["condition"]
    assert outputs["gen_rand"].keywords["neg_condition"] is data["neg_condition"]

    generated = outputs["gen_rand"]()
    assert generated.shape == data["real"].shape


def test_optimizers(get_model_data):
    model, data = get_model_data
    # Test for both student and fake_score optimizer
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)


def test_single_step_path_uses_its_source_grid(get_model_data, monkeypatch):
    model, data = get_model_data
    sampling = StudentGridConfig(steps=1, grid_kwargs={"base_steps": 3})
    model.config.student_grid_cfg = sampling
    model.net.noise_scheduler.student_grid_cfg = sampling
    denoise_path = torch.stack([torch.full_like(data["real"], value) for value in range(3)], dim=1)
    data = {key: value for key, value in data.items() if key != "noise"}
    data["path"] = denoise_path
    captured = {}

    def forward(input_student, t_student, **kwargs):
        captured["input_student"] = input_student
        captured["t_student"] = t_student
        return input_student

    monkeypatch.setattr(model.net, "forward", forward)
    monkeypatch.setattr(model, "_get_outputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(model, "apply_reward", lambda gen_data, data, loss_map: loss_map)

    model.single_train_step(data, iteration=0)

    torch.testing.assert_close(captured["input_student"], denoise_path[:, 0])
    torch.testing.assert_close(
        captured["t_student"],
        model.net.noise_scheduler.student_grid(1, device=model.device)[:1],
    )


def test_path_kd_requires_an_anchored_student_grid(get_model_data):
    model, data = get_model_data
    data = {key: value for key, value in data.items() if key != "noise"}
    data["path"] = torch.stack([torch.full_like(data["real"], value) for value in range(3)], dim=1)

    with pytest.raises(ValueError, match="KD path data requires an anchored student grid"):
        model.single_train_step(data, iteration=0)


def test_multistep_kd_supports_nondivisible_path_and_student_steps(get_model_data, monkeypatch):
    model, data = get_model_data
    sampling = StudentGridConfig(steps=4, grid_kwargs={"base_steps": 6})
    model.config.student_grid_cfg = sampling
    model.net.noise_scheduler.student_grid_cfg = sampling

    denoise_path = torch.stack([torch.full_like(data["real"], value) for value in range(6)], dim=1)
    data = {**data, "path": denoise_path}
    sampled_ids = torch.tensor([1], device=model.device)
    captured = {}

    def sample_student_step(low, high, size, device=None, **kwargs):
        assert (low, high, size) == (0, 4, (1,))
        return sampled_ids.to(device=device)

    def forward(input_student, t_student, condition=None, **kwargs):
        captured["input_student"] = input_student
        captured["t_student"] = t_student
        captured["condition"] = condition
        captured["kwargs"] = kwargs
        return input_student

    monkeypatch.setattr(torch, "randint", sample_student_step)
    monkeypatch.setattr(model.net, "forward", forward)
    monkeypatch.setattr(model, "_get_outputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(model, "apply_reward", lambda gen_data, data, loss_map: loss_map)

    model.single_train_step(data, iteration=0)

    # Four student steps partition six path intervals at [0, 1, 3, 4, 6].
    torch.testing.assert_close(captured["input_student"], denoise_path[:, 1])
    torch.testing.assert_close(
        captured["t_student"],
        model.net.noise_scheduler.student_grid(4, device=model.device)[sampled_ids],
    )
    assert captured["condition"] is data["condition"]
    assert captured["kwargs"] == {"fwd_pred_type": "x0"}


def test_multistep_kd_rejects_a_path_with_the_wrong_source_resolution(get_model_data):
    model, data = get_model_data
    sampling = StudentGridConfig(
        steps=4,
        grid_type="explicit",
        grid_kwargs={"timesteps": torch.linspace(model.net.noise_scheduler.max_t, 0.0, 7).tolist()},
    )
    model.config.student_grid_cfg = sampling
    model.net.noise_scheduler.student_grid_cfg = sampling
    data = {
        **data,
        "path": torch.stack([torch.full_like(data["real"], value) for value in range(5)], dim=1),
    }

    with pytest.raises(
        ValueError,
        match="path_steps=5, student_base_steps=6",
    ):
        model.single_train_step(data, iteration=0)


def test_causal_kd_forwards_inhomogeneous_student_grid():
    student_step_indices = torch.tensor([[0, 0, 1, 1, 2, 2]])
    t_student = torch.tensor([[0.9, 0.9, 0.5, 0.5, 0.0, 0.0]])
    captured = {}

    class Scheduler:
        default_student_steps = 2
        student_base_steps = 4

        @staticmethod
        def sample_student_timesteps_inhom(*args, **kwargs):
            assert kwargs["num_steps"] == 2
            return t_student, student_step_indices

        @staticmethod
        def student_grid_indices(target_steps, device=None):
            assert target_steps == 2
            return torch.tensor([0, 2, 4], device=device)

    class Net:
        chunk_size = 2
        noise_scheduler = Scheduler()

        @staticmethod
        def __call__(input_student, t, condition=None, **kwargs):
            captured["input_student"] = input_student
            captured["t_student"] = t
            captured["condition"] = condition
            captured["kwargs"] = kwargs
            return input_student

    model = object.__new__(CausalKDModel)
    torch.nn.Module.__init__(model)
    model.net = Net()
    model.device = torch.device("cpu")
    model._get_outputs = lambda *args, **kwargs: {}
    model.apply_reward = lambda gen_data, data, loss_map: loss_map

    denoise_path = torch.stack(
        [torch.full((1, 1, 6, 1, 1), float(value)) for value in range(4)],
        dim=1,
    )
    condition = object()
    model.single_train_step(
        {
            "path": denoise_path,
            "real": torch.full((1, 1, 6, 1, 1), 4.0),
            "condition": condition,
        },
        iteration=0,
    )

    torch.testing.assert_close(captured["input_student"][:, :, :2], torch.zeros(1, 1, 2, 1, 1))
    torch.testing.assert_close(captured["input_student"][:, :, 2:4], torch.full((1, 1, 2, 1, 1), 2.0))
    torch.testing.assert_close(captured["input_student"][:, :, 4:], torch.full((1, 1, 2, 1, 1), 4.0))
    assert captured["t_student"] is t_student
    assert captured["condition"] is condition
    causal_ctx = captured["kwargs"].pop("causal_ctx")
    assert captured["kwargs"] == {"fwd_pred_type": "x0"}
    # The student trains block-causally over the full video, like CausalSFT and CausVid.
    assert (causal_ctx.cache_tag, causal_ctx.is_ar, causal_ctx.store_kv) == ("pos", False, False)
    assert (causal_ctx.chunk_size, causal_ctx.total_num_frames, causal_ctx.mask_kind) == (2, 6, "blockwise_causal")


@RunIf(min_gpus=1)
def test_causal_kd_trains_on_a_frame_count_the_chunk_size_does_not_divide(tmp_path):
    """8 frames in chunks of 3 train with one path state per mask chunk, [0, 5) and [5, 8), at that chunk's time."""
    from tests.test_network import _tiny_causal_wan_family_network

    device = torch.device("cuda")
    model = object.__new__(CausalKDModel)
    torch.nn.Module.__init__(model)
    model.device = device
    model.precision = torch.float32
    model.use_ema = []
    model.net = _tiny_causal_wan_family_network("causal_wan", tmp_path).to(device)
    model.net.chunk_size = 3
    scheduler = model.net.noise_scheduler
    scheduler.student_grid_cfg = StudentGridConfig(steps=2, grid_kwargs={"base_steps": 4})
    forward, inputs = model.net.forward, []

    def record_inputs(x, t, **kwargs):
        inputs.append((x, t))
        return forward(x, t, **kwargs)

    model.net.forward = record_inputs
    # Each path state holds its base-grid time, as the clean data holds the grid's clean end 0.
    shape = (2, 4, 8, 4, 4)
    data = {
        "path": torch.stack(
            [torch.full(shape, float(time), device=device) for time in scheduler.student_grid(4)[:-1]], dim=1
        ),
        "real": torch.zeros(shape, device=device),
        "condition": torch.randn(2, 6, 8, device=device),
    }
    loss_map, _ = model.single_train_step(data, iteration=0)
    loss_map["total_loss"].backward()

    ((x, t),) = inputs
    for start, end in _build_block_token_ranges(num_frames=8, frame_seqlen=1, chunk_size=3):
        assert (t[:, start:end] == t[:, start : start + 1]).all()
    torch.testing.assert_close(x, t[:, None, :, None, None].to(x.dtype).expand_as(x))
    assert torch.isfinite(loss_map["total_loss"])
    assert any(param.grad is not None for param in model.net.parameters())
