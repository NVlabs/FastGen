# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import gc
import pytest
import torch

import fastgen.methods.distribution_matching.self_forcing as self_forcing
from fastgen.configs.methods.config_self_forcing import ModelConfig
from fastgen.configs.net import CausalWan_1_3B_Config, Wan_1_3B_Config
from fastgen.methods import DMD2Model, SelfForcingModel
from fastgen.networks.network_causal import CausalFastGenNetwork
from tests.utils import RunIf, check_grad_zero
from fastgen.utils.env import set_env_vars


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    set_env_vars()

    config = ModelConfig()
    config.net = CausalWan_1_3B_Config
    config.teacher = Wan_1_3B_Config  # teacher is still bidirectional network

    # Set device and precision
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.precision = "float32"
    config.precision_model = "float64" if config.device.type == "cpu" else "bfloat16"
    config.pretrained_model_path = None  # disable ckpt loading
    config.student_update_freq = 2
    config.student_grid_cfg.steps = 4  # multistep student distillation

    # Set the correct timestep range for Wan models
    config.sample_t_cfg.max_t = 0.999
    config.sample_t_cfg.min_t = 0.001
    config.sample_t_cfg.time_dist_type = "uniform"

    # Use dimensions appropriate for Wan models
    config.input_shape = [16, 21, 30, 52]  # [C, T, H, W] - smaller for testing
    config.enable_preprocessors = False  # The data below holds pre-embedded conditions and latents
    config.gan_loss_weight_gen = 0.0  # Set GAN loss weight

    # Self-forcing specific settings
    config.start_gradient_frame = 0
    config.same_step_across_blocks = True
    config.last_step_only = False

    model = SelfForcingModel(config)
    # only use a single block for testing
    model.net.transformer.blocks = model.net.transformer.blocks[:1]
    model.teacher.transformer.blocks = model.teacher.transformer.blocks[:1]

    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    channels, n_frames, height, width = config.input_shape

    # Create mock video data appropriate for Wan models.
    # Forwards take conditioning in the network's compute dtype, as ``build_conditioning`` returns it.
    condition_dtype = model.net.precision_model
    data = {
        "real": torch.randn(batch_size, channels, n_frames, height, width, device=model.device, dtype=model.precision),
        "condition": torch.randn(batch_size, 512, 4096, device=model.device, dtype=condition_dtype),
        "neg_condition": torch.zeros(batch_size, 512, 4096, device=model.device, dtype=condition_dtype),
        "idx": ["sample-0"],
    }

    return model, data


def test_sf_requires_backward_simulation():
    config = ModelConfig()
    assert config.backward_simulation == "default"

    config.backward_simulation = None
    with pytest.raises(ValueError, match="requires backward_simulation='default'"):
        SelfForcingModel(config)


def test_sf_rejects_sample_t_in_student_bucket():
    class Scheduler:
        @staticmethod
        def student_grid(device=None):
            return torch.tensor([1.0, 0.0], device=device)

        @staticmethod
        def latents(noise, t_init=None):
            del t_init
            return noise

    class Net:
        noise_scheduler = Scheduler()

    model = object.__new__(SelfForcingModel)
    torch.nn.Module.__init__(model)
    model.config = ModelConfig()
    model.config.sample_t_in_student_bucket = True
    model.net = Net()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.input_shape = [1, 2, 1, 1]

    with pytest.raises(ValueError, match="SelfForcingModel provided student_step of shape None"):
        model._training_inputs({"idx": [0, 1]})


def test_sf_rejects_a_start_gradient_frame_off_the_block_starts():
    """Gradients start at a block: 6 frames in chunks of 3 roll out as [0, 3) and [3, 6), so only 0 and 3 are valid."""

    class Net(CausalFastGenNetwork):
        chunk_size = 3

        def forward(self, *args, **kwargs):
            raise AssertionError("the check precedes every forward")

        def clear_caches(self):
            pass

    net = object.__new__(Net)
    torch.nn.Module.__init__(net)
    model = object.__new__(SelfForcingModel)
    torch.nn.Module.__init__(model)
    model.config = ModelConfig()
    model.net = net

    for start_gradient_frame in (-1, 1, 2, 4, 6):
        model.config.start_gradient_frame = start_gradient_frame
        with pytest.raises(ValueError, match=r"start_gradient_frame must fall on a chunk boundary .* one of \[0, 3\]"):
            model.gen_data_from_net(torch.zeros(1, 1, 6, 1, 1), None)


def test_sf_rejects_step_generations():
    config = ModelConfig()
    config.student_generation = "step"

    with pytest.raises(ValueError, match=r"SelfForcingModel supports student_generation in \('x0',\)"):
        SelfForcingModel(config)


def test_sf_training_inputs_start_at_the_first_student_grid_node(monkeypatch):
    """Fresh noise is placed at ``student_grid[0]``, like every other rollout start state."""

    class RecordingScheduler:
        def __init__(self):
            self.t_inits = []

        @staticmethod
        def student_grid(device=None):
            return torch.tensor([0.9, 0.5, 0.0], dtype=torch.float64, device=device)

        def latents(self, noise, t_init=None):
            self.t_inits.append(t_init)
            return noise * t_init.to(noise.dtype)

    class Net:
        noise_scheduler = RecordingScheduler()

    model = object.__new__(SelfForcingModel)
    torch.nn.Module.__init__(model)
    model.net = Net()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.input_shape = [1, 2, 1, 1]
    noise = torch.full((2, 1, 2, 1, 1), 2.0)
    monkeypatch.setattr(self_forcing, "noise_from_shape", lambda *_args, **_kwargs: noise)
    monkeypatch.setattr(
        model, "_sample_t_and_eps", lambda batch_size, sample_t_cfg=None, student_step=None: (None, None)
    )

    _inputs = model._training_inputs({"idx": [0, 1]}, iteration=0)
    input_student = _inputs.input_student
    student_step = _inputs.student_step
    trajectory_step = _inputs.trajectory_step
    generation_kwargs = _inputs.generation_kwargs
    loss_scale = _inputs.loss_scale

    assert student_step is None and trajectory_step is None
    assert generation_kwargs == {}
    assert loss_scale is None
    assert len(model.net.noise_scheduler.t_inits) == 1
    torch.testing.assert_close(model.net.noise_scheduler.t_inits[0], torch.tensor(0.9, dtype=torch.float64))
    torch.testing.assert_close(input_student, torch.full_like(noise, 1.8))


def test_sf_setup_warns_once_without_a_causal_initialization(monkeypatch):
    """The causal student starts from model.pretrained_student_net_path or from the checkpoint the trainer loads
    from pretrained_ckpt_path, which sets resume_iter, before the optimizers are built; without either, setup warns
    once."""
    warnings = []
    monkeypatch.setattr(self_forcing.logger, "warning", warnings.append)
    monkeypatch.setattr(DMD2Model, "init_optimizers", lambda self: None)
    model = object.__new__(SelfForcingModel)
    torch.nn.Module.__init__(model)
    model.config = ModelConfig()

    model.init_optimizers()
    [message] = warnings
    assert "pretrained_student_net_path" in message and "pretrained_ckpt_path" in message

    warnings.clear()
    model.config.pretrained_student_net_path = "ode_init.pt"
    model.init_optimizers()
    model.config.pretrained_student_net_path = None
    model.resume_iter = 0
    model.init_optimizers()
    assert warnings == []


@RunIf(min_gpus=1)
def test_sf_single_train_step_does_not_require_real_data(get_model_data):
    model, data = get_model_data
    assert model.config.gan_loss_weight_gen == 0.0

    data = {key: value for key, value in data.items() if key != "real"}
    loss_map, outputs = model.single_train_step(data, iteration=0)

    assert torch.isfinite(loss_map["total_loss"])
    assert "gan_loss_disc" not in loss_map
    assert outputs["input_rand"].shape == (1, *model.input_shape)


@RunIf(min_gpus=1)
def test_sf_exit_step_sampling(get_model_data, monkeypatch):
    model, data = get_model_data
    calls = []

    def record_choice(high, num_samples=1, device="cpu", src=0):
        del src
        calls.append((high, device, num_samples))
        choices = [idx % high for idx in range(num_samples)]
        return choices[0] if num_samples == 1 else choices

    monkeypatch.setattr(self_forcing, "broadcast_choice", record_choice)

    # Test the exit-step sampling specific to self-forcing
    num_frames = data["real"].shape[2]

    # The rollout folds the remainder frames into the first block
    num_blocks = max(1, num_frames // model.net.chunk_size)
    sample_steps = model.net.noise_scheduler.default_student_steps

    # Sample one exit step per block
    exit_steps = model._sample_exit_steps(num_blocks)

    assert exit_steps == [0] * num_blocks
    assert calls == [(sample_steps, model.device, 1)]

    model.config.same_step_across_blocks = False
    assert model._sample_exit_steps(num_blocks) == [idx % sample_steps for idx in range(num_blocks)]
    assert calls[-1] == (sample_steps, model.device, num_blocks)
    assert model._sample_exit_steps(1) == [0]
    assert calls[-1] == (sample_steps, model.device, 1)

    model.config.last_step_only = True
    calls.clear()
    assert model._sample_exit_steps(num_blocks) == [sample_steps - 1] * num_blocks
    assert calls == []


@RunIf(min_gpus=1)
def test_sf_gen_data_from_net_runs_gradient_rollout(get_model_data, monkeypatch):
    model, data = get_model_data
    model.config.last_step_only = True

    step_calls = 0
    store_calls = 0
    scheduler_step = model.net.noise_scheduler.step
    store_causal_chunk = model._store_causal_chunk

    def record_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        assert args[2].ndim == 0
        return scheduler_step(*args, **kwargs)

    def record_store(*args, **kwargs):
        nonlocal store_calls
        store_calls += 1
        return store_causal_chunk(*args, **kwargs)

    def reset_peak_memory_stats(*args, **kwargs):
        raise AssertionError("the rollout must keep the peak-memory statistics that GPUStatsCallback reports")

    monkeypatch.setattr(model.net.noise_scheduler, "step", record_step)
    monkeypatch.setattr(model, "_store_causal_chunk", record_store)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak_memory_stats)

    # Test the Self-Forcing gen_data_from_net override.
    noise = torch.randn_like(data["real"])
    condition = data["condition"]

    # Student-update callers preserve gradients at the sampled exit step.
    model.config.start_gradient_frame = 0
    generation = model.gen_data_from_net(noise, None, condition)
    gen_frames = generation.data
    generation_metrics = generation.metrics

    # Check output shape matches input
    assert gen_frames.shape == noise.shape
    assert gen_frames.requires_grad
    assert generation.next_state is None
    assert generation.x0 is generation.data
    torch.testing.assert_close(
        generation_metrics["avg_denoising_steps"],
        torch.tensor(float(model.net.noise_scheduler.default_student_steps), device=model.device),
    )

    # Fake-score and validation callers disable gradients through their outer context.
    with torch.no_grad():
        generation = model.gen_data_from_net(noise, None, condition)
        gen_frames_no_grad = generation.data
        generation_metrics = generation.metrics

    assert gen_frames_no_grad.shape == noise.shape
    assert not gen_frames_no_grad.requires_grad
    assert set(generation_metrics) == {"avg_denoising_steps"}
    assert generation.next_state is None
    assert generation.x0 is generation.data
    assert step_calls > 0
    assert store_calls > 0
    assert not hasattr(model, "rollout_with_gradient")


@RunIf(min_gpus=1)
def test_sf_single_train_step_student_update(get_model_data):
    model, data = get_model_data
    # Run the training step for student update
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "vsd_loss" in loss_map
    assert "gan_loss_gen" in loss_map
    assert "avg_denoising_steps" in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)
    assert isinstance(outputs["input_rand"], torch.Tensor)

    gen_tensor = outputs["gen_rand"]()
    assert gen_tensor.shape == data["real"].shape
    assert outputs["input_rand"].shape == data["real"].shape


@RunIf(min_gpus=1)
def test_sf_single_train_step_fake_score_update(get_model_data):
    model, data = get_model_data

    # Run the training step for fake score update
    loss_map, outputs = model.single_train_step(data, 1)

    # Assertions
    assert "fake_score_loss" in loss_map
    assert ("gan_loss_disc" in loss_map) == (model.config.gan_loss_weight_gen > 0)
    assert "avg_denoising_steps" in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)
    assert isinstance(outputs["input_rand"], torch.Tensor)

    # Check that losses are finite
    assert torch.isfinite(loss_map["total_loss"])
    assert torch.isfinite(loss_map["fake_score_loss"])


@RunIf(min_gpus=1)
def test_sf_optimizers(get_model_data):
    model, data = get_model_data

    gc.collect()
    torch.cuda.empty_cache()

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
    check_grad_zero(model.fake_score)
