# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import contextlib
import os
import gc
import tempfile
from types import SimpleNamespace

import pytest
import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)

from fastgen.callbacks.ema import EMACallback
from fastgen.configs.methods.config_dmd2 import create_config
from fastgen.methods.model import FastGenModel
from fastgen.networks.network import FastGenModule
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.configs.net import EDM2_IN64_S_Config
from fastgen.methods import DMD2Model
from fastgen.trainer import Trainer
from fastgen.configs.instantiate import instantiate
from fastgen.utils import basic_utils
from fastgen.utils.distributed.fsdp import create_fsdp_context, reshard_fsdp_modules
from fastgen.utils.env import set_env_vars
from fastgen.configs.callbacks import (
    CTSchedule_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    WANDB_CALLBACK,
    VLM_IMG_CALLBACK,
    EMA_CALLBACK,
    TrainProfiler_CALLBACK,
    GPUStats_CALLBACK,
    ForcedWeightNorm_CALLBACK,
)
from fastgen.callbacks.callback import CallbackDict
from fastgen.callbacks.wandb import WandbCallback, vae_decode_context
from tests.utils import RunIf, run_distributed_test


@pytest.mark.parametrize("as_dict", [False, True])
def test_wandb_audio_latents_decode_to_wav_and_log_audio(as_dict, tmp_path, monkeypatch):
    import wave

    import fastgen.callbacks.wandb as callback_module

    class AudioVAE(torch.nn.Module):
        modalities = ("audio",)

        def decode(self, latents):
            waveform = torch.full((2, 2, 240), 0.25)
            return {"audio": waveform} if isinstance(latents, dict) else waveform

    latents = torch.zeros(2, 8, 4, 16)
    media = {"audio": latents} if as_dict else latents
    assert callback_module.is_savable(media)
    callback = WandbCallback()
    entries = callback._save_batch(
        tmp_path,
        "student",
        media,
        0,
        ["first", "second"],
        AudioVAE(),
        None,
        {"audio_sample_rate": 48000, "vid_format": "mp4"},
    )
    for entry in entries:
        assert entry.path.suffix == ".wav"
        with wave.open(str(entry.path)) as stream:
            assert stream.getnchannels() == 2
            assert stream.getframerate() == 48000
            assert stream.getnframes() == 240

    logged = {}
    monkeypatch.setattr(callback_module.wandb, "run", object())
    monkeypatch.setattr(callback_module.wandb, "Audio", lambda path, caption: (path, caption))
    monkeypatch.setattr(callback_module.wandb, "log", lambda payload, step: logged.update(payload))
    callback._log_panels(entries, iteration=1, group="train", suffix="")
    assert next(iter(logged.values())) == [(str(entry.path), entry.prompt) for entry in entries]


class _EMAValidationModule(torch.nn.Module):
    def __init__(
        self,
        parameter_shape: tuple[int, ...] | None = None,
        buffer_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if parameter_shape is not None:
            self.weight = torch.nn.Parameter(torch.zeros(parameter_shape))
        if buffer_shape is not None:
            self.register_buffer("state", torch.zeros(buffer_shape))


@pytest.fixture(autouse=True)
def _finish_wandb_runs():
    """Finish the disabled runs ``on_app_begin`` opens, so no test leaves a global ``wandb.run`` behind."""
    yield
    if wandb.run is not None:
        wandb.finish()


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    dmd_config = create_config()
    dmd_config.log_config.name = "test"

    instance = dmd_config.model
    opts = ["-", "backbone.img_resolution=8", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)
    instance.use_ema = True
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.input_shape = [3, 8, 8]

    dmd_model = DMD2Model(instance)
    dmd_model.on_train_begin()
    dmd_model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    neg_condition = torch.zeros(batch_size, 10)

    # Create mock data; EDM takes its conditioning in FP32, as its ``build_conditioning`` returns it.
    data = {
        "idx": list(range(batch_size)),
        "real": torch.randn(batch_size, 3, 8, 8).to(dmd_model.device, dmd_model.precision),
        "condition": labels.to(dmd_model.device, torch.float32),
        "neg_condition": neg_condition.to(dmd_model.device, torch.float32),
    }
    return dmd_model, data, dmd_config


def test_ema_callback_validates_plain_parameter_shapes():
    model = SimpleNamespace(
        net=_EMAValidationModule(parameter_shape=(2, 3)),
        ema=_EMAValidationModule(parameter_shape=(2, 4)),
    )

    with pytest.raises(RuntimeError, match=r"EMA parameter 'weight'.*shape"):
        EMACallback(beta=0.5).on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=1,
        )


def test_ema_callback_validates_gathered_parameter_shape():
    model = SimpleNamespace(
        net=_EMAValidationModule(parameter_shape=(2, 3)),
        ema=_EMAValidationModule(parameter_shape=(2, 3)),
    )
    model.net.weight.full_tensor = lambda: torch.zeros(2, 4)

    with pytest.raises(RuntimeError, match=r"EMA parameter 'weight'.*shape"):
        EMACallback(beta=0.5).on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=1,
        )


def test_ema_callback_validates_buffer_shapes():
    model = SimpleNamespace(
        net=_EMAValidationModule(buffer_shape=(3,)),
        ema=_EMAValidationModule(buffer_shape=(4,)),
    )

    with pytest.raises(RuntimeError, match=r"EMA buffer 'state'.*shape"):
        EMACallback(beta=0.5).on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=1,
        )


@pytest.mark.parametrize(
    ("parameter_dtype", "buffer_dtype", "warns"),
    [
        (torch.bfloat16, torch.float32, True),
        (torch.float16, torch.float32, True),
        (torch.float32, torch.float32, False),
        (torch.float32, torch.bfloat16, False),
    ],
    ids=["bf16", "fp16", "fp32", "bf16-buffer"],
)
def test_ema_callback_warns_about_16_bit_ema_parameters(monkeypatch, parameter_dtype, buffer_dtype, warns):
    warning_messages = []
    monkeypatch.setattr("fastgen.callbacks.ema.logger.warning", warning_messages.append)
    ema = _EMAValidationModule(parameter_shape=(2,), buffer_shape=(2,)).to(parameter_dtype)
    ema.state = ema.state.to(buffer_dtype)
    ema.eval().requires_grad_(False)

    EMACallback(ema_name="ema_slow").on_model_init_end(SimpleNamespace(ema_slow=ema))

    assert len(warning_messages) == int(warns), warning_messages
    if warns:
        assert "ema_slow" in warning_messages[0] and str(parameter_dtype) in warning_messages[0]
        assert 'precision_opt="float32"' in warning_messages[0]


def test_ema_callback_warns_once_per_missing_tensor(monkeypatch):
    warning_messages = []
    monkeypatch.setattr("fastgen.callbacks.ema.logger.warning", warning_messages.append)
    model = SimpleNamespace(
        net=_EMAValidationModule(parameter_shape=(2,), buffer_shape=(2,)),
        ema=_EMAValidationModule(),
    )
    callback = EMACallback(beta=0.5)

    for iteration in range(5, 8):
        callback.on_training_step_end(model, data_batch=None, output_batch=None, loss_dict=None, iteration=iteration)

    assert warning_messages == [
        "EMA parameter weight not found, skipping update.",
        "EMA buffer state not found, skipping update.",
    ]


def _all_true(flag: bool) -> bool:
    probe = torch.tensor(int(flag), dtype=torch.int64)
    dist.all_reduce(probe, op=dist.ReduceOp.MIN)
    return bool(probe.item())


def _module_with_parameter(parameter: torch.Tensor) -> torch.nn.Module:
    module = torch.nn.Module()
    module.register_parameter("weight", torch.nn.Parameter(parameter, requires_grad=False))
    return module


def _ema_dtensor_validation_impl(rank: int, world_size: int) -> dict[str, bool]:
    mesh = init_device_mesh("cuda", (world_size,))
    device = torch.device("cuda", rank)
    callback = EMACallback(beta=0.5)

    def validation_error(student: DTensor, ema: DTensor, expected: str) -> bool:
        model = SimpleNamespace(
            net=_module_with_parameter(student),
            ema=_module_with_parameter(ema),
        )
        try:
            callback.on_training_step_end(
                model,
                data_batch=None,
                output_batch=None,
                loss_dict=None,
                iteration=1,
            )
        except RuntimeError as error:
            return expected in str(error)
        return False

    student = distribute_tensor(torch.zeros(8, device=device), mesh, [Shard(0)])
    replicated_ema = distribute_tensor(torch.zeros(8, device=device), mesh, [Replicate()])
    placements_checked = validation_error(student, replicated_ema, "placements")

    student = distribute_tensor(torch.zeros(8, device=device), mesh, [Shard(0)])
    wrong_shape_ema = distribute_tensor(torch.zeros(10, device=device), mesh, [Shard(0)])
    global_shape_checked = validation_error(student, wrong_shape_ema, "global shape")

    return {
        "placements_checked": _all_true(placements_checked),
        "global_shape_checked": _all_true(global_shape_checked),
    }


@RunIf(min_gpus=2)
def test_ema_callback_validates_dtensor_layout_and_global_shape():
    result = run_distributed_test(test_fn=_ema_dtensor_validation_impl, world_size=2, timeout=240)
    assert result is not None
    assert all(result.values()), result


def test_ema_callback(get_model_data):
    """Test EMA callback basic functionality (non-FSDP mode)."""
    model, data, config = get_model_data

    for callback_name, callback_config in EMA_CALLBACK.items():
        assert callback_name == "ema"
        assert model.ema is not None

        ema_callback = instantiate(callback_config)
        ema_callback.config = config
        # on_app_begin reads the total batch size from the config
        ema_callback.on_app_begin()

        assert ema_callback.beta == 0.9999
        assert ema_callback.type == "constant"
        assert ema_callback.gamma == 16.97
        assert ema_callback.ema_halflife_kimg == 500
        assert ema_callback.ema_rampup_ratio == 0.05

        ema_callback.on_model_init_end(model)
        assert ema_callback._enabled is True

        # EMA should be initialized from net during model.build_model()
        ema_state = model.ema.state_dict()
        net_state = model.net.state_dict()
        assert all(torch.allclose(net_state[k], p_ema) for k, p_ema in ema_state.items())
        assert not any(p_ema.requires_grad for p_ema in ema_state.values())

        # Modify network parameters and compute expected EMA update
        buffers = [k for k, _ in model.net.named_buffers()]
        expected_ema_state = {}
        for k, p_net in net_state.items():
            torch.nn.init.normal_(p_net)
            if k in buffers:
                expected_ema_state[k] = p_net.detach().clone()
            else:
                expected_ema_state[k] = torch.lerp(ema_state[k], p_net.detach(), 1.0 - ema_callback.beta)

        # Run EMA update step
        ema_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=1,
        )

        # Verify EMA was updated correctly
        new_ema_state = model.ema.state_dict()
        assert all(torch.allclose(expected_ema_state[k], p_ema) for k, p_ema in new_ema_state.items())
        assert not any(p_ema.requires_grad for p_ema in new_ema_state.values())

        # Test that EMA update is skipped when ema is None
        model.ema = None
        ema_callback.on_model_init_end(model)
        assert ema_callback._enabled is False
        ema_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=1,
        )
        assert model.ema is None


def test_ema_initialization_after_build(get_model_data):
    """Test that EMA is correctly initialized from net state during model build."""
    model, data, config = get_model_data

    # Verify EMA exists and matches net state
    assert model.ema is not None
    assert model.use_ema == ["ema"]

    ema_state = model.ema.state_dict()
    net_state = model.net.state_dict()

    # All EMA parameters should match net parameters exactly after initialization
    for k in net_state.keys():
        assert k in ema_state, f"Key {k} not found in EMA state"
        assert torch.allclose(net_state[k], ema_state[k]), f"EMA state mismatch for {k}"

    # EMA should not require gradients
    assert not any(p.requires_grad for p in model.ema.parameters())
    assert model.ema.training is False  # EMA should be in eval mode


def test_ema_callback_multiple_steps(get_model_data):
    """Test EMA callback over multiple training steps to verify accumulation."""
    model, data, config = get_model_data

    ema_callback = instantiate(EMA_CALLBACK["ema"])
    ema_callback.config = config
    ema_callback.on_app_begin()

    beta = ema_callback.beta

    # Store initial EMA state
    initial_ema_state = {k: v.clone() for k, v in model.ema.state_dict().items()}
    buffers = [k for k, _ in model.net.named_buffers()]

    # Run multiple EMA update steps
    for iteration in range(1, 5):
        # Modify network parameters
        for p in model.net.parameters():
            torch.nn.init.normal_(p)

        # Update expected EMA
        net_state = model.net.state_dict()
        for k in initial_ema_state.keys():
            if k in buffers:
                initial_ema_state[k] = net_state[k].clone()
            else:
                initial_ema_state[k].lerp_(net_state[k], 1.0 - beta)

        ema_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=iteration,
        )

    # Verify final EMA state
    final_ema_state = model.ema.state_dict()
    for k, expected in initial_ema_state.items():
        assert torch.allclose(expected, final_ema_state[k], atol=1e-6), f"Mismatch at {k}"


def test_ema_callback_gathers_a_sharded_student_for_a_replicated_ema(get_model_data):
    """A replicated EMA takes the compatibility branch: student parameters exposing ``full_tensor()`` are gathered
    before the update (a live sharded EMA updates local shards instead). The mock stands in for the all-gather."""
    model, data, config = get_model_data

    # Mock FSDP by adding full_tensor method to parameters
    # In real FSDP, this gathers the full tensor from all shards
    original_params = {}
    for name, param in model.net.named_parameters():
        original_params[name] = param.data.clone()
        # Add a mock full_tensor method that returns the parameter itself
        param.full_tensor = lambda p=param: p.data.clone()

    ema_callback = instantiate(EMA_CALLBACK["ema"])
    ema_callback.config = config
    ema_callback.on_app_begin()

    # Get initial EMA state
    initial_ema_state = {k: v.clone() for k, v in model.ema.state_dict().items()}
    buffers = [k for k, _ in model.net.named_buffers()]

    # Modify network parameters
    for p in model.net.named_parameters():
        torch.nn.init.normal_(p[1])

    # Compute expected EMA update
    expected_ema_state = {}
    net_state = model.net.state_dict()
    for k in initial_ema_state.keys():
        if k in buffers:
            expected_ema_state[k] = net_state[k].clone()
        else:
            expected_ema_state[k] = torch.lerp(initial_ema_state[k], net_state[k], 1.0 - ema_callback.beta)

    # Run EMA update (should use full_tensor() for FSDP)
    ema_callback.on_training_step_end(
        model,
        data_batch=None,
        output_batch=None,
        loss_dict=None,
        iteration=1,
    )

    # Verify EMA was updated correctly
    final_ema_state = model.ema.state_dict()
    for k, expected in expected_ema_state.items():
        assert torch.allclose(expected, final_ema_state[k], atol=1e-6), f"Mismatch at {k}"


# =============================================================================
# True FSDP EMA Test Implementation
# =============================================================================


def _test_ema_callback_fsdp_distributed_impl(rank: int, world_size: int) -> dict:
    """Test EMA callback with real FSDP in a distributed setting using EDM model.

    This test uses the same EDM model architecture as other callback tests to ensure
    we're testing the actual model code paths. It verifies that:
    1. EMA callback correctly gathers full tensors from FSDP-sharded parameters
    2. EMA state remains consistent after update

    Args:
        rank: Process rank
        world_size: Total number of processes

    Returns:
        dict with test results
    """
    from fastgen.callbacks.ema import EMACallback
    from fastgen.configs.config_utils import override_config_with_opts
    from fastgen.utils.distributed.collectives import synchronize
    from fastgen.utils.distributed import is_rank0
    from fastgen.utils.module import module_state_tensors

    device_mesh = init_device_mesh("cuda", (world_size,))
    device = torch.cuda.current_device()

    # Create EDM network using the same configuration as other tests
    # Use small resolution and simple architecture for fast testing
    dmd_config = create_config()
    instance = dmd_config.model
    opts = ["-", "backbone.img_resolution=8", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device(f"cuda:{rank}")
    instance.precision = "float32"
    instance.pretrained_model_path = None  # disable ckpt loading

    # Instantiate the network (EDM architecture)
    net = instantiate(instance.net).to(device)

    # Get state dict before FSDP sharding for broadcast
    if is_rank0():
        broadcast_state_dict = copy.deepcopy(net.state_dict())
    else:
        broadcast_state_dict = None

    synchronize()

    # Apply FSDP sharding using the network's fully_shard method
    # This follows the same pattern as test_fsdp.py
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    net.fully_shard(mesh=device_mesh, mp_policy=mp_policy)

    # Materialize meta tensors and reset parameters (following test_fsdp.py pattern)
    net.model.to_empty(device=device)
    if hasattr(net, "reset_parameters"):
        net.reset_parameters()
    synchronize()

    # Broadcast state dict from rank 0 (following test_fsdp.py pattern)
    # Extract only the inner model's state dict since that's what's sharded
    if broadcast_state_dict is not None:
        inner_model_prefix = "model."
        inner_broadcast_state_dict = {
            k[len(inner_model_prefix) :]: v for k, v in broadcast_state_dict.items() if k.startswith(inner_model_prefix)
        }
    else:
        inner_broadcast_state_dict = None

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(net.model, model_state_dict=inner_broadcast_state_dict, options=options)
    synchronize()

    # Create EMA model (matching production behavior)
    ema_init_state = {}
    for name, p in net.named_parameters():
        if hasattr(p, "full_tensor"):
            # All ranks must participate in full_tensor() gather
            full_p = p.full_tensor().detach().clone()
        else:
            full_p = p.detach().clone()
        ema_init_state[name] = full_p

    # Only persistent buffers belong in load_state_dict; runtime scheduler caches are rebuilt.
    for _kind, name, buf in module_state_tensors(net, parameters=False, nonpersistent_buffers=False):
        ema_init_state[name] = buf.detach().clone()

    # Create a fresh EDM network for EMA
    ema = instantiate(instance.net).to(device)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad = False
    ema.load_state_dict(ema_init_state)
    initial_ema_state = {k: v.clone() for k, v in ema.state_dict().items()}

    synchronize()

    # Create EMA callback (the update path dispatches on the sharded parameter type)
    ema_callback = EMACallback(
        type="constant",
        beta=0.9,  # Use larger learning rate for visible updates
    )

    # Modify network parameters (simulate training step)
    with torch.no_grad():
        for p in net.parameters():
            # The modification happens via the sharded parameter
            p.data.add_(torch.randn_like(p.data) * 0.1)

    synchronize()

    # Create a mock model object with net and ema attributes
    class MockModel:
        def __init__(self, net, ema):
            self.net = net
            self.ema = ema
            self.ema_enabled = True
            self.resume_iter = 0

    mock_model = MockModel(net, ema)

    # Run EMA update
    ema_callback.on_training_step_end(
        mock_model,
        data_batch=None,
        output_batch=None,
        loss_dict=None,
        iteration=1,
    )

    synchronize()

    # Verify results
    assert ema is not None, "EMA should exist"

    # Check that EMA was updated (should be different from initial)
    final_ema_state = ema.state_dict()

    # Check if EMA was updated (at least one param should differ)
    params_changed = 0
    for k in initial_ema_state:
        if not torch.allclose(initial_ema_state[k], final_ema_state[k], atol=1e-6):
            params_changed += 1

    results = {
        "ema_different_from_initial": params_changed > 0,
        "params_changed": params_changed,
        "total_params": len(initial_ema_state),
        "ema_no_grad": not any(p.requires_grad for p in ema.parameters()),
        "model_type": "EDM",
    }
    return results


@RunIf(min_gpus=2)
def test_ema_callback_fsdp_distributed():
    """Test EMA callback with real FSDP distributed training using EDM model.

    This test requires at least 2 GPUs and uses the actual EDM network architecture
    (same as other callback tests) to verify that the EMA callback correctly handles
    FSDP-sharded parameters by:
    1. Gathering full tensors from all shards using full_tensor()
    2. Performing EMA updates
    3. Maintaining proper synchronization across ranks
    """
    gc.collect()
    torch.cuda.empty_cache()

    result = run_distributed_test(
        test_fn=_test_ema_callback_fsdp_distributed_impl,
        world_size=2,
        timeout=180,  # Slightly longer for model instantiation
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("model_type") == "EDM", "Test should use EDM model"
    assert result["ema_different_from_initial"], (
        f"EMA should have been updated after training step. "
        f"Only {result.get('params_changed', 0)}/{result.get('total_params', 0)} params changed."
    )
    assert result["ema_no_grad"], "EMA parameters should not require gradients"

    gc.collect()
    torch.cuda.empty_cache()


class _LiveEMANet(FastGenModule):
    """Tiny mixed-storage net with a root-owned parameter that stays unsharded after forward."""

    def __init__(self):
        super().__init__()
        self.trainable_layer = torch.nn.Linear(8, 8)
        self.frozen_layer = torch.nn.Linear(8, 8)
        self.frozen_layer.requires_grad_(False)
        self.register_buffer("scale", torch.ones(8))
        self.root_bias = torch.nn.Parameter(torch.zeros(8))
        # The model treats ``net`` as the preprocessor host, like ``FastGenNetwork``.
        self.preprocessors = {}

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        self.frozen_layer.requires_grad_(False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trainable = self.trainable_layer(x.to(dtype=self.precision_model))
        frozen = self.frozen_layer(x.to(dtype=self.precision_model))
        output = (trainable + frozen) * self.scale.to(device=trainable.device, dtype=trainable.dtype)
        return output.to(dtype=x.dtype) + self.root_bias.to(device=output.device, dtype=x.dtype)


class _LiveEMAModel(FastGenModel):
    @property
    def model_dict(self):
        return torch.nn.ModuleDict({"net": self.net, "ema": self.ema})

    def _get_outputs(self, *args, **kwargs):
        raise NotImplementedError

    def single_train_step(self, data, iteration):
        del iteration
        prediction = self.net(data["x"])
        loss = (prediction - data["target"]).square().mean()
        return {"total_loss": loss}, {"prediction": prediction}


def _canonical_parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name.replace("_checkpoint_wrapped_module.", ""): parameter for name, parameter in module.named_parameters()}


def _fsdp_live_ema_impl(rank: int, world_size: int) -> dict:
    del world_size
    model = _LiveEMAModel.__new__(_LiveEMAModel)
    torch.nn.Module.__init__(model)
    model.module_dict = {}
    model.config = SimpleNamespace(
        inference=False,
        ddp_find_unused_parameters=False,
        enable_preprocessors=False,
        extra_fsdp_modules=[],
        net_adapter=SimpleNamespace(type=None),
        torch_compile_mode=None,
    )
    model.device = torch.device("cuda", rank)
    model._resolve_precisions(
        precision="float32",
        precision_model="bfloat16",
        precision_opt="float32",
        precision_amp="bfloat16",
        precision_amp_infer="bfloat16",
        precision_fsdp_reduce="bfloat16",
    )
    model.use_ema = ["ema"]

    torch.manual_seed(2468)
    model.net = _LiveEMANet()
    model.ema = _LiveEMANet()
    model.ema.load_state_dict(model.net.state_dict())
    model.module_dict.update(model.model_dict)

    fsdp_context = create_fsdp_context(
        reduce_dtype=model.precision_fsdp_reduce, min_num_params=1, apply_cpu_offload=False
    )
    model.on_train_begin(fsdp_context=fsdp_context)

    callback = EMACallback(type="constant", beta=0.5)
    callback.on_model_init_end(model)

    net_params = _canonical_parameters(model.net)
    ema_params = _canonical_parameters(model.ema)
    initially_sharded = all(
        isinstance(parameter, DTensor) for parameter in (*net_params.values(), *ema_params.values())
    )
    optim_storage_precision = all(
        parameter.to_local().dtype == torch.float32 for parameter in (*net_params.values(), *ema_params.values())
    )

    # Simulate an optimizer update directly on the student's local shards.
    with torch.no_grad():
        for index, parameter in enumerate(net_params.values(), start=1):
            parameter.to_local().fill_(float(index))
        model.net.scale.fill_(7.0)

    ema_before = {name: parameter.to_local().detach().clone() for name, parameter in ema_params.items()}
    net_after = {name: parameter.to_local().detach().clone() for name, parameter in net_params.items()}

    # Generic inference/no-grad scopes do not own the FSDP lifecycle. In particular,
    # an auxiliary training iteration may forward the student without ever running its
    # backward hook, leaving both roots materialized before the callback phase.
    sample = torch.ones(2, 8, device=model.device, dtype=torch.float32)
    with basic_utils.inference_mode(model.ema, precision_amp=None, device_type=model.device.type):
        model.ema(sample)
    with torch.no_grad():
        model.net(sample)
    unsharded_without_backward = not isinstance(model.net.root_bias, DTensor) and not isinstance(
        model.ema.root_bias, DTensor
    )

    callback_state = {}

    class CallbackChain:
        @staticmethod
        def on_training_step_end(**kwargs):
            callback_state["sharded_before_ema_update"] = isinstance(model.net.root_bias, DTensor) and isinstance(
                model.ema.root_bias, DTensor
            )
            callback.on_training_step_end(**kwargs)

    # The trainer restores every callback input once at the phase boundary, so EMA can
    # update local shards without repairing either root itself.
    trainer = object.__new__(Trainer)
    trainer.callbacks = CallbackChain()
    trainer.fsdp_context = fsdp_context

    original_full_tensor = DTensor.full_tensor

    def fail_on_full_tensor(*args, **kwargs):
        del args, kwargs
        raise AssertionError("A sharded live EMA update must not gather a full tensor")

    DTensor.full_tensor = fail_on_full_tensor
    try:
        reshard_fsdp_modules(model.module_dict.values())
        trainer.callbacks.on_training_step_end(
            model=model,
            data_batch={},
            output_batch={},
            loss_dict={},
            iteration=1,
        )
    finally:
        DTensor.full_tensor = original_full_tensor

    updated_ema_params = _canonical_parameters(model.ema)
    sharded_after_update = isinstance(model.ema.root_bias, DTensor) and all(
        isinstance(parameter, DTensor) for parameter in updated_ema_params.values()
    )
    update_ok = True
    for name, parameter in updated_ema_params.items():
        expected = ema_before[name].clone()
        expected.lerp_(net_after[name].to(device=expected.device, dtype=expected.dtype), 0.5)
        update_ok = update_ok and torch.equal(parameter.to_local(), expected)

    buffer_ok = torch.equal(model.ema.scale, model.net.scale.to(dtype=model.ema.scale.dtype))
    return {
        "initially_sharded": _all_true(initially_sharded),
        "optim_storage_precision": _all_true(optim_storage_precision),
        "unsharded_without_backward": _all_true(unsharded_without_backward),
        "sharded_before_ema_update": _all_true(callback_state["sharded_before_ema_update"]),
        "sharded_after_update": _all_true(sharded_after_update),
        "shard_local_update_ok": _all_true(update_ok),
        "buffer_update_ok": _all_true(buffer_ok),
    }


@RunIf(min_gpus=2)
def test_ema_callback_fsdp_live_update_uses_local_shards_after_no_grad_forwards():
    result = run_distributed_test(test_fn=_fsdp_live_ema_impl, world_size=2, timeout=240)
    assert result is not None
    assert all(result.values()), result


# =============================================================================
# Non-Distributed Tests (continue below)
# =============================================================================


def test_ema_checkpoint_save_load(get_model_data):
    """Test that EMA state is correctly saved and loaded from checkpoints."""
    model, data, config = get_model_data

    # Initialize EMA callback and run a few updates
    ema_callback = instantiate(EMA_CALLBACK["ema"])
    ema_callback.config = config
    ema_callback.on_app_begin()
    ema_callback.on_model_init_end(model)

    # Modify network and update EMA a few times
    for i in range(3):
        for p in model.net.parameters():
            torch.nn.init.normal_(p)
        ema_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=i + 1,
        )

    # Store EMA state before saving
    ema_state_before = {k: v.clone() for k, v in model.ema.state_dict().items()}

    # Create a temporary directory for checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        from fastgen.configs.config import BaseCheckpointerConfig
        from fastgen.utils.checkpointer import Checkpointer

        checkpointer = Checkpointer(BaseCheckpointerConfig(save_dir=tmpdir, use_s3=False))

        # Save checkpoint
        checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=None,
            scheduler_dict=None,
            grad_scaler=None,
            callbacks=None,
            path=os.path.join(tmpdir, "test_ema.pth"),
            iteration=100,
        )

        # Verify checkpoint file exists
        assert os.path.exists(os.path.join(tmpdir, "test_ema.pth"))

        # Reset EMA state to verify loading works
        for k in model.ema.state_dict():
            model.ema.state_dict()[k].zero_()

        # Verify EMA is zeroed
        for k, v in model.ema.state_dict().items():
            assert torch.all(v == 0), f"EMA {k} should be zeroed"

        # Load checkpoint
        loaded_iter = checkpointer.load(
            model_dict=model.model_dict,
            optimizer_dict=None,
            scheduler_dict=None,
            grad_scaler=None,
            callbacks=None,
            path=os.path.join(tmpdir, "test_ema.pth"),
        )

        assert loaded_iter == 100

        # Verify EMA state was restored
        ema_state_after = model.ema.state_dict()
        for k, v_before in ema_state_before.items():
            assert torch.allclose(v_before, ema_state_after[k]), f"EMA state mismatch for {k}"


def test_ema_callback_beta_types(get_model_data):
    """Test EMA callback with different beta calculation types."""
    model, data, config = get_model_data

    # Test power function beta
    ema_callback_power = instantiate(EMA_CALLBACK["ema"])
    ema_callback_power.type = "power"
    ema_callback_power.config = config
    ema_callback_power.on_app_begin()

    # Power function should return beta = (1 - 1/iteration)^(gamma + 1)
    iteration = 10
    expected_power_beta = (1 - 1 / iteration) ** (ema_callback_power.gamma + 1)
    actual_power_beta = ema_callback_power._power_function_beta(iteration)
    assert np.isclose(expected_power_beta, actual_power_beta)

    # Test halflife beta
    ema_callback_halflife = instantiate(EMA_CALLBACK["ema"])
    ema_callback_halflife.type = "halflife"
    ema_callback_halflife.config = config
    ema_callback_halflife.on_app_begin()

    # Halflife beta should use the formula 0.5^(batch_size / ema_halflife_nimg)
    iteration = 100
    halflife_beta = ema_callback_halflife._halflife_beta(iteration)
    assert 0 < halflife_beta < 1, f"Halflife beta should be between 0 and 1, got {halflife_beta}"


@pytest.mark.parametrize("q,late_stage", [(2.0, 1023), (4.0, 511), (4, 10000)])
def test_ct_schedule_saturates_without_overflow(q, late_stage):
    from fastgen.callbacks.ct_schedule import CTScheduleCallback

    callback = CTScheduleCallback(q=q, ratio_limit=0.9961, kimg_per_stage=1)
    model = SimpleNamespace()
    for stage in (0, 1, late_stage):
        callback.on_train_begin(model, iteration=stage * 1000)
        expected = 1 - 1 / q ** (stage + 1) if stage < 2 else callback.ratio_limit
        assert callback.stage == stage
        assert model.ratio == expected


@pytest.mark.parametrize("q", [0.0, 0.5, 1.0, float("inf"), float("nan")])
def test_ct_schedule_rejects_non_growing_or_nonfinite_base(q):
    from fastgen.callbacks.ct_schedule import CTScheduleCallback

    with pytest.raises(ValueError, match="q must be finite and greater than 1"):
        CTScheduleCallback(q=q)


def test_ct_schedule_callback(get_model_data):
    model, data, config = get_model_data

    for callback_name, callback_config in CTSchedule_CALLBACK.items():
        assert callback_name == "ct_schedule"
        assert config.dataloader_train.batch_size == 256

        ct_schedule_callback = instantiate(callback_config)
        ct_schedule_callback.config = config

        assert ct_schedule_callback.q == 2.0
        assert ct_schedule_callback.ratio_limit == 0.999
        assert ct_schedule_callback.kimg_per_stage == 12500

        ct_schedule_callback.on_train_begin(model, iteration=0)
        assert np.isclose(ct_schedule_callback.stage, 0)
        assert model.ratio == 0.5

        model.resume_iter = 100000
        ct_schedule_callback.on_train_begin(model, iteration=0)
        assert np.isclose(ct_schedule_callback.stage, 2)
        assert model.ratio == 0.875

        ct_schedule_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=100000,
        )

        assert np.isclose(ct_schedule_callback.stage, 4)
        assert np.isclose(model.ratio, 0.96875)


def test_grad_clip_callback(get_model_data):
    model, data, config = get_model_data
    for callback_name, callback_config in GradClip_CALLBACK.items():
        assert callback_name == "grad_clip"
        assert callback_config.grad_norm == 10.0

        grad_clip_callback = instantiate(callback_config)
        grad_clip_callback.config = config

        assert grad_clip_callback.grad_norm == 10.0
        assert grad_clip_callback.model_key == "net"
        # Without a backward pass there is nothing to clip, and the hook materializes no gradients.
        grad_clip_callback.on_optimizer_step_begin(model)
        assert all(param.grad is None for param in model.net.parameters())


def test_grad_clip_skips_a_model_key_without_an_optimizer(monkeypatch):
    """A disabled loss term (e.g. ``gan_loss_weight_gen=0``) builds no optimizer to clip; warn once and skip it."""
    import fastgen.callbacks.grad_clip as grad_clip_module
    from fastgen.callbacks.grad_clip import GradClipCallback

    warnings = []
    monkeypatch.setattr(grad_clip_module.logger, "warning", warnings.append)
    callback = GradClipCallback(grad_norm=None, model_key="discriminator")
    model = SimpleNamespace(optimizer_dict={"net": object(), "fake_score": object()})

    for iteration in (1, 2):
        callback.on_optimizer_step_begin(model, iteration=iteration)

    assert len(warnings) == 1
    assert "'discriminator'" in warnings[0]


def _grad_clip_test_model(scaler_enabled=False):
    model = FastGenModel.__new__(FastGenModel)
    torch.nn.Module.__init__(model)
    model.net = torch.nn.Linear(2, 2)
    model.net_optimizer = torch.optim.SGD(model.net.parameters(), lr=0.1)
    model.grad_scaler = torch.amp.GradScaler("cpu", enabled=scaler_enabled, init_scale=8.0, growth_interval=1)
    model.get_lr_schedulers = lambda iteration: []
    return model


@pytest.mark.parametrize("scaler_enabled", [False, True])
def test_grad_clip_unscales_before_clipping(scaler_enabled):
    from fastgen.callbacks.grad_clip import GradClipCallback

    model = _grad_clip_test_model(scaler_enabled=scaler_enabled)
    callback = GradClipCallback(grad_norm=0.5)
    for _ in range(2):
        loss = sum(param.sum() for param in model.net.parameters())
        model.grad_scaler.scale(loss).backward()
        callback.on_optimizer_step_begin(model)
        grads = torch.cat([param.grad.flatten() for param in model.net.parameters()])
        torch.testing.assert_close(grads.norm(), torch.tensor(0.5))

        expected = [param.detach() - 0.1 * param.grad for param in model.net.parameters()]
        model.optimizers_schedulers_step(0)
        for param, value in zip(model.net.parameters(), expected):
            torch.testing.assert_close(param, value)
        model.optimizers_zero_grad(0)


def test_grad_clip_preserves_overflow_skip():
    from fastgen.callbacks.grad_clip import GradClipCallback

    model = _grad_clip_test_model(scaler_enabled=True)
    callback = GradClipCallback(posinf=0.0, neginf=0.0)
    for overflow in (True, False):
        loss = sum(param.sum() for param in model.net.parameters())
        model.grad_scaler.scale(loss).backward()
        if overflow:
            model.net.weight.grad.fill_(float("inf"))
        callback.on_optimizer_step_begin(model)
        assert all(torch.isfinite(param.grad).all() for param in model.net.parameters())

        expected = [
            param.detach().clone() if overflow else param.detach() - 0.1 * param.grad
            for param in model.net.parameters()
        ]
        previous_scale = model.grad_scaler.get_scale()
        model.optimizers_schedulers_step(0)
        for param, value in zip(model.net.parameters(), expected):
            torch.testing.assert_close(param, value)
        assert model.grad_scaler.get_scale() == previous_scale * (0.5 if overflow else 2.0)
        model.optimizers_zero_grad(0)


@pytest.mark.parametrize("overflow_key", ["fake_score", "discriminator"])
@pytest.mark.parametrize("clip_auxiliary", [False, True], ids=["fdistill", "dmd2"])
def test_shared_grad_scaler_preserves_independent_auxiliary_overflow_skip(overflow_key, clip_auxiliary):
    """Sanitizing one auxiliary root must preserve its skip while its finite sibling still updates."""
    from fastgen.callbacks.grad_clip import GradClipCallback

    model = DMD2Model.__new__(DMD2Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(student_update_freq=5, vsd_loss_weight=1.0, gan_loss_weight_gen=0.001)
    for name in ("net", "fake_score", "discriminator"):
        module = torch.nn.Linear(1, 1, bias=False)
        setattr(model, name, module)
        setattr(model, f"{name}_optimizer", torch.optim.SGD(module.parameters(), lr=0.1))
    model.grad_scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=1)
    model.get_lr_schedulers = lambda iteration: []
    callbacks = [GradClipCallback(model_key="net")]
    if clip_auxiliary:
        callbacks.extend(GradClipCallback(grad_norm=None, model_key=name) for name in ("fake_score", "discriminator"))

    for iteration, overflow in ((1, True), (2, False)):
        model.optimizers_zero_grad(iteration)
        model.grad_scaler.scale(model.fake_score.weight.sum() + model.discriminator.weight.sum()).backward()
        if overflow:
            getattr(model, overflow_key).weight.grad.fill_(float("inf"))
        before = {name: getattr(model, name).weight.detach().clone() for name in model.optimizer_dict}
        for callback in callbacks:
            callback.on_optimizer_step_begin(model, iteration)
        if clip_auxiliary:
            assert all(
                torch.isfinite(getattr(model, name).weight.grad).all() for name in ("fake_score", "discriminator")
            )

        previous_scale = model.grad_scaler.get_scale()
        model.optimizers_schedulers_step(iteration)

        for name, previous in before.items():
            should_update = name != "net" and not (overflow and name == overflow_key)
            expected = previous - 0.1 if should_update else previous
            torch.testing.assert_close(getattr(model, name).weight, expected)
        assert model.grad_scaler.get_scale() == previous_scale * (0.5 if overflow else 2.0)


@RunIf(min_gpus=1)
def test_gpu_stats_callback(get_model_data):
    model, data, config = get_model_data
    for callback_name, callback_config in GPUStats_CALLBACK.items():
        assert callback_name == "gpu_stats"
        assert "every_n" not in callback_config

        gpu_stats_callback = instantiate(callback_config)
        gpu_stats_callback.config = config
        assert gpu_stats_callback.every_n is None

        gpu_stats_callback.on_train_begin(model, iteration=0)
        gpu_stats_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=0,
        )


def test_gpu_stats_counts_a_child_that_exits_while_measured_as_zero(monkeypatch):
    """A worker that exits between listing and querying must not crash one rank while the others gather."""
    import psutil

    import fastgen.callbacks.gpu_stats as gpu_stats_module

    class _ExitedChild:
        def memory_info(self):
            raise psutil.NoSuchProcess(pid=0)

    gathered = []
    monkeypatch.setattr(psutil.Process, "children", lambda self, recursive=False: [_ExitedChild()])
    monkeypatch.setattr(gpu_stats_module, "log_prof_data", gathered.append)
    callback = gpu_stats_module.GPUStatsCallback(every_n=1)

    callback.on_training_step_end(None, data_batch=None, output_batch=None, loss_dict=None, iteration=1)

    assert gpu_stats_module._rss(_ExitedChild()) == 0
    assert gathered[0][0]["cpu_mem_gb"] > 0  # the main process still counts


def test_param_count_callback(get_model_data):
    model, data, config = get_model_data
    for callback_name, callback_config in ParamCount_CALLBACK.items():
        assert callback_name == "param_count"
        param_count_callback = instantiate(callback_config)
        param_count_callback.config = config
        param_count_callback.on_train_begin(model)


def test_train_profiler_callback(get_model_data):
    model, data, config = get_model_data
    for callback_name, callback_config in TrainProfiler_CALLBACK.items():
        assert callback_name == "train_profiler"
        assert "every_n" not in callback_config

        train_profiler_callback = instantiate(callback_config)
        train_profiler_callback.config = config
        assert train_profiler_callback.last_log_time is None
        assert train_profiler_callback.every_n is None

        train_profiler_callback.on_train_begin(model, iteration=0)
        assert train_profiler_callback.every_n == config.trainer.logging_iter
        train_profiler_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=0,
        )
        assert train_profiler_callback.last_log_time is not None


def test_train_profiler_times_each_phase_before_the_step_end_callbacks(monkeypatch):
    """Data loading sums over accumulation rounds, forward and backward passes average over them, and the
    optimizer step ends with the last round, before step-end callbacks such as W&B media logging run."""
    import time

    import fastgen.callbacks.train_profiler as profiler_module

    # Hook times of two accumulation rounds: step begin, then per round accumulation begin, backward begin,
    # (optimizer step begin,) and accumulation end.
    clock = iter([0.0, 1.0, 3.0, 6.0, 7.0, 9.0, 12.0, 13.0, 50.0])
    monkeypatch.setattr(profiler_module, "time", SimpleNamespace(perf_counter=lambda: next(clock), time=time.time))
    syncs = []
    monkeypatch.setattr(profiler_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiler_module.torch.cuda, "synchronize", lambda: syncs.append(True))
    logged = {}
    monkeypatch.setattr(profiler_module.wandb, "run", SimpleNamespace())
    monkeypatch.setattr(profiler_module.wandb, "log", lambda metrics, step: logged.update(metrics))

    profiler = profiler_module.TrainProfilerCallback(every_n=4)
    profiler.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=4))
    profiler.on_training_step_begin(None, iteration=4)
    for accum_iter in range(2):
        profiler.on_training_accum_step_begin(None, {}, iteration=4, accum_iter=accum_iter)
        profiler.on_backward_begin(None, {}, {}, {}, iteration=4, accum_iter=accum_iter)
        if accum_iter == 1:
            profiler.on_optimizer_step_begin(None, iteration=4)
        profiler.on_training_accum_step_end(None, {}, {}, iteration=4, accum_iter=accum_iter)
    profiler.on_training_step_end(None, {}, {}, {}, iteration=4)

    assert logged == {
        "profiler/data_loading_time": 2.0,  # (1 - 0) + (7 - 6)
        "profiler/avg_forward_pass_time": 2.0,  # ((3 - 1) + (9 - 7)) / 2
        "profiler/avg_backward_pass_time": 3.0,  # ((6 - 3) + (12 - 9)) / 2
        "profiler/optimizer_step_time": 1.0,  # 13 - 12
    }
    # Only logged iterations wait for the GPU.
    assert len(syncs) == 8
    profiler.on_training_step_begin(None, iteration=5)
    assert len(syncs) == 8


def test_forced_weight_norm_callback(get_model_data, monkeypatch):
    import fastgen.callbacks.forced_weight_norm as forced_weight_norm_module

    model, data, config = get_model_data
    warned = []
    monkeypatch.setattr(forced_weight_norm_module.logger, "warning", warned.append)
    for callback_name, callback_config in ForcedWeightNorm_CALLBACK.items():
        assert callback_name == "forced_weight_norm"
        forced_weight_norm_callback = instantiate(callback_config)
        forced_weight_norm_callback.config = config
        # The EDM net has no forced_weight_normalization: one warning, however often the hook runs.
        forced_weight_norm_callback.on_training_accum_step_begin(model, data)
        forced_weight_norm_callback.on_training_accum_step_begin(model, data)
        assert len(warned) == 1 and "forced_weight_normalization" in warned[0]

        net_config = EDM2_IN64_S_Config
        net_config = override_config_with_opts(net_config, ["-", "img_resolution=2", "channel_mult=[1]"])
        net = instantiate(net_config)
        model.net = net
        initial = {name: param.clone() for name, param in net.named_parameters()}

        forced_weight_norm_callback.on_training_accum_step_begin(model, data)
        normalized = {name: param.clone() for name, param in net.named_parameters()}
        assert any(not torch.equal(initial[name], param) for name, param in normalized.items())
        # Normalizing is a fixed point up to the 1e-4 eps of EDM2's normalize.
        forced_weight_norm_callback.on_training_accum_step_begin(model, data)
        for name, param in net.named_parameters():
            torch.testing.assert_close(param, normalized[name], rtol=1e-4, atol=0)
        assert len(warned) == 1


def test_wandb_callback(get_model_data):
    """``WANDB_CALLBACK`` instantiates and auto-resolves ``media_path``
    / ``logging_iter`` / ``sample_logging_iter`` from the trainer config
    in ``on_app_begin``."""
    model, _data, config = get_model_data
    config.log_config.wandb_mode = "disabled"
    for callback_name, callback_config in WANDB_CALLBACK.items():
        assert callback_name == "wandb"
        wandb_callback = instantiate(callback_config)
        wandb_callback.config = config
        wandb_callback.on_app_begin()

        # The unified callback owns its own media-dir lifecycle.
        assert wandb_callback.media_path is not None
        # Default: both cadences inherit ``trainer.logging_iter``.
        assert wandb_callback.logging_iter == config.trainer.logging_iter
        assert wandb_callback.sample_logging_iter == config.trainer.logging_iter
        wandb_callback.on_optimizer_step_begin(model)


def test_wandb_callback_cadence(get_model_data):
    """Explicit ``__init__`` values for ``logging_iter`` and
    ``sample_logging_iter`` survive ``on_app_begin`` (config only fills
    unset fields). When only ``logging_iter`` is set, the media cadence
    inherits it; when only ``sample_logging_iter`` is set, scalars still
    fall back to ``trainer.logging_iter``."""
    _model, _data, config = get_model_data
    config.log_config.wandb_mode = "disabled"
    cfg_iter = int(config.trainer.logging_iter)

    cases = [
        # (logging_iter, sample_logging_iter, expected_scalar, expected_media)
        (7, None, 7, 7),
        (None, 42, cfg_iter, 42),
        (5, 50, 5, 50),
    ]
    for log_iter, sample_iter, exp_scalar, exp_media in cases:
        cb = WandbCallback(logging_iter=log_iter, sample_logging_iter=sample_iter)
        cb.config = config
        cb.on_app_begin()
        assert cb.logging_iter == exp_scalar
        assert cb.sample_logging_iter == exp_media


def test_wandb_callback_can_disable_validation_media_without_dropping_losses(monkeypatch):
    """Release smokes suppress validation samples but retain validation scalars."""
    callback = WandbCallback(log_validation_media=False)
    stepped = []
    logged = []
    monkeypatch.setattr(callback, "_step", lambda *args, **kwargs: stepped.append((args, kwargs)))
    monkeypatch.setattr(callback, "_log_stats", lambda record, iteration, group: logged.append((iteration, group)))

    callback.on_validation_step_end(
        model=None,
        data_batch={},
        output_batch={},
        loss_dict={"loss": torch.tensor(1.0)},
        step=0,
        iteration=6,
        idx=0,
    )
    callback.on_validation_end(model=None, iteration=6, idx=0)

    assert stepped == []
    assert logged == [(6, "val0")]


@pytest.mark.parametrize("rank0", [True, False], ids=["rank0", "peer"])
def test_wandb_rank0_work_runs_inside_guards_every_rank_enters(tmp_path, monkeypatch, rank0):
    """wandb init and the log modules run on rank 0 inside ``raise_across_ranks``, which every rank enters, so
    a rank-0 failure there raises on every rank instead of leaving the peers waiting in a collective."""
    import fastgen.callbacks.wandb as wandb_callback_module
    from fastgen.callbacks.wandb import LogModule

    guarded = []

    @contextlib.contextmanager
    def recording_guard(operation):
        try:
            yield
        except Exception as error:
            guarded.append((operation, str(error)))
            raise
        guarded.append((operation, None))

    class _FailingModule(LogModule):
        def process(self, *args, **kwargs):
            raise TypeError("bad row")

    def failing_init(config):
        raise RuntimeError("no wandb")

    monkeypatch.setattr(wandb_callback_module, "raise_across_ranks", recording_guard)
    monkeypatch.setattr(wandb_callback_module, "is_rank0", lambda: rank0)
    monkeypatch.setattr(wandb_callback_module, "init_wandb", failing_init)
    callback = WandbCallback(log_modules={"bad": _FailingModule()})
    callback.config = SimpleNamespace(
        log_config=SimpleNamespace(save_path=str(tmp_path)),
        trainer=SimpleNamespace(logging_iter=1, offload_module_in_decoding=False, fsdp=False),
    )
    monkeypatch.setattr(callback, "_write_media", lambda *args: (tmp_path, []))

    def step():
        callback._step(None, {}, {}, 1, "train", "", 0, callback.loss_dict_record)

    if rank0:
        with pytest.raises(RuntimeError, match="no wandb"):
            callback.on_app_begin()
        with pytest.raises(TypeError, match="bad row"):
            step()
        assert guarded == [("wandb setup", "no wandb"), ("wandb log modules", "bad row")]
    else:
        callback.on_app_begin()
        step()
        assert guarded == [("wandb setup", None), ("wandb log modules", None)]


def test_vae_decode_context_offloads_registered_components(monkeypatch):
    import contextlib
    import fastgen.callbacks.wandb as wandb_callback_module

    class TrackedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moves = []

        def to(self, device):
            self.moves.append(torch.device(device))
            return self

    class Host(TrackedModule):
        def __init__(self, vae):
            super().__init__()
            self.preprocessors = {"vae": vae}

        def init_vae(self):
            raise AssertionError("existing VAE must be reused")

        @contextlib.contextmanager
        def vae_context(self):
            yield self.preprocessors["vae"]

    vae = TrackedModule()
    host = Host(vae)
    teacher = TrackedModule()
    text_encoder = TrackedModule()

    model = SimpleNamespace(
        module_dict={
            "net": host,
            "teacher": teacher,
            "vae": vae,
            "text_encoder": text_encoder,
        },
        device=torch.device("cuda:0"),
        net_preprocessors=host,
        precision_amp_enc=None,
    )
    monkeypatch.setattr(wandb_callback_module.torch.cuda, "memory_allocated", lambda _device: 0)
    monkeypatch.setattr(
        wandb_callback_module.basic_utils,
        "inference_mode",
        lambda **_kwargs: contextlib.nullcontext(),
    )

    with vae_decode_context(model, offload_modules=True) as yielded:
        assert yielded is vae
        assert host.moves == [torch.device("cpu")]
        assert teacher.moves == [torch.device("cpu")]
        assert text_encoder.moves == [torch.device("cpu")]
        assert vae.moves == []

    restored = [torch.device("cpu"), model.device]
    assert host.moves == restored
    assert teacher.moves == restored
    assert text_encoder.moves == restored
    assert vae.moves == []


@pytest.mark.parametrize("persistent", [False, True])
def test_vae_decode_context_prepares_each_vae_once_across_logging_calls(persistent):
    from fastgen.networks.network import FastGenNetwork

    class Host(FastGenNetwork):
        def forward(self, *args, **kwargs):
            raise NotImplementedError

        def init_vae(self):
            vae = FastGenModule()
            vae.proj = torch.nn.Linear(2, 2)
            self.preprocessors["vae"] = vae

    host = Host()
    host.prepare_module(device="cpu", precision_model=torch.float32)
    if persistent:
        host.init_vae()
        host.preprocessors["vae"].prepare_module(device="cpu", precision_model=torch.float32)
    model = SimpleNamespace(net_preprocessors=host, device=torch.device("cpu"), precision_amp_enc=None)
    borrowed = []
    for _ in range(2):
        with vae_decode_context(model) as vae:
            assert vae.is_prepared
            assert not any(parameter.is_inference() for parameter in vae.parameters())
            borrowed.append(vae)
        assert ("vae" in host.preprocessors) is persistent

    assert (borrowed[0] is borrowed[1]) is persistent


def test_vae_decode_context_cleans_up_partial_vae_initialization():
    from fastgen.networks.network import FastGenNetwork

    partial_vae = torch.nn.Identity()

    class Host(FastGenNetwork):
        def forward(self, *args, **kwargs):
            raise NotImplementedError

        def init_vae(self):
            self.preprocessors["vae"] = partial_vae
            raise RuntimeError("VAE initialization failed")

    host = Host()
    host.prepare_module(device="cpu", precision_model=torch.float32)
    model = SimpleNamespace(
        net_preprocessors=host,
        module_dict={},
        device=torch.device("cpu"),
        precision_amp_enc=None,
    )

    with pytest.raises(RuntimeError, match="VAE initialization failed"):
        with vae_decode_context(model):
            pytest.fail("the context must not yield after failed VAE initialization")

    assert "vae" not in host.preprocessors


@pytest.mark.parametrize(
    ("ddp", "fsdp", "expected"),
    [(False, False, True), (True, False, True), (False, True, False), (True, True, False)],
)
def test_wandb_decode_offload_supports_ddp_but_not_fsdp(tmp_path, monkeypatch, ddp, fsdp, expected):
    import fastgen.callbacks.wandb as wandb_callback_module

    callback = WandbCallback(media_path=tmp_path)
    callback.config = SimpleNamespace(
        log_config=SimpleNamespace(save_path=str(tmp_path)),
        trainer=SimpleNamespace(
            ddp=ddp,
            fsdp=fsdp,
            logging_iter=1,
            offload_module_in_decoding=True,
        ),
    )
    monkeypatch.setattr(wandb_callback_module, "init_wandb", lambda _config: None)

    callback.on_app_begin()

    assert callback.offload_module_in_decoding is expected


@pytest.mark.parametrize("conditional", [False, True])
def test_wandb_teacher_sampling_uses_full_batch_and_preserves_the_teacher_default(tmp_path, monkeypatch, conditional):
    """Teacher visualization mirrors full-batch student generation without inheriting its NFE."""
    import contextlib
    from types import SimpleNamespace

    import fastgen.callbacks.wandb as wandb_callback_module
    from fastgen.utils.global_vars import set_temp_global_vars

    recorded = {}

    class Model:
        teacher = object()

        @staticmethod
        def reshard_fsdp_modules():
            pass

        @staticmethod
        def sample(teacher, noise, **kwargs):
            recorded["teacher"] = teacher
            recorded["noise"] = noise
            recorded["kwargs"] = kwargs
            return torch.zeros_like(noise)

    callback = WandbCallback(log_batchsize=1)
    callback.config = SimpleNamespace(trainer=SimpleNamespace(visualize_teacher=True))
    callback.media_path = tmp_path
    monkeypatch.setattr(callback, "_do_save", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        wandb_callback_module,
        "vae_decode_context",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    input_noise = torch.randn(2, 3, 4, 4)
    condition = torch.randn(2, 5) if conditional else None
    neg_condition = torch.randn(2, 5) if conditional else None
    data_batch = {"condition": condition, "neg_condition": neg_condition} if conditional else {}
    with set_temp_global_vars({"STUDENT_STEPS": 2}):
        callback._write_media(
            Model(),
            data_batch,
            {"gen_rand": torch.zeros_like(input_noise), "input_rand": input_noise},
            iteration=1,
            group="val0",
            suffix="_0",
        )

    assert recorded["teacher"] is Model.teacher
    torch.testing.assert_close(recorded["noise"], input_noise)
    assert recorded["kwargs"]["condition"] is condition
    if conditional:
        assert recorded["kwargs"]["neg_condition"] is neg_condition
    else:
        assert "neg_condition" not in recorded["kwargs"]
    assert "sample_steps" not in recorded["kwargs"]


def test_wandb_teacher_sampling_passes_mixed_condition_structure_unchanged(tmp_path, monkeypatch):
    """Full-batch sampling needs no inference about which condition leaves are batched."""
    import contextlib
    from types import SimpleNamespace

    import fastgen.callbacks.wandb as wandb_callback_module

    recorded = {}

    class Model:
        teacher = object()

        @staticmethod
        def reshard_fsdp_modules():
            pass

        @staticmethod
        def sample(_teacher, noise, condition=None, neg_condition=None):
            del neg_condition
            recorded["condition"] = condition
            return torch.zeros_like(noise)

    callback = WandbCallback(log_batchsize=1)
    callback.config = SimpleNamespace(trainer=SimpleNamespace(visualize_teacher=True))
    callback.media_path = tmp_path
    monkeypatch.setattr(callback, "_do_save", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        wandb_callback_module,
        "vae_decode_context",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    input_noise = torch.randn(3, 3, 4, 4)
    condition = {
        "prompt_embeds": torch.randn(3, 3, 4),
        "text_token_tags": torch.ones(3, dtype=torch.long),
    }
    callback._write_media(
        Model(),
        {"condition": condition},
        {"gen_rand": torch.zeros_like(input_noise), "input_rand": input_noise},
        iteration=1,
        group="train",
        suffix="",
    )

    sampled_condition = recorded["condition"]
    assert sampled_condition is condition
    torch.testing.assert_close(sampled_condition["prompt_embeds"], condition["prompt_embeds"])
    torch.testing.assert_close(sampled_condition["text_token_tags"], condition["text_token_tags"])
    assert sampled_condition["text_token_tags"].numel() == sampled_condition["prompt_embeds"].shape[1]


def test_wandb_teacher_media_is_sliced_only_when_saved(tmp_path, monkeypatch):
    """Teacher inference may use the full batch while media still honors log_batchsize."""
    from types import SimpleNamespace

    callback = WandbCallback()
    captured = {}

    def fake_save_batch(_save_dir, kind, tensor, *_args, **_kwargs):
        captured[kind] = tensor
        return []

    monkeypatch.setattr(callback, "_save_batch", fake_save_batch)
    model = SimpleNamespace(
        net=SimpleNamespace(),
        net_preprocessors=SimpleNamespace(preprocessors={}),
        precision_amp_enc=None,
    )
    teacher = torch.randn(3, 3, 4, 4)

    callback._do_save(tmp_path, model, {}, {"gen_teacher": teacher}, n_per_rank=1)

    torch.testing.assert_close(captured["teacher"], teacher[:1])


def test_wandb_reshards_between_fsdp_inference_phases(tmp_path, monkeypatch):
    """Student, teacher, and batched VAE work have explicit FSDP phase boundaries."""
    import contextlib

    import fastgen.callbacks.wandb as wandb_callback_module

    events = []

    class Model:
        teacher = object()
        uses_fsdp = True

        @staticmethod
        def reshard_fsdp_modules():
            events.append("reshard")

        @staticmethod
        def sample(_teacher, noise, condition=None, neg_condition=None):
            del condition, neg_condition
            events.append("teacher")
            return torch.zeros_like(noise)

    @contextlib.contextmanager
    def fake_vae_decode_context(*_args, **_kwargs):
        events.append("vae_enter")
        yield
        events.append("vae_exit")

    callback = WandbCallback(log_batchsize=1)
    callback.config = SimpleNamespace(trainer=SimpleNamespace(visualize_teacher=True))
    callback.media_path = tmp_path
    monkeypatch.setattr(wandb_callback_module, "vae_decode_context", fake_vae_decode_context)
    monkeypatch.setattr(callback, "_do_save", lambda *_args, **_kwargs: events.append("vae") or [])
    input_noise = torch.randn(1, 3, 4, 4)

    def sample_student():
        events.append("student")
        return torch.zeros_like(input_noise)

    callback._write_media(
        Model(),
        {"condition": torch.randn(1, 5)},
        {"gen_rand": sample_student, "input_rand": input_noise},
        iteration=1,
        group="val0",
        suffix="_0",
    )

    assert events == [
        "student",
        "reshard",
        "teacher",
        "reshard",
        "vae_enter",
        "vae",
        "vae_exit",
        "reshard",
    ]


def test_wandb_callback_source_panel(tmp_path):
    """Raw reference pixels are displayed directly; encoded references are decoded."""

    decode_calls = []

    class _StubVAE:
        def decode(self, x):
            decode_calls.append(x)
            # Latent -> RGB, mirroring a real VAE decode.
            return torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])

    class _StubNet:
        preprocessors = {"vae": _StubVAE()}

    class _StubModel:
        net = _StubNet()
        net_preprocessors = net  # the VAE is read off the preprocessor host
        precision_amp_enc = None
        device = torch.device("cpu")

    callback = WandbCallback()
    callback.media_path = tmp_path
    model = _StubModel()

    pixel = torch.zeros(1, 3, 8, 8)
    latent = torch.zeros(1, 16, 8, 8)

    pix_dir = tmp_path / "iter0_pixel"
    pix_dir.mkdir()
    entries = callback._do_save(pix_dir, model, {"condition_image_raw": pixel}, {}, n_per_rank=1)
    src = [e for e in entries if e.kind == "source"]
    assert len(src) == 1
    assert src[0].path.exists()
    assert src[0].path.parent == pix_dir / "data" / "source"
    assert decode_calls == []

    lat_dir = tmp_path / "iter0_latent"
    lat_dir.mkdir()
    entries = callback._do_save(lat_dir, model, {"condition_image": latent}, {}, n_per_rank=1)
    src = [e for e in entries if e.kind == "source"]
    assert len(src) == 1
    assert src[0].path.exists()
    assert len(decode_calls) == 1
    torch.testing.assert_close(decode_calls[0], latent)

    both_dir = tmp_path / "iter0_both"
    both_dir.mkdir()
    entries = callback._do_save(
        both_dir,
        model,
        {"condition_image_raw": pixel, "condition_image": latent},
        {},
        n_per_rank=1,
    )
    assert len([entry for entry in entries if entry.kind == "source"]) == 1
    assert len(decode_calls) == 1

    empty_dir = tmp_path / "iter0_empty"
    empty_dir.mkdir()
    entries = callback._do_save(empty_dir, model, {}, {}, n_per_rank=1)
    assert not any(e.kind == "source" for e in entries)


def test_wandb_max_ranks_filters_entries_after_all_rank_write(tmp_path, monkeypatch):
    """With a shared nonempty schema, every rank decodes/writes; max_ranks only filters W&B."""
    import contextlib

    import fastgen.callbacks.wandb as wandb_callback_module

    decode_calls = []

    class _StubVAE(torch.nn.Module):
        def decode(self, x):
            decode_calls.append(x)
            return torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])

    vae = _StubVAE()
    net = SimpleNamespace(preprocessors={"vae": vae})
    model = SimpleNamespace(
        net=net,
        net_preprocessors=net,
        precision_amp_enc=None,
        reshard_fsdp_modules=lambda preprocessors_only=False: None,
    )
    callback = WandbCallback(max_ranks=1)
    callback.config = SimpleNamespace(trainer=SimpleNamespace(visualize_teacher=False))
    callback.media_path = tmp_path
    monkeypatch.setattr(wandb_callback_module, "get_rank", lambda: 1)
    monkeypatch.setattr(
        wandb_callback_module,
        "vae_decode_context",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    save_dir, entries = callback._write_media(
        model,
        {},
        {"gen_rand": torch.zeros(1, 16, 8, 8)},
        iteration=1,
        group="val0",
        suffix="_0",
    )

    assert entries == []
    assert len(decode_calls) == 1
    assert (save_dir / "student" / "generation" / "frame_001_000.png").exists()


class _WandbFSDPVAE(FastGenModule):
    """Tiny VAE whose decode enters a real FSDP2 root forward."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(4).reshape(4, 4, 1, 1))
        self.decode_calls = 0

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = torch.nn.functional.conv2d(latent.to(dtype=self.precision_model), self.weight)
        return decoded[:, :3].to(dtype=latent.dtype)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        # Calling the sharded root, rather than its forward method directly, is what
        # exercises FSDP2's collective parameter all-gather on every logging rank.
        return self(latent)


class _WandbFSDPVAEHost:
    def __init__(self, vae: _WandbFSDPVAE):
        self.preprocessors = {"vae": vae}

    def init_vae(self):
        raise AssertionError("the persistent sharded VAE must be reused")

    @contextlib.contextmanager
    def vae_context(self):
        yield self.preprocessors["vae"]


def _wandb_fsdp_max_ranks_impl(rank: int, world_size: int, media_path: str) -> dict:
    from unittest.mock import patch

    from fastgen.utils import media_utils

    device = torch.device("cuda", rank)
    vae = _WandbFSDPVAE()
    fsdp_context = create_fsdp_context(
        reduce_dtype=torch.float32,
        min_num_params=1,
        apply_cpu_offload=False,
    )
    vae.prepare_module(
        device=device,
        precision_model=torch.float32,
        fsdp_context=fsdp_context,
    )
    assert isinstance(vae, FSDPModule)

    host = _WandbFSDPVAEHost(vae)
    model = SimpleNamespace(
        net=host,
        net_preprocessors=host,
        module_dict={"vae": vae},
        uses_fsdp=True,
        device=device,
        precision_amp_enc=None,
        reshard_fsdp_modules=lambda preprocessors_only=False: reshard_fsdp_modules((vae,)),
    )
    callback = WandbCallback(media_path=media_path, max_ranks=1, log_batchsize=1)
    callback.config = SimpleNamespace(trainer=SimpleNamespace(visualize_teacher=False))
    callback.trainer = SimpleNamespace(fsdp_context=fsdp_context)

    local_result = {}
    write_media = callback._write_media
    save_image = media_utils.save_image

    def delayed_save(*args, **kwargs):
        # The entry gather must wait for a slow writer before rank 0 reads anyone's files.
        if rank == 1:
            import time

            time.sleep(0.5)
        return save_image(*args, **kwargs)

    def record_media(*args, **kwargs):
        save_dir, entries = write_media(*args, **kwargs)
        own_path = save_dir / "student" / "generation" / f"frame_{rank:03d}_000.png"
        local_result.update(
            decode_calls=vae.decode_calls,
            own_file_exists=own_path.exists(),
            entry_count=len(entries),
            entry_ranks=[entry.rank for entry in entries],
        )
        return save_dir, entries

    def inspect_panels(entries, *args):
        generation_dir = callback._current_dir / "student" / "generation"
        local_result["visible_file_count"] = len(list(generation_dir.glob("frame_*_000.png")))
        local_result["uploaded_ranks"] = [entry.rank for entry in entries]

    callback._write_media = record_media
    callback._log_panels = inspect_panels
    with patch.object(media_utils, "save_image", side_effect=delayed_save):
        callback._step(
            model,
            {},
            {"gen_rand": torch.full((1, 4, 4, 4), float(rank), device=device)},
            iteration=1,
            group="fsdp",
            suffix="",
            step=0,
            record=callback.loss_dict_record,
        )

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_result)
    return {"per_rank": gathered}


@RunIf(min_gpus=2)
def test_wandb_fsdp_max_ranks_only_filters_upload_entries(tmp_path):
    """A shared nonempty schema enters the sharded VAE everywhere; max_ranks filters uploads."""
    result = run_distributed_test(
        test_fn=_wandb_fsdp_max_ranks_impl,
        world_size=2,
        timeout=240,
        media_path=str(tmp_path / "wandb_fsdp_media"),
    )

    assert result is not None
    per_rank = result["per_rank"]
    assert [rank_result["decode_calls"] for rank_result in per_rank] == [1, 1]
    assert all(rank_result["own_file_exists"] for rank_result in per_rank)
    assert per_rank[0]["visible_file_count"] == 2
    assert per_rank[0]["uploaded_ranks"] == [0]
    assert [rank_result["entry_count"] for rank_result in per_rank] == [1, 0]
    assert [rank_result["entry_ranks"] for rank_result in per_rank] == [[0], []]


def test_vlm_t2i_callback(get_model_data, monkeypatch):
    """``VLM_IMG_CALLBACK`` builds a ``WandbCallback`` (with
    ``max_ranks=4``) whose ``log_modules`` is a dict keyed by ``"vlm"``
    with a single ``VLMLogger``. ``on_app_begin`` wires up the module
    on rank 0. ``master_prompt`` stays ``None`` — the default rubric is
    resolved per call by ``score_sample`` via
    ``DEFAULT_MASTER_PROMPTS[sample.media_type]``. No network call —
    scoring is exercised by ``test_logger.py``."""
    _model, _data, config = get_model_data
    config.log_config.wandb_mode = "disabled"
    monkeypatch.setenv("VLM_BASE_URL", "http://stub")
    assert set(VLM_IMG_CALLBACK.keys()) == {"wandb"}
    wandb_callback = instantiate(VLM_IMG_CALLBACK["wandb"])
    wandb_callback.config = config
    wandb_callback.on_app_begin()
    assert wandb_callback.max_ranks == 4
    assert set(wandb_callback.log_modules) == {"vlm"}
    vlm_logger = wandb_callback.log_modules["vlm"]
    assert vlm_logger.media_mode == "image"
    assert vlm_logger.master_prompt is None  # default rubric resolved in score_sample


def test_callback_list(get_model_data):
    model, data, config = get_model_data
    config.trainer.callbacks = DictConfig({**GradClip_CALLBACK, **ParamCount_CALLBACK})
    config.trainer.callbacks.update({**ForcedWeightNorm_CALLBACK})

    trainer = Trainer(config)
    callbacks = CallbackDict(config=config, trainer=trainer)
    assert len(callbacks._callbacks) == 3
    callbacks.on_train_begin(model)


def test_wandb_extract_prompt_uses_raw_condition_namespace():
    from fastgen.callbacks.wandb import extract_prompt

    assert extract_prompt({"condition_raw": ["raw"], "condition": ["encoded fallback"]}) == "raw"
    assert extract_prompt({"condition": ["wrongly namespaced raw text"]}) == ""
    assert extract_prompt({"caption": ["alias"]}) == ""
    assert extract_prompt({"prompt": ["alias"]}) == ""


def test_carry_callback_overlays_each_preprocessed_accumulation_slot():
    """An active carry replaces a fresh preprocessed batch independently per accumulation slot.

    Dispatch asserts every registered callback has the hook, so the plain ``Callback`` here
    (which only inherits the base no-op ``on_training_accum_step_end``) also proves that method
    exists on the base -- without it, dispatching the end hook would trip the ``hasattr`` assert.
    """
    import types

    from fastgen.callbacks.callback import Callback, CallbackDict
    from fastgen.callbacks.carry import CARRY_OUTPUT_KEY, CARRY_OVERLAY_KEYS, CarryCallback
    from fastgen.configs.instantiate import LazyCall as L

    # FastGen loaders normalize text to ``condition`` / ``condition_raw``;
    # aliases are not part of the canonical model/logger/carry contract.
    assert "condition" in CARRY_OVERLAY_KEYS
    assert "condition_raw" in CARRY_OVERLAY_KEYS
    assert "condition_image" in CARRY_OVERLAY_KEYS
    assert "condition_image_raw" in CARRY_OVERLAY_KEYS
    assert "caption" not in CARRY_OVERLAY_KEYS
    assert "prompt" not in CARRY_OVERLAY_KEYS

    callbacks_cfg = {
        "base": L(Callback)(),
        "carry": L(CarryCallback)(),
    }
    config = types.SimpleNamespace(trainer=types.SimpleNamespace(callbacks=callbacks_cfg))
    callbacks = CallbackDict(config=config, trainer=None)

    states = {0: torch.tensor([10.0]), 1: torch.tensor([20.0])}
    prompts = {0: ["prompt zero"], 1: ["prompt one"]}

    # Iteration 1: nothing to inject yet; the end hook captures each complete carry.
    for accum in (0, 1):
        data = {"condition_raw": prompts[accum], "idx": [f"sample-{accum}"]}
        callbacks.on_training_accum_step_begin(None, data, iteration=1, accum_iter=accum)
        assert "state_carried" not in data, "nothing to carry on the first iteration"
        carry = {key: data[key] for key in CARRY_OVERLAY_KEYS if key in data}
        carry.update({"state_carried": states[accum], "step_carried": 1})
        callbacks.on_training_accum_step_end(None, data, {CARRY_OUTPUT_KEY: carry}, iteration=1, accum_iter=accum)

    # Iteration 2: each active slot atomically replaces its already-preprocessed fresh batch.
    for accum in (0, 1):
        data = {
            "condition_raw": ["wrong fresh prompt"],
            "idx": ["wrong-fresh-id"],
            "fresh_only": object(),
        }
        callbacks.on_training_accum_step_begin(None, data, iteration=2, accum_iter=accum)
        assert "fresh_only" not in data
        assert torch.equal(data["state_carried"], states[accum])
        assert data["condition_raw"] is prompts[accum]
        assert data["idx"] == [f"sample-{accum}"]

        # Replacing the active trajectory with its boundary marker drops every omitted canonical
        # field from the slot, while leaving this iteration's already-overlaid data intact for loggers.
        callbacks.on_training_accum_step_end(
            None,
            data,
            {
                CARRY_OUTPUT_KEY: {
                    "step_carried": 4,
                }
            },
            iteration=2,
            accum_iter=accum,
        )
        assert data["condition_raw"] is prompts[accum]

    # The next fresh batch is loaded and gets only the terminal marker; stale prompt/identity
    # metadata cannot leak across the completed trajectory boundary.
    for accum in (0, 1):
        data = {"condition_raw": ["new prompt"], "idx": ["new-id"]}
        callbacks.on_training_accum_step_begin(None, data, iteration=3, accum_iter=accum)
        assert "state_carried" not in data and data["step_carried"] == 4
        assert data["condition_raw"] == ["new prompt"] and data["idx"] == ["new-id"]


def test_carry_callback_advances_validation_after_each_batch_is_preprocessed():
    """Opt-in validation carry overlays already-preprocessed batches."""
    from fastgen.callbacks.carry import CARRY_OUTPUT_KEY, CarryCallback

    callback = CarryCallback(enable_validation=True)
    state = torch.tensor([3.0])

    callback.on_validation_begin(None, iteration=10, idx=0)

    first = {"condition": "first", "idx": [0]}
    callback.on_validation_step_begin(None, first, step=0, iteration=10, idx=0)
    assert first["condition"] == "first"
    callback.on_validation_step_end(
        None,
        first,
        {CARRY_OUTPUT_KEY: {"condition": "first", "idx": [0], "state_carried": state, "step_carried": 1}},
        {},
        step=0,
        iteration=10,
        idx=0,
    )

    second = {"condition": "discarded", "idx": [1], "fresh_only": object()}
    callback.on_validation_step_begin(None, second, step=1, iteration=10, idx=0)
    assert second["state_carried"] is state and second["condition"] == "first"
    assert "fresh_only" not in second
    callback.on_validation_step_end(
        None,
        second,
        {CARRY_OUTPUT_KEY: {"step_carried": 2}},
        {},
        step=1,
        iteration=10,
        idx=0,
    )

    third = {"condition": "second", "idx": [2]}
    callback.on_validation_step_begin(None, third, step=2, iteration=10, idx=0)
    assert third == {"condition": "second", "idx": [2], "step_carried": 2}

    callback.on_validation_end(None, iteration=10, idx=0)
    reset = {"condition": "next pass"}
    callback.on_validation_step_begin(None, reset, step=0, iteration=10, idx=1)
    assert reset == {"condition": "next pass"}


def test_carry_callback_leaves_validation_batches_independent_by_default():
    """The default callback ignores validation carry outputs."""
    from fastgen.callbacks.carry import CARRY_OUTPUT_KEY, CarryCallback

    callback = CarryCallback()
    assert callback.enable_validation is False

    callback.on_validation_begin(None, iteration=10, idx=0)
    first = {"condition": "first", "idx": [0]}
    callback.on_validation_step_end(
        None,
        first,
        {CARRY_OUTPUT_KEY: {"condition": "first", "state_carried": torch.tensor([1.0])}},
        {},
        step=0,
        iteration=10,
        idx=0,
    )

    second = {"condition": "second", "idx": [1]}
    callback.on_validation_step_begin(None, second, step=1, iteration=10, idx=0)
    assert second == {"condition": "second", "idx": [1]}
    assert callback._validation_slot is None


# ---------------------------------------------------------------------------
# RankConsistencyCallback
# ---------------------------------------------------------------------------


def test_rank_consistency_callback_gates_and_selects(monkeypatch):
    """The check runs every ``every_n`` iterations on the selected roots only, and never single-process."""
    import fastgen.callbacks.rank_consistency as rank_consistency_module
    from fastgen.callbacks.rank_consistency import RankConsistencyCallback

    checked = []
    monkeypatch.setattr(rank_consistency_module, "world_size", lambda: 2)
    monkeypatch.setattr(
        rank_consistency_module,
        "validate_module_values_across_ranks",
        lambda module, **kwargs: checked.append((module, kwargs["parameters"], kwargs["operation"])),
    )
    net, teacher = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    model = SimpleNamespace(module_dict={"net": net, "teacher": teacher})

    callback = RankConsistencyCallback(every_n=10, modules=["teacher"])
    callback.on_train_begin(model)
    callback.on_training_step_end(model, {}, {}, {}, iteration=5)
    assert checked == []
    callback.on_training_step_end(model, {}, {}, {}, iteration=10)
    assert [(module is teacher, parameters) for module, parameters, _ in checked] == [(True, False)]
    assert "'teacher'" in checked[0][2] and "iteration 10" in checked[0][2]

    checked.clear()
    RankConsistencyCallback(every_n=1, check_parameters=True).on_training_step_end(model, {}, {}, {}, iteration=3)
    assert [(module, parameters) for module, parameters, _ in checked] == [(net, True), (teacher, True)]

    monkeypatch.setattr(rank_consistency_module, "world_size", lambda: 1)
    checked.clear()
    RankConsistencyCallback(every_n=1).on_training_step_end(model, {}, {}, {}, iteration=1)
    assert checked == []

    with pytest.raises(ValueError, match="not registered in module_dict"):
        RankConsistencyCallback(modules=["missing"]).on_train_begin(model)
    with pytest.raises(ValueError, match="every_n must be positive"):
        RankConsistencyCallback(every_n=0)


def test_rank_consistency_callback_preflights_canonical_root_plan(monkeypatch):
    """Root collectives start only after every rank agrees on the full sorted schedule."""
    import fastgen.callbacks.rank_consistency as rank_consistency_module
    from fastgen.callbacks.rank_consistency import RankConsistencyCallback

    plans = []
    checked = []
    source_plan = [None]

    def validate_plan(plan, operation):
        plans.append((plan, operation))
        if source_plan[0] is not None and plan != source_plan[0]:
            raise RuntimeError(f"{operation} mismatch with rank 0")

    monkeypatch.setattr(rank_consistency_module, "world_size", lambda: 2)
    monkeypatch.setattr(rank_consistency_module, "validate_object_across_ranks", validate_plan)
    monkeypatch.setattr(
        rank_consistency_module,
        "validate_module_values_across_ranks",
        lambda module, **kwargs: checked.append((module, kwargs["operation"])),
    )

    net, teacher = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    # Deliberately use non-canonical insertion order.
    model = SimpleNamespace(module_dict={"teacher": teacher, "net": net})
    callback = RankConsistencyCallback(every_n=1)
    callback.on_train_begin(model)

    expected = {"roots": ("net", "teacher"), "missing": (), "every_n": 1, "check_parameters": False}
    assert plans == [(expected, "rank consistency callback initialization root-plan validation")]

    plans.clear()
    callback.on_training_step_end(model, {}, {}, {}, iteration=4)

    assert plans == [(expected, "rank consistency check at iteration 4 root-plan validation")]
    assert [module for module, _operation in checked] == [net, teacher]

    checked.clear()
    source_plan[0] = {**expected, "roots": ("net",)}
    with pytest.raises(RuntimeError, match="root-plan validation mismatch"):
        callback.on_training_step_end(model, {}, {}, {}, iteration=5)
    assert checked == []


def _rank_consistency_callback_impl(rank: int, world_size: int) -> dict:
    """Equal state passes, shards are skipped, and a fingerprint mismatch raises on every rank."""
    from fastgen.callbacks.rank_consistency import RankConsistencyCallback

    def every_rank(flag: bool) -> bool:
        agreed = torch.tensor([1 if flag else 0], device="cuda")
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
        return bool(agreed.item())

    device = torch.device("cuda", rank)
    net = torch.nn.Linear(4, 4).to(device)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.fill_(0.5)  # identical replicated parameters
    teacher = torch.nn.Module()
    teacher.register_buffer("table", torch.arange(4.0, device=device))
    teacher.register_buffer("cache", torch.ones(2, device=device), persistent=False)
    mesh = init_device_mesh("cuda", (world_size,))
    sharded = torch.nn.Module()
    sharded.weight = torch.nn.Parameter(
        distribute_tensor(torch.arange(float(4 * world_size), device=device).reshape(world_size, 4), mesh, [Shard(0)])
    )
    roots = {"net": net, "teacher": teacher, "sharded": sharded}
    if rank == 1:
        roots = {name: roots[name] for name in reversed(tuple(roots))}
    model = SimpleNamespace(module_dict=roots)

    callback = RankConsistencyCallback(every_n=1, check_parameters=True)
    callback.on_train_begin(model)
    callback.on_training_step_end(model, {}, {}, {}, iteration=1)  # local shards differ by design: skipped

    settings_rejected = {}
    for field, value in (("every_n", rank + 1), ("check_parameters", bool(rank))):
        mismatched = RankConsistencyCallback(**{field: value})
        try:
            mismatched.on_train_begin(model)
        except RuntimeError as error:
            settings_rejected[field] = "root-plan validation mismatch" in str(error) and field in str(error)
        else:
            settings_rejected[field] = False

    if rank == 1:
        teacher.cache[0] = 2.0  # one changed element avoids cancellation in the XOR fingerprint
    try:
        callback.on_training_step_end(model, {}, {}, {}, iteration=2)
        message = ""
    except RuntimeError as error:
        message = str(error)
    rejected = "'fingerprints'" in message and "'teacher'" in message and "rank 1" in message
    expected = torch.tensor([2.0 if rank == 1 else 1.0, 1.0], device=device)

    if rank == 1:
        model.module_dict.pop("sharded")
    try:
        callback.on_training_step_end(model, {}, {}, {}, iteration=3)
        root_plan_message = ""
    except RuntimeError as error:
        root_plan_message = str(error)
    root_plan_rejected = "root-plan validation mismatch" in root_plan_message and "'roots'" in root_plan_message

    configured_callback = RankConsistencyCallback(every_n=1, modules=["sharded"])
    try:
        configured_callback.on_train_begin(model)
        selection_message = ""
    except RuntimeError as error:
        selection_message = str(error)
    missing_selection_rejected = (
        "root-plan validation" in selection_message
        and "rank 1" in selection_message
        and "'missing'" in selection_message
    )
    return {
        "rejected_collectively": every_rank(rejected),
        "check_did_not_mutate": every_rank(torch.equal(teacher.cache, expected)),
        "root_plan_rejected_collectively": every_rank(root_plan_rejected),
        "missing_selection_rejected_collectively": every_rank(missing_selection_rejected),
        **{f"{field}_rejected_collectively": every_rank(rejected) for field, rejected in settings_rejected.items()},
    }


@RunIf(min_gpus=2)
def test_rank_consistency_callback_detects_drift_across_ranks():
    result = run_distributed_test(test_fn=_rank_consistency_callback_impl, world_size=2, timeout=240)
    assert result is not None, "distributed test returned nothing (a rank likely crashed)"
    assert all(result.values()), result
