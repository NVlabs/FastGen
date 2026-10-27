# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``FastGenModel`` base class (``fastgen/methods/model.py``).

The model resolves the run's precisions and dispatches roles onto the ``FastGenModule`` boundary:
the trainable student, frozen auxiliaries, EMA copies, preprocessors, and ``extra_fsdp_modules``
through ``prepare_module`` / ``on_train_begin`` / ``init_preprocessors``, plus the gradient scaler,
AMP scopes, pretrained-weight loading, and the reward path. The module boundary itself is tested in
``tests/test_fastgen_module.py``; concrete methods are tested in the ``tests/test_*model.py`` files.
"""

import contextlib
import io
import warnings
from types import SimpleNamespace

import attrs
import pytest
import torch

from fastgen.configs.config import BaseModelConfig
from fastgen.methods.model import FastGenModel
from fastgen.networks.discriminators import Discriminator_Audio2D
from fastgen.networks.lora_utils import LoRAMixin
from fastgen.networks.network import FastGenNetwork, FastGenModule
from tests.utils import RunIf, run_distributed_test


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _TinyNetwork(FastGenNetwork):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def forward(self, x_t, t, **kwargs):
        del t, kwargs
        return self.proj(x_t.to(dtype=self.precision_model)).to(dtype=x_t.dtype)


class _TinyNet(torch.nn.Module):
    """Minimal module for pretrained-weight loading tests."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


class _PlainPreprocessor(FastGenModule):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Linear(2, 2)


class _FP32Preprocessor(_PlainPreprocessor):
    @property
    def precision_model(self):
        return torch.float32


class _PreprocessorNetwork(_TinyNetwork):
    def init_preprocessors(self):
        for name in ("vae", "text_encoder", "image_encoder"):
            if name not in self.preprocessors:
                self.preprocessors[name] = _PlainPreprocessor()


class _TinyModel(FastGenModel):
    def _get_outputs(self, *args, **kwargs):
        raise NotImplementedError

    def single_train_step(self, *args, **kwargs):
        raise NotImplementedError


class _RecordingFSDPContext:
    """CPU test double for a logical root's one-shot FSDP preparation."""

    device = torch.device("cpu")

    def __init__(self):
        self.modules = []
        self.calls = []

    def _shard_module(self, module):
        self.calls.append(module)
        self.modules.append(module)
        return module


def _preprocessor_initialization_failures_worker(rank, world_size):
    import torch.distributed as dist

    for failure in ("construction", "shared_root", "invalid_type", "disabled"):
        model = _uninitialized_model()
        model.net = _TinyNetwork()
        model.module_dict["net"] = model.net
        model.config.enable_preprocessors = not (rank == 1 and failure == "disabled")
        prepare_calls = []
        model.prepare_module = lambda *args, **kwargs: prepare_calls.append(True)

        def init_preprocessors():
            if rank == 1 and failure == "construction":
                raise FileNotFoundError("missing preprocessor weights")
            preprocessor = _PlainPreprocessor()
            if rank == 1 and failure == "shared_root":
                preprocessor.core = model.net.proj
            if rank == 1 and failure == "invalid_type":
                preprocessor = torch.nn.Linear(2, 2)
            model.net.preprocessors["vae"] = preprocessor

        model.net.init_preprocessors = init_preprocessors
        expected = "preprocessor preparation plan mismatch" if failure == "disabled" else "preprocessor initialization"
        with pytest.raises(RuntimeError, match=expected):
            model.init_preprocessors()
        checks = [None] * world_size
        dist.all_gather_object(checks, not prepare_calls)
        assert all(checks), (failure, checks)
    return True


@RunIf(min_gpus=2)
def test_preprocessor_initialization_failures_propagate_before_preparation():
    assert run_distributed_test(
        _preprocessor_initialization_failures_worker, world_size=2, timeout=120, nccl_timeout=30
    )


def _uninitialized_model() -> _TinyModel:
    model = _TinyModel.__new__(_TinyModel)
    torch.nn.Module.__init__(model)
    # ``model_dict`` reads ``net``; tests that need a real root assign one themselves.
    model.net = None
    model.module_dict = {}
    model.config = SimpleNamespace(inference=False, extra_fsdp_modules=[], autocast_cache_enabled=False)
    model.use_ema = []
    model.precision_opt = torch.float32
    return model


def _reference_state_dict() -> dict:
    net = _TinyNet()
    torch.nn.init.constant_(net.lin.weight, 0.5)
    torch.nn.init.constant_(net.lin.bias, -0.25)
    return net.state_dict()


def _zeroed_net() -> _TinyNet:
    net = _TinyNet()
    torch.nn.init.zeros_(net.lin.weight)
    torch.nn.init.zeros_(net.lin.bias)
    return net


# ---------------------------------------------------------------------------
# Student rollout policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("student_sample_type", ["euler", "0.5", -1.0, None])
def test_init_rejects_an_unknown_student_sample_type_before_building(student_sample_type):
    config = BaseModelConfig(device="cpu", student_sample_type=student_sample_type)
    with pytest.raises(ValueError, match="_TinyModel student_sample_type must be one of"):
        _TinyModel(config)


# ---------------------------------------------------------------------------
# Precision resolution and AMP scopes
# ---------------------------------------------------------------------------


def test_precision_model_none_defaults_to_method_precision():
    model = _uninitialized_model()
    model._resolve_precisions(precision="float32", precision_model=None)
    assert model.precision_model == model.precision == torch.float32


@pytest.mark.parametrize("name", ["precision_amp", "precision_amp_infer", "precision_amp_enc"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_main_amp_policies_accept_full_precision_cuda_autocast_targets(name, precision):
    model = _uninitialized_model()
    model._resolve_precisions(**{name: precision})
    assert getattr(model, name) == getattr(torch, precision)


def test_autocast_weight_cache_is_off_by_default():
    """The config comment on ``autocast_cache_enabled`` records why the weight cache is off."""
    assert attrs.fields(BaseModelConfig).autocast_cache_enabled.default is False


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_training_autocast_weight_cache_follows_config(cache_enabled):
    """``model.autocast()`` forwards ``autocast_cache_enabled`` and nested regions inherit it."""
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model._resolve_precisions(precision_amp="bfloat16")
    model.config.autocast_cache_enabled = cache_enabled
    assert torch.is_autocast_cache_enabled()  # torch's global default outside any region
    with model.autocast():
        assert torch.is_autocast_enabled("cpu")
        assert torch.is_autocast_cache_enabled() is cache_enabled
        # Nested regions (JVP, inference, network-local autocast) copy the current cache flag.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            assert torch.is_autocast_cache_enabled() is cache_enabled
    assert torch.is_autocast_cache_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_autocast_supports_full_precision_targets(dtype):
    value = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    linear = torch.nn.Linear(8, 8, device="cuda", dtype=torch.bfloat16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert (value @ value).dtype == torch.bfloat16
            assert linear(value).dtype == torch.bfloat16

            with torch.autocast("cuda", dtype=dtype):
                assert torch.is_autocast_enabled("cuda")
                assert torch.get_autocast_dtype("cuda") == dtype
                assert (value @ value).dtype == dtype
                assert linear(value).dtype == dtype

            assert torch.get_autocast_dtype("cuda") == torch.bfloat16
            assert (value @ value).dtype == torch.bfloat16
            assert linear(value).dtype == torch.bfloat16

    assert not caught


def test_model_sample_owns_inference_amp_and_does_not_forward_it():
    class _Sampler(_TinyNetwork):
        def sample(self, noise, guidance_cfg=None):
            assert guidance_cfg is None
            self.cpu_amp_enabled = torch.is_autocast_enabled("cpu")
            return noise

    model = _uninitialized_model()
    model.precision_amp_infer = torch.bfloat16
    model.config = SimpleNamespace(guidance_cfg=None)
    network = _Sampler()

    result = model.sample(network, torch.ones(1, 2))

    assert network.cpu_amp_enabled
    assert result.dtype == torch.float32


@pytest.mark.parametrize("name", ["precision_amp", "precision_amp_infer"])
def test_model_sample_does_not_accept_amp_kwargs(name):
    class _Sampler(_TinyNetwork):
        def sample(self, noise, guidance_cfg=None):
            del guidance_cfg
            return noise

    model = _uninitialized_model()
    model.precision_amp_infer = None
    model.config = SimpleNamespace(guidance_cfg=None)

    with pytest.raises(TypeError, match=name):
        model.sample(_Sampler(), torch.ones(1, 2), **{name: torch.bfloat16})


# ---------------------------------------------------------------------------
# Pretrained model loading
# ---------------------------------------------------------------------------


def test_load_pretrained_model_from_local_path(tmp_path):
    """A local path loads in place and creates no cache directory."""
    expected = _reference_state_dict()
    checkpoint_path = tmp_path / "model.pth"
    torch.save(expected, checkpoint_path)

    net = _zeroed_net()
    FastGenModel._load_pretrained_model(net, str(checkpoint_path), device="cpu")

    assert torch.equal(net.lin.weight, expected["lin.weight"])
    assert torch.equal(net.lin.bias, expected["lin.bias"])


def test_load_pretrained_model_requires_at_least_one_matching_key(tmp_path):
    checkpoint_path = tmp_path / "unrelated.pth"
    torch.save({"unrelated.weight": torch.ones(1)}, checkpoint_path)

    with pytest.raises(RuntimeError, match="matched no target parameters"):
        FastGenModel._load_pretrained_model(_zeroed_net(), str(checkpoint_path), device="cpu")


def test_load_pretrained_model_caches_s3_download(tmp_path, monkeypatch):
    """S3 weights are cached under ``FASTGEN_OUTPUT_ROOT/MODEL`` and reused."""
    expected = _reference_state_dict()
    payload = io.BytesIO()
    torch.save(expected, payload)
    fetched = []

    def fake_download(path, output):
        fetched.append(path)
        output.write(payload.getvalue())
        return 0

    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr("fastgen.utils.s3.resumable_download", fake_download)

    net = _zeroed_net()
    FastGenModel._load_pretrained_model(net, "s3://bucket/model.pth", device="cpu")
    assert torch.equal(net.lin.weight, expected["lin.weight"])
    assert fetched == ["s3://bucket/model.pth"]
    # MODEL matches the ``CKPT_ROOT_DIR`` default in ``fastgen.configs.net``.
    assert (tmp_path / "MODEL" / "bucket" / "model.pth").is_file(), "download should be cached"

    net = _zeroed_net()
    FastGenModel._load_pretrained_model(net, "s3://bucket/model.pth", device="cpu")
    assert torch.equal(net.lin.weight, expected["lin.weight"])
    assert fetched == ["s3://bucket/model.pth"], "cache hit must not re-download from S3"


def test_load_pretrained_model_caches_http_download(tmp_path, monkeypatch):
    """Single-file HTTP model sources are mirrored once and then loaded from the shared cache."""
    expected = _reference_state_dict()
    payload = io.BytesIO()
    torch.save(expected, payload)
    url = "https://example.com/model.pt?revision=immutable"
    fetched = []

    def fake_download(path, output):
        fetched.append(path)
        output.write(payload.getvalue())

    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr("fastgen.utils.cache_utils._download_http", fake_download)

    for _ in range(2):
        net = _zeroed_net()
        FastGenModel._load_pretrained_model(net, url, device="cpu")
        assert torch.equal(net.lin.weight, expected["lin.weight"])

    assert fetched == [url]
    assert len(list((tmp_path / "MODEL" / "url").iterdir())) == 1


def test_load_pretrained_model_falls_back_to_uncached_http(monkeypatch):
    """A failed mirror is non-fatal now that the loader can read HTTP directly."""
    expected = _reference_state_dict()
    url = "https://example.com/model.pt"
    calls = []

    def fake_cache(path, strict):
        calls.append(("cache", path, strict))
        return path

    def fake_load(path, device):
        calls.append(("load", path, device))
        return expected

    monkeypatch.setattr("fastgen.methods.model.cache", fake_cache)
    monkeypatch.setattr("fastgen.methods.model.load_checkpoint_state_dict", fake_load)

    net = _zeroed_net()
    FastGenModel._load_pretrained_model(net, url, device="cpu")

    assert torch.equal(net.lin.weight, expected["lin.weight"])
    assert calls == [("cache", url, False), ("load", url, "cpu")]


# ---------------------------------------------------------------------------
# Role dispatch through ``prepare_module``, ``on_train_begin``, and EMA setup
# ---------------------------------------------------------------------------


def test_model_prepare_module_requires_name():
    model = _uninitialized_model()

    with pytest.raises(TypeError, match="name"):
        model.prepare_module(_TinyNetwork())


def test_model_prepare_module_rejects_plain_modules():
    """Every component owns its setup; third-party modules must be wrapped in a FastGenModule."""
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float64
    model.precision_opt = torch.float32

    with pytest.raises(TypeError, match="must be a FastGenModule"):
        model.prepare_module(torch.nn.Linear(2, 2), name="auxiliary")


def test_model_prepare_module_sets_network_and_discriminator_model_precision():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float64
    model.precision_opt = torch.float32
    model.net = _TinyNetwork()
    model.discriminator = Discriminator_Audio2D(feature_indices={0}, inner_dim=2, hidden_channels=4, num_blocks=1)
    model.discriminator.requires_grad_(False)

    model.prepare_module(model.net, name="net")
    model.prepare_module(model.discriminator, name="discriminator")

    assert model.net.precision_model == torch.float64
    # Trainable parameters keep the trainable storage; the frozen discriminator follows the model precision.
    assert model.net.proj.weight.dtype == torch.float32
    assert model.discriminator.precision_model == torch.float64
    assert {p.dtype for p in model.discriminator.parameters()} == {torch.float64}


def test_model_prepare_module_is_frozen_unsharded_and_does_not_register():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float64
    model.precision_opt = torch.float32
    model.net = _TinyNetwork()
    component = _TinyNetwork()
    fsdp_context = _RecordingFSDPContext()

    model.prepare_module(component, name="auxiliary", fsdp_context=fsdp_context)

    assert model.module_dict == {}
    assert component not in fsdp_context.modules
    assert not component.training
    assert not any(parameter.requires_grad for parameter in component.parameters())
    assert component.proj.weight.dtype == torch.float64
    assert component.precision_opt == torch.float64


def test_model_prepare_module_rejects_late_requested_fsdp_extra():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision_model = torch.float32
    model.precision_opt = torch.float32
    model.net = _TinyNetwork()
    model.module_dict = {"net": model.net}
    model.config.extra_fsdp_modules = ["auxiliary"]
    fsdp_context = _RecordingFSDPContext()
    model.prepare_module(model.net, name="net", fsdp_context=fsdp_context)

    auxiliary = _PlainPreprocessor()
    with pytest.raises(RuntimeError, match=r"auxiliary.*after FSDP setup.*active FSDP context"):
        model.prepare_module(auxiliary, name="auxiliary")

    assert not auxiliary.is_prepared
    assert not auxiliary._is_fsdp
    assert fsdp_context.calls == [model.net]


def test_model_prepare_module_rejects_an_already_sharded_extra_without_context():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision_model = torch.float32
    model.precision_opt = torch.float32
    model.net = _TinyNetwork()
    model.module_dict = {"net": model.net}
    model.config.extra_fsdp_modules = ["auxiliary"]
    auxiliary = _PlainPreprocessor()
    fsdp_context = _RecordingFSDPContext()

    model.prepare_module(model.net, name="net", fsdp_context=fsdp_context)
    model.prepare_module(auxiliary, name="auxiliary", fsdp_context=fsdp_context)
    with pytest.raises(RuntimeError, match="already prepared"):
        model.prepare_module(auxiliary, name="auxiliary")

    assert auxiliary.is_prepared
    assert auxiliary._is_fsdp
    assert fsdp_context.calls == [model.net, auxiliary]


def test_model_prepare_module_allows_requested_extra_without_fsdp_run():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision_model = torch.float32
    model.precision_opt = torch.float32
    model.config.extra_fsdp_modules = ["auxiliary"]
    auxiliary = _PlainPreprocessor()

    model.prepare_module(auxiliary, name="auxiliary")

    assert auxiliary.is_prepared
    assert not auxiliary._is_fsdp


def test_model_prepare_module_shards_frozen_model_during_fsdp_inference():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float64
    model.precision_opt = torch.float32
    model.net = _TinyNetwork()
    model.config = SimpleNamespace(inference=True, extra_fsdp_modules=[])
    fsdp_context = _RecordingFSDPContext()

    model.prepare_module(model.net, name="net", fsdp_context=fsdp_context)

    assert model.net in fsdp_context.modules
    assert not model.net.training
    assert not any(parameter.requires_grad for parameter in model.net.parameters())
    assert model.net.proj.weight.dtype == torch.float64


@pytest.mark.parametrize(
    ("trainable", "source_dtype", "expected_dtype"),
    [
        (False, torch.float32, torch.float64),
        (True, torch.float64, torch.float32),
    ],
)
def test_model_prepare_module_rejects_repeat_requests_without_mutation(trainable, source_dtype, expected_dtype):
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float64
    model.precision_opt = torch.float32

    component = _TinyNetwork().to(dtype=source_dtype)
    component.register_buffer("running", torch.arange(2, dtype=source_dtype))
    component.train(not trainable)
    component.requires_grad_(not trainable)
    name = "net" if trainable else "auxiliary"
    model.net = component if trainable else _TinyNetwork()

    model.prepare_module(component, name=name)
    assert component.is_prepared

    prepared_state = {name: tensor.detach().clone() for name, tensor in component.state_dict().items()}
    prepared_parameters = dict(component.named_parameters())
    prepared_buffers = dict(component.named_buffers())
    prepared_storage = {
        name: tensor.untyped_storage().data_ptr()
        for name, tensor in (*component.named_parameters(), *component.named_buffers())
    }

    precision_setup_calls = 0
    private_set_precision = component._set_precision

    def record_private_set_precision(*args, **kwargs):
        nonlocal precision_setup_calls
        precision_setup_calls += 1
        return private_set_precision(*args, **kwargs)

    component._set_precision = record_private_set_precision
    with pytest.raises(RuntimeError, match="already prepared"):
        model.prepare_module(component, name=name)
    assert precision_setup_calls == 0

    assert component.precision_model == model.precision_model
    assert all(module.training is trainable for module in component.modules())
    assert all(parameter.requires_grad is trainable for parameter in component.parameters())
    assert all(tensor.dtype == expected_dtype for tensor in component.state_dict().values())
    assert all(tensor.device == model.device for tensor in component.state_dict().values())
    assert all(torch.equal(tensor, prepared_state[name]) for name, tensor in component.state_dict().items())
    assert all(parameter is prepared_parameters[name] for name, parameter in component.named_parameters())
    assert all(buffer is prepared_buffers[name] for name, buffer in component.named_buffers())
    assert {
        name: tensor.untyped_storage().data_ptr()
        for name, tensor in (*component.named_parameters(), *component.named_buffers())
    } == prepared_storage


def test_on_train_begin_shards_student_and_ema_with_role_policies():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float64
    model.precision_opt = torch.float32
    model.precision_fsdp_reduce = torch.float32
    model.precision_amp = None
    model.net = _TinyNetwork()
    model.ema = _TinyNetwork()
    model.use_ema = ["ema"]
    model.module_dict = {"net": model.net, "ema": model.ema}
    model.config = SimpleNamespace(
        inference=False,
        enable_preprocessors=False,
        extra_fsdp_modules=[],
        net_adapter=SimpleNamespace(type=None),
    )
    fsdp_context = _RecordingFSDPContext()

    model.on_train_begin(fsdp_context=fsdp_context)

    assert fsdp_context.modules == [model.net, model.ema]
    assert fsdp_context.calls == [model.net, model.ema]
    assert model.uses_fsdp and model.net._is_fsdp and model.ema._is_fsdp
    assert model.net.training
    assert all(parameter.requires_grad for parameter in model.net.parameters())
    assert model.net.proj.weight.dtype == torch.float32
    assert not model.ema.training
    assert not any(parameter.requires_grad for parameter in model.ema.parameters())
    assert model.ema.proj.weight.dtype == torch.float32


def test_on_train_begin_allows_network_owned_eager_split_precision(monkeypatch):
    class NetworkOwnedSplit(_TinyNetwork):
        def forward(self, x_t, t, **kwargs):
            del t, kwargs
            return self.proj(x_t.to(dtype=self.proj.weight.dtype)).to(dtype=x_t.dtype)

    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision_model = torch.bfloat16
    model.precision_opt = torch.float32
    model.precision_fsdp_reduce = torch.float32
    model.precision_amp = None
    model.precision_amp_infer = None
    model.net = NetworkOwnedSplit()
    model.module_dict = {"net": model.net}
    model.config = SimpleNamespace(
        inference=False,
        enable_preprocessors=False,
        extra_fsdp_modules=[],
        net_adapter=SimpleNamespace(type=None),
    )
    warning_messages = []
    monkeypatch.setattr("fastgen.methods.model.logger.warning", warning_messages.append)

    model.on_train_begin()
    output = model.net(torch.ones(1, 2, dtype=torch.bfloat16), None)

    assert model.net.proj.weight.dtype == torch.float32
    assert output.dtype == torch.bfloat16
    assert len(warning_messages) == 1
    assert "network-owned dtype handling" in warning_messages[0]
    assert "may otherwise fail" in warning_messages[0]


def _root_order_worker(rank, world_size):
    import torch.distributed as dist

    results = {}
    for difference in ("none", "order", "count", "parameter_alias", "shared_owner", "lora"):
        model = _uninitialized_model()
        model.config.enable_preprocessors = False
        model.config.inference = True
        model.config.net_adapter = SimpleNamespace(type=None)
        model.device = torch.device("cuda", rank)
        model.precision_model = model.precision_fsdp_reduce = torch.float32
        model.precision_amp = model.precision_amp_infer = None
        model.use_ema = ["ema"]
        with torch.device("meta" if rank else "cpu"):
            model.net, model.ema = _TinyNetwork(), _TinyNetwork()
        if rank == 0:
            with torch.no_grad():
                for value, module in [(1, model.net), (2, model.ema)]:
                    for parameter in module.parameters():
                        parameter.fill_(value)
        roots = [("net", model.net), ("ema", model.ema)]
        if rank == 1:
            if difference == "order":
                roots.reverse()
            elif difference == "count":
                roots.pop()
            elif difference == "parameter_alias":
                model.ema.proj.weight = model.net.proj.weight
            elif difference == "shared_owner":
                model.net.dropout = model.ema.dropout = torch.nn.Dropout()
            elif difference == "lora":
                model.config.net_adapter.type = "lora"
        model.module_dict = dict(roots)
        before = tuple(model.parameters())
        try:
            model.on_train_begin(fsdp_context=object() if difference == "lora" else None)
        except RuntimeError as error:
            expected = (
                "model preparation plan mismatch"
                if difference in {"order", "count"}
                else "model preparation validation"
            )
            passed = difference != "none" and expected in str(error)
            passed &= all(a is b for a, b in zip(before, model.parameters()))
            passed &= all(parameter.is_meta == bool(rank) for parameter in model.parameters())
            passed &= not any(module.is_prepared for module in model.module_dict.values())
        else:
            passed = difference == "none" and all(
                bool((parameter == value).all())
                for value, module in [(1, model.net), (2, model.ema)]
                for parameter in module.parameters()
            )
        checks = [None] * world_size
        dist.all_gather_object(checks, passed)
        results[difference] = checks
    return results


@RunIf(min_gpus=2)
def test_model_preparation_agrees_on_root_order_before_mutation():
    result = run_distributed_test(_root_order_worker, world_size=2, timeout=120, nccl_timeout=30)
    assert result is not None and all(all(checks) for checks in result.values()), result


def test_plain_inference_ema_uses_model_precision():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float64
    model.precision_opt = torch.float32
    model.net = _TinyNetwork()
    model.ema = _TinyNetwork()
    model.use_ema = ["ema"]
    model.config = SimpleNamespace(inference=True, extra_fsdp_modules=[])

    model.prepare_module(model.ema, name="ema")

    assert model.ema.proj.weight.dtype == torch.float64
    assert model.ema.precision_opt == torch.float64
    assert not model.ema.training
    assert not any(parameter.requires_grad for parameter in model.ema.parameters())


def test_plain_training_ema_uses_optimization_precision():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float64
    model.precision_opt = torch.float32
    model.net = _TinyNetwork()
    model.ema = _TinyNetwork()
    model.use_ema = ["ema"]
    model.config = SimpleNamespace(inference=False, extra_fsdp_modules=[], autocast_cache_enabled=False)

    model.prepare_module(model.ema, name="ema")

    assert model.ema.proj.weight.dtype == torch.float32
    assert model.ema.precision_opt == torch.float32
    assert not model.ema.training
    assert not any(parameter.requires_grad for parameter in model.ema.parameters())


@pytest.mark.parametrize("rank0", [True, False])
def test_setup_ema_uses_standard_meta_init_lifecycle(monkeypatch, rank0):
    import fastgen.methods.model as model_module

    model = _uninitialized_model()
    model.net = _TinyNetwork()
    model.use_ema = ["ema"]
    model.config = SimpleNamespace(meta_init=True, net=object(), student_grid_cfg=None)
    with torch.no_grad():
        model.net.proj.weight.fill_(2.0)
        model.net.proj.bias.fill_(3.0)

    monkeypatch.setattr(model_module, "is_rank0", lambda: rank0)
    monkeypatch.setattr(model_module, "instantiate", lambda *_args, **_kwargs: _TinyNetwork())
    guarded = []
    loads = []
    load_module_state_dict = model_module.load_module_state_dict

    @contextlib.contextmanager
    def record_guarded_block(operation):
        guarded.append(operation)
        yield

    def record_load(module, state_dict, **kwargs):
        loads.append(module)
        return load_module_state_dict(module, state_dict, **kwargs)

    monkeypatch.setattr(model_module, "raise_across_ranks", record_guarded_block)
    monkeypatch.setattr(model_module, "load_module_state_dict", record_load)

    model._setup_ema()

    # Rank 0 copies inside the collective boundary; a meta rank only takes part in the poll.
    assert guarded == ["EMA network ema"]
    assert loads == ([model.ema] if rank0 else [])
    if rank0:
        assert all(not parameter.is_meta for parameter in model.ema.parameters())
        for name, tensor in model.net.state_dict().items():
            assert torch.equal(model.ema.state_dict()[name], tensor)
    else:
        assert all(parameter.is_meta for parameter in model.ema.parameters())


def test_setup_ema_requires_an_exact_state_clone(monkeypatch):
    import fastgen.methods.model as model_module

    class IncompatibleEMA(_TinyNetwork):
        def __init__(self):
            super().__init__()
            self.ema_only = torch.nn.Parameter(torch.ones(1))

    model = _uninitialized_model()
    model.net = _TinyNetwork()
    model.use_ema = ["ema"]
    model.config = SimpleNamespace(meta_init=False, net=object(), student_grid_cfg=None)

    monkeypatch.setattr(model_module, "instantiate", lambda *_args, **_kwargs: IncompatibleEMA())

    with pytest.raises(RuntimeError, match="Missing key.*ema_only"):
        model._setup_ema()


class _LoRANetwork(LoRAMixin, _TinyNetwork):
    lora_base_name = ""


@pytest.mark.parametrize("meta", [False, True], ids=["rank0", "meta-rank"])
def test_net_adapter_rejects_meta_init_on_every_rank(meta):
    model = _uninitialized_model()
    with torch.device("meta" if meta else "cpu"):
        model.net = _LoRANetwork()
    model.config = SimpleNamespace(net_adapter=SimpleNamespace(type="lora", rank=2), meta_init=True)

    # Rank 0 builds real weights and the other ranks meta ones; both reject before injecting anything.
    with pytest.raises(ValueError, match="meta_init is not supported with a net_adapter"):
        model._init_net_adapter()
    assert not model.net.lora_enabled


# ---------------------------------------------------------------------------
# Preprocessors, ``extra_fsdp_modules``, and component ownership
# ---------------------------------------------------------------------------


def test_register_modules_is_idempotent_for_the_same_modules():
    model = _uninitialized_model()
    auxiliary = _TinyNetwork()

    model.register_modules(auxiliary=auxiliary)
    model.register_modules(auxiliary=auxiliary)

    assert model.module_dict == {"auxiliary": auxiliary}


def test_register_modules_rejects_a_collision_before_updating():
    model = _uninitialized_model()
    registered = _TinyNetwork()
    replacement = _TinyNetwork()
    new_component = _TinyNetwork()
    model.register_modules(auxiliary=registered)

    with pytest.raises(ValueError, match="auxiliary"):
        model.register_modules(new_component=new_component, auxiliary=replacement)

    assert model.module_dict == {"auxiliary": registered}


@pytest.mark.parametrize("already_registered", [False, True])
def test_register_modules_rejects_duplicate_physical_roots_before_updating(already_registered):
    model = _uninitialized_model()
    shared = _TinyNetwork()
    if already_registered:
        model.register_modules(first=shared)
    before = model.module_dict.copy()
    with pytest.raises(AssertionError, match="root 'second' shares modules with another root"):
        model.register_modules(first=shared, second=shared, independent=_TinyNetwork())
    assert model.module_dict == before


@pytest.mark.parametrize("already_registered", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "parameter",
        "buffer",
        "nonpersistent_buffer",
        "cross_kind",
        "shared_child",
        "stateless_child",
        "nested_root",
        "root_as_child",
    ],
)
def test_register_modules_rejects_shared_state_across_roots(already_registered, kind):
    model = _uninitialized_model()
    first, second = _TinyNetwork(), _TinyNetwork()
    if kind == "parameter":
        second.proj.weight = first.proj.weight
    elif kind == "shared_child":
        second.proj = first.proj
    elif kind == "stateless_child":
        first.dropout = torch.nn.Dropout()
        second.container = torch.nn.Sequential(first.dropout)
    elif kind == "nested_root":
        second.child = first
    elif kind == "root_as_child":
        first.child = second
    elif kind == "cross_kind":
        second.register_buffer("cache", first.proj.weight)
    else:
        first.register_buffer("cache", torch.ones(2), persistent=kind == "buffer")
        second.register_buffer("cache", first.cache, persistent=kind == "buffer")
    if already_registered:
        model.register_modules(first=first)
    before = model.module_dict.copy()
    expected = "modules" if kind in {"shared_child", "stateless_child", "nested_root", "root_as_child"} else "tensors"
    with pytest.raises(AssertionError, match=f"root 'second' shares {expected} with another root"):
        model.register_modules(first=first, second=second)
    assert model.module_dict == before


def test_register_modules_accepts_parameter_ties_and_shared_children_within_a_root():
    model = _uninitialized_model()
    root = _TinyNetwork()
    root.shared = root.proj
    root.tied = root.proj.weight
    root.dropout = torch.nn.Dropout()
    root.dropout_alias = root.dropout
    model.register_modules(first=root, second=_TinyNetwork())
    assert model.module_dict["first"] is root


@pytest.mark.parametrize("kind", ["parameter", "stateless_child"])
def test_training_setup_rechecks_roots_before_preparation(kind):
    model = _uninitialized_model()
    model.net, other = _TinyNetwork(), _TinyNetwork()
    model.register_modules(net=model.net, other=other)
    if kind == "parameter":
        other.proj.weight = model.net.proj.weight
    else:
        model.net.dropout = torch.nn.Dropout()
        other.dropout = model.net.dropout
    before = model.net.proj.weight
    expected = "tensors" if kind == "parameter" else "modules"
    with pytest.raises(AssertionError, match=f"root 'other' shares {expected} with another root"):
        model.on_train_begin()
    assert model.net.proj.weight is before
    assert not model.net.is_prepared and not other.is_prepared


def test_net_preprocessors_selects_candidate_before_registry_exists():
    model = _uninitialized_model()
    candidate = _TinyNetwork.__new__(_TinyNetwork)
    torch.nn.Module.__init__(candidate)
    model.net = candidate

    assert not hasattr(candidate, "preprocessors")
    assert model.net_preprocessors is candidate


def test_net_preprocessors_rejects_a_model_without_a_network_host():
    model = _uninitialized_model()

    with pytest.raises(RuntimeError, match="no network available to own preprocessors"):
        _ = model.net_preprocessors


def test_reshard_fsdp_modules_selects_preprocessor_or_all_registered_roots(monkeypatch):
    import fastgen.utils.distributed.fsdp as fsdp_utils

    model = _uninitialized_model()
    host = _TinyNetwork()
    vae = _PlainPreprocessor()
    text_encoder = _PlainPreprocessor()
    auxiliary = _PlainPreprocessor()
    host.preprocessors.update(vae=vae, text_encoder=text_encoder)
    model.net = host
    model.module_dict.update(net=host, vae=vae, text_encoder=text_encoder, auxiliary=auxiliary)

    fsdp_context = _RecordingFSDPContext()
    for module in (host, vae, auxiliary):
        module.prepare_module(device="cpu", fsdp_context=fsdp_context)

    calls = []
    monkeypatch.setattr(fsdp_utils, "reshard_fsdp_modules", lambda modules: calls.append(tuple(modules)))

    model.reshard_fsdp_modules(True)
    model.reshard_fsdp_modules()

    assert calls == [(vae,), (host, vae, auxiliary)]


def test_preprocessor_setup_uses_model_precision():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float64
    model.net = _PreprocessorNetwork()
    model.use_ema = []
    model.config = SimpleNamespace(inference=False, extra_fsdp_modules=[], enable_preprocessors=True)

    host = model.init_preprocessors()

    assert host is model.net
    for name in ("vae", "text_encoder", "image_encoder"):
        preprocessor = host.preprocessors[name]
        assert preprocessor.core.weight.dtype == torch.float64
        assert preprocessor.precision_model == torch.float64
        assert preprocessor._precision_opt is None
        assert preprocessor.precision_opt == torch.float64
        assert preprocessor.is_prepared
        assert all(not module.training for module in preprocessor.modules())
        assert not any(parameter.requires_grad for parameter in preprocessor.parameters())


@pytest.mark.parametrize("fsdp", [False, True])
def test_init_preprocessors_prepares_each_preprocessor_once(fsdp):
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision_model = torch.float32
    model.net = _PreprocessorNetwork()
    model.config = SimpleNamespace(inference=False, enable_preprocessors=True, extra_fsdp_modules=["vae"])
    context = _RecordingFSDPContext() if fsdp else None

    host = model.init_preprocessors(fsdp_context=context)
    assert all(module.is_prepared for module in host.preprocessors.values())
    if fsdp:
        assert context.calls == [host.preprocessors["vae"]]

    # A second call raises before any mutation, even when it requests FSDP for an eager preprocessor.
    late_context = _RecordingFSDPContext()
    with pytest.raises(RuntimeError, match="already prepared"):
        model.init_preprocessors(fsdp_context=late_context)
    assert late_context.calls == []
    assert host.preprocessors["vae"]._is_fsdp == fsdp


def _prepare_shared_preprocessor(model, helper):
    model.net.preprocessors["vae"] = helper
    model.init_preprocessors()


def assert_shared_helper_cannot_be_prepared_twice(prepare_helper):
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision_model = torch.bfloat16
    model.precision_amp_enc = None
    model.config.enable_preprocessors = True
    model.net = _PreprocessorNetwork()
    helper = _PlainPreprocessor()
    # Even the model's own frozen policy cannot be applied a second time.
    helper.prepare_module(device=model.device, precision_model=model.precision_model)
    before = {name: tensor.clone() for name, tensor in helper.state_dict().items()}
    with pytest.raises(RuntimeError, match="already prepared"):
        prepare_helper(model, helper)
    for name, tensor in helper.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_shared_helpers_cannot_be_prepared_twice():
    assert_shared_helper_cannot_be_prepared_twice(_prepare_shared_preprocessor)


def _assert_shared_helper_double_preparation(rank, world_size, prepare_helper):
    import torch.distributed as dist

    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision_model = torch.float32
    model.precision_amp_enc = None
    model.config.enable_preprocessors = True
    model.net = _PreprocessorNetwork()
    helper = _PlainPreprocessor()
    helper.prepare_module("cpu", precision_model=torch.float32)
    if rank == 1:
        helper._is_prepared = False
    before = helper.core.weight.clone()
    # Only rank 0 repeats a preparation; every rank must raise instead of waiting for it.
    with pytest.raises(RuntimeError, match="rank 0 failed.*already prepared"):
        prepare_helper(model, helper)
    results = [None] * world_size
    dist.all_gather_object(results, torch.equal(helper.core.weight, before))
    return results


def _shared_helper_double_preparation_worker(rank, world_size):
    return _assert_shared_helper_double_preparation(rank, world_size, _prepare_shared_preprocessor)


@RunIf(min_gpus=2)
def test_shared_helper_double_preparation_fails_on_every_rank():
    checks = run_distributed_test(_shared_helper_double_preparation_worker, world_size=2, timeout=120, nccl_timeout=30)
    assert checks is not None and all(checks)


def test_preprocessor_precision_hook_overrides_model_boundary():
    model = _uninitialized_model()
    model.precision = torch.float32
    model.precision_model = torch.bfloat16
    model.device = torch.device("cpu")
    model.config = SimpleNamespace(inference=False, extra_fsdp_modules=[], enable_preprocessors=True)
    net = _PreprocessorNetwork()
    net.init_preprocessors()
    net.preprocessors["vae"] = _FP32Preprocessor()
    model.net = net
    model.use_ema = []

    model.init_preprocessors()

    vae = net.preprocessors["vae"]
    text_encoder = net.preprocessors["text_encoder"]
    assert vae.precision_model == torch.float32
    assert text_encoder.precision_model == torch.bfloat16
    # Storage follows each preprocessor's effective model precision.
    assert vae.core.weight.dtype == torch.float32
    assert text_encoder.core.weight.dtype == torch.bfloat16


def test_preprocessor_is_registered_by_name_in_module_dict():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float32
    model.net = _PreprocessorNetwork()
    model.use_ema = []
    model.config = SimpleNamespace(inference=False, enable_preprocessors=True, extra_fsdp_modules=["vae"])

    model.init_preprocessors()

    assert model.module_dict["vae"] is model.net.preprocessors["vae"]


def test_init_preprocessors_rejects_a_component_name_collision():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float64
    model.net = _PreprocessorNetwork()
    model.use_ema = []
    model.config = SimpleNamespace(inference=False, extra_fsdp_modules=[], enable_preprocessors=True)
    registered_vae = _PlainPreprocessor()
    model.register_modules(vae=registered_vae)

    with pytest.raises(ValueError, match="vae"):
        model.init_preprocessors()

    assert model.module_dict["vae"] is registered_vae


def test_fsdp_extras_and_lazy_preprocessors_stay_at_model_precision():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float32
    model.precision_opt = torch.float64
    model.precision_fsdp_reduce = torch.float64
    model.net = _PreprocessorNetwork()
    model.use_ema = []
    model.module_dict["net"] = model.net
    model.config = SimpleNamespace(
        inference=False,
        enable_preprocessors=True,
        extra_fsdp_modules=["vae"],
        net_adapter=SimpleNamespace(type=None),
    )

    fsdp_context = _RecordingFSDPContext()
    model.on_train_begin(fsdp_context=fsdp_context)

    # The trainable FSDP root stores everything in the optimization precision; frozen preprocessors
    # (including the one selected as an FSDP extra) stay at the model precision.
    assert model.net.proj.weight.dtype == torch.float64
    for name in ("vae", "text_encoder", "image_encoder"):
        preprocessor = model.net.preprocessors[name]
        assert preprocessor.core.weight.dtype == torch.float32
        assert preprocessor.precision_model == torch.float32
        assert preprocessor.precision_opt == torch.float32
    vae = model.net.preprocessors["vae"]
    assert model.module_dict["vae"] is vae
    assert model.net._is_fsdp and vae._is_fsdp
    assert not model.net.preprocessors["text_encoder"]._is_fsdp
    assert not model.net.preprocessors["image_encoder"]._is_fsdp


@pytest.fixture
def vace_depth_checkpoint(monkeypatch, tmp_path):
    from fastgen.networks.VaceWan.modules import vace_depth_annotator

    monkeypatch.setattr(vace_depth_annotator, "DepthAnythingV2", lambda **_kwargs: torch.nn.Linear(2, 2))
    model_path = tmp_path / "depth.pth"
    torch.save({"weight": torch.full((2, 2), 2.0), "bias": torch.zeros(2)}, model_path)
    return str(model_path)


def test_vace_depth_annotator_is_sharded_during_fsdp_extra_startup(vace_depth_checkpoint):
    from fastgen.networks.VaceWan.modules.vace_depth_annotator import VACEDepthExtractor

    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.bfloat16
    model.precision_opt = torch.float32
    model.precision_fsdp_reduce = torch.float32
    model.precision_amp = None

    class _VaceHost(_TinyNetwork):
        def init_preprocessors(self):
            self.preprocessors["depth_annotator"] = VACEDepthExtractor(model_path=vace_depth_checkpoint)

    model.net = _VaceHost()
    model.module_dict["net"] = model.net
    model.config = SimpleNamespace(
        inference=False,
        enable_preprocessors=True,
        extra_fsdp_modules=["depth_annotator"],
        net_adapter=SimpleNamespace(type=None),
    )
    fsdp_context = _RecordingFSDPContext()

    model.on_train_begin(fsdp_context=fsdp_context)

    depth_annotator = model.net.preprocessors["depth_annotator"]
    assert model.module_dict["depth_annotator"] is depth_annotator
    assert isinstance(depth_annotator.model, torch.nn.Linear)
    assert depth_annotator.is_prepared and depth_annotator._is_fsdp
    assert depth_annotator.device == torch.device("cpu")
    assert depth_annotator.model.weight.dtype == torch.float32
    assert not any(parameter.requires_grad for parameter in depth_annotator.parameters())
    assert torch.equal(depth_annotator.model.weight, torch.full((2, 2), 2.0))
    assert fsdp_context.calls == [model.net, depth_annotator]


def test_loaded_vace_depth_annotator_is_sharded_as_an_fsdp_extra(vace_depth_checkpoint):
    from fastgen.networks.VaceWan.modules.vace_depth_annotator import VACEDepthExtractor

    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision_model = torch.bfloat16
    model.precision_opt = torch.float32
    model.net = _TinyNetwork()
    model.config = SimpleNamespace(inference=False, extra_fsdp_modules=["depth_annotator"])

    depth_annotator = VACEDepthExtractor(model_path=vace_depth_checkpoint)
    loaded_model = depth_annotator.model
    fsdp_context = _RecordingFSDPContext()

    model.prepare_module(depth_annotator, name="depth_annotator", fsdp_context=fsdp_context)

    assert depth_annotator.model is loaded_model
    assert depth_annotator.is_prepared
    assert depth_annotator.device == torch.device("cpu")
    assert fsdp_context.calls == [depth_annotator]
    assert fsdp_context.modules == [depth_annotator]

    with pytest.raises(RuntimeError, match="already prepared"):
        model.prepare_module(depth_annotator, name="depth_annotator", fsdp_context=fsdp_context)
    assert fsdp_context.calls == [depth_annotator]
    assert fsdp_context.modules == [depth_annotator]


def test_repeat_preparation_of_loaded_vace_depth_annotator_raises_without_mutation(vace_depth_checkpoint):
    from fastgen.networks.VaceWan.modules.vace_depth_annotator import VACEDepthExtractor

    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision_model = torch.bfloat16
    model.precision_opt = torch.float32
    model.net = _TinyNetwork()
    model.config = SimpleNamespace(inference=False, extra_fsdp_modules=["depth_annotator"])

    depth_annotator = VACEDepthExtractor(model_path=vace_depth_checkpoint)
    loaded_model = depth_annotator.model
    depth_annotator.prepare_module(device="cpu", precision_model=torch.bfloat16, trainable=False)
    assert depth_annotator.is_prepared
    fsdp_context = _RecordingFSDPContext()

    with pytest.raises(RuntimeError, match="already prepared"):
        model.prepare_module(depth_annotator, name="depth_annotator", fsdp_context=fsdp_context)

    assert depth_annotator.model is loaded_model
    assert depth_annotator.is_prepared
    assert not depth_annotator._is_fsdp
    assert depth_annotator.device == torch.device("cpu")
    assert fsdp_context.calls == []
    assert fsdp_context.modules == []


def test_fsdp_completion_markers_belong_to_registered_components(monkeypatch):
    import fastgen.utils.distributed.fsdp as fsdp_utils

    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float32
    model.precision_opt = torch.float64
    model.precision_fsdp_reduce = torch.float32
    model.net = _PreprocessorNetwork()
    model.use_ema = []
    model.module_dict["net"] = model.net
    model.config = SimpleNamespace(
        inference=False,
        enable_preprocessors=True,
        extra_fsdp_modules=["vae"],
        net_adapter=SimpleNamespace(type=None),
    )

    monkeypatch.setattr(fsdp_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(fsdp_utils, "sync_any", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(fsdp_utils, "_fully_shard_module", lambda *_args, **_kwargs: None)
    fsdp_context = fsdp_utils.FSDPContext(
        mesh=SimpleNamespace(device_type="cpu"),
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=None,
        cpu_offload=False,
    )
    _shard_module = fsdp_context._shard_module

    def shard_registered_component(module):
        assert any(module is component for component in model.module_dict.values())
        return _shard_module(module)

    monkeypatch.setattr(fsdp_context, "_shard_module", shard_registered_component)

    model.on_train_begin(fsdp_context=fsdp_context)

    sharded_components = {id(component) for component in model.module_dict.values() if component._is_fsdp}
    assert sharded_components == {id(model.net), id(model.net.preprocessors["vae"])}


def test_parent_is_sharded_before_lazy_preprocessors_are_initialized():
    events = []

    class _OrderedPreprocessorNetwork(_PreprocessorNetwork):
        def init_preprocessors(self):
            events.append(("init_preprocessors", self))
            super().init_preprocessors()

    class _OrderedFSDPContext(_RecordingFSDPContext):
        def _shard_module(self, module):
            events.append(("shard", module))
            return super()._shard_module(module)

    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float32
    model.precision_opt = torch.float32
    model.precision_fsdp_reduce = torch.float32
    model.net = _OrderedPreprocessorNetwork()
    model.module_dict["net"] = model.net
    model.config = SimpleNamespace(
        inference=False,
        enable_preprocessors=True,
        extra_fsdp_modules=["vae"],
        net_adapter=SimpleNamespace(type=None),
    )

    model.on_train_begin(fsdp_context=_OrderedFSDPContext())

    vae = model.net.preprocessors["vae"]
    assert events[:3] == [("shard", model.net), ("init_preprocessors", model.net), ("shard", vae)]


def test_preconstructed_fsdp_extra_is_prepared_once_during_training_setup():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float32
    model.precision_opt = torch.float64
    model.precision_fsdp_reduce = torch.float64
    model.net = _PreprocessorNetwork()
    model.use_ema = []
    model.module_dict["net"] = model.net
    model.config = SimpleNamespace(
        inference=False,
        enable_preprocessors=True,
        extra_fsdp_modules=["vae"],
        net_adapter=SimpleNamespace(type=None),
    )

    # Construct the VAE early without preparing it. Model startup adds it to the
    # component dict before the unified preparation pass.
    model.net.init_preprocessors()
    vae = model.net.preprocessors["vae"]
    assert not vae.is_prepared

    precision_setup_calls = 0
    private_set_precision = vae._set_precision

    def record_private_set_precision(*args, **kwargs):
        nonlocal precision_setup_calls
        precision_setup_calls += 1
        return private_set_precision(*args, **kwargs)

    vae._set_precision = record_private_set_precision
    fsdp_context = _RecordingFSDPContext()
    model.on_train_begin(fsdp_context=fsdp_context)

    # The unified component pass prepares the root exactly once.
    assert precision_setup_calls == 1
    assert vae.is_prepared
    assert fsdp_context.modules.count(vae) == 1
    assert model.module_dict["vae"] is vae


def test_on_train_begin_with_prepared_fsdp_extra_raises_without_lifecycle_setup():
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.float32
    model.precision_opt = torch.float64
    model.precision_fsdp_reduce = torch.float64
    model.net = _PreprocessorNetwork()
    model.use_ema = []
    model.module_dict["net"] = model.net
    model.config = SimpleNamespace(
        inference=False,
        enable_preprocessors=True,
        extra_fsdp_modules=["vae"],
        net_adapter=SimpleNamespace(type=None),
    )

    model.init_preprocessors()
    vae = model.net.preprocessors["vae"]
    assert vae.is_prepared

    precision_setup_calls = 0
    private_set_precision = vae._set_precision

    def record_private_set_precision(*args, **kwargs):
        nonlocal precision_setup_calls
        precision_setup_calls += 1
        return private_set_precision(*args, **kwargs)

    vae._set_precision = record_private_set_precision
    fsdp_context = _RecordingFSDPContext()
    with pytest.raises(RuntimeError, match="already prepared"):
        model.on_train_begin(fsdp_context=fsdp_context)

    # A late FSDP request neither repeats lifecycle setup nor mutates the eager-prepared root.
    assert precision_setup_calls == 0
    assert vae.is_prepared
    assert not vae._is_fsdp
    assert vae not in fsdp_context.calls
    assert vae not in fsdp_context.modules
    assert model.module_dict["vae"] is vae


def test_concrete_wan_preprocessor_extra_fsdp_uses_model_precision():
    """A lazily built real wrapper receives model precision during its first FSDP preparation."""
    from fastgen.networks.Wan.network import WanVideoEncoder

    class _WanPreprocessorNetwork(_TinyNetwork):
        def init_preprocessors(self):
            vae = WanVideoEncoder.__new__(WanVideoEncoder)
            torch.nn.Module.__init__(vae)
            vae.vae = torch.nn.Linear(2, 2)
            vae.fixed_precision_model = None
            self.preprocessors["vae"] = vae

    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.precision_model = torch.bfloat16
    model.precision_opt = torch.float64
    model.precision_fsdp_reduce = torch.float32
    model.net = _WanPreprocessorNetwork()
    model.use_ema = []
    model.module_dict["net"] = model.net
    model.config = SimpleNamespace(
        inference=False,
        enable_preprocessors=True,
        extra_fsdp_modules=["vae"],
        net_adapter=SimpleNamespace(type=None),
    )

    fsdp_context = _RecordingFSDPContext()
    model.on_train_begin(fsdp_context=fsdp_context)

    vae = model.net.preprocessors["vae"]
    assert model.net.proj.weight.dtype == torch.float64
    assert vae.vae.weight.dtype == torch.bfloat16
    assert vae.precision_model == torch.bfloat16
    assert vae in fsdp_context.modules
    assert model.module_dict["vae"] is vae


def test_register_modules_rejects_non_fastgen_modules():
    model = _uninitialized_model()

    with pytest.raises(AssertionError, match="'auxiliary' must be a FastGenModule"):
        model.register_modules(auxiliary=torch.nn.Linear(2, 2))
    assert "auxiliary" not in model.module_dict


@pytest.mark.parametrize("sharded", [False, True])
def test_init_preprocessors_warns_about_sharded_preprocessor_control_flow(monkeypatch, sharded):
    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision_model = torch.bfloat16
    model.net = _PreprocessorNetwork()
    model.register_modules(net=model.net)
    model.config.enable_preprocessors = True
    model.config.extra_fsdp_modules = ["vae"]
    warnings = []
    monkeypatch.setattr("fastgen.methods.model.logger.warning", warnings.append)

    model.init_preprocessors(fsdp_context=_RecordingFSDPContext() if sharded else None)
    if not sharded:
        assert warnings == []
        return

    assert len(warnings) == 1
    assert all(
        term in warnings[0]
        for term in ("vae", "preprocessing", "callback decode", "raw-versus-cached", "media schemas", "same order")
    )


def test_init_preprocessors_rejects_a_vae_that_disagrees_with_the_declared_compression():
    """``pixel_shape`` sizes pixel inputs from the network's declaration, so the loaded VAE must match it."""

    class _CompressingNetwork(_PreprocessorNetwork):
        latent_compression_factors = (4, 8, 8)

    model = _uninitialized_model()
    model.device = torch.device("cpu")
    model.precision_model = torch.bfloat16
    model.net = _CompressingNetwork()
    model.register_modules(net=model.net)
    model.config.enable_preprocessors = True
    vae = _PlainPreprocessor()
    vae.latent_compression_factors = (4, 16, 16)
    model.net.preprocessors["vae"] = vae

    with pytest.raises(ValueError, match=r"declares latent compression \(4, 8, 8\), but its VAE compresses by"):
        model.init_preprocessors()


def test_validate_modules_requires_trainable_parameters_on_model_roots():
    """Non-EMA model roots must keep a parameter that requires gradients outside inference."""
    model = _uninitialized_model()
    model.net = _TinyNetwork()
    model.module_dict = {"net": model.net}
    model.net.prepare_module(device="cpu", precision_model=torch.bfloat16, precision_opt=torch.float32, trainable=True)
    model.validate_modules()

    model.net.requires_grad_(False)
    with pytest.raises(AssertionError, match="Trainable model root 'net' has no parameter requiring gradients"):
        model.validate_modules()

    # Inference roots are frozen at model precision; the check does not apply there.
    model.config.inference = True
    model.validate_modules()


@pytest.mark.parametrize("root_name", ["net", "frozen"])
@pytest.mark.parametrize("failed_preparation", [False, True])
def test_validate_modules_requires_every_registered_root_to_complete_preparation(root_name, failed_preparation):
    model = _uninitialized_model()
    model.net = _TinyNetwork()
    model.register_modules(net=model.net)
    if root_name == "net":
        pending = model.net
    else:
        model.net.prepare_module("cpu", precision_opt=model.precision_opt, trainable=True)
        pending = _PlainPreprocessor()
        model.register_modules(frozen=pending)
    if failed_preparation:

        def fail_placement(*args, **kwargs):
            raise RuntimeError("injected placement failure")

        pending.to = fail_placement
        with pytest.raises(RuntimeError, match="injected placement failure"):
            pending.prepare_module("cpu")
    assert not pending.is_prepared
    with pytest.raises(AssertionError, match=f"Registered root '{root_name}' contains an unprepared FastGen module"):
        model.validate_modules()


def test_validate_modules_rejects_children_attached_after_preparation():
    model = _uninitialized_model()
    model.net = _TinyNetwork()
    model.register_modules(net=model.net)
    model.net.prepare_module("cpu", precision_model=torch.bfloat16, precision_opt=torch.float32, trainable=True)
    model.net.late = _PlainPreprocessor()
    with pytest.raises(AssertionError, match="Registered root 'net' contains an unprepared FastGen module"):
        model.validate_modules()


# ---------------------------------------------------------------------------
# Gradient scaler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("precision_model", "enabled"), [("float16", True), ("bfloat16", False)])
def test_requested_grad_scaler_warns_when_nothing_runs_in_float16(monkeypatch, precision_model, enabled):
    """Loss scaling protects only FP16 arithmetic, so a requested scaler without any FP16 precision stays disabled
    and says so instead of being silently ignored."""
    model = _uninitialized_model()
    model._resolve_precisions(precision="float32", precision_model=precision_model, precision_opt="float32")
    model.device = torch.device("cpu")
    model.net = _TinyNetwork()
    model.module_dict = {"net": model.net}
    model.config = BaseModelConfig(device="cpu", grad_scaler_enabled=True)
    warning_messages = []
    monkeypatch.setattr("fastgen.methods.model.logger.warning", warning_messages.append)

    model.init_optimizers()

    scaler_warnings = [message for message in warning_messages if "Grad scaler" in message]
    assert model.grad_scaler.is_enabled() is enabled
    assert len(scaler_warnings) == (0 if enabled else 1)
    assert enabled or "no FP16 arithmetic to scale" in scaler_warnings[0]


def test_grad_scaler_steps_multiple_optimizers_before_one_update():
    class _RecordingScaler:
        def __init__(self):
            self.optimizers = []
            self.update_calls = 0

        def is_enabled(self):
            return True

        def step(self, optimizer):
            self.optimizers.append(optimizer)
            optimizer.step()

        def update(self):
            self.update_calls += 1

    model = _uninitialized_model()
    params = [torch.nn.Parameter(torch.ones(())) for _ in range(2)]
    optimizers = [torch.optim.SGD([param], lr=0.1) for param in params]
    for param in params:
        param.grad = torch.ones_like(param)
    model.grad_scaler = _RecordingScaler()
    model.get_optimizers = lambda iteration: optimizers
    model.get_lr_schedulers = lambda iteration: []

    model.optimizers_schedulers_step(0)

    assert model.grad_scaler.optimizers == optimizers
    assert model.grad_scaler.update_calls == 1


def test_grad_scaler_rejects_low_precision_gradients_before_optimizer_step():
    class _RecordingScaler:
        def __init__(self):
            self.step_calls = 0

        def is_enabled(self):
            return True

        def step(self, optimizer):
            del optimizer
            self.step_calls += 1

        def update(self):
            raise AssertionError("update must not run after a rejected gradient")

    model = _uninitialized_model()
    param = torch.nn.Parameter(torch.ones((), dtype=torch.float16))
    param.grad = torch.ones_like(param)
    optimizer = torch.optim.SGD([param], lr=0.1)
    model.grad_scaler = _RecordingScaler()
    model.get_optimizers = lambda iteration: [optimizer]
    model.get_lr_schedulers = lambda iteration: []

    with pytest.raises(RuntimeError, match="cannot unscale FP16 gradients"):
        model.optimizers_schedulers_step(0)
    assert model.grad_scaler.step_calls == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_grad_scaler_accepts_non_fp16_gradients(dtype):
    model = _uninitialized_model()
    model.grad_scaler = SimpleNamespace(is_enabled=lambda: True)
    param = torch.nn.Parameter(torch.ones((), dtype=dtype))
    param.grad = torch.ones_like(param)
    optimizer = torch.optim.SGD([param], lr=0.1)

    assert model.should_use_grad_scaler(optimizer)


# ---------------------------------------------------------------------------
# Reward integration dtype boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("total_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("reward_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_optimized_reward_uses_natural_dtype_promotion(total_dtype, reward_dtype):
    model = _uninitialized_model()
    model.config.optimize_reward = True
    model.reward = lambda gen_data, data, condition_dropout_mask=None: {"total_loss": gen_data.mean()}
    reward_input = torch.ones(2, dtype=reward_dtype, requires_grad=True)
    total_loss = torch.ones((), dtype=total_dtype, requires_grad=True)

    loss_map = model.apply_reward(reward_input, {}, {"total_loss": total_loss})

    assert loss_map["reward_loss"].dtype == reward_dtype
    assert loss_map["total_loss"].dtype == torch.promote_types(total_dtype, reward_dtype)
    loss_map["total_loss"].backward()
    assert total_loss.grad is not None and total_loss.grad.dtype == total_dtype
    assert reward_input.grad is not None and reward_input.grad.dtype == reward_dtype
