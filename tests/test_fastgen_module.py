# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``FastGenModule`` and the ``FastGenNetwork`` base (``fastgen/networks/network.py``).

Each FastGen module owns its precision policy: private ``_set_precision`` records and optionally
casts storage dtypes (with fixed-precision overrides) for one module island. Device placement instead
belongs to a physical root. The exposed ``prepare_module`` method is the lifecycle boundary: it records one target
across nested FastGen modules and moves or shards the complete tree. ``FastGenNetwork`` adds the plain
``preprocessors`` dict and ``vae_context``.

How ``FastGenModel`` dispatches roles onto this boundary is tested in ``tests/test_fastgen_model.py``;
network-specific pins live next to their networks in ``tests/test_network.py``.
"""

import pytest
import torch
from diffusers import ModelMixin
from transformers import BertConfig, BertModel

from fastgen.networks.network import FastGenNetwork, FastGenModule
import fastgen.networks.module as module_lifecycle
import fastgen.utils.module as module_state
from tests.utils import RunIf, run_distributed_test


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _MixedNetwork(FastGenNetwork):
    def __init__(self, device=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": torch.float64}
        self.weight = torch.nn.Parameter(torch.ones(2, **factory_kwargs))
        self.frozen = torch.nn.Parameter(torch.ones(2, **factory_kwargs), requires_grad=False)
        self.fixed = torch.nn.Parameter(torch.ones(2, **factory_kwargs))
        self.register_buffer("running", torch.ones(2, **factory_kwargs))
        self.register_buffer("indices", torch.ones(2, device=device, dtype=torch.long))

    def _fixed_precision_overrides(self):
        return {"fixed": torch.float64}

    def forward(self, x_t, t, **kwargs):
        del t, kwargs
        return x_t


class _TinyNetwork(FastGenNetwork):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def forward(self, x_t, t, **kwargs):
        del t, kwargs
        return self.proj(x_t.to(dtype=self.precision_model)).to(dtype=x_t.dtype)


class _Wrapper(FastGenModule):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Linear(2, 2, dtype=torch.float64)


class _SharedBufferModule(FastGenModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((2,), 3.0, dtype=torch.float64))
        self.child = torch.nn.Module()
        self.child.register_parameter("weight_alias", self.weight)
        shared = torch.tensor([1.25, -2.5], dtype=torch.float64)
        self.register_buffer("running", shared)
        self.register_buffer("running_alias", shared, persistent=False)
        self.child.register_buffer("cached", shared, persistent=False)
        indices = torch.tensor([1, 2], dtype=torch.long)
        self.register_buffer("indices", indices)
        self.child.register_buffer("indices", indices, persistent=False)


class _FixedPrecisionWrapper(_Wrapper):
    @property
    def precision_model(self):
        return torch.float32


class _PlainPreprocessor(FastGenModule):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Linear(2, 2)


class _VAEOnlyNetwork(_TinyNetwork):
    def __init__(self):
        super().__init__()
        self.init_vae_calls = 0

    def init_vae(self):
        self.init_vae_calls += 1
        self.preprocessors["vae"] = _PlainPreprocessor()


class _DeclaredTransformer(torch.nn.Module):
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table"]

    def __init__(self):
        super().__init__()
        self.time_embedder = torch.nn.Linear(2, 2)
        self.proj = torch.nn.Linear(2, 2)
        self.scale_shift_table = torch.nn.Parameter(torch.ones(2))
        self.rope = torch.nn.Module()
        self.rope.register_buffer("freqs_cos", torch.ones(2))


class _DiffusersDtypeProbe(ModelMixin):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)


def _set_precision(component, **kwargs):
    component._set_precision(precision_model=torch.bfloat16, **kwargs)


# ---------------------------------------------------------------------------
# Validated state-dict loading
# ---------------------------------------------------------------------------


class _NestedCheckpointLinear(torch.nn.Linear):
    def load_state_dict(self, state_dict, **kwargs):
        return super().load_state_dict(state_dict["model"], **kwargs)


def test_require_any_uses_matched_target_keys_after_nested_source_unwrapping(monkeypatch):
    module = _NestedCheckpointLinear(2, 2)
    weight = torch.full_like(module.weight, 3.0)
    messages = []
    # A partial load (missing bias, unexpected checkpoint_only) is announced as a warning, not a success.
    monkeypatch.setattr(module_state.logger, "warning", messages.append)

    load_info = module_state.load_module_state_dict(
        module,
        {"model": {"weight": weight, "checkpoint_only": torch.ones(1)}},
        False,
        True,
    )

    assert torch.equal(module.weight, weight)
    assert load_info.missing_keys == ["bias"]
    assert load_info.unexpected_keys == ["checkpoint_only"]
    assert len(messages) == 1 and "_NestedCheckpointLinear, loading info:" in messages[0]


def test_require_any_rejects_a_permissive_load_with_no_matched_target_keys():
    module = _NestedCheckpointLinear(2, 2)

    with pytest.raises(RuntimeError, match="matched no target parameters"):
        module_state.load_module_state_dict(
            module,
            {"model": {"checkpoint_only": torch.ones(1)}},
            False,
            True,
        )


# ---------------------------------------------------------------------------
# ``_set_precision``: the storage policy
# ---------------------------------------------------------------------------


def test_frozen_component_stores_everything_in_model_precision_except_pins():
    module = _MixedNetwork()

    _set_precision(module)

    assert module.precision_model == torch.bfloat16
    assert module.precision_opt == torch.bfloat16
    assert module.weight.dtype == torch.bfloat16
    assert module.frozen.dtype == torch.bfloat16
    assert module.running.dtype == torch.bfloat16
    assert module.fixed.dtype == torch.float64
    assert module.indices.dtype == torch.long


def test_trainable_component_stores_everything_in_optimization_precision_except_pins():
    module = _MixedNetwork()

    _set_precision(module, precision_opt=torch.float32)

    # The whole component follows ``precision_opt``: ``requires_grad`` can change during training, so
    # frozen parameters and buffers are not split off.
    assert module.weight.dtype == torch.float32
    assert module.frozen.dtype == torch.float32
    assert module.running.dtype == torch.float32
    assert module.fixed.dtype == torch.float64
    assert module.indices.dtype == torch.long
    assert module.weight.requires_grad and not module.frozen.requires_grad


def test_set_precision_casts_gradients_with_their_parameters():
    module = _MixedNetwork()
    module.weight.grad = torch.ones_like(module.weight)

    _set_precision(module, precision_opt=torch.float32)

    assert module.weight.grad.dtype == torch.float32


def test_set_precision_updates_meta_dtypes_without_materializing():
    module = _MixedNetwork(device=torch.device("meta"))

    _set_precision(module, precision_opt=torch.float32)

    assert module.weight.is_meta and module.weight.dtype == torch.float32
    assert module.running.is_meta and module.running.dtype == torch.float32
    assert module.fixed.is_meta and module.fixed.dtype == torch.float64


def test_set_precision_rejects_conflicting_policies_for_a_shared_tensor():
    module = _MixedNetwork()
    module.alias = module.fixed  # the same tensor is pinned under one name and free under another

    with pytest.raises(ValueError, match="shared tensor"):
        _set_precision(module)


def test_none_model_precision_disables_casts():
    class _Uncast(_MixedNetwork):
        @property
        def precision_model(self):
            return None

    module = _Uncast()
    _set_precision(module)

    assert module.precision_model is None and module.precision_opt is None
    assert {module.weight.dtype, module.frozen.dtype, module.running.dtype} == {torch.float64}
    state = torch.ones(2)
    assert state.to(dtype=module.precision_model) is state

    # Trainable storage still follows ``precision_opt``.
    _set_precision(module, precision_opt=torch.float32)
    assert module.weight.dtype == torch.float32 and module.fixed.dtype == torch.float64


def test_module_to_uses_native_uniform_cast_without_redefining_precision():
    module = _MixedNetwork()
    _set_precision(module)

    module.to(dtype=torch.float32)
    assert module.weight.dtype == torch.float32
    assert module.fixed.dtype == torch.float32
    assert module.precision_model == torch.bfloat16

    module.to(device="cpu")  # device-only moves do not touch dtypes
    assert module.weight.dtype == torch.float32 and module.fixed.dtype == torch.float32


def test_unconfigured_component_behaves_like_a_plain_module():
    module = _MixedNetwork()
    assert module.precision_model is None and module.precision_opt is None
    state = torch.ones(2)
    assert state.to(dtype=module.precision_model) is state


def test_preprocessor_casts_owned_modules_through_its_own_to():
    preprocessor = _Wrapper()

    _set_precision(preprocessor)

    assert preprocessor.precision_model == torch.bfloat16
    assert preprocessor.core.weight.dtype == torch.bfloat16


def test_preprocessor_resolves_architecture_model_precision_before_casting():
    preprocessor = _FixedPrecisionWrapper()

    _set_precision(preprocessor)

    assert preprocessor.precision_model == torch.float32
    assert preprocessor.core.weight.dtype == torch.float32


def test_set_precision_logs_architecture_model_precision_override(monkeypatch):
    messages = []
    monkeypatch.setattr(module_lifecycle.logger, "info", messages.append)

    _FixedPrecisionWrapper()._set_precision(precision_model=torch.bfloat16)
    _Wrapper()._set_precision(precision_model=torch.bfloat16)

    assert messages == ["_FixedPrecisionWrapper overrides precision_model from torch.bfloat16 to torch.float32."]


def test_cast_to_model_precision_promotes_with_the_active_autocast_dtype():
    """Under autocast, a primary input is cast to ``precision_model`` promoted with the autocast dtype (FP16 with BF16:
    FP32), so the narrower model precision does not round it, while wider FP64 inputs are still narrowed; a pinned
    module promotes its own. Without an active autocast region the network runs nominally in ``precision_model``, so
    the input is cast to it, and ``None`` casts nothing."""

    class _PinnedBF16(_Wrapper):
        @property
        def precision_model(self):
            return torch.bfloat16

    network = _TinyNetwork()
    network.pinned = _PinnedBF16()
    state = torch.full((2,), 1 + 2**-12, dtype=torch.float64)  # FP16 rounds it to 1
    assert network.cast_to_model_precision(state) is state
    network._set_precision(precision_model=torch.float16)

    assert network.cast_to_model_precision(state).dtype == torch.float16
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
        assert network.cast_to_model_precision(state).dtype == torch.float16
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        promoted = network.cast_to_model_precision(state)
        assert promoted.dtype == torch.float32 and torch.equal(promoted, state.float())
        assert network.pinned.cast_to_model_precision(state).dtype == torch.bfloat16


def test_preprocessor_uses_native_module_traversal():
    class _ModulePreprocessor(FastGenModule):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Linear(2, 2, dtype=torch.float64)

    preprocessor = _ModulePreprocessor()
    _set_precision(preprocessor)
    assert preprocessor.core.weight.dtype == torch.bfloat16


def _record_storage_casts(monkeypatch) -> tuple[list, list]:
    """Record every island resolution and every executed cast of ``FastGenModule._set_precision``."""
    resolved, casts = [], []

    def record_resolve(module, default_dtype, dtype_overrides, exclude=()):
        exclude = list(exclude)
        resolved.append((module, default_dtype, dict(dtype_overrides), exclude))
        return module_state.resolve_storage_dtypes(module, default_dtype, dtype_overrides, exclude)

    def record_cast(module, entries):
        casts.append(module)
        return module_state.cast_to_storage_dtypes(module, entries)

    monkeypatch.setattr(module_lifecycle, "resolve_storage_dtypes", record_resolve)
    monkeypatch.setattr(module_lifecycle, "cast_to_storage_dtypes", record_cast)
    return resolved, casts


def test_set_precision_resolves_each_island_and_casts_once(monkeypatch):
    preprocessor = _Wrapper()
    resolved, casts = _record_storage_casts(monkeypatch)

    _set_precision(preprocessor)

    assert resolved == [(preprocessor, torch.bfloat16, {}, [])] and casts == [preprocessor]
    assert preprocessor.core.weight.dtype == torch.bfloat16


def test_set_precision_refuses_to_cast_wrapped_state(monkeypatch):
    module = _Wrapper()
    # Treat ordinary tensors as wrapper subclasses (e.g. DTensors) without a process group.
    monkeypatch.setattr(module_state, "is_traceable_wrapper_subclass", lambda tensor: True)

    module._set_precision(precision_model=torch.float64)  # a no-op remains valid after wrapping
    with pytest.raises(AssertionError, match="storage before wrapping"):
        module._set_precision(precision_model=torch.bfloat16)


def test_precision_component_is_shared_by_networks_discriminators_and_preprocessors():
    from fastgen.networks.discriminators import Discriminator

    assert issubclass(FastGenNetwork, FastGenModule)
    assert issubclass(Discriminator, FastGenModule)


def test_tensor_boundary_none_dtype_is_identity_and_split_dtype_preserves_gradients():
    tensor = torch.randn(2, 2, requires_grad=True)
    assert tensor.to(dtype=None) is tensor

    network = _TinyNetwork().double()
    network._set_precision(precision_model=torch.float64)
    output = network(tensor, torch.ones(2, dtype=torch.float64))

    assert output.dtype == tensor.dtype == torch.float32
    output.sum().backward()
    assert tensor.grad is not None and tensor.grad.dtype == torch.float32
    assert network.proj.weight.grad is not None and network.proj.weight.grad.dtype == torch.float64


def test_hugging_face_dtype_properties_follow_unsharded_module_to():
    diffusers_model = _DiffusersDtypeProbe()
    transformers_model = BertModel(
        BertConfig(hidden_size=8, intermediate_size=16, num_attention_heads=2, num_hidden_layers=1)
    )

    for module in (diffusers_model, transformers_model):
        assert module.dtype == torch.float32
        module.to(dtype=torch.float64)
        assert module.dtype == torch.float64


# ---------------------------------------------------------------------------
# Root-wide device placement and the persistent ``prepare_module`` lifecycle
# ---------------------------------------------------------------------------


def test_device_is_read_only_and_has_no_exposed_lifecycle_mutators():
    module = _Wrapper()
    with pytest.raises(AttributeError):
        module.device = "cpu"

    assert not hasattr(module, "set_device")
    assert not hasattr(module, "set_precision")
    assert not hasattr(module, "move_to_device")
    assert not hasattr(module, "cast_to_precision")
    assert not hasattr(module, "_move_to_device")

    module._set_device("cpu")
    assert module.device == torch.device("cpu")


def test_device_is_none_until_recorded():
    module = _Wrapper()  # real CPU parameters, but no recorded device: no fallback to their location
    assert module.device is None
    state = torch.ones(2)
    assert state.to(device=module.device) is state

    module.to("cpu")  # a plain move never records
    assert module.device is None

    module._set_device("cpu")
    assert module.device == torch.device("cpu")


def test_private_set_device_stamps_one_normalized_target_across_the_physical_root():
    root = _Composite()
    child = root.container["child"]
    child.inner = _PassThroughChild()

    root._set_device("cpu")

    target = torch.device("cpu")
    for module in (root, child, child.inner):
        assert module._device == target
        assert module.device == target


def test_private_set_device_records_target_without_moving_state():
    root = _Composite()
    before = module_state.tensor_registrations(root)

    root._set_device("meta")

    assert root.device == torch.device("meta")
    assert root.container["child"].device == torch.device("meta")
    after = module_state.tensor_registrations(root)
    assert after.keys() == before.keys()
    for key, registration in after.items():
        assert registration.tensor is before[key].tensor
        assert registration.tensor.device.type == "cpu"


@pytest.mark.parametrize("trainable", [False, True])
def test_private_set_train_mode_applies_mode_and_gradient_policy(trainable):
    module = _Wrapper()
    module.requires_grad_(not trainable)

    module._set_train_mode(trainable)

    assert module.training is trainable
    assert all(parameter.requires_grad is trainable for parameter in module.parameters())


def test_private_set_train_mode_warns_for_parameters_it_enables(monkeypatch):
    module = _Wrapper().requires_grad_(False)
    warnings = []
    monkeypatch.setattr(module_lifecycle.logger, "warning", warnings.append)

    module._set_train_mode(True)

    assert module.training
    assert all(parameter.requires_grad for parameter in module.parameters())
    assert len(warnings) == 1
    assert "_Wrapper:" in warnings[0]
    assert "core.weight" in warnings[0]
    assert "core.bias" in warnings[0]


def test_private_set_train_mode_does_not_warn_for_parameters_kept_frozen(monkeypatch):
    class _PartiallyProtected(FastGenModule):
        def __init__(self):
            super().__init__()
            self.trainable = torch.nn.Linear(2, 2)
            self.protected = torch.nn.Linear(2, 2).requires_grad_(False)

        def requires_grad_(self, requires_grad: bool = True):
            super().requires_grad_(requires_grad)
            if requires_grad:
                self.protected.requires_grad_(False)
            return self

    module = _PartiallyProtected()
    warnings = []
    monkeypatch.setattr(module_lifecycle.logger, "warning", warnings.append)

    module._set_train_mode(True)

    assert module.training
    assert all(parameter.requires_grad for parameter in module.trainable.parameters())
    assert not any(parameter.requires_grad for parameter in module.protected.parameters())
    assert warnings == []


def test_private_set_train_mode_lets_nested_modules_apply_their_own_policy(monkeypatch):
    class _Protected(FastGenModule):
        """A nested module whose own policy keeps ``protected`` frozen, like a LoRA base."""

        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Linear(2, 2)
            self.protected = torch.nn.Linear(2, 2).requires_grad_(False)

        def requires_grad_(self, requires_grad: bool = True):
            super().requires_grad_(requires_grad)
            self.protected.requires_grad_(False)
            return self

    class _Root(FastGenModule):
        def __init__(self):
            super().__init__()
            self.own = torch.nn.Linear(2, 2)
            self.container = torch.nn.ModuleDict({"child": _Protected()})

    root = _Root()
    child = root.container["child"]
    warnings = []
    monkeypatch.setattr(module_lifecycle.logger, "warning", warnings.append)

    root._set_train_mode(True)

    # ``nn.Module.requires_grad_`` never calls the nested override; the root must let the child re-apply it.
    assert root.training and child.training
    assert all(parameter.requires_grad for parameter in root.own.parameters())
    assert all(parameter.requires_grad for parameter in child.adapter.parameters())
    assert not any(parameter.requires_grad for parameter in child.protected.parameters())
    # The root unfreezes ``protected`` only until the child re-freezes it, so nothing is reported.
    assert warnings == []

    root._set_train_mode(False)

    assert not root.training and not child.training
    assert not any(parameter.requires_grad for parameter in root.parameters())


def test_prepare_module_without_device_skips_placement(monkeypatch):
    def unexpected_move(*args, **kwargs):
        raise AssertionError(f"move_and_sync_module was called with {args=}, {kwargs=}")

    monkeypatch.setattr(module_lifecycle, "move_and_sync_module", unexpected_move)
    root = _Composite()
    child = root.container["child"]

    root.prepare_module(device=None, precision_model=torch.bfloat16)

    # Precision, mode, and gradients are applied; placement is the caller's.
    assert root.is_prepared and child.is_prepared
    assert root.device is None and child.device is None
    assert root.own.weight.dtype == torch.bfloat16 and child.core.weight.dtype == torch.float32
    assert not root.training and not any(parameter.requires_grad for parameter in root.parameters())

    with pytest.raises(RuntimeError, match="already prepared"):
        root.prepare_module(device=None, precision_model=torch.bfloat16)
    with pytest.raises(RuntimeError, match="already prepared"):
        root.prepare_module(device="cpu", precision_model=torch.bfloat16)


def test_prepare_module_eagerly_places_the_complete_physical_root_once(monkeypatch):
    root = _Composite()
    child = root.container["child"]
    calls = []

    def recording_move(module, device):
        calls.append((module, torch.device(device)))
        module.to(device=device)

    monkeypatch.setattr(module_lifecycle, "move_and_sync_module", recording_move)

    root.prepare_module(device="meta", precision_model=torch.bfloat16)

    target = torch.device("meta")
    assert calls == [(root, target)]
    assert root.device == target and child.device == target
    assert root.own.weight.device == target and child.core.weight.device == target


def test_prepare_module_places_the_requested_device_when_a_base_shadows_device(monkeypatch):
    class _ParameterDevice:
        @property
        def device(self):
            return next(self.parameters()).device

    # Listed before FastGenModule, the base's ``device`` wins without tripping the class-definition guard.
    class _BaseFirst(_ParameterDevice, _Wrapper):
        pass

    calls = []
    monkeypatch.setattr(
        module_lifecycle, "move_and_sync_module", lambda module, device: calls.append(torch.device(device))
    )
    module = _BaseFirst()
    assert module.device == torch.device("cpu")

    module.prepare_module(device="meta")

    assert calls == [torch.device("meta")]


@pytest.mark.parametrize("device", [None, "cpu"])
def test_prepare_module_rejects_buffer_aliases_before_any_mutation(device):
    module = _SharedBufferModule()
    shared = module.running
    indices = module.indices

    with pytest.raises(AssertionError, match="multiple distinct buffer slots"):
        module.prepare_module(device=device, precision_model=torch.bfloat16)

    assert module.running is module.running_alias is module.child.cached is shared
    assert module.weight is module.child.weight_alias
    assert module.indices is module.child.indices is indices
    assert module.running.dtype == module.weight.dtype == torch.float64
    assert module.indices.dtype == torch.long
    assert module._non_persistent_buffers_set == {"running_alias"}
    assert module.child._non_persistent_buffers_set == {"cached", "indices"}
    torch.testing.assert_close(module.running, torch.tensor([1.25, -2.5], dtype=torch.float64))
    assert not module.is_prepared
    assert module.device is None and module.precision_model is None
    assert module.training and module.weight.requires_grad


def _prepare_tied_parameters_worker(rank, world_size, precision, source_device):
    with torch.device(source_device if rank else "cpu"):
        module = FastGenModule()
        module.left = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        module.right = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        module.right.weight = module.left.weight
        module.left.register_buffer("cache", torch.full((2,), 7.0, dtype=torch.float64), persistent=False)
        module.left.register_buffer("indices", torch.arange(2))
        module.left_alias = module.left
    device = torch.device("cuda", torch.cuda.current_device())

    module.prepare_module(device=device, precision_model=precision)

    assert module.left is module.left_alias
    assert module.left.cache is module.left_alias.cache
    assert module.left.weight is module.right.weight
    assert all(tensor.device == device for tensor in [*module.parameters(), *module.buffers()])
    assert module.left.cache.dtype == module.left.weight.dtype == (precision or torch.float64)
    assert module.left.indices.dtype == torch.long
    assert module.left._non_persistent_buffers_set == {"cache"}
    assert module.is_prepared and module.device == device
    torch.testing.assert_close(
        module.left.cache, torch.full((2,), 7.0, dtype=precision or torch.float64, device=device)
    )
    return True


@RunIf(min_gpus=2)
@pytest.mark.parametrize("precision", [None, torch.bfloat16])
@pytest.mark.parametrize("source_device", ["cpu", "meta"])
def test_prepare_module_preserves_tied_parameters_and_shared_submodules(precision, source_device):
    assert run_distributed_test(
        _prepare_tied_parameters_worker,
        world_size=2,
        timeout=120,
        nccl_timeout=30,
        precision=precision,
        source_device=source_device,
    )


def test_prepare_module_rejects_parameter_buffer_aliases_before_casting():
    module = _SharedBufferModule()
    module.register_buffer("weight_as_buffer", module.weight)

    with pytest.raises(AssertionError, match="both a parameter and a buffer"):
        module.prepare_module(device="cpu", precision_model=torch.bfloat16)

    assert module.weight is module.child.weight_alias is module.weight_as_buffer
    assert module.weight.dtype == torch.float64
    torch.testing.assert_close(module.weight, torch.full((2,), 3.0, dtype=torch.float64))
    assert module.device is None and module.precision_model is None
    assert not module.is_prepared


@RunIf(min_gpus=1)
def test_prepare_module_device_move_rejects_parameter_buffer_aliases_without_splitting_them():
    module = _SharedBufferModule()
    module.register_buffer("weight_as_buffer", module.weight)

    with pytest.raises(AssertionError, match="both a parameter and a buffer"):
        module.prepare_module(device="cuda")

    assert module.weight is module.child.weight_alias is module.weight_as_buffer
    assert module.running is module.running_alias is module.child.cached
    assert all(tensor.device.type == "cpu" for tensor in [*module.parameters(), *module.buffers()])
    assert not module.is_prepared


def test_duplicate_preparation_rejects_runtime_buffer_ties_without_mutation():
    root = _Composite()
    child = root.container["child"]
    child.register_buffer("cache", torch.ones(2))
    root.prepare_module(device="cpu", precision_model=torch.bfloat16)
    # A forward (e.g. a dynamic RoPE cache reset) may create a buffer tie.
    child.register_buffer("original_cache", child.cache, persistent=False)
    shared = child.cache
    weight = root.own.weight

    with pytest.raises(RuntimeError, match="already prepared"):
        root.prepare_module(device="cpu", precision_model=torch.bfloat16)
    with pytest.raises(RuntimeError, match="already prepared"):
        root.prepare_module(device="cpu", precision_model=torch.float32, trainable=True)

    assert child.cache is child.original_cache is shared
    assert root.own.weight is weight and weight.dtype == torch.bfloat16
    assert root.precision_model == torch.bfloat16
    assert root.is_prepared and child.is_prepared
    assert not root.training and not child.training and not weight.requires_grad


@pytest.mark.parametrize("setting", ["overwrite", "swap"])
def test_prepare_module_rejects_nondefault_torch_settings_before_lifecycle_mutation(setting):
    module = _Wrapper()
    before = (module.is_prepared, module.device, module.precision_model, module.training)
    weight = module.core.weight
    values = weight.detach().clone()
    setter = getattr(torch.__future__, f"set_{setting}_module_params_on_conversion")
    original = getattr(torch.__future__, f"get_{setting}_module_params_on_conversion")()
    try:
        setter(True)
        with pytest.raises(AssertionError, match=f"set_{setting}_module_params_on_conversion"):
            module.prepare_module(device="cpu", precision_model=torch.bfloat16, trainable=True)
    finally:
        setter(original)

    assert (module.is_prepared, module.device, module.precision_model, module.training) == before
    assert module.core.weight is weight
    torch.testing.assert_close(weight, values)


def test_placement_free_preparation_without_a_process_group_is_local(monkeypatch):
    def forbidden_collective(*args, **kwargs):
        pytest.fail("Preparation without an active process group must not run collectives")

    monkeypatch.setattr(torch.distributed, "all_reduce", forbidden_collective)
    local = _Wrapper()
    local.prepare_module(device=None, precision_model=torch.bfloat16)
    assert local.is_prepared and local.core.weight.dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match="already prepared"):
        local.prepare_module(device=None, precision_model=torch.bfloat16)


def _prepare_module_rank_local_state_failures_impl(rank, world_size, mode):
    import torch.distributed as dist

    results = {}
    for failure in (
        "buffer_alias",
        "parameter_buffer_alias",
        "inference_state",
        "complex_parameter",
        "overwrite",
        "swap",
        "prepared_root",
        "prepared_child",
    ):
        module = FastGenModule()
        module.child = FastGenModule()
        module.child.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        module.weight = module.child.weight
        module.register_buffer("cache", torch.ones(2, dtype=torch.float64))
        if rank == 1:
            if failure == "buffer_alias":
                module.register_buffer("cache_alias", module.cache)
            elif failure == "parameter_buffer_alias":
                module.register_buffer("weight_as_buffer", module.weight)
            elif failure == "inference_state":
                with torch.inference_mode():
                    module.cache = torch.ones_like(module.cache)
            elif failure == "complex_parameter":
                module.child.phase = torch.nn.Parameter(torch.ones(2, dtype=torch.complex128))
            elif failure == "prepared_root":
                module._is_prepared = True
            elif failure == "prepared_child":
                module.child._is_prepared = True

        registrations = module_state.tensor_registrations(module)
        signature = module_state.registration_signature(registrations, include_device=True)
        values = {key: entry.tensor.clone() for key, entry in registrations.items()}
        lifecycle = [
            (child.is_prepared, child.device, child.precision_model, child.precision_opt, child.training)
            for child in module.modules()
        ]
        shard_calls = []

        class ShardingContext:
            device = torch.device("cpu")

            def _shard_module(self, candidate):
                shard_calls.append(candidate)
                raise AssertionError("Sharding began before all ranks passed state validation")

        context = ShardingContext() if mode == "fsdp" else None
        target = torch.device("cuda", rank)
        if context is not None:
            target = None
        setter = None
        if rank == 1 and failure in {"overwrite", "swap"}:
            setter = getattr(torch.__future__, f"set_{failure}_module_params_on_conversion")
            original = getattr(torch.__future__, f"get_{failure}_module_params_on_conversion")()
            setter(True)
        try:
            module.prepare_module(device=target, precision_model=torch.bfloat16, trainable=True, fsdp_context=context)
        except RuntimeError as error:
            expected = {
                "buffer_alias": "multiple distinct buffer slots",
                "parameter_buffer_alias": "both a parameter and a buffer",
                "inference_state": "inference tensor",
                "complex_parameter": "does not support complex parameters",
                "overwrite": "set_overwrite_module_params_on_conversion",
                "swap": "set_swap_module_params_on_conversion",
                "prepared_root": "already prepared",
                "prepared_child": "already prepared",
            }[failure]
            rejected = expected in str(error) and "rank 1" in str(error)
        else:
            rejected = False
        finally:
            if setter is not None:
                setter(original)

        after = module_state.tensor_registrations(module)
        per_rank = [None] * world_size
        dist.all_gather_object(
            per_rank,
            {
                "rejected": rejected,
                "no_sharding": not shard_calls,
                "metadata_unchanged": module_state.registration_signature(after, include_device=True) == signature,
                "identity_unchanged": all(entry.tensor is registrations[key].tensor for key, entry in after.items()),
                "values_unchanged": all(torch.equal(entry.tensor, values[key]) for key, entry in after.items()),
                "lifecycle_unchanged": lifecycle
                == [
                    (child.is_prepared, child.device, child.precision_model, child.precision_opt, child.training)
                    for child in module.modules()
                ],
            },
        )
        results[failure] = per_rank
    return results


@RunIf(min_gpus=2)
@pytest.mark.parametrize("mode", ["eager", "fsdp"])
def test_prepare_module_propagates_rank_local_state_failures_before_mutation(mode):
    results = run_distributed_test(
        test_fn=_prepare_module_rank_local_state_failures_impl,
        world_size=2,
        timeout=120,
        nccl_timeout=30,
        mode=mode,
    )
    assert results is not None
    assert all(all(checks.values()) for per_rank in results.values() for checks in per_rank), results


def test_prepare_module_without_device_rejects_unmaterialized_meta_modules():
    with torch.device("meta"):
        module = _Wrapper()

    with pytest.raises(RuntimeError, match="meta tensors"):
        module.prepare_module(device=None, precision_model=torch.bfloat16)
    assert not module.is_prepared

    # An FSDP context owns placement and materialization, so no target device is fine there.
    class ShardingContext:
        device = torch.device("cpu")

        def __init__(self):
            self.modules = []

        def _shard_module(self, candidate):
            self.modules.append(candidate)
            return candidate

    context = ShardingContext()
    module.prepare_module(device=None, precision_model=torch.bfloat16, fsdp_context=context)
    assert context.modules == [module] and module.is_prepared and module._is_fsdp


def test_prepare_module_rejects_inference_context_and_inference_tensors():
    module = _Wrapper()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="must not run under torch.inference_mode"):
        module.prepare_module(device="cpu", precision_model=torch.float32)
    assert not module.is_prepared

    with torch.inference_mode():
        built_in_inference = _Wrapper()
    with pytest.raises(RuntimeError, match=r"2 inference tensor\(s\), first 8: \['core.weight', 'core.bias'\]"):
        built_in_inference.prepare_module(device="cpu", precision_model=torch.float32)
    assert not built_in_inference.is_prepared

    module.prepare_module(device="cpu", precision_model=torch.float32)
    assert module.is_prepared

    with torch.inference_mode(), pytest.raises(RuntimeError, match="already prepared"):
        module.prepare_module(device="cpu", precision_model=torch.float32)
    assert module.is_prepared


def test_set_precision_can_record_without_casting_and_casts_by_default():
    root = _Composite()
    child = root.container["child"]

    root._record_precision(precision_model=torch.bfloat16)

    # Recorded on the root and on the nested module; nothing is cast yet.
    assert root.precision_model == torch.bfloat16 and root.precision_opt == torch.bfloat16
    assert child.precision_model == torch.float32 and child.precision_opt == torch.float32
    assert root.own.weight.dtype == torch.float64 and child.core.weight.dtype == torch.float64

    root._set_precision(precision_model=torch.bfloat16)

    assert root.own.weight.dtype == torch.bfloat16 and child.core.weight.dtype == torch.float32


def test_set_precision_default_cast_includes_modules_attached_after_a_record_only_call():
    root = _Composite()
    root._record_precision(precision_model=torch.bfloat16)
    late = _PassThroughChild()
    root.container["late"] = late

    root._set_precision(precision_model=torch.bfloat16)

    assert late.precision_model == torch.bfloat16
    assert late.core.weight.dtype == torch.bfloat16


def test_private_set_precision_configures_storage_without_marking_prepared():
    component = _TinyNetwork()

    component._set_precision(
        precision_model=torch.float64,
        precision_opt=torch.float32,
    )
    assert component.proj.weight.dtype is torch.float32
    assert component.precision_model is torch.float64
    assert component.precision_opt is torch.float32
    assert not component.is_prepared


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"device": None},
        {"device": "meta"},
        {"precision_model": torch.float32},
        {"precision_opt": torch.float32},
        {"trainable": True},
    ],
    ids=["identical", "no_device", "device", "model_precision", "storage_precision", "trainable"],
)
def test_prepare_module_rejects_every_repeat_request_without_mutation(overrides, monkeypatch):
    module = _Wrapper()
    assert module.prepare_module(device="cpu", precision_model=torch.bfloat16) is module
    weight = module.core.weight
    values = weight.detach().clone()

    def unexpected_lifecycle_setup(*args, **kwargs):
        pytest.fail("A duplicate request must fail before any lifecycle mutation")

    monkeypatch.setattr(module, "_set_precision", unexpected_lifecycle_setup)
    monkeypatch.setattr(module, "_set_train_mode", unexpected_lifecycle_setup)
    monkeypatch.setattr(module, "_set_device", unexpected_lifecycle_setup)
    kwargs = {"device": "cpu", "precision_model": torch.bfloat16} | overrides
    with pytest.raises(RuntimeError, match="already prepared.*only be prepared once"):
        module.prepare_module(**kwargs)

    assert module.is_prepared and module.device == torch.device("cpu")
    assert module.precision_model is torch.bfloat16 and module._precision_opt is None
    assert module.core.weight is weight and weight.dtype == torch.bfloat16
    torch.testing.assert_close(weight, values)
    assert not module.training and not any(parameter.requires_grad for parameter in module.parameters())


def test_prepare_module_delegates_to_fsdp_after_precision_and_role_setup(monkeypatch):
    component = _TinyNetwork()
    events = []

    def unexpected_move(*args, **kwargs):
        raise AssertionError(f"move_and_sync_module was called with {args=}, {kwargs=}")

    monkeypatch.setattr(module_lifecycle, "move_and_sync_module", unexpected_move)

    class InspectingFSDPContext:
        device = torch.device("cpu")

        def _shard_module(self, candidate):
            assert candidate is component
            assert candidate.precision_model is torch.bfloat16
            assert candidate.precision_opt is torch.float32
            assert candidate.proj.weight.dtype is torch.float32
            assert candidate.device == torch.device("cpu")
            assert candidate.training
            assert all(parameter.requires_grad for parameter in candidate.parameters())
            assert not candidate.is_prepared
            events.append("shard")
            return candidate

    component.prepare_module(
        device="cpu",
        precision_model=torch.bfloat16,
        precision_opt=torch.float32,
        trainable=True,
        fsdp_context=InspectingFSDPContext(),
    )

    assert events == ["shard"]
    assert component.is_prepared and component._is_fsdp


def test_repeat_preparation_with_same_or_no_fsdp_context_raises(monkeypatch):
    component = _TinyNetwork()
    calls = []

    class RecordingFSDPContext:
        device = torch.device("cpu")

        def _shard_module(self, candidate):
            assert candidate is component
            assert not candidate.is_prepared
            calls.append(candidate)
            return candidate

    context = RecordingFSDPContext()
    component.prepare_module(
        device="cpu",
        precision_model=torch.bfloat16,
        precision_opt=torch.float32,
        trainable=True,
        fsdp_context=context,
    )

    def unexpected_lifecycle_setup(*args, **kwargs):
        raise AssertionError(f"lifecycle setup was repeated with {args=}, {kwargs=}")

    monkeypatch.setattr(component, "_set_precision", unexpected_lifecycle_setup)
    for requested_context in (context, None):
        with pytest.raises(RuntimeError, match="already prepared"):
            component.prepare_module(
                device="cpu",
                precision_model=torch.bfloat16,
                precision_opt=torch.float32,
                trainable=True,
                fsdp_context=requested_context,
            )

    assert calls == [component]
    assert component.is_prepared and component._is_fsdp


def test_repeat_preparation_with_other_fsdp_context_raises_without_dispatch():
    component = _TinyNetwork()

    class RecordingFSDPContext:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = []

        def _shard_module(self, candidate):
            self.calls.append(candidate)

    installed_context = RecordingFSDPContext()
    replacement_context = RecordingFSDPContext()
    kwargs = {
        "device": "cpu",
        "precision_model": torch.bfloat16,
        "precision_opt": torch.float32,
        "trainable": True,
    }
    component.prepare_module(**kwargs, fsdp_context=installed_context)

    with pytest.raises(RuntimeError, match="already prepared"):
        component.prepare_module(**kwargs, fsdp_context=replacement_context)

    assert installed_context.calls == [component]
    assert replacement_context.calls == []
    assert component.is_prepared and component._is_fsdp


def test_repeat_preparation_of_eager_prepared_module_raises_before_sharding():
    root = _Composite()
    child = root.container["child"]
    root.prepare_module(device="cpu", precision_model=torch.bfloat16)

    class UnexpectedFSDPContext:
        device = torch.device("cpu")

        def _shard_module(self, _candidate):
            raise AssertionError("late FSDP transition must fail before sharding")

    with pytest.raises(RuntimeError, match="already prepared"):
        root.prepare_module(
            device="cpu",
            precision_model=torch.bfloat16,
            fsdp_context=UnexpectedFSDPContext(),
        )

    assert root.is_prepared and child.is_prepared
    assert not root._is_fsdp and not child._is_fsdp


def test_prepare_module_does_not_reach_preprocessors_outside_the_module_tree():
    network = _MixedNetwork()
    network.preprocessors = {"vae": _Wrapper()}

    network.prepare_module(device="cpu", precision_model=torch.float32)

    # Preprocessors live in a plain dict outside the module tree and are prepared individually
    # by ``FastGenModel.init_preprocessors``; preparing the network must not reach them.
    assert network.preprocessors["vae"]._device is None
    assert "vae" not in dict(network.named_children())


def test_preprocessor_dict_does_not_register_modules_on_the_network():
    network = _TinyNetwork()
    preprocessor = _PlainPreprocessor()
    network.preprocessors["vae"] = preprocessor

    assert network.preprocessors == {"vae": preprocessor}
    assert "preprocessors" not in network._modules
    network_parameters = {id(parameter) for parameter in network.parameters()}
    assert network_parameters.isdisjoint(id(parameter) for parameter in preprocessor.parameters())
    assert all(not name.startswith("preprocessors.") for name in network.state_dict())


# ---------------------------------------------------------------------------
# Nested FastGen modules: precision islands within one physical device root
# ---------------------------------------------------------------------------


class _PinnedChild(FastGenModule):
    """A nested module that resolves fp32 for itself regardless of its parent's request."""

    def __init__(self):
        super().__init__()
        self.core = torch.nn.Linear(2, 2, dtype=torch.float64)

    @property
    def precision_model(self):
        return torch.float32


class _PassThroughChild(FastGenModule):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Linear(2, 2, dtype=torch.float64)


class _Composite(FastGenModule):
    """A root without overrides whose plain container holds a nested FastGen module."""

    def __init__(self):
        super().__init__()
        self.own = torch.nn.Linear(2, 2, dtype=torch.float64)
        self.container = torch.nn.ModuleDict({"child": _PinnedChild()})


def test_nearest_nested_modules_do_not_descend_into_fastgen_modules():
    root = _Composite()
    child = root.container["child"]
    child.inner = _PassThroughChild()  # a grandchild belongs to the child, not to the root
    root.alias = child  # the same nested module registered twice is yielded once

    assert list(root._nearest_nested_modules()) == [child]
    assert list(child._nearest_nested_modules()) == [child.inner]
    assert list(child.inner._nearest_nested_modules()) == []


def test_fastgen_modules_include_the_root_and_every_depth():
    root = _Composite()
    child = root.container["child"]
    child.inner = _PassThroughChild()
    root.alias = child  # the same nested module registered twice is yielded once

    assert list(root._fastgen_modules()) == [root, child, child.inner]
    assert list(root._named_fastgen_modules()) == [
        ("", root),
        ("container.child", child),
        ("container.child.inner", child.inner),
    ]
    assert list(child.inner._fastgen_modules()) == [child.inner]


def test_root_without_overrides_keeps_a_nested_modules_own_precision():
    root = _Composite()

    _set_precision(root)

    child = root.container["child"]
    assert root.precision_model == torch.bfloat16 and root.own.weight.dtype == torch.bfloat16
    assert child._precision_model == torch.bfloat16
    assert child._precision_opt is None
    assert child.precision_model == torch.float32 and child.precision_opt == torch.float32
    assert child.core.weight.dtype == torch.float32


def test_parent_precision_override_does_not_replace_child_requests():
    class _FP64Root(_Composite):
        @property
        def precision_model(self):
            return torch.float64

    root = _FP64Root()
    root.container["plain"] = _PassThroughChild()

    _set_precision(root)

    assert root.precision_model == torch.float64 and root.own.weight.dtype == torch.float64
    assert root.container["plain"]._precision_model == torch.bfloat16
    assert root.container["plain"].precision_model == torch.bfloat16
    assert root.container["plain"].core.weight.dtype == torch.bfloat16
    assert root.container["child"].precision_model == torch.float32


def test_precision_opt_override_is_rejected_at_class_definition():
    with pytest.raises(TypeError, match="overrides precision_opt; only precision_model"):

        class _FP64StorageRoot(_Composite):
            @property
            def precision_opt(self):
                return torch.float64


def test_precision_opt_reaches_nested_modules():
    root = _Composite()
    root.container["plain"] = _PassThroughChild()

    _set_precision(root, precision_opt=torch.float64)

    assert root.own.weight.dtype == torch.float64
    assert root.container["plain"].core.weight.dtype == torch.float64
    # An explicit storage precision also applies to a child with its own model precision.
    assert root.container["child"].precision_model == torch.float32
    assert root.container["child"].core.weight.dtype == torch.float64


@pytest.mark.parametrize("sharing", ["parameter", "plain_child"])
@pytest.mark.parametrize("conflict", ["model_precision", "fixed_override", "preserve_dtype"])
def test_nested_shared_storage_conflicts_fail_before_casting(sharing, conflict):
    root = _Composite()
    left = root.container["child"]
    right = root.other = _PassThroughChild()
    if sharing == "parameter":
        right.core.weight = left.core.weight
    else:
        right.core = left.core
    if conflict == "fixed_override":
        right._fixed_precision_overrides = lambda: {"core.weight": torch.float64}
    original = [(tensor, tensor.clone()) for tensor in (*root.parameters(), *root.buffers())]

    with pytest.raises(ValueError, match="shared tensor has conflicting storage dtypes"):
        root.prepare_module(
            "cpu",
            precision_model=None if conflict == "preserve_dtype" else torch.float64,
            precision_opt=torch.float32 if conflict == "fixed_override" else None,
        )

    for tensor, value in original:
        torch.testing.assert_close(tensor, value)
    assert left.core.weight is right.core.weight
    assert not any(module.is_prepared for module in root.modules() if isinstance(module, FastGenModule))


@pytest.mark.parametrize("sharing", ["parameter", "plain_child", "fastgen_child", "parent_parameter"])
@pytest.mark.parametrize("precision_opt", [None, torch.float64])
def test_compatible_shared_storage_survives_nested_preparation(sharing, precision_opt):
    root = _Composite()
    child = root.container["child"]
    if sharing == "fastgen_child":
        root.alias = child
    elif sharing == "parent_parameter":
        root.weight = child.core.weight
    else:
        # Different compute policies are compatible when an explicit storage dtype unifies them.
        root.other = _PassThroughChild() if precision_opt is not None else _PinnedChild()
        if sharing == "plain_child":
            root.other.core = child.core
        else:
            root.other.core.weight = child.core.weight
    weight = child.core.weight

    root.prepare_module("cpu", precision_model=torch.bfloat16, precision_opt=precision_opt)

    assert root.is_prepared and child.is_prepared
    assert child.core.weight is weight and weight.dtype == (precision_opt or torch.float32)
    if sharing == "parent_parameter":
        assert root.weight is weight
    elif sharing == "fastgen_child":
        assert root.alias is child
    else:
        assert root.other.core.weight is weight


def test_matching_fixed_overrides_take_precedence_for_shared_storage():
    root = _Composite()
    child = root.container["child"]
    root.other = _PassThroughChild()
    root.other.core.weight = child.core.weight
    for module in (child, root.other):
        module._fixed_precision_overrides = lambda: {"core.weight": torch.float32}

    root.prepare_module("cpu", precision_model=torch.bfloat16, precision_opt=torch.float64)

    assert child.core.weight is root.other.core.weight
    assert child.core.weight.dtype == torch.float32
    assert child.core.bias.dtype == root.other.core.bias.dtype == torch.float64


def test_device_override_is_rejected_at_class_definition():
    with pytest.raises(TypeError, match="overrides device; only precision_model"):

        class _PinnedDeviceChild(_PinnedChild):
            @property
            def device(self):
                return torch.device("meta")


def test_override_naming_a_nested_tensor_raises():
    class _Overreaching(_Composite):
        def _fixed_precision_overrides(self):
            return {"container.child.core.weight": torch.float64}

    with pytest.raises(AssertionError, match="nested module state"):
        _set_precision(_Overreaching())


@pytest.mark.parametrize("child_precision", [torch.float64, torch.bfloat16])
def test_nested_module_prepared_separately_is_rejected(child_precision):
    root = _Composite()
    child = root.container["child"]
    child.prepare_module(device="cpu", precision_model=child_precision)
    weight = root.own.weight
    values = weight.detach().clone()

    with pytest.raises(RuntimeError, match="_PinnedChild is already prepared"):
        root.prepare_module(device="cpu", precision_model=torch.bfloat16)
    assert not root.is_prepared and child.is_prepared
    assert root.precision_model is None and root.device is None
    assert root.own.weight is weight
    torch.testing.assert_close(weight, values)
    assert child._precision_model == child_precision and child.precision_model == torch.float32


def test_composite_root_resolves_each_island_once_and_casts_the_root_once(monkeypatch):
    root = _Composite()
    child = root.container["child"]
    resolved, casts = _record_storage_casts(monkeypatch)

    _set_precision(root)

    # Each precision island resolves its own state once; the parent excludes child-owned state.
    assert [(module, exclude) for module, _dtype, _overrides, exclude in resolved] == [(root, [child]), (child, [])]
    assert casts == [root]
    assert root.own.weight.dtype == torch.bfloat16
    assert child.core.weight.dtype == torch.float32


def test_prepare_module_marks_nested_modules_prepared():
    root = _Composite()
    child = root.container["child"]

    root.prepare_module(device="cpu", precision_model=torch.bfloat16)

    assert root.is_prepared and child.is_prepared
    assert child.device == torch.device("cpu") and child.core.weight.dtype == torch.float32
    with pytest.raises(AttributeError, match="is_prepared"):
        child.is_prepared = False

    # Preparation through the root also completes the child's preparation.
    for precision in (torch.bfloat16, torch.float64):
        with pytest.raises(RuntimeError, match="already prepared"):
            child.prepare_module(device="cpu", precision_model=precision)
    assert child.core.weight.dtype == torch.float32


def test_preparation_failure_before_mutation_allows_retry(monkeypatch):
    root = _Composite()
    child = root.container["child"]
    weight = root.own.weight
    values = weight.detach().clone()

    def failing_set_precision(*args, **kwargs):
        raise RuntimeError("boom")

    with monkeypatch.context() as patch:
        patch.setattr(root, "_set_precision", failing_set_precision)
        with pytest.raises(RuntimeError, match="boom"):
            root.prepare_module(device="cpu", precision_model=torch.bfloat16)

    assert not root.is_prepared and not child.is_prepared
    assert root.precision_model is None and root.device is None
    assert root.own.weight is weight
    torch.testing.assert_close(weight, values)

    root.prepare_module(device="cpu", precision_model=torch.bfloat16)
    assert root.is_prepared and child.is_prepared


@pytest.mark.parametrize("failure", ["cast", "placement", "shard"])
def test_failed_preparation_leaves_tree_unprepared_without_rollback(monkeypatch, failure):
    root = _Composite()
    child = root.container["child"]
    torch.nn.init.constant_(root.own.weight, 1.0 + 2**-12)

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    class PartialShard:
        device = torch.device("cpu")

        def _shard_module(self, module):
            module.own.register_forward_pre_hook(lambda *args: None)
            fail()

    with monkeypatch.context() as patch:
        if failure == "cast":
            patch.setattr(child, "_apply", fail)
        elif failure == "placement":
            patch.setattr(module_lifecycle, "move_and_sync_module", fail)
        with pytest.raises(RuntimeError, match="injected failure"):
            root.prepare_module(
                "cpu",
                precision_model=torch.bfloat16,
                fsdp_context=PartialShard() if failure == "shard" else None,
            )

    # The first cast already lost information, so converting back to FP32 cannot repair it.
    assert root.own.weight.dtype == torch.bfloat16
    assert root.own.weight[0, 0].item() == 1.0
    assert not root.is_prepared and not child.is_prepared
    assert not root._is_fsdp and not child._is_fsdp


def test_preparation_validation_failure_can_be_fixed_before_retrying():
    module = _Wrapper()
    module.register_buffer("invalid_alias", module.core.weight)
    with pytest.raises(AssertionError, match="both a parameter and a buffer"):
        module.prepare_module("cpu", precision_model=torch.bfloat16)
    assert not module.is_prepared
    assert module.core.weight.dtype == torch.float64
    del module.invalid_alias
    module.prepare_module("cpu", precision_model=torch.float32)
    assert module.is_prepared and module.core.weight.dtype == torch.float32


def test_prepare_module_rejects_extra_state_before_mutation():
    class _Stateful(torch.nn.Linear):
        def get_extra_state(self):
            return {"step": 3}

        def set_extra_state(self, state):
            pass

    module = _Wrapper()
    module.stateful = _Stateful(2, 2, dtype=torch.float64)
    with pytest.raises(RuntimeError, match=r"extra_state \(\['stateful'\]\)"):
        module.prepare_module("cpu", precision_model=torch.bfloat16)
    assert not module.is_prepared
    assert module.core.weight.dtype == torch.float64 and module.stateful.weight.dtype == torch.float64


def test_prepare_module_rejects_complex_parameters_but_keeps_complex_buffers():
    module = _Wrapper()
    rotation = torch.polar(torch.ones(2), torch.arange(2.0))
    module.core.phase = torch.nn.Parameter(rotation.clone())
    with pytest.raises(RuntimeError, match=r"does not support complex parameters.*\['core\.phase'\]"):
        module.prepare_module("cpu", precision_model=torch.bfloat16)
    assert not module.is_prepared
    assert module.core.weight.dtype == torch.float64 and module.core.phase.dtype == torch.complex64

    del module.core.phase
    module.register_buffer("rotation", rotation.clone())
    module.prepare_module("cpu", precision_model=torch.bfloat16)
    assert module.is_prepared and module.core.weight.dtype == torch.bfloat16
    assert module.rotation.dtype == torch.complex64 and torch.equal(module.rotation, rotation)


def _failed_cast_readiness_worker(rank, world_size):
    import torch.distributed as dist

    root = _Composite()
    child = root.container["child"]
    if rank == 1:

        def fail(*args, **kwargs):
            raise RuntimeError("injected cast failure")

        child._apply = fail
    with pytest.raises(RuntimeError, match="injected cast failure"):
        root.prepare_module("cuda", precision_model=torch.bfloat16)
    checks = [None] * world_size
    dist.all_gather_object(
        checks,
        not root.is_prepared
        and not child.is_prepared
        and not root._is_fsdp
        and root.own.weight.dtype == torch.bfloat16,
    )
    return checks


@RunIf(min_gpus=2)
def test_failed_cast_leaves_the_tree_unprepared_on_every_rank():
    checks = run_distributed_test(_failed_cast_readiness_worker, world_size=2, timeout=120, nccl_timeout=30)
    assert checks is not None and all(checks)


@pytest.mark.parametrize("network_type", ["FastGenNetwork", "LTX2", "MiniMaxH3", "WanMoE"])
def test_noise_schedule_initializers_assert_before_changes_after_preparation(network_type):
    from fastgen.networks.LTX2.network import LTX2
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3
    from fastgen.networks.Wan.wan_moe import WanMoE

    network = _TinyNetwork()
    scheduler = network.noise_scheduler
    schedule_type = network.schedule_type
    network.prepare_module("cpu", precision_model=torch.float32)

    # Call each construction hook on a lightweight host: the assertion must precede even
    # architecture-specific field access, as well as scheduler or schedule_type mutation.
    cls = {"FastGenNetwork": FastGenNetwork, "LTX2": LTX2, "MiniMaxH3": MiniMaxH3, "WanMoE": WanMoE}[network_type]
    with pytest.raises(AssertionError, match="Initialize the noise schedule before preparation"):
        cls._set_noise_schedule(network, "rf")
    assert network.noise_scheduler is scheduler and network.schedule_type == schedule_type


def test_duplicate_preparation_rejects_added_meta_tensors_without_clearing_state():
    root = _Composite()
    child = root.container["child"]
    root.prepare_module(device="cpu", precision_model=torch.bfloat16)
    root.own = torch.nn.Linear(2, 2, device="meta")

    with pytest.raises(RuntimeError, match="already prepared"):
        root.prepare_module(device=None, precision_model=torch.float32)

    assert root.is_prepared and child.is_prepared
    assert root.device == torch.device("cpu") and root.precision_model is torch.bfloat16


def _meta_wan_moe():
    """A WanMoE stub whose experts are meta-initialized, as on non-primary FSDP ranks."""
    from fastgen.networks.Wan.network import Wan
    from fastgen.networks.Wan.wan_moe import WanMoE

    def expert():
        network = object.__new__(Wan)
        torch.nn.Module.__init__(network)
        with torch.device("meta"):
            network.transformer = _DeclaredTransformer()
        network._keep_original_prec = True
        return network

    network = object.__new__(WanMoE)
    torch.nn.Module.__init__(network)
    network.network_high = expert()
    network.network_low = expert()
    return network


def test_private_set_device_reaches_wan_moe_experts():
    network = _meta_wan_moe()
    experts = (network.network_high, network.network_low)
    # Meta-initialized modules have no real tensor to fall back on before setup.
    assert network.device is None
    assert all(expert.device is None for expert in experts)

    network._set_device("cpu")

    assert network.device == torch.device("cpu")
    for expert in experts:
        assert expert._device == torch.device("cpu")
        assert expert.device == torch.device("cpu")


def test_private_set_device_reaches_composite_discriminator_modalities():
    from fastgen.networks.discriminators import CompositeDiscriminator, Discriminator

    class _MetaDiscriminator(Discriminator):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, device="meta"))

    discriminator = object.__new__(CompositeDiscriminator)
    torch.nn.Module.__init__(discriminator)
    discriminator.modality_discs = torch.nn.ModuleDict({"video": _MetaDiscriminator(), "audio": _MetaDiscriminator()})
    assert all(child.device is None for child in discriminator.modality_discs.values())

    discriminator._set_device("cpu")

    for child in discriminator.modality_discs.values():
        assert child._device == torch.device("cpu")
        assert child.device == torch.device("cpu")


def test_private_set_device_reaches_wrapped_network_of_multimodal_wrapper():
    from tests.model_utils import MultimodalNet

    wrapper = object.__new__(MultimodalNet)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner = _MixedNetwork(device="meta")
    assert wrapper.inner.device is None

    wrapper._set_device("cpu")

    assert wrapper.inner._device == torch.device("cpu")
    assert wrapper.inner.device == torch.device("cpu")


def test_multimodal_wrapper_delegates_precision_without_an_override():
    from tests.model_utils import MultimodalNet

    assert "_set_precision" not in MultimodalNet.__dict__
    wrapper = object.__new__(MultimodalNet)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner = _MixedNetwork()

    _set_precision(wrapper, precision_opt=torch.float32)

    assert wrapper.precision_model == torch.bfloat16
    assert wrapper.inner.precision_model == torch.bfloat16
    assert wrapper.inner.weight.dtype == torch.float32 and wrapper.inner.fixed.dtype == torch.float64


@pytest.mark.parametrize("trainable", [False, True])
def test_prepare_module_configures_nested_modules(trainable):
    from fastgen.networks.Wan.network import Wan
    from fastgen.networks.Wan.wan_moe import WanMoE

    def expert():
        network = object.__new__(Wan)
        torch.nn.Module.__init__(network)
        network.transformer = _DeclaredTransformer()
        network._keep_original_prec = True
        return network

    network = object.__new__(WanMoE)
    torch.nn.Module.__init__(network)
    network.network_high = expert()
    network.network_low = expert()
    network.requires_grad_(not trainable)

    network.prepare_module(
        device="cpu",
        precision_model=torch.bfloat16,
        precision_opt=torch.float64,
        trainable=trainable,
    )

    # Storage precision is explicit and independent of train/eval or gradient state.
    storage = torch.float64
    for expert in (network.network_high, network.network_low):
        assert expert._device == torch.device("cpu")
        assert expert.precision_model == torch.bfloat16
        assert expert.training is trainable
        assert all(param.requires_grad is trainable for param in expert.parameters())
        assert expert.transformer.proj.weight.dtype == storage
        assert expert.transformer.time_embedder.weight.dtype == storage
        assert expert.transformer.rope.freqs_cos.dtype == torch.float32


# ---------------------------------------------------------------------------
# ``FastGenNetwork.vae_context``: persistent versus borrowed VAEs
# ---------------------------------------------------------------------------


def _make_prepared_vae_context_network(network_type=_VAEOnlyNetwork):
    network = network_type()
    network.prepare_module(
        device="cpu",
        precision_model=torch.float64,
        precision_opt=torch.float32,
        trainable=True,
    )
    return network


def _temporary_vae_initialization_failures_worker(rank, world_size):
    import torch.distributed as dist

    for failure in ("construction", "unprepared_host", "invalid_type"):
        network = _make_prepared_vae_context_network()
        if rank == 1 and failure == "unprepared_host":
            network._is_prepared = False
        prepare_calls = []

        def init_vae():
            network.preprocessors["vae"] = _PlainPreprocessor()
            if rank == 1 and failure == "construction":
                raise FileNotFoundError("missing VAE weights")
            if rank == 1 and failure == "invalid_type":
                network.preprocessors["vae"] = torch.nn.Linear(2, 2)

        network.init_vae = init_vae
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(FastGenModule, "prepare_module", lambda *args, **kwargs: prepare_calls.append(True))
            with pytest.raises(RuntimeError, match="temporary VAE initialization: rank 1"):
                with torch.inference_mode(), network.vae_context():
                    pytest.fail("A failed temporary VAE must not be yielded")
        checks = [None] * world_size
        dist.all_gather_object(checks, not prepare_calls and "vae" not in network.preprocessors)
        assert all(checks), (failure, checks)
    return True


@RunIf(min_gpus=2)
def test_temporary_vae_initialization_failures_propagate_before_preparation():
    assert run_distributed_test(
        _temporary_vae_initialization_failures_worker, world_size=2, timeout=120, nccl_timeout=30
    )


def test_vae_context_borrows_an_unregistered_temporary_vae():
    network = _make_prepared_vae_context_network()

    with network.vae_context() as vae:
        assert network.preprocessors["vae"] is vae
        assert vae.is_prepared
        assert vae.core.weight.dtype == torch.float64
        # The temporary frozen VAE follows the model-precision request, not the
        # trainable host's optimizer/storage precision.
        assert network.proj.weight.dtype == torch.float32
        assert vae.precision_model == torch.float64
        assert vae.precision_opt == torch.float64
        assert not vae.training
        assert not any(parameter.requires_grad for parameter in vae.parameters())

    assert network.init_vae_calls == 1
    assert "vae" not in network.preprocessors


def test_vae_context_prepares_a_fresh_vae_for_each_logging_call():
    network = _make_prepared_vae_context_network()
    borrowed = []
    for _ in range(2):
        with network.vae_context() as vae:
            assert vae.is_prepared
            borrowed.append(vae)
            # Nested decoding in the same call borrows the current temporary instance.
            with network.vae_context() as nested:
                assert nested is vae
        assert "vae" not in network.preprocessors

    assert network.init_vae_calls == 2
    assert borrowed[0] is not borrowed[1]


def test_vae_context_prepares_a_temporary_vae_inside_an_inference_context():
    """VACE borrows its VAE from within inference preprocessing; the temporary VAE must still be
    built and prepared as ordinary tensors."""
    network = _make_prepared_vae_context_network()

    with torch.inference_mode(), network.vae_context() as vae:
        assert vae.is_prepared
        assert not any(tensor.is_inference() for tensor in (*vae.parameters(), *vae.buffers()))
        assert torch.is_inference_mode_enabled()

    assert "vae" not in network.preprocessors


def test_vae_context_reuses_a_persistent_vae():
    network = _make_prepared_vae_context_network()
    network.init_vae()
    vae = network.preprocessors["vae"]
    vae.prepare_module(device=network.device, precision_model=network._precision_model)

    with network.vae_context() as yielded:
        assert yielded is vae

    assert network.init_vae_calls == 1
    assert network.preprocessors["vae"] is vae


def test_vae_context_requires_a_prepared_host_before_temporary_initialization():
    network = _VAEOnlyNetwork()

    with pytest.raises(RuntimeError, match="host network to be prepared"):
        with network.vae_context():
            pytest.fail("the context must not yield from an unprepared host")

    assert network.init_vae_calls == 0
    assert "vae" not in network.preprocessors


def test_vae_context_cleans_up_a_partial_initialization():
    class _FailingVAEOnlyNetwork(_VAEOnlyNetwork):
        def init_vae(self):
            super().init_vae()
            raise RuntimeError("VAE initialization failed")

    network = _make_prepared_vae_context_network(_FailingVAEOnlyNetwork)

    with pytest.raises(RuntimeError, match="VAE initialization failed"):
        with network.vae_context():
            pytest.fail("the context must not yield after failed VAE initialization")

    assert "vae" not in network.preprocessors


def test_vae_context_cleans_up_after_prepare_failure(monkeypatch):
    network = _make_prepared_vae_context_network()

    def fail_prepare(_self, *_args, **_kwargs):
        raise RuntimeError("VAE preparation failed")

    monkeypatch.setattr(_PlainPreprocessor, "prepare_module", fail_prepare)

    with pytest.raises(RuntimeError, match="VAE preparation failed"):
        with network.vae_context():
            pytest.fail("the context must not yield after failed VAE preparation")

    assert "vae" not in network.preprocessors


def test_vae_context_preserves_a_replacement_on_exception():
    network = _make_prepared_vae_context_network()
    replacement = _PlainPreprocessor()

    with pytest.raises(RuntimeError, match="decode failed"):
        with network.vae_context():
            network.preprocessors["vae"] = replacement
            raise RuntimeError("decode failed")

    assert network.preprocessors["vae"] is replacement
