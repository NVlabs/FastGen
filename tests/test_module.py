# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

import fastgen.utils.module as module_state
from tests.utils import RunIf


@pytest.mark.parametrize("unexpected", [False, True], ids=["empty", "unrelated"])
@pytest.mark.parametrize("check_state_preservation", [False, True])
def test_require_any_rejects_all_missing_checkpoint_wrapper_state(unexpected, check_state_preservation):
    inner = torch.nn.Linear(2, 2)
    inner.register_buffer("scale", torch.ones(2))
    inner.register_buffer("cache", torch.zeros(2), persistent=False)
    module = checkpoint_wrapper(inner)
    state_dict = {"unrelated": torch.ones(1)} if unexpected else {}
    before = {name: tensor.clone() for name, tensor in inner.state_dict().items()}

    # Export hooks strip the wrapper prefix, but native missing-key reports retain it.
    target_names = set(module_state.module_state_names(module, nonpersistent_buffers=False))
    assert set(module.state_dict()).isdisjoint(target_names)
    assert set(module.load_state_dict(state_dict, strict=False).missing_keys) == target_names
    with pytest.raises(RuntimeError, match="matched no target parameters"):
        module_state.load_module_state_dict(
            module, state_dict, strict=False, require_any=True, check_state_preservation=check_state_preservation
        )

    assert all(torch.equal(inner.state_dict()[name], tensor) for name, tensor in before.items())


@pytest.mark.parametrize(
    "loaded_names",
    [("weight",), ("weight", "bias", "scale")],
    ids=["partial-parameter", "full"],
)
def test_require_any_accepts_matching_checkpoint_wrapper_state(loaded_names):
    inner = torch.nn.Linear(2, 2)
    inner.register_buffer("scale", torch.ones(2))
    inner.register_buffer("cache", torch.zeros(2), persistent=False)
    module = checkpoint_wrapper(inner)
    original = {name: tensor.clone() for name, tensor in inner.state_dict().items()}
    state_dict = {name: torch.full_like(original[name], 7.0) for name in loaded_names}

    load_info = module_state.load_module_state_dict(module, state_dict, strict=False, require_any=True)

    assert set(load_info.missing_keys) == {
        f"_checkpoint_wrapped_module.{name}" for name in original.keys() - state_dict.keys()
    }
    assert not load_info.unexpected_keys
    for name, tensor in inner.state_dict().items():
        assert torch.equal(tensor, state_dict.get(name, original[name]))
    assert torch.equal(inner.cache, torch.zeros(2))


@pytest.mark.parametrize("source", ["buffer", "synthesized", "parameter_free"])
def test_require_any_rejects_buffer_only_matches(source):
    # Without affine parameters BatchNorm holds only buffers; ``strict`` is the check for such modules.
    module = torch.nn.BatchNorm1d(2, affine=source != "parameter_free")
    state_dict = {"unrelated": torch.ones(1)} if source == "synthesized" else {"running_mean": torch.ones(2)}
    # A plain dict has no version metadata, so BatchNorm synthesizes ``num_batches_tracked`` itself.
    assert "num_batches_tracked" not in module.load_state_dict(dict(state_dict), strict=False).missing_keys

    with pytest.raises(RuntimeError, match="matched no target parameters"):
        module_state.load_module_state_dict(module, state_dict, strict=False, require_any=True)


def test_require_any_accepts_exact_strict_loads_of_parameter_free_modules():
    source = torch.nn.BatchNorm1d(2, affine=False)
    source.running_mean.fill_(3.0)
    target = torch.nn.BatchNorm1d(2, affine=False)

    module_state.load_module_state_dict(target, source.state_dict(), strict=True, require_any=True)

    assert torch.equal(target.running_mean, source.running_mean)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("check_state_preservation", [False, True])
def test_empty_required_keys_are_rejected_before_loading(strict, check_state_preservation):
    module = torch.nn.Linear(2, 2)
    before = module.weight.detach().clone()

    with pytest.raises(ValueError, match="required_keys must name at least one target key"):
        module_state.load_module_state_dict(
            module,
            {"weight": torch.zeros(2, 2)},
            strict=strict,
            required_keys=(),
            check_state_preservation=check_state_preservation,
        )

    assert torch.equal(module.weight, before)


def test_require_any_counts_a_loaded_alias_inside_a_checkpoint_wrapper():
    inner = torch.nn.Module()
    inner.weight = torch.nn.Parameter(torch.zeros(2))
    inner.weight_alias = inner.weight
    module = checkpoint_wrapper(inner)

    load_info = module_state.load_module_state_dict(
        module, {"weight_alias": torch.full((2,), 7.0)}, strict=False, require_any=True
    )

    assert load_info.missing_keys == ["_checkpoint_wrapped_module.weight"]
    assert not load_info.unexpected_keys
    assert inner.weight is inner.weight_alias
    assert torch.equal(inner.weight, torch.full((2,), 7.0))


def test_wrapped_required_keys_load_requires_the_rest_completely_or_not_at_all(monkeypatch):
    inner = torch.nn.Module()
    for name in ("adapter", "base_a", "base_b"):
        inner.register_parameter(name, torch.nn.Parameter(torch.zeros(2)))
    module = torch.nn.Module()
    module.block = checkpoint_wrapper(inner)
    wrapped = "block._checkpoint_wrapped_module"
    warnings, successes = [], []
    monkeypatch.setattr(module_state.logger, "warning", warnings.append)
    monkeypatch.setattr(module_state.logger, "success", successes.append)

    def load(names, value, **kwargs):
        state_dict = {f"block.{name}": torch.full((2,), value) for name in names}
        # Target names keep the wrapper prefix, as the loader reports them.
        return module_state.load_module_state_dict(
            module, state_dict, required_keys={f"{wrapped}.adapter"}, description="wrapped", **kwargs
        )

    # The rest may be absent (as without a frozen base) or complete (as in a full checkpoint).
    load_info = load(["adapter"], 1.0)
    assert load_info.missing_keys == [f"{wrapped}.base_a", f"{wrapped}.base_b"]
    assert torch.equal(inner.base_a, torch.zeros(2))
    load(["adapter", "base_a", "base_b"], 2.0)
    assert torch.equal(inner.base_b, torch.full((2,), 2.0))
    assert successes == [
        "wrapped, loading info: missing_keys=[], unexpected_keys=[]",
        "wrapped, loading info: missing_keys=[], unexpected_keys=[], 2 of 2 non-required tensors loaded "
        f"(first 8: ['{wrapped}.base_a', '{wrapped}.base_b'])",
    ]

    # Part of the rest, a missing required tensor, or a key the target lacks fails strictly ...
    with pytest.raises(
        RuntimeError, match=rf"1 of 2 non-required .*, 1 not loaded \(first 8: \['{wrapped}.base_b'\]\)"
    ):
        load(["adapter", "base_a"], 3.0)
    with pytest.raises(RuntimeError, match=rf"missing_keys=\['{wrapped}.adapter'\], unexpected_keys=\['{wrapped}.x'\]"):
        load(["x", "base_a", "base_b"], 3.0)

    # ... and is reported when permissive.
    load(["adapter", "base_a"], 4.0, strict=False)
    assert torch.equal(inner.base_a, torch.full((2,), 4.0))
    assert warnings == [
        f"wrapped, loading info: missing_keys=[], unexpected_keys=[], 1 of 2 non-required tensors loaded "
        f"(first 8: ['{wrapped}.base_a']), 1 not loaded (first 8: ['{wrapped}.base_b'])"
    ]


def test_required_keys_load_judges_keys_after_load_hooks():
    module = torch.nn.Module()
    module.base = torch.nn.Parameter(torch.zeros(2))
    module.adapter = torch.nn.Parameter(torch.zeros(2))
    renames = {"legacy_adapter": "adapter"}

    def remap(state_dict, prefix, *args):
        for old, new in list(renames.items()):
            if prefix + old in state_dict:
                value = state_dict.pop(prefix + old)
                if new is not None:
                    state_dict[prefix + new] = value

    module._register_load_state_dict_pre_hook(remap)

    # A key that a hook renames counts as loaded ...
    module_state.load_module_state_dict(module, {"legacy_adapter": torch.ones(2)}, required_keys={"adapter"})
    assert torch.equal(module.adapter, torch.ones(2))

    # ... and a required key that a hook drops is missing, although the state dict held it.
    renames["adapter"] = None
    with pytest.raises(RuntimeError, match=r"missing_keys=\['adapter'\]"):
        module_state.load_module_state_dict(
            module, {"adapter": torch.full((2,), 2.0), "base": torch.ones(2)}, required_keys={"adapter"}
        )


def test_required_keys_load_ignores_buffers_that_the_loader_synthesizes():
    module = torch.nn.Module()
    module.adapter = torch.nn.Parameter(torch.zeros(2))
    module.norm = torch.nn.BatchNorm1d(2)

    # Without version metadata BatchNorm inserts its num_batches_tracked, which must not make a state dict
    # without the rest look like one that holds part of it. Required keys may come from a one-shot iterable.
    module_state.load_module_state_dict(module, {"adapter": torch.ones(2)}, required_keys=iter(["adapter"]))
    assert torch.equal(module.adapter, torch.ones(2))
    with pytest.raises(RuntimeError, match=r"missing_keys=\['adapter'\]"):
        module_state.load_module_state_dict(module, {"norm.weight": torch.ones(2)}, required_keys=iter(["adapter"]))


def test_temporarily_disable_activation_checkpointing_restores_flags_on_error():
    module = torch.nn.Sequential(torch.nn.Linear(2, 2))
    module.gradient_checkpointing = True
    module[0].gradient_checkpointing = False

    with pytest.raises(RuntimeError, match="forward failed"):
        with module_state.temporarily_disable_activation_checkpointing(module):
            assert module.gradient_checkpointing is False
            assert module[0].gradient_checkpointing is False
            raise RuntimeError("forward failed")

    assert module.gradient_checkpointing is True
    assert module[0].gradient_checkpointing is False


def test_temporarily_disable_activation_checkpointing_rejects_wrappers_only_with_grad():
    module = checkpoint_wrapper(torch.nn.Linear(2, 2))

    with pytest.raises(RuntimeError, match="activation-checkpoint wrappers"):
        with module_state.temporarily_disable_activation_checkpointing(module):
            pass

    with torch.no_grad(), module_state.temporarily_disable_activation_checkpointing(module):
        pass


def test_tensor_registrations_capture_raw_slots_and_current_values():
    root = torch.nn.Module()
    root.child = torch.nn.Module()
    weight = torch.nn.Parameter(torch.ones(2))
    root.child.register_parameter("weight", weight)
    root.child.register_parameter("optional_weight", None)
    root.child.register_buffer("cache", torch.ones(2), persistent=False)
    root.child.register_buffer("optional_cache", None)

    registrations = module_state.tensor_registrations(root)

    registration = registrations[("parameter", "child.weight")]
    assert isinstance(registration, module_state.TensorRegistration)
    assert registration.owner is root.child
    assert registration.local_name == "weight"
    assert registration.tensor is weight
    assert registration.persistent is None

    owner, local_name, captured_weight, persistent = registration
    assert (owner, local_name, persistent) == (root.child, "weight", None)
    assert captured_weight is weight
    assert registrations[("buffer", "child.cache")][3] is False
    assert registrations[("parameter", "child.optional_weight")][2:] == (None, None)
    assert registrations[("buffer", "child.optional_cache")][2:] == (None, True)

    replacement = torch.nn.Parameter(torch.zeros(2))
    root.child.weight = replacement
    assert registrations[("parameter", "child.weight")][2] is weight
    assert root.child._parameters["weight"] is replacement


def test_module_state_tensors_selects_aliases_before_deduplicating():
    module = torch.nn.Module()
    weight = torch.nn.Parameter(torch.ones(2))
    module.register_parameter("z_weight", weight)
    module.register_parameter("a_weight", weight)
    module.register_parameter("optional_weight", None)

    first = torch.ones(2)
    module.register_buffer("z_persistent", first)
    module.register_buffer("a_nonpersistent", first, persistent=False)
    second = torch.zeros(2)
    module.register_buffer("z_nonpersistent", second, persistent=False)
    module.register_buffer("a_persistent", second)
    module.register_buffer("parameter_buffer", weight)
    module.register_buffer("optional_buffer", None, persistent=False)

    def entries(**kwargs):
        return [(kind, name, id(tensor)) for kind, name, tensor in module_state.module_state_tensors(module, **kwargs)]

    assert entries(parameters=False, persistent_buffers=True, nonpersistent_buffers=False) == [
        ("buffer", "a_persistent", id(second)),
        ("buffer", "parameter_buffer", id(weight)),
        ("buffer", "z_persistent", id(first)),
    ]
    assert entries(parameters=False, persistent_buffers=False, nonpersistent_buffers=True) == [
        ("buffer", "a_nonpersistent", id(first)),
        ("buffer", "z_nonpersistent", id(second)),
    ]
    assert entries() == [
        ("parameter", "a_weight", id(weight)),
        ("buffer", "a_nonpersistent", id(first)),
        ("buffer", "a_persistent", id(second)),
        ("buffer", "parameter_buffer", id(weight)),
    ]
    assert entries(remove_duplicate=False) == [
        ("parameter", "a_weight", id(weight)),
        ("parameter", "z_weight", id(weight)),
        ("buffer", "a_nonpersistent", id(first)),
        ("buffer", "a_persistent", id(second)),
        ("buffer", "parameter_buffer", id(weight)),
        ("buffer", "z_nonpersistent", id(second)),
        ("buffer", "z_persistent", id(first)),
    ]
    assert entries(names={"z_weight", "a_persistent", "z_nonpersistent"}) == [
        ("parameter", "z_weight", id(weight)),
        ("buffer", "a_persistent", id(second)),
    ]


def test_module_state_tensors_order_does_not_depend_on_registration_order():
    def make_module(names):
        module = torch.nn.Module()
        for name in names:
            module.register_buffer(name, torch.ones(1))
        return module

    forward = make_module(("a", "b"))
    reverse = make_module(("b", "a"))

    assert [name for _kind, name, _tensor in module_state.module_state_tensors(forward)] == ["a", "b"]
    assert [name for _kind, name, _tensor in module_state.module_state_tensors(reverse)] == ["a", "b"]


def test_module_state_tensors_deduplicates_shared_module_paths():
    child = torch.nn.Module()
    child.register_buffer("cache", torch.ones(1))
    root = torch.nn.Module()
    root.right = child
    root.left = child

    names = [name for _kind, name, _tensor in module_state.module_state_tensors(root)]
    all_names = [name for _kind, name, _tensor in module_state.module_state_tensors(root, remove_duplicate=False)]

    assert names == ["left.cache"]
    assert all_names == ["left.cache", "right.cache"]


def test_module_state_tensors_excludes_every_name_of_state_held_by_excluded_submodules():
    root = torch.nn.Module()
    root.child = torch.nn.Module()
    root.child.weight = torch.nn.Parameter(torch.ones(2))
    cache = torch.ones(2)
    root.child.register_buffer("cache", cache)
    root.register_buffer("cache", cache)
    root.register_buffer("own", torch.zeros(2))

    assert list(module_state.module_state_names(root)) == ["child.weight", "cache", "child.cache", "own"]
    # The shared cache belongs to the excluded child under both of its names; exclusions may be one-shot.
    assert list(module_state.module_state_names(root, exclude=iter([root.child]))) == ["own"]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_validate_state_dict_export_checks_tensor_coverage_by_identity(wrapped, device, monkeypatch):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2, device=device))
    module.weight_alias = module.weight
    module.hidden = torch.nn.Parameter(torch.zeros(2, device=device))
    module.register_buffer("hidden_scale", torch.ones(2, device=device))
    module.register_buffer("cache", torch.zeros(2, device=device), persistent=False)
    target = checkpoint_wrapper(module) if wrapped else module
    state_with_vars = target.state_dict(keep_vars=True)
    exported = target.state_dict()

    def fail_export(*args, **kwargs):
        raise AssertionError("Validation must not run export hooks")

    with monkeypatch.context() as patch:
        patch.setattr(target, "state_dict", fail_export)
        module_state.validate_state_dict_export(target, state_with_vars)
        module_state.validate_state_dict_export(target, state_with_vars, exported)

    def rename(_module, state_dict, prefix, _local_metadata):
        del state_dict[prefix + "weight_alias"]
        state_dict[prefix + "renamed"] = state_dict.pop(prefix + "weight")

    module._register_state_dict_hook(rename)
    # Any exported alias/renamed key covers the same tensor; nonpersistent state need not export.
    state_with_vars = target.state_dict(keep_vars=True)
    module_state.validate_state_dict_export(target, state_with_vars)
    exported = target.state_dict()
    module_state.validate_state_dict_export(target, state_with_vars, exported)
    del exported["renamed"]
    with pytest.raises(RuntimeError, match="does not export 2 registered tensor"):
        module_state.validate_state_dict_export(target, state_with_vars, exported)

    def hide(_module, state_dict, prefix, _local_metadata):
        del state_dict[prefix + "hidden"]
        del state_dict[prefix + "hidden_scale"]

    module._register_state_dict_hook(hide)
    state_with_vars = target.state_dict(keep_vars=True)
    with pytest.raises(RuntimeError, match="does not export 2 registered tensor"):
        module_state.validate_state_dict_export(target, state_with_vars)


def test_tied_parameter_registration_groups_returns_physical_registration_groups():
    root = torch.nn.Module()
    root.direct = torch.nn.Parameter(torch.ones(2))
    root.register_parameter("direct_alias", root.direct)
    root.left = torch.nn.Linear(2, 2, bias=False)
    root.right = torch.nn.Linear(2, 2, bias=False)
    root.right.weight = root.left.weight

    groups = {
        frozenset((id(registration.owner), registration.local_name) for registration in group)
        for group in module_state.tied_parameter_registration_groups(module_state.tensor_registrations(root))
    }

    assert groups == {
        frozenset(((id(root), "direct"), (id(root), "direct_alias"))),
        frozenset(((id(root.left), "weight"), (id(root.right), "weight"))),
    }


def test_tied_parameter_registration_groups_ignores_buffers():
    root = torch.nn.Module()
    cache = torch.ones(2)
    root.register_buffer("z_cache", cache)
    root.register_buffer("a_cache", cache)

    assert module_state.tied_parameter_registration_groups(module_state.tensor_registrations(root)) == ()


@pytest.mark.parametrize("allow_buffer_aliases", [False, True])
def test_reject_unsupported_state_aliases_rejects_parameter_buffer_aliases(allow_buffer_aliases):
    root = torch.nn.Module()
    tensor = torch.nn.Parameter(torch.ones(2))
    root.register_parameter("weight", tensor)
    root.register_buffer("cache", tensor)

    registrations = module_state.tensor_registrations(root)
    assert module_state.tied_parameter_registration_groups(registrations) == ()
    with pytest.raises(AssertionError, match="both a parameter and a buffer"):
        module_state.reject_unsupported_state_aliases(root, allow_buffer_aliases=allow_buffer_aliases)


@pytest.mark.parametrize("different_owners", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_reject_unsupported_state_aliases_rejects_distinct_buffer_slots(different_owners, dtype):
    root = torch.nn.Module()
    root.child = torch.nn.Module()
    cache = torch.ones(2, dtype=dtype)
    root.register_buffer("cache", cache)
    owner = root.child if different_owners else root
    owner.register_buffer("alias", cache, persistent=False)

    with pytest.raises(AssertionError, match="multiple distinct buffer slots"):
        module_state.reject_unsupported_state_aliases(root)
    module_state.reject_unsupported_state_aliases(root, allow_buffer_aliases=True)
    assert root.cache is owner.alias is cache


def test_reject_unsupported_state_aliases_accepts_tied_parameters_and_shared_submodules():
    root = torch.nn.Module()
    root.left = torch.nn.Linear(2, 2, bias=False)
    root.right = torch.nn.Linear(2, 2, bias=False)
    root.right.weight = root.left.weight
    root.left.register_buffer("cache", torch.ones(2))
    root.right.register_buffer("cache", torch.ones(2))
    root.right.register_buffer("optional_cache", None)
    root.left_alias = root.left
    root.nested = torch.nn.Module()
    root.nested.right_alias = root.right

    module_state.reject_unsupported_state_aliases(root)


def test_tied_parameter_registration_groups_does_not_count_multiple_paths_to_one_module():
    shared = torch.nn.Linear(2, 2)
    root = torch.nn.Module()
    root.left = shared
    root.right = shared

    weight_registrations = [
        (key, registration)
        for key, registration in module_state.tensor_registrations(root).items()
        if registration[1] == "weight"
    ]
    assert {qualified_name for (_kind, qualified_name), _registration in weight_registrations} == {
        "left.weight",
        "right.weight",
    }
    assert (
        len(
            {
                (id(owner), kind, local_name)
                for (kind, _qualified_name), (owner, local_name, _tensor, _persistent) in weight_registrations
            }
        )
        == 1
    )
    assert module_state.tied_parameter_registration_groups(module_state.tensor_registrations(root)) == ()


def test_reject_inference_state_names_tensors_and_refuses_the_context():
    module = torch.nn.Linear(2, 2)
    module_state.reject_inference_state(module)

    with torch.inference_mode():
        module.register_buffer("cache", torch.ones(2), persistent=False)
        with pytest.raises(RuntimeError, match="must not run under torch.inference_mode"):
            module_state.reject_inference_state(module)
    with pytest.raises(RuntimeError, match=r"1 inference tensor\(s\), first 8: \['cache'\]"):
        module_state.reject_inference_state(module)

    # Low-level materialization is reachable without prepare_module and refuses the same states.
    with torch.device("meta"):
        meta = torch.nn.Linear(2, 2)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="meta materialization of Linear must not run"):
        module_state.materialize_meta_module(meta, "cpu")
    assert all(parameter.is_meta for parameter in meta.parameters())


class _DeclaredTransformer(torch.nn.Module):
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table"]

    def __init__(self):
        super().__init__()
        self.time_embedder = torch.nn.Linear(2, 2)
        self.proj = torch.nn.Linear(2, 2)
        self.scale_shift_table = torch.nn.Parameter(torch.ones(2))


def _cast_storage(module, default_dtype, dtype_overrides, exclude=()):
    entries = module_state.resolve_storage_dtypes(module, default_dtype, dtype_overrides, exclude)
    module_state.cast_to_storage_dtypes(module, entries)


def test_storage_cast_applies_exact_overrides():
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))

    overrides = {name: torch.bfloat16 for name in module_state.module_state_names(module) if name.startswith("1.")}
    _cast_storage(module, default_dtype=None, dtype_overrides=overrides)

    assert module[0].weight.dtype == torch.float32 and module[1].weight.dtype == torch.bfloat16
    weights = module[0].weight, module[1].weight
    _cast_storage(module, default_dtype=None, dtype_overrides={})
    assert module[0].weight is weights[0] and module[1].weight is weights[1]
    assert module[0].weight.dtype == torch.float32 and module[1].weight.dtype == torch.bfloat16


def test_storage_cast_preserves_tied_parameters_and_shared_owners():
    root = torch.nn.Module()
    root.left = torch.nn.Module()
    root.right = root.left
    root.left.register_buffer("cache", torch.ones(2), persistent=False)
    weight = torch.nn.Parameter(torch.ones(2))
    weight.grad = torch.ones_like(weight)
    root.register_parameter("weight", weight)
    root.left.register_parameter("weight", weight)

    _cast_storage(root, default_dtype=torch.bfloat16, dtype_overrides={})

    assert root.left.cache is root.right.cache and root.left.cache.dtype == torch.bfloat16
    assert root.weight is root.left.weight is weight and weight.dtype == torch.bfloat16
    assert weight.grad.dtype == torch.bfloat16
    assert "cache" in root.right._non_persistent_buffers_set


def test_storage_cast_rejects_casting_wrapper_subclass_buffers():
    class Wrapper(torch.Tensor):
        @staticmethod
        def __new__(cls, inner):
            return torch.Tensor._make_wrapper_subclass(cls, inner.shape, dtype=inner.dtype, device=inner.device)

        def __init__(self, inner):
            self.inner = inner

        def __tensor_flatten__(self):
            return ["inner"], None

        @staticmethod
        def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
            return Wrapper(inner_tensors["inner"])

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            raise NotImplementedError("the rejection must happen before any operator runs")

    module = torch.nn.Module()
    module.register_buffer("ordinary", torch.ones(2))
    module.register_buffer("wrapped", Wrapper(torch.ones(2)))

    # No dtype change: nothing to cast, so the wrapper is left alone.
    _cast_storage(module, default_dtype=torch.float32, dtype_overrides={})
    with pytest.raises(AssertionError, match="Cast Wrapper storage before wrapping"):
        _cast_storage(module, default_dtype=torch.bfloat16, dtype_overrides={})


def test_storage_cast_keeps_registrations_shared_with_excluded_modules_in_sync():
    parent = torch.nn.Module()
    parent.child = torch.nn.Module()
    shared = torch.nn.Parameter(torch.ones(2))
    parent.register_parameter("weight", shared)
    parent.child.register_parameter("weight", shared)

    # The parent leaves state owned by the nested module alone ...
    _cast_storage(parent, torch.bfloat16, {}, exclude=[parent.child])
    assert parent.weight.dtype == torch.float32
    # ... and the nested cast converts the one tied parameter, so the parent's registration follows.
    _cast_storage(parent.child, torch.bfloat16, {})
    assert parent.weight is parent.child.weight is shared
    assert shared.dtype == torch.bfloat16


def test_storage_cast_rejects_overrides_for_state_held_by_excluded_modules():
    parent = torch.nn.Module()
    parent.child = torch.nn.Module()
    parent.child.register_buffer("cache", torch.ones(2))

    with pytest.raises(AssertionError, match="Storage dtype overrides.*nested module state.*child.cache"):
        _cast_storage(parent, torch.bfloat16, {"child.cache": torch.float32}, exclude=[parent.child])
    assert parent.child.cache.dtype == torch.float32


def test_storage_cast_rejects_unknown_overrides_before_casting():
    module = torch.nn.Linear(2, 2)
    with pytest.raises(AssertionError, match="Storage dtype overrides.*wieght"):
        _cast_storage(module, torch.bfloat16, {"wieght": torch.float32})
    assert module.weight.dtype == module.bias.dtype == torch.float32


def test_storage_cast_accepts_integer_override_names_without_casting_them():
    module = torch.nn.Linear(2, 2)
    module.register_buffer("counter", torch.tensor(3))
    _cast_storage(module, torch.bfloat16, {"counter": torch.float32})
    assert module.weight.dtype == torch.bfloat16
    assert module.counter.dtype == torch.int64 and module.counter.item() == 3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_storage_cast_keeps_complex_tensors(dtype):
    module = torch.nn.Linear(2, 2)
    rotation = torch.polar(torch.ones(4), torch.arange(4.0))
    module.register_buffer("rotation", rotation.clone())
    module.phase = torch.nn.Parameter(rotation.clone())

    # A real storage dtype would discard the imaginary parts, even when an override names the tensor.
    _cast_storage(module, dtype, {"rotation": torch.float32})

    assert module.weight.dtype == dtype
    assert module.rotation.dtype == module.phase.dtype == torch.complex64
    assert torch.equal(module.rotation, rotation) and torch.equal(module.phase, rotation)


def test_matches_fp32_declaration_uses_name_components_below_the_prefix():
    transformer = _DeclaredTransformer()
    assert module_state.matches_fp32_declaration("transformer.time_embedder.weight", transformer, prefix="transformer.")
    assert module_state.matches_fp32_declaration("transformer.scale_shift_table", transformer, prefix="transformer.")
    assert not module_state.matches_fp32_declaration("transformer.proj.weight", transformer, prefix="transformer.")
    assert not module_state.matches_fp32_declaration("other.time_embedder.weight", transformer, prefix="transformer.")
    assert not module_state.matches_fp32_declaration("time_embedder.weight", torch.nn.Module(), prefix="")


def test_materialize_meta_module_resets_root_and_repairs_nested_ties():
    events = []

    class Leaf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2))
            self.register_buffer("cache", torch.empty(2), persistent=False)

    class TiedOwner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(3, 2)
            self.projection = torch.nn.Linear(2, 3, bias=False)
            self.tie_weights()

        def tie_weights(self):
            events.append("tie")
            self.projection.weight = self.embedding.weight

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.leaf = Leaf()
            self.tied = TiedOwner()

        def reset_parameters(self):
            # The architecture root owns the reset contract for its descendants' non-persistent state.
            assert self.tied.projection.weight is self.tied.embedding.weight
            events.append("root_reset")
            torch.nn.init.constant_(self.leaf.weight, 3.0)
            self.leaf.cache.fill_(5.0)

    with torch.device("meta"):
        module = Root()
    # Method-specific state may be attached after leaving the meta init context; materialization
    # must move it with its values instead of discarding it.
    module.register_buffer("method_state", torch.full((2,), 7.0))
    events.clear()

    module_state.materialize_meta_module(module, "cpu")

    assert events == ["root_reset"]
    assert torch.equal(module.leaf.weight, torch.full((2,), 3.0))
    assert torch.equal(module.leaf.cache, torch.full((2,), 5.0))
    assert torch.equal(module.method_state, torch.full((2,), 7.0))
    assert module.tied.projection.weight is module.tied.embedding.weight
    assert not any(tensor.device.type == "meta" for tensor in module.state_dict().values())


def test_materialize_meta_module_restores_ties_without_calling_tie_hooks():
    hook_calls = []

    class TiedChild(torch.nn.Module):
        def tie_weights(self):
            hook_calls.append(id(self))

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = TiedChild()
            self.first_alias = self.first
            self.second = TiedChild()

            shared_parameter = torch.nn.Parameter(torch.empty(2))
            self.first.register_parameter("weight", shared_parameter)
            self.first.register_parameter("weight_alias", shared_parameter)
            self.second.register_parameter("weight", shared_parameter)

            self.first.register_buffer("cache", torch.empty(2))
            self.second.register_buffer("cache", torch.empty(2))

        def reset_parameters(self):
            pass

    with torch.device("meta"):
        module = Root()

    module_state.materialize_meta_module(module, "cpu")

    assert hook_calls == []
    assert module.first.weight is module.first.weight_alias is module.second.weight
    assert module.first.cache is module.first_alias.cache
    assert module.first.cache is not module.second.cache


def test_materialize_meta_module_restores_shared_descendant_ties_without_hooks():
    calls = []

    class SharedChild(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2))
            self.register_buffer("cache", torch.empty(2), persistent=False)

        def tie_weights(self):
            calls.append("child")
            self.weight = torch.nn.Parameter(torch.full_like(self.weight, 7.0))

    class Parent(torch.nn.Module):
        def __init__(self, name, child):
            super().__init__()
            self.name = name
            self.child = child
            self.weight = child.weight

        def tie_weights(self):
            calls.append(self.name)
            self.weight = self.child.weight

    with torch.device("meta"):
        root = torch.nn.Module()
        child = SharedChild()
        root.left = Parent("left", child)
        root.right = Parent("right", child)

    module_state.materialize_meta_module(root, "cpu")

    assert calls == []
    assert root.left.child is root.right.child
    assert root.left.child.cache is root.right.child.cache
    assert root.left.weight is root.right.weight is child.weight
    assert child.weight.device.type == "cpu"


def test_materialize_meta_module_preserves_non_meta_parameter_values():
    with torch.device("meta"):
        module = torch.nn.Linear(2, 2)
    shared = torch.nn.Linear(2, 2, bias=False)
    module.left = shared
    module.right = shared
    aliased_weight = torch.nn.Parameter(torch.ones(2))
    module.register_parameter("z_weight", aliased_weight)
    module.register_parameter("a_weight", aliased_weight)
    module.register_buffer("method_state", torch.ones(2))
    module.register_buffer("other_state", torch.ones(2))
    module.register_buffer("optional_state", None)
    original_weight = shared.weight.detach().clone()

    module_state.materialize_meta_module(module, "cpu")

    torch.testing.assert_close(shared.weight, original_weight)
    assert module.left is module.right
    assert module.z_weight is module.a_weight is aliased_weight
    torch.testing.assert_close(module.z_weight, torch.ones(2))


def test_materialize_meta_module_preserves_tied_parameters_and_shared_submodules(monkeypatch):
    class SharedOwner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(2, 2, bias=False)
            self.right = torch.nn.Linear(2, 2, bias=False)
            self.right.weight = self.left.weight
            self.left_alias = self.left
            self.left.register_buffer("cache", torch.full((2,), 4.0))

    with torch.device("meta"):
        module = SharedOwner()
    warnings = []
    monkeypatch.setattr(module_state.logger, "warning", warnings.append)

    module_state.materialize_meta_module(module, "cpu")

    assert module.left.weight is module.right.weight
    assert module.left.cache is module.left_alias.cache
    assert module.left.weight.device.type == "cpu"
    assert not any("replaced" in warning for warning in warnings)


def test_materialize_meta_module_rejects_buffer_aliases_before_allocation_and_reset():
    resets = []

    class SharedBuffers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            cache = torch.empty(2)
            self.register_buffer("z_cache", cache)
            self.register_buffer("a_cache", cache)

        def reset_parameters(self):
            resets.append(True)
            self.z_cache = torch.full_like(self.z_cache, 3.0)
            self.a_cache = torch.full_like(self.a_cache, 7.0)

    with torch.device("meta"):
        module = SharedBuffers()

    original = module.z_cache
    with pytest.raises(AssertionError, match="multiple distinct buffer slots"):
        module_state.materialize_meta_module(module, "cpu")

    assert module.z_cache is module.a_cache is original
    assert original.is_meta
    assert not resets


@pytest.mark.parametrize("mutation", ["tensor", "owner", "dtype", "persistence", "none_slot"])
def test_module_state_snapshot_reuses_its_baseline(mutation):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2))
    module.weight_alias = module.weight
    module.block = torch.nn.Identity()
    module.register_buffer("cache", torch.ones(2))
    module.register_buffer("optional", None)
    snapshot = module_state.ModuleStateSnapshot.capture(module)

    # Repeated value-only operations must pass without refreshing the structural baseline.
    with torch.no_grad():
        module.weight.fill_(2)
    snapshot.validate()
    with snapshot.validate_after("second copy"):
        module.cache.fill_(3)

    if mutation == "tensor":
        module.weight_alias = torch.nn.Parameter(module.weight.detach().clone())
    elif mutation == "owner":
        module.block = torch.nn.Identity()
    elif mutation == "dtype":
        module.weight.data = module.weight.data.double()
    elif mutation == "persistence":
        module._non_persistent_buffers_set.add("cache")
    else:
        module.optional = torch.ones(2)

    # Even a failed check must not accept the changed state as a new baseline.
    for operation in ("third copy", "fourth copy"):
        with pytest.raises(RuntimeError, match=f"{operation} changed registered state"):
            with snapshot.validate_after(operation):
                pass


def test_load_module_state_dict_copies_tied_state_preserving_identity_and_dtype():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    module.weight_alias = module.weight
    module.register_buffer("cache", torch.ones(2, dtype=torch.float64))
    module.register_buffer("cache_alias", module.cache)
    weight, cache = module.weight, module.cache
    state = {name: torch.zeros(2, dtype=torch.float32) for name in module.state_dict()}

    module_state.load_module_state_dict(module, state)

    assert module.weight is module.weight_alias is weight
    assert module.cache is module.cache_alias is cache
    torch.testing.assert_close(module.weight, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(module.cache, torch.zeros(2, dtype=torch.float64))


def test_load_module_state_dict_rejects_assignment_without_splitting_parameter_ties():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2))
    module.weight_alias = module.weight
    weight = module.weight

    with pytest.raises(TypeError, match="unexpected keyword argument 'assign'"):
        module_state.load_module_state_dict(
            module, {"weight": torch.zeros(2), "weight_alias": torch.zeros(2)}, assign=True
        )

    assert module.weight is module.weight_alias is weight
    torch.testing.assert_close(weight, torch.ones(2))


@pytest.mark.parametrize("overwrite,swap", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("operation", ["prepare", "materialize", "load"])
def test_state_operations_reject_nondefault_torch_settings_before_mutation(overwrite, swap, operation):
    from fastgen.networks.module import FastGenModule

    module = torch.nn.Linear(2, 2, dtype=torch.float64, device="meta" if operation == "materialize" else "cpu")
    module.weight_alias = module.weight
    module.register_buffer("cache", torch.ones(2, dtype=torch.float64, device=module.weight.device))
    before = module_state.tensor_registrations(module)
    values = {key: entry.tensor.clone() for key, entry in before.items()}
    original = (
        torch.__future__.get_overwrite_module_params_on_conversion(),
        torch.__future__.get_swap_module_params_on_conversion(),
    )
    try:
        torch.__future__.set_overwrite_module_params_on_conversion(overwrite)
        torch.__future__.set_swap_module_params_on_conversion(swap)
        with pytest.raises(AssertionError, match="FastGen requires torch.__future__") as caught:
            if operation == "prepare":
                # Casting and placement rely on preparation's up-front settings check.
                root = FastGenModule()
                root.inner = module
                root.prepare_module("cpu", precision_model=torch.bfloat16)
            elif operation == "materialize":
                module_state.materialize_meta_module(module, "cpu")
            else:
                module_state.load_module_state_dict(
                    module, {name: torch.zeros_like(tensor) for name, tensor in module.state_dict().items()}
                )
        # The overwrite setting is checked first.
        first = "overwrite" if overwrite else "swap"
        assert f"set_{first}_module_params_on_conversion(False)" in str(caught.value)
        assert torch.__future__.get_overwrite_module_params_on_conversion() == overwrite
        assert torch.__future__.get_swap_module_params_on_conversion() == swap
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(original[0])
        torch.__future__.set_swap_module_params_on_conversion(original[1])

    after = module_state.tensor_registrations(module)
    assert after.keys() == before.keys()
    for key, entry in after.items():
        assert entry.tensor is before[key].tensor
        assert entry.tensor.device == values[key].device
        assert entry.tensor.dtype == values[key].dtype
        if not entry.tensor.is_meta:
            torch.testing.assert_close(entry.tensor, values[key])
    assert module.weight is module.weight_alias


def test_has_meta_state_checks_parameters_buffers_and_none_slots():
    real = torch.nn.Module()
    real.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    real.register_parameter("optional_weight", None)
    real.register_buffer("cache", torch.ones(1), persistent=False)
    real.register_buffer("optional_cache", None)
    assert not module_state.has_meta_state(real)

    meta_parameter = torch.nn.Module()
    meta_parameter.register_parameter("weight", torch.nn.Parameter(torch.empty(1, device="meta")))
    assert module_state.has_meta_state(meta_parameter)

    meta_buffer = torch.nn.Module()
    meta_buffer.register_buffer("cache", torch.empty(1, device="meta"), persistent=False)
    assert module_state.has_meta_state(meta_buffer)


def test_has_meta_state_short_circuits_before_buffers():
    class MetaParameter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1, device="meta"))

        def buffers(self, *args, **kwargs):
            del args, kwargs
            pytest.fail("buffers() must not run after a meta parameter is found")

    assert module_state.has_meta_state(MetaParameter())


def test_registration_signature_includes_parameter_buffer_aliases():
    module = torch.nn.Module()
    tensor = torch.nn.Parameter(torch.ones(1))
    module.register_parameter("weight", tensor)
    module.register_parameter("weight_alias", tensor)
    module.register_buffer("cache", tensor)

    signature = module_state.registration_signature(module_state.tensor_registrations(module), include_device=False)
    aliases = (("buffer", "cache"), ("parameter", "weight"), ("parameter", "weight_alias"))
    assert isinstance(signature[("parameter", "weight")], module_state._TensorMetadata)
    assert signature[("parameter", "weight")].aliases == aliases
    assert signature[("parameter", "weight_alias")].aliases == aliases
    assert signature[("buffer", "cache")].aliases == aliases


def test_materialize_meta_module_allows_reset_values_built_in_compute_dtype():
    """A reset may compute in FP32, but must write into the dtype-selected storage."""

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16))
            self.register_buffer("cache", torch.empty(4, dtype=torch.bfloat16), persistent=False)

        def reset_parameters(self):
            torch.nn.init.ones_(self.weight)
            values = torch.arange(4, dtype=torch.float32, device=self.cache.device)
            self.cache.copy_(values)

    with torch.device("meta"):
        module = Root()

    module_state.materialize_meta_module(module, "cpu")

    assert module.weight.dtype == module.cache.dtype == torch.bfloat16
    assert torch.equal(module.weight, torch.ones(4, dtype=torch.bfloat16))
    assert torch.equal(module.cache, torch.arange(4, dtype=torch.bfloat16))


def test_materialize_meta_module_rebuilds_unregistered_tensor_state():
    """The root reset also repairs derived tensors invisible to module state traversal."""

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2))
            self.reset_parameters()

        def reset_parameters(self):
            torch.nn.init.ones_(self.weight)
            self.runtime_grid = torch.arange(3, dtype=torch.float32, device=self.weight.device)

    with torch.device("meta"):
        module = Root()
    assert module.runtime_grid.device.type == "meta"

    module_state.materialize_meta_module(module, "cpu")

    assert torch.equal(module.runtime_grid, torch.arange(3, dtype=torch.float32))


@pytest.mark.parametrize("mutation", ["dtype", "device"])
def test_materialize_meta_module_validates_allocation_before_accepting_reset(mutation):
    class InvalidAllocation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", torch.empty(2, device="meta"))

        def _apply(self, fn, recurse=True):
            super()._apply(fn, recurse=recurse)
            self.cache = self.cache.to(dtype=torch.float64) if mutation == "dtype" else self.cache.to("meta")
            return self

    with pytest.raises(RuntimeError, match="failed validation while materializing meta tensors"):
        module_state.materialize_meta_module(InvalidAllocation(), "cpu")


@pytest.mark.parametrize(
    "mutation",
    [
        "dtype",
        "shape",
        "layout",
        "requires_grad",
        "persistence",
        "missing",
        "unexpected",
        "kind",
        "none_to_tensor",
        "alias",
        "device",
    ],
)
def test_materialize_meta_module_rejects_reset_state_metadata_mismatch(mutation):
    class InvalidReset(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2))
            self.register_buffer("cache", torch.empty(2), persistent=False)
            self.register_buffer("optional", None, persistent=False)
            self.register_buffer("other", torch.empty(2), persistent=False)

        def reset_parameters(self):
            if mutation == "dtype":
                self.cache = torch.empty_like(self.cache, dtype=torch.float64)
            elif mutation == "shape":
                self.cache = self.cache.new_empty(3)
            elif mutation == "layout":
                indices = torch.tensor([[0]], device=self.cache.device)
                values = torch.ones(1, dtype=self.cache.dtype, device=self.cache.device)
                self.cache = torch.sparse_coo_tensor(indices, values, self.cache.shape)
            elif mutation == "requires_grad":
                self.weight.requires_grad_(False)
            elif mutation == "persistence":
                self._non_persistent_buffers_set.discard("cache")
            elif mutation == "missing":
                del self._buffers["cache"]
            elif mutation == "unexpected":
                self.register_buffer("extra", torch.empty_like(self.cache))
            elif mutation == "kind":
                replacement = torch.nn.Parameter(torch.empty_like(self.cache))
                del self._buffers["cache"]
                self.register_parameter("cache", replacement)
            elif mutation == "none_to_tensor":
                self.optional = torch.empty_like(self.cache)
            elif mutation == "alias":
                self.other = self.cache
            elif mutation == "device":
                self.cache = torch.empty_like(self.cache, device="meta")

    with torch.device("meta"):
        module = InvalidReset()

    with pytest.raises(RuntimeError, match="changed registered state"):
        module_state.materialize_meta_module(module, "cpu")


@pytest.mark.parametrize("shared_owner", [False, True])
def test_materialize_meta_module_rejects_parameter_replacement_by_reset(shared_owner):
    with torch.device("meta"):
        module = torch.nn.Module()
        module.left = torch.nn.Linear(2, 2, bias=False)
        if shared_owner:
            module.right = module.left

    def reset():
        module.left.weight = torch.nn.Parameter(torch.empty_like(module.left.weight))

    module.reset_parameters = reset
    with pytest.raises(RuntimeError, match="tensor identities.*must be preserved"):
        module_state.materialize_meta_module(module, "cpu")


def test_materialize_meta_module_rejects_stateless_owner_replacement_by_reset():
    with torch.device("meta"):
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.ones(2))
        module.dropout = torch.nn.Dropout()
    module.reset_parameters = lambda: setattr(module, "dropout", torch.nn.Dropout())
    with pytest.raises(RuntimeError, match="module owners must be preserved"):
        module_state.materialize_meta_module(module, "cpu")


def _materialize_and_collect_rng_warnings(module: torch.nn.Module, device: torch.device | str) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module_state.materialize_meta_module(module, device)
    messages = [str(warning.message) for warning in caught]
    return [message for message in messages if "RNG state during meta materialization" in message]


@pytest.mark.parametrize("draw", ["none", "forked", "global"])
def test_materialize_meta_module_warns_when_reset_advances_global_rng(draw):
    """Reset hooks should not draw random values; draws isolated by ``fork_rng`` leave the global state alone."""

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(4))

        def reset_parameters(self):
            if draw == "none":
                torch.nn.init.ones_(self.weight)
            elif draw == "forked":
                with torch.random.fork_rng(devices=[]):
                    torch.nn.init.normal_(self.weight)
            else:
                torch.nn.init.normal_(self.weight)

    with torch.device("meta"):
        module = Root()

    messages = _materialize_and_collect_rng_warnings(module, "cpu")

    if draw == "global":
        assert len(messages) == 1 and messages[0].startswith("Root.reset_parameters() advanced the cpu RNG state")
    else:
        assert messages == []
    assert module.weight.device.type == "cpu"


@RunIf(min_gpus=1)
def test_materialize_meta_module_warns_when_reset_draws_on_cuda():
    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(4))

        def reset_parameters(self):
            torch.nn.init.normal_(self.weight)

    with torch.device("meta"):
        module = Root()
    device = torch.device("cuda", torch.cuda.current_device())

    messages = _materialize_and_collect_rng_warnings(module, device)

    assert len(messages) == 1 and messages[0].startswith(f"Root.reset_parameters() advanced the {device} RNG state")


def test_materialize_meta_module_propagates_reset_exception_directly():
    class InvalidReset(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2))

        def reset_parameters(self):
            self.missing_buffer.copy_(torch.ones(2))

    with torch.device("meta"):
        module = InvalidReset()

    with pytest.raises(AttributeError, match="missing_buffer"):
        module_state.materialize_meta_module(module, "cpu")
