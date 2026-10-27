# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression cases at preparation, materialization, and checkpoint state boundaries."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from fastgen.networks.module import FastGenModule
from fastgen.utils.checkpointer import Checkpointer
import fastgen.utils.module as state
from tests.test_checkpointing import _get_checkpointer_config, _tiny_lora_net
from tests.utils import RunIf, run_distributed_test


@pytest.mark.parametrize("meta_kind", ["parameter", "buffer"])
def test_copy_load_rejects_meta_before_loading_real_state(meta_kind):
    module = torch.nn.Module()
    module.real = torch.nn.Parameter(torch.ones(2))
    if meta_kind == "parameter":
        module.unallocated = torch.nn.Parameter(torch.empty(2, device="meta"))
    else:
        module.register_buffer("unallocated", torch.empty(2, device="meta"))
    with pytest.raises(RuntimeError, match="materialize meta tensors first"):
        state.load_module_state_dict(module, {"real": torch.zeros(2), "unallocated": torch.zeros(2)}, require_any=True)
    assert torch.equal(module.real, torch.ones(2))
    assert module.unallocated.is_meta


@pytest.mark.parametrize("operation", ["prepare", "move", "monolithic", "dcp"])
@pytest.mark.parametrize("meta_kind", ["parameter", "persistent_buffer", "nonpersistent_buffer"])
def test_single_process_meta_synchronization_requires_real_source(tmp_path, operation, meta_kind):
    from fastgen.utils.distributed.module import move_and_sync_module

    def make_module(device):
        module = FastGenModule()
        module.weight = torch.nn.Parameter(torch.full((2,), 7.0))
        if meta_kind == "parameter":
            module.unallocated = torch.nn.Parameter(torch.full((2,), 9.0, device=device))
        else:
            module.register_buffer(
                "unallocated", torch.full((2,), 9.0, device=device), persistent=meta_kind == "persistent_buffer"
            )
        return module

    target = make_module("meta")
    before = target.weight, target.unallocated
    hooks = []
    target.reset_parameters = lambda: hooks.append("reset")
    target.register_load_state_dict_post_hook(lambda *args: hooks.append("load"))
    if operation in ("monolithic", "dcp"):
        source = make_module("cpu")
        with torch.no_grad():
            source.weight.zero_()
        checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=operation == "dcp")
        path = checkpointer.save(torch.nn.ModuleDict({"net": source}))

    with pytest.raises(RuntimeError, match="Rank 0 must hold the complete real state"):
        if operation == "prepare":
            target.prepare_module("cpu", trainable=True)
        elif operation == "move":
            move_and_sync_module(target, "cpu")
        else:
            checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path)
    assert target.weight is before[0] and target.unallocated is before[1]
    torch.testing.assert_close(target.weight, torch.full((2,), 7.0))
    assert target.unallocated.is_meta and not target.is_prepared and not hooks


def test_unknown_required_key_rejected_before_loading():
    module = torch.nn.Linear(2, 2, bias=False)
    before = module.weight.detach().clone()
    with pytest.raises(ValueError, match="unknown required target keys.*typo"):
        state.load_module_state_dict(module, {"weight": torch.zeros_like(before)}, required_keys=iter(["typo"]))
    assert torch.equal(module.weight, before)


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("mutation", ["omit_lora", "omit_extra_parameter", "omit_extra_buffer", "include_base"])
def test_checkpoint_rejects_adapter_exports_that_disagree_with_selection(tmp_path, monkeypatch, operation, mutation):
    config = _get_checkpointer_config(str(tmp_path))
    config.save_adapter_only = True
    checkpointer = Checkpointer(config)
    module = _tiny_lora_net(seed=0)
    module.extra.register_buffer("statistics", torch.ones(2))
    models = torch.nn.ModuleDict({"net": module})
    path = checkpointer.save(models, iteration=3)
    checkpoint_before = Path(path).read_bytes()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(7)
    before = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    original_export = module.adapter_state_dict

    def broken_export():
        exported = original_export()
        if mutation == "omit_lora":
            del exported[next(name for name in exported if "lora_" in name)]
        elif mutation == "omit_extra_parameter":
            del exported["extra.weight"]
        elif mutation == "omit_extra_buffer":
            del exported["extra.statistics"]
        else:
            name = next(name for name in before if not module._is_adapter_tensor(name))
            exported[name] = before[name]
        return exported

    monkeypatch.setattr(module, "adapter_state_dict", broken_export)
    loads = []
    module.register_load_state_dict_pre_hook(lambda *args: loads.append(True))
    with pytest.raises(RuntimeError, match=r"adapter_state_dict\(\).*exactly.*_is_adapter_tensor"):
        getattr(checkpointer, operation)(models, path=path)
    assert not loads
    assert Path(path).read_bytes() == checkpoint_before
    torch.testing.assert_close(module.state_dict(), before)


class _ExtraStateLinear(torch.nn.Linear):
    def __init__(self, features=2):
        super().__init__(features, features)
        self.extra_state_calls = []

    def get_extra_state(self):
        self.extra_state_calls.append("get")
        return {"version": 1}

    def set_extra_state(self, value):
        self.extra_state_calls.append("set")


class _LoadOnlyExtraStateLinear(_ExtraStateLinear):
    # PyTorch restores extra state through the setter even without an export hook.
    get_extra_state = torch.nn.Module.get_extra_state


@pytest.mark.parametrize("module_type", [_ExtraStateLinear, _LoadOnlyExtraStateLinear])
@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize(
    "kind,save_sharded",
    [("root", False), ("nested", False), ("root", True), ("nested", True), ("adapter", False)],
)
def test_checkpoint_rejects_extra_state_before_hooks_or_io(tmp_path, operation, kind, save_sharded, module_type):
    import fastgen.utils.checkpointer as checkpointing

    config = _get_checkpointer_config(str(tmp_path))
    config.save_adapter_only = kind == "adapter"
    checkpointer = Checkpointer(config, save_sharded=save_sharded)
    if kind == "adapter":
        source, target = _tiny_lora_net(seed=0), _tiny_lora_net(seed=1)
        stateful = target.extra = module_type(4)
    else:
        source, target = torch.nn.Linear(2, 2), module_type()
        stateful = target
        if kind == "nested":
            source, target = torch.nn.Sequential(source), torch.nn.Sequential(target)
    path = checkpointer.save(torch.nn.ModuleDict({"net": source}), iteration=3)
    checkpoint_before = Path(path).read_bytes()
    before = {name: tensor.clone() for name, tensor in target.named_parameters()}
    calls = []
    target.register_state_dict_pre_hook(lambda *args: calls.append("export"))
    target.register_load_state_dict_pre_hook(lambda *args: calls.append("load"))
    with patch.object(checkpointing.dcp, "save") as save, patch.object(checkpointing.dcp, "load") as load:
        with pytest.raises(RuntimeError, match=f"checkpoint {operation}:.*extra_state"):
            getattr(checkpointer, operation)(torch.nn.ModuleDict({"net": target}), path=path)
        save.assert_not_called()
        load.assert_not_called()
    assert not calls and not stateful.extra_state_calls
    assert Path(path).read_bytes() == checkpoint_before
    torch.testing.assert_close(dict(target.named_parameters()), before)


@pytest.mark.parametrize("module_type", [_ExtraStateLinear, _LoadOnlyExtraStateLinear])
@pytest.mark.parametrize("check_state_preservation", [False, True])
def test_direct_copy_load_rejects_extra_state_before_hooks(check_state_preservation, module_type):
    module = module_type()
    before = {name: tensor.clone() for name, tensor in module.named_parameters()}
    checkpoint = {name: torch.zeros_like(tensor) for name, tensor in before.items()}
    checkpoint["_extra_state"] = {"version": 1}
    with pytest.raises(RuntimeError, match="extra_state"):
        state.load_module_state_dict(module, checkpoint, check_state_preservation=check_state_preservation)
    assert not module.extra_state_calls
    torch.testing.assert_close(dict(module.named_parameters()), before)


def _cross_kind_module(device="cpu"):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2, device=device), requires_grad=False)
    # Matching type and requires_grad make per-kind metadata insufficient to detect this alias.
    module.register_buffer("cache", torch.nn.Parameter(torch.zeros(2, device=device), requires_grad=False))
    return module


def _tie_buffer_to_parameter(module, *args):
    module.register_buffer("cache", module.weight)


def test_copy_load_rejects_alias_created_by_post_hook():
    module = _cross_kind_module()
    module.register_load_state_dict_post_hook(_tie_buffer_to_parameter)
    with pytest.raises(RuntimeError, match="changed registered state"):
        state.load_module_state_dict(module, {"weight": torch.ones(2), "cache": torch.zeros(2)})


@pytest.mark.parametrize("mutation", ["split", "replace_owner", "dtype", "persistence"])
def test_copy_load_rejects_registration_changes_by_post_hook(mutation):
    module = torch.nn.Module()
    module.left = torch.nn.Linear(2, 2, bias=False)
    module.right = module.left
    module.tied = module.left.weight
    module.register_buffer("cache", torch.ones(2))
    checkpoint = module.state_dict()

    def mutate(target, _info):
        if mutation == "split":
            target.tied = torch.nn.Parameter(target.tied.detach().clone())
        elif mutation == "replace_owner":
            replacement = torch.nn.Module()
            replacement.weight = target.left.weight
            target.left = target.right = replacement
        elif mutation == "dtype":
            target.tied.data = target.tied.data.double()
        else:
            target._non_persistent_buffers_set.add("cache")

    module.register_load_state_dict_post_hook(mutate)
    with pytest.raises(RuntimeError, match="changed registered state"):
        state.load_module_state_dict(module, checkpoint)


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
@pytest.mark.parametrize("mutation", ["existing_cross_kind", "hook_cross_kind", "hook_split"])
def test_checkpoint_load_checks_the_entire_model_collection(tmp_path, save_sharded, mutation):
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    sources = torch.nn.ModuleDict({"a": _cross_kind_module(), "b": _cross_kind_module()})
    path = checkpointer.save(sources)
    targets = torch.nn.ModuleDict({"a": _cross_kind_module(), "b": _cross_kind_module()})
    targets["b"].tied = targets["b"].weight
    with torch.no_grad():
        targets["a"].weight.fill_(7)
    calls = []
    targets["b"].register_load_state_dict_pre_hook(lambda *args: calls.append("b"))
    if mutation == "existing_cross_kind":
        targets["b"].register_buffer("cache", targets["a"].weight)
    else:

        def mutate(*args):
            if mutation == "hook_cross_kind":
                targets["b"].register_buffer("cache", targets["a"].weight)
            else:
                targets["b"].tied = torch.nn.Parameter(targets["b"].weight.detach().clone())

        targets["a"].register_load_state_dict_post_hook(mutate)
    error, message = (
        (AssertionError, "both a parameter and a buffer")
        if mutation == "existing_cross_kind"
        else (RuntimeError, "changed registered state")
    )
    with pytest.raises(error, match=message):
        checkpointer.load(targets, path=path, strict=False)
    if save_sharded or mutation == "existing_cross_kind":
        assert not calls, "Invalid state must fail before mutation or the next DCP collective"
    if mutation == "existing_cross_kind":
        torch.testing.assert_close(targets["a"].weight, torch.full((2,), 7.0))


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
def test_checkpointer_reuses_one_collection_snapshot(tmp_path, save_sharded):
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    models = torch.nn.ModuleDict({"a": torch.nn.Linear(2, 2), "b": torch.nn.Linear(2, 2)})
    optimizers = {name: torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9) for name, module in models.items()}
    for name, module in models.items():
        module(torch.ones(2)).sum().backward()
        optimizers[name].step()
    callbacks = SimpleNamespace(state_dict=lambda: {"counter": 1}, load_state_dict=lambda _: None)

    for operation in ("save", "load"):
        with patch.object(state.ModuleStateSnapshot, "capture", wraps=state.ModuleStateSnapshot.capture) as capture:
            result = getattr(checkpointer, operation)(models, optimizer_dict=optimizers, callbacks=callbacks)
        assert capture.call_count == 1, "Model/optimizer/scalar operations must share one baseline"
        assert capture.call_args.args[0] is models, "Internal helpers must not capture per-model baselines"
        if operation == "load":
            assert result == 0


def _checkpoint_materialization_snapshot_worker(rank, world_size, checkpoint_dir, save_sharded):
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=save_sharded)
    torch.manual_seed(0)
    sources = torch.nn.ModuleDict({"a": torch.nn.Linear(2, 2), "b": torch.nn.Linear(2, 2)})
    path = checkpointer.save(sources, iteration=7)
    with torch.device("meta" if rank else "cpu"):
        targets = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in sources})
    original_capture = state.ModuleStateSnapshot.capture
    collection_meta_states = []

    def capture(module, operation):
        if module is targets:
            collection_meta_states.append(state.has_meta_state(module))
        return original_capture(module, operation)

    with patch.object(state.ModuleStateSnapshot, "capture", side_effect=capture):
        assert checkpointer.load(targets, path=path) == 7
    assert collection_meta_states == [False], "Capture only the final, fully materialized collection"
    torch.testing.assert_close(targets.state_dict(), sources.state_dict())
    return True


@RunIf(min_gpus=2)
@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
def test_checkpoint_captures_collection_once_after_all_materialization(tmp_path, save_sharded):
    assert run_distributed_test(
        _checkpoint_materialization_snapshot_worker,
        world_size=2,
        timeout=120,
        nccl_timeout=30,
        checkpoint_dir=str(tmp_path),
        save_sharded=save_sharded,
    )


def test_dcp_read_mutation_fails_before_the_next_collective(tmp_path, monkeypatch):
    import fastgen.utils.checkpointer as checkpointing

    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=True)
    models = torch.nn.ModuleDict({"a": _cross_kind_module(), "b": _cross_kind_module()})
    path = checkpointer.save(models)
    reads, copies = [], []
    original_load = checkpointing.dcp.load

    def mutate_after_read(*args, **kwargs):
        original_load(*args, **kwargs)
        reads.append(True)
        models["b"].weight = torch.nn.Parameter(models["b"].weight.detach().clone(), requires_grad=False)

    models["a"].register_load_state_dict_pre_hook(lambda *args: copies.append("a"))
    models["b"].register_load_state_dict_pre_hook(lambda *args: copies.append("b"))
    monkeypatch.setattr(checkpointing.dcp, "load", mutate_after_read)
    with pytest.raises(RuntimeError, match="changed registered state"):
        checkpointer.load(models, path=path)
    assert len(reads) == 1
    assert copies == ["a"], "Validate after the local copy, before the next model's DCP read"


def _checkpoint_extension_mutation_worker(rank, world_size, checkpoint_dir):
    import fastgen.utils.checkpointer as checkpointing

    results = {}
    for hook in ("optimizer_export", "optimizer_load", "callbacks_load"):
        checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=True)
        models = torch.nn.ModuleDict({"a": torch.nn.Linear(2, 2), "b": torch.nn.Linear(2, 2)})
        optimizers = {
            name: torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9) for name, module in models.items()
        }
        for name, module in models.items():
            module(torch.ones(2)).sum().backward()
            optimizers[name].step()

        def mutate(*args):
            if rank == 1:
                models["b"].weight = torch.nn.Parameter(models["b"].weight.detach().clone())

        callbacks = SimpleNamespace(state_dict=lambda: {"counter": 1}, load_state_dict=mutate)
        path = checkpointer.save(models, optimizer_dict=optimizers, callbacks=callbacks)
        later_calls = []
        if hook == "optimizer_export":
            optimizers["a"].register_state_dict_pre_hook(mutate)
            optimizers["b"].register_state_dict_pre_hook(lambda *args: later_calls.append(True))
        elif hook == "optimizer_load":
            optimizers["a"].register_load_state_dict_post_hook(mutate)
            optimizers["b"].register_load_state_dict_pre_hook(lambda *args: later_calls.append(True))

        operation = checkpointer.save if hook == "optimizer_export" else checkpointer.load
        with patch.object(checkpointing.dcp, "save", side_effect=AssertionError("Unexpected shard write")) as writes:
            try:
                operation(models, optimizer_dict=optimizers, callbacks=callbacks, path=path)
            except RuntimeError as error:
                # Batched exports may fail in a later local conversion before the phase's final
                # snapshot check. Every rank must still stop before writing or the next DCP read.
                rejected = "rank 1 failed" in str(error)
            else:
                rejected = False
        checks = [None] * world_size
        dist.all_gather_object(
            checks, rejected and not writes.called and (hook == "optimizer_export" or not later_calls)
        )
        results[hook] = checks
    return results


@RunIf(min_gpus=2)
def test_checkpoint_extension_mutations_fail_before_the_next_collective(tmp_path):
    results = run_distributed_test(
        _checkpoint_extension_mutation_worker,
        world_size=2,
        timeout=180,
        nccl_timeout=30,
        checkpoint_dir=str(tmp_path),
    )
    for checks in results.values():
        assert all(checks), results


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
def test_checkpoint_save_rejects_state_hidden_only_in_detached_export(tmp_path, monkeypatch, save_sharded):
    class HiddenWhenDetached(torch.nn.Linear):
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            super()._save_to_state_dict(destination, prefix, keep_vars)
            if not keep_vars:
                del destination[prefix + "weight"]

    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    path = checkpointer.save(torch.nn.ModuleDict({"net": torch.nn.Linear(2, 2)}), iteration=1)
    previous = Path(path).read_bytes()
    monkeypatch.setattr(checkpointer, "_save_checkpoint", lambda *args: pytest.fail("Unexpected checkpoint write"))
    monkeypatch.setattr(
        "fastgen.utils.checkpointer.dcp.save", lambda *args, **kwargs: pytest.fail("Unexpected shard write")
    )
    with pytest.raises(RuntimeError, match="does not export 1 registered tensor"):
        checkpointer.save(torch.nn.ModuleDict({"net": HiddenWhenDetached(2, 2)}), path=path, iteration=2)
    assert Path(path).read_bytes() == previous


@pytest.mark.parametrize("wrapper", ["checkpoint", "compile"])
def test_dcp_export_coverage_uses_canonical_wrapper_keys(tmp_path, wrapper):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=True)
    module = torch.nn.Linear(2, 2)
    target = checkpoint_wrapper(module) if wrapper == "checkpoint" else torch.compile(module, backend="eager")
    path = checkpointer.save(torch.nn.ModuleDict({"net": target}))
    fresh = torch.nn.Linear(2, 2)
    checkpointer.load(torch.nn.ModuleDict({"net": fresh}), path=path)
    for name, tensor in module.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[name], tensor)


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
@pytest.mark.parametrize("mutation", ["replace_local", "replace_other", "tie_roots"])
def test_checkpoint_export_validates_all_roots_before_any_write(tmp_path, monkeypatch, save_sharded, mutation):
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    models = torch.nn.ModuleDict({"a": _cross_kind_module(), "b": _cross_kind_module()})
    path = checkpointer.save(models, iteration=1)
    previous = Path(path).read_bytes()

    def mutate(*args):
        if mutation == "tie_roots":
            models["a"].weight = models["b"].weight
        else:
            target = models["a" if mutation == "replace_other" else "b"]
            target.weight = torch.nn.Parameter(target.weight.detach().clone(), requires_grad=False)

    # The later root's hook must fail before even the first root's shards are overwritten.
    models["b"].register_state_dict_post_hook(mutate)
    monkeypatch.setattr(checkpointer, "_save_checkpoint", lambda *args: pytest.fail("Unexpected checkpoint write"))
    monkeypatch.setattr(
        "fastgen.utils.checkpointer.dcp.save", lambda *args, **kwargs: pytest.fail("Unexpected shard write")
    )
    message = (
        r"state_dict\(\) does not export"
        if mutation == "replace_local"
        else "checkpoint state export changed registered state"
    )
    with pytest.raises(RuntimeError, match=message):
        checkpointer.save(models, path=path, iteration=2)
    assert Path(path).read_bytes() == previous
    fresh = torch.nn.ModuleDict({"a": _cross_kind_module(), "b": _cross_kind_module()})
    assert checkpointer.load(fresh, path=path) == 1
    for module in fresh.values():
        torch.testing.assert_close(module.weight, torch.ones(2))


def _checkpoint_export_mutation_worker(rank, world_size, checkpoint_dir, save_sharded, hook_root):
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=save_sharded)
    models = torch.nn.ModuleDict({"a": _cross_kind_module(), "b": _cross_kind_module()})
    path = checkpointer.save(models, iteration=1)
    previous = Path(path).read_bytes()

    def mutate(*args):
        models["a"].weight = torch.nn.Parameter(models["a"].weight.detach().clone(), requires_grad=False)

    # DCP exports on every rank; monolithic export is owned by rank zero.
    if rank == int(save_sharded):
        models[hook_root].register_state_dict_post_hook(mutate)
    with (
        patch("fastgen.utils.checkpointer.dcp.save", side_effect=AssertionError("Unexpected shard write")),
        patch.object(checkpointer, "_save_checkpoint", side_effect=AssertionError("Unexpected checkpoint write")),
    ):
        try:
            checkpointer.save(models, path=path, iteration=2)
        except RuntimeError as error:
            expected = (
                "state_dict() does not export"
                if hook_root == "a"
                else "checkpoint state export changed registered state"
            )
            rejected = expected in str(error)
        else:
            rejected = False
    checks = [None] * world_size
    dist.all_gather_object(checks, rejected and Path(path).read_bytes() == previous)
    return checks


@RunIf(min_gpus=2)
@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
@pytest.mark.parametrize("hook_root", ["a", "b"], ids=["first-root", "last-root"])
def test_checkpoint_export_mutations_fail_collectively_before_writes(tmp_path, save_sharded, hook_root):
    checks = run_distributed_test(
        _checkpoint_export_mutation_worker,
        world_size=2,
        timeout=120,
        nccl_timeout=30,
        checkpoint_dir=str(tmp_path),
        save_sharded=save_sharded,
        hook_root=hook_root,
    )
    assert checks is not None and all(checks), checks


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
@pytest.mark.parametrize("kind", ["parameter", "buffer", "nonpersistent_buffer", "stateless_child"])
@pytest.mark.parametrize("operation", ["save", "load"])
def test_checkpoint_rejects_cross_root_state_before_hooks(tmp_path, save_sharded, kind, operation):
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    sources = torch.nn.ModuleDict({"a": _cross_kind_module(), "b": _cross_kind_module()})
    path = checkpointer.save(sources)
    targets = torch.nn.ModuleDict({"a": _cross_kind_module(), "b": _cross_kind_module()})
    if kind == "stateless_child":
        targets["a"].dropout = torch.nn.Dropout()
        targets["b"].dropout = targets["a"].dropout
    elif kind == "parameter":
        targets["b"].weight = targets["a"].weight
    else:
        targets["b"].register_buffer("cache", targets["a"].cache, persistent=kind != "nonpersistent_buffer")
    with torch.no_grad():
        targets["a"].weight.fill_(7)
    calls = []
    targets["a"].register_state_dict_pre_hook(lambda *args: calls.append("export"))
    targets["a"].register_load_state_dict_pre_hook(lambda *args: calls.append("load"))
    expected = "modules" if kind == "stateless_child" else "tensors"
    with pytest.raises(AssertionError, match=f"root 'b' shares {expected} with another root"):
        getattr(checkpointer, operation)(targets, path=path)
    assert not calls
    torch.testing.assert_close(targets["a"].weight, torch.full((2,), 7.0))


@pytest.mark.parametrize("failure", ["export", "writer", "late_writer", "scalar"])
def test_dcp_setup_failure_preserves_previous_completion_marker(tmp_path, monkeypatch, failure):
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=True)
    models = torch.nn.ModuleDict({"net": _cross_kind_module(), "other": _cross_kind_module()})
    path = checkpointer.save(models, iteration=1)
    previous = Path(path).read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("injected initial setup failure")

    if failure == "export":
        models["net"].register_state_dict_pre_hook(fail)
    elif failure == "scalar":
        monkeypatch.setattr(checkpointer, "_scalar_state_dict", fail)
    elif failure == "late_writer":
        get_writer = checkpointer.get_storage_writer

        def writer(checkpoint_path):
            if checkpoint_path.endswith(".other_model"):
                fail()
            return get_writer(checkpoint_path)

        monkeypatch.setattr(checkpointer, "get_storage_writer", writer)
    else:
        monkeypatch.setattr(checkpointer, "get_storage_writer", fail)
    monkeypatch.setattr(
        "fastgen.utils.checkpointer.dcp.save", lambda *args, **kwargs: pytest.fail("Unexpected shard write")
    )
    with pytest.raises(RuntimeError, match="injected initial setup failure"):
        checkpointer.save(models, path=path, iteration=2)
    assert Path(path).read_bytes() == previous
    # The old marker still names intact shards, so a fresh target can restart from iteration 1.
    assert checkpointer.load(torch.nn.ModuleDict({k: _cross_kind_module() for k in models}), path=path) == 1


def _dcp_save_local_failure_worker(rank, world_size, checkpoint_dir):
    import fastgen.utils.checkpointer as checkpointing

    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=True)
    models = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in ("a", "b")})
    path = checkpointer.save(models, iteration=1)
    previous = Path(path).read_bytes()
    results = {}

    for failure in ("path", "writer", "scalar", "marker"):

        def fail(*args, **kwargs):
            raise RuntimeError(f"injected {failure} failure")

        def get_writer(checkpoint_path):
            if rank == 1 and checkpoint_path.endswith(".b_model"):
                fail()
            return checkpointing.FileSystemWriter(checkpoint_path)

        remove_artifact = checkpointing._remove_checkpoint_artifact

        def remove(path):
            if path.endswith(".pth"):
                fail()
            remove_artifact(path)

        with contextlib.ExitStack() as stack:
            writes = stack.enter_context(patch.object(checkpointing.dcp, "save"))
            publishes = stack.enter_context(patch.object(checkpointer, "_save_checkpoint"))
            if failure == "path" and rank == 1:
                stack.enter_context(patch.object(checkpointer, "resolve_save_path", side_effect=fail))
            elif failure == "writer":
                stack.enter_context(patch.object(checkpointer, "get_storage_writer", side_effect=get_writer))
            elif failure == "scalar" and rank == 0:
                stack.enter_context(patch.object(checkpointer, "_scalar_state_dict", side_effect=fail))
            elif failure == "marker":
                stack.enter_context(patch.object(checkpointing, "_remove_checkpoint_artifact", side_effect=remove))

            optimizers = {name: torch.optim.Adam(module.parameters()) for name, module in models.items()}
            try:
                checkpointer.save(models, optimizer_dict=optimizers, path=path, iteration=2)
            except RuntimeError as error:
                rejected = f"injected {failure} failure" in str(error)
            else:
                rejected = False

            marker_intact = Path(path).read_bytes() == previous
            checks = [None] * world_size
            dist.all_gather_object(
                checks,
                rejected
                and marker_intact
                and not publishes.called
                and not writes.called
                and all(not optimizer.state for optimizer in optimizers.values()),
            )
            results[failure] = checks
    return results


@RunIf(min_gpus=2)
def test_dcp_save_local_failures_stop_all_ranks_before_publication(tmp_path):
    results = run_distributed_test(
        _dcp_save_local_failure_worker, world_size=2, timeout=120, nccl_timeout=30, checkpoint_dir=str(tmp_path)
    )
    assert results is not None and all(all(checks) for checks in results.values()), results


def _checkpoint_mapping_failure_worker(rank, world_size, checkpoint_dir, save_sharded):
    import fastgen.utils.checkpointer as checkpointing
    from tests.test_checkpointing import _DCPMetaLoadModule

    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=save_sharded)
    path = checkpointer.save(torch.nn.ModuleDict({"net": _DCPMetaLoadModule(10.0, include_optional=False)}))
    with torch.device("meta" if rank else "cpu"):
        target = _DCPMetaLoadModule(100.0)

    helper = "_registration_names_for_dcp_keys" if save_sharded else "validate_state_dict_export"
    helper_module = checkpointing if save_sharded else state
    original_mapping = getattr(helper_module, helper)

    def map_missing_keys(*args):
        if rank == 1:
            raise RuntimeError("injected missing-key mapping failure")
        return original_mapping(*args)

    with (
        patch.object(helper_module, helper, side_effect=map_missing_keys),
        patch.object(checkpointing, "broadcast_module") as broadcast,
    ):
        try:
            checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path, strict=False, device="cpu")
        except RuntimeError as error:
            rejected = "injected missing-key mapping failure" in str(error)
        else:
            rejected = False
        checks = [None] * world_size
        dist.all_gather_object(
            checks, rejected and not broadcast.called and (not save_sharded or bool((target.weight == 11).all()))
        )
    return checks


@RunIf(min_gpus=2)
@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
def test_mapping_failure_stops_all_ranks_before_uncovered_state_broadcast(tmp_path, save_sharded):
    checks = run_distributed_test(
        _checkpoint_mapping_failure_worker,
        world_size=2,
        timeout=120,
        nccl_timeout=30,
        checkpoint_dir=str(tmp_path),
        save_sharded=save_sharded,
    )
    assert checks is not None and all(checks), checks


def _checkpoint_materialization_cross_root_tie_worker(rank, world_size, checkpoint_dir, save_sharded):
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=save_sharded)
    path = checkpointer.save(torch.nn.ModuleDict({name: torch.nn.Linear(2, 2, bias=False) for name in ("a", "b")}))
    targets = torch.nn.ModuleDict(
        {
            "a": torch.nn.Linear(2, 2, bias=False, device="meta" if rank else "cpu"),
            "b": torch.nn.Linear(2, 2, bias=False),
        }
    )
    with torch.no_grad():
        targets["b"].weight.fill_(7)

    def reset():
        targets["a"].weight = targets["b"].weight

    targets["a"].reset_parameters = reset
    with pytest.raises(RuntimeError, match="changed registered state"):
        checkpointer.load(targets, path=path)
    torch.testing.assert_close(targets["b"].weight, torch.full((2, 2), 7.0))
    return True


@RunIf(min_gpus=2)
@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
def test_checkpoint_materialization_cannot_tie_separate_roots(tmp_path, save_sharded):
    assert run_distributed_test(
        _checkpoint_materialization_cross_root_tie_worker,
        world_size=2,
        timeout=120,
        nccl_timeout=30,
        checkpoint_dir=str(tmp_path),
        save_sharded=save_sharded,
    )


def test_materialization_rejects_cross_kind_alias_created_by_allocation_without_reset():
    class CrossKindAllocation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, device="meta"), requires_grad=False)
            self.register_buffer("cache", torch.nn.Parameter(torch.empty(2, device="meta"), requires_grad=False))

        def _apply(self, fn, recurse=True):
            super()._apply(fn, recurse=recurse)
            # Keep both slots and their tensor metadata; only the cross-kind alias changes.
            self.register_buffer("cache", self.weight)
            return self

    with pytest.raises(RuntimeError, match="failed validation while materializing meta tensors"):
        state.materialize_meta_module(CrossKindAllocation(), "cpu")


def test_materialization_rejects_cross_kind_alias_created_by_reset():
    module = _cross_kind_module("meta")
    module.reset_parameters = lambda: _tie_buffer_to_parameter(module)
    with pytest.raises(RuntimeError, match="changed registered state"):
        state.materialize_meta_module(module, "cpu")


def test_materialization_restores_original_ties_before_reset():
    class Canonical(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.empty(2, device="meta"))
            self.second = self.first
            self.canonical = self.first

        def reset_parameters(self):
            assert self.first is self.second is self.canonical
            self.first.fill_(1)
            self.second.fill_(1)
            self.canonical.fill_(2)

        def tie_weights(self):
            raise AssertionError("materialization must not call tie_weights")

    module = Canonical()
    state.materialize_meta_module(module, "cpu")
    assert module.first is module.second is module.canonical
    torch.testing.assert_close(module.first, torch.full((2,), 2.0))


def test_materialization_rejects_parameter_ties_split_by_reset():
    module = torch.nn.Module()
    module.a = torch.nn.Parameter(torch.empty(2, device="meta"))
    module.b = module.c = module.d = module.a

    def reset():
        module.a = torch.nn.Parameter(torch.empty_like(module.a))

    module.reset_parameters = reset
    with pytest.raises(RuntimeError, match="changed registered state"):
        state.materialize_meta_module(module, "cpu")


@pytest.mark.parametrize("none_slot", [False, True])
def test_signature_distinguishes_shared_owners_from_distinct_tied_owners(none_slot):
    shared = torch.nn.Module()
    shared.left = torch.nn.Linear(2, 2, bias=False)
    shared.right = shared.left
    distinct = torch.nn.Module()
    distinct.left = torch.nn.Linear(2, 2, bias=False)
    distinct.right = torch.nn.Linear(2, 2, bias=False)
    distinct.right.weight = distinct.left.weight
    if none_slot:
        shared.left.weight = distinct.left.weight = distinct.right.weight = None
    first = state.registration_signature(state.tensor_registrations(shared), include_device=False)
    second = state.registration_signature(state.tensor_registrations(distinct), include_device=False)
    assert first == second
    assert state.module_topology_signature(shared) != state.module_topology_signature(distinct)


@pytest.mark.parametrize("operation", ["materialize", "prepare", "prepare_without_placement"])
def test_differentiable_buffers_rejected_before_placement(operation):
    module = FastGenModule()
    module.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    # Preparation also rejects rank-zero meta state, so only materialization starts from a meta buffer.
    buffer_device = "meta" if operation == "materialize" else "cpu"
    module.register_buffer("cache", torch.empty(2, device=buffer_device, requires_grad=True))
    before = state.registration_signature(state.tensor_registrations(module), include_device=True)
    assert before[("buffer", "cache")].requires_grad is True
    with pytest.raises(RuntimeError, match="buffers with requires_grad=True"):
        if operation == "materialize":
            state.materialize_meta_module(module, "cpu")
        else:
            module.prepare_module(
                None if operation == "prepare_without_placement" else "cpu", precision_model=torch.float32
            )
    assert state.registration_signature(state.tensor_registrations(module), include_device=True) == before


def test_duplicate_prepare_rejects_even_after_cross_kind_alias_added():
    module = FastGenModule()
    module.weight = torch.nn.Parameter(torch.ones(2))
    module.prepare_module("cpu")
    module.register_buffer("cache", module.weight)
    with pytest.raises(RuntimeError, match="already prepared"):
        module.prepare_module("cpu")


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
def test_checkpoint_save_rejects_cross_kind_alias_before_overwrite(tmp_path, save_sharded):
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    module = _cross_kind_module()
    marker = tmp_path / "checkpoint.pth"
    marker.write_bytes(b"previous completed checkpoint")
    _tie_buffer_to_parameter(module)
    with pytest.raises(AssertionError, match="both a parameter and a buffer"):
        checkpointer.save(torch.nn.ModuleDict({"net": module}), path=str(marker))
    assert marker.read_bytes() == b"previous completed checkpoint"
    assert not (tmp_path / "checkpoint.net_model").exists()


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
def test_checkpoint_save_rejects_alias_created_by_state_export(tmp_path, save_sharded):
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    module = _cross_kind_module()
    module.register_state_dict_post_hook(_tie_buffer_to_parameter)
    marker = tmp_path / "checkpoint.pth"
    with pytest.raises(RuntimeError, match="changed registered state"):
        checkpointer.save(torch.nn.ModuleDict({"net": module}), path=str(marker))
    assert not marker.exists()


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
def test_checkpoint_load_rejects_post_hook_cross_kind_alias(tmp_path, save_sharded):
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    module = _cross_kind_module()
    path = checkpointer.save(torch.nn.ModuleDict({"net": module}))
    module.register_load_state_dict_post_hook(_tie_buffer_to_parameter)
    with pytest.raises(RuntimeError, match="changed registered state"):
        checkpointer.load(torch.nn.ModuleDict({"net": module}), path=path)


@pytest.mark.parametrize("during_decode", [False, True])
def test_vae_offload_validates_all_components_before_either_move(monkeypatch, during_decode):
    import fastgen.callbacks.wandb as callbacks

    class Tracked(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moves = []
            self.register_buffer("cache", torch.ones(2))

        def to(self, device):
            self.moves.append(str(device))
            return super().to(device)

    first, second = Tracked(), Tracked()
    host = SimpleNamespace(init_vae=lambda: None, vae_context=lambda: contextlib.nullcontext(None))
    model = SimpleNamespace(
        net_preprocessors=host,
        module_dict={"first": first, "second": second},
        device=torch.device("cpu"),
        precision_amp_enc=None,
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: 0)
    monkeypatch.setattr(callbacks.basic_utils, "inference_mode", lambda **kwargs: contextlib.nullcontext())
    if not during_decode:
        second.register_buffer("alias", second.cache)
    with pytest.raises(AssertionError, match="multiple distinct buffer slots"):
        with callbacks.vae_decode_context(model, offload_modules=True):
            assert during_decode
            second.register_buffer("alias", second.cache)
    assert first.moves == second.moves == (["cpu"] if during_decode else [])


def _preparation_failure_worker(rank, world_size):
    from fastgen.utils.distributed.fsdp import FSDPContext

    results = {}
    for failure in ("cast", "fsdp_export"):
        module = FastGenModule()
        module.weight = torch.nn.Parameter(torch.ones(2, device="meta" if failure == "fsdp_export" and rank else "cpu"))
        mesh = SimpleNamespace(device_type="cuda", mesh=torch.arange(world_size), mesh_dim_names=None)
        context = FSDPContext(mesh, 1, torch.float32, None, False)
        if failure == "cast" and rank == 1:
            module._set_precision = lambda *args: (_ for _ in ()).throw(RuntimeError("injected cast failure"))
        if failure == "fsdp_export" and rank == 0:
            module.register_state_dict_pre_hook(
                lambda *args: (_ for _ in ()).throw(RuntimeError("injected export failure"))
            )
        with patch("fastgen.utils.distributed.fsdp._fully_shard_module") as shard:
            try:
                module.prepare_module(None, precision_model=torch.float32, fsdp_context=context)
            except RuntimeError as error:
                rejected = f"injected {'cast' if failure == 'cast' else 'export'} failure" in str(error)
            else:
                rejected = False
            checks = [None] * world_size
            dist.all_gather_object(checks, rejected and not shard.called and not module.is_prepared)
            results[failure] = checks
    return results


@RunIf(min_gpus=2)
def test_preparation_cast_and_fsdp_export_failures_reach_all_ranks():
    result = run_distributed_test(_preparation_failure_worker, world_size=2, timeout=120, nccl_timeout=30)
    assert result is not None and all(all(checks) for checks in result.values()), result


def _checkpoint_failure_worker(rank, world_size, checkpoint_dir, checkpoint_format):
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=checkpoint_format == "dcp")
    source = _cross_kind_module()
    path = checkpointer.save(torch.nn.ModuleDict({"net": source}), iteration=1)
    results = {}
    for failure in ("read", "export", "load_hook", "post_alias", "topology", "shared_owner", "cache_difference"):
        if failure == "export" and checkpoint_format != "dcp":
            continue
        target = _cross_kind_module()
        expected = "injected" if failure in {"read", "export", "load_hook"} else "changed registered state"
        if failure == "topology":
            independent = torch.nn.Parameter(target.cache.detach().clone(), requires_grad=False)
            target.register_buffer("cache_alias", target.cache if rank == 1 else independent)
            expected = "'registrations'"
        elif failure == "shared_owner":
            target.left = torch.nn.Linear(2, 2, bias=False)
            target.right = target.left if rank else torch.nn.Linear(2, 2, bias=False)
            target.right.weight = target.left.weight
            expected = "'topology'"
        elif failure == "cache_difference" and rank == 1:
            target.register_buffer("local_cache", torch.ones(3), persistent=False)

        def fail(*args, **kwargs):
            raise RuntimeError("injected checkpoint hook failure")

        with contextlib.ExitStack() as stack:
            if failure == "read" and rank == 0:
                # Every rank reads the file; a rank-0 read failure must reach rank 1 through raise_across_ranks.
                stack.enter_context(patch("fastgen.utils.checkpointer.load_training_checkpoint", side_effect=fail))
            elif rank == 1:
                if failure == "export":
                    target.register_state_dict_pre_hook(fail)
                elif failure == "load_hook":
                    target.register_load_state_dict_pre_hook(fail)
                elif failure == "post_alias":
                    target.register_load_state_dict_post_hook(_tie_buffer_to_parameter)
            try:
                checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path)
            except RuntimeError as error:
                passed = failure != "cache_difference" and expected in str(error)
            else:
                passed = failure == "cache_difference"
        checks = [None] * world_size
        dist.all_gather_object(checks, passed)
        results[failure] = checks
    return results


@RunIf(min_gpus=2)
@pytest.mark.parametrize("checkpoint_format", ["monolithic", "dcp"])
def test_checkpoint_local_failures_and_real_topology_are_collective(tmp_path, checkpoint_format):
    result = run_distributed_test(
        _checkpoint_failure_worker,
        world_size=2,
        timeout=180,
        nccl_timeout=30,
        checkpoint_dir=str(tmp_path),
        checkpoint_format=checkpoint_format,
    )
    assert result is not None and all(all(checks) for checks in result.values()), result


def _preparation_request_worker(rank, world_size):
    results = {}
    for difference in (
        "precision_model",
        "precision_opt",
        "trainable",
        "device",
        "device_type",
        "context",
        "context_device",
    ):
        module = FastGenModule()
        module.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        before = module.weight
        request = dict(device=None, precision_model=torch.bfloat16, precision_opt=torch.float32, trainable=False)
        if difference == "device":
            request["device"] = "cuda" if rank == 0 else None
        elif difference == "device_type":
            request["device"] = "cuda" if rank == 0 else "cpu"
        elif difference == "context":
            request["fsdp_context"] = (
                SimpleNamespace(device=torch.device("cuda", rank), _shard_module=lambda module: None)
                if rank == 0
                else None
            )
        elif difference == "context_device":
            # Under FSDP the context's execution device is the preparation target.
            request["fsdp_context"] = SimpleNamespace(
                device=torch.device("cpu" if rank == 1 else "cuda"), _shard_module=lambda module: None
            )
        elif rank == 1:
            request[difference] = {"precision_model": torch.float32, "precision_opt": torch.float64, "trainable": True}[
                difference
            ]
        try:
            module.prepare_module(**request)
        except RuntimeError as error:
            rejected = "prepare_module request mismatch" in str(error)
        else:
            rejected = False
        checks = [None] * world_size
        dist.all_gather_object(
            checks,
            rejected
            and module.weight is before
            and module.weight.dtype == torch.float64
            and module.precision_model is None
            and module.training
            and not module.is_prepared,
        )
        results[difference] = checks

    # Both placement-free preparation and rank-local CUDA ordinals remain valid when agreed.
    for target in (None, torch.device("cuda", rank)):
        module = FastGenModule()
        module.weight = torch.nn.Parameter(torch.ones(2))
        module.prepare_module(target, precision_model=torch.bfloat16)
        assert module.is_prepared and module.weight.dtype == torch.bfloat16
    return results


@RunIf(min_gpus=2)
def test_preparation_requests_must_agree_before_mutation():
    result = run_distributed_test(_preparation_request_worker, world_size=2, timeout=120, nccl_timeout=30)
    assert result is not None and all(all(checks) for checks in result.values()), result


def _broadcast_selection_worker(rank, world_size):
    from fastgen.utils.distributed.module import broadcast_module

    results = {}
    for difference in ("kind", "names", "empty_kind", "empty_names", "src", "invalid_src"):
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.full((2,), float(rank + 1), device="cuda"))
        module.register_buffer("cache", torch.full((2,), float(rank + 4), device="cuda"))
        before = {key: tensor.clone() for key, tensor in module.state_dict().items()}
        request = {}
        if difference == "kind":
            request.update(parameters=rank == 0, persistent_buffers=rank == 1)
        elif difference == "names":
            request["names"] = {"weight"} if rank == 0 else {"cache"}
        elif difference == "empty_kind" and rank == 1:
            request.update(parameters=False, persistent_buffers=False, nonpersistent_buffers=False)
        elif difference == "empty_names":
            request["names"] = None if rank == 0 else set()
        elif difference == "src":
            request["src"] = rank
        elif difference == "invalid_src":
            request["src"] = 0 if rank == 0 else world_size
        with patch("fastgen.utils.distributed.module._sync_params_and_buffers") as transport:
            try:
                broadcast_module(module, **request)
            except RuntimeError as error:
                rejected = "broadcast_module mismatch with rank 0" in str(error)
            else:
                rejected = False
        checks = [None] * world_size
        dist.all_gather_object(
            checks,
            rejected
            and not transport.called
            and all(torch.equal(tensor, before[key]) for key, tensor in module.state_dict().items()),
        )
        results[difference] = checks
    broadcast_module(module, src=1)
    assert torch.equal(module.weight, torch.full_like(module.weight, 2))
    assert torch.equal(module.cache, torch.full_like(module.cache, 5))
    return results


@RunIf(min_gpus=2)
def test_broadcast_source_and_payload_selection_must_agree():
    result = run_distributed_test(_broadcast_selection_worker, world_size=2, timeout=120, nccl_timeout=30)
    assert result is not None and all(all(checks) for checks in result.values()), result


def _fsdp_agreement_worker(rank, world_size):
    import fastgen.utils.distributed.fsdp as fsdp
    from fastgen.utils.distributed.fsdp import FSDPContext
    from torch.distributed.fsdp import CPUOffloadPolicy

    class Pinned(FastGenModule):
        @property
        def precision_model(self):
            return self.pin

    results = {}
    for difference in (
        "owners",
        "ties",
        "precision",
        "checkpointing",
        "wrap_failure",
        "placement_failure",
        "buffer_placement_failure",
    ):
        module = FastGenModule()
        module.left = Pinned()
        module.left.pin = torch.float32
        module.left.weight = torch.nn.Parameter(torch.ones(2))
        module.right = Pinned()
        module.right.pin = torch.float32
        module.right.weight = module.left.weight
        mesh = SimpleNamespace(device_type="cuda", mesh=torch.arange(world_size), mesh_dim_names=None)
        context = FSDPContext(mesh, 1, torch.float32, None, False)
        if difference == "buffer_placement_failure":
            context.cpu_offload = True
            context.offload_policy = CPUOffloadPolicy()
        if rank == 1:
            if difference == "owners":
                module.right = module.left
            elif difference == "ties":
                module.right.weight = torch.nn.Parameter(torch.ones(2))
            elif difference == "precision":
                # Untie both ranks below to avoid a local precision-policy conflict.
                module.left.pin = torch.bfloat16
            elif difference == "checkpointing":
                module.left.gradient_checkpointing = True
            elif difference == "placement_failure":
                module.to = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected placement failure"))
        if difference == "precision":
            module.right.weight = torch.nn.Parameter(torch.ones(2))

        def shard(*args, **kwargs):
            if not difference.endswith("failure"):
                raise AssertionError("Sharding began before agreement")
            if difference == "wrap_failure" and rank == 1:
                raise RuntimeError("injected wrap failure")

        original_move_buffers = fsdp._move_registered_buffers_

        def move_buffers(*args, **kwargs):
            if difference == "buffer_placement_failure" and rank == 1:
                raise RuntimeError("injected buffer placement failure")
            original_move_buffers(*args, **kwargs)

        with (
            patch.object(fsdp, "_fully_shard_module", side_effect=shard) as wrapped,
            patch.object(fsdp, "_move_registered_buffers_", side_effect=move_buffers),
        ):
            try:
                module.prepare_module(
                    None, precision_model=torch.float32, precision_opt=torch.float32, fsdp_context=context
                )
            except RuntimeError as error:
                rejected = (
                    "injected" in str(error)
                    if difference.endswith("failure")
                    else "FSDP pre-wrap" in str(error) and "mismatch" in str(error)
                )
            else:
                rejected = False
        checks = [None] * world_size
        dist.all_gather_object(
            checks,
            rejected
            and not module.is_prepared
            and not module._is_fsdp
            and (wrapped.called == difference.endswith("failure")),
        )
        results[difference] = checks
    return results


@RunIf(min_gpus=2)
def test_fsdp_validates_before_wrapping_and_commits_collectively():
    result = run_distributed_test(_fsdp_agreement_worker, world_size=2, timeout=180, nccl_timeout=30)
    assert result is not None and all(all(checks) for checks in result.values()), result


def _fsdp_parameter_order_worker(rank, world_size):
    from torch.distributed.device_mesh import init_device_mesh
    import fastgen.utils.distributed.fsdp as fsdp

    class Ordered(FastGenModule):
        def __init__(self, reverse):
            super().__init__()
            for name in ("b", "a") if reverse else ("a", "b"):
                self.register_parameter(name, torch.nn.Parameter(torch.full((4,), 1.0 if name == "a" else 2.0)))

        def forward(self, x):
            return x * (self.a + 10 * self.b)

    device = torch.device("cuda", rank)
    context = fsdp.FSDPContext(init_device_mesh("cuda", (world_size,)), 100, torch.float32, None, False)
    module = Ordered(reverse=bool(rank))
    before = module.a, module.b
    with patch.object(fsdp, "_fully_shard_module", wraps=fsdp._fully_shard_module) as wrapped:
        with pytest.raises(RuntimeError, match="FSDP pre-wrap policy.*mismatch"):
            module.prepare_module(device, precision_model=torch.float32, trainable=True, fsdp_context=context)
    assert not wrapped.called and not module.is_prepared and not module._is_fsdp
    assert module.a is before[0] and module.b is before[1]
    torch.testing.assert_close(module.a, torch.ones(4))
    torch.testing.assert_close(module.b, torch.full((4,), 2.0))

    # Identical registration order must still yield the correct gathered values on both ranks.
    module = Ordered(reverse=False)
    module.prepare_module(device, precision_model=torch.float32, trainable=True, fsdp_context=context)
    torch.testing.assert_close(module(torch.ones(4, device=device)), torch.full((4,), 21.0, device=device))
    return True


@RunIf(min_gpus=2)
def test_fsdp_rejects_parameter_order_mismatch_before_wrapping():
    assert run_distributed_test(_fsdp_parameter_order_worker, world_size=2, timeout=120, nccl_timeout=30)


def _fsdp_copy_hook_worker(rank, world_size):
    from torch.distributed.device_mesh import init_device_mesh
    from fastgen.utils.distributed.fsdp import FSDPContext

    module = FastGenModule()
    module.weight = torch.nn.Parameter(torch.ones(4, device="meta" if rank else "cpu"))
    mesh = init_device_mesh("cuda", (world_size,))
    context = FSDPContext(mesh, 1, torch.float32, None, False)

    def replace_parameter(target, _info):
        target.weight = torch.nn.Parameter(target.weight.detach().clone(), requires_grad=target.weight.requires_grad)

    if rank == 1:
        module.register_load_state_dict_post_hook(replace_parameter)
    try:
        module.prepare_module(None, precision_model=torch.float32, fsdp_context=context)
    except RuntimeError as error:
        rejected = "changed registered state" in str(error) and "rank 1" in str(error)
    else:
        rejected = False
    checks = [None] * world_size
    dist.all_gather_object(checks, rejected and not module.is_prepared and not module._is_fsdp)
    return checks


@RunIf(min_gpus=2)
def test_fsdp_state_load_rejects_hook_parameter_replacement_collectively():
    result = run_distributed_test(_fsdp_copy_hook_worker, world_size=2, timeout=120, nccl_timeout=30)
    assert result is not None and all(result), result


def _fsdp_export_hook_worker(rank, world_size, hook):
    from torch.distributed.device_mesh import init_device_mesh
    from fastgen.utils.distributed.fsdp import FSDPContext

    module = FastGenModule()
    module.weight = torch.nn.Parameter(torch.ones(4, device="meta" if rank else "cpu"))
    original = module.weight
    mesh = init_device_mesh("cuda", (world_size,))
    context = FSDPContext(mesh, 1, torch.float32, None, False)

    calls = []

    def replace_parameter(target, *_args):
        if hook == "pre_once" and calls:
            return
        calls.append(True)
        target.weight = torch.nn.Parameter(target.weight.detach().clone(), requires_grad=target.weight.requires_grad)

    if rank == 0:
        if hook == "pre_once":
            module.register_state_dict_pre_hook(replace_parameter)
        else:
            module.register_state_dict_post_hook(replace_parameter)
    with patch(
        "fastgen.utils.distributed.fsdp._fully_shard_module", side_effect=AssertionError("Unexpected FSDP wrapping")
    ) as wrapped:
        try:
            module.prepare_module(None, precision_model=torch.float32, fsdp_context=context)
        except RuntimeError as error:
            rejected = "FSDP source state export" in str(error) and "rank 0" in str(error)
        else:
            rejected = False
    checks = [None] * world_size
    dist.all_gather_object(
        checks,
        rejected
        and not wrapped.called
        and not module.is_prepared
        and not module._is_fsdp
        # Validation detects the rank-zero mutation; it does not roll it back.
        and (module.weight is original) == (rank != 0),
    )
    return checks


@RunIf(min_gpus=2)
@pytest.mark.parametrize("hook", ["post", "pre_once"])
def test_fsdp_source_export_rejects_hook_parameter_replacement_collectively(hook):
    result = run_distributed_test(_fsdp_export_hook_worker, world_size=2, timeout=120, nccl_timeout=30, hook=hook)
    assert result is not None and all(result), result


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("hidden", ["weight", "cache"])
def test_checkpoint_requires_complete_ordinary_exports(tmp_path, save_sharded, operation, hidden):
    module = torch.nn.Linear(2, 2)
    module.register_buffer("cache", torch.ones(2))
    models = torch.nn.ModuleDict({"net": module})
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    path = checkpointer.save(models, iteration=3)
    marker_before = Path(path).read_bytes()
    weight_before = module.weight.detach().clone()
    loaded = []

    def hide(_module, state_dict, prefix, _local_metadata):
        del state_dict[prefix + hidden]

    module.register_state_dict_post_hook(hide)
    module.register_load_state_dict_post_hook(lambda *args: loaded.append(True))
    with pytest.raises(RuntimeError, match="does not export 1 registered tensor"):
        if operation == "save":
            checkpointer.save(models, path=path, iteration=4)
        else:
            checkpointer.load(models, path=path, strict=False)
    assert Path(path).read_bytes() == marker_before
    assert not loaded
    torch.testing.assert_close(module.weight, weight_before)


def test_direct_copy_load_rejects_a_target_with_hidden_persistent_state():
    module = torch.nn.Linear(2, 2)
    source = module.state_dict()

    def hide(_module, state_dict, prefix, _local_metadata):
        del state_dict[prefix + "weight"]

    module.register_state_dict_post_hook(hide)
    with pytest.raises(RuntimeError, match="does not export 1 registered tensor"):
        state.load_module_state_dict(module, source)
