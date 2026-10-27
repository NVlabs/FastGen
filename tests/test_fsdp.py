# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for FSDP2 weight synchronization and forward pass correctness.

These tests verify:
1. FSDP weight synchronization from rank 0 to all ranks
2. Buffer synchronization across ranks
3. Forward pass consistency between FSDP and reference models
4. State dict gathering and comparison
5. Meta device initialization for memory-efficient loading (meta_init)
6. End-to-end FSDP-context preparation with real model instantiation
7. Standard FSDP path without meta init (all ranks load independently)
8. Multi-network models (DMD2 with net, teacher, fake_score)

Tests require at least 2 GPUs and are run using torch.multiprocessing.spawn
to simulate distributed training.

Usage:
    pytest tests/test_fsdp.py -v

    # Or run with specific tests:
    pytest tests/test_fsdp.py::test_fsdp_weight_sync -v
    pytest tests/test_fsdp.py::test_fsdp_meta_device_init -v
    pytest tests/test_fsdp.py::test_fsdp_context_code_path -v
    pytest tests/test_fsdp.py::test_fsdp_context_no_meta_init -v
    pytest tests/test_fsdp.py::test_dmd2_fsdp_context -v
"""

import contextlib
import copy
import math
from types import SimpleNamespace
from typing import Dict
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
from diffusers import ModelMixin
from transformers import BertConfig, PreTrainedModel
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.utils._python_dispatch import TorchDispatchMode
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from fastgen.configs.instantiate import instantiate
import fastgen.utils.distributed.fsdp as fsdp_utils
from tests.utils import RunIf, run_distributed_test
from fastgen.utils.basic_utils import clear_gpu_memory
from fastgen.utils.env import set_env_vars
from fastgen.utils.tree_utils import to
from fastgen.networks.network import FastGenModule
from fastgen.methods.model import FastGenModel


class _DiffusersDynamicDtypeChild(ModelMixin):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.inside_dtype = None

    def forward(self, x):
        self.inside_dtype = self.dtype
        return self.proj(x)


class _TransformersDynamicDtypeChild(PreTrainedModel):
    config_class = BertConfig

    def __init__(self):
        super().__init__(BertConfig(hidden_size=4, intermediate_size=8, num_attention_heads=1, num_hidden_layers=1))
        self.proj = torch.nn.Linear(4, 4)
        self.inside_dtype = None

    def forward(self, x):
        self.inside_dtype = self.dtype
        return self.proj(x)


class _ExplicitBoundaryPreprocessor(FastGenModule):
    def __init__(self):
        super().__init__()
        self.diffusers_child = _DiffusersDynamicDtypeChild()
        self.transformers_child = _TransformersDynamicDtypeChild()
        # This fixture exercises the forward boundary while FSDP owns storage precision.
        self._record_precision(precision_model=torch.bfloat16)
        self.child_dtypes_at_parent = None

    def encode(self, state):
        return self(state)

    def forward(self, state):
        input_dtype = state.dtype
        self.child_dtypes_at_parent = (self.diffusers_child.dtype, self.transformers_child.dtype)
        model_input = state.to(dtype=self.precision_model)
        output = self.diffusers_child(model_input) + self.transformers_child(model_input)
        return output.to(dtype=input_dtype)


def _test_nested_fsdp_dynamic_dtype_impl(rank: int, world_size: int) -> Dict:
    del rank
    mesh = init_device_mesh("cuda", (world_size,))
    model = _ExplicitBoundaryPreprocessor().to(device=torch.cuda.current_device(), dtype=torch.float32)
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=False,
    )
    fully_shard(model.diffusers_child, mesh=mesh, mp_policy=policy)
    fully_shard(model.transformers_child, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)

    state = torch.randn(2, 4, device=torch.cuda.current_device(), dtype=torch.float32, requires_grad=True)
    output = model.encode(state)
    output.sum().backward()
    return {
        "parent_child_dtypes": model.child_dtypes_at_parent,
        "diffusers_inside_dtype": model.diffusers_child.inside_dtype,
        "transformers_inside_dtype": model.transformers_child.inside_dtype,
        "output_dtype": output.dtype,
        "output_device": output.device.type,
        "input_grad_dtype": state.grad.dtype,
    }


@RunIf(min_gpus=1)
def test_nested_fsdp_dynamic_dtype_is_storage_at_parent_boundary():
    result = run_distributed_test(
        test_fn=_test_nested_fsdp_dynamic_dtype_impl,
        world_size=1,
        timeout=120,
        setup_fn=set_env_vars,
    )
    assert result == {
        "parent_child_dtypes": (torch.float32, torch.float32),
        "diffusers_inside_dtype": torch.bfloat16,
        "transformers_inside_dtype": torch.bfloat16,
        "output_dtype": torch.float32,
        "output_device": "cuda",
        "input_grad_dtype": torch.float32,
    }


def _install_mp_dtype_patch(monkeypatch, param_dtypes):
    """Patch a frozen-group ``_init_mp_dtypes`` and capture the fallback logs.

    ``param_dtypes`` is the (orig, reduce) dtype pair of each param in the group.
    """

    def upstream_init(self):
        self._orig_dtype = None
        self._reduce_dtype = None
        for param, (orig_dtype, reduce_dtype) in zip(self.fsdp_params, param_dtypes):
            param.orig_dtype = orig_dtype
            param.reduce_dtype = reduce_dtype

    monkeypatch.setattr(FSDPParamGroup, "_init_mp_dtypes", upstream_init)
    logs = {"debug": [], "info": [], "warning": []}
    for level, messages in logs.items():
        monkeypatch.setattr(fsdp_utils.logger, level, messages.append)

    assert fsdp_utils.patch_fsdp2_mp_dtypes()

    group = SimpleNamespace(
        fsdp_params=[SimpleNamespace() for _ in param_dtypes],
        _module_fqn="discriminator",
    )
    FSDPParamGroup._init_mp_dtypes(group)
    return group, logs


def test_patch_fsdp2_mp_dtypes_fills_uniform_frozen_group_dtypes(monkeypatch):
    group, logs = _install_mp_dtype_patch(monkeypatch, [(torch.float32, torch.float32)] * 2)

    assert group._orig_dtype is torch.float32
    assert group._reduce_dtype is torch.float32
    # Frozen groups are expected (frozen connectors / teacher), so this stays at debug.
    assert any("FSDP2 mp-dtype fallback applied" in message for message in logs["debug"])
    assert not logs["info"] and not logs["warning"]


def test_patch_fsdp2_mp_dtypes_treats_clamped_reduce_dtype_as_resolved(monkeypatch):
    """A None reduce dtype is upstream's "reduce in the compute dtype" clamp, not a disagreement."""
    group, logs = _install_mp_dtype_patch(monkeypatch, [(torch.float32, None)] * 2)

    assert group._orig_dtype is torch.float32
    assert group._reduce_dtype is None
    assert any("FSDP2 mp-dtype fallback applied" in message for message in logs["debug"])
    assert any("reduce in the compute dtype" in message for message in logs["debug"])
    assert not logs["info"] and not logs["warning"]


def test_patch_fsdp2_mp_dtypes_logs_info_when_dtypes_disagree(monkeypatch):
    group, logs = _install_mp_dtype_patch(
        monkeypatch, [(torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16)]
    )

    # Non-uniform per-param dtypes cannot be resolved, so the group is reported.
    assert group._orig_dtype is None
    assert group._reduce_dtype is None
    assert any("unresolved" in message and "orig=" in message and "reduce=" in message for message in logs["info"])
    assert not logs["debug"]


def test_make_fully_shard_kwargs():
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )

    kwargs = {"mesh": "mesh", "mp_policy": policy}
    fp32_kwargs = fsdp_utils.make_fully_shard_kwargs(
        kwargs,
        param_dtype=torch.float32,
        storage_dtype=torch.bfloat16,
        enabled=True,
        reshard_after_forward=False,
    )

    assert kwargs == {"mesh": "mesh", "mp_policy": policy}
    assert fp32_kwargs is not None
    assert fp32_kwargs["mp_policy"].param_dtype is torch.float32
    assert fp32_kwargs["mp_policy"].reduce_dtype is torch.float32
    assert fp32_kwargs["mp_policy"].output_dtype is torch.bfloat16
    assert fp32_kwargs["mp_policy"].cast_forward_inputs is True
    assert fp32_kwargs["reshard_after_forward"] is False

    direct_kwargs = fsdp_utils.make_fully_shard_kwargs(
        kwargs,
        param_dtype=torch.float32,
        storage_dtype=torch.bfloat16,
        enabled=True,
        cast_forward_inputs=False,
    )
    assert direct_kwargs is not None
    direct_policy = direct_kwargs["mp_policy"]
    assert direct_policy.param_dtype is torch.float32
    assert direct_policy.cast_forward_inputs is False

    fp16_kwargs = fsdp_utils.make_fully_shard_kwargs(
        kwargs,
        param_dtype=torch.float16,
        storage_dtype=torch.bfloat16,
        enabled=True,
    )
    assert fp16_kwargs is not None
    assert fp16_kwargs["mp_policy"].param_dtype is torch.float16

    assert (
        fsdp_utils.make_fully_shard_kwargs(
            kwargs,
            param_dtype=torch.float32,
            storage_dtype=torch.bfloat16,
            enabled=False,
        )
        is None
    )
    assert (
        fsdp_utils.make_fully_shard_kwargs(
            {"mesh": "mesh"},
            param_dtype=torch.float32,
            storage_dtype=torch.bfloat16,
            enabled=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("parent_compute_dtype", "storage_dtype", "expected_boundary"),
    [
        pytest.param(torch.bfloat16, torch.bfloat16, True, id="different-compute"),
        pytest.param(torch.float32, torch.bfloat16, True, id="different-storage"),
        pytest.param(torch.float32, torch.float32, False, id="uniform-fp32"),
        pytest.param(None, torch.bfloat16, True, id="no-cast-bf16-storage"),
        pytest.param(None, torch.float32, False, id="no-cast-fp32-storage"),
    ],
)
def test_make_fully_shard_kwargs_uses_compute_and_storage_boundaries(
    parent_compute_dtype,
    storage_dtype,
    expected_boundary,
):
    policy = MixedPrecisionPolicy(param_dtype=parent_compute_dtype, reduce_dtype=torch.float32)

    result = fsdp_utils.make_fully_shard_kwargs(
        {"mesh": "mesh", "mp_policy": policy},
        param_dtype=torch.float32,
        storage_dtype=storage_dtype,
        enabled=True,
    )

    assert (result is not None) is expected_boundary
    if result is not None:
        assert result["mp_policy"].param_dtype is torch.float32


def test_create_fsdp_context_prepares_run_wide_state_once(monkeypatch):
    mesh = SimpleNamespace(device_type="cuda")
    calls = []

    monkeypatch.setattr(fsdp_utils, "world_size", lambda: 4)
    monkeypatch.setattr("fastgen.utils.distributed.collectives.world_size", lambda: 1)
    monkeypatch.setattr(fsdp_utils.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(fsdp_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp_utils.torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        fsdp_utils,
        "init_device_mesh",
        lambda *args, **kwargs: calls.append(("mesh", args, kwargs)) or mesh,
    )
    monkeypatch.setattr(
        fsdp_utils,
        "patch_fsdp2_mp_dtypes",
        lambda: calls.append(("patch",)) or True,
    )
    monkeypatch.setattr(fsdp_utils.torch.cuda, "current_device", lambda: 2)

    context = fsdp_utils.create_fsdp_context(
        reduce_dtype=torch.bfloat16,
        min_num_params=123,
        apply_cpu_offload=False,
        sharding_group_size=2,
    )

    assert calls == [
        ("patch",),
        ("mesh", ("cuda", (2, 2)), {"mesh_dim_names": ("replicate", "shard")}),
    ]
    assert context.mesh is mesh
    assert context.min_num_params == 123
    assert context.reduce_dtype is torch.bfloat16
    assert context.offload_policy is None
    assert context.device == torch.device("cuda", 2)
    assert context.cpu_offload is False


def test_create_fsdp_context_fails_before_mesh_when_mp_dtype_patch_fails(monkeypatch):
    monkeypatch.setattr(fsdp_utils, "world_size", lambda: 2)
    monkeypatch.setattr("fastgen.utils.distributed.collectives.world_size", lambda: 1)
    monkeypatch.setattr(fsdp_utils.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(fsdp_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        fsdp_utils,
        "init_device_mesh",
        lambda *_args, **_kwargs: pytest.fail("device mesh was initialized before the compatibility check"),
    )
    monkeypatch.setattr(
        fsdp_utils, "patch_fsdp2_mp_dtypes", Mock(side_effect=ImportError("FSDPParamGroup internals changed"))
    )

    with pytest.raises(ImportError, match="FSDPParamGroup internals changed"):
        fsdp_utils.create_fsdp_context(reduce_dtype=torch.bfloat16)


def test_create_fsdp_context_requires_initialized_process_group(monkeypatch):
    monkeypatch.setattr(fsdp_utils.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(fsdp_utils.torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="requires an initialized distributed process group"):
        fsdp_utils.create_fsdp_context(reduce_dtype=torch.float32)


@pytest.mark.parametrize("group_size", [0, -1, True, 1.5, 3])
def test_create_fsdp_context_rejects_invalid_group_sizes_before_mesh_construction(monkeypatch, group_size):
    monkeypatch.setattr(fsdp_utils, "world_size", lambda: 2)
    monkeypatch.setattr("fastgen.utils.distributed.collectives.world_size", lambda: 1)
    monkeypatch.setattr(fsdp_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp_utils, "patch_fsdp2_mp_dtypes", lambda: True)
    with patch.object(fsdp_utils, "init_device_mesh") as mesh:
        with pytest.raises(ValueError, match="positive integer|divisible"):
            fsdp_utils.create_fsdp_context(torch.float32, sharding_group_size=group_size)
    mesh.assert_not_called()


def _fsdp_context_initialization_agreement_worker(rank, world_size):
    for difference in ("invalid_group", "mesh", "patch", "reduce_dtype", "threshold", "offload"):
        kwargs = {"reduce_dtype": torch.float32}
        if rank == 1:
            kwargs.update(
                {
                    "invalid_group": {"sharding_group_size": 3},
                    "mesh": {"sharding_group_size": 1},
                    "patch": {},
                    "reduce_dtype": {"reduce_dtype": torch.bfloat16},
                    "threshold": {"min_num_params": 1},
                    "offload": {"apply_cpu_offload": True},
                }[difference]
            )
        with (
            patch.object(
                fsdp_utils,
                "patch_fsdp2_mp_dtypes",
                side_effect=ImportError("injected") if rank == 1 and difference == "patch" else None,
            ),
            patch.object(fsdp_utils, "init_device_mesh") as mesh,
        ):
            expected = (
                "FSDP context initialization" if difference in ("invalid_group", "patch") else "FSDP context plan"
            )
            with pytest.raises(RuntimeError, match=expected):
                fsdp_utils.create_fsdp_context(**kwargs)
        checks = [None] * world_size
        dist.all_gather_object(checks, not mesh.called)
        assert all(checks), (difference, checks)
    return True


@RunIf(min_gpus=2)
def test_fsdp_context_agrees_before_mesh_construction():
    assert run_distributed_test(
        _fsdp_context_initialization_agreement_worker, world_size=2, timeout=120, nccl_timeout=30
    )


def test_reshard_fsdp_modules_reshards_every_fsdp_module_below_the_roots(monkeypatch):
    calls = []

    class FakeFSDPModule(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def reshard(self):
            calls.append(self.name)

    root = FakeFSDPModule("root")
    root.child = FakeFSDPModule("child")
    root.child.grandchild = FakeFSDPModule("grandchild")
    root.plain = torch.nn.Linear(2, 2)
    monkeypatch.setattr(fsdp_utils, "FSDPModule", FakeFSDPModule)

    fsdp_utils.reshard_fsdp_modules((root.child, root))

    # ``reshard`` is per group and idempotent, so order and repeats under overlapping roots do not matter.
    assert set(calls) == {"grandchild", "child", "root"}


def test_apply_fsdp_checkpointing_wraps_block_types_in_their_containers():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])
            self.head = torch.nn.Linear(2, 2)

    net = Net()
    blocks = list(net.blocks)

    assert fsdp_utils.apply_fsdp_checkpointing(net, Block) == 2

    # The wrappers replace the blocks in their container, so the sharding loops that run afterwards
    # form the FSDP groups on the wrappers, outside the checkpoint boundary.
    assert all(isinstance(block, CheckpointWrapper) for block in net.blocks)
    assert all(wrapped._checkpoint_wrapped_module is block for wrapped, block in zip(net.blocks, blocks))
    assert not isinstance(net.head, CheckpointWrapper)
    assert not isinstance(blocks[0].proj, CheckpointWrapper)


def test_fsdp_context_shards_a_precision_configured_module_once_across_contexts(monkeypatch):
    module = FastGenModule()
    module.projection = torch.nn.Linear(2, 2)
    module.nested = FastGenModule()
    module.nested.projection = torch.nn.Linear(2, 2)
    shard_calls = []
    sync_calls = []
    warnings = []

    monkeypatch.setattr(fsdp_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        fsdp_utils,
        "sync_any",
        lambda value, **_kwargs: sync_calls.append(value) or False,
    )
    monkeypatch.setattr(fsdp_utils.logger, "warning", warnings.append)
    monkeypatch.setattr(
        fsdp_utils,
        "_fully_shard_module",
        lambda candidate, **kwargs: shard_calls.append((candidate, kwargs)),
    )

    context = fsdp_utils.FSDPContext(
        mesh=SimpleNamespace(device_type="cpu"),
        min_num_params=7,
        reduce_dtype=torch.float32,
        offload_policy=None,
        cpu_offload=False,
    )
    assert (
        module.prepare_module("cpu", precision_model=torch.bfloat16, precision_opt=torch.float64, fsdp_context=context)
        is module
    )
    second_context = fsdp_utils.FSDPContext(
        mesh=SimpleNamespace(device_type="cpu"),
        min_num_params=999,
        reduce_dtype=torch.bfloat16,
        offload_policy=object(),
        cpu_offload=True,
    )
    with pytest.raises(RuntimeError, match="already prepared"):
        module.prepare_module("cpu", fsdp_context=second_context)

    assert len(shard_calls) == 1
    assert sync_calls == [False]
    assert warnings == []
    assert module._is_fsdp and module.is_prepared
    assert not module.nested._is_fsdp
    assert module.nested.is_prepared
    candidate, kwargs = shard_calls[0]
    assert candidate is module
    assert kwargs["min_num_params"] == 7
    assert kwargs["mesh"] is context.mesh
    assert kwargs["offload_policy"] is None
    policy = kwargs["mp_policy"]
    assert policy.param_dtype is torch.bfloat16
    assert policy.reduce_dtype is torch.float32
    assert policy.output_dtype is None
    assert policy.cast_forward_inputs is False
    assert module.projection.weight.dtype is torch.float64


def test_fsdp_context_preserves_native_dtype_without_precision_configuration(monkeypatch):
    module = FastGenModule()
    module.projection = torch.nn.Linear(2, 2, dtype=torch.float64)
    shard_calls = []

    monkeypatch.setattr(fsdp_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(fsdp_utils, "sync_any", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        fsdp_utils,
        "_fully_shard_module",
        lambda candidate, **kwargs: shard_calls.append((candidate, kwargs)),
    )

    context = fsdp_utils.FSDPContext(
        mesh=SimpleNamespace(device_type="cpu"),
        min_num_params=7,
        reduce_dtype=torch.float32,
        offload_policy=None,
        cpu_offload=False,
    )
    context._shard_module(module)

    assert len(shard_calls) == 1
    policy = shard_calls[0][1]["mp_policy"]
    assert policy.param_dtype is None
    assert module.projection.weight.dtype is torch.float64


@pytest.mark.parametrize("source_device", ["cpu", "meta"])
@pytest.mark.parametrize("alias_kind", ["parameter_buffer", "buffer"])
def test_fsdp_preparation_rejects_aliases_before_casting_and_sharding(monkeypatch, source_device, alias_kind):
    with torch.device(source_device):
        module = FastGenModule()
        module.weight = torch.nn.Parameter(torch.ones(2))
        module.register_buffer("cache", torch.ones(2))
        module.register_buffer("alias", module.weight if alias_kind == "parameter_buffer" else module.cache)
    original_weight, original_buffer = module.weight, module.cache
    shard_calls = []

    monkeypatch.setattr(fsdp_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(fsdp_utils, "sync_any", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(fsdp_utils, "_fully_shard_module", lambda *_args, **_kwargs: shard_calls.append(True))

    context = fsdp_utils.FSDPContext(
        mesh=SimpleNamespace(device_type="cpu"),
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=None,
        cpu_offload=False,
    )

    with pytest.raises(AssertionError, match="both a parameter and a buffer|multiple distinct buffer slots"):
        module.prepare_module("cpu", precision_model=torch.bfloat16, fsdp_context=context)

    assert shard_calls == []
    assert not module._is_fsdp
    assert not module.is_prepared
    assert module.device is None and module.precision_model is None
    assert module.weight is original_weight and module.cache is original_buffer
    assert module.weight.dtype == module.cache.dtype == torch.float32
    assert module.weight.device.type == module.cache.device.type == source_device
    assert module.training and module.weight.requires_grad


def test_fsdp_preparation_sets_completion_markers_only_after_placement_succeeds(monkeypatch):
    module = FastGenModule()
    module.projection = torch.nn.Linear(2, 2)
    module._set_precision(torch.float32)
    shard_calls = []

    monkeypatch.setattr(fsdp_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(fsdp_utils, "sync_any", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(fsdp_utils, "_fully_shard_module", lambda candidate, **_kwargs: shard_calls.append(candidate))

    def fail_placement(*_args, **_kwargs):
        raise RuntimeError("injected placement failure")

    monkeypatch.setattr(module, "to", fail_placement)
    context = fsdp_utils.FSDPContext(
        mesh=SimpleNamespace(device_type="cpu"),
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=None,
        cpu_offload=False,
    )

    with pytest.raises(RuntimeError, match="injected placement failure"):
        module.prepare_module("cpu", fsdp_context=context)

    assert shard_calls == [module]
    assert not module._is_fsdp and not module.is_prepared


def test_fsdp_context_keeps_meta_sync_steps_atomic(monkeypatch):
    module = FastGenModule()
    module.projection = torch.nn.Linear(2, 2)
    module._set_precision(torch.bfloat16)
    events = []

    monkeypatch.setattr(fsdp_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(fsdp_utils, "sync_any", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(fsdp_utils, "is_rank0", lambda: True)
    monkeypatch.setattr(
        fsdp_utils,
        "validate_module_sync_source",
        lambda candidate, **kwargs: events.append(("validate_source", candidate, kwargs)),
    )
    monkeypatch.setattr(
        fsdp_utils,
        "_fully_shard_module",
        lambda candidate, **_kwargs: events.append(("shard", candidate)),
    )
    monkeypatch.setattr(
        fsdp_utils,
        "materialize_meta_module",
        lambda candidate, device: events.append(("materialize", candidate, device)),
    )

    @contextlib.contextmanager
    def record_guarded_block(operation):
        events.append(("guarded_block", operation))
        yield

    monkeypatch.setattr(fsdp_utils, "raise_across_ranks", record_guarded_block)
    monkeypatch.setattr(
        fsdp_utils,
        "broadcast_module",
        lambda candidate, **kwargs: events.append(("broadcast_nonpersistent", candidate, kwargs)),
    )

    def record_state_load(candidate, model_state_dict, options):
        events.append(("load", candidate, model_state_dict, options))

    monkeypatch.setattr(fsdp_utils, "set_model_state_dict", record_state_load)
    monkeypatch.setattr(fsdp_utils.torch.cuda, "empty_cache", lambda: events.append(("empty_cache",)))

    context = fsdp_utils.FSDPContext(
        mesh=SimpleNamespace(device_type="cpu"),
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=None,
        cpu_offload=False,
    )
    context._shard_module(module)

    guards = [event[1] for event in events if event[0] == "guarded_block"]
    assert guards == [
        "FSDP source state export",
        "FSDP wrapping",
        "FSDP state synchronization",
        "FSDP state load",
        "FSDP preparation completion",
    ]
    events = [event for event in events if event[0] != "guarded_block"]
    assert [event[0] for event in events] == [
        "validate_source",
        "shard",
        "materialize",
        "broadcast_nonpersistent",
        "load",
        "empty_cache",
    ]
    assert events[0][1:] == (module, {"src": 0, "operation": "FSDP state synchronization"})
    assert events[1][1:] == (module,)
    assert events[2][1:] == (module, torch.device("cpu"))
    assert events[3][1:] == (
        module,
        {
            "parameters": False,
            "persistent_buffers": False,
            "nonpersistent_buffers": True,
            "operation": "FSDP state synchronization",
        },
    )
    assert events[4][1] is module
    assert set(events[4][2]) == set(module.state_dict())
    assert events[4][3].full_state_dict is True
    assert events[4][3].broadcast_from_rank0 is True


def test_move_registered_buffers_preserves_shared_modules_dtype_and_persistence():
    root = torch.nn.Module()
    root.weight = torch.nn.Parameter(torch.ones(2))
    root.left = torch.nn.Module()
    root.right = torch.nn.Module()
    root.left_alias = root.left
    root.left.register_buffer("cache", torch.ones(2, dtype=torch.float64))
    root.right.register_buffer("cache", torch.ones(2, dtype=torch.float64), persistent=False)
    root.right.register_buffer("optional_cache", None, persistent=False)

    fsdp_utils._move_registered_buffers_(root, "meta")

    assert root.left.cache is root.left_alias.cache
    assert root.left.cache is not root.right.cache
    assert root.left.cache.device.type == "meta"
    assert root.left.cache.dtype == torch.float64
    assert root.weight.device.type == "cpu"
    assert "cache" not in root.left._non_persistent_buffers_set
    assert "cache" in root.right._non_persistent_buffers_set
    assert root.right.optional_cache is None


def test_move_registered_buffers_does_not_build_an_autograd_graph():
    module = torch.nn.Module()
    source = torch.ones(2, dtype=torch.float32, requires_grad=True)
    module.register_buffer("differentiable", source)

    fsdp_utils._move_registered_buffers_(module, torch.device("cpu"))
    assert module.differentiable is source

    fsdp_utils._move_registered_buffers_(module, torch.device("meta"))
    assert not module.differentiable.requires_grad
    assert module.differentiable.grad_fn is None


def test_fsdp_context_places_cpu_offloaded_buffers_after_state_load(monkeypatch):
    module = FastGenModule()
    module.projection = torch.nn.Linear(2, 2)
    module.register_buffer("persistent", torch.ones(2))
    module.register_buffer("runtime", torch.ones(2), persistent=False)
    module._set_precision(torch.bfloat16)
    module._set_device("cuda:0")
    events = []

    monkeypatch.setattr(fsdp_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(fsdp_utils.torch.cuda, "empty_cache", lambda: events.append(("empty_cache",)))
    monkeypatch.setattr(fsdp_utils, "sync_any", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(fsdp_utils, "is_rank0", lambda: True)
    monkeypatch.setattr(fsdp_utils, "validate_module_sync_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        fsdp_utils,
        "_fully_shard_module",
        lambda candidate, **_kwargs: events.append(("shard", candidate)),
    )
    monkeypatch.setattr(
        fsdp_utils,
        "materialize_meta_module",
        lambda candidate, device: events.append(("materialize", candidate, device)),
    )
    monkeypatch.setattr(
        fsdp_utils,
        "_move_registered_buffers_",
        lambda candidate, device: events.append(("move_buffers", candidate, device)),
    )

    @contextlib.contextmanager
    def record_guarded_block(operation):
        events.append(("guarded_block", operation))
        yield

    monkeypatch.setattr(fsdp_utils, "raise_across_ranks", record_guarded_block)
    monkeypatch.setattr(
        fsdp_utils,
        "broadcast_module",
        lambda candidate, **_kwargs: events.append(("broadcast_nonpersistent", candidate)),
    )
    monkeypatch.setattr(
        fsdp_utils,
        "set_model_state_dict",
        lambda candidate, model_state_dict, options: events.append(("load", candidate, model_state_dict, options)),
    )

    context = fsdp_utils.FSDPContext(
        mesh=SimpleNamespace(device_type="cuda"),
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=object(),
        cpu_offload=True,
    )
    context._shard_module(module)

    guards = [event[1] for event in events if event[0] == "guarded_block"]
    assert guards == [
        "FSDP source state export",
        "FSDP wrapping",
        "FSDP state synchronization",
        "FSDP state load",
        "FSDP preparation completion",
    ]
    events = [event for event in events if event[0] != "guarded_block"]
    assert [event[0] for event in events] == [
        "shard",
        "materialize",
        "broadcast_nonpersistent",
        "load",
        "move_buffers",
        "empty_cache",
    ]
    assert events[3][0] == "load"
    assert events[4][1:] == (module, torch.device("cuda:0"))


def _fsdp_rejects_meta_sync_source_impl(rank: int, world_size: int) -> dict:
    context = torch.device("meta") if rank == 0 else contextlib.nullcontext()
    with context:
        module = FastGenModule()
        module.projection = torch.nn.Linear(2, 2)

    fsdp_context = fsdp_utils.FSDPContext(
        mesh=SimpleNamespace(device_type="cuda", mesh=torch.arange(world_size), mesh_dim_names=None),
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=None,
        cpu_offload=False,
    )
    try:
        fsdp_context._shard_module(module)
    except RuntimeError as error:
        rejected = "Rank 0 must hold the complete real state" in str(error)
    else:
        rejected = False

    result = torch.tensor(int(rejected), device=torch.device("cuda", rank))
    dist.all_reduce(result, op=dist.ReduceOp.MIN)
    return {"rejected_collectively": bool(result.item())}


@RunIf(min_gpus=2)
def test_fsdp_rejects_meta_sync_source_before_sharding():
    result = run_distributed_test(test_fn=_fsdp_rejects_meta_sync_source_impl, world_size=2)
    assert result is not None
    assert result["rejected_collectively"]


def _fsdp_rejects_state_hidden_from_state_dict_impl(rank: int, world_size: int, detached_only: bool) -> dict:
    """The sharded load restores only ``state_dict()`` entries, so meta sync refuses anything it hides."""
    from torch.distributed.device_mesh import init_device_mesh

    context = torch.device("meta") if rank != 0 else contextlib.nullcontext()
    with context:
        module = FastGenModule()
        module.projection = torch.nn.Linear(2, 2)
        module.hidden = torch.nn.Parameter(torch.ones(2))

    def hide(_module, state_dict, prefix, _local_metadata):
        if not detached_only or state_dict[prefix + "hidden"] is not _module.hidden:
            del state_dict[prefix + "hidden"]

    def ignore_missing_hidden(_module, incompatible_keys):
        incompatible_keys.missing_keys.remove("hidden")

    # Kept out of checkpoints on both sides, so even PyTorch's strict load would accept the gap.
    module._register_state_dict_hook(hide)
    module.register_load_state_dict_post_hook(ignore_missing_hidden)
    fsdp_context = fsdp_utils.FSDPContext(
        mesh=init_device_mesh("cuda", (world_size,)),
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=None,
        cpu_offload=False,
    )
    try:
        fsdp_context._shard_module(module)
    except RuntimeError as error:
        rejected = "does not export 1 registered tensor(s) of FastGenModule; first 8: ['hidden']" in str(error)
    else:
        rejected = False

    result = torch.tensor(int(rejected), device=torch.device("cuda", rank))
    dist.all_reduce(result, op=dist.ReduceOp.MIN)
    return {"rejected_collectively": bool(result.item())}


@RunIf(min_gpus=2)
@pytest.mark.parametrize("detached_only", [False, True])
def test_fsdp_rejects_meta_sync_of_state_hidden_from_state_dict(detached_only):
    result = run_distributed_test(
        test_fn=_fsdp_rejects_state_hidden_from_state_dict_impl, world_size=2, detached_only=detached_only
    )
    assert result is not None
    assert result["rejected_collectively"]


def _shared_preparation_module(kind):
    module = FastGenModule()
    module.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    module.left = torch.nn.Module()
    if kind in ("shared_container", "different_parents"):
        module.left.child = torch.nn.Module()
        owner = module.left.child
    else:
        owner = module.left
    if kind in ("parameter", "shared_container", "different_parents", "tied_parameter"):
        owner.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    elif kind in ("persistent_buffer", "nonpersistent_buffer"):
        owner.register_buffer("cache", torch.arange(8, dtype=torch.float32), persistent=kind == "persistent_buffer")
    else:
        owner.register_parameter("optional_weight", None)
        owner.register_buffer("optional_cache", None)
    if kind == "tied_parameter":
        module.right = torch.nn.Module()
        module.right.weight = module.left.weight
    elif kind == "different_parents":
        module.right = torch.nn.Module()
        module.right.child = module.left.child
    else:
        module.right = module.left
    return module


def _fsdp_shared_preparation_worker(rank, world_size, cpu_offload):
    device = torch.device("cuda", rank)
    context = fsdp_utils.FSDPContext(
        mesh=init_device_mesh("cuda", (world_size,)),
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=fsdp_utils.CPUOffloadPolicy() if cpu_offload else None,
        cpu_offload=cpu_offload,
    )
    for preparation in ("fsdp_meta", "fsdp_real", "eager_meta"):
        for kind in (
            "parameter",
            "persistent_buffer",
            "shared_container",
            "different_parents",
            "nonpersistent_buffer",
            "stateless",
            "tied_parameter",
        ):
            source_device = "meta" if rank and preparation != "fsdp_real" else "cpu"
            with torch.device(source_device):
                module = _shared_preparation_module(kind)
            rejected = preparation == "fsdp_meta" and kind in (
                "parameter",
                "persistent_buffer",
                "shared_container",
                "different_parents",
            )
            kwargs = dict(precision_model=torch.float32, fsdp_context=None if preparation == "eager_meta" else context)
            with patch.object(fsdp_utils, "_fully_shard_module", wraps=fsdp_utils._fully_shard_module) as shard:
                if rejected:
                    with pytest.raises(RuntimeError, match="FSDP meta-state synchronization.*shared submodules"):
                        module.prepare_module(device, **kwargs)
                    assert not shard.called and not module.is_prepared and not module._is_fsdp
                    assert all(tensor.device.type == source_device for tensor in module.parameters())
                    assert all(tensor.device.type == source_device for tensor in module.buffers())
                else:
                    module.prepare_module(device, **kwargs)
                    assert module.is_prepared
                    for tensor in (*module.parameters(), *module.buffers()):
                        full = tensor.full_tensor() if isinstance(tensor, torch.distributed.tensor.DTensor) else tensor
                        torch.testing.assert_close(full.cpu(), torch.arange(8, dtype=torch.float32), rtol=0, atol=0)
            if kind == "tied_parameter":
                assert module.left is not module.right and module.left.weight is module.right.weight
            elif kind == "different_parents":
                assert module.left.child is module.right.child
            else:
                assert module.left is module.right
    return True


@RunIf(min_gpus=2)
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_fsdp_meta_sync_rejects_shared_persistent_modules_before_wrapping(cpu_offload):
    assert run_distributed_test(
        _fsdp_shared_preparation_worker, world_size=2, timeout=180, nccl_timeout=30, cpu_offload=cpu_offload
    )


class _GainedChild(torch.nn.Module):
    def __init__(self, child):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.linspace(0.5, 1.5, 8))
        self.child = child

    def forward(self, h):
        return self.child(h) * self.gain


class _SharedTrainingNet(FastGenModule):
    """Calls one ``Linear`` through two module paths, or ties one weight across two modules."""

    def __init__(self, kind):
        super().__init__()
        self.inp = torch.nn.Linear(8, 8)
        shared = torch.nn.Linear(8, 8)
        if kind == "shared_module":
            self.left = self.right = shared
        elif kind == "shared_under_wrapped_parents":
            # Parents with their own parameters get FSDP groups above the shared module's group.
            self.left, self.right = _GainedChild(shared), _GainedChild(shared)
        else:
            self.left, self.right = shared, torch.nn.Linear(8, 8)
            self.right.weight = shared.weight
        self.out = torch.nn.Linear(8, 1)

    def forward(self, x):
        h = torch.relu(self.inp(x))
        return self.out(self.left(h) + self.right(torch.tanh(h))).squeeze(-1)


def _fsdp_shared_training_worker(rank, world_size, kind, cpu_offload, checkpoint_dir):
    from fastgen.configs.config import BaseCheckpointerConfig
    from fastgen.utils.checkpointer import Checkpointer

    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", rank)
    generator = torch.Generator().manual_seed(0)
    batches = [
        (torch.randn(4 * world_size, 8, generator=generator), torch.randn(4 * world_size, generator=generator))
        for _ in range(6)
    ]

    def full(tensor):
        tensor = tensor.full_tensor() if isinstance(tensor, torch.distributed.tensor.DTensor) else tensor
        return tensor.detach().cpu()

    def train(module, optimizer, steps, shard):
        first_grads = None
        for x, y in steps:
            if shard:
                x, y = x.chunk(world_size)[rank], y.chunk(world_size)[rank]
            torch.nn.functional.mse_loss(module(x.to(device)), y.to(device)).backward()
            if first_grads is None:
                first_grads = {name: full(parameter.grad) for name, parameter in module.named_parameters()}
            torch.nn.utils.clip_grad_norm_(module.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
        return first_grads

    def prepared(seed):
        torch.manual_seed(seed)
        module = _SharedTrainingNet(kind)
        context = fsdp_utils.FSDPContext(
            mesh=init_device_mesh("cuda", (world_size,)),
            min_num_params=1,
            reduce_dtype=torch.float32,
            offload_policy=fsdp_utils.CPUOffloadPolicy() if cpu_offload else None,
            cpu_offload=cpu_offload,
        )
        module.prepare_module(device, precision_model=torch.float32, trainable=True, fsdp_context=context)
        return module, torch.optim.AdamW(module.parameters(), lr=1e-2)

    # The reference trains the same initialization on the global batch in one process.
    torch.manual_seed(0)
    reference = _SharedTrainingNet(kind).to(device)
    reference_grads = train(reference, torch.optim.AdamW(reference.parameters(), lr=1e-2), batches, shard=False)

    config = BaseCheckpointerConfig()
    config.save_dir, config.use_s3 = checkpoint_dir, False
    checkpointer = Checkpointer(config, save_sharded=True)
    module, optimizer = prepared(seed=0)
    grads = train(module, optimizer, batches[:3], shard=True)
    path = checkpointer.save(torch.nn.ModuleDict({"net": module}), {"net": optimizer}, iteration=3)
    train(module, optimizer, batches[3:], shard=True)

    # A differently initialized model must continue the run exactly from the sharded checkpoint.
    resumed, resumed_optimizer = prepared(seed=1)
    iteration = checkpointer.load(
        torch.nn.ModuleDict({"net": resumed}), {"net": resumed_optimizer}, path=path, device=device
    )
    train(resumed, resumed_optimizer, batches[3:], shard=True)

    assert iteration == 3
    assert grads.keys() == reference_grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grad, reference_grads[name], rtol=1e-5, atol=1e-6)
    final = {key: full(value) for key, value in module.state_dict().items()}
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(final[key], value.cpu(), rtol=1e-5, atol=1e-6)
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(full(value), final[key], rtol=0, atol=0)
    return True


@RunIf(min_gpus=2)
@pytest.mark.parametrize("cpu_offload", [False, True], ids=["gpu", "offload"])
@pytest.mark.parametrize("kind", ["shared_module", "shared_under_wrapped_parents", "tied_parameter"])
def test_fsdp_trains_and_resumes_shared_submodules_with_real_state(kind, cpu_offload, tmp_path):
    assert run_distributed_test(
        _fsdp_shared_training_worker,
        world_size=2,
        timeout=300,
        nccl_timeout=60,
        kind=kind,
        cpu_offload=cpu_offload,
        checkpoint_dir=str(tmp_path),
    )


class _FSDPSetterStub(fsdp_utils.FSDPModule, torch.nn.Module):
    """Exercise FSDPModule's real flag setters without initializing distributed."""

    def __new__(cls):
        # FSDPModule.__new__ normally unwraps its dynamically mixed-in class.
        return object.__new__(cls)

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.param_group = SimpleNamespace(reduce_grads=True, all_reduce_grads=True)
        self.state_ctx = SimpleNamespace(is_last_backward=True)
        self.fsdp_state = SimpleNamespace(
            _fsdp_param_group=self.param_group,
            _fsdp_param_groups=[self.param_group],
            _state_ctx=self.state_ctx,
        )
        self.sync_calls = []
        self.last_backward_calls = []

    def _get_fsdp_state(self):
        return self.fsdp_state

    def set_requires_gradient_sync(self, requires_gradient_sync, recurse=True):
        self.sync_calls.append((requires_gradient_sync, recurse))
        return super().set_requires_gradient_sync(requires_gradient_sync, recurse=recurse)

    def set_is_last_backward(self, is_last_backward):
        self.last_backward_calls.append(is_last_backward)
        return super().set_is_last_backward(is_last_backward)


def test_fsdp_sync_grad_is_idempotent_for_overlapping_roots():
    shared_fsdp_module = _FSDPSetterStub()
    parent = torch.nn.Module()
    parent.add_module("shared", shared_fsdp_module)
    model = SimpleNamespace(module_dict={"parent": parent, "shared": shared_fsdp_module})

    # ``parent.modules()`` and ``shared.modules()`` both yield the same FSDP
    # module. Repeating PyTorch's flag assignments must leave one coherent state.
    with fsdp_utils.fsdp_sync_grad(model, enabled=False):
        assert shared_fsdp_module.param_group.reduce_grads is False
        assert shared_fsdp_module.param_group.all_reduce_grads is False
        assert shared_fsdp_module.state_ctx.is_last_backward is False
        assert shared_fsdp_module.sync_calls == [(False, False), (False, False)]
        assert shared_fsdp_module.last_backward_calls == [False, False]

    assert shared_fsdp_module.param_group.reduce_grads is True
    assert shared_fsdp_module.param_group.all_reduce_grads is True
    assert shared_fsdp_module.state_ctx.is_last_backward is True
    assert shared_fsdp_module.sync_calls == [
        (False, False),
        (False, False),
        (True, False),
        (True, False),
    ]
    assert shared_fsdp_module.last_backward_calls == [False, False, True, True]


def test_fsdp_sync_grad_toggles_frozen_fsdp_modules():
    """Frozen groups are toggled too, so roles flipped inside the step never meet stale flags."""
    frozen_fsdp_module = _FSDPSetterStub()
    frozen_fsdp_module.weight.requires_grad_(False)
    model = SimpleNamespace(module_dict={"teacher": frozen_fsdp_module})

    with fsdp_utils.fsdp_sync_grad(model, enabled=False):
        assert frozen_fsdp_module.param_group.reduce_grads is False
        assert frozen_fsdp_module.state_ctx.is_last_backward is False

    assert frozen_fsdp_module.param_group.reduce_grads is True
    assert frozen_fsdp_module.state_ctx.is_last_backward is True
    assert frozen_fsdp_module.sync_calls == [(False, False), (True, False)]
    assert frozen_fsdp_module.last_backward_calls == [False, True]


class _CollectiveCounter(TorchDispatchMode):
    """Count dispatched collectives by operator name, without CommDebugMode's module tracking."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.name in str(func):
            self.count += 1
        return func(*args, **(kwargs or {}))


def _test_trainer_fsdp_gradient_accumulation_impl(rank: int, world_size: int) -> Dict:
    """Exercise Trainer.train_step with an unwrapped FastGen root and nested FSDP groups."""
    from fastgen.methods.model import FastGenModel
    from fastgen.trainer import Trainer

    class NestedBody(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Linear(8, 8)
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)])

        def forward(self, x):
            x = self.input(x)
            for block in self.blocks:
                x = torch.tanh(block(x))
            return x

    class NestedFSDPNet(FastGenModule):
        def __init__(self):
            super().__init__()
            self.body = NestedBody()
            # This test uses a FastGenModule as the model's network root, so provide the
            # normal FastGenNetwork preprocessor registry expected by model validation.
            self.preprocessors = {}

        def forward(self, x):
            return self.body(x)

        def fully_shard(self, **kwargs):
            for block in self.body.blocks:
                fully_shard(block, **kwargs, reshard_after_forward=True)
            fully_shard(self.body, **kwargs)

    class ToyModel(FastGenModel):
        @property
        def model_dict(self):
            return torch.nn.ModuleDict({"net": self.net})

        def _get_outputs(self, *args, **kwargs):
            raise NotImplementedError

        def single_train_step(self, data, iteration):
            del iteration
            prediction = self.net(data["x"])
            return {"total_loss": prediction.square().mean()}, {"prediction": prediction}

        def optimizers_schedulers_step(self, iteration):
            self.optimizer_step_iterations.append(iteration)

        def optimizers_zero_grad(self, iteration):
            self.zero_grad_iterations.append(iteration)

    class IdentityGradScaler:
        @staticmethod
        def scale(loss):
            return loss

    class NoOpCallbacks:
        @staticmethod
        def on_backward_begin(*args, **kwargs):
            pass

        @staticmethod
        def on_optimizer_step_begin(*args, **kwargs):
            pass

    model = ToyModel.__new__(ToyModel)
    torch.nn.Module.__init__(model)
    model.module_dict = {}
    model.config = SimpleNamespace(
        inference=False,
        enable_preprocessors=False,
        extra_fsdp_modules=[],
        net_adapter=SimpleNamespace(type=None),
        autocast_cache_enabled=False,
    )
    model.device = torch.device("cuda", rank)
    model._resolve_precisions(
        precision="float32",
        precision_model="float32",
        precision_opt="float32",
        precision_fsdp_reduce="float32",
    )
    model.use_ema = []
    torch.manual_seed(1234)
    model.net = NestedFSDPNet()
    model.module_dict["net"] = model.net
    model.grad_scaler = IdentityGradScaler()
    model.optimizer_step_iterations = []
    model.zero_grad_iterations = []

    fsdp_context = fsdp_utils.create_fsdp_context(reduce_dtype=model.precision_fsdp_reduce, min_num_params=1)
    model.on_train_begin(fsdp_context=fsdp_context)

    wrapped_modules = [module for module in model.net.modules() if isinstance(module, fsdp_utils.FSDPModule)]
    root_is_unwrapped = not isinstance(model.net, fsdp_utils.FSDPModule)
    nested_topology_is_wrapped = (
        len(wrapped_modules) == 3
        and isinstance(model.net.body, fsdp_utils.FSDPModule)
        and all(isinstance(block, fsdp_utils.FSDPModule) for block in model.net.body.blocks)
    )

    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(fsdp=True, grad_accum_rounds=2))
    trainer.callbacks = NoOpCallbacks()

    def make_data(accum_iter):
        generator = torch.Generator(device=model.device).manual_seed(10_000 + 100 * rank + accum_iter)
        return {"x": torch.randn(4, 8, generator=generator, device=model.device)}

    with _CollectiveCounter("reduce_scatter") as first_microbatch_counter:
        trainer.train_step(model, model, make_data(0), iteration=0, grad_accum_iter=0)
    first_reduce_scatter_count = first_microbatch_counter.count
    no_early_optimizer_step = not model.optimizer_step_iterations and not model.zero_grad_iterations

    with _CollectiveCounter("reduce_scatter") as final_microbatch_counter:
        trainer.train_step(model, model, make_data(1), iteration=0, grad_accum_iter=1)
    final_reduce_scatter_count = final_microbatch_counter.count

    assert root_is_unwrapped
    assert nested_topology_is_wrapped
    assert first_reduce_scatter_count == 0
    assert final_reduce_scatter_count > 0
    assert no_early_optimizer_step
    assert model.optimizer_step_iterations == [0]
    assert model.zero_grad_iterations == [0]

    return {
        "world_size": world_size,
        "wrapped_module_count": len(wrapped_modules),
        "first_reduce_scatter_count": first_reduce_scatter_count,
        "final_reduce_scatter_count": final_reduce_scatter_count,
    }


@RunIf(min_gpus=2)
def test_trainer_fsdp_gradient_accumulation_skips_early_gradient_sync():
    result = run_distributed_test(
        test_fn=_test_trainer_fsdp_gradient_accumulation_impl,
        world_size=2,
        timeout=240,
    )
    assert result == {
        "world_size": 2,
        "wrapped_module_count": 3,
        "first_reduce_scatter_count": 0,
        "final_reduce_scatter_count": 3,
    }


def _test_trainer_fsdp_role_flip_accumulation_impl(rank: int, world_size: int) -> Dict:
    """DMD2-style roles flipped inside ``single_train_step`` while the trainer accumulates gradients.

    Two sharded roots alternate between trainable and frozen per iteration; the frozen one takes part in
    the backward through its input like a frozen score network. Every microbatch except the last must
    skip the reduce-scatter (also right after a flip), frozen roots must never receive gradients, and the
    accumulated gradient must equal one synchronized step over the concatenated microbatches.
    """
    from fastgen.methods.model import FastGenModel
    from fastgen.trainer import Trainer

    # Compare different GEMM batch sizes in full FP32: a container's TF32 default
    # can introduce size-dependent rounding larger than this equivalence check's tolerance.
    torch.set_float32_matmul_precision("highest")

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Linear(8, 8)
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)])

        def forward(self, x):
            x = self.input(x)
            for block in self.blocks:
                x = torch.tanh(block(x))
            return x

    class Root(FastGenModule):
        def __init__(self):
            super().__init__()
            self.body = Body()
            self.preprocessors = {}

        def forward(self, x):
            return self.body(x)

        def fully_shard(self, **kwargs):
            for block in self.body.blocks:
                fully_shard(block, **kwargs, reshard_after_forward=True)
            fully_shard(self.body, **kwargs)

    class TwoRoleModel(FastGenModel):
        @property
        def model_dict(self):
            return torch.nn.ModuleDict({"net": self.net, "aux": self.aux})

        def _get_outputs(self, *args, **kwargs):
            raise NotImplementedError

        def single_train_step(self, data, iteration):
            # Like DMD2's ``_setup_grad_requirements``: the roles are decided inside the step.
            student_turn = iteration % 2 == 0
            self.net.requires_grad_(student_turn)
            self.aux.requires_grad_(not student_turn)
            active, frozen = (self.net, self.aux) if student_turn else (self.aux, self.net)
            prediction = frozen(active(data["x"]))
            return {"total_loss": prediction.square().mean()}, {"prediction": prediction}

        def optimizers_schedulers_step(self, iteration):
            pass

        def optimizers_zero_grad(self, iteration):
            pass

    class IdentityGradScaler:
        @staticmethod
        def scale(loss):
            return loss

    class NoOpCallbacks:
        @staticmethod
        def on_backward_begin(*args, **kwargs):
            pass

        @staticmethod
        def on_optimizer_step_begin(*args, **kwargs):
            pass

    model = TwoRoleModel.__new__(TwoRoleModel)
    torch.nn.Module.__init__(model)
    model.module_dict = {}
    model.config = SimpleNamespace(
        inference=False,
        enable_preprocessors=False,
        extra_fsdp_modules=[],
        net_adapter=SimpleNamespace(type=None),
        autocast_cache_enabled=False,
    )
    model.device = torch.device("cuda", rank)
    model._resolve_precisions(
        precision="float32",
        precision_model="float32",
        precision_opt="float32",
        precision_fsdp_reduce="float32",
    )
    model.use_ema = []
    torch.manual_seed(1234)
    model.net = Root()
    model.aux = Root()
    model.module_dict.update(net=model.net, aux=model.aux)
    model.grad_scaler = IdentityGradScaler()

    fsdp_context = fsdp_utils.create_fsdp_context(reduce_dtype=model.precision_fsdp_reduce, min_num_params=1)
    model.on_train_begin(fsdp_context=fsdp_context)

    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(fsdp=True, grad_accum_rounds=2))
    trainer.callbacks = NoOpCallbacks()

    def make_data(iteration, accum_iter):
        seed = 10_000 + 1_000 * iteration + 100 * rank + accum_iter
        generator = torch.Generator(device=model.device).manual_seed(seed)
        return {"x": torch.randn(4, 8, generator=generator, device=model.device)}

    def local_grads(root):
        grads = []
        for parameter in root.parameters():
            grad = parameter.grad
            if grad is not None:
                grad = (grad.to_local() if hasattr(grad, "to_local") else grad).detach().clone()
            grads.append(grad)
        return grads

    def clear_grads():
        for parameter in model.parameters():
            parameter.grad = None

    counts = {}
    for iteration in (0, 1):
        for accum_iter in (0, 1):
            with _CollectiveCounter("reduce_scatter") as counter:
                trainer.train_step(
                    model, model, make_data(iteration, accum_iter), iteration=iteration, grad_accum_iter=accum_iter
                )
            counts[f"it{iteration}_mb{accum_iter}"] = counter.count
        if iteration == 0:
            net_got_grads = all(parameter.grad is not None for parameter in model.net.parameters())
            aux_stayed_clean = all(parameter.grad is None for parameter in model.aux.parameters())
            clear_grads()

    accumulated = local_grads(model.aux)
    aux_got_grads = all(grad is not None for grad in accumulated)
    net_stayed_clean_after_flip = all(parameter.grad is None for parameter in model.net.parameters())

    # Reference: one synchronized step over both microbatches of iteration 1. The mean over eight samples
    # equals the average of the two four-sample means, so the gradients must agree.
    clear_grads()
    trainer.config.trainer.grad_accum_rounds = 1
    both = {"x": torch.cat([make_data(1, 0)["x"], make_data(1, 1)["x"]])}
    trainer.train_step(model, model, both, iteration=1, grad_accum_iter=0)
    reference = local_grads(model.aux)
    grads_match = all(
        grad is not None and ref is not None and torch.allclose(grad, ref, rtol=1e-5, atol=1e-6)
        for grad, ref in zip(accumulated, reference)
    )

    return {
        "counts": counts,
        "net_got_grads": net_got_grads,
        "aux_stayed_clean": aux_stayed_clean,
        "aux_got_grads": aux_got_grads,
        "net_stayed_clean_after_flip": net_stayed_clean_after_flip,
        "grads_match": grads_match,
    }


@RunIf(min_gpus=2)
def test_trainer_fsdp_gradient_accumulation_with_roles_flipped_inside_the_step():
    result = run_distributed_test(
        test_fn=_test_trainer_fsdp_role_flip_accumulation_impl,
        world_size=2,
        timeout=240,
    )
    assert result is not None, "distributed test returned nothing (a rank likely crashed)"
    counts = result["counts"]
    # No reduce-scatter on any non-final microbatch, including the first one after the role flip.
    assert counts["it0_mb0"] == 0 and counts["it1_mb0"] == 0, counts
    assert counts["it0_mb1"] > 0 and counts["it1_mb1"] == counts["it0_mb1"], counts
    assert result["net_got_grads"] and result["aux_stayed_clean"]
    assert result["aux_got_grads"] and result["net_stayed_clean_after_flip"]
    assert result["grads_match"]


def test_fsdp_dispatch_uses_fastgen_module_owned_boundaries(monkeypatch):
    """Any FastGen module with a sharding hook owns its component boundaries."""

    generic_shard_calls = []
    monkeypatch.setattr(
        fsdp_utils,
        "fully_shard",
        lambda module, **kwargs: generic_shard_calls.append((module, kwargs)),
    )

    class ComponentShardedModule(FastGenModule):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2)
            self.custom_shard_kwargs = None

        def fully_shard(self, **kwargs):
            self.custom_shard_kwargs = kwargs
            fsdp_utils.fully_shard(self.projection, **kwargs)

    module = ComponentShardedModule()
    fsdp_kwargs = {"mesh": object(), "mp_policy": object(), "offload_policy": object()}

    fsdp_utils._fully_shard_module(module, min_num_params=0, **fsdp_kwargs)

    assert module.custom_shard_kwargs == fsdp_kwargs
    assert generic_shard_calls == [(module.projection, fsdp_kwargs)]
    assert all(sharded_module is not module for sharded_module, _ in generic_shard_calls)


def test_fsdp_dispatch_generically_shards_fastgen_module_without_hook(monkeypatch):
    """A FastGen module without the capability still uses generic traversal."""

    generic_shard_calls = []
    monkeypatch.setattr(
        fsdp_utils,
        "fully_shard",
        lambda module, **kwargs: generic_shard_calls.append((module, kwargs)),
    )
    module = FastGenModule()
    module.projection = torch.nn.Linear(2, 2)

    fsdp_utils._fully_shard_module(module, min_num_params=10**9, mesh=object())

    assert len(generic_shard_calls) == 1
    sharded_module, kwargs = generic_shard_calls[0]
    assert sharded_module is module
    assert "reshard_after_forward" not in kwargs


@pytest.mark.parametrize("with_policy", [False, True], ids=["inherit-kwargs", "derive-param-dtype"])
def test_generic_fsdp_shards_a_parameterized_nested_precision_island_first(monkeypatch, with_policy):
    root = FastGenModule()
    root.island = FastGenModule()
    root.island.projection = torch.nn.Linear(2, 2)
    root.tail = torch.nn.Linear(2, 2)
    root._record_precision(torch.bfloat16, torch.bfloat16)
    root.island._record_precision(torch.float32, torch.float32)
    fsdp_kwargs = {"mesh": "mesh"}
    if with_policy:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    shard_calls = []
    monkeypatch.setattr(fsdp_utils, "fully_shard", _record_fully_shard_in_place(shard_calls))

    fsdp_utils._fully_shard_module(root, min_num_params=10**9, **fsdp_kwargs)

    assert [module for module, _ in shard_calls] == [root.island, root]
    island_kwargs, root_kwargs = shard_calls[0][1], shard_calls[1][1]
    assert island_kwargs["reshard_after_forward"] is True and "reshard_after_forward" not in root_kwargs
    if with_policy:
        assert island_kwargs["mp_policy"].param_dtype is torch.float32
        assert island_kwargs["mp_policy"].reduce_dtype is torch.float32
        assert root_kwargs["mp_policy"] is fsdp_kwargs["mp_policy"]
    else:
        assert island_kwargs == {"mesh": "mesh", "reshard_after_forward": True}


def test_generic_fsdp_allows_same_policy_and_buffer_only_precision_islands(monkeypatch):
    root = FastGenModule()
    root.same_policy = FastGenModule()
    root.same_policy.projection = torch.nn.Linear(2, 2)
    root.buffer_only = FastGenModule()
    root.buffer_only.register_buffer("state", torch.ones(2))
    root._record_precision(torch.bfloat16, torch.bfloat16)
    root.buffer_only._record_precision(torch.float32, torch.float32)
    shard_calls = []
    monkeypatch.setattr(
        fsdp_utils,
        "fully_shard",
        lambda candidate, **kwargs: shard_calls.append((candidate, kwargs)),
    )

    fsdp_utils._fully_shard_module(root, min_num_params=10**9, mesh="mesh")

    assert shard_calls == [(root, {"mesh": "mesh"})]


def test_generic_fsdp_restores_the_root_policy_below_a_different_precision_boundary(monkeypatch):
    root = FastGenModule()
    root.outer = FastGenModule()
    root.outer.weight = torch.nn.Parameter(torch.ones(2))
    root.outer.inner = FastGenModule()
    root.outer.inner.weight = torch.nn.Parameter(torch.ones(2))
    root._record_precision(torch.bfloat16, torch.float32)
    root.outer._record_precision(torch.float32, torch.float32)
    root.outer.inner._record_precision(torch.bfloat16, torch.float32)
    shard_calls = []
    monkeypatch.setattr(fsdp_utils, "fully_shard", _record_fully_shard_in_place(shard_calls))

    fsdp_utils._fully_shard_module(
        root, min_num_params=10**9, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    )

    assert [module for module, _ in shard_calls] == [root.outer.inner, root.outer, root]
    assert [kwargs["mp_policy"].param_dtype for _, kwargs in shard_calls] == [
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
    ]


def test_generic_fsdp_rejects_ties_across_distinct_boundaries_with_the_same_policy(monkeypatch):
    root = FastGenModule()
    root.left, root.right = FastGenModule(), FastGenModule()
    weight = root.left.weight = root.right.weight = torch.nn.Parameter(torch.ones(2))
    root._record_precision(torch.bfloat16, torch.float32)
    root.left._record_precision(torch.float32, torch.float32)
    root.right._record_precision(torch.float32, torch.float32)
    shard_calls = []
    monkeypatch.setattr(fsdp_utils, "fully_shard", lambda module, **kwargs: shard_calls.append(module))

    with pytest.raises(RuntimeError, match="separate precision boundaries") as error:
        fsdp_utils._fully_shard_module(root, min_num_params=1, mesh="mesh")

    assert "left.weight" in str(error.value) and "right.weight" in str(error.value)
    assert root.left.weight is root.right.weight is weight
    assert shard_calls == []


def test_generic_fsdp_keeps_ties_within_one_precision_boundary(monkeypatch):
    root = FastGenModule()
    root.island = FastGenModule()
    root.island.left, root.island.right = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    root.island.right.weight = root.island.left.weight
    root._record_precision(torch.bfloat16, torch.float32)
    root.island._record_precision(torch.float32, torch.float32)
    shard_calls = []
    monkeypatch.setattr(fsdp_utils, "fully_shard", _record_fully_shard_in_place(shard_calls))

    fsdp_utils._fully_shard_module(root, min_num_params=1, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))

    assert [module for module, _ in shard_calls] == [root.island, root]
    assert shard_calls[0][1]["mp_policy"].param_dtype == torch.float32


def test_generic_fsdp_size_groups_inherit_the_enclosing_precision(monkeypatch):
    root = FastGenModule()
    root.island = FastGenModule()
    root.island.projection = torch.nn.Linear(2, 2)
    root._record_precision(torch.bfloat16, torch.float32)
    root.island._record_precision(torch.float32, torch.float32)
    shard_calls = []
    monkeypatch.setattr(fsdp_utils, "fully_shard", _record_fully_shard_in_place(shard_calls))

    fsdp_utils._fully_shard_module(root, min_num_params=1, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))

    assert [module for module, _ in shard_calls] == [root.island.projection, root.island, root]
    assert all(kwargs["mp_policy"].param_dtype == torch.float32 for _, kwargs in shard_calls[:-1])


@pytest.mark.parametrize("alias_first", [False, True])
def test_generic_fsdp_wraps_a_shared_fastgen_boundary_once(monkeypatch, alias_first):
    root, outer, shared = FastGenModule(), FastGenModule(), FastGenModule()
    outer.weight = torch.nn.Parameter(torch.ones(2))
    shared.weight = torch.nn.Parameter(torch.ones(2))
    outer.child = shared
    children = [("outer", outer), ("shared", shared)]
    for name, child in reversed(children) if alias_first else children:
        root.add_module(name, child)
    root._record_precision(torch.bfloat16, torch.float32)
    outer._record_precision(torch.float32, torch.float32)
    shard_calls = []
    monkeypatch.setattr(fsdp_utils, "fully_shard", _record_fully_shard_in_place(shard_calls))

    fsdp_utils._fully_shard_module(
        root, min_num_params=10**9, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    )

    assert [module for module, _ in shard_calls] == [shared, outer, root]
    assert all(kwargs["mp_policy"].param_dtype == torch.float32 for _, kwargs in shard_calls[:-1])


def _generic_fsdp_precision_plan_impl(rank: int, world_size: int) -> dict:
    from torch.distributed.fsdp import FSDPModule

    class Leaf(FastGenModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 4))

        def forward(self, value):
            return value.to(self.precision_model) @ self.weight

    class FP32Leaf(Leaf):
        @property
        def precision_model(self):
            return torch.float32

    class Outer(FP32Leaf):
        def __init__(self):
            super().__init__()
            self.inner = Leaf()

        def forward(self, value):
            return super().forward(self.inner(value))

    class Pair(FastGenModule):
        def __init__(self):
            super().__init__()
            self.left, self.right = torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False)
            torch.nn.init.ones_(self.left.weight)
            self.right.weight = self.left.weight

        @property
        def precision_model(self):
            return torch.float32

        def forward(self, value):
            return self.left(value) + 2 * self.right(value)

    class Root(FastGenModule):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, value):
            return self.child(value)

    device = torch.device("cuda", rank)
    context = fsdp_utils.create_fsdp_context(torch.float32, min_num_params=1)

    def prepare(module):
        module.prepare_module(
            device, precision_model=torch.bfloat16, precision_opt=torch.float32, trainable=True, fsdp_context=context
        )

    nested = Root(Outer())
    prepare(nested)
    optimizer = torch.optim.SGD(nested.parameters(), lr=0.1)
    output = nested(torch.ones(1, 4, device=device))
    torch.testing.assert_close(output, torch.full((1, 4), 16.0, device=device))
    output.sum().backward()
    optimizer.step()
    for parameter in nested.parameters():
        torch.testing.assert_close(parameter.to_local(), torch.full_like(parameter.to_local(), 0.6))
    nested_precision = (
        fully_shard.state(nested.child.inner)._fsdp_param_group.mp_policy.param_dtype == torch.bfloat16
        and fully_shard.state(nested.child)._fsdp_param_group.mp_policy.param_dtype == torch.float32
    )

    shared = Root(Outer())
    shared.child.inner = shared.alias = FP32Leaf()
    prepare(shared)
    optimizer = torch.optim.SGD(shared.parameters(), lr=0.1)
    shared(torch.ones(1, 4, device=device)).sum().backward()
    optimizer.step()
    shared_boundary = shared.child.inner is shared.alias
    for parameter in shared.parameters():
        torch.testing.assert_close(parameter.to_local(), torch.full_like(parameter.to_local(), 0.6))

    tied = Root(Pair())
    prepare(tied)
    assert tied.child.left.weight is tied.child.right.weight
    optimizer = torch.optim.SGD(tied.parameters(), lr=0.1)
    tied(torch.ones(1, 4, device=device)).sum().backward()
    optimizer.step()
    weight = tied.child.left.weight.to_local()
    torch.testing.assert_close(weight, torch.full_like(weight, 0.7))
    ties_preserved = tied.child.left.weight is tied.child.right.weight

    invalid = FastGenModule()
    invalid.left, invalid.right = FP32Leaf(), FP32Leaf()
    weight = invalid.right.weight = invalid.left.weight
    with pytest.raises(RuntimeError, match="separate precision boundaries"):
        prepare(invalid)
    cross_boundary_ties_rejected = (
        invalid.left.weight is invalid.right.weight is weight
        and not invalid.is_prepared
        and not any(isinstance(child, FSDPModule) for child in invalid.modules())
    )
    return {
        "nested_precision": _all_true(nested_precision),
        "shared_boundary": _all_true(shared_boundary),
        "ties_preserved": _all_true(ties_preserved),
        "cross_boundary_ties_rejected": _all_true(cross_boundary_ties_rejected),
    }


@RunIf(min_gpus=2)
def test_generic_fsdp_precision_plan_preserves_forward_gradients_and_ties():
    result = run_distributed_test(
        _generic_fsdp_precision_plan_impl, world_size=2, timeout=180, nccl_timeout=60, setup_fn=set_env_vars
    )
    assert result is not None and all(result.values()), result


def test_generic_fsdp_rejects_tied_parameter_across_precision_islands(monkeypatch):
    root = FastGenModule()
    root.weight = torch.nn.Parameter(torch.ones(2))
    root.island = FastGenModule()
    root.island.register_parameter("tied_weight", root.weight)
    root._record_precision(torch.bfloat16, torch.bfloat16)
    root.island._record_precision(torch.float32, torch.float32)
    shard_calls = []
    monkeypatch.setattr(
        fsdp_utils,
        "fully_shard",
        lambda candidate, **kwargs: shard_calls.append((candidate, kwargs)),
    )

    with pytest.raises(RuntimeError, match="architecture-owned fully_shard") as error:
        fsdp_utils._fully_shard_module(root, min_num_params=1, mesh="mesh")

    assert "weight" in str(error.value)
    assert "island.tied_weight" in str(error.value)
    assert shard_calls == []


def test_generic_fsdp_rejects_shared_module_reached_through_different_precision_islands(monkeypatch):
    root = FastGenModule()
    shared = torch.nn.Linear(2, 2)
    root.shared = shared
    root.island = FastGenModule()
    root.island.shared = shared
    root._record_precision(torch.bfloat16, torch.bfloat16)
    root.island._record_precision(torch.float32, torch.float32)
    shard_calls = []
    monkeypatch.setattr(
        fsdp_utils,
        "fully_shard",
        lambda candidate, **kwargs: shard_calls.append((candidate, kwargs)),
    )

    with pytest.raises(RuntimeError, match="architecture-owned fully_shard") as error:
        fsdp_utils._fully_shard_module(root, min_num_params=1, mesh="mesh")

    assert "shared.weight" in str(error.value)
    assert "island.shared.weight" in str(error.value)
    assert shard_calls == []


def _fake_fsdp_groups(monkeypatch):
    """Install FSDP2 doubles for ``_validate_fsdp_pins``.

    Returns ``(FakeFSDPModule, groups)``; ``groups`` maps a fake module to
    ``(param_dtype, [(holder, param_name, storage_dtype), ...])``.
    """
    groups = {}

    class FakeFSDPModule(torch.nn.Module):
        pass

    def state(candidate):
        param_dtype, members = groups[candidate]
        fsdp_params = [
            SimpleNamespace(
                _module_info=SimpleNamespace(module=holder, param_name=name, shared_modules=[], shared_param_names=[]),
                sharded_param=SimpleNamespace(dtype=storage_dtype),
            )
            for holder, name, storage_dtype in members
        ]
        group = SimpleNamespace(mp_policy=SimpleNamespace(param_dtype=param_dtype), fsdp_params=fsdp_params)
        return SimpleNamespace(_fsdp_param_group=group)

    monkeypatch.setattr(fsdp_utils, "FSDPModule", FakeFSDPModule)
    monkeypatch.setattr(fsdp_utils, "fully_shard", SimpleNamespace(state=state))
    return FakeFSDPModule, groups


class _PinnedRoot(FastGenModule):
    def __init__(self):
        super().__init__()
        self.pinned = torch.nn.Linear(2, 2)
        self.plain = torch.nn.Linear(2, 2)
        self.register_buffer("table", torch.ones(2))

    def _fixed_precision_overrides(self):
        return {"pinned.weight": torch.float32, "pinned.bias": torch.float32, "table": torch.float32}


@pytest.mark.parametrize(
    ("param_dtype", "storage_dtype", "managed", "error"),
    [
        pytest.param(torch.float32, torch.float32, True, None, id="carved-fp32-group"),
        pytest.param(None, torch.float32, True, None, id="compute-in-storage-dtype"),
        pytest.param(
            torch.bfloat16, torch.float32, True, "computes in torch.bfloat16, pinned torch.float32", id="root-policy"
        ),
        pytest.param(torch.float32, torch.float32, False, "not managed by any FSDP group", id="unsharded"),
    ],
)
def test_validate_fsdp_pins(monkeypatch, param_dtype, storage_dtype, managed, error):
    FakeFSDPModule, groups = _fake_fsdp_groups(monkeypatch)
    root = _PinnedRoot()
    root.group = FakeFSDPModule()
    members = [(root.plain, "weight", torch.bfloat16), (root.plain, "bias", torch.bfloat16)]
    if managed:
        members += [(root.pinned, "weight", storage_dtype), (root.pinned, "bias", storage_dtype)]
    groups[root.group] = (param_dtype, members)

    if error is None:
        fsdp_utils._validate_fsdp_pins(root)
        return
    with pytest.raises(RuntimeError, match=error) as excinfo:
        fsdp_utils._validate_fsdp_pins(root)
    # Buffers are never cast by FSDP, so the pinned buffer is not reported.
    assert "pinned.weight" in str(excinfo.value) and "table" not in str(excinfo.value)


def test_shard_module_validates_pins_after_wrapping(monkeypatch):
    calls = []
    monkeypatch.setattr(fsdp_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(fsdp_utils, "sync_any", lambda value, **_kwargs: False)
    monkeypatch.setattr(fsdp_utils, "_fully_shard_module", lambda candidate, **_kwargs: calls.append("shard"))
    monkeypatch.setattr(fsdp_utils, "_validate_fsdp_pins", lambda candidate: calls.append("pins"))
    module = FastGenModule()
    module.projection = torch.nn.Linear(2, 2)
    module._set_precision(precision_model=torch.bfloat16, precision_opt=torch.bfloat16)
    context = fsdp_utils.FSDPContext(
        mesh=SimpleNamespace(device_type="cpu"),
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=None,
        cpu_offload=False,
    )

    context._shard_module(module)

    assert calls == ["shard", "pins"]


def test_fsdp_dispatch_rejects_an_unmarked_prewrapped_exact_root(monkeypatch):
    class FakeFSDPModule(torch.nn.Module):
        pass

    module = FakeFSDPModule()
    shard_calls = []
    monkeypatch.setattr(fsdp_utils, "FSDPModule", FakeFSDPModule)
    monkeypatch.setattr(
        fsdp_utils,
        "fully_shard",
        lambda candidate, **kwargs: shard_calls.append((candidate, kwargs)),
    )

    with pytest.raises(RuntimeError, match="exact root is already an FSDPModule"):
        fsdp_utils._fully_shard_module(module, min_num_params=1, mesh="mesh")

    assert shard_calls == []


def test_generic_fsdp_allows_a_presharded_descendant(monkeypatch):
    class FakeFSDPModule(FastGenModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.grandchild = torch.nn.Linear(2, 2)

    root = FastGenModule()
    root.weight = torch.nn.Parameter(torch.ones(2))
    child = FakeFSDPModule()
    root.add_module("child", child)
    root._record_precision(torch.bfloat16, torch.bfloat16)
    child._record_precision(torch.float32, torch.float32)
    shard_calls = []
    monkeypatch.setattr(fsdp_utils, "FSDPModule", FakeFSDPModule)
    monkeypatch.setattr(
        fsdp_utils,
        "fully_shard",
        lambda candidate, **kwargs: shard_calls.append((candidate, kwargs)),
    )

    fsdp_utils._fully_shard_module(root, min_num_params=1, mesh="mesh")

    assert shard_calls == [(root, {"mesh": "mesh"})]


def _generic_fsdp_presharded_descendant_impl(rank: int, world_size: int) -> dict:
    """Compose a generic outer root with a real, already-sharded child."""
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    class Child(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.grandchild = torch.nn.Linear(4, 4, bias=False)

        def forward(self, value):
            return self.grandchild(value)

    class Root(FastGenModule):
        def __init__(self):
            super().__init__()
            self.child = Child()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))

        def forward(self, value):
            return torch.nn.functional.linear(self.child(value), self.weight)

    torch.manual_seed(1234)
    device = torch.device("cuda", rank)
    root = Root().to(device)
    reference = copy.deepcopy(root)
    inputs = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4) / 8
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_output = reference(reference_inputs)
    reference_output.square().mean().backward()

    mesh = init_device_mesh("cuda", (world_size,))
    child = root.child
    grandchild = child.grandchild
    fully_shard(child, mesh=mesh, reshard_after_forward=True)
    child_wrapped_before_outer = isinstance(child, FSDPModule)
    grandchild_unwrapped_before_outer = not isinstance(grandchild, FSDPModule)
    grandchild_weight = grandchild.weight

    fsdp_utils._fully_shard_module(root, min_num_params=1, mesh=mesh)
    wrapped_paths = {name for name, module in root.named_modules() if isinstance(module, FSDPModule)}
    all_parameters_sharded = all(isinstance(parameter, DTensor) for parameter in root.parameters())

    inputs = inputs.detach().clone().requires_grad_()
    output = root(inputs)
    output.square().mean().backward()
    local_gradients = [
        parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        for parameter in root.parameters()
    ]

    local_result = {
        "child_identity_preserved": root.child is child and child.grandchild is grandchild,
        "child_wrapped_before_outer": child_wrapped_before_outer,
        "grandchild_unwrapped_before_outer": grandchild_unwrapped_before_outer,
        "grandchild_weight_preserved": grandchild.weight is grandchild_weight,
        "wrapped_paths": sorted(wrapped_paths),
        "all_parameters_sharded": all_parameters_sharded,
        "all_gradients_finite": all(
            gradient is not None and torch.isfinite(gradient).all().item() for gradient in local_gradients
        ),
        "all_gradients_nonzero": all(
            gradient is not None and gradient.abs().sum().item() > 0 for gradient in local_gradients
        ),
        "output_error": float((output - reference_output).abs().max().item()),
        "input_grad_error": float((inputs.grad - reference_inputs.grad).abs().max().item()),
    }
    per_rank = [None] * world_size
    dist.all_gather_object(per_rank, local_result)
    return {"per_rank": per_rank}


@RunIf(min_gpus=2)
def test_generic_fsdp_composes_with_a_real_presharded_descendant():
    """The generic outer traversal must leave an existing FSDP subtree intact."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_generic_fsdp_presharded_descendant_impl,
        world_size=2,
        timeout=180,
        nccl_timeout=120,
        setup_fn=set_env_vars,
    )
    assert result is not None
    for rank, entry in enumerate(result["per_rank"]):
        assert entry["child_identity_preserved"], (rank, entry)
        assert entry["child_wrapped_before_outer"], (rank, entry)
        assert entry["grandchild_unwrapped_before_outer"], (rank, entry)
        assert entry["grandchild_weight_preserved"], (rank, entry)
        assert entry["wrapped_paths"] == ["", "child"], (rank, entry)
        assert entry["all_parameters_sharded"], (rank, entry)
        assert entry["all_gradients_finite"], (rank, entry)
        assert entry["all_gradients_nonzero"], (rank, entry)
        assert entry["output_error"] < 1e-6, (rank, entry)
        assert entry["input_grad_error"] < 1e-6, (rank, entry)
    clear_gpu_memory()


def _record_fully_shard_in_place(shard_calls):
    """Return a test double that mirrors fully_shard's in-place FSDPModule mixin."""

    def record(module, **kwargs):
        shard_calls.append((module, kwargs))
        module_type = type(module)
        module.__class__ = type(f"FSDP{module_type.__name__}", (fsdp_utils.FSDPModule, module_type), {})

    return record


def test_generic_fsdp_shards_bottom_up_and_root_once(monkeypatch):
    class ParameterModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))

    root = ParameterModule()
    child = ParameterModule()
    grandchild = ParameterModule()
    root.add_module("child", child)
    child.add_module("grandchild", grandchild)

    shard_calls = []
    monkeypatch.setattr(fsdp_utils, "fully_shard", _record_fully_shard_in_place(shard_calls))

    fsdp_utils._fully_shard_module(root, min_num_params=1, mesh=object())

    assert [module for module, _ in shard_calls] == [grandchild, child, root]
    assert all(kwargs["reshard_after_forward"] is True for _, kwargs in shard_calls[:-1])
    assert "reshard_after_forward" not in shard_calls[-1][1]


def _preprocess_boundary_reshard_impl(rank: int, world_size: int) -> dict:
    from torch.distributed.tensor import DTensor

    class TinyPreprocessor(FastGenModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16)
            self.saw_unsharded_cuda = []

        def encode(self, value):
            return self(value)

        def forward(self, value):
            self.saw_unsharded_cuda.append(not isinstance(self.weight, DTensor) and self.weight.device.type == "cuda")
            return value @ self.weight

    class TinyNet:
        def __init__(self, text_encoder):
            self.preprocessors = {"text_encoder": text_encoder}
            self.retained_between_encodes = False

        def build_conditioning(self, data, input_shape=None, precision=None):
            del input_shape
            condition = self.preprocessors["text_encoder"].encode(data["condition"])
            self.retained_between_encodes = not isinstance(self.preprocessors["text_encoder"].weight, DTensor)
            neg_condition = self.preprocessors["text_encoder"].encode(data["neg_condition"])
            return {"condition": condition, "neg_condition": neg_condition}

    device = torch.device("cuda", rank)
    text_encoder = TinyPreprocessor()
    reference_weight = text_encoder.weight.detach().to(device)
    context = fsdp_utils.create_fsdp_context(
        reduce_dtype=torch.float32,
        min_num_params=10**9,
        apply_cpu_offload=True,
    )
    text_encoder.prepare_module(
        device=device,
        precision_model=torch.float32,
        trainable=False,
        fsdp_context=context,
    )

    def is_cpu_shard() -> bool:
        parameter = text_encoder.weight
        return isinstance(parameter, DTensor) and parameter.to_local().device.type == "cpu"

    initially_resharded = is_cpu_shard()
    net = TinyNet(text_encoder)

    class TinyModel(SimpleNamespace):
        reshard_fsdp_modules = FastGenModel.reshard_fsdp_modules
        prepare_batch = FastGenModel.prepare_batch
        state_fields = FastGenModel.state_fields

    model = TinyModel(
        device=device,
        precision=torch.float32,
        precision_amp_enc=None,
        net_preprocessors=net,
        input_shape=(4,),
    )
    condition = torch.eye(4, dtype=torch.float32)
    neg_condition = torch.flip(condition, dims=(0,))
    output = model.prepare_batch({"condition": condition, "neg_condition": neg_condition})

    local_result = {
        "root_is_fsdp": isinstance(text_encoder, fsdp_utils.FSDPModule),
        "root_marked_fsdp": text_encoder._is_fsdp,
        "initially_resharded": initially_resharded,
        "retained_between_encodes": net.retained_between_encodes,
        "unsharded_during_encodes": text_encoder.saw_unsharded_cuda == [True, True],
        "resharded_after_preprocessing": is_cpu_shard(),
        "condition_error": float((output["condition"] - condition.to(device) @ reference_weight).abs().max()),
        "neg_condition_error": float(
            (output["neg_condition"] - neg_condition.to(device) @ reference_weight).abs().max()
        ),
    }
    per_rank = [None] * world_size
    dist.all_gather_object(per_rank, local_result)
    return {"per_rank": per_rank}


@RunIf(min_gpus=2)
def test_preprocess_boundary_retains_then_reshards_cpu_offloaded_root():
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_preprocess_boundary_reshard_impl,
        world_size=2,
        timeout=180,
        nccl_timeout=120,
        setup_fn=set_env_vars,
    )
    assert result is not None
    for rank, entry in enumerate(result["per_rank"]):
        assert entry["root_is_fsdp"], (rank, entry)
        assert entry["root_marked_fsdp"], (rank, entry)
        assert entry["initially_resharded"], (rank, entry)
        assert entry["retained_between_encodes"], (rank, entry)
        assert entry["unsharded_during_encodes"], (rank, entry)
        assert entry["resharded_after_preprocessing"], (rank, entry)
        assert entry["condition_error"] < 1e-6, (rank, entry)
        assert entry["neg_condition_error"] < 1e-6, (rank, entry)
    clear_gpu_memory()


def test_generic_fsdp_shards_a_shared_submodule_once(monkeypatch):
    shared = torch.nn.Linear(2, 2)
    left = torch.nn.Module()
    left.add_module("shared", shared)
    right = torch.nn.Module()
    right.add_module("shared", shared)
    root = torch.nn.Module()
    root.add_module("left", left)
    root.add_module("right", right)

    shard_calls = []
    monkeypatch.setattr(fsdp_utils, "fully_shard", _record_fully_shard_in_place(shard_calls))

    fsdp_utils._fully_shard_module(root, min_num_params=1, mesh=object())

    assert [module for module, _ in shard_calls] == [shared, root]


def test_generic_fsdp_keeps_shared_parameters_in_the_root_shard(monkeypatch):
    left = torch.nn.Linear(2, 2)
    right = torch.nn.Linear(2, 2)
    right.weight = left.weight
    root = torch.nn.Module()
    root.add_module("left", left)
    root.add_module("right", right)

    shard_calls = []
    monkeypatch.setattr(fsdp_utils, "fully_shard", lambda module, **kwargs: shard_calls.append((module, kwargs)))

    fsdp_utils._fully_shard_module(root, min_num_params=1, mesh=object())

    # Sharding either owner separately would split the tied weight across FSDP
    # units. The root sees both aliases and owns the parameter exactly once.
    assert [module for module, _ in shard_calls] == [root]


class _CustomMethodDepthBackbone(torch.nn.Module):
    """Toy DINOv2 analogue whose parameter is consumed outside ``forward``."""

    def __init__(self):
        super().__init__()
        self.channel_weight = torch.nn.Parameter(torch.tensor([0.5, 1.0, 1.5]))
        self.saw_unsharded_weight = False

    def get_intermediate_layers(self, image: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        self.saw_unsharded_weight = not isinstance(self.channel_weight, DTensor)
        return torch.einsum("chw,c->hw", image, self.channel_weight)

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # pragma: no cover - deliberate tripwire
        del image
        raise AssertionError("the depth model must use get_intermediate_layers, not child.forward")


class _CustomMethodDepthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = _CustomMethodDepthBackbone()

    def infer_tensor(self, image: torch.Tensor) -> torch.Tensor:
        return self.pretrained.get_intermediate_layers(image)


def _vace_depth_root_only_fsdp_impl(rank: int, world_size: int, model_path: str) -> dict:
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    from fastgen.networks.VaceWan.modules import vace_depth_annotator

    device = torch.device("cuda", rank)
    with patch.object(vace_depth_annotator, "DepthAnythingV2", side_effect=lambda **_kwargs: _CustomMethodDepthModel()):
        extractor = vace_depth_annotator.VACEDepthExtractor(model_path=model_path)
    context = fsdp_utils.create_fsdp_context(
        reduce_dtype=torch.float32,
        min_num_params=1,
    )
    extractor.prepare_module(
        device=device,
        precision_model=torch.float32,
        trainable=False,
        fsdp_context=context,
    )

    child_weight_is_dtensor = isinstance(extractor.model.pretrained.channel_weight, DTensor)
    frames = torch.arange(12, device=device, dtype=torch.float32).reshape(1, 3, 1, 2, 2)
    with torch.no_grad():
        output = extractor(frames)
    retained_after_forward = not isinstance(extractor.model.pretrained.channel_weight, DTensor)
    fsdp_utils.reshard_fsdp_modules((extractor,))

    return {
        "root_wrapped": isinstance(extractor, FSDPModule),
        "model_not_wrapped": not isinstance(extractor.model, FSDPModule),
        "child_not_wrapped": not isinstance(extractor.model.pretrained, FSDPModule),
        "child_weight_is_dtensor": child_weight_is_dtensor,
        "custom_method_saw_unsharded_weight": extractor.model.pretrained.saw_unsharded_weight,
        "child_weight_retained_until_boundary": retained_after_forward,
        "child_weight_resharded_at_boundary": isinstance(extractor.model.pretrained.channel_weight, DTensor),
        "output_succeeded": output.shape == frames.shape
        and output.dtype == frames.dtype
        and output.device == device
        and torch.isfinite(output).all().item(),
        "root_marked_fsdp": extractor._is_fsdp,
        "world_size": world_size,
    }


@RunIf(min_gpus=2)
def test_vace_depth_extractor_fsdp_uses_one_root_group_for_custom_child_method(tmp_path):
    """The root hook must unshard DINOv2-like state before a direct custom-method call."""
    clear_gpu_memory()
    model_path = tmp_path / "depth.pth"
    torch.save(_CustomMethodDepthModel().state_dict(), model_path)
    result = run_distributed_test(
        test_fn=_vace_depth_root_only_fsdp_impl,
        world_size=2,
        timeout=240,
        setup_fn=set_env_vars,
        model_path=str(model_path),
    )
    assert result == {
        "root_wrapped": True,
        "model_not_wrapped": True,
        "child_not_wrapped": True,
        "child_weight_is_dtensor": True,
        "custom_method_saw_unsharded_weight": True,
        "child_weight_retained_until_boundary": True,
        "child_weight_resharded_at_boundary": True,
        "output_succeeded": True,
        "root_marked_fsdp": True,
        "world_size": 2,
    }
    clear_gpu_memory()


# =============================================================================
# Network Loading and FSDP Utilities (using internal library code)
# =============================================================================
# This section drives FSDP2 and ``EDMPrecond.fully_shard`` directly (``to_empty``, ``reset_parameters``,
# ``set_model_state_dict``), not ``prepare_module``; the FSDP-context tests below cover the production path.


def _get_meta_init_context(meta_init: bool):
    """Get context manager for FSDP meta initialization.

    When meta_init is enabled, non-rank-0 processes use meta device
    for memory-efficient loading. Rank 0 loads weights normally, then
    FSDP detects the meta parameters and syncs weights to the other ranks.

    Args:
        meta_init: Whether to use meta initialization.

    Returns:
        Context manager (meta device for non-rank-0, nullcontext otherwise)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    use_meta = meta_init and rank != 0
    if use_meta:
        return torch.device("meta")
    return contextlib.nullcontext()


def create_edm_network(meta_init: bool = False):
    """Create an EDM CIFAR10 network using the internal library config.

    This is a small, fast network suitable for testing FSDP functionality
    without the overhead of loading large models like Wan.

    Args:
        meta_init: If True, non-rank-0 processes use meta device

    Returns:
        Instantiated EDM network
    """
    from fastgen.configs.net import EDM_CIFAR10_Config

    # Clone config
    config = copy.deepcopy(EDM_CIFAR10_Config)

    # Use meta device context for non-rank-0 processes when meta_init is True
    with _get_meta_init_context(meta_init):
        network = instantiate(config)

    return network


def apply_fsdp_to_network(network, device_mesh):
    """Apply FSDP sharding to a FastGenNetwork.

    Uses the network's built-in fully_shard method.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )

    # Use the network's fully_shard method
    network.fully_shard(mesh=device_mesh, mp_policy=mp_policy)


def gather_fsdp_state_dict(model) -> Dict[str, torch.Tensor]:
    """Gather full state dict from FSDP model."""
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )
    return get_model_state_dict(model, options=options)


def generate_dummy_inputs(
    batch_size: int = 2,
    img_resolution: int = 32,
    img_channels: int = 3,
    label_dim: int = 10,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
):
    """Generate dummy input data for EDM CIFAR10 model forward pass."""
    if device is None:
        device = torch.cuda.current_device()

    generator = torch.Generator(device="cpu").manual_seed(42)

    # EDM expects image-shaped input
    x = torch.randn(batch_size, img_channels, img_resolution, img_resolution, generator=generator, dtype=dtype).to(
        device
    )

    # Timestep (sigma value for EDM); schedulers consume timesteps in their own precision (float64 default).
    timestep = torch.tensor([0.5] * batch_size, dtype=torch.float64, device=device)

    # One-hot labels for class conditioning
    labels = torch.zeros(batch_size, label_dim, dtype=dtype, device=device)
    labels[:, 0] = 1.0  # Set first class as the label

    return {
        "x_t": x,
        "t": timestep,
        "condition": labels,
    }


def load_reference_network():
    """Instantiate a fresh reference network on CPU.

    Its random initialization differs from the FSDP network's, so callers load rank 0's
    pre-shard ``state_dict`` into it before comparing outputs.
    """
    from fastgen.configs.net import EDM_CIFAR10_Config

    config = copy.deepcopy(EDM_CIFAR10_Config)

    # Load on CPU to avoid GPU memory issues
    network = instantiate(config)

    return network


# =============================================================================
# Test Functions (run inside distributed workers)
# =============================================================================


def _test_fsdp_weight_sync_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test that FSDP weights are correctly synced from rank 0 to all ranks.

    This test:
    1. Rank 0 loads full model, other ranks use meta device
    2. Apply FSDP sharding with state dict broadcast
    3. Gather state dict back and compare to the ORIGINAL state dict from rank 0

    This verifies that ALL weights (including randomly-initialized ones like
    logvar_linear) are correctly synced via FSDP2's set_model_state_dict.
    """
    device_mesh = init_device_mesh("cuda", (world_size,))

    # Load network (rank 0 loads weights, others use meta device via meta_init)
    network = create_edm_network(meta_init=True)

    # Extract state dict from rank 0 BEFORE FSDP sharding
    # We need TWO copies:
    # - broadcast_state_dict: passed to set_model_state_dict (gets modified in place to DTensor)
    # - reference_state_dict: kept intact for comparison after gathering
    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
        # Create a separate reference copy with tensors moved to CPU to avoid DTensor issues
        reference_state_dict = {k: v.cpu().clone() for k, v in broadcast_state_dict.items()}
    else:
        broadcast_state_dict = None
        reference_state_dict = None

    dist.barrier()

    # Apply FSDP sharding using the network's method
    apply_fsdp_to_network(network, device_mesh)

    # Materialize meta tensors and reset parameters
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    # Broadcast state dict from rank 0
    # NOTE: This modifies broadcast_state_dict in place (tensors become DTensors)
    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Gather state dict back from FSDP model
    gathered_state_dict = gather_fsdp_state_dict(network)
    dist.barrier()

    # Only rank 0 performs comparison
    result = {"weights_match": True, "mismatches": [], "num_keys_compared": 0}
    if rank == 0:
        # Compare gathered state dict against the REFERENCE state dict
        # (the unmodified copy we kept before set_model_state_dict)
        reference_keys = set(reference_state_dict.keys())
        gathered_keys = set(gathered_state_dict.keys())
        common_keys = reference_keys & gathered_keys

        mismatches = []
        for key in common_keys:
            # reference_state_dict tensors are already on CPU
            reference_tensor = reference_state_dict[key].float()
            gathered_tensor = gathered_state_dict[key].cpu().float()

            if reference_tensor.shape != gathered_tensor.shape:
                mismatches.append((key, "shape_mismatch", str(reference_tensor.shape), str(gathered_tensor.shape)))
                continue

            max_diff = (reference_tensor - gathered_tensor).abs().max().item()
            if max_diff > 1e-5:
                mismatches.append((key, "value_mismatch", max_diff))

        # Also check for missing/extra keys
        missing_keys = reference_keys - gathered_keys
        extra_keys = gathered_keys - reference_keys
        for key in missing_keys:
            mismatches.append((key, "missing_in_gathered"))
        for key in extra_keys:
            mismatches.append((key, "extra_in_gathered"))

        result["weights_match"] = len(mismatches) == 0
        result["mismatches"] = mismatches
        result["num_keys_compared"] = len(common_keys)
        result["missing_keys"] = len(missing_keys)
        result["extra_keys"] = len(extra_keys)

        torch.cuda.empty_cache()

    return result


def _test_fsdp_tensor_sharding_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test that FSDP actually shards tensors across ranks.

    This test verifies that:
    1. After FSDP sharding, local tensor shapes differ from full tensor shapes
    2. The total number of local elements is less than full elements
    3. When gathered, the full shapes are restored

    This is important to verify that FSDP is actually sharding the model
    and not just wrapping it without sharding.
    """
    from torch.distributed.tensor import DTensor

    device_mesh = init_device_mesh("cuda", (world_size,))

    # Load network
    network = create_edm_network(meta_init=True)

    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
        # Store original shapes for comparison
        original_shapes = {k: v.shape for k, v in broadcast_state_dict.items()}
    else:
        broadcast_state_dict = None
        original_shapes = None

    dist.barrier()

    # Apply FSDP sharding
    apply_fsdp_to_network(network, device_mesh)
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Check that parameters are DTensors and have different local vs full shapes
    dtensor_params = []
    local_vs_full_diffs = []
    total_local_numel = 0
    total_full_numel = 0

    for name, param in network.named_parameters():
        if isinstance(param, DTensor):
            dtensor_params.append(name)
            local_shape = tuple(param._local_tensor.shape)
            full_shape = tuple(param.shape)
            local_numel = param._local_tensor.numel()
            full_numel = param.numel()

            total_local_numel += local_numel
            total_full_numel += full_numel

            # Record if shapes differ (they should for sharded params)
            if local_shape != full_shape:
                local_vs_full_diffs.append(
                    {
                        "name": name,
                        "local_shape": local_shape,
                        "full_shape": full_shape,
                        "local_numel": local_numel,
                        "full_numel": full_numel,
                    }
                )

    # Gather state dict and verify full shapes are restored
    gathered_state_dict = gather_fsdp_state_dict(network)
    dist.barrier()

    result = {
        "rank": rank,
        "world_size": world_size,
        "num_dtensor_params": len(dtensor_params),
        "num_params_with_shape_diff": len(local_vs_full_diffs),
        "total_local_numel": total_local_numel,
        "total_full_numel": total_full_numel,
        "local_vs_full_diffs": local_vs_full_diffs[:5],  # First 5 for debugging
    }

    # Verify that at least some parameters have different local vs full shapes
    # This confirms FSDP is actually sharding
    has_sharded_params = len(local_vs_full_diffs) > 0
    result["has_sharded_params"] = has_sharded_params

    # Verify local numel is less than full numel (sharding reduces per-rank memory)
    # For FSDP with world_size ranks, local should be approximately 1/world_size of full
    if total_full_numel > 0:
        sharding_ratio = total_local_numel / total_full_numel
        expected_ratio = 1.0 / world_size
        # Allow some tolerance since not all params may be sharded equally
        ratio_is_reasonable = sharding_ratio < 0.8  # Should be significantly less than 1.0
        result["sharding_ratio"] = sharding_ratio
        result["expected_ratio"] = expected_ratio
        result["ratio_is_reasonable"] = ratio_is_reasonable
    else:
        result["ratio_is_reasonable"] = False
        result["sharding_ratio"] = None

    # Verify gathered shapes match original shapes (on rank 0)
    if rank == 0 and original_shapes is not None:
        shapes_match = True
        shape_mismatches = []
        for key in gathered_state_dict:
            if key in original_shapes:
                gathered_shape = tuple(gathered_state_dict[key].shape)
                orig_shape = tuple(original_shapes[key])
                if gathered_shape != orig_shape:
                    shapes_match = False
                    shape_mismatches.append((key, orig_shape, gathered_shape))
        result["gathered_shapes_match"] = shapes_match
        result["shape_mismatches"] = shape_mismatches[:5]
    else:
        result["gathered_shapes_match"] = True
        result["shape_mismatches"] = []

    result["all_passed"] = (
        has_sharded_params and result.get("ratio_is_reasonable", False) and result.get("gathered_shapes_match", False)
    )

    return result


def _test_fsdp_buffer_sync_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test that FSDP buffers are correctly synced across ranks.

    ``to_empty`` leaves the meta ranks' persistent buffers uninitialized; the rank-0 broadcast in
    ``set_model_state_dict`` must restore them, so every rank agrees afterwards.
    """
    device_mesh = init_device_mesh("cuda", (world_size,))

    # Load network
    network = create_edm_network(meta_init=True)

    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
    else:
        broadcast_state_dict = None

    dist.barrier()

    # Apply FSDP
    apply_fsdp_to_network(network, device_mesh)

    # Materialize the meta ranks; the rank-0 broadcast below restores the persistent buffers
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Check buffer synchronization across ranks
    buffer_max_diff = 0.0
    buffer_issues = []

    for name, buffer in network.named_buffers():
        # Clone rank 0's buffer and broadcast
        rank0_buffer = buffer.clone()
        dist.broadcast(rank0_buffer, src=0)

        # Compare local buffer to rank 0's buffer
        local_vs_rank0_diff = (buffer - rank0_buffer).abs().max().item()

        if local_vs_rank0_diff > buffer_max_diff:
            buffer_max_diff = local_vs_rank0_diff
        if local_vs_rank0_diff > 1e-6:
            buffer_issues.append((name, local_vs_rank0_diff))

    # Reduce buffer_max_diff across all ranks
    buffer_max_diff_tensor = torch.tensor([buffer_max_diff], device=torch.cuda.current_device())
    dist.all_reduce(buffer_max_diff_tensor, op=dist.ReduceOp.MAX)
    global_buffer_max_diff = buffer_max_diff_tensor.item()

    result = {
        "buffers_synced": global_buffer_max_diff < 1e-6,
        "max_diff": global_buffer_max_diff,
        "issues": buffer_issues if rank == 0 else [],
    }

    return result


def _test_fsdp_forward_pass_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test that FSDP forward pass produces consistent results.

    Tests:
    1. FSDP forward is deterministic (same input -> same output)
    2. All ranks produce the same output
    3. FSDP output matches a reference model holding rank 0's pre-shard weights
    """
    device_mesh = init_device_mesh("cuda", (world_size,))

    # Load and setup FSDP model
    network = create_edm_network(meta_init=True)

    # Keep an intact CPU copy of rank 0's weights: set_model_state_dict turns the broadcast copy into DTensors.
    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
        reference_state_dict = {k: v.cpu().clone() for k, v in broadcast_state_dict.items()}
    else:
        broadcast_state_dict = None
        reference_state_dict = None

    dist.barrier()

    apply_fsdp_to_network(network, device_mesh)
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Generate inputs (same on all ranks due to fixed seed)
    inputs = generate_dummy_inputs(
        batch_size=2,
        img_resolution=32,
        img_channels=3,
        label_dim=10,
        device=torch.cuda.current_device(),
        dtype=torch.float32,
    )

    network.eval()

    # Test 1: FSDP forward is deterministic
    with torch.no_grad():
        output1 = network(
            x_t=inputs["x_t"],
            t=inputs["t"],
            condition=inputs["condition"],
            return_features_early=False,
            feature_indices=None,
        )

        output2 = network(
            x_t=inputs["x_t"],
            t=inputs["t"],
            condition=inputs["condition"],
            return_features_early=False,
            feature_indices=None,
        )

    consistency_diff = (output1.cpu() - output2.cpu()).abs().max().item()

    # Test 2: All ranks produce same output
    if hasattr(output1, "full_tensor"):
        output_full = output1.full_tensor()
    else:
        output_full = output1

    rank0_output = output_full.clone()
    dist.broadcast(rank0_output, src=0)
    rank_consistency_diff = _max_across_ranks((output_full - rank0_output).abs().max().item())

    fsdp_output_cpu = output_full.cpu().float()

    dist.barrier()

    # Test 3: Compare to reference (only on rank 0)
    result = {
        "deterministic": consistency_diff < 1e-6,
        "consistency_diff": consistency_diff,
        "ranks_consistent": rank_consistency_diff < 1e-6,
        "rank_consistency_diff": rank_consistency_diff,
    }

    if rank == 0:
        # Offload FSDP model
        network.to("cpu")
        torch.cuda.empty_cache()

        # Load the reference model with rank 0's pre-shard weights
        ref_network = load_reference_network()
        ref_network.load_state_dict(reference_state_dict, strict=True)
        ref_network = ref_network.to(device=torch.cuda.current_device(), dtype=torch.float32)
        ref_network.eval()

        ref_inputs = to(inputs, device=torch.cuda.current_device())

        with torch.no_grad():
            ref_output = ref_network(
                x_t=ref_inputs["x_t"],
                t=ref_inputs["t"],
                condition=ref_inputs["condition"],
                return_features_early=False,
                feature_indices=None,
            )

        ref_output_cpu = ref_output.cpu().float()
        del ref_network
        torch.cuda.empty_cache()

        # Compare outputs
        max_diff = (fsdp_output_cpu - ref_output_cpu).abs().max().item()
        mean_diff = (fsdp_output_cpu - ref_output_cpu).abs().mean().item()
        ref_norm = ref_output_cpu.abs().mean().item()
        relative_error = mean_diff / (ref_norm + 1e-8)

        result["max_diff"] = max_diff
        result["mean_diff"] = mean_diff
        result["relative_error"] = relative_error
        result["forward_matches"] = relative_error < 1e-3 and max_diff < 1e-2

    return result


def _test_fsdp_meta_device_init_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test that meta_init correctly uses meta device for non-rank-0 processes.

    This test verifies:
    1. Rank 0 loads real tensors on CPU/CUDA
    2. Non-rank-0 processes have tensors on meta device
    3. After FSDP sync, all ranks have real tensors
    """
    # Load network with meta_init=True
    network = create_edm_network(meta_init=True)

    # Check tensor devices BEFORE FSDP sharding
    meta_tensors_before = []
    real_tensors_before = []

    for name, param in network.named_parameters():
        if param.device.type == "meta":
            meta_tensors_before.append(name)
        else:
            real_tensors_before.append(name)

    # Verify meta device usage based on rank
    if rank == 0:
        # Rank 0 should have real tensors (not meta)
        has_correct_device_before = len(meta_tensors_before) == 0
    else:
        # Non-rank-0 should have meta tensors
        has_correct_device_before = len(meta_tensors_before) > 0 and len(real_tensors_before) == 0
    # Only rank 0's result reaches the test, so every rank's verdict is reduced into it.
    has_correct_device_before = _all_true(has_correct_device_before)

    result = {
        "rank": rank,
        "meta_tensors_before_fsdp": len(meta_tensors_before),
        "real_tensors_before_fsdp": len(real_tensors_before),
        "has_correct_device_before": has_correct_device_before,
    }

    # Now test that after FSDP sharding and sync, all tensors are materialized
    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
    else:
        broadcast_state_dict = None

    dist.barrier()

    device_mesh = init_device_mesh("cuda", (world_size,))
    apply_fsdp_to_network(network, device_mesh)

    # Materialize meta tensors and reset parameters
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    # Broadcast state dict from rank 0
    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Check tensor devices AFTER FSDP sync
    meta_tensors_after = []
    real_tensors_after = []

    for name, param in network.named_parameters():
        # After FSDP, parameters may be DTensor or regular tensor on CUDA
        # Check the underlying device
        if hasattr(param, "_local_tensor"):
            # DTensor case
            device = param._local_tensor.device
        else:
            device = param.device

        if device.type == "meta":
            meta_tensors_after.append(name)
        else:
            real_tensors_after.append(name)

    # After FSDP sync, no tensors should be on meta device
    has_correct_device_after = _all_true(len(meta_tensors_after) == 0)

    result["meta_tensors_after_fsdp"] = len(meta_tensors_after)
    result["real_tensors_after_fsdp"] = len(real_tensors_after)
    result["has_correct_device_after"] = has_correct_device_after
    result["all_passed"] = has_correct_device_before and has_correct_device_after

    return result


def _test_fsdp_gathered_weights_forward_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test that a model loaded with gathered FSDP weights produces correct output.

    This isolates whether issues come from FSDP forward computation vs weight gathering.
    """
    device_mesh = init_device_mesh("cuda", (world_size,))

    # Setup FSDP model
    network = create_edm_network(meta_init=True)

    # Keep an intact CPU copy of rank 0's weights: set_model_state_dict turns the broadcast copy into DTensors.
    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
        reference_state_dict = {k: v.cpu().clone() for k, v in broadcast_state_dict.items()}
    else:
        broadcast_state_dict = None
        reference_state_dict = None

    dist.barrier()

    apply_fsdp_to_network(network, device_mesh)
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Gather state dict
    gathered_state_dict = gather_fsdp_state_dict(network)
    dist.barrier()

    result = {}

    if rank == 0:
        # Generate inputs
        inputs = generate_dummy_inputs(
            batch_size=2,
            img_resolution=32,
            img_channels=3,
            label_dim=10,
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        )

        # Load fresh model with gathered weights
        gathered_network = load_reference_network()
        gathered_network = gathered_network.to(device=torch.cuda.current_device(), dtype=torch.float32)
        gathered_network.load_state_dict(gathered_state_dict, strict=True)

        gathered_network.eval()
        with torch.no_grad():
            gathered_output = gathered_network(
                x_t=inputs["x_t"],
                t=inputs["t"],
                condition=inputs["condition"],
                return_features_early=False,
                feature_indices=None,
            )

        gathered_output_cpu = gathered_output.cpu().float()
        del gathered_network
        torch.cuda.empty_cache()

        # Load the reference model with rank 0's pre-shard weights
        ref_network = load_reference_network()
        ref_network.load_state_dict(reference_state_dict, strict=True)
        ref_network = ref_network.to(device=torch.cuda.current_device(), dtype=torch.float32)

        ref_network.eval()
        with torch.no_grad():
            ref_output = ref_network(
                x_t=inputs["x_t"],
                t=inputs["t"],
                condition=inputs["condition"],
                return_features_early=False,
                feature_indices=None,
            )

        ref_output_cpu = ref_output.cpu().float()
        del ref_network
        torch.cuda.empty_cache()

        # Compare
        max_diff = (gathered_output_cpu - ref_output_cpu).abs().max().item()
        mean_diff = (gathered_output_cpu - ref_output_cpu).abs().mean().item()

        result = {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "matches": max_diff < 1e-4,
        }

    return result


# =============================================================================
# Pytest Test Functions
# =============================================================================


@RunIf(min_gpus=2)
def test_fsdp_weight_sync():
    """Test that FSDP correctly synchronizes weights from rank 0 to all ranks."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_weight_sync_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result[
        "weights_match"
    ], f"Weight sync failed with {len(result['mismatches'])} mismatches: {result['mismatches'][:5]}"

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_tensor_sharding():
    """Test that FSDP actually shards tensors across ranks.

    This verifies that:
    - Local tensor shapes differ from full tensor shapes for sharded parameters
    - The sharding ratio is reasonable (local < full)
    - Gathered shapes match original shapes
    """
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_tensor_sharding_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_sharded_params", False), (
        f"No parameters were sharded. num_dtensor_params={result.get('num_dtensor_params', 'N/A')}, "
        f"num_params_with_shape_diff={result.get('num_params_with_shape_diff', 'N/A')}"
    )
    assert result.get("ratio_is_reasonable", False), (
        f"Sharding ratio is not reasonable: {result.get('sharding_ratio', 'N/A')} "
        f"(expected ~{result.get('expected_ratio', 'N/A')})"
    )
    assert result.get(
        "gathered_shapes_match", False
    ), f"Gathered shapes don't match original shapes: {result.get('shape_mismatches', 'N/A')}"
    assert result.get("all_passed", False), (
        f"Tensor sharding test failed. Details: " f"local_vs_full_diffs={result.get('local_vs_full_diffs', 'N/A')}"
    )

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_buffer_sync():
    """Test that FSDP correctly synchronizes buffers across ranks."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_buffer_sync_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result["buffers_synced"], f"Buffer sync failed with max_diff={result['max_diff']}: {result['issues']}"

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_forward_pass():
    """FSDP forward is deterministic, rank-consistent, and matches the reference holding rank 0's weights."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_forward_pass_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result["deterministic"], f"FSDP forward not deterministic: diff={result['consistency_diff']}"
    assert result["ranks_consistent"], f"Ranks inconsistent: diff={result['rank_consistency_diff']}"
    assert result.get("forward_matches", False), (
        f"FSDP forward doesn't match reference: "
        f"max_diff={result.get('max_diff', 'N/A')}, "
        f"relative_error={result.get('relative_error', 'N/A')}"
    )

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_gathered_weights_forward():
    """Test that model loaded with gathered FSDP weights produces correct output."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_gathered_weights_forward_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get(
        "matches", False
    ), f"Gathered weights model doesn't match reference: max_diff={result.get('max_diff', 'N/A')}"

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_meta_device_init():
    """Test that meta_init correctly uses meta device for non-rank-0 processes.

    This verifies:
    - Rank 0 loads real tensors (not meta)
    - Non-rank-0 processes use meta device for memory-efficient loading
    - After FSDP sync, all tensors are properly materialized
    """
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_meta_device_init_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_correct_device_before", False), (
        f"Meta device not used correctly before FSDP: "
        f"rank={result.get('rank', 'N/A')}, "
        f"meta_tensors={result.get('meta_tensors_before_fsdp', 'N/A')}, "
        f"real_tensors={result.get('real_tensors_before_fsdp', 'N/A')}"
    )
    assert result.get("has_correct_device_after", False), (
        f"Tensors not properly materialized after FSDP: " f"meta_tensors={result.get('meta_tensors_after_fsdp', 'N/A')}"
    )

    clear_gpu_memory()


# =============================================================================
# End-to-End Model FSDP Context Test
# =============================================================================


def _test_fsdp_context_impl(
    rank: int,
    world_size: int,
    sharding_group_size: int = None,
) -> Dict:
    """Test complete inline FSDP preparation with a real model.

    This test verifies:
    1. Model instantiation with meta_init=True
    2. meta device usage for non-rank-0 processes
    3. The shared FSDP context correctly wraps and syncs the model during preparation
    4. After FSDP, all tensors are properly materialized
    5. Forward pass produces consistent results across ranks

    Args:
        sharding_group_size: If set, creates a 2D mesh with (replicate, shard) dimensions.
    """
    import copy
    from fastgen.configs.methods.config_sft import ModelConfig
    from fastgen.configs.net import EDM_CIFAR10_Config
    from fastgen.methods import SFTModel
    from fastgen.utils.distributed.fsdp import create_fsdp_context

    # Create a minimal model config with EDM (small, fast network)
    config = ModelConfig()
    config.net = copy.deepcopy(EDM_CIFAR10_Config)
    config.input_shape = [3, 32, 32]
    config.precision = "float32"
    config.device = "cuda"
    config.meta_init = True
    config.pretrained_model_path = None  # No pretrained weights needed
    config.use_ema = False  # Disable EMA for simpler testing

    result = {
        "rank": rank,
        "world_size": world_size,
        "sharding_group_size": sharding_group_size,
    }

    # Step 1: Build model with meta_init=True
    # The model's build_model() uses _get_meta_init_context() which wraps
    # network instantiation in meta device context for non-rank-0 processes
    model = SFTModel(config)

    # Step 2: Verify meta device usage before module preparation
    meta_tensors_before = []
    real_tensors_before = []

    for name, param in model.net.named_parameters():
        if param.device.type == "meta":
            meta_tensors_before.append(name)
        else:
            real_tensors_before.append(name)

    if rank == 0:
        # Rank 0 should have real tensors (not meta)
        has_correct_device_before = len(meta_tensors_before) == 0
    else:
        # Non-rank-0 should have meta tensors
        has_correct_device_before = len(meta_tensors_before) > 0 and len(real_tensors_before) == 0
    # Only rank 0's result reaches the test, so every rank's verdict is reduced into it.
    has_correct_device_before = _all_true(has_correct_device_before)

    result["meta_tensors_before_fsdp"] = len(meta_tensors_before)
    result["real_tensors_before_fsdp"] = len(real_tensors_before)
    result["has_correct_device_before"] = has_correct_device_before

    dist.barrier()

    # Step 3: Prepare the shared context, then cast, shard, place, and synchronize each
    # selected root through the production module-preparation path.
    fsdp_context = create_fsdp_context(
        reduce_dtype=model.precision_fsdp_reduce,
        min_num_params=1_000_000,  # Lower threshold for the smaller EDM network
        apply_cpu_offload=False,
        sharding_group_size=sharding_group_size,
    )
    model.on_train_begin(fsdp_context=fsdp_context)
    result["preparation_complete"] = _all_true(model.net.is_prepared and model.net._is_fsdp)

    dist.barrier()

    # Step 4: Verify tensors are materialized after inline FSDP preparation
    meta_tensors_after = []
    real_tensors_after = []

    for name, param in model.net.named_parameters():
        # After FSDP, parameters may be DTensor or regular tensor on CUDA
        if hasattr(param, "_local_tensor"):
            device = param._local_tensor.device
        else:
            device = param.device

        if device.type == "meta":
            meta_tensors_after.append(name)
        else:
            real_tensors_after.append(name)

    has_correct_device_after = _all_true(len(meta_tensors_after) == 0)
    result["meta_tensors_after_fsdp"] = len(meta_tensors_after)
    result["real_tensors_after_fsdp"] = len(real_tensors_after)
    result["has_correct_device_after"] = has_correct_device_after

    # Step 5: Test forward pass consistency across ranks
    model.net.eval()

    # Generate same random input on all ranks
    generator = torch.Generator(device="cpu").manual_seed(42)
    dummy_input = torch.randn(2, 3, 32, 32, generator=generator, dtype=torch.float32).cuda()
    dummy_t = torch.tensor(
        [0.5, 0.5],
        dtype=model.net.noise_scheduler.t_precision,
        device="cuda",
    )
    dummy_labels = torch.zeros(2, 10, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        output = model.net(dummy_input, dummy_t, condition=dummy_labels)

    # Compare output across ranks
    if hasattr(output, "full_tensor"):
        output_full = output.full_tensor()
    else:
        output_full = output

    rank0_output = output_full.clone()
    dist.broadcast(rank0_output, src=0)
    rank_consistency_diff = _max_across_ranks((output_full - rank0_output).abs().max().item())

    result["forward_rank_consistent"] = rank_consistency_diff < 1e-5
    result["rank_consistency_diff"] = rank_consistency_diff
    result["all_passed"] = (
        has_correct_device_before
        and has_correct_device_after
        and result["preparation_complete"]
        and result["forward_rank_consistent"]
    )

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_fsdp_context_code_path():
    """Test complete inline FSDP preparation with a real model.

    This is an end-to-end test that verifies:
    - Model builds correctly with meta_init=True
    - Non-rank-0 processes use meta device
    - The shared FSDP context correctly wraps and syncs weights
    - Forward pass is consistent across ranks
    """
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_context_impl,
        world_size=2,
        timeout=300,  # Allow more time for model instantiation
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_correct_device_before", False), (
        f"Meta device not used correctly before FSDP: "
        f"rank={result.get('rank', 'N/A')}, "
        f"meta_tensors={result.get('meta_tensors_before_fsdp', 'N/A')}, "
        f"real_tensors={result.get('real_tensors_before_fsdp', 'N/A')}"
    )
    assert result.get("has_correct_device_after", False), (
        f"Tensors not properly materialized after FSDP preparation: "
        f"meta_tensors={result.get('meta_tensors_after_fsdp', 'N/A')}"
    )
    assert result.get("preparation_complete", False), "The logical root did not complete FSDP preparation"
    assert result.get("forward_rank_consistent", False), (
        f"Forward pass inconsistent across ranks: " f"diff={result.get('rank_consistency_diff', 'N/A')}"
    )
    assert result.get("all_passed", False), "FSDP-context preparation test failed"

    clear_gpu_memory()


@RunIf(min_gpus=4)
def test_fsdp_context_sharding_group():
    """Test inline FSDP preparation with a sharding group size (2D mesh).

    This test verifies FSDP with a 2D device mesh where:
    - sharding_group_size=2 means weights are sharded within groups of 2 GPUs
    - The mesh has (replicate, shard) dimensions

    Requires at least 4 GPUs and world_size must be divisible by 2.
    """

    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_context_impl,
        world_size=4,
        timeout=300,
        setup_fn=set_env_vars,
        sharding_group_size=2,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_correct_device_before", False), (
        f"Meta device not used correctly before FSDP: "
        f"rank={result.get('rank', 'N/A')}, "
        f"meta_tensors={result.get('meta_tensors_before_fsdp', 'N/A')}, "
        f"real_tensors={result.get('real_tensors_before_fsdp', 'N/A')}"
    )
    assert result.get("has_correct_device_after", False), (
        f"Tensors not properly materialized after FSDP preparation: "
        f"meta_tensors={result.get('meta_tensors_after_fsdp', 'N/A')}"
    )
    assert result.get("forward_rank_consistent", False), (
        f"Forward pass inconsistent across ranks: " f"diff={result.get('rank_consistency_diff', 'N/A')}"
    )
    assert result.get("all_passed", False), "FSDP-context sharding group test failed"

    clear_gpu_memory()


def _test_fsdp_context_no_meta_init_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test inline FSDP preparation without meta init.

    This test verifies that the standard (non-meta-init) FSDP path works:
    1. All ranks load real tensors (no meta device)
    2. The shared FSDP context correctly wraps the model during preparation
    3. Forward pass produces consistent results across ranks
    """
    import copy
    from fastgen.configs.methods.config_sft import ModelConfig
    from fastgen.configs.net import EDM_CIFAR10_Config
    from fastgen.methods import SFTModel
    from fastgen.utils.distributed.fsdp import create_fsdp_context

    # Create a minimal model config with EDM (small, fast network)
    # meta_init=False means all ranks load their own weights
    config = ModelConfig()
    config.net = copy.deepcopy(EDM_CIFAR10_Config)
    config.input_shape = [3, 32, 32]
    config.precision = "float32"
    config.device = "cuda"
    config.meta_init = False  # All ranks load weights independently
    config.pretrained_model_path = None
    config.use_ema = False

    result = {
        "rank": rank,
        "world_size": world_size,
    }

    # Step 1: Build model with meta_init=False
    model = SFTModel(config)

    # Step 2: Verify all ranks have real tensors (no meta device)
    meta_tensors_before = []
    real_tensors_before = []

    for name, param in model.net.named_parameters():
        if param.device.type == "meta":
            meta_tensors_before.append(name)
        else:
            real_tensors_before.append(name)

    # All ranks should have real tensors (not meta)
    has_correct_device_before = _all_true(len(meta_tensors_before) == 0)
    result["meta_tensors_before_fsdp"] = len(meta_tensors_before)
    result["real_tensors_before_fsdp"] = len(real_tensors_before)
    result["has_correct_device_before"] = has_correct_device_before

    dist.barrier()

    # Step 3: Apply inline FSDP preparation
    fsdp_context = create_fsdp_context(
        reduce_dtype=model.precision_fsdp_reduce,
        min_num_params=1_000_000,
        apply_cpu_offload=False,
    )
    model.on_train_begin(fsdp_context=fsdp_context)

    dist.barrier()

    # Step 4: Verify tensors are on CUDA after FSDP
    meta_tensors_after = []
    real_tensors_after = []

    for name, param in model.net.named_parameters():
        if hasattr(param, "_local_tensor"):
            device = param._local_tensor.device
        else:
            device = param.device

        if device.type == "meta":
            meta_tensors_after.append(name)
        else:
            real_tensors_after.append(name)

    has_correct_device_after = _all_true(len(meta_tensors_after) == 0)
    result["meta_tensors_after_fsdp"] = len(meta_tensors_after)
    result["real_tensors_after_fsdp"] = len(real_tensors_after)
    result["has_correct_device_after"] = has_correct_device_after

    # Step 5: Test forward pass consistency across ranks
    model.net.eval()

    generator = torch.Generator(device="cpu").manual_seed(42)
    dummy_input = torch.randn(2, 3, 32, 32, generator=generator, dtype=torch.float32).cuda()
    dummy_t = torch.tensor([0.5, 0.5], dtype=model.net.noise_scheduler.t_precision, device="cuda")
    dummy_labels = torch.zeros(2, 10, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        output = model.net(dummy_input, dummy_t, condition=dummy_labels)

    if hasattr(output, "full_tensor"):
        output_full = output.full_tensor()
    else:
        output_full = output

    rank0_output = output_full.clone()
    dist.broadcast(rank0_output, src=0)
    rank_consistency_diff = _max_across_ranks((output_full - rank0_output).abs().max().item())

    result["forward_rank_consistent"] = rank_consistency_diff < 1e-5
    result["rank_consistency_diff"] = rank_consistency_diff
    result["all_passed"] = has_correct_device_before and has_correct_device_after and result["forward_rank_consistent"]

    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_fsdp_context_no_meta_init():
    """Test inline FSDP preparation without meta init.

    This verifies the standard FSDP path where all ranks load weights independently.
    """
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_context_no_meta_init_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_correct_device_before", False), (
        f"All ranks should have real tensors before FSDP: "
        f"meta_tensors={result.get('meta_tensors_before_fsdp', 'N/A')}"
    )
    assert result.get("has_correct_device_after", False), (
        f"Tensors not on CUDA after FSDP preparation: " f"meta_tensors={result.get('meta_tensors_after_fsdp', 'N/A')}"
    )
    assert result.get("forward_rank_consistent", False), (
        f"Forward pass inconsistent across ranks: " f"diff={result.get('rank_consistency_diff', 'N/A')}"
    )

    clear_gpu_memory()


def _test_dmd2_fsdp_context_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test inline FSDP preparation with a multi-network DMD2Model.

    This configuration has:
    - net: student network
    - teacher: frozen teacher network
    - fake_score: trainable fake score network

    GAN loss is disabled here, so no discriminator is built. This tests that FSDP correctly
    handles the trainable mappings plus a frozen ``extra_fsdp_modules`` entry.
    """
    import copy
    from fastgen.configs.methods.config_dmd2 import ModelConfig
    from fastgen.configs.net import EDM_CIFAR10_Config
    from fastgen.methods import DMD2Model
    from fastgen.utils.distributed.fsdp import create_fsdp_context

    # Create DMD2 config with EDM network
    config = ModelConfig()
    config.net = copy.deepcopy(EDM_CIFAR10_Config)
    config.input_shape = [3, 32, 32]
    config.precision = "float32"
    config.precision_model = "bfloat16"
    config.precision_opt = "float32"
    config.device = "cuda"
    config.meta_init = True
    config.pretrained_model_path = None
    config.use_ema = False
    config.extra_fsdp_modules = ["teacher"]
    # Disable GAN loss to avoid discriminator complexity
    config.gan_loss_weight_gen = 0.0

    result = {
        "rank": rank,
        "world_size": world_size,
    }

    # Step 1: Build DMD2 model with meta_init=True
    model = DMD2Model(config)

    # Step 2: Verify meta device usage for all networks
    networks_to_check = list(model.module_dict)
    meta_status = {}

    for net_name in networks_to_check:
        if hasattr(model, net_name):
            network = getattr(model, net_name)
            meta_count = sum(1 for p in network.parameters() if p.device.type == "meta")
            real_count = sum(1 for p in network.parameters() if p.device.type != "meta")
            meta_status[net_name] = {
                "meta": meta_count,
                "real": real_count,
            }

    # Verify correct meta device usage based on rank
    if rank == 0:
        # Rank 0: all networks should have real tensors
        has_correct_device_before = all(status["meta"] == 0 for status in meta_status.values())
    else:
        # Other ranks: every network uses meta init.
        has_correct_device_before = all(status["real"] == 0 for status in meta_status.values())
    has_correct_device_before = _all_true(has_correct_device_before)

    result["meta_status_before"] = meta_status
    result["has_correct_device_before"] = has_correct_device_before

    dist.barrier()

    # Step 3: Prepare every selected root inline through one shared context.
    fsdp_context = create_fsdp_context(
        reduce_dtype=model.precision_fsdp_reduce,
        min_num_params=1_000_000,
        apply_cpu_offload=False,
    )
    model.on_train_begin(fsdp_context=fsdp_context)

    dist.barrier()

    # Step 4: Verify all networks are materialized after FSDP
    meta_status_after = {}
    for net_name in networks_to_check:
        if hasattr(model, net_name):
            network = getattr(model, net_name)
            meta_count = 0
            real_count = 0
            for p in network.parameters():
                if hasattr(p, "_local_tensor"):
                    device = p._local_tensor.device
                else:
                    device = p.device
                if device.type == "meta":
                    meta_count += 1
                else:
                    real_count += 1
            meta_status_after[net_name] = {
                "meta": meta_count,
                "real": real_count,
            }

    has_correct_device_after = _all_true(all(status["meta"] == 0 for status in meta_status_after.values()))

    def has_correct_storage_dtype(name, network):
        default = torch.bfloat16 if name in config.extra_fsdp_modules else torch.float32
        # EDM's FP32 mapping path keeps its pinned storage in every role.
        pinned = network._fixed_precision_overrides()
        return all(
            parameter.dtype == pinned.get(parameter_name, default)
            for parameter_name, parameter in network.named_parameters()
            if parameter.is_floating_point()
        )

    has_correct_storage_dtype_after = _all_true(
        all(has_correct_storage_dtype(name, getattr(model, name)) for name in networks_to_check)
    )
    result["meta_status_after"] = meta_status_after
    result["has_correct_device_after"] = has_correct_device_after
    result["has_correct_storage_dtype_after"] = has_correct_storage_dtype_after

    # Step 5: Test each runnable network.
    for name in networks_to_check:
        getattr(model, name).eval()

    generator = torch.Generator(device="cpu").manual_seed(42)
    dummy_input = torch.randn(2, 3, 32, 32, generator=generator, dtype=torch.float32).cuda()
    dummy_t = torch.tensor([0.5, 0.5], dtype=model.net.noise_scheduler.t_precision, device="cuda")
    dummy_labels = torch.zeros(2, 10, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        forward_outputs = {
            "net": model.net(dummy_input, dummy_t, condition=dummy_labels),
            "fake_score": model.fake_score(dummy_input, dummy_t, condition=dummy_labels),
            "teacher": model.teacher(dummy_input, dummy_t, condition=dummy_labels),
        }

    # Check consistency across ranks for all networks
    forward_results = {}
    for name, output in forward_outputs.items():
        assert output.dtype == torch.float32
        if hasattr(output, "full_tensor"):
            output_full = output.full_tensor()
        else:
            output_full = output

        rank0_output = output_full.clone()
        dist.broadcast(rank0_output, src=0)
        diff = _max_across_ranks((output_full - rank0_output).abs().max().item())
        forward_results[name] = {
            "consistent": diff < 1e-5,
            "diff": diff,
        }

    result["forward_results"] = forward_results
    all_forward_consistent = all(r["consistent"] for r in forward_results.values())
    result["all_forward_consistent"] = all_forward_consistent

    result["all_passed"] = (
        has_correct_device_before
        and has_correct_device_after
        and has_correct_storage_dtype_after
        and all_forward_consistent
    )

    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_dmd2_fsdp_context():
    """Test inline FSDP preparation with a multi-network DMD2Model.

    This verifies distinct storage and compute precision for trainable DMD2 networks and its
    frozen teacher in ``extra_fsdp_modules``.
    """
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_dmd2_fsdp_context_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_correct_device_before", False), (
        f"Meta device not used correctly before FSDP: " f"meta_status={result.get('meta_status_before', 'N/A')}"
    )
    assert result.get("has_correct_device_after", False), (
        f"Tensors not properly materialized after FSDP preparation: "
        f"meta_status={result.get('meta_status_after', 'N/A')}"
    )
    assert result.get(
        "has_correct_storage_dtype_after", False
    ), "Trainable FSDP shards did not remain float32 or frozen extras did not remain at model precision"
    assert result.get("all_forward_consistent", False), (
        f"Forward pass inconsistent across ranks: " f"forward_results={result.get('forward_results', 'N/A')}"
    )
    assert result.get("all_passed", False), "DMD2 FSDP-context test failed"

    clear_gpu_memory()


# =============================================================================
# Stock clip_grad_norm_ on FSDP CPU-offload
# =============================================================================
#
# Verifies that the stock ``torch.nn.utils.clip_grad_norm_`` works on
# FSDP-CPU-offloaded DTensors under the dual ``cuda:nccl,cpu:gloo`` backend:
# DTensor reductions on CPU route through gloo automatically, so no bespoke
# clipping helper is needed. Generic collective and module-synchronization
# contracts live in ``tests/test_distributed.py``; the mixed-backend
# default-PG contract remains in ``tests/test_ddp.py``.


def _stock_clip_grad_norm_on_fsdp_cpu_offload(rank: int, world_size: int) -> dict:
    """Tiny FSDP-wrapped model, CPU offload on, run one backward, then call
    stock ``torch.nn.utils.clip_grad_norm_``.

    Contract: under the mixed-backend default PG (``cuda:nccl,cpu:gloo``),
    DTensor reductions on CPU route through gloo, so stock ``clip_grad_norm_``
    returns a finite scalar without raising on a CPU-offloaded DTensor.
    """
    mesh = init_device_mesh("cuda", (world_size,))
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
    ).cuda(rank)

    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

    fully_shard(
        model,
        mesh=mesh,
        offload_policy=CPUOffloadPolicy(),
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )

    x = torch.randn(4, 64, device=f"cuda:{rank}")
    loss = model(x).sum()
    loss.backward()

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    return {"total_norm": float(total_norm)}


@RunIf(min_gpus=2)
def test_stock_clip_grad_norm_on_fsdp_cpu_offload():
    """Stock ``torch.nn.utils.clip_grad_norm_`` works on FSDP CPU-offloaded
    parameters under the dual gloo+nccl backend — no bespoke FSDP grad-clip
    helper required."""
    result = run_distributed_test(
        test_fn=_stock_clip_grad_norm_on_fsdp_cpu_offload,
        world_size=2,
        timeout=120,
    )
    assert result is not None
    tn = result["total_norm"]
    assert math.isfinite(tn)
    assert tn > 0


# ---------------------------------------------------------------------------
# Mixed frozen / trainable parameters inside one FSDP group
# ---------------------------------------------------------------------------
#
# ``MiniMaxH3(freeze_adaln=True)`` holds the per-block AdaLN projections fixed. They live inside
# the per-block FSDP groups next to the trainable ff/attn parameters (they are disjoint from the
# float32 carve-out, which gets its own groups), so a block group becomes partly frozen and partly
# trainable. Everything else frozen in this repo is whole-module -- the teacher, or the fake score
# between updates -- which is what ``patch_fsdp2_mp_dtypes`` covers.

_MIXED_DIM, _MIXED_BLOCKS = 32, 4


class _MixedAdaLNBlock(torch.nn.Module):
    """A block named like the release: ``adaln_proj`` beside the trainable rest."""

    def __init__(self):
        super().__init__()
        self.adaln_proj = torch.nn.ModuleDict({"linear": torch.nn.Linear(_MIXED_DIM, 2 * _MIXED_DIM)})
        self.ff = torch.nn.Linear(_MIXED_DIM, _MIXED_DIM)

    def forward(self, x):
        shift, scale = self.adaln_proj["linear"](x).chunk(2, dim=-1)
        return self.ff(x * (1 + scale) + shift)


class _MixedAdaLNNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_MixedAdaLNBlock() for _ in range(_MIXED_BLOCKS)])
        self.proj_out = torch.nn.Linear(_MIXED_DIM, _MIXED_DIM)

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return self.proj_out(x)


def _freeze_mixed_adaln(model: torch.nn.Module) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        if "adaln" in name:
            param.requires_grad_(False)
            frozen += 1
    return frozen


def _mixed_requires_grad_impl(rank: int, world_size: int, **kwargs):
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

    from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing

    checkpointing = kwargs.get("checkpointing", False)
    cpu_offload = kwargs.get("cpu_offload", False)
    steps = kwargs.get("steps", 1)
    toggle = kwargs.get("toggle", False)

    torch.manual_seed(0)
    device_mesh = init_device_mesh("cuda", (world_size,))
    model = _MixedAdaLNNet().cuda()
    n_frozen = _freeze_mixed_adaln(model)

    # ``MiniMaxH3.fully_shard`` checkpoints every block, so the frozen AdaLN sits inside a
    # recomputed region.
    if checkpointing:
        apply_fsdp_checkpointing(model, _MixedAdaLNBlock)

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    shard_kwargs = {"mesh": device_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        shard_kwargs["offload_policy"] = CPUOffloadPolicy()
    for block in model.transformer_blocks:
        fully_shard(block, **shard_kwargs)
    fully_shard(model, **shard_kwargs)

    # FSDP2 preserves ``requires_grad`` (``_init_sharded_param`` copies it), so re-applying the freeze here
    # is only defensive; the real guard is MiniMaxH3's ``requires_grad_`` override, modelled by ``toggle``.
    _freeze_mixed_adaln(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-2)

    def _local(t):
        return t.to_local() if hasattr(t, "to_local") else t

    before = {n: _local(p.detach()).float().clone() for n, p in model.named_parameters()}

    for _ in range(steps):
        # DMD2's ``_setup_grad_requirements`` freezes the whole fake score on student iterations
        # and re-enables it on fake-score ones; the freeze has to survive that round trip.
        if toggle:
            model.requires_grad_(False)
            model.requires_grad_(True)
            _freeze_mixed_adaln(model)
        opt.zero_grad(set_to_none=True)
        model(torch.randn(2, _MIXED_DIM, device="cuda")).square().mean().backward()
        opt.step()

    moved = [n for n, p in model.named_parameters() if not torch.allclose(_local(p.detach()).float(), before[n])]
    return {
        "n_frozen": n_frozen,
        "n_trainable_in_opt": len(trainable),
        "adaln_with_grad": [n for n, p in model.named_parameters() if "adaln" in n and p.grad is not None],
        "n_other_with_grad": len([n for n, p in model.named_parameters() if "adaln" not in n and p.grad is not None]),
        "adaln_moved": [n for n in moved if "adaln" in n],
        "n_other_moved": len([n for n in moved if "adaln" not in n]),
    }


def _assert_mixed_requires_grad_clean(result):
    assert result is not None, "distributed test returned nothing (a rank likely crashed)"
    assert result["adaln_with_grad"] == [], f"frozen AdaLN received grads: {result['adaln_with_grad']}"
    assert result["adaln_moved"] == [], f"frozen AdaLN was updated: {result['adaln_moved']}"
    assert result["n_other_with_grad"] > 0, "no trainable parameter received a gradient"
    assert result["n_other_moved"] > 0, "no trainable parameter was updated"
    assert result["n_frozen"] == 2 * _MIXED_BLOCKS, f"expected {2 * _MIXED_BLOCKS} frozen tensors"
    assert result["n_trainable_in_opt"] == 2 * _MIXED_BLOCKS + 2, "optimizer param count is wrong"


@RunIf(min_gpus=2)
def test_fsdp_mixed_requires_grad_group():
    """A group holding both frozen AdaLN and trainable ff must shard, backward and step."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_mixed_requires_grad_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )
    _assert_mixed_requires_grad_clean(result)


@RunIf(min_gpus=2)
def test_fsdp_mixed_requires_grad_full_stack():
    """Everything the H3 run enables at once: CPU offload, activation checkpointing, several
    steps, and DMD2's per-iteration requires_grad toggle."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_mixed_requires_grad_impl,
        world_size=2,
        setup_fn=set_env_vars,
        cpu_offload=True,
        checkpointing=True,
        steps=3,
        toggle=True,
    )
    _assert_mixed_requires_grad_clean(result)


def _tied_hf_encoder_meta_sync_impl(rank: int, world_size: int, **kwargs):
    """A wrapped HF causal LM with tied embeddings and a non-persistent rotary table, meta-built off rank 0."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from fastgen.utils.distributed import fsdp as fsdp_utils_module

    class TiedEncoder(FastGenModule):
        def __init__(self):
            super().__init__()
            self.model = Qwen2ForCausalLM(
                Qwen2Config(
                    vocab_size=64,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=64,
                    tie_word_embeddings=True,
                )
            )

    torch.manual_seed(0)
    with torch.device("meta") if rank else contextlib.nullcontext():
        module = TiedEncoder()
    assert module.model.lm_head.weight is module.model.model.embed_tokens.weight
    assert "inv_freq" in module.model.model.rotary_emb._non_persistent_buffers_set
    reference = {name: value.detach().clone() for name, value in module.state_dict().items()} if rank == 0 else None
    reference_inv_freq = module.model.model.rotary_emb.inv_freq.detach().clone().cuda() if rank == 0 else None

    context = fsdp_utils_module.FSDPContext(
        mesh=init_device_mesh("cuda", (world_size,)),
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=None,
        cpu_offload=False,
    )
    module.prepare_module(device=None, precision_model=torch.float32, fsdp_context=context)

    tied_after = module.model.lm_head.weight is module.model.model.embed_tokens.weight
    inv_freq = module.model.model.rotary_emb.inv_freq
    expected_inv_freq = reference_inv_freq if rank == 0 else torch.empty_like(inv_freq)
    dist.broadcast(expected_inv_freq, src=0)
    inv_freq_matches = not inv_freq.is_meta and torch.equal(inv_freq, expected_inv_freq)

    gathered = get_model_state_dict(module, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    mismatched = []
    if rank == 0:
        mismatched = [
            name for name, value in reference.items() if name not in gathered or not torch.equal(gathered[name], value)
        ]
    return {"tied_after": tied_after, "inv_freq_matches": inv_freq_matches, "mismatched": mismatched}


@RunIf(min_gpus=2)
def test_fsdp_meta_sync_preserves_hf_ties_and_nonpersistent_buffers():
    """The no-reset path: HF encoders lack ``reset_parameters``, so meta ranks must receive rank 0's
    non-persistent buffers by broadcast, keep the embedding tie, and load every weight."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_tied_hf_encoder_meta_sync_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )
    assert result is not None
    assert result["tied_after"]
    assert result["inv_freq_matches"]
    assert result["mismatched"] == []


# ---------------------------------------------------------------------------
# Reverse split-precision FSDP boundary
#
# The root computes in FP32 but stores ordinary parameters in BF16, while an explicit fixed
# precision island retains FP32 storage. The island still requires an independent FSDP group.
# ---------------------------------------------------------------------------


class _ReverseSplitPrecisionNet(FastGenModule):
    """FP32 compute with BF16 default storage and one explicitly FP32 storage island."""

    def __init__(self):
        super().__init__()
        self.default_layer = torch.nn.Linear(8, 8)
        self.fp32_layer = torch.nn.Linear(8, 8)
        self.fp32_boundary_created = False
        self.last_compute_dtypes = None

    def _fixed_precision_overrides(self):
        return {
            "fp32_layer.weight": torch.float32,
            "fp32_layer.bias": torch.float32,
        }

    def fully_shard(self, **kwargs):
        fp32_kwargs = fsdp_utils.make_fully_shard_kwargs(
            kwargs,
            param_dtype=torch.float32,
            storage_dtype=self.precision_opt,
            enabled=True,
        )
        self.fp32_boundary_created = fp32_kwargs is not None
        if fp32_kwargs is not None:
            fully_shard(self.fp32_layer, **fp32_kwargs)
        fully_shard(self, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        default = self.default_layer(x.to(dtype=self.precision_model))
        fp32 = self.fp32_layer(x.to(dtype=torch.float32))
        self.last_compute_dtypes = (default.dtype, fp32.dtype)
        return default + fp32


# ---------------------------------------------------------------------------
# Split model/optimization precision through ``FastGenModel.on_train_begin`` + FSDP
#
# Method state stays fp32 while the network computes in bfloat16 and keeps fp32 trainable
# storage; covers real and meta initialization, CPU offload, ``reset_parameters`` on
# fixed-precision buffers, and the cross-rank state checks that guard meta materialization.
# ---------------------------------------------------------------------------


class _TinyPrecisionNet(FastGenModule):
    """Tiny mixed-storage component whose forward records the real compute dtype."""

    def __init__(self):
        super().__init__()
        self.trainable_layer = torch.nn.Linear(8, 8)
        self.frozen_layer = torch.nn.Linear(8, 8)
        self.frozen_layer.requires_grad_(False)
        self.register_buffer("scale", torch.ones(8))
        self.last_compute_dtypes = None
        # The model treats ``net`` as the preprocessor host, like ``FastGenNetwork``.
        self.preprocessors = {}

    def requires_grad_(self, requires_grad: bool = True):
        # A permanently frozen part (like LTX2's connectors) survives the generic trainable toggle.
        super().requires_grad_(requires_grad)
        self.frozen_layer.requires_grad_(False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trainable = self.trainable_layer(x.to(dtype=self.precision_model))
        frozen = self.frozen_layer(x.to(dtype=self.precision_model))
        self.last_compute_dtypes = (trainable.dtype, frozen.dtype)
        output = (trainable + frozen) * self.scale
        return output.to(dtype=x.dtype)


class _TinyDeferredPreprocessor(FastGenModule):
    """Late preprocessor that records whether preparation received meta state."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.meta_before_prepare = None

    def prepare_module(self, *args, **kwargs):
        self.meta_before_prepare = any(parameter.is_meta for parameter in self.parameters())
        return super().prepare_module(*args, **kwargs)

    def forward(self, value):
        return self.proj(value)


class _TinyLatePreprocessorHost(_TinyPrecisionNet):
    """Construct rank-0-real/meta-peer state when lazy preprocessors are requested."""

    def __init__(self, rank: int):
        super().__init__()
        self._construction_rank = rank

    @staticmethod
    def reference_weight():
        return torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16

    def init_preprocessors(self):
        if "text_encoder" in self.preprocessors:
            return
        context = torch.device("meta") if self._construction_rank != 0 else contextlib.nullcontext()
        with context:
            text_encoder = _TinyDeferredPreprocessor()
        if self._construction_rank == 0:
            with torch.no_grad():
                text_encoder.proj.weight.copy_(self.reference_weight())
        self.preprocessors["text_encoder"] = text_encoder


class _ResetBufferPrecisionNet(FastGenModule):
    """Meta-init component whose reset hook rebuilds both kinds of buffer."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(4))
        self.register_buffer("persistent_state", torch.empty(4))
        self.register_buffer("nonpersistent_state", torch.empty(4), persistent=False)
        # The model treats ``net`` as the preprocessor host, like ``FastGenNetwork``.
        self.preprocessors = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Reconstruct values in the allocated storage without changing identity or precision.
        self.persistent_state.fill_(2.0)
        self.nonpersistent_state.fill_(3.0)

    def _fixed_precision_overrides(self):
        return {
            "persistent_state": torch.float64,
            "nonpersistent_state": torch.float64,
        }

    def forward(self, x):
        return x


class _UnresetBufferPrecisionNet(FastGenModule):
    """Meta-init component whose non-persistent state must come from rank zero."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4,), 11.0))
        self.register_buffer("persistent_state", torch.full((4,), 17.0))
        self.register_buffer("nonpersistent_state", torch.full((4,), 23.0), persistent=False)
        self.preprocessors = {}

    def _fixed_precision_overrides(self):
        return {
            "persistent_state": torch.float64,
            "nonpersistent_state": torch.float64,
        }

    def forward(self, x):
        return x


class _InvalidResetBufferPrecisionNet(_ResetBufferPrecisionNet):
    """Reset hook that silently discards the fixed-precision buffer policy."""

    def reset_parameters(self) -> None:
        self.persistent_state = torch.ones(
            self.persistent_state.shape,
            dtype=torch.float32,
            device=self.persistent_state.device,
        )
        self.nonpersistent_state.copy_(torch.full_like(self.nonpersistent_state, 3.0))


class _RankDependentResetBufferPrecisionNet(_ResetBufferPrecisionNet):
    """Reset hook whose non-persistent values differ across ranks."""

    def reset_parameters(self) -> None:
        self.persistent_state.fill_(2.0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.nonpersistent_state.fill_(3.0 + rank)


class _TinyPrecisionModel(FastGenModel):
    @property
    def model_dict(self):
        return torch.nn.ModuleDict({"net": self.net})

    def _get_outputs(self, *args, **kwargs):
        raise NotImplementedError

    def single_train_step(self, data, iteration):
        del iteration
        prediction = self.net(data["x"])
        loss = (prediction - data["target"]).square().mean()
        return {"total_loss": loss}, {"prediction": prediction}


def _uninitialized_precision_model(rank: int, **precisions) -> _TinyPrecisionModel:
    model = _TinyPrecisionModel.__new__(_TinyPrecisionModel)
    torch.nn.Module.__init__(model)
    model.module_dict = {}
    model.config = SimpleNamespace(
        ddp_find_unused_parameters=False,
        inference=False,
        enable_preprocessors=False,
        extra_fsdp_modules=[],
        net_adapter=SimpleNamespace(type=None),
        autocast_cache_enabled=False,
    )
    model.device = torch.device("cuda", rank)
    model._resolve_precisions(precision="float32", precision_model="bfloat16", precision_opt="float32", **precisions)
    model.use_ema = []
    return model


def _make_split_precision_model(rank: int, meta_init: bool, cpu_offload: bool = False) -> _TinyPrecisionModel:
    model = _uninitialized_precision_model(
        rank, precision_amp="bfloat16", precision_amp_infer="bfloat16", precision_fsdp_reduce="bfloat16"
    )
    torch.manual_seed(1234)
    context = torch.device("meta") if meta_init and rank != 0 else contextlib.nullcontext()
    with context:
        model.net = _TinyPrecisionNet()
    model.module_dict.update(model.model_dict)
    model._meta_before_prepare = any(parameter.is_meta for parameter in model.net.parameters())
    if not model._meta_before_prepare:
        with torch.no_grad():
            for parameter in model.net.parameters():
                parameter.fill_(0.05)
            model.net.scale.fill_(1.0)
    fsdp_context = fsdp_utils.create_fsdp_context(
        reduce_dtype=model.precision_fsdp_reduce, min_num_params=1, apply_cpu_offload=cpu_offload
    )
    model.on_train_begin(fsdp_context=fsdp_context)
    return model


def _make_reset_buffer_model(
    rank: int,
    network_type: type[_ResetBufferPrecisionNet] = _ResetBufferPrecisionNet,
    cpu_offload: bool = False,
) -> _TinyPrecisionModel:
    model = _uninitialized_precision_model(rank, precision_fsdp_reduce="bfloat16")
    context = torch.device("meta") if rank != 0 else contextlib.nullcontext()
    with context:
        model.net = network_type()
    model.module_dict.update(model.model_dict)
    model._reset_state_keys_before = set(model.net.state_dict())
    model._meta_before_prepare = any(tensor.is_meta for tensor in (*model.net.parameters(), *model.net.buffers()))
    if rank == 0:
        with torch.no_grad():
            model.net.weight.fill_(11.0)
            # The persistent value must be restored from rank 0; the non-persistent value is
            # broadcast from rank 0, overwriting ``reset_parameters`` on the meta ranks.
            model.net.persistent_state.fill_(17.0)
    fsdp_context = fsdp_utils.create_fsdp_context(
        reduce_dtype=model.precision_fsdp_reduce, min_num_params=1, apply_cpu_offload=cpu_offload
    )
    model.on_train_begin(fsdp_context=fsdp_context)
    return model


def _all_true(flag: bool) -> bool:
    probe = torch.tensor(int(flag), dtype=torch.int64)
    dist.all_reduce(probe, op=dist.ReduceOp.MIN)
    return bool(probe.item())


def _max_across_ranks(value: float) -> float:
    probe = torch.tensor(value, dtype=torch.float64)
    dist.all_reduce(probe, op=dist.ReduceOp.MAX)
    return probe.item()


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _fsdp_reverse_split_precision_impl(rank: int, world_size: int) -> dict:
    del rank, world_size
    # The 1e-6 comparisons require full FP32 matmuls, even when the image enables TF32 by default.
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(1234)
    module = _ReverseSplitPrecisionNet()
    reference_default = copy.deepcopy(module.default_layer).bfloat16().double()
    reference_fp32 = copy.deepcopy(module.fp32_layer).double()
    context = fsdp_utils.create_fsdp_context(
        reduce_dtype=torch.float32,
        min_num_params=1,
    )
    module.prepare_module(
        device=torch.device("cuda", torch.cuda.current_device()),
        precision_model=torch.float32,
        precision_opt=torch.bfloat16,
        trainable=True,
        fsdp_context=context,
    )

    storage_ok = all(_local(parameter).dtype == torch.bfloat16 for parameter in module.default_layer.parameters())
    storage_ok = storage_ok and all(
        _local(parameter).dtype == torch.float32 for parameter in module.fp32_layer.parameters()
    )
    sharded = isinstance(module, fsdp_utils.FSDPModule) and isinstance(module.fp32_layer, fsdp_utils.FSDPModule)

    # Build the same input independently of per-device CUDA RNG streams, then
    # broadcast it so this remains a test of sharding rather than fixture RNG.
    generator = torch.Generator(device="cpu").manual_seed(9000)
    inputs = torch.randn(4, 8, generator=generator, dtype=torch.float32).to(
        device=torch.device("cuda", torch.cuda.current_device())
    )
    dist.broadcast(inputs, src=0)
    output = module(inputs)
    with torch.no_grad():
        reference_inputs = inputs.cpu().double()
        expected = reference_default(reference_inputs) + reference_fp32(reference_inputs)
    torch.testing.assert_close(output.detach().cpu().double(), expected, rtol=0, atol=1e-6)
    output.square().mean().backward()

    grads = (module.default_layer.weight.grad, module.fp32_layer.weight.grad)
    grad_ok = all(
        grad is not None
        and bool(torch.isfinite(_local(grad)).all().item())
        and float(_local(grad).float().norm().item()) > 0
        for grad in grads
    )
    comparable_output = output.detach()
    if hasattr(comparable_output, "full_tensor"):
        comparable_output = comparable_output.full_tensor()
    rank0_output = comparable_output.clone()
    dist.broadcast(rank0_output, src=0)
    rank_consistent = torch.allclose(comparable_output, rank0_output, rtol=0, atol=1e-6)

    return {
        "boundary_created": _all_true(module.fp32_boundary_created),
        "storage_ok": _all_true(storage_ok),
        "sharded": _all_true(sharded),
        "compute_ok": _all_true(module.last_compute_dtypes == (torch.float32, torch.float32)),
        "grad_ok": _all_true(grad_ok),
        "rank_consistent": _all_true(rank_consistent),
    }


@RunIf(min_gpus=2)
def test_fsdp_fp32_compute_bf16_storage_preserves_fp32_island():
    """An FP32 island still needs its own group when the enclosing storage is BF16."""
    result = run_distributed_test(
        test_fn=_fsdp_reverse_split_precision_impl,
        world_size=2,
        timeout=240,
    )
    assert result is not None
    assert all(result.values()), result


def _fsdp_split_precision_impl(rank: int, world_size: int, meta_init: bool, cpu_offload: bool) -> dict:
    del world_size
    model = _make_split_precision_model(rank, meta_init=meta_init, cpu_offload=cpu_offload)
    meta_before = model._meta_before_prepare
    expected_meta_before = meta_init and rank != 0
    parameters = list(model.net.parameters())
    local_parameters = [_local(parameter) for parameter in parameters]
    # A sharded trainable component stores all floating state in the optimization precision.
    storage_ok = all(parameter.dtype == torch.float32 for parameter in parameters)
    storage_ok = storage_ok and model.net.scale.dtype == torch.float32
    materialized = all(not parameter.is_meta for parameter in local_parameters)
    placement_ok = all(parameter.device.type == ("cpu" if cpu_offload else "cuda") for parameter in local_parameters)
    buffer_placement_ok = model.net.scale.device == model.device
    sharded = hasattr(model.net, "set_requires_gradient_sync")
    precision_ok = model.net.precision_model == torch.bfloat16

    generator = torch.Generator(device=model.device).manual_seed(9000)
    data = {
        "x": torch.randn(4, 8, generator=generator, device=model.device, dtype=torch.float32),
        "target": torch.zeros(4, 8, device=model.device, dtype=torch.float32),
    }
    with model.autocast():
        loss_map, outputs = model.single_train_step(data, iteration=0)
    loss_map["total_loss"].backward()

    grad = model.net.trainable_layer.weight.grad
    local_grad = _local(grad) if grad is not None else None
    grad_ok = (
        local_grad is not None
        and local_grad.dtype == torch.float32
        and bool(torch.isfinite(local_grad).all().item())
        and float(local_grad.float().norm().item()) > 0
    )
    frozen_ok = all(parameter.grad is None for parameter in model.net.frozen_layer.parameters())
    compute_ok = model.net.last_compute_dtypes == (torch.bfloat16, torch.bfloat16)
    output = outputs["prediction"].detach()
    output_ok = output.dtype == torch.float32
    rank0_output = output.clone()
    dist.broadcast(rank0_output, src=0)
    rank_consistent = float((output - rank0_output).abs().max().item()) < 1e-6

    return {
        "meta_before_ok": _all_true(meta_before == expected_meta_before),
        "storage_ok": _all_true(storage_ok),
        "materialized": _all_true(materialized),
        "placement_ok": _all_true(placement_ok),
        "buffer_placement_ok": _all_true(buffer_placement_ok),
        "sharded": _all_true(sharded),
        "precision_ok": _all_true(precision_ok),
        "compute_ok": _all_true(compute_ok),
        "output_ok": _all_true(output_ok),
        "rank_consistent": _all_true(rank_consistent),
        "grad_ok": _all_true(grad_ok),
        "frozen_ok": _all_true(frozen_ok),
    }


def _run_fsdp_split_precision_case(meta_init: bool, cpu_offload: bool) -> None:
    result = run_distributed_test(
        test_fn=_fsdp_split_precision_impl,
        world_size=2,
        timeout=240,
        meta_init=meta_init,
        cpu_offload=cpu_offload,
    )
    assert result is not None
    assert all(result.values()), result


def _fsdp_late_meta_preprocessor_impl(rank: int, world_size: int) -> dict:
    from torch.distributed.tensor import DTensor

    torch.manual_seed(1234)
    model = _uninitialized_precision_model(rank, precision_fsdp_reduce="bfloat16")
    model.config.enable_preprocessors = True
    model.config.extra_fsdp_modules = ["text_encoder"]
    model.net = _TinyLatePreprocessorHost(rank)
    model.module_dict.update(model.model_dict)

    fsdp_context = fsdp_utils.create_fsdp_context(reduce_dtype=model.precision_fsdp_reduce, min_num_params=1)
    model.on_train_begin(fsdp_context=fsdp_context)

    text_encoder = model.net_preprocessors.preprocessors["text_encoder"]
    weight = text_encoder.proj.weight
    local_weight = weight.to_local() if isinstance(weight, DTensor) else weight
    inputs = torch.eye(4, device=model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        output = text_encoder(inputs)
    expected = (
        inputs
        @ _TinyLatePreprocessorHost.reference_weight()
        .to(
            device=model.device,
            dtype=torch.bfloat16,
        )
        .T
    )

    local_result = {
        "meta_before_prepare": text_encoder.meta_before_prepare,
        "registered": model.module_dict["text_encoder"] is text_encoder,
        "is_fsdp": text_encoder._is_fsdp and isinstance(text_encoder, fsdp_utils.FSDPModule),
        "weight_is_dtensor": isinstance(weight, DTensor),
        "local_weight_materialized": not local_weight.is_meta,
        "local_weight_device": local_weight.device.type,
        "output_error": float((output - expected).abs().max().item()),
    }
    per_rank = [None] * world_size
    dist.all_gather_object(per_rank, local_result)
    return {"per_rank": per_rank}


@RunIf(min_gpus=2)
def test_fsdp_late_preprocessor_keeps_meta_peers_until_outer_preparation():
    """Outer preparation shards a late rank-0-real/meta-peer preprocessor directly."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_fsdp_late_meta_preprocessor_impl,
        world_size=2,
        timeout=240,
        setup_fn=set_env_vars,
    )
    assert result is not None
    for rank, entry in enumerate(result["per_rank"]):
        assert entry["meta_before_prepare"] == (rank != 0), (rank, entry)
        assert entry["registered"], (rank, entry)
        assert entry["is_fsdp"], (rank, entry)
        assert entry["weight_is_dtensor"], (rank, entry)
        assert entry["local_weight_materialized"], (rank, entry)
        assert entry["local_weight_device"] == "cuda", (rank, entry)
        assert entry["output_error"] == 0.0, (rank, entry)
    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_split_precision_real_init_forward_backward():
    _run_fsdp_split_precision_case(meta_init=False, cpu_offload=False)


@RunIf(min_gpus=2)
def test_fsdp_split_precision_real_cpu_offload_forward_backward():
    _run_fsdp_split_precision_case(meta_init=False, cpu_offload=True)


@RunIf(min_gpus=2)
def test_fsdp_split_precision_meta_init_forward_backward():
    _run_fsdp_split_precision_case(meta_init=True, cpu_offload=False)


@RunIf(min_gpus=2)
def test_fsdp_split_precision_meta_cpu_offload_forward_backward():
    _run_fsdp_split_precision_case(meta_init=True, cpu_offload=True)


def _fsdp_meta_reset_buffers_impl(rank: int, world_size: int, cpu_offload: bool) -> dict:
    del world_size
    model = _make_reset_buffer_model(rank, cpu_offload=cpu_offload)
    state_keys = model._reset_state_keys_before
    meta_before = model._meta_before_prepare

    parameter = _local(model.net.weight)
    persistent_state = _local(model.net.persistent_state)
    nonpersistent_state = _local(model.net.nonpersistent_state)
    rank0_persistent = persistent_state.clone()
    rank0_nonpersistent = nonpersistent_state.clone()
    dist.broadcast(rank0_persistent, src=0)
    dist.broadcast(rank0_nonpersistent, src=0)

    return {
        "meta_before_ok": _all_true(meta_before == (rank != 0)),
        "state_dict_membership_ok": _all_true(
            "persistent_state" in state_keys and "nonpersistent_state" not in state_keys
        ),
        "materialized": _all_true(not persistent_state.is_meta and not nonpersistent_state.is_meta),
        "fixed_dtypes_ok": _all_true(
            persistent_state.dtype == torch.float64 and nonpersistent_state.dtype == torch.float64
        ),
        "parameter_device_ok": _all_true(parameter.device.type == ("cpu" if cpu_offload else "cuda")),
        "buffer_devices_ok": _all_true(
            persistent_state.device == model.device and nonpersistent_state.device == model.device
        ),
        "persistent_value_restored": _all_true(torch.equal(persistent_state, torch.full_like(persistent_state, 17.0))),
        "nonpersistent_value_from_rank0": _all_true(
            torch.equal(nonpersistent_state, torch.full_like(nonpersistent_state, 3.0))
        ),
        "rank_agreement": _all_true(
            torch.equal(persistent_state, rank0_persistent) and torch.equal(nonpersistent_state, rank0_nonpersistent)
        ),
    }


@pytest.mark.parametrize("cpu_offload", [False, True])
@RunIf(min_gpus=2)
def test_meta_init_reset_buffers_fixed_precision(cpu_offload):
    result = run_distributed_test(
        test_fn=_fsdp_meta_reset_buffers_impl,
        world_size=2,
        timeout=240,
        cpu_offload=cpu_offload,
    )
    assert result is not None
    assert all(result.values()), result


def _fsdp_meta_unreset_buffers_impl(rank: int, world_size: int, cpu_offload: bool) -> dict:
    del world_size
    model = _make_reset_buffer_model(
        rank,
        network_type=_UnresetBufferPrecisionNet,
        cpu_offload=cpu_offload,
    )
    persistent_state = _local(model.net.persistent_state)
    nonpersistent_state = _local(model.net.nonpersistent_state)
    rank0_nonpersistent = nonpersistent_state.clone()
    dist.broadcast(rank0_nonpersistent, src=0)

    return {
        "meta_before_ok": _all_true(model._meta_before_prepare == (rank != 0)),
        "materialized": _all_true(not nonpersistent_state.is_meta),
        "fixed_dtype_ok": _all_true(nonpersistent_state.dtype == torch.float64),
        "buffer_devices_ok": _all_true(
            persistent_state.device == model.device and nonpersistent_state.device == model.device
        ),
        "source_value_broadcast": _all_true(
            torch.equal(nonpersistent_state, torch.full_like(nonpersistent_state, 23.0))
        ),
        "rank_agreement": _all_true(torch.equal(nonpersistent_state, rank0_nonpersistent)),
    }


@pytest.mark.parametrize("cpu_offload", [False, True])
@RunIf(min_gpus=2)
def test_meta_init_broadcasts_nonpersistent_buffers_without_reset(cpu_offload):
    result = run_distributed_test(
        test_fn=_fsdp_meta_unreset_buffers_impl,
        world_size=2,
        timeout=240,
        cpu_offload=cpu_offload,
    )
    assert result is not None
    assert all(result.values()), result


def _fsdp_rank0_nonpersistent_buffer_impl(rank: int, world_size: int, cpu_offload: bool) -> dict:
    del world_size
    model = _make_reset_buffer_model(
        rank,
        network_type=_RankDependentResetBufferPrecisionNet,
        cpu_offload=cpu_offload,
    )
    nonpersistent_state = _local(model.net.nonpersistent_state)
    return {"rank0_value_wins": _all_true(torch.equal(nonpersistent_state, torch.full_like(nonpersistent_state, 3.0)))}


@pytest.mark.parametrize("cpu_offload", [False, True])
@RunIf(min_gpus=2)
def test_fsdp_rank0_nonpersistent_buffers_override_rank_local_reset(cpu_offload):
    result = run_distributed_test(
        test_fn=_fsdp_rank0_nonpersistent_buffer_impl,
        world_size=2,
        timeout=240,
        cpu_offload=cpu_offload,
    )
    assert result is not None
    assert all(result.values()), result


def _fsdp_rejects_reset_buffer_dtype_mismatch_impl(rank: int, world_size: int) -> dict:
    del world_size
    model = _uninitialized_precision_model(rank, precision_fsdp_reduce="bfloat16")
    context = torch.device("meta") if rank != 0 else contextlib.nullcontext()
    with context:
        model.net = _InvalidResetBufferPrecisionNet()
    model.module_dict.update(model.model_dict)

    fsdp_context = fsdp_utils.create_fsdp_context(
        reduce_dtype=model.precision_fsdp_reduce, min_num_params=1, apply_cpu_offload=False
    )
    try:
        model.on_train_begin(fsdp_context=fsdp_context)
    except RuntimeError as error:
        message = str(error)
        rejected = "FSDP state synchronization" in message and "changed registered state" in message
    else:
        rejected = False

    return {"rejected_collectively": _all_true(rejected)}


@RunIf(min_gpus=2)
def test_fsdp_rejects_reset_buffer_dtype_mismatch_before_state_load():
    result = run_distributed_test(
        test_fn=_fsdp_rejects_reset_buffer_dtype_mismatch_impl,
        world_size=2,
        timeout=240,
    )
    assert result is not None
    assert result["rejected_collectively"]


def _unsharded_root_checkpoint_impl(rank: int, world_size: int, checkpoint_dir: str) -> Dict:
    """Checkpointing reshards a root that a forward left unsharded, as with Wan's ``reshard_after_forward=False``."""
    from fastgen.utils.checkpointer import Checkpointer
    from tests.test_checkpointing import _get_checkpointer_config

    del rank
    mesh = init_device_mesh("cuda", (world_size,))
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).cuda()
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    expected = [parameter.full_tensor().clone() for parameter in model.parameters()]
    inputs = torch.ones(2, 4, device="cuda")
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=True)

    with torch.no_grad():
        model(inputs)
    path = checkpointer.save(torch.nn.ModuleDict({"net": model}), iteration=3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model(inputs)
    iteration = checkpointer.load(torch.nn.ModuleDict({"net": model}), path=path)
    restored = all(
        torch.equal(parameter.full_tensor(), value) for parameter, value in zip(model.parameters(), expected)
    )
    return {"loaded": _all_true(iteration == 3 and restored)}


@RunIf(min_gpus=2)
def test_checkpointing_reshards_roots_left_unsharded_by_a_forward(tmp_path):
    result = run_distributed_test(
        test_fn=_unsharded_root_checkpoint_impl, world_size=2, timeout=240, checkpoint_dir=str(tmp_path)
    )
    assert result is not None
    assert result["loaded"]
