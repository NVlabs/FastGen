# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical regression tests for FastGen's FSDP2 workarounds.

The integration tests deliberately keep three independent oracles:

* a non-sharded model over the union of every rank's microbatches;
* stock FSDP2, with the FastGen monkey patch disabled;
* FSDP2 with the FastGen patch installed.

This makes the tests fail if a workaround stops fixing its upstream bug, while
also making an upstream fix visible so the corresponding patch can be removed.
"""

from __future__ import annotations

import contextlib
import copy
from types import SimpleNamespace
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
from torch.distributed.tensor import DTensor

from fastgen.methods.model import FastGenModel
from fastgen.networks.network import FastGenModule
import fastgen.utils.distributed.fsdp as fsdp_utils
from fastgen.utils.basic_utils import clear_gpu_memory
from fastgen.utils.env import set_env_vars
from tests.utils import RunIf, run_distributed_test


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _placement_names(tensor: torch.Tensor) -> tuple[str, ...]:
    return tuple(type(placement).__name__ for placement in getattr(tensor, "placements", ()))


def _dtype_name(dtype: torch.dtype | None) -> str | None:
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _all_rank_results(local_result: dict, world_size: int) -> list[dict]:
    results = [None] * world_size
    dist.all_gather_object(results, local_result)
    return results


@pytest.mark.parametrize(
    ("param_dtypes", "expected_orig", "expected_reduce"),
    [
        (
            [(torch.float32, torch.bfloat16), (torch.float32, torch.bfloat16)],
            torch.float32,
            torch.bfloat16,
        ),
        (
            [(torch.float32, torch.bfloat16), (torch.float32, torch.float32)],
            torch.float32,
            None,
        ),
        (
            [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.bfloat16)],
            None,
            torch.bfloat16,
        ),
    ],
)
def test_mp_dtype_patch_resolves_master_and_reduce_dtypes_independently(
    monkeypatch,
    param_dtypes,
    expected_orig,
    expected_reduce,
):
    """A mixed field must not prevent the other uniform field from being recovered."""

    def upstream_init(group):
        group._orig_dtype = None
        group._reduce_dtype = None
        for parameter, (orig_dtype, reduce_dtype) in zip(group.fsdp_params, param_dtypes):
            parameter.orig_dtype = orig_dtype
            parameter.reduce_dtype = reduce_dtype

    monkeypatch.setattr(FSDPParamGroup, "_init_mp_dtypes", upstream_init)
    assert fsdp_utils.patch_fsdp2_mp_dtypes()

    group = SimpleNamespace(
        fsdp_params=[SimpleNamespace() for _ in param_dtypes],
        _module_fqn="mixed_precision_group",
    )
    FSDPParamGroup._init_mp_dtypes(group)

    assert group._orig_dtype is expected_orig
    assert group._reduce_dtype is expected_reduce


def test_mp_dtype_patch_preserves_upstream_trainable_group_dtypes(monkeypatch):
    """The fallback is inert when upstream initialized a normally trainable group."""

    def upstream_init(group):
        group._orig_dtype = torch.float64
        group._reduce_dtype = torch.float32
        for parameter in group.fsdp_params:
            parameter.orig_dtype = torch.bfloat16
            parameter.reduce_dtype = torch.bfloat16

    monkeypatch.setattr(FSDPParamGroup, "_init_mp_dtypes", upstream_init)
    assert fsdp_utils.patch_fsdp2_mp_dtypes()

    group = SimpleNamespace(fsdp_params=[SimpleNamespace()], _module_fqn="active_group")
    FSDPParamGroup._init_mp_dtypes(group)

    assert group._orig_dtype is torch.float64
    assert group._reduce_dtype is torch.float32


class _RecordingLinear(torch.nn.Linear):
    """Linear layer that records the materialized parameter dtype seen by its forward."""

    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype):
        super().__init__(in_features, out_features, dtype=dtype)
        self.forward_weight_dtype: torch.dtype | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_weight_dtype = self.weight.dtype
        return super().forward(inputs)


class _MixedPrecisionPatchNet(torch.nn.Module):
    """Small network with a permanently frozen branch and an optional FP32 island."""

    def __init__(self, master_dtype: torch.dtype, fp32_carveout: bool):
        super().__init__()
        self.input = _RecordingLinear(5, 8, dtype=master_dtype)
        self.frozen = _RecordingLinear(8, 8, dtype=master_dtype)
        self.output = _RecordingLinear(8, 3, dtype=master_dtype)
        # FSDP2 intentionally rejects scalar parameters; a one-element vector
        # keeps this parameter in the root group while preserving the math.
        self.root_scale = torch.nn.Parameter(torch.tensor([0.75], dtype=master_dtype))
        self.carveout = _RecordingLinear(8, 8, dtype=torch.float32) if fp32_carveout else None
        self.forward_root_dtype: torch.dtype | None = None

    def set_roles(self, active: bool) -> None:
        self.requires_grad_(False)
        if active:
            self.input.requires_grad_(True)
            self.output.requires_grad_(True)
            self.root_scale.requires_grad_(True)
            if self.carveout is not None:
                self.carveout.requires_grad_(True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_root_dtype = self.root_scale.dtype
        hidden = torch.nn.functional.silu(self.input(inputs))
        hidden = hidden + 0.125 * self.frozen(hidden)
        if self.carveout is not None:
            # This mirrors the network-owned FP32 islands that remain FP32 even
            # when an outer training autocast context is active.
            with torch.autocast(device_type="cuda", enabled=False):
                carved = self.carveout(hidden.float())
            hidden = hidden + 0.25 * carved.to(dtype=hidden.dtype)
        return self.output(torch.tanh(hidden)) * self.root_scale

    def observed_compute_dtypes(self) -> dict[str, str | None]:
        observed = {
            "input": _dtype_name(self.input.forward_weight_dtype),
            "frozen": _dtype_name(self.frozen.forward_weight_dtype),
            "output": _dtype_name(self.output.forward_weight_dtype),
            "root": _dtype_name(self.forward_root_dtype),
        }
        if self.carveout is not None:
            observed["carveout"] = _dtype_name(self.carveout.forward_weight_dtype)
        return observed


_MP_CASES = {
    # Control: stock and patched FSDP must agree when the group is trainable at lazy init.
    "active_root_fp32_master_bf16_compute_fp32_reduce": {
        "topology": "root",
        "master_dtype": torch.float32,
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.float32,
        "amp_dtype": torch.bfloat16,
        "initially_frozen": False,
        "fp32_carveout": False,
        "grad_rtol": 0.025,
    },
    "frozen_root_fp32_master_bf16_compute_fp32_reduce": {
        "topology": "root",
        "master_dtype": torch.float32,
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.float32,
        "amp_dtype": torch.bfloat16,
        "initially_frozen": True,
        "fp32_carveout": False,
        "grad_rtol": 0.025,
    },
    "frozen_nested_fp32_master_bf16_compute_bf16_reduce": {
        "topology": "nested",
        "master_dtype": torch.float32,
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.bfloat16,
        "amp_dtype": torch.bfloat16,
        "initially_frozen": True,
        "fp32_carveout": False,
        "grad_rtol": 0.04,
    },
    "frozen_auto_bf16_master_bf16_compute_fp32_reduce": {
        "topology": "auto",
        "master_dtype": torch.bfloat16,
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.float32,
        "amp_dtype": torch.bfloat16,
        "initially_frozen": True,
        "fp32_carveout": False,
        "grad_rtol": 0.04,
    },
    "frozen_nested_fp32_master_fp16_compute_fp16_reduce": {
        "topology": "nested",
        "master_dtype": torch.float32,
        "param_dtype": torch.float16,
        "reduce_dtype": torch.float16,
        "amp_dtype": torch.float16,
        "initially_frozen": True,
        "fp32_carveout": False,
        "grad_rtol": 0.025,
    },
    "frozen_nested_bf16_master_with_fp32_master_carveout": {
        "topology": "nested",
        "master_dtype": torch.bfloat16,
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.float32,
        "amp_dtype": torch.bfloat16,
        "initially_frozen": True,
        "fp32_carveout": True,
        "grad_rtol": 0.05,
    },
    # Regression for the former global-only activation gate: every model-level dtype is BF16,
    # but the network-owned carveout still needs FP32 master/compute and BF16 reduction metadata.
    "frozen_nested_all_bf16_with_fp32_master_carveout": {
        "topology": "nested",
        "master_dtype": torch.bfloat16,
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.bfloat16,
        "amp_dtype": torch.bfloat16,
        "initially_frozen": True,
        "fp32_carveout": True,
        "grad_rtol": 0.05,
    },
}


def _autocast_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        return model(inputs)


def _make_mp_inputs(
    rank: int,
    microbatch: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(17_000 + 100 * rank + microbatch)
    return torch.randn(3, 5, generator=generator, device=device, dtype=torch.float32) * 0.25


def _shard_mp_model(
    model: _MixedPrecisionPatchNet,
    mesh,
    case: dict,
) -> None:
    policy = MixedPrecisionPolicy(
        param_dtype=case["param_dtype"],
        reduce_dtype=case["reduce_dtype"],
        output_dtype=None,
        cast_forward_inputs=False,
    )
    topology = case["topology"]
    if topology == "root":
        fully_shard(model, mesh=mesh, mp_policy=policy)
        return
    if topology == "auto":
        # _MixedPrecisionPatchNet intentionally has no fully_shard method.
        fsdp_utils._fully_shard_module(model, min_num_params=1, mesh=mesh, mp_policy=policy)
        return
    if topology != "nested":
        raise AssertionError(f"Unknown test topology: {topology}")

    for child in (model.input, model.frozen, model.output):
        fully_shard(child, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    if model.carveout is not None:
        carveout_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=case["reduce_dtype"],
            output_dtype=None,
            cast_forward_inputs=False,
        )
        fully_shard(model.carveout, mesh=mesh, mp_policy=carveout_policy, reshard_after_forward=True)
    # The direct root_scale parameter gets the default policy. Like FastGen's
    # network roots, it deliberately retains parameters until backward or an
    # explicit phase-boundary reshard.
    fully_shard(model, mesh=mesh, mp_policy=policy)


def _group_dtype_records(model: torch.nn.Module) -> list[dict]:
    records = []
    seen = set()
    for module in model.modules():
        state = _get_module_fsdp_state(module)
        group = None if state is None else state._fsdp_param_group
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))
        per_param_orig = {parameter.orig_dtype for parameter in group.fsdp_params}
        per_param_reduce = {parameter.reduce_dtype for parameter in group.fsdp_params}
        expected_orig = next(iter(per_param_orig)) if len(per_param_orig) == 1 else None
        expected_reduce = next(iter(per_param_reduce)) if len(per_param_reduce) == 1 else None
        records.append(
            {
                "module": getattr(group, "_module_fqn", "?"),
                "orig": _dtype_name(group._orig_dtype),
                "reduce": _dtype_name(group._reduce_dtype),
                "expected_orig": _dtype_name(expected_orig),
                "expected_reduce": _dtype_name(expected_reduce),
                "matches": group._orig_dtype == expected_orig and group._reduce_dtype == expected_reduce,
            }
        )
    return records


def _expected_master_dtype(name: str, case: dict) -> torch.dtype:
    if name.startswith("carveout."):
        return torch.float32
    return case["master_dtype"]


def _parameter_storage_is_correct(model: torch.nn.Module, case: dict, device: torch.device) -> bool:
    for name, parameter in model.named_parameters():
        expected_dtype = _expected_master_dtype(name, case)
        if not isinstance(parameter, DTensor) or _placement_names(parameter) != ("Shard",):
            return False
        local = parameter.to_local()
        if parameter.dtype != expected_dtype or local.dtype != expected_dtype or local.device != device:
            return False
    return True


def _full_fsdp_grads(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], bool, bool]:
    grads = {}
    placement_ok = True
    finite = True
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        placement_ok &= isinstance(grad, DTensor) and _placement_names(grad) == ("Shard",)
        finite &= bool(torch.isfinite(_local(grad)).all().item())
        grads[name] = grad.full_tensor() if isinstance(grad, DTensor) else grad
    return grads, placement_ok, finite


def _test_mp_dtype_patch_matrix_impl(
    rank: int,
    world_size: int,
    case_name: str,
    patched: bool,
) -> dict:
    """Run one precision/topology case in a fresh process group."""

    case = _MP_CASES[case_name]
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))
    if patched:
        assert fsdp_utils.patch_fsdp2_mp_dtypes()

    torch.manual_seed(3107)
    model = _MixedPrecisionPatchNet(
        case["master_dtype"],
        fp32_carveout=case["fp32_carveout"],
    ).to(device)
    torch.manual_seed(3107)
    reference = _MixedPrecisionPatchNet(
        case["master_dtype"],
        fp32_carveout=case["fp32_carveout"],
    ).to(device)
    _shard_mp_model(model, mesh, case)

    model.set_roles(not case["initially_frozen"])
    reference.set_roles(not case["initially_frozen"])
    local_inputs = [_make_mp_inputs(rank, microbatch, device) for microbatch in range(2)]
    union_inputs = torch.cat(
        [_make_mp_inputs(other_rank, microbatch, device) for other_rank in range(world_size) for microbatch in range(2)]
    )

    # The first forward is the trigger for the frozen-group bug. Keep the
    # no-grad forward in both the active control and frozen regression case.
    with torch.no_grad():
        warm_output = _autocast_forward(model, local_inputs[0], case["amp_dtype"])
        warm_reference = _autocast_forward(reference, local_inputs[0], case["amp_dtype"])
    warm_forward_error = float((warm_output.float() - warm_reference.float()).abs().max().item())
    observed_compute_dtypes = model.observed_compute_dtypes()
    expected_compute_dtypes = {
        "input": _dtype_name(case["param_dtype"]),
        "frozen": _dtype_name(case["param_dtype"]),
        "output": _dtype_name(case["param_dtype"]),
        "root": _dtype_name(case["param_dtype"]),
    }
    if case["fp32_carveout"]:
        expected_compute_dtypes["carveout"] = "float32"
    group_records = _group_dtype_records(model)

    # Root groups use reshard_after_forward=False, so explicitly establish the
    # sharded phase before changing roles and accumulating gradients.
    fsdp_utils.reshard_fsdp_modules((model,))
    model.set_roles(True)
    reference.set_roles(True)
    storage_ok = _parameter_storage_is_correct(model, case, device)

    reference.zero_grad(set_to_none=True)
    reference_output = _autocast_forward(reference, union_inputs, case["amp_dtype"])
    reference_output.float().square().mean().backward()
    reference_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }

    max_forward_error = 0.0
    failure = None
    try:
        model.zero_grad(set_to_none=True)
        container = SimpleNamespace(module_dict={"net": model})
        for microbatch, inputs in enumerate(local_inputs):
            sync = microbatch == len(local_inputs) - 1
            with fsdp_utils.fsdp_sync_grad(container, enabled=sync):
                output = _autocast_forward(model, inputs, case["amp_dtype"])
                start = (rank * len(local_inputs) + microbatch) * inputs.shape[0]
                expected_output = reference_output[start : start + inputs.shape[0]]
                max_forward_error = max(
                    max_forward_error,
                    float((output.float() - expected_output.float()).abs().max().item()),
                )
                (output.float().square().mean() / len(local_inputs)).backward()
    except Exception as error:  # Stock FSDP2 is expected to fail in the frozen cases.
        failure = f"{type(error).__name__}: {error}"

    local_result = {
        "failure": failure,
        "warm_forward_error": warm_forward_error,
        "max_forward_error": max_forward_error,
        "compute_dtypes_match": observed_compute_dtypes == expected_compute_dtypes,
        "observed_compute_dtypes": observed_compute_dtypes,
        "expected_compute_dtypes": expected_compute_dtypes,
        "storage_ok": storage_ok,
        "group_records": group_records,
        "group_dtypes_match": bool(group_records) and all(record["matches"] for record in group_records),
        # None is FSDP2's valid sentinel for "reduce in the compute dtype".
        # A group is unresolved only when its group-level values disagree with
        # the uniform per-parameter values, not merely when one value is None.
        "unresolved_groups": sum(not record["matches"] for record in group_records),
    }

    if failure is None:
        fsdp_grads, grad_placement_ok, grads_finite = _full_fsdp_grads(model)
        frozen_grad_ok = all(
            parameter.grad is None for name, parameter in model.named_parameters() if name.startswith("frozen.")
        )
        trainable_names = set(reference_grads)
        max_grad_rel_error = 0.0
        for name in trainable_names:
            actual = fsdp_grads[name].float()
            expected = reference_grads[name].float()
            relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-6)
            max_grad_rel_error = max(max_grad_rel_error, float(relative_error.item()))

        grad_dtype_ok = all(
            parameter.grad is None or parameter.grad.dtype == _expected_master_dtype(name, case)
            for name, parameter in model.named_parameters()
        )
        local_result.update(
            {
                "all_trainable_grads_present": set(fsdp_grads) == trainable_names,
                "frozen_grad_ok": frozen_grad_ok,
                "grad_placement_ok": grad_placement_ok,
                "grad_dtype_ok": grad_dtype_ok,
                "grads_finite": grads_finite,
                "max_grad_rel_error": max_grad_rel_error,
                "params_resharded_after_backward": _parameter_storage_is_correct(model, case, device),
            }
        )

    return {
        "case": case_name,
        "patched": patched,
        "per_rank": _all_rank_results(local_result, world_size),
    }


def _assert_correct_mp_result(result: dict, case: dict) -> None:
    for rank, entry in enumerate(result["per_rank"]):
        assert entry["failure"] is None, f"rank {rank}: {entry['failure']}"
        assert entry["warm_forward_error"] < 0.02, (rank, entry)
        assert entry["max_forward_error"] < 0.02, (rank, entry)
        assert entry["compute_dtypes_match"], (rank, entry)
        assert entry["storage_ok"], (rank, entry)
        assert entry["group_dtypes_match"], (rank, entry["group_records"])
        assert entry["unresolved_groups"] == 0, (rank, entry["group_records"])
        assert entry["all_trainable_grads_present"], (rank, entry)
        assert entry["frozen_grad_ok"], (rank, entry)
        assert entry["grad_placement_ok"], (rank, entry)
        assert entry["grad_dtype_ok"], (rank, entry)
        assert entry["grads_finite"], (rank, entry)
        assert entry["max_grad_rel_error"] < case["grad_rtol"], (rank, entry)
        assert entry["params_resharded_after_backward"], (rank, entry)


@pytest.mark.parametrize("case_name", tuple(_MP_CASES))
@RunIf(min_gpus=2)
def test_fsdp_mp_dtype_patch_against_stock_and_non_fsdp(case_name):
    """Patch correctness across AMP, dtype, role, accumulation, and topology combinations."""

    clear_gpu_memory()
    stock = run_distributed_test(
        test_fn=_test_mp_dtype_patch_matrix_impl,
        world_size=2,
        timeout=180,
        nccl_timeout=120,
        setup_fn=set_env_vars,
        case_name=case_name,
        patched=False,
    )
    patched = run_distributed_test(
        test_fn=_test_mp_dtype_patch_matrix_impl,
        world_size=2,
        timeout=180,
        nccl_timeout=120,
        setup_fn=set_env_vars,
        case_name=case_name,
        patched=True,
    )
    assert stock is not None and patched is not None

    case = _MP_CASES[case_name]
    _assert_correct_mp_result(patched, case)
    if case["initially_frozen"]:
        # Depending on the dtype combination, stock FSDP either raises while
        # assigning the reduced gradient or silently leaves reduction metadata
        # unresolved. Both are the upstream bug this patch guards.
        for rank, entry in enumerate(stock["per_rank"]):
            known_gap = entry["failure"] is not None or not entry["group_dtypes_match"]
            assert known_gap, f"rank {rank}: stock FSDP no longer reproduces the frozen-init bug: {entry}"
    else:
        _assert_correct_mp_result(stock, case)

    clear_gpu_memory()


def _test_wan_moe_empty_attention_patch_impl(rank: int, world_size: int) -> dict:
    """A/B the diffusers native-attention workaround in a fresh GPU process."""

    del world_size
    from diffusers.models import attention_dispatch

    from fastgen.networks.Wan.wan_moe import _patch_diffusers_native_attention_for_empty_batch

    device = torch.device("cuda", rank)
    native_backend = attention_dispatch.AttentionBackendName.NATIVE
    stock_native = attention_dispatch._native_attention
    stock_was_unpatched = not bool(getattr(stock_native, "_fastgen_empty_batch_safe", False))

    def run_empty_case() -> dict:
        query = torch.randn(0, 6, 2, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
        key = torch.randn(0, 9, 2, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
        value = torch.randn(0, 9, 2, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
        try:
            output = attention_dispatch.dispatch_attention_fn(query, key, value, backend=native_backend)
            if output is None:
                return {"failure": "native attention returned None"}
            connected = output.grad_fn is not None
            output.sum().backward()
            grads_present = all(tensor.grad is not None for tensor in (query, key, value))
            return {
                "failure": None,
                "shape_ok": output.shape == query.shape,
                "dtype_ok": output.dtype == query.dtype,
                "connected": connected,
                "grads_present": grads_present,
            }
        except Exception as error:
            return {"failure": f"{type(error).__name__}: {error}"}

    stock = run_empty_case()
    _patch_diffusers_native_attention_for_empty_batch()
    patched_native = attention_dispatch._native_attention
    _patch_diffusers_native_attention_for_empty_batch()
    patched = run_empty_case()

    generator = torch.Generator(device=device).manual_seed(1977)
    query = torch.randn(2, 6, 2, 8, generator=generator, device=device, dtype=torch.bfloat16)
    key = torch.randn(2, 9, 2, 8, generator=generator, device=device, dtype=torch.bfloat16)
    value = torch.randn(2, 9, 2, 8, generator=generator, device=device, dtype=torch.bfloat16)
    populated_output = attention_dispatch.dispatch_attention_fn(query, key, value, backend=native_backend)
    populated_reference = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    ).transpose(1, 2)

    return {
        "stock_was_unpatched": stock_was_unpatched,
        "stock": stock,
        "patched": patched,
        "patch_installed": bool(getattr(patched_native, "_fastgen_empty_batch_safe", False)),
        "patch_idempotent": attention_dispatch._native_attention is patched_native,
        "registry_updated": (attention_dispatch._AttentionBackendRegistry._backends[native_backend] is patched_native),
        "populated_forward_error": float((populated_output.float() - populated_reference.float()).abs().max().item()),
    }


@RunIf(min_gpus=1)
def test_wan_moe_empty_attention_patch_against_stock_native_attention():
    """The patch preserves populated attention and connects empty Q/K/V graphs."""

    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_wan_moe_empty_attention_patch_impl,
        world_size=1,
        timeout=90,
        setup_fn=set_env_vars,
    )
    assert result is not None
    assert result["stock_was_unpatched"]
    assert result["patch_installed"]
    assert result["patch_idempotent"]
    assert result["registry_updated"]

    patched = result["patched"]
    assert patched["failure"] is None, patched
    assert patched["shape_ok"] and patched["dtype_ok"], patched
    assert patched["connected"] and patched["grads_present"], patched
    assert result["populated_forward_error"] < 0.02

    stock = result["stock"]
    if stock["failure"] is None:
        # Accept an upstream-fixed backend only if it already provides the
        # complete graph-preserving contract that the patch guarantees.
        assert stock["shape_ok"] and stock["dtype_ok"], stock
        assert stock["connected"] and stock["grads_present"], stock

    clear_gpu_memory()


class _RetainedRoot(FastGenModule):
    """Generic FastGen module whose direct parameter is retained after no-grad forwards."""

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(4, 4, kernel_size=1)
        self.root_bias = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 4))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projection(inputs) + self.root_bias.view(1, 4, 1, 1))


def _root_is_sharded(module: _RetainedRoot) -> bool:
    return isinstance(module.root_bias, DTensor) and _placement_names(module.root_bias) == ("Shard",)


def _test_forced_reshard_boundaries_impl(rank: int, world_size: int) -> dict:
    """Exercise the real logger and validation phase-boundary reshard calls."""

    from fastgen.callbacks.wandb import WandbCallback
    from fastgen.trainer import Trainer

    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))
    torch.manual_seed(2029)
    base = _RetainedRoot()
    net = copy.deepcopy(base)
    teacher = copy.deepcopy(base)
    ema = copy.deepcopy(base)
    reference = copy.deepcopy(base).to(device)

    fsdp_context = fsdp_utils.FSDPContext(
        mesh=mesh,
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=None,
        cpu_offload=False,
    )
    net.prepare_module(
        device=device,
        precision_model=torch.float32,
        precision_opt=torch.float32,
        trainable=True,
        fsdp_context=fsdp_context,
    )
    teacher.prepare_module(
        device=device,
        precision_model=torch.float32,
        trainable=False,
        fsdp_context=fsdp_context,
    )
    ema.prepare_module(
        device=device,
        precision_model=torch.float32,
        precision_opt=torch.float32,
        trainable=False,
        fsdp_context=fsdp_context,
    )

    generator = torch.Generator(device=device).manual_seed(4001)
    inputs = torch.randn(2, 4, 3, 3, generator=generator, device=device)
    condition = torch.randn(2, 3, generator=generator, device=device)
    phases = {
        "initially_sharded": all(_root_is_sharded(module) for module in (net, teacher, ema)),
        "generic_nested_and_root_wrapped": isinstance(net, FSDPModule) and isinstance(net.projection, FSDPModule),
    }

    class LoggingModel:
        net_preprocessors = SimpleNamespace()
        reshard_fsdp_modules = FastGenModel.reshard_fsdp_modules

        def __init__(self):
            self.teacher = teacher
            self.device = device
            self.uses_fsdp = True
            self.module_dict = {"net": net, "teacher": teacher, "ema": ema}

        def sample(self, network, noise, condition=None, neg_condition=None):
            del condition, neg_condition
            phases["student_sharded_before_teacher"] = _root_is_sharded(net)
            result = network(noise)
            phases["teacher_retained_after_forward"] = not _root_is_sharded(teacher)
            return result

    def deferred_student():
        result = net(inputs)
        phases["student_retained_after_forward"] = not _root_is_sharded(net)
        return result

    def fake_save(*_args, **_kwargs):
        phases["all_sharded_before_decode"] = all(_root_is_sharded(module) for module in (net, teacher, ema))
        return []

    with tempfile.TemporaryDirectory(prefix=f"fsdp_phase_rank{rank}_") as media_path:
        callback = WandbCallback(media_path=media_path, max_ranks=None, log_batchsize=1)
        callback.config = SimpleNamespace(trainer=SimpleNamespace(visualize_teacher=True))
        callback._do_save = fake_save
        with torch.no_grad():
            callback._write_media(
                LoggingModel(),
                {"condition": condition},
                {"gen_rand": deferred_student, "input_rand": inputs},
                iteration=1,
                group="train",
                suffix="",
            )
    phases["all_sharded_after_logger"] = all(_root_is_sharded(module) for module in (net, teacher, ema))

    validation_state = {}

    class ValidationModel:
        reshard_fsdp_modules = FastGenModel.reshard_fsdp_modules

        def __init__(self):
            self.device = device
            self.uses_fsdp = True
            self.module_dict = {"net": net, "teacher": teacher, "ema": ema}

        @staticmethod
        def autocast():
            return contextlib.nullcontext()

        @staticmethod
        def prepare_batch(data, **_kwargs):
            return data

        @staticmethod
        def single_train_step(data, iteration):
            del iteration
            prediction = net(data["x"])
            return {"total_loss": prediction.square().mean()}, {"prediction": prediction}

    class ValidationCallbacks:
        @staticmethod
        def on_validation_begin(*_args, **_kwargs):
            pass

        @staticmethod
        def on_validation_step_begin(*_args, **_kwargs):
            pass

        @staticmethod
        def on_validation_step_end(_model, _data, outputs, _losses, **_kwargs):
            validation_state["root_retained_inside_validation"] = not _root_is_sharded(net)
            validation_state["output"] = outputs["prediction"].detach().clone()

        @staticmethod
        def on_validation_end(*_args, **_kwargs):
            pass

    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(global_vars_val=[{}], val_seed=123))
    trainer.callbacks = ValidationCallbacks()
    validation_model = ValidationModel()
    trainer.validate(validation_model, validation_model, [{"x": inputs}], iteration=2)

    phases["all_sharded_after_validation"] = all(_root_is_sharded(module) for module in (net, teacher, ema))
    with torch.no_grad():
        reference_validation_output = reference(inputs)
    validation_forward_error = float((validation_state["output"] - reference_validation_output).abs().max().item())

    net.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    net_output = net(inputs)
    reference_output = reference(inputs)
    net_output.square().mean().backward()
    reference_output.square().mean().backward()
    fsdp_grads, grad_placement_ok, grads_finite = _full_fsdp_grads(net)
    reference_grads = {name: parameter.grad for name, parameter in reference.named_parameters()}
    max_grad_error = max(
        float((fsdp_grads[name] - reference_grads[name]).abs().max().item()) for name in reference_grads
    )

    local_result = {
        **phases,
        "root_retained_inside_validation": validation_state["root_retained_inside_validation"],
        "validation_forward_error": validation_forward_error,
        "training_forward_error": float((net_output - reference_output).abs().max().item()),
        "max_grad_error": max_grad_error,
        "grad_placement_ok": grad_placement_ok,
        "grads_finite": grads_finite,
        "net_resharded_after_backward": _root_is_sharded(net),
        "frozen_roots_have_no_grads": all(
            parameter.grad is None for module in (teacher, ema) for parameter in module.parameters()
        ),
        "frozen_roots_remain_sharded": _root_is_sharded(teacher) and _root_is_sharded(ema),
        "marked_root_count": sum(module._is_fsdp for module in (net, teacher, ema)),
    }
    return {"per_rank": _all_rank_results(local_result, world_size)}


@RunIf(min_gpus=2)
def test_logger_and_validation_force_root_reshards_before_the_next_phase():
    """No-backward phases retain root parameters until the real lifecycle hooks reshard them."""

    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_test_forced_reshard_boundaries_impl,
        world_size=2,
        timeout=180,
        nccl_timeout=120,
        setup_fn=set_env_vars,
    )
    assert result is not None
    for rank, entry in enumerate(result["per_rank"]):
        assert entry["initially_sharded"], (rank, entry)
        assert entry["generic_nested_and_root_wrapped"], (rank, entry)
        assert entry["student_retained_after_forward"], (rank, entry)
        assert entry["student_sharded_before_teacher"], (rank, entry)
        assert entry["teacher_retained_after_forward"], (rank, entry)
        assert entry["all_sharded_before_decode"], (rank, entry)
        assert entry["all_sharded_after_logger"], (rank, entry)
        assert entry["root_retained_inside_validation"], (rank, entry)
        assert entry["all_sharded_after_validation"], (rank, entry)
        assert entry["validation_forward_error"] < 1e-6, (rank, entry)
        assert entry["training_forward_error"] < 1e-6, (rank, entry)
        assert entry["max_grad_error"] < 1e-6, (rank, entry)
        assert entry["grad_placement_ok"], (rank, entry)
        assert entry["grads_finite"], (rank, entry)
        assert entry["net_resharded_after_backward"], (rank, entry)
        assert entry["frozen_roots_have_no_grads"], (rank, entry)
        assert entry["frozen_roots_remain_sharded"], (rank, entry)
        assert entry["marked_root_count"] == 3, (rank, entry)

    clear_gpu_memory()
