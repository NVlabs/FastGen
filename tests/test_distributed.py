# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for rank helpers, collectives, and distributed module synchronization."""

import pytest
import torch
import torch.distributed as dist

import fastgen.utils.distributed as distributed
from tests.utils import RunIf, run_distributed_test


@pytest.mark.parametrize(
    ("initialized", "cuda_available", "expected"),
    (
        (False, False, 1),
        (False, True, 1),
        (True, False, 1),
        (True, True, 8),
    ),
)
def test_world_size_preserves_initialized_cuda_semantics(monkeypatch, initialized, cuda_available, expected):
    get_world_size_calls = []
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(distributed.dist, "get_world_size", lambda: get_world_size_calls.append(None) or 8)

    assert distributed.world_size() == expected
    assert len(get_world_size_calls) == int(initialized and cuda_available)


@pytest.mark.parametrize(
    ("environment_value", "visible_devices", "expected"),
    ((None, 0, 1), (None, 4, 4), ("8", 4, 8), ("0", 4, 4)),
)
def test_local_world_size_prefers_launcher_environment(monkeypatch, environment_value, visible_devices, expected):
    if environment_value is None:
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", environment_value)
    monkeypatch.setattr(distributed.torch.cuda, "device_count", lambda: visible_devices)

    assert distributed.local_world_size() == expected


def test_get_rank_does_not_inspect_initialization_when_distributed_is_unavailable(monkeypatch):
    monkeypatch.setattr(distributed.dist, "is_available", lambda: False)
    monkeypatch.setattr(
        distributed.dist,
        "is_initialized",
        lambda: pytest.fail("is_initialized() must be short-circuited when distributed is unavailable"),
    )
    monkeypatch.setattr(
        distributed.dist,
        "get_rank",
        lambda *_args, **_kwargs: pytest.fail("get_rank() must not be called without distributed support"),
    )

    assert distributed.get_rank() == 0


def test_get_rank_returns_zero_for_an_uninitialized_process_group(monkeypatch):
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        distributed.dist,
        "get_rank",
        lambda *_args, **_kwargs: pytest.fail("get_rank() must not be called before process-group initialization"),
    )

    assert distributed.get_rank() == 0


def test_get_rank_forwards_the_process_group(monkeypatch):
    group = object()
    received_groups = []
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "get_rank", lambda received: received_groups.append(received) or 3)

    assert distributed.get_rank(group) == 3
    assert received_groups == [group]


@pytest.mark.parametrize(("worker_rank", "expected"), ((0, True), (1, False)))
def test_is_rank0_uses_get_rank(monkeypatch, worker_rank, expected):
    monkeypatch.setattr(distributed, "get_rank", lambda: worker_rank)

    assert distributed.is_rank0() is expected


def test_rank0_only_executes_and_returns_the_wrapped_result_on_rank_zero(monkeypatch):
    calls = []

    @distributed.rank0_only
    def add(value, increment=0):
        """Add an increment."""
        calls.append((value, increment))
        return value + increment

    monkeypatch.setattr(distributed, "is_rank0", lambda: True)

    assert add(2, increment=5) == 7
    assert calls == [(2, 5)]
    assert add.__name__ == "add"
    assert add.__doc__ == "Add an increment."


def test_rank0_only_suppresses_the_wrapped_call_on_other_ranks(monkeypatch):
    calls = []

    @distributed.rank0_only
    def record(value):
        calls.append(value)
        return value

    monkeypatch.setattr(distributed, "is_rank0", lambda: False)

    assert record("unreachable") is None
    assert calls == []


def _move_and_sync_module_ties_worker(rank, world_size, source_device, target_device):
    from fastgen.utils.distributed.module import move_and_sync_module

    class SharedBuffers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((2,), 3.0, dtype=torch.float64))
            self.child = torch.nn.Module()
            self.child_alias = self.child
            self.child.register_parameter("weight_alias", self.weight)
            self.child.register_buffer("cached", torch.full((2,), 2.0, dtype=torch.float64), persistent=False)

        def reset_parameters(self):
            # Distinct from rank 0's values, so the final value checks prove the broadcast.
            self.weight.fill_(0.0)
            self.child.cached.fill_(0.0)

    local_source = source_device if rank else "cpu"
    with torch.device(local_source):
        module = SharedBuffers()
    device = torch.device("cuda", torch.cuda.current_device()) if target_device == "cuda" else torch.device("cpu")

    # Buffer aliases are rejected by preparation before it moves anything, not by this helper.
    move_and_sync_module(module, device=device)

    assert module.child.cached is module.child_alias.cached
    assert module.weight is module.child.weight_alias
    assert module.weight.dtype == module.child.cached.dtype == torch.float64
    assert module.weight.device == module.child.cached.device == device
    assert module.child._non_persistent_buffers_set == {"cached"}
    torch.testing.assert_close(module.weight, torch.full((2,), 3.0, device=device, dtype=torch.float64))
    torch.testing.assert_close(module.child.cached, torch.full((2,), 2.0, device=device, dtype=torch.float64))
    return True


@RunIf(min_gpus=2)
@pytest.mark.parametrize("source_device", ["cpu", "meta"])
@pytest.mark.parametrize("target_device", ["cpu", "cuda"])
def test_move_and_sync_module_preserves_ties_and_shared_owners(source_device, target_device):
    assert run_distributed_test(
        _move_and_sync_module_ties_worker,
        world_size=2,
        timeout=120,
        nccl_timeout=30,
        source_device=source_device,
        target_device=target_device,
    )


def _agree(value: float) -> list:
    """Reduce ``value`` across ranks, returning ``[min, max]``.

    A rank that missed a broadcast still holds its own distinct value, which pulls the
    two apart; equal min and max means every rank agrees. The probe tensor is on CPU so
    it rides the gloo half of the default PG whatever device the value came from.
    """
    local = torch.tensor([value], dtype=torch.float64)
    low, high = local.clone(), local.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    return [float(low.item()), float(high.item())]


def _agree_all(flag: bool) -> bool:
    """True only when ``flag`` holds on *every* rank (MIN over a 0/1 probe)."""
    agreed = torch.tensor([1 if flag else 0], dtype=torch.int64)
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
    return bool(agreed.item())


def test_broadcast_choice_single_process(monkeypatch):
    import fastgen.utils.distributed.collectives as collectives

    monkeypatch.setattr(collectives, "world_size", lambda: 1)
    torch.manual_seed(17)
    expected = int(torch.randint(0, 23, (1,), dtype=torch.long).item())
    torch.manual_seed(17)
    assert collectives.broadcast_choice(23) == expected

    torch.manual_seed(17)
    expected = torch.randint(0, 23, (4,), dtype=torch.long).tolist()
    torch.manual_seed(17)
    assert collectives.broadcast_choice(23, 4) == expected

    for invalid in (0, -1):
        with pytest.raises(AssertionError):
            collectives.broadcast_choice(invalid)
        with pytest.raises(AssertionError):
            collectives.broadcast_choice(23, invalid)


def test_synchronize_and_broadcast_object_follow_world_size(monkeypatch):
    """Every collective helper gates on the one ``world_size`` predicate (CUDA-aware)."""
    import fastgen.utils.distributed.collectives as collectives

    collective_calls = []
    monkeypatch.setattr(collectives.dist, "barrier", lambda **kwargs: collective_calls.append(("barrier", kwargs)))
    monkeypatch.setattr(
        collectives.dist,
        "broadcast_object_list",
        lambda values, **kwargs: collective_calls.append(("broadcast_object_list", kwargs)),
    )

    monkeypatch.setattr(collectives, "world_size", lambda: 1)
    collectives.synchronize()
    assert collectives.broadcast_object({"a": 1}) == {"a": 1}
    assert collective_calls == []

    monkeypatch.setattr(collectives, "world_size", lambda: 2)
    collectives.synchronize()
    assert collectives.broadcast_object({"a": 1}, src=1) == {"a": 1}
    assert [name for name, _ in collective_calls] == ["barrier", "broadcast_object_list"]
    assert "device_ids" in collective_calls[0][1]
    assert collective_calls[1][1]["src"] == 1


def test_warn_on_rng_state_drift_single_process(monkeypatch):
    import fastgen.utils.distributed.collectives as collectives

    def fail(*_args):
        pytest.fail("one rank has nothing to compare")

    monkeypatch.setattr(collectives, "world_size", lambda: 1)
    monkeypatch.setattr(collectives.dist, "all_gather", fail)
    monkeypatch.setattr(collectives.dist, "all_gather_object", fail)
    monkeypatch.setattr(collectives.logger, "warning", pytest.fail)

    collectives.warn_on_rng_state_drift("Setup", cpu=True)


class _HostState:
    """A generator state that the simulated current device, ``cuda:1``, receives as a host copy."""

    def __init__(self, state):
        self.state = state

    def to(self, device):
        assert device == torch.device("cuda", 1)
        return self.state.clone()


class _GeneratorStub:
    def __init__(self, state, offset=0):
        self.state = state
        self.offset = offset

    def get_state(self):
        return _HostState(self.state)

    def get_offset(self):
        return self.offset


def _simulate_rng_ranks(monkeypatch, collectives, cpu_generator, cuda_generators, all_gather, all_gather_object):
    """Simulate three ranks whose current device is ``cuda:1`` and return the warnings rank 0 would log."""
    messages = []
    monkeypatch.setattr(collectives, "world_size", lambda: 3)
    monkeypatch.setattr(collectives.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(collectives.torch, "default_generator", cpu_generator)
    monkeypatch.setattr(collectives.torch.cuda, "default_generators", cuda_generators)
    monkeypatch.setattr(collectives.dist, "all_gather", all_gather)
    monkeypatch.setattr(collectives.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(collectives.logger, "warning", messages.append)
    return messages


@pytest.mark.parametrize(
    ("diverged", "cpu", "expected"),
    [
        (None, True, None),
        ("cpu", True, "CPU on ranks [2]"),
        ("cpu", False, None),
        ("cuda:1", False, "CUDA on ranks [2]"),
        ("cuda:0", True, None),
    ],
    ids=["equal", "cpu-checked", "cpu-exempt", "own-device", "other-device"],
)
def test_warn_on_rng_state_drift_compares_cpu_and_own_device(monkeypatch, diverged, cpu, expected):
    """CUDA ordinals are rank-local, so only the current device's generator is compared across ranks."""
    import fastgen.utils.distributed.collectives as collectives

    states = {
        "cpu": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "cuda:0": torch.tensor([6, 7], dtype=torch.uint8),
        "cuda:1": torch.tensor([4, 5], dtype=torch.uint8),
    }

    def all_gather(outputs, value):
        for output in outputs:
            output.copy_(value)
        if diverged is not None and torch.equal(value, states[diverged]):
            outputs[2][-1] += 1

    def all_gather_object(outputs, value):
        outputs[:] = [value] * len(outputs)

    messages = _simulate_rng_ranks(
        monkeypatch,
        collectives,
        _GeneratorStub(states["cpu"]),
        (_GeneratorStub(states["cuda:0"]), _GeneratorStub(states["cuda:1"])),
        all_gather,
        all_gather_object,
    )

    collectives.warn_on_rng_state_drift("Setup", cpu=cpu)

    if expected is None:
        assert messages == []
    else:
        assert len(messages) == 1 and messages[0].startswith(f"Setup: RNG states differ from rank 0 ({expected})")


@pytest.mark.parametrize(
    ("offsets", "rank2_draws", "expected"),
    [
        ((0, 8, 0), [], None),
        ((4, 8, 0), [], "rank 1: cuda:0"),
        ((4, 8, 12), ["cuda:1"], "rank 1: cuda:0, cuda:2; rank 2: cuda:1"),
    ],
    ids=["own-device-only", "other-device", "several"],
)
def test_warn_on_rng_state_drift_reports_draws_on_other_gpus(monkeypatch, offsets, rank2_draws, expected):
    """Seeding zeroes a CUDA generator's offset, so a nonzero offset on another GPU's generator reveals a draw there."""
    import fastgen.utils.distributed.collectives as collectives

    state = torch.tensor([1, 2], dtype=torch.uint8)

    def all_gather(outputs, value):
        for output in outputs:
            output.copy_(value)

    def all_gather_object(outputs, value):
        outputs[:] = [[], value, rank2_draws]

    messages = _simulate_rng_ranks(
        monkeypatch,
        collectives,
        _GeneratorStub(state),
        tuple(_GeneratorStub(state, offset) for offset in offsets),
        all_gather,
        all_gather_object,
    )

    collectives.warn_on_rng_state_drift("Setup", cpu=True)

    if expected is None:
        assert messages == []
    else:
        prefix = f"Setup: ranks drew random values on GPUs other than their own ({expected})"
        assert len(messages) == 1 and messages[0].startswith(prefix)


def test_broadcast_module_selects_names_and_coalesces(monkeypatch):
    """The transport is DDP's coalesced broadcast over the canonical selection; ``names`` narrows it."""
    import fastgen.utils.distributed.module as dist_module

    sent = []
    # The collectives' own ``world_size()`` guard short-circuits the cross-rank checks without a process group.
    monkeypatch.setattr(dist_module, "world_size", lambda: 2)
    monkeypatch.setattr(
        dist_module,
        "_sync_params_and_buffers",
        lambda group, tensors, bucket_bytes, src: sent.append((list(tensors), bucket_bytes, src)),
    )

    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.zeros(2))
    module.register_buffer("b", torch.ones(2))
    module.register_buffer("alias", module.b)  # a second registration of the same buffer
    module.register_buffer("scratch", torch.full((2,), 2.0), persistent=False)

    dist_module.broadcast_module(module, names={"alias", "weight"})

    assert len(sent) == 1
    tensors, bucket_bytes, src = sent[0]
    assert bucket_bytes == dist_module._BROADCAST_BUCKET_BYTES and src == 0
    # Canonical order (parameters, then buffers by name), one entry per shared tensor, detached payloads.
    assert [tensor.data_ptr() for tensor in tensors] == [module.weight.data_ptr(), module.b.data_ptr()]
    assert not any(tensor.requires_grad for tensor in tensors)

    sent.clear()
    dist_module.broadcast_module(module, parameters=False, persistent_buffers=False, names={"b"})
    assert sent == []  # nothing selected by kind and name: no transport call


def _validate_values_across_ranks(rank: int, world_size: int) -> dict:
    """Fingerprints cover replicated parameters and buffers while skipping non-replicated DTensors."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

    from fastgen.utils.distributed.module import validate_module_values_across_ranks

    device = torch.device(f"cuda:{rank}")
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.full((2,), 1.0, device=device))
    module.register_buffer("table", torch.tensor([float("nan"), 2.0], device=device))
    mesh = init_device_mesh("cuda", (world_size,))
    module.replicated = torch.nn.Parameter(
        DTensor.from_local(torch.ones(2, device=device), mesh, [Replicate()], run_check=False)
    )
    module.shard = torch.nn.Parameter(
        distribute_tensor(torch.arange(float(2 * world_size), device=device).reshape(world_size, 2), mesh, [Shard(0)])
    )
    module.partial = torch.nn.Parameter(
        DTensor.from_local(torch.full((2,), float(rank + 1), device=device), mesh, [Partial()], run_check=False)
    )

    # The all-Replicate DTensor agrees. Sharded and partial local values differ by design and are skipped.
    validate_module_values_across_ranks(module, parameters=True)

    # A single-element drift cannot cancel through repeated values in the XOR fingerprint.
    with torch.no_grad():
        module.weight[0] = 1.0 + 2.0 * rank
    try:
        validate_module_values_across_ranks(module, parameters=True, operation="value check")
        message = ""
    except RuntimeError as error:
        message = str(error)
    rejected = "value check mismatch with rank 0" in message and "'fingerprints'" in message and "rank 1" in message

    with torch.no_grad():
        module.weight.fill_(1.0)
        module.replicated.to_local()[0] = 1.0 + 2.0 * rank
    replicated_before = module.replicated.to_local().detach().clone()
    try:
        validate_module_values_across_ranks(module, parameters=True, operation="replicated DTensor value check")
        replicated_message = ""
    except RuntimeError as error:
        replicated_message = str(error)
    replicated_rejected = "'fingerprints'" in replicated_message and "rank 1" in replicated_message

    validate_module_values_across_ranks(module, parameters=False)  # buffers alone still agree
    sharded_only = torch.nn.Module()
    sharded_only.shard = module.shard
    validate_module_values_across_ranks(sharded_only, parameters=True)  # no replicated values to compare
    return {
        "rejects_parameter_drift": _agree_all(rejected),
        "rejects_replicated_dtensor_drift": _agree_all(replicated_rejected),
        "did_not_mutate": _agree_all(torch.equal(module.replicated.to_local(), replicated_before)),
    }


@RunIf(min_gpus=2)
def test_validate_module_values_across_ranks():
    result = run_distributed_test(test_fn=_validate_values_across_ranks, world_size=2)
    assert result is not None
    assert all(result.values()), result


def test_sync_extrema_single_process(monkeypatch):
    import fastgen.utils.distributed.collectives as collectives

    monkeypatch.setattr(collectives, "world_size", lambda: 1)
    assert collectives.sync_max(7, device=torch.device("cpu")) == 7
    assert collectives.sync_min(7, device=torch.device("cpu")) == 7


def _sync_extrema_agree(rank: int, world_size: int) -> dict:
    from fastgen.utils.distributed.collectives import sync_max, sync_min

    value = 2 * rank + 1
    device = torch.device(f"cuda:{rank}")
    minimum = sync_min(value, device=device)
    maximum = sync_max(value, device=device)
    return {
        "minimum": _agree(float(minimum)),
        "maximum": _agree(float(maximum)),
        "expected": _agree_all(minimum == 1 and maximum == 2 * (world_size - 1) + 1),
    }


@RunIf(min_gpus=2)
def test_sync_extrema_agree_across_ranks():
    result = run_distributed_test(test_fn=_sync_extrema_agree, world_size=2)
    assert result["minimum"] == [1.0, 1.0]
    assert result["maximum"] == [3.0, 3.0]
    assert result["expected"]


def _broadcast_choice_agrees(rank: int, world_size: int) -> dict:
    from fastgen.utils.distributed.collectives import broadcast_choice

    torch.manual_seed(100 + rank)
    choices = broadcast_choice(23, 4, device=f"cuda:{rank}")
    assert isinstance(choices, list)
    return {
        "choices": [_agree(float(choice)) for choice in choices],
        "in_range": _agree_all(all(0 <= choice < 23 for choice in choices)),
    }


@RunIf(min_gpus=2)
def test_broadcast_choice_agrees_across_ranks():
    result = run_distributed_test(test_fn=_broadcast_choice_agrees, world_size=2)
    assert all(low == high for low, high in result["choices"])
    assert result["in_range"]


def _broadcast_object_agrees(rank: int, world_size: int) -> dict:
    """Broadcast one nested object through both backends of the mixed group."""
    del world_size
    from fastgen.utils.distributed.collectives import broadcast_object

    expected = {"prompts": ["first", "second"], "metadata": {"seed": 17, "optional": None}}
    local = expected if rank == 0 else {"wrong_rank": rank}
    via_nccl = broadcast_object(local, src=0, device=torch.device(f"cuda:{rank}"))
    via_default = broadcast_object(local, src=0)
    return {
        "nccl": _agree_all(via_nccl == expected),
        "default": _agree_all(via_default == expected),
    }


@RunIf(min_gpus=2)
def test_broadcast_object_agrees_across_ranks():
    result = run_distributed_test(test_fn=_broadcast_object_agrees, world_size=2)
    assert all(result.values()), result


def _warn_on_rng_state_drift_detects_rank_local_draws(rank: int, world_size: int) -> dict:
    """A draw on one rank warns about the generator it advanced on every rank; equal own-device draws do not."""
    del world_size
    import fastgen.utils.distributed.collectives as collectives

    messages = []
    rank0_warning = collectives.logger.warning
    collectives.logger.warning = messages.append

    def check(cpu: bool) -> list[str]:
        messages.clear()
        collectives.warn_on_rng_state_drift("Check", cpu=cpu)
        return list(messages)

    try:
        torch.manual_seed(7)
        torch.rand(1, device=torch.device("cuda", rank))
        own_device_draws = check(cpu=True)
        if rank == 1:
            torch.rand(1)
        cpu_exempt, cpu_checked = check(cpu=False), check(cpu=True)
        torch.manual_seed(7)
        if rank == 1:
            torch.rand(1, device=torch.device("cuda", rank))
        cuda = check(cpu=True)
        torch.manual_seed(7)
        if rank == 1:
            torch.rand(1, device=torch.device("cuda", 0))
        other_gpu = check(cpu=True)
    finally:
        collectives.logger.warning = rank0_warning
    return {
        "own_device_draws": _agree_all(own_device_draws == []),
        "cpu_exempt": _agree_all(cpu_exempt == []),
        "cpu_checked": _agree_all(len(cpu_checked) == 1 and "(CPU on ranks [1])" in cpu_checked[0]),
        "cuda": _agree_all(len(cuda) == 1 and "(CUDA on ranks [1])" in cuda[0]),
        "other_gpu": _agree_all(len(other_gpu) == 1 and "(rank 1: cuda:0)" in other_gpu[0]),
    }


@RunIf(min_gpus=2)
def test_warn_on_rng_state_drift_detects_rank_local_draws():
    result = run_distributed_test(test_fn=_warn_on_rng_state_drift_detects_rank_local_draws, world_size=2)
    assert all(result.values()), result


def _move_module_handles_meta(rank: int, world_size: int) -> dict:
    """Non-rank-0 starts on meta; ``move_and_sync_module`` materializes the
    meta tensors, moves rank 0 to the target device, and syncs every
    parameter/buffer from rank 0."""
    from fastgen.utils.distributed.module import move_and_sync_module

    if rank == 0:
        # Real weights, deterministic so we can compare across ranks.
        m = torch.nn.Linear(8, 8)
        torch.nn.init.constant_(m.weight, 0.5)
        torch.nn.init.constant_(m.bias, -0.25)
    else:
        with torch.device("meta"):
            m = torch.nn.Linear(8, 8)

    move_and_sync_module(m, device="cpu")

    return {
        "weight_sum": _agree(float(m.weight.detach().sum().item())),
        "bias_sum": _agree(float(m.bias.detach().sum().item())),
        "all_on_cpu": _agree_all(next(m.parameters()).device.type == "cpu"),
    }


@RunIf(min_gpus=2)
def test_move_module_handles_meta():
    """``move_and_sync_module`` materializes meta tensors on receiving ranks
    and broadcasts rank 0's values; the mixed-backend default PG dispatches the
    CPU broadcast through gloo automatically."""
    result = run_distributed_test(test_fn=_move_module_handles_meta, world_size=2)
    # Linear(8, 8) with weight=0.5 sums to 32, bias=-0.25 sums to -2.0. These must hold
    # on the meta rank too — that is the half actually exercising materialization.
    assert result["weight_sum"] == [32.0, 32.0]
    assert result["bias_sum"] == [-2.0, -2.0]
    assert result["all_on_cpu"]


def _move_module_rejects_meta_source(rank: int, world_size: int) -> dict:
    """Every rank must reject a meta broadcast source before placement mutates state."""
    from fastgen.utils.distributed.module import move_and_sync_module

    if rank == 0:
        with torch.device("meta"):
            m = torch.nn.Linear(8, 8)
    else:
        m = torch.nn.Linear(8, 8)
        torch.nn.init.constant_(m.weight, 0.5)
        torch.nn.init.constant_(m.bias, -0.25)

    try:
        move_and_sync_module(m, device="cpu")
    except RuntimeError as error:
        rejected = "Rank 0 must hold the complete real state" in str(error)
    else:
        rejected = False

    unchanged = m.weight.is_meta if rank == 0 else m.weight.device.type == "cpu"
    return {
        "rejected": _agree_all(rejected),
        "unchanged": _agree_all(unchanged),
    }


@RunIf(min_gpus=2)
def test_move_module_rejects_meta_rank0_collectively():
    """A meta source would broadcast reset/random state, so all ranks fail first."""
    result = run_distributed_test(test_fn=_move_module_rejects_meta_source, world_size=2)
    assert result["rejected"]
    assert result["unchanged"]


def _validate_module_sync_source_is_read_only(rank: int, world_size: int) -> dict:
    del world_size
    from fastgen.utils.distributed.collectives import raise_across_ranks
    from fastgen.utils.distributed.module import validate_module_sync_source

    if rank == 0:
        with torch.device("meta"):
            module = torch.nn.Linear(2, 2)
    else:
        module = torch.nn.Linear(2, 2)
        torch.nn.init.constant_(module.weight, 3.0)
        torch.nn.init.constant_(module.bias, 4.0)

    weight = module.weight
    bias = module.bias
    weight_value = None if weight.is_meta else weight.detach().clone()
    bias_value = None if bias.is_meta else bias.detach().clone()
    try:
        with raise_across_ranks("test synchronization"):
            validate_module_sync_source(module, src=0, operation="test synchronization")
    except RuntimeError as error:
        rejected = "Rank 0 must hold the complete real state" in str(error)
    else:
        rejected = False

    unchanged = module.weight is weight and module.bias is bias
    if weight_value is not None:
        unchanged = unchanged and torch.equal(module.weight, weight_value) and torch.equal(module.bias, bias_value)
    return {
        "rejected_collectively": _agree_all(rejected),
        "module_unchanged": _agree_all(unchanged),
    }


@RunIf(min_gpus=2)
def test_validate_module_sync_source_is_read_only():
    result = run_distributed_test(test_fn=_validate_module_sync_source_is_read_only, world_size=2)
    assert result is not None
    assert all(result.values()), result


def _move_module_rejects_rank_local_reset_mismatch(rank: int, world_size: int) -> dict:
    """A reset violation on only the meta rank must fail collectively before broadcast."""
    del world_size
    from fastgen.utils.distributed.module import move_and_sync_module

    class InvalidReset(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2))
            self.register_buffer("cache", torch.empty(2), persistent=False)
            self.reset_parameters()

        def reset_parameters(self):
            torch.nn.init.ones_(self.weight)
            # Valid during ordinary fp32 construction, but invalid after preparation selected bf16.
            self.cache = torch.ones_like(self.cache, dtype=torch.float32)

    if rank == 0:
        module = InvalidReset()
    else:
        with torch.device("meta"):
            module = InvalidReset()
    module.to(dtype=torch.bfloat16)

    try:
        move_and_sync_module(module, device="cpu")
    except RuntimeError as error:
        message = str(error)
        rejected = "changed registered state" in message
        reported_offender = "rank 1" in message
    else:
        rejected = reported_offender = False

    return {
        "rejected_collectively": _agree_all(rejected),
        "same_offender_reported": _agree_all(reported_offender),
    }


@RunIf(min_gpus=2)
def test_move_module_rejects_rank_local_reset_mismatch_collectively():
    result = run_distributed_test(test_fn=_move_module_rejects_rank_local_reset_mismatch, world_size=2)
    assert result is not None
    assert all(result.values()), result


def _move_module_reports_rank_local_reset_exception(rank: int, world_size: int) -> dict:
    """A hook exception on only the meta rank must reach every rank before broadcast."""
    del world_size
    from fastgen.utils.distributed.module import move_and_sync_module

    class InvalidReset(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2))
            self.reset_count = 0
            self.reset_parameters()

        def reset_parameters(self):
            self.reset_count += 1
            if self.reset_count > 1:
                self.missing_buffer.copy_(torch.ones(2))
            torch.nn.init.ones_(self.weight)

    if rank == 0:
        module = InvalidReset()
    else:
        with torch.device("meta"):
            module = InvalidReset()

    try:
        move_and_sync_module(module, device="cpu")
    except RuntimeError as error:
        message = str(error)
        rejected = "AttributeError" in message and "missing_buffer" in message
        reported_offender = "rank 1" in message
    else:
        rejected = reported_offender = False

    return {
        "rejected_collectively": _agree_all(rejected),
        "same_offender_reported": _agree_all(reported_offender),
    }


@RunIf(min_gpus=2)
def test_move_module_reports_rank_local_reset_exception_collectively():
    result = run_distributed_test(test_fn=_move_module_reports_rank_local_reset_exception, world_size=2)
    assert result is not None
    assert all(result.values()), result


def _move_module_reports_source_rank_placement_failure(rank: int, world_size: int) -> dict:
    """A placement failure on the real source rank must reach the meta ranks instead of stalling them."""
    del world_size
    from fastgen.utils.distributed.module import move_and_sync_module

    class FailingMove(torch.nn.Linear):
        def _apply(self, fn, recurse=True):
            if rank == 0:
                raise RuntimeError("simulated placement failure")
            return super()._apply(fn, recurse)

    if rank == 0:
        module = FailingMove(2, 2)
    else:
        with torch.device("meta"):
            module = FailingMove(2, 2)

    try:
        move_and_sync_module(module, device="cpu")
    except RuntimeError as error:
        message = str(error)
        rejected = "simulated placement failure" in message and "rank 0" in message
    else:
        rejected = False

    return {"rejected_collectively": _agree_all(rejected)}


@RunIf(min_gpus=2)
def test_move_module_reports_source_rank_placement_failure_collectively():
    result = run_distributed_test(test_fn=_move_module_reports_source_rank_placement_failure, world_size=2)
    assert result is not None
    assert all(result.values()), result


def _raise_across_ranks_contract(rank: int, world_size: int) -> dict:
    """Success runs each rank's own block; one failing rank raises the same error everywhere."""
    del world_size
    from fastgen.utils.distributed.collectives import raise_across_ranks

    with raise_across_ranks("test placement"):
        returned = rank + 10

    try:
        with raise_across_ranks("test placement"):
            if rank == 1:
                raise ValueError("buffer 'cache': dtype mismatch")
    except RuntimeError as error:
        message = str(error)
        rejected = (
            "test placement" in message
            and "rank 1" in message
            and "ValueError: buffer 'cache': dtype mismatch" in message
        )
        cause_on_failing_rank_only = isinstance(error.__cause__, ValueError) == (rank == 1)
    else:
        rejected = cause_on_failing_rank_only = False

    return {
        "runs_local_block": _agree_all(returned == rank + 10),
        "rejected_collectively": _agree_all(rejected),
        "cause_on_failing_rank_only": _agree_all(cause_on_failing_rank_only),
    }


@RunIf(min_gpus=2)
def test_raise_across_ranks_contract():
    result = run_distributed_test(test_fn=_raise_across_ranks_contract, world_size=2)
    assert result is not None
    assert all(result.values()), result


def _synced_iter_contract(rank: int, world_size: int) -> dict:
    from itertools import islice

    from fastgen.utils.distributed.collectives import synced_iter

    device = torch.device("cuda", rank)
    equal = list(synced_iter([rank, rank + 10], device)) == [rank, rank + 10]
    empty = list(synced_iter([], device)) == []
    none_payload = list(synced_iter([None], device)) == [None]
    limited = list(islice(synced_iter(range(rank + 2), device), 2)) == [0, 1]
    unequal = False
    steps = 0
    try:
        for _ in synced_iter(range(rank + 1), device):
            # Represent a model collective: exhaustion must be detected before entering it.
            dist.all_reduce(torch.ones(1, device=device))
            steps += 1
    except ValueError as error:
        unequal = "different numbers of items" in str(error) and steps == 1

    def broken(fail_all, peer_exhausted):
        yield rank
        if fail_all or rank == world_size - 1:
            raise ValueError(f"invalid item on rank {rank}")
        if not peer_exhausted:
            yield rank + 10

    failures_shared = []
    # An iterator error must also win over exhaustion, including when every rank has no item.
    for fail_all, peer_exhausted in [(False, False), (False, True), (True, False)]:
        failed_together = False
        try:
            list(synced_iter(broken(fail_all, peer_exhausted), device))
        except RuntimeError as error:
            first_failed = 0 if fail_all else world_size - 1
            expected = f"rank {first_failed} failed: ValueError: invalid item on rank {first_failed}"
            local_failure = fail_all or rank == world_size - 1
            failed_together = expected in str(error) and isinstance(error.__cause__, ValueError) == local_failure
        failures_shared.append(failed_together)

    class BrokenIterable:
        def __iter__(self):
            if rank == world_size - 1:
                raise ValueError("cannot initialize iterator")
            return iter([rank])

    initialization_failed = False
    try:
        list(synced_iter(BrokenIterable(), device))
    except RuntimeError as error:
        initialization_failed = "cannot initialize iterator" in str(error)
    return {
        "equal": _agree_all(equal),
        "empty": _agree_all(empty),
        "none_payload": _agree_all(none_payload),
        "limited": _agree_all(limited),
        "unequal_rejected": _agree_all(unequal),
        "iterator_error_shared": _agree_all(all(failures_shared)),
        "initialization_error_shared": _agree_all(initialization_failed),
    }


@RunIf(min_gpus=2)
def test_synced_iter_finishes_or_fails_on_every_rank():
    result = run_distributed_test(test_fn=_synced_iter_contract, world_size=2)
    assert result is not None
    assert all(result.values()), result


def test_synced_iter_single_process_preserves_values_and_errors(monkeypatch):
    import fastgen.utils.distributed.collectives as collectives

    monkeypatch.setattr(collectives, "world_size", lambda: 1)
    assert list(collectives.synced_iter([None, 1], "cpu")) == [None, 1]
    assert list(collectives.synced_iter([], "cpu")) == []

    def broken():
        yield 1
        raise ValueError("invalid item")

    items = collectives.synced_iter(broken(), "cpu")
    assert next(items) == 1
    with pytest.raises(ValueError, match="invalid item"):
        next(items)


def test_broadcast_rank0_result_single_process(monkeypatch):
    """Without peers, ``fn`` runs locally and its own exception propagates unwrapped."""
    import fastgen.utils.distributed.collectives as collectives

    monkeypatch.setattr(collectives, "world_size", lambda: 1)
    assert collectives.broadcast_rank0_result(lambda: {"a": 1}, "test result") == {"a": 1}

    def fail():
        raise ValueError("marker unreadable")

    with pytest.raises(ValueError, match="marker unreadable"):
        collectives.broadcast_rank0_result(fail, "test result")


@pytest.mark.parametrize("rank", [0, 1])
def test_broadcast_rank0_result_runs_fn_only_on_rank0(monkeypatch, rank):
    import fastgen.utils.distributed.collectives as collectives

    calls = []
    broadcasts = []
    device = torch.device("cuda", 0)
    monkeypatch.setattr(collectives, "world_size", lambda: 2)
    monkeypatch.setattr(collectives, "get_rank", lambda: rank)
    monkeypatch.setattr(collectives.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(collectives, "sync_min", lambda value, device: 2)  # no rank failed
    monkeypatch.setattr(
        collectives, "broadcast_object", lambda value, **kwargs: broadcasts.append((value, kwargs)) or "result"
    )

    result = collectives.broadcast_rank0_result(lambda: calls.append(rank) or "result", "test result", device=device)

    assert result == "result"
    assert calls == ([0] if rank == 0 else [])
    assert broadcasts == [("result" if rank == 0 else None, {"device": device})]


@pytest.mark.parametrize("rank", [0, 1])
def test_broadcast_rank0_result_reports_rank0_failure_instead_of_broadcasting(monkeypatch, rank):
    import fastgen.utils.distributed.collectives as collectives

    broadcasts = []
    monkeypatch.setattr(collectives, "world_size", lambda: 2)
    monkeypatch.setattr(collectives, "get_rank", lambda: rank)
    monkeypatch.setattr(collectives.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(collectives, "sync_min", lambda value, device: 0)  # rank 0 failed
    monkeypatch.setattr(
        collectives,
        "broadcast_object",
        lambda value, **kwargs: broadcasts.append(value) or "ValueError: marker unreadable",
    )

    def fail():
        raise ValueError("marker unreadable")

    with pytest.raises(RuntimeError, match="test result: rank 0 failed: ValueError: marker unreadable") as error:
        collectives.broadcast_rank0_result(fail, "test result")

    assert isinstance(error.value.__cause__, ValueError) if rank == 0 else error.value.__cause__ is None
    assert broadcasts == ["ValueError: marker unreadable" if rank == 0 else None], "only the failure report is sent"


def _broadcast_rank0_result_contract(rank: int, world_size: int) -> dict:
    """Only rank 0 runs ``fn``; every rank returns its result through either backend or raises its failure."""
    del world_size
    from fastgen.utils.distributed.collectives import broadcast_rank0_result

    calls = []
    via_nccl = broadcast_rank0_result(
        lambda: calls.append(rank) or {"source": rank}, "test result", device=torch.device(f"cuda:{rank}")
    )
    via_default = broadcast_rank0_result(lambda: calls.append(rank) or {"source": rank}, "test result")

    def fail():
        raise ValueError("marker unreadable")

    try:
        broadcast_rank0_result(fail, "test result")
    except RuntimeError as error:
        rejected = "test result: rank 0 failed: ValueError: marker unreadable" in str(error)
        cause_on_rank0_only = isinstance(error.__cause__, ValueError) == (rank == 0)
    else:
        rejected = cause_on_rank0_only = False

    return {
        "nccl": _agree_all(via_nccl == {"source": 0}),
        "default": _agree_all(via_default == {"source": 0}),
        "runs_on_rank0_only": _agree_all(calls == ([0, 0] if rank == 0 else [])),
        "rejected_collectively": _agree_all(rejected),
        "cause_on_rank0_only": _agree_all(cause_on_rank0_only),
    }


@RunIf(min_gpus=2)
def test_broadcast_rank0_result_contract():
    result = run_distributed_test(test_fn=_broadcast_rank0_result_contract, world_size=2)
    assert result is not None
    assert all(result.values()), result


def _validate_module_state_mismatches(rank: int, world_size: int) -> dict:
    """Exercise multiple metadata disagreements without entering an unsafe state broadcast."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard

    from fastgen.utils.distributed.module import validate_module_state_across_ranks

    device = torch.device(f"cuda:{rank}")
    mesh = init_device_mesh("cuda", (world_size,))
    caught = {}
    for mismatch in (
        "dtype",
        "shape",
        "layout",
        "requires_grad",
        "persistence",
        "device_type",
        "placements",
        "missing",
        "extra",
        "kind",
        "none",
        "alias",
    ):
        module = torch.nn.Module()
        if mismatch == "placements":
            placement = Replicate() if rank == 0 else Shard(0)
            local_size = 2 if rank == 0 else 1
            weight = DTensor.from_local(
                torch.ones(local_size, device=device),
                mesh,
                [placement],
                run_check=False,
                shape=torch.Size([2]),
                stride=(1,),
            )
        else:
            weight = torch.ones(2, device=device)
        module.register_parameter(
            "weight",
            torch.nn.Parameter(weight, requires_grad=not (rank and mismatch == "requires_grad")),
        )

        if not (rank and mismatch == "missing"):
            if rank and mismatch == "kind":
                module.register_parameter("state", torch.nn.Parameter(torch.ones(2, device=device)))
            elif rank and mismatch == "none":
                module.register_buffer("state", None)
            else:
                shape = (3,) if rank and mismatch == "shape" else (2,)
                dtype = torch.float64 if rank and mismatch == "dtype" else torch.float32
                persistent = not (rank and mismatch == "persistence")
                state_device = torch.device("cpu") if rank and mismatch == "device_type" else device
                if rank and mismatch == "layout":
                    indices = torch.tensor([[0]], device=state_device)
                    values = torch.ones(1, device=state_device)
                    state = torch.sparse_coo_tensor(indices, values, (2,), device=state_device)
                else:
                    state = torch.ones(shape, dtype=dtype, device=state_device)
                module.register_buffer("state", state, persistent=persistent)
        if rank and mismatch == "extra":
            module.register_buffer("extra", torch.ones(1, device=device))
        if mismatch == "alias":
            other = module.state if rank else torch.ones(2, device=device)
            module.register_buffer("other", other)

        if mismatch == "dtype":
            weight = module.weight
            state = module.state
            weight_value = weight.detach().clone()
            state_value = state.detach().clone()

        try:
            validate_module_state_across_ranks(module, operation="test synchronization")
        except RuntimeError as error:
            message = str(error)
            rejected = "'registrations'" in message and "rank 1" in message
        else:
            rejected = False
        caught[mismatch] = _agree_all(rejected)
        if mismatch == "dtype":
            caught["metadata_validation_did_not_mutate"] = _agree_all(
                module.weight is weight
                and module.state is state
                and torch.equal(module.weight, weight_value)
                and torch.equal(module.state, state_value)
            )

    return caught


@RunIf(min_gpus=2)
def test_validate_module_state_across_ranks_rejects_metadata_mismatches():
    result = run_distributed_test(test_fn=_validate_module_state_mismatches, world_size=2)
    assert result is not None
    assert all(result.values()), result


def _validate_nonpersistent_buffer_values(rank: int, world_size: int) -> dict:
    """Fingerprints ignore checkpointed buffers and reject a non-colliding reset-state divergence."""
    del world_size
    from fastgen.utils.distributed.module import validate_module_values_across_ranks

    device = torch.device(f"cuda:{rank}")
    module = torch.nn.Module()
    module.register_buffer("persistent", torch.tensor([float(rank), 2.0], device=device))
    module.register_buffer("cache", torch.full((2,), 3.0, device=device), persistent=False)

    validate_module_values_across_ranks(module, persistent_buffers=False, operation="buffer reset validation")

    # Vary one element so the mismatch survives the XOR reduction.
    module.cache[0] = 3.0 + rank
    cache_before = module.cache.clone()
    try:
        validate_module_values_across_ranks(module, persistent_buffers=False, operation="buffer reset validation")
    except RuntimeError as error:
        message = str(error)
        rejected = (
            "buffer reset validation mismatch with rank 0" in message
            and "'fingerprints'" in message
            and "rank 1" in message
        )
    else:
        rejected = False

    return {
        "rejected_collectively": _agree_all(rejected),
        "validation_did_not_mutate": _agree_all(torch.equal(module.cache, cache_before)),
    }


@RunIf(min_gpus=2)
def test_validate_module_values_across_ranks_checks_only_nonpersistent_buffers():
    result = run_distributed_test(test_fn=_validate_nonpersistent_buffer_values, world_size=2)
    assert result is not None
    assert all(result.values()), result


def _broadcast_module_real_values(rank: int, world_size: int) -> dict:
    """``broadcast_module`` is pure broadcast — no meta handling.
    All ranks construct real values; rank 0's distinct value wins."""
    from fastgen.utils.distributed.module import broadcast_module

    m = torch.nn.Linear(8, 8)
    val = 0.5 if rank == 0 else 0.99
    torch.nn.init.constant_(m.weight, val)
    torch.nn.init.constant_(m.bias, -val)

    broadcast_module(m)

    return {
        "weight_sum": _agree(float(m.weight.detach().sum().item())),
        "bias_sum": _agree(float(m.bias.detach().sum().item())),
    }


@RunIf(min_gpus=2)
def test_broadcast_module_real_values():
    """Pure-broadcast contract: ``broadcast_module`` overwrites every
    parameter and buffer with rank 0's values across the world."""
    result = run_distributed_test(test_fn=_broadcast_module_real_values, world_size=2)
    # Linear(8, 8) with weight=0.5 sums to 32, bias=-0.5 sums to -4.0; a rank left on
    # its own 0.99 would show up as a spread between min and max.
    assert result["weight_sum"] == [32.0, 32.0]
    assert result["bias_sum"] == [-4.0, -4.0]


def _broadcast_module_alias_contract(rank: int, world_size: int) -> dict:
    del world_size
    from unittest.mock import patch

    from fastgen.utils.distributed.module import broadcast_module

    device = torch.device("cuda", rank)
    results = {}
    for alias_rank in (0, 1):
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.full((2,), 1.0 + rank, device=device), requires_grad=False)
        # Match every per-kind metadata field: only the cross-kind identity differs across ranks.
        cache = torch.nn.Parameter(torch.full((2,), 3.0 + rank, device=device), requires_grad=False)
        module.register_buffer("cache", module.weight if rank == alias_rank else cache)
        identities = (module.weight, module.cache)
        values = (module.weight.clone(), module.cache.clone())
        with patch("fastgen.utils.distributed.module._sync_params_and_buffers") as transport:
            try:
                broadcast_module(module)
            except RuntimeError as error:
                rejected = "both a parameter and a buffer" in str(error) and f"rank {alias_rank}" in str(error)
            else:
                rejected = False

        results[f"rank_{alias_rank}_alias_rejected_before_transport"] = _agree_all(rejected and not transport.called)
        results[f"rank_{alias_rank}_state_unchanged"] = _agree_all(
            module.weight is identities[0]
            and module.cache is identities[1]
            and torch.equal(module.weight, values[0])
            and torch.equal(module.cache, values[1])
        )

    module = torch.nn.Module()
    child = torch.nn.Module()
    child.weight = torch.nn.Parameter(torch.full((2,), 4.0 + rank, device=device))
    child.register_buffer("cache", torch.full((2,), 6.0 + rank, device=device))
    child.register_buffer("cache_alias", child.cache, persistent=False)
    module.left = module.right = child
    module.tied_weight = child.weight
    weight, cache = child.weight, child.cache
    broadcast_module(module)

    results["supported_aliases_preserved"] = _agree_all(
        module.left is module.right is child
        and module.tied_weight is child.weight is weight
        and child.cache_alias is child.cache is cache
    )
    results["supported_alias_values_synchronized"] = _agree_all(
        torch.equal(weight, torch.full_like(weight, 4.0)) and torch.equal(cache, torch.full_like(cache, 6.0))
    )
    return results


@RunIf(min_gpus=2)
def test_broadcast_module_validates_alias_contract_before_copying():
    result = run_distributed_test(test_fn=_broadcast_module_alias_contract, world_size=2, timeout=120, nccl_timeout=30)
    assert result is not None
    assert all(result.values()), result


def _broadcast_module_buffer_only(rank: int, world_size: int) -> dict:
    """A parameterless module checks metadata on the backend-safe device, then broadcasts its CUDA buffer."""
    from fastgen.utils.distributed.module import broadcast_module

    m = torch.nn.Module()
    m.register_buffer("state", torch.full((4,), 0.5 if rank == 0 else 0.99, device=f"cuda:{rank}"))
    broadcast_module(m)

    return {
        "state_sum": _agree(float(m.state.sum().item())),
        # Broadcast is in-place: every rank keeps its buffer on its own device, not src's.
        "all_on_own_device": _agree_all(m.state.device == torch.device(f"cuda:{rank}")),
    }


@RunIf(min_gpus=2)
def test_broadcast_module_buffer_only():
    """Buffer-only contract: ``broadcast_module`` syncs registered buffers on a module
    with no parameters to every rank, in place."""
    result = run_distributed_test(test_fn=_broadcast_module_buffer_only, world_size=2)
    # Rank 0's 0.5 * 4 == 2.0 must land on every rank; a rank still holding its own
    # 0.99 * 4 == 3.96 shows up as a spread. 0.5 is exact in fp32.
    assert result["state_sum"] == [2.0, 2.0]
    assert result["all_on_own_device"]


def _broadcast_module_nonpersistent_buffer_only(rank: int, world_size: int) -> dict:
    del world_size
    from fastgen.utils.distributed.module import broadcast_module

    device = torch.device("cuda", rank)
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.full((2,), 1.0 + rank, device=device))
    module.register_buffer("persistent", torch.full((2,), 3.0 + rank, device=device))
    module.register_buffer("nonpersistent", torch.full((2,), 5.0 + rank, device=device), persistent=False)

    broadcast_module(
        module,
        parameters=False,
        persistent_buffers=False,
        nonpersistent_buffers=True,
    )

    return {
        "parameter_mean": _agree(float(module.weight.mean().item())),
        "persistent_mean": _agree(float(module.persistent.mean().item())),
        "nonpersistent_mean": _agree(float(module.nonpersistent.mean().item())),
    }


@RunIf(min_gpus=2)
def test_broadcast_module_can_select_only_nonpersistent_buffers():
    result = run_distributed_test(test_fn=_broadcast_module_nonpersistent_buffer_only, world_size=2)
    assert result["parameter_mean"] == [1.0, 2.0]
    assert result["persistent_mean"] == [3.0, 4.0]
    assert result["nonpersistent_mean"] == [5.0, 5.0]


def _broadcast_module_uses_canonical_state_order(rank: int, world_size: int) -> dict:
    """Collective order follows qualified names, not rank-local registration order."""
    del world_size
    from fastgen.utils.distributed.module import broadcast_module

    device = torch.device("cuda", rank)
    module = torch.nn.Module()
    values = {"a": 1.0 + 10 * rank, "b": 2.0 + 10 * rank}
    names = ("a", "b") if rank == 0 else ("b", "a")
    for name in names:
        module.register_buffer(name, torch.full((2,), values[name], device=device))

    broadcast_module(module, parameters=False)

    return {
        "a_mean": _agree(float(module.a.mean().item())),
        "b_mean": _agree(float(module.b.mean().item())),
    }


@RunIf(min_gpus=2)
def test_broadcast_module_uses_canonical_state_order():
    result = run_distributed_test(test_fn=_broadcast_module_uses_canonical_state_order, world_size=2)
    assert result["a_mean"] == [1.0, 1.0]
    assert result["b_mean"] == [2.0, 2.0]


def _broadcast_module_dtensor_worker(rank, world_size):
    from unittest.mock import patch
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
    import fastgen.utils.distributed.module as module_sync

    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))
    for kind in ("parameter", "buffer"):
        for size, placement in ((4, Shard(0)), (3, Shard(0)), (2, Partial())):
            module = torch.nn.Module()
            values = torch.arange(size, dtype=torch.float32, device=device)
            tensor = (
                DTensor.from_local(values + rank, mesh, [placement], run_check=False)
                if isinstance(placement, Partial)
                else distribute_tensor(values, mesh, [placement])
            )
            if kind == "parameter":
                module.weight = torch.nn.Parameter(tensor)
                tensor = module.weight
            else:
                module.register_buffer("sharded", tensor)
            module.register_buffer("cache", torch.full((2,), float(rank + 1), device=device))
            before = tensor.to_local().detach().clone()
            with patch.object(module_sync, "_sync_params_and_buffers") as transport:
                with pytest.raises(RuntimeError, match="cannot broadcast non-replicated DTensors"):
                    module_sync.broadcast_module(module)
            assert not transport.called
            torch.testing.assert_close(tensor.to_local(), before)
            torch.testing.assert_close(module.cache, torch.full((2,), float(rank + 1), device=device))

            # Unselected sharded state does not prevent ordinary buffer synchronization.
            module_sync.broadcast_module(module, parameters=False, names={"cache"})
            torch.testing.assert_close(tensor.to_local(), before)
            torch.testing.assert_close(module.cache, torch.ones(2, device=device))

        module = torch.nn.Module()
        tensor = DTensor.from_local(torch.full((4,), float(rank + 1), device=device), mesh, [Replicate()])
        if kind == "parameter":
            module.weight = torch.nn.Parameter(tensor)
            tensor = module.weight
        else:
            module.register_buffer("replicated", tensor)
        module_sync.broadcast_module(module)
        torch.testing.assert_close(tensor.to_local(), torch.ones(4, device=device))
    return True


@RunIf(min_gpus=2)
def test_broadcast_module_rejects_selected_shards_and_accepts_replicated_dtensors():
    assert run_distributed_test(_broadcast_module_dtensor_worker, world_size=2, timeout=120, nccl_timeout=30)


def _broadcast_module_copies_back_noncontiguous_buffer(rank: int, world_size: int) -> dict:
    """Contiguous collective staging preserves a registered view and its backing storage."""
    del world_size
    from fastgen.utils.distributed.module import broadcast_module

    device = torch.device("cuda", rank)
    backing = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    backing.add_(100 * rank)
    view = backing.t()
    original_stride = view.stride()
    module = torch.nn.Module()
    module.register_buffer("state", view)

    broadcast_module(module, parameters=False)

    expected = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4).t()
    return {
        "values": _agree_all(torch.equal(module.state, expected)),
        "identity": _agree_all(module.state is view),
        "stride": _agree_all(module.state.stride() == original_stride),
        "storage": _agree_all(module.state.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()),
    }


@RunIf(min_gpus=2)
def test_broadcast_module_copies_back_noncontiguous_buffer():
    result = run_distributed_test(test_fn=_broadcast_module_copies_back_noncontiguous_buffer, world_size=2)
    assert result is not None
    assert all(result.values()), result


def _move_module_no_meta_no_broadcast(rank: int, world_size: int) -> dict:
    """No rank has meta — broadcast is skipped, so each rank keeps its
    locally-distinct values. Rank 0 verifies the values diverged across ranks
    via an all_reduce-min/max comparison."""
    from fastgen.utils.distributed.module import move_and_sync_module

    m = torch.nn.Linear(8, 8)
    val = 0.5 if rank == 0 else 0.99
    torch.nn.init.constant_(m.weight, val)
    torch.nn.init.constant_(m.bias, -val)

    move_and_sync_module(m, device="cpu")

    # This is the one test that wants *disagreement*: if the broadcast had fired,
    # min == max == 0.5 (rank 0 wins everywhere). If gating worked, min == 0.5 and
    # max == 0.99 (ranks kept their distinct values).
    local_val = float(m.weight.detach().mean().item())

    return {"rank0_val": local_val, "spread": _agree(local_val)}


@RunIf(min_gpus=2)
def test_move_module_no_meta_skips_broadcast():
    """Gating contract: when no rank has meta, ``move_and_sync_module`` does not broadcast. Callers needing an
    unconditional sync should call ``broadcast_module`` directly."""
    result = run_distributed_test(test_fn=_move_module_no_meta_no_broadcast, world_size=2)
    assert result["rank0_val"] == 0.5  # rank 0's own value, untouched (0.5 is exact in fp32)
    # Distinct min/max proves the broadcast did NOT fire across ranks.
    # 0.99 has no exact fp32 representation, so compare with a small tolerance.
    min_val, max_val = result["spread"]
    assert min_val == 0.5
    assert max_val == pytest.approx(0.99, abs=1e-3)


def _move_module_meta_to_cuda_after_cast(rank: int, world_size: int) -> dict:
    """Mirrors the eager ``prepare_module`` path: cast storage, then ``move_and_sync_module``.
    Rank 0 has real FP32 values (loaded weights), while non-rank-0 has FP16 meta state.
    The cast must normalize that intentional pre-broadcast mismatch on meta state too.
    After the call every rank should hold rank 0's values on CUDA in bfloat16."""
    from fastgen.utils.distributed.module import move_and_sync_module
    from fastgen.utils.module import cast_to_storage_dtypes, resolve_storage_dtypes

    target_device = torch.device(f"cuda:{rank}")

    if rank == 0:
        m = torch.nn.Linear(8, 8)
        torch.nn.init.constant_(m.weight, 0.5)
        torch.nn.init.constant_(m.bias, -0.25)
    else:
        with torch.device("meta"):
            m = torch.nn.Linear(8, 8, dtype=torch.float16)

    cast_to_storage_dtypes(m, resolve_storage_dtypes(m, torch.bfloat16, {}))
    move_and_sync_module(m, device=target_device)

    return {
        "weight_sum": _agree(float(m.weight.detach().to(torch.float32).sum().item())),
        "bias_sum": _agree(float(m.bias.detach().to(torch.float32).sum().item())),
        "all_on_cuda": _agree_all(m.weight.device.type == "cuda"),
        "all_bfloat16": _agree_all(m.weight.dtype == torch.bfloat16),
    }


@RunIf(min_gpus=2)
def test_move_module_meta_to_cuda_after_cast():
    """Locks the eager preparation contract: storage is cast on every rank, meta ranks included,
    before ``move_and_sync_module`` materializes, validates, and broadcasts cross-rank state.
    ``prepare_module`` uses this order for the teacher and student networks under ``meta_init=True``."""
    result = run_distributed_test(test_fn=_move_module_meta_to_cuda_after_cast, world_size=2)
    # Linear(8, 8) with weight=0.5 sums to 32, bias=-0.25 sums to -2.0 (cast back to
    # fp32 in the probe to dodge bf16 rounding). Both bounds are checked, so the meta
    # rank cannot quietly disagree.
    assert all(abs(value - 32.0) < 1e-1 for value in result["weight_sum"])
    assert all(abs(value - (-2.0)) < 1e-2 for value in result["bias_sum"])
    assert result["all_on_cuda"]
    assert result["all_bfloat16"]
