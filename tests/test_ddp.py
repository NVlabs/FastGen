# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for DDP initialization, its mixed backend, and end-to-end DDP behavior.

Tests require at least 2 GPUs and are run using torch.multiprocessing.spawn
to simulate distributed training.

Usage:
    pytest tests/test_ddp.py -v

    # Or run with specific tests:
    pytest tests/test_ddp.py::test_both_backends_alive -v
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from fastgen.methods.model import FastGenModel
from fastgen.networks.network import FastGenModule
from fastgen.utils.distributed.ddp import model_to_ddp
from tests.utils import RunIf, run_distributed_test


def _clear_launcher_env(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)


def test_init_is_idempotent(monkeypatch):
    import fastgen.utils.distributed.ddp as ddp

    monkeypatch.setattr(ddp.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        ddp.dist,
        "init_process_group",
        lambda **kwargs: pytest.fail("initialized an existing process group"),
    )

    assert ddp.init(required=True)


@pytest.mark.parametrize("launcher_env", [{}, {"WORLD_SIZE": "2"}])
def test_init_without_complete_launcher_is_optional_or_required(monkeypatch, launcher_env):
    import fastgen.utils.distributed.ddp as ddp

    _clear_launcher_env(monkeypatch)
    for name, value in launcher_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ddp.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp.dist, "is_initialized", lambda: False)

    assert not ddp.init()
    with pytest.raises(RuntimeError, match="requires torchrun"):
        ddp.init(required=True)


def test_init_uses_complete_launcher_environment(monkeypatch):
    import fastgen.utils.distributed.ddp as ddp

    calls = []
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("NCCL_TIMEOUT", "17")
    monkeypatch.setattr(ddp.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(ddp.dist, "init_process_group", lambda **kwargs: calls.append(("init", kwargs)))
    monkeypatch.setattr(ddp.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(ddp.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ddp.torch.cuda, "set_device", lambda device: calls.append(("device", device)))

    assert ddp.init()
    assert calls == [
        (
            "init",
            {
                "backend": "cuda:nccl,cpu:gloo",
                "init_method": "env://",
                "rank": 2,
                "world_size": 4,
                "timeout": ddp.timedelta(seconds=17),
            },
        ),
        ("device", 1),
    ]


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


def _total(value: float) -> float:
    """Sum ``value`` across ranks — for counting work that should happen on one rank only."""
    local = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return float(local.item())


def _both_backends_alive(rank: int, world_size: int) -> dict:
    """Default PG dispatches CUDA tensors via nccl, CPU tensors via gloo."""
    cuda_t = torch.tensor([float(rank)], device=f"cuda:{rank}")
    dist.broadcast(cuda_t, src=0)  # dispatched via nccl
    cpu_t = torch.tensor([float(rank) + 100.0])
    dist.broadcast(cpu_t, src=0)  # dispatched via gloo
    return {"cuda_value": _agree(float(cuda_t.item())), "cpu_value": _agree(float(cpu_t.item()))}


@RunIf(min_gpus=2)
def test_both_backends_alive():
    """Default PG handles both CUDA (nccl) and CPU (gloo) tensors in the same job."""
    result = run_distributed_test(test_fn=_both_backends_alive, world_size=2)
    # Each broadcast must reach every rank, not merely leave rank 0's own value intact.
    assert result["cuda_value"] == [0.0, 0.0]
    assert result["cpu_value"] == [100.0, 100.0]


def _load_pretrained_fetches_s3_once(rank: int, world_size: int, cache_dir: str, payload_path: str) -> dict:
    """``_load_pretrained_model`` must pull from S3 on one rank only, then let every rank
    read the mirror. All ranks read; only the download is rank-guarded."""
    import fastgen.methods.model as model_module
    import fastgen.utils.s3 as s3

    with open(payload_path, "rb") as handle:
        blob = handle.read()

    fetches = []

    def fake_download(path, output):
        fetches.append(path)
        output.write(blob)
        return 0

    s3.resumable_download = fake_download
    os.environ["FASTGEN_OUTPUT_ROOT"] = cache_dir

    net = torch.nn.Linear(8, 8)
    torch.nn.init.zeros_(net.weight)
    torch.nn.init.zeros_(net.bias)
    model_module.FastGenModel._load_pretrained_model(net, "s3://bucket/pretrained.pth", device="cpu")

    return {
        "total_fetches": _total(len(fetches)),
        "weight_sum": _agree(float(net.weight.detach().sum().item())),
        "bias_sum": _agree(float(net.bias.detach().sum().item())),
    }


@RunIf(min_gpus=2)
def test_load_pretrained_model_fetches_s3_once_across_ranks(tmp_path):
    """One S3 download for the whole job, yet every rank ends up with the weights.

    Both halves matter: counting fetches alone would pass if the other ranks never got
    the weights, and checking weights alone would pass if every rank re-downloaded.
    """
    payload_path = tmp_path / "pretrained.pth"
    torch.save({"weight": torch.full((8, 8), 0.5), "bias": torch.full((8,), -0.25)}, payload_path)

    result = run_distributed_test(
        test_fn=_load_pretrained_fetches_s3_once,
        world_size=2,
        cache_dir=str(tmp_path / "root"),
        payload_path=str(payload_path),
    )

    assert result["total_fetches"] == 1.0, "the checkpoint must be downloaded once, on rank 0"
    # Zero-initialized on every rank, so a rank that never loaded stays at 0.0.
    assert result["weight_sum"] == [32.0, 32.0]
    assert result["bias_sum"] == [-2.0, -2.0]


def _rank_guarded_s3_download(rank: int, world_size: int, cache_dir: str, payload_path: str) -> dict:
    """Rank 0 primes the shared mirror, the barrier publishes it, then every rank resolves
    to it locally. Mirrors the ``_load_pretrained_model`` call shape."""
    import fastgen.utils.cache_utils as cache_utils
    import fastgen.utils.io_utils as io_utils
    import fastgen.utils.s3 as s3
    from fastgen.utils.distributed.collectives import synchronize
    from fastgen.utils.distributed import is_rank0

    with open(payload_path, "rb") as handle:
        blob = handle.read()

    fetches = []

    def fake_download(path, output):
        fetches.append(path)
        output.write(blob)
        return 0

    s3.resumable_download = fake_download

    if is_rank0():
        cache_utils.cache("s3://bucket/model.pth", cache_dir=cache_dir, strict=False)
    synchronize()

    checkpoint_path = cache_utils.resolve_cached_path("s3://bucket/model.pth", cache_dir=cache_dir)
    state = io_utils.load_checkpoint_state_dict(checkpoint_path, device="cpu")

    return {
        "total_fetches": _total(len(fetches)),
        "weight_sum": _agree(float(state["w"].sum().item())),
    }


@RunIf(min_gpus=2)
def test_cache_downloads_once_across_ranks(tmp_path):
    """Rank-guard contract: ``cache`` behind ``is_rank0`` plus a barrier means exactly one
    rank pulls from S3, while every rank still ends up with the weights.

    ``cache`` runs no collectives itself, so the barrier here — not the helper — is what
    makes the mirror visible to the other ranks.
    """
    payload_path = tmp_path / "payload.pth"
    torch.save({"w": torch.full((8, 8), 0.5), "b": torch.full((8,), -0.25)}, payload_path)

    result = run_distributed_test(
        test_fn=_rank_guarded_s3_download,
        world_size=2,
        cache_dir=str(tmp_path / "cache"),
        payload_path=str(payload_path),
    )

    # Without the guard every rank fetches, so this would be 2.0 on a 2-rank job.
    assert result["total_fetches"] == 1.0
    # 0.5 * 8 * 8 == 32.0 must still land on both ranks, from the mirror.
    assert result["weight_sum"] == [32.0, 32.0]


# ---------------------------------------------------------------------------
# Split model/optimization precision through ``FastGenModel.on_train_begin`` + ``model_to_ddp``
# ---------------------------------------------------------------------------


class _SplitPrecisionNet(FastGenModule):
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
        output = (trainable + frozen) * self.scale.to(device=trainable.device, dtype=trainable.dtype)
        return output.to(dtype=x.dtype)


class _SplitPrecisionModel(FastGenModel):
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


def _split_precision_model(rank: int) -> _SplitPrecisionModel:
    """fp32 method state, bfloat16 network compute, fp32 trainable storage; prepared by ``on_train_begin``."""
    model = _SplitPrecisionModel.__new__(_SplitPrecisionModel)
    torch.nn.Module.__init__(model)
    model.module_dict = {}
    model.config = SimpleNamespace(
        inference=False,
        ddp_find_unused_parameters=False,
        enable_preprocessors=False,
        extra_fsdp_modules=[],
        net_adapter=SimpleNamespace(type=None),
        autocast_cache_enabled=False,
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
    model.use_ema = []
    torch.manual_seed(1234)
    model.net = _SplitPrecisionNet()
    model.module_dict.update(model.model_dict)
    with torch.no_grad():
        for parameter in model.net.parameters():
            parameter.fill_(0.05)
        model.net.scale.fill_(1.0)
    model.on_train_begin(fsdp_context=None)
    return model


def _ddp_split_precision_impl(rank: int, world_size: int) -> dict:
    del world_size
    model = _split_precision_model(rank)

    # A trainable component stores all of its floating state in the optimization precision (the
    # frozen layer included: ``requires_grad`` may change during training); FSDP/plain agree.
    storage_ok = all(parameter.dtype == torch.float32 for parameter in model.net.parameters())
    storage_ok = storage_ok and model.net.scale.dtype == torch.float32
    precision_ok = model.net.precision_model == torch.bfloat16

    wrapped = model_to_ddp(model)
    optimizer = torch.optim.SGD(model.net.trainable_layer.parameters(), lr=0.05)
    before = model.net.trainable_layer.weight.detach().clone()
    generator = torch.Generator(device=model.device).manual_seed(8000 + rank)
    data = {
        "x": torch.randn(4, 8, generator=generator, device=model.device, dtype=torch.float32),
        "target": torch.zeros(4, 8, device=model.device, dtype=torch.float32),
    }
    with model.autocast():
        loss_map, outputs = wrapped.single_train_step(data, iteration=0)
    loss_map["total_loss"].backward()

    grad = model.net.trainable_layer.weight.grad
    grad_norm_range = _agree(float(grad.float().norm().item()))
    grad_ok = grad.dtype == torch.float32 and torch.isfinite(grad).all().item()
    frozen_ok = all(parameter.grad is None for parameter in model.net.frozen_layer.parameters())
    compute_ok = model.net.last_compute_dtypes == (torch.bfloat16, torch.bfloat16)
    output_ok = outputs["prediction"].dtype == torch.float32

    optimizer.step()
    update_ok = not torch.equal(before, model.net.trainable_layer.weight.detach())
    synchronized = abs(grad_norm_range[1] - grad_norm_range[0]) < 1e-7

    return {
        "storage_ok": _agree_all(storage_ok),
        "precision_ok": _agree_all(precision_ok),
        "compute_ok": _agree_all(compute_ok),
        "output_ok": _agree_all(output_ok),
        "grad_ok": _agree_all(bool(grad_ok)),
        "frozen_ok": _agree_all(frozen_ok),
        "update_ok": _agree_all(update_ok),
        "gradients_synchronized": _agree_all(synchronized),
        "gradient_norm_range": grad_norm_range,
    }


@RunIf(min_gpus=2)
def test_ddp_split_precision_forward_backward():
    result = run_distributed_test(test_fn=_ddp_split_precision_impl, world_size=2, timeout=180)
    assert result is not None
    for key in ("storage_ok", "precision_ok", "compute_ok", "output_ok", "grad_ok", "frozen_ok", "update_ok"):
        assert result[key], key
    assert result["gradients_synchronized"], result["gradient_norm_range"]
