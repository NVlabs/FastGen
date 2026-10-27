# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpointer tests.

Monolithic and DCP save/load of model, optimizer, scheduler and iteration state; EMA, teacher and
extra-FSDP-module handling; adapter-only and permissive loads; pretrained key maps; S3 paths; completion
markers, publication and auto-resume; DCP manifests and plan agreement; meta-init materialization and
topology checks; plus the two-rank FSDP cases. Tests use small networks and temporary directories.
"""

import contextlib
import copy
import gc
import io
import tempfile
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from fastgen.methods import SFTModel, DMD2Model
from fastgen.trainer import Trainer
from fastgen.networks.network import FastGenModule
from fastgen.configs.methods.config_sft import ModelConfig as SFTModelConfig
from fastgen.configs.methods.config_dmd2 import ModelConfig as DMD2ModelConfig
from fastgen.configs.net import EDM_CIFAR10_Config
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.configs.config import BaseCheckpointerConfig
from fastgen.utils.checkpointer import (
    Checkpointer,
    is_dcp_checkpoint,
    validate_checkpoint_path,
)
from fastgen.utils.io_utils import load_training_checkpoint, path_exists
from tests.utils import RunIf, run_distributed_test
from fastgen.utils.env import set_env_vars
from fastgen.utils.basic_utils import clear_gpu_memory


# =============================================================================
# Helper Functions
# =============================================================================


class _PrecisionLinear(FastGenModule, torch.nn.Linear):
    pass


def _get_small_edm_config():
    """Get a small EDM config for fast testing."""
    config = copy.deepcopy(EDM_CIFAR10_Config)
    opts = [
        "-",
        "backbone.img_resolution=8",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=False",
    ]
    return override_config_with_opts(config, opts)


def _get_small_discriminator_config():
    """Get a small discriminator config for DMD2 testing."""
    from fastgen.configs.discriminator import Discriminator_EDM_CIFAR10_Config

    opts = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    return override_config_with_opts(copy.deepcopy(Discriminator_EDM_CIFAR10_Config), opts)


def _create_sft_model(use_ema: bool = False, device: str = "cpu") -> SFTModel:
    """Create a small SFT model for testing."""
    config = SFTModelConfig()
    config.net = _get_small_edm_config()
    config.input_shape = [3, 8, 8]
    config.precision = "float32"
    config.device = device
    config.pretrained_model_path = None
    config.use_ema = use_ema

    model = SFTModel(config)
    model.on_train_begin()
    model.init_optimizers()
    return model


def _create_dmd2_model(use_ema: bool = False, device: str = "cpu", with_gan: bool = False) -> DMD2Model:
    """Create a small DMD2 model for testing."""
    config = DMD2ModelConfig()
    config.net = _get_small_edm_config()
    config.input_shape = [3, 8, 8]
    config.precision = "float32"
    config.device = device
    config.pretrained_model_path = None
    config.use_ema = use_ema
    config.student_update_freq = 2

    # Disable GAN by default to simplify testing
    if with_gan:
        config.gan_loss_weight_gen = 0.001
        config.discriminator = _get_small_discriminator_config()
    else:
        config.gan_loss_weight_gen = 0.0

    # Don't add teacher to FSDP dict for standard checkpointing tests
    config.extra_fsdp_modules = []

    model = DMD2Model(config)
    model.on_train_begin()
    model.init_optimizers()
    return model


def _modify_model_weights(model, scale: float = 2.0):
    """Modify model weights by scaling them."""
    with torch.no_grad():
        for param in model.net.parameters():
            param.mul_(scale)


def _compare_state_dicts(state_dict1, state_dict2, tolerance: float = 1e-6) -> bool:
    """Compare two state dicts and return True if they match."""
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]

        if tensor1.shape != tensor2.shape:
            return False

        if not torch.allclose(tensor1.float(), tensor2.float(), atol=tolerance):
            return False

    return True


def _module_placement_errors(
    module: torch.nn.Module,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
    dtype_overrides: dict[str, torch.dtype] | None = None,
) -> list[str]:
    """Describe misplaced or wrongly cast parameters and buffers in an FSDP module."""
    tensors = [
        *((f"parameter:{name}", tensor) for name, tensor in module.named_parameters()),
        *((f"buffer:{name}", tensor) for name, tensor in module.named_buffers()),
    ]
    errors = []
    for name, tensor in tensors:
        local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        if local.device != expected_device:
            errors.append(f"{name} is on {local.device}, expected {expected_device}")
        tensor_dtype = (dtype_overrides or {}).get(name, expected_dtype)
        if local.is_floating_point() and local.dtype != tensor_dtype:
            errors.append(f"{name} has dtype {local.dtype}, expected {tensor_dtype}")
    return errors


def _get_checkpointer_config(save_dir: str) -> BaseCheckpointerConfig:
    """Create a checkpointer config with the given save directory."""
    config = BaseCheckpointerConfig()
    config.save_dir = save_dir
    config.use_s3 = False
    return config


def test_dcp_checkpointer_rejects_save_adapter_only(tmp_path):
    """Construction stays legal (inference uses it as a loader); DCP saving must raise."""
    config = _get_checkpointer_config(str(tmp_path))
    config.save_adapter_only = True
    with pytest.raises(NotImplementedError, match="save_adapter_only"):
        Checkpointer(config, save_sharded=True).save(torch.nn.ModuleDict())


def _tiny_lora_net(seed: int):
    """Plain module with a LoRA adapter on ``linear`` (PEFT low-level injection)."""
    from fastgen.networks.lora_utils import LoRAMixin

    class _LoRANet(LoRAMixin, torch.nn.Module):
        lora_base_name = ""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)
            self.extra = torch.nn.Linear(4, 4)
            self.lora_extra_trainable_modules = ("extra",)

        def forward(self, x):
            return self.linear(x) + self.extra(x)

    torch.manual_seed(seed)
    net = _LoRANet()
    net.init_lora(rank=2, alpha=2, target_modules=["linear"])
    return net


def test_adapter_only_checkpoint_round_trips_and_stays_strict_for_the_adapter(tmp_path, monkeypatch):
    """``save_adapter_only`` stores just the adapter; the strict default requires a target's adapter and leaves
    its frozen base optional, so both save modes load the same way but an incomplete adapter fails."""
    import fastgen.utils.logging_utils as fg_logger

    config = _get_checkpointer_config(str(tmp_path))
    config.save_adapter_only = True
    checkpointer = Checkpointer(config)

    net = _tiny_lora_net(seed=0)
    with torch.no_grad():
        for tensor in net.adapter_state_dict().values():
            tensor.normal_()
    expected = {name: tensor.clone() for name, tensor in net.adapter_state_dict().items()}

    path = checkpointer.save(torch.nn.ModuleDict({"net": net}), iteration=3)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint["model"]["net"]
    assert set(saved) == set(expected)
    assert any("lora_" in name for name in saved)
    assert {"extra.weight", "extra.bias"} <= set(saved)
    assert all("lora_" in name or name.startswith("extra.") for name in saved)

    # Loading follows the target, not the current job's save policy; the omitted base keeps its values.
    load_config = _get_checkpointer_config(str(tmp_path))
    target = _tiny_lora_net(seed=1)
    base = {name: tensor.clone() for name, tensor in target.state_dict().items() if name not in expected}
    assert Checkpointer(load_config).load(torch.nn.ModuleDict({"net": target}), path=path) == 3
    for name, tensor in target.adapter_state_dict().items():
        torch.testing.assert_close(tensor, expected[name])
    for name, tensor in base.items():
        assert torch.equal(target.state_dict()[name], tensor)

    # Saved tensors that the target keeps outside its adapter (e.g. an extra module it no longer trains)
    # make the file neither adapter-only nor full: strict loading refuses it, permissive loading names them.
    base_names = "['linear.base_layer.bias', 'linear.base_layer.weight']"
    frozen_extra = _tiny_lora_net(seed=2)
    frozen_extra.lora_extra_trainable_modules = ()
    with pytest.raises(RuntimeError, match=re.escape(f"2 not loaded (first 8: {base_names})")):
        checkpointer.load(torch.nn.ModuleDict({"net": frozen_extra}), path=path)
    warnings = []
    monkeypatch.setattr(fg_logger, "warning", warnings.append)
    frozen_extra = _tiny_lora_net(seed=2)
    frozen_extra.lora_extra_trainable_modules = ()
    checkpointer.load(torch.nn.ModuleDict({"net": frozen_extra}), path=path, strict=False)
    torch.testing.assert_close(frozen_extra.extra.weight, expected["extra.weight"])
    assert any("2 of 4 non-required tensors loaded (first 8: ['extra.bias', 'extra.weight'])" in m for m in warnings)

    # A full checkpoint loads its base too, and only completely.
    full_config = _get_checkpointer_config(str(tmp_path))
    full_source = _tiny_lora_net(seed=3)
    full_path = Checkpointer(full_config).save(torch.nn.ModuleDict({"net": full_source}), iteration=4)
    full_target = _tiny_lora_net(seed=4)
    checkpointer.load(torch.nn.ModuleDict({"net": full_target}), path=full_path)
    for name, tensor in full_target.state_dict().items():
        torch.testing.assert_close(tensor, full_source.state_dict()[name])
    incomplete = torch.load(full_path, map_location="cpu", weights_only=False)
    incomplete["model"]["net"].pop("linear.base_layer.weight")
    incomplete_path = tmp_path / "incomplete_full.pth"
    torch.save(incomplete, incomplete_path)
    with pytest.raises(RuntimeError, match=re.escape("1 not loaded (first 8: ['linear.base_layer.weight'])")):
        checkpointer.load(torch.nn.ModuleDict({"net": _tiny_lora_net(seed=5)}), path=str(incomplete_path))

    missing_names = [next(name for name in expected if "lora_" in name), "extra.weight"]
    for missing_name in missing_names:
        partial = copy.deepcopy(checkpoint)
        partial["model"]["net"].pop(missing_name)
        partial_path = tmp_path / f"partial_{missing_name.replace('.', '_')}.pth"
        torch.save(partial, partial_path)
        with pytest.raises(RuntimeError, match=re.escape(f"missing_keys=['{missing_name}']")):
            checkpointer.load(torch.nn.ModuleDict({"net": _tiny_lora_net(seed=5)}), path=str(partial_path))

    oversized = copy.deepcopy(checkpoint)
    oversized["model"]["net"]["outside_adapter_namespace.weight"] = torch.ones(1)
    oversized_path = tmp_path / "oversized_adapter.pth"
    torch.save(oversized, oversized_path)
    with pytest.raises(RuntimeError, match=re.escape("unexpected_keys=['outside_adapter_namespace.weight']")):
        checkpointer.load(torch.nn.ModuleDict({"net": _tiny_lora_net(seed=6)}), path=str(oversized_path))

    # A target without an adapter rejects the adapter tensors before copying anything.
    plain = _checkpoint_test_net(weight=2.0)
    with pytest.raises(RuntimeError, match="no initialized adapter"):
        checkpointer.load(torch.nn.ModuleDict({"net": plain}), path=path)
    assert torch.equal(plain.weight, torch.full((4, 4), 2.0))


def test_permissive_load_accepts_an_incomplete_adapter_checkpoint(tmp_path):
    source = _tiny_lora_net(seed=0)
    state = dict(source.adapter_state_dict())
    missing_name = next(iter(state))
    state.pop(missing_name)
    state["outside_adapter_namespace.weight"] = torch.ones(1)
    checkpoint_path = tmp_path / "partial_adapter.pth"
    torch.save({"model": {"net": state}, "iteration": 7}, checkpoint_path)
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)))

    with pytest.raises(RuntimeError, match=re.escape(f"missing_keys=['{missing_name}']")):
        checkpointer.load(torch.nn.ModuleDict({"net": _tiny_lora_net(seed=1)}), path=str(checkpoint_path))

    target = _tiny_lora_net(seed=1)
    missing_before = target.state_dict()[missing_name].clone()
    assert checkpointer.load(torch.nn.ModuleDict({"net": target}), path=str(checkpoint_path), strict=False) == 7

    assert torch.equal(target.state_dict()[missing_name], missing_before)
    for name, tensor in source.adapter_state_dict().items():
        if name != missing_name:
            torch.testing.assert_close(target.state_dict()[name], tensor)


@pytest.mark.parametrize("use_s3", [False, True], ids=["local", "s3"])
def test_dcp_optimizer_save_failure_invalidates_reused_marker(tmp_path, monkeypatch, use_s3):
    """A failed optimizer DCP save cannot leave ``latest_ar`` resumable.

    ``latest_ar`` is reused across autoresume requests, so merely avoiding a new
    scalar save is insufficient: its previous completion marker must be removed
    before any model or optimizer shard is overwritten.
    """

    class InjectedDCPFailure(BaseException):
        pass

    net = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(net.parameters())
    net(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    config = _get_checkpointer_config(str(tmp_path))
    deleted_s3_paths = []
    published_s3_paths = []
    monkeypatch.setattr(
        Checkpointer,
        "get_storage_writer",
        lambda self, checkpoint_path: checkpoint_path,
    )
    if use_s3:
        config.save_dir = "s3://bucket/fastgen/checkpoints"
        requested_path = os.path.join(config.save_dir, "latest_ar.pth")
        marker_path = requested_path
        monkeypatch.setattr("fastgen.utils.s3.delete", deleted_s3_paths.append)
        monkeypatch.setattr(
            "fastgen.utils.s3.save",
            lambda path, data: published_s3_paths.append((path, data)),
        )
    else:
        marker_path = tmp_path / "latest_ar.pth"
        requested_path = str(marker_path)
        torch.save({"iteration": 1}, marker_path)
    save_calls = []

    def fail_optimizer_save(state_dict, storage_writer):
        del state_dict, storage_writer
        if use_s3:
            assert deleted_s3_paths == [marker_path]
        else:
            assert not marker_path.exists()
        save_calls.append(None)
        if len(save_calls) == 2:
            raise InjectedDCPFailure("injected optimizer save failure")

    monkeypatch.setattr("fastgen.utils.checkpointer.dcp.save", fail_optimizer_save)

    checkpointer = Checkpointer(config, save_sharded=True)
    with pytest.raises(InjectedDCPFailure, match="injected optimizer save failure"):
        checkpointer.save(
            model_dict=torch.nn.ModuleDict({"net": net}),
            optimizer_dict={"net": optimizer},
            path=requested_path,
            iteration=2,
        )

    assert len(save_calls) == 2, "the injected failure must occur during optimizer saving"
    if use_s3:
        assert deleted_s3_paths == [marker_path]
        assert not published_s3_paths, "a failed DCP save must not publish an S3 completion marker"
    else:
        assert not marker_path.exists(), "a failed DCP save must not leave a resumable completion marker"


@pytest.mark.parametrize(
    ("requested_path", "expected_path"),
    [
        (None, "s3://bucket/fastgen/checkpoints/0000007.pth"),
        (
            "s3://other-bucket/run/latest_ar.pth",
            "s3://other-bucket/run/latest_ar.pth",
        ),
    ],
    ids=["iteration", "explicit"],
)
def test_checkpointer_saves_s3_paths_as_given(monkeypatch, requested_path, expected_path):
    """Iteration paths below an S3 ``save_dir`` and explicit S3 paths are written without rewriting."""
    config = _get_checkpointer_config("s3://bucket/fastgen/checkpoints")
    saved_objects = []
    monkeypatch.setattr(
        "fastgen.utils.s3.save",
        lambda path, data: saved_objects.append((path, data)),
    )

    resolved_path = Checkpointer(config).save(
        model_dict=torch.nn.ModuleDict(),
        path=requested_path,
        iteration=7,
    )

    assert resolved_path == expected_path
    assert [path for path, _ in saved_objects] == [expected_path]
    state = torch.load(io.BytesIO(saved_objects[0][1]), weights_only=False)
    assert state["iteration"] == 7


def test_local_checkpoint_publication_atomically_replaces_existing_file(tmp_path, monkeypatch):
    """The old marker remains readable until a complete temporary file replaces it."""
    checkpoint_path = tmp_path / "latest_ar.pth"
    torch.save({"iteration": 1}, checkpoint_path)
    original_replace = os.replace
    replacements = []

    def inspect_then_replace(source, destination):
        assert destination == str(checkpoint_path)
        assert torch.load(checkpoint_path, weights_only=False)["iteration"] == 1
        assert torch.load(source, weights_only=False)["iteration"] == 2
        replacements.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr("fastgen.utils.cache_utils.os.replace", inspect_then_replace)

    Checkpointer(_get_checkpointer_config(str(tmp_path)))._save_checkpoint({"iteration": 2}, str(checkpoint_path))

    assert len(replacements) == 1
    assert torch.load(checkpoint_path, weights_only=False)["iteration"] == 2
    assert list(tmp_path.glob("*.tmp")) == []


def test_local_checkpoint_publication_failure_preserves_existing_file(tmp_path, monkeypatch):
    """A serialization failure removes its temporary file without damaging the old marker."""
    checkpoint_path = tmp_path / "latest_ar.pth"
    torch.save({"iteration": 1}, checkpoint_path)
    original_contents = checkpoint_path.read_bytes()

    def fail_after_partial_write(_state, destination):
        destination.write(b"partial checkpoint")
        raise OSError("injected serialization failure")

    monkeypatch.setattr("fastgen.utils.checkpointer.torch.save", fail_after_partial_write)

    with pytest.raises(OSError, match="injected serialization failure"):
        Checkpointer(_get_checkpointer_config(str(tmp_path)))._save_checkpoint({"iteration": 2}, str(checkpoint_path))

    assert checkpoint_path.read_bytes() == original_contents
    assert list(tmp_path.glob("*.tmp")) == []


def test_local_checkpoint_publication_removes_directory_orphans(tmp_path):
    """A later iteration sweeps full temporary checkpoints that will never reuse their path."""
    checkpoint_path = tmp_path / "0000201.pth"
    stale_paths = [
        tmp_path / "0000200.pth.11111111111111111111111111111111.tmp",
        tmp_path / "latest_ar.pth.22222222222222222222222222222222.tmp",
    ]
    unrelated_temporary = tmp_path / "model.pt.33333333333333333333333333333333.tmp"
    for stale_path in stale_paths:
        stale_path.write_bytes(b"orphaned checkpoint")
    unrelated_temporary.write_bytes(b"not a FastGen checkpoint temporary")

    Checkpointer(_get_checkpointer_config(str(tmp_path)))._save_checkpoint({"iteration": 201}, str(checkpoint_path))

    assert all(not stale_path.exists() for stale_path in stale_paths)
    assert unrelated_temporary.is_file()
    assert torch.load(checkpoint_path, weights_only=False)["iteration"] == 201


def test_autoresume_fallback_uses_resolved_checkpoint_path(monkeypatch):
    """The already-saved fallback supplies autoresume with a loadable S3 path."""
    config = _get_checkpointer_config("s3://bucket/fastgen/checkpoints")
    requested_resumes = []
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(checkpointer=config))
    trainer.checkpointer = Checkpointer(config)
    trainer.auto_resume = SimpleNamespace(
        termination_requested=lambda: True,
        get_resume_details=lambda: {},
        request_resume=lambda user_dict: requested_resumes.append(user_dict),
    )
    monkeypatch.setattr("fastgen.trainer.is_rank0", lambda: True)

    assert trainer.auto_resume_exit(
        SimpleNamespace(device=torch.device("cpu")),
        iteration=7,
        skip_if_just_saved=True,
    )
    assert requested_resumes == [{"save_path": "s3://bucket/fastgen/checkpoints/0000007.pth"}]


@pytest.mark.parametrize("use_s3", [False, True], ids=["local", "s3"])
def test_dcp_empty_optimizer_manifest_excludes_stale_metadata(tmp_path, monkeypatch, use_s3):
    """Reusing a prefix cannot make an unstepped optimizer load old state."""
    net = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(net.parameters())
    config = _get_checkpointer_config(str(tmp_path))
    storage_paths = []
    monkeypatch.setattr(
        Checkpointer,
        "get_storage_writer",
        lambda self, checkpoint_path: storage_paths.append(checkpoint_path) or checkpoint_path,
    )
    monkeypatch.setattr("fastgen.utils.checkpointer.dcp.save", lambda state_dict, storage_writer: None)

    if use_s3:
        config.save_dir = "s3://bucket/fastgen/checkpoints"
        requested_path = os.path.join(config.save_dir, "latest_ar.pth")
        marker_path = requested_path
        prefix = marker_path[: -len(".pth")]
        metadata_path = os.path.join(f"{prefix}.net_optim", ".metadata")
        objects = {marker_path: b"old marker", metadata_path: b"old metadata"}

        def delete_object(path):
            objects.pop(path, None)

        monkeypatch.setattr("fastgen.utils.s3.delete", delete_object)
        monkeypatch.setattr("fastgen.utils.s3.save", objects.__setitem__)
    else:
        marker_path = tmp_path / "latest_ar.pth"
        requested_path = str(marker_path)
        prefix = str(marker_path)[: -len(".pth")]
        metadata_path = os.path.join(f"{prefix}.net_optim", ".metadata")
        torch.save({"iteration": 1}, marker_path)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "wb") as metadata_file:
            metadata_file.write(b"old metadata")

    resolved_path = Checkpointer(config, save_sharded=True).save(
        model_dict=torch.nn.ModuleDict({"net": net}),
        optimizer_dict={"net": optimizer},
        path=requested_path,
        iteration=2,
    )

    assert resolved_path == str(marker_path)
    assert storage_paths == [f"{prefix}.net_model"]
    assert not optimizer.state, "checking an unstepped optimizer must not synthesize moments"
    if use_s3:
        assert metadata_path in objects
        state = torch.load(io.BytesIO(objects[marker_path]), weights_only=False)
    else:
        assert os.path.exists(metadata_path)
        state = torch.load(marker_path, weights_only=False)
    assert state["iteration"] == 2
    assert state["dcp_manifest"] == {"models": ("net",), "optimizers": ()}


def _dcp_exports_alias_live_state(exports: dict, module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """Report whether captured ``dcp.save`` payloads share storage with exactly the live state they export."""
    from fastgen.utils.module import local_tensor, module_state_tensors

    def storages(tensors) -> set[int]:
        return {local_tensor(tensor).untyped_storage().data_ptr() for tensor in tensors if torch.is_tensor(tensor)}

    live_model = (tensor for _kind, _name, tensor in module_state_tensors(module, nonpersistent_buffers=False))
    live_optimizer = (value for state in optimizer.state.values() for value in state.values())
    exported_optimizer = (value for state in exports["net_optim"]["state"].values() for value in state.values())
    return {
        "model": storages(exports["net_model"].values()) == storages(live_model),
        "optimizer": storages(exported_optimizer) == storages(live_optimizer),
    }


def test_dcp_save_exports_alias_live_state(tmp_path, monkeypatch):
    """DCP saves hold every export until the writes finish, so exports must share live storage, not copy it."""
    net = torch.nn.Linear(2, 2)
    net.register_buffer("scale", torch.ones(2))
    optimizer = torch.optim.AdamW(net.parameters())
    net(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    exports = {}
    monkeypatch.setattr(
        Checkpointer, "get_storage_writer", lambda self, checkpoint_path: checkpoint_path.rsplit(".", 1)[1]
    )
    monkeypatch.setattr(
        "fastgen.utils.checkpointer.dcp.save",
        lambda state_dict, storage_writer: exports.__setitem__(storage_writer, state_dict),
    )
    Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=True).save(
        torch.nn.ModuleDict({"net": net}), optimizer_dict={"net": optimizer}, iteration=1
    )

    assert sorted(exports) == ["net_model", "net_optim"]
    assert _dcp_exports_alias_live_state(exports, net, optimizer) == {"model": True, "optimizer": True}


def _fsdp_dcp_save_exports_alias_live_state_impl(
    rank: int, world_size: int, checkpoint_dir: str, cpu_offload: bool
) -> dict:
    """Capture each rank's DCP exports of a stepped FSDP2 model and compare them with its live shards."""
    import fastgen.utils.checkpointer as checkpointer_module
    from fastgen.utils.distributed.fsdp import create_fsdp_context
    from fastgen.utils.module import local_tensor

    config = SFTModelConfig()
    config.net = _get_small_edm_config()
    config.input_shape = [3, 8, 8]
    config.precision = "float32"
    config.precision_model = "bfloat16"
    config.precision_opt = "float32"
    config.device = "cuda"
    config.pretrained_model_path = None
    config.use_ema = False
    config.meta_init = True
    model = SFTModel(config)
    model.on_train_begin(
        fsdp_context=create_fsdp_context(
            reduce_dtype=model.precision_fsdp_reduce, min_num_params=1_000_000, apply_cpu_offload=cpu_offload
        )
    )
    model.init_optimizers()

    # Saving skips an unstepped optimizer; zero gradients create its moments without a forward pass.
    for parameter in model.net.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)
    model.net_optimizer.step()
    model.net_optimizer.zero_grad(set_to_none=True)

    exports = {}
    original_dcp_save = checkpointer_module.dcp.save
    checkpointer_module.dcp.save = lambda state_dict, storage_writer: exports.__setitem__(storage_writer, state_dict)
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=True)
    checkpointer.get_storage_writer = lambda checkpoint_path: checkpoint_path.rsplit(".", 1)[1]
    try:
        checkpointer.save(model.model_dict, optimizer_dict=model.optimizer_dict, iteration=1)
    finally:
        checkpointer_module.dcp.save = original_dcp_save

    local_result = {
        "shard_devices": sorted({local_tensor(parameter).device.type for parameter in model.net.parameters()}),
        "exports": sorted(exports),
        **_dcp_exports_alias_live_state(exports, model.net, model.net_optimizer),
    }
    per_rank = [None] * world_size
    dist.all_gather_object(per_rank, local_result)
    return {"per_rank": per_rank}


@RunIf(min_gpus=2)
@pytest.mark.parametrize("cpu_offload", [False, True], ids=["device_shards", "cpu_offload"])
def test_fsdp_dcp_save_exports_alias_live_state(tmp_path, cpu_offload):
    """Sharded exports alias each rank's live local shards and optimizer moments, with or without CPU offload."""
    clear_gpu_memory()
    result = run_distributed_test(
        test_fn=_fsdp_dcp_save_exports_alias_live_state_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
        checkpoint_dir=str(tmp_path),
        cpu_offload=cpu_offload,
    )
    clear_gpu_memory()
    assert result is not None
    expected = {
        "shard_devices": ["cpu" if cpu_offload else "cuda"],
        "exports": ["net_model", "net_optim"],
        "model": True,
        "optimizer": True,
    }
    assert result["per_rank"] == [expected, expected]


def _dcp_rank_local_optimizer_state_impl(rank: int, world_size: int, checkpoint_dir: str) -> dict:
    """Only one rank steps a routed expert, but every rank must enter optimizer checkpointing."""
    import fastgen.utils.checkpointer as checkpointer_module

    class _ToyMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)])

        def forward(self, tokens: torch.Tensor, expert: int) -> torch.Tensor:
            return self.experts[expert](tokens)

    model = _ToyMoE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    if rank == 0:
        model(torch.ones(1, 2), expert=0).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    state_parameter_ids = {id(parameter) for parameter in optimizer.state}
    raw_state_names = sorted(
        name for name, parameter in model.named_parameters() if id(parameter) in state_parameter_ids
    )

    save_paths = []
    original_dcp_save = checkpointer_module.dcp.save
    checkpointer_module.dcp.save = lambda _state_dict, storage_writer: save_paths.append(storage_writer)
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=True)
    checkpointer.get_storage_writer = lambda checkpoint_path: checkpoint_path
    try:
        checkpoint_path = checkpointer.save(
            model_dict=torch.nn.ModuleDict({"net": model}),
            optimizer_dict={"net": optimizer},
            path=os.path.join(checkpoint_dir, "toy_moe.pth"),
            iteration=3,
        )
    finally:
        checkpointer_module.dcp.save = original_dcp_save

    local_result = {
        "rank": rank,
        "raw_state_names": raw_state_names,
        "save_paths": save_paths,
        "checkpoint_path": checkpoint_path,
        "marker_exists": os.path.isfile(checkpoint_path),
    }
    per_rank = [None] * world_size
    dist.all_gather_object(per_rank, local_result)
    return {"per_rank": per_rank}


@RunIf(min_gpus=2)
def test_dcp_rank_local_optimizer_state_uses_global_predicate(tmp_path):
    """A stepped expert on one rank keeps all ranks on the collective optimizer-save branch."""
    result = run_distributed_test(
        test_fn=_dcp_rank_local_optimizer_state_impl,
        world_size=2,
        timeout=60,
        nccl_timeout=30,
        checkpoint_dir=str(tmp_path),
    )
    assert result is not None
    assert [entry["raw_state_names"] for entry in result["per_rank"]] == [["experts.0.weight"], []]
    for entry in result["per_rank"]:
        assert entry["save_paths"] == [
            f"{entry['checkpoint_path'].removesuffix('.pth')}.net_model",
            f"{entry['checkpoint_path'].removesuffix('.pth')}.net_optim",
        ]
        assert entry["marker_exists"]


def _fsdp_scalar_publication_failure_propagates_impl(rank: int, world_size: int, checkpoint_dir: str) -> dict:
    """A rank-0 scalar-state export failure reaches every rank before writing."""
    del world_size

    class FailingCallbacks:
        @staticmethod
        def state_dict():
            raise ValueError("injected scalar-state failure")

    marker_path = os.path.join(checkpoint_dir, "latest_ar.pth")
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=True)
    try:
        checkpointer.save(
            model_dict=torch.nn.ModuleDict(),
            callbacks=FailingCallbacks(),
            path=marker_path,
            iteration=9,
        )
    except RuntimeError as error:
        message = str(error)
        rejected = (
            "DCP save setup" in message
            and "rank 0 failed" in message
            and "ValueError: injected scalar-state failure" in message
        )
        cause_is_local = isinstance(error.__cause__, ValueError) == (rank == 0)
    else:
        rejected = cause_is_local = False

    checks = torch.tensor(
        [rejected, cause_is_local, not os.path.exists(marker_path)],
        dtype=torch.uint8,
    )
    dist.all_reduce(checks, op=dist.ReduceOp.MIN)
    return {
        "rejected_on_all_ranks": bool(checks[0].item()),
        "cause_only_on_rank0": bool(checks[1].item()),
        "marker_absent_on_all_ranks": bool(checks[2].item()),
    }


@RunIf(min_gpus=2)
def test_fsdp_scalar_publication_failure_propagates_across_ranks(tmp_path):
    result = run_distributed_test(
        test_fn=_fsdp_scalar_publication_failure_propagates_impl,
        world_size=2,
        timeout=60,
        nccl_timeout=30,
        checkpoint_dir=str(tmp_path),
    )
    assert result is not None
    assert all(result.values()), result


# =============================================================================
# Standard Checkpointer Tests
# =============================================================================


def test_checkpointer_refuses_inference_targets():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = Checkpointer(_get_checkpointer_config(tmpdir))
        real = torch.nn.Linear(2, 2)
        path = checkpointer.save(model_dict=torch.nn.ModuleDict({"net": real}), iteration=1)

        # Loads happen before prepare_module too, so they refuse inference contexts and tensors themselves.
        with torch.inference_mode(), pytest.raises(RuntimeError, match="monolithic load of Linear must not run"):
            checkpointer.load(model_dict=torch.nn.ModuleDict({"net": real}), path=path)
        with torch.inference_mode():
            built_in_inference = torch.nn.Linear(2, 2)
        with pytest.raises(RuntimeError, match=r"2 inference tensor\(s\)"):
            checkpointer.load(model_dict=torch.nn.ModuleDict({"net": built_in_inference}), path=path)


def test_standard_checkpoint_save_load():
    """Test basic checkpoint save and load with Checkpointer class."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model and checkpointer
        model = _create_sft_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Save original state
        original_state = copy.deepcopy(model.net.state_dict())
        iteration = 100

        # Save checkpoint
        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            callbacks=None,
            iteration=iteration,
        )

        # Verify checkpoint file exists
        assert os.path.exists(path), f"Checkpoint file not found at {path}"

        # Modify weights
        _modify_model_weights(model, scale=2.0)
        modified_state = model.net.state_dict()
        assert not _compare_state_dicts(original_state, modified_state), "States should differ after modification"

        # Load checkpoint
        loaded_iteration = checkpointer.load(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            callbacks=None,
            path=path,
        )

        # Verify loaded state matches original
        loaded_state = model.net.state_dict()
        assert loaded_iteration == iteration, f"Iteration mismatch: {loaded_iteration} != {iteration}"
        assert _compare_state_dicts(original_state, loaded_state), "Loaded state doesn't match original"


def test_checkpoint_multiple_networks():
    """Test checkpoint save/load with multiple networks (DMD2Model)."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create DMD2 model (has net and fake_score)
        model = _create_dmd2_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Save original states
        original_net_state = copy.deepcopy(model.net.state_dict())
        original_fake_score_state = copy.deepcopy(model.fake_score.state_dict())
        iteration = 200

        # Save checkpoint
        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            iteration=iteration,
        )

        # Modify both networks
        _modify_model_weights(model, scale=2.0)
        with torch.no_grad():
            for param in model.fake_score.parameters():
                param.mul_(3.0)

        # Load checkpoint
        loaded_iteration = checkpointer.load(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            path=path,
        )

        # Verify both networks are restored
        assert loaded_iteration == iteration
        assert _compare_state_dicts(original_net_state, model.net.state_dict()), "Net state doesn't match"
        assert _compare_state_dicts(
            original_fake_score_state, model.fake_score.state_dict()
        ), "Fake score state doesn't match"


def test_optimizer_state_save_load():
    """Test that optimizer states are correctly saved and loaded."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_sft_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Run a training step to populate optimizer state
        batch_size = 2
        data = {
            "real": torch.randn(batch_size, 3, 8, 8),
            "condition": torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10).float(),
            "neg_condition": torch.zeros(batch_size, 10),
        }

        model.optimizers_zero_grad(0)
        loss_map, _ = model.single_train_step(data, 0)
        loss_map["total_loss"].backward()
        model.optimizers_schedulers_step(0)

        # Save optimizer state
        original_optimizer_state = copy.deepcopy(model.net_optimizer.state_dict())
        iteration = 50

        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            iteration=iteration,
        )

        # Create a fresh model and load the checkpoint
        fresh_model = _create_sft_model(use_ema=False, device="cpu")

        checkpointer.load(
            model_dict=fresh_model.model_dict,
            optimizer_dict=fresh_model.optimizer_dict,
            scheduler_dict=fresh_model.scheduler_dict,
            grad_scaler=fresh_model.grad_scaler,
            path=path,
        )

        # Verify optimizer state
        loaded_optimizer_state = fresh_model.net_optimizer.state_dict()
        assert (
            original_optimizer_state["param_groups"] == loaded_optimizer_state["param_groups"]
        ), "Optimizer param_groups mismatch"
        assert original_optimizer_state["state"], "The update must create optimizer moments to exercise restoration"
        torch.testing.assert_close(original_optimizer_state["state"], loaded_optimizer_state["state"], rtol=0, atol=0)


def test_load_without_saved_optimizer_state_keeps_fresh_optimizer(tmp_path, monkeypatch):
    """A model-only checkpoint normalizes its explicit optimizer None and warns."""
    net = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(net.parameters())
    path = tmp_path / "model_only.pth"
    torch.save({"model": {"net": net.state_dict()}, "optimizer": None, "iteration": 7}, path)
    warnings = []
    monkeypatch.setattr("fastgen.utils.checkpointer.logger.warning", warnings.append)

    iteration = Checkpointer(_get_checkpointer_config(str(tmp_path))).load(
        model_dict=torch.nn.ModuleDict({"net": net}),
        optimizer_dict={"net": optimizer},
        path=str(path),
    )

    assert iteration == 7
    assert not optimizer.state
    assert warnings == ["Optimizer net not found in checkpoint."]


# =============================================================================
# EMA Checkpointing Tests
# =============================================================================


def test_ema_checkpoint_save_load():
    """Test that EMA states are correctly saved and loaded."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model with EMA enabled
        model = _create_sft_model(use_ema=True, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Verify EMA exists
        assert hasattr(model, "ema"), "Model should have EMA network"
        assert model.ema is not None, "EMA network should not be None"

        # Save original EMA state
        original_ema_state = copy.deepcopy(model.ema.state_dict())
        iteration = 100

        # Save checkpoint
        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            iteration=iteration,
        )

        # Modify EMA weights
        with torch.no_grad():
            for param in model.ema.parameters():
                param.mul_(5.0)

        modified_ema_state = model.ema.state_dict()
        assert not _compare_state_dicts(original_ema_state, modified_ema_state), "EMA states should differ"

        # Load checkpoint
        checkpointer.load(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            path=path,
        )

        # Verify EMA state is restored
        loaded_ema_state = model.ema.state_dict()
        assert _compare_state_dicts(original_ema_state, loaded_ema_state), "EMA state should be restored"


def test_ema_independent_from_net():
    """Test that EMA and net states are independent in checkpoints."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_sft_model(use_ema=True, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Make EMA different from net
        with torch.no_grad():
            for ema_param, net_param in zip(model.ema.parameters(), model.net.parameters()):
                ema_param.copy_(net_param * 0.5)  # EMA is half of net

        original_net_state = copy.deepcopy(model.net.state_dict())
        original_ema_state = copy.deepcopy(model.ema.state_dict())

        # Verify they're different
        assert not _compare_state_dicts(original_net_state, original_ema_state), "Net and EMA should be different"

        # Save checkpoint
        path = checkpointer.save(
            model_dict=model.model_dict,
            iteration=100,
        )

        # Modify both
        _modify_model_weights(model, scale=3.0)
        with torch.no_grad():
            for param in model.ema.parameters():
                param.mul_(7.0)

        # Load checkpoint
        checkpointer.load(
            model_dict=model.model_dict,
            path=path,
        )

        # Verify both are restored to their original (different) states
        assert _compare_state_dicts(original_net_state, model.net.state_dict()), "Net state mismatch"
        assert _compare_state_dicts(original_ema_state, model.ema.state_dict()), "EMA state mismatch"
        assert not _compare_state_dicts(
            model.net.state_dict(), model.ema.state_dict()
        ), "Net and EMA should still be different"


# =============================================================================
# Teacher Exclusion Tests
# =============================================================================


def test_teacher_excluded_from_checkpoint():
    """Test that teacher network is NOT included in the saved checkpoint."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create DMD2 model which has a teacher network
        model = _create_dmd2_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Verify teacher exists
        assert hasattr(model, "teacher"), "DMD2 model should have teacher"
        assert model.teacher is not None, "Teacher should not be None"

        # Verify teacher is NOT in model_dict (which is what gets checkpointed)
        model_dict = model.model_dict
        assert "teacher" not in model_dict, "Teacher should NOT be in model_dict for checkpointing"

        # Verify other networks ARE in model_dict
        assert "net" in model_dict, "Net should be in model_dict"
        assert "fake_score" in model_dict, "Fake score should be in model_dict"

        # Save checkpoint
        iteration = 100
        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            iteration=iteration,
        )

        # Load the checkpoint file and verify teacher is not in it
        checkpoint = torch.load(path, weights_only=False)
        assert "teacher" not in checkpoint["model"], "Teacher should NOT be saved in checkpoint"
        assert "net" in checkpoint["model"], "Net should be in checkpoint"
        assert "fake_score" in checkpoint["model"], "Fake score should be in checkpoint"


def test_teacher_state_unchanged_after_load():
    """Test that loading a checkpoint doesn't affect teacher state."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_dmd2_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Save original teacher state
        original_teacher_state = copy.deepcopy(model.teacher.state_dict())

        # Save checkpoint (teacher not included)
        path = checkpointer.save(
            model_dict=model.model_dict,
            iteration=100,
        )

        # Modify teacher weights manually
        with torch.no_grad():
            for param in model.teacher.parameters():
                param.mul_(10.0)

        modified_teacher_state = copy.deepcopy(model.teacher.state_dict())
        assert not _compare_state_dicts(original_teacher_state, modified_teacher_state), "Teacher should be modified"

        # Load checkpoint - teacher should remain modified (not overwritten)
        checkpointer.load(
            model_dict=model.model_dict,
            path=path,
        )

        # Verify teacher is still in modified state (not overwritten by load)
        current_teacher_state = model.teacher.state_dict()
        assert _compare_state_dicts(
            modified_teacher_state, current_teacher_state
        ), "Teacher should remain modified after checkpoint load"


def test_extra_fsdp_module_excluded_from_checkpoint():
    """An FSDP-only precision component must be neither saved nor restored."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_dmd2_model(use_ema=False, device="cpu")
        model.auxiliary = _PrecisionLinear(4, 4, bias=False).requires_grad_(False)
        model.config.extra_fsdp_modules = ["teacher", "auxiliary"]
        model.register_modules(auxiliary=model.auxiliary)
        model.prepare_module(model.auxiliary, name="auxiliary")

        assert "auxiliary" in model.module_dict
        assert "auxiliary" not in model.model_dict
        original_state = copy.deepcopy(model.auxiliary.state_dict())

        checkpointer = Checkpointer(_get_checkpointer_config(tmpdir))
        path = checkpointer.save(model_dict=model.model_dict, iteration=100)

        checkpoint = torch.load(path, weights_only=False)
        assert "auxiliary" not in checkpoint["model"]

        with torch.no_grad():
            model.auxiliary.weight.mul_(10.0)
        modified_state = copy.deepcopy(model.auxiliary.state_dict())
        assert not _compare_state_dicts(original_state, modified_state)

        checkpointer.load(model_dict=model.model_dict, path=path)

        assert _compare_state_dicts(modified_state, model.auxiliary.state_dict())


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_load_without_path_discovers_latest_checkpoint():
    """``load(path=None)`` discovers the latest completed checkpoint."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_sft_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Save multiple checkpoints
        checkpointer.save(model_dict=model.model_dict, iteration=100)
        checkpointer.save(model_dict=model.model_dict, iteration=200)
        checkpointer.save(model_dict=model.model_dict, iteration=300)

        # Create fresh model
        fresh_model = _create_sft_model(use_ema=False, device="cpu")

        # Load without specifying path (should find latest)
        loaded_iteration = checkpointer.load(
            model_dict=fresh_model.model_dict,
            path=None,  # None discovers the latest completed checkpoint
        )

        assert loaded_iteration == 300, f"Should load latest checkpoint, got iteration {loaded_iteration}"


def test_checkpoint_missing_model_key_logging(monkeypatch):
    """Test that loading warns when the target model has keys absent from the checkpoint."""
    gc.collect()
    import fastgen.utils.logging_utils as fg_logger

    with tempfile.TemporaryDirectory() as tmpdir:
        sft_model = _create_sft_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        path = checkpointer.save(
            model_dict=sft_model.model_dict,
            optimizer_dict=sft_model.optimizer_dict,
            iteration=100,
        )

        dmd2_model = _create_dmd2_model(use_ema=False, device="cpu")

        log_messages = []
        _orig_warning, _orig_info, _orig_success = fg_logger.warning, fg_logger.info, fg_logger.success
        monkeypatch.setattr(
            fg_logger, "warning", lambda msg: (log_messages.append(("warning", msg)), _orig_warning(msg))
        )
        monkeypatch.setattr(fg_logger, "info", lambda msg: (log_messages.append(("info", msg)), _orig_info(msg)))
        monkeypatch.setattr(
            fg_logger, "success", lambda msg: (log_messages.append(("success", msg)), _orig_success(msg))
        )

        loaded_iteration = checkpointer.load(
            model_dict=dmd2_model.model_dict,
            optimizer_dict=dmd2_model.optimizer_dict,
            path=path,
            strict=False,
        )

        assert loaded_iteration == 100, "Should still return correct iteration"

        warnings = [m for level, m in log_messages if level == "warning"]
        model_missing = [w for w in warnings if "Model" in w and "not found in checkpoint" in w]
        optim_missing = [w for w in warnings if "Optimizer" in w and "not found in checkpoint" in w]

        assert any(
            "fake_score" in w for w in model_missing
        ), f"Expected 'fake_score' model-not-found warning, got: {warnings}"
        assert any(
            "fake_score" in w for w in optim_missing
        ), f"Expected 'fake_score' optimizer-not-found warning, got: {warnings}"

        net_info = [m for level, m in log_messages if level == "success" and "Model net" in m and "loading info" in m]
        assert len(net_info) > 0, "Matching key 'net' should still log its loading info"


def test_checkpoint_param_mismatch_logging(monkeypatch):
    """Test that missing and unexpected *parameter* keys are reported in the loading info log."""
    gc.collect()
    import fastgen.utils.logging_utils as fg_logger

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_sft_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        path = checkpointer.save(model_dict=model.model_dict, iteration=100)

        # Tamper with the checkpoint: remove a real key, add a fake key
        ckpt = torch.load(path, weights_only=False)
        net_sd = ckpt["model"]["net"]
        removed_key = next(iter(net_sd))
        del net_sd[removed_key]
        net_sd["fake_layer.weight"] = torch.randn(10, 10)
        torch.save(ckpt, path)

        log_messages = []
        _orig_warning = fg_logger.warning
        # Missing or unexpected keys downgrade the loading-info line from success to warning.
        monkeypatch.setattr(
            fg_logger, "warning", lambda msg: (log_messages.append(("warning", msg)), _orig_warning(msg))
        )

        fresh_model = _create_sft_model(use_ema=False, device="cpu")
        loaded_iteration = checkpointer.load(model_dict=fresh_model.model_dict, path=path, strict=False)

        assert loaded_iteration == 100

        load_info = [m for level, m in log_messages if level == "warning" and "loading info" in m]
        assert len(load_info) > 0, "Should log model loading info"
        info_str = load_info[0]
        assert removed_key in info_str, f"Should report missing key '{removed_key}' in: {info_str}"
        assert "fake_layer.weight" in info_str, f"Should report unexpected key in: {info_str}"

        strict_model = _create_sft_model(use_ema=False, device="cpu")
        with pytest.raises(RuntimeError, match="Error\\(s\\) in loading state_dict"):
            checkpointer.load(model_dict=strict_model.model_dict, path=path)


class _MappedCheckpointModule(torch.nn.Module):
    """Tiny root with one checkpointed and one deliberately missing parameter."""

    def __init__(self, selected: float, retained: float):
        super().__init__()
        self.selected = torch.nn.Parameter(torch.tensor([selected]))
        self.retained = torch.nn.Parameter(torch.tensor([retained]))


def test_trainer_pretrained_key_map_loads_partial_net_and_ema_roots_from_pth(tmp_path):
    """The key map is destination-to-source for a monolithic DDP checkpoint."""
    checkpoint_path = tmp_path / "selected_roots.pth"
    torch.save(
        {
            "model": {
                "net": {"selected": torch.tensor([11.0])},
                "ema": {"selected": torch.tensor([22.0])},
            },
            "iteration": 37,
        },
        checkpoint_path,
    )

    model = SimpleNamespace(
        net=_MappedCheckpointModule(selected=1.0, retained=101.0),
        ema_9999=_MappedCheckpointModule(selected=2.0, retained=102.0),
        ema_99995=_MappedCheckpointModule(selected=3.0, retained=103.0),
        ema_9996=_MappedCheckpointModule(selected=4.0, retained=104.0),
    )
    untouched_state = {name: tensor.clone() for name, tensor in model.ema_9996.state_dict().items()}

    checkpointer_config = BaseCheckpointerConfig()
    checkpointer_config.pretrained_ckpt_path = str(checkpoint_path)
    checkpointer_config.pretrained_ckpt_key_map = {
        "net": "ema",
        "ema_9999": "net",
        "ema_99995": "ema",
    }
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(checkpointer=checkpointer_config))

    trainer.checkpointer = Checkpointer(checkpointer_config)
    trainer.load_pretrained_ckpt(model)

    assert torch.equal(model.net.selected, torch.tensor([22.0]))  # net <- checkpoint EMA
    assert torch.equal(model.ema_9999.selected, torch.tensor([11.0]))  # EMA <- checkpoint net
    assert torch.equal(model.ema_99995.selected, torch.tensor([22.0]))  # second EMA <- checkpoint EMA
    assert torch.equal(model.net.retained, torch.tensor([101.0]))
    assert torch.equal(model.ema_9999.retained, torch.tensor([102.0]))
    assert torch.equal(model.ema_99995.retained, torch.tensor([103.0]))
    assert all(torch.equal(model.ema_9996.state_dict()[name], value) for name, value in untouched_state.items())
    assert model.resume_iter == 37


@pytest.mark.parametrize(
    "key_map, should_warn",
    [({"net": "net"}, True), ({"net": "net", "ema_9999": "ema"}, False), ({"fake_score": "fake_score"}, False)],
    ids=["net-only", "net-and-ema", "without-net"],
)
def test_trainer_pretrained_key_map_warns_only_when_net_leaves_ema_stale(key_map, should_warn, monkeypatch):
    """EMA clones are copied from net before this load, so they only go stale when net itself is overwritten."""
    import fastgen.utils.logging_utils as fg_logger

    warnings = []
    monkeypatch.setattr(fg_logger, "warning", warnings.append)
    checkpointer_config = BaseCheckpointerConfig()
    checkpointer_config.pretrained_ckpt_path = "unused.pth"
    checkpointer_config.pretrained_ckpt_key_map = key_map
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(checkpointer=checkpointer_config))
    trainer.checkpointer = SimpleNamespace(load=lambda *args, **kwargs: 0)
    model = SimpleNamespace(
        use_ema=["ema_9999"], net=torch.nn.Identity(), ema_9999=torch.nn.Identity(), fake_score=torch.nn.Identity()
    )

    trainer.load_pretrained_ckpt(model)

    assert any("EMA module(s) ['ema_9999']" in message for message in warnings) == should_warn


def test_trainer_propagates_missing_pretrained_checkpoint(tmp_path):
    checkpointer_config = BaseCheckpointerConfig()
    checkpointer_config.pretrained_ckpt_path = str(tmp_path / "missing.pth")
    checkpointer_config.pretrained_ckpt_key_map = {"net": "net"}
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(checkpointer=checkpointer_config))
    trainer.checkpointer = Checkpointer(checkpointer_config)

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        trainer.load_pretrained_ckpt(SimpleNamespace(net=torch.nn.Identity()))


# =============================================================================
# FSDP Checkpointing Tests (require multiple GPUs)
# =============================================================================


_DCP_LOAD_INFO_LAYOUTS = {
    "exact": (True, True, False, False),
    "missing_parameter": (True, True, True, False),
    "missing_buffer": (True, True, False, True),
    "missing_parameter_and_buffer": (True, True, True, True),
    "unexpected_parameter": (False, True, False, False),
    "unexpected_buffer": (True, False, False, False),
    "unexpected_parameter_and_buffer": (False, False, False, False),
    "missing_and_unexpected_parameter_and_buffer": (False, False, True, True),
}
_DCP_SOURCE_VALUES = {
    "shared_parameter": 11.0,
    "checkpoint_only_parameter": 12.0,
    "shared_buffer": 21.0,
    "checkpoint_only_buffer": 22.0,
}
_DCP_TARGET_VALUES = {
    "shared_parameter": -11.0,
    "checkpoint_only_parameter": -12.0,
    "target_only_parameter": 13.0,
    "shared_buffer": -21.0,
    "checkpoint_only_buffer": -22.0,
    "target_only_buffer": 23.0,
}


class _DCPModelLoadInfoModule(torch.nn.Module):
    """Small model with independently optional checkpoint and target-only state."""

    def __init__(
        self,
        source: bool,
        include_checkpoint_parameter: bool = True,
        include_checkpoint_buffer: bool = True,
        include_target_parameter: bool = False,
        include_target_buffer: bool = False,
    ):
        super().__init__()
        values = _DCP_SOURCE_VALUES if source else _DCP_TARGET_VALUES
        self.shared_parameter = torch.nn.Parameter(torch.full((12,), values["shared_parameter"]))
        if include_checkpoint_parameter:
            self.checkpoint_only_parameter = torch.nn.Parameter(torch.full((12,), values["checkpoint_only_parameter"]))
        if include_target_parameter:
            self.target_only_parameter = torch.nn.Parameter(torch.full((12,), values["target_only_parameter"]))

        self.register_buffer("shared_buffer", torch.full((12,), values["shared_buffer"]))
        if include_checkpoint_buffer:
            self.register_buffer(
                "checkpoint_only_buffer",
                torch.full((12,), values["checkpoint_only_buffer"]),
            )
        if include_target_buffer:
            self.register_buffer("target_only_buffer", torch.full((12,), values["target_only_buffer"]))


def _make_dcp_model_load_info_target(layout: str) -> _DCPModelLoadInfoModule:
    (
        include_checkpoint_parameter,
        include_checkpoint_buffer,
        include_target_parameter,
        include_target_buffer,
    ) = _DCP_LOAD_INFO_LAYOUTS[layout]
    return _DCPModelLoadInfoModule(
        source=False,
        include_checkpoint_parameter=include_checkpoint_parameter,
        include_checkpoint_buffer=include_checkpoint_buffer,
        include_target_parameter=include_target_parameter,
        include_target_buffer=include_target_buffer,
    )


def _prepare_dcp_model_load_info_module(
    module: _DCPModelLoadInfoModule,
    sharded: bool,
    world_size: int,
) -> _DCPModelLoadInfoModule:
    module.cuda()
    if sharded:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cuda", (world_size,))
        fully_shard(module, mesh=mesh)
    return module


def _expected_dcp_model_load_info(layout: str) -> tuple[list[str], list[str]]:
    (
        include_checkpoint_parameter,
        include_checkpoint_buffer,
        include_target_parameter,
        include_target_buffer,
    ) = _DCP_LOAD_INFO_LAYOUTS[layout]
    missing_keys = []
    unexpected_keys = []
    if include_target_parameter:
        missing_keys.append("target_only_parameter")
    if include_target_buffer:
        missing_keys.append("target_only_buffer")
    if not include_checkpoint_parameter:
        unexpected_keys.append("checkpoint_only_parameter")
    if not include_checkpoint_buffer:
        unexpected_keys.append("checkpoint_only_buffer")
    return sorted(missing_keys), sorted(unexpected_keys)


def _expected_dcp_model_load_values(layout: str) -> dict[str, float]:
    (
        include_checkpoint_parameter,
        include_checkpoint_buffer,
        include_target_parameter,
        include_target_buffer,
    ) = _DCP_LOAD_INFO_LAYOUTS[layout]
    expected = {
        "shared_parameter": _DCP_SOURCE_VALUES["shared_parameter"],
        "shared_buffer": _DCP_SOURCE_VALUES["shared_buffer"],
    }
    if include_checkpoint_parameter:
        expected["checkpoint_only_parameter"] = _DCP_SOURCE_VALUES["checkpoint_only_parameter"]
    if include_checkpoint_buffer:
        expected["checkpoint_only_buffer"] = _DCP_SOURCE_VALUES["checkpoint_only_buffer"]
    if include_target_parameter:
        expected["target_only_parameter"] = _DCP_TARGET_VALUES["target_only_parameter"]
    if include_target_buffer:
        expected["target_only_buffer"] = _DCP_TARGET_VALUES["target_only_buffer"]
    return expected


def _save_dcp_model_load_info_checkpoint_impl(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
    sharded: bool,
) -> dict:
    """Save the same source state from two ranks, with or without FSDP2."""
    module = _prepare_dcp_model_load_info_module(
        _DCPModelLoadInfoModule(source=True),
        sharded=sharded,
        world_size=world_size,
    )
    path = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=True).save(
        torch.nn.ModuleDict({"net": module}),
        iteration=17,
    )
    dist.barrier()
    del module
    torch.cuda.empty_cache()
    return {"path": path, "rank": rank}


def _run_dcp_model_load_info_case(
    rank: int,
    world_size: int,
    path: str,
    source_sharded: bool,
    target_sharded: bool,
    layout: str,
) -> dict:
    """Load one topology/layout case and collect diagnostics from every rank."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.tensor import DTensor

    module = _prepare_dcp_model_load_info_module(
        _make_dcp_model_load_info_target(layout),
        sharded=target_sharded,
        world_size=world_size,
    )
    import fastgen.utils.checkpointer as checkpointer_module

    load_messages = []
    original_warning, original_success = checkpointer_module.logger.warning, checkpointer_module.logger.success
    checkpointer_module.logger.warning = lambda message: load_messages.append(f"warning: {message}")
    checkpointer_module.logger.success = lambda message: load_messages.append(f"success: {message}")
    try:
        iteration = Checkpointer(_get_checkpointer_config(os.path.dirname(path))).load(
            torch.nn.ModuleDict({"net": module}),
            path=path,
            strict=False,
        )
    finally:
        checkpointer_module.logger.warning, checkpointer_module.logger.success = original_warning, original_success

    model_load_info = [message for message in load_messages if "Model net, loading info:" in message]
    rank_messages = [None] * world_size
    dist.all_gather_object(rank_messages, model_load_info)

    rank_iterations = [None] * world_size
    dist.all_gather_object(rank_iterations, iteration)

    parameters_are_sharded = all(isinstance(parameter, DTensor) for parameter in module.parameters())
    rank_sharding = [None] * world_size
    dist.all_gather_object(rank_sharding, parameters_are_sharded)

    state = get_model_state_dict(module, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    state_matches = True
    if rank == 0:
        expected_values = _expected_dcp_model_load_values(layout)
        state_matches = set(state) == set(expected_values) and all(
            torch.equal(state[name], torch.full_like(state[name], value)) for name, value in expected_values.items()
        )

    missing_keys, unexpected_keys = _expected_dcp_model_load_info(layout)
    level = "warning" if missing_keys or unexpected_keys else "success"
    expected_message = (
        f"{level}: Model net, loading info: missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
    )
    result = {
        "case": (
            f"save={'sharded' if source_sharded else 'unsharded'}-ws2/"
            f"load={'sharded' if target_sharded else 'unsharded'}-ws{world_size}/{layout}"
        ),
        "rank_messages": rank_messages,
        "expected_message": expected_message,
        "rank_iterations": rank_iterations,
        "rank_sharding": rank_sharding,
        "expected_sharding": target_sharded,
        "state_matches": state_matches,
    }

    del state, module
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def _load_dcp_model_load_info_matrix_impl(
    rank: int,
    world_size: int,
    checkpoint_paths: dict[str, str],
    target_sharding: list[bool],
    layouts: list[str],
) -> dict:
    """Run several load cases in one process group to keep the GPU test affordable."""
    results = []
    for source_name, path in checkpoint_paths.items():
        source_sharded = source_name == "sharded"
        for target_sharded in target_sharding:
            for layout in layouts:
                results.append(
                    _run_dcp_model_load_info_case(
                        rank=rank,
                        world_size=world_size,
                        path=path,
                        source_sharded=source_sharded,
                        target_sharded=target_sharded,
                        layout=layout,
                    )
                )
    return {"results": results}


@pytest.fixture(scope="module")
def dcp_model_load_info_checkpoints(tmp_path_factory):
    """Create real DCP checkpoints saved on two ranks for topology-change tests."""
    checkpoint_root = tmp_path_factory.mktemp("dcp_model_load_info")
    checkpoint_paths = {}
    clear_gpu_memory()
    for sharded in (False, True):
        checkpoint_dir = str(checkpoint_root / ("sharded" if sharded else "unsharded"))
        result = run_distributed_test(
            test_fn=_save_dcp_model_load_info_checkpoint_impl,
            world_size=2,
            timeout=180,
            setup_fn=set_env_vars,
            checkpoint_dir=checkpoint_dir,
            sharded=sharded,
        )
        assert result is not None
        checkpoint_paths["sharded" if sharded else "unsharded"] = result["path"]
    clear_gpu_memory()
    return checkpoint_paths


def _assert_dcp_model_load_info_results(result: dict) -> None:
    assert result is not None
    assert result["results"]
    for case in result["results"]:
        assert all(messages == [case["expected_message"]] for messages in case["rank_messages"]), case["case"]
        assert all(iteration == 17 for iteration in case["rank_iterations"]), case["case"]
        assert all(sharded is case["expected_sharding"] for sharded in case["rank_sharding"]), case["case"]
        assert case["state_matches"], case["case"]


@RunIf(min_gpus=3)
def test_dcp_model_load_info_across_world_sizes_and_sharding(dcp_model_load_info_checkpoints):
    """A 2-rank save logs the same four incompatibility kinds on 1-, 2-, and 3-rank loads."""
    mixed_layout = ["missing_and_unexpected_parameter_and_buffer"]
    for load_world_size in (1, 2, 3):
        result = run_distributed_test(
            test_fn=_load_dcp_model_load_info_matrix_impl,
            world_size=load_world_size,
            timeout=240,
            setup_fn=set_env_vars,
            checkpoint_paths=dcp_model_load_info_checkpoints,
            target_sharding=[False, True],
            layouts=mixed_layout,
        )
        _assert_dcp_model_load_info_results(result)
    clear_gpu_memory()


@RunIf(min_gpus=3)
def test_dcp_model_load_info_for_parameters_and_buffers(dcp_model_load_info_checkpoints):
    """Real FSDP2 loads distinguish missing/unexpected parameters, buffers, both, and exact state."""
    result = run_distributed_test(
        test_fn=_load_dcp_model_load_info_matrix_impl,
        world_size=3,
        timeout=300,
        setup_fn=set_env_vars,
        checkpoint_paths={"sharded": dcp_model_load_info_checkpoints["sharded"]},
        target_sharding=[True],
        layouts=list(_DCP_LOAD_INFO_LAYOUTS),
    )
    _assert_dcp_model_load_info_results(result)
    clear_gpu_memory()


def _test_fsdp_pretrained_key_map_impl(rank: int, world_size: int, checkpoint_dir: str) -> dict:
    """Map partial source roots into live FSDP2 roots without replacing their DTensors."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, fully_shard
    from torch.distributed.tensor import DTensor

    class _SelectedRoot(torch.nn.Module):
        def __init__(self, selected: float):
            super().__init__()
            self.selected = torch.nn.Parameter(torch.full((4,), selected))

    class _TargetRoot(_SelectedRoot):
        def __init__(self, selected: float, retained: float):
            super().__init__(selected)
            self.retained = torch.nn.Parameter(torch.full((4,), retained))

    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (world_size,))

    source_roots = torch.nn.ModuleDict(
        {
            "net": _SelectedRoot(11.0).to(device),
            "ema": _SelectedRoot(22.0).to(device),
        }
    )
    for root in source_roots.values():
        fully_shard(root, mesh=mesh)

    checkpointer_config = _get_checkpointer_config(checkpoint_dir)
    checkpoint_path = Checkpointer(checkpointer_config, save_sharded=True).save(
        model_dict=source_roots,
        iteration=37,
    )

    model = SimpleNamespace(
        net=_TargetRoot(selected=1.0, retained=101.0).to(device),
        ema_9999=_TargetRoot(selected=2.0, retained=102.0).to(device),
        ema_99995=_TargetRoot(selected=3.0, retained=103.0).to(device),
    )
    target_roots = (model.net, model.ema_9999, model.ema_99995)
    for root in target_roots:
        fully_shard(root, mesh=mesh)
    retained_before = [root.retained.to_local().clone() for root in target_roots]

    checkpointer_config.pretrained_ckpt_path = checkpoint_path
    checkpointer_config.pretrained_ckpt_key_map = {
        "net": "ema",
        "ema_9999": "net",
        "ema_99995": "ema",
    }
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(checkpointer=checkpointer_config))
    trainer.checkpointer = Checkpointer(checkpointer_config)
    trainer.load_pretrained_ckpt(model)

    def _local_parameter_matches(root: torch.nn.Module, name: str, expected: float) -> bool:
        parameter = getattr(root, name)
        return isinstance(parameter, DTensor) and torch.equal(
            parameter.to_local(), torch.full_like(parameter.to_local(), expected)
        )

    selected_mapped = all(
        (
            _local_parameter_matches(model.net, "selected", 22.0),
            _local_parameter_matches(model.ema_9999, "selected", 11.0),
            _local_parameter_matches(model.ema_99995, "selected", 22.0),
        )
    )
    retained_preserved = all(
        torch.equal(root.retained.to_local(), before) for root, before in zip(target_roots, retained_before)
    )
    roots_still_sharded = all(isinstance(root, FSDPModule) for root in target_roots) and all(
        isinstance(parameter, DTensor) for root in target_roots for parameter in root.parameters()
    )

    assert selected_mapped, f"rank {rank}: selected parameters did not follow the root key map"
    assert retained_preserved, f"rank {rank}: parameters absent from the checkpoint changed"
    assert roots_still_sharded, f"rank {rank}: checkpoint loading replaced or unsharded an FSDP root"
    assert model.resume_iter == 37

    resume_iter = model.resume_iter
    del source_roots, model, target_roots, retained_before
    clear_gpu_memory()
    return {
        "selected_mapped": selected_mapped,
        "retained_preserved": retained_preserved,
        "roots_still_sharded": roots_still_sharded,
        "resume_iter": resume_iter,
    }


@RunIf(min_gpus=2)
def test_fsdp_pretrained_key_map_loads_partial_net_and_ema_roots():
    """A real DCP marker supports partial net/EMA remapping into FSDP2 roots."""
    clear_gpu_memory()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_distributed_test(
            test_fn=_test_fsdp_pretrained_key_map_impl,
            world_size=2,
            timeout=180,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )
    assert result is not None
    assert result["selected_mapped"]
    assert result["retained_preserved"]
    assert result["roots_still_sharded"]
    assert result["resume_iter"] == 37
    clear_gpu_memory()


def _test_fsdp_checkpoint_sft_impl(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
) -> dict:
    """Test FSDP checkpointing with SFTModel (single network)."""
    from unittest.mock import patch

    from fastgen.utils.distributed.fsdp import create_fsdp_context

    # Create model config
    config = SFTModelConfig()
    config.net = _get_small_edm_config()
    config.input_shape = [3, 8, 8]
    config.precision = "float32"
    config.precision_model = "bfloat16"
    config.precision_opt = "float32"
    config.device = "cuda"
    config.pretrained_model_path = None
    config.use_ema = False
    config.meta_init = True

    # Build model
    model = SFTModel(config)

    dist.barrier()

    # Prepare the model through one shared FSDP context.
    fsdp_context = create_fsdp_context(
        reduce_dtype=model.precision_fsdp_reduce,
        min_num_params=1_000_000,
        apply_cpu_offload=False,
    )
    model.on_train_begin(fsdp_context=fsdp_context)

    model.init_optimizers()

    dist.barrier()

    # Create DCP checkpointer
    checkpointer_config = _get_checkpointer_config(checkpoint_dir)
    checkpointer = Checkpointer(checkpointer_config, save_sharded=True)

    # Get original net state for comparison (gather to rank 0)
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    original_state = get_model_state_dict(model.net, options=options)

    dist.barrier()

    # Save checkpoint
    iteration = 150
    path = checkpointer.save(
        model_dict=model.model_dict,
        optimizer_dict=model.optimizer_dict,
        scheduler_dict=model.scheduler_dict,
        grad_scaler=model.grad_scaler,
        iteration=iteration,
    )

    dist.barrier()

    # Modify weights
    with torch.no_grad():
        for param in model.net.parameters():
            if hasattr(param, "_local_tensor"):
                param._local_tensor.mul_(2.0)
            else:
                param.mul_(2.0)

    dist.barrier()

    # Read the marker once on each rank; load() already has its format in memory.
    with patch("fastgen.utils.checkpointer.load_training_checkpoint", wraps=load_training_checkpoint) as read:
        loaded_iteration = checkpointer.load(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            path=path,
        )
    assert read.call_count == 1

    dist.barrier()

    # Verify state is restored
    loaded_state = get_model_state_dict(model.net, options=options)

    result = {
        "iteration_match": loaded_iteration == iteration,
        "loaded_iteration": loaded_iteration,
        "expected_iteration": iteration,
    }

    if rank == 0:
        # Compare states on rank 0
        result["state_match"] = _compare_state_dicts(original_state, loaded_state)
        result["all_passed"] = result["iteration_match"] and result["state_match"]
    else:
        result["state_match"] = True
        result["all_passed"] = True

    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_fsdp_checkpoint_sft():
    """Test FSDP checkpointing with SFTModel."""
    clear_gpu_memory()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_distributed_test(
            test_fn=_test_fsdp_checkpoint_sft_impl,
            world_size=2,
            timeout=300,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )

        assert result is not None, "Test did not return a result"
        assert result.get("iteration_match", False), (
            f"Iteration mismatch: got {result.get('loaded_iteration')}, " f"expected {result.get('expected_iteration')}"
        )
        assert result.get("state_match", False), "Model state not correctly restored"
        assert result.get("all_passed", False), "FSDP checkpoint test failed"

    clear_gpu_memory()


def _test_fsdp_checkpoint_dmd2_impl(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
) -> dict:
    """Test FSDP checkpointing with DMD2Model (multiple networks)."""
    from fastgen.utils.distributed.fsdp import create_fsdp_context

    # Create DMD2 model config
    config = DMD2ModelConfig()
    config.net = _get_small_edm_config()
    config.input_shape = [3, 8, 8]
    config.precision = "float32"
    config.precision_opt = "bfloat16"
    config.device = "cuda"
    config.pretrained_model_path = None
    config.use_ema = False
    config.meta_init = True
    config.gan_loss_weight_gen = 0.0  # Disable GAN for simpler testing
    config.extra_fsdp_modules = ["teacher", "auxiliary"]

    # Build model
    model = DMD2Model(config)
    with model._get_meta_init_context(config.meta_init):
        model.auxiliary = _PrecisionLinear(4, 4, bias=False)
    model.auxiliary.requires_grad_(False)
    model.register_modules(auxiliary=model.auxiliary)

    dist.barrier()

    # Prepare every selected root through one shared FSDP context.
    fsdp_context = create_fsdp_context(
        reduce_dtype=model.precision_fsdp_reduce,
        min_num_params=1_000_000,
        apply_cpu_offload=False,
    )
    model.on_train_begin(fsdp_context=fsdp_context)

    expected_device = torch.device("cuda", torch.cuda.current_device())
    components = model.module_dict
    assert set(config.extra_fsdp_modules) <= set(components), "Not all configured FSDP extras resolved"
    placement_errors = {}
    for name, module in components.items():
        expected_dtype = model.precision_model if name in config.extra_fsdp_modules else model.precision_opt
        pinned = {
            f"{kind}:{tensor_name}": dtype
            for tensor_name, dtype in module._fixed_precision_overrides().items()
            for kind in ("parameter", "buffer")
        }
        errors = _module_placement_errors(module, expected_dtype, expected_device, dtype_overrides=pinned)
        if errors:
            placement_errors[name] = errors
    assert not placement_errors, f"FSDP module placement mismatch: {placement_errors}"

    model.init_optimizers()

    dist.barrier()

    # Create DCP checkpointer
    checkpointer_config = _get_checkpointer_config(checkpoint_dir)
    checkpointer = Checkpointer(checkpointer_config, save_sharded=True)

    # Get original states for comparison
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    original_net_state = get_model_state_dict(model.net, options=options)
    original_fake_score_state = get_model_state_dict(model.fake_score, options=options)
    original_auxiliary_state = get_model_state_dict(model.auxiliary, options=options)

    dist.barrier()

    # Save checkpoint
    iteration = 250
    path = checkpointer.save(
        model_dict=model.model_dict,
        optimizer_dict=model.optimizer_dict,
        scheduler_dict=model.scheduler_dict,
        grad_scaler=model.grad_scaler,
        iteration=iteration,
    )

    dist.barrier()
    assert not os.path.exists(
        f"{path.removesuffix('.pth')}.auxiliary_model"
    ), "FSDP-only auxiliary module was checkpointed"

    # Modify checkpointed networks and the FSDP-only auxiliary module.
    with torch.no_grad():
        for param in model.net.parameters():
            if hasattr(param, "_local_tensor"):
                param._local_tensor.mul_(2.0)
            else:
                param.mul_(2.0)
        for param in model.fake_score.parameters():
            if hasattr(param, "_local_tensor"):
                param._local_tensor.mul_(3.0)
            else:
                param.mul_(3.0)
        for param in model.auxiliary.parameters():
            if hasattr(param, "_local_tensor"):
                param._local_tensor.mul_(4.0)
            else:
                param.mul_(4.0)

    dist.barrier()
    modified_auxiliary_state = get_model_state_dict(model.auxiliary, options=options)

    # Load checkpoint
    loaded_iteration = checkpointer.load(
        model_dict=model.model_dict,
        optimizer_dict=model.optimizer_dict,
        scheduler_dict=model.scheduler_dict,
        grad_scaler=model.grad_scaler,
        path=path,
    )

    dist.barrier()

    # Verify states are restored
    loaded_net_state = get_model_state_dict(model.net, options=options)
    loaded_fake_score_state = get_model_state_dict(model.fake_score, options=options)
    loaded_auxiliary_state = get_model_state_dict(model.auxiliary, options=options)

    result = {
        "iteration_match": loaded_iteration == iteration,
        "loaded_iteration": loaded_iteration,
        "expected_iteration": iteration,
        "placement_match": not placement_errors,
    }

    if rank == 0:
        result["net_state_match"] = _compare_state_dicts(original_net_state, loaded_net_state)
        result["fake_score_state_match"] = _compare_state_dicts(original_fake_score_state, loaded_fake_score_state)
        auxiliary_was_modified = not _compare_state_dicts(original_auxiliary_state, modified_auxiliary_state)
        result["extra_module_excluded"] = auxiliary_was_modified and _compare_state_dicts(
            modified_auxiliary_state, loaded_auxiliary_state
        )
        result["all_passed"] = (
            result["iteration_match"]
            and result["net_state_match"]
            and result["fake_score_state_match"]
            and result["placement_match"]
            and result["extra_module_excluded"]
        )
    else:
        result["net_state_match"] = True
        result["fake_score_state_match"] = True
        result["extra_module_excluded"] = True
        result["all_passed"] = True

    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_fsdp_checkpoint_dmd2():
    """Test FSDP checkpointing with DMD2Model (multiple networks)."""
    clear_gpu_memory()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_distributed_test(
            test_fn=_test_fsdp_checkpoint_dmd2_impl,
            world_size=2,
            timeout=300,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )

        assert result is not None, "Test did not return a result"
        assert result.get("iteration_match", False), (
            f"Iteration mismatch: got {result.get('loaded_iteration')}, " f"expected {result.get('expected_iteration')}"
        )
        assert result.get("net_state_match", False), "Net state not correctly restored"
        assert result.get("fake_score_state_match", False), "Fake score state not correctly restored"
        assert result.get("placement_match", False), "FSDP module precision/device mismatch"
        assert result.get("extra_module_excluded", False), "FSDP-only auxiliary module was restored"
        assert result.get("all_passed", False), "FSDP checkpoint DMD2 test failed"

    clear_gpu_memory()


def _test_fsdp_ema_checkpoint_impl(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
) -> dict:
    """Update a CPU-offloaded live FSDP EMA and round-trip it through DCP."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    from fastgen.callbacks.ema import EMACallback
    from fastgen.utils.distributed.fsdp import create_fsdp_context

    # Create model config with EMA
    config = SFTModelConfig()
    config.net = _get_small_edm_config()
    config.input_shape = [3, 8, 8]
    config.precision = "float32"
    config.precision_model = "bfloat16"
    config.precision_opt = "float32"
    config.device = "cuda"
    config.pretrained_model_path = None
    config.use_ema = True
    config.meta_init = True

    # Build the EMA like every other model root: rank 0 owns real state while
    # meta initialization leaves the other ranks allocation-free until FSDP preparation.
    model = SFTModel(config)
    assert list(model.ema.parameters()), "EMA test network should have parameters"
    expected_meta = rank != 0
    assert all(parameter.is_meta == expected_meta for parameter in model.ema.parameters())
    assert all(buffer.is_meta == expected_meta for buffer in model.ema.buffers())

    dist.barrier()

    # Prepare both live roots through the same ordinary FSDP path.
    fsdp_context = create_fsdp_context(
        reduce_dtype=model.precision_fsdp_reduce,
        min_num_params=1_000_000,
        apply_cpu_offload=True,
    )
    model.on_train_begin(fsdp_context=fsdp_context)
    assert any(isinstance(module, FSDPModule) for module in model.ema.modules())
    assert all(isinstance(parameter, DTensor) for parameter in model.net.parameters())
    assert all(parameter.to_local().device.type == "cpu" for parameter in model.net.parameters())
    assert all(isinstance(parameter, DTensor) for parameter in model.ema.parameters())
    assert all(parameter.to_local().device.type == "cpu" for parameter in model.ema.parameters())
    assert all(parameter.to_local().dtype == torch.float32 for parameter in model.ema.parameters())

    model.init_optimizers()

    dist.barrier()

    result = {"rank": rank}

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    initialized_net_state = get_model_state_dict(model.net, options=options)
    initialized_ema_state = get_model_state_dict(model.ema, options=options)
    if rank == 0:
        assert _compare_state_dicts(
            initialized_net_state, initialized_ema_state
        ), "EMA initialization did not copy the student state"

    callback = EMACallback(type="constant", beta=0.5)
    callback.on_model_init_end(model)
    net_parameters = {
        name.replace("_checkpoint_wrapped_module.", ""): parameter for name, parameter in model.net.named_parameters()
    }
    ema_parameters = {
        name.replace("_checkpoint_wrapped_module.", ""): parameter for name, parameter in model.ema.named_parameters()
    }
    assert net_parameters.keys() == ema_parameters.keys()
    ema_before = {name: parameter.to_local().detach().clone() for name, parameter in ema_parameters.items()}

    # Simulate an optimizer update directly on CPU-offloaded student shards. Rank-distinct deltas
    # ensure that both local shards participate in the subsequent EMA checkpoint.
    with torch.no_grad():
        for parameter in net_parameters.values():
            parameter.to_local().add_(rank + 1)
    net_after = {name: parameter.to_local().detach().clone() for name, parameter in net_parameters.items()}

    callback.on_training_step_end(
        model=model,
        data_batch={},
        output_batch={},
        loss_dict={},
        iteration=1,
    )

    ema_parameters = {
        name.replace("_checkpoint_wrapped_module.", ""): parameter for name, parameter in model.ema.named_parameters()
    }
    ema_update_match = all(
        torch.equal(
            ema_parameters[name].to_local(),
            ema_before[name].lerp(net_after[name], 0.5),
        )
        for name in ema_parameters
    )
    update_match = torch.tensor(int(ema_update_match), dtype=torch.int32, device=torch.device("cuda", rank))
    dist.all_reduce(update_match, op=dist.ReduceOp.MIN)
    ema_update_match = bool(update_match.item())

    original_ema_state = get_model_state_dict(model.ema, options=options)
    if rank == 0:
        assert not _compare_state_dicts(
            initialized_net_state, original_ema_state
        ), "Live EMA update did not diverge from the student"

    # Create DCP checkpointer
    checkpointer_config = _get_checkpointer_config(checkpoint_dir)
    checkpointer = Checkpointer(checkpointer_config, save_sharded=True)

    dist.barrier()

    # Save checkpoint
    iteration = 175
    path = checkpointer.save(
        model_dict=model.model_dict,
        optimizer_dict=model.optimizer_dict,
        scheduler_dict=model.scheduler_dict,
        iteration=iteration,
    )
    assert os.path.exists(f"{path.removesuffix('.pth')}.ema_model/.metadata"), "EMA DCP metadata was not written"

    dist.barrier()

    # Corrupt only the local shards, then restore the saved sharded EMA through DCP.
    with torch.no_grad():
        for param in model.ema.parameters():
            param.to_local().add_(10.0)

    dist.barrier()

    # Load checkpoint
    loaded_iteration = checkpointer.load(
        model_dict=model.model_dict,
        optimizer_dict=model.optimizer_dict,
        scheduler_dict=model.scheduler_dict,
        path=path,
    )

    dist.barrier()

    result["iteration_match"] = loaded_iteration == iteration

    # Verify DCP restored the EMA without replacing or unsharding it.
    loaded_ema_state = get_model_state_dict(model.ema, options=options)
    ema_still_sharded = all(isinstance(parameter, DTensor) for parameter in model.ema.parameters())
    ema_still_offloaded = all(parameter.to_local().device.type == "cpu" for parameter in model.ema.parameters())
    assert ema_still_sharded, f"rank {rank}: EMA was not FSDP-sharded after DCP load"
    assert ema_still_offloaded, f"rank {rank}: EMA was not CPU-offloaded after DCP load"
    result["ema_update_match"] = ema_update_match
    result["ema_sharded"] = ema_still_sharded
    result["ema_cpu_offloaded"] = ema_still_offloaded
    if rank == 0:
        result["ema_state_match"] = _compare_state_dicts(original_ema_state, loaded_ema_state)
        result["all_passed"] = (
            result["iteration_match"]
            and ema_update_match
            and result["ema_state_match"]
            and ema_still_sharded
            and ema_still_offloaded
        )
    else:
        result["ema_state_match"] = True
        result["all_passed"] = True

    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_fsdp_ema_checkpoint():
    """DCP restores a CPU-offloaded EMA after a real local-shard update."""
    clear_gpu_memory()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_distributed_test(
            test_fn=_test_fsdp_ema_checkpoint_impl,
            world_size=2,
            timeout=300,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )

        assert result is not None, "Test did not return a result"
        assert result.get("iteration_match", False), "Iteration mismatch in FSDP EMA checkpoint test"
        assert result.get("ema_update_match", False), "CPU-offloaded EMA local-shard update was incorrect"
        assert result.get("ema_sharded", False), "EMA was not FSDP-sharded after DCP load"
        assert result.get("ema_cpu_offloaded", False), "EMA was not CPU-offloaded after DCP load"
        assert result.get("ema_state_match", False), "EMA state not correctly restored"
        assert result.get("all_passed", False), "FSDP EMA checkpoint test failed"

    clear_gpu_memory()


# =============================================================================
# FSDP Optimizer State Verification Tests
# =============================================================================


def _make_fsdp_test_model(world_size):
    """Create an FSDP2 model with nested per-block sharding for optimizer checkpoint tests.

    We use a plain nn.Module instead of FastGenModel + its FSDP preparation context because the EDM
    model triggers an unpicklable-object issue in DCP under mp.spawn (the test
    infrastructure). The checkpointer logic being tested is architecture-independent.

    Imports are inside the function to avoid issues with mp.spawn serialization.
    """
    import torch.nn as nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from fastgen.utils.basic_utils import set_random_seed

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 128)
            self.linear2 = nn.Linear(128, 64)

        def forward(self, x):
            return x + self.linear2(torch.relu(self.linear1(x)))

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(64, 64)
            self.blocks = nn.ModuleList([_Block() for _ in range(3)])
            self.head = nn.Linear(64, 64)

        def forward(self, x):
            x = self.embed(x)
            for b in self.blocks:
                x = b(x)
            return self.head(x)

    mesh = init_device_mesh("cuda", (world_size,))
    set_random_seed(42)
    model = _Model().cuda()
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    return model


def _get_fqn_optim_state(model, optimizer):
    """Extract optimizer state as {fqn: {key: cpu_tensor}} for comparison across save/load."""
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    osd = get_optimizer_state_dict(model, optimizer)
    result = {}
    for fqn, vals in osd.get("state", {}).items():
        result[fqn] = {}
        for k, v in vals.items():
            if hasattr(v, "to_local"):
                result[fqn][k] = v.to_local().detach().cpu().clone()
            elif torch.is_tensor(v):
                result[fqn][k] = v.detach().cpu().clone()
            else:
                result[fqn][k] = v
    return result


def _compare_optim_states(state_a, state_b):
    """Compare two FQN-keyed optimizer states. Returns (match: bool, message: str)."""
    if set(state_a) != set(state_b):
        return False, (
            f"FQN sets differ: only_in_a={sorted(set(state_a) - set(state_b))}, "
            f"only_in_b={sorted(set(state_b) - set(state_a))}"
        )
    for fqn in state_a:
        for k in set(state_a[fqn]) | set(state_b[fqn]):
            va, vb = state_a[fqn].get(k), state_b[fqn].get(k)
            if va is None or vb is None:
                return False, f"{fqn}.{k}: missing in {'a' if va is None else 'b'}"
            if torch.is_tensor(va) and torch.is_tensor(vb):
                if not torch.allclose(va.float(), vb.float(), atol=1e-6):
                    return False, f"{fqn}.{k}: max_diff={(va.float()-vb.float()).abs().max().item()}"
            elif va != vb:
                return False, f"{fqn}.{k}: {va} vs {vb}"
    return True, "all match"


def _save_fsdp_checkpoint(net, optimizer, checkpoint_dir, iteration=1):
    """Save model+optimizer as a DCP checkpoint, return (checkpointer, path)."""
    ckpt_config = _get_checkpointer_config(checkpoint_dir)
    checkpointer = Checkpointer(ckpt_config, save_sharded=True)
    md = torch.nn.ModuleDict({"net": net})
    path = checkpointer.save(model_dict=md, optimizer_dict={"net": optimizer}, iteration=iteration)
    dist.barrier()
    return checkpointer, path


def _load_fsdp_checkpoint(checkpointer, path, net, optimizer):
    """Load model+optimizer from a DCP checkpoint, return loaded iteration."""
    md = torch.nn.ModuleDict({"net": net})
    loaded_iter = checkpointer.load(model_dict=md, optimizer_dict={"net": optimizer}, path=path)
    dist.barrier()
    return loaded_iter


def _test_fsdp_optimizer_state_restored_impl(rank, world_size, checkpoint_dir):
    """Verify every rank's optimizer state survives save/load within the comparison tolerance."""
    from fastgen.utils.basic_utils import set_random_seed

    net = _make_fsdp_test_model(world_size)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    set_random_seed(100, by_rank=True)
    net(torch.randn(8, 64, device="cuda")).sum().backward()
    opt.step()
    opt.zero_grad()

    ref = _get_fqn_optim_state(net, opt)
    checkpointer, path = _save_fsdp_checkpoint(net, opt, checkpoint_dir)

    net2 = _make_fsdp_test_model(world_size)
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    _load_fsdp_checkpoint(checkpointer, path, net2, opt2)

    loaded = _get_fqn_optim_state(net2, opt2)
    assert ref and loaded, "Expected populated optimizer state before and after loading"
    match, msg = _compare_optim_states(ref, loaded)
    assert match, msg
    result = {}
    if rank == 0:
        result = {"optim_match": match, "optim_msg": msg, "ref_fqns": len(ref), "loaded_fqns": len(loaded)}
    del net, net2
    clear_gpu_memory()
    return result


@RunIf(min_gpus=2)
def test_fsdp_optimizer_state_restored():
    """Test that FSDP optimizer state (momentum, step count) is correctly restored from checkpoint."""
    clear_gpu_memory()
    with tempfile.TemporaryDirectory() as tmpdir:
        r = run_distributed_test(
            test_fn=_test_fsdp_optimizer_state_restored_impl,
            world_size=2,
            timeout=300,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )
        assert r is not None, "No result"
        assert r.get("optim_match", False), (
            f"Optimizer state mismatch: {r.get('optim_msg', '?')}  "
            f"(ref={r.get('ref_fqns', '?')}, loaded={r.get('loaded_fqns', '?')})"
        )
    clear_gpu_memory()


def _test_fsdp_optimizer_training_continues_impl(rank, world_size, checkpoint_dir):
    """Verify resumed optimizer state on every rank and output within their comparison tolerances."""
    from fastgen.utils.basic_utils import set_random_seed

    net1 = _make_fsdp_test_model(world_size)
    opt1 = torch.optim.Adam(net1.parameters(), lr=1e-3)
    set_random_seed(100, by_rank=True)
    net1(torch.randn(8, 64, device="cuda")).sum().backward()
    opt1.step()
    opt1.zero_grad()

    checkpointer, path = _save_fsdp_checkpoint(net1, opt1, checkpoint_dir)

    set_random_seed(200, by_rank=True)
    net1(torch.randn(8, 64, device="cuda")).sum().backward()
    opt1.step()
    opt1.zero_grad()
    ref_state = _get_fqn_optim_state(net1, opt1)
    net1.eval()
    with torch.no_grad():
        set_random_seed(300, by_rank=True)
        ref_out = net1(torch.randn(4, 64, device="cuda")).cpu().clone()

    net2 = _make_fsdp_test_model(world_size)
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    _load_fsdp_checkpoint(checkpointer, path, net2, opt2)

    set_random_seed(200, by_rank=True)
    net2(torch.randn(8, 64, device="cuda")).sum().backward()
    opt2.step()
    opt2.zero_grad()
    loaded_state = _get_fqn_optim_state(net2, opt2)
    net2.eval()
    with torch.no_grad():
        set_random_seed(300, by_rank=True)
        loaded_out = net2(torch.randn(4, 64, device="cuda")).cpu().clone()

    assert ref_state and loaded_state, "Expected populated optimizer state for both training paths"
    match, msg = _compare_optim_states(ref_state, loaded_state)
    assert match, msg
    result = {}
    if rank == 0:
        out_diff = (ref_out.float() - loaded_out.float()).abs().max().item()
        result = {"state_match": match, "state_msg": msg, "output_match": out_diff < 1e-5, "output_diff": out_diff}
    del net1, net2
    clear_gpu_memory()
    return result


@RunIf(min_gpus=2)
def test_fsdp_optimizer_training_continues():
    """Test that training after checkpoint resume produces identical results to uninterrupted training."""
    clear_gpu_memory()
    with tempfile.TemporaryDirectory() as tmpdir:
        r = run_distributed_test(
            test_fn=_test_fsdp_optimizer_training_continues_impl,
            world_size=2,
            timeout=300,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )
        assert r is not None
        assert r.get("state_match", False), f"Optimizer state diverged: {r.get('state_msg', '?')}"
        assert r.get("output_match", False), f"Model output diverged: diff={r.get('output_diff', '?')}"
    clear_gpu_memory()


def test_dcp_optimizer_load_reads_metadata_once(tmp_path, monkeypatch):
    models = torch.nn.ModuleDict({"net": torch.nn.Linear(2, 2)})
    optimizer = torch.optim.Adam(models.parameters(), lr=1e-3)
    models["net"](torch.ones(2)).sum().backward()
    optimizer.step()
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=True)
    path = checkpointer.save(models, optimizer_dict={"net": optimizer}, iteration=7)
    original_get_reader = checkpointer.get_storage_reader
    metadata_reads = []

    def get_reader(checkpoint_path):
        reader = original_get_reader(checkpoint_path)
        original_read_metadata = reader.read_metadata

        def read_metadata():
            metadata_reads.append(checkpoint_path)
            return original_read_metadata()

        monkeypatch.setattr(reader, "read_metadata", read_metadata)
        return reader

    monkeypatch.setattr(checkpointer, "get_storage_reader", get_reader)
    assert checkpointer.load(models, optimizer_dict={"net": optimizer}, path=path) == 7
    assert metadata_reads.count(f"{path.removesuffix('.pth')}.net_optim") == 1


def _test_fsdp_optimizer_logging_impl(rank, world_size, checkpoint_dir):
    """Verify load logging reports correct counts and no spurious warnings."""
    import fastgen.utils.logging_utils as fg_logger
    from fastgen.utils.basic_utils import set_random_seed

    net = _make_fsdp_test_model(world_size)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    set_random_seed(100, by_rank=True)
    net(torch.randn(8, 64, device="cuda")).sum().backward()
    opt.step()
    opt.zero_grad()

    checkpointer, path = _save_fsdp_checkpoint(net, opt, checkpoint_dir)

    log_messages = []
    _orig_success, _orig_warning, _orig_info = fg_logger.success, fg_logger.warning, fg_logger.info
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            fg_logger, "success", lambda msg: (log_messages.append(("success", msg)), _orig_success(msg))
        )
        monkeypatch.setattr(
            fg_logger, "warning", lambda msg: (log_messages.append(("warning", msg)), _orig_warning(msg))
        )
        monkeypatch.setattr(fg_logger, "info", lambda msg: (log_messages.append(("info", msg)), _orig_info(msg)))

        net2 = _make_fsdp_test_model(world_size)
        opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
        _load_fsdp_checkpoint(checkpointer, path, net2, opt2)

    result = {}
    if rank == 0:
        success_msgs = [m for level, m in log_messages if level == "success" and "loaded optimizer state" in m.lower()]
        info_match_msgs = [m for level, m in log_messages if level == "info" and "params match checkpoint" in m.lower()]
        bad_warns = [
            m
            for level, m in log_messages
            if level == "warning" and ("missing from checkpoint" in m or "not in current model" in m)
        ]

        result = {
            "has_success_log": len(success_msgs) > 0,
            "success_log": success_msgs[0] if success_msgs else "",
            "has_param_count_log": len(info_match_msgs) > 0,
            "param_count_log": info_match_msgs[0] if info_match_msgs else "",
            "no_bad_warnings": len(bad_warns) == 0,
            "bad_warnings": bad_warns,
            "has_state": len(opt2.state) > 0,
        }
    del net, net2
    clear_gpu_memory()
    return result


@RunIf(min_gpus=2)
def test_fsdp_optimizer_logging():
    """Test that optimizer load logging reports correct FQNs with no false warnings."""
    clear_gpu_memory()
    with tempfile.TemporaryDirectory() as tmpdir:
        r = run_distributed_test(
            test_fn=_test_fsdp_optimizer_logging_impl,
            world_size=2,
            timeout=300,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )
        assert r is not None
        assert r.get("has_success_log", False), f"No success log found; got: {r}"
        assert r.get(
            "has_param_count_log", False
        ), f"Expected an info log of the form '... params match checkpoint'; got: {r}"
        assert r.get("no_bad_warnings", False), f"Spurious warnings: {r.get('bad_warnings', [])}"
        assert r.get("has_state", False), "Optimizer has no state after load"
    clear_gpu_memory()


def _test_fsdp_optimizer_mismatch_logging_impl(rank, world_size, checkpoint_dir):
    """Verify optimizer load logging for mismatched parameter sets (extra / missing)."""
    import fastgen.utils.logging_utils as fg_logger
    from fastgen.utils.basic_utils import set_random_seed

    @contextlib.contextmanager
    def _capture_logs():
        msgs = []
        _w, _s, _i = fg_logger.warning, fg_logger.success, fg_logger.info
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(fg_logger, "warning", lambda m: (msgs.append(("warning", m)), _w(m)))
            monkeypatch.setattr(fg_logger, "success", lambda m: (msgs.append(("success", m)), _s(m)))
            monkeypatch.setattr(fg_logger, "info", lambda m: (msgs.append(("info", m)), _i(m)))
            yield msgs

    result = {}

    # ---- Part 1: Extra params in checkpoint ----
    # Save with ALL params optimized
    net_full = _make_fsdp_test_model(world_size)
    opt_full = torch.optim.Adam(net_full.parameters(), lr=1e-3)
    set_random_seed(100, by_rank=True)
    net_full(torch.randn(8, 64, device="cuda")).sum().backward()
    opt_full.step()
    opt_full.zero_grad()

    full_dir = os.path.join(checkpoint_dir, "full")
    os.makedirs(full_dir, exist_ok=True)
    ckpt_full, path_full = _save_fsdp_checkpoint(net_full, opt_full, full_dir)

    # Load with frozen last block (optimizer tracks fewer params)
    net_frozen = _make_fsdp_test_model(world_size)
    for p in net_frozen.blocks[2].parameters():
        p.requires_grad = False
    opt_frozen = torch.optim.Adam(filter(lambda p: p.requires_grad, net_frozen.parameters()), lr=1e-3)
    set_random_seed(300, by_rank=True)
    net_frozen(torch.randn(8, 64, device="cuda")).sum().backward()
    opt_frozen.step()
    opt_frozen.zero_grad()

    with _capture_logs() as extra_logs:
        _load_fsdp_checkpoint(ckpt_full, path_full, net_frozen, opt_frozen)

    if rank == 0:
        extra_warns = [
            m
            for lvl, m in extra_logs
            if lvl == "warning" and "params in checkpoint" in m and "not in current model" in m
        ]
        result["extra_has_warning"] = len(extra_warns) > 0
        result["extra_warning_text"] = extra_warns[0] if extra_warns else ""
        result["extra_load_ok"] = any(
            lvl == "success" and "Successfully loaded optimizer state" in m for lvl, m in extra_logs
        )

    del net_full, net_frozen
    clear_gpu_memory()
    dist.barrier()

    # ---- Part 2: Missing params from checkpoint ----
    # Save with frozen last block (optimizer only has non-frozen params)
    net_frozen2 = _make_fsdp_test_model(world_size)
    for p in net_frozen2.blocks[2].parameters():
        p.requires_grad = False
    opt_frozen2 = torch.optim.Adam(filter(lambda p: p.requires_grad, net_frozen2.parameters()), lr=1e-3)
    set_random_seed(200, by_rank=True)
    net_frozen2(torch.randn(8, 64, device="cuda")).sum().backward()
    opt_frozen2.step()
    opt_frozen2.zero_grad()

    partial_dir = os.path.join(checkpoint_dir, "partial")
    os.makedirs(partial_dir, exist_ok=True)
    ckpt_part, path_part = _save_fsdp_checkpoint(net_frozen2, opt_frozen2, partial_dir)

    # Load with ALL params optimized
    net_full2 = _make_fsdp_test_model(world_size)
    opt_full2 = torch.optim.Adam(net_full2.parameters(), lr=1e-3)
    set_random_seed(400, by_rank=True)
    net_full2(torch.randn(8, 64, device="cuda")).sum().backward()
    opt_full2.step()
    opt_full2.zero_grad()

    with _capture_logs() as missing_logs:
        _load_fsdp_checkpoint(ckpt_part, path_part, net_full2, opt_full2)

    if rank == 0:
        missing_warns = [
            m
            for lvl, m in missing_logs
            if lvl == "warning" and "params in current model" in m and "missing from checkpoint" in m
        ]
        result["missing_has_warning"] = len(missing_warns) > 0
        result["missing_warning_text"] = missing_warns[0] if missing_warns else ""
        result["missing_load_ok"] = any(
            lvl == "success" and "Successfully loaded optimizer state" in m for lvl, m in missing_logs
        )

    del net_frozen2, net_full2
    clear_gpu_memory()
    return result


@RunIf(min_gpus=2)
def test_fsdp_optimizer_mismatch_logging():
    """Test optimizer logging when checkpoint and model have different parameter sets."""
    clear_gpu_memory()
    with tempfile.TemporaryDirectory() as tmpdir:
        r = run_distributed_test(
            test_fn=_test_fsdp_optimizer_mismatch_logging_impl,
            world_size=2,
            timeout=300,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )
        assert r is not None, "No result returned"

        assert r.get(
            "extra_has_warning", False
        ), f"Expected warning about extra params in checkpoint: {r.get('extra_warning_text', 'none')}"
        assert r.get("extra_load_ok", False), "Load should succeed despite extra params in checkpoint"

        assert r.get(
            "missing_has_warning", False
        ), f"Expected warning about missing params from checkpoint: {r.get('missing_warning_text', 'none')}"
        assert r.get("missing_load_ok", False), "Load should succeed despite missing params from checkpoint"
    clear_gpu_memory()


# =============================================================================
# Checkpoint File Structure Tests
# =============================================================================


def test_checkpoint_file_structure():
    """Test that checkpoint file has expected structure."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_sft_model(use_ema=True, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        iteration = 150

        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            callbacks=None,
            iteration=iteration,
        )

        # Load and verify structure
        checkpoint = torch.load(path, weights_only=False)

        # Verify top-level keys
        assert "model" in checkpoint, "Checkpoint should have 'model' key"
        assert "optimizer" in checkpoint, "Checkpoint should have 'optimizer' key"
        assert "scheduler" in checkpoint, "Checkpoint should have 'scheduler' key"
        assert "grad_scaler" in checkpoint, "Checkpoint should have 'grad_scaler' key"
        assert "iteration" in checkpoint, "Checkpoint should have 'iteration' key"

        # Verify model contains expected networks
        assert "net" in checkpoint["model"], "Checkpoint should contain 'net'"
        assert "ema" in checkpoint["model"], "Checkpoint should contain 'ema' (EMA is enabled)"

        # Verify iteration
        assert checkpoint["iteration"] == iteration, "Iteration mismatch"


# =============================================================================
# S3 Checkpointing Tests (resume)
# =============================================================================


def _checkpoint_test_net(weight: float = 0.0, bias: float = 0.0) -> torch.nn.Linear:
    net = torch.nn.Linear(4, 4)
    torch.nn.init.constant_(net.weight, weight)
    torch.nn.init.constant_(net.bias, bias)
    return net


@pytest.mark.parametrize("path", [None, ""])
def test_unset_checkpoint_path_helpers_preserve_the_no_checkpoint_sentinel(path):
    assert validate_checkpoint_path(path) is None
    assert not path_exists(path)


@pytest.mark.parametrize(
    "path",
    ["0000042", "/checkpoints/0000042", "s3://bucket/run/0000042", "/checkpoints/0000042.pt"],
    ids=["relative-local", "absolute-local", "s3", "wrong-extension"],
)
def test_validate_checkpoint_path_rejects_non_marker_paths(path):
    with pytest.raises(ValueError, match=r"\.pth"):
        validate_checkpoint_path(path)


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "sharded"])
@pytest.mark.parametrize(
    "path",
    ["0000042", "/checkpoints/0000042", "s3://bucket/run/0000042", "/checkpoints/0000042.pt"],
    ids=["relative-local", "absolute-local", "s3", "wrong-extension"],
)
def test_checkpointers_reject_non_marker_save_and_load_paths_before_io(tmp_path, monkeypatch, save_sharded, path):
    def unexpected_storage_access(*args, **kwargs):
        pytest.fail("invalid checkpoint paths must not read, invalidate, or publish checkpoints")

    monkeypatch.setattr("fastgen.utils.checkpointer.path_exists", unexpected_storage_access)
    monkeypatch.setattr("fastgen.utils.checkpointer._remove_checkpoint_artifact", unexpected_storage_access)
    monkeypatch.setattr(Checkpointer, "_save_checkpoint", unexpected_storage_access)
    monkeypatch.setattr("fastgen.utils.checkpointer.load_training_checkpoint", unexpected_storage_access)
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    for operation in (
        lambda: checkpointer.resolve_save_path(path, iteration=42),
        lambda: checkpointer.resolve_load_path(path),
        lambda: checkpointer.save(torch.nn.ModuleDict(), path=path, iteration=42),
        lambda: checkpointer.load(torch.nn.ModuleDict(), path=path),
    ):
        with pytest.raises(ValueError, match=r"\.pth"):
            operation()
    assert not list(tmp_path.iterdir()), "validation must not create checkpoint artifacts"


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "sharded"])
def test_checkpointers_reject_relative_save_locations_before_io(tmp_path, monkeypatch, save_sharded):
    """Without ``setup``, a relative ``save_dir`` would silently resolve against the working directory."""

    def unexpected_storage_access(*args, **kwargs):
        pytest.fail("relative checkpoint locations must be rejected before storage access")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("fastgen.utils.checkpointer.latest_checkpoint", unexpected_storage_access)
    monkeypatch.setattr("fastgen.utils.checkpointer._remove_checkpoint_artifact", unexpected_storage_access)
    monkeypatch.setattr(Checkpointer, "_save_checkpoint", unexpected_storage_access)
    checkpointer = Checkpointer(_get_checkpointer_config("checkpoints"), save_sharded=save_sharded)
    for operation in (
        lambda: checkpointer.save(torch.nn.ModuleDict(), iteration=42),
        lambda: checkpointer.save(torch.nn.ModuleDict(), path="checkpoints/latest_ar.pth", iteration=42),
        lambda: checkpointer.load(torch.nn.ModuleDict()),
    ):
        with pytest.raises(ValueError, match="absolute path or an s3:// URI"):
            operation()
    assert not list(tmp_path.iterdir()), "rejection must not create checkpoint artifacts"


@pytest.mark.parametrize("payload", [[{"iteration": 42}], torch.ones(1), None], ids=["list", "tensor", "none"])
def test_training_checkpoint_load_rejects_a_non_dict_marker(tmp_path, payload):
    marker_path = str(tmp_path / "0000042.pth")
    torch.save(payload, marker_path)

    with pytest.raises(TypeError, match=f"holds a {type(payload).__name__}, not a state dict"):
        load_training_checkpoint(marker_path)


@pytest.mark.parametrize("discover_latest", [False, True], ids=["explicit-marker", "discover-latest"])
@pytest.mark.parametrize("save_sharded", [False, True], ids=["save-monolithic", "save-sharded"])
def test_dcp_checkpointer_loads_sharded_marker(tmp_path, discover_latest, save_sharded, monkeypatch):
    source = _checkpoint_test_net(weight=2.0, bias=3.0)
    config = _get_checkpointer_config(str(tmp_path))
    saver = Checkpointer(config, save_sharded=True)
    marker_path = saver.save(torch.nn.ModuleDict({"net": source}), iteration=42)
    checkpointer = Checkpointer(config, save_sharded=save_sharded)
    assert marker_path == str(tmp_path / "0000042.pth")
    marker_checks = []
    original_exists = os.path.exists

    def record_exists(path):
        if str(path) == marker_path:
            marker_checks.append(path)
        return original_exists(path)

    monkeypatch.setattr("fastgen.utils.io_utils.os.path.exists", record_exists)
    target = _checkpoint_test_net()
    path = None if discover_latest else marker_path
    assert checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path, device="cpu") == 42
    assert marker_checks == [marker_path], "load should check the completion marker only once"
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


@pytest.mark.parametrize("sharded_keys", [(), ("net",)], ids=["no-shards", "missing-ema-shards"])
def test_dcp_checkpointer_names_missing_shards_before_loading(tmp_path, monkeypatch, sharded_keys):
    marker_path = tmp_path / "0000042.pth"
    torch.save({"iteration": 42, "dcp_manifest": {"models": ("net", "ema"), "optimizers": ()}}, marker_path)
    for key in sharded_keys:
        metadata_path = tmp_path / f"0000042.{key}_model" / ".metadata"
        metadata_path.parent.mkdir()
        metadata_path.touch()
    monkeypatch.setattr(
        "fastgen.utils.checkpointer.dcp.load",
        lambda *args, **kwargs: pytest.fail("a checkpoint with missing shards must not load any key"),
    )
    model_dict = torch.nn.ModuleDict({"net": _checkpoint_test_net(), "ema": _checkpoint_test_net()})

    with pytest.raises(FileNotFoundError) as error:
        Checkpointer(_get_checkpointer_config(str(tmp_path))).load(model_dict, path=str(marker_path))

    missing = [key for key in model_dict if key not in sharded_keys]
    message = str(error.value)
    assert message.startswith(f"{marker_path} is neither a monolithic checkpoint (it has no 'model' entry)")
    assert f"{tmp_path / '0000042'}.<key>_model/.metadata is missing for model key(s) {missing}" in message


@pytest.mark.parametrize("adapter_only", [False, True])
@pytest.mark.parametrize("save_sharded", [False, True], ids=["save-monolithic", "save-sharded"])
def test_format_aware_checkpointer_delegates_monolithic_save_and_load(
    tmp_path, adapter_only, save_sharded, monkeypatch
):
    config = _get_checkpointer_config(str(tmp_path))
    config.save_adapter_only = adapter_only
    source = _tiny_lora_net(1) if adapter_only else _checkpoint_test_net(weight=2.0, bias=3.0)
    target = _tiny_lora_net(1) if adapter_only else _checkpoint_test_net()
    # The constructor accepts save_sharded positionally as well as by keyword.
    saver = Checkpointer(config, False)
    assert saver.save_sharded is False
    path = saver.save(torch.nn.ModuleDict({"net": source}), iteration=9)
    assert path == str(tmp_path / "0000009.pth")
    assert torch.load(path, weights_only=False)["model"]
    checkpointer = Checkpointer(config, save_sharded=save_sharded)
    loaded_paths = []
    monkeypatch.setattr(
        "fastgen.utils.checkpointer.load_training_checkpoint",
        lambda path, device: loaded_paths.append(path) or load_training_checkpoint(path, device),
    )
    assert checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path) == 9
    assert loaded_paths == [path], "delegation must not read a full monolithic checkpoint twice"
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


@pytest.mark.parametrize("source", ["zipfile", "legacy", "s3"])
def test_format_probe_skips_monolithic_payload_without_shards(tmp_path, monkeypatch, source):
    if source == "s3":
        path = "s3://bucket/checkpoints/0000001.pth"
        monkeypatch.setattr("fastgen.utils.s3.exists", lambda _: False)
        monkeypatch.setattr("fastgen.utils.s3.load", lambda _: pytest.fail("Probe downloaded checkpoint"))
    else:
        path = str(tmp_path / "0000001.pth")
        torch.save(
            {"model": {"net": torch.nn.Linear(3, 4).state_dict()}, "iteration": 1},
            path,
            _use_new_zipfile_serialization=source != "legacy",
        )
    monkeypatch.setattr(
        "fastgen.utils.checkpointer.load_training_checkpoint", lambda *a, **kw: pytest.fail("Probe read checkpoint")
    )
    assert not is_dcp_checkpoint(path, ("net", "net_ema"))


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("source", ["local", "s3"])
def test_format_probe_uses_contents_and_meta_storage(tmp_path, monkeypatch, sharded, source):
    state = {"iteration": 1} if sharded else {"model": {"net": torch.nn.Linear(3, 4).state_dict()}, "iteration": 1}
    # Stale shard directories must not misclassify a monolithic overwrite.
    if source == "s3":
        path = "s3://bucket/checkpoints/0000001.pth"
        metadata = "s3://bucket/checkpoints/0000001.net_model/.metadata"
        payload = io.BytesIO()
        torch.save(state, payload)
        monkeypatch.setattr("fastgen.utils.s3.exists", lambda p: p in (path, metadata))
        monkeypatch.setattr("fastgen.utils.s3.load", lambda _: io.BytesIO(payload.getvalue()))
    else:
        path = str(tmp_path / "0000001.pth")
        torch.save(state, path)
        metadata = tmp_path / "0000001.net_model" / ".metadata"
        metadata.parent.mkdir()
        metadata.touch()
    reads = []

    def read(*args, **kwargs):
        assert kwargs == {"device": "meta", "mmap": True}
        loaded = load_training_checkpoint(*args, **kwargs)
        if not sharded:
            assert loaded["model"]["net"]["weight"].is_meta
        reads.append(args[0])
        return loaded

    monkeypatch.setattr("fastgen.utils.checkpointer.load_training_checkpoint", read)
    assert is_dcp_checkpoint(path, ("missing_root", "net")) is sharded
    assert reads == [path]
    assert not is_dcp_checkpoint(None, ("net",))
    assert not is_dcp_checkpoint("", ("net",))
    assert reads == [path]


@pytest.mark.parametrize("use_s3", [False, True], ids=["local", "s3"])
def test_checkpoint_load_checks_marker_existence_once(tmp_path, monkeypatch, use_s3):
    source = _checkpoint_test_net(weight=2.0, bias=3.0)
    state = {"model": {"net": source.state_dict()}, "iteration": 42}
    local_path = tmp_path / "0000042.pth"
    torch.save(state, local_path)
    checkpoint_path = "s3://bucket/run/0000042.pth" if use_s3 else str(local_path)
    checks = []
    downloads = []
    original_exists = os.path.exists

    def local_exists(path):
        if str(path) == checkpoint_path:
            checks.append(path)
        return original_exists(path)

    def s3_exists(path):
        assert use_s3, "a local checkpoint must not query S3"
        checks.append(path)
        return True

    def s3_load(path):
        assert use_s3
        downloads.append(path)
        return io.BytesIO(local_path.read_bytes())

    monkeypatch.setattr("fastgen.utils.io_utils.os.path.exists", local_exists)
    monkeypatch.setattr("fastgen.utils.s3.exists", s3_exists)
    monkeypatch.setattr("fastgen.utils.s3.load", s3_load)
    target = _checkpoint_test_net()
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)))

    iteration = checkpointer.load(torch.nn.ModuleDict({"net": target}), path=checkpoint_path, device="cpu")

    assert iteration == 42
    assert checks == [checkpoint_path], "load and dispatch must not duplicate the marker existence check"
    assert downloads == ([checkpoint_path] if use_s3 else [])
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


@pytest.mark.parametrize("use_s3", [False, True], ids=["local", "s3"])
def test_checkpoint_load_missing_marker_fails_before_read(tmp_path, monkeypatch, use_s3):
    checkpoint_path = "s3://bucket/run/missing.pth" if use_s3 else str(tmp_path / "missing.pth")
    checks = []
    original_exists = os.path.exists

    def local_exists(path):
        if str(path) == checkpoint_path:
            checks.append(path)
        return original_exists(path)

    def s3_exists(path):
        assert use_s3
        checks.append(path)
        return False

    def unexpected_read(*args, **kwargs):
        pytest.fail("a missing checkpoint must not be opened or deserialized")

    monkeypatch.setattr("fastgen.utils.io_utils.os.path.exists", local_exists)
    monkeypatch.setattr("fastgen.utils.s3.exists", s3_exists)
    monkeypatch.setattr("fastgen.utils.s3.load", unexpected_read)
    monkeypatch.setattr(torch, "load", unexpected_read)
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)))

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        checkpointer.load(torch.nn.ModuleDict(), path=checkpoint_path)

    assert checks == [checkpoint_path]


@pytest.mark.parametrize("failure_phase", ["exists", "load"])
def test_checkpoint_load_propagates_s3_access_errors(monkeypatch, failure_phase):
    from botocore.exceptions import ClientError

    checkpoint_path = "s3://bucket/run/0000042.pth"
    operation = "HeadObject" if failure_phase == "exists" else "GetObject"
    storage_error = ClientError({"Error": {"Code": "AccessDenied", "Message": "permission denied"}}, operation)
    checks = []
    downloads = []

    def s3_exists(path):
        checks.append(path)
        if failure_phase == "exists":
            raise storage_error
        return True

    def s3_load(path):
        downloads.append(path)
        raise storage_error

    monkeypatch.setattr("fastgen.utils.s3.exists", s3_exists)
    monkeypatch.setattr("fastgen.utils.s3.load", s3_load)
    checkpointer = Checkpointer(_get_checkpointer_config("unused"))

    with pytest.raises(ClientError) as error:
        checkpointer.load(torch.nn.ModuleDict(), path=checkpoint_path)

    assert error.value is storage_error
    assert checks == [checkpoint_path]
    assert downloads == ([checkpoint_path] if failure_phase == "load" else [])


def test_checkpointer_load_resumes_from_s3(monkeypatch):
    """``load`` must ask S3 whether the checkpoint exists.

    ``os.path.exists`` is always False for an s3:// URL, so an unguarded check would
    discard the checkpoint ``latest_checkpoint`` just found and silently restart the
    run from iteration 0.
    """
    expected = _checkpoint_test_net(weight=0.5, bias=-0.25).state_dict()
    payload = io.BytesIO()
    torch.save(
        {
            "model": {"net": expected},
            "optimizer": None,
            "scheduler": None,
            "grad_scaler": None,
            "callbacks": None,
            "iteration": 42,
        },
        payload,
    )

    config = _get_checkpointer_config("s3://bucket/fastgen/run")

    discovery_roots = []
    existence_checks = []
    monkeypatch.setattr(
        "fastgen.utils.checkpointer.latest_checkpoint",
        lambda path: discovery_roots.append(path) or "s3://bucket/fastgen/run/0000042.pth",
    )
    monkeypatch.setattr("fastgen.utils.s3.exists", lambda path: existence_checks.append(path) or True)
    monkeypatch.setattr("fastgen.utils.s3.load", lambda path: io.BytesIO(payload.getvalue()))

    net = _checkpoint_test_net()
    iteration = Checkpointer(config).load(model_dict=torch.nn.ModuleDict({"net": net}), device="cpu")

    assert discovery_roots == ["s3://bucket/fastgen/run"]
    assert existence_checks == ["s3://bucket/fastgen/run/0000042.pth"]
    assert iteration == 42, "an existing S3 checkpoint must not be discarded"
    assert torch.equal(net.weight, expected["weight"])


def test_checkpointer_reads_relative_load_paths_locally_with_use_s3(tmp_path, monkeypatch):
    """``use_s3`` never sends a relative load path to S3; ``s3_container`` only roots ``save_dir`` in ``setup``."""
    torch.save({"model": {}, "iteration": 42}, tmp_path / "0000042.pth")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("fastgen.utils.s3.exists", lambda path: pytest.fail(f"queried S3 for {path}"))
    config = _get_checkpointer_config("s3://bucket/fastgen/run")
    config.use_s3 = True

    iteration = Checkpointer(config).load(model_dict=torch.nn.ModuleDict(), path="0000042.pth", device="cpu")

    assert iteration == 42


def test_checkpointer_load_returns_zero_when_no_checkpoint_is_discovered(tmp_path):
    config = _get_checkpointer_config(str(tmp_path))

    iteration = Checkpointer(config).load(model_dict=torch.nn.ModuleDict({"net": _checkpoint_test_net()}), device="cpu")

    assert iteration == 0


def test_checkpointer_load_raises_when_discovered_s3_checkpoint_is_missing(monkeypatch):
    """A checkpoint that disappears after discovery must not silently restart training."""
    config = _get_checkpointer_config("s3://bucket/fastgen/run")

    monkeypatch.setattr(
        "fastgen.utils.checkpointer.latest_checkpoint", lambda path: "s3://bucket/fastgen/run/0000042.pth"
    )
    monkeypatch.setattr("fastgen.utils.s3.exists", lambda path: False)

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        Checkpointer(config).load(model_dict=torch.nn.ModuleDict({"net": _checkpoint_test_net()}), device="cpu")


def test_checkpointer_loads_an_explicit_s3_checkpoint_without_use_s3(monkeypatch):
    """Reads dispatch on the path: a filesystem-saving run can load a pretrained checkpoint from S3."""
    expected = _checkpoint_test_net(weight=0.5, bias=-0.25).state_dict()
    payload = io.BytesIO()
    torch.save({"model": {"net": expected}, "iteration": 7}, payload)
    loaded_paths = []
    monkeypatch.setattr("fastgen.utils.s3.exists", lambda path: True)
    monkeypatch.setattr(
        "fastgen.utils.s3.load", lambda path: loaded_paths.append(path) or io.BytesIO(payload.getvalue())
    )

    net = _checkpoint_test_net()
    iteration = Checkpointer(_get_checkpointer_config("checkpoints")).load(
        model_dict=torch.nn.ModuleDict({"net": net}), path="s3://bucket/pretrained/0000007.pth", device="cpu"
    )

    assert loaded_paths == ["s3://bucket/pretrained/0000007.pth"]
    assert iteration == 7
    assert torch.equal(net.weight, expected["weight"])


def test_checkpointer_loads_an_absolute_local_checkpoint_with_use_s3(tmp_path):
    """Absolute local paths are not prefixed with the S3 container, so an S3-saving run can start from shared storage."""
    expected = _checkpoint_test_net(weight=0.5, bias=-0.25).state_dict()
    path = tmp_path / "pretrained.pth"
    torch.save({"model": {"net": expected}, "iteration": 3}, path)
    config = BaseCheckpointerConfig()
    config.use_s3 = True
    config.s3_container = "s3://bucket/fastgen"

    net = _checkpoint_test_net()
    iteration = Checkpointer(config).load(model_dict=torch.nn.ModuleDict({"net": net}), path=str(path), device="cpu")

    assert iteration == 3
    assert torch.equal(net.weight, expected["weight"])


@pytest.mark.parametrize(
    ("strict_kwargs", "key_mismatch", "expected_partial_load", "expect_strict_error"),
    [
        ({}, "none", False, False),
        ({"strict": False}, "mixed", True, False),
        ({}, "unexpected", False, True),
    ],
    ids=["strict-default", "partial-pretrained", "strict-unexpected"],
)
def test_dcp_checkpointer_load_uses_s3_metadata_objects(
    monkeypatch,
    strict_kwargs,
    key_mismatch,
    expected_partial_load,
    expect_strict_error,
):
    """DCP S3 resume uses object metadata and explicit model-load strictness."""
    prefix = "s3://bucket/fastgen/run/0000042"
    scalar = io.BytesIO()
    torch.save(
        {
            "scheduler": None,
            "grad_scaler": None,
            "callbacks": None,
            "iteration": 42,
            "dcp_manifest": {"models": ("net",), "optimizers": ()},
        },
        scalar,
    )

    config = _get_checkpointer_config("s3://bucket/fastgen/run")

    existence_checks = []
    existing = {f"{prefix}.pth", f"{prefix}.net_model/.metadata"}
    monkeypatch.setattr("fastgen.utils.checkpointer.latest_checkpoint", lambda path: f"{prefix}.pth")
    monkeypatch.setattr(
        "fastgen.utils.s3.exists",
        lambda path: existence_checks.append(path) or path in existing,
    )
    monkeypatch.setattr("fastgen.utils.s3.load", lambda path: io.BytesIO(scalar.getvalue()))
    monkeypatch.setattr(Checkpointer, "get_storage_reader", lambda self, checkpoint_path: object())
    load_planners = []
    load_messages = []
    monkeypatch.setattr("fastgen.utils.checkpointer.logger.warning", lambda m: load_messages.append(("warning", m)))
    monkeypatch.setattr("fastgen.utils.checkpointer.logger.success", lambda m: load_messages.append(("success", m)))

    def record_model_load(state_dict, storage_reader, planner=None):
        metadata_keys = {key: object() for key in state_dict}
        if key_mismatch == "mixed":
            metadata_keys.pop("bias")
            metadata_keys["unexpected.weight"] = object()
        elif key_mismatch == "unexpected":
            metadata_keys["unexpected.weight"] = object()
        planner.set_up_planner(
            state_dict,
            SimpleNamespace(state_dict_metadata=metadata_keys),
            is_coordinator=True,
        )
        load_planners.append(planner)

    monkeypatch.setattr(
        "fastgen.utils.checkpointer.dcp.load",
        record_model_load,
    )

    net = _checkpoint_test_net()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    load_kwargs = {
        "model_dict": torch.nn.ModuleDict({"net": net}),
        "optimizer_dict": {"net": optimizer},
        "device": "cpu",
        **strict_kwargs,
    }
    if expect_strict_error:
        with pytest.raises(RuntimeError, match="unexpected.weight"):
            Checkpointer(config).load(**load_kwargs)
        iteration = None
    else:
        iteration = Checkpointer(config).load(**load_kwargs)

    assert iteration == (None if expect_strict_error else 42)
    assert len(load_planners) == 1
    assert load_planners[0].allow_partial_load is expected_partial_load
    model_load_info = [(level, message) for level, message in load_messages if "Model net, loading info:" in message]
    assert len(model_load_info) == 1
    level, message = model_load_info[0]
    assert level == ("success" if key_mismatch == "none" else "warning")
    if key_mismatch == "mixed":
        assert "bias" in message
        assert "unexpected.weight" in message
    elif key_mismatch == "unexpected":
        assert "missing_keys=[]" in message
        assert "unexpected.weight" in message
    expected_existence_checks = [
        f"{prefix}.pth",
        f"{prefix}.net_model/.metadata",
    ]
    assert existence_checks == expected_existence_checks


def _legacy_load_aligns_uncovered_state_impl(rank: int, world_size: int, checkpoint_dir: str) -> dict:
    """A monolithic ``.pth`` loaded under meta init leaves every rank with rank 0's uncovered state,
    including missing checkpoint keys and nonpersistent buffers."""

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)
            self.register_buffer("scale", torch.randn(4), persistent=False)

    path = os.path.join(checkpoint_dir, "pretrained.pth")
    torch.manual_seed(100 + rank)  # rank-dependent values, like ``reset_parameters`` on a meta rank
    if rank == 0:
        net = Net()
        covered = {name: value for name, value in net.state_dict().items() if name != "lin.bias"}
        torch.save({"model": {"net": covered}, "iteration": 7}, path)
    else:
        with torch.device("meta"):
            net = Net()
    dist.barrier()

    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir))
    iteration = checkpointer.load(torch.nn.ModuleDict({"net": net}), path=path, strict=False)

    def agrees_everywhere(tensor: torch.Tensor) -> bool:
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.detach().contiguous())
        return all(torch.equal(gathered[0], other) for other in gathered[1:])

    return {
        "iteration": iteration,
        "no_meta_left": not any(t.is_meta for t in [*net.parameters(), *net.buffers()]),
        "covered_weight_agrees": agrees_everywhere(net.lin.weight),
        "missing_bias_agrees": agrees_everywhere(net.lin.bias),
        "nonpersistent_buffer_agrees": agrees_everywhere(net.scale),
    }


@RunIf(min_gpus=2)
def test_legacy_pretrained_load_aligns_uncovered_state_under_meta_init():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_distributed_test(
            test_fn=_legacy_load_aligns_uncovered_state_impl,
            world_size=2,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )
    assert result is not None, "distributed test returned nothing (a rank likely crashed)"
    assert result["iteration"] == 7
    assert result["no_meta_left"]
    assert result["covered_weight_agrees"]
    assert result["missing_bias_agrees"]
    assert result["nonpersistent_buffer_agrees"]


@pytest.mark.parametrize("wrapper", ["checkpoint", "compile", "checkpoint-and-compile"])
def test_dcp_key_mapping_resolves_wrapper_registration_names(wrapper):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from fastgen.utils.checkpointer import _registration_names_for_dcp_keys

    leaf = torch.nn.Linear(2, 2)
    leaf.register_buffer("scale", torch.ones(2))
    use_checkpoint = "checkpoint" in wrapper
    use_compile = "compile" in wrapper
    module = torch.nn.Sequential(checkpoint_wrapper(leaf) if use_checkpoint else leaf)
    if use_compile:
        # No compiled forward is needed: the real wrapper's registration names are the contract.
        module = torch.compile(module, backend="eager")
    prefix = ("_orig_mod." if use_compile else "") + "0."
    if use_checkpoint:
        prefix += "_checkpoint_wrapped_module."

    assert _registration_names_for_dcp_keys(module, {"0.weight", "0.scale"}) == {
        f"{prefix}weight",
        f"{prefix}scale",
    }


def test_dcp_key_mapping_keeps_shared_module_alias_registrations():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from fastgen.utils.checkpointer import _registration_names_for_dcp_keys

    wrapped = checkpoint_wrapper(torch.nn.Linear(2, 2))
    module = torch.nn.ModuleDict({"first": wrapped, "second": wrapped})

    assert _registration_names_for_dcp_keys(module, {"first.weight", "second.weight"}) == {
        "first._checkpoint_wrapped_module.weight",
        "second._checkpoint_wrapped_module.weight",
    }


@pytest.mark.parametrize("key", ["missing.weight", "cache"])
def test_dcp_key_mapping_rejects_unregistered_or_nonpersistent_keys(key):
    from fastgen.utils.checkpointer import _registration_names_for_dcp_keys

    module = torch.nn.Linear(2, 2)
    module.register_buffer("cache", torch.ones(2), persistent=False)

    with pytest.raises(AssertionError, match="DCP keys without a registered tensor") as error:
        _registration_names_for_dcp_keys(module, {key})
    assert key in str(error.value)


def test_dcp_key_mapping_skips_registration_work_for_empty_keys(monkeypatch):
    import fastgen.utils.checkpointer as checkpointer_module

    monkeypatch.setattr(
        checkpointer_module, "_get_fqns", lambda *args, **kwargs: pytest.fail("empty key selection inspected state")
    )
    assert checkpointer_module._registration_names_for_dcp_keys(torch.nn.Linear(2, 2), []) == set()


class _DCPMetaLoadModule(FastGenModule):
    """Checkpointed and constructor-only state, deliberately without a root reset hook."""

    def __init__(
        self,
        value_base: float,
        include_optional: bool = True,
        include_cache: bool = True,
        include_wrapped_child: bool = False,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((8,), value_base + 1))
        self.register_buffer("scale", torch.full((8,), value_base + 3))
        if include_optional:
            self.retained_weight = torch.nn.Parameter(torch.full((8,), value_base + 2))
            self.register_buffer("retained_scale", torch.full((8,), value_base + 4))
        if include_cache:
            self.register_buffer("cache", torch.full((8,), value_base + 5), persistent=False)
        if include_wrapped_child:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

            child = torch.nn.Module()
            child.weight = torch.nn.Parameter(torch.full((8,), value_base + 6))
            child.register_buffer("scale", torch.full((8,), value_base + 7))
            self.wrapped = checkpoint_wrapper(child)


@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
@pytest.mark.parametrize("persistent", [False, True])
def test_checkpoint_save_and_copy_load_allow_runtime_buffer_aliases(tmp_path, save_sharded, persistent):
    source = _DCPMetaLoadModule(10.0)
    source.prepare_module("cpu")
    # Runtime updates can tie existing buffer slots after preparation.
    source.register_buffer("scale_alias", source.scale, persistent=persistent)
    source.register_buffer("cache_alias", source.cache, persistent=False)
    target = _DCPMetaLoadModule(100.0)
    target.prepare_module("cpu")
    target.register_buffer("scale_alias", target.scale, persistent=persistent)
    target.register_buffer("cache_alias", target.cache, persistent=False)
    original_weight, original_scale, original_cache = target.weight, target.scale, target.cache
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)

    path = checkpointer.save(torch.nn.ModuleDict({"net": source}), iteration=19)
    iteration = checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path, device="cpu")

    assert iteration == 19
    assert source.scale is source.scale_alias
    assert source.cache is source.cache_alias
    assert target.weight is original_weight
    assert target.scale is target.scale_alias is original_scale
    assert target.cache is target.cache_alias is original_cache
    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(target.scale, source.scale)
    torch.testing.assert_close(target.cache, torch.full((8,), 105.0))
    assert target.is_prepared and target.device == torch.device("cpu")


def _checkpoint_load_tied_parameters_worker(rank, world_size, checkpoint_dir, save_sharded, target_device):
    def make_module(device):
        with torch.device(device):
            module = torch.nn.Module()
            module.left = torch.nn.Linear(2, 2, bias=False)
            module.right = torch.nn.Linear(2, 2, bias=False)
            module.right.weight = module.left.weight
            module.left.register_buffer("cache", torch.ones(2))
            module.left_alias = module.left
        return module

    torch.manual_seed(0)
    source, target = make_module("cpu"), make_module(target_device if rank else "cpu")
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=save_sharded)
    path = checkpointer.save(torch.nn.ModuleDict({"net": source}), iteration=19)

    assert checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path, device="cpu") == 19

    assert target.left.weight is target.right.weight
    assert target.left is target.left_alias
    assert target.left.cache is target.left_alias.cache
    torch.testing.assert_close(target.left.weight, source.left.weight)
    torch.testing.assert_close(target.left.cache, source.left.cache)
    return True


@RunIf(min_gpus=2)
@pytest.mark.parametrize("save_sharded", [False, True], ids=["monolithic", "dcp"])
@pytest.mark.parametrize("target_device", ["cpu", "meta"])
def test_checkpoint_load_accepts_tied_parameters_and_shared_submodules(tmp_path, save_sharded, target_device):
    assert run_distributed_test(
        _checkpoint_load_tied_parameters_worker,
        world_size=2,
        timeout=120,
        nccl_timeout=30,
        checkpoint_dir=str(tmp_path),
        save_sharded=save_sharded,
        target_device=target_device,
    )


def _checkpoint_load_preserves_meta_target_state_impl(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
    scenario: str,
    source_device: str,
    load_device: str,
    checkpoint_format: str = "dcp",
) -> dict:
    """Checkpoint loading precedes eager preparation, or reuses an already-prepared FSDP root."""
    from unittest.mock import patch

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    import fastgen.utils.checkpointer as checkpointer_module
    import fastgen.utils.distributed.module as distributed_module
    from fastgen.utils.distributed.fsdp import FSDPContext
    from fastgen.utils.module import has_meta_state, local_tensor, module_state_tensors
    from scripts.inference.inference_utils import load_checkpoint

    partial = scenario in {"partial", "partial_wrapped"}
    wrapped_missing = scenario == "partial_wrapped"
    already_prepared = scenario == "prepared_fsdp"
    include_cache = scenario != "no_uncovered"
    device = torch.device("cuda", torch.cuda.current_device())
    target_source_device = device if source_device == "cuda" else torch.device("cpu")
    target_load_device = device if load_device == "cuda" else torch.device("cpu")
    mesh = init_device_mesh("cuda", (world_size,))
    source = _DCPMetaLoadModule(10.0, include_optional=not partial, include_cache=include_cache).to(device)
    if checkpoint_format == "dcp":
        fully_shard(source, mesh=mesh)
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=checkpoint_format == "dcp")
    path = checkpointer.save(torch.nn.ModuleDict({"net": source}), iteration=19)

    if already_prepared:
        target = _DCPMetaLoadModule(100.0, include_cache=include_cache)
        context = FSDPContext(
            mesh, min_num_params=1, reduce_dtype=torch.float32, offload_policy=None, cpu_offload=False
        )
        target.prepare_module(device, precision_model=torch.float32, trainable=True, fsdp_context=context)
    elif rank == 0:
        # Exercise the checkpoint loader's own CUDA-to-CPU placement path.
        with torch.device(target_source_device):
            target = _DCPMetaLoadModule(
                100.0,
                include_cache=include_cache,
                include_wrapped_child=wrapped_missing,
            )
    else:
        with torch.device("meta"):
            target = _DCPMetaLoadModule(
                100.0 + rank,
                include_cache=include_cache,
                include_wrapped_child=wrapped_missing,
            )

    initial_meta_split = has_meta_state(target) == (not already_prepared and rank != 0)
    parameter_ids = {name: id(parameter) for name, parameter in target.named_parameters()}
    initial_devices = {
        name: local_tensor(tensor).device for name, tensor in [*target.named_parameters(), *target.named_buffers()]
    }
    events = []
    before_load_materialized = []
    broadcast_selections = []
    tensor_payload_names = []
    original_materialize = checkpointer_module.materialize_meta_module
    original_to = target.to
    original_broadcast = checkpointer_module.broadcast_module
    original_payload_broadcast = distributed_module._sync_params_and_buffers
    load_owner = checkpointer_module.dcp if checkpoint_format == "dcp" else target
    load_method = "load" if checkpoint_format == "dcp" else "load_state_dict"
    original_load = getattr(load_owner, load_method)

    def record_materialize(module, *args, **kwargs):
        events.append("materialize")
        assert not already_prepared and rank != 0
        assert module is target
        original_materialize(module, *args, **kwargs)
        # Missing reset hooks leave unspecified storage; poison it to ensure loading and the
        # uncovered-state broadcast each restore their own fields, with no pre-load alignment.
        with torch.no_grad():
            for tensor in [*module.parameters(), *module.buffers()]:
                tensor.fill_(float("nan"))

    def record_to(*args, **kwargs):
        events.append("move")
        assert not already_prepared, "Loading an already-prepared FSDP root must not move its state"
        return original_to(*args, **kwargs)

    def inspect_then_load(*args, **kwargs):
        events.append("load_start")
        before_load_materialized.append(
            not has_meta_state(target)
            and all(
                local_tensor(tensor).device == (device if already_prepared else target_load_device)
                for tensor in [*target.parameters(), *target.buffers()]
            )
        )
        result = original_load(*args, **kwargs)
        events.append("load_end")
        return result

    def record_broadcast(module, *args, **kwargs):
        assert module is target
        assert not already_prepared, "Already-prepared FSDP roots must not broadcast initial state"
        assert "load_end" in events, "Uncovered state must be broadcast after loading covered tensors"
        assert kwargs.get("names") is not None, "Checkpoint loading must never broadcast the whole model"
        broadcast_selections.append(set(kwargs["names"]))
        events.append("broadcast")
        return original_broadcast(module, *args, **kwargs)

    def record_tensor_payload(group, tensors, *args, **kwargs):
        assert "load_end" in events, "No module tensor payload may be transferred before checkpoint loading"
        names_by_storage = {
            local_tensor(tensor).data_ptr(): name for _kind, name, tensor in module_state_tensors(target)
        }
        tensor_payload_names.append(
            tuple(sorted(names_by_storage.get(tensor.data_ptr(), "unknown") for tensor in tensors))
        )
        return original_payload_broadcast(group, tensors, *args, **kwargs)

    with (
        patch.object(checkpointer_module, "materialize_meta_module", record_materialize),
        patch.object(target, "to", record_to),
        patch.object(checkpointer_module, "broadcast_module", record_broadcast),
        patch.object(distributed_module, "_sync_params_and_buffers", record_tensor_payload),
        patch.object(load_owner, load_method, inspect_then_load),
    ):
        if partial or load_device == "cuda":
            iteration = checkpointer.load(
                torch.nn.ModuleDict({"net": target}), path=path, strict=not partial, device=target_load_device
            )
        else:
            # Native inference loads a newly constructed net before setup_inference_modules prepares it.
            iteration = load_checkpoint(checkpointer, SimpleNamespace(net=target, ema_dict={}), path)

    expected = {
        "weight": 11.0,
        "retained_weight": 102.0 if partial else 12.0,
        "scale": 13.0,
        "retained_scale": 104.0 if partial else 14.0,
    }
    if include_cache:
        expected["cache"] = 105.0
    if wrapped_missing:
        expected["wrapped._checkpoint_wrapped_module.weight"] = 106.0
        expected["wrapped._checkpoint_wrapped_module.scale"] = 107.0

    def values_match():
        registered = {name: tensor for _kind, name, tensor in module_state_tensors(target, remove_duplicate=False)}
        return all(
            torch.equal(local_tensor(registered[name]), torch.full_like(local_tensor(registered[name]), value))
            for name, value in expected.items()
        )

    expected_uncovered = ({"cache"} if include_cache else set()) | (
        {"retained_weight", "retained_scale"} if partial else set()
    )
    if wrapped_missing:
        expected_uncovered.update(
            {"wrapped._checkpoint_wrapped_module.weight", "wrapped._checkpoint_wrapped_module.scale"}
        )
    if already_prepared:
        expected_selections = []
        expected_payloads = []
    else:
        expected_selections = [expected_uncovered]
        expected_payloads = [tuple(sorted(expected_uncovered))] if expected_uncovered else []
    selection_matches = broadcast_selections == expected_selections or (
        not expected_uncovered and not broadcast_selections
    )
    checks = {
        "initial_meta_split": initial_meta_split,
        "iteration": iteration == 19,
        "materialized_and_placed_before_load": before_load_materialized == [True],
        "only_meta_rank_materialized": events.count("materialize") == int(not already_prepared and rank != 0),
        "broadcast_only_uncovered_names": selection_matches,
        "covered_tensors_never_broadcast": tensor_payload_names == expected_payloads,
        "loaded_and_uncovered_values": values_match(),
        "no_meta_left": not has_meta_state(target),
    }
    if already_prepared:
        checks["no_prepared_state_mutation"] = events == ["load_start", "load_end"]
        checks["prepared_policy_preserved"] = target.is_prepared and target._is_fsdp
        checks["dtensor_identity_preserved"] = all(
            isinstance(parameter, DTensor) and id(parameter) == parameter_ids[name]
            for name, parameter in target.named_parameters()
        )
        checks["placement_preserved"] = all(
            local_tensor(tensor).device == initial_devices[name]
            for name, tensor in [*target.named_parameters(), *target.named_buffers()]
        )
    else:
        checks["load_uses_requested_device"] = all(
            tensor.device == target_load_device for tensor in [*target.parameters(), *target.buffers()]
        )
        checks["not_yet_lifecycle_prepared"] = not target.is_prepared
        target.prepare_module(device, precision_model=torch.float32)
        checks["later_eager_preparation_preserves_values"] = target.is_prepared and values_match()

    if wrapped_missing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
        from torch.nn.parallel import DistributedDataParallel

        ddp_target = DistributedDataParallel(
            checkpoint_wrapper(torch.nn.Linear(2, 2)).to(device), device_ids=[device.index]
        )
        checks["ddp_wrapper_mapping_uses_raw_names"] = checkpointer_module._registration_names_for_dcp_keys(
            ddp_target, {"weight", "bias"}
        ) == {"module._checkpoint_wrapped_module.weight", "module._checkpoint_wrapped_module.bias"}

    flags = torch.tensor(list(checks.values()), dtype=torch.uint8)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    return dict(zip(checks, flags.bool().tolist()))


@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    ("scenario", "source_device", "load_device"),
    [
        ("full", "cpu", "cpu"),
        ("partial", "cpu", "cpu"),
        ("no_uncovered", "cpu", "cpu"),
        ("prepared_fsdp", "cuda", "cpu"),
        ("full", "cuda", "cpu"),
        ("partial", "cpu", "cuda"),
        ("partial_wrapped", "cpu", "cpu"),
    ],
    ids=[
        "full",
        "partial",
        "no-uncovered",
        "prepared_fsdp",
        "full-cuda-to-cpu",
        "partial-cpu-to-cuda",
        "partial-wrapped",
    ],
)
def test_dcp_load_preserves_meta_target_state(tmp_path, scenario, source_device, load_device):
    """Full/partial DCP loads preserve rank-0 defaults; prepared FSDP roots retain their placement."""
    result = run_distributed_test(
        test_fn=_checkpoint_load_preserves_meta_target_state_impl,
        world_size=2,
        timeout=180,
        nccl_timeout=60,
        checkpoint_dir=str(tmp_path),
        scenario=scenario,
        source_device=source_device,
        load_device=load_device,
    )
    assert result is not None
    assert all(result.values()), result


@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    ("scenario", "source_device", "load_device"),
    [
        ("full", "cpu", "cpu"),
        ("partial", "cpu", "cpu"),
        ("no_uncovered", "cpu", "cpu"),
        ("full", "cuda", "cpu"),
        ("partial", "cpu", "cuda"),
        ("partial_wrapped", "cpu", "cpu"),
    ],
    ids=[
        "full",
        "partial",
        "no-uncovered",
        "full-cuda-to-cpu",
        "partial-cpu-to-cuda",
        "partial-wrapped",
    ],
)
def test_monolithic_load_preserves_meta_target_state(tmp_path, scenario, source_device, load_device):
    result = run_distributed_test(
        test_fn=_checkpoint_load_preserves_meta_target_state_impl,
        world_size=2,
        timeout=180,
        nccl_timeout=60,
        checkpoint_dir=str(tmp_path),
        scenario=scenario,
        source_device=source_device,
        load_device=load_device,
        checkpoint_format="monolithic",
    )
    assert result is not None
    assert all(result.values()), result


class _LegacyLayoutNet(torch.nn.Module):
    """A checkpoint layout that loads only after renaming ``old_proj`` and expanding the half-height ``head``."""

    def __init__(self):
        super().__init__()
        self.keep = torch.nn.Linear(4, 4)
        self.old_proj = torch.nn.Linear(4, 6)
        self.head = torch.nn.Linear(6, 2, bias=False)


class _ExpandingLinear(torch.nn.Linear):
    """Repeats a checkpoint weight with half the output rows in its own load, like PDD's single-head expansion."""

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "weight"
        if key in state_dict and state_dict[key].shape[0] * 2 == self.out_features:
            state_dict[key] = state_dict[key].repeat(2, 1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class _ConvertingNet(torch.nn.Module):
    """Loads ``_LegacyLayoutNet`` checkpoints, renaming like Wan's, DiT's, and Cosmos' ``load_state_dict``
    overrides or in a load pre-hook, and expanding ``head`` in its own load."""

    def __init__(self, rename_in_hook: bool):
        super().__init__()
        self.keep = torch.nn.Linear(4, 4)
        self.proj = torch.nn.Linear(4, 6)
        self.head = _ExpandingLinear(6, 4, bias=False)
        self.rename_in_hook = rename_in_hook
        if rename_in_hook:
            self.register_load_state_dict_pre_hook(_ConvertingNet._rename_in_place)

    @staticmethod
    def _rename_in_place(module, state_dict, prefix, *args):
        for key in [key for key in state_dict if key.startswith(f"{prefix}old_proj.")]:
            state_dict[key.replace(f"{prefix}old_proj.", f"{prefix}proj.", 1)] = state_dict.pop(key)

    def load_state_dict(self, state_dict, *args, **kwargs):
        if not self.rename_in_hook:
            state_dict = {key.replace("old_proj.", "proj.", 1): value for key, value in state_dict.items()}
        return super().load_state_dict(state_dict, *args, **kwargs)


def _monolithic_load_conversion_impl(rank: int, world_size: int, checkpoint_dir: str) -> list:
    """Ordinary module loading handles key/shape conversions before the result is sharded."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    legacy = _LegacyLayoutNet()
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir))
    path = checkpointer.save(torch.nn.ModuleDict({"net": legacy}), iteration=6)
    expected = {
        **{f"keep.{key}": value for key, value in legacy.keep.state_dict().items()},
        **{f"proj.{key}": value for key, value in legacy.old_proj.state_dict().items()},
        "head.weight": legacy.head.weight.detach().repeat(2, 1),
    }
    mesh = init_device_mesh("cuda", (world_size,))
    checks = {}
    for rename_in_hook in (False, True):
        for sharded in (False, True):
            torch.manual_seed(1)
            target = _ConvertingNet(rename_in_hook)
            iteration = checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path)
            target.to(device)
            if sharded:
                fully_shard(target, mesh=mesh)
            # Every rank gathers the same tensors in the same order.
            state = {
                key: (value.full_tensor() if isinstance(value, DTensor) else value).cpu()
                for key, value in target.state_dict().items()
            }
            name = f"{'pre-hook' if rename_in_hook else 'override'}-{'fsdp' if sharded else 'unsharded'}"
            checks[name] = (
                iteration == 6
                and state.keys() == expected.keys()
                and all(torch.equal(state[key], value) for key, value in expected.items())
            )
    per_rank = [None] * world_size
    dist.all_gather_object(per_rank, checks)
    return per_rank


@RunIf(min_gpus=2)
def test_monolithic_load_before_fsdp_follows_network_load_conversions(tmp_path):
    result = run_distributed_test(
        _monolithic_load_conversion_impl, world_size=2, timeout=180, nccl_timeout=60, checkpoint_dir=str(tmp_path)
    )
    assert result is not None and all(all(checks.values()) for checks in result), result


class _LinearWithBuffers(FastGenModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.register_buffer("offset", torch.ones(4))
        self.register_buffer("cache", torch.full((2,), 7.0), persistent=False)

    def forward(self, x):
        return self.linear(x) + self.offset


def _monolithic_load_before_fsdp_training_impl(rank, world_size, checkpoint_dir, partial, offload):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import CPUOffloadPolicy
    from torch.distributed.tensor import DTensor

    from fastgen.utils.distributed.fsdp import FSDPContext

    source = _LinearWithBuffers()
    saved = source.state_dict()
    if partial:
        del saved["linear.bias"]
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir))
    path = checkpointer.resolve_save_path(None, 8)
    if rank == 0:
        # Old optimizer state is deliberately not part of a weights-only warm start.
        torch.save({"model": {"net": saved}, "optimizer": {"net": {"unused": True}}, "iteration": 8}, path)
    dist.barrier()
    torch.manual_seed(17)
    reference = _LinearWithBuffers()
    reference.load_state_dict(saved, strict=not partial)
    torch.manual_seed(17)
    with torch.device("meta" if rank else "cpu"):
        target = _LinearWithBuffers()
    assert checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path, strict=not partial) == 8
    torch.testing.assert_close(target.state_dict(), reference.state_dict())
    torch.testing.assert_close(target.cache, reference.cache)

    device = torch.device("cuda", torch.cuda.current_device())
    context = FSDPContext(
        init_device_mesh("cuda", (world_size,)),
        min_num_params=1,
        reduce_dtype=torch.float32,
        offload_policy=CPUOffloadPolicy() if offload else None,
        cpu_offload=offload,
    )
    target.prepare_module(
        device, precision_model=torch.float32, precision_opt=torch.float32, trainable=True, fsdp_context=context
    )
    reference.to(device)
    optimizer = torch.optim.Adam(target.parameters(), lr=0.01)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
    assert not optimizer.state
    inputs = torch.arange(6.0, device=device).reshape(2, 3)
    for net, opt in ((reference, reference_optimizer), (target, optimizer)):
        net(inputs).square().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    for actual, expected in zip(target.parameters(), reference.parameters()):
        assert isinstance(actual, DTensor)
        assert actual.to_local().device.type == ("cpu" if offload else "cuda")
        torch.testing.assert_close(actual.to(device).full_tensor(), expected)
        assert optimizer.state[actual]["step"].item() == 1
    return True


@RunIf(min_gpus=2)
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("offload", [False, True])
def test_monolithic_weights_load_before_fsdp_and_train_with_fresh_optimizer(tmp_path, partial, offload):
    assert run_distributed_test(
        _monolithic_load_before_fsdp_training_impl,
        world_size=2,
        timeout=180,
        nccl_timeout=45,
        checkpoint_dir=str(tmp_path),
        partial=partial,
        offload=offload,
    )


def _invalid_checkpoint_operations_impl(rank, world_size, checkpoint_dir, scenario):
    from unittest.mock import patch

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
    from torch.distributed.tensor import distribute_tensor, Shard

    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir))
    path = checkpointer.save(torch.nn.ModuleDict({"net": torch.nn.Linear(3, 4)}), iteration=8)
    if scenario == "missing_dcp_shards":
        if rank == 0:
            torch.save({"iteration": 8, "dcp_manifest": {"models": ("net",), "optimizers": ()}}, path)
        dist.barrier()
    target = torch.nn.Linear(3, 4).cuda()
    mesh = init_device_mesh("cuda", (world_size,))
    if scenario == "buffer":
        target.register_buffer("sharded_buffer", distribute_tensor(torch.ones(4, device="cuda"), mesh, [Shard(0)]))
    elif scenario != "one_rank" or rank == 1:
        fully_shard(target, mesh=mesh, **({"offload_policy": CPUOffloadPolicy()} if scenario == "offload" else {}))
    parameters = list(target.parameters())
    tensors = [*parameters, *target.buffers()]
    originals = [(t.to_local() if hasattr(t, "to_local") else t).detach().clone() for t in tensors]
    optimizers = {"net": torch.optim.Adam(parameters)} if scenario == "optimizer_only" else None
    model = torch.nn.ModuleDict({} if scenario == "optimizer_only" else {"net": target})
    reads = []

    def read(*args, **kwargs):
        reads.append(kwargs["device"])
        return load_training_checkpoint(*args, **kwargs)

    message = (
        r"\.metadata is missing"
        if scenario == "missing_dcp_shards"
        else "Monolithic checkpoints require unsharded model and optimizer state"
    )
    with patch("fastgen.utils.checkpointer.load_training_checkpoint", read):
        with pytest.raises(RuntimeError, match=message):
            checkpointer.load(model, optimizer_dict=optimizers, path=path, device="cuda", strict=False)
    assert reads == ["cuda"]
    if scenario != "missing_dcp_shards":
        with patch.object(checkpointer, "_save_checkpoint") as publish:
            with pytest.raises(RuntimeError, match=message):
                checkpointer.save(model, optimizer_dict=optimizers, path=path, iteration=9)
        publish.assert_not_called()
        assert load_training_checkpoint(path)["iteration"] == 8
    for tensor, original in zip([*target.parameters(), *target.buffers()], originals):
        torch.testing.assert_close(tensor.to_local() if hasattr(tensor, "to_local") else tensor, original)
    assert all(before is after for before, after in zip(parameters, target.parameters()))
    return True


@RunIf(min_gpus=2)
@pytest.mark.parametrize("scenario", ["model", "buffer", "offload", "optimizer_only", "one_rank", "missing_dcp_shards"])
def test_invalid_checkpoint_operations_preserve_state_on_every_rank(tmp_path, scenario):
    assert run_distributed_test(
        _invalid_checkpoint_operations_impl,
        world_size=2,
        timeout=120,
        nccl_timeout=30,
        checkpoint_dir=str(tmp_path),
        scenario=scenario,
    )


def test_monolithic_load_rejects_absent_roots_without_real_rank0_state(tmp_path):
    """An absent root keeps its state, so a meta root on rank 0 has no values for any rank to receive."""
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)))
    path = checkpointer.save(torch.nn.ModuleDict({"present": torch.nn.Linear(2, 2)}), iteration=1)
    present = torch.nn.Linear(2, 2)
    before = present.weight.detach().clone()
    with torch.device("meta"):
        absent = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="'absent' is absent from the checkpoint.*complete real state"):
        checkpointer.load(torch.nn.ModuleDict({"present": present, "absent": absent}), path=path)

    assert torch.equal(present.weight, before) and absent.weight.is_meta


def _checkpoint_meta_preparation_failure_impl(
    rank: int, world_size: int, failure: str, checkpoint_format: str = "dcp"
) -> dict:
    """A failure affecting one rank must stop every rank before restoring tensor state."""
    from unittest.mock import patch

    import fastgen.utils.checkpointer as checkpointer_module
    from fastgen.utils.module import has_meta_state

    class ResetFailureModule(_DCPMetaLoadModule):
        def reset_parameters(self):
            if rank == 1:
                raise RuntimeError("injected rank-local reset failure")

    alias_failure = failure in {"buffer_alias_real", "buffer_alias_meta", "parameter_buffer_alias"}
    should_be_meta = failure not in {"inference_state", "parameter_buffer_alias", "overwrite", "swap"} and rank == (
        0 if failure == "meta_source" else 1
    )
    target_class = ResetFailureModule if failure == "reset" else _DCPMetaLoadModule
    # Even without nonpersistent state, registrations must match before a tensor load can begin.
    include_cache = failure != "schema_mismatch"
    if failure == "inference_state" and rank == 1:
        with torch.inference_mode():
            target = target_class(100.0, include_cache=include_cache)
    else:
        with torch.device("meta" if should_be_meta else "cpu"):
            target = target_class(100.0, include_cache=include_cache)
            if failure == "schema_mismatch" and rank == 1:
                target.weight = torch.nn.Parameter(torch.full((9,), 101.0))
    if failure == "parameter_buffer_alias" and rank == 1:
        target.register_buffer("weight_as_buffer", target.weight)
    elif failure == "buffer_alias_meta" and rank == 1 or failure == "buffer_alias_real" and rank == 0:
        target.register_buffer("cache_alias", target.cache, persistent=False)
    # An invalid later target must prevent loading or materializing an earlier valid one too.
    with torch.device("meta" if should_be_meta else "cpu"):
        first_target = _DCPMetaLoadModule(200.0)
    first_parameters = list(first_target.parameters())
    targets = {"first": first_target, "net": target} if alias_failure else {"net": target}
    parameter_ids = [id(parameter) for parameter in target.parameters()]
    buffer_ids = [id(buffer) for buffer in target.buffers()]
    optimizers = {"net": torch.optim.SGD(target.parameters(), lr=0.1)} if failure == "optimizer" else None
    checkpointer = Checkpointer(_get_checkpointer_config("unused"))
    entered_load = []
    checkpoint_state = {"iteration": 19}
    if checkpoint_format == "monolithic":
        checkpoint_state["model"] = {name: {"weight": torch.ones(8)} for name in targets}
    else:
        checkpoint_state["dcp_manifest"] = {"models": tuple(targets), "optimizers": ()}

    def forbidden_load(*args, **kwargs):
        entered_load.append(True)
        raise AssertionError("A rank started restoring state despite an invalid meta target")

    with (
        patch.object(checkpointer_module, "path_exists", return_value=True),
        patch.object(checkpointer_module, "load_training_checkpoint", return_value=checkpoint_state),
        patch.object(checkpointer, "get_storage_reader", return_value=object(), create=True),
        patch.object(checkpointer_module.dcp, "load", forbidden_load),
        patch.object(target, "load_state_dict", forbidden_load),
        patch.object(first_target, "load_state_dict", forbidden_load),
        patch.object(
            torch.__future__,
            "get_overwrite_module_params_on_conversion",
            return_value=rank == 1 and failure == "overwrite",
        ),
        patch.object(
            torch.__future__, "get_swap_module_params_on_conversion", return_value=rank == 1 and failure == "swap"
        ),
    ):
        try:
            checkpointer.load(torch.nn.ModuleDict(targets), optimizer_dict=optimizers, path="unused/checkpoint.pth")
        except RuntimeError as error:
            message = str(error).lower()
            if alias_failure:
                alias_rank = 0 if failure == "buffer_alias_real" else 1
                aliased = "both a parameter and a buffer" in message or "multiple distinct buffer slots" in message
                rejected = aliased and f"rank {alias_rank}" in message
            elif failure == "inference_state":
                rejected = "inference tensor" in message and "rank 1" in message
            elif failure in {"overwrite", "swap"}:
                rejected = f"set_{failure}_module_params_on_conversion" in message and "rank 1" in message
            elif failure == "reset":
                rejected = "injected rank-local reset failure" in message and "rank 1" in message
            elif failure == "schema_mismatch":
                rejected = "'registrations'" in message and "rank 1" in message
            else:
                rejected = "meta" in message and (
                    "optimizer" in message if failure == "optimizer" else "rank 0" in message
                )
        else:
            rejected = False

    per_rank = [None] * world_size
    preflight_failure = alias_failure or failure in {"optimizer", "meta_source", "inference_state", "overwrite", "swap"}
    dist.all_gather_object(
        per_rank,
        {
            "rejected": rejected,
            "never_entered_tensor_load": not entered_load,
            "preflight_meta_state_unchanged": not preflight_failure or has_meta_state(target) == should_be_meta,
            "preflight_parameter_identity_unchanged": (
                not preflight_failure or [id(parameter) for parameter in target.parameters()] == parameter_ids
            ),
            "preflight_buffer_identity_unchanged": (
                not preflight_failure or [id(buffer) for buffer in target.buffers()] == buffer_ids
            ),
            "earlier_valid_target_unchanged": not alias_failure
            or (
                has_meta_state(first_target) == should_be_meta
                and all(before is after for before, after in zip(first_parameters, first_target.parameters()))
                and (should_be_meta or bool(torch.all(first_target.weight == 201.0)))
            ),
        },
    )
    return {"per_rank": per_rank}


@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    "failure",
    [
        "optimizer",
        "meta_source",
        "inference_state",
        "reset",
        "schema_mismatch",
        "buffer_alias_real",
        "buffer_alias_meta",
        "parameter_buffer_alias",
        "overwrite",
        "swap",
    ],
)
def test_dcp_meta_preparation_failure_is_collective(failure):
    result = run_distributed_test(
        test_fn=_checkpoint_meta_preparation_failure_impl,
        world_size=2,
        timeout=60,
        nccl_timeout=30,
        failure=failure,
    )
    assert result is not None
    assert all(all(entry.values()) for entry in result["per_rank"]), result


@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    "failure",
    [
        "optimizer",
        "meta_source",
        "inference_state",
        "reset",
        "schema_mismatch",
        "buffer_alias_real",
        "buffer_alias_meta",
        "parameter_buffer_alias",
        "overwrite",
        "swap",
    ],
)
def test_monolithic_meta_preparation_failure_is_collective(failure):
    result = run_distributed_test(
        test_fn=_checkpoint_meta_preparation_failure_impl,
        world_size=2,
        timeout=60,
        nccl_timeout=30,
        failure=failure,
        checkpoint_format="monolithic",
    )
    assert result is not None
    assert all(all(entry.values()) for entry in result["per_rank"]), result


# =============================================================================
# Checkpoint Plans and Module-Owner Preservation Tests
# =============================================================================


def _plan_root(value=1.0, name="weight"):
    module = torch.nn.Module()
    module.register_parameter(name, torch.nn.Parameter(torch.full((2,), value)))
    module.dropout = torch.nn.Dropout()
    return module


@pytest.mark.parametrize("strict", [False, True])
def test_monolithic_load_rejects_zero_matching_roots_before_mutation(tmp_path, strict):
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)))
    path = checkpointer.save(torch.nn.ModuleDict({"source": _plan_root()}), iteration=11)
    target = _plan_root(9.0)
    calls = []
    target.register_load_state_dict_post_hook(lambda *args: calls.append(True))
    with pytest.raises(RuntimeError, match="matched no target model roots"):
        checkpointer.load(torch.nn.ModuleDict({"target": target, "source": None}), path=path, strict=strict)
    torch.testing.assert_close(target.weight, torch.full((2,), 9.0))
    assert not calls

    # At least one matching root suffices; unmatched roots keep their constructor values.
    source = _plan_root(7.0)
    assert checkpointer.load(torch.nn.ModuleDict({"source": source, "target": target}), path=path, strict=strict) == 11
    torch.testing.assert_close(source.weight, torch.ones(2))
    torch.testing.assert_close(target.weight, torch.full((2,), 9.0))
    # Scalar-only restoration deliberately requests no model roots.
    assert checkpointer.load(torch.nn.ModuleDict(), path=path, strict=strict) == 11


@pytest.mark.parametrize("save_sharded", [False, True])
def test_none_entries_are_skipped_in_both_formats(tmp_path, save_sharded):
    """A ``None`` entry names a root, optimizer, or scheduler the caller does not hold; both formats skip it."""
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    source = _plan_root(3.0)
    optimizer = torch.optim.Adam(source.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    source.weight.sum().backward()
    optimizer.step()
    scheduler.step()
    models = torch.nn.ModuleDict({"source": source, "unused": None})
    path = checkpointer.save(
        models, {"source": optimizer, "unused": None}, {"source": scheduler, "unused": None}, iteration=11
    )

    target = _plan_root(9.0)
    target_optimizer = torch.optim.Adam(target.parameters())
    target_scheduler = torch.optim.lr_scheduler.LambdaLR(target_optimizer, lambda step: 1.0)
    models = torch.nn.ModuleDict({"source": target, "unused": None})
    optimizers = {"source": target_optimizer, "unused": None}
    assert checkpointer.load(models, optimizers, {"source": target_scheduler, "unused": None}, path=path) == 11
    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(
        target_optimizer.state[target.weight]["exp_avg"], optimizer.state[source.weight]["exp_avg"]
    )
    assert target_scheduler.last_epoch == scheduler.last_epoch == 1


@pytest.mark.parametrize("save_sharded", [False, True])
def test_only_none_roots_are_rejected_in_both_formats(tmp_path, save_sharded):
    """Only ``None`` roots would transfer no weights without an error; an empty dict requests scalar state alone."""
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    with pytest.raises(ValueError, match="every model root"):
        checkpointer.save(torch.nn.ModuleDict({"source": None}), iteration=12)
    assert not any(tmp_path.iterdir())

    path = checkpointer.save(torch.nn.ModuleDict({"source": _plan_root()}), iteration=12)
    with pytest.raises(ValueError, match="every model root"):
        checkpointer.load(torch.nn.ModuleDict({"source": None}), path=path)
    assert checkpointer.load(torch.nn.ModuleDict(), path=path) == 12

    # main's loader indexes these fields directly, so a save without the dicts stores them empty, not None.
    state = load_training_checkpoint(path)
    assert state["scheduler"] == {}
    assert "optimizer" not in state if save_sharded else state["optimizer"] == {}


@pytest.mark.parametrize("save_sharded", [False, True])
@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("mutation", ["share", "replace"])
def test_checkpoint_hooks_preserve_stateless_owners(tmp_path, save_sharded, operation, mutation):
    models = torch.nn.ModuleDict({"a": _plan_root(), "b": _plan_root()})
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    path = checkpointer.save(models, iteration=1)
    previous = Path(path).read_bytes()

    def mutate(*args):
        models["b"].dropout = models["a"].dropout if mutation == "share" else torch.nn.Dropout()

    if operation == "save":
        models["a"].register_state_dict_post_hook(mutate)
    else:
        models["a"].register_load_state_dict_post_hook(mutate)
    with pytest.raises(RuntimeError, match="module owners must be preserved"):
        getattr(checkpointer, operation)(models, path=path)
    assert Path(path).read_bytes() == previous


def _materialization_owner_mutation_worker(rank, world_size, checkpoint_dir, save_sharded):
    checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=save_sharded)
    path = checkpointer.save(torch.nn.ModuleDict({"a": _plan_root(), "b": _plan_root()}))
    with torch.device("meta" if rank else "cpu"):
        target = _plan_root()
    models = torch.nn.ModuleDict({"a": target, "b": _plan_root()})

    def mutate():
        models["b"].dropout = torch.nn.Dropout()

    target.reset_parameters = mutate
    with pytest.raises(RuntimeError, match="module owners must be preserved"):
        checkpointer.load(models, path=path)
    return True


@RunIf(min_gpus=2)
@pytest.mark.parametrize("save_sharded", [False, True])
def test_materialization_hook_cannot_replace_another_roots_stateless_owner(tmp_path, save_sharded):
    assert run_distributed_test(
        _materialization_owner_mutation_worker,
        world_size=2,
        timeout=120,
        nccl_timeout=30,
        checkpoint_dir=str(tmp_path),
        save_sharded=save_sharded,
    )


@pytest.mark.parametrize("save_sharded", [False, True])
def test_shared_stateless_owners_within_one_root_remain_valid(tmp_path, save_sharded):
    module = _plan_root()
    module.dropout_alias = module.dropout
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    path = checkpointer.save(torch.nn.ModuleDict({"net": module}), iteration=3)
    assert checkpointer.load(torch.nn.ModuleDict({"net": module}), path=path) == 3
    assert module.dropout_alias is module.dropout


@pytest.mark.parametrize("save_sharded", [False, True])
@pytest.mark.parametrize("hook", ["optimizer", "callbacks"])
def test_optimizer_and_callback_load_hooks_preserve_model_owners(tmp_path, save_sharded, hook):
    models = torch.nn.ModuleDict({"a": _plan_root(), "b": _plan_root()})
    optimizer = torch.optim.SGD(models["a"].parameters(), lr=0.1, momentum=0.9)
    models["a"].weight.sum().backward()
    optimizer.step()

    def mutate(*args):
        models["b"].dropout = torch.nn.Dropout()

    class Callbacks:
        def state_dict(self):
            return {"counter": 1}

        def load_state_dict(self, state):
            mutate()

    callbacks = Callbacks() if hook == "callbacks" else None
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    path = checkpointer.save(models, optimizer_dict={"a": optimizer}, callbacks=callbacks)
    if hook == "optimizer":
        optimizer.register_load_state_dict_post_hook(mutate)
    with pytest.raises(RuntimeError, match="module owners must be preserved"):
        checkpointer.load(models, optimizer_dict={"a": optimizer}, callbacks=callbacks, path=path)


def test_permissive_dcp_load_requires_a_parameter_match(tmp_path):
    source = _plan_root(name="source")
    target = _plan_root(name="target")
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=True)
    path = checkpointer.save(torch.nn.ModuleDict({"net": source}), iteration=11)
    original = target.target.detach().clone()
    calls = []
    target.register_load_state_dict_post_hook(lambda *args: calls.append(True))
    with pytest.raises(RuntimeError, match="matched no target parameters"):
        checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path, strict=False)
    torch.testing.assert_close(target.target, original)
    assert not calls


@pytest.mark.parametrize("save_sharded", [False, True])
def test_strict_loads_accept_parameter_free_roots(tmp_path, save_sharded):
    source = torch.nn.BatchNorm1d(2, affine=False)
    source.running_mean.fill_(3.0)
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=save_sharded)
    path = checkpointer.save(torch.nn.ModuleDict({"net": source}), iteration=6)
    target = torch.nn.BatchNorm1d(2, affine=False)

    assert checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path) == 6
    torch.testing.assert_close(target.running_mean, source.running_mean)


@pytest.mark.parametrize("unstepped_optimizer", [False, True])
def test_dcp_overwrite_manifest_excludes_previous_models_and_optimizers(tmp_path, unstepped_optimizer):
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=True)
    models = torch.nn.ModuleDict({"a": _plan_root(), "b": _plan_root(2.0)})
    optimizer = torch.optim.SGD(models["a"].parameters(), lr=0.1, momentum=0.9)
    models["a"].weight.sum().backward()
    optimizer.step()
    path = checkpointer.save(models, optimizer_dict={"a": optimizer}, iteration=1)
    with torch.no_grad():
        models["a"].weight.fill_(3)
    fresh_optimizer = torch.optim.SGD(models["a"].parameters(), lr=0.1, momentum=0.9)
    checkpointer.save(
        torch.nn.ModuleDict({"a": models["a"]}),
        optimizer_dict={"a": fresh_optimizer} if unstepped_optimizer else None,
        path=path,
        iteration=2,
    )
    assert Path(path.removesuffix(".pth") + ".b_model/.metadata").exists()
    assert Path(path.removesuffix(".pth") + ".a_optim/.metadata").exists()
    targets = torch.nn.ModuleDict({"a": _plan_root(8.0), "b": _plan_root(9.0)})
    with pytest.raises(FileNotFoundError, match="absent from the DCP shard manifest"):
        checkpointer.load(targets, path=path, strict=False)
    torch.testing.assert_close(targets["a"].weight, torch.full((2,), 8.0))
    target_optimizer = torch.optim.SGD(targets["a"].parameters(), lr=0.1, momentum=0.9)
    assert (
        checkpointer.load(torch.nn.ModuleDict({"a": targets["a"]}), optimizer_dict={"a": target_optimizer}, path=path)
        == 2
    )
    torch.testing.assert_close(targets["a"].weight, torch.full((2,), 3.0))
    assert not target_optimizer.state


def test_dcp_listed_optimizer_shard_must_exist_before_loading(tmp_path):
    source = _plan_root()
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    source.weight.sum().backward()
    optimizer.step()
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=True)
    path = checkpointer.save(torch.nn.ModuleDict({"net": source}), optimizer_dict={"net": optimizer})
    Path(path.removesuffix(".pth") + ".net_optim/.metadata").unlink()
    target = _plan_root(7.0)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    with pytest.raises(FileNotFoundError, match="listed in the DCP manifest are missing"):
        checkpointer.load(torch.nn.ModuleDict({"net": target}), optimizer_dict={"net": target_optimizer}, path=path)
    torch.testing.assert_close(target.weight, torch.full((2,), 7.0))


@pytest.mark.parametrize("optimizer_shard", [True, False])
def test_legacy_dcp_marker_without_manifest_loads_like_main(tmp_path, optimizer_shard):
    source = _plan_root()
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    source.weight.sum().backward()
    optimizer.step()
    checkpointer = Checkpointer(_get_checkpointer_config(str(tmp_path)), save_sharded=True)
    path = checkpointer.save(torch.nn.ModuleDict({"net": source}), optimizer_dict={"net": optimizer}, iteration=4)
    # Markers saved by ``main`` carry no shard manifest.
    marker = torch.load(path, weights_only=False)
    del marker["dcp_manifest"]
    torch.save(marker, path)
    if not optimizer_shard:
        Path(path.removesuffix(".pth") + ".net_optim/.metadata").unlink()
    target = _plan_root(7.0)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)

    loaded = checkpointer.load(
        torch.nn.ModuleDict({"net": target}), optimizer_dict={"net": target_optimizer}, path=path
    )

    assert loaded == 4
    torch.testing.assert_close(target.weight, source.weight)
    if optimizer_shard:
        torch.testing.assert_close(
            target_optimizer.state[target.weight]["momentum_buffer"], optimizer.state[source.weight]["momentum_buffer"]
        )
    else:
        assert not target_optimizer.state
    # As in older versions, every requested model still needs its shards.
    with pytest.raises(FileNotFoundError, match="metadata is missing"):
        checkpointer.load(torch.nn.ModuleDict({"net": _plan_root(), "extra": _plan_root()}), path=path)


def _checkpoint_plan_worker(rank, world_size, checkpoint_dir):
    from unittest.mock import patch

    import fastgen.utils.checkpointer as checkpointing

    config = _get_checkpointer_config(checkpoint_dir)
    saver = Checkpointer(config)
    sources = torch.nn.ModuleDict({"a": _plan_root(), "b": _plan_root(2.0)})
    first = saver.save(sources, iteration=11)
    second = saver.save(sources, iteration=12)
    before_marker = Path(first).read_bytes()
    results = {}
    for operation in ("save", "load"):
        differences = ["path", "model_order", "model_count", "optimizer_order", "optimizer_count"]
        if operation == "save":
            differences.append("format")
        else:
            # Every rank loads the format that rank 0 read, so it cannot differ between ranks.
            differences += ["strict", "no_checkpoint"]
        for difference in differences:
            checkpointer = Checkpointer(config)
            models = torch.nn.ModuleDict({"a": _plan_root(7.0), "b": _plan_root(7.0)})
            optimizers = {name: torch.optim.SGD(module.parameters(), lr=0.1) for name, module in models.items()}
            path, strict = first, True
            if rank == 1:
                if difference == "path":
                    path = second
                elif difference == "model_order":
                    models = torch.nn.ModuleDict(reversed(list(models.items())))
                elif difference == "model_count":
                    del models["b"]
                elif difference == "optimizer_order":
                    optimizers = dict(reversed(list(optimizers.items())))
                elif difference == "optimizer_count":
                    del optimizers["b"]
                elif difference == "strict":
                    strict = False
                elif difference == "no_checkpoint":
                    path = ""
                elif difference == "format":
                    checkpointer.save_sharded = True

            with contextlib.ExitStack() as stack:
                writes = stack.enter_context(patch.object(checkpointer, "_save_checkpoint"))
                shard_writes = stack.enter_context(patch.object(checkpointing.dcp, "save"))
                try:
                    if operation == "save":
                        checkpointer.save(models, optimizer_dict=optimizers, path=path)
                    else:
                        checkpointer.load(models, optimizer_dict=optimizers, path=path, strict=strict)
                except RuntimeError as error:
                    rejected = f"checkpoint {operation} plan mismatch" in str(error)
                else:
                    rejected = False
                checks = [None] * world_size
                dist.all_gather_object(
                    checks,
                    rejected
                    and not writes.called
                    and not shard_writes.called
                    and all(bool((m.weight == 7).all()) for m in models.values())
                    and Path(first).read_bytes() == before_marker,
                )
                results[f"{operation}/{difference}"] = checks

    alias = Path(checkpoint_dir) / "alias.pth"
    with checkpointing.raise_across_ranks("test checkpoint symlink"):
        if rank == 0:
            alias.symlink_to(first)
    targets = torch.nn.ModuleDict({"a": _plan_root(), "b": _plan_root()})
    iteration = saver.load(targets, path=first if rank == 0 else str(alias))
    checks = [None] * world_size
    dist.all_gather_object(checks, iteration == 11 and bool((targets["b"].weight == 2).all()))
    results["canonical_path"] = checks
    return results


@RunIf(min_gpus=2)
def test_checkpoint_plans_agree_before_loading_or_writing(tmp_path):
    results = run_distributed_test(
        _checkpoint_plan_worker, world_size=2, timeout=180, nccl_timeout=30, checkpoint_dir=str(tmp_path)
    )
    assert results is not None and all(all(checks) for checks in results.values()), results


# =============================================================================
# Module Topology Agreement and Root Isolation Tests
# =============================================================================


def _topology_root(value=1.0, difference="none", dtype=torch.float32):
    module = FastGenModule()
    module.weight = torch.nn.Parameter(torch.full((3 if difference == "shape" else 2,), value, dtype=dtype))
    module.weight_alias = module.weight
    module.register_buffer("cache", torch.full((2,), value, dtype=dtype))
    module.left = torch.nn.Dropout() if difference != "type" else torch.nn.Identity()
    module.right = torch.nn.Dropout() if difference == "shared_owner" else module.left
    if difference == "path":
        module.extra = torch.nn.Identity()
    return module


def _eager_topology_worker(rank, world_size):
    from unittest.mock import patch

    results = {}
    for placement in (None, "cpu", "cuda"):
        # Meta state needs a placement target to materialize on.
        for state in ("real",) if placement is None else ("real", "meta"):
            for difference in ("none", "shape", "path", "type", "shared_owner"):
                local_difference = difference if rank else "none"
                if state == "meta" and rank:
                    with torch.device("meta"):
                        module = _topology_root(difference=local_difference)
                else:
                    # Initial dtype differences are valid when the requested cast resolves them.
                    module = _topology_root(rank + 1.0, local_difference, torch.float32 if rank else torch.float64)
                device = torch.device("cuda", rank) if placement == "cuda" else placement
                with patch.object(module, "to", wraps=module.to) as move:
                    try:
                        module.prepare_module(device, precision_model=torch.float32, trainable=True)
                    except RuntimeError as error:
                        # Only meta materialization compares topology, after casting and before placement.
                        expected = "'registrations'" if difference == "shape" else "'topology'"
                        passed = state == "meta" and difference != "none" and expected in str(error)
                        passed &= not move.called and not module.is_prepared
                        passed &= module.weight.device.type == ("meta" if state == "meta" and rank else "cpu")
                    else:
                        # Real eager modules are rank-local and keep their values; meta ranks receive rank 0's.
                        passed = (state == "real" or difference == "none") and module.is_prepared
                        passed &= module.weight.device.type == (placement or "cpu")
                        passed &= bool((module.weight == (rank + 1 if state == "real" else 1)).all())
                        passed &= module.weight_alias is module.weight
                        passed &= (module.right is module.left) == (local_difference != "shared_owner")
                    passed &= module.weight.dtype == torch.float32
                checks = [None] * world_size
                dist.all_gather_object(checks, passed)
                results[f"{placement}/{state}/{difference}"] = checks
    return results


@RunIf(min_gpus=2)
def test_eager_preparation_validates_topology_only_for_meta_materialization():
    results = run_distributed_test(_eager_topology_worker, world_size=2, timeout=180, nccl_timeout=30)
    assert results is not None and all(all(checks) for checks in results.values()), results


def _real_checkpoint_topology_worker(rank, world_size, checkpoint_dir):
    results = {}
    for save_sharded in (False, True):
        checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=save_sharded)
        path = checkpointer.save(torch.nn.ModuleDict({"net": _topology_root(3.0)}), iteration=7)
        for difference in ("none", "shape", "path", "type", "shared_owner"):
            target = _topology_root(9.0, difference if rank else "none")
            loaded = []
            target.register_load_state_dict_post_hook(lambda *args: loaded.append(True))
            try:
                iteration = checkpointer.load(torch.nn.ModuleDict({"net": target}), path=path)
            except RuntimeError as error:
                expected = "'registrations'" if difference == "shape" else "'topology'"
                passed = difference != "none" and expected in str(error)
                passed &= not loaded and bool((target.weight == 9).all())
            else:
                passed = difference == "none" and iteration == 7 and bool(loaded)
                passed &= bool((target.weight == 3).all()) and target.weight_alias is target.weight
                passed &= target.right is target.left
            checks = [None] * world_size
            dist.all_gather_object(checks, passed)
            results[f"{save_sharded}/{difference}"] = checks
    return results


@RunIf(min_gpus=2)
def test_real_checkpoint_targets_validate_stateless_topology_before_loading(tmp_path):
    results = run_distributed_test(
        _real_checkpoint_topology_worker, world_size=2, timeout=180, nccl_timeout=30, checkpoint_dir=str(tmp_path)
    )
    assert results is not None and all(all(checks) for checks in results.values()), results


def _check_materialization_preserves_other_roots(rank, checkpointer, victim_position, kind):
    order = ("victim", "net") if victim_position == "before" else ("net", "victim")
    path = checkpointer.save(torch.nn.ModuleDict({name: _topology_root() for name in order}))
    with torch.device("meta" if rank else "cpu"):
        target = _topology_root()
    with torch.device("meta" if rank and victim_position == "pending_meta" else "cpu"):
        victim = _topology_root()
    models = torch.nn.ModuleDict({name: target if name == "net" else victim for name in order})

    def mutate():
        if kind == "parameter":
            # Keep the victim's tensor alias topology unchanged; replacing its identity is enough.
            victim.weight = torch.nn.Parameter(torch.empty_like(victim.weight))
            victim.weight_alias = victim.weight
        else:
            victim.cache = torch.empty_like(victim.cache)

    target.reset_parameters = mutate
    loaded = []
    target.register_load_state_dict_post_hook(lambda *args: loaded.append(True))
    with pytest.raises(RuntimeError, match="tensor identities.*must be preserved"):
        checkpointer.load(models, path=path)
    assert not loaded


def _check_materialization_rejects_own_parameter_replacement(rank, checkpointer):
    path = checkpointer.save(
        torch.nn.ModuleDict({"net": _topology_root(3.0), "other": _topology_root(4.0)}), iteration=8
    )
    with torch.device("meta" if rank else "cpu"):
        target = _topology_root()
    other = _topology_root()
    before = other.weight, other.cache

    def reset():
        target.weight = torch.nn.Parameter(torch.ones_like(target.weight))
        target.weight_alias = target.weight

    target.reset_parameters = reset
    with pytest.raises(RuntimeError, match="changed registered state"):
        checkpointer.load(torch.nn.ModuleDict({"net": target, "other": other}), path=path)
    assert other.weight is before[0] and other.cache is before[1]
    assert bool((other.weight == 1).all())


def _check_load_hook_replacement_after_materialization(rank, checkpointer):
    path = checkpointer.save(torch.nn.ModuleDict({"a": _topology_root(3.0), "b": _topology_root(4.0)}))
    with torch.device("meta" if rank else "cpu"):
        target = _topology_root()
    models = torch.nn.ModuleDict({"a": _topology_root(), "b": target})
    prepared_before_copy = []

    def mutate(*args):
        prepared_before_copy.append(not target.weight.is_meta)
        target.weight = torch.nn.Parameter(target.weight.detach().clone())
        target.weight_alias = target.weight

    models["a"].register_load_state_dict_post_hook(mutate)
    with pytest.raises(RuntimeError, match="changed registered state"):
        checkpointer.load(models, path=path)
    assert prepared_before_copy == [True]


def _materialization_isolation_worker(rank, world_size, checkpoint_dir):
    for save_sharded in (False, True):
        checkpointer = Checkpointer(_get_checkpointer_config(checkpoint_dir), save_sharded=save_sharded)
        for victim_position in ("before", "after", "pending_meta"):
            for kind in ("parameter", "buffer"):
                _check_materialization_preserves_other_roots(rank, checkpointer, victim_position, kind)
        _check_materialization_rejects_own_parameter_replacement(rank, checkpointer)
        _check_load_hook_replacement_after_materialization(rank, checkpointer)
    return True


@RunIf(min_gpus=2)
def test_materialization_and_load_hook_replacements_fail_collectively(tmp_path):
    assert run_distributed_test(
        _materialization_isolation_worker, world_size=2, timeout=180, nccl_timeout=30, checkpoint_dir=str(tmp_path)
    )
