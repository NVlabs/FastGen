# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for script packaging and shared command-line setup."""

import argparse
import contextlib
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastgen.configs.config import BaseConfig
from fastgen.utils import checkpointer, scripts
from fastgen.utils.cache_utils import remove_atomic_write_orphans
from scripts import fsdp_convert
from tests.test_configs import REPO_ROOT, build_config


def _loaded_config() -> BaseConfig:
    """A config as ``load_config`` returns it, named after its file."""
    config = BaseConfig()
    config.log_config.name = "config_probe"
    return config


def test_parse_args_takes_hydra_overrides_after_a_single_dash(monkeypatch):
    parser = argparse.ArgumentParser()
    monkeypatch.setattr("sys.argv", ["train.py", "--config", "config.py", "-", "trainer.max_iter=3"])

    args = scripts.parse_args(parser)

    # ``override_config_with_opts`` requires the leading '-' that the help documents.
    assert args.config == "config.py" and args.opts == ["-", "trainer.max_iter=3"]
    help_text = next(action.help for action in parser._actions if action.dest == "opts")
    assert "after a single '-'" in help_text and "Yacs" not in help_text


@pytest.mark.parametrize("torch_compile_enabled", [False, True])
@pytest.mark.parametrize(("tf32_matmul", "tf32_conv", "fp16_reduction"), [(False, True, True), (True, False, False)])
def test_set_cuda_backend_uses_one_tf32_api_family(
    monkeypatch, torch_compile_enabled, tf32_matmul, tf32_conv, fp16_reduction
):
    backends = SimpleNamespace(
        cudnn=SimpleNamespace(conv=SimpleNamespace()),
        cuda=SimpleNamespace(matmul=SimpleNamespace()),
    )
    monkeypatch.setattr(scripts.torch, "backends", backends)
    monkeypatch.setattr(scripts.logger, "critical", lambda *_args, **_kwargs: None)

    scripts.set_cuda_backend(
        deterministic=True,
        benchmark=False,
        tf32_matmul=tf32_matmul,
        tf32_conv=tf32_conv,
        fp16_reduced_precision_reduction=fp16_reduction,
        torch_compile_enabled=torch_compile_enabled,
    )

    assert backends.cudnn.deterministic is True
    assert backends.cudnn.benchmark is False
    assert backends.cuda.matmul.allow_fp16_reduced_precision_reduction is fp16_reduction
    if torch_compile_enabled:
        assert backends.cudnn.allow_tf32 is tf32_conv
        assert backends.cuda.matmul.allow_tf32 is tf32_matmul
        assert not hasattr(backends.cudnn.conv, "fp32_precision")
        assert not hasattr(backends.cuda.matmul, "fp32_precision")
    else:
        precision = {True: "tf32", False: "ieee"}
        assert backends.cudnn.conv.fp32_precision == precision[tf32_conv]
        assert backends.cuda.matmul.fp32_precision == precision[tf32_matmul]
        assert not hasattr(backends.cudnn, "allow_tf32")
        assert not hasattr(backends.cuda.matmul, "allow_tf32")


def test_tf32_defaults_are_pytorch_defaults():
    """IEEE FP32 matmuls, TF32 convolutions and fp16 reduced-precision reductions, in the config and the setter."""
    trainer = BaseConfig().trainer
    names = ("tf32_matmul", "tf32_conv", "fp16_reduced_precision_reduction")
    parameters = inspect.signature(scripts.set_cuda_backend).parameters
    assert [getattr(trainer, name) for name in names] == [False, True, True]
    assert [parameters[name].default for name in names] == [False, True, True]


@pytest.mark.parametrize(
    ("save_dir", "expected"),
    [
        ("checkpoints", "s3://bucket/fastgen/proj/grp/run/checkpoints"),
        ("s3://other-bucket/run", "s3://other-bucket/run"),
        ("/shared/checkpoints", "/shared/checkpoints"),
    ],
    ids=["relative", "s3-uri", "absolute"],
)
def test_setup_resolves_s3_save_dir_to_a_per_run_uri(monkeypatch, save_dir, expected):
    config = _loaded_config()
    config.trainer.checkpointer.use_s3 = True
    config.trainer.checkpointer.s3_container = "s3://bucket/fastgen"
    config.trainer.checkpointer.save_dir = save_dir
    config.log_config.project, config.log_config.group, config.log_config.name = "proj", "grp", "run"

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: False)
    monkeypatch.setattr(scripts, "is_rank0", lambda: False)
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)

    result = scripts.setup(argparse.Namespace(config="unused.py"))

    # The checkpointer uses ``save_dir`` as given. The run triple keeps two default-config S3 runs
    # from sharing or resuming each other's checkpoints; explicit locations are kept.
    assert result.trainer.checkpointer.save_dir == expected


@pytest.mark.parametrize("use_s3", [False, True])
@pytest.mark.parametrize("save_dir", ["/shared/checkpoints", "s3://bucket/explicit/checkpoints"])
def test_explicit_checkpoint_save_dir_is_independent_of_storage_default(use_s3, save_dir):
    config = _loaded_config()
    config.trainer.checkpointer.use_s3 = use_s3
    config.trainer.checkpointer.save_dir = save_dir
    assert scripts.resolve_save_dir(config) == save_dir


def test_relative_local_checkpoint_save_dir_is_below_the_run_directory():
    config = _loaded_config()
    config.trainer.checkpointer.use_s3 = False
    config.trainer.checkpointer.save_dir = "checkpoints"
    assert scripts.resolve_save_dir(config) == str(Path(config.log_config.save_path) / "checkpoints")


@pytest.mark.parametrize(
    ("opts", "expected"),
    [(None, "config_dmd2_fast"), (["-", "log_config.name=named_by_cli"], "named_by_cli")],
    ids=["config-file-name", "cli-override"],
)
def test_setup_names_runs_after_the_config_file_unless_the_cli_names_them(monkeypatch, opts, expected):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: False)
    monkeypatch.setattr(scripts, "is_rank0", lambda: False)
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)

    args = argparse.Namespace(config="fastgen/configs/experiments/EDM/config_dmd2_fast.py", opts=opts)
    result = scripts.setup(args)

    assert result.log_config.name == expected
    # The run directory and its checkpoints follow the resolved name.
    assert result.trainer.checkpointer.save_dir == os.path.join(result.log_config.save_path, "checkpoints")
    assert os.path.basename(result.log_config.save_path) == expected


@pytest.mark.parametrize(("own_name", "expected"), [(None, "config_probe"), ("own_name", "own_name")])
def test_load_config_names_an_unnamed_config_after_its_file(monkeypatch, own_name, expected):
    config = BaseConfig()
    config.log_config.name = own_name
    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)

    assert scripts.load_config("configs/config_probe.py").log_config.name == expected


def test_load_config_resolves_the_config_without_overrides(monkeypatch):
    """Every entry point gets the config as Hydra resolves it, whether or not it passes overrides."""
    config = _loaded_config()
    config.log_config.group = "${log_config.project}"
    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)

    assert scripts.load_config("config_probe.py").log_config.group == config.log_config.project


@pytest.mark.parametrize("fsdp_enabled", [False, True], ids=["ddp", "fsdp"])
@pytest.mark.parametrize("checkpoint_format", ["monolithic", "sharded", "unavailable"])
def test_setup_preserves_meta_init_without_inspecting_checkpoint(
    monkeypatch, tmp_path, fsdp_enabled, checkpoint_format
):
    config = _loaded_config()
    config.model.meta_init = True
    config.trainer.fsdp = fsdp_enabled
    config.trainer.ddp = not fsdp_enabled
    marker_path = str(tmp_path / "0000042.pth")
    config.trainer.checkpointer.pretrained_ckpt_path = marker_path
    config.trainer.checkpointer.pretrained_ckpt_key_map = {"net": "saved_net"}
    events = []
    if checkpoint_format != "unavailable":
        state = {"iteration": 42}
        if checkpoint_format == "monolithic":
            state["model"] = {"saved_net": {"weight": scripts.torch.ones(1)}}
        else:
            shard_dir = tmp_path / "0000042.saved_net_model"
            shard_dir.mkdir()
            (shard_dir / ".metadata").write_bytes(b"metadata")
        scripts.torch.save(state, marker_path)

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: True)
    monkeypatch.setattr(scripts, "world_size", lambda: 4)
    monkeypatch.setattr(scripts, "is_rank0", lambda: False)
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: events.append("credentials"))
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)

    def unexpected_checkpoint_io(*args, **kwargs):
        pytest.fail("setup must not inspect or load a checkpoint")

    monkeypatch.setattr(scripts.torch, "load", unexpected_checkpoint_io)
    monkeypatch.setattr(checkpointer, "path_exists", unexpected_checkpoint_io)
    monkeypatch.setattr(scripts.s3, "exists", unexpected_checkpoint_io)
    monkeypatch.setattr(scripts.s3, "load", unexpected_checkpoint_io)
    result = scripts.setup(argparse.Namespace(config="unused.py"))
    assert result.model.meta_init is True
    assert result.trainer.checkpointer.pretrained_ckpt_path == marker_path
    assert events == ["credentials"]


@pytest.mark.parametrize("path", ["/checkpoints/0000042", "s3://bucket/run/0000042"])
def test_setup_leaves_pretrained_checkpoint_validation_to_consumers(monkeypatch, path):
    config = _loaded_config()
    config.trainer.checkpointer.pretrained_ckpt_path = path
    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: False)
    monkeypatch.setattr(scripts, "is_rank0", lambda: False)
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)

    result = scripts.setup(argparse.Namespace(config="unused.py"))

    assert result.trainer.checkpointer.pretrained_ckpt_path == path


def test_setup_dryrun_renders_config_without_writing(monkeypatch):
    config = _loaded_config()
    calls = []
    logs = []

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(
        scripts,
        "serialize_config",
        lambda config, **kwargs: calls.append(kwargs) or "resolved: true\n",
    )
    monkeypatch.setattr(scripts.logger, "info", logs.append)
    monkeypatch.setattr(scripts.ddp, "init", lambda: pytest.fail("dryrun initialized distributed state"))

    with pytest.raises(SystemExit) as exc_info:
        scripts.setup(argparse.Namespace(config="unused.py", dryrun=True))

    assert exc_info.value.code == 0
    assert calls == [{"return_type": "yaml"}]
    assert logs[-1] == "resolved: true\n"


def test_setup_serializes_config_after_distributed_init(monkeypatch):
    config = _loaded_config()
    config.trainer.ddp = True
    events = []

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: events.append(("ddp.init", required)) or True)
    monkeypatch.setattr(scripts, "is_rank0", lambda: events.append("is_rank0") or True)
    monkeypatch.setattr(
        scripts,
        "serialize_config",
        lambda *args, **kwargs: events.append("serialize_config"),
    )
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)

    scripts.setup(argparse.Namespace(config="unused.py"))

    assert events == [("ddp.init", True), "is_rank0", "serialize_config"]


def test_setup_writes_the_config_inside_a_cross_rank_guard(monkeypatch):
    config = _loaded_config()
    events = []

    @contextlib.contextmanager
    def record_guard(operation):
        events.append(("enter", operation))
        yield
        events.append(("exit", operation))

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: False)
    monkeypatch.setattr(scripts, "raise_across_ranks", record_guard)
    monkeypatch.setattr(scripts, "is_rank0", lambda: True)
    monkeypatch.setattr(scripts, "serialize_config", lambda *args, **kwargs: events.append("serialize_config"))
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)

    scripts.setup(argparse.Namespace(config="unused.py"))

    # A rank-0-only write failure must reach the peers instead of leaving them in their next collective.
    assert events == [("enter", "config save"), "serialize_config", ("exit", "config save")]


@pytest.mark.parametrize(("compile_mode", "expected"), [(None, False), ("default", True)])
def test_setup_selects_compile_compatible_tf32_api(monkeypatch, compile_mode, expected):
    config = _loaded_config()
    config.model.torch_compile_mode = compile_mode
    calls = []

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: False)
    monkeypatch.setattr(scripts, "is_rank0", lambda: False)
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: calls.append((args, kwargs)))

    scripts.setup(argparse.Namespace(config="unused.py"))

    assert len(calls) == 1
    assert calls[0][1] == {"torch_compile_enabled": expected}


@pytest.mark.parametrize("evaluation", [False, True])
def test_setup_probes_optional_distributed_without_trainer_flags(monkeypatch, evaluation):
    config = _loaded_config()
    calls = []

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: calls.append(required) or True)
    monkeypatch.setattr(scripts, "is_rank0", lambda: False)
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)

    scripts.setup(argparse.Namespace(config="unused.py"), evaluation=evaluation)

    assert calls == [False]


@pytest.mark.parametrize(
    ("ddp_enabled", "evaluation", "required"),
    [
        (True, False, True),
        (False, True, False),
    ],
)
def test_setup_keeps_meta_init_without_fsdp_when_distributed(monkeypatch, ddp_enabled, evaluation, required):
    config = _loaded_config()
    config.model.meta_init = True
    config.trainer.ddp = ddp_enabled
    calls = []

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: calls.append(required) or True)
    monkeypatch.setattr(scripts, "world_size", lambda: 4)
    monkeypatch.setattr(scripts, "is_rank0", lambda: False)
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)

    result = scripts.setup(argparse.Namespace(config="unused.py"), evaluation=evaluation)

    assert calls == [required]
    assert result.trainer.fsdp is False
    assert result.model.meta_init is True


def test_setup_evaluation_allows_single_process_with_inherited_fsdp(monkeypatch):
    config = _loaded_config()
    config.trainer.fsdp = True
    config.model.meta_init = True
    calls = []
    logs = []

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: calls.append(required) or False)
    monkeypatch.setattr(scripts.logger, "info", logs.append)
    monkeypatch.setattr(scripts, "is_rank0", lambda: False)
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)

    result = scripts.setup(argparse.Namespace(config="unused.py"), evaluation=True)

    assert calls == [False]
    assert result.model.meta_init is False
    assert (
        "Distributed process group initialized=False, world size=1, " "DDP configured=False, FSDP configured=False."
    ) in logs


@pytest.mark.parametrize(
    ("evaluation", "ddp_enabled", "save_adapter_only", "error", "match"),
    [
        (False, True, False, ValueError, "mutually exclusive"),
        (True, True, False, ValueError, "mutually exclusive"),
        (False, False, True, NotImplementedError, "save_adapter_only"),
    ],
    ids=["ddp-and-fsdp", "ddp-and-fsdp-evaluation", "fsdp-adapter-only"],
)
def test_setup_rejects_invalid_parallelism_before_joining_the_process_group(
    monkeypatch, evaluation, ddp_enabled, save_adapter_only, error, match
):
    config = _loaded_config()
    config.trainer.ddp = ddp_enabled
    config.trainer.fsdp = True
    config.trainer.checkpointer.save_adapter_only = save_adapter_only

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: pytest.fail("setup joined the process group"))

    with pytest.raises(error, match=match):
        scripts.setup(argparse.Namespace(config="unused.py"), evaluation=evaluation)


def test_setup_refuses_multi_rank_training_without_ddp_or_fsdp(monkeypatch):
    config = _loaded_config()

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: True)
    monkeypatch.setattr(scripts, "world_size", lambda: 4)
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)
    monkeypatch.setattr(scripts, "serialize_config", lambda *args, **kwargs: pytest.fail("setup saved the config"))

    with pytest.raises(ValueError, match="independent replica"):
        scripts.setup(argparse.Namespace(config="unused.py"))


@pytest.mark.parametrize("supports_fsdp", [False, True])
@pytest.mark.parametrize(("ddp_enabled", "fsdp_enabled"), [(False, False), (True, False), (False, True)])
def test_setup_drops_evaluation_fsdp_unless_the_entry_point_shards(
    monkeypatch, ddp_enabled, fsdp_enabled, supports_fsdp
):
    config = _loaded_config()
    config.trainer.ddp = ddp_enabled
    config.trainer.fsdp = fsdp_enabled
    warnings_logged = []
    calls = []

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda path: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: calls.append(required) or True)
    monkeypatch.setattr(scripts.logger, "warning", warnings_logged.append)
    monkeypatch.setattr(scripts, "world_size", lambda: 4)
    monkeypatch.setattr(scripts, "is_rank0", lambda: False)
    monkeypatch.setattr(scripts, "set_env_vars", lambda path: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)

    result = scripts.setup(argparse.Namespace(config="unused.py"), evaluation=True, supports_fsdp=supports_fsdp)

    assert (result.trainer.ddp, result.trainer.fsdp) == (ddp_enabled, fsdp_enabled and supports_fsdp)
    assert calls == [fsdp_enabled and supports_fsdp]
    assert any("trainer.fsdp" in message for message in warnings_logged) == (fsdp_enabled and not supports_fsdp)


def test_setup_evaluation_enables_preprocessors(monkeypatch):
    config = build_config("fastgen/configs/experiments/SDXL/config_dmd2.py")
    assert not config.model.enable_preprocessors

    monkeypatch.setattr(scripts.config_utils, "import_config_from_python_file", lambda _: config)
    monkeypatch.setattr(scripts.ddp, "init", lambda required=False: False)
    monkeypatch.setattr(scripts, "serialize_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(scripts, "set_env_vars", lambda *args, **kwargs: None)
    monkeypatch.setattr(scripts, "set_cuda_backend", lambda *args, **kwargs: None)
    args = argparse.Namespace(config="unused.py", opts=None, log_level="INFO", dryrun=False)

    assert scripts.setup(args, evaluation=True).model.enable_preprocessors


def test_fsdp_conversion_failure_does_not_publish_a_checkpoint_and_can_retry(tmp_path, monkeypatch):
    source = tmp_path / "0000001.net_model"
    source.mkdir()
    target = tmp_path / "0000001.net_model.pth"
    monkeypatch.setattr(sys, "argv", ["fsdp_convert.py", str(tmp_path), "--iters", "1"])

    def interrupted_conversion(source_dir, output_path):
        assert source_dir == str(source)
        Path(output_path).write_bytes(b"partial checkpoint")
        raise OSError("interrupted conversion")

    monkeypatch.setattr(fsdp_convert, "dcp_to_torch_save", interrupted_conversion)
    with pytest.raises(OSError, match="interrupted conversion"):
        fsdp_convert.main()
    assert set(tmp_path.iterdir()) == {source}

    monkeypatch.setattr(
        fsdp_convert, "dcp_to_torch_save", lambda source_dir, output_path: Path(output_path).write_bytes(b"complete")
    )
    fsdp_convert.main()
    assert target.read_bytes() == b"complete"
    assert set(tmp_path.iterdir()) == {source, target}


def test_fsdp_conversion_survives_a_trainer_checkpoint_sweep(tmp_path, monkeypatch):
    (tmp_path / "0000001.net_model").mkdir()
    monkeypatch.setattr(sys, "argv", ["fsdp_convert.py", str(tmp_path), "--iters", "1"])

    def convert_while_the_trainer_saves(source_dir, output_path):
        Path(output_path).write_bytes(b"complete")
        assert remove_atomic_write_orphans(tmp_path, "*.pth") == []

    monkeypatch.setattr(fsdp_convert, "dcp_to_torch_save", convert_while_the_trainer_saves)
    fsdp_convert.main()
    assert (tmp_path / "0000001.net_model.pth").read_bytes() == b"complete"
