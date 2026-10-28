# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process import-order and dependency-boundary tests."""

import subprocess
import sys
from pathlib import Path

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run_in_fresh_process(statement: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", statement],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_importing_utils_package_is_lightweight():
    statement = (
        "import sys; import fastgen.utils; "
        "removed = {'LazyCall', 'instantiate', 'locate', 'expand_like', "
        "'set_global_vars', 'set_temp_global_vars'}; "
        "assert removed.isdisjoint(vars(fastgen.utils)), sorted(removed.intersection(vars(fastgen.utils))); "
        "forbidden = {'torch', 'omegaconf', 'fastgen.configs.instantiate', "
        "'fastgen.utils.logging_utils', 'fastgen.utils.distributed', "
        "'fastgen.utils.global_vars', 'fastgen.utils.io_utils', 'fastgen.utils.tensor'}; "
        "assert forbidden.isdisjoint(sys.modules), sorted(forbidden.intersection(sys.modules))"
    )

    _run_in_fresh_process(statement)


def test_importing_config_instantiation_does_not_load_runtime_utilities():
    statement = (
        "import sys; from fastgen.configs.instantiate import LazyCall, instantiate; "
        "assert callable(LazyCall) and callable(instantiate); "
        "forbidden = {'torch', 'fastgen.utils.logging_utils', 'fastgen.utils.distributed', "
        "'fastgen.utils.global_vars', 'fastgen.utils.io_utils', 'fastgen.utils.tensor'}; "
        "assert forbidden.isdisjoint(sys.modules), sorted(forbidden.intersection(sys.modules))"
    )

    _run_in_fresh_process(statement)


def test_importing_global_state_for_reads_does_not_load_logging():
    statement = (
        "import sys; import fastgen.utils.global_vars; "
        "forbidden = {'torch', 'omegaconf', 'fastgen.utils.logging_utils', "
        "'fastgen.utils.distributed', 'fastgen.utils.io_utils'}; "
        "assert forbidden.isdisjoint(sys.modules), sorted(forbidden.intersection(sys.modules))"
    )

    _run_in_fresh_process(statement)


def test_importing_tensor_helpers_does_not_load_unrelated_domains():
    statement = (
        "import sys; "
        "from fastgen.utils.tensor import (PRECISION_MAP, assert_tensor, expand_like, "
        "precision_to_dtype, temp_disable_efficient_attn); "
        "assert PRECISION_MAP and all(callable(value) for value in "
        "(assert_tensor, expand_like, precision_to_dtype, temp_disable_efficient_attn)); "
        "forbidden = {'omegaconf', 'fastgen.configs.instantiate', "
        "'fastgen.utils.logging_utils', 'fastgen.utils.distributed', "
        "'fastgen.utils.global_vars', 'fastgen.utils.io_utils', 'fastgen.utils.basic_utils'}; "
        "assert forbidden.isdisjoint(sys.modules), sorted(forbidden.intersection(sys.modules))"
    )

    _run_in_fresh_process(statement)


def test_importing_s3_leaf_does_not_load_other_utility_domains():
    statement = (
        "import sys; import fastgen.utils.s3; "
        "forbidden_prefixes = ('torch', 'omegaconf', 'requests', 'safetensors', "
        "'fastgen.configs.instantiate', "
        "'fastgen.utils.logging_utils', 'fastgen.utils.distributed', "
        "'fastgen.utils.global_vars', 'fastgen.utils.cache_utils', "
        "'fastgen.utils.io_utils', 'fastgen.utils.tensor'); "
        "loaded = {name for name in sys.modules if any(name == prefix or "
        "name.startswith(prefix + '.') for prefix in forbidden_prefixes)}; "
        "assert not loaded, sorted(loaded)"
    )

    _run_in_fresh_process(statement)


def test_importing_cache_utils_does_not_load_checkpoint_deserialization():
    statement = (
        "import sys; import fastgen.utils.cache_utils; "
        "forbidden_prefixes = ('requests', 'safetensors', 'fastgen.utils.io_utils'); "
        "loaded = {name for name in sys.modules if any(name == prefix or "
        "name.startswith(prefix + '.') for prefix in forbidden_prefixes)}; "
        "assert not loaded, sorted(loaded)"
    )

    _run_in_fresh_process(statement)


def test_importing_io_utils_does_not_load_the_s3_cli():
    statement = (
        "import sys; import fastgen.utils.io_utils; " "assert 'scripts.resumable_s3_download' not in sys.modules"
    )

    _run_in_fresh_process(statement)


def test_io_utils_does_not_reexport_cache_helpers():
    statement = (
        "import fastgen.utils.io_utils as io_utils; "
        "names = ('cache', 'resolve_cached_path', 'checkpoint_cache_dir'); "
        "assert not [name for name in names if hasattr(io_utils, name)]"
    )

    _run_in_fresh_process(statement)


def test_importing_distributed_package_does_not_import_distributed_features():
    statement = (
        "import sys; "
        "import fastgen.utils.distributed; "
        "loaded = {name for name in sys.modules if name == 'fastgen.utils.distributed' "
        "or name.startswith('fastgen.utils.distributed.')}; "
        "expected = {'fastgen.utils.distributed'}; "
        "assert loaded == expected, sorted(loaded)"
    )

    _run_in_fresh_process(statement)


def test_importing_logging_facade_only_imports_the_lightweight_distributed_package():
    statement = (
        "import sys; "
        "import fastgen.utils.logging_utils; "
        "loaded = {name for name in sys.modules if name == 'fastgen.utils.distributed' "
        "or name.startswith('fastgen.utils.distributed.')}; "
        "expected = {'fastgen.utils.distributed'}; "
        "assert loaded == expected, sorted(loaded); "
        "forbidden = {'fastgen.utils.distributed.collectives', 'fastgen.utils.distributed.module', "
        "'fastgen.utils.distributed.ddp', 'fastgen.utils.distributed.fsdp', "
        "'fastgen.utils.distributed.s3_filesystem'}; "
        "assert forbidden.isdisjoint(sys.modules), sorted(forbidden.intersection(sys.modules))"
    )

    _run_in_fresh_process(statement)


@pytest.mark.parametrize(
    "statement",
    (
        "import fastgen.utils.logging_utils; import fastgen.utils.distributed",
        "import fastgen.utils.distributed; import fastgen.utils.logging_utils",
    ),
    ids=("logging-statement-first", "distributed-statement-first"),
)
def test_logging_and_distributed_import_statements_succeed(statement):
    _run_in_fresh_process(statement)


def test_importing_env_only_loads_the_lightweight_distributed_package():
    statement = (
        "import sys; import fastgen.utils.env; "
        "loaded = {name for name in sys.modules if name == 'fastgen.utils.distributed' "
        "or name.startswith('fastgen.utils.distributed.')}; "
        "assert loaded == {'fastgen.utils.distributed'}, sorted(loaded); "
        "forbidden = {'fastgen.utils.basic_utils', 'fastgen.utils.io_utils', "
        "'fastgen.utils.module'}; "
        "assert forbidden.isdisjoint(sys.modules), sorted(forbidden.intersection(sys.modules))"
    )

    _run_in_fresh_process(statement)


@pytest.mark.parametrize(
    "statement",
    (
        "import fastgen.utils.env; import fastgen.utils.distributed",
        "import fastgen.utils.distributed; import fastgen.utils.env",
    ),
    ids=("env-statement-first", "distributed-statement-first"),
)
def test_env_and_distributed_import_statements_succeed(statement):
    _run_in_fresh_process(statement)


def test_importing_module_does_not_load_broad_or_distributed_feature_modules():
    statement = (
        "import sys; import fastgen.utils.module; "
        "loaded = {name for name in sys.modules if name == 'fastgen.utils.distributed' "
        "or name.startswith('fastgen.utils.distributed.')}; "
        "assert loaded == {'fastgen.utils.distributed'}, sorted(loaded); "
        "forbidden = {'fastgen.utils.basic_utils', 'fastgen.utils.env', "
        "'fastgen.utils.io_utils'}; "
        "assert forbidden.isdisjoint(sys.modules), sorted(forbidden.intersection(sys.modules))"
    )

    _run_in_fresh_process(statement)


@pytest.mark.parametrize(
    "statement",
    (
        "import torch; from fastgen.utils.distributed.module import move_and_sync_module; "
        "from fastgen.utils.module import has_meta_state, materialize_meta_module, "
        "tensor_registrations, tied_parameter_registration_groups; assert not has_meta_state(torch.nn.Module())",
        "from fastgen.utils.module import has_meta_state, materialize_meta_module, "
        "tensor_registrations, tied_parameter_registration_groups; "
        "from fastgen.utils.distributed.module import move_and_sync_module",
    ),
    ids=("distributed-module-then-module", "module-then-distributed-module"),
)
def test_distributed_module_and_module_state_import_without_a_cycle(statement):
    _run_in_fresh_process(statement)


@pytest.mark.parametrize(
    "statement",
    (
        "import fastgen.utils.distributed.fsdp",
        "import fastgen.utils.checkpointer",
        "import fastgen.networks.network",
    ),
    ids=("fsdp", "checkpointer", "network"),
)
def test_high_level_modules_import_successfully(statement):
    _run_in_fresh_process(statement)
