# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU selection must not acquire the GPU test suite's imports or CUDA context."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


def test_cpu_selection_does_not_import_network_suite_or_probe_cuda(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    # Exercise the real global fixtures in an isolated pytest process. An adjacent GPU test
    # ensures its RunIf decorator is collected even though only the CPU test is selected.
    (tmp_path / "conftest.py").write_text((repo / "tests/conftest.py").read_text())
    probe = tmp_path / "test_selected.py"
    probe.write_text(
        textwrap.dedent(
            """
            import sys
            import torch
            from tests.utils import RunIf

            @RunIf(min_gpus=2, min_gpu_mem=80)
            def test_gpu_not_selected():
                raise AssertionError("GPU test must not run")

            def test_cpu_selected():
                assert "tests.test_network" not in sys.modules
                assert "test_network" not in sys.modules
                assert torch.ones(2, device="cpu").sum().item() == 2
            """
        )
    )
    runner = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import torch

        def forbidden(*args, **kwargs):
            raise AssertionError("CPU-only pytest selection called CUDA")

        for name in ("is_available", "device_count", "get_device_properties", "init", "_lazy_init",
                     "empty_cache", "synchronize"):
            setattr(torch.cuda, name, forbidden)
        torch.cuda.is_initialized = lambda: False

        class BlockNetworkSuite(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"tests.test_network", "test_network"}:
                    raise AssertionError("CPU-only pytest selection imported the full network suite")

        sys.meta_path.insert(0, BlockNetworkSuite())
        import pytest
        raise SystemExit(pytest.main([sys.argv[1] + "::test_cpu_selected", "-q"]))
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", runner, str(probe)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_fixture_teardown_collects_cpu_before_emptying_cuda(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    # Replace only the copied fixture module's cleanup hooks, without initializing CUDA.
    recording = textwrap.dedent(
        """
        from types import SimpleNamespace

        cleanup_events = []
        gc = SimpleNamespace(collect=lambda: cleanup_events.append("gc"))
        torch.cuda.is_initialized = lambda: True
        torch.cuda.empty_cache = lambda: cleanup_events.append("empty_cache")
        torch.cuda.synchronize = lambda: cleanup_events.append("synchronize")

        def pytest_sessionfinish(session, exitstatus):
            assert cleanup_events == ["gc", "empty_cache", "synchronize"]
        """
    )
    (tmp_path / "conftest.py").write_text((repo / "tests/conftest.py").read_text() + recording)
    probe = tmp_path / "test_cleanup_order.py"
    probe.write_text("def test_selected():\n    pass\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


@pytest.mark.parametrize("initialized", [False, True])
def test_cuda_cleanup_requires_an_existing_context(monkeypatch, initialized):
    from tests import conftest

    calls = []
    monkeypatch.setattr(conftest.torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(conftest.torch.cuda, "is_available", lambda: pytest.fail("cleanup must not probe CUDA"))
    monkeypatch.setattr(conftest.torch.cuda, "device_count", lambda: pytest.fail("cleanup must not enumerate GPUs"))
    monkeypatch.setattr(conftest.torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(conftest.torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    cleanup = conftest.cleanup_cuda_after_test.__wrapped__()
    next(cleanup)
    with pytest.raises(StopIteration):
        next(cleanup)
    assert calls == (["empty_cache", "synchronize"] if initialized else [])


@pytest.mark.parametrize(
    ("memories_gb", "min_gpus", "min_gpu_mem", "skip"),
    [([], 1, None, True), ([85.5], 2, None, True), ([85.5, 85.5], 2, 80, False), ([85.5, 40], 2, 80, True)],
)
def test_runif_defers_gpu_checks_and_preserves_requirements(monkeypatch, memories_gb, min_gpus, min_gpu_mem, skip):
    from tests import utils

    queried = []

    def device_count():
        queried.append("count")
        return len(memories_gb)

    def device_properties(index):
        queried.append(index)
        return SimpleNamespace(total_memory=memories_gb[index] * 1e9)

    monkeypatch.setattr(utils.torch.cuda, "device_count", device_count)
    monkeypatch.setattr(utils.torch.cuda, "get_device_properties", device_properties)
    mark = utils.RunIf(min_gpus=min_gpus, min_gpu_mem=min_gpu_mem)
    assert queried == []
    assert bool(mark.kwargs["condition"]) is skip
    assert queried[0] == "count"


def test_runif_non_gpu_failure_skips_without_cuda_queries(monkeypatch, tmp_path):
    from tests import utils

    monkeypatch.setattr(
        utils.torch.cuda, "device_count", lambda: pytest.fail("a missing file already requires skipping")
    )
    missing = tmp_path / "missing-checkpoint.pth"
    mark = utils.RunIf(min_gpus=1, requires_file=str(missing))
    assert bool(mark.kwargs["condition"]) is True
    assert f"requires file={missing}" in mark.kwargs["reason"]
