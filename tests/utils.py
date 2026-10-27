# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test helpers for the FastGen test suite."""

import os
import pickle
import socket
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from importlib.metadata import version as get_pkg_version
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from packaging.version import Version
from pytest import MarkDecorator

# =============================================================================
# Distributed Test Infrastructure
# =============================================================================


def _init_distributed(rank: int, world_size: int, port: int, nccl_timeout: int = 600):
    """Initialize distributed process group for testing.

    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Port for distributed communication
        nccl_timeout: Timeout in seconds for NCCL operations (default: 600)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    from datetime import timedelta

    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=nccl_timeout),
    )
    torch.cuda.set_device(rank)


def _cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _reap_process(p: "mp.Process", term_timeout: float = 5.0, kill_timeout: float = 5.0) -> None:
    """SIGTERM a worker, then escalate to SIGKILL if it does not exit.

    A hung NCCL collective (or any C-extension blocked in a long syscall) can
    mask SIGTERM, leaving the worker holding its CUDA context and wedging the
    GPU for every subsequent test in the session. Always escalate to SIGKILL
    so the kernel reclaims the device memory before the next test starts.
    """
    if not p.is_alive():
        return
    p.terminate()
    p.join(timeout=term_timeout)
    if p.is_alive():
        p.kill()
        p.join(timeout=kill_timeout)
        if p.is_alive():
            warnings.warn(f"Process {p.pid} did not exit after SIGKILL; CUDA context may still be held.", stacklevel=2)


def _error_watchdog(error_event: mp.Event, local_failure: threading.Event, check_interval: float = 0.5):
    """Watchdog thread that monitors for errors from other ranks.

    When the error_event is set by another rank, this thread will terminate
    the current process to prevent hanging on NCCL collectives.

    Args:
        error_event: Shared multiprocessing Event that signals an error occurred
        local_failure: Process-local flag set by this rank's own failure, which reports before it exits
        check_interval: How often to check for errors (seconds)
    """
    while True:
        if error_event.is_set() and not local_failure.is_set():
            # Another rank failed, force exit to prevent hanging on collectives
            os._exit(1)
        time.sleep(check_interval)


def _run_distributed_test(
    rank: int,
    world_size: int,
    port: int,
    test_fn: Callable,
    result_file: str,
    error_file: str,
    error_event: mp.Event,
    setup_fn: Optional[Callable] = None,
    nccl_timeout: int = 600,
    **kwargs,
):
    """Worker function to run a distributed test.

    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Port for distributed communication
        test_fn: The test function to run with signature fn(rank, world_size, **kwargs)
        result_file: Path to file where rank 0 writes results
        error_file: Path to file where any rank writes error details
        error_event: Shared event to signal errors across ranks
        setup_fn: Optional setup function to run after distributed init (e.g., set_env_vars)
        nccl_timeout: Timeout in seconds for NCCL operations (default: 600)
        **kwargs: Additional arguments to pass to test_fn
    """
    from fastgen.utils.basic_utils import set_random_seed

    # Start watchdog thread to monitor for errors from other ranks
    local_failure = threading.Event()
    watchdog = threading.Thread(target=_error_watchdog, args=(error_event, local_failure), daemon=True)
    watchdog.start()

    try:
        _init_distributed(rank, world_size, port, nccl_timeout=nccl_timeout)
        # Spawned workers otherwise start from per-process random seeds. Seed every rank identically as
        # ``Trainer.__init__`` does before building models, so real-initialized ranks build identical
        # weights. Tests that need rank-dependent randomness derive it from ``rank``.
        set_random_seed(0, by_rank=False)

        # Run optional setup function (e.g., set_env_vars for library-specific setup)
        if setup_fn is not None:
            setup_fn()

        # Run the test function
        result = test_fn(rank=rank, world_size=world_size, **kwargs)

        # Only rank 0 writes the result to file
        if rank == 0:
            with open(result_file, "wb") as f:
                pickle.dump(("success", result), f)

    # pytest outcomes derive from BaseException, so a missed ``pytest.raises`` must fail the test too.
    except (Exception, pytest.fail.Exception, pytest.skip.Exception) as e:
        import traceback

        error_info = {
            "rank": rank,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

        # Signal error to all other ranks FIRST (before any cleanup); the local flag keeps this rank's own
        # watchdog from exiting before the traceback below is written.
        local_failure.set()
        error_event.set()

        # Write error details to file (use rank-specific suffix to avoid race conditions)
        try:
            error_file_rank = f"{error_file}.rank{rank}"
            with open(error_file_rank, "wb") as f:
                pickle.dump(error_info, f)
        except Exception:
            pass  # Best effort - error signaling via event is the priority

        # Also write to main result file if rank 0
        if rank == 0:
            with open(result_file, "wb") as f:
                pickle.dump(("error", (str(e), traceback.format_exc())), f)

    finally:
        _cleanup_distributed()


def run_distributed_test(
    test_fn: Callable,
    world_size: int = 2,
    timeout: int = 300,
    setup_fn: Optional[Callable] = None,
    nccl_timeout: Optional[int] = None,
    **kwargs,
) -> Optional[Dict]:
    """Run a test function in a distributed setting using multiprocessing.spawn.

    This utility handles:
    - Finding a free port for NCCL communication
    - Spawning worker processes with proper distributed initialization
    - Collecting results from rank 0 via pickle file (avoiding Queue issues with CUDA)
    - Cross-rank error signaling to prevent hanging when one rank fails
    - Proper cleanup of process groups

    Args:
        test_fn: Test function with signature fn(rank, world_size, **kwargs)
                 Should return a dict with test results from rank 0.
        world_size: Number of processes to spawn (default: 2)
        timeout: Timeout in seconds for each process to complete (default: 300)
        setup_fn: Optional setup function to run after distributed init on each rank.
                  Useful for library-specific setup like set_env_vars().
        nccl_timeout: Timeout in seconds for NCCL operations. Defaults to match `timeout`.
        **kwargs: Additional arguments to pass to test_fn

    Returns:
        Result dict from rank 0 test execution, or None if no result was written

    Raises:
        TimeoutError: If any process exceeds the timeout
        AssertionError: If any process exits with non-zero code or test fails

    Example:
        ```python
        def _test_my_distributed_feature(rank: int, world_size: int, some_arg: int) -> dict:
            # Setup distributed model/training
            ...
            # Run test logic
            ...
            # Return results (only rank 0's result is collected)
            return {"passed": True, "value": computed_value}

        @RunIf(min_gpus=2)
        def test_my_distributed_feature():
            result = run_distributed_test(
                test_fn=_test_my_distributed_feature,
                world_size=2,
                some_arg=42,
            )
            assert result["passed"]
        ```
    """
    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    # Create temp files for results and errors
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        result_file = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".err") as f:
        error_file = f.name

    # Create shared error event for cross-rank signaling
    ctx = mp.get_context("spawn")
    error_event = ctx.Event()

    # Default nccl_timeout to match test timeout
    if nccl_timeout is None:
        nccl_timeout = timeout

    processes = []
    try:
        # Spawn processes
        for rank in range(world_size):
            p = ctx.Process(
                target=_run_distributed_test,
                args=(rank, world_size, port, test_fn, result_file, error_file, error_event),
                kwargs={"setup_fn": setup_fn, "nccl_timeout": nccl_timeout, **kwargs},
            )
            p.start()
            processes.append(p)

        # Wait for processes with periodic error checking
        start_time = time.time()
        while True:
            # A rank that dies without raising (e.g. a segfault) never sets the event, and its peers
            # would block in NCCL until ``nccl_timeout``.
            crashed = {
                i: p.exitcode for i, p in enumerate(processes) if not p.is_alive() and p.exitcode not in (0, None)
            }
            if error_event.is_set() or crashed:
                # Give processes a moment to write error files and exit
                time.sleep(0.5)
                for p in processes:
                    _reap_process(p)
                break

            # Check if all processes have finished
            all_done = all(not p.is_alive() for p in processes)
            if all_done:
                break

            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                for p in processes:
                    _reap_process(p)
                raise TimeoutError(f"Distributed test timed out after {timeout}s")

            # Brief sleep to avoid busy waiting
            time.sleep(0.1)

        # Join all processes to collect exit codes
        for p in processes:
            p.join(timeout=5)

        # Collect error information from all ranks
        error_messages = []
        for rank in range(world_size):
            error_file_rank = f"{error_file}.rank{rank}"
            if os.path.exists(error_file_rank):
                try:
                    with open(error_file_rank, "rb") as f:
                        error_info = pickle.load(f)
                    error_messages.append(
                        f"Rank {error_info['rank']} failed:\n{error_info['error']}\n{error_info['traceback']}"
                    )
                except Exception:
                    error_messages.append(f"Rank {rank} failed (unable to read error details)")

        # If we collected errors, raise them
        if error_messages:
            raise AssertionError(
                f"Distributed test failed on {len(error_messages)} rank(s):\n\n" + "\n---\n".join(error_messages)
            )

        # Name the rank that died first; its peers were terminated above.
        if crashed:
            crash_report = ", ".join(f"rank {i} exited with code {code}" for i, code in crashed.items())
            raise AssertionError(f"Distributed test crashed without an error report: {crash_report}")

        # Check if any process failed with non-zero exit code
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                raise AssertionError(f"Process {i} exited with code {p.exitcode}")

        # Read result from file
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            with open(result_file, "rb") as f:
                status, result = pickle.load(f)
            if status == "error":
                error_msg, traceback_str = result
                raise AssertionError(f"Distributed test failed:\n{error_msg}\n{traceback_str}")
            return result

        return None

    finally:
        # Ensure all processes are terminated (and their CUDA contexts released)
        for p in processes:
            _reap_process(p)

        # Clean up temp files
        for f in [result_file, error_file]:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception:
                    pass
        # Clean up rank-specific error files
        for rank in range(world_size):
            error_file_rank = f"{error_file}.rank{rank}"
            if os.path.exists(error_file_rank):
                try:
                    os.unlink(error_file_rank)
                except Exception:
                    pass


# =============================================================================
# Conditional Test Skipping
# =============================================================================


@dataclass(frozen=True)
class _GPURequirement:
    """Defer CUDA discovery until pytest evaluates a selected test's skip condition."""

    min_gpus: int
    min_gpu_mem: Optional[float]
    other_requirement_unmet: bool = False

    def __bool__(self) -> bool:
        if self.other_requirement_unmet:
            return True
        count = torch.cuda.device_count()
        if count < self.min_gpus:
            return True
        if self.min_gpu_mem is not None:
            if count == 0:
                return True
            # Decimal GB matches marketed capacity, and every visible rank must have enough memory.
            smallest = min(torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(count))
            return smallest < self.min_gpu_mem
        return False


class RunIf:
    """RunIf wrapper for conditional skipping of tests.

    Fully compatible with `@pytest.mark`. GPU requirements are checked at test setup,
    so collecting a module or selecting only its CPU tests does not query CUDA.

    Example:

    ```python
        @RunIf(min_torch="1.8")
        @pytest.mark.parametrize("arg1", [1.0, 2.0])
        def test_wrapper(arg1):
            assert arg1 > 0
    ```
    """

    def __new__(
        cls,
        min_gpus: int = 0,
        min_gpu_mem: Optional[float] = None,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        requires_file: Optional[Union[str, List[str]]] = None,
        **kwargs: Dict[Any, Any],
    ) -> MarkDecorator:
        """Creates a new `@RunIf` `MarkDecorator` decorator.

        :param min_gpus: Min number of GPUs required to run test.
        :param min_gpu_mem: Min total memory in GB as marketed (an "80GB" H100 satisfies ``80``) on every visible GPU.
        :param min_torch: Minimum pytorch version to run test.
        :param max_torch: Maximum pytorch version to run test.
        :param min_python: Minimum python version required to run test.
        :param requires_file: File or list of files required to run test.
        :param kwargs: Native `pytest.mark.skipif` keyword arguments.
        """
        conditions = []
        reasons = []

        if min_torch:
            torch_version = get_pkg_version("torch")
            conditions.append(Version(torch_version) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            torch_version = get_pkg_version("torch")
            conditions.append(Version(torch_version) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if requires_file:
            if isinstance(requires_file, str):
                requires_file = [requires_file]
            conditions.append(any(not Path(file).exists() for file in requires_file))
            reasons.append(f"requires file={','.join(requires_file)}")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        condition = any(conditions)
        if min_gpus or min_gpu_mem is not None:
            condition = _GPURequirement(min_gpus, min_gpu_mem, other_requirement_unmet=condition)
            if min_gpus:
                reasons.append(f"GPUs>={min_gpus}")
            if min_gpu_mem is not None:
                reasons.append(f"GPU memory>={min_gpu_mem}GB")
        return pytest.mark.skipif(
            condition=condition,
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )


# =============================================================================
# Gradient Assertions
# =============================================================================


def check_grad_zero(model: torch.nn.Module):
    """Check that every accumulated gradient is zero, whatever the parameter's current ``requires_grad``."""
    for param in model.parameters():
        if param.grad is not None:
            assert torch.allclose(param.grad, torch.zeros_like(param.grad))


def _gradients_by_parameter(
    net: torch.nn.Module,
    run_loss: Callable[[], torch.Tensor],
    device_type: str,
    **autocast_kwargs,
) -> dict[str, torch.Tensor]:
    """Run backward under BF16 autocast and return gradients by parameter name."""
    net.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, **autocast_kwargs):
        loss = run_loss()
    loss.backward()
    return {name: p.grad.detach().clone() for name, p in net.named_parameters() if p.grad is not None}


def assert_autocast_weight_cache_neutral(
    net: torch.nn.Module,
    run_loss: Callable[[], torch.Tensor],
    device_type: str,
) -> None:
    """Assert that BF16 autocast cache state does not affect parameter gradients."""
    reference = _gradients_by_parameter(net, run_loss, device_type, enabled=False)
    assert reference, "the fp32 reference run produced no gradients"
    cache_on = _gradients_by_parameter(net, run_loss, device_type, enabled=True, cache_enabled=True)
    cache_off = _gradients_by_parameter(net, run_loss, device_type, enabled=True, cache_enabled=False)
    assert set(cache_on) == set(reference), f"missing with the cache on: {sorted(set(reference) - set(cache_on))}"
    assert set(cache_off) == set(reference), f"missing with the cache off: {sorted(set(reference) - set(cache_off))}"
    for name, grad in cache_on.items():
        assert torch.isfinite(grad).all(), name
        assert torch.allclose(grad, cache_off[name], rtol=1e-3, atol=1e-4), f"gradient of {name} depends on the cache"


# =============================================================================
# Training Probes
# =============================================================================


def record_generation_renoising(model: torch.nn.Module, monkeypatch: pytest.MonkeyPatch) -> list:
    """Record ``(t, t_start, student_step)`` of each renoising of a DMD2-family model's generations.

    Generations at a grid time ``t_start`` are noised to ``t`` with ``forward_process(..., t_start=t_start)``;
    real data and student inputs are noised from clean data without ``t_start``, so only the former are recorded.
    """
    renoisings, student_steps = [], []
    gen_data_from_net = model.gen_data_from_net
    forward_process = model.net.noise_scheduler.forward_process

    def record_generation(input_student, student_step, **kwargs):
        student_steps.append(student_step)
        return gen_data_from_net(input_student, student_step, **kwargs)

    def record_forward_process(x, eps, t, t_start=None):
        if t_start is not None:
            renoisings.append((t, t_start, student_steps[-1]))
        return forward_process(x, eps, t, t_start=t_start)

    monkeypatch.setattr(model, "gen_data_from_net", record_generation)
    monkeypatch.setattr(model.net.noise_scheduler, "forward_process", record_forward_process)
    return renoisings


# =============================================================================
# Network Configs
# =============================================================================


def network_schedule_kwargs(net_config: Any) -> Dict[str, Any]:
    """Return the kwargs ``net_config``'s network passes to its noise schedule, constructor defaults included.

    The constructor runs only until it calls ``FastGenNetwork.__init__``, so no network is built and nothing loads.
    Wrappers that build their schedule from sub-networks, such as the Wan MoE ones, report only their own call.
    """
    from fastgen.configs.instantiate import instantiate
    from fastgen.networks.network import FastGenNetwork

    class _ScheduleKwargs(Exception):
        pass

    def capture(self, net_pred_type="x0", schedule_type="edm", **schedule_kwargs):
        raise _ScheduleKwargs(schedule_kwargs)

    with mock.patch.object(FastGenNetwork, "__init__", capture):
        try:
            instantiate(net_config)
        except _ScheduleKwargs as captured:
            return captured.args[0]
    raise AssertionError(f"{net_config['_target_']} never called FastGenNetwork.__init__")
