# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# Inductor's subprocess compile pool spawns workers that re-initialize CUDA
# independently. Once the runner GPU is briefly busy or wedged, those workers
# fail with "Could not find an active GPU backend" and every test that JIT-
# compiles a Triton kernel (e.g. flex_attention in WAN/CausVid) fails. Compile
# in-process instead: slower per-test, but the workers cannot diverge from the
# parent's CUDA state. MUST be set before `import torch`.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# The DataLoader pin-memory thread's ``torch.set_num_threads(1)`` disables MKL_DYNAMIC
# process-wide, so the CI image's MKL 2021 threads VML ops (``torch.cos``/``exp`` on large
# CPU tensors) inside ATen's OpenMP regions, where its racy lazy init intermittently
# segfaults in ``mkl_vml_serv_threader_*``. MUST be set before `import torch`.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# nv25.11 still reads CUDA-specific options from the legacy allocator alias.
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512"),
)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ["PYTORCH_ALLOC_CONF"])

import gc
import multiprocessing
import torch
import pytest

# Avoid "use of fork() may lead to deadlocks in the child" on Python 3.12+
multiprocessing.set_start_method("forkserver", force=True)


def pytest_addoption(parser):
    """Add command-line options for pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require real data",
    )
    parser.addoption(
        "--no-offload-preprocessors",
        action="store_true",
        default=False,
        help="Keep VAE / text-encoder / image-encoder on GPU for the whole test.  "
        "Default offloads them to CPU between encode/decode calls, freeing GPU "
        "memory for the transformer under test.",
    )
    parser.addoption(
        "--credentials-dir",
        default="./credentials",
        help="Folder with the optional credential files loaded like the config's credentials_dir",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "large_model: marks tests that load large models (deselect with '-m \"not large_model\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests (run with --run-integration)")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        # --run-integration given: do not skip integration tests
        return
    skip_integration = pytest.mark.skip(reason="Requires real data - use pytest --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def cleanup_after_test(cleanup_cuda_after_test):
    """Collect CPU resources before CUDA cleanup releases their cached allocations."""
    yield
    gc.collect()


@pytest.fixture(autouse=True)
def cleanup_cuda_after_test():
    """Clean an existing CUDA context without probing GPUs or creating a context for CPU tests."""
    yield
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture(autouse=True, scope="session")
def set_default_env(pytestconfig):
    """Set HF_HOME and other env vars to the same defaults used during training."""
    from fastgen.utils.env import set_env_vars

    set_env_vars(pytestconfig.getoption("--credentials-dir"))
    yield


@pytest.fixture(autouse=True, scope="session")
def increase_file_limit():
    """
    Attempt to increase file descriptor limit at the session level.
    This runs once at the start of the test session.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Try to increase to 4096, but don't exceed hard limit
        new_soft = 4096 if hard == resource.RLIM_INFINITY else min(4096, hard)
        if soft != resource.RLIM_INFINITY and soft < new_soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        else:
            new_soft = soft
        print(f"\n✓ Increased file descriptor limit: {soft} -> {new_soft}")
    except Exception as e:
        print(f"\n⚠ Could not increase file descriptor limit: {e}")
        print("  This may cause 'Too many open files' errors with many tests")

    yield
