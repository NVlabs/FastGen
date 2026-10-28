# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise a built distribution outside the checkout; editable installs hide missing files."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOTS = ("fastgen", "scripts", "comfyui")


def _run(command, cwd, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    root = tmp_path_factory.mktemp("packaging")
    source = root / "source"
    source.mkdir()
    for name in ("setup.py", "pyproject.toml", "MANIFEST.in", "requirements.txt", "README.md", "LICENSE"):
        shutil.copy2(_REPOSITORY_ROOT / name, source / name)
    for name in (*_PACKAGE_ROOTS, "licenses"):
        shutil.copytree(_REPOSITORY_ROOT / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Build the wheel from the sdist so both distribution formats must carry the inputs.
    dist = root / "dist"
    _run([sys.executable, "setup.py", "sdist", "--dist-dir", str(dist)], source)
    (sdist,) = dist.glob("*.tar.gz")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist),
            str(sdist),
        ],
        root,
    )
    (wheel,) = dist.glob("*.whl")
    return wheel


def test_distribution_includes_all_python_modules(built_wheel):
    with zipfile.ZipFile(built_wheel) as archive:
        packaged = set(archive.namelist())
    expected = {
        path.relative_to(_REPOSITORY_ROOT).as_posix()
        for name in _PACKAGE_ROOTS
        for path in (_REPOSITORY_ROOT / name).rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert not expected - packaged, sorted(expected - packaged)


def test_distribution_includes_runtime_data_and_vendor_licenses(built_wheel):
    with zipfile.ZipFile(built_wheel) as archive:
        packaged = set(archive.namelist())
    expected = {
        path.relative_to(_REPOSITORY_ROOT).as_posix()
        for name in _PACKAGE_ROOTS
        for path in (_REPOSITORY_ROOT / name).rglob("*")
        if path.is_file() and (path.name == "LICENSE" or path.suffix in {".json", ".txt", ".png", ".mp4", ".npy"})
    }
    assert not expected - packaged, sorted(expected - packaged)


def test_distribution_includes_third_party_license_notices(built_wheel):
    with zipfile.ZipFile(built_wheel) as archive:
        for source in (_REPOSITORY_ROOT / "licenses").rglob("LICENSE"):
            relative = source.relative_to(_REPOSITORY_ROOT).as_posix()
            matches = [name for name in archive.namelist() if name.endswith("/" + relative)]
            assert len(matches) == 1, relative
            assert archive.read(matches[0]) == source.read_bytes(), relative


def test_installed_wheel_imports_configs_and_preprocessors_without_checkout(built_wheel, tmp_path):
    installed = tmp_path / "installed"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-compile",
            "--target",
            str(installed),
            str(built_wheel),
        ],
        tmp_path,
    )
    env = dict(os.environ, PYTHONPATH=str(installed))
    _run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import fastgen; "
            "assert Path(fastgen.__file__).is_relative_to(Path('installed').resolve()); "
            "from fastgen.configs.experiments.EDM.config_dmd2_fast import create_config; "
            "assert create_config().model.net is not None; "
            "from fastgen.datasets.wds_dataloaders import PRESET_CONSTANTS; "
            "assert PRESET_CONSTANTS['neg_prompt_wan']; "
            "from fastgen.third_party.annotators.depth_anything_v2.dpt import DepthAnythingV2",
        ],
        tmp_path,
        env,
    )
