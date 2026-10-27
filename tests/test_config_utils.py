# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``--config`` loader and the config serializer."""

from __future__ import annotations

import importlib
import os
import sys
from enum import Enum

import attrs
import pytest
import yaml
from omegaconf import DictConfig

from fastgen.configs.config_utils import import_config_from_python_file, serialize_config
from fastgen.configs.instantiate import LazyCall as L


@pytest.fixture
def probe_package(tmp_path, monkeypatch):
    """A tiny importable config package below ``tmp_path``, which becomes the working directory.

    The package name starts with ``py`` on purpose: the module name used to be derived by stripping
    every ``.py`` substring, which silently mangled such path segments.
    """
    package = tmp_path / "pyprobe_configs"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "config_probe.py").write_text("def create_config():\n    return 'probe-config'\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield package
    for name in [name for name in sys.modules if name.split(".")[0] == package.name]:
        del sys.modules[name]


@pytest.mark.parametrize("prefix", ["", "."])
def test_loader_accepts_both_spellings_of_a_relative_config_path(probe_package, prefix):
    """``--config`` is written by hand, so a leading ``./`` must reach the same module."""
    config_file = os.path.join(prefix, probe_package.name, "config_probe.py")

    assert import_config_from_python_file(config_file) == "probe-config"


def test_loader_rejects_an_absolute_config_path(probe_package):
    """The same config that imports as a relative path has no module name when spelled absolutely."""
    with pytest.raises(ValueError, match="relative to the working directory"):
        import_config_from_python_file(str(probe_package / "config_probe.py"))


class _ProbeMode(str, Enum):
    FAST = "fast"


class _ProbeNet:
    pass


@attrs.define
class _ProbeConfig:
    net: DictConfig = attrs.field(
        factory=lambda: L(_ProbeNet)(width=8, scale=7e-05, causal=True, label=None, lr="1e-4", mode=_ProbeMode.FAST)
    )
    sizes: list = attrs.field(factory=lambda: [1, 2])


def test_serializer_keeps_yaml_scalars_and_writes_other_values_as_strings():
    """W&B and ``config.yaml`` get numbers and booleans back. A class or an enum, which YAML would tag as a Python
    object that safe loaders reject, is written as its string."""
    serialized = serialize_config(_ProbeConfig())

    assert serialized == {
        "net": {
            "_target_": str(_ProbeNet),
            "width": 8,
            "scale": 7e-05,
            "causal": True,
            "label": None,
            "lr": "1e-4",
            "mode": str(_ProbeMode.FAST),
        },
        "sizes": [1, 2],
    }
    assert yaml.safe_load(serialize_config(_ProbeConfig(), return_type="yaml")) == serialized
