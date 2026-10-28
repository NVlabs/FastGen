# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for FastGen's config-instantiation primitives."""

from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace

from omegaconf import ListConfig

import fastgen.configs.instantiate as instantiate_module
from fastgen.configs.instantiate import LazyCall, instantiate, locate


def _capture_config(required, selected, nested):
    return {
        "required": required,
        "selected": selected,
        "nested": nested,
    }


@dataclass
class _DataclassTarget:
    value: int
    label: str = "default"


def test_instantiate_recurses_through_list_config():
    config = ListConfig(
        [
            LazyCall(SimpleNamespace)(value=1),
            LazyCall(SimpleNamespace)(value=2),
        ],
        flags={"allow_objects": True},
    )

    result = instantiate(config)

    assert isinstance(result, ListConfig)
    assert [item.value for item in result] == [1, 2]


def test_instantiate_recurses_through_target_mapping_and_call_kwargs_win():
    config = {
        "_target_": _capture_config,
        "selected": "from-config",
        "nested": LazyCall(SimpleNamespace)(value=3),
    }

    result = instantiate(config, "required", selected="from-call")

    assert result["required"] == "required"
    assert result["selected"] == "from-call"
    assert isinstance(result["nested"], SimpleNamespace)
    assert result["nested"].value == 3


def test_lazy_call_converts_dataclass_target_to_import_path_and_round_trips():
    config = LazyCall(_DataclassTarget)(value=7, label="configured")

    target_path = config["_target_"]
    assert isinstance(target_path, str)
    assert locate(target_path) is _DataclassTarget
    assert instantiate(config) == _DataclassTarget(value=7, label="configured")


def test_locate_resolves_dotted_import_path():
    assert locate("types.SimpleNamespace") is SimpleNamespace


def test_locate_uses_hydra_fallback_when_pydoc_cannot_resolve(monkeypatch):
    sentinel = object()
    requested_names = []
    fake_hydra = ModuleType("hydra")
    fake_hydra_utils = ModuleType("hydra.utils")

    def fake_locate(name):
        requested_names.append(name)
        return sentinel

    fake_hydra_utils._locate = fake_locate
    fake_hydra.utils = fake_hydra_utils
    monkeypatch.setitem(sys.modules, "hydra", fake_hydra)
    monkeypatch.setitem(sys.modules, "hydra.utils", fake_hydra_utils)
    monkeypatch.setattr(instantiate_module.pydoc, "locate", lambda _name: None)

    assert locate("package.hidden.Target") is sentinel
    assert requested_names == ["package.hidden.Target"]
