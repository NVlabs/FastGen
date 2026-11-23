# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from fastgen.utils.basic_utils import (
    attr_override,
    eval_context,
    nested_get,
    safe_update,
    train_mode,
)


def test_nested_get_requires_an_explicit_missing_value_default():
    cfg = SimpleNamespace(model={"student_grid_cfg": {}})

    with pytest.raises(AttributeError, match="missing path component 'grid_kwargs'"):
        nested_get(cfg, "model.student_grid_cfg.grid_kwargs", separator=".")

    assert nested_get(cfg, "model.student_grid_cfg.grid_kwargs", default=None, separator=".") is None


@pytest.mark.parametrize("raise_in_context", [False, True])
def test_attr_override_restores_nested_values_and_removes_added_keys(raise_in_context):
    cfg = SimpleNamespace(model=SimpleNamespace(scale=1.0, grid_kwargs={}))

    try:
        with attr_override(cfg, "model.scale", 5.0), attr_override(cfg, "model.grid_kwargs.shift", 3.0):
            assert (cfg.model.scale, cfg.model.grid_kwargs) == (5.0, {"shift": 3.0})
            if raise_in_context:
                raise RuntimeError("test failure")
    except RuntimeError:
        assert raise_in_context

    assert (cfg.model.scale, cfg.model.grid_kwargs) == (1.0, {})


def test_attr_override_rejects_a_missing_attribute_without_side_effects():
    cfg = SimpleNamespace(model=SimpleNamespace())

    with pytest.raises(AttributeError), attr_override(cfg, "model.scale", 5.0):
        pass

    assert vars(cfg.model) == {}


def test_safe_update():
    target = {"existing": 1}

    safe_update(target, {"new": 2})

    assert target == {"existing": 1, "new": 2}


def test_safe_update_rejects_duplicates_atomically():
    target = {"existing": 1}

    with pytest.raises(ValueError, match=r"Duplicate keys: \['existing'\]"):
        safe_update(target, {"existing": 2, "new": 3})

    assert target == {"existing": 1}


@pytest.mark.parametrize("raise_in_context", [False, True])
def test_train_mode_restores_mixed_child_modes_and_overlapping_roots(raise_in_context):
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout())
    model[0].eval()
    original = {module: module.training for module in model.modules()}

    try:
        with train_mode(model[0], model, mode=False):
            assert all(not module.training for module in model.modules())
            with train_mode(model, mode=True):
                assert all(module.training for module in model.modules())
            assert all(not module.training for module in model.modules())
            if raise_in_context:
                raise RuntimeError("test failure")
    except RuntimeError:
        assert raise_in_context

    assert {module: module.training for module in model.modules()} == original


@pytest.mark.parametrize("inference_mode", [False, True])
@pytest.mark.parametrize("precision_amp", [None, torch.bfloat16])
def test_eval_context_manages_grad_autocast_and_restores_mixed_modes(inference_mode, precision_amp):
    network = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    network[1].eval()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="context failure"):
            with eval_context(
                network,
                precision_amp=precision_amp,
                device_type="cpu",
                inference_mode=inference_mode,
            ):
                assert not torch.is_grad_enabled() and not any(module.training for module in network.modules())
                assert torch.is_inference_mode_enabled() is inference_mode
                assert torch.is_autocast_enabled("cpu") is (precision_amp is not None)
                output = network(torch.ones(1, 2, requires_grad=True))
                assert not output.requires_grad and output.is_inference() is inference_mode
                assert output.dtype == (precision_amp or torch.float32)
                raise RuntimeError("context failure")
        assert torch.is_autocast_enabled("cpu") and torch.is_grad_enabled()

    assert network.training and network[0].training and not network[1].training
    assert not torch.is_inference_mode_enabled() and not torch.is_autocast_enabled("cpu")
