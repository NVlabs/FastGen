# SPDX-FileCopyrightText: Copyright (c) 2021 Facebook, Inc. and its affiliates.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction helpers for object specifications in FastGen configs."""

from collections import abc
from dataclasses import is_dataclass
import pydoc
from typing import Any

from omegaconf import DictConfig, ListConfig


def _convert_target_to_string(target: Any) -> str:
    """Return the shortest import path that resolves to ``target``."""
    module, qualname = target.__module__, target.__qualname__

    # Compress paths such as ``module.submodule._impl.Class`` when the object
    # is also re-exported from a parent package.
    module_parts = module.split(".")
    for index in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:index])
        candidate = f"{prefix}.{qualname}"
        try:
            if locate(candidate) is target:
                return candidate
        except ImportError:
            pass
    return f"{module}.{qualname}"


def locate(name: str) -> Any:
    """Resolve ``name`` to the object identified by its import path."""
    obj = pydoc.locate(name)

    # Some paths (for example ``torch.optim.sgd.SGD``) are not handled by
    # pydoc.locate. Fall back to Hydra's implementation for those cases.
    if obj is None:
        try:
            # Importing hydra.utils.get_method prints many errors, so use the
            # underlying locator directly.
            from hydra.utils import _locate
        except ImportError as error:
            import fastgen.utils.logging_utils as logger

            logger.error(f"Cannot dynamically locate object {name}!")
            raise error
        else:
            obj = _locate(name)

    return obj


def instantiate(cfg, *args, **kwargs):
    """Recursively instantiate mappings that contain a ``_target_`` key."""
    if isinstance(cfg, ListConfig):
        values = [instantiate(value) for value in cfg]
        return ListConfig(values, flags={"allow_objects": True})
    if isinstance(cfg, list):
        # Many constructors accept a list of instantiated objects.
        return [instantiate(value) for value in cfg]

    if isinstance(cfg, abc.Mapping) and "_target_" in cfg:
        # This is equivalent to hydra.utils.instantiate(cfg, _convert_="all")
        # but avoids the overhead tracked in Hydra issue #1200.
        cfg = {key: instantiate(value) for key, value in cfg.items()}
        target = instantiate(cfg.pop("_target_"))

        if isinstance(target, str):
            target_name = target
            target = locate(target_name)
            assert target is not None, target_name
        else:
            try:
                target_name = target.__module__ + "." + target.__qualname__
            except Exception:
                # A target may be any callable, including one without these
                # descriptive attributes.
                target_name = str(target)
        assert callable(target), f"_target_ {target} does not define a callable object"

        try:
            additional_kwargs = {}
            additional_kwargs.update(cfg)
            additional_kwargs.update(kwargs)
            return target(*args, **additional_kwargs)
        except TypeError:
            import fastgen.utils.logging_utils as logger

            logger.error(f"Error when instantiating {target_name}!")
            raise

    return cfg


class LazyCall:
    """Build a config specification instead of immediately calling a target.

    ``LazyCall`` accepts a callable or import path and must itself be called with
    keyword arguments only. The result can be edited before it is passed to ``instantiate``.

    Example::

        from fastgen.configs.instantiate import LazyCall, instantiate

        layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
        layer_cfg.out_channels = 64
        layer = instantiate(layer_cfg)
    """

    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(f"target of LazyCall must be a callable or defines a callable! Got {target}")
        self._target = target

    def __call__(self, **kwargs):
        if is_dataclass(self._target):
            # OmegaConf cannot hold a dataclass type directly.
            # See https://github.com/hydra-ecosystem/omegaconf/issues/784.
            target = _convert_target_to_string(self._target)
        else:
            target = self._target
        kwargs["_target_"] = target
        return DictConfig(content=kwargs, flags={"allow_objects": True})
