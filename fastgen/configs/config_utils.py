# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config loading, command-line overrides, and serialization.

Overrides use Hydra's grammar. A dict value such as ``key={a: 1}`` merges into the existing dict and cannot add keys to
it, so to replace a dict, clear it with ``key=null`` and add its entries with ``++key.a=1``.
"""

import os
from collections.abc import Mapping
from typing import Any, Optional, List, Dict, Tuple
from copy import deepcopy

import attrs
import yaml
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

import importlib
from dataclasses import fields as dataclass_fields
import attr
from dataclasses import is_dataclass
import fastgen.utils.logging_utils as logger


def import_config_from_python_file(config_file: str) -> Any:
    """
    Import a config from a python file.

    Args:
        config_file (str): The path to the python file, relative to the working directory.

    Returns:
        Any: The config object.
    """

    if not config_file.endswith(".py"):
        raise ValueError("Config file must be a Python file with a .py extension. " f"Received: {config_file}")

    if os.path.isabs(config_file):
        raise ValueError(
            "Config file must be a path relative to the working directory, since it is converted to "
            f"a module name. Received: {config_file}"
        )

    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"FastGen config file ({config_file}) not found.")

    # Convert to importable module format.
    normalized = os.path.normpath(config_file)
    if normalized == os.pardir or normalized.startswith(os.pardir + os.sep):
        raise ValueError(
            "Config file must live below the working directory, since it is converted to a module name. "
            f"Received: {config_file}"
        )
    config_module = os.path.splitext(normalized)[0].replace(os.sep, ".")

    # Import the module
    try:
        module = importlib.import_module(config_module)
    except ImportError as e:
        logger.error(f"Failed to import config from python file: {e}")
        raise e

    return module.create_config()


def model_pixel_shape(model_config: Any) -> Tuple[int, ...]:
    """``(H, W)`` or ``(T, H, W)`` of the pixels that encode to ``model_config.input_shape``.

    Asks the class of ``model_config.net`` through ``pixel_shape_for``, so a config can size its loaders before any
    network exists. A modality dict contributes its ``"video"`` entry.
    """
    input_shape = model_config.input_shape
    if isinstance(input_shape, Mapping):
        input_shape = input_shape["video"]
    return model_config.net._target_.pixel_shape_for(input_shape)


def config_from_dict(ref_instance: Any, kwargs: Any) -> Any:
    """
    Construct an instance of the same type as ref_instance using the provided dictionary or data or unstructured data

    Args:
        ref_instance: The reference instance to determine the type and fields when needed
        kwargs: A dictionary of keyword arguments to use for constructing the new instance or primitive data or unstructured data

    Returns:
        Any: A new instance of the same type as ref_instance constructed using the provided kwargs or the primitive data or unstructured data

    Raises:
        AssertionError: If the fields do not match or if extra keys are found.
        Exception: If there is an error constructing the new instance.
    """
    is_type = is_attrs_or_dataclass(ref_instance)
    if not is_type:
        return kwargs
    else:
        ref_fields = set(get_fields(ref_instance))
        assert isinstance(kwargs, dict) or isinstance(kwargs, DictConfig), "kwargs must be a dictionary or a DictConfig"
        keys = set(kwargs.keys())

        # ref_fields must include all keys
        extra_keys = keys - ref_fields
        assert keys <= ref_fields, f"Unknown config fields {extra_keys} for {type(ref_instance)}"

        resolved_kwargs: Dict[str, Any] = {}
        for f in keys:
            resolved_kwargs[f] = config_from_dict(getattr(ref_instance, f), kwargs[f])
        try:
            new_instance = type(ref_instance)(**resolved_kwargs)
        except Exception as e:
            logger.error(f"Error when constructing {type(ref_instance)} with {resolved_kwargs}")
            logger.error(e)
            raise e
        return new_instance


def override_config_with_opts(config: Any, opts: Optional[List[str]] = None) -> Any:
    """
    Override the config with the opts.

    Args:
        config (Any): The config object.
        opts (Optional[List[str]]): Hydra override strings, starting with the "-" separator
            (e.g. ["-", "trainer.max_iter=10"]).

    Returns:
        Any: The config object.
    """

    # Convert Config object to a DictConfig object for Hydra

    if not isinstance(config, DictConfig):
        config_dict = attrs.asdict(config)
        config_dict = DictConfig(content=config_dict, flags={"allow_objects": True})
    else:
        config_dict = config

    # Use Hydra to handle overrides
    cs = ConfigStore.instance()
    cs.store(name="config", node=config_dict)

    if opts is None:
        opts = []

    if len(opts) > 0 and opts[0] != "-":
        raise ValueError(f"opts must start with '-' to separate from other arguments. Got: {opts}")
    opts = opts[1:]
    with initialize(version_base=None):
        try:
            cfg = compose(config_name="config", overrides=opts)
        except Exception as e:
            raise ValueError(f"Failed to compose config with opts: {e}")

        OmegaConf.resolve(cfg)

    config = config_from_dict(config, cfg)

    return config


def is_attrs_or_dataclass(obj) -> bool:
    """
    Check if the object is an instance of an attrs class or a dataclass.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is an instance of an attrs class or a dataclass, False otherwise.
    """
    return is_dataclass(obj) or attr.has(type(obj))


def get_fields(obj):
    """
    Get the fields of an attrs class or a dataclass.

    Args:
        obj: The object to get fields from. Must be an instance of an attrs class or a dataclass.

    Returns:
        list: A list of field names.

    Raises:
        ValueError: If the object is neither an attrs class nor a dataclass.
    """
    if is_dataclass(obj):
        return [field.name for field in dataclass_fields(obj)]
    elif attr.has(type(obj)):
        return [field.name for field in attr.fields(type(obj))]
    else:
        raise ValueError("The object is neither an attrs class nor a dataclass.")


def serialize_config(
    config: Any,
    return_type: str = "dict",
    path: str | bytes | None = None,
    filename: str = "config.yaml",
) -> Dict[str, Any] | str:
    """
    Serialize a config (BaseConfig or DictConfig) to various formats.

    Args:
        config: The config to serialize (BaseConfig attrs object or DictConfig).
        return_type: Output format - "dict" (plain dict), "yaml" (YAML string), or "file" (save to file).
        path: Directory path to save the file. Required if return_type is "file".
        filename: Name of the file to save. Only used if return_type is "file".

    Returns:
        Dict[str, Any] if return_type is "dict"
        str (YAML) if return_type is "yaml" or "file"

    Raises:
        ValueError: If return_type is "file" but path is not provided.
    """
    if return_type == "file" and path is None:
        raise ValueError("path must be provided when return_type is 'file'")

    # Deep copy to avoid modifying original
    config = deepcopy(config)

    # Normalize to DictConfig with object support
    if not isinstance(config, DictConfig):
        config_dict = attrs.asdict(config)
        config_omegaconf = DictConfig(content=config_dict, flags={"allow_objects": True})
    else:
        config_omegaconf = config

    def is_serializable(item) -> bool:
        # Exact types: YAML tags subclasses, such as str-valued enums, as Python objects that safe loaders reject.
        return type(item) in (bool, int, float, str)

    def process_config(conf):
        if isinstance(conf, DictConfig):
            for key, value in conf.items():
                if isinstance(value, (DictConfig, ListConfig)):
                    process_config(value)
                else:
                    if not is_serializable(value) and value is not None:
                        conf[key] = str(value)
        elif isinstance(conf, ListConfig):
            for i, item in enumerate(conf):
                if isinstance(item, (DictConfig, ListConfig)):
                    process_config(item)
                else:
                    if not is_serializable(item) and item is not None:
                        conf[i] = str(item)
        else:
            raise NotImplementedError("Input config must be a DictConfig or ListConfig.")
        return conf

    config_omegaconf = process_config(config_omegaconf)
    result_dict: Dict[str, Any] = OmegaConf.to_container(config_omegaconf, resolve=True)  # type: ignore

    if return_type == "dict":
        return result_dict

    # For yaml and file, convert to YAML string
    yaml_str = yaml.dump(result_dict, default_flow_style=False, sort_keys=True)

    if return_type == "file":
        os.makedirs(path, exist_ok=True)  # type: ignore
        with open(f"{path}/{filename}", "w") as f:
            f.write(yaml_str)
        logger.info(f"Config is saved at {path}/{filename}")

    return yaml_str
