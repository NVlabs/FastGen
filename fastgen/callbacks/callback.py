# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Callable, Any, TYPE_CHECKING
import torch
from torch.utils.data import DataLoader

from fastgen.configs.instantiate import instantiate
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.config import BaseConfig
    from fastgen.trainer import Trainer
    from fastgen.methods import FastGenModel


class CallbackDict:
    def __init__(self, config: BaseConfig, trainer: Trainer):
        self._callbacks = {}
        callback_configs = config.trainer.callbacks
        if callback_configs:
            if isinstance(callback_configs, list):
                logger.warning(msg="The 'config.trainer.callbacks' parameter should be a dict instead of a list. ")
                callback_configs = {f"callback_{k}": v for k, v in enumerate(callback_configs)}
            for callback_name, current_callback_cfg in callback_configs.items():
                if "_target_" not in current_callback_cfg:
                    logger.critical(
                        f"Callback {callback_name} is missing the '_target_' field. \n Skip {current_callback_cfg}"
                    )
                    continue
                logger.critical(f"Instantiating callback {callback_name}: {current_callback_cfg}")
                _callback = instantiate(current_callback_cfg)
                assert isinstance(_callback, Callback), f"{current_callback_cfg} is not a valid callback."
                _callback.config = config
                _callback.trainer = trainer
                _callback.on_app_begin()
                self._callbacks[callback_name] = _callback

    def __getattr__(self, method_name: str) -> Callable:
        # Protocol probes (``copy``, ``pickle``) must see a missing dunder, not a broadcasting wrapper.
        if method_name.startswith("__"):
            raise AttributeError(method_name)

        def load_state_dict(state_dict: dict[str, Any]) -> None:
            for name, callback in self._callbacks.items():
                if name in state_dict:
                    callback.load_state_dict(state_dict[name])
                else:
                    logger.warning(f"Callback {name} not found in checkpoint.")

        def state_dict() -> dict[str, Any]:
            return {name: self._callbacks[name].state_dict() for name in self._callbacks}

        def callbacks_wrapper(*args, **kwargs):
            for callback in self._callbacks.values():
                assert hasattr(callback, method_name)
                method = getattr(callback, method_name)
                assert callable(method), f"{method_name} is not callable."
                method(*args, **kwargs)

        if method_name == "state_dict":
            return state_dict
        if method_name == "load_state_dict":
            return load_state_dict
        return callbacks_wrapper


class Callback:
    config: "BaseConfig"
    trainer: "Trainer"

    def on_app_begin(self) -> None:
        pass

    def on_model_init_start(self, model: FastGenModel) -> None:
        pass

    def on_model_init_end(self, model: FastGenModel | torch.nn.parallel.DistributedDataParallel) -> None:
        pass

    def on_optimizer_init_start(self, model: FastGenModel) -> None:
        pass

    def on_optimizer_init_end(self, model: FastGenModel) -> None:
        pass

    def on_load_checkpoint_start(self, model: FastGenModel) -> None:
        pass

    def on_load_checkpoint_end(self, model: FastGenModel, iteration: int = 0) -> None:
        pass

    def on_dataloader_init_start(self, model: FastGenModel, iteration: int = 0) -> None:
        pass

    def on_dataloader_init_end(
        self, model: FastGenModel, dataloader_train: DataLoader, dataloader_val: DataLoader, iteration: int = 0
    ) -> None:
        pass

    def on_train_begin(self, model: FastGenModel, iteration: int = 0) -> None:
        pass

    def on_training_step_begin(
        self,
        model: FastGenModel,
        iteration: int = 0,
    ) -> None:
        pass

    def on_training_accum_step_begin(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        iteration: int = 0,
        accum_iter: int = 0,
    ) -> None:
        pass

    def on_training_accum_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        iteration: int = 0,
        accum_iter: int = 0,
    ) -> None:
        pass

    def on_backward_begin(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
        accum_iter: int = 0,
    ) -> None:
        pass

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        pass

    def on_optimizer_step_begin(self, model: FastGenModel, iteration: int = 0) -> None:
        pass

    def on_train_end(self, model: FastGenModel, iteration: int = 0) -> None:
        pass

    def on_validation_begin(self, model: FastGenModel, iteration: int = 0, idx: int = 0) -> None:
        pass

    def on_validation_step_begin(
        self, model: FastGenModel, data_batch: dict[str, torch.Tensor], step: int = 0, iteration: int = 0, idx: int = 0
    ) -> None:
        pass

    def on_validation_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        step: int = 0,
        iteration: int = 0,
        idx: int = 0,
    ) -> None:
        pass

    def on_validation_end(self, model: FastGenModel, iteration: int = 0, idx: int = 0) -> None:
        pass

    def on_save_checkpoint_start(self, model: FastGenModel, iteration: int = 0) -> None:
        pass

    def on_save_checkpoint_success(self, model: FastGenModel, iteration: int = 0, path: str = None) -> None:
        pass

    def on_save_checkpoint_end(self, model: FastGenModel, iteration: int = 0) -> None:
        pass

    def on_app_end(self, model: FastGenModel, iteration: int = 0) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass
