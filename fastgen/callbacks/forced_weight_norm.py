# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING


from fastgen.callbacks.callback import Callback
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


class ForcedWeightNormCallback(Callback):
    def __init__(self) -> None:
        self._warned = False

    def on_training_accum_step_begin(
        self,
        model: FastGenModel,
        *args,
        **kwargs,
    ) -> None:
        if hasattr(model.net, "forced_weight_normalization"):
            model.net.forced_weight_normalization()
        elif not self._warned:
            self._warned = True
            logger.warning(
                "Enabled ForcedWeightNormCallback but model.net does not have the forced_weight_normalization method."
            )
