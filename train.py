# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The training entry script for the FastGen project. Works for both DDP and FSDP training.
"""

import argparse

from fastgen.configs.config import BaseConfig
from fastgen.trainer import Trainer
from fastgen.configs.instantiate import instantiate
import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed.collectives import clean_up, synchronize
from fastgen.utils.scripts import parse_args, setup


def main(config: BaseConfig):
    # initiate the trainer first so that ``set_random_seed(..., by_rank=False)``
    # in ``Trainer.__init__`` runs *before* model construction. This makes the
    # global RNG identical across ranks during model construction, so every
    # rank builds the same initial weights instead of diverging on PyTorch's
    # process-default seed.
    logger.info("Initializing trainer...")
    fastgen_trainer = Trainer(config)
    logger.success("Trainer initialized successfully")
    synchronize()

    # initiate the model under the seed established by ``Trainer.__init__``.
    # ``config=config.model`` is passed as an explicit kwarg so the attrs
    # ``ModelConfig`` reaches the model unchanged (avoids OmegaConf coercing
    # the subtree to ``DictConfig`` on assignment into ``model_class``).
    model = instantiate(config.model_class, config=config.model)
    synchronize()

    # Start training
    fastgen_trainer.run(model)
    synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    args = parse_args(parser)
    config = setup(args)

    main(config)

    clean_up()
    logger.info("Training finished.")
