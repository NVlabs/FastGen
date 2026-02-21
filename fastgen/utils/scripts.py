# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import argparse
import torch
from omegaconf import OmegaConf

from fastgen.configs.config_utils import serialize_config
from fastgen.configs import config_utils
import fastgen.utils.logging_utils as logger
from fastgen.utils.basic_utils import get_batch_size_total
from fastgen.utils.distributed import ddp, world_size, is_rank0
from fastgen.utils.io_utils import set_env_vars
from fastgen.configs.config import BaseConfig


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("--config", default="configs.config", help="Path to the config file")
    parser.add_argument("--log_level", default="INFO", help="Log level (e.g. DEBUG, INFO)")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="""Modify config options at the end of the command. 
                                For Yacs configs, use space-separated "PATH.KEY VALUE" pairs.
                                For python-based LazyConfig, use "path.key=value".""",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    args = parser.parse_args()
    return args


def set_cuda_backend(deterministic: bool = True, benchmark: bool = True, tf32_enabled: bool = True):
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = tf32_enabled
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = tf32_enabled
    logger.critical(
        f"cuDNN deterministic: {deterministic}, " f"cuDNN benchmark: {benchmark}, " f"enable TF32: {tf32_enabled}"
    )


def setup(args: argparse.Namespace, evaluation: bool = False) -> BaseConfig:
    if hasattr(args, "log_level"):
        # set log level for logger (INFO, by default)
        logger.set_log_level(args.log_level)

    # Import the config from the python file
    config: BaseConfig = config_utils.import_config_from_python_file(args.config)
    if hasattr(args, "opts"):
        # Override the config with the command line arguments
        config = config_utils.override_config_with_opts(config, args.opts)
    # Update checkpointer save_dir
    config.trainer.checkpointer.save_dir = f"{config.log_config.save_path}/{config.trainer.checkpointer.save_dir}"

    # save config
    config_save_path = config.log_config.save_path
    if evaluation:
        config_save_path = os.path.join(config_save_path, config.eval.samples_dir)
    if is_rank0():
        serialize_config(config, return_type="file", path=config_save_path, filename="config.yaml")

    # Check for dryrun
    if getattr(args, "dryrun", False):
        logger.info("Dryrun")
        logger.info(OmegaConf.to_yaml(OmegaConf.load(f"{config_save_path}/config.yaml")))
        logger.info(f"config.yaml is saved at {config_save_path}")
        exit(0)

    # distributed setup
    if config.trainer.ddp or config.trainer.fsdp:
        # check if ddp is available
        if not torch.distributed.is_available():
            raise RuntimeError("Distributed training is not available, please check your PyTorch installation.")
        # initialize DDP
        ddp.init()
        logger.info(f"Distributed training initialized, world size: {world_size()}")
    else:
        logger.info("No DDP or FSDP parallelism")

    # Check if we can use memory-efficient FSDP meta init
    if config.model.fsdp_meta_init:
        if not config.trainer.fsdp:
            logger.warning("Ignoring fsdp_meta_init since FSDP is disabled.")
            config.model.fsdp_meta_init = False
        elif evaluation:
            logger.warning("Ignoring fsdp_meta_init for evaluation/inference.")
            config.model.fsdp_meta_init = False
        elif config.trainer.checkpointer.pretrained_ckpt_path:
            # TODO: this functionality is not implemented yet
            logger.warning("Ignoring fsdp_meta_init for loading pretrained checkpoint.")
            config.model.fsdp_meta_init = False

    # Global batch size
    if getattr(config.trainer, "batch_size_global", None) is not None:
        batch_size = config.dataloader_train.batch_size * world_size()
        accum_rounds = max(config.trainer.batch_size_global // batch_size, 1)
        new_batch_size_global = accum_rounds * batch_size
        if new_batch_size_global != config.trainer.batch_size_global:
            logger.critical(
                f"Requested global batch size {config.trainer.batch_size_global} is not divisible by current batch size {batch_size}. New global batch size will be {new_batch_size_global}."
            )

        if accum_rounds != config.trainer.grad_accum_rounds:
            logger.info(
                f"Changing gradient accumulation rounds from {config.trainer.grad_accum_rounds} to {accum_rounds} to match requested global batch size."
            )
            config.trainer.grad_accum_rounds = accum_rounds

    logger.critical(
        f"Global batch size: {get_batch_size_total(config)} (Batch size per GPU: {config.dataloader_train.batch_size}, Gradient accumulation rounds: {config.trainer.grad_accum_rounds}, World size: {world_size()})"
    )

    # Set up s3 environmental variables
    set_env_vars(config.trainer.checkpointer.s3_credential)

    # Set up CUDA backend
    set_cuda_backend(config.trainer.cudnn.deterministic, config.trainer.cudnn.benchmark, config.trainer.tf32_enabled)

    return config
