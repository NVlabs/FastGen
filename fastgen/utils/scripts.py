# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
import warnings

import torch

from fastgen.configs.config_utils import serialize_config
from fastgen.configs import config_utils
import fastgen.utils.logging_utils as logger
from fastgen.utils.basic_utils import get_batch_size_total
import fastgen.utils.distributed.ddp as ddp
from fastgen.utils.distributed import is_rank0, world_size
from fastgen.utils.distributed.collectives import raise_across_ranks
from fastgen.utils.env import set_env_vars
import fastgen.utils.s3 as s3
from fastgen.configs.config import BaseConfig


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument(
        "--config", required=True, help="Path to the config Python file, relative to the working directory"
    )
    parser.add_argument("--log_level", default="INFO", help="Log level (e.g. DEBUG, INFO)")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Config overrides at the end of the command, after a single '-': `- path.key=value ...` (Hydra syntax).",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    args = parser.parse_args()
    return args


def set_cuda_backend(
    deterministic: bool = True,
    benchmark: bool = True,
    tf32_matmul: bool = False,
    tf32_conv: bool = True,
    fp16_reduced_precision_reduction: bool = True,
    torch_compile_enabled: bool = False,
):
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    # Floating-point precision settings, set even at PyTorch's defaults: NGC images set
    # TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which makes TF32 the matmul default there.
    if torch_compile_enabled:
        # TODO: Remove this legacy-API branch once the maintained PyTorch/Inductor build no longer
        # reads allow_tf32 while lowering FP32 matmul/bmm and Triton convolution graphs.
        # The maintained PyTorch/Inductor build still reads the legacy flags while lowering some
        # graphs. Setting the new precision API first makes that internal read fail because PyTorch
        # forbids mixing both API families in one process. Use one family consistently until the
        # maintained image includes the upstream Inductor fix.
        # Suppress only the expected deprecation warning from this deliberate compatibility path;
        # other warnings raised while configuring the backends must remain visible.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Please use the new API settings to control TF32 behavior.*",
                category=UserWarning,
            )
            torch.backends.cudnn.allow_tf32 = tf32_conv
            torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
    else:
        torch.backends.cudnn.conv.fp32_precision = "tf32" if tf32_conv else "ieee"
        torch.backends.cuda.matmul.fp32_precision = "tf32" if tf32_matmul else "ieee"
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = fp16_reduced_precision_reduction
    logger.critical(
        f"cuDNN deterministic: {deterministic}, cuDNN benchmark: {benchmark}, TF32 matmul: {tf32_matmul}, "
        f"TF32 conv: {tf32_conv}, fp16 reduced-precision reduction: {fp16_reduced_precision_reduction}"
    )


def resolve_save_dir(config: BaseConfig) -> str:
    """Return the checkpoint directory, which the checkpointer uses as given.

    An S3 URI or absolute ``save_dir`` is kept, whatever ``use_s3`` is. Any other ``save_dir`` goes
    below the run's directory, so two runs never share a checkpoint prefix and ``resume`` cannot pick
    up another run's checkpoints: below the log path, or with ``use_s3`` below the run's S3 prefix.
    """
    log_config, checkpointer = config.log_config, config.trainer.checkpointer
    if s3.is_uri(checkpointer.save_dir) or os.path.isabs(checkpointer.save_dir):
        return checkpointer.save_dir
    if not checkpointer.use_s3:
        return os.path.join(log_config.save_path, checkpointer.save_dir)
    return os.path.join(
        checkpointer.s3_container, log_config.project, log_config.group, log_config.name, checkpointer.save_dir
    )


def load_config(config_file: str, opts: list[str] | None = None, evaluation: bool = False) -> BaseConfig:
    """Load a config file, apply command-line overrides, and prepare it for training or evaluation.

    An unnamed config takes the file's name. The config passes through Hydra even without overrides, so every entry
    point gets the same resolved config. Callers resolve the checkpoint directory with ``resolve_save_dir`` in their
    own working directory.
    """
    config: BaseConfig = config_utils.import_config_from_python_file(config_file)
    if config.log_config.name is None:
        config.log_config.name = os.path.splitext(os.path.basename(config_file))[0]
    config = config_utils.override_config_with_opts(config, opts)

    # Evaluation skips training-only networks and consumes raw prompts/images.
    config.model.inference = evaluation
    if evaluation:
        # Training configs may disable these when their conditions are precomputed.
        logger.info("Enabling preprocessors for evaluation")
        config.model.enable_preprocessors = True
    return config


def setup(args: argparse.Namespace, evaluation: bool = False, supports_fsdp: bool = False) -> BaseConfig:
    """Load and validate the config, join the distributed process group, and save the resolved config.

    Evaluation entry points run a full model replica on every rank, so ``trainer.fsdp`` is dropped with a
    warning. An entry point that FSDP-shards the model itself passes ``supports_fsdp=True`` to keep it.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`.*",
        category=UserWarning,
        module=r"huggingface_hub\.utils\._validators",
    )

    if hasattr(args, "log_level"):
        # set log level for logger (INFO, by default)
        logger.set_log_level(args.log_level)

    config = load_config(args.config, getattr(args, "opts", None), evaluation=evaluation)
    config.trainer.checkpointer.save_dir = resolve_save_dir(config)

    if config.trainer.ddp and config.trainer.fsdp:
        raise ValueError("trainer.ddp and trainer.fsdp are mutually exclusive; enable only one of them.")
    if evaluation and config.trainer.fsdp and not supports_fsdp:
        logger.warning(
            "Ignoring trainer.fsdp=True: evaluation does not FSDP-shard the model, so every rank runs a full "
            "model replica."
        )
        config.trainer.fsdp = False
    ddp_enabled = config.trainer.ddp
    fsdp_enabled = config.trainer.fsdp
    if fsdp_enabled and config.trainer.checkpointer.save_adapter_only:
        raise NotImplementedError("save_adapter_only is not supported with FSDP (DCP checkpointing)")

    # Check for dryrun
    if getattr(args, "dryrun", False):
        logger.info("Dryrun")
        logger.info(serialize_config(config, return_type="yaml"))
        sys.exit(0)

    # Always join a launcher-provided process group. FSDP and DDP training require one; evaluation never
    # wraps the model in DDP.
    distributed = ddp.init(required=fsdp_enabled or (ddp_enabled and not evaluation))
    if not evaluation and world_size() > 1 and not (ddp_enabled or fsdp_enabled):
        raise ValueError(
            f"{world_size()} ranks are running without trainer.ddp or trainer.fsdp, so each rank would train "
            "an independent replica; enable one of them or launch a single process."
        )

    logger.info(
        f"Distributed process group initialized={distributed}, world size={world_size()}, "
        f"DDP configured={ddp_enabled}, FSDP configured={fsdp_enabled}."
    )

    # Configure credentials and the Hugging Face cache before model/trainer initialization.
    set_env_vars(config.credentials_dir)

    # Meta initialization needs a distributed source rank, but does not require FSDP:
    # eager FastGenModule preparation materializes and synchronizes meta ranks too.
    if config.model.meta_init and not distributed:
        logger.warning("Ignoring meta_init since distributed execution is not active.")
        config.model.meta_init = False

    # Global batch size
    if config.trainer.batch_size_global is not None:
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

    # Set up CUDA backend
    set_cuda_backend(
        config.trainer.cudnn.deterministic,
        config.trainer.cudnn.benchmark,
        config.trainer.tf32_matmul,
        config.trainer.tf32_conv,
        config.trainer.fp16_reduced_precision_reduction,
        torch_compile_enabled=config.model.torch_compile_mode is not None,
    )

    # Save config (after all mutations above)
    config_save_path = config.log_config.save_path
    if evaluation:
        config_save_path = os.path.join(config_save_path, config.eval.samples_dir)
    with raise_across_ranks("config save"):
        if is_rank0():
            serialize_config(config, return_type="file", path=config_save_path, filename="config.yaml")

    return config
