# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, List, Optional, Dict

import copy
import attrs
from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from fastgen.configs.callbacks import WANDB_CALLBACK
from fastgen.configs.data import CIFAR10_Loader_Config
from fastgen.configs.net import EDM_CIFAR10_Config as EDMConfig
from fastgen.configs.opt import BaseOptimizerConfig, BaseSchedulerConfig
from fastgen.methods import FastGenModel


@attrs.define(slots=False)
class CuDNNConfig:
    # If set to True, cudnn will use deterministic cudnn functions for better reproducibility.
    deterministic: bool = False
    # If set to True, cudnn will benchmark several algorithms and pick the fastest one.
    benchmark: bool = True


@attrs.define(slots=False)
class LogConfig:
    # Project name
    project: str = "fastgen"
    # Experiment name
    group: str = "cifar10"
    # Run/job name
    name: str = "debug"
    # W&B mode, can be "online" or "disabled".
    wandb_mode: str = "online"
    # Wandb credential path
    wandb_credential: str = "./credentials/wandb_api.txt"

    # save path
    @property
    def save_path(self) -> str:
        return os.path.join(
            os.environ.get("FASTGEN_OUTPUT_ROOT", "FASTGEN_OUTPUT"), f"{self.project}/{self.group}/{self.name}"
        )


@attrs.define(slots=False)
class EvalConfig:
    # Number of samples to generate
    num_samples: int = 50000
    # Save a small batch of images
    save_images: bool = False
    # Minimum checkpoint to evaluate
    min_ckpt: int = 0
    # Maximum checkpoint to evaluate
    max_ckpt: int = 100000000
    # Directory to save samples
    samples_dir: str = "samples"


@attrs.define(slots=False)
class BaseCheckpointerConfig:
    save_dir: str = "checkpoints"
    use_s3: bool = False
    s3_container: str = "s3://checkpoints/fastgen"
    s3_credential: str = "./credentials/s3.json"

    # path to pretrained model (from previous stages),
    # it's used by loading fsdp/ddp trained ckpt to an fsdp/ddp pipeline
    pretrained_ckpt_path: str = ""
    # submodule names of model and keys of a pretrained checkpoint of the form {"model": {"submodule_key": ...}, ...}
    pretrained_ckpt_key_map: Dict[str, str] = {"net": "net"}


@attrs.define(slots=False)
class SampleTConfig:
    """Config for sampling t from a time distribution."""

    # time distribution (currently supporting: uniform, lognormal, polynomial, logitnormal, shift, and log_t)
    time_dist_type: str = "uniform"
    # mu in lognormal, logitnormal, and log_t distributions
    train_p_mean: float = -1.1
    # sigma in lognormal, logitnormal, and log_t distributions
    train_p_std: float = 2.0
    # shift value in shifted sampling (t_shifted = t * shift / (t * (shift - 1) + 1))
    shift: float = 5.0
    # lowest value in truncated range
    min_t: float = 0.002
    # highest value in truncated range
    max_t: float = 80.0
    # If provided, it is in the form [t_max, ..., 0] where len(t_list) needs to equal student_sample_steps + 1
    t_list: Optional[List[float]] = None
    # degree of freedom in log-transformed student-t distribution
    log_t_df: float = 0.01


@attrs.define(slots=False)
class BaseModelConfig:
    # Use factory functions to ensure each instance gets its own copy
    net: dict = attrs.field(factory=lambda: copy.deepcopy(EDMConfig))
    teacher: Optional[dict] = None  # Usually not used, only used when teacher is different from net (i.e. Causvid)

    # guidance scale for classifier-free guidance in teacher diffusion model. None means no guidance.
    guidance_scale: Optional[float] = None

    # enable skip layer guidance (currently only wan network has the skip_layers option in cfg)
    skip_layers: List[int] | None = None

    # optimizer and scheduler for the main net (i.e., one-step generator in DMD)
    net_optimizer: dict = attrs.field(factory=lambda: copy.deepcopy(BaseOptimizerConfig))
    net_scheduler: dict = attrs.field(factory=lambda: copy.deepcopy(BaseSchedulerConfig))

    # sampling t from a given distribution
    sample_t_cfg: SampleTConfig = attrs.field(factory=SampleTConfig)

    # shape of the input to the model (defaults to CIFAR-10)
    input_shape: List[int] = [3, 32, 32]
    # device ("cuda" or "cpu")
    device: str = "cuda"

    # enable gradient scaler
    grad_scaler_enabled: bool = False
    grad_scaler_init_scale: float = 65536.0
    grad_scaler_growth_interval: int = 2000

    # path to the pretrained teacher model ckpt
    pretrained_model_path: str = ""
    # path to the pretrained student net ckpt (if different from the teacher)
    pretrained_student_net_path: str = ""
    # initialize student from the above checkpoints (can be turned off to only load weights to the teacher)
    load_student_weights: bool = True

    # enable preprocessors in the model
    enable_preprocessors: bool = True

    # EMA for the main net (requires EMACallback)
    use_ema: Any = False

    # multistep generation if larger than 1 (default: single-step generation)
    student_sample_steps: int = 1
    # sampling type in multistep generation ('sde', 'ode')
    student_sample_type: str = "sde"

    # Enable memory-efficient model loading with meta device:
    # - Rank 0 loads pretrained weights normally
    # - Other ranks use torch.device("meta") for ZERO memory allocation (just metadata)
    # - FSDP materializes meta tensors and broadcasts weights from rank 0
    # This dramatically speeds up initialization for large models (14B+):
    # - Reduces RAM from N*model_size to 1*model_size
    # - Eliminates disk I/O contention (N parallel reads -> 1 read)
    # - Expected speedup: 30+ min -> <1 min for 14B models on 8 GPUs
    fsdp_meta_init: bool = False

    # whether to add the teacher model to the fsdp_dict
    add_teacher_to_fsdp_dict: bool = True

    # whether to find unused parameters in ddp
    # - can be turned off for improved performance
    # - however, it is required if the model has a discriminator or the net initializes unused modules (e.g., for logvar predictions)
    ddp_find_unused_parameters: bool = True

    # precision variables (choose from "float64", "float32", "bfloat16", or "float16")
    # (precision of the time steps is handled in the noise scheduler, defaulting to float64 for numerical stability)

    # precision for model/optimizer states and data - recommended to be float32 if precision_amp is not None
    precision: str = "float32"
    # AMP during training - if None or equal to precision, AMP is disabled during training.
    precision_amp: str | None = None
    # AMP during inference - if None or equal to precision, AMP is disabled during inference.
    precision_amp_infer: str | None = None
    # AMP during en-/decoding (e.g., for VAEs or text encoders) - if None or equal to precision, AMP is disabled during en-/decoding.
    precision_amp_enc: str | None = None
    # FSDP2 precision for parameter storage and gradient reduction.
    # If None, defaults to `precision`. Useful for storing params/grads in float32 while computing in bfloat16.
    precision_fsdp: str | None = None


@attrs.define(slots=False)
class BaseTrainerConfig:
    cudnn: CuDNNConfig = attrs.field(factory=CuDNNConfig)
    checkpointer: BaseCheckpointerConfig = attrs.field(factory=BaseCheckpointerConfig)

    # Callbacks configs.
    callbacks: dict = DictConfig(WANDB_CALLBACK)

    # save checkpoint frequency
    save_ckpt_iter: int = 5000
    # test on validation set frequency
    validation_iter: int = 1000
    # logging frequency
    logging_iter: int = 1000
    # maximum training iteration
    max_iter: int = 1000000
    # whether to visualize multistep teacher generation
    visualize_teacher: bool = False

    # Set the random seed.
    seed: int = 0
    # Validation seed
    val_seed: int | None = None
    # Resume
    resume: bool = True

    # DDP Parallelism
    ddp: bool = False
    # FSDP Parallelism
    fsdp: bool = False
    # Enable TensorFloat32 (convolution and matmul)
    tf32_enabled: bool = True

    # Number of gradient accumulation rounds
    grad_accum_rounds: int = 1

    # Global batch size (if not None, overrides grad_accum_rounds to match the specified batch size)
    batch_size_global: int | None = None

    # offload other modules to cpu during latent decoding
    offload_module_in_decoding: bool = False

    # apply cpu offloading in fsdp
    fsdp_cpu_offload: bool = False
    # Fallback minimum number of parameters for FSDP wrapping
    # (10M wraps large models into fairly small shards)
    # The FastGenNetwork should provide a fully_shard method that can be used to shard the network.
    # If we need to shard a different module, we fall back to an auto-sharding policy based on this value.
    fsdp_min_num_params: int = 10_000_000
    # Sharding group size for FSDP. If None, fully shard across all ranks.
    # If set, creates a 2D mesh with (replicate, shard) dimensions.
    fsdp_sharding_group_size: Optional[int] = None

    # global variables
    global_vars: Optional[dict] = None
    global_vars_val: List[dict | None] = [None]

    # augment config
    augment_pipe: Optional[DictConfig] = None


@attrs.define(slots=False)
class BaseConfig:
    # Log config.
    log_config: LogConfig = attrs.field(factory=LogConfig)

    # Trainer configs.
    trainer: BaseTrainerConfig = attrs.field(factory=BaseTrainerConfig)

    # Model configs.
    model: BaseModelConfig = attrs.field(factory=BaseModelConfig)
    model_class: DictConfig = L(FastGenModel)(config=None)

    # Data configs.
    dataloader_train: dict = CIFAR10_Loader_Config
    dataloader_val: Any = None

    # Eval configs.
    eval: EvalConfig = attrs.field(factory=EvalConfig)
