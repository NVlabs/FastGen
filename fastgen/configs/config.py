# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, List, Optional, Dict

import copy
import attrs
from omegaconf import DictConfig

from fastgen.configs.instantiate import LazyCall as L
from fastgen.utils.env import output_root
from fastgen.utils.guidance import GuidanceConfig
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.data import CIFAR10_Loader_Config
from fastgen.configs.net import EDM_CIFAR10_Config as EDMConfig
from fastgen.configs.opt import BaseOptimizerConfig, BaseSchedulerConfig
from fastgen.methods import FastGenModel
from fastgen.networks.noise_schedule import StudentGridConfig


@attrs.define
class CuDNNConfig:
    # If set to True, cudnn will use deterministic cudnn functions for better reproducibility.
    deterministic: bool = False
    # If set to True, cudnn will benchmark several algorithms and pick the fastest one.
    benchmark: bool = True


@attrs.define
class LogConfig:
    # Project name
    project: str = "fastgen"
    # Experiment name
    group: str = "cifar10"
    # Run/job name; None takes the name of the config file it is loaded from
    name: Optional[str] = None
    # W&B mode, can be "online", "offline", or "disabled".
    wandb_mode: str = "online"

    # save path
    @property
    def save_path(self) -> str:
        assert self.name is not None, "log_config.name is unset: set a name or load the config with load_config"
        return os.path.join(output_root(), f"{self.project}/{self.group}/{self.name}")


@attrs.define
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


@attrs.define
class BaseCheckpointerConfig:
    # Without S3, this path must be visible at the same location to every rank.
    # Multi-node checkpointing to separate node-local filesystems is unsupported.
    # The checkpointer requires an absolute path or s3:// URI; setup() places a relative one below the run.
    save_dir: str = "checkpoints"
    # Save only the adapter tensors (LoRA + extra modules); non-adapter modules save in full.
    save_adapter_only: bool = False
    # Use the shared S3 backend when no shared POSIX filesystem is available.
    use_s3: bool = False
    s3_container: str = "s3://checkpoints/fastgen"

    # Explicit monolithic .pth checkpoint file or DCP completion marker from a previous stage.
    # Empty or None means no pretrained checkpoint; either format can load into FSDP or DDP.
    pretrained_ckpt_path: Optional[str] = ""
    # submodule names of model and keys of a pretrained checkpoint of the form {"model": {"submodule_key": ...}, ...}
    pretrained_ckpt_key_map: Dict[str, str] = attrs.field(factory=lambda: {"net": "net"})


@attrs.define
class SampleTConfig:
    """Config for sampling t from a time distribution."""

    # time distribution (uniform, logitnormal, shifted, shifted_logitnormal, lognormal, log_t, or polynomial, on every
    # schedule), drawn over the schedule's own time t. So lognormal and polynomial are EDM's noise-level distributions
    # only on EDM: on RF (unshifted, where EDM's sigma is t / (1 - t)), logitnormal with the same train_p_mean and
    # train_p_std gives EDM's lognormal, up to how min_t and max_t apply.
    time_dist_type: str = "uniform"
    # mu in lognormal, logitnormal, and log_t distributions
    train_p_mean: float = -1.1
    # sigma in lognormal, logitnormal, and log_t distributions
    train_p_std: float = 2.0
    # shift value in shifted sampling (t_shifted = t * shift / (t * (shift - 1) + 1), applied within
    # [min_t, max_t])
    shift: float = 5.0
    # lowest value in truncated range (bounds the sampled time itself, after any shift); with DMD2 step generations
    # (student_generation="step") or sample_t_in_student_bucket, the sampled time's distance above the next
    # student-grid time t_{i+1}
    min_t: float = 0.002
    # highest value in truncated range (bounds the sampled time itself, after any shift)
    max_t: float = 80.0
    # degree of freedom in log-transformed student-t distribution
    log_t_df: float = 0.01
    # exponent of the polynomial distribution: t ** (1 / rho) is uniform between min_t ** (1 / rho) and
    # max_t ** (1 / rho), the polynomial spacing of EDM's sampling grids
    rho: float = 7.0

    def sample_t_kwargs(self) -> Dict[str, Any]:
        """The arguments of ``BaseNoiseSchedule.sample_t``, without the fields that a method's subclass adds."""
        return {field.name: getattr(self, field.name) for field in attrs.fields(SampleTConfig)}


@attrs.define
class NetAdapterConfig:
    """Optional adapter overlay on ``net``."""

    # None: train the full net. "lora": train an adapter on a frozen base. When
    # ``teacher`` is unset, it shares the net with adapters disabled; otherwise it is separate.
    # TODO: Enable FSDP with LoRA (currently only DDP is supported)
    type: Optional[str] = None
    # LoRA rank.
    rank: int = 64
    # LoRA alpha (scaling); defaults to the rank when None.
    alpha: Optional[int] = None
    # None -> the network's default targets; "all-linear" -> every Linear except the
    # network's extra trainable modules; or an explicit list of module-name suffixes.
    target_modules: Any = None
    dropout: float = 0.0
    # fresh modules that train alongside the adapter, e.g. ["r_embedder"].
    extra_trainable_modules: Optional[List[str]] = None
    # attribute of the net the adapter is injected into.
    base_name: str = "transformer"
    # attribute overrides applied during teacher (base-model) calls; None -> none.
    base_model_overrides: Optional[Dict[str, Any]] = None


@attrs.define
class BaseModelConfig:
    # Use factory functions to ensure each instance gets its own copy
    net: dict = attrs.field(factory=lambda: copy.deepcopy(EDMConfig))
    teacher: Optional[dict] = None  # Usually not used, only used when teacher is different from net (i.e. Causvid)
    # Optional adapter overlay on the net (e.g. a LoRA student on a frozen base).
    net_adapter: NetAdapterConfig = attrs.field(factory=NetAdapterConfig)

    # Unified guidance config (CFG / mix / rescale); a network's recipe may be a subclass with
    # its own terms.
    # Defaults to an empty ``GuidanceConfig()`` (``cfg.scale=None`` → no CFG).
    # Configs should be explicit: either set individual fields (e.g.
    # ``config.model.guidance_cfg.cfg.scale = 6.0``) or replace the whole
    # object with a network's recipe (e.g.
    # ``config.model.guidance_cfg = CogVideoX.default_guidance_cfg()``).
    # For multimodal models (e.g. LTX-2 audio+video), set per-modality scales
    # via ``cfg.scale={"video": 3.0, "audio": 7.0}``.
    guidance_cfg: GuidanceConfig = attrs.field(factory=GuidanceConfig)

    # optimizer and scheduler for the main net (i.e., one-step generator in DMD)
    net_optimizer: dict = attrs.field(factory=lambda: copy.deepcopy(BaseOptimizerConfig))
    net_scheduler: dict = attrs.field(factory=lambda: copy.deepcopy(BaseSchedulerConfig))

    # sampling t from a given distribution
    sample_t_cfg: SampleTConfig = attrs.field(factory=SampleTConfig)
    # Scheduler-owned policy for constructing a distilled student's time grid.
    student_grid_cfg: StudentGridConfig = attrs.field(factory=StudentGridConfig)

    # shape of the input to the model (defaults to CIFAR-10)
    input_shape: List[int] = attrs.field(factory=lambda: [3, 32, 32])
    # device ("cuda" or "cpu")
    device: str = "cuda"

    # enable gradient scaler
    grad_scaler_enabled: bool = False
    grad_scaler_init_scale: float = 65536.0
    grad_scaler_growth_interval: int = 2000

    # path to the pretrained teacher model ckpt
    pretrained_model_path: Optional[str] = None
    # path to the pretrained student net ckpt (if different from the teacher)
    pretrained_student_net_path: Optional[str] = None
    # initialize student from the above checkpoints (can be turned off to only load weights to the teacher)
    load_student_weights: bool = True

    # enable preprocessors in the model
    enable_preprocessors: bool = True

    # EMA for the main net (requires EMACallback)
    use_ema: Any = False

    # additional reward models
    reward: Any = None
    optimize_reward: bool = False

    # sampling type in multistep generation, see ``BaseNoiseSchedule.step``: 'ode' (DDIM), 'sde' (fresh noise),
    # 'ddpm' (ancestral), or a float eta >= 0 (0 is 'ode', 1 is 'ddpm'); methods raise on types their student
    # rollout does not support (``FastGenModel.student_sample_types``)
    student_sample_type: str | float = "sde"

    # Enable memory-efficient construction and loading for supported ``FastGenModule`` roots:
    # - Rank 0 constructs and loads weights normally
    # - Other ranks construct on torch.device("meta") with ZERO storage allocation
    # - ``prepare_module`` materializes and synchronizes those ranks from rank 0, with or without FSDP
    # Components that do not support the FastGen meta-init lifecycle are constructed normally.
    # This dramatically speeds up initialization for large models (14B+):
    # - Reduces RAM from N*model_size to 1*model_size
    # - Eliminates disk I/O contention (N parallel reads -> 1 read)
    # - Expected speedup: 30+ min -> <1 min for 14B models on 8 GPUs
    # A monolithic ``trainer.checkpointer.pretrained_ckpt_path`` forgoes both savings: every rank reads the file
    # and materializes the full model before wrapping. DCP checkpoints load directly into FSDP shards.
    meta_init: bool = False

    # Frozen components to shard in addition to the model_dict roots. EMA copies are model_dict roots
    # too: they are sharded like the live network and share its ``precision_opt`` storage. Entries are names in
    # FastGenModel.module_dict, e.g. "teacher", "cm_teacher", or "text_encoder". Do not list
    # model_dict entries; they are selected automatically. Extras must be ``FastGenModule`` roots
    # and are not checkpointed. They are stored in ``precision_model`` and use it as their FSDP
    # parameter/nominal compute dtype rather than the ``precision_opt`` storage of trainable
    # components; active AMP may still select another dtype for eligible operators.
    extra_fsdp_modules: List[str] = attrs.field(factory=lambda: ["teacher"])

    # whether to find unused parameters in ddp
    # - can be turned off for improved performance
    # - however, it is required if the model has a discriminator or the net initializes unused modules (e.g., for logvar predictions)
    ddp_find_unused_parameters: bool = True

    # Skip training-only sub-networks (fake_score, discriminator, reward, etc.) during build_model.
    # Set to True in inference scripts to avoid allocating memory for modules that are never used.
    inference: bool = False

    # torch.compile mode for the training networks ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
    # applied in apply_torch_compile. None disables torch.compile.
    torch_compile_mode: Optional[str] = None

    # precision variables (choose from "float64", "float32", "bfloat16", or "float16")
    # (precision of the time steps is handled in the noise scheduler, defaulting to float64 for numerical stability)

    # precision for data and method/sampling arithmetic; loss kernels may promote to a wider dtype
    precision: str = "float32"
    # precision of primary neural inputs, nominal execution without autocast, and storage of frozen
    # state (teachers, preprocessors, and ordinary frozen parameters). Active AMP may choose another
    # dtype for eligible operators; strict compute regions must own their local autocast policy. If
    # None, follows `precision`.
    precision_model: str | None = None
    # storage precision of optimized components (master weights and Adam state) and live EMA weights.
    # If None, follows `precision_model`. Recommend float32 for adapter training and for bfloat16
    # compute. Without FSDP, ordinary modules usually need compatible AMP or network-owned dtype
    # handling when this differs from `precision_model`; otherwise an incompatible operation may
    # fail. FP16 optimized storage runs without loss scaling and is discouraged. LoRA networks keep
    # their frozen base weights in `precision_model`.
    precision_opt: str | None = None
    # AMP during training (supported dtypes are device/backend-specific) - if None, AMP is disabled.
    precision_amp: str | None = None
    # torch's autocast weight cache (``torch.autocast(cache_enabled=...)``) for training and validation
    # AMP; standalone inference and encoder/decoder autocast do not use it. Nested regions inherit it.
    # Off by default: the cache memoizes the autocast copy of every trainable weight
    # for the whole outermost region and never records whether the copy was made with gradients enabled,
    # so any no_grad use of trainable weights before their gradient forward silently drops those
    # gradients, a copy leaks into nested regions of another dtype, and the copies stay resident through
    # FSDP2 resharding and activation checkpointing. Each weight is cast once per forward anyway; only a
    # weight reused within the region saves one forward cast and one backward gradient cast per extra
    # use, and its reuse gradients are then summed in the autocast dtype instead of the weight dtype.
    autocast_cache_enabled: bool = False
    # AMP during inference (supported dtypes are device/backend-specific) - if None, AMP is disabled.
    precision_amp_infer: str | None = None
    # AMP during en-/decoding (e.g., for VAEs or text encoders) - if None, AMP is disabled during en-/decoding.
    precision_amp_enc: str | None = None
    # FSDP2 gradient-reduce dtype. If None, defaults to ``precision_opt``. Setting this to
    # ``"bfloat16"`` while keeping ``precision_opt="float32"`` keeps the FSDP master + Adam
    # state in fp32 while reducing gradients in bf16.
    precision_fsdp_reduce: str | None = None


@attrs.define
class BaseTrainerConfig:
    cudnn: CuDNNConfig = attrs.field(factory=CuDNNConfig)
    checkpointer: BaseCheckpointerConfig = attrs.field(factory=BaseCheckpointerConfig)

    # Callbacks configs.
    callbacks: dict = attrs.field(factory=lambda: DictConfig(BASE_CALLBACKS))

    # Auto-resume spec, instantiated by the Trainer. ``None`` (default) -> ``NoOpAutoResume``.
    # Set it to a LazyCall spec building an ``AutoResumeInterface`` to enable cluster auto-resume.
    auto_resume: Optional[DictConfig] = None

    # save checkpoint frequency
    save_ckpt_iter: int = 5000
    # test on validation set frequency
    validation_iter: int = 1000
    # logging frequency
    logging_iter: int = 1000
    # Total optimizer steps, including any restored from a checkpoint; excludes accumulation microbatches.
    # A fresh run executes iterations 1 through max_iter, saving the final state at max_iter.
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
    # FP32 precision of cuBLAS matmuls and cuDNN convolutions: TensorFloat32 if True, IEEE float32 if False.
    # The defaults are PyTorch's, which references that leave these settings alone run with.
    tf32_matmul: bool = False
    tf32_conv: bool = True
    # Allow reduced-precision reductions in cuBLAS fp16 matmuls (PyTorch's default).
    fp16_reduced_precision_reduction: bool = True

    # Number of gradient accumulation rounds
    grad_accum_rounds: int = 1

    # Global batch size (if not None, overrides grad_accum_rounds to match the specified batch size)
    batch_size_global: int | None = None

    # Temporarily offload registered non-decoder components during latent decoding (not supported with FSDP)
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
    global_vars_val: List[dict | None] = attrs.field(factory=lambda: [None])

    # augment config
    augment_pipe: Optional[DictConfig] = None


@attrs.define
class BaseConfig:
    # Log config.
    log_config: LogConfig = attrs.field(factory=LogConfig)

    # Trainer configs.
    trainer: BaseTrainerConfig = attrs.field(factory=BaseTrainerConfig)

    # Model configs.
    model: BaseModelConfig = attrs.field(factory=BaseModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(FastGenModel)(config=None))

    # Data configs.
    dataloader_train: dict = attrs.field(factory=lambda: CIFAR10_Loader_Config.copy())
    dataloader_val: Any = None

    # Eval configs.
    eval: EvalConfig = attrs.field(factory=EvalConfig)

    # Folder with the optional credential files: wandb_api.txt, hf_token.txt, vlm_api_token.txt, and s3.json.
    credentials_dir: str = "./credentials"
