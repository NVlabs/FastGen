# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Dict, Any, Optional, TYPE_CHECKING, Callable, Tuple
import contextlib

import torch

from fastgen.configs.opt import get_scheduler
from fastgen.networks.lora_utils import SharedBaseTeacher
from fastgen.networks.network import FastGenNetwork, FastGenModule, validate_latent_compression
from fastgen.networks.noise_schedule import SAMPLE_TYPES, check_sample_type
from fastgen.configs.instantiate import instantiate
from fastgen.utils.cache_utils import cache
from fastgen.utils.distributed.collectives import raise_across_ranks, synchronize, validate_object_across_ranks
from fastgen.utils.io_utils import load_checkpoint_state_dict
import fastgen.utils.logging_utils as logger
import fastgen.utils.basic_utils as basic_utils
from fastgen.utils.module import load_module_state_dict, validate_module_roots
from fastgen.utils.tensor import precision_to_dtype
from fastgen.utils.tree_utils import (
    TensorOrDict,
    assert_same,
    per_modality,
    get_batch_size,
    get_device,
    cast_like,
    randn_like,
    to,
    detach,
)
from fastgen.utils.distributed import is_rank0

if TYPE_CHECKING:
    from fastgen.configs.config import BaseModelConfig
    from fastgen.utils.distributed.fsdp import FSDPContext


class FastGenModel(torch.nn.Module):
    # The ``config.student_sample_type`` values this method's student rollout supports, see ``check_sample_type``.
    student_sample_types: Tuple[Any, ...] = SAMPLE_TYPES
    # Batch fields consumed as method state. Subclasses extend this for method-specific inputs.
    state_fields: frozenset[str] = frozenset({"real"})

    def __init__(self, config: BaseModelConfig):
        """FastGenModel class for implementing training interface for all fastgen networks.

        Args:
            config (BaseModelConfig): The configuration for the FastGen model
        """
        super().__init__()
        self.config = config
        # Registry of physical FastGenModule roots whose device, precision, and FSDP lifecycle is
        # managed independently. Entries are added through register_modules for model_dict,
        # separate teachers, method-specific roots, and lazy preprocessors.
        self.module_dict: Dict[str, FastGenModule] = {}

        # device
        self.device = torch.device(config.device)
        if self.device.type not in ["cuda", "cpu"]:
            raise ValueError(f"FastGenModel models only support cuda and cpu devices, got {self.device.type}")

        # precision and autocast
        self._resolve_precisions(
            precision=self.config.precision,
            precision_model=self.config.precision_model,
            precision_opt=self.config.precision_opt,
            precision_amp=self.config.precision_amp,
            precision_amp_infer=self.config.precision_amp_infer,
            precision_amp_enc=self.config.precision_amp_enc,
            precision_fsdp_reduce=self.config.precision_fsdp_reduce,
        )

        # input shape (values are updated after build_model via net.infer_shape)
        self.input_shape = config.input_shape

        # define the name of the EMA networks to use
        use_ema = config.use_ema
        if isinstance(use_ema, bool):
            use_ema = ["ema"] if use_ema else []
        if not all(isinstance(name, str) and name.startswith("ema") for name in use_ema):
            raise ValueError(f"use_ema must be a bool or a list of strings starting with `ema`, got {use_ema}.")
        self.use_ema = use_ema

        # student rollout policy
        check_sample_type(
            config.student_sample_type, self.student_sample_types, name=f"{type(self).__name__} student_sample_type"
        )

        # instantiate all necessary nets and submodules
        self.build_model()
        self.register_modules(**self.model_dict)

        # Update input_shape via the network
        if hasattr(self.net, "infer_shape"):
            self.input_shape = self.net.infer_shape(self.input_shape)
        logger.info(f"Input shape is {self.input_shape}.")

    def register_modules(self, **modules: FastGenModule) -> None:
        """Register independent FastGen roots under unique names, without changing existing owners."""
        for name, module in modules.items():
            assert isinstance(module, FastGenModule), f"Module {name!r} must be a FastGenModule."
            if name in self.module_dict and self.module_dict[name] is not module:
                raise ValueError(f"Cannot register module {name!r}: module_dict already contains a different module.")
        validate_module_roots(self.module_dict | modules)
        self.module_dict.update(modules)

    @property
    def uses_fsdp(self) -> bool:
        """Whether a registered logical root completed FSDP preparation."""
        return any(module._is_fsdp for module in self.module_dict.values())

    def reshard_fsdp_modules(self, preprocessors_only: bool = False) -> None:
        """Release retained FSDP parameters below selected logical roots.

        Args:
            preprocessors_only: Select only the active host's preprocessor roots instead of every
                registered model root.
        """
        if preprocessors_only:
            roots = self.net_preprocessors.preprocessors.values()
        else:
            roots = self.module_dict.values()
        roots = tuple(root for root in roots if root._is_fsdp)
        if not roots:
            return

        # Avoid adding the FSDP utility stack to this module's eager import path.
        from fastgen.utils.distributed.fsdp import reshard_fsdp_modules

        reshard_fsdp_modules(roots)

    def prepare_batch(
        self,
        data: dict[str, Any],
        augment_pipe: Callable | None = None,
        inference_mode: bool = False,
        materialize_real: bool = True,
    ) -> dict[str, Any]:
        """Turn one dataloader batch into method state and network conditioning.

        The model owns placement, method-state precision, execution scopes, and
        preprocessor lifecycle. Its active preprocessor network owns the
        representation-specific ``real_raw`` encoding and complete conditioning
        builder.

        Args:
            data: Per-batch input mapping. The mapping is not mutated.
            augment_pipe: Optional data-domain augmenter, run before encoder autocast.
            inference_mode: Use strict ``torch.inference_mode`` instead of
                training-safe ``torch.no_grad``.
            materialize_real: Encode a missing ``real`` from ``real_raw``. Condition
                building always runs and may still consume ``real_raw``.

        Returns:
            A new mapping with fields in ``state_fields`` cast to ``precision`` and
            network-owned conditioning updates; ``real_raw`` is removed.
        """
        mode = torch.inference_mode if inference_mode else torch.no_grad
        net = self.net_preprocessors

        with mode():
            data = to(data, device=self.device)
            if augment_pipe is not None:
                data = augment_pipe(data)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.precision_amp_enc,
                enabled=self.precision_amp_enc is not None,
            ):
                if materialize_real and data.get("real") is None and data.get("real_raw") is not None:
                    data["real"] = net.encode_real(data["real_raw"])

                data = {
                    key: to(value, dtype=self.precision) if key in self.state_fields else value
                    for key, value in data.items()
                }
                data.update(
                    net.build_conditioning(
                        data,
                        input_shape=self.input_shape,
                        precision=self.precision,
                    )
                )
                data.pop("real_raw", None)

            # The builder runs after state placement/casting, so state it returns must already match both.
            assert_same(
                dtype=self.precision,
                device=self.device,
                **{key: data[key] for key in self.state_fields & data.keys()},
            )

        # Keep generic FSDP roots materialized across every VAE/text call in this
        # phase, then release them before the main network phase.
        self.reshard_fsdp_modules(preprocessors_only=True)
        return data

    def _setup_ema(self):
        """Initialize EMA networks. Only call during build_model(), before checkpoint loading."""
        for name in self.use_ema:
            logger.info(f"Initializing EMA network {name}")
            with self._get_meta_init_context():
                ema = instantiate(self.config.net, student_grid_cfg=self.config.student_grid_cfg)
            setattr(self, name, ema)
            self._load_state_from(ema, self.net, f"EMA network {name}")

    def _load_state_from(
        self, target: torch.nn.Module, source: torch.nn.Module, description: str, strict: bool = True
    ) -> None:
        """Initialize ``target`` from ``source``'s state.

        Under meta initialization only rank 0 holds real state, so only it copies; its failure is
        raised on every rank instead of leaving the peers waiting in their next collective.
        """
        with raise_across_ranks(description):
            if not self.config.meta_init or is_rank0():
                load_module_state_dict(
                    target, source.state_dict(), strict=strict, require_any=True, description=description
                )

    def _get_meta_init_context(self, meta_init: bool = None):
        """Get the rank-aware context manager for meta initialization.

        When meta_init is enabled, non-rank-0 processes use meta device
        for memory-efficient loading. Rank 0 loads weights normally; module
        preparation later materializes and synchronizes the other ranks,
        either through FSDP or eager ``FastGenModule`` preparation.

        Args:
            meta_init: Whether to use meta initialization. If None, uses
                self.config.meta_init.
        """
        if meta_init is None:
            meta_init = self.config.meta_init
        use_meta = meta_init and not is_rank0()
        if use_meta:
            return torch.device("meta")
        return contextlib.nullcontext()

    def _resolve_precisions(
        self,
        precision: str = "float32",
        precision_model: str | None = None,
        precision_opt: str | None = None,
        precision_amp: str | None = None,
        precision_amp_infer: str | None = None,
        precision_amp_enc: str | None = None,
        precision_fsdp_reduce: str | None = None,
    ) -> None:
        """Resolve the method-state, model, storage, and automatic mixed precision (AMP) dtypes.

        All precision arguments are strings that are mapped to torch dtypes according to PRECISION_MAP:
            "float16" -> torch.float16
            "bfloat16" -> torch.bfloat16
            "float32" -> torch.float32
            "float64" -> torch.float64
        AMP dtype support is device/backend-specific; use None to disable autocast.

        Note that the precision of the time steps is handled in the noise scheduler (defaulting to float64 for
        numerical stability).

        Args:
            precision: Precision for data, trajectories, and method arithmetic. Loss kernels may
                promote their operands and return a wider dtype.
            precision_model: Precision of primary neural inputs, nominal execution when autocast is
                disabled, and storage of frozen state. Active AMP may select another dtype for
                eligible operators; a network requiring strict operator precision must own its local
                autocast policy. Output dtype follows each module's boundary contract. If None,
                follows `precision`.
            precision_opt: Storage precision of optimized modules (master weights and optimizer
                state) and live EMA weights. If None, follows `precision_model`. Without FSDP,
                ordinary modules usually need compatible AMP or network-owned dtype handling when
                this differs from `precision_model`; otherwise an incompatible operation may fail.
                FP16 optimized storage is allowed but runs without loss scaling and is discouraged.
            precision_amp: Autocast precision for eligible training operations. If None, AMP is disabled.
            precision_amp_infer: Autocast precision for eligible inference operations. If None, AMP is disabled.
            precision_amp_enc: Precision for AMP en-/decoder (e.g., for VAEs or text encoders).
                If None, AMP is disabled during en-/decoding.
            precision_fsdp_reduce: Precision for FSDP2 gradient reduction. If None, defaults to
                `precision_opt`. Set this independently to keep the master / Adam state in one dtype
                while reducing gradients in another.
        """

        # Method state, neural execution, and optimized storage are intentionally independent.
        self.precision = precision_to_dtype(precision)
        self.precision_model = precision_to_dtype(precision_model, default=self.precision)
        self.precision_opt = precision_to_dtype(precision_opt, default=self.precision_model)
        self.precision_fsdp_reduce = precision_to_dtype(precision_fsdp_reduce, default=self.precision_opt)

        # AMP is disabled when unset; kept even when it matches `precision_model` (mixed params)
        self.precision_amp = precision_to_dtype(precision_amp)  # training
        self.precision_amp_infer = precision_to_dtype(precision_amp_infer)  # inference
        self.precision_amp_enc = precision_to_dtype(precision_amp_enc)  # en-/decoders (VAEs, text encoders)

        if self.precision_amp is not None and self.precision_opt != torch.float32:
            logger.warning(
                f"Autocast to {self.precision_amp} is enabled with optimized parameters stored in "
                f"{self.precision_opt}. It is recommended to keep them in `float32`."
            )

        logger.critical(
            f"Method-state precision: {self.precision}. Model precision: {self.precision_model}. "
            f"Optimization precision: {self.precision_opt}. "
            f"AMP training precision: {self.precision_amp}. "
            f"AMP en-/decoder precision: {self.precision_amp_enc}. AMP inference precision: {self.precision_amp_infer}. "
            f"FSDP reduce precision: {self.precision_fsdp_reduce}."
        )

    def prepare_module(
        self,
        module: Any,
        name: str,
        fsdp_context: Optional["FSDPContext"] = None,
    ) -> None:
        """Prepare a module for the role its name selects, with this model's device and precisions.

        ``model_dict`` roots are trainable outside inference and store in ``precision_opt``; a live EMA
        shares that storage but stays frozen. Every other name, preprocessors included, is frozen at
        the model precision; ``extra_fsdp_modules`` only adds sharding. Each module is prepared once.
        Once FSDP setup has started, a requested extra cannot first be prepared without its active context.
        """
        module_type = type(module).__name__
        if not isinstance(module, FastGenModule):
            raise TypeError(f"Module {module_type} must be a FastGenModule.")

        is_model = name in self.model_dict
        requested_extra = name in self.config.extra_fsdp_modules
        if requested_extra and fsdp_context is None and self.uses_fsdp and not module._is_fsdp:
            raise RuntimeError(
                f"extra_fsdp_modules requested {name!r}, but it is being prepared after FSDP setup "
                "without the active FSDP context. Late FSDP roots are unsupported; construct and "
                "prepare the module during the initial FSDP setup or remove it from extra_fsdp_modules."
            )

        inference = self.config.inference
        trainable = not inference and is_model and name not in self.use_ema
        use_fsdp = fsdp_context is not None and (is_model or requested_extra)

        description = f"{name} ({module_type}; trainable={trainable}, fsdp={use_fsdp})"
        logger.debug(f"Preparing {description}.")
        module.prepare_module(
            device=self.device,
            precision_model=self.precision_model,
            precision_opt=self.precision_opt if is_model and not inference else None,
            trainable=trainable,
            fsdp_context=fsdp_context if use_fsdp else None,
        )

    @property
    def teacher_config(self) -> dict:
        teacher_config = self.config.net
        if self.config.teacher is not None:
            logger.critical("Using teacher config (usually due to teacher architecture being different from student)")
            teacher_config = self.config.teacher
        return teacher_config

    @property
    def separate_teacher(self) -> Optional[torch.nn.Module]:
        """Return the independently instantiated teacher, if any."""
        teacher = getattr(self, "teacher", None)
        return None if isinstance(teacher, SharedBaseTeacher) else teacher

    def _init_net_adapter(self):
        """Realize ``config.net_adapter`` and inject it into the network.

        ``prepare_module`` later applies the adapter-aware trainability policy.
        """
        adapter = self.config.net_adapter
        if adapter.type is None:
            return
        if adapter.type != "lora":
            raise ValueError(f"Unknown net_adapter.type: {adapter.type!r} (expected None or 'lora')")
        if not hasattr(self.net, "init_lora"):
            raise ValueError(f"net_adapter.type='lora' requires a LoRA-capable net, got {type(self.net).__name__}")
        if adapter.rank is None or adapter.rank <= 0:
            raise ValueError(f"net_adapter.type='lora' requires rank > 0, got {adapter.rank}")
        if self.use_ema:
            # EMA copies are built from config.net (a clean base) and cannot represent the adapter.
            raise ValueError("use_ema is not supported with a net_adapter")
        if self.config.meta_init:
            # PEFT cannot inject into the meta-initialized bases of ranks other than 0; reject on every rank.
            raise ValueError("meta_init is not supported with a net_adapter")
        student_path = self.config.pretrained_student_net_path
        if self.config.load_student_weights:
            # Nets that don't self-load (e.g. non-diffusers bases): initialize the
            # vanilla base BEFORE injection, while checkpoint keys still match.
            model_path = student_path or self.config.pretrained_model_path
            if model_path is not None:
                FastGenModel._load_pretrained_model(self.net, model_path, meta_init=self.config.meta_init)
        elif student_path is not None:
            logger.warning("Ignoring `pretrained_student_net_path` since `load_student_weights` is False.")
        # Set before injection: "all-linear" excludes these paths.
        self.net.lora_base_name = adapter.base_name
        self.net.lora_extra_trainable_modules = tuple(adapter.extra_trainable_modules or ())
        self.net.base_model_overrides = dict(adapter.base_model_overrides or {})
        self.net.init_lora(
            rank=adapter.rank,
            alpha=adapter.alpha,
            target_modules=adapter.target_modules,
            dropout=adapter.dropout,
        )

    def build_teacher(self):
        # Without an explicit teacher, use a parameter-less adapters-off view of the student.
        if self.config.net_adapter.type is not None and self.config.teacher is None:
            if self.config.load_student_weights and self.config.pretrained_student_net_path is not None:
                # The distinct student checkpoint was loaded before adapter injection. Sharing that
                # base with the teacher would silently replace the teacher's intended base weights.
                raise ValueError("pretrained_student_net_path with a net_adapter requires an explicit separate teacher")
            logger.info("Building shared-base teacher (student net with adapters disabled)")
            if self.config.pretrained_model_path is not None and not self.config.load_student_weights:
                logger.warning(
                    "pretrained_model_path is ignored (load_student_weights=False); "
                    "the base net loads its own weights."
                )
            self.teacher = SharedBaseTeacher(self.net)
            return

        # instantiate the teacher
        logger.info("Instantiating the teacher")
        meta_init_teacher = "teacher" in self.config.extra_fsdp_modules and self.config.meta_init
        logger.info(f"build_teacher: meta_init={self.config.meta_init}, meta_init_teacher={meta_init_teacher}")
        with self._get_meta_init_context(meta_init_teacher):
            self.teacher = instantiate(self.teacher_config)
        logger.info(f"Teacher guidance config: {self.config.guidance_cfg}")

        # load pre-trained teacher model
        model_path = self.config.pretrained_model_path
        if model_path is not None:
            FastGenModel._load_pretrained_model(self.teacher, model_path, meta_init=meta_init_teacher)
        self.register_modules(teacher=self.teacher)

    def load_student_weights_and_ema(self):
        if self.config.net_adapter.type is not None:
            # The base already loaded pre-injection (see _init_net_adapter); the
            # adapted key layout cannot load a vanilla checkpoint.
            return
        # external network ckpt different from teacher (e.g. pretrained kd, pretrained self-forcing, etc.)
        has_student_path = self.config.pretrained_student_net_path is not None
        # pretrained teacher model ckpt
        has_model_path = self.config.pretrained_model_path is not None

        if self.config.load_student_weights:
            logger.info("Loading student weights")
            if has_student_path:
                FastGenModel._load_pretrained_model(
                    self.net, self.config.pretrained_student_net_path, meta_init=self.config.meta_init
                )
            elif has_model_path:
                teacher = self.separate_teacher
                if teacher is not None:
                    logger.info("Loading student weights from teacher weights")
                    self._load_state_from(self.net, teacher, "Student network initialized from teacher", strict=False)
                else:
                    FastGenModel._load_pretrained_model(
                        self.net, self.config.pretrained_model_path, meta_init=self.config.meta_init
                    )
            else:
                logger.warning(
                    "No student weights specified. This might be intended if the student initialization already "
                    "loads pretrained weights (e.g., from diffusers)."
                )

        elif has_student_path:
            logger.warning("Ignoring `pretrained_student_net_path` since `load_student_weights` is False.")
        elif has_model_path and getattr(self, "teacher", None) is None:
            logger.warning("Ignoring `pretrained_model_path` since `load_student_weights` is False.")

        # load EMA weights
        self._setup_ema()

    def build_model(self):
        # instantiate the generator network
        logger.info("Instantiating the generator network")
        with self._get_meta_init_context():
            self.net = instantiate(self.config.net, student_grid_cfg=self.config.student_grid_cfg)
        self._init_net_adapter()

        # initialize the reward function (training only)
        if self.config.reward is not None and not self.config.inference:
            self.reward = instantiate(self.config.reward)

    def init_preprocessors(self, fsdp_context: Optional["FSDPContext"] = None) -> FastGenNetwork:
        """Build the host's missing preprocessors and prepare all of them as frozen modules.

        Call once per host: like every module, a preprocessor can only be prepared once.

        Returns:
            The host network selected by ``net_preprocessors``.
        """
        with raise_across_ranks("preprocessor initialization"):
            net = self.net_preprocessors
            preprocessors = {}
            if self.config.enable_preprocessors:
                net.init_preprocessors()
                preprocessors = dict(net.preprocessors)
                # Register before preparation so FSDP roots are tracked even after a setup failure.
                self.register_modules(**preprocessors)
                if "vae" in preprocessors:
                    validate_latent_compression(net, preprocessors["vae"])
        validate_object_across_ranks(tuple(preprocessors), operation="preprocessor preparation plan")

        for name, module in preprocessors.items():
            self.prepare_module(module, name=name, fsdp_context=fsdp_context)

        sharded = tuple(name for name, module in preprocessors.items() if module._is_fsdp)
        if sharded:
            logger.warning(
                f"FSDP-sharded preprocessors {sharded} require synchronized preprocessing and callback decode "
                "control flow within each shard group. Input samples may differ, but optional-field availability, "
                "raw-versus-cached inputs, augmentation decisions, and callback media schemas (including zero/nonzero "
                "work) must produce the same preprocessor calls in the same order on every participating rank; "
                "otherwise FSDP collectives may deadlock."
            )
        return net

    def apply_torch_compile(self):
        """Compile the runnable networks in place with torch.compile.

        No-op when ``config.torch_compile_mode`` is None. Otherwise the mode is
        passed to ``torch.nn.Module.compile``; supported values are
        ``"default"``, ``"reduce-overhead"``, ``"max-autotune"``, and
        ``"max-autotune-no-cudagraphs"``.

        Called after DDP/FSDP wrapping (trainer) or after the root pruning in
        ``setup_inference_modules`` (inference) so compilation composes with the
        distributed wrappers and only targets the net(s) that actually run.
        """
        mode = self.config.torch_compile_mode
        if mode is None:
            return

        # Training: skip EMA copies (not run during training). Inference: skip
        # nothing — setup_inference_modules already reduced model_dict to the
        # running net(s) and set the rest to None (dropped by ``net is not None``).
        skip_modules = {} if self.config.inference else self.ema_dict
        modules = {name: net for name, net in self.model_dict.items() if name not in skip_modules and net is not None}
        # The teacher is not part of model_dict; add it when present.
        teacher = self.separate_teacher
        if teacher is not None:
            modules["teacher"] = teacher
        # Preprocessors deliberately live outside the generator's registered tree. Their exposed
        # encode/decode methods route through ``forward``, so compile the root to preserve both
        # torch.compile and FSDP hooks around every operation.
        for name, preprocessor in self.net_preprocessors.preprocessors.items():
            modules[f"preprocessor.{name}"] = preprocessor

        for name, module in modules.items():
            if getattr(module, "_compiled_call_impl", None) is not None:
                logger.info(f"Skipping torch.compile for {name} (already compiled)")
                continue
            logger.info(f"Applying torch.compile (mode={mode}) to {name}")
            module.compile(mode=mode)

    def on_train_begin(self, fsdp_context: Optional["FSDPContext"] = None):
        """Prepare every module before DDP wrapping and optimizer creation.

        Roles are stable for the run: non-EMA ``model_dict`` entries are trainable outside
        inference, and everything else in ``module_dict`` is frozen. Model roots (including
        EMA) and names selected by ``extra_fsdp_modules`` are cast, sharded, materialized, and
        placed immediately through the shared context.
        """
        self._validate_preparation_plan(fsdp_context)

        # Preprocessors are registered and prepared afterwards, by init_preprocessors.
        for name, module in self.module_dict.items():
            self.prepare_module(module, name=name, fsdp_context=fsdp_context)

        # Parent modules are sharded before loading the lazy preprocessors, which are then
        # prepared immediately instead of waiting for another model-wide pass.
        self.init_preprocessors(fsdp_context=fsdp_context)

        with raise_across_ranks("model validation"):
            self.validate_modules()

        if hasattr(self, "reward"):
            self.reward.build_model(self)
            synchronize()

    def _validate_preparation_plan(self, fsdp_context: Optional["FSDPContext"]) -> None:
        """Check the registered roots before any is prepared; warn about precisions only FSDP resolves.

        Each root's ``prepare_module`` compares its request and rejects already prepared modules. The
        root order must agree too: same-shaped roots would otherwise exchange state during synchronization.
        """
        # A local rejection must reach every rank before the plan collective below.
        with raise_across_ranks("model preparation validation"):
            if fsdp_context is not None and self.config.net_adapter.type is not None:
                raise ValueError("net_adapter does not support FSDP")
            # Roots may have changed since registration.
            validate_module_roots(self.module_dict, "model preparation")
        validate_object_across_ranks(tuple(self.module_dict), operation="model preparation plan")
        if fsdp_context is not None:
            return

        if self.precision_fsdp_reduce != self.precision_opt:
            logger.warning("`precision_fsdp_reduce` is ignored without FSDP.")
        if self.precision_opt != self.precision_model and self.precision_amp is None:
            logger.warning(
                f"Optimized modules are stored in {self.precision_opt} while the networks run in "
                f"{self.precision_model}; without FSDP, ordinary modules usually need compatible "
                "`precision_amp` or network-owned dtype handling, and may otherwise fail at an "
                "incompatible operation."
            )
        if self.use_ema and self.precision_opt != self.precision_model and self.precision_amp_infer is None:
            logger.warning(
                f"Live EMA weights are stored in {self.precision_opt} while the networks run in "
                f"{self.precision_model}; without FSDP, ordinary modules usually need compatible "
                "`precision_amp_infer` or network-owned dtype handling during EMA sampling, and may "
                "otherwise fail at an incompatible operation."
            )

    def _batch_size(self, data: Dict[str, Any]) -> int:
        """The batch size, read off the loader's per-sample index list (``idx``).

        Every loader tags each sample with an ``idx`` entry, collated into a list, so the count
        holds even for an unconditional batch -- whose ``condition`` may carry no tensor to read a
        batch size from.
        """
        if "idx" not in data:
            raise KeyError(
                "Batch has no 'idx' key to read the batch size from. Every dataloader must tag each "
                "sample with a per-sample 'idx'; add one to your loader's output. "
                f"Got keys: {sorted(data.keys())}."
            )
        return len(data["idx"])

    def _get_outputs(
        self,
        gen_data: TensorOrDict,
        condition: Any = None,
        neg_condition: Any = None,
    ) -> Dict[str, TensorOrDict | Callable]:
        """Build outputs for validation sampling and training-sample logging.

        The training sample is an x0 approximation; callers whose generation is a noisy trajectory state convert it
        to x0 first. It is returned detached: DDP requires returned tensors to be detached or covered by the losses
        (see ``DDPWrapper.single_train_step``).
        """
        noise = randn_like(gen_data, dtype=self.precision)

        return {
            "gen_rand": partial(
                self.generator_fn,
                net=self.net_inference,
                noise=noise,
                condition=condition,
                neg_condition=neg_condition,
            ),
            "input_rand": noise,
            "gen_rand_train": detach(gen_data),
        }

    def _student_sample_loop(
        self,
        net: FastGenNetwork,
        x: TensorOrDict,
        student_grid: torch.Tensor,
        condition: Any = None,
        neg_condition: Any = None,
    ) -> TensorOrDict:
        """Advance a student across every interval of its runtime grid.

        Handles both plain tensors and dict-of-tensors (multimodal).

        Args:
            net: Student network.
            x: State at the first node of ``student_grid``.
            student_grid: Descending runtime grid with one more node than intervals.
            condition: Optional conditioning information.
            neg_condition: Optional negative conditioning information.

        Returns:
            State at the terminal grid node.
        """
        del neg_condition

        num_steps = len(student_grid) - 1
        for step_index in range(num_steps):
            x = self._advance_student_step(
                net,
                x,
                step_index,
                student_grid,
                condition=condition,
            )

        return x

    def _advance_student_step(
        self,
        net: FastGenNetwork,
        state: TensorOrDict,
        step_index: int,
        student_grid: torch.Tensor,
        condition: Any = None,
    ) -> TensorOrDict:
        """Advance across ``student_grid[step_index:step_index + 2]``.

        ``step_index`` is an ordinal in the supplied runtime grid. It is not a
        persistent base-grid index used by methods such as PDD.
        """
        num_steps = len(student_grid) - 1
        if not isinstance(step_index, int) or not 0 <= step_index < num_steps:
            raise ValueError(f"step_index must be an int in [0, {num_steps - 1}], got {step_index!r}.")
        t = student_grid[step_index].expand(get_batch_size(state))
        x0 = net(
            state,
            t,
            condition=condition,
            fwd_pred_type="x0",
            output_dtype=None,
        )
        if step_index + 1 == num_steps:
            return cast_like(x0, state)
        next_state = net.noise_scheduler.step(
            state,
            x0,
            student_grid[step_index],
            student_grid[step_index + 1],
            sample_type=self.config.student_sample_type,
        )
        return next_state

    def generator_fn(
        self,
        net: FastGenNetwork,
        noise: TensorOrDict,
        data: TensorOrDict = None,
        condition: Any = None,
        neg_condition: Any = None,
    ) -> TensorOrDict:
        """
        Single-step or multistep generation with the distilled network.

        Handles both plain tensors and dict-of-tensors (multimodal).

        Args:
            net: The FastGenNetwork network
            noise: Pure noise to start from (zero-mean, unit-variance Gaussian tensor or dict-of-tensors)
            data (torch.Tensor, optional): Additional data to add to initial latents.
                Useful for inpainting or other conditional tasks. Defaults to None.
            condition: Optional conditioning information.
            neg_condition: Accepted by the shared generator interface. Generic
                distilled students do not apply guidance again at inference.

        Returns:
            Generated sample from the distilled single-step or multistep student.
        """
        device = get_device(noise)
        with basic_utils.inference_mode(net, precision_amp=self.precision_amp_infer, device_type=device.type):
            student_grid = net.noise_scheduler.student_grid(device=device)

            # Initialize the state at the first student-grid node.
            latents = net.noise_scheduler.latents(noise, t_init=student_grid[0])

            # Add optional data (e.g., for inpainting)
            if data is not None:
                latents = per_modality(torch.add, latents, data)

            # Advance through every interval of the student grid.
            sample = self._student_sample_loop(
                net,
                latents,
                student_grid=student_grid,
                condition=condition,
                neg_condition=neg_condition,
            )
            return cast_like(sample, noise)

    def sample(
        self,
        net: FastGenNetwork,
        noise: TensorOrDict,
        **kwargs,
    ) -> TensorOrDict:
        """Run a network-owned sampler under model-owned inference autocast."""
        assert hasattr(net, "sample")
        with basic_utils.inference_mode(
            net,
            precision_amp=self.precision_amp_infer,
            device_type=get_device(noise).type,
        ):
            # Teacher sampling uses the network's own default integration policy.
            # SFTModel.generator_fn passes its student_sample_type explicitly.
            result = net.sample(noise, guidance_cfg=self.config.guidance_cfg, **kwargs)
        return cast_like(result, noise)

    @abstractmethod
    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for the FastGen model.

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output
        """

    def init_optimizers(self):
        """Initialize optimizers, lr_schedulers and grad_scalers"""
        # instantiate the optimizer for the generator network
        self.net_optimizer = instantiate(self.config.net_optimizer, model=self.net)
        # instantiate the lr scheduler for the generator network
        self.net_lr_scheduler = get_scheduler(self.net_optimizer, self.config.net_scheduler)
        # Loss scaling protects FP16 arithmetic/reduction. Parameter storage alone is not a reason
        # to enable it; PyTorch only forbids unscaling gradients that are themselves FP16.
        fp16_sources = {
            "precision": self.precision,
            "precision_model": self.precision_model,
            "precision_amp": self.precision_amp,
        }
        if self.uses_fsdp:
            fp16_sources["precision_fsdp_reduce"] = self.precision_fsdp_reduce
        fp16_sources = [name for name, dtype in fp16_sources.items() if dtype == torch.float16]
        grad_scaler_required = bool(fp16_sources)
        grad_scaler_supported = self.precision_opt != torch.float16
        grad_scaler_enabled = self.config.grad_scaler_enabled and grad_scaler_required and grad_scaler_supported
        if grad_scaler_required:
            if not grad_scaler_supported:
                logger.warning(
                    "Grad scaler cannot unscale FP16 trainable/master gradients; use BF16 or preferably FP32 "
                    "parameter storage to scale FP16 arithmetic."
                )
            elif grad_scaler_enabled:
                logger.info(
                    f"Grad scaler enabled with init scale {self.config.grad_scaler_init_scale} and growth interval {self.config.grad_scaler_growth_interval}."
                )
            else:
                logger.warning(
                    f"Grad scaler disabled but recommended when using float16 precision (set via {fp16_sources})."
                )
        elif self.config.grad_scaler_enabled:
            logger.warning(
                "Grad scaler requested but disabled: none of precision, precision_model, precision_amp, or "
                "precision_fsdp_reduce under FSDP is float16, so there is no FP16 arithmetic to scale."
            )
        self.grad_scaler = torch.amp.GradScaler(
            device=self.device.type,
            init_scale=self.config.grad_scaler_init_scale,
            growth_interval=self.config.grad_scaler_growth_interval,
            enabled=grad_scaler_enabled,
        )

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """
        Get the optimizers for the current iteration
        Args:
            iteration (int): The current training iteration
        """
        return [self.net_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler]:
        """
        Get the lr schedulers for the current iteration
        Args:
            iteration (int): The current training iteration
        """
        return [self.net_lr_scheduler]

    def optimizers_zero_grad(self, iteration: int) -> None:
        """
        Zero the gradients of the optimizers based on the iteration
        """
        for optimizer in self.get_optimizers(iteration):
            optimizer.zero_grad(set_to_none=True)

    def should_use_grad_scaler(self, optimizer: torch.optim.Optimizer) -> bool:
        """Whether this optimizer has gradients that the enabled scaler can unscale.

        PyTorch rejects FP16 gradients during unscale (BF16/FP32/FP64 are
        supported). Fail before any optimizer steps so scaled FP16 gradients
        can never reach an unscaled ``optimizer.step()``.

        Args:
            optimizer: The optimizer to check

        Returns:
            True for an enabled scaler and at least one supported gradient.
        """
        if not self.grad_scaler.is_enabled():
            return False

        has_grad = False
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    continue
                has_grad = True
                if param.grad.dtype == torch.float16:
                    raise RuntimeError(
                        "GradScaler cannot unscale FP16 gradients; keep trainable/master parameters in BF16 or "
                        "preferably FP32."
                    )
        return has_grad

    def optimizers_schedulers_step(self, iteration: int) -> None:
        """
        Step the optimizer and scheduler step based on the iteration,
        and gradient scaler is also updated
        """
        optimizers = self.get_optimizers(iteration)
        use_grad_scaler = [self.should_use_grad_scaler(optimizer) for optimizer in optimizers]
        for optimizer, use_scaler in zip(optimizers, use_grad_scaler):
            if use_scaler:
                self.grad_scaler.step(optimizer)
            else:
                optimizer.step()
        if any(use_grad_scaler):
            # One scale factor belongs to the shared backward pass, even when a
            # method updates multiple optimizers (for example DMD2's discriminator).
            self.grad_scaler.update()

        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    @staticmethod
    def _load_pretrained_model(
        model: torch.nn.Module,
        pretrained_model_path: str,
        device: torch.device | str = "cpu",
        meta_init: bool = False,
    ) -> None:
        """
        Load the pre-trained model from the given path
        Args:
            model (torch.nn.Module): The model to load
            pretrained_model_path (str): The path to the pretrained model
            device (torch.device | str): Map location for the checkpoint tensors; the module's placement is unchanged
            meta_init (bool): Whether only rank 0 loads state for a meta-initialized module
        """
        # One rank mirrors a remote checkpoint and leaving the block publishes it; this call site
        # is collective, so that is safe here. Every rank then reads the mirror rather than
        # broadcasting off rank 0: the model is still on CPU at this point, so a broadcast
        # would ride gloo, which measured slower than just reading. The mirror is only an
        # optimization: if it cannot be written, loading falls back to the original remote source.
        with raise_across_ranks(f"mirror pretrained model {pretrained_model_path}"):
            if is_rank0():
                cache(pretrained_model_path, strict=False)

        # Only rank 0 loads weights if using meta initialization. Propagate local read,
        # conversion, and coverage failures before peers continue into model setup; the
        # poll also lines the ranks up, so no further barrier is needed.
        with raise_across_ranks(f"load pretrained model {pretrained_model_path}"):
            if not meta_init or is_rank0():
                model_dict = load_checkpoint_state_dict(pretrained_model_path, device=device)

                for k, v in model_dict.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 0:
                        # since FSDP2 cannot handle 0-dim. tensors, we adapted all network definitions to use 1-dim.
                        # tensors with numel equal to 1
                        model_dict[k] = v.unsqueeze(0)
                        logger.debug(f"Changed {k} from 0-dim. tensor to 1-dim. tensor with numel equal to 1.")

                load_module_state_dict(
                    model,
                    model_dict,
                    strict=False,
                    require_any=True,
                    description=f"Pretrained {type(model).__name__}",
                )
                torch.cuda.empty_cache()

    def apply_reward(
        self,
        gen_data: TensorOrDict,
        data: Dict[str, Any],
        loss_map: Dict[str, torch.Tensor],
        condition_dropout_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply rewards, optionally excluding samples whose prompt was dropped (``True`` in a ``[B]`` mask)."""
        if hasattr(self, "reward"):
            assert isinstance(
                gen_data, torch.Tensor
            ), f"Rewards currently require unimodal (tensor) gen_data, got {type(gen_data).__name__}"
            if self.config.optimize_reward and torch.is_grad_enabled():
                assert gen_data.requires_grad, "`gen_data` must require grad for rewards"
            else:
                gen_data = gen_data.detach()
            reward_loss_map = self.reward(gen_data, data, condition_dropout_mask=condition_dropout_mask)
            reward_loss = reward_loss_map.pop("total_loss")
            basic_utils.safe_update(reward_loss_map, {"reward_loss": reward_loss})
            basic_utils.safe_update(loss_map, reward_loss_map)
            if self.config.optimize_reward:
                # subtract the reward from the loss since we want to maximize the reward
                loss_map["total_loss"] = loss_map["total_loss"] + reward_loss
        return loss_map

    def autocast(self):
        """Return the autocast context manager for training and validation.

        ``config.autocast_cache_enabled`` controls this context; standalone inference and
        encoder/decoder autocast do not use it. Nested regions inherit the setting. The
        ``torch.clear_autocast_cache()`` calls in the methods only matter while it is enabled.
        """
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.precision_amp,
            enabled=self.precision_amp is not None,
            cache_enabled=self.config.autocast_cache_enabled,
        )

    @property
    def ema_dict(self):
        """Return dict containing all EMA networks"""
        return {name: getattr(self, name) for name in self.use_ema}

    @property
    def net_preprocessors(self) -> FastGenNetwork:
        """Return the network that owns the preprocessors (VAE, text/image encoders).

        Prefer ``net`` -- the only network that ever owns encoders already, so trying it first
        avoids building a second copy -- then the teacher, then the EMA copy retained by inference
        pruning. A model without any of these roots has no valid preprocessing lifecycle.
        """
        candidates = (getattr(self, name, None) for name in ("net", "teacher", *self.use_ema))
        host = next((candidate for candidate in candidates if candidate is not None), None)
        if host is None:
            raise RuntimeError(f"{type(self).__name__} has no network available to own preprocessors.")
        return host

    @property
    def net_inference(self):
        """Return the network to use for inference.

        Uses the first configured EMA network when available. EMA model roots follow the same
        FSDP lifecycle as the live network, so this selection is independent of sharding.
        """
        return getattr(self, self.use_ema[0]) if self.use_ema else self.net

    def validate_modules(self) -> None:
        """Assert the registry invariants left by preparation, which otherwise hold by construction.

        Children attached after their root was prepared stay unprepared, and an adapter or override
        can leave a trainable root without trainable parameters.
        """
        for name, module in self.module_dict.items():
            assert all(
                child.is_prepared for child in module._fastgen_modules()
            ), f"Registered root {name!r} contains an unprepared FastGen module; register all children first."
        for name in self.config.extra_fsdp_modules:
            if name not in self.module_dict:
                logger.info(f"extra_fsdp_modules: {name!r} is not in module_dict; not sharded.")
        if self.config.inference:
            return
        model_names = set(self.model_dict)
        for name, module in self.module_dict.items():
            if name in model_names and name not in self.use_ema:
                assert any(
                    parameter.requires_grad for parameter in module.parameters()
                ), f"Trainable model root {name!r} has no parameter requiring gradients."

    @property
    def model_dict(self):
        """Return the model dict containing the student and EMA networks"""
        return torch.nn.ModuleDict({"net": self.net, **self.ema_dict})

    @property
    def optimizer_dict(self):
        """Return a dict containing all the optimizers"""
        return {
            "net": self.net_optimizer,
        }

    @property
    def scheduler_dict(self):
        """Return a dict containing all the lr schedulers"""
        return {
            "net": self.net_lr_scheduler,
        }
