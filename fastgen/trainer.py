# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any, Iterable, Mapping
from itertools import islice
import os
import time
import gc

import torch
from torch.utils.data import DataLoader

from fastgen.methods import FastGenModel
from fastgen.utils.basic_utils import set_random_seed, set_tmp_random_seed
from fastgen.utils.checkpointer import Checkpointer, is_dcp_checkpoint
from fastgen.utils.io_utils import path_exists
import fastgen.utils.logging_utils as logger
from fastgen.configs.config import BaseConfig
from fastgen.callbacks.callback import CallbackDict
from fastgen.configs.instantiate import instantiate
from fastgen.utils import basic_utils
from fastgen.utils.tree_utils import detach
import fastgen.utils.distributed.ddp as ddp
import fastgen.utils.distributed.fsdp as fsdp
from fastgen.utils.distributed import is_rank0, world_size
from fastgen.utils.distributed.collectives import (
    broadcast_rank0_result,
    raise_across_ranks,
    synced_iter,
    synchronize,
    warn_on_rng_state_drift,
)
import fastgen.utils.global_vars as global_vars
from fastgen.utils.global_vars import set_global_vars, set_temp_global_vars
from fastgen.utils.autoresume import create_auto_resume


def should_load_after_fsdp(path: str | None, model_keys: Iterable[str], fsdp_enabled: bool) -> bool:
    """Whether checkpoint loading should wait until after FSDP wrapping.

    All ranks must call with the same path and FSDP setting. Only rank 0 probes the file;
    its decision or error is shared with every rank. An unset path or disabled FSDP skips the probe.
    """
    if not fsdp_enabled or not path:
        return False
    return broadcast_rank0_result(lambda: is_dcp_checkpoint(path, model_keys), "checkpoint load order")


def _validate_ema_key_map(key_map: Mapping[str, str], ema_names: Iterable[str]) -> None:
    """Warn when ``key_map`` loads ``net`` but not every EMA clone of it.

    The clones were copied from ``net`` before the pretrained load and keep its old weights unless the EMA
    callback re-initializes them, which constant beta with ``start_iter=0`` never does and a non-zero
    ``resume_iter`` can skip.
    """
    unmapped_ema = [name for name in ema_names if name not in key_map]
    if "net" in key_map and unmapped_ema:
        logger.warning(
            f"pretrained_ckpt_key_map loads net but not the EMA module(s) {unmapped_ema}; map them too "
            "(e.g. {'net': 'net', 'ema': 'ema'}) unless the stale clone is intended."
        )


class Trainer:
    def __init__(self, config: BaseConfig):
        """
        Initialize the Trainer.

        Args:
            config (BaseConfig): FastGen config. Auto-resume is built from
                ``config.trainer.auto_resume`` (defaults to None -> NoOpAutoResume).
                See fastgen.utils.autoresume for the interface and examples.
        """
        if config.trainer.ddp and config.trainer.fsdp:
            raise ValueError("trainer.ddp and trainer.fsdp are mutually exclusive; enable only one of them.")
        if config.trainer.fsdp and config.trainer.checkpointer.save_adapter_only:
            raise NotImplementedError("save_adapter_only is not supported with FSDP (DCP checkpointing)")

        self.config = config
        self.fsdp_context: fsdp.FSDPContext | None = None
        set_global_vars(self.config.trainer.global_vars)

        # Initialize auto-resume from ``config.trainer.auto_resume`` (None -> NoOpAutoResume).
        self.auto_resume = create_auto_resume(self.config.trainer.auto_resume)
        logger.info(f"Auto-resume: {type(self.auto_resume).__name__}")

        # Set the random seed identically on every rank. ``train.py`` calls
        # ``Trainer(config)`` *before* model construction, so this seed also
        # gates model construction — every rank builds the same initial weights
        # rather than diverging on PyTorch's non-deterministic process-default
        # seed. Per-rank-distinct seeding is re-applied below via
        # ``set_random_seed(..., by_rank=True)`` right before the dataloader and
        # training loop, which is where data shuffle / augmentation / diffusion
        # noise / timestep sampling actually need rank-divergent RNG.
        set_random_seed(config.trainer.seed, by_rank=False)

        # Initialize the callback functions.
        logger.info("Initializing callbacks (including wandb)...")
        self.callbacks = CallbackDict(config=config, trainer=self)
        logger.success("Callbacks initialized successfully")

        # Synchronize after callback initialization to handle wandb timing differences
        synchronize()
        logger.info("Callback synchronization complete")

        # Initialize the checkpointer.
        logger.info("Initializing checkpointer...")
        self.checkpointer = Checkpointer(self.config.trainer.checkpointer, save_sharded=self.config.trainer.fsdp)
        logger.success("Checkpointer initialized successfully")

    def run(
        self,
        model: FastGenModel,
    ) -> None:
        """
        Run the training loop

        Args:
            model (FastGenModel): Distillation model.
        """
        logger.info("Starting training")
        if world_size() > 1 and not (self.config.trainer.ddp or self.config.trainer.fsdp):
            raise ValueError(
                f"{world_size()} ranks are running without trainer.ddp or trainer.fsdp, so each rank would train "
                "an independent replica; enable one of them or launch a single process."
            )

        iter_start = 0
        logger.info("Initializing callbacks and model ...")
        self.callbacks.on_model_init_start(model)
        pretrained_ckpt_path = self.config.trainer.checkpointer.pretrained_ckpt_path
        # Monolithic weights load into ordinary modules before wrapping. DCP loads directly into
        # the prepared shards; inspect once on rank 0 so every rank takes the same setup path.
        defer_pretrained_ckpt = should_load_after_fsdp(
            pretrained_ckpt_path,
            self.config.trainer.checkpointer.pretrained_ckpt_key_map.values(),
            fsdp_enabled=self.config.trainer.fsdp,
        )
        if pretrained_ckpt_path and not defer_pretrained_ckpt:
            # This typically only affects the first job in the auto-resume chain
            self.load_pretrained_ckpt(model)

        if self.config.trainer.fsdp and self.config.model.precision_amp is not None:
            logger.warning(
                f"Autocast to {self.config.model.precision_amp} is enabled together with FSDP, which already "
                f"casts parameters to precision_model={model.precision_model} per forward. While this is "
                "possible, it is not recommended."
            )

        self.fsdp_context = None
        if self.config.trainer.fsdp:
            logger.info("Preparing FSDP context ..")
            self.fsdp_context = fsdp.create_fsdp_context(
                reduce_dtype=model.precision_fsdp_reduce,
                min_num_params=self.config.trainer.fsdp_min_num_params,
                apply_cpu_offload=self.config.trainer.fsdp_cpu_offload,
                sharding_group_size=self.config.trainer.fsdp_sharding_group_size,
            )

        logger.info("Starting model.on_train_begin ...")
        model.on_train_begin(fsdp_context=self.fsdp_context)
        logger.info("model.on_train_begin completed")

        # DDP wraps the complete method object. FSDP2 has already modified each selected
        # component in place during ``on_train_begin``.
        if self.config.trainer.ddp:
            logger.info("Wrapping model into ddp ..")
            model_ddp = ddp.model_to_ddp(model)
            logger.info("DDP wrapping completed")
        else:
            model_ddp = model

        if defer_pretrained_ckpt:
            logger.info("Loading pretrained checkpoint after FSDP wrapping ..")
            self.load_pretrained_ckpt(model)

        # Compile networks after DDP/FSDP wrapping so torch.compile composes
        # with the distributed wrappers (no-op if torch_compile_mode is None).
        model.apply_torch_compile()

        self.callbacks.on_model_init_end(model_ddp)
        synchronize()

        self.callbacks.on_optimizer_init_start(model)
        model.init_optimizers()
        self.callbacks.on_optimizer_init_end(model)

        self.callbacks.on_load_checkpoint_start(model)

        # Check if we are resuming from an auto-resume checkpoint
        self.auto_resume.init()
        auto_resume_details = self.auto_resume.get_resume_details()
        logger.info(f"Auto-Resume Details: {auto_resume_details}")
        autoresume_ckpt = auto_resume_details["save_path"] if auto_resume_details else None

        resume_requested = self.config.trainer.resume or autoresume_ckpt is not None
        if autoresume_ckpt is not None and not path_exists(self.checkpointer.resolve_load_path(autoresume_ckpt)):
            # A preemption during the ``latest_ar`` publication leaves the continuation pointing at a
            # checkpoint that never completed; resume from the newest complete one instead.
            logger.warning(
                f"Autoresume checkpoint {autoresume_ckpt} not found; resuming from the latest checkpoint in "
                f"{self.config.trainer.checkpointer.save_dir}"
            )
            autoresume_ckpt = None

        if resume_requested:
            logger.info("Loading checkpoints for resuming ..")

            # load previous checkpoint
            iter_start = self.checkpointer.load(
                model.model_dict,
                optimizer_dict=model.optimizer_dict,
                scheduler_dict=model.scheduler_dict,
                grad_scaler=model.grad_scaler,
                callbacks=self.callbacks,
                path=autoresume_ckpt,
                device=model.device,
            )
        self.callbacks.on_load_checkpoint_end(model, iteration=iter_start)

        basic_utils.clear_gpu_memory()
        synchronize()

        # ``reset_parameters`` hooks should not consume randomness, so after the identical seed in
        # ``__init__`` only rank 0's real meta-init construction may leave the CPU generators unequal.
        warn_on_rng_state_drift("Before the per-rank reseed", cpu=not self.config.model.meta_init)

        # re-seed based on the current iteration for resuming
        set_random_seed(self.config.trainer.seed, iteration=iter_start, by_rank=True)

        # resume samplers and initiate the dataloaders
        self.callbacks.on_dataloader_init_start(model, iteration=iter_start)
        self.resolve_sampler_start_idx(iter_start)

        logger.info("Instantiating dataloader...")
        dataloader_train = instantiate(self.config.dataloader_train)
        dataloader_val = instantiate(self.config.dataloader_val) if self.config.dataloader_val else None
        augment_pipe = instantiate(self.config.trainer.augment_pipe)
        self.callbacks.on_dataloader_init_end(model, dataloader_train, dataloader_val, iteration=iter_start)

        self.callbacks.on_train_begin(model, iteration=iter_start)
        logger.info(f"iter_start: {iter_start}")
        if iter_start > 0 and iter_start >= self.config.trainer.max_iter:
            logger.warning(
                f"Resumed at iteration {iter_start}, which already reaches trainer.max_iter="
                f"{self.config.trainer.max_iter}; nothing is left to train."
            )

        if iter_start == 0 and dataloader_val is not None:
            # validation before first training step
            self.validate(model_ddp, model, dataloader_val, iteration=iter_start)

        dataloader_train_iter = synced_iter(dataloader_train, device=model.device)
        # ``max_iter`` is inclusive, so the final checkpoint, labelled with the last completed iteration,
        # holds the state after ``max_iter`` optimizer iterations.
        iter_cur = iter_start
        for iter_cur in range(iter_start + 1, self.config.trainer.max_iter + 1):
            self.callbacks.on_training_step_begin(model, iteration=iter_cur)
            for grad_accum_iter in range(self.config.trainer.grad_accum_rounds):
                data = next(dataloader_train_iter)
                data = model.prepare_batch(data, augment_pipe)
                logger.debug(
                    f"iteration: {iter_cur} | grad_accum_iter: {grad_accum_iter} | data: {basic_utils.to_str(data)}"
                )

                # single training step
                self.callbacks.on_training_accum_step_begin(model, data, iteration=iter_cur, accum_iter=grad_accum_iter)
                loss_map, outputs = self.train_step(model_ddp, model, data, iter_cur, grad_accum_iter)
                self.callbacks.on_training_accum_step_end(
                    model, data, outputs, iteration=iter_cur, accum_iter=grad_accum_iter
                )

            # Not every root forwarded during a training step participates in backward. In
            # particular, auxiliary updates may run a frozen student under no_grad. Establish
            # sharded callback inputs once at this phase boundary.
            model.reshard_fsdp_modules()
            self.callbacks.on_training_step_end(
                model=model,
                data_batch=data,
                output_batch=outputs,
                loss_dict=loss_map,
                iteration=iter_cur,
            )
            is_final_iter = iter_cur == self.config.trainer.max_iter

            # validation
            if (is_final_iter or iter_cur % self.config.trainer.validation_iter == 0) and dataloader_val is not None:
                self.validate(model_ddp, model, dataloader_val, iteration=iter_cur)

            # save checkpoint
            just_saved_checkpoint = False
            latest_checkpoint_path = None
            if is_final_iter or iter_cur % self.config.trainer.save_ckpt_iter == 0:
                latest_checkpoint_path = self.save_checkpoint(model, iter_cur)
                just_saved_checkpoint = True

            # The run is complete and saved after the final iteration, so a termination request needs no continuation.
            if not is_final_iter and self.auto_resume_exit(
                model, iter_cur, skip_if_just_saved=just_saved_checkpoint, recent_checkpoint_path=latest_checkpoint_path
            ):
                # termination requested
                self.callbacks.on_train_end(model, iteration=iter_cur)
                self.callbacks.on_app_end(model, iteration=iter_cur)
                logger.info("Taking a 10 sec nap and exiting training.")
                time.sleep(10)
                return

        logger.info("Training complete.")
        self.callbacks.on_train_end(model, iteration=iter_cur)
        self.callbacks.on_app_end(model, iteration=iter_cur)
        logger.info("Taking a 10 sec nap and exiting training.")
        time.sleep(10)

    def load_pretrained_ckpt(self, model: FastGenModel, device: torch.device | str = "cpu"):
        """
        Load pretrained model weights from a checkpoint.
        """
        key_map = self.config.trainer.checkpointer.pretrained_ckpt_key_map
        _validate_ema_key_map(key_map, getattr(model, "use_ema", None) or ())
        # The format-aware checkpointer accepts the same .pth path for both formats.
        pretrained_ckpt_path = self.config.trainer.checkpointer.pretrained_ckpt_path
        resume_iter = None
        for k_model, k_ckpt in key_map.items():
            if hasattr(model, k_model):
                model_dict = torch.nn.ModuleDict({k_ckpt: getattr(model, k_model)})
                resume_iter = self.checkpointer.load(
                    model_dict,
                    path=pretrained_ckpt_path,
                    device=device,
                    strict=False,
                )
                logger.info(
                    f"Loaded {k_model} model from {k_ckpt} in {pretrained_ckpt_path} at iteration {resume_iter}"
                )
            else:
                logger.warning(
                    f"Model does not have submodule {k_model}. Skipping loading {k_ckpt} from {pretrained_ckpt_path}."
                )
        if resume_iter is not None:
            logger.info(f"Setting resume_iter for model to {resume_iter}.")
            model.resume_iter = resume_iter

    def resolve_sampler_start_idx(self, iteration: int) -> None:
        """
        Resolve unset (``None``) dataloader sampler start indices for a run starting at ``iteration``.
        """
        nimg = (
            iteration * self.config.dataloader_train.batch_size * self.config.trainer.grad_accum_rounds * world_size()
        )
        if getattr(self.config.dataloader_train, "sampler_start_idx", 0) is None:
            logger.info(f"Setting sampler start index to {nimg} images for dataloader_train")
            self.config.dataloader_train.sampler_start_idx = nimg
        # Only the training stream consumed ``nimg`` samples, so the validation loader starts at its beginning.
        if self.config.dataloader_val and getattr(self.config.dataloader_val, "sampler_start_idx", 0) is None:
            self.config.dataloader_val.sampler_start_idx = 0

    def save_checkpoint(self, model: FastGenModel, iteration: int, path: str | None = None) -> str:
        logger.info(f"Saving checkpoint iteration {iteration}")
        self.callbacks.on_save_checkpoint_start(model, iteration=iteration)
        path = self.checkpointer.save(
            model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            callbacks=self.callbacks,
            path=path,
            iteration=iteration,
        )
        self.callbacks.on_save_checkpoint_success(model, iteration=iteration, path=path)
        # Explicitly clear memory after checkpointing: we need this to
        # avoid OOM during wandb logging where the VAE is loaded and
        # used for decoding
        gc.collect()
        torch.cuda.empty_cache()
        self.callbacks.on_save_checkpoint_end(model, iteration=iteration)
        return path

    def train_step(
        self,
        model_ddp: FastGenModel | torch.nn.parallel.DistributedDataParallel,
        model: FastGenModel,
        data: Dict[str, Any],
        iteration: int,
        grad_accum_iter: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Single training step

        Args:
            model_ddp (FastGenModel | torch.nn.parallel.DistributedDataParallel): Distillation model with ddp wrapper.
            model (FastGenModel): Distillation model.
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration: Current training iteration
            grad_accum_iter (int): Gradient accumulation iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output
        """
        grad_accum_rounds = self.config.trainer.grad_accum_rounds
        sync_grads = grad_accum_iter == grad_accum_rounds - 1

        if not self.config.trainer.fsdp:
            with ddp.ddp_sync_grad(model_ddp, sync_grads):
                # forward pass
                with model.autocast():
                    loss_map, outputs = model_ddp.single_train_step(data, iteration)
                # backward pass
                self.callbacks.on_backward_begin(
                    model, data, outputs, loss_map, iteration=iteration, accum_iter=grad_accum_iter
                )
                model.grad_scaler.scale(loss_map["total_loss"] / grad_accum_rounds).backward()
        else:
            with fsdp.fsdp_sync_grad(model, sync_grads):
                # forward pass
                with model.autocast():
                    loss_map, outputs = model_ddp.single_train_step(data, iteration)
                # backward pass
                self.callbacks.on_backward_begin(
                    model, data, outputs, loss_map, iteration=iteration, accum_iter=grad_accum_iter
                )
                model.grad_scaler.scale(loss_map["total_loss"] / grad_accum_rounds).backward()

        if grad_accum_iter == grad_accum_rounds - 1:
            # optimizer step, scheduler step, and more
            self.callbacks.on_optimizer_step_begin(model=model, iteration=iteration)
            model.optimizers_schedulers_step(iteration)
            # Zero after step to free memory on active optimizers
            model.optimizers_zero_grad(iteration)

        # detach loss_map and outputs
        return detach(loss_map), detach(outputs)

    @torch.no_grad()
    def validate(
        self,
        model_ddp: FastGenModel | torch.nn.parallel.DistributedDataParallel,
        model: FastGenModel,
        dataloader_val: DataLoader,
        iteration: int = 0,
    ) -> None:
        for idx, val_vars in enumerate(self.config.trainer.global_vars_val):
            with set_temp_global_vars(val_vars), set_tmp_random_seed(
                self.config.trainer.val_seed,
                by_rank=True,
                devices=[model.device] if model.device.type == "cuda" else [],
            ):
                self.callbacks.on_validation_begin(model, iteration=iteration, idx=idx)
                logger.info(f"Validation iteration {iteration}")
                batches = synced_iter(dataloader_val, device=model.device)
                for step, data in enumerate(islice(batches, getattr(global_vars, "MAX_VAL_STEPS", None))):
                    data = model.prepare_batch(data)
                    self.callbacks.on_validation_step_begin(model, data, step=step, iteration=iteration, idx=idx)
                    logger.debug(f"iteration: {iteration} | validation step: {step} | data: {basic_utils.to_str(data)}")
                    with model.autocast():
                        loss_map, outputs = model_ddp.single_train_step(data, iteration)
                    self.callbacks.on_validation_step_end(
                        model, data, outputs, loss_map, step=step, iteration=iteration, idx=idx
                    )
                self.callbacks.on_validation_end(model, iteration=iteration, idx=idx)
                synchronize()

        # Validation has no backward pass to release roots configured with
        # reshard_after_forward=False. Restore shards before checkpointing or training resumes.
        model.reshard_fsdp_modules()

    def auto_resume_exit(
        self,
        model: FastGenModel,
        iteration: int,
        skip_if_just_saved: bool = False,
        recent_checkpoint_path: str | None = None,
    ) -> bool:
        """
        Check if the training should be terminated and auto-resume should be triggered.

        Args:
            model (FastGenModel): Distillation model.
            iteration (int): Current training iteration
            skip_if_just_saved (bool): Skip saving checkpoint if we just saved one
            recent_checkpoint_path (str): Path to the most recently saved checkpoint

        Returns:
            bool: True if the training should be terminated, False otherwise

        """
        termination_requested = broadcast_rank0_result(
            lambda: bool(self.auto_resume.termination_requested()),
            "autoresume termination poll",
            device=model.device,
        )

        if not termination_requested:
            return False

        # Termination requested - save checkpoint and request resume
        ar_details = self.auto_resume.get_resume_details() or {}

        # Only save checkpoint if we haven't just saved one
        if not skip_if_just_saved:
            save_path = self.save_checkpoint(
                model, iteration, path=os.path.join(self.config.trainer.checkpointer.save_dir, "latest_ar.pth")
            )
            ar_details["save_path"] = save_path
        else:
            # Use the most recent checkpoint path
            logger.info("Skipping AutoResume checkpoint save as we just saved a regular checkpoint")
            if recent_checkpoint_path:
                save_path = recent_checkpoint_path
                logger.info(f"Using recently saved checkpoint: {save_path}")
            else:
                # Fallback: construct the path (this should rarely happen)
                logger.warning("No recent checkpoint path provided, constructing path")
                save_path = self.checkpointer.resolve_save_path(path=None, iteration=iteration)
            ar_details["save_path"] = save_path

        # Rank-0-only I/O: fail every rank together instead of letting the peers run the end-of-training callbacks.
        with raise_across_ranks("autoresume resume request"):
            if is_rank0():
                self.auto_resume.request_resume(user_dict=ar_details)

        logger.info("Autoresume requested. Terminating training.")
        return True
