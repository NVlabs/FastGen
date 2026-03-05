# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any, List, Optional, Callable
import os
import time
import gc

import torch
from torch.utils.data import DataLoader

from fastgen.methods import FastGenModel
from fastgen.utils.basic_utils import set_random_seed, set_tmp_random_seed
from fastgen.utils.checkpointer import Checkpointer, FSDPCheckpointer
import fastgen.utils.logging_utils as logger
from fastgen.configs.config import BaseConfig
from fastgen.callbacks.callback import CallbackDict
from fastgen.utils import instantiate, basic_utils
import fastgen.utils.distributed.ddp as ddp
import fastgen.utils.distributed.fsdp as fsdp
from fastgen.utils.distributed import synchronize, is_rank0, world_size
import torch.distributed as dist
from fastgen.utils import set_global_vars, set_temp_global_vars
from fastgen.utils import global_vars
from fastgen.utils.autoresume import AutoResumeInterface, create_auto_resume


class Trainer:
    def __init__(self, config: BaseConfig, auto_resume: Optional[AutoResumeInterface] = None):
        """
        Initialize the Trainer.

        Args:
            config (BaseConfig): FastGen config
            auto_resume (Optional[AutoResumeInterface]): Custom auto-resume implementation.
                If None, defaults to NoOpAutoResume (auto-resume disabled).
                See fastgen.utils.autoresume for the interface and examples.
        """

        self.config = config
        set_global_vars(self.config.trainer.global_vars)

        # Initialize auto-resume (defaults to NoOpAutoResume if not provided)
        self.auto_resume = create_auto_resume(auto_resume)
        logger.info(f"Auto-resume: {type(self.auto_resume).__name__}")

        # Set random seed
        set_random_seed(config.trainer.seed, by_rank=True)

        # Initialize the callback functions.
        logger.info("Initializing callbacks (including wandb)...")
        self.callbacks = CallbackDict(config=config, trainer=self)
        logger.success("Callbacks initialized successfully")

        # Synchronize after callback initialization to handle wandb timing differences
        synchronize()
        logger.info("Callback synchronization complete")

        # Initialize the checkpointer.
        logger.info("Initializing checkpointer...")
        if self.config.trainer.fsdp:
            self.checkpointer = FSDPCheckpointer(self.config.trainer.checkpointer)
        else:
            self.checkpointer = Checkpointer(self.config.trainer.checkpointer)
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

        iter_start = 0
        logger.info("Initializing callbacks and model ...")
        self.callbacks.on_model_init_start(model)
        if self.config.trainer.checkpointer.pretrained_ckpt_path:
            # This typically only affects the first job in the auto-resume chain
            self.load_pretrained_ckpt(model)

        if self.config.trainer.fsdp and (
            self.config.model.precision_amp is not None
            and self.config.model.precision_amp != self.config.model.precision
        ):
            logger.warning(
                f"Autocast to {self.config.model.precision_amp} is enabled and FSDP is enabled. "
                f"While this is possible, it is not recommended."
            )

        logger.info("Starting model.on_train_begin ...")
        synchronize()
        model.on_train_begin(is_fsdp=self.config.trainer.fsdp)
        synchronize()
        logger.info("model.on_train_begin completed")

        # wrap model into DDP or FSDP
        assert not (
            self.config.trainer.ddp and self.config.trainer.fsdp
        ), "Model cannot be wrapped into both DDP and FSDP"
        if self.config.trainer.ddp:
            logger.info("Wrapping model into ddp ..")
            model_ddp = ddp.model_to_ddp(model)
            logger.info("DDP wrapping completed")
        elif self.config.trainer.fsdp:
            logger.info("Wrapping model into fsdp ..")
            model_ddp = fsdp.model_to_fsdp(
                model,
                min_num_params=self.config.trainer.fsdp_min_num_params,
                apply_cpu_offload=self.config.trainer.fsdp_cpu_offload,
                sync_module_states=self.config.model.fsdp_meta_init,
                sharding_group_size=self.config.trainer.fsdp_sharding_group_size,
            )
            logger.info("FSDP wrapping completed")
        else:
            model_ddp = model
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

        if self.config.trainer.resume or autoresume_ckpt is not None:
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

        # re-seed based on the current iteration for resuming
        set_random_seed(self.config.trainer.seed, iteration=iter_start, by_rank=True)

        # resume samplers and initiate the dataloaders
        self.callbacks.on_dataloader_init_start(model, iteration=iter_start)
        nimg = (
            iter_start * self.config.dataloader_train.batch_size * self.config.trainer.grad_accum_rounds * world_size()
        )
        for loader in ["dataloader_train", "dataloader_val"]:
            dataloader_config = getattr(self.config, loader, None)
            if getattr(dataloader_config, "sampler_start_idx", 0) is None:
                logger.info(f"Setting sampler start index to {nimg} images for {loader}")
                dataloader_config.sampler_start_idx = nimg

        logger.info("Instantiating dataloader...")
        dataloader_train = instantiate(self.config.dataloader_train)
        dataloader_val = (
            instantiate(self.config.dataloader_val) if getattr(self.config, "dataloader_val", None) else None
        )
        augment_pipe = instantiate(self.config.trainer.augment_pipe)
        self.callbacks.on_dataloader_init_end(model, dataloader_train, dataloader_val, iteration=iter_start)

        self.callbacks.on_train_begin(model, iteration=iter_start)
        logger.info(f"iter_start: {iter_start}")

        if iter_start == 0 and dataloader_val is not None:
            # validation before first training step
            self.validate(model_ddp, model, dataloader_val, iteration=iter_start)

        dataloader_train_iter = iter(dataloader_train)
        for iter_cur in range(iter_start + 1, self.config.trainer.max_iter):
            self.callbacks.on_training_step_begin(model, iteration=iter_cur)
            for grad_accum_iter in range(self.config.trainer.grad_accum_rounds):
                data = next(dataloader_train_iter)
                data = self.preprocess_data(model, data, augment_pipe)
                logger.debug(
                    f"iteration: {iter_cur} | grad_accum_iter: {grad_accum_iter} | data: {basic_utils.to_str(data)}"
                )

                # single training step
                self.callbacks.on_training_accum_step_begin(model, data, iteration=iter_cur, accum_iter=grad_accum_iter)
                loss_map, outputs = self.train_step(model_ddp, model, data, iter_cur, grad_accum_iter)

            self.callbacks.on_training_step_end(
                model=model,
                data_batch=data,
                output_batch=outputs,
                loss_dict=loss_map,
                iteration=iter_cur,
            )

            # validation
            if iter_cur % self.config.trainer.validation_iter == 0 and dataloader_val is not None:
                self.validate(model_ddp, model, dataloader_val, iteration=iter_cur)

            # save checkpoint
            just_saved_checkpoint = False
            latest_checkpoint_path = None
            if iter_cur % self.config.trainer.save_ckpt_iter == 0:
                latest_checkpoint_path = self.save_checkpoint(model, iter_cur)
                just_saved_checkpoint = True

            if self.auto_resume_exit(
                model, iter_cur, skip_if_just_saved=just_saved_checkpoint, recent_checkpoint_path=latest_checkpoint_path
            ):
                # termination requested
                self.callbacks.on_train_end(model, iteration=iter_cur)
                self.callbacks.on_app_end(model, iteration=iter_cur)
                logger.info("Taking a 10 sec nap and exiting training.")
                time.sleep(10)
                return

        logger.info("Training complete.")
        # validation in the end
        if dataloader_val is not None:
            self.validate(model_ddp, model, dataloader_val, iteration=self.config.trainer.max_iter)
        self.save_checkpoint(model, self.config.trainer.max_iter)
        self.callbacks.on_train_end(model, iteration=self.config.trainer.max_iter)
        self.callbacks.on_app_end(model, iteration=self.config.trainer.max_iter)
        logger.info("Taking a 10 sec nap and exiting training.")
        time.sleep(10)

    def load_pretrained_ckpt(self, model: FastGenModel, device: torch.device | str = "cpu"):
        """
        Load pretrained model weights from a checkpoint.
        """
        key_map = self.config.trainer.checkpointer.pretrained_ckpt_key_map
        # use FSDP checkpointer to load the pretrained checkpoint
        # (which falls back to the basic checkpointer if the checkpoint ends with .pth)
        _checkpointer = FSDPCheckpointer(self.config.trainer.checkpointer)
        resume_iter = None
        for k_model, k_ckpt in key_map.items():
            if hasattr(model, k_model):
                model_dict = torch.nn.ModuleDict({k_ckpt: getattr(model, k_model)})
                resume_iter = _checkpointer.load(
                    model_dict,
                    path=self.config.trainer.checkpointer.pretrained_ckpt_path,
                    device=device,
                )
                logger.info(
                    f"Loaded {k_model} model from {k_ckpt} in {self.config.trainer.checkpointer.pretrained_ckpt_path} "
                    f"at iteration {resume_iter}"
                )
            else:
                logger.warning(
                    f"Model does not have submodule {k_model}. Skipping loading {k_ckpt} from "
                    f"{self.config.trainer.checkpointer.pretrained_ckpt_path}."
                )
        if resume_iter is not None:
            logger.info(f"Setting resume_iter for model to {resume_iter}.")
            model.resume_iter = resume_iter

    def save_checkpoint(self, model: FastGenModel, iteration: int, path: str | None = None) -> str:
        logger.info(f"Saving checkpoint iteration {iteration}")
        self.callbacks.on_save_checkpoint_start(model, iteration=iteration)
        # awaken the dataloader to avoid timeout
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
            model_ddp (FastGenModel | torch.nn.parallel.DistributedDataParallel): Distillation model with ddp wraaper.
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
        return basic_utils.detach(loss_map), basic_utils.detach(outputs)

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
                for step, data in enumerate(dataloader_val):
                    if getattr(global_vars, "MAX_VAL_STEPS", None) is not None and step >= getattr(
                        global_vars, "MAX_VAL_STEPS"
                    ):
                        break

                    self.callbacks.on_validation_step_begin(model, data, step=step, iteration=iteration, idx=idx)
                    data = self.preprocess_data(model, data)
                    logger.debug(f"iteration: {iteration} | validation step: {step} | data: {basic_utils.to_str(data)}")
                    with model.autocast():
                        loss_map, outputs = model_ddp.single_train_step(data, iteration)
                    self.callbacks.on_validation_step_end(
                        model, data, outputs, loss_map, step=step, iteration=iteration, idx=idx
                    )
                self.callbacks.on_validation_end(model, iteration=iteration, idx=idx)
                synchronize()

    @torch.no_grad()
    def preprocess_data(
        self, model: FastGenModel, data: Dict[str, Any], augment_pipe: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Preprocess the data before passing to the model.
        Args:
            model: FastGenModel
            data: Dict[str, Any]

        Returns:
            Dict[str, Any]: Preprocessed data

        """
        ctx = dict(device=model.device, dtype=model.precision)
        data = basic_utils.to(data, **ctx)
        if augment_pipe is not None:
            data = augment_pipe(data)

        # we do not use torch.inference_mode here since resulting inference tensors
        # give runtime errors with gradient computations/checkpointing
        with torch.autocast(
            device_type=model.device.type, dtype=model.precision_amp_enc, enabled=model.precision_amp_enc is not None
        ):
            # Data/noise
            raw = "{}_raw".format
            for k in ["real", "noise"]:
                if k in data and raw(k) not in data:
                    data[raw(k)] = data[k]
                    # dataloader returns real of shape [B, C, T, H, W]
                    if hasattr(model.net, "vae") and data[k].shape[1] != model.input_shape[0]:
                        # Encode the data/noise to latent space
                        data[k] = model.net.vae.encode(data[k])

            # Text conditions
            for k in ["condition", "neg_condition"]:
                if k in data and raw(k) not in data:
                    data[raw(k)] = data[k]
                    if hasattr(model.net, "text_encoder") and isinstance(data[k], List):
                        # Encode the prompt to embedding
                        data[k] = model.net.text_encoder.encode(data[k])

            # Context for i2v/vid2vid
            if "real_raw" in data:
                if getattr(model.net, "is_i2v", False):  # extra vid context for i2v
                    # compute input for I2V models
                    real_raw_first_frame = data["real_raw"][:, :, 0:1]
                    bsz, channels = real_raw_first_frame.shape[0:2]
                    num_frames, height, width = data["real_raw"].shape[2:]
                    first_frame_cond = real_raw_first_frame

                    # Wan 2.1 I2V model concatenates first_frame_cond with noisy latents and mask.
                    # Wan 2.2 5B model replaces the first noisy latent frame with the first clean latent frame.
                    if model.net.concat_mask:
                        padding_shape = (bsz, channels, num_frames - 1, height, width)
                        first_frame_cond = torch.cat(
                            [real_raw_first_frame, real_raw_first_frame.new_zeros(*padding_shape)], dim=2
                        )

                    if hasattr(model.net, "vae"):
                        # Official Wan I2V implementation uses the VAE encoder with "argmax" mode to avoid stochasticity
                        data["first_frame_cond"] = model.net.vae.encode(first_frame_cond, mode="argmax")
                    else:
                        data["first_frame_cond"] = first_frame_cond
                        if model.net.concat_mask:
                            logger.debug(
                                "Consider using a pixel-space dataloader: "
                                "Current latent dataloader uses latent zeros for image condition "
                                "while pretrained Wan2.1 I2V uses VAE-encoded zeros."
                            )

                if hasattr(model.net, "image_encoder"):
                    # Encode the first video frame with CLIP
                    data["encoder_hidden_states_image"] = model.net.image_encoder.encode(data["real_raw"][:, :, 0])

                if getattr(model.net, "is_vid2vid", False):  # extra vid context for vid2vid
                    assert hasattr(
                        model.net, "prepare_vid_conditioning"
                    ), "model.net must have prepare_vid_conditioning method"
                    if "depth_latent" in data:
                        data["vid_context"] = model.net.prepare_vid_conditioning(
                            data["real_raw"], condition_latents=data["depth_latent"]
                        )
                    else:
                        data["vid_context"] = model.net.prepare_vid_conditioning(data["real_raw"])

                # Cosmos video2world conditioning: use first a few frames as conditioning
                if getattr(model.net, "is_video2world", False):
                    num_cond_frames = getattr(model.net, "num_conditioning_frames", 1)
                    real_raw_first_frames = data["real_raw"][:, :, :num_cond_frames]
                    bsz, channels, _, height, width = data["real_raw"].shape

                    # Encode conditioning frames with VAE
                    if hasattr(model.net, "vae"):
                        data["conditioning_latents"] = model.net.vae.encode(real_raw_first_frames, mode="argmax")
                    else:
                        data["conditioning_latents"] = real_raw_first_frames

                    # Create condition mask: 1 for conditioning frames, 0 for generated frames
                    t_latent = data["real"].shape[2]
                    t_cond_latent = data["conditioning_latents"].shape[2]
                    condition_mask = torch.zeros(bsz, 1, t_latent, height // 8, width // 8, device=data["real"].device)
                    condition_mask[:, :, :t_cond_latent] = 1.0
                    data["condition_mask"] = condition_mask

        # Move encoded data to dtype and device
        data = basic_utils.to(data, **ctx)

        return data

    def auto_resume_exit(
        self, model: FastGenModel, iteration: int, skip_if_just_saved: bool = False, recent_checkpoint_path: str = None
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
        # Check termination on rank 0 and broadcast to all ranks
        termination_requested = False

        # Ensure all ranks are ready before rank 0 checks termination
        synchronize()

        if is_rank0():
            termination_requested = self.auto_resume.termination_requested()

        # Broadcast the decision from rank 0 to all other ranks
        if world_size() > 1:
            termination_tensor = torch.tensor([1.0 if termination_requested else 0.0], device=model.device)
            dist.broadcast(termination_tensor, src=0)
            termination_requested = termination_tensor.item() > 0.5

        # Ensure all ranks have received the broadcast before proceeding
        synchronize()

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
                if isinstance(self.checkpointer, FSDPCheckpointer):
                    save_path = os.path.join(self.config.trainer.checkpointer.save_dir, f"{iteration:07d}")
                else:
                    save_path = os.path.join(self.config.trainer.checkpointer.save_dir, f"{iteration:07d}.pth")
            ar_details["save_path"] = save_path

        if is_rank0():
            self.auto_resume.request_resume(user_dict=ar_details)

        logger.info("Autoresume requested. Terminating training.")
        return True
