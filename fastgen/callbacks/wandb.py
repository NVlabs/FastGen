# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
from dataclasses import dataclass, field
import time
from typing import Optional, Dict, Callable, TYPE_CHECKING
import gc


import torch
import torchvision
from torchvision.transforms import functional as tv_F

import wandb
import wandb.util

from fastgen.callbacks.callback import Callback
from fastgen.configs.config_utils import serialize_config
from fastgen.utils import basic_utils

from fastgen.utils.distributed import rank0_only, synchronize, world_size
from fastgen.utils import logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.config import BaseConfig
    from fastgen.methods import FastGenModel


def to_wandb(
    tensor: torch.Tensor,
    rgb_range: float = 255.0,
    normalized: bool = False,
    max_plot_img: int = 16,
    max_plot_vid: int = 2,
    fps: int = 16,
    channel_before_time: bool = True,
    caption: str | None = None,
    vid_format: str = "mp4",
) -> wandb.Image | wandb.Video:
    """
    Convert a tensor to a wandb.Image or wandb.Video.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B,C,H,W], [B,T,C,H,W], or [B,T,C,H,W,D].
        rgb_range (float, optional): Output target RGB range (can almost definitely be kept as 255).
            Defaults to 255.0.
        normalized (bool, optional): Whether the tensor is normalized to [0,1]. Defaults to False which assumes [-1,1] range.
        max_plot_img (int, optional): Max number of images to plot. Defaults to 16.
        max_plot_vid (int, optional): Max number of videos to plot. Defaults to 2.
        fps (int, optional): Frames per second. Defaults to 8.
        channel_before_time (bool, optional): Whether the tensor is in the format [B,C,T,..]. Set False if the [B,T,C,..] format is used.
        caption (str, optional): Caption for the image or video. Defaults to None.
        vid_format (str, optional): Format of the video file. Defaults to "mp4".

    Returns:
        wandb.Image | wandb.Video: Format a tensor for logging to W&B.
    """

    if tensor.ndim == 5:
        max_plot = max_plot_vid
        if channel_before_time:
            tensor = tensor.permute(0, 2, 1, 3, 4)
    elif tensor.ndim == 4:
        max_plot = max_plot_img
    else:
        raise ValueError(f"Tensor must be 4 or 5 dimensional, but got {tensor.ndim} dimensions")

    # slice and adjust range
    if normalized:
        factor = rgb_range
        offset = 0.0
    else:
        factor = rgb_range / 2.0
        offset = rgb_range / 2.0
    tensor = tensor[:max_plot].mul(factor).add(offset).clip_(0, rgb_range).to(torch.uint8)

    # convert to wandb.Image or wandb.Video
    assert tensor.shape[-3] == 3, "Make sure that the data is in ..., C, H, W format"
    if tensor.ndim == 5:
        return wandb.Video(tensor.cpu().numpy(), fps=fps, format=vid_format, caption=caption)
    else:
        image_grid = torchvision.utils.make_grid(tensor, nrow=4, pad_value=1)
        image_grid = tv_F.to_pil_image(image_grid)
        return wandb.Image(image_grid, caption=caption)


@rank0_only
def init_wandb(config: BaseConfig):
    # wandb login
    wandb_credential = config.log_config.wandb_credential
    if os.path.isfile(wandb_credential):
        with open(wandb_credential, encoding="utf-8") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip("\n")
        logger.info(f"Loading WANDB_API_KEY from {wandb_credential}")

    wandb_config = config.log_config

    # Resume with or generate a wandb id
    logger.info(f"wandb_config.save_path: {wandb_config.save_path}")
    os.makedirs(wandb_config.save_path, exist_ok=True)
    wandb_id_path = f"{wandb_config.save_path}/wandb_id.txt"
    if os.path.isfile(wandb_id_path):
        with open(wandb_id_path, encoding="utf-8") as f:
            wandb_id = f.read().strip()
        logger.info(f"Resuming with an existing wandb id: {wandb_id}")
    else:
        wandb_id = wandb.util.generate_id()
        with open(wandb_id_path, "w", encoding="utf-8") as f:
            f.write(f"{wandb_id}\n")
        logger.info(f"Generating a wandb id: {wandb_id}")

    # Get config as plain dict
    config_resolved = serialize_config(config, return_type="dict")

    # Initialize the wandb library.
    wandb.init(
        id=wandb_id,
        project=wandb_config.project,
        group=wandb_config.group,
        name=wandb_config.name,
        config=config_resolved,
        dir=wandb_config.save_path,
        resume="allow",
        mode=wandb_config.wandb_mode,
    )

    # Save a copy of code to a wandb Artifact (this can be slow)
    # Make code upload optional to avoid distributed training delays
    upload_code = basic_utils.str2bool(os.getenv("WANDB_UPLOAD_CODE", "false"))
    if upload_code:
        logger.info("Uploading code to wandb (this may take a few minutes)...")
        wandb.run.log_code(".")
        logger.info("Code upload to wandb completed")
    else:
        logger.info("Wandb code upload disabled (set WANDB_UPLOAD_CODE=true to enable)")


@dataclass
class _LossDictRecord:
    loss_dict: dict = field(default_factory=dict)
    iter_count_dict: dict = field(default_factory=dict)

    def add(self, loss_dict: Optional[Dict[str, torch.Tensor]]) -> None:
        if loss_dict is not None:
            for loss_name, loss_val in loss_dict.items():
                self.loss_dict[loss_name] = self.loss_dict.get(loss_name, 0.0) + loss_val.float().item()
                self.iter_count_dict[loss_name] = self.iter_count_dict.get(loss_name, 0) + 1

    def reset(self) -> None:
        self.loss_dict = {}
        self.iter_count_dict = {}

    def gather_dict(self, dictionary: Dict[str, float | int]) -> Dict[str, float | int]:
        n_ranks = world_size()
        if n_ranks > 1:
            dict_list = [None for _ in range(n_ranks)]
            torch.distributed.all_gather_object(dict_list, dictionary)
            # from list of dicts to dict of summed values
            dictionary = {}
            for d in dict_list:
                for key, value in d.items():
                    dictionary[key] = dictionary.get(key, 0.0) + value
        return dictionary

    def get_stat(self) -> Dict[str, float]:
        # number of ranks that logged this loss
        rank_dict = self.gather_dict({k: 1 for k in self.loss_dict.keys()})
        # number of times this loss was computed
        count_dict = self.gather_dict(self.iter_count_dict)
        # sum of all losses
        loss_dict = self.gather_dict(self.loss_dict)

        avg_loss_dict = {}
        for loss_name, loss_val in loss_dict.items():
            count = count_dict.get(loss_name, 0)
            ranks = rank_dict.get(loss_name, 1)
            iter_count = count / ranks
            avg_loss = (loss_val / count) * (ranks / world_size()) if count > 0 else 0.0
            logger.info(f"avg_{loss_name}: {avg_loss:.4f}".ljust(30) + f"iter count: {iter_count}")
            avg_loss_dict[loss_name] = avg_loss
        self.reset()
        return avg_loss_dict


class WandbCallback(Callback):
    """
    The callback gets precision for data from model
    """

    def __init__(
        self,
        *args,
        validation_logging_step: int = 1,
        sample_logging_iter: Optional[int] = None,
        vid_format: str = "mp4",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.validation_logging_step = validation_logging_step
        self.sample_logging_iter = sample_logging_iter
        self.val_sample_map = None
        self.vid_format = vid_format
        self.loss_dict_record = _LossDictRecord()
        self.val_loss_dict_record = _LossDictRecord()

    def on_app_begin(self) -> None:
        assert hasattr(self, "config"), "Missing config in WandbCallback."
        init_wandb(self.config)
        self.offload_module_in_decoding = self.config.trainer.offload_module_in_decoding
        # disable offloading if using FSDP
        if self.config.trainer.fsdp:
            self.offload_module_in_decoding = False
        if self.sample_logging_iter is None:
            self.sample_logging_iter = self.config.trainer.logging_iter
        synchronize()

    @rank0_only
    def on_optimizer_step_begin(self, model: FastGenModel, iteration: int = 0) -> None:
        assert hasattr(self, "config"), "Missing config in WandbCallback."
        if iteration % self.config.trainer.logging_iter == 0:
            for name, scheduler in model.scheduler_dict.items():
                wandb.log({f"optimizer/lr_{name}": scheduler.get_last_lr()[0]}, step=iteration)

    def get_sample_map(
        self, model: FastGenModel, data_batch: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor | Callable]
    ) -> dict[str, wandb.Image | wandb.Video]:
        # Collect generated and real data and create copies to avoid modifying the original dicts
        sample_map = {}
        gen_rand = output_batch["gen_rand"]
        if isinstance(gen_rand, Callable):
            synchronize()
            gen_rand = gen_rand()
            synchronize()

        # Avoid modifying the original dicts
        data_batch = data_batch.copy()
        output_batch = output_batch.copy()

        # Decide whether we want to visualize multistep teacher generation
        if self.config.trainer.visualize_teacher:
            assert "input_rand" in output_batch, "We need to know the noise to visualize teacher generation"
            teacher_output = model.sample(
                model.teacher,
                output_batch["input_rand"][0:1],
                data_batch["condition"][0:1],  # e.g. text condition encoded by the text encoder
                data_batch["neg_condition"][0:1],  # e.g. negative text condition encoded by the text encoder
            )
            output_batch["gen_teacher"] = teacher_output

        # Decode to pixel if it's in latent space
        if hasattr(model.net, "init_preprocessors"):
            torch.cuda.empty_cache()
            device_nets = model.device

            has_vae = hasattr(model.net, "vae")
            if not has_vae:
                model.net.init_vae()
                model.net.vae.to(device=device_nets, dtype=model.precision)

            if self.offload_module_in_decoding:
                # offload the unneeded models to CPU (enable it if hitting OOM here)
                logger.info(
                    f"GPU Memory BEFORE moving nets to CPU: {torch.cuda.memory_allocated(device_nets) / 1024 ** 2:.2f} MB"
                )
                if hasattr(model, "fake_score"):
                    model.fake_score = model.fake_score.to("cpu")
                if hasattr(model, "teacher"):
                    model.teacher = model.teacher.to("cpu")
                logger.info(
                    f"GPU Memory AFTER moving nets to CPU: {torch.cuda.memory_allocated(device_nets) / 1024 ** 2:.2f} MB"
                )
                synchronize()

            with basic_utils.inference_mode(precision_amp=model.precision_amp_enc, device_type=device_nets.type):
                if "real" in data_batch:
                    # only generate one sample for video
                    limit = 1 if len(data_batch["real"].shape) == 5 else len(data_batch["real"])
                    data_batch["real"] = model.net.vae.decode(data_batch["real"][:limit])
                if isinstance(gen_rand, dict):
                    for k in gen_rand:
                        limit = 1 if len(gen_rand[k].shape) == 5 else len(gen_rand[k])
                        gen_rand[k] = model.net.vae.decode(gen_rand[k][:limit])
                else:
                    limit = 1 if len(gen_rand.shape) == 5 else len(gen_rand)
                    gen_rand = model.net.vae.decode(gen_rand[:limit])

                if "gen_teacher" in output_batch:
                    output_batch["gen_teacher"] = model.net.vae.decode(output_batch["gen_teacher"][:limit])
                if logger.LOG_LEVEL == "DEBUG" and "gen_rand_train" in output_batch:
                    output_batch["gen_rand_train"] = model.net.vae.decode(output_batch["gen_rand_train"][:limit])

            if not has_vae:
                del model.net.vae

            if self.offload_module_in_decoding:
                # move back fake_score to gpu
                if hasattr(model, "fake_score"):
                    model.fake_score = model.fake_score.to(device_nets)
                if hasattr(model, "teacher"):
                    model.teacher = model.teacher.to(device_nets)
                logger.info(
                    f"GPU Memory AFTER moving nets back to GPU: {torch.cuda.memory_allocated(device_nets) / 1024 ** 2:.2f} MB"
                )
                synchronize()

        if wandb.run:
            if (
                "condition_raw" in data_batch
                and isinstance(data_batch["condition_raw"], (list, tuple))
                and isinstance(data_batch["condition_raw"][0], str)
            ):
                caption = "\n".join(data_batch["condition_raw"][: len(gen_rand)])
            else:
                caption = None
            if isinstance(gen_rand, dict):
                for k in gen_rand:
                    sample_map[f"student/generation/{k}"] = to_wandb(
                        gen_rand[k], caption=caption, vid_format=self.vid_format
                    )
            else:
                sample_map["student/generation"] = to_wandb(gen_rand, caption=caption, vid_format=self.vid_format)
            if "real" in data_batch:
                sample_map["data/real"] = to_wandb(data_batch["real"], caption=caption, vid_format=self.vid_format)
            if "gen_teacher" in output_batch:
                sample_map["teacher/generation"] = to_wandb(
                    output_batch["gen_teacher"], caption=caption, vid_format=self.vid_format
                )
            if logger.LOG_LEVEL == "DEBUG" and "gen_rand_train" in output_batch:
                sample_map["student/generation_train"] = to_wandb(
                    output_batch["gen_rand_train"], caption=caption, vid_format=self.vid_format
                )

        return sample_map

    def log_sample_map(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        suffix: str = "",
        iteration: int = 0,
        group: str = "train",
    ) -> None:
        sample_map = self.get_sample_map(model, data_batch, output_batch)
        sample_map = {f"{group}_media/{k}{suffix}": v for k, v in sample_map.items()}
        if wandb.run:
            wandb.log(sample_map, step=iteration)
        synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    def log_stats(self, loss_dict_record: _LossDictRecord, iteration: int = 0, group: str = "train") -> None:
        logger.info(f"logging {group} stats at iteration {iteration}" + "-" * 20)
        # Collect distributed statistics
        avg_loss_dict = loss_dict_record.get_stat()
        stats = {f"{group}/{name}": val for name, val in avg_loss_dict.items()}
        base_info = {"optimizer/iteration": iteration}

        # log stats and base info
        if wandb.run:
            wandb.log(stats, step=iteration)
            wandb.log(base_info, step=iteration)

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        self.loss_dict_record.add(loss_dict)
        time_start = time.perf_counter()
        logged = False
        if iteration % self.config.trainer.logging_iter == 0 or iteration == 1:
            self.log_stats(self.loss_dict_record, iteration=iteration, group="train")
            logged = True
        if iteration % self.sample_logging_iter == 0 or iteration == 1:
            self.log_sample_map(model, data_batch, output_batch, iteration=iteration, group="train")
            logged = True
        if logged:
            time_taken = time.perf_counter() - time_start
            logger.info(f"WandB logging complete after {time_taken:.2f} seconds")

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
        self.val_loss_dict_record.add(loss_dict)

        if step % self.validation_logging_step == 0:
            self.log_sample_map(
                model, data_batch, output_batch, suffix=f"_{step}", iteration=iteration, group=f"val{idx}"
            )

    def on_validation_end(self, model: FastGenModel, iteration: int = 0, idx: int = 0) -> None:
        self.log_stats(self.val_loss_dict_record, iteration=iteration, group=f"val{idx}")
