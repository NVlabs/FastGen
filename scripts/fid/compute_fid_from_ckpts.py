# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample images from every saved checkpoint and compute their FID.

Checkpoints are discovered in ``trainer.checkpointer.save_dir``, which may be a
local path or an ``s3://`` URI. Only ``.pth`` files with a numeric name between
``eval.min_ckpt`` and ``eval.max_ckpt`` (inclusive) are evaluated; these include
DCP completion markers but not ``latest_ar.pth``.

Samples and ``fid.json`` are written to ``{log_config.save_path}/{eval.samples_dir}``,
a local directory that all ranks must share during distributed evaluation. For
each checkpoint, the student (its first EMA network when EMA is enabled) generates
``eval.num_samples`` images in ``iter_<ckpt>/``, and their FID against the
training dataset's reference statistics (see ``DATASETS``) is merged into
``fid.json``. Reruns skip checkpoints already in ``fid.json`` and reuse complete
sample directories; the samples of newly scored checkpoints are deleted afterwards.

With ``eval.save_images=True``, rank 0 instead writes one preview grid per
checkpoint to ``iter_<ckpt>/vis/<ckpt>.png``, and no FID is computed.

Example:
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=8 --standalone \
        scripts/fid/compute_fid_from_ckpts.py \
        --config fastgen/configs/experiments/EDM/config_dmd2_cifar10.py
"""

import argparse
import os
import tqdm
import time
import numpy as np
import PIL.Image
import json
from pathlib import Path
import shutil

import torch
from torchvision.utils import make_grid, save_image

import fastgen.utils.logging_utils as logger
from fastgen.configs.config import BaseConfig
from fastgen.configs.instantiate import instantiate
from fastgen.utils.checkpointer import Checkpointer
from fastgen.utils.io_utils import list_checkpoints
from fastgen.utils import basic_utils
from fastgen.utils.distributed import get_rank, world_size
from fastgen.utils.distributed.collectives import broadcast_rank0_result, clean_up, synchronize
from fastgen.utils.scripts import setup, parse_args

from scripts.fid.fid import calc

DATASETS = {
    "cifar10-32x32.zip": "cifar10",
    "imagenet-64x64.zip": "imagenet64",
    "imagenet-64x64-edmv2.zip": "imagenet64-edmv2",
    "imagenet_256_sd.zip": "imagenet256",
}


def remove_iter_dirs(root_dir: str, ckpt_num_visited: list) -> None:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"✗ Folder not found: {root.resolve()}")

    removed = 0
    visited_set = set(str(x) for x in ckpt_num_visited)
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("iter_"):
            suffix = path.name[5:]  # remove "iter_" prefix
            if suffix in visited_set:
                try:
                    shutil.rmtree(path)
                    removed += 1
                except Exception as exc:
                    raise RuntimeError(f"  – Could not delete {path}: {exc}")

    logger.info(f"✓ Removed {removed} directorie(s) from {root.resolve()}")


def _completed_fid_iterations(path: str) -> list[int]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        history = json.load(file)
    assert "ckpt_num" in history and "fid" in history
    assert len(history["ckpt_num"]) == len(history["fid"])
    return history["ckpt_num"]


def main(config: BaseConfig):
    # fix seeds
    basic_utils.set_random_seed(config.trainer.seed, by_rank=True)

    # Initialize the model. ``config.model`` is passed as an explicit kwarg
    # so OmegaConf doesn't coerce the attrs subtree (and
    # ``config.model.inference`` was already set to True by
    # ``setup(..., evaluation=True)``).
    model = instantiate(config.model_class, config=config.model)

    # Initialize the checkpointer and samples directory.
    checkpointer = Checkpointer(config.trainer.checkpointer)
    samples_dir = os.path.join(config.log_config.save_path, config.eval.samples_dir)

    try:
        dataset = DATASETS[config.dataloader_train.dataset_path.split("/")[-1]]
    except (KeyError, AttributeError):
        dataset = DATASETS[config.dataloader_train.datatags[0].split("/")[-1]]

    # Initialize the batches.
    num_batches = (
        (config.eval.num_samples - 1) // (config.dataloader_train.batch_size * world_size()) + 1
    ) * world_size()
    logger.info(f"world size: {world_size()}, num batches: {num_batches}")
    all_batches = torch.as_tensor(np.arange(config.eval.num_samples)).tensor_split(num_batches)
    rank_batches = all_batches[get_rank() :: world_size()]

    # Every rank must enter checkpoint loads in the same order, including for S3 listings.
    checkpoints = broadcast_rank0_result(
        lambda: list_checkpoints(config.trainer.checkpointer.save_dir), "FID checkpoint discovery"
    )

    # Resume decisions must agree before any rank enters a collective checkpoint load.
    fid_runs_file = f"{samples_dir}/fid.json"
    runs_visited = broadcast_rank0_result(lambda: _completed_fid_iterations(fid_runs_file), "FID sweep history loading")

    logger.info(f"Evaluating student with {model.net.noise_scheduler.default_student_steps} steps.")
    # Modules can only be prepared once; each checkpoint load below copies into the prepared tensors.
    model.on_train_begin()
    # sweep over all checkpoints
    for ckpt_num, ckpt_path in checkpoints:
        # Load network.
        synchronize()

        if ckpt_num < config.eval.min_ckpt or ckpt_num > config.eval.max_ckpt:
            continue

        # check if we evaluated fid already
        if ckpt_num in runs_visited:
            logger.info(f"Skipping checkpoint {ckpt_path} since we already evaluated its FID in fid.json")
            continue

        outdir = os.path.join(samples_dir, f"iter_{ckpt_num}")
        samples_complete = broadcast_rank0_result(
            lambda: not config.eval.save_images
            and os.path.exists(outdir)
            and sum(path.is_file() for path in Path(outdir).rglob("*.png")) >= config.eval.num_samples,
            "FID sweep sample discovery",
        )
        if samples_complete:
            logger.info(f"Skipping checkpoint {ckpt_path} since there are already {config.eval.num_samples} samples")
            continue

        logger.info(f'Loading model from "{ckpt_path}"...')
        checkpointer.load(
            model.model_dict,
            path=ckpt_path,
        )
        inference_net = getattr(model, model.use_ema[0]) if model.use_ema else model.net
        inference_net.eval()
        ctx = dict(device=model.device, dtype=model.precision)

        # ``on_train_begin`` already built and placed the preprocessors on their host
        # net; read the VAE off that host rather than building a second copy on the EMA.
        vae = model.net_preprocessors.preprocessors.get("vae")

        # Loop over batches.
        conditional = inference_net.label_dim > 0
        logger.info(
            f"{'Conditional' if conditional else 'Unconditional'} sampling of {config.eval.num_samples} "
            f"images to {outdir}..."
        )

        for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(get_rank() != 0)):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Prepare condition data.
            if conditional:
                # One-hot labels pair with the zero null negative, as in ``ImageDataset``.
                class_ids = torch.randint(inference_net.label_dim, size=[batch_size], device=model.device)
                condition = torch.nn.functional.one_hot(class_ids, num_classes=inference_net.label_dim).float()
                data = {"condition": condition, "neg_condition": torch.zeros_like(condition)}
            else:
                data = {}

            data = model.prepare_batch(data, inference_mode=True, materialize_real=False)

            # Pick noise.
            noise = torch.randn([batch_size, *model.input_shape], **ctx)

            images = model.generator_fn(
                inference_net,
                noise,
                condition=data.get("condition"),
                neg_condition=data.get("neg_condition"),
            )

            if vae is not None:
                with basic_utils.inference_mode(
                    vae, precision_amp=model.precision_amp_enc, device_type=model.device.type
                ):
                    images = vae.decode(images)

            if config.eval.save_images:
                if get_rank() == 0:
                    visdir = os.path.join(outdir, "vis")
                    os.makedirs(visdir, exist_ok=True)
                    images_ = (images + 1) / 2.0
                    image_grid = make_grid(images_, nrow=int(np.sqrt(len(images))), padding=0)
                    save_image(image_grid, os.path.join(visdir, f"{ckpt_num}.png"))
                    logger.info(f"Saved {len(images)} images to {visdir}/{ckpt_num}.png")
                break

            # Save images.
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f"{seed - seed % 1000:06d}")
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f"{seed:06d}.png")
                PIL.Image.fromarray(image_np).save(image_path)

    if config.eval.save_images:
        return

    synchronize()
    time.sleep(10)

    calc(
        samples_dir,
        config.eval.num_samples,
        config.trainer.seed,
        config.eval.min_ckpt,
        config.eval.max_ckpt,
        config.dataloader_train.batch_size,
        dataset,
        device=model.device,
    )

    if get_rank() == 0:
        # remove generated samples to free the occupied space
        runs_visited_new = [
            ckpt_num
            for ckpt_num in _completed_fid_iterations(fid_runs_file)
            if ckpt_num >= config.eval.min_ckpt and ckpt_num <= config.eval.max_ckpt and ckpt_num not in runs_visited
        ]
        if runs_visited_new:
            remove_iter_dirs(samples_dir, runs_visited_new)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FID evaluation")
    args = parse_args(parser)
    config = setup(args, evaluation=True)
    synchronize()
    main(config)

    clean_up()

# ----------------------------------------------------------------------------
