# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import glob
import pickle
import json
from pathlib import Path
import tqdm
import numpy as np
import scipy.linalg

import torch

from scripts.fid.inception import InceptionV3
from fastgen.datasets.class_cond_dataset import ImageFolderDataset
from fastgen.utils.cache_utils import atomic_write
from fastgen.utils.distributed import get_rank, is_rank0, world_size
from fastgen.utils.distributed.collectives import broadcast_rank0_result, raise_across_ranks
import fastgen.utils.logging_utils as logger
from fastgen.configs.data import DATA_ROOT_DIR


def calculate_inception_stats(
    detector_net,
    feature_dim,
    image_path,
    num_expected=None,
    seed=0,
    max_batch_size=64,
    num_workers=3,
    prefetch_factor=2,
    device=torch.device("cuda"),
):
    # List images.
    logger.info(f'Loading images from "{image_path}"...')
    with raise_across_ranks("FID image discovery"):
        dataset_obj = ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
        if num_expected is not None and len(dataset_obj) < num_expected:
            raise ValueError(f"Found {len(dataset_obj)} images, but expected at least {num_expected}")
        if len(dataset_obj) < 2:
            raise ValueError(f"Found {len(dataset_obj)} images, but need at least 2 to compute statistics")

    # Accumulate statistics.
    logger.info(f"Calculating statistics for {len(dataset_obj)} images...")
    with raise_across_ranks("FID feature extraction"):
        num_batches = ((len(dataset_obj) - 1) // (max_batch_size * world_size()) + 1) * world_size()
        all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
        rank_batches = [batch for batch in all_batches[get_rank() :: world_size()] if len(batch)]
        data_loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_sampler=rank_batches,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers else None,
        )
        mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
        sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
        for data in tqdm.tqdm(data_loader, unit="batch", disable=(get_rank() != 0)):
            images = data["real_raw"]
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])

            with torch.no_grad():
                features = detector_net(images.to(device))
                features = features.to(torch.float64)

            mu += features.sum(0)
            sigma += features.T @ features

    # Calculate grand totals.
    if world_size() > 1:
        torch.distributed.all_reduce(mu)
        torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    mu, sigma, mu_ref, sigma_ref = (np.asarray(value) for value in (mu, sigma, mu_ref, sigma_ref))
    if mu.ndim != 1 or mu_ref.shape != mu.shape or sigma.shape != (mu.size, mu.size) or sigma_ref.shape != sigma.shape:
        raise ValueError("FID requires equal-length mean vectors and matching square covariance matrices")
    if not all(np.isrealobj(value) and np.isfinite(value).all() for value in (mu, sigma, mu_ref, sigma_ref)):
        raise ValueError("FID statistics must be real and finite")
    m = np.square(mu - mu_ref).sum()
    s = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref))
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    if not np.isfinite(fid):
        raise ValueError("FID computation produced a non-finite result")
    return float(np.real(fid))


def calc(samples_dir, num_expected, seed, min_ckpt, max_ckpt, batch, dataset, device=torch.device("cuda")):
    """Calculate FID for a given set of images."""

    ref = None
    if dataset == "cifar10":
        ref_path = f"{DATA_ROOT_DIR}/fid-refs/cifar10-32x32.npz"
    elif dataset == "imagenet64":
        ref_path = f"{DATA_ROOT_DIR}/fid-refs/imagenet-64x64.npz"
    elif dataset == "imagenet64-edmv2":
        ref_path = f"{DATA_ROOT_DIR}/fid-refs/imagenet-64x64-edmv2.pkl"
    elif dataset == "imagenet256":
        ref_path = f"{DATA_ROOT_DIR}/fid-refs/imagenet_256.pkl"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    logger.info(f'Loading dataset reference statistics from "{ref_path}"...')
    with raise_across_ranks("FID reference loading"):
        if is_rank0():
            if ref_path.endswith(".npz"):
                with np.load(ref_path) as f:
                    ref = dict(f)
            else:
                assert ref_path.endswith(".pkl"), f"Unknown file type: {ref_path}"
                with open(ref_path, "rb") as f:
                    ref = pickle.load(f)["fid"]

    def load_evaluation_state():
        stats = glob.glob(f"{samples_dir}/iter_[0-9]*")
        stats = [path for path in stats if Path(path).is_dir() and Path(path).name[5:].isdecimal()]
        stats.sort(key=lambda path: int(Path(path).name[5:]))
        metric_scores = {"ckpt_num": [], "fid": []}
        if os.path.exists(f"{samples_dir}/fid.json"):
            with open(f"{samples_dir}/fid.json", "r") as f:
                metric_scores = json.load(f)
            logger.info(f"metric_scores in the existing file: {metric_scores}")
        return stats, metric_scores["ckpt_num"], metric_scores["fid"]

    stats, ckpt_num_list, fid_list = broadcast_rank0_result(load_evaluation_state, "FID evaluation discovery")

    # Load Inception-v3 model.
    logger.info("Loading Inception-v3 model...")
    feature_dim = 2048
    with raise_across_ranks("FID detector initialization"):
        detector_net = InceptionV3().to(device)
        detector_net.eval()

    for path in stats:
        ckpt_num = int(Path(path).name[5:])

        if ckpt_num in ckpt_num_list:
            logger.info(f"ckpt {ckpt_num} already has metrics. Skip.")
            continue

        if ckpt_num < min_ckpt or ckpt_num > max_ckpt:
            continue

        mu, sigma = calculate_inception_stats(
            detector_net,
            feature_dim,
            image_path=path,
            num_expected=num_expected,
            seed=seed,
            max_batch_size=batch,
            device=device,
        )

        logger.info(f"Calculating FID for {path}... ")
        with raise_across_ranks("FID score computation"):
            if is_rank0():
                fid = calculate_fid_from_inception_stats(mu, sigma, ref["mu"], ref["sigma"])
                logger.info(f"path: {path}")
                logger.info(f"FID: {fid}")
                logger.info("=" * 20)
                fid_list.append(fid)
        ckpt_num_list.append(ckpt_num)

    # dump the FID scores to a json file
    with raise_across_ranks("FID result writing"):
        if is_rank0():
            metric_scores = {}
            # read metrics again in case another process altered file
            if os.path.exists(f"{samples_dir}/fid.json"):
                with open(f"{samples_dir}/fid.json", "r") as f:
                    metric_scores = json.load(f)
                metric_scores = {ckpt: fid for ckpt, fid in zip(metric_scores["ckpt_num"], metric_scores["fid"])}

            # merge metrics
            for ckpt, fid in zip(ckpt_num_list, fid_list):
                metric_scores[ckpt] = fid
            metric_scores = sorted(metric_scores.items(), key=lambda x: x[0])
            metric_scores = {"ckpt_num": [ckpt for ckpt, _ in metric_scores], "fid": [fid for _, fid in metric_scores]}
            # Publish the ledger atomically so an interruption cannot truncate it.
            atomic_write(f"{samples_dir}/fid.json", lambda f: f.write(json.dumps(metric_scores).encode("utf-8")))
