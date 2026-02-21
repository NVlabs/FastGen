# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video generation inference script.

Supports:
- Text-to-video (T2V): Wan 2.1/2.2
- Image-to-video (I2V): Wan I2V
- Video-to-video (V2V): VACE Wan, Self-Forcing
- Video2World: Cosmos Predict2

Examples:

    # T2V: eval teacher only (Wan)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py --do_student_sampling False \\
        --config fastgen/configs/experiments/WanT2V/config_dmd2.py \\
        - trainer.seed=1 trainer.ddp=True model.guidance_scale=5.0 log_config.name=wan_t2v_inference

    # I2V: image-to-video (Wan I2V)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py --do_student_sampling False \\
        --input_image_file scripts/inference/prompts/source_image_paths.txt \\
        --config fastgen/configs/experiments/WanI2V/config_dmd2_14b.py \\
        - trainer.seed=1 trainer.ddp=True model.guidance_scale=5.0 log_config.name=wan_i2v_inference

    # V2V: video-to-video with VACE
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py --do_student_sampling False \\
        --source_video_file scripts/inference/prompts/source_video_paths.txt \\
        --config fastgen/configs/experiments/WanV2V/config_sft_latent.py \\
        - trainer.seed=1 trainer.ddp=True model.guidance_scale=5.0 log_config.name=vace_wan_inference

    # Video2World: Cosmos Predict2
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py --do_student_sampling False \\
        --input_image_file scripts/inference/prompts/source_image_paths.txt --num_conditioning_frames 1 \\
        --config fastgen/configs/experiments/CosmosPredict2/config_sft.py \\
        - trainer.seed=1 trainer.ddp=True model.guidance_scale=5.0 model.net.is_video2world=True \\
        log_config.name=cosmos_v2w_inference

    # Eval with skip-layer guidance (SLG)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py --do_student_sampling False \\
        --config fastgen/configs/experiments/WanT2V/config_dmd2.py \\
        - trainer.seed=1 trainer.ddp=True model.guidance_scale=6.0 model.skip_layers=[10] \\
        log_config.name=wan_slg_inference

    # Eval student and teacher together
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py --ckpt_path /path/to/checkpoint.pth \\
        --do_student_sampling True --do_teacher_sampling True \\
        --config fastgen/configs/experiments/WanT2V/config_dmd2.py \\
        - trainer.seed=1 trainer.ddp=True log_config.name=wan_student_teacher_inference
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import imageio.v3 as iio
import numpy as np
import torch
from tqdm.auto import tqdm

from fastgen.configs.config import BaseConfig
from fastgen.networks.WanI2V import WanI2V
from fastgen.networks.cosmos_predict2 import CosmosPredict2
import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed import clean_up, is_rank0, world_size
from fastgen.utils import basic_utils
from fastgen.utils.scripts import parse_args, setup
from fastgen.utils.checkpointer import FSDPCheckpointer
from fastgen.third_party.wan_prompt_expand.prompt_expand import QwenPromptExpander
from fastgen.datasets.wds_dataloaders import transform_video
from scripts.inference.inference_utils import (
    expand_path,
    load_prompts,
    init_model,
    load_checkpoint,
    cleanup_unused_modules,
    setup_inference_modules,
    add_common_args,
)

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


def load_video_frames(video_path: str, num_frames: int, height: int, width: int) -> Optional[torch.Tensor]:
    """
    Load video, align spatial preprocessing with dataset pipeline via transform_video,
    and return tensor shaped [1, C, T, H, W] in [-1, 1].
    """
    try:
        frames_np = iio.imread(video_path, plugin="pyav")  # [T, H, W, C], uint8
    except Exception as e:
        logger.error(f"Failed to read video file: {video_path} with error {e}")
        return None

    if frames_np is None or len(frames_np) == 0:
        logger.error(f"No frames decoded from video file: {video_path}")
        return None

    # Ensure enough frames by padding with the last frame (avoid tiling entire clip)
    T = len(frames_np)
    if T < num_frames:
        pad_count = num_frames - T
        last = frames_np[-1:]
        frames_np = np.concatenate([frames_np, np.repeat(last, pad_count, axis=0)], axis=0)
    else:
        # Use a centered segment to better match training decode behavior
        start = max(0, (T - num_frames) // 2)
        frames_np = frames_np[start : start + num_frames]

    # Convert to torch and apply the same preprocessing as training
    frames_t = torch.from_numpy(frames_np)  # [T, H, W, C], uint8
    out = transform_video(frames_t, sequence_length=num_frames, img_size=(width, height))
    frames_tensor = out["real"]  # [C, T, H, W], float in [-1, 1]
    return frames_tensor.unsqueeze(0)


def load_conditioning_image(
    image_path: str, height: int, width: int, num_latent_frames: int = 1
) -> Optional[torch.Tensor]:
    """
    Load an image as conditioning frames for image-to-video generation.

    The image is replicated to create the pixel frames needed by temporal VAE
    (which has 4x temporal compression). For N latent frames, we need (N-1)*4+1 pixel frames.

    Args:
        image_path: Path to the input image.
        height: Target height in pixels.
        width: Target width in pixels.
        num_latent_frames: Number of latent frames to generate from the image (default 1).

    Returns:
        Tensor of shape [1, C, T, H, W] in [-1, 1] range, where T is the number of
        pixel frames needed for the requested latent frames. Returns None on failure.
    """
    try:
        # Load image using imageio
        img_np = iio.imread(image_path)  # [H, W, C], uint8
    except Exception as e:
        logger.error(f"Failed to read image file: {image_path} with error {e}")
        return None

    if img_np is None:
        logger.error(f"Failed to load image: {image_path}")
        return None

    # Handle grayscale images
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[-1] == 4:  # RGBA
        img_np = img_np[..., :3]

    # For temporal VAE with 4x compression, we need (T-1)*4+1 pixel frames for T latent frames
    # For 1 latent frame: 1 pixel frame; For 2 latent frames: 5 pixel frames, etc.
    num_pixel_frames = (num_latent_frames - 1) * 4 + 1 if num_latent_frames > 1 else 1

    # Replicate image to create video-like input
    # Shape: [T, H, W, C]
    frames_np = np.stack([img_np] * num_pixel_frames, axis=0)

    # Convert to torch and apply preprocessing
    frames_t = torch.from_numpy(frames_np)  # [T, H, W, C], uint8
    out = transform_video(frames_t, sequence_length=num_pixel_frames, img_size=(width, height))
    frames_tensor = out["real"]  # [C, T, H, W], float in [-1, 1]
    return frames_tensor.unsqueeze(0)


def prepare_wani2v_condition(
    conditioning_frames: torch.Tensor,
    conditioning_latents: torch.Tensor,
    condition: torch.Tensor,
    neg_condition: Optional[torch.Tensor],
    model: FastGenModel,
    vae: torch.nn.Module,
    t_latent: int,
    use_concat_mask: bool,
) -> tuple:
    """Prepare condition dicts for WanI2V models.

    Args:
        conditioning_frames: Raw pixel frames [B, C, T, H, W] in [-1, 1]
        conditioning_latents: VAE-encoded latents of conditioning frames
        condition: Text embeddings for positive prompt
        neg_condition: Text embeddings for negative prompt
        model: The model instance (for precision and device info)
        vae: VAE model for encoding
        t_latent: Total number of latent frames
        use_concat_mask: Whether model uses concat mask (Wan 2.1) or frame replacement (Wan 2.2)

    Returns:
        Tuple of (condition_dict, neg_condition_dict, i2v_tag)
    """
    if use_concat_mask:
        # Wan 2.1 14B: first_frame_cond must be created in pixel space then encoded
        # This matches training: [first_frame, zeros, zeros, ...] -> VAE encode
        num_pixel_frames = (t_latent - 1) * 4 + 1
        B, C_pixel, _, H_pixel, W_pixel = conditioning_frames.shape
        pixel_cond = torch.zeros(
            B,
            C_pixel,
            num_pixel_frames,
            H_pixel,
            W_pixel,
            device=conditioning_frames.device,
            dtype=conditioning_frames.dtype,
        )
        pixel_cond[:, :, 0] = conditioning_frames[:, :, 0]  # First frame only
        # Encode through VAE (zeros become VAE-encoded zeros, not latent zeros)
        with basic_utils.inference_mode(vae, precision_amp=model.precision_amp_infer, device_type=model.device.type):
            first_frame_cond = vae.encode(pixel_cond)
        logger.info(f"Wan 2.1 I2V: created first_frame_cond via VAE, shape {first_frame_cond.shape}")
    else:
        # Wan 2.2 5B: pad with zeros in latent space (simpler, no concat)
        first_frame_cond = torch.zeros(
            1,
            conditioning_latents.shape[1],
            t_latent,
            conditioning_latents.shape[3],
            conditioning_latents.shape[4],
            device=conditioning_latents.device,
            dtype=conditioning_latents.dtype,
        )
        first_frame_cond[:, :, : conditioning_latents.shape[2]] = conditioning_latents

    condition_dict = {"text_embeds": condition, "first_frame_cond": first_frame_cond}
    neg_condition_dict = {"text_embeds": neg_condition, "first_frame_cond": first_frame_cond}

    # Add image encoder embeddings if available (Wan 2.1 14B I2V)
    if hasattr(model.net, "image_encoder"):
        with basic_utils.inference_mode(
            model.net.image_encoder, precision_amp=model.precision_amp_infer, device_type=model.device.type
        ):
            # Use first pixel frame for image encoder
            first_pixel_frame = conditioning_frames[:, :, 0:1]  # [B, C, 1, H, W]
            img_embeds = model.net.image_encoder.encode(first_pixel_frame[:, :, 0])
        # Ensure embeddings are on the correct device and dtype
        img_embeds = img_embeds.to(device=model.device, dtype=model.precision)
        condition_dict["encoder_hidden_states_image"] = img_embeds
        neg_condition_dict["encoder_hidden_states_image"] = img_embeds

    return condition_dict, neg_condition_dict, "_i2v"


def prepare_cosmos_v2w_condition(
    conditioning_latents: torch.Tensor,
    condition: torch.Tensor,
    neg_condition: Optional[torch.Tensor],
    latent_shape: Sequence[int],
    num_conditioning_frames: int,
) -> tuple:
    """Prepare condition dicts for CosmosPredict2 video2world mode.

    Args:
        conditioning_latents: VAE-encoded latents of conditioning frames
        condition: Text embeddings for positive prompt
        neg_condition: Text embeddings for negative prompt
        latent_shape: Shape of latent tensor [C, T, H, W]
        num_conditioning_frames: Number of frames to condition on

    Returns:
        Tuple of (condition_dict, neg_condition_dict, i2v_tag)
    """
    t_latent, h_latent, w_latent = latent_shape[1], latent_shape[2], latent_shape[3]

    # Create condition mask: 1 for conditioning frames, 0 for generated
    condition_mask = torch.zeros(
        1, 1, t_latent, h_latent, w_latent, device=conditioning_latents.device, dtype=conditioning_latents.dtype
    )
    condition_mask[:, :, :num_conditioning_frames] = 1.0

    # Build condition dict for forward() compatibility
    condition_dict = {
        "text_embeds": condition,
        "conditioning_latents": conditioning_latents,
        "condition_mask": condition_mask,
    }
    neg_condition_dict = {
        "text_embeds": neg_condition,
        "conditioning_latents": conditioning_latents,
        "condition_mask": condition_mask,
    }

    return condition_dict, neg_condition_dict, f"_v2w{num_conditioning_frames}"


def prepare_vacewan_condition(
    source_video_path: str,
    depth_latent_path: Optional[str],
    model: FastGenModel,
    latent_shape: Sequence[int],
    condition: torch.Tensor,
    neg_condition: Optional[torch.Tensor],
    ctx: dict,
    num_segments: int = 1,
    overlap_frames: int = 0,
) -> tuple:
    """Prepare condition dicts for VACE Wan models (depth-to-video).

    Args:
        source_video_path: Path to the source video for conditioning
        depth_latent_path: Optional path to precomputed depth latents
        model: The model instance
        latent_shape: Shape of latent tensor [C, T, H, W]
        condition: Text embeddings for positive prompt
        neg_condition: Text embeddings for negative prompt
        ctx: Device/dtype context dict
        num_segments: Number of segments for extrapolation (default 1)
        overlap_frames: Number of overlapping latent frames between segments (default 0)

    Returns:
        Tuple of (condition_dict, neg_condition_dict)
    """
    t_latent = latent_shape[1]

    # For extrapolation with multiple segments, compute total latent frames needed
    # Total = segment_frames + (num_segments - 1) * (segment_frames - overlap_frames)
    #       = num_segments * segment_frames - (num_segments - 1) * overlap_frames
    if num_segments > 1:
        total_latent_frames = num_segments * t_latent - (num_segments - 1) * overlap_frames
    else:
        total_latent_frames = t_latent

    # Convert latent frames to pixel frames (VAE has 4x temporal compression)
    target_frames = (total_latent_frames - 1) * 4 + 1

    # VAE spatial compression factor: 16 for Wan 2.2 (48ch latents), 8 for others
    vae_spatial_factor = 16 if latent_shape[0] == 48 else 8
    height = latent_shape[2] * vae_spatial_factor
    width = latent_shape[3] * vae_spatial_factor

    video = load_video_frames(source_video_path, num_frames=target_frames, height=height, width=width)  # [-1, 1]
    video = video.to(**ctx)

    if depth_latent_path is None:
        depth_latent = model.net.prepare_vid_conditioning(video=video, condition_latents=None)
    else:
        depth_latent = torch.load(depth_latent_path)
        depth_latent = depth_latent[:, :total_latent_frames]
        depth_latent = depth_latent.unsqueeze(0)
        depth_latent = depth_latent.to(**ctx)
        depth_latent = model.net.prepare_vid_conditioning(video=video, condition_latents=depth_latent)

    condition_dict = {"text_embeds": condition, "vid_context": depth_latent}
    neg_condition_dict = {"text_embeds": neg_condition, "vid_context": depth_latent}

    return condition_dict, neg_condition_dict


def prepare_i2v_condition(
    input_image_path: str,
    model: FastGenModel,
    vae: Optional[torch.nn.Module],
    latent_shape: Sequence[int],
    condition: torch.Tensor,
    neg_condition: Optional[torch.Tensor],
    num_conditioning_frames: int,
    ctx: dict,
) -> tuple:
    """Load and prepare I2V/video2world conditioning from an input image.

    Args:
        input_image_path: Path to the input image
        model: The model instance
        vae: VAE model for encoding
        latent_shape: Shape of latent tensor [C, T, H, W]
        condition: Text embeddings for positive prompt
        neg_condition: Text embeddings for negative prompt
        num_conditioning_frames: Number of frames to condition on
        ctx: Device/dtype context dict

    Returns:
        Tuple of (condition, neg_condition, i2v_tag) where condition/neg_condition
        may be updated dicts for I2V mode, or unchanged if loading fails.
    """
    i2v_tag = ""

    if not input_image_path or not Path(input_image_path).exists():
        if input_image_path:
            logger.warning(f"Conditioning image not found: {input_image_path}")
        return condition, neg_condition, i2v_tag

    # VAE spatial compression factor: 16 for Wan 2.2 (48ch latents), 8 for others
    vae_spatial_factor = 16 if latent_shape[0] == 48 else 8
    height = latent_shape[2] * vae_spatial_factor
    width = latent_shape[3] * vae_spatial_factor

    # Load and preprocess input image
    conditioning_frames = load_conditioning_image(
        input_image_path,
        height=height,
        width=width,
        num_latent_frames=num_conditioning_frames,
    )

    if conditioning_frames is None or vae is None:
        logger.warning(f"Failed to encode conditioning image: {input_image_path}")
        return condition, neg_condition, i2v_tag

    conditioning_frames = conditioning_frames.to(**ctx)
    with basic_utils.inference_mode(vae, precision_amp=model.precision_amp_infer, device_type=model.device.type):
        conditioning_latents = vae.encode(conditioning_frames)
    logger.info(f"I2V: encoded image to latents shape {conditioning_latents.shape}")

    if getattr(model.net, "is_i2v", False):
        assert isinstance(model.net, WanI2V), f"Expected WanI2V model but got {type(model.net).__name__}"
        # WanI2V model
        use_concat_mask = getattr(model.net, "concat_mask", False)
        return prepare_wani2v_condition(
            conditioning_frames=conditioning_frames,
            conditioning_latents=conditioning_latents,
            condition=condition,
            neg_condition=neg_condition,
            model=model,
            vae=vae,
            t_latent=latent_shape[1],
            use_concat_mask=use_concat_mask,
        )
    elif getattr(model.net, "is_video2world", False):
        assert isinstance(
            model.net, CosmosPredict2
        ), f"Expected CosmosPredict2 model but got {type(model.net).__name__}"
        # CosmosPredict2 video2world
        return prepare_cosmos_v2w_condition(
            conditioning_latents=conditioning_latents,
            condition=condition,
            neg_condition=neg_condition,
            latent_shape=latent_shape,
            num_conditioning_frames=num_conditioning_frames,
        )
    else:
        raise NotImplementedError(f"I2V mode not implemented for {type(model.net).__name__}")


def expand_prompts_with_qwen(
    prompts: list[str],
    model_name: str,
    device: torch.device,
    seed: int,
) -> list[str]:
    """Expand prompts using Qwen model on rank 0 and broadcast to all ranks.

    Args:
        prompts: List of prompts to expand
        model_name: Qwen model name
        device: Device to run on
        seed: Random seed for prompt expansion

    Returns:
        List of expanded prompts
    """
    logger.info("Expanding prompts on rank 0 ...")
    if is_rank0():
        prompt_expander = QwenPromptExpander(
            model_name=model_name,
            is_vl=False,
            device=device,
        )
        for prompt_idx in tqdm(range(len(prompts))):
            logger.debug(f"Expanding prompt {prompts[prompt_idx]} with seed {seed}")
            basic_utils.set_random_seed(seed)
            prompt_output = prompt_expander(prompts[prompt_idx], tar_lang="en", seed=seed)
            logger.info(f"Expanded prompt: {prompt_output.prompt}")
            prompts[prompt_idx] = prompt_output.prompt
        # Free memory
        del prompt_expander
        gc.collect()
        torch.cuda.empty_cache()
    else:
        prompts = [None] * len(prompts)

    if world_size() > 1:
        torch.distributed.broadcast_object_list(prompts, src=0)

    return prompts


def main(args, config: BaseConfig):
    # Load prompts
    pos_prompt_set = load_prompts(args.prompt_file, relative_to="cwd")

    # Prompt expansion if specified
    if args.prompt_expand_model:
        pos_prompt_set = expand_prompts_with_qwen(
            pos_prompt_set,
            args.prompt_expand_model,
            torch.device(config.model.device),
            args.prompt_expand_model_seed,
        )

    # Load depth latent paths
    depth_latent_paths = None
    if args.depth_latent_file is not None:
        depth_latent_path = expand_path(args.depth_latent_file, relative_to="cwd")
        if depth_latent_path.is_file():
            with depth_latent_path.open("r") as f:
                depth_latent_paths = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"depth_latent_file: {depth_latent_path} not found!")

    # Load source video paths
    source_video_paths = None
    if args.source_video_file is not None:
        source_video_path = expand_path(args.source_video_file, relative_to="cwd")
        if source_video_path.is_file():
            with source_video_path.open("r") as f:
                source_video_paths = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"source_video_path: {source_video_path} not found!")

    # Load input images for I2V mode (or video2world mode in cosmos)
    input_image_paths = None
    if args.input_image_file is not None:
        input_image_file_path = expand_path(args.input_image_file, relative_to="cwd")
        if input_image_file_path.is_file():
            with input_image_file_path.open("r") as f:
                input_image_paths = [line.strip() for line in f.readlines() if line.strip()]

            # Align with prompts: repeat last image if fewer images than prompts
            num_prompts = len(pos_prompt_set)
            num_images = len(input_image_paths)
            if num_images < num_prompts:
                last_image = input_image_paths[-1] if input_image_paths else ""
                input_image_paths.extend([last_image] * (num_prompts - num_images))
                logger.info(f"I2V: extended {num_images} images to {num_prompts} by repeating last image")
            elif num_images > num_prompts:
                input_image_paths = input_image_paths[:num_prompts]
                logger.info(f"I2V: truncated {num_images} images to {num_prompts} prompts")

            logger.info(f"I2V mode: {len(input_image_paths)} input images for {num_prompts} prompts")
        else:
            raise FileNotFoundError(f"input_image_file_path: {input_image_file_path} not found!")

    # Fix sampling seeds
    seed = basic_utils.set_random_seed(config.trainer.seed, by_rank=True)

    # Initialize model and checkpointer
    model = init_model(config)
    # FSDP checkpointer falls back to basic checkpointer if the checkpoint ends with .pth
    checkpointer = FSDPCheckpointer(config.trainer.checkpointer)

    # Load checkpoint
    ckpt_iter, save_dir = load_checkpoint(checkpointer, model, args.ckpt_path, config)

    if ckpt_iter is None and args.do_student_sampling:
        logger.warning(f"Performing {model.config.student_sample_steps}-step generation on the non-distilled model")

    if args.video_save_dir:  # overwrite the save_dir
        save_dir = args.video_save_dir
        logger.info(f"video_save_dir: {save_dir}")

    save_dir = Path(save_dir)
    prompt_name = Path(args.prompt_file).stem
    if args.prompt_expand_model:
        prompt_name += f"_{args.prompt_expand_model}"
    save_dir = save_dir / prompt_name

    save_video_kwargs = {"precision_amp": model.precision_amp_infer, "save_as_gif": args.save_as_gif, "fps": args.fps}
    if args.save_high_quality:
        save_video_kwargs = {
            "quality": 18,
            "preset": "medium",
            "fps": args.fps,
        }
        save_dir = save_dir.parent / (save_dir.name + "_hq")

    # Remove unused modules
    cleanup_unused_modules(model, args.do_teacher_sampling)

    # Get precision and set up inference modules
    teacher, student, vae = setup_inference_modules(
        model, config, args.do_teacher_sampling, args.do_student_sampling, model.precision
    )
    ctx = {"dtype": model.precision, "device": model.device}

    # Check if we have at least one valid sampling path
    has_teacher_sampling = teacher is not None and hasattr(teacher, "sample")
    has_student_sampling = student is not None and hasattr(model, "generator_fn")
    assert (
        has_teacher_sampling or has_student_sampling
    ), "At least one of teacher or student (with generator_fn) must be provided for sampling"

    # Load negative condition
    neg_condition = None
    if args.neg_prompt_file is not None:
        neg_condition = load_prompts(args.neg_prompt_file, relative_to="cwd")
        if len(neg_condition) > 0:
            neg_condition = neg_condition[:1]
            logger.warning(f"Found {len(neg_condition)} negative prompts, only using the first one.")
        logger.debug(f"Loaded negative prompt: {neg_condition[0]}")
        if hasattr(model.net, "text_encoder"):
            with basic_utils.inference_mode(
                model.net.text_encoder, precision_amp=model.precision_amp_enc, device_type=model.device.type
            ):
                neg_condition = basic_utils.to(model.net.text_encoder.encode(neg_condition), **ctx)

    slg_tag = "" if config.model.skip_layers is None else f"_slg{'_'.join([str(x) for x in config.model.skip_layers])}"

    # Fix noise for all generated samples
    noise = torch.randn(
        [1, *config.model.input_shape],
        **ctx,
    )

    for i, prompt in enumerate(pos_prompt_set):
        logger.info(f"[{i+1}/{len(pos_prompt_set)}] Generating: {prompt[:80]}...")

        # Encode prompt
        condition = [prompt]
        if hasattr(model.net, "text_encoder"):
            with basic_utils.inference_mode(
                model.net.text_encoder, precision_amp=model.precision_amp_enc, device_type=model.device.type
            ):
                condition = basic_utils.to(model.net.text_encoder.encode(condition), **ctx)

        # VACE Wan mode: depth-to-video conditioning
        is_net_v2v = hasattr(model.net, "prepare_vid_conditioning")
        if source_video_paths is not None and i < len(source_video_paths) and is_net_v2v:
            depth_latent_path = depth_latent_paths[i] if depth_latent_paths is not None else None
            condition, neg_condition_sample = prepare_vacewan_condition(
                source_video_path=source_video_paths[i],
                depth_latent_path=depth_latent_path,
                model=model,
                latent_shape=config.model.input_shape,
                condition=condition,
                neg_condition=neg_condition,
                ctx=ctx,
                num_segments=args.num_segments,
                overlap_frames=args.overlap_frames,
            )
        else:
            neg_condition_sample = neg_condition

        # Image-to-video / Video2world mode: load and encode conditioning image
        # Skip if already using VACE video conditioning (model has prepare_vid_conditioning)
        i2v_tag = ""
        is_net_i2v = getattr(model.net, "is_i2v", False) or getattr(model.net, "is_video2world", False)
        if input_image_paths is not None and i < len(input_image_paths) and is_net_i2v:
            condition, neg_condition_sample, i2v_tag = prepare_i2v_condition(
                input_image_path=input_image_paths[i],
                model=model,
                vae=vae,
                latent_shape=config.model.input_shape,
                condition=condition,
                neg_condition=neg_condition,
                num_conditioning_frames=args.num_conditioning_frames,
                ctx=ctx,
            )

        # Student sampling
        if has_student_sampling:
            use_extrapolation = args.num_segments != 1 or args.overlap_frames != 0
            start_time = time.time()

            # Build student sampling kwargs
            student_kwargs = {
                "condition": condition,
                "neg_condition": neg_condition_sample,
                "student_sample_steps": model.config.student_sample_steps,
                "student_sample_type": model.config.student_sample_type,
                "t_list": model.config.sample_t_cfg.t_list,
                "precision_amp": model.precision_amp_infer,
            }

            if use_extrapolation:
                if not hasattr(model, "generator_fn_extrapolation"):
                    raise RuntimeError("Extrapolation is only supported for causal autoregressive networks")
                if not hasattr(model.net, "vae"):
                    raise RuntimeError("VAE is required for extrapolation but was not initialized")
                student_kwargs["num_segments"] = args.num_segments
                student_kwargs["overlap_frames"] = args.overlap_frames
                video_student = model.generator_fn_extrapolation(student, noise, **student_kwargs)
            else:
                video_student = model.generator_fn(student, noise, **student_kwargs)

            sampling_time = time.time() - start_time
            logger.info(f"Student sampling time: {sampling_time:.2f}s")
            save_path = save_dir / f"student_step{model.config.student_sample_steps}{i2v_tag}_{i:04d}_seed{seed}.mp4"
            basic_utils.save_media(video_student, str(save_path), vae=vae, **save_video_kwargs)

        # Teacher sampling
        if has_teacher_sampling:
            start_time = time.time()
            teacher_kwargs = {
                "condition": condition,
                "neg_condition": neg_condition_sample,
                "num_steps": args.num_steps,
                "second_order": False,
                "precision_amp": model.precision_amp_infer,
                "fps": torch.full((noise.shape[0],), float(args.fps), device=noise.device),
            }
            if config.model.skip_layers is not None:
                teacher_kwargs["skip_layers"] = config.model.skip_layers

            video_teacher = model.sample(teacher, noise, **teacher_kwargs)
            sampling_time = time.time() - start_time
            logger.info(f"Teacher sampling time: {sampling_time:.2f}s")
            save_path = (
                save_dir
                / f"teacher_cfg{config.model.guidance_scale}_steps{args.num_steps}{slg_tag}{i2v_tag}_{i:04d}_seed{seed}.mp4"
            )
            basic_utils.save_media(video_teacher, str(save_path), vae=vae, **save_video_kwargs)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video model inference")

    # Add common args
    add_common_args(parser)

    # Video-specific args
    parser.add_argument(
        "--save_as_gif",
        default=False,
        type=basic_utils.str2bool,
        help="Whether to save videos as GIF (True) or MP4 (False)",
    )
    parser.add_argument(
        "--fps",
        default=16,
        type=int,
        help="Frames per second for saved video and model temporal encoding (default: 16, matches Wan base_fps)",
    )
    parser.add_argument(
        "--save_high_quality",
        default=False,
        type=basic_utils.str2bool,
        help="Whether to save videos in high-quality (codec: libx265 vs libx264)",
    )
    parser.add_argument(
        "--prompt_file",
        default="scripts/inference/prompts/validation_aug_qwen_2_5_14b_seed42.txt",
        type=str,
        help="File containing prompts (one per line). Relative paths are resolved from script directory.",
    )
    parser.add_argument(
        "--neg_prompt_file",
        default="scripts/inference/prompts/negative_prompt.txt",
        type=str,
        help="The file containing the negative prompt to use for CFG.",
    )
    parser.add_argument(
        "--prompt_expand_model",
        type=str,
        help="If specified, perform prompt expansion using the specified Qwen model.",
        choices=["QwenVL2.5_3B", "QwenVL2.5_7B", "Qwen2.5_3B", "Qwen2.5_7B", "Qwen2.5_14B"],
    )
    parser.add_argument(
        "--prompt_expand_model_seed",
        type=int,
        help="Seed for prompt expansion.",
        default=0,
    )
    parser.add_argument(
        "--depth_latent_file",
        default=None,
        type=str,
        help="The file containing the depth latent paths to use for sampling.",
    )
    parser.add_argument(
        "--source_video_file",
        default="scripts/inference/prompts/source_video_paths.txt",
        type=str,
        help="The file containing the source video paths to use for sampling.",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=1,
        help="Number of autoregressive segments to generate when using extrapolation (default: 1)",
    )
    parser.add_argument(
        "--overlap_frames",
        type=int,
        default=0,
        help="Number of latent frames to overlap between segments when extrapolating (default: 0)",
    )
    parser.add_argument(
        "--video_save_dir",
        type=str,
        help="Path to the video save directory.",
        default=None,
    )
    parser.add_argument(
        "--num_steps",
        default=50,
        type=int,
        help="Number of sampling steps for teacher (default: 50)",
    )
    # I2V arguments
    parser.add_argument(
        "--input_image_file",
        type=str,
        default="scripts/inference/prompts/source_image_paths.txt",
        help="File containing paths to input images (one per line) for I2V mode (or video2world mode in cosmos). "
        "Images are aligned with prompts; if fewer images than prompts, the last image is repeated.",
    )
    parser.add_argument(
        "--num_conditioning_frames",
        type=int,
        default=1,
        help="Number of latent frames to condition on for I2V mode (default: 1).",
    )
    parser.add_argument(
        "--conditional_frame_timestep",
        type=float,
        default=0.0,
        help="Timestep value for conditioning frames in I2V mode. "
        "Use 0.0 (default) to indicate clean conditioning frames. "
        "Use -1.0 to disable timestep modification. "
        "Use small positive value (e.g., 0.1) for noisy conditioning.",
    )

    args = parse_args(parser)
    config = setup(args, evaluation=True)
    main(args, config)

    clean_up()

# ----------------------------------------------------------------------------
