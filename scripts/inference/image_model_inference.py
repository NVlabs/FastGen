# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image generation inference script.

Supports:
- Text-conditional models: SD15, SDXL, Flux
- Class-conditional models: EDM, SiT, DiT (ImageNet)
- Unconditional generation

Examples:

    # Text-conditional: eval teacher only (SDXL)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \
        scripts/inference/image_model_inference.py --do_student_sampling False \
        --config fastgen/configs/experiments/SDXL/config_sft.py \
        - trainer.seed=1 trainer.ddp=True model.guidance_scale=5.0 log_config.name=sdxl_inference

    # Class-conditional: eval teacher (SiT on ImageNet)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \
        scripts/inference/image_model_inference.py --do_student_sampling False \
        --prompt_file scripts/inference/prompts/classes.txt --classes 1000 \
        --config fastgen/configs/experiments/SiT/config_sft.py \
        - trainer.seed=1 trainer.ddp=True log_config.name=sit_inference

    # Unconditional generation (EDM CIFAR-10)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \
        scripts/inference/image_model_inference.py --do_student_sampling False \
        --unconditional --num_samples 16 \
        --config fastgen/configs/experiments/EDM/config_sft_edm_cifar10.py \
        - trainer.seed=1 trainer.ddp=True log_config.name=edm_cifar10_inference

    # Eval both student and teacher
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \
        scripts/inference/image_model_inference.py --ckpt_path /path/to/checkpoints/0003000.pth \
        --do_student_sampling True --do_teacher_sampling True \
        --config fastgen/configs/experiments/SD15/config_dmd2.py \
        - trainer.seed=1 trainer.ddp=True log_config.name=sd15_inference
"""

import argparse
import time
from pathlib import Path

import torch

from fastgen.configs.config import BaseConfig
import fastgen.utils.logging_utils as logger
from fastgen.utils import basic_utils
from fastgen.utils.distributed import clean_up
from fastgen.utils.scripts import parse_args, setup
from fastgen.utils.checkpointer import FSDPCheckpointer
from scripts.inference.inference_utils import (
    load_prompts,
    init_model,
    load_checkpoint,
    cleanup_unused_modules,
    setup_inference_modules,
    add_common_args,
)


def _prepare_condition(args, prompt, model, ctx):
    """Prepare conditioning based on generation mode.

    Args:
        args: Command line arguments
        prompt: Text prompt or class label (None for unconditional)
        model: The model instance
        ctx: Device/dtype context

    Returns:
        Encoded condition tensor or None
    """
    if args.unconditional:
        # Unconditional: use zeros for class-conditional, None for text-conditional
        if args.classes is not None:
            return torch.zeros(1, args.classes, **ctx)
        return None

    if args.classes is not None:
        # Class-conditional: one-hot encode the class label
        assert prompt.isdigit(), f"Each prompt must be an integer class label, got: {prompt}"
        condition = torch.zeros(1, args.classes, **ctx)
        condition[0, int(prompt)] = 1
        return condition

    # Text-conditional: encode the prompt
    condition = [prompt]
    if hasattr(model.net, "text_encoder"):
        with basic_utils.inference_mode(
            model.net.text_encoder, precision_amp=model.precision_amp_enc, device_type=model.device.type
        ):
            condition = basic_utils.to(model.net.text_encoder.encode(condition), **ctx)
    return condition


def main(args, config: BaseConfig):
    # Load prompts or set up unconditional generation
    if args.unconditional:
        pos_prompt_set = [None] * args.num_samples
        prompt_name = "unconditional"
    else:
        pos_prompt_set = load_prompts(args.prompt_file, relative_to="cwd")
        prompt_name = Path(args.prompt_file).stem

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

    # Set up save directory
    if args.image_save_dir:
        save_dir = args.image_save_dir
        logger.info(f"image_save_dir: {save_dir}")
    save_dir = Path(save_dir) / prompt_name

    # Remove unused modules to free memory
    cleanup_unused_modules(model, args.do_teacher_sampling)

    # Set up inference modules
    teacher, student, vae = setup_inference_modules(
        model, config, args.do_teacher_sampling, args.do_student_sampling, model.precision
    )
    ctx = {"dtype": model.precision, "device": model.device}

    # Validate sampling configuration
    has_teacher_sampling = teacher is not None and hasattr(teacher, "sample")
    has_student_sampling = student is not None and hasattr(model, "generator_fn")
    assert (
        has_teacher_sampling or has_student_sampling
    ), "At least one of teacher or student (with generator_fn) must be provided for sampling"

    # Prepare negative condition for CFG
    neg_condition = None
    if args.classes is not None:
        # Class-conditional: use zero vector as negative
        neg_condition = torch.zeros(1, args.classes, **ctx)
    elif args.neg_prompt_file is not None:
        neg_prompts = load_prompts(args.neg_prompt_file, relative_to="cwd")
        if len(neg_prompts) > 1:
            logger.warning(f"Found {len(neg_prompts)} negative prompts, only using the first one.")
        neg_condition = neg_prompts[:1]
        logger.debug(f"Loaded negative prompt: {neg_condition[0]}")
        if hasattr(model.net, "text_encoder"):
            with basic_utils.inference_mode(
                model.net.text_encoder, precision_amp=model.precision_amp_enc, device_type=model.device.type
            ):
                neg_condition = basic_utils.to(model.net.text_encoder.encode(neg_condition), **ctx)

    # Build skip-layer guidance tag for filenames
    slg_tag = ""
    if config.model.skip_layers is not None:
        slg_tag = f"_slg{'_'.join([str(x) for x in config.model.skip_layers])}"

    # Initialize noise (regenerated per sample for unconditional mode)
    noise = torch.randn([1, *config.model.input_shape], **ctx)

    # Main generation loop
    for i, prompt in enumerate(pos_prompt_set):
        # Log progress
        if args.unconditional:
            logger.info(f"[{i+1}/{len(pos_prompt_set)}] Generating unconditional sample...")
            # Generate different noise for each unconditional sample (diversity)
            noise = torch.randn([1, *config.model.input_shape], **ctx)
        else:
            logger.info(f"[{i+1}/{len(pos_prompt_set)}] Generating: {prompt[:80]}...")

        # Prepare condition based on model type
        condition = _prepare_condition(args, prompt, model, ctx)

        # Student sampling
        if has_student_sampling:
            start_time = time.time()
            image_student = model.generator_fn(
                student,
                noise,
                condition=condition,
                student_sample_steps=model.config.student_sample_steps,
                student_sample_type=model.config.student_sample_type,
                t_list=model.config.sample_t_cfg.t_list,
                precision_amp=model.precision_amp_infer,
            )
            logger.info(f"Student sampling time: {time.time() - start_time:.2f}s")

            save_path = save_dir / f"student_step{model.config.student_sample_steps}_{i:04d}_seed{seed}.png"
            basic_utils.save_media(image_student, str(save_path), vae=vae, precision_amp=model.precision_amp_infer)

        # Teacher sampling
        if has_teacher_sampling:
            start_time = time.time()
            teacher_kwargs = {
                "num_steps": args.num_steps,
                "second_order": False,
                "precision_amp": model.precision_amp_infer,
            }
            if config.model.skip_layers is not None:
                teacher_kwargs["skip_layers"] = config.model.skip_layers

            image_teacher = model.sample(
                teacher, noise, condition=condition, neg_condition=neg_condition, **teacher_kwargs
            )
            logger.info(f"Teacher sampling time: {time.time() - start_time:.2f}s")

            save_path = (
                save_dir
                / f"teacher_cfg{config.model.guidance_scale}_steps{args.num_steps}{slg_tag}_{i:04d}_seed{seed}.png"
            )
            basic_utils.save_media(image_teacher, str(save_path), vae=vae, precision_amp=model.precision_amp_infer)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image generation inference for text-conditional, class-conditional, and unconditional models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add common args
    add_common_args(parser)

    # Prompt/condition arguments
    parser.add_argument(
        "--prompt_file",
        default="scripts/inference/prompts/image_prompts.txt",
        type=str,
        help="File containing prompts (one per line). For class-conditional models, use integer class labels.",
    )
    parser.add_argument(
        "--neg_prompt_file",
        default=None,
        type=str,
        help="File containing negative prompt for CFG (only first line used).",
    )
    parser.add_argument(
        "--classes",
        default=None,
        type=int,
        help="Number of classes for class-conditional generation (e.g., 1000 for ImageNet). "
        "Prompts should be integer class labels.",
    )
    parser.add_argument(
        "--unconditional",
        action="store_true",
        help="Generate unconditional samples (no class or text conditioning).",
    )
    parser.add_argument(
        "--num_samples",
        default=10,
        type=int,
        help="Number of samples for unconditional generation (default: 10).",
    )

    # Output arguments
    parser.add_argument(
        "--image_save_dir",
        default=None,
        type=str,
        help="Directory to save generated images (overrides default).",
    )

    # Sampling arguments
    parser.add_argument(
        "--num_steps",
        default=50,
        type=int,
        help="Number of sampling steps for teacher (default: 50).",
    )

    args = parse_args(parser)
    config = setup(args, evaluation=True)
    main(args, config)

    clean_up()
