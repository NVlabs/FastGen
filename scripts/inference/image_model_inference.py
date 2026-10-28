# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image generation inference script.

Supports:
- Text-conditional models: SD15, SDXL, Flux, QwenImage
- Class-conditional models: EDM, SiT, DiT (ImageNet)
- Unconditional generation
- Image-editing / image-to-image (I2I): Flux.2 image-edit

Examples:

    # Text-conditional: sample the fine-tuned student (SDXL)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \
        scripts/inference/image_model_inference.py \
        --neg_prompt_file scripts/inference/prompts/empty_prompt.txt \
        --config fastgen/configs/experiments/SDXL/config_sft.py \
        - trainer.seed=1 model.guidance_cfg.cfg.scale=5.0 log_config.name=sdxl_inference

    # Class-conditional: sample the fine-tuned student (SiT on ImageNet)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \
        scripts/inference/image_model_inference.py \
        --prompt_file scripts/inference/prompts/classes.txt --classes 1000 \
        --config fastgen/configs/experiments/DiT/config_sft_sit_xl.py \
        - trainer.seed=1 log_config.name=sit_inference

    # Unconditional generation (EDM CIFAR-10)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \
        scripts/inference/image_model_inference.py \
        --unconditional --num_samples 16 \
        --config fastgen/configs/experiments/EDM/config_sft_edm_cifar10.py \
        - trainer.seed=1 log_config.name=edm_cifar10_inference

    # Eval both student and teacher
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \
        scripts/inference/image_model_inference.py --ckpt_path /path/to/checkpoints/0003000.pth \
        --do_student_sampling True --do_teacher_sampling True \
        --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \
        --config fastgen/configs/experiments/SD15/config_dmd2.py \
        - trainer.seed=1 log_config.name=sd15_inference

    # Image-editing / I2I (Flux.2): prompts are edit instructions, aligned with source images
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \
        scripts/inference/image_model_inference.py \
        --prompt_file scripts/inference/prompts/edit_prompts.txt \
        --input_image_file scripts/inference/prompts/source_image_paths.txt \
        --config fastgen/configs/experiments/Flux2/config_sft_image_edit.py \
        - trainer.seed=1 log_config.name=flux2_image_edit_inference
"""

import argparse
from pathlib import Path

import torch

from fastgen.configs.config import BaseConfig
import fastgen.utils.logging_utils as logger
from fastgen.configs.instantiate import instantiate
from fastgen.utils import basic_utils
from fastgen.utils.media_utils import save_media
from fastgen.utils.basic_utils import cuda_timer
from fastgen.utils.distributed.collectives import clean_up
from fastgen.utils.scripts import parse_args, setup
from fastgen.utils.checkpointer import Checkpointer
from scripts.inference.inference_utils import (
    load_source_image_for_net,
    load_file_entries,
    load_checkpoint,
    default_inference_output_dir,
    setup_inference_modules,
    add_common_args,
)


def main(args, config: BaseConfig):
    # Load prompts or set up unconditional generation
    if args.unconditional:
        pos_prompt_set = [None] * args.num_samples
        prompt_name = "unconditional"
    else:
        pos_prompt_set = load_file_entries(args.prompt_file)
        prompt_name = Path(args.prompt_file).stem

    # One negative prompt per positive one.  Unconditional runs have no negative branch, and
    # class-conditional runs spell out the null class vector below instead of a prompt.
    neg_prompt_set = None
    if args.neg_prompt_file and not args.unconditional and args.classes is None:
        neg_prompt_set = load_file_entries(args.neg_prompt_file, num=len(pos_prompt_set))

    # Optional image-editing (I2I) reference(s), aligned with the prompts (edit
    # instructions). The complete builder resolves text for every network and,
    # for image-edit networks, also bundles the VAE-encoded reference.
    input_image_paths = load_file_entries(args.input_image_file, num=len(pos_prompt_set), drop_empty=True)

    # Fix sampling seeds
    seed = basic_utils.set_random_seed(config.trainer.seed, by_rank=True)

    # Initialize model and checkpointer (config.model.inference was set to
    # True by ``setup(..., evaluation=True)``, skipping training-only modules).
    model = instantiate(config.model_class, config=config.model)
    # Checkpoint contents select sharded loading or delegation to the basic checkpointer.
    checkpointer = Checkpointer(config.trainer.checkpointer)

    # Load checkpoint
    ckpt_iter = load_checkpoint(checkpointer, model, args.ckpt_path)

    if ckpt_iter is None and args.do_student_sampling:
        logger.warning(
            f"Performing {model.net.noise_scheduler.default_student_steps}-step generation on the non-distilled model"
        )

    # Set up save directory. Explicit destinations are caller-owned; the default includes
    # checkpoint identity so evaluating another checkpoint cannot overwrite this run.
    save_root = (
        Path(args.image_save_dir)
        if args.image_save_dir
        else default_inference_output_dir(config.log_config.save_path, args.ckpt_path, ckpt_iter)
    )
    save_dir = save_root / prompt_name
    logger.info(f"image_save_dir: {save_dir}")

    # Prune and set up inference modules (also calls apply_torch_compile internally)
    teacher, student, net = setup_inference_modules(model, args.do_teacher_sampling, args.do_student_sampling)
    vae = net.preprocessors.get("vae")
    ctx = {"dtype": model.precision, "device": model.device}

    # Which sampling paths setup selected. ``setup_inference_modules`` already rejects a request
    # that selects neither, and builds a requested teacher that the method lacks.
    has_teacher_sampling = teacher is not None and hasattr(teacher, "sample")
    has_student_sampling = student is not None

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

        # Assemble raw task inputs. Shared preprocessing encodes the positive prompt and
        # lets the network build any image-edit conditioning.  Branches this run cannot supply
        # stay out of the request entirely: `validate_conditioning`'s `absent_fields` tests key
        # presence, so a present-but-None branch is a different claim from an absent one.
        if args.unconditional:
            data = {}
        elif args.classes is not None:
            if not prompt.isdigit():
                raise ValueError(f"Each prompt must be an integer class label, got: {prompt}")
            condition = torch.zeros(1, args.classes)
            condition[0, int(prompt)] = 1
            # Batched CFG needs a concrete negative: the zero vector is the null class, as in ``ImageDataset``.
            data = {"condition": condition, "neg_condition": torch.zeros_like(condition)}
        else:
            data = {"condition_raw": [prompt]}
            if neg_prompt_set is not None:
                data["neg_condition_raw"] = [neg_prompt_set[i]]
        has_image = input_image_paths is not None
        if has_image:
            data["condition_image_raw"] = load_source_image_for_net(input_image_paths[i], net, config.model.input_shape)
        data = model.prepare_batch(data, inference_mode=True, materialize_real=False)

        # Student sampling
        if has_student_sampling:
            with cuda_timer("Student sampling time"):
                image_student = model.generator_fn(
                    student,
                    noise,
                    condition=data.get("condition"),
                    neg_condition=data.get("neg_condition"),
                )

            save_path = save_dir / f"student_{i:04d}_seed{seed}.png"
            save_media(image_student, str(save_path), vae=vae, precision_amp=model.precision_amp_enc)

        # Teacher sampling
        if has_teacher_sampling:
            teacher_kwargs = {}
            # Only override sample_steps when explicitly set; otherwise let net.sample()
            # use its own default.
            if args.sample_steps is not None:
                teacher_kwargs["sample_steps"] = args.sample_steps

            with cuda_timer("Teacher sampling time"):
                image_teacher = model.sample(
                    teacher,
                    noise,
                    condition=data.get("condition"),
                    neg_condition=data.get("neg_condition"),
                    **teacher_kwargs,
                )

            save_path = save_dir / f"teacher_{i:04d}_seed{seed}.png"
            save_media(image_teacher, str(save_path), vae=vae, precision_amp=model.precision_amp_enc)


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
        help="Negative prompt file for CFG, one prompt per --prompt_file prompt. Omit it for no "
        "negative conditioning, or pass scripts/inference/prompts/empty_prompt.txt for the empty "
        "negative prompt. Omitting it does not disable an active model.guidance_cfg.cfg.scale. "
        "Class-conditional runs ignore it.",
    )
    parser.add_argument(
        "--input_image_file",
        default=None,
        type=str,
        help="File of source/reference image paths (one per line) for image-editing (I2I). "
        "Aligned with --prompt_file (edit instructions). Providing it enables I2I mode; only "
        "image-edit networks consume the reference, others ignore it.",
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
        help="Directory for generated images, with one subdirectory per prompt file. Defaults to a directory in "
        "the run directory (log_config.save_path): inference/ without --ckpt_path, otherwise one named after "
        "the checkpoint.",
    )

    # Sampling arguments
    parser.add_argument(
        "--sample_steps",
        default=None,
        type=int,
        help="Number of teacher denoising steps, i.e. sampling-grid intervals rather than grid points (default: the "
        "network's own default). The student, including an SFT model, samples model.student_grid_cfg.steps steps "
        "instead.",
    )

    args = parse_args(parser)
    config = setup(args, evaluation=True)
    main(args, config)

    clean_up()
