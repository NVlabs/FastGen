# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastGen sampling nodes for ComfyUI — noise, teacher, and student."""

from __future__ import annotations

from fastgen.utils import basic_utils
from fastgen.utils.tree_utils import noise_from_shape
from fastgen.utils.basic_utils import cuda_timer
import fastgen.utils.logging_utils as logger

from ._config_helpers import temporary_overrides

_CONFIG_OVERRIDES_TOOLTIP = (
    "Temporary config overrides, one per line: dotted.key=value, with values parsed as Python literals.\n"
    "Switching to an explicit student grid replaces the whole grid_kwargs dict (quote it).\n"
    "e.g.:\n"
    "  model.guidance_cfg.cfg.scale=5.0\n"
    "  model.student_grid_cfg.grid_type=explicit\n"
    "  model.student_grid_cfg.grid_kwargs='{\"timesteps\": [0.999, 0.937, 0.833, 0.624, 0.0]}'"
)


class FastGenNoiseGenerator:
    """Generate fixed noise from the model's latent shape and a seed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_bundle": ("FASTGEN_MODEL",),
                "seed": (
                    "INT",
                    {"default": 1, "min": 0, "max": 2**31 - 1, "tooltip": "Random seed for noise generation"},
                ),
            },
        }

    RETURN_TYPES = ("FASTGEN_NOISE",)
    RETURN_NAMES = ("noise",)
    FUNCTION = "generate"
    CATEGORY = "FastGen"
    DESCRIPTION = "Generate deterministic noise for FastGen sampling."

    def generate(self, model_bundle: dict, seed: int = 1):
        model = model_bundle["model"]
        ctx = model_bundle["ctx"]

        basic_utils.set_random_seed(seed)
        noise = noise_from_shape(model.input_shape, batch_size=1, **ctx)
        logger.info(
            f"[FastGen] Noise generated with seed={seed}, shape={noise.shape if hasattr(noise, 'shape') else 'tree'}"
        )
        return (noise,)


class FastGenTeacherSample:
    """Run multi-step teacher sampling (e.g. 50-step DDPM/Euler)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_bundle": ("FASTGEN_MODEL",),
                "conditioning": ("FASTGEN_CONDITIONING",),
                "noise": ("FASTGEN_NOISE",),
            },
            "optional": {
                "sample_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "Teacher denoising steps (sampling-grid intervals, not grid points). "
                        "0 = use the model/scheduler default (matches the CLI).",
                    },
                ),
                "config_overrides": (
                    "STRING",
                    {"forceInput": True, "tooltip": _CONFIG_OVERRIDES_TOOLTIP},
                ),
            },
        }

    RETURN_TYPES = ("FASTGEN_VIDEO",)
    RETURN_NAMES = ("video_latent",)
    FUNCTION = "sample"
    CATEGORY = "FastGen/sampling"
    DESCRIPTION = "Run multi-step teacher sampling to generate a video."

    def sample(
        self,
        model_bundle: dict,
        conditioning: dict,
        noise,
        sample_steps: int = 0,
        config_overrides: str = "",
    ):
        model = model_bundle["model"]
        teacher = model_bundle["teacher"]
        config = model_bundle["config"]

        if teacher is None or not model_bundle["has_teacher"]:
            raise RuntimeError("Teacher network was not loaded. Enable load_teacher in FastGenModelLoader.")

        teacher_kwargs = {
            "condition": conditioning["condition"],
            "neg_condition": conditioning["neg_condition"],
        }
        # Match the CLI: only override the network's default when the user set a
        # positive value. Passing 0/None would crash schedulers like UniPC
        # (set_timesteps(None + 1)); leaving it unset uses the model default.
        if sample_steps:
            teacher_kwargs["sample_steps"] = sample_steps

        with temporary_overrides(config, config_overrides):
            with cuda_timer("Teacher sampling time"):
                video = model.sample(teacher, noise, **teacher_kwargs)

        steps_tag = sample_steps if sample_steps else "default"
        logger.info(f"[FastGen] Teacher sampling done — steps={steps_tag}")
        return ({"latent": video},)


class FastGenStudentSample:
    """Run few-step student sampling (distilled model)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_bundle": ("FASTGEN_MODEL",),
                "conditioning": ("FASTGEN_CONDITIONING",),
                "noise": ("FASTGEN_NOISE",),
            },
            "optional": {
                "config_overrides": (
                    "STRING",
                    {"forceInput": True, "tooltip": _CONFIG_OVERRIDES_TOOLTIP},
                ),
            },
        }

    RETURN_TYPES = ("FASTGEN_VIDEO",)
    RETURN_NAMES = ("video_latent",)
    FUNCTION = "sample"
    CATEGORY = "FastGen/sampling"
    DESCRIPTION = "Run few-step student sampling; extrapolation settings come from FastGen Preprocess."

    def sample(
        self,
        model_bundle: dict,
        conditioning: dict,
        noise,
        config_overrides: str = "",
    ):
        model = model_bundle["model"]
        net = model_bundle["net"]
        student = model_bundle["student"]
        config = model_bundle["config"]

        if student is None or not model_bundle["has_student"]:
            raise RuntimeError("Student network was not loaded. Enable load_student in FastGenModelLoader.")

        num_segments = conditioning.get("num_segments", 1)
        overlap_frames = conditioning.get("overlap_frames", 0)
        use_extrapolation = num_segments != 1 or overlap_frames != 0

        with temporary_overrides(config, config_overrides):
            effective_steps = student.noise_scheduler.default_student_steps
            with cuda_timer("Student sampling time"):
                if use_extrapolation:
                    if not hasattr(model, "generator_fn_extrapolation"):
                        raise RuntimeError(
                            "Extrapolation (num_segments/overlap_frames) requires a method with "
                            "generator_fn_extrapolation, e.g. a CausVid or Self-Forcing config; "
                            f"{type(model).__name__} has none."
                        )
                    if overlap_frames > 0 and "vae" not in net.preprocessors:
                        raise RuntimeError("VAE is required for overlapping extrapolation but was not initialized.")
                    video = model.generator_fn_extrapolation(
                        student,
                        noise,
                        num_segments=num_segments,
                        overlap_frames=overlap_frames,
                        condition=conditioning["condition"],
                    )
                else:
                    video = model.generator_fn(
                        student,
                        noise,
                        condition=conditioning["condition"],
                        neg_condition=conditioning["neg_condition"],
                    )

        logger.info(f"[FastGen] Student sampling done — steps={effective_steps}, segments={num_segments}")
        return ({"latent": video},)
