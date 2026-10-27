# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastGen model loader node for ComfyUI."""

from __future__ import annotations

import hashlib
import json

import folder_paths
import torch

from fastgen.configs.config import BaseConfig
from fastgen.configs.instantiate import instantiate
from fastgen.utils.checkpointer import Checkpointer
import fastgen.utils.logging_utils as logger
from scripts.inference.inference_utils import (
    load_checkpoint,
    setup_inference_modules,
)


class FastGenModelLoader:
    """Initialize a FastGen model, load checkpoint, and prepare teacher/student/VAE.

    Outputs an opaque FASTGEN_MODEL bundle consumed by all downstream nodes.
    The model is only reloaded when config, checkpoint, or teacher/student flags change.
    """

    @classmethod
    def INPUT_TYPES(cls):
        ckpt_files = ["(pretrained)"] + folder_paths.get_filename_list("fastgen_checkpoints")
        return {
            "required": {
                "config": ("FASTGEN_CONFIG",),
                "load_teacher": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Load teacher network for multi-step sampling"},
                ),
                "load_student": ("BOOLEAN", {"default": True, "tooltip": "Load student network for few-step sampling"}),
            },
            "optional": {
                "ckpt_name": (
                    ckpt_files,
                    {
                        "tooltip": "Select a .pth checkpoint from the fastgen_checkpoints folder. "
                        "For DCP, place or symlink the complete checkpoint directory there. "
                        "Or type a path in ckpt_path."
                    },
                ),
                "ckpt_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Type an absolute .pth checkpoint path here, with any DCP shard directories beside it "
                        "(overrides dropdown). Leave empty to use dropdown.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FASTGEN_MODEL",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_model"
    CATEGORY = "FastGen"
    DESCRIPTION = "Load a FastGen model with teacher and/or student networks."

    def load_model(
        self,
        config: BaseConfig,
        load_teacher: bool = True,
        load_student: bool = True,
        ckpt_name: str = "(pretrained)",
        ckpt_path: str = "",
    ):
        if not load_teacher and not load_student:
            raise ValueError("At least one of load_teacher or load_student must be enabled.")

        # Resolve checkpoint: explicit path wins, then dropdown, then empty (pretrained)
        effective_ckpt = ""
        if ckpt_path.strip():
            effective_ckpt = ckpt_path.strip()
        elif ckpt_name and ckpt_name != "(pretrained)":
            effective_ckpt = folder_paths.get_full_path("fastgen_checkpoints", ckpt_name)
            if effective_ckpt is None:
                raise FileNotFoundError(
                    f"Checkpoint {ckpt_name!r} was not found in the registered fastgen_checkpoints folders."
                )

        # ComfyUI executes nodes under torch.inference_mode, where FastGen refuses to build, load or place
        # modules: tensors created there reject later in-place updates such as state loading.
        with torch.inference_mode(False):
            model = instantiate(config.model_class, config=config.model)
            checkpointer = Checkpointer(config.trainer.checkpointer)
            ckpt_iter = load_checkpoint(checkpointer, model, effective_ckpt)

            if ckpt_iter is None and load_student:
                logger.warning(
                    f"No checkpoint loaded — running {model.net.noise_scheduler.default_student_steps}-step "
                    "generation on the non-distilled model"
                )

            teacher, student, net = setup_inference_modules(model, load_teacher, load_student)

        has_teacher = teacher is not None and hasattr(teacher, "sample")
        has_student = student is not None and hasattr(model, "generator_fn")
        if not has_teacher and not has_student:
            raise RuntimeError(
                "Neither teacher (with .sample) nor student (with generator_fn) is available after setup."
            )

        vae = net.preprocessors.get("vae")
        ctx = {"dtype": model.precision, "device": model.device}

        bundle = {
            "model": model,
            "net": net,
            "teacher": teacher,
            "student": student,
            "vae": vae,
            "config": config,
            "ctx": ctx,
            "ckpt_iter": ckpt_iter,
            "has_teacher": has_teacher,
            "has_student": has_student,
        }

        logger.info(
            f"[FastGen] Model loaded — teacher={'yes' if has_teacher else 'no'}, "
            f"student={'yes' if has_student else 'no'}, "
            f"vae={'yes' if vae is not None else 'no'}, "
            f"device={model.device}, precision={model.precision}"
        )
        return (bundle,)

    @classmethod
    def IS_CHANGED(cls, config, load_teacher, load_student, ckpt_name="(pretrained)", ckpt_path=""):
        effective = ckpt_path.strip() if ckpt_path.strip() else ckpt_name
        # Hash actual config content, not object identity (id(config) changes on
        # every load and would force needless re-execution). default=str makes
        # non-JSON values deterministic; fall back to repr if json can't encode.
        try:
            config_repr = json.dumps(config, sort_keys=True, default=str)
        except Exception:
            config_repr = repr(config)
        key = f"{config_repr}|{effective}|{load_teacher}|{load_student}"
        return hashlib.md5(key.encode()).hexdigest()
