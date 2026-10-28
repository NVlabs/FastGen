# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastGen prompt expansion for ComfyUI."""

from __future__ import annotations

import torch
from PIL import Image

import fastgen.utils.logging_utils as logger
from fastgen.utils.tensor import unit_pixels_to_uint8
from scripts.inference.inference_utils import expand_prompts_with_qwen

PROMPT_EXPAND_CHOICES = ["None", "QwenVL2.5_3B", "QwenVL2.5_7B", "Qwen2.5_3B", "Qwen2.5_7B", "Qwen2.5_14B"]


def _image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert the first frame of a ComfyUI IMAGE tensor (``[B, H, W, C]`` in ``[0, 1]``) to a PIL image."""
    return Image.fromarray(unit_pixels_to_uint8(image[0]).cpu().numpy())


class FastGenPromptExpand:
    """Expand a prompt with a Qwen model before preprocessing.

    Wire the output ``prompt`` into FastGenPreprocess (or any STRING input).
    The vision-language choices (``QwenVL2.5_*``) require a reference ``image``;
    the text-only choices (``Qwen2.5_*``) ignore it.  ``None`` is a passthrough.

    ``system_prompt`` overrides the built-in master prompt when non-empty;
    otherwise the English master prompt is used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prompt to expand.",
                    },
                ),
                "prompt_expand_model": (PROMPT_EXPAND_CHOICES, {"default": "None"}),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Reference image for vision-language (QwenVL) expansion. Ignored by text-only models."},
                ),
                "prompt_expand_seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "system_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Override the built-in master prompt. Leave disconnected to use the default.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "expand"
    CATEGORY = "FastGen"
    DESCRIPTION = "Expand a prompt with a Qwen / QwenVL model before FastGen preprocessing."

    def expand(
        self,
        prompt: str,
        prompt_expand_model: str = "None",
        image=None,
        prompt_expand_seed: int = 0,
        system_prompt: str = "",
    ):
        if prompt_expand_model == "None":
            return (prompt,)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        system_prompt_override = system_prompt.strip() or None
        expanded = expand_prompts_with_qwen(
            [prompt],
            prompt_expand_model,
            device,
            prompt_expand_seed,
            images=[_image_tensor_to_pil(image)] if image is not None else None,
            system_prompt=system_prompt_override,
        )[0]
        logger.info(f"[FastGen] Expanded prompt: {expanded[:120]}...")
        return (expanded,)
