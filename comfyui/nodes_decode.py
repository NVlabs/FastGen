# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastGen unified decode + preview + save node for ComfyUI."""

from __future__ import annotations

import os

import folder_paths
import torch

from fastgen.utils import basic_utils
from fastgen.utils.media_utils import DEFAULT_AUDIO_SAMPLE_RATE, save_media
from fastgen.utils.tree_utils import get_device
import fastgen.utils.logging_utils as logger


def _comfy_image(pixels: torch.Tensor | None, frame_index: int) -> torch.Tensor | None:
    """Convert ``[B, C, T, H, W]`` video or ``[B, C, H, W]`` image in ``[-1, 1]`` to a ComfyUI IMAGE."""
    if pixels is None:
        return None
    frames = pixels[0]  # drop batch dim -> [C, T, H, W] or [C, H, W]

    if frames.ndim == 3:
        image = frames.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, C]
    elif frames.ndim == 4:
        if frame_index == -2:
            image = frames.permute(1, 2, 3, 0)  # all frames [T, H, W, C]
        else:
            idx = frame_index if frame_index >= 0 else frames.shape[1] + frame_index + 1
            idx = min(idx, frames.shape[1] - 1)
            image = frames[:, idx].permute(1, 2, 0).unsqueeze(0)  # [1, H, W, C]
    else:
        raise ValueError(f"Unexpected decoded tensor shape: {frames.shape}")

    return ((image.float() + 1.0) * 0.5).clamp(0, 1).cpu()


def _comfy_audio(waveform: torch.Tensor | None, sample_rate: int) -> dict | None:
    """Convert a ``[B, samples]`` or ``[B, C, samples]`` waveform to a ComfyUI AUDIO."""
    if waveform is None:
        return None
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    return {"waveform": waveform.clamp(-1, 1).float().cpu(), "sample_rate": sample_rate}


class FastGenDecodeVideo:
    """Decode FASTGEN_VIDEO latents through the VAE, preview inline, save to
    ComfyUI's output directory, and return frames as a ComfyUI IMAGE and audio
    as a ComfyUI AUDIO.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_bundle": ("FASTGEN_MODEL",),
                "video_latent": ("FASTGEN_VIDEO",),
            },
            "optional": {
                "frame_index": (
                    "INT",
                    {
                        "default": -2,
                        "min": -2,
                        "tooltip": (
                            "Frame to extract as IMAGE output. " "0 = first, -1 = last, -2 = ALL frames as a batch."
                        ),
                    },
                ),
                "vid_format": (
                    ["mp4", "gif"],
                    {
                        "default": "mp4",
                        "tooltip": (
                            "Container format for the preview / saved video. Audio-video outputs always use mp4, "
                            "and audio-only outputs wav."
                        ),
                    },
                ),
                "save_high_quality": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Use high-quality encoding (libx264, crf 18)."},
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "fastgen/comfyui",
                        "tooltip": "Filename prefix for the saved output (may include subdirs like 'fastgen/run1').",
                    },
                ),
                # ComfyUI's audio player. The frontend finds it by this name, fills it from ``audio`` UI
                # results and never sends its value, so it must stay optional.
                "audioUI": ("AUDIO_UI", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO")
    RETURN_NAMES = ("image", "saved_path", "audio")
    FUNCTION = "decode"
    OUTPUT_NODE = True
    CATEGORY = "FastGen"
    DESCRIPTION = (
        "Decode output latents, save to ComfyUI's output directory, and return frames as a ComfyUI IMAGE and "
        "audio as a ComfyUI AUDIO. Audio-only models return no IMAGE; models without audio return no AUDIO."
    )

    def decode(
        self,
        model_bundle: dict,
        video_latent: dict,
        frame_index: int = -2,
        vid_format: str = "mp4",
        save_high_quality: bool = False,
        filename_prefix: str = "fastgen/comfyui",
    ):
        model = model_bundle["model"]
        vae = model_bundle["vae"]

        latent = video_latent["latent"]
        # Playback follows the network that hosts preprocessing. Packaged teacher/student configs
        # are tested to match it; custom configurations are responsible for the same invariant.
        fps = float(getattr(model_bundle["net"], "fps", 16))

        # ---- single VAE decode (the expensive step — done once, reused below) ----
        if vae is not None:
            with basic_utils.inference_mode(
                vae,
                precision_amp=model.precision_amp_enc,
                device_type=get_device(latent).type,
            ):
                decoded = vae.decode(latent)
        else:
            decoded = latent

        assert not isinstance(decoded, dict) or set(decoded) in (
            {"video"},
            {"image"},
            {"audio"},
            {"video", "audio"},
        ), f"Unexpected decoded output keys: {set(decoded)}"

        # Split the output into frames and a waveform. Bare tensors follow ``save_media``'s
        # rank rule: [B, samples] and [B, C, samples] are audio.
        if isinstance(decoded, dict):
            pixels = decoded.get("video", decoded.get("image"))
            waveform = decoded.get("audio")
        elif decoded.ndim in (2, 3):
            pixels, waveform = None, decoded
        else:
            pixels, waveform = decoded, None

        # pixels: [B, C, T, H, W] (video) or [B, C, H, W] (image), in [-1, 1]
        # Modality follows tensor rank: a 5D tensor remains video even when T == 1.
        is_video = pixels is not None and pixels.ndim == 5

        # Match the container ``save_media`` writes: MP4 for audio-video (GIF cannot carry audio), WAV for audio.
        if waveform is not None:
            ext = "mp4" if pixels is not None else "wav"
        else:
            ext = vid_format if is_video else "png"
        audio_sample_rate = getattr(vae, "vocoder_sample_rate", DEFAULT_AUDIO_SAMPLE_RATE)
        save_kwargs: dict = {"fps": fps, "vid_format": vid_format, "audio_sample_rate": audio_sample_rate}
        if save_high_quality:
            save_kwargs.update(quality=18, preset="medium")

        full_output_folder, base_filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        # ComfyUI's output counter parses the digits before a trailing "_", its saver convention.
        out_filename = f"{base_filename}_{counter:05}_.{ext}"
        media_path = os.path.join(full_output_folder, out_filename)
        save_media(decoded, media_path, vae=None, **save_kwargs)
        saved_path = media_path
        preview = {"filename": out_filename, "subfolder": subfolder, "type": "output"}
        logger.info(f"[FastGen] Output saved to {media_path}")

        ui: dict = {"images": [preview]} if pixels is not None else {"audio": [preview]}
        if is_video:
            ui["animated"] = (True,)

        image = _comfy_image(pixels, frame_index)
        audio = _comfy_audio(waveform, audio_sample_rate)

        logger.info(
            f"[FastGen] Output decoded -> IMAGE {None if image is None else image.shape}, "
            f"AUDIO {None if audio is None else audio['waveform'].shape}, media {media_path}"
        )
        return {"ui": ui, "result": (image, saved_path, audio)}
