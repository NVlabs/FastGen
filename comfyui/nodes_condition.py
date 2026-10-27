# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw task-input loaders and unified FastGen preprocessing for ComfyUI."""

from __future__ import annotations

from collections.abc import Mapping

import torch

import fastgen.utils.logging_utils as logger
from fastgen.datasets.video_resampling import constant_rate_frame_indices, validate_frame_rate
from scripts.inference.inference_utils import (
    prepare_source_image_for_net,
    prepare_conditioning_image_for_net,
    prepare_source_video_for_net,
    total_latent_frames,
    load_raw_depth_latent,
    prepare_depth_latent,
    load_camera_poses,
    prepare_camera_poses,
)


def _video_frames(video, target_fps: float | None = None) -> tuple[torch.Tensor, list[int] | None]:
    """Extract decoded frames from a native ComfyUI VIDEO object, resampled to ``target_fps``.

    Returns the frames and, when the source frame rate is known, each kept frame's index in the source
    stream (for camera-pose alignment). A bare tensor is taken to be at ``target_fps`` already.
    """
    if target_fps is not None:
        target_fps = validate_frame_rate(target_fps, "target_fps")
    if isinstance(video, torch.Tensor):
        return video, None
    if not hasattr(video, "get_components"):
        raise TypeError("The video input must be a native ComfyUI VIDEO object")
    components = video.get_components()
    if not hasattr(components, "images"):
        raise TypeError("The ComfyUI VIDEO object did not expose image components")
    frames = components.images
    source_fps = getattr(components, "frame_rate", None)
    if target_fps is None or source_fps is None:
        return frames, None
    # Components omit PTS, so place their frames on a CFR grid and apply the decoder's shared cadence.
    indices = constant_rate_frame_indices(len(frames), source_fps, target_fps)
    return frames[indices], indices


class FastGenCameraLoader:
    """Load raw camera poses and intrinsics without FastGen model encoding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "camera_poses_path": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Folder containing poses.npy and intrinsics.npy.",
                    },
                ),
                "camera_source_video_path": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Source video for VIPE camera-pose extraction.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FASTGEN_CAMERA",)
    RETURN_NAMES = ("camera",)
    FUNCTION = "load"
    CATEGORY = "FastGen/inputs"
    DESCRIPTION = "Load raw camera poses and intrinsics for FastGen preprocessing."

    def load(self, camera_poses_path: str = "", camera_source_video_path: str = ""):
        poses_path = (camera_poses_path or "").strip()
        source_path = (camera_source_video_path or "").strip()
        if bool(poses_path) == bool(source_path):
            raise ValueError("Connect exactly one of camera_poses_path or camera_source_video_path")
        poses, intrinsics = load_camera_poses(poses_path or None, source_path or None)
        return ({"poses": poses, "intrinsics": intrinsics},)


class FastGenDepthLatentLoader:
    """Load a raw precomputed depth latent without model preprocessing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_latent_path": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Path to a precomputed depth latent tensor (.pt).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FASTGEN_DEPTH_LATENT",)
    RETURN_NAMES = ("depth_latent",)
    FUNCTION = "load"
    CATEGORY = "FastGen/inputs"
    DESCRIPTION = "Load a raw depth latent for FastGen preprocessing."

    def load(self, depth_latent_path: str):
        return (load_raw_depth_latent(depth_latent_path),)


class FastGenPreprocess:
    """Encode raw task inputs into one finalized FastGen conditioning object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_bundle": ("FASTGEN_MODEL",),
                "prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Positive prompt. FastGen Prompt Expand can be connected directly.",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Negative prompt for CFG. Leave it disconnected for no negative "
                        "conditioning, which guidance-distilled networks want, or connect an empty "
                        "string for the empty negative prompt. Disconnecting it does not disable an "
                        "active model.guidance_cfg.cfg.scale.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Raw image from Load Image for image editing, I2V, or Video2World.",
                    },
                ),
                "video": (
                    "VIDEO",
                    {
                        "tooltip": "Raw video from ComfyUI's Load Video node for V2V conditioning; "
                        "it is resampled to the inference network's FPS when its frame rate is known.",
                    },
                ),
                "camera": (
                    "FASTGEN_CAMERA",
                    {
                        "forceInput": True,
                        "tooltip": "Raw camera data from FastGen Camera Loader.",
                    },
                ),
                "depth_latent": (
                    "FASTGEN_DEPTH_LATENT",
                    {
                        "forceInput": True,
                        "tooltip": "Raw depth latent from FastGen Depth Latent Loader.",
                    },
                ),
                "num_segments": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "tooltip": "Autoregressive segments; also controls V2V conditioning length.",
                    },
                ),
                "overlap_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "tooltip": "Latent overlap between autoregressive segments.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FASTGEN_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "preprocess"
    CATEGORY = "FastGen/conditioning"
    DESCRIPTION = "Encode text and optional image, video, camera, or depth inputs exactly once."

    def preprocess(
        self,
        model_bundle: dict,
        prompt: str,
        negative_prompt=None,
        image=None,
        video=None,
        camera=None,
        depth_latent=None,
        num_segments: int = 1,
        overlap_frames: int = 0,
    ):
        model = model_bundle["model"]
        net = model_bundle["net"]
        shape = model.input_shape

        if image is not None and video is not None:
            raise ValueError("Connect either image or video conditioning, not both")
        # Widget minima keep num_segments >= 1 and overlap_frames >= 0; generator_fn_extrapolation re-checks them.

        # Both prompts stay raw: ``prepare_batch`` encodes them for ordinary text networks and
        # hands them to ``build_conditioning`` for networks that tokenize themselves (Cosmos3).
        # Like an omitted --neg_prompt_file, a disconnected negative prompt (``None``) builds no
        # negative conditioning, which does not disable an active CFG scale; a connected empty
        # string is the empty negative prompt, like scripts/inference/prompts/empty_prompt.txt.
        data = {"condition_raw": [prompt.strip()]}
        if negative_prompt is not None:
            data["neg_condition_raw"] = [negative_prompt.strip()]
        cropping_params = None
        source_frame_indices = None

        if image is not None:
            video_shape = shape["video"] if isinstance(shape, Mapping) else shape
            if len(video_shape) == 3:
                reference = prepare_source_image_for_net(image, net, shape)
                key = "condition_image_raw"
            elif len(video_shape) == 4:
                prepared = prepare_conditioning_image_for_net(image, net, shape)
                reference = prepared["real_raw"]
                cropping_params = prepared["cropping_params"]
                key = "real_raw"
            else:
                raise ValueError(f"Unsupported FastGen input shape for image conditioning: {video_shape}")
            data[key] = reference

        if video is not None or depth_latent is not None:
            latent_frame_count = total_latent_frames(shape, num_segments, overlap_frames)
            if video is not None:
                frames, frame_indices = _video_frames(video, target_fps=getattr(net, "fps", None))
                prepared = prepare_source_video_for_net(frames, net, shape, latent_frame_count, frame_indices)
                data["real_raw"] = prepared["real_raw"]
                cropping_params = prepared["cropping_params"]
                source_frame_indices = prepared["source_frame_indices"]
            if depth_latent is not None:
                data["depth_latent"] = prepare_depth_latent(depth_latent, latent_frame_count)

        if camera is not None:
            try:
                poses, intrinsics = camera["poses"], camera["intrinsics"]
            except (KeyError, TypeError) as error:
                raise ValueError("camera must come from FastGen Camera Loader") from error
            if cropping_params is None:
                raise ValueError("Camera conditioning requires an input image or video")
            data["camera_poses"], data["camera_intrinsics"] = prepare_camera_poses(
                poses,
                intrinsics,
                cropping_params,
                source_frame_indices,
            )

        processed = model.prepare_batch(data, inference_mode=True, materialize_real=False)
        conditioning = {
            "condition": processed["condition"],
            "neg_condition": processed.get("neg_condition"),
            "num_segments": num_segments,
            "overlap_frames": overlap_frames,
        }
        logger.info(
            "[FastGen] Preprocessed conditioning "
            f"(image={image is not None}, video={video is not None}, "
            f"camera={camera is not None}, depth={depth_latent is not None})"
        )
        return (conditioning,)
