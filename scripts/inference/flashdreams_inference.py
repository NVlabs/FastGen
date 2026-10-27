# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example CLI for FastGen-distilled Self-Forcing Wan 2.1 T2V checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from zipfile import is_zipfile

from fastgen.configs.config import BaseConfig
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.noise_schedule import get_noise_schedule
from fastgen.utils.scripts import load_config
from scripts.inference.inference_utils import load_file_entries

# diffusers WanTransformer3DModel -> FlashDreams WanDiTNetwork.
_TOP_LEVEL_KEY_MAPPING: dict[str, str] = {
    "patch_embedding.weight": "patch_embedding.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "proj_out.weight": "head.head.weight",
    "proj_out.bias": "head.head.bias",
    "scale_shift_table": "head.modulation",
}
_CONTAINER_KEYS = ("state_dict", "module", "model", "generator_ema", "generator")
_KEY_PREFIXES = ("_fsdp_wrapped_module.", "module.", "model.", "net.", "transformer.")
_NUM_TRAIN_TIMESTEPS = 1000
_WAN_LATENT_CHANNELS = 16
_SINK_PREFIX_SIZE_T = 5
_SINK_ATTENTION_WINDOW_SIZE_T = 7


def _convert_block_key(rest: str) -> str:
    """Convert one diffusers Wan block key suffix to the native Wan layout."""
    if rest == "scale_shift_table":
        return "modulation"
    if rest.startswith("norm2."):
        return "norm3." + rest.removeprefix("norm2.")
    if rest.startswith("ffn.net.0.proj."):
        return "ffn.0." + rest.removeprefix("ffn.net.0.proj.")
    if rest.startswith("ffn.net.2."):
        return "ffn.2." + rest.removeprefix("ffn.net.2.")
    if rest.startswith(("attn1.", "attn2.")):
        branch = "self_attn" if rest.startswith("attn1.") else "cross_attn"
        subkey = rest.split(".", 1)[1]
        for source, target in (
            ("to_out.0.", "o."),
            ("to_q.", "q."),
            ("to_k.", "k."),
            ("to_v.", "v."),
        ):
            if subkey.startswith(source):
                subkey = target + subkey.removeprefix(source)
                break
        return f"{branch}.{subkey}"
    raise KeyError(f"Unsupported FastGen Wan block key: {rest}")


def _strip_key_prefixes(key: str) -> str:
    """Strip nested module wrappers from a checkpoint key."""
    while True:
        for prefix in _KEY_PREFIXES:
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break
        else:
            return key


def fastgen_state_dict_transform(state_dict: dict[str, Any], root: str = "net") -> dict[str, Any]:
    """Convert a FastGen diffusers-layout Wan checkpoint to native Wan keys.

    A monolithic FastGen checkpoint stores each network under its root name; ``root`` selects the one to convert.
    """
    while True:
        for container_key in (root, *_CONTAINER_KEYS):
            nested = state_dict.get(container_key)
            if isinstance(nested, dict):
                state_dict = nested
                break
        else:
            break

    converted: dict[str, Any] = {}
    for key, value in state_dict.items():
        key = _strip_key_prefixes(key)
        if key in _TOP_LEVEL_KEY_MAPPING:
            converted[_TOP_LEVEL_KEY_MAPPING[key]] = value
        elif key.startswith("blocks."):
            _, block_index, rest = key.split(".", 2)
            converted[f"blocks.{block_index}.{_convert_block_key(rest)}"] = value
        else:
            raise KeyError(f"Unsupported FastGen Wan checkpoint key: {key}")
    return converted


@dataclass(frozen=True)
class FastGenInferenceSettings:
    """FastGen config values required by the FlashDreams pipeline."""

    model_size: str
    chunk_size: int
    total_num_frames: int
    height: int
    width: int
    fps: int
    seed: int
    timesteps: list[int]
    checkpoint_root: str


def _load_fastgen_config(config_path: str, opts: list[str]) -> BaseConfig:
    """Load a FastGen config, apply CLI overrides, and verify the method."""
    config = load_config(config_path, opts, evaluation=True)

    method = config.model_class.get("_target_")
    if getattr(method, "__name__", None) != "SelfForcingModel":
        raise ValueError(
            f"FlashDreams inference requires a Self-Forcing config; got {getattr(method, '__name__', method)!r}."
        )
    return config


def _settings_from_config(config: BaseConfig) -> FastGenInferenceSettings:
    """Extract and validate the Wan 2.1 T2V inference settings."""
    network = config.model.net
    network_target = network.get("_target_")
    if getattr(network_target, "__name__", None) != "CausalWan":
        raise ValueError(
            f"Expected a causal Wan T2V network; got {getattr(network_target, '__name__', network_target)!r}."
        )

    model_id = str(network.model_id_or_local_path).lower()
    if "wan2.1-t2v" not in model_id:
        raise ValueError(f"Expected a Wan 2.1 T2V model, got {network.model_id_or_local_path!r}.")
    if "1.3b" in model_id:
        model_size = "1.3b"
    elif "14b" in model_id:
        model_size = "14b"
    else:
        raise ValueError(f"Could not infer model size from {network.model_id_or_local_path!r}.")

    input_shape = config.model.input_shape
    if len(input_shape) != 4 or int(input_shape[0]) != _WAN_LATENT_CHANNELS:
        raise ValueError(f"Expected Wan latent shape [{_WAN_LATENT_CHANNELS}, T, H, W], got {input_shape!r}.")
    _, height, width = model_pixel_shape(config.model)

    chunk_size = int(network.chunk_size)
    total_num_frames = int(network.total_num_frames)
    if chunk_size <= 0 or total_num_frames <= 0:
        raise ValueError("chunk_size and total_num_frames must be positive.")
    if int(input_shape[1]) != total_num_frames:
        raise ValueError(
            f"model.input_shape[1] ({input_shape[1]}) must match model.net.total_num_frames ({total_num_frames})."
        )

    # Resolve the canonical scheduler grid instead of interpreting its config here.
    schedule = get_noise_schedule(
        network.get("schedule_type", "rf"),
        student_grid_cfg=config.model.student_grid_cfg,
    )
    student_grid = schedule.student_grid()
    timesteps = [round(float(timestep) * _NUM_TRAIN_TIMESTEPS) for timestep in student_grid[:-1]]

    fps = network.get("fps", 16)
    fps = 16 if fps is None else int(fps)
    if fps <= 0:
        raise ValueError("model.net.fps must be positive.")

    # Like the other inference scripts, sample the first configured EMA network, else the live one.
    use_ema = config.model.use_ema
    if isinstance(use_ema, bool):
        use_ema = ["ema"] if use_ema else []

    return FastGenInferenceSettings(
        model_size=model_size,
        chunk_size=chunk_size,
        total_num_frames=total_num_frames,
        height=height,
        width=width,
        fps=fps,
        seed=int(config.trainer.seed),
        timesteps=timesteps,
        checkpoint_root=use_ema[0] if use_ema else "net",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a FastGen-distilled Self-Forcing Wan 2.1 T2V checkpoint with FlashDreams."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Consolidated FastGen .pth checkpoint. A DCP conversion holds one network, so convert the one "
        "inference uses (scripts/fsdp_convert.py --model-type ema_model when the config uses EMA).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="FastGen Python config used to train the checkpoint.",
    )
    parser.add_argument(
        "--prompt-file",
        default="scripts/inference/prompts/video_prompts.txt",
        type=str,
        help="File containing prompts (one per line); only the first one is used.",
    )
    parser.add_argument(
        "--total-blocks",
        type=int,
        help="Autoregressive chunks to generate (default: total_num_frames / chunk_size from the config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the generated video (default: <log_config.save_path>/flashdreams).",
    )
    parser.add_argument(
        "--sink",
        action="store_true",
        help="Use FlashDreams' 1.3B sink preset for stable long rollouts.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="FastGen config overrides, separated from script arguments by '-'.",
    )

    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not is_zipfile(args.checkpoint):
        parser.error(
            f"checkpoint is not a complete PyTorch archive: {args.checkpoint}. "
            "If it is being copied, wait for the transfer to finish."
        )
    if args.total_blocks is not None and args.total_blocks <= 0:
        parser.error("--total-blocks must be positive")
    return args


def main() -> None:
    """Build the FlashDreams runner config and generate a video."""
    args = _parse_args()
    fastgen_config = _load_fastgen_config(args.config, args.opts)
    settings = _settings_from_config(fastgen_config)
    output_dir = args.output_dir or Path(fastgen_config.log_config.save_path) / "flashdreams"

    prompt = load_file_entries(args.prompt_file, num=1)[0]
    if not prompt:
        raise ValueError(f"No prompt found in {args.prompt_file}.")

    if args.sink and (settings.model_size != "1.3b" or settings.chunk_size != 3):
        raise ValueError("The FlashDreams sink preset requires a 1.3B model with chunk_size=3.")

    try:
        from flashdreams.infra.diffusion.model import DiffusionModelConfig
        from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
        from flashdreams.recipes.wan import (
            Wan21TransformerConfig,
            WanDiTNetwork1pt3BConfig,
            WanDiTNetwork14BConfig,
            WanInferencePipelineConfig,
            WanVAEDecoderConfig,
        )
        from self_forcing.runner import SelfForcingT2VRunnerConfig
    except ImportError as exc:
        raise SystemExit(
            "This script targets the integrations/self_forcing package of FlashDreams v0.1.0. From a v0.1.0 "
            "checkout (git clone --branch v0.1.0 https://github.com/NVIDIA/flashdreams.git), run: "
            "uv sync --project integrations/self_forcing"
        ) from exc

    network_config_class = WanDiTNetwork1pt3BConfig if settings.model_size == "1.3b" else WanDiTNetwork14BConfig
    network_config = network_config_class(
        patch_embedding_type="conv3d",
        cp_method="ring",
        apply_rope_before_kvcache=not args.sink,
    )
    transformer_config = Wan21TransformerConfig(
        network=network_config,
        checkpoint_path=str(args.checkpoint),
        state_dict_transform=partial(fastgen_state_dict_transform, root=settings.checkpoint_root),
        batch_shape=(),
        len_t=settings.chunk_size,
        guidance_scale=1.0,
        # Sink settings bound KV-cache attention context; total_blocks controls rollout length.
        window_size_t=_SINK_ATTENTION_WINDOW_SIZE_T if args.sink else settings.total_num_frames,
        sink_size_t=_SINK_PREFIX_SIZE_T if args.sink else 0,
        stamp_image_latent=False,
        compile_network=not args.sink,
        use_cuda_graph=not args.sink,
    )
    scheduler_config = FlowMatchSchedulerConfig(
        num_inference_steps=len(settings.timesteps),
        denoising_timesteps=settings.timesteps,
        # shift=1 preserves FastGen's normalized student_grid while resolving schedule entries.
        warp_denoising_step=True,
        shift=1.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
    )

    name = f"fastgen-self-forcing-wan2.1-t2v-{settings.model_size}"
    if args.sink:
        name += "-sink"

    pipeline_config = WanInferencePipelineConfig(
        name=name,
        enable_sync_and_profile=False,
        encoder=None,
        decoder=WanVAEDecoderConfig(),
        diffusion_model=DiffusionModelConfig(
            seed=settings.seed,
            transformer=transformer_config,
            scheduler=scheduler_config,
        ),
    )

    if args.total_blocks is None:
        total_blocks, remainder = divmod(settings.total_num_frames, settings.chunk_size)
        if remainder:
            raise ValueError(
                "model.net.total_num_frames must be divisible by model.net.chunk_size when --total-blocks is omitted."
            )
    else:
        total_blocks = args.total_blocks

    runner_config = SelfForcingT2VRunnerConfig(
        runner_name=name,
        description=f"FastGen-distilled Self-Forcing Wan 2.1 T2V {settings.model_size} inference.",
        pipeline=pipeline_config,
        prompt=prompt,
        total_blocks=total_blocks,
        output_dir=output_dir,
        pixel_height=settings.height,
        pixel_width=settings.width,
        fps=settings.fps,
    )
    runner_config.setup().run()


if __name__ == "__main__":
    main()
