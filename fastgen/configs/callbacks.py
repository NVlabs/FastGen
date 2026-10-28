# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.instantiate import LazyCall as L

from fastgen.callbacks.ct_schedule import CTScheduleCallback
from fastgen.callbacks.grad_clip import GradClipCallback
from fastgen.callbacks.param_count import ParamCountCallback
from fastgen.callbacks.wandb import WandbCallback
from fastgen.callbacks.vlm_logger import VLMLogger
from fastgen.callbacks.vipe_logger import VipeLogger
from fastgen.callbacks.ema import EMACallback
from fastgen.callbacks.train_profiler import TrainProfilerCallback
from fastgen.callbacks.gpu_stats import GPUStatsCallback
from fastgen.callbacks.forced_weight_norm import ForcedWeightNormCallback
from fastgen.callbacks.gpu_mem_profiler import MemTrackerCallback
from fastgen.callbacks.rank_consistency import RankConsistencyCallback


CTSchedule_CALLBACK = dict(
    ct_schedule=L(CTScheduleCallback)(q=2.0, ratio_limit=0.999, kimg_per_stage=12500),
)

EMA_CALLBACK = dict(
    ema=L(EMACallback)(
        type="constant", beta=0.9999, gamma=16.97, ema_halflife_kimg=500, ema_rampup_ratio=0.05, start_iter=0
    ),
)

EMA_CONST_CALLBACKS = dict(
    ema_9999=L(EMACallback)(type="constant", beta=0.9999, ema_name="ema_9999"),
    ema_99995=L(EMACallback)(type="constant", beta=0.99995, ema_name="ema_99995"),
    ema_9996=L(EMACallback)(type="constant", beta=0.9996, ema_name="ema_9996"),
)

EMA_POWER_CALLBACKS = dict(
    ema_1=L(EMACallback)(type="power", gamma=96.99, ema_name="ema_1"),
    ema_5=L(EMACallback)(type="power", gamma=16.97, ema_name="ema_5"),
    ema_10=L(EMACallback)(type="power", gamma=6.94, ema_name="ema_10"),
)

ForcedWeightNorm_CALLBACK = dict(
    forced_weight_norm=L(ForcedWeightNormCallback)(),
)

GradClip_CALLBACK = dict(
    grad_clip=L(GradClipCallback)(grad_norm=10.0, model_key="net"),
)

GPUStats_CALLBACK = dict(
    gpu_stats=L(GPUStatsCallback)(),
)

ParamCount_CALLBACK = dict(
    param_count=L(ParamCountCallback)(),
)

TrainProfiler_CALLBACK = dict(
    train_profiler=L(TrainProfilerCallback)(),
)

WANDB_CALLBACK = dict(
    wandb=L(WandbCallback)(),
)

MemTracker_CALLBACK = dict(
    mem_tracker=L(MemTrackerCallback)(),
)

# Debug aid, deliberately not in BASE_CALLBACKS: every ``every_n`` iterations, assert that the
# replicated buffers (and with ``check_parameters=True`` the non-sharded parameters) of every
# registered root are identical across ranks.
RankConsistency_CALLBACK = dict(
    rank_consistency=L(RankConsistencyCallback)(every_n=100),
)

# Typical callbacks every training run gets.
BASE_CALLBACKS = {
    **WANDB_CALLBACK,
    **GPUStats_CALLBACK,
    **TrainProfiler_CALLBACK,
    **ParamCount_CALLBACK,
}


# ---------------------------------------------------------------------------
# Drop-in dicts wiring one log module (``VLMLogger`` or ``VipeLogger``) onto a
# ``WandbCallback`` that logs media from the first four ranks.
#
# Each preset returns ``{"wandb": L(WandbCallback)(log_modules={...})}``;
# experiment configs override the default ``wandb`` entry in place, keeping
# its position in the callback order, via
# ``callbacks.update(VLM_VID_CALLBACK)``, then tweak knobs by name —
# ``wandb.log_modules["vlm"].max_samples = 8`` — instead of fragile
# positional ``log_modules[0]`` indexing.

# T2I default — flash-lite judge, single-image scoring. ``master_prompt``
# is resolved per call by ``score_sample`` via
# ``DEFAULT_MASTER_PROMPTS[sample.media_type]``. ``model`` must be the id
# that the OpenAI-compatible endpoint at ``base_url`` expects.
VLM_IMG_CALLBACK = dict(
    wandb=L(WandbCallback)(
        max_ranks=4,
        log_modules={
            "vlm": L(VLMLogger)(
                media_mode="image",
                model="gemini-3.1-flash-lite",
                base_url=None,  # falls back to the VLM_BASE_URL env var (set e.g. in .env)
            ),
        },
    ),
)

# I2V / T2V default — frame-grid scoring against the tiled-grid rubric
# (resolved per call by ``score_sample`` from ``media_mode``) with the
# video-native gemini-3.1-pro-preview judge. ``model`` must be the id that
# the OpenAI-compatible endpoint at ``base_url`` expects.
VLM_VID_CALLBACK = dict(
    wandb=L(WandbCallback)(
        max_ranks=4,
        log_modules={
            "vlm": L(VLMLogger)(
                media_mode="frame_grid",
                video_frames=4,
                model="gemini-3.1-pro-preview",
                base_url=None,  # falls back to the VLM_BASE_URL env var (set e.g. in .env)
            ),
        },
    ),
)

# Camera-conditioned video models (e.g. LingBot-world): log the GT camera
# trajectory each step. Set ``run_vipe_on_generated=True`` (requires the
# optional ``vipe`` package) to also estimate the generated clip's trajectory
# and report scale-aligned pose / intrinsics error. Override by name, e.g.
# ``callbacks.wandb.log_modules["vipe"].run_vipe_on_generated = True``.
VIPE_CALLBACK = dict(
    wandb=L(WandbCallback)(
        max_ranks=4,
        log_modules={
            "vipe": L(VipeLogger)(
                run_vipe_on_generated=False,
            ),
        },
    ),
)
