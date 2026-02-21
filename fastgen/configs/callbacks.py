# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.utils import LazyCall as L

from fastgen.callbacks.ct_schedule import CTScheduleCallback
from fastgen.callbacks.grad_clip import GradClipCallback
from fastgen.callbacks.param_count import ParamCountCallback
from fastgen.callbacks.wandb import WandbCallback
from fastgen.callbacks.ema import EMACallback
from fastgen.callbacks.train_profiler import TrainProfilerCallback
from fastgen.callbacks.gpu_stats import GPUStatsCallback
from fastgen.callbacks.forced_weight_norm import ForcedWeightNormCallback
from fastgen.callbacks.gpu_mem_profiler import MemTrackerCallback


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
    gpu_stats=L(GPUStatsCallback)(every_n=100),
)

ParamCount_CALLBACK = dict(
    param_count=L(ParamCountCallback)(),
)

TrainProfiler_CALLBACK = dict(
    train_profiler=L(TrainProfilerCallback)(every_n=100),
)

WANDB_CALLBACK = dict(
    wandb=L(WandbCallback)(sample_logging_iter=None),
)

MemTracker_CALLBACK = dict(
    mem_tracker=L(MemTrackerCallback)(),
)
