# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config schema for the AnyFlow on-policy method (paper Stage 2).

AnyFlow's on-policy stage is DMD2 with a flow-map student that generates via
a multi-step rollout-with-gradient, so the config is based on the DMD2 model
config. The flow-map pretrain stage (paper Stage 1) is MeanFlow with
AnyFlow's hyperparameters — see ``configs/experiments/WanT2V/config_anyflow.py``.
"""

from typing import Any, Dict, List, Optional

import attrs
from omegaconf import DictConfig

from fastgen.callbacks.grad_clip import GradClipCallback
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    EMA_CALLBACK,
    GradClip_CALLBACK,
)
from fastgen.configs.config import BaseConfig
from fastgen.configs.methods.config_dmd2 import ModelConfig as DMD2ModelConfig
from fastgen.configs.methods.config_mean_flow import (
    LossConfig as MeanFlowLossConfig,
    SampleTConfig as MeanFlowSampleTConfig,
)
from fastgen.methods import AnyFlowModel
from fastgen.configs.instantiate import LazyCall as L


@attrs.define
class ModelConfig(DMD2ModelConfig):
    """AnyFlow on-policy model config — DMD2 plus the rollout / cotrain knobs.

    The MeanFlow loss / sampling configs drive the co-trained Stage-1
    flow-map loss inside the student update (the reference's
    ``cotrain_forward_kl``).
    """

    # AnyFlow starts from fresh noise and completes its flow-map rollout in-iteration, so it
    # requires the carry-free "default" mode. Real data is required only when the co-trained
    # flow-map loss is enabled; cotrain_pretrain_weight=0 permits data-free training (the
    # inherited GAN terms likewise require real data to be active).
    backward_simulation: str | None = "default"

    # The reference rollout uses flow-map ODE steps.
    student_sample_type: str = "ode"

    # MeanFlow-style (t, r) sampling for the co-trained flow-map loss. Separate from
    # the inherited `sample_t_cfg`, which DMD2 uses for the noising time: the reference
    # draws these from its `scheduler` and the noising time from its `dmd_scheduler`.
    # `cotrain_sample_r_overrides` changes `cotrain_sample_t_cfg` for r (None samples r like t).
    cotrain_sample_t_cfg: MeanFlowSampleTConfig = attrs.field(factory=MeanFlowSampleTConfig)
    cotrain_sample_r_overrides: Optional[Dict[str, Any]] = None
    loss_config: MeanFlowLossConfig = attrs.field(factory=MeanFlowLossConfig)

    # Weight of the co-trained Stage-1 flow-map loss in the student update.
    # The reference runs it at weight 1 (cotrain_forward_kl: True); 0 disables.
    cotrain_pretrain_weight: float = 1.0

    # Rollout NFE list, sampled uniformly per iteration with rank-0 broadcast
    # (reference rollout_cfg.num_inference_steps_list). None falls back to
    # ``student_grid_cfg.steps``.
    student_steps_list: Optional[List[int]] = None

    # Maximum number of differentiable student forwards in the compressed rollout.
    # The reference first step -> fine step -> last step construction corresponds to 3.
    max_steps_rollout: int = 3

    # Prediction-side guidance fusion for the co-trained flow-map loss; see
    # `config_mean_flow.ModelConfig`.
    #
    # Deviates from the AnyFlow reference, which divides dF/dt by g for the whole
    # batch. We divide only on samples that kept their condition: a dropped sample's
    # fused prediction is plain `u_uncond`, so an ungated 1/g would regress it onto
    # a different fixed point.
    guidance_fuse_scale: Optional[float] = None

    # Text dropout for the co-trained flow-map loss (reference drop_text_ratio).
    cond_dropout_prob: Optional[float] = None
    cond_keys_no_dropout: List[str] = attrs.field(factory=list)

    # Autocast dtype for the co-trained loss JVP (None disables autocast there). The JVP state and its network input
    # are kept in at least this dtype; the conditioning stays in precision_model. Unlike FastGen's own dropout, native
    # dropout layers, such as LoRA's, replay the forward's masks only if this dtype equals precision_amp.
    precision_amp_jvp: str | None = None


@attrs.define
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(AnyFlowModel)(config=None))


def create_config():
    config = Config()
    # Same callback set as DMD2's `create_config` (this stage trains a fake
    # score net too, so it keeps the per-model grad-clip hooks, disabled by
    # default) plus the EMA the flow-map student needs.
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            "grad_clip_fake_score": L(GradClipCallback)(grad_norm=None, model_key="fake_score"),
            "grad_clip_discriminator": L(GradClipCallback)(grad_norm=None, model_key="discriminator"),
            **EMA_CALLBACK,
            **BASE_CALLBACKS,
        }
    )

    # The student is a flow-map network with a dual-timestep input.
    config.model.use_ema = True
    config.model.net.backbone.r_timestep = True
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.discriminator_scheduler.warm_up_steps = [0]

    return config
