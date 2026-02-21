# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import torch
from omegaconf import DictConfig
from fastgen.configs.methods.config_dmd2 import create_config
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.trainer import Trainer
from fastgen.utils import LazyCall as L, instantiate
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
    EMA_CALLBACK,
    GPUStats_CALLBACK,
    TrainProfiler_CALLBACK,
    ParamCount_CALLBACK,
)


class SyntheticLoader:
    """Minimal loader that yields synthetic data with the given shape (e.g. (3, 2, 2) for C, H, W)."""

    def __init__(
        self,
        shape: tuple[int, ...],
        batch_size: int = 1,
        sampler_start_idx: int = 0,
        condition_dim: int = 10,
        **kwargs,
    ):
        self.shape = tuple(shape)
        self.batch_size = batch_size
        self.sampler_start_idx = sampler_start_idx
        self.condition_dim = condition_dim

    def __iter__(self):
        while True:
            real = torch.randn(self.batch_size, *self.shape, dtype=torch.float32)
            condition = torch.zeros(self.batch_size, self.condition_dim, dtype=torch.float32)
            condition[:, 0] = 1.0
            neg_condition = torch.zeros(self.batch_size, self.condition_dim, dtype=torch.float32)
            idx = torch.zeros(self.batch_size, dtype=torch.int64)
            yield {
                "real": real,
                "condition": condition,
                "neg_condition": neg_condition,
                "idx": idx,
            }


def test_trainer():
    config = create_config()
    config.log_config.name = "test"
    config.trainer.callbacks = DictConfig(
        {**GradClip_CALLBACK, **EMA_CALLBACK, **GPUStats_CALLBACK, **TrainProfiler_CALLBACK, **ParamCount_CALLBACK}
    )

    model_config = config.model
    model_config.gan_loss_weight_gen = 0.01
    # 8x8 minimum: discriminator head uses 4x4 kernels in 8->4->1 pipeline
    input_shape = [3, 8, 8]
    opts_network = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1"]
    model_config.teacher = override_config_with_opts(model_config.net, opts_network)
    model_config.net = override_config_with_opts(model_config.net, opts_network)
    model_config.use_ema = True
    model_config.student_update_freq = 1

    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    model_config.discriminator = override_config_with_opts(model_config.discriminator, opts_discriminator)

    model_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config.pretrained_model_path = ""  # disable ckpt loading
    model_config.input_shape = input_shape

    # Use synthetic loader so real_data shape matches gen_data (avoids shape mismatch in forward_process)
    config.dataloader_train = L(SyntheticLoader)(shape=input_shape, batch_size=1, sampler_start_idx=None)
    config.dataloader_val = L(SyntheticLoader)(shape=input_shape, batch_size=1, sampler_start_idx=None)

    config.trainer = override_config_with_opts(
        config.trainer, ["-", "max_iter=2", "save_ckpt_iter=1", "validation_iter=1", "logging_iter=1"]
    )
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 2}]  # limit validation to 1 step

    with tempfile.TemporaryDirectory() as tmpdir:
        config.trainer.checkpointer.save_dir = tmpdir

        # initiate the model
        config.model_class.config = config.model
        model = instantiate(config.model_class)
        config.model_class.config = None

        # initiate the trainer
        fastgen_trainer = Trainer(config)
        fastgen_trainer.run(model)
