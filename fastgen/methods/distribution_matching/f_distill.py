# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Dict, TYPE_CHECKING, Optional
import torch

from fastgen.methods import DMD2Model
from fastgen.methods.common_loss import variational_score_distillation_loss
from fastgen.utils.tensor import to_high_precision
from fastgen.utils.tree_utils import TensorOrDict, assert_same, per_modality, to_dict
import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed import world_size

if TYPE_CHECKING:
    from fastgen.configs.methods.config_f_distill import ModelConfig

# The collection of weighting functions, depending on the f-divergence and the ratio r
all_f_div_weighting_function = {
    "rkl": torch.ones_like,
    "kl": lambda r: r,
    "js": lambda r: 1 - 1 / (1 + r),
    "sf": lambda r: 1 / (1 + r),
    "neyman": lambda r: 1 / torch.clamp(r, min=1e-8),  # Add clamp to prevent division by zero
    "sh": lambda r: r**0.5,  # Squared Hellinger distance
    "jf": lambda r: 1 + r,  # Jeffreys divergence
}


class FdistillModel(DMD2Model):
    def __init__(self, config: ModelConfig):
        """

        Args:
            config (ModelConfig): The configuration for the F-distill model
        """
        if config.vsd_loss_weight <= 0:
            raise ValueError("f-distill requires vsd_loss_weight > 0")
        if config.gan_loss_weight_gen <= 0:
            raise ValueError("f-distill requires gan_loss_weight_gen > 0")
        if config.f_distill.f_div not in all_f_div_weighting_function:
            raise ValueError(f"Unsupported f-divergence {config.f_distill.f_div}")
        super().__init__(config)

        logger.info(f"Using {self.config.f_distill.f_div}-divergence")
        self.f_div_weighting_function = all_f_div_weighting_function[self.config.f_distill.f_div]

    def build_model(self):
        super().build_model()

        if self.config.f_distill.ratio_normalization and not self.config.inference:
            assert hasattr(self, "discriminator"), "f-distill ratio normalization requires a discriminator"
            self.bin_num = self.config.f_distill.bin_num
            # Persistent, so the density-ratio history is checkpointed and resumed with the discriminator.
            self.discriminator.register_buffer("bins", torch.ones(self.bin_num))

    def _get_f_div_weighting_h(self, fake_logits: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert fake_logits.ndim == 2, f"fake_logits expected to be of rank 2, got {fake_logits.ndim}"

        # Compute in high precision for numerical stability.
        fake_logits, t = to_high_precision(fake_logits, t)

        # Clamp fake_logits to prevent exponential overflow.
        fake_logits_clamped = torch.clamp(fake_logits.mean(dim=1), min=-10.0, max=10.0)
        ratio = torch.exp(fake_logits_clamped).detach()
        ratio = torch.clamp(ratio, self.config.f_distill.ratio_lower, self.config.f_distill.ratio_upper)
        assert_same(shape=True, device=True, ratio=ratio, t=t)

        if self.config.f_distill.ratio_normalization:
            # get bin idx
            max_t = float(self.net.noise_scheduler.max_t)
            min_t = float(self.net.noise_scheduler.min_t)
            bin_width = (max_t - min_t) / self.bin_num
            idx = ((t - min_t) / bin_width).floor().long().clamp(0, self.bin_num - 1)

            # Bin statistics use the same promoted loss dtype.
            cnt = torch.bincount(idx, minlength=self.bin_num).to(fake_logits.dtype)
            ratio_sum = torch.bincount(idx, weights=ratio, minlength=self.bin_num)

            if world_size() > 1:
                torch.distributed.all_reduce(cnt)
                torch.distributed.all_reduce(ratio_sum)

            # EMA update only where cnt > 0.
            valid = cnt > 0
            new_vals = ratio_sum / (cnt + 1e-6)
            ema_rate = self.config.f_distill.ratio_ema_rate

            # Update in the promoted loss dtype, then copy back to the buffer dtype.
            bins = self.discriminator.bins.to(fake_logits.dtype)
            if torch.is_grad_enabled():
                # Validation runs under no_grad and must not change the training density-ratio history.
                bins[valid] = bins[valid] * ema_rate + (1 - ema_rate) * new_vals[valid]
                self.discriminator.bins.copy_(bins.to(self.discriminator.bins.dtype))

            # Normalize ratio based on the EMA histogram with a stability epsilon.
            ratio = ratio / (bins[idx] + 1e-6)

        # Normalize the weighting function h in the promoted dtype.
        h = self.f_div_weighting_function(ratio)
        h_sum = h.sum()
        if world_size() > 1:
            torch.distributed.all_reduce(h_sum)
        h_mean = h_sum / (len(h) * world_size())

        # Add epsilon to prevent division by zero in h normalization.
        h = h / (h_mean + 1e-6)

        return h

    def _compute_distribution_matching_loss(
        self,
        gen_data: TensorOrDict,
        teacher_x0: TensorOrDict,
        fake_score_x0: TensorOrDict,
        fake_logits: Optional[torch.Tensor],
        t: torch.Tensor,
        loss_scale: Optional[torch.Tensor] = None,
        gen_x0: Optional[TensorOrDict] = None,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Compute the density-ratio-weighted VSD objective and h metrics."""
        if fake_logits is None:
            raise ValueError("F-distill weights by the discriminator, which trains only on data['real']")
        h = self._get_f_div_weighting_h(fake_logits, t)
        # Since the pseudo-target is detached, outer scaling gives
        # dL/d(gen_data) = h * vsd_grad, just like scaling its residual by h.
        vsd_loss_scale = h if loss_scale is None else h * loss_scale
        f_distill_per_modality = per_modality(
            variational_score_distillation_loss,
            gen_data,
            teacher_x0,
            fake_score_x0,
            loss_scale=vsd_loss_scale,
            gen_x0=gen_x0,
        )
        loss_map = to_dict(f_distill_per_modality, suffix="f_distill_loss")
        f_distill_loss = sum(loss_map.values())

        log_h = self.config.f_distill.f_div != "rkl"
        one = torch.ones_like(f_distill_loss)
        loss_map.update(
            {
                "min_h": h.min() if log_h else one,
                "avg_h": h.mean() if log_h else one,
                "max_h": h.max() if log_h else one,
            }
        )
        return loss_map, f_distill_loss
