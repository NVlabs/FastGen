# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Dict, Any, TYPE_CHECKING, Optional
import torch

from fastgen.methods import DMD2Model
from fastgen.methods.common_loss import (
    variational_score_distillation_loss,
    gan_loss_generator,
)
import fastgen.utils.logging_utils as logger
from fastgen.methods.loss_support import f_divergence_weights

if TYPE_CHECKING:
    from fastgen.configs.methods.config_f_distill import ModelConfig

# The collection of weighting functions, depending on the f-divergence and the ratio r
all_f_div_weighting_function = {
    "rkl": lambda r: torch.tensor(1, dtype=r.dtype, device=r.device),
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
        super().__init__(config)
        self.config = config

        assert self.config.gan_loss_weight_gen > 0, "f-distill requires gan_loss_weight_gen > 0"

        assert self.config.f_distill.f_div in all_f_div_weighting_function, (
            f"Unsupported f-divergence {self.config.f_distill.f_div}"
        )
        logger.info(f"Using {self.config.f_distill.f_div}-divergence")
        self.f_div_weighting_function = all_f_div_weighting_function[self.config.f_distill.f_div]

    def build_model(self):
        super().build_model()

        if self.config.f_distill.ratio_normalization:
            self.bin_num = self.config.f_distill.bin_num
            # register buffer for bins, to save/load it as part of the student model
            # Note: These are persistent, so function with FSDP memory-efficient loading
            # IF you make them not persistent, you need to ensure they are initialized in net.reset_parameters
            self.net.register_buffer("bins", torch.ones(self.bin_num))

    def _get_f_div_weighting_h(
        self,
        fake_logits: torch.Tensor,
        t: torch.Tensor,
        *,
        logit_weight: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
        statistics_group=None,
        update_state: bool = True,
    ) -> torch.Tensor:
        """Compute f-distill weights and maintain the student's ratio bins.

        Use ``logit_weight``, ``sample_mask``, and ``statistics_group`` with
        the semantics of ``f_divergence_weights``. Set ``update_state=False``
        to use saved bins without updating them. Call ``f_divergence_weights``
        directly when your training loop owns the statistics.
        """
        options = self.config.f_distill
        bins = self.net.bins if options.ratio_normalization else None
        h, updated = f_divergence_weights(
            fake_logits,
            t,
            self.f_div_weighting_function,
            ratio_lower=options.ratio_lower,
            ratio_upper=options.ratio_upper,
            bins=bins,
            min_t=float(self.net.noise_scheduler.min_t),
            max_t=float(self.net.noise_scheduler.max_t),
            ema_rate=options.ratio_ema_rate,
            logit_weight=logit_weight,
            sample_mask=sample_mask,
            statistics_group=statistics_group,
            update_state=update_state,
        )
        if bins is not None and update_state:
            bins.copy_(updated.to(bins.dtype))
        return h

    def _student_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        data: Dict[str, Any],
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Perform student model update step.

        Args:
            input_student: Input tensor to student network
            t_student: Input time to student network
            t: Time step
            eps: Noise tensor
            data: Original data batch
            condition: Conditioning information
            neg_condition: Negative conditioning

        Returns:
            tuple of (loss_map, outputs)
        """
        # Generate data from student
        gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)
        perturbed_data = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        # Compute the fake score with x0-prediction
        with torch.no_grad():
            fake_score_x0 = self.fake_score(perturbed_data, t, condition=condition, fwd_pred_type="x0")

        # Compute the teacher x0-prediction and gan loss for generator
        teacher_x0, fake_feat = self.teacher(
            perturbed_data,
            t,
            condition=condition,
            feature_indices=self.discriminator.feature_indices,
            fwd_pred_type="x0",
        )
        fake_logits = self.discriminator(fake_feat)
        gan_loss_gen = gan_loss_generator(fake_logits)

        # Apply classifier-free guidance if needed
        if self.config.guidance_scale is not None:
            teacher_x0 = self._apply_classifier_free_guidance(
                perturbed_data, t, teacher_x0, neg_condition=neg_condition
            )

        # Get f-divergence weighting
        h = self._get_f_div_weighting_h(fake_logits, t)

        # Compute the f-distill loss
        f_distill_loss = variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0, additional_scale=h)

        # Compute the final loss
        loss = f_distill_loss + self.config.gan_loss_weight_gen * gan_loss_gen

        # Build output dictionaries
        loss_map = {
            "total_loss": loss,
            "f_distill_loss": f_distill_loss,
            "gan_loss_gen": gan_loss_gen,
            "min_h": h.min() if self.config.f_distill.f_div != "rkl" else torch.tensor(1).to(self.device),
            "avg_h": h.mean() if self.config.f_distill.f_div != "rkl" else torch.tensor(1).to(self.device),
            "max_h": h.max() if self.config.f_distill.f_div != "rkl" else torch.tensor(1).to(self.device),
        }
        outputs = self._get_outputs(gen_data, input_student, condition=condition)

        return loss_map, outputs
