# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from typing import Optional

from fastgen.networks.noise_schedule import BaseNoiseSchedule
from fastgen.utils import expand_like
from fastgen.methods.loss_support import LossReduction


def denoising_score_matching_loss(
    pred_type: str,
    net_pred: torch.Tensor,
    x0: torch.Tensor = None,
    eps: torch.Tensor = None,
    noise_scheduler: Optional[BaseNoiseSchedule] = None,
    t: Optional[torch.Tensor] = None,
    *,
    loss_weight: Optional[torch.Tensor] = None,
    flow_target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute mean squared error against the selected denoising target.

    The forward process is ``x_t = alpha_t * x_0 + sigma_t * eps``.
    By default, every residual has unit weight. Supply ``loss_weight`` for
    masking or time-dependent weighting. The mean divides by the full element
    count. For ``pred_type="flow"``, ``flow_target``
    replaces the default velocity ``eps - x0`` for a custom diffusion path.

    Args:
        pred_type (str): Prediction type: 'x0', 'eps', 'v', 'flow'.
        net_pred (torch.Tensor): The network output, and its meaning is determined by pred_type.
        noise_scheduler (BaseNoiseSchedule): Noise scheduler.
        x0 (torch.Tensor): The clean data x_0.
        eps (torch.Tensor): The epsilon used to compute noised data.
        t (torch.Tensor): The target data t.
        loss_weight (Optional[torch.Tensor]): Finite nonnegative weights that broadcast
            to net_pred. Zero entries exclude residuals. Positive values scale squared errors.
        flow_target (Optional[torch.Tensor]): Conditional velocity target used only
            for flow prediction. Defaults to eps - x0.

    Raises:
        NotImplementedError: If an unknown pred_type is used.

    Returns:
        loss (torch.Tensor): The denoising diffusion loss.
    """
    if pred_type == "x0":
        assert x0 is not None, "x0 cannot be None"
        target = x0
    elif pred_type == "eps":
        assert eps is not None, "eps cannot be None"
        target = eps
    elif pred_type == "v":
        assert x0 is not None and eps is not None and t is not None, "x0, eps, t should not be None"
        assert noise_scheduler is not None, "noise_scheduler should not be None"
        alpha_t = expand_like(noise_scheduler.alpha(t), x0).to(device=x0.device, dtype=x0.dtype)
        sigma_t = expand_like(noise_scheduler.sigma(t), x0).to(device=x0.device, dtype=x0.dtype)
        v = alpha_t * eps - sigma_t * x0
        target = v
    elif pred_type == "flow":
        if flow_target is None:
            assert x0 is not None and eps is not None, "x0 and eps cannot be None"
            target = eps - x0
        else:
            target = flow_target
    else:
        raise NotImplementedError(f"Unknown prediction type {pred_type}")
    if loss_weight is None:
        return F.mse_loss(target, net_pred, reduction="mean")
    return LossReduction(net_pred, loss_weight).residual(net_pred - target, mean=True).mean()


def variational_score_distillation_loss(
    gen_data: torch.Tensor,
    teacher_x0: torch.Tensor,
    fake_score_x0: torch.Tensor,
    additional_scale: Optional[torch.Tensor] = None,
    *,
    loss_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the variational score distillation loss.

    By default, all entries contribute to the target normalization and loss.
    Supply ``loss_weight`` to mask entries or scale squared residuals. Zero
    weights also exclude entries from the normalization statistics. Positive
    magnitudes affect only the residual. The mean divides by the full element
    count.

    Args:
        gen_data (torch.Tensor): generated data
        teacher_x0 (torch.Tensor): x0-prediction from the teacher
        fake_score_x0 (torch.Tensor): x0-prediction from the fake score
        additional_scale (Optional[torch.Tensor]): Additional scale parameter for the VSD loss.
        loss_weight (Optional[torch.Tensor]): Finite nonnegative weights that broadcast
            to gen_data. Zero entries exclude residuals and target normalization statistics.

    Returns:
        loss (torch.Tensor): The variational score distillation loss.
    """
    dims = tuple(range(1, teacher_x0.ndim))
    reduction = LossReduction(gen_data, loss_weight) if loss_weight is not None else None

    with torch.no_grad():
        # Perform weight calculation in fp32 for numerical stability
        original_dtype = gen_data.dtype
        gen_data_fp32 = gen_data.float()
        teacher_x0_fp32 = teacher_x0.float()

        # Compute weight in fp32 to avoid numerical instability
        if reduction is None:
            diff_abs_mean = (gen_data_fp32 - teacher_x0_fp32).abs().mean(dim=dims, keepdim=True)
        else:
            difference = reduction.clean(gen_data_fp32 - teacher_x0_fp32)
            diff_abs_mean = difference.abs().sum(dims) / reduction.count.clamp_min(1)
            diff_abs_mean = expand_like(diff_abs_mean, gen_data)
        w_fp32 = 1 / (diff_abs_mean + 1e-6)

        # Apply additional scale if provided
        if additional_scale is not None:
            w_fp32 *= expand_like(additional_scale.float(), w_fp32)

        # Convert weight back to original precision
        w = w_fp32.to(dtype=original_dtype)

        vsd_grad = (fake_score_x0 - teacher_x0) * w
        if reduction is not None:
            vsd_grad = reduction.clean(vsd_grad)
        pseudo_target = gen_data - vsd_grad

    loss = 0.5 * (
        F.mse_loss(gen_data, pseudo_target, reduction="mean")
        if reduction is None
        else reduction.residual(gen_data - pseudo_target, mean=True).mean()
    )
    return loss


def gan_loss_generator(fake_logits: torch.Tensor, *, loss_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the generator's mean softplus loss over discriminator scores.

    By default, every score contributes equally. Supply ``loss_weight`` to
    exclude or scale score losses while retaining the full score count in the
    mean. Determine valid scores during feature extraction and pooling. A
    score mask cannot undo corruption that has spread through the backbone.

    Args:
        fake_logits (torch.Tensor): The logits for the fake data.
        loss_weight (Optional[torch.Tensor]): Finite nonnegative weights that broadcast
            to fake_logits. Zero entries exclude scores. Positive values scale their losses.

    Returns:
        gan_loss (torch.Tensor): The GAN loss for the generator.

    """

    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    if loss_weight is None:
        gan_loss = F.softplus(-fake_logits).mean()
    else:
        reduction = LossReduction(fake_logits, loss_weight)
        gan_loss = (reduction.clean(F.softplus(-reduction.clean(fake_logits))) * reduction.weight).mean()
    return gan_loss


def gan_loss_discriminator(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    *,
    real_weight: Optional[torch.Tensor] = None,
    fake_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sum the discriminator's mean softplus losses for real and fake scores.

    By default, every score contributes equally. Use ``real_weight`` and
    ``fake_weight`` to exclude or scale each set of score losses, with the
    same reduction and feature-validity requirements as ``gan_loss_generator``.

    Args:
        real_logits (torch.Tensor): The logits for the real data.
        fake_logits (torch.Tensor): The logits for the fake data.
        real_weight, fake_weight (Optional[torch.Tensor]): Finite nonnegative weights
            that broadcast to the corresponding logits. Zero entries exclude scores.

    Returns:
        gan_loss (torch.Tensor): The GAN loss for the discriminator.
    """
    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    assert real_logits.ndim == 2, f"real_logits has shape {real_logits.shape}"
    gan_loss = gan_loss_generator(-fake_logits, loss_weight=fake_weight) + gan_loss_generator(
        real_logits, loss_weight=real_weight
    )

    return gan_loss
