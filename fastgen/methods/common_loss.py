# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from typing import Any, Optional

from fastgen.networks.noise_schedule import BaseNoiseSchedule
from fastgen.utils.tensor import assert_tensor, expand_like, to_high_precision
from fastgen.utils.tree_utils import TensorOrDict, assert_same, per_modality


def mean_with_loss_scale(
    elementwise_loss: torch.Tensor,
    loss_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """Scale each sample's mean loss, then average over the original batch."""
    if loss_scale is None:
        return elementwise_loss.mean()
    assert elementwise_loss.ndim > 0, "A per-sample loss scale cannot be applied to a scalar loss."
    assert_same(shape=(elementwise_loss.shape[0],), device=elementwise_loss.device, loss_scale=loss_scale)

    non_batch_dims = tuple(range(1, elementwise_loss.ndim))
    per_sample_loss = elementwise_loss.mean(dim=non_batch_dims) if non_batch_dims else elementwise_loss
    # Per sample, so in the widest dtype, also that of a time-dependent scale.
    per_sample_loss, loss_scale = to_high_precision(per_sample_loss, loss_scale)
    return (per_sample_loss * loss_scale).mean()


def mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean squared error with an optional multiplicative scale per sample."""
    # Dtypes may differ, but shapes must not: broadcasting would silently change the loss.
    assert_same(shape=True, device=True, prediction=prediction, target=target)
    prediction, target = to_high_precision(prediction, target)
    if loss_scale is None:
        return F.mse_loss(prediction, target, reduction="mean")
    squared_error = (prediction - target).square()
    return mean_with_loss_scale(squared_error, loss_scale)


def flow_matching_target(
    pred_type: str,
    x0: Optional[torch.Tensor] = None,
    eps: Optional[torch.Tensor] = None,
    noise_scheduler: Optional[BaseNoiseSchedule] = None,
    t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The flow matching target of a ``pred_type`` prediction at ``x_t = alpha_t * x_0 + sigma_t * eps``.

    ``x0`` and ``eps`` are returned as given; the v and flow targets are computed in high precision.

    Args:
        pred_type (str): Prediction type: 'x0', 'eps', 'v', 'flow'.
        x0 (torch.Tensor): The clean data x_0.
        eps (torch.Tensor): The epsilon used to compute noised data.
        noise_scheduler (BaseNoiseSchedule): Noise scheduler (required for pred_types 'v' and 'flow').
        t (torch.Tensor): The diffusion timestep (required for pred_types 'v' and 'flow').

    Raises:
        NotImplementedError: If an unknown pred_type is used.
    """
    if pred_type == "x0":
        assert x0 is not None, "x0 cannot be None"
        return x0
    if pred_type == "eps":
        assert eps is not None, "eps cannot be None"
        return eps
    if pred_type == "v":
        assert x0 is not None and eps is not None and t is not None, "x0, eps, t should not be None"
        assert noise_scheduler is not None, "v target needs the noise_scheduler"
        return noise_scheduler.cond_v(x0, eps, t, output_dtype=None)
    if pred_type == "flow":
        assert x0 is not None and eps is not None and t is not None, "x0, eps, t cannot be None"
        assert noise_scheduler is not None, "flow target needs the noise_scheduler"
        return noise_scheduler.cond_velocity(x0, eps, t, output_dtype=None)
    raise NotImplementedError(f"Unknown prediction type {pred_type}")


def flow_matching_loss(
    net_pred: torch.Tensor,
    pred_type: str,
    x0: Optional[torch.Tensor] = None,
    eps: Optional[torch.Tensor] = None,
    noise_scheduler: Optional[BaseNoiseSchedule] = None,
    t: Optional[torch.Tensor] = None,
    loss_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the flow matching objective.
        Forward process:
            x_t = alpha_t * x_0 + sigma_t * eps

        Note: This function does not derive a time-dependent weight (for example,
            from SNR), but callers may provide an explicit per-sample loss scale.

    Args:
        net_pred (torch.Tensor): The network output, and its meaning is determined by pred_type.
        pred_type (str): Prediction type: 'x0', 'eps', 'v', 'flow'.
        x0 (torch.Tensor): The clean data x_0.
        eps (torch.Tensor): The epsilon used to compute noised data.
        noise_scheduler (BaseNoiseSchedule): Noise scheduler (required for pred_type 'v').
        t (torch.Tensor): The diffusion timestep.
        loss_scale (Optional[torch.Tensor]): Multiplicative scale for each
            sample's loss before the ordinary batch mean.

    Raises:
        NotImplementedError: If an unknown pred_type is used.

    Returns:
        loss (torch.Tensor): The flow matching loss.
    """
    assert_same(shape=True, dtype=True, device=True, net_pred=net_pred, x0=x0, eps=eps)
    target = flow_matching_target(pred_type, x0=x0, eps=eps, noise_scheduler=noise_scheduler, t=t)
    return mse_loss(net_pred, target, loss_scale)


def transition_score_weight(
    weighting: str,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    variance: torch.Tensor,
    data_variance: float = 1.0,
    clamp_min: float = 0.0,
) -> torch.Tensor:
    """Return the unit-noise residual weight ``W`` for transition variance ``v = variance``.

    For equivalent x0-loss weight ``w``, ``W = w sigma^4 / (alpha^2 v)``. With
    ``rho = alpha^2 tau^2 / (alpha^2 tau^2 + sigma^2 - v)`` and ``kappa = sigma^2 / (rho v)``,
    ``weighting`` selects ``w`` as ``kappa^-1/2`` (``variance_matched``), ``kappa^-1``
    (``inverse_variance``), ``v / sigma^2`` (``conservative``), or 1 (``x0``);
    ``unweighted`` returns ``W = 1``. Divisors are floored at ``clamp_min``, and carried-over
    noise variance ``sigma^2 - v`` is clipped at zero.
    """
    if weighting == "unweighted":
        return torch.ones_like(sigma)
    alpha = alpha.clamp(min=clamp_min)
    if weighting == "conservative":
        return sigma**2 / alpha**2
    if weighting == "x0":
        return sigma**4 / (alpha**2 * variance)
    signal = alpha**2 * data_variance
    # sigma**2 - variance is carried-over noise variance; clamp roundoff or flooring artifacts to zero.
    signal_fraction = signal / (signal + (sigma**2 - variance).clamp(min=0)).clamp(min=clamp_min)
    if weighting == "inverse_variance":
        return sigma**2 * signal_fraction / alpha**2
    if weighting == "variance_matched":
        return sigma**3 * signal_fraction.sqrt() / (alpha**2 * variance.sqrt())
    raise ValueError(
        f"Unknown transition score weighting {weighting!r}; choose one of "
        "variance_matched, inverse_variance, conservative, x0, unweighted."
    )


def transition_score_loss(
    eps_pred: torch.Tensor,
    eps: torch.Tensor,
    noise_scheduler: BaseNoiseSchedule,
    t_start: torch.Tensor,
    t: torch.Tensor,
    weighting: str,
    data_variance: float = 1.0,
    loss_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted unit-noise loss of an eps prediction at ``t`` for states carried there from ``t_start``.

    ``eps`` is the innovation of the forward transition; its variance ``v`` is floored at the schedule's
    ``clamp_min``, like its other divisors, so transitions of (near) zero length stay finite. Their gradient carries
    ``sqrt(v)`` from the residual, which keeps it bounded for every weighting but ``"x0"``, whose ``W ~ 1/v`` is
    bounded only by that floor: a transition on the floor weighs ``sigma^4 / (alpha^2 clamp_min)``.
    """
    assert_same(shape=True, device=True, eps_pred=eps_pred, eps=eps)
    # Per sample, so in the widest dtype of the times; the states resolve their own below.
    t_start, t = to_high_precision(t_start, t)
    alpha, sigma = noise_scheduler.alpha_sigma(t)
    sigma = noise_scheduler.non_zero_clamp(sigma)
    variance = noise_scheduler.transition_std(t, t_start).square().clamp(min=noise_scheduler.clamp_min)
    weight = transition_score_weight(weighting, alpha, sigma, variance, data_variance, noise_scheduler.clamp_min)

    eps_pred, eps = to_high_precision(eps_pred, eps)
    residual = eps - expand_like((variance.sqrt() / sigma).to(eps.dtype), eps) * eps_pred
    elementwise_loss = expand_like(weight.to(eps.dtype), residual) * residual.square()
    # ``t`` and ``t_start`` can broadcast the time-dependent factors past the inputs' shape.
    assert_same(shape=True, device=True, eps=eps, elementwise_loss=elementwise_loss)
    return mean_with_loss_scale(elementwise_loss, loss_scale)


def variational_score_distillation_loss(
    gen_data: torch.Tensor,
    teacher_x0: torch.Tensor,
    fake_score_x0: torch.Tensor,
    loss_scale: Optional[torch.Tensor] = None,
    gen_x0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the variational score distillation loss.
    Args:
        gen_data (torch.Tensor): generated data
        teacher_x0 (torch.Tensor): x0-prediction from the teacher
        fake_score_x0 (torch.Tensor): x0-prediction from the fake score
        loss_scale (Optional[torch.Tensor]): Multiplicative scale applied to
            each sample's final surrogate loss, outside the detached pseudo-target
            construction.
        gen_x0 (Optional[torch.Tensor]): Clean estimate of the generated data, whose
            residual against ``teacher_x0`` normalizes the gradient. Defaults to ``gen_data``.

    Returns:
        loss (torch.Tensor): The variational score distillation loss.
    """
    if gen_x0 is None:
        gen_x0 = gen_data
    assert_same(
        shape=True,
        device=True,
        gen_data=gen_data,
        teacher_x0=teacher_x0,
        fake_score_x0=fake_score_x0,
        gen_x0=gen_x0,
    )
    gen_data, teacher_x0, fake_score_x0, gen_x0 = to_high_precision(gen_data, teacher_x0, fake_score_x0, gen_x0)
    dims = tuple(range(1, teacher_x0.ndim))

    with torch.no_grad():
        diff_abs_mean = (gen_x0 - teacher_x0).abs().mean(dim=dims, keepdim=True)
        normalization = 1 / (diff_abs_mean + 1e-6)
        vsd_grad = (fake_score_x0 - teacher_x0) * normalization
        pseudo_target = gen_data - vsd_grad

    loss = 0.5 * mse_loss(gen_data, pseudo_target, loss_scale)
    return loss


def gan_loss_generator(
    fake_logits: torch.Tensor,
    loss_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the GAN loss for the generator
    Args:
        fake_logits (torch.Tensor): The logits for the fake data.
        loss_scale (Optional[torch.Tensor]): Multiplicative scale for each
            sample's loss before the ordinary batch mean.

    Returns:
        gan_loss (torch.Tensor): The GAN loss for the generator.

    """

    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    fake_logits = to_high_precision(fake_logits)
    gan_loss = mean_with_loss_scale(F.softplus(-fake_logits), loss_scale)
    return gan_loss


def gan_loss_discriminator(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    loss_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the GAN loss for the discriminator
    Args:
        real_logits (torch.Tensor): The logits for the real data.
        fake_logits (torch.Tensor): The logits for the fake data.
        loss_scale (Optional[torch.Tensor]): Multiplicative scale applied to
            each sample's paired real and fake terms before the ordinary batch mean.

    Returns:
        gan_loss (torch.Tensor): The GAN loss for the discriminator.
    """
    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    assert real_logits.ndim == 2, f"real_logits has shape {real_logits.shape}"
    real_logits, fake_logits = to_high_precision(real_logits, fake_logits)
    gan_loss = mean_with_loss_scale(F.softplus(fake_logits), loss_scale) + mean_with_loss_scale(
        F.softplus(-real_logits), loss_scale
    )

    return gan_loss


def gan_loss_approx_r1(
    real_feat_logit: torch.Tensor,
    perturbed_real: TensorOrDict,
    t_real: torch.Tensor,
    teacher: torch.nn.Module,
    discriminator: torch.nn.Module,
    perturbation_scale: float,
    condition: Optional[Any] = None,
    loss_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the approximate R1 regularization loss for the discriminator.

    Finite-difference form of the R1 gradient penalty (APT,
    https://arxiv.org/abs/2501.08316). For discriminator logits ``F``, a Gaussian
    direction ``eta`` and a perturbation scale ``d``, a first-order expansion together
    with ``E[eta eta^T] = I`` gives

        E ||F(z + d * eta) - F(z)||^2 = d^2 ||J_z F||_F^2 + O(d^3),

    so the squared logit difference estimates the squared Jacobian norm up to the
    factor ``d**2``, which the caller absorbs into its regularization weight.

    The perturbation is applied in the network's *input* space, i.e. to the already
    noised ``x_t``. Perturbing the clean latent instead would displace the input by
    ``alpha_t * d * eta``, targeting ``||J_x F||_F^2`` but weighting each sample by
    ``alpha_t**2``, which under a shifted timestep schedule suppresses the penalty for
    most of the batch. Input-space perturbation weights all timesteps equally, at the
    cost of targeting ``J_{x_t}`` rather than ``J_x``.

    Args:
        real_feat_logit (torch.Tensor): Discriminator logits for the unperturbed
            noised real data.
        perturbed_real (TensorOrDict): Noised real data ``x_t`` that produced
            ``real_feat_logit``.
        t_real (torch.Tensor): Time step used to produce ``perturbed_real``.
        teacher (torch.nn.Module): Frozen feature extractor.
        discriminator (torch.nn.Module): Discriminator head.
        perturbation_scale (float): Standard deviation of the Gaussian perturbation.
        condition (Optional[Any]): Conditioning information.
        loss_scale (Optional[torch.Tensor]): Multiplicative scale for each
            sample's loss before the ordinary batch mean.

    Returns:
        gan_loss_ar1 (torch.Tensor): The approximate R1 regularization loss.
    """
    perturbed_real_alpha = per_modality(
        lambda z: z + perturbation_scale * torch.randn_like(z),
        perturbed_real,
    )
    with torch.no_grad():
        real_feat_alpha = teacher(
            perturbed_real_alpha,
            t_real,
            condition=condition,
            return_features_early=True,
            feature_indices=discriminator.feature_indices,
        )
    real_feat_alpha_logit = discriminator(real_feat_alpha)
    # Match APT's unnormalized convention; the caller's weight absorbs the d**2 factor.
    return mse_loss(real_feat_logit, real_feat_alpha_logit, loss_scale)


def finite_difference_points(
    x_t: torch.Tensor,
    t: torch.Tensor,
    dxt_dt: torch.Tensor,
    step: torch.Tensor,
    forward_valid: torch.Tensor,
    backward_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The state-time points of a directional finite difference with *step* along ``(dxt_dt, 1)``, and the factor
    that turns the difference of the evaluations at them into a derivative.

    The difference is central (``t ± step``) where both steps are valid, one-sided where only one is, and zero where
    neither is: there ``t_plus = t_minus = t``, so both evaluations stay at the original point, and the factor is 0.
    Everything is computed in the precision of the inputs, in which a valid step must be positive and must not round
    away; callers cast the returned times to the network's.
    """
    t_plus = torch.where(forward_valid, t + step, t)
    t_minus = torch.where(backward_valid, t - step, t)
    span = t_plus - t_minus
    assert_tensor(
        ((span > 0) | ~(forward_valid | backward_valid)).all(),
        "The finite-difference step is not positive or rounds away at t; take a larger step or the exact JVP",
        error_type=ValueError,
    )
    x_t_plus = x_t + expand_like((t_plus - t).to(x_t.dtype), dxt_dt) * dxt_dt
    x_t_minus = x_t + expand_like((t_minus - t).to(x_t.dtype), dxt_dt) * dxt_dt
    factor = torch.where(span > 0, 1.0 / torch.where(span > 0, span, 1.0), 0.0)
    return x_t_plus, t_plus, x_t_minus, t_minus, factor
