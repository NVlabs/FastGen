# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss support and ratio statistics for custom training integrations."""

from collections.abc import Callable

import torch


class LossReduction:
    r"""Reduce squared residuals and compute norms over valid entries.

    Omit ``weight`` to include every entry. Set weights to zero to exclude
    invalid entries before nonlinear operations. Positive weights scale
    squared residuals linearly. Norms and valid counts use only their
    positive support. Means divide by the full element count, including
    masked entries.

    Parameters
    ----------
    reference : torch.Tensor
        Shape reference for broadcasting the weights.
    weight : torch.Tensor or None, optional, default=None
        Finite nonnegative weights. Omit to give every entry unit weight.
        Check these values before calling compiled loss code.

    Examples
    --------
    >>> x = torch.tensor([[2.0, 99.0]])
    >>> LossReduction(x, torch.tensor([[0.5, 0.0]])).residual(x, mean=True)
    tensor([1.])
    """

    def __init__(self, reference: torch.Tensor, weight: torch.Tensor | None = None):
        self.weight = torch.ones_like(reference) if weight is None else torch.broadcast_to(weight, reference.shape)
        self.valid = self.weight > 0
        self.dims = tuple(range(1, reference.ndim))
        self.count = self.valid.sum(self.dims)
        self.active = self.count > 0

    def clean(self, value: torch.Tensor) -> torch.Tensor:
        """Remove excluded entries before nonlinear operations."""
        return torch.where(self.valid, value, torch.zeros_like(value))

    def norm(self, value: torch.Tensor) -> torch.Tensor:
        """Compute each sample's Euclidean norm using only unmasked entries."""
        return torch.linalg.vector_norm(self.clean(value).flatten(1), dim=1)

    def residual(self, value: torch.Tensor, *, mean: bool) -> torch.Tensor:
        """Reduce linearly weighted squared residuals over non-batch dimensions."""
        weighted = self.clean(value).square() * self.weight
        return weighted.mean(self.dims) if mean else weighted.sum(self.dims)


@torch.no_grad()
def f_divergence_weights(
    logits: torch.Tensor,
    t: torch.Tensor,
    weighting_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    ratio_lower: float = 0.1,
    ratio_upper: float = 20.0,
    bins: torch.Tensor | None = None,
    min_t: float = 0.0,
    max_t: float = 1.0,
    ema_rate: float = 0.0,
    logit_weight: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    statistics_group: torch.distributed.ProcessGroup | None = None,
    update_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Compute normalized f-distill weights and return ratio statistics.

    Use this function when your training loop owns the running statistics.
    Pass previous ``bins`` to normalize ratios by diffusion time, then save
    the returned bins for the next call or a checkpoint. Input bins remain
    unchanged. With ``update_state=False``, normalization uses the saved
    values without updating them. With ``bins=None``, it skips bin normalization.

    By default, every score and sample contributes. Zero ``logit_weight``
    entries exclude scores, and positive magnitudes do not change their
    relative contribution. False ``sample_mask`` entries exclude whole samples.
    Samples with no valid scores receive zero weight. Weight normalization
    uses the global valid sample count and supports unequal batch sizes.

    Call on every rank in ``statistics_group``, even ranks with no valid samples.
    Choose a group joining data replicas, with spatial features combined
    within each sample. During distributed execution, ``None``
    uses the default process group. The function leaves its inputs unchanged.

    Examples
    --------
    >>> weights, bins = f_divergence_weights(
    ...     torch.zeros(2, 3), torch.tensor([0.2, 0.8]), torch.ones_like,
    ... )
    >>> weights.shape
    torch.Size([2])
    >>> bins is None
    True
    """
    if logits.ndim != 2 or t.shape != logits.shape[:1]:
        raise ValueError("Expected logits [B, scores] and times [B].")
    if bins is not None and (bins.ndim != 1 or bins.numel() == 0 or max_t <= min_t):
        raise ValueError("Ratio bins require a nonempty vector and increasing time bounds.")
    if logit_weight is None:
        mean = logits.float().mean(1)
        active = torch.ones_like(t, dtype=torch.bool)
    else:
        valid = torch.broadcast_to(logit_weight, logits.shape) > 0
        count = valid.sum(1)
        mean = torch.where(valid, logits.float(), 0).sum(1) / count.clamp_min(1)
        active = count > 0
    if sample_mask is not None:
        active = active & sample_mask
    ratio = mean.clamp(-10, 10).exp().clamp(ratio_lower, ratio_upper)
    updated = bins
    if bins is not None:
        bin_num = bins.numel()
        idx = ((t.double() - min_t) / ((max_t - min_t) / bin_num)).floor().long().clamp(0, bin_num - 1)
        count = torch.bincount(idx, weights=active.float(), minlength=bin_num).float()
        total = torch.bincount(idx, weights=torch.where(active, ratio, 0), minlength=bin_num).float()
        statistics = torch.stack((count, total))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(statistics, group=statistics_group)
        count, total = statistics.unbind(0)
        updated = bins.to(device=t.device, dtype=torch.float32)
        if update_state:
            new_bins = updated * ema_rate + (1 - ema_rate) * total / (count + 1e-6)
            updated = torch.where(count > 0, new_bins, updated)
        ratio = ratio / (updated[idx] + 1e-6)
    h = weighting_fn(ratio).expand_as(ratio)
    totals = torch.stack((torch.where(active, h, 0).sum(), active.float().sum()))
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals, group=statistics_group)
    h_mean = totals[0] / totals[1].clamp_min(1)
    return torch.where(active, h / (h_mean + 1e-6), 0).to(logits.dtype), updated
