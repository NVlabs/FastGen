# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss extension points preserve defaults and exclude invalid entries."""

from types import SimpleNamespace
from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp

from fastgen.methods.common_loss import (
    denoising_score_matching_loss,
    gan_loss_generator,
    variational_score_distillation_loss,
)
from fastgen.methods.consistency_model.mean_flow import MeanFlowModel
from fastgen.methods.consistency_model.sCM import SCMModel
from fastgen.methods.loss_support import LossReduction, f_divergence_weights
from fastgen.networks.noise_schedule import TrigNoiseSchedule


def _empty_rank_worker(rank, rendezvous):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        logits = torch.zeros(rank, 1)
        h, bins = f_divergence_weights(
            logits,
            torch.full((rank,), 0.25),
            torch.ones_like,
            bins=torch.ones(2),
        )
        torch.testing.assert_close(h, torch.ones(rank))
        torch.testing.assert_close(bins, torch.ones(2))
    finally:
        torch.distributed.destroy_process_group()


def test_empty_rank_ratio_collectives(tmp_path):
    mp.spawn(_empty_rank_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_masked_residual_excludes_nan():
    values = torch.tensor([[1.0, float("nan")]], requires_grad=True)
    reduction = LossReduction(values, torch.tensor([[2.0, 0.0]]))
    loss = reduction.residual(values, mean=True).sum()
    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(torch.autograd.grad(loss, values)[0], torch.tensor([[2.0, 0.0]]))


@pytest.mark.parametrize("kind", ("scm", "l2", "opt_grad"))
@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("input_grad", (False, True))
def test_unit_support_matches_default(kind, empty, input_grad):
    cls = SCMModel if kind == "scm" else MeanFlowModel
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.loss_config = SimpleNamespace(
        tangent_warmup_steps=10,
        tangent_warmup_const=0.1,
        g_norm_spatial_invariance=True,
        prior_weighting_enabled=True,
        divide_x_0_spatial_dim=True,
        norm_method="poly_0.5",
        norm_const=0.01,
        loss_type=kind,
        weight_type=None,
        tangent_spatial_invariance=True,
    )
    model.sigma_data = 0.5
    model.net_trigflow = SimpleNamespace(noise_scheduler=TrigNoiseSchedule())
    model.net = SimpleNamespace(noise_scheduler=SimpleNamespace(min_t=0.0, max_t=1.0))
    prediction = torch.randn(2, 3, 4, requires_grad=True)
    tangent, state, velocity = [torch.randn_like(prediction) for _ in range(3)]
    state.requires_grad_(input_grad)
    t = torch.tensor([0.3, 0.7])
    logvar = torch.full((2, 1), float("nan") if empty else 0.0, requires_grad=True)
    if kind == "scm":
        args = (prediction, tangent, state, velocity, logvar, t, t, 5)
        operation = model._scm_pred_to_loss
    else:
        args = (prediction, tangent, state, velocity, t, t / 2, 5)
        operation = model._mf_pred_to_loss
    if empty:
        actual = operation(*args, loss_weight=torch.zeros_like(prediction))[0]
        torch.testing.assert_close(actual, torch.zeros_like(actual))
        inputs = (prediction, logvar) if kind == "scm" else (prediction,)
        for value, grad in zip(inputs, torch.autograd.grad(actual.sum(), inputs)):
            torch.testing.assert_close(grad, torch.zeros_like(value))
        return
    expected = operation(*args)[0]
    actual = operation(*args, loss_weight=torch.ones_like(prediction))[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), prediction, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), prediction)[0],
    )


def test_vsd_unit_and_masked_support():
    generated = torch.randn(2, 3, 4, requires_grad=True)
    teacher, fake = torch.randn_like(generated), torch.randn_like(generated)
    expected = variational_score_distillation_loss(generated, teacher, fake)
    actual = variational_score_distillation_loss(generated, teacher, fake, loss_weight=torch.ones_like(generated))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, generated, retain_graph=True)[0], torch.autograd.grad(expected, generated)[0]
    )
    mask = torch.ones_like(generated)
    mask[..., 0] = 0
    teacher[..., 0] = float("nan")
    actual = variational_score_distillation_loss(generated, teacher, fake, loss_weight=mask)
    assert torch.isfinite(actual)
    assert torch.isfinite(torch.autograd.grad(actual, generated)[0]).all()


def test_ratio_support_and_frozen_state():
    logits = torch.tensor([[0.0, float("nan")], [float("nan"), float("nan")]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    bins = torch.ones(4)
    h, updated = f_divergence_weights(
        logits,
        torch.tensor([0.2, 0.7]),
        lambda r: torch.ones_like(r),
        bins=bins,
        min_t=0,
        max_t=1,
        logit_weight=weight,
        update_state=False,
    )
    torch.testing.assert_close(updated, bins)
    torch.testing.assert_close(h, torch.tensor([1.0, 0.0]))


def test_masked_score_and_adversarial_losses():
    prediction = torch.tensor([[1.0, float("nan")]], requires_grad=True)
    mask = torch.tensor([[2.0, 0.0]])
    target = torch.zeros_like(prediction)
    score = denoising_score_matching_loss("flow", prediction, flow_target=target, loss_weight=mask)
    torch.testing.assert_close(score, torch.tensor(1.0))
    gan = gan_loss_generator(prediction, loss_weight=mask)
    assert torch.isfinite(gan)
    assert torch.isfinite(torch.autograd.grad(score + gan, prediction)[0]).all()
