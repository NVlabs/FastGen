# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for `fastgen.methods.elastification.budget_utils.get_num_parameters`.

Covers:
  - Concrete int inputs return a plain int
  - Soft tensor inputs return a differentiable tensor
  - Linearity: params scale linearly with mlp_hidden, ~quadratically with hidden
  - Layer skip weights per-block contribution correctly
  - Gradient flows back to router-like soft probs in a budget-loss-style call
"""

from __future__ import annotations

import pytest
import torch

from fastgen.methods.elastification.budget_utils import get_num_parameters


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def wan_1_3b_kwargs() -> dict:
    """Approximate Wan-1.3B-T2V config (hidden 1536, mlp 8960, 30 blocks)."""
    return dict(
        num_layers=30,
        hidden_size=1536,
        mlp_hidden_size=8960,
        num_attention_heads=12,
        head_dim=128,
        text_encoder_dim=4096,
        in_channels=16,
        out_channels=16,
        patch_dim=8,
    )


# ─── Basic shape / type ──────────────────────────────────────────────────


class TestReturnType:
    def test_int_inputs_return_int(self, wan_1_3b_kwargs):
        p = get_num_parameters(**wan_1_3b_kwargs)
        assert isinstance(p, int)

    def test_tensor_hidden_returns_tensor(self, wan_1_3b_kwargs):
        kw = {**wan_1_3b_kwargs, "hidden_size": torch.tensor(1152.0, requires_grad=True)}
        p = get_num_parameters(**kw)
        assert isinstance(p, torch.Tensor)
        assert p.requires_grad

    def test_tensor_mlp_returns_tensor(self, wan_1_3b_kwargs):
        kw = {**wan_1_3b_kwargs, "mlp_hidden_size": torch.tensor(4480.0, requires_grad=True)}
        p = get_num_parameters(**kw)
        assert isinstance(p, torch.Tensor)
        assert p.requires_grad

    def test_positive_value(self, wan_1_3b_kwargs):
        assert get_num_parameters(**wan_1_3b_kwargs) > 0


# ─── Scaling behavior ────────────────────────────────────────────────────


class TestScaling:
    def test_halving_mlp_hidden_reduces_params(self, wan_1_3b_kwargs):
        p_full = get_num_parameters(**wan_1_3b_kwargs)
        p_half = get_num_parameters(**{**wan_1_3b_kwargs, "mlp_hidden_size": 4480})
        assert p_half < p_full

    def test_halving_hidden_reduces_params_more_than_mlp(self, wan_1_3b_kwargs):
        """`hidden_size` appears in linear+quadratic terms; `mlp_hidden` only linear.
        Halving `hidden` should therefore shrink the total more than halving `mlp_hidden`."""
        p_full = get_num_parameters(**wan_1_3b_kwargs)
        p_half_hidden = get_num_parameters(**{**wan_1_3b_kwargs, "hidden_size": 768})
        p_half_mlp = get_num_parameters(**{**wan_1_3b_kwargs, "mlp_hidden_size": 4480})
        assert p_half_hidden < p_half_mlp < p_full

    def test_more_layers_more_params(self, wan_1_3b_kwargs):
        p_30 = get_num_parameters(**wan_1_3b_kwargs)
        p_60 = get_num_parameters(**{**wan_1_3b_kwargs, "num_layers": 60})
        assert p_60 > p_30

    def test_zero_layers_reduces_to_input_output_boilerplate(self, wan_1_3b_kwargs):
        p_zero = get_num_parameters(**{**wan_1_3b_kwargs, "num_layers": 0})
        p_full = get_num_parameters(**wan_1_3b_kwargs)
        assert 0 < p_zero < p_full


# ─── Layer-skip weighting ────────────────────────────────────────────────


class TestLayerSkip:
    def test_all_zeros_skip_matches_no_skip(self, wan_1_3b_kwargs):
        p_no_skip = get_num_parameters(**wan_1_3b_kwargs)
        p_zeros = get_num_parameters(
            **wan_1_3b_kwargs,
            layer_skip_probs=torch.zeros(wan_1_3b_kwargs["num_layers"]),
        )
        assert p_no_skip == pytest.approx(p_zeros.item(), rel=1e-6)

    def test_all_ones_skip_reduces_to_boilerplate(self, wan_1_3b_kwargs):
        p_ones = get_num_parameters(
            **wan_1_3b_kwargs,
            layer_skip_probs=torch.ones(wan_1_3b_kwargs["num_layers"]),
        )
        p_zero_layers = get_num_parameters(**{**wan_1_3b_kwargs, "num_layers": 0})
        assert p_ones.item() == pytest.approx(p_zero_layers, rel=1e-6)

    def test_half_skip_reduces_params(self, wan_1_3b_kwargs):
        p_full = get_num_parameters(**wan_1_3b_kwargs)
        skip = torch.zeros(wan_1_3b_kwargs["num_layers"])
        skip[::2] = 1.0
        p_half_skip = get_num_parameters(**wan_1_3b_kwargs, layer_skip_probs=skip)
        assert p_half_skip.item() < p_full

    def test_wrong_skip_shape_raises(self, wan_1_3b_kwargs):
        with pytest.raises(AssertionError):
            get_num_parameters(
                **wan_1_3b_kwargs,
                layer_skip_probs=torch.zeros(wan_1_3b_kwargs["num_layers"] + 1),
            )


# ─── Differentiability (budget-loss-style call) ─────────────────────────


class TestGradientFlow:
    def test_gradient_flows_to_soft_router_probs(self, wan_1_3b_kwargs):
        """Mirrors how the manager will use this: soft router probs @ candidate
        list → expected width → param count → budget-loss."""
        emb_list = torch.tensor([1536.0, 1152.0, 768.0, 384.0])
        mlp_list = torch.tensor([8960.0, 6720.0, 4480.0, 2240.0])
        router_emb = torch.tensor([0.4, 0.3, 0.2, 0.1], requires_grad=True)
        router_mlp = torch.tensor([0.4, 0.3, 0.2, 0.1], requires_grad=True)

        exp_h = router_emb @ emb_list
        exp_mlp = router_mlp @ mlp_list

        kw = {**wan_1_3b_kwargs, "hidden_size": exp_h, "mlp_hidden_size": exp_mlp}
        p_expected = get_num_parameters(**kw)
        p_full = get_num_parameters(**wan_1_3b_kwargs)

        loss = torch.abs(p_expected / p_full - 0.5)
        loss.backward()

        assert router_emb.grad is not None
        assert router_emb.grad.abs().sum() > 0
        assert router_mlp.grad is not None
        assert router_mlp.grad.abs().sum() > 0
