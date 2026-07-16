# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for `fastgen.methods.elastification.router.FlextronRouter`.

Covers:
  - Construction: config wiring, gate MLP shapes + init, DP awareness, scaler
  - Forward path: trained-budget lookup, `budget=1.0` fallback, interpolation,
    fwd_pass_count increment
  - DP-seeded Gumbel-softmax: RNG isolation, determinism, rank/iter/fwd-count
    sensitivity
  - Tau decay schedule
  - Gradient flow (with non-invariant loss — softmax outputs sum to 1)
  - add_skipping=True path (skip gate + per-layer mask)
  - flex_hetero_ffn=True path (per-block choice list)
  - Scaler-off default config (was a Megatron NameError we fixed)
"""

from __future__ import annotations

import pytest
import torch

from fastgen.methods.elastification.config import FlextronConfig
from fastgen.methods.elastification.router import FlextronRouter


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def basic_config() -> FlextronConfig:
    """Config with elasticity on both axes, scaler off (the default v1 recipe)."""
    return FlextronConfig(
        emb_int_list=[5120, 3840, 2560, 1280],
        mlp_int_list=[13824, 10368, 6912, 3456],
        budget_list=[1.0, 0.75, 0.5, 0.25],
        router_std=0.1,
    )


@pytest.fixture
def scaler_config() -> FlextronConfig:
    """Config identical to `basic_config` but with the linear scaler schedule on."""
    return FlextronConfig(
        emb_int_list=[5120, 3840, 2560, 1280],
        mlp_int_list=[13824, 10368, 6912, 3456],
        budget_list=[1.0, 0.75, 0.5, 0.25],
        router_std=0.1,
        linear_scaler_start=1.0,
        linear_scaler_end=10.0,
        train_iters=1000,
    )


@pytest.fixture
def skipping_config() -> FlextronConfig:
    """Config with `add_skipping=True`."""
    return FlextronConfig(
        emb_int_list=[5120, 3840, 2560, 1280],
        mlp_int_list=[13824, 10368, 6912, 3456],
        budget_list=[1.0, 0.75, 0.5],
        add_skipping=True,
        layer_ranking_list=[0, 5, 10, 15, 20],
        num_layers=30,
    )


@pytest.fixture
def hetero_config() -> FlextronConfig:
    """Config with `flex_hetero_ffn=True`."""
    return FlextronConfig(
        emb_int_list=[5120, 3840],
        mlp_int_list=[13824, 10368, 6912],
        budget_list=[1.0, 0.5],
        flex_hetero_ffn=True,
        num_mlp_blocks=30,
    )


# ─── Construction ────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_construction_succeeds(self, basic_config):
        r = FlextronRouter(basic_config)
        assert r.config is basic_config

    def test_input_dim_matches_budget_list_length(self, basic_config):
        r = FlextronRouter(basic_config)
        assert r.input_dim == len(basic_config.budget_list)

    def test_n_dim_matches_router_inter_dim(self, basic_config):
        r = FlextronRouter(basic_config)
        assert r.n_dim == basic_config.router_inter_dim

    def test_budget_map_populated(self, basic_config):
        r = FlextronRouter(basic_config)
        assert set(r.budget_map.keys()) == set(basic_config.budget_list)
        # Descending order: index 0 = largest budget.
        for i, b in enumerate(basic_config.budget_list):
            assert r.budget_map[b].item() == i

    def test_dp_awareness_defaults(self, basic_config):
        r = FlextronRouter(basic_config)
        # In an uninitialized-dist context, rank=0 and size=1.
        assert r.dp_rank == 0
        assert r.dp_size == 1

    def test_fwd_pass_count_starts_at_zero(self, basic_config):
        r = FlextronRouter(basic_config)
        assert r.fwd_pass_count == 0

    def test_hard_sample_th_copied_from_config(self, basic_config):
        r = FlextronRouter(basic_config)
        assert r.hard_sample_th == basic_config.hard_sample_th

    def test_gate_mlp_shape(self, basic_config):
        r = FlextronRouter(basic_config)
        # 3-element Sequential: Linear(input_dim, n_dim) → LeakyReLU → Linear(n_dim, len(mlp_int_list))
        assert r.gate_mlp[0].weight.shape == (r.n_dim, r.input_dim)
        assert r.gate_mlp[2].weight.shape == (len(basic_config.mlp_int_list), r.n_dim)

    def test_gate_emb_shape(self, basic_config):
        r = FlextronRouter(basic_config)
        assert r.gate_emb[0].weight.shape == (r.n_dim, r.input_dim)
        assert r.gate_emb[2].weight.shape == (len(basic_config.emb_int_list), r.n_dim)

    def test_gates_have_no_bias(self, basic_config):
        r = FlextronRouter(basic_config)
        assert r.gate_mlp[0].bias is None
        assert r.gate_mlp[2].bias is None
        assert r.gate_emb[0].bias is None
        assert r.gate_emb[2].bias is None

    def test_weight_init_std_close_to_router_std(self, basic_config):
        r = FlextronRouter(basic_config)
        # 500-dim weight, N(0, 0.1) → sample std within ~7% of target 99% of the time
        assert abs(r.gate_emb[0].weight.std().item() - basic_config.router_std) < 0.02

    def test_scaler_none_by_default(self, basic_config):
        r = FlextronRouter(basic_config)
        assert r.scaler is None

    def test_scaler_built_when_configured(self, scaler_config):
        r = FlextronRouter(scaler_config)
        assert r.scaler is not None
        assert r.scaler.shape == (scaler_config.train_iters,)
        assert r.scaler[0].item() == pytest.approx(scaler_config.linear_scaler_start)
        assert r.scaler[-1].item() == pytest.approx(scaler_config.linear_scaler_end)


# ─── Forward at various budgets ──────────────────────────────────────────


class TestForward:
    def test_forward_at_trained_budget_returns_three_tuple(self, basic_config):
        r = FlextronRouter(basic_config)
        out = r.forward(budget=0.75, curr_iteration=0)
        assert len(out) == 3

    def test_mlp_out_shape_matches_choice_list(self, basic_config):
        r = FlextronRouter(basic_config)
        mlp_out, _, _ = r.forward(budget=0.75, curr_iteration=0)
        assert mlp_out[0].shape == (len(basic_config.mlp_int_list),)

    def test_emb_out_shape_matches_choice_list(self, basic_config):
        r = FlextronRouter(basic_config)
        _, _, emb_out = r.forward(budget=0.75, curr_iteration=0)
        assert emb_out[0].shape == (len(basic_config.emb_int_list),)

    def test_hard_choice_is_valid_member_of_int_list(self, basic_config):
        r = FlextronRouter(basic_config)
        mlp_out, _, emb_out = r.forward(budget=0.75, curr_iteration=0)
        assert mlp_out[1] in basic_config.mlp_int_list
        assert emb_out[1] in basic_config.emb_int_list

    def test_skip_output_none_when_add_skipping_false(self, basic_config):
        r = FlextronRouter(basic_config)
        _, skip_out, _ = r.forward(budget=0.75, curr_iteration=0)
        assert skip_out is None

    @pytest.mark.parametrize("budget", [1.0, 0.75, 0.5, 0.25])
    def test_forward_at_each_trained_budget(self, basic_config, budget):
        r = FlextronRouter(basic_config)
        out = r.forward(budget=budget, curr_iteration=0)
        assert out[0][0].shape == (len(basic_config.mlp_int_list),)
        assert out[2][0].shape == (len(basic_config.emb_int_list),)

    def test_forward_at_interpolated_budget(self, basic_config):
        r = FlextronRouter(basic_config)
        # 0.625 is not in [1.0, 0.75, 0.5, 0.25] → interpolation branch
        out = r.forward(budget=0.625, curr_iteration=100)
        assert out[0][1] in basic_config.mlp_int_list

    def test_forward_at_budget_1_when_1_not_in_list(self):
        """Fallback branch: `budget=1.0` requested but not in `budget_list`."""
        cfg = FlextronConfig(
            emb_int_list=[1024, 512],
            mlp_int_list=[4096, 2048],
            budget_list=[0.75, 0.5, 0.25],
        )
        r = FlextronRouter(cfg)
        # Should fall back to largest configured budget (0.75) without crashing.
        out = r.forward(budget=1.0, curr_iteration=0)
        assert out[0][1] in cfg.mlp_int_list

    def test_fwd_pass_count_increments_each_forward(self, basic_config):
        r = FlextronRouter(basic_config)
        n0 = r.fwd_pass_count
        _ = r.forward(budget=0.75, curr_iteration=0)
        _ = r.forward(budget=0.5, curr_iteration=1)
        _ = r.forward(budget=0.75, curr_iteration=2)
        assert r.fwd_pass_count == n0 + 3


# ─── DP-seeded Gumbel-softmax ────────────────────────────────────────────


class TestDPGumbelSoftmax:
    def test_cpu_rng_state_restored(self, basic_config):
        r = FlextronRouter(basic_config)
        torch.manual_seed(1234)
        x = torch.randn(4)  # advance RNG so `before` isn't the post-seed state
        before = torch.get_rng_state()
        _ = r._dp_gumbel_softmax(x, tau=0.5, hard=False, curr_iteration=100)
        after = torch.get_rng_state()
        assert torch.equal(before, after), "CPU RNG state must be restored"

    def test_same_seed_produces_identical_sample(self, basic_config):
        r = FlextronRouter(basic_config)
        logits = torch.randn(4)
        r.dp_rank = 0
        r.fwd_pass_count = 0
        s1 = r._dp_gumbel_softmax(logits.clone(), tau=1.0, hard=False, curr_iteration=42)
        r.dp_rank = 0
        r.fwd_pass_count = 0
        s2 = r._dp_gumbel_softmax(logits.clone(), tau=1.0, hard=False, curr_iteration=42)
        assert torch.equal(s1, s2)

    def test_different_dp_rank_produces_different_sample(self, basic_config):
        r = FlextronRouter(basic_config)
        logits = torch.randn(4)
        r.dp_rank = 0
        r.fwd_pass_count = 0
        s0 = r._dp_gumbel_softmax(logits.clone(), tau=1.0, hard=False, curr_iteration=42)
        r.dp_rank = 1
        r.fwd_pass_count = 0
        s1 = r._dp_gumbel_softmax(logits.clone(), tau=1.0, hard=False, curr_iteration=42)
        assert not torch.equal(s0, s1)

    def test_different_iteration_produces_different_sample(self, basic_config):
        r = FlextronRouter(basic_config)
        logits = torch.randn(4)
        r.dp_rank = 0
        r.fwd_pass_count = 0
        s_iter42 = r._dp_gumbel_softmax(logits.clone(), tau=1.0, hard=False, curr_iteration=42)
        s_iter43 = r._dp_gumbel_softmax(logits.clone(), tau=1.0, hard=False, curr_iteration=43)
        assert not torch.equal(s_iter42, s_iter43)

    def test_different_fwd_pass_count_produces_different_sample(self, basic_config):
        r = FlextronRouter(basic_config)
        logits = torch.randn(4)
        r.dp_rank = 0
        r.fwd_pass_count = 0
        s_a = r._dp_gumbel_softmax(logits.clone(), tau=1.0, hard=False, curr_iteration=42)
        r.fwd_pass_count = 1
        s_b = r._dp_gumbel_softmax(logits.clone(), tau=1.0, hard=False, curr_iteration=42)
        assert not torch.equal(s_a, s_b)


# ─── Tau schedule ────────────────────────────────────────────────────────


class TestTauDecay:
    def test_tau_at_iteration_zero_equals_tau_init(self, basic_config):
        r = FlextronRouter(basic_config)
        assert r.get_curr_tau(0).item() == pytest.approx(basic_config.tau_init)

    def test_tau_decreases_monotonically(self, basic_config):
        r = FlextronRouter(basic_config)
        taus = [r.get_curr_tau(i).item() for i in (0, 1000, 10000, 30000)]
        assert taus == sorted(taus, reverse=True)

    def test_tau_approaches_zero_at_large_iteration(self, basic_config):
        r = FlextronRouter(basic_config)
        # 0.9999 ** 30000 ≈ 0.0498
        assert r.get_curr_tau(30000).item() < 0.06


# ─── Gradient flow ───────────────────────────────────────────────────────


class TestGradientFlow:
    def test_gradient_flows_to_all_router_params(self, basic_config):
        """
        Uses a non-invariant loss.  A common pitfall: `sum(softmax(x)) = 1`
        identically, so `mlp_probs.sum() + emb_probs.sum()` is a constant
        w.r.t. the logits and produces zero gradient.  We use an
        arange-weighted sum instead.
        """
        r = FlextronRouter(basic_config)
        for p in r.parameters():
            p.grad = None

        mlp_out, _, emb_out = r.forward(budget=0.75, curr_iteration=0)
        w_mlp = torch.arange(len(basic_config.mlp_int_list), dtype=torch.float32)
        w_emb = torch.arange(len(basic_config.emb_int_list), dtype=torch.float32)
        loss = (mlp_out[0] * w_mlp).sum() + (emb_out[0] * w_emb).sum()
        loss.backward()

        for name, p in r.named_parameters():
            assert p.grad is not None, f"{name} missing gradient"
            assert p.grad.abs().sum() > 0, f"{name} received zero gradient"


# ─── add_skipping branch ─────────────────────────────────────────────────


class TestAddSkipping:
    def test_gate_skip_layer_built_when_add_skipping_true(self, skipping_config):
        r = FlextronRouter(skipping_config)
        assert hasattr(r, "gate_skip_layer")

    def test_gate_skip_layer_not_built_when_add_skipping_false(self, basic_config):
        r = FlextronRouter(basic_config)
        assert not hasattr(r, "gate_skip_layer")

    def test_skip_gate_output_dim(self, skipping_config):
        r = FlextronRouter(skipping_config)
        # Last-layer output dim = len(layer_ranking_list) + 1 (one extra "skip 0 layers" option)
        assert r.gate_skip_layer[2].weight.shape[0] == len(skipping_config.layer_ranking_list) + 1

    def test_skip_output_non_none_when_add_skipping_true(self, skipping_config):
        r = FlextronRouter(skipping_config)
        _, skip_out, _ = r.forward(budget=0.75, curr_iteration=0)
        assert skip_out is not None

    def test_skip_probs_shape(self, skipping_config):
        r = FlextronRouter(skipping_config)
        _, skip_out, _ = r.forward(budget=0.75, curr_iteration=0)
        soft_probs, _ = skip_out
        assert soft_probs.shape == (len(skipping_config.layer_ranking_list) + 1,)

    def test_skip_mask_shape_matches_num_layers(self, skipping_config):
        r = FlextronRouter(skipping_config)
        _, skip_out, _ = r.forward(budget=0.75, curr_iteration=0)
        _, skip_mask = skip_out
        assert skip_mask.shape == (skipping_config.num_layers,)

    def test_skip_mask_values_are_zero_or_one(self, skipping_config):
        r = FlextronRouter(skipping_config)
        _, skip_out, _ = r.forward(budget=0.5, curr_iteration=0)
        _, skip_mask = skip_out
        # Per Megatron's construction: exactly 0 or 1 per layer.
        assert set(skip_mask.unique().tolist()).issubset({0.0, 1.0})


# ─── flex_hetero_ffn branch ──────────────────────────────────────────────


class TestFlexHeteroFfn:
    def test_gate_mlp_output_dim_scales_with_num_blocks(self, hetero_config):
        r = FlextronRouter(hetero_config)
        expected_output_dim = len(hetero_config.mlp_int_list) * hetero_config.num_mlp_blocks
        assert r.gate_mlp[2].weight.shape[0] == expected_output_dim

    def test_hetero_mlp_choice_is_a_list(self, hetero_config):
        r = FlextronRouter(hetero_config)
        mlp_out, _, _ = r.forward(budget=0.5, curr_iteration=0)
        assert isinstance(mlp_out[1], list)

    def test_hetero_choice_list_has_num_mlp_blocks_entries(self, hetero_config):
        r = FlextronRouter(hetero_config)
        mlp_out, _, _ = r.forward(budget=0.5, curr_iteration=0)
        assert len(mlp_out[1]) == hetero_config.num_mlp_blocks

    def test_hetero_choices_are_all_valid(self, hetero_config):
        r = FlextronRouter(hetero_config)
        mlp_out, _, _ = r.forward(budget=0.5, curr_iteration=0)
        for c in mlp_out[1]:
            assert c in hetero_config.mlp_int_list


# ─── Scaler-off default config (Megatron NameError we fixed) ────────────


class TestScalerOffPath:
    """The default config (`scaler=None`, `normalize_router_logits=False`) used
    to raise `NameError: local variable 'scale' referenced before assignment`
    in Megatron's `mlp_forward`.  We default `scale = 1.0` so the multiplication
    is a no-op.  See PLAN.md "Known Megatron quirks"."""

    def test_default_config_forward_does_not_raise(self, basic_config):
        r = FlextronRouter(basic_config)
        r.forward(budget=0.75, curr_iteration=0)  # must not raise NameError

    def test_hetero_forward_with_scaler_off_does_not_raise(self, hetero_config):
        r = FlextronRouter(hetero_config)
        r.forward(budget=0.5, curr_iteration=0)  # must not raise NameError

    def test_skipping_forward_with_scaler_off_does_not_raise(self, skipping_config):
        r = FlextronRouter(skipping_config)
        r.forward(budget=0.75, curr_iteration=0)  # must not raise NameError
