# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``fastgen.utils.guidance``.

Covers:

* Schema construction and ``GuidanceConfig`` defaults.
* ``resolve_piecewise`` over the (value, boundary) cross product.
* ``per_modality(resolve_piecewise, ...)`` dispatch over per-modality scales.
* ``predict_with_guidance`` for every shared term, every cache-key
  short-circuit, the mix-term contract, the algebraic identity vs
  mean-flow's 3-way blend, and the rejection of providers that are not
  networks.
* Network terms of a ``GuidanceConfig`` subclass, and LTX-2's own recipe
  for its STG and cross-modal terms and its rescale on x0.

These tests do **not** require any real network or model weights — small
``FunctionNet`` stub networks simulate the network forward.
"""

from __future__ import annotations

import attrs
import pytest
import torch

import fastgen.utils.tensor as tensor_module
from fastgen.networks.noise_schedule import RFNoiseSchedule
from fastgen.utils.guidance import (
    GuidanceConfig,
    GuidanceTerm,
    apply_cfg,
    apply_guidance,
    apply_mix,
    apply_rescale,
    cfg_rescale,
    get_guidance_context,
    is_active,
    predict_with_guidance,
    resolve_piecewise,
)
from fastgen.utils.tree_utils import per_modality
from tests.model_utils import FunctionNet


def test_guidance_composes_half_precision_branches_without_overflow():
    net = FunctionNet(lambda x_t, t, condition, **kwargs: torch.full_like(x_t, 6e4 if condition == "pos" else -6e4))
    x_t, t = torch.zeros(1, 2, dtype=torch.float16), torch.tensor([0.5], dtype=torch.float64)
    g = GuidanceConfig()
    g.cfg.scale = 0.0
    result = predict_with_guidance(net, x_t, t, guidance_cfg=g, condition="pos", neg_condition="neg")
    assert result.dtype == x_t.dtype
    torch.testing.assert_close(result, torch.full_like(x_t, -60000.0))


def test_guidance_keeps_a_working_dtype_per_modality(monkeypatch):
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)
    net = FunctionNet(
        lambda x_t, t, condition, **kwargs: {key: value + 1 for key, value in x_t.items()},
        modalities=("video", "audio"),
    )
    x_t = {"video": torch.zeros(2, 3, dtype=torch.bfloat16), "audio": torch.zeros(2, 2, dtype=torch.float64)}
    t = torch.tensor([0.25, 0.75], dtype=net.noise_scheduler.t_precision)
    neg = {key: torch.full_like(value, -1.0, dtype=torch.float64) for key, value in x_t.items()}
    g = GuidanceConfig()
    g.cfg.scale = 3.0
    working = {"video": torch.float32, "audio": torch.float64}
    narrow = {"video": torch.bfloat16, "audio": torch.float32}
    wide = {"video": torch.float64, "audio": torch.float32}

    ctx = get_guidance_context(net, x_t, t, "pos", g, "neg", precomputed={"neg": neg})
    assert ctx.dtype == working
    assert per_modality(lambda x: x.dtype, ctx.cache["neg"]) == working
    for output_dtype, working_dtype, result_dtype in (
        ("same", working, {"video": torch.bfloat16, "audio": torch.float64}),
        (None, working, working),
        (narrow, working, narrow),
        # A requested dtype widens its modality's working dtype but never narrows it.
        (wide, {"video": torch.float64, "audio": torch.float64}, wide),
    ):
        result, cache = predict_with_guidance(
            net, x_t, t, "pos", g, "neg", precomputed={"neg": neg}, return_cache=True, output_dtype=output_dtype
        )
        assert per_modality(lambda x: x.dtype, cache["pos"]) == working_dtype
        # pos + (3 - 1) * (pos - neg) with pos = 1 and the precomputed neg = -1.
        expected = per_modality(lambda x, d: torch.full_like(x, 5.0, dtype=d), x_t, result_dtype)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_guidance_rejects_a_branch_outside_its_modality_working_dtype(monkeypatch):
    monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)

    class DemotesAudio(FunctionNet):
        def forward(self, x_t, t, **kwargs):
            prediction = super().forward(x_t, t, **kwargs)
            return {**prediction, "audio": prediction["audio"].float()}

    net = DemotesAudio(lambda x_t, t, condition, **kwargs: x_t, modalities=("video", "audio"))
    x_t = {"video": torch.zeros(2, 3, dtype=torch.bfloat16), "audio": torch.zeros(2, 2, dtype=torch.float64)}
    t = torch.tensor([0.25, 0.75], dtype=net.noise_scheduler.t_precision)
    # The float32 audio branch has the video's working dtype, not its own float64 one.
    with pytest.raises(AssertionError, match=r"pos\.dtype \(torch\.float32\) must be torch\.float64"):
        predict_with_guidance(net, x_t, t, "pos")


@pytest.mark.parametrize(
    "multimodal, output_dtype, match",
    [
        (True, "float32", "'same' or None"),
        (False, {"video": torch.float32}, "needs multimodal data"),
        (True, {"video": torch.float32}, "per modality of"),
        (True, {"video": torch.float32, "audio": None}, "per modality of"),
    ],
    ids=["unknown-string", "mapping-for-a-tensor", "missing-modality", "none-value"],
)
def test_guidance_rejects_an_invalid_output_dtype_before_any_forward(multimodal, output_dtype, match):
    def native(*args, **kwargs):
        raise AssertionError("no forward may run")

    net = FunctionNet(native, modalities=("video", "audio") if multimodal else None)
    x_t = {"video": torch.zeros(2, 3), "audio": torch.zeros(2, 2)} if multimodal else torch.zeros(2, 3)
    t = torch.tensor([0.25, 0.75], dtype=net.noise_scheduler.t_precision)
    with pytest.raises(ValueError, match=match):
        predict_with_guidance(net, x_t, t, "pos", output_dtype=output_dtype)


def test_cfg_rescale_rejects_a_singleton_std_reduction():
    guided = torch.tensor([[2.0], [4.0]])
    cond = torch.tensor([[1.0], [3.0]])
    with pytest.raises(ValueError, match="more than one element"):
        cfg_rescale(guided, cond, rescale=0.5, rescale_dims=(-1,))


def test_cfg_rescale_floors_a_constant_prediction():
    guided = torch.full((1, 4), 2.0)
    cond = torch.tensor([[-2.0, 2.0, -2.0, 2.0]])
    result = cfg_rescale(guided, cond, rescale=0.5, eps=1.0)
    torch.testing.assert_close(result, torch.full_like(guided, 3.0))


# ───────────────── Stub network and fixtures ─────────────────────────────────


class StubNet(FunctionNet):
    """Deterministic stand-in network.

    Each forward predicts ``base + offsets[key]`` where ``key`` is derived
    from the kwargs (used to differentiate cond/uncond/perturbed calls).
    Records every invocation for cache-hit / cache-miss counting.
    """

    def __init__(self, base: torch.Tensor):
        super().__init__()
        self.base = base
        self.calls: list[dict] = []

    def native(self, x_t, t, condition=None, **kw):
        self.calls.append({"condition": condition, **kw})
        if isinstance(condition, str):
            tag = condition
        elif condition is None:
            tag = "uncond"
        else:
            tag = "tensor_cond"
        offset = {
            "pos": 1.0,
            "neg": -1.0,
            "uncond": -2.0,
            "tensor_cond": 0.5,
        }.get(tag, 0.0)
        out = self.base + offset
        if kw.get("skip_layers") is not None:
            out = out + 0.1
        if kw.get("skip_self_attn_layers") is not None:
            out = out + 0.2
        if kw.get("skip_cross_modal_attn"):
            out = out + 0.3
        return out


@pytest.fixture
def t_batch():
    """Timesteps in the stub networks' schedule precision."""
    return torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)


@pytest.fixture
def x_t():
    return torch.zeros(3, 4)


@pytest.fixture
def x_t_dict():
    return {"video": torch.zeros(3, 4), "audio": torch.zeros(3, 2)}


@pytest.fixture
def stub():
    return StubNet(base=torch.zeros(3, 4))


@attrs.define
class PerturbTerm(GuidanceTerm):
    """A network term with one forward of its own, cached under its field name."""


@attrs.define
class NetworkGuidance(GuidanceConfig):
    """A ``GuidanceConfig`` subclass with a network term, cached as the ``"perturb"`` branch."""

    perturb: PerturbTerm = attrs.field(factory=PerturbTerm)

    def pred_keys(self):
        return [*super().pred_keys(), "perturb"]

    def active_pred_keys(self):
        return super().active_pred_keys() + (["perturb"] if self.perturb.is_ever_active() else [])

    def validate(self, supported_terms, supports_cfg_skip_layers=False, network_name="network"):
        super().validate(supported_terms, supports_cfg_skip_layers, network_name)
        if "perturb" not in supported_terms and self.perturb.scale is not None:
            raise NotImplementedError(f"{network_name} does not support guidance_cfg.perturb")


@attrs.define
class PairTerm(GuidanceTerm):
    """A network term with two forwards of its own, the second with the negative condition."""


@attrs.define
class ReuseTerm(GuidanceTerm):
    """A network term without a forward of its own: it reuses the CFG negative."""


@attrs.define
class BranchingGuidance(GuidanceConfig):
    """A ``GuidanceConfig`` subclass whose network terms need two forwards of their own and none."""

    pair: PairTerm = attrs.field(factory=PairTerm)
    reuse: ReuseTerm = attrs.field(factory=ReuseTerm)

    def pred_keys(self):
        return [*super().pred_keys(), "pair", "pair_neg"]

    def active_pred_keys(self):
        keys = super().active_pred_keys()
        if self.pair.is_ever_active():
            keys += ["pair", "pair_neg"]
        if self.reuse.is_ever_active() and "neg" not in keys:
            keys.append("neg")
        return keys

    def condition_for_pred_key(self, key, condition, neg_condition):
        if key == "pair_neg":
            return neg_condition
        return super().condition_for_pred_key(key, condition, neg_condition)


# ═════════════════════════════════════════════════════════════════════════
# Guidance config defaults
# ═════════════════════════════════════════════════════════════════════════


class TestGuidanceConfig:
    def test_default_all_disabled(self):
        g = GuidanceConfig()
        assert g.cfg.scale is None
        assert g.mix.scale is None
        assert g.rescale is None

    def test_default_terms_independent_instances(self):
        g1 = GuidanceConfig()
        g2 = GuidanceConfig()
        g1.cfg.scale = 5.0
        assert g2.cfg.scale is None  # factory'd, not shared

    def test_default_returns_x_cond(self, stub, x_t, t_batch):
        # Disabled config → result == cond prediction.
        out = predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=GuidanceConfig(),
            condition="pos",
        )
        expected = stub.base + 1.0  # cond offset
        assert torch.allclose(out, expected)
        assert len(stub.calls) == 1  # only the cond forward

    def test_none_guidance_is_empty_config(self, stub, x_t, t_batch):
        # ``guidance_cfg=None`` is equivalent to ``guidance_cfg=GuidanceConfig()``:
        # one cond forward, returned unchanged.  Lets callers (DMD2, CM,
        # sCM, f_distill, etc.) skip the ``if guidance_cfg is None`` boilerplate.
        out = predict_with_guidance(stub, x_t, t_batch, guidance_cfg=None, condition="pos")
        expected = stub.base + 1.0
        assert torch.allclose(out, expected)
        assert len(stub.calls) == 1


# ═════════════════════════════════════════════════════════════════════════
# resolve_piecewise
# ═════════════════════════════════════════════════════════════════════════


class TestResolvePiecewise:
    def test_none(self, t_batch):
        assert resolve_piecewise(None, t_batch, None) is None

    def test_scalar_no_boundary(self, t_batch):
        assert resolve_piecewise(3.0, t_batch, None) == 3.0

    def test_scalar_with_boundary_is_user_error(self, t_batch):
        # Scalar + boundary makes no sense; we don't pretend it does.
        with pytest.raises(TypeError):
            resolve_piecewise(3.0, t_batch, 0.5)

    def test_sequence_no_boundary_passes_through(self, t_batch):
        # No boundary = always-on, return as-is. Natural form for
        # always-on skip_layers=[5, 10] etc.
        assert resolve_piecewise([5, 10], t_batch, None) == [5, 10]
        assert resolve_piecewise([1.0, 2.0], t_batch, None) == [1.0, 2.0]

    def test_sequence_length_mismatch(self, t_batch):
        with pytest.raises(ValueError, match="len\\(value\\) must"):
            resolve_piecewise([1.0, 2.0, 3.0], t_batch, 0.5)

    def test_numeric_scalar_boundary(self, t_batch):
        # value=[3.0, 7.0], boundary=0.5 → t<0.5 → 3.0, t>=0.5 → 7.0.
        # bucketize(right=True): t < boundary → bucket 0; t >= boundary → bucket 1.
        out = resolve_piecewise([3.0, 7.0], t_batch, 0.5)
        assert torch.allclose(out, t_batch.new_tensor([3.0, 7.0, 7.0]))

    def test_numeric_sequence_boundary(self, t_batch):
        # 3 regions: t<0.3 → 1.0, 0.3<=t<0.7 → 2.0, t>=0.7 → 3.0.
        out = resolve_piecewise([1.0, 2.0, 3.0], t_batch, [0.3, 0.7])
        assert torch.allclose(out, t_batch.new_tensor([1.0, 2.0, 3.0]))

    def test_sequence_pieces_collapse_via_first_t(self):
        # Sequence schedules return the single piece chosen by t.
        # Batch-uniform t=[0.6, ...] → bucket 1 → second entry.
        t = torch.tensor([0.6, 0.6, 0.6])
        out = resolve_piecewise([None, [5, 10]], t, 0.3, sequence=True)
        assert out == [5, 10]

    def test_sequence_pieces_off_region(self):
        t = torch.tensor([0.1, 0.1, 0.1])
        out = resolve_piecewise([None, [5, 10]], t, 0.3, sequence=True)
        assert out is None

    def test_sequence_pieces_per_sample_same_bucket_ok(self):
        # Per-sample t that all happen to fall in the same bucket is fine
        # — no actual disagreement on which value to return.
        t = torch.tensor([0.4, 0.5, 0.6])
        out = resolve_piecewise([None, [5, 10]], t, 0.3, sequence=True)
        assert out == [5, 10]

    def test_sequence_pieces_cross_bucket_raises(self):
        # Per-sample t that span multiple buckets raise.
        t = torch.tensor([0.1, 0.6, 0.6])
        with pytest.raises(ValueError, match="same bucket"):
            resolve_piecewise([None, [5, 10]], t, 0.3, sequence=True)

    @pytest.mark.parametrize("value", [[5, 10], [None, 5]])
    def test_sequence_pieces_reject_numbers(self, t_batch, value):
        # Without sequence=True, [5, 10] would resolve as a numeric schedule.
        with pytest.raises(ValueError, match="a sequence or None per piece"):
            resolve_piecewise(value, t_batch, 0.5, sequence=True)


# ═════════════════════════════════════════════════════════════════════════
# per_modality(resolve_piecewise, ...) dispatch over guidance scales
# ═════════════════════════════════════════════════════════════════════════


class TestResolveGuidanceScale:
    """``per_modality(resolve_piecewise, ...)`` dispatches resolve_piecewise
    per-modality for dict-shaped scales and broadcasts otherwise."""

    def test_scalar(self, t_batch):
        assert per_modality(resolve_piecewise, 3.0, t_batch, None) == 3.0

    def test_sequence_with_boundary(self, t_batch):
        out = per_modality(resolve_piecewise, [3.0, 7.0], t_batch, 0.5)
        assert torch.allclose(out, t_batch.new_tensor([3.0, 7.0, 7.0]))

    def test_per_modality_dict_scalar(self, t_batch):
        out = per_modality(resolve_piecewise, {"video": 3.0, "audio": 5.0}, t_batch, None)
        assert out == {"video": 3.0, "audio": 5.0}

    def test_per_modality_dict_sequence(self, t_batch):
        out = per_modality(
            resolve_piecewise,
            {"video": [3.0, 7.0], "audio": 5.0},
            t_batch,
            {"video": 0.5, "audio": None},
        )
        assert torch.allclose(out["video"], t_batch.new_tensor([3.0, 7.0, 7.0]))
        assert out["audio"] == 5.0

    def test_per_modality_scalar_boundary_broadcast(self, t_batch):
        out = per_modality(resolve_piecewise, {"video": [3.0, 7.0], "audio": [1.0, 9.0]}, t_batch, 0.5)
        assert torch.allclose(out["video"], t_batch.new_tensor([3.0, 7.0, 7.0]))
        assert torch.allclose(out["audio"], t_batch.new_tensor([1.0, 9.0, 9.0]))

    def test_none(self, t_batch):
        assert per_modality(resolve_piecewise, None, t_batch, None) is None


# ═════════════════════════════════════════════════════════════════════════
# apply_guidance dtype + shape handling for per-sample (Tensor[B]) scales
# ═════════════════════════════════════════════════════════════════════════


class TestApplyGuidancePerSampleScale:
    """When boundary scheduling is used, ``resolve_piecewise`` returns a
    ``Tensor[B]`` scale.  ``apply_guidance`` must (a) broadcast it to the
    prediction rank and (b) cast it to the prediction dtype, so the result
    keeps that dtype."""

    def test_per_sample_scale_broadcasts_to_pred_rank(self):
        # Predictions have rank 4 (B, C, H, W); scale is Tensor[B].
        state = torch.zeros(3, 4, 5, 6)
        cond = torch.full_like(state, 1.0)
        uncond = torch.full_like(state, -1.0)
        scale = torch.tensor([3.0, 5.0, 7.0])  # Tensor[B]

        out = apply_guidance(state, cond, uncond, scale)

        # state + (scale - 1) * (cond - uncond) = 0 + (s-1)*(2.0)
        # Per-sample: s=3→4.0, s=5→8.0, s=7→12.0.
        expected_per_sample = torch.tensor([4.0, 8.0, 12.0])
        for b in range(3):
            assert torch.allclose(out[b], torch.full_like(out[b], expected_per_sample[b].item()))

    def test_per_sample_scale_dtype_does_not_promote(self):
        # Reproduces the production scenario: t in fp64, predictions in bf16.
        state = torch.zeros(3, 4, dtype=torch.bfloat16)
        cond = torch.full_like(state, 1.0)
        uncond = torch.full_like(state, -1.0)
        scale_fp64 = torch.tensor([3.0, 5.0, 7.0], dtype=torch.float64)

        out = apply_guidance(state, cond, uncond, scale_fp64)

        assert out.dtype == torch.bfloat16, (
            f"apply_guidance must preserve the prediction dtype (bfloat16), "
            f"got {out.dtype} — fp64 scale silently promoted the result"
        )

    def test_predict_with_guidance_piecewise_dtype(self, x_t):
        # End-to-end: predict_with_guidance with a piecewise scale and
        # an fp64 t against a stub that returns bf16 predictions.
        net = FunctionNet(lambda x_t, t, condition, **kw: torch.full_like(x_t, 1.0 if condition == "pos" else -1.0))
        x_t_bf16 = torch.zeros(3, 4, dtype=torch.bfloat16)
        t_fp64 = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)

        g = GuidanceConfig()
        g.cfg.scale = [3.0, 7.0]
        g.cfg.boundary = 0.5  # per-sample piecewise scale
        out = predict_with_guidance(
            net,
            x_t_bf16,
            t_fp64,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
        )
        assert out.dtype == torch.bfloat16

    def test_predict_with_guidance_rank_matched_t(self):
        # User pre-expands t to rank-4 (common: noise-schedule code keeps t
        # as ``[B, 1, 1, 1]`` so it broadcasts directly against ``[B, C, H, W]``
        # state).  ``resolve_piecewise`` must preserve t's shape so the
        # resulting ``Tensor[B, 1, 1, 1]`` scale lands in ``apply_guidance``
        # already rank-aligned (``expand_like`` is then a no-op).
        net = FunctionNet(lambda x_t, t, condition, **kw: torch.full_like(x_t, 1.0 if condition == "pos" else -1.0))
        x_t = torch.zeros(3, 4, 5, 6)
        # t in (0.2, 0.5, 0.8); boundary=0.5 ⇒ buckets (0, 1, 1) ⇒ scales
        # (3.0, 7.0, 7.0).  ``predict_with_guidance`` initialises
        # ``x_pred = x_cond = pos`` (= 1.0 from the stub), so the result is
        # ``pos + (s-1)*(pos - neg) = 1 + (s-1)*2``.
        t_4d = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64).view(3, 1, 1, 1)

        g = GuidanceConfig()
        g.cfg.scale = [3.0, 7.0]
        g.cfg.boundary = 0.5
        out = predict_with_guidance(
            net,
            x_t,
            t_4d,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
        )

        assert out.shape == x_t.shape
        expected_per_sample = torch.tensor([5.0, 13.0, 13.0])
        for b in range(3):
            assert torch.allclose(out[b], torch.full_like(out[b], expected_per_sample[b].item()))


# ═════════════════════════════════════════════════════════════════════════
# predict_with_guidance — single-term branches
# ═════════════════════════════════════════════════════════════════════════


class TestPredictWithGuidanceCFG:
    def test_cfg_matches_apply_guidance(self, stub, x_t, t_batch):
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        out = predict_with_guidance(stub, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")
        x_cond = stub.base + 1.0  # pos
        x_uncond = stub.base + (-1.0)  # neg
        expected = apply_guidance(x_cond, x_cond, x_uncond, 3.0)
        assert torch.allclose(out, expected)

    def test_cfg_precomputed_skips_forward(self, stub, x_t, t_batch):
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        prebuilt_cond = torch.full_like(x_t, 11.0)
        prebuilt_cfg = torch.full_like(x_t, -7.0)
        out = predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
            precomputed={"pos": prebuilt_cond, "neg": prebuilt_cfg},
        )
        assert len(stub.calls) == 0  # both cached
        expected = apply_guidance(prebuilt_cond, prebuilt_cond, prebuilt_cfg, 3.0)
        assert torch.allclose(out, expected)

    def test_cfg_skip_layers_propagates(self, stub, x_t, t_batch):
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        g.cfg.skip_layers = [5, 10]
        predict_with_guidance(stub, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")
        # Second call (cfg uncond) carries skip_layers.
        cfg_call = next(c for c in stub.calls if c["condition"] == "neg")
        assert cfg_call["skip_layers"] == [5, 10]

    def test_cfg_flat_skip_layers_with_boundary_raises(self, stub, x_t, t_batch):
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        g.cfg.skip_layers, g.cfg.skip_layers_boundary = [5, 10], 0.5
        with pytest.raises(ValueError, match="a sequence or None per piece"):
            predict_with_guidance(stub, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")


class TestPredictWithGuidanceNetworkTerms:
    def test_standard_recipe_rejects_network_terms_before_forwards(self, stub, x_t, t_batch):
        g = NetworkGuidance()
        # Reject even when the current time window would make the term inactive.
        g.perturb.scale, g.perturb.boundary = [1.0, 3.0], 0.9
        with pytest.raises(NotImplementedError, match="StubNet does not support guidance_cfg.perturb"):
            predict_with_guidance(
                stub, x_t, t_batch, guidance_cfg=g, condition="pos", precomputed={"perturb": torch.zeros_like(x_t)}
            )
        assert stub.calls == []

    def test_network_term_names_are_branch_keys(self, stub, x_t, t_batch):
        # A branch already in the working dtype enters the cache as it is.
        perturbed = torch.zeros_like(x_t, dtype=torch.float64)
        _, cache = predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=NetworkGuidance(),
            condition="pos",
            precomputed={"perturb": perturbed},
            return_cache=True,
        )
        assert set(cache) == {"pos", "perturb"} and cache["perturb"] is perturbed
        with pytest.raises(ValueError, match="unknown guidance branch keys"):
            predict_with_guidance(stub, x_t, t_batch, condition="pos", precomputed={"perturb": perturbed})


class TestPredictWithGuidanceContexts:
    """Per-key kwarg overrides via the ``contexts`` parameter."""

    def test_contexts_route_to_right_forward(self, stub, x_t, t_batch):
        # Every forward picks up the per-key extras; pos and neg get
        # different ``cache_tag`` values.
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        out = predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
            contexts={"pos": {"cache_tag": "pos"}, "neg": {"cache_tag": "neg"}},
        )
        cond_call = next(c for c in stub.calls if c["condition"] == "pos")
        cfg_call = next(c for c in stub.calls if c["condition"] == "neg")
        assert cond_call["cache_tag"] == "pos"
        assert cfg_call["cache_tag"] == "neg"
        # Numerical result unchanged by contexts (extras are passthrough).
        x_cond = stub.base + 1.0
        x_uncond = stub.base - 1.0
        expected = apply_guidance(x_cond, x_cond, x_uncond, 3.0)
        assert torch.allclose(out, expected)

    def test_contexts_partial(self, stub, x_t, t_batch):
        # Only "pos" supplied; neg call gets no extras.
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
            contexts={"pos": {"cache_tag": "pos"}},
        )
        cond_call = next(c for c in stub.calls if c["condition"] == "pos")
        cfg_call = next(c for c in stub.calls if c["condition"] == "neg")
        assert cond_call["cache_tag"] == "pos"
        assert "cache_tag" not in cfg_call

    def test_contexts_skipped_on_cache_hit(self, stub, x_t, t_batch):
        # When a precomputed entry satisfies a key, the corresponding
        # contexts entry is silently ignored (no forward to apply it to).
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        precomp_uncond = torch.full_like(x_t, -1.0)
        predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
            precomputed={"neg": precomp_uncond},
            contexts={"neg": {"cache_tag": "neg"}},
        )
        # Only the cond forward ran; no neg forward → no cache_tag observed.
        assert all(c.get("cache_tag") is None for c in stub.calls)

    def test_contexts_apply_to_mix(self, stub, x_t, t_batch):
        # Per-key contexts also apply to non-CFG terms.
        g = GuidanceConfig()
        g.mix.scale = 2.0
        predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
            contexts={"pos_mix": {"cache_tag": "pos_mix"}},
        )
        tagged = [c for c in stub.calls if c.get("cache_tag") == "pos_mix"]
        assert len(tagged) == 1 and tagged[0]["condition"] == "pos"


# ═════════════════════════════════════════════════════════════════════════
# LTX-2: default recipe (Cases 1a/1b/2) and its own guidance composition
# ═════════════════════════════════════════════════════════════════════════


def _import_ltx2():
    """``(LTX2, LTX2GuidanceConfig)``; skips where the LTX-2 module cannot be imported."""
    try:
        from fastgen.networks.LTX2.network import LTX2, LTX2GuidanceConfig
    except (ImportError, OSError) as e:
        pytest.skip(f"LTX-2 module not importable in this env: {e}")
    return LTX2, LTX2GuidanceConfig


def _ltx2_recipe(forward, modalities=None):
    """A T2V ``FunctionNet`` over *forward* with ``LTX2``'s guidance recipe."""
    LTX2, _ = _import_ltx2()

    class Recipe(FunctionNet):
        predict_with_guidance = LTX2.predict_with_guidance
        _rescale_denoised = LTX2._rescale_denoised

    return Recipe(forward, modalities)


def _flow_forward(x_t, t, condition, **kwargs):
    """The velocity ``0.5 * x_t + condition``; ``FunctionNet`` converts it to the requested ``fwd_pred_type``."""
    return 0.5 * x_t + condition


class TestLTX2DefaultGuidance:
    """End-to-end (CPU-level) verification that
    ``LTX2.default_guidance_cfg(modalities=...)`` threads correctly through
    ``predict_with_guidance`` for every supported sample-time input shape:

    * **Case 1a** — multimodal-capable network with full dict input
      (``{"video": ..., "audio": ...}``): config has dict CFG scale, modality
      cross-guidance enabled.
    * **Case 1b** — multimodal-capable network with single-key dict input
      (``{"video": ...}``): ``active_modalities`` collapses to a single
      modality, so ``default_guidance_cfg`` produces a scalar CFG scale with
      cross-modal guidance disabled (modality.scale is ``None``); inputs
      remain dict-shaped through ``per_modality`` dispatch.
    * **Case 2** — unimodal-only network with bare-Tensor input: same scalar
      CFG scale as 1b, but inputs are bare Tensors throughout.

    Companion to the GPU smoke tests in ``test_network_ltx2`` /
    ``test_network_ltx2_unimodal`` — runs without LTX-2 weights.
    """

    @staticmethod
    def _make_dict_stub():
        """Stand-in network that accepts dict or Tensor x_t and mirrors shape."""

        class _Stub:
            def __init__(self):
                self.calls: list[dict] = []

            def __call__(self, x_t, t, **kw):
                self.calls.append(kw)
                cond = kw.get("condition")
                offset = {"pos": 1.0, "neg": -1.0}.get(cond, 0.0)
                bump = 0.0
                if kw.get("skip_self_attn_layers") is not None:
                    bump += 0.2
                if kw.get("skip_cross_modal_attn"):
                    bump += 0.3
                if isinstance(x_t, dict):
                    return {k: torch.zeros_like(v) + offset + bump for k, v in x_t.items()}
                return torch.zeros_like(x_t) + offset + bump

        return _Stub()

    # ---- shape of the GuidanceConfig itself ------------------------------

    def test_config_shape_multimodal(self):
        """Case 1a: ``modalities=("video","audio")`` → dict CFG scale, modality on."""
        LTX2, LTX2GuidanceConfig = _import_ltx2()
        g = LTX2.default_guidance_cfg(modalities=("video", "audio"))
        assert isinstance(g, LTX2GuidanceConfig)
        assert isinstance(g.cfg.scale, dict) and set(g.cfg.scale.keys()) == {"video", "audio"}
        assert g.modality.scale == 3.0  # cross-modal active
        assert g.stg.scale == 2.0
        assert g.rescale == 0.7

    def test_config_shape_video_only(self):
        """Cases 1b / 2: ``modalities=("video",)`` → scalar CFG scale, modality off."""
        LTX2, _ = _import_ltx2()
        g = LTX2.default_guidance_cfg(modalities=("video",))
        assert g.cfg.scale == 3.0  # scalar (looked up from {"video": 3.0, "audio": 7.0})
        assert g.modality.scale is None  # cross-modal disabled in unimodal mode
        assert g.stg.scale == 2.0
        assert g.rescale == 0.7

    def test_config_shape_audio_only(self):
        """Symmetry: ``modalities=("audio",)`` → audio CFG scale (7.0), modality off."""
        LTX2, _ = _import_ltx2()
        g = LTX2.default_guidance_cfg(modalities=("audio",))
        assert g.cfg.scale == 7.0
        assert g.modality.scale is None

    def test_unknown_release_is_rejected(self):
        LTX2, _ = _import_ltx2()
        with pytest.raises(ValueError, match="Unsupported LTX-2 version"):
            LTX2.default_guidance_cfg(version="future")

    # ---- end-to-end through predict_with_guidance ------------------------

    def test_case_1a_multimodal_full_dict(self, t_batch):
        """Full-dict input: cfg(dict) + stg(scalar) + modality(scalar) + rescale,
        all per-modality.  Output mirrors input shape."""
        LTX2, _ = _import_ltx2()
        x_t = {"video": torch.zeros(3, 4), "audio": torch.zeros(3, 2)}
        g = LTX2.default_guidance_cfg(modalities=("video", "audio"))
        net = _ltx2_recipe(self._make_dict_stub(), modalities=("video", "audio"))
        out = predict_with_guidance(net, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")
        assert isinstance(out, dict) and set(out.keys()) == {"video", "audio"}
        assert out["video"].shape == x_t["video"].shape
        assert out["audio"].shape == x_t["audio"].shape

    def test_case_1b_multimodal_single_key_dict(self, t_batch):
        """Single-key dict input on a multimodal-capable network (active_modalities
        collapses to one).  `default_guidance_cfg(("video",))` is the recipe;
        per-modality dispatch only iterates over present keys."""
        LTX2, _ = _import_ltx2()
        x_t = {"video": torch.zeros(3, 4)}
        g = LTX2.default_guidance_cfg(modalities=("video",))
        stub = self._make_dict_stub()
        net = _ltx2_recipe(stub, modalities=("video", "audio"))
        out = predict_with_guidance(net, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")
        assert isinstance(out, dict) and set(out.keys()) == {"video"}
        assert out["video"].shape == x_t["video"].shape
        # No modality (cross-modal) forward — only cond + cfg-uncond + stg.
        assert all(not c.get("skip_cross_modal_attn") for c in stub.calls)

    def test_case_2_unimodal_bare_tensor(self, t_batch):
        """Bare-Tensor input on a unimodal-only network — no per_modality
        dispatch needed (`per_modality` short-circuits on Tensor primary)."""
        LTX2, _ = _import_ltx2()
        x_t = torch.zeros(3, 4)
        g = LTX2.default_guidance_cfg(modalities=("video",))
        net = _ltx2_recipe(self._make_dict_stub())
        out = predict_with_guidance(net, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")
        assert isinstance(out, torch.Tensor)
        assert out.shape == x_t.shape


class TestLTX2GuidanceRecipe:
    """``LTX2.predict_with_guidance``: STG and cross-modal terms, and the rescale on x0 for every caller."""

    def test_active_stg_requires_perturbation_layers(self, stub, x_t, t_batch):
        _, LTX2GuidanceConfig = _import_ltx2()
        g = LTX2GuidanceConfig()
        g.stg.scale = 2.0
        with pytest.raises(ValueError, match="active STG requires"):
            predict_with_guidance(_ltx2_recipe(stub), x_t, t_batch, guidance_cfg=g, condition="pos")

    def test_stg(self, stub, x_t, t_batch):
        _, LTX2GuidanceConfig = _import_ltx2()
        g = LTX2GuidanceConfig()
        g.stg.scale = 2.0
        g.stg.skip_self_attn_layers = [3, 7]
        out = predict_with_guidance(_ltx2_recipe(stub), x_t, t_batch, guidance_cfg=g, condition="pos")
        x_cond = stub.base + 1.0
        x_perturbed = stub.base + 1.0 + 0.2  # pos offset + STG offset
        expected = apply_guidance(x_cond, x_cond, x_perturbed, 2.0)
        assert torch.allclose(out, expected)
        stg_call = next(c for c in stub.calls if c.get("skip_self_attn_layers") is not None)
        assert stg_call["skip_self_attn_layers"] == [3, 7]

    def test_stg_flat_layers_with_boundary_raises(self, stub, x_t, t_batch):
        _, LTX2GuidanceConfig = _import_ltx2()
        g = LTX2GuidanceConfig()
        g.stg.scale = 2.0
        g.stg.skip_self_attn_layers, g.stg.skip_self_attn_layers_boundary = [3, 7], 0.5
        with pytest.raises(ValueError, match="a sequence or None per piece"):
            predict_with_guidance(_ltx2_recipe(stub), x_t, t_batch, guidance_cfg=g, condition="pos")

    def test_contexts_apply_to_stg(self, stub, x_t, t_batch):
        _, LTX2GuidanceConfig = _import_ltx2()
        g = LTX2GuidanceConfig()
        g.stg.scale = 2.0
        g.stg.skip_self_attn_layers = [1]
        predict_with_guidance(
            _ltx2_recipe(stub),
            x_t,
            t_batch,
            guidance_cfg=g,
            condition="pos",
            contexts={"stg": {"cache_tag": "stg"}},
        )
        stg_call = next(c for c in stub.calls if c.get("skip_self_attn_layers"))
        assert stg_call["cache_tag"] == "stg"

    def test_cfg_and_stg_compose_linearly(self, stub, x_t, t_batch):
        # CFG, STG and modality are all linear, so the result is a hand-computed sum.
        _, LTX2GuidanceConfig = _import_ltx2()
        g = LTX2GuidanceConfig()
        g.cfg.scale = 3.0
        g.stg.scale = 2.0
        g.stg.skip_self_attn_layers = [1]
        out = predict_with_guidance(
            _ltx2_recipe(stub), x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg"
        )
        x_cond = stub.base + 1.0
        x_uncond = stub.base - 1.0
        x_perturbed = stub.base + 1.0 + 0.2
        # state += (3-1)*(x_cond - x_uncond) + (2-1)*(x_cond - x_perturbed)
        expected = x_cond + 2.0 * (x_cond - x_uncond) + 1.0 * (x_cond - x_perturbed)
        assert torch.allclose(out, expected)

    def test_stg_runs_sequentially_after_batched_cfg(self, x_t, t_batch):
        # CFG is batched into one forward; STG runs as a sequential second forward.
        _, LTX2GuidanceConfig = _import_ltx2()
        net = BatchedStubNet()
        g = LTX2GuidanceConfig()
        g.cfg.scale = 3.0
        g.stg.scale = 2.0
        g.stg.skip_self_attn_layers = [1]
        out = predict_with_guidance(
            _ltx2_recipe(net),
            x_t,
            t_batch,
            batched=True,
            guidance_cfg=g,
            condition=["pos"] * x_t.shape[0],
            neg_condition=["neg"] * x_t.shape[0],
            cat_fn=lambda neg, pos: list(neg) + list(pos),
        )
        assert len(net.calls) == 2
        assert any(c.get("skip_self_attn_layers") is not None for c in net.calls)
        x_cond = torch.full_like(x_t, 1.0)
        x_uncond = torch.full_like(x_t, -1.0)
        x_perturbed = x_cond + 0.2
        expected = x_cond + 2.0 * (x_cond - x_uncond) + 1.0 * (x_cond - x_perturbed)
        assert torch.allclose(out, expected)

    def test_active_modality_rejected_on_unimodal_tensor(self, stub, x_t, t_batch):
        _, LTX2GuidanceConfig = _import_ltx2()
        g = LTX2GuidanceConfig()
        g.modality.scale = 3.0
        with pytest.raises(ValueError, match="cross-modal guidance requires"):
            predict_with_guidance(_ltx2_recipe(stub), x_t, t_batch, guidance_cfg=g, condition="pos")
        assert len(stub.calls) == 1

    def test_inactive_modality_allows_unimodal_tensor(self, stub, x_t, t_batch):
        _, LTX2GuidanceConfig = _import_ltx2()
        g = LTX2GuidanceConfig()
        g.modality.scale = 1.0
        out = predict_with_guidance(_ltx2_recipe(stub), x_t, t_batch, guidance_cfg=g, condition="pos")
        assert torch.allclose(out, stub.base + 1.0)
        assert len(stub.calls) == 1

    def test_modality_runs_on_dict(self, t_batch, x_t_dict):
        # Build a stub that returns dict outputs to match x_t_dict.
        class DictStub:
            def __init__(self):
                self.calls = []

            def __call__(self, x_t, t, **kw):
                self.calls.append(kw)
                cond = kw.get("condition")
                offset = {"pos": 1.0, "neg": -1.0}.get(cond, 0.0)
                bump = 0.3 if kw.get("skip_cross_modal_attn") else 0.0
                return {k: torch.zeros_like(v) + offset + bump for k, v in x_t.items()}

        _, LTX2GuidanceConfig = _import_ltx2()
        stub = DictStub()
        g = LTX2GuidanceConfig()
        g.modality.scale = 2.0  # scalar — must broadcast into both modalities
        net = _ltx2_recipe(stub, modalities=("video", "audio"))
        out = predict_with_guidance(net, x_t_dict, t_batch, guidance_cfg=g, condition="pos")
        # Ran cond + modality forwards.
        assert len(stub.calls) == 2
        assert any(c.get("skip_cross_modal_attn") for c in stub.calls)
        assert set(out.keys()) == {"video", "audio"}
        # Scalar-scale → dict-state broadcast: ``per_modality`` indexes the dict
        # state and broadcasts the scalar scale, so every modality gets the same
        # guidance strength.  cond pass returns 1.0; modality pass
        # (same ``"pos"`` condition + ``skip_cross_modal_attn=True``) returns
        # 1.3; therefore each modality of x_pred ends up at
        # ``1.0 + (2-1)*(1.0 - 1.3) = 0.7``.
        for k, v in out.items():
            assert v.shape == x_t_dict[k].shape
            assert torch.allclose(v, torch.full_like(v, 0.7), atol=1e-6), (
                f"modality '{k}': scalar modality.scale=2.0 must broadcast "
                f"into the dict structure; got mean {v.mean().item():.4f}, expected 0.7"
            )

    def test_active_pred_keys_name_its_branches(self):
        _, LTX2GuidanceConfig = _import_ltx2()
        g = LTX2GuidanceConfig()
        g.cfg.scale, g.stg.scale, g.modality.scale = 3.0, 2.0, 4.0
        assert g.active_pred_keys() == ["pos", "neg", "stg", "modality"]

    @pytest.mark.parametrize(
        "terms",
        [
            {"cfg": ({"video": 1.0, "audio": 1.0}, None), "stg": ([2.0, 1.0], [0.5])},
            {"mix": ([1.0, 2.0], 0.5), "modality": (1.0, None)},
            {"stg": (1.0, None), "modality": ([1.0, 3.0], 0.5)},
        ],
    )
    def test_active_pred_keys_are_the_branches_it_computes(self, terms):
        """Run at a time in every piece, the recipe computes exactly these branches."""
        _, LTX2GuidanceConfig = _import_ltx2()
        g = LTX2GuidanceConfig()
        g.stg.skip_self_attn_layers = [0]
        for name, (scale, boundary) in terms.items():
            term = getattr(g, name)
            term.scale, term.boundary = scale, boundary

        def forward(x_t, t, **kwargs):
            return {key: torch.zeros_like(value) for key, value in x_t.items()}

        net = _ltx2_recipe(forward, modalities=("video", "audio"))
        computed = set()
        for t in (0.2, 0.8):
            x_t = {"video": torch.zeros(1, 4), "audio": torch.zeros(1, 2)}
            _, cache = predict_with_guidance(
                net,
                x_t,
                torch.tensor([t], dtype=torch.float64),
                guidance_cfg=g,
                condition="pos",
                neg_condition="neg",
                return_cache=True,
            )
            computed |= cache.keys()
        keys = g.active_pred_keys()
        assert len(keys) == len(set(keys)) and set(keys) == computed

    @pytest.mark.parametrize("rescale_type", ["std", "norm"])
    def test_flow_callers_rescale_x0_like_the_sampler(self, rescale_type):
        _, LTX2GuidanceConfig = _import_ltx2()
        ns = RFNoiseSchedule()
        net = _ltx2_recipe(_flow_forward)
        generator = torch.Generator().manual_seed(0)
        x, pos, neg = (torch.randn(2, 6, generator=generator, dtype=torch.float64) for _ in range(3))
        t = torch.tensor([0.3, 0.8], dtype=torch.float64)
        g = LTX2GuidanceConfig()
        g.cfg.scale, g.rescale, g.rescale_type = 3.0, 0.7, rescale_type
        flow = predict_with_guidance(net, x, t, pos, g, neg, fwd_pred_type="flow")
        x0 = predict_with_guidance(net, x, t, pos, g, neg, fwd_pred_type="x0")
        torch.testing.assert_close(flow, ns.x0_to_flow(x, x0, t))
        v_pos, v_neg = _flow_forward(x, t, pos), _flow_forward(x, t, neg)
        v_guided = v_pos + 2 * (v_pos - v_neg)
        x0_cond, x0_guided = ns.flow_to_x0(x, v_pos, t), ns.flow_to_x0(x, v_guided, t)
        rescaled = cfg_rescale(x0_guided, x0_cond, 0.7, rescale_type=rescale_type)
        torch.testing.assert_close(flow, ns.x0_to_flow(x, rescaled, t))
        # Rescaling the velocity itself would distill another target.
        assert not torch.allclose(flow, cfg_rescale(v_guided, v_pos, 0.7, rescale_type=rescale_type))

    def test_flow_callers_keep_the_unrescaled_composition_at_sigma_zero(self):
        _, LTX2GuidanceConfig = _import_ltx2()
        ns = RFNoiseSchedule()
        generator = torch.Generator().manual_seed(0)
        x, pos, neg = (torch.randn(2, 6, generator=generator, dtype=torch.float64) for _ in range(3))
        t = torch.tensor([0.0, 0.5], dtype=torch.float64)
        g = LTX2GuidanceConfig()
        g.cfg.scale, g.rescale = 3.0, 0.7
        flow = predict_with_guidance(_ltx2_recipe(_flow_forward), x, t, pos, g, neg, fwd_pred_type="flow")
        v_pos, v_neg = _flow_forward(x, t, pos), _flow_forward(x, t, neg)
        v_guided = v_pos + 2 * (v_pos - v_neg)
        torch.testing.assert_close(flow[0], v_guided[0], rtol=0, atol=0)
        rescaled = cfg_rescale(ns.flow_to_x0(x, v_guided, t), ns.flow_to_x0(x, v_pos, t), 0.7)
        torch.testing.assert_close(flow[1], ns.x0_to_flow(x, rescaled, t)[1])

    def test_rescale_requires_the_unshifted_rf_schedule(self, stub, x_t, t_batch):
        _, LTX2GuidanceConfig = _import_ltx2()
        net = _ltx2_recipe(stub)
        net.schedule_type = "shifted_rf"
        g = LTX2GuidanceConfig()
        g.rescale = 0.7
        with pytest.raises(NotImplementedError, match="unshifted 'rf'"):
            predict_with_guidance(net, x_t, t_batch, guidance_cfg=g, condition="pos")

    def test_guidance_composes_in_the_working_dtype_and_rounds_once(self):
        _, LTX2GuidanceConfig = _import_ltx2()

        def forward(*args, **kwargs):
            raise AssertionError("Every branch is cached")

        generator = torch.Generator().manual_seed(0)
        branches = {key: torch.randn(2, 64, generator=generator).to(torch.bfloat16) for key in ("pos", "neg", "stg")}
        g = LTX2GuidanceConfig()
        g.cfg.scale, g.stg.scale, g.stg.skip_self_attn_layers, g.rescale = 3.0, 2.0, [1], 0.7
        t = torch.tensor([0.3, 0.7], dtype=torch.float64)
        actual = predict_with_guidance(
            _ltx2_recipe(forward),
            torch.zeros(2, 64, dtype=torch.bfloat16),
            t,
            "pos",
            g,
            "neg",
            precomputed=branches,
            fwd_pred_type="x0",
        )
        # At a zero state, x0 = -sigma * v: the per-sample sigma cancels in the rescale factor.
        pos, neg, stg = (branches[key].double() for key in ("pos", "neg", "stg"))
        velocity = cfg_rescale(pos + 2 * (pos - neg) + (pos - stg), pos, 0.7)
        expected = (-t.view(-1, 1) * velocity).to(torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rescale_on_x0_keeps_a_working_dtype_per_modality(self, monkeypatch):
        _, LTX2GuidanceConfig = _import_ltx2()
        monkeypatch.setattr(tensor_module, "HIGH_PRECISION", torch.float32)

        def forward(x_t, t, condition, **kwargs):
            return {key: 0.5 * value + (1.0 if condition == "pos" else -1.0) for key, value in x_t.items()}

        net = _ltx2_recipe(forward, modalities=("video", "audio"))
        generator = torch.Generator().manual_seed(0)
        x_t = {
            "video": torch.randn(2, 6, generator=generator).to(torch.bfloat16),
            "audio": torch.randn(2, 4, generator=generator, dtype=torch.float64),
        }
        t = torch.tensor([0.3, 0.8], dtype=net.noise_scheduler.t_precision)
        g = LTX2GuidanceConfig()
        g.cfg.scale, g.rescale = 3.0, 0.7
        actual, cache = predict_with_guidance(net, x_t, t, "pos", g, "neg", return_cache=True, output_dtype=None)
        working = {"video": torch.float32, "audio": torch.float64}
        for prediction in (actual, *cache.values()):
            assert per_modality(lambda x: x.dtype, prediction) == working
        # Requesting float64 widens only the video, so the audio is computed alike.
        reference = predict_with_guidance(net, x_t, t, "pos", g, "neg", output_dtype=torch.float64)
        torch.testing.assert_close(actual["video"], reference["video"].float())
        torch.testing.assert_close(actual["audio"], reference["audio"], rtol=0, atol=0)


# ═════════════════════════════════════════════════════════════════════════
# predict_with_guidance — mix term
# ═════════════════════════════════════════════════════════════════════════


class TestPredictWithGuidanceMix:
    def test_mix_auto_computes_pos_mix(self, stub, x_t, t_batch):
        # When mix.scale is set and "pos_mix" is not in precomputed,
        # predict_with_guidance auto-computes it via a fresh cond forward
        # (same kwargs as "pos").  No raise.
        g = GuidanceConfig()
        g.mix.scale = 2.0
        predict_with_guidance(stub, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")
        # 3 forwards: cond ("pos") + cfg-uncond ("neg") + auto-computed "pos_mix".
        assert len(stub.calls) == 3
        # Two of them are cond passes (same condition) — "pos" and "pos_mix".
        cond_calls = [c for c in stub.calls if c["condition"] == "pos"]
        assert len(cond_calls) == 2

    def test_mix_pos_mix_override(self, stub, x_t, t_batch):
        # When CFG is enabled and "pos_mix" is supplied, mix uses the
        # override instead of running an extra forward; cfg's "neg"
        # is shared between cfg and mix.
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        g.mix.scale = 2.0
        pos_mix = torch.full_like(x_t, 4.0)
        predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
            precomputed={"pos_mix": pos_mix},
        )
        # 2 forwards: cond ("pos") + cfg uncond ("neg"). pos_mix overridden.
        assert len(stub.calls) == 2

    def test_mix_triggers_uncond_when_cfg_off(self, stub, x_t, t_batch):
        # When CFG is off but mix is on, the mix term still requires the
        # cfg-uncond reference, so a "neg" forward is run.  pos_mix is
        # auto-computed (one more forward).
        g = GuidanceConfig()
        g.mix.scale = 2.0
        predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
        )
        # 3 forwards: cond + uncond (mix-induced) + pos_mix.
        assert len(stub.calls) == 3
        cfg_call = next(c for c in stub.calls if c["condition"] == "neg")
        assert cfg_call.get("skip_layers") is None  # no cfg term to supply skip_layers

    def test_mix_algebraic_identity_vs_meanflow(self, x_t, t_batch):
        """``predict_with_guidance`` with ``cfg.scale=ω``, ``mix.scale=κ+1``
        and ``precomputed={pos: dxt_dt, neg: neg, pos_mix: pos}`` matches
        the MeanFlow paper's eq. 21 ``ω·dxt_dt + κ·pos + (1-ω-κ)·neg``
        up to float32 rounding.
        """
        torch.manual_seed(0)
        dxt_dt = torch.randn(3, 4)
        neg = torch.randn(3, 4)
        pos = torch.randn(3, 4)

        for omega, kappa in [
            (3.0, 0.5),
            (1.0, 0.0),
            (0.0, 0.5),
            (1.0, 0.5),
            (2.0, -0.5),  # ω + κ != 1
        ]:
            g = GuidanceConfig()
            g.cfg.scale = omega
            g.mix.scale = kappa + 1.0
            out = predict_with_guidance(
                StubNet(torch.zeros(3, 4)),  # never invoked; everything precomputed
                x_t,
                t_batch,
                guidance_cfg=g,
                condition="pos",
                neg_condition="neg",
                precomputed={"pos": dxt_dt, "neg": neg, "pos_mix": pos},
            )
            expected = omega * dxt_dt + kappa * pos + (1.0 - omega - kappa) * neg
            assert torch.allclose(out, expected, atol=1e-6), f"ω={omega}, κ={kappa}"

    def test_mix_3way_with_t_window(self):
        """3-way teacher mixture gated by a t-window via piecewise (scale, boundary).

        Setting ``cfg.scale=[ω, 1.0]`` + ``cfg.boundary=[t_end]`` and
        ``mix.scale=[κ+1, 1.0]`` + ``mix.boundary=[t_end]`` must yield
        the MeanFlow paper's eq. 21 blend ``ω·dxt_dt + κ·pos + (1-ω-κ)·neg``
        for samples with ``t < t_end`` and pass ``dxt_dt`` through unchanged
        for samples with ``t >= t_end`` (both terms collapse to a no-op at
        scale=1).
        """
        torch.manual_seed(0)
        # Mix in-window and out-of-window samples in the same batch.
        omega, kappa, t_end = 0.2, 0.92, 0.75  # config_mf_xl.py settings
        t = torch.tensor([0.10, 0.50, t_end - 1e-3, t_end, t_end + 1e-3, 0.90], dtype=torch.float64)
        in_window = t < t_end  # bucketize(right=True) → t == t_end goes to the upper bucket
        B = t.shape[0]
        dxt_dt = torch.randn(B, 4)
        neg = torch.randn(B, 4)
        pos = torch.randn(B, 4)

        g = GuidanceConfig()
        g.cfg.scale = [omega, 1.0]
        g.cfg.boundary = [t_end]
        g.mix.scale = [kappa + 1.0, 1.0]
        g.mix.boundary = [t_end]

        out = predict_with_guidance(
            StubNet(torch.zeros(B, 4)),  # never invoked; everything precomputed
            torch.zeros(B, 4),
            t,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
            precomputed={"pos": dxt_dt, "neg": neg, "pos_mix": pos},
        )

        expected_in = omega * dxt_dt + kappa * pos + (1.0 - omega - kappa) * neg
        expected = torch.where(in_window.unsqueeze(-1), expected_in, dxt_dt)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_mix_only_with_t_window(self):
        """Mix-only window (cfg disabled, ω implicitly 1) — config_mf_b.py's mix scale, windowed.

        With ``cfg.scale=None`` and ``mix.scale=[κ+1, 1.0]`` +
        ``mix.boundary=[t_end]``, the result (paper eq. 21 with ω=1) is
        ``dxt_dt + κ·(pos - neg)`` inside the window and ``dxt_dt`` outside
        (mix at scale=1 is a no-op).
        """
        torch.manual_seed(1)
        kappa, t_end = 0.5, 0.75
        t = torch.tensor([0.20, 0.74, 0.75, 0.80], dtype=torch.float64)
        in_window = t < t_end
        B = t.shape[0]
        dxt_dt = torch.randn(B, 4)
        neg = torch.randn(B, 4)
        pos = torch.randn(B, 4)

        g = GuidanceConfig()
        g.mix.scale = [kappa + 1.0, 1.0]
        g.mix.boundary = [t_end]

        out = predict_with_guidance(
            StubNet(torch.zeros(B, 4)),
            torch.zeros(B, 4),
            t,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
            precomputed={"pos": dxt_dt, "neg": neg, "pos_mix": pos},
        )

        expected_in = dxt_dt + kappa * (pos - neg)
        expected = torch.where(in_window.unsqueeze(-1), expected_in, dxt_dt)
        assert torch.allclose(out, expected, atol=1e-6)


# ═════════════════════════════════════════════════════════════════════════
# predict_with_guidance — combinations and rescale
# ═════════════════════════════════════════════════════════════════════════


class TestPredictWithGuidanceCombinations:
    def test_rescale_applied_last(self, stub, x_t, t_batch):
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        g.rescale = 0.7
        out = predict_with_guidance(stub, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")
        x_cond = stub.base + 1.0
        x_uncond = stub.base - 1.0
        guided = apply_guidance(x_cond, x_cond, x_uncond, 3.0)
        expected = cfg_rescale(guided, x_cond, 0.7)
        assert torch.allclose(out, expected)


# ═════════════════════════════════════════════════════════════════════════
# cfg_rescale — std vs norm, dims selection
# ═════════════════════════════════════════════════════════════════════════


class TestCfgRescaleVariants:
    @pytest.mark.parametrize("rescale", [None, 0.0])
    @pytest.mark.parametrize("rescale_type", ["std", "norm"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
    def test_disabled_rescale_preserves_singleton_predictions(self, rescale, rescale_type, dtype):
        # Sample std is undefined for one element; a disabled rescale must not evaluate it.
        guided = torch.tensor([[2.0], [3.0]], dtype=dtype, requires_grad=True)
        cond = torch.ones_like(guided)
        out = cfg_rescale(guided, cond, rescale, rescale_type=rescale_type)
        assert out is guided
        out.sum().backward()
        torch.testing.assert_close(guided.grad, torch.ones_like(guided))

    def test_norm_pure_ratio_matches_cond_norm(self):
        # rescale=1.0 + type='norm' + dims=(-1,) is a pure per-row L2 rescale:
        # the resulting tensor's per-row L2 norm should equal cond's per-row L2 norm.
        torch.manual_seed(0)
        cond = torch.randn(3, 5, 8)  # [B, seq, D]
        guided = torch.randn(3, 5, 8) * 4.0  # different scale
        out = cfg_rescale(guided, cond, 1.0, rescale_type="norm", rescale_dims=(-1,))
        cond_n = torch.linalg.vector_norm(cond, dim=-1)
        out_n = torch.linalg.vector_norm(out, dim=-1)
        assert torch.allclose(out_n, cond_n, atol=1e-5)

    def test_norm_blend_at_half_strength(self):
        # rescale=0.5 + type='norm' should blend halfway between guided and the
        # cond-norm-rescaled version.
        torch.manual_seed(0)
        cond = torch.randn(3, 5, 8)
        guided = torch.randn(3, 5, 8) * 4.0
        full = cfg_rescale(guided, cond, 1.0, rescale_type="norm", rescale_dims=(-1,))
        half = cfg_rescale(guided, cond, 0.5, rescale_type="norm", rescale_dims=(-1,))
        assert torch.allclose(half, 0.5 * (guided + full), atol=1e-5)

    def test_std_dims_default_is_all_non_batch(self):
        # rescale_dims=None ⇒ all non-batch dims (preserves the original
        # CFG-rescale behavior on [B, C, H, W]).
        torch.manual_seed(0)
        cond = torch.randn(3, 4, 5, 6)
        guided = torch.randn(3, 4, 5, 6) * 3.0
        out = cfg_rescale(guided, cond, 1.0, rescale_type="std", rescale_dims=None)
        # rescale=1.0 with std → factor = cond_std / guided_std; out_std == cond_std.
        cond_std = cond.std(dim=(1, 2, 3))
        out_std = out.std(dim=(1, 2, 3))
        assert torch.allclose(out_std, cond_std, atol=1e-5)

        # Dict-state + scalar-rescale broadcast: ``per_modality`` indexes the
        # dict state and broadcasts the scalar rescale, so every modality
        # rescales independently with the same strength.  Same property as
        # above (out_std == cond_std) must hold per modality.
        cond_dict = {
            "video": torch.randn(3, 4, 5, 6),
            "audio": torch.randn(3, 2, 7),
        }
        guided_dict = {
            "video": torch.randn(3, 4, 5, 6) * 3.0,
            "audio": torch.randn(3, 2, 7) * 5.0,
        }
        out_dict = per_modality(
            cfg_rescale,
            guided_dict,
            cond_dict,
            1.0,
            rescale_type="std",
            rescale_dims=None,
        )
        assert set(out_dict.keys()) == {"video", "audio"}
        for k in cond_dict:
            non_batch_dims = tuple(range(1, cond_dict[k].ndim))
            assert torch.allclose(
                out_dict[k].std(dim=non_batch_dims),
                cond_dict[k].std(dim=non_batch_dims),
                atol=1e-5,
            ), f"modality '{k}': scalar rescale=1.0 must broadcast into the dict structure"

    def test_unknown_rescale_type_raises(self):
        cond = torch.randn(3, 4)
        with pytest.raises(ValueError, match="rescale_type"):
            cfg_rescale(cond, cond, 1.0, rescale_type="foo")

    def test_predict_with_guidance_routes_norm_rescale(self, x_t, t_batch):
        # End-to-end: GuidanceConfig.rescale_type='norm' + rescale_dims=(-1,)
        # threads through predict_with_guidance to cfg_rescale.  The output's
        # per-row L2 norm should match the cond pred's per-row L2 norm.
        net = StubNet(base=torch.zeros(3, 4))
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        g.rescale = 1.0
        g.rescale_type = "norm"
        g.rescale_dims = (-1,)
        out = predict_with_guidance(net, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")
        x_cond = net.base + 1.0  # what predict_with_guidance uses for the rescale anchor
        cond_n = torch.linalg.vector_norm(x_cond, dim=-1)
        out_n = torch.linalg.vector_norm(out, dim=-1)
        assert torch.allclose(out_n, cond_n, atol=1e-5)


# ═════════════════════════════════════════════════════════════════════════
# predict_with_guidance — providers
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("provider", ["function", "module"])
def test_guidance_rejects_a_provider_that_is_not_a_network(stub, x_t, t_batch, provider):
    net = (lambda *args, **kwargs: stub(*args, **kwargs)) if provider == "function" else torch.nn.Identity()
    with pytest.raises(TypeError, match="needs a FastGenNetwork"):
        predict_with_guidance(net, x_t, t_batch, condition="pos")
    assert stub.calls == []


# ═════════════════════════════════════════════════════════════════════════
# predict_with_guidance
# ═════════════════════════════════════════════════════════════════════════


class BatchedStubNet(FunctionNet):
    """Stub network whose output depends on the *condition* tag on a per-row basis.

    Lets the batched helper run one forward over a doubled batch and
    produce ``out[:B] == uncond`` and ``out[B:] == cond`` after chunking.
    Also reacts to perturbation kwargs (``skip_self_attn_layers``,
    ``skip_cross_modal_attn``) so LTX-2's STG / modality terms can be exercised.
    """

    OFFSETS = {"pos": 1.0, "neg": -1.0}

    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def native(self, x, t, condition=None, **kw):
        self.calls.append({"condition": condition, **kw})
        cond = condition
        # cond may be a list of tags ("pos"/"neg") or a single tag broadcast.
        if isinstance(cond, list) and cond and isinstance(cond[0], str):
            offsets = torch.tensor([self.OFFSETS[c] for c in cond], dtype=torch.float32)
        else:
            offsets = torch.tensor([self.OFFSETS[cond]] * x.shape[0], dtype=torch.float32)
        out = offsets.view(-1, *([1] * (x.ndim - 1))).expand_as(x).clone()
        if kw.get("skip_self_attn_layers") is not None:
            out = out + 0.2
        if kw.get("skip_cross_modal_attn"):
            out = out + 0.3
        return out


class TestPredictWithBatchedGuidance:
    def test_matches_sequential(self, x_t, t_batch):
        # Same numerical result as sequential predict_with_guidance.
        net = BatchedStubNet()
        g = GuidanceConfig()
        g.cfg.scale = 3.0

        # Sequential path: pass single tag, stub broadcasts.
        seq = predict_with_guidance(net, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")

        # Batched path: cat_fn concats list of single-element tags into a list of B tags.
        net_b = BatchedStubNet()
        batched = predict_with_guidance(
            net_b,
            x_t,
            t_batch,
            batched=True,
            guidance_cfg=g,
            condition=["pos"] * x_t.shape[0],
            neg_condition=["neg"] * x_t.shape[0],
            cat_fn=lambda neg, pos: list(neg) + list(pos),
        )
        assert torch.allclose(seq, batched)
        # Sequential = 2 forwards (cond + cfg-uncond); batched = 1 forward.
        assert len(net.calls) == 2
        assert len(net_b.calls) == 1

    def test_no_cfg_falls_back_to_sequential(self, stub, x_t, t_batch):
        # cfg.scale=None (or guidance_cfg=None) ⇒ no batched forward, single cond forward.
        out = predict_with_guidance(
            stub,
            x_t,
            t_batch,
            batched=True,
            guidance_cfg=None,
            condition="pos",
            neg_condition="neg",
        )
        assert torch.allclose(out, stub.base + 1.0)
        assert len(stub.calls) == 1

    def test_rescale_applied_after_the_batched_pair(self, x_t, t_batch):
        # Rescale is applied after the batched CFG forward. Verify the result matches.
        net = BatchedStubNet()
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        g.rescale = 0.7
        out = predict_with_guidance(
            net,
            x_t,
            t_batch,
            batched=True,
            guidance_cfg=g,
            condition=["pos"] * x_t.shape[0],
            neg_condition=["neg"] * x_t.shape[0],
            cat_fn=lambda neg, pos: list(neg) + list(pos),
        )
        x_cond = torch.full_like(x_t, 1.0)
        x_uncond = torch.full_like(x_t, -1.0)
        guided = apply_guidance(x_cond, x_cond, x_uncond, 3.0)
        expected = cfg_rescale(guided, x_cond, 0.7)
        assert torch.allclose(out, expected)

    def test_rejects_multimodal_x_t(self, t_batch):
        # x_t must be a Tensor; dict (multimodal) is rejected because we
        # explicitly torch.cat([x_t, x_t]) and would need a per-modality cat.
        net = BatchedStubNet()
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        with pytest.raises(TypeError, match="Tensor x_t"):
            predict_with_guidance(
                net,
                {"video": torch.zeros(3, 4)},
                t_batch,
                batched=True,
                guidance_cfg=g,
                condition="pos",
                neg_condition="neg",
            )

    def test_skip_layers_raises(self, x_t, t_batch):
        # The single batched forward cannot apply per-batch-element skip lists.
        net = BatchedStubNet()
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        g.cfg.skip_layers = [5, 10]
        with pytest.raises(ValueError, match="skip_layers"):
            predict_with_guidance(
                net,
                x_t,
                t_batch,
                batched=True,
                guidance_cfg=g,
                condition=["pos"] * x_t.shape[0],
                neg_condition=["neg"] * x_t.shape[0],
                cat_fn=lambda neg, pos: list(neg) + list(pos),
            )

    def test_cat_fn_failure_is_wrapped(self, x_t, t_batch):
        # A failing ``cat_fn`` (most common cause: ``neg_condition=None``
        # while CFG is active) must surface as an actionable ``ValueError``
        # mentioning the CFG scale, not as a cryptic ``TypeError`` from
        # deep inside the caller's lambda. The original exception is
        # chained via ``__cause__`` so the traceback is preserved.
        net = BatchedStubNet()
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        with pytest.raises(ValueError, match="cat_fn.*failed") as exc_info:
            predict_with_guidance(
                net,
                x_t,
                t_batch,
                batched=True,
                guidance_cfg=g,
                condition=["pos"] * x_t.shape[0],
                neg_condition=None,
                cat_fn=lambda neg, pos: list(neg) + list(pos),
            )
        assert "guidance_cfg.cfg.scale=3.0" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, TypeError)

    @pytest.mark.parametrize("attribute", ["shape", "dtype", "device"])
    def test_default_cat_requires_matching_tensor_metadata(self, attribute, x_t, t_batch):
        net = BatchedStubNet()
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        condition = torch.ones(x_t.shape[0], 4)
        if attribute == "shape":
            neg_condition = torch.ones(x_t.shape[0] + 1, 4)
        elif attribute == "dtype":
            neg_condition = condition.double()
        else:
            neg_condition = torch.empty_like(condition, device="meta")

        with pytest.raises(ValueError, match="cat_fn.*failed") as exc_info:
            predict_with_guidance(
                net,
                x_t,
                t_batch,
                batched=True,
                guidance_cfg=g,
                condition=condition,
                neg_condition=neg_condition,
            )

        cause = exc_info.value.__cause__
        assert isinstance(cause, AssertionError)
        assert f"neg.{attribute}" in str(cause)
        assert f"pos.{attribute}" in str(cause)
        assert net.calls == []

    def test_list_condition_with_custom_cat(self, x_t, t_batch):
        # Conditions shaped like SDXL's list-of-two-tensors. Custom
        # cat_fn concatenates each list element along dim 0.
        condition = [torch.full((3, 8), 1.0), torch.full((3, 16), 2.0)]
        neg_condition = [torch.full((3, 8), -1.0), torch.full((3, 16), -2.0)]

        captured = {}

        def native(x, t, condition=None, **kw):
            captured["condition"] = condition
            captured["x_shape"] = tuple(x.shape)
            return torch.zeros(x.shape[0], 4)

        g = GuidanceConfig()
        g.cfg.scale = 3.0
        out = predict_with_guidance(
            FunctionNet(native),
            x_t,
            t_batch,
            batched=True,
            guidance_cfg=g,
            condition=condition,
            neg_condition=neg_condition,
            cat_fn=lambda neg, pos: [torch.cat([n, p], dim=0) for n, p in zip(neg, pos, strict=True)],
        )
        # Verify the cat happened correctly: first element shape (6, 8), second (6, 16).
        assert captured["x_shape"] == (6, 4)
        assert captured["condition"][0].shape == (6, 8)
        assert captured["condition"][1].shape == (6, 16)
        # Negative half = -1, positive half = 1, applied per row.
        assert (captured["condition"][0][:3] == -1.0).all()
        assert (captured["condition"][0][3:] == 1.0).all()
        assert out.shape == x_t.shape


# ═════════════════════════════════════════════════════════════════════════
# is_active + skip_inactive (forward-skipping optimization)
# ═════════════════════════════════════════════════════════════════════════


class TestIsActive:
    def test_none_is_inactive(self):
        assert not is_active(None)

    def test_scalar_one_is_inactive(self):
        assert not is_active(1.0)
        assert not is_active(1)  # int 1 also counts

    def test_scalar_other_is_active(self):
        assert is_active(2.0)
        assert is_active(0.0)  # 0 is *not* a no-op (pushes toward uncond)
        assert is_active(0.999)

    def test_tensor_uniform_one_is_inactive(self):
        # Returns a 0-d bool tensor (Python's `if` materialises via __bool__).
        assert not is_active(torch.ones(4))

    def test_tensor_with_nonone_is_active(self):
        assert is_active(torch.tensor([1.0, 1.0, 2.0, 1.0]))
        assert is_active(torch.tensor([0.0, 0.0, 0.0, 0.0]))  # 0 is not no-op

    def test_dict_all_one_is_inactive(self):
        assert not is_active({"video": 1.0, "audio": 1.0})
        assert not is_active({"video": torch.ones(4), "audio": 1.0})

    def test_dict_any_active_is_active(self):
        assert is_active({"video": 1.0, "audio": 3.0})
        assert is_active({"video": torch.tensor([1.0, 1.0, 2.0]), "audio": 1.0})


class TestSkipInactive:
    def test_scalar_one_skips_forward(self, stub, x_t, t_batch):
        # cfg.scale=1.0 is a static no-op → skip_inactive (default) drops the neg forward.
        g = GuidanceConfig()
        g.cfg.scale = 1.0
        out = predict_with_guidance(stub, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")
        # Only the cond ("pos") forward runs; no "neg" forward.
        assert len(stub.calls) == 1
        assert stub.calls[0]["condition"] == "pos"
        # And the result is just x_cond (no perturbation): base + "pos" offset (1.0).
        assert torch.equal(out, stub.base + 1.0)

    def test_scalar_one_force_compute(self, stub, x_t, t_batch):
        # skip_inactive=False bypasses the optimization → neg forward still runs.
        g = GuidanceConfig()
        g.cfg.scale = 1.0
        predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
            skip_inactive=False,
        )
        # Two forwards: "pos" + "neg" (apply_guidance with s=1 yields x_pred unchanged).
        assert len(stub.calls) == 2

    def test_piecewise_uniform_one_skips(self, stub, x_t):
        # Whole batch lands in the "off" segment (scale=1.0) → skip.
        g = GuidanceConfig()
        g.cfg.scale = [3.0, 1.0]
        g.cfg.boundary = [0.5]
        # All t > 0.5 → out-of-window → s_eff uniformly 1.0.
        t_out = torch.tensor([0.6, 0.7, 0.8], dtype=torch.float64)
        predict_with_guidance(stub, x_t, t_out, guidance_cfg=g, condition="pos", neg_condition="neg")
        assert len(stub.calls) == 1  # only cond pass

    def test_piecewise_partial_window_does_not_skip(self, stub, x_t):
        # Mixed batch (some samples in-window, some out) → s_eff has a non-1 entry → run.
        g = GuidanceConfig()
        g.cfg.scale = [3.0, 1.0]
        g.cfg.boundary = [0.5]
        t_mixed = torch.tensor([0.4, 0.6, 0.4], dtype=torch.float64)
        predict_with_guidance(stub, x_t, t_mixed, guidance_cfg=g, condition="pos", neg_condition="neg")
        # Two forwards: "pos" + "neg" (the in-window samples get the perturbation;
        # out-of-window get s_eff=1.0 which is a no-op via apply_guidance).
        assert len(stub.calls) == 2

    def test_mix_scale_one_skips_pos_mix_forward(self, stub, x_t, t_batch):
        # mix.scale=1.0 is a no-op → both pos_mix and neg are skipped (mix is the only
        # term that would force the neg forward).
        g = GuidanceConfig()
        g.mix.scale = 1.0
        predict_with_guidance(stub, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")
        # Only the cond ("pos") forward — no neg, no pos_mix.
        assert len(stub.calls) == 1


# ═════════════════════════════════════════════════════════════════════════
# GuidanceConfig.active_pred_keys / condition_for_pred_key
# ═════════════════════════════════════════════════════════════════════════


class TestActivePredKeys:
    def test_disabled(self):
        assert GuidanceConfig().active_pred_keys() == ["pos"]

    def test_cfg_adds_neg(self):
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        assert g.active_pred_keys() == ["pos", "neg"]

    def test_network_terms_added_by_name_when_active(self):
        g = NetworkGuidance()
        g.cfg.scale = 3.0
        g.perturb.scale = 2.0
        assert g.active_pred_keys() == ["pos", "neg", "perturb"]

    def test_a_subclass_overrides_the_branches_of_unusual_terms(self):
        g = BranchingGuidance()
        g.reuse.scale = 2.0
        assert g.active_pred_keys() == ["pos", "neg"]
        g.pair.scale = 2.0
        assert g.active_pred_keys() == ["pos", "pair", "pair_neg", "neg"]

    def test_branch_methods_match_the_branches_a_recipe_computes(self, x_t, t_batch):
        conditions = []

        class BranchingNet(FunctionNet):
            def native(self, x_t, t, condition=None, **kwargs):
                conditions.append(condition)
                return torch.full_like(x_t, 1.0 if condition == "pos" else -1.0)

            def predict_with_guidance(self, ctx):
                g = ctx.guidance_cfg
                guided = cond = ctx.predict("pos", condition=ctx.condition)
                scale = ctx.scale(g.pair)
                if scale is not None:
                    pair = ctx.predict("pair", condition=ctx.condition)
                    guided = apply_guidance(guided, pair, ctx.predict("pair_neg", condition=ctx.neg_condition), scale)
                scale = ctx.scale(g.reuse)
                if scale is not None:
                    guided = apply_guidance(guided, cond, ctx.predict("neg", condition=ctx.neg_condition), scale)
                return guided

        g = BranchingGuidance()
        g.pair.scale, g.reuse.scale = 2.0, 3.0
        _, cache = predict_with_guidance(
            BranchingNet(), x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg", return_cache=True
        )
        assert set(cache) == set(g.active_pred_keys())
        assert conditions == [g.condition_for_pred_key(key, "pos", "neg") for key in cache]

    def test_mix_triggers_neg_when_cfg_off(self):
        g = GuidanceConfig()
        g.mix.scale = 2.0
        # Mix needs cfg-uncond reference; "neg" is added even though cfg is off.
        # "pos_mix" is also added (auto-computed cond forward).
        assert g.active_pred_keys() == ["pos", "neg", "pos_mix"]

    def test_mix_does_not_duplicate_neg(self):
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        g.mix.scale = 2.0
        # cfg already adds "neg"; mix doesn't add a second one but does add "pos_mix".
        keys = g.active_pred_keys()
        assert keys.count("neg") == 1
        assert "pos_mix" in keys

    def test_scale_of_one_everywhere_adds_no_branch(self):
        # Causal samplers provision a KV cache per key, so a no-op term must add no forward (nor need neg_condition).
        g = NetworkGuidance()
        g.cfg.scale = 1.0
        g.mix.scale = 1
        for scale, boundary in [([1.0, 1.0], 0.5), ({"video": 1.0, "audio": None}, None)]:
            g.perturb.scale, g.perturb.boundary = scale, boundary
            assert g.active_pred_keys() == ["pos"]
        # One active piece or modality is enough.
        g.cfg.scale, g.cfg.boundary = [1.0, 4.0], 0.5
        g.perturb.scale = {"video": 1.0, "audio": 3.0}
        assert g.active_pred_keys() == ["pos", "neg", "perturb"]

    @pytest.mark.parametrize(
        "terms",
        [
            {"cfg": (1.0, None)},
            {"cfg": ([1.0, 1.0], 0.5), "mix": (1.0, None)},
            {"cfg": ([1.0, 4.0], 0.5)},
            {"cfg": ({"video": 1.0, "audio": [1.0, 3.0]}, {"video": None, "audio": 0.5})},
            {"mix": ([1.0, 2.0], 0.5)},
        ],
    )
    def test_keys_are_the_branches_predict_with_guidance_computes_at_some_time(self, terms):
        """Run at a time in every piece, ``predict_with_guidance`` computes exactly these branches."""
        g = GuidanceConfig()
        for name, (scale, boundary) in terms.items():
            term = getattr(g, name)
            term.scale, term.boundary = scale, boundary

        net = FunctionNet(
            lambda x_t, t, **kwargs: {key: torch.zeros_like(value) for key, value in x_t.items()},
            modalities=("video", "audio"),
        )
        computed = set()
        for t in (0.2, 0.8):
            x_t = {"video": torch.zeros(1, 4), "audio": torch.zeros(1, 2)}
            _, cache = predict_with_guidance(
                net,
                x_t,
                torch.tensor([t], dtype=torch.float64),
                guidance_cfg=g,
                condition="pos",
                neg_condition="neg",
                return_cache=True,
            )
            computed |= cache.keys()
        keys = g.active_pred_keys()
        assert len(keys) == len(set(keys)) and set(keys) == computed


class TestConditionForPredKey:
    def test_neg_uses_neg_condition(self):
        assert GuidanceConfig().condition_for_pred_key("neg", "POS", "NEG") == "NEG"

    def test_pos_pos_mix_and_network_terms_use_condition(self):
        for key in ("pos", "pos_mix", "perturb"):
            assert NetworkGuidance().condition_for_pred_key(key, "POS", "NEG") == "POS"

    @pytest.mark.parametrize("key", ["negative", "perturb"])
    def test_unknown_key_raises(self, key):
        with pytest.raises(ValueError, match="unknown guidance branch key"):
            GuidanceConfig().condition_for_pred_key(key, "POS", "NEG")


# ═════════════════════════════════════════════════════════════════════════
# return_cache (sequential and batched predict_with_guidance)
# ═════════════════════════════════════════════════════════════════════════


class TestReturnCache:
    def test_return_cache_exposes_pos_pred(self, stub, x_t, t_batch):
        # cache["pos"] is the cond pred — useful for callers that want
        # to post-process the guided result (e.g. DiT's SIT convention).
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        out, cache = predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
            return_cache=True,
        )
        assert set(cache.keys()) == {"pos", "neg"}
        # Cache entries are in the working dtype.
        torch.testing.assert_close(cache["pos"], (stub.base + 1.0).double())
        torch.testing.assert_close(cache["neg"], (stub.base - 1.0).double())
        # Composed pred unchanged by return_cache.
        x_cond = stub.base + 1.0
        x_uncond = stub.base - 1.0
        expected = apply_guidance(x_cond, x_cond, x_uncond, 3.0)
        assert torch.allclose(out, expected)

    def test_return_cache_default_false(self, stub, x_t, t_batch):
        # Without return_cache, returns just the prediction.
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        out = predict_with_guidance(stub, x_t, t_batch, guidance_cfg=g, condition="pos", neg_condition="neg")
        assert isinstance(out, torch.Tensor)

    def test_return_cache_includes_precomputed(self, stub, x_t, t_batch):
        # Precomputed entries are inside the returned cache, so callers
        # see the full per-pass picture.
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        prebuilt_cond = torch.full_like(x_t, 5.0)
        out, cache = predict_with_guidance(
            stub,
            x_t,
            t_batch,
            guidance_cfg=g,
            condition="pos",
            neg_condition="neg",
            precomputed={"pos": prebuilt_cond},
            return_cache=True,
            output_dtype=torch.float64,
        )
        assert out.dtype == torch.float64
        assert torch.equal(cache["pos"], prebuilt_cond.double())
        assert cache["pos"].dtype == cache["neg"].dtype == torch.float64
        assert "neg" in cache

    def test_return_cache_batched(self, x_t, t_batch):
        # predict_with_guidance also threads return_cache through.
        net = BatchedStubNet()
        g = GuidanceConfig()
        g.cfg.scale = 3.0
        out, cache = predict_with_guidance(
            net,
            x_t,
            t_batch,
            batched=True,
            guidance_cfg=g,
            condition=["pos"] * x_t.shape[0],
            neg_condition=["neg"] * x_t.shape[0],
            cat_fn=lambda neg, pos: list(neg) + list(pos),
            return_cache=True,
            output_dtype=torch.float64,
        )
        assert {"pos", "neg"} <= set(cache.keys())
        assert out.dtype == torch.float64
        assert cache["pos"].dtype == cache["neg"].dtype == torch.float64
        assert torch.equal(cache["pos"], torch.full_like(x_t, 1.0, dtype=torch.float64))
        assert torch.equal(cache["neg"], torch.full_like(x_t, -1.0, dtype=torch.float64))


# ───────────────── network-owned full guided prediction ──────────────────────


@pytest.mark.parametrize("batched", [False, True])
def test_network_guidance_owns_recipe_and_dispatcher_owns_finalization(batched):
    class OwnedGuidance(FunctionNet):
        def native(self, *args, **kwargs):
            raise AssertionError("Only the recipe may request forwards")

        def predict_with_guidance(self, ctx):
            self.received = ctx
            return result

        def transform_output(self, prediction, x_t, t, condition=None, fwd_pred_type=None, output_dtype="same", **kw):
            self.finalized = (prediction, x_t, t, condition, fwd_pred_type, output_dtype, kw)
            return (prediction + 2).to(output_dtype)

    net = OwnedGuidance()
    x, t = torch.zeros(2, 4), torch.zeros(2)
    cfg = NetworkGuidance()
    cfg.perturb.scale = 3.0  # The standard recipe would reject this network term.
    cfg.rescale = 1.0
    # The recipe returns the working dtype; the transform rounds to the requested one.
    result = torch.ones_like(x, dtype=torch.float64)
    cache = {"pos": torch.zeros_like(x, dtype=torch.float64)}
    contexts = {"pos": {"tag": "custom-context"}}

    actual, actual_cache = predict_with_guidance(
        net,
        x,
        t,
        batched=batched,
        guidance_cfg=cfg,
        condition="positive",
        neg_condition="negative",
        precomputed=cache,
        contexts=contexts,
        return_cache=True,
        skip_inactive=False,
        output_dtype=torch.float32,
        fwd_pred_type="x0",
    )
    torch.testing.assert_close(actual, (result + 2).float())
    ctx = net.received
    assert ctx.net is net and ctx.x_t is x and ctx.t is t
    assert ctx.guidance_cfg is cfg and ctx.contexts is contexts
    assert ctx.condition == "positive" and ctx.neg_condition == "negative"
    assert not ctx.skip_inactive and ctx.batched == batched
    assert ctx.output_dtype == torch.float32 and ctx.dtype == torch.float64
    assert ctx.fwd_pred_type == "x0" and ctx.fwd_kwargs == {}
    assert actual_cache is ctx.cache and actual_cache["pos"] is cache["pos"]
    assert net.finalized[0] is result and net.finalized[1] is x and net.finalized[2] is t
    assert net.finalized[3:6] == ("positive", "x0", torch.float32)
    assert net.finalized[6] == {"apply_output_conditioning": True, "tag": "custom-context"}


@pytest.fixture(params=["sit", "dit", "qwen"])
def owned_guidance_net(request):
    from fastgen.networks.DiT.network import DiT
    from fastgen.networks.QwenImage.network import QwenImage

    class TinyNet(FunctionNet):
        # Use the real guided-prediction methods without loading model weights.
        predict_with_guidance = (
            QwenImage.predict_with_guidance if request.param == "qwen" else DiT.predict_with_guidance
        )
        _pack_latents = QwenImage._pack_latents
        _unpack_latents = QwenImage._unpack_latents

        def __init__(self):
            super().__init__(net_pred_type="x0")
            self.kind = request.param
            self.use_sit_convention = request.param == "sit"
            self.cfg_channels = 3 if request.param == "sit" else None  # as the SiT reference configs
            self.calls = []

        def forward(self, x_t, t, condition=None, **kwargs):
            self.calls.append((condition, kwargs))
            return super().forward(x_t, t, condition=condition, **kwargs)

        def native(self, x_t, t, condition=None, **kwargs):
            pattern = torch.arange(x_t[0].numel(), dtype=x_t.dtype).reshape(1, *x_t.shape[1:]) / 16
            return x_t + pattern + condition.reshape(-1, 1, 1, 1)

    return TinyNet()


def _owned_inputs():
    x = torch.zeros(2, 4, 4, 4, dtype=torch.bfloat16)
    t = torch.tensor([0.25, 0.75], dtype=torch.float64)
    return x, t, torch.ones(2, 1, dtype=x.dtype), torch.zeros(2, 1, dtype=x.dtype)


def _expected_owned(net, cond, neg, scale, cfg):
    scale = scale.to(cond.dtype)
    guided = cond + (scale.reshape(-1, 1, 1, 1) - 1) * (cond - neg)
    if net.kind == "sit":
        guided[:, 3:] = cond[:, 3:]
    if net.kind == "qwen":
        return net._unpack_latents(
            cfg_rescale(
                net._pack_latents(guided),
                net._pack_latents(cond),
                cfg.rescale,
                rescale_type=cfg.rescale_type,
                rescale_dims=cfg.rescale_dims,
            ),
            guided.shape[-2],
            guided.shape[-1],
        )
    return cfg_rescale(guided, cond, cfg.rescale, rescale_type=cfg.rescale_type, rescale_dims=cfg.rescale_dims)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("rescale", [None, 0.7])
def test_real_network_guidance_numeric(owned_guidance_net, batched, rescale):
    net = owned_guidance_net
    x, t, cond, neg = _owned_inputs()
    cfg = GuidanceConfig()
    cfg.cfg.scale, cfg.cfg.boundary = [1.0, 3.0], 0.5
    cfg.rescale, cfg.rescale_type, cfg.rescale_dims = rescale, "norm", (-1,)
    actual, cache = predict_with_guidance(
        net,
        x,
        t,
        batched=batched,
        guidance_cfg=cfg,
        condition=cond,
        neg_condition=neg,
        output_dtype=torch.float64,
        return_cache=True,
    )
    expected = _expected_owned(net, cache["pos"], cache["neg"], torch.tensor([1.0, 3.0]), cfg)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == cache["pos"].dtype == cache["neg"].dtype == torch.float64
    assert len(net.calls) == (1 if batched else 2)


@pytest.mark.parametrize("batched", [False, True])
def test_common_forward_kwargs_reach_every_branch_forward(batched):
    calls = []

    def native(x, t, condition=None, **kwargs):
        calls.append(kwargs)
        return x + condition.reshape(-1, 1)

    cfg = GuidanceConfig()
    cfg.cfg.scale = 3.0
    x, t = torch.zeros(2, 4), torch.zeros(2, dtype=torch.float64)
    net = FunctionNet(native)
    result = predict_with_guidance(net, x, t, torch.ones(2), cfg, torch.zeros(2), batched=batched, tag="common")
    torch.testing.assert_close(result, torch.full_like(x, 3))
    expected_kwargs = [{"tag": "common"}]
    if not batched:
        expected_kwargs.append({"tag": "common", "skip_layers": None})
    assert calls == expected_kwargs


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen_cfg_packing_commutes_but_token_norm_requires_packing(dtype):
    from fastgen.networks.QwenImage.network import QwenImage

    class LatentLayout:
        pack = QwenImage._pack_latents
        unpack = QwenImage._unpack_latents

    generator = torch.Generator().manual_seed(42)
    pos = torch.randn(2, 4, 4, 6, generator=generator).to(dtype)
    neg = torch.randn(2, 4, 4, 6, generator=generator).to(dtype)
    scale = torch.tensor([2.0, 4.0], dtype=dtype)
    guided = pos + (scale.reshape(-1, 1, 1, 1) - 1) * (pos - neg)
    layout = LatentLayout()
    pack, unpack = layout.pack, layout.unpack
    packed_pos, packed_neg = pack(pos), pack(neg)
    packed_guided = packed_pos + (scale.reshape(-1, 1, 1) - 1) * (packed_pos - packed_neg)
    torch.testing.assert_close(pack(guided), packed_guided, rtol=0, atol=0)
    expected = unpack(cfg_rescale(packed_guided, packed_pos, 1.0, rescale_type="norm", rescale_dims=(-1,)), 4, 6)
    actual = unpack(cfg_rescale(pack(guided), pack(pos), 1.0, rescale_type="norm", rescale_dims=(-1,)), 4, 6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    wrong_axes = cfg_rescale(guided, pos, 1.0, rescale_type="norm", rescale_dims=(-1,))
    assert not torch.allclose(wrong_axes, expected, rtol=0.01, atol=0.01)


def test_real_network_guidance_cached_teacher_and_contexts(owned_guidance_net):
    net = owned_guidance_net
    x, t, cond, neg = _owned_inputs()
    cfg = GuidanceConfig()
    cfg.cfg.scale = 3.0
    positive = net(x, t, condition=cond).to(torch.bfloat16)
    net.calls.clear()
    actual, cache = predict_with_guidance(
        net,
        x,
        t,
        guidance_cfg=cfg,
        condition=cond,
        neg_condition=neg,
        precomputed={"pos": positive},
        contexts={"neg": {"tag": "teacher"}},
        skip_inactive=False,
        fwd_pred_type="x0",
        output_dtype=torch.float32,
        return_cache=True,
    )
    assert len(net.calls) == 1
    assert net.calls[0][1]["tag"] == "teacher" and "fwd_pred_type" not in net.calls[0][1]
    expected = _expected_owned(net, positive.float(), cache["neg"].float(), torch.full((2,), 3.0), cfg)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("skip_inactive", [False, True])
def test_real_network_guidance_inactive_cfg(owned_guidance_net, skip_inactive):
    net = owned_guidance_net
    x, t, cond, neg = _owned_inputs()
    cfg = GuidanceConfig()
    cfg.cfg.scale = 1.0
    actual, cache = predict_with_guidance(
        net,
        x,
        t,
        guidance_cfg=cfg,
        condition=cond,
        neg_condition=neg,
        skip_inactive=skip_inactive,
        return_cache=True,
    )
    torch.testing.assert_close(actual, cache["pos"].to(actual.dtype))
    assert len(net.calls) == (1 if skip_inactive else 2)


def test_real_network_guidance_mix_rescale_last(owned_guidance_net):
    net = owned_guidance_net
    x, t, cond, neg = _owned_inputs()
    cfg = GuidanceConfig()
    cfg.cfg.scale = 3.0
    cfg.mix.scale = 1.5
    cfg.rescale, cfg.rescale_type, cfg.rescale_dims = 0.7, "norm", (-1,)
    positive = torch.arange(x.numel(), dtype=x.dtype).reshape(x.shape) / 32
    cache = {"pos": positive, "neg": positive.flip(-1), "pos_mix": positive + 3}
    actual = predict_with_guidance(net, x, t, guidance_cfg=cfg, condition=cond, neg_condition=neg, precomputed=cache)
    guided = cache["pos"] + 2 * (cache["pos"] - cache["neg"])
    if net.kind == "sit":
        guided[:, 3:] = cache["pos"][:, 3:]
    guided += 0.5 * (cache["pos_mix"] - cache["neg"])
    if net.kind == "qwen":
        expected = net._unpack_latents(
            cfg_rescale(
                net._pack_latents(guided), net._pack_latents(positive), 0.7, rescale_type="norm", rescale_dims=(-1,)
            ),
            4,
            4,
        )
    else:
        expected = cfg_rescale(guided, positive, 0.7, rescale_type="norm", rescale_dims=(-1,))
    torch.testing.assert_close(actual, expected)
    assert not net.calls


@pytest.mark.parametrize("with_rescale", [False, True])
def test_real_network_guidance_without_cfg(owned_guidance_net, with_rescale):
    net = owned_guidance_net
    x, t, cond, _ = _owned_inputs()
    cfg = GuidanceConfig() if with_rescale else None
    if cfg is not None:
        cfg.rescale, cfg.rescale_type, cfg.rescale_dims = 1.0, "norm", (-1,)
    actual, cache = predict_with_guidance(net, x, t, guidance_cfg=cfg, condition=cond, return_cache=True)
    torch.testing.assert_close(actual, cache["pos"].to(actual.dtype))
    assert len(net.calls) == 1


def test_network_guidance_runs_inside_shared_teacher_context(owned_guidance_net):
    from contextlib import contextmanager
    from fastgen.networks.lora_utils import SharedBaseTeacher

    net = owned_guidance_net
    x, t, cond, neg = _owned_inputs()
    original_cfg_channels = net.cfg_channels
    net.base_model_overrides = {"cfg_channels": 3}
    observations = []

    @contextmanager
    def adapters_disabled():
        observations.append(net.training)
        yield

    net.adapters_disabled = adapters_disabled
    teacher = SharedBaseTeacher(net)
    cfg = GuidanceConfig()
    cfg.cfg.scale = 3.0
    actual, cache = predict_with_guidance(
        teacher,
        x,
        t,
        guidance_cfg=cfg,
        condition=cond,
        neg_condition=neg,
        return_cache=True,
    )
    expected = cache["pos"] + 2 * (cache["pos"] - cache["neg"])
    if net.kind != "qwen":
        expected[:, 3:] = cache["pos"][:, 3:]
    torch.testing.assert_close(actual, expected.to(actual.dtype))
    # The custom recipe, branches included, and the final transform each run once under the teacher context.
    assert observations == [False, False]
    assert net.training and net.cfg_channels == original_cfg_channels


@pytest.mark.parametrize("term", ["stg", "modality", "skip_layers"])
@pytest.mark.parametrize("batched", [False, True])
def test_network_rejects_unsupported_guidance_before_forwards(owned_guidance_net, term, batched):
    net = owned_guidance_net
    x, t, cond, neg = _owned_inputs()
    # Another network's recipe: LTX-2's terms are not DiT's or Qwen-Image's.
    _, LTX2GuidanceConfig = _import_ltx2()
    cfg = LTX2GuidanceConfig()
    if term == "skip_layers":
        cfg.cfg.scale = 3.0
        cfg.cfg.skip_layers = [1]
    else:
        # Reject even when the current time window would make the term inactive.
        getattr(cfg, term).scale = [1.0, 3.0]
        getattr(cfg, term).boundary = 0.9
    with pytest.raises(NotImplementedError, match=term):
        predict_with_guidance(
            net,
            x,
            t,
            guidance_cfg=cfg,
            condition=cond,
            neg_condition=neg,
            batched=batched,
            precomputed={"stg": torch.zeros_like(x)},
        )
    assert net.calls == []


@pytest.mark.parametrize("batched", [False, True])
def test_network_ignores_skip_layers_without_a_cfg_term(owned_guidance_net, batched):
    """``cfg.skip_layers`` reach only the negative forward, so without a CFG scale they need no support."""
    net = owned_guidance_net
    x, t, cond, neg = _owned_inputs()
    cfg = GuidanceConfig()
    cfg.cfg.skip_layers = [1]
    actual, cache = predict_with_guidance(
        net, x, t, guidance_cfg=cfg, condition=cond, neg_condition=neg, batched=batched, return_cache=True
    )
    torch.testing.assert_close(actual, cache["pos"].to(actual.dtype))
    assert len(net.calls) == 1


@pytest.mark.parametrize("cached", [False, True])
def test_batched_context_preserves_cached_branches_and_forward_contexts(cached):
    net = BatchedStubNet()
    x, t = torch.zeros(2, 4), torch.tensor([0.25, 0.75], dtype=torch.float64)
    cfg = GuidanceConfig()
    cfg.cfg.scale = 3.0
    actual, cache = predict_with_guidance(
        net,
        x,
        t,
        guidance_cfg=cfg,
        condition="pos",
        neg_condition="neg",
        batched=True,
        precomputed={"pos": torch.full_like(x, 5.0)} if cached else None,
        contexts={"neg": {"tag": "negative-context"}},
        return_cache=True,
    )
    assert len(net.calls) == (1 if cached else 2)
    assert net.calls[-1]["tag"] == "negative-context"
    torch.testing.assert_close(actual, (cache["pos"] + 2 * (cache["pos"] - cache["neg"])).to(actual.dtype))
    if cached:
        torch.testing.assert_close(cache["pos"], torch.full_like(x, 5.0, dtype=torch.float64))


@pytest.mark.parametrize("skip_inactive", [False, True])
def test_batched_context_honors_inactive_cfg(skip_inactive):
    net = BatchedStubNet()
    x, t = torch.zeros(2, 4), torch.tensor([0.25, 0.75], dtype=torch.float64)
    cfg = GuidanceConfig()
    cfg.cfg.scale = 1.0
    _, cache = predict_with_guidance(
        net,
        x,
        t,
        guidance_cfg=cfg,
        condition=["pos"] * 2,
        neg_condition=["neg"] * 2,
        batched=True,
        cat_fn=lambda neg, pos: neg + pos,
        skip_inactive=skip_inactive,
        return_cache=True,
    )
    assert len(net.calls) == 1
    assert ("neg" in cache) == (not skip_inactive)
    assert len(net.calls[0]["condition"]) == (2 if skip_inactive else 4)


def test_guidance_blocks_can_be_composed_by_networks(x_t_dict, t_batch):
    cfg = GuidanceConfig()
    cfg.cfg.scale = {"video": 2.0, "audio": 3.0}
    cfg.mix.scale = 1.25
    cfg.rescale, cfg.rescale_type = 0.5, "norm"
    cached = {
        key: {name: torch.arange(x.numel()).reshape(x.shape).float() + offset for name, x in x_t_dict.items()}
        for key, offset in [("pos", 2), ("neg", -1), ("pos_mix", 5)]
    }

    def native(*args, **kwargs):
        raise AssertionError("All block forwards are cached")

    net = FunctionNet(native, modalities=tuple(x_t_dict))
    ctx = get_guidance_context(net, x_t_dict, t_batch, guidance_cfg=cfg, condition="pos", precomputed=cached)
    result = ctx.predict("pos")
    for block in (apply_cfg, apply_mix):
        result = block(ctx, result)
    result = apply_rescale(ctx, result)
    for name in x_t_dict:
        # Blocks compose in the working dtype.
        pos, neg = cached["pos"][name].double(), cached["neg"][name].double()
        expected = pos + (cfg.cfg.scale[name] - 1) * (pos - neg)
        expected = expected + 0.25 * (cached["pos_mix"][name].double() - neg)
        torch.testing.assert_close(result[name], cfg_rescale(expected, pos, 0.5, rescale_type="norm"))


def test_owned_guidance_applies_its_output_constraint_once_after_its_recipe(owned_guidance_net):
    net = owned_guidance_net
    x, t, pos, neg = _owned_inputs()
    cfg = GuidanceConfig()
    cfg.cfg.scale = 3.0
    cfg.rescale = 0.7
    cfg.rescale_type = "norm" if net.kind == "qwen" else "std"
    raw, cache = predict_with_guidance(net, x, t, pos, cfg, neg, return_cache=True)
    constrained = []

    def constraint(prediction, x_t, t, condition=None, **context):
        constrained.append((prediction, x_t, t, condition))
        return prediction * 0.5 + 2

    net._condition_output = constraint
    net.calls.clear()
    actual, actual_cache = predict_with_guidance(net, x, t, pos, cfg, neg, return_cache=True)
    composed = constrained[0][0]
    torch.testing.assert_close(composed.to(raw.dtype), raw, rtol=0, atol=0)
    torch.testing.assert_close(actual, (composed * 0.5 + 2).to(actual.dtype), rtol=0, atol=0)
    assert len(constrained) == 1 and constrained[0][1] is x and constrained[0][3] is pos
    assert all(
        kwargs["apply_output_conditioning"] is False and kwargs["output_dtype"] == torch.float64
        for _, kwargs in net.calls
    )
    for key in cache:
        torch.testing.assert_close(actual_cache[key], cache[key], rtol=0, atol=0)
