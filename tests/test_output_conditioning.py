# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output constraints apply once, in ``transform_output`` after guidance, over cache-safe native predictions."""

from types import SimpleNamespace

import pytest
import torch

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import CompositeNoiseSchedule, RFNoiseSchedule
from fastgen.utils.guidance import (
    GuidanceConfig,
    apply_cfg,
    apply_rescale,
    cfg_rescale,
    predict_with_guidance,
)
from fastgen.utils.tree_utils import per_modality


class BlendedNet(FastGenNetwork):
    """A differentiable soft constraint independent of any I2V layout."""

    def __init__(self, strength=0.4):
        super().__init__(net_pred_type="x0", schedule_type="rf")
        self.strength = strength
        self.calls = []
        self.finalizations = []

    def forward(
        self, x_t, t, condition, fwd_pred_type=None, apply_output_conditioning=True, output_dtype="same", **kwargs
    ):
        self.calls.append((apply_output_conditioning, kwargs))
        p = x_t.square() + condition
        if kwargs.get("skip_self_attn_layers"):
            p = p * 0.7
        if kwargs.get("skip_cross_modal_attn"):
            p = p * 0.6
        return self.transform_output(
            p, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning, **kwargs
        )

    def _condition_output(self, prediction, x_t, t, condition, **context):
        self.finalizations.append((x_t, t, condition, context))
        return (1 - self.strength) * prediction + self.strength * 2

    def sample(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.parametrize("strength", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("rescale_type", ["std", "norm"])
@pytest.mark.parametrize("batched", [False, True])
def test_guidance_finalizes_once_and_reusable_cache_is_unconditioned(strength, rescale_type, batched):
    net = BlendedNet(strength)
    x = torch.arange(12, dtype=torch.float64).view(2, 6) / 10
    t = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    pos, neg = torch.ones_like(x), -torch.ones_like(x)
    g = GuidanceConfig()
    g.cfg.scale, g.mix.scale, g.rescale, g.rescale_type = 3.0, 1.5, 0.7, rescale_type
    p, n = x.square() + pos, x.square() + neg
    expected = (1 - strength) * cfg_rescale(p + 2.5 * (p - n), p, 0.7, rescale_type=rescale_type) + strength * 2
    actual, cache = predict_with_guidance(net, x, t, pos, g, neg, batched=batched, return_cache=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(cache["pos"], p)
    assert all(flag is False for flag, _ in net.calls)
    assert len(net.finalizations) == 1
    saved = {k: v.clone() for k, v in cache.items()}
    calls = len(net.calls)
    again = predict_with_guidance(net, x, t, pos, g, neg, precomputed=cache, batched=batched)
    assert len(net.calls) == calls and len(net.finalizations) == 2
    torch.testing.assert_close(again, actual, rtol=0, atol=0)
    for key in saved:
        torch.testing.assert_close(cache[key], saved[key], rtol=0, atol=0)


def test_no_guidance_finalizes_and_retains_gradient():
    net = BlendedNet()
    x = torch.randn(2, 6, requires_grad=True)
    t, condition = torch.full((2,), 0.5, dtype=net.noise_scheduler.t_precision), torch.ones_like(x)
    direct = net(x, t, condition)
    actual = predict_with_guidance(net, x, t, condition)
    # The forward transforms in the state dtype and guidance in the working dtype; both round once.
    torch.testing.assert_close(actual, direct)
    actual.sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach() * 0.6)


@pytest.mark.parametrize("stage", ["recipe", "constraint"])
@pytest.mark.parametrize(
    "output_dtype,changed_dtype", [(torch.float32, torch.bfloat16), ("same", torch.float32), (None, torch.float32)]
)
def test_guidance_rejects_recipes_and_constraints_that_change_the_dtype(stage, output_dtype, changed_dtype):
    def change_dtype(prediction, *args, **kwargs):
        return prediction.to(changed_dtype)

    net = BlendedNet()
    if stage == "recipe":
        net.predict_with_guidance = lambda ctx: change_dtype(ctx.predict("pos", condition=ctx.condition))
    else:
        net._condition_output = change_dtype
    x, t = torch.ones(2, 6, dtype=torch.bfloat16), torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    match = r"prediction\.dtype" if stage == "recipe" else r"conditioned\.dtype"
    with pytest.raises(AssertionError, match=match):
        predict_with_guidance(net, x, t, torch.ones_like(x), output_dtype=output_dtype)


@pytest.mark.parametrize("source", ["precomputed", "contexts", "recipe"])
def test_guidance_rejects_unknown_branch_keys(source):
    net = BlendedNet()
    x, t = torch.ones(2, 6), torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    kwargs = {}
    if source == "precomputed":
        kwargs["precomputed"] = {"uncond": x}
    elif source == "contexts":
        kwargs["contexts"] = {"uncond": {}}
    else:
        net.predict_with_guidance = lambda ctx: ctx.predict("uncond", condition=ctx.condition)
    with pytest.raises(ValueError, match="unknown guidance branch keys"):
        predict_with_guidance(net, x, t, x, **kwargs)


@pytest.mark.parametrize("source", ["forward", "precomputed"])
def test_guidance_composes_branches_of_any_dtype_in_the_working_dtype(source):
    net = BlendedNet(strength=0.0)
    x, t = torch.zeros(2, 6, dtype=torch.bfloat16), torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    branch = torch.full((2, 6), 1 / 3)  # FP32 beside a BF16 state
    condition = branch if source == "forward" else torch.zeros_like(x)
    precomputed = {"pos": branch} if source == "precomputed" else None
    actual, cache = predict_with_guidance(net, x, t, condition, precomputed=precomputed, return_cache=True)
    torch.testing.assert_close(cache["pos"], branch.double(), rtol=0, atol=0)
    torch.testing.assert_close(actual, branch.to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize(
    "state_dtype,output_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float64, torch.float32),
    ],
)
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("owned_recipe", [False, True])
@pytest.mark.parametrize("rescale_type", ["std", "norm"])
def test_guidance_composes_in_the_working_dtype_and_reuses_its_cache(
    state_dtype, output_dtype, batched, owned_recipe, rescale_type
):
    net = BlendedNet(strength=0.4)
    x = torch.linspace(-0.8, 1.3, 12, dtype=state_dtype).reshape(2, 6).requires_grad_()
    pos = torch.linspace(0.1, 0.7, 12, dtype=state_dtype).reshape_as(x)
    neg = -torch.linspace(0.3, 0.8, 12, dtype=state_dtype).reshape_as(x)
    t = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    g = GuidanceConfig()
    g.cfg.scale, g.rescale, g.rescale_type = 2.375, 0.7, rescale_type
    if owned_recipe:
        net.predict_with_guidance = lambda ctx: apply_rescale(
            ctx, apply_cfg(ctx, ctx.predict("pos", condition=ctx.condition))
        )
    seen = []
    constrain = net._condition_output

    def record(prediction, x_t, t, condition, **kwargs):
        seen.append(prediction)
        assert x_t is x and condition is pos
        return constrain(prediction, x_t, t, condition, **kwargs)

    net._condition_output = record
    raw_pos, raw_neg = (x.square() + pos).double(), (x.square() + neg).double()

    def expected(p, n):
        # The branches, the composition and the constraint share the working dtype; the result rounds once.
        guided = cfg_rescale(p + 1.375 * (p - n), p, 0.7, rescale_type=rescale_type)
        return guided, 0.6 * guided + 0.8

    guided, conditioned = expected(raw_pos, raw_neg)
    actual, cache = predict_with_guidance(
        net, x, t, pos, g, neg, batched=batched, output_dtype=output_dtype, return_cache=True
    )
    torch.testing.assert_close(seen[0], guided, rtol=0, atol=0)
    torch.testing.assert_close(actual, conditioned.to(output_dtype), rtol=0, atol=0)
    torch.testing.assert_close(cache["pos"], raw_pos, rtol=0, atol=0)
    torch.testing.assert_close(cache["neg"], raw_neg, rtol=0, atol=0)
    assert len(seen) == 1 and seen[0].dtype == torch.float64
    assert actual.dtype == output_dtype and actual.requires_grad

    # Reusing the cache at another dtype skips both forwards and rounds the same composition there.
    calls = len(net.calls)
    again, reused = predict_with_guidance(
        net, x, t, pos, g, neg, precomputed=cache, output_dtype=torch.float64, return_cache=True
    )
    torch.testing.assert_close(again, conditioned, rtol=0, atol=0)
    assert len(net.calls) == calls and len(seen) == 2
    assert reused["pos"] is cache["pos"] and reused["neg"] is cache["neg"]
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("output_dtype", ["same", None, torch.float32])
def test_multimodal_guidance_rounds_to_the_requested_or_each_state_dtype(output_dtype):
    x = {
        "video": torch.linspace(0.1, 1, 12, dtype=torch.bfloat16).reshape(2, 6),
        "audio": torch.linspace(0.2, 1.3, 8, dtype=torch.float64).reshape(2, 4),
    }
    pos, neg = per_modality(torch.ones_like, x), per_modality(lambda v: -torch.ones_like(v), x)
    seen = []

    class MultimodalNet(FastGenNetwork):
        def __init__(self):
            super().__init__(net_pred_type="x0", schedule_type="rf")
            self.noise_scheduler = CompositeNoiseSchedule(schedules={m: {"name": "rf"} for m in x})

        def forward(
            self, x_t, t, condition=None, fwd_pred_type=None, apply_output_conditioning=True, output_dtype="same", **kw
        ):
            assert apply_output_conditioning is False
            prediction = per_modality(lambda state, cond: state.square() + cond, x_t, condition)
            return self.transform_output(
                prediction, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
            )

        def _condition_output(self, prediction, x_t, t, condition=None, **context):
            seen.append(prediction)
            assert x_t is x and condition is pos
            return per_modality(lambda pred, clean: 0.6 * pred + 0.4 * clean.to(pred.dtype), prediction, condition)

        def sample(self, *args, **kwargs):
            raise NotImplementedError

    net = MultimodalNet()
    t = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    g = GuidanceConfig()
    g.cfg.scale, g.rescale = 2.375, 0.7
    actual, cache = predict_with_guidance(net, x, t, pos, g, neg, output_dtype=output_dtype, return_cache=True)
    assert len(seen) == 1
    for key, state in x.items():
        p, n = (state.square() + pos[key]).double(), (state.square() + neg[key]).double()
        guided = cfg_rescale(p + 1.375 * (p - n), p, 0.7)
        # None keeps the working dtype, FP64 here.
        dtype = state.dtype if output_dtype == "same" else output_dtype or torch.float64
        expected = (0.6 * guided + 0.4 * pos[key].double()).to(dtype)
        torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)
        assert seen[0][key].dtype == cache["pos"][key].dtype == cache["neg"][key].dtype == torch.float64


def test_finalizer_uses_positive_context_and_rejects_branch_flag_overrides():
    net = BlendedNet()
    x, t = torch.ones(2, 6), torch.full((2,), 0.5, dtype=net.noise_scheduler.t_precision)
    g = GuidanceConfig()
    g.cfg.scale = 3
    positive = torch.full_like(x, 4)
    marker = object()
    predict_with_guidance(
        net,
        x,
        t,
        torch.ones_like(x),
        g,
        -x,
        contexts={"pos": {"condition": positive, "causal_ctx": marker}},
        batched=True,
    )
    assert net.finalizations[0][2] is positive
    assert net.finalizations[0][3]["causal_ctx"] is marker
    with pytest.raises(ValueError, match="guidance owns"):
        predict_with_guidance(net, x, t, x, contexts={"neg": {"apply_output_conditioning": True}})
    with pytest.raises(ValueError, match="guidance owns"):
        predict_with_guidance(net, x, t, x, contexts={"pos": {"output_dtype": torch.float32}})
    with pytest.raises(ValueError, match="guidance owns"):
        predict_with_guidance(net, x, t, x, contexts={"pos": {"fwd_pred_type": "x0"}})


def test_guidance_can_skip_the_final_output_constraint():
    net = BlendedNet()
    x = torch.arange(12, dtype=torch.float64).view(2, 6) / 10
    t = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    pos, neg = torch.ones_like(x), -torch.ones_like(x)
    g = GuidanceConfig()
    g.cfg.scale = 3.0
    actual, cache = predict_with_guidance(
        net, x, t, pos, g, neg, output_dtype=torch.float32, return_cache=True, apply_output_conditioning=False
    )
    p, n = x.square() + pos, x.square() + neg
    torch.testing.assert_close(actual, (p + 2.0 * (p - n)).float(), rtol=0, atol=0)
    torch.testing.assert_close(cache["pos"], p, rtol=0, atol=0)
    assert net.finalizations == []
    assert [flag for flag, _ in net.calls] == [False, False]


def _constraint_network(kind):
    from fastgen.networks.WanI2V.network import WanI2V
    from fastgen.networks.WanI2V.network_cam import WanCamI2V
    from fastgen.networks.WanI2V.network_causal import CausalWanI2V
    from fastgen.networks.cosmos_predict2.network import CosmosPredict2
    from fastgen.networks.cosmos3.network import Cosmos3
    from fastgen.networks.LTX2.network import LTX2

    cls = {
        "wan": WanI2V,
        "camera": WanCamI2V,
        "causal": CausalWanI2V,
        "cosmos2": CosmosPredict2,
        "cosmos3": Cosmos3,
        "ltx": LTX2,
    }[kind]
    net = cls.__new__(cls)
    torch.nn.Module.__init__(net)
    net.net_pred_type, net.noise_scheduler = "flow", RFNoiseSchedule()
    net.concat_mask = False
    return net


@pytest.mark.parametrize("kind", ["wan", "camera", "causal", "cosmos2", "cosmos3", "ltx"])
@pytest.mark.parametrize("pred_type", ["x0", "eps", "v", "flow"])
@pytest.mark.parametrize(
    "state_dtype,output_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float16),
        (torch.float32, torch.float32),
        (torch.float32, torch.float64),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
        (torch.float64, torch.float32),
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
    ],
)
def test_network_constraints_match_clean_target_and_preserve_endpoints(kind, pred_type, state_dtype, output_dtype):
    net = _constraint_network(kind)
    # A native prediction of the requested type isolates the constraint from the conversion.
    net.net_pred_type = pred_type
    if pred_type == "v":
        from fastgen.networks.noise_schedule import AlphasNoiseSchedule

        net.noise_scheduler = AlphasNoiseSchedule(torch.linspace(1, 0.01, 1000))
    torch.manual_seed(4)
    x = torch.randn(2, 2, 3, 2, 2, dtype=state_dtype)
    out = torch.randn_like(x, requires_grad=True)
    clean = torch.randn_like(x[:, :, :1])
    mask = torch.zeros(2, 1, 3, 1, 1, dtype=torch.bool)
    mask[:, :, 0] = 1
    t = torch.tensor([0.0, 0.5], dtype=net.noise_scheduler.t_precision)
    cond = {
        "text_embeds": None,
        "first_frame_cond": clean,
        "frame_cond": clean,
        "conditioning_latents": clean,
        "condition_mask": mask,
    }
    saved = out.detach().clone()
    saved_conditions = {key: value.clone() for key, value in cond.items() if isinstance(value, torch.Tensor)}
    if pred_type != "x0":
        # No non-x0 prediction carries the clean frame at sigma = 0.
        with pytest.raises(ValueError, match="sigma = 0"):
            net._condition_output(out.to(output_dtype), x, t, cond, pred_type)
        with pytest.raises(ValueError, match="sigma = 0"):
            predict_with_guidance(
                net, x, t, cond, precomputed={"pos": out}, output_dtype=output_dtype, fwd_pred_type=pred_type
            )
        t = torch.tensor([0.25, 0.5], dtype=net.noise_scheduler.t_precision)
    # Positional and keyword calls share the same constraint contract.
    direct = net._condition_output(out.to(output_dtype), x, t, cond, pred_type)
    assert direct.dtype == output_dtype
    actual = predict_with_guidance(
        net, x, t, cond, precomputed={"pos": out}, output_dtype=output_dtype, fwd_pred_type=pred_type
    )
    torch.testing.assert_close(actual, direct, rtol=0, atol=0)
    if pred_type == "x0":
        torch.testing.assert_close(actual[:, :, :1], clean.to(output_dtype), rtol=0, atol=0)
    else:
        expected = net.noise_scheduler.convert_model_output(
            x[:, :, :1], clean, t, src_pred_type="x0", target_pred_type=pred_type, output_dtype=output_dtype
        )
        torch.testing.assert_close(actual[:, :, :1], expected)
    torch.testing.assert_close(actual[:, :, 1:], out[:, :, 1:].to(output_dtype), rtol=0, atol=0)
    torch.testing.assert_close(out, saved, rtol=0, atol=0)
    assert actual.dtype == output_dtype and torch.isfinite(actual).all()
    assert cond["first_frame_cond"] is clean and clean.dtype == x.dtype
    for key, value in saved_conditions.items():
        torch.testing.assert_close(cond[key], value, rtol=0, atol=0)
    actual.sum().backward()
    assert torch.equal(out.grad[:, :, 1:], torch.ones_like(out.grad[:, :, 1:]))


@pytest.mark.parametrize("pred_type", ["x0", "flow"])
@pytest.mark.parametrize(
    "state_dtype,prediction_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float16),
    ],
)
def test_ltx_output_blend_promotes_operands_before_arithmetic(state_dtype, prediction_dtype, pred_type):
    net = _constraint_network("ltx")
    x = torch.zeros(1, 2, 3, 2, 2, dtype=state_dtype)
    prediction = torch.linspace(-1.234, 2.345, x.numel(), dtype=prediction_dtype).reshape_as(x)
    clean = torch.full_like(x[:, :, :1], 1.234)
    strength = torch.tensor([0.4], dtype=torch.float32)
    condition = {"text_embeds": None, "frame_cond": clean, "strength": strength}
    blend_dtype = torch.promote_types(torch.promote_types(state_dtype, prediction_dtype), strength.dtype)
    weight = (1 - strength).reshape(1, 1, 1, 1, 1).to(dtype=blend_dtype)
    t = torch.full((1,), 0.5, dtype=net.noise_scheduler.t_precision)
    target = (
        clean.to(blend_dtype)
        if pred_type == "x0"
        else net.noise_scheduler.convert_model_output(
            x[:, :, :1], clean, t, src_pred_type="x0", target_pred_type=pred_type, output_dtype=blend_dtype
        )
    )
    expected = prediction.clone()
    expected[:, :, :1] = (prediction[:, :, :1].to(blend_dtype) * weight + target * (1 - weight)).to(prediction_dtype)
    saved = {key: value.clone() for key, value in condition.items() if isinstance(value, torch.Tensor)}
    saved_prediction = prediction.clone()
    actual = net._condition_output(prediction, x, t, condition, pred_type)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(prediction, saved_prediction, rtol=0, atol=0)
    for key, value in saved.items():
        torch.testing.assert_close(condition[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["wan", "camera", "causal", "cosmos2", "cosmos3"])
@pytest.mark.parametrize("pred_type", ["x0", "flow"])
@pytest.mark.parametrize("invalid_prediction", [float("nan"), float("inf"), -float("inf")])
def test_hard_constraints_select_clean_targets_over_nonfinite_predictions(kind, pred_type, invalid_prediction):
    net = _constraint_network(kind)
    x = torch.zeros(2, 2, 3, 2, 2, dtype=torch.bfloat16)
    prediction = torch.randn(x.shape, dtype=torch.float32)
    prediction[:, :, :1] = invalid_prediction
    clean = torch.ones_like(x[:, :, :1])
    mask = torch.zeros(2, 1, 3, 1, 1, dtype=torch.bool)
    mask[:, :, :1] = True
    condition = {"first_frame_cond": clean, "conditioning_latents": clean, "condition_mask": mask}
    # x0 is constrained at the clean end too; a non-x0 prediction only where it is noised.
    t = torch.tensor([0 if pred_type == "x0" else 0.5, 0.5], dtype=net.noise_scheduler.t_precision)
    actual = net._condition_output(prediction, x, t, condition, pred_type)
    target = clean if pred_type == "x0" else -2 * clean
    torch.testing.assert_close(actual[:, :, :1], target.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual[:, :, 1:], prediction[:, :, 1:], rtol=0, atol=0)


def test_causal_constraint_uses_chunk_position_and_per_frame_times():
    from fastgen.networks.network_causal import CausalContext

    net = _constraint_network("causal")
    x, out = torch.randn(2, 2, 3, 2, 2), torch.randn(2, 2, 3, 2, 2)
    cond = {"first_frame_cond": torch.randn_like(x[:, :, :1])}
    t = torch.tensor([[0.25, 0.5, 0.5], [0.5, 0, 0]], dtype=net.noise_scheduler.t_precision)
    kwargs = {"x_t": x, "t": t, "condition": cond, "fwd_pred_type": "flow"}
    assert net._condition_output(out, **kwargs, causal_ctx=CausalContext(cur_start_frame=3)) is out
    actual = net._condition_output(out, **kwargs, causal_ctx=CausalContext(cur_start_frame=0))
    first_t = t[:, None, :1, None, None].to(x.dtype)
    torch.testing.assert_close(actual[:, :, :1], (x[:, :, :1] - cond["first_frame_cond"]) / first_t)
    torch.testing.assert_close(actual[:, :, 1:], out[:, :, 1:], rtol=0, atol=0)
    # Only frame zero is constrained, so only its time must be noised.
    with pytest.raises(ValueError, match="sigma = 0"):
        net._condition_output(out, **{**kwargs, "t": t.flip(1)}, causal_ctx=CausalContext(cur_start_frame=0))


@pytest.mark.parametrize("rescale_type", ["std", "norm"])
@pytest.mark.parametrize("multimodal", [False, True])
def test_ltx_real_forward_partial_strength_and_all_guidance_terms(rescale_type, multimodal):
    from fastgen.networks.LTX2.network import LTX2GuidanceConfig
    from tests.test_network import _ltx2_i2v_sample_stub

    net = _ltx2_i2v_sample_stub(("video", "audio") if multimodal else ("video",))
    torch.manual_seed(42)
    video = torch.randn(1, 2, 4, 2, 2)
    x = {"video": video} if multimodal else video
    t = torch.tensor([0.5], dtype=net.noise_scheduler.t_precision)
    mask = torch.ones(1, 2, dtype=torch.long)
    cond = {
        "text_embeds": (torch.randn(1, 2, 3), mask),
        "frame_cond": torch.randn(1, 2, 3, 2, 2),
        "frame_idx": torch.tensor([0, 1, 3]),
        "strength": torch.tensor([0.0, 0.4, 1.0]),
    }
    neg = {**cond, "text_embeds": (torch.randn(1, 2, 3), mask)}
    g = LTX2GuidanceConfig()
    g.cfg.scale, g.mix.scale, g.stg.scale = 3.0, 1.5, 2.0
    g.modality.scale = 2.0 if multimodal else None
    g.rescale, g.rescale_type = 0.7, rescale_type
    # Cached native velocities, in guidance's working dtype; a cached perturbation is valid even for this
    # block-free transformer.
    pos = net(x, t, condition=cond, apply_output_conditioning=False, output_dtype=torch.float64)
    negative = net(x, t, condition=neg, apply_output_conditioning=False, output_dtype=torch.float64)
    p, n = (pos["video"], negative["video"]) if multimodal else (pos, negative)
    stg = {"video": p * 0.8} if multimodal else p * 0.8
    modality = {"video": p * 0.9} if multimodal else p * 0.9
    cache = {"pos": pos, "neg": negative, "stg": stg, "modality": modality, "pos_mix": pos}
    guided = p + 2.5 * (p - n) + 0.2 * p + (0.1 * p if multimodal else 0)

    def to_x0(velocity):
        wrap = (lambda value: {"video": value}) if multimodal else (lambda value: value)
        x0 = net.transform_output(
            wrap(velocity), x, t, cond, "x0", output_dtype=torch.float64, apply_output_conditioning=False
        )
        return x0["video"] if multimodal else x0

    # The reference rescales the per-token x0, then blends the conditioned frames.
    expected = cfg_rescale(to_x0(guided), to_x0(p), 0.7, rescale_type=rescale_type)
    idx = cond["frame_idx"]
    weight = (1 - cond["strength"]).reshape(1, 1, -1, 1, 1).double()
    expected[:, :, idx] = expected[:, :, idx] * weight + cond["frame_cond"].double() * (1 - weight)
    actual = predict_with_guidance(net, x, t, cond, g, neg, precomputed=cache, fwd_pred_type="x0")
    actual = actual["video"] if multimodal else actual
    torch.testing.assert_close(actual, expected.float())
    torch.testing.assert_close(actual[:, :, 3], cond["frame_cond"][:, :, 2], rtol=0, atol=0)
    direct = net(x, t, condition=cond, fwd_pred_type="x0")
    plain = predict_with_guidance(net, x, t, cond, fwd_pred_type="x0")
    torch.testing.assert_close(plain, direct, rtol=0, atol=0)


def test_ltx_audio_and_cache_are_untouched_by_video_constraint():
    net = _constraint_network("ltx")
    video, audio = torch.randn(2, 2, 3, 2, 2), torch.randn(2, 4, 6)
    prediction = {"video": video, "audio": audio}
    clean = torch.randn_like(video[:, :, :1])
    actual = net._condition_output(
        prediction,
        x_t=prediction,
        t=torch.full((2,), 0.5, dtype=net.noise_scheduler.t_precision),
        condition={"text_embeds": None, "frame_cond": clean},
        fwd_pred_type="x0",
    )
    assert actual is not prediction and actual["audio"] is audio and prediction["video"] is video
    torch.testing.assert_close(actual["video"][:, :, :1], clean, rtol=0, atol=0)


def test_trigflow_defers_native_constraint_and_rejects_finalizing_it():
    from fastgen.methods.trajectory_matching.sCM import TrigFlowPrecond

    native = _wan_forward_network()
    wrapper = TrigFlowPrecond(native, sigma_data=0.5)
    x = torch.randn(2, 2, 3, 2, 2)
    t = torch.tensor([0.3, 0.7], dtype=native.noise_scheduler.t_precision)
    cond = {"text_embeds": torch.randn(2, 3, 4), "first_frame_cond": torch.randn_like(x[:, :, :1])}
    neg = {**cond, "text_embeds": -cond["text_embeds"]}
    g = GuidanceConfig()
    g.cfg.scale = 3.0
    actual, cache = predict_with_guidance(
        wrapper, x, t, cond, g, neg, return_cache=True, apply_output_conditioning=False
    )
    # F_theta of the wrapped network's unconstrained x0.
    deferred = wrapper(x, t, condition=cond, apply_output_conditioning=False, output_dtype=None)
    torch.testing.assert_close(cache["pos"], deferred, rtol=0, atol=0)
    torch.testing.assert_close(actual, (cache["pos"] + 2 * (cache["pos"] - cache["neg"])).float())
    with pytest.raises(NotImplementedError, match="apply_output_conditioning=False"):
        predict_with_guidance(wrapper, x, t, cond, g, neg, precomputed=cache)
    with pytest.raises(NotImplementedError, match="apply_output_conditioning=False"):
        wrapper.transform_output(actual, x, t, cond, "flow")


def _wan_forward_network(kind="wan"):
    net = _constraint_network(kind)
    net._record_precision(precision_model=torch.float32)
    net.expand_timesteps = False
    net.latent_compression_factors = (4, 8, 8)
    net._mask_when_storing_cache = False

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
            self.config = SimpleNamespace(patch_size=(1, 1, 1))

        def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, **kw):
            p = 0.5 * hidden_states[:, :2] + encoder_hidden_states.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
            features = [hidden_states]
            out = [p, features] if kw.get("feature_indices") is not None else p
            if kw.get("return_features_early"):
                return features
            return (out, torch.zeros(p.shape[0], 1)) if kw.get("return_logvar") else out

    net.transformer = Transformer()
    return net


@pytest.mark.parametrize("kind", ["wan", "camera", "causal"])
@pytest.mark.parametrize("features", [False, True])
@pytest.mark.parametrize("logvar", [False, True])
def test_wan_forward_control_preserves_auxiliary_outputs(kind, features, logvar):
    net = _wan_forward_network(kind)
    x, t = torch.randn(2, 2, 3, 2, 2), torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    condition = {"text_embeds": torch.randn(2, 3, 4), "first_frame_cond": torch.randn_like(x[:, :, :1])}
    kw = dict(
        condition=condition,
        feature_indices={0} if features else None,
        return_logvar=logvar,
        unpatchify_features=False,
    )
    raw = net(x, t, apply_output_conditioning=False, output_dtype=None, **kw)
    final = net(x, t, fwd_pred_type="x0", **kw)
    if logvar:
        torch.testing.assert_close(raw[1], final[1], rtol=0, atol=0)
        raw, final = raw[0], final[0]
    if features:
        torch.testing.assert_close(raw[1], final[1], rtol=0, atol=0)
        raw, final = raw[0], final[0]
    torch.testing.assert_close(final, net.transform_output(raw, x, t, condition, "x0"), rtol=0, atol=0)
    torch.testing.assert_close(final[:, :, :1], condition["first_frame_cond"], rtol=0, atol=0)


def _transform_case(case):
    """``(network, x_t, t, condition, forward kwargs)`` for a network whose forward runs on CPU without weights."""
    if case in ("wan", "camera", "causal", "causal-per-frame"):
        net = _wan_forward_network("causal" if case.startswith("causal") else case)
        x = torch.randn(2, 2, 3, 2, 2)
        times = [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]] if case == "causal-per-frame" else [0.3, 0.7]
        t = torch.tensor(times, dtype=net.noise_scheduler.t_precision)
        condition = {"text_embeds": torch.randn(2, 3, 4), "first_frame_cond": torch.randn_like(x[:, :, :1])}
        return net, x, t, condition, {}
    if case == "ltx":
        from tests.test_network import _ltx2_i2v_sample_stub

        net = _ltx2_i2v_sample_stub()
        x = torch.randn(1, 2, 4, 2, 2)
        condition = {
            "text_embeds": (torch.randn(1, 2, 3), torch.ones(1, 2, dtype=torch.long)),
            "frame_cond": torch.randn(1, 2, 2, 2, 2),
            "frame_idx": torch.tensor([0, 2]),
            "strength": torch.tensor([0.4, 1.0]),
        }
        return net, x, torch.tensor([0.5], dtype=net.noise_scheduler.t_precision), condition, {}
    from tests.test_network import _tiny_reference_dit

    net = _tiny_reference_dit(learn_sigma=True)
    t = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    return net, torch.randn(2, 4, 8, 8), t, torch.tensor([1, 3]), {"return_variance": case == "dit-variance"}


@pytest.mark.parametrize("pred_type", ["x0", "eps", "flow"])
@pytest.mark.parametrize("case", ["wan", "camera", "causal", "causal-per-frame", "ltx", "dit", "dit-variance"])
def test_transforming_the_native_prediction_reproduces_the_forward(case, pred_type):
    """``forward`` ends with ``transform_output``, which only casts a native prediction to the native type, so
    transforming the unconstrained native output gives the same output."""
    torch.manual_seed(0)
    net, x, t, condition, kwargs = _transform_case(case)
    with torch.no_grad():
        native = net(x, t, condition=condition, apply_output_conditioning=False, output_dtype=None, **kwargs)
        direct = net(x, t, condition=condition, fwd_pred_type=pred_type, **kwargs)
        transformed = net.transform_output(native, x, t, condition, pred_type, **kwargs)
        again = net.transform_output(
            native, x, t, condition, output_dtype=native.dtype, apply_output_conditioning=False, **kwargs
        )
    torch.testing.assert_close(transformed, direct, rtol=0, atol=0)
    torch.testing.assert_close(again, native, rtol=0, atol=0)


def test_moe_routes_raw_branches_and_finalizes_once():
    from fastgen.networks.Wan.wan_moe import WanMoE

    net = WanMoE.__new__(WanMoE)
    torch.nn.Module.__init__(net)
    net.network_high, net.network_low = _wan_forward_network(), _wan_forward_network()
    net.noise_scheduler, net.net_pred_type = RFNoiseSchedule(), "flow"
    net.guidance_boundary, net.enable_single_expert_shortcut = 0.5, False
    net.cond_keys_no_routing = ()
    x = torch.randn(2, 2, 3, 2, 2)
    t = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    cond = {"text_embeds": torch.randn(2, 3, 4), "first_frame_cond": torch.randn_like(x[:, :, :1])}
    neg = {**cond, "text_embeds": -cond["text_embeds"]}
    g = GuidanceConfig()
    g.cfg.scale, g.rescale = 3, 0.7
    actual, cache = predict_with_guidance(net, x, t, cond, g, neg, return_cache=True, fwd_pred_type="x0")
    assert not torch.equal(cache["pos"][:, :, :1], cond["first_frame_cond"].double())
    torch.testing.assert_close(actual[:, :, :1], cond["first_frame_cond"], rtol=0, atol=0)
    positional = net.transform_output(cache["pos"], x, t, cond, "x0", torch.float64)
    assert positional.dtype == torch.float64
    torch.testing.assert_close(positional[:, :, :1], cond["first_frame_cond"].double(), rtol=0, atol=0)
    raw = net(x, t, condition=cond, apply_output_conditioning=False, output_dtype=None)
    torch.testing.assert_close(cache["pos"], raw, rtol=0, atol=0)


@pytest.mark.parametrize("use_gan", [False, True])
def test_dmd_teacher_cache_producer_requests_prediction_before_conditioning(use_gan):
    from fastgen.methods.distribution_matching.dmd2 import DMD2Model, Generation

    class Teacher(BlendedNet):
        def forward(self, *args, **kwargs):
            p = super().forward(*args, **kwargs)
            return [p, [p]] if kwargs.get("feature_indices") is not None else p

    net = Teacher()
    x = torch.randn(2, 6)
    t = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    cond, neg = torch.ones_like(x), -torch.ones_like(x)
    g = GuidanceConfig()
    g.cfg.scale, g.rescale = 3.0, 0.7
    seen = []
    generation = Generation(data=x, x0=x)

    def capture_loss(data, teacher, fake, *args, **kw):
        seen.append(teacher)
        return {"vsd_loss": torch.tensor(0.0)}, torch.tensor(0.0)

    model = SimpleNamespace(
        use_gan=use_gan,
        use_vsd=True,
        teacher=net,
        net=net,
        device=x.device,
        config=SimpleNamespace(guidance_cfg=g, gan_loss_weight_gen=0, vsd_loss_weight=1),
        gen_data_from_net=lambda *a, **kw: generation,
        _generation_time=lambda *a: None,
        fake_score=lambda x, *a, **kw: torch.zeros_like(x),
        _vsd_input=lambda x, *a: x,
        _compute_distribution_matching_loss=capture_loss,
        _training_outputs=lambda *a: {},
        apply_reward=lambda x, data, losses: losses,
    )

    class Discriminator:
        feature_indices = {0}

        def __call__(self, features):
            return features[0].mean(dim=1, keepdim=True)

    model.discriminator = Discriminator()
    inputs = SimpleNamespace(
        input_student=x,
        student_step=torch.zeros(2, dtype=torch.long),
        generation_kwargs={},
        eps=torch.randn_like(x),
        t=t,
        loss_scale=None,
        trajectory_step=None,
    )
    # The generator GAN term needs real data.
    DMD2Model._student_update_step(model, inputs, {"condition": cond, "neg_condition": neg, "real": x})
    assert len(net.calls) == 2 and all(flag is False for flag, _ in net.calls)
    assert len(net.finalizations) == 1
    noisy = net.noise_scheduler.forward_process(x, inputs.eps, t)
    # The VSD loss receives the teacher's x0 unrounded, in guidance's working dtype.
    p, n = (noisy.square() + cond).double(), (noisy.square() + neg).double()
    expected = 0.6 * cfg_rescale(p + 2 * (p - n), p, 0.7) + 0.8
    torch.testing.assert_close(seen[0], expected)


def test_optional_network_hooks_default_to_none():
    assert FastGenNetwork.predict_with_guidance is None
    assert FastGenNetwork._condition_output is None


@pytest.mark.parametrize("kind", ["wan", "camera", "causal", "cosmos2", "cosmos3", "ltx"])
@pytest.mark.parametrize("pred_type", ["x0", "flow"])
def test_output_constraint_treats_a_narrower_condition_like_its_exact_upcast(kind, pred_type):
    net = _constraint_network(kind)
    x = torch.randn(2, 2, 3, 2, 2)
    clean = torch.randn(2, 2, 1, 2, 2, dtype=torch.bfloat16)
    mask = torch.zeros(2, 1, 3, 1, 1, dtype=torch.bool)
    mask[:, :, 0] = True
    t = torch.ones(2, dtype=net.noise_scheduler.t_precision) * 0.5

    def condition_output(clean):
        condition = {
            "text_embeds": None,
            "first_frame_cond": clean,
            "frame_cond": clean,
            "conditioning_latents": clean,
            "condition_mask": mask,
        }
        return net._condition_output(x.double(), x_t=x, t=t, condition=condition, fwd_pred_type=pred_type)

    assert torch.equal(condition_output(clean), condition_output(clean.to(x.dtype)))


@pytest.mark.parametrize("kind", ["wan", "cosmos2", "cosmos3", "ltx"])
def test_absent_or_disabled_condition_is_identity(kind):
    net = _constraint_network(kind)
    x = torch.randn(2, 2, 3, 2, 2)
    t = torch.ones(2, dtype=net.noise_scheduler.t_precision) * 0.5
    assert net._condition_output(x, x_t=x, t=t, condition=None, fwd_pred_type="flow") is x
    if kind == "wan":
        net.concat_mask = True
        assert net._condition_output(x, x_t=x, t=t, condition={}, fwd_pred_type="flow") is x


@pytest.mark.parametrize(
    "state_dtype,output_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float16),
        (torch.float64, torch.float32),
    ],
)
@pytest.mark.parametrize("pred_type", ["x0", "flow"])
def test_ltx_guidance_rounds_once_to_the_output_dtype_and_keeps_prepared_conditions(
    state_dtype, output_dtype, pred_type
):
    from tests.test_network import _ltx2_i2v_sample_stub

    net = _ltx2_i2v_sample_stub()
    x = torch.randn(2, 2, 3, 2, 2, dtype=state_dtype)
    t = torch.full((2,), 0.5, dtype=net.noise_scheduler.t_precision)
    clean = torch.randn_like(x[:, :, :1])
    condition = {
        "text_embeds": (torch.randn(2, 2, 3), torch.ones(2, 2, dtype=torch.long)),
        "frame_cond": clean,
        "strength": torch.tensor([0.4], dtype=torch.float32),
    }
    g = GuidanceConfig()
    g.cfg.scale, g.rescale = 3, 0.7
    actual, cache = predict_with_guidance(
        net,
        x,
        t,
        condition,
        g,
        condition,
        fwd_pred_type=pred_type,
        output_dtype=output_dtype,
        return_cache=True,
    )
    # Every request composes the same working-dtype result and rounds it once.
    reference = predict_with_guidance(
        net, x, t, condition, g, condition, precomputed=cache, fwd_pred_type=pred_type, output_dtype=torch.float64
    )
    torch.testing.assert_close(actual, reference.to(output_dtype), rtol=0, atol=0)
    assert actual.dtype == output_dtype and condition["frame_cond"] is clean and clean.dtype == state_dtype
    assert cache["pos"].dtype == cache["neg"].dtype == torch.float64

    # PDD can request a decoding dtype below the teacher's state dtype.
    from fastgen.methods.trajectory_matching.pdd import PDDModel

    model = SimpleNamespace(
        teacher=net, precision=state_dtype, precision_decoding=output_dtype, config=SimpleNamespace(guidance_cfg=g)
    )
    pdd_prediction = PDDModel._guided_teacher(model, x, t, condition, condition, pred_type)
    torch.testing.assert_close(pdd_prediction, actual, rtol=0, atol=0)
    assert pdd_prediction.dtype == output_dtype


@pytest.mark.parametrize("strength", [0.4, 1.0])
def test_ltx_flow_and_x0_teacher_targets_share_the_x0_rescale(strength):
    """PDD's flow target is DMD2's x0 target converted: both rescale x0, then constrain the conditioned frame."""
    from fastgen.methods.trajectory_matching.pdd import PDDModel
    from tests.test_network import _ltx2_i2v_sample_stub

    net = _ltx2_i2v_sample_stub()
    torch.manual_seed(0)
    x = torch.randn(2, 2, 3, 2, 2)
    t = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    mask = torch.ones(2, 2, dtype=torch.long)
    condition = {
        "text_embeds": (torch.randn(2, 2, 3), mask),
        "frame_cond": torch.randn_like(x[:, :, :1]),
        "strength": torch.tensor([strength]),
    }
    negative = {**condition, "text_embeds": (torch.randn(2, 2, 3), mask)}
    g = GuidanceConfig()
    g.cfg.scale, g.rescale = 3.0, 0.7
    x0 = predict_with_guidance(net, x, t, condition, g, negative, fwd_pred_type="x0", output_dtype=torch.float64)
    model = SimpleNamespace(
        teacher=net, precision=x.dtype, precision_decoding=torch.float64, config=SimpleNamespace(guidance_cfg=g)
    )
    flow = PDDModel._guided_teacher(model, x, t, condition, negative, "flow")
    expected = net.noise_scheduler.x0_to_flow(x, x0, t, output_dtype=torch.float64)
    torch.testing.assert_close(flow, expected, rtol=1e-5, atol=1e-5)
    # Rescaling the velocities instead changes the unconditioned frames by far more than rounding.
    _, cache = predict_with_guidance(
        net, x, t, condition, g, negative, fwd_pred_type="flow", output_dtype=torch.float64, return_cache=True
    )
    pos, neg = cache["pos"], cache["neg"]
    flow_space = cfg_rescale(pos + 2 * (pos - neg), pos, 0.7)
    assert not torch.allclose(flow[:, :, 1:], flow_space[:, :, 1:], rtol=1e-3, atol=1e-3)


def test_ltx_guidance_composes_a_negative_with_other_i2v_fields_as_its_own_native_prediction():
    """``forward`` expresses each branch against the scalar ``t`` with its own condition, so any negative composes."""
    from tests.test_network import _ltx2_i2v_sample_stub

    net = _ltx2_i2v_sample_stub()
    x = torch.randn(1, 2, 3, 2, 2)
    t = torch.tensor([0.5], dtype=net.noise_scheduler.t_precision)
    text = (torch.randn(1, 2, 3), torch.ones(1, 2, dtype=torch.long))
    condition = {"text_embeds": text, "frame_cond": torch.randn(1, 2, 1, 2, 2), "strength": torch.tensor([0.5])}
    cfg = GuidanceConfig()
    cfg.cfg.scale = 3.0
    for negative in (text, {**condition, "strength": torch.tensor([0.4])}, {**condition, "preserve_cond_input": False}):
        with torch.no_grad():
            p = net(x, t, condition=condition, apply_output_conditioning=False, output_dtype=None)
            n = net(x, t, condition=negative, apply_output_conditioning=False, output_dtype=None)
            expected = net.transform_output(p + 2 * (p - n), x, t, condition, "x0")
            actual = predict_with_guidance(net, x, t, condition, cfg, negative, fwd_pred_type="x0")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strength", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("rescale_type", ["std", "norm"])
def test_ltx_sampling_trajectory_conditions_in_guidance_precision(state_dtype, strength, rescale_type, monkeypatch):
    """Compare every Euler step to independent CFG/rescale/blending with prepared conditions."""
    from tests.test_network import _ltx2_i2v_sample_stub

    net = _ltx2_i2v_sample_stub()
    torch.manual_seed(42)
    noise = torch.randn(1, 2, 3, 2, 2, dtype=state_dtype)
    clean = torch.randn_like(noise[:, :, :1])
    text_mask = torch.ones(1, 2, dtype=torch.long)
    condition = {
        "text_embeds": (torch.randn(1, 2, 3), text_mask),
        "frame_cond": clean,
        "strength": torch.tensor([strength], dtype=torch.float32),
    }
    negative = {**condition, "text_embeds": (torch.randn(1, 2, 3), text_mask)}
    g = GuidanceConfig()
    g.cfg.scale, g.rescale, g.rescale_type = 3.0, 0.7, rescale_type
    stages = []
    step = net.noise_scheduler.step

    def record_step(*args, **kwargs):
        result = step(*args, **kwargs)
        stages.append(result.clone())
        return result

    monkeypatch.setattr(net.noise_scheduler, "step", record_step)
    with torch.no_grad():
        actual = net.sample(noise.clone(), condition, negative, sample_steps=4, guidance_cfg=g)
    monkeypatch.setattr(net.noise_scheduler, "step", step)

    # The oracle uses native forwards and writes the sampling equations explicitly;
    # it never invokes guidance, an output constraint or a sampler step.
    _, grid = net.sample_scheduler_and_grid(4)
    ns = net.noise_scheduler
    latents = ns.latents(noise, t_init=grid[0])
    # Initialization uses the same clean-latent blend as output conditioning.
    mask = (1 - condition["strength"]).reshape(1, 1, 1, 1, 1)
    latents[:, :, :1] = (latents[:, :, :1].to(mask.dtype) * mask + clean.to(mask.dtype) * (1 - mask)).to(state_dtype)
    pos = {**condition, "preserve_cond_input": False}
    neg = {**negative, "preserve_cond_input": False}
    dims = tuple(range(1, noise.ndim))

    def to_x0(velocity, time):
        return net.transform_output(
            velocity, latents, time, pos, "x0", output_dtype=torch.float64, apply_output_conditioning=False
        )

    with torch.no_grad():
        for index, (time, time_next) in enumerate(zip(grid[:-1], grid[1:], strict=True)):
            dt = (time_next - time).double()
            time = time.expand(1)
            # Guidance composes, rescales and blends in FP64, and the sampler steps from that x0.
            p = net(latents, time, pos, apply_output_conditioning=False, output_dtype=torch.float64)
            n = net(latents, time, neg, apply_output_conditioning=False, output_dtype=torch.float64)
            p, guided = to_x0(p, time), to_x0(p + 2 * (p - n), time)
            if rescale_type == "std":
                pos_stat = p.std(dim=dims, keepdim=True, correction=0)
                guided_stat = guided.std(dim=dims, keepdim=True, correction=0)
            else:
                pos_stat = torch.linalg.vector_norm(p, dim=dims, keepdim=True)
                guided_stat = torch.linalg.vector_norm(guided, dim=dims, keepdim=True)
            x0 = guided * (0.7 * pos_stat / guided_stat.clamp(min=1e-5) + (1 - 0.7))
            # Conditions keep the state dtype; the blend runs in the working precision.
            x0[:, :, :1] = x0[:, :, :1] * mask.double() + clean.double() * (1 - mask.double())
            velocity = ns.x0_to_flow(latents, x0, time, output_dtype=torch.float64)
            latents = (latents.double() + dt * velocity).to(state_dtype)
            torch.testing.assert_close(stages[index], latents, rtol=0, atol=0)
    assert len(stages) == 4
    torch.testing.assert_close(actual, latents, rtol=0, atol=0)
    if strength == 1:
        assert torch.equal(actual[:, :, :1], clean)


@pytest.mark.parametrize("rescale_type", ["std", "norm"])
def test_trigflow_rescale_floor_is_in_network_output_units(rescale_type):
    from fastgen.methods.trajectory_matching.sCM import TrigFlowPrecond
    from fastgen.networks.Wan.network import Wan

    native = Wan.__new__(Wan)
    torch.nn.Module.__init__(native)
    native.noise_scheduler = RFNoiseSchedule()
    wrapper = TrigFlowPrecond(native, sigma_data=0.5)
    x = torch.zeros(2, 6, dtype=torch.float64)
    t = torch.tensor([0.3, 0.7], dtype=wrapper.noise_scheduler.t_precision)
    pos = torch.linspace(-1e-6, 1e-6, x.numel(), dtype=x.dtype).reshape_as(x)
    neg = torch.zeros_like(pos)
    g = GuidanceConfig()
    g.cfg.scale, g.rescale, g.rescale_type = 3.0, 1.0, rescale_type
    actual, cache = predict_with_guidance(
        wrapper, x, t, None, g, precomputed={"pos": pos, "neg": neg}, return_cache=True
    )
    expected = cfg_rescale(3 * pos, pos, 1.0, rescale_type=rescale_type)
    torch.testing.assert_close(actual, expected)
    assert cache["pos"] is pos and cache["neg"] is neg
    # Both statistics are below epsilon: scaling before rescale would add an
    # erroneous second factor of sigma_data to the velocity.


def _cosmos_forward_inputs(kind):
    if kind == "cosmos3":
        from tests.test_network import _tiny_cosmos3, _cosmos3_i2v_condition

        net = _tiny_cosmos3(is_i2v=True).double()
        net._record_precision(precision_model=torch.float64)
        x = torch.randn(2, 8, 2, 4, 4, dtype=torch.float64)
        condition = _cosmos3_i2v_condition(2, 8, 2, 4, 4, x.dtype)
    else:
        from fastgen.networks.cosmos_predict2.network import CosmosPredict2

        net = CosmosPredict2(
            model_channels=48,
            num_blocks=1,
            num_heads=4,
            max_img_h=4,
            max_img_w=4,
            max_frames=4,
            in_channels=2,
            out_channels=2,
            patch_spatial=1,
            patch_temporal=1,
            concat_padding_mask=False,
            add_video_condition_mask=True,
            mlp_ratio=2,
            crossattn_emb_channels=8,
            use_adaln_lora=False,
            enable_logvar_linear=False,
            use_crossattn_projection=False,
            use_wan_fp32_strategy=False,
            is_video2world=True,
            sample_sigma_schedule="karras",
        ).eval()
        x = torch.randn(2, 2, 3, 2, 2)
        mask = torch.zeros(2, 1, 3, 2, 2, dtype=torch.bool)
        mask[:, :, 0] = 1
        condition = {
            "text_embeds": torch.randn(2, 3, 8),
            "conditioning_latents": torch.randn_like(x[:, :, :1]),
            "condition_mask": mask,
        }
    t = torch.tensor([0.3, 0.7], dtype=net.noise_scheduler.t_precision)
    return net, x, t, condition


@pytest.mark.parametrize("scattered_mask", [False, True])
@pytest.mark.parametrize("replace_frames", [False, True])
@pytest.mark.parametrize("rescale", [None, 0.7])
def test_cosmos_predict2_sample_keeps_raw_state_and_honors_frame_replacement(
    scattered_mask, replace_frames, rescale, monkeypatch
):
    net, noise, _, condition = _cosmos_forward_inputs("cosmos2")
    if scattered_mask:
        # Per-sample frames that no count of leading frames describes.
        scattered = torch.zeros_like(condition["condition_mask"])
        scattered[0, :, 0::2] = True
        scattered[1, :, 1] = True
        condition = {**condition, "conditioning_latents": torch.randn_like(noise), "condition_mask": scattered}
    negative = {**condition, "text_embeds": -condition["text_embeds"]}
    scheduler, times = net.sample_scheduler_and_grid(2)
    initial = net.noise_scheduler.latents(noise, t_init=times[0])
    clean, _, mask = net._get_conditioning_tensors(initial, condition)
    pinned_velocity = (initial - clean) / scheduler.sigmas[0]
    branches, steps = [], []
    handle = net.register_forward_hook(
        lambda module, args, kwargs, output: branches.append(
            (args[0].clone(), kwargs["condition"]["condition_mask"], output.clone())
        ),
        with_kwargs=True,
    )
    step = scheduler.step

    def record_step(velocity, time, state, **kwargs):
        steps.append((state.clone(), velocity.clone()))
        return step(velocity, time, state, **kwargs)

    monkeypatch.setattr(scheduler, "step", record_step)
    guidance = GuidanceConfig()
    guidance.cfg.scale, guidance.rescale = 3.0, rescale
    try:
        with torch.no_grad():
            net.sample(
                noise,
                condition,
                negative,
                sample_steps=2,
                guidance_cfg=guidance,
                denoise_replace_gt_frames=replace_frames,
            )
    finally:
        handle.remove()

    assert len(steps) == 3 and len(branches) == 6
    assert not torch.equal(initial[:, :, :1], clean[:, :, :1])
    for index, (state, velocity) in enumerate(steps):
        (positive_state, positive_mask, positive), (negative_state, negative_mask, negative_prediction) = branches[
            2 * index : 2 * index + 2
        ]
        assert torch.equal(positive_mask, condition["condition_mask"])
        assert torch.equal(negative_mask, condition["condition_mask"])
        torch.testing.assert_close(positive_state, state, rtol=0, atol=0)
        torch.testing.assert_close(negative_state, state, rtol=0, atol=0)
        # Native branches compose in the working dtype; the result rounds once to the state dtype.
        positive, negative_prediction = positive.double(), negative_prediction.double()
        expected = cfg_rescale(positive + 2 * (positive - negative_prediction), positive, rescale).to(velocity.dtype)
        if replace_frames:
            expected = torch.where(mask, pinned_velocity, expected)
        torch.testing.assert_close(velocity, expected, rtol=0, atol=0)


def test_cosmos_predict2_sample_rejects_mask_frames_beyond_conditioning_latents():
    net, noise, _, condition = _cosmos_forward_inputs("cosmos2")
    mask = condition["condition_mask"].clone()
    mask[1, :, 2] = True
    with pytest.raises(AssertionError, match="beyond the 1 conditioning_latents frames"):
        net.sample(noise, {**condition, "condition_mask": mask}, sample_steps=1, guidance_cfg=GuidanceConfig())


@pytest.mark.parametrize("missing", ["conditioning_latents", "condition_mask"])
def test_cosmos_predict2_sample_rejects_incomplete_video2world_condition(missing):
    net, noise, _, condition = _cosmos_forward_inputs("cosmos2")
    del condition[missing]
    with pytest.raises(ValueError, match="conditioning_latents and condition_mask must be provided together"):
        net.sample(noise, condition, sample_steps=1, guidance_cfg=GuidanceConfig())


@pytest.mark.parametrize("kind", ["cosmos2", "cosmos3"])
@pytest.mark.parametrize("pred_type", ["x0", "eps", "flow"])
def test_cosmos_real_forward_transform_and_guidance_cache(kind, pred_type):
    net, x, t, condition = _cosmos_forward_inputs(kind)
    with torch.no_grad():
        raw = net(x, t, condition=condition, apply_output_conditioning=False, output_dtype=None)
        direct = net(x, t, condition=condition, fwd_pred_type=pred_type)
        transformed = net.transform_output(raw, x, t, condition, pred_type)
        torch.testing.assert_close(transformed, direct, rtol=0, atol=0)
        g = GuidanceConfig()
        g.cfg.scale, g.rescale = 3, 0.7
        # Use distinct cached predictions to make rescaling nontrivial without another backbone pass.
        guided = predict_with_guidance(
            net,
            x,
            t,
            condition,
            g,
            condition,
            precomputed={"pos": raw, "neg": raw * 0.7 + 0.5},
            fwd_pred_type=pred_type,
        )
        x0 = net.noise_scheduler.convert_model_output(x, guided, t, src_pred_type=pred_type, target_pred_type="x0")
    torch.testing.assert_close(x0[:, :, :1], condition["conditioning_latents"])


@pytest.mark.parametrize("rescale_type", ["std", "norm"])
def test_constant_cached_predictions_and_full_constraint_remain_finite(rescale_type):
    net = BlendedNet(1)
    x, t = torch.zeros(2, 6), torch.full((2,), 0.5, dtype=net.noise_scheduler.t_precision)
    cfg = GuidanceConfig()
    cfg.cfg.scale, cfg.rescale, cfg.rescale_type = 5, 1, rescale_type
    actual = predict_with_guidance(net, x, t, x, cfg, x, precomputed={"pos": x, "neg": x})
    torch.testing.assert_close(actual, torch.full_like(x, 2), rtol=0, atol=0)
    assert not net.calls and len(net.finalizations) == 1
