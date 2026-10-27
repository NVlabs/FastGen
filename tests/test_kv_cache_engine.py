# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest
import torch
from diffusers.models.transformers.transformer_wan import WanAttnProcessor
from torch.utils.checkpoint import checkpoint

import fastgen.networks.network_causal as network_causal
import fastgen.networks.Wan.network_causal as wan_causal
from fastgen.networks.kv_cache import KVCacheEngine
from fastgen.networks.kv_cache.base import _build_block_token_ranges
from fastgen.networks.network_causal import CausalContext, causal_chunk_ranges
from fastgen.networks.noise_schedule import causal_chunk_frames
from fastgen.networks.Wan.network_causal import CausalWanAttnProcessor, WanAttention
from tests.utils import RunIf, run_distributed_test


def _build_engine(
    block_count: int = 1,
    cache_tag: str = "pos",
    self_shape: tuple[int, int, int, int] = (1, 8, 2, 4),
    cross_shape: tuple[int, int, int, int] = (1, 4, 2, 4),
) -> KVCacheEngine:
    engine = KVCacheEngine()
    engine.ensure_capacity(
        block_count=block_count,
        cache_tag=cache_tag,
        self_shape=self_shape,
        cross_shape=cross_shape,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return engine


@pytest.fixture(
    params=[
        pytest.param(("cpu", torch.float32), id="cpu-fp32"),
        pytest.param(("cuda", torch.float32), id="cuda-fp32", marks=RunIf(min_gpus=1)),
        pytest.param(("cuda", torch.bfloat16), id="cuda-bf16", marks=RunIf(min_gpus=1)),
        pytest.param(("cuda", torch.float16), id="cuda-fp16", marks=RunIf(min_gpus=1)),
    ]
)
def device_dtype(request):
    return request.param


def _make_case(device_dtype, with_image=True, fused=False):
    device, dtype = device_dtype
    torch.manual_seed(42)
    attn = WanAttention(
        dim=32,
        heads=2,
        dim_head=16,
        added_kv_proj_dim=32 if with_image else None,
        cross_attention_dim_head=16,
    ).to(device=device, dtype=dtype)
    if fused:
        attn.fuse_projections()
    hidden = torch.randn(2, 7, 32, device=device, dtype=dtype)
    context = torch.randn(2, 512 + (257 if with_image else 0), 32, device=device, dtype=dtype)
    engine = KVCacheEngine()
    engine.register(attn, block_idx=0)
    ctx = CausalContext(cache_engines={"main": engine}, is_ar=True, store_kv=True)
    _ensure_capacity(engine, context, ctx.cache_tag)
    return attn, hidden, context, engine, ctx


def _ensure_capacity(engine, context, cache_tag):
    engine.ensure_capacity(
        block_count=1,
        cache_tag=cache_tag,
        self_shape=(context.shape[0], 14, 2, 16),
        cross_shape=(context.shape[0], context.shape[1], 2, 16),
        device=context.device,
        dtype=context.dtype,
    )


def test_causal_context_set_store_kv_restores_previous_value():
    ctx = CausalContext(store_kv=False)

    with ctx.set_store_kv(True):
        assert ctx.store_kv is True

    assert ctx.store_kv is False

    ctx.store_kv = True
    with pytest.raises(RuntimeError):
        with ctx.set_store_kv(False):
            assert ctx.store_kv is False
            raise RuntimeError("force restore path")

    assert ctx.store_kv is True


def test_causal_context_ar_prepare_block_mask_is_noop_without_total_num_frames():
    ctx = CausalContext(is_ar=True)

    ctx.prepare_block_mask(device=torch.device("cpu"), num_frames=3)

    assert ctx.active_block_mask() is None


@pytest.mark.parametrize("is_ar,expected", [(True, 2), (False, 0)], ids=["ar", "full-sequence"])
def test_causal_context_infers_the_start_frame_from_the_cache_only_in_ar_passes(is_ar, expected):
    """A rollout continues after its cached tokens; a full-sequence pass never inherits a stale cache."""

    class _CachedTwoFrames:
        def get_cached_self_token_count(self, cache_tag):
            return 2 * 4

    ctx = CausalContext(cache_engines={"main": _CachedTwoFrames()}, is_ar=is_ar)

    assert ctx.resolve_start_frame(frame_seqlen=4) == expected


def test_causal_context_teacher_forcing_block_mask_requires_flex_attention(monkeypatch):
    monkeypatch.setattr(network_causal, "CAUSAL_FLEX_ATTENTION_AVAILABLE", False)

    ctx = CausalContext(is_ar=False, total_num_frames=3)

    with pytest.raises(RuntimeError, match="FlexAttention is required"):
        ctx.prepare_block_mask(device=torch.device("cpu"), num_frames=3)


def test_causal_context_teacher_forcing_block_mask_requires_total_num_frames(monkeypatch):
    monkeypatch.setattr(network_causal, "CAUSAL_FLEX_ATTENTION_AVAILABLE", True)

    ctx = CausalContext(is_ar=False)

    with pytest.raises(ValueError, match="total_num_frames"):
        ctx.prepare_block_mask(device=torch.device("cpu"), num_frames=3)


def test_causal_wan_attn_processor_cross_cache_reuses_cached_kv_in_no_grad(monkeypatch):
    cache_tag = "pos"
    batch_size, seq_len, cond_len, dim, heads, dim_head = 1, 2, 4, 8, 2, 4
    engine = _build_engine(
        cache_tag=cache_tag,
        self_shape=(batch_size, 8, heads, dim_head),
        cross_shape=(batch_size, cond_len, heads, dim_head),
    )

    # Mirror the real Wan block layout where self-attention and cross-attention are
    # distinct modules that must both resolve to the same cache-engine block index.
    attn_self = WanAttention(dim=dim, heads=heads, dim_head=dim_head)
    attn_cross = WanAttention(dim=dim, heads=heads, dim_head=dim_head)
    engine.register(attn_self, block_idx=0)
    engine.register(attn_cross, block_idx=0)

    processor = CausalWanAttnProcessor()
    captured_kv = []

    def fake_dispatch_attention_fn(query, key, value, **kwargs):
        del kwargs
        captured_kv.append((key.detach().clone(), value.detach().clone()))
        return torch.zeros_like(query)

    monkeypatch.setattr(wan_causal, "dispatch_attention_fn", fake_dispatch_attention_fn)

    ctx = CausalContext(cache_tag=cache_tag, store_kv=True, cache_engines={"main": engine})

    hidden_states = torch.randn(batch_size, seq_len, dim)
    encoder_hidden_states = torch.randn(batch_size, cond_len, dim)
    out = processor(
        attn=attn_cross,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        causal_ctx=ctx,
    )
    assert tuple(out.shape) == (batch_size, seq_len, dim)

    cross_entry = engine.get_cross_kv(block_idx=0, cache_tag=cache_tag)
    assert cross_entry is not None
    cached_k, cached_v, is_init = cross_entry
    cached_k_first = cached_k.clone()
    cached_v_first = cached_v.clone()
    assert is_init is True
    assert torch.allclose(captured_kv[-1][0], cached_k)
    assert torch.allclose(captured_kv[-1][1], cached_v)

    ctx.store_kv = False
    hidden_states_2 = torch.randn(batch_size, seq_len, dim)
    encoder_hidden_states_2 = torch.randn(batch_size, cond_len, dim)
    with torch.no_grad():
        _ = processor(
            attn=attn_cross,
            hidden_states=hidden_states_2,
            encoder_hidden_states=encoder_hidden_states_2,
            causal_ctx=ctx,
        )
    assert torch.allclose(captured_kv[-1][0], cached_k)
    assert torch.allclose(captured_kv[-1][1], cached_v)

    ctx.store_kv = True
    hidden_states_3 = torch.randn(batch_size, seq_len, dim)
    encoder_hidden_states_3 = torch.randn(batch_size, cond_len, dim)
    _ = processor(
        attn=attn_cross,
        hidden_states=hidden_states_3,
        encoder_hidden_states=encoder_hidden_states_3,
        causal_ctx=ctx,
    )
    cross_entry_updated = engine.get_cross_kv(block_idx=0, cache_tag=cache_tag)
    assert cross_entry_updated is not None
    cached_k_updated, cached_v_updated, _ = cross_entry_updated
    assert torch.allclose(captured_kv[-1][0], cached_k_updated)
    assert torch.allclose(captured_kv[-1][1], cached_v_updated)
    assert not torch.allclose(cached_k_first, cached_k_updated)
    assert not torch.allclose(cached_v_first, cached_v_updated)


@pytest.mark.parametrize("with_image", [True, False], ids=["i2v", "t2v"])
@pytest.mark.parametrize("fused", [False, True], ids=["separate", "fused"])
def test_cross_cache_matches_reference_and_skips_projections(device_dtype, with_image, fused):
    attn, hidden, context, engine, ctx = _make_case(device_dtype, with_image=with_image, fused=fused)
    processor = CausalWanAttnProcessor()
    reference = WanAttnProcessor()
    mask = torch.ones(2, 1, 1, 512, dtype=torch.bool, device=context.device)
    mask[..., -8:] = False  # Only text attention takes this mask; the image branch remains unmasked.
    with torch.no_grad():
        expected = reference(attn, hidden, context, attention_mask=mask)
        expected_next = reference(attn, hidden * 0.5, context, attention_mask=mask)

    names = ["to_kv"] if fused else ["to_k", "to_v"]
    if with_image:
        names += ["to_added_kv"] if fused else ["add_k_proj", "add_v_proj"]
    calls = Counter()

    def count_call(name):
        def hook(*_):
            calls[name] += 1

        return hook

    handles = [getattr(attn, name).register_forward_hook(count_call(name)) for name in names]
    try:
        with torch.inference_mode():
            ctx.store_kv = False
            actual = processor(attn, hidden, context, attention_mask=mask, causal_ctx=ctx)
            torch.testing.assert_close(actual, expected)
            assert not engine.get_cross_kv(0, "pos")[2]

            ctx.store_kv = True
            actual = processor(attn, hidden, context, attention_mask=mask, causal_ctx=ctx)
            torch.testing.assert_close(actual, expected)
            assert calls == Counter({name: 2 for name in names})
            key, value, initialized = engine.get_cross_kv(0, "pos")
            assert initialized and key.shape == value.shape == (2, context.shape[1], 2, 16)

            # This is repeated by classify_forward_prepare at every chunk/denoising step.
            _ensure_capacity(engine, context, "pos")
            calls.clear()
            for store_kv in (False, True):
                ctx.store_kv = store_kv
                actual = processor(attn, hidden * 0.5, context, attention_mask=mask, causal_ctx=ctx)
                torch.testing.assert_close(actual, expected_next)
                assert not calls
                cached_key, cached_value, initialized = engine.get_cross_kv(0, "pos")
                assert initialized
                assert cached_key.data_ptr() == key.data_ptr()
                assert cached_value.data_ptr() == value.data_ptr()
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("store_kv", [False, True], ids=["read-only", "refresh"])
@pytest.mark.parametrize("checkpointed", [False, True], ids=["eager", "checkpointed"])
def test_image_cross_cache_preserves_reference_gradients(device_dtype, store_kv, checkpointed):
    attn, hidden, context, engine, ctx = _make_case(device_dtype)
    processor = CausalWanAttnProcessor()
    # Inference tensors must never enter a subsequent differentiable forward.
    with torch.inference_mode():
        processor(attn, hidden, context + 1, causal_ctx=ctx)
    old_key, old_value, _ = engine.get_cross_kv(0, "pos")
    hidden.requires_grad_()
    context.requires_grad_()
    ctx.store_kv = store_kv
    params = dict(attn.named_parameters())
    inputs = (hidden, context, *params.values())

    expected = WanAttnProcessor()(attn, hidden, context)
    expected_grads = torch.autograd.grad(expected.float().square().sum(), inputs)

    def forward(h, c):
        return processor(attn, h, c, causal_ctx=ctx)

    actual = checkpoint(forward, hidden, context, use_reentrant=False) if checkpointed else forward(hidden, context)
    actual_grads = torch.autograd.grad(actual.float().square().sum(), inputs)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    # Guard against a shared missing-gradient bug in either implementation.
    for name in ("to_k.weight", "to_v.weight", "add_k_proj.weight", "add_v_proj.weight"):
        grad = actual_grads[2 + list(params).index(name)]
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    assert torch.count_nonzero(actual_grads[1][:, :257]) > 0
    assert torch.count_nonzero(actual_grads[1][:, 257:]) > 0

    key, value, initialized = engine.get_cross_kv(0, "pos")
    assert initialized
    if store_kv:
        assert key.requires_grad and value.requires_grad
        assert not key.is_inference() and not value.is_inference()
    else:
        assert key is old_key and value is old_value


@pytest.mark.parametrize("delete", [False, True], ids=["clear", "delete"])
def test_image_cross_cache_isolates_guidance_and_resets_conditioning(device_dtype, delete):
    attn, hidden, positive, engine, ctx = _make_case(device_dtype)
    processor = CausalWanAttnProcessor()
    negative = positive + 1
    with torch.no_grad():
        expected_pos = WanAttnProcessor()(attn, hidden, positive)
        expected_neg = WanAttnProcessor()(attn, hidden, negative)
        processor(attn, hidden, positive, causal_ctx=ctx)
        ctx.cache_tag = "neg"
        _ensure_capacity(engine, negative, "neg")
        processor(attn, hidden, negative, causal_ctx=ctx)
        assert not torch.allclose(expected_pos, expected_neg)
        ctx.store_kv = False
        for tag, context, expected in (("pos", positive, expected_pos), ("neg", negative, expected_neg)):
            ctx.cache_tag = tag
            torch.testing.assert_close(processor(attn, hidden, context, causal_ctx=ctx), expected)

        engine.clear(delete=delete)
        replacement = positive - 1
        ctx.cache_tag = "pos"
        ctx.store_kv = True
        _ensure_capacity(engine, replacement, "pos")
        assert not engine.get_cross_kv(0, "pos")[2]
        expected = WanAttnProcessor()(attn, hidden, replacement)
        torch.testing.assert_close(processor(attn, hidden, replacement, causal_ctx=ctx), expected)
        assert engine.get_cross_kv(0, "pos")[2]


def test_kv_cache_engine_prepare_self_attention_kv_resolves_start_idx_and_prefix():
    cache_tag = "pos"
    engine = _build_engine(cache_tag=cache_tag, self_shape=(1, 8, 2, 4), cross_shape=(1, 4, 2, 4))

    key_cached = torch.randn(1, 3, 2, 4)
    value_cached = torch.randn(1, 3, 2, 4)
    engine.append_self_kv(
        block_idx=0,
        cache_tag=cache_tag,
        start_idx=0,
        key=key_cached,
        value=value_cached,
        detach=True,
    )

    key_cur = torch.randn(1, 2, 2, 4)
    value_cur = torch.randn(1, 2, 2, 4)
    k_full, v_full = engine.prepare_self_attention_kv(
        block_idx=0,
        cache_tag=cache_tag,
        requested_start_idx=None,
        key=key_cur,
        value=value_cur,
        store_kv=False,
        allow_cache_grad=False,
    )

    assert torch.allclose(k_full, torch.cat([key_cached, key_cur], dim=1))
    assert torch.allclose(v_full, torch.cat([value_cached, value_cur], dim=1))


def test_kv_cache_engine_refuses_to_drop_cached_tokens_mid_rollout():
    """Re-creating a populated entry or reading past the cached tokens would silently drop the earlier chunks."""
    engine = _build_engine(self_shape=(1, 8, 2, 4))
    engine.append_self_kv(
        block_idx=0,
        cache_tag="pos",
        start_idx=0,
        key=torch.randn(1, 4, 2, 4),
        value=torch.randn(1, 4, 2, 4),
        detach=True,
    )

    def ensure(self_shape):
        engine.ensure_capacity(
            block_count=1,
            cache_tag="pos",
            self_shape=self_shape,
            cross_shape=(1, 4, 2, 4),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    ensure((1, 8, 2, 4))
    assert engine.get_self_len(block_idx=0, cache_tag="pos") == 4
    with pytest.raises(ValueError, match="holds 4 tokens"):
        ensure((1, 12, 2, 4))
    with pytest.raises(ValueError, match="only 4 are cached"):
        engine.get_self_prefix(block_idx=0, cache_tag="pos", end_idx=6)

    engine.clear()
    ensure((1, 12, 2, 4))
    assert engine.get_self_len(block_idx=0, cache_tag="pos") == 0
    with pytest.raises(ValueError, match="only 0 are cached"):
        engine.get_self_prefix(block_idx=0, cache_tag="pos", end_idx=4)


@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16])
def test_kv_cache_engine_anchor_last_self_kv_to_output_keeps_values_and_graph(output_dtype):
    cache_tag = "pos"
    engine = _build_engine(cache_tag=cache_tag, self_shape=(1, 8, 2, 4), cross_shape=(1, 4, 2, 4))

    key = torch.randn(1, 3, 2, 4, requires_grad=True)
    value = torch.randn(1, 3, 2, 4, requires_grad=True)
    # Every value is finite, but their FP16 sum overflows.
    output = torch.full((1, 32, 64), 64.0, dtype=output_dtype, requires_grad=True)

    engine.append_self_kv(
        block_idx=0,
        cache_tag=cache_tag,
        start_idx=0,
        key=key,
        value=value,
        detach=False,
    )
    engine.anchor_last_self_kv_to_output(block_idx=0, cache_tag=cache_tag, output=output)

    k_prefix, v_prefix = engine.get_self_prefix(block_idx=0, cache_tag=cache_tag, end_idx=3)
    assert torch.allclose(k_prefix, key)
    assert torch.allclose(v_prefix, value)

    (k_prefix.sum() + v_prefix.sum()).backward()
    assert key.grad is not None
    assert value.grad is not None
    assert output.grad is not None
    assert torch.count_nonzero(output.grad) == 0


def test_block_token_ranges_put_remainder_frames_in_the_first_chunk():
    """Masks put remainder frames in the first chunk, as the AR sampling loops and per-chunk training timesteps do."""
    assert _build_block_token_ranges(num_frames=8, frame_seqlen=2, chunk_size=3) == [(0, 10), (10, 16)]
    assert _build_block_token_ranges(num_frames=2, frame_seqlen=2, chunk_size=3) == [(0, 4)]


@pytest.mark.parametrize(
    ("num_frames", "chunk_size"), [(1, 1), (5, 1), (1, 3), (5, 3), (7, 3), (8, 3), (21, 3), (9, 4), (30, 4)]
)
def test_causal_chunk_frames_follow_the_mask_chunks(num_frames, chunk_size):
    """Per-chunk training times follow ``causal_chunk_frames`` and the AR loops ``causal_chunk_ranges``, which chunk
    frames exactly like the attention masks."""
    ranges = _build_block_token_ranges(num_frames, frame_seqlen=1, chunk_size=chunk_size)
    assert causal_chunk_frames(num_frames, chunk_size) == [end - start for start, end in ranges]
    assert causal_chunk_ranges(num_frames, chunk_size) == ranges


def test_kv_cache_import_keeps_dynamo_ddp_graph_splitting():
    """Importing the KV-cache package leaves ``optimize_ddp`` at its default: Dynamo's DDP graph splitting handles
    FlexAttention's higher-order op, so compiled DDP networks keep their communication overlap."""
    import torch._dynamo

    assert torch._dynamo.config.optimize_ddp is True


def _ddp_flex_attention_impl(rank, world_size):
    from torch.nn.parallel import DistributedDataParallel

    from fastgen.networks.kv_cache import flex_attention, prepare_blockwise_causal_attn_mask

    class FlexNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query = torch.nn.Sequential(torch.nn.Linear(64, 4096), torch.nn.Linear(4096, 64))
            self.out = torch.nn.Sequential(torch.nn.Linear(64, 4096), torch.nn.Linear(4096, 64))

        def forward(self, x, block_mask):
            return self.out(flex_attention(self.query(x), x, x, block_mask=block_mask))

    device = torch.device("cuda", rank)
    torch.manual_seed(0)
    # 1 MB buckets split the 4 MB of parameters, so DDP's graph splitting runs on the compiled module.
    model = DistributedDataParallel(FlexNet().to(device), device_ids=[rank], bucket_cap_mb=1)
    block_mask = prepare_blockwise_causal_attn_mask(device, num_frames=2, frame_seqlen=64, chunk_size=1)
    x = torch.randn(1, 2, 128, 64, device=device)
    for compiled in (False, True):
        if compiled:
            model.compile()
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            model(x, block_mask).sum().backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    return {"optimize_ddp": torch._dynamo.config.optimize_ddp}


@RunIf(min_gpus=1)
def test_ddp_networks_run_compiled_flex_attention_with_ddp_graph_splitting():
    """FlexAttention runs inside DDP, eager and compiled, under Dynamo's default DDP graph splitting."""
    assert run_distributed_test(_ddp_flex_attention_impl, world_size=1, timeout=600) == {"optimize_ddp": True}
