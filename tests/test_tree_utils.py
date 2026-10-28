# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pytree utilities, including traversal, casting, metadata, and assertions."""

from collections import namedtuple

import pytest
import torch

from fastgen.utils.tree_utils import (
    DICT_TYPES,
    assert_same,
    cast_like,
    detach,
    expand_first_dim,
    get_batch_size,
    get_device,
    get_dtype,
    get_shape,
    index,
    noise_from_shape,
    per_modality,
    randn_like,
    slice_dim,
    tmap,
    to,
    to_dict,
)


# ── fixtures ──────────────────────────────────────────────────────────────

Pair = namedtuple("Pair", ["first", "second"])


def _make_dict(**kw):
    return {"x": torch.ones(2, 3, **kw), "y": torch.ones(2, 5, **kw)}


def _make_list():
    return [torch.ones(2, 3), torch.ones(2, 5)]


def _make_tuple():
    return (torch.ones(2, 3), torch.ones(2, 5))


def _make_namedtuple():
    return Pair(first=torch.ones(2, 3), second=torch.ones(2, 5))


# ═════════════════════════════════════════════════════════════════════════
# per_modality (canonical multi-arg dispatcher)
# ═════════════════════════════════════════════════════════════════════════


class TestPerModality:
    # ── single-arg / non-dict primary ──────────────────────────────────

    def test_tensor_primary_short_circuits(self):
        """Non-dict primary → fn(primary, *args, **kwargs); no per-key indexing."""
        t = torch.ones(3)
        result = per_modality(lambda x: x * 2, t)
        assert torch.equal(result, torch.full((3,), 2.0))

    def test_tensor_primary_with_extra_args(self):
        t = torch.ones(3)
        result = per_modality(lambda x, s: x * s, t, 5.0)
        assert torch.equal(result, torch.full((3,), 5.0))

    # ── dict primary, multi-arg ─────────────────────────────────────────

    def test_dict_primary_multi_arg_zip(self):
        a = {"x": torch.ones(3), "y": torch.full((4,), 2.0)}
        b = {"x": torch.full((3,), 10.0), "y": torch.full((4,), 20.0)}
        result = per_modality(lambda u, v: u + v, a, b)
        assert isinstance(result, dict)
        assert torch.equal(result["x"], torch.full((3,), 11.0))
        assert torch.equal(result["y"], torch.full((4,), 22.0))

    def test_dict_primary_scalar_broadcast(self):
        """Non-dict positional arg is broadcast to every per-modality call."""
        a = {"x": torch.ones(3), "y": torch.ones(4)}
        result = per_modality(lambda u, s: u * s, a, 5.0)
        assert torch.equal(result["x"], torch.full((3,), 5.0))
        assert torch.equal(result["y"], torch.full((4,), 5.0))

    def test_dict_primary_none_broadcast(self):
        """``None`` is broadcast as-is — useful for per-modality dicts whose
        entries can be ``None`` to disable a single modality."""
        a = {"x": torch.ones(3), "y": torch.ones(4)}
        result = per_modality(lambda u, opt: u if opt is None else u * opt, a, None)
        assert torch.equal(result["x"], torch.ones(3))
        assert torch.equal(result["y"], torch.ones(4))

    def test_dict_primary_callable_broadcast(self):
        """Callable args are broadcast (not indexed)."""
        a = {"x": torch.ones(3), "y": torch.ones(4)}
        result = per_modality(lambda u, fn: fn(u), a, torch.nn.functional.silu)
        assert torch.equal(result["x"], torch.nn.functional.silu(torch.ones(3)))

    def test_dict_primary_dict_arg_with_none_entry(self):
        """Dict arg with per-modality ``None`` lets callers skip individual modalities."""
        a = {"x": torch.ones(3), "y": torch.ones(4)}
        scales = {"x": 5.0, "y": None}
        result = per_modality(lambda u, s: u if s is None else u * s, a, scales)
        assert torch.equal(result["x"], torch.full((3,), 5.0))
        assert torch.equal(result["y"], torch.ones(4))

    # ── kwargs symmetry ─────────────────────────────────────────────────

    def test_kwarg_broadcast(self):
        """Non-dict kwarg is broadcast to every per-modality call (matches
        existing ``CompositeNoiseSchedule`` callers like
        ``latents(noise, t_init=…)``)."""
        a = {"x": torch.ones(3), "y": torch.ones(4)}
        result = per_modality(lambda u, scale: u * scale, a, scale=7.0)
        assert torch.equal(result["x"], torch.full((3,), 7.0))
        assert torch.equal(result["y"], torch.full((4,), 7.0))

    def test_kwarg_per_key_indexing(self):
        """Dict-shaped kwarg is indexed per-modality, mirroring positional arg behavior."""
        a = {"x": torch.ones(3), "y": torch.ones(4)}
        result = per_modality(lambda u, scale: u * scale, a, scale={"x": 2.0, "y": 3.0})
        assert torch.equal(result["x"], torch.full((3,), 2.0))
        assert torch.equal(result["y"], torch.full((4,), 3.0))

    def test_mixed_kwargs_broadcast_and_indexed(self):
        a = {"x": torch.ones(3), "y": torch.ones(4)}
        # `dtype` (non-dict) broadcasts; `scale` (dict) indexes per-key.
        result = per_modality(
            lambda u, scale, dtype: (u * scale).to(dtype),
            a,
            scale={"x": 2.0, "y": 3.0},
            dtype=torch.float16,
        )
        assert result["x"].dtype == torch.float16
        assert result["y"].dtype == torch.float16

    # ── ModuleDict primary ─────────────────────────────────────────────

    def test_module_dict_primary(self):
        """``torch.nn.ModuleDict`` is in ``DICT_TYPES`` (it is NOT a
        ``Mapping`` subclass).  ``CompositeNoiseSchedule.subschedules`` is
        a ``ModuleDict`` and must be traversable."""
        # Pin the membership invariant so future ``DICT_TYPES`` edits notice.
        assert torch.nn.ModuleDict in DICT_TYPES
        md = torch.nn.ModuleDict({"x": torch.nn.Linear(3, 1, bias=False), "y": torch.nn.Linear(3, 1, bias=False)})
        result = per_modality(lambda mod: mod.weight.shape, md)
        assert isinstance(result, dict)
        assert result["x"] == torch.Size([1, 3])

    def test_module_dict_arg(self):
        """``ModuleDict`` arg is also indexed per-key."""
        a = {"x": torch.ones(3), "y": torch.ones(3)}
        md = torch.nn.ModuleDict({"x": torch.nn.Linear(3, 1, bias=False), "y": torch.nn.Linear(3, 1, bias=False)})
        result = per_modality(lambda u, mod: mod(u), a, md)
        assert "x" in result and "y" in result

    # ── error modes ─────────────────────────────────────────────────────

    def test_key_mismatch_raises(self):
        a = {"x": torch.ones(3), "y": torch.ones(4)}
        b = {"x": torch.ones(3), "z": torch.ones(4)}  # wrong key
        with pytest.raises(KeyError):
            per_modality(lambda u, v: u + v, a, b)

    def test_extra_keys_in_arg_silently_ignored(self):
        """Primary drives the iteration; extra keys in args are ignored."""
        a = {"x": torch.ones(3)}
        b = {"x": torch.full((3,), 2.0), "extra": torch.ones(3)}
        result = per_modality(lambda u, v: u + v, a, b)
        assert set(result) == {"x"}
        assert torch.equal(result["x"], torch.full((3,), 3.0))

    # ── edge cases ──────────────────────────────────────────────────────

    def test_empty_dict_primary(self):
        result = per_modality(lambda u: u * 2, {})
        assert result == {}

    def test_single_key_dict_primary(self):
        """A single-key dict is dispatched per modality like any other dict."""
        a = {"video": torch.ones(3)}
        result = per_modality(lambda u: u * 2, a)
        assert set(result) == {"video"}
        assert torch.equal(result["video"], torch.full((3,), 2.0))

    def test_dict_arg_on_tensor_primary_passes_through(self):
        """When primary is non-dict, dict args/kwargs are passed verbatim (not indexed)."""
        primary_t = torch.ones(3)
        captured = {}

        def _capture(t, d):
            captured["d"] = d
            return t

        per_modality(_capture, primary_t, {"video": 1.0, "audio": 2.0})
        assert captured["d"] == {"video": 1.0, "audio": 2.0}


# ═════════════════════════════════════════════════════════════════════════
# tmap
# ═════════════════════════════════════════════════════════════════════════


class TestTmap:
    def test_dict_single(self):
        result = tmap(lambda t: t * 2, _make_dict())
        assert isinstance(result, dict)
        assert torch.allclose(result["x"], torch.full((2, 3), 2.0))
        assert torch.allclose(result["y"], torch.full((2, 5), 2.0))

    def test_dict_multi_input(self):
        a, b = _make_dict(), _make_dict()
        result = tmap(lambda x, y: x + y, a, b)
        assert torch.allclose(result["x"], torch.full((2, 3), 2.0))
        assert torch.allclose(result["y"], torch.full((2, 5), 2.0))

    def test_plain_tensor(self):
        t = torch.ones(4, 3)
        result = tmap(lambda x: x * 3, t)
        assert torch.allclose(result, torch.full((4, 3), 3.0))

    def test_list(self):
        result = tmap(lambda t: t + 1, _make_list())
        assert isinstance(result, list)
        assert len(result) == 2
        assert torch.allclose(result[0], torch.full((2, 3), 2.0))

    def test_tuple(self):
        result = tmap(lambda t: t + 1, _make_tuple())
        assert isinstance(result, tuple)
        assert torch.allclose(result[1], torch.full((2, 5), 2.0))

    def test_namedtuple(self):
        result = tmap(lambda t: t * 5, _make_namedtuple())
        assert isinstance(result, Pair)
        assert torch.allclose(result.first, torch.full((2, 3), 5.0))
        assert torch.allclose(result.second, torch.full((2, 5), 5.0))

    def test_non_tensor_leaf_passthrough(self):
        tree = {"x": torch.ones(2, 3), "label": "hello"}
        result = tmap(lambda t: t * 2, tree)
        assert result["label"] == "hello"
        assert torch.allclose(result["x"], torch.full((2, 3), 2.0))

    def test_nested_dict(self):
        tree = {"outer": {"inner": torch.ones(2, 3)}}
        result = tmap(lambda t: t + 10, tree)
        assert torch.allclose(result["outer"]["inner"], torch.full((2, 3), 11.0))

    def test_multi_input_with_scalar(self):
        a = {"x": torch.ones(3)}
        result = tmap(lambda t, s: t * s, a, {"x": 4.0})
        assert torch.allclose(result["x"], torch.full((3,), 4.0))

    def test_preserves_keys(self):
        tree = {"alpha": torch.ones(1), "beta": torch.zeros(1)}
        result = tmap(lambda t: t, tree)
        assert set(result.keys()) == {"alpha", "beta"}


# ═════════════════════════════════════════════════════════════════════════
# assert_same
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        pytest.param(
            torch.zeros(2, 3, dtype=torch.float32),
            torch.ones(4, dtype=torch.float32),
            id="single-tensor",
        ),
        pytest.param(
            {
                "video": torch.zeros(2, 3, dtype=torch.float32),
                "audio": torch.zeros(2, 4, dtype=torch.bfloat16),
            },
            {
                "video": torch.ones(5, dtype=torch.float32),
                "audio": torch.ones(6, dtype=torch.bfloat16),
            },
            id="multimodal-state-dict",
        ),
        pytest.param(
            (torch.zeros(2, 3, dtype=torch.bfloat16), torch.zeros(2, 3, dtype=torch.long)),
            (torch.ones(4, 5, dtype=torch.bfloat16), torch.ones(4, 5, dtype=torch.long)),
            id="sd15-qwen-ltx-embedding-mask-tuple",
        ),
        pytest.param(
            [torch.zeros(2, 3, dtype=torch.bfloat16), torch.zeros(2, 4, dtype=torch.bfloat16)],
            [torch.ones(5, 6, dtype=torch.bfloat16), torch.ones(5, 7, dtype=torch.bfloat16)],
            id="sdxl-embedding-list",
        ),
        pytest.param(
            (torch.zeros(2, 4, dtype=torch.bfloat16), torch.zeros(2, 3, 4, dtype=torch.bfloat16)),
            (torch.ones(5, 4, dtype=torch.bfloat16), torch.ones(5, 6, 4, dtype=torch.bfloat16)),
            id="flux-pooled-sequence-tuple",
        ),
        pytest.param(
            {
                "text_embeds": (None, torch.zeros(2, 3, dtype=torch.bfloat16)),
                "condition_image": torch.zeros(2, 4, dtype=torch.float32),
            },
            {
                "text_embeds": (None, torch.ones(5, 3, dtype=torch.bfloat16)),
                "condition_image": torch.ones(5, 4, dtype=torch.float32),
            },
            id="flux2-image-edit-dict",
        ),
        pytest.param(
            {
                "input_ids": [torch.zeros(3, dtype=torch.long), torch.zeros(5, dtype=torch.long)],
                "condition_mask": torch.zeros(2, 1, dtype=torch.bool),
                "conditioning_latents": torch.zeros(2, 4, dtype=torch.float32),
            },
            {
                "input_ids": [torch.ones(4, dtype=torch.long), torch.ones(2, dtype=torch.long)],
                "condition_mask": torch.ones(2, 1, dtype=torch.bool),
                "conditioning_latents": torch.ones(2, 4, dtype=torch.float32),
            },
            id="cosmos3-i2v-dict",
        ),
        pytest.param(
            {
                "prompt_embeds": torch.zeros(2, 3, dtype=torch.bfloat16),
                "text_token_tags": torch.zeros(3, dtype=torch.long),
                "cond_video_rows": torch.zeros(1, 2, 4, dtype=torch.float32),
                "keyframe_anchors": ("first", "last"),
            },
            {
                "prompt_embeds": torch.ones(2, 5, dtype=torch.bfloat16),
                "text_token_tags": torch.ones(5, dtype=torch.long),
                "cond_video_rows": torch.ones(1, 3, 4, dtype=torch.float32),
                "keyframe_anchors": ("first", "last"),
            },
            id="minimax-h3-fl2va-dict",
        ),
        pytest.param(
            {
                "orig_condition": torch.zeros(2, 3, dtype=torch.float32),
                "aug_condition": torch.zeros(2, 4, dtype=torch.float32),
            },
            {
                "orig_condition": torch.ones(5, 3, dtype=torch.float32),
                "aug_condition": torch.ones(5, 4, dtype=torch.float32),
            },
            id="edm-augmentation-dict",
        ),
        pytest.param(
            Pair(
                first=torch.zeros(2, dtype=torch.float32),
                second=torch.zeros(3, dtype=torch.bool),
            ),
            Pair(
                first=torch.ones(4, dtype=torch.float32),
                second=torch.ones(5, dtype=torch.bool),
            ),
            id="namedtuple-tree",
        ),
    ],
)
def test_assert_same_accepts_conditioning_and_multimodal_trees(positive, negative):
    # Dtype is compared per corresponding leaf; different leaves and branches may
    # intentionally have different dtypes and shapes.
    assert_same(dtype=True, positive=positive, negative=negative)


def test_assert_same_accepts_optional_none_and_multiple_trees():
    reference = torch.zeros(2, 3, dtype=torch.bfloat16)
    assert_same(
        shape=True,
        dtype=True,
        device=True,
        optional=None,
        reference=reference,
        peer=torch.ones_like(reference),
        third=torch.empty_like(reference),
    )
    assert_same(shape=True, optional=None, only=reference)
    assert_same(shape=True, first=None, second=None)


def test_assert_same_skips_none_leaves():
    assert_same(
        dtype=True,
        reference={
            "missing_from_reference": None,
            "missing_from_peer": torch.zeros(1, dtype=torch.float32),
            "checked": torch.zeros(1, dtype=torch.bfloat16),
        },
        peer={
            "missing_from_reference": torch.zeros(1, dtype=torch.float64),
            "missing_from_peer": None,
            "checked": torch.ones(2, dtype=torch.bfloat16),
        },
    )


def test_assert_same_first_structure_drives_the_walk():
    # Like ``tmap``, the walk visits the positions where the first structure holds a
    # tensor; a ``None`` there skips the position for every structure.
    peers = {
        "second": {"text_embeds": torch.zeros(1, dtype=torch.float32)},
        "third": {"text_embeds": torch.zeros(1, dtype=torch.float64)},
    }
    with pytest.raises(AssertionError, match=r"third\.dtype"):
        assert_same(dtype=True, **peers)
    assert_same(dtype=True, first={"text_embeds": None}, **peers)


def test_assert_same_names_mismatched_metadata():
    reference = torch.zeros(2, 3, dtype=torch.float32)

    with pytest.raises(AssertionError, match=r"condition\.shape .*\[3, 2\].*reference\.shape .*\[2, 3\]"):
        assert_same(shape=True, reference=reference, condition=torch.ones(3, 2))
    with pytest.raises(AssertionError, match=r"condition\.dtype .*torch\.float64.*reference\.dtype .*torch\.float32"):
        assert_same(dtype=True, reference=reference, condition=torch.ones(2, 3, dtype=torch.float64))
    with pytest.raises(AssertionError, match=r"condition\.device .*meta.*reference\.device .*cpu"):
        assert_same(device=True, reference=reference, condition=torch.empty(1, device="meta"))


def test_assert_same_accepts_metadata_targets():
    values = {
        "first": torch.zeros(2, 3, dtype=torch.bfloat16),
        "second": [torch.ones(2, 3, dtype=torch.bfloat16)],
    }

    assert_same(shape=(2, 3), dtype=torch.bfloat16, device=torch.device("cpu"), values=values)


@pytest.mark.parametrize(
    ("constraint", "message"),
    [
        pytest.param({"shape": (3, 2)}, r"value\.shape .*must be \(3, 2\)", id="shape"),
        pytest.param({"dtype": torch.float64}, r"value\.dtype .*must be torch\.float64", id="dtype"),
        pytest.param({"device": torch.device("meta")}, r"value\.device .*must be meta", id="device"),
    ],
)
def test_assert_same_names_target_metadata_mismatch(constraint, message):
    value = torch.zeros(2, 3, dtype=torch.float32)

    with pytest.raises(AssertionError, match=message):
        assert_same(value=value, **constraint)


def test_assert_same_checks_targets_on_each_structure():
    # Unlike comparisons, a target reaches positions where an earlier structure holds ``None``
    # and needs no common layout.
    first = {"video": torch.zeros(1), "audio": None}
    second = {"video": torch.zeros(1), "audio": torch.zeros(1, dtype=torch.float64)}
    with pytest.raises(AssertionError, match=r"second\.dtype .*must be torch\.float32"):
        assert_same(dtype=torch.float32, first=first, second=second)

    assert_same(
        dtype=torch.float32,
        device=torch.device("cpu"),
        real=first,
        noise=torch.zeros(2),
        path=[torch.ones(3)],
    )


def test_assert_same_comparisons_keep_the_layout_check_next_to_targets():
    with pytest.raises(AssertionError, match="dict key mismatch"):
        assert_same(shape=True, dtype=torch.float32, positive={"video": torch.zeros(1)}, negative=torch.zeros(1))


def test_assert_same_rejects_non_tensor_structures_for_targets():
    with pytest.raises(TypeError, match="weight to be a Tensor, a container, or None, got float"):
        assert_same(dtype=torch.float32, loss=torch.zeros(1), weight=1.0)


def test_assert_same_none_and_false_skip_properties():
    reference = torch.zeros(2, 3, dtype=torch.float32)
    condition = torch.zeros(3, 2, dtype=torch.float64)

    assert_same(shape=None, dtype=False, device=True, reference=reference, condition=condition)


def test_assert_same_checks_nothing_when_every_property_is_skipped():
    """``dtype=module.precision_model`` checks nothing for a module without one (no casts)."""
    assert_same(dtype=None, reference=torch.zeros(2), condition=torch.zeros(3, dtype=torch.float64))
    assert_same(condition=torch.zeros(1))


def test_assert_same_rejects_nested_dtype_mismatch():
    positive = {"text_embeds": (None, torch.zeros(2, dtype=torch.bfloat16))}
    negative = {"text_embeds": (None, torch.zeros(3, dtype=torch.float32))}

    with pytest.raises(AssertionError, match=r"negative\.dtype"):
        assert_same(dtype=True, positive=positive, negative=negative)


@pytest.mark.parametrize(
    ("positive", "negative", "message"),
    [
        pytest.param(
            {"video": torch.zeros(1), "audio": torch.zeros(1)},
            {"video": torch.zeros(1)},
            "dict key mismatch",
            id="dict-keys",
        ),
        pytest.param(
            [torch.zeros(1), torch.zeros(1)],
            [torch.zeros(1)],
            "sequence length mismatch",
            id="sequence-length",
        ),
        pytest.param(
            {"video": torch.zeros(1)},
            torch.zeros(1),
            "dict key mismatch",
            id="no-tensor-broadcast",
        ),
    ],
)
def test_assert_same_requires_matching_tree_structure(positive, negative, message):
    with pytest.raises(AssertionError, match=message):
        assert_same(dtype=True, positive=positive, negative=negative)


@pytest.mark.parametrize("tree", [{}, {"optional": None}, {"metadata": "text"}, [None, ("metadata",)]])
def test_assert_same_accepts_tensor_free_nested_trees(tree):
    assert_same(dtype=True, first=tree, second=tree)


def test_assert_same_observes_actual_tensor_dtype_under_autocast():
    reference = torch.ones(2, 2, dtype=torch.float32)
    peer = torch.ones_like(reference)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert_same(dtype=True, reference=reference, peer=peer)
        autocast_output = reference @ peer

    assert autocast_output.dtype == torch.bfloat16
    with pytest.raises(AssertionError, match=r"autocast_output\.dtype"):
        assert_same(dtype=True, reference=reference, autocast_output=autocast_output)


# ═════════════════════════════════════════════════════════════════════════
# get_device / get_dtype / get_batch_size / get_shape
# ═════════════════════════════════════════════════════════════════════════


class TestIntrospectionHelpers:
    def test_get_device_tensor(self):
        assert get_device(torch.zeros(3)).type == "cpu"

    def test_get_device_dict(self):
        assert get_device({"x": torch.zeros(3), "y": torch.zeros(4)}).type == "cpu"

    def test_get_dtype_tensor(self):
        assert get_dtype(torch.zeros(3, dtype=torch.float64)) is torch.float64

    def test_get_dtype_dict(self):
        assert get_dtype({"x": torch.zeros(3, dtype=torch.float16)}) is torch.float16

    def test_get_batch_size_tensor(self):
        assert get_batch_size(torch.zeros(7, 3, 5)) == 7

    def test_get_batch_size_dict(self):
        assert get_batch_size({"video": torch.zeros(7, 4, 8), "audio": torch.zeros(7, 4, 128)}) == 7

    def test_get_device_nested_tuple(self):
        # Walks into nested tuples to find a tensor leaf (e.g. Flux2's
        # ``(None, prompt_embeds)`` text-encoder contract).
        assert get_device((None, torch.zeros(3))).type == "cpu"

    def test_get_dtype_nested_namedtuple(self):
        tree = {"k": Pair(first=None, second=[torch.zeros(2, dtype=torch.float64)])}
        assert get_dtype(tree) is torch.float64

    def test_get_batch_size_nested_dict(self):
        assert get_batch_size({"k": (None, torch.zeros(7, 3))}) == 7

    def test_get_device_raises_on_empty(self):
        with pytest.raises(ValueError, match="no Tensor leaf"):
            get_device({})

    def test_get_device_raises_on_non_tensor(self):
        with pytest.raises(ValueError, match="no Tensor leaf"):
            get_device("hello")

    def test_get_device_raises_on_all_none_tree(self):
        with pytest.raises(ValueError, match="no Tensor leaf"):
            get_device({"a": (None, None), "b": [None]})

    def test_get_shape_tensor(self):
        assert get_shape(torch.zeros(2, 3)) == torch.Size([2, 3])

    def test_get_shape_dict(self):
        out = get_shape({"video": torch.zeros(2, 4, 8), "audio": torch.zeros(2, 4, 128)})
        assert out == {"video": torch.Size([2, 4, 8]), "audio": torch.Size([2, 4, 128])}


# ═════════════════════════════════════════════════════════════════════════
# to_dict
# ═════════════════════════════════════════════════════════════════════════


class TestToDict:
    def test_dict_with_suffix(self):
        pred = {"x": torch.randn(2, 3), "y": torch.randn(2, 5)}
        target = {"x": torch.randn(2, 3), "y": torch.randn(2, 5)}
        losses = tmap(torch.nn.functional.mse_loss, pred, target)
        result = to_dict(losses, suffix="loss")
        assert "x_loss" in result and "y_loss" in result
        assert all(v.item() >= 0 for v in result.values())

    def test_arbitrary_keys(self):
        pred = {"x": torch.randn(2, 3), "y": torch.randn(2, 5)}
        target = {"x": torch.randn(2, 3), "y": torch.randn(2, 5)}
        losses = tmap(torch.nn.functional.mse_loss, pred, target)
        result = to_dict(losses, suffix="l")
        assert "x_l" in result and "y_l" in result

    def test_sum_pattern(self):
        tree = _make_dict()
        per_key_sums = tmap(lambda x: x.sum(), tree)
        losses = to_dict(per_key_sums, suffix="loss")
        total = sum(losses.values())
        assert total.item() == 2 * 3 + 2 * 5

    def test_single_tensor(self):
        t = torch.tensor(3.14)
        result = to_dict(t, suffix="loss")
        assert "loss" in result and result["loss"] is t

    def test_no_suffix(self):
        result = to_dict({"a": torch.tensor(1.0)})
        assert "a_" in result


# ═════════════════════════════════════════════════════════════════════════
# randn_like
# ═════════════════════════════════════════════════════════════════════════


class TestRandnLike:
    def test_dict(self):
        noise = randn_like(_make_dict())
        assert isinstance(noise, dict)
        assert noise["x"].shape == (2, 3)
        assert noise["y"].shape == (2, 5)

    def test_dtype_override(self):
        noise = randn_like(_make_dict(), dtype=torch.float16)
        assert noise["x"].dtype == torch.float16
        assert noise["y"].dtype == torch.float16

    def test_plain_tensor(self):
        t = torch.ones(4, 6)
        noise = randn_like(t)
        assert noise.shape == (4, 6)
        assert noise.dtype == t.dtype

    def test_stochastic(self):
        """Two calls should produce different noise."""
        a = randn_like(_make_dict())
        b = randn_like(_make_dict())
        assert not torch.allclose(a["x"], b["x"])


# ═════════════════════════════════════════════════════════════════════════
# cast_like
# ═════════════════════════════════════════════════════════════════════════


class TestCastLike:
    def test_same_takes_each_modality_dtype(self):
        data = {"video": torch.ones(2, dtype=torch.float64), "audio": torch.ones(3, dtype=torch.float64)}
        reference = {"video": torch.zeros(1, dtype=torch.bfloat16), "audio": torch.zeros(1, dtype=torch.float32)}
        for result in (cast_like(data, reference), cast_like(data, reference, "same")):
            assert result["video"].dtype == torch.bfloat16
            assert result["audio"].dtype == torch.float32

    def test_mapping_gives_each_modality_dtype(self):
        data = {"video": torch.ones(2, dtype=torch.float64), "audio": torch.ones(3, dtype=torch.float64)}
        result = cast_like(data, data, {"video": torch.bfloat16, "audio": torch.float32})
        assert result["video"].dtype == torch.bfloat16
        assert result["audio"].dtype == torch.float32

    def test_none_keeps_the_data(self):
        data = _make_dict()
        assert cast_like(data, torch.zeros(1, dtype=torch.float16), None) is data

    def test_rejects_other_strings(self):
        with pytest.raises(ValueError, match="'same' or None"):
            cast_like(_make_dict(), torch.zeros(1), "float32")

    @pytest.mark.parametrize(
        "dtype",
        [
            {"x": torch.float32},
            {"x": torch.float32, "y": torch.float32, "z": torch.float32},
            {"x": torch.float32, "y": None},
            {"x": torch.float32, "y": "same"},
        ],
        ids=["missing-modality", "extra-modality", "none-value", "string-value"],
    )
    def test_rejects_a_mapping_without_a_dtype_per_modality(self, dtype):
        with pytest.raises(ValueError, match="per modality of"):
            cast_like(_make_dict(), _make_dict(), dtype)

    def test_rejects_a_mapping_for_a_tensor(self):
        with pytest.raises(ValueError, match="needs multimodal data"):
            cast_like(torch.ones(2), torch.ones(2), {"x": torch.float32})

    def test_dtype_overrides_reference(self):
        reference = {"x": torch.zeros(1, dtype=torch.bfloat16), "y": torch.zeros(1, dtype=torch.float32)}
        result = cast_like(_make_dict(), reference, dtype=torch.float64)
        assert result["x"].dtype == torch.float64
        assert result["y"].dtype == torch.float64

    def test_tensor_reference_applies_to_every_modality(self):
        result = cast_like(_make_dict(), torch.zeros(1, dtype=torch.float16))
        assert result["x"].dtype == torch.float16
        assert result["y"].dtype == torch.float16

    def test_plain_tensor_without_copy_when_dtype_matches(self):
        t = torch.ones(2, 3, dtype=torch.float64)
        assert cast_like(t, torch.zeros(1, dtype=torch.bfloat16)).dtype == torch.bfloat16
        assert cast_like(t, torch.zeros(1, dtype=torch.float64)) is t

    def test_casts_integer_dtypes_and_keeps_device(self):
        """Integer tensors are cast too, as with ``to``, and no tensor follows the reference's device."""
        data = {"latents": torch.ones(2), "ids": torch.arange(2)}
        result = cast_like(data, torch.empty(1, dtype=torch.float16, device="meta"))
        assert result["latents"].dtype == torch.float16 and result["latents"].device.type == "cpu"
        assert result["ids"].dtype == torch.float16 and result["ids"].device.type == "cpu"


# ═════════════════════════════════════════════════════════════════════════
# index / slice_dim
# ═════════════════════════════════════════════════════════════════════════


class TestIndex:
    def test_plain_tensor(self):
        t = torch.arange(10)
        assert torch.equal(index(t, 3), t[3])

    def test_dict_shares_the_index(self):
        tree = {"video": torch.arange(10), "audio": torch.arange(10) * 2}
        out = index(tree, 4)
        assert torch.equal(out["video"], tree["video"][4])
        assert torch.equal(out["audio"], tree["audio"][4])

    def test_negative_and_tensor_index(self):
        tree = {"video": torch.arange(10), "audio": torch.arange(10) * 2}
        assert torch.equal(index(tree, -1)["audio"], tree["audio"][-1])
        ids = torch.tensor([0, 2, 2])
        assert torch.equal(index(tree, ids)["video"], tree["video"][ids])

    def test_tuple_index_spans_axes(self):
        t = torch.arange(24).reshape(6, 4)
        rows = torch.tensor([0, 3])
        assert torch.equal(index(t, (rows, 1)), t[rows, 1])

    def test_per_modality_index(self):
        """A mapping index follows ``per_modality``: each modality gets its own."""
        tree = {"video": torch.arange(10), "audio": torch.arange(10) * 2}
        out = index(tree, {"video": 1, "audio": 5})
        assert int(out["video"]) == 1 and int(out["audio"]) == 10


class TestSliceDim:
    # ── 1-/2-/3-arg overloads on a plain tensor (default dim=0) ─────────

    def test_stop_only(self):
        """``slice_dim(t, n)`` mirrors ``slice(n)`` → ``t[:n]``."""
        t = torch.arange(10)
        assert torch.equal(slice_dim(t, 3), t[:3])

    def test_start_stop(self):
        t = torch.arange(10)
        assert torch.equal(slice_dim(t, 2, 7), t[2:7])

    def test_start_stop_step(self):
        t = torch.arange(10)
        assert torch.equal(slice_dim(t, 0, 10, 2), t[0:10:2])

    def test_none_placeholders(self):
        """Any of start/stop/step may be ``None`` (Python slice semantics)."""
        t = torch.arange(10)
        assert torch.equal(slice_dim(t, None, None, 2), t[::2])
        assert torch.equal(slice_dim(t, 3, None), t[3:])
        assert torch.equal(slice_dim(t, None, 5), t[:5])

    # ── default-dim batch slicing ───────────────────────────────────────

    def test_default_dim_is_batch(self):
        """``dim`` defaults to the batch dimension 0."""
        t = torch.arange(24).reshape(6, 4)
        assert torch.equal(slice_dim(t, 2), t[:2])

    # ── positive dim ────────────────────────────────────────────────────

    def test_positive_dim(self):
        t = torch.arange(24).reshape(2, 3, 4)
        assert torch.equal(slice_dim(t, 2, dim=1), t[:, :2])
        assert torch.equal(slice_dim(t, 1, 3, dim=2), t[:, :, 1:3])

    def test_positive_dim_step(self):
        t = torch.arange(60).reshape(2, 3, 10)
        assert torch.equal(slice_dim(t, 0, 10, 2, dim=2), t[:, :, 0:10:2])

    # ── negative dim (Ellipsis-based, rank-agnostic) ────────────────────

    def test_negative_dim_last(self):
        """``dim=-1`` slices the last axis regardless of rank."""
        t = torch.arange(24).reshape(2, 3, 4)
        assert torch.equal(slice_dim(t, 2, dim=-1), t[..., :2])

    def test_negative_dim_second_to_last(self):
        t = torch.arange(24).reshape(2, 3, 4)
        assert torch.equal(slice_dim(t, 2, dim=-2), t[..., :2, :])

    def test_negative_dim_step(self):
        t = torch.arange(40).reshape(2, 4, 5)
        assert torch.equal(slice_dim(t, None, None, 2, dim=-1), t[..., ::2])

    # ── tensor bounds ───────────────────────────────────────────────────

    def test_tensor_bounds(self):
        """``slice(start, stop)`` takes whatever it is given, so Tensor bounds work as written."""
        t = torch.arange(10)
        start = torch.zeros(1, dtype=torch.long)
        for block in (1, 3, 4):
            assert torch.equal(slice_dim(t, start, start + block), t[start : start + block])
            assert torch.equal(slice_dim(t, start, start + block), t[:block])

    def test_tensor_bounds_zero_dim(self):
        t = torch.arange(10)
        start = torch.tensor(2)
        assert torch.equal(slice_dim(t, start, start + 3), t[start : start + 3])

    def test_tensor_bounds_dict_and_dim(self):
        data = {"video": torch.arange(20).reshape(4, 5), "audio": torch.arange(12).reshape(4, 3)}
        start = torch.zeros(1, dtype=torch.long)
        out = slice_dim(data, start, start + 2, dim=1)
        assert torch.equal(out["video"], data["video"][:, :2])
        assert torch.equal(out["audio"], data["audio"][:, :2])

    # ── multimodal: dict primary ───────────────────────────────────────

    def test_dict_default_dim(self):
        """Slicing batch dim broadcasts over modalities (the wandb.py call site)."""
        data = {"video": torch.arange(20).reshape(5, 4), "audio": torch.arange(15).reshape(5, 3)}
        out = slice_dim(data, 2)
        assert isinstance(out, dict)
        assert torch.equal(out["video"], data["video"][:2])
        assert torch.equal(out["audio"], data["audio"][:2])

    def test_dict_positive_dim(self):
        data = {"x": torch.arange(12).reshape(3, 4), "y": torch.arange(6).reshape(3, 2)}
        out = slice_dim(data, 1, dim=1)
        assert torch.equal(out["x"], data["x"][:, :1])
        assert torch.equal(out["y"], data["y"][:, :1])

    def test_dict_negative_dim_with_mixed_ndim(self):
        """``dim=-1`` works when modalities have different ranks (Ellipsis-based indexing)."""
        data = {
            "video": torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5),
            "audio": torch.arange(2 * 5).reshape(2, 5),
        }
        out = slice_dim(data, 2, dim=-1)
        assert out["video"].shape == (2, 3, 4, 2)
        assert out["audio"].shape == (2, 2)
        assert torch.equal(out["video"], data["video"][..., :2])
        assert torch.equal(out["audio"], data["audio"][..., :2])

    # ── view / aliasing ─────────────────────────────────────────────────

    def test_returns_view_no_copy(self):
        """Pure slicing returns a view; no underlying data is copied."""
        t = torch.arange(20).reshape(5, 4)
        out = slice_dim(t, 2)
        assert out.data_ptr() == t.data_ptr()

    # ── error modes ─────────────────────────────────────────────────────

    def test_no_args_raises(self):
        """Mirrors builtin ``slice()`` which requires at least one arg."""
        with pytest.raises(TypeError):
            slice_dim(torch.arange(4))


# ═════════════════════════════════════════════════════════════════════════
# detach
# ═════════════════════════════════════════════════════════════════════════


class TestDetach:
    def test_dict(self):
        v = torch.randn(2, 3, requires_grad=True)
        tree = {"x": v, "y": torch.randn(2, 5)}
        detached = detach(tree)
        assert not detached["x"].requires_grad

    def test_plain_tensor(self):
        t = torch.randn(3, requires_grad=True)
        result = detach(t)
        assert not result.requires_grad

    def test_list(self):
        lst = [torch.randn(2, requires_grad=True), torch.randn(3, requires_grad=True)]
        result = detach(lst)
        assert all(not r.requires_grad for r in result)

    def test_values_preserved(self):
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = detach(t)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))


# ═════════════════════════════════════════════════════════════════════════
# to (dtype / device casting)
# ═════════════════════════════════════════════════════════════════════════


class TestTo:
    def test_dtype_dict(self):
        result = to(_make_dict(), dtype=torch.float16)
        assert isinstance(result, dict)
        assert result["x"].dtype == torch.float16
        assert result["y"].dtype == torch.float16

    def test_dtype_plain_tensor(self):
        result = to(torch.ones(2, 3), dtype=torch.bfloat16)
        assert result.dtype == torch.bfloat16

    def test_nested_dict(self):
        data = {"real": _make_dict(), "condition": torch.randn(2, 10)}
        result = to(data, dtype=torch.float16)
        assert result["real"]["x"].dtype == torch.float16
        assert result["condition"].dtype == torch.float16

    def test_list(self):
        result = to(_make_list(), dtype=torch.float64)
        assert all(r.dtype == torch.float64 for r in result)

    def test_to_cpu(self):
        result = to(_make_dict(), device="cpu")
        assert result["x"].device.type == "cpu"

    def test_no_context_is_identity(self):
        data = {"tensor": torch.ones(1), "optional": None}
        assert to(data) is data

    def test_integer_and_bool_tensors_are_cast_unless_cast_ints_is_false(self):
        data = {"index": torch.arange(3), "mask": torch.ones(3, dtype=torch.bool), "x": torch.ones(3)}
        converted = to(data, dtype=torch.float64)
        assert all(value.dtype == torch.float64 for value in converted.values())
        kept = to(data, dtype=torch.float64, cast_ints=False)
        assert (kept["index"].dtype, kept["mask"].dtype, kept["x"].dtype) == (torch.int64, torch.bool, torch.float64)


# ═════════════════════════════════════════════════════════════════════════
# noise_from_shape
# ═════════════════════════════════════════════════════════════════════════


class TestNoiseFromShape:
    def test_unimodal(self):
        noise = noise_from_shape([3, 8, 8], batch_size=4, device="cpu", dtype=torch.float32)
        assert isinstance(noise, torch.Tensor)
        assert noise.shape == (4, 3, 8, 8)

    def test_multimodal(self):
        spec = {"x": [3, 16, 16], "y": [1, 80]}
        noise = noise_from_shape(spec, batch_size=2, device="cpu", dtype=torch.float32)
        assert isinstance(noise, dict)
        assert noise["x"].shape == (2, 3, 16, 16)
        assert noise["y"].shape == (2, 1, 80)

    def test_dtype(self):
        noise = noise_from_shape([3], batch_size=1, device="cpu", dtype=torch.float16)
        assert noise.dtype == torch.float16

    def test_multimodal_dtype(self):
        spec = {"v": [4], "a": [8]}
        noise = noise_from_shape(spec, batch_size=1, device="cpu", dtype=torch.bfloat16)
        assert noise["v"].dtype == torch.bfloat16
        assert noise["a"].dtype == torch.bfloat16


# ═════════════════════════════════════════════════════════════════════════
# expand_first_dim
# ═════════════════════════════════════════════════════════════════════════


class TestExpandFirstDim:
    def test_dict(self):
        tree = {"x": torch.randn(1, 3), "y": torch.randn(1, 5)}
        result = expand_first_dim(tree, batch_size=4)
        assert result["x"].shape == (4, 3)
        assert result["y"].shape == (4, 5)

    def test_plain_tensor(self):
        t = torch.randn(1, 8)
        result = expand_first_dim(t, batch_size=3)
        assert result.shape == (3, 8)

    def test_no_copy(self):
        t = torch.randn(1, 4)
        result = expand_first_dim(t, batch_size=2)
        assert result.data_ptr() == t.data_ptr()

    def test_list(self):
        lst = [torch.randn(1, 3), torch.randn(1, 5)]
        result = expand_first_dim(lst, batch_size=4)
        assert result[0].shape == (4, 3)
        assert result[1].shape == (4, 5)
