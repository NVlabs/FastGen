# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FlextronConfig — all Elastification config fields in one place.

Ports `FlextronConfig` from `megatron/elastification/flextron_config.py`.

Section order, field order, field names, and defaults mirror the reference.
Dropped (out of Wan scope): all Mamba fields, all MoE fields, all memory-mode
fields, `basemodel_type`. Added (Wan needs): `lr_mult_router`, `train_iters`,
`num_mlp_blocks` — noted inline.

Framework: `@attrs.define(slots=False)` is used instead of Megatron's
`@dataclass` so FlextronConfig composes with FastGen's attrs-based ModelConfig
hierarchy (see `fastgen/configs/methods/config_sft.py`).
"""

from __future__ import annotations

from typing import List, Optional

import attrs


@attrs.define(slots=False)
class FlextronConfig:
    # ── Core flags ────────────────────────────────────────────────────────────
    flextron: bool = False
    binary_mask: bool = False
    add_skipping: bool = False
    no_attn_skip: bool = False
    slice: bool = False
    soft_mask: bool = False

    # ── Router ────────────────────────────────────────────────────────────────
    enable_router: bool = False
    router_inter_dim: int = 128
    hard_sample_th: float = 0.996
    router_beta: float = 1.0
    loss_alpha: float = 1.0
    tau_init: float = 1.0
    tau_decay: float = 0.9999
    router_std: float = 0.1
    router_gbs: int = 32
    normalize_router_logits: bool = False
    linear_scaler_start: Optional[float] = None
    linear_scaler_end: Optional[float] = None
    # Megatron: CLI arg on the training script.
    lr_mult_router: float = 1.0
    # Megatron: pulled from `args.train_iters`.
    train_iters: Optional[int] = None

    # ── Budget ────────────────────────────────────────────────────────────────
    budget_probs: Optional[List[float]] = None
    budget_list: Optional[List[float]] = None
    budget_type: str = 'param'
    disable_budget: bool = False

    # ── Training / eval control ───────────────────────────────────────────────
    is_flex_eval: bool = False
    freeze_router: bool = False
    freeze_model: bool = False
    curr_iteration: Optional[int] = None
    original_model_sample_prob: float = 0.33
    override_selected_budget: Optional[List[float]] = None

    # ── Layer-skip constraints ────────────────────────────────────────────────
    skip_num_attn_layer_constraint: Optional[int] = None
    skip_total_layer_constraint: Optional[int] = None
    layer_ranking_list: Optional[List[int]] = None
    # Megatron inherits `num_layers` from TransformerConfig; the manager
    # populates this from `net.transformer.blocks`.
    num_layers: Optional[int] = None

    # ── Force overrides (eval / frozen-router mode) ───────────────────────────
    force_router_skip: Optional[List[int]] = None
    force_mlp: Optional[List[float]] = None
    force_emb: Optional[List[float]] = None

    # ── Choice lists (converted to int at model-setup time) ───────────────────
    mlp_per_list: Optional[List[float]] = None
    emb_per_list: Optional[List[float]] = None
    mlp_int_list: Optional[List[int]] = None
    emb_int_list: Optional[List[int]] = None

    # ── Heterogeneous per-layer routing ───────────────────────────────────────
    flex_hetero_ffn: bool = False
    # Megatron derives per-block counts from `hybrid_layer_pattern`; Wan has no
    # such string, so the manager populates this from `net.transformer.blocks`.
    num_mlp_blocks: Optional[int] = None

    # ── Distillation ──────────────────────────────────────────────────────────
    distillation: bool = False
    distill_coeff: float = 0.0
    distill_only: bool = False

    def __attrs_post_init__(self):
        # Mirrors Megatron's `validate_flextron_per_int_lists` +
        # `sort_budget_list_descending` from `elastification/arguments.py`,
        # inlined here because we don't have a separate args-injection step.

        # Per-list / int-list mutual exclusion. Defaults per-list to [1.0]
        # (elasticity off) when neither is set on an axis.
        pairs = (
            ('mlp', 'mlp_per_list', 'mlp_int_list'),
            ('emb', 'emb_per_list', 'emb_int_list'),
        )
        for name, per_attr, int_attr in pairs:
            per_val = getattr(self, per_attr)
            int_val = getattr(self, int_attr)
            per_set = per_val is not None
            int_set = int_val is not None
            if per_set:
                for x in per_val:
                    assert 0.0 <= x <= 1.0, (
                        f'{per_attr} values must be in [0, 1], got {x}.'
                    )
            assert not (per_set and int_set), (
                f'Use either {per_attr} or {int_attr} for {name}, not both.'
            )
            if not per_set and not int_set:
                setattr(self, per_attr, [1.0])

        # Sort budget_list descending; permute budget_probs to match.
        if self.budget_list is not None and len(self.budget_list) > 1:
            if self.budget_probs is not None:
                assert len(self.budget_probs) == len(self.budget_list), (
                    f'budget_probs length {len(self.budget_probs)} does not '
                    f'match budget_list length {len(self.budget_list)}'
                )
            order = sorted(
                range(len(self.budget_list)),
                key=lambda i: -self.budget_list[i],
            )
            self.budget_list = [self.budget_list[i] for i in order]
            if self.budget_probs is not None:
                self.budget_probs = [self.budget_probs[i] for i in order]
