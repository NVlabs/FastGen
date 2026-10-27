# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exports for the vanilla KV-cache package."""

from fastgen.networks.kv_cache.base import (
    BlockMask,
    BaseKVCacheEngine,
    CAUSAL_FLEX_ATTENTION_AVAILABLE,
    FLEX_ATTENTION_AVAILABLE,
    flex_attention,
    prepare_blockwise_causal_attn_mask,
    prepare_teacher_forcing_attn_mask,
)
from fastgen.networks.kv_cache.vanilla_engine import VanillaKVCacheEngine

KVCacheEngine = VanillaKVCacheEngine

__all__ = [
    "BlockMask",
    "BaseKVCacheEngine",
    "CAUSAL_FLEX_ATTENTION_AVAILABLE",
    "FLEX_ATTENTION_AVAILABLE",
    "KVCacheEngine",
    "VanillaKVCacheEngine",
    "flex_attention",
    "prepare_blockwise_causal_attn_mask",
    "prepare_teacher_forcing_attn_mask",
]
