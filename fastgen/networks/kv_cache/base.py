# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base abstractions and shared helpers for vanilla Wan-style KV-cache management.

A `BaseKVCacheEngine` serves one block branch of a causal model: `"main"` for `transformer.blocks`, plus `"vace"` for
`transformer.vace_blocks` in causal VACE. Its storage:

- `_self_kv[block_idx][cache_tag]`: one self-attention entry per block and tag, with an engine-specific payload (see
  `vanilla_engine.py`).
- `_cross_kv[block_idx][cache_tag]`: one cross-attention entry `{"k", "v", "is_init"}` with K/V of shape
  `[B, L_enc, H, D]`, where `L_enc` covers the text tokens, preceded by the image tokens for Wan2.1 I2V blocks.
- `_module_to_block[id(attn_module)]`: the block index of each registered attention module, so a processor finds its
  entries without knowing its own index.
- `_block_masks[(device, num_frames, frame_seqlen, chunk_size, mask_kind)]`: memoized FlexAttention masks for
  full-sequence teacher- and diffusion-forcing passes.

`block_idx` is local to the engine, so block 0 of the `"vace"` engine is `transformer.vace_blocks[0]`. `cache_tag`
separates guidance branches, usually `"pos"` and `"neg"`.

Cross-attention entries are written once and read many times per rollout. `ensure_capacity` allocates zero K/V with
`is_init=False`. The attention processor fills the entry (`set_cross_kv`) on the first forward with `store_kv=True`,
and later no-grad forwards read it (`get_cross_kv`) instead of recomputing the K/V projections. Forwards with
gradients enabled always recompute K/V, so gradients reach the projections, and refresh the entry when
`store_kv=True`. A tag therefore assumes fixed conditioning and weights for the whole rollout: `clear` the cache
before changing either.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import os
from typing import Any

import torch

import fastgen.utils.logging_utils as logger


_disable_flex_env = os.environ.get("FASTGEN_DISABLE_FLEX_ATTENTION", "0") == "1"
try:
    if _disable_flex_env:
        raise ImportError("FlexAttention disabled via FASTGEN_DISABLE_FLEX_ATTENTION=1")
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

    CAUSAL_FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    CAUSAL_FLEX_ATTENTION_AVAILABLE = False
    create_block_mask = None  # type: ignore
    flex_attention = None  # type: ignore

    class BlockMask:  # type: ignore
        pass


if CAUSAL_FLEX_ATTENTION_AVAILABLE:
    _compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
    _disable_compile_wrap = (
        os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1" or os.environ.get("FASTGEN_FLEX_COMPILE", "1") == "0"
    )
    if not _disable_compile_wrap:
        flex_attention = torch.compile(flex_attention, dynamic=False, mode=_compile_mode)  # type: ignore[arg-type]

FLEX_ATTENTION_AVAILABLE = CAUSAL_FLEX_ATTENTION_AVAILABLE


def _build_block_token_ranges(
    num_frames: int,
    frame_seqlen: int,
    chunk_size: int,
) -> list[tuple[int, int]]:
    """Return token index ranges for causal chunks, matching FastGen chunk semantics."""
    num_frames = int(num_frames)
    frame_seqlen = int(frame_seqlen)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    total_tokens = num_frames * frame_seqlen
    if total_tokens <= 0:
        return []

    num_chunks = num_frames // chunk_size
    remaining_size = num_frames % chunk_size

    # FastGen puts any remainder frames in the first causal chunk. For example,
    # 8 frames with chunk_size=3 becomes chunks [0, 1, 2, 3, 4] and [5, 6, 7],
    # as in the AR sampling loops, Self-Forcing rollouts, and per-chunk training
    # timesteps (``sample_t_inhom``, ``sample_student_timesteps_inhom``).
    frame_counts = (
        [remaining_size] if num_chunks == 0 else [chunk_size + remaining_size] + [chunk_size] * (num_chunks - 1)
    )

    # Convert per-chunk frame counts into contiguous token ranges. The mask
    # builders operate on flattened video tokens, so every frame contributes
    # frame_seqlen consecutive tokens.
    block_ranges: list[tuple[int, int]] = []
    current_start = 0
    for frames_in_chunk in frame_counts:
        chunk_len_tokens = int(frames_in_chunk) * int(frame_seqlen)
        block_ranges.append((current_start, current_start + chunk_len_tokens))
        current_start += chunk_len_tokens

    return block_ranges


def prepare_blockwise_causal_attn_mask(
    device: torch.device | str,
    num_frames: int = 21,
    frame_seqlen: int = 1560,
    chunk_size: int = 1,
) -> BlockMask:
    """Construct a frame-chunk causal mask for teacher-forcing style full-video passes."""
    if not CAUSAL_FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "FlexAttention is required for teacher-forcing block masks. "
            "Install a PyTorch build with torch.nn.attention.flex_attention or set CausalContext.is_ar=True "
            "to use KV-cache autoregressive mode instead."
        )

    logger.info("creating blockwise causal attn mask for teacher-forcing or diffusion-forcing")

    total_length = num_frames * frame_seqlen
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

    for block_start, block_end in _build_block_token_ranges(
        num_frames=num_frames,
        frame_seqlen=frame_seqlen,
        chunk_size=chunk_size,
    ):
        ends[block_start:block_end] = block_end

    def attention_mask(b, h, q_idx, kv_idx) -> torch.Tensor:
        del b, h
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    return create_block_mask(  # type: ignore[misc]
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )


def prepare_teacher_forcing_attn_mask(
    device: torch.device | str,
    num_frames: int = 21,
    frame_seqlen: int = 1560,
    chunk_size: int = 1,
) -> BlockMask:
    """Construct the dual-stream teacher-forcing attention mask."""
    if not CAUSAL_FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "FlexAttention is required for teacher-forcing block masks. "
            "Install a PyTorch build with torch.nn.attention.flex_attention."
        )

    logger.info("creating teacher-forcing attention mask")

    clean_length = num_frames * frame_seqlen
    total_length = clean_length * 2
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

    block_ranges = _build_block_token_ranges(num_frames=num_frames, frame_seqlen=frame_seqlen, chunk_size=chunk_size)

    for block_start, block_end in block_ranges:
        context_ends[block_start:block_end] = block_end

    for block_start, block_end in block_ranges:
        noisy_start = clean_length + block_start
        noisy_end = clean_length + block_end
        noise_noise_starts[noisy_start:noisy_end] = noisy_start
        noise_noise_ends[noisy_start:noisy_end] = noisy_end
        noise_context_ends[noisy_start:noisy_end] = block_start

    def attention_mask(b, h, q_idx, kv_idx) -> torch.Tensor:
        del b, h
        # Clean stream: each clean query can see clean tokens through the end of
        # its current chunk, matching the blockwise full-context teacher signal.
        clean_mask = (q_idx < clean_length) & (kv_idx < context_ends[q_idx])

        # Noisy stream: each noisy query can see other noisy tokens in the same
        # chunk plus clean tokens from previous chunks only.
        same_noisy_block = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
        previous_clean_context = kv_idx < noise_context_ends[q_idx]
        noise_mask = (q_idx >= clean_length) & (same_noisy_block | previous_clean_context)
        return (q_idx == kv_idx) | clean_mask | noise_mask

    return create_block_mask(  # type: ignore[misc]
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )


class BaseKVCacheEngine(ABC):
    """Shared cache storage and registration helpers for self- and cross-attention KV state."""

    def __init__(self) -> None:
        self._self_kv: list[dict[str, dict[str, Any]]] = []
        self._cross_kv: list[dict[str, dict[str, Any]]] = []
        self._module_to_block: dict[int, int] = {}
        self._block_masks: dict[tuple[str, int, int, int, str], Any] = {}

    def _ensure_block_storage(self, block_count: int) -> None:
        """Resize per-block storage when the network block count changes."""
        need_reinit = len(self._self_kv) != block_count or len(self._cross_kv) != block_count
        if need_reinit:
            self._self_kv = [{} for _ in range(block_count)]
            self._cross_kv = [{} for _ in range(block_count)]

    def _ensure_cross_cache_entry(
        self,
        idx: int,
        cache_tag: str,
        cross_shape: tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Allocate or refresh the fixed-shape cross-attention cache entry for one block."""
        cross_entry = self._cross_kv[idx].get(cache_tag, {})
        k_cross = cross_entry.get("k", None)
        v_cross = cross_entry.get("v", None)
        need_alloc_cross = (
            not isinstance(k_cross, torch.Tensor)
            or not isinstance(v_cross, torch.Tensor)
            or tuple(k_cross.shape) != cross_shape
            or tuple(v_cross.shape) != cross_shape
        )
        if need_alloc_cross:
            self._cross_kv[idx][cache_tag] = {
                "k": torch.zeros(cross_shape, device=device, dtype=dtype),
                "v": torch.zeros(cross_shape, device=device, dtype=dtype),
                "is_init": False,
            }

    @staticmethod
    def _get_empty_self_tensor(entry: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build an empty `[B, 0, H, D]` tensor pair using cached metadata."""
        self_shape = tuple(entry.get("self_shape", (0, 0, 0, 0)))
        if len(self_shape) != 4:
            raise ValueError("invalid self_shape metadata for KV cache entry")
        batch_size, _, num_heads, head_dim = self_shape
        self_device = entry.get("self_device", torch.device("cpu"))
        self_dtype = entry.get("self_dtype", torch.float32)
        empty = torch.zeros((batch_size, 0, num_heads, head_dim), device=self_device, dtype=self_dtype)
        return empty, empty

    def _get_entry_token_count(self, entry: dict[str, Any]) -> int:
        """Return the number of cached self-attention tokens stored in one entry."""
        return int(entry.get("len", 0))

    def get_cached_self_token_count(self, cache_tag: str) -> int:
        """Return cached self-attention tokens for the first populated block under `cache_tag`."""
        for block_idx in range(len(self._self_kv)):
            cached_tokens = self.get_self_len(block_idx=block_idx, cache_tag=cache_tag)
            if cached_tokens > 0:
                return cached_tokens
        return 0

    def get_self_len(self, block_idx: int, cache_tag: str) -> int:
        """Return the cached self-attention token count for one block."""
        entry = self._self_kv[block_idx].get(cache_tag, None) if block_idx < len(self._self_kv) else None
        if not isinstance(entry, dict):
            return 0
        return self._get_entry_token_count(entry)

    def resolve_self_attention_start_idx(
        self,
        block_idx: int,
        cache_tag: str,
        requested_start_idx: int | None,
    ) -> int:
        """Choose the self-attention prefix boundary for the next read/write operation."""
        if requested_start_idx is None:
            return self.get_self_len(block_idx=block_idx, cache_tag=cache_tag)
        return int(requested_start_idx)

    def get_cross_kv(self, block_idx: int, cache_tag: str) -> tuple[torch.Tensor, torch.Tensor, bool] | None:
        """Return cached cross-attention KV and its initialization flag for one block."""
        if block_idx >= len(self._cross_kv):
            return None
        entry = self._cross_kv[block_idx].get(cache_tag, None)
        if not isinstance(entry, dict):
            return None
        return entry["k"], entry["v"], bool(entry.get("is_init", False))

    def set_cross_kv(self, block_idx: int, cache_tag: str, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write cross-attention KV into an initialized cache slot."""
        if block_idx >= len(self._cross_kv) or cache_tag not in self._cross_kv[block_idx]:
            raise KeyError(f"cross cache is not initialized for block_idx={block_idx}, cache_tag={cache_tag}")
        self._cross_kv[block_idx][cache_tag] = {
            "k": key,
            "v": value,
            "is_init": True,
        }

    @abstractmethod
    def _clear_self_entry(self, entry: dict[str, Any]) -> None:
        """Reset one self-cache entry without changing its allocated capacity."""
        raise NotImplementedError

    def clear(self, delete: bool = False) -> None:
        """Reset cached tensors, or drop storage entirely when `delete=True`."""
        if delete:
            self._self_kv.clear()
            self._cross_kv.clear()
            self._block_masks.clear()
            return

        def _safe_zero_tensor(entry: dict[str, Any], key: str) -> None:
            tensor = entry.get(key, None)
            if not isinstance(tensor, torch.Tensor):
                return
            entry[key] = torch.zeros_like(tensor)

        for kvb in self._self_kv:
            for sub in kvb.values():
                if isinstance(sub, dict):
                    self._clear_self_entry(sub)

        for kvb in self._cross_kv:
            for sub in kvb.values():
                if isinstance(sub, dict):
                    sub["is_init"] = False
                    _safe_zero_tensor(sub, "k")
                    _safe_zero_tensor(sub, "v")

    def register(self, module: torch.nn.Module, block_idx: int) -> None:
        """Associate an attention module with a block index."""
        self._module_to_block[id(module)] = block_idx

    def is_registered(self, module: torch.nn.Module) -> bool:
        """Return whether `module` has been registered with this engine."""
        return id(module) in self._module_to_block

    def resolve_block_idx(self, module: torch.nn.Module) -> int:
        """Map a registered attention module back to its owning block index."""
        key = id(module)
        if key not in self._module_to_block:
            raise KeyError(
                f"Module {type(module).__name__} at {hex(key)} is not registered. "
                f"Call engine.register(module, block_idx) during model init."
            )
        return self._module_to_block[key]

    def get_or_create_block_mask(
        self,
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        chunk_size: int,
        mask_kind: str = "blockwise_causal",
    ) -> Any:
        """Memoize and return a blockwise causal mask for the requested frame geometry."""
        key = (str(device), int(num_frames), int(frame_seqlen), int(chunk_size), str(mask_kind))
        if key not in self._block_masks:
            if mask_kind == "blockwise_causal":
                self._block_masks[key] = prepare_blockwise_causal_attn_mask(
                    device=device,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    chunk_size=chunk_size,
                )
            elif mask_kind == "teacher_forcing":
                self._block_masks[key] = prepare_teacher_forcing_attn_mask(
                    device=device,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    chunk_size=chunk_size,
                )
            else:
                raise ValueError(f"Unsupported block mask kind: {mask_kind!r}")
        return self._block_masks[key]

    @abstractmethod
    def ensure_capacity(
        self,
        block_count: int,
        cache_tag: str,
        self_shape: tuple[int, int, int, int],
        cross_shape: tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def has_self_cache(self, block_idx: int, cache_tag: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def append_self_kv(
        self,
        block_idx: int,
        cache_tag: str,
        start_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        detach: bool,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def anchor_last_self_kv_to_output(
        self,
        block_idx: int,
        cache_tag: str,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_self_prefix(
        self,
        block_idx: int,
        cache_tag: str,
        end_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def prepare_self_attention_kv(
        self,
        block_idx: int,
        cache_tag: str,
        requested_start_idx: int | None,
        key: torch.Tensor,
        value: torch.Tensor,
        store_kv: bool,
        allow_cache_grad: bool,
        block_mask: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        raise NotImplementedError
