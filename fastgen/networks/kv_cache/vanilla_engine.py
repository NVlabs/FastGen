# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vanilla KV-cache engine: per-chunk self-attention K/V lists and dense cross-attention K/V buffers.

Each self-cache entry, one per block and `cache_tag`, holds:

- `k_chunks`, `v_chunks`: the cached K/V, one `[B, chunk_tokens, H, D]` tensor per stored chunk.
- `len`: the number of cached tokens across all chunks.
- `self_shape`, `self_device`, `self_dtype`: the `(B, seq_len, H, D)` layout, device, and dtype from the latest
  `ensure_capacity`. They shape the prefix of an empty cache, and a `self_shape` change re-creates the entry.

Lifecycle:

1. `ensure_capacity` creates each entry empty (`len=0`), so `len` always counts the cached tokens, and keeps an
   existing entry whose `self_shape` matches. A different `self_shape` re-creates the entry, but raises `ValueError`
   instead if the entry still holds tokens. Call `clear_caches()` before a new rollout, and keep `self_shape` fixed
   within one by setting `CausalContext.total_num_frames`; without it, the shape follows each chunk's length.
2. `append_self_kv` appends a chunk's K/V and advances `len`. A `start_idx` below `len` (activation-checkpoint
   recompute or chunk restaging) first truncates the stale suffix; a `start_idx` above `len` raises `ValueError`.
   `detach` controls whether gradients flow through the cached chunks, as in Self-Forcing training.
3. `get_self_prefix(end_idx=start_idx)` concatenates the cached chunks up to the boundary, and attention runs on
   `[cached_prefix | current_chunk]`. It raises `ValueError` if `end_idx` exceeds `len`, i.e. when earlier chunks
   were cleared or never stored.
4. `anchor_last_self_kv_to_output` adds a zero-valued term derived from the block output to the newest chunk. This
   keeps an autograd edge between them, without changing values, when activation checkpointing restages K/V.
5. `clear` empties every entry (`len=0`) but keeps its metadata; `clear(delete=True)` drops the storage.
"""

from __future__ import annotations

from typing import Any

import torch

from fastgen.networks.kv_cache.base import BaseKVCacheEngine


class VanillaKVCacheEngine(BaseKVCacheEngine):
    """Baseline engine that stores self-cache as raw chunk lists and cross-cache as dense buffers."""

    def ensure_capacity(
        self,
        block_count: int,
        cache_tag: str,
        self_shape: tuple[int, int, int, int],
        cross_shape: tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Allocate per-block raw self-cache metadata and fixed-size cross-cache buffers."""
        self._ensure_block_storage(block_count=block_count)

        for idx in range(block_count):
            self_entry = self._self_kv[idx].get(cache_tag, {})
            need_alloc_self = (
                not isinstance(self_entry, dict)
                or not isinstance(self_entry.get("k_chunks", None), list)
                or not isinstance(self_entry.get("v_chunks", None), list)
                or tuple(self_entry.get("self_shape", ())) != self_shape
            )

            if need_alloc_self:
                cached_len = self._get_entry_token_count(self_entry)
                if cached_len > 0:
                    raise ValueError(
                        f"self_shape changed from {self_entry.get('self_shape')} to {self_shape} while the self cache "
                        f"holds {cached_len} tokens (block_idx={idx}, cache_tag={cache_tag}); call clear_caches() "
                        "before a new rollout and keep self_shape fixed within one, e.g. by setting "
                        "CausalContext.total_num_frames"
                    )
                self._self_kv[idx][cache_tag] = {
                    "k_chunks": [],
                    "v_chunks": [],
                    "len": 0,
                    "self_shape": self_shape,
                    "self_device": device,
                    "self_dtype": dtype,
                }
            else:
                self._self_kv[idx][cache_tag]["self_shape"] = self_shape
                self._self_kv[idx][cache_tag]["self_device"] = device
                self._self_kv[idx][cache_tag]["self_dtype"] = dtype

            self._ensure_cross_cache_entry(
                idx=idx,
                cache_tag=cache_tag,
                cross_shape=cross_shape,
                device=device,
                dtype=dtype,
            )

    def has_self_cache(self, block_idx: int, cache_tag: str) -> bool:
        """Return whether a raw self-cache entry exists for this block and tag."""
        if block_idx >= len(self._self_kv):
            return False
        entry = self._self_kv[block_idx].get(cache_tag, None)
        return (
            isinstance(entry, dict)
            and isinstance(entry.get("k_chunks", None), list)
            and isinstance(entry.get("v_chunks", None), list)
            and "len" in entry
        )

    def append_self_kv(
        self,
        block_idx: int,
        cache_tag: str,
        start_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        detach: bool,
    ) -> int:
        """Append a new raw KV segment, truncating any stale suffix first."""
        if block_idx >= len(self._self_kv) or cache_tag not in self._self_kv[block_idx]:
            raise KeyError(f"self cache is not initialized for block_idx={block_idx}, cache_tag={cache_tag}")

        entry = self._self_kv[block_idx][cache_tag]
        current_len = int(entry.get("len", 0))
        start_idx = int(start_idx)
        if start_idx < current_len:
            # Rewinding can happen during activation-checkpoint recompute or chunk restaging.
            k_chunks = entry["k_chunks"]
            v_chunks = entry["v_chunks"]
            kept_k: list[torch.Tensor] = []
            kept_v: list[torch.Tensor] = []
            kept_len = 0
            remaining = start_idx
            for k_chunk, v_chunk in zip(k_chunks, v_chunks):
                chunk_len = int(k_chunk.shape[1])
                if remaining <= 0:
                    break
                if remaining >= chunk_len:
                    kept_k.append(k_chunk)
                    kept_v.append(v_chunk)
                    kept_len += chunk_len
                else:
                    kept_k.append(k_chunk[:, :remaining, :, :])
                    kept_v.append(v_chunk[:, :remaining, :, :])
                    kept_len += remaining
                    break
                remaining -= chunk_len
            entry["k_chunks"] = kept_k
            entry["v_chunks"] = kept_v
            current_len = kept_len
            entry["len"] = current_len

        if start_idx != current_len:
            raise ValueError(
                f"start_idx={start_idx} must match current cached len={current_len} "
                f"for dynamic self cache (block_idx={block_idx}, cache_tag={cache_tag})"
            )
        seqlen = int(key.shape[1])
        end_idx = current_len + seqlen
        entry["len"] = end_idx

        entry["k_chunks"].append(key.detach() if detach else key)
        entry["v_chunks"].append(value.detach() if detach else value)
        return end_idx

    def anchor_last_self_kv_to_output(
        self,
        block_idx: int,
        cache_tag: str,
        output: torch.Tensor,
    ) -> None:
        """Tie the newest self-KV chunk to a block output without changing values.

        The zero-valued anchor keeps an autograd edge from the block output back
        to the cached chunk when activation checkpointing restages KV tensors.
        """
        if block_idx >= len(self._self_kv) or cache_tag not in self._self_kv[block_idx]:
            raise KeyError(f"self cache is not initialized for block_idx={block_idx}, cache_tag={cache_tag}")

        entry = self._self_kv[block_idx][cache_tag]
        if len(entry.get("k_chunks", [])) == 0 or len(entry.get("v_chunks", [])) == 0:
            return

        # Zero before reducing: a finite FP16 output can have an overflowing sum,
        # in which case multiplying the sum by zero would poison the cache with NaNs.
        anchor = (output.reshape(output.shape[0], -1) * 0).sum(dim=1, keepdim=True)
        anchor = anchor.to(device=entry["k_chunks"][-1].device, dtype=entry["k_chunks"][-1].dtype)
        anchor = anchor.view(output.shape[0], 1, 1, 1)

        entry["k_chunks"][-1] = entry["k_chunks"][-1] + anchor
        entry["v_chunks"][-1] = entry["v_chunks"][-1] + anchor

    def get_self_prefix(
        self,
        block_idx: int,
        cache_tag: str,
        end_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize a raw self-cache prefix ending at `end_idx` tokens."""
        if block_idx >= len(self._self_kv) or cache_tag not in self._self_kv[block_idx]:
            raise KeyError(f"self cache is not initialized for block_idx={block_idx}, cache_tag={cache_tag}")

        entry = self._self_kv[block_idx][cache_tag]
        cached_len = self._get_entry_token_count(entry)
        if end_idx > cached_len:
            raise ValueError(
                f"requested a self-cache prefix of {end_idx} tokens, but only {cached_len} are cached "
                f"(block_idx={block_idx}, cache_tag={cache_tag}); earlier chunks were cleared or never stored"
            )
        k_chunks = entry.get("k_chunks", [])
        v_chunks = entry.get("v_chunks", [])
        if len(k_chunks) == 0:
            return self._get_empty_self_tensor(entry)

        remaining = max(0, int(end_idx))
        parts_k = []
        parts_v = []
        for k_chunk, v_chunk in zip(k_chunks, v_chunks):
            if remaining <= 0:
                break
            chunk_len = int(k_chunk.shape[1])
            if remaining >= chunk_len:
                parts_k.append(k_chunk)
                parts_v.append(v_chunk)
            else:
                parts_k.append(k_chunk[:, :remaining, :, :])
                parts_v.append(v_chunk[:, :remaining, :, :])
                break
            remaining -= chunk_len

        if len(parts_k) == 0:
            return k_chunks[0][:, :0, :, :], v_chunks[0][:, :0, :, :]
        if len(parts_k) == 1:
            return parts_k[0], parts_v[0]
        return torch.cat(parts_k, dim=1), torch.cat(parts_v, dim=1)

    def build_self_attention_kv(
        self,
        block_idx: int,
        cache_tag: str,
        start_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        store_kv: bool,
        allow_cache_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the KV pair used by attention from cached prefix plus the current chunk."""
        if store_kv:
            self.append_self_kv(
                block_idx=block_idx,
                cache_tag=cache_tag,
                start_idx=start_idx,
                key=key,
                value=value,
                detach=not allow_cache_grad,
            )

        if start_idx == 0:
            return key, value

        with torch.set_grad_enabled(allow_cache_grad):
            k_cached, v_cached = self.get_self_prefix(
                block_idx=block_idx,
                cache_tag=cache_tag,
                end_idx=start_idx,
            )
        return torch.cat([k_cached, key], dim=1), torch.cat([v_cached, value], dim=1)

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
        """Prepare self-attention KV when raw-cache reuse is active for this block."""
        if block_mask is not None or not self.has_self_cache(block_idx=block_idx, cache_tag=cache_tag):
            return None

        start_idx = self.resolve_self_attention_start_idx(
            block_idx=block_idx,
            cache_tag=cache_tag,
            requested_start_idx=requested_start_idx,
        )
        return self.build_self_attention_kv(
            block_idx=block_idx,
            cache_tag=cache_tag,
            start_idx=start_idx,
            key=key,
            value=value,
            store_kv=store_kv,
            allow_cache_grad=allow_cache_grad,
        )

    def _clear_self_entry(self, entry: dict[str, Any]) -> None:
        """Reset a raw self-cache entry while preserving its metadata and allocation."""
        entry["len"] = 0
        entry["k_chunks"] = []
        entry["v_chunks"] = []
