# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base network interface and runtime context for causal generation."""

from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import itertools
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch

from fastgen.networks.kv_cache import (
    BaseKVCacheEngine,
    CAUSAL_FLEX_ATTENTION_AVAILABLE,
)
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import causal_chunk_frames

__all__ = ["CausalContext", "CausalFastGenNetwork", "causal_chunk_ranges", "check_chunk_boundary"]


def causal_chunk_ranges(seq_len: int, chunk_size: int) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` frame ranges of the ``causal_chunk_frames`` chunks, in order."""
    bounds = list(itertools.accumulate(causal_chunk_frames(seq_len, chunk_size), initial=0))
    return list(zip(bounds[:-1], bounds[1:]))


def check_chunk_boundary(frame: int, chunk_ranges: Sequence[tuple[int, int]], name: str) -> None:
    """Raise unless ``frame`` is where one of ``chunk_ranges`` starts, i.e. a chunk boundary other than the end."""
    starts = [start for start, _ in chunk_ranges]
    if frame not in starts:
        raise ValueError(
            f"{name} must fall on a chunk boundary inside the {chunk_ranges[-1][1]}-frame sequence, one of "
            f"{starts}; got {frame}"
        )


@dataclass
class CausalContext:
    """Bundles all KV cache state for a single causal forward pass or rollout.

    Attributes:
        cache_tag: Cache namespace ("pos" for positive, "neg" for negative/unconditional).
        store_kv: Whether the current forward pass should write KV pairs to cache.
        cur_start_frame: Starting frame index for this chunk (used for RoPE offset).
        frame_offset: Global frame offset for multi-segment autoregressive extrapolation.
            Control-conditioned causal nets (e.g. VaceWan) add it to ``cur_start_frame``
            to index the control tensor at the segment's global window.
        is_ar: If True, use cached KV (AR mode). If False, use block mask (teacher-forcing).
        total_num_frames: Total frames in the full video sequence. Required for
            teacher-forcing block masks (``is_ar=False``). AR rollouts may omit it
            only if all chunks have the same length: without it, the KV-cache shape
            follows the current chunk, and a shape change mid-rollout raises.
        frame_seqlen: Tokens per frame. Set automatically by classify_forward_prepare().
        chunk_size: Number of frames per chunk for causal processing.
        cache_engines: Named KV cache engines, one per block branch (``"main"``; causal VACE adds ``"vace"``).
        grad_through_kv_cache: If ``True``, keep graph through cached self-KV reads/writes.
        mask_kind: Selects which non-AR mask to build. ``"blockwise_causal"`` is
            the existing noisy-only path, while ``"teacher_forcing"`` uses the
            dual-stream clean/noisy teacher-forcing mask.
    """

    cache_tag: str = "pos"
    store_kv: bool = False
    cur_start_frame: int = 0
    frame_offset: int = 0
    is_ar: bool = False
    total_num_frames: Optional[int] = None
    frame_seqlen: int = 0
    chunk_size: int = 3
    cache_engines: dict[str, BaseKVCacheEngine] = field(default_factory=dict, repr=False)
    grad_through_kv_cache: bool = False
    mask_kind: str = "blockwise_causal"
    _block_masks: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def cache_start_idx(self) -> int:
        """Token-level cache offset derived from frame position and tokens-per-frame."""
        return self.cur_start_frame * self.frame_seqlen

    def get_engine(self, key: str = "main") -> Optional[BaseKVCacheEngine]:
        """Return the engine for `key`."""
        return self.cache_engines.get(key)

    def set_engine(self, key: str, engine: BaseKVCacheEngine) -> None:
        """Set the engine for `key`."""
        self.cache_engines[key] = engine

    def attach_missing_engines(self, engines: Mapping[str, BaseKVCacheEngine]) -> None:
        """Attach model-owned engines that are not already present on this context.

        Callers often create ``CausalContext`` with only rollout metadata. The
        model owns the canonical engines because attention modules are registered
        there during construction, so attach them here to let blocks resolve
        their engine without requiring every caller to pass cache engines.
        """
        for key, engine in engines.items():
            self.cache_engines.setdefault(key, engine)

    def resolve_registered_engine(
        self, module: torch.nn.Module
    ) -> tuple[Optional[str], Optional[BaseKVCacheEngine], Optional[int]]:
        """Resolve which registered engine owns `module`.

        Returns `(engine_key, engine, block_idx)` when a matching branch is found,
        otherwise `(None, None, None)`.
        """
        for engine_key, engine in self.cache_engines.items():
            if engine.is_registered(module):
                return engine_key, engine, engine.resolve_block_idx(module)
        return None, None, None

    def snapshot(self) -> "CausalContext":
        """
        Create a shallow copy (snapshot) of the CausalContext.
        This is useful during gradient checkpointing so that the backward pass
        sees the exactly frozen metadata (`cur_start_frame`, `frame_seqlen`, etc.)
        for a specific chunk, while retaining references to the shared caches.
        """

        return replace(self)

    @contextmanager
    def set_store_kv(self, enabled: bool = True) -> Iterator[None]:
        """Temporarily set whether this context writes KV pairs to cache."""
        previous = self.store_kv
        self.store_kv = enabled
        try:
            yield
        finally:
            self.store_kv = previous

    def active_block_mask(self, engine_key: str = "main") -> Any:
        """Block mask for self-attention, suppressed in AR mode."""
        if self.is_ar:
            return None
        return self.get_block_mask(engine_key)

    def get_block_mask(self, engine_key: str = "main") -> Any:
        """Return the cached block mask for `engine_key`."""
        return self._block_masks.get(engine_key)

    def resolve_start_frame(self, frame_seqlen: int, engine_key: str = "main") -> int:
        """Resolve actual start frame, auto-detecting from cache when cur_start_frame is 0.

        Sets ``frame_seqlen`` on this context and, for an AR pass at ``cur_start_frame`` 0,
        derives the true start position from the number of tokens already cached. A full-sequence
        pass starts at ``cur_start_frame`` even if a rollout left tokens in the cache.
        """
        self.frame_seqlen = frame_seqlen
        engine = self.get_engine(engine_key)
        if self.is_ar and self.cur_start_frame == 0 and engine is not None:
            cached_tokens = int(engine.get_cached_self_token_count(cache_tag=self.cache_tag))
            if cached_tokens > 0:
                self.cur_start_frame = cached_tokens // frame_seqlen
        return self.cur_start_frame

    def prepare_block_mask(self, device: torch.device, num_frames: int, engine_key: str = "main") -> None:
        """Build (or retrieve cached) block mask for teacher-forcing mode.

        The result is stored in ``_block_masks[engine_key]`` and exposed via
        ``active_block_mask``, which additionally suppresses the mask
        in AR mode.
        """
        if self.is_ar:
            self._block_masks[engine_key] = None
            return
        if not CAUSAL_FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError(
                "FlexAttention is required for teacher-forcing block masks. "
                "Install a PyTorch build with torch.nn.attention.flex_attention or set CausalContext.is_ar=True "
                "for KV-cache autoregressive passes."
            )
        if self.total_num_frames is None:
            raise ValueError("total_num_frames must be set when preparing teacher-forcing block masks")
        if num_frames != self.total_num_frames:
            raise ValueError(
                "teacher-forcing block masks require a full-video forward pass "
                f"(num_frames={num_frames}, total_num_frames={self.total_num_frames}); "
                "use is_ar=True for chunked KV-cache autoregressive passes"
            )
        engine = self.get_engine(engine_key)
        if engine is None:
            raise KeyError(f"No KV cache engine registered for key={engine_key!r}")
        self._block_masks[engine_key] = engine.get_or_create_block_mask(
            device=device,
            num_frames=num_frames,
            frame_seqlen=self.frame_seqlen,
            chunk_size=self.chunk_size,
            mask_kind=self.mask_kind,
        )

    def ensure_kv_cache_capacity(
        self,
        block_count: int,
        self_shape: tuple[int, int, int, int],
        cross_shape: tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
        engine_key: str = "main",
    ) -> None:
        """Allocate KV cache buffers using caller-provided cache layout."""
        engine = self.get_engine(engine_key)
        if engine is None:
            raise KeyError(f"No KV cache engine registered for key={engine_key!r}")
        engine.ensure_capacity(
            block_count=block_count,
            cache_tag=self.cache_tag,
            self_shape=self_shape,
            cross_shape=cross_shape,
            device=device,
            dtype=dtype,
        )


class CausalFastGenNetwork(FastGenNetwork):
    """Base class for chunked causal video networks.

    Frames are processed ``chunk_size`` at a time: an AR rollout attends to the KV cache written by the earlier
    chunks, teacher forcing to a block-causal mask over the full sequence (see ``CausalContext``). The cache is
    sized for ``total_num_frames`` frames, and ``clear_caches`` empties it between rollouts.

    Args:
        chunk_size (int): Frames per chunk. Defaults to 3.
        total_num_frames (int): Frames of the full sequence the KV cache is sized for (AR sampling and
            Self-Forcing pass it as ``CausalContext.total_num_frames``). Defaults to 21.
        **kwargs: Keyword arguments for the next base class: the wrapped network
            (e.g. ``Wan``), else FastGenNetwork's noise-schedule kwargs.
    """

    def __init__(
        self,
        net_pred_type: str = "x0",
        schedule_type: str = "edm",
        chunk_size: int = 3,
        total_num_frames: int = 21,
        **kwargs,
    ):
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **kwargs)
        self.chunk_size = chunk_size
        self.total_num_frames = total_num_frames

    def _convert_output(self, prediction, x_t, t, condition, fwd_pred_type, **context):
        """Convert with one timestep per frame: a ``[B, T]`` time broadcasts over ``[B, C, T, H, W]``."""
        t = t[:, None, :, None, None] if t.ndim == 2 else t
        return super()._convert_output(prediction, x_t, t, condition, fwd_pred_type, **context)

    @abstractmethod
    def clear_caches(self):
        """Empty the KV caches so the next rollout starts from a clean cache."""
