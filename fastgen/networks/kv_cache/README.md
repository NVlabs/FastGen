# KV Cache Module

This folder contains the modular KV-cache backend used by causal video models.

Current scope:

- vanilla KV cache only
- multi-branch routing supported at the `CausalContext` / model layer

This README describes the structure. Entry payloads, lifecycles, and error cases are documented in the module docstrings of `base.py` and `vanilla_engine.py`.

## Files

- `base.py`: shared cache storage, registration, cross-attention entries, and block-mask helpers
- `vanilla_engine.py`: baseline raw-chunk self-cache + dense cross-cache engine (`VanillaKVCacheEngine`, exported as `KVCacheEngine`)
- `__init__.py`: package exports for the active KV-cache backend

## Layering

The design is split into three layers:

1. Model layer
   The model owns named KV-cache engine branches on `transformer.kv_cache_engines`.

2. Context layer
   `CausalContext` carries the branch map for the current rollout in `cache_engines`.

3. Engine layer
   Each engine stores per-block cache entries keyed by `cache_tag`.

## Model-Level Structure Example for Wan T2V / WanI2V / VACE-Wan

Causal Wan T2V / WanI2V:

```python
self.transformer.kv_cache_engines = {
    "main": VanillaKVCacheEngine(),
}
```

Causal VACE:

```python
self.transformer.kv_cache_engines = {
    "main": VanillaKVCacheEngine(),
    "vace": VanillaKVCacheEngine(),
}
```

`"main"` owns `transformer.blocks[*]` and `"vace"` owns `transformer.vace_blocks[*]`. Each branch registers only the attention modules it owns.

## Context-Level Structure

`CausalContext` (`fastgen/networks/network_causal.py`) stores rollout metadata plus references to the named engine objects owned by the causal model; its docstring lists the fields. Two keys and a block index select one cache entry:

```text
branch key  -> choose engine: "main", "vace"
cache_tag   -> choose the pos/neg namespace inside that engine
block_idx   -> choose the block-local entry inside that engine/tag
```

## Engine-Level Structure

Each engine is single-branch internally. Shared storage in `BaseKVCacheEngine`:

| Attribute | Indexed by | Holds |
|---|---|---|
| `_self_kv` | `[block_idx][cache_tag]` | one self-attention entry |
| `_cross_kv` | `[block_idx][cache_tag]` | one cross-attention entry |
| `_module_to_block` | `id(attn_module)` | block index of a registered attention module |
| `_block_masks` | `(device, num_frames, frame_seqlen, chunk_size, mask_kind)` | memoized FlexAttention block mask |

All `block_idx` values are **local to the engine branch**, not global across the transformer:

| Engine branch | `block_idx = i` refers to |
|---|---|
| `"main"` | `transformer.blocks[i]` |
| `"vace"` | `transformer.vace_blocks[i]` |

### Self-attention entries

The vanilla engine keeps each block's K/V as a list with one tensor per stored chunk, plus the cached token count `len`:

1. `ensure_capacity()` creates each entry empty.
2. `append_self_kv()` appends each chunk's K/V.
3. Attention runs on `[cached_prefix | current_chunk]`, reading the prefix with `get_self_prefix()`.
4. `clear()`, called through `clear_caches()` at the start of each rollout, empties every entry.

### Cross-attention entries

Each entry holds the K/V of the full cross-attention context (the text tokens, preceded by the image tokens for Wan2.1 I2V) and is written once, then read many times per rollout:

1. `ensure_capacity()` allocates zero K/V with `is_init=False`.
2. The first forward with `store_kv=True` fills the entry.
3. Later no-grad forwards read it and skip the K/V projections; forwards with gradients enabled recompute K/V.
4. `clear()` zeros every entry; clear before changing a tag's conditioning or starting a new rollout.

## Concrete Hierarchy

```text
model.transformer.kv_cache_engines
├── "main" -> engine_main
│   ├── _module_to_block[id(attn)] = block_idx
│   ├── _self_kv[block_idx][cache_tag]
│   └── _cross_kv[block_idx][cache_tag]
└── "vace" -> engine_vace
    ├── _module_to_block[id(attn)] = block_idx
    ├── _self_kv[block_idx][cache_tag]
    └── _cross_kv[block_idx][cache_tag]
```

## Routing Rule

Attention processors do not hardcode a branch. They resolve the active engine by asking which registered engine owns the current attention module (`CausalContext.resolve_registered_engine`). That lets Wan T2V and WanI2V stay single-branch and VACE use separate `"main"` and `"vace"` caches without changing the engine internals.
