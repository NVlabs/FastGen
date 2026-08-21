# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-format conversions for the Wan networks.

FastGen's Wan wraps a diffusers ``WanTransformer3DModel`` and expects keys under a
``transformer.`` prefix. Third-party checkpoints come in other layouts, and the
converters here normalise them. They are applied in ``Wan.load_state_dict``:

* ``convert_wan_official_state_dict`` — the original (non-diffusers) Wan release
  layout, e.g. ``self_attn.q`` -> ``attn1.to_q``.
* ``remap_anyflow_keys`` — the AnyFlow HF releases, which store the flow-map
  r-pathway inside ``condition_embedder`` as ``delta_embedder``.

Each converter is a no-op when its marker keys are absent, so they are safe to call
unconditionally and in sequence.
"""

from typing import Any, Dict, List, Mapping, Tuple

import torch

import fastgen.utils.logging_utils as logger


# Rename mapping for the original Wan checkpoint layout, as an ordered list of
# (old, new) substring pairs — order matters for the norm swap below.
WAN_OFFICIAL_RENAME_MAPPING: List[Tuple[str, str]] = [
    ("time_embedding.0", "condition_embedder.time_embedder.linear_1"),
    ("time_embedding.2", "condition_embedder.time_embedder.linear_2"),
    ("text_embedding.0", "condition_embedder.text_embedder.linear_1"),
    ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
    ("time_projection.1", "condition_embedder.time_proj"),
    ("head.modulation", "scale_shift_table"),
    ("head.head", "proj_out"),
    ("modulation", "scale_shift_table"),
    ("ffn.0", "ffn.net.0.proj"),
    ("ffn.2", "ffn.net.2"),
    # swap norm names: norm1, norm3, norm2 -> norm1, norm2, norm3
    ("norm2", "norm__placeholder"),
    ("norm3", "norm2"),
    ("norm__placeholder", "norm3"),
    # I2V extras
    ("img_emb.proj.0", "condition_embedder.image_embedder.norm1"),
    ("img_emb.proj.1", "condition_embedder.image_embedder.ff.net.0.proj"),
    ("img_emb.proj.3", "condition_embedder.image_embedder.ff.net.2"),
    ("img_emb.proj.4", "condition_embedder.image_embedder.norm2"),
    ("img_emb.emb_pos", "condition_embedder.image_embedder.pos_embed"),
    # attention parts
    ("self_attn.q", "attn1.to_q"),
    ("self_attn.k", "attn1.to_k"),
    ("self_attn.v", "attn1.to_v"),
    ("self_attn.o", "attn1.to_out.0"),
    ("self_attn.norm_q", "attn1.norm_q"),
    ("self_attn.norm_k", "attn1.norm_k"),
    ("cross_attn.q", "attn2.to_q"),
    ("cross_attn.k", "attn2.to_k"),
    ("cross_attn.v", "attn2.to_v"),
    ("cross_attn.o", "attn2.to_out.0"),
    ("cross_attn.norm_q", "attn2.norm_q"),
    ("cross_attn.norm_k", "attn2.norm_k"),
    ("attn2.to_k_img", "attn2.add_k_proj"),
    ("attn2.to_v_img", "attn2.add_v_proj"),
    ("attn2.norm_k_img", "attn2.norm_added_k"),
]

# Keys a checkpoint may nest the model state under.
_NESTED_STATE_KEYS = ["generator", "state_dict", "model", "module", "net"]

# Prefixes stripped before applying the rename mapping.
_STRIPPED_PREFIXES = ["model.", "module.", "transformer."]


def _unwrap_state_dict(state_dict: Mapping[str, Any]) -> Tuple[Dict[str, Any], str | None]:
    """Return the tensor state dict, unwrapping one level of nesting if present."""
    for key in _NESTED_STATE_KEYS:
        if isinstance(state_dict, dict) and key in state_dict and isinstance(state_dict[key], dict):
            return state_dict[key], key
    # fallback: assume the loaded object IS the state dict
    if isinstance(state_dict, dict) and all(isinstance(v, torch.Tensor) for v in state_dict.values()):
        return state_dict, None
    raise ValueError("Could not find a state_dict in checkpoint.")


def rename_wan_official_key(key: str) -> str:
    """Map one original-Wan parameter name onto the diffusers layout."""
    for prefix in _STRIPPED_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
    for old, new in WAN_OFFICIAL_RENAME_MAPPING:
        if old in key:
            key = key.replace(old, new)
    return key


def convert_wan_official_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert an original (non-diffusers) Wan checkpoint to FastGen's layout.

    Unwraps a nested state dict if needed, renames the parameters via
    ``WAN_OFFICIAL_RENAME_MAPPING``, and adds the ``transformer.`` prefix the
    model expects.
    """
    state, nested_key = _unwrap_state_dict(state_dict)
    logger.info(f"Loading original Wan checkpoint format from key: {nested_key}")
    return {f"transformer.{rename_wan_official_key(k)}": v for k, v in state.items()}


def remap_anyflow_keys(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remap an AnyFlow HF release state_dict to FastGen's Wan layout.

    AnyFlow's ``FAR_Wan_Transformer3DModel`` stores the r-pathway inside the main
    ``condition_embedder`` as ``delta_embedder``, and uses ONE shared ``time_proj``
    for both t and (t, r). FastGen exposes the r-pathway as a top-level
    ``r_embedder``; in gated mode it keeps only the ``time_embedder`` and reuses
    ``condition_embedder.time_proj`` (see ``_fuse_r_embedding`` in
    ``Wan/network.py``), so the layouts differ by a rename only.

    The function is a no-op when no ``condition_embedder.delta_embedder.*`` keys are
    present, so it's safe to call unconditionally. Keys with or without the
    ``transformer.`` module prefix are both handled.
    """
    delta_marker = "condition_embedder.delta_embedder."
    delta_keys = [k for k in state_dict if delta_marker in k]
    if not delta_keys:
        return state_dict

    new_sd = dict(state_dict)
    for k in delta_keys:
        # [transformer.]condition_embedder.delta_embedder.linear_1.weight
        #   -> [transformer.]r_embedder.time_embedder.linear_1.weight
        prefix, _, suffix = k.partition(delta_marker)
        target = f"{prefix}r_embedder.time_embedder.{suffix}"
        if target in new_sd:
            raise ValueError(
                f"remap_anyflow_keys: rewriting {k!r} would overwrite the existing {target!r}. "
                "This checkpoint carries both the AnyFlow and the FastGen r-pathway layouts; "
                "drop one of them before loading."
            )
        new_sd[target] = new_sd.pop(k)
    logger.info(f"remap_anyflow_keys: rewrote {len(delta_keys)} delta_embedder tensors into r_embedder.")
    return new_sd
