# SPDX-FileCopyrightText: Copyright (c) 2025 The Wan Team and The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from contextlib import nullcontext
from typing import Any, Callable, Optional, List, Set, Dict, Union, Tuple, Mapping, Sequence
import types
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from diffusers import UniPCMultistepScheduler

from diffusers.models import WanTransformer3DModel, AutoencoderKLWan
from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

from transformers import T5TokenizerFast, UMT5EncoderModel

from fastgen.networks.lora_utils import LoRAMixin, scaled_lora_layers
from fastgen.networks.network import FastGenNetwork, FastGenModule, pixel_shape_from_factors
from fastgen.networks.Wan.utils import (
    WAN_OFFICIAL_KEY_MARKERS,
    _unwrap_state_dict,
    convert_wan_official_state_dict,
    remap_anyflow_keys,
)

from fastgen.utils.basic_utils import prompt_clean
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.tensor import promote_dtype
from fastgen.utils.module import attention_forward_without_cudnn, module_state_names
from fastgen.utils.env import hf_load
from fastgen.utils.tree_utils import OutputDtype, assert_same
import fastgen.utils.logging_utils as logger


def _original_fp32_region(enabled: bool, device_type: str):
    """Original Wan's ``autocast(dtype=torch.float32)`` region, where eligible operators compute in at least FP32:
    autocast upcasts reduced-precision operands and leaves FP64 ones alone. PyTorch supports FP32 autocast only
    on CUDA; elsewhere the region disables autocast, as PyTorch would after a warning, so it computes in the
    parameters' dtype."""
    if not enabled:
        return nullcontext()
    if device_type == "cuda":
        return torch.autocast(device_type, dtype=torch.float32)
    return torch.autocast(device_type, enabled=False)


def wan_rms_norm_forward(self: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """Instance-bound ``forward`` of an attention Q/K norm that rounds like original Wan's ``WanRMSNorm``.

    ``WanRMSNorm`` normalizes in FP32 and rounds to the input dtype before scaling by the weight, so a BF16 weight
    rounds the product a second time and an FP32 weight keeps it in FP32 for RoPE. ``torch.nn.RMSNorm`` rounds
    once after the weight, which is more precise for a BF16 weight; this follows the reference for parity. FP64
    inputs normalize in FP64.
    """
    dims = tuple(range(-len(self.normalized_shape), 0))
    eps = torch.finfo(x.dtype).eps if self.eps is None else self.eps
    wide = x.to(promote_dtype(x, torch.float32))
    normed = (wide * torch.rsqrt(wide.pow(2).mean(dim=dims, keepdim=True) + eps)).type_as(x)
    return normed if self.weight is None else normed * self.weight


def flatten_timestep(timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Flatten timestep and return seq_len if timestep has a shape of [batch_size, seq_len]
    """
    ts_seq_len = None
    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()  # batch_size * seq_len

    return timestep, ts_seq_len


def unflatten_timestep_proj(timestep_proj: torch.Tensor, ts_seq_len: Optional[int] = None) -> torch.Tensor:
    """
    Unflatten timestep, taking expand_timesteps (seq_len) into account
    """
    if ts_seq_len is not None:
        # batch_size, seq_len, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        # batch_size, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    return timestep_proj


def project_output(self, hidden_states: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Apply ``proj_out`` to the modulated output norm.

    With ``keep_original_prec``, the projection runs under FP32 autocast like original Wan's head, on an input in the
    wider of FP32 and its weight dtype. Otherwise it computes in *dtype*, the text projection's, like Diffusers.
    """
    if not self.keep_original_prec:
        return self.proj_out(hidden_states.to(dtype))
    head_dtype = promote_dtype(self.proj_out.weight, torch.float32)
    with _original_fp32_region(True, hidden_states.device.type):
        output = self.proj_out(hidden_states.to(head_dtype))
    assert_same(dtype=head_dtype, output=output)
    return output


def project_logvar(self, temb: torch.Tensor, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Apply ``logvar_linear`` to one time embedding per sample: ``temb`` averaged over its noised tokens or frames.

    ``_compute_timestep_inputs`` applies the image-to-video first-frame mask by zeroing those timesteps, and
    teacher forcing's clean context is at timestep zero too, so the zeros mark the clean inputs to leave out. A
    sample with every timestep at zero has one embedding throughout and averages all of it.

    The average is taken in ``temb``'s dtype, the time MLP's, and the head computes in *dtype*, the text
    projection's, so callers pass ``encoder_hidden_states.dtype``.
    """
    if temb.ndim == 3:
        noised = (timestep != 0).reshape(temb.shape[0], -1).expand(temb.shape[:2])
        noised = (noised | ~noised.any(dim=1, keepdim=True)).unsqueeze(-1)
        temb = (temb * noised).sum(dim=1) / noised.sum(dim=1)
    return self.logvar_linear(temb.to(dtype))


def normalize(x, dim=-1, eps=1e-4):
    """Scale each modulation vector to unit norm, per sample, frame or token alike."""
    dtype = promote_dtype(x, torch.float32)
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=dtype)
    norm = torch.add(eps, norm, alpha=1.0)
    return x / norm.to(x.dtype)


def sinusoidal_embedding_1d_wan(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal embedding matching the original Wan2.1/2.2 implementation.

    Computes in float64 and outputs [cos, sin] ordering. With the Wan-specific
    diffusers config (flip_sin_to_cos=True, downscale_freq_shift=0), the formula
    and ordering are identical to diffusers' Timesteps — the only difference is
    float64 vs float32 intermediate precision.

    Args:
        dim: Total embedding dimension (split into half for cos and half for sin).
        position: Input positions/timesteps tensor.

    Returns:
        Sinusoidal embedding of shape [len(position), dim].
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half, device=position.device).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.float()


def wan_timestep_embedding_forward(
    self: torch.nn.Module,
    sample: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a Diffusers timestep MLP using the live first projection dtype, widened to the autocast dtype.

    Apart from the input cast, this mirrors the pinned Diffusers
    ``TimestepEmbedding.forward`` implementation. Under original Wan's FP32 autocast
    region the projections compute in at least FP32, so the FP32 sinusoid enters
    them unrounded.

    This forward executes after the enclosing FSDP/offload pre-forward hook, so
    ``linear_1.weight.dtype`` describes the materialized projection rather than the
    sharded master storage. If a child projection is ever wrapped separately, this
    cast must move into that projection's forward instead.
    """
    if condition is not None:
        sample = sample + self.cond_proj(condition)

    projection_dtype = self.linear_1.weight.dtype
    # Quantized projections consume floating activations and own their input policy.
    if projection_dtype != torch.int8:
        if torch.is_autocast_enabled(sample.device.type):
            projection_dtype = torch.promote_types(projection_dtype, torch.get_autocast_dtype(sample.device.type))
        sample = sample.to(dtype=projection_dtype)

    sample = self.linear_1(sample)
    if self.act is not None:
        sample = self.act(sample)
    sample = self.linear_2(sample)
    if self.post_act is not None:
        sample = self.post_act(sample)
    return sample


def wan_condition_embedding_forward(
    self: torch.nn.Module,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    timestep_seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run a Diffusers Wan condition embedder with original Wan's time-embedding precision.

    Apart from the time path, this mirrors the pinned Diffusers ``WanTimeTextImageEmbedding.forward``
    implementation. Diffusers casts ``temb`` to the text's dtype before ``time_proj``; original Wan
    computes the time embedding and its projection in FP32 under autocast and asserts that both are FP32.
    ``temb`` therefore keeps the time embedder's dtype. With ``keep_original_prec``, both run under
    FP32 autocast and are asserted to be in the wider of FP32 and the time embedder's dtype, so FP64
    models keep FP64.

    The sinusoid is cast in ``wan_timestep_embedding_forward``, after the FSDP/offload
    pre-forward hook has materialized the time embedder's projection.
    """
    timestep = self.timesteps_proj(timestep)
    if timestep_seq_len is not None:
        timestep = timestep.unflatten(0, (-1, timestep_seq_len))

    with _original_fp32_region(self.keep_original_prec, timestep.device.type):
        temb = self.time_embedder(timestep)
        timestep_proj = self.time_proj(self.act_fn(temb))
    if self.keep_original_prec:
        time_dtype = promote_dtype(self.time_embedder.linear_1.weight, torch.float32)
        assert_same(dtype=time_dtype, temb=temb, timestep_proj=timestep_proj)

    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
    if encoder_hidden_states_image is not None:
        encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


def compute_temb_modulation(scale_shift_table, temb, norm_temb):
    """Chunk temb into 6 shift/scale/gate modulation parameters."""
    if temb.ndim == 4:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)
    else:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (scale_shift_table + temb.float()).chunk(
            6, dim=1
        )

    if norm_temb:
        shift_msa = normalize(shift_msa)
        scale_msa = normalize(scale_msa)
        c_shift_msa = normalize(c_shift_msa)
        c_scale_msa = normalize(c_scale_msa)

    return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa


def wan_block_forward_impl(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    shift_msa,
    scale_msa,
    gate_msa,
    c_shift_msa,
    c_scale_msa,
    c_gate_msa,
    post_self_attn_fn: Optional[Callable] = None,
):
    """Self-attn → cross-attn → FFN shared by Wan and WanCamI2V."""
    core_dtype = encoder_hidden_states.dtype

    # 1. Self-attention
    norm_hidden_states = self.norm1(hidden_states).float()
    norm_hidden_states = (norm_hidden_states * (1 + scale_msa) + shift_msa).to(core_dtype)
    attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
    hidden_states = hidden_states.float() + attn_output * gate_msa

    if post_self_attn_fn is not None:
        hidden_states = post_self_attn_fn(hidden_states)

    # 2. Cross-attention
    norm_hidden_states = self.norm2(hidden_states.float()).to(core_dtype)
    attn_output = self.attn2(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states.to(core_dtype),
    )
    hidden_states = hidden_states + attn_output

    # 3. Feed-forward
    norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).to(core_dtype)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = hidden_states.float() + ff_output.float() * c_gate_msa

    # Blocks used outside a Wan network (tests, standalone modules) keep the reference fp32 carrier.
    if not getattr(self, "fp32_residual", True):
        hidden_states = hidden_states.to(core_dtype)
    return hidden_states


def block_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    norm_temb: bool,
):
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = compute_temb_modulation(
        self.scale_shift_table, temb, norm_temb
    )
    return wan_block_forward_impl(
        self,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        shift_msa,
        scale_msa,
        gate_msa,
        c_shift_msa,
        c_scale_msa,
        c_gate_msa,
    )


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
    skip_layers: Optional[List[int]] = None,
    **kwargs,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    # get the sizes
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    with scaled_lora_layers(self, lora_scale):
        # prepare the input for the block forward passes
        (
            hidden_states,
            timestep_proj,
            r_timestep_proj,
            encoder_hidden_states,
            temb,
            rotary_emb,
        ) = self.classify_forward_prepare(
            hidden_states,
            timestep,
            encoder_hidden_states,
            r_timestep,
            encoder_hidden_states_image,
            attention_kwargs,
        )

        hidden_states, features = self.classify_forward_block_forward(
            hidden_states,
            timestep_proj,
            encoder_hidden_states,
            rotary_emb,
            r_timestep_proj,
            skip_layers,
            feature_indices,
            return_features_early,
            attention_kwargs,
        )

        # If we have all the features, we can exit early
        if return_features_early:
            assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
            return features

        # 5. Output norm, projection & unpatchify
        if temb.dim() == 3:
            # Per-frame (or per-token) modulation
            # Compute per-frame (or per-token) shift/scale from model-level table and per-frame (or per-token) temb
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)

            # Two possible cases:
            #  - Wan 2.1: Per-frame modulation (causal architecture)
            #  - Wan 2.2 IT2V 5B: Per-token modulation (default)
            if shift.shape[1] == hidden_states.shape[1]:
                # Wan 2.2 IT2V 5B model
                # Apply shift/scale per token on normalized tokens
                hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
            else:
                # Apply shift/scale per frame on normalized tokens
                frame_seqlen = post_patch_height * post_patch_width
                hs_norm_out = self.norm_out(hidden_states.float()).unflatten(1, (post_patch_num_frames, frame_seqlen))
                hidden_states = (
                    (hs_norm_out * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)).flatten(1, 2).type_as(hidden_states)
                )
        else:
            # Global modulation
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)
            hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = project_output(self, hidden_states, encoder_hidden_states.dtype)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            hidden_states.shape[-1] // (p_t * p_h * p_w),
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        out = output if feature_indices is None else [output, features]

        if return_logvar:
            assert hasattr(
                self, "logvar_linear"
            ), "logvar_linear layer is required when return_logvar=True. Set enable_logvar_linear=True in model config."
            # One value per sample: average the time embeddings of the noised tokens or frames.
            logvar = project_logvar(self, temb, timestep, encoder_hidden_states.dtype)
            return out, logvar
        return out


def _fuse_r_embedding(
    self,
    temb: torch.Tensor,
    timestep_proj: torch.Tensor,
    remb: torch.Tensor,
    rs_seq_len: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Combine the t- and r-time embeddings into the final timestep projection.

    * ``additive``: project each embedding separately and sum — ``temb + remb``,
      ``timestep_proj + r_embedder.time_proj(...)``.
    * ``gated``: interpolate first, ``(1-g)·temb + g·remb``, then project the
      blend through the condition_embedder's own ``time_proj`` — shared with the
      t-only path, so ``Wan.__init__`` drops the unused ``r_embedder.time_proj``
      rather than let it drift during training.

    Returns ``(temb, timestep_proj, r_timestep_proj)``: ``temb`` [B, D] modulates
    the output head (``scale_shift_table + temb``), ``timestep_proj`` [B, 6, D] is
    the AdaLN modulation every block consumes, and ``r_timestep_proj`` is what the
    blocks switch to at ``encoder_depth`` — ``None`` when there is no switch.
    """
    fusion = getattr(self.r_embedder, "fusion_mode", "additive")

    if fusion == "gated":
        gate = self.r_embedder.gate_value
        rt_emb = (1 - gate) * temb + gate * remb
        rt_ts_proj = self.condition_embedder.time_proj(self.condition_embedder.act_fn(rt_emb))
        rt_ts_proj = unflatten_timestep_proj(rt_ts_proj, rs_seq_len)
        if self.encoder_depth is None:
            return rt_emb, rt_ts_proj, None
        return rt_emb, timestep_proj, rt_ts_proj

    # additive (MeanFlow): project r through the r-embedder's own time_proj and sum
    r_ts_proj = self.r_embedder.time_proj(self.r_embedder.act_fn(remb))
    r_ts_proj = unflatten_timestep_proj(r_ts_proj, rs_seq_len)
    if self.encoder_depth is None:
        return temb + remb, timestep_proj + r_ts_proj, None
    return remb, timestep_proj, r_ts_proj


def apply_r_embedding(
    self,
    timestep: torch.Tensor,
    r_timestep: Optional[torch.Tensor],
    temb: torch.Tensor,
    timestep_proj: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Embed ``r_timestep`` and fuse it into ``temb`` / ``timestep_proj``.

    ``timestep`` must already be the flattened tensor ``flatten_timestep`` returns. ``r_timestep``
    gets the same treatment as ``t``: the forwards expand both alike (``_compute_timestep_inputs``),
    so it is flattened here before ``time_cond_type="diff"`` subtracts it, and its embedding
    unflattens like ``timestep_proj``. Like ``temb``, the embedding keeps its time embedder's dtype, and
    with ``keep_original_prec`` the r path runs under FP32 autocast and is asserted at least FP32 like the t path.
    Fusion math is ``_fuse_r_embedding`` (additive or gated). Shared by Wan, VACE, and causal prepares.
    """
    if self.r_embedder is not None and r_timestep is not None:
        r_timestep, rs_seq_len = flatten_timestep(r_timestep)
        if self.time_cond_type == "abs":
            pass
        elif self.time_cond_type == "diff":
            r_timestep = timestep - r_timestep
        else:
            raise ValueError(f"Invalid time condition: {self.time_cond_type}")

        with _original_fp32_region(self.r_embedder.keep_original_prec, r_timestep.device.type):
            r_timestep = self.r_embedder.timesteps_proj(r_timestep)
            remb = self.r_embedder.time_embedder(r_timestep)
            if rs_seq_len is not None:
                # [B, S, D] like the ``temb`` the condition embedder returns for expanded timesteps.
                remb = remb.unflatten(0, (-1, rs_seq_len))
            embeddings = _fuse_r_embedding(self, temb, timestep_proj, remb, rs_seq_len)
        if self.r_embedder.keep_original_prec:
            time_dtype = promote_dtype(self.r_embedder.time_embedder.linear_1.weight, torch.float32)
            assert_same(dtype=time_dtype, embeddings=embeddings)
        return embeddings
    if r_timestep is not None:
        raise ValueError("r_timestep provided but no r_embedder is present")
    return temb, timestep_proj, None


def classify_forward_prepare(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    prepare the input for the block forward passes inside the classify_forward function.
    will be used as the prepare_op function in the classify_forward function.

    args:
        self: WanTransformer3DModel
        hidden_states: [B, C, F, H, W]
        timestep: [B], or [B, N_tokens] for Wan2.2 TI2V 5B
        encoder_hidden_states: [B, L_c, D_c]
        r_timestep: same shape as timestep
        encoder_hidden_states_image: [B, L_img, D_img]
        attention_kwargs: Optional[Dict[str, Any]] = None

    Returns:
        hidden_states: [B, N_tokens, D_model], where N_tokens=(F/p_t)*(H/p_h)*(W/p_w)
        timestep_proj: [B, 6, D_model], or [B, N_tokens, 6, D_model] for Wan2.2 TI2V 5B
        r_timestep_proj: same shape as timestep_proj, or None when no block switches to it
        encoder_hidden_states: [B, L, D_model], the image tokens (if any) followed by the text tokens
        temb: [B, D_model], or [B, N_tokens, D_model] for Wan2.2 TI2V 5B
        rotary_emb: (cos, sin), each [1, N_tokens, 1, D_head]
    """
    rotary_emb = self.rope(hidden_states)
    assert_same(
        device=True,
        hidden_states=hidden_states,
        freqs_cos=rotary_emb[0],
        freqs_sin=rotary_emb[1],
    )

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # Contiguous before the block loop: mirrors diffusers #14236 (~8% faster block loop).
    hidden_states = hidden_states.contiguous()

    # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
    timestep, ts_seq_len = flatten_timestep(timestep)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )
    timestep_proj = unflatten_timestep_proj(timestep_proj, ts_seq_len)

    temb, timestep_proj, r_timestep_proj = apply_r_embedding(self, timestep, r_timestep, temb, timestep_proj)

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    return (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        temb,
        rotary_emb,
    )


def classify_forward_block_forward(
    self,
    hidden_states: torch.Tensor,
    timestep_proj: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    r_timestep_proj: Optional[torch.Tensor] = None,
    skip_layers: Optional[List[int]] = None,
    feature_indices: Optional[Set[int]] = None,
    return_features_early: Optional[bool] = False,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    block forward pass inside the classify_forward function.
    will be used as the block_forward_op function in the classify_forward function.

    args:
        self: WanTransformer3DModel
        hidden_states: [B, N_tokens, D_model]
        timestep_proj: [B, 6, D_model], or [B, N_tokens, 6, D_model] for Wan2.2 TI2V 5B
        encoder_hidden_states: [B, L, D_model]
        rotary_emb: (cos, sin), each [1, N_tokens, 1, D_head]
        r_timestep_proj: same shape as timestep_proj, or None
        skip_layers: Optional[List[int]]
        feature_indices: Optional[Set[int]]
        attention_kwargs: Optional[Dict[str, Any]]

    Returns:
        hidden_states: [B, N_tokens, D_model]
        features: List[[B, N_tokens, D_model]]
    """
    features = []
    for idx, block in enumerate(self.blocks):
        if self.encoder_depth is not None and idx == self.encoder_depth and r_timestep_proj is not None:
            timestep_proj = r_timestep_proj
        # A skipped block acts as the identity; still record its feature so each requested index yields one.
        if skip_layers is None or idx not in skip_layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, timestep_proj, encoder_hidden_states, rotary_emb, self.norm_temb
                )
            else:
                hidden_states = block(hidden_states, timestep_proj, encoder_hidden_states, rotary_emb, self.norm_temb)
        if feature_indices is not None and idx in feature_indices:
            features.append(hidden_states)

        # If we have all the features, we can exit early
        if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
            return hidden_states, features

    return hidden_states, features


class WanTextEncoder(FastGenModule):
    def __init__(self, model_id_or_local_path):
        super().__init__()
        self.text_encoder = hf_load(
            UMT5EncoderModel.from_pretrained,
            model_id_or_local_path,
            subfolder="text_encoder",
            # Reference dtype: WanPipeline runs UMT5 in bf16 (only the VAE is fp32).
            dtype=torch.bfloat16,
            # Pin the official eager attention behavior across Transformers releases.
            attn_implementation="eager",
        )

        # Wan explicitly ships a T5 tokenizer under ``tokenizer/``. Selecting
        # it directly avoids AutoTokenizer trying to resolve a nonexistent
        # repository-root config during fully cached offline runs.
        self.tokenizer = hf_load(
            T5TokenizerFast.from_pretrained,
            model_id_or_local_path,
            subfolder="tokenizer",
        )

    def encode(self, conditioning: Optional[Any] = None) -> torch.Tensor:
        conditioning = [conditioning] if isinstance(conditioning, str) else conditioning
        conditioning = [prompt_clean(u) for u in conditioning]

        text_inputs = self.tokenizer(
            conditioning,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self(text_input_ids.to(self.device), mask.to(self.device))
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        return torch.stack([torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0)

    def forward(self, text_input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text_input_ids, mask).last_hidden_state


class WanVideoEncoder(FastGenModule):
    """Wan VAE wrapper with checkpoint normalization at the VAE boundary.

    Original Wan 2.1/2.2 and VACE construct both the VAE and mean/std tensors in
    FP32. Their encode/decode methods run under ``amp.autocast(dtype=self.dtype)``;
    with the default ``self.dtype=torch.float32``, the autocast target, VAE inputs,
    and statistics are all FP32, so the forward and inverse mean/std transforms execute
    in FP32. Diffusers externalizes the transforms around a raw ``AutoencoderKLWan``,
    so ``fixed_precision_model`` selects the wrapper precision here; normalization
    and the returned result stay in that VAE precision.
    """

    def __init__(
        self,
        model_id_or_local_path: str,
        fixed_precision_model: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.vae: AutoencoderKLWan = hf_load(
            AutoencoderKLWan.from_pretrained, model_id_or_local_path, subfolder="vae", dtype=torch.float32
        )
        self.fixed_precision_model = fixed_precision_model

    @property
    def precision_model(self) -> Optional[torch.dtype]:
        """Apply a consumer-fixed VAE dtype, or otherwise inherit the requested precision."""
        fixed_precision = self.fixed_precision_model
        return super().precision_model if fixed_precision is None else fixed_precision

    @property
    def latent_compression_factors(self) -> tuple[int, int, int]:
        """Pixel-to-latent downsampling factors of the video VAE: ``(T, H, W)``.

        Defaults to ``(4, 8, 8)``.
        """
        cfg = self.vae.config
        return (
            getattr(cfg, "scale_factor_temporal", 4),
            getattr(cfg, "scale_factor_spatial", 8),
            getattr(cfg, "scale_factor_spatial", 8),
        )

    def forward(self, data: torch.Tensor, operation: str, mode="sample") -> torch.Tensor:
        if operation == "encode":
            real_images = data.to(dtype=self.precision_model)
            if mode == "sample":
                latent_images = self.vae.encode(real_images, return_dict=False)[0].sample()
            elif mode == "argmax":
                latent_images = self.vae.encode(real_images, return_dict=False)[0].mode()
            else:
                raise ValueError(f"Invalid mode: {mode}. Supported modes: ['sample', 'argmax']")
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent_images.device, latent_images.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latent_images.device, latent_images.dtype
            )

            return (latent_images - latents_mean) * latents_std

        if operation == "decode":
            latent_images = data.to(dtype=self.precision_model)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent_images.device, latent_images.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latent_images.device, latent_images.dtype
            )
            latents = latent_images / latents_std + latents_mean
            videos = self.vae.decode(latents, return_dict=False)[0]
            return videos.clip_(-1.0, 1.0)

        raise ValueError(f"Unsupported preprocessor operation: {operation!r}")

    def encode(self, real_images: torch.Tensor, mode="sample") -> torch.Tensor:
        return self(real_images, mode=mode, operation="encode")

    def decode(self, latent_images: torch.Tensor) -> torch.Tensor:
        return self(latent_images, operation="decode")

    def reencode_overlap(self, latents: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Re-encode the last ``num_frames`` latent frames of ``latents`` as the start of a new clip.

        This VAE encodes a clip's first pixel frame alone and each later latent from ``t`` frames given the ones
        before, so no generated latent looks like a clip's start. This decodes all of ``latents`` (the earlier
        latents give the overlap its context) and encodes afresh the pixel frames the overlap covers, from the last
        frame of its first latent: ``1 + t * (num_frames - 1)`` frames, which give ``num_frames`` latents.
        """
        pixels = self.decode(latents)
        t_factor = self.latent_compression_factors[0]
        # The posterior mean, as the original Wan VAE encodes.
        return self.encode(pixels[:, :, pixels.shape[2] - t_factor * (num_frames - 1) - 1 :], mode="argmax")


class Wan(LoRAMixin, FastGenNetwork):
    """A Wan teacher model for text-to-video diffusion distillation."""

    MODEL_ID_14B = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    MODEL_ID_1_3B = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    MODEL_ID_VER_2_2_T2V_A14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    MODEL_ID_VER_2_2_TI2V_5B_720P = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    def _transformer_blocks(self) -> Tuple[nn.Module, ...]:
        """Return every transformer block; subclasses with additional block lists extend this."""
        return tuple(self.transformer.blocks)

    def _fixed_precision_overrides(self) -> Dict[str, torch.dtype]:
        """Return Wan state that retains the reference model's mixed precision: the rotary tables in FP32.

        Every parameter follows the requested storage precision. Original Wan's FP32 regions compute in at
        least FP32 from it: the time path and the output projection under FP32 autocast, the modulation and
        the norms through explicit upcasts. Later ordinary ``nn.Module.to`` calls intentionally remain uniform.
        """
        overrides = super()._fixed_precision_overrides()
        if not self._keep_original_prec:
            return overrides
        # Diffusers computes the rotary tables in FP64 but registers them in FP32, so FP64 storage only
        # widens the rotation arithmetic ahead of the bf16 q/k rounding: it moves ~1e-5 of the q/k elements
        # by one ulp, is indistinguishable from the attention-kernel choice at the network and sample level,
        # and costs ~3% forward time. Keep FP32.
        overrides.update(
            {name: torch.float32 for name in module_state_names(self) if name.startswith("transformer.rope.")}
        )
        return overrides

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID_1_3B,
        transformer_subfolder: str = "transformer",
        r_timestep: bool = False,
        disable_efficient_attn: bool = False,
        disable_grad_ckpt: bool = False,
        enable_logvar_linear: bool = False,  # Enable logvar_linear if necessary
        r_embedder_init: str = "pretrained",
        time_cond_type: str = "diff",
        norm_temb: bool = False,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        encoder_depth: int | None = None,
        load_pretrained: bool = True,
        use_fsdp_checkpoint: bool = True,
        fps: int = 16,
        sdpa_no_cudnn_blocks: Optional[list] = None,
        r_embedder_fusion: str = "additive",
        r_embedder_gate_value: float = 0.25,
        keep_enc_original_prec: bool = True,
        keep_original_prec: bool = True,
        fp32_residual: bool = True,
        **schedule_kwargs,
    ):
        """Wan2.1/2.2 model constructor.

        Args:
            model_id_or_local_path (str, optional): The huggingface model ID or local path to load.
                Defaults to "Wan-AI/Wan2.1-T2V-1.3B-Diffusers".
            transformer_subfolder (str, optional): Transformer subfolder to load from.
                Defaults to "transformer".
            r_timestep (bool): Whether to support meanflow-like models with r timestep. Defaults to False.
            disable_efficient_attn (bool, optional): Whether to disable efficient attention. Defaults to False.
            disable_grad_ckpt (bool, optional): Whether to disable checkpoints during training. Defaults to False.
            enable_logvar_linear (bool, optional): Whether to enable logvar linear prediction. Defaults to False.
            r_embedder_init (str, optional): Initialization method for the r embedder: "pretrained" copies
                the condition embedder (non-zero r contribution at init), "zero" keeps the copied hidden
                layer and zeroes the output layers (trainable; a no-op at init only for additive fusion
                without ``encoder_depth``), "random" re-initializes the hidden layer randomly and zeroes the
                output layers. Defaults to "pretrained".
            time_cond_type (str, optional): Time condition type for r timestep. Defaults to "diff".
            norm_temb (bool, optional): Whether to normalize the time embeddings. Defaults to False.
            net_pred_type (str, optional): Prediction type. Defaults to "flow".
            schedule_type (str, optional): Schedule type. Defaults to "rf".
            encoder_depth (int, optional): The depth of the encoder (i.e. the number of blocks taking in t embeddings).
                Defaults to None, meaning all blocks take in [t embeddings + r embeddings].
            load_pretrained (bool, optional): Whether to load the model from pretrained.
                If False, the model will be initialized from the config.
                Defaults to True.
            use_fsdp_checkpoint (bool, optional): Whether to use FSDP gradient checkpointing. Defaults to True.
            fps (int): Frames per second. Defaults to 16 for all Wan models.
            sdpa_no_cudnn_blocks (list, optional): List of self-attention block indices
                whose self-attention SDPA is forced off the cuDNN backend (onto Flash / mem-efficient).
                Works around the cuDNN bf16 SDPA-backward kernel emitting non-finite query gradients on
                those blocks' large activations. Defaults to None (disabled).
            r_embedder_fusion (str, optional): How the t and r time embeddings are combined,
                "additive" (MeanFlow default) or "gated" (AnyFlow). Defaults to "additive".
            r_embedder_gate_value (float, optional): Convex gate ``g`` of the "gated" fusion,
                ``(1-g)*temb + g*remb``. Ignored for "additive". Defaults to 0.25.
            keep_enc_original_prec (bool, optional): Keep the frozen encoders in original Wan's precision
                while returning their outputs in the requested model dtype: the VAE in float32 and, for
                I2V models, the CLIP image encoder in float16 whatever the enclosing autocast. Defaults
                to True.
            keep_original_prec (bool, optional): Preserve the transformer's reference mixed
                precision: as in original Wan, the time path and the output projection compute in at
                least FP32 under FP32 autocast, which needs CUDA for reduced-precision parameters, the
                attention Q/K norms round like ``WanRMSNorm``, and the rotary tables remain FP32 (Diffusers'
                registered precision). Parameters use the requested dtype either way. When disabled, the
                time path and the output projection compute like the rest of the transformer, and the Q/K
                norms are Diffusers' ``torch.nn.RMSNorm``. Defaults to True.
            fp32_residual (bool, optional): Keep the residual stream in FP32 across blocks as in original
                Wan. When disabled, every block rounds its output to the model dtype, which removes the FP32
                checkpoint-boundary activations (about 2 bytes per token per hidden unit per block; 11 GiB
                per GPU for the 14B 480p recipe). Defaults to True.
            **schedule_kwargs: Noise-schedule keyword arguments, forwarded to the FastGenNetwork constructor.
        """
        self.transformer_subfolder = transformer_subfolder

        # Initialize FastGenNetwork with Wan-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)
        self._keep_original_prec = keep_original_prec

        if disable_efficient_attn:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

        self._use_fsdp_checkpoint = use_fsdp_checkpoint

        inner_dim = self._initialize_network(model_id_or_local_path, load_pretrained)
        self.model_id_or_local_path = model_id_or_local_path
        # Only Wan2.2's TI2V-5B works on the Wan2.2 VAE's 48-channel latents: it downsamples 16x spatially, as in
        # ``pixel_shape_for``, and conditions every latent token on its own timestep.
        wan22_vae = self.transformer.config.out_channels == 48
        self.latent_compression_factors = (4, 16, 16) if wan22_vae else (4, 8, 8)
        self.expand_timesteps = wan22_vae

        self._keep_enc_original_prec = keep_enc_original_prec
        self.fps = fps

        # Self-attn blocks to force off the cuDNN SDPA backend; consumed by
        # _install_sdpa_no_cudnn_hooks inside override_transformer_forward() below.
        self._sdpa_no_cudnn_blocks = sdpa_no_cudnn_blocks

        # Add logvar_linear layer (conditional based on enable_logvar_linear flag)
        if enable_logvar_linear:
            self.transformer.logvar_linear = torch.nn.Linear(inner_dim, 1)
            logger.info("Added logvar_linear layer")
        else:
            logger.info("Skipped logvar_linear layer (disabled by enable_logvar_linear=False)")
        self.transformer.norm_temb = norm_temb
        self.transformer.encoder_depth = encoder_depth
        self.transformer.time_cond_type = time_cond_type
        if r_timestep:
            logger.info(f"Initializing r embedder with {r_embedder_init}")
            self.transformer.r_embedder = self.init_embedder(r_embedder_init, r_embedder_fusion, r_embedder_gate_value)
        else:
            self.transformer.r_embedder = None

        # core functionality to override forward function and other methods in the transformer
        self.override_transformer_forward()
        for block in self._transformer_blocks():
            block.fp32_residual = fp32_residual
        # Resolve time-MLP inputs from the projections that consume them, and keep ``temb``
        # in the time MLP's dtype. Both are installed here so every Wan/VACE/causal prepare
        # path shares the same dtype boundary without threading a policy through their forwards.
        for embedder in (self.transformer.condition_embedder, self.transformer.r_embedder):
            if embedder is not None:
                embedder.time_embedder.forward = types.MethodType(
                    wan_timestep_embedding_forward, embedder.time_embedder
                )
        self.transformer.condition_embedder.forward = types.MethodType(
            wan_condition_embedding_forward, self.transformer.condition_embedder
        )

        # Use lazy initialization as this wont work in a meta context when doing FSDP2.
        self._unipc_scheduler = None

        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    def sample_scheduler_and_grid(
        self, sample_steps: int, flow_shift: float, device: Optional[torch.device] = None
    ) -> Tuple[UniPCMultistepScheduler, torch.Tensor]:
        """Lazily load UniPC and set it to the noise schedule's grid, shifted by ``flow_shift``.

        The grid is built in float32 and passed pre-shifted (UniPC's own ``flow_shift`` is 1), so UniPC stores it bit
        for bit. Each call also resets UniPC's multistep state.

        Returns:
            The scheduler and the times to query the network at, one per step: its sigmas in ``t_precision``.
        """
        if self._unipc_scheduler is None:
            self._unipc_scheduler = hf_load(
                UniPCMultistepScheduler.from_pretrained,
                self.model_id_or_local_path,
                subfolder="scheduler",
                flow_shift=1.0,  # the grid below is pre-shifted
            )
        sigmas = self.noise_scheduler.shift_t(
            self.noise_scheduler.grid(sample_steps, device=device, dtype=torch.float32), shift=flow_shift
        )
        sigmas_np = sigmas[:-1].cpu().numpy()  # the scheduler appends the terminal sigma 0 itself
        self._unipc_scheduler.set_timesteps(num_inference_steps=sample_steps, sigmas=sigmas_np, device=device)
        # Start at step 0: a fine grid repeats its first integer timestep, and UniPC's lookup takes the second match.
        self._unipc_scheduler.set_begin_index(0)
        t_grid = self.noise_scheduler.safe_clamp(
            self._unipc_scheduler.sigmas[:-1].to(device=device, dtype=self.noise_scheduler.t_precision),
            min=self.noise_scheduler.min_t,
            max=self.noise_scheduler.max_t,
            warn=True,
        )
        return self._unipc_scheduler, t_grid

    def _initialize_network(self, model_id_or_local_path: str, load_pretrained: bool) -> int:
        """Initialize the network.
        Args:
            model_id_or_local_path (str): The model ID or local path to load. This can be either a HuggingFace
                model ID or a local directory holding the model weights.
            load_pretrained (bool): Whether to load the model from pretrained. If False, the model will be initialized from the config.

        Returns:
            int: The transformer's inner dimension.
        """
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info(f"Loading Wan transformer from subfolder '{self.transformer_subfolder}'")
            self.transformer: WanTransformer3DModel = hf_load(
                WanTransformer3DModel.from_pretrained,
                model_id_or_local_path,
                subfolder=self.transformer_subfolder,
            )
        else:
            # Load config and create model structure
            # If we're in a meta context, tensors will automatically be on meta device
            config = hf_load(
                WanTransformer3DModel.load_config,
                model_id_or_local_path,
                subfolder=self.transformer_subfolder,
            )
            if in_meta_context:
                logger.info(
                    "Initializing Wan transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing Wan transformer from config (no pretrained weights)")
                logger.warning("Wan Transformer being initializated from config. No weights are loaded!")
            self.transformer: WanTransformer3DModel = WanTransformer3DModel.from_config(config)

        config = self.transformer.config
        return config.num_attention_heads * config.attention_head_dim

    def reset_parameters(self):
        """Recompute the non-persistent RoPE tables in place after meta materialization.

        ``freqs_cos``/``freqs_sin`` are absent from the state dict, so every rank rebuilds them from the
        transformer config; the identity-preserving ``copy_`` keeps the FSDP registrations intact.
        """
        # Reinitialize RoPE buffers by re-instantiating the rope module
        # We need this as they are no persistent, and wont be synced by set_model_state_dict
        # when using FSDP2
        # The rope module computes freqs_cos/freqs_sin from the transformer config
        if hasattr(self.transformer, "rope") and self.transformer.rope is not None:
            config = self.transformer.config
            rope_config = {
                "attention_head_dim": config.attention_head_dim,
                "patch_size": config.patch_size,
                "max_seq_len": config.rope_max_seq_len,
                "theta": getattr(config, "rope_theta", 10000.0),
            }

            # Get the current device (after to_empty, buffers are on the target device)
            device = next(self.transformer.rope.buffers()).device

            # Re-instantiate rope on the correct device to compute buffers
            new_rope = WanRotaryPosEmbed(**rope_config).to(device)

            # Copy the computed buffers to the existing rope module
            # This preserves the module identity (important for FSDP)
            with torch.no_grad():
                self.transformer.rope.freqs_cos.copy_(new_rope.freqs_cos)
                self.transformer.rope.freqs_sin.copy_(new_rope.freqs_sin)

            del new_rope
            logger.debug("Reinitialized RoPE buffers (freqs_cos, freqs_sin)")

        super().reset_parameters()

    def fully_shard(self, **kwargs):
        """Shard ``self.transformer``; see ``FastGenNetwork.fully_shard``."""
        # Checkpointing has to happen first, for proper casting during backward pass recomputation.
        if self.transformer.gradient_checkpointing and self._use_fsdp_checkpoint:
            self.transformer.disable_gradient_checkpointing()
            block_types = tuple({type(block) for block in self._transformer_blocks()})
            apply_fsdp_checkpointing(self.transformer, block_types)

        # FSDP2 keeps the root group gathered after forward, so every block needs its own group.
        for block in self._transformer_blocks():
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

    @classmethod
    def pixel_shape_for(cls, latent_shape: Sequence[int]) -> Tuple[int, ...]:
        """``FastGenNetwork.pixel_shape_for`` with the compression of the VAE that the latent channels identify.

        Wan2.2's VAE (TI2V-5B) has 48 latent channels and downsamples 16x spatially, the Wan2.1 VAE 16 and 8x.
        """
        factors = (4, 16, 16) if latent_shape[0] == 48 else (4, 8, 8)
        return pixel_shape_from_factors(latent_shape, factors)

    def init_embedder(
        self,
        embedder_init: str,
        r_embedder_fusion: str = "additive",
        r_embedder_gate_value: float = 0.25,
    ) -> torch.nn.Module:
        """Build the r-embedder and stash its fusion config on the module.

        The fusion settings live on the module (not on `self`) so the
        method-bound forward override can branch on them without changing its
        signature. They are set before the meta-device early return so both
        paths carry them.

        The copied text and image branches are always dropped because the r path
        consumes only timestep modules. For gated fusion, its own `time_proj` /
        `act_fn` are also dropped because the fused embedding goes through the
        condition embedder's shared projection.
        """
        embedder = copy.deepcopy(self.transformer.condition_embedder)
        del embedder.text_embedder
        del embedder.image_embedder

        if r_embedder_fusion not in ("additive", "gated"):
            raise ValueError(f"r_embedder_fusion must be 'additive' or 'gated', got {r_embedder_fusion!r}")
        embedder.fusion_mode = r_embedder_fusion
        # Plain float (not a buffer) so FSDP's reset_parameters does not need to
        # re-materialize it after meta-device init.
        embedder.gate_value = float(r_embedder_gate_value)
        logger.info(
            f"r_embedder fusion={r_embedder_fusion}"
            + (f" gate_value={r_embedder_gate_value}" if r_embedder_fusion == "gated" else "")
        )

        # Skip initialization if using meta device (weights will be broadcast via FSDP)
        if self._is_in_meta_context():
            logger.info("Skipping r_embedder initialization on meta device (will receive weights via FSDP sync)")
        elif embedder_init == "zero":
            # Zero only the output layers: no-op at init, but the copied hidden layer
            # keeps the gradients flowing (all-zero weights would never leave zero).
            for linear in (embedder.time_embedder.linear_2, embedder.time_proj):
                torch.nn.init.zeros_(linear.weight)
                if linear.bias is not None:
                    torch.nn.init.zeros_(linear.bias)
        elif embedder_init == "random":
            # following https://github.com/Wan-Video/Wan2.1/blob/7c81b2f27defa56c7e627a4b6717c8f2292eee58/wan/modules/model.py#L609
            for param in embedder.modules():
                if isinstance(param, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(param.weight)
                    if param.bias is not None:
                        torch.nn.init.zeros_(param.bias)
            for param in embedder.time_embedder.modules():
                if isinstance(param, torch.nn.Linear):
                    torch.nn.init.normal_(param.weight, std=0.02)

            # to have zero remb initially
            torch.nn.init.zeros_(embedder.time_embedder.linear_2.weight)
            torch.nn.init.zeros_(embedder.time_embedder.linear_2.bias)
            # to have zero r_timestep_proj initially
            torch.nn.init.zeros_(embedder.time_proj.weight)
            torch.nn.init.zeros_(embedder.time_proj.bias)
        elif embedder_init == "pretrained":
            # Since the r_embedder is already copied from the condition_embedder, we don't need to do anything
            pass
        else:
            raise ValueError(f"Invalid embedder_init: {embedder_init}")

        if r_embedder_fusion == "gated":
            # The gated path projects the fused (t, r) embedding through the
            # condition_embedder's own time_proj, shared with the t-only path.
            # Drop the r_embedder's copy so it cannot linger as an untrained
            # dead parameter. Done after the init branches above because
            # "random" initializes time_proj.
            del embedder.time_proj
            del embedder.act_fn
        return embedder

    def _install_sdpa_no_cudnn_hooks(self) -> None:
        """Force the self-attention SDPA of the configured blocks onto a non-cuDNN backend
        (Flash / mem-efficient), per the ``sdpa_no_cudnn_blocks`` constructor argument.

        Findings (Wan2.2-I2V-A14B DMD2, bf16, torch>=2.10): training NaN'd sporadically because
        the **cuDNN scaled-dot-product-attention backward** kernel emits non-finite query
        gradients (dq) on the low-noise expert's block-33 self-attention, where the qk-norm
        produces "massive-activation" outliers (|Q-hat|,|K-hat| ~ 90-120). cuDNN is the *default*
        SDPA backend for these shapes, so it NaN'd from ~iter 30. The Flash and mem-efficient
        backends compute the identical attention finitely on the exact same inputs (and fp32 is
        also clean), so this is a cuDNN-bf16-backward precision artifact -- NOT a fundamental
        overflow in the attention math. Forcing the affected blocks off cuDNN is
        therefore a free, distortion-free fix.

        What else we tried (on the culprit block) and its effect:
          - Bounding the qk-norm WEIGHTS -- hard clamp and soft-clamp to ~3.5:
            ALL still diverged. The driver is the activation magnitude (massive-activation channels),
            which weight-bounding barely reduces.
          - Bounding the ACTIVATION -- hard output-clamp (+/-15), soft-cap (20*tanh), uniform
            down-scale (x0.6), per-head L2-renorm (<=20): all reached 0 NaN, but they distort
            attention values.
          - fp32 attention on the block: 0 NaN, no value distortion, but adds compute.
          - Reactive grad sanitizing (zero the non-finite attn grads): keeps training alive but is
            a band-aid -- the overflowing layer's grads are zeroed every step, so it stops learning.

        ``sdpa_no_cudnn_blocks``: list of self-attn block indices (e.g. ``[33]`` for the low-noise
        expert culprit). Empty (batch-0) MoE-routed expert forwards fall back to the default
        backend, since Flash/mem-efficient reject batch size 0.
        """
        if not self._sdpa_no_cudnn_blocks:
            return
        n_blocks = len(self.transformer.blocks)
        only = sorted({i for i in self._sdpa_no_cudnn_blocks if 0 <= i < n_blocks})
        for i in only:
            attn = getattr(self.transformer.blocks[i], "attn1", None)
            if attn is None:
                continue
            if "forward" in attn.__dict__:
                raise RuntimeError(
                    f"blocks.{i}.attn1.forward is already instance-bound; the no-cuDNN SDPA hook cannot be stacked on it"
                )
            attn.forward = types.MethodType(attention_forward_without_cudnn, attn)
        logger.info(f"[sdpa-no-cudnn] forced Flash/mem-efficient on self-attn blocks={only}.")

    def override_transformer_forward(self) -> None:
        # Override transformer forward methods with custom implementations
        for block in self.transformer.blocks:
            block.forward = types.MethodType(block_forward, block)
        self._install_sdpa_no_cudnn_hooks()
        self.transformer.classify_forward_prepare = types.MethodType(classify_forward_prepare, self.transformer)
        self.transformer.classify_forward_block_forward = types.MethodType(
            classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Original Wan fp64 timestep sinusoid instead of diffusers' fp32 ``Timesteps``.
        self._override_timesteps_proj(self.transformer.condition_embedder.timesteps_proj)
        if self.transformer.r_embedder is not None:
            self._override_timesteps_proj(self.transformer.r_embedder.timesteps_proj)

        # Original Wan's FP32 autocast regions around the time path and the head.
        for module in (self.transformer, self.transformer.condition_embedder, self.transformer.r_embedder):
            if module is not None:
                module.keep_original_prec = self._keep_original_prec

        # Original Wan's Q/K norm rounding in every attention.
        if self._keep_original_prec:
            for block in self._transformer_blocks():
                for module in block.modules():
                    if isinstance(module, nn.RMSNorm):
                        module.forward = types.MethodType(wan_rms_norm_forward, module)

    def _override_timesteps_proj(self, timesteps_proj_module) -> None:
        """Replace a diffusers ``Timesteps`` forward with the original Wan fp64 sinusoid.

        Diffusers' fp32 ``Timesteps`` (same formula and [cos, sin] order) is a drop-in alternative.
        """
        num_channels = timesteps_proj_module.num_channels

        def new_forward(self, timesteps):
            # Use original WAN sinusoidal embedding instead of diffusers default sinusoidal embedding
            return sinusoidal_embedding_1d_wan(num_channels, timesteps)

        timesteps_proj_module.forward = types.MethodType(new_forward, timesteps_proj_module)

    def init_preprocessors(self):
        """Initialize the text and video encoders for the Wan model."""
        if "text_encoder" not in self.preprocessors:
            self.init_text_encoder()
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder for Wan model."""
        self.preprocessors["text_encoder"] = WanTextEncoder(model_id_or_local_path=self.model_id_or_local_path)

    def init_vae(self):
        """Initialize the video encoder for Wan model."""
        self.preprocessors["vae"] = WanVideoEncoder(
            model_id_or_local_path=self.model_id_or_local_path,
            fixed_precision_model=torch.float32 if self._keep_enc_original_prec else None,
        )

    def _unpatchify_features(self, x_t: torch.Tensor, features: List[torch.Tensor]) -> List[torch.Tensor]:
        B, _, T, H, W = x_t.shape
        p_t, p_h, p_w = self.transformer.config.patch_size
        post_patch_num_frames = T // p_t
        post_patch_height = H // p_h
        post_patch_width = W // p_w
        feats = []
        for feat in features:
            feat = feat.reshape(
                B,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                p_t,
                p_h,
                p_w,
                feat.shape[-1] // (p_t * p_h * p_w),
            )
            feat = feat.permute(0, 7, 1, 4, 2, 5, 3, 6)
            feat = feat.flatten(6, 7).flatten(4, 5).flatten(2, 3)
            feats.append(feat)
        return feats

    def _compute_timestep_inputs(self, timestep: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute timestep input used for Wan models.
        Optionally Expand or mask the timestep input for Wan 2.2 TI2V models.
            - I2V: Apply a mask that zeroes out the timestep for the first latent frame.
            - T2V: Use a mask tensor filled with ones.
        """
        timestep = self.noise_scheduler.rescale_t(timestep)

        if self.expand_timesteps and mask is not None:
            p_t, p_h, p_w = self.transformer.config.patch_size
            # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
            timestep = (mask[:, ::p_t, ::p_h, ::p_w].to(timestep.dtype) * timestep.view(-1, 1, 1, 1)).flatten(1)
        return timestep

    @staticmethod
    def default_guidance_cfg() -> GuidanceConfig:
        """Default guidance recipe: CFG only at scale 5.0.

        Experiments may override via ``config.model.guidance_cfg``.
        """
        g = GuidanceConfig()
        g.cfg.scale = 5.0
        return g

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        sample_steps: int = 50,
        sample_type: str = "ode",
        shift: float = 5.0,
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Multistep sample using the UniPC method.

        Args:
            noise (torch.Tensor): The noisy latents to start from.
            condition: The condition.
            neg_condition: The negative condition (used for CFG when enabled).
            sample_steps: The number of sampling steps.
            sample_type: Integration policy; only ``"ode"`` is supported.
            shift: Noise schedule shift parameter.
            guidance_cfg: Composite guidance config.  When ``None`` (the
                default), ``default_guidance_cfg()`` is used.

        Returns:
            torch.Tensor: The sample output.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )

        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()

        scheduler, t_grid = self.sample_scheduler_and_grid(sample_steps, flow_shift=shift, device=noise.device)

        # Initialize latents with proper scaling based on the initial timestep
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])

        # main sampling loop
        for sched_step, t in tqdm(zip(scheduler.timesteps, t_grid, strict=True), total=len(t_grid), desc="Sampling"):
            t = t.expand(latents.shape[0])

            flow_pred = predict_with_guidance(
                self,
                latents,
                t,
                guidance_cfg=guidance_cfg,
                condition=condition,
                neg_condition=neg_condition,
                fwd_pred_type="flow",
            )

            latents = scheduler.step(flow_pred, sched_step, latents, return_dict=False)[0]

        return latents

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        """Load state dict with automatic extraction for original Wan checkpoint formats.

        Also handles original Wan checkpoints nested under a common key, e.g., {'generator': state_dict}, and bare
        diffusers keys without the ``transformer.`` prefix, including AnyFlow releases (see ``remap_anyflow_keys``).
        """
        if not any(k.startswith("transformer.") for k in state_dict.keys()):
            state_dict, nested_key = _unwrap_state_dict(state_dict)
            # Unwrapping can reveal adapter keys hidden from the generic load preflight.
            if nested_key is not None and not self.lora_enabled and any("lora_" in name for name in state_dict):
                raise RuntimeError(
                    "The state dict holds LoRA adapter tensors but the target has no initialized adapter; "
                    "merge the adapter into the base weights or initialize it before loading."
                )
        if not any(k.startswith("transformer.") for k in state_dict.keys()):
            if any(marker in k for k in state_dict for marker in WAN_OFFICIAL_KEY_MARKERS):
                state_dict = convert_wan_official_state_dict(state_dict)
            else:
                state_dict = {f"transformer.{k}": v for k, v in remap_anyflow_keys(state_dict).items()}

        return super().load_state_dict(state_dict, **kwargs)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        skip_layers: Optional[List[int]] = None,
        unpatchify_features: bool = True,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the Wan diffusion score model.

        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (torch.Tensor): The text conditioning embeddings. Defaults to None.
            r (torch.Tensor): Another timestep mainly used by meanflow.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. When provided, features are returned alongside the model output;
                an empty set returns an empty feature list. By default, None returns only the model output.
            return_logvar: If true, the forward pass also returns the logvar, shape [B, 1].
            fwd_pred_type: Update the network prediction type, must be in ['x0', 'eps', 'v', 'flow'].
                None means using the original net_pred_type.
            skip_layers: Apply skip-layer guidance by skipping layers of the unconditional network during forward pass.
            unpatchify_features: If true, the features are unpatchified to [B, C', T, H, W] (C' = hidden size / patch
                volume); otherwise they stay packed as [B, N_tokens, D].
            apply_output_conditioning: Apply the output constraint after the conversion.
            output_dtype: ``"same"`` rounds the prediction to the state dtype; ``None`` keeps it in high precision.
        Returns:
            torch.Tensor: The score model output.
        """
        self._validate_forward_kwargs(
            fwd_kwargs,
            r=r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            skip_layers=skip_layers,
            supports_r_timestep=getattr(self.transformer, "r_embedder", None) is not None,
            supports_skip_layers=True,
            num_features=len(self.transformer.blocks),
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        assert isinstance(condition, torch.Tensor), "condition must be a tensor"
        x_t_model = self.cast_to_model_precision(x_t)
        assert_same(dtype=self.precision_model, device=x_t_model.device, condition=condition)
        assert_same(
            shape=True,
            dtype=self.noise_scheduler.t_precision,
            device=True,
            t=t,
            r=r,
        )
        timestep_mask = torch.ones_like(x_t[:, 0])  # shape: [batch_size, num_latent_frames, H, W]
        timestep = self._compute_timestep_inputs(t, timestep_mask)
        r_timestep = None if r is None else self._compute_timestep_inputs(r, timestep_mask)

        # Pass the primary tensor inputs positionally so FSDP2 input handling is
        # preserved across activation-checkpoint recompute.
        model_outputs = self.transformer(
            x_t_model,
            timestep,
            condition,
            r_timestep=r_timestep,
            attention_kwargs=None,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            skip_layers=skip_layers,
        )

        if return_features_early:
            assert len(model_outputs) == len(feature_indices)
            return self._unpatchify_features(x_t_model, model_outputs) if unpatchify_features else model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        if feature_indices is None:
            assert isinstance(out, torch.Tensor)
            out = self.transform_output(out, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning)
        else:
            assert isinstance(out, list)
            out[0] = self.transform_output(
                out[0], x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
            )
            out[1] = self._unpatchify_features(x_t_model, out[1]) if unpatchify_features else out[1]

        if return_logvar:
            return out, logvar.to(dtype=x_t.dtype)
        return out
