# SPDX-FileCopyrightText: Copyright (c) 2025 The Lightricks team and The HuggingFace Team. All rights reserved.
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

"""LTX-2.0, LTX-2.3 and LTX-2.5 FastGen network.

The canonical model IDs (``Lightricks/LTX-2``, ``diffusers/LTX-2.3-Diffusers``, ``Lightricks/LTX-2.5-Diffusers``)
select their release; other IDs need an explicit ``ltx2_version``. 2.3/2.5 follow current ltx_core for RoPE
coordinate precision and cross-modal AdaLN inputs; 2.0 keeps its release-era behavior.
"""

import math
import types
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import attrs
import torch
import torch.nn as nn
from diffusers.models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from diffusers.models.transformers import LTX2VideoTransformer3DModel
from diffusers.models.transformers.transformer_ltx2 import LTX2Attention, LTX2VideoTransformerBlock
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder, LTX2VocoderWithBWE
from torch.distributed.fsdp import fully_shard
from transformers import (
    Gemma3ForConditionalGeneration,
    Gemma4UnifiedForConditionalGeneration,
    GemmaTokenizerFast,
)

from fastgen.networks.lora_utils import scaled_lora_layers
from fastgen.networks.network import FastGenNetwork, FastGenModule, validate_conditioning
from fastgen.networks.noise_schedule import BaseNoiseSchedule, CompositeNoiseSchedule
from fastgen.utils.guidance import (
    GuidanceConfig,
    GuidanceContext,
    GuidanceTerm,
    apply_cfg,
    apply_guidance,
    apply_mix,
    apply_rescale,
    is_active,
    predict_with_guidance,
    resolve_piecewise,
)
from fastgen.utils.tree_utils import (
    OutputDtype,
    TensorOrDict,
    assert_same,
    expand_first_dim,
    get_batch_size,
    get_device,
    per_modality,
    to,
)
from fastgen.utils.tensor import assert_tensor, expand_like, high_precision_dtype, promote_dtype, to_promoted_dtype
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing, make_fully_shard_kwargs
from fastgen.utils.distributed import is_rank0
import fastgen.utils.logging_utils as logger
from fastgen.utils.module import module_state_names
from fastgen.utils.env import hf_load
from fastgen.utils.audio import MelSpectrogram


def _validate_modalities(modalities: Sequence[str]) -> tuple[str, ...]:
    modalities = tuple(modalities)
    supported = {"video", "audio"}
    if not modalities or len(modalities) != len(set(modalities)) or not set(modalities) <= supported:
        raise ValueError(
            f"modalities must be a non-empty selection without duplicates from {supported}; got {modalities}"
        )
    return modalities


def _validate_input(
    data: Any,
    modalities: Sequence[str],
    input_name: str = "forward()",
    multimodal_owner: str = "network",
) -> None:
    """Validate the shared Tensor-or-modality-dict input contract."""
    if isinstance(data, torch.Tensor):
        if len(modalities) != 1:
            raise ValueError(
                f"bare Tensor input requires exactly one modality; got modalities={modalities}. "
                f"Pass a dict for a multimodal {multimodal_owner}."
            )
        return

    if not isinstance(data, dict):
        raise TypeError(f"{input_name} input must be a Tensor or dict, got {type(data).__name__}")

    keys = set(data)
    if not keys or not keys <= set(modalities):
        raise ValueError(
            f"{input_name} dict input keys must be a non-empty subset of configured modalities "
            f"{set(modalities)}; got {keys}"
        )


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    """Pack video latents [B, C, F, H, W] → [B, F//pt * H//p * W//p, C*pt*p*p]."""
    B, C, F, H, W = latents.shape
    pF = F // patch_size_t
    pH = H // patch_size
    pW = W // patch_size
    latents = latents.reshape(B, C, pF, patch_size_t, pH, patch_size, pW, patch_size)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    """Unpack video latents [B, T, D] → [B, C, F, H, W]."""
    B = latents.size(0)
    latents = latents.reshape(B, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack audio latents [B, C, L, M] → [B, L, C*M]."""
    return latents.transpose(1, 2).flatten(2, 3)


def _unpack_audio_latents(latents: torch.Tensor, num_mel_bins: int) -> torch.Tensor:
    """Unpack audio latents [B, L, C*M] -> [B, C, L, M]."""
    return latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)


def _audio_latent_frames(waveform_samples: int, hop_length: int, temporal_compression: int) -> int:
    """Latent frames the audio VAE emits for ``waveform_samples``.

    The centered mel has ``samples // hop_length + 1`` frames; each causal stride-2 stage emits
    ``ceil(T / 2)``. This is the reference trainer's count. Reference and diffusers inference use
    ``round(duration * latents_per_second)``, often one frame fewer (251 vs. 252 for 241 frames at 24 fps).
    """
    mel_frames = waveform_samples // hop_length + 1
    return math.ceil(mel_frames / temporal_compression)


def _validate_audio_video_fps(
    real_raw: Mapping[str, torch.Tensor], fps: float, sampling_rate: int, hop_length: int, temporal_compression: int
) -> None:
    """Check that a raw audio-video clip runs at ``fps``: its audio must encode to the latent count inferred for it.

    The loader trims each waveform to the clip's frames at the clip's own frame rate, while ``LTX2.infer_shape`` sizes
    the audio latent from the frames at the network's ``fps`` (``_compute_audio_shape``). The two counts must agree,
    or the encoded audio drifts from ``input_shape`` and audio and video RoPE positions disagree.
    """
    frames, samples = real_raw["video"].shape[2], real_raw["audio"].shape[-1]
    encoded = _audio_latent_frames(samples, hop_length, temporal_compression)
    inferred = _audio_latent_frames(int(frames * sampling_rate / fps), hop_length, temporal_compression)
    if encoded != inferred:
        raise ValueError(
            f"Audio-video data with {frames} frames and {samples} audio samples encodes to {encoded} audio latents, "
            f"but this LTX-2 network infers {inferred} for {frames} frames at fps={fps} and {sampling_rate} Hz. "
            f"Resample the data with the loader's target_fps={fps} (and target_sample_rate={sampling_rate}), or set "
            f"the network's fps."
        )


# The token counts at which ``LTX2Scheduler``'s shift reaches ``base_shift`` and ``max_shift``.
_LTX2_BASE_SHIFT_ANCHOR = 1024
_LTX2_MAX_SHIFT_ANCHOR = 4096


def _ltx2_sigma_schedule(
    sample_steps: int,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    image_seq_len: int = _LTX2_MAX_SHIFT_ANCHOR,
    stretch: bool = True,
    terminal: float = 0.1,
) -> torch.Tensor:
    """Sigmas of ltx_core's ``LTX2Scheduler``: exponential time shift, then stretch to ``terminal``.

    Returns ``sample_steps + 1`` float32 values ending at the terminal zero, bit for bit the reference's: the float32
    CPU arithmetic follows it operation by operation, and its shift expression rounds differently from
    ``time_shift``'s. The defaults are the reference's, which every release's scheduler config repeats;
    ``image_seq_len=4096`` matches the reference pipelines.
    ``stretch=True`` needs ``sample_steps >= 2``: one step would divide by zero in the stretch.
    """
    if stretch and sample_steps < 2:
        raise ValueError(
            f"_ltx2_sigma_schedule requires sample_steps >= 2 with stretch=True; got {sample_steps}. "
            "The stretch step divides by zero when there is no non-terminal sigma."
        )
    m = (max_shift - base_shift) / (_LTX2_MAX_SHIFT_ANCHOR - _LTX2_BASE_SHIFT_ANCHOR)
    shift = math.exp(image_seq_len * m + (base_shift - m * _LTX2_BASE_SHIFT_ANCHOR))
    sigmas = torch.linspace(1.0, 0.0, sample_steps + 1, dtype=torch.float32, device="cpu")
    sigmas = torch.where(sigmas != 0, shift / (shift + (1 / sigmas - 1)), 0)
    if stretch:
        nonzero = sigmas != 0
        one_minus_z = 1.0 - sigmas[nonzero]
        sigmas[nonzero] = 1.0 - one_minus_z / (one_minus_z[-1] / (1.0 - terminal))
    return sigmas


_LTX2_VERSION_BY_MODEL_ID = {
    "Lightricks/LTX-2": "2.0",
    # HF repo with transformer/vae/vocoder configs for Diffusers `from_pretrained`; Lightricks/LTX-2.3 is weights-only.
    "diffusers/LTX-2.3-Diffusers": "2.3",
    "Lightricks/LTX-2.5-Diffusers": "2.5",
}


def _validate_ltx2_version(version: Optional[str]) -> str:
    if version is None:
        raise ValueError("LTX-2 version must be specified")
    version = str(version)
    supported_versions = tuple(_LTX2_VERSION_BY_MODEL_ID.values())
    if version not in supported_versions:
        raise ValueError(f"Unsupported LTX-2 version: {version!r}. Expected one of {supported_versions}.")
    return version


def _resolve_ltx2_version(model_id_or_local_path: str, version: Optional[str]) -> str:
    """Validate an explicit release, or use the exact canonical-model default."""
    if version is None:
        version = _LTX2_VERSION_BY_MODEL_ID.get(str(model_id_or_local_path))
        if version is None:
            raise ValueError(
                f"No default LTX-2 release is registered for model_id_or_local_path={model_id_or_local_path!r}; "
                "set ltx2_version explicitly."
            )
    return _validate_ltx2_version(version)


def _ltx2_transformer_subfolder(
    model_id_or_local_path: str,
    override: Optional[str] = None,
    version: Optional[str] = None,
) -> str:
    """Return ``override``, else ``"transformer_full"`` (full/SFT) for 2.5 and ``"transformer"`` otherwise."""
    if override is not None:
        return override
    resolved_version = _resolve_ltx2_version(model_id_or_local_path, version)
    return "transformer_full" if resolved_version == "2.5" else "transformer"


def _ltx2_cross_modal_timestep_inputs(
    version: str,
    video_timestep: torch.Tensor,
    audio_timestep: torch.Tensor,
    video_sigma: torch.Tensor,
    audio_sigma: torch.Tensor,
    gate_scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-modal AdaLN timestep inputs: video scale/shift, A2V gate, audio scale/shift, V2A gate.

    Inputs are already in the embedding domain; only the gate factor is applied here. 2.3/2.5 follow
    current ltx_core: a modality's per-token timestep drives its scale/shift and the other modality's
    global sigma drives its gate. Upstream supports only 2.3/2.5 and routes 2.3 through the current
    pipeline (https://github.com/Lightricks/LTX-2/issues/160#issuecomment-5328260608,
    https://github.com/Lightricks/LTX-2/issues/178#issuecomment-5328255960); the unsupported 2.0 keeps
    its historical policy of its own per-token timestep for both.
    """
    if version == "2.0":
        video_scale_shift = video_timestep
        video_gate = video_timestep
        audio_scale_shift = audio_timestep
        audio_gate = audio_timestep
    elif version in ("2.3", "2.5"):
        video_scale_shift = video_timestep
        video_gate = audio_sigma
        audio_scale_shift = audio_timestep
        audio_gate = video_sigma
    else:
        raise ValueError(f"Unsupported LTX-2 version: {version}")

    return (
        video_scale_shift.flatten(),
        video_gate.flatten() * gate_scale_factor,
        audio_scale_shift.flatten(),
        audio_gate.flatten() * gate_scale_factor,
    )


def _apply_ltx2_first_frame_embedding(
    hidden_states: torch.Tensor,
    keyframes_embedding: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Apply LTX-2.5's learned embedding to the first causal latent frame."""
    first_frame_mask = torch.zeros_like(hidden_states[..., :1])
    first_frame_mask[:, : height * width] = 1
    return hidden_states + first_frame_mask * keyframes_embedding.to(hidden_states.dtype)


def _ltx2_first_frame_embedding_enabled(transformer: nn.Module) -> bool:
    """Whether this loaded checkpoint declares and provides the embedding."""
    return (
        bool(getattr(transformer.config, "use_keyframes_abs_pos_embedding", False))
        and getattr(transformer, "keyframes_abs_pos_embedding", None) is not None
    )


# ---------------------------------------------------------------------------
# Native-dtype RoPE attention processor (matching ltx_core reference precision)
# ---------------------------------------------------------------------------


def _apply_interleaved_rotary_emb_native(x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Interleaved RoPE in native dtype (no float32 upcast)."""
    cos, sin = freqs
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return x * cos + x_rotated * sin


def _apply_split_rotary_emb_native(x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Split RoPE in native dtype (no float32 upcast)."""
    cos, sin = freqs
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        b, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True
    last = x.shape[-1]
    r = last // 2
    split_x = x.reshape(*x.shape[:-1], 2, r)
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]
    output = split_x * cos.unsqueeze(-2)
    output[..., :1, :].addcmul_(-sin.unsqueeze(-2), second_x)
    output[..., 1:, :].addcmul_(sin.unsqueeze(-2), first_x)
    output = output.reshape(*output.shape[:-2], last)
    if needs_reshape:
        output = output.swapaxes(1, 2).reshape(b, t, -1)
    return output


class NativeDtypeAttnProcessor:
    """Attention processor matching ltx_core ``Attention.forward``.

    RoPE runs in the activation dtype without an FP32 upcast, as in ltx_core. ``all_perturbed`` returns V
    (STG passthrough); without it, attention is unperturbed, so one processor serves every module.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        gate_logits = None
        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(hidden_states)

        value = attn.to_v(encoder_hidden_states)

        if all_perturbed:
            hidden_states = value
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            if query_rotary_emb is not None:
                k_emb = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                if attn.rope_type == "interleaved":
                    query = _apply_interleaved_rotary_emb_native(query, query_rotary_emb)
                    key = _apply_interleaved_rotary_emb_native(key, k_emb)
                elif attn.rope_type == "split":
                    query = _apply_split_rotary_emb_native(query, query_rotary_emb)
                    key = _apply_split_rotary_emb_native(key, k_emb)

            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)

        if gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            gates = 2.0 * torch.sigmoid(gate_logits)
            hidden_states = hidden_states * gates.unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Block forward — monkey-patched onto each LTX2VideoTransformerBlock
# ---------------------------------------------------------------------------


def _prepare_attention_mask(mask: torch.Tensor, x_dtype: torch.dtype) -> torch.Tensor:
    """Convert an int64 ``{0, 1}`` mask ``[B, T]`` to ltx_core's additive bias ``[B, 1, T]``.

    ``(mask - 1).to(x_dtype) * finfo.max``: valid tokens get ``0``, padding ``-finfo.max``.
    """
    assert_same(dtype=torch.int64, attention_mask=mask)
    return ((mask - 1).to(x_dtype) * torch.finfo(x_dtype).max).unsqueeze(1)


def _prepare_timestep(
    timestep: torch.Tensor, adaln: nn.Module, hidden_states: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ltx_core ``_prepare_timestep`` for ``hidden_states`` ``[B, N, D]``, on a timestep already in the embedding domain.

    Embeds at ``hidden_states``' dtype and returns the AdaLN modulation and the embedded timestep as ``[B, T, D]``,
    where ``T`` is 1 for a global timestep and ``N`` for a per-token one.
    """
    batch_size, num_tokens = hidden_states.shape[:2]
    modulation, embedded_timestep = adaln(timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype)
    modulation = modulation.view(batch_size, -1, modulation.shape[-1])
    assert modulation.shape[1] in (1, num_tokens), f"expected 1 or {num_tokens} timesteps, got {modulation.shape[1]}"
    return modulation, embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])


def _get_ada_values(
    scale_shift_table: torch.Tensor,
    batch_size: int,
    temb: torch.Tensor,
    indices: slice = slice(None),
) -> Tuple[torch.Tensor, ...]:
    """ltx_core ``get_ada_values``: the AdaLN rows ``indices`` as a tuple of ``[B, T, D]`` tensors.

    Unlike Diffusers ``get_mod_params``, the table is cast to ``temb``'s dtype before the add, as in ltx_core.
    """
    num_ada_params = scale_shift_table.shape[0]
    ada_values = (
        scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=temb.device, dtype=temb.dtype)
        + temb.reshape(batch_size, temb.shape[1], num_ada_params, -1)[:, :, indices, :]
    )
    return ada_values.unbind(dim=2)


def _text_cross_attn(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    norm_module,
    attn_module,
    temb: torch.Tensor,
    temb_prompt: Optional[torch.Tensor],
    encoder_attention_mask: Optional[torch.Tensor],
    is_video: bool,
) -> torch.Tensor:
    """Text cross-attention with the optional LTX-2.3+ query and prompt AdaLN (``cross_attn_adaln``)."""
    batch_size = hidden_states.size(0)
    cross_attn_adaln = self.video_cross_attn_adaln if is_video else self.audio_cross_attn_adaln
    sst = self.scale_shift_table if is_video else self.audio_scale_shift_table

    shift_q = scale_q = gate_q = None
    if cross_attn_adaln:
        shift_q, scale_q, gate_q = _get_ada_values(sst, batch_size, temb, slice(6, 9))

    if self.cross_attn_adaln:
        psst = self.prompt_scale_shift_table if is_video else self.audio_prompt_scale_shift_table
        if temb_prompt is not None:
            shift_kv, scale_kv = _get_ada_values(psst, batch_size, temb_prompt)
        else:
            # `use_prompt_adaln_single=False` (LTX-2.5): no `prompt_adaln` MLP, so the prompt-side
            # scale/shift is timestep-independent and comes from the static table. [2,D] -> 2x [1,1,D].
            shift_kv, scale_kv = (
                psst[None, None].to(device=hidden_states.device, dtype=hidden_states.dtype).unbind(dim=2)
            )
        encoder_hidden_states = encoder_hidden_states * (1 + scale_kv) + shift_kv

    norm_hs = norm_module(hidden_states)
    if cross_attn_adaln:
        norm_hs = norm_hs * (1 + scale_q) + shift_q

    out = attn_module(
        norm_hs,
        encoder_hidden_states=encoder_hidden_states,
        query_rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    if cross_attn_adaln:
        out = out * gate_q
    return hidden_states + out


def _block_forward_audio_optional(
    self,
    hidden_states: Optional[torch.Tensor] = None,
    audio_hidden_states: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    audio_encoder_hidden_states: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    temb_audio: Optional[torch.Tensor] = None,
    temb_ca_scale_shift: Optional[torch.Tensor] = None,
    temb_ca_audio_scale_shift: Optional[torch.Tensor] = None,
    temb_ca_gate: Optional[torch.Tensor] = None,
    temb_ca_audio_gate: Optional[torch.Tensor] = None,
    temb_prompt: Optional[torch.Tensor] = None,
    temb_prompt_audio: Optional[torch.Tensor] = None,
    video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ca_video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ca_audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    audio_encoder_attention_mask: Optional[torch.Tensor] = None,
    a2v_cross_attention_mask: Optional[torch.Tensor] = None,
    v2a_cross_attention_mask: Optional[torch.Tensor] = None,
    skip_video_self_attn: bool = False,
    skip_audio_self_attn: bool = False,
    skip_cross_modal_attn: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """ltx_core ``BasicAVTransformerBlock.forward`` with either modality optional.

    The Diffusers block reads the audio tensors unconditionally and fails when they are ``None``.
    """
    run_vx = hidden_states is not None
    run_ax = audio_hidden_states is not None
    run_a2v = run_vx and run_ax and not skip_cross_modal_attn
    batch_size = (hidden_states if run_vx else audio_hidden_states).size(0)

    # ---- 1. Video self-attention ----
    if run_vx:
        vshift_msa, vscale_msa, vgate_msa = _get_ada_values(self.scale_shift_table, batch_size, temb, slice(0, 3))
        vshift_mlp, vscale_mlp, vgate_mlp = _get_ada_values(self.scale_shift_table, batch_size, temb, slice(3, 6))

        norm_vx = self.norm1(hidden_states) * (1 + vscale_msa) + vshift_msa
        vx_attn = self.attn1(
            hidden_states=norm_vx,
            encoder_hidden_states=None,
            query_rotary_emb=video_rotary_emb,
            all_perturbed=skip_video_self_attn,
        )
        hidden_states = hidden_states + vx_attn * vgate_msa

    # ---- 1b. Audio self-attention ----
    if run_ax:
        ashift_msa, ascale_msa, agate_msa = _get_ada_values(
            self.audio_scale_shift_table, batch_size, temb_audio, slice(0, 3)
        )
        ashift_mlp, ascale_mlp, agate_mlp = _get_ada_values(
            self.audio_scale_shift_table, batch_size, temb_audio, slice(3, 6)
        )

        norm_ax = self.audio_norm1(audio_hidden_states) * (1 + ascale_msa) + ashift_msa
        ax_attn = self.audio_attn1(
            hidden_states=norm_ax,
            encoder_hidden_states=None,
            query_rotary_emb=audio_rotary_emb,
            all_perturbed=skip_audio_self_attn,
        )
        audio_hidden_states = audio_hidden_states + ax_attn * agate_msa

    # ---- 2. Video text cross-attention ----
    if run_vx:
        hidden_states = _text_cross_attn(
            self,
            hidden_states,
            encoder_hidden_states,
            self.norm2,
            self.attn2,
            temb,
            temb_prompt,
            encoder_attention_mask,
            is_video=True,
        )

    # ---- 2b. Audio text cross-attention ----
    if run_ax:
        audio_hidden_states = _text_cross_attn(
            self,
            audio_hidden_states,
            audio_encoder_hidden_states,
            self.audio_norm2,
            self.audio_attn2,
            temb_audio,
            temb_prompt_audio,
            audio_encoder_attention_mask,
            is_video=False,
        )

    # ---- 3. Audio-video cross-attention (a2v + v2a) ----
    if run_a2v:
        norm_vx_ca = self.audio_to_video_norm(hidden_states)
        norm_ax_ca = self.video_to_audio_norm(audio_hidden_states)

        v_ca_ss = self.video_a2v_cross_attn_scale_shift_table
        a_ca_ss = self.audio_a2v_cross_attn_scale_shift_table

        # Video modulation
        v_a2v_scale, v_a2v_shift = _get_ada_values(v_ca_ss[:4, :], batch_size, temb_ca_scale_shift, slice(0, 2))
        v_v2a_scale, v_v2a_shift = _get_ada_values(v_ca_ss[:4, :], batch_size, temb_ca_scale_shift, slice(2, 4))
        (a2v_gate,) = _get_ada_values(v_ca_ss[4:, :], batch_size, temb_ca_gate)
        a2v_gate = a2v_gate.squeeze(2)

        # Audio modulation
        a_a2v_scale, a_a2v_shift = _get_ada_values(a_ca_ss[:4, :], batch_size, temb_ca_audio_scale_shift, slice(0, 2))
        a_v2a_scale, a_v2a_shift = _get_ada_values(a_ca_ss[:4, :], batch_size, temb_ca_audio_scale_shift, slice(2, 4))
        (v2a_gate,) = _get_ada_values(a_ca_ss[4:, :], batch_size, temb_ca_audio_gate)
        v2a_gate = v2a_gate.squeeze(2)

        # a2v: Q=Video, K/V=Audio
        mod_vx = norm_vx_ca * (1 + v_a2v_scale.squeeze(2)) + v_a2v_shift.squeeze(2)
        mod_ax = norm_ax_ca * (1 + a_a2v_scale.squeeze(2)) + a_a2v_shift.squeeze(2)
        hidden_states = hidden_states + a2v_gate * self.audio_to_video_attn(
            mod_vx,
            encoder_hidden_states=mod_ax,
            query_rotary_emb=ca_video_rotary_emb,
            key_rotary_emb=ca_audio_rotary_emb,
            attention_mask=a2v_cross_attention_mask,
        )

        # v2a: Q=Audio, K/V=Video
        mod_vx = norm_vx_ca * (1 + v_v2a_scale.squeeze(2)) + v_v2a_shift.squeeze(2)
        mod_ax = norm_ax_ca * (1 + a_v2a_scale.squeeze(2)) + a_v2a_shift.squeeze(2)
        audio_hidden_states = audio_hidden_states + v2a_gate * self.video_to_audio_attn(
            mod_ax,
            encoder_hidden_states=mod_vx,
            query_rotary_emb=ca_audio_rotary_emb,
            key_rotary_emb=ca_video_rotary_emb,
            attention_mask=v2a_cross_attention_mask,
        )

    # ---- 4. Video FFN ----
    if run_vx:
        norm_vx = self.norm3(hidden_states) * (1 + vscale_mlp) + vshift_mlp
        hidden_states = hidden_states + self.ff(norm_vx) * vgate_mlp

    # ---- 4b. Audio FFN ----
    if run_ax:
        norm_ax = self.audio_norm3(audio_hidden_states) * (1 + ascale_mlp) + ashift_mlp
        audio_hidden_states = audio_hidden_states + self.audio_ff(norm_ax) * agate_mlp

    return hidden_states, audio_hidden_states


# ---------------------------------------------------------------------------
# classify_forward — monkey-patched onto self.transformer
# ---------------------------------------------------------------------------


def classify_forward(
    self,
    hidden_states: Optional[torch.Tensor] = None,
    audio_hidden_states: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    audio_encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: Optional[torch.Tensor] = None,
    audio_timestep: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    audio_sigma: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    audio_encoder_attention_mask: Optional[torch.Tensor] = None,
    num_frames: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    fps: float = 24.0,
    audio_num_frames: Optional[int] = None,
    video_coords: Optional[torch.Tensor] = None,
    audio_coords: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[dict] = None,
    return_dict: bool = True,  # accepted for API compatibility; always ignored
    # FastGen distillation kwargs
    return_features_early: bool = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: bool = False,
    skip_layers: Optional[List[int]] = None,
    # Perturbation flags for STG / modality guidance
    skip_self_attn_layers: Optional[List[int]] = None,
    skip_cross_modal_attn: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[List]]:
    """Patched ``LTX2VideoTransformer3DModel.forward`` for FastGen.

    Adds feature extraction, early exit, logvar and STG/modality perturbations. As in ltx_core, a ``None``
    ``hidden_states`` or ``audio_hidden_states`` skips that modality's RoPE, timestep embeddings,
    cross-modal modulation and output head.

    Perturbations:
        skip_self_attn_layers: Layers whose video and audio self-attention return V (STG).
        skip_cross_modal_attn: Skip a2v/v2a cross-attention in every block (modality guidance).

    Returns:
        Always ``(video_output, audio_output, logvar, features)``, with ``None`` for entries that do not
        apply; early exit sets only ``features``.
    """
    lora_scale = 1.0 if attention_kwargs is None else attention_kwargs.get("scale", 1.0)

    with scaled_lora_layers(self, lora_scale):
        run_video = hidden_states is not None
        run_audio = audio_hidden_states is not None
        ltx2_version = _validate_ltx2_version(getattr(self, "_fastgen_ltx2_version", None))

        # ------------------------------------------------------------------ #
        # Steps 1-5: mirroring the original LTX-2 None-guarded pattern
        # ------------------------------------------------------------------ #
        if run_audio:
            audio_timestep = audio_timestep if audio_timestep is not None else timestep

        batch_size = (hidden_states if run_video else audio_hidden_states).size(0)

        # 1. Patchify input projections. Under autocast these outputs, rather
        # than the wrapper's input dtype, define the downstream activation dtype.
        if run_video:
            hidden_states = self.proj_in(hidden_states)
            # Current ltx_core gates this on checkpoint capability, not a model-version
            # branch. Older 2.3 checkpoints do not provide/enable the embedding, so this
            # remains a no-op for them while matching the supported current pipeline.
            if _ltx2_first_frame_embedding_enabled(self):
                hidden_states = _apply_ltx2_first_frame_embedding(
                    hidden_states,
                    self.keyframes_abs_pos_embedding,
                    height,
                    width,
                )
        if run_audio:
            audio_hidden_states = self.audio_proj_in(audio_hidden_states)

        if run_video and run_audio:
            assert_same(dtype=True, device=True, hidden_states=hidden_states, audio_hidden_states=audio_hidden_states)

        # 2. Convert attention masks after projection so additive masks use the
        # actual autocast-selected activation dtype.
        if run_video and encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = _prepare_attention_mask(encoder_attention_mask, hidden_states.dtype)

        if run_audio and audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = _prepare_attention_mask(
                audio_encoder_attention_mask, audio_hidden_states.dtype
            )

        # 3. RoPE positional embeddings use the same runtime activation dtype.
        video_rotary_emb = None
        video_cross_attn_rotary_emb = None
        if run_video:
            if video_coords is None:
                video_coords = self.rope.prepare_video_coords(
                    batch_size, num_frames, height, width, hidden_states.device, fps=fps
                )
            video_dtype = hidden_states.dtype
            # Current upstream 2.3/2.5 keeps positions in FP32. The unsupported
            # 2.0 compatibility path preserves its release-era latent dtype.
            video_coords = video_coords.to(video_dtype if ltx2_version == "2.0" else torch.float32)
            video_rotary_emb = to(self.rope(video_coords, device=hidden_states.device), dtype=video_dtype)
            video_cross_attn_rotary_emb = to(
                self.cross_attn_rope(video_coords[:, 0:1, :], device=hidden_states.device), dtype=video_dtype
            )

        if run_audio:
            if audio_coords is None:
                audio_coords = self.audio_rope.prepare_audio_coords(
                    batch_size, audio_num_frames, audio_hidden_states.device
                )
            audio_dtype = audio_hidden_states.dtype
            audio_rotary_emb = to(self.audio_rope(audio_coords, device=audio_hidden_states.device), dtype=audio_dtype)
            audio_cross_attn_rotary_emb = to(
                self.cross_attn_audio_rope(audio_coords[:, 0:1, :], device=audio_hidden_states.device),
                dtype=audio_dtype,
            )
            # Ref: audio positions stay float32; cos/sin cast to out_dtype=latent dtype
            assert_same(dtype=torch.float32, device=audio_hidden_states.device, audio_coords=audio_coords)
        else:
            audio_rotary_emb = None
            audio_cross_attn_rotary_emb = None

        # 4. Timestep embeddings and modulation parameters
        timestep_cross_attn_gate_scale_factor = (
            self.config.cross_attn_timestep_scale_multiplier / self.config.timestep_scale_multiplier
        )

        temb = None
        embedded_timestep = None
        # T2V: timesteps is [B] → temb is [B, 1, D] (broadcast across tokens in the block)
        # I2V: timesteps is [B, N] → temb is [B, N, D] (per-token modulation)
        if run_video:
            temb, embedded_timestep = _prepare_timestep(timestep, self.time_embed, hidden_states)
            assert_same(
                dtype=True,
                device=True,
                hidden_states=hidden_states,
                temb=temb,
                embedded_timestep=embedded_timestep,
            )

        temb_audio = None
        audio_embedded_timestep = None
        if run_audio:
            temb_audio, audio_embedded_timestep = _prepare_timestep(
                audio_timestep, self.audio_time_embed, audio_hidden_states
            )
            assert_same(
                dtype=True,
                device=True,
                audio_hidden_states=audio_hidden_states,
                temb_audio=temb_audio,
                audio_embedded_timestep=audio_embedded_timestep,
            )

        sigma_v = sigma if sigma is not None else timestep
        audio_sigma_v = audio_sigma if audio_sigma is not None else sigma_v
        # Ref: modality.sigma is global [B], not per-token.
        for name, value in (("sigma", sigma_v), ("audio_sigma", audio_sigma_v)):
            assert value.shape[1:] in ((), (1,)), f"{name} must be global [B] or scalar, got shape {value.shape}"

        temb_prompt = None
        temb_prompt_audio = None
        if run_video and self.prompt_modulation and hasattr(self, "prompt_adaln"):
            temb_prompt, _ = _prepare_timestep(sigma_v, self.prompt_adaln, hidden_states)
            assert_same(dtype=True, device=True, hidden_states=hidden_states, temb_prompt=temb_prompt)
        if run_audio and self.prompt_modulation and hasattr(self, "audio_prompt_adaln"):
            temb_prompt_audio, _ = _prepare_timestep(audio_sigma_v, self.audio_prompt_adaln, audio_hidden_states)
            assert_same(
                dtype=True,
                device=True,
                audio_hidden_states=audio_hidden_states,
                temb_prompt_audio=temb_prompt_audio,
            )

        video_cross_attn_scale_shift = None
        video_cross_attn_a2v_gate = None
        audio_cross_attn_scale_shift = None
        audio_cross_attn_v2a_gate = None
        if run_video and run_audio:
            (
                video_ca_scale_shift_t,
                video_ca_gate_t,
                audio_ca_scale_shift_t,
                audio_ca_gate_t,
            ) = _ltx2_cross_modal_timestep_inputs(
                ltx2_version,
                video_timestep=timestep,
                audio_timestep=audio_timestep,
                video_sigma=sigma_v,
                audio_sigma=audio_sigma_v,
                gate_scale_factor=timestep_cross_attn_gate_scale_factor,
            )

            video_cross_attn_scale_shift, _ = _prepare_timestep(
                video_ca_scale_shift_t, self.av_cross_attn_video_scale_shift, hidden_states
            )
            video_cross_attn_a2v_gate, _ = _prepare_timestep(
                video_ca_gate_t, self.av_cross_attn_video_a2v_gate, hidden_states
            )
            audio_cross_attn_scale_shift, _ = _prepare_timestep(
                audio_ca_scale_shift_t, self.av_cross_attn_audio_scale_shift, audio_hidden_states
            )
            audio_cross_attn_v2a_gate, _ = _prepare_timestep(
                audio_ca_gate_t, self.av_cross_attn_audio_v2a_gate, audio_hidden_states
            )
            assert_same(
                dtype=True,
                device=True,
                hidden_states={"video": hidden_states, "audio": audio_hidden_states},
                cross_attn_scale_shift={
                    "video": video_cross_attn_scale_shift,
                    "audio": audio_cross_attn_scale_shift,
                },
                cross_attn_gate={
                    "video": video_cross_attn_a2v_gate,
                    "audio": audio_cross_attn_v2a_gate,
                },
            )

        # 5. Prompt embeddings (LTX-2.0: caption_projection in DiT; LTX-2.3: per-modality projections in connectors only)
        # Ref: context shape = [B, T_text, inner_dim]
        if run_video:
            if self.config.use_prompt_embeddings:
                encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            assert (
                encoder_hidden_states.shape[-1] == hidden_states.shape[-1]
            ), f"video context dim {encoder_hidden_states.shape[-1]} != inner_dim {hidden_states.shape[-1]}"
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))
        if run_audio:
            if self.config.use_prompt_embeddings:
                audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
            assert (
                audio_encoder_hidden_states.shape[-1] == audio_hidden_states.shape[-1]
            ), f"audio context dim {audio_encoder_hidden_states.shape[-1]} != audio_inner_dim {audio_hidden_states.shape[-1]}"
            audio_encoder_hidden_states = audio_encoder_hidden_states.view(batch_size, -1, audio_hidden_states.size(-1))

        # ------------------------------------------------------------------ #
        # Step 6: Block loop with active-modality feature extraction
        # ------------------------------------------------------------------ #
        features: List[torch.Tensor] = []

        skip_sa_layers = set(skip_self_attn_layers) if skip_self_attn_layers else set()

        for idx, block in enumerate(self.transformer_blocks):
            # A skipped block acts as the identity; still record its feature so each requested index yields one.
            if skip_layers is None or idx not in skip_layers:
                block_kwargs = dict(
                    hidden_states=hidden_states,
                    audio_hidden_states=audio_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    temb=temb,
                    temb_audio=temb_audio,
                    temb_ca_scale_shift=video_cross_attn_scale_shift,
                    temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                    temb_ca_gate=video_cross_attn_a2v_gate,
                    temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                    temb_prompt=temb_prompt,
                    temb_prompt_audio=temb_prompt_audio,
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=video_cross_attn_rotary_emb,
                    ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                    skip_video_self_attn=idx in skip_sa_layers,
                    skip_audio_self_attn=idx in skip_sa_layers,
                    skip_cross_modal_attn=skip_cross_modal_attn,
                )

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    # Pass tensor inputs positionally so torch.utils.checkpoint can detect
                    # requires_grad and properly track activations across the checkpoint
                    # boundary; bind only the non-tensor flags via the closure.
                    flag_kwargs = {
                        "skip_video_self_attn": block_kwargs.pop("skip_video_self_attn"),
                        "skip_audio_self_attn": block_kwargs.pop("skip_audio_self_attn"),
                        "skip_cross_modal_attn": block_kwargs.pop("skip_cross_modal_attn"),
                    }
                    tensor_keys = tuple(block_kwargs.keys())

                    def _ckpt_block(*tensor_args, _block=block, _keys=tensor_keys, _flags=flag_kwargs):
                        return _block(**dict(zip(_keys, tensor_args)), **_flags)

                    hidden_states, audio_hidden_states = self._gradient_checkpointing_func(
                        _ckpt_block, *block_kwargs.values()
                    )
                else:
                    hidden_states, audio_hidden_states = block(**block_kwargs)

            if feature_indices is not None and idx in feature_indices:
                if hidden_states is not None and audio_hidden_states is not None:
                    features.append({"video": hidden_states.clone(), "audio": audio_hidden_states.clone()})
                elif hidden_states is not None:
                    features.append(hidden_states.clone())
                elif audio_hidden_states is not None:
                    features.append(audio_hidden_states.clone())

            if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
                return None, None, None, features

        # ------------------------------------------------------------------ #
        # Step 7: Output layers
        # ------------------------------------------------------------------ #
        video_output = None
        if run_video:
            scale_shift_values = (
                self.scale_shift_table[None, None].to(device=hidden_states.device, dtype=hidden_states.dtype)
                + embedded_timestep[:, :, None]
            )
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            video_output = self.proj_out(hidden_states)

        audio_output = None
        if run_audio:
            audio_scale_shift_values = (
                self.audio_scale_shift_table[None, None].to(
                    device=audio_hidden_states.device, dtype=audio_hidden_states.dtype
                )
                + audio_embedded_timestep[:, :, None]
            )
            audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]
            audio_hidden_states = self.audio_norm_out(audio_hidden_states)
            audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
            audio_output = self.audio_proj_out(audio_hidden_states)

        # ------------------------------------------------------------------ #
        # Assemble output
        # ------------------------------------------------------------------ #
        if return_features_early:
            assert feature_indices is not None
            assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
            return None, None, None, features

        logvar = None
        if return_logvar:
            assert hasattr(
                self, "logvar_linear"
            ), "logvar_linear is required when return_logvar=True. Set enable_logvar_linear=True in model config."
            # ``logvar_linear`` is sized to the video ``inner_dim`` while
            # ``audio_logvar_linear`` is sized to the audio ``inner_dim`` (they
            # differ for LTX-2: 4096 vs 2048).  When both modalities run we use
            # the video timestep; in audio-only forwards we must use the audio
            # logvar head to match the audio embedded-timestep feature width.
            if embedded_timestep is not None:
                logvar = self.logvar_linear(embedded_timestep.mean(dim=1))
            else:
                assert hasattr(self, "audio_logvar_linear"), (
                    "audio_logvar_linear is required for audio-only forward with "
                    "return_logvar=True. It is added when enable_logvar_linear=True "
                    "and 'audio' is in modalities."
                )
                logvar = self.audio_logvar_linear(audio_embedded_timestep.mean(dim=1))

        return video_output, audio_output, logvar, features if feature_indices is not None else None


# ---------------------------------------------------------------------------
# Text encoder wrapper
# ---------------------------------------------------------------------------


class LTX2TextEncoder(FastGenModule):
    """The checkpoint's Gemma text encoder (Gemma 3 for LTX-2.0/2.3, Gemma 4 for LTX-2.5).

    Returns the stacked hidden states together with the tokenizer attention mask ``LTX2TextConnectors`` needs.
    """

    def __init__(
        self,
        model_id_or_local_path: str,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        ltx2_version: Optional[str] = None,
    ):
        """
        Args:
            model_id_or_local_path: HuggingFace model ID or local path.
            dtype: Gemma weight dtype. Defaults to BF16 like the LTX-2 reference; the HF checkpoint is mostly
                FP32, and running it in FP32 changes the hidden states and degrades generation.
            ltx2_version: Release selected by the parent network; ``None`` maps canonical model IDs.
        """
        super().__init__()
        # LTX-2.0/2.3 ship Gemma 3, LTX-2.5 ships Gemma 4; the version determines the class.
        resolved_version = _resolve_ltx2_version(model_id_or_local_path, ltx2_version)
        enc_cls = Gemma4UnifiedForConditionalGeneration if resolved_version == "2.5" else Gemma3ForConditionalGeneration
        enc_name = enc_cls.__name__
        self.tokenizer = hf_load(GemmaTokenizerFast.from_pretrained, model_id_or_local_path, subfolder="tokenizer")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if is_rank0():
            logger.debug(f"Loading {enc_name} from pretrained...")
            self.text_encoder = hf_load(
                enc_cls.from_pretrained, model_id_or_local_path, subfolder="text_encoder", dtype=dtype
            )
        else:
            logger.debug(f"Meta-init {enc_name} (weights will be broadcasted from rank 0)...")
            config = hf_load(enc_cls.config_class.from_pretrained, model_id_or_local_path, subfolder="text_encoder")
            with torch.device("meta"):
                self.text_encoder = enc_cls._from_config(config, dtype=dtype)

    def encode(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt(s) into raw stacked Gemma hidden states; ``LTX2TextConnectors`` normalizes them.

        Returns:
            ``prompt_embeds`` ``[B, T, H * (num_layers + 1)]`` (embedding output and every layer) and the
            binary ``attention_mask`` ``[B, T]`` (1 = token, 0 = padding).
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        prompt = [p.strip() for p in prompt]

        # Every LTX-2 prompt must start with <bos>. ``add_special_tokens=True`` is not enough:
        # Gemma 3 emits BOS from its post-processor, Gemma 4 (LTX-2.5) does not, so relying on the
        # flag silently drops it and shifts every token one position. The reference prepends it
        # unconditionally (``LTXGemmaTokenizer.tokenize_with_weights``).
        encoded = self.tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=max_sequence_length,
            add_special_tokens=True,
        )
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("LTX-2 requires a tokenizer with a bos_token_id; got None.")
        ids = [
            seq if seq and seq[0] == bos_id else [bos_id, *seq][:max_sequence_length] for seq in encoded["input_ids"]
        ]
        text_inputs = self.tokenizer.pad(
            {"input_ids": ids},
            padding="max_length",
            max_length=max_sequence_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return self(
            text_inputs.input_ids.to(self.device),
            text_inputs.attention_mask.to(self.device),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,  # only hidden_states is used; skips the full-vocabulary lm_head projection
        )

        # Connectors normalize the raw per-layer states after restoring this shape.
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        return hidden_states.flatten(2, 3), attention_mask


# ---------------------------------------------------------------------------
# Unified VAE (video-only, audio-only, or joint; dispatches on multimodal dict)
# ---------------------------------------------------------------------------


class MelSpectrogramFP32(FastGenModule, MelSpectrogram):
    """Keep spectrogram state and inputs in FP32."""

    @property
    def precision_model(self) -> torch.dtype:
        return torch.float32

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(dtype=self.precision_model)
        return super().forward(waveform)


class LTX2VocoderWithBWEFP32(FastGenModule, LTX2VocoderWithBWE):
    """Keep BWE vocoder state and inputs in FP32."""

    @property
    def precision_model(self) -> torch.dtype:
        return torch.float32

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        mel_spec = mel_spec.to(dtype=self.precision_model)
        return super().forward(mel_spec)


class AutoencoderKLLTX2AudioFP32(FastGenModule, AutoencoderKLLTX2Audio):
    """Keep the audio VAE state and inputs in FP32; the reference encodes training audio in FP32 for quality."""

    @property
    def precision_model(self) -> torch.dtype:
        return torch.float32

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        return super().encode(x.to(dtype=self.precision_model), return_dict=return_dict)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        return super().decode(z.to(dtype=self.precision_model), return_dict=return_dict)


class LTX2VAE(FastGenModule):
    """Video/audio VAE wrapper; audio decode includes the vocoder.

    As in ltx_core, which casts the statistics with ``.to(x)`` and does not autocast its VAEs, latent
    (de)normalization runs in the VAE-side latent dtype: the posterior's on encode, that of the latent staged
    in ``precision_model`` on decode. The audio VAE, mel transform and BWE vocoder are pinned to FP32.
    """

    def __init__(
        self,
        model_id_or_local_path: str,
        enable_tiling: bool = False,
        modalities: Tuple[str, ...] = ("video", "audio"),
    ):
        super().__init__()

        self.modalities = _validate_modalities(modalities)

        self._video_vae: Optional[AutoencoderKLLTX2Video] = None
        if "video" in self.modalities:
            self._video_vae = hf_load(AutoencoderKLLTX2Video.from_pretrained, model_id_or_local_path, subfolder="vae")
            if enable_tiling:
                self._video_vae.enable_tiling()

        self._audio_vae: Optional[AutoencoderKLLTX2Audio] = None
        self._vocoder: Optional[Union[LTX2Vocoder, LTX2VocoderWithBWE]] = None
        self._mel_transform: Optional[MelSpectrogram] = None

        if "audio" in self.modalities:
            # FP32 like the reference's training-data audio encode (see ``AutoencoderKLLTX2AudioFP32``).
            self._audio_vae = hf_load(
                AutoencoderKLLTX2AudioFP32.from_pretrained, model_id_or_local_path, subfolder="audio_vae"
            )

            # LTX-2.3/2.5 use LTX2VocoderWithBWE (48 kHz output with bandwidth extension);
            # LTX-2.0 uses LTX2Vocoder (24 kHz). Detect from the config on disk.
            vocoder_config = hf_load(LTX2Vocoder.load_config, model_id_or_local_path, subfolder="vocoder")
            vocoder_cls = LTX2VocoderWithBWEFP32 if "bwe_in_channels" in vocoder_config else LTX2Vocoder
            self._vocoder = hf_load(vocoder_cls.from_pretrained, model_id_or_local_path, subfolder="vocoder")

            # Mel feature extraction is signal processing: a nested FastGen module fixed to fp32.
            self._mel_transform = MelSpectrogramFP32(
                sample_rate=16000,
                n_fft=1024,
                win_length=1024,
                hop_length=160,
                f_min=0.0,
                f_max=8000.0,
                n_mels=64,
                window_fn=torch.hann_window,
                center=True,
                pad_mode="reflect",
                power=1.0,
                mel_scale="slaney",
                norm="slaney",
            )

    @property
    def latent_compression_factors(self) -> tuple[int, int, int]:
        """Pixel-to-latent ``(T, H, W)`` factors of the video VAE; ``(8, 32, 32)`` when unavailable."""
        if self._video_vae is None:
            return (8, 32, 32)
        return (
            getattr(self._video_vae, "temporal_compression_ratio", 8),
            getattr(self._video_vae, "spatial_compression_ratio", 32),
            getattr(self._video_vae, "spatial_compression_ratio", 32),
        )

    def forward(
        self,
        data,
        operation: str,
        mode: str = "argmax",
    ):
        if operation == "encode_video":
            return self._encode_video(data, mode=mode)
        if operation == "encode":
            fn_by_modality = {
                "video": lambda x: self._encode_video(x, mode),
                "audio": lambda x: self._encode_audio(x, mode),
            }
        elif operation == "decode":
            fn_by_modality = {"video": self._decode_video, "audio": self._decode_audio}
        else:
            raise ValueError(f"Unsupported preprocessor operation: {operation!r}")
        _validate_input(data, self.modalities, operation, "VAE")
        if isinstance(data, torch.Tensor):
            return fn_by_modality[self.modalities[0]](data)
        return {modality: fn(data[modality]) for modality, fn in fn_by_modality.items() if modality in data}

    # -- encode -----------------------------------------------------------

    def encode(self, data, mode: str = "argmax"):
        """Encode video pixels and/or audio waveforms into normalized latents.

        Like ``LTX2.forward``, a Tensor requires exactly one configured modality; a dict may hold any
        non-empty subset of them and returns the same keys. ``mode`` is ``"argmax"`` (the posterior mean,
        which the reference trainer caches and its pipelines condition on) or ``"sample"``.
        """
        return self(data, operation="encode", mode=mode)

    def encode_video(self, video: torch.Tensor, mode: str = "argmax") -> torch.Tensor:
        return self(video, mode=mode, operation="encode_video")

    def _encode_video(self, video: torch.Tensor, mode: str = "argmax") -> torch.Tensor:
        assert mode in {"sample", "argmax"}, f"Invalid mode: {mode}. Supported modes: ['sample', 'argmax']"
        assert self._video_vae is not None, "video VAE not loaded (missing 'video' in modalities?)"
        model_video = video.to(dtype=self.precision_model)
        dist = self._video_vae.encode(model_video, return_dict=False)[0]
        latents = dist.sample() if mode == "sample" else dist.mode()
        mean = self._video_vae.latents_mean.view(1, -1, 1, 1, 1).to(dtype=latents.dtype)
        std = self._video_vae.latents_std.view(1, -1, 1, 1, 1).to(dtype=latents.dtype)
        return (latents - mean) * self._video_vae.config.scaling_factor / std

    def _encode_audio(self, audio: torch.Tensor, mode: str = "argmax") -> torch.Tensor:
        assert mode in {"sample", "argmax"}, f"Invalid mode: {mode}. Supported modes: ['sample', 'argmax']"
        assert (
            self._audio_vae is not None and self._mel_transform is not None
        ), "audio VAE not loaded (missing 'audio' in modalities?)"
        waveform_samples = audio.shape[-1]
        # audio: [batch, 2, samples] stereo waveform → log-mel [batch, 2, time, mel_bins]
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        mel = torch.log(torch.clamp(self._mel_transform(audio), min=1e-5))  # fp32 log-mel features
        mel = mel.permute(0, 1, 3, 2).contiguous()  # [batch, 2, time, mel_bins]
        if mel.shape[1] == 1:
            mel = mel.expand(-1, 2, -1, -1)
        # The FP32 audio VAE takes the FP32 log-mel as is (reference training-data policy).
        dist = self._audio_vae.encode(mel, return_dict=False)[0]
        latents = dist.sample() if mode == "sample" else dist.mode()

        # ``LTX2._compute_audio_shape`` sizes audio from the video shape with this
        # formula, so a VAE that disagrees with it fails loudly rather than let
        # ``model.input_shape`` and the encoded latents drift apart. ``LTX2.encode_real``
        # checks the waveform length against the video frames at ``LTX2.fps``.
        expected_frames = _audio_latent_frames(
            waveform_samples,
            hop_length=self._audio_vae.config.mel_hop_length,
            temporal_compression=2 ** (len(self._audio_vae.config.ch_mult) - 1),
        )
        encoded_frames = latents.shape[2]
        if encoded_frames != expected_frames:
            raise ValueError(
                "LTX-2 audio VAE produced an unexpected latent duration: "
                f"got {encoded_frames} frames for {waveform_samples} waveform samples, "
                f"expected {expected_frames}."
            )

        # Per-(C*F) channel stats: ``latents_mean``/``latents_std`` (each ``[C*F]``) follow the ``C``-major order of
        # the packed ``[B, T, C*F]`` latents, so they broadcast over ``[B, C, T, F]`` as ``(1, C, 1, F)``.
        _, C, _, F = latents.shape
        mean = self._audio_vae.latents_mean.to(dtype=latents.dtype).view(1, C, 1, F)
        std = self._audio_vae.latents_std.to(dtype=latents.dtype).view(1, C, 1, F)
        return (latents - mean) / std

    # -- decode -----------------------------------------------------------

    def decode(self, data):
        """Decode latents to video pixels and/or audio waveforms (vocoder included).

        Accepts the same Tensor/dict inputs as ``encode``.
        """
        return self(data, operation="decode")

    def _decode_video(
        self,
        latents: torch.Tensor,
        decode_timestep: float = 0.05,
        decode_noise_scale: float = 0.025,
    ) -> torch.Tensor:
        assert self._video_vae is not None, "video VAE not loaded (missing 'video' in modalities?)"
        # Ref: VideoDecoder.forward — order is:
        #   1. noise on normalized latents  (decode_noise_scale=0.025)
        #   2. per_channel_statistics.un_normalize
        #   3. conv_in + decoder blocks with temb (decode_timestep=0.05, scaled by 1000)
        model_latents = latents.to(dtype=self.precision_model)
        temb = None
        if getattr(self._video_vae.config, "timestep_conditioning", False):
            batch_size = model_latents.shape[0]
            noise = torch.randn_like(model_latents) * decode_noise_scale
            model_latents = noise + (1.0 - decode_noise_scale) * model_latents
            temb = torch.full((batch_size,), decode_timestep, device=model_latents.device, dtype=model_latents.dtype)

        mean = self._video_vae.latents_mean.view(1, -1, 1, 1, 1).to(dtype=model_latents.dtype)
        std = self._video_vae.latents_std.view(1, -1, 1, 1, 1).to(dtype=model_latents.dtype)
        model_latents = model_latents * std / self._video_vae.config.scaling_factor + mean
        return self._video_vae.decode(model_latents, temb, return_dict=False)[0]

    def _decode_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode audio latents to a waveform through the audio VAE and vocoder.

        The plain ``LTX2Vocoder`` runs in model precision, as in ltx_core; the BWE wrapper pins FP32.
        Autocast policy is left to the caller.
        """
        assert self._audio_vae is not None and self._vocoder is not None
        # Inverse of the per-(C*F) channel normalization in ``_encode_audio``.
        model_latents = latents.to(dtype=self.precision_model)
        _, C, _, F = model_latents.shape
        mean = self._audio_vae.latents_mean.to(dtype=model_latents.dtype).view(1, C, 1, F)
        std = self._audio_vae.latents_std.to(dtype=model_latents.dtype).view(1, C, 1, F)
        mel = self._audio_vae.decode(model_latents * std + mean, return_dict=False)[0]
        # The FP32 audio VAE returns an FP32 mel; stage it in model precision for the vocoder.
        return self._vocoder(mel.to(dtype=self.precision_model))

    @property
    def vocoder_sample_rate(self) -> int:
        """Vocoder output rate (24 kHz for 2.0, 48 kHz with BWE for 2.3/2.5); 16 kHz without a vocoder."""
        if self._vocoder is not None and hasattr(self._vocoder, "config"):
            return int(self._vocoder.config.output_sampling_rate)
        return 16000


# ---------------------------------------------------------------------------
# Guidance terms that only LTX2.predict_with_guidance composes
# ---------------------------------------------------------------------------


@attrs.define
class STGTerm(GuidanceTerm):
    """Spatio-temporal guidance: perturbed pass with self-attention layers skipped."""

    skip_self_attn_layers: Any = None
    skip_self_attn_layers_boundary: Any = None


@attrs.define
class ModalityGuidanceTerm(GuidanceTerm):
    """Cross-modal guidance: perturbed pass with cross-modal attention disabled.

    Rejected when ``x_t`` is a bare ``Tensor`` because there is no
    cross-modal attention to perturb in unimodal mode.
    """


@attrs.define
class LTX2GuidanceConfig(GuidanceConfig):
    """``GuidanceConfig`` with LTX-2's STG and cross-modal terms, whose branches are ``"stg"`` and ``"modality"``."""

    stg: STGTerm = attrs.field(factory=STGTerm)
    modality: ModalityGuidanceTerm = attrs.field(factory=ModalityGuidanceTerm)

    def pred_keys(self) -> List[str]:
        return [*super().pred_keys(), "stg", "modality"]

    def active_pred_keys(self) -> List[str]:
        keys = super().active_pred_keys()
        if self.stg.is_ever_active():
            keys.append("stg")
        if self.modality.is_ever_active():
            keys.append("modality")
        return keys

    def validate(
        self, supported_terms: Sequence[str], supports_cfg_skip_layers: bool = False, network_name: str = "network"
    ) -> None:
        super().validate(supported_terms, supports_cfg_skip_layers, network_name)
        for name, term in (("stg", self.stg), ("modality", self.modality)):
            if name not in supported_terms and term.scale is not None:
                raise NotImplementedError(f"{network_name} does not support guidance_cfg.{name}")


def apply_stg(ctx: GuidanceContext, prediction: TensorOrDict) -> TensorOrDict:
    """Add STG against a conditional pass that skips the configured self-attention layers."""
    g = ctx.guidance_cfg
    term = g.stg if isinstance(g, LTX2GuidanceConfig) else STGTerm()
    scale = ctx.scale(term)
    if scale is None:
        return prediction
    layers = resolve_piecewise(term.skip_self_attn_layers, ctx.t, term.skip_self_attn_layers_boundary, sequence=True)
    if "stg" not in ctx.cache and is_active(scale) and (not isinstance(layers, Sequence) or not layers):
        raise ValueError(
            "active STG requires a non-empty guidance_cfg.stg.skip_self_attn_layers "
            "unless precomputed['stg'] is supplied"
        )
    return per_modality(
        apply_guidance,
        prediction,
        ctx.predict("pos", condition=ctx.condition),
        ctx.predict("stg", condition=ctx.condition, skip_self_attn_layers=layers),
        scale,
    )


def apply_modality(ctx: GuidanceContext, prediction: TensorOrDict) -> TensorOrDict:
    """Add cross-modal guidance against a conditional pass without cross-modal attention."""
    g = ctx.guidance_cfg
    term = g.modality if isinstance(g, LTX2GuidanceConfig) else ModalityGuidanceTerm()
    scale = ctx.scale(term)
    if scale is None:
        return prediction
    if isinstance(ctx.x_t, torch.Tensor):
        raise ValueError(
            "guidance_cfg.modality is enabled for a unimodal Tensor input; "
            "cross-modal guidance requires a multimodal mapping"
        )
    return per_modality(
        apply_guidance,
        prediction,
        ctx.predict("pos", condition=ctx.condition),
        ctx.predict("modality", condition=ctx.condition, skip_cross_modal_attn=True),
        scale,
    )


# ---------------------------------------------------------------------------
# Main LTX-2 network
# ---------------------------------------------------------------------------


class LTX2(FastGenNetwork):
    """FastGen wrapper for LTX-2.0 / 2.3 / 2.5 audio-video generation.

    ``modalities`` selects video-only, audio-only or joint audio-video. Canonical model IDs imply their
    release; others must set ``ltx2_version``. The network holds the text ``connectors`` and the patched
    ``transformer``; the preprocessors are ``text_encoder`` (Gemma) and ``vae`` (video VAE, audio VAE, vocoder).

    Image-to-video uses the same transformer and VAE and follows ltx_core's ``VideoConditionByLatentIndex``:
    per-token ``1 - strength`` timesteps and ``to_denoised``, the ``post_process_latent`` blend, and a
    full-sequence Euler step. ``forward`` and ``sample`` switch to I2V whenever ``condition`` is a dict;
    ``is_i2v=True`` only makes ``build_conditioning`` encode the first frame.

    Condition format:
        - T2V: ``(prompt_embeds, attention_mask)`` from ``LTX2TextEncoder``.
        - I2V::

            condition = {
                "text_embeds": (prompt_embeds, attention_mask),
                "frame_cond": Tensor [B, C, N, H, W],  # clean latents, one per entry of frame_idx
                "frame_idx": LongTensor [N],  # optional, default [0]: latent frames to condition
                "strength": FloatTensor [N] or [B, N],  # optional, per frame or per sample; FP32 on frame_cond's
                                                        # device; default 1 (fully conditioned)
                "preserve_cond_input": bool,  # optional, default True: blend the clean frames into x_t
                                              # before the transformer (REF training); sample() sets False
            }
    """

    MODEL_ID = "Lightricks/LTX-2"
    # From the Lightricks/LTX-2 VAE configs, so shape inference works before ``init_preprocessors`` loads the VAEs.
    latent_compression_factors = (8, 32, 32)

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID,
        modalities: Tuple[str, ...] = ("video", "audio"),
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        disable_grad_ckpt: bool = False,
        load_pretrained: bool = True,
        enable_logvar_linear: bool = False,
        fps: float = 24.0,
        enable_vae_tiling: bool = False,
        is_i2v: bool = False,
        transformer_subfolder: Optional[str] = None,
        ltx2_version: Optional[str] = None,
        train_connectors: bool = False,
        **schedule_kwargs,
    ):
        """
        Args:
            model_id_or_local_path: HuggingFace model ID or local path. Defaults to ``"Lightricks/LTX-2"``.
            modalities: Enabled modalities, default ``("video", "audio")``, like the reference ``LTXModelType``.
                Selects the VAEs/vocoder, logvar heads and noise schedules; inputs may hold any non-empty subset.
            net_pred_type: Prediction type. Defaults to ``"flow"``.
            schedule_type: Noise schedule type. Defaults to ``"rf"`` (rectified flow).
            disable_grad_ckpt: Disable activation checkpointing. FSDP needs no override: ``fully_shard``
                replaces the Diffusers checkpointing with FSDP-compatible wrappers.
            load_pretrained: Load pretrained weights; ``False`` initializes from the config.
            enable_logvar_linear: Add logvar heads for uncertainty weighting (sCM). Defaults to False.
            fps: Training-data frame rate, used for RoPE coordinates and audio-shape inference.
            enable_vae_tiling: Tile VAE encode/decode to avoid 32-bit index overflow at high resolution
                (e.g. 1080p).
            is_i2v: Make ``build_conditioning`` encode the first frame into ``frame_cond``. Requires ``"video"``.
            transformer_subfolder: Transformer subfolder; defaults to ``"transformer_full"`` for 2.5 and
                ``"transformer"`` otherwise.
            ltx2_version: Release policy; ``None`` is accepted only for canonical model IDs.
            train_connectors: Fine-tune the text connectors. Defaults to False, like the official LTX trainer and
                released distilled checkpoint; frozen connectors stay in BF16 with their own BF16 FSDP groups.
        """
        self.modalities = _validate_modalities(modalities)
        self.train_connectors = train_connectors
        # The native sampler grids start at sigma 1.0, so the raw-time bound defaults to 1.0 (configs may lower it).
        schedule_kwargs.setdefault("max_t", 1.0)
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)
        self.model_id_or_local_path = model_id_or_local_path
        self.ltx2_version = _resolve_ltx2_version(model_id_or_local_path, ltx2_version)
        self.transformer_subfolder = _ltx2_transformer_subfolder(
            model_id_or_local_path,
            transformer_subfolder,
            version=self.ltx2_version,
        )
        self.fps = fps
        self._enable_vae_tiling = enable_vae_tiling
        self.is_i2v = is_i2v
        if is_i2v:
            assert "video" in self.modalities, "is_i2v=True requires 'video' in modalities"

        self._initialize_network(model_id_or_local_path, load_pretrained, enable_logvar_linear)
        if self.ltx2_version == "2.5" and not _ltx2_first_frame_embedding_enabled(self.transformer):
            logger.warning(
                "The selected LTX-2.5 transformer must enable use_keyframes_abs_pos_embedding "
                "and provide keyframes_abs_pos_embedding."
            )

        # ``classify_forward`` is bound to the Diffusers transformer below, so
        # attach the wrapper's release selection to that runtime object. Keep
        # it out of the serialized Diffusers config/state dict.
        self.transformer._fastgen_ltx2_version = self.ltx2_version

        # Monkey-patch classify_forward onto self.transformer
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Monkey-patch each transformer block so audio is optional (matching
        # the original LTX-2 BasicAVTransformerBlock run_ax/run_a2v pattern).
        for block in self.transformer.transformer_blocks:
            block.forward = types.MethodType(_block_forward_audio_optional, block)

        # Replace the default Diffusers attention processor with a native-dtype
        # RoPE variant so that FastGen matches the ltx_core reference precision.
        processor = NativeDtypeAttnProcessor()
        for root in (self.transformer, self.connectors):
            for module in root.modules():
                if isinstance(module, LTX2Attention):
                    module.set_processor(processor)

        # Gradient checkpointing
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_network(
        self, model_id_or_local_path: str, load_pretrained: bool, enable_logvar_linear: bool = False
    ) -> None:
        """Initialize the transformer and supporting modules."""
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading LTX-2 transformer from pretrained")
            self.transformer: LTX2VideoTransformer3DModel = hf_load(
                LTX2VideoTransformer3DModel.from_pretrained,
                model_id_or_local_path,
                subfolder=self.transformer_subfolder,
            )
        else:
            config = hf_load(
                LTX2VideoTransformer3DModel.load_config, model_id_or_local_path, subfolder=self.transformer_subfolder
            )
            if in_meta_context:
                logger.info(
                    "Initializing LTX-2 transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing LTX-2 transformer from config (no pretrained weights)")
                logger.warning("LTX-2 transformer being initialized from config. No weights are loaded!")
            self.transformer: LTX2VideoTransformer3DModel = LTX2VideoTransformer3DModel.from_config(config)

        # inner_dim = num_attention_heads * attention_head_dim
        inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        audio_inner_dim = (
            self.transformer.config.audio_num_attention_heads * self.transformer.config.audio_attention_head_dim
        )

        if enable_logvar_linear:
            self.transformer.logvar_linear = nn.Linear(inner_dim, 1)
            logger.debug(f"Added logvar_linear ({inner_dim} → 1) to LTX-2 transformer")
            # Audio embeddings have a different feature width than video
            # (``audio_inner_dim`` vs ``inner_dim``).  An audio-only forward
            # uses the audio embedded-timestep, so it needs its own logvar
            # head when 'audio' is an enabled modality.
            if "audio" in self.modalities:
                self.transformer.audio_logvar_linear = nn.Linear(audio_inner_dim, 1)
                logger.debug(f"Added audio_logvar_linear ({audio_inner_dim} → 1) to LTX-2 transformer")
        else:
            logger.debug("Skipped logvar_linear layer (disabled by enable_logvar_linear=False)")

        # Keep connectors in checkpoints, but mirror the official trainer by
        # freezing them in their checkpoint dtype unless explicitly requested.
        if should_load_weights:
            self.connectors: LTX2TextConnectors = hf_load(
                LTX2TextConnectors.from_pretrained, model_id_or_local_path, subfolder="connectors"
            )
        else:
            config = hf_load(LTX2TextConnectors.load_config, model_id_or_local_path, subfolder="connectors")
            self.connectors: LTX2TextConnectors = LTX2TextConnectors.from_config(config)
        if not self.train_connectors:
            self.connectors.to(dtype=torch.bfloat16)
            self._freeze_connectors_()

        # Compression constants from the Lightricks/LTX-2 configs, so shape inference and
        # token unpacking work before ``init_preprocessors`` loads the VAEs.
        self.transformer_spatial_patch_size = self.transformer.config.patch_size  # 1
        self.transformer_temporal_patch_size = self.transformer.config.patch_size_t  # 1
        self.audio_sampling_rate = 16000
        self.audio_hop_length = 160
        self.audio_vae_temporal_compression = 4
        self.audio_vae_mel_compression_ratio = 4
        self.audio_vae_mel_bins = 64
        self.audio_vae_latent_channels = 8

    # ------------------------------------------------------------------
    # Preprocessor initialisation (lazy)
    # ------------------------------------------------------------------

    def init_preprocessors(self):
        """Initialize the text encoder and the VAE bundle (with audio VAE and vocoder when audio is enabled).

        Only rank 0 loads the text encoder weights; other ranks stay on meta until ``prepare_module``
        materializes and synchronizes or shards them.
        """
        if "text_encoder" not in self.preprocessors:
            self.init_text_encoder()
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_text_encoder(self, dtype: Optional[torch.dtype] = torch.bfloat16):
        """Initialize the release's Gemma text encoder; ``dtype`` defaults to BF16 like the LTX-2 reference."""
        self.preprocessors["text_encoder"] = LTX2TextEncoder(
            model_id_or_local_path=self.model_id_or_local_path,
            dtype=dtype,
            ltx2_version=self.ltx2_version,
        )

    def init_vae(self):
        """Initialize the VAE bundle (video VAE, audio VAE and vocoder) for the configured modalities."""
        self.preprocessors["vae"] = LTX2VAE(
            self.model_id_or_local_path,
            enable_tiling=self._enable_vae_tiling,
            modalities=self.modalities,
        )

    def encode_real(self, real_raw: TensorOrDict) -> TensorOrDict:
        """Encode raw video and/or audio with the VAE, after checking that an audio-video clip runs at ``fps``."""
        if isinstance(real_raw, Mapping) and {"video", "audio"} <= real_raw.keys():
            _validate_audio_video_fps(
                real_raw, self.fps, self.audio_sampling_rate, self.audio_hop_length, self.audio_vae_temporal_compression
            )
        return super().encode_real(real_raw)

    # ------------------------------------------------------------------
    # Frozen connector placement
    # ------------------------------------------------------------------

    def _freeze_connectors_(self) -> None:
        self.connectors.eval().requires_grad_(False)

    def requires_grad_(self, requires_grad: bool = True):
        """Respect the connector freeze across generic student grad toggles."""
        super().requires_grad_(requires_grad)
        if requires_grad and not self.train_connectors and hasattr(self, "connectors"):
            self._freeze_connectors_()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and not self.train_connectors and hasattr(self, "connectors"):
            self.connectors.eval()
        return self

    def _fixed_precision_overrides(self) -> Dict[str, torch.dtype]:
        """Keep frozen text connectors in their release BF16 precision."""
        if self.train_connectors:
            return {}
        return {name: torch.bfloat16 for name in module_state_names(self) if name.startswith("connectors.")}

    # ------------------------------------------------------------------
    # FSDP
    # ------------------------------------------------------------------

    def fully_shard(self, **kwargs):
        """Shard ``self.transformer`` and the text connectors; see ``FastGenNetwork.fully_shard``."""
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(self.transformer, LTX2VideoTransformerBlock)

        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

        # Frozen connectors are stored and computed in bf16 (``_fixed_precision_overrides``), so they
        # need their own group whenever the parent policy computes in another dtype.
        connector_kwargs = kwargs
        if not self.train_connectors:
            self._freeze_connectors_()
            connector_kwargs = (
                make_fully_shard_kwargs(
                    kwargs, param_dtype=torch.bfloat16, storage_dtype=self.precision_opt, enabled=True
                )
                or kwargs
            )

        # Wrap connector leaves before their parents to bound collectives and
        # place frozen connectors as sharded tensors on the compute device.
        for connector in self.connectors.children():
            connector_blocks = getattr(connector, "transformer_blocks", None)
            if connector_blocks is not None:
                for block in connector_blocks:
                    fully_shard(block, **connector_kwargs)
            if any(True for _ in connector.parameters()):
                fully_shard(connector, **connector_kwargs)
        fully_shard(self.connectors, **connector_kwargs)

        if self.train_connectors:
            logger.info("Applied fine-grained FSDP wrapping to trainable LTX-2 text connectors")
        else:
            logger.info("Applied fine-grained bf16 FSDP wrapping to frozen LTX-2 text connectors")

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def _set_noise_schedule(self, schedule_type=None, **schedule_kwargs):
        assert not self.is_prepared, "Initialize the noise schedule before preparation"
        if schedule_type is not None:
            self.schedule_type = schedule_type
        if len(self.modalities) <= 1:
            super()._set_noise_schedule(self.schedule_type, **schedule_kwargs)
        else:
            schedule_cfg = {"name": self.schedule_type, **schedule_kwargs}
            self.noise_scheduler = CompositeNoiseSchedule(
                schedules={m: schedule_cfg for m in self.modalities},
                primary_key=self.modalities[0],
            )

    # ------------------------------------------------------------------
    # Shape inference (populates None values in input_shape)
    # ------------------------------------------------------------------

    def infer_shape(self, input_shape: Union[List[int], Dict[str, Optional[List[int]]]]):
        """Fill a missing multimodal ``"audio"`` shape from the ``"video"`` shape and fps.

        Unimodal networks use flat shapes (bare tensors), which are returned unchanged.
        """
        if not isinstance(input_shape, Mapping):
            assert (
                len(self.modalities) == 1
            ), f"A plain LTX-2 input shape requires exactly one configured modality; got {self.modalities}."
            return input_shape

        input_shape = dict(input_shape)  # shallow copy
        if "audio" in input_shape and input_shape["audio"] is None:
            video_shape = input_shape["video"]
            assert video_shape is not None, "Cannot infer audio shape without video shape"
            audio_L, audio_M, audio_C = self._compute_audio_shape(video_shape, fps=self.fps)
            input_shape["audio"] = [audio_C, audio_L, audio_M]
        return input_shape

    # ------------------------------------------------------------------
    # Audio latent sizing helper
    # ------------------------------------------------------------------

    @property
    def audio_latent_mel_bins(self) -> int:
        """Mel bins of an audio latent: the audio VAE's input bins over its mel compression."""
        return self.audio_vae_mel_bins // self.audio_vae_mel_compression_ratio

    def _compute_audio_shape(self, video_shape: Sequence[int], fps: float) -> Tuple[int, int, int]:
        """Return ``(audio_num_frames, latent_mel_bins, num_audio_ch)`` for a ``[C, F, H, W]`` video latent."""
        pixel_frames = self.pixel_shape(video_shape)[0]

        # Mirror ``AudioVideoWDSLoader``'s waveform length (``int(frames * sr / fps)``,
        # multiplied first so integral counts survive float rounding) so the inferred
        # shape equals the encoder output for that clip. This follows the reference
        # trainer rather than its inference contract ``round(duration *
        # latents_per_second)``; see ``_audio_latent_frames``.
        waveform_samples = int(pixel_frames * self.audio_sampling_rate / fps)
        audio_num_frames = _audio_latent_frames(
            waveform_samples,
            hop_length=self.audio_hop_length,
            temporal_compression=self.audio_vae_temporal_compression,
        )
        return audio_num_frames, self.audio_latent_mel_bins, self.audio_vae_latent_channels

    # ------------------------------------------------------------------
    # Feature unpacking helper
    # ------------------------------------------------------------------

    def _unpatchify_features(self, video_latents: Optional[torch.Tensor], features: list) -> list:
        """Unpack token features: video to ``[B, C, F, H, W]`` and audio to ``[B, C, L, M]``.

        Plain tensors are video features, or audio features when ``video_latents`` is ``None``.
        """
        latent_mel_bins = self.audio_latent_mel_bins
        if video_latents is None:
            return [_unpack_audio_latents(feat, latent_mel_bins) for feat in features]

        _, _, latent_f, latent_h, latent_w = video_latents.shape
        ps = self.transformer_spatial_patch_size
        pt = self.transformer_temporal_patch_size

        unpacked = []
        for feat in features:
            if not isinstance(feat, torch.Tensor):
                v = _unpack_latents(feat["video"], latent_f, latent_h, latent_w, ps, pt)
                a = _unpack_audio_latents(feat["audio"], latent_mel_bins)
                unpacked.append({"video": v, "audio": a})
            else:
                unpacked.append(_unpack_latents(feat, latent_f, latent_h, latent_w, ps, pt))
        return unpacked

    @property
    def connector_precision(self) -> Optional[torch.dtype]:
        """Input dtype of the text connectors: BF16 when frozen, else the transformer's ``precision_model``."""
        return self.precision_model if self.train_connectors else torch.bfloat16

    # ------------------------------------------------------------------
    # I2V conditioning (active when ``is_i2v=True``)
    # ------------------------------------------------------------------

    def build_conditioning(self, data, input_shape=None, precision=None):
        """Resolve the text branches and, with ``is_i2v=True``, add the first-frame I2V condition.

        Text is resolved in ``connector_precision`` rather than ``precision_model``, since an FP16 detour can
        overflow values the BF16 connectors represent. For I2V, the first frame of ``data["real_raw"]`` (its
        ``"video"`` entry for dicts) is VAE-encoded with ``mode="argmax"`` into ``precision`` and returned as
        ``{"condition": {"text_embeds": ..., "frame_cond": latent}}``; a present ``"neg_condition"`` gets the
        same frame fields. An optional ``data["strength"]``, a scalar or one per sample, is added in FP32 on the
        latent's device. Unlike ``WanI2V`` there is no concat mask; LTX-2 uses a per-token denoise mask.
        ``input_shape`` is unused. I2V raises ``ValueError`` without ``data["real_raw"]``.
        """
        out = self._resolve_conditions(data, dtype=self.connector_precision)
        required_fields = ("condition",) if self.is_i2v else ()
        validate_conditioning(out, required_fields=required_fields)
        text_embeds = out.get("condition")
        if not self.is_i2v:
            return out

        real_raw = data.get("real_raw")
        if real_raw is None:
            raise ValueError(
                f"{type(self).__name__} requires data['real_raw'] (pixel-space). "
                "Use a pixel dataloader, or supply mp4 in the loader's key_map."
            )

        # Call the video branch directly so this path works uniformly for
        # unimodal- and multimodal-init VAEs (the generic ``vae.encode`` would
        # require a dict on a multimodal VAE).
        video = real_raw["video"] if isinstance(real_raw, Mapping) else real_raw
        vae = self.get_preprocessor("vae", "encode the first frame")
        first_frame = vae.encode_video(video[:, :, :1], mode="argmax").to(dtype=precision)

        frame_condition = {"frame_cond": first_frame}
        if data.get("strength") is not None:
            strength = torch.as_tensor(data["strength"], dtype=torch.float32, device=first_frame.device)
            frame_condition["strength"] = strength.reshape(1) if strength.numel() == 1 else strength.reshape(-1, 1)

        out["condition"] = {"text_embeds": text_embeds, **frame_condition}
        if out.get("neg_condition") is not None:
            out["neg_condition"] = {"text_embeds": out["neg_condition"], **frame_condition}
        return out

    @staticmethod
    def _blend_with_clean(
        x: torch.Tensor,
        clean: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Return REF ``post_process_latent``'s ``x * weight + clean * (1 - weight)``, cast to ``x.dtype``.

        Operands must share a device and are promoted to a common dtype for the blend. ``x`` and ``clean`` are
        ``(B, C, N, H, W)``; ``weight`` is ``(1, 1, N, 1, 1)`` or ``(B, 1, N, 1, 1)``, 1 for full denoising and 0 for
        fully conditioned.
        """
        assert_same(device=True, x=x, clean=clean, weight=weight)
        output_dtype = x.dtype
        x, clean, weight = to_promoted_dtype(x, clean, weight)
        return (x * weight + clean * (1.0 - weight)).to(dtype=output_dtype)

    @staticmethod
    def _unpack_i2v_condition(
        condition: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate an I2V condition and return ``(frame_idx, cond_latent, inv_strength)``.

        ``frame_idx`` is a 1-D ``torch.long`` tensor (default ``[0]``) and ``cond_latent`` the clean
        ``(B, C, len(frame_idx), H, W)`` latent. ``strength`` holds one value per conditioned frame, ``[N]``, or one
        per sample and frame, ``[B, N]``; ``inv_strength`` is ``1 - strength`` shaped ``(1, 1, N, 1, 1)`` or
        ``(B, 1, N, 1, 1)`` to broadcast against video tensors. ``strength`` must be FP32 (the reference mask dtype)
        on ``cond_latent``'s device; blends promote it and timestep products cast it to the timestep dtype.
        """
        assert (
            "text_embeds" in condition and "frame_cond" in condition
        ), "I2V condition (dict) requires 'text_embeds' and 'frame_cond' keys"
        cond_latent = condition["frame_cond"]
        batch, _, num_frames = cond_latent.shape[:3]
        frame_idx = condition.get("frame_idx")
        if frame_idx is None:
            frame_idx = torch.tensor([0], dtype=torch.long, device=cond_latent.device)
        assert_same(shape=(num_frames,), dtype=torch.long, device=cond_latent.device, frame_idx=frame_idx)
        strength = condition.get("strength")
        if strength is None:
            strength = torch.ones(num_frames, device=cond_latent.device, dtype=torch.float32)
        assert_same(dtype=torch.float32, device=cond_latent.device, strength=strength)
        shapes = ((num_frames,), (batch, num_frames))
        assert strength.shape in shapes, f"strength shape {tuple(strength.shape)} is not one of {shapes}"
        inv_strength = (1.0 - strength).view(-1, 1, num_frames, 1, 1)
        return frame_idx, cond_latent, inv_strength

    @property
    def _video_noise_scheduler(self) -> BaseNoiseSchedule:
        """The video noise schedule: the network's own, or the ``"video"`` sub-schedule of a composite one."""
        ns = self.noise_scheduler
        return ns.subschedules["video"] if hasattr(ns, "subschedules") else ns

    def _i2v_model_input(self, video_latents: torch.Tensor, condition: Dict[str, Any]) -> torch.Tensor:
        """The video the transformer sees for an I2V ``condition``: REF's training-style input blend.

        ``model_input_video[cond] = video_latents[cond] * (1-strength) + clean * strength`` collapses to a pure
        replace at ``strength == 1`` (REF ``text_to_video.py``'s ``torch.where(cond_mask, video_latents,
        noisy_video)``). The raw ``video_latents`` still anchor the final ``x0 → fwd_pred_type`` round trip, so
        consumers (training loss, sample loop) recover ``x0 = clean`` at conditioned tokens. ``sample()`` opts out
        with ``preserve_cond_input=False``: its init blend and Euler evolution already maintain the REF-faithful
        state, and re-blending the input each step would shortcut the trajectory.
        """
        if not condition.get("preserve_cond_input", True):
            return video_latents
        frame_idx, cond_latent, inv_strength = self._unpack_i2v_condition(condition)
        model_input_video = video_latents.clone()
        model_input_video[:, :, frame_idx] = self._blend_with_clean(
            video_latents[:, :, frame_idx], cond_latent, inv_strength
        )
        return model_input_video

    def _convert_output(self, prediction, x_t, t, condition, fwd_pred_type, **context):
        """Convert every position with the scalar ``t``; an x0 at the clean end restores the conditioned frames.

        ``forward`` expresses the I2V conditioned frames against the raw state and scalar ``t``
        (``_convert_i2v_output``), so this is REF's conversion everywhere, and the only one audio needs. Where
        ``sigma = 0`` that velocity cannot carry their per-token x0, which REF ``X0Model.to_denoised`` gives there as
        the transformer input, independent of the velocity.
        """
        converted = super()._convert_output(prediction, x_t, t, condition, fwd_pred_type)
        if fwd_pred_type != "x0" or not isinstance(condition, dict):
            return converted
        video_ns = self._video_noise_scheduler
        video_latents = x_t["video"] if isinstance(x_t, dict) else x_t
        video = converted["video"] if isinstance(converted, dict) else converted
        frame_idx = self._unpack_i2v_condition(condition)[0]
        model_input = self._i2v_model_input(video_latents, condition)[:, :, frame_idx].to(dtype=video.dtype)
        noised = expand_like(video_ns.sigma(t) > 0, model_input)
        video = video.clone()
        video[:, :, frame_idx] = torch.where(noised, video[:, :, frame_idx], model_input)
        return {**converted, "video": video} if isinstance(converted, dict) else video

    def _condition_output(self, prediction, x_t, t, condition, fwd_pred_type, **context):
        """Apply REF ``post_process_latent`` once, after guidance and rescaling; a no-op for T2V.

        Per-token ``to_denoised`` is part of the conversion before it. Conversion is affine, so blending with the
        clean target converted to ``fwd_pred_type`` imposes the x0 constraint in any prediction space without
        inverting a partial blend. At ``sigma = 0`` a non-x0 prediction cannot carry it, which raises unless all of
        that sample's strengths are zero. Audio and unconditioned frames are untouched.
        """
        if not isinstance(condition, dict):
            return prediction
        video = prediction["video"] if isinstance(prediction, dict) else prediction
        video_latents = x_t["video"] if isinstance(x_t, dict) else x_t
        frame_idx, clean, inv_strength = self._unpack_i2v_condition(condition)
        ns = self._video_noise_scheduler
        blend_dtype = promote_dtype(video, clean, inv_strength)
        if fwd_pred_type != "x0":
            noised = ns.sigma(t) > 0
            unconstrained = (inv_strength == 1).flatten(1).all(dim=1)
            assert_tensor(
                (noised | unconstrained).all(),
                "A non-x0 prediction cannot carry the output constraint at sigma = 0; request fwd_pred_type='x0' "
                "or apply_output_conditioning=False there",
                error_type=ValueError,
            )
            # Zero-strength frames keep the prediction, so their target only needs to be finite at sigma = 0.
            t_safe = torch.where(noised, t, torch.full_like(t, ns.max_t))
            clean = ns.convert_model_output(
                video_latents[:, :, frame_idx],
                clean,
                t_safe,
                src_pred_type="x0",
                target_pred_type=fwd_pred_type,
                output_dtype=blend_dtype,
            )
        # The shared blend promotes operands and returns video.dtype; stored conditions stay untouched.
        conditioned = self._blend_with_clean(video[:, :, frame_idx], clean, inv_strength)
        result = video.clone()
        result[:, :, frame_idx] = conditioned
        return {**prediction, "video": result} if isinstance(prediction, dict) else result

    def _convert_i2v_output(
        self,
        model_output: TensorOrDict,
        model_input_video: torch.Tensor,
        video_latents: torch.Tensor,
        frame_idx: torch.Tensor,
        inv_strength: torch.Tensor,
        t: torch.Tensor,
    ) -> TensorOrDict:
        """Express the conditioned frames' native prediction against the raw state and scalar ``t``.

        Their x0 is REF ``X0Model.to_denoised``, from the effective timestep ``t * (1 - strength)`` and the actual
        transformer input (after any input blend). Converted back to ``net_pred_type`` against the raw state and
        scalar ``t``, every position converts with the scalar ``t``. The video is returned in high precision, which
        ``transform_output`` keeps, so the prediction is rounded only once. At a clean endpoint x0 does not
        determine a non-x0 prediction, so the conditioned frames keep the transformer's value there and
        ``_convert_output`` restores their x0.
        """
        video_ns = self._video_noise_scheduler
        video = model_output["video"] if isinstance(model_output, dict) else model_output
        # Always a copy, so the in-place merge below never writes into the transformer output.
        video = video.to(dtype=high_precision_dtype(video), copy=True)

        # Per-frame to_denoised at the cond frames: ``x0_cond = state.latent -
        # velocity_cond * (1-strength[i])*t`` with one effective timestep per
        # cond frame.  REF uses ``state.latent`` (= our ``model_input_video``
        # at cond positions, see docstring above). The timestep keeps ``t``'s
        # dtype, as ``convert_model_output`` requires, and ``safe_clamp`` lifts
        # its strength-1 value 0 to ``min_t`` for schedules that start above 0.
        t_cond = video_ns.safe_clamp(
            t.view(-1, 1, 1, 1, 1) * inv_strength.to(dtype=t.dtype),
            min=video_ns.min_t,
            max=video_ns.max_t,
        )
        x0_cond = video_ns.convert_model_output(
            model_input_video[:, :, frame_idx],
            video[:, :, frame_idx],
            t_cond,
            src_pred_type=self.net_pred_type,
            target_pred_type="x0",
            output_dtype=video.dtype,
        )

        # x0 → net_pred_type with scalar t (downstream consumer convention,
        # uses raw ``video_latents`` so ``x0_recon = video_latents -
        # fwd_cond * t`` recovers the per-token x0 at the cond frames).
        noised = None
        if self.net_pred_type == "x0":
            fwd_cond = x0_cond
        else:
            noised = video_ns.sigma(t) > 0
            t_safe = torch.where(noised, t, torch.full_like(t, video_ns.max_t))
            fwd_cond = video_ns.convert_model_output(
                video_latents[:, :, frame_idx],
                x0_cond,
                t_safe,
                src_pred_type="x0",
                target_pred_type=self.net_pred_type,
                output_dtype=video.dtype,
            )

        # Merge: write ``fwd_cond`` (B, C, len(frame_idx), H, W) into the
        # native output at the conditioned frame indices via fancy indexing.
        # Non-cond frames keep their native value; at a clean endpoint, a
        # non-x0 output keeps it at the conditioned frames too.
        if noised is not None:
            fwd_cond = torch.where(
                expand_like(noised, fwd_cond),
                fwd_cond,
                video[:, :, frame_idx],
            )
        video[:, :, frame_idx] = fwd_cond
        if isinstance(model_output, dict):
            return {**model_output, "video": video}
        return video

    # ------------------------------------------------------------------
    # forward() — active-modality distillation interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: TensorOrDict,
        t: torch.Tensor,
        condition: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]] = None,
        fps: Optional[float] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        unpatchify_features: bool = True,
        skip_layers: Optional[List[int]] = None,
        skip_self_attn_layers: Optional[List[int]] = None,
        skip_cross_modal_attn: bool = False,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[TensorOrDict, List[torch.Tensor], Tuple]:
        """Forward pass for the modalities present in ``x_t``.

        A plain tensor is the sole configured modality; a dict runs its ``"video"``/``"audio"`` entries. A dict
        ``condition`` selects I2V (see the class docstring): per-token timesteps and ``to_denoised``, which expresses
        the conditioned frames against the scalar ``t``, then the clean-frame blend unless
        ``apply_output_conditioning=False``. Arguments not listed follow ``FastGenNetwork.forward``.

        Args:
            x_t: Video ``[B, C, F, H, W]`` or audio ``[B, C, L, M]`` latents, or a dict of them.
            t: Timestep ``[B]``.
            condition: T2V ``(prompt_embeds, attention_mask)`` from ``LTX2TextEncoder``, or the I2V dict.
            fps: Frame rate for RoPE coordinates; defaults to ``self.fps``.
            skip_layers: Transformer blocks to skip entirely (SLG).
            skip_self_attn_layers: Blocks whose self-attention returns V (STG).
            skip_cross_modal_attn: Skip a2v/v2a cross-attention (modality guidance).

        Returns:
            The prediction, a dict for dict ``x_t``; ``[prediction, features]`` with ``feature_indices``;
            ``(above, logvar [B, 1])`` with ``return_logvar``; only the features with ``return_features_early``.
            ``unpatchify_features`` unpacks the features to the latent layouts.
        """
        num_blocks = len(self.transformer.transformer_blocks)
        self._validate_forward_kwargs(
            fwd_kwargs,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            skip_layers=skip_layers,
            supports_skip_layers=True,
            num_features=num_blocks,
        )
        if skip_self_attn_layers:
            self._validate_index_range("skip_self_attn_layers", skip_self_attn_layers, num_blocks)
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        _validate_input(x_t, self.modalities, "forward()", "network")
        if isinstance(x_t, dict):
            video_latents = x_t.get("video")
            audio_latents = x_t.get("audio")
        elif self.modalities[0] == "video":
            video_latents, audio_latents = x_t, None
        else:
            video_latents, audio_latents = None, x_t
        run_video = video_latents is not None
        run_audio = audio_latents is not None

        if fps is None:
            fps = self.fps

        first_tensor = video_latents if video_latents is not None else audio_latents
        batch_size = first_tensor.shape[0]
        device = first_tensor.device

        # ---- Unpack text conditioning (dispatch on condition shape) ----
        # T2V: ``condition = (prompt_embeds, attention_mask)`` (2-tuple).
        # I2V: ``condition = {"text_embeds": (...), "frame_cond": ..., ...}``.
        # The dispatch is purely on the condition shape, so the same ``LTX2``
        # instance can serve both modes (useful for ablations or T2V → I2V
        # finetuning where you want to flip behavior without re-instantiating).
        # The init-time ``self.is_i2v`` flag only steers ``build_conditioning``
        # (whether to VAE-encode a first frame).
        is_i2v = isinstance(condition, dict)
        if is_i2v:
            assert run_video, "I2V conditioning requires a video input"
            frame_idx, _, inv_strength = self._unpack_i2v_condition(condition)
            prompt_embeds, attention_mask = condition["text_embeds"]
        else:
            if not isinstance(condition, (tuple, list)) or len(condition) != 2:
                raise ValueError(
                    f"T2V condition must be a 2-tuple (prompt_embeds, attention_mask), "
                    f"got {type(condition).__name__}"
                    + (f" of length {len(condition)}" if hasattr(condition, "__len__") else "")
                )
            prompt_embeds, attention_mask = condition

        # ---- Optional input-side conditioning preservation (REF training-style) ----
        model_input_video = self._i2v_model_input(video_latents, condition) if is_i2v else video_latents

        # ---- Run connectors to get per-modality encoder hidden states ----
        assert_same(dtype=self.connector_precision, device=device, prompt_embeds=prompt_embeds)
        connector_video_embeds, connector_audio_embeds, connector_attn_mask = self.connectors(
            prompt_embeds, attention_mask
        )
        # The frozen connectors are an internal bf16 island (release dtype). Stage
        # their outputs in the conditioning dtype (``precision_model``) here; any later
        # mixed-precision operations follow the active autocast policy.
        connector_video_embeds = connector_video_embeds.to(dtype=self.precision_model)
        connector_audio_embeds = connector_audio_embeds.to(dtype=self.precision_model)

        # ---- Timestep: per-token to match the reference (ltx_core) ----
        # The reference passes per-token timesteps [B, T] through the adaln MLP,
        # while a global [B] timestep (broadcast) produces different bf16 rounding
        # in the matmul due to a different batch dimension.  Using per-token
        # timesteps keeps the two codebases bitwise-identical.
        timestep_scalars = expand_first_dim(self.noise_scheduler.rescale_t(t), batch_size)
        if isinstance(timestep_scalars, Mapping):
            video_timestep_scalar = timestep_scalars.get("video")
            audio_timestep_scalar = timestep_scalars.get("audio")
        else:
            video_timestep_scalar = audio_timestep_scalar = timestep_scalars

        # ---- Pack video latents (use ``model_input_video`` so the transformer
        # sees clean conditioning tokens at conditioned positions when
        # ``preserve_cond_input=True``; equal to ``video_latents`` in T2V mode). ----
        hidden_states = None
        video_coords = None
        latent_f = latent_h = latent_w = None
        if run_video:
            _, _, latent_f, latent_h, latent_w = video_latents.shape
            hidden_states = _pack_latents(
                self.cast_to_model_precision(model_input_video),
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
            video_coords = self.transformer.rope.prepare_video_coords(
                batch_size, latent_f, latent_h, latent_w, device, fps=fps
            )

        # ---- Audio: optional, following original LTX-2 pattern ----
        audio_hidden_states = None
        audio_num_frames = None
        audio_coords = None

        if run_audio:
            audio_hidden_states = _pack_audio_latents(self.cast_to_model_precision(audio_latents))
            audio_num_frames = audio_hidden_states.shape[1]
            audio_coords = self.transformer.audio_rope.prepare_audio_coords(batch_size, audio_num_frames, device)

        # Expand global timestep to per-token [B, T] to match the reference adaln path.
        # In I2V mode, video tokens use the denoise_mask-weighted timesteps (``mask *
        # sigma`` per REF ``timesteps_from_mask``: ``1 - strength`` at conditioned
        # frame tokens, ``1`` elsewhere), so conditioned tokens see timestep 0 (full
        # conditioning) at strength=1.
        video_timestep = None
        audio_timestep_expanded = None
        if run_video:
            num_video_tokens = hidden_states.shape[1]
            video_timestep = video_timestep_scalar.unsqueeze(1).expand(batch_size, num_video_tokens)
            if is_i2v:
                # Build per-frame weight ``(B, F, tokens_per_frame)`` with ``1.0``
                # everywhere except ``1 - strength[i]`` at the conditioned-frame
                # rows ``frame_idx[i]``, then flatten to per-token ``(B, T)``.
                # Reshape ``inv_strength`` to ``(1 or B, N, 1)`` so the per-frame
                # weights broadcast across the ``tokens_per_frame`` axis of the
                # LHS slice.
                tokens_per_frame = latent_h * latent_w
                denoise_weight = video_timestep.new_ones(batch_size, latent_f, tokens_per_frame)
                # Use the scheduler's timestep dtype for both operands.
                denoise_weight[:, frame_idx, :] = inv_strength.flatten(1).unsqueeze(-1).to(denoise_weight.dtype)
                denoise_weight = denoise_weight.view(batch_size, num_video_tokens)
                video_timestep = video_timestep * denoise_weight
        if run_audio:
            audio_timestep_expanded = audio_timestep_scalar.unsqueeze(1).expand(batch_size, audio_num_frames)

        # ---- Transformer forward ----
        video_packed, audio_packed, logvar, features = self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=connector_video_embeds if run_video else None,
            audio_encoder_hidden_states=connector_audio_embeds if run_audio else None,
            encoder_attention_mask=connector_attn_mask if run_video else None,
            audio_encoder_attention_mask=connector_attn_mask if run_audio else None,
            timestep=video_timestep,
            audio_timestep=audio_timestep_expanded,
            sigma=video_timestep_scalar if run_video else audio_timestep_scalar,
            audio_sigma=audio_timestep_scalar if run_audio else None,
            num_frames=latent_f,
            height=latent_h,
            width=latent_w,
            fps=fps,
            audio_num_frames=audio_num_frames,
            video_coords=video_coords,
            audio_coords=audio_coords,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            skip_layers=skip_layers,
            skip_self_attn_layers=skip_self_attn_layers,
            skip_cross_modal_attn=skip_cross_modal_attn,
        )

        # ---- Early exit: list of active-modality feature tensors ----
        if return_features_early:
            return self._unpatchify_features(video_latents, features) if unpatchify_features else features

        # ---- Unpack packed tokens to spatial form ----
        video_out = None
        if run_video:
            video_out = _unpack_latents(
                video_packed,
                latent_f,
                latent_h,
                latent_w,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
        audio_out = None
        if run_audio:
            audio_out = _unpack_audio_latents(audio_packed, self.audio_latent_mel_bins)

        # ---- Convert model output to requested prediction type (REF X0Model + post_process) ----
        if isinstance(x_t, dict):
            model_output = {}
            if video_out is not None:
                model_output["video"] = video_out
            if audio_out is not None:
                model_output["audio"] = audio_out
        elif run_video:
            model_output = video_out
        else:
            model_output = audio_out
        if is_i2v:
            model_output = self._convert_i2v_output(
                model_output, model_input_video, video_latents, frame_idx, inv_strength, t
            )
        model_output = self.transform_output(
            model_output, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
        )

        if features is not None:
            out = [
                model_output,
                self._unpatchify_features(video_latents, features) if unpatchify_features else features,
            ]
        else:
            out = model_output

        if return_logvar:
            return out, logvar.to(dtype=first_tensor.dtype)
        return out

    @staticmethod
    def default_guidance_cfg(
        modalities: Sequence[str] = ("video", "audio"),
        version: str = "2.0",
    ) -> LTX2GuidanceConfig:
        """Canonical LTX-2 inference recipe for ``modalities`` and release ``version``.

        CFG ``video=3.0, audio=7.0``; STG scale 2.0 at self-attention layer 28 (2.3/2.5) or 29 (2.0); cross-modal
        guidance 3.0 with several modalities; final rescale 0.7.
        """
        modalities = _validate_modalities(modalities)
        version = _validate_ltx2_version(version)
        cfg_scales = {"video": 3.0, "audio": 7.0}
        if len(modalities) == 1:
            cfg_scale = cfg_scales[modalities[0]]  # scalar — unimodal Tensor input
        else:
            cfg_scale = {m: cfg_scales[m] for m in modalities}  # multi-key dict — multimodal input

        g = LTX2GuidanceConfig()
        g.cfg.scale = cfg_scale
        # STG is 2.0 here and 1.0 in the reference: FastGen composes every term as
        # ``state + (scale - 1) * delta`` while ltx_core's ``MultiModalGuider`` special-cases STG
        # as ``scale * delta``. So our scale is the reference's plus one; setting the literal 1.0
        # would disable STG entirely.
        g.stg.scale = 2.0
        g.stg.skip_self_attn_layers = [28] if version in ("2.3", "2.5") else [29]
        # Cross-modal guidance is only meaningful when more than one modality is active.
        if len(modalities) > 1:
            g.modality.scale = 3.0
        g.rescale = 0.7
        return g

    def predict_with_guidance(self, ctx: GuidanceContext) -> TensorOrDict:
        """Compose ltx_core's ``MultiModalGuider`` on the native velocities, rescaling on x0.

        CFG, STG, cross-modal guidance and mix add differences of branches, so they compose on the native branches
        in guidance's working dtype; the rescale is not linear and runs on x0 as in the reference sampler.
        """
        g = ctx.guidance_cfg
        g.validate(
            supported_terms=("cfg", "stg", "modality", "mix", "rescale"),
            supports_cfg_skip_layers=True,
            network_name=type(self).__name__,
        )
        cond = ctx.predict("pos", condition=ctx.condition)
        guided = apply_cfg(ctx, cond)
        guided = apply_stg(ctx, guided)
        guided = apply_modality(ctx, guided)
        guided = apply_mix(ctx, guided)
        if g.rescale is not None:
            guided = self._rescale_denoised(ctx, guided, cond)
        return guided

    def _rescale_denoised(self, ctx: GuidanceContext, guided: TensorOrDict, cond: TensorOrDict) -> TensorOrDict:
        """``cfg_rescale`` of the x0 of ``guided`` toward that of ``cond``, returned as native velocities.

        The schedule converts both to x0, without the output constraint. ``forward`` expresses every position,
        conditioned frames included, against the scalar ``t``, and on the unshifted ``"rf"`` schedule ``sigma = t``,
        so a velocity ``v`` gives ``x0 = state - t * v`` and the rescaled x0 maps back as ``v - (x0' - x0) / t``.
        Where ``t`` is zero, x0 does not determine ``v`` and the prediction stays unrescaled; at the clean end the x0
        rescale is the identity.
        """
        if self.net_pred_type != "flow" or self.schedule_type != "rf":
            raise NotImplementedError(
                f"{type(self).__name__} maps its x0 rescale back to 'flow' predictions on the unshifted 'rf' "
                f"schedule, not {self.net_pred_type!r} predictions on {self.schedule_type!r}"
            )

        def to_x0(prediction: TensorOrDict) -> TensorOrDict:
            return self.noise_scheduler.convert_model_output(
                ctx.x_t, prediction, ctx.t, src_pred_type="flow", target_pred_type="x0", output_dtype=ctx.dtype
            )

        def correct(velocity, rescaled, x0):
            t = expand_like(ctx.t, velocity).to(dtype=velocity.dtype)
            noised = t > 0
            return torch.where(noised, velocity - (rescaled - x0) / torch.where(noised, t, 1.0), velocity)

        x0 = to_x0(guided)
        rescaled = apply_rescale(ctx, x0, to_x0(cond))
        return per_modality(correct, guided, rescaled, x0)

    def sample_scheduler_and_grid(
        self, sample_steps: int, device: Optional[torch.device] = None
    ) -> Tuple[BaseNoiseSchedule, torch.Tensor]:
        """Return the noise schedule, which takes the sampling steps, and the LTX-2 sigmas (``_ltx2_sigma_schedule``).

        The grid holds ``sample_steps + 1`` times in ``t_precision`` ending at the clean 0; the query times among
        them are clamped to the noise schedule. Needs ``sample_steps >= 2``.
        """
        ns = self.noise_scheduler
        grid = _ltx2_sigma_schedule(sample_steps).to(device=device, dtype=ns.t_precision)
        return ns, ns.clamp_grid(grid, warn=True)

    def sample(
        self,
        noise: TensorOrDict,
        condition: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]] = None,
        neg_condition: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]] = None,
        sample_steps: Optional[int] = None,
        sample_type: str = "ode",
        fps: Optional[float] = None,
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> TensorOrDict:
        """Denoise ``noise``; the sigma schedule, guidance and rescaling match the LTX-2 reference.

        A dict ``condition`` selects I2V: conditioned frames start from REF ``GaussianNoiser``'s
        ``noise * mask + clean * (1 - mask)`` with ``mask = 1 - strength``, the Euler step covers the full
        sequence, ``forward`` applies the per-token timesteps, and guidance applies the clean-frame blend.

        Args:
            noise: Noise for the sole configured modality, or a dict with ``"video"`` ``[B, C, F, H, W]``
                and/or ``"audio"`` ``[B, C, L, M]``.
            condition: T2V ``(prompt_embeds, attention_mask)``, or the I2V dict (see the class docstring).
            neg_condition: Negative condition for CFG, with the same structure.
            sample_steps: Denoising steps; ``None`` uses 30 for 2.3/2.5 and 40 for 2.0.
            sample_type: Only ``"ode"`` is supported.
            fps: Frame rate for RoPE coordinates; defaults to ``self.fps``.
            guidance_cfg: ``None`` uses ``default_guidance_cfg`` for the modalities in ``noise``; a supplied config
                is used verbatim and must match them.

        Returns:
            Denoised latents with the structure of ``noise``.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )
        if sample_steps is None:
            # LTX-2.5 follows LTX2Pipeline's 30-step default, same as LTX-2.3.
            sample_steps = 30 if self.ltx2_version in ("2.3", "2.5") else 40
        if fps is None:
            fps = self.fps

        active_modalities = list(noise) if isinstance(noise, dict) else [self.modalities[0]]

        batch_size = get_batch_size(noise)
        device = get_device(noise)

        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg(modalities=active_modalities, version=self.ltx2_version)

        # ---- Scheduler setup (matches original LTX-2 scheduler) ----
        ns, grid = self.sample_scheduler_and_grid(sample_steps, device=device)
        latents = ns.latents(noise, t_init=grid[0])

        # ---- I2V: initialize latents with the shared clean-latent blend ----
        # This is algebraically equivalent to REF ``GaussianNoiser``'s lerp. It is the
        # *only* place we set ``latents[cond] = clean`` in the stored sample
        # state; the loop's post-guidance x0 blend + Euler step then maintain
        # it bit-exactly at strength=1 (and REF-faithfully at strength<1).
        # Dispatched on condition shape (dict ⇒ I2V), mirroring ``forward``.
        # We also stamp ``preserve_cond_input=False`` onto shallow copies of
        # ``condition``/``neg_condition`` so the per-step forward skips the REF
        # training-style input replace (which would shortcut the trajectory at
        # strength<1 and is a no-op at strength=1). The shared guidance finalizer
        # applies ``post_process_latent`` once after rescaling, as in REF.
        if isinstance(condition, dict):
            video_init = latents["video"] if isinstance(latents, dict) else latents
            frame_idx, cond_latent, inv_strength = self._unpack_i2v_condition(condition)
            video_init[:, :, frame_idx] = self._blend_with_clean(video_init[:, :, frame_idx], cond_latent, inv_strength)
            condition = {**condition, "preserve_cond_input": False}
            if isinstance(neg_condition, dict):
                neg_condition = {**neg_condition, "preserve_cond_input": False}

        # ---- Denoising loop ----
        # Guidance composes the native velocities, whose I2V conditioned frames
        # ``forward`` expresses against the scalar t (per-token to_denoised), and
        # its final ``transform_output`` returns x0 with the scalar-t
        # ``flow_to_x0``, then the clean-frame blend.
        for t, t_next in zip(grid[:-1], grid[1:], strict=True):
            t = t.expand(batch_size)

            x0_pred = predict_with_guidance(
                self,
                latents,
                t,
                guidance_cfg=guidance_cfg,
                condition=condition,
                neg_condition=neg_condition,
                output_dtype=None,
                fps=fps,
                fwd_pred_type="x0",
            )
            # x0 → velocity → Euler step on FULL sequence, in high precision until the step returns
            # ``latents.dtype``.
            # In I2V mode at strength=1 guidance's output conditioning gives ``x0_pred[video] =
            # clean`` at conditioned positions, where ``latents[video] = clean``,
            # so ``flow = (clean - clean)/t = 0`` and the step leaves
            # cond positions clean bit-exactly.  Audio (and any other
            # non-conditioning modality) just steps normally.
            vel = ns.x0_to_flow(latents, x0_pred, t, output_dtype=None)
            latents = ns.step(latents, vel, t, t_next, pred_type="flow")

        return latents
