# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
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

import inspect
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import BaseOutput, apply_lora_scale, is_torch_version, logging
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings, PixArtAlphaTextProjection
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def apply_interleaved_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, C // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


def apply_split_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs

    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        b, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(f"Expected x.shape[-1] to be even for split rotary, got {last}.")
    r = last // 2

    split_x = x.reshape(*x.shape[:-1], 2, r).float()
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]

    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]

    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)

    out = out.reshape(*out.shape[:-2], last)

    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(b, t, -1)

    out = out.to(dtype=x_dtype)
    return out


@dataclass
class AudioVisualModelOutput(BaseOutput):
    r"""
    Holds the output of an audiovisual model which produces both visual (e.g. video) and audio outputs.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, num_frames, height, width)`):
            The hidden states output conditioned on the `encoder_hidden_states` input, representing the visual output
            of the model.
        audio_sample (`torch.Tensor` or `None`):
            The audio output of the audiovisual model. None when audio is disabled.
    """

    sample: "torch.Tensor"  # noqa: F821
    audio_sample: "torch.Tensor | None"  # noqa: F821


class LTX2AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://huggingface.co/papers/2310.00426; Section 2.3) and adapted by the LTX-2.0
    model.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_mod_params (`int`, *optional*, defaults to `6`):
            The number of modulation parameters.
        use_additional_conditions (`bool`, *optional*, defaults to `False`):
            Whether to use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, num_mod_params: int = 6, use_additional_conditions: bool = False):
        super().__init__()
        self.num_mod_params = num_mod_params

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, self.num_mod_params * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        batch_size: int | None = None,
        hidden_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        added_cond_kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class LTX2AudioVideoAttnProcessor:
    r"""
    Processor for implementing attention (SDPA) for the LTX-2.0 model.
    Supports separate RoPE embeddings for queries and keys (a2v / v2a cross attention).
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX attention processors require a minimum PyTorch version of 2.0. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "LTX2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if query_rotary_emb is not None:
            if attn.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )
            elif attn.rope_type == "split":
                query = apply_split_rotary_emb(query, query_rotary_emb)
                key = apply_split_rotary_emb(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )

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

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LTX2Attention(torch.nn.Module, AttentionModuleMixin):
    r"""
    Attention class for all LTX-2.0 attention layers.
    Supports separate query and key RoPE embeddings for a2v / v2a cross-attention.
    """

    _default_processor_cls = LTX2AudioVideoAttnProcessor
    _available_processors = [LTX2AudioVideoAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        kv_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        cross_attention_dim: int | None = None,
        out_bias: bool = True,
        qk_norm: str = "rms_norm_across_heads",
        norm_eps: float = 1e-6,
        norm_elementwise_affine: bool = True,
        rope_type: str = "interleaved",
        processor=None,
    ):
        super().__init__()
        if qk_norm != "rms_norm_across_heads":
            raise NotImplementedError("Only 'rms_norm_across_heads' is supported as a valid value for `qk_norm`.")

        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = query_dim
        self.heads = heads
        self.rope_type = rope_type

        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head * kv_heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        hidden_states = self.processor(
            self, hidden_states, encoder_hidden_states, attention_mask, query_rotary_emb, key_rotary_emb, **kwargs
        )
        return hidden_states


class LTX2VideoTransformerBlock(nn.Module):
    r"""
    Transformer block used in LTX-2.0.

    Supports two-level audio gating (Option C):
      - Construction-time: pass ``audio_dim=None`` to skip all audio module allocation.
      - Runtime: pass ``audio_enabled=False`` in ``forward()`` to skip audio ops this step.

    The a2v cross-attention (video attending to audio) is intentionally decoupled from
    ``audio_enabled`` — video can still attend to audio as conditioning even when the
    audio update branch is disabled, matching the original BasicAVTransformerBlock behaviour.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        # Audio args — all optional. Pass None to build a video-only block.
        audio_dim: int | None = None,
        audio_num_attention_heads: int | None = None,
        audio_attention_head_dim: int | None = None,
        audio_cross_attention_dim: int | None = None,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        rope_type: str = "interleaved",
    ):
        super().__init__()

        # Construction-time gate
        self.has_audio = audio_dim is not None

        if self.has_audio:
            assert audio_num_attention_heads is not None
            assert audio_attention_head_dim is not None
            assert audio_cross_attention_dim is not None

        # --- 1. Video Self-Attention (always built) ---
        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # --- 1b. Audio Self-Attention (conditional) ---
        if self.has_audio:
            self.audio_norm1 = RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
            self.audio_attn1 = LTX2Attention(
                query_dim=audio_dim,
                heads=audio_num_attention_heads,
                kv_heads=audio_num_attention_heads,
                dim_head=audio_attention_head_dim,
                bias=attention_bias,
                cross_attention_dim=None,
                out_bias=attention_out_bias,
                qk_norm=qk_norm,
                rope_type=rope_type,
            )

        # --- 2. Video Prompt Cross-Attention (always built) ---
        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = LTX2Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # --- 2b. Audio Prompt Cross-Attention (conditional) ---
        if self.has_audio:
            self.audio_norm2 = RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
            self.audio_attn2 = LTX2Attention(
                query_dim=audio_dim,
                cross_attention_dim=audio_cross_attention_dim,
                heads=audio_num_attention_heads,
                kv_heads=audio_num_attention_heads,
                dim_head=audio_attention_head_dim,
                bias=attention_bias,
                out_bias=attention_out_bias,
                qk_norm=qk_norm,
                rope_type=rope_type,
            )

        # --- 3. Audio-Video Cross-Attention (conditional — both modalities) ---
        if self.has_audio:
            # a2v: Q=Video, K/V=Audio
            self.audio_to_video_norm = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
            self.audio_to_video_attn = LTX2Attention(
                query_dim=dim,
                cross_attention_dim=audio_dim,
                heads=audio_num_attention_heads,
                kv_heads=audio_num_attention_heads,
                dim_head=audio_attention_head_dim,
                bias=attention_bias,
                out_bias=attention_out_bias,
                qk_norm=qk_norm,
                rope_type=rope_type,
            )

            # v2a: Q=Audio, K/V=Video
            self.video_to_audio_norm = RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
            self.video_to_audio_attn = LTX2Attention(
                query_dim=audio_dim,
                cross_attention_dim=dim,
                heads=audio_num_attention_heads,
                kv_heads=audio_num_attention_heads,
                dim_head=audio_attention_head_dim,
                bias=attention_bias,
                out_bias=attention_out_bias,
                qk_norm=qk_norm,
                rope_type=rope_type,
            )

            # Per-layer cross-attention modulation params
            self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, dim))
            self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, audio_dim))

        # --- 4. Feedforward (video always built, audio conditional) ---
        self.norm3 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.ff = FeedForward(dim, activation_fn=activation_fn)

        if self.has_audio:
            self.audio_norm3 = RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
            self.audio_ff = FeedForward(audio_dim, activation_fn=activation_fn)

        # --- 5. AdaLN modulation params (video always, audio conditional) ---
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        if self.has_audio:
            self.audio_scale_shift_table = nn.Parameter(torch.randn(6, audio_dim) / audio_dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor | None,
        temb: torch.Tensor,
        temb_audio: torch.Tensor | None,
        temb_ca_scale_shift: torch.Tensor | None,
        temb_ca_audio_scale_shift: torch.Tensor | None,
        temb_ca_gate: torch.Tensor | None,
        temb_ca_audio_gate: torch.Tensor | None,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        ca_video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        ca_audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        a2v_cross_attention_mask: torch.Tensor | None = None,
        v2a_cross_attention_mask: torch.Tensor | None = None,
        audio_enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = hidden_states.size(0)

        # Runtime gates — mirrors BasicAVTransformerBlock exactly
        # run_ax:  audio updates (self-attn, cross-attn, ffn, v2a)
        # run_a2v: video attending to audio — asymmetrically decoupled from audio_enabled
        #          so video can still use audio as conditioning even when audio_enabled=False
        # run_v2a: audio updates from video — tied to run_ax
        run_ax  = self.has_audio and audio_enabled and audio_hidden_states is not None
        run_a2v = self.has_audio and audio_hidden_states is not None
        run_v2a = run_ax

        # 1. Video Self-Attention (always runs)
        norm_hidden_states = self.norm1(hidden_states)
        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None].to(temb.device) + temb.reshape(
            batch_size, temb.size(1), num_ada_params, -1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            query_rotary_emb=video_rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        # 1b. Audio Self-Attention (conditional)
        if run_ax:
            norm_audio_hidden_states = self.audio_norm1(audio_hidden_states)
            num_audio_ada_params = self.audio_scale_shift_table.shape[0]
            audio_ada_values = self.audio_scale_shift_table[None, None].to(temb_audio.device) + temb_audio.reshape(
                batch_size, temb_audio.size(1), num_audio_ada_params, -1
            )
            audio_shift_msa, audio_scale_msa, audio_gate_msa, audio_shift_mlp, audio_scale_mlp, audio_gate_mlp = (
                audio_ada_values.unbind(dim=2)
            )
            norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_scale_msa) + audio_shift_msa
            attn_audio_hidden_states = self.audio_attn1(
                hidden_states=norm_audio_hidden_states,
                encoder_hidden_states=None,
                query_rotary_emb=audio_rotary_emb,
            )
            audio_hidden_states = audio_hidden_states + attn_audio_hidden_states * audio_gate_msa

        # 2. Video Cross-Attention with text (always runs)
        norm_hidden_states = self.norm2(hidden_states)
        attn_hidden_states = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states

        # 2b. Audio Cross-Attention with text (conditional)
        if run_ax:
            norm_audio_hidden_states = self.audio_norm2(audio_hidden_states)
            attn_audio_hidden_states = self.audio_attn2(
                norm_audio_hidden_states,
                encoder_hidden_states=audio_encoder_hidden_states,
                query_rotary_emb=None,
                attention_mask=audio_encoder_attention_mask,
            )
            audio_hidden_states = audio_hidden_states + attn_audio_hidden_states

        # 3. Audio-Video Cross-Attention
        if run_a2v or run_v2a:
            norm_hidden_states_av = self.audio_to_video_norm(hidden_states)
            norm_audio_hidden_states_av = self.video_to_audio_norm(audio_hidden_states)

            # Video modulation params
            video_per_layer_ca_scale_shift = self.video_a2v_cross_attn_scale_shift_table[:4, :]
            video_per_layer_ca_gate        = self.video_a2v_cross_attn_scale_shift_table[4:, :]
            video_ca_scale_shift_table = (
                video_per_layer_ca_scale_shift[:, :, ...].to(temb_ca_scale_shift.dtype)
                + temb_ca_scale_shift.reshape(batch_size, temb_ca_scale_shift.shape[1], 4, -1)
            ).unbind(dim=2)
            video_ca_gate = (
                video_per_layer_ca_gate[:, :, ...].to(temb_ca_gate.dtype)
                + temb_ca_gate.reshape(batch_size, temb_ca_gate.shape[1], 1, -1)
            ).unbind(dim=2)
            video_a2v_ca_scale, video_a2v_ca_shift, video_v2a_ca_scale, video_v2a_ca_shift = video_ca_scale_shift_table
            a2v_gate = video_ca_gate[0].squeeze(2)

            # Audio modulation params
            audio_per_layer_ca_scale_shift = self.audio_a2v_cross_attn_scale_shift_table[:4, :]
            audio_per_layer_ca_gate        = self.audio_a2v_cross_attn_scale_shift_table[4:, :]
            audio_ca_scale_shift_table = (
                audio_per_layer_ca_scale_shift[:, :, ...].to(temb_ca_audio_scale_shift.dtype)
                + temb_ca_audio_scale_shift.reshape(batch_size, temb_ca_audio_scale_shift.shape[1], 4, -1)
            ).unbind(dim=2)
            audio_ca_gate = (
                audio_per_layer_ca_gate[:, :, ...].to(temb_ca_audio_gate.dtype)
                + temb_ca_audio_gate.reshape(batch_size, temb_ca_audio_gate.shape[1], 1, -1)
            ).unbind(dim=2)
            audio_a2v_ca_scale, audio_a2v_ca_shift, audio_v2a_ca_scale, audio_v2a_ca_shift = audio_ca_scale_shift_table
            v2a_gate = audio_ca_gate[0].squeeze(2)

            # 3a. a2v: Q=Video, K/V=Audio (runs even when audio_enabled=False)
            if run_a2v:
                mod_norm_hidden_states = (
                    norm_hidden_states_av * (1 + video_a2v_ca_scale.squeeze(2))
                    + video_a2v_ca_shift.squeeze(2)
                )
                mod_norm_audio_hidden_states = (
                    norm_audio_hidden_states_av * (1 + audio_a2v_ca_scale.squeeze(2))
                    + audio_a2v_ca_shift.squeeze(2)
                )
                a2v_attn_hidden_states = self.audio_to_video_attn(
                    mod_norm_hidden_states,
                    encoder_hidden_states=mod_norm_audio_hidden_states,
                    query_rotary_emb=ca_video_rotary_emb,
                    key_rotary_emb=ca_audio_rotary_emb,
                    attention_mask=a2v_cross_attention_mask,
                )
                hidden_states = hidden_states + a2v_gate * a2v_attn_hidden_states

            # 3b. v2a: Q=Audio, K/V=Video (only when audio branch is fully active)
            if run_v2a:
                mod_norm_hidden_states = (
                    norm_hidden_states_av * (1 + video_v2a_ca_scale.squeeze(2))
                    + video_v2a_ca_shift.squeeze(2)
                )
                mod_norm_audio_hidden_states = (
                    norm_audio_hidden_states_av * (1 + audio_v2a_ca_scale.squeeze(2))
                    + audio_v2a_ca_shift.squeeze(2)
                )
                v2a_attn_hidden_states = self.video_to_audio_attn(
                    mod_norm_audio_hidden_states,
                    encoder_hidden_states=mod_norm_hidden_states,
                    query_rotary_emb=ca_audio_rotary_emb,
                    key_rotary_emb=ca_video_rotary_emb,
                    attention_mask=v2a_cross_attention_mask,
                )
                audio_hidden_states = audio_hidden_states + v2a_gate * v2a_attn_hidden_states

        # 4. Video Feedforward (always runs)
        norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + self.ff(norm_hidden_states) * gate_mlp

        # 4b. Audio Feedforward (conditional)
        if run_ax:
            norm_audio_hidden_states = (
                self.audio_norm3(audio_hidden_states) * (1 + audio_scale_mlp) + audio_shift_mlp
            )
            audio_hidden_states = audio_hidden_states + self.audio_ff(norm_audio_hidden_states) * audio_gate_mlp

        # Return None for audio when it didn't run — prevents stale tensor propagation
        return hidden_states, audio_hidden_states if run_ax else None


class LTX2AudioVideoRotaryPosEmbed(nn.Module):
    """
    Video and audio rotary positional embeddings (RoPE) for the LTX-2.0 model.
    """

    def __init__(
        self,
        dim: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        scale_factors: tuple[int, ...] = (8, 32, 32),
        theta: float = 10000.0,
        causal_offset: int = 1,
        modality: str = "video",
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t

        if rope_type not in ["interleaved", "split"]:
            raise ValueError(f"{rope_type=} not supported. Choose between 'interleaved' and 'split'.")
        self.rope_type = rope_type

        self.base_num_frames = base_num_frames
        self.num_attention_heads = num_attention_heads

        self.base_height = base_height
        self.base_width = base_width

        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.audio_latents_per_second = float(sampling_rate) / float(hop_length) / float(scale_factors[0])

        self.scale_factors = scale_factors
        self.theta = theta
        self.causal_offset = causal_offset

        self.modality = modality
        if self.modality not in ["video", "audio"]:
            raise ValueError(f"Modality {modality} is not supported. Supported modalities are `video` and `audio`.")
        self.double_precision = double_precision

    def prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        fps: float = 24.0,
    ) -> torch.Tensor:
        grid_f = torch.arange(start=0, end=num_frames, step=self.patch_size_t, dtype=torch.float32, device=device)
        grid_h = torch.arange(start=0, end=height, step=self.patch_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(start=0, end=width, step=self.patch_size, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)

        patch_size = (self.patch_size_t, self.patch_size, self.patch_size)
        patch_size_delta = torch.tensor(patch_size, dtype=grid.dtype, device=grid.device)
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

        latent_coords = torch.stack([grid, patch_ends], dim=-1)
        latent_coords = latent_coords.flatten(1, 3)
        latent_coords = latent_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        scale_tensor = torch.tensor(self.scale_factors, device=latent_coords.device)
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)

        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + self.causal_offset - self.scale_factors[0]).clamp(min=0)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

        return pixel_coords

    def prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        shift: int = 0,
    ) -> torch.Tensor:
        grid_f = torch.arange(
            start=shift, end=num_frames + shift, step=self.patch_size_t, dtype=torch.float32, device=device
        )

        audio_scale_factor = self.scale_factors[0]
        grid_start_mel = grid_f * audio_scale_factor
        grid_start_mel = (grid_start_mel + self.causal_offset - audio_scale_factor).clip(min=0)
        grid_start_s = grid_start_mel * self.hop_length / self.sampling_rate

        grid_end_mel = (grid_f + self.patch_size_t) * audio_scale_factor
        grid_end_mel = (grid_end_mel + self.causal_offset - audio_scale_factor).clip(min=0)
        grid_end_s = grid_end_mel * self.hop_length / self.sampling_rate

        audio_coords = torch.stack([grid_start_s, grid_end_s], dim=-1)
        audio_coords = audio_coords.unsqueeze(0).expand(batch_size, -1, -1)
        audio_coords = audio_coords.unsqueeze(1)
        return audio_coords

    def prepare_coords(self, *args, **kwargs):
        if self.modality == "video":
            return self.prepare_video_coords(*args, **kwargs)
        elif self.modality == "audio":
            return self.prepare_audio_coords(*args, **kwargs)

    def forward(
        self, coords: torch.Tensor, device: str | torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or coords.device

        num_pos_dims = coords.shape[1]

        if coords.ndim == 4:
            coords_start, coords_end = coords.chunk(2, dim=-1)
            coords = (coords_start + coords_end) / 2.0
            coords = coords.squeeze(-1)

        if self.modality == "video":
            max_positions = (self.base_num_frames, self.base_height, self.base_width)
        elif self.modality == "audio":
            max_positions = (self.base_num_frames,)
        grid = torch.stack([coords[:, i] / max_positions[i] for i in range(num_pos_dims)], dim=-1).to(device)
        num_rope_elems = num_pos_dims * 2

        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(start=0.0, end=1.0, steps=self.dim // num_rope_elems, dtype=freqs_dtype, device=device),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)

        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs
        freqs = freqs.transpose(-1, -2).flatten(2)

        if self.rope_type == "interleaved":
            cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
            sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

            if self.dim % num_rope_elems != 0:
                cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % num_rope_elems])
                sin_padding = torch.zeros_like(cos_freqs[:, :, : self.dim % num_rope_elems])
                cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
                sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        elif self.rope_type == "split":
            expected_freqs = self.dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq = freqs.cos()
            sin_freq = freqs.sin()

            if pad_size != 0:
                cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
                cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
                sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

            b = cos_freq.shape[0]
            t = cos_freq.shape[1]

            cos_freq = cos_freq.reshape(b, t, self.num_attention_heads, -1)
            sin_freq = sin_freq.reshape(b, t, self.num_attention_heads, -1)

            cos_freqs = torch.swapaxes(cos_freq, 1, 2)
            sin_freqs = torch.swapaxes(sin_freq, 1, 2)

        return cos_freqs, sin_freqs


class LTX2VideoTransformer3DModel(
    ModelMixin, ConfigMixin, AttentionMixin, FromOriginalModelMixin, PeftAdapterMixin, CacheMixin
):
    r"""
    A Transformer model for video-like data used in LTX-2.0.

    Supports two-level audio gating (Option C):
      - Construction-time: set ``audio_enabled=False`` to build a video-only model with
        no audio weights allocated at all. Ideal for finetuning video-only.
      - Runtime: pass ``audio_hidden_states=None`` to skip all audio ops for a given
        forward pass on a full AV checkpoint.

    When loading a pretrained AV checkpoint into a video-only model, use ``strict=False``:
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        # unexpected = all audio.* keys — expected, safe to ignore
        # missing    = should be [] — all video keys present
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["LTX2VideoTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_attention_mask": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int | None = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        vae_scale_factors: tuple[int, int, int] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        audio_in_channels: int = 128,
        audio_out_channels: int | None = 128,
        audio_patch_size: int = 1,
        audio_patch_size_t: int = 1,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_cross_attention_dim: int = 2048,
        audio_scale_factor: int = 4,
        audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000,
        audio_hop_length: int = 160,
        num_layers: int = 48,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        timestep_scale_multiplier: int = 1000,
        cross_attn_timestep_scale_multiplier: int = 1000,
        rope_type: str = "interleaved",
        audio_enabled: bool = True,  # <-- NEW: construction-time gate
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        audio_out_channels = audio_out_channels or audio_in_channels
        inner_dim = num_attention_heads * attention_head_dim
        audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim

        # 1. Patchification input projections
        self.proj_in = nn.Linear(in_channels, inner_dim)
        if audio_enabled:
            self.audio_proj_in = nn.Linear(audio_in_channels, audio_inner_dim)

        # 2. Prompt embeddings
        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
        if audio_enabled:
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels, hidden_size=audio_inner_dim
            )

        # 3. Timestep modulation
        self.time_embed = LTX2AdaLayerNormSingle(inner_dim, num_mod_params=6, use_additional_conditions=False)
        if audio_enabled:
            self.audio_time_embed = LTX2AdaLayerNormSingle(
                audio_inner_dim, num_mod_params=6, use_additional_conditions=False
            )
            self.av_cross_attn_video_scale_shift = LTX2AdaLayerNormSingle(
                inner_dim, num_mod_params=4, use_additional_conditions=False
            )
            self.av_cross_attn_audio_scale_shift = LTX2AdaLayerNormSingle(
                audio_inner_dim, num_mod_params=4, use_additional_conditions=False
            )
            self.av_cross_attn_video_a2v_gate = LTX2AdaLayerNormSingle(
                inner_dim, num_mod_params=1, use_additional_conditions=False
            )
            self.av_cross_attn_audio_v2a_gate = LTX2AdaLayerNormSingle(
                audio_inner_dim, num_mod_params=1, use_additional_conditions=False
            )

        # 3.3. Output layer modulation params
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        if audio_enabled:
            self.audio_scale_shift_table = nn.Parameter(torch.randn(2, audio_inner_dim) / audio_inner_dim**0.5)

        # 4. RoPE — video always built, audio ropes only when audio_enabled
        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=inner_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=vae_scale_factors,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        if audio_enabled:
            self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
                dim=audio_inner_dim,
                patch_size=audio_patch_size,
                patch_size_t=audio_patch_size_t,
                base_num_frames=audio_pos_embed_max_pos,
                sampling_rate=audio_sampling_rate,
                hop_length=audio_hop_length,
                scale_factors=[audio_scale_factor],
                theta=rope_theta,
                causal_offset=causal_offset,
                modality="audio",
                double_precision=rope_double_precision,
                rope_type=rope_type,
                num_attention_heads=audio_num_attention_heads,
            )
            cross_attn_pos_embed_max_pos = max(pos_embed_max_pos, audio_pos_embed_max_pos)
            self.cross_attn_rope = LTX2AudioVideoRotaryPosEmbed(
                dim=audio_cross_attention_dim,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
                base_num_frames=cross_attn_pos_embed_max_pos,
                base_height=base_height,
                base_width=base_width,
                theta=rope_theta,
                causal_offset=causal_offset,
                modality="video",
                double_precision=rope_double_precision,
                rope_type=rope_type,
                num_attention_heads=num_attention_heads,
            )
            self.cross_attn_audio_rope = LTX2AudioVideoRotaryPosEmbed(
                dim=audio_cross_attention_dim,
                patch_size=audio_patch_size,
                patch_size_t=audio_patch_size_t,
                base_num_frames=cross_attn_pos_embed_max_pos,
                sampling_rate=audio_sampling_rate,
                hop_length=audio_hop_length,
                theta=rope_theta,
                causal_offset=causal_offset,
                modality="audio",
                double_precision=rope_double_precision,
                rope_type=rope_type,
                num_attention_heads=audio_num_attention_heads,
            )

        # 5. Transformer blocks — pass None dims when audio disabled
        self.transformer_blocks = nn.ModuleList(
            [
                LTX2VideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    audio_dim=audio_inner_dim if audio_enabled else None,
                    audio_num_attention_heads=audio_num_attention_heads if audio_enabled else None,
                    audio_attention_head_dim=audio_attention_head_dim if audio_enabled else None,
                    audio_cross_attention_dim=audio_cross_attention_dim if audio_enabled else None,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    rope_type=rope_type,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Output layers
        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)
        if audio_enabled:
            self.audio_norm_out = nn.LayerNorm(audio_inner_dim, eps=1e-6, elementwise_affine=False)
            self.audio_proj_out = nn.Linear(audio_inner_dim, audio_out_channels)

        self.gradient_checkpointing = False

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor = None,
        audio_encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.LongTensor = None,
        audio_timestep: torch.LongTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        audio_num_frames: int | None = None,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Single runtime gate — all audio ops key off this.
        # Requires: modules exist (construction-time) AND caller supplied a tensor.
        run_audio = self.config.audio_enabled and audio_hidden_states is not None

        audio_timestep = audio_timestep if audio_timestep is not None else timestep

        # Attention mask conversion
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if run_audio and audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = (
                1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)
            ) * -10000.0
            audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)

        # 1. RoPE — video always, audio only when run_audio
        if video_coords is None:
            video_coords = self.rope.prepare_video_coords(
                batch_size, num_frames, height, width, hidden_states.device, fps=fps
            )
        video_rotary_emb = self.rope(video_coords, device=hidden_states.device)

        if run_audio:
            if audio_coords is None:
                audio_coords = self.audio_rope.prepare_audio_coords(
                    batch_size, audio_num_frames, audio_hidden_states.device
                )
            audio_rotary_emb = self.audio_rope(audio_coords, device=audio_hidden_states.device)
            video_cross_attn_rotary_emb = self.cross_attn_rope(
                video_coords[:, 0:1, :], device=hidden_states.device
            )
            audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
                audio_coords[:, 0:1, :], device=audio_hidden_states.device
            )
        else:
            audio_rotary_emb = None
            video_cross_attn_rotary_emb = None
            audio_cross_attn_rotary_emb = None

        # 2. Patchify
        hidden_states = self.proj_in(hidden_states)
        if run_audio:
            audio_hidden_states = self.audio_proj_in(audio_hidden_states)

        # 3. Timestep embeddings and modulation params
        timestep_cross_attn_gate_scale_factor = (
            self.config.cross_attn_timestep_scale_multiplier / self.config.timestep_scale_multiplier
        )

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        if run_audio:
            temb_audio, audio_embedded_timestep = self.audio_time_embed(
                audio_timestep.flatten(), batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype,
            )
            temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
            audio_embedded_timestep = audio_embedded_timestep.view(
                batch_size, -1, audio_embedded_timestep.size(-1)
            )

            video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype,
            )
            video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
                timestep.flatten() * timestep_cross_attn_gate_scale_factor,
                batch_size=batch_size, hidden_dtype=hidden_states.dtype,
            )
            video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
                batch_size, -1, video_cross_attn_scale_shift.shape[-1]
            )
            video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(
                batch_size, -1, video_cross_attn_a2v_gate.shape[-1]
            )

            audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
                audio_timestep.flatten(), batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype,
            )
            audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
                audio_timestep.flatten() * timestep_cross_attn_gate_scale_factor,
                batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype,
            )
            audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
                batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
            )
            audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(
                batch_size, -1, audio_cross_attn_v2a_gate.shape[-1]
            )
        else:
            temb_audio = None
            audio_embedded_timestep = None
            video_cross_attn_scale_shift = None
            video_cross_attn_a2v_gate = None
            audio_cross_attn_scale_shift = None
            audio_cross_attn_v2a_gate = None

        # 4. Prompt embeddings
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))
        if run_audio:
            audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
            audio_encoder_hidden_states = audio_encoder_hidden_states.view(
                batch_size, -1, audio_hidden_states.size(-1)
            )

        # 5. Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # Note: _gradient_checkpointing_func only accepts positional tensor args.
                # audio_enabled is not passed explicitly here — the block derives
                # run_ax from audio_hidden_states being None when run_audio=False.
                hidden_states, audio_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    audio_hidden_states,
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    temb,
                    temb_audio,
                    video_cross_attn_scale_shift,
                    audio_cross_attn_scale_shift,
                    video_cross_attn_a2v_gate,
                    audio_cross_attn_v2a_gate,
                    video_rotary_emb,
                    audio_rotary_emb,
                    video_cross_attn_rotary_emb,
                    audio_cross_attn_rotary_emb,
                    encoder_attention_mask,
                    audio_encoder_attention_mask,
                )
            else:
                hidden_states, audio_hidden_states = block(
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
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=video_cross_attn_rotary_emb,
                    ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                    audio_enabled=run_audio,
                )

        # 6. Output layers
        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        if run_audio:
            audio_scale_shift_values = (
                self.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
            )
            audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]
            audio_hidden_states = self.audio_norm_out(audio_hidden_states)
            audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
            audio_output = self.audio_proj_out(audio_hidden_states)
        else:
            audio_output = None

        if not return_dict:
            return (output, audio_output)
        return AudioVisualModelOutput(sample=output, audio_sample=audio_output)