# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""FastGen network for NVIDIA Cosmos3 (video generation tower).

Cosmos3 is an omni-modal Mixture-of-Transformers. FastGen integrates the *video*
generation pathway only: a single Qwen3-VL transformer processes one packed sequence
holding the prompt's text token-IDs (understanding / causal) followed by the noised VAE
vision latents (generation / full-attention). Conditioning is therefore the prompt token
IDs (embedded inline via the shared ``embed_tokens`` and attended to by the vision tokens),
NOT a separate cross-attention embedding tensor. The backbone (``Cosmos3Backbone``) lives in
``modules.py``; this file is the ``FastGenNetwork`` wrapper providing forward / sample /
conditioning / weight-loading / FSDP, mirroring ``cosmos_predict2``.

Diffusion: rectified flow (``net_pred_type="flow"``, ``schedule_type="rf"``); velocity is
predicted only on noisy frames (zero on the clean conditioning frame for I2V). Conditioning
modes: T2V, and single-anchor-frame I2V via ``is_i2v`` (multi-frame V2V conditioning
is not implemented yet). VAE: Wan2.2-TI2V-5B (48-channel), reused via ``WanVideoEncoder``.
"""

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch.distributed.fsdp import fully_shard
from diffusers import UniPCMultistepScheduler

from fastgen.networks.network import FastGenNetwork, FastGenModule, validate_conditioning
from fastgen.networks.Wan import WanVideoEncoder
from fastgen.networks.cosmos3.modules import (
    Cosmos3Backbone,
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)
import fastgen.utils.logging_utils as logger
from fastgen.utils.tensor import assert_tensor
from fastgen.utils.tree_utils import OutputDtype, assert_same
from fastgen.utils.distributed.fsdp import make_fully_shard_kwargs
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.io_utils import load_checkpoint_state_dict
from fastgen.utils.module import load_module_state_dict, module_state_names
from fastgen.utils.env import hf_download, hf_load


# =====================================================================================
# Text tokenizer. Cosmos3 has no separate text encoder: the joint transformer embeds raw
# Qwen2 token IDs. This wrapper applies the Cosmos3 chat template + metadata templates and
# returns token-ID lists. It is exposed as ``text_tokenizer`` — NOT ``text_encoder`` — because
# Cosmos3's complete builder owns geometry/FPS-aware positive and negative templates.
# =====================================================================================

_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."
# Metadata attached to the negative prompt: none, the positive templates ("same", the native
# cosmos_framework inference default), or the "...is not..." inverse templates (the diffusers
# Cosmos3OmniPipeline default).
_NEGATIVE_METADATA_MODES = ("none", "same", "inverse")


class Cosmos3TextTokenizer(FastGenModule):
    """Qwen2 chat-template tokenizer that produces Cosmos3 prompt token IDs.

    Mirrors the native ``cosmos_framework`` prompt construction: the optional system prompt (off by
    default, as in every native training config), the duration / resolution metadata templates
    (``negative_metadata_mode`` selects the negative prompt's variant), and appends ``<eos>`` +
    ``<|vision_start|>``. ``encode`` returns a list of 1-D ``LongTensor`` token-ID sequences (one
    per prompt) — variable length, packed per-sample downstream.
    """

    def __init__(self, model_name_or_path: str, subfolder: str = "text_tokenizer"):
        super().__init__()
        from transformers import AutoTokenizer

        path = model_name_or_path
        sub = (
            subfolder
            if (subfolder and not os.path.isfile(os.path.join(model_name_or_path, "tokenizer_config.json")))
            else None
        )
        # Prefer the dedicated text_tokenizer subfolder when present.
        if os.path.isdir(os.path.join(model_name_or_path, subfolder)):
            path, sub = os.path.join(model_name_or_path, subfolder), None
        # transformers 5 no longer tolerates an explicit subfolder=None in
        # its local-path resolution. Omit the keyword when the tokenizer path
        # was already resolved to the dedicated subdirectory.
        subfolder_kwargs = {} if sub is None else {"subfolder": sub}
        self.tokenizer = hf_load(AutoTokenizer.from_pretrained, path, **subfolder_kwargs)
        self.start_of_generation_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.eos_token_id = self.tokenizer.eos_token_id
        self.duration_template = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
        self.image_resolution_template = "This image is of {height}x{width} resolution."
        self.video_resolution_template = "This video is of {height}x{width} resolution."
        self.inverse_duration_template = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
        self.inverse_image_resolution_template = "This image is not of {height}x{width} resolution."
        self.inverse_video_resolution_template = "This video is not of {height}x{width} resolution."

    def tokenize_prompt(
        self,
        prompt: str,
        is_image: bool = False,
        is_negative: bool = False,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        use_system_prompt: bool = False,
        add_resolution_template: bool = True,
        add_duration_template: bool = True,
        negative_metadata_mode: str = "same",
    ) -> List[int]:
        if negative_metadata_mode not in _NEGATIVE_METADATA_MODES:
            raise ValueError(
                f"negative_metadata_mode must be one of {_NEGATIVE_METADATA_MODES}, got {negative_metadata_mode!r}"
            )
        if is_negative and negative_metadata_mode == "none":
            add_resolution_template = add_duration_template = False
        inverse = is_negative and negative_metadata_mode == "inverse"
        res_t = self.image_resolution_template if is_image else self.video_resolution_template
        inv_res_t = self.inverse_image_resolution_template if is_image else self.inverse_video_resolution_template

        def _append(base: str, addition: str) -> str:
            base = base.rstrip(".")
            return f"{base}. {addition}" if base else addition

        text = prompt
        if not is_image and add_duration_template:
            dt = self.inverse_duration_template if inverse else self.duration_template
            text = _append(text, dt.format(duration=num_frames / fps, fps=fps))
        if add_resolution_template:
            rt = inv_res_t if inverse else res_t
            text = _append(text, rt.format(height=height, width=width))

        conversations = []
        if use_system_prompt:
            conversations.append(
                {"role": "system", "content": _SYSTEM_PROMPT_IMAGE if is_image else _SYSTEM_PROMPT_VIDEO}
            )
        conversations.append({"role": "user", "content": text})
        enc = self.tokenizer.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, add_vision_id=False, return_dict=True
        )
        return list(enc["input_ids"]) + [self.eos_token_id, self.start_of_generation_id]

    def encode(
        self,
        conditioning: Union[str, List[str]],
        is_negative: bool = False,
        fps: float = 24.0,
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        """Tokenize prompt(s) into ``LongTensor`` sequences, one per prompt."""
        if isinstance(conditioning, str):
            conditioning = [conditioning]
        ids = [
            torch.tensor(
                self.tokenize_prompt(p, is_negative=is_negative, fps=fps, **kwargs),
                dtype=torch.long,
                device=self.device,
            )
            for p in conditioning
        ]
        return self(ids)

    def forward(self, input_ids: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        return {"input_ids": input_ids}


# =====================================================================================
# Main network
# =====================================================================================


class Cosmos3(FastGenNetwork):
    """Cosmos3 video-generation network (text->video / ->image / image->video)."""

    MODEL_REVISION = "03c14e74a6ddb51985d614b75d70f2443efc6a05"
    TRANSFORMER_INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"
    NATIVE_NUM_TRAIN_TIMESTEPS = 1000
    # Native sampling shift per resolution tier for the 2B/8B (Edge/Nano) generators, matched on the
    # shorter clip side (cosmos_framework ``_RESOLUTION_SHIFT_DEFAULTS``; the 32B Super model uses 5 throughout).
    RESOLUTION_SHIFT_DEFAULTS: Dict[int, float] = {256: 3.0, 480: 5.0, 720: 10.0}
    # The Wan2.2 VAE the checkpoint ships.
    latent_compression_factors = (4, 16, 16)

    def __init__(
        self,
        # Backbone architecture (released Cosmos3-Nano 8B-class generator defaults)
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 12288,
        latent_channel: int = 48,
        latent_patch_size: int = 2,
        patch_latent_dim: int = 192,
        vocab_size: int = 151936,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 5000000.0,
        rope_axes_dim: Tuple[int, int, int] = (24, 20, 20),
        temporal_modality_margin: int = 15000,
        reset_spatial_indices: bool = True,
        enable_fps_modulation: bool = True,
        base_fps: int = 24,
        enable_logvar_linear: bool = False,
        # FastGen / noise schedule
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        # Encoders
        # Released repo root: supplies the transformer checkpoint, VAE, and text tokenizer.
        model_id_or_local_path: str = "nvidia/Cosmos3-Nano",
        revision: Optional[str] = MODEL_REVISION,
        load_pretrained: bool = True,
        # Generation defaults
        fps: float = 24.0,
        is_i2v: bool = False,
        # Prompt construction, native cosmos_framework defaults: no system prompt (every native training
        # config sets use_system_prompt=False) and the negative prompt carries the same duration/resolution
        # metadata as the positive one. The diffusers Cosmos3OmniPipeline instead defaults to a system
        # prompt and "inverse" ("...is not...") negative templates.
        use_system_prompt: bool = False,
        negative_metadata_mode: str = "same",
        gradient_checkpointing: bool = False,
        freeze_understanding_tower: bool = False,
        keep_time_embedder_fp32: bool = True,
        **schedule_kwargs,
    ):
        # The native shifted grids start above the RF default of 0.999, and student grids start at max_t like the
        # DMD2 cookbook's t = 1, so the raw-time bound defaults to 1.0 (configs may lower it).
        schedule_kwargs.setdefault("max_t", 1.0)
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)
        if negative_metadata_mode not in _NEGATIVE_METADATA_MODES:
            raise ValueError(
                f"negative_metadata_mode must be one of {_NEGATIVE_METADATA_MODES}, got {negative_metadata_mode!r}"
            )
        self.use_system_prompt = use_system_prompt
        self.negative_metadata_mode = negative_metadata_mode

        self.enable_fps_modulation = enable_fps_modulation
        self.temporal_modality_margin = temporal_modality_margin
        self.reset_spatial_indices = reset_spatial_indices
        self.base_fps = base_fps
        self.latent_patch_size = latent_patch_size
        self.fps = fps
        self.is_i2v = is_i2v
        self.freeze_understanding_tower = freeze_understanding_tower
        self.keep_time_embedder_fp32 = keep_time_embedder_fp32
        self.model_id_or_local_path = model_id_or_local_path
        self.revision = revision

        self.transformer = Cosmos3Backbone(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            latent_channel=latent_channel,
            latent_patch_size=latent_patch_size,
            patch_latent_dim=patch_latent_dim,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_axes_dim=rope_axes_dim,
            enable_logvar_linear=enable_logvar_linear,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.sample_scheduler = None
        if load_pretrained:
            # Resolve the immutable component root on every rank. FSDP only synchronizes tensors;
            # leaving this as a Hub ID on meta ranks would make their tokenizer/VAE resolve unpinned
            # ``main`` while rank 0 uses the requested revision's local snapshot.
            self.model_id_or_local_path = hf_download(
                self.model_id_or_local_path,
                revision=self.revision,
                allow_patterns=["transformer/*", "text_tokenizer/*", "vae/*"],
            )
            if self._is_in_meta_context():
                logger.info("Skipping Cosmos3 weight loading on a meta-initialized rank")
            else:
                self._load_pretrained_transformer()

    # ----- packing -----

    def _build_sample_inputs(
        self,
        input_ids: torch.Tensor,
        latent_shape: Tuple[int, int, int, int],
        has_image_condition: bool,
        fps: float,
        device,
    ) -> Dict[str, Any]:
        """Build the static (timestep-invariant) packed-sequence fields for one sample.

        ``latent_shape`` is ``(C, T, H, W)`` of the VAE latent. Mirrors the per-segment
        construction in ``Cosmos3OmniPipeline`` (text segment then vision segment, with the
        ``temporal_modality_margin`` gap between them).
        """
        und_len = int(input_ids.shape[0])
        text_ids, next_off = get_3d_mrope_ids_text_tokens(und_len, 0.0, use_float_positions=self.enable_fps_modulation)
        vision_offset = next_off + self.temporal_modality_margin

        c, t, h, w = latent_shape
        p = self.latent_patch_size
        ph = -(-h // p)  # ceil
        pw = -(-w // p)
        num_vis = t * ph * pw
        noisy_start = 1 if has_image_condition else 0
        noisy_frames = torch.arange(noisy_start, t, device=device, dtype=torch.long)
        stride = ph * pw
        mse_idx = []
        for f in range(noisy_start, t):
            s = und_len + f * stride
            mse_idx.extend(range(s, s + stride))
        vis_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=t,
            grid_h=ph,
            grid_w=pw,
            temporal_offset=vision_offset,
            reset_spatial_indices=self.reset_spatial_indices,
            fps=fps if self.enable_fps_modulation else None,
            base_fps=float(self.base_fps),
            temporal_compression_factor=self.latent_compression_factors[0],
        )
        position_ids = torch.cat([text_ids, vis_ids], dim=1).to(device)
        return {
            "input_ids": input_ids.to(device),
            "text_indexes": torch.arange(und_len, device=device, dtype=torch.long),
            "und_len": und_len,
            "sequence_length": und_len + num_vis,
            "vision_token_shapes": [(t, ph, pw)],
            "vision_sequence_indexes": torch.arange(und_len, und_len + num_vis, device=device, dtype=torch.long),
            "vision_mse_loss_indexes": torch.tensor(mse_idx, device=device, dtype=torch.long),
            "vision_noisy_frame_indexes": [noisy_frames],
            "position_ids": position_ids,
            "num_noisy_vision_tokens": (t - noisy_start) * stride,
        }

    @staticmethod
    def _extract_condition(condition: Any) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """Return ``(input_ids_list, condition_mask)`` from a condition object."""
        if isinstance(condition, Mapping):
            return condition.get("input_ids"), condition.get("condition_mask")
        return condition, None

    # ----- forward -----

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Mapping[str, Any]] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        fps: Optional[Union[float, torch.Tensor]] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Predict velocity for the noised vision latents ``x_t`` ``[B, C, T, H, W]``.

        ``condition`` carries the prompt token IDs (``{"input_ids": [LongTensor]*B}``) and an
        optional ``condition_mask`` ``[B, 1, T, 1, 1]`` (I2V clean frames). Rotary positions use
        ``fps`` (one value for the batch or one per sample) when given, else the network's
        configured frame rate. Runs the packed transformer once per batch item (the joint
        sequence is inherently single-sample). ``apply_output_conditioning``, which pins the I2V
        frames, and ``output_dtype`` follow ``FastGenNetwork.forward``.
        """
        self._validate_forward_kwargs(
            fwd_kwargs,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            num_features=len(self.transformer.layers),
        )
        if return_logvar and self.transformer.logvar_linear is None:
            raise RuntimeError("return_logvar=True requires enable_logvar_linear=True")
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        input_ids_list, condition_mask = self._extract_condition(condition)
        assert input_ids_list is not None, "Cosmos3.forward requires condition['input_ids']"
        # I2V: the reference feeds conditioning frames as clean latents at sigma 0 (it zeroes their noise), so
        # replace them in the model input here and pin them in the output below.
        x_t_model = self.cast_to_model_precision(self.preserve_conditioning(x_t, condition))
        assert all(
            ids.dtype in (torch.int32, torch.int64) for ids in input_ids_list
        ), "Cosmos3 input_ids must contain integer token IDs"

        B, C, T, H, W = x_t_model.shape
        device = x_t_model.device
        p_ = self.latent_patch_size
        ph, pw = (H + p_ - 1) // p_, (W + p_ - 1) // p_  # patch grid (for reshaping gen features)

        def _featmap(stacked: torch.Tensor) -> torch.Tensor:
            # gen-stream tokens [B, T*ph*pw, D] (T-major) -> [B, D, T, ph, pw] for Discriminator_VideoDiT
            return stacked.reshape(B, T, ph, pw, stacked.shape[-1]).permute(0, 4, 1, 2, 3).contiguous()

        # FastGen t in [0, 1] feeds the backbone's timestep embedding directly: the released
        # pipeline's discrete [0, 1000] timesteps were multiplied by timestep_scale=0.001
        # inside the backbone anyway, so the rescale round-trip is dropped entirely.
        if t.ndim == 0:
            t = t.expand(B)
        # The timestep embedding and logvar are computed INSIDE the backbone's forward
        # (see `Cosmos3Backbone.forward`): FSDP2 only all-gathers a unit's
        # parameters in its forward pre-hook, so `time_embedder` / `logvar_linear` cannot
        # be reached through a plain method call on a sharded backbone.

        velocities: List[torch.Tensor] = []
        feats_per_sample: List[List[torch.Tensor]] = []
        logvars: List[torch.Tensor] = []
        has_img_cond = condition_mask is not None
        fps_vals = [self.fps] if fps is None else fps.flatten().tolist() if torch.is_tensor(fps) else [fps]
        if len(fps_vals) == 1:
            fps_vals *= B
        assert len(fps_vals) == B, f"fps needs one value or one per sample, got {len(fps_vals)} for batch size {B}"
        for b in range(B):
            packed = self._build_sample_inputs(
                input_ids_list[b], (C, T, H, W), has_image_condition=has_img_cond, fps=float(fps_vals[b]), device=device
            )
            preds, feats, logvar_b = self.transformer(
                input_ids=packed["input_ids"],
                text_indexes=packed["text_indexes"],
                position_ids=packed["position_ids"],
                und_len=packed["und_len"],
                sequence_length=packed["sequence_length"],
                vision_tokens=[x_t_model[b : b + 1]],
                vision_token_shapes=packed["vision_token_shapes"],
                vision_sequence_indexes=packed["vision_sequence_indexes"],
                vision_mse_loss_indexes=packed["vision_mse_loss_indexes"],
                num_noisy_vision_tokens=packed["num_noisy_vision_tokens"],
                vision_noisy_frame_indexes=packed["vision_noisy_frame_indexes"],
                t=t[b : b + 1],
                feature_indices=feature_indices,
                return_features_early=return_features_early,
                return_logvar=return_logvar,
            )
            if return_logvar:
                logvars.append(logvar_b)
            if return_features_early:
                feats_per_sample.append(feats)
                continue
            velocities.append(preds[0])  # [1, C, T, H, W]
            if feature_indices is not None:
                feats_per_sample.append(feats)

        if return_features_early:
            return [
                _featmap(torch.stack([feats_per_sample[b][i] for b in range(B)], dim=0))
                for i in range(len(feature_indices))
            ]

        out = torch.cat(velocities, dim=0)  # velocity, [B, C, T, H, W]
        out = self.transform_output(out, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning)

        if feature_indices is not None:
            features = [
                _featmap(torch.stack([feats_per_sample[b][i] for b in range(B)], dim=0))
                for i in range(len(feature_indices))
            ]
            out = [out, features]

        if return_logvar:
            # [B, 1], one row per sample forward.
            return out, torch.cat(logvars, dim=0).to(dtype=x_t.dtype)
        return out

    # ----- conditioning -----

    def text_encode_kwargs(self, key: str, input_shape: Sequence[int]) -> Dict[str, Any]:
        """Per-key kwargs for ``text_tokenizer.encode`` in ``build_conditioning``.

        Cosmos3's tokenizer bakes duration/resolution/fps templates into the prompt text and
        selects the negative prompt's template variant (``negative_metadata_mode``), so it needs the
        REAL clip geometry + ``is_negative`` rather than the hardcoded encode defaults
        (189f/720/24fps). Deriving from ``input_shape`` (= ``[C, T_latent, H_latent,
        W_latent]``) keeps the text metadata consistent with the actual latent for ANY
        configured geometry. ``fps`` is the network's configured frame rate (keep the loader's
        ``target_fps`` aligned to it). ``input_shape`` is required: the tokenizer's 189-frame
        720x1280 defaults would silently mislabel any other geometry.
        """
        is_negative = key == "neg_condition"
        num_frames, height, width = self.pixel_shape(input_shape)
        return {
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "fps": self.fps,
            # One frame is a text2image request in the reference: image resolution template, no duration.
            "is_image": num_frames == 1,
            "is_negative": is_negative,
            "use_system_prompt": getattr(self, "use_system_prompt", False),
            "negative_metadata_mode": getattr(self, "negative_metadata_mode", "same"),
        }

    def build_conditioning(
        self, data: dict, input_shape: Sequence[int], precision: Optional[torch.dtype] = None
    ) -> Dict[str, Any]:
        """Resolve explicit raw/token branches and optionally build I2V context.

        Raw prompts are read only from ``condition_raw`` / ``neg_condition_raw``; a
        non-``None`` plain branch is an already encoded token mapping and is reused as is.
        Prompt templates use the network's configured frame rate. The I2V anchor latent is
        returned in the state ``precision``.
        """
        condition_keys = (("condition", "condition_raw"), ("neg_condition", "neg_condition_raw"))

        def resolve(key, raw_key):
            if data.get(key) is not None:
                return data[key]
            if data.get(raw_key) is None:
                return None
            tokenizer = self.get_preprocessor("text_tokenizer", "tokenize raw prompts")
            return tokenizer.encode(
                data[raw_key],
                **self.text_encode_kwargs(key, input_shape),
            )

        conditions = {}
        for key, raw_key in condition_keys:
            value = resolve(key, raw_key)
            if value is not None:
                conditions[key] = value
        validate_conditioning(
            conditions,
            required_fields=("condition",),
            require_mapping=True,
            absent_mapping_keys=("condition_mask", "conditioning_latents"),
        )

        if not self.is_i2v:
            return conditions

        # I2V: anchor frame 0 with a binary clean-conditioning-frame mask (mirrors cosmos_predict2 V2W).
        # Only single-frame anchoring is supported; multi-frame (V2V) conditioning is not implemented.
        if input_shape is None:
            raise ValueError("Cosmos3 I2V requires input_shape to size the condition mask.")
        real_raw = data.get("real_raw")
        if real_raw is None:
            raise ValueError("Cosmos3 I2V requires data['real_raw'] (pixel-space).")
        _, t_latent, _, _ = input_shape
        bsz = real_raw.shape[0]
        condition_mask = torch.zeros(bsz, 1, t_latent, 1, 1, device=real_raw.device, dtype=torch.bool)
        condition_mask[:, :, 0] = True  # anchor frame 0

        # VAE-encode the anchor frame so ``preserve_conditioning`` can inject it into the
        # clean slot during sampling (mirrors cosmos_predict2 V2W); the Wan2.2 VAE maps a
        # single pixel frame to a single latent frame.
        vae = self.get_preprocessor("vae", "encode the I2V anchor frame")
        conditioning_latents = vae.encode(real_raw[:, :, :1], mode="argmax").to(dtype=precision)

        return {
            key: {
                **value,
                "condition_mask": condition_mask,
                "conditioning_latents": conditioning_latents,
            }
            for key, value in conditions.items()
        }

    def mix_condition(
        self, condition: Any, neg_condition: Any, cond_dropout_prob: float
    ) -> tuple[Any, torch.Tensor | None]:
        """Per-sample CFG dropout for SFT: swap the prompt token IDs with the negative
        prompt's for a ``cond_dropout_prob`` fraction of samples.

        Cosmos3 has no null (empty-string) condition — the negative branch used by CFG at
        inference is the explicit negative-prompt encoding (metadata templates included), so
        training must occasionally see exactly that. The token-ID lists are variable-length (one 1-D tensor per
        sample), so whole samples are swapped by index instead of tensor-mixed. Entries
        shared by both dicts (``condition_mask``, ``conditioning_latents``) are identical
        across positive/negative and pass through unchanged.
        Returns the mixed condition and a ``[B]`` boolean mask on the token IDs' device with ``True``
        for dropped samples, or ``None`` when no prompt condition is available.
        """
        ids, _ = self._extract_condition(condition)
        neg_ids, _ = self._extract_condition(neg_condition)
        if ids is None or neg_ids is None:
            return condition, None
        keep = torch.rand(len(ids)) >= cond_dropout_prob
        mixed_ids = [pos if keep[b] else neg for b, (pos, neg) in enumerate(zip(ids, neg_ids))]
        mixed = {**condition, "input_ids": mixed_ids} if isinstance(condition, Mapping) else mixed_ids
        return mixed, (~keep).to(ids[0].device)

    def _conditioning_frames(self, x: torch.Tensor, condition: Any) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """``(mask, clean)`` broadcast to ``x`` for an I2V condition, ``None`` without a condition mask."""
        _, condition_mask = self._extract_condition(condition)
        clean = condition.get("conditioning_latents") if isinstance(condition, Mapping) else None
        if (condition_mask is None) != (clean is None):
            raise ValueError("conditioning_latents and condition_mask must be provided together")
        if condition_mask is None:
            return None
        assert_same(device=True, x=x, conditioning_latents=clean)
        assert_same(dtype=torch.bool, device=x.device, condition_mask=condition_mask)
        # ``_build_sample_inputs`` packs the sequence assuming exactly latent frame 0 is clean.
        assert_tensor(
            condition_mask[:, :, 0].all() & ~condition_mask[:, :, 1:].any(),
            "Cosmos3 conditions on latent frame 0 only (single-frame I2V contract)",
        )
        mask_c = condition_mask.expand(-1, x.shape[1], -1, x.shape[3], x.shape[4])
        clean_full = torch.zeros_like(x)
        clean_full[:, :, : clean.shape[2]] = clean
        return mask_c, clean_full

    def preserve_conditioning(self, x: torch.Tensor, condition: Any) -> torch.Tensor:
        """Replace the conditioning frames of ``x`` (I2V) with the clean latents.

        The reference trains and samples with those frames at sigma 0 (their noise is zeroed), so ``forward``
        applies this to its input before the backbone runs.
        """
        frames = self._conditioning_frames(x, condition)
        if frames is None:
            return x
        mask_c, clean_full = frames
        return torch.where(mask_c, clean_full, x)

    def _condition_output(self, prediction, x_t, t, condition, fwd_pred_type, **context):
        """Keep the reference's noiseless conditioning frames after guidance.

        Convert the clean target against the caller's raw state/time so every
        prediction domain reconstructs the same x0. At sigma zero a non-x0
        prediction cannot carry it, which raises.
        """
        frames = self._conditioning_frames(x_t, condition)
        if frames is None:
            return prediction
        mask_c, clean_full = frames
        if fwd_pred_type != "x0":
            assert_tensor(
                (self.noise_scheduler.sigma(t) > 0).all(),
                "A non-x0 prediction cannot carry the output constraint at sigma = 0; request fwd_pred_type='x0' "
                "or apply_output_conditioning=False there",
                error_type=ValueError,
            )
            clean_full = self.noise_scheduler.convert_model_output(
                x_t,
                clean_full,
                t,
                src_pred_type="x0",
                target_pred_type=fwd_pred_type,
                output_dtype=prediction.dtype,
            )
        return torch.where(mask_c, clean_full.to(dtype=prediction.dtype), prediction)

    # ----- sampling -----

    @staticmethod
    def default_guidance_cfg() -> GuidanceConfig:
        g = GuidanceConfig()
        g.cfg.scale = 6.0  # Cosmos3 T2V default
        return g

    # Native Cosmos3 sampling (cosmos_framework ``FlowUniPCMultistepScheduler``): UniPC order 2 / BH2
    # predicting x0 from the flow prediction on a shifted-linear grid with final sigma 0. The released
    # scheduler_config.json instead selects an EDM-style Karras grid (``use_karras_sigmas`` with
    # sigma_min=0.147, sigma_max=200) whose diffusers branch ignores ``flow_shift``; loading it sampled
    # the 995..128 grid instead of the native 999..227 one and made the shift a no-op.

    @classmethod
    def native_flow_sigmas(cls, sample_steps: int) -> np.ndarray:
        """The native unshifted grid ``linspace(sigma_max, 0, steps + 1)[:-1]``.

        ``sigma_max`` is ``1 - 1/num_train_timesteps`` rounded to float32, exactly as the native scheduler
        derives it from its float32 training sigmas; the scheduler applies the shift and appends sigma 0.
        """
        sigma_max = float(np.float32(1.0 - 1.0 / cls.NATIVE_NUM_TRAIN_TIMESTEPS))
        return np.linspace(sigma_max, 0.0, sample_steps + 1)[:-1]

    def sample_scheduler_and_grid(
        self,
        sample_steps: int,
        height: int,
        width: int,
        flow_shift: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[UniPCMultistepScheduler, torch.Tensor]:
        """Return the cached UniPC, configured like the native Cosmos3 flow scheduler and set to its native grid.

        Timesteps and sigmas match the native scheduler bit for bit. ``height`` and ``width`` are in pixels; with
        ``flow_shift=None`` the clip's shorter side selects the shift of the nearest native resolution tier.

        Returns:
            The scheduler and the times to query the network at: the FP32 sigmas it integrates, in ``t_precision``,
            the time the reference trains the network on. The reference sampler instead truncates them to integer
            timesteps, which offsets the network's time from the solver's by up to 0.001.
        """
        if flow_shift is None:
            short_side = min(int(height), int(width))
            tier = min(self.RESOLUTION_SHIFT_DEFAULTS, key=lambda resolution: abs(resolution - short_side))
            flow_shift = self.RESOLUTION_SHIFT_DEFAULTS[tier]
        if self.sample_scheduler is None or float(self.sample_scheduler.config.flow_shift) != float(flow_shift):
            self.sample_scheduler = UniPCMultistepScheduler(
                num_train_timesteps=self.NATIVE_NUM_TRAIN_TIMESTEPS,
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                flow_shift=float(flow_shift),
                solver_order=2,
                solver_type="bh2",
                predict_x0=True,
                lower_order_final=True,
                final_sigmas_type="zero",
                thresholding=False,
            )
        self.sample_scheduler.set_timesteps(
            num_inference_steps=sample_steps, device=device, sigmas=self.native_flow_sigmas(sample_steps)
        )
        # High shifts and step counts repeat the leading int64 timestep (999, 999, ...); UniPC would resolve the
        # first step to its second occurrence, skip a sigma, and step past the grid's end.
        self.sample_scheduler.set_begin_index(0)
        t_grid = self.noise_scheduler.safe_clamp(
            self.sample_scheduler.sigmas[:-1].to(device=device, dtype=self.noise_scheduler.t_precision),
            min=self.noise_scheduler.min_t,
            max=self.noise_scheduler.max_t,
            warn=True,
        )
        return self.sample_scheduler, t_grid

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        sample_steps: int = 35,
        sample_type: str = "ode",
        # None: the native per-resolution default (3 / 5 / 10 for the 256 / 480 / 720 tiers); the Cosmos3-Nano
        # model card's flow_shift=10.0 is that 720p value.
        shift: Optional[float] = None,
        fps: Optional[Union[float, torch.Tensor]] = None,
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """UniPC flow multistep sampling on the native Cosmos3 schedule.

        Unlike the reference sampler, which starts from unscaled noise and queries the network at truncated integer
        timesteps, it starts from noise at the first sigma and queries the network at the solver's sigmas.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )
        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()
        _, height, width = self.pixel_shape(noise.shape[1:])
        scheduler, t_grid = self.sample_scheduler_and_grid(
            sample_steps,
            height=height,
            width=width,
            flow_shift=shift,
            device=noise.device,
        )

        # Unlike the reference, which starts from unscaled noise, start at the first query time's noise level.
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])
        # ``forward`` feeds the clean I2V frames to the backbone and guidance pins them in its output, and UniPC's
        # corrector rebuilds its sample from ``last_sample``, so the re-injection below only makes the returned
        # anchor exact (the final sigma-0 step alone reproduces it up to rounding), like the reference's blend.
        # It never changes the generated frames: the backbone sees the clean anchor and emits zeros there.

        for sched_step, t in zip(scheduler.timesteps, t_grid, strict=True):
            t = t.expand(latents.shape[0])
            velocity = predict_with_guidance(
                self,
                latents,
                t,
                guidance_cfg=guidance_cfg,
                condition=condition,
                neg_condition=neg_condition,
                fwd_pred_type="flow",
                fps=fps,
            )
            latents = scheduler.step(velocity, sched_step, latents, return_dict=False)[0]
            latents = self.preserve_conditioning(latents, condition)
        return latents

    # ----- weight loading -----

    def _load_pretrained_transformer(self) -> None:
        """Load the released HF transformer shards into the FastGen backbone."""
        index_path = os.path.join(self.model_id_or_local_path, self.TRANSFORMER_INDEX)

        state_dict = load_checkpoint_state_dict(index_path, device="cpu")
        load_module_state_dict(
            self,
            state_dict,
            strict=False,
            require_any=True,
            description=(
                f"Cosmos3 transformer from {self.model_id_or_local_path}"
                + (f" at revision {self.revision}" if self.revision is not None else "")
            ),
        )

    @staticmethod
    def remap_checkpoint_keys(state_dict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Map released flat Cosmos3 transformer keys onto ``self.transformer.*``.

        Keeps the understanding tower (needed for in-context conditioning), the generation
        tower (``*_moe_gen`` / ``add_*`` / ``norm_added_*``), and the diffusion adapters
        (``proj_in/out``, ``time_embedder``). Drops the unused text head (``lm_head``), the
        audio/action towers, modality embeds, and Transformer-Engine ``_extra_state``.
        """
        drop_prefixes = ("lm_head", "audio_", "action_", "sound_")
        drop_substr = ("_modality_embed", "_extra_state")
        remapped, dropped = {}, 0
        for key, value in state_dict.items():
            k = key
            if k.startswith("transformer."):
                k = k[len("transformer.") :]
            if k.startswith("model.net."):
                k = k[len("model.net.") :]
            elif k.startswith("net."):
                k = k[len("net.") :]
            k = k.replace("._checkpoint_wrapped_module", "").replace("._orig_mod", "")
            if any(k.startswith(p) for p in drop_prefixes) or any(s in k for s in drop_substr):
                dropped += 1
                continue
            remapped["transformer." + k] = value
        if dropped:
            logger.info(f"Cosmos3.remap_checkpoint_keys: dropped {dropped} non-video-generation keys")
        return remapped

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        return super().load_state_dict(self.remap_checkpoint_keys(state_dict), **kwargs)

    # ----- encoders -----

    def init_preprocessors(self):
        # The tokenizer is intentionally not the base builder's ordinary ``text_encoder``:
        # Cosmos3's complete builder needs clip geometry, FPS, and inverse-negative templates.
        if "text_tokenizer" not in self.preprocessors:
            self.init_text_tokenizer()
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_text_tokenizer(self):
        self.preprocessors["text_tokenizer"] = Cosmos3TextTokenizer(self.model_id_or_local_path)

    def init_vae(self):
        # Unlike native Wan's default FP32 path, Cosmos3's native tokenizer explicitly casts VAE
        # weights, inputs, and mean/std statistics to BF16 without autocast. Its sampler remains
        # FP32, so callers cast the BF16 latents to the state precision.
        self.preprocessors["vae"] = WanVideoEncoder(
            model_id_or_local_path=self.model_id_or_local_path,
            fixed_precision_model=torch.bfloat16,
        )

    def _fixed_precision_overrides(self) -> Dict[str, torch.dtype]:
        """Return the transformer's FP32 state overrides (the rotary frequencies and the reference time embedder).

        FastGen selects storage precision before DDP/FSDP wrapping. The reference asserts FP32 rotary
        frequencies in every precision. The explicit precision boundary must retain ``time_embedder``
        before either DDP preserves its mixed parameter dtypes or the custom FSDP child policy selects
        float32 compute.
        """
        prefixes = ("transformer.rotary_emb.",)
        if self.keep_time_embedder_fp32:
            prefixes += ("transformer.time_embedder.",)
        return {name: torch.float32 for name in module_state_names(self) if name.startswith(prefixes)}

    # ----- understanding-tower freeze (DMD2 distillation) -----

    # Generation-tower / diffusion-adapter param markers. A transformer param whose name
    # contains NONE of these belongs to the understanding (text) tower. (``moe_gen`` covers
    # ``mlp_moe_gen`` and the ``*_layernorm_moe_gen`` / ``norm_moe_gen`` norms, so the und
    # variants like ``input_layernorm`` — a prefix of ``input_layernorm_moe_gen`` — are NOT
    # mis-classified.) Splits the 14.53B backbone into ~6.97B gen / ~7.57B und.
    _GEN_PARAM_MARKERS = (
        "moe_gen",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
        "norm_added",
        "proj_in",
        "proj_out",
        "time_embedder",
        "logvar",
    )

    def _is_understanding_param(self, name: str) -> bool:
        return not any(m in name for m in self._GEN_PARAM_MARKERS)

    def _freeze_understanding_tower_(self) -> None:
        """Set ``requires_grad=False`` on every understanding-tower param (train only the
        generation tower). For DMD2 the student/fake_score should keep the teacher's text
        conditioning fixed (cf. Wan DMD2 freezing its text encoder); this also drops the und
        tower's grad + AdamW state since the optimizer is built from ``requires_grad`` params."""
        for n, p in self.transformer.named_parameters():
            if self._is_understanding_param(n):
                p.requires_grad_(False)

    def requires_grad_(self, requires_grad: bool = True):
        """When freezing the understanding tower, intercept the module-level
        ``requires_grad_(True)`` calls (``build_model`` and DMD2's per-iter grad setup) so only
        the generation tower is (re-)enabled; ``requires_grad_(False)`` still freezes everything
        (e.g. the teacher)."""
        super().requires_grad_(requires_grad)
        if requires_grad and self.freeze_understanding_tower:
            self._freeze_understanding_tower_()
        return self

    # ----- FSDP / meta-init -----

    def reset_parameters(self):
        self.transformer.reset_parameters()
        super().reset_parameters()
        logger.info("Reset parameters for Cosmos3 network")

    def fully_shard(self, **kwargs):
        fp32_kwargs = make_fully_shard_kwargs(
            kwargs,
            param_dtype=torch.float32,
            storage_dtype=self.precision_opt,
            enabled=self.keep_time_embedder_fp32,
        )

        # Shard each MoT decoder layer, the optional FP32 timestep embedder,
        # then the whole backbone. Never shard ``self``.
        for layer in self.transformer.layers:
            fully_shard(layer, **kwargs)
        if fp32_kwargs is not None:
            fully_shard(self.transformer.time_embedder, **fp32_kwargs)
            logger.info("Sharded time_embedder with a float32 param_dtype")
        fully_shard(self.transformer, **kwargs)
        # Re-apply the freeze post-shard / pre-optimizer-build so the optimizer (built from
        # requires_grad params at trainer.init_optimizers) excludes the und tower — covers
        # both the student (net) and fake_score, whose grad-state is what frees the memory.
        if self.freeze_understanding_tower:
            self._freeze_understanding_tower_()
