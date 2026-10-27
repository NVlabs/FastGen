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

import types
from typing import Any, Optional, List, Set, Dict, Union, Tuple
import torch
from einops import rearrange
from diffusers.models import WanVACETransformer3DModel

from fastgen.networks.lora_utils import scaled_lora_layers
from fastgen.networks.network import pixel_shape_from_factors, validate_conditioning
from fastgen.networks.Wan.network import (
    Wan,
    apply_r_embedding,
    block_forward,
    compute_temb_modulation,
    flatten_timestep,
    project_logvar,
    project_output,
    unflatten_timestep_proj,
    wan_block_forward_impl,
)
from fastgen.networks.VaceWan.modules.vace_depth_annotator import (
    DEPTH_ANYTHING_V2_VITL_URL,
    VACEDepthExtractor,
)

import fastgen.utils.logging_utils as logger
from fastgen.utils.tree_utils import OutputDtype, assert_same
from fastgen.utils.distributed.collectives import sync_all, sync_any
from fastgen.utils.env import hf_load


def vace_classify_forward_prepare(
    self: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    control_hidden_states: Optional[torch.Tensor] = None,
    control_hidden_states_scale: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[List[torch.Tensor]],
]:
    """Prepare inputs for VACE transformer forward pass."""

    del attention_kwargs  # unused for base VACE path

    num_frames = hidden_states.shape[2]

    if control_hidden_states is not None and control_hidden_states_scale is None:
        control_hidden_states_scale = control_hidden_states.new_ones(len(self.config.vace_layers))

    rotary_emb = self.rope(hidden_states)
    assert_same(
        device=True,
        hidden_states=hidden_states,
        freqs_cos=rotary_emb[0],
        freqs_sin=rotary_emb[1],
    )

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    processed_control_states: Optional[torch.Tensor] = None
    processed_control_scales: Optional[List[torch.Tensor]] = None
    if control_hidden_states is not None:
        if control_hidden_states.shape[2] != num_frames:
            raise ValueError(
                f"VACE control has {control_hidden_states.shape[2]} latent frames, but the video has {num_frames}"
            )
        processed_control_states = self.vace_patch_embedding(control_hidden_states)
        processed_control_states = processed_control_states.flatten(2).transpose(1, 2)
        assert_same(
            shape=True, device=True, hidden_states=hidden_states, processed_control_states=processed_control_states
        )
        processed_control_scales = (
            list(torch.unbind(control_hidden_states_scale)) if control_hidden_states_scale is not None else None
        )

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
        processed_control_states,
        processed_control_scales,
    )


def vace_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    control_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
    norm_temb: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """VACE control block with the original Wan activation boundaries."""
    core_dtype = encoder_hidden_states.dtype
    if self.proj_in is not None:
        control_hidden_states = self.proj_in(control_hidden_states.to(core_dtype))
        control_hidden_states = control_hidden_states + hidden_states.to(core_dtype)

    modulation = compute_temb_modulation(self.scale_shift_table, temb, norm_temb)
    control_hidden_states = wan_block_forward_impl(
        self,
        control_hidden_states,
        encoder_hidden_states,
        rotary_emb,
        *modulation,
    )

    conditioning_states = None
    if self.proj_out is not None:
        conditioning_states = self.proj_out(control_hidden_states.to(core_dtype))
    return conditioning_states, control_hidden_states


def vace_classify_forward_block_forward(
    self: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep_proj: torch.Tensor,
    r_timestep_proj: Optional[torch.Tensor],
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    control_hidden_states: Optional[torch.Tensor] = None,
    control_hidden_states_scale: Optional[List[torch.Tensor]] = None,
    skip_layers: Optional[List[int]] = None,
    feature_indices: Optional[Set[int]] = None,
    return_features_early: bool = False,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Execute transformer blocks with optional VACE control."""

    del attention_kwargs  # unused for base VACE path

    features: List[torch.Tensor] = []

    gradient_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing
    has_control = control_hidden_states is not None and hasattr(self, "vace_blocks")

    def run_block(module: torch.nn.Module, *args: Any) -> Any:
        return self._gradient_checkpointing_func(module, *args) if gradient_ckpt else module(*args)

    control_hidden_states_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
    current_control_states = control_hidden_states
    if has_control:
        assert control_hidden_states_scale is not None
        for layer_idx, block in enumerate(self.vace_blocks):
            conditioning_states, current_control_states = run_block(
                block,
                hidden_states,
                encoder_hidden_states,
                current_control_states,
                timestep_proj,
                rotary_emb,
                self.norm_temb,
            )
            control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[layer_idx]))
        control_hidden_states_list = control_hidden_states_list[::-1]

    for block_idx, block in enumerate(self.blocks):
        if self.encoder_depth is not None and block_idx == self.encoder_depth and r_timestep_proj is not None:
            timestep_proj = r_timestep_proj

        # Skip only the backbone block; still apply its VACE hint and record its feature so both stay aligned.
        if skip_layers is None or block_idx not in skip_layers:
            hidden_states = run_block(
                block, hidden_states, timestep_proj, encoder_hidden_states, rotary_emb, self.norm_temb
            )

        if has_control and block_idx in self.config.vace_layers:
            assert control_hidden_states_list, "Control states depleted before applying to configured layers"
            control_hint, scale = control_hidden_states_list.pop()
            hidden_states = hidden_states + control_hint * scale

        if feature_indices is not None and block_idx in feature_indices:
            features.append(hidden_states)
            if return_features_early and len(features) == len(feature_indices):
                break

    if return_features_early:
        assert feature_indices is not None
        assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"

    return hidden_states, features


def vace_classify_forward(
    self: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    control_hidden_states: Optional[torch.Tensor] = None,
    control_hidden_states_scale: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
    skip_layers: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Forward pass with feature collection and logvar support for VACE model."""
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    with scaled_lora_layers(self, lora_scale):
        (
            hidden_states,
            timestep_proj,
            r_timestep_proj,
            encoder_hidden_states,
            temb,
            rotary_emb,
            processed_control_states,
            processed_control_scales,
        ) = self.classify_forward_prepare(
            hidden_states,
            timestep,
            encoder_hidden_states,
            r_timestep,
            encoder_hidden_states_image,
            control_hidden_states,
            control_hidden_states_scale,
            attention_kwargs,
        )

        hidden_states, features = self.classify_forward_block_forward(
            hidden_states,
            timestep_proj,
            r_timestep_proj,
            encoder_hidden_states,
            rotary_emb,
            processed_control_states,
            processed_control_scales,
            skip_layers,
            feature_indices,
            bool(return_features_early),
            attention_kwargs,
        )

        if return_features_early:
            return features

        if temb.dim() == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)

            if shift.shape[1] == hidden_states.shape[1]:
                hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
            else:
                frame_seqlen = post_patch_height * post_patch_width
                hs_norm_out = self.norm_out(hidden_states.float()).unflatten(1, (post_patch_num_frames, frame_seqlen))
                hidden_states = (
                    (hs_norm_out * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)).flatten(1, 2).type_as(hidden_states)
                )
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)

            hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = project_output(self, hidden_states, encoder_hidden_states.dtype)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        out: Union[torch.Tensor, List[torch.Tensor]] = output if feature_indices is None else [output, features]

        if return_logvar:
            assert hasattr(
                self, "logvar_linear"
            ), "logvar_linear layer is required when return_logvar=True. Set enable_logvar_linear=True in model config."
            # One value per sample: average the time embeddings of the noised tokens.
            logvar = project_logvar(self, temb, timestep, encoder_hidden_states.dtype)
            return out, logvar
        return out


class VACEWan(Wan):
    """VACE-WAN model following the exact VACE pattern."""

    MODEL_ID_1_3B = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    MODEL_ID_14B = "Wan-AI/Wan2.1-VACE-14B-diffusers"

    def _transformer_blocks(self) -> Tuple[torch.nn.Module, ...]:
        return (*super()._transformer_blocks(), *self.transformer.vace_blocks)

    def __init__(
        self,
        depth_model_path: str = DEPTH_ANYTHING_V2_VITL_URL,
        context_scale: float = 1.0,
        model_id_or_local_path: str = MODEL_ID_1_3B,
        r_timestep: bool = False,
        disable_efficient_attn: bool = False,
        disable_grad_ckpt: bool = False,
        enable_logvar_linear: bool = False,
        r_embedder_init: str = "pretrained",
        time_cond_type: str = "abs",
        norm_temb: bool = False,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        encoder_depth: int | None = None,
        load_pretrained: bool = True,
        use_fsdp_checkpoint: bool = True,
        **kwargs,
    ):
        """Initialize VACE-WAN model.

        Args:
            depth_model_path: Local path, S3 URI, or HTTP(S) URL for the depth model.
            context_scale: Scale factor for context influence
            model_id_or_local_path: HuggingFace model ID or local path for base WAN
            r_timestep (bool): Whether to support meanflow-like models with r timestep. Defaults to False.
            disable_efficient_attn (bool, optional): Whether to disable efficient attention. Defaults to False.
            disable_grad_ckpt (bool, optional): Whether to disable checkpoints during training. Defaults to False.
            enable_logvar_linear (bool, optional): Whether to enable logvar linear prediction. Defaults to False.
            r_embedder_init (str, optional): Initialization method for the r embedder. Defaults to "pretrained".
            time_cond_type (str, optional): Time condition type. Defaults to "abs".
            norm_temb (bool, optional): Whether to normalize the time embeddings of the main and control blocks.
                Defaults to False.
            net_pred_type (str, optional): Prediction type. Defaults to "flow".
            schedule_type (str, optional): Schedule type. Defaults to "rf".
            encoder_depth (int, optional): The depth of the encoder (i.e. the number of blocks taking in t embeddings).
                Defaults to None, meaning all blocks take in [t embeddings + r embeddings].
            load_pretrained (bool, optional): Whether to load the model from pretrained.
                If False, the model will be initialized from the config.
                Defaults to True.
            use_fsdp_checkpoint (bool, optional): Whether to use FSDP gradient checkpointing. Defaults to True.
            **kwargs: Additional keyword arguments passed to ``Wan``.
        """
        super().__init__(
            model_id_or_local_path=model_id_or_local_path,
            r_timestep=r_timestep,
            disable_efficient_attn=disable_efficient_attn,
            disable_grad_ckpt=disable_grad_ckpt,
            enable_logvar_linear=enable_logvar_linear,
            r_embedder_init=r_embedder_init,
            time_cond_type=time_cond_type,
            norm_temb=norm_temb,
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            encoder_depth=encoder_depth,
            load_pretrained=load_pretrained,
            use_fsdp_checkpoint=use_fsdp_checkpoint,
            **kwargs,
        )
        self.context_scale = context_scale

        self.depth_model_path = depth_model_path
        self._all_zero_latent = None

        torch.cuda.empty_cache()
        logger.info("Initialized VACE-WAN model")

    def override_transformer_forward(self) -> None:
        """Override the transformer forward method to use VACE-specific methods."""
        # Base Wan patches (no-cuDNN SDPA hooks, official sinusoidal embedding, r-embedding fusion); the
        # block and transformer forwards are rebound to the VACE implementations below.
        super().override_transformer_forward()
        for block in self.transformer.blocks:
            block.forward = types.MethodType(block_forward, block)
        for block in self.transformer.vace_blocks:
            block.forward = types.MethodType(vace_block_forward, block)
        self.transformer.classify_forward_prepare = types.MethodType(vace_classify_forward_prepare, self.transformer)
        self.transformer.classify_forward_block_forward = types.MethodType(
            vace_classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(vace_classify_forward, self.transformer)

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
            logger.info("Loading VaceWan transformer")
            self.transformer: WanVACETransformer3DModel = hf_load(
                WanVACETransformer3DModel.from_pretrained,
                model_id_or_local_path,
                subfolder=self.transformer_subfolder,
            )
        else:
            # Load config and create model structure
            # If we're in a meta context, tensors will automatically be on meta device
            config = hf_load(
                WanVACETransformer3DModel.load_config,
                model_id_or_local_path,
                subfolder=self.transformer_subfolder,
            )
            if in_meta_context:
                logger.info(
                    "Initializing VaceWan transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing VaceWan transformer from config (no pretrained weights)")
                logger.warning("VaceWan Transformer being initializated from config. No weights are loaded!")
            self.transformer: WanVACETransformer3DModel = WanVACETransformer3DModel.from_config(config)
        config = self.transformer.config
        return config.num_attention_heads * config.attention_head_dim

    def init_preprocessors(self) -> None:
        """Initialize Wan's encoders and load the depth annotator before preparation."""
        super().init_preprocessors()
        if "depth_annotator" not in self.preprocessors:
            self.preprocessors["depth_annotator"] = VACEDepthExtractor(model_path=self.depth_model_path)

    def build_conditioning(self, data, input_shape=None, precision=None):
        """Resolve text and build VaceWan context from pixels or a depth latent.

        Delegates the depth-extraction + VAE-encoding work to
        ``_build_vid_context`` and returns the context, which only the backbone
        reads, in ``precision_model``. ``data["depth_latent"]`` (if
        present) is forwarded as a precomputed shortcut for the depth step
        and ``real_raw`` is optional when every distributed rank has that
        shortcut. If any rank lacks it, all ranks run the depth annotator on
        ``real_raw`` (which must therefore be pixel-space and present everywhere).

        ``input_shape`` is unused (the condition shape is fully determined by
        ``real_raw`` / ``depth_latent``); accepted for hook signature
        compatibility.

        The inherited builder resolves each raw or pre-encoded text branch.
        This method enriches those branches without mutating ``data``.
        """
        out = super().build_conditioning(data, input_shape=input_shape, precision=precision)
        validate_conditioning(out, required_fields=("condition",))
        text_embeds = out.get("condition")

        depth_latent = data.get("depth_latent")
        real_raw = data.get("real_raw")
        # _build_vid_context validates that at least one depth source is available.
        vid_context = self._build_vid_context(
            video=real_raw,
            condition_latents=depth_latent,
        )
        vid_context = vid_context.to(dtype=self.precision_model)

        out["condition"] = {"text_embeds": text_embeds, "vid_context": vid_context}
        neg_text_embeds = out.get("neg_condition")
        if neg_text_embeds is not None:
            out["neg_condition"] = {"text_embeds": neg_text_embeds, "vid_context": vid_context}
        return out

    def _build_vid_context(
        self,
        video: Optional[torch.Tensor],
        condition_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prepare VACE conditioning by extracting depth and creating latents.

        Private helper of ``build_conditioning``; not part of the public network interface.

        Args:
            video: [B, C, T, H, W] pixel-space video frames in range [-1, 1],
                used to run the depth annotator + VAE encode. Ignored when every
                rank supplies ``condition_latents``; required on every rank when
                any rank needs depth extraction.
            condition_latents: [B, C_latent, T_latent, H_latent, W_latent]
                precomputed depth latents (VAE-encoded), used as the reactive
                half of the VACE context.

        Returns:
            conditioning: [B, 96, T_latent, H_latent, W_latent]
                (16 inactive + 16 reactive depth latents + 64 mask channels)
        """

        has_precomputed_depth = condition_latents is not None
        sync_input = condition_latents if has_precomputed_depth else video
        if sync_input is None:
            raise ValueError(
                f"{type(self).__name__} requires data['real_raw'] (pixel-space). "
                "Use a pixel dataloader for depth extraction, or supply data['depth_latent'] precomputed."
            )
        sync_device = sync_input.device

        # Both paths can contain collectives (sharded depth extraction, temporary VAE preparation,
        # sharded VAE encodes), so use the shortcut only when every rank can take it.
        precomputed_on_all_ranks = sync_all(has_precomputed_depth, device=sync_device)
        if not precomputed_on_all_ranks and not sync_all(video is not None, device=sync_device):
            raise RuntimeError(
                "At least one VACE rank requires depth extraction because `depth_latent` is absent, but "
                "`real_raw` is not available on every rank. Mixed precomputed/raw batches require pixels "
                "on every rank."
            )

        if not precomputed_on_all_ranks:
            assert video is not None
            if "vae" not in self.preprocessors:
                raise ValueError("VAE must be initialized to prepare video latents")
            depth_annotator = self.preprocessors.get("depth_annotator")
            if depth_annotator is None:
                raise ValueError("Depth annotator must be initialized to extract depth from video")

            # Extract depth maps from video
            with torch.no_grad():  # Ensure no gradients flow through depth extraction
                inp_depth_annotator = rearrange(video, "b c t h w -> (b t) c h w")
                inp_depth_annotator = (inp_depth_annotator + 1) * 0.5  # from [-1, 1] to [0, 1]
                depth_video = depth_annotator(inp_depth_annotator)  # Returns in [0, 1]
                depth_video = rearrange(depth_video, "(b t) c h w -> b c t h w", b=video.shape[0])

            # Convert depth to proper range for VAE encoding [-1, 1]
            depth_video = 2.0 * depth_video - 1.0

            # For VACE, we need to encode depth with a mask to get 32 channels
            # Create an all-ones mask to indicate the entire frame contains depth information
            mask = torch.ones_like(depth_video)

            # Prepare video latents WITH mask to get proper 32-channel format
            video_latents = self._prepare_video_latents(depth_video, mask=mask)  # Shape: [B, 32, T, H, W]
        else:
            assert condition_latents is not None
            # Cache a deterministic batch-one inactive latent. Batch is expanded below;
            # only the non-batch latent geometry determines whether it can be reused.
            local_cache_miss = (
                self._all_zero_latent is None or self._all_zero_latent.shape[1:] != condition_latents.shape[1:]
            )

            # Temporary preparation and a persistent FSDP VAE's encode contain
            # collectives, so the cache state itself must agree before either path runs.
            cache_miss_on_any_rank = sync_any(local_cache_miss, device=condition_latents.device)
            cache_miss_on_all_ranks = sync_all(local_cache_miss, device=condition_latents.device)
            assert (
                cache_miss_on_any_rank == cache_miss_on_all_ranks
            ), "VACE inactive-latent cache misses must be consistent across distributed ranks."
            if cache_miss_on_any_rank:
                with self.vae_context() as vae:
                    assert vae is not None, "VAE must be available to prepare video latents."

                    # The inactive half is the VAE encoding of a black clip. Reconstruct
                    # its pixel geometry from the precomputed latent shape, so this path
                    # needs neither ``real_raw`` nor persistent preprocessors.
                    pixel_shape = pixel_shape_from_factors(condition_latents.shape[1:], vae.latent_compression_factors)
                    zeros_clip = condition_latents.new_zeros((1, 3, *pixel_shape))
                    self._all_zero_latent = vae.encode(zeros_clip, mode="argmax")

            inactive_latents = self._all_zero_latent.to(condition_latents.device, condition_latents.dtype).expand(
                condition_latents.shape[0], -1, -1, -1, -1
            )
            video_latents = torch.cat([inactive_latents, condition_latents], dim=1)

        # Prepare mask for latent space (always all ones for full frame generation)
        b, _, t, h, w = video_latents.shape
        mask_latents = torch.ones(b, 64, t, h, w, device=video_latents.device, dtype=video_latents.dtype)

        # Concatenate video latents and mask
        conditioning = torch.cat([video_latents, mask_latents], dim=1)  # Shape: [B, 96, T, H, W]

        return conditioning

    def _prepare_video_latents(
        self,
        video: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """General method to prepare video latents by encoding with VAE.

        Args:
            video: [B, C, T, H, W] video frames in range [-1, 1]
            mask: [B, C, T, H, W] optional mask tensor
            generator: Random generator for sampling

        Returns:
            latents: [B, C_latent*2, T_latent, H_latent, W_latent] if mask provided,
                        else [B, C_latent, T_latent, H_latent, W_latent]
        """
        if "vae" not in self.preprocessors:
            raise ValueError("VAE must be initialized to prepare video latents")

        if mask is None:
            # Encode video directly with argmax mode for deterministic results
            latents = self.preprocessors["vae"].encode(
                video,
                mode="argmax",
            )
        else:
            # Apply mask to separate inactive/reactive regions
            assert_same(shape=True, device=True, video=video, mask=mask)
            mask = (mask > 0.5).to(dtype=video.dtype)
            inactive = video * (1 - mask)
            reactive = video * mask

            # Encode separately with argmax mode
            inactive_latents = self.preprocessors["vae"].encode(inactive, mode="argmax")
            reactive_latents = self.preprocessors["vae"].encode(reactive, mode="argmax")

            # Concatenate
            latents = torch.cat([inactive_latents, reactive_latents], dim=1)

        return latents

    def sample(
        self,
        noise: torch.Tensor,
        sample_steps: int = 50,
        sample_type: str = "ode",
        shift: float = 3.0,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the model.

        ``shift`` defaults to 3.0, the ``flow_shift`` in the Diffusers VACE checkpoints' scheduler config. The
        official Wan2.1 ``generate.py`` and VACE inference script sample VACE at shift 16.
        """
        return super().sample(noise, sample_steps=sample_steps, sample_type=sample_type, shift=shift, **kwargs)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
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
    ):
        """Forward pass with VACE context support.
        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (dict[str, torch.Tensor]): Prepared ``text_embeds`` and
                ``vid_context``. ``vid_context`` has shape ``[B, 96, T, H, W]``.
            r (torch.Tensor): Another timestep mainly used by meanflow.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. Any provided set returns a feature list alongside the output,
                including an empty list for an empty set. None returns only the model output.
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

        assert isinstance(condition, dict), "condition must be a dict"
        assert "text_embeds" in condition, "condition must contain 'text_embeds'"
        assert "vid_context" in condition, "condition must contain 'vid_context'"

        x_t_model = self.cast_to_model_precision(x_t)
        assert_same(
            shape=True,
            dtype=self.noise_scheduler.t_precision,
            device=True,
            t=t,
            r=r,
        )

        # Handle VACE mode
        encoder_hidden_states = condition["text_embeds"]
        control_hidden_states = condition["vid_context"]
        assert_same(
            dtype=self.precision_model,
            device=x_t_model.device,
            text_embeds=encoder_hidden_states,
            vid_context=control_hidden_states,
        )

        # Prepare control_hidden_states_scale
        control_hidden_states_scale = self.context_scale
        if isinstance(control_hidden_states_scale, (int, float)):
            control_hidden_states_scale = [control_hidden_states_scale] * len(self.transformer.config.vace_layers)
        if isinstance(control_hidden_states_scale, list):
            if len(control_hidden_states_scale) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `control_hidden_states_scale` {len(control_hidden_states_scale)} does not "
                    f"match number of layers {len(self.transformer.config.vace_layers)}."
                )
            control_hidden_states_scale = torch.tensor(control_hidden_states_scale)
        if isinstance(control_hidden_states_scale, torch.Tensor):
            if control_hidden_states_scale.size(0) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `control_hidden_states_scale` {control_hidden_states_scale.size(0)} does not "
                    f"match number of layers {len(self.transformer.config.vace_layers)}."
                )
            control_hidden_states_scale = control_hidden_states_scale.to(device=x_t_model.device, dtype=x_t_model.dtype)

        # Call transformer
        model_outputs = self.transformer(
            hidden_states=x_t_model,
            timestep=self.noise_scheduler.rescale_t(t),
            encoder_hidden_states=encoder_hidden_states,
            r_timestep=None if r is None else self.noise_scheduler.rescale_t(r),
            encoder_hidden_states_image=None,
            control_hidden_states=control_hidden_states,
            control_hidden_states_scale=control_hidden_states_scale,
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
