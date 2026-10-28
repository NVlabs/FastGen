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

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Optional, List, Set, Union, Tuple, Dict
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffusers.models import WanTransformer3DModel
from transformers import CLIPImageProcessor, CLIPVisionModel

from fastgen.networks.Wan.network import (
    Wan,
    WanTextEncoder,
    WanVideoEncoder,
)
from fastgen.networks.network import FastGenModule, validate_conditioning
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.tensor import assert_tensor
from fastgen.utils.tree_utils import OutputDtype, assert_same
from fastgen.utils.env import hf_load
import fastgen.utils.logging_utils as logger


class WanImageEncoder(FastGenModule):
    """CLIP image encoder of the Wan I2V models.

    Original Wan stores and computes CLIP in FP16. A ``fixed_precision_model`` stores and computes CLIP in that
    dtype whatever the enclosing autocast; without one, CLIP follows the requested precision and the caller's
    autocast.
    """

    def __init__(
        self,
        model_id_or_local_path: str,
        fixed_precision_model: Optional[torch.dtype] = None,
    ):
        super().__init__()
        logger.info("Loading CLIP image encoder")
        self.image_encoder = hf_load(
            CLIPVisionModel.from_pretrained,
            model_id_or_local_path,
            subfolder="image_encoder",
            # Load without an early precision loss. Preparation applies final precision.
            dtype=torch.float32,
        )
        logger.info("Loading CLIP image processor")
        self.image_processor = hf_load(
            CLIPImageProcessor.from_pretrained,
            model_id_or_local_path,
            subfolder="image_processor",
        )
        self.fixed_precision_model = fixed_precision_model

    @property
    def precision_model(self) -> Optional[torch.dtype]:
        """Apply a consumer-fixed CLIP dtype, or otherwise inherit the requested precision."""
        fixed_precision = self.fixed_precision_model
        return super().precision_model if fixed_precision is None else fixed_precision

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # As original Wan's CLIPModel.visual: a bicubic resize without the image processor's antialiasing.
        size = (self.image_encoder.config.image_size,) * 2
        images = F.interpolate(images.to(dtype=torch.float32), size=size, mode="bicubic", align_corners=False)
        mean = images.new_tensor(self.image_processor.image_mean).view(-1, 1, 1)
        std = images.new_tensor(self.image_processor.image_std).view(-1, 1, 1)
        pixel_values = images.mul(0.5).add(0.5).sub(mean).div(std)
        return self(pixel_values=pixel_values.to(self.device, dtype=self.precision_model))

    def forward(self, **images: torch.Tensor) -> torch.Tensor:
        # Original Wan's FP16 autocast keeps its CLIP in FP16 only because its LayerNorms cast back to the input
        # dtype. These LayerNorms would return FP32 under autocast, so a fixed precision disables it instead.
        device_type = images["pixel_values"].device.type
        with nullcontext() if self.fixed_precision_model is None else torch.autocast(device_type, enabled=False):
            image_embeds = self.image_encoder(**images, output_hidden_states=True)
        return image_embeds.hidden_states[-2]


class WanI2V(Wan):
    """A Wan teacher model for image-to-video diffusion distillation."""

    MODEL_ID_VER_2_1_I2V_14B_480P = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    MODEL_ID_VER_2_1_I2V_14B_720P = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    MODEL_ID_VER_2_2_I2V_A14B = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    MODEL_ID_VER_2_2_TI2V_5B_720P = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID_VER_2_2_TI2V_5B_720P,
        r_timestep: bool = False,
        disable_efficient_attn: bool = False,
        disable_grad_ckpt: bool = False,
        enable_logvar_linear: bool = False,
        r_embedder_init: str = "pretrained",
        time_cond_type: str = "diff",
        norm_temb: bool = False,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        encoder_depth: int | None = None,
        load_pretrained: bool = True,
        **kwargs,
    ):
        """Wan I2V model constructor.

        Args:
            model_id_or_local_path (str, optional): The huggingface model ID or local path to load.
                Defaults to "Wan-AI/Wan2.2-TI2V-5B-Diffusers".
            r_timestep (bool): Whether to support meanflow-like models with r timestep. Defaults to False.
            disable_efficient_attn (bool, optional): Whether to disable efficient attention. Defaults to False.
            disable_grad_ckpt (bool, optional): Whether to disable checkpoints during training. Defaults to False.
            enable_logvar_linear (bool, optional): Whether to enable logvar linear. Defaults to False.
            r_embedder_init (str, optional): Initialization method for the r embedder. Defaults to "pretrained".
            time_cond_type (str, optional): Time condition type for r timestep. Defaults to "diff".
            norm_temb (bool, optional): Whether to normalize the time embeddings. Defaults to False.
            net_pred_type (str, optional): Prediction type. Defaults to "flow".
            schedule_type (str, optional): Schedule type. Defaults to "rf".
            encoder_depth (int, optional): The depth of the encoder (i.e. the number of blocks taking in t embeddings).
                Defaults to None, meaning all blocks take in [t embeddings + r embeddings].
            load_pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
            **kwargs: Additional keyword arguments passed to ``Wan``.
        """
        # Initialize FastGenNetwork with Wan-specific defaults
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
            **kwargs,
        )

    def _initialize_network(self, model_id_or_local_path: str, load_pretrained: bool) -> int:
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info(f"Loading Wan I2V transformer from subfolder '{self.transformer_subfolder}'")
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
                    "Initializing Wan I2V transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing Wan I2V transformer from config (no pretrained weights)")
                logger.warning("Wan I2V Transformer being initializated from config. No weights are loaded!")
            self.transformer: WanTransformer3DModel = WanTransformer3DModel.from_config(config)

        config = self.transformer.config
        # Wan2.1-style I2V transformers take a frame mask and the first-frame latents next to the noisy ones, so their
        # input is wider than their output; only Wan2.1 I2V also attends to CLIP image features.
        self.concat_mask = config.in_channels != config.out_channels
        self.use_image_encoder = config.image_dim is not None
        return config.num_attention_heads * config.attention_head_dim

    def init_preprocessors(self):
        """Initialize the text and image encoders for the Wan model."""
        if self.use_image_encoder and "image_encoder" not in self.preprocessors:
            self.init_image_encoder()
        if "text_encoder" not in self.preprocessors:
            self.init_text_encoder()
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_image_encoder(self):
        """Initialize the image encoder for WanI2V model."""
        self.preprocessors["image_encoder"] = WanImageEncoder(
            model_id_or_local_path=self.model_id_or_local_path,
            fixed_precision_model=torch.float16 if self._keep_enc_original_prec else None,
        )

    def init_text_encoder(self):
        """Initialize the text encoder for WanI2V model."""
        self.preprocessors["text_encoder"] = WanTextEncoder(model_id_or_local_path=self.model_id_or_local_path)

    def init_vae(self):
        """Initialize the video encoder for WanI2V model."""
        self.preprocessors["vae"] = WanVideoEncoder(
            model_id_or_local_path=self.model_id_or_local_path,
            fixed_precision_model=torch.float32 if self._keep_enc_original_prec else None,
        )

    def build_conditioning(self, data, input_shape=None, precision=None):
        """Resolve text and build WanI2V context from a pixel ``real_raw``.

        Slices the first pixel frame of ``data["real_raw"]`` and VAE-encodes it
        in deterministic ``mode="argmax"`` (matches the pretrained Wan-2.1 I2V
        contract) into a ``first_frame_cond`` in the state ``precision``.
        ``concat_mask`` checkpoints (Wan-2.1) additionally pad the
        first frame to the full pixel temporal length with zeros before
        encoding, so the latent padding is "VAE-encoded zeros" rather than
        literal zeros.  When the network owns a CLIP image encoder, the raw
        first frame is also CLIP-embedded into ``encoder_hidden_states_image``,
        which only the backbone reads, in ``precision_model``.

        Requires ``data["real_raw"]`` to be present and pixel-space; raises
        ``ValueError`` otherwise so misconfigured pipelines fail loud rather
        than training on degraded conditioning.

        The inherited builder resolves each raw or pre-encoded text branch.
        Returns a fresh positive structure and, when resolved, a matching
        negative structure without mutating ``data``.
        """
        out = super().build_conditioning(data, input_shape=input_shape, precision=precision)
        validate_conditioning(out, required_fields=("condition",))
        text_embeds = out.get("condition")

        real_raw = data.get("real_raw")
        if real_raw is None:
            raise ValueError(
                f"{type(self).__name__} requires data['real_raw'] (pixel-space). "
                "Use a Latent+Raw or pixel dataloader, or supply mp4 in the loader's key_map."
            )

        vae = self.get_preprocessor("vae", "encode the first frame")

        pixel_first = real_raw[:, :, :1]
        encode_input = pixel_first
        if self.concat_mask:
            if input_shape is None or isinstance(input_shape, Mapping):
                raise ValueError(
                    "WanI2V with concat_mask requires input_shape as a "
                    "(C, T, H, W) sequence to derive the pixel temporal "
                    f"length for zero-padding; got {type(input_shape).__name__}."
                )
            num_pixel_frames = self.pixel_shape(input_shape)[0]
            bsz, channels, _, height, width = pixel_first.shape
            padding = pixel_first.new_zeros(bsz, channels, num_pixel_frames - 1, height, width)
            encode_input = torch.cat([pixel_first, padding], dim=2)

        extras = {"first_frame_cond": vae.encode(encode_input, mode="argmax").to(dtype=precision)}
        if "image_encoder" in self.preprocessors:
            image_encoder = self.get_preprocessor("image_encoder", "embed the first frame")
            extras["encoder_hidden_states_image"] = image_encoder.encode(pixel_first[:, :, 0]).to(
                dtype=self.precision_model
            )

        out["condition"] = {"text_embeds": text_embeds, **extras}
        neg_text_embeds = out.get("neg_condition")
        if neg_text_embeds is not None:
            out["neg_condition"] = {"text_embeds": neg_text_embeds, **extras}
        return out

    def _replace_first_frame(
        self, first_frame_cond: torch.Tensor, latents: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Replace the first frame of the latents with the first frame condition.

        Args:
            first_frame_cond (torch.Tensor): The clean first frame padded with encoded zero pixels, on
                ``latents``' device; it is cast to their dtype here.
            latents (torch.Tensor): The latents to start from.
            return_mask (bool): Whether to return the mask.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]: The latents with the first frame replaced.
                If return_mask is True, return a tuple of the latents and the mask.
        """
        bsz, _, num_latent_frames, latent_height, latent_width = latents.shape
        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width, dtype=torch.bool, device=latents.device
        )

        first_frame_mask[:, :, 0] = False
        latents = torch.where(first_frame_mask, latents, first_frame_cond.to(dtype=latents.dtype))
        if return_mask:
            return latents, first_frame_mask
        return latents

    def _condition_output(self, prediction, x_t, t, condition, fwd_pred_type, causal_ctx=None, **context):
        """Pin frame zero in the requested prediction space, only in the first causal chunk.

        Wan2.2 TI2V keeps the clean first frame outside its denoising trajectory.
        Concatenation-based Wan I2V models impose no output constraint.
        """
        if self.concat_mask or condition is None or (causal_ctx is not None and causal_ctx.cur_start_frame != 0):
            return prediction
        first_frame_cond = condition["first_frame_cond"]
        assert_same(device=True, x_t=x_t, first_frame_cond=first_frame_cond)
        t = t[:, None, :, None, None] if t.ndim == 2 else t
        clean = first_frame_cond[:, :, :1]
        if fwd_pred_type != "x0":
            t_first = t[:, :, :1] if t.ndim == 5 else t
            assert_tensor(
                (self.noise_scheduler.sigma(t_first) > 0).all(),
                "A non-x0 prediction cannot carry the output constraint at sigma = 0; request fwd_pred_type='x0' "
                "or apply_output_conditioning=False there",
                error_type=ValueError,
            )
            clean = self.noise_scheduler.convert_model_output(
                x_t[:, :, :1],
                clean,
                t_first,
                src_pred_type="x0",
                target_pred_type=fwd_pred_type,
                output_dtype=prediction.dtype,
            )
        return self._replace_first_frame(clean, prediction)

    def _compute_i2v_inputs(
        self,
        x_t: torch.Tensor,
        first_frame_cond: torch.Tensor,
        timestep: torch.Tensor,
        r_timestep: Optional[torch.Tensor] = None,
        cur_start_frame: int = 0,
        mask_all_frames: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute the input tensor for the I2V model. Matches official Wan-AI/Wan2.1-I2V-14B-480P.

        Supports two cases:

            1. Wan 2.1 I2V (14B, concat_mask=True):
                Input = concat([x_t, mask, first_frame_cond], dim=1). Mask: frame0=1, rest=0,
                reshaped to [bsz, 4, num_latent_frames, H, W] to align with 4x temporal VAE.

            2. Wan 2.2 TI2V (5B, concat_mask=False):
                The first latent frame of x_t is replaced with the clean first frame.
                - first_frame_cond: clean latent of the first frame only
                    shape: [bsz, num_channels, 1, H, W]
                We also expand timesteps as Wan TI2V uses expanded timesteps.
                Frame 0 of the output is pinned to the clean frame in the requested prediction domain
                (``_condition_output``).

        Args:
            x_t (torch.Tensor): The noisy latents to start from.
            first_frame_cond (torch.Tensor): The clean first frame padded with encoded zero pixels, on ``x_t``'s
                device; it is cast to ``x_t``'s dtype where it is used.
            timestep (torch.Tensor): The timestep.
            r_timestep (torch.Tensor): The optional timestep for meanflow-like models.
            cur_start_frame (int): The current start frame index. Varies during autoregressive sampling, where only
                the chunk starting at frame 0 holds the clean first frame.
            mask_all_frames (bool): Whether to mask all frames. Used optionally for KV-caching during
                autoregressive sampling.
        """
        bsz, _, num_latent_frames, H, W = x_t.shape
        if self.concat_mask:
            scale_factor_temporal = self.latent_compression_factors[0]

            # with clean images
            if mask_all_frames:
                # Need to mask all frames
                num_frames = num_latent_frames * scale_factor_temporal
                mask_lat_size = torch.ones(bsz, 1, num_frames, H, W).to(dtype=x_t.dtype, device=first_frame_cond.device)
            elif cur_start_frame == 0:
                # Not masking all, but the first frame needs to be masked
                # mask tensor to indicate whether the frame is real (clean) frame or noisy one
                num_frames = self.pixel_shape(x_t.shape[1:])[0]
                mask_lat_size = torch.zeros(bsz, 1, num_frames, H, W).to(
                    dtype=x_t.dtype, device=first_frame_cond.device
                )
                # first frame is real one, so set it as one
                mask_lat_size[:, :, 0] = 1
                first_frame_mask = mask_lat_size[:, :, :1]
                first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=scale_factor_temporal)
                # mask_lat_size: (bsz, 1, scale_factor_temporal * num_latent_frames, H, W)
                mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
            else:
                # Not masking any frames, as we're past first frame too
                num_frames = num_latent_frames * scale_factor_temporal
                mask_lat_size = torch.zeros(bsz, 1, num_frames, H, W).to(
                    dtype=x_t.dtype, device=first_frame_cond.device
                )
            # mask_lat_size: (bsz, num_latent_frames, scale_factor_temporal, H, W)
            mask_lat_size = mask_lat_size.view(bsz, num_latent_frames, scale_factor_temporal, H, W)
            # mask_lat_size: (bsz, scale_factor_temporal, num_latent_frames, H, W)
            mask_lat_size = mask_lat_size.transpose(1, 2)
            latent_model_input = torch.concat([x_t, mask_lat_size, first_frame_cond.to(dtype=x_t.dtype)], dim=1)

            timestep = self._compute_timestep_inputs(timestep, mask=None)
            if r_timestep is not None:
                r_timestep = self._compute_timestep_inputs(r_timestep, mask=None)
        else:
            latent_model_input = x_t
            timestep_mask = None
            if mask_all_frames:
                # Keep every frame's timestep; same ``[1, T, H, W]`` layout as ``first_frame_mask[:, 0]`` below.
                timestep_mask = torch.ones(1, num_latent_frames, H, W, dtype=torch.bool, device=x_t.device)
            if cur_start_frame == 0:
                # At the start, so force the first frame replacement
                latent_model_input, first_frame_mask = self._replace_first_frame(
                    first_frame_cond, x_t, return_mask=True
                )
                # If we're not masking all frames, we need to mask just the first frame
                if not mask_all_frames:
                    timestep_mask = first_frame_mask[:, 0]
            timestep = self._compute_timestep_inputs(timestep, timestep_mask)

            if r_timestep is not None:
                r_timestep = self._compute_timestep_inputs(r_timestep, timestep_mask)
        return latent_model_input, timestep, r_timestep

    @staticmethod
    def default_guidance_cfg() -> GuidanceConfig:
        """Default guidance recipe: CFG only at scale 5.0.

        Experiments may override via ``config.model.guidance_cfg``.
        """
        g = GuidanceConfig()
        g.cfg.scale = 5.0
        return g

    def resolve_sample_settings(
        self, input_shape: Sequence[int], sample_steps: Optional[int] = None, shift: Optional[float] = None
    ) -> Tuple[int, float]:
        """Fill ``None`` sampler arguments with the official defaults for latents of ``input_shape`` ``[C, T, H, W]``.

        Wan2.1 ``generate.py`` samples I2V for 40 steps, at shift 3.0 within the 480P pixel budget (832x480) and 5.0
        above it, whichever checkpoint is loaded. Its sizes are area budgets, since I2V fits the output to the input
        image's aspect ratio. The Wan2.2 task configs sample I2V-A14B for 40 steps and TI2V-5B for 50, both at shift
        5.0 at every resolution. Wan2.1 I2V is the one I2V checkpoint conditioned on CLIP image features, and TI2V-5B
        the one with expanded timesteps.
        """
        if sample_steps is None:
            sample_steps = 50 if self.expand_timesteps else 40
        if shift is None:
            _, height, width = self.pixel_shape(input_shape)
            shift = 3.0 if self.use_image_encoder and height * width <= 832 * 480 else 5.0
        return sample_steps, shift

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        sample_steps: Optional[int] = None,
        sample_type: str = "ode",
        shift: Optional[float] = None,
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Sample from the WanI2V model with proper first-frame conditioning.

        For I2V models, the first latent frame must be preserved as the clean
        conditioning frame after each scheduler step. ``sample_steps`` and ``shift``
        default (None) to the official settings from ``resolve_sample_settings``.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )

        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()
        sample_steps, shift = self.resolve_sample_settings(noise.shape[1:], sample_steps, shift)

        scheduler, t_grid = self.sample_scheduler_and_grid(sample_steps, flow_shift=shift, device=noise.device)

        # Initialize latents with proper scaling based on the initial timestep
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])
        first_frame_cond = condition.get("first_frame_cond") if isinstance(condition, dict) else None

        # Main sampling loop
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

            # For I2V: replace flow prediction for first frame with identity (keeps it clean)
            if first_frame_cond is not None and not self.concat_mask:
                # For Wan 2.2 TI2V: set flow to zero for first frame (no change)
                flow_pred = flow_pred.clone()
                flow_pred[:, :, 0] = 0.0

            latents = scheduler.step(flow_pred, sched_step, latents, return_dict=False)[0]

            # For I2V: restore first frame to clean conditioning after scheduler step
            if first_frame_cond is not None and not self.concat_mask:
                latents = latents.clone()
                latents[:, :, 0] = first_frame_cond[:, :, 0]

        return latents

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
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the Wan I2V diffusion score model.

        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (Dict[str, torch.Tensor], optional): Optional conditioning
                information. Defaults to None.
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
            apply_output_conditioning: Pin frame zero after the conversion.
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
        assert "first_frame_cond" in condition, "condition must contain 'first_frame_cond'"

        # In TI2V mode the same state-domain tensor is reused for both the input
        # and output first-frame replacements.
        first_frame_cond = condition["first_frame_cond"]
        assert_same(
            shape=self.concat_mask,
            device=True,
            x_t=x_t,
            first_frame_cond=first_frame_cond,
        )
        assert_same(
            shape=True,
            dtype=self.noise_scheduler.t_precision,
            device=True,
            t=t,
            r=r,
        )

        i2v_inputs, timestep, r_timestep = self._compute_i2v_inputs(
            x_t,
            first_frame_cond=first_frame_cond,
            timestep=t,
            r_timestep=r,
        )
        i2v_inputs = self.cast_to_model_precision(i2v_inputs)
        text_embeds = condition["text_embeds"]
        encoder_hidden_states_image = condition.get("encoder_hidden_states_image")
        assert_same(
            dtype=self.precision_model,
            device=i2v_inputs.device,
            text_embeds=text_embeds,
            encoder_hidden_states_image=encoder_hidden_states_image,
        )

        model_outputs = self.transformer(
            hidden_states=i2v_inputs,
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            r_timestep=r_timestep,
            attention_kwargs=None,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            skip_layers=skip_layers,
            encoder_hidden_states_image=encoder_hidden_states_image,
        )

        if return_features_early:
            assert len(model_outputs) == len(feature_indices)
            return self._unpatchify_features(i2v_inputs, model_outputs) if unpatchify_features else model_outputs

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
            out[1] = self._unpatchify_features(i2v_inputs, out[1]) if unpatchify_features else out[1]

        if return_logvar:
            return out, logvar.to(dtype=x_t.dtype)
        return out
