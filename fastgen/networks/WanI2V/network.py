# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
#
# See licenses/diffusers/LICENSE for more details.
# -----------------------------------------------------------------------------

from typing import Optional, List, Set, Union, Tuple, Dict
import os
import torch
from tqdm.auto import tqdm

from diffusers.models import WanTransformer3DModel
from diffusers.image_processor import PipelineImageInput
from transformers import CLIPImageProcessor, CLIPVisionModel

from fastgen.networks.Wan.network import (
    Wan,
    WanTextEncoder,
    WanVideoEncoder,
)
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils.basic_utils import str2bool
import fastgen.utils.logging_utils as logger


class WanImageEncoder:
    def __init__(
        self,
        model_id_or_local_path: str,
    ):
        # image encoder
        logger.info("Loading CLIP image encoder")
        self.image_encoder = CLIPVisionModel.from_pretrained(
            model_id_or_local_path,
            cache_dir=os.environ["HF_HOME"],
            subfolder="image_encoder",
            torch_dtype=torch.float32,
        )
        logger.info("Loading CLIP image processor")
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_id_or_local_path,
            cache_dir=os.environ["HF_HOME"],
            subfolder="image_processor",
        )
        # We never update the encoder, so freeze it
        self.image_encoder.requires_grad_(False)

    def encode(self, images: PipelineImageInput) -> torch.Tensor:
        device = self.image_encoder.device
        images = (images + 1) / 2.0
        images = self.image_processor(images=images.to(dtype=torch.float32), return_tensors="pt", do_rescale=False).to(
            device, dtype=self.image_encoder.dtype
        )
        image_embeds = self.image_encoder(**images, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        self.image_encoder.to(*args, **kwargs)
        return self


class WanI2V(Wan):
    """A Wan teacher model for image-to-video diffusion distillation."""

    MODEL_ID_VER_2_1_I2V_14B_480P = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    MODEL_ID_VER_2_1_I2V_14B_720P = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    MODEL_ID_VER_2_2_TI2V_5B_720P = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID_VER_2_2_TI2V_5B_720P,
        r_timestep: bool = False,
        disable_efficient_attn: bool = False,
        disable_grad_ckpt: bool = False,
        enable_logvar_linear: bool = True,
        r_embedder_init: str = "pretrained",
        time_cond_type: str = "diff",
        norm_temb: bool = False,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        encoder_depth: int | None = None,
        load_pretrained: bool = True,
        **model_kwargs,
    ):
        """Wan I2V model constructor.

        Args:
            model_id_or_local_path (str, optional): The huggingface model ID or local path to load.
                Defaults to "Wan-AI/Wan2.2-TI2V-5B-Diffusers".
            r_timestep (bool): Whether to support meanflow-like models with r timestep. Defaults to False.
            disable_efficient_attn (bool, optional): Whether to disable efficient attention. Defaults to False.
            disable_grad_ckpt (bool, optional): Whether to disable checkpoints during training. Defaults to False.
            enable_logvar_linear (bool, optional): Whether to enable logvar linear. Defaults to True.
            r_embedder_init (str, optional): Initialization method for the r embedder. Defaults to "pretrained".
            time_cond_type (str, optional): Time condition type for r timestep. Defaults to "diff".
            norm_temb (bool, optional): Whether to normalize the time embeddings. Defaults to False.
            net_pred_type (str, optional): Prediction type. Defaults to "flow".
            schedule_type (str, optional): Schedule type. Defaults to "rf".
            encoder_depth (int, optional): The depth of the encoder (i.e. the number of blocks taking in t embeddings).
                Defaults to None, meaning all blocks take in [t embeddings + r embeddings].
            load_pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
            **model_kwargs: Additional keyword arguments to pass to the FastGenNetwork constructor.
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
            **model_kwargs,
        )

        # Flag to indicate this is an image-to-video model
        self.is_i2v = True

    def _initialize_network(self, model_id_or_local_path: str, load_pretrained: bool) -> Tuple[str, int]:
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading Wan I2V transformer")
            self.transformer: WanTransformer3DModel = WanTransformer3DModel.from_pretrained(
                model_id_or_local_path,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
        else:
            # Load config and create model structure
            # If we're in a meta context, tensors will automatically be on meta device
            config = WanTransformer3DModel.load_config(
                model_id_or_local_path,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
            if in_meta_context:
                logger.info(
                    "Initializing Wan I2V transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing Wan I2V transformer from config (no pretrained weights)")
                logger.warning("Wan I2V Transformer being initializated from config. No weights are loaded!")
            self.transformer: WanTransformer3DModel = WanTransformer3DModel.from_config(config)
        model_id = Wan.get_model_id(model_id_or_local_path)
        if model_id == self.MODEL_ID_VER_2_1_I2V_14B_480P:
            inner_dim = 5120
            self.concat_mask = True
            self.use_image_encoder = True
            self.expand_timesteps = False
        elif model_id == self.MODEL_ID_VER_2_1_I2V_14B_720P:
            inner_dim = 5120
            self.concat_mask = True
            self.use_image_encoder = True
            self.expand_timesteps = False
        elif model_id == self.MODEL_ID_VER_2_2_TI2V_5B_720P:
            inner_dim = 3072
            self.concat_mask = False
            self.use_image_encoder = False
            self.expand_timesteps = True
        else:
            raise NotImplementedError("Model id does not exist.")
        return model_id, inner_dim

    def init_preprocessors(self):
        """Initialize the text and image encoders for the Wan model."""
        if self.use_image_encoder and not hasattr(self, "image_encoder"):
            self.init_image_encoder()
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_image_encoder(self):
        """Initialize the image encoder for WanI2V model."""
        self.image_encoder = WanImageEncoder(model_id_or_local_path=self.model_id)

    def init_text_encoder(self):
        """Initialize the text encoder for WanI2V model."""
        self.text_encoder = WanTextEncoder(model_id_or_local_path=self.model_id)

    def init_vae(self):
        """Initialize the video encoder for WanI2V model."""
        self.vae = WanVideoEncoder(model_id_or_local_path=self.model_id)

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        if hasattr(self, "image_encoder"):
            self.image_encoder.to(*args, **kwargs)

        return self

    def preserve_conditioning(self, x: torch.Tensor, condition: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Preserve conditioning frames for I2V models during student sampling.

        This hook is called by _student_sample_loop to ensure conditioning frames
        are preserved after each denoising step.

        Args:
            x: The tensor to modify (either x0 prediction or noisy latents)
            condition: The condition dict containing first_frame_cond

        Returns:
            The tensor with conditioning frames preserved
        """
        # Only apply for Wan 2.2 style (concat_mask=False)
        # Wan 2.1 (concat_mask=True) handles conditioning through concatenation
        if self.concat_mask:
            return x

        if not isinstance(condition, dict) or "first_frame_cond" not in condition:
            return x

        first_frame_cond = condition["first_frame_cond"]
        x = x.clone()
        x[:, :, 0] = first_frame_cond[:, :, 0]
        return x

    def _replace_first_frame(
        self, first_frame_cond: torch.Tensor, latents: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Replace the first frame of the latents with the first frame condition.

        Args:
            first_frame_cond (torch.Tensor): The clean first frame padded with encoded zero pixels.
            latents (torch.Tensor): The latents to start from.
            return_mask (bool): Whether to return the mask.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]: The latents with the first frame replaced.
                If return_mask is True, return a tuple of the latents and the mask.
        """
        bsz, _, num_latent_frames, latent_height, latent_width = latents.shape
        first_frame_mask = torch.ones(1, 1, num_latent_frames, latent_height, latent_width).to(
            dtype=latents.dtype, device=latents.device
        )

        first_frame_mask[:, :, 0] = 0
        latents = (1 - first_frame_mask) * first_frame_cond + first_frame_mask * latents
        if return_mask:
            return latents, first_frame_mask
        return latents

    def _compute_i2v_inputs(
        self,
        x_t: torch.Tensor,
        first_frame_cond: torch.Tensor,
        timestep: torch.Tensor,
        r_timestep: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute the input tensor for the I2V model. Supports two cases:

            1. Wan 2.1 I2V:
                The input is formed by concatenating three tensors along the channel dimension:
                - x_t: noisy latent video frames
                    shape: [bsz, num_channels, num_latent_frames, H, W]
                - mask_lat_size: binary mask indicating which frames are clean (first) vs noisy (others)
                    shape: [bsz, scale_factor_temporal, num_latent_frames, H, W]
                - first_frame_cond: latent of the clean first frame, with zero-padded frames for the rest
                    shape: [bsz, num_channels, num_latent_frames, H, W]

            2. Wan 2.2 TI2V (5B model):
                The first latent frame of x_t is replaced with the clean first frame.
                In this case, no zero padding is applied.
                - first_frame_cond: clean latent of the first frame only
                    shape: [bsz, num_channels, 1, H, W]
                We also expand timesteps as Wan TI2V uses expanded timesteps.

        Args:
            x_t (torch.Tensor): The noisy latents to start from.
            first_frame_cond (torch.Tensor): The clean first frame padded with encoded zero pixels.
            timestep (torch.Tensor): The timestep.
            r_timestep (torch.Tensor): The optional timestep for meanflow-like models.
        """
        if self.concat_mask:
            bsz, _, num_latent_frames, H, W = x_t.shape
            scale_factor_temporal = 4
            num_frames = (num_latent_frames - 1) * scale_factor_temporal + 1
            # mask tensor to indicate whether the frame is real (clean) frame or noisy one
            mask_lat_size = torch.ones(bsz, 1, num_frames, H, W).to(dtype=x_t.dtype, device=first_frame_cond.device)
            # first frame is real one, so set it as one
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=scale_factor_temporal)
            # mask_lat_size: (bsz, 1, scale_factor_temporal * num_latent_frames, H, W)
            mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
            # mask_lat_size: (bsz, num_latent_frames, scale_factor_temporal, H, W)
            mask_lat_size = mask_lat_size.view(bsz, -1, scale_factor_temporal, H, W)
            # mask_lat_size: (bsz, scale_factor_temporal, num_latent_frames, H, W)
            mask_lat_size = mask_lat_size.transpose(1, 2)

            latent_model_input = torch.concat([x_t, mask_lat_size, first_frame_cond], dim=1)

            timestep = self._compute_timestep_inputs(timestep, mask=None)
            if r_timestep is not None:
                r_timestep = self._compute_timestep_inputs(r_timestep, mask=None)

        else:
            latent_model_input, first_frame_mask = self._replace_first_frame(first_frame_cond, x_t, return_mask=True)
            timestep_mask = first_frame_mask[:, 0]
            timestep = self._compute_timestep_inputs(timestep, timestep_mask)

            if r_timestep is not None:
                timestep_mask = first_frame_mask[:, 0]
                r_timestep = self._compute_timestep_inputs(r_timestep, timestep_mask)

        return latent_model_input, timestep, r_timestep

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 40,
        shift: float = 3.0,
        skip_layers: Optional[List[int]] = None,
        skip_layers_start_percent: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the WanI2V model with proper first-frame conditioning.

        For I2V models, the first latent frame must be preserved as the clean
        conditioning frame after each scheduler step.
        """
        assert self.schedule_type == "rf", f"{self.schedule_type} is not supported"

        # Extract first_frame_cond for replacement after scheduler steps
        first_frame_cond = None
        if isinstance(condition, dict) and "first_frame_cond" in condition:
            first_frame_cond = condition["first_frame_cond"]

        self.unipc_scheduler.config.flow_shift = shift
        self.unipc_scheduler.set_timesteps(num_inference_steps=num_steps, device=noise.device)
        timesteps = self.unipc_scheduler.timesteps

        # Initialize latents with proper scaling based on the initial timestep
        t_init = timesteps[0] / self.unipc_scheduler.config.num_train_timesteps
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        # Main sampling loop
        for idx, timestep in tqdm(enumerate(timesteps), total=num_steps - 1):
            t = (timestep / self.unipc_scheduler.config.num_train_timesteps).expand(latents.shape[0])
            t = self.noise_scheduler.safe_clamp(t, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t).to(
                latents.dtype
            )

            flow_pred = self(
                latents,
                t,
                condition=condition,
                r=None,
                return_features_early=False,
                feature_indices={},
                return_logvar=False,
            )

            if guidance_scale is not None:
                flow_uncond = self(
                    latents,
                    t,
                    condition=neg_condition,
                    r=None,
                    return_features_early=False,
                    feature_indices={},
                    return_logvar=False,
                    skip_layers=skip_layers if idx >= skip_layers_start_percent * num_steps else None,
                )
                flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)

            # For I2V: replace flow prediction for first frame with identity (keeps it clean)
            if first_frame_cond is not None and not self.concat_mask:
                # For Wan 2.2 TI2V: set flow to zero for first frame (no change)
                flow_pred = flow_pred.clone()
                flow_pred[:, :, 0] = 0.0

            latents = self.unipc_scheduler.step(flow_pred, timestep, latents, return_dict=False)[0]

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
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the StableDiffusion latent diffusion score model.

        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (Dict[str, torch.Tensor], optional): Optional conditioning
                information.  Defaults to None.
            r (torch.Tensor): Another timestep mainly used by meanflow.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. If the set is non-empty, then features will be returned.
                By default, feature_indices=None means extract no features.
            return_logvar: If true, the foward pass returns the logvar.
            fwd_pred_type: Update the network prediction type, must be in ['x0', 'eps', 'v', 'flow'].
                None means using the original net_pred_type.
            skip_layers: Apply skip-layer guidance by skipping layers of the unconditional network during forward pass.
            unpatchify_features: If true, the features will be unpatchified and returned in shape of [B, T, H, W, C].

        Returns:
            torch.Tensor: The score model output.
        """
        assert isinstance(condition, dict), "condition must be a dict"
        assert "text_embeds" in condition, "condition must contain 'text_embeds'"
        assert "first_frame_cond" in condition, "condition must contain 'first_frame_cond'"
        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            # Exit immediately if user requested this.
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"

        text_embeds, first_frame_cond = condition["text_embeds"], condition["first_frame_cond"]
        text_embeds = torch.stack(text_embeds, dim=0) if isinstance(text_embeds, list) else text_embeds
        i2v_inputs, timestep, r_timestep = self._compute_i2v_inputs(
            x_t,
            first_frame_cond=first_frame_cond,
            timestep=t,
            r_timestep=r,
        )
        kwargs = dict()
        if "encoder_hidden_states_image" in condition:
            kwargs["encoder_hidden_states_image"] = condition["encoder_hidden_states_image"]
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
            **kwargs,
        )

        if return_features_early:
            assert len(model_outputs) == len(feature_indices)
            return self._unpatchify_features(x_t, model_outputs) if unpatchify_features else model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        if len(feature_indices) == 0:
            assert isinstance(out, torch.Tensor)
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            if not self.concat_mask:
                out = self._replace_first_frame(first_frame_cond, out)

        else:
            assert isinstance(out, list)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            if not self.concat_mask:
                out[0] = self._replace_first_frame(first_frame_cond, out[0])
            out[1] = self._unpatchify_features(x_t, out[1]) if unpatchify_features else out[1]

        if return_logvar:
            return out, logvar
        return out
