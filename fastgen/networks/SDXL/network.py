# SPDX-FileCopyrightText: Copyright (c) 2025 The HuggingFace Team. All rights reserved.
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

from typing import Any, Optional, List, Set
import types
import torch
from torch.distributed.fsdp import fully_shard

from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UpBlock2D,
    UNetMidBlock2DCrossAttn,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from fastgen.networks.network import FastGenNetwork, FastGenModule
from fastgen.networks.SD15.network import classify_forward
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.tree_utils import OutputDtype, assert_same
from fastgen.utils.env import hf_load
import fastgen.utils.logging_utils as logger


class SDXLTextEncoder(FastGenModule):
    def __init__(self, model_id_or_local_path):
        super().__init__()
        self.text_encoder_one = hf_load(
            CLIPTextModel.from_pretrained,
            model_id_or_local_path,
            subfolder="text_encoder",
            # Reference is fp16 (StableDiffusionXLPipeline); `dtype="auto"` would also give
            # fp16 (fp16 checkpoint). Keeping fp32 for parity with prior FastGen versions.
            dtype=torch.float32,
        )

        self.text_encoder_two = hf_load(
            CLIPTextModelWithProjection.from_pretrained,
            model_id_or_local_path,
            subfolder="text_encoder_2",
            # Same as text_encoder above: reference fp16, `dtype="auto"` fp16, kept fp32.
            dtype=torch.float32,
        )

        self.tokenizer_one = hf_load(
            CLIPTokenizer.from_pretrained,
            model_id_or_local_path,
            subfolder="tokenizer",
        )
        self.tokenizer_two = hf_load(
            CLIPTokenizer.from_pretrained,
            model_id_or_local_path,
            subfolder="tokenizer_2",
        )

    def encode(self, conditioning: Optional[Any] = None) -> List[torch.Tensor]:
        text_input_ids_one = self.tokenizer_one(
            conditioning,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_input_ids_two = self.tokenizer_two(
            conditioning,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return self(text_input_ids_one.to(self.device), text_input_ids_two.to(self.device))

    def forward(
        self,
        text_input_ids_one: torch.Tensor,
        text_input_ids_two: torch.Tensor,
    ) -> List[torch.Tensor]:
        prompt_embeds_list = []

        for text_input_ids, text_encoder in zip(
            [text_input_ids_one, text_input_ids_two], [self.text_encoder_one, self.text_encoder_two]
        ):
            prompt_embeds = text_encoder(
                text_input_ids,
                output_hidden_states=True,
            )

            pooled_prompt_embeds = prompt_embeds[0]

            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            len(text_input_ids_one), -1
        )  # use the second text encoder's pooled prompt embeds (overwrite in for loop)

        return [prompt_embeds, pooled_prompt_embeds]


class SDXLImageEncoder(FastGenModule):
    """SDXL VAE with the reference FP32 latent-normalization boundary.

    The original Stability SGM first stage remains FP32 with first-stage autocast
    explicitly disabled. ``DiffusionEngine`` owns the scale outside ``AutoencoderKL``
    and applies it in FP32. FastGen keeps a released ``force_upcast`` VAE at that
    boundary, scales in the VAE dtype and returns the result in it, and enters the VAE
    dtype before reversing the scale.
    """

    def __init__(
        self,
        model_id_or_local_path: str,
    ):
        super().__init__()
        self.vae: AutoencoderKL = hf_load(AutoencoderKL.from_pretrained, model_id_or_local_path, subfolder="vae")

    @property
    def precision_model(self) -> Optional[torch.dtype]:
        """Keep VAEs that require upcasting in FP32; otherwise honor the requested precision."""
        if getattr(self.vae.config, "force_upcast", False):
            return torch.float32
        return super().precision_model

    @property
    def latent_compression_factors(self) -> tuple[int, int]:
        """Pixel-to-latent ``(H, W)`` downsampling of the VAE: 8x for SDXL (a 1024x1024 image -> 128x128 latent).

        Every down block but the last halves the resolution, as in the reference pipeline's ``vae_scale_factor``.
        """
        factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return (factor, factor)

    def encode(self, real_images: torch.Tensor) -> torch.Tensor:
        return self(real_images, operation="encode")

    def decode(self, latent_images: torch.Tensor) -> torch.Tensor:
        return self(latent_images, operation="decode")

    def forward(self, data: torch.Tensor, operation: str) -> torch.Tensor:
        if operation == "encode":
            latent_images = self.vae.encode(data.to(dtype=self.precision_model), return_dict=False)[0].sample()
            return latent_images * self.vae.config.scaling_factor
        if operation == "decode":
            latents = 1 / self.vae.config.scaling_factor * data.to(dtype=self.precision_model)
            images = self.vae.decode(latents, return_dict=False)[0]
            return images.clip_(-1.0, 1.0)
        raise ValueError(f"Unsupported preprocessor operation: {operation!r}")


class StableDiffusionXL(FastGenNetwork):
    """A StableDiffusion XL teacher model for text-to-image diffusion distillation."""

    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    latent_compression_factors = (8, 8)

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID,
        net_pred_type="eps",
        schedule_type="sdxl",
        load_pretrained: bool = True,
        **schedule_kwargs,
    ):
        """StableDiffusion score model constructor.

        Args:
            model_id_or_local_path (str, optional): The huggingface model ID or local path to load.
                Defaults to "stabilityai/stable-diffusion-xl-base-1.0".
            net_pred_type (str, optional): Prediction type. Defaults to "eps".
            schedule_type (str, optional): Schedule type. Defaults to "sdxl".
            load_pretrained (bool, optional): Whether to load pretrained weights.
                Defaults to True.
        """
        # Initialize FastGenNetwork with SDXL-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)

        self.model_id_or_local_path = model_id_or_local_path

        # Initialize the network (handles meta device and pretrained loading)
        self._initialize_network(model_id_or_local_path, load_pretrained)

        self.unet.forward = types.MethodType(classify_forward, self.unet)
        self.unet.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    def _initialize_network(self, model_id_or_local_path: str, load_pretrained: bool) -> None:
        """Initialize the UNet network.

        Args:
            model_id_or_local_path: The HuggingFace model ID or local path.
            load_pretrained: Whether to load pretrained weights.
        """
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading SDXL UNet from pretrained")
            self.unet: UNet2DConditionModel = hf_load(
                UNet2DConditionModel.from_pretrained,
                model_id_or_local_path,
                subfolder="unet",
            )
        else:
            # Load config and create model structure
            # If we're in a meta context, tensors will automatically be on meta device
            config = hf_load(
                UNet2DConditionModel.load_config,
                model_id_or_local_path,
                subfolder="unet",
            )
            if in_meta_context:
                logger.info("Initializing SDXL UNet on meta device (zero memory, will receive weights via FSDP sync)")
            else:
                logger.info("Initializing SDXL UNet from config (no pretrained weights)")
                logger.warning("SDXL UNet being initialized from config. No weights are loaded!")
            self.unet: UNet2DConditionModel = UNet2DConditionModel.from_config(config)

        # Logvar head on the time embedding (``time_embed_dim`` wide; 1280 for the released UNets).
        self.unet.logvar_linear = torch.nn.Linear(self.unet.time_embedding.linear_2.out_features, 1)

    def fully_shard(self, **kwargs):
        """Shard ``self.unet``; see ``FastGenNetwork.fully_shard``.

        SDXL uses a U-Net architecture with:
        - down_blocks: Encoder blocks (CrossAttnDownBlock2D, DownBlock2D)
        - mid_block: Middle block (UNetMidBlock2DCrossAttn)
        - up_blocks: Decoder blocks (CrossAttnUpBlock2D, UpBlock2D)
        """
        unet_block_types = (
            CrossAttnDownBlock2D,
            CrossAttnUpBlock2D,
            DownBlock2D,
            UpBlock2D,
            UNetMidBlock2DCrossAttn,
        )

        # Checkpointing has to happen first, for proper casting during backward pass recomputation.
        # Diffusers keeps the checkpointing flag on the blocks, so query the U-Net root's property.
        if self.unet.is_gradient_checkpointing:
            self.unet.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(self.unet, unet_block_types)

        for block in self.unet.down_blocks:
            fully_shard(block, **kwargs)
        if self.unet.mid_block is not None:
            fully_shard(self.unet.mid_block, **kwargs)
        for block in self.unet.up_blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.unet, **kwargs)

    def init_preprocessors(self):
        """Initialize the text and image encoders for the SDXL model."""
        if "text_encoder" not in self.preprocessors:
            self.init_text_encoder()
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder for SDXL model."""
        self.preprocessors["text_encoder"] = SDXLTextEncoder(model_id_or_local_path=self.model_id_or_local_path)

    def init_vae(self):
        """Initialize the image encoder for SDXL model."""
        self.preprocessors["vae"] = SDXLImageEncoder(model_id_or_local_path=self.model_id_or_local_path)

    def build_condition_input(self, height: int, width: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return SDXL's micro-conditioning ``time_ids`` for ``height x width`` pixel images, ``[1, 6]``.

        Original and target size are ``(height, width)`` and the crop is ``(0, 0)``, the reference
        pipeline's defaults.
        """
        original_size = (height, width)
        target_size = (height, width)
        crop_top_left = (0, 0)
        return torch.tensor([original_size + crop_top_left + target_size], dtype=torch.float32, device=device)

    def forward(
        self,
        x_t: torch.FloatTensor,
        t: torch.Tensor,
        condition: Optional[List[torch.Tensor]] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ):
        """Forward pass of the StableDiffusion latent diffusion score model.

        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (list[torch.Tensor]): Optional conditioning information.  Defaults to None.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. Any provided set returns a feature list alongside the output,
                including an empty list for an empty set. None returns only the model output.
            return_logvar: If true, the forward pass returns the logvar.
            fwd_pred_type: Update the network prediction type, must be in ['x0', 'eps', 'v', 'flow'].
                None means using the original net_pred_type.
            apply_output_conditioning: Apply the output constraint after the conversion.
            output_dtype: ``"same"`` rounds the prediction to the state dtype; ``None`` keeps it in high precision.
        Returns:
            torch.Tensor: The score model output.
        """
        self._validate_forward_kwargs(
            fwd_kwargs,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            num_features=len(self.unet.down_blocks) + 1,
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        x_t_model = self.cast_to_model_precision(x_t)
        # Get prompt embeddings
        text_embedding, pooled_text_embedding = condition
        assert_same(
            dtype=self.precision_model,
            device=x_t_model.device,
            text_embedding=text_embedding,
            pooled_text_embedding=pooled_text_embedding,
        )
        # The micro-conditioning sizes are in pixels.
        height, width = self.pixel_shape(x_t_model.shape[1:])
        add_time_ids = self.build_condition_input(height, width, device=x_t_model.device).repeat(x_t_model.shape[0], 1)
        unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_text_embedding}

        # Stability AI and Diffusers both project the raw timestep in FP32, then
        # cast the sinusoidal features to the UNet activation dtype.
        model_outputs = self.unet(
            x_t_model,
            self.noise_scheduler.rescale_t(t),
            text_embedding,
            added_cond_kwargs=unet_added_conditions,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
        )

        if return_features_early:
            return model_outputs

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

        if return_logvar:
            return out, logvar.to(dtype=x_t.dtype)
        return out

    @staticmethod
    def default_guidance_cfg() -> GuidanceConfig:
        """Default guidance recipe: CFG only at scale 7.5.

        Experiments may override via ``config.model.guidance_cfg``.
        """
        g = GuidanceConfig()
        g.cfg.scale = 7.5
        return g

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[List[torch.Tensor]] = None,
        neg_condition: Optional[List[torch.Tensor]] = None,
        sample_steps: int = 50,
        sample_type: str = "ode",
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Generate samples using DDIM-style sampling.

        Uses ``predict_with_guidance`` to fuse the cond +
        cfg-uncond pair into one batched forward.

        Args:
            noise: Initial standard-normal noise tensor [B, C, H, W]; the sampler scales it to the first sigma.
            condition: List of [text_embedding, pooled_text_embedding].
            neg_condition: List of negative [text_embedding, pooled_text_embedding] for CFG.
            sample_steps: Number of sampling steps.
            sample_type: Integration policy; only ``"ode"`` is supported.
            guidance_cfg: Composite guidance config; ``None`` →
                ``default_guidance_cfg()`` (CFG at 7.5).

        Returns:
            Generated samples in latent space.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("sdxl",),
            supported_sample_types=("ode",),
        )
        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()

        # Get timestep schedule (SDXL uses t in [0, 1], mapped to [0, 1000])
        grid = self.noise_scheduler.grid(sample_steps, device=noise.device)

        x = self.noise_scheduler.latents(noise=noise, t_init=grid[0])

        for t, t_next in zip(grid[:-1], grid[1:]):
            # Expand t for batch
            t_batch = t.expand(x.shape[0])

            eps_pred = predict_with_guidance(
                self,
                x,
                t_batch,
                batched=True,
                guidance_cfg=guidance_cfg,
                condition=condition,
                neg_condition=neg_condition,
                output_dtype=None,
                fwd_pred_type="eps",
            )

            # DDIM step: x_next = alpha_next * x0 + sigma_next * eps, and x0 at the grid's clean end.
            x = self.noise_scheduler.step(x, eps_pred, t_batch, t_next, pred_type="eps")

        return x
