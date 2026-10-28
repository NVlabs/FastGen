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

from typing import Any, Dict, List, Optional, Tuple, Union, Set
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
from transformers import CLIPTextModel, CLIPTokenizer
from fastgen.networks.network import FastGenNetwork, FastGenModule
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.tree_utils import OutputDtype, assert_same
from fastgen.utils.env import hf_load
import fastgen.utils.logging_utils as logger


def classify_forward(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
):
    r"""
    The [`UNet2DConditionModel`] forward method.

    Args:
        sample (`torch.FloatTensor`):
            The noisy input tensor with the following shape `(batch, channel, height, width)`.
        timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
        class_labels (`torch.Tensor`, *optional*, defaults to `None`):
            Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
        timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
            Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
            through the `self.time_embedding` layer to obtain the timestep embeddings.
        attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
            is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
            negative values to the attention scores corresponding to "discard" tokens.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        encoder_attention_mask (`torch.Tensor`):
            A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
            `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
            which adds large negative values to the attention scores corresponding to "discard" tokens.
        feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. Any provided set returns a feature list alongside the output,
                including an empty list for an empty set. None returns only the model output.
        return_features_early: If true, the forward pass returns the features once the set is complete.
            This means the forward pass will not finish completely and no final output is returned.
        return_logvar: If true, the forward pass returns the logvar.

    Returns:
        The sample tensor, or ``[sample, features]`` when ``feature_indices`` is given, or only the features
        with ``return_features_early``; ``return_logvar`` returns ``(output, logvar)``.
    """
    # By default, samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb_timestep = self.time_embedding(t_emb, timestep_cond)

    emb = emb_timestep

    class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None:
        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    aug_emb = self.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )
    if self.config.addition_embed_type == "image_hint":
        aug_emb, hint = aug_emb
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    # 2.5 GLIGEN position net
    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

    # 3. down
    down_block_res_samples = (sample,)
    idx, features = 0, []
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        if feature_indices is not None and idx in feature_indices:
            features.append(sample)
        idx += 1
        if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
            return features

        down_block_res_samples += res_samples

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = self.mid_block(sample, emb)

    if feature_indices is not None and idx in feature_indices:
        features.append(sample)

    # If we have all the features, we can exit early
    if return_features_early:
        assert feature_indices is not None
        assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
        return features

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
            )

    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    out = [sample, features] if feature_indices is not None else sample

    if return_logvar:
        logvar = self.logvar_linear(emb_timestep)
        return out, logvar

    return out


class StableDiffusionTextEncoder(FastGenModule):
    def __init__(
        self,
        model_id_or_local_path: str,
    ):
        super().__init__()
        self.text_encoder = hf_load(
            CLIPTextModel.from_pretrained,
            model_id_or_local_path,
            subfolder="text_encoder",
            # Reference is fp16 (StableDiffusionPipeline); `dtype="auto"` would give fp32
            # (fp32 checkpoint). Keeping fp32 for parity with prior FastGen versions.
            dtype=torch.float32,
        )
        self.tokenizer: CLIPTokenizer = hf_load(
            CLIPTokenizer.from_pretrained,
            model_id_or_local_path,
            subfolder="tokenizer",
        )

    def encode(self, conditioning: Optional[Any] = None) -> Tuple[Any, Any]:
        tokenized = self.tokenizer(
            conditioning,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self(tokenized.input_ids.to(self.device), tokenized.attention_mask.to(self.device))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[Any, Any]:
        return self.text_encoder(input_ids)[0], attention_mask


class StableDiffusionImageEncoder(FastGenModule):
    """Stable Diffusion VAE with pipeline-owned scalar latent normalization.

    The original CompVis ``AutoencoderKL`` has FP32 parameters and returns unscaled
    latents; ``LatentDiffusion`` owns the ``scale_factor`` transform outside it. That
    transform and image encode are FP32, while the default native pipeline decodes the
    neural VAE under FP16 autocast. FastGen mirrors that ordering for released
    ``force_upcast`` VAEs and returns each result in the dtype the VAE computed it in.
    """

    def __init__(
        self,
        model_id_or_local_path: str,
    ):
        super().__init__()
        self.vae: AutoencoderKL = hf_load(AutoencoderKL.from_pretrained, model_id_or_local_path, subfolder="vae")

    @property
    def precision_model(self) -> Optional[torch.dtype]:
        """Keep the released FP32 VAE fixed while allowing custom replacements to opt out."""
        if getattr(self.vae.config, "force_upcast", False):
            return torch.float32
        return super().precision_model

    @property
    def latent_compression_factors(self) -> tuple[int, int]:
        """Pixel-to-latent ``(H, W)`` downsampling of the VAE; every down block but the last halves resolution."""
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
            if getattr(self.vae.config, "force_upcast", False):
                with torch.autocast(device_type=latents.device.type, dtype=torch.float16):
                    images = self.vae.decode(latents, return_dict=False)[0]
            else:
                images = self.vae.decode(latents, return_dict=False)[0]
            return images.clip_(-1.0, 1.0)
        raise ValueError(f"Unsupported preprocessor operation: {operation!r}")


class StableDiffusion15(FastGenNetwork):
    """A StableDiffusion teacher model for text-to-image diffusion distillation."""

    MODEL_ID_15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    latent_compression_factors = (8, 8)

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID_15,
        net_pred_type="eps",
        schedule_type="sd",
        load_pretrained: bool = True,
        **schedule_kwargs,
    ):
        """StableDiffusion score model constructor.

        Args:
            model_id_or_local_path (str, optional): The huggingface model ID or local path to load.
                Defaults to "stable-diffusion-v1-5/stable-diffusion-v1-5".
            net_pred_type (str, optional): Prediction type. Defaults to "eps".
            schedule_type (str, optional): Schedule type. Defaults to "sd".
            load_pretrained (bool, optional): Whether to load pretrained weights.
                Defaults to True.
        """
        # Initialize FastGenNetwork with SD15-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)

        self.model_id_or_local_path = model_id_or_local_path

        # Initialize the network (handles meta device and pretrained loading)
        self._initialize_network(model_id_or_local_path, load_pretrained)

        # Rewrite the unet forward to get bottleneck feature
        self.unet.forward = types.MethodType(classify_forward, self.unet)
        # Block-level diffusers checkpointing for eager training.
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
            logger.info("Loading SD15 UNet from pretrained")
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
                logger.info("Initializing SD15 UNet on meta device (zero memory, will receive weights via FSDP sync)")
            else:
                logger.info("Initializing SD15 UNet from config (no pretrained weights)")
                logger.warning("SD15 UNet being initialized from config. No weights are loaded!")
            self.unet: UNet2DConditionModel = UNet2DConditionModel.from_config(config)

        # Logvar head on the time embedding (``time_embed_dim`` wide; 1280 for the released UNets).
        self.unet.logvar_linear = torch.nn.Linear(self.unet.time_embedding.linear_2.out_features, 1)

    def fully_shard(self, **kwargs):
        """Shard ``self.unet``; see ``FastGenNetwork.fully_shard``.

        SD15 uses a U-Net architecture with:
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

        # Shard down blocks
        for block in self.unet.down_blocks:
            fully_shard(block, **kwargs)

        # Shard mid block
        if self.unet.mid_block is not None:
            fully_shard(self.unet.mid_block, **kwargs)

        # Shard up blocks
        for block in self.unet.up_blocks:
            fully_shard(block, **kwargs)

        fully_shard(self.unet, **kwargs)

    def init_preprocessors(self):
        """Initialize the text and image encoders for the Stable Diffusion model."""
        if "text_encoder" not in self.preprocessors:
            self.init_text_encoder()
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder for Stable Diffusion model."""
        self.preprocessors["text_encoder"] = StableDiffusionTextEncoder(
            model_id_or_local_path=self.model_id_or_local_path
        )

    def init_vae(self):
        """Initialize the image encoder for Stable Diffusion model."""
        self.preprocessors["vae"] = StableDiffusionImageEncoder(model_id_or_local_path=self.model_id_or_local_path)

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
        prompt_embeddings, prompt_mask = condition
        assert_same(dtype=self.precision_model, device=x_t_model.device, prompt_embeddings=prompt_embeddings)
        assert prompt_mask.dtype in (
            torch.bool,
            torch.int32,
            torch.int64,
        ), f"prompt_mask must have bool, int32, or int64 dtype, got {prompt_mask.dtype}"
        # CompVis and Diffusers both project the raw timestep in FP32, then cast
        # the sinusoidal features to the UNet activation dtype.
        model_outputs = self.unet(
            x_t_model,
            self.noise_scheduler.rescale_t(t),
            encoder_hidden_states=prompt_embeddings,
            encoder_attention_mask=prompt_mask,
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
        """Default guidance recipe: CFG only at scale 7.5."""
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
        set_alpha_to_one: bool = False,
    ) -> torch.Tensor:
        """Generate samples using DDIM-style sampling.

        Uses ``predict_with_guidance`` to fuse the cond +
        cfg-uncond pair into one batched forward.

        Args:
            noise: Initial standard-normal noise tensor [B, C, H, W]; the sampler scales it to the first sigma.
            condition: List of [encoder_hidden_states, encoder_attention_mask].
            neg_condition: List of negative [encoder_hidden_states, encoder_attention_mask] for CFG.
            sample_steps: Number of sampling steps.
            sample_type: Integration policy; only ``"ode"`` is supported.
            guidance_cfg: Composite guidance config; ``None`` →
                ``default_guidance_cfg()`` (CFG at 7.5).
            set_alpha_to_one: The final step's target, named as in diffusers' DDIM/PNDM schedulers.
                ``False`` (default), as the reference pipeline's scheduler config, renoises the last x0
                prediction to the table's first entry, ``alpha(0) x0 + sigma(0) eps``, so the output keeps the
                reference's small residual noise (``sigma(0) = sqrt(beta_0) ≈ 0.03``); ``True`` returns x0.

        Returns:
            Generated samples in latent space.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("sd",),
            supported_sample_types=("ode",),
        )
        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()

        # Get timestep schedule (SD uses t in [0, 1], mapped to [0, 1000])
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

            # DDIM step: x_next = alpha_next * x0 + sigma_next * eps; the grid's end 0 is x0 with
            # ``set_alpha_to_one`` and the table's noisy first entry otherwise.
            x = self.noise_scheduler.step(x, eps_pred, t_batch, t_next, pred_type="eps", clean_end=set_alpha_to_one)

        return x
