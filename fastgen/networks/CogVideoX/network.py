# SPDX-FileCopyrightText: Copyright (c) 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
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
from typing import Any, Optional, Tuple, Union, Dict, Set, List
from einops import rearrange
import math
import attrs
import torch

from diffusers import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from torch.distributed.fsdp import fully_shard
from transformers import T5EncoderModel, T5Tokenizer

from fastgen.networks.network import FastGenNetwork, FastGenModule
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.module import attention_forward_without_cudnn
from fastgen.utils.tree_utils import OutputDtype, assert_same
import fastgen.utils.logging_utils as logger
from fastgen.utils.env import hf_load


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    encoder_hidden_states: torch.Tensor,
    timestep_cond: Optional[torch.Tensor] = None,
    ofs: Optional[Union[int, float, torch.LongTensor]] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, num_frames, channels, height, width = hidden_states.shape

    # 1. Time embedding
    timesteps = timestep
    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb_timestep = self.time_embedding(t_emb, timestep_cond)
    emb = emb_timestep

    if self.ofs_embedding is not None:
        ofs_emb = self.ofs_proj(ofs)
        ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
        ofs_emb = self.ofs_embedding(ofs_emb)
        emb = emb + ofs_emb

    # 2. Patch embedding
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
    hidden_states = self.embedding_dropout(hidden_states)

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = hidden_states[:, :text_seq_length]
    hidden_states = hidden_states[:, text_seq_length:]

    # 3. Transformer blocks
    features = []
    for i, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                block,
                hidden_states,
                encoder_hidden_states,
                emb,
                image_rotary_emb,
                attention_kwargs,
                use_reentrant=False,
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )

        if feature_indices is not None and i in feature_indices:
            features.append(hidden_states)

        # If we have all the features, we can exit early
        if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
            return features

    hidden_states = self.norm_final(hidden_states)

    # 4. Final block
    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    # 5. Unpatchify (spatial patches only; CogVideoX 1.5's temporal patching is rejected at construction)
    p = self.config.patch_size
    output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
    output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    out = [output, features] if feature_indices is not None else output

    if return_logvar:
        logvar = self.logvar_linear(emb_timestep)
        return out, logvar

    return out


class CogVideoXTextEncoder(FastGenModule):
    def __init__(
        self,
        model_id_or_local_path: str,
    ):
        super().__init__()
        self.text_encoder = hf_load(
            T5EncoderModel.from_pretrained,
            model_id_or_local_path,
            subfolder="text_encoder",
            # Load in FP32; preparation casts the frozen encoder to the model precision.
            dtype=torch.float32,
        )
        self.tokenizer: T5Tokenizer = hf_load(
            T5Tokenizer.from_pretrained,
            model_id_or_local_path,
            subfolder="tokenizer",
        )

    def encode(self, conditioning: Union[str, List[str]]) -> torch.Tensor:
        tokenized = self.tokenizer(
            conditioning,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return self(tokenized.input_ids.to(self.device))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids)[0]


class CogVideoXVideoEncoder(FastGenModule):
    """CogVideoX VAE preserving the native SAT latent-scaling boundary.

    The SAT loader casts the assembled model, including this VAE, to its selected global
    dtype (FP16 for the 2B recipe, BF16 for 5B/1.5) rather than keeping the VAE in
    FP32. Its shipped configs disable first-stage encode autocast, and decode has no
    VAE-specific autocast. SAT also owns
    ``scale_factor`` outside the VAE: it scales after encode and unscales before decode.
    FastGen mirrors that boundary in ``precision_model`` and returns results in it,
    after the adjacent scalar transform. Ambient autocast remains caller-controlled.
    """

    def __init__(
        self,
        model_id_or_local_path: str,
    ):
        super().__init__()
        self.vae: AutoencoderKLCogVideoX = hf_load(
            AutoencoderKLCogVideoX.from_pretrained, model_id_or_local_path, subfolder="vae"
        )

    @property
    def latent_compression_factors(self) -> tuple[int, int, int]:
        """Pixel-to-latent downsampling factors of the video VAE: ``(T, H, W)``.

        Spatially, every down block but the last halves the resolution, as in the reference pipeline's
        ``vae_scale_factor_spatial``.
        """
        spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return (int(self.vae.config.temporal_compression_ratio), spatial, spatial)

    def encode(self, real_videos: torch.Tensor) -> torch.Tensor:
        """Encode raw videos to latent space.

        Args:
            real_videos: Input video tensor of shape (B, C, T, H, W).

        Returns:
            Latent tensor of shape (B, C, T, H, W) scaled by VAE config.
        """
        return self(real_videos, operation="encode")

    def decode(self, latent_videos: torch.Tensor) -> torch.Tensor:
        """Decode latents back to video space.

        Args:
            latent_videos: Latent tensor of shape (B, C, T, H, W).

        Returns:
            Decoded video tensor of shape (B, C, T, H, W).
        """
        return self(latent_videos, operation="decode")

    def forward(self, data: torch.Tensor, operation: str) -> torch.Tensor:
        if operation == "encode":
            latent_videos = self.vae.encode(data.to(dtype=self.precision_model), return_dict=False)[0].sample()
            return latent_videos * self.vae.config.scaling_factor
        if operation == "decode":
            latents = 1 / self.vae.config.scaling_factor * data.to(dtype=self.precision_model)
            return self.vae.decode(latents).sample
        raise ValueError(f"Unsupported preprocessor operation: {operation!r}")


class CogVideoX(FastGenNetwork):
    """A CogVideoX teacher model for text-to-video diffusion distillation."""

    MODEL_PATH_5B = "THUDM/CogVideoX-5b"
    MODEL_PATH_2B = "THUDM/CogVideoX-2b"
    # The released scheduler configs' ``snr_shift_scale``, the only field in which they differ.
    SNR_SHIFT_SCALES = {MODEL_PATH_2B: 3.0, MODEL_PATH_5B: 1.0}
    latent_compression_factors = (4, 8, 8)

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_PATH_2B,
        net_pred_type="v",
        schedule_type="cogvideox",
        load_pretrained: bool = True,
        disable_grad_ckpt: bool = False,
        fps: int = 8,
        sdpa_no_cudnn: bool = False,
        **schedule_kwargs,
    ):
        """CogVideoX score model constructor.

        Args:
            model_id_or_local_path (str, optional): The huggingface model ID or local path to load.
                Defaults to "THUDM/CogVideoX-2b".
            net_pred_type (str, optional): Prediction type. Defaults to "v".
            schedule_type (str, optional): Schedule type. Defaults to "cogvideox".
            load_pretrained (bool, optional): Whether to load pretrained weights.
                Defaults to True.
            disable_grad_ckpt (bool, optional): Whether to disable gradient checkpointing.
                Defaults to False (checkpointing enabled).
            fps (int): Frames per second. Defaults to 8.
            sdpa_no_cudnn (bool, optional): Run every block's attention off the cuDNN SDPA backend, on Flash or
                mem-efficient attention. cuDNN's BF16 and FP16 SDPA backward returns non-finite gradients for
                CogVideoX-2B, which the other backends compute finitely; 5B trains finitely on the faster cuDNN.
                Defaults to False.
            **schedule_kwargs: Noise-schedule keyword arguments. ``snr_shift_scale`` defaults to the
                released value for the two hub checkpoints and must be given for any other model.
        """
        if "snr_shift_scale" not in schedule_kwargs:
            if model_id_or_local_path not in self.SNR_SHIFT_SCALES:
                raise ValueError(
                    f"No released snr_shift_scale is known for {model_id_or_local_path!r}; pass snr_shift_scale "
                    f"(released values: {self.SNR_SHIFT_SCALES})."
                )
            schedule_kwargs["snr_shift_scale"] = self.SNR_SHIFT_SCALES[model_id_or_local_path]

        # Initialize FastGenNetwork with CogVideoX-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **schedule_kwargs)

        self.model_id_or_local_path = model_id_or_local_path
        self.fps = fps

        # Initialize the network (handles meta device and pretrained loading)
        self._initialize_network(model_id_or_local_path, load_pretrained)

        # Overwrite the transformer forward to get bottleneck feature
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        if sdpa_no_cudnn:
            for block in self.transformer.transformer_blocks:
                block.attn1.forward = types.MethodType(attention_forward_without_cudnn, block.attn1)

        # Enable/disable gradient checkpointing (controlled at init, like Wan)
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    def _initialize_network(self, model_id_or_local_path: str, load_pretrained: bool) -> None:
        """Initialize the transformer network.

        Args:
            model_id_or_local_path: The HuggingFace model ID or local path.
            load_pretrained: Whether to load pretrained weights.
        """
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading CogVideoX transformer from pretrained")
            self.transformer: CogVideoXTransformer3DModel = hf_load(
                CogVideoXTransformer3DModel.from_pretrained,
                model_id_or_local_path,
                subfolder="transformer",
            )
        else:
            # Load config and create model structure
            # If we're in a meta context, tensors will automatically be on meta device
            config = hf_load(
                CogVideoXTransformer3DModel.load_config,
                model_id_or_local_path,
                subfolder="transformer",
            )
            if in_meta_context:
                logger.info(
                    "Initializing CogVideoX transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing CogVideoX transformer from config (no pretrained weights)")
                logger.warning("CogVideoX transformer being initialized from config. No weights are loaded!")
            self.transformer: CogVideoXTransformer3DModel = CogVideoXTransformer3DModel.from_config(config)
        if self.transformer.config.patch_size_t is not None:
            raise NotImplementedError(
                f"{model_id_or_local_path!r} patches time (patch_size_t, CogVideoX 1.5), which FastGen does not "
                "support; use a CogVideoX 1.0 checkpoint (CogVideoX-2b or CogVideoX-5b)."
            )

        # Add logvar_linear layer
        time_embed_dim = self.transformer.time_embedding.linear_2.out_features
        self.transformer.logvar_linear = torch.nn.Linear(time_embed_dim, 1)

    def fully_shard(self, **kwargs):
        """Shard ``self.transformer``; see ``FastGenNetwork.fully_shard``."""
        # Checkpointing has to happen first, for proper casting during backward pass recomputation.
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(self.transformer, CogVideoXBlock)

        # Apply FSDP sharding to each transformer block
        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

    def init_preprocessors(self):
        """Initialize the T5 text encoder and the video VAE."""
        if "text_encoder" not in self.preprocessors:
            self.init_text_encoder()
        if "vae" not in self.preprocessors:
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder for CogVideoX model."""
        self.preprocessors["text_encoder"] = CogVideoXTextEncoder(model_id_or_local_path=self.model_id_or_local_path)

    def init_vae(self):
        """Video VAE used to encode training data and decode samples."""
        self.preprocessors["vae"] = CogVideoXVideoEncoder(model_id_or_local_path=self.model_id_or_local_path)

    def _prepare_rotary_positional_embeddings(
        self, latent_height: int, latent_width: int, num_frames: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.transformer.config.patch_size
        grid_height = latent_height // p
        grid_width = latent_width // p

        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )

        freqs_cos = freqs_cos.to(device=device, dtype=dtype)
        freqs_sin = freqs_sin.to(device=device, dtype=dtype)
        return freqs_cos, freqs_sin

    def _unpatchify_features(self, x_t: torch.Tensor, features: List[torch.Tensor]) -> List[torch.Tensor]:
        B, T, C, H, W = x_t.shape
        p = self.transformer.config.patch_size
        feats = []
        for feat in features:
            feat = feat.reshape(feat.shape[0], T, H // p, W // p, -1, p, p)
            # (B, D, T, H, W)
            feat = rearrange(feat, "B T Hp Wp D p1 p2 -> B D T (Hp p1) (Wp p2)")
            feats.append(feat)
        return feats

    def _get_variables(self, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back=None):
        lamb = ((alpha_prod_t / (1 - alpha_prod_t)) ** 0.5).log()
        lamb_next = ((alpha_prod_t_prev / (1 - alpha_prod_t_prev)) ** 0.5).log()
        h = lamb_next - lamb

        if alpha_prod_t_back is not None:
            lamb_previous = ((alpha_prod_t_back / (1 - alpha_prod_t_back)) ** 0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def _get_mult(self, h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back):
        mult1 = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5 * (-h).exp()
        mult2 = (-2 * h).expm1() * alpha_prod_t_prev**0.5

        if alpha_prod_t_back is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    @staticmethod
    def _local_guidance_cfg(
        guidance_cfg: GuidanceConfig, step: int, use_dynamic_cfg: bool, sample_steps: int
    ) -> GuidanceConfig:
        """Return a step-local copy of *guidance_cfg* with ``cfg.scale`` modulated.

        With ``use_dynamic_cfg=True``, applies the cosine ramp to
        ``guidance_cfg.cfg.scale`` as a function of the integer loop
        index ``step`` (0..``sample_steps``-1), matching the intended
        CogVideoX cosine annealing schedule (see diffusers #9641).
        Otherwise returns the input unchanged.  Sets ``cfg.scale = None``
        when the resulting per-step scale would be a no-op (``<= 1``),
        to skip the uncond forward in ``predict_with_guidance``.
        """
        scale = guidance_cfg.cfg.scale
        if scale is None:
            return guidance_cfg
        if use_dynamic_cfg:
            progress = max(0.0, min(1.0, (sample_steps - step) / sample_steps))
            scale = 1 + scale * ((1 - math.cos(math.pi * progress**5.0)) / 2)
        step_scale = scale if scale > 1 else None
        return attrs.evolve(guidance_cfg, cfg=attrs.evolve(guidance_cfg.cfg, scale=step_scale))

    @staticmethod
    def default_guidance_cfg() -> GuidanceConfig:
        """Default guidance recipe: CFG only at scale 6.0."""
        g = GuidanceConfig()
        g.cfg.scale = 6.0
        return g

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        use_dynamic_cfg: Optional[bool] = False,
        sample_steps: Optional[int] = 50,
        sample_type: str = "sde",
        timestep_spacing: str = "native",
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Multistep sample using CogVideoX's stochastic DPM-Solver++ method.

        Args:
            noise (torch.Tensor): The noisy latents to start from.
            condition (torch.Tensor, optional): The condition
            neg_condition (torch.Tensor, optional): The negative condition
            use_dynamic_cfg (bool, optional): Whether to modulate the CFG scale per step via a cosine ramp.
            sample_steps (int, optional): The number of steps to sample
            sample_type: Integration policy; only ``"sde"`` is supported because every step injects
                fresh noise (``mult_noise``), as in the reference ``CogVideoXDPMScheduler``.
            timestep_spacing (str, optional): ``"native"`` queries the noise schedule's uniform ``grid``;
                ``"trailing"`` queries ``CogVideoXDPMScheduler``'s trailing spacing before its rounding to integer
                timesteps, which are its exact timesteps when ``sample_steps`` divides 1000.
            guidance_cfg (Optional[GuidanceConfig]): Composite guidance config; ``None`` →
                ``default_guidance_cfg()`` (CFG at 6.0).

        Returns:
            torch.Tensor: The sample output.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("cogvideox",),
            supported_sample_types=("sde",),
        )

        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()

        if timestep_spacing == "native":
            t_grid = self.noise_scheduler.grid(sample_steps, device=noise.device)[:-1]
        elif timestep_spacing == "trailing":
            num_train_timesteps = self.noise_scheduler.num_train_timesteps
            if not 1 <= sample_steps <= num_train_timesteps:
                raise ValueError(
                    f"Trailing spacing needs sample_steps in [1, {num_train_timesteps}], got {sample_steps}."
                )
            # diffusers' CogVideoXDPMScheduler spacing before it rounds to integer timesteps. CPU division is
            # correctly rounded, so integer timesteps land exactly on the schedule's table entries.
            steps = torch.arange(sample_steps, device="cpu", dtype=self.noise_scheduler.t_precision)
            t_grid = (num_train_timesteps - 1 - steps * num_train_timesteps / sample_steps) / num_train_timesteps
            t_grid = self.noise_scheduler.safe_clamp(
                t_grid.to(noise.device), min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t, warn=True
            )
        else:
            raise NotImplementedError(f"Unsupported timestep_spacing {timestep_spacing!r}")

        # alpha^2 at every query time, then 1 at the clean end the last step reaches.
        alpha_prod = torch.cat([self.noise_scheduler.alpha(t_grid) ** 2, t_grid.new_ones(1)])

        # Initialize latents with proper scaling based on the initial timestep
        x_t = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])

        prev_pred_xhat = None
        for i in range(sample_steps):
            # Step to the next grid time; diffusers' trailing scheduler steps to t - 1000 // sample_steps, the same
            # time when sample_steps divides 1000.
            alpha_prod_t, alpha_prod_t_prev = alpha_prod[i], alpha_prod[i + 1]
            alpha_prod_t_back = alpha_prod[i - 1] if i > 0 else None

            local_guidance_cfg = CogVideoX._local_guidance_cfg(guidance_cfg, i, use_dynamic_cfg, sample_steps)
            model_t = t_grid[i].expand(x_t.shape[0])
            pred_xhat = predict_with_guidance(
                self,
                x_t,
                model_t,
                guidance_cfg=local_guidance_cfg,
                condition=condition,
                neg_condition=neg_condition,
                fwd_pred_type="x0",
            )

            h, r, _, _ = self._get_variables(alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
            mult = list(self._get_mult(h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back))
            mult_noise = (1 - alpha_prod_t_prev) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5

            if prev_pred_xhat is None or i == sample_steps - 1:
                # First order on the first step, which has no previous prediction, and on the last, where r = 0 makes
                # the second-order weights infinite.
                denoised_d = pred_xhat
            else:
                denoised_d = mult[2] * pred_xhat - mult[3] * prev_pred_xhat
            x_t = mult[0] * x_t - mult[1] * denoised_d + mult_noise * torch.randn_like(x_t)

            prev_pred_xhat = pred_xhat

        return x_t

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the CogVideoX latent video diffusion model.

        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (torch.Tensor): The condition information.
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
            num_features=len(self.transformer.transformer_blocks),
        )
        if return_features_early and feature_indices is not None and not feature_indices:
            return []

        # rearrange x_t from B, C, T, H, W to B, T, C, H, W since CogVideoX expects time first, then channels
        assert isinstance(condition, torch.Tensor), "condition must be a tensor"
        x_t_model = self.cast_to_model_precision(rearrange(x_t, "b c t h w -> b t c h w"))
        assert_same(dtype=self.precision_model, device=x_t_model.device, condition=condition)
        if self.transformer.config.use_rotary_positional_embeddings:
            latent_height, latent_width = x_t_model.shape[3], x_t_model.shape[4]

            # Native SAT builds the rotary tables in float32, then casts them to the query/key dtype and
            # rotates at that precision. This Diffusers-backed path rotates in float32 before restoring the
            # query/key dtype, so retaining float32 tables avoids an additional quantization step.
            image_rotary_emb = self._prepare_rotary_positional_embeddings(
                latent_height, latent_width, x_t_model.shape[1], device=x_t_model.device, dtype=torch.float32
            )
        else:
            image_rotary_emb = None

        model_outputs = self.transformer(
            x_t_model,
            self.noise_scheduler.rescale_t(t),
            condition,
            image_rotary_emb=image_rotary_emb,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
        )

        if return_features_early:
            assert len(model_outputs) == len(feature_indices)
            return self._unpatchify_features(x_t_model, model_outputs)

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        if feature_indices is None:
            assert isinstance(out, torch.Tensor)
            out = rearrange(out, "b t c h w -> b c t h w")
            out = self.transform_output(out, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning)
        else:
            assert isinstance(out, list)
            out[0] = rearrange(out[0], "b t c h w -> b c t h w")
            out[0] = self.transform_output(
                out[0], x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
            )
            out[1] = self._unpatchify_features(x_t_model, out[1])  # already in B, C, T, H, W

        if return_logvar:
            return out, logvar.to(dtype=x_t.dtype)
        return out
