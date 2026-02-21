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

import types
from typing import Optional, List, Set, Union, Tuple, Dict

from tqdm.auto import tqdm
import torch
from fastgen.networks.network import CausalFastGenNetwork
from fastgen.networks.Wan.network import (
    classify_forward,
)
from fastgen.networks.Wan.network_causal import (
    _rope_forward_with_time_offset,
    _prepare_blockwise_causal_attn_mask,
    _create_external_caches,
    _wan_set_attn_processor,
    _wan_block_forward_inline_cache,
    classify_forward_prepare,
    classify_forward_block_forward,
    CausalWanAttnProcessor,
)
from fastgen.networks.WanI2V.network import WanI2V
from fastgen.networks.noise_schedule import NET_PRED_TYPES


class CausalWanI2V(CausalFastGenNetwork, WanI2V):
    def __init__(
        self,
        model_id_or_local_path: str = WanI2V.MODEL_ID_VER_2_2_TI2V_5B_720P,
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
        use_fsdp_checkpoint: bool = True,
        mask_when_storing_cache: bool = False,
        chunk_size: int = 3,
        total_num_frames: int = 21,
        delete_cache_on_clear: bool = False,
        **model_kwargs,
    ):
        """Causal Wan I2V model constructor.

        Args:
            model_id_or_local_path (str, optional): The huggingface model ID or local path to load.
                Defaults to "Wan-AI/Wan2.2-TI2V-5B-Diffusers".
            r_timestep (bool, optional): Whether to support meanflow-like models with r timestep. Defaults to False.
            disable_efficient_attn (bool, optional): Whether to disable efficient attention. Defaults to False.
            disable_grad_ckpt (bool, optional): Whether to disable checkpoints during training. Defaults to False.
            enable_logvar_linear (bool, optional): Whether to enable logvar linear. Defaults to False.
            r_embedder_init (str, optional): Initialization method for the r embedder. Defaults to "pretrained".
            time_cond_type (str, optional): Time condition type for r timestep. Defaults to "diff".
            norm_temb (bool, optional): Whether to normalize the time embeddings. Defaults to False.
            net_pred_type (str, optional): Prediction type. Defaults to "flow".
            schedule_type (str, optional): Schedule type. Defaults to "rf".
            encoder_depth (int, optional): The depth of the encoder (i.e. the number of blocks taking in t embeddings).
            load_pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
            use_fsdp_checkpoint (bool, optional): Whether to use FSDP gradient checkpointing. Defaults to True.
            mask_when_storing_cache (bool, optional): Whether to mask all frames when storing KV cache during autoregressive sampling.
                Otherwise only the first frame is masked. Defaults to False.
            chunk_size (int, optional): The chunk size for the transformer. Defaults to 3.
            total_num_frames (int, optional): The total number of frames. Defaults to 21.
            delete_cache_on_clear (bool, optional): Whether to delete the cache on clear. Defaults to False.
            **model_kwargs: Additional keyword arguments to pass to the FastGenNetwork constructor.
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
            chunk_size=chunk_size,
            total_num_frames=total_num_frames,
            **model_kwargs,
        )
        self._mask_when_storing_cache = mask_when_storing_cache
        self._delete_cache_on_clear = delete_cache_on_clear

    def override_transformer_forward(self, inner_dim: int) -> None:
        # Patch rope offset and helper methods
        if hasattr(self.transformer, "rope"):
            self.transformer.rope.forward = types.MethodType(_rope_forward_with_time_offset, self.transformer.rope)
        self.transformer._prepare_blockwise_causal_attn_mask = types.MethodType(
            _prepare_blockwise_causal_attn_mask, self.transformer
        )
        self.transformer._create_external_caches = types.MethodType(_create_external_caches, self.transformer)
        self.transformer.set_attn_processor = types.MethodType(_wan_set_attn_processor, self.transformer)
        self.transformer.classify_forward_prepare = types.MethodType(classify_forward_prepare, self.transformer)
        self.transformer.classify_forward_block_forward = types.MethodType(
            classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Patch block forward to inline-cache path
        for block in self.transformer.blocks:
            block.forward = types.MethodType(_wan_block_forward_inline_cache, block)

        # Install attention processor with inline caches
        self.transformer.set_attn_processor(CausalWanAttnProcessor())
        self.transformer.block_mask = None

    def clear_caches(self) -> None:
        if self._delete_cache_on_clear:
            if hasattr(self.transformer, "_external_self_kv_list"):
                del self.transformer._external_self_kv_list
            if hasattr(self.transformer, "_external_cross_kv_list"):
                del self.transformer._external_cross_kv_list
            torch.cuda.empty_cache()
            return

        # Reset external KV lengths if present (mapping by tag)
        if hasattr(self.transformer, "_external_self_kv_list"):
            for kvb in self.transformer._external_self_kv_list:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["len"] = 0
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])
        # Reset external cross-attn cache init flags if present (mapping by tag)
        if hasattr(self.transformer, "_external_cross_kv_list"):
            for kvb in self.transformer._external_cross_kv_list:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["is_init"] = False
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])

    def _compute_timestep_inputs(self, timestep: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute timestep input used for Causal Wan models.
        Optionally Expand or mask the timestep input for Wan 2.2 TI2V models.
            - I2V: Apply a mask that zeroes out the timestep for the first latent frame.
            - T2V: Use a mask tensor filled with ones.
        Different from WanI2V/network.py that expands timestep to [B, num_tokens] for Wan 2.2 TI2V models only,
            we expand timestep to [B, num_frames] for all Wan models to perform causal training/inference.

        Args:
            timestep (torch.Tensor): shape: (B, T) or (B, )
            mask (torch.Tensor): shape: (B, T, H, W)

        Return:
            timestep (torch.Tensor): shape: (B, T)
        """
        timestep = self.noise_scheduler.rescale_t(timestep)
        if timestep.ndim == 1:
            timestep = timestep.view(-1, 1)
        if mask is not None:
            p_t, _, _ = self.transformer.config.patch_size
            timestep = mask[:, ::p_t, 0, 0] * timestep

        return timestep

    def _compute_i2v_inputs(
        self,
        x_t: torch.Tensor,
        first_frame_cond: torch.Tensor,
        timestep: torch.Tensor,
        r_timestep: Optional[torch.Tensor] = None,
        cur_start_frame: int = 0,
        mask_all_frames: bool = False,
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
            cur_start_frame (int): The current start frame index. Varies during autoregressive sampling.
            mask_all_frames (bool): Whether to mask all frames. Used optionally for KV-caching during
                autoregressive sampling.
        """
        bsz, _, num_latent_frames, H, W = x_t.shape
        if self.concat_mask:
            scale_factor_temporal = 4

            # with clean images
            if mask_all_frames:
                # Need to mask all frames
                num_frames = num_latent_frames * scale_factor_temporal
                mask_lat_size = torch.ones(bsz, 1, num_frames, H, W).to(dtype=x_t.dtype, device=first_frame_cond.device)
            elif cur_start_frame == 0:
                # Not masking all, but the first frame needs to be masked
                # mask tensor to indicate whether the frame is real (clean) frame or noisy one
                num_frames = (num_latent_frames - 1) * scale_factor_temporal + 1
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
            mask_lat_size = mask_lat_size.view(bsz, -1, scale_factor_temporal, H, W)
            # mask_lat_size: (bsz, scale_factor_temporal, num_latent_frames, H, W)
            mask_lat_size = mask_lat_size.transpose(1, 2)
            latent_model_input = torch.concat([x_t, mask_lat_size, first_frame_cond], dim=1)

            timestep = self._compute_timestep_inputs(timestep)
            if r_timestep is not None:
                r_timestep = self._compute_timestep_inputs(r_timestep)
        else:
            latent_model_input = x_t
            timestep_mask = None
            if mask_all_frames:
                # Need to mask all frames
                latent_model_input = x_t
                timestep_mask = torch.ones(1, 1, num_latent_frames, H, W).to(dtype=x_t.dtype, device=x_t.device)
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

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        unpatchify_features: bool = True,
        fwd_pred_type: Optional[str] = None,
        skip_layers: Optional[List[int]] = None,
        cache_tag: str = "pos",
        cur_start_frame: int = 0,
        store_kv: bool = False,
        is_ar: bool = False,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the StableDiffusion latent diffusion score model.

        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (Dict[str, torch.Tensor]): The conditioning information.  Defaults to None.
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
            images: The images given as condition to the model.

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
        # Slice the first frame cond to match the current frame start
        if self.concat_mask:
            first_frame_cond = first_frame_cond[:, :, cur_start_frame : cur_start_frame + x_t.shape[2]]
        text_embeds = torch.stack(text_embeds, dim=0) if isinstance(text_embeds, list) else text_embeds

        i2v_inputs, timestep, r_timestep = self._compute_i2v_inputs(
            x_t,
            first_frame_cond=first_frame_cond,
            timestep=t,
            r_timestep=r,
            cur_start_frame=cur_start_frame,
            mask_all_frames=store_kv and self._mask_when_storing_cache,
        )
        kwargs = dict()
        if "encoder_hidden_states_image" in condition:
            kwargs["encoder_hidden_states_image"] = condition["encoder_hidden_states_image"]

        attention_kwargs = {
            "cache_tag": cache_tag,
            "chunk_size": self.chunk_size,
            "store_kv": store_kv,
            "cur_start_frame": cur_start_frame,
            "total_num_frames": self.total_num_frames,
            "is_ar": is_ar,
        }
        model_outputs = self.transformer(
            hidden_states=i2v_inputs,
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            r_timestep=r_timestep,
            attention_kwargs=attention_kwargs,
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
            if cur_start_frame == 0 and not self.concat_mask:
                out = self._replace_first_frame(first_frame_cond, out)

        else:
            assert isinstance(out, list)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            if cur_start_frame == 0 and not self.concat_mask:
                out[0] = self._replace_first_frame(first_frame_cond, out[0])
            return self._unpatchify_features(x_t, model_outputs) if unpatchify_features else model_outputs

        if return_logvar:
            return out, logvar
        return out

    def sample(
        self,
        noise: torch.FloatTensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 5.0,
        sample_steps: Optional[int] = 50,
        shift: float = 5.0,
        context_noise: float = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive sampling for teacher network with CFG and unipc_scheduler

        Args:
            noise (torch.Tensor): The pure noise to start from (usually a zero-mean, unit-variance Gaussian)
            condition (Dict[str, torch.Tensor], optional): conditioning information.
            neg_condition (Dict[str, torch.Tensor], optional): Optional negative conditioning information. Defaults to None.
            guidance_scale (Optional[float]): Scale of guidance. Defaults to 5.0.
            sample_steps (Optional[int]): Number of time steps to sample. Defaults to 4.
            shift (Optional[float]): Shift value of timestep scheduler. Defaults to 5.0.
            context_noise (Optional[float]): Scale of context noise in the range [0, 1]. Defaults to 0.

        Returns:
            torch.Tensor: The sample output.
        """
        assert self.schedule_type == "rf", f"{self.schedule_type} is not supported"
        # Extract first_frame_cond for replacement during sampling
        first_frame_cond = None
        if isinstance(condition, dict) and "first_frame_cond" in condition:
            first_frame_cond = condition["first_frame_cond"]
        # Temporarily set to eval mode and revert back after generation
        was_training = self.training
        self.eval()
        # Ensure caches are clean at the start of sampling
        self.clear_caches()

        x = noise
        batch_size = x.shape[0]
        num_frames = x.shape[2]

        num_chunks = num_frames // self.chunk_size
        remaining_size = num_frames % self.chunk_size
        time_rescale_factor = self.unipc_scheduler.config.num_train_timesteps
        for i in range(max(1, num_chunks)):
            if num_chunks == 0:
                start, end = 0, remaining_size
            else:
                start = 0 if i == 0 else self.chunk_size * i + remaining_size
                end = self.chunk_size * (i + 1) + remaining_size

            x_next = x[:, :, start:end]
            self.unipc_scheduler.config.flow_shift = shift
            self.unipc_scheduler.set_timesteps(num_inference_steps=sample_steps, device=noise.device)
            timesteps = self.unipc_scheduler.timesteps
            for timestep in tqdm(timesteps, total=sample_steps - 1):
                t = (timestep / time_rescale_factor).expand(batch_size)
                x_cur = x_next
                flow_pred = self(
                    x_cur,
                    t,
                    condition,
                    cache_tag="pos",  # kv-cache positive
                    cur_start_frame=start,
                    store_kv=False,
                    is_ar=True,
                )
                if guidance_scale is not None:
                    flow_uncond = self(
                        x_cur,
                        t,
                        neg_condition,
                        cache_tag="neg",  # kv-cache negative
                        cur_start_frame=start,
                        store_kv=False,
                        is_ar=True,
                    )
                    flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)
                # For Wan 2.2 TI2V: keep the first frame clean in the first chunk
                if first_frame_cond is not None and not self.concat_mask and start == 0:
                    flow_pred = flow_pred.clone()
                    flow_pred[:, :, 0] = 0.0
                x_next = self.unipc_scheduler.step(flow_pred, timestep, x_next, return_dict=False)[0]
                if first_frame_cond is not None and not self.concat_mask and start == 0:
                    x_next = x_next.clone()
                    x_next[:, :, 0] = first_frame_cond[:, :, 0]

            x[:, :, start:end] = x_next

            # rerun with timestep zero or context_noise to finalize the cache slice
            x_cache = x_next
            t_cache = torch.full((batch_size,), 0, device=x.device, dtype=x.dtype)
            if context_noise > 0:
                # Add context noise to denoised frames before caching
                t_cache = torch.full((batch_size,), context_noise, device=x.device, dtype=x.dtype)
                x_cache = self.noise_scheduler.forward_process(x_next, torch.randn_like(x_next), t_cache)

            _ = self(
                x_cache,
                t_cache,
                condition,
                cache_tag="pos",
                cur_start_frame=start,
                store_kv=True,
                is_ar=True,
            )
            if guidance_scale is not None:
                _ = self(
                    x_cache,
                    t_cache,
                    neg_condition,
                    cache_tag="neg",
                    cur_start_frame=start,
                    store_kv=True,
                    is_ar=True,
                )

        # cleanup caches after full sampling
        self.clear_caches()
        # Revert to original mode
        self.train(was_training)

        # Replace first frame
        self._replace_first_frame(condition["first_frame_cond"], x)
        return x
