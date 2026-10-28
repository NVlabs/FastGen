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

from typing import Optional, List, Set, Union, Tuple, Dict

from tqdm.auto import tqdm
import torch
from fastgen.networks.network_causal import CausalContext, causal_chunk_ranges
from fastgen.networks.Wan.network_causal import CausalWan
from fastgen.networks.WanI2V.network import WanI2V
from fastgen.utils.tree_utils import OutputDtype, assert_same
from fastgen.utils.guidance import (
    GuidanceConfig,
    predict_with_guidance,
)


class CausalWanI2V(CausalWan, WanI2V):
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
        **kwargs,
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
            **kwargs: Additional keyword arguments passed to ``WanI2V``.
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
            **kwargs,
        )
        self._mask_when_storing_cache = mask_when_storing_cache

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
        causal_ctx: Optional[CausalContext] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the WanI2V diffusion score model.

        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (Optional[Dict[str, torch.Tensor]]): The conditioning information. Defaults to None.
            r (torch.Tensor): Another timestep mainly used by meanflow.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. Any provided set returns a feature list alongside the output,
                including an empty list for an empty set. None returns only the model output.
            return_logvar: If true, the forward pass also returns the logvar, shape [B, 1].
            unpatchify_features: If true, the features are unpatchified to [B, C', T, H, W] (C' = hidden size / patch
                volume); otherwise they stay packed as [B, N_tokens, D].
            fwd_pred_type: Update the network prediction type, must be in ['x0', 'eps', 'v', 'flow'].
                None means using the original net_pred_type.
            skip_layers: Apply skip-layer guidance by skipping layers of the unconditional network during forward pass.
            causal_ctx: Causal rollout state for the KV-cache path. None attends to the whole sequence without a
                mask; causal training and sampling always pass a context.
            apply_output_conditioning: Pin frame zero in the first chunk after the conversion.
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

        cur_start_frame = 0 if causal_ctx is None else causal_ctx.cur_start_frame
        store_kv = False if causal_ctx is None else causal_ctx.store_kv

        first_frame_cond = condition["first_frame_cond"]
        # Slice the first frame cond to match the current frame start
        if self.concat_mask:
            first_frame_cond = first_frame_cond[:, :, cur_start_frame : cur_start_frame + x_t.shape[2]]

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
            cur_start_frame=cur_start_frame,
            mask_all_frames=store_kv and self._mask_when_storing_cache,
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

        # Pass the primary tensor inputs positionally so FSDP2 input handling is
        # preserved across activation-checkpoint recompute.
        model_outputs = self.transformer(
            i2v_inputs,
            timestep,
            text_embeds,
            r_timestep=r_timestep,
            causal_ctx=causal_ctx,
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
            out = self.transform_output(
                out, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning, causal_ctx=causal_ctx
            )
        else:
            assert isinstance(out, list)
            out[0] = self.transform_output(
                out[0], x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning, causal_ctx=causal_ctx
            )
            out[1] = self._unpatchify_features(i2v_inputs, out[1]) if unpatchify_features else out[1]

        if return_logvar:
            return out, logvar.to(dtype=x_t.dtype)
        return out

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        sample_steps: Optional[int] = None,
        sample_type: str = "ode",
        shift: Optional[float] = None,
        context_noise: Optional[float] = None,
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Autoregressive sampling for teacher network with CFG and UniPC.

        Args:
            noise (torch.Tensor): The pure noise to start from (usually a zero-mean, unit-variance Gaussian)
            condition (Optional[Dict[str, torch.Tensor]]): Conditioning information. Defaults to None.
            neg_condition (Optional[Dict[str, torch.Tensor]]): Optional negative conditioning information. Defaults to None.
            sample_steps (Optional[int]): Number of time steps to sample. Defaults to None, the official setting from
                ``resolve_sample_settings``.
            sample_type: Integration policy; only ``"ode"`` is supported.
            shift (Optional[float]): Shift value of timestep scheduler. Defaults to None, the official setting from
                ``resolve_sample_settings``.
            context_noise (Optional[float]): Time in [0, 1] to which each finished chunk is re-noised before
                caching; None caches the clean chunk. Defaults to None.
            guidance_cfg (Optional[GuidanceConfig]): Composite guidance config; ``None`` →
                ``WanI2V.default_guidance_cfg()`` (CFG at 5.0).

        Returns:
            torch.Tensor: The sample output.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )
        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg()
        sample_steps, shift = self.resolve_sample_settings(noise.shape[1:], sample_steps, shift)

        # Ensure caches are clean at the start of sampling.
        self.clear_caches()

        kv_cache_engine = self.transformer.kv_cache_engines["main"]

        # One ``(cache_key, ctx, cond)`` triple per active guidance forward.
        def new_ctx(cache_tag):
            return CausalContext(
                cache_tag=cache_tag,
                cache_engines={"main": kv_cache_engine},
                chunk_size=self.chunk_size,
                total_num_frames=self.total_num_frames,
                is_ar=True,
            )

        ctxs = [
            (key, new_ctx(key), guidance_cfg.condition_for_pred_key(key, condition, neg_condition))
            for key in guidance_cfg.active_pred_keys()
        ]

        # Create initial latents.
        scheduler, t_grid = self.sample_scheduler_and_grid(sample_steps, flow_shift=shift, device=noise.device)
        x = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])
        first_frame_cond = condition.get("first_frame_cond") if isinstance(condition, dict) else None

        batch_size = x.shape[0]
        num_frames = x.shape[2]

        for start, end in causal_chunk_ranges(num_frames, self.chunk_size):
            x_next = x[:, :, start:end]
            # Reset scheduler state (model_outputs, lower_order_nums, etc.).
            # Reset the UniPC multistep state for this chunk; the grid (``timesteps`` / ``t_grid``) is unchanged.
            self.sample_scheduler_and_grid(sample_steps, flow_shift=shift, device=noise.device)

            # Configure all contexts for denoising (read-only cache access).
            for _, ctx, _ in ctxs:
                ctx.cur_start_frame = start
                ctx.store_kv = False

            for sched_step, t in tqdm(
                zip(scheduler.timesteps, t_grid, strict=True), total=len(t_grid), desc="Sampling"
            ):
                t = t.expand(batch_size)
                x_cur = x_next
                flow_pred = predict_with_guidance(
                    self,
                    x_cur,
                    t,
                    guidance_cfg=guidance_cfg,
                    condition=condition,
                    neg_condition=neg_condition,
                    fwd_pred_type="flow",
                    contexts={key: {"causal_ctx": ctx} for key, ctx, _ in ctxs},
                )
                # For Wan 2.2 TI2V: keep the first frame clean in the first chunk
                if first_frame_cond is not None and not self.concat_mask and start == 0:
                    flow_pred = flow_pred.clone()
                    flow_pred[:, :, 0] = 0.0
                x_next = scheduler.step(flow_pred, sched_step, x_next, return_dict=False)[0]
                if first_frame_cond is not None and not self.concat_mask and start == 0:
                    x_next = x_next.clone()
                    x_next[:, :, 0] = first_frame_cond[:, :, 0]

            x[:, :, start:end] = x_next

            # rerun with timestep zero or context_noise to finalize the cache slice
            x_cache = x_next
            t_cache = torch.full((batch_size,), 0, device=x.device, dtype=self.noise_scheduler.t_precision)
            if context_noise is not None:
                # Add context noise to denoised frames before caching
                t_cache = torch.full(
                    (batch_size,), context_noise, device=x.device, dtype=self.noise_scheduler.t_precision
                )
                x_cache = self.noise_scheduler.forward_process(x_next, torch.randn_like(x_next), t_cache)

            # Store KV cache for this chunk (one finalization pass per active ctx).
            for _, ctx, cond in ctxs:
                with ctx.set_store_kv(True):
                    _ = self(x_cache, t_cache, condition=cond, causal_ctx=ctx, apply_output_conditioning=False)

        # cleanup caches after full sampling
        self.clear_caches()
        return x
