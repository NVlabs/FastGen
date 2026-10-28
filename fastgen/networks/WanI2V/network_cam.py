# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# WanCamI2V: Wan I2V with camera-pose conditioning via Plücker embeddings.

import types
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import torch
from einops import rearrange

from fastgen.networks.lora_utils import scaled_lora_layers
from fastgen.networks.Wan.network import (
    classify_forward_prepare,
    compute_temb_modulation,
    project_logvar,
    project_output,
    wan_block_forward_impl,
)
from fastgen.networks.WanI2V.network import WanI2V
from fastgen.utils.camera_utils import (
    compute_relative_poses,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from fastgen.utils.tensor import as_array
from fastgen.utils.tree_utils import OutputDtype, assert_same
import fastgen.utils.logging_utils as logger


def _compute_plucker_cond(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    latent_compression_factors: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Compute Plücker embeddings for camera conditioning.

    ``intrinsics`` must already be expressed in the target ``(height, width)``
    pixel grid.

    Returns c2ws_plucker_emb [B, C, lat_f, lat_h, lat_w] ready for patch embedding.
    """
    t_factor, h_factor, w_factor = latent_compression_factors
    n_frames = len(poses)
    len_c2ws = ((n_frames - 1) // t_factor) * t_factor + 1
    poses = poses[:len_c2ws]
    Ks = intrinsics[:len_c2ws]

    # ``interpolate_camera_poses`` wraps scipy (Slerp / interp1d), which only
    # accepts numpy, so convert just the pose slices handed to it (in fp32,
    # since numpy has no bf16).
    poses_np = as_array(poses, torch.float32)
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
        src_rot_mat=poses_np[:, :3, :3],
        src_trans_vec=poses_np[:, :3, 3],
        tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // t_factor) + 1),
    )
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
    # Keep poses + intrinsics in fp32 so ``get_plucker_embeddings`` (whose meshgrid
    # inherits ``c2ws_infer``'s dtype) runs entirely in fp32.
    c2ws_infer = c2ws_infer.to(device=poses.device, dtype=torch.float32)
    Ks = Ks[0:1].to(device=poses.device, dtype=torch.float32).repeat(len(c2ws_infer), 1)

    lat_f = (len_c2ws - 1) // t_factor + 1
    lat_h = height // h_factor // patch_size[1] * patch_size[1]
    lat_w = width // w_factor // patch_size[2] * patch_size[2]

    c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, height, width)
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb,
        "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
        c1=int(height // lat_h),
        c2=int(width // lat_w),
    )
    c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
    c2ws_plucker_emb = rearrange(c2ws_plucker_emb, "b (f h w) c -> b c f h w", f=lat_f, h=lat_h, w=lat_w)
    return c2ws_plucker_emb


def _wancami2v_classify_forward_prepare(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple:
    """Prepare inputs including camera conditioning."""
    # Run standard prepare
    result = classify_forward_prepare(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        r_timestep,
        encoder_hidden_states_image,
        attention_kwargs,
    )
    (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        temb,
        rotary_emb,
    ) = result

    # Process camera conditioning if provided
    camera_hidden_states = None
    if attention_kwargs is not None and "camera_hidden_states" in attention_kwargs:
        c2ws_plucker_emb = attention_kwargs["camera_hidden_states"]
        if c2ws_plucker_emb is not None and hasattr(self, "camera_patch_embedding"):
            # c2ws_plucker_emb: [B, C, lat_f, lat_h, lat_w], C=6*c1*c2
            # Patchify like the video latent -> [B, N, C * pt * ph * pw]
            b, c, lat_f, lat_h, lat_w = c2ws_plucker_emb.shape
            pt, ph, pw = self.config.patch_size
            plucker_patched = rearrange(
                c2ws_plucker_emb,
                "b c (f ft) (h fh) (w fw) -> b (f h w) (c ft fh fw)",
                ft=pt,
                fh=ph,
                fw=pw,
            )
            plucker_embedded = self.camera_patch_embedding(plucker_patched)
            camera_hidden_states = self.camera_hidden_layer2(
                torch.nn.functional.silu(self.camera_hidden_layer1(plucker_embedded))
            )
            camera_hidden_states = camera_hidden_states + plucker_embedded

            # One camera token per video token; padding or trimming would silently inject the wrong
            # rays (a grid built for another VAE compression has 4x the tokens).
            seq_len = hidden_states.size(1)
            cam_len = camera_hidden_states.size(1)
            if cam_len != seq_len:
                raise ValueError(
                    f"camera conditioning has {cam_len} tokens but the video latent has {seq_len}: the Plücker "
                    f"grid {lat_f}x{lat_h}x{lat_w} (patch size {(pt, ph, pw)}) must be built with this model's "
                    "latent_compression_factors"
                )

            out_kwargs = dict(attention_kwargs) if attention_kwargs else {}
            out_kwargs["camera_hidden_states"] = camera_hidden_states
        else:
            out_kwargs = dict(attention_kwargs) if attention_kwargs else {}
    else:
        out_kwargs = dict(attention_kwargs) if attention_kwargs else {}

    return (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        temb,
        rotary_emb,
        out_kwargs,
    )


def _wancami2v_block_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    norm_temb: bool,
    camera_hidden_states: Optional[torch.Tensor] = None,
):
    """Block forward with optional camera conditioning injection after self-attention."""
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = compute_temb_modulation(
        self.scale_shift_table, temb, norm_temb
    )

    def _camera_inject(hidden_states):
        if camera_hidden_states is not None and hasattr(self, "cam_scale_layer"):
            # The camera layers run in the camera-conditioning dtype; as in LingBot, the modulation promotes
            # to the (fp32) residual instead of rounding it.
            cam_hs = camera_hidden_states
            cam_h = self.cam_injector_layer2(torch.nn.functional.silu(self.cam_injector_layer1(cam_hs)))
            cam_h = cam_h + cam_hs
            cam_scale = self.cam_scale_layer(cam_h)
            cam_shift = self.cam_shift_layer(cam_h)
            assert_same(dtype=True, device=True, camera_hidden_states=cam_hs, cam_scale=cam_scale, cam_shift=cam_shift)
            residual = hidden_states
            hidden_states = (1.0 + cam_scale) * residual + cam_shift
            assert_same(dtype=True, device=True, residual=residual, hidden_states=hidden_states)
        return hidden_states

    return wan_block_forward_impl(
        self,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        shift_msa,
        scale_msa,
        gate_msa,
        c_shift_msa,
        c_scale_msa,
        c_gate_msa,
        post_self_attn_fn=_camera_inject,
    )


def _wancami2v_classify_forward_block_forward(
    self,
    hidden_states: torch.Tensor,
    timestep_proj: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    r_timestep_proj: Optional[torch.Tensor] = None,
    skip_layers: Optional[list] = None,
    feature_indices: Optional[Set[int]] = None,
    return_features_early: Optional[bool] = False,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, list]:
    """Block forward loop with camera conditioning."""
    features = []
    camera_hidden_states = None
    if attention_kwargs is not None and "camera_hidden_states" in attention_kwargs:
        camera_hidden_states = attention_kwargs.get("camera_hidden_states")

    for idx, block in enumerate(self.blocks):
        if self.encoder_depth is not None and idx == self.encoder_depth and r_timestep_proj is not None:
            timestep_proj = r_timestep_proj
        # A skipped block acts as the identity; still record its feature so each requested index yields one.
        if skip_layers is None or idx not in skip_layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    timestep_proj,
                    encoder_hidden_states,
                    rotary_emb,
                    self.norm_temb,
                    camera_hidden_states,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    timestep_proj,
                    encoder_hidden_states,
                    rotary_emb,
                    self.norm_temb,
                    camera_hidden_states,
                )

        if feature_indices is not None and idx in feature_indices:
            features.append(hidden_states)
        if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
            return hidden_states, features

    return hidden_states, features


def _wancami2v_classify_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
    skip_layers: Optional[list] = None,
    **kwargs,
):
    """Classify forward that handles camera conditioning in prepare and block forward."""
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
        attention_kwargs = {}

    batch_size, num_channels, num_frames, height, width = hidden_states.shape

    p_t, p_h, p_w = self.config.patch_size

    with scaled_lora_layers(self, lora_scale):
        # Use our custom prepare that processes camera
        prep_result = self.classify_forward_prepare(
            hidden_states,
            timestep,
            encoder_hidden_states,
            r_timestep,
            encoder_hidden_states_image,
            attention_kwargs,
        )
        (
            hidden_states,
            timestep_proj,
            r_timestep_proj,
            encoder_hidden_states,
            temb,
            rotary_emb,
            attention_kwargs,
        ) = prep_result

        hidden_states, features = self.classify_forward_block_forward(
            hidden_states,
            timestep_proj,
            encoder_hidden_states,
            rotary_emb,
            r_timestep_proj,
            skip_layers,
            feature_indices,
            return_features_early,
            attention_kwargs,
        )

        if return_features_early:
            return features

        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if temb.dim() == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
            if shift.shape[1] == hidden_states.shape[1]:
                hidden_states = self.norm_out(hidden_states.float()) * (1 + scale) + shift
            else:
                frame_seqlen = post_patch_height * post_patch_width
                hs_norm_out = self.norm_out(hidden_states.float()).unflatten(1, (post_patch_num_frames, frame_seqlen))
                hidden_states = (hs_norm_out * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)).flatten(1, 2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)
            hidden_states = self.norm_out(hidden_states.float()) * (1 + scale) + shift
        hidden_states = project_output(self, hidden_states, encoder_hidden_states.dtype)

        # Use explicit channel dim instead of -1 because PyTorch cannot infer
        # -1 when the total element count is 0 (which happens when the MoE
        # router sends an empty batch through one of the experts at sample time).
        _patch_volume = p_t * p_h * p_w
        _out_channels = hidden_states.shape[-1] // _patch_volume
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, _out_channels
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        out = output if feature_indices is None else [output, features]
        if return_logvar:
            assert hasattr(
                self, "logvar_linear"
            ), "logvar_linear layer is required when return_logvar=True. Set enable_logvar_linear=True in model config."
            # One value per sample: average the time embeddings of the noised tokens.
            return out, project_logvar(self, temb, timestep, encoder_hidden_states.dtype)
        return out


class WanCamI2V(WanI2V):
    """WanCamI2V: Wan I2V with camera-pose conditioning via Plücker embeddings.

    Subclasses FastGen's WanI2V (e.g. Wan2.2-TI2V-5B, or a single ~14B expert of the Wan2.2-I2V-A14B LingBot-world MoE).
    Camera conditioning is optional; when not provided, behaves identical to WanI2V.

    For the full MoE model, use LingBotWorld which wraps two WanCamI2V experts.

    Conditioning:
        - text_embeds, first_frame_cond: Same as WanI2V
        - c2ws_plucker_emb (optional): Precomputed Plücker tensor
    """

    def __init__(
        self,
        model_id_or_local_path: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        **kwargs,
    ):
        """WanCamI2V constructor.

        Args:
            model_id_or_local_path: Wan checkpoint path (e.g. Wan2.2-TI2V-5B or Wan2.2-I2V-A14B).
            **kwargs: Passed to WanI2V.
        """
        super().__init__(model_id_or_local_path=model_id_or_local_path, **kwargs)

        self._add_camera_layers()

    @property
    def plucker_channels(self) -> int:
        """Channels of ``c2ws_plucker_emb``: six ray coordinates per pixel of a latent cell."""
        return 6 * self.latent_compression_factors[1] * self.latent_compression_factors[2]

    def _add_camera_layers(self) -> None:
        """Add camera conditioning layers to transformer and blocks."""
        # Get inner_dim from config (num_attention_heads * attention_head_dim)
        config = self.transformer.config
        inner_dim = config["num_attention_heads"] * config["attention_head_dim"]

        # Global camera embedding (Plücker -> hidden); its input is the Plücker grid patchified like the video latent
        pt, ph, pw = config["patch_size"]
        self.transformer.camera_patch_embedding = torch.nn.Linear(self.plucker_channels * pt * ph * pw, inner_dim)
        self.transformer.camera_hidden_layer1 = torch.nn.Linear(inner_dim, inner_dim)
        self.transformer.camera_hidden_layer2 = torch.nn.Linear(inner_dim, inner_dim)

        # Per-block camera injection
        for block in self.transformer.blocks:
            block.cam_injector_layer1 = torch.nn.Linear(inner_dim, inner_dim)
            block.cam_injector_layer2 = torch.nn.Linear(inner_dim, inner_dim)
            block.cam_scale_layer = torch.nn.Linear(inner_dim, inner_dim)
            block.cam_shift_layer = torch.nn.Linear(inner_dim, inner_dim)

        # Override block forward and classify_forward
        for block in self.transformer.blocks:
            block.forward = types.MethodType(_wancami2v_block_forward, block)

        self.transformer.classify_forward_prepare = types.MethodType(
            _wancami2v_classify_forward_prepare, self.transformer
        )
        self.transformer.classify_forward_block_forward = types.MethodType(
            _wancami2v_classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(_wancami2v_classify_forward, self.transformer)

        self._init_camera_layers()

        logger.info("Added camera conditioning layers to WanCamI2V")

    def _init_camera_layers(self) -> None:
        """Initialize the camera conditioning layers.

        The global Plücker embedding stack and per-block injector MLPs are
        xavier-initialized. Only the terminal scale/shift projections are
        zero-initialized, so camera conditioning starts as an identity (no-op)
        residual without permanently blocking gradients through the injector
        MLPs.
        """
        if not hasattr(self.transformer, "camera_patch_embedding"):
            return
        for m in [
            self.transformer.camera_patch_embedding,
            self.transformer.camera_hidden_layer1,
            self.transformer.camera_hidden_layer2,
        ]:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        for block in self.transformer.blocks:
            for m in [block.cam_injector_layer1, block.cam_injector_layer2]:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            for m in [block.cam_scale_layer, block.cam_shift_layer]:
                torch.nn.init.zeros_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        skip_layers: Optional[list] = None,
        unpatchify_features: bool = True,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ):
        """Forward pass. ``condition`` may carry ``c2ws_plucker_emb`` (built by ``build_conditioning``)."""
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
        assert "text_embeds" in condition
        assert "first_frame_cond" in condition

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
        attention_kwargs = {}
        camera_hidden_states = None
        if "c2ws_plucker_emb" in condition:
            camera_hidden_states = condition["c2ws_plucker_emb"]
        i2v_inputs, timestep, r_timestep = self._compute_i2v_inputs(
            x_t,
            first_frame_cond=first_frame_cond,
            timestep=t,
            r_timestep=r,
        )
        i2v_inputs = self.cast_to_model_precision(i2v_inputs)
        if camera_hidden_states is not None:
            attention_kwargs["camera_hidden_states"] = camera_hidden_states
        text_embeds = condition["text_embeds"]

        encoder_hidden_states_image = condition.get("encoder_hidden_states_image")
        assert_same(
            dtype=self.precision_model,
            device=i2v_inputs.device,
            text_embeds=text_embeds,
            c2ws_plucker_emb=camera_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
        )

        model_outputs = self.transformer(
            hidden_states=i2v_inputs,
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            r_timestep=r_timestep,
            encoder_hidden_states_image=encoder_hidden_states_image,
            attention_kwargs=attention_kwargs,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            skip_layers=skip_layers,
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

        return (out, logvar.to(dtype=x_t.dtype)) if return_logvar else out

    def compute_camera_conditioning(self, data: Dict[str, Any]) -> torch.Tensor:
        """Compute Plücker embeddings from batched camera poses/intrinsics in a data dict.

        Args:
            data: Must contain 'camera_poses' [B, N, 4, 4], 'camera_intrinsics'
                  [B, N, 4], and 'real_raw' [B, C, T, H, W] (the pixel-space
                  snapshot the trainer/loader populates). Pixel resolution is
                  read from ``real_raw.shape``; intrinsics are required to
                  already be in that pixel grid (the camera dataloader handles
                  this via ``transform_intrinsics``). The trainer has already
                  placed both camera inputs on the model device before
                  ``build_conditioning`` runs. Geometry remains fp32 on that
                  device, and ``build_conditioning`` casts the result to
                  ``precision_model``.

        Returns:
            c2ws_plucker_emb [B, C, lat_f, lat_h, lat_w]

        Raises:
            ValueError: if ``data['real_raw']`` is missing — without it the
                pixel grid is unknown and the Plücker embedding would silently
                disagree with the intrinsics.
        """
        real_raw = data.get("real_raw")
        if real_raw is None:
            raise ValueError(
                f"{type(self).__name__}.compute_camera_conditioning requires "
                "data['real_raw'] (pixel-space) to derive the Plücker pixel "
                "grid. Use a Latent+Raw or pixel dataloader, or supply mp4 in "
                "the loader's key_map."
            )
        height, width = int(real_raw.shape[3]), int(real_raw.shape[4])

        plucker_list = []
        for b in range(data["camera_poses"].shape[0]):
            plucker = _compute_plucker_cond(
                poses=data["camera_poses"][b],
                intrinsics=data["camera_intrinsics"][b],
                height=height,
                width=width,
                latent_compression_factors=self.latent_compression_factors,
                patch_size=tuple(self.transformer.config.patch_size),
            )
            plucker_list.append(plucker)
        return torch.cat(plucker_list, dim=0)

    def build_conditioning(self, data, input_shape=None, precision=None):
        """Build WanI2V conditioning, then attach camera Plücker conditioning.

        Extends `WanI2V.build_conditioning`: when camera conditioning is
        enabled and the batch carries ``camera_poses``, compute the Plücker
        embedding once and add it as ``c2ws_plucker_emb`` to the positive
        ``condition`` dict (and to ``neg_condition`` when one is present).
        ``forward`` consumes ``condition["c2ws_plucker_emb"]``, which only the
        backbone reads, in ``precision_model``. The trainer merges the returned
        dict into ``data``.
        """
        out = super().build_conditioning(data, input_shape=input_shape, precision=precision)
        if "camera_poses" in data:
            # The Plücker conditioning carries one (interpolated) frame per output
            # latent frame, so the supplied trajectory length must match the target
            # generation length. Compare in pixel-frame space: convert the target
            # latent length (input_shape[1]) to the pose-frame count it expects. A
            # mismatch otherwise surfaces as a cryptic deep shape error.
            if input_shape is not None and len(input_shape) >= 2:
                target_lat_f = int(input_shape[1])
                expected_pose_frames = self.pixel_shape(input_shape)[0]
                n_pose = int(data["camera_poses"].shape[1])
                if n_pose != expected_pose_frames:
                    raise ValueError(
                        f"camera_poses has {n_pose} pose frames but input_shape expects "
                        f"{target_lat_f} latent frames (= {expected_pose_frames} pose frames "
                        f"at {self.latent_compression_factors[0]}x temporal compression); truncate "
                        f"poses.npy/intrinsics.npy to {expected_pose_frames} frames to match the generation length."
                    )
            c2ws_plucker_emb = self.compute_camera_conditioning(data).to(dtype=self.precision_model)
            out["condition"]["c2ws_plucker_emb"] = c2ws_plucker_emb
            if out.get("neg_condition") is not None:
                out["neg_condition"]["c2ws_plucker_emb"] = c2ws_plucker_emb
        return out
