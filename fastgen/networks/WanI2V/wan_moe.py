# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""I2V variants of the WanMoE wrapper.

- ``WanMoEI2V``: Wan2.2 I2V A14B (two ``WanI2V`` experts).
- ``LingBotWorld``: LingBot-world camera-conditioned MoE (two ``WanCamI2V`` experts).
"""

import json
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

from fastgen.networks.Wan.wan_moe import WanMoE
from fastgen.networks.WanI2V.network import WanI2V
from fastgen.networks.WanI2V.network_cam import WanCamI2V
from fastgen.utils.guidance import GuidanceConfig, predict_with_guidance
from fastgen.utils.env import hf_download
from fastgen.utils.module import load_module_state_dict
import fastgen.utils.logging_utils as logger


class WanMoEI2V(WanMoE):
    """WanMoE for I2V (Wan2.2 I2V A14B)."""

    DEFAULT_GUIDANCE_BOUNDARY = 0.9

    def __init__(
        self,
        model_id_or_local_path: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        guidance_boundary: float = DEFAULT_GUIDANCE_BOUNDARY,
        **kwargs,
    ):
        super().__init__(
            wan_cls=WanI2V,
            model_id_or_local_path=model_id_or_local_path,
            guidance_boundary=guidance_boundary,
            **kwargs,
        )

    @classmethod
    def default_guidance_cfg(cls, guidance_boundary: Optional[float] = None) -> GuidanceConfig:
        """Default guidance recipe: CFG ``[3.5, 3.5]`` across the expert-routing boundary.

        Wan2.2 I2V A14B uses the same CFG scale (3.5) for the low-noise
        (``t < boundary``) and high-noise (``t >= boundary``) experts.
        ``guidance_boundary`` defaults to the model's base
        ``DEFAULT_GUIDANCE_BOUNDARY``. Experiments may override via
        ``config.model.guidance_cfg``.
        """
        g = GuidanceConfig()
        g.cfg.scale = [3.5, 3.5]
        g.cfg.boundary = [cls.DEFAULT_GUIDANCE_BOUNDARY if guidance_boundary is None else guidance_boundary]
        return g

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
        """I2V multistep sample with MoE routing and CFG via ``predict_with_guidance``.

        Keeps the Wan2.2 TI2V first-frame pinning (clean first latent frame
        preserved across scheduler steps) when ``concat_mask`` is False.
        ``sample_steps`` and ``shift`` default (None) to the official settings
        from ``WanI2V.resolve_sample_settings``.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )

        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg(self.guidance_boundary)
        sample_steps, shift = self.resolve_sample_settings(noise.shape[1:], sample_steps, shift)

        scheduler, t_grid = self.sample_scheduler_and_grid(sample_steps, flow_shift=shift, device=noise.device)
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_grid[0])
        first_frame_cond = condition.get("first_frame_cond") if isinstance(condition, dict) else None

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

            if first_frame_cond is not None and not self.concat_mask:
                flow_pred = flow_pred.clone()
                flow_pred[:, :, 0] = 0.0

            latents = scheduler.step(flow_pred, sched_step, latents, return_dict=False)[0]

            if first_frame_cond is not None and not self.concat_mask:
                latents = latents.clone()
                latents[:, :, 0] = first_frame_cond[:, :, 0]

        return latents


class LingBotWorld(WanMoE):
    """LingBot-world: MoE camera-conditioned I2V model.

    LingBot-world ships as a Wan2.2-I2V-A14B MoE with two expert checkpoint subfolders
    (``high_noise_model`` and ``low_noise_model``), each a WanCamI2V with
    camera-pose conditioning via Plücker embeddings.

    See: https://github.com/robbyant/lingbot-world
    """

    MODEL_ID = "robbyant/lingbot-world-base-cam"

    DEFAULT_GUIDANCE_BOUNDARY = 0.947

    LINGBOT_CAMERA_KEY_MAPPING = [
        ("patch_embedding_wancamctrl", "camera_patch_embedding"),
        ("c2ws_hidden_states_layer1", "camera_hidden_layer1"),
        ("c2ws_hidden_states_layer2", "camera_hidden_layer2"),
    ]

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID,
        guidance_boundary: float = DEFAULT_GUIDANCE_BOUNDARY,
        load_pretrained: bool = True,
        **kwargs,
    ):
        # Build the two 14B I2V expert architectures without loading base Wan weights;
        # actual weights come from the LingBot checkpoint below.
        super().__init__(
            wan_cls=WanCamI2V,
            model_id_or_local_path=WanI2V.MODEL_ID_VER_2_2_I2V_A14B,
            guidance_boundary=guidance_boundary,
            load_pretrained=False,
            **kwargs,
        )
        if load_pretrained:
            if not self._is_in_meta_context():
                self._load_expert_weights(
                    self.network_high, model_id_or_local_path, checkpoint_subfolder="high_noise_model"
                )
                self._load_expert_weights(
                    self.network_low, model_id_or_local_path, checkpoint_subfolder="low_noise_model"
                )
            else:
                logger.info("Skipping LingBot weight loading (meta device init, weights will be synced via FSDP)")

    def _remap_lingbot_keys(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Rename LingBot checkpoint keys to WanCamI2V camera layer names."""
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            for old, new in self.LINGBOT_CAMERA_KEY_MAPPING:
                if old in new_k:
                    new_k = new_k.replace(old, new)
                    break
            remapped[new_k] = v
        return remapped

    def _load_expert_weights(self, expert: WanCamI2V, model_id_or_local_path: str, checkpoint_subfolder: str) -> Any:
        """Load weights from a LingBot-world expert subfolder of a local directory or a Hugging Face repo.

        Returns the ``expert.load_state_dict`` result (a torch ``_IncompatibleKeys``
        with the missing/unexpected keys).
        """
        logger.info(f"Loading LingBot-world weights from {model_id_or_local_path}/{checkpoint_subfolder}")
        index_path = hf_download(
            model_id_or_local_path, "diffusion_pytorch_model.safetensors.index.json", subfolder=checkpoint_subfolder
        )
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        shard_files = sorted(set(index.get("weight_map", {}).values()))

        state_dict = {}
        for shard in tqdm(shard_files, desc=f"Loading {checkpoint_subfolder}"):
            state_dict.update(load_file(hf_download(model_id_or_local_path, shard, subfolder=checkpoint_subfolder)))

        state_dict = self._remap_lingbot_keys(state_dict)
        # WanCamI2V inherits Wan.load_state_dict, which converts official Wan keys
        # after the LingBot-specific camera-key remapping above.
        return load_module_state_dict(
            expert,
            state_dict,
            strict=False,
            require_any=True,
            description=f"LingBot-world {checkpoint_subfolder}",
        )

    def load_state_dict(self, state_dict, **kwargs):
        """Remap LingBot camera-layer keys before loading."""
        state_dict = self._remap_lingbot_keys(state_dict)
        return super().load_state_dict(state_dict, **kwargs)

    def compute_camera_conditioning(self, data: Dict[str, Any]) -> torch.Tensor:
        """Compute camera conditioning (delegates to high expert)."""
        return self.network_high.compute_camera_conditioning(data)

    @classmethod
    def default_guidance_cfg(cls, guidance_boundary: Optional[float] = None) -> GuidanceConfig:
        """Default guidance recipe: CFG ``[5.0, 5.0]`` across the expert-routing boundary.

        LingBot-world uses the same CFG scale (5.0) for the low-noise
        (``t < boundary``) and high-noise (``t >= boundary``) experts.
        ``guidance_boundary`` defaults to the model's base
        ``DEFAULT_GUIDANCE_BOUNDARY``. Experiments may override via
        ``config.model.guidance_cfg``.
        """
        g = GuidanceConfig()
        g.cfg.scale = [5.0, 5.0]
        g.cfg.boundary = [cls.DEFAULT_GUIDANCE_BOUNDARY if guidance_boundary is None else guidance_boundary]
        return g

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        sample_steps: int = 70,
        sample_type: str = "ode",
        shift: float = 10.0,
        guidance_cfg: Optional[GuidanceConfig] = None,
    ) -> torch.Tensor:
        """Sample from LingBotWorld using the routed MoE forward (camera-conditioned).

        CFG (incl. skip-layer guidance via ``guidance_cfg.cfg.skip_layers``) is
        driven by ``predict_with_guidance``; each guidance pass runs through the
        per-frame high/low expert routing in ``forward``.
        """
        self._validate_sample_kwargs(
            sample_type,
            supported_schedule_types=("rf",),
            supported_sample_types=("ode",),
        )

        if guidance_cfg is None:
            guidance_cfg = self.default_guidance_cfg(self.guidance_boundary)

        noise_scheduler = self.network_high.noise_scheduler
        scheduler, t_grid = self.network_high.sample_scheduler_and_grid(
            sample_steps, flow_shift=shift, device=noise.device
        )
        latents = noise_scheduler.latents(noise=noise, t_init=t_grid[0])

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

            latents = scheduler.step(flow_pred, sched_step, latents, return_dict=False)[0]

        return latents
