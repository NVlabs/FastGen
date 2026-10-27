# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Any, Callable

import torch
from fastgen.methods import FastGenModel
from fastgen.methods.causal_sampling import CausalStudentSamplingMixin
from fastgen.methods.common_loss import mse_loss
from fastgen.networks.network_causal import CausalContext
from fastgen.utils.tensor import expand_like


class KDModel(FastGenModel):
    state_fields = FastGenModel.state_fields | {"noise", "path"}

    def build_model(self):
        super().build_model()
        self.load_student_weights_and_ema()

    def _kd_path_indices(self, path_steps: int, target_steps: int) -> torch.Tensor:
        """Validate a KD path against the anchored grid and map target step indices into it."""
        noise_scheduler = self.net.noise_scheduler
        base_steps = noise_scheduler.student_base_steps
        if base_steps is None:
            raise ValueError(
                "KD path data requires an anchored student grid; set "
                "student_grid_cfg.grid_kwargs.base_steps for a native grid or "
                "configure an explicit grid"
            )
        if path_steps != base_steps:
            raise ValueError(
                "KD path resolution must match the configured student base grid: "
                f"path_steps={path_steps}, student_base_steps={base_steps}"
            )
        return noise_scheduler.student_grid_indices(
            target_steps=target_steps,
            device=self.device,
        )

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for knowledge distillation model.

        Important! For path-backed KD, the anchored student base grid must be aligned with the
        `path` timesteps. The path contains one non-clean state per base-grid interval, while `real`
        contains the clean endpoint. Please check `path_timesteps` in the paired dataset's index.json.

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """
        denoised_data = data["real"]
        condition = data.get("condition")
        neg_condition = data.get("neg_condition")
        batch_size = denoised_data.shape[0]

        noise_scheduler = self.net.noise_scheduler
        student_steps = noise_scheduler.default_student_steps
        denoise_path = data.get("path")
        use_noise = student_steps == 1 and "noise" in data
        if not use_noise and denoise_path is None:
            required = "Noise or path" if student_steps == 1 else "Path"
            raise ValueError(f"{required} must be provided for KDModel")

        base_indices = None if use_noise else self._kd_path_indices(denoise_path.shape[1], student_steps)
        if student_steps == 1:
            t_student = noise_scheduler.student_grid(device=self.device)[0].expand(batch_size)
            if use_noise:
                input_student = noise_scheduler.latents(noise=data["noise"], t_init=t_student)
            else:
                input_student = denoise_path[:, base_indices[0], ...]
        else:
            t_student, student_step_indices = noise_scheduler.sample_student_timesteps(
                batch_size,
                num_steps=student_steps,
                return_indices=True,
                device=self.device,
            )
            path_indices = base_indices[student_step_indices]
            path_indices = expand_like(path_indices, denoise_path).expand(
                -1, -1, *denoise_path.shape[2:]
            )  # [batch_size, 1, ...]
            input_student = torch.gather(denoise_path, 1, path_indices).squeeze(1)

        gen_data = self.net(
            input_student,
            t_student,
            condition=condition,
            fwd_pred_type="x0",
        )

        # Compute the l2 loss between the generated data and the denoised data
        loss = 0.5 * mse_loss(gen_data, denoised_data)

        # Build output dictionaries
        loss_map = {
            "total_loss": loss,
            "recon_loss": loss,
        }
        outputs = self._get_outputs(gen_data, condition=condition, neg_condition=neg_condition)

        # Compute rewards
        loss_map = self.apply_reward(gen_data, data, loss_map)

        return loss_map, outputs


class CausalKDModel(CausalStudentSamplingMixin, KDModel):
    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for knowledge distillation model.

        Important! The anchored student base grid must be aligned with the `path` timesteps.
        Please check the `path_timesteps` item in index.json of the paired dataset.

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """
        denoise_path = data["path"]  # shape is [batch_size, num_inf_steps, C, num_frames, H, W]
        denoised_data = data["real"]  # [batch_size, C, num_frames, H, W]
        condition = data.get("condition")
        neg_condition = data.get("neg_condition")
        batch_size, num_frames = denoise_path.shape[0], denoise_path.shape[3]
        chunk_size = self.net.chunk_size
        noise_scheduler = self.net.noise_scheduler
        target_steps = noise_scheduler.default_student_steps
        base_indices = self._kd_path_indices(denoise_path.shape[1], target_steps)

        # Sample one runtime student-grid step index per causal chunk.
        t_inhom, student_step_indices = noise_scheduler.sample_student_timesteps_inhom(
            batch_size,
            num_frames,
            chunk_size,
            num_steps=target_steps,
            device=self.device,
        )  # [batch_size, num_frames]
        path_step = base_indices[student_step_indices]
        expand_shape = [path_step.shape[0], 1, 1, path_step.shape[1]] + [1] * max(0, denoise_path.ndim - 4)
        path_indices = path_step.view(expand_shape).expand(
            -1, -1, *denoise_path.shape[2:]
        )  # [batch_size, 1, C, num_frames, H, W]

        denoise_path_all = torch.cat([denoise_path, denoised_data.unsqueeze(1)], dim=1)  # gather clean data
        noisy_data = torch.gather(denoise_path_all, 1, path_indices).squeeze(1)  # [batch_size, C, num_frames, H, W]

        # generate data, attending block-causally over the full video like CausalSFT and CausVid
        causal_ctx = CausalContext(
            cache_tag="pos",
            chunk_size=chunk_size,
            total_num_frames=num_frames,
            store_kv=False,
            is_ar=False,
        )
        gen_data = self.net(
            noisy_data,
            t_inhom,
            condition=condition,
            fwd_pred_type="x0",
            causal_ctx=causal_ctx,
        )

        # Compute the l2 loss between the generated data and the denoised data
        loss = 0.5 * mse_loss(gen_data, denoised_data)

        # Build output dictionaries
        loss_map = {
            "total_loss": loss,
            "recon_loss": loss,
        }
        outputs = self._get_outputs(gen_data, condition=condition, neg_condition=neg_condition)

        # Compute rewards
        loss_map = self.apply_reward(gen_data, data, loss_map)

        return loss_map, outputs
