# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Dict, Any, TYPE_CHECKING, Callable, Optional
import torch
import torch.nn.functional as F

from fastgen.configs.opt import get_scheduler
from fastgen.methods import FastGenModel
from fastgen.methods.common_loss import (
    gan_loss_generator,
    gan_loss_discriminator,
)
from fastgen.utils import instantiate
import fastgen.utils.logging_utils as logger
from fastgen.utils.basic_utils import convert_cfg_to_dict

if TYPE_CHECKING:
    from fastgen.configs.methods.config_ladd import ModelConfig


class LADDModel(FastGenModel):
    def __init__(self, config: ModelConfig):
        """

        Args:
            config (ModelConfig): The configuration for the LADD model
        """
        super().__init__(config)
        self.config = config

    def build_model(self):
        super().build_model()

        # instantiate the teacher and load student weights
        self.build_teacher()
        self.load_student_weights_and_ema()

        # instantiate the discriminator
        logger.info("Instantiating the discriminator")
        # TODO: Discriminators do not yet support meta initialization
        self.discriminator = instantiate(self.config.discriminator)

    def _setup_grad_requirements(self, iteration: int) -> None:
        if iteration % self.config.student_update_freq == 0:
            # update the student
            self.net.train().requires_grad_(True)
            self.discriminator.eval().requires_grad_(False)
        else:
            # update the discriminator
            self.net.eval().requires_grad_(False)
            self.discriminator.train().requires_grad_(True)

    def _generate_noise_and_time(
        self, real_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random noises and time step

        Args:
            batch_size: Batch size
            real_data: Real data tensor for dtype/device reference

        Returns:
            rand_z_max: Random noise used by the student
            t_max: Time step used by the student
            t: Time step
            eps: Random noise used by a forward process
        """
        batch_size = real_data.shape[0]
        eps_student = torch.randn(batch_size, *self.input_shape, device=self.device, dtype=real_data.dtype)

        if self.config.student_sample_steps == 1:
            # perform single-step distillation
            # input noise to student (sigma * eps)
            t_student = torch.full(
                (batch_size,),
                self.net.noise_scheduler.max_t,
                device=self.device,
                dtype=self.net.noise_scheduler.t_precision,
            )
            input_student = self.net.noise_scheduler.latents(noise=eps_student)
        else:
            # perform multiple-step distillation
            # Add noise to real image data (for multistep generation)
            t_student = self.net.noise_scheduler.sample_from_t_list(
                batch_size,
                sample_steps=self.config.student_sample_steps,
                t_list=self.config.sample_t_cfg.t_list,
                device=self.device,
            )
            input_student = self.net.noise_scheduler.forward_process(real_data, eps_student, t_student)

        t = self.net.noise_scheduler.sample_t(
            batch_size, **convert_cfg_to_dict(self.config.sample_t_cfg), device=self.device
        )
        eps = torch.randn_like(real_data, device=self.device, dtype=real_data.dtype)

        return input_student, t_student, t, eps

    def _student_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        data: Dict[str, Any],
        condition: Optional[Any] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """Perform student model update step.

        Args:
            input_student: Input tensor to student network
            t_student: Input time to student network
            t: Time step
            eps: Noise tensor
            data: Original data batch
            condition: Conditioning information

        Returns:
            tuple of (loss_map, outputs)
        """
        # Generate data from student
        gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)
        perturbed_data = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        # Compute the discriminator score
        fake_feat = self.teacher(
            perturbed_data,
            t,
            condition=condition,
            return_features_early=True,
            feature_indices=self.discriminator.feature_indices,
        )

        # Compute the GAN loss for the generator
        gan_loss_gen = gan_loss_generator(self.discriminator(fake_feat))

        # Build output dictionaries
        loss_map = {
            "total_loss": gan_loss_gen,
            "gan_loss_gen": gan_loss_gen,
        }
        outputs = self._get_outputs(gen_data, input_student, condition=condition)

        return loss_map, outputs

    def _compute_real_feat(
        self, real_data: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, condition: Optional[Any] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute discriminator features for both real and fake data.

        Args:
            real_data: Real data tensor
            condition: Conditioning information
            t: Time step
            eps: Noise tensor

        Returns:
            tuple of (real_feat, t_real)
        """
        # decide whether to use the same t and noise for real and fake data
        if self.config.gan_use_same_t_noise:
            t_real = t
            eps_real = eps
        else:
            t_real = self.net.noise_scheduler.sample_t(
                real_data.shape[0],
                **convert_cfg_to_dict(self.config.sample_t_cfg),
                device=self.device,
            )
            eps_real = torch.randn_like(real_data)
        # Perturb the real data according to the given forward process
        perturbed_real = self.net.noise_scheduler.forward_process(real_data, eps_real, t_real)
        real_feat = self.teacher(
            perturbed_real,
            t_real,
            condition=condition,
            return_features_early=True,
            feature_indices=self.discriminator.feature_indices,
        )

        return real_feat, t_real

    def _compute_r1_regularization(
        self,
        real_feat_logit: torch.Tensor,
        real_data: torch.Tensor,
        t_real: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> torch.Tensor:
        """Compute R1 regularization loss for discriminator.

        Args:
            real_feat_logit: Real feature logits
            real_data: Real data tensor
            t_real: Time step for real data
            condition: Conditioning information

        Returns:
            R1 regularization loss
        """
        perturbed_real_alpha = real_data.add(self.config.gan_r1_reg_alpha * torch.randn_like(real_data))
        with torch.no_grad():
            real_feat_alpha = self.teacher(
                perturbed_real_alpha,
                t_real,
                condition=condition,
                return_features_early=True,
                feature_indices=self.discriminator.feature_indices,
            )
        real_feat_alpha_logit = self.discriminator(real_feat_alpha)
        gan_loss_ar1 = F.mse_loss(real_feat_logit, real_feat_alpha_logit, reduction="mean")

        return gan_loss_ar1

    def _discriminator_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        real_data: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Perform fake score and discriminator update step.

        Args:
            input_student: Input tensor to student network
            t_student: Input time to student network
            t: Time steps
            eps: Noise tensor
            real_data: Real data tensor
            condition: Conditioning information

        Returns:
            tuple of (loss_map, outputs)
        """
        with torch.no_grad():
            # generate data and compute fake score loss
            gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)
            x_t_sg = self.net.noise_scheduler.forward_process(gen_data, eps, t)

            # extract real and fake features from teacher
            fake_feat = self.teacher(
                x_t_sg,
                t,
                condition=condition,
                return_features_early=True,
                feature_indices=self.discriminator.feature_indices,
            )

            real_feat, t_real = self._compute_real_feat(real_data=real_data, t=t, eps=eps, condition=condition)

        real_feat_logit = self.discriminator(real_feat)
        gan_loss_disc = gan_loss_discriminator(real_feat_logit, self.discriminator(fake_feat))

        # Use approximate R1 regularization in the APT paper to regularize the discriminator head
        gan_loss_ar1 = torch.zeros_like(gan_loss_disc)
        if self.config.gan_r1_reg_weight > 0:
            gan_loss_ar1 = self._compute_r1_regularization(real_feat_logit, real_data, t_real, condition=condition)

        total_loss = gan_loss_disc + self.config.gan_r1_reg_weight * gan_loss_ar1
        loss_map = {
            "gan_loss_disc": gan_loss_disc,
            "total_loss": total_loss,
        }
        if self.config.gan_r1_reg_weight > 0:
            loss_map.update({"gan_loss_ar1": gan_loss_ar1})
        outputs = self._get_outputs(gen_data, input_student, condition=condition)

        return loss_map, outputs

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        if self.config.student_sample_steps == 1:
            assert input_student is not None, "input_student must not be None"
            noise = input_student / self.net.noise_scheduler.max_sigma
            return {"gen_rand": gen_data, "input_rand": noise}
        else:
            noise = torch.randn_like(gen_data, dtype=self.precision)
            gen_rand_func = partial(
                self.generator_fn,
                net=self.net_inference,
                noise=noise,
                condition=condition,
                student_sample_steps=self.config.student_sample_steps,
                student_sample_type=self.config.student_sample_type,
                t_list=self.config.sample_t_cfg.t_list,
                precision_amp=self.precision_amp_infer,
            )
            return {"gen_rand": gen_rand_func, "input_rand": noise, "gen_rand_train": gen_data}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for Latent Adversarial Distillation (LADD)

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """

        # Prepare training data and conditions
        real_data, condition, _ = self._prepare_training_data(data)

        # Set up gradient requirements based on training phase
        self._setup_grad_requirements(iteration)

        # Generate noise and time steps
        input_student, t_student, t, eps = self._generate_noise_and_time(real_data)

        # Choose between student update or fake_score/discriminator update
        if iteration % self.config.student_update_freq == 0:
            return self._student_update_step(input_student, t_student, t, eps, data, condition=condition)
        else:
            return self._discriminator_update_step(input_student, t_student, t, eps, real_data, condition=condition)

    def init_optimizers(self):
        """Initialize optimizers, lr_schedulers and grad_scalers"""
        super().init_optimizers()

        # instantiate the optimizer for discriminator
        self.discriminator_optimizer = instantiate(self.config.discriminator_optimizer, model=self.discriminator)

        # instantiate the lr scheduler for discriminator
        self.discriminator_lr_scheduler = get_scheduler(
            self.discriminator_optimizer, self.config.discriminator_scheduler
        )

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """
        Get the optimizers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if iteration % self.config.student_update_freq == 0:
            return [self.net_optimizer]
        else:
            return [self.discriminator_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler]:
        """
        Get the lr schedulers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if iteration % self.config.student_update_freq == 0:
            return [self.net_lr_scheduler]
        else:
            return [self.discriminator_lr_scheduler]

    @property
    def model_dict(self):
        """Return the model dict containing the student and discriminator models"""
        _model_dict = super().model_dict
        _model_dict["discriminator"] = self.discriminator
        return _model_dict

    @property
    def optimizer_dict(self):
        """Return a dict containing all the optimizers"""
        _optimizer_dict = super().optimizer_dict
        _optimizer_dict["discriminator"] = self.discriminator_optimizer
        return _optimizer_dict

    @property
    def scheduler_dict(self):
        """Return a dict containing all the lr schedulers"""
        _scheduler_dict = super().scheduler_dict
        _scheduler_dict["discriminator"] = self.discriminator_lr_scheduler
        return _scheduler_dict
