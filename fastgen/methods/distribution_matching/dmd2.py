# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Dict, Any, TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F

from fastgen.configs.opt import get_scheduler
from fastgen.utils import instantiate
from fastgen.methods import FastGenModel
from fastgen.methods.common_loss import (
    denoising_score_matching_loss,
    variational_score_distillation_loss,
    gan_loss_generator,
    gan_loss_discriminator,
)
import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed import synchronize, is_rank0
from fastgen.utils.basic_utils import convert_cfg_to_dict


if TYPE_CHECKING:
    from fastgen.configs.methods.config_dmd2 import ModelConfig


class DMD2Model(FastGenModel):
    def __init__(self, config: ModelConfig):
        """

        Args:
            config (ModelConfig): The configuration for the DMD model
        """
        super().__init__(config)
        self.config = config

    def build_model(self):
        super().build_model()
        self.build_teacher()
        self.load_student_weights_and_ema()

        # instantiate the fake_score and load weights from teacher
        logger.info("Instantiating the fake_score")
        with self._get_meta_init_context():
            self.fake_score = instantiate(self.teacher_config)
        model_path = self.config.pretrained_model_path
        if model_path is not None and len(model_path) > 0:
            if (not self.config.fsdp_meta_init) or is_rank0():
                # Only rank 0 loads weights if using meta initialization
                self.fake_score.load_state_dict(self.teacher.state_dict())
            synchronize()

        if self.config.gan_loss_weight_gen > 0:
            logger.info(f"gan_loss_weight_gen: {self.config.gan_loss_weight_gen}")
            # instantiate the discriminator in DMD2
            logger.info("Instantiating the discriminator")
            if getattr(self.config.discriminator, "disc_type", None) is not None:
                logger.info(f"Discriminator type: {self.config.discriminator.disc_type}")
            # TODO: Discriminators do not yet support meta initialization
            self.discriminator = instantiate(self.config.discriminator)
            synchronize()
        torch.cuda.empty_cache()

    def _setup_grad_requirements(self, iteration: int) -> None:
        if iteration % self.config.student_update_freq == 0:
            # update the student
            self.fake_score.eval().requires_grad_(False)
            if self.config.gan_loss_weight_gen > 0:
                self.discriminator.eval().requires_grad_(False)
        else:
            # update the fake_score and discriminator
            self.fake_score.train().requires_grad_(True)
            if self.config.gan_loss_weight_gen > 0:
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

    def _compute_teacher_x0(
        self, perturbed_data: torch.Tensor, t: torch.Tensor, condition: Optional[Any] = None
    ) -> torch.Tensor:
        """Compute the teacher x0 target used by VSD."""
        with torch.no_grad():
            teacher_x0 = self.teacher(
                perturbed_data,
                t,
                condition=condition,
                fwd_pred_type="x0",
            )
        return teacher_x0.detach()

    def _compute_generator_gan_loss(
        self, perturbed_data: torch.Tensor, t: torch.Tensor, condition: Optional[Any] = None
    ) -> torch.Tensor:
        """Compute the generator-side GAN loss from fake_score features."""
        if self.config.gan_loss_weight_gen <= 0:
            return torch.tensor(0.0, device=self.device, dtype=perturbed_data.dtype)

        fake_feat = self.fake_score(
            perturbed_data,
            t,
            condition=condition,
            return_features_early=True,
            feature_indices=self.discriminator.feature_indices,
        )
        return gan_loss_generator(self.discriminator(fake_feat))

    def _apply_classifier_free_guidance(
        self,
        perturbed_data: torch.Tensor,
        t: torch.Tensor,
        teacher_x0: torch.Tensor,
        neg_condition: Optional[Any] = None,
    ) -> torch.Tensor:
        """Apply classifier-free guidance to teacher predictions.

        Args:
            perturbed_data: Perturbed data
            t: Time step
            teacher_x0: Original teacher x0 prediction
            neg_condition: Negative conditioning for CFG

        Returns:
            CFG-adjusted teacher_x0
        """
        assert self.config.guidance_scale is not None, "guidance_scale must be provided"
        # classifier-free guidance
        with torch.no_grad():
            kwargs = {"condition": neg_condition, "fwd_pred_type": "x0"}
            if self.config.skip_layers is not None:
                kwargs["skip_layers"] = self.config.skip_layers
            teacher_x0_neg = self.teacher(perturbed_data, t, **kwargs)

        teacher_x0 = teacher_x0 + (self.config.guidance_scale - 1) * (teacher_x0 - teacher_x0_neg)
        return teacher_x0

    def _student_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        data: Dict[str, Any],
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Perform student model update step.

        Args:
            input_student: Input tensor to student network
            t_student: Input time to student network
            t: Time step
            eps: Noise tensor
            data: Original data batch
            condition: Conditioning information
            neg_condition: Negative conditioning

        Returns:
            tuple of (loss_map, outputs)
        """
        # Generate data from student
        gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)
        perturbed_data = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        # Compute the fake score with x0-prediction
        with torch.no_grad():
            fake_score_x0 = self.fake_score(perturbed_data, t, condition=condition, fwd_pred_type="x0")

        # Compute the teacher x0-prediction and gan loss for generator
        assert (
            perturbed_data.dtype == data["real"].dtype == input_student.dtype
        ), f"perturbed_data.dtype: {perturbed_data.dtype}, data['real'].dtype: {data['real'].dtype}, input_student.dtype: {input_student.dtype}"
        assert (
            t.dtype == t_student.dtype == self.net.noise_scheduler.t_precision
        ), f"t.dtype: {t.dtype}, t_student.dtype: {t_student.dtype}, self.net.noise_scheduler.t_precision: {self.net.noise_scheduler.t_precision}"
        teacher_x0 = self._compute_teacher_x0(perturbed_data, t, condition=condition)
        gan_loss_gen = self._compute_generator_gan_loss(perturbed_data, t, condition=condition)

        # Apply classifier-free guidance if needed
        if self.config.guidance_scale is not None:
            teacher_x0 = self._apply_classifier_free_guidance(
                perturbed_data, t, teacher_x0, neg_condition=neg_condition
            )

        # Compute the VSD loss
        vsd_loss = variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0)

        # Compute the final loss
        loss = vsd_loss + self.config.gan_loss_weight_gen * gan_loss_gen

        # Build output dictionaries
        loss_map = {
            "total_loss": loss,
            "vsd_loss": vsd_loss,
            "gan_loss_gen": gan_loss_gen,
        }
        outputs = self._get_outputs(gen_data, input_student, condition=condition)

        return loss_map, outputs

    def _compute_real_feat(
        self, real_data: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, condition: Optional[Any] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute real features for the discriminator using the fake_score backbone.

        Args:
            real_data: Real data tensor
            t: Time step
            eps: Noise tensor
            condition: Conditioning information

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
        perturbed_real = self.net.noise_scheduler.forward_process(real_data, eps_real, t_real)
        real_feat = self.fake_score(
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
            real_feat_alpha = self.fake_score(
                perturbed_real_alpha,
                t_real,
                condition=condition,
                return_features_early=True,
                feature_indices=self.discriminator.feature_indices,
            )
        real_feat_alpha_logit = self.discriminator(real_feat_alpha)
        gan_loss_ar1 = F.mse_loss(real_feat_logit, real_feat_alpha_logit, reduction="mean")

        return gan_loss_ar1

    def _fake_score_discriminator_update_step(
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
            condition: Conditioning information
            real_data: Real data tensor

        Returns:
            tuple of (loss_map, outputs)
        """
        # Generate data and compute fake score loss
        with torch.no_grad():
            gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)
            x_t_sg = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        # The fake score matches the teacher, but we want to do SDS in x0 space
        fake_score_pred_type = self.config.fake_score_pred_type or self.teacher.net_pred_type
        assert (
            x_t_sg.dtype == real_data.dtype == input_student.dtype
        ), f"x_t_sg.dtype: {x_t_sg.dtype}, real_data.dtype: {real_data.dtype}, input_student.dtype: {input_student.dtype}"
        assert (
            t.dtype == t_student.dtype == self.net.noise_scheduler.t_precision
        ), f"t.dtype: {t.dtype}, t_student.dtype: {t_student.dtype}, self.net.noise_scheduler.t_precision: {self.net.noise_scheduler.t_precision}"
        fake_score_pred = self.fake_score(x_t_sg, t, condition=condition, fwd_pred_type=fake_score_pred_type)
        loss_fakescore = denoising_score_matching_loss(
            fake_score_pred_type,
            net_pred=fake_score_pred,
            noise_scheduler=self.net.noise_scheduler,
            x0=gen_data,
            eps=eps,
            t=t,
        )

        gan_loss_disc = torch.zeros_like(loss_fakescore)
        gan_loss_ar1 = torch.zeros_like(loss_fakescore)
        if self.config.gan_loss_weight_gen > 0:
            # Compute the GAN loss for the discriminator
            fake_feat = self.fake_score(
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
            if self.config.gan_r1_reg_weight > 0:
                gan_loss_ar1 = self._compute_r1_regularization(real_feat_logit, real_data, t_real, condition=condition)

        loss = loss_fakescore + gan_loss_disc + self.config.gan_r1_reg_weight * gan_loss_ar1

        loss_map = {
            "total_loss": loss,
            "fake_score_loss": loss_fakescore,
            "gan_loss_disc": gan_loss_disc,
        }
        if self.config.gan_loss_weight_gen > 0 and self.config.gan_r1_reg_weight > 0:
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
            assert input_student is not None, "input_student must be provided"
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
        Single training step for distribution matching distillation (DMD)

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """
        # Prepare training data and conditions
        real_data, condition, neg_condition = self._prepare_training_data(data)

        # Set up gradient requirements based on training phase
        self._setup_grad_requirements(iteration)

        # Generate noise and time steps
        input_student, t_student, t, eps = self._generate_noise_and_time(real_data)

        # Choose between student update or fake_score/discriminator update
        if iteration % self.config.student_update_freq == 0:
            return self._student_update_step(
                input_student, t_student, t, eps, data, condition=condition, neg_condition=neg_condition
            )
        else:
            return self._fake_score_discriminator_update_step(
                input_student, t_student, t, eps, real_data, condition=condition
            )

    def init_optimizers(self):
        """Initialize optimizers, lr_schedulers and grad_scalers"""
        super().init_optimizers()

        # instantiate the optimizers for fake_score and discriminator
        self.fake_score_optimizer = instantiate(self.config.fake_score_optimizer, model=self.fake_score)
        # instantiate the lr schedulers for fake_score and discriminator
        self.fake_score_lr_scheduler = get_scheduler(self.fake_score_optimizer, self.config.fake_score_scheduler)

        if self.config.gan_loss_weight_gen > 0:
            # instantiate the discriminator in DMD2
            self.discriminator_optimizer = instantiate(self.config.discriminator_optimizer, model=self.discriminator)
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
            if self.config.gan_loss_weight_gen > 0:
                return [self.fake_score_optimizer, self.discriminator_optimizer]
            else:
                return [self.fake_score_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler]:
        """
        Get the lr schedulers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if iteration % self.config.student_update_freq == 0:
            return [self.net_lr_scheduler]
        else:
            if self.config.gan_loss_weight_gen > 0:
                return [self.fake_score_lr_scheduler, self.discriminator_lr_scheduler]
            else:
                return [self.fake_score_lr_scheduler]

    @property
    def model_dict(self):
        """Return the model dict containing the student, fake_score, and discriminator models"""
        _model_dict = super().model_dict
        _model_dict["fake_score"] = self.fake_score
        if self.config.gan_loss_weight_gen > 0:
            _model_dict["discriminator"] = self.discriminator

        return _model_dict

    @property
    def optimizer_dict(self):
        """Return a dict containing all the optimizers"""
        _optimizer_dict = super().optimizer_dict
        _optimizer_dict["fake_score"] = self.fake_score_optimizer
        if self.config.gan_loss_weight_gen > 0:
            _optimizer_dict["discriminator"] = self.discriminator_optimizer

        return _optimizer_dict

    @property
    def scheduler_dict(self):
        """Return a dict containing all the lr schedulers"""
        _scheduler_dict = super().scheduler_dict
        _scheduler_dict["fake_score"] = self.fake_score_lr_scheduler
        if self.config.gan_loss_weight_gen > 0:
            _scheduler_dict["discriminator"] = self.discriminator_lr_scheduler

        return _scheduler_dict
