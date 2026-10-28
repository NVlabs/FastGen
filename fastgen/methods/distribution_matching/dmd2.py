# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distribution Matching Distillation 2 (DMD2).

Handles both unimodal (plain tensor) and multimodal (dict-of-tensors) inputs.

With ``student_generation="step"`` (Phased DMD, Fan et al., 2025), the student takes one sampler step per
student-grid interval. For a noise schedule x_t = alpha_t x_0 + sigma_t eps, its ``student_sample_type`` step
(``BaseNoiseSchedule.step``) from grid time t_i to t_gen_data = t_{i+1} gives the generation
gen_data = alpha(t_gen_data) x0_hat + sigma(t_gen_data) eps_gen, with x0_hat and, for the DDIM "ode", eps_gen implied
by its prediction. Sampling and carried trajectories chain these steps. As gen_data is noisy before the last step,
training-sample outputs and rewards use x0_hat.

The VSD and GAN losses match the distribution of gen_data to the data's at t_gen_data. They carry gen_data to t, drawn
from ``sample_t_cfg``'s distribution conditioned on the bucket [t_{i+1}, t_i], or without ``sample_t_in_student_bucket``
on [t_{i+1}, max_t], with the marginal-preserving forward transition (``BaseNoiseSchedule.forward_process``) from
t_start = t_gen_data

    X_t = a gen_data + sqrt(v) eps,    a = alpha_t / alpha(t_gen_data),    v = sigma_t^2 - a^2 sigma(t_gen_data)^2,

which for "sde" (fresh eps_gen) is bucketed DMD2's forward process of x0_hat. The VSD takes gen_data on its x0
scale, gen_data / alpha(t_gen_data), and normalizes by the teacher residual of x0_hat: X_t carries gen_data with a
where DMD2's forward process carries x0 with alpha_t, so this is DMD2's x0 gradient carried to gen_data by the chain
rule. For "sde", the gradient reaching x0_hat is bucketed DMD2's.

The fake score predicts the marginal noise eps_theta(X_t, t) and trains on the unit-noise loss, the x0 loss of its
denoiser D_theta against the conditionally unbiased target Y* = (X_t - sigma_t^2 / sqrt(v) eps) / alpha_t,

    W || eps - sqrt(v) / sigma_t eps_theta ||^2 = w || D_theta - Y* ||^2,    W = w sigma_t^4 / (alpha_t^2 v),

with w set by ``fake_score_weighting`` (see ``common_loss.transition_score_weight``). For clean generations
(v = sigma_t^2, e.g. the last step of a rectified-flow grid), the losses reduce to bucketed DMD2's, with x0 targets
for the fake score (eps for "unweighted"). Without the bucket, several student steps reach each t, and the fake
score, which sees only (X_t, t), fits the w-weighted average of their denoisers. That is their mixture's denoiser
only if w is the same for all of them, so this mode requires "x0" (w = 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import attrs
import torch

from fastgen.configs.opt import get_scheduler
from fastgen.methods.backward_simulation import (
    attach_trajectory_carry,
    resolve_trajectory_point,
)
from fastgen.methods.common_loss import (
    flow_matching_loss,
    gan_loss_approx_r1,
    gan_loss_discriminator,
    gan_loss_generator,
    transition_score_loss,
    variational_score_distillation_loss,
)
from fastgen.methods.model import FastGenModel
from fastgen.configs.instantiate import instantiate
from fastgen.utils.basic_utils import safe_update
from fastgen.utils.distributed.collectives import synchronize
from fastgen.utils.guidance import predict_with_guidance
import fastgen.utils.logging_utils as logger
from fastgen.utils.tensor import expand_like, to_high_precision
from fastgen.utils.tree_utils import (
    TensorOrDict,
    assert_same,
    cast_like,
    detach,
    get_batch_size,
    get_dtype,
    noise_from_shape,
    per_modality,
    to_dict,
)

if TYPE_CHECKING:
    from fastgen.configs.config import SampleTConfig
    from fastgen.networks.network import FastGenNetwork


@attrs.frozen(eq=False)
class TrainingInputs:
    """Resolved student state and noising/trajectory/loss metadata, passed unchanged to update hooks."""

    input_student: TensorOrDict
    student_step: torch.Tensor | None
    t: torch.Tensor
    eps: TensorOrDict
    trajectory_step: int | None = None
    generation_kwargs: dict[str, Any] = attrs.field(factory=dict)
    loss_scale: torch.Tensor | None = None


@attrs.frozen(eq=False)
class Generation:
    """A student generation ``data`` and what the losses, outputs, and carries need from it.

    ``x0`` is its clean estimate, used by training-sample outputs, rewards, and the VSD normalizer; clean generations
    are their own. An estimate converted from a prediction stays in high precision for the VSD normalizer, and
    outputs and rewards take it in ``data``'s dtype. ``next_state`` is the state at the next grid time that a carried
    trajectory continues from, if any. ``metrics`` may include differentiable losses.
    """

    data: TensorOrDict
    x0: TensorOrDict
    next_state: TensorOrDict | None = None
    metrics: dict[str, torch.Tensor] = attrs.field(factory=dict)


class DMD2Model(FastGenModel):
    """Weighted VSD and adversarial distribution-matching trainer.

    Standard DMD2 enables VSD and may optionally enable the adversarial
    objective. LADD is the GAN-only specialization with VSD disabled.
    ``config.student_generation`` selects whether the student generates its
    clean estimate or its sampler step (Phased DMD, see the module docstring).
    """

    # The config.student_generation values this method's generator supports; subclasses whose gen_data_from_net
    # supports fewer narrow it.
    student_generations: Tuple[str, ...] = ("x0", "step")

    def __init__(self, config):
        if config.student_generation not in self.student_generations:
            raise ValueError(
                f"{type(self).__name__} supports student_generation in {self.student_generations}, "
                f"got {config.student_generation!r}."
            )
        super().__init__(config)

    @property
    def use_vsd(self) -> bool:
        """Whether this configuration enables VSD and fake-score training."""
        return self.config.vsd_loss_weight > 0

    @property
    def use_gan(self) -> bool:
        """Whether this configuration enables adversarial training."""
        return self.config.gan_loss_weight_gen > 0

    def build_model(self):
        super().build_model()

        self.build_teacher()
        self.load_student_weights_and_ema()

        if not self.config.inference:
            if self.use_vsd:
                self._check_fake_score_config()
                logger.info("Instantiating the fake_score")
                with self._get_meta_init_context():
                    self.fake_score = instantiate(self.teacher_config)
                model_path = self.config.pretrained_model_path
                if model_path is not None:
                    if self.separate_teacher is None:
                        # An adapter teacher is a parameter-less view over the
                        # student, so its state_dict is deliberately empty. Load
                        # an independently-instantiated fake score from the same
                        # explicit base checkpoint instead.
                        FastGenModel._load_pretrained_model(
                            self.fake_score,
                            model_path,
                            meta_init=self.config.meta_init,
                        )
                    else:
                        self._load_state_from(self.fake_score, self.teacher, "Fake score initialized from teacher")

            if self.use_gan:
                logger.info(f"gan_loss_weight_gen: {self.config.gan_loss_weight_gen}")
                logger.info("Instantiating the discriminator")
                if getattr(self.config.discriminator, "disc_type", None) is not None:
                    logger.info(f"Discriminator type: {self.config.discriminator.disc_type}")
                # TODO: Discriminators do not yet support meta initialization
                self.discriminator = instantiate(self.config.discriminator)
                synchronize()

        torch.cuda.empty_cache()

    def _check_fake_score_config(self) -> None:
        """Check that the fake-score loss is valid for the generations it trains on, and warn where it reduces to
        ``fake_score_weighting="native"``."""
        weighting = self.config.fake_score_weighting
        if self.config.student_generation == "step":
            if weighting == "native":
                raise ValueError(
                    "fake_score_weighting='native' regresses the fake score on generations as clean data, but step "
                    "generations are noisy; choose a transition weighting (see common_loss.transition_score_weight)."
                )
            if not self.config.sample_t_in_student_bucket and weighting != "x0":
                raise ValueError(
                    "Without sample_t_in_student_bucket, the fake score learns from several student steps at each t, "
                    "and only fake_score_weighting='x0' keeps its loss unbiased over them; "
                    f"got {weighting!r}."
                )
        if weighting != "native":
            pred_type = self.config.fake_score_pred_type or self.teacher.net_pred_type
            if pred_type != "eps":
                raise ValueError(
                    f"fake_score_weighting={weighting!r} trains the fake score with a unit-noise loss on eps "
                    f"predictions; set fake_score_pred_type='eps', got {pred_type!r}."
                )
            if self.config.student_generation == "x0":
                native_pred_type = "eps" if weighting == "unweighted" else "x0"
                message = (
                    f"fake_score_weighting={weighting!r} only differs from 'native' on step generations; on x0 "
                    f"generations it is the 'native' loss with fake_score_pred_type={native_pred_type!r}."
                )
                if native_pred_type == "x0":
                    message += (
                        " It computes that loss from eps predictions weighted by sigma_t^2 / alpha_t^2, which "
                        "amplifies rounding where alpha_t is small."
                    )
                logger.warning(message)

    def _setup_grad_requirements(self, iteration: int) -> None:
        update_auxiliary = iteration % self.config.student_update_freq != 0

        self.net.train(not update_auxiliary).requires_grad_(not update_auxiliary)
        if self.use_gan:
            self.discriminator.train(update_auxiliary).requires_grad_(update_auxiliary)
        if self.use_vsd:
            self.fake_score.train(update_auxiliary).requires_grad_(update_auxiliary)

    def _sample_t_and_eps(
        self,
        batch_size: int,
        sample_t_cfg: Optional[SampleTConfig] = None,
        student_step: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, TensorOrDict]:
        """Sample a timestep and matching noise from one explicit distribution.

        Given each sample's generator-input student step, the draw is conditioned on its ``_sample_t_window``.
        """
        if sample_t_cfg is None:
            sample_t_cfg = self.config.sample_t_cfg
        window = self._sample_t_window(student_step, batch_size, min_t=sample_t_cfg.min_t)
        t = self.net.noise_scheduler.sample_t(
            batch_size,
            **sample_t_cfg.sample_t_kwargs(),
            window=window,
            device=self.device,
        )
        eps = noise_from_shape(self.input_shape, batch_size, self.device, self.precision)
        return t, eps

    def _student_grid(self, device: torch.device) -> torch.Tensor:
        """Return the student grid that ``student_step`` and the carried ``trajectory_step`` index."""
        return self.net.noise_scheduler.student_grid(device=device)

    def _generation_time(self, student_step: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Return each generation's time: None for clean x0 generations, else the next grid time ``t_{i+1}``."""
        if self.config.student_generation == "x0":
            return None
        return self._student_grid(student_step.device)[student_step + 1]

    def _sample_t_window(
        self,
        student_step: Optional[torch.Tensor],
        batch_size: int,
        min_t: Optional[float] = None,
    ) -> Optional[tuple[torch.Tensor, float | torch.Tensor]]:
        """Return each sample's noising-time window, which conditions the ``[min_t, max_t]`` distribution, or None.

        For step generations, ``min_t`` is the sampled time's distance above the generation's grid time
        ``t_{i+1}``, as it is above the clean data for x0 generations: a transition of (near) zero length carries
        no information, and the ``"x0"`` fake-score weight grows like its inverse variance. With
        ``sample_t_in_student_bucket`` the window is ``[t_{i+1} + min_t, t_i]``, which ``[min_t, max_t]`` must
        overlap; without it, step generations are noised from ``t_{i+1} + min_t`` up to ``max_t``. ``sample_t``
        intersects the window with ``sample_t_cfg``'s ``[min_t, max_t]`` in float64, so its upper end stays ``t_i``,
        or the schedule's ``max_t`` without the bucket: clamping the grid times here would round ``max_t`` to their
        ``t_precision``. The window conditions the distribution rather than redefining it on the window, so every
        window keeps that distribution's shape (for the shift, Phased DMD's uniform draw before the global shift).
        """
        in_bucket = self.config.sample_t_in_student_bucket
        if not in_bucket and self.config.student_generation == "x0":
            return None
        if student_step is None or student_step.shape != (batch_size,):
            shape = None if student_step is None else tuple(student_step.shape)
            raise ValueError(
                "Bounding noising times by the student grid needs one student step per sample, but "
                f"{type(self).__name__} provided student_step of shape {shape} for batch size {batch_size}."
            )
        grid = self._student_grid(student_step.device)
        lower = grid[student_step + 1] + (0.0 if min_t is None else min_t)
        upper = grid[student_step] if in_bucket else self.net.noise_scheduler.max_t
        return lower, upper

    def _training_inputs(
        self,
        data: Dict[str, Any],
        iteration: int = 1,
    ) -> TrainingInputs:
        """Build shared data-driven or backward-simulation inputs."""
        condition = data.get("condition")
        real_data = data.get("real")
        batch_size = self._batch_size(data)
        noise_scheduler = self.net.noise_scheduler
        student_grid = self._student_grid(self.device)
        num_steps = len(student_grid) - 1
        trajectory_step = None

        if real_data is not None and get_batch_size(real_data) != batch_size:
            raise ValueError(
                f"Training input batch {batch_size} does not match its paired real batch "
                f"{get_batch_size(real_data)}."
            )

        if self.config.backward_simulation is None:
            eps_student = noise_from_shape(self.input_shape, batch_size, self.device, self.precision)
            if num_steps == 1:
                t_student = student_grid[0].expand(batch_size)
                student_step = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                input_student = noise_scheduler.latents(noise=eps_student, t_init=t_student)
            else:
                if real_data is None:
                    raise ValueError("Multi-step data-driven training requires data['real'].")
                student_step = torch.randint(num_steps, (batch_size,), device=self.device)
                input_student = noise_scheduler.forward_process(real_data, eps_student, student_grid[student_step])
        else:
            input_student, trajectory_step = resolve_trajectory_point(
                data,
                net=self.net,
                student_grid=student_grid,
                input_shape=self.input_shape,
                batch_size=batch_size,
                dtype=self.precision,
                condition=condition,
                advance_step=self._advance_student_step,
                iteration=iteration,
                update_period=self.config.student_update_freq,
                owner=f"backward-simulation {type(self).__name__.removesuffix('Model')}",
                advance_empty=torch.is_grad_enabled(),
                advance_type=self.config.backward_simulation,
            )
            student_step = torch.full((batch_size,), trajectory_step, dtype=torch.long, device=self.device)

        t, eps = self._sample_t_and_eps(batch_size, student_step=student_step)
        return TrainingInputs(
            input_student=input_student,
            student_step=student_step,
            t=t,
            eps=eps,
            trajectory_step=trajectory_step,
        )

    def gen_data_from_net(
        self,
        input_student: TensorOrDict,
        student_step: Optional[torch.Tensor],
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> Generation:
        """Generate from one student prediction at grid time t_i: the x0 it implies, or for step generations its
        sampler step to t_{i+1}, from which carried trajectories continue in either mode."""
        del neg_condition
        if student_step is None:
            raise ValueError("Base DMD2 generation requires a student step.")
        student_grid = self._student_grid(student_step.device)
        t_student = student_grid[student_step]
        metrics = {"avg_t_student": t_student.mean()}

        prediction = self.net(input_student, t_student, condition=condition, output_dtype=None)
        x0 = self.net.noise_scheduler.convert_model_output(
            input_student,
            prediction,
            t_student,
            src_pred_type=self.net.net_pred_type,
            target_pred_type="x0",
            output_dtype=None,
        )

        is_step = self.config.student_generation == "step"
        next_state = None
        if is_step or self.config.backward_simulation is not None:
            with torch.set_grad_enabled(is_step and torch.is_grad_enabled()):
                next_state = self.net.noise_scheduler.step(
                    input_student,
                    prediction,
                    t_student,
                    student_grid[student_step + 1],
                    pred_type=self.net.net_pred_type,
                    sample_type=self.config.student_sample_type,
                )

        data = next_state if is_step else cast_like(x0, input_student)
        return Generation(data=data, x0=x0, next_state=next_state, metrics=metrics)

    def _advance_student_step(
        self,
        net: FastGenNetwork,
        state: TensorOrDict,
        step_index: int,
        student_grid: torch.Tensor,
        condition: Any = None,
    ) -> TensorOrDict:
        """Advance across ``student_grid[step_index:step_index + 2]`` by the sampler step of the net's own prediction,
        as in training."""
        num_steps = len(student_grid) - 1
        if not isinstance(step_index, int) or not 0 <= step_index < num_steps:
            raise ValueError(f"step_index must be an int in [0, {num_steps - 1}], got {step_index!r}.")
        t = student_grid[step_index].expand(get_batch_size(state))
        return net.noise_scheduler.step(
            state,
            net(state, t, condition=condition, output_dtype=None),
            t,
            student_grid[step_index + 1],
            pred_type=net.net_pred_type,
            sample_type=self.config.student_sample_type,
        )

    def _fake_score_loss_map(
        self,
        fake_score_pred: TensorOrDict,
        pred_type: str,
        gen_data: TensorOrDict,
        eps: TensorOrDict,
        t: torch.Tensor,
        t_gen_data: Optional[torch.Tensor],
        loss_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Fake-score loss on generations at ``t_gen_data`` (None if clean) noised to ``t``.

        ``fake_score_weighting="native"`` is denoising score matching in the prediction space ``pred_type``; the
        other weightings weight the unit-noise loss of the forward transition (see ``transition_score_loss``).
        """
        if self.config.fake_score_weighting == "native":
            losses = per_modality(
                flow_matching_loss,
                fake_score_pred,
                pred_type,
                gen_data,
                eps,
                self.net.noise_scheduler.schedules,
                t,
                loss_scale=loss_scale,
            )
        else:
            losses = per_modality(
                transition_score_loss,
                fake_score_pred,
                eps,
                self.net.noise_scheduler.schedules,
                torch.zeros_like(t) if t_gen_data is None else t_gen_data,
                t,
                self.config.fake_score_weighting,
                self.config.fake_score_data_variance,
                loss_scale=loss_scale,
            )
        return to_dict(losses, suffix="fake_score_loss")

    def _training_outputs(
        self,
        generation: Generation,
        data: Dict[str, Any],
        trajectory_step: Optional[int] = None,
    ) -> Dict[str, TensorOrDict | Callable]:
        """Build an update step's outputs and attach the carry of a trajectory at ``trajectory_step``."""
        outputs = self._get_outputs(
            cast_like(generation.x0, generation.data),
            condition=data.get("condition"),
            neg_condition=data.get("neg_condition"),
        )
        if trajectory_step is not None:
            attach_trajectory_carry(
                outputs,
                data,
                current_step=trajectory_step,
                num_steps=len(self._student_grid(self.device)) - 1,
                next_state=generation.next_state,
                advance_type=self.config.backward_simulation,
            )
        return outputs

    def _vsd_input(self, gen_data: TensorOrDict, t_gen_data: Optional[torch.Tensor]) -> TensorOrDict:
        """Return each generation on its x0 scale ``gen_data / alpha(t_gen_data)``, whose VSD gradient on ``gen_data``
        is DMD2's x0 gradient divided by ``alpha(t_gen_data)``, the scale its forward transition starts from; clean
        generations are their own x0."""
        if t_gen_data is None:
            return gen_data
        alpha_gen_data = self.net.noise_scheduler.alpha_start(t_gen_data)

        def to_x0_scale(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
            original_dtype = x.dtype
            x = to_high_precision(x)
            return (x / expand_like(alpha.to(x.dtype), x)).to(original_dtype)

        return per_modality(to_x0_scale, gen_data, alpha_gen_data)

    def _compute_distribution_matching_loss(
        self,
        gen_data: TensorOrDict,
        teacher_x0: TensorOrDict,
        fake_score_x0: TensorOrDict,
        fake_logits: Optional[torch.Tensor],
        t: torch.Tensor,
        loss_scale: Optional[torch.Tensor] = None,
        gen_x0: Optional[TensorOrDict] = None,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Compute the VSD objective on ``gen_data``, the generation on its x0 scale (see ``_vsd_input``), normalized
        by the teacher residual of the clean estimate ``gen_x0`` (by default the generation itself), and its
        logging entries."""
        del fake_logits, t
        vsd_per_modality = per_modality(
            variational_score_distillation_loss,
            gen_data,
            teacher_x0,
            fake_score_x0,
            loss_scale=loss_scale,
            gen_x0=gen_x0,
        )
        loss_map = to_dict(vsd_per_modality, suffix="vsd_loss")
        return loss_map, sum(loss_map.values())

    def _student_update_step(
        self,
        inputs: TrainingInputs,
        data: Dict[str, Any],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, TensorOrDict | Callable]]:
        """Update the student with the enabled VSD and generator GAN terms."""
        # The auxiliary update trains the discriminator only on real data, so only then does the generator face it.
        # `real` must be present or absent on every rank: the discriminator path and f-distill's histogram
        # reduce collectively.
        use_gan = self.use_gan and data.get("real") is not None
        if not self.use_vsd and not use_gan:
            raise ValueError("DMD2 student updates require VSD training, or GAN training with data['real'].")

        condition = data.get("condition")
        neg_condition = data.get("neg_condition")
        generation = self.gen_data_from_net(
            inputs.input_student,
            inputs.student_step,
            condition=condition,
            neg_condition=neg_condition,
            **inputs.generation_kwargs,
        )
        t_gen_data = self._generation_time(inputs.student_step)
        perturbed_data = self.net.noise_scheduler.forward_process(
            generation.data, inputs.eps, inputs.t, t_start=t_gen_data
        )

        # The teacher's native prediction is the conditional branch of its guided target below.
        fake_logits = None
        if use_gan:
            teacher_outputs = self.teacher(
                perturbed_data,
                inputs.t,
                condition=condition,
                return_features_early=not self.use_vsd,
                feature_indices=self.discriminator.feature_indices,
                apply_output_conditioning=False,
                output_dtype=None,
            )
            if self.use_vsd:
                teacher_output, fake_feat = teacher_outputs
                teacher_output = detach(teacher_output)
            else:
                fake_feat = teacher_outputs
            assert isinstance(fake_feat, list)
            fake_logits = self.discriminator(fake_feat)
            loss_map = {
                "gan_loss_gen": gan_loss_generator(
                    fake_logits,
                    loss_scale=inputs.loss_scale,
                ),
            }
        else:
            with torch.no_grad():
                teacher_output = self.teacher(
                    perturbed_data,
                    inputs.t,
                    condition=condition,
                    apply_output_conditioning=False,
                    output_dtype=None,
                )
            loss_map = {
                "gan_loss_gen": torch.tensor(0.0, device=self.device, dtype=get_dtype(perturbed_data)),
            }

        total_loss = self.config.gan_loss_weight_gen * loss_map["gan_loss_gen"]
        if self.use_vsd:
            # The VSD loss upcasts these clean estimates, so they are not rounded to the state's dtype first.
            with torch.no_grad():
                fake_score_x0 = self.fake_score(
                    perturbed_data, inputs.t, condition=condition, fwd_pred_type="x0", output_dtype=None
                )
                teacher_x0 = predict_with_guidance(
                    self.teacher,
                    perturbed_data,
                    inputs.t,
                    guidance_cfg=self.config.guidance_cfg,
                    condition=condition,
                    neg_condition=neg_condition,
                    precomputed={"pos": teacher_output},
                    skip_inactive=False,
                    output_dtype=None,
                    fwd_pred_type="x0",
                )

            distribution_loss_map, distribution_loss = self._compute_distribution_matching_loss(
                self._vsd_input(generation.data, t_gen_data),
                teacher_x0,
                fake_score_x0,
                fake_logits,
                inputs.t,
                loss_scale=inputs.loss_scale,
                gen_x0=generation.x0,
            )
            safe_update(loss_map, distribution_loss_map)
            total_loss = self.config.vsd_loss_weight * distribution_loss + total_loss

        loss_map["total_loss"] = total_loss
        safe_update(loss_map, generation.metrics)
        outputs = self._training_outputs(generation, data, inputs.trajectory_step)
        loss_map = self.apply_reward(cast_like(generation.x0, generation.data), data, loss_map)
        return loss_map, outputs

    def _compute_real_feat(
        self,
        real_data: TensorOrDict,
        t: torch.Tensor,
        eps: TensorOrDict,
        condition: Optional[Any] = None,
        student_step: Optional[torch.Tensor] = None,
    ) -> tuple[list, torch.Tensor, TensorOrDict]:
        """Compute discriminator features from perturbed real data."""
        if self.config.gan_use_same_t_noise:
            t_real = t
            eps_real = eps
        else:
            t_real, eps_real = self._sample_t_and_eps(get_batch_size(real_data), student_step=student_step)

        perturbed_real = self.net.noise_scheduler.forward_process(real_data, eps_real, t_real)
        real_feat = self.teacher(
            perturbed_real,
            t_real,
            condition=condition,
            return_features_early=True,
            feature_indices=self.discriminator.feature_indices,
        )
        assert isinstance(real_feat, list), f"Expected list of features, got {type(real_feat)}"
        return real_feat, t_real, perturbed_real

    def _compute_gan_discriminator_losses(
        self,
        fake_data: TensorOrDict,
        t: torch.Tensor,
        eps: TensorOrDict,
        real_data: TensorOrDict,
        condition: Optional[Any] = None,
        loss_scale: Optional[torch.Tensor] = None,
        student_step: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Compute discriminator classification and optional approximate-R1 losses."""
        with torch.no_grad():
            fake_feat = self.teacher(
                fake_data,
                t,
                condition=condition,
                return_features_early=True,
                feature_indices=self.discriminator.feature_indices,
            )
            real_feat, t_real, perturbed_real = self._compute_real_feat(
                real_data=real_data,
                t=t,
                eps=eps,
                condition=condition,
                student_step=student_step,
            )

        assert isinstance(fake_feat, list) and isinstance(real_feat, list)
        real_feat_logit = self.discriminator(real_feat)
        gan_loss_disc = gan_loss_discriminator(
            real_feat_logit,
            self.discriminator(fake_feat),
            loss_scale=loss_scale,
        )

        gan_loss_ar1 = torch.zeros_like(gan_loss_disc)
        loss_map = {"gan_loss_disc": gan_loss_disc}
        if self.config.gan_r1_reg_weight > 0:
            gan_loss_ar1 = gan_loss_approx_r1(
                real_feat_logit,
                perturbed_real,
                t_real,
                teacher=self.teacher,
                discriminator=self.discriminator,
                perturbation_scale=self.config.gan_r1_reg_alpha,
                condition=condition,
                loss_scale=loss_scale,
            )
            loss_map["gan_loss_ar1"] = gan_loss_ar1

        total_loss = gan_loss_disc + self.config.gan_r1_reg_weight * gan_loss_ar1
        return loss_map, total_loss

    def _auxiliary_update_step(
        self,
        inputs: TrainingInputs,
        data: Dict[str, Any],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, TensorOrDict | Callable]]:
        """Update the enabled fake-score and discriminator models."""
        if not self.use_vsd and not self.use_gan:
            raise ValueError("DMD2 requires at least one of VSD or GAN training to be enabled.")

        condition = data.get("condition")
        neg_condition = data.get("neg_condition")
        real_data = data.get("real")
        if self.use_gan and not self.use_vsd and real_data is None:
            raise ValueError("GAN-only discriminator updates require data['real'].")

        t_gen_data = self._generation_time(inputs.student_step)
        with torch.no_grad():
            generation = self.gen_data_from_net(
                inputs.input_student,
                inputs.student_step,
                condition=condition,
                neg_condition=neg_condition,
                **inputs.generation_kwargs,
            )
            x_t = self.net.noise_scheduler.forward_process(generation.data, inputs.eps, inputs.t, t_start=t_gen_data)

        loss_map: Dict[str, torch.Tensor] = {}
        gan_loss_map: Dict[str, torch.Tensor] = {}
        total_loss = None
        if self.use_gan and real_data is not None:
            gan_loss_map, total_loss = self._compute_gan_discriminator_losses(
                x_t,
                inputs.t,
                inputs.eps,
                real_data,
                condition=condition,
                loss_scale=inputs.loss_scale,
                student_step=inputs.student_step,
            )

        if self.use_vsd:
            fake_score_t = inputs.t
            fake_score_eps = inputs.eps
            fake_score_sample_t_overrides = self.config.fake_score_sample_t_overrides
            if fake_score_sample_t_overrides is not None:
                fake_score_t, fake_score_eps = self._sample_t_and_eps(
                    get_batch_size(generation.data),
                    sample_t_cfg=attrs.evolve(self.config.sample_t_cfg, **fake_score_sample_t_overrides),
                    student_step=inputs.student_step,
                )
                with torch.no_grad():
                    x_t = self.net.noise_scheduler.forward_process(
                        generation.data, fake_score_eps, fake_score_t, t_start=t_gen_data
                    )

            fake_score_pred_type = self.config.fake_score_pred_type or self.teacher.net_pred_type
            assert_same(
                dtype=self.precision,
                device=self.device,
                x_t=x_t,
                input_student=inputs.input_student,
                real_data=real_data,
            )
            assert_same(dtype=self.net.noise_scheduler.t_precision, device=self.device, fake_score_t=fake_score_t)

            fake_score_pred = self.fake_score(
                x_t,
                fake_score_t,
                condition=condition,
                fwd_pred_type=fake_score_pred_type,
            )
            fake_score_loss_map = self._fake_score_loss_map(
                fake_score_pred,
                fake_score_pred_type,
                generation.data,
                fake_score_eps,
                fake_score_t,
                t_gen_data,
                loss_scale=inputs.loss_scale,
            )
            safe_update(loss_map, fake_score_loss_map)
            fake_score_loss = sum(fake_score_loss_map.values())
            total_loss = fake_score_loss if total_loss is None else total_loss + fake_score_loss

        safe_update(loss_map, gan_loss_map)
        if total_loss is None:
            raise ValueError("This auxiliary update has no active objective.")
        loss_map["total_loss"] = total_loss
        safe_update(loss_map, generation.metrics)
        outputs = self._training_outputs(generation, data, inputs.trajectory_step)
        return loss_map, outputs

    def single_train_step(
        self,
        data: Dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, TensorOrDict | Callable]]:
        """Run one alternating student or auxiliary training step."""
        self._setup_grad_requirements(iteration)

        inputs = self._training_inputs(data, iteration)

        update_step = (
            self._student_update_step
            if iteration % self.config.student_update_freq == 0
            else self._auxiliary_update_step
        )
        return update_step(inputs, data)

    def init_optimizers(self):
        """Initialize optimizers, learning-rate schedulers, and grad scalers."""
        super().init_optimizers()
        if self.use_vsd:
            self.fake_score_optimizer = instantiate(
                self.config.fake_score_optimizer,
                model=self.fake_score,
            )
            self.fake_score_lr_scheduler = get_scheduler(
                self.fake_score_optimizer,
                self.config.fake_score_scheduler,
            )

        if self.use_gan:
            self.discriminator_optimizer = instantiate(
                self.config.discriminator_optimizer,
                model=self.discriminator,
            )
            self.discriminator_lr_scheduler = get_scheduler(
                self.discriminator_optimizer,
                self.config.discriminator_scheduler,
            )

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """Return the optimizers active on this alternating iteration."""
        if iteration % self.config.student_update_freq == 0:
            return [self.net_optimizer]

        optimizers = []
        if self.use_vsd:
            optimizers.append(self.fake_score_optimizer)
        if self.use_gan:
            optimizers.append(self.discriminator_optimizer)
        return optimizers

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """Return the learning-rate schedulers active on this iteration."""
        if iteration % self.config.student_update_freq == 0:
            return [self.net_lr_scheduler]

        schedulers = []
        if self.use_vsd:
            schedulers.append(self.fake_score_lr_scheduler)
        if self.use_gan:
            schedulers.append(self.discriminator_lr_scheduler)
        return schedulers

    @property
    def model_dict(self):
        """Return the student and all constructed auxiliary models."""
        model_dict = super().model_dict
        if hasattr(self, "fake_score"):
            model_dict["fake_score"] = self.fake_score
        if hasattr(self, "discriminator"):
            model_dict["discriminator"] = self.discriminator
        return model_dict

    @property
    def optimizer_dict(self):
        """Return all initialized optimizers."""
        optimizer_dict = super().optimizer_dict
        if hasattr(self, "fake_score_optimizer"):
            optimizer_dict["fake_score"] = self.fake_score_optimizer
        if hasattr(self, "discriminator_optimizer"):
            optimizer_dict["discriminator"] = self.discriminator_optimizer
        return optimizer_dict

    @property
    def scheduler_dict(self):
        """Return all initialized learning-rate schedulers."""
        scheduler_dict = super().scheduler_dict
        if hasattr(self, "fake_score_lr_scheduler"):
            scheduler_dict["fake_score"] = self.fake_score_lr_scheduler
        if hasattr(self, "discriminator_lr_scheduler"):
            scheduler_dict["discriminator"] = self.discriminator_lr_scheduler
        return scheduler_dict
