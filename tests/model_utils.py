# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model-dependent test utilities."""

import contextlib

import torch

from fastgen.networks.lora_utils import SharedBaseTeacher
from fastgen.networks.network import FastGenModule, FastGenNetwork
from fastgen.networks.noise_schedule import CompositeNoiseSchedule, StudentGridConfig
from fastgen.networks.utils.pdd_mixin import PDDMixin, PDDSpec, make_pdd


class FunctionNet(FastGenNetwork):
    """A flow-predicting network whose native prediction is ``native(x_t, t, condition=..., **kwargs)``.

    *native* defaults to a subclass's ``native`` method; *modalities* selects a composite of RF schedules for dict
    inputs.
    """

    def __init__(self, native=None, modalities=None, net_pred_type="flow"):
        super().__init__(net_pred_type=net_pred_type, schedule_type="rf")
        if native is not None:
            # A stub network passed as *native* must not become a submodule.
            object.__setattr__(self, "native", native)
        if modalities is not None:
            self.noise_scheduler = CompositeNoiseSchedule(schedules={m: {"name": "rf"} for m in modalities})

    def forward(
        self, x_t, t, condition=None, fwd_pred_type=None, apply_output_conditioning=True, output_dtype="same", **kwargs
    ):
        prediction = self.native(x_t, t, condition=condition, **kwargs)
        return self.transform_output(
            prediction, x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
        )

    def sample(self, *args, **kwargs):
        raise NotImplementedError


class MultimodalNet(FastGenModule):
    """Test-only wrapper that makes an EDM net handle multimodal dict inputs.

    Processes each modality independently through the wrapped network.
    Video features are expanded to 5D for Discriminator_VideoDiT compatibility.
    It delegates the rest of the network interface to the wrapped network, so it is registered as a virtual
    ``FastGenNetwork``.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        sched = inner.noise_scheduler
        self.noise_scheduler = CompositeNoiseSchedule(
            schedules={
                "video": {
                    "name": "edm",
                    "min_t": sched._min_t,
                    "max_t": sched._max_t,
                    "student_grid_cfg": sched.student_grid_cfg,
                },
                "audio": {"name": "edm", "min_t": sched._min_t, "max_t": sched._max_t},
            }
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    @staticmethod
    def _to_5d(feat):
        """Convert [B, C, H, W] to [B, C, 2, H, W] for Conv3d kernels."""
        return feat.unsqueeze(2).repeat(1, 1, 2, 1, 1)

    def forward(self, x_t, t, **kwargs):
        if isinstance(x_t, torch.Tensor):
            return self.inner(x_t, t, **kwargs)

        feature_indices = kwargs.get("feature_indices")
        return_features_early = kwargs.get("return_features_early", False)

        v_out = self.inner(x_t["video"], t, **kwargs)
        a_out = self.inner(x_t["audio"], t, **kwargs)

        if return_features_early:
            return [{"video": self._to_5d(vf), "audio": af} for vf, af in zip(v_out, a_out)]

        if feature_indices is not None:
            v_pred, v_feats = v_out[0], v_out[1]
            a_pred, a_feats = a_out[0], a_out[1]
            av_feats = [{"video": self._to_5d(vf), "audio": af} for vf, af in zip(v_feats, a_feats)]
            return [{"video": v_pred, "audio": a_pred}, av_feats]

        return {"video": v_out, "audio": a_out}

    def transform_output(self, prediction, x_t, t, **kwargs):
        if isinstance(x_t, torch.Tensor):
            return self.inner.transform_output(prediction, x_t, t, **kwargs)
        return {key: self.inner.transform_output(prediction[key], x_t[key], t, **kwargs) for key in x_t}


FastGenNetwork.register(MultimodalNet)


class StudentAsTeacher(SharedBaseTeacher):
    """Teacher view over a student's own weights: a ``SharedBaseTeacher`` for a net without adapters.

    The net is held by reference and not registered, so the view owns no parameters; attribute lookups
    fall through to the student, which runs as it is. Use it to exercise no_grad teacher passes that share
    trainable weights with the gradient forward.
    """

    @contextlib.contextmanager
    def _as_base_model(self):
        yield


def _native_grid(base_steps, steps=None, shift=None, blocks=None):
    grid_kwargs = {"base_steps": base_steps}
    if shift is not None:
        grid_kwargs["shift"] = shift
    if blocks is not None:
        indices = [0]
        for block in blocks:
            indices.append(indices[-1] + block)
        grid_kwargs["indices"] = {len(blocks): indices}
        steps = len(blocks)
    return StudentGridConfig(
        steps=base_steps if steps is None else steps,
        grid_type="native",
        grid_kwargs=grid_kwargs,
    )


def _explicit_grid(timesteps, steps=None, blocks=None):
    base_steps = len(timesteps) - 1
    grid_kwargs = {"timesteps": timesteps}
    if blocks is not None:
        indices = [0]
        for block in blocks:
            indices.append(indices[-1] + block)
        grid_kwargs["indices"] = {len(blocks): indices}
        steps = len(blocks)
    return StudentGridConfig(
        steps=base_steps if steps is None else steps,
        grid_type="explicit",
        grid_kwargs=grid_kwargs,
    )


class _FakePDDNet:
    def __init__(
        self,
        schedule,
        net_pred_type="flow",
        pdd_steps=None,
        base_grid=None,
        blocks=None,
    ):
        self.noise_scheduler = schedule
        self.net_pred_type = net_pred_type
        if base_grid is not None:
            schedule.student_grid_cfg = _explicit_grid(base_grid, blocks=blocks)
        elif pdd_steps is not None:
            schedule.student_grid_cfg = _native_grid(pdd_steps, blocks=blocks)
        self.pdd_steps = len(base_grid) - 1 if pdd_steps is None and base_grid is not None else pdd_steps
        self.predictions = None
        self.unfused_predictions = None
        self.conditions = []

    pdd_primary_schedule = PDDMixin.pdd_primary_schedule
    pdd_base_grid = PDDMixin.pdd_base_grid
    pdd_advance_prediction = PDDMixin.pdd_advance_prediction
    _validate_pdd_indices = PDDMixin._validate_pdd_indices
    _validate_pdd_schedule_contract = PDDMixin._validate_pdd_schedule_contract
    pdd_block_weights = PDDMixin.pdd_block_weights
    pdd_advance_blocks = PDDMixin.pdd_advance_blocks
    pdd_teacher_target = PDDMixin.pdd_teacher_target
    _pdd_rk_mean_velocity = PDDMixin._pdd_rk_mean_velocity

    def __call__(
        self,
        x_t,
        step_index,
        condition=None,
        fused=False,
        next_step_index=None,
        precision_decoding=None,
        fwd_pred_type=None,
    ):
        assert fwd_pred_type in (None, "flow")
        self.conditions.append(condition)
        if not fused:
            assert next_step_index is None and precision_decoding is None
            source = self.unfused_predictions if self.unfused_predictions is not None else self.predictions
            if callable(source):
                return source(x_t, step_index, condition)
            return source
        start, end = step_index, next_step_index
        assert 0 <= start < end <= self.pdd_steps
        if callable(self.predictions):
            pred = self.predictions(x_t, start, end, condition)
        else:
            pred = self.predictions[start]
        return pred


class _ToyPredictionBase(FastGenNetwork):
    """One-channel network whose widened bias gives directly inspectable PDD heads."""

    def __init__(
        self,
        net_pred_type="flow",
        load_pretrained=True,
        schedule_shift=1.0,
        noise_scheduler=None,
        student_grid_cfg=None,
    ):
        super().__init__(
            net_pred_type=net_pred_type,
            schedule_type="shifted_rf",
            min_t=0.0,
            max_t=0.999,
            shift=schedule_shift,
            student_grid_cfg=student_grid_cfg,
        )
        if noise_scheduler is not None:
            noise_scheduler.student_grid_cfg = self.noise_scheduler.student_grid_cfg
            self.noise_scheduler = noise_scheduler
        self.proj = torch.nn.Linear(1, 1)

    @staticmethod
    def _is_in_meta_context():
        return False

    def forward(
        self, x_t, t, condition=None, fwd_pred_type=None, apply_output_conditioning=True, output_dtype="same", **kwargs
    ):
        del kwargs
        return self.transform_output(
            self.proj(x_t), x_t, t, condition, fwd_pred_type, output_dtype, apply_output_conditioning
        )


_ToyPDD = make_pdd(
    _ToyPredictionBase,
    PDDSpec(final_linear="proj", channel_layout="channel_major"),
)
