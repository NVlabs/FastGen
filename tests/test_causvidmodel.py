# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import gc
from contextlib import contextmanager
from typing import Callable

import pytest
import torch

import fastgen.methods.distribution_matching.causvid as causvid
from fastgen.configs.methods.config_causvid import ModelConfig
from fastgen.configs.net import CausalWan_1_3B_Config, Wan_1_3B_Config
from fastgen.configs.discriminator import Discriminator_Wan_1_3B_Config
from fastgen.methods import CausalKDModel, CausVidModel, DMD2Model
from fastgen.methods.causal_sampling import CausalStudentSamplingMixin
from fastgen.networks.kv_cache.base import _build_block_token_ranges
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.network_causal import CausalContext
from fastgen.networks.Wan.network import WanVideoEncoder
from fastgen.networks.noise_schedule import StudentGridConfig
from tests.utils import RunIf, check_grad_zero
from fastgen.utils.env import set_env_vars


@pytest.mark.parametrize("model_cls", [CausVidModel, CausalKDModel])
def test_causal_methods_share_instance_student_grid_cfg_loop(model_cls):
    class RecordingNet:
        chunk_size = 2
        total_num_frames = 4

        def __init__(self):
            self.clear_count = 0
            self.store_kv = []
            self.conditioned = []
            self.output_dtypes = []

        def clear_caches(self):
            self.clear_count += 1

        def __call__(
            self,
            x,
            t,
            condition=None,
            fwd_pred_type=None,
            causal_ctx=None,
            apply_output_conditioning=True,
            output_dtype="same",
        ):
            self.store_kv.append(causal_ctx.store_kv)
            self.conditioned.append(apply_output_conditioning)
            self.output_dtypes.append(output_dtype)
            assert fwd_pred_type == "x0"
            return x + 1

    assert model_cls._student_sample_loop is CausalStudentSamplingMixin._student_sample_loop
    model = object.__new__(model_cls)
    model.config = type("Config", (), {"student_sample_type": "sde", "context_noise": None})()
    net = RecordingNet()
    latents = torch.zeros(1, 1, 4, 1, 1)
    result = model._student_sample_loop(net, latents, torch.tensor([1.0, 0.0]))

    torch.testing.assert_close(result, torch.ones_like(result))
    assert net.clear_count == 2
    assert net.store_kv == [False, True, False, True]
    assert net.conditioned == [True, False, True, False]
    assert net.output_dtypes == [None, "same", None, "same"]


def test_store_causal_chunk_builds_the_context_noise_time_in_grid_precision():
    """Context noise is a scheduler time, so it follows the grid dtype rather than the state dtype."""

    class RecordingScheduler:
        def __init__(self):
            self.forward_process_times = []

        def forward_process(self, x, eps, t):
            del eps
            self.forward_process_times.append(t)
            return x

    class RecordingNet:
        def __init__(self):
            self.noise_scheduler = RecordingScheduler()
            self.cache_times = []

        def __call__(self, x, t, condition=None, fwd_pred_type=None, causal_ctx=None, apply_output_conditioning=True):
            assert causal_ctx.store_kv
            self.cache_times.append(t)
            return x

    model = object.__new__(CausVidModel)
    model.config = type("Config", (), {"student_sample_type": "ode", "context_noise": 0.3})()
    net = RecordingNet()
    ctx = CausalContext(cache_tag="pos", is_ar=True, chunk_size=2, total_num_frames=2)
    x_clean = torch.zeros(3, 1, 2, 1, 1, dtype=torch.bfloat16)
    student_grid = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)

    model._store_causal_chunk(net, x_clean, student_grid, None, ctx, 0)

    (t_forward,) = net.noise_scheduler.forward_process_times
    (t_cache,) = net.cache_times
    assert t_forward is t_cache
    assert t_cache.dtype == torch.float64
    assert t_cache.shape == (3,)
    torch.testing.assert_close(t_cache, torch.full((3,), 0.3, dtype=torch.float64))


def test_denoise_causal_chunk_can_stop_with_gradient_only_at_stop():
    class RecordingScheduler:
        def __init__(self):
            self.recorded_steps = []

        def step(self, x, x0, t, t_next, sample_type):
            del x, sample_type
            self.recorded_steps.append((t.clone(), t_next.clone()))
            return x0

    class RecordingNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.noise_scheduler = RecordingScheduler()
            self.calls = []

        def forward(self, x, t, condition=None, fwd_pred_type=None, causal_ctx=None, output_dtype="same"):
            del condition, fwd_pred_type, causal_ctx, output_dtype
            self.calls.append((t, torch.is_grad_enabled()))
            return x + self.weight

    model = object.__new__(CausVidModel)
    torch.nn.Module.__init__(model)
    model.config = type("Config", (), {"student_sample_type": "ode"})()
    net = RecordingNet()
    student_grid = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])

    result = model._denoise_causal_chunk(
        net,
        torch.zeros(1, 1),
        student_grid,
        condition=None,
        ctx=CausalContext(is_ar=True),
        start_frame=3,
        stop_step=2,
        grad_only_at_stop=True,
    )

    torch.testing.assert_close(torch.stack([call[0] for call in net.calls]), student_grid[:3, None])
    assert [call[1] for call in net.calls] == [False, False, True]
    assert len(net.noise_scheduler.recorded_steps) == 2
    assert result.requires_grad
    result.sum().backward()
    torch.testing.assert_close(net.weight.grad, torch.ones_like(net.weight))


@RunIf(min_gpus=1)
def test_denoise_causal_chunk_clears_detached_amp_weight_cache_at_stop():
    """A no-grad prefix must not poison the AMP weight cache used by the exit step."""

    class Scheduler:
        @staticmethod
        def step(x, x0, t, t_next, sample_type):
            del x, t, t_next, sample_type
            return x0

    class LinearNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False, device="cuda", dtype=torch.float32)
            self.noise_scheduler = Scheduler()

        def forward(self, x, t, condition=None, fwd_pred_type=None, causal_ctx=None, output_dtype="same"):
            del t, condition, fwd_pred_type, causal_ctx, output_dtype
            return self.proj(x)

    model = object.__new__(CausVidModel)
    torch.nn.Module.__init__(model)
    model.config = type("Config", (), {"student_sample_type": "ode"})()
    net = LinearNet()

    # The first call casts and caches the FP32 weight under no_grad. The stop
    # call in the same autocast context must rebuild a grad-bearing cast.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model._denoise_causal_chunk(
            net,
            torch.randn(2, 4, device="cuda", dtype=torch.float32),
            torch.tensor([1.0, 0.5, 0.0], device="cuda"),
            condition=None,
            ctx=CausalContext(is_ar=True),
            start_frame=0,
            stop_step=1,
            grad_only_at_stop=True,
        )

    assert result.requires_grad
    result.float().sum().backward()
    assert net.proj.weight.grad is not None
    assert torch.isfinite(net.proj.weight.grad).all()


@pytest.mark.parametrize("model_cls", [CausVidModel, CausalKDModel])
def test_causal_methods_honor_ode_student_grid_cfg(model_cls):
    class RecordingScheduler:
        def __init__(self):
            self.step_calls = []
            self.x0_to_eps_calls = []
            self.forward_process_calls = []

        def x0_to_eps(self, xt, x0, t):
            self.x0_to_eps_calls.append((xt.clone(), x0.clone(), t.clone()))
            return torch.full_like(x0, 2)

        def forward_process(self, x0, eps, t):
            self.forward_process_calls.append((x0.clone(), eps.clone(), t.clone()))
            return x0 + eps

        def step(self, x, x_pred, t, t_next, sample_type):
            self.step_calls.append((x.clone(), x_pred.clone(), t.clone(), t_next.clone(), sample_type))
            assert sample_type == "ode"
            eps = self.x0_to_eps(xt=x, x0=x_pred, t=t)
            return self.forward_process(x_pred, eps, t_next.expand(x_pred.shape[0]))

    class RecordingNet:
        chunk_size = 2
        total_num_frames = 2

        def __init__(self):
            self.noise_scheduler = RecordingScheduler()

        def clear_caches(self):
            pass

        def __call__(
            self,
            x,
            t,
            condition=None,
            fwd_pred_type=None,
            causal_ctx=None,
            apply_output_conditioning=True,
            output_dtype="same",
        ):
            assert fwd_pred_type == "x0"
            return x if causal_ctx.store_kv else x + 1

    model = object.__new__(model_cls)
    model.config = type("Config", (), {"student_sample_type": "ode", "context_noise": None})()
    net = RecordingNet()
    latents = torch.zeros(1, 1, 2, 1, 1)

    result = model._student_sample_loop(net, latents, torch.tensor([1.0, 0.5, 0.0]))

    assert torch.equal(result, torch.full_like(result, 4))
    assert len(net.noise_scheduler.step_calls) == 1
    assert net.noise_scheduler.step_calls[0][-1] == "ode"
    assert len(net.noise_scheduler.x0_to_eps_calls) == 1
    assert len(net.noise_scheduler.forward_process_calls) == 1
    xt, x0, t_cur = net.noise_scheduler.x0_to_eps_calls[0]
    assert torch.equal(xt, torch.zeros_like(xt))
    assert torch.equal(x0, torch.ones_like(x0))
    assert t_cur.ndim == 0
    assert torch.equal(t_cur, torch.ones_like(t_cur))
    _, eps, t_next = net.noise_scheduler.forward_process_calls[0]
    assert torch.equal(eps, torch.full_like(eps, 2))
    assert torch.equal(t_next, torch.full_like(t_next, 0.5))


def test_training_inputs_use_shared_t_and_eps_sampler(monkeypatch):
    frame_steps = torch.ones(2, 4, dtype=torch.long)

    class Scheduler:
        @staticmethod
        def sample_student_timesteps_inhom(batch_size, num_frames, chunk_size, device=None):
            del chunk_size
            return torch.zeros(batch_size, num_frames, device=device), frame_steps

        @staticmethod
        def forward_process(real_data, eps, t):
            assert eps.shape == real_data.shape
            assert t.shape == (real_data.shape[0], 1, real_data.shape[2], 1, 1)
            return real_data

    class Net:
        chunk_size = 2
        noise_scheduler = Scheduler()

    model = object.__new__(CausVidModel)
    torch.nn.Module.__init__(model)
    model.config = type("Config", (), {"backward_simulation": None})()
    model.net = Net()
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.input_shape = [1, 4, 2, 2]

    shared_t = torch.full((2,), 0.25)
    shared_eps = torch.full((2, 1, 4, 2, 2), 3.0)
    sampler_calls = []

    def sample_t_and_eps(batch_size, sample_t_cfg=None, student_step=None):
        sampler_calls.append((batch_size, sample_t_cfg, student_step))
        return shared_t, shared_eps

    monkeypatch.setattr(model, "_sample_t_and_eps", sample_t_and_eps)
    real_data = torch.zeros(2, 1, 4, 2, 2)

    _inputs = model._training_inputs({"real": real_data}, iteration=7)
    student_step = _inputs.student_step
    t = _inputs.t
    eps = _inputs.eps

    assert sampler_calls == [(2, None, None)]
    assert student_step is frame_steps
    assert t is shared_t
    assert eps is shared_eps


def test_causvid_rejects_sample_t_in_student_bucket():
    class Scheduler:
        @staticmethod
        def sample_student_timesteps_inhom(batch_size, num_frames, chunk_size, device=None):
            del chunk_size
            t = torch.zeros(batch_size, num_frames, device=device)
            return t, torch.zeros_like(t, dtype=torch.long)

        @staticmethod
        def forward_process(real_data, eps, t):
            del eps, t
            return real_data

    class Net:
        chunk_size = 2
        noise_scheduler = Scheduler()

    model = object.__new__(CausVidModel)
    torch.nn.Module.__init__(model)
    model.config = ModelConfig()
    model.config.sample_t_in_student_bucket = True
    model.net = Net()
    model.device = torch.device("cpu")
    model.input_shape = [1, 4, 2, 2]

    with pytest.raises(ValueError, match="CausVidModel provided student_step of shape None"):
        model._training_inputs({"real": torch.zeros(2, 1, 4, 2, 2)})


def test_causvid_rejects_step_generations():
    config = ModelConfig()
    config.student_generation = "step"

    with pytest.raises(ValueError, match=r"CausVidModel supports student_generation in \('x0',\)"):
        CausVidModel(config)


def _preprocessor_host(vae):
    """A network stand-in that owns ``vae``."""
    host = type("PreprocessorHost", (), {"get_preprocessor": FastGenNetwork.get_preprocessor})()
    host.preprocessors = {"vae": vae}
    return host


def test_causvid_extrapolation_uses_shared_chunk_primitives():
    class Scheduler:
        default_student_steps = 1

        @staticmethod
        def latents(x, t_init):
            return x

        @staticmethod
        def student_grid(num_steps=None, device=None, dtype=None):
            assert num_steps is None
            return torch.tensor([1.0, 0.0], device=device, dtype=dtype)

    class RecordingNet(torch.nn.Module):
        chunk_size = 2
        total_num_frames = 4
        vae = None

        def __init__(self):
            super().__init__()
            self.noise_scheduler = Scheduler()
            self.clear_count = 0

        def clear_caches(self):
            self.clear_count += 1

    class RecordingCausVid(CausVidModel):
        def _denoise_causal_chunk(
            self, net, x_chunk, student_grid, condition, ctx, start_frame, stop_step=None, grad_only_at_stop=False
        ):
            self.denoise_calls.append((start_frame, condition, ctx))
            return x_chunk + 1

        def _store_causal_chunk(self, net, x_clean, student_grid, condition, ctx, start_frame, t_cache=None):
            self.store_calls.append((start_frame, condition, ctx))

    model = object.__new__(RecordingCausVid)
    torch.nn.Module.__init__(model)
    model.precision_amp_infer = None
    model.denoise_calls = []
    model.store_calls = []
    net = RecordingNet()
    noise = torch.zeros(1, 1, 4, 1, 1)

    result = model.generator_fn_extrapolation(
        net,
        noise,
        num_segments=1,
        overlap_frames=0,
        condition="prompt",
    )

    torch.testing.assert_close(result, torch.ones_like(noise))
    assert [(start, condition) for start, condition, _ in model.denoise_calls] == [
        (0, "prompt"),
        (2, "prompt"),
    ]
    assert [(start, condition) for start, condition, _ in model.store_calls] == [
        (0, "prompt"),
        (2, "prompt"),
    ]
    assert all(
        denoise_ctx is store_ctx
        for (_, _, denoise_ctx), (_, _, store_ctx) in zip(model.denoise_calls, model.store_calls)
    )
    assert net.clear_count == 3


def test_causvid_extrapolation_segments_use_the_training_chunk_layout():
    """Every segment chunks like the training masks, with the remainder first: 5 frames in chunks of 2 run as
    [0, 3) and [3, 5). A continuation prefills its 3-frame overlap as that leading chunk; other overlaps fail."""

    class Scheduler:
        @staticmethod
        def latents(x, t_init):
            return x

        @staticmethod
        def student_grid(device=None, dtype=None):
            return torch.tensor([1.0, 0.0], device=device, dtype=dtype)

    class Net(torch.nn.Module):
        chunk_size = 2
        total_num_frames = 5

        def __init__(self):
            super().__init__()
            self.noise_scheduler = Scheduler()

        def clear_caches(self):
            pass

    class IdentityVAE:
        reencode_overlap = WanVideoEncoder.reencode_overlap
        latent_compression_factors = (1, 1, 1)

        @staticmethod
        def decode(latents):
            return latents

        @staticmethod
        def encode(pixels, mode):
            return pixels

    class RecordingCausVid(CausVidModel):
        def _denoise_causal_chunk(
            self, net, x_chunk, student_grid, condition, ctx, start_frame, stop_step=None, grad_only_at_stop=False
        ):
            self.denoised.append((start_frame, start_frame + x_chunk.shape[2]))
            return x_chunk

        def _store_causal_chunk(self, net, x_clean, student_grid, condition, ctx, start_frame, t_cache=None):
            self.stored.append((start_frame, start_frame + x_clean.shape[2]))

    model = object.__new__(RecordingCausVid)
    torch.nn.Module.__init__(model)
    model.net = _preprocessor_host(IdentityVAE())
    model.use_ema = []
    model.precision_amp_infer = None
    model.precision_amp_enc = None
    model.denoised, model.stored = [], []
    noise = torch.zeros(1, 1, 5, 1, 1)

    result = model.generator_fn_extrapolation(Net(), noise, num_segments=2, overlap_frames=3)

    assert _build_block_token_ranges(num_frames=5, frame_seqlen=1, chunk_size=2) == [(0, 3), (3, 5)]
    assert result.shape == (1, 1, 7, 1, 1)
    assert model.denoised == [(0, 3), (3, 5), (3, 5)]
    assert model.stored == [(0, 3), (3, 5), (0, 3), (3, 5)]
    for overlap_frames in (-1, 1, 2, 4, 5):
        with pytest.raises(ValueError, match="chunk boundary"):
            model.generator_fn_extrapolation(Net(), noise, num_segments=2, overlap_frames=overlap_frames)


def test_causvid_extrapolation_uses_vae_from_preprocessor_host_and_encoder_amp(monkeypatch):
    class Scheduler:
        @staticmethod
        def latents(x, t_init):
            return x

        @staticmethod
        def student_grid(device=None, dtype=None):
            return torch.tensor([1.0, 0.0], device=device, dtype=dtype)

    class SampledNet(torch.nn.Module):
        chunk_size = 1
        total_num_frames = 2

        def __init__(self):
            super().__init__()
            self.noise_scheduler = Scheduler()
            self.preprocessors = {}

        def forward(self, x, t, **kwargs):
            del t, kwargs
            return x

        def clear_caches(self):
            pass

    class RecordingVAE:
        reencode_overlap = WanVideoEncoder.reencode_overlap
        latent_compression_factors = (1, 1, 1)

        def __init__(self):
            self.decode_calls = 0
            self.encode_calls = 0

        def decode(self, latents):
            self.decode_calls += 1
            return latents

        def encode(self, pixels, mode):
            self.encode_calls += 1
            return pixels

    class RecordingCausVid(CausVidModel):
        def _denoise_causal_chunk(
            self, net, x_chunk, student_grid, condition, ctx, start_frame, stop_step=None, grad_only_at_stop=False
        ):
            del net, student_grid, condition, ctx, start_frame, stop_step, grad_only_at_stop
            return x_chunk + 1

        def _store_causal_chunk(self, net, x_clean, student_grid, condition, ctx, start_frame, t_cache=None):
            del net, x_clean, student_grid, condition, ctx, start_frame, t_cache

    vae = RecordingVAE()
    preprocessor_host = _preprocessor_host(vae)
    model = object.__new__(RecordingCausVid)
    torch.nn.Module.__init__(model)
    model.net = preprocessor_host
    model.use_ema = []
    model.precision_amp_infer = torch.float16
    model.precision_amp_enc = torch.bfloat16

    inference_contexts = []

    @contextmanager
    def record_inference_mode(*modules, precision_amp, device_type):
        inference_contexts.append((modules, precision_amp, device_type))
        yield

    monkeypatch.setattr(
        "fastgen.methods.distribution_matching.causvid.basic_utils.inference_mode",
        record_inference_mode,
    )

    sampled_net = SampledNet()
    result = model.generator_fn_extrapolation(
        sampled_net,
        torch.zeros(1, 1, 2, 1, 1),
        num_segments=2,
        overlap_frames=1,
    )

    assert result.shape == (1, 1, 3, 1, 1)
    assert vae.decode_calls == 1
    assert vae.encode_calls == 1
    assert inference_contexts == [
        ((sampled_net,), model.precision_amp_infer, "cpu"),
        ((vae,), model.precision_amp_enc, "cpu"),
    ]


@pytest.mark.parametrize("overlap_frames", [1, 2, 3])
def test_causvid_extrapolation_bridges_the_overlap_at_the_vae_temporal_compression(overlap_frames):
    """The next segment opens with the overlapped latents re-encoded as a clip start: the VAE encodes the pixel frames
    they cover, from the last frame of the first one, so the first becomes a single-frame latent and the others are
    encoded with that context. They prefill the KV cache like finished chunks, which applies the context noise."""

    class Scheduler:
        @staticmethod
        def latents(x, t_init):
            return x

        @staticmethod
        def student_grid(device=None, dtype=None):
            return torch.tensor([1.0, 0.0], device=device, dtype=dtype)

    class Net(torch.nn.Module):
        chunk_size = 1
        total_num_frames = 4

        def __init__(self):
            super().__init__()
            self.noise_scheduler = Scheduler()

        def clear_caches(self):
            pass

    class TemporalVAE:
        reencode_overlap = WanVideoEncoder.reencode_overlap
        latent_compression_factors = (4, 8, 8)

        def __init__(self):
            self.encoded = []

        def decode(self, latents):
            # Pixel frame p holds p: latent 0 decodes to frame 0 and latent i > 0 to frames 4i - 3 to 4i.
            num_frames = 1 + 4 * (latents.shape[2] - 1)
            frames = torch.arange(num_frames, dtype=latents.dtype).view(1, 1, -1, 1, 1)
            return frames.expand(latents.shape[0], latents.shape[1], -1, *latents.shape[3:])

        def encode(self, pixels, mode):
            # A causal clip's latent j > 0 ends at pixel frame 4j, so it stands for that frame here.
            assert mode == "argmax"  # the posterior mean, as the original Wan VAE encodes
            self.encoded.append(pixels)
            return 100 + pixels[:, :, ::4]

    class RecordingCausVid(CausVidModel):
        def _denoise_causal_chunk(
            self, net, x_chunk, student_grid, condition, ctx, start_frame, stop_step=None, grad_only_at_stop=False
        ):
            # Latent frame i of every segment denoises to 10 + i.
            return torch.full_like(x_chunk, 10.0 + start_frame)

        def _store_causal_chunk(self, net, x_clean, student_grid, condition, ctx, start_frame, t_cache=None):
            self.stored.append((start_frame, x_clean.flatten().tolist()))

    vae = TemporalVAE()
    model = object.__new__(RecordingCausVid)
    torch.nn.Module.__init__(model)
    model.net = _preprocessor_host(vae)
    model.use_ema = []
    model.precision_amp_infer = None
    model.precision_amp_enc = None
    model.stored = []

    result = model.generator_fn_extrapolation(
        Net(), torch.zeros(1, 1, 4, 1, 1), num_segments=2, overlap_frames=overlap_frames
    )

    # The 4 latents decode to 13 pixel frames; the first overlapped latent 4 - k ends at pixel frame 4 * (4 - k), and
    # the clip from there to the end encodes to k latents ending at frames 4 * (4 - k), ..., 12.
    [clip_pixels] = vae.encoded
    assert clip_pixels.flatten().tolist() == [float(frame) for frame in range(4 * (4 - overlap_frames), 13)]
    bridge = [100.0 + frame for frame in range(4 * (4 - overlap_frames), 13, 4)]
    # The first segment stores its 4 chunks before the second prefills its overlap.
    assert model.stored[4 : 4 + overlap_frames] == [(i, [value]) for i, value in enumerate(bridge)]
    expected = [10.0, 11.0, 12.0, 13.0] + [10.0 + i for i in range(overlap_frames, 4)]
    assert result.flatten().tolist() == expected


def test_gen_data_from_net_uses_teacher_forcing_causal_context():
    grid = torch.tensor([0.75, 0.25, 0.0])

    class GridScheduler:
        @staticmethod
        def student_grid(device=None):
            return grid.to(device)

    class RecordingNet:
        chunk_size = 3
        noise_scheduler = GridScheduler()

        def __init__(self):
            self.call = None

        def __call__(self, x, t, condition=None, fwd_pred_type=None, causal_ctx=None):
            self.call = {
                "x": x,
                "t": t,
                "condition": condition,
                "fwd_pred_type": fwd_pred_type,
                "causal_ctx": causal_ctx,
            }
            return x + 1

    net = RecordingNet()
    model = CausVidModel.__new__(CausVidModel)
    model.net = net

    input_student = torch.zeros(2, 4, 5, 6, 7)
    student_step = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 1, 2, 2]])
    condition = torch.ones(2, 8, 9)

    generation = model.gen_data_from_net(input_student, student_step, condition=condition)
    output = generation.data
    generation_metrics = generation.metrics

    assert torch.equal(output, input_student + 1)
    assert generation_metrics == {}
    assert generation.next_state is None
    assert generation.x0 is generation.data
    assert net.call["x"] is input_student
    assert torch.equal(net.call["t"], grid[student_step])
    assert net.call["condition"] is condition
    assert net.call["fwd_pred_type"] == "x0"

    causal_ctx = net.call["causal_ctx"]
    assert isinstance(causal_ctx, CausalContext)
    assert causal_ctx.cache_tag == "pos"
    assert causal_ctx.chunk_size == net.chunk_size
    assert causal_ctx.total_num_frames == input_student.shape[2]
    assert causal_ctx.store_kv is False
    assert causal_ctx.is_ar is False


@RunIf(min_gpus=1)
def test_causvid_trains_on_a_frame_count_the_chunk_size_does_not_divide(tmp_path):
    """8 frames in chunks of 3 train with one student step per mask chunk, [0, 5) and [5, 8), in the student and
    the fake-score updates."""
    from tests.test_network import _tiny_causal_wan_family_network

    device = torch.device("cuda")
    model = object.__new__(CausVidModel)
    torch.nn.Module.__init__(model)
    model.config = ModelConfig()
    model.config.gan_loss_weight_gen = 0.0
    model.config.sample_t_cfg.max_t = 0.999
    model.device = device
    model.precision = torch.float32
    model.use_ema = []
    model.input_shape = [4, 8, 4, 4]
    # Without a causal context, the teacher and fake score attend to the whole video.
    model.net, model.teacher, model.fake_score = (
        _tiny_causal_wan_family_network("causal_wan", tmp_path / name).to(device)
        for name in ("net", "teacher", "fake_score")
    )
    model.net.chunk_size = 3
    model.net.noise_scheduler.student_grid_cfg = StudentGridConfig(steps=4)
    forward, times = model.net.forward, []

    def record_times(x, t, **kwargs):
        times.append(t)
        return forward(x, t, **kwargs)

    model.net.forward = record_times
    data = {
        "real": torch.randn(2, *model.input_shape, device=device),
        "condition": torch.randn(2, 6, 8, device=device),
        "neg_condition": torch.zeros(2, 6, 8, device=device),
    }
    for iteration in (0, 1):
        loss_map, _ = model.single_train_step(data, iteration)
        loss_map["total_loss"].backward()
        assert torch.isfinite(loss_map["total_loss"])

    assert len(times) == 2
    for t in times:
        for start, end in _build_block_token_ranges(num_frames=8, frame_seqlen=1, chunk_size=3):
            assert (t[:, start:end] == t[:, start : start + 1]).all()
    assert any(param.grad is not None for param in model.net.parameters())
    assert any(param.grad is not None for param in model.fake_score.parameters())


def test_causvid_setup_warns_once_without_a_causal_initialization(monkeypatch):
    """The causal student starts from model.pretrained_student_net_path or from the checkpoint the trainer loads
    from pretrained_ckpt_path, which sets resume_iter, before the optimizers are built; without either, setup warns
    once."""
    warnings = []
    monkeypatch.setattr(causvid.logger, "warning", warnings.append)
    monkeypatch.setattr(DMD2Model, "init_optimizers", lambda self: None)
    model = object.__new__(CausVidModel)
    torch.nn.Module.__init__(model)
    model.config = ModelConfig()

    model.init_optimizers()
    [message] = warnings
    assert "pretrained_student_net_path" in message and "pretrained_ckpt_path" in message

    warnings.clear()
    model.config.pretrained_student_net_path = "causal_init.pt"
    model.init_optimizers()
    model.config.pretrained_student_net_path = None
    model.resume_iter = 0
    model.init_optimizers()
    assert warnings == []


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    set_env_vars()

    config = ModelConfig()
    config.net = copy.deepcopy(CausalWan_1_3B_Config)
    config.net.chunk_size = 2  # Override chunk_size to match 2-frame test data
    config.teacher = Wan_1_3B_Config  # teacher is still bidirectional network

    # Set discriminator config with lightweight overrides. Deep-copy so the num_blocks
    # edit below stays out of the shared module-level config.
    config.discriminator = copy.deepcopy(Discriminator_Wan_1_3B_Config)
    config.discriminator.num_blocks = 1

    # Set device and precision
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.precision = "float32"
    config.precision_model = "float64" if config.device.type == "cpu" else "bfloat16"
    config.pretrained_model_path = None  # disable ckpt loading
    config.student_update_freq = 2
    config.student_grid_cfg.steps = 4  # multistep student distillation

    # Set the correct timestep range
    config.sample_t_cfg.max_t = 0.999
    config.sample_t_cfg.min_t = 0.001

    # Use small dimensions for testing (Wan uses 16 channels)
    config.input_shape = [16, 2, 4, 4]  # [C, T, H, W]
    config.enable_preprocessors = False  # The data below holds pre-embedded conditions and latents
    config.gan_loss_weight_gen = 0.001  # Set GAN loss weight

    model = CausVidModel(config)
    # only use a single block for testing
    model.net.transformer.blocks = model.net.transformer.blocks[:1]
    model.teacher.transformer.blocks = model.teacher.transformer.blocks[:1]

    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    channels, n_frames, height, width = config.input_shape

    # Create much smaller mock video data (Wan uses UMT5: seq_len=512, hidden_dim=4096).
    # Forwards take conditioning in the network's compute dtype, as ``build_conditioning`` returns it.
    condition_dtype = model.net.precision_model
    data = {
        "real": torch.randn(batch_size, channels, n_frames, height, width, device=model.device, dtype=model.precision),
        "condition": torch.randn(batch_size, 512, 4096, device=model.device, dtype=condition_dtype),
        "neg_condition": torch.zeros(batch_size, 512, 4096, device=model.device, dtype=condition_dtype),
    }

    return model, data


@RunIf(min_gpus=1)
def test_causal_timestep_sampling(get_model_data):
    model, data = get_model_data
    _inputs = model._training_inputs(data, iteration=0)
    generation_kwargs = _inputs.generation_kwargs
    loss_scale = _inputs.loss_scale
    assert generation_kwargs == {}
    assert loss_scale is None

    # Test that the causal model uses heterogeneous timesteps
    batch_size = data["real"].shape[0]
    num_frames = data["real"].shape[2]

    # Use model's chunk size (overridden to 2 in fixture to match 2-frame test data)
    chunk_size = getattr(model.net, "chunk_size", 2)

    # Access noise scheduler correctly through the model's noise_scheduler attribute
    t_inhom, _ = model.teacher.noise_scheduler.sample_student_timesteps_inhom(
        n=batch_size, seq_len=num_frames, chunk_size=chunk_size, num_steps=4
    )

    # Check that timesteps have the right shape
    assert t_inhom.shape == (batch_size, num_frames)

    # Timesteps within a mask chunk should be the same
    for start, end in _build_block_token_ranges(num_frames, frame_seqlen=1, chunk_size=chunk_size):
        assert torch.allclose(t_inhom[:, start : start + 1], t_inhom[:, start:end], atol=1e-6)


@RunIf(min_gpus=1)
def test_single_train_step_student_update(get_model_data):
    model, data = get_model_data
    # Run the training step
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "vsd_loss" in loss_map
    assert "gan_loss_gen" in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)
    assert isinstance(outputs["input_rand"], torch.Tensor)

    gen_tensor = outputs["gen_rand"]()
    assert gen_tensor.shape == data["real"].shape
    assert outputs["input_rand"].shape == data["real"].shape


@RunIf(min_gpus=1)
def test_single_train_step_fake_score_update(get_model_data):
    model, data = get_model_data

    # Run the training step
    loss_map, outputs = model.single_train_step(data, 1)

    # Assertions
    assert "fake_score_loss" in loss_map
    assert "gan_loss_disc" in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)
    assert isinstance(outputs["input_rand"], torch.Tensor)

    # Check that losses are finite
    assert torch.isfinite(loss_map["total_loss"])
    assert torch.isfinite(loss_map["fake_score_loss"])


@RunIf(min_gpus=1)
def test_optimizers(get_model_data):
    model, data = get_model_data

    gc.collect()
    torch.cuda.empty_cache()

    # Test for both student and fake_score optimizer
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)

    # Test for both student and fake_score optimizer zero grad, after at least one backward pass
    model.optimizers_zero_grad(2)
    check_grad_zero(model.net)
    model.optimizers_zero_grad(3)
    check_grad_zero(model.discriminator)
    check_grad_zero(model.fake_score)
