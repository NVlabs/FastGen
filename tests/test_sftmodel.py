# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import gc
from types import SimpleNamespace

import pytest
import torch

from fastgen.methods import CausalSFTModel, CausalTeacherForcingSFTModel, SFTModel
from fastgen.configs.methods.config_sft import ModelConfig
from fastgen.configs.methods.config_sft_causal import ModelConfig as CausalModelConfig
from fastgen.configs.methods.config_sft_teacher_forcing import ModelConfig as TeacherForcingModelConfig
from fastgen.configs.net import CausalWan_1_3B_Config
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.networks.kv_cache.base import _build_block_token_ranges
from fastgen.networks.noise_schedule import RFNoiseSchedule, ShiftedRFNoiseSchedule
from fastgen.utils.global_vars import set_temp_global_vars
from fastgen.utils.env import set_env_vars
from tests.model_utils import MultimodalNet
from tests.utils import RunIf


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    instance = ModelConfig()
    opts = [
        "-",
        "backbone.img_resolution=8",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=False",
    ]
    instance.net = override_config_with_opts(instance.net, opts)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None  # disable ckpt loading
    instance.input_shape = [3, 8, 8]

    # SFT-specific configurations
    instance.cond_dropout_prob = 0.1
    instance.cond_keys_no_dropout = []

    model = SFTModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    # Use one-hot encoded labels as EDM network expects them
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
    neg_condition = torch.zeros(batch_size, 10)

    # Create mock data
    data = {
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_condition.to(model.device, model.precision),
    }

    return model, data


def test_single_train_step(get_model_data):
    """Test the single training step of SFT model."""
    model, data = get_model_data

    # Run the training step
    loss_map, outputs = model.single_train_step(data, 0)

    # Test loss_map structure
    assert isinstance(loss_map, dict)
    assert "total_loss" in loss_map
    assert "dsm_loss" in loss_map
    assert isinstance(loss_map["total_loss"], torch.Tensor)
    assert isinstance(loss_map["dsm_loss"], torch.Tensor)
    assert loss_map["total_loss"].detach().item() >= 0.0
    assert loss_map["dsm_loss"].detach().item() >= 0.0
    # For SFT, total_loss should equal dsm_loss
    assert torch.allclose(loss_map["total_loss"], loss_map["dsm_loss"])

    # Test outputs structure
    assert isinstance(outputs, dict)
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["input_rand"], torch.Tensor)
    assert outputs["input_rand"].shape == data["real"].shape
    # gen_rand should be a callable (partial function)
    assert callable(outputs["gen_rand"])
    assert outputs["gen_rand"].keywords["neg_condition"] is data["neg_condition"]
    assert isinstance(outputs["gen_rand_train"], torch.Tensor)
    assert outputs["gen_rand_train"].shape == data["real"].shape


@pytest.mark.parametrize("optimize_reward", [False, True])
@pytest.mark.parametrize("dropout_probability", [None, 0.0, 1.0])
def test_single_train_step_applies_a_configured_reward_to_the_x0_prediction(
    get_model_data, optimize_reward, dropout_probability
):
    """SFT scores its x0 prediction like the other methods; only an optimized reward joins the objective."""
    model, data = get_model_data
    model.config.optimize_reward = optimize_reward
    model.config.cond_dropout_prob = dropout_probability
    rewarded = []

    def reward(gen_data, reward_data, condition_dropout_mask=None):
        rewarded.append((gen_data, reward_data, condition_dropout_mask))
        return {"total_loss": gen_data.mean(), "aesthetic": torch.tensor(0.5)}

    model.reward = reward
    loss_map, outputs = model.single_train_step(data, 0)

    ((scored, scored_data, dropout_mask),) = rewarded
    assert scored_data is data
    if dropout_probability is None:
        assert dropout_mask is None
    else:
        assert dropout_mask.shape == (data["real"].shape[0],) and dropout_mask.dtype == torch.bool
        assert dropout_mask.eq(dropout_probability == 1.0).all()
    assert scored.requires_grad == optimize_reward
    torch.testing.assert_close(scored.detach(), outputs["gen_rand_train"])
    torch.testing.assert_close(loss_map["aesthetic"], torch.tensor(0.5))
    expected_total = loss_map["dsm_loss"] + loss_map["reward_loss"] if optimize_reward else loss_map["dsm_loss"]
    torch.testing.assert_close(loss_map["total_loss"], expected_total)


def test_mix_condition_tensor(get_model_data):
    """Test the _mix_condition method with tensor inputs."""
    model, data = get_model_data

    condition = data["condition"]
    neg_condition = data["neg_condition"]

    # Test with cond_dropout_prob = None (no dropout)
    model.config.cond_dropout_prob = None
    mixed_condition, dropout_mask = model._mix_condition(condition, neg_condition)
    assert dropout_mask is None
    assert torch.allclose(mixed_condition, condition)

    # Test with cond_dropout_prob = 1.0 (full dropout)
    model.config.cond_dropout_prob = 1.0
    mixed_condition, dropout_mask = model._mix_condition(condition, neg_condition)
    assert dropout_mask.all()
    assert torch.allclose(mixed_condition, neg_condition)

    # Test with intermediate cond_dropout_prob
    model.config.cond_dropout_prob = 0.5
    mixed_condition, dropout_mask = model._mix_condition(condition, neg_condition)
    assert mixed_condition.shape == condition.shape
    assert dropout_mask.dtype == torch.bool
    torch.testing.assert_close(mixed_condition, torch.where(dropout_mask[:, None], neg_condition, condition))


def test_mix_condition_dict(get_model_data):
    """Test the _mix_condition method with dictionary inputs."""
    model, data = get_model_data

    # Create dictionary conditions
    dict_condition = {"text_embeds": data["condition"], "other_info": torch.ones_like(data["condition"])}
    dict_neg_condition = {"text_embeds": data["neg_condition"], "other_info": torch.zeros_like(data["neg_condition"])}

    # Test with cond_dropout_prob = None (no dropout)
    model.config.cond_dropout_prob = None
    mixed_condition, dropout_mask = model._mix_condition(dict_condition, dict_neg_condition)
    assert dropout_mask is None
    assert torch.allclose(mixed_condition["text_embeds"], dict_condition["text_embeds"])
    assert torch.allclose(mixed_condition["other_info"], dict_condition["other_info"])

    # Test with cond_dropout_prob = 1.0 (full dropout)
    model.config.cond_dropout_prob = 1.0
    mixed_condition, dropout_mask = model._mix_condition(dict_condition, dict_neg_condition)
    assert dropout_mask.all()
    assert torch.allclose(mixed_condition["text_embeds"], dict_neg_condition["text_embeds"])
    assert torch.allclose(mixed_condition["other_info"], dict_neg_condition["other_info"])


def test_mix_condition_with_no_dropout_keys(get_model_data):
    """Test the _mix_condition method with keys that should not be dropped."""
    model, data = get_model_data

    # Set cond_keys_no_dropout
    model.config.cond_keys_no_dropout = {"other_info"}

    # Create dictionary conditions
    dict_condition = {"text_embeds": data["condition"], "other_info": torch.ones_like(data["condition"])}
    dict_neg_condition = {"text_embeds": data["neg_condition"], "other_info": torch.zeros_like(data["neg_condition"])}

    # Test with cond_dropout_prob = 1.0 (full dropout)
    model.config.cond_dropout_prob = 1.0
    mixed_condition, dropout_mask = model._mix_condition(dict_condition, dict_neg_condition)
    assert dropout_mask.all()
    # text_embeds should be dropped (replaced with neg_condition)
    assert torch.allclose(mixed_condition["text_embeds"], dict_neg_condition["text_embeds"])
    # other_info should NOT be dropped (kept as original condition)
    assert torch.allclose(mixed_condition["other_info"], dict_condition["other_info"])


def test_mix_condition_nested_tuple_value(get_model_data):
    """Cover Flux2-style ``text_embeds = (None, prompt_embeds)`` dict values.

    Exercises the nested tuple/list recursion alongside
    ``cond_keys_no_dropout``: ``first_leaf`` must walk through the tuple
    of the droppable key to derive (batch_size, device, dtype), ``tmap``
    must rebuild the tuple element-wise while leaving the ``None`` slot
    untouched, and the no-drop key must pass through unchanged via the
    ``condition | mixed`` merge.
    """
    model, data = get_model_data

    model.config.cond_keys_no_dropout = {"other_info"}
    other = torch.ones_like(data["condition"])
    other_neg = torch.zeros_like(data["neg_condition"])
    cond = {"text_embeds": (None, data["condition"]), "other_info": other}
    neg = {"text_embeds": (None, data["neg_condition"]), "other_info": other_neg}

    # cond_dropout_prob = None: pass-through unchanged.
    model.config.cond_dropout_prob = None
    out, dropout_mask = model._mix_condition(cond, neg)
    assert dropout_mask is None
    assert out["text_embeds"][0] is None
    assert torch.allclose(out["text_embeds"][1], cond["text_embeds"][1])
    assert torch.allclose(out["other_info"], other)

    # cond_dropout_prob = 1.0: droppable tensor slot replaced with neg, None slot
    # preserved, no-drop key still equal to the original (NOT to ``other_neg``).
    model.config.cond_dropout_prob = 1.0
    out, dropout_mask = model._mix_condition(cond, neg)
    assert dropout_mask.all()
    assert isinstance(out["text_embeds"], tuple)
    assert out["text_embeds"][0] is None
    assert torch.allclose(out["text_embeds"][1], neg["text_embeds"][1])
    assert torch.allclose(out["other_info"], other)

    # cond_dropout_prob = 0.0: tensor slot kept as cond, None slot preserved,
    # no-drop key unchanged.
    model.config.cond_dropout_prob = 0.0
    out, dropout_mask = model._mix_condition(cond, neg)
    assert not dropout_mask.any()
    assert out["text_embeds"][0] is None
    assert torch.allclose(out["text_embeds"][1], cond["text_embeds"][1])
    assert torch.allclose(out["other_info"], other)


def test_mix_condition_all_keys_protected_returns_no_dropout_mask(get_model_data):
    model, data = get_model_data
    model.config.cond_dropout_prob = 1.0
    model.config.cond_keys_no_dropout = {"image"}
    condition = {"image": data["condition"]}
    mixed, dropout_mask = model._mix_condition(condition, {"image": data["neg_condition"]})
    assert mixed is condition
    assert dropout_mask is None


def test_mix_condition_all_none_droppable_raises(get_model_data):
    """All-``None`` droppable values + cond dropout requested → loud failure.

    With ``cond_dropout_prob`` set there must be *something* to drop; if
    every droppable value is ``None`` (no tensor leaf reachable to derive
    the mask shape / device / dtype), ``first_leaf`` raises.
    """
    model, _ = get_model_data
    model.config.cond_dropout_prob = 0.5
    model.config.cond_keys_no_dropout = []

    cond = {"text_embeds": None}
    neg = {"text_embeds": None}

    with pytest.raises(ValueError, match="no Tensor leaf"):
        model._mix_condition(cond, neg)


def test_mix_condition_delegates_to_network(get_model_data):
    """Networks exposing ``mix_condition`` own the mixing (e.g. Cosmos3's
    variable-length token-ID lists cannot be tensor-mixed)."""
    model, data = get_model_data
    model.config.cond_dropout_prob = 0.25

    calls = {}
    expected_mask = torch.ones(data["condition"].shape[0], dtype=torch.bool)

    class _StructuredNet(torch.nn.Module):
        def mix_condition(self, condition, neg_condition, cond_dropout_prob):
            calls["args"] = (condition, neg_condition, cond_dropout_prob)
            return neg_condition, expected_mask

    model.net = _StructuredNet()
    out, dropout_mask = model._mix_condition(data["condition"], data["neg_condition"])

    assert out is data["neg_condition"]
    assert dropout_mask is expected_mask
    assert calls["args"] == (data["condition"], data["neg_condition"], 0.25)

    # cond_dropout_prob = None short-circuits before the delegation.
    model.config.cond_dropout_prob = None
    calls.clear()
    out, dropout_mask = model._mix_condition(data["condition"], data["neg_condition"])
    assert out is data["condition"]
    assert dropout_mask is None
    assert not calls


def test_cosmos3_mix_condition_swaps_token_id_samples():
    """``Cosmos3.mix_condition`` swaps whole per-sample token-ID entries and passes
    shared tensor entries (condition_mask, conditioning_latents) through."""
    from fastgen.networks.cosmos3.network import Cosmos3

    stub = Cosmos3.__new__(Cosmos3)
    ids = [torch.arange(5), torch.arange(7)]
    neg_ids = [torch.arange(5) + 100, torch.arange(3) + 100]
    mask = torch.zeros(2, 1, 4, 1, 1, dtype=torch.bool)
    cond = {"input_ids": ids, "condition_mask": mask}
    neg = {"input_ids": neg_ids, "condition_mask": mask}

    dropped, dropout_mask = Cosmos3.mix_condition(stub, cond, neg, cond_dropout_prob=1.0)
    assert dropout_mask.all()
    assert all(torch.equal(m, n) for m, n in zip(dropped["input_ids"], neg_ids))
    assert dropped["condition_mask"] is mask

    kept, dropout_mask = Cosmos3.mix_condition(stub, cond, neg, cond_dropout_prob=0.0)
    assert not dropout_mask.any()
    assert all(torch.equal(m, p) for m, p in zip(kept["input_ids"], ids))


def test_cosmos3_mix_condition_mask_follows_token_ids_device():
    from fastgen.networks.cosmos3.network import Cosmos3

    stub = Cosmos3.__new__(Cosmos3)
    ids = [torch.arange(5, device="meta"), torch.arange(7, device="meta")]
    _, dropout_mask = Cosmos3.mix_condition(stub, {"input_ids": ids}, {"input_ids": ids}, cond_dropout_prob=0.5)
    assert dropout_mask.device == ids[0].device


def test_generator_fn(get_model_data):
    """SFT forwards only arguments declared by the network sampler."""
    model, data = get_model_data
    model.net.noise_scheduler.student_grid_cfg.steps = 4
    received = {}

    # Keep this signature strict so generic sampler kwargs cannot be hidden by the test.
    class MockNet(torch.nn.Module):
        noise_scheduler = model.net.noise_scheduler

        def sample(
            self,
            noise,
            condition=None,
            neg_condition=None,
            sample_steps=None,
            sample_type="ode",
            guidance_cfg=None,
        ):
            received.update(
                condition=condition,
                neg_condition=neg_condition,
                sample_steps=sample_steps,
                sample_type=sample_type,
                guidance_cfg=guidance_cfg,
            )
            return torch.randn_like(noise)

    mock_net = MockNet()
    noise = torch.randn_like(data["real"])

    # Test generator function
    with set_temp_global_vars({"STUDENT_STEPS": 2}):
        expected_steps = mock_net.noise_scheduler.default_student_steps
        result = model.generator_fn(
            net=mock_net,
            noise=noise,
            condition=data["condition"],
            neg_condition=data["neg_condition"],
        )

    assert isinstance(result, torch.Tensor)
    assert result.shape == noise.shape
    assert expected_steps == 2
    assert received["sample_steps"] == expected_steps
    assert received["sample_type"] == model.config.student_sample_type == "ode"
    assert received["guidance_cfg"] is model.config.guidance_cfg
    assert set(received) == {"condition", "neg_condition", "sample_steps", "sample_type", "guidance_cfg"}


def test_sample_types_the_network_samplers_cannot_run_are_rejected_before_building():
    config = ModelConfig(device="cpu", student_sample_type="ddpm")
    with pytest.raises(ValueError, match="SFTModel student_sample_type must be one of 'ode', 'sde'; got 'ddpm'"):
        SFTModel(config)


def test_causal_generator_fn_forwards_context_noise():
    """Context noise is a causal-sampling kwarg, not a generic SFT kwarg."""
    model = object.__new__(CausalSFTModel)
    torch.nn.Module.__init__(model)
    model.config = CausalModelConfig(context_noise=0.25)
    model.precision_amp_infer = None
    received = {}

    class MockNet(torch.nn.Module):
        noise_scheduler = SimpleNamespace(default_student_steps=1)

        def sample(
            self,
            noise,
            condition=None,
            neg_condition=None,
            sample_steps=None,
            sample_type="ode",
            guidance_cfg=None,
            context_noise=None,
        ):
            received.update(
                condition=condition,
                neg_condition=neg_condition,
                sample_steps=sample_steps,
                sample_type=sample_type,
                guidance_cfg=guidance_cfg,
                context_noise=context_noise,
            )
            return torch.zeros_like(noise)

    noise = torch.randn(1, 3, 8, 8)
    result = model.generator_fn(net=MockNet(), noise=noise)

    assert result.shape == noise.shape
    assert received["context_noise"] == model.config.context_noise


def test_causal_sft_loss_uses_frame_aligned_timesteps(monkeypatch):
    """The DSM target must vary over video frames, not channels."""
    batch_size, channels, num_frames, height, width = 2, 3, 4, 2, 2
    sampled_t = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ]
    )

    class RecordingSchedule(ShiftedRFNoiseSchedule):
        def __init__(self):
            super().__init__(shift=2.0)
            self.loss_t = None

        def sample_t_inhom(self, batch_size, seq_len, chunk_size, device=None, **kwargs):
            del kwargs
            assert (batch_size, seq_len, chunk_size) == (2, num_frames, 2)
            return sampled_t.to(device=device)

        def cond_velocity(self, x, eps, t, output_dtype="same"):
            self.loss_t = t.detach().clone()
            return super().cond_velocity(x, eps, t, output_dtype=output_dtype)

    class ZeroFlowNet(torch.nn.Module):
        chunk_size = 2
        net_pred_type = "flow"

        def __init__(self, noise_scheduler):
            super().__init__()
            self.noise_scheduler = noise_scheduler

        def forward(self, x, t, condition=None, causal_ctx=None):
            del condition
            assert torch.equal(t, sampled_t)
            assert causal_ctx.total_num_frames == num_frames
            return torch.zeros_like(x)

    model = object.__new__(CausalSFTModel)
    torch.nn.Module.__init__(model)
    model.config = ModelConfig()
    model.config.cond_dropout_prob = None
    model.device = torch.device("cpu")
    model.input_shape = [channels, num_frames, height, width]
    schedule = RecordingSchedule()
    model.net = ZeroFlowNet(schedule)

    monkeypatch.setattr(CausalSFTModel, "_to_x0_prediction", lambda self, noisy_data, net_pred, t: net_pred)
    monkeypatch.setattr(CausalSFTModel, "_get_outputs", lambda self, gen_data, **kwargs: {})

    data = {"real": torch.randn(batch_size, channels, num_frames, height, width)}
    loss_map, _ = model.single_train_step(data, iteration=0)

    expected_t = sampled_t[:, None, :, None, None]
    assert torch.equal(schedule.loss_t, expected_t)
    assert torch.isfinite(loss_map["total_loss"])


@pytest.mark.parametrize(
    ("model_cls", "config_cls"),
    [(CausalSFTModel, ModelConfig), (CausalTeacherForcingSFTModel, TeacherForcingModelConfig)],
)
@pytest.mark.parametrize("dropout_probability", [None, 1.0])
def test_causal_single_train_step_applies_a_configured_reward_to_the_x0_prediction(
    model_cls, config_cls, dropout_probability
):
    """The causal SFT steps score the x0 prediction they return as the training sample."""

    class ZeroFlowNet(torch.nn.Module):
        chunk_size = 2
        net_pred_type = "flow"

        def __init__(self):
            super().__init__()
            self.noise_scheduler = RFNoiseSchedule()

        def forward(self, x, t, condition=None, causal_ctx=None):
            del t, condition, causal_ctx
            return torch.zeros_like(x)

    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.config = config_cls()
    model.config.cond_dropout_prob = dropout_probability
    model.config.sample_t_cfg.max_t = 0.999
    model.device = torch.device("cpu")
    model.precision = torch.float32
    model.use_ema = []
    model.input_shape = [3, 4, 2, 2]
    model.net = ZeroFlowNet()
    rewarded = []

    def reward(gen_data, reward_data, condition_dropout_mask=None):
        rewarded.append((gen_data, reward_data, condition_dropout_mask))
        return {"total_loss": torch.tensor(0.25)}

    model.reward = reward
    data = {
        "real": torch.randn(2, *model.input_shape),
        "condition": torch.ones(2, 1),
        "neg_condition": torch.zeros(2, 1),
    }
    loss_map, outputs = model.single_train_step(data, iteration=0)

    ((scored, scored_data, dropout_mask),) = rewarded
    assert scored_data is data
    if dropout_probability is None:
        assert dropout_mask is None
    else:
        assert dropout_mask.dtype == torch.bool and dropout_mask.all()
    torch.testing.assert_close(scored, outputs["gen_rand_train"])
    torch.testing.assert_close(loss_map["reward_loss"], torch.tensor(0.25))
    torch.testing.assert_close(loss_map["total_loss"], loss_map["dsm_loss"])


@RunIf(min_gpus=1)
@pytest.mark.parametrize(
    ("model_cls", "config_cls"),
    [(CausalSFTModel, CausalModelConfig), (CausalTeacherForcingSFTModel, TeacherForcingModelConfig)],
)
def test_causal_sft_trains_on_a_frame_count_the_chunk_size_does_not_divide(model_cls, config_cls, tmp_path):
    """8 frames in chunks of 3 train a causal Wan with one time per mask chunk, [0, 5) and [5, 8); teacher forcing
    also draws its context-stream times per chunk."""
    from tests.test_network import _tiny_causal_wan_family_network

    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.config = config_cls()
    model.config.cond_dropout_prob = None
    model.config.sample_t_cfg.max_t = 0.999
    if model_cls is CausalTeacherForcingSFTModel:
        model.config.sample_t_context_cfg.enabled = True
    model.device = torch.device("cuda")
    model.precision = torch.float32
    model.use_ema = []
    model.input_shape = [4, 8, 4, 4]
    model.net = _tiny_causal_wan_family_network("causal_wan", tmp_path).to(model.device)
    model.net.chunk_size = 3
    forward, times = model.net.forward, []

    def record_times(x, t, **kwargs):
        times.append(t)
        return forward(x, t, **kwargs)

    model.net.forward = record_times
    data = {
        "real": torch.randn(2, *model.input_shape, device=model.device),
        "condition": torch.randn(2, 6, 8, device=model.device),
    }
    loss_map, _ = model.single_train_step(data, iteration=0)
    loss_map["total_loss"].backward()

    (t,) = times
    # Teacher forcing passes the context stream's times before the denoising stream's.
    for stream_t in t.split(8, dim=1):
        for start, end in _build_block_token_ranges(num_frames=8, frame_seqlen=1, chunk_size=3):
            assert (stream_t[:, start:end] == stream_t[:, start : start + 1]).all()
    assert torch.isfinite(loss_map["total_loss"])
    assert any(param.grad is not None for param in model.net.parameters())


def test_optimizers(get_model_data):
    """Test optimizer functionality."""
    model, data = get_model_data

    # Test that optimizer operations run without errors
    model.optimizers_zero_grad(0)
    loss_map, _ = model.single_train_step(data, 0)

    # Test that loss requires gradients and can be backpropagated
    assert loss_map["total_loss"].requires_grad
    model.grad_scaler.scale(loss_map["total_loss"]).backward()

    # Test that gradients exist after backward pass
    has_gradients = False
    for param in model.net.parameters():
        if param.requires_grad and param.grad is not None:
            has_gradients = True
            break
    assert has_gradients, "No gradients found after backward pass"

    # Test that optimizer step runs without errors
    model.optimizers_schedulers_step(0)


def test_loss_computation(get_model_data):
    """Test that loss computation produces reasonable values."""
    model, data = get_model_data

    loss_map, outputs = model.single_train_step(data, 0)

    # Test loss properties
    total_loss = loss_map["total_loss"]
    dsm_loss = loss_map["dsm_loss"]

    assert total_loss.requires_grad
    assert dsm_loss.requires_grad
    assert not torch.isnan(total_loss)
    assert not torch.isnan(dsm_loss)
    assert not torch.isinf(total_loss)
    assert not torch.isinf(dsm_loss)


# ---------------------------------------------------------------------------
# Teacher-forcing causal SFT tests
# ---------------------------------------------------------------------------


def test_teacher_forcing_context_match_train_t_default_off():
    assert TeacherForcingModelConfig().teacher_forcing_context_match_train_t is False


def test_teacher_forcing_context_aug_default_off():
    assert TeacherForcingModelConfig().sample_t_context_cfg.enabled is False


def test_sample_t_inhom_requires_positive_lengths():
    scheduler = RFNoiseSchedule()

    with pytest.raises(ValueError, match="seq_len must be > 0"):
        scheduler.sample_t_inhom(2, seq_len=0, chunk_size=4)
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        scheduler.sample_t_inhom(2, seq_len=3, chunk_size=0)


def test_sample_t_inhom_omitted_bounds_use_scheduler_bounds(monkeypatch):
    scheduler = RFNoiseSchedule(min_t=0.0, max_t=1.0)

    def zeros_rand(*size, device=None, dtype=None, **kwargs):
        del kwargs
        return torch.zeros(size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "rand", zeros_rand)

    t = scheduler.sample_t_inhom(2, seq_len=4, chunk_size=2, time_dist_type="uniform")

    assert t.shape == (2, 4)
    assert torch.equal(t, torch.full_like(t, scheduler.min_t))


def test_sample_t_direct_defaults_use_scheduler_bounds(monkeypatch):
    scheduler = RFNoiseSchedule(min_t=0.0, max_t=1.0)

    def zeros_rand(*size, device=None, dtype=None, **kwargs):
        del kwargs
        return torch.zeros(size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "rand", zeros_rand)

    t = scheduler.sample_t(2, time_dist_type="uniform")

    assert torch.equal(t, torch.full_like(t, scheduler.min_t))


@pytest.fixture(params=[(1, 5), (3, 6)], ids=["framewise", "chunkwise"])
def get_teacher_forcing_model_data(request):
    chunk_size, n_frames = request.param
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    set_env_vars()

    config = TeacherForcingModelConfig()
    config.net = copy.deepcopy(CausalWan_1_3B_Config)
    config.net.chunk_size = chunk_size
    config.net.total_num_frames = n_frames
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.precision = "float32"
    config.precision_model = "bfloat16"
    config.pretrained_model_path = None
    config.student_grid_cfg.steps = 4
    config.sample_t_cfg.max_t = 0.999
    config.sample_t_cfg.min_t = 0.001
    config.sample_t_cfg.time_dist_type = "uniform"
    config.input_shape = [16, n_frames, 8, 8]
    config.enable_preprocessors = False

    model = CausalTeacherForcingSFTModel(config)
    model.net.transformer.blocks = model.net.transformer.blocks[:1]
    model.on_train_begin()

    batch_size = 1
    channels, n_frames, height, width = config.input_shape
    # Forwards take conditioning in the network's compute dtype, as ``build_conditioning`` returns it.
    condition_dtype = model.net.precision_model
    data = {
        "real": torch.randn(batch_size, channels, n_frames, height, width, device=model.device, dtype=model.precision),
        "condition": torch.randn(batch_size, 512, 4096, device=model.device, dtype=condition_dtype),
        "neg_condition": torch.zeros(batch_size, 512, 4096, device=model.device, dtype=condition_dtype),
    }
    yield model, data, n_frames

    del data
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@RunIf(min_gpus=1)
def test_teacher_forcing_single_train_step_smoke(get_teacher_forcing_model_data):
    model, data, n_frames = get_teacher_forcing_model_data
    loss_map, outputs = model.single_train_step(data, 0)

    assert "total_loss" in loss_map
    assert "dsm_loss" in loss_map
    assert torch.isfinite(loss_map["total_loss"])
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert callable(outputs["gen_rand"])
    assert isinstance(outputs["input_rand"], torch.Tensor)
    assert isinstance(outputs["gen_rand_train"], torch.Tensor)
    assert outputs["gen_rand_train"].shape == data["real"].shape
    assert outputs["input_rand"].shape == data["real"].shape
    assert outputs["input_rand"].shape[2] == n_frames


@RunIf(min_gpus=1)
def test_teacher_forcing_context_match_train_t_packs_matching_context(get_teacher_forcing_model_data):
    model, data, _ = get_teacher_forcing_model_data
    model.config.teacher_forcing_context_match_train_t = True

    captured = {}
    real_forward = model.net.forward

    def spy(x, t, *args, **kwargs):
        assert "clean_x" not in kwargs
        assert "clean_t" not in kwargs
        captured["x"] = x.detach().clone()
        captured["t"] = t.detach().clone()
        return real_forward(x, t, *args, **kwargs)

    model.net.forward = spy
    try:
        model.single_train_step(data, 0)
    finally:
        model.net.forward = real_forward

    # The context stream sees exactly the noisy frames the denoising stream is trained on.
    num_frames = data["real"].shape[2]
    assert captured["x"].shape[2] == 2 * num_frames
    assert captured["t"].shape[1] == 2 * num_frames
    assert torch.equal(captured["x"][:, :, :num_frames], captured["x"][:, :, num_frames:])
    assert torch.equal(captured["t"][:, :num_frames], captured["t"][:, num_frames:])


# ---------------------------------------------------------------------------
# Multimodal SFT tests
# ---------------------------------------------------------------------------


@pytest.fixture
def get_multimodal_model_data():
    gc.collect()
    instance = ModelConfig()
    opts = [
        "-",
        "backbone.img_resolution=8",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "backbone.r_timestep=False",
    ]
    instance.net = override_config_with_opts(instance.net, opts)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None
    instance.input_shape = [3, 8, 8]

    instance.cond_dropout_prob = 0.1
    instance.cond_keys_no_dropout = []

    model = SFTModel(instance)
    model.net = MultimodalNet(model.net)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
    neg_condition = torch.zeros(batch_size, 10)

    video = torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision)
    audio = torch.randn(batch_size, 3, 4, 4).to(model.device, model.precision)
    real = {"video": video, "audio": audio}

    data = {
        "real": real,
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_condition.to(model.device, model.precision),
    }

    return model, data


def test_multimodal_single_train_step(get_multimodal_model_data):
    model, data = get_multimodal_model_data
    loss_map, outputs = model.single_train_step(data, 0)

    assert isinstance(loss_map, dict)
    assert "total_loss" in loss_map
    assert "video_dsm_loss" in loss_map
    assert "audio_dsm_loss" in loss_map

    expected_total = loss_map["video_dsm_loss"] + loss_map["audio_dsm_loss"]
    assert torch.allclose(loss_map["total_loss"], expected_total, atol=1e-5)


def test_multimodal_backward_pass(get_multimodal_model_data):
    model, data = get_multimodal_model_data
    model.optimizers_zero_grad(0)
    loss_map, _ = model.single_train_step(data, 0)

    assert loss_map["total_loss"].requires_grad
    model.grad_scaler.scale(loss_map["total_loss"]).backward()

    has_gradients = any(p.grad is not None for p in model.net.parameters() if p.requires_grad)
    assert has_gradients


def test_multimodal_outputs_structure(get_multimodal_model_data):
    model, data = get_multimodal_model_data
    _, outputs = model.single_train_step(data, 0)

    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert callable(outputs["gen_rand"])
    assert isinstance(outputs["input_rand"], dict)
    assert isinstance(outputs["gen_rand_train"], dict)
    assert outputs["gen_rand_train"].keys() == data["real"].keys()
    for key in data["real"]:
        assert outputs["gen_rand_train"][key].shape == data["real"][key].shape
