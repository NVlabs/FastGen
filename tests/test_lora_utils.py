# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``fastgen.networks.lora_utils`` (``LoRAMixin`` + helpers).

Uses a tiny ``FluxTransformer2DModel`` (1 joint + 1 single block, 16-dim) so the
PEFT adapter API is exercised end-to-end without loading pretrained weights. The
load-bearing contract is the "teacher = base model with adapters disabled" trick:
disabling adapters must restore the base output exactly, and re-enabling must
restore the adapted output.
"""

import gc

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

import fastgen.utils.module as module_state
from fastgen.networks.lora_utils import (
    ALL_LINEAR,
    LoRAMixin,
    SharedBaseTeacher,
    all_linear_target_modules,
)
from fastgen.networks.Wan.network import Wan
from fastgen.networks.network import FastGenNetwork


class _TinyLoRANet(LoRAMixin, torch.nn.Module):
    """Minimal stand-in for a diffusers-backed FastGen network (owns ``self.transformer``)."""

    def __init__(self):
        super().__init__()
        from diffusers import FluxTransformer2DModel

        self.transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            pooled_projection_dim=16,
            guidance_embeds=False,
            axes_dims_rope=(2, 2, 4),
        )

    def forward(self, x):
        # One adapted layer is enough to observe adapter on/off behavior.
        return self.transformer.transformer_blocks[0].attn.to_q(x)

    def sample(self, x):
        return self.forward(x)


class _TinyPlainLoRANet(LoRAMixin, torch.nn.Module):
    """Plain-PyTorch network exercising PEFT's low-level injection path."""

    lora_base_name = ""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.extra = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def lora_net():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    torch.manual_seed(0)
    net = _TinyLoRANet()
    net.init_lora(rank=4, alpha=4, target_modules=["to_q", "to_k", "to_v"])
    return net


def _adapted_layer(net):
    """A LoRA-adapted Linear we can call directly (no full-transformer plumbing)."""
    return net.transformer.transformer_blocks[0].attn.to_q


def _perturb_lora_b(net):
    """Make the adapter non-identity (lora_B is zero-initialized by PEFT)."""
    for name, p in net.transformer.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)


def test_init_lora_injects(lora_net):
    param_names = [n for n, _ in lora_net.transformer.named_parameters()]
    assert any("lora_A" in n for n in param_names) and any("lora_B" in n for n in param_names)
    assert lora_net.lora_enabled
    # PEFT freezes the base on injection; the owning model applies the final
    # trainable set via the adapter-aware requires_grad_.
    assert all(p.requires_grad == ("lora_" in n) for n, p in lora_net.transformer.named_parameters())


def test_adapters_disabled_is_noop_without_lora():
    net = _TinyLoRANet()
    assert not net.lora_enabled
    with net.adapters_disabled():  # no-op context must still work
        pass


def test_plain_pytorch_lora_injection_and_nested_disable():
    torch.manual_seed(0)
    net = _TinyPlainLoRANet()
    net.lora_extra_trainable_modules = ("extra",)
    net.init_lora(rank=2, alpha=2, target_modules=["linear"])
    net.requires_grad_(True)

    assert net.lora_base is net
    assert all(p.requires_grad == ("lora_" in name or name.startswith("extra.")) for name, p in net.named_parameters())

    x = torch.randn(2, 4)
    with net.adapters_disabled():
        base_out = net(x)
        with net.adapters_disabled():
            torch.testing.assert_close(net(x), base_out)

    for name, parameter in net.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.5)
    adapted_out = net(x)
    assert not torch.allclose(adapted_out, base_out)
    with net.adapters_disabled():
        torch.testing.assert_close(net(x), base_out)
    torch.testing.assert_close(net(x), adapted_out)

    adapter_state = net.adapter_state_dict()
    assert any("lora_A" in name for name in adapter_state)
    assert {"extra.weight", "extra.bias"} <= set(adapter_state)
    assert not any(name.startswith(".") for name in adapter_state)


def test_init_lora_raises_on_meta_device():
    with torch.device("meta"):
        net = _TinyLoRANet()
        with pytest.raises(RuntimeError, match="meta"):
            net.init_lora(rank=4, alpha=4, target_modules=["to_q"])


def test_requires_grad_is_adapter_aware(lora_net):
    lora_net.lora_extra_trainable_modules = ("proj_out", "not_a_module")
    extra_param_ids = {id(p) for p in lora_net.transformer.proj_out.parameters()}
    lora_net.requires_grad_(True)
    for name, p in lora_net.named_parameters():
        expected = ("lora_" in name) or (id(p) in extra_param_ids)
        assert p.requires_grad is expected, name
    # Disabling freezes everything; a net without an adapter keeps stock semantics.
    lora_net.requires_grad_(False)
    assert not any(p.requires_grad for p in lora_net.parameters())
    plain = _TinyLoRANet()
    plain.requires_grad_(True)
    assert all(p.requires_grad for p in plain.parameters())


def test_adapters_disabled_roundtrip(lora_net):
    layer = _adapted_layer(lora_net)
    x = torch.randn(2, 5, 16)

    with lora_net.adapters_disabled():
        base_out = layer(x)
    # PEFT init (gaussian A, zero B) keeps the adapter an identity at init.
    assert torch.allclose(layer(x), base_out, atol=1e-6)

    _perturb_lora_b(lora_net)
    adapted_out = layer(x)
    assert not torch.allclose(adapted_out, base_out, atol=1e-4)

    # The teacher trick: disabling restores the base exactly, re-enabling restores the student.
    with lora_net.adapters_disabled():
        assert torch.allclose(layer(x), base_out, atol=1e-6)
    assert torch.allclose(layer(x), adapted_out, atol=1e-6)


def test_adapters_disabled_restores_the_prior_state_when_nested_or_already_disabled(lora_net):
    layer = _adapted_layer(lora_net)
    x = torch.randn(2, 5, 16)
    _perturb_lora_b(lora_net)
    adapted_out = layer(x)

    with lora_net.adapters_disabled():
        base_out = layer(x)
        with lora_net.adapters_disabled():
            torch.testing.assert_close(layer(x), base_out)
        # The inner exit must not re-enable the adapter inside the outer context.
        torch.testing.assert_close(layer(x), base_out)
    torch.testing.assert_close(layer(x), adapted_out)

    # Entering while diffusers' own API already disabled the adapter leaves it disabled afterwards.
    lora_net.lora_base.disable_adapters()
    with lora_net.adapters_disabled():
        torch.testing.assert_close(layer(x), base_out)
    torch.testing.assert_close(layer(x), base_out)
    lora_net.lora_base.enable_adapters()
    torch.testing.assert_close(layer(x), adapted_out)


@pytest.mark.parametrize("all_frozen", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_adapters_disabled_preserves_parameter_trainability(lora_net, all_frozen, fail):
    for name, parameter in lora_net.named_parameters():
        parameter.requires_grad_(not all_frozen and "lora_B" in name)
    expected = {name: parameter.requires_grad for name, parameter in lora_net.named_parameters()}
    try:
        with lora_net.adapters_disabled():
            with lora_net.adapters_disabled():
                assert not any(parameter.requires_grad for parameter in lora_net.parameters())
            if fail:
                raise RuntimeError("forward failed")
    except RuntimeError as error:
        assert fail and str(error) == "forward failed"
    assert {name: parameter.requires_grad for name, parameter in lora_net.named_parameters()} == expected


def test_checkpointed_lora_recompute_keeps_adapter(lora_net):
    """Activation checkpointing recomputes at backward time: a teacher (adapters-off)
    pass between the student forward and backward must not corrupt the recompute."""
    _perturb_lora_b(lora_net)
    layer = _adapted_layer(lora_net)
    lora_params = [p for n, p in layer.named_parameters() if "lora_" in n]
    x = torch.randn(2, 5, 16)

    ref = layer(x)
    ref_grads = torch.autograd.grad(ref.square().sum(), lora_params)

    out = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
    assert torch.allclose(out, ref, atol=1e-6)
    with torch.no_grad():
        SharedBaseTeacher(lora_net)(x)  # teacher pass between forward and backward
    out.square().sum().backward()  # recompute fires here; adapters must be back on
    for p, g in zip(lora_params, ref_grads, strict=True):
        assert p.grad is not None and torch.allclose(p.grad, g, atol=1e-6)

    # Negative control: a recompute with adapters off is detected loudly (the
    # recomputed graph saves fewer tensors), not silently corrupted.
    out2 = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
    with lora_net.adapters_disabled(), pytest.raises(torch.utils.checkpoint.CheckpointError):
        out2.square().sum().backward()


def test_all_linear_target_modules_excludes_prefixes():
    net = _TinyLoRANet()
    all_linears = all_linear_target_modules(net.transformer)
    assert all_linears and all(isinstance(n, str) for n in all_linears)
    excluded = all_linear_target_modules(net.transformer, exclude_prefixes=["time_text_embed"])
    assert set(excluded) < set(all_linears)
    assert not any(n.startswith("time_text_embed") for n in excluded)


def _with_extras() -> _TinyLoRANet:
    """Extras set as ``_init_net_adapter`` does; one path uninstantiated (skip path)."""
    net = _TinyLoRANet()
    net.lora_extra_trainable_modules = ("proj_out", "not_instantiated")
    return net


def test_resolve_lora_target_modules():
    net = _with_extras()
    # "all-linear" adapts everything except the declared extra trainable modules.
    resolved = net.resolve_lora_target_modules(ALL_LINEAR)
    assert resolved and not any(n == "proj_out" or n.startswith("proj_out.") for n in resolved)
    # Explicit lists pass through; None requires a network default.
    assert net.resolve_lora_target_modules(["to_q"]) == ["to_q"]
    with pytest.raises(NotImplementedError):
        _TinyLoRANet().resolve_lora_target_modules(None)


def test_adapter_state_dict_membership():
    """LoRA + extra-module tensors (buffers too); invariant to requires_grad flips."""
    net = _with_extras()
    net.init_lora(rank=4, alpha=4, target_modules=["to_q", "to_k", "to_v"])
    net.transformer.proj_out.register_buffer("stat", torch.ones(3))

    state = net.adapter_state_dict()
    assert any("lora_A" in k for k in state) and any("lora_B" in k for k in state)
    assert "transformer.proj_out.stat" in state  # extra-module buffers are kept
    assert all("lora_" in k or k.startswith("transformer.proj_out.") for k in state)
    assert not any("base_layer" in k for k in state)  # frozen base stays out

    # Structural selection: flipping requires_grad must not change the saved set.
    for p in net.parameters():
        p.requires_grad_(False)
    assert net.adapter_state_dict().keys() == state.keys()


def test_permissive_adapter_load_logs_only_adapter_incompatibilities(monkeypatch):
    def make_net():
        net = _TinyPlainLoRANet()
        net.lora_extra_trainable_modules = ("extra",)
        net.init_lora(rank=2, alpha=2, target_modules=["linear"])
        return net

    source = make_net()
    target = make_net()
    state = dict(source.adapter_state_dict())
    missing_name = "extra.weight"
    unexpected_name = "extra.unexpected"
    state.pop(missing_name)
    state[unexpected_name] = torch.ones(1)

    warnings = []
    monkeypatch.setattr(module_state.logger, "warning", warnings.append)
    load_info = module_state.load_module_state_dict(
        target,
        state,
        strict=False,
        require_any=True,
        required_keys=target.adapter_state_dict().keys(),
    )

    adapter_names = set(target.adapter_state_dict())
    base_names = set(target.state_dict()) - adapter_names
    assert base_names <= set(load_info.missing_keys)
    assert missing_name in load_info.missing_keys
    assert load_info.unexpected_keys == [unexpected_name]
    assert len(warnings) == 1
    assert f"missing_keys=['{missing_name}']" in warnings[0]
    assert f"unexpected_keys=['{unexpected_name}']" in warnings[0]
    assert not any(name in warnings[0] for name in base_names)


def test_shared_base_teacher_is_parameterless_view(lora_net):
    teacher = SharedBaseTeacher(lora_net)
    # Held by reference, never registered: no parameters, empty state_dict, so it
    # stays out of checkpoints / FSDP dicts / cast sweeps.
    assert len(list(teacher.parameters())) == 0
    assert teacher.state_dict() == {}
    # Attribute reads pass through to the net.
    assert teacher.lora_enabled is True
    assert teacher.transformer is lora_net.transformer


def test_shared_base_teacher_to_is_noop_on_base(lora_net):
    """``.to`` must not touch the shared base: offload paths move ``module_dict`` roots with ``.to(...)``;
    the view must stay inert even if one is applied to it."""
    teacher = SharedBaseTeacher(lora_net)
    assert teacher.to(torch.float64) is teacher
    assert all(p.dtype == torch.float32 for p in lora_net.parameters())


def test_shared_base_teacher_forward_disables_adapters(lora_net):
    teacher = SharedBaseTeacher(lora_net)
    x = torch.randn(2, 5, 16)
    with lora_net.adapters_disabled():
        base_out = lora_net(x)

    _perturb_lora_b(lora_net)
    adapted_out = lora_net(x)
    assert not torch.allclose(adapted_out, base_out, atol=1e-4)

    # The view runs the base model bitwise and re-enables the adapter afterwards.
    assert torch.equal(teacher(x), base_out)
    assert torch.equal(lora_net(x), adapted_out)
    # Wrapped prediction helpers run adapters-off too.
    assert torch.equal(teacher.sample(x), base_out)
    assert torch.equal(lora_net.sample(x), adapted_out)


@pytest.mark.parametrize("fail", [False, True])
def test_shared_base_teacher_nested_calls_restore_only_after_outer_exit(lora_net, monkeypatch, fail):
    teacher = SharedBaseTeacher(lora_net)
    x = torch.randn(2, 5, 16)
    with lora_net.adapters_disabled():
        base_out = lora_net(x)
    _perturb_lora_b(lora_net)
    adapted_out = lora_net(x)

    def sample(x):
        first = teacher(x)  # Re-enter the provider from a delegated recipe.
        # The outer recipe must still see the base model after this nested call.
        torch.testing.assert_close(lora_net(x), base_out, rtol=0, atol=0)
        if fail:
            raise RuntimeError("recipe failed")
        return first + teacher(x)

    monkeypatch.setattr(lora_net, "sample", sample)
    if fail:
        with pytest.raises(RuntimeError, match="recipe failed"):
            teacher.sample(x)
    else:
        torch.testing.assert_close(teacher.sample(x), 2 * base_out, rtol=0, atol=0)
    torch.testing.assert_close(lora_net(x), adapted_out, rtol=0, atol=0)
    assert lora_net.training
    # An exception must also leave the provider ready for another teacher call.
    torch.testing.assert_close(teacher(x), base_out, rtol=0, atol=0)


def test_lora_network_keeps_base_weights_in_model_precision():
    """Only the adapter (LoRA tensors + extra modules) is optimized; the frozen base stays in model precision."""

    class _LoRANetwork(LoRAMixin, FastGenNetwork):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.base = torch.nn.Linear(2, 2, dtype=torch.float64)
            self.transformer.lora_A = torch.nn.Linear(2, 2, dtype=torch.float64)
            self.transformer.extra = torch.nn.Linear(2, 2, dtype=torch.float64)
            self.lora_base_name = "transformer"
            self.lora_extra_trainable_modules = ("extra",)

        def forward(self, x_t, t, **kwargs):
            del t, kwargs
            return x_t

    network = _LoRANetwork()
    network._lora_enabled = True

    network._set_precision(precision_model=torch.bfloat16, precision_opt=torch.float32)

    assert network.transformer.base.weight.dtype == torch.bfloat16
    assert network.transformer.lora_A.weight.dtype == torch.float32
    assert network.transformer.extra.weight.dtype == torch.float32

    # Plain ``to`` is a native uniform cast; re-applying the policy restores the split.
    network.to(dtype=torch.float64)
    assert network.transformer.base.weight.dtype == torch.float64
    assert network.transformer.lora_A.weight.dtype == torch.float64

    network._set_precision(precision_model=torch.bfloat16, precision_opt=torch.float32)
    assert network.transformer.base.weight.dtype == torch.bfloat16
    assert network.transformer.lora_A.weight.dtype == torch.float32


def test_init_lora_keeps_extra_modules_trainable(monkeypatch):
    """PEFT freezes the extra modules on injection; they train with the adapter, so trainable
    preparation does not report them as newly trainable."""

    class _PlainLoRANetwork(LoRAMixin, FastGenNetwork):
        lora_base_name = ""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)
            self.extra = torch.nn.Linear(4, 4)

        def forward(self, x_t, t, **kwargs):
            del t, kwargs
            return self.linear(x_t)

    network = _PlainLoRANetwork()
    network.lora_extra_trainable_modules = ("extra",)
    network.init_lora(rank=2, alpha=2, target_modules=["linear"])
    expected = {name: "lora_" in name or name.startswith("extra.") for name, _ in network.named_parameters()}
    assert {name: p.requires_grad for name, p in network.named_parameters()} == expected

    warnings = []
    monkeypatch.setattr(module_state.logger, "warning", warnings.append)
    network._set_train_mode(True)

    assert warnings == []
    assert {name: p.requires_grad for name, p in network.named_parameters()} == expected


class _CheckpointedLoRANet(LoRAMixin, torch.nn.Module):
    """Plain network whose block checkpoints diffusers-style: a flag read at forward time."""

    lora_base_name = ""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.gradient_checkpointing = False

    def forward(self, x):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.linear, x, use_reentrant=False)
        return self.linear(x)


def _checkpointed_lora_net():
    torch.manual_seed(0)
    net = _CheckpointedLoRANet()
    net.init_lora(rank=2, alpha=2, target_modules=["linear"])
    for name, parameter in net.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.5)  # make the adapter non-identity
    return net


def test_shared_base_teacher_keeps_differentiable_forward_out_of_checkpoint_recompute():
    """Checkpointing is off during the teacher forward, so backward never recomputes it as the student."""
    net = _checkpointed_lora_net()
    net.gradient_checkpointing = True
    x = torch.randn(3, 4, requires_grad=True)

    out = SharedBaseTeacher(net)(x)
    assert net.gradient_checkpointing is True  # restored after the call
    (grad,) = torch.autograd.grad(out.square().sum(), x)

    net.gradient_checkpointing = False
    with net.adapters_disabled():
        ref = net(x)
    (ref_grad,) = torch.autograd.grad(ref.square().sum(), x)
    assert torch.allclose(out, ref, atol=1e-6) and torch.allclose(grad, ref_grad, atol=1e-6)
    assert not torch.allclose(net(x), ref, atol=1e-6)  # the student still runs with its adapter


def test_shared_base_teacher_refuses_differentiable_forward_through_checkpoint_wrappers():
    net = _checkpointed_lora_net()
    net.linear = checkpoint_wrapper(net.linear)
    teacher = SharedBaseTeacher(net)
    x = torch.randn(3, 4, requires_grad=True)

    with pytest.raises(RuntimeError, match="activation-checkpoint wrappers"):
        teacher(x)
    with torch.no_grad():
        assert teacher(x).shape == (3, 4)  # nothing is recomputed without gradients


def _delegate_one_grad_branch(net, teacher):
    """Install a delegated recipe on ``net`` whose branch enables grad inside the caller's teacher call."""

    def sample(x):
        with torch.enable_grad():
            return teacher(x)

    net.sample = sample


def test_shared_base_teacher_nested_grad_call_keeps_forward_out_of_checkpoint_recompute():
    """A no-grad outer call leaves checkpointing on, so a nested call that enables grad must turn it off."""
    net = _checkpointed_lora_net()
    net.gradient_checkpointing = True
    teacher = SharedBaseTeacher(net)
    _delegate_one_grad_branch(net, teacher)
    flags_at_forward = []
    net.register_forward_pre_hook(lambda module, args: flags_at_forward.append(module.gradient_checkpointing))
    x = torch.randn(3, 4, requires_grad=True)

    with torch.no_grad():
        out = teacher.sample(x)
    assert flags_at_forward == [False]
    assert net.gradient_checkpointing is True  # restored after the call
    (grad,) = torch.autograd.grad(out.square().sum(), x)

    net.gradient_checkpointing = False
    with net.adapters_disabled():
        ref = net(x)
    (ref_grad,) = torch.autograd.grad(ref.square().sum(), x)
    assert torch.allclose(out, ref, atol=1e-6) and torch.allclose(grad, ref_grad, atol=1e-6)


def test_shared_base_teacher_nested_grad_call_refuses_checkpoint_wrappers():
    net = _checkpointed_lora_net()
    net.linear = checkpoint_wrapper(net.linear)
    teacher = SharedBaseTeacher(net)
    _delegate_one_grad_branch(net, teacher)

    with torch.no_grad(), pytest.raises(RuntimeError, match="activation-checkpoint wrappers"):
        teacher.sample(torch.randn(3, 4, requires_grad=True))


def test_wan_declares_lora_capability():
    assert issubclass(Wan, LoRAMixin)
