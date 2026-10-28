# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA helpers for FastGen networks.

``LoRAMixin`` uses a model's native PEFT adapter API when it has one and falls
back to PEFT's low-level in-place injection for ordinary ``torch.nn.Module``
backbones. It provides an adapter-aware ``requires_grad_`` and adapter toggling
(``adapters_disabled``). ``FastGenModel`` drives all of it via
``config.model.net_adapter``, which also selects the module to adapt and the
extra trainable modules. Networks may override ``default_lora_target_modules``
to supply the targets used when ``target_modules`` is unset. With no explicit
teacher, ``SharedBaseTeacher`` provides a parameter-less adapters-off view over
the same net (no second copy in memory).

Adapters train with DDP, but not with FSDP, ``model.use_ema``, or
``model.meta_init``. ``trainer.checkpointer.save_adapter_only=True`` saves only
the adapter and extra trainable modules; loading such a checkpoint keeps the
frozen base at its pretrained weights.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from fastgen.utils.basic_utils import attr_override, train_mode
import fastgen.utils.logging_utils as logger
from fastgen.utils.module import module_state_names, temporarily_disable_activation_checkpointing


# Sentinel for ``target_modules``: adapt every ``nn.Linear`` in the transformer
# except the separately-trained new modules (e.g. ``r_embedder``).
ALL_LINEAR = "all-linear"


def all_linear_target_modules(module: torch.nn.Module, exclude_prefixes: Iterable[str] = ()) -> List[str]:
    """Full dotted names of every ``nn.Linear`` in ``module`` that is not one of
    ``exclude_prefixes`` or below one (explicit full names, not suffixes, so
    PEFT matches exactly the intended set)."""
    exclude = tuple(exclude_prefixes)
    return [
        name
        for name, sub in module.named_modules()
        if isinstance(sub, torch.nn.Linear) and not any(name == p or name.startswith(f"{p}.") for p in exclude)
    ]


@contextlib.contextmanager
def scaled_lora_layers(module: torch.nn.Module, lora_scale: float):
    """Apply ``lora_scale`` to ``module``'s PEFT LoRA layers for the duration of the block.

    Unscaling happens in ``finally`` so an exception inside the forward cannot leave a
    stale scale on the module. Without the PEFT backend the scale cannot be applied at
    all, so a non-default one is warned about and ignored.
    """
    if not USE_PEFT_BACKEND:
        if lora_scale != 1.0:
            logger.warning(f"Ignoring LoRA scale {lora_scale}: scaling requires the PEFT backend.")
        yield
        return

    scale_lora_layers(module, lora_scale)
    try:
        yield
    finally:
        unscale_lora_layers(module, lora_scale)


class LoRAMixin:
    """Mixin adding LoRA support to a network whose ``lora_base_name`` attribute
    (default ``"transformer"``) identifies the module to adapt. An empty name
    selects the whole network. Networks without an injected adapter behave
    exactly as before.
    """

    _lora_enabled: bool = False

    # Attribute name of the module the adapter is injected into (set from the config).
    lora_base_name: str = "transformer"

    # Extra trainable module paths, relative to ``lora_base`` (set from the config).
    lora_extra_trainable_modules: Tuple[str, ...] = ()

    # Attribute overrides applied around every teacher call so the net behaves as the
    # original base model (set from ``net_adapter.base_model_overrides``).
    base_model_overrides: Optional[Dict[str, Any]] = None

    @property
    def lora_base(self) -> torch.nn.Module:
        return self if not self.lora_base_name else getattr(self, self.lora_base_name)

    def default_lora_target_modules(self) -> List[str]:
        """Per-network default LoRA target list (networks override)."""
        raise NotImplementedError(
            f"{type(self).__name__} defines no default LoRA target modules; "
            "set model.net_adapter.target_modules explicitly."
        )

    def _resolve_extra_modules(self, paths: Iterable[str]) -> List[torch.nn.Module]:
        modules = []
        for path in paths:
            try:
                modules.append(self.lora_base.get_submodule(path))
            except AttributeError:
                pass  # declared but not instantiated (e.g. no guidance_embedder)
        return modules

    def _unfreeze_extra_modules(self) -> int:
        """Make the extra trainable modules trainable; return how many are instantiated."""
        modules = self._resolve_extra_modules(self.lora_extra_trainable_modules)
        for module in modules:
            module.requires_grad_(True)
        return len(modules)

    def resolve_lora_target_modules(self, target_modules: Optional[str | List[str]]) -> str | List[str]:
        """Resolve ``target_modules``: None -> the network's defaults; "all-linear" ->
        every Linear except the extra trainable modules (only those, so the trainable
        set stays stable for optimizer-state compatibility); any other string is a PEFT
        full-match regex and passes through unchanged, as does a list."""
        if target_modules is None:
            return self.default_lora_target_modules()
        if target_modules == ALL_LINEAR:
            return all_linear_target_modules(self.lora_base, exclude_prefixes=self.lora_extra_trainable_modules)
        if isinstance(target_modules, str):
            return target_modules
        return list(target_modules)

    def init_lora(
        self,
        rank: int,
        alpha: Optional[int] = None,
        target_modules: Optional[str | List[str]] = None,
        dropout: float = 0.0,
        init_lora_weights: str = "gaussian",
    ) -> None:
        """Inject a LoRA adapter into ``lora_base``.

        Args:
            rank: LoRA rank (> 0).
            alpha: LoRA alpha (scaling). Defaults to ``rank`` when ``None``.
            target_modules: Module-name suffixes to adapt, ``"all-linear"``, a PEFT full-match
                regex string, or ``None`` for the network's defaults (see ``resolve_lora_target_modules``).
            dropout: LoRA dropout probability.
            init_lora_weights: PEFT weight init (``"gaussian"`` keeps the adapter
                an identity at init, since ``lora_B`` is zero-initialized).
        """
        if any(p.is_meta for p in self.lora_base.parameters()):
            # PEFT cannot inject adapters on meta tensors.
            raise RuntimeError("LoRA injection is not supported on a meta-initialized transformer.")

        from peft import LoraConfig  # local import: optional dependency

        if alpha is None:
            alpha = rank
        target_modules = self.resolve_lora_target_modules(target_modules)

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            init_lora_weights=init_lora_weights,
            target_modules=target_modules,
        )
        add_adapter = getattr(self.lora_base, "add_adapter", None)
        if callable(add_adapter):
            add_adapter(lora_config)
        else:
            # Plain PyTorch networks do not expose diffusers' model-level PEFT
            # helpers. The low-level API wraps matching modules in place while
            # preserving the network's type and methods.
            from peft import inject_adapter_in_model  # local import: optional dependency

            inject_adapter_in_model(lora_config, self.lora_base)
        # PEFT freezes every non-adapter parameter, but the extra modules train with the adapter.
        num_extra = self._unfreeze_extra_modules()
        self._lora_enabled = True
        targets = f"regex {target_modules!r}" if isinstance(target_modules, str) else f"{len(target_modules)} modules"
        logger.info(
            f"Injected LoRA adapter (rank={rank}, alpha={alpha}, dropout={dropout}) "
            f"into {type(self.lora_base).__name__} targeting {targets}; "
            f"{num_extra} extra modules are trainable."
        )

    @property
    def lora_enabled(self) -> bool:
        return getattr(self, "_lora_enabled", False)

    def requires_grad_(self, requires_grad: bool = True):
        """With an adapter, "trainable" means the LoRA params + the extra modules:
        enabling re-freezes the base, so generic flips (e.g. GAN phase toggles) stay safe."""
        if not (requires_grad and self.lora_enabled):
            return super().requires_grad_(requires_grad)

        super().requires_grad_(False)
        num_lora = 0
        for name, p in self.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
                num_lora += 1

        num_extra = self._unfreeze_extra_modules()
        logger.info(f"Froze base weights; {num_lora} LoRA tensors and {num_extra} extra modules are trainable.")
        return self

    def _is_adapter_tensor(self, name: str) -> bool:
        """Whether the qualified tensor ``name`` is a LoRA tensor or belongs to an extra trainable module.

        Structural selection -- ``requires_grad`` flips don't change it."""
        base_prefix = f"{self.lora_base_name}." if self.lora_base_name else ""
        prefixes = tuple(f"{base_prefix}{path}." for path in self.lora_extra_trainable_modules)
        return "lora_" in name or name.startswith(prefixes)

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict restricted to the adapter: LoRA tensors plus the extra modules' params and buffers."""
        assert self.lora_enabled, "adapter_state_dict requires an initialized LoRA adapter"
        return {name: t for name, t in self.state_dict().items() if self._is_adapter_tensor(name)}

    def _fixed_precision_overrides(self) -> Dict[str, torch.dtype]:
        """Return frozen base state that stays in ``precision_model``.

        With an adapter, only its tensors are optimized, so the ``_set_precision(precision_opt=...)``
        boundary casts the LoRA tensors and extra trainable modules to optimization precision while
        the base weights stay in model precision (no full-size upcast of the frozen backbone).
        Architecture-specific overrides take precedence over this broad base rule, and nested FastGen
        modules (e.g. the noise scheduler) keep their own precision policy.
        """
        overrides = super()._fixed_precision_overrides()
        if self.lora_enabled and self.precision_model is not None:
            for name in module_state_names(self, exclude=self._nearest_nested_modules()):
                if not self._is_adapter_tensor(name):
                    overrides.setdefault(name, self.precision_model)
        return overrides

    @contextlib.contextmanager
    def adapters_disabled(self):
        """Context manager that disables LoRA adapters within the block.

        Used to run the *teacher* (base model) forward through the same network.
        No-op when LoRA is not enabled; restores the prior adapter state on exit, so contexts nest.
        """
        if not self.lora_enabled:
            yield
            return
        # Both injection paths wrap the targets in PEFT tuner layers, which diffusers' own
        # ``disable_adapters``/``enable_adapters`` only toggle, forgetting the prior state.
        # Toggle the layers here and restore that state, so a nested or already-disabled
        # context does not re-enable the adapters on exit.
        from peft.tuners.tuners_utils import BaseTunerLayer  # local import: optional dependency

        layers = [module for module in self.lora_base.modules() if isinstance(module, BaseTunerLayer)]
        if not layers:
            raise RuntimeError("LoRA is enabled but no PEFT tuner layers were found.")
        disabled_states = [bool(layer.disable_adapters) for layer in layers]
        # PEFT's enable_adapters also toggles trainability; preserve phase-level freezes and partial adapters.
        grad_states = [(parameter, parameter.requires_grad) for parameter in self.lora_base.parameters()]
        for layer in layers:
            layer.enable_adapters(False)
        try:
            yield
        finally:
            for layer, was_disabled in zip(layers, disabled_states, strict=True):
                layer.enable_adapters(not was_disabled)
            for parameter, requires_grad in grad_states:
                parameter.requires_grad_(requires_grad)


class SharedBaseTeacher(torch.nn.Module):
    """Teacher view over an adapter student: the same network run as the base model.

    Holds the net by reference under ``base_net`` (never registered), so the view
    owns no parameters and stays out of checkpoints, FSDP dicts, and cast sweeps.
    ``forward`` and every callable reached through ``__getattr__`` run as the base
    model (adapters off, eval; both restored on exit) -- anything invoked on a
    teacher is by definition a base-model computation. Other lookups pass through.
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        # Bypass Module registration: the net's parameters must not become ours.
        object.__setattr__(self, "base_net", net)
        self._base_model_active = False

    @contextlib.contextmanager
    def _as_base_model(self):
        # A delegated guidance recipe calls back through this provider for each
        # branch. Keep adapters disabled until the outermost teacher call exits.
        if self._base_model_active:
            # The outer call resolved checkpointing for its own grad mode; this call may have enabled grad.
            with temporarily_disable_activation_checkpointing(self.base_net):
                yield
            return
        net = self.base_net
        self._base_model_active = True
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(train_mode(net, mode=False))
                stack.enter_context(net.adapters_disabled())
                # Student-only attribute defaults (e.g. an embedded-guidance scale)
                # must not steer base-model computations.
                for name, value in (net.base_model_overrides or {}).items():
                    stack.enter_context(attr_override(net, name, value))
                stack.enter_context(temporarily_disable_activation_checkpointing(net))
                yield
        finally:
            self._base_model_active = False

    def forward(self, *args, **kwargs):
        with self._as_base_model():
            return self.base_net(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        attr = getattr(self.base_net, name)
        # Sub-modules (callable too) pass through raw -- wrapping would hide their
        # attributes; only methods/functions run under the base-model context.
        if callable(attr) and not isinstance(attr, torch.nn.Module):

            @functools.wraps(attr)
            def as_base_model(*args, **kwargs):
                with self._as_base_model():
                    return attr(*args, **kwargs)

            return as_base_model
        return attr
