# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local operations on module registrations, runtime state, and placement.

Names follow raw parameter/buffer registrations (``prefix + name`` over the module tree), not
state-dict hooks that may rewrite wrapper prefixes. The snapshot also keeps non-persistent buffers,
``None`` slots, aliases, persistence, and trainability, which meta materialization and cross-rank
structure checks need. Names have no DCP-style stripping of ``_orig_mod`` or checkpoint-wrapper
prefixes, so name-keyed policies must run before wrapping or within one tree.

Unlike plain PyTorch, FastGen does not support:

* buffers with ``requires_grad=True`` (register trainable state as parameters) or modules with
  ``extra_state``;
* complex parameters (register complex constants as buffers or parameterize complex weights with
  real tensors);
* a tensor registered as both a parameter and a buffer, or in several buffer slots, except runtime
  buffer ties such as RoPE cache aliases when saving or loading into materialized modules;
* modules or tensors shared between independently managed roots (ties and shared submodules within
  one root are fine);
* ``state_dict()`` exports that omit a parameter or persistent buffer;
* assignment loading or non-default ``torch.__future__`` module-conversion settings.

The checks compare module owners, registration slots, tensor identities, and metadata, never values,
so hooks must uphold the value contracts themselves. Export hooks leave live values unchanged, since
exports share their storage. During meta materialization, ``reset_parameters()`` rebuilds only
deterministic state within the root being materialized. Hooks can run on only some ranks, e.g. rank-0
exports and meta-rank resets, so they must not issue collectives.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Collection, Iterable, Iterator, Literal, Mapping, NamedTuple
import warnings

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

import fastgen.utils.logging_utils as logger
from fastgen.utils.tensor import resolve_device


TensorKind = Literal["parameter", "buffer"]
RegistrationKey = tuple[TensorKind, str]


def load_module_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    strict: bool = True,
    require_any: bool = False,
    description: str | None = None,
    required_keys: Iterable[str] | None = None,
    check_state_preservation: bool = True,
) -> Any:
    """Copy values into existing module state, validate optional coverage, and log the result.

    Loading preserves the target's tensor objects, devices, and dtypes. Assignment loading is
    deliberately unsupported because replacing registrations can split tied parameters.

    ``require_any`` guards permissive loads (``strict=False`` or ``required_keys``): at least one
    registered parameter must not be reported missing by the module's loader. Strict loads already
    require every key. Loaders may synthesize buffers (e.g. BatchNorm's ``num_batches_tracked``), so
    buffers never count; load modules without parameters strictly.
    Target names include aliases and retain raw wrapper prefixes, matching native ``missing_keys`` after
    source unwrapping or load-hook key conversion; exported ``state_dict`` names may strip those
    prefixes. Nonpersistent buffers do not count toward coverage. Non-empty
    ``required_keys``, in the same target names, relax ``strict`` for a partial state dict such as an
    adapter without its frozen base. In strict mode those entries must load, and loading any optional
    parameter requires the whole optional state to load; optional buffers alone do not trigger this
    rule. Both checks read the loader's own report, so load hooks that rename or drop keys are judged by
    their result.
    Targets must be materialized first. Native loading and post-load validation are not transactional:
    after a failure, discard or restore the target before using it.

    Set ``check_state_preservation=False`` only when the caller already guards structural preservation,
    such as with a snapshot of the complete model collection. Request and coverage checks always run.
    """
    description = description or type(module).__name__
    required_keys = None if required_keys is None else set(required_keys)  # The keys are read several times.
    guard = validate_module_state_preserved(module, description) if check_state_preservation else nullcontext()
    with guard:
        _validate_load_request(module, state_dict, required_keys, description)
        state_with_vars = module.state_dict(keep_vars=True)
        validate_state_dict_export(module, state_with_vars)
        native_strict = strict and required_keys is None
        load_info = module.load_state_dict(state_dict, strict=native_strict)
        missing_keys, not_loaded, coverage = load_info.missing_keys, [], ""
        if required_keys is not None:
            missing_keys, not_loaded, coverage = _partial_load_coverage(module, load_info, required_keys)
        info = f"missing_keys={missing_keys}, unexpected_keys={load_info.unexpected_keys}{coverage}"
        if required_keys is not None and strict and (missing_keys or load_info.unexpected_keys or not_loaded):
            raise RuntimeError(f"{description}: the state dict does not match, {info}.")
        # Native strict loading already required every key, including those of parameter-free modules.
        if require_any and not native_strict:
            parameter_names = set(module_state_names(module, persistent_buffers=False, nonpersistent_buffers=False))
            if not parameter_names.difference(load_info.missing_keys):
                raise RuntimeError(
                    f"State-dict load matched no target parameters for {type(module).__name__} "
                    f"(target parameters={len(parameter_names)}, unexpected keys={len(load_info.unexpected_keys)})."
                )
        log = logger.warning if missing_keys or load_info.unexpected_keys or not_loaded else logger.success
        log(f"{description}, loading info: {info}")
        return load_info


def _validate_load_request(
    module: nn.Module, state_dict: Mapping[str, Any], required_keys: set[str] | None, description: str
) -> None:
    """Check a copy-load request before any value changes."""
    validate_torch_state_settings()
    reject_extra_state(module, description)
    if has_meta_state(module):
        raise RuntimeError(f"{description}: copy loading requires materialized state; materialize meta tensors first.")
    if not getattr(module, "lora_enabled", False):
        # A permissive load would otherwise drop the adapter and keep the unmerged base weights.
        adapter_keys = [name for name in state_dict if "lora_" in name]
        if adapter_keys:
            raise RuntimeError(
                f"{description}: the state dict holds {len(adapter_keys)} LoRA adapter tensors (first 8: "
                f"{adapter_keys[:8]}) but the target has no initialized adapter; merge the adapter into the "
                "base weights or initialize it before loading."
            )
    if required_keys is None:
        return
    if not required_keys:
        # An empty contract would silently turn a strict load into an unchecked one.
        raise ValueError(f"{description}: required_keys must name at least one target key; pass None instead.")
    unknown_keys = required_keys.difference(module_state_names(module, nonpersistent_buffers=False))
    if unknown_keys:
        raise ValueError(f"{description}: unknown required target keys: {sorted(unknown_keys)[:8]}")


def _partial_load_coverage(module: nn.Module, load_info: Any, required_keys: set[str]) -> tuple[list, list, str]:
    """Return the missing required keys, the unloaded optional keys, and a coverage note for the log.

    Loading any optional parameter requires the whole optional state; buffers alone do not imply a base load.
    """
    optional_keys = set(module_state_names(module, nonpersistent_buffers=False)) - required_keys
    parameter_names = set(module_state_names(module, persistent_buffers=False, nonpersistent_buffers=False))
    not_loaded, coverage = [], ""
    if not optional_keys.intersection(parameter_names).issubset(load_info.missing_keys):
        loaded = sorted(optional_keys.difference(load_info.missing_keys))
        not_loaded = sorted(optional_keys.intersection(load_info.missing_keys))
        coverage = f", {len(loaded)} of {len(optional_keys)} non-required tensors loaded (first 8: {loaded[:8]})"
        if not_loaded:
            coverage += f", {len(not_loaded)} not loaded (first 8: {not_loaded[:8]})"
    return sorted(required_keys.intersection(load_info.missing_keys)), not_loaded, coverage


class TensorRegistration(NamedTuple):
    """One qualified parameter or buffer registration slot."""

    owner: nn.Module
    local_name: str
    tensor: torch.Tensor | None
    persistent: bool | None


class _TensorMetadata(NamedTuple):
    """Serializable metadata for one registered tensor path."""

    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    layout: str | None = None
    requires_grad: bool | None = None
    persistent: bool | None = None
    tensor_type: str | None = None
    placements: tuple[str, ...] | None = None
    aliases: tuple[RegistrationKey, ...] | None = None
    device_type: str | None = None


@contextmanager
def temporarily_disable_activation_checkpointing(module: nn.Module):
    """Temporarily disable flag-gated activation checkpointing for a module tree.

    Diffusers-style checkpointing reads ``gradient_checkpointing`` during the forward, so its
    enabled flags can be cleared and restored exactly. ``CheckpointWrapper`` modules checkpoint
    independently of those flags and cannot be toggled per forward. With autograd disabled there
    can be no backward recomputation, so this context is a no-op.
    """
    if not torch.is_grad_enabled():
        yield
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    if any(isinstance(submodule, CheckpointWrapper) for submodule in module.modules()):
        raise RuntimeError(
            f"{type(module).__name__} contains activation-checkpoint wrappers, which cannot be disabled per forward."
        )

    checkpointing_states = [
        (submodule, submodule.gradient_checkpointing)
        for submodule in module.modules()
        if bool(getattr(submodule, "gradient_checkpointing", False))
    ]
    for submodule, _state in checkpointing_states:
        submodule.gradient_checkpointing = False
    try:
        yield
    finally:
        for submodule, state in checkpointing_states:
            submodule.gradient_checkpointing = state


def attention_forward_without_cudnn(attention: nn.Module, *args, **kwargs):
    """Instance-bound ``forward`` of an attention module whose SDPA stays off the cuDNN backend.

    cuDNN's BF16/FP16 SDPA backward returns non-finite gradients on some attention that Flash and mem-efficient
    attention compute finitely on the same inputs, such as Wan2.2-I2V-A14B's low-noise block 33 and CogVideoX-2B.
    Only cuDNN is removed from the enabled backends, so an enclosing restriction, such as a math-only JVP region,
    still applies. Empty (batch-0) inputs run unrestricted, since Flash and mem-efficient reject batch size 0.
    Install it with ``types.MethodType`` so that ``copy.deepcopy`` rebinds it to the copied module; it runs inside
    the module, so activation checkpointing recomputes the attention on the same backend.
    """
    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    forward = type(attention).forward
    backend = torch.backends.cuda
    enabled = [
        sdp
        for sdp, is_enabled in (
            (SDPBackend.FLASH_ATTENTION, backend.flash_sdp_enabled()),
            (SDPBackend.EFFICIENT_ATTENTION, backend.mem_efficient_sdp_enabled()),
            (SDPBackend.MATH, backend.math_sdp_enabled()),
        )
        if is_enabled
    ]
    if not enabled or (torch.is_tensor(hidden_states) and hidden_states.shape[0] == 0):
        return forward(attention, *args, **kwargs)
    with sdpa_kernel(enabled):
        return forward(attention, *args, **kwargs)


def tensor_registrations(module: nn.Module) -> dict[RegistrationKey, TensorRegistration]:
    """Snapshot raw tensor registrations by ``(kind, qualified_name)``."""
    # Standard module iteration omits None registrations and cannot expose buffer persistence. Raw
    # registries preserve both facts. Repeated module paths are retained for structural signatures;
    # owner identity and local name identify the same physical slot.
    registrations: dict[RegistrationKey, TensorRegistration] = {}
    for module_name, owner in module.named_modules(remove_duplicate=False):
        prefix = f"{module_name}." if module_name else ""
        registries: tuple[tuple[TensorKind, Mapping[str, torch.Tensor | None]], ...] = (
            ("parameter", owner._parameters),
            ("buffer", owner._buffers),
        )
        for kind, slots in registries:
            for local_name, tensor in slots.items():
                persistent = local_name not in owner._non_persistent_buffers_set if kind == "buffer" else None
                registrations[(kind, f"{prefix}{local_name}")] = TensorRegistration(
                    owner=owner,
                    local_name=local_name,
                    tensor=tensor,
                    persistent=persistent,
                )
    return registrations


def _validate_module_owners(module: nn.Module, before: Mapping[str, nn.Module], operation: str) -> None:
    """Require every module path to retain its captured owner."""
    after = dict(module.named_modules(remove_duplicate=False))
    if before.keys() != after.keys() or any(owner is not after[name] for name, owner in before.items()):
        raise RuntimeError(f"{operation} changed registered state: module owners must be preserved.")


@contextmanager
def validate_module_owners_preserved(module: nn.Module, operation: str):
    """Require every module path, including stateless and shared descendants, to retain its owner."""
    before = dict(module.named_modules(remove_duplicate=False))
    yield
    _validate_module_owners(module, before, operation)


@dataclass(frozen=True)
class ModuleStateSnapshot:
    """Reusable baseline of module owners, slots, tensor identities, and metadata.

    Keep references as well as metadata: a hook may replace an owner or tensor without changing
    its shape or dtype. Tensor contents are neither snapshotted nor compared, so in-place value
    mutations are not detected. Copy loading intentionally changes values; export hooks must keep
    live values unchanged, and materialization hooks must not change another root's values.
    This checks a structural postcondition; it does not roll back mutations after a failure.
    Capture an entire ModuleDict to cover hooks that mutate another model entry. Repeated
    validation always compares against the original baseline, without capturing it again.
    """

    _module: nn.Module
    _owners: dict[str, nn.Module]
    _registrations: dict[RegistrationKey, TensorRegistration]
    _signature: dict[RegistrationKey, _TensorMetadata]
    _devices: dict[RegistrationKey, torch.device]
    _operation: str

    @classmethod
    def capture(cls, module: nn.Module, operation: str = "module state operation") -> "ModuleStateSnapshot":
        reject_unsupported_state_aliases(module, operation, allow_buffer_aliases=True)
        registrations = tensor_registrations(module)
        return cls(
            module,
            dict(module.named_modules(remove_duplicate=False)),
            registrations,
            registration_signature(registrations, include_device=True),
            {key: entry.tensor.device for key, entry in registrations.items() if entry.tensor is not None},
            operation,
        )

    def validate(self, operation: str | None = None) -> None:
        """Check the captured collection at an operation boundary, without refreshing the baseline."""
        operation = operation or self._operation
        # Capture rejects unsupported aliases; preserving slots and tensor identities preserves that invariant.
        _validate_module_owners(self._module, self._owners, operation)
        after = tensor_registrations(self._module)
        # Equal signatures imply equal registration names. Owner preservation above covers physical slots.
        if self._signature != registration_signature(after, include_device=True) or any(
            before.tensor is not after[key].tensor
            or (before.tensor is not None and after[key].tensor.device != self._devices[key])
            for key, before in self._registrations.items()
        ):
            raise RuntimeError(
                f"{operation} changed registered state; physical slots, tensor identities, and metadata "
                "must be preserved."
            )

    @contextmanager
    def validate_after(self, operation: str | None = None):
        """Validate a completed local phase against this baseline, without recapturing it."""
        yield
        self.validate(operation)


@contextmanager
def validate_module_state_preserved(module: nn.Module, operation: str):
    """Check one operation against a fresh structural snapshot; failures do not roll back mutations."""
    with ModuleStateSnapshot.capture(module, operation).validate_after():
        yield


def module_state_tensors(
    module: nn.Module,
    parameters: bool = True,
    persistent_buffers: bool = True,
    nonpersistent_buffers: bool = True,
    names: Collection[str] | None = None,
    remove_duplicate: bool = True,
    exclude: Iterable[nn.Module] = (),
) -> Iterator[tuple[TensorKind, str, torch.Tensor]]:
    """Yield selected non-``None`` module state in canonical parameter-then-buffer order.

    The flags select by kind; ``names`` additionally restricts the selection to those qualified
    names. ``exclude`` drops every tensor that the given submodules' trees hold, under all of its
    names, so state shared with an excluded submodule belongs to that submodule. Selection happens
    per registration before duplicate removal, so a shared tensor is still selected when only one of
    its aliases is enabled or named. When duplicate removal is enabled, tensor identity is unique
    within each kind; an object registered as both a parameter and a buffer is yielded once for each
    kind, matching PyTorch's separate named iterators.
    """
    excluded = {id(tensor) for submodule in exclude for tensor in (*submodule.parameters(), *submodule.buffers())}
    registrations = sorted(
        tensor_registrations(module).items(),
        key=lambda item: (item[0][0] == "buffer", item[0][1]),
    )
    seen: set[tuple[TensorKind, int]] = set()
    for (kind, qualified_name), registration in registrations:
        tensor = registration.tensor
        if kind == "parameter":
            selected = parameters
        elif registration.persistent:
            selected = persistent_buffers
        else:
            selected = nonpersistent_buffers
        if not selected or tensor is None or id(tensor) in excluded:
            continue
        if names is not None and qualified_name not in names:
            continue
        if remove_duplicate:
            identity = kind, id(tensor)
            if identity in seen:
                continue
            seen.add(identity)
        yield kind, qualified_name, tensor


def module_state_names(
    module: nn.Module,
    parameters: bool = True,
    persistent_buffers: bool = True,
    nonpersistent_buffers: bool = True,
    exclude: Iterable[nn.Module] = (),
) -> Iterator[str]:
    """Yield every selected non-``None`` parameter and buffer name, including aliases."""
    for _kind, name, _tensor in module_state_tensors(
        module,
        parameters=parameters,
        persistent_buffers=persistent_buffers,
        nonpersistent_buffers=nonpersistent_buffers,
        remove_duplicate=False,
        exclude=exclude,
    ):
        yield name


def validate_state_dict_export(
    module: nn.Module,
    state_with_vars: Mapping[str, Any],
    state_dict: Mapping[str, Any] | None = None,
    *,
    key_mapping: Callable[[str], Iterable[str]] | None = None,
) -> None:
    """Check captured exports for coverage of every parameter and persistent buffer, without running hooks.

    ``state_with_vars`` is the caller's ``state_dict(keep_vars=True)`` export and must retain registered
    tensor identities. Any exported alias or wrapper-rewritten key counts; nonpersistent buffers
    and explicit adapter-only exports are exempt.
    When checking an actual export, count only identities whose keys occur in that payload;
    ``key_mapping`` applies native DCP key canonicalization when needed. This catches hooks that
    omit state only when ``keep_vars=False``. Values remain the export hook's responsibility.
    Callers must capture both exports within their state-preservation validation.
    """
    keys = set(state_dict) if state_dict is not None else None
    exported = {
        id(tensor)
        for name, tensor in state_with_vars.items()
        if keys is None or not keys.isdisjoint(key_mapping(name) if key_mapping is not None else (name,))
    }
    hidden = [
        name
        for _kind, name, tensor in module_state_tensors(module, nonpersistent_buffers=False, remove_duplicate=False)
        if id(tensor) not in exported
    ]
    if hidden:
        raise RuntimeError(
            f"state_dict() does not export {len(hidden)} registered tensor(s) of {type(module).__name__}; "
            f"first 8: {hidden[:8]}. Ordinary exports must expose every parameter and persistent buffer."
        )


def validate_torch_state_settings() -> None:
    """Require PyTorch's default conversion and loading semantics before mutating state.

    Overwrite mode can split tied parameters. Swap mode dispatches loading through ``module_load``
    instead of ``copy_`` and is outside FastGen's copy-only loading contract. Check at operation
    boundaries because these process-global settings may change after import or preparation.
    """
    assert (
        not torch.__future__.get_overwrite_module_params_on_conversion()
    ), "FastGen requires torch.__future__.set_overwrite_module_params_on_conversion(False)."
    assert (
        not torch.__future__.get_swap_module_params_on_conversion()
    ), "FastGen requires torch.__future__.set_swap_module_params_on_conversion(False)."


def validate_module_roots(modules: Mapping[str, nn.Module], operation: str = "module registration") -> None:
    """Require independently managed roots to own distinct modules and registered tensor objects.

    Tied parameters and shared submodules within one root remain valid. Across roots, even
    stateless descendants and nonpersistent buffers must be independent: training mode, placement,
    sharding, and checkpointing are managed separately.
    """
    owners: set[int] = set()
    tensors: set[int] = set()
    for name, root in modules.items():
        root_owners = {id(module) for module in root.modules()}
        root_tensors = {id(tensor) for tensor in (*root.parameters(), *root.buffers())}
        assert owners.isdisjoint(root_owners), f"{operation}: root {name!r} shares modules with another root."
        assert tensors.isdisjoint(root_tensors), f"{operation}: root {name!r} shares tensors with another root."
        owners |= root_owners
        tensors |= root_tensors


def reject_unsupported_state_aliases(
    module: nn.Module,
    operation: str = "module preparation",
    allow_buffer_aliases: bool = False,
) -> None:
    """Reject unsafe tensor identities before changing registered state.

    Parameters may be tied. A tensor registered as both a parameter and a buffer is always
    rejected; multiple distinct buffer slots are also rejected unless the operation only copies
    values into existing tensors or saves/reuses state. Runtime buffer ties are allowed on those
    paths. This checks object identity, not shared storage or views. Visiting each owner once deduplicates repeated
    paths to the same physical slot, so shared submodules remain valid.
    """
    owners = list(module.modules())
    parameters = {id(tensor) for owner in owners for tensor in owner._parameters.values() if tensor is not None}
    buffers = [id(tensor) for owner in owners for tensor in owner._buffers.values() if tensor is not None]
    where = f"{operation} of {type(module).__name__}"
    assert parameters.isdisjoint(buffers), f"{where}: the same tensor is registered as both a parameter and a buffer."
    distinct_buffers = allow_buffer_aliases or len(set(buffers)) == len(buffers)
    assert distinct_buffers, f"{where}: the same tensor is in multiple distinct buffer slots."


def reject_differentiable_buffers(module: nn.Module, operation: str = "module placement") -> None:
    """Reject buffer gradients before placement, which uses no-grad tensor conversions."""
    names = [name for name, tensor in module.named_buffers() if tensor.requires_grad]
    if names:
        raise RuntimeError(
            f"{operation} of {type(module).__name__} does not support buffers with requires_grad=True; "
            f"register trainable state as parameters instead. First 8: {names[:8]}"
        )


def reject_unsafe_placement(module: nn.Module, operation: str = "module placement") -> None:
    """Reject non-default torch conversion settings, buffer aliases, and buffer gradients before placement."""
    validate_torch_state_settings()
    reject_unsupported_state_aliases(module, operation)
    reject_differentiable_buffers(module, operation)


def tied_parameter_registration_groups(
    registrations: Mapping[RegistrationKey, TensorRegistration],
) -> tuple[tuple[TensorRegistration, ...], ...]:
    """Return tied physical parameter slots in registration order.

    ``None`` registrations are ignored, and repeated qualified paths to one physical slot are
    counted once. Both group and slot order follow the registration snapshot.
    Registrations identify slots; their ``tensor`` values are snapshots and may be stale after
    module placement, so mutation consumers must re-read ``owner`` and ``local_name``.
    """
    groups: dict[int, dict[tuple[int, str], TensorRegistration]] = {}
    for (kind, _name), registration in registrations.items():
        if kind == "parameter" and registration.tensor is not None:
            slots = groups.setdefault(id(registration.tensor), {})
            slots.setdefault((id(registration.owner), registration.local_name), registration)
    return tuple(tuple(slots.values()) for slots in groups.values() if len(slots) > 1)


def resolve_storage_dtypes(
    module: nn.Module,
    default_dtype: torch.dtype | None,
    dtype_overrides: Mapping[str, torch.dtype],
    exclude: Iterable[nn.Module] = (),
) -> list[tuple[str, torch.Tensor, torch.dtype]]:
    """Return ``(qualified name, tensor, storage dtype)`` for every floating tensor ``module`` owns.

    ``dtype_overrides`` maps exact names to their dtype; other names use ``default_dtype``, and
    ``None`` keeps the current dtype. ``exclude`` lists submodules that own and cast their state.
    Storage dtypes are real, so integer and complex tensors keep their dtype, overrides included.
    """
    owned = {
        name: tensor for _kind, name, tensor in module_state_tensors(module, remove_duplicate=False, exclude=exclude)
    }
    assert dtype_overrides.keys() <= owned.keys(), (
        f"Storage dtype overrides must name state owned by {type(module).__name__}, not unknown or nested "
        f"module state: {sorted(dtype_overrides.keys() - owned.keys())[:8]}"
    )
    return [
        (name, tensor, dtype_overrides.get(name, default_dtype) or tensor.dtype)
        for name, tensor in owned.items()
        if tensor.is_floating_point()
    ]


@torch.no_grad()
def cast_to_storage_dtypes(module: nn.Module, entries: Iterable[tuple[str, torch.Tensor, torch.dtype]]) -> None:
    """Cast each listed tensor and its gradient in one ``nn.Module._apply`` over ``module``.

    Every alias of a tensor must request the same dtype; all requests are checked before any cast.
    Parameter objects and ties survive under the default torch settings that callers validate;
    buffer objects may change. Casting must precede FSDP wrapping.
    """
    targets: dict[int, tuple[torch.dtype, str]] = {}
    for name, tensor, dtype in entries:
        previous = targets.setdefault(id(tensor), (dtype, name))
        if previous[0] != dtype:
            raise ValueError(
                f"A shared tensor has conflicting storage dtypes: {previous[1]!r} requires {previous[0]}, "
                f"but {name!r} requires {dtype}. Make the dtype policy agree for every alias."
            )
    for parameter in module.parameters():
        if parameter.grad is not None and id(parameter) in targets:
            targets[id(parameter.grad)] = targets[id(parameter)]

    def convert(tensor: torch.Tensor) -> torch.Tensor:
        dtype = targets.get(id(tensor), (tensor.dtype,))[0]
        if dtype == tensor.dtype:
            return tensor
        assert not is_traceable_wrapper_subclass(tensor), f"Cast {type(tensor).__name__} storage before wrapping."
        return tensor.to(dtype=dtype)

    module._apply(convert)


def matches_fp32_declaration(name: str, declaring: nn.Module, prefix: str = "") -> bool:
    """Whether ``name`` is pinned by ``declaring._keep_in_fp32_modules``.

    Name-component matching follows Diffusers: a pattern pins a tensor when it equals one dotted
    component of the name relative to ``prefix``.
    """
    patterns = tuple(getattr(declaring, "_keep_in_fp32_modules", None) or ())
    return name.startswith(prefix) and any(pattern in name[len(prefix) :].split(".") for pattern in patterns)


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Expose a DTensor's local shard for storage-device checks."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def reject_inference_state(module: nn.Module, operation: str = "module preparation") -> None:
    """Refuse an inference-mode context and inference tensors before placing or loading ``module``.

    Both are construction bugs rather than states to repair: tensors created under
    ``torch.inference_mode`` reject every later in-place update (state loading, broadcasts,
    optimizer and EMA steps), and placement under that context would mint more of them.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            f"{operation} of {type(module).__name__} must not run under torch.inference_mode; its tensors would "
            "reject later in-place updates."
        )
    inference_names = [
        name
        for name, tensor in (*module.named_parameters(), *module.named_buffers())
        if tensor.is_inference() or local_tensor(tensor).is_inference()
    ]
    if inference_names:
        raise RuntimeError(
            f"{type(module).__name__} holds {len(inference_names)} inference tensor(s), first 8: "
            f"{inference_names[:8]}. Construct module state outside torch.inference_mode."
        )


def reject_extra_state(module: nn.Module, operation: str = "module preparation") -> None:
    """Refuse modules with ``extra_state``, which no FastGen state operation synchronizes or checks.

    Synchronization, sharding, and checkpoint coverage handle registered parameters and buffers only.
    """
    owners = [
        name or type(submodule).__name__
        for name, submodule in module.named_modules()
        if type(submodule).get_extra_state is not nn.Module.get_extra_state
        or type(submodule).set_extra_state is not nn.Module.set_extra_state
    ]
    if owners:
        raise RuntimeError(
            f"{operation}: {type(module).__name__} contains modules with extra_state ({owners[:8]}), which FastGen "
            "does not support; keep such state in parameters or buffers."
        )


def reject_complex_parameters(module: nn.Module, operation: str = "module preparation") -> None:
    """Refuse complex parameters, which FSDP mixed precision corrupts or fails to reduce.

    A real compute dtype drops their imaginary parts in forward and backward; a real reduce dtype fails on
    their gradients. Complex buffers remain valid: storage casts keep their dtype, and FSDP never casts buffers.
    """
    names = [name for name, parameter in module.named_parameters() if parameter.is_complex()]
    if names:
        raise RuntimeError(
            f"{operation} of {type(module).__name__} does not support complex parameters; register complex "
            f"constants as buffers or parameterize complex weights with real tensors. First 8: {names[:8]}"
        )


def has_meta_state(module: nn.Module) -> bool:
    """Return whether any registered parameter or buffer has meta storage."""
    return any(local_tensor(tensor).is_meta for tensor in module.parameters()) or any(
        local_tensor(tensor).is_meta for tensor in module.buffers()
    )


def module_topology_signature(module: nn.Module) -> tuple[tuple[str, str, str], ...]:
    """Describe ordered module paths, shared owners, and types without process-local identities."""
    owners: dict[int, str] = {}
    return tuple(
        (name, owners.setdefault(id(child), name), f"{type(child).__module__}.{type(child).__qualname__}")
        for name, child in module.named_modules(remove_duplicate=False)
    )


def registration_signature(
    registrations: Mapping[RegistrationKey, TensorRegistration], include_device: bool
) -> dict[RegistrationKey, _TensorMetadata]:
    """Describe registered tensor metadata and aliases, including ``None`` slots.

    Owner topology is checked separately by ``module_topology_signature`` or owner preservation;
    together with qualified registration names it already identifies shared physical slots.
    """
    groups: dict[int, list[RegistrationKey]] = {}
    for key, registration in registrations.items():
        if registration.tensor is not None:
            groups.setdefault(id(registration.tensor), []).append(key)
    alias_groups = {tensor_key: tuple(sorted(keys)) for tensor_key, keys in groups.items() if len(keys) > 1}
    signature: dict[RegistrationKey, _TensorMetadata] = {}
    for key, registration in registrations.items():
        tensor = registration.tensor
        if tensor is None:
            signature[key] = _TensorMetadata(persistent=registration.persistent)
            continue
        signature[key] = _TensorMetadata(
            dtype=str(tensor.dtype),
            shape=tuple(tensor.shape),
            layout=str(tensor.layout),
            requires_grad=bool(tensor.requires_grad),
            persistent=registration.persistent,
            tensor_type=f"{type(tensor).__module__}.{type(tensor).__qualname__}",
            placements=tuple(repr(placement) for placement in tensor.placements)
            if isinstance(tensor, DTensor)
            else None,
            aliases=alias_groups.get(id(tensor)),
            device_type=local_tensor(tensor).device.type if include_device else None,
        )
    return signature


@contextmanager
def _warn_if_rng_advanced(module: nn.Module):
    """Warn when a completed reset advances the CPU or, if CUDA is already initialized, any CUDA default generator."""
    generators = {"cpu": torch.default_generator}
    if torch.cuda.is_initialized():
        generators.update({f"cuda:{index}": generator for index, generator in enumerate(torch.cuda.default_generators)})
    states = {name: generator.get_state() for name, generator in generators.items()}
    yield
    advanced = [name for name, state in states.items() if not torch.equal(state, generators[name].get_state())]
    if advanced:
        # Rank 0 skips materialization in eager meta preparation, so do not use its logger.
        warnings.warn(
            f"{module.__class__.__name__}.reset_parameters() advanced the {', '.join(advanced)} RNG state during meta "
            "materialization. Reset hooks should not draw random values: callers overwrite the materialized values, "
            "and draws on only some ranks desynchronize their random streams."
        )


@torch.no_grad()
def materialize_meta_module(module: nn.Module, device: torch.device | str) -> None:
    """Allocate meta state, restore recorded parameter ties, and rebuild deterministic buffers.

    Existing real values survive placement. The root ``reset_parameters()`` hook may rebuild
    deterministic buffers and unregistered tensors, but must preserve every registered object,
    its metadata, and module-owner topology. Parameter ties are restored before the reset; generic
    ``tie_weights()`` hooks are not called. Callers restore authoritative parameter and persistent
    buffer values afterwards. Unregistered tensors are the reset hook's responsibility.

    This function issues no collectives directly, but a reset hook operating on DTensors may
    trigger collectives inside PyTorch's DTensor dispatch. Reset hooks should not draw random
    values; advancing a global RNG state emits a ``UserWarning``.
    """
    operation = "meta materialization"
    with validate_module_owners_preserved(module, operation):
        reject_inference_state(module, operation)
        reject_unsafe_placement(module, operation)
        registrations = tensor_registrations(module)
        expected_signature = registration_signature(registrations, include_device=False)
        tied_parameters = tied_parameter_registration_groups(registrations)
        target_device = resolve_device(device)

        def place_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if local_tensor(tensor).is_meta:
                return torch.empty_like(tensor, device=target_device)
            return tensor.to(device=target_device)

        module._apply(place_tensor)
        # Allocation may split parameter objects. Their values have not been initialized yet,
        # so any member can represent the original group before the reset or checkpoint copy.
        for group in tied_parameters:
            tensor = getattr(group[0].owner, group[0].local_name)
            for slot in group[1:]:
                setattr(slot.owner, slot.local_name, tensor)

        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            with validate_module_state_preserved(module, "meta reset_parameters"), _warn_if_rng_advanced(module):
                reset_parameters()
        # Wrapped third-party models may have no reset hook. Their caller must overwrite all
        # allocated state, including nonpersistent buffers, from rank 0.

        actual = tensor_registrations(module)
        signature = registration_signature(actual, include_device=False)
        misplaced = sorted(
            key
            for key, entry in actual.items()
            if entry.tensor is not None and local_tensor(entry.tensor).device != target_device
        )
        if signature != expected_signature or misplaced:
            changed = sorted(
                key
                for key in signature.keys() | expected_signature.keys()
                if signature.get(key) != expected_signature.get(key)
            )
            raise RuntimeError(
                f"{type(module).__name__} failed validation while materializing meta tensors: registered "
                "metadata and parameter ties must be preserved, and every tensor must finish on the requested device. "
                f"Changed registrations (first 8): {changed[:8]}; misplaced (first 8): {misplaced[:8]}"
            )
