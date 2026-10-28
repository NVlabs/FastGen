# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-architecture contracts shared by every FastGen diffusion network."""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from fastgen.networks.network import FastGenNetwork, pixel_shape_from_factors, validate_latent_compression
from fastgen.utils.guidance import GuidanceConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FeatureImplementation:
    name: str
    relative_path: str
    qualname: tuple[str, ...]
    packages_features: bool = True
    collects_features: bool = True


# Every distinct feature-collection/output implementation used by a released network.
# Wrappers that share a transformer implementation intentionally point to the same entry.
FEATURE_IMPLEMENTATIONS = (
    FeatureImplementation("edm-song", "fastgen/networks/EDM/network.py", ("SongUNet", "forward")),
    FeatureImplementation("edm-dhariwal", "fastgen/networks/EDM/network.py", ("DhariwalUNet", "forward")),
    FeatureImplementation("edm2", "fastgen/networks/EDM2/network.py", ("EDM2UNet", "forward")),
    FeatureImplementation("dit", "fastgen/networks/DiT/network.py", ("DiT", "forward")),
    FeatureImplementation("sd-unet", "fastgen/networks/SD15/network.py", ("classify_forward",)),
    FeatureImplementation("flux", "fastgen/networks/Flux/network.py", ("classify_forward",)),
    FeatureImplementation("flux2", "fastgen/networks/Flux2/network.py", ("classify_forward",)),
    FeatureImplementation("qwen-image", "fastgen/networks/QwenImage/network.py", ("classify_forward",)),
    FeatureImplementation("cogvideox", "fastgen/networks/CogVideoX/network.py", ("classify_forward",)),
    FeatureImplementation(
        "cosmos-predict2",
        "fastgen/networks/cosmos_predict2/network.py",
        ("CosmosPredict2DiT", "forward"),
    ),
    FeatureImplementation(
        "cosmos3-core",
        "fastgen/networks/cosmos3/modules.py",
        ("Cosmos3Backbone", "forward"),
    ),
    FeatureImplementation(
        "cosmos3-wrapper",
        "fastgen/networks/cosmos3/network.py",
        ("Cosmos3", "forward"),
        collects_features=False,
    ),
    FeatureImplementation("ltx2", "fastgen/networks/LTX2/network.py", ("classify_forward",)),
    FeatureImplementation("minimax-h3", "fastgen/networks/MiniMaxH3/network.py", ("classify_forward",)),
    FeatureImplementation(
        "wan-blocks",
        "fastgen/networks/Wan/network.py",
        ("classify_forward_block_forward",),
        packages_features=False,
    ),
    FeatureImplementation(
        "wan",
        "fastgen/networks/Wan/network.py",
        ("classify_forward",),
        collects_features=False,
    ),
    FeatureImplementation(
        "causal-wan-blocks",
        "fastgen/networks/Wan/network_causal.py",
        ("classify_forward_block_forward",),
        packages_features=False,
    ),
    FeatureImplementation(
        "causal-wan",
        "fastgen/networks/Wan/network_causal.py",
        ("classify_forward",),
        collects_features=False,
    ),
    FeatureImplementation(
        "wan-camera-blocks",
        "fastgen/networks/WanI2V/network_cam.py",
        ("_wancami2v_classify_forward_block_forward",),
        packages_features=False,
    ),
    FeatureImplementation(
        "wan-camera",
        "fastgen/networks/WanI2V/network_cam.py",
        ("_wancami2v_classify_forward",),
        collects_features=False,
    ),
    FeatureImplementation(
        "vace-blocks",
        "fastgen/networks/VaceWan/network.py",
        ("vace_classify_forward_block_forward",),
        packages_features=False,
    ),
    FeatureImplementation(
        "vace",
        "fastgen/networks/VaceWan/network.py",
        ("vace_classify_forward",),
        collects_features=False,
    ),
    FeatureImplementation(
        "causal-vace-blocks",
        "fastgen/networks/VaceWan/network_causal.py",
        ("vace_causal_classify_forward_block_forward",),
        packages_features=False,
    ),
    FeatureImplementation(
        "causal-vace",
        "fastgen/networks/VaceWan/network_causal.py",
        ("vace_causal_classify_forward",),
        collects_features=False,
    ),
)


@dataclass(frozen=True)
class NetworkContract:
    name: str
    module: str
    class_name: str
    optional_inputs: frozenset[str]
    feature_implementations: tuple[str, ...]


R = frozenset({"r"})
SLG = frozenset({"skip_layers"})
R_SLG = R | SLG


# This registry is deliberately explicit. A new released network must choose the feature
# implementation(s) that establish its return contract before this suite will cover it.
NETWORK_CONTRACTS = (
    NetworkContract("EDM", "fastgen.networks.EDM.network", "EDMPrecond", R, ("edm-song", "edm-dhariwal")),
    NetworkContract("EDM2", "fastgen.networks.EDM2.network", "EDM2Precond", R, ("edm2",)),
    NetworkContract("DiT", "fastgen.networks.DiT.network", "DiT", R, ("dit",)),
    NetworkContract("SD15", "fastgen.networks.SD15.network", "StableDiffusion15", frozenset(), ("sd-unet",)),
    NetworkContract("SDXL", "fastgen.networks.SDXL.network", "StableDiffusionXL", frozenset(), ("sd-unet",)),
    NetworkContract("Flux", "fastgen.networks.Flux.network", "Flux", R, ("flux",)),
    NetworkContract("Flux2", "fastgen.networks.Flux2.network", "Flux2", frozenset(), ("flux2",)),
    NetworkContract("QwenImage", "fastgen.networks.QwenImage.network", "QwenImage", R, ("qwen-image",)),
    NetworkContract("CogVideoX", "fastgen.networks.CogVideoX.network", "CogVideoX", frozenset(), ("cogvideox",)),
    NetworkContract(
        "CosmosPredict2",
        "fastgen.networks.cosmos_predict2.network",
        "CosmosPredict2",
        SLG,
        ("cosmos-predict2",),
    ),
    NetworkContract(
        "Cosmos3", "fastgen.networks.cosmos3.network", "Cosmos3", frozenset(), ("cosmos3-core", "cosmos3-wrapper")
    ),
    NetworkContract("LTX2", "fastgen.networks.LTX2.network", "LTX2", SLG, ("ltx2",)),
    NetworkContract("MiniMaxH3", "fastgen.networks.MiniMaxH3.network", "MiniMaxH3", SLG, ("minimax-h3",)),
    NetworkContract("Wan", "fastgen.networks.Wan.network", "Wan", R_SLG, ("wan-blocks", "wan")),
    NetworkContract(
        "CausalWan",
        "fastgen.networks.Wan.network_causal",
        "CausalWan",
        R_SLG,
        ("causal-wan-blocks", "causal-wan"),
    ),
    NetworkContract("WanI2V", "fastgen.networks.WanI2V.network", "WanI2V", R_SLG, ("wan-blocks", "wan")),
    NetworkContract(
        "CausalWanI2V",
        "fastgen.networks.WanI2V.network_causal",
        "CausalWanI2V",
        R_SLG,
        ("causal-wan-blocks", "causal-wan"),
    ),
    NetworkContract(
        "WanCamI2V",
        "fastgen.networks.WanI2V.network_cam",
        "WanCamI2V",
        R_SLG,
        ("wan-camera-blocks", "wan-camera"),
    ),
    NetworkContract("VACEWan", "fastgen.networks.VaceWan.network", "VACEWan", R_SLG, ("vace-blocks", "vace")),
    NetworkContract(
        "CausalVACEWan",
        "fastgen.networks.VaceWan.network_causal",
        "CausalVACEWan",
        R_SLG,
        ("causal-vace-blocks", "causal-vace"),
    ),
    NetworkContract("WanMoE", "fastgen.networks.Wan.wan_moe", "WanMoE", R_SLG, ("wan-blocks", "wan")),
    NetworkContract("WanMoET2V", "fastgen.networks.Wan.wan_moe", "WanMoET2V", R_SLG, ("wan-blocks", "wan")),
    NetworkContract("WanMoEI2V", "fastgen.networks.WanI2V.wan_moe", "WanMoEI2V", R_SLG, ("wan-blocks", "wan")),
    NetworkContract(
        "LingBotWorld",
        "fastgen.networks.WanI2V.wan_moe",
        "LingBotWorld",
        R_SLG,
        ("wan-camera-blocks", "wan-camera"),
    ),
)


def _resolve_network(case: NetworkContract):
    return getattr(importlib.import_module(case.module), case.class_name)


def _definition(spec: FeatureImplementation) -> tuple[str, ast.AST]:
    source = (REPO_ROOT / spec.relative_path).read_text()
    body: list[ast.stmt] = ast.parse(source).body
    node: ast.AST | None = None
    for part in spec.qualname:
        matches = [
            candidate
            for candidate in body
            if isinstance(candidate, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and candidate.name == part
        ]
        assert len(matches) == 1, f"{spec.name}: could not resolve {'.'.join(spec.qualname)}"
        node = matches[0]
        body = node.body
    assert node is not None
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment, node


_PROHIBITED_FEATURE_PATTERNS = {
    "aliases the feature return condition": r"\binclude_features\s*=",
    "constructs a traced set": r"\bfeature_indices\s*=\s*set\s*\(",
    "uses a truthy fallback": r"\bfeature_indices\s*=\s*feature_indices\s+or\b",
    "collapses None and empty by length": r"\blen\(\s*feature_indices\s*\)\s*==\s*0\b",
    "collapses None and empty by truthiness": r"\bif\s+(?:not\s+)?feature_indices\s*:",
}


@pytest.mark.parametrize("spec", FEATURE_IMPLEMENTATIONS, ids=lambda spec: spec.name)
def test_feature_implementations_preserve_none_and_empty_set(spec: FeatureImplementation):
    source, node = _definition(spec)
    compact = re.sub(r"\s+", "", source)

    for description, pattern in _PROHIBITED_FEATURE_PATTERNS.items():
        assert re.search(pattern, source) is None, f"{spec.name} {description}"

    if spec.packages_features:
        has_feature_ternary = any(
            re.search(pattern, source) is not None
            for pattern in (
                r"\bif\s+feature_indices\s+is\s+not\s+None\s+else\b",
                r"\bfeature_indices\s+is\s+None\s+else\s+\[",
            )
        )
        if not has_feature_ternary:
            assert "iffeature_indicesisnotNone:" in compact
            assert "out=[out,features]" in compact
        assert "features" in compact, f"{spec.name} does not include features in its output contract"

    if spec.collects_features:
        assert "features.append(" in compact, f"{spec.name} does not collect features in traversal order"
        assert (
            "len(features)==len(feature_indices)" in compact
        ), f"{spec.name} does not stop after all requested features are collected"
        assert any(
            isinstance(candidate, ast.Return)
            and candidate.value is not None
            and any(isinstance(value, ast.Name) and value.id == "features" for value in ast.walk(candidate.value))
            for candidate in ast.walk(node)
        ), f"{spec.name} does not return the collected feature list"


@pytest.mark.parametrize("case", NETWORK_CONTRACTS, ids=lambda case: case.name)
def test_every_released_network_exposes_the_feature_contract(case: NetworkContract):
    network_cls = _resolve_network(case)
    parameters = inspect.signature(network_cls.forward).parameters

    assert parameters["feature_indices"].default is None
    assert parameters["return_features_early"].default is False
    assert "_validate_forward_kwargs" in network_cls.forward.__code__.co_names

    optional_inputs = {"r", "skip_layers"}
    assert optional_inputs & parameters.keys() == case.optional_inputs
    assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    required = {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    assert required == {"self", "x_t", "t"}

    known_implementations = {spec.name for spec in FEATURE_IMPLEMENTATIONS}
    assert case.feature_implementations
    assert set(case.feature_implementations) <= known_implementations


@pytest.mark.parametrize("case", NETWORK_CONTRACTS, ids=lambda case: case.name)
def test_every_released_network_forward_defaults_constrain_and_round_to_the_state_dtype(case: NetworkContract):
    """Guidance requests unconstrained native branches with ``apply_output_conditioning=False`` and
    ``output_dtype=None``; by default a forward constrains its output and rounds it to the state dtype."""
    parameters = inspect.signature(_resolve_network(case).forward).parameters
    assert parameters["apply_output_conditioning"].default is True
    assert parameters["output_dtype"].default == "same"
    assert "transform_output" not in parameters


def test_every_feature_implementation_is_used_by_a_released_network():
    registered = {name for case in NETWORK_CONTRACTS for name in case.feature_implementations}
    assert registered == {spec.name for spec in FEATURE_IMPLEMENTATIONS}


# Feature-collecting loops of the networks that accept ``skip_layers``.
SKIP_LAYER_FEATURE_IMPLEMENTATIONS = tuple(
    spec
    for spec in FEATURE_IMPLEMENTATIONS
    if spec.collects_features
    and any(
        "skip_layers" in case.optional_inputs and spec.name in case.feature_implementations
        for case in NETWORK_CONTRACTS
    )
)
# A ``continue`` inside these targets their own loop, not the block loop.
_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.For, ast.AsyncFor, ast.While)


def _walk_same_iteration(node: ast.AST):
    yield node
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, _NESTED_SCOPES):
            yield from _walk_same_iteration(child)


@pytest.mark.parametrize("spec", SKIP_LAYER_FEATURE_IMPLEMENTATIONS, ids=lambda spec: spec.name)
def test_skipped_layers_still_record_requested_features(spec: FeatureImplementation):
    """A skipped block acts as the identity, but each requested index still yields one feature, as the
    ``len(features) == len(feature_indices)`` early exits and checks require."""
    _, node = _definition(spec)
    guards = [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.If)
        and any(isinstance(name, ast.Name) and name.id == "skip_layers" for name in ast.walk(candidate.test))
    ]
    assert guards, f"{spec.name} accepts skip_layers but never checks them"
    for guard in guards:
        guarded = list(_walk_same_iteration(guard))
        assert not any(
            isinstance(statement, ast.Continue) for statement in guarded
        ), f"{spec.name} skips the feature collection of a skipped layer"
        assert not any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and (call.func.value.id, call.func.attr) == ("features", "append")
            for call in guarded
        ), f"{spec.name} records features only for executed layers"


class _ForwardContractNetwork(FastGenNetwork):
    """Weightless executable model for the shared feature-return truth table."""

    net_pred_type = "flow"

    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(
        self,
        x_t,
        t,
        condition=None,
        return_features_early=False,
        feature_indices=None,
        **fwd_kwargs,
    ):
        del t, condition
        self._validate_forward_kwargs(
            fwd_kwargs,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            num_features=4,
        )

        features = []
        for index in range(4):
            feature = x_t + index
            if feature_indices is not None and index in feature_indices:
                features.append(feature)
            if return_features_early and feature_indices is not None and len(features) == len(feature_indices):
                return features

        output = x_t + 10
        return output if feature_indices is None else [output, features]


def test_feature_return_truth_table_and_traversal_order():
    network = _ForwardContractNetwork()
    state = torch.tensor(1)
    timestep = torch.tensor([0.5])

    assert network(state, timestep, feature_indices=None).item() == 11
    output, features = network(state, timestep, feature_indices=set())
    assert output.item() == 11
    assert features == []
    assert network(state, timestep, return_features_early=True, feature_indices=set()) == []

    features = network(state, timestep, return_features_early=True, feature_indices={3, 1, 2})
    assert [feature.item() for feature in features] == [2, 3, 4]

    with pytest.raises(ValueError, match="requires feature_indices"):
        network(state, timestep, return_features_early=True, feature_indices=None)


@pytest.mark.parametrize("return_features_early", [False, True])
@pytest.mark.parametrize(
    ("indices", "error", "message"),
    [
        ({-1}, ValueError, "out-of-range indices"),
        ({4}, ValueError, "out-of-range indices"),
        ({1, 8}, ValueError, "out-of-range indices"),
        ({"1"}, TypeError, "contain integers excluding bool"),
        ({1.0}, TypeError, "contain integers excluding bool"),
        ({True}, TypeError, "contain integers excluding bool"),
        ({None}, TypeError, "contain integers excluding bool"),
    ],
)
def test_feature_validation_rejects_invalid_indices_before_compute(indices, error, message, return_features_early):
    network = _ForwardContractNetwork()
    # These sentinels cannot participate in tensor arithmetic: validation must fail first.
    with pytest.raises(error, match=message):
        network(object(), object(), feature_indices=indices, return_features_early=return_features_early)


def _feature_contract_stub(case: NetworkContract):
    """Build only the architecture metadata needed to validate a real wrapper's forward call."""
    network_cls = _resolve_network(case)
    network = network_cls.__new__(network_cls)
    torch.nn.Module.__init__(network)
    network.net_pred_type = "flow"
    network.r_embedder = None
    network.blocks = [None] * 4
    network.transformer = SimpleNamespace(
        blocks=[None] * 4,
        transformer_blocks=[None] * 4,
        single_transformer_blocks=[None] * 3,
        layers=[None] * 4,
        r_embedder=None,
    )
    # EDM indices address resolution stages, not every residual/auxiliary encoder block.
    encoder = {
        "32x32_conv": None,
        "32x32_block0": None,
        "32x32_block1": None,
        "16x16_aux_down": None,
        "16x16_block0": None,
        "16x16_block1": None,
        "8x8_block0": None,
        "8x8_block1": None,
    }
    network.model = SimpleNamespace(enc=encoder, num_blocks=2, r_timestep=None)
    network.unet = SimpleNamespace(enc=encoder, num_blocks=2, emb_noise_r=None, down_blocks=[None] * 3)
    network.network_high = SimpleNamespace(transformer=network.transformer)
    network.network_low = SimpleNamespace(transformer=network.transformer)
    num_features = 3 if case.name in {"EDM", "EDM2"} else 7 if case.name in {"Flux", "Flux2"} else 4
    return network, num_features


@pytest.mark.parametrize("case", NETWORK_CONTRACTS, ids=lambda case: case.name)
def test_released_networks_validate_feature_indices_before_compute(case: NetworkContract):
    network, num_features = _feature_contract_stub(case)
    for early in (False, True):
        for indices in ({-1}, {num_features}, {0, num_features}):
            with pytest.raises(ValueError, match=rf"feature_indices.*valid range is \[0, {num_features}\)"):
                network(object(), object(), feature_indices=indices, return_features_early=early)
        for indices in ({False}, {0.0}, {"0"}):
            with pytest.raises(TypeError, match="contain integers excluding bool"):
                network(object(), object(), feature_indices=indices, return_features_early=early)

    assert network(object(), object(), feature_indices=set(), return_features_early=True) == []

    class ValidatedBeforeCompute(Exception):
        pass

    validate = network._validate_forward_kwargs

    def stop_after_validation(*args, **kwargs):
        assert kwargs["num_features"] == num_features
        validate(*args, **kwargs)
        raise ValidatedBeforeCompute

    with patch.object(network, "_validate_forward_kwargs", side_effect=stop_after_validation):
        # Counts are always supplied, even without feature extraction. The highest valid
        # index must survive each wrapper's architecture-specific count.
        for indices in (None, set(), {0, num_features - 1}):
            with pytest.raises(ValidatedBeforeCompute):
                network(object(), object(), feature_indices=indices)


def test_transform_output_resolves_fwd_pred_type():
    network = _ForwardContractNetwork()
    resolved = []

    def convert(prediction, x_t, t, condition, fwd_pred_type, **context):
        resolved.append(fwd_pred_type)
        return prediction

    state = torch.zeros(1, dtype=torch.float64)
    with patch.object(network, "_convert_output", side_effect=convert):
        network.transform_output(state, state, state)
        network.transform_output(state, state, state, fwd_pred_type="eps")
    assert resolved == ["flow", "eps"]

    with pytest.raises(ValueError, match="fwd_pred_type='typo'"):
        network.transform_output(object(), object(), object(), fwd_pred_type="typo")


def test_transform_output_rejects_a_conversion_that_rounds():
    network = _ForwardContractNetwork()
    state = torch.zeros(1, dtype=torch.float64)

    def convert(prediction, x_t, t, condition, fwd_pred_type, **context):
        return prediction.float()

    with patch.object(network, "_convert_output", side_effect=convert):
        with pytest.raises(AssertionError, match=r"converted\.dtype \(torch\.float32\) must be torch\.float64"):
            network.transform_output(state, state, state)


def test_feature_indices_none_guard_is_torch_dynamo_safe():
    def probe(x, feature_indices=None):
        count = 0
        for index in range(2):
            if feature_indices is not None and index in feature_indices:
                count += 1
        return x + count + (10 if feature_indices is not None else 0)

    compiled = torch.compile(probe, backend="eager", fullgraph=True)
    assert compiled(torch.tensor(1), None).item() == 1
    assert compiled(torch.tensor(1), set()).item() == 11
    assert compiled(torch.tensor(1), {0}).item() == 12


def test_validate_forward_kwargs_rejects_unsupported_controls():
    network = _ForwardContractNetwork()
    assert network._validate_forward_kwargs({"skip_layers": None}) is None
    with pytest.raises(TypeError, match="feature_indices to be a set or None"):
        network._validate_forward_kwargs({}, feature_indices=[1])
    with pytest.raises(ValueError, match="requires feature_indices"):
        network._validate_forward_kwargs({}, return_features_early=True, feature_indices=None)
    assert network._validate_forward_kwargs({}, return_features_early=True, feature_indices=set()) is None
    with pytest.raises(NotImplementedError, match="secondary r timesteps"):
        network._validate_forward_kwargs({}, r=object())
    with pytest.raises(NotImplementedError, match="skip_layers"):
        network._validate_forward_kwargs({"skip_layers": [1]})
    # LTX-2's STG and cross-modal flags are its own forward parameters, unknown to every other network.
    for name, value in (("skip_layer", [1]), ("skip_self_attn_layers", [1]), ("skip_cross_modal_attn", True)):
        with pytest.raises(TypeError, match=f"unexpected keyword argument.*'{name}'"):
            network._validate_forward_kwargs({name: value})


def test_validate_forward_kwargs_accepts_supported_controls_and_checks_bounds():
    network = _ForwardContractNetwork()
    result = network._validate_forward_kwargs(
        {},
        r=object(),
        skip_layers=[0, 3],
        supports_r_timestep=True,
        supports_skip_layers=True,
        feature_indices={0, 3},
        num_features=4,
    )
    assert result is None
    with pytest.raises(ValueError, match=r"out-of-range indices \[4\]"):
        network._validate_forward_kwargs({}, skip_layers=[4], supports_skip_layers=True, num_features=4)


@pytest.mark.parametrize("control", ["feature_indices", "skip_layers"])
def test_forward_index_controls_share_feature_count(control):
    network = _ForwardContractNetwork()
    controls = {"supports_skip_layers": True}
    with pytest.raises(RuntimeError, match="must provide num_features"):
        network._validate_forward_kwargs({}, **{control: {0}}, **controls)

    assert network._validate_forward_kwargs({}, **{control: {0, 3}}, num_features=4, **controls) is None
    for index in (-1, 4):
        with pytest.raises(ValueError, match=rf"{control}.*valid range is \[0, 4\)"):
            network._validate_forward_kwargs({}, **{control: {index}}, num_features=4, **controls)


ODE_ONLY_SAMPLERS = (
    ("fastgen.networks.EDM.network", "EDMPrecond"),
    ("fastgen.networks.EDM2.network", "EDM2Precond"),
    ("fastgen.networks.SD15.network", "StableDiffusion15"),
    ("fastgen.networks.SDXL.network", "StableDiffusionXL"),
    ("fastgen.networks.Flux.network", "Flux"),
    ("fastgen.networks.Flux2.network", "Flux2"),
    ("fastgen.networks.QwenImage.network", "QwenImage"),
    ("fastgen.networks.LTX2.network", "LTX2"),
    ("fastgen.networks.MiniMaxH3.network", "MiniMaxH3"),
    ("fastgen.networks.cosmos_predict2.network", "CosmosPredict2"),
    ("fastgen.networks.cosmos3.network", "Cosmos3"),
    ("fastgen.networks.Wan.network", "Wan"),
    ("fastgen.networks.Wan.network_causal", "CausalWan"),
    ("fastgen.networks.WanI2V.network", "WanI2V"),
    ("fastgen.networks.WanI2V.network_causal", "CausalWanI2V"),
    ("fastgen.networks.VaceWan.network", "VACEWan"),
    ("fastgen.networks.VaceWan.network_causal", "CausalVACEWan"),
    ("fastgen.networks.Wan.wan_moe", "WanMoET2V"),
    ("fastgen.networks.WanI2V.wan_moe", "WanMoEI2V"),
    ("fastgen.networks.WanI2V.wan_moe", "LingBotWorld"),
)


@pytest.mark.parametrize(("module_name", "class_name"), ODE_ONLY_SAMPLERS)
def test_ode_only_samplers_reject_sde_before_accessing_state(module_name, class_name):
    network_type = getattr(importlib.import_module(module_name), class_name)
    parameters = inspect.signature(network_type.sample).parameters
    assert parameters["sample_type"].default == "ode"
    required = {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    assert required == {"self", "noise"}
    network = object.__new__(network_type)
    schedule_type = {
        "EDMPrecond": "edm",
        "EDM2Precond": "edm",
        "StableDiffusion15": "sd",
        "StableDiffusionXL": "sdxl",
    }.get(class_name, "rf")
    object.__setattr__(network, "schedule_type", schedule_type)

    with pytest.raises(ValueError, match=r"sample_type='sde'"):
        network_type.sample(network, torch.empty(0), sample_type="sde")
    with pytest.raises(TypeError, match="unexpected keyword argument 'typo'"):
        network_type.sample(network, torch.empty(0), typo=True)

    object.__setattr__(network, "schedule_type", "unsupported")
    with pytest.raises(ValueError, match=r"schedule_type='unsupported'"):
        network_type.sample(network, torch.empty(0), sample_type="ode")


def test_cogvideox_sampler_is_stochastic_and_rejects_ode():
    from fastgen.networks.CogVideoX.network import CogVideoX

    assert inspect.signature(CogVideoX.sample).parameters["sample_type"].default == "sde"
    network = object.__new__(CogVideoX)
    object.__setattr__(network, "schedule_type", "cogvideox")
    with pytest.raises(ValueError, match=r"sample_type='ode'"):
        CogVideoX.sample(network, torch.empty(0), sample_type="ode")


def test_dit_sample_type_defaults_follow_schedule():
    from fastgen.networks.DiT.network import DiT

    parameters = inspect.signature(DiT.sample).parameters
    assert {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    } == {"self", "noise"}
    network = object.__new__(DiT)
    torch.nn.Module.__init__(network)
    network.schedule_type = "rf"
    noise = torch.zeros(1, 4, 2, 2)

    with (
        patch.object(DiT, "_sample_flow", return_value=noise) as sample_ode,
        patch.object(DiT, "_sample_flow_sde", return_value=noise) as sample_sde,
    ):
        assert network.sample(noise, sample_steps=1, guidance_cfg=GuidanceConfig()) is noise
        assert network.sample(noise, sample_type="ode", sample_steps=1, guidance_cfg=GuidanceConfig()) is noise

    sample_ode.assert_called_once()
    sample_sde.assert_called_once()
    with pytest.raises(ValueError, match="sample_type"):
        network.sample(noise, sample_type="typo", guidance_cfg=GuidanceConfig())
    network.schedule_type = "sd"
    with pytest.raises(ValueError, match="sample_type"):
        network.sample(noise, sample_type="sde", guidance_cfg=GuidanceConfig())
    network.schedule_type = "ddpm"
    with patch.object(DiT, "_sample_ddpm", return_value=noise) as sample_ddpm:
        assert network.sample(noise, sample_steps=1, guidance_cfg=GuidanceConfig()) is noise
        assert network.sample(noise, sample_type="ode", sample_steps=1, guidance_cfg=GuidanceConfig()) is noise
    assert [call.kwargs["sample_type"] for call in sample_ddpm.call_args_list] == ["ddpm", "ode"]
    network.schedule_type = "unsupported"
    with pytest.raises(ValueError, match="schedule_type"):
        network.sample(noise, guidance_cfg=GuidanceConfig())


def _fastgen_network_classes() -> list[type]:
    """Every ``FastGenNetwork`` subclass defined under ``fastgen.networks`` or ``fastgen.methods``."""
    import fastgen.methods
    import fastgen.networks

    for package in (fastgen.networks, fastgen.methods):
        for module in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
            importlib.import_module(module.name)
    classes, pending = set(), [FastGenNetwork]
    while pending:
        for subclass in pending.pop().__subclasses__():
            if subclass not in classes:
                classes.add(subclass)
                pending.append(subclass)
    return sorted(
        (cls for cls in classes if cls.__module__.startswith("fastgen.")),
        key=lambda cls: f"{cls.__module__}.{cls.__qualname__}",
    )


def _forwarded_parameter_names(cls: type, stop: type) -> set[str]:
    """Named ``__init__`` parameters along ``cls``'s chain of ``**kwargs`` forwarding, up to ``stop``."""
    names = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        parameters = inspect.signature(init).parameters.values()
        names |= {
            parameter.name
            for parameter in parameters
            if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY) and parameter.name != "self"
        }
        if klass is stop or not any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters):
            break
    return names


def test_network_constructors_leave_schedule_parameters_to_schedule_kwargs():
    """A network constructor naming a noise-schedule parameter would silently keep it from the schedule.

    Every network forwards the constructor kwargs it does not name to its schedule as ``**schedule_kwargs``,
    and a config may pair a network with any registered schedule, so every schedule parameter name is
    reserved. Discovery imports ``fastgen.networks`` and ``fastgen.methods``, so new networks are covered.
    """
    from fastgen.networks.noise_schedule import NOISE_SCHEDULES, BaseNoiseSchedule

    schedule_names = set().union(
        *(_forwarded_parameter_names(schedule, BaseNoiseSchedule) for schedule in NOISE_SCHEDULES.values())
    )
    assert {"max_t", "shift", "student_grid_cfg"} <= schedule_names
    networks = _fastgen_network_classes()
    assert {"EDMPrecond", "Wan", "TrigFlowPrecond"} <= {network.__name__ for network in networks}

    shadowing = {
        f"{network.__module__}.{network.__qualname__}": sorted(names)
        for network in networks
        if (names := _forwarded_parameter_names(network, FastGenNetwork) & schedule_names)
    }
    assert not shadowing, f"pass these noise-schedule parameters through **schedule_kwargs instead: {shadowing}"


def test_pixel_shape_from_factors_maps_image_and_causal_video_latents():
    assert pixel_shape_from_factors([16, 60, 104], (8, 8)) == (480, 832)
    # The first latent frame covers one pixel frame and every later one four.
    assert pixel_shape_from_factors([16, 21, 60, 104], (4, 8, 8)) == (81, 480, 832)
    assert pixel_shape_from_factors([16, 1, 60, 104], (4, 8, 8)) == (1, 480, 832)
    mismatched_ranks = [([16, 60, 104], (4, 8, 8)), ([16, 21, 60, 104], (8, 8)), ([1, 16, 21, 60, 104], (4, 8, 8))]
    for latent_shape, factors in mismatched_ranks:
        with pytest.raises(ValueError, match=r"\[C, H, W\] latent with \(H, W\) factors"):
            pixel_shape_from_factors(latent_shape, factors)


@pytest.mark.parametrize("case", NETWORK_CONTRACTS, ids=lambda case: case.name)
def test_every_released_network_knows_its_pixel_shape_without_an_instance(case: NetworkContract):
    """Configs size pixel inputs with ``pixel_shape_for`` before any network or VAE exists."""
    network_cls = _resolve_network(case)
    # Wan networks declare no class-level factors; they follow the latent channels, 16 for the Wan2.1 VAE.
    factors = getattr(network_cls, "latent_compression_factors", (4, 8, 8))

    h_factor, w_factor = factors[-2:]
    if len(factors) == 2:
        assert network_cls.pixel_shape_for([16, 2, 3]) == (2 * h_factor, 3 * w_factor)
        return
    # Seven latent frames are also a valid MiniMax-H3 ``5 * n + 2`` count.
    num_frames, height, width = network_cls.pixel_shape_for([16, 7, 2, 3])
    assert (height, width) == (2 * h_factor, 3 * w_factor)
    assert num_frames >= 7


def test_latent_compression_is_validated_only_for_networks_that_declare_it():
    vae = SimpleNamespace(latent_compression_factors=(4, 8, 8))
    validate_latent_compression(SimpleNamespace(), vae)
    validate_latent_compression(SimpleNamespace(latent_compression_factors=(4, 8, 8)), vae)
    with pytest.raises(ValueError, match=r"\(4, 16, 16\), but its VAE compresses by \(4, 8, 8\)"):
        validate_latent_compression(SimpleNamespace(latent_compression_factors=(4, 16, 16)), vae)
