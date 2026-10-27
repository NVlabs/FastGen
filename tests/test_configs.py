# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests that every experiment config builds.

Every ``config_*.py`` directly under a model directory in
``fastgen/configs/experiments`` must be importable and return a
:class:`~fastgen.configs.config.BaseConfig` from ``create_config()``.

This guards against broken imports and attribute typos in configs that are not
otherwise exercised by the test suite -- e.g. a config importing a dataloader
that is not included in the package.
"""

import ast
import importlib
import inspect
import re
import textwrap
from copy import deepcopy
from pathlib import Path

import attrs
import pytest
from omegaconf import DictConfig, ListConfig

from fastgen.configs.config import BaseConfig, BaseModelConfig, BaseTrainerConfig
from fastgen.configs.config_utils import model_pixel_shape, override_config_with_opts
from fastgen.configs.instantiate import locate
from fastgen.networks.noise_schedule import get_noise_schedule
from fastgen.utils.scripts import load_config
from tests.utils import network_schedule_kwargs

# tests/test_configs.py -> parents[1] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "fastgen" / "configs" / "experiments"


def discover_config_files() -> list[str]:
    """Return repo-root-relative POSIX paths to experiment configs."""
    return [path.relative_to(REPO_ROOT).as_posix() for path in sorted(EXPERIMENTS_DIR.glob("*/config_*.py"))]


def config_id(rel_path: str) -> str:
    """Short, readable pytest id, e.g. ``WanT2V/config_kd_path_14b.py``."""
    return rel_path.split("experiments/", 1)[-1]


def build_config(config_file: str):
    """Import a config module and return the result of ``create_config()``.

    Mirrors ``fastgen.configs.config_utils.import_config_from_python_file`` (the
    loader ``train.py`` uses) but derives the fully-qualified module name so it
    does not depend on the process working directory.
    """
    assert config_file.endswith(".py"), config_file
    module_name = config_file[: -len(".py")].replace("/", ".")
    module = importlib.import_module(module_name)
    config = module.create_config()
    # Like ``load_config``, an unnamed config takes its file's name.
    if config.log_config.name is None:
        config.log_config.name = Path(config_file).stem
    return config


def assert_discriminator_callback_matches_config(config, config_file: str):
    """Zero-GAN DMD-derived configs must not register discriminator callbacks."""
    if getattr(config.model, "gan_loss_weight_gen", None) == 0:
        assert "grad_clip_discriminator" not in config.trainer.callbacks, (
            f"{config_file}: zero-GAN config registers grad_clip_discriminator "
            "even though the model does not build a discriminator"
        )


def _registered_ema_names(callbacks) -> set[str]:
    """EMA networks that a registered ``EMACallback`` keeps updated, keyed by the name it writes."""
    from fastgen.callbacks.ema import EMACallback

    default_name = inspect.signature(EMACallback).parameters["ema_name"].default
    names = set()
    for node in callbacks.values():
        if "_target_" not in node:
            continue
        target = node["_target_"]
        if not callable(target):
            target = locate(target)
        if isinstance(target, type) and issubclass(target, EMACallback):
            names.add(node.get("ema_name", default_name))
    return names


def assert_ema_callback_matches_config(config, config_file: str):
    """Every EMA network in ``use_ema`` must have a callback that updates it.

    ``use_ema`` only builds the EMA copy and makes ``net_inference`` prefer it; the averaging
    itself lives in ``EMACallback``. An unregistered callback therefore leaves the copy pinned to
    its initialization while validation, media, and checkpoints all silently read from it.
    """
    use_ema = getattr(config.model, "use_ema", False)
    if isinstance(use_ema, bool):
        use_ema = ["ema"] if use_ema else []
    missing = sorted(set(use_ema) - _registered_ema_names(config.trainer.callbacks))
    assert not missing, (
        f"{config_file}: model.use_ema requests {missing} without a matching EMACallback; "
        "register one, e.g. trainer.callbacks.update(EMA_CALLBACK)"
    )


def assert_parallelism_is_unambiguous(config, config_file: str):
    """A recipe may select DDP or FSDP, but never both."""
    assert not (
        config.trainer.ddp and config.trainer.fsdp
    ), f"{config_file}: trainer.ddp and trainer.fsdp are both enabled"


def assert_loaders_match_input_shape(config, config_file: str):
    """Raw-media loaders produce the pixels that encode to ``input_shape``, as ``model_pixel_shape`` derives them."""
    if config.model.net is None or tuple(config.model.net.get("modalities") or ()) == ("audio",):
        return
    for name in ("dataloader_train", "dataloader_val"):
        loader = getattr(config, name)
        if loader is None or all(loader.get(key) is None for key in ("input_res", "img_size", "sequence_length")):
            continue
        *frames, height, width = model_pixel_shape(config.model)
        input_res = loader.get("input_res")
        actual = {
            "input_res": (input_res, input_res) if isinstance(input_res, int) else input_res,
            "img_size": loader.get("img_size"),
            "sequence_length": loader.get("sequence_length"),
        }
        expected = {
            "input_res": (width, height),
            "img_size": (width, height),
            "sequence_length": next(iter(frames), None),
        }
        for key, value in actual.items():
            if value is not None:
                value = value if key == "sequence_length" else tuple(value)
                assert value == expected[key], (
                    f"{config_file}: {name}.{key} is {value}, but input_shape {config.model.input_shape} "
                    f"decodes to {expected[key]}"
                )


def assert_validation_settings_have_a_loader(config, config_file: str):
    """The trainer validates only with a ``dataloader_val``; without one, validation settings never take effect."""
    if config.dataloader_val is not None:
        return
    defaults = BaseTrainerConfig()
    for field in ("validation_iter", "global_vars_val", "val_seed"):
        assert getattr(config.trainer, field) == getattr(
            defaults, field
        ), f"{config_file}: trainer.{field} is set without a dataloader_val"


def test_ladd_and_dmd2_configs_share_only_adversarial_settings():
    from fastgen.configs.callbacks import BASE_CALLBACKS, GradClip_CALLBACK
    from fastgen.configs.methods.config_dmd2 import ModelConfig as DMD2ModelConfig
    from fastgen.configs.methods.config_ladd import ModelConfig as LADDModelConfig
    from fastgen.configs.methods.config_ladd import create_config as create_ladd_config

    assert not issubclass(LADDModelConfig, DMD2ModelConfig)
    assert DMD2ModelConfig().vsd_loss_weight == 1.0

    ladd_config = create_ladd_config()
    for field in (
        "vsd_loss_weight",
        "fake_score_optimizer",
        "fake_score_scheduler",
        "fake_score_sample_t_overrides",
        "fake_score_pred_type",
    ):
        assert not hasattr(ladd_config.model, field)
    with pytest.raises(ValueError, match="vsd_loss_weight"):
        override_config_with_opts(ladd_config, ["-", "model.vsd_loss_weight=0"])
    assert ladd_config.model.gan_loss_weight_gen == 1.0
    assert set(ladd_config.trainer.callbacks) == {*GradClip_CALLBACK, *BASE_CALLBACKS}
    assert "grad_clip_fake_score" not in ladd_config.trainer.callbacks
    assert "grad_clip_discriminator" not in ladd_config.trainer.callbacks
    assert ladd_config.dataloader_train.batch_size == 256
    assert ladd_config.model.discriminator_scheduler.warm_up_steps == [0]
    assert ladd_config.model.net_scheduler.warm_up_steps == [0]


def test_causvid_and_self_forcing_configs_expose_context_noise():
    from fastgen.configs.methods.config_causvid import ModelConfig as CausVidModelConfig
    from fastgen.configs.methods.config_causvid import create_config as create_causvid_config
    from fastgen.configs.methods.config_self_forcing import ModelConfig as SelfForcingModelConfig

    assert issubclass(SelfForcingModelConfig, CausVidModelConfig)
    causvid_config = create_causvid_config()
    assert causvid_config.model.context_noise is None
    overridden = override_config_with_opts(causvid_config, ["-", "model.context_noise=0.1"])
    assert overridden.model.context_noise == 0.1


CONFIG_FILES = discover_config_files()


def _mutable_config_paths(value, path="config", result=None):
    """Collect mutable config nodes while allowing shared immutable classes and callables."""
    if result is None:
        result = {}
    if attrs.has(type(value)):
        children = ((field.name, getattr(value, field.name)) for field in attrs.fields(type(value)))
    elif isinstance(value, (dict, DictConfig)):
        children = value.items()
    elif isinstance(value, (list, ListConfig)):
        children = enumerate(value)
    elif isinstance(value, tuple):
        for index, child in enumerate(value):
            _mutable_config_paths(child, f"{path}.{index}", result)
        return result
    else:
        return result
    if id(value) in result:
        return result
    result[id(value)] = path
    for name, child in children:
        _mutable_config_paths(child, f"{path}.{name}", result)
    return result


# Every shipped config: the experiment recipes plus the runnable method defaults.
SHIPPED_CONFIG_FILES = CONFIG_FILES + [
    path.relative_to(REPO_ROOT).as_posix()
    for path in sorted((REPO_ROOT / "fastgen/configs/methods").glob("config_*.py"))
]


def assert_config_calls_are_isolated(config_file):
    first = build_config(config_file)
    second = build_config(config_file)
    first_nodes = _mutable_config_paths(first)
    second_nodes = _mutable_config_paths(second)

    shared = sorted(first_nodes[node] for node in first_nodes.keys() & second_nodes.keys())
    assert not shared, f"{config_file}: create_config() shares mutable nodes: {shared}"


@pytest.mark.parametrize("config_file", SHIPPED_CONFIG_FILES)
def test_config_calls_own_their_mutable_nodes(config_file):
    assert_config_calls_are_isolated(config_file)


def test_wan_recipe_edits_do_not_change_other_recipes_or_presets():
    from fastgen.configs.net import Wan_1_3B_Config

    first = build_config("fastgen/configs/experiments/WanT2V/config_dmd2.py")
    second = build_config("fastgen/configs/experiments/WanT2V/config_sft.py")
    assert first.model.net is not second.model.net
    assert first.model.net is not Wan_1_3B_Config
    expected_net = deepcopy(Wan_1_3B_Config)
    expected_data = deepcopy(second.dataloader_train)

    first.model.net.fps = 123
    first.dataloader_train.img_size[0] = 32
    first.model_class.config = {"probe": [1]}

    assert second.model.net == Wan_1_3B_Config == expected_net
    assert second.dataloader_train == expected_data
    assert second.model_class.config is None
    assert build_config("fastgen/configs/experiments/WanT2V/config_dmd2.py").model.net == expected_net


def test_teacher_presets_are_independent_of_student_and_other_recipes():
    from fastgen.configs.net import CosmosPredict2_14B_Aggressive_Config

    distill = build_config("fastgen/configs/experiments/CosmosPredict2/config_dmd2_14b.py")
    finetune = build_config("fastgen/configs/experiments/CosmosPredict2/config_sft_14b.py")
    assert distill.model.teacher is not CosmosPredict2_14B_Aggressive_Config
    assert distill.model.teacher is not distill.model.net
    assert distill.model.teacher is not finetune.model.net
    expected = deepcopy(CosmosPredict2_14B_Aggressive_Config)

    distill.model.teacher.model_channels = 1
    assert distill.model.net == finetune.model.net == CosmosPredict2_14B_Aggressive_Config == expected


# Configs that README examples ask the reader to create.
README_EXAMPLE_CONFIG_FILES = {Path("fastgen/configs/experiments/EDM/config_dmd2_fast_small.py")}


def test_readme_experiment_config_paths_exist():
    """Every canonical experiment-config path in a README resolves in this checkout."""
    pattern = re.compile(r"(?:fastgen/)?configs/experiments/[A-Za-z0-9_./-]+\.py")
    references = []
    missing = []

    for readme in sorted(REPO_ROOT.rglob("README.md")):
        for match in pattern.finditer(readme.read_text(encoding="utf-8")):
            reference = match.group()
            config_path = Path(reference if reference.startswith("fastgen/") else f"fastgen/{reference}")
            if config_path in README_EXAMPLE_CONFIG_FILES:
                continue
            references.append((readme, config_path))
            if not (REPO_ROOT / config_path).is_file():
                missing.append(f"{readme.relative_to(REPO_ROOT)}: {config_path}")

    assert references, "No experiment-config paths found in README files"
    assert not missing, "README files reference missing experiment configs:\n" + "\n".join(missing)


def test_student_grid_cfg_config_uses_an_independent_factory():
    first = BaseModelConfig()
    second = BaseModelConfig()

    first.student_grid_cfg.grid_kwargs["shift"] = 5.0
    assert "shift" not in second.student_grid_cfg.grid_kwargs

    overridden = override_config_with_opts(second, ["-", "++student_grid_cfg.grid_kwargs.shift=3.0"])
    assert overridden.student_grid_cfg.grid_kwargs["shift"] == 3.0


def test_legacy_student_grid_cli_override_is_rejected():
    for legacy in ("student_t_grid", "student_sampling"):
        assert not hasattr(BaseModelConfig(), legacy)
    with pytest.raises(ValueError):
        override_config_with_opts(BaseModelConfig(), ["-", "student_t_grid.sample_steps=3"])
    with pytest.raises(ValueError):
        override_config_with_opts(BaseModelConfig(), ["-", "student_sampling.default_steps=3"])


def test_explicit_student_grid_round_trips_through_hydra():
    overridden = override_config_with_opts(
        BaseModelConfig(),
        [
            "-",
            "student_grid_cfg.steps=2",
            "student_grid_cfg.grid_type=explicit",
            "++student_grid_cfg.grid_kwargs.timesteps=[0.999,0.5,0.0]",
        ],
    )

    grid = get_noise_schedule("rf", student_grid_cfg=overridden.student_grid_cfg).student_grid()
    assert grid.tolist() == pytest.approx([0.999, 0.5, 0.0])


def test_native_student_grid_shift_round_trips_through_hydra():
    overridden = override_config_with_opts(
        BaseModelConfig(),
        [
            "-",
            "student_grid_cfg.steps=4",
            "++student_grid_cfg.grid_kwargs.shift=5",
        ],
    )

    schedule = get_noise_schedule("rf", max_t=1.0, student_grid_cfg=overridden.student_grid_cfg)
    assert schedule.student_grid().tolist() == pytest.approx([1.0, 0.9375, 5 / 6, 0.625, 0.0])


def test_sample_t_overrides_round_trip_through_hydra():
    """The command line sets and clears the changes to sample_t_cfg; the recipe supplies every other field."""
    dmd2 = build_config("fastgen/configs/experiments/WanT2V/config_dmd2.py")
    opts = ["-", "++model.fake_score_sample_t_overrides.time_dist_type=shifted_logitnormal"]
    model = override_config_with_opts(dmd2, opts).model
    fake_score_sample_t_cfg = attrs.evolve(model.sample_t_cfg, **model.fake_score_sample_t_overrides)
    assert fake_score_sample_t_cfg == attrs.evolve(dmd2.model.sample_t_cfg, time_dist_type="shifted_logitnormal")
    assert (fake_score_sample_t_cfg.min_t, fake_score_sample_t_cfg.max_t) == (0.001, 0.999)

    onpolicy = build_config("fastgen/configs/experiments/WanT2V/config_anyflow_onpolicy.py")
    assert onpolicy.model.fake_score_sample_t_overrides is not None
    cleared = override_config_with_opts(onpolicy, ["-", "model.fake_score_sample_t_overrides=null"])
    assert cleared.model.fake_score_sample_t_overrides is None

    meanflow = build_config("fastgen/configs/experiments/EDM/config_mf_cifar10.py")
    model = override_config_with_opts(meanflow, ["-", "model.sample_r_overrides.train_p_mean=-3.0"]).model
    assert model.sample_r_overrides == {"train_p_mean": -3.0}


def test_network_guidance_subclass_round_trips_through_hydra():
    """A network's GuidanceConfig subclass survives overrides, which reach its terms at the usual paths."""
    from fastgen.networks.LTX2.network import LTX2GuidanceConfig

    pdd = build_config("fastgen/configs/experiments/LTX2/config_pdd_ltx23.py")
    opts = ["-", "model.guidance_cfg.stg.scale=1.5", "model.guidance_cfg.modality.scale=null"]
    guidance_cfg = override_config_with_opts(pdd, opts).model.guidance_cfg
    assert type(guidance_cfg) is LTX2GuidanceConfig
    assert guidance_cfg.stg.scale == 1.5 and guidance_cfg.stg.skip_self_attn_layers == [28]
    assert guidance_cfg.modality.scale is None and guidance_cfg.cfg.scale == {"video": 4.5, "audio": 7.0}


@pytest.mark.parametrize(
    "module_name",
    [
        "fastgen.configs.experiments.WanT2V.config_anyflow",
        "fastgen.configs.experiments.WanT2V.config_anyflow_onpolicy",
    ],
)
def test_anyflow_configs_sample_on_a_native_shifted_student_grid(module_name):
    """The flow-map student grid carries its own shift instead of explicit timesteps."""
    config = importlib.import_module(module_name).create_config()
    assert config.model.student_grid_cfg.grid_type == "native"
    assert config.model.student_grid_cfg.steps == 4
    assert "timesteps" not in config.model.student_grid_cfg.grid_kwargs
    assert config.model.student_grid_cfg.grid_kwargs["shift"] == 5.0


@pytest.mark.parametrize(
    ("config_file", "shift"),
    [
        ("fastgen/configs/experiments/CogVideoX/config_dmd2.py", 5),
        ("fastgen/configs/experiments/Cosmos3/config_dmd2.py", 5),
        ("fastgen/configs/experiments/CosmosPredict2/config_dmd2.py", 5),
        ("fastgen/configs/experiments/Flux2/config_dmd2.py", 9.9),
        ("fastgen/configs/experiments/Flux2/config_dmd2_image_edit.py", 9.9),
        ("fastgen/configs/experiments/Flux2/config_dmd2_image_edit_klein_base_4b.py", 9.9),
        ("fastgen/configs/experiments/Flux2/config_dmd2_image_edit_klein_base_9b.py", 9.9),
        ("fastgen/configs/experiments/Flux2/config_dmd2_klein_base_4b.py", 9.9),
        ("fastgen/configs/experiments/Flux2/config_dmd2_klein_base_9b.py", 9.9),
        ("fastgen/configs/experiments/LTX2/config_dmd2.py", 5),
        ("fastgen/configs/experiments/LTX2/config_dmd2_ltx23.py", 5),
        ("fastgen/configs/experiments/MiniMaxH3/config_dmd2.py", 12),
        ("fastgen/configs/experiments/QwenImage/config_dmd2.py", 1.714),
    ],
)
def test_non_wan_distillation_configs_use_native_student_grid_shifts(config_file, shift):
    config = build_config(config_file)

    assert config.model.student_grid_cfg.grid_type == "native"
    assert config.model.student_grid_cfg.grid_kwargs == {"shift": shift}


@pytest.mark.parametrize(
    "config_file",
    [
        "fastgen/configs/experiments/Flux/config_ayf.py",
        "fastgen/configs/experiments/QwenImage/config_ayf.py",
    ],
)
def test_recipes_lower_a_raised_network_bound_to_the_training_range(config_file):
    """These networks raise the RF bound to 1.0 for their sampler grids; the recipes distill at 0.999, the top of
    their student grid, with the teacher on the same schedule range."""
    config = build_config(config_file)

    net = config.model.net
    assert net.max_t == 0.999
    if config.model.teacher is None:
        assert config.model.net_adapter.type == "lora"  # the teacher is the adapters-off student
    else:
        assert (config.model.teacher.get("min_t"), config.model.teacher.max_t) == (net.get("min_t"), net.max_t)
    assert (config.model.sample_t_cfg.min_t, config.model.sample_t_cfg.max_t) == (0.001, 0.999)
    assert "max_t" not in config.model.student_grid_cfg.grid_kwargs


def test_cosmos3_dmd2_uses_the_reference_student_grid_and_bounded_loss_times():
    """The sampler reaches pure noise like the reference grid, while DMD2 loss times remain below that endpoint."""
    config = build_config("fastgen/configs/experiments/Cosmos3/config_dmd2.py")
    schedule_kwargs = network_schedule_kwargs(config.model.net)

    assert schedule_kwargs["max_t"] == network_schedule_kwargs(config.model.teacher)["max_t"] == 1.0
    assert (config.model.sample_t_cfg.min_t, config.model.sample_t_cfg.max_t) == (0.001, 0.999)
    schedule = get_noise_schedule("rf", student_grid_cfg=config.model.student_grid_cfg, **schedule_kwargs)
    assert schedule.student_grid().tolist() == pytest.approx([1.0, 15 / 16, 5 / 6, 5 / 8, 0.0])


def test_cosmos3_sft_uses_native_state_precision_and_bf16_compute():
    config = build_config("fastgen/configs/experiments/Cosmos3/config_sft.py")

    assert config.model.precision == "float32"
    assert config.model.precision_model == "bfloat16"
    assert config.model.precision_opt == "float32"


@pytest.mark.parametrize(
    "config_file",
    [
        "fastgen/configs/experiments/QwenImage/config_pdd.py",
        "fastgen/configs/experiments/LTX2/config_pdd_ltx23.py",
    ],
)
def test_pdd_recipes_start_their_grid_at_the_network_bound(config_file):
    """Student grid and teacher sampler both start at sigma 1.0 (pure noise); nothing is clamped."""
    config = build_config(config_file)
    net, teacher = (network_schedule_kwargs(network) for network in (config.model.net, config.model.teacher))

    assert net["max_t"] == teacher["max_t"] == 1.0
    assert "max_t" not in config.model.student_grid_cfg.grid_kwargs


# ---------------------------------------------------------------------------
# Shared PDD recipe properties.
# ---------------------------------------------------------------------------

BASE_PDD_RECIPE_MODULES = [
    "fastgen.configs.experiments.DiT.config_pdd_repa_xl",
    "fastgen.configs.experiments.LTX2.config_pdd_ltx23",
    "fastgen.configs.experiments.MiniMaxH3.config_pdd",
    "fastgen.configs.experiments.QwenImage.config_pdd",
    "fastgen.configs.experiments.WanT2V.config_pdd",
    "fastgen.configs.experiments.WanT2V.config_pdd_14b",
]

BASE_BF16_MODEL_PDD_RECIPE_MODULES = [
    module_name
    for module_name in BASE_PDD_RECIPE_MODULES
    if module_name != "fastgen.configs.experiments.DiT.config_pdd_repa_xl"
]

BASE_WAN_PDD_RECIPE_MODULES = [
    "fastgen.configs.experiments.WanT2V.config_pdd",
    "fastgen.configs.experiments.WanT2V.config_pdd_14b",
]

BASE_MINIMAX_H3_RECIPE_MODULES = [
    "fastgen.configs.experiments.MiniMaxH3.config_dmd2",
    "fastgen.configs.experiments.MiniMaxH3.config_pdd",
    "fastgen.configs.experiments.MiniMaxH3.config_sft",
]

# (module, GRID_SIZE, NFES) of the PDD recipes whose block sizes follow from GRID_SIZE // NFES.
BASE_PDD_RECIPE_GRID_SIZES = [
    ("fastgen.configs.experiments.DiT.config_pdd_repa_xl", 64, (1, 2, 4, 8)),
    ("fastgen.configs.experiments.QwenImage.config_pdd", 128, (2, 4, 8)),
    ("fastgen.configs.experiments.WanT2V.config_pdd", 128, (2, 4, 8)),
    ("fastgen.configs.experiments.WanT2V.config_pdd_14b", 128, (2, 4, 8)),
    ("fastgen.configs.experiments.LTX2.config_pdd_ltx23", 256, (4, 8)),
    ("fastgen.configs.experiments.MiniMaxH3.config_pdd", 32, (4, 8)),
]


@pytest.mark.parametrize("module_name", BASE_PDD_RECIPE_MODULES)
def test_checked_in_pdd_recipes_use_fp32_method_and_decoding_precision(module_name):
    config = importlib.import_module(module_name).create_config()

    assert config.model.precision == "float32"
    assert config.model.precision_decoding is None
    assert not hasattr(config.model.net, "precision_decoding")


@pytest.mark.parametrize("module_name", BASE_BF16_MODEL_PDD_RECIPE_MODULES)
def test_large_model_pdd_recipes_use_bf16_models_with_fp32_optimization(module_name):
    config = importlib.import_module(module_name).create_config()

    assert config.model.precision_model == "bfloat16"
    assert config.model.precision_opt == "float32"


def test_dit_pdd_recipe_uses_fp32_models_with_bf16_autocast():
    config = importlib.import_module("fastgen.configs.experiments.DiT.config_pdd_repa_xl").create_config()

    assert config.model.precision_model is None
    assert config.model.precision_opt is None
    assert config.model.precision_amp == "bfloat16"
    assert config.model.precision_amp_infer == "bfloat16"


@pytest.mark.parametrize("module_name", BASE_WAN_PDD_RECIPE_MODULES)
def test_wan_pdd_recipes_inherit_encoder_precision_default(module_name):
    config = importlib.import_module(module_name).create_config()

    assert "keep_enc_original_prec" not in config.model.net
    assert "keep_enc_original_prec" not in config.model.teacher


@pytest.mark.parametrize("module_name", BASE_MINIMAX_H3_RECIPE_MODULES)
def test_minimax_h3_recipes_offload_fp32_optimizer_state(module_name):
    """Four-rank H3 recipes cannot fit their FP32 Adam state on device."""
    config = importlib.import_module(module_name).create_config()

    assert config.trainer.fsdp is True
    assert config.trainer.ddp is False
    assert config.trainer.fsdp_cpu_offload is True
    assert config.model.precision_opt == "float32"


@pytest.mark.parametrize(("module_name", "grid_size", "nfes"), BASE_PDD_RECIPE_GRID_SIZES)
def test_checked_in_pdd_recipes_derive_grid_and_block_sizes_programmatically(
    module_name,
    grid_size,
    nfes,
):
    config = importlib.import_module(module_name).create_config()

    assert config.model.net.pdd_steps == grid_size
    assert config.model.student_grid_cfg.grid_kwargs["base_steps"] == grid_size
    assert config.model.loss_config.block_size_min == grid_size // max(nfes)
    assert config.model.loss_config.block_size_max == grid_size // min(nfes)


@pytest.mark.parametrize(
    ("module_name", "max_iter", "save_ckpt_iter"),
    [
        ("fastgen.configs.experiments.DiT.config_pdd_repa_xl", 300000, 10000),
        ("fastgen.configs.experiments.QwenImage.config_pdd", 3000, 250),
        ("fastgen.configs.experiments.WanT2V.config_pdd", 250, 25),
        ("fastgen.configs.experiments.WanT2V.config_pdd_14b", 3500, 25),
        ("fastgen.configs.experiments.LTX2.config_pdd_ltx23", 250, 25),
    ],
)
def test_published_pdd_recipes_use_paper_training_lengths_and_checkpoint_cadence(
    module_name,
    max_iter,
    save_ckpt_iter,
):
    config = importlib.import_module(module_name).create_config()

    assert config.trainer.max_iter == max_iter
    assert config.trainer.save_ckpt_iter == save_ckpt_iter


def test_wan_14b_pdd_recipe_disables_unsharded_ema():
    config = importlib.import_module("fastgen.configs.experiments.WanT2V.config_pdd_14b").create_config()

    assert config.model.use_ema is False


def test_dit_sft_trains_on_the_dit_linear_beta_schedule():
    """Facebook's DiT was trained with ADM's linear betas (1e-4..0.02), not SD1.5's scaled-linear ones."""
    config = build_config("fastgen/configs/experiments/DiT/config_sft_dit_xl.py")

    assert config.model.net.schedule_type == "ddpm"
    assert config.model.net.net_pred_type == "eps"


def test_cosmos_predict2_dmd2_v2w_configures_student_and_teacher():
    config = build_config("fastgen/configs/experiments/CosmosPredict2/config_dmd2_v2w.py")

    for network in (config.model.net, config.model.teacher):
        assert network.is_video2world is True
        assert network.num_conditioning_frames == 1


def test_configs_discovered():
    """Guard against the glob silently matching nothing (which would make the
    parametrized test below vacuously pass)."""
    assert CONFIG_FILES, "No experiment configs discovered"


def test_sdxl_sft_disables_encoder_autocast():
    config = build_config("fastgen/configs/experiments/SDXL/config_sft.py")

    assert config.model.precision_amp_enc is None


@pytest.mark.parametrize(
    "config_file",
    [
        "fastgen/configs/experiments/SDXL/config_dmd2.py",
        "fastgen/configs/experiments/SDXL/config_f_distill.py",
        "fastgen/configs/experiments/SDXL/config_kd.py",
    ],
)
def test_sdxl_distillation_trains_in_fp32_without_autocast(config_file):
    """Use the reference entry point's default FP32 denoiser policy; encoder AMP is independent."""
    config = build_config(config_file)

    assert config.model.precision == "float32"
    assert config.model.precision_model is None
    assert config.model.precision_opt is None
    assert config.model.precision_amp is None
    assert config.model.precision_amp_infer is None
    assert config.model.precision_amp_enc is None


@pytest.mark.parametrize(
    "config_file",
    [
        "fastgen/configs/experiments/SD15/config_dmd2.py",
        "fastgen/configs/experiments/SD15/config_f_distill.py",
        "fastgen/configs/experiments/SD15/config_sft.py",
    ],
)
def test_sd15_recipes_train_in_fp32(config_file):
    """Use the reference training entry point's default FP32 denoiser policy."""
    config = build_config(config_file)

    assert config.model.precision == "float32"
    assert config.model.precision_model is None
    assert config.model.precision_opt is None
    assert config.model.precision_amp is None
    assert config.model.precision_amp_enc is None
    assert config.model.grad_scaler_enabled is False


def test_cosmos_predict2_14b_sft_uses_aggressive_sac_at_full_geometry():
    from fastgen.networks.cosmos_predict2.modules import CheckpointMode
    from fastgen.networks.cosmos_predict2.network import CosmosPredict2

    config = build_config("fastgen/configs/experiments/CosmosPredict2/config_sft_14b.py")
    net = config.model.net

    target = net["_target_"]
    if not callable(target):
        target = locate(target)
    assert target is CosmosPredict2
    assert (net.model_channels, net.num_blocks, net.num_heads) == (5120, 36, 40)
    assert net.sac_config.mode == CheckpointMode.AGGRESSIVE
    assert list(config.model.input_shape) == [16, 24, 88, 160]
    assert list(config.dataloader_train.img_size) == [1280, 704]
    assert config.dataloader_train.sequence_length == 93
    assert config.trainer.fsdp is True
    assert (config.model.precision, config.model.precision_model, config.model.precision_opt) == (
        "float32",
        "bfloat16",
        "float32",
    )


def test_cosmos_predict2_14b_dmd2_uses_aggressive_sac_at_480p_geometry():
    from fastgen.networks.cosmos_predict2.modules import CheckpointMode
    from fastgen.networks.cosmos_predict2.network import CosmosPredict2

    config = build_config("fastgen/configs/experiments/CosmosPredict2/config_dmd2_14b.py")

    for model_config in (config.model.net, config.model.teacher):
        target = model_config["_target_"]
        if not callable(target):
            target = locate(target)
        assert target is CosmosPredict2
        assert (model_config.model_channels, model_config.num_blocks, model_config.num_heads) == (5120, 36, 40)
        assert model_config.sac_config.mode == CheckpointMode.AGGRESSIVE

    assert list(config.model.input_shape) == [16, 24, 60, 104]
    assert list(config.dataloader_train.img_size) == [832, 480]
    assert config.dataloader_train.sequence_length == 93
    assert config.trainer.fsdp is True
    assert (config.model.precision, config.model.precision_opt) == ("bfloat16", "float32")
    # Three discriminator heads, at the 2B recipe's relative depths of the 36 blocks.
    assert list(config.model.discriminator.feature_indices) == [17, 26, 35]


@pytest.mark.parametrize(
    ("config_file", "precision_model", "precision_opt"),
    [
        ("fastgen/configs/experiments/EDM2/config_tcm_s.py", None, None),
        ("fastgen/configs/experiments/EDM2/config_tcm_xl.py", "float16", "float32"),
    ],
)
def test_edm2_consistency_recipes_configure_precision(config_file, precision_model, precision_opt):
    """TCM-S uses the default FP32 for its reproduced dropout overflow; TCM-XL retains native FP16 with FP32
    master weights and scaling."""
    config = build_config(config_file)

    assert config.model.precision == "float32"
    assert config.model.precision_model == precision_model
    assert config.model.precision_opt == precision_opt
    assert config.model.precision_amp is None
    assert config.model.grad_scaler_enabled is (precision_model == "float16")
    if config.model.grad_scaler_enabled:
        assert (config.model.grad_scaler_init_scale, config.model.grad_scaler_growth_interval) == (16, 20000)
    assert config.trainer.tf32_conv is False
    assert config.trainer.tf32_matmul is False
    assert config.trainer.fp16_reduced_precision_reduction is False
    if "config_tcm_" in config_file:
        assert dict(config.trainer.checkpointer.pretrained_ckpt_key_map) == {
            name: "net" for name in ("cm_teacher", "net", "ema_1", "ema_5", "ema_10")
        }


@pytest.mark.parametrize(
    "config_file",
    [
        "fastgen/configs/experiments/EDM2/config_cm_s.py",
        "fastgen/configs/experiments/EDM2/config_cm_xl.py",
        "fastgen/configs/experiments/EDM/config_cm_in64.py",
    ],
)
def test_cm_recipes_train_in_fp32(config_file):
    """The CM recipes default to FP32; their comments describe the optional FP16 training."""
    config = build_config(config_file)

    assert config.model.precision == "float32"
    assert config.model.precision_model is None
    assert config.model.precision_opt is None
    assert config.model.precision_amp is None
    assert config.model.grad_scaler_enabled is False
    assert config.trainer.tf32_conv is False
    assert config.trainer.tf32_matmul is False
    assert config.trainer.fp16_reduced_precision_reduction is False


def test_edm2_scm_xl_trains_in_fp16_mode_with_an_fp32_jvp():
    """sCM-XL trains EDM2 in FP16 from FP32 master weights without autocast, and computes the JVP in FP32."""
    config = build_config("fastgen/configs/experiments/EDM2/config_scm_xl.py")

    assert config.model.precision_model == "float16"
    assert config.model.precision_opt == "float32"
    assert config.model.precision_amp is None
    assert config.model.precision_amp_jvp == "float32"


@pytest.mark.parametrize(
    ("config_file", "init_scale"),
    [
        ("fastgen/configs/experiments/EDM/config_sft_edm_in64.py", 65536),
        ("fastgen/configs/experiments/EDM/config_dmd2_in64.py", 32768),
        ("fastgen/configs/experiments/EDM/config_f_distill_in64.py", 32768),
        ("fastgen/configs/experiments/EDM2/config_sft_s.py", 65536),
        ("fastgen/configs/experiments/EDM2/config_sft_xl.py", 65536),
        ("fastgen/configs/experiments/EDM/config_scd_in64.py", 1048576),
        ("fastgen/configs/experiments/EDM2/config_scm_xl.py", 4096),
    ],
)
def test_fp16_recipes_configure_dynamic_loss_scaling(config_file, init_scale):
    """Pin recipe settings; numerical underflow and overflow depend on the loss, batch, and model state."""
    config = build_config(config_file)

    assert (config.model.precision_amp or config.model.precision_model) == "float16"
    assert config.model.precision_opt == "float32"
    assert config.model.grad_scaler_enabled is True
    assert (config.model.grad_scaler_init_scale, config.model.grad_scaler_growth_interval) == (init_scale, 20000)


@pytest.mark.parametrize(
    ("config_file", "sample_type"),
    [
        ("fastgen/configs/experiments/SDXL/config_sft.py", "ode"),
        ("fastgen/configs/experiments/DiT/config_sft_dit_xl.py", "ode"),
        ("fastgen/configs/experiments/DiT/config_sft_sit_xl.py", "sde"),
    ],
)
def test_sft_sample_type_is_method_owned(config_file, sample_type):
    config = build_config(config_file)

    assert config.model.student_sample_type == sample_type
    assert "sample_type" not in config.model.net


def assert_sft_network_samplers_ignore_student_grid_policy(config_files):
    """Assert SFT gives network samplers only the NFE, not a student-grid policy."""
    from fastgen.methods import SFTModel

    forbidden_attributes = {
        "default_student_steps",
        "sample_student_timesteps",
        "sample_student_timesteps_inhom",
        "student_base_steps",
        "student_grid",
        "student_grid_cfg",
        "student_grid_indices",
    }
    sampler_methods = {}

    for config_file in config_files:
        config = build_config(config_file)

        model_target = config.model_class["_target_"]
        if not callable(model_target):
            model_target = locate(model_target)
        assert issubclass(model_target, SFTModel), f"{config_file}: expected an SFT model, got {model_target}"

        net_target = config.model.net["_target_"]
        if not callable(net_target):
            net_target = locate(net_target)
        assert inspect.isclass(net_target), f"{config_file}: network target must be a class, got {net_target}"

        for cls in net_target.__mro__:
            for method_name in ("sample", "sample_scheduler_and_grid"):
                method = cls.__dict__.get(method_name)
                if method is not None and inspect.isfunction(method):
                    sampler_methods[(method.__module__, method.__qualname__)] = (method, config_file)

    assert sampler_methods, "No SFT network sampling methods were discovered"
    violations = []
    for (module_name, qualname), (method, config_file) in sorted(sampler_methods.items()):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        referenced = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} & forbidden_attributes
        if referenced:
            violations.append(f"{config_file}: {module_name}.{qualname} references {sorted(referenced)}")

    assert not violations, (
        "SFT network samplers must derive grid values independently of student_grid_cfg; "
        "only SFTModel may read its step count:\n" + "\n".join(violations)
    )


def test_sft_network_samplers_ignore_student_grid_policy():
    sft_config_files = [path for path in CONFIG_FILES if Path(path).name.startswith("config_sft")]
    assert sft_config_files, "No SFT experiment configs were discovered"
    assert_sft_network_samplers_ignore_student_grid_policy(sft_config_files)


@pytest.mark.parametrize(
    "config_file",
    [
        "fastgen/configs/experiments/Flux/config_sft.py",
        "fastgen/configs/experiments/SD15/config_sft.py",
    ],
)
def test_flux_and_sd15_sft_disable_shared_encoder_autocast(config_file):
    config = build_config(config_file)

    assert config.model.precision_amp_enc is None


def assert_wan22_14b_dmd2_offloads_fp32_optimizer_state(config):
    """Four-rank Wan2.2 MoE DMD2 cannot keep both FP32 Adam states on device."""
    assert config.trainer.fsdp is True
    assert config.trainer.ddp is False
    assert config.trainer.fsdp_cpu_offload is True
    assert config.model.meta_init is True
    assert config.model.precision == "bfloat16"
    assert config.model.precision_opt == "float32"


@pytest.mark.parametrize(
    "config_file",
    [
        "fastgen/configs/experiments/WanI2V/config_dmd2_wan22_14b.py",
        "fastgen/configs/experiments/WanT2V/config_dmd2_wan22_14b.py",
    ],
)
def test_wan22_14b_dmd2_offloads_fp32_optimizer_state(config_file):
    assert_wan22_14b_dmd2_offloads_fp32_optimizer_state(build_config(config_file))


def assert_wan22_5b_native_settings(config, resample_video=False):
    """Keep network, video loaders, and optimizer precision on the validated 5B defaults."""
    assert config.trainer.fsdp is True
    assert config.trainer.ddp is False
    if hasattr(config.model, "fake_score_optimizer"):
        assert config.model.precision == "bfloat16"
    else:
        # SFT keeps the data and sampler state in fp32 and only the network in bf16.
        assert config.model.precision == "float32"
        assert config.model.precision_model == "bfloat16"
    assert config.model.precision_opt == "float32"
    is_causal = "chunk_size" in config.model.net
    latent_frames = 30 if is_causal else 31
    assert list(config.model.input_shape) == [48, latent_frames, 44, 80]
    assert config.model.net.fps == 24
    if is_causal:
        assert config.model.net.chunk_size == 3
        assert config.model.net.total_num_frames == latent_frames
        assert latent_frames % config.model.net.chunk_size == 0
        # Exercise the training sampler, not just config construction.
        times = get_noise_schedule("rf").sample_t_inhom(
            2,
            seq_len=latent_frames,
            chunk_size=config.model.net.chunk_size,
            device="cpu",
            **config.model.sample_t_cfg.sample_t_kwargs(),
        )
        assert times.shape == (2, latent_frames)
        assert times.isfinite().all()
        assert (times >= config.model.sample_t_cfg.min_t).all()
        assert (times <= config.model.sample_t_cfg.max_t).all()
        chunks = times.reshape(2, -1, config.model.net.chunk_size)
        assert (chunks == chunks[..., :1]).all()
    for loader in (config.dataloader_train, config.dataloader_val):
        if loader is not None:
            assert tuple(loader.img_size) == (1280, 704)
            assert loader.sequence_length == (117 if is_causal else 121)
            assert loader.get("target_fps") == (config.model.net.fps if resample_video else None)

    if hasattr(config.model, "fake_score_optimizer"):
        reference = build_config("fastgen/configs/experiments/WanT2V/config_dmd2_14b.py")
        for name in ("net_optimizer", "fake_score_optimizer", "discriminator_optimizer"):
            optimizer = getattr(config.model, name)
            assert optimizer == getattr(reference.model, name)
            assert optimizer.optim_type == "adamw"
            assert optimizer.lr == 3e-6
            assert tuple(optimizer.betas) == (0.9, 0.999)
            assert optimizer.weight_decay == 0.01
        assert config.trainer.callbacks.grad_clip.grad_norm == 10.0
        assert config.trainer.callbacks.grad_clip_fake_score.grad_norm is None
        assert config.trainer.callbacks.grad_clip_discriminator.grad_norm is None
    else:
        # Keep the 1e-5 fine-tuning rate when adjusting precision or native video geometry.
        assert config.model.net_optimizer.lr == 1e-5


@pytest.mark.parametrize(
    "config_file",
    [path for path in CONFIG_FILES if "wan22_5b" in path],
    ids=config_id,
)
def test_wan22_5b_native_settings(config_file):
    assert_wan22_5b_native_settings(build_config(config_file))


def test_wan22_5b_fps_change_is_scoped_to_5b_presets():
    from fastgen.configs import net

    for name in (
        "Wan_1_3B_Config",
        "Wan21_T2V_14B_Config",
        "Wan21_I2V_14B_480P_Config",
        "Wan22_T2V_14B_Config",
        "Wan22_I2V_14B_Config",
        "CausalWan21_I2V_14B_480P_Config",
    ):
        assert getattr(net, name).fps == 16


def assert_lingbot_world_sft_offloads_fp32_optimizer_state(config_file):
    """LingBot SFT must offload its FP32 optimizer state before world-8 routing."""
    config = build_config(config_file)

    assert config.trainer.fsdp is True
    assert config.trainer.ddp is False
    assert config.trainer.fsdp_cpu_offload is True
    assert config.model.meta_init is True
    assert config.model.precision == "float32"
    assert config.model.precision_model == "bfloat16"
    assert config.model.precision_opt == "float32"


def test_lingbot_world_sft_offloads_fp32_optimizer_state():
    assert_lingbot_world_sft_offloads_fp32_optimizer_state(
        "fastgen/configs/experiments/WanI2V/config_sft_lingbot_world.py"
    )


@pytest.mark.parametrize(
    ("config_file", "field", "expected"),
    [
        (
            "fastgen/configs/experiments/DiT/config_sft_dit_xl.py",
            "pretrained_model_path",
            "DIT_XL_2_256_URL",
        ),
        (
            "fastgen/configs/experiments/DiT/config_sft_sit_xl.py",
            "pretrained_model_path",
            "SIT_XL_2_256_URL",
        ),
        (
            "fastgen/configs/experiments/DiT/config_pdd_repa_xl.py",
            "pretrained_model_path",
            "SIT_XL_2_REPA_256_URL",
        ),
        (
            "fastgen/configs/experiments/WanT2V/config_sf.py",
            "pretrained_student_net_path",
            "SELF_FORCING_ODE_INIT_URL",
        ),
        (
            "fastgen/configs/experiments/WanT2V/config_causvid.py",
            "pretrained_student_net_path",
            "SELF_FORCING_ODE_INIT_URL",
        ),
    ],
)
def test_single_file_pretrained_checkpoints_use_shared_http_cache(config_file, field, expected):
    from fastgen.configs import net as net_configs

    config = build_config(config_file)

    assert getattr(config.model, field) == getattr(net_configs, expected)
    assert getattr(config.model, field).startswith("https://")


@pytest.mark.parametrize(
    "preset_name",
    [
        "VACE_Wan_1_3B_Config",
        "VACE_Wan_14B_Config",
        "CausalVACE_Wan_1_3B_Config",
        "CausalVACE_Wan_14B_Config",
    ],
)
def test_vace_presets_use_auto_downloaded_depth_checkpoint(preset_name):
    from fastgen.configs import net as net_configs

    preset = getattr(net_configs, preset_name)

    assert preset.depth_model_path == net_configs.DEPTH_ANYTHING_V2_VITL_URL
    assert preset.depth_model_path.startswith("https://")


@pytest.mark.parametrize("config_file", CONFIG_FILES, ids=config_id)
def test_config_builds(config_file):
    """``create_config()`` must import cleanly and return a ``BaseConfig``."""
    config = build_config(config_file)
    assert isinstance(
        config, BaseConfig
    ), f"{config_file}: create_config() returned {type(config)!r}, expected BaseConfig"
    assert_discriminator_callback_matches_config(config, config_file)
    assert_ema_callback_matches_config(config, config_file)
    assert_parallelism_is_unambiguous(config, config_file)
    assert_loaders_match_input_shape(config, config_file)
    assert_validation_settings_have_a_loader(config, config_file)


def _shared_loader_presets() -> dict[str, object]:
    """Module-level ``LazyCall`` loader presets that configs must copy before mutating."""
    import fastgen.configs.data as data_presets

    return {name: value for name, value in vars(data_presets).items() if name.endswith("LoaderConfig")}


@pytest.mark.parametrize("config_file", CONFIG_FILES, ids=config_id)
def test_config_does_not_alias_shared_loader_presets(config_file):
    """A config that assigns a preset by reference mutates it for every config built later in the process."""
    config = build_config(config_file)
    presets = _shared_loader_presets()
    assert presets, "no loader presets discovered"
    for field in ("dataloader_train", "dataloader_val"):
        loader = getattr(config, field, None)
        aliased = [name for name, preset in presets.items() if loader is preset]
        assert not aliased, f"{config_file}: {field} aliases the shared preset {aliased[0]}; use copy.deepcopy"


def assert_loaders_leave_the_start_offset_to_the_trainer(data_presets, config_files: list[str]):
    """The trainer sets a loader's sampler_start_idx only when it is None: the next unseen sample for training,
    0 for validation."""
    suffixes = ("LoaderConfig", "Loader_Config")
    presets = {name: value for name, value in vars(data_presets).items() if name.endswith(suffixes)}
    assert presets, "no loader presets discovered"
    fixed = [name for name, preset in presets.items() if preset.get("sampler_start_idx", 0) is not None]
    for config_file in config_files:
        config = build_config(config_file)
        for field in ("dataloader_train", "dataloader_val"):
            loader = getattr(config, field, None)
            if loader is not None and loader.get("sampler_start_idx", 0) is not None:
                fixed.append(f"{config_id(config_file)}:{field}")
    assert not fixed, f"loaders that fix sampler_start_idx instead of leaving it to the trainer: {fixed}"


def test_loaders_leave_the_start_offset_to_the_trainer():
    import fastgen.configs.data as data_presets

    assert_loaders_leave_the_start_offset_to_the_trainer(data_presets, CONFIG_FILES)


def _configured_fps(node):
    """Frame rate a lazy network config resolves to: the explicit kwarg, else the target's default."""
    if "fps" in node:
        return float(node["fps"])
    target = node["_target_"]
    if not callable(target):
        target = locate(target)
    parameter = inspect.signature(target).parameters.get("fps")
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return float(parameter.default)


def test_teacher_configs_share_the_student_frame_rate():
    """Conditions are prepared with the student network's ``fps`` (Cosmos3 bakes it into its prompt
    templates and rotary positions), so a separately configured teacher must resolve to the same rate."""
    mismatches = []
    for config_file in CONFIG_FILES:
        config = build_config(config_file)
        if config.model.teacher is None:
            continue  # the teacher is instantiated from the student config itself
        student_fps, teacher_fps = _configured_fps(config.model.net), _configured_fps(config.model.teacher)
        if student_fps is not None and teacher_fps is not None and student_fps != teacher_fps:
            mismatches.append(f"{config_id(config_file)}: net fps {student_fps} vs teacher fps {teacher_fps}")
    assert not mismatches, "teacher and student frame rates differ:\n" + "\n".join(mismatches)


def _optimized_storage_precision(model_config) -> str:
    """Storage of optimized modules and live EMA weights, resolved like ``FastGenModel._resolve_precisions``."""
    for precision in (model_config.precision_opt, model_config.precision_model):
        if precision is not None:
            return precision
    return model_config.precision


@pytest.mark.parametrize("config_file", SHIPPED_CONFIG_FILES, ids=config_id)
def test_ema_weights_are_not_stored_in_16_bit(config_file):
    """A 16-bit EMA rounds each ``(1 - beta)``-weighted update away, so it never leaves its initialization."""
    model_config = build_config(config_file).model
    if not model_config.use_ema:
        return
    storage = _optimized_storage_precision(model_config)
    assert storage not in (
        "float16",
        "bfloat16",
    ), f"{config_file}: model.use_ema keeps the EMA in {storage} storage; set model.precision_opt='float32'"


# Negative prompt each family's raw-text loaders inject: the one the model samples with at inference.
NEGATIVE_PROMPT_PRESETS = {
    "Cosmos3": "neg_prompt_cosmos3",
    "CosmosPredict2": "neg_prompt_cosmos",
    "LTX2": "neg_prompt_ltx2",
    "LTX2I2V": "neg_prompt_ltx2",
    "WanI2V": "neg_prompt_wan",
    "WanT2V": "neg_prompt_wan",
    "WanV2V": "neg_prompt_wan",
}


def _negative_prompt_preset(config_file: str) -> str | None:
    if "lingbot_world" in Path(config_file).stem:
        return "neg_prompt_lingbot_world"
    return NEGATIVE_PROMPT_PRESETS.get(Path(config_file).parent.name)


@pytest.mark.parametrize(
    "config_file", [path for path in CONFIG_FILES if _negative_prompt_preset(path) is not None], ids=config_id
)
def test_training_negative_prompts_match_the_model(config_file):
    config = build_config(config_file)
    for field in ("dataloader_train", "dataloader_val"):
        loader = getattr(config, field)
        presets_map = None if loader is None else loader.get("presets_map")
        if presets_map is not None:
            assert presets_map.get("neg_condition_raw") == _negative_prompt_preset(
                config_file
            ), f"{config_file}: {field}.presets_map={dict(presets_map)}"


def test_14b_fsdp_recipes_initialize_on_the_meta_device():
    """Only rank 0 constructs and loads a 14B FSDP model; the other ranks receive its weights when sharded."""
    missing = []
    for config_file in CONFIG_FILES:
        if "14b" not in Path(config_file).stem:
            continue
        config = build_config(config_file)
        if config.trainer.fsdp and not config.model.meta_init:
            missing.append(config_id(config_file))
    assert not missing, f"14B FSDP recipes without model.meta_init: {missing}"


def test_recipes_write_to_distinct_run_directories():
    """Two recipes sharing a run directory would resume each other's checkpoints.

    A config without its own ``log_config.name`` runs under its file name.
    """
    runs = {}
    for config_file in CONFIG_FILES:
        log_config = build_config(config_file).log_config
        runs.setdefault((log_config.project, log_config.group, log_config.name), []).append(config_id(config_file))
    shared = [f"{'/'.join(run)}: {files}" for run, files in runs.items() if len(files) > 1]
    assert not shared, "recipes share a run directory:\n" + "\n".join(shared)


@pytest.mark.parametrize("config_file", CONFIG_FILES, ids=config_id)
def test_loaded_recipes_carry_a_run_name(config_file, monkeypatch):
    """``load_config`` names every recipe after its file; no recipe sets its own name."""
    monkeypatch.chdir(REPO_ROOT)
    loaded = load_config(config_file)
    assert loaded.log_config.name == build_config(config_file).log_config.name == Path(config_file).stem


def test_unknown_optimizer_type_is_rejected():
    import torch

    from fastgen.configs.opt import get_optimizer

    with pytest.raises(ValueError, match="sgd"):
        get_optimizer(torch.nn.Linear(2, 2), optim_type="sgd")


def test_edm_cifar10_sft_keeps_the_augment_labels_of_the_pretrained_edm_model():
    """The EDM checkpoint embeds 9 augment labels: x/y flips, scale, rotation, anisotropy, and translation."""
    config = build_config("fastgen/configs/experiments/EDM/config_sft_edm_cifar10.py")
    pipe = config.trainer.augment_pipe
    geometric = ("xflip", "yflip", "rotate_int", "translate_int", "scale", "rotate_frac", "aniso", "translate_frac")
    color = ("brightness", "contrast", "lumaflip", "hue", "saturation")
    enabled = {name for name in geometric + color if pipe.get(name, 0)}
    assert enabled == {"xflip", "yflip", "scale", "rotate_frac", "aniso", "translate_frac"}
    assert pipe.p == 0.12
    assert config.model.net.backbone.augment_dim == 9
    assert not config.dataloader_train.get("xflip", False)


def test_composite_discriminators_state_their_feature_indices_per_modality():
    """A modality whose own ``feature_indices`` differ from the composite's is overridden with a warning."""
    from fastgen.networks.discriminators import CompositeDiscriminator

    mismatches = []
    for config_file in CONFIG_FILES:
        discriminator = getattr(build_config(config_file).model, "discriminator", None)
        if discriminator is None or discriminator.get("feature_indices") is None:
            continue
        target = discriminator["_target_"]
        if (target if callable(target) else locate(target)) is not CompositeDiscriminator:
            continue
        for key, modality in discriminator.modality_discs.items():
            if set(modality.get("feature_indices") or ()) != set(discriminator.feature_indices):
                mismatches.append(f"{config_id(config_file)}: {key}")
    assert not mismatches, f"modality discriminators without the composite feature_indices: {mismatches}"


def test_cm_end_points_start_at_the_network_min_t():
    """``CMModel`` warns at every start when ``sample_t_cfg.min_r`` lies below the network's ``min_t``."""
    below = []
    for config_file in SHIPPED_CONFIG_FILES:
        model = build_config(config_file).model
        min_r = getattr(model.sample_t_cfg, "min_r", None)
        if min_r is None:
            continue
        target = model.net["_target_"]
        if not callable(target):
            target = locate(target)
        schedule_type = model.net.get("schedule_type", inspect.signature(target).parameters["schedule_type"].default)
        schedule_kwargs = {key: model.net[key] for key in ("min_t", "max_t") if key in model.net}
        min_t = get_noise_schedule(schedule_type, **schedule_kwargs).min_t
        if min_r < min_t:
            below.append(f"{config_id(config_file)}: min_r {min_r} < min_t {min_t}")
    assert not below, "CM end points below the network's min_t:\n" + "\n".join(below)
