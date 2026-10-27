# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig
from fastgen.configs.methods.config_dmd2 import create_config
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.trainer import Trainer, should_load_after_fsdp
from fastgen.callbacks.callback import Callback
from fastgen.configs.instantiate import LazyCall as L, instantiate
from fastgen.methods.model import FastGenModel
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
    EMA_CALLBACK,
    GPUStats_CALLBACK,
    TrainProfiler_CALLBACK,
    ParamCount_CALLBACK,
)
from tests.utils import RunIf, run_distributed_test


class SyntheticLoader:
    """Minimal loader that yields synthetic data with the given shape (e.g. (3, 2, 2) for C, H, W)."""

    def __init__(
        self,
        shape: tuple[int, ...],
        batch_size: int = 1,
        sampler_start_idx: int = 0,
        condition_dim: int = 10,
        **kwargs,
    ):
        self.shape = tuple(shape)
        self.batch_size = batch_size
        self.sampler_start_idx = sampler_start_idx
        self.condition_dim = condition_dim

    def __iter__(self):
        while True:
            real = torch.randn(self.batch_size, *self.shape, dtype=torch.float32)
            condition = torch.zeros(self.batch_size, self.condition_dim, dtype=torch.float32)
            condition[:, 0] = 1.0
            neg_condition = torch.zeros(self.batch_size, self.condition_dim, dtype=torch.float32)
            idx = torch.zeros(self.batch_size, dtype=torch.int64)
            yield {
                "real": real,
                "condition": condition,
                "neg_condition": neg_condition,
                "idx": idx,
            }


class AccumEndCounterCallback(Callback):
    def __init__(self):
        self.count = 0

    def on_training_accum_step_end(self, *args, **kwargs):
        self.count += 1


def test_validation_carry_advances_while_loading_and_preprocessing_every_step():
    """Every validation step preprocesses fresh data before carry replacement and logging."""
    from fastgen.callbacks.carry import CARRY_OUTPUT_KEY, CarryCallback

    class CountingLoader:
        def __init__(self):
            self.loads = 0

        def __iter__(self):
            for idx in range(3):
                self.loads += 1
                yield {"condition_raw": [f"prompt {idx}"], "idx": [idx]}

    class Preprocessors:
        def __init__(self):
            self.calls = 0
            self.conditions = []
            self.preprocessors = {}

        def build_conditioning(self, data, input_shape=None, precision=None):
            self.calls += 1
            raw = data["condition_raw"]
            self.conditions.append(raw)
            prompt_idx = int(raw[0].rsplit(maxsplit=1)[-1])
            return {"condition": torch.tensor([prompt_idx], dtype=torch.float32)}

    class ValidationModel:
        device = torch.device("cpu")
        precision = torch.float32
        precision_amp_enc = None
        input_shape = [1]
        state_fields = FastGenModel.state_fields
        prepare_batch = FastGenModel.prepare_batch

        def __init__(self):
            self.steps = []
            self.net_preprocessors = Preprocessors()

        @staticmethod
        def reshard_fsdp_modules(preprocessors_only=False):
            del preprocessors_only

        @staticmethod
        def autocast():
            return nullcontext()

        def single_train_step(self, data, iteration):
            step = data.get("step_carried", 0)
            self.steps.append((step, data["condition"].item()))
            if step == 2:
                carry = {"step_carried": 3}
            else:
                carry = {
                    "condition": data["condition"],
                    "idx": data["idx"],
                    "state_carried": torch.tensor([step + 1.0]),
                    "step_carried": step + 1,
                }
            return {"total_loss": torch.tensor(float(step))}, {CARRY_OUTPUT_KEY: carry}

    class ValidationCallbacks:
        def __init__(self):
            self.carry = CarryCallback(enable_validation=True)
            self.wandb_steps = []

        def on_validation_begin(self, *args, **kwargs):
            self.carry.on_validation_begin(*args, **kwargs)

        def on_validation_step_begin(self, *args, **kwargs):
            self.carry.on_validation_step_begin(*args, **kwargs)

        def on_validation_step_end(self, *args, **kwargs):
            self.wandb_steps.append(kwargs["step"])
            self.carry.on_validation_step_end(*args, **kwargs)

        def on_validation_end(self, *args, **kwargs):
            self.carry.on_validation_end(*args, **kwargs)

    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            global_vars_val=[{"MAX_VAL_STEPS": 3}],
            val_seed=123,
        )
    )
    trainer.callbacks = ValidationCallbacks()
    trainer.fsdp_context = None
    model = ValidationModel()
    loader = CountingLoader()

    trainer.validate(model, model, loader, iteration=10)

    assert model.steps == [(0, 0.0), (1, 0.0), (2, 0.0)]
    assert model.net_preprocessors.calls == 3
    assert model.net_preprocessors.conditions == [["prompt 0"], ["prompt 1"], ["prompt 2"]]
    assert trainer.callbacks.wandb_steps == [0, 1, 2]
    assert loader.loads == 3
    assert trainer.callbacks.carry._validation_slot is None


def test_validation_calls_model_reshard_after_validation():
    """Validation delegates its phase-boundary cleanup to the model lifecycle."""
    reshard = MagicMock()
    model = SimpleNamespace(
        device=torch.device("cpu"),
        reshard_fsdp_modules=reshard,
    )
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            global_vars_val=[{}],
            val_seed=123,
        )
    )
    trainer.callbacks = MagicMock()

    trainer.validate(model, model, [], iteration=10)

    reshard.assert_called_once_with()


def test_validation_step_limit_does_not_read_an_unused_batch():
    def batches():
        yield {"idx": [0]}
        raise AssertionError("read beyond MAX_VAL_STEPS")

    model = SimpleNamespace(
        device=torch.device("cpu"),
        prepare_batch=lambda batch: batch,
        autocast=nullcontext,
        single_train_step=lambda *_: ({}, {}),
        reshard_fsdp_modules=MagicMock(),
    )
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(global_vars_val=[{"MAX_VAL_STEPS": 1}], val_seed=123))
    trainer.callbacks = MagicMock()
    trainer.validate(model, model, batches())
    trainer.callbacks.on_validation_step_end.assert_called_once()


def test_zero_validation_steps_does_not_initialize_the_loader():
    class UnusedLoader:
        def __iter__(self):
            raise AssertionError("initialized an unused validation loader")

    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(global_vars_val=[{"MAX_VAL_STEPS": 0}], val_seed=123))
    trainer.callbacks = MagicMock()
    model = SimpleNamespace(device=torch.device("cpu"), reshard_fsdp_modules=MagicMock())

    trainer.validate(model, model, UnusedLoader())

    trainer.callbacks.on_validation_step_begin.assert_not_called()
    trainer.callbacks.on_validation_end.assert_called_once()


def _synced_trainer_iterator_contract(rank, world_size, num_workers):
    """Real rank-local WDS streams must stop the driver before its next model collective."""
    import io
    from pathlib import Path
    import tarfile
    import torch.distributed as dist

    import fastgen.trainer as trainer_module
    from fastgen.datasets.wds_dataloaders import WDSLoader

    device = torch.device("cuda", rank)
    results = {}

    class BrokenIterable:
        def __iter__(self):
            if rank == world_size - 1:
                raise ValueError("cannot initialize loader")
            return iter([{}])

    with pytest.MonkeyPatch.context() as patch, tempfile.TemporaryDirectory() as directory:
        patch.setattr(trainer_module.ddp, "model_to_ddp", lambda model: model)
        patch.setattr(trainer_module, "warn_on_rng_state_drift", lambda *args, **kwargs: None)
        patch.setattr(trainer_module.basic_utils, "clear_gpu_memory", lambda: None)
        patch.setattr(trainer_module, "time", SimpleNamespace(sleep=lambda _: None))

        cases = [
            ("repeating", 1, True, True, None, "", 4),
            ("finite_at_step_limit", 8, False, True, None, "", 4),
            ("unequal", 3, False, False, ValueError, "different numbers of items", 1),
            ("dropped_tail", 3, False, True, StopIteration, "", 1),
            ("empty_rank", 1, False, False, ValueError, "different numbers of items", 0),
            ("decode_error", 4, False, True, RuntimeError, "rank 1 failed", 0),
            ("balanced_eof", 2, False, True, StopIteration, "", 1),
            ("initialization_error", 0, False, True, RuntimeError, "cannot initialize loader", 0),
        ]
        for name, size, repeat, drop_last, expected_error, message, expected_steps in cases:
            shard_dir = Path(directory) / name
            shard_dir.mkdir()
            shard = shard_dir / "00000.tar"
            with tarfile.open(shard, "w") as archive:
                for index in range(size):
                    # Fail on rank 1's first sample, independent of batching lookahead.
                    content = b"\xff" if name == "decode_error" and index == 1 else str(index).encode()
                    info = tarfile.TarInfo(f"{index:05d}.txt")
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))

            loader = (
                BrokenIterable()
                if name == "initialization_error"
                else WDSLoader(
                    datatags=[f"WDS:{shard_dir}"],
                    batch_size=1,
                    num_workers=num_workers,
                    deterministic=True,
                    strict=True,
                    train=repeat,
                    drop_last=drop_last,
                    sampler_start_idx=0,
                    key_map={"condition_raw": "txt"},
                )
            )
            loader_config = SimpleNamespace(batch_size=1, sampler_start_idx=0)
            trainer = object.__new__(Trainer)
            trainer.config = SimpleNamespace(
                trainer=SimpleNamespace(
                    ddp=True,
                    fsdp=False,
                    resume=False,
                    seed=0,
                    grad_accum_rounds=2,
                    max_iter=2,
                    validation_iter=100,
                    save_ckpt_iter=100,
                    augment_pipe=None,
                    checkpointer=SimpleNamespace(pretrained_ckpt_path="", pretrained_ckpt_key_map={}),
                    global_vars_val=[{"MAX_VAL_STEPS": 1}],
                    val_seed=0,
                ),
                model=SimpleNamespace(meta_init=False),
                dataloader_train=loader_config,
                dataloader_val=None,
            )
            trainer.callbacks = MagicMock()
            trainer.auto_resume = MagicMock()
            trainer.auto_resume.get_resume_details.return_value = None
            trainer.auto_resume_exit = MagicMock(return_value=False)
            trainer.save_checkpoint = MagicMock()
            patch.setattr(trainer_module, "instantiate", lambda cfg: loader if cfg is loader_config else cfg)
            model = MagicMock()
            model.device = device
            model.prepare_batch.side_effect = lambda data, *args: data
            steps = []

            def train_step(_model_ddp, _model, data, iteration, accum_iter):
                dist.all_reduce(torch.ones(1, device=device))
                steps.append((iteration, accum_iter))
                return {}, {}

            trainer.train_step = train_step
            caught = None
            try:
                trainer.run(model)
            except Exception as error:
                caught = error
            matches = caught is None if expected_error is None else isinstance(caught, expected_error)
            results[name] = (
                matches
                and (not message or message in str(caught))
                and steps == [(1, 0), (1, 1), (2, 0), (2, 1)][:expected_steps]
            )
            if not results[name]:
                print(f"rank={rank}, case={name}, steps={steps}, error={caught!r}", flush=True)

        # islice must not initialize the underlying iterable outside the synchronized guard.
        try:
            trainer.validate(model, model, BrokenIterable())
        except RuntimeError as error:
            results["validation_initialization_error"] = "cannot initialize loader" in str(error)
        else:
            results["validation_initialization_error"] = False

    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)
    return {name: all(result[name] for result in all_results) for name in results}


@RunIf(min_gpus=2)
@pytest.mark.parametrize("num_workers", [0, 3])
def test_training_and_validation_iterators_fail_together(num_workers):
    result = run_distributed_test(_synced_trainer_iterator_contract, world_size=2, num_workers=num_workers, timeout=180)
    assert result is not None
    assert all(result.values()), result


@pytest.mark.parametrize("fsdp_enabled,path", [(False, "checkpoint.pth"), (True, None), (True, "")])
def test_load_order_skips_format_probe_when_unused(monkeypatch, fsdp_enabled, path):
    def unexpected_probe(*args, **kwargs):
        pytest.fail("Unused checkpoints must not be probed or broadcast")

    monkeypatch.setattr("fastgen.trainer.is_dcp_checkpoint", unexpected_probe)
    monkeypatch.setattr("fastgen.trainer.broadcast_rank0_result", unexpected_probe)
    assert not should_load_after_fsdp(path, ("net",), fsdp_enabled=fsdp_enabled)


def _load_order_worker(rank, world_size):
    from unittest.mock import patch

    for sharded in (False, True):

        def probe(path, model_keys):
            assert rank == 0
            assert path == "checkpoint.pth"
            assert tuple(model_keys) == ("saved_net",)
            return sharded

        with patch("fastgen.trainer.is_dcp_checkpoint", side_effect=probe) as read:
            assert should_load_after_fsdp("checkpoint.pth", ("saved_net",), fsdp_enabled=True) is sharded
        assert read.call_count == (1 if rank == 0 else 0)

    with patch("fastgen.trainer.is_dcp_checkpoint", side_effect=OSError("checkpoint unavailable")) as read:
        with pytest.raises(RuntimeError, match="checkpoint unavailable"):
            should_load_after_fsdp("checkpoint.pth", ("saved_net",), fsdp_enabled=True)
    assert read.call_count == (1 if rank == 0 else 0)
    return True


@RunIf(min_gpus=2)
def test_load_order_shares_rank0_decision_and_errors():
    assert run_distributed_test(_load_order_worker, world_size=2, timeout=120)


@pytest.mark.parametrize(
    ("path", "fsdp_enabled", "is_dcp", "expected_events"),
    [
        ("/tmp/checkpoints/0000200.pth", True, True, ["fsdp_context", "on_train_begin", "stage_load"]),
        ("/tmp/checkpoints/0000200.pth", True, False, ["stage_load", "fsdp_context", "on_train_begin"]),
        ("/tmp/checkpoints/0000200.pth", False, True, ["stage_load", "on_train_begin", "ddp_wrap"]),
        ("/tmp/checkpoints/0000200.pth", False, False, ["stage_load", "on_train_begin", "ddp_wrap"]),
        (None, True, False, ["fsdp_context", "on_train_begin"]),
        ("", True, False, ["fsdp_context", "on_train_begin"]),
        (None, False, False, ["on_train_begin", "ddp_wrap"]),
        ("", False, False, ["on_train_begin", "ddp_wrap"]),
    ],
    ids=[
        "fsdp-dcp",
        "fsdp-monolithic",
        "ddp-dcp",
        "ddp-monolithic",
        "fsdp-none",
        "fsdp-empty",
        "ddp-none",
        "ddp-empty",
    ],
)
def test_pretrained_checkpoint_load_order(path, fsdp_enabled, is_dcp, expected_events, monkeypatch):
    """Monolithic weights load before wrapping; FSDP loads DCP directly into the prepared shards."""

    class StopAfterPreparation(Exception):
        pass

    events = []
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            fsdp=fsdp_enabled,
            ddp=not fsdp_enabled,
            fsdp_min_num_params=10,
            fsdp_cpu_offload=True,
            fsdp_sharding_group_size=None,
            checkpointer=SimpleNamespace(
                pretrained_ckpt_path=path,
                pretrained_ckpt_key_map={"net": "net_ema"},
            ),
        ),
        model=SimpleNamespace(
            meta_init=True,
            precision_amp=None,
            precision="bfloat16",
        ),
    )
    trainer = object.__new__(Trainer)
    trainer.config = config
    trainer.callbacks = MagicMock()

    model = MagicMock()
    fsdp_context = object()

    def record_train_begin(**kwargs):
        assert kwargs == {"fsdp_context": fsdp_context if fsdp_enabled else None}
        events.append("on_train_begin")

    model.on_train_begin.side_effect = record_train_begin
    model.apply_torch_compile.side_effect = StopAfterPreparation

    import fastgen.trainer as trainer_module

    probe = MagicMock(return_value=is_dcp)
    monkeypatch.setattr(trainer_module, "is_dcp_checkpoint", probe)
    monkeypatch.setattr(trainer_module, "world_size", lambda: 2)
    monkeypatch.setattr(trainer_module, "synchronize", lambda: None)
    monkeypatch.setattr(
        trainer_module.fsdp,
        "create_fsdp_context",
        lambda **_: events.append("fsdp_context") or fsdp_context,
    )
    monkeypatch.setattr(trainer_module.ddp, "model_to_ddp", lambda model: events.append("ddp_wrap") or model)

    def record_load(_model):
        assert _model is model
        assert bool(path)
        events.append("stage_load")

    trainer.load_pretrained_ckpt = record_load

    with pytest.raises(StopAfterPreparation):
        trainer.run(model)

    assert events == expected_events
    if fsdp_enabled and path:
        probe.assert_called_once()
        assert probe.call_args.args[0] == path
        assert tuple(probe.call_args.args[1]) == ("net_ema",)
    else:
        probe.assert_not_called()


def test_autoresume_rank0_calls_run_inside_rank_failure_guards(monkeypatch):
    import fastgen.trainer as trainer_module
    from fastgen.utils.distributed import collectives

    events = []

    @contextmanager
    def guard(operation):
        events.append(f"enter {operation}")
        yield
        events.append(f"exit {operation}")

    trainer = object.__new__(Trainer)
    trainer.auto_resume = SimpleNamespace(
        termination_requested=lambda: events.append("termination_requested") or True,
        get_resume_details=lambda: {},
        request_resume=lambda user_dict: events.append(f"request_resume {user_dict}"),
    )
    monkeypatch.setattr(trainer_module, "raise_across_ranks", guard)
    monkeypatch.setattr(collectives, "raise_across_ranks", guard)
    monkeypatch.setattr(trainer_module, "synchronize", lambda: None)
    monkeypatch.setattr(trainer_module, "is_rank0", lambda: True)

    assert trainer.auto_resume_exit(
        SimpleNamespace(device=torch.device("cpu")),
        iteration=7,
        skip_if_just_saved=True,
        recent_checkpoint_path="checkpoint.pth",
    )
    assert events == [
        "enter autoresume termination poll",
        "termination_requested",
        "exit autoresume termination poll",
        "enter autoresume resume request",
        "request_resume {'save_path': 'checkpoint.pth'}",
        "exit autoresume resume request",
    ]


def test_run_refuses_a_multi_rank_launch_without_ddp_or_fsdp(monkeypatch):
    """Several ranks without DDP or FSDP would train independent replicas; the trainer refuses up front."""
    import fastgen.trainer as trainer_module

    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(ddp=False, fsdp=False))
    trainer.callbacks = MagicMock()
    monkeypatch.setattr(trainer_module, "world_size", lambda: 2)

    with pytest.raises(ValueError, match="independent replica"):
        trainer.run(MagicMock())

    trainer.callbacks.on_model_init_start.assert_not_called()


def test_gpu_memory_is_cleared_after_checkpoint_callbacks(monkeypatch):
    """GPU memory is cleared after load callbacks, even without a resume."""

    class StopAfterCleanup(Exception):
        pass

    events = []
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            fsdp=False,
            ddp=False,
            resume=False,
            seed=0,
            checkpointer=SimpleNamespace(pretrained_ckpt_path="", pretrained_ckpt_key_map={}),
        ),
        model=SimpleNamespace(),
    )
    trainer = object.__new__(Trainer)
    trainer.config = config
    trainer.callbacks = MagicMock()
    trainer.callbacks.on_load_checkpoint_end.side_effect = lambda *_args, **_kwargs: events.append("load_callback")
    trainer.auto_resume = MagicMock()
    trainer.auto_resume.get_resume_details.return_value = None
    trainer.checkpointer = MagicMock()

    model = MagicMock()
    model.device = torch.device("cpu")

    import fastgen.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "synchronize", lambda: None)

    def stop_after_cleanup():
        events.append("clear_gpu_memory")
        raise StopAfterCleanup

    monkeypatch.setattr(trainer_module.basic_utils, "clear_gpu_memory", stop_after_cleanup)

    with pytest.raises(StopAfterCleanup):
        trainer.run(model)

    trainer.checkpointer.load.assert_not_called()
    assert events == ["load_callback", "clear_gpu_memory"]


@pytest.mark.parametrize("marker_exists", [False, True], ids=["missing", "present"])
def test_autoresume_continuation_without_its_checkpoint_resumes_from_the_latest(monkeypatch, tmp_path, marker_exists):
    """A continuation whose ``latest_ar`` never completed loads the latest checkpoint instead."""
    save_path = tmp_path / "latest_ar.pth"
    if marker_exists:
        save_path.touch()
    checkpointer = SimpleNamespace(pretrained_ckpt_path="", pretrained_ckpt_key_map={}, save_dir=str(tmp_path))
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(fsdp=False, ddp=False, resume=False, checkpointer=checkpointer)
    )
    trainer.callbacks = MagicMock()
    trainer.auto_resume = MagicMock()
    trainer.auto_resume.get_resume_details.return_value = {"save_path": str(save_path)}
    trainer.checkpointer = MagicMock()
    trainer.checkpointer.resolve_load_path.side_effect = lambda path: path
    trainer.checkpointer.load.side_effect = RuntimeError("stop after the load")
    warned = []
    monkeypatch.setattr("fastgen.trainer.synchronize", lambda: None)
    monkeypatch.setattr("fastgen.utils.logging_utils.warning", warned.append)

    with pytest.raises(RuntimeError, match="stop after the load"):
        trainer.run(MagicMock(device=torch.device("cpu")))

    assert trainer.checkpointer.load.call_args.kwargs["path"] == (str(save_path) if marker_exists else None)
    assert any("not found; resuming from the latest checkpoint" in message for message in warned) == (not marker_exists)


@pytest.mark.parametrize("meta_init", [False, True], ids=["real-init", "meta-init"])
def test_rng_states_are_compared_before_the_per_rank_reseed(monkeypatch, meta_init):
    """CUDA is always compared and CPU only without meta init, right before the per-rank reseed."""

    class StopAtReseed(Exception):
        pass

    events = []
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            fsdp=False,
            ddp=False,
            resume=False,
            seed=3,
            checkpointer=SimpleNamespace(pretrained_ckpt_path="", pretrained_ckpt_key_map={}),
        ),
        model=SimpleNamespace(meta_init=meta_init),
    )
    trainer.callbacks = MagicMock()
    trainer.auto_resume = MagicMock()
    trainer.auto_resume.get_resume_details.return_value = None
    trainer.checkpointer = MagicMock()

    import fastgen.trainer as trainer_module

    def reseed(seed, iteration=0, by_rank=False, devices=None):
        events.append(("reseed", seed, iteration, by_rank))
        raise StopAtReseed

    monkeypatch.setattr(trainer_module, "synchronize", lambda: None)
    monkeypatch.setattr(trainer_module.basic_utils, "clear_gpu_memory", lambda: events.append(("clear_gpu_memory",)))
    monkeypatch.setattr(
        trainer_module,
        "warn_on_rng_state_drift",
        lambda operation, cpu: events.append(("compare", operation, cpu)),
    )
    monkeypatch.setattr(trainer_module, "set_random_seed", reseed)

    with pytest.raises(StopAtReseed):
        trainer.run(MagicMock())

    assert events == [
        ("clear_gpu_memory",),
        ("compare", "Before the per-rank reseed", not meta_init),
        ("reseed", 3, 0, True),
    ]


@pytest.mark.parametrize(
    ("resume_iteration", "max_iter", "expected_steps"),
    [
        (0, 6, [1, 2, 3, 4, 5, 6]),
        (2, 6, [3, 4, 5, 6]),
        (6, 6, []),
        (8, 6, []),
        (0, 1, [1]),
        (0, 0, []),
    ],
)
@pytest.mark.parametrize("interval", [3, 5, 100])
def test_final_checkpoint_holds_max_iter_iterations(monkeypatch, resume_iteration, max_iter, expected_steps, interval):
    """``max_iter`` is inclusive and the final checkpoint, saved once, is labelled with the iterations it holds, so
    a completed run can be extended without skipping an optimizer step or input batch. The final iteration validates
    and saves but does not poll for termination, because a completed run needs no continuation; a resume from its
    checkpoint neither trains, saves, nor validates, and warns that nothing is left to train."""
    import fastgen.trainer as trainer_module

    loader = SimpleNamespace(batch_size=2, sampler_start_idx=None)
    validation = SimpleNamespace(sampler_start_idx=None)
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            ddp=False,
            fsdp=False,
            resume=True,
            seed=0,
            grad_accum_rounds=1,
            max_iter=max_iter,
            validation_iter=interval,
            save_ckpt_iter=interval,
            augment_pipe=None,
            checkpointer=SimpleNamespace(pretrained_ckpt_path="", pretrained_ckpt_key_map={}),
        ),
        model=SimpleNamespace(meta_init=False),
        dataloader_train=loader,
        dataloader_val=validation,
    )
    trainer.callbacks = MagicMock()
    trainer.auto_resume = MagicMock()
    trainer.auto_resume.get_resume_details.return_value = None
    trainer.checkpointer = MagicMock()
    trainer.checkpointer.load.return_value = resume_iteration
    events = []

    def poll(_model, iteration, skip_if_just_saved, **_):
        events.append(("poll", iteration, skip_if_just_saved))
        return False

    trainer.train_step = MagicMock(return_value=({}, {}))
    trainer.validate = MagicMock(side_effect=lambda *args, iteration: events.append(("validate", iteration)))
    trainer.save_checkpoint = MagicMock(side_effect=lambda _model, iteration: events.append(("save", iteration)))
    trainer.auto_resume_exit = MagicMock(side_effect=poll)
    monkeypatch.setattr(trainer_module, "world_size", lambda: 1)
    monkeypatch.setattr(trainer_module, "synchronize", lambda: None)
    monkeypatch.setattr(trainer_module, "set_random_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "warn_on_rng_state_drift", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module.basic_utils, "clear_gpu_memory", lambda: None)
    monkeypatch.setattr(trainer_module.time, "sleep", lambda _: None)

    def batches():
        for _ in expected_steps:
            yield {}
        raise AssertionError("read beyond max_iter")

    monkeypatch.setattr(trainer_module, "instantiate", lambda cfg: batches() if cfg is loader else cfg)
    warned = []
    monkeypatch.setattr(trainer_module.logger, "warning", warned.append)
    model = MagicMock()
    model.prepare_batch.side_effect = lambda data, *args: data

    trainer.run(model)

    final = expected_steps[-1] if expected_steps else resume_iteration
    saved = [call.args[1] for call in trainer.save_checkpoint.call_args_list]
    validated = [call.kwargs["iteration"] for call in trainer.validate.call_args_list]
    assert [call.args[3] for call in trainer.train_step.call_args_list] == expected_steps
    if expected_steps:
        assert saved[-1] == final and saved.count(final) == 1
        assert validated[-1] == final and validated.count(final) == 1
        assert events[-2:] == [("validate", final), ("save", final)]
    else:
        # Only a fresh run validates before its first step.
        assert saved == [] and validated == ([0] if resume_iteration == 0 else [])
    assert [event[1] for event in events if event[0] == "poll"] == expected_steps[:-1]
    assert len(warned) == int(resume_iteration > 0 and not expected_steps)
    trainer.callbacks.on_train_end.assert_called_once_with(model, iteration=final)
    trainer.callbacks.on_app_end.assert_called_once_with(model, iteration=final)
    assert loader.sampler_start_idx == resume_iteration * loader.batch_size
    assert validation.sampler_start_idx == 0


@pytest.mark.parametrize("grad_accum_rounds", [1, 3])
def test_extending_final_checkpoint_matches_uninterrupted_optimizer_steps(monkeypatch, tmp_path, grad_accum_rounds):
    """A saved step count resumes the optimizer, scheduler, and next batch without a gap."""
    import itertools
    import fastgen.trainer as trainer_module
    from fastgen.configs.config import BaseCheckpointerConfig
    from fastgen.utils.checkpointer import Checkpointer

    class CountingModel(torch.nn.Module):
        device = torch.device("cpu")

        def __init__(self):
            super().__init__()
            self.model_dict = torch.nn.ModuleDict({"net": torch.nn.Linear(1, 1, bias=False)})
            with torch.no_grad():
                self.model_dict["net"].weight.fill_(0.25)
            self.grad_scaler = torch.amp.GradScaler("cpu", enabled=False)
            self.steps = []
            self.samples = []

        def init_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
            self.optimizer_dict = {"net": optimizer}
            self.scheduler_dict = {"net": torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)}

        def on_train_begin(self, **kwargs):
            pass

        def apply_torch_compile(self):
            pass

        def reshard_fsdp_modules(self):
            pass

        def autocast(self):
            return nullcontext()

        def prepare_batch(self, data, augment_pipe):
            return data

        def single_train_step(self, data, iteration):
            self.samples.extend(data)
            inputs = torch.tensor(data, dtype=torch.float32).reshape(-1, 1) / 10
            loss = (self.model_dict["net"](inputs) - 2 * inputs).square().mean()
            return {"total_loss": loss}, {}

        def optimizers_schedulers_step(self, iteration):
            self.optimizer_dict["net"].step()
            self.scheduler_dict["net"].step()
            self.steps.append(iteration)

        def optimizers_zero_grad(self, iteration):
            self.optimizer_dict["net"].zero_grad(set_to_none=True)

    monkeypatch.setattr(trainer_module, "world_size", lambda: 1)
    monkeypatch.setattr(trainer_module, "synchronize", lambda: None)
    monkeypatch.setattr(trainer_module, "set_random_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "warn_on_rng_state_drift", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module.basic_utils, "clear_gpu_memory", lambda: None)
    monkeypatch.setattr(trainer_module.time, "sleep", lambda _: None)

    def run(directory, max_iter):
        loader = SimpleNamespace(batch_size=2, sampler_start_idx=None)

        def instantiate_loader(config):
            if config is loader:
                return (
                    list(range(start, start + loader.batch_size))
                    for start in itertools.count(loader.sampler_start_idx, loader.batch_size)
                )
            return config

        monkeypatch.setattr(trainer_module, "instantiate", instantiate_loader)
        trainer = object.__new__(Trainer)
        checkpoint_config = BaseCheckpointerConfig(save_dir=str(directory))
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(
                ddp=False,
                fsdp=False,
                resume=True,
                seed=0,
                grad_accum_rounds=grad_accum_rounds,
                max_iter=max_iter,
                validation_iter=100,
                save_ckpt_iter=100,
                augment_pipe=None,
                checkpointer=checkpoint_config,
            ),
            model=SimpleNamespace(meta_init=False),
            dataloader_train=loader,
            dataloader_val=None,
        )
        trainer.callbacks = MagicMock()
        trainer.callbacks.state_dict.return_value = {}
        trainer.auto_resume = MagicMock()
        trainer.auto_resume.get_resume_details.return_value = None
        trainer.auto_resume_exit = MagicMock(return_value=False)
        trainer.checkpointer = Checkpointer(checkpoint_config)
        model = CountingModel()
        trainer.run(model)
        state = torch.load(directory / f"{max_iter:07d}.pth", weights_only=True)
        assert state["iteration"] == state["scheduler"]["net"]["last_epoch"] == max_iter
        return model, state

    continuous, expected = run(tmp_path / "continuous", 5)
    first, first_state = run(tmp_path / "extended", 2)
    resumed, actual = run(tmp_path / "extended", 5)
    assert first_state["iteration"] == 2
    assert continuous.steps == [1, 2, 3, 4, 5]
    assert first.steps == [1, 2]
    assert resumed.steps == [3, 4, 5]
    assert continuous.samples == first.samples + resumed.samples == list(range(5 * 2 * grad_accum_rounds))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    final_checkpoint = tmp_path / "extended" / "0000005.pth"
    saved_at = final_checkpoint.stat().st_mtime_ns
    completed, unchanged = run(tmp_path / "extended", 5)
    assert completed.steps == completed.samples == []
    assert final_checkpoint.stat().st_mtime_ns == saved_at
    torch.testing.assert_close(unchanged, expected, rtol=0, atol=0)


def test_trainer():
    config = create_config()
    config.log_config.name = "test"
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **EMA_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            "accum_end_counter": L(AccumEndCounterCallback)(),
        }
    )

    model_config = config.model
    model_config.gan_loss_weight_gen = 0.01
    # 8x8 minimum: discriminator head uses 4x4 kernels in 8->4->1 pipeline
    input_shape = [3, 8, 8]
    opts_network = [
        "-",
        "backbone.img_resolution=8",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
    ]
    model_config.teacher = override_config_with_opts(model_config.net, opts_network)
    model_config.net = override_config_with_opts(model_config.net, opts_network)
    model_config.use_ema = True
    model_config.student_update_freq = 1

    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    model_config.discriminator = override_config_with_opts(model_config.discriminator, opts_discriminator)

    model_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config.pretrained_model_path = None  # disable ckpt loading
    model_config.input_shape = input_shape

    # Use synthetic loader so real_data shape matches gen_data (avoids shape mismatch in forward_process)
    config.dataloader_train = L(SyntheticLoader)(shape=input_shape, batch_size=1, sampler_start_idx=None)
    config.dataloader_val = L(SyntheticLoader)(shape=input_shape, batch_size=1, sampler_start_idx=None)

    config.trainer = override_config_with_opts(
        config.trainer, ["-", "max_iter=2", "save_ckpt_iter=1", "validation_iter=1", "logging_iter=1"]
    )
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 2}]  # limit validation to 2 steps

    with tempfile.TemporaryDirectory() as tmpdir:
        config.trainer.checkpointer.save_dir = tmpdir

        # initiate the model (config.model passed as explicit kwarg so
        # OmegaConf doesn't coerce the attrs subtree into DictConfig)
        model = instantiate(config.model_class, config=config.model)

        # initiate the trainer
        fastgen_trainer = Trainer(config)
        fastgen_trainer.run(model)
        expected_calls = config.trainer.max_iter * config.trainer.grad_accum_rounds
        assert fastgen_trainer.callbacks._callbacks["accum_end_counter"].count == expected_calls
