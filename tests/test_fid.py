# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FID must handle small distributed evaluations and propagate setup failures."""

from pathlib import Path
from types import SimpleNamespace
import json
import warnings

from PIL import Image
import numpy as np
import pytest
import torch

from fastgen.utils.io_utils import parse_checkpoint_iteration
from scripts.fid import fid
from scripts.fid.inception import InceptionV3
from tests.utils import RunIf, run_distributed_test


def test_inception_without_internal_resize_preserves_input_normalization():
    seen = []

    class Layers(torch.nn.Module):
        def forward(self, images):
            seen.append(images)
            return images.new_zeros(len(images), 2048)

    detector = InceptionV3.__new__(InceptionV3)
    torch.nn.Module.__init__(detector)
    detector.layers = Layers()
    detector.resize_inside = False
    images = torch.tensor([-1.0, 1.0]).reshape(2, 1, 1, 1).expand(2, 3, 299, 299)

    features = detector(images)

    assert features.shape == (2, 2048)
    torch.testing.assert_close(seen[0][0], torch.full_like(images[0], -1.0))
    torch.testing.assert_close(seen[0][1], torch.full_like(images[1], 127.0 / 128.0))


def test_fid_matches_the_closed_form_for_diagonal_gaussians():
    mu, sigma = np.array([1.0, 0.0]), np.diag([4.0, 1.0])
    # |mu - mu_ref|^2 = 1 and tr(sigma + sigma_ref - 2 (sigma sigma_ref)^(1/2)) = (4 + 1 - 4) + (1 + 1 - 2) = 1.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        score = fid.calculate_fid_from_inception_stats(mu, sigma, np.zeros(2), np.eye(2))
    assert score == pytest.approx(2.0)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_fid_rejects_nonfinite_statistics(index, invalid):
    values = [np.zeros(2), np.eye(2), np.zeros(2), np.eye(2)]
    values[index].flat[0] = invalid
    with pytest.raises(ValueError, match="real and finite"):
        fid.calculate_fid_from_inception_stats(*values)


@pytest.mark.parametrize("mu_ref, sigma_ref", [(np.zeros(1), np.eye(2)), (np.zeros(2), np.eye(3))])
def test_fid_rejects_incompatible_statistics(mu_ref, sigma_ref):
    with pytest.raises(ValueError, match="matching square covariance"):
        fid.calculate_fid_from_inception_stats(np.zeros(2), np.eye(2), mu_ref, sigma_ref)


def test_fid_rejects_a_nonfinite_matrix_square_root(monkeypatch):
    monkeypatch.setattr(fid.scipy.linalg, "sqrtm", lambda value: np.full_like(value, np.nan))
    with pytest.raises(ValueError, match="non-finite result"):
        fid.calculate_fid_from_inception_stats(np.zeros(2), np.eye(2), np.zeros(2), np.eye(2))


def test_fid_uses_the_iteration_directory_name(tmp_path, monkeypatch):
    samples = tmp_path / "iter_9000" / "samples"
    (samples / "iter_2").mkdir(parents=True)
    (samples / "iter_3_incomplete").mkdir()
    (samples / "iter_4").touch()
    refs = tmp_path / "fid-refs"
    refs.mkdir()
    np.savez(refs / "cifar10-32x32.npz", mu=np.zeros(2), sigma=np.eye(2))
    monkeypatch.setattr(fid, "DATA_ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(fid, "InceptionV3", torch.nn.Identity)
    measured = []

    def statistics(*args, image_path, **kwargs):
        measured.append(image_path)
        return np.zeros(2), np.eye(2)

    monkeypatch.setattr(fid, "calculate_inception_stats", statistics)
    fid.calc(str(samples), 2, 0, 0, 10, 1, "cifar10", device="cpu")

    assert measured == [str(samples / "iter_2")]
    assert json.loads((samples / "fid.json").read_text()) == {"ckpt_num": [2], "fid": [0.0]}


@pytest.mark.parametrize("rank", [0, 3])
def test_fid_small_sample_sets_never_collate_an_empty_batch(tmp_path, monkeypatch, rank):
    for index in range(2):
        Image.new("RGB", (2, 2), color=(index * 255,) * 3).save(tmp_path / f"{index}.png")
    monkeypatch.setattr(fid, "world_size", lambda: 4)
    monkeypatch.setattr(fid, "get_rank", lambda: rank)
    reductions = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value: reductions.append(value.clone()))
    batch_sizes = []

    def detector(images):
        batch_sizes.append(len(images))
        return images.flatten(1)[:, :2]

    fid.calculate_inception_stats(detector, 2, str(tmp_path), num_expected=2, num_workers=0, device="cpu")

    assert batch_sizes == ([1] if rank == 0 else [])
    assert len(reductions) == 2
    torch.testing.assert_close(reductions[0], torch.full((2,), -1.0 if rank == 0 else 0.0, dtype=torch.float64))
    torch.testing.assert_close(reductions[1], torch.full((2, 2), 1.0 if rank == 0 else 0.0, dtype=torch.float64))


def _fid_setup_failure(rank, world_size, path, operation):
    del world_size
    if operation == "reference":
        fid.DATA_ROOT_DIR = path
        with pytest.raises(RuntimeError, match="FID reference loading"):
            fid.calc(path, 2, 0, 0, 10, 1, "cifar10", device="cpu")
    elif operation in ("history", "detector", "scores"):
        labels = {
            "history": "FID evaluation discovery",
            "detector": "FID detector initialization",
            "scores": "FID score computation",
        }

        def detector():
            if operation == "detector" and rank == 1:
                raise ValueError("detector initialization failed")
            return torch.nn.Identity()

        def score(*args):
            raise ValueError("invalid reference statistics")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fid, "DATA_ROOT_DIR", path)
            patch.setattr(fid, "InceptionV3", detector)
            patch.setattr(fid, "calculate_inception_stats", lambda *args, **kwargs: (np.zeros(2), np.eye(2)))
            patch.setattr(fid, "calculate_fid_from_inception_stats", score)
            with pytest.raises(RuntimeError, match=labels[operation]):
                fid.calc(path, 2, 0, 0, 10, 1, "cifar10", device="cpu")
    elif operation == "features":

        def detector(images):
            if rank == 1:
                raise ValueError("feature extraction failed")
            return torch.zeros(len(images), 2)

        with pytest.raises(RuntimeError, match="FID feature extraction"):
            fid.calculate_inception_stats(detector, 2, path, num_expected=2, num_workers=0, device="cpu")
    else:
        # Only one rank has a bad view of the input directory.
        image_path = str(Path(path) / "missing") if rank == 1 else path
        with pytest.raises(RuntimeError, match="FID image discovery"):
            fid.calculate_inception_stats(None, 2, image_path, num_expected=2, device="cpu")
    return {"raised": True}


@RunIf(min_gpus=2)
@pytest.mark.parametrize("operation", ["reference", "images", "features", "history", "detector", "scores"])
def test_fid_setup_failure_reaches_every_rank(tmp_path, operation):
    for index in range(2):
        Image.new("RGB", (2, 2)).save(tmp_path / f"{index}.png")
    if operation in ("history", "detector", "scores"):
        (tmp_path / "fid-refs").mkdir()
        np.savez(tmp_path / "fid-refs/cifar10-32x32.npz", mu=np.zeros(2), sigma=np.eye(2))
        (tmp_path / "iter_1").mkdir()
        (tmp_path / "iter_2").mkdir()
    if operation == "history":
        (tmp_path / "fid.json").write_text("invalid JSON")
    result = run_distributed_test(
        _fid_setup_failure, world_size=2, timeout=120, path=str(tmp_path), operation=operation
    )
    assert result == {"raised": True}


def _fid_duplicate_iterations(rank, world_size, path):
    del rank, world_size
    measured = []

    def statistics(*args, image_path, **kwargs):
        measured.append(image_path)
        return np.zeros(2), np.eye(2)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fid, "DATA_ROOT_DIR", path)
        patch.setattr(fid, "InceptionV3", torch.nn.Identity)
        patch.setattr(fid, "calculate_inception_stats", statistics)
        fid.calc(path, 2, 0, 0, 10, 1, "cifar10", device="cpu")
    assert len(measured) == 1
    return {"measured": 1}


@RunIf(min_gpus=2)
def test_fid_duplicate_iteration_names_keep_rank_skip_decisions_aligned(tmp_path):
    (tmp_path / "fid-refs").mkdir()
    np.savez(tmp_path / "fid-refs/cifar10-32x32.npz", mu=np.zeros(2), sigma=np.eye(2))
    for name in ("iter_1", "iter_0001"):
        (tmp_path / name).mkdir()
    result = run_distributed_test(_fid_duplicate_iterations, world_size=2, timeout=120, path=str(tmp_path))
    assert result == {"measured": 1}
    assert json.loads((tmp_path / "fid.json").read_text()) == {"ckpt_num": [1], "fid": [0.0]}


def _fid_sweep_resume(rank, world_size, path, broken_history):
    del world_size
    from scripts.fid import compute_fid_from_ckpts as sweep

    loaded = []
    net = SimpleNamespace(noise_scheduler=SimpleNamespace(default_student_steps=1), label_dim=0, eval=lambda: None)
    model = SimpleNamespace(
        net=net,
        model_dict={},
        use_ema=[],
        net_preprocessors=SimpleNamespace(preprocessors={}),
        device=torch.device("cpu"),
        precision=torch.float32,
        input_shape=(3, 2, 2),
        on_train_begin=lambda: None,
        prepare_batch=lambda data, **kwargs: data,
        generator_fn=lambda network, noise, **kwargs: torch.zeros_like(noise),
    )
    config = SimpleNamespace(
        model_class=None,
        model=None,
        trainer=SimpleNamespace(seed=1, checkpointer=SimpleNamespace(save_dir=path)),
        log_config=SimpleNamespace(save_path=str(Path(path) / str(rank))),
        dataloader_train=SimpleNamespace(dataset_path="cifar10-32x32.zip", batch_size=1),
        eval=SimpleNamespace(samples_dir="samples", num_samples=2, min_ckpt=0, max_ckpt=10, save_images=False),
    )
    checkpointer = SimpleNamespace(load=lambda modules, path: loaded.append(parse_checkpoint_iteration(path)))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sweep, "instantiate", lambda *args, **kwargs: model)
        patch.setattr(sweep, "Checkpointer", lambda config: checkpointer)
        patch.setattr(
            sweep, "list_checkpoints", lambda path: [(iteration, f"{iteration}.pth") for iteration in (1, 2, 3)]
        )
        patch.setattr(sweep, "calc", lambda *args, **kwargs: None)
        patch.setattr(sweep.time, "sleep", lambda *args: None)
        if broken_history:
            with pytest.raises(RuntimeError, match="FID sweep history loading"):
                sweep.main(config)
            assert loaded == []
        else:
            sweep.main(config)
            assert loaded == [3]
    return {"loaded": loaded}


@RunIf(min_gpus=2)
@pytest.mark.parametrize("broken_history", [False, True])
def test_fid_sweep_resume_decisions_use_rank_zero_filesystem_view(tmp_path, broken_history):
    # Rank one sees stale history and no completed samples. Both must still load only checkpoint 3.
    for rank, completed in ((0, 1), (1, 2)):
        samples = tmp_path / str(rank) / "samples"
        samples.mkdir(parents=True)
        (samples / "fid.json").write_text(json.dumps({"ckpt_num": [completed], "fid": [0.0]}))
    if broken_history:
        (tmp_path / "0/samples/fid.json").write_text("invalid JSON")
    completed_samples = tmp_path / "0/samples/iter_2"
    completed_samples.mkdir()
    for index in range(2):
        (completed_samples / f"{index}.png").touch()
    result = run_distributed_test(
        _fid_sweep_resume, world_size=2, timeout=120, path=str(tmp_path), broken_history=broken_history
    )
    assert result == {"loaded": [] if broken_history else [3]}
