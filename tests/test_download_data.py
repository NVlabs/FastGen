# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data preparation must never publish incomplete outputs as ready."""

import hashlib
import io
import json
import pickle
import sys
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import download_data
from scripts.download_data import convert_pickle_to_pth


@pytest.mark.parametrize(
    "arguments, data_root, checkpoint_root, message",
    [
        (["--only-data"], "s3://data", "/models", "local --output-dir"),
        (["--only-models"], "/data", "s3://checkpoints", "local --ckpt-dir"),
        (["--only-models"], "/data", "https://example.test/models", "local --ckpt-dir"),
    ],
)
def test_remote_download_destinations_fail_before_network_access(
    monkeypatch, capsys, arguments, data_root, checkpoint_root, message
):
    monkeypatch.setattr(download_data, "DATA_ROOT_DIR", data_root)
    monkeypatch.setattr(download_data, "CKPT_ROOT_DIR", checkpoint_root)
    monkeypatch.setattr(sys, "argv", ["download_data.py", "--dataset", "cifar10", *arguments])
    monkeypatch.setattr(download_data, "clone_repo", lambda *args: pytest.fail("must validate before network access"))
    with pytest.raises(SystemExit) as error:
        download_data.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def _source_checkpoint(path):
    model = torch.nn.Linear(2, 2, dtype=torch.float64)
    path.write_bytes(pickle.dumps({"ema": model}))
    return model.state_dict()


def test_converted_checkpoint_preserves_state_and_publishes_only_the_destination(tmp_path):
    source = tmp_path / "source.pkl"
    expected = _source_checkpoint(source)
    target = tmp_path / "converted" / "model.pth"

    convert_pickle_to_pth(source, target, tmp_path)

    actual = torch.load(target, weights_only=True)
    assert actual.keys() == expected.keys()
    for name, tensor in actual.items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure_stage", ["save", "verify"])
def test_conversion_failure_never_publishes_partial_checkpoint(tmp_path, monkeypatch, existing, failure_stage):
    source = tmp_path / "source.pkl"
    expected = _source_checkpoint(source)
    target = tmp_path / "model.pth"
    if existing:
        torch.save(expected, target)
        previous = target.read_bytes()

    def fail_save(state, path):
        Path(path).write_bytes(b"incomplete write")
        raise OSError("interrupted checkpoint write")

    def fail_verify(*args, **kwargs):
        raise OSError("checkpoint verification failed")

    monkeypatch.setattr(
        torch, "save" if failure_stage == "save" else "load", fail_save if failure_stage == "save" else fail_verify
    )
    with pytest.raises(OSError, match="checkpoint"):
        convert_pickle_to_pth(source, target, tmp_path)

    if existing:
        assert target.read_bytes() == previous
    else:
        assert not target.exists()
    assert set(tmp_path.iterdir()) == ({source, target} if existing else {source})


@pytest.mark.parametrize("operation", ["dataset", "fid"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["exit", "exception", "missing"])
def test_preparation_failure_preserves_destination_and_can_retry(tmp_path, monkeypatch, operation, existing, failure):
    repo = tmp_path / "converter"
    repo.mkdir()
    (repo / ("dataset_tool.py" if operation == "dataset" else "fid.py")).touch()
    source = tmp_path / "source.zip"
    source.write_bytes(b"source")
    target = tmp_path / ("dataset.zip" if operation == "dataset" else "reference.npz")
    if existing:
        target.write_bytes(b"previous complete output")
    initial_files = set(tmp_path.iterdir())

    def run(command, **kwargs):
        destination = Path(next(arg.removeprefix("--dest=") for arg in command if arg.startswith("--dest=")))
        assert destination.name == target.name
        if failure != "missing":
            destination.write_bytes(b"partial" if failure else b"complete output")
        if failure == "exception":
            raise OSError("interrupted conversion")
        return SimpleNamespace(returncode=int(failure == "exit"), stdout="", stderr="")

    def prepare():
        if operation == "dataset":
            return download_data.run_dataset_tool(repo, source, target, force=existing)
        return download_data.compute_fid_refs(source, target, repo, force=existing)

    monkeypatch.setattr(download_data.subprocess, "run", run)
    with pytest.raises((RuntimeError, OSError)):
        prepare()

    if existing:
        assert target.read_bytes() == b"previous complete output"
    else:
        assert not target.exists()
    assert set(tmp_path.iterdir()) == initial_files

    failure = None
    prepare()
    assert target.read_bytes() == b"complete output"
    assert set(tmp_path.iterdir()) == initial_files | {target}


@pytest.mark.parametrize("operation", ["dataset", "fid"])
def test_existing_preparation_output_is_reused_without_force(tmp_path, monkeypatch, operation):
    target = tmp_path / "complete"
    target.write_bytes(b"complete output")

    def unexpected_run(*args, **kwargs):
        pytest.fail("Existing complete output must not be regenerated without --force")

    monkeypatch.setattr(download_data.subprocess, "run", unexpected_run)
    if operation == "dataset":
        assert download_data.run_dataset_tool(tmp_path, tmp_path / "source", target) is False
    else:
        download_data.compute_fid_refs(tmp_path / "source", target, tmp_path)
    assert target.read_bytes() == b"complete output"


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", [None, "read", "truncated", "checksum"])
def test_download_publishes_only_complete_verified_bytes(tmp_path, monkeypatch, existing, failure):
    target = tmp_path / "model.pkl"
    if existing:
        target.write_bytes(b"previous complete output")
    payload = b"new complete output"
    digest = hashlib.md5(payload).hexdigest()

    class Response(io.BytesIO):
        headers = {"content-length": str(len(payload) + int(failure == "truncated"))}

        def read(self, size):
            if failure == "read" and self.tell():
                raise OSError("interrupted download")
            return super().read(size)

    monkeypatch.setattr(urllib.request, "urlopen", lambda request: Response(payload))
    if failure:
        with pytest.raises((OSError, ValueError)):
            download_data.download_file(
                "https://example.test/model", target, expected_md5="wrong" if failure == "checksum" else digest
            )
        if existing:
            assert target.read_bytes() == b"previous complete output"
        else:
            assert not target.exists()
        assert set(tmp_path.iterdir()) == ({target} if existing else set())
    else:
        download_data.download_file("https://example.test/model", target, expected_md5=digest)
        assert target.read_bytes() == payload
        assert list(tmp_path.iterdir()) == [target]


def test_verified_download_is_reused_without_network_access(tmp_path, monkeypatch):
    target = tmp_path / "model.pkl"
    payload = b"already verified"
    target.write_bytes(payload)

    def unexpected_download(*args, **kwargs):
        pytest.fail("Verified download must be reused")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_download)
    download_data.download_file("https://example.test/model", target, expected_md5=hashlib.md5(payload).hexdigest())
    assert target.read_bytes() == payload


@pytest.mark.parametrize("existing", [False, True])
def test_vae_encoding_failure_preserves_destination_and_can_retry(tmp_path, monkeypatch, existing):
    from diffusers.models import AutoencoderKL

    source = tmp_path / "images.zip"
    with zipfile.ZipFile(source, "w") as archive:
        image = io.BytesIO()
        Image.new("RGB", (8, 8)).save(image, format="PNG")
        for name in ["a.png", "b.png"]:
            archive.writestr(name, image.getvalue())
        archive.writestr("dataset.json", json.dumps({"labels": [["a.png", 1], ["b.png", 2]]}))
    target = tmp_path / "latents.zip"
    if existing:
        target.write_bytes(b"previous complete output")
    initial_files = set(tmp_path.iterdir())

    class VAE:
        calls = 0
        fail = True

        def eval(self):
            return self

        def requires_grad_(self, requires_grad):
            return self

        def cuda(self):
            return self

        def encode(self, image):
            self.calls += 1
            if self.fail and self.calls == 2:
                raise RuntimeError("interrupted encoding")
            return {"latent_dist": SimpleNamespace(mean=torch.zeros(1, 4, 1, 1), std=torch.ones(1, 4, 1, 1))}

    vae = VAE()
    monkeypatch.setattr(AutoencoderKL, "from_pretrained", lambda *args, **kwargs: vae)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    with pytest.raises(RuntimeError, match="interrupted encoding"):
        download_data.encode_dit_style(source, target, force=existing)
    if existing:
        assert target.read_bytes() == b"previous complete output"
    else:
        assert not target.exists()
    assert set(tmp_path.iterdir()) == initial_files

    vae.fail = False
    assert download_data.encode_dit_style(source, target, force=existing) is True
    with zipfile.ZipFile(target) as archive:
        labels = json.loads(archive.read("dataset.json"))["labels"]
        assert [label for _, label in labels] == [1, 2]
        assert len(archive.namelist()) == 3
        for name, _ in labels:
            actual = np.load(io.BytesIO(archive.read(name)))
            np.testing.assert_array_equal(actual[:4], np.zeros((4, 1, 1)))
            np.testing.assert_array_equal(actual[4:], np.ones((4, 1, 1)))
    assert set(tmp_path.iterdir()) == initial_files | {target}
