# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest

from fastgen.utils.env import hf_download, hf_load, output_root, set_env_vars, str2bool

_BEARER_ENV_KEYS = ("WANDB_API_KEY", "HF_TOKEN", "OPENAI_API_KEY")
_AWS_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "S3_ENDPOINT_URL",
)


@pytest.fixture(autouse=True)
def _isolated_environment():
    """``set_env_vars`` writes ``os.environ`` directly; restore it so stub credentials never reach later tests."""
    saved = os.environ.copy()
    for key in (*_BEARER_ENV_KEYS, *_AWS_ENV_KEYS):
        os.environ.pop(key, None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _write(path: Path, value: str) -> Path:
    path.write_text(value)
    return path


@pytest.mark.parametrize("value", (True, "yes", "YES", "true", "t", "1"))
def test_str2bool_accepts_true_values(value):
    assert str2bool(value) is True


@pytest.mark.parametrize("value", (False, "no", "NO", "false", "f", "0"))
def test_str2bool_accepts_false_values(value):
    assert str2bool(value) is False


def test_str2bool_rejects_invalid_strings():
    with pytest.raises(ValueError, match="Boolean value expected"):
        str2bool("sometimes")


def test_output_root_keeps_a_caller_supplied_value_verbatim(monkeypatch):
    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", "/shared/fastgen-output")

    assert output_root() == "/shared/fastgen-output"


def test_output_root_defaults_to_the_in_tree_directory_when_unset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FASTGEN_OUTPUT_ROOT", raising=False)

    assert output_root() == os.path.join(os.getcwd(), "FASTGEN_OUTPUT")


@pytest.mark.parametrize("value", ("", "   "))
def test_output_root_treats_a_blank_value_as_unset(monkeypatch, tmp_path, value):
    """An unexpanded shell variable arrives blank, which must not collapse the path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", value)

    assert output_root() == os.path.join(os.getcwd(), "FASTGEN_OUTPUT")


def test_output_root_pins_a_relative_value_on_read(monkeypatch, tmp_path):
    """A resolved root must survive a later ``chdir``, which config import performs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", "FASTGEN_OUTPUT")
    expected = os.path.join(os.getcwd(), "FASTGEN_OUTPUT")

    resolved = output_root()
    monkeypatch.chdir("/")

    assert resolved == expected


def test_output_root_strips_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", "  /shared/fastgen-output  ")

    assert output_root() == "/shared/fastgen-output"


def _record(model_id, *args, **kwargs):
    return model_id, args, kwargs


def test_hf_load_uses_fastgen_hugging_face_environment(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/tmp/fastgen-hf-cache")
    monkeypatch.setenv("LOCAL_FILES_ONLY", "true")

    assert hf_load(_record, "org/model") == (
        "org/model",
        (),
        {"cache_dir": "/tmp/fastgen-hf-cache", "local_files_only": True},
    )


def test_hf_load_omits_unset_cache_and_defaults_online(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("LOCAL_FILES_ONLY", raising=False)

    assert hf_load(_record, "org/model") == ("org/model", (), {"local_files_only": False})


def test_hf_load_forwards_arguments_and_lets_the_caller_override_the_defaults(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/tmp/fastgen-hf-cache")
    monkeypatch.delenv("LOCAL_FILES_ONLY", raising=False)

    assert hf_load(_record, "org/model", "weights.pt", subfolder="vae", local_files_only=True) == (
        "org/model",
        ("weights.pt",),
        {"cache_dir": "/tmp/fastgen-hf-cache", "local_files_only": True, "subfolder": "vae"},
    )


def test_hf_download_uses_a_local_directory_in_place(monkeypatch, tmp_path):
    import huggingface_hub

    for name in ("hf_hub_download", "snapshot_download"):
        monkeypatch.setattr(huggingface_hub, name, lambda *args, **kwargs: pytest.fail("a local directory uses no Hub"))
    root = str(tmp_path)

    assert hf_download(root) == root
    assert hf_download(root, "weights.pt") == str(tmp_path / "weights.pt")
    assert hf_download(root, "index.json", subfolder="vae", revision="main") == str(tmp_path / "vae" / "index.json")


def test_hf_download_fetches_a_repo_file_or_snapshot_through_hf_load(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: ("file", args, kwargs))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: ("snapshot", args, kwargs))
    monkeypatch.setenv("HF_HOME", "/tmp/fastgen-hf-cache")
    monkeypatch.setenv("LOCAL_FILES_ONLY", "true")
    defaults = {"cache_dir": "/tmp/fastgen-hf-cache", "local_files_only": True}

    assert hf_download("org/model", "weights.pt", subfolder="vae") == (
        "file",
        ("org/model", "weights.pt"),
        {"subfolder": "vae", **defaults},
    )
    assert hf_download("org/model", allow_patterns=["vae/*"]) == (
        "snapshot",
        ("org/model",),
        {"allow_patterns": ["vae/*"], **defaults},
    )


def _hub_error(name: str) -> Exception:
    """Build the ``huggingface_hub.errors`` exception ``name`` for an anonymous 401 response."""
    import httpx
    from huggingface_hub import errors

    response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co/api/models/org/model"))
    return getattr(errors, name)(f"401 Client Error: {name}", response=response)


def _raise_like_diffusers(hub_error: Exception):
    """Return a loader that, like diffusers, re-raises ``hub_error`` as a plain ``OSError`` while handling it."""

    def loader(model_id, **kwargs):
        try:
            raise hub_error
        except OSError:
            raise OSError(f"{model_id} is not a local folder and is not a valid model identifier")

    return loader


def test_hf_load_explains_a_gated_repo_wrapped_by_diffusers():
    gated = _hub_error("GatedRepoError")

    with pytest.raises(RuntimeError, match=r"accept the license at https://huggingface\.co/org/gated, ") as error:
        hf_load(_raise_like_diffusers(gated), "org/gated")

    assert error.value.__cause__.__context__ is gated


def test_hf_load_explains_a_missing_repo_without_a_license_step():
    with pytest.raises(RuntimeError, match="hf auth login") as error:
        hf_load(_raise_like_diffusers(_hub_error("RepositoryNotFoundError")), "org/missing")

    assert "accept the license" not in str(error.value)


@pytest.mark.parametrize("unrelated", (OSError("No space left on device"), ValueError("bad config")))
def test_hf_load_passes_unrelated_errors_through(unrelated):
    def loader(model_id, **kwargs):
        raise unrelated

    with pytest.raises(type(unrelated)) as error:
        hf_load(loader, "org/model")

    assert error.value is unrelated


def test_loads_all_three_bearer_tokens(tmp_path):
    _write(tmp_path / "wandb_api.txt", "wandb-key\n")
    _write(tmp_path / "hf_token.txt", "hf-key\n")
    _write(tmp_path / "vlm_api_token.txt", "vlm-key\n")

    set_env_vars(str(tmp_path))

    assert os.environ["WANDB_API_KEY"] == "wandb-key"
    assert os.environ["HF_TOKEN"] == "hf-key"
    assert os.environ["OPENAI_API_KEY"] == "vlm-key"


def test_defaults_to_the_credentials_folder_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "credentials").mkdir()
    _write(tmp_path / "credentials" / "hf_token.txt", "hf-key")

    set_env_vars()

    assert os.environ["HF_TOKEN"] == "hf-key"


def test_strips_surrounding_whitespace(tmp_path):
    _write(tmp_path / "vlm_api_token.txt", "  raw-token \n\n")

    set_env_vars(str(tmp_path))

    assert os.environ["OPENAI_API_KEY"] == "raw-token"


@pytest.mark.parametrize("content", ("s3cr3t\nline\n", "s3cr3t\rline"))
def test_rejects_a_multi_line_token_file_without_printing_it(tmp_path, content):
    _write(tmp_path / "wandb_api.txt", content)

    with pytest.raises(ValueError, match="wandb_api.txt must hold WANDB_API_KEY on a single line") as error:
        set_env_vars(str(tmp_path))

    assert "s3cr3t" not in str(error.value)
    assert "WANDB_API_KEY" not in os.environ


def test_missing_folder_and_files_are_silent_noops(tmp_path):
    set_env_vars(str(tmp_path / "no-such-folder"))
    set_env_vars(str(tmp_path))

    for key in (*_BEARER_ENV_KEYS, *_AWS_ENV_KEYS):
        assert key not in os.environ


def test_a_present_file_overrides_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    _write(tmp_path / "vlm_api_token.txt", "from-file")

    set_env_vars(str(tmp_path))

    assert os.environ["OPENAI_API_KEY"] == "from-file"


def test_override_false_keeps_set_variables_and_fills_the_missing_ones(tmp_path, monkeypatch):
    """Release payloads keep an injected secret and load the files only for variables that are unset or empty."""
    monkeypatch.setenv("WANDB_API_KEY", "from-scheduler")
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "id-from-scheduler")
    _write(tmp_path / "wandb_api.txt", "from-file")
    _write(tmp_path / "hf_token.txt", "hf-from-file")
    _write(
        tmp_path / "s3.json",
        json.dumps({"aws_access_key_id": "id-from-file", "aws_secret_access_key": "secret-from-file"}),
    )

    set_env_vars(str(tmp_path), override=False)

    assert os.environ["WANDB_API_KEY"] == "from-scheduler"
    assert os.environ["HF_TOKEN"] == "hf-from-file"
    assert os.environ["AWS_ACCESS_KEY_ID"] == "id-from-scheduler"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "secret-from-file"


def test_a_missing_file_keeps_the_environment_value(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "from-shell")
    _write(tmp_path / "hf_token.txt", "hf-key")

    set_env_vars(str(tmp_path))

    assert os.environ["WANDB_API_KEY"] == "from-shell"
    assert os.environ["HF_TOKEN"] == "hf-key"


@pytest.mark.parametrize("content", ("", " \n\n"))
def test_an_empty_file_keeps_the_environment_value(tmp_path, monkeypatch, content):
    monkeypatch.setenv("WANDB_API_KEY", "from-shell")
    _write(tmp_path / "wandb_api.txt", content)
    _write(tmp_path / "hf_token.txt", content)

    set_env_vars(str(tmp_path))

    assert os.environ["WANDB_API_KEY"] == "from-shell"
    assert "HF_TOKEN" not in os.environ


def test_loads_aws_credentials_from_s3_json(tmp_path):
    aws_file = tmp_path / "s3.json"
    aws_file.write_text(
        json.dumps(
            {
                "aws_access_key_id": "AKIA-stub",
                "aws_secret_access_key": "secret-stub",
                "aws_session_token": "token-stub",
                "region_name": "us-west-2",
                "endpoint_url": "https://s3.example.com",
            }
        )
    )

    set_env_vars(str(tmp_path))

    assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIA-stub"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "secret-stub"
    assert os.environ["AWS_SESSION_TOKEN"] == "token-stub"
    assert os.environ["AWS_DEFAULT_REGION"] == "us-west-2"
    assert os.environ["AWS_ENDPOINT_URL"] == "https://s3.example.com"
    assert os.environ["S3_ENDPOINT_URL"] == "https://s3.example.com"


def test_s3_json_without_session_token_preserves_inherited_token(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_SESSION_TOKEN", "inherited-token")
    aws_file = tmp_path / "s3.json"
    aws_file.write_text(
        json.dumps(
            {
                "aws_access_key_id": "AKIA-stub",
                "aws_secret_access_key": "secret-stub",
                "region_name": "us-west-2",
                "endpoint_url": "https://s3.example.com",
            }
        )
    )

    set_env_vars(str(tmp_path))

    assert os.environ["AWS_SESSION_TOKEN"] == "inherited-token"


def test_s3_json_overrides_credentials_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    aws_file = tmp_path / "s3.json"
    aws_file.write_text(
        json.dumps(
            {
                "aws_access_key_id": "file-key",
                "aws_secret_access_key": "file-secret",
                "region_name": "us-west-2",
                "endpoint_url": "https://s3.example.com",
            }
        )
    )

    set_env_vars(str(tmp_path))

    assert os.environ["AWS_ACCESS_KEY_ID"] == "file-key"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "file-secret"


def test_hf_home_default_set(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", str(tmp_path))
    set_env_vars(None)

    assert os.environ["HF_HOME"].startswith(str(tmp_path))
    assert os.environ["HF_HOME"].endswith(".cache")


def test_hf_home_is_absolute_for_a_relative_output_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("FASTGEN_OUTPUT_ROOT", "FASTGEN_OUTPUT")
    set_env_vars(None)

    assert os.environ["HF_HOME"] == os.path.join(os.getcwd(), "FASTGEN_OUTPUT", ".cache")


def test_existing_hf_home_is_preserved(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/shared/hugging-face-cache")
    set_env_vars(None)

    assert os.environ["HF_HOME"] == "/shared/hugging-face-cache"
