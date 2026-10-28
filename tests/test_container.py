# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the generated container entrypoint without building an image."""

import json
from pathlib import Path
import shlex
import subprocess
import sys


def _assert_container_entrypoint(dockerfile: Path, tmp_path: Path):
    instructions = [
        line.removeprefix("RUN ")
        for line in dockerfile.read_text().splitlines()
        if line.startswith("RUN ") and "/entry.sh" in line
    ]
    assert len(instructions) == 1
    entrypoint = tmp_path / "container entrypoint.sh"
    # Docker's shell-form RUN preserves escapes within a line. Execute that
    # instruction with only the destination redirected into the test directory.
    build_command = instructions[0].replace("/entry.sh", shlex.quote(str(entrypoint)))
    subprocess.run(["/bin/sh", "-c", build_command], cwd=tmp_path, check=True, capture_output=True, timeout=15)

    arguments = ["two words", "", 'a"b', "a'b", "$HOME", "$(false)", "*.py", "a;b"]
    child = (
        "import json, os, sys; "
        "print(json.dumps({'pid': os.getpid(), 'arguments': sys.argv[2:]})); "
        "sys.exit(int(sys.argv[1]))"
    )
    for exit_code in (0, 17):
        process = subprocess.Popen(
            [str(entrypoint), sys.executable, "-c", child, str(exit_code), *arguments],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=15)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        assert process.returncode == exit_code, stderr
        # exec must replace the wrapper so the command receives container signals.
        assert json.loads(stdout) == {"pid": process.pid, "arguments": arguments}


def test_container_entrypoint_preserves_arguments_process_and_exit_status(tmp_path):
    _assert_container_entrypoint(Path(__file__).resolve().parents[1] / "Dockerfile", tmp_path)
