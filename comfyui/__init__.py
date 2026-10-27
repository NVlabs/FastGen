# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ComfyUI custom nodes wrapping FastGen video inference pipeline."""

import os
import sys

FASTGEN_REPO = os.environ.get(
    "FASTGEN_REPO",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "fastgen"),
)
FASTGEN_REPO = os.path.abspath(FASTGEN_REPO)

# Fail fast on a bad path: inserting it into sys.path would otherwise surface
# as a cryptic ModuleNotFoundError when the nodes try to import fastgen.
if not os.path.isdir(FASTGEN_REPO):
    raise RuntimeError(
        f"FASTGEN_REPO does not point to a valid directory: {FASTGEN_REPO!r}. "
        "Set the FASTGEN_REPO environment variable to your fastgen checkout before starting ComfyUI."
    )

os.environ["FASTGEN_REPO"] = FASTGEN_REPO

if FASTGEN_REPO not in sys.path:
    sys.path.insert(0, FASTGEN_REPO)

if "FASTGEN_OUTPUT_ROOT" not in os.environ:
    os.environ["FASTGEN_OUTPUT_ROOT"] = os.path.join(FASTGEN_REPO, "FASTGEN_OUTPUT")

# Register custom folder paths so ComfyUI file-list dropdowns can browse them.
# Users can add more search directories via folder_paths.add_model_folder_path().
# folder_paths is provided by ComfyUI itself; a clear error beats a bare
# ModuleNotFoundError when these nodes are imported outside ComfyUI.
try:
    import folder_paths  # noqa: E402
except ImportError as e:
    raise RuntimeError(
        "FastGen ComfyUI nodes require ComfyUI's 'folder_paths' module, which could not be imported. "
        "Install these nodes under ComfyUI's 'custom_nodes' directory and launch them from ComfyUI "
        "(or add ComfyUI's root to PYTHONPATH)."
    ) from e

# Dedicated checkpoint directory — never fall back to a huge parent dir.
# Place monolithic .pth files here. For DCP, symlink or copy the complete
# containing checkpoint directory below this directory so each .pth marker
# stays beside its <name>.<key>_model shard directories. Alternatively, add
# the containing directory via folder_paths.add_model_folder_path().
_input_dir_base = folder_paths.get_input_directory()
_ckpt_dir = os.path.join(_input_dir_base, "fastgen_checkpoints")
os.makedirs(_ckpt_dir, exist_ok=True)
folder_paths.folder_names_and_paths.setdefault("fastgen_checkpoints", ([_ckpt_dir], {".pth"}))

from .nodes_config import FastGenConfigLoader  # noqa: E402
from .nodes_model import FastGenModelLoader  # noqa: E402
from .nodes_prompt import FastGenPromptExpand  # noqa: E402
from .nodes_condition import FastGenCameraLoader, FastGenDepthLatentLoader, FastGenPreprocess  # noqa: E402
from .nodes_sampling import FastGenNoiseGenerator, FastGenTeacherSample, FastGenStudentSample  # noqa: E402
from .nodes_decode import FastGenDecodeVideo  # noqa: E402

NODE_CLASS_MAPPINGS = {
    "FastGenConfigLoader": FastGenConfigLoader,
    "FastGenModelLoader": FastGenModelLoader,
    "FastGenPromptExpand": FastGenPromptExpand,
    "FastGenCameraLoader": FastGenCameraLoader,
    "FastGenDepthLatentLoader": FastGenDepthLatentLoader,
    "FastGenPreprocess": FastGenPreprocess,
    "FastGenNoiseGenerator": FastGenNoiseGenerator,
    "FastGenTeacherSample": FastGenTeacherSample,
    "FastGenStudentSample": FastGenStudentSample,
    "FastGenDecodeVideo": FastGenDecodeVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastGenConfigLoader": "FastGen Config Loader",
    "FastGenModelLoader": "FastGen Model Loader",
    "FastGenPromptExpand": "FastGen Prompt Expand",
    "FastGenCameraLoader": "FastGen Camera Loader",
    "FastGenDepthLatentLoader": "FastGen Depth Latent Loader",
    "FastGenPreprocess": "FastGen Preprocess",
    "FastGenNoiseGenerator": "FastGen Noise Generator",
    "FastGenTeacherSample": "FastGen Teacher Sample",
    "FastGenStudentSample": "FastGen Student Sample",
    "FastGenDecodeVideo": "FastGen Decode & Preview",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
