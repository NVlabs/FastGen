# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the camera utilities: Plücker rays, the trajectory alignment behind the VIPE metrics, and VIPE runs."""

import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from fastgen.utils.camera_utils import (
    align_camera_trajectories,
    encode_video_with_vipe,
    extract_ijkl_from_poses,
    extract_wasd_from_poses,
    get_plucker_embeddings,
)


@pytest.mark.parametrize("extract", [extract_wasd_from_poses, extract_ijkl_from_poses])
@pytest.mark.parametrize("num_frames", [1, 3])
@pytest.mark.parametrize("normalize", [False, True])
def test_stationary_camera_actions_are_zero(extract, num_frames, normalize):
    poses = np.repeat(np.eye(4)[None], num_frames, axis=0)
    actions = extract(poses, normalize=normalize)

    assert actions.dtype == np.float32
    np.testing.assert_array_equal(actions, np.zeros((num_frames, 4), dtype=np.float32))


def _dolly_with_pan(n: int = 81, pan_deg: float = 8.0) -> np.ndarray:
    """A straight forward path with a slow pan, as (n, 4, 4) camera-to-world poses."""
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = Rotation.from_euler("y", np.linspace(0.0, pan_deg, n), degrees=True).as_matrix()
    poses[:, 2, 3] = np.linspace(0.0, 2.0, n)
    return poses


def test_plucker_rays_ignore_an_enclosing_autocast():
    """Rays keep the poses' FP32 under an enclosing autocast, such as encoder AMP around conditioning."""
    c2ws = torch.eye(4).repeat(3, 1, 1)
    c2ws[:, :3, :3] = torch.as_tensor(
        Rotation.from_euler("y", [0.0, 7.0, 15.0], degrees=True).as_matrix(), dtype=torch.float32
    )
    c2ws[:, :3, 3] = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.3], [0.2, -0.1, 0.6]])
    intrinsics = torch.tensor([[60.0, 60.0, 16.0, 12.0]]).repeat(3, 1)

    expected = get_plucker_embeddings(c2ws, intrinsics, height=24, width=32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = get_plucker_embeddings(c2ws, intrinsics, height=24, width=32)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_exact_estimate_of_a_straight_path_in_another_frame_and_scale_has_no_error():
    gt = _dolly_with_pan()
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("xyz", [30.0, -50.0, 110.0], degrees=True).as_matrix()
    world[:3, 3] = [3.0, -1.0, 7.0]
    est = world[None] @ gt
    est[:, :3, 3] *= 0.1

    _, est_aligned, metrics = align_camera_trajectories(gt, est)

    assert metrics["rotation_error_deg"] < 1e-4
    assert metrics["ate_rel"] < 1e-9
    assert np.isclose(metrics["sim3_scale"], 10.0)
    assert est_aligned.shape == gt.shape


def test_position_noise_on_a_straight_path_does_not_create_rotation_error():
    gt = _dolly_with_pan()
    est = gt.copy()
    est[1:, :2, 3] += np.random.default_rng(0).normal(scale=0.01, size=(len(gt) - 1, 2))

    _, _, metrics = align_camera_trajectories(gt, est)

    assert metrics["rotation_error_deg"] < 1e-4
    assert metrics["ate_rel"] < 0.02


def test_rotation_error_is_the_mean_angle_between_anchored_orientations():
    gt = _dolly_with_pan()
    est = _dolly_with_pan(pan_deg=16.0)

    _, _, metrics = align_camera_trajectories(gt, est)

    assert np.isclose(metrics["rotation_error_deg"], 4.0)
    assert metrics["ate_rel"] < 1e-9


def test_estimate_moving_the_wrong_way_is_not_mirrored_onto_gt():
    gt = _dolly_with_pan()
    est = gt.copy()
    est[:, :3, 3] *= -1.0

    _, _, metrics = align_camera_trajectories(gt, est)

    assert metrics["sim3_scale"] == 0.0
    assert np.isclose(metrics["ate_rel"], 0.5)


def test_non_finite_estimate_frames_leave_the_others_paired_with_their_gt_frames():
    gt = _dolly_with_pan()
    est = gt.copy()
    est[:, :3, 3] *= 0.5
    est[5] = np.nan

    _, est_aligned, metrics = align_camera_trajectories(gt, est)

    assert metrics["ate_rel"] < 1e-9
    assert metrics["rotation_error_deg"] < 1e-4
    assert np.isclose(metrics["sim3_scale"], 2.0)
    assert est_aligned.shape == gt.shape
    assert np.isnan(est_aligned[5]).all()
    assert np.isfinite(np.delete(est_aligned, 5, axis=0)).all()


def test_estimate_with_a_non_finite_first_frame_is_not_aligned():
    gt = _dolly_with_pan()
    est = gt.copy()
    est[0] = np.nan

    _, est_aligned, metrics = align_camera_trajectories(gt, est)

    assert est_aligned is None
    assert "ate_mean" not in metrics


def test_vipe_temporary_directory_is_removed_when_composing_the_config_fails(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    temp_dir = tmp_path / "vipe_out"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: str(temp_dir))

    def failing_config_dir(**kwargs):
        raise RuntimeError("bad VIPE config")

    # Stand-ins for the optional VIPE install; composing its config fails.
    modules = {
        "hydra": SimpleNamespace(initialize_config_dir=failing_config_dir),
        "vipe": SimpleNamespace(get_config_path=lambda: str(tmp_path), make_pipeline=None),
        "vipe.streams": SimpleNamespace(),
        "vipe.streams.base": SimpleNamespace(ProcessedVideoStream=None),
        "vipe.streams.raw_mp4_stream": SimpleNamespace(RawMp4Stream=None),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(RuntimeError, match="bad VIPE config"):
        encode_video_with_vipe(video)

    assert not temp_dir.exists()
