# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Camera pose utilities: VIPE I/O, on-the-fly pose estimation, and visualization helpers.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Camera math
# ---------------------------------------------------------------------------


def transform_intrinsics(intrinsics: torch.Tensor, cropping_params: dict) -> torch.Tensor:
    """Transform camera intrinsics [N, 4] to account for resize and crop.

    Args:
        intrinsics: [N, 4] as [fx, fy, cx, cy].
        cropping_params: Dict with resize_w, resize_h, orig_w, orig_h, crop_x0, crop_y0.
    """
    orig_dtype = intrinsics.dtype
    intrinsics = intrinsics.float()
    scale_x = cropping_params["resize_w"] / cropping_params["orig_w"]
    scale_y = cropping_params["resize_h"] / cropping_params["orig_h"]
    result = torch.stack(
        [
            intrinsics[:, 0] * scale_x,
            intrinsics[:, 1] * scale_y,
            intrinsics[:, 2] * scale_x - cropping_params["crop_x0"],
            intrinsics[:, 3] * scale_y - cropping_params["crop_y0"],
        ],
        dim=-1,
    )
    return result.to(orig_dtype)


def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    """Interpolate camera poses at target frame indices using SLERP for rotation.

    Target indices must lie within the source index span; neither rotation nor translation extrapolates.
    """
    interp_func_trans = interp1d(src_indices, src_trans_vec, axis=0, kind="linear")
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    quats = src_quat_vec.as_quat().copy()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4))
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Compute inverse of SE3 transformation matrices."""
    Rot = T[:, :3, :3]
    trans = T[:, :3, 3:]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(T.shape[0], 1, 1)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    """Compute relative poses w.r.t. first frame."""
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        relative_poses_framewise = torch.bmm(SE3_inverse(relative_poses[:-1]), relative_poses[1:])
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:
        translations = relative_poses[:, :3, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    bias: float = 0.5,
    device="cuda",
    dtype=torch.float32,
) -> torch.Tensor:
    """Create pixel grid for ray computation."""
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias
    grid_xy = grid_xy[None, ...].repeat(n_frames, 1, 1)
    return grid_xy


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Compute per-pixel ``[origin, direction]`` ray embeddings from camera poses and intrinsics.

    Returns ``[T, H, W, 6]`` rows of the camera origin followed by the unit world-space ray direction,
    the embedding the shipped WanCamI2V weights were trained on; it is not the ``(d, o x d)`` Plücker form.
    """
    n_frames = c2ws_mat.shape[0]
    # Ray directions are computed in fp32: pixel grids and intrinsics carry large
    # magnitudes that lose precision in low-precision dtypes (e.g., bf16 cannot
    # represent integer pixel coordinates above 256 exactly).
    grid_xy = create_meshgrid(n_frames, height, width, device=c2ws_mat.device, dtype=torch.float32)
    fx, fy, cx, cy = Ks.float().chunk(4, dim=-1)

    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    # Pinhole unprojection at unit depth (z = 1)
    xs = (i - cx) / fx
    ys = (j - cy) / fy
    zs = torch.ones_like(i)

    directions = torch.stack([xs, ys, zs], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # An enclosing autocast (e.g. encoder AMP around ``build_conditioning``) would lower this rotation.
    with torch.autocast(device_type=c2ws_mat.device.type, enabled=False):
        rays_d = directions.to(c2ws_mat.dtype) @ c2ws_mat[:, :3, :3].transpose(-1, -2)
    rays_o = c2ws_mat[:, :3, 3]
    rays_o = rays_o[:, None, :].expand_as(rays_d)
    plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1)
    plucker_embeddings = plucker_embeddings.view([n_frames, height, width, 6])
    return plucker_embeddings


# ---------------------------------------------------------------------------
# VIPE I/O
# ---------------------------------------------------------------------------


def load_vipe_poses(pose_path: Union[str, Path]) -> np.ndarray:
    """Load poses from VIPE npz format."""
    data = np.load(pose_path)
    pose_data = data["data"]
    inds = data["inds"]
    order = np.argsort(inds)
    return pose_data[order]


def load_vipe_intrinsics(intrinsics_path: Union[str, Path]) -> np.ndarray:
    """Load intrinsics from VIPE npz format."""
    data = np.load(intrinsics_path)
    intr_data = data["data"]
    inds = data["inds"]
    order = np.argsort(inds)
    return intr_data[order]


def prepare_camera_poses_from_vipe(
    vipe_base_path: Union[str, Path],
    artifact_name: Optional[str] = None,
    video_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera poses and intrinsics from VIPE output.

    Returns:
        Tuple of (poses [N,4,4], intrinsics [N,4]).
    """
    vipe_base_path = Path(vipe_base_path)

    if artifact_name is None:
        if video_path is not None:
            artifact_name = Path(video_path).stem
        else:
            pose_dir = vipe_base_path / "pose"
            if pose_dir.exists():
                npz_files = list(pose_dir.glob("*.npz"))
                if npz_files:
                    artifact_name = npz_files[0].stem
                else:
                    raise ValueError(f"No pose artifacts found in {pose_dir}. " "Run VIPE on the video first.")
            else:
                raise ValueError(f"VIPE base path has no pose/ directory: {vipe_base_path}")

    poses = load_vipe_poses(vipe_base_path / "pose" / f"{artifact_name}.npz")
    intrinsics = load_vipe_intrinsics(vipe_base_path / "intrinsics" / f"{artifact_name}.npz")

    n_min = min(len(poses), len(intrinsics))
    poses = poses[:n_min]
    intrinsics = intrinsics[:n_min]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        np.save(output_path / "poses.npy", poses)
        np.save(output_path / "intrinsics.npy", intrinsics)

    return poses, intrinsics


# ---------------------------------------------------------------------------
# On-the-fly VIPE estimation
# ---------------------------------------------------------------------------


def encode_video_with_vipe(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    pipeline: str = "default",
    keep_temp: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run VIPE on-the-fly on a video and return camera poses + intrinsics."""
    import tempfile

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        import hydra
        from vipe import get_config_path, make_pipeline
        from vipe.streams.base import ProcessedVideoStream
        from vipe.streams.raw_mp4_stream import RawMp4Stream
    except ImportError as e:
        raise ImportError(
            "VIPE is required for on-the-fly encoding. Install NVIDIA ViPE (https://github.com/nv-tlabs/vipe), "
            "published on PyPI as nvidia-vipe; the PyPI package named vipe is unrelated."
        ) from e

    use_temp = output_path is None
    if use_temp:
        temp_dir = tempfile.mkdtemp(prefix="vipe_wancami2v_")
        vipe_out = Path(temp_dir)
    else:
        vipe_out = Path(output_path)
        vipe_out.mkdir(parents=True, exist_ok=True)

    # Drop the AdaptiveDepthProcessor, since callers only read poses +
    # intrinsics, and the moving-object masks, whose GroundingDINO detector
    # needs Transformers 4 (BertModel.get_head_mask).
    overrides = [
        f"pipeline={pipeline}",
        f"pipeline.output.path={vipe_out}",
        "pipeline.init.instance=null",
        "pipeline.post.depth_align_model=null",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz=false",
    ]

    try:
        with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
            args = hydra.compose("default", overrides)
        vipe_pipeline = make_pipeline(args.pipeline)
        video_stream = ProcessedVideoStream(RawMp4Stream(video_path), []).cache(desc="Reading video stream")
        vipe_pipeline.run(video_stream)

        artifact_name = video_path.stem
        poses, intrinsics = prepare_camera_poses_from_vipe(
            vipe_out,
            artifact_name=artifact_name,
            output_path=output_path if not use_temp else None,
        )
    finally:
        # Remove the temporary directory even when composing the config or running VIPE fails.
        if use_temp and not keep_temp:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    return poses, intrinsics


# ---------------------------------------------------------------------------
# Action derivation from camera poses (WASD translation, IJKL rotation)
# ---------------------------------------------------------------------------


def _rotation_matrix_to_euler_zyx(Rot: np.ndarray) -> Tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to ZYX Euler angles, returned as ``(z, y, x)``.

    For OpenCV camera axes (z forward, y down, x right) these are ``(roll, yaw, pitch)``, which is how
    ``extract_ijkl_from_poses`` consumes them.

    Uses the OpenCV/robotics convention where the singular case is handled at
    near-gimbal-lock (sy < 1e-6).
    """
    sy = np.sqrt(Rot[0, 0] ** 2 + Rot[1, 0] ** 2)
    if sy < 1e-6:
        z_angle = 0.0
        y_angle = float(np.arctan2(-Rot[2, 0], Rot[0, 0]))
        x_angle = float(np.arctan2(-Rot[1, 2], Rot[1, 1]))
    else:
        z_angle = float(np.arctan2(Rot[1, 0], Rot[0, 0]))
        y_angle = float(np.arctan2(-Rot[2, 0], sy))
        x_angle = float(np.arctan2(Rot[2, 1], Rot[2, 2]))
    return z_angle, y_angle, x_angle


def extract_wasd_from_poses(
    c2ws: np.ndarray,
    threshold: float = 0.01,
    normalize: bool = False,
) -> np.ndarray:
    """Derive per-frame WASD one-hot from a c2w trajectory via dominant camera-frame translation.

    Args:
        c2ws: [N, 4, 4] camera-to-world SE(3) matrices.
        threshold: Minimum translation magnitude to register a key press. With
            ``normalize=False`` (default) this is the raw camera-frame
            translation magnitude per frame; the default ``0.01`` was tuned
            for the lingbot game-engine corpus and is meaningful for any
            16 fps dataset whose poses live on a similar scale (including
            real-world camera-annotated datasets after FPS resampling). With
            ``normalize=True`` it is instead a *fraction of the maximum
            per-frame translation in the clip*, useful as an opt-in
            escape hatch for future datasets at very different pose scales.
        normalize: If True, divide the per-frame camera-frame translations by
            their max norm across the clip before thresholding. Default
            ``False`` preserves the original lingbot overlay behavior.

    Returns:
        [N, 4] one-hot array [W, A, S, D] = [forward, left, backward, right].
        The last frame is always zeros (no delta).
    """
    c2ws = np.asarray(c2ws, dtype=np.float32)
    assert c2ws.ndim == 3 and c2ws.shape[-2:] == (4, 4), f"expected [N,4,4], got {c2ws.shape}"

    c2w_a_inv = np.linalg.inv(c2ws[:-1])
    T_rels = np.matmul(c2w_a_inv, c2ws[1:])

    trans = T_rels[:, :3, 3]  # (N-1, 3)  [right, down, fwd]
    if normalize and trans.size:
        max_norm = float(np.linalg.norm(trans, axis=-1).max())
        if max_norm > 0:
            trans = trans / max_norm

    wasd = np.zeros((len(c2ws), 4), dtype=np.float32)
    for i in range(trans.shape[0]):
        right, _down, forward = trans[i]
        if max(abs(right), abs(forward)) < threshold:
            continue
        if abs(forward) > abs(right):
            wasd[i, 0 if forward > 0 else 2] = 1.0  # W or S
        else:
            wasd[i, 3 if right > 0 else 1] = 1.0  # D or A
    return wasd


def extract_ijkl_from_poses(
    c2ws: np.ndarray,
    threshold: float = 0.002,
    normalize: bool = False,
) -> np.ndarray:
    """Derive per-frame IJKL one-hot from a c2w trajectory via dominant rotation axis.

    Args:
        c2ws: [N, 4, 4] camera-to-world SE(3) matrices.
        threshold: Minimum rotation magnitude to register a key press. With
            ``normalize=False`` (default) this is the raw per-frame absolute
            Euler-angle magnitude in radians. The default ``0.002`` rad
            (~0.11 deg/frame at 16 fps) is loose enough to pick up gentle
            real-world camera rotations (median ~0.0017 rad) while still
            meaningful for lingbot's harder pans.
            The original lingbot-only default was ``0.005`` rad. With
            ``normalize=True`` it is interpreted as a *fraction of the
            maximum per-frame absolute Euler angle in the clip*.
        normalize: If True, divide the per-frame rotation magnitudes by their
            max across the clip before thresholding. Default ``False``
            keeps the raw-radians semantics.

    Returns:
        [N, 4] one-hot array [I, J, K, L] = [pitch-up, yaw-left, pitch-down, yaw-right].
        The last frame is always zeros (no delta).
    """
    c2ws = np.asarray(c2ws, dtype=np.float32)
    assert c2ws.ndim == 3 and c2ws.shape[-2:] == (4, 4), f"expected [N,4,4], got {c2ws.shape}"

    c2w_a_inv = np.linalg.inv(c2ws[:-1])
    T_rels = np.matmul(c2w_a_inv, c2ws[1:])

    # Decompose each framewise rotation into Euler ZYX (roll, yaw, pitch).
    eulers = np.array(
        [_rotation_matrix_to_euler_zyx(T_rels[i, :3, :3]) for i in range(T_rels.shape[0])],
        dtype=np.float32,
    )  # (N-1, 3)  [roll, yaw, pitch]
    if normalize and eulers.size:
        max_mag = float(np.abs(eulers).max())
        if max_mag > 0:
            eulers = eulers / max_mag

    ijkl = np.zeros((len(c2ws), 4), dtype=np.float32)
    for i in range(eulers.shape[0]):
        z, yaw_right, pitch_up = eulers[i]
        mag_z, mag_yaw, mag_pitch = abs(z), abs(yaw_right), abs(pitch_up)
        m = max(mag_z, mag_yaw, mag_pitch)
        if m < threshold:
            continue
        if m == mag_yaw:
            ijkl[i, 3 if yaw_right > 0 else 1] = 1.0  # L or J
        elif m == mag_pitch:
            ijkl[i, 0 if pitch_up > 0 else 2] = 1.0  # I or K
    return ijkl


# ---------------------------------------------------------------------------
# WASD/IJKL overlay rendering
# ---------------------------------------------------------------------------

# BGR colors for OpenCV
_WASD_ACTIVE_BGR = (0, 100, 200)  # blue
_IJKL_ACTIVE_BGR = (200, 100, 0)  # orange
_KEY_INACTIVE_BGR = (50, 50, 50)
_KEY_TEXT_BGR = (255, 255, 255)


def _draw_rounded_rect(img, pt1, pt2, color, r=5, alpha=0.7):
    """Alpha-blend a filled rounded rectangle onto img (in-place)."""
    import cv2

    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, -1)
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def _draw_key_cluster(img, top_left, labels_and_active, active_color):
    """Draw a 4-key cross cluster (top, left, center, right) with given (label, is_active)."""
    import cv2

    key_size, spacing = 40, 5
    x, y = top_left
    positions = [
        (x + key_size + spacing, y),  # top
        (x, y + key_size + spacing),  # left
        (x + key_size + spacing, y + key_size + spacing),  # center / bottom-center
        (x + 2 * (key_size + spacing), y + key_size + spacing),  # right
    ]
    for (label, active), (kx, ky) in zip(labels_and_active, positions):
        color = active_color if active else _KEY_INACTIVE_BGR
        _draw_rounded_rect(img, (kx, ky), (kx + key_size, ky + key_size), color)
        font_scale, thickness = 0.6, 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.putText(
            img,
            label,
            (kx + (key_size - tw) // 2, ky + (key_size + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _KEY_TEXT_BGR,
            thickness,
            cv2.LINE_AA,
        )
    return img


def render_wasd_ijkl_overlay(
    frames: np.ndarray,
    c2ws: Optional[np.ndarray] = None,
    wasd: Optional[np.ndarray] = None,
    ijkl: Optional[np.ndarray] = None,
    translation_threshold: float = 0.01,
    rotation_threshold: float = 0.002,
) -> np.ndarray:
    """Render WASD (bottom-left, blue) and IJKL (bottom-right, orange) overlays onto video frames.

    One of {c2ws, wasd+ijkl} must be provided; missing action arrays are auto-derived
    from c2ws via ``extract_wasd_from_poses`` / ``extract_ijkl_from_poses``.

    Args:
        frames: [F, H, W, 3] uint8 RGB, or float in [0, 1].
        c2ws: Optional [F, 4, 4] camera poses used to derive actions when not given.
        wasd: Optional [F, 4] WASD one-hot (overrides auto-derivation).
        ijkl: Optional [F, 4] IJKL one-hot (overrides auto-derivation).

    Returns:
        Same shape and dtype as input frames, with overlays drawn.
    """
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "render_wasd_ijkl_overlay requires OpenCV. On headless systems install "
            "opencv-python-headless (pip install opencv-python-headless)."
        ) from e

    assert frames.ndim == 4 and frames.shape[-1] == 3, f"expected [F,H,W,3], got {frames.shape}"

    if wasd is None and ijkl is None and c2ws is None:
        raise ValueError("Must provide either c2ws or wasd+ijkl arrays.")
    if wasd is None:
        wasd = extract_wasd_from_poses(c2ws, translation_threshold) if c2ws is not None else None
    if ijkl is None:
        ijkl = extract_ijkl_from_poses(c2ws, rotation_threshold) if c2ws is not None else None

    input_was_float = np.issubdtype(frames.dtype, np.floating)
    if input_was_float:
        frames_u8 = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        frames_u8 = frames.astype(np.uint8, copy=True)

    # RGB -> BGR for OpenCV drawing
    bgr = frames_u8[..., ::-1].copy()
    F, H, W, _ = bgr.shape

    wasd_labels = ["W", "A", "S", "D"]
    ijkl_labels = ["I", "J", "K", "L"]

    key_size, spacing, margin = 40, 5, 50
    cluster_y = H - margin - 2 * (key_size + spacing) + spacing
    wasd_tl = (margin, cluster_y)
    ijkl_tl = (W - margin - 3 * (key_size + spacing) + spacing, cluster_y)

    n = min(F, len(wasd) if wasd is not None else F, len(ijkl) if ijkl is not None else F)
    for i in range(n):
        frame = bgr[i]
        if wasd is not None:
            _draw_key_cluster(
                frame,
                wasd_tl,
                list(zip(wasd_labels, wasd[i].astype(bool))),
                _WASD_ACTIVE_BGR,
            )
        if ijkl is not None:
            _draw_key_cluster(
                frame,
                ijkl_tl,
                list(zip(ijkl_labels, ijkl[i].astype(bool))),
                _IJKL_ACTIVE_BGR,
            )

    out = bgr[..., ::-1]  # BGR -> RGB
    if input_was_float:
        return out.astype(frames.dtype) / 255.0
    return out.astype(frames.dtype)


# ---------------------------------------------------------------------------
# Trajectory alignment (for both metrics and visualization)
# ---------------------------------------------------------------------------


def _anchor_poses(poses: np.ndarray) -> np.ndarray:
    """Return poses re-expressed in the frame-0 camera coordinate system.

    After anchoring, ``poses[0]`` is the identity and every subsequent pose is
    ``T0^{-1} @ T[i]``, so translations represent motion relative to the start
    (not absolute world coordinates).
    """
    T0_inv = np.linalg.inv(poses[0])
    return T0_inv[None] @ poses


def align_camera_trajectories(
    poses_gt: np.ndarray,
    poses_est: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Anchor both trajectories to their first camera and scale the estimate onto GT.

    The estimated trajectory from monocular SLAM (e.g. VIPE) lives in its own
    world frame and is only recovered up to scale. Comparing it to GT in the
    dataset's absolute world frame (which for e.g. ``lingbot_world`` sits
    thousands of meters from the origin) produces essentially arbitrary
    translation/rotation "errors". Re-expressing each trajectory in its own
    first camera's frame puts both in one coordinate system: rotations are
    directly comparable and positions differ only by scale. No rotation is
    fitted, because a rotation fitted to camera positions is unconstrained
    about the path axis for the near-straight paths of short clips.

    Args:
        poses_gt: (T, 4, 4) camera-to-world poses for the ground-truth clip.
        poses_est: (T', 4, 4) camera-to-world poses estimated on the generated
            clip (from VIPE), or ``None``.

    Returns:
        ``(poses_gt_anchored, poses_est_aligned, metrics)`` where::

            poses_gt_anchored: (T, 4, 4), poses_gt[0] == identity.
            poses_est_aligned: (min(T, T'), 4, 4) anchored with positions scaled
                onto GT, frame by frame (non-finite input frames stay non-finite and
                are left out of the metrics), or None if poses_est is None or has a
                non-finite first frame or fewer than two finite frames.
            metrics: dict with keys (all floats, some optional):
                gt_path_length  - sum ||Δt|| of anchored GT.
                gt_motion_max   - max ||t|| of anchored GT (useful denominator).
                ate_mean/median/max - per-frame ||t_est_aligned - t_gt|| stats.
                ate_rel         - ate_mean / gt_motion_max (0 when GT is static).
                sim3_scale      - least-squares scale of the anchored est
                    positions onto GT (est -> gt), clamped at 0.
                rotation_error_deg - mean angle between the anchored est and
                    GT rotations (scale free).
    """
    poses_gt = np.asarray(poses_gt, dtype=np.float64)
    gt_anch = _anchor_poses(poses_gt)
    t_gt = gt_anch[:, :3, 3]
    gt_motion_max = float(np.linalg.norm(t_gt, axis=1).max())
    gt_path_length = float(np.linalg.norm(np.diff(t_gt, axis=0), axis=1).sum()) if len(t_gt) > 1 else 0.0
    metrics: dict = {
        "gt_path_length": gt_path_length,
        "gt_motion_max": gt_motion_max,
    }

    if poses_est is None:
        return gt_anch, None, metrics

    poses_est = np.asarray(poses_est, dtype=np.float64)
    n = min(len(gt_anch), len(poses_est))
    if n < 2:
        return gt_anch, None, metrics
    # Skip frames with non-finite poses so SLAM glitches don't poison the fit; the others stay paired with
    # their GT frame. Frame 0 anchors both trajectories, so it must be finite.
    finite = np.isfinite(poses_est[:n]).reshape(n, -1).all(axis=1)
    if not finite[0] or finite.sum() < 2:
        return gt_anch, None, metrics

    est_anch = _anchor_poses(poses_est[:n])
    t_est = est_anch[finite, :3, 3]
    t_gt_n = t_gt[:n][finite]

    # Least-squares scale of positions that both start at the origin. A negative
    # scale would mirror an estimate that moves the wrong way onto GT.
    est_sq = float((t_est**2).sum())
    scale = max(float((t_est * t_gt_n).sum()) / est_sq, 0.0) if est_sq > 0 else 1.0
    est_aligned = est_anch.copy()
    est_aligned[:, :3, 3] *= scale

    ate = np.linalg.norm(scale * t_est - t_gt_n, axis=1)
    metrics["ate_mean"] = float(ate.mean())
    metrics["ate_median"] = float(np.median(ate))
    metrics["ate_max"] = float(ate.max())
    metrics["ate_rel"] = float(ate.mean() / gt_motion_max) if gt_motion_max > 0 else 0.0
    metrics["sim3_scale"] = scale

    # Rotation error: angle between R_gt_anch[i] and R_est_anch[i].
    rel = np.einsum("nij,nkj->nik", gt_anch[:n][finite, :3, :3], est_anch[finite, :3, :3])  # R_gt @ R_est^T
    cos = (np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5
    cos = np.clip(cos, -1.0, 1.0)
    metrics["rotation_error_deg"] = float(np.degrees(np.arccos(cos)).mean())

    return gt_anch, est_aligned, metrics


# ---------------------------------------------------------------------------
# Visualization (matplotlib / wandb)
# ---------------------------------------------------------------------------


def _set_axes_equal_3d(ax) -> None:
    """Make a 3D matplotlib plot use equal scaling so shapes aren't distorted."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * (limits[:, 1] - limits[:, 0]).max()
    radius = max(radius, 1e-6)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_camera_trajectory(
    poses_gt: np.ndarray,
    poses_est: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    intrinsics_est: Optional[np.ndarray] = None,
    title: str = "Camera Trajectory",
) -> "Figure":
    """Plot camera trajectory: 3D path (left) and position-over-time (right).

    Both trajectories are re-expressed in the frame-0 camera frame before
    plotting, and (when an estimate is provided) the estimated positions are
    scaled onto GT. This removes the dataset's absolute world offset
    and the monocular-SLAM scale ambiguity so the plotted shapes are directly
    comparable. Alignment stats (ATE, scale, rotation) are written to the
    figure suptitle.

    Args:
        poses_gt: (T, 4, 4) GT camera-to-world poses.
        poses_est: Optional (T', 4, 4) estimated poses (e.g. from VIPE).
        intrinsics: Optional (T, 4) or (4,) GT intrinsics [fx, fy, cx, cy].
        intrinsics_est: Optional estimated intrinsics (same format).
        title: Figure suptitle (alignment stats are appended).
    """
    import matplotlib.pyplot as plt  # plotting only; keep it off the dataloader import path

    gt_anch, est_aligned, align_metrics = align_camera_trajectories(poses_gt, poses_est)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    pos_gt = gt_anch[:, :3, 3]
    ax1.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], "b-o", label="GT", markersize=3, linewidth=1)
    ax1.scatter(pos_gt[0, 0], pos_gt[0, 1], pos_gt[0, 2], c="green", s=100, marker="^", label="Start")

    if est_aligned is not None:
        pos_est = est_aligned[:, :3, 3]
        ax1.plot(
            pos_est[:, 0],
            pos_est[:, 1],
            pos_est[:, 2],
            "r-x",
            label="VIPE Est. (scaled)",
            markersize=3,
            linewidth=1,
        )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Camera Path (anchored to frame 0)")
    ax1.legend()
    _set_axes_equal_3d(ax1)

    ax2 = fig.add_subplot(122)
    frames = np.arange(len(pos_gt))
    ax2.plot(frames, pos_gt[:, 0], "b-", label="GT X", alpha=0.7)
    ax2.plot(frames, pos_gt[:, 1], "g-", label="GT Y", alpha=0.7)
    ax2.plot(frames, pos_gt[:, 2], "r-", label="GT Z", alpha=0.7)

    if est_aligned is not None:
        pos_est = est_aligned[:, :3, 3]
        ax2.plot(frames[: len(pos_est)], pos_est[:, 0], "b--", label="Est X", alpha=0.7)
        ax2.plot(frames[: len(pos_est)], pos_est[:, 1], "g--", label="Est Y", alpha=0.7)
        ax2.plot(frames[: len(pos_est)], pos_est[:, 2], "r--", label="Est Z", alpha=0.7)

    def _intr1(x):
        if x is None:
            return None
        x = np.asarray(x)
        return x[0] if x.ndim == 2 else x

    k_gt = _intr1(intrinsics)
    k_est = _intr1(intrinsics_est)
    if k_gt is not None or k_est is not None:
        lines = []
        if k_gt is not None:
            lines.append(f"GT : fx={k_gt[0]:.1f}  fy={k_gt[1]:.1f}  cx={k_gt[2]:.1f}  cy={k_gt[3]:.1f}")
        if k_est is not None:
            lines.append(f"Est: fx={k_est[0]:.1f}  fy={k_est[1]:.1f}  cx={k_est[2]:.1f}  cy={k_est[3]:.1f}")
        ax2.text(
            0.02,
            0.02,
            "\n".join(lines),
            transform=ax2.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Position (anchored)")
    ax2.set_title("Camera Position over Time")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    subtitle_bits = [f"GT motion max={align_metrics['gt_motion_max']:.3f}"]
    if "ate_mean" in align_metrics:
        subtitle_bits += [
            f"ATE mean={align_metrics['ate_mean']:.3f}",
            f"ATE rel={align_metrics['ate_rel']:.3f}",
            f"scale={align_metrics['sim3_scale']:.3g}",
            f"rot err={align_metrics['rotation_error_deg']:.1f}°",
        ]
    fig.suptitle(title + "\n" + "  |  ".join(subtitle_bits), fontsize=10)
    plt.tight_layout()
    return fig
