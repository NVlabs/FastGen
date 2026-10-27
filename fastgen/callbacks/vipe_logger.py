# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``VipeLogger`` — pluggable camera-trajectory ``LogModule`` for ``WandbCallback``.

Mirrors :class:`~fastgen.callbacks.vlm_logger.VLMLogger`: each logged step it
reads the ground-truth camera poses from **every rank's** ``data_batch``
(populated by ``CameraVideoWDSLoader``) — gathered
cross-rank via ``collect`` so rank 0 sees every logged clip, not just its
own — and ships camera diagnostics to W&B:

* a camera-trajectory plot per clip (GT, plus the VIPE estimate when enabled);
* WASD/IJKL **action overlays** drawn on every rank's saved student/real/teacher
  clips (the actions are derived from the camera poses), one tile per rank;
* when ``run_vipe_on_generated`` is set (and the optional ``vipe`` package is
  installed), pose / intrinsics error scalars from a frame-0-anchored,
  scale-aligned VIPE estimate of the generated clip.

The camera diagnostics are a self-contained, opt-in ``LogModule`` like the VLM
judge. The module is a no-op for batches without ``camera_poses`` /
``camera_intrinsics``.
"""

from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import wandb

from fastgen.callbacks.wandb import LogModule, MediaEntry
from fastgen.utils.camera_utils import (
    align_camera_trajectories,
    encode_video_with_vipe,
    plot_camera_trajectory,
    render_wasd_ijkl_overlay,
)
from fastgen.utils.tensor import as_array
from fastgen.utils import logging_utils as logger

if TYPE_CHECKING:
    from fastgen.callbacks.wandb import WandbCallback


class VipeLogger(LogModule):
    """``LogModule`` that logs camera-trajectory diagnostics for camera-conditioned models.

    Constructor knobs (all optional):

    * ``run_vipe_on_generated`` — when True, run VIPE on each generated student
      clip to estimate its camera trajectory and report scale-aligned pose /
      intrinsics error against the GT poses (averaged across clips). Requires the
      optional ``vipe`` package; a missing install or a VIPE failure is logged
      and skipped per clip (leaving just the GT trajectory plot). Default
      ``False``.
    * ``overlay_actions`` — when True (default), draw WASD/IJKL action-key
      overlays (derived from camera poses) on the saved student / real /
      teacher clips and log them as ``wandb.Video`` panels.
    * ``fps`` — frame rate for the logged overlay videos (default 16, the Wan
      training cadence).
    * ``pipeline`` — VIPE pipeline name forwarded to ``encode_video_with_vipe``.
    """

    # kind -> ("gt" | "gen") action source for the overlay: real and teacher
    # clips show GT camera actions, student clips show the generated trajectory
    # (VIPE estimate when available, else GT).
    _OVERLAY_KINDS = (("real", "gt"), ("student", "gen"), ("teacher", "gt"))

    def __init__(
        self,
        run_vipe_on_generated: bool = False,
        overlay_actions: bool = True,
        fps: int = 16,
        pipeline: str = "default",
    ) -> None:
        super().__init__()
        self.run_vipe_on_generated = run_vipe_on_generated
        self.overlay_actions = overlay_actions
        self.fps = fps
        self.pipeline = pipeline

    # ----- LogModule lifecycle (rank 0 only) ---------------------------------

    def setup(self, callback: "WandbCallback") -> None:
        super().setup(callback)
        logger.info(
            f"vipe: VipeLogger enabled (run_vipe_on_generated={self.run_vipe_on_generated}, "
            f"overlay_actions={self.overlay_actions})"
        )

    # ----- All-ranks collect (runs on every rank) ----------------------------

    def collect(self, data_batch: Optional[Dict[str, Any]]) -> Any:
        """Hand this rank's GT camera poses/intrinsics to rank 0 (all-gathered
        cross-rank by ``WandbCallback._step``).

        The poses live only in each rank's own ``data_batch``, so without this
        rank 0 could overlay only its own clip. Returning them here lets
        ``process`` overlay / plot **every** rank's clip.
        """
        if not data_batch or "camera_poses" not in data_batch or "camera_intrinsics" not in data_batch:
            return None
        return {
            "camera_poses": as_array(data_batch["camera_poses"], torch.float32),  # [B, F, 4, 4]
            "camera_intrinsics": as_array(data_batch["camera_intrinsics"], torch.float32),  # [B, F, 4]
        }

    # ----- Per-step processing (rank 0 only) ---------------------------------

    def process(
        self,
        save_dir: Path,
        entries: List[MediaEntry],
        iteration: int,
        group: str,
        step: int,
        data_batch: Optional[Dict[str, Any]],
        output_batch: Optional[Dict[str, Any]],
        gathered: Optional[List[Any]] = None,
    ) -> Optional[Dict[str, float]]:
        if wandb.run is None:
            return None
        # ``gathered`` is the per-rank list of ``collect`` outputs (each rank's
        # camera poses/intrinsics), so it carries the poses for EVERY logged
        # clip — overlays/plots span all ranks, not just rank 0's local batch.
        if not gathered or all(g is None for g in gathered):
            return None

        # Optional per-clip VIPE estimate of each generated (student) clip's
        # trajectory. Opt-in (needs the ``vipe`` package); isolated per clip.
        est_by_key: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        if self.run_vipe_on_generated:
            for e in entries:
                if e.kind != "student":
                    continue
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    est_by_key[(e.rank, e.batch_idx)] = encode_video_with_vipe(
                        e.path, output_path=None, pipeline=self.pipeline, keep_temp=False
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(f"vipe: VIPE estimation failed for rank{e.rank} — {exc}")
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

        # Camera-trajectory plot per generated clip (GT + estimate), shipped as
        # one multi-tile panel spanning every rank.
        metrics: Dict[str, List[float]] = defaultdict(list)
        try:
            import matplotlib.pyplot as plt

            traj_tiles: List[Any] = []
            for e in entries:
                if e.kind != "student":
                    continue
                poses_gt, intrinsics_gt = self._poses_for(e, gathered)
                if poses_gt is None:
                    continue
                poses_est, intrinsics_est = est_by_key.get((e.rank, e.batch_idx), (None, None))
                fig = plot_camera_trajectory(
                    poses_gt, poses_est, intrinsics=intrinsics_gt, intrinsics_est=intrinsics_est
                )
                traj_tiles.append(wandb.Image(fig, caption=f"rank{e.rank}"))
                plt.close(fig)
                if poses_est is not None:
                    self._accumulate_metrics(metrics, poses_gt, poses_est, intrinsics_gt, intrinsics_est)
            if traj_tiles:
                wandb.log({f"{group}_media/camera/trajectory": traj_tiles}, step=iteration)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"vipe: camera trajectory logging failed — {exc}")

        if self.overlay_actions:
            self._log_action_overlays(entries, group, iteration, gathered, est_by_key)

        return {k: float(np.mean(v)) for k, v in metrics.items()} or None

    # ----- Internals ---------------------------------------------------------

    def _poses_for(self, entry: MediaEntry, gathered: List[Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """``(poses, intrinsics)`` for ``entry``'s ``(rank, batch_idx)`` from the
        cross-rank ``gathered`` list — ``(None, None)`` when absent."""
        g = gathered[entry.rank] if (gathered and entry.rank < len(gathered)) else None
        if not g:
            return None, None
        cp, ci = g.get("camera_poses"), g.get("camera_intrinsics")
        if cp is None or entry.batch_idx >= len(cp):
            return None, None
        intr = ci[entry.batch_idx] if (ci is not None and entry.batch_idx < len(ci)) else None
        return cp[entry.batch_idx], intr

    @staticmethod
    def _accumulate_metrics(
        metrics: Dict[str, List[float]],
        poses_gt: np.ndarray,
        poses_est: np.ndarray,
        intrinsics_gt: Optional[np.ndarray],
        intrinsics_est: Optional[np.ndarray],
    ) -> None:
        """Append this clip's scale-aligned pose / intrinsics errors; the caller
        means them across clips."""
        _, _, align_metrics = align_camera_trajectories(poses_gt, poses_est)
        for src, dst in (
            ("ate_mean", "camera_translation_error"),
            ("ate_rel", "camera_translation_error_rel"),
            ("ate_max", "camera_translation_error_max"),
            ("sim3_scale", "camera_sim3_scale"),
            ("rotation_error_deg", "camera_rotation_error_deg"),
        ):
            if src in align_metrics:
                metrics[dst].append(float(align_metrics[src]))
        if intrinsics_est is not None and intrinsics_gt is not None:
            n_frames = min(len(intrinsics_gt), len(intrinsics_est))
            k_gt = np.asarray(intrinsics_gt[:n_frames], dtype=np.float64)
            k_est = np.asarray(intrinsics_est[:n_frames], dtype=np.float64)
            abs_err = np.abs(k_gt - k_est)
            metrics["camera_intrinsics_abs_error"].append(float(abs_err.mean()))
            # Dimensionless, resolution-independent: focal by GT focal,
            # principal point by ~image size (2*cx, 2*cy).
            denom = np.maximum(
                np.stack([k_gt[..., 0], k_gt[..., 1], 2 * k_gt[..., 2], 2 * k_gt[..., 3]], axis=-1),
                1e-6,
            )
            metrics["camera_intrinsics_rel_error"].append(float((abs_err / denom).mean()))

    def _log_action_overlays(
        self,
        entries: List[MediaEntry],
        group: str,
        iteration: int,
        gathered: List[Any],
        est_by_key: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Draw WASD/IJKL action overlays (from camera poses) on **every** saved
        clip and ship them per kind as one multi-tile ``wandb.Video`` panel (one
        tile per rank), mirroring ``WandbCallback._log_panels``.

        ``real`` and ``teacher`` show GT camera actions; ``student`` shows the
        generated trajectory's actions (per-clip VIPE estimate when available,
        else GT). Best-effort and per-clip isolated — a decode / OpenCV failure
        on one clip logs a warning and skips just that tile.
        """
        import imageio.v3 as iio

        tiles_by_kind: Dict[str, List[Any]] = defaultdict(list)
        for kind, source in self._OVERLAY_KINDS:
            for e in entries:
                if e.kind != kind:
                    continue
                poses_gt, _ = self._poses_for(e, gathered)
                if poses_gt is None:
                    continue
                if source == "gen":
                    est = est_by_key.get((e.rank, e.batch_idx), (None, None))[0]
                    poses = est if est is not None else poses_gt
                else:
                    poses = poses_gt
                try:
                    frames = np.asarray(iio.imread(e.path))  # [F, H, W, 3] uint8
                    if frames.ndim != 4 or frames.shape[-1] != 3:
                        continue
                    overlaid = render_wasd_ijkl_overlay(frames, c2ws=poses)
                    # wandb.Video wants (time, channel, height, width) uint8.
                    vid = np.ascontiguousarray(np.asarray(overlaid).transpose(0, 3, 1, 2))
                    tiles_by_kind[kind].append(wandb.Video(vid, fps=self.fps, format="mp4", caption=f"rank{e.rank}"))
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(f"vipe: action overlay for kind={kind} rank{e.rank} failed — {exc}")
        for kind, tiles in tiles_by_kind.items():
            if tiles:
                wandb.log({f"{group}_media/camera/{kind}_actions": tiles}, step=iteration)
