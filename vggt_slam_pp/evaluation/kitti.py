"""Strict conversion of KITTI poses to TUM trajectory rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


_ROTATION_TOLERANCE = 1e-5


def read_kitti_pose_rows(path: Path) -> np.ndarray:
    poses: list[np.ndarray] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        fields = stripped.split()
        if len(fields) != 12:
            raise ValueError(
                f"{path}:{line_number}: expected 12 columns, got {len(fields)}"
            )
        try:
            values = np.asarray([float(field) for field in fields], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric KITTI row") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{path}:{line_number}: KITTI rows must contain only finite values"
            )

        pose = values.reshape(3, 4)
        rotation = pose[:, :3]
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3, dtype=np.float64),
            rtol=0.0,
            atol=_ROTATION_TOLERANCE,
        ) or not np.isclose(
            np.linalg.det(rotation),
            1.0,
            rtol=0.0,
            atol=_ROTATION_TOLERANCE,
        ):
            raise ValueError(f"{path}:{line_number}: invalid rotation matrix")
        poses.append(pose)

    if len(poses) < 2:
        raise ValueError("KITTI trajectories require at least two poses")

    rows = np.empty((len(poses), 8), dtype=np.float64)
    for frame_id, pose in enumerate(poses):
        rows[frame_id, 0] = frame_id
        rows[frame_id, 1:4] = pose[:, 3]
        rows[frame_id, 4:8] = Rotation.from_matrix(pose[:, :3]).as_quat()
    return rows
