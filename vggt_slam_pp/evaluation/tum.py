"""Strict TUM trajectory parsing and duplicate-timestamp canonicalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _validate_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("TUM trajectories must have exactly 8 columns")
    if values.shape[0] < 2:
        raise ValueError("TUM trajectories require at least two poses")
    if not np.all(np.isfinite(values)):
        raise ValueError("TUM trajectories must contain only finite values")
    if np.any(np.diff(values[:, 0]) < 0):
        raise ValueError("TUM timestamps must not decrease")
    quaternion_norms = np.linalg.norm(values[:, 4:8], axis=1)
    if np.any(quaternion_norms <= np.finfo(np.float64).eps):
        raise ValueError("TUM quaternions must have non-zero norm")
    return values


@dataclass(frozen=True)
class CanonicalTrajectory:
    """Unique-timestamp rows plus diagnostics for discarded boundary poses."""

    rows: np.ndarray
    duplicate_count: int
    max_duplicate_translation: float
    max_duplicate_rotation_deg: float

    def __post_init__(self) -> None:
        copied = np.array(self.rows, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "rows", copied)


def read_tum_rows(path: Path) -> np.ndarray:
    # TUM 每行固定为 timestamp tx ty tz qx qy qz qw。
    rows: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 8:
            raise ValueError(
                f"{path}:{line_number}: expected 8 columns, got {len(fields)}"
            )
        try:
            rows.append([float(field) for field in fields])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric TUM row") from exc
    if not rows:
        raise ValueError(f"no TUM poses found in {path}")
    return np.array(_validate_rows(np.asarray(rows, dtype=np.float64)), copy=True)


def canonicalize_tum(rows: np.ndarray) -> CanonicalTrajectory:
    values = _validate_rows(rows)
    keep_indices: list[int] = []
    duplicate_count = 0
    max_translation = 0.0
    max_rotation_deg = 0.0

    # 相邻子图共享边界帧，因此原始 VGGT-SLAM 日志可能重复同一时间戳。
    # 规范轨迹保留第一次出现，并记录被丢弃项与首项的最大分歧。
    first_index_by_timestamp: dict[float, int] = {}
    for index, row in enumerate(values):
        timestamp = float(row[0])
        first_index = first_index_by_timestamp.get(timestamp)
        if first_index is None:
            first_index_by_timestamp[timestamp] = index
            keep_indices.append(index)
            continue

        duplicate_count += 1
        first = values[first_index]
        translation = float(np.linalg.norm(row[1:4] - first[1:4]))
        max_translation = max(max_translation, translation)

        first_quaternion = first[4:8] / np.linalg.norm(first[4:8])
        duplicate_quaternion = row[4:8] / np.linalg.norm(row[4:8])
        # q 与 -q 表示同一旋转，取点积绝对值后再计算最小夹角。
        cosine = float(
            np.clip(abs(np.dot(first_quaternion, duplicate_quaternion)), -1.0, 1.0)
        )
        angle_deg = float(np.degrees(2.0 * np.arccos(cosine)))
        max_rotation_deg = max(max_rotation_deg, angle_deg)

    canonical_rows = np.array(values[keep_indices], copy=True)
    if canonical_rows.shape[0] < 2:
        raise ValueError("canonical trajectory requires at least two unique timestamps")
    return CanonicalTrajectory(
        rows=canonical_rows,
        duplicate_count=duplicate_count,
        max_duplicate_translation=max_translation,
        max_duplicate_rotation_deg=max_rotation_deg,
    )


def write_tum_rows(path: Path, rows: np.ndarray) -> None:
    values = _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(format(value, ".17g") for value in row) for row in values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
