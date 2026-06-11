"""Strict comparison of export-disabled and export-enabled baseline runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vggt_slam_pp.evaluation.tum import read_tum_rows
from vggt_slam_pp.io.checksums import sha256_file


def _load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    required = {"submap_count", "loop_count", "pose_log"}
    missing = required - summary.keys()
    if missing:
        raise ValueError(f"run summary missing fields: {sorted(missing)}")
    return summary


def _resolve_pose_path(summary_path: Path, value: str) -> Path:
    pose_path = Path(value)
    return pose_path if pose_path.is_absolute() else summary_path.parent / pose_path


def compare_baseline_runs(
    left_summary_path: Path,
    right_summary_path: Path,
    *,
    absolute_tolerance: float = 1e-8,
) -> dict[str, Any]:
    left = _load_summary(left_summary_path)
    right = _load_summary(right_summary_path)
    if left["submap_count"] != right["submap_count"]:
        raise ValueError("submap count differs")
    if left["loop_count"] != right["loop_count"]:
        raise ValueError("loop count differs")

    left_pose_path = _resolve_pose_path(left_summary_path, left["pose_log"])
    right_pose_path = _resolve_pose_path(right_summary_path, right["pose_log"])
    left_rows = read_tum_rows(left_pose_path)
    right_rows = read_tum_rows(right_pose_path)
    if left_rows.shape != right_rows.shape:
        raise ValueError("pose count differs")
    if not np.array_equal(left_rows[:, 0], right_rows[:, 0]):
        raise ValueError("timestamp sequence differs")
    if not np.allclose(
        left_rows[:, 1:],
        right_rows[:, 1:],
        rtol=0.0,
        atol=absolute_tolerance,
    ):
        max_difference = float(np.max(np.abs(left_rows[:, 1:] - right_rows[:, 1:])))
        raise ValueError(f"pose values differ; max absolute difference {max_difference}")

    return {
        "schema_version": 1,
        "equal": True,
        "absolute_tolerance": absolute_tolerance,
        "submap_count": int(left["submap_count"]),
        "loop_count": int(left["loop_count"]),
        "pose_count": int(left_rows.shape[0]),
        "left_pose_sha256": sha256_file(left_pose_path),
        "right_pose_sha256": sha256_file(right_pose_path),
    }
