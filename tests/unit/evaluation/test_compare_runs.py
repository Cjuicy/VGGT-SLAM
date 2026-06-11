import json
from pathlib import Path

import numpy as np
import pytest

from vggt_slam_pp.evaluation.compare_runs import compare_baseline_runs
from vggt_slam_pp.evaluation.tum import write_tum_rows


def _write_run(root: Path, name: str, poses: np.ndarray, **updates: int) -> Path:
    pose_path = root / f"{name}.txt"
    write_tum_rows(pose_path, poses)
    summary = {
        "submap_count": updates.get("submap_count", 2),
        "loop_count": updates.get("loop_count", 1),
        "pose_log": str(pose_path),
    }
    summary_path = root / f"{name}.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def _poses() -> np.ndarray:
    return np.array(
        [
            [0.0, 0, 0, 0, 0, 0, 0, 1],
            [1.0, 1, 0, 0, 0, 0, 0, 1],
            [2.0, 1, 1, 1, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def test_identical_baseline_runs_compare_equal(tmp_path: Path) -> None:
    left = _write_run(tmp_path, "left", _poses())
    right = _write_run(tmp_path, "right", _poses())

    report = compare_baseline_runs(left, right)
    assert report["equal"] is True
    assert report["pose_count"] == 3


@pytest.mark.parametrize("difference", ["submap", "loop", "timestamp", "pose"])
def test_comparison_rejects_behavioral_difference(
    tmp_path: Path, difference: str
) -> None:
    left = _write_run(tmp_path, "left", _poses())
    poses = _poses()
    updates: dict[str, int] = {}
    if difference == "submap":
        updates["submap_count"] = 3
    elif difference == "loop":
        updates["loop_count"] = 0
    elif difference == "timestamp":
        poses[1, 0] += 0.1
    else:
        poses[1, 1] += 0.1
    right = _write_run(tmp_path, "right", poses, **updates)

    with pytest.raises(ValueError, match=difference):
        compare_baseline_runs(left, right)
