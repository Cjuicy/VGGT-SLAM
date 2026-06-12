from pathlib import Path

import numpy as np
import pytest

from vggt_slam_pp.evaluation.kitti import read_kitti_pose_rows


def test_reads_identity_and_rotated_translated_poses(tmp_path: Path) -> None:
    path = tmp_path / "poses.txt"
    path.write_text(
        "\n".join(
            [
                "# frame poses",
                "1 0 0 0 0 1 0 0 0 0 1 0",
                "0 -1 0 1 1 0 0 2 0 0 1 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = read_kitti_pose_rows(path)

    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)],
        ],
        dtype=np.float64,
    )
    assert rows.dtype == np.float64
    np.testing.assert_allclose(rows, expected, atol=1e-12)


def test_rejects_malformed_column_count(tmp_path: Path) -> None:
    path = tmp_path / "poses.txt"
    path.write_text(
        "1 0 0 0 0 1 0 0 0 0 1\n"
        "1 0 0 0 0 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="12 columns"):
        read_kitti_pose_rows(path)


def test_rejects_non_finite_value(tmp_path: Path) -> None:
    path = tmp_path / "poses.txt"
    path.write_text(
        "1 0 0 nan 0 1 0 0 0 0 1 0\n"
        "1 0 0 0 0 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        read_kitti_pose_rows(path)


def test_rejects_invalid_rotation(tmp_path: Path) -> None:
    path = tmp_path / "poses.txt"
    path.write_text(
        "2 0 0 0 0 1 0 0 0 0 1 0\n"
        "1 0 0 0 0 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rotation"):
        read_kitti_pose_rows(path)
