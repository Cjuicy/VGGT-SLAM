from pathlib import Path

import numpy as np
import pytest

from vggt_slam_pp.evaluation.tum import (
    canonicalize_tum,
    read_tum_rows,
    write_tum_rows,
)


def test_canonicalization_keeps_first_duplicate_and_reports_disagreement() -> None:
    rows = np.array(
        [
            [0.0, 0, 0, 0, 0, 0, 0, 1],
            [1.0, 1, 0, 0, 0, 0, 0, 1],
            [1.0, 1.1, 0, 0, 0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)],
            [2.0, 2, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    original = rows.copy()
    canonical = canonicalize_tum(rows)

    np.testing.assert_array_equal(rows, original)
    np.testing.assert_array_equal(canonical.rows, rows[[0, 1, 3]])
    assert canonical.duplicate_count == 1
    assert canonical.max_duplicate_translation == pytest.approx(0.1)
    assert canonical.max_duplicate_rotation_deg == pytest.approx(45.0)


def test_tum_round_trip(tmp_path: Path) -> None:
    rows = np.array(
        [
            [0.0, 0, 0, 0, 0, 0, 0, 1],
            [1.0, 1, 2, 3, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "trajectory.txt"
    write_tum_rows(path, rows)

    np.testing.assert_allclose(read_tum_rows(path), rows)


@pytest.mark.parametrize(
    "rows",
    [
        np.array([[0.0, 0, 0, 0, 0, 0, 0, 1]]),
        np.array(
            [
                [1.0, 0, 0, 0, 0, 0, 0, 1],
                [0.0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
        np.array(
            [
                [0.0, 0, 0, np.nan, 0, 0, 0, 1],
                [1.0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
        np.zeros((2, 7)),
    ],
)
def test_invalid_trajectories_are_rejected(rows: np.ndarray) -> None:
    with pytest.raises(ValueError):
        canonicalize_tum(rows)


def test_malformed_text_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("0 0 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="8 columns"):
        read_tum_rows(path)
