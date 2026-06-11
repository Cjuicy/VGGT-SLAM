import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from vggt_slam_pp.evaluation.tum import write_tum_rows


def test_cli_reports_sim3_aligned_raw_and_canonical_ate(tmp_path: Path) -> None:
    timestamps = np.arange(6, dtype=np.float64)
    reference_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.5],
            [2.5, 2.0, 1.0],
            [3.0, 2.5, 1.5],
        ]
    )
    reference = np.column_stack(
        [
            timestamps,
            reference_positions,
            np.zeros((6, 3)),
            np.ones(6),
        ]
    )

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    estimate_positions = 2.5 * (rotation @ reference_positions.T).T + [4, -3, 2]
    estimate = np.column_stack(
        [
            timestamps,
            estimate_positions,
            np.tile([0.0, 0.0, np.sqrt(0.5)], (6, 1)),
            np.full(6, np.sqrt(0.5)),
        ]
    )

    reference_path = tmp_path / "groundtruth.txt"
    estimate_path = tmp_path / "estimate.txt"
    output_path = tmp_path / "ate.json"
    write_tum_rows(reference_path, reference)
    write_tum_rows(estimate_path, estimate)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vggt_slam_pp.cli.evaluate_ate",
            "--groundtruth",
            str(reference_path),
            "--estimate",
            str(estimate_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["paper_compatible"]["translation_ape_rmse"] < 1e-8
    assert report["canonical_unique"]["translation_ape_rmse"] < 1e-8
    assert report["paper_compatible"]["associated_pose_count"] == 6
    assert len(report["inputs"]["groundtruth_sha256"]) == 64
    assert len(report["inputs"]["estimate_sha256"]) == 64
