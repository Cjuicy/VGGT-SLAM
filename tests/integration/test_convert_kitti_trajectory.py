import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from vggt_slam_pp.evaluation.tum import read_tum_rows
from vggt_slam_pp.io.checksums import sha256_file


def run_conversion(input_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "vggt_slam_pp.cli.convert_kitti_trajectory",
            "--input",
            os.path.relpath(input_path, Path.cwd()),
            "--output",
            os.path.relpath(output_path, Path.cwd()),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_converts_kitti_trajectory_and_reports_checksums(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "poses.txt"
    output_path = tmp_path / "nested" / "trajectory.txt"
    input_path.write_text(
        "\n".join(
            [
                "1 0 0 0 0 1 0 0 0 0 1 0",
                "0 -1 0 1 1 0 0 2 0 0 1 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_conversion(input_path, output_path)

    assert result.returncode == 0, result.stderr
    rows = read_tum_rows(output_path)
    np.testing.assert_allclose(rows[:, 0], [0.0, 1.0])
    np.testing.assert_allclose(rows[:, 1:4], [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

    report = json.loads(result.stdout)
    assert set(report) == {
        "schema_version",
        "input",
        "output",
        "pose_count",
        "input_sha256",
        "output_sha256",
    }
    assert report == {
        "schema_version": 1,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "pose_count": 2,
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
    }


def test_cli_rejects_identical_input_and_output_without_modifying_source(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "poses.txt"
    original = (
        "1 0 0 0 0 1 0 0 0 0 1 0\n"
        "0 -1 0 1 1 0 0 2 0 0 1 3\n"
    ).encode()
    input_path.write_bytes(original)
    original_sha256 = sha256_file(input_path)

    result = run_conversion(input_path, input_path)

    assert result.returncode != 0
    assert "input and output must refer to different files" in result.stderr
    assert input_path.read_bytes() == original
    assert sha256_file(input_path) == original_sha256


def test_cli_rejects_hard_link_output_without_modifying_source(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "poses.txt"
    output_path = tmp_path / "trajectory.txt"
    original = (
        "1 0 0 0 0 1 0 0 0 0 1 0\n"
        "0 -1 0 1 1 0 0 2 0 0 1 3\n"
    ).encode()
    input_path.write_bytes(original)
    os.link(input_path, output_path)
    original_sha256 = sha256_file(input_path)

    result = run_conversion(input_path, output_path)

    assert result.returncode != 0
    assert "input and output must refer to different files" in result.stderr
    assert input_path.read_bytes() == original
    assert output_path.read_bytes() == original
    assert sha256_file(input_path) == original_sha256
