"""Convert a KITTI trajectory to TUM format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vggt_slam_pp.evaluation.kitti import read_kitti_pose_rows
from vggt_slam_pp.evaluation.tum import write_tum_rows
from vggt_slam_pp.io.checksums import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path or (
        output_path.exists() and input_path.samefile(output_path)
    ):
        parser.error("input and output must refer to different files")

    input_sha256 = sha256_file(input_path)
    rows = read_kitti_pose_rows(input_path)
    write_tum_rows(output_path, rows)

    report = {
        "schema_version": 1,
        "input": str(input_path),
        "output": str(output_path),
        "pose_count": len(rows),
        "input_sha256": input_sha256,
        "output_sha256": sha256_file(output_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
