"""Evaluate raw and canonical TUM trajectories with Sim(3)-aligned ATE."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from vggt_slam_pp.evaluation.ate import evaluate_translation_ape
from vggt_slam_pp.evaluation.tum import (
    canonicalize_tum,
    read_tum_rows,
    write_tum_rows,
)
from vggt_slam_pp.io.checksums import canonical_json_bytes, sha256_file


def build_report(
    groundtruth_path: Path,
    estimate_path: Path,
    *,
    max_timestamp_difference: float = 0.01,
) -> dict[str, object]:
    groundtruth_rows = read_tum_rows(groundtruth_path)
    estimate_rows = read_tum_rows(estimate_path)
    canonical_groundtruth = canonicalize_tum(groundtruth_rows)
    canonical_estimate = canonicalize_tum(estimate_rows)

    # paper_compatible 保留原始边界重复帧，便于复现既有基线口径。
    raw_result = evaluate_translation_ape(
        groundtruth_path,
        estimate_path,
        max_timestamp_difference=max_timestamp_difference,
    )
    # canonical_unique 只用于第二次评测；临时去重轨迹不会写入正式产物。
    with tempfile.TemporaryDirectory(prefix="vggt-slam-pp-ate-") as temporary:
        temporary_root = Path(temporary)
        canonical_groundtruth_path = temporary_root / "groundtruth.txt"
        canonical_estimate_path = temporary_root / "estimate.txt"
        write_tum_rows(canonical_groundtruth_path, canonical_groundtruth.rows)
        write_tum_rows(canonical_estimate_path, canonical_estimate.rows)
        canonical_result = evaluate_translation_ape(
            canonical_groundtruth_path,
            canonical_estimate_path,
            max_timestamp_difference=max_timestamp_difference,
        )

    duplicate_diagnostics = {
        "groundtruth": {
            "duplicate_count": canonical_groundtruth.duplicate_count,
            "max_translation": canonical_groundtruth.max_duplicate_translation,
            "max_rotation_deg": canonical_groundtruth.max_duplicate_rotation_deg,
        },
        "estimate": {
            "duplicate_count": canonical_estimate.duplicate_count,
            "max_translation": canonical_estimate.max_duplicate_translation,
            "max_rotation_deg": canonical_estimate.max_duplicate_rotation_deg,
        },
    }
    return {
        "schema_version": 1,
        "alignment": "Sim3",
        "metric": "translation_APE_RMSE",
        "max_timestamp_difference": max_timestamp_difference,
        "inputs": {
            "groundtruth": str(groundtruth_path),
            "estimate": str(estimate_path),
            "groundtruth_sha256": sha256_file(groundtruth_path),
            "estimate_sha256": sha256_file(estimate_path),
        },
        "duplicate_diagnostics": duplicate_diagnostics,
        "paper_compatible": raw_result,
        "canonical_unique": canonical_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-timestamp-difference", type=float, default=0.01)
    args = parser.parse_args()

    report = build_report(
        args.groundtruth.resolve(),
        args.estimate.resolve(),
        max_timestamp_difference=args.max_timestamp_difference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
