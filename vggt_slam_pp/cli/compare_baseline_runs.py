"""Compare two baseline run summaries and their TUM pose logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vggt_slam_pp.evaluation.compare_runs import compare_baseline_runs
from vggt_slam_pp.io.checksums import canonical_json_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    report = compare_baseline_runs(
        args.left.resolve(),
        args.right.resolve(),
        absolute_tolerance=args.absolute_tolerance,
    )
    encoded = canonical_json_bytes(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
