"""Verify and summarize one M0 run cache as deterministic JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vggt_slam_pp.io.graph_state import read_final_graph_state
from vggt_slam_pp.io.submap_cache import read_submap_cache


def inspect_run_cache(run_root: Path) -> dict[str, Any]:
    submaps_root = run_root / "submaps"
    submap_reports = []
    for cache_path in sorted(path for path in submaps_root.iterdir() if path.is_dir()):
        metadata, arrays = read_submap_cache(cache_path)
        array_report = {}
        for field_name in arrays.__dataclass_fields__:
            value = getattr(arrays, field_name)
            array_report[field_name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": True,
                "read_only": not value.flags.writeable,
            }
        submap_reports.append(
            {
                "submap_id": metadata.submap_id,
                "frame_ids": list(metadata.frame_ids),
                "last_non_loop_frame_index": metadata.last_non_loop_frame_index,
                "coordinate_frame": metadata.coordinate_frame,
                "unit_state": metadata.unit_state,
                "scale_baked_into_geometry": metadata.scale_baked_into_geometry,
                "loop_sources": list(metadata.loop_sources),
                "arrays": array_report,
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "ok": True,
        "run_root": str(run_root),
        "submap_count": len(submap_reports),
        "submaps": submap_reports,
    }
    if (run_root / "final_state.json").is_file():
        state = read_final_graph_state(run_root)
        report["final_state"] = {
            "update_index": state.update_index,
            "transform_kind": state.transform_kind,
            "world_from_submap_shape": list(state.world_from_submap.shape),
            "edge_count": state.edge_count,
            "loop_count": state.loop_count,
            "trajectory_sha256": state.trajectory_sha256,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    report = inspect_run_cache(args.run_root.resolve())
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
