import json
import subprocess
import sys
from pathlib import Path

from tests.fixtures.submap_factory import make_baseline_submap
from vggt_slam_pp.adapters.export_session import ExportSession
from vggt_slam_pp.contracts.runtime import RunIdentity


def test_inspector_reports_verified_shapes_frames_and_transforms(tmp_path: Path) -> None:
    run = RunIdentity(
        run_id="inspect",
        solver_mode="baseline_sim3_compat",
        run_purpose="pp_frontend_bridge",
        max_loops=0,
        submap_size=2,
        min_disparity=50.0,
    )
    session = ExportSession(
        output_root=tmp_path,
        run=run,
        baseline_sha256="a" * 64,
        weight_sha256="b" * 64,
    )
    submap = make_baseline_submap()
    session.export_latest_submap(submap, ())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vggt_slam_pp.cli.inspect_submap_cache",
            str(session.run_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["submaps"][0]["frame_ids"] == ["100.0", "101.0"]
    assert report["submaps"][0]["arrays"]["points_submap"]["shape"] == [2, 2, 3, 3]
    assert report["submaps"][0]["arrays"]["colors_rgb"]["dtype"] == "uint8"
