from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.fixtures.submap_factory import make_baseline_submap
from vggt_slam_pp.adapters.export_session import ExportSession
from vggt_slam_pp.contracts.runtime import RunIdentity
from vggt_slam_pp.io.graph_state import read_final_graph_state
from vggt_slam_pp.io.submap_cache import read_submap_cache

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "VGGT-SLAM-version1.0" / "main.py"


class FakeMap:
    def __init__(self, submaps: list[object]) -> None:
        self._submaps = submaps

    def get_latest_submap(self) -> object:
        return self._submaps[-1]

    def ordered_submaps_by_key(self):
        return iter(self._submaps)


class FakeFactorGraph:
    def __init__(self, count: int) -> None:
        self._count = count

    def size(self) -> int:
        return self._count


def _session(tmp_path: Path) -> ExportSession:
    run = RunIdentity(
        run_id="bridge-run",
        solver_mode="baseline_sim3_compat",
        run_purpose="pp_frontend_bridge",
        max_loops=0,
        submap_size=2,
        min_disparity=50.0,
    )
    return ExportSession(
        output_root=tmp_path,
        run=run,
        baseline_sha256="a" * 64,
        weight_sha256="b" * 64,
    )


def test_main_keeps_export_opt_in_and_after_graph_update() -> None:
    source = MAIN.read_text(encoding="utf-8")
    assert "if args.export_submaps_dir is not None" in source
    assert source.index("solver.add_points(predictions)") < source.index(
        "export_session.export_latest_submap"
    )
    assert source.index("solver.graph.optimize()") < source.index(
        "export_session.export_latest_submap"
    )
    assert source.index("solver.map.update_submap_homographies") < source.index(
        "export_session.export_latest_submap"
    )


def test_session_writes_geometry_once_and_all_graph_states(tmp_path: Path) -> None:
    submap = make_baseline_submap()
    original_points = submap.pointclouds.copy()
    graph_map = FakeMap([submap])
    graph = SimpleNamespace(
        graph=FakeFactorGraph(1),
        get_num_loops=lambda: 0,
    )
    session = _session(tmp_path)

    geometry_path = session.export_latest_submap(submap, loop_sources=())
    state_path = session.export_graph_state(graph_map, graph)
    final_path = session.finalize(graph_map, graph)

    metadata, arrays = read_submap_cache(geometry_path)
    final_state = read_final_graph_state(session.run_root)
    assert metadata.submap_id == 4
    np.testing.assert_array_equal(arrays.points_submap, original_points)
    assert state_path.name == "000000.json"
    assert final_state.world_from_submap.shape == (1, 4, 4)
    assert final_path == session.run_root / "final_state.json"
    np.testing.assert_array_equal(submap.pointclouds, original_points)

    # Geometry is immutable and cannot silently be exported twice.
    try:
        session.export_latest_submap(submap, loop_sources=())
    except FileExistsError:
        pass
    else:
        raise AssertionError("duplicate submap export must fail")
