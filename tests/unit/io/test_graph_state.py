import json
from pathlib import Path

import numpy as np
import pytest

from vggt_slam_pp.contracts.graph_state import GraphState
from vggt_slam_pp.contracts.runtime import RunIdentity
from vggt_slam_pp.io.graph_state import (
    GraphStateIntegrityError,
    read_final_graph_state,
    write_graph_state,
)


def _state(index: int) -> GraphState:
    run = RunIdentity(
        run_id="graph-test",
        solver_mode="baseline_sim3_compat",
        run_purpose="pp_frontend_bridge",
        max_loops=0,
        submap_size=32,
        min_disparity=50.0,
    )
    return GraphState(
        run=run,
        update_index=index,
        world_from_submap=np.repeat(np.eye(4)[None], index + 1, axis=0),
        transform_kind="SE3_with_baked_scale",
        edge_count=index,
        loop_count=0,
        trajectory_sha256=f"{index:x}" * 64,
    )


def test_graph_states_append_and_final_points_to_latest(tmp_path: Path) -> None:
    write_graph_state(tmp_path, _state(0))
    write_graph_state(tmp_path, _state(1))

    loaded = read_final_graph_state(tmp_path)
    assert loaded.update_index == 1
    assert (tmp_path / "states" / "000000.json").is_file()
    assert (tmp_path / "states" / "000001.json").is_file()


def test_graph_state_indices_must_be_monotonic(tmp_path: Path) -> None:
    write_graph_state(tmp_path, _state(0))

    with pytest.raises(ValueError, match="next update_index"):
        write_graph_state(tmp_path, _state(2))


def test_modified_graph_state_fails_checksum(tmp_path: Path) -> None:
    state_path = write_graph_state(tmp_path, _state(0))
    envelope = json.loads(state_path.read_text(encoding="utf-8"))
    envelope["state"]["edge_count"] = 99
    state_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(GraphStateIntegrityError, match="checksum"):
        read_final_graph_state(tmp_path)
