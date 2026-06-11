import numpy as np
import pytest

from vggt_slam_pp.contracts.graph_state import GraphState
from vggt_slam_pp.contracts.runtime import RunIdentity


def _run(solver_mode: str) -> RunIdentity:
    return RunIdentity(
        run_id=solver_mode,
        solver_mode=solver_mode,
        run_purpose="baseline_reference",
        max_loops=1,
        submap_size=32,
        min_disparity=50.0,
    )


def test_graph_state_copies_and_freezes_transforms() -> None:
    transforms = np.repeat(np.eye(4)[None], 2, axis=0)
    state = GraphState(
        run=_run("baseline_sim3_compat"),
        update_index=3,
        world_from_submap=transforms,
        transform_kind="SE3_with_baked_scale",
        edge_count=2,
        loop_count=1,
        trajectory_sha256="c" * 64,
    )
    transforms[0, 0, 0] = 7.0

    assert state.world_from_submap[0, 0, 0] == 1.0
    assert state.world_from_submap.flags.writeable is False


def test_baked_scale_transform_requires_sim3_compat() -> None:
    with pytest.raises(ValueError, match="baseline_sim3_compat"):
        GraphState(
            run=_run("baseline_sl4"),
            update_index=0,
            world_from_submap=np.eye(4)[None],
            transform_kind="SE3_with_baked_scale",
            edge_count=0,
            loop_count=0,
            trajectory_sha256="d" * 64,
        )


def test_sl4_transform_requires_sl4_solver() -> None:
    with pytest.raises(ValueError, match="baseline_sl4"):
        GraphState(
            run=_run("baseline_sim3_compat"),
            update_index=0,
            world_from_submap=np.eye(4)[None],
            transform_kind="SL4",
            edge_count=0,
            loop_count=0,
            trajectory_sha256="d" * 64,
        )


def test_graph_state_rejects_bad_shape_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        GraphState(
            run=_run("baseline_sl4"),
            update_index=0,
            world_from_submap=np.eye(3)[None],
            transform_kind="SL4",
            edge_count=0,
            loop_count=0,
            trajectory_sha256="e" * 64,
        )

    transforms = np.eye(4)[None]
    transforms[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        GraphState(
            run=_run("baseline_sl4"),
            update_index=0,
            world_from_submap=transforms,
            transform_kind="SL4",
            edge_count=0,
            loop_count=0,
            trajectory_sha256="e" * 64,
        )
