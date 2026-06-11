import numpy as np
import pytest

from tests.fixtures.submap_factory import make_baseline_submap
from vggt_slam_pp.adapters.vggt_slam_v1 import adapt_submap
from vggt_slam_pp.contracts.runtime import RunIdentity


def _bridge_run() -> RunIdentity:
    return RunIdentity(
        run_id="adapter-test",
        solver_mode="baseline_sim3_compat",
        run_purpose="pp_frontend_bridge",
        max_loops=0,
        submap_size=2,
        min_disparity=50.0,
    )


def test_adapter_uses_finalized_scale_baked_submap_fields() -> None:
    source = make_baseline_submap()
    metadata, arrays = adapt_submap(
        source,
        run=_bridge_run(),
        baseline_sha256="a" * 64,
        weight_sha256="b" * 64,
        loop_sources=(),
    )

    np.testing.assert_array_equal(arrays.points_submap, source.pointclouds)
    assert not np.any(arrays.points_submap == -999.0)
    assert metadata.scale_baked_into_geometry is True


def test_adapter_converts_3x4_camera_poses_to_homogeneous() -> None:
    source = make_baseline_submap(poses_as_3x4=True)
    _, arrays = adapt_submap(
        source,
        run=_bridge_run(),
        baseline_sha256="a" * 64,
        weight_sha256="b" * 64,
        loop_sources=(),
    )

    assert arrays.camera_to_submap.shape == (2, 4, 4)
    np.testing.assert_array_equal(arrays.camera_to_submap[:, 3, :], [[0, 0, 0, 1]] * 2)


def test_adapter_uses_baseline_confidence_threshold() -> None:
    source = make_baseline_submap()
    _, arrays = adapt_submap(
        source,
        run=_bridge_run(),
        baseline_sha256="a" * 64,
        weight_sha256="b" * 64,
        loop_sources=(),
    )

    np.testing.assert_array_equal(
        arrays.confidence_mask,
        source.conf >= source.conf_threshold,
    )


def test_adapter_returns_defensive_copies() -> None:
    source = make_baseline_submap()
    _, arrays = adapt_submap(
        source,
        run=_bridge_run(),
        baseline_sha256="a" * 64,
        weight_sha256="b" * 64,
        loop_sources=(),
    )
    source.pointclouds[0, 0, 0, 0] = 12345.0

    assert arrays.points_submap[0, 0, 0, 0] != 12345.0
    with pytest.raises(ValueError):
        arrays.points_submap[0, 0, 0, 0] = 1.0


def test_bridge_rejects_loop_sources() -> None:
    with pytest.raises(ValueError, match="loop_sources"):
        adapt_submap(
            make_baseline_submap(),
            run=_bridge_run(),
            baseline_sha256="a" * 64,
            weight_sha256="b" * 64,
            loop_sources=(9,),
        )


def test_baseline_reference_labels_appended_loop_frames() -> None:
    source = make_baseline_submap()
    source.pointclouds = np.concatenate(
        [source.pointclouds, source.pointclouds[:1]],
        axis=0,
    )
    source.colors = np.concatenate([source.colors, source.colors[:1]], axis=0)
    source.conf = np.concatenate([source.conf, source.conf[:1]], axis=0)
    source.vggt_intrinscs = np.concatenate(
        [source.vggt_intrinscs, source.vggt_intrinscs[:1]],
        axis=0,
    )
    source.poses = np.concatenate([source.poses, source.poses[:1]], axis=0)
    run = RunIdentity(
        run_id="baseline-loop",
        solver_mode="baseline_sim3_compat",
        run_purpose="baseline_reference",
        max_loops=1,
        submap_size=2,
        min_disparity=50.0,
    )

    metadata, _ = adapt_submap(
        source,
        run=run,
        baseline_sha256="a" * 64,
        weight_sha256="b" * 64,
        loop_sources=(3,),
    )

    assert metadata.frame_ids[-1] == "loop_source:3:0"
