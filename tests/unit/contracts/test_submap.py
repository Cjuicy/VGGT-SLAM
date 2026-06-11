import numpy as np
import pytest
from pydantic import ValidationError

from vggt_slam_pp.contracts.runtime import RunIdentity
from vggt_slam_pp.contracts.submap import SubmapArrays, SubmapMetadata


def _bridge_run() -> RunIdentity:
    return RunIdentity(
        run_id="bridge",
        solver_mode="baseline_sim3_compat",
        run_purpose="pp_frontend_bridge",
        max_loops=0,
        submap_size=2,
        min_disparity=50.0,
    )


def _valid_arrays() -> dict[str, np.ndarray]:
    return {
        "points_submap": np.zeros((2, 3, 4, 3), dtype=np.float32),
        "colors_rgb": np.zeros((2, 3, 4, 3), dtype=np.uint8),
        "confidence": np.ones((2, 3, 4), dtype=np.float32),
        "confidence_mask": np.ones((2, 3, 4), dtype=bool),
        "intrinsics": np.repeat(np.eye(3)[None], 2, axis=0),
        "camera_to_submap": np.repeat(np.eye(4)[None], 2, axis=0),
    }


def test_arrays_are_defensive_read_only_copies() -> None:
    source = _valid_arrays()
    arrays = SubmapArrays(**source)
    source["points_submap"][0, 0, 0, 0] = 9.0

    assert arrays.points_submap[0, 0, 0, 0] == 0.0
    assert arrays.points_submap.flags.writeable is False
    with pytest.raises(ValueError):
        arrays.colors_rgb[0, 0, 0, 0] = 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("points_submap", np.zeros((2, 3, 4, 2))),
        ("colors_rgb", np.zeros((2, 3, 4, 3), dtype=np.float32)),
        ("confidence", np.zeros((2, 3, 5))),
        ("confidence_mask", np.zeros((2, 3, 4), dtype=np.uint8)),
        ("intrinsics", np.zeros((2, 4, 4))),
        ("camera_to_submap", np.zeros((2, 3, 3))),
    ],
)
def test_arrays_reject_invalid_shape_or_dtype(field: str, value: np.ndarray) -> None:
    payload = _valid_arrays()
    payload[field] = value

    with pytest.raises(ValueError):
        SubmapArrays(**payload)


def test_arrays_reject_non_finite_values() -> None:
    payload = _valid_arrays()
    payload["points_submap"][0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        SubmapArrays(**payload)


def test_bridge_metadata_rejects_loop_sources() -> None:
    with pytest.raises(ValidationError, match="loop_sources"):
        SubmapMetadata(
            run=_bridge_run(),
            submap_id=0,
            frame_ids=("0", "1"),
            last_non_loop_frame_index=1,
            scale_baked_into_geometry=True,
            loop_sources=(4,),
            baseline_sha256="a" * 64,
            weight_sha256="b" * 64,
        )


def test_bridge_metadata_records_relative_scale_state() -> None:
    metadata = SubmapMetadata(
        run=_bridge_run(),
        submap_id=0,
        frame_ids=("0", "1"),
        last_non_loop_frame_index=1,
        scale_baked_into_geometry=True,
        loop_sources=(),
        baseline_sha256="a" * 64,
        weight_sha256="b" * 64,
    )

    assert metadata.coordinate_frame == "vggt_submap"
    assert metadata.unit_state == "relative_map_unit"
    assert metadata.scale_baked_into_geometry is True
