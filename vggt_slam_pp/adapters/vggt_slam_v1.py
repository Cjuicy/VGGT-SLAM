"""Read-only adapter for the public fields of VGGT-SLAM v1 `Submap`."""

from __future__ import annotations

from typing import Any

import numpy as np

from vggt_slam_pp.contracts.runtime import RunIdentity
from vggt_slam_pp.contracts.submap import SubmapArrays, SubmapMetadata

_REQUIRED_FIELDS = (
    "submap_id",
    "pointclouds",
    "colors",
    "conf",
    "conf_threshold",
    "vggt_intrinscs",
    "poses",
    "frame_ids",
    "last_non_loop_frame_index",
)


def _require_finalized_fields(submap: object) -> None:
    missing = [
        name
        for name in _REQUIRED_FIELDS
        if not hasattr(submap, name) or getattr(submap, name) is None
    ]
    if missing:
        raise ValueError(
            "VGGT-SLAM submap is not finalized; missing fields: "
            + ", ".join(missing)
        )


def _homogeneous_camera_poses(poses: Any) -> np.ndarray:
    poses_array = np.asarray(poses)
    if poses_array.ndim != 3:
        raise ValueError("submap.poses must have shape (S,3,4) or (S,4,4)")
    if poses_array.shape[1:] == (4, 4):
        return np.array(poses_array, copy=True)
    if poses_array.shape[1:] != (3, 4):
        raise ValueError("submap.poses must have shape (S,3,4) or (S,4,4)")

    homogeneous = np.repeat(
        np.eye(4, dtype=poses_array.dtype)[None],
        poses_array.shape[0],
        axis=0,
    )
    homogeneous[:, :3, :] = poses_array
    return homogeneous


def adapt_submap(
    submap: object,
    *,
    run: RunIdentity,
    baseline_sha256: str,
    weight_sha256: str,
    loop_sources: tuple[int, ...],
) -> tuple[SubmapMetadata, SubmapArrays]:
    """Copy a post-`add_points()` baseline submap into M0 cache contracts."""
    _require_finalized_fields(submap)

    points = np.asarray(getattr(submap, "pointclouds"))
    confidence = np.asarray(getattr(submap, "conf"))
    confidence_threshold = float(getattr(submap, "conf_threshold"))
    camera_to_submap = _homogeneous_camera_poses(getattr(submap, "poses"))
    frame_ids = tuple(str(value) for value in getattr(submap, "frame_ids"))

    if points.shape[0] != len(frame_ids):
        raise ValueError(
            "submap.frame_ids must cover every exported frame; "
            "disable baseline loop frames for pp_frontend_bridge"
        )

    arrays = SubmapArrays(
        # These are finalized fields. Do not reconstruct from raw VGGT predictions:
        # baseline_sim3_compat has already multiplied scale into both arrays.
        points_submap=points,
        colors_rgb=np.asarray(getattr(submap, "colors")),
        confidence=confidence,
        confidence_mask=confidence >= confidence_threshold,
        intrinsics=np.asarray(getattr(submap, "vggt_intrinscs")),
        camera_to_submap=camera_to_submap,
    )
    metadata = SubmapMetadata(
        run=run,
        submap_id=int(getattr(submap, "submap_id")),
        frame_ids=frame_ids,
        last_non_loop_frame_index=int(
            getattr(submap, "last_non_loop_frame_index")
        ),
        scale_baked_into_geometry=run.solver_mode == "baseline_sim3_compat",
        loop_sources=loop_sources,
        baseline_sha256=baseline_sha256,
        weight_sha256=weight_sha256,
    )
    return metadata, arrays
