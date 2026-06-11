"""Versioned global transforms kept separate from immutable submap geometry."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from vggt_slam_pp.contracts.runtime import RunIdentity

TransformKind = Literal["SL4", "SE3_with_baked_scale"]


@dataclass(frozen=True)
class GraphState:
    """One complete snapshot of mutable global graph state."""

    run: RunIdentity
    update_index: int
    world_from_submap: np.ndarray
    transform_kind: TransformKind
    edge_count: int
    loop_count: int
    trajectory_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.update_index < 0 or self.edge_count < 0 or self.loop_count < 0:
            raise ValueError("graph indices and counts must be non-negative")
        if self.loop_count > self.edge_count:
            raise ValueError("loop_count cannot exceed edge_count")

        transforms = np.array(self.world_from_submap, copy=True)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError("world_from_submap must have shape (N,4,4)")
        if not np.all(np.isfinite(transforms)):
            raise ValueError("world_from_submap must contain only finite values")
        transforms.setflags(write=False)
        object.__setattr__(self, "world_from_submap", transforms)

        expected_solver = {
            "SL4": "baseline_sl4",
            "SE3_with_baked_scale": "baseline_sim3_compat",
        }[self.transform_kind]
        if self.run.solver_mode != expected_solver:
            raise ValueError(
                f"{self.transform_kind} requires solver mode {expected_solver}"
            )

        checksum = self.trajectory_sha256.lower()
        if len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum
        ):
            raise ValueError("trajectory_sha256 must be a 64-character SHA-256")
        object.__setattr__(self, "trajectory_sha256", checksum)
