"""Immutable metadata and dense arrays exported from one VGGT submap."""

from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vggt_slam_pp.contracts.runtime import RunIdentity


def _read_only_copy(array: np.ndarray, name: str) -> np.ndarray:
    copied = np.array(array, copy=True)
    if not np.issubdtype(copied.dtype, np.number) and copied.dtype != np.bool_:
        raise ValueError(f"{name} must be numeric or boolean")
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{name} must contain only finite values")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class SubmapArrays:
    """Dense submap payload in the finalized, scale-adjusted local frame."""

    points_submap: np.ndarray
    colors_rgb: np.ndarray
    confidence: np.ndarray
    confidence_mask: np.ndarray
    intrinsics: np.ndarray
    camera_to_submap: np.ndarray

    def __post_init__(self) -> None:
        for field in fields(self):
            value = _read_only_copy(getattr(self, field.name), field.name)
            object.__setattr__(self, field.name, value)

        points_shape = self.points_submap.shape
        if len(points_shape) != 4 or points_shape[-1] != 3:
            raise ValueError("points_submap must have shape (S,H,W,3)")
        spatial_shape = points_shape[:3]
        if self.colors_rgb.shape != points_shape:
            raise ValueError("colors_rgb must match points_submap shape")
        if self.colors_rgb.dtype != np.uint8:
            raise ValueError("colors_rgb must use uint8")
        if self.confidence.shape != spatial_shape:
            raise ValueError("confidence must have shape (S,H,W)")
        if self.confidence_mask.shape != spatial_shape:
            raise ValueError("confidence_mask must have shape (S,H,W)")
        if self.confidence_mask.dtype != np.bool_:
            raise ValueError("confidence_mask must use bool")

        submap_frames = points_shape[0]
        if self.intrinsics.shape != (submap_frames, 3, 3):
            raise ValueError("intrinsics must have shape (S,3,3)")
        if self.camera_to_submap.shape != (submap_frames, 4, 4):
            raise ValueError("camera_to_submap must have shape (S,4,4)")


class SubmapMetadata(BaseModel):
    """Small JSON-safe description of an immutable submap payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    run: RunIdentity
    submap_id: int = Field(ge=0)
    frame_ids: tuple[str, ...] = Field(min_length=1)
    last_non_loop_frame_index: int = Field(ge=0)
    coordinate_frame: Literal["vggt_submap"] = "vggt_submap"
    unit_state: Literal["relative_map_unit"] = "relative_map_unit"
    scale_baked_into_geometry: bool
    loop_sources: tuple[int, ...] = ()
    baseline_sha256: str
    weight_sha256: str

    @field_validator("baseline_sha256", "weight_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("must be a 64-character SHA-256")
        return normalized

    @model_validator(mode="after")
    def validate_submap_semantics(self) -> "SubmapMetadata":
        if self.last_non_loop_frame_index >= len(self.frame_ids):
            raise ValueError("last_non_loop_frame_index must index frame_ids")
        if self.run.run_purpose == "pp_frontend_bridge" and self.loop_sources:
            raise ValueError("pp_frontend_bridge requires empty loop_sources")

        expected_baked_scale = self.run.solver_mode == "baseline_sim3_compat"
        if self.scale_baked_into_geometry != expected_baked_scale:
            raise ValueError(
                "scale_baked_into_geometry must match the baseline solver mode"
            )
        return self
