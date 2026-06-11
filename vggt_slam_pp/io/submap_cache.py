"""Atomic storage for immutable submap metadata and dense geometry."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np

from vggt_slam_pp.contracts.submap import SubmapArrays, SubmapMetadata
from vggt_slam_pp.io.checksums import (
    canonical_json_bytes,
    fsync_directory,
    sha256_file,
    write_fsynced,
)


class CacheIntegrityError(RuntimeError):
    """Raised when a cache file no longer matches its committed checksum."""


def write_submap_cache(
    submaps_root: Path,
    metadata: SubmapMetadata,
    arrays: SubmapArrays,
) -> Path:
    """Commit one immutable submap directory with an atomic rename."""
    submaps_root.mkdir(parents=True, exist_ok=True)
    final_path = submaps_root / f"{metadata.submap_id:06d}"
    if final_path.exists():
        raise FileExistsError(f"immutable submap already exists: {final_path}")

    temporary_path = submaps_root / (
        f".{metadata.submap_id:06d}.tmp-{uuid.uuid4().hex}"
    )
    temporary_path.mkdir()
    try:
        metadata_path = temporary_path / "metadata.json"
        geometry_path = temporary_path / "geometry.npz"
        checksums_path = temporary_path / "checksums.json"

        write_fsynced(
            metadata_path,
            canonical_json_bytes(metadata.model_dump(mode="json")),
        )
        np.savez_compressed(
            geometry_path,
            points_submap=arrays.points_submap,
            colors_rgb=arrays.colors_rgb,
            confidence=arrays.confidence,
            confidence_mask=arrays.confidence_mask,
            intrinsics=arrays.intrinsics,
            camera_to_submap=arrays.camera_to_submap,
        )
        with geometry_path.open("rb") as stream:
            os.fsync(stream.fileno())

        checksums = {
            "schema_version": 1,
            "files": {
                "geometry.npz": sha256_file(geometry_path),
                "metadata.json": sha256_file(metadata_path),
            },
        }
        write_fsynced(checksums_path, canonical_json_bytes(checksums))
        fsync_directory(temporary_path)
        temporary_path.rename(final_path)
        fsync_directory(submaps_root)
        return final_path
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


def read_submap_cache(cache_path: Path) -> tuple[SubmapMetadata, SubmapArrays]:
    """Validate checksums before reconstructing immutable contract objects."""
    checksums_path = cache_path / "checksums.json"
    try:
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
        declared_files = checksums["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CacheIntegrityError(f"invalid checksums.json in {cache_path}") from exc

    for filename in ("metadata.json", "geometry.npz"):
        path = cache_path / filename
        if not path.is_file():
            raise CacheIntegrityError(f"missing committed cache file: {filename}")
        actual = sha256_file(path)
        expected = declared_files.get(filename)
        if actual != expected:
            raise CacheIntegrityError(f"checksum mismatch: {filename}")

    metadata = SubmapMetadata.model_validate_json(
        (cache_path / "metadata.json").read_text(encoding="utf-8")
    )
    with np.load(cache_path / "geometry.npz", allow_pickle=False) as archive:
        arrays = SubmapArrays(
            points_submap=archive["points_submap"],
            colors_rgb=archive["colors_rgb"],
            confidence=archive["confidence"],
            confidence_mask=archive["confidence_mask"],
            intrinsics=archive["intrinsics"],
            camera_to_submap=archive["camera_to_submap"],
        )
    return metadata, arrays
