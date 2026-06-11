from pathlib import Path

import numpy as np
import pytest

from vggt_slam_pp.contracts.runtime import RunIdentity
from vggt_slam_pp.contracts.submap import SubmapArrays, SubmapMetadata
from vggt_slam_pp.io.submap_cache import (
    CacheIntegrityError,
    read_submap_cache,
    write_submap_cache,
)


def _payload() -> tuple[SubmapMetadata, SubmapArrays]:
    run = RunIdentity(
        run_id="cache-test",
        solver_mode="baseline_sim3_compat",
        run_purpose="pp_frontend_bridge",
        max_loops=0,
        submap_size=2,
        min_disparity=50.0,
    )
    metadata = SubmapMetadata(
        run=run,
        submap_id=7,
        frame_ids=("10", "11"),
        last_non_loop_frame_index=1,
        scale_baked_into_geometry=True,
        loop_sources=(),
        baseline_sha256="a" * 64,
        weight_sha256="b" * 64,
    )
    arrays = SubmapArrays(
        points_submap=np.arange(2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3),
        colors_rgb=np.zeros((2, 2, 3, 3), dtype=np.uint8),
        confidence=np.ones((2, 2, 3), dtype=np.float32),
        confidence_mask=np.ones((2, 2, 3), dtype=bool),
        intrinsics=np.repeat(np.eye(3)[None], 2, axis=0),
        camera_to_submap=np.repeat(np.eye(4)[None], 2, axis=0),
    )
    return metadata, arrays


def test_submap_cache_round_trip(tmp_path: Path) -> None:
    metadata, arrays = _payload()
    cache_path = write_submap_cache(tmp_path, metadata, arrays)
    loaded_metadata, loaded_arrays = read_submap_cache(cache_path)

    assert loaded_metadata == metadata
    for field_name in arrays.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(loaded_arrays, field_name),
            getattr(arrays, field_name),
        )


def test_existing_submap_cannot_be_overwritten(tmp_path: Path) -> None:
    metadata, arrays = _payload()
    write_submap_cache(tmp_path, metadata, arrays)

    with pytest.raises(FileExistsError):
        write_submap_cache(tmp_path, metadata, arrays)


def test_failed_write_removes_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, arrays = _payload()

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(np, "savez_compressed", fail_save)
    with pytest.raises(RuntimeError, match="simulated"):
        write_submap_cache(tmp_path, metadata, arrays)

    assert not list(tmp_path.glob(".000007.tmp-*"))
    assert not (tmp_path / "000007").exists()


@pytest.mark.parametrize("filename", ["geometry.npz", "metadata.json"])
def test_modified_cache_file_fails_checksum(tmp_path: Path, filename: str) -> None:
    metadata, arrays = _payload()
    cache_path = write_submap_cache(tmp_path, metadata, arrays)
    target = cache_path / filename
    target.write_bytes(target.read_bytes() + b"corruption")

    with pytest.raises(CacheIntegrityError, match=filename):
        read_submap_cache(cache_path)
