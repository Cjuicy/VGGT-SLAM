"""Append-only global graph snapshots with an atomic final-state pointer."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from vggt_slam_pp.contracts.graph_state import GraphState
from vggt_slam_pp.contracts.runtime import RunIdentity
from vggt_slam_pp.io.checksums import (
    canonical_json_bytes,
    fsync_directory,
    sha256_bytes,
    sha256_file,
    write_fsynced,
)


class GraphStateIntegrityError(RuntimeError):
    """Raised when a graph snapshot or final pointer is incomplete or modified."""


def _state_payload(state: GraphState) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "run": state.run.model_dump(mode="json"),
        "update_index": state.update_index,
        "world_from_submap": state.world_from_submap.tolist(),
        "transform_kind": state.transform_kind,
        "edge_count": state.edge_count,
        "loop_count": state.loop_count,
        "trajectory_sha256": state.trajectory_sha256,
    }


def _atomic_write(path: Path, value: bytes) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        write_fsynced(temporary_path, value)
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_graph_state(run_root: Path, state: GraphState) -> Path:
    """Append the next state, then atomically advance `final_state.json`."""
    states_root = run_root / "states"
    states_root.mkdir(parents=True, exist_ok=True)
    existing_indices = sorted(
        int(path.stem)
        for path in states_root.glob("*.json")
        if path.stem.isdigit()
    )
    expected_index = 0 if not existing_indices else existing_indices[-1] + 1
    if state.update_index != expected_index:
        raise ValueError(f"next update_index must be {expected_index}")

    state_path = states_root / f"{state.update_index:06d}.json"
    if state_path.exists():
        raise FileExistsError(f"graph state already exists: {state_path}")

    payload = _state_payload(state)
    payload_checksum = sha256_bytes(canonical_json_bytes(payload))
    envelope = {
        "checksum": payload_checksum,
        "state": payload,
    }
    _atomic_write(state_path, canonical_json_bytes(envelope))

    pointer = {
        "schema_version": 1,
        "update_index": state.update_index,
        "path": f"states/{state_path.name}",
        "sha256": sha256_file(state_path),
    }
    _atomic_write(run_root / "final_state.json", canonical_json_bytes(pointer))
    return state_path


def _read_state_file(path: Path) -> GraphState:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["state"]
        expected_checksum = envelope["checksum"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GraphStateIntegrityError(f"invalid graph state: {path}") from exc

    actual_checksum = sha256_bytes(canonical_json_bytes(payload))
    if actual_checksum != expected_checksum:
        raise GraphStateIntegrityError(f"graph state checksum mismatch: {path}")

    return GraphState(
        schema_version=payload["schema_version"],
        run=RunIdentity.model_validate(payload["run"]),
        update_index=payload["update_index"],
        world_from_submap=np.asarray(payload["world_from_submap"], dtype=np.float64),
        transform_kind=payload["transform_kind"],
        edge_count=payload["edge_count"],
        loop_count=payload["loop_count"],
        trajectory_sha256=payload["trajectory_sha256"],
    )


def read_final_graph_state(run_root: Path) -> GraphState:
    """Resolve and verify the latest fully committed graph state."""
    pointer_path = run_root / "final_state.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        relative_path = Path(pointer["path"])
        expected_hash = pointer["sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GraphStateIntegrityError("invalid final_state.json") from exc

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise GraphStateIntegrityError("final state path must stay inside run root")
    state_path = run_root / relative_path
    if not state_path.is_file() or sha256_file(state_path) != expected_hash:
        raise GraphStateIntegrityError("final graph state file checksum mismatch")

    state = _read_state_file(state_path)
    if state.update_index != pointer.get("update_index"):
        raise GraphStateIntegrityError("final state update_index mismatch")
    return state
