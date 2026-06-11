"""Opt-in bridge from baseline objects to immutable M0 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vggt_slam_pp.adapters.vggt_slam_v1 import adapt_submap
from vggt_slam_pp.contracts.graph_state import GraphState
from vggt_slam_pp.contracts.runtime import RunIdentity
from vggt_slam_pp.io.checksums import canonical_json_bytes, sha256_bytes
from vggt_slam_pp.io.graph_state import write_graph_state
from vggt_slam_pp.io.submap_cache import write_submap_cache


class ExportSession:
    """Own one run directory without mutating any baseline object."""

    def __init__(
        self,
        *,
        output_root: Path,
        run: RunIdentity,
        baseline_sha256: str,
        weight_sha256: str,
    ) -> None:
        self.run = run
        self.baseline_sha256 = baseline_sha256
        self.weight_sha256 = weight_sha256
        self.run_root = Path(output_root) / run.run_id
        self.submaps_root = self.run_root / "submaps"
        self.run_root.mkdir(parents=True, exist_ok=True)
        existing_states = [
            int(path.stem)
            for path in (self.run_root / "states").glob("*.json")
            if path.stem.isdigit()
        ]
        self._next_update_index = 0 if not existing_states else max(existing_states) + 1
        self._last_state_path: Path | None = None

    def export_latest_submap(
        self,
        submap: object,
        loop_sources: tuple[int, ...],
    ) -> Path:
        metadata, arrays = adapt_submap(
            submap,
            run=self.run,
            baseline_sha256=self.baseline_sha256,
            weight_sha256=self.weight_sha256,
            loop_sources=loop_sources,
        )
        return write_submap_cache(self.submaps_root, metadata, arrays)

    @staticmethod
    def _ordered_submaps(graph_map: object) -> list[Any]:
        return list(graph_map.ordered_submaps_by_key())

    @staticmethod
    def _trajectory_checksum(submaps: list[Any]) -> str:
        # This representation captures the same local-to-global pose ingredients
        # as GraphMap.write_poses_to_file without creating an intermediate file.
        payload = []
        for submap in submaps:
            payload.append(
                {
                    "submap_id": int(submap.submap_id),
                    "frame_ids": [str(value) for value in submap.frame_ids],
                    "last_non_loop_frame_index": int(
                        submap.last_non_loop_frame_index
                    ),
                    "camera_to_submap": np.asarray(submap.poses).tolist(),
                    "world_from_submap": np.asarray(submap.H_world_map).tolist(),
                }
            )
        return sha256_bytes(canonical_json_bytes(payload))

    def export_graph_state(self, graph_map: object, graph: object) -> Path:
        submaps = self._ordered_submaps(graph_map)
        if not submaps:
            raise ValueError("cannot export graph state without submaps")
        transforms = np.stack(
            [np.asarray(submap.H_world_map) for submap in submaps],
            axis=0,
        )
        factor_count = int(graph.graph.size())
        loop_count = int(graph.get_num_loops())
        state = GraphState(
            run=self.run,
            update_index=self._next_update_index,
            world_from_submap=transforms,
            transform_kind=(
                "SE3_with_baked_scale"
                if self.run.solver_mode == "baseline_sim3_compat"
                else "SL4"
            ),
            edge_count=factor_count,
            loop_count=loop_count,
            trajectory_sha256=self._trajectory_checksum(submaps),
        )
        self._last_state_path = write_graph_state(self.run_root, state)
        self._next_update_index += 1
        return self._last_state_path

    def finalize(self, graph_map: object, graph: object) -> Path:
        if self._last_state_path is None:
            self.export_graph_state(graph_map, graph)
        final_path = self.run_root / "final_state.json"
        if not final_path.is_file():
            raise RuntimeError("final graph-state pointer was not committed")
        return final_path
