"""Translation APE computed through evo's Python API."""

from __future__ import annotations

from pathlib import Path

from evo.core import metrics, sync
from evo.tools import file_interface


def evaluate_translation_ape(
    groundtruth_path: Path,
    estimate_path: Path,
    *,
    max_timestamp_difference: float = 0.01,
) -> dict[str, float | int]:
    """Associate, Sim(3)-align, and report translation APE RMSE."""
    reference = file_interface.read_tum_trajectory_file(groundtruth_path)
    estimate = file_interface.read_tum_trajectory_file(estimate_path)
    reference, estimate = sync.associate_trajectories(
        reference,
        estimate,
        max_diff=max_timestamp_difference,
        first_name="groundtruth",
        snd_name="estimate",
    )
    if reference.num_poses < 2:
        raise ValueError("ATE requires at least two associated poses")

    estimate.align(reference, correct_scale=True)
    metric = metrics.APE(metrics.PoseRelation.translation_part)
    metric.process_data((reference, estimate))
    return {
        "translation_ape_rmse": float(
            metric.get_statistic(metrics.StatisticsType.rmse)
        ),
        "associated_pose_count": int(reference.num_poses),
    }
