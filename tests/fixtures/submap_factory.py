from types import SimpleNamespace

import numpy as np


def make_baseline_submap(*, poses_as_3x4: bool = False) -> SimpleNamespace:
    points = np.arange(2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    poses[1, :3, 3] = (1.0, 2.0, 3.0)
    if poses_as_3x4:
        poses = poses[:, :3, :]

    return SimpleNamespace(
        submap_id=4,
        pointclouds=points,
        colors=np.full((2, 2, 3, 3), 127, dtype=np.uint8),
        conf=np.array(
            [
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                [[60.0, 50.0, 40.0], [30.0, 20.0, 10.0]],
            ],
            dtype=np.float32,
        ),
        conf_threshold=30.0,
        vggt_intrinscs=np.repeat(np.eye(3)[None], 2, axis=0),
        poses=poses,
        frame_ids=[100.0, 101.0],
        last_non_loop_frame_index=1,
        H_world_map=np.eye(4),
        raw_predictions={"world_points": np.full_like(points, -999.0)},
    )
