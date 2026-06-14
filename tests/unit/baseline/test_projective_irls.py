import sys
from pathlib import Path

import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[3] / "VGGT-SLAM-version1.0"
sys.path.insert(0, str(BASELINE_ROOT))

from vggt_slam.h_solve import (  # noqa: E402
    apply_homography,
    estimate_3D_homography_weighted,
    ransac_irls_projective,
    ransac_projective,
)


def _normalized_homography() -> np.ndarray:
    homography = np.array(
        [
            [1.08, -0.04, 0.02, 0.3],
            [0.03, 0.96, -0.05, -0.2],
            [-0.01, 0.06, 1.03, 0.15],
            [0.01, -0.015, 0.005, 1.0],
        ],
        dtype=np.float64,
    )
    return homography / np.linalg.det(homography) ** 0.25


def test_weighted_dlt_ignores_zero_weight_outlier() -> None:
    rng = np.random.default_rng(4)
    source = rng.uniform(-1.0, 1.0, size=(40, 3))
    target = apply_homography(_normalized_homography(), source)
    target[-1] = np.array([20.0, -15.0, 8.0])
    weights = np.ones(len(source), dtype=np.float64)
    weights[-1] = 0.0

    estimated = estimate_3D_homography_weighted(source, target, weights)
    prediction = apply_homography(estimated, source[:-1])

    np.testing.assert_allclose(prediction, target[:-1], atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(estimated), 1.0, atol=1e-8)


def test_weighted_dlt_supports_five_point_minimal_sample() -> None:
    source = np.array(
        [
            [-0.8, -0.4, 0.2],
            [0.7, -0.5, -0.3],
            [-0.2, 0.9, -0.6],
            [0.4, 0.3, 0.8],
            [-0.6, 0.5, 0.7],
        ],
        dtype=np.float64,
    )
    target = apply_homography(_normalized_homography(), source)

    estimated = estimate_3D_homography_weighted(source, target)

    np.testing.assert_allclose(
        apply_homography(estimated, source),
        target,
        atol=1e-8,
    )


def test_ransac_irls_refines_noisy_inliers_without_dense_weight_matrix() -> None:
    rng = np.random.default_rng(12)
    source = rng.uniform(-1.0, 1.0, size=(240, 3))
    clean_target = apply_homography(_normalized_homography(), source)
    observed_target = clean_target + rng.normal(0.0, 0.0015, clean_target.shape)
    outlier_indices = rng.choice(len(source), size=60, replace=False)
    observed_target[outlier_indices] = rng.uniform(-3.0, 3.0, size=(60, 3))
    confidence = np.ones(len(source), dtype=np.float64)
    confidence[outlier_indices] = 0.1

    initial = ransac_projective(
        source,
        observed_target,
        threshold=0.01,
        max_iter=300,
        random_seed=9,
    )
    refined = ransac_irls_projective(
        source,
        observed_target,
        confidence=confidence,
        threshold=0.01,
        ransac_max_iter=300,
        irls_max_iter=10,
        random_seed=9,
    )

    inlier_mask = np.ones(len(source), dtype=bool)
    inlier_mask[outlier_indices] = False
    initial_rmse = np.sqrt(
        np.mean(
            np.sum(
                (apply_homography(initial, source[inlier_mask]) - clean_target[inlier_mask])
                ** 2,
                axis=1,
            )
        )
    )
    refined_rmse = np.sqrt(
        np.mean(
            np.sum(
                (apply_homography(refined, source[inlier_mask]) - clean_target[inlier_mask])
                ** 2,
                axis=1,
            )
        )
    )

    assert refined_rmse < initial_rmse
    source_text = (BASELINE_ROOT / "vggt_slam" / "h_solve.py").read_text(
        encoding="utf-8"
    )
    assert "torch.diag" not in source_text
