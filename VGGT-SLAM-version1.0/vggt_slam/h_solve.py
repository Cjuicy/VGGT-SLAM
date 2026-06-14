import open3d as o3d
import numpy as np
import torch
from scipy.linalg import null_space

def to_homogeneous(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])

def apply_homography(H, X, debug=False):
    X_h = to_homogeneous(X)
    X_trans = (H @ X_h.T).T
    if debug:
        print(X_trans[:, 3])
    return X_trans[:, :3] / X_trans[:, 3:]

def apply_homography_batch(H_batch: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Efficiently apply batched 4x4 homographies to 3D points.
    
    Args:
        H_batch: Tensor of shape (B, 4, 4)
        X:       Tensor of shape (N, 3)
    Returns:
        Transformed points: Tensor of shape (B, N, 3)
    """
    B = H_batch.shape[0]
    N = X.shape[0]
    
    # Append 1 to each point: (N, 4)
    ones = torch.ones((N, 1), dtype=X.dtype, device=X.device)
    X_h = torch.cat([X, ones], dim=1)  # (N, 4)

    # Apply homographies: (B, 4, 4) x (N, 4)^T → (B, 4, N)
    X_h = X_h.T.unsqueeze(0).expand(B, 4, N)  # (B, 4, N)
    X_trans = torch.bmm(H_batch, X_h)  # (B, 4, N)

    # Perspective divide
    X_trans = X_trans[:, :3, :] / X_trans[:, 3:4, :]  # (B, 3, N)
    
    # Transpose to (B, N, 3)
    return X_trans.permute(0, 2, 1)

def _validate_correspondences(X_src, X_dst):
    source = np.asarray(X_src, dtype=np.float64)
    target = np.asarray(X_dst, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("X_src must have shape (N, 3)")
    if target.shape != source.shape:
        raise ValueError("X_dst must match X_src shape")
    if source.shape[0] < 5:
        raise ValueError("at least five 3D correspondences are required")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("3D correspondences must contain only finite values")
    return source, target


def _normalization_transform(points, weights):
    """Return a weighted 3D Hartley-style normalization transform."""
    normalized_weights = weights / weights.sum()
    centroid = np.sum(points * normalized_weights[:, None], axis=0)
    centered = points - centroid
    rms_distance = np.sqrt(
        np.sum(normalized_weights * np.sum(centered * centered, axis=1))
    )
    if rms_distance <= np.finfo(np.float64).eps:
        raise ValueError("3D correspondences are degenerate")

    scale_factor = np.sqrt(3.0) / rms_distance
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale_factor
    transform[:3, 3] = -scale_factor * centroid
    normalized = apply_homography(transform, points)
    return transform, normalized


def _projective_design_matrix(source, target):
    """Build the three DLT equations contributed by each 3D correspondence."""
    count = source.shape[0]
    ones = np.ones(count, dtype=np.float64)
    source_h = np.column_stack((source, ones))
    xp, yp, zp = target.T

    design = np.zeros((3 * count, 16), dtype=np.float64)
    design[0::3, 0:4] = -source_h
    design[0::3, 12:16] = source_h * xp[:, None]
    design[1::3, 4:8] = -source_h
    design[1::3, 12:16] = source_h * yp[:, None]
    design[2::3, 8:12] = -source_h
    design[2::3, 12:16] = source_h * zp[:, None]
    return design


def _normalize_sl4(homography):
    """Fix projective scale so a valid 4x4 transform has determinant one."""
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("homography must be a finite 4x4 matrix")
    determinant = float(np.linalg.det(matrix))
    if determinant <= np.finfo(np.float64).eps:
        raise ValueError("homography must have positive non-zero determinant")
    return matrix / determinant**0.25


def estimate_3D_homography_weighted(X_src, X_dst, weights=None):
    """
    使用加权 DLT 估计一个 SL(4) 3D projective homography。

    每个点对应三行 DLT 方程。加权最小二乘要求这三行同时乘
    ``sqrt(w_i)``，不能把点坐标直接写成 ``W @ X``。
    """
    source, target = _validate_correspondences(X_src, X_dst)
    if weights is None:
        point_weights = np.ones(source.shape[0], dtype=np.float64)
    else:
        point_weights = np.asarray(weights, dtype=np.float64)
        if point_weights.shape != (source.shape[0],):
            raise ValueError("weights must have shape (N,)")
        if not np.all(np.isfinite(point_weights)) or np.any(point_weights < 0):
            raise ValueError("weights must be finite and non-negative")
    if np.count_nonzero(point_weights > 0) < 5:
        raise ValueError("at least five correspondences must have positive weight")

    source_transform, normalized_source = _normalization_transform(
        source, point_weights
    )
    target_transform, normalized_target = _normalization_transform(
        target, point_weights
    )
    design = _projective_design_matrix(normalized_source, normalized_target)
    design *= np.repeat(np.sqrt(point_weights), 3)[:, None]

    # Five correspondences produce a 15x16 matrix. In that underdetermined
    # minimal case full V is required to expose the one-dimensional nullspace.
    full_matrices = design.shape[0] < design.shape[1]
    _, _, right_vectors = np.linalg.svd(design, full_matrices=full_matrices)
    normalized_homography = right_vectors[-1].reshape(4, 4)
    homography = (
        np.linalg.inv(target_transform)
        @ normalized_homography
        @ source_transform
    )
    return _normalize_sl4(homography)


def estimate_3D_homography(X_src_batch, X_dst_batch, device=None):
    """
    Estimate batch of 3D Homography.
    
    Inputs:
        X_src_batch: (B, N, 3)
        X_dst_batch: (B, N, 3)
        
    Returns:
        H_batch: (B, 4, 4)
    """
    source_batch = np.asarray(X_src_batch, dtype=np.float64)
    target_batch = np.asarray(X_dst_batch, dtype=np.float64)
    if source_batch.ndim != 3 or source_batch.shape[2] != 3:
        raise ValueError("X_src_batch must have shape (B, N, 3)")
    if target_batch.shape != source_batch.shape:
        raise ValueError("X_dst_batch must match X_src_batch shape")

    B, N, _ = source_batch.shape
    ones = np.ones((B, N))
    x, y, z = (
        source_batch[:, :, 0],
        source_batch[:, :, 1],
        source_batch[:, :, 2],
    )
    xp, yp, zp = (
        target_batch[:, :, 0],
        target_batch[:, :, 1],
        target_batch[:, :, 2],
    )
    source_h = np.stack([x, y, z, ones], axis=2)
    design = np.zeros((B, 3 * N, 16))
    design[:, 0::3, 0:4] = -source_h
    design[:, 0::3, 12:16] = np.stack([x * xp, y * xp, z * xp, xp], axis=2)
    design[:, 1::3, 4:8] = -source_h
    design[:, 1::3, 12:16] = np.stack([x * yp, y * yp, z * yp, yp], axis=2)
    design[:, 2::3, 8:12] = -source_h
    design[:, 2::3, 12:16] = np.stack([x * zp, y * zp, z * zp, zp], axis=2)

    H_batch = np.zeros((B, 4, 4))
    for i in range(B):
        null_vectors = null_space(design[i])
        if null_vectors.shape[1] == 0:
            H_batch[i] = np.eye(4)
            continue
        homography = null_vectors[:, 0].reshape(4, 4)
        if abs(homography[3, 3]) <= np.finfo(np.float64).eps:
            H_batch[i] = np.eye(4)
            continue
        homography = homography / homography[3, 3]
        determinant = np.linalg.det(homography)
        if not np.isfinite(determinant) or determinant < 0.0001:
            H_batch[i] = np.eye(4)
        else:
            H_batch[i] = homography / determinant**0.25

    output_device = torch.device("cpu") if device is None else torch.device(device)
    return torch.tensor(H_batch, dtype=torch.float32, device=output_device)

def is_planar(X, threshold=5e-2):
    X_centered = X - X.mean(axis=0)
    _, S, _ = np.linalg.svd(X_centered)
    normal_strength = S[-1] / S[0]
    return normal_strength < threshold

def scale(X):
    centroid = X.mean(axis=0)
    X_centered = X - centroid  # move centroid to origin

    # Compute average distance to the origin after centering
    avg_norm = np.linalg.norm(X_centered, axis=1).mean()

    # Desired average distance is sqrt(3)
    desired_avg_norm = np.sqrt(3)

    # Compute the uniform scaling factor
    scale = desired_avg_norm / avg_norm

    # Construct the 4x4 similarity transform matrix
    T = np.eye(4)
    T[:3, :3] *= scale  # apply scaling
    T[:3, 3] = -scale * centroid  # apply translation

    X_h = np.hstack([X, np.ones((X.shape[0], 1))])  # shape: (N, 4)

    # Step 2: Apply the transform
    X_transformed_h = (T @ X_h.T).T  # shape: (N, 4)

    # Step 3: Convert back to 3D (drop the homogeneous coordinate)
    X_transformed = X_transformed_h[:, :3]

    return T, X_transformed

# SL4 + RANSAC 用于 单应性(两个不同视角的相机之间一对一的映射关系)矩阵估计的鲁棒求解
def ransac_projective(
    X1_np,
    X2_np,
    threshold=0.01,
    max_iter=300,
    sample_size=5,
    random_seed=None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to torch tensors on GPU
    source, target = _validate_correspondences(X1_np, X2_np)
    if threshold <= 0 or max_iter <= 0:
        raise ValueError("threshold and max_iter must be positive")
    if sample_size < 5 or sample_size > source.shape[0]:
        raise ValueError("sample_size must be between 5 and the point count")
    X1 = torch.tensor(source, dtype=torch.float32, device=device)
    X2 = torch.tensor(target, dtype=torch.float32, device=device)
    N = X1.shape[0]

    # Sample indices for each hypothesis. 批量随机采样
    rng = np.random.default_rng(random_seed)
    indices = torch.tensor(
        rng.integers(0, N, size=(max_iter, sample_size)),
        dtype=torch.long,
        device=device,
    )

    # Gather sampled point sets.
    X1_samples = torch.stack([X1[idx] for idx in indices])  # (max_iter, sample_size, 3)
    X2_samples = torch.stack([X2[idx] for idx in indices])  # (max_iter, sample_size, 3)

    # Estimate homographies. 批量DLT估计 单应性矩阵
    H_ests = estimate_3D_homography(
        X1_samples.cpu().numpy(),
        X2_samples.cpu().numpy(),
        device=device,
    )

    # Apply homographies to all points. 将每个假设的单应矩阵应用到所有点上，看看预测的点和实际的点之间的误差
    X2_preds = apply_homography_batch(H_ests, X1)

    # Compute Euclidean error. 误差与内点判定
    errors = torch.norm(X2_preds - X2[None, :, :], dim=2)

    # Compute inlier masks and counts.
    inlier_masks = errors < threshold  # (max_iter, N)
    inlier_counts = inlier_masks.sum(dim=1)

    # Select best hypothesis 选择最优假设
    best_idx = torch.argmax(inlier_counts)
    best_H = H_ests[best_idx].cpu().numpy()

    return best_H


def irls_projective(
    X1_np,
    X2_np,
    initial_homography,
    confidence=None,
    threshold=0.01,
    max_iter=10,
    convergence_tolerance=1e-7,
):
    """
    从 RANSAC 初值鲁棒精修 SL(4) 单应矩阵,综合考虑两点权重:
      1) 上游给出的 per-point 置信度 (硬过滤后剩余的 [0, 1] 软权重)
      2) 当前残差经过 Tukey biweight 的鲁棒权重
    最终有效权重: w_i = conf_i * tukey(error_i)

    Args:
        X1_np:    (N, 3) 源点云
        X2_np:    (N, 3) 目标点云
        initial_homography: RANSAC 给出的鲁棒初值
        confidence: (N,) 每点非负置信度,None 表示全部置 1
        threshold: Tukey 截断阈值 (归一化坐标空间下)
        max_iter:  IRLS 迭代次数

    Returns:
        H_est:    (4, 4) 属于 SL(4) 的单应矩阵
    """
    source, target = _validate_correspondences(X1_np, X2_np)
    if threshold <= 0 or max_iter <= 0 or convergence_tolerance <= 0:
        raise ValueError("IRLS thresholds and iteration count must be positive")
    if confidence is None:
        confidence_weights = np.ones(source.shape[0], dtype=np.float64)
    else:
        confidence_weights = np.asarray(confidence, dtype=np.float64)
        if confidence_weights.shape != (source.shape[0],):
            raise ValueError("confidence must have shape (N,)")
        if not np.all(np.isfinite(confidence_weights)) or np.any(
            confidence_weights < 0
        ):
            raise ValueError("confidence must be finite and non-negative")
        maximum = float(confidence_weights.max())
        if maximum <= np.finfo(np.float64).eps:
            raise ValueError("confidence must contain a positive value")
        confidence_weights = confidence_weights / maximum

    H_est = _normalize_sl4(initial_homography)

    for _ in range(max_iter):
        prediction = apply_homography(H_est, source)
        errors = np.linalg.norm(prediction - target, axis=1)

        # Tukey biweight on error. 误差方向的鲁棒权重 (硬阈值 + 平滑衰减)
        ratio = errors / threshold
        tukey = np.zeros_like(errors)
        inlier_mask = ratio < 1.0
        tukey[inlier_mask] = (1.0 - ratio[inlier_mask] ** 2) ** 2

        # Combined weight = confidence * error-robustness.
        # 高置信度内点主导拟合,低置信度内点按比例贡献,外点(error>=threshold)归零
        weights = confidence_weights * tukey
        if np.count_nonzero(weights > 0) < 5:
            break

        try:
            candidate = estimate_3D_homography_weighted(source, target, weights)
        except (ValueError, np.linalg.LinAlgError):
            break
        difference = np.linalg.norm(candidate - H_est)
        H_est = candidate
        if difference <= convergence_tolerance:
            break

    return H_est


def ransac_irls_projective(
    X1_np,
    X2_np,
    confidence=None,
    threshold=0.01,
    ransac_max_iter=300,
    irls_max_iter=10,
    sample_size=5,
    random_seed=None,
):
    """先用 RANSAC 抵抗大外点,再用置信度加权 IRLS 精修全部对应点。"""
    initial = ransac_projective(
        X1_np,
        X2_np,
        threshold=threshold,
        max_iter=ransac_max_iter,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    return irls_projective(
        X1_np,
        X2_np,
        initial,
        confidence=confidence,
        threshold=threshold,
        max_iter=irls_max_iter,
    )
