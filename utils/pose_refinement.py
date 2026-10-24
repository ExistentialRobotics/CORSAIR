"""Geometry-only ICP refinement for unit-radius-normalized point clouds."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def guard_pose_update(source_xyz, target_xyz, initial_transform, proposed_transform, *, max_distance=.08):
    """Accept lower full-source nearest-neighbor RMSE with no loss of overlap.

    The source must be the observed cloud when one cloud is partial. Overlap is
    its fraction of points within max_distance of the target. No pose labels
    or evaluation errors enter this decision.
    """
    source = np.asarray(source_xyz, dtype=np.float64)
    tree = cKDTree(np.asarray(target_xyz, dtype=np.float64))

    def quality(transform):
        transform = np.asarray(transform, dtype=np.float64)
        if not np.isfinite(transform).all():
            return float('inf'), 0.
        transformed = source @ transform[:3, :3].T + transform[:3, 3]
        distances = tree.query(transformed, k=1)[0]
        return float(np.sqrt(np.mean(distances ** 2))), float(np.mean(distances <= max_distance))

    before_rmse, before_overlap = quality(initial_transform)
    after_rmse, after_overlap = quality(proposed_transform)
    accepted = bool(after_rmse < before_rmse - 1e-12 and after_overlap >= before_overlap - 1e-12)
    details = dict(accepted=accepted, before_rmse=before_rmse, proposed_rmse=after_rmse,
                   before_overlap=before_overlap, proposed_overlap=after_overlap,
                   overlap_distance=max_distance)
    selected = proposed_transform if accepted else initial_transform
    return np.asarray(selected, dtype=np.float64).copy(), details


def refine_pose_icp(source_xyz, target_xyz, initial_transform, multiscale=False, *, partial_target=False,
                    max_iterations=10, guard=True):
    """Refine a source-to-target transform without pose labels or correspondences.

    Thresholds use the normalized coordinates of the ShapeNet pose evaluator.
    If the target is partial, align its observed points into the complete source
    and invert the result, so unobserved source surfaces do not drive the fit.
    max_iterations is the standard ICP cap, or a per-stage cap for multiscale.
    By default, retain the initial transform unless scan-to-target RMSE improves
    and the fraction of observed points within 0.08 does not decrease.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if partial_target:
        reverse = refine_pose_icp(target_xyz, source_xyz, np.linalg.inv(np.asarray(initial_transform)), multiscale,
                                  max_iterations=max_iterations, guard=guard)
        return np.linalg.inv(reverse)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.asarray(source_xyz, dtype=np.float64))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.asarray(target_xyz, dtype=np.float64))
    transform = np.asarray(initial_transform, dtype=np.float64).copy()
    levels = ([(.08, .20, 50), (.04, .10, 40), (.02, .05, 30), (0, .025, 30)]
              if multiscale else [(0, .08, max_iterations)])
    for voxel_size, max_distance, iterations in levels:
        src = source.voxel_down_sample(voxel_size) if voxel_size else source
        dst = target.voxel_down_sample(voxel_size) if voxel_size else target
        result = o3d.pipelines.registration.registration_icp(
            src, dst, max_distance, transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=min(iterations, max_iterations)),
        )
        transform = result.transformation.copy()
    if guard:
        transform, _ = guard_pose_update(source_xyz, target_xyz, initial_transform, transform)
    return transform
