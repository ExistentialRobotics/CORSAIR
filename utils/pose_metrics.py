"""Pose errors with stable float64 reporting for very small rotations."""
import numpy as np


def pose_errors(estimated_transform, ground_truth_transform, symmetry_order=1):
    """Return symmetry-aware RRE in degrees and RTE in input coordinate units.

    Symmetry is rotation about the canonical source Y axis, matching the
    ShapeNet evaluator. Labels are used only for scoring.
    """
    estimated = np.asarray(estimated_transform, dtype=np.float64)
    truth = np.asarray(ground_truth_transform, dtype=np.float64)
    if symmetry_order < 1:
        raise ValueError("symmetry_order must be positive")
    angles = []
    for k in range(symmetry_order):
        angle = 2 * np.pi * k / symmetry_order
        c, s = np.cos(angle), np.sin(angle)
        symmetry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        expected_rotation = truth[:3, :3] @ symmetry.T
        relative = estimated[:3, :3].T @ expected_rotation
        skew = .5 * np.array([relative[2, 1] - relative[1, 2],
                              relative[0, 2] - relative[2, 0],
                              relative[1, 0] - relative[0, 1]])
        angles.append(np.degrees(np.arctan2(np.linalg.norm(skew), .5 * (np.trace(relative) - 1))))
    return float(min(angles)), float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
