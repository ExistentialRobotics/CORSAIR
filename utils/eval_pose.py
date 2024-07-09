import os

import numpy as np
import open3d as o3d
import torch
import transforms3d
from tqdm import tqdm

from utils.find_nn import find_nn_gpu, find_nn_cpu, find_knn_cpu


def find_corr(xyz0, xyz1, F0, F1, nn_max_n, subsample_size=-1):
    """
    Find top-1 matching point pairs based on local features (F0 and F1)
    """
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=nn_max_n)
    if subsample_size > 0 and subsample:
        return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
        return xyz0, xyz1[nn_inds]


def find_kcorr(F0, F1, k=1, nn_max_n=500, subsample_size=-1):
    """
    Find top-k matching point pairs based on local features (F0 and F1)
    """
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]
    # Compute the nn
    # device = torch.device("cuda")
    # if not torch.is_tensor(F0):
    #     F0 = torch.from_numpy(F0).to(device)
    # if not torch.is_tensor(F1):
    #     F1 = torch.from_numpy(F1).to(device)

    if k == 1:
        # nn_inds = find_nn_gpu(F0, F1, nn_max_n=nn_max_n)
        nn_inds = find_nn_cpu(F0, F1).flatten()
    else:
        # nn_inds = find_knn_gpu(F0, F1, k=k, nn_max_n=nn_max_n)
        nn_inds = find_knn_cpu(F0, F1, k=k).flatten()

    if subsample_size > 0 and subsample:
        inds0 = np.repeat(inds0, k)
        return inds0, inds1[nn_inds]
    else:
        inds0 = list(range(len(F0)))
        inds0 = np.repeat(inds0, k)
        return inds0, nn_inds


def registration_based_on_corr(source_pcd, target_pcd, max_corr_dist=0.03, seed=0):
    source_pcd_o3d = o3d.geometry.PointCloud()
    source_pcd_o3d.points = o3d.utility.Vector3dVector(source_pcd.astype(np.float64))
    target_pcd_o3d = o3d.geometry.PointCloud()
    target_pcd_o3d.points = o3d.utility.Vector3dVector(target_pcd.astype(np.float64))
    corr_o3d = o3d.utility.Vector2iVector(np.asarray(list(map(lambda x: [x, x], range(source_pcd.shape[0])))))
    # hyperparameter to tune
    # seems to be better when max_corr_dist is larger
    # max_corr_dist = 0.03
    # there are several other parameters of RANSAC
    o3d.utility.random.seed(seed)
    # tqdm.write("running registration_ransac_based_on_correspondence")
    # os.environ["OMP_NUM_THREADS"] = "1"
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source_pcd_o3d, target_pcd_o3d, corr_o3d, max_corr_dist, ransac_n=10
    )
    # os.environ.pop("OMP_NUM_THREADS")
    T_est = result.transformation
    return T_est


def eval_pose(T_est, T0, T1, axis_symmetry=1):
    """
    Eval rotation error and translation error.
    Rotational symmetry is taken into consideration
    """

    t_loss_best = np.inf
    r_loss_best = np.inf

    for i in range(axis_symmetry):

        R = transforms3d.euler.euler2mat(0, i * (2 * np.pi / axis_symmetry), 0)
        trans = np.eye(4)
        trans[:3, :3] = R
        T_gt = np.matmul(T1, np.matmul(np.linalg.inv(trans), np.linalg.inv(T0))).astype(np.float32)

        r_loss = np.arccos(np.clip((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2, -1, 1))
        t_loss = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])

        # avg_dist = np.linalg.norm(apply_transform(xyz0_corr, T_est)-xyz1_corr, axis=1).mean()

        if r_loss_best > r_loss:
            r_loss_best = r_loss
            t_loss_best = t_loss

    return t_loss_best, r_loss_best
