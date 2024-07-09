import os
import numpy as np
import torch
import open3d as o3d
import matplotlib
import matplotlib.cm
from tqdm import tqdm

# from open3d import JVisualizer
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from utils.visualize import Jvisualize
from utils.read_json import build_pcd
from utils.preprocess import apply_transform, chamfer_gpu_1direction
from utils.preprocess import chamfer_kdtree_1direction
from utils.eval_pose import *


def sample_line(p0, p1, sample_num=100):
    """
    Draw line
    """
    d = p0 - p1
    line = p1[:, None] + d[:, None] * np.arange(sample_num + 1) / sample_num
    return line


def draw_alignment_lines(feat0, feat1, xyz0, xyz1, T0, T1, horizon_shift=0):
    """
    Draw lines to visualize perdicted matching pairs
    """
    baseB = feat0
    baseP = feat1

    predict_match = []

    for i in range(len(baseB)):
        j = torch.argmin(torch.norm(baseP - baseB[i, :], dim=1)).item()

        predict_match.append([i, j])

    predict_match = np.array(predict_match)

    xyz0_ali = apply_transform(xyz0, np.linalg.inv(T0)) + np.array([horizon_shift, 0, 0])
    xyz1_ali = apply_transform(xyz1, np.linalg.inv(T1))

    visualizer = JVisualizer()

    n = 30
    for i in range(n):
        m0, m1 = predict_match[i]
        # print(xyz0_ali[m0].shape, xyz1_ali[m1].shape)
        line = sample_line(xyz0_ali[m0], xyz1_ali[m1])
        l = build_pcd(line.T, np.array([0, 0, 0]))
        visualizer.add_geometry(l)

    pcd0 = build_pcd(xyz0_ali, np.array([1, 0, 0]))
    pcd1 = build_pcd(xyz1_ali, np.array([0, 1, 0]))

    visualizer.add_geometry(pcd0)
    visualizer.add_geometry(pcd1)

    visualizer.show()


def generate_heat_map(raw_pc, feat, T, index):
    """
    Visualize the heat map of local features
    """
    local_dist = squareform(pdist(feat.detach().cpu().numpy(), "cosine"))
    local_sort = np.argsort(-local_dist[index, :])

    ranking = np.arange(len(local_sort))

    local_rank = np.zeros([len(local_sort)])

    local_rank[local_sort] = ranking

    colors = matplotlib.cm.ScalarMappable(cmap="hot").to_rgba(local_rank)

    ali_pc = apply_transform(raw_pc, np.linalg.inv(T))

    Jvisualize([ali_pc, ali_pc[index : index + 1, :]], [colors[:, :3], "BLACK"])


def visual_symmetry_points(feat, raw_pc, i, T):
    local_dist = squareform(pdist(feat.detach().cpu().numpy(), "cosine"))
    visualizer = JVisualizer()

    mid_points = []

    index = i
    local_sort = np.argsort(local_dist[index, :])

    ranking = np.arange(len(local_sort))

    local_rank = np.zeros([len(local_sort)])

    local_rank[local_sort] = ranking

    colors = np.zeros([len(local_sort), 3])

    colors[local_rank >= 200] = np.array([1, 0, 0])

    visualizer = JVisualizer()

    xyz0_ali = apply_transform(raw_pc, np.linalg.inv(T))

    chair = build_pcd(xyz0_ali, colors)

    visualizer.add_geometry(chair)

    visualizer.show()

    nns = raw_pc[local_rank < 100]

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(nns)

    label = kmeans.predict(nns)

    c = np.mean(kmeans.cluster_centers_, axis=0)

    mid_points.append(c)

    mid_points = np.array(mid_points)


def split_corr(pcsA, pcsB, featsA, featsB, knn, subsample_size=1000):
    xyzA_corrs = []
    xyzB_corrs = []

    for pcA, pcB, featA, featB in zip(pcsA, pcsB, featsA, featsB):
        idx_0, idx_1 = find_kcorr(featA, featB, k=knn, subsample_size=subsample_size)

        xyzA_corr = pcA[idx_0]
        xyzB_corr = pcB[idx_1]

        xyzA_corrs.append(xyzA_corr)
        xyzB_corrs.append(xyzB_corr)
    # import open3d as o3d
    # pcds_a = []
    # pcds_b = []
    # line_sets = []
    # colors = np.array([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1],
    #     [1, 1, 0]
    # ])
    # offset = np.array([2.0, 0, 0])
    # for pcA, pcB, xyzA_corr, xyzB_corr, color in zip(pcsA, pcsB, xyzA_corrs, xyzB_corrs, colors):
    #     pcds_a.append(build_pcd(pcA, color * 0.5))
    #     pcds_b.append(build_pcd(pcB - offset, color))
    #     line_set = o3d.geometry.LineSet()
    #     line_set.points = o3d.utility.Vector3dVector(np.concatenate([xyzA_corr, xyzB_corr - offset], axis=0))
    #     line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i + len(xyzA_corr)] for i in range(len(xyzA_corr))]))
    #     line_set.colors = o3d.utility.Vector3dVector(np.tile(color, [len(xyzA_corr), 1]))
    #     line_sets.append(line_set)
    # o3d.visualization.draw_geometries(pcds_a + pcds_b + line_sets)
    xyzA_corrs = np.concatenate(xyzA_corrs, axis=0)
    xyzB_corrs = np.concatenate(xyzB_corrs, axis=0)
    return xyzA_corrs, xyzB_corrs


def symmetric_cut4(feat, raw_pc, K, max_sample=100):
    """
    Cut a symmetric object alone its plan of symmetry
    Only support 2 and 4 centers clustering
    feat: [N, feat] geometric feature
    raw_pc: [N, 3] coordinate
    """

    mid_points = []
    centers = []

    random_points = np.random.choice(len(raw_pc), max_sample, replace=False)

    curr_var = 100
    curr_model = None

    for i in range(max_sample):
        index = random_points[i]

        local_dist = np.linalg.norm(feat[index : index + 1, :] - feat, axis=1)

        local_sort = np.argsort(local_dist)

        ranking = np.arange(len(local_sort))

        local_rank = np.zeros([len(local_sort)])

        local_rank[local_sort] = ranking

        if isinstance(raw_pc, torch.Tensor):
            raw_pc = raw_pc.detach().cpu()

        nns = raw_pc[local_rank < 50, :]

        kmeans = KMeans(n_clusters=K, random_state=0, n_init=10).fit(nns)

        centers = kmeans.cluster_centers_

        dist = np.linalg.norm(centers[None, :, :] - centers[:, None, :], 2, 2)

        diag = np.arange(K)

        dist[diag, diag] = 100

        labels = kmeans.predict(raw_pc)
        ratios = [(labels == i).sum() / len(labels) for i in range(K)]

        nn_label = kmeans.predict(nns)
        error = []
        for l in range(K):
            error.append(np.linalg.norm(nns[nn_label == l] - kmeans.cluster_centers_[l], axis=1).mean())

        if dist.min() > 0.15 > max(error) and curr_var > np.sqrt(np.var(ratios)):
            curr_var = np.sqrt(np.var(ratios))
            curr_model = kmeans

    centers = curr_model.cluster_centers_

    dist = np.linalg.norm(centers[None, :, :] - centers[:, None, :], 2, 2)

    diag = np.arange(K)

    dist[diag, diag] = 100
    rank_centers = np.argsort(dist[0, 1:])

    labels = curr_model.predict(raw_pc)

    if K == 2:
        return [labels == 0, labels == 1]
    elif K == 4:
        return [
            labels == 0,
            labels == rank_centers[0] + 1,
            labels == rank_centers[2] + 1,
            labels == rank_centers[1] + 1,
        ]
    else:
        raise ValueError("not defined")


def sym_pose(baseF, xyz0, posF, xyz1, pos_sym, k_nn=5, max_corr=0.20, seed=0):
    """
    Estimate pose with and without symmetry
    """

    idx_0, idx_1 = find_kcorr(baseF, posF, k=k_nn, subsample_size=-1)

    source_pcd = xyz0[idx_0]
    target_pcd = xyz1[idx_1]

    T_est_ransac = registration_based_on_corr(source_pcd, target_pcd, max_corr, seed)

    T_est_ransac = torch.from_numpy(T_est_ransac.astype(np.float32))

    # chamf_dist_ransac = chamfer_gpu_1direction(
    #     apply_transform(xyz0, T_est_ransac).cuda(),
    #     torch.from_numpy(xyz1).cuda(),
    # )
    chamf_dist_ransac = chamfer_kdtree_1direction(apply_transform(xyz0, T_est_ransac), xyz1)

    chamf_dist_best = chamf_dist_ransac
    T_est_best = T_est_ransac

    # if pos_sym >= 2:
    #     base_masks = symmetric_cut4(baseF, xyz0, 4, max_sample=100)
    #     pos_masks = symmetric_cut4(posF, xyz1, 4, max_sample=100)
    # else:
    #     base_masks = symmetric_cut4(baseF, xyz0, 2, max_sample=100)
    #     pos_masks = symmetric_cut4(posF, xyz1, 2, max_sample=100)

    try:
        if pos_sym >= 2:
            base_masks = symmetric_cut4(baseF, xyz0, 4, max_sample=100)
            pos_masks = symmetric_cut4(posF, xyz1, 4, max_sample=100)
        else:
            base_masks = symmetric_cut4(baseF, xyz0, 2, max_sample=100)
            pos_masks = symmetric_cut4(posF, xyz1, 2, max_sample=100)
    except Exception as e:
        tqdm.write(f"symmetry failed use ransac: {e}")
        return T_est_best, chamf_dist_best, T_est_ransac, chamf_dist_ransac, False

    for _ in range(len(base_masks)):
        pcsA = [xyz0[base_mask] for base_mask in base_masks]
        pcsB = [xyz1[pos_mask] for pos_mask in pos_masks]

        featsA = [baseF[base_mask] for base_mask in base_masks]
        featsB = [posF[pos_mask] for pos_mask in pos_masks]

        xyzA_corrs, xyzB_corrs = split_corr(pcsA, pcsB, featsA, featsB, k_nn, subsample_size=-1)

        T_est = registration_based_on_corr(xyzA_corrs, xyzB_corrs, max_corr, seed)

        T_est = torch.from_numpy(T_est.astype(np.float32))

        # chamf_dist = chamfer_gpu_1direction(apply_transform(xyz0, T_est).cuda(), torch.from_numpy(xyz1).cuda())
        chamf_dist = chamfer_kdtree_1direction(apply_transform(xyz0, T_est), xyz1)

        rot_item = pos_masks.pop(0)
        pos_masks.append(rot_item)

        if chamf_dist_best > chamf_dist:
            chamf_dist_best = chamf_dist
            T_est_best = T_est

    if pos_sym >= 2:
        # base_masks = [base_mask1, base_mask2, base_mask3, base_mask4]
        pos_masks = [pos_masks[0], pos_masks[3], pos_masks[2], pos_masks[1]]
    else:
        base_masks = []
        pos_masks = []

    for _ in range(len(base_masks)):
        pcsA = [xyz0[base_mask] for base_mask in base_masks]
        pcsB = [xyz1[pos_mask] for pos_mask in pos_masks]

        featsA = [baseF[base_mask] for base_mask in base_masks]
        featsB = [posF[pos_mask] for pos_mask in pos_masks]

        xyzA_corrs, xyzB_corrs = split_corr(pcsA, pcsB, featsA, featsB, k_nn, subsample_size=-1)

        T_est = registration_based_on_corr(xyzA_corrs, xyzB_corrs, max_corr, seed)

        T_est = torch.from_numpy(T_est.astype(np.float32))

        # chamf_dist = chamfer_gpu_1direction(apply_transform(xyz0, T_est).cuda(), torch.from_numpy(xyz1).cuda())
        chamf_dist = chamfer_kdtree_1direction(apply_transform(xyz0, T_est), xyz1)

        rot_item = pos_masks.pop(0)
        pos_masks.append(rot_item)

        # print(chamf_dist)

        if chamf_dist_best > chamf_dist:
            chamf_dist_best = chamf_dist
            T_est_best = T_est

    return T_est_best, chamf_dist_best, T_est_ransac, chamf_dist_ransac, True
