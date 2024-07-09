import torch
import numpy as np
import open3d as o3d
import torch.functional as F
from scipy.spatial import KDTree


def pdist(A, B, dist_type="L2"):
    if dist_type == "L2":
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == "SquareL2":
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError("Not implemented")


def find_nn_cpu(feat0, feat1, return_distance=False):
    feat1tree = KDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=1, workers=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def find_knn_cpu(feat0, feat1, k, return_distance=False):
    feat1tree = KDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=k, workers=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def find_nn_gpu(F0, F1, nn_max_n=-1, return_distance=False, dist_type="SquareL2"):
    # Too much memory if F0 or F1 large. Divide the F0
    if nn_max_n > 1:
        N = len(F0)
        C = int(np.ceil(N / nn_max_n))
        stride = nn_max_n
        dists, inds = [], []
        for i in range(C):
            dist = pdist(F0[i * stride : (i + 1) * stride], F1, dist_type=dist_type)
            min_dist, ind = dist.min(dim=1)
            dists.append(min_dist.detach().unsqueeze(1).cpu())
            inds.append(ind.cpu())

        if C * stride < N:
            dist = pdist(F0[C * stride :], F1, dist_type=dist_type)
            min_dist, ind = dist.min(dim=1)
            dists.append(min_dist.detach().unsqueeze(1).cpu())
            inds.append(ind.cpu())

        dists = torch.cat(dists)
        inds = torch.cat(inds)
        assert len(inds) == N
    else:
        dist = pdist(F0, F1, dist_type=dist_type)
        min_dist, inds = dist.min(dim=1)
        dists = min_dist.detach().unsqueeze(1).cpu()
        inds = inds.cpu()
    if return_distance:
        return inds, dists
    else:
        return inds


# for each point in
def find_knn_gpu(F0, F1, k, nn_max_n=-1, return_distance=False, dist_type="SquareL2"):
    # Too much memory if F0 or F1 large. Divide the F0
    if nn_max_n > 1:
        N = len(F0)
        C = int(np.floor(N / nn_max_n))
        stride = nn_max_n
        dists, inds = [], []
        for i in range(C):
            dist = pdist(F0[i * stride : (i + 1) * stride], F1, dist_type=dist_type)
            # min_dist, ind = dist.min(dim=1)
            min_dist, ind = dist.topk(k=k, dim=1, largest=False)
            dists.append(min_dist.reshape((-1,)).detach().unsqueeze(1).cpu())
            inds.append(ind.reshape((-1,)).cpu())

        if C * stride < N:
            dist = pdist(F0[C * stride :], F1, dist_type=dist_type)
            # min_dist, ind = dist.min(dim=1)
            min_dist, ind = dist.topk(k=k, dim=1, largest=False)
            dists.append(min_dist.reshape((-1,)).detach().unsqueeze(1).cpu())
            inds.append(ind.reshape((-1,)).cpu())

        dists = torch.cat(dists)
        inds = torch.cat(inds)
        assert len(inds) == N * k
    else:
        dist = pdist(F0, F1, dist_type=dist_type)
        # min_dist, inds = dist.min(dim=1)
        min_dist, inds = dist.topk(k=k, dim=1, largest=False)
        dists = min_dist.reshape((-1,)).detach().unsqueeze(1).cpu()
        inds = inds.reshape((-1,)).cpu()
    if return_distance:
        return inds, dists
    else:
        return inds
