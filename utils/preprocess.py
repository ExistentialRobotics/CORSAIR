import os
import numpy as np
import torch
import csv
import copy
import open3d as o3d
import transforms3d
from scipy.spatial import KDTree


def load_raw_pc(path, samples):
    pc0 = np.load(path)[:samples, :]
    return pc0


def load_norm_pc(path, samples):
    pc0 = np.load(path)[:samples, :]
    pc0 -= pc0.mean(0)
    pc0 = pc0 / np.max(np.linalg.norm(pc0, 2, 1))
    return pc0


def apply_transform(pointcloud, T):
    """
    pointcloud: [N, 3]
    T: [4, 4]
    """
    pointcloud_homo = np.concatenate([pointcloud, np.ones([len(pointcloud), 1])], 1)
    rot_pointcloud_homo = np.matmul(T, pointcloud_homo.T).T
    rot_pointcloud = rot_pointcloud_homo[:, :3]

    return rot_pointcloud


def chamfer_gpu(pc0, pc1):
    pc0 = pc0[None, :, :]
    pc1 = pc1[:, None, :]
    delta = pc0 - pc1

    return delta.norm(dim=2).min(0)[0].mean() + delta.norm(dim=2).min(1)[0].mean()


def chamfer_gpu_1direction(pc0, pc1):

    pc0 = pc0[None, :, :]
    pc1 = pc1[:, None, :]
    delta = pc0 - pc1
    return delta.norm(dim=2).min(0)[0].mean().item()


def chamfer_kdtree_1direction(pc0, pc1):
    tree = KDTree(pc1)
    dd, ii = tree.query(pc0, k=1)
    return dd.mean()


def random_rotation(pointcloud):
    np.random.seed()

    R = transforms3d.euler.euler2mat(
        np.random.uniform(0, 2 * np.pi),
        np.random.uniform(0, 2 * np.pi),
        np.random.uniform(0, 2 * np.pi),
    )
    T = np.random.uniform(-0.5, 0.5, 3)

    trans = np.eye(4)
    trans[:3, :3] = R
    trans[:3, 3] = T
    return np.matmul(R, pointcloud[:, :, None])[:, :, 0] + T, trans


def random_rotation_id(pointcloud):

    trans = np.eye(4)
    return pointcloud, trans


def read_match(path_match, path_mismatch):
    with open(path_match, "r") as f:
        lines = f.readlines()
    # print(lines[0])
    file_list = lines[0].strip(",").split(",")

    match_map = [line.strip("\n").strip(" ").split(" ") for line in lines[1:]]

    with open(path_mismatch, "r") as f:
        lines = f.readlines()

    mismatch_map = [line.strip("\n").strip(" ").split(" ") for line in lines[1:]]

    return file_list, match_map, mismatch_map


def read_catname(path):
    name2id = {}
    id2name = {}
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        # print(line.strip('\n').split(' '))
        catid, name = line.strip("\n").strip(" ").split(" ")
        name2id[name] = catid
        id2name[catid] = name

    return id2name, name2id


def read_label(path):
    """
    organized as follow:
    catid/split/modelid.npy catid subcatid
    """
    file_list = []
    label2data = {}
    data2label = {}
    with open(path, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if len(line.strip("\n")):
            file_path, catid, subcatid = line.strip("\n").split(" ")
            file_list.append(file_path)

            data2label[file_path] = [catid, subcatid]

            if not catid in label2data.keys():
                label2data[catid] = {}
            if not subcatid in label2data[catid].keys():
                label2data[catid][subcatid] = [[file_path], [idx]]
            else:
                label2data[catid][subcatid][0].append(file_path)
                label2data[catid][subcatid][1].append(idx)

    return file_list, label2data, data2label


def read_file(path):

    res = []
    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if len(line.strip("\n")):
            res.append(line.strip("\n"))

    return res


def read_split(path):
    """
    id, CatId, SubcatId, modelId, split
    """
    Cat2Id = {"train": {}, "test": {}, "val": {}}
    Id2Cat = {}
    with open(path, newline="") as csv_file:
        reader = csv.reader(
            csv_file,
        )
        lines = [line for line in reader]

    lines = lines[1:]

    for line in lines:
        _, catid, subcatid, modelid, split = line
        Id2Cat[modelid] = {"CatId": catid, "SubcatId": subcatid, "split": split}
        if not catid in Cat2Id[split].keys():
            Cat2Id[split][catid] = {}

        if not subcatid in Cat2Id[split][catid].keys():
            Cat2Id[split][catid][subcatid] = [modelid]

        else:
            Cat2Id[split][catid][subcatid].append(modelid)

    return Cat2Id, Id2Cat


def print_stat(Cat2Id, Id2Cat, split, catid):

    # print(Cat2Id["train"][catid].keys())
    stat = {}
    print(split)
    for key in list(Cat2Id[split][catid].keys()):
        print("SubcatId:{}, count:{}".format(key, len(Cat2Id[split][catid][key])))
        stat[key] = len(Cat2Id[split][catid][key])
    return stat


def get_matching_indices(source_pcd, target_pcd, search_voxel_size, K=None):

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pcd)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pcd)

    pcd_tree = o3d.geometry.KDTreeFlann(target)

    base_inds = []
    match_inds = []
    for i, point in enumerate(source.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)

        if K is not None:
            idx = idx[:K]

        base_inds += len(idx) * [i]
        match_inds += idx

    return list(zip(base_inds, match_inds))


def generate_rand_negative_pairs(positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
        positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
        N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]


def _hash(arr, M):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M**d
        else:
            hash_vec += arr[d] * M**d
    return hash_vec


def path_dict(root):
    """
    Designed for Shapenet pointcloud 15k
    """
    id2path = {}
    catids = os.listdir(root)
    for catid in catids:
        if catid == ".DS_Store":
            continue
        for split in ["train", "val", "test"]:
            for f in os.listdir(os.path.join(root, catid, split)):
                id2path[f.split(".")[0]] = os.path.join(root, catid, split, f)
                # print(f.split('.')[0], os.path.join(root, catid, split, f))
    return id2path
