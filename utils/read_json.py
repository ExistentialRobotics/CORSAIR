# script for scan2cad data collection
# save cropped raw point cloud

import os
import json
import csv
from tqdm import tqdm
import numpy as np
import open3d as o3d
import transforms3d
from utils.preprocess import path_dict, load_raw_pc, apply_transform


def load_json(path):
    """
    Read .json files
    """
    f = open(path)
    data = json.load(f)

    return data


def load_csv(path):
    """
    Read .csv files
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        content = [row for row in reader]

    return content


def build_pcd(points, colors):
    """
    Build a open3d Point Cloud with given points and colors
    Input:
        points: array of [n, 3]
        colors: array of [n, 3] or  array of [3, ]
    """
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(points)

    if colors.shape[0] == points.shape[0]:
        pcd0.colors = o3d.utility.Vector3dVector(colors)
    elif colors.shape[0] == 3:
        pcd0.paint_uniform_color(colors)
    else:
        raise ValueError("unknown color dimension")

    return pcd0


def to_T(translation, quaternion, scale):
    translation = np.array(translation)
    quaternion = np.array(quaternion)
    scale = np.array(scale)

    # euler = transforms3d.euler.quat2euler(quaternion)
    # R = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2])
    R = transforms3d.quaternions.quat2mat(quaternion)
    M, S = np.eye(4), np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = translation
    S[0, 0] = scale[0]
    S[1, 1] = scale[1]
    S[2, 2] = scale[2]
    T = np.matmul(M, S)

    return T


def apply_trans(pointcloud, translation, quaternion, scale, mode):
    """
    Apply transformation to point cloud coords
    Input:
        pointcloud: array of [n, 3]
        trans, quat, scale: load from json
        mode: normal, homo
            normal: return [n, 3] coords
            homo: return [n, 4] coords
    """

    T = to_T(translation, quaternion, scale)

    pc = apply_transform(pointcloud, T)

    return pc


def convert_tri(tri, indices):
    d = {}
    for i in range(indices.shape[0]):
        d[indices[i]] = i

    assert tri.shape[1] == 3
    for i in range(tri.shape[0]):
        for j in range(3):
            tri[i][j] = d[tri[i][j]]

    return tri
