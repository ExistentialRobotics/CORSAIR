import os
import numpy as np
import open3d as o3d
import random
from utils.preprocess import read_label, random_rotation, load_norm_pc, apply_transform
from utils.read_json import *

# from open3d import JVisualizer


def visualize_pc(pcs, colors=None, txt=""):
    pcds = []
    for idx, pc in enumerate(pcs):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        if not colors == None:
            pcd.paint_uniform_color(colors[idx])
        else:
            pcd.paint_uniform_color(np.array([1, 1, 1]) * idx / len(pcs))
        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds, txt)


def visualize_path(
    root, pathes, random_rot=False, colors=None, txt="open3d", samples=2000
):
    pcs = []
    objs = []
    for idx, path in enumerate(pathes):
        pc0 = load_norm_pc(os.path.join(root, path), samples)
        if random_rot:
            pc0, trans = random_rotation(pc0)
        # pc0 = np.matmul(np.linalg.inv(trans), pc0[:,:,None])[:,:,0]
        # print(type(pc0))
        pcs.append(pc0)
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(pc0)
        if not colors == None:
            pcd0.colors = colors[idx]
        else:
            pcd0.paint_uniform_color(np.array([1, 1, 1]) * idx / len(pathes))
        objs.append(pcd0)

    o3d.visualization.draw_geometries(objs, txt)


def Jvisualize(pcs, colors):
    """
    Visualization in Jupyter notebook
    pcs: list of [N, 3] array
    colors: list of [3, ] array
    """

    COLORS_DICT = {
        "BLACK": np.array([0, 0, 0]),
        "RED": np.array([1, 0, 0]),
        "GREEN": np.array([0, 1, 0]),
        "BLUE": np.array([0, 0, 1]),
    }

    visualizer = JVisualizer()

    for pc, color in zip(pcs, colors):
        if isinstance(color, str):
            color = COLORS_DICT[color]

        pcd = build_pcd(pc, color)
        visualizer.add_geometry(pcd)

    visualizer.show()


def visual_retrieval(idx0, idx1, dataset):
    """
    Visualize point cloud pairs given index and dataset
    dataset: pytorch dataset
    """
    data_a = dataset.__getitem__(idx0)[0]
    data_ret = dataset.__getitem__(idx1)[0]

    a = data_a["origin"]
    ret = data_ret["origin"]

    a_T = data_a["T"]
    ret_T = data_ret["T"]

    a_ali = apply_transform(a, np.linalg.inv(a_T))
    ret_ali = apply_transform(ret, np.linalg.inv(ret_T))

    Jvisualize([a_ali, ret_ali], [np.array([1, 0, 0]), np.array([0, 1, 0])])


def visual_pose(raw_pc0, raw_pc1, Test, T0, T1):
    xyz0_est = apply_transform(raw_pc0, Test)
    xyz1_est = raw_pc1

    xyz0_est = apply_transform(xyz0_est, np.linalg.inv(T1))
    xyz1_est = apply_transform(xyz1_est, np.linalg.inv(T1))

    Jvisualize([xyz0_est, xyz1_est], ["RED", "GREEN"])
