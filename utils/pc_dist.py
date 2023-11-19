import transforms3d.euler as euler
import open3d as o3d
import numpy as np
import torch
import os
from tqdm import tqdm
import threading
import argparse
from utils.preprocess import (
    read_file,
    read_split,
    print_stat,
    load_raw_pc,
    load_norm_pc,
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dir",
    type=str,
    default="/Users/tyzhao/Desktop/workspace/data/ShapeNetCore.v2.PC15k/",
)
parser.add_argument("--mode", type=str, default="test")
parser.add_argument("--num_mismatch", type=int, default=20)
parser.add_argument("--num_match", type=int, default=10)


def chamfer(pc0, pc1):
    """
    Chamfer distance on gpu
    """
    pc0 = pc0[None, :, :]
    pc1 = pc1[:, None, :]
    delta = pc0 - pc1
    return delta.norm(dim=2).min(0)[0].mean() + delta.norm(dim=2).min(1)[0].mean()


def compute_chamfer(base_idx, pc0, pcs, domain, table):

    for i in range(domain[0], domain[1]):
        table[base_idx, i] = chamfer(pc0, pcs[i])


def compute_dist(pcs):
    """
    Compute pair-wise Chamfer distance matrix within a set
    """
    length = len(pcs)
    table = np.eye(length) * 100

    for base_idx in tqdm(range(len(pcs))):
        # print("start: {}".format(base_idx))
        if base_idx % 10 == 0:
            print("{}/{}".format(base_idx + 1, len(pcs)))

        total = len(pcs) - base_idx - 1
        threds = []
        if total > 8:
            for i in range(8):
                domain = [
                    base_idx + 1 + i * total // 8,
                    base_idx + 1 + (i + 1) * total // 8,
                ]
                t = threading.Thread(
                    target=compute_chamfer,
                    args=(base_idx, pcs[base_idx], pcs, domain, table),
                )
                t.start()
                threds.append(t)
        else:
            t = threading.Thread(
                target=compute_chamfer,
                args=(base_idx, pcs[base_idx], pcs, [base_idx + 1, len(pcs)], table),
            )
            t.start()
            threds.append(t)

        for t in threds:
            t.join()

    table += table.T
    return table


def shapenet_cat(split, catid):
    root = "/scannet/ShapeNetCore.v2.PC15k"
    target = "/scannet/tables"

    pcs = []
    print(catid, len(os.listdir(os.path.join(root, catid, split))))
    files = os.listdir(os.path.join(root, catid, split))
    files.sort()
    for _, objid in enumerate(files):
        pcs.append(
            torch.Tensor(
                load_norm_pc(os.path.join(root, catid, split, objid), 2000),
                device=torch.device("cuda"),
            )
        )

    table = compute_dist(pcs)
    np.save(os.path.join(target, "{}_{}.npy".format(catid, split)), table)


if __name__ == "__main__":
    shapenet_cat("train", "03001627")
    shapenet_cat("val", "03001627")
    shapenet_cat("test", "03001627")
