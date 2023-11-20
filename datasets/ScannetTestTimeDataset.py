import os
import torch
import csv
import random
from tqdm import tqdm
import numpy as np
import open3d as o3d
import transforms3d
import MinkowskiEngine as ME

from utils.preprocess import *
from utils.visualize import *
from utils.read_json import *
from datasets.CategoryDataset import CategoryDataset
from datasets.Reader import *


class ScannetTestTimeDataset(CategoryDataset):
    """
    Scan2CAD test time dataset.
    """

    def __init__(
        self,
        scan_root,
        cad_root,
        CADLib,
        Scan2CadInfo,
        split,
        catid,
        pos_ratio,
        neg_ratio,
        voxel_size,
        preload=True,
    ):

        self.scan_root = scan_root
        self.cad_root = cad_root
        self.catid = catid
        self.split = split
        self.voxel_size = voxel_size

        self.preload = preload

        self.badscans = []

        self.CADLib = CADLib
        self.table = self.CADLib.table
        # TEST: Modify the distance representation. ground truth is included in the top match
        r = np.arange(len(self.table))
        self.table[r, r] = 0

        self.id2idx = self.CADLib.id2idx

        (
            self.files,
            self.BestMatches,
            self.CadPoses,
            self.ScanPoses,
        ) = Scan2CadInfo.get_split(split)

        # Read Scannet scans
        print("Loading Scannet scans")
        self.pcs = []
        if self.preload:
            reader = ScannetReader(self.scan_root, self.files, 10000)
            readerloader = torch.utils.data.DataLoader(
                reader, batch_size=1, shuffle=False, num_workers=8
            )

            for data in tqdm(readerloader):
                self.pcs.append(data[0, :, :].numpy())
        else:
            self.pcs = [os.path.join(self.scan_root, file) for file in self.files]

    def _getscan(self, idx):
        if self.preload:
            return self.pcs[idx]
        else:
            return load_raw_pc(self.pcs[idx], 10000)

    def _getcad(self, idx):
        if self.preload:
            return self.CADLib.CadPcs[idx]
        else:
            return load_raw_pc(self.CADLib.CadPcs[idx], 10000)

    def __getitem__(self, idx):

        # sample valid positive cad model
        positive_idx = self.id2idx[self.BestMatches[idx]]

        base_coords = self._getscan(idx)
        pos_coords = self._getcad(positive_idx)

        T_base = to_T(
            self.ScanPoses[idx]["translation"],
            self.ScanPoses[idx]["rotation"],
            self.ScanPoses[idx]["scale"],
        )

        T_pos = to_T(
            self.CadPoses[idx]["translation"],
            self.CadPoses[idx]["rotation"],
            self.CadPoses[idx]["scale"],
        )

        base_coords = apply_transform(
            base_coords, np.matmul(np.linalg.inv(T_pos), T_base)
        )
        # pos_coords = apply_transform(pos_coords, T_pos)

        # normalize. Scan normalize with the positive CAD model

        # t = pos_coords.mean(0)
        # TODO: -t before, but the alignment is bad, double check!
        base_coords -= base_coords.mean(0)
        pos_coords -= pos_coords.mean(0)

        r = np.max(np.linalg.norm(pos_coords, 2, 1))

        base_coords = base_coords / r
        pos_coords = pos_coords / r

        rot_base_coords, base_trans = random_rotation(base_coords)
        # rot_pos_coords, pos_trans = random_rotation(pos_coords)
        rot_pos_coords = pos_coords
        pos_trans = np.eye(4)

        base_feat = np.ones([len(rot_base_coords), 1])
        pos_feat = np.ones([len(rot_pos_coords), 1])

        # tag: data output
        base = {
            "coord": rot_base_coords,
            "origin": rot_base_coords,
            "feat": base_feat,
            "T": base_trans,
            "idx": idx,
        }
        pos = {
            "coord": rot_pos_coords,
            "origin": rot_pos_coords,
            "feat": pos_feat,
            "T": pos_trans,
            "idx": self.id2idx[self.BestMatches[idx]],
        }

        return base, pos
