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
from datasets.CategoryDataset import *
from datasets.Reader import Reader


class CategoryTestTimeDataset(CategoryDataset):
    """
    Test time CAD database (no random rotation)
    """

    def __init__(self, root, split, catid, voxel_size):

        self.root = root
        self.split = split
        self.catid = catid
        self.voxel_size = voxel_size
        reader = Reader(self.root, self.catid, self.split, 10000)
        readerloader = torch.utils.data.DataLoader(
            reader, batch_size=1, shuffle=False, num_workers=8
        )
        pcs_ref = []
        for data in tqdm(readerloader):
            pcs_ref.append(data[0, :, :].numpy())
        self.pcs = pcs_ref

    def __getitem__(self, idx):

        base_coords = self._getpc(idx)

        rot_base_coords, rot_base_coords_grid, base_coords = self.quant(
            base_coords, base_coords
        )

        base_feat = np.ones([len(rot_base_coords), 1])

        base = {
            "coord": rot_base_coords_grid,
            "origin": rot_base_coords,
            "feat": base_feat,
            "idx": idx,
        }

        return base

    def collate_pair_fn(self, list_data):

        base_dict = list_data

        base_coords = []
        base_feat = []
        base_T = []
        base_origin = []
        base_idx = []

        for idx in range(len(base_dict)):

            base_coords.append(torch.from_numpy(base_dict[idx]["coord"]))
            base_origin.append(torch.from_numpy(base_dict[idx]["origin"]))
            base_feat.append(torch.from_numpy(base_dict[idx]["feat"]))
            base_idx.append(base_dict[idx]["idx"])

        batch_base_coords, batch_base_feat = ME.utils.sparse_collate(
            base_coords, base_feat
        )

        data = {}

        data["base_coords"] = batch_base_coords.int()
        data["base_feat"] = batch_base_feat.float()
        data["base_origin"] = torch.cat(base_origin, 0).float()
        data["base_idx"] = torch.Tensor(base_idx)

        return data
