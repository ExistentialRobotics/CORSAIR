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
from datasets.ChairDataset import *


class HardCategoryDataset(ChairDataset):
    def __init__(
        self, root, split, catid, table_root, pos_ratio, neg_ratio, voxel_size
    ):
        self.root = root
        self.split = split
        self.catid = catid
        self.table_root = table_root
        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio
        # self.rotation = rotation
        self.voxel_size = voxel_size

        self.id2name, self.name2id = read_catname("./config/CatName.txt")
        pcs_ref = self.load_data()

        table_ref = np.load(
            os.path.join(self.table_root, "{}_{}.npy".format(catid, split))
        )

        self.table, self.pcs = self.filter_data(table_ref, pcs_ref)
        # self.table, self.pcs = table_ref, pcs_ref

        self.rank_a = np.argsort(self.table, 1)
        self.rank_d = np.argsort(-self.table, 1)

        self.pos_n = int(self.__len__() * pos_ratio)
        self.neg_n = int(self.__len__() * neg_ratio)
        # print(self.pos_n, self.neg_n)

    def filter_data(self, table_ref, pcs_ref, thres=0.15, num=3):

        while True:
            z = []
            for t in table_ref:
                z.append((t <= thres).sum())
            z = np.array(z)

            mask = (z >= num).nonzero()[0]

            if len(mask) == len(table_ref):
                break
            print(len(mask))

            table = table_ref[mask, :]
            table = table[:, mask]
            table_ref = table.copy()

            pcs = [pcs_ref[item] for item in mask]
            pcs_ref = pcs

        return table, pcs

    def load_data(
        self,
    ):
        pcs = []
        files = os.listdir(os.path.join(self.root, self.catid, self.split))
        files.sort()
        for name in files:
            pcs.append(os.path.join(self.root, self.catid, self.split, name))

        return pcs

    def generate_positive_inst(self, idx):

        topn = self.pos_n
        dist_rank = self.rank_a[idx, :]
        valid = (self.table[idx, :] < 0.15).sum()
        topn = min(topn, valid)
        # print(topn, valid)
        select_idx = np.random.choice(np.arange(topn), 1, replace=False)
        return dist_rank[select_idx]

    def generate_negative_inst(self, idx):

        topn = self.neg_n
        dist_rank = self.rank_d[idx, :]
        valid = (self.table[idx, :] > 0.2).sum() - 1
        topn = min(topn, valid)
        # print(topn, valid)
        select_idx = np.random.choice(np.arange(topn), 4, replace=False) + 1

        return dist_rank[select_idx]

    def _getpc(self, idx):
        return load_norm_pc(self.pcs[idx], 10000)

    def _getlabel(self, idx):
        return 0

    def generate_local_pair(self, base, pos, neg):
        """
        base, pos, neg: np array
        """
        # TODO: merge positive and negative
        N0, N1, N2 = len(base), len(pos), len(neg)
        pip = np.array(get_matching_indices(base, pos, 0.03))
        if pip.shape[0] < 0.1 * min(N0, N1):
            return None, None, None
        pin = np.array(
            generate_rand_negative_pairs(
                pip, max(N0, N1), N0, N1, N_neg=np.floor(len(pip))
            )
        )
        nin = np.array(
            generate_rand_negative_pairs(
                [[0, 0]], max(N0, N2), N0, N2, N_neg=np.floor(len(pip))
            )
        )

        dist_pin = np.linalg.norm(base[pin[:, 0]] - pos[pin[:, 1]], 2, 1)
        dist_nin = np.linalg.norm(base[nin[:, 0]] - neg[nin[:, 1]], 2, 1)

        return pip, pin[dist_pin > 0.1], nin[dist_nin > 0.1]

    def _getlabel(self, idx):
        return 0

    def _getpc(self, idx):
        return load_norm_pc(self.pcs[idx], 10000)

    def _getidx(self, index):
        return index

    def quant(self, rot_coords, coords):
        unique_idx = ME.utils.sparse_quantize(
            np.floor(rot_coords / self.voxel_size), return_index=True
        )
        rot_coords = rot_coords[unique_idx, :]
        coords = coords[unique_idx, :]
        rot_coords_grid = np.floor(rot_coords / self.voxel_size)

        return rot_coords, rot_coords_grid, coords

    def __getitem__(self, index):
        idx = self._getidx(index)
        label = self._getlabel(idx)

        pos_inst_pos_pairs, pos_inst_neg_pairs, neg_inst_neg_pairs = None, None, None
        # while(not isinstance(pos_inst_pos_pairs, np.ndarray) or not isinstance(pos_inst_neg_pairs, np.ndarray) or not isinstance(neg_inst_neg_pairs, np.ndarray)):

        positive_idx = self.generate_positive_inst(index)
        negative_idx = self.generate_negative_inst(index)

        base_coords = self._getpc(idx)
        pos_coords = [self._getpc(i) for i in positive_idx]
        neg_coords = [self._getpc(i) for i in negative_idx]

        rot_pos_coords = []
        pos_trans = []
        rot_neg_coords = []
        neg_trans = []

        rot_base_coords, base_trans = random_rotation(base_coords)
        for coords in pos_coords:
            rot_coords, trans = random_rotation(coords)
            rot_pos_coords.append(rot_coords)
            pos_trans.append(trans)
        for coords in neg_coords:
            rot_coords, trans = random_rotation(coords)
            rot_neg_coords.append(rot_coords)
            neg_trans.append(trans)

        rot_pos_coords_grid = []
        rot_neg_coords_grid = []
        rot_base_coords, rot_base_coords_grid, base_coords = self.quant(
            rot_base_coords, base_coords
        )
        for i in range(len(pos_coords)):
            rot_coords, rot_coords_grid, coords = self.quant(
                rot_pos_coords[i], pos_coords[i]
            )
            rot_pos_coords[i] = rot_coords
            pos_coords[i] = coords
            rot_pos_coords_grid.append(rot_coords_grid)

        for i in range(len(neg_coords)):
            rot_coords, rot_coords_grid, coords = self.quant(
                rot_neg_coords[i], neg_coords[i]
            )
            rot_neg_coords[i] = rot_coords
            neg_coords[i] = coords
            rot_neg_coords_grid.append(rot_coords_grid)

        # generate positive and negative pairs

        # for i in range(len(pos_coords)):
        #    pos_inst_pos_pairs, pos_inst_neg_pairs, neg_inst_neg_pairs = self.generate_local_pair(
        #                                                                    base_coords, pos_coords[i], neg_coords[i])

        base_feat = np.ones([len(rot_base_coords), 1])
        pos_feat = [
            np.ones([len(rot_pos_coords[i]), 1]) for i in range(len(rot_pos_coords))
        ]
        neg_feat = [
            np.ones([len(rot_neg_coords[i]), 1]) for i in range(len(rot_neg_coords))
        ]

        # tag: data output
        base = {
            "coord": rot_base_coords_grid,
            "origin": rot_base_coords,
            "feat": base_feat,
            "T": base_trans,
            "idx": idx,
        }
        pos = [
            {
                "coord": rot_pos_coords_grid[i],
                "origin": rot_pos_coords[i],
                "feat": pos_feat[i],
                "T": pos_trans[i],
                "idx": positive_idx[i],
            }
            for i in range(len(rot_pos_coords))
        ]
        neg = [
            {
                "coord": rot_neg_coords_grid[i],
                "origin": rot_neg_coords[i],
                "feat": neg_feat[i],
                "T": neg_trans[i],
                "idx": negative_idx[i],
            }
            for i in range(len(rot_neg_coords))
        ]

        return (
            base,
            pos,
            neg,
        )  # pos_inst_pos_pairs, pos_inst_neg_pairs, neg_inst_neg_pairs, label

    def __len__(
        self,
    ):
        return len(self.pcs)

    def collate_pair_fn(self, list_data):
        # print(type(list_data))
        # print(len(list_data))

        base_dict, pos_dict, neg_dict = list(zip(*list_data))

        base_coords = []
        pos_coords = []
        neg_coords = []
        base_feat = []
        pos_feat = []
        neg_feat = []
        base_idx = []
        pos_idx = []
        neg_idx = []

        for idx in range(len(base_dict)):

            base_coords.append(torch.from_numpy(base_dict[idx]["coord"]))

            for pos_i in range(len(pos_dict[idx])):
                pos_coords.append(torch.from_numpy(pos_dict[idx][pos_i]["coord"]))
            for neg_i in range(len(neg_dict[idx])):
                neg_coords.append(torch.from_numpy(neg_dict[idx][neg_i]["coord"]))

            base_feat.append(torch.from_numpy(base_dict[idx]["feat"]))
            for pos_i in range(len(pos_dict[idx])):
                pos_feat.append(torch.from_numpy(pos_dict[idx][pos_i]["feat"]))
            for neg_i in range(len(neg_dict[idx])):
                neg_feat.append(torch.from_numpy(neg_dict[idx][neg_i]["feat"]))
            base_idx.append(base_dict[idx]["idx"])
            for pos_i in range(len(pos_dict[idx])):
                pos_idx.append(pos_dict[idx][pos_i]["idx"])
            for neg_i in range(len(neg_dict[idx])):
                neg_idx.append(neg_dict[idx][neg_i]["idx"])

        batch_base_coords, batch_base_feat = ME.utils.sparse_collate(
            base_coords, base_feat
        )
        batch_pos_coords, batch_pos_feat = ME.utils.sparse_collate(pos_coords, pos_feat)
        batch_neg_coords, batch_neg_feat = ME.utils.sparse_collate(neg_coords, neg_feat)

        data = {}

        data["base_coords"] = batch_base_coords.int()
        data["base_feat"] = batch_base_feat.float()
        data["pos_coords"] = batch_pos_coords.int()
        data["pos_feat"] = batch_pos_feat.float()
        data["neg_coords"] = batch_neg_coords.int()
        data["neg_feat"] = batch_neg_feat.float()

        data["base_idx"] = torch.tensor(base_idx)
        data["pos_idx"] = torch.tensor(pos_idx)
        data["neg_idx"] = torch.tensor(neg_idx)

        return data


if __name__ == "__main__":
    root = "/zty-vol/data/ShapeNetCore.v2.PC15k"
    catid = "03001627"

    H = HardDataset(root, "train", catid, "/scannet/tables", 0.1, 0.5, True)

    train_loader = torch.utils.data.DataLoader(
        H, batch_size=4, shuffle=False, num_workers=4, collate_fn=H.collate_pair_fn
    )

    for data in train_loader:
        print(data.keys())
