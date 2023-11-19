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


class ChairDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, pos_ratio, neg_ratio, voxel_size):

        self.root = root
        self.file_list, self.lable2data, self.data2label = read_label(
            "./config/label_{}".format(split)
        )

        # self.int_label = self.to_int_label(self.lable2data)
        self.table = np.load("./config/max_dist_{}.npy".format(split))
        self.voxel_size = voxel_size
        print(voxel_size)
        self.pcs = []

        for file_path in self.file_list:
            self.pcs.append(os.path.join(self.root, file_path))

        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio
        self.pos_n = int(self.__len__() * pos_ratio)
        self.neg_n = int(self.__len__() * neg_ratio)

    def to_int_label(self, LabelDict):
        IntLabel = {}
        i = 0
        for catid in LabelDict.keys():
            for subcatid in LabelDict[catid].keys():
                IntLabel[subcatid] = i
                i += 1
        return IntLabel

    def generate_positive_inst(self, idx):

        topn = self.pos_n
        # rank the chamfer dist
        dist_rank = np.argsort(self.table[idx, :])
        # only chamfer(i,j)<0.15 is valid positive
        # valid = (self.table[idx, :]<0.15).nonzero()[0]
        #
        # topn = max(min(topn, len(valid)), 1)
        # ramdom sample rate
        # prob = 2*(np.arange(topn)+1)/((1+topn)*topn)
        # prob = np.flip(prob)
        # select_idx = np.random.choice(np.arange(topn), p=prob)
        select_idx = np.random.choice(np.arange(topn))

        return dist_rank[select_idx]

    def generate_negative_inst(self, idx):
        # diagnal of table is 200
        topn = self.neg_n
        dist_rank = np.argsort(-self.table[idx, :])
        # valid = (self.table[idx, :]>0.2).nonzero()[0]
        # topn = max(min(topn, len(valid)-1), 1)
        # prob = 2*(np.arange(topn)+1)/((1+topn)*topn)
        # prob = np.flip(prob)
        # select_idx = np.random.choice(np.arange(topn), p=prob)+1
        select_idx = np.random.choice(np.arange(topn)) + 1

        return dist_rank[select_idx]

    def generate_local_pair(self, base, pos, neg):
        """
        base, pos, neg: np array
        """
        # TODO: merge positive and negative
        N0, N1, N2 = base.shape[0], pos.shape[0], neg.shape[0]
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

    def __len__(
        self,
    ):
        return len(self.pcs)

    def collate_pair_fn(self, list_data):

        (
            base_dict,
            pos_dict,
            neg_dict,
            pos_inst_pos_pairs,
            pos_inst_neg_pairs,
            neg_inst_neg_pairs,
        ) = list(zip(*list_data))

        PiP = []
        PiN = []
        NiN = []

        base_coords = []
        pos_coords = []
        neg_coords = []
        base_feat = []
        pos_feat = []
        neg_feat = []
        base_T = []
        pos_T = []
        neg_T = []
        base_origin = []
        pos_origin = []
        neg_origin = []
        base_idx = []
        pos_idx = []
        neg_idx = []
        base_sym = []
        pos_sym = []
        neg_sym = []

        pos_curr_idx = np.zeros([1, 2])
        neg_curr_idx = np.zeros([1, 2])

        for idx in range(len(base_dict)):
            Nbase = base_dict[idx]["coord"].shape[0]
            Npos = pos_dict[idx]["coord"].shape[0]
            Nneg = neg_dict[idx]["coord"].shape[0]

            PiP.append(
                torch.from_numpy(np.array(pos_inst_pos_pairs[idx]) + pos_curr_idx)
            )
            PiN.append(
                torch.from_numpy(np.array(pos_inst_neg_pairs[idx]) + pos_curr_idx)
            )
            NiN.append(
                torch.from_numpy(np.array(neg_inst_neg_pairs[idx]) + neg_curr_idx)
            )

            pos_curr_idx += np.array([Nbase, Npos])
            neg_curr_idx += np.array([Nbase, Nneg])

            base_coords.append(torch.from_numpy(base_dict[idx]["coord"]))
            pos_coords.append(torch.from_numpy(pos_dict[idx]["coord"]))
            neg_coords.append(torch.from_numpy(neg_dict[idx]["coord"]))
            base_origin.append(torch.from_numpy(base_dict[idx]["origin"]))
            pos_origin.append(torch.from_numpy(pos_dict[idx]["origin"]))
            neg_origin.append(torch.from_numpy(neg_dict[idx]["origin"]))
            base_feat.append(torch.from_numpy(base_dict[idx]["feat"]))
            pos_feat.append(torch.from_numpy(pos_dict[idx]["feat"]))
            neg_feat.append(torch.from_numpy(neg_dict[idx]["feat"]))
            base_T.append(torch.from_numpy(base_dict[idx]["T"]))
            pos_T.append(torch.from_numpy(pos_dict[idx]["T"]))
            neg_T.append(torch.from_numpy(neg_dict[idx]["T"]))
            base_idx.append(base_dict[idx]["idx"])
            pos_idx.append(pos_dict[idx]["idx"])
            neg_idx.append(neg_dict[idx]["idx"])
            base_sym.append(base_dict[idx]["sym"])
            pos_sym.append(pos_dict[idx]["sym"])
            neg_sym.append(neg_dict[idx]["sym"])

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

        data["PiP_pairs"] = torch.cat(PiP, 0).int()
        data["PiN_pairs"] = torch.cat(PiN, 0).int()
        data["NiN_pairs"] = torch.cat(NiN, 0).int()
        data["base_origin"] = torch.cat(base_origin, 0).float()
        data["pos_origin"] = torch.cat(pos_origin, 0).float()
        data["neg_origin"] = torch.cat(neg_origin, 0).float()

        data["base_T"] = torch.stack(base_T, 0).float()
        data["pos_T"] = torch.stack(pos_T, 0).float()
        data["neg_T"] = torch.stack(neg_T, 0).float()
        data["base_idx"] = torch.Tensor(base_idx).int()
        data["pos_idx"] = torch.Tensor(pos_idx).int()
        data["neg_idx"] = torch.Tensor(neg_idx).int()

        data["base_sym"] = torch.Tensor(base_sym).int()
        data["pos_sym"] = torch.Tensor(pos_sym).int()
        data["neg_sym"] = torch.Tensor(neg_sym).int()

        return data
