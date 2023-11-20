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
from datasets.Reader import Reader


class CategoryDataset(ChairDataset):
    """
    Shapenet category dataset
    """

    def __init__(
        self, root, split, catid, dist_mat_root, pos_ratio, neg_ratio, voxel_size
    ):
        self.root = root
        self.split = split
        self.catid = catid
        self.dist_mat_root = dist_mat_root
        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio
        self.voxel_size = voxel_size

        script_dir = os.path.dirname(__file__)
        self.fix_trans = np.load(os.path.join(script_dir, "../configs/fix_trans.npy"))
        self.id2name, self.name2id = read_catname(os.path.join(script_dir, "../configs/CatName.txt"))

        dist_mat_ref = np.load(
            os.path.join(self.dist_mat_root, "{}_{}.npy".format(catid, split))
        )

        # Load data to memory with Reader
        reader = Reader(self.root, self.catid, self.split, 10000)
        readerloader = torch.utils.data.DataLoader(
            reader, batch_size=1, shuffle=False, num_workers=8
        )
        pcs_ref = []
        for data in tqdm(readerloader):
            pcs_ref.append(data[0, :, :].numpy())

        # Load symmetry label
        if os.path.exists(
            "./config/{}_{}_rot_sym_label.txt".format(self.catid, self.split)
        ):

            with open(
                "./config/{}_{}_rot_sym_label.txt".format(self.catid, self.split), "r"
            ) as f:

                lines = f.readlines()

                names = [line.strip("\n").split(" ")[0] for line in lines]
                sym_ref = [int(line.strip("\n").split(" ")[1]) for line in lines]
        else:
            sym_ref = None

        self.dist_mat, self.pcs, self.sym_label = self.filter_data(
            dist_mat_ref, pcs_ref, sym_ref
        )

        self.rank_a = np.argsort(self.dist_mat, 1)
        self.rank_d = np.argsort(-1 * self.dist_mat, 1)

        self.pos_n = int(self.__len__() * pos_ratio)
        self.neg_n = int(self.__len__() * neg_ratio)

    def filter_data(self, dist_mat_ref, pcs_ref, sym_ref, thres=0.15, num=3):
        """
        Remove objects that have less than num similar objects. CD < thres is considered similar.
        """
        while True:
            z = []
            for t in dist_mat_ref:
                z.append((t <= thres).sum())
            z = np.array(z)

            mask = (z >= num).nonzero()[0]

            if len(mask) == len(dist_mat_ref):
                break
            print(len(mask))

            dist_mat = dist_mat_ref[mask, :]
            dist_mat = dist_mat[:, mask]
            dist_mat_ref = dist_mat.copy()

            pcs = [pcs_ref[item] for item in mask]
            pcs_ref = pcs

            if not sym_ref is None:
                sym = [sym_ref[item] for item in mask]
                sym_ref = sym

        return dist_mat, pcs, sym_ref

    def generate_local_pair(self, base, pos, neg, sample=1024, radius=0.03):
        """
        Generate local matching pairs.
        """
        N0, N1, N2 = base.shape[0], pos.shape[0], neg.shape[0]
        pip = np.array(get_matching_indices(base, pos, radius))

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

        pin = pin[dist_pin > 0.1]
        nin = nin[dist_nin > 0.1]

        np.random.shuffle(pip)
        np.random.shuffle(pin)
        np.random.shuffle(nin)

        return pip[:sample, :], pin[:sample, :], nin[:sample, :]

    def generate_positive_inst(self, idx):
        """
        Generate positive (similar) objects from the dataset
        """
        topn = self.pos_n
        dist_rank = np.argsort(self.dist_mat[idx, :])
        valid = (self.dist_mat[idx, :] < 0.15).nonzero()[0]
        topn = max(min(topn, len(valid)), 1)
        prob = 2 * (np.arange(topn) + 1) / ((1 + topn) * topn)
        prob = np.flip(prob)
        select_idx = np.random.choice(np.arange(topn), p=prob)
        return dist_rank[select_idx]

    def generate_negative_inst(self, idx):
        """
        Generate negative (dissimilar) objects from the dataset
        """
        topn = self.neg_n
        dist_rank = np.argsort(-self.dist_mat[idx, :])
        valid = (self.dist_mat[idx, :] > 0.2).nonzero()[0]
        topn = max(min(topn, len(valid) - 1), 1)
        prob = 2 * (np.arange(topn) + 1) / ((1 + topn) * topn)
        prob = np.flip(prob)
        select_idx = np.random.choice(np.arange(topn), p=prob) + 1
        return dist_rank[select_idx]

    def quant(self, rot_coords, coords):
        """
        Quantize point clouds
        """
        if ME.__version__ >= "0.5.4":
            unique_idx = ME.utils.sparse_quantize(
                np.ascontiguousarray(np.floor(rot_coords / self.voxel_size)),
                return_index=True,
                return_maps_only=True,
            )
        else:
            unique_idx = ME.utils.sparse_quantize(
                np.floor(rot_coords / self.voxel_size), return_index=True
            )
        rot_coords = rot_coords[unique_idx, :]
        coords = coords[unique_idx, :]
        rot_coords_grid = np.floor(rot_coords / self.voxel_size)

        return rot_coords, rot_coords_grid, coords

    def _get_sym(self, idx):
        """
        Get symmetry label
        """
        if self.sym_label is None:
            return 1
        else:
            return self.sym_label[idx]

    def __getitem__(self, index):
        idx = self._getidx(index)

        # Try to find a valid positive object with enough matching pairs
        pos_inst_pos_pairs, pos_inst_neg_pairs, neg_inst_neg_pairs = None, None, None
        while (
            not isinstance(pos_inst_pos_pairs, np.ndarray)
            or not isinstance(pos_inst_neg_pairs, np.ndarray)
            or not isinstance(neg_inst_neg_pairs, np.ndarray)
        ):
            positive_idx = self.generate_positive_inst(index)
            negative_idx = self.generate_negative_inst(index)

            base_coords = self._getpc(idx)
            pos_coords = self._getpc(positive_idx)
            neg_coords = self._getpc(negative_idx)

            base_sym = self._get_sym(idx)
            pos_sym = self._get_sym(positive_idx)
            neg_sym = self._get_sym(negative_idx)

            if self.split == "train":
                rot_base_coords, base_trans = random_rotation(base_coords)
                rot_pos_coords, pos_trans = random_rotation(pos_coords)
                rot_neg_coords, neg_trans = random_rotation(neg_coords)
            else:
                # Use fixed rotation for evaluation
                base_trans = self.fix_trans[index, 0, :, :]
                pos_trans = self.fix_trans[index, 1, :, :]
                neg_trans = self.fix_trans[index, 2, :, :]

                rot_base_coords = apply_transform(base_coords, base_trans)
                rot_pos_coords = apply_transform(pos_coords, pos_trans)
                rot_neg_coords = apply_transform(neg_coords, neg_trans)

            rot_base_coords, rot_base_coords_grid, base_coords = self.quant(
                rot_base_coords, base_coords
            )
            rot_pos_coords, rot_pos_coords_grid, pos_coords = self.quant(
                rot_pos_coords, pos_coords
            )
            rot_neg_coords, rot_neg_coords_grid, neg_coords = self.quant(
                rot_neg_coords, neg_coords
            )

            # generate positive and negative matching pairs
            (
                pos_inst_pos_pairs,
                pos_inst_neg_pairs,
                neg_inst_neg_pairs,
            ) = self.generate_local_pair(base_coords, pos_coords, neg_coords)

        base_feat = np.ones([len(rot_base_coords), 1])
        pos_feat = np.ones([len(rot_pos_coords), 1])
        neg_feat = np.ones([len(rot_neg_coords), 1])

        base = {
            "coord": rot_base_coords_grid,
            "origin": rot_base_coords,
            "feat": base_feat,
            "T": base_trans,
            "idx": idx,
            "sym": base_sym,
        }
        pos = {
            "coord": rot_pos_coords_grid,
            "origin": rot_pos_coords,
            "feat": pos_feat,
            "T": pos_trans,
            "idx": positive_idx,
            "sym": pos_sym,
        }
        neg = {
            "coord": rot_neg_coords_grid,
            "origin": rot_neg_coords,
            "feat": neg_feat,
            "T": neg_trans,
            "idx": negative_idx,
            "sym": neg_sym,
        }

        return (
            base,
            pos,
            neg,
            pos_inst_pos_pairs,
            pos_inst_neg_pairs,
            neg_inst_neg_pairs,
        )

    def _getpc(self, idx):
        """
        Get point cloud
        """
        return self.pcs[idx]
