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


class ScannetDataset(CategoryDataset):
    """
    For finetuning FCGF on Scannet.
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
        # Modify the distance representation. ground truth is included in the top match
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
            readerloader = torch.utils.data.DataLoader(reader, batch_size=1, shuffle=False, num_workers=8)

            for data in tqdm(readerloader):
                self.pcs.append(data[0, :, :].numpy())
        else:
            self.pcs = [os.path.join(self.scan_root, file) for file in self.files]

        script_dir = os.path.dirname(__file__)
        sym_label_file = os.path.join(script_dir, "../configs", f"{self.catid}_scan2cad_rot_sym_label.txt")
        if os.path.exists(sym_label_file):
            with open(sym_label_file, "r") as f:
                lines = f.readlines()
                # names = [line.strip("\n").split(" ")[0] for line in lines]
                sym_ref = [int(line.strip("\n").split(" ")[1]) for line in lines]
        else:
            sym_ref = None

        self.sym_label = sym_ref

        self.rank_a = np.argsort(self.table, 1)
        self.rank_d = np.argsort(-1 * self.table, 1)

        self.pos_n = int(len(self.CADLib.CadPcs) * pos_ratio)
        self.neg_n = int(len(self.CADLib.CadPcs) * neg_ratio)

        self.fix_trans = np.load(os.path.join(script_dir, "../configs/fix_trans.npy"))

    def _get_sym(self, idx):
        """
        Get symmetry label
        """
        if self.sym_label is None:
            return 1
        else:
            return self.sym_label[idx]

    def _getscan(self, idx):
        """
        Get segmentated scanned object
        """
        if self.preload:
            return self.pcs[idx]
        else:
            return load_raw_pc(self.pcs[idx], 10000)

    def _getcad(self, idx):
        """
        Get corresponding CAD model
        """
        if self.preload:
            return self.CADLib.CadPcs[idx]
        else:
            return load_raw_pc(self.CADLib.CadPcs[idx], 10000)

    def generate_positive_inst(self, scanidx):
        """
        Positive CAD models
        """
        if self.pos_n > 0:
            cadidx = self.id2idx[self.BestMatches[scanidx]]
            topn = self.pos_n
            dist_rank = np.argsort(self.table[cadidx, :])
            valid = (self.table[cadidx, :] < 0.15).nonzero()[0]
            topn = max(min(topn, len(valid)), 1)
            prob = 2 * (np.arange(topn) + 1) / ((1 + topn) * topn)
            prob = np.flip(prob)
            select_idx = np.random.choice(np.arange(topn), p=prob)
            return dist_rank[select_idx]
        else:
            return self.id2idx[self.BestMatches[scanidx]]

    def generate_negative_inst(self, scanidx):
        """
        Negative CAD models
        """
        cadidx = self.id2idx[self.BestMatches[scanidx]]
        topn = self.neg_n
        dist_rank = np.argsort(-self.table[cadidx, :])
        valid = (self.table[cadidx, :] > 0.2).nonzero()[0]
        topn = max(min(topn, len(valid) - 1), 1)
        prob = 2 * (np.arange(topn) + 1) / ((1 + topn) * topn)
        prob = np.flip(prob)
        select_idx = np.random.choice(np.arange(topn), p=prob)

        return dist_rank[select_idx]

    """
    def generate_local_pair(self, base, pos, neg, sample=1024):

        # Generate local matching pairs.

        N0, N1, N2 = base.shape[0], pos.shape[0], neg.shape[0]
        pip = np.array(get_matching_indices(base, pos, 0.03))

        if pip.shape[0] < 0.1*min(N0, N1):
            return None, None, None
        pin = np.array(generate_rand_negative_pairs(pip, max(N0, N1), N0, N1, N_neg=np.floor(len(pip))))
        nin = np.array(generate_rand_negative_pairs([[0, 0]], max(N0, N2), N0, N2, N_neg=np.floor(len(pip))))

        dist_pin = np.linalg.norm(base[pin[:, 0]]-pos[pin[:, 1]], 2, 1)
        dist_nin = np.linalg.norm(base[nin[:, 0]]-neg[nin[:, 1]], 2, 1)

        pin = pin[dist_pin>0.1]
        nin = nin[dist_nin>0.1]

        np.random.shuffle(pip)
        np.random.shuffle(pin)
        np.random.shuffle(nin)

        return pip[:sample, :], pin[:sample, :], nin[:sample, :]
        """

    def __getitem__(self, idx):
        label = self._getlabel(idx)

        failed_count = 0

        pos_inst_pos_pairs, pos_inst_neg_pairs, neg_inst_neg_pairs = None, None, None

        while pos_inst_pos_pairs is None:
            # raise error after failed over 100 times
            if failed_count >= 100:
                self.badscans.append(self.files[idx])
                print("Low inlier rate idx: {} path: {} for over 100 times".format(idx, self.files[idx]))
                while self.files[idx] in self.badscans:
                    idx = random.randint(0, len(self.files) - 1)
                failed_count = 0
                # raise ValueError("Low in lier rate idx: {} path: {} for over 100 times".format(idx, self.files[idx]))

            # sample valid positive cad model
            positive_idx = self.generate_positive_inst(idx)
            negative_idx = self.generate_negative_inst(idx)

            base_coords = self._getscan(idx)
            pos_coords = self._getcad(positive_idx)
            neg_coords = self._getcad(negative_idx)

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

            base_coords = apply_transform(base_coords, np.matmul(np.linalg.inv(T_pos), T_base))

            # import open3d as o3d
            # base_pcd = o3d.geometry.PointCloud()
            # base_pcd.points = o3d.utility.Vector3dVector(base_coords)
            # base_pcd.paint_uniform_color([1, 0, 0])
            # pos_pcd = o3d.geometry.PointCloud()
            # pos_pcd.points = o3d.utility.Vector3dVector(pos_coords)
            # pos_pcd.paint_uniform_color([0, 1, 0])
            # o3d.visualization.draw_geometries([base_pcd, pos_pcd])

            # normalize. Scan normalize with the positive CAD model
            t = pos_coords.mean(0)
            base_coords -= t
            neg_coords -= t
            pos_coords -= t
            # TODO: -t before, but the alignment is bad, double check!
            # base_coords -= base_coords.mean(0)
            # neg_coords -= neg_coords.mean(0)
            # pos_coords -= pos_coords.mean(0)

            # import open3d as o3d
            # base_pcd = o3d.geometry.PointCloud()
            # base_pcd.points = o3d.utility.Vector3dVector(base_coords)
            # base_pcd.paint_uniform_color([1, 0, 0])
            # pos_pcd = o3d.geometry.PointCloud()
            # pos_pcd.points = o3d.utility.Vector3dVector(pos_coords)
            # pos_pcd.paint_uniform_color([0, 1, 0])
            # o3d.visualization.draw_geometries([base_pcd, pos_pcd])

            r = np.max(np.linalg.norm(pos_coords, 2, 1))

            base_coords = base_coords / r
            neg_coords = neg_coords / r
            pos_coords = pos_coords / r

            pos_sym = self._get_sym(positive_idx)

            if self.split == "train":
                rot_base_coords, base_trans = random_rotation(base_coords)
                rot_pos_coords, pos_trans = random_rotation(pos_coords)
                rot_neg_coords, neg_trans = random_rotation(neg_coords)
            else:
                base_trans = self.fix_trans[idx, 0, :, :]
                pos_trans = self.fix_trans[idx, 1, :, :]
                neg_trans = self.fix_trans[idx, 2, :, :]

                rot_base_coords = apply_transform(base_coords, base_trans)
                rot_pos_coords = apply_transform(pos_coords, pos_trans)
                rot_neg_coords = apply_transform(neg_coords, neg_trans)

            rot_base_coords, rot_base_coords_grid, base_coords = self.quant(rot_base_coords, base_coords)
            rot_pos_coords, rot_pos_coords_grid, pos_coords = self.quant(rot_pos_coords, pos_coords)
            rot_neg_coords, rot_neg_coords_grid, neg_coords = self.quant(rot_neg_coords, neg_coords)

            # generate positive and negative pairs
            (
                pos_inst_pos_pairs,
                pos_inst_neg_pairs,
                neg_inst_neg_pairs,
            ) = self.generate_local_pair(base_coords, pos_coords, neg_coords)
            failed_count += 1

        base_feat = np.ones([len(rot_base_coords), 1])
        pos_feat = np.ones([len(rot_pos_coords), 1])
        neg_feat = np.ones([len(rot_neg_coords), 1])

        base = {
            "coord": rot_base_coords_grid,
            "origin": rot_base_coords,
            "feat": base_feat,
            "T": base_trans,
            "idx": idx,
            "sym": 1,
        }
        pos = {
            "coord": rot_pos_coords_grid,
            "origin": rot_pos_coords,
            "feat": pos_feat,
            "T": pos_trans,
            "idx": self.id2idx[self.BestMatches[idx]],
            "sym": pos_sym,
        }
        neg = {
            "coord": rot_neg_coords_grid,
            "origin": rot_neg_coords,
            "feat": neg_feat,
            "T": neg_trans,
            "idx": negative_idx,
            "sym": 1,
        }

        return (
            base,
            pos,
            neg,
            pos_inst_pos_pairs,
            pos_inst_neg_pairs,
            neg_inst_neg_pairs,
        )
