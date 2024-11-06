import argparse
import os
from dataclasses import dataclass

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import vedo
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets.ScannetDataset import ScannetDataset
from datasets.ScannetTestTimeDataset import ScannetTestTimeDataset
from model import fc
from model import load_model
from utils.Info.CADLib import CustomizeCADLib
from utils.Info.Scan2cadInfo import Scan2cadInfo
from utils.eval_pose import eval_pose
from utils.logger import logger
from utils.retrieval import scan2cad_retrieval_eval
from utils.symmetry import sym_pose
from utils.visualization import embed_tsne
from utils.visualization import get_color_map
from utils.preprocess import load_norm_pc, normalize_pc, apply_transform, load_raw_pc, chamfer_kdtree_1direction
import open3d as o3d

@dataclass
class Config:
    shapenet_pc15k_root: str
    scan2cad_pc_root: str
    scan2cad_annotation_root: str
    category: str
    checkpoint: str
    catid: str = None
    voxel_size: float = 0.03
    k_nn: int = 5
    max_corr: float = 0.2
    distance: str = "l2"
    random_seed: int = 31
    cache_dir: str = ""
    register_top1: bool = True
    device: str = "cuda"
    ignore_cache: bool = False

    def __post_init__(self):
        if self.category == "table":
            self.catid = "04379243"
        elif self.category == "chair":
            self.catid = "03001627"
        else:
            raise ValueError("Invalid category")

class App:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        parser = argparse.ArgumentParser(description="Evaluate CORSAIR")
        parser.add_argument(
            "--shapenet-pc15k-root",
            type=str,
            default="/mnt/data/ShapeNetCore.v2.PC15k",
            help="Path to ShapeNetCore.v2.PC15k",
        )
        parser.add_argument(
            "--scan2cad-pc-root",
            type=str,
            default="/mnt/data/Scan2CAD_pc",
            help="Path to Scan2CAD",
        )
        parser.add_argument(
            "--scan2cad-annotation-root",
            type=str,
            default="/mnt/data/Scan2CAD_annotations",
            help="Path to Scan2CAD annotations",
        )
        parser.add_argument(
            "--category",
            type=str,
            default="table",
            choices=["table", "chair"],
            help="Category to evaluate",
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=os.path.join(script_dir, "ckpts", f"scannet_ret_table_best"),
            help="Path to the checkpoint",
        )
        parser.add_argument(
            "--cache-dir",
            type=str,
            default=os.path.join(script_dir, "data"),
            help="Path to load / save the result of registration.",
        )
        parser.add_argument(
            "--register-gt",
            action="store_false",
            dest="register_top1",
            help="Registering gt CAD model",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cuda", "cpu"],
            help="Device to use for evaluation",
        )
        parser.add_argument(
            "--ignore-cache",
            action="store_true",
            help="Ignore cached results",
        )

        args = parser.parse_args()
        self.config = Config(**vars(args))

        self.logger = logger("./logs", "evaluation_scan2cad_gsplat_recon.txt")
        self.logger.log(f"category: {self.config.category}")

        self.scan2cad_info = Scan2cadInfo(
            cad_root=self.config.shapenet_pc15k_root,
            scan_root=self.config.scan2cad_pc_root,
            catid=self.config.catid,
            annotation_dir=self.config.scan2cad_annotation_root,
        )

        self.cad_lib = CustomizeCADLib(
            root=self.config.shapenet_pc15k_root,
            catid=self.config.catid,
            ids=self.scan2cad_info.UsedObjId,
            table_path=os.path.join(script_dir, "config", f"{self.config.catid}_scan2cad.npy"),
            voxel_size=self.config.voxel_size,
            preload=False,
        )

        self.dataset = ScannetDataset(
            scan_root=self.config.scan2cad_pc_root,
            cad_root=self.config.shapenet_pc15k_root,
            CADLib=self.cad_lib,
            Scan2CadInfo=self.scan2cad_info,
            split="test",
            catid=self.config.catid,
            pos_ratio=0.1,
            neg_ratio=0.5,
            voxel_size=self.config.voxel_size,
            preload=False,
        )

        self.dataset.pos_n = 1  # ask the dataset to load the ground truth best match

        self.cad_lib_loader = DataLoader(
            self.cad_lib,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=self.cad_lib.collate_pair_fn,
        )

        self.scan2cad_loader = DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            collate_fn=self.dataset.collate_pair_fn,
        )

        with open(os.path.join(script_dir, "config", f"{self.config.catid}_scan2cad_rot_sym_label.txt"), "r") as f:
            lines = f.readlines()
            # names = [line.strip("\n").split(" ")[0] for line in lines]
            self.sym_label = [int(line.strip("\n").split(" ")[1]) for line in lines]
        
        # feature extraction network for registration
        model = load_model("ResUNetBN2C")(
            in_channels=1,
            out_channels=16,
            bn_momentum=0.05,
            normalize_feature=True,
            conv1_kernel_size=3,
            D=3,
        )
        model = model.to(self.config.device)

        # embedding network for retrieval
        embedding = fc.conv1_max_embedding(1024, 512, 256).to(self.config.device)

        # load weights
        checkpoint = torch.load(self.config.checkpoint, map_location=self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        embedding.load_state_dict(checkpoint["embedding_state_dict"])
        self.logger.log(f"Checkpoint epoch: {checkpoint['epoch']}")

        model.eval()
        embedding.eval()

        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)

        # Extract Retrieval Features for CAD models
        self.lib_feats = []
        self.lib_outputs = []
        self.lib_Ts = []
        self.lib_origins = []
        self.logger.log("Updating global feature in the CAD library")
        with torch.no_grad():
            for data in tqdm(self.cad_lib_loader, ncols=80):
                base_input = ME.SparseTensor(
                    data["base_feat"].to(self.config.device),
                    data["base_coords"].to(self.config.device),
                )

                self.lib_Ts.append(data["base_T"])
                batch_size = len(data["base_idx"])

                base_output, base_feat = model(base_input)
                base_feat = embedding(base_feat)

                for i in range(batch_size):
                    base_mask = base_output.C[:, 0] == i
                    self.lib_outputs.append(base_output.F[base_mask, :])
                    self.lib_origins.append(data["base_origin"].to(self.config.device)[base_mask, :])

                base_feat_norm = nn.functional.normalize(base_feat, dim=1)
                self.lib_feats.append(base_feat_norm)
        self.lib_feats = torch.cat(self.lib_feats, dim=0)
        self.lib_Ts = torch.cat(self.lib_Ts, dim=0)

        # Extract Retrieval Features for objects
        self.base_outputs = []
        self.base_origins = []
        self.base_feats = []
        self.base_Ts = []
        self.best_match_idx = []
        self.best_match_syms = []
        self.logger.log("Updating global feature in the Scan2CAD dataset")
        with torch.no_grad():
            for data in tqdm(self.scan2cad_loader, ncols=80):
                base_input = ME.SparseTensor(
                    data["base_feat"].to(self.config.device),
                    data["base_coords"].to(self.config.device),
                )

                self.base_Ts.append(data["base_T"])
                self.best_match_idx.append(data["pos_idx"])
                self.best_match_syms.append(data["pos_sym"])
                batch_size = len(data["pos_idx"])

                base_output, base_feat = model(base_input)
                base_feat = embedding(base_feat)

                for i in range(batch_size):
                    base_mask = base_output.C[:, 0] == i
                    self.base_outputs.append(base_output.F[base_mask, :])
                    self.base_origins.append(data["base_origin"].to("cuda")[base_mask, :])

                base_feat_norm = nn.functional.normalize(base_feat, dim=1)
                self.base_feats.append(base_feat_norm)
        self.base_feats = torch.cat(self.base_feats, dim=0)
        self.base_Ts = torch.cat(self.base_Ts, dim=0)
        self.best_match_idx = torch.cat(self.best_match_idx, dim=0)
        self.best_match_syms = torch.cat(self.best_match_syms, dim=0)

        # Evaluate Retrieval Results
        self.descriptors = self.base_feats.detach().cpu().numpy()
        self.lib_descriptors = self.lib_feats.detach().cpu().numpy()
        self.best_match_idx = self.best_match_idx.detach().cpu().numpy()
        self.stat = scan2cad_retrieval_eval(
            self.descriptors,
            self.lib_descriptors,
            self.best_match_idx,
            self.dataset.table,
            int(0.1 * self.dataset.table.shape[1]),  # Precision@M=0.1n
        )
        self.logger.log(f"top1_error: {self.stat['top1_error']}")
        self.logger.log(f"precision: {self.stat['precision']}")

        self.CADLib_idx2id = {v: k for k, v in self.cad_lib.id2idx.items()}

        fixed_transform_gsplat = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])

        chamfer_dist_list = []

        for i in tqdm(range(len(self.base_origins)), ncols=80):
            tqdm.write(f"Processing {i}th query")

            # grount truth point cloud
            align_cad_xyz = load_raw_pc(self.cad_lib.CadPcs[self.cad_lib.id2idx[self.dataset.BestMatches[i]]], 15000)

            # get retrieved object ID
            retrieved_idx = self.stat['top1_predict'][i]
            retrieved_modelID = self.CADLib_idx2id[retrieved_idx]

            # get corresponding Gaussian splat and reconstruct point cloud
            gsplat_root = "/mnt/data/RaDe-GS"
            gsplat_recon_path = os.path.join(gsplat_root, self.config.catid, retrieved_modelID, "recon.ply")
            retrieved_gsplat_recon_mesh = o3d.io.read_triangle_mesh(gsplat_recon_path)
            retrieved_gsplat_recon_pcd = retrieved_gsplat_recon_mesh.sample_points_uniformly(number_of_points=15000)
            retrieved_gsplat_xyz = apply_transform(np.asarray(retrieved_gsplat_recon_pcd.points), fixed_transform_gsplat)

            # compute Chamfer distance
            chamfer_dist = chamfer_kdtree_1direction(align_cad_xyz, retrieved_gsplat_xyz) + \
                            chamfer_kdtree_1direction(retrieved_gsplat_xyz, align_cad_xyz)
            tqdm.write(f"CD: {chamfer_dist}")
            chamfer_dist_list.append(chamfer_dist)
        
        self.logger.log(f"average chamfer distance (GT CAD vs RaDe-GS reconstructed PCD): {np.mean(chamfer_dist_list)}")

if __name__ == "__main__":
    App()
