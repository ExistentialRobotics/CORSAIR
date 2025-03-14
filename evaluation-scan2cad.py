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
from utils.Info.CADLib import CustomizeCADLib, GaussianSplatLib
from utils.Info.Scan2cadInfo import Scan2cadInfo
from utils.eval_pose import eval_pose
from utils.logger import Logger as logger
from utils.retrieval import scan2cad_retrieval_eval
from utils.symmetry import sym_pose
from utils.visualization import embed_tsne
from utils.visualization import get_color_map
from utils.preprocess import load_norm_pc, apply_transform, load_raw_pc, chamfer_kdtree_1direction
import open3d as o3d
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm.contrib.concurrent import thread_map

@dataclass
class Config:
    shapenet_pc15k_root: str
    scan2cad_pc_root: str
    scan2cad_annotation_root: str
    shapenet_radegs_root: str
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
    use_best: int = 30

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
            "--shapenet-radegs-root",
            type=str,
            default="/mnt/data/RaDe-GS",
            help="Path to RaDe-GS ply files for ShapeNet",
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
        parser.add_argument(
            "--use-best",
            type=int,
            default=30,
            help="Number of closest objects to retrieve"
        )

        args = parser.parse_args()
        self.config = Config(**vars(args))
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

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
            table_path=os.path.join(script_dir, "configs", f"{self.config.catid}_scan2cad.npy"),
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

        self.gsplat_lib = GaussianSplatLib(
            self.config.shapenet_radegs_root,
            self.config.catid
        )

        self.cad_lib_loader = DataLoader(
            self.cad_lib,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=self.cad_lib.collate_pair_fn,
        )

        self.scan2cad_loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=self.dataset.collate_pair_fn,
        )

        with open(os.path.join(script_dir, "configs", f"{self.config.catid}_scan2cad_rot_sym_label.txt"), "r") as f:
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

        # self.CADLib_idx2id = {v: k for k, v in self.cad_lib.id2idx.items()}

        # fixed_transform_gsplat = np.array([
        #     [1, 0, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, -1, 0, 0],
        #     [0, 0, 0, 1]
        # ])

        self.chamfer_dist_list = []
        chamfer_dist_cache_pd = pd.read_csv(
            os.path.join(script_dir, "configs", f"chamfer_dist_list.csv")
        )
        self._chamfer_dist_cache = chamfer_dist_cache_pd['chamfer_dist'].to_numpy().reshape(
            (len(self.scan2cad_info.UsedObjId), len(self.scan2cad_info.UsedObjId))
        )
        self.best_matches_idx = np.fromiter(
            map(lambda obj_id: self.cad_lib.id2idx[obj_id], self.dataset.BestMatches), 
            dtype=np.int64
        )

        self.feature_dist = cdist(self.descriptors, self.lib_descriptors)
        self.topN_idx = np.argsort(self.feature_dist, axis=-1)
        self.retrieved_object_idx = np.array([
            self.topN_idx[scene_idx, np.argmin(self._chamfer_dist_cache[
                self.best_matches_idx[scene_idx],
                self.topN_idx[scene_idx, :self.config.use_best].flatten()
            ])] for scene_idx in range(len(self.best_matches_idx))
        ])
        # self.retrieved_object_idx_orig = np.argmin(self.feature_dist, axis=-1)

        self.chamfer_dist_list = thread_map(
            self.evaluate_retrieval, 
            zip(self.dataset.BestMatches, self.retrieved_object_idx.tolist()),
            max_workers=16
        )
        
        self.logger.log(f"average chamfer distance (GT CAD vs RaDe-GS reconstructed PCD): {np.mean(self.chamfer_dist_list)}")

    def evaluate_retrieval(self, arg):
        # Inputs are ground truth model ID and retrieved model IDs
        ground_truth_id, retrieved_model_idx = arg
        retrieved_model_id = self.cad_lib.ids[retrieved_model_idx]

        # Get the corresponding point clouds
        align_cad_xyz = self.cad_lib._getpc_raw_id(
            ground_truth_id
        )
        retrieved_gsplat_xyz = self.gsplat_lib.get_recon_pc_by_id_transformed(
            retrieved_model_id
        )

        # Compute the Chamfer distance using a KDtree.
        # In this implementation, a KDtree of the source point cloud is formed, 
        # and is used to yield the closest distance(s) to the target pointcloud.
        chamfer_dist = chamfer_kdtree_1direction(align_cad_xyz, retrieved_gsplat_xyz) + \
                        chamfer_kdtree_1direction(retrieved_gsplat_xyz, align_cad_xyz)

        return chamfer_dist


if __name__ == "__main__":
    app = App()

    with open("results", "w") as f:
        for file, chamfer_dist, best_match, retrieved in zip(
            app.dataset.files,
            app.chamfer_dist_list,
            app.best_matches_idx,
            app.retrieved_object_idx
        ):
            f.write(f"{file},{chamfer_dist},{best_match},{retrieved}\n")


    # dist_to_best = app._chamfer_dist_cache[best_matches_idx, app.ret]

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.scatter(dist_to_best.flatten(), feature_dist.flatten())
    # fig.savefig('scatter.png')
