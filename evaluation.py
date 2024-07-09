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
from model import fc
from model import load_model
from utils.Info.CADLib import CustomizeCADLib
from utils.Info.Scan2cadInfo import Scan2cadInfo
from utils.eval_pose import eval_pose
from utils.logger import Logger
from utils.retrieval import scan2cad_retrieval_eval
from utils.symmetry import sym_pose
from utils.visualization import embed_tsne
from utils.visualization import get_color_map


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
            default=os.path.join(script_dir, "data", "ShapeNetCore.v2.PC15k"),
            help="Path to ShapeNetCore.v2.PC15k",
        )
        parser.add_argument(
            "--scan2cad-pc-root",
            type=str,
            default=os.path.join(script_dir, "data", "Scan2CAD_pc"),
            help="Path to Scan2CAD",
        )
        parser.add_argument(
            "--scan2cad-annotation-root",
            type=str,
            default=os.path.join(script_dir, "data", "Scan2CAD_annotations"),
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

        self.logger = Logger("./logs", "evaluation.txt")
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

        # Evaluate Registration Results
        # registration
        if not self._load_data():
            self.Ts_est_ransac = []
            self.Ts_est_best = []
            self.t_losses_ransac = []
            self.t_losses_sym = []
            self.r_losses_ransac = []
            self.r_losses_sym = []
            self.sym_ransac_success = []
            self.chamfer_dist_ransac = []
            self.chamfer_dist_sym = []
            for i in tqdm(range(len(self.base_outputs)), ncols=80):
                tqdm.write(f"Processing {i}th query")
                baseF = self.base_outputs[i]
                xyz0 = self.base_origins[i]
                pos_idx = self.stat["top1_predict" if self.config.register_top1 else "gt"][i]
                cadF = self.lib_outputs[pos_idx]
                xyz1 = self.lib_origins[pos_idx]
                cad_sym = self.sym_label[pos_idx]
                T_est_best, chamfer_dist_best, T_est_ransac, chamfer_dist_ransac, sym_success = sym_pose(
                    baseF.detach().cpu().numpy(),
                    xyz0.detach().cpu().numpy(),
                    cadF.detach().cpu().numpy(),
                    xyz1.detach().cpu().numpy(),
                    cad_sym,
                    self.config.k_nn,
                    self.config.max_corr,
                )

                T0 = self.base_Ts[i].cpu().numpy()
                T1 = self.lib_Ts[pos_idx].cpu().numpy()
                t_loss_ransac, r_loss_ransac = eval_pose(T_est_ransac, T0, T1, axis_symmetry=cad_sym)
                t_loss_sym, r_loss_sym = eval_pose(T_est_best, T0, T1, axis_symmetry=cad_sym)

                tqdm.write(f"ransac: translation error: {t_loss_ransac}, rotation error: {r_loss_ransac}")
                tqdm.write(f"sym: translation error: {t_loss_sym}, rotation error: {r_loss_sym}")

                self.Ts_est_ransac.append(T_est_ransac.detach().cpu().numpy())
                self.Ts_est_best.append(T_est_best.detach().cpu().numpy())
                self.t_losses_ransac.append(t_loss_ransac)
                self.t_losses_sym.append(t_loss_sym)
                self.r_losses_ransac.append(r_loss_ransac)
                self.r_losses_sym.append(r_loss_sym)
                self.sym_ransac_success.append(sym_success)
                self.chamfer_dist_ransac.append(chamfer_dist_ransac)
                self.chamfer_dist_sym.append(chamfer_dist_best)
            self._save_data()

        def _compute_rte(t_losses):
            rte_002 = np.sum(np.array(t_losses) <= 0.02) / len(t_losses)
            rte_005 = np.sum(np.array(t_losses) <= 0.05) / len(t_losses)
            rte_010 = np.sum(np.array(t_losses) <= 0.10) / len(t_losses)
            rte_015 = np.sum(np.array(t_losses) <= 0.15) / len(t_losses)
            return rte_002, rte_005, rte_010, rte_015

        def _compute_rre(r_losses):
            r_losses = np.rad2deg(np.array(r_losses))
            rre_005 = np.sum(r_losses <= 5) / len(r_losses)
            rre_015 = np.sum(r_losses <= 15) / len(r_losses)
            rre_045 = np.sum(r_losses <= 45) / len(r_losses)
            return rre_005, rre_015, rre_045

        t_loss_ransac = np.mean(self.t_losses_ransac)
        rte_002_ransac, rte_005_ransac, rte_010_ransac, rte_015_ransac = _compute_rte(self.t_losses_ransac)
        t_loss_sym = np.mean(self.t_losses_sym)
        rte_002_sym, rte_005_sym, rte_010_sym, rte_015_sym = _compute_rte(self.t_losses_sym)
        r_loss_ransac = np.mean(self.r_losses_ransac)
        rre_005_ransac, rre_015_ransac, rre_045_ransac = _compute_rre(self.r_losses_ransac)
        r_loss_sym = np.mean(self.r_losses_sym)
        rre_005_sym, rre_015_sym, rre_045_sym = _compute_rre(self.r_losses_sym)
        chamfer_dist_ransac = np.mean(self.chamfer_dist_ransac)
        chamfer_dist_sym = np.mean(self.chamfer_dist_sym)
        sym_success_rate = np.mean(self.sym_ransac_success)
        self.logger.log(
            f"\n==================================================================\n"
            f"vanilla ransac:\n"
            f"translation error: {t_loss_ransac},\n"
            f"rte 0.02: {rte_002_ransac}, rte 0.05: {rte_005_ransac}, "
            f"rte 0.10: {rte_010_ransac}, rte 0.15: {rte_015_ransac}\n"
            f"------------------------------------------------------------------\n"
            f"rotation error: {r_loss_ransac},\n"
            f"rre 5: {rre_005_ransac}, rre 15: {rre_015_ransac}, rre 45: {rre_045_ransac},\n"
            f"chamfer distance: {chamfer_dist_ransac}"
        )
        self.logger.log(
            f"\n==================================================================\n"
            f"sym ransac:\n"
            f"translation error: {t_loss_sym},\n"
            f"rte 0.02: {rte_002_sym}, rte 0.05: {rte_005_sym}, "
            f"rte 0.10: {rte_010_sym}, rte 0.15: {rte_015_sym}\n"
            f"------------------------------------------------------------------\n"
            f"rotation error: {r_loss_sym},\n"
            f"rre 5: {rre_005_sym}, rre 15: {rre_015_sym}, rre 45: {rre_045_sym},\n"
            f"chamfer distance: {chamfer_dist_sym}\n"
            f"==================================================================\n"
        )

        self.logger.log(f"sym success rate: {sym_success_rate}")

        # Visualize Results
        self.logger.log("Visualizing retrieval results")
        self._init_gui()
        self.visualize()

    def _load_data(self) -> bool:
        if self.config.ignore_cache:
            return False

        def _load_np(name: str):
            name = os.path.join(self.config.cache_dir, name)
            if not os.path.exists(name):
                raise FileNotFoundError(f"{name} not found")
            return np.load(name)

        if self.config.register_top1:
            suffix = "_top1.npy"
        else:
            suffix = "_gt.npy"

        try:
            self.Ts_est_ransac = _load_np(f"Ts_est_ransac_{self.config.category}{suffix}")
            self.Ts_est_ransac = [x.reshape(4, 4) for x in self.Ts_est_ransac]
            self.Ts_est_best = _load_np(f"Ts_est_best_{self.config.category}{suffix}")
            self.Ts_est_best = [x.reshape(4, 4) for x in self.Ts_est_best]
            self.t_losses_ransac = _load_np(f"t_losses_ransac_{self.config.category}{suffix}")
            self.t_losses_sym = _load_np(f"t_losses_sym_{self.config.category}{suffix}")
            self.r_losses_ransac = _load_np(f"r_losses_ransac_{self.config.category}{suffix}")
            self.r_losses_sym = _load_np(f"r_losses_sym_{self.config.category}{suffix}")
            self.sym_ransac_success = _load_np(f"sym_ransac_success_{self.config.category}{suffix}")
            self.chamfer_dist_ransac = _load_np(f"chamfer_dist_ransac_{self.config.category}{suffix}")
            self.chamfer_dist_sym = _load_np(f"chamfer_dist_sym_{self.config.category}{suffix}")
            return True
        except FileNotFoundError:
            return False

    def _save_data(self):
        def _save_np(name: str, data):
            name = os.path.join(self.config.cache_dir, name)
            np.save(name, data)

        if self.config.register_top1:
            suffix = "_top1.npy"
        else:
            suffix = "_gt.npy"

        Ts_est_ransac = np.array([x.flatten() for x in self.Ts_est_ransac])
        _save_np(f"Ts_est_ransac_{self.config.category}{suffix}", Ts_est_ransac)
        Ts_est_best = np.array([x.flatten() for x in self.Ts_est_best])
        _save_np(f"Ts_est_best_{self.config.category}{suffix}", Ts_est_best)
        _save_np(f"t_losses_ransac_{self.config.category}{suffix}", self.t_losses_ransac)
        _save_np(f"t_losses_sym_{self.config.category}{suffix}", self.t_losses_sym)
        _save_np(f"r_losses_ransac_{self.config.category}{suffix}", self.r_losses_ransac)
        _save_np(f"r_losses_sym_{self.config.category}{suffix}", self.r_losses_sym)
        _save_np(f"sym_ransac_success_{self.config.category}{suffix}", self.sym_ransac_success)
        _save_np(f"chamfer_dist_ransac_{self.config.category}{suffix}", self.chamfer_dist_ransac)
        _save_np(f"chamfer_dist_sym_{self.config.category}{suffix}", self.chamfer_dist_sym)

    def _init_gui(self):
        # layout:
        # |0: The full window
        # |---------------------------|--------------------------|--------------------------|
        # | 1. The query point cloud  | 2. The top pos CAD model | 3. The top neg CAD model |
        # | 4. The colored features   | 5. Vanilla RANSAC        | 6. Symmetry RANSAC       |
        # |---------------------------|--------------------------|--------------------------|
        dx = 0.01
        dy = 0.01
        nx = 3
        ny = 2
        ux = (1 - (nx + 1) * dx) / nx
        uy = (1 - (ny + 1) * dy) / ny
        bottom_left_xs = np.linspace(dx, 1, nx, endpoint=False)
        bottom_left_ys = np.linspace(dy, 1, ny, endpoint=False)[::-1]
        top_right_xs = bottom_left_xs + ux
        top_right_ys = bottom_left_ys + uy
        shape = [dict(bottomleft=(0, 0), topright=(1, 1), bg="k1")]
        for j in range(ny):
            for i in range(nx):
                shape.append(
                    dict(
                        bottomleft=(bottom_left_xs[i], bottom_left_ys[j]),
                        topright=(top_right_xs[i], top_right_ys[j]),
                        bg="w",
                    )
                )
        self.plotter = vedo.Plotter(shape=shape, sharecam=False, size=(1800, 1000))
        self.display_pc_idx = 0
        self.plotter.add_callback("KeyPress", self._keyboard_callback)
        self.vedo_query_pcd1 = None
        self.vedo_query_pcd2 = None
        self.vedo_query_pcd3 = None
        self.vedo_query_flagpole1 = None
        self.vedo_query_flagpole2 = None
        self.vedo_query_flagpole3 = None
        self.vedo_pos_pcd1 = None
        self.vedo_pos_pcd2 = None
        self.vedo_pos_pcd3 = None
        self.vedo_neg_pcd = None
        self.vedo_pos_flagpole1 = None
        self.vedo_pos_flagpole2 = None
        self.vedo_pos_flagpole3 = None
        self.vedo_neg_flagpole = None
        self.vedo_colored_pcd = None
        self.vedo_loss_text_ransac = None
        self.vedo_loss_text_sym = None
        print("Press Right/Left to change the query point cloud")

    def visualize(self):
        self._update_gui()
        self.plotter.at(0).show(interactive=True)

    def _keyboard_callback(self, event):
        if event.name != "KeyPressEvent":
            return
        if event.keypress == "Right":
            self.display_pc_idx += 1
        elif event.keypress == "Left":
            self.display_pc_idx -= 1
        elif event.keypress == "q":
            self.plotter.close()
            return
        if self.display_pc_idx < 0:
            self.display_pc_idx = 0
        elif self.display_pc_idx >= len(self.base_outputs):
            self.display_pc_idx = len(self.base_outputs) - 1

        self._update_gui()

    def _update_gui(self):
        pcd_file = self.scan2cad_info.test_files[self.display_pc_idx]
        tqdm.write(f"Query {self.config.category} {self.display_pc_idx}: {pcd_file}")
        tqdm.write(f"Predicted Matched CAD: {self.cad_lib.pathes[self.stat['top1_predict'][self.display_pc_idx]]}")
        tqdm.write(f"Ground Truth Matched CAD: {self.cad_lib.pathes[self.stat['gt'][self.display_pc_idx]]}")

        if self.vedo_query_pcd1 is None:
            self.plotter.at(1).add(vedo.Text2D("Query Point Cloud"))
        else:
            self.plotter.at(1).remove(self.vedo_query_pcd1, self.vedo_query_flagpole1)
            self.plotter.at(2).remove(self.vedo_query_pcd1)
            self.plotter.at(3).remove(self.vedo_query_pcd1)
        pcd_points = self.base_origins[self.display_pc_idx].detach().cpu().numpy()
        transform = self.base_Ts[self.display_pc_idx].detach().cpu().numpy()
        self.vedo_query_pcd1 = vedo.Points(pcd_points).apply_transform(np.linalg.inv(transform)).color("red")
        self.vedo_query_flagpole1 = self.vedo_query_pcd1.flagpole(
            f"Query {self.config.category} {self.display_pc_idx}",
            s=0.05,
        )
        self.plotter.at(1).add(self.vedo_query_pcd1, self.vedo_query_flagpole1).render(resetcam=True)

        if self.vedo_pos_pcd1 is None:
            self.plotter.at(2).add(vedo.Text2D("Predicted Closest CAD PC"))
            self.plotter.at(3).add(vedo.Text2D("Predicted Farthest CAD PC"))
        else:
            self.plotter.at(2).remove(self.vedo_pos_pcd1, self.vedo_pos_flagpole1)
            self.plotter.at(3).remove(self.vedo_neg_pcd, self.vedo_neg_flagpole)
        dists = np.linalg.norm(
            self.descriptors[self.display_pc_idx, None, :] - self.lib_descriptors[None, :, :],
            ord=2,
            axis=2,
        )[0, :]
        pos_idx = np.argmin(dists)  # top1-predict
        assert pos_idx == self.stat["top1_predict"][self.display_pc_idx]
        neg_idx = np.argmax(dists)
        self.vedo_pos_pcd1 = vedo.Points(self.lib_origins[pos_idx].detach().cpu().numpy()).color("green")
        self.vedo_neg_pcd = vedo.Points(self.lib_origins[neg_idx].detach().cpu().numpy()).color("blue")
        self.vedo_pos_flagpole1 = self.vedo_pos_pcd1.flagpole(f"Positive CAD {pos_idx}", s=0.05)
        self.vedo_neg_flagpole = self.vedo_neg_pcd.flagpole(f"Negative CAD {neg_idx}", s=0.05)
        self.plotter.at(2).add(self.vedo_query_pcd1, self.vedo_pos_pcd1, self.vedo_pos_flagpole1).render(resetcam=True)
        self.plotter.at(3).add(self.vedo_query_pcd1, self.vedo_neg_pcd, self.vedo_neg_flagpole).render(resetcam=True)

        # color point features
        if self.vedo_colored_pcd is None:
            self.plotter.at(4).add(vedo.Text2D("Point features coloring"))
        else:
            self.plotter.at(4).remove(self.vedo_colored_pcd)
        all_points = np.concatenate([pcd_points, self.vedo_pos_pcd1.vertices + np.array([2, 0, 0])], axis=0)
        all_feats = np.concatenate(
            [
                self.base_outputs[self.display_pc_idx].detach().cpu().numpy(),
                self.lib_outputs[pos_idx].detach().cpu().numpy(),
            ],
            axis=0,
        )
        tsne_results = embed_tsne(all_feats)
        colors = (get_color_map(tsne_results) * 255).astype(np.uint8)
        self.vedo_colored_pcd = vedo.Points(all_points)
        self.vedo_colored_pcd.pointcolors = colors
        self.plotter.at(4).add(self.vedo_colored_pcd).render(resetcam=True)

        # visualize registration by vanilla ransac
        pos_idx = self.stat["top1_predict" if self.config.register_top1 else "gt"][self.display_pc_idx]
        if self.vedo_query_pcd2 is None:
            self.plotter.at(5).add(vedo.Text2D("Registration (Vanilla RANSAC)"))
        else:
            self.plotter.at(5).remove(
                self.vedo_query_pcd2,
                self.vedo_pos_pcd2,
                self.vedo_query_flagpole2,
                self.vedo_pos_flagpole2,
                self.vedo_loss_text_ransac,
            )
        T_est_ransac = self.Ts_est_ransac[self.display_pc_idx]
        self.vedo_query_pcd2 = vedo.Points(pcd_points).apply_transform(T_est_ransac).color("red")
        self.vedo_pos_pcd2 = vedo.Points(self.lib_origins[pos_idx].detach().cpu().numpy()).color("green")
        self.vedo_query_flagpole2 = self.vedo_query_pcd2.flagpole(
            f"Query {self.config.category} {self.display_pc_idx}", s=0.05
        )
        self.vedo_pos_flagpole2 = self.vedo_pos_pcd2.flagpole(f"CAD {pos_idx}", s=0.05)
        self.vedo_loss_text_ransac = vedo.Text2D(
            f"translation error: {self.t_losses_ransac[self.display_pc_idx]:.3f}\n"
            f"rotation error: {self.r_losses_ransac[self.display_pc_idx]:.3f}",
            pos="bottom-right",
        )
        self.plotter.at(5).add(
            self.vedo_query_pcd2,
            self.vedo_pos_pcd2,
            self.vedo_query_flagpole2,
            self.vedo_pos_flagpole2,
            self.vedo_loss_text_ransac,
        ).render(resetcam=True)

        # visualize registration by symmetry ransac
        if self.vedo_query_pcd3 is None:
            self.plotter.at(6).add(vedo.Text2D("Registration (Symmetry RANSAC)"))
        else:
            self.plotter.at(6).remove(
                self.vedo_query_pcd3,
                self.vedo_pos_pcd3,
                self.vedo_query_flagpole3,
                self.vedo_pos_flagpole3,
                self.vedo_loss_text_sym,
            )
        T_est_best = self.Ts_est_best[self.display_pc_idx]
        self.vedo_query_pcd3 = vedo.Points(pcd_points).apply_transform(T_est_best).color("red")
        self.vedo_pos_pcd3 = vedo.Points(self.lib_origins[pos_idx].detach().cpu().numpy()).color("green")
        self.vedo_query_flagpole3 = self.vedo_query_pcd3.flagpole(
            f"Query {self.config.category} {self.display_pc_idx}", s=0.05
        )
        self.vedo_pos_flagpole3 = self.vedo_pos_pcd3.flagpole(f"CAD {pos_idx}", s=0.05)
        self.vedo_loss_text_sym = vedo.Text2D(
            f"translation error: {self.t_losses_sym[self.display_pc_idx]:.3f}\n"
            f"rotation error: {self.r_losses_sym[self.display_pc_idx]:.3f}",
            pos="bottom-right",
        )
        self.plotter.at(6).add(
            self.vedo_query_pcd3,
            self.vedo_pos_pcd3,
            self.vedo_query_flagpole3,
            self.vedo_pos_flagpole3,
            self.vedo_loss_text_sym,
        ).render(resetcam=True)


if __name__ == "__main__":
    App()
