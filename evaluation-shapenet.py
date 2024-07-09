import argparse
import os
import random
from dataclasses import dataclass

import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import pandas as pd
import torch
import transforms3d as t3d
import vedo
from joblib import Parallel
from joblib import delayed
from scipy.spatial import KDTree
from tqdm import tqdm

from model import load_model
from utils.eval_pose import eval_pose
from utils.symmetry import sym_pose


@dataclass
class Config:
    shapenet_root: str
    category: str
    n_models: int
    n_poses_per_model: int
    max_roll_deg: float
    max_pitch_deg: float
    max_yaw_deg: float
    max_translation_x: float
    max_translation_y: float
    max_translation_z: float

    model_ckpt: str
    device: str
    random_seed: int

    category_id: str = None
    voxel_size: float = 0.03
    k_nn: int = 5
    max_corr: float = 0.4

    def __post_init__(self):
        if self.category == "table":
            self.category_id = "04379243"
        elif self.category == "chair":
            self.category_id = "03001627"
        else:
            raise ValueError(f"Unsupported category: {self.category}")
        assert self.n_poses_per_model > 0, "n_poses_per_model must be positive"


def load_pc(path):
    pc = np.load(path)
    t = pc.mean(axis=0, keepdims=True)
    pc -= t
    r = np.linalg.norm(pc, axis=1).max()
    pc /= r
    return pc


def generate_random_pose(config: Config):
    roll = np.random.uniform(-config.max_roll_deg, config.max_roll_deg)
    pitch = np.random.uniform(-config.max_pitch_deg, config.max_pitch_deg)
    yaw = np.random.uniform(-config.max_yaw_deg, config.max_yaw_deg)
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    translation_x = np.random.uniform(-config.max_translation_x, config.max_translation_x)
    translation_y = np.random.uniform(-config.max_translation_y, config.max_translation_y)
    translation_z = np.random.uniform(-config.max_translation_z, config.max_translation_z)
    pose = np.eye(4)
    pose[:3, :3] = t3d.euler.euler2mat(roll, pitch, yaw)
    pose[0, 3] = translation_x
    pose[1, 3] = translation_y
    pose[2, 3] = translation_z
    return pose


def quantize_pc(pc, voxel_size):
    unique_idx = ME.utils.sparse_quantize(
        np.ascontiguousarray(np.floor(pc / voxel_size)),
        return_index=True,
        return_maps_only=True,
    )
    pc = pc[unique_idx, :]
    pc_grid = np.ascontiguousarray(np.floor(pc / voxel_size))
    pc = torch.from_numpy(pc).float()
    pc_grid = torch.from_numpy(pc_grid).int()
    return pc, pc_grid


def batch_coords_and_feats(coords, feats):
    batch_coords, batch_feats = ME.utils.sparse_collate(coords, feats)
    return batch_coords, batch_feats


def generate_test_pc_pair(config: Config, pc_file):
    pc = load_pc(pc_file)
    pose = generate_random_pose(config)
    pc_transformed = pc @ pose[:3, :3].T + pose[:3, [3]].T
    return pc, pc_transformed, pose


def chamfer_max(pc0, pc0_tree, pc1, pc1_tree):
    max0 = 0
    for i in range(len(pc0)):
        dist, idx = pc1_tree.query(pc0[i], k=1)
        if dist > max0:
            max0 = dist

    max1 = 0
    for i in range(len(pc1)):
        dist, idx = pc0_tree.query(pc1[i], k=1)
        if dist > max1:
            max1 = dist

    return max(max0, max1)


def test_symmetry_label(sym_label: int, pc: np.ndarray, cd_threshold: float):
    pc_tree = KDTree(pc)
    for i in range(1, int(sym_label / 2) + 1):
        # in ShapeNet, y-axis is upward
        R = t3d.euler.euler2mat(0, i * (2 * np.pi) / sym_label, 0)
        pc_rot = pc @ R.T
        pc_rot_tree = KDTree(pc_rot)
        error = chamfer_max(pc, pc_tree, pc_rot, pc_rot_tree)
        if error > cd_threshold:
            return False
    return True


def get_symmetry_label(pc: np.ndarray, cd_threshold: float) -> int:
    for sym_label in [12, 8, 6, 4, 3, 2, 1]:  # 1 is non-symmetry
        if test_symmetry_label(sym_label, pc, cd_threshold):
            return sym_label
    return 0


class App:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--shapenet-root", type=str, required=True)
        parser.add_argument("--category", type=str, required=True)
        parser.add_argument("--n-models", type=int, default=1)
        parser.add_argument("--n-poses-per-model", type=int, default=10)
        parser.add_argument("--max-roll-deg", type=float, default=360)
        parser.add_argument("--max-pitch-deg", type=float, default=360)
        parser.add_argument("--max-yaw-deg", type=float, default=360)
        parser.add_argument("--max-translation-x", type=float, default=1.0)
        parser.add_argument("--max-translation-y", type=float, default=1.0)
        parser.add_argument("--max-translation-z", type=float, default=1.0)
        parser.add_argument("--model-ckpt", type=str, required=True)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--random-seed", type=int, default=0)
        args = parser.parse_args()
        self.config = Config(
            shapenet_root=args.shapenet_root,
            category=args.category,
            n_models=args.n_models,
            n_poses_per_model=args.n_poses_per_model,
            max_roll_deg=args.max_roll_deg,
            max_pitch_deg=args.max_pitch_deg,
            max_yaw_deg=args.max_yaw_deg,
            max_translation_x=args.max_translation_x,
            max_translation_y=args.max_translation_y,
            max_translation_z=args.max_translation_z,
            model_ckpt=args.model_ckpt,
            device=args.device,
            random_seed=args.random_seed,
        )

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        o3d.utility.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)

        self.category_dir = os.path.join(self.config.shapenet_root, self.config.category_id, "test")
        self.pc_files = os.listdir(self.category_dir)
        self.pc_files = [os.path.join(self.category_dir, f) for f in self.pc_files]
        if self.config.n_models <= 0:
            self.config.n_models = len(self.pc_files)
        else:
            self.config.n_models = min(self.config.n_models, len(self.pc_files))

        if self.config.n_models < len(self.pc_files):
            self.pc_files = np.random.choice(self.pc_files, self.config.n_models, replace=False)
        self.pc_files = sorted(self.pc_files)

        postfix = (
            f"shapenet-seed{self.config.random_seed}-{self.config.category}-"
            f"{self.config.n_models}-{self.config.n_poses_per_model}"
        )
        self.csv_file = f"results-{postfix}.csv"
        self.npz_file = f"poses-{postfix}.npz"
        if os.path.exists(self.csv_file) and os.path.exists(self.npz_file):
            with open(self.csv_file, "r") as f:
                self.df = pd.read_csv(f)
            with open(self.npz_file, "rb") as f:
                data = np.load(f)
                self.poses_gt = data["poses_gt"]
                self.poses_pred_sym = data["poses_pred_sym"]
                self.poses_pred_ransac = data["poses_pred_ransac"]
        else:
            self.generate_test_results()

        # print the statistics based on collected results
        rte_002_sym = (self.df["rte_sym"] <= 0.02).sum() / len(self.df)
        rte_002_ransac = (self.df["rte_ransac"] <= 0.02).sum() / len(self.df)
        rre_05_sym = (self.df["rre_sym"] <= np.deg2rad(5)).sum() / len(self.df)
        rre_05_ransac = (self.df["rre_ransac"] <= np.deg2rad(5)).sum() / len(self.df)
        tqdm.write(f"RTE <= 0.02: sym: {rte_002_sym:.4f}, ransac: {rte_002_ransac:.4f}")
        tqdm.write(f"RRE <= 5 deg: sym: {rre_05_sym:.4f}, ransac: {rre_05_ransac:.4f}")

        # Visualize Results
        self._init_gui()
        self.visualize()

    def registration_worker(self, feats_pc, feats_pc_transformed, coords_pc, pc_quant, pc_transformed_quant, pose_gt):
        symmetry_label = get_symmetry_label(coords_pc, 0.1)

        T_est_sym, chamfer_dist_sym, T_est_ransac, chamfer_dist_ransac, sym_success = sym_pose(
            feats_pc,
            pc_quant,
            feats_pc_transformed,
            pc_transformed_quant,
            symmetry_label,
            self.config.k_nn,
            self.config.max_corr,
            seed=self.config.random_seed,
        )
        rte_sym, rre_sym = eval_pose(T_est_sym, np.eye(4), pose_gt, axis_symmetry=symmetry_label)
        rte_ransac, rre_ransac = eval_pose(T_est_ransac, np.eye(4), pose_gt, axis_symmetry=symmetry_label)

        results = dict(
            symmetry_label=symmetry_label,
            T_est_sym=T_est_sym,
            chamfer_dist_sym=chamfer_dist_sym,
            T_est_ransac=T_est_ransac,
            chamfer_dist_ransac=chamfer_dist_ransac,
            sym_success=sym_success,
            rte_sym=rte_sym,
            rre_sym=rre_sym,
            rte_ransac=rte_ransac,
            rre_ransac=rre_ransac,
        )

        # tqdm.write(f"symmetry label: {symmetry_label}, success: {sym_success}")
        # tqdm.write(f"sym: RTE={rte_sym:.4f}, RRE={rre_sym:.4f}, CD={chamfer_dist_sym:.4f}")
        # tqdm.write(f"ransac: RTE={rte_ransac:.4f}, RRE={rre_ransac:.4f}, CD={chamfer_dist_ransac:.4f}")

        return results

    def registration_producer(self):
        model = load_model("ResUNetBN2C")(
            in_channels=1,
            out_channels=16,
            bn_momentum=0.05,
            normalize_feature=True,
            conv1_kernel_size=3,
            D=3,
        ).to(self.config.device)
        # embedding = fc.conv1_max_embedding(1024, 512, 256).to(self.config.device)
        checkpoint = torch.load(self.config.model_ckpt, map_location=self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        # embedding.load_state_dict(checkpoint["embedding_state_dict"])
        model.eval()
        # embedding.eval()

        for pc_file in tqdm(self.pc_files, ncols=160, desc="model", position=0):
            for _ in tqdm(range(self.config.n_poses_per_model), ncols=160, desc="pose", position=1, leave=False):
                coords_pc, coords_pc_transformed, pose_gt = generate_test_pc_pair(self.config, pc_file)

                pc_quant, pc_grid = quantize_pc(coords_pc, self.config.voxel_size)
                pc_transformed_quant, pc_transformed_grid = quantize_pc(coords_pc_transformed, self.config.voxel_size)

                batch_pc, batch_feats = batch_coords_and_feats(
                    [pc_grid, pc_transformed_grid],
                    [torch.ones((pc_grid.shape[0], 1)), torch.ones((pc_transformed_grid.shape[0], 1))],
                )
                batch_input = ME.SparseTensor(batch_feats.to(self.config.device), batch_pc.to(self.config.device))
                batch_local_feats, batch_global_feats = model(batch_input)

                mask_pc = batch_local_feats.C[:, 0] == 0
                feats_pc = batch_local_feats.F[mask_pc, :]
                feats_pc_transformed = batch_local_feats.F[~mask_pc, :]
                self.poses_gt.append(pose_gt)

                yield (
                    feats_pc.detach().cpu().numpy(),
                    feats_pc_transformed.detach().cpu().numpy(),
                    coords_pc,
                    pc_quant.detach().cpu().numpy(),
                    pc_transformed_quant.detach().cpu().numpy(),
                    pose_gt,
                )

    def generate_test_results(self):
        self.df = pd.DataFrame(
            columns=[
                "model",
                "pose_idx",
                "symmetry_label",
                "sym_success",
                "rte_sym",
                "rre_sym",
                "cd_sym",
                "rte_ransac",
                "rre_ransac",
                "cd_ransac",
            ]
        )

        self.poses_gt = []
        self.poses_pred_sym = []
        self.poses_pred_ransac = []

        results = Parallel(n_jobs=-1, verbose=100, pre_dispatch="1.5*n_jobs")(
            delayed(self.registration_worker)(*x) for x in self.registration_producer()
        )

        for idx, result in enumerate(results):
            symmetry_label = result["symmetry_label"]
            T_est_sym = result["T_est_sym"]
            chamfer_dist_sym = result["chamfer_dist_sym"]
            T_est_ransac = result["T_est_ransac"]
            chamfer_dist_ransac = result["chamfer_dist_ransac"]
            sym_success = result["sym_success"]
            rte_sym = result["rte_sym"]
            rre_sym = result["rre_sym"]
            rte_ransac = result["rte_ransac"]
            rre_ransac = result["rre_ransac"]

            self.poses_pred_sym.append(T_est_sym)
            self.poses_pred_ransac.append(T_est_ransac)

            pc_file_idx = idx // self.config.n_poses_per_model
            pose_idx = idx % self.config.n_poses_per_model
            pc_file = self.pc_files[pc_file_idx]
            self.df.loc[len(self.df)] = [
                os.path.basename(pc_file),
                pose_idx,
                symmetry_label,
                sym_success,
                rte_sym,
                rre_sym,
                chamfer_dist_sym,
                rte_ransac,
                rre_ransac,
                chamfer_dist_ransac,
            ]

        self.df.to_csv(self.csv_file, index=False)

        with open(self.npz_file, "wb") as f:
            np.savez(
                f, poses_gt=self.poses_gt, poses_pred_sym=self.poses_pred_sym, poses_pred_ransac=self.poses_pred_ransac
            )

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
        elif self.display_pc_idx >= len(self.df):
            self.display_pc_idx = len(self.df) - 1

        self._update_gui()

    def _init_gui(self):
        # layout:
        # |0: The full window
        # |-------------------------|------------------------------|
        # | 1. The org point cloud and the transformed point cloud |
        # | 2. Vanilla RANSAC       | 3. Symmetry RANSAC           |
        # |-------------------------|------------------------------|
        dx = 0.01
        dy = 0.01
        nx = 2
        ny = 2
        ux = (1 - (nx + 1) * dx) / nx
        uy = (1 - (ny + 1) * dy) / ny
        bottom_left_xs = np.linspace(dx, 1, nx, endpoint=False)
        bottom_left_ys = np.linspace(dy, 1, ny, endpoint=False)[::-1]
        top_right_xs = bottom_left_xs + ux
        top_right_ys = bottom_left_ys + uy
        shape = [
            dict(bottomleft=(0, 0), topright=(1, 1), bg="k1"),
            dict(
                bottomleft=(bottom_left_xs[0], bottom_left_ys[0]), topright=(top_right_xs[1], top_right_ys[0]), bg="w"
            ),
            dict(
                bottomleft=(bottom_left_xs[0], bottom_left_ys[1]), topright=(top_right_xs[0], top_right_ys[1]), bg="w"
            ),
            dict(
                bottomleft=(bottom_left_xs[1], bottom_left_ys[1]), topright=(top_right_xs[1], top_right_ys[1]), bg="w"
            ),
        ]

        self.plotter = vedo.Plotter(shape=shape, sharecam=False, size=(1800, 1000))
        self.display_pc_idx = 0
        self.plotter.add_callback("KeyPress", self._keyboard_callback)
        self.vedo_pcd1 = None
        self.vedo_pcd2 = None
        self.vedo_pcd_ransac = None
        self.vedo_pcd_sym = None
        self.vedo_flagpole1 = None
        self.vedo_flagpole2 = None
        self.vedo_pcd_gt1 = None
        self.vedo_pcd_gt2 = None
        self.vedo_loss_text_ransac = None
        self.vedo_loss_text_sym = None
        print("Press Right/Left to change the query point cloud")

    def _update_gui(self):
        pcd_file = self.df["model"][self.display_pc_idx]
        pose_idx = self.df["pose_idx"][self.display_pc_idx]
        tqdm.write(f"Align {self.config.category} {self.display_pc_idx}: {pcd_file}, pose_idx: {pose_idx}")

        pcd_file = os.path.join(self.category_dir, pcd_file)
        pcd_points = load_pc(pcd_file)
        pose_gt = self.poses_gt[self.display_pc_idx]
        pcd_points_transformed = pcd_points @ pose_gt[:3, :3].T + pose_gt[:3, [3]].T

        if self.vedo_pcd1 is None:
            self.plotter.at(1).add(vedo.Text2D("Point Clouds"))
        else:
            self.plotter.at(1).remove(self.vedo_pcd1, self.vedo_flagpole1, self.vedo_pcd2, self.vedo_flagpole2)
        self.vedo_pcd1 = vedo.Points(pcd_points).color("red")
        self.vedo_pcd2 = vedo.Points(pcd_points_transformed).color("green")
        self.vedo_flagpole1 = self.vedo_pcd1.flagpole(f"Original point cloud", s=0.05)
        self.vedo_flagpole2 = self.vedo_pcd2.flagpole(f"Transformed point cloud (GT)", s=0.05)
        self.plotter.at(1).add(
            self.vedo_pcd1,
            self.vedo_flagpole1,
            self.vedo_pcd2,
            self.vedo_flagpole2,
        ).render(resetcam=True)

        # visualize registration by vanilla ransac
        if self.vedo_pcd_gt1 is None:
            self.plotter.at(2).add(vedo.Text2D("Registration (Vanilla RANSAC)"))
        else:
            self.plotter.at(2).remove(self.vedo_pcd_ransac, self.vedo_pcd_gt1, self.vedo_loss_text_ransac)
        pose_pred_ransac = self.poses_pred_ransac[self.display_pc_idx]
        self.vedo_pcd_ransac = vedo.Points(pcd_points).apply_transform(pose_pred_ransac).color("red")
        self.vedo_pcd_gt1 = vedo.Points(pcd_points_transformed).color("green")
        rte_ransac = self.df["rte_ransac"][self.display_pc_idx]
        rre_ransac = self.df["rre_ransac"][self.display_pc_idx]
        self.vedo_loss_text_ransac = vedo.Text2D(
            f"translation error: {rte_ransac:.3f}\n" f"rotation error: {rre_ransac:.3f}",
            pos="bottom-right",
        )
        self.plotter.at(2).add(
            self.vedo_pcd_ransac,
            self.vedo_pcd_gt1,
            self.vedo_loss_text_ransac,
        ).render(resetcam=True)

        # visualize registration by symmetry ransac
        if self.vedo_pcd_sym is None:
            self.plotter.at(3).add(vedo.Text2D("Registration (Symmetry RANSAC)"))
        else:
            self.plotter.at(3).remove(self.vedo_pcd_sym, self.vedo_pcd_gt2, self.vedo_loss_text_sym)
        pose_pred_sym = self.poses_pred_sym[self.display_pc_idx]
        self.vedo_pcd_sym = vedo.Points(pcd_points).apply_transform(pose_pred_sym).color("red")
        self.vedo_pcd_gt2 = vedo.Points(pcd_points_transformed).color("green")
        rte_sym = self.df["rte_sym"][self.display_pc_idx]
        rre_sym = self.df["rre_sym"][self.display_pc_idx]
        self.vedo_loss_text_sym = vedo.Text2D(
            f"translation error: {rte_sym:.3f}\n" f"rotation error: {rre_sym:.3f}",
            pos="bottom-right",
        )
        self.plotter.at(3).add(
            self.vedo_pcd_sym,
            self.vedo_pcd_gt2,
            self.vedo_loss_text_sym,
        ).render(resetcam=True)


if __name__ == "__main__":
    App()
