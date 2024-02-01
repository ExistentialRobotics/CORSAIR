from dataclasses import dataclass
import numpy as np
import os
import transforms3d as t3d
from tqdm import tqdm
from model import load_model
import torch
from model import fc
import MinkowskiEngine as ME
from utils.symmetry import sym_pose
from utils.eval_pose import eval_pose
from scipy.spatial import KDTree
import vedo
import argparse
import pandas as pd
import open3d as o3d
import random


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


def main():
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
    config = Config(
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

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    o3d.utility.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)

    category_dir = os.path.join(config.shapenet_root, config.category_id, "test")
    pc_files = os.listdir(category_dir)
    pc_files = [os.path.join(category_dir, f) for f in pc_files]
    if config.n_models <= 0:
        config.n_models = len(pc_files)
    else:
        config.n_models = min(config.n_models, len(pc_files))
    if config.n_models < len(pc_files):
        pc_files = np.random.choice(pc_files, config.n_models, replace=False)

    model = load_model("ResUNetBN2C")(
        in_channels=1,
        out_channels=16,
        bn_momentum=0.05,
        normalize_feature=True,
        conv1_kernel_size=3,
        D=3,
    ).to(config.device)
    embedding = fc.conv1_max_embedding(1024, 512, 256).to(config.device)
    checkpoint = torch.load(config.model_ckpt, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])
    embedding.load_state_dict(checkpoint["embedding_state_dict"])
    model.eval()
    embedding.eval()

    df = pd.DataFrame(
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

    for pc_file in tqdm(pc_files, ncols=160, desc="model", position=0):
        for pose_idx in tqdm(range(config.n_poses_per_model), ncols=160, desc="pose", position=1, leave=False):
            coords_pc, coords_pc_transformed, pose_gt = generate_test_pc_pair(config, pc_file)

            pc_quant, pc_grid = quantize_pc(coords_pc, config.voxel_size)
            pc_transformed_quant, pc_transformed_grid = quantize_pc(coords_pc_transformed, config.voxel_size)

            batch_pc, batch_feats = batch_coords_and_feats(
                [pc_grid, pc_transformed_grid],
                [torch.ones((pc_grid.shape[0], 1)), torch.ones((pc_transformed_grid.shape[0], 1))],
            )
            batch_input = ME.SparseTensor(batch_feats.to(config.device), batch_pc.to(config.device))
            batch_local_feats, batch_global_feats = model(batch_input)
            batch_global_feats = embedding(batch_global_feats)

            mask_pc = batch_local_feats.C[:, 0] == 0
            feats_pc = batch_local_feats.F[mask_pc, :]
            feats_pc_transformed = batch_local_feats.F[~mask_pc, :]

            symmetry_label = get_symmetry_label(coords_pc, 0.1)

            T_est_sym, chamfer_dist_sym, T_est_ransac, chamfer_dist_ransac, sym_success = sym_pose(
                feats_pc.detach().cpu().numpy(),
                pc_quant.detach().cpu().numpy(),
                feats_pc_transformed.detach().cpu().numpy(),
                pc_transformed_quant.detach().cpu().numpy(),
                symmetry_label,
                config.k_nn,
                config.max_corr,
                seed=config.random_seed,
            )
            rte_sym, rre_sym = eval_pose(T_est_sym, np.eye(4), pose_gt, axis_symmetry=symmetry_label)
            rte_ransac, rre_ransac = eval_pose(T_est_ransac, np.eye(4), pose_gt, axis_symmetry=symmetry_label)

            tqdm.write(f"symmetry label: {symmetry_label}, success: {sym_success}")
            tqdm.write(f"sym: RTE={rte_sym:.4f}, RRE={rre_sym:.4f}, CD={chamfer_dist_sym:.4f}")
            tqdm.write(f"ransac: RTE={rte_ransac:.4f}, RRE={rre_ransac:.4f}, CD={chamfer_dist_ransac:.4f}")

            # T_est_sym = T_est_sym.detach().cpu().numpy()
            # vedo_pc_gt = vedo.Points(coords_pc_transformed).color("green")
            # pc_sym = coords_pc @ T_est_sym[:3, :3].T + T_est_sym[:3, [3]].T
            # vedo_pc_sym = vedo.Points(pc_sym).color("red")
            # vedo.show(vedo_pc_gt, vedo_pc_sym)

            df.loc[len(df)] = [
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

            # percentage that rte <= 0.02
            rte_002_sym = (df["rte_sym"] <= 0.02).sum() / len(df)
            rte_002_ransac = (df["rte_ransac"] <= 0.02).sum() / len(df)
            rre_05_sym = (df["rre_sym"] <= np.deg2rad(5)).sum() / len(df)
            rre_05_ransac = (df["rre_ransac"] <= np.deg2rad(5)).sum() / len(df)
            tqdm.write(f"RTE <= 0.02: sym: {rte_002_sym:.4f}, ransac: {rte_002_ransac:.4f}")
            tqdm.write(f"RRE <= 5 deg: sym: {rre_05_sym:.4f}, ransac: {rre_05_ransac:.4f}")

    df.to_csv(f"results-shapenet-{config.category}.csv", index=False)


if __name__ == "__main__":
    main()
