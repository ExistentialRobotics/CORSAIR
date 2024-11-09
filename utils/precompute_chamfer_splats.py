import numpy as np
from tqdm.contrib.concurrent import process_map, thread_map
from dataclasses import dataclass
import argparse
from utils.Info.CADLib import CustomizeCADLib
from utils.Info.Scan2cadInfo import Scan2cadInfo
from utils.preprocess import apply_transform, load_raw_pc, chamfer_kdtree_1direction
import os
import open3d as o3d
import itertools
import pandas as pd

@dataclass
class Config:
    shapenet_pc15k_root: str
    scan2cad_pc_root: str
    scan2cad_annotation_root: str
    shapenet_radegs_root: str
    category: str
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

        parser = argparse.ArgumentParser(description="Generate CD matrix")
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

        args = parser.parse_args()
        self.config = Config(**vars(args))

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

        num_objects = len(self.scan2cad_info.UsedObjId)

        # For all objects, compute the chamfer distance between GT object I and splat render J
        chamfer_dist_matrix = np.empty( (num_objects, num_objects) )

        fixed_transform_gsplat = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])

        self.args = itertools.product(
            self.scan2cad_info.UsedObjId,
            self.scan2cad_info.UsedObjId
        )

        self.result = thread_map(self.compute_cd, self.args, max_workers=32)

        # with tqdm(total = num_objects * num_objects) as pbar:
        #     for obj_idx, obj_id in tqdm(enumerate(self.scan2cad_info.UsedObjId)):
        #         for splat_idx, splat_id in tqdm(enumerate(self.scan2cad_info.UsedObjId)):
        
        # np.save('configs/cd_table.npy', chamfer_dist_matrix)

    def compute_cd(self, arg):
        obj_id, splat_id = arg
        align_cad_xyz = load_raw_pc(
            self.cad_lib.CadPcs[self.cad_lib.id2idx[obj_id]],
            15000
        )
        gsplat_recon_path = os.path.join(
            self.config.shapenet_radegs_root, 
            self.config.catid, 
            splat_id, 
            "recon.ply"
        )
        fixed_transform_gsplat = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])


        retrieved_gsplat_recon_mesh = o3d.io.read_triangle_mesh(gsplat_recon_path)
        retrieved_gsplat_recon_pcd = retrieved_gsplat_recon_mesh.sample_points_uniformly(number_of_points=15000)
        retrieved_gsplat_xyz = apply_transform(np.asarray(retrieved_gsplat_recon_pcd.points), fixed_transform_gsplat)

        # compute Chamfer distance
        chamfer_dist = chamfer_kdtree_1direction(align_cad_xyz, retrieved_gsplat_xyz) + \
            chamfer_kdtree_1direction(retrieved_gsplat_xyz, align_cad_xyz)

        return obj_id, splat_id, chamfer_dist
        
if __name__ == "__main__":
    app = App()
    df = pd.DataFrame(app.result, columns=['obj_id', 'splat_id', 'chamfer_dist'])
    df.to_csv('chamfer_dist_list.csv', index=False)