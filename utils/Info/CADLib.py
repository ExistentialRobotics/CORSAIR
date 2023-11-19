import torch
import numpy as np
from tqdm import tqdm
import MinkowskiEngine as ME

from datasets.Reader import CategoryLibReader, ReaderWithPath

from utils.preprocess import path_dict, load_norm_pc


class CatCADLib:
    """
    Class that store and query CAD models.
    """

    def __init__(self, root, catid, table_path, preload=True):
        # Read Shapenet CAD model

        self.root = root
        self.catid = catid
        self.preload = preload
        self.table = np.load(table_path)

        print("Loading Shapenet CAD models")
        CadReader = CategoryLibReader(
            self.root, self.catid, ["train", "test", "val"], 10000, normal=False
        )
        CadReaderLoader = torch.utils.data.DataLoader(
            CadReader, batch_size=1, shuffle=False, num_workers=8
        )

        self.CadPcs = []
        if self.preload:
            for data in tqdm(CadReaderLoader):
                self.CadPcs.append(data[0, :, :].numpy())
        else:
            self.CadPcs = CadReader.files
        self.id2idx = CadReader.Id2Index


class CustomizeCADLib(torch.utils.data.Dataset):
    """
    Class that stores and query from a given collection CAD models.
    """

    def __init__(self, root, catid, ids, table_path, voxel_size, preload=True):
        """
        ids: list of cad model ids
        """

        self.root = root
        self.catid = catid
        self.voxel_size = voxel_size
        print(self.voxel_size)
        self.ids = ids
        self.preload = preload
        self.pathes = []
        self.id2path = path_dict(self.root)
        self.id2idx = {}
        self.table = np.load(table_path)

        """
        # select needed cad models distance function
        CatCadReader = CategoryLibReader(self.root, self.catid, ["train", "test", "val"], 10000, normal=False)
        """
        # id_in_category = []
        for idx, id in enumerate(self.ids):
            self.pathes.append(self.id2path[id])
            self.id2idx[id] = idx
            # id_in_category.append(CatCadReader.Id2Index[id])
        """
        id_in_category = np.array(id_in_category)
        self.table = self.table[id_in_category, :][:, id_in_category]
        """

        print("Loading Shapenet CAD models")
        CadReader = ReaderWithPath(self.pathes, 10000, normal=False)
        CadReaderLoader = torch.utils.data.DataLoader(
            CadReader, batch_size=1, shuffle=False, num_workers=8
        )

        self.CadPcs = []
        if self.preload:
            for data in tqdm(CadReaderLoader):
                self.CadPcs.append(data[0, :, :].numpy())
        else:
            self.CadPcs = CadReader.files

    def _getpc(self, idx):
        if self.preload:
            return self.CadPcs[idx]
        else:
            return load_norm_pc(self.CadPcs[idx], 10000)

    def quant(self, rot_coords, coords):
        if ME.__version__ >= "0.5.4":
            unique_idx = ME.utils.sparse_quantize(
                np.floor(rot_coords / self.voxel_size),
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

    def __len__(self):
        return len(self.CadPcs)

    def __getitem__(self, idx):

        base_coords = self._getpc(idx)

        rot_base_coords, rot_base_coords_grid, base_coords = self.quant(
            base_coords, base_coords
        )

        base_feat = np.ones([len(rot_base_coords), 1])

        identity = np.eye(4)

        base = {
            "coord": rot_base_coords_grid,
            "origin": rot_base_coords,
            "feat": base_feat,
            "T": identity,
            "idx": idx,
        }

        return base

    def collate_pair_fn(self, list_data):

        base_dict = list_data

        base_coords = []
        base_feat = []
        base_T = []
        base_origin = []
        base_idx = []

        for idx in range(len(base_dict)):

            base_coords.append(torch.from_numpy(base_dict[idx]["coord"]))
            base_origin.append(torch.from_numpy(base_dict[idx]["origin"]))
            base_feat.append(torch.from_numpy(base_dict[idx]["feat"]))
            base_idx.append(base_dict[idx]["idx"])
            base_T.append(torch.from_numpy(base_dict[idx]["T"]))

        batch_base_coords, batch_base_feat = ME.utils.sparse_collate(
            base_coords, base_feat
        )

        data = {}

        data["base_coords"] = batch_base_coords.int()
        data["base_feat"] = batch_base_feat.float()
        data["base_origin"] = torch.cat(base_origin, 0).float()
        data["base_idx"] = torch.Tensor(base_idx).int()
        data["base_T"] = torch.stack(base_T, 0).float()

        return data
