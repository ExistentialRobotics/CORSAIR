import os
import torch
from utils.preprocess import load_norm_pc, load_raw_pc, path_dict


class Reader(torch.utils.data.Dataset):
    def __init__(self, root, catid, split, npoint):
        self.root = root
        self.catid = catid
        self.split = split
        self.npoints = npoint
        self.files = os.listdir(os.path.join(root, catid, split))
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return load_norm_pc(
            os.path.join(self.root, self.catid, self.split, self.files[index]),
            self.npoints,
        )


class ScannetReader(Reader):
    def __init__(self, root, files, npoints):
        self.root = root
        self.files = files
        self.npoints = npoints

    def __getitem__(self, index):
        return load_raw_pc(os.path.join(self.root, self.files[index]), self.npoints)


class CategoryLibReader(Reader):
    def __init__(self, root, catid, splits, npoint, normal=False):
        self.root = root
        self.catid = catid
        self.splits = splits
        self.npoints = npoint
        self.normal = normal

        self.Id2Index = {}
        self.files = []
        for split in self.splits:
            pointcloudfiles = os.listdir(os.path.join(root, catid, split))
            pointcloudfiles.sort()
            for file in pointcloudfiles:
                self.Id2Index[file.split(".")[0]] = len(self.files)
                self.files.append(os.path.join(self.root, self.catid, split, file))

    def __getitem__(self, index):
        if self.normal:
            return load_norm_pc(self.files[index], self.npoints)
        else:
            return load_raw_pc(self.files[index], self.npoints)


class ReaderWithPath(Reader):
    def __init__(self, files, npoints, normal=False):
        self.files = files
        self.normal = normal
        self.npoints = npoints

    def __getitem__(self, index):

        if self.normal:
            return load_norm_pc(self.files[index], self.npoints)
        else:
            return load_raw_pc(self.files[index], self.npoints)


class Scan2cadLibReader(Reader):
    def __init__(self, root, catid, ids, id2path, npoint):
        """
        Read scan2cad used cad models only
        ids: ids of the used cad models
        """
        self.root = root
        self.catid = catid
        self.npoints = npoint
        self.id2path = id2path

        self.files = []
        for id in ids:
            self.files.append(self.id2path[id])

    def __getitem__(self, idx):
        return load_norm_pc(self.files[idx], self.npoints)
