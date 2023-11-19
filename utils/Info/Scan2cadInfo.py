import os

from utils.read_json import load_json, load_csv
from utils.preprocess import path_dict


class Scan2cadInfo:
    def __init__(self, cad_root, scan_root, catid, annotation_dir):
        """
        A class for scan2cad dataset infos
        Contains:
            UsedObjId: used cad models in scan2cad (catid)
            UsedObjPath: path to cad models
            ScanObjPathes: path to scan objects
            {train/val/test}_keys: scene_ids in train/val/test
            {train/val/test}_files: scan objects in train/val/test
            {Train/Val/Test}BestMatches: best cad model corresponding to the scan objects
            {Train/Val/Test}CadPoses: cad poses
            {Train/Val/Test}ScanPoses: scan poses

        """

        self.scan_root = scan_root
        self.cad_root = cad_root

        self.annotation = load_json(os.path.join(annotation_dir, "full_annotations.json"))
        self.all_cad = load_csv(os.path.join(annotation_dir, "unique_cads.csv"))

        self.catid = catid

        self.id2path = path_dict(self.cad_root)
        self.UsedObjId, self.UsedObjPath = self.filter_cad()

        self.scan2pose = self.read_scan2pose()
        (
            self.ScanObjPathes,
            self.BestMatchesDict,
            self.CadPosesDict,
            self.ScanPosesDict,
        ) = self.read_scans()
        self.ScanObjPathes.sort()

        script_dir = os.path.dirname(__file__)
        path_train = os.path.join(script_dir, "../../configs/scannet_train.txt")
        path_test = os.path.join(script_dir, "../../configs/scannet_val.txt")
        self.train_and_val_scans, self.test_scans = self.read_scannet_split(path_train, path_test)

        self.scene_dict = self.get_scene_dict(self.ScanObjPathes)

        self.train_keys, self.val_keys, self.test_keys = self.filter_scan()

        (
            self.train_files,
            self.TrainBestMatches,
            self.TrainCadPoses,
            self.TrainScanPoses,
        ) = self.keys_to_label(self.train_keys, self.scene_dict)
        (
            self.val_files,
            self.ValBestMatches,
            self.ValCadPoses,
            self.ValScanPoses,
        ) = self.keys_to_label(self.val_keys, self.scene_dict)
        (
            self.test_files,
            self.TestBestMatches,
            self.TestCadPoses,
            self.TestScanPoses,
        ) = self.keys_to_label(self.test_keys, self.scene_dict)

    def get_split(self, split):
        if split == "train":
            return (
                self.train_files,
                self.TrainBestMatches,
                self.TrainCadPoses,
                self.TrainScanPoses,
            )
        elif split == "val":
            return (
                self.val_files,
                self.ValBestMatches,
                self.ValCadPoses,
                self.ValScanPoses,
            )
        elif split == "test":
            return (
                self.test_files,
                self.TestBestMatches,
                self.TestCadPoses,
                self.TestScanPoses,
            )
        else:
            raise ValueError("No such split")

    def print_stats(self):
        print("CADs:")
        print("\tnumber of CADs: {}".format(len(self.UsedObjPath)))

        print("Scans:")
        print(
            "\tnumber of scenes train: {}, val: {}, test: {}".format(
                len(self.train_keys), len(self.val_keys), len(self.test_keys)
            )
        )
        print(
            "\tnumber of objects train: {}, val: {}, test: {}".format(
                len(self.train_files), len(self.val_files), len(self.test_files)
            )
        )

    def filter_cad(
        self,
    ):
        """
        Use only specific catid
        """
        used_objId = []
        used_objPath = []
        for catId, objId in self.all_cad:
            if catId == self.catid:
                used_objId.append(objId)
                used_objPath.append(self.id2path[objId])

        return used_objId, used_objPath

    def get_scene_dict(self, files):
        """
        Get the dict of all the scene appeared.
        """
        scene_dict = {}
        for file in files:
            if not file[:12] in scene_dict.keys():
                scene_dict[file[:12]] = [file]
            else:
                scene_dict[file[:12]].append(file)

        return scene_dict

    def filter_scan(self):
        """
        Split the dataset according to scene_id.
        """

        keys = list(self.scene_dict.keys())
        keys.sort()

        # "split train val test here"

        train_and_val_keys = [key for key in keys if key in self.train_and_val_scans]
        test_keys = [key for key in keys if key in self.test_scans]

        train_keys = train_and_val_keys[: int(0.9 * (len(train_and_val_keys)))]
        val_keys = train_and_val_keys[int(0.9 * (len(train_and_val_keys))) :]

        return train_keys, val_keys, test_keys

    def keys_to_label(self, keys, scene_dict):
        """
        Given the scene, split the labels.
        """
        files = []
        for key in keys:
            files += scene_dict[key]

        BestMatches = [self.BestMatchesDict[file] for file in files]
        CadPoses = [self.CadPosesDict[file] for file in files]
        ScanPoses = [self.ScanPosesDict[file] for file in files]

        return files, BestMatches, CadPoses, ScanPoses

    def read_scannet_split(self, path_train, path_test):
        """
        Using scannet val set as our test set. Spliting scannet train set into our train set and our set.
        """

        with open(path_train) as f:
            train_and_val_scans = [line.strip("\n") for line in f.readlines()]

        with open(path_test) as f:
            test_scans = [line.strip("\n") for line in f.readlines()]

        return train_and_val_scans, test_scans

    def read_scans(self):
        script_dir = os.path.dirname(__file__)
        scannet_omit = os.path.join(script_dir, "../../configs/scannet_omit.txt")
        with open(scannet_omit, "r") as f:
            lines = f.readlines()
            omits = [line.strip("\n") for line in lines]

        files = os.listdir(self.scan_root)

        pcs = []
        BestMatches = {}
        CadPoses = {}
        ScanPoses = {}

        for file in files:
            if not file.endswith(".npy"):
                continue
            SceneID, NumModel, CatId, ModelId, _ = file.split(".")

            if not file in omits and CatId == self.catid:
                pcs.append(file)
                BestMatches[file] = ModelId
                CadPoses[file] = self.scan2pose[SceneID]["aligned_models"][int(NumModel)]["trs"]
                ScanPoses[file] = self.scan2pose[SceneID]["trs"]

        return pcs, BestMatches, CadPoses, ScanPoses

    def read_scan2pose(self):
        """
        Decompose annotation
        """
        scan2pose = {}
        for scan in self.annotation:
            scan2pose[scan["id_scan"]] = {
                "aligned_models": scan["aligned_models"],
                "trs": scan["trs"],
            }

        return scan2pose
