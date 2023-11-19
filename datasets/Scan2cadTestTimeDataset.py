from datasets.CategoryTestTimeDataset import *
from datasets.Reader import Scan2cadLibReader
from utils.preprocess import path_dict
from utils.read_json import load_csv


class Scan2cadTestTimeDataset(CategoryTestTimeDataset):
    def __init__(self, root, catid, scan2cad_dict, voxel_size):

        self.root = root
        self.catid = catid
        self.voxel_size = voxel_size
        self.id2path = path_dict(self.root)
        scan2cad_objs = load_csv(scan2cad_dict)

        self.ids = []
        for catId, objId in scan2cad_objs:
            if catId == self.catid:
                self.ids.append(objId)

        reader = Scan2cadLibReader(self.root, self.catid, self.ids, self.id2path, 10000)
        readerloader = torch.utils.data.DataLoader(
            reader, batch_size=1, shuffle=False, num_workers=8
        )
        pcs_ref = []
        for data in tqdm(readerloader):
            pcs_ref.append(data[0, :, :].numpy())
        self.pcs = pcs_ref
