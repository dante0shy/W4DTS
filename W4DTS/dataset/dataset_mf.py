

elastic_deformation = False
import MinkowskiEngine as ME

# print(ME.__file__)
import torch, numpy as np, glob, math
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod


class BaseDataset(Dataset, ABC):

    def __init__(self, data_base, batch_size, window_size, scale, mid_label=None, type='train', config = None,sv_dir = None):
        super(BaseDataset, self).__init__()

        self.type = type
        self.files = []
        self.data_base = data_base
        self.batch_size = batch_size
        self.window_size = window_size
        self.scale = scale
        self.config = config
        self.sv_dir = sv_dir
        self.mid_label = mid_label
        self.get_files(type)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]


    @abstractmethod
    def get_files(self,part):
        pass

    @abstractmethod
    def aug(self,a, seed):
        pass

    # @abstractmethod
    # def get_frames(self,tbl):
    #     pass

    @abstractmethod
    def get_data_loader(self,):
        pass

    @abstractmethod
    def train_consist(self, x):
        pass