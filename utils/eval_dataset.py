import os, random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.get_yaml import const


class EvaluationDataset(Dataset):
    def __init__(self):
        self.eval_folder = os.path.abspath(const['io']['eval'])
        self.dataset = []

        self.initialize()

    def __getitem__(self, item):
        x, name = self.dataset[item]
        return self.read_file(x), name

    def __len__(self):
        return len(self.dataset)

    def initialize(self):
        self.load_dataset()

    def load_dataset(self):
        for name_qz in os.listdir(self.eval_folder):
            path_qz = os.path.join(self.eval_folder, name_qz)
            name = name_qz.split('_')[0]
            if os.path.exists(path_qz):
                self.dataset.append((path_qz, name))

    @staticmethod
    def read_file(path):
        numbers = np.loadtxt(path)
        numbers = torch.from_numpy(numbers).float()
        return numbers


class EvaluationDataset_hdf5(Dataset):
    def __init__(self):
        self.setup_random_seed(123)
        self.eval_folder = os.path.abspath(const['io']['eval_input'])
        self.eval_folder_l = os.path.abspath(const['io']['eval_label'])
        self.dataset = []

        self.initialize()
        # import pdb
        # pdb.set_trace()

    @staticmethod
    def norm(x, y):
        return x/np.max(np.abs(x)), y/ np.max(np.abs(y))

    @staticmethod
    def setup_random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def __getitem__(self, item):
        x, y = self.dataset[item][..., 0], self.dataset[item][..., 1]
        # x,y = self.norm(x,y)
        x = y + x*random.randint(40,65) * 0.01
        # y = self.read_file(y)
        # y = self.norm(y, None)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), item
        # x, name = self.dataset[item]
        # return self.read_file(x), name

    def __len__(self):
        return len(self.dataset)

    def initialize(self):
        self.load_dataset()

    def load_dataset(self):
        self.dataset = self.load_h5py_dataset(self.eval_folder, self.eval_folder_l)

        # for name_qz in os.listdir(self.eval_folder):
        #     path_qz = os.path.join(self.eval_folder, name_qz)
        #     name = name_qz.split('_')[0]
        #     if os.path.exists(path_qz):
        #         self.dataset.append((path_qz, name))

    @staticmethod
    def read_file(path):
        numbers = np.loadtxt(path)
        numbers = torch.from_numpy(numbers).float()
        return numbers

    def load_h5py_dataset(self, file, label_f):
        # import pdb
        # pdb.set_trace()
        with h5py.File(file, "r") as f:
            data = f['traces'][:]
            idx_i = f['metadata'][:]
        with h5py.File(label_f, "r") as f:
            label = f['traces'][:]
            idx_l = f['metadata'][:]

        dataset = []
        for i in range(len(data)):
            x, y = data[i, ...], label[i, ...]
            x,y = self.norm(x,y)
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            dataset.append([x , y])
        # import pdb
        # pdb.set_trace()
        return np.array(dataset).transpose((0, 3, 2, 1))
        # self.dataset = np.stack([data, label],axis=3).transpose((0, 2, 1, 3))