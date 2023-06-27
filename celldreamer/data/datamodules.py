# Adapted from https://github.com/theislab/cellnet/blob/main/cellnet/datamodules.py
import os.path
from os.path import join
import lightning.pytorch as pl
import scanpy as sc
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from celldreamer.data.toy.shape_color import ShapeColorDataset
from celldreamer.data.hlca.hlca import HLCADataset


class ShapeColorDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_samples: int):
        """
        Lightning Datamodule for the toy dataset ShapeColorDataset
        :param batch_size: int, batch size
        :param num_samples: int, number of samples to generate
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.dataset = ShapeColorDataset(self.num_samples)
        # train val test split
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        # set random seed
        torch.manual_seed(0)
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset,
                                                                                                [train_size,
                                                                                                 val_size,
                                                                                                 test_size]
                                                                                                )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class HLCADataModule(pl.LightningDataModule):
    """
    Base class for HLCA data module, load train, val and test adata into memory.
    """
    def __init__(self,
                 path: str = None,
                 batch_size: int = 4096):
        super(HLCADataModule, self).__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        if not path:  # use default path
            # get root directory that is two levels above this file's directory
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.path = join(root, 'data', 'hlca')
        else:
            self.path = path

        # self.train_dataset = HLCADataset(adata_path=self.path, mode='train')
        # self.val_dataset = HLCADataset(adata_path=self.path, mode='val')
        # self.test_dataset = HLCADataset(adata_path=self.path, mode='test')

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = HLCADataset(adata_path=self.path, mode='train')
            self.val_dataset = HLCADataset(adata_path=self.path, mode='val')
        elif stage == "test":
            self.test_dataset = HLCADataset(adata_path=self.path, mode='test')
        else:
            raise ValueError("Stage must be either fit or test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
