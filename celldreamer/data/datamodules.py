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

    def valid_dataloader(self):
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
        self.batch_size = batch_size
        if not path:  # use default path
            # get root directory that is two levels above this file's directory
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.path = join(root, 'data', 'hlca')
        else:
            self.path = path

        train_adata = sc.read_h5ad(join(self.path, 'train_adata.h5ad'))

        self.train_dataset = CustomDataset(torch.tensor(train_adata.X.todense()), train_adata.obs['ann_level_5'].values)
        val_adata = sc.read_h5ad(join(self.path, 'val_adata.h5ad'))
        self.val_dataset = CustomDataset(torch.tensor(val_adata.X.todense()), val_adata.obs['ann_level_5'].values)
        test_adata = sc.read_h5ad(join(self.path, 'test_adata.h5ad'))
        self.test_dataset = CustomDataset(torch.tensor(test_adata.X.todense()), test_adata.obs['ann_level_5'].values)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def valid_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class CustomDataset(Dataset):

    def __init__(self, x, obs=None):
        super(CustomDataset).__init__()
        assert any([isinstance(x, np.ndarray), isinstance(x, csr_matrix), isinstance(x, torch.Tensor)])
        self.x = x
        self.obs = obs

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        x = self.x[idx, :]
        if isinstance(x, csr_matrix):
            x = x.toarray()

        if self.obs is not None:
            # replicate merlin dataloader output format
            out = (
                {
                    'X': x.squeeze(),
                    'idx': self.obs.iloc[idx]['idx'].to_numpy().reshape((-1, 1)),
                    'cell_type': self.obs.iloc[idx]['cell_type'].to_numpy().reshape((-1, 1))
                }, None
            )
        else:
            out = ({'X': x.squeeze()}, None)

        return out