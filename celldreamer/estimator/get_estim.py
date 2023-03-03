# Adapted from:
# https://github.com/theislab/cellnet/blob/main/cellnet/estimators.py
from os.path import join
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from celldreamer.models.base.utils import MLP
from celldreamer.models.diffusion.classifier_free_ddpm import GaussianDDPMClassifierFreeGuidance


class EstimatorCellDreamer:
    datamodule: MerlinDataModule
    model: pl.LightningModule
    trainer: pl.Trainer

    def __init__(self, data_path: str):
        self.data_path = data_path

    def init_datamodule(
            self,
            batch_size: int = 2048,
            dataset_kwargs_train: Dict = None,
            dataset_kwargs_inference: Dict = None
    ):
        if 'merlin' in self.data_path:
            self.datamodule = MerlinDataModule(
                self.data_path,
                columns=['cell_type'],
                batch_size=batch_size,
                drop_last=True,
                merlin_dataset_kwargs_train=dataset_kwargs_train,
                merlin_dataset_kwargs_inference=dataset_kwargs_inference
            )
        elif 'chemCPA' in self.data_path:
            raise NotImplementedError

    def init_model(self, generative_model: str, denoiser_module: str, model_kwargs):
        if generative_model == 'diffusion':
            if denoiser_module == 'mlp':
                # MLP is a class that inherits from torch.nn.Module
                denoiser_module = MLP(**model_kwargs.pop('denoiser_module_kwargs'))
                self.model = GaussianDDPMClassifierFreeGuidance(
                    denoiser_module=denoiser_module,
                    **model_kwargs  # model_kwargs should contain the rest of the arguments
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def init_trainer(self, trainer_kwargs):
        self.trainer = pl.Trainer(**trainer_kwargs)

    def _check_is_initialized(self):
        if not self.model:
            raise RuntimeError('You need to call self.init_model before calling self.train')
        if not self.datamodule:
            raise RuntimeError('You need to call self.init_datamodule before calling self.train')
        if not self.trainer:
            raise RuntimeError('You need to call self.init_trainer before calling self.train')

    def get_fixed_rna_model_params(self):
        return {
            'gene_dim': len(pd.read_parquet(join(self.data_path, 'var.parquet'))),
            'feature_means': np.load(join(self.data_path, 'norm/zero_centering/means.npy')),
            'train_set_size': sum(self.datamodule.train_dataset.partition_lens),
            'val_set_size': sum(self.datamodule.val_dataset.partition_lens),
            'batch_size': self.datamodule.batch_size,
        }

    def find_lr(self, lr_find_kwargs, plot_results: bool = False):
        self._check_is_initialized()
        lr_finder = self.trainer.tuner.lr_find(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(),
            val_dataloaders=self.datamodule.val_dataloader(),
            **lr_find_kwargs
        )
        if plot_results:
            lr_finder.plot(suggest=True)

        return lr_finder.suggestion(), lr_finder.results

    def train(self, ckpt_path: str = None):
        self._check_is_initialized()
        self.trainer.fit(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(),
            val_dataloaders=self.datamodule.val_dataloader(),
            ckpt_path=ckpt_path
        )

    def validate(self, ckpt_path: str = None):
        self._check_is_initialized()
        return self.trainer.validate(self.model, dataloaders=self.datamodule.val_dataloader(), ckpt_path=ckpt_path)

    def test(self, ckpt_path: str = None):
        self._check_is_initialized()
        return self.trainer.test(self.model, dataloaders=self.datamodule.test_dataloader(), ckpt_path=ckpt_path)

    def generate(self, batch_size: int, attributes: List[int], ckpt_path: str = None) -> torch.Tensor:
        """
        To Do: Generate samples from the model.
        :param batch_size: Number of samples to generate.
        :param attributes: List of attributes to generate samples for.
        :param ckpt_path: Path to checkpoint to load model from.
        :return: Tensor of shape (batch_size, gene_dim).
        """
        self._check_is_initialized()
        return self.model.generate(batch_size, attributes, ckpt_path=ckpt_path)

    def predict(self, ckpt_path: str = None) -> np.ndarray:
        self._check_is_initialized()
        predictions_batched: List[torch.Tensor] = self.trainer.predict(
            self.model,
            dataloaders=self.datamodule.predict_dataloader(),
            ckpt_path=ckpt_path
        )
        return torch.vstack(predictions_batched).numpy()
