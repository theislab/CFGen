import abc
import gc
from typing import Callable, Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import ContinuousBernoulli
from torchmetrics import ExplainedVariance, MeanSquaredError, MetricCollection
from celldreamer.models.base.base import _get_norm_layer, MLP


class BaseAutoEncoder(pl.LightningModule, abc.ABC):

    autoencoder: nn.Module  # autoencoder mapping von gene_dim to gene_dim

    def __init__(
            self,
            # fixed params
            gene_dim: int,
            feature_means: np.ndarray,
            # params from datamodule
            train_set_size: int,
            val_set_size: int,
            batch_size: int,
            # model specific params
            reconst_loss: str = 'continuous_bernoulli',
            learning_rate: float = 0.005,
            weight_decay: float = 0.1,
            optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
            lr_scheduler: Callable = None,
            lr_scheduler_kwargs: Dict = None,
            gc_frequency: int = 5
    ):
        super(BaseAutoEncoder, self).__init__()

        self.gene_dim = gene_dim
        self.train_set_size = train_set_size
        self.val_set_size = val_set_size
        self.batch_size = batch_size
        self.gc_freq = gc_frequency

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optim = optimizer
        self.reconst_loss = reconst_loss
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.register_buffer('feature_means', torch.tensor(feature_means))

        metrics = MetricCollection({
            'explained_var_weighted': ExplainedVariance(multioutput='variance_weighted'),
            'explained_var_uniform': ExplainedVariance(multioutput='uniform_average'),
            'mse': MeanSquaredError()
        })

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def _calc_reconstruction_loss(self, preds: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean'):
        if self.reconst_loss == 'continuous_bernoulli':
            loss = - ContinuousBernoulli(probs=preds).log_prob(targets)
            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'sum':
                loss = loss.sum()
        elif self.reconst_loss == 'bce':
            loss = F.binary_cross_entropy(preds, targets, reduction=reduction)
        elif self.reconst_loss == 'mae':
            loss = F.l1_loss(preds, targets, reduction=reduction)
        else:
            loss = F.mse_loss(preds, targets, reduction=reduction)
        return loss

    @abc.abstractmethod
    def _step(self, batch) -> (torch.Tensor, torch.Tensor):
        """Calculate predictions (int64 tensor) and loss"""
        pass

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch = batch[0]
        # zero center data
        # batch['X'] = batch['X'] - self.feature_means

        return batch

    def forward(self, batch):
        x_in = batch['X'] - self.feature_means  # zero center data
        # do not use covariates
        x_latent = self.encoder(x_in)
        x_reconst = self.decoder(x_latent)
        return x_latent, x_reconst

    def training_step(self, batch, batch_idx):
        preds, loss = self._step(batch)
        self.log('train_loss', loss, on_epoch=True)
        self.log_dict(self.train_metrics(preds, batch['X']), on_epoch=True)

        if batch_idx % self.gc_freq == 0:
            gc.collect()

        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss = self._step(batch)
        self.log('val_loss', loss)
        metrics = self.val_metrics(preds, batch['X'])
        self.log_dict(metrics)
        self.log('hp_metric', metrics['val_mse'])

        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def test_step(self, batch, batch_idx):
        preds, loss = self._step(batch)
        self.log('test_loss', loss)
        self.log_dict(self.test_metrics(preds, batch['X']))

        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def on_train_epoch_end(self) -> None:
        gc.collect()

    def on_validation_epoch_end(self) -> None:
        gc.collect()

    def configure_optimizers(self):
        optimizer_config = {'optimizer': self.optim(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)}
        if self.lr_scheduler is not None:
            lr_scheduler_kwargs = {} if self.lr_scheduler_kwargs is None else self.lr_scheduler_kwargs
            interval = lr_scheduler_kwargs.pop('interval', 'epoch')
            monitor = lr_scheduler_kwargs.pop('monitor', 'val_loss_epoch')
            frequency = lr_scheduler_kwargs.pop('frequency', 1)
            scheduler = self.lr_scheduler(optimizer_config['optimizer'], **lr_scheduler_kwargs)
            optimizer_config['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': interval,
                'monitor': monitor,
                'frequency': frequency
            }

        return optimizer_config


class MLP_AutoEncoder(BaseAutoEncoder):

    def __init__(
            self,
            # fixed params
            gene_dim: int,
            feature_means: np.ndarray,
            # params from datamodule
            train_set_size: int,
            val_set_size: int,
            batch_size: int,
            # model specific params
            units_encoder: List[int],
            units_decoder: List[int],
            batch_norm: bool = True,
            layer_norm: bool = False,
            activation: Callable[[], torch.nn.Module] = nn.SELU,
            output_activation: Callable[[], torch.nn.Module] = nn.Sigmoid,
            dropout: float = 0.2,
            learning_rate: float = 0.005,
            weight_decay: float = 0.1,
            optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
            lr_scheduler: Callable = None,
            lr_scheduler_kwargs: Dict = None,
    ):
        super(MLP_AutoEncoder, self).__init__(
            gene_dim=gene_dim,
            feature_means=feature_means,
            train_set_size=train_set_size,
            val_set_size=val_set_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs
        )
        # save hyperparameters
        self.save_hyperparameters(ignore=['feature_means'])

        # Define encoder network
        self.encoder = MLP(
            in_channels=gene_dim,
            hidden_channels=units_encoder,
            norm_layer=_get_norm_layer(batch_norm=batch_norm, layer_norm=layer_norm),
            activation_layer=activation,
            inplace=False,
            dropout=dropout
        )
        # Define decoder network
        self.decoder = nn.Sequential(
            MLP(
                in_channels=units_encoder[-1],
                hidden_channels=units_decoder + [gene_dim],
                norm_layer=_get_norm_layer(batch_norm=batch_norm, layer_norm=layer_norm),
                activation_layer=activation,
                inplace=False,
                dropout=dropout
            ),
            output_activation()
        )
        self.predict_bottleneck = False

    def _step(self, batch: Dict[str, torch.Tensor]):
        """
        Step function for training, validation and testing
        :param batch: batch of data
        :return: loss, predictions, targets
        """
        # print('batch keys', batch.keys())
        x_latent, x_reconst = self(batch)
        loss = self._calc_reconstruction_loss(x_reconst, batch['X'], reduction='mean')

        return x_reconst, loss
