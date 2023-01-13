# This is the BaseModel from Sfairazero
import abc
from typing import Callable, Dict, List, Optional
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from math import sqrt
import collections


def is_last_layer(i, units):
    return i == (len(units) - 2)


def _get_norm_layer(batch_norm: bool, layer_norm: bool):
    if batch_norm:
        norm_layer = nn.BatchNorm1d
    elif layer_norm:
        norm_layer = nn.LayerNorm
    else:
        norm_layer = None

    return norm_layer


class DenseResidualBlock(nn.Module):

    def __init__(
            self,
            n_features: int,
            activation: Callable[[], torch.nn.Module],
            gain_weight_init: float = sqrt(2.)
    ):
        super(DenseResidualBlock, self).__init__()
        self.linear1 = nn.Linear(n_features, n_features)
        nn.init.xavier_uniform_(self.linear1.weight, gain=gain_weight_init)
        self.activation = activation()

    def forward(self, x):
        return self.activation(self.linear1(x)) + x


class DenseLayerStack(nn.Module):

    def __init__(
            self,
            in_features: int,
            units: List[int],
            activation: Callable[[], torch.nn.Module],
            batch_norm: bool = False,
            layer_norm: bool = False,
            dropout: float = 0.,
            gain_weight_init: float = sqrt(2.),
            add_residual_blocks: bool = False
    ):
        super(DenseLayerStack, self).__init__()

        layers_dim = [in_features] + units
        layers = []
        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            cell = [nn.Linear(n_in, n_out)]
            if gain_weight_init is not None:
                nn.init.xavier_uniform_(cell[0].weight, gain=gain_weight_init)
            if batch_norm and not is_last_layer(i, layers_dim):
                cell.append(nn.BatchNorm1d(n_out, eps=1e-2))
            if layer_norm and not is_last_layer(i, layers_dim):
                cell.append(nn.LayerNorm(n_out))
            if not is_last_layer(i, layers_dim):
                cell.append(activation())
                if add_residual_blocks:
                    cell.append(DenseResidualBlock(n_out, activation, gain_weight_init))
            if (dropout > 0.) and not is_last_layer(i, layers_dim):
                cell.append(nn.Dropout(dropout))

            layers.append((f'Layer {i}', nn.Sequential(*cell)))

        self.layers = nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.
    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    Implementation slightly adapted from https://pytorch.org/vision/main/generated/torchvision.ops.MLP.html
    (removed Dropout from last layer + log_api_usage call)
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: List[int],
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            inplace: Optional[bool] = True,
            bias: bool = True,
            dropout: float = 0.0,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim, eps=0.001))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))

        super().__init__(*layers)


class BaseModel(pl.LightningModule, abc.ABC):

    def __init__(
            self,
            gene_dim: int,
            feature_normalizations: List[str],
            feature_means: np.ndarray = None,
            learning_rate: float = 0.01,
            weight_decay: float = 0.01,
            optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
            lr_scheduler: Callable = None,
            lr_scheduler_kwargs: Dict = None,
            log_param_histograms: bool = False
    ):
        super(BaseModel, self).__init__()
        for norm in feature_normalizations:
            if norm not in ['log1p', 'zero_center', 'none']:
                raise ValueError(
                    f"Freature normalizations have to be in ['log1p', 'zero_center', 'none']. "
                    f"You supplied: {norm}"
                )
        if 'zero_center' in feature_normalizations:
            if feature_means is None:
                raise ValueError('You need to supply feature_means to use "zero_center" normalization')
            if not feature_means.shape == (1, gene_dim):
                raise ValueError('Shape of feature_means has to be (1, gene_dim)')
            self.register_buffer('feature_means', torch.tensor(feature_means))

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.log_param_histogram = log_param_histograms

    def get_normalized_counts(self, x, normalization: str) -> torch.Tensor:
        if normalization == 'log1p':
            x_normed = torch.log1p(x['X'])
        elif normalization == 'zero_center':
            x_normed = x['X'] - self.feature_means
        else:
            x_normed = x['X']

        return x_normed

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if ('labels' in batch) and (batch['labels'].shape[1] == 1):
            batch['labels'] = torch.squeeze(batch['labels'])
        if 'batch' in batch:
            batch['batch'] = torch.squeeze(batch['batch'])
        if 'assay_sc' in batch:
            batch['assay_sc'] = torch.squeeze(batch['assay_sc'])
        if 'organ' in batch:
            batch['organ'] = torch.squeeze(batch['organ'])

        return batch

    def training_epoch_end(self, outputs):
        if self.log_param_histogram:
            for name, params in self.named_parameters():
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        optimizer_config = {'optimizer': self.optim(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)}
        if self.lr_scheduler is not None:
            lr_scheduler_kwargs = {} if self.lr_scheduler_kwargs is None else self.lr_scheduler_kwargs
            interval = lr_scheduler_kwargs.pop('interval', 'epoch')
            monitor = lr_scheduler_kwargs.pop('monitor', 'train_loss_epoch')
            frequency = lr_scheduler_kwargs.pop('frequency', 1)
            scheduler = self.lr_scheduler(optimizer_config['optimizer'], **lr_scheduler_kwargs)
            optimizer_config['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': interval,
                'monitor': monitor,
                'frequency': frequency
            }

        return optimizer_config
