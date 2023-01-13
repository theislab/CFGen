from typing import Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import ContinuousBernoulli
from torchmetrics import ExplainedVariance, MeanSquaredError, MetricCollection
from scgm.models.generative_models.base import BaseModel, _get_norm_layer, MLP


class AutoEncoder(BaseModel):

    def __init__(
            self,
            # fixed params
            feature_means: np.ndarray,
            gene_dim: int,
            n_batches: int,
            n_organs: int,
            n_assay_sc: int,
            # params from datamodule
            train_set_size: int,
            val_set_size: int,
            batch_size: int,
            # model specific hparams
            units_encoder: List[int],
            units_decoder: List[int],
            embedding_dim_batch: Optional[int] = 25,
            encode_organ: bool = False,
            encode_assay_sc: bool = False,
            encode_tech_sample: bool = False,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.001,
            activation: Callable[[], torch.nn.Module] = nn.SELU,
            batch_norm: bool = True,
            layer_norm: bool = False,
            dropout: float = 0.2,
            output_activation: Callable[[], torch.nn.Module] = nn.Sigmoid,
            normalize_input: str = 'zero_center',
            normalize_output: str = 'none',
            reconstruction_loss: str = 'continuous_bernoulli',
            masking_rate: Optional[float] = None,
            optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
            lr_scheduler: Optional[Callable] = None,
            lr_scheduler_kwargs: Optional[Dict] = None,
            log_param_histograms: bool = False,
            gain_weight_init: float = 0.75,
            gain_weight_init_first_layer: float = 4.
    ):
        # check input
        assert 0. <= dropout <= 1.
        assert reconstruction_loss in ['mse', 'mae', 'continuous_bernoulli', 'bce']
        if reconstruction_loss in ['continuous_bernoulli', 'bce']:
            assert output_activation == nn.Sigmoid
            assert normalize_output == 'none'

        self.train_set_size = train_set_size
        self.val_set_size = val_set_size
        self.batch_size = batch_size

        super(AutoEncoder, self).__init__(
            gene_dim=gene_dim,
            feature_normalizations=[normalize_input, normalize_output],
            feature_means=feature_means,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            log_param_histograms=log_param_histograms
        )
        self.save_hyperparameters(ignore=['feature_means', 'gene_dim', 'n_batches', 'n_organs', 'n_assay_sc'])

        self.gene_dim = gene_dim
        self.encode_organ = encode_organ
        self.n_organ = n_organs
        self.n_assay_sc = n_assay_sc
        self.encode_assay_sc = encode_assay_sc
        self.encode_tech_sample = encode_tech_sample
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.reconst_loss = reconstruction_loss
        self.masking_rate = masking_rate

        # Define embedding layer for batches
        if embedding_dim_batch:
            self.batch_embedding = torch.nn.Embedding(n_batches, embedding_dim_batch)
        else:
            self.batch_embedding = None
        # Define encoder network
        in_features_encoder = (gene_dim +
                               self.encode_tech_sample * embedding_dim_batch +
                               self.encode_organ * n_organs +
                               self.encode_assay_sc * n_assay_sc)
        self.encoder = MLP(
            in_channels=in_features_encoder,
            hidden_channels=units_encoder,
            norm_layer=_get_norm_layer(batch_norm=batch_norm, layer_norm=layer_norm),
            activation_layer=activation,
            inplace=False,
            dropout=dropout
        )
        # Define decoder network
        in_features_decoder = units_encoder[-1] + embedding_dim_batch * (n_batches > 0)
        in_features_decoder += self.encode_organ * n_organs + self.encode_assay_sc * n_assay_sc
        self.decoder = nn.Sequential(
            MLP(
                in_channels=in_features_decoder,
                hidden_channels=units_decoder + [gene_dim],
                norm_layer=_get_norm_layer(batch_norm=batch_norm, layer_norm=layer_norm),
                activation_layer=activation,
                inplace=False,
                dropout=dropout
            ),
            output_activation()
        )
        self._init_weights(gain_weight_init_first_layer=gain_weight_init_first_layer, gain_weight_init=gain_weight_init)

        metrics = MetricCollection({
            'explained_var_weighted': ExplainedVariance(multioutput='variance_weighted'),
            'explained_var_uniform': ExplainedVariance(multioutput='uniform_average'),
            'mse': MeanSquaredError()
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def _init_weights(self, gain_weight_init: float, gain_weight_init_first_layer: float):
        for ix, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                if ix == 0:
                    nn.init.xavier_uniform_(layer.weight, gain=gain_weight_init_first_layer)
                else:
                    nn.init.xavier_uniform_(layer.weight, gain=gain_weight_init)

        for ix, layer in enumerate(self.decoder[0]):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain_weight_init)

    def forward(self, x, batch, assay_sc, organ):
        covariates = tuple()
        if self.batch_embedding:
            batch_embedding = self.batch_embedding(batch)
            if self.encode_tech_sample:
                covariates += (batch_embedding,)
        if self.encode_assay_sc:
            covariates += (F.one_hot(assay_sc, num_classes=self.n_assay_sc),)
        if self.encode_organ:
            covariates += (F.one_hot(organ, num_classes=self.n_organ),)
        x_in = torch.cat((x,) + covariates, dim=1)
        x_latent = self.encoder(x_in)
        # add batch embedding if not already added above
        if (self.batch_embedding is not None) and (not self.encode_tech_sample):
            covariates += (batch_embedding,)
        x_reconst = self.decoder(torch.cat((x_latent,) + covariates, dim=1))

        return x_latent, x_reconst

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

    def _step(self, batch: Dict[str, torch.Tensor]):
        targets = self.get_normalized_counts(batch, self.normalize_output)
        inputs = self.get_normalized_counts(batch, self.normalize_input)
        print('Batch: ', batch)
        x_latent, x_reconst = self(inputs, batch['batch'], batch['assay_sc'], batch['organ'])
        loss = self._calc_reconstruction_loss(x_reconst, targets, reduction='mean')

        return loss, x_reconst, targets

    def training_step(self, batch, batch_idx):
        loss, x_reconst, targets = self._step(batch)
        self.log_dict(self.train_metrics(x_reconst, targets), on_epoch=True, on_step=True)
        self.log('train_loss', loss, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, x_reconst, targets = self._step(batch)
        self.log_dict(self.val_metrics(x_reconst, targets))
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss, x_reconst, targets = self._step(batch)
        self.log_dict(self.test_metrics(x_reconst, targets))
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x_latent, _ = self(batch)
        return x_latent
