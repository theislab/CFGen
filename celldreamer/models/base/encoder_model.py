import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from scvi.distributions import NegativeBinomial
from torch.distributions import Poisson, Bernoulli
from celldreamer.models.base.utils import MLP
import time


class EncoderModel(pl.LightningModule):
    """
    PyTorch Lightning Module for an encoder-decoder model.

    Args:
        in_dim (dict): Dictionary specifying the input dimensions for each modality.
        x0_from_x_kwargs (dict): If multimodal, dictionary with arguments, one per effect.
        scaler (dict): If multimodal, dictionary with one scaler per modality.
        learning_rate (float): Learning rate for optimization.
        weight_decay (float): Weight decay for optimization.
        covariate_specific_theta (bool): Flag indicating whether theta is specific to covariates.
        encoder_type (str): Type of encoder, either 'learnt_encoder' or 'learnt_autoencoder'.
        conditioning_covariate (str): Covariate used for conditioning.
        n_cat (int): Number of categories for the theta parameter.
        multimodal (bool): Flag indicating whether the model is multimodal.
        is_binarized (bool): Flag indicating whether the input data is binarized.

    Methods:
        training_step: Executes a training step.
        validation_step: Executes a validation step.
        configure_optimizers: Configures the optimizer for training.
        encode: Encodes input data.
        decode: Decodes encoded data.

    """
    def __init__(self,
                 in_dim,
                 x0_from_x_kwargs,
                 learning_rate,
                 weight_decay,
                 covariate_specific_theta,
                 encoder_type,
                 conditioning_covariate,
                 n_cat=None,
                 multimodal=False,
                 is_binarized=False, 
                 encoder_multimodal_joint_layers=None,
                 ):
        """
        Initializes the EncoderModel.
        """

        super().__init__()
        # Input dimension
        self.in_dim = in_dim

        # Initialize attributes 
        self.x0_from_x_kwargs = x0_from_x_kwargs  # if multimodal, dictionary with arguments, one per effect
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.covariate_specific_theta = covariate_specific_theta
        self.encoder_type = encoder_type
        self.conditioning_covariate = conditioning_covariate
        self.n_cat = n_cat
        self.multimodal = multimodal
        self.is_binarized = is_binarized
        # Joint into a single latent space or not 
        self.encoder_multimodal_joint_layers = encoder_multimodal_joint_layers
        if self.encoder_multimodal_joint_layers:
            self.x0_from_x_joint = None

        # List of modalities present in the data 
        if multimodal:
            self.modality_list = list(self.x0_from_x_kwargs.keys())

        # Theta for the negative binomial parameterization of scRNA-seq
        in_dim_rna = self.in_dim if not self.multimodal else self.in_dim["rna"]
        if not covariate_specific_theta:
            self.theta = torch.nn.Parameter(torch.randn(in_dim_rna), requires_grad=True)
        else:
            self.theta = torch.nn.Parameter(torch.randn(n_cat, in_dim_rna), requires_grad=True)

        # Initialize all the metrics
        if encoder_type == "learnt_encoder":
            if not self.multimodal:
                x0_from_x_kwargs["dims"] = [self.in_dim, *x0_from_x_kwargs["dims"], self.in_dim]
                self.x0_from_x = MLP(**x0_from_x_kwargs)
            else:
                raise NotImplementedError
        else:
            if not self.multimodal:
                x0_from_x_kwargs["dims"] = [self.in_dim, *x0_from_x_kwargs["dims"]]  # Encoder params
                self.x0_from_x = MLP(**x0_from_x_kwargs)
                x0_from_x_kwargs["dims"] = x0_from_x_kwargs["dims"][::-1]  # Decoder params
                self.x_from_x0 = MLP(**x0_from_x_kwargs)
            else:
                # Modality specific part 
                self.x0_from_x = {}
                self.x_from_x0 = {}
                for mod in self.modality_list:
                    x0_from_x_kwargs[mod]["dims"] = [self.in_dim[mod], *x0_from_x_kwargs[mod]["dims"]]
                    self.x0_from_x[mod] = MLP(**x0_from_x_kwargs[mod])
                    if self.encoder_multimodal_joint_layers:
                        x0_from_x_kwargs[mod]["dims"].append(self.encoder_multimodal_joint_layers["dims"][-1])
                    x0_from_x_kwargs[mod]["dims"] = x0_from_x_kwargs[mod]["dims"][::-1]
                    self.x_from_x0[mod] = MLP(**x0_from_x_kwargs[mod])
                self.x0_from_x = torch.nn.ModuleDict(self.x0_from_x)
                self.x_from_x0 = torch.nn.ModuleDict(self.x_from_x0)
                
                # Shared modality part in the encoder 
                if self.encoder_multimodal_joint_layers:
                    joint_inputs = sum([x0_from_x_kwargs[mod]["dims"][0] for mod in self.modality_list])
                    self.encoder_multimodal_joint_layers["dims"] = [joint_inputs, *self.encoder_multimodal_joint_layers["dims"]]
                    self.x0_from_x_joint = MLP(**self.encoder_multimodal_joint_layers)

        self.save_hyperparameters()

    def _step(self, batch, dataset_type):
        """
        Executes a single step of training or validation.

        Args:
            batch (dict): Batch of input data.
            dataset_type (str): Type of dataset, either 'train' or 'valid'.

        Returns:
            loss (tensor): Loss value for the step.

        """
        if not self.multimodal:
            X = batch["X"].to(self.device)
            size_factor = X.sum(1).unsqueeze(1).to(self.device)
        else:
            X = {mod: batch["X"][mod].to(self.device) for mod in batch["X"]}
            size_factor = {}
            for mod in X:
                size_factor_mod = X[mod].sum(1).unsqueeze(1).to(self.device)
                size_factor[mod] = size_factor_mod

        # Conditioning covariate encodings
        y = batch["y"][self.conditioning_covariate].to(self.device)

        # Make the encoding multimodal
        z = self.encode(batch)
        mu_hat = self.decode(z, size_factor)

        # Compute the negative log-likelihood of the data under the model
        if not self.multimodal:
            if not self.covariate_specific_theta:
                px = NegativeBinomial(mu=mu_hat, theta=torch.exp(self.theta))
            else:
                px = NegativeBinomial(mu=mu_hat, theta=torch.exp(self.theta[y]))
            loss = - px.log_prob(X).sum(1).mean()
        else:
            loss = 0
            for mod in mu_hat:
                if mod == "rna":
                    # Negative Binomial log-likelihood
                    if not self.covariate_specific_theta:
                        px = NegativeBinomial(mu=mu_hat[mod], theta=torch.exp(self.theta))
                    else:
                        px = NegativeBinomial(mu=mu_hat[mod], theta=torch.exp(self.theta[y]))
                elif mod == "atac":
                    if not self.is_binarized:
                        px = Poisson(rate=mu_hat[mod])
                    else:
                        px = Bernoulli(probs=mu_hat[mod])
                else:
                    raise NotImplementedError
                loss -= px.log_prob(X[mod]).sum(1).mean()

        self.log(f'{dataset_type}/loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Executes a training step.

        Args:
            batch (dict): Batch of input data.
            batch_idx (int): Index of the current batch.

        Returns:
            loss (tensor): Loss value for the step.

        """
        loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Executes a validation step.

        Args:
            batch (dict): Batch of input data.
            batch_idx (int): Index of the current batch.

        Returns:
            loss (tensor): Loss value for the step.

        """
        loss = self._step(batch, "valid")
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            optimizer (torch.optim.Optimizer): Optimizer for training.

        """
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.learning_rate,
                                 weight_decay=self.weight_decay)

    def encode(self, batch):
        """
        Encodes input data.

        Args:
            batch (dict): Batch of input data.

        Returns:
            z (tensor or dict): Encoded data.

        """
        if not self.multimodal:
            return self.x0_from_x(batch["X_norm"].to(self.device))
        else:
            z = {}
            for mod in self.modality_list:
                z_mod = self.x0_from_x[mod](batch["X_norm"][mod].to(self.device))
                z[mod] = z_mod
                
            # Implement joint layers if defined
            if self.encoder_multimodal_joint_layers:
                z_joint = torch.cat([z[mod] for mod in z], dim=-1)
                z = self.x0_from_x_joint(z_joint)     
            return z

    def decode(self, x, size_factor):
        """
        Decodes encoded data.

        Args:
            x (tensor or dict): Encoded data.
            size_factor (tensor or dict): Size factor.

        Returns:
            mu_hat (tensor or dict): Decoded data.

        """
        if not self.multimodal:
            if self.encoder_type == "learnt_autoencoder":
                x = self.x_from_x0(x)
            mu_hat = F.softmax(x, dim=1)    
            mu_hat = mu_hat * size_factor  # assume single modality is RNA
        else:
            mu_hat = {}
            for mod in self.modality_list:
                if self.encoder_type == "learnt_autoencoder":
                    if not self.encoder_multimodal_joint_layers:
                        x_mod = self.x_from_x0[mod](x[mod])
                    else:
                        x_mod = self.x_from_x0[mod](x)
                else:
                    raise NotImplementedError
                
                if mod != "atac" or (mod == "atac" and not self.is_binarized):
                    mu_hat_mod = F.softmax(x_mod, dim=1)  # for Poisson counts the parameterization is similar to RNA 
                    mu_hat_mod = mu_hat_mod * size_factor[mod]
                else:
                    mu_hat_mod = F.sigmoid(x_mod)
                mu_hat[mod] = mu_hat_mod
        return mu_hat
