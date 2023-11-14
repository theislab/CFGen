from typing import Literal
import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch import sigmoid, sqrt, autograd, linspace
from torch.special import expm1
import torch.nn.functional as F
import pytorch_lightning as pl

from tqdm import trange

from scvi.distributions import NegativeBinomial

from celldreamer.models.vdm.variance_scheduling import FixedLinearSchedule, LearnedLinearSchedule
from celldreamer.models.base.utils import MLP, unsqueeze_right, kl_std_normal
from celldreamer.eval.evaluate import compute_umap_and_wasserstein

class VDM(pl.LightningModule):
    def __init__(self,
                 denoising_model: nn.Module,
                 feature_embeddings: dict, 
                 z0_from_x_kwargs: dict,
                 plotting_folder: Path,
                 in_dim: int,
                 learning_rate: float = 0.001, 
                 weight_decay: float = 0.0001, 
                 noise_schedule: str = "fixed_linear", 
                 gamma_min: float = -5.0, 
                 gamma_max: float = 1.0,
                 antithetic_time_sampling: bool = True, 
                 scaling_method: int = "log_normalization",
                 train_library_size: bool = False, 
                 generative_library_size: float = 1000):
        """
        Variational Diffusion Model (VDM).

        Args:
            denoising_model (nn.Module): Denoising model.
            feature_embeddings (dict): Feature embeddings for covariates.
            z0_from_x_kwargs (dict): Arguments for the z0_from_x MLP.
            plotting_folder (Path): Folder for saving plots.
            in_dim (int): Number of genes.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            weight_decay (float, optional): Weight decay. Defaults to 0.0001.
            noise_schedule (str, optional): Noise schedule type. Defaults to "fixed_linear".
            gamma_min (float, optional): Minimum value for gamma in noise schedule. Defaults to -5.0.
            gamma_max (float, optional): Maximum value for gamma in noise schedule. Defaults to 1.0.
            antithetic_time_sampling (bool, optional): Use antithetic time sampling. Defaults to True.
            scaling_method (int, optional): Scaling method for input data. Defaults to "log_normalization".
            train_library_size (bool, optional): Whether the library size should be trained with the densoising.
            generative_library_size (float, optional): library size for generated samples. 
        """
        super().__init__()
        
        self.denoising_model = denoising_model.to(self.device)
        self.feature_embeddings = feature_embeddings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.in_dim = in_dim
        self.antithetic_time_sampling = antithetic_time_sampling
        self.scaling_method = scaling_method
        self.plotting_folder = plotting_folder
        self.train_library_size = train_library_size
        self.generative_library_size = generative_library_size
        
        # Used to collect test outputs
        self.testing_ouputs = []
        
        # Define a dimensionality-preserving encoder to optimize jointly with the diffusion model
        z0_from_x_kwargs["dims"] = [self.in_dim, *z0_from_x_kwargs["dims"], self.in_dim]
        self.z0_from_x = MLP(**z0_from_x_kwargs)
        
        # If we train library size, design a library size encoder
        if self.train_library_size:
            library_size_enc_kwargs = z0_from_x_kwargs.copy()
            library_size_enc_kwargs["dims"] = [self.in_dim, *z0_from_x_kwargs["dims"], 1]
            self.library_size_enc = MLP(**library_size_enc_kwargs)
        
        # Define the inverse dispersion parameter (negative binomial)
        self.theta = torch.nn.Parameter(torch.randn(self.in_dim))
            
        assert noise_schedule in ["fixed_linear", "learned_linear"]
        if noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(gamma_min, gamma_max)
        elif noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(gamma_min, gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {noise_schedule}")
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):
        """
        Training step for VDM.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        return self._step(batch, dataset='train')

    def validation_step(self, batch, batch_idx):
        """
        Validation step for VDM.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        return self._step(batch, dataset='valid')

    
    def _step(self, batch, dataset: Literal['train', 'valid']):
        """
        Common step for training and validation.

        Args:
            batch: Batch data.
            dataset (Literal['train', 'valid']): Dataset type.

        Returns:
            torch.Tensor: Loss value.
        """
        # Collect observation
        x = batch["X"].to(self.device)
        # Scale batch to reasonable range 
        x_scaled = self._scale_batch(x)
        z0 = self.z0_from_x(x_scaled)
        # Quantify library size 
        if self.train_library_size:
            library_size = x.sum(1).unsqueeze(1)
        else:
            library_size = torch.exp(self.library_size_enc(z0))
        
        # Collect concatenated labels
        y = self._featurize_batch_y(batch)  #TODO: For now we don't implement conditional version
        
        # Sample time 
        times = self._sample_times(z0.shape[0]).requires_grad_(True)
        
        # Sample noise 
        noise = torch.rand_like(z0)
        
        # Sample an observation undergoing noising 
        z_t, gamma_t, gamma_t_padded = self.sample_q_t_0(x=z0, times=times, noise=noise)
        
        # Forward through the model 
        model_out = self.denoising_model(z_t, gamma_t_padded)
        
        # Diffusion loss
        gamma_grad = autograd.grad(  # gamma_grad shape: (B, )
            gamma_t,  # (B, )
            times,  # (B, )
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        pred_loss = ((model_out - noise) ** 2).sum(1)  # (B, )
        diffusion_loss = 0.5 * pred_loss * gamma_grad 

        # Latent loss
        gamma_1 = self.gamma(torch.tensor([1.0], device=self.device))
        sigma_1_sq = sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * z0**2  # (alpha_1 * x)**2
        latent_loss = kl_std_normal(mean_sq, sigma_1_sq).sum(1) # (B, )
        
        # Compute log p(x | z_0) for all possible values of each pixel in x.
        recons_loss = self.log_probs_x_z0(x, z0, library_size)  
        recons_loss = recons_loss.sum(1)
        
        # Total loss    
        loss = diffusion_loss + latent_loss + recons_loss

        with torch.no_grad():
            gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        
        # Save results
        metrics = {
            f"{dataset}/loss": loss.mean(),
            f"{dataset}/diff_loss": diffusion_loss.mean(),
            f"{dataset}/latent_loss": latent_loss.mean(),
            f"{dataset}/loss_recon": recons_loss.mean(),
            f"{dataset}/gamma_0": gamma_0.item(),
            f"{dataset}/gamma_1": gamma_1.item(),
        }
        self.log_dict(metrics, prog_bar=True)
        return loss.mean()

    # Private methods
    def _featurize_batch_y(self, batch):
        """
        Featurize all the covariates 

        Args:
            batch: Batch data.

        Returns:
            torch.Tensor: Featurized covariates.
        """
        y = []     
        for feature_cat in batch["y"]:
            y_cat = self.feature_embeddings[feature_cat](batch["y"][feature_cat])
            y.append(y_cat)
        y = torch.cat(y, dim=1).to(self.device)
        return y
    
    def _sample_times(self, batch_size):
        """
        Sample times, can be sampled to cover the 

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Sampled times.
        """
        if self.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
        else:
            times = torch.rand(batch_size, device=self.device)
        return times

    def sample_q_t_0(self, x, times, noise=None):
        """
        Samples from the distributions q(x_t | x_0) at the given time steps.

        Args:
            x: Input data.
            times: Sampled times.
            noise: Noise tensor.

        Returns:
            tuple: Tuple containing mean, gamma_t, and padded gamma_t.
        """
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            gamma_t = self.gamma(times)
        gamma_t_padded = unsqueeze_right(gamma_t, x.ndim - gamma_t.ndim)
        mean = x * sqrt(sigmoid(-gamma_t_padded))  # x * alpha
        scale = sqrt(sigmoid(gamma_t_padded))
        if noise is None:
            noise = torch.randn_like(x)
        return mean + noise * scale, gamma_t, gamma_t_padded

    @torch.no_grad()
    def sample_p_s_t(self, z, t, s, clip_samples):
        """
        Samples from p(z_s | z_t, x). Used for standard ancestral sampling.

        Args:
            z: Latent variable.
            t: Initial time.
            s: Target time.
            clip_samples: Whether to clip samples.

        Returns:
            torch.Tensor: Sampled values.
        """
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        
        c = -expm1(gamma_s - gamma_t)
        alpha_t = sqrt(sigmoid(-gamma_t))
        alpha_s = sqrt(sigmoid(-gamma_s))
        sigma_t = sqrt(sigmoid(gamma_t))
        sigma_s = sqrt(sigmoid(gamma_s))

        pred_noise = self.denoising_model(z, unsqueeze_right(gamma_t, z.ndim - gamma_t.ndim))
        if clip_samples:
            x_start = (z - sigma_t * pred_noise) / alpha_t
            x_start.clamp_(-1.0, 1.0)
            mean = alpha_s * (z * (1 - c) / alpha_t + c * x_start)
        else:
            mean = alpha_s / alpha_t * (z - c * sigma_t * pred_noise)
        scale = sigma_s * sqrt(c)
        return mean + scale * torch.randn_like(z) 

    @torch.no_grad()
    def sample(self, batch_size, n_sample_steps, clip_samples, library_size=1e4):
        """
        Sample from the model.

        Args:
            batch_size (int): Batch size.
            n_sample_steps (int): Number of sample steps.
            clip_samples: Whether to clip samples.
            library_size (float): Library size.

        Returns:
            torch.Tensor: Sampled values.
        """
        z = torch.randn((batch_size, self.denoising_model.in_dim), device=self.device)
        steps = linspace(1.0, 0.0, n_sample_steps + 1, device=self.device)
        for i in trange(n_sample_steps, desc="sampling"):
            z = self.sample_p_s_t(z, steps[i], steps[i + 1], clip_samples)
        if self.train_library_size:
            library_size = torch.exp(self.library_size_enc(z))
        z_softmax = F.softmax(z, dim=1)
        distr = NegativeBinomial(mu=library_size*z_softmax, theta=torch.exp(self.theta))
        return distr.sample()
    
    def log_probs_x_z0(self, x, z_0, library_size):
        """
        Compute log p(x | z_0) for all possible values of each pixel in x.

        Args:
            x: Input data.
            z_0: Latent variable.
            library_size (float): Library size.

        Returns:
            torch.Tensor: Log probabilities.
        """
        z_0_softmax = F.softmax(z_0, dim=1)
        distr = NegativeBinomial(mu=library_size*z_0_softmax, theta=torch.exp(self.theta))
        recon_loss = -distr.log_prob(x)
        return recon_loss
    
    def configure_optimizers(self):
        """
        Optimizer configuration 

        Returns:
            dict: Optimizer configuration.
        """
        optimizer_config = {'optimizer': torch.optim.AdamW(self.parameters(), 
                                                           self.learning_rate, 
                                                           betas=(0.9, 0.99), 
                                                           weight_decay=self.weight_decay, 
                                                           eps=1e-8)}

        return optimizer_config

    def _scale_batch(self, x):
        """
        Scale input batch.

        Args:
            x: Input data.

        Returns:
            torch.Tensor: Scaled data.
        """
        if self.scaling_method == "log_normalization":
            x_scaled = torch.log(1 + x)  # scale input 
        elif self.scaling_method == "z_score_normalization":
            x_scaled = self._z_score_normalize(x)
        elif self.scaling_method == "minmax_normalization":
            x_scaled = self._min_max_scale(x)
        elif self.scaling_method == "unnormalized":
            x_scaled = x
        else:
            raise ValueError(f"Unknown normalization {self.scaling_method}")
        return x_scaled
            
    def _z_score_normalize(self, data, epsilon=1e-8):
        """
        Z-score normalize data.

        Args:
            data: Input data.
            epsilon (float, optional): Small constant. Defaults to 1e-8.

        Returns:
            torch.Tensor: Normalized data.
        """
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        
        # Handle zero variance by adding epsilon to the standard deviation
        std += epsilon
        
        normalized_data = (data - mean) / std
        return normalized_data

    def _min_max_scale(self, data, epsilon=1e-8):
        """
        Min-max scale data.

        Args:
            data: Input data.
            epsilon (float, optional): Small constant. Defaults to 1e-8.

        Returns:
            torch.Tensor: Scaled data.
        """
        min_val = torch.min(data, dim=0)[0]
        max_val = torch.max(data, dim=0)[0]
        
        # Handle zero variance by setting a small constant value for the range
        range_val = max_val - min_val + epsilon
        
        scaled_data = (data - min_val) / range_val * 2 - 1
        return scaled_data
    
    def test_step(self, batch, batch_idx):
        """
        Training step for VDM.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        self.testing_ouputs.append(batch["X"])
    
    def on_test_epoch_end(self, *arg, **kwargs):
        """
        Concatenates all observations from the test data loader in a single dataset.

        Args:
            outputs: List of outputs from the test step.

        Returns:
            None
        """
        # Concatenate all test observations
        testing_outputs = torch.cat(self.testing_ouputs, dim=0)
        # Plot UMAP of generated cells and real test cells
        wd = compute_umap_and_wasserstein(model=self, 
                                            batch_size=1000, 
                                            n_sample_steps=1000, 
                                            clip_samples=False, 
                                            library_size=self.generative_library_size,
                                            plotting_folder=self.plotting_folder, 
                                            X_real=testing_outputs)
        # Compute Wasserstein distance between real test set and generated data 
        self.log("wasserstein_distance", wd)
        return wd
    