from typing import Literal
import numpy as np

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


class VDM(pl.LightningModule):
    def __init__(self,
                 denoising_model: nn.Module,
                 feature_embeddings: dict, 
                 one_hot_encode_features: bool,
                 z0_from_x_kwargs: dict,
                 learning_rate: float = 0.001, 
                 weight_decay: float = 0.0001, 
                 noise_schedule: str = "fixed_linear", 
                 gamma_min: float = -13.3, 
                 gamma_max: float = 5.0,
                 antithetic_time_sampling: bool = True, 
                 scaling_method: int = "log_normalization"
                 ):
        
        super().__init__()
        self.denoising_model = denoising_model.to(self.device)
        self.feature_embeddings = feature_embeddings
        self.one_hot_encode_features = one_hot_encode_features
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.in_dim = denoising_model.in_dim
        self.antithetic_time_sampling = antithetic_time_sampling
        self.scaling_method = scaling_method
        
        # Define a dimensionality-preserving encoder 
        z0_from_x_kwargs["dims"] = [self.in_dim, *z0_from_x_kwargs["dims"], self.in_dim]
        self.z0_from_x = MLP(**z0_from_x_kwargs)
        
        # Define the inverse dispersion parameter 
        self.theta = torch.nn.Parameter(torch.randn(self.in_dim))
            
        assert noise_schedule in ["fixed_linear", "learned_linear"]
        if noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(gamma_min, gamma_max)
        elif noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(gamma_min, gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {noise_schedule}")
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, dataset='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, dataset='valid')
    
    # Forward
    def _step(self, batch, dataset: Literal['train', 'valid']):
        """
        train/validation step of DDPM. The logic is mostly taken from the original DDPM paper,
        except for the class conditioning part.
        """
        # Collect observation
        x = batch["X"].to(self.device)
        library_size = x.sum(1).unsqueeze(1)
        x_scaled = self._scale_batch(x)
        x0 = self.z0_from_x(x_scaled)
        
        # Collect concatenated labels
        y = self._featurize_batch_y(batch)
        
        # Sample time 
        times = self._sample_times(x.shape[0]).requires_grad_(True)
        
        # Sample noise 
        noise = torch.rand_like(x)
        
        # Sample an observation undergoing noising 
        x_t, gamma_t = self.sample_q_t_0(x=x0, times=times, noise=noise)
        
        # Forward through the model 
        model_out = self.denoising_model(x_t, gamma_t)
        
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
        mean_sq = (1 - sigma_1_sq) * x**2  # (alpha_1 * x)**2
        latent_loss = kl_std_normal(mean_sq, sigma_1_sq).sum(1) # (B, )
        
        # Compute log p(x | z_0) for all possible values of each pixel in x.
        recons_loss = self.log_probs_x_z0(x, x0, library_size)  
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
        self.log_dict(metrics)
        return loss.mean()

    # Private methods
    def _featurize_batch_y(self, batch):
        """
        Featurize all the covariates 
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
        """
        if self.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
        else:
            times = torch.rand(batch_size, device=self.device)
        return times

    def sample_q_t_0(self, x, times, noise=None):
        """Samples from the distributions q(x_t | x_0) at the given time steps."""
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            gamma_t = self.gamma(times)
        gamma_t_padded = unsqueeze_right(gamma_t, x.ndim - gamma_t.ndim)
        mean = x * sqrt(sigmoid(-gamma_t_padded))  # x * alpha
        scale = sqrt(sigmoid(gamma_t_padded))
        if noise is None:
            noise = torch.randn_like(x)
        return mean + noise * scale, gamma_t    

    @torch.no_grad()
    def sample_p_s_t(self, z, t, s, clip_samples):
        """Samples from p(z_s | z_t, x). Used for standard ancestral sampling."""
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        alpha_t = sqrt(sigmoid(-gamma_t))
        alpha_s = sqrt(sigmoid(-gamma_s))
        sigma_t = sqrt(sigmoid(gamma_t))
        sigma_s = sqrt(sigmoid(gamma_s))

        pred_noise = self.denoising_model(z, gamma_t)
        if clip_samples:
            x_start = (z - sigma_t * pred_noise) / alpha_t
            x_start.clamp_(-1, 1)
            mean = alpha_s * (z * (1 - c) / alpha_t + c * x_start)
        else:
            mean = alpha_s / alpha_t * (z - c * sigma_t * pred_noise)
        scale = sigma_s * sqrt(c)
        return mean + scale * torch.randn_like(z)
    
    @torch.no_grad()
    def sample(self, batch_size, n_sample_steps, clip_samples, library_size=1e4):
        z = torch.randn((batch_size, self.in_dim), device=self.device)
        steps = linspace(1.0, 0.0, n_sample_steps + 1, device=self.device)
        for i in trange(n_sample_steps, desc="sampling"):
            z = self.sample_p_s_t(z, steps[i], steps[i + 1], clip_samples)
        # Sample negative binomial 
        z_softmax = F.softmax(z, dim=1)
        distr = NegativeBinomial(mu=library_size*z_softmax, theta=torch.exp(self.theta))
        return distr.sample(), z
    
    def log_probs_x_z0(self, x, z_0, library_size):
        gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        gamma_0_padded = unsqueeze_right(gamma_0, x.ndim - gamma_0.ndim)
        mean = sqrt(sigmoid(-gamma_0_padded))  # x * alpha
        scale = sqrt(sigmoid(gamma_0_padded))
        z_0 = mean * z_0 + scale * torch.rand_like(z_0)
        z_0 = z_0 / mean
        z_0_softmax = F.softmax(z_0, dim=1)
        distr = NegativeBinomial(mu=library_size*z_0_softmax, theta=torch.exp(self.theta))
        recon_loss = -distr.log_prob(x)
        return recon_loss
    
    def configure_optimizers(self):
        """
        Optimizer configuration 
        """ 
        optimizer_config = {'optimizer': torch.optim.AdamW(self.parameters(), 
                                                           self.learning_rate, 
                                                           betas=(0.9, 0.99), 
                                                           weight_decay=self.weight_decay, 
                                                           eps=1e-8)}

        return optimizer_config

    def _scale_batch(self, x):
        if self.scaling_method == "log_normalization":
            x_scaled = torch.log(1+x)  # scale input 
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
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        
        # Handle zero variance by adding epsilon to the standard deviation
        std += epsilon
        
        normalized_data = (data - mean) / std
        return normalized_data

    def _min_max_scale(self, data, epsilon=1e-8):
        min_val = torch.min(data, dim=0)[0]
        max_val = torch.max(data, dim=0)[0]
        
        # Handle zero variance by setting a small constant value for the range
        range_val = max_val - min_val + epsilon
        
        scaled_data = (data - min_val) / range_val * 2 - 1
        return scaled_data
