from typing import Literal
import numpy as np

import torch
from torch import nn
from torch import sigmoid, exp, sqrt, autograd, linspace, argmax
from torch.special import expm1
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Callable, Optional
from tqdm import tqdm
from functools import partial
from tqdm import trange

from scvi.distributions import NegativeBinomial

from celldreamer.models.vdm import FixedLinearSchedule, LearnedLinearSchedule
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
                 antithetic_time_sampling: bool = True 
                 ):
        
        self.denoising_model = denoising_model
        self.feature_embeddings = feature_embeddings
        self.one_hot_encoder = one_hot_encode_features
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.in_dim = denoising_model.in_dim
        self.antithetic_time_sampling = antithetic_time_sampling
        
        # Define a dimensionality-preserving encoder 
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
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'valid')
    
    # Forward
    def _step(self, batch, dataset: Literal['train', 'valid']):
        """
        train/validation step of DDPM. The logic is mostly taken from the original DDPM paper,
        except for the class conditioning part.
        """
        # Collect observation
        x = batch["X"].to(self.device)
        library_size = x.sum(1)
        x_log = torch.log(1+x)
        x0 = self.z0_from_x(x_log)
        
        # Collect concatenated labels
        y = self._featurize_batch_y(batch)
        
        # Sample time 
        times = self._sample_times(x.shape[0])
        
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
        pred_loss = ((model_out - noise) ** 2).sum(x_t.shape[1:])  # (B, )
        diffusion_loss = 0.5 * pred_loss * gamma_grad 

        # Latent loss
        gamma_1 = self.gamma(torch.tensor([1.0], device=self.device))
        sigma_1_sq = sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * x**2  # (alpha_1 * x)**2
        latent_loss = kl_std_normal(mean_sq, sigma_1_sq).sum(x_t.shape[1:]) # (B, )
        
        # Compute log p(x | z_0) for all possible values of each pixel in x.
        recons_loss = self.log_probs_x_z0(x, x0)  # (B, C, H, W, vocab_size)
        recons_loss = recons_loss.sum(recons_loss.shape[1:])
        
        # Total loss    
        loss = diffusion_loss + latent_loss + recons_loss

        with torch.no_grad():
            gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        
        # Save results
        metrics = {
            f"{dataset}/bpd": loss.mean(),
            f"{dataset}/diff_loss": diffusion_loss.mean(),
            f"{dataset}/latent_loss": latent_loss.mean(),
            f"{dataset}/loss_recon": recons_loss.mean(),
            f"{dataset}/gamma_0": gamma_0.item(),
            f"{dataset}/gamma_1": gamma_1.item(),
        }
        self.log_dict(metrics)
        return loss.mean(), metrics

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

        pred_noise = self.model(z, gamma_t)
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
        z = torch.randn((batch_size, *self.in_dim), device=self.device)
        steps = linspace(1.0, 0.0, n_sample_steps + 1, device=self.device)
        for i in trange(n_sample_steps, desc="sampling"):
            z = self.sample_p_s_t(z, steps[i], steps[i + 1], clip_samples)
        # Sample negative binomial 
        z_softmax = F.softmax(z, dim=1)
        distr = NegativeBinomial(mu=library_size*z_softmax, theta=torch.exp(self.theta))
        return distr.sample(batch_size)
    
    def log_probs_x_z0(self, x, z_0, library_size):
        z_0_softmax = F.softmax(z_0, dim=1)
        distr = NegativeBinomial(mu=library_size*z_0_softmax, theta=torch.exp(self.theta))
        recon_loss = -distr.log_prob(x)
        return recon_loss
    