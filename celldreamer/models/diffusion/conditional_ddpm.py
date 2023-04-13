import numpy as np
from typing import Literal
import pytorch_lightning as pl
import torch
from pathlib import Path
from torch import nn
from typing import Callable, Union, Optional, List

from celldreamer.models.diffusion.variance_scheduler.cosine import CosineScheduler
from celldreamer.models.diffusion.distributions import x0_to_xt


class ConditionalGaussianDDPM(pl.LightningModule):
    """
    Implementation of "Classifier-Free Diffusion Guidance"
    """

    def __init__(self,
                 denoising_model: nn.Module,  # MLP
                 autoencoder_model: nn.Module, 
                 feature_embeddings: dict, 
                 T: int,  # default: 4_000
                 w: float,  # default: 0.3
                 v: float, 
                 n_covariates: int, 
                 p_uncond: float,
                 logging_freq: int,   
                 classifier_free: bool, 
                 task: str, 
                 metric_collector, 
                 optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
                 variance_scheduler= CosineScheduler()  # default: cosine
                 ):

        super().__init__()
        
        # Denoising model and autpoencoder
        self.denoising_model = denoising_model
        self.autoencoder_model = autoencoder_model
        self.metric_collector = metric_collector
        
        # Number of classes per covariate 
        self.num_classes = self.denoising_model.num_classes
        self.n_covariates = n_covariates
        
        # Diffusion hyperparameters 
        self.T = T
        self.w = w        
        self.v = v
        self.p_uncond = p_uncond
        self.in_dim = self.denoising_model.in_dim 
        self.task = task
        self.feature_embeddings = feature_embeddings
        self.classifier_free = classifier_free
        
        # Scheduler and the associated variances
        self.var_scheduler = variance_scheduler
        self.alphas_hat = self.var_scheduler.get_alpha_hat().to(self.device)
        self.alphas = self.var_scheduler.get_alphas().to(self.device)
        self.betas = self.var_scheduler.get_betas().to(self.device)
        self.betas_hat = self.var_scheduler.get_betas_hat().to(self.device)
        
        # Optimization 
        self.mse = nn.MSELoss()
        self.logging_freq = logging_freq
        self.iteration = 0
        self.optim = optimizer

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor: 
        return self.denoising_model(x, t, c)

    def training_step(self, batch):
        return self._step(batch, 'train')

    def validation_step(self, batch):
        return self._step(batch, 'valid')

    def _step(self, batch, dataset: Literal['train', 'valid']) -> torch.Tensor:
        """
        train/validation step of DDPM. The logic is mostly taken from the original DDPM paper,
        except for the class conditioning part.
        """
        X = batch["X"].to(self.device)
        
        # If we are training a latent model, we encode X first 
        if self.autoencoder_model != None:
            X = self.autoencoder_model.encoder(X)
        
        # Encode covariates 
        y = []
        for feature_cat in batch["y"]:
            y_cat = self.feature_embeddings[feature_cat](batch["y"][feature_cat])
            y.append(y_cat)
        y = torch.cat(y, dim=1).to(self.device) # Concatenate covariate encodings 
        
        # Dummy flags that with probability p_uncond, we train without class conditioning
        if self.classifier_free:
            is_class_cond = torch.rand(size=(X.shape[0],1), device=X.device) >= self.p_uncond
            y = y * is_class_cond.float() 
            
        # Sample t uniformly from [0, T-1]
        t = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device) 
        t_expanded = t.reshape(-1, 1)
        eps = torch.randn_like(X)  
        alpha_hat_t = self.alphas_hat[t_expanded] 
        x_t = x0_to_xt(X, alpha_hat_t, eps)  # go from x_0 to x_t in a single equation thanks to the step
        pred_eps = self(x_t, t / self.T, y) # predict the noise to transition from x_t to x_{t-1}
        loss = self.mse(eps, pred_eps) # compute the MSE between the predicted noise and the real noise
        
        self._log(dataset, loss)

        self.iteration += 1
        return loss
    
    def generate(self, 
                 batch_size: Optional[int] = None, 
                 z_t: Optional[torch.Tensor] = None,
                 c: Optional[torch.Tensor] = None, 
                 T: Optional[int] = None,
                 get_intermediate_steps: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Generation time 
        T = T or self.T
        batch_size = batch_size or 1
        is_c_none = c is None
        
        # Optional conditioning
        if is_c_none:
            c = torch.zeros(batch_size, 
                            np.sum(list(self.num_classes.values())),
                            device=self.device)
        if get_intermediate_steps:
            steps = []
            
        # Starting random noise 
        if z_t is None:   
            z_t = torch.randn(batch_size, self.in_dim, device=self.device)

        for t in range(T - 1, 0, -1):
            if get_intermediate_steps:
                steps.append(z_t)
            t = torch.LongTensor([t] * batch_size).to(self.device).view(-1, 1)
            t_expanded = t.view(-1, 1)
            if is_c_none:
                # compute unconditioned noise
                eps = self(z_t, t / T, c)  # predict via nn the noise
            else:
                if self.classifier_free:
                    # compute class conditioned noise
                    eps1 = (1 + self.w) * self(z_t, t / T, c)
                    eps2 = self.w * self(z_t, t / T, c * 0)
                    eps = eps1 - eps2
                else: 
                    eps = self(z_t, t / T, c)
                    
            alpha_t = self.alphas[t_expanded]
            z = torch.randn_like(z_t)
            alpha_hat_t = self.alphas_hat[t_expanded]
            # denoise step from x_t to x_{t-1} following the DDPM paper
            z_t = (z_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * eps) / (torch.sqrt(alpha_t)) + \
                  self.betas[t_expanded] * z
                  
        if get_intermediate_steps:
            steps.append(z_t)
            
        if self.autoencoder_model != None:
            X_hat = self.autoencoder_model.decoder(z_t)
            
        return X_hat, steps
    
    def reconstruct(self, batch, T):
        # Encode X to latent space 
        X = batch["X"].to(self.device)
        if self.autoencoder_model != None:
            X = self.autoencoder_model.encoder(X)
        
        # Encode and concatenate variables
        y = []     
        for feature_cat in batch["y"]:
            y_cat = self.feature_embeddings[feature_cat](batch["y"][feature_cat])
            y.append(y_cat)
        y = torch.cat(y, dim=1).to(self.device)
        
        t = T * torch.ones(X.shape[0], 1, device=X.device) 
        t_expanded = t.reshape(-1, 1)
        eps = torch.randn_like(X)  # [bs, c, w, h]
        alpha_hat_t = self.alphas_hat[t_expanded] # get \hat{\alpha}_t
        x_t = x0_to_xt(X, alpha_hat_t, eps)  # go from x_0 to x_t in a single equation thanks to the step
        
        # Generate observation from x_t 
        x_hat = self.generate(x_t.shape[0], 
                              x_t, 
                              y, 
                              T, 
                              get_intermediate_steps=False)[0]
        return x_hat, x_t

    def configure_optimizers(self):
        if self.task == "perturbation_modelling":
            optimizer_config = {'optimizer': self.optim(list(self.parameters())+
                                                        list(self.feature_embeddings["y_drug"]).parameters(),
                                                        lr=self.lr, 
                                                        weight_decay=self.weight_decay)}
        else:
            optimizer_config = {'optimizer': self.optim(self.parameters(),
                                                        lr=self.lr, 
                                                        weight_decay=self.weight_decay)}
        return optimizer_config

    def on_fit_start(self) -> None:
        self.betas = self.betas.to(self.device)
        self.betas_hat = self.betas_hat.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_hat = self.alphas_hat.to(self.device)
        
    def _log(self, dataset, loss):        
        # log every batch on validation set, otherwise log every self.logging_freq batches on training set
        if dataset == "valid" or (self.iteration % self.logging_freq) == 0:
            self.log(f"loss/{dataset}_loss", loss, on_step=True)
            
        if dataset == "valid":
            real, generated, reconstructed = self.get_generated_and_reconstructed(self, dataset)
            self.metric_collector[dataset].compute_generation_metrics(real, 
                                                                      generated, 
                                                                      reconstructed)
            self.log_dict(self.metric_collector.metric_dict)
            self.metric_collector[dataset].reset_metrics()
    
    def get_generated_and_reconstructed(self, dataset):
        """
        Compute metrics obtained from generating from random noise 
        """
        real = {}
        generated = {}
        reconstructed = {}
        
        for batch in self.metric_collector[dataset].dataloader:
            if self.task == "perturbation_modelling":
                
                # Get feature combination embedding to condition generation 
                y = []
                for feature_cat in batch["y"]:
                    y_cat = self.feature_embeddings[feature_cat](batch["y"][feature_cat])
                    y.append(y_cat)
                y = torch.cat(y, dim=1).to(self.device)
                
                # Perform generation and reconstruction 
                X_generated = self.generate(batch["X"].shape[0],
                                                  y)[0]
                X_reconstructed = self.reconstruct(batch["X"].to(self.device),
                                                    y)[0]
                
                # Extract key by name 
                key = self._extract_batch_key_name(batch)
                for idx, k in enumerate(key): 
                    if k in self.real:
                        self.real[k] = torch.cat([self.real[k], batch["X"][idx].unsqueeze(0)], dim=0)
                        self.generated[k] = torch.cat([self.generated[k], X_generated[idx].unsqueeze(0)], dim=0)
                        self.reconstructed[k] = torch.cat([self.reconstructed[k], X_reconstructed[idx].unsqueeze(0)], dim=0)
                    else:
                        self.real[k] = batch["X"][idx].unsqueeze(0)
                        self.generated[k] = X_reconstructed[idx].unsqueeze(0)
                        self.reconstructed[k] = X_generated[idx].unsqueeze(0)
            else:
                raise NotImplementedError 
            return real, generated, reconstructed
        
    def _extract_batch_key_name(self, batch):
        if self.task == "perturbation_modelling":
            y_drug = batch["y"].pop("y_drug")
            y_cov = [batch["y"][key].argmax(1).tolist() for key in batch["y"]]
            key = list(zip(y_drug[0].tolist(),
                            y_drug[1].tolist(), 
                            *y_cov))
        else:
            raise NotImplementedError
        return key 
    