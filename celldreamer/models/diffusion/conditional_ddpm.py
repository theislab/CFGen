import numpy as np
from typing import Literal
import pytorch_lightning as pl
import torch
from anndata import AnnData
from torch import nn
from typing import Callable, Union, Optional, List
import pandas as pd 

from celldreamer.models.diffusion.variance_scheduler.cosine import CosineScheduler
from celldreamer.models.diffusion.distributions import x0_to_xt


class ConditionalGaussianDDPM(pl.LightningModule):
    """
    Implementation of "Classifier-Free Diffusion Guidance"
    """

    def __init__(self,
                 denoising_model: nn.Module,
                 autoencoder_model: nn.Module, 
                 feature_embeddings: dict, 
                 T: int,  # default: 1_000
                 w: float,  # default: 0.3
                 classifier_free: bool, 
                 p_uncond: float,
                 task: str, 
                 use_drugs: bool,
                 one_hot_encode_features: bool,
                 metric_collector, 
                 optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
                 variance_scheduler= CosineScheduler,  # default: cosine
                 learning_rate: float = 0.001, 
                 weight_decay: float = 0.0001, 
                 ):

        super().__init__()
        
        # Denoising model and autpoencoder
        self.denoising_model = denoising_model
        self.autoencoder_model = autoencoder_model
        self.metric_collector = metric_collector
        
        # Number of classes per covariate 
        self.num_classes = self.denoising_model.num_classes
        
        # Diffusion hyperparameters 
        self.T = T
        self.w = w        
        self.p_uncond = p_uncond
        self.in_dim = self.denoising_model.in_dim 
        self.task = task
        self.feature_embeddings = feature_embeddings
        self.classifier_free = classifier_free
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_drugs = use_drugs
        self.one_hot_encode_features = one_hot_encode_features
        
        # Scheduler and the associated variances
        self.var_scheduler = variance_scheduler(T = self.T)
        self.alphas_hat = self.var_scheduler.get_alpha_hat().to(self.device)
        self.alphas = self.var_scheduler.get_alphas().to(self.device)
        self.betas = self.var_scheduler.get_betas().to(self.device)
        self.betas_hat = self.var_scheduler.get_betas_hat().to(self.device)
        
        # Optimization 
        self.mse = nn.MSELoss()
        self.iteration = 0
        self.optim = optimizer

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor: 
        return self.denoising_model(x, t, c)

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'valid')

    # def on_validation_epoch_end(self):
    #     return self.eval_distr_matching(dataset="valid")
    
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
        y=self._featurize_batch_y(batch)
        
        # Dummy flags that with probability p_uncond, we train without class conditioning
        if self.classifier_free:
            is_class_cond = torch.rand(size=(X.shape[0], 1), device=X.device) >= self.p_uncond
            y = y * is_class_cond.float() 
            
        # Sample t uniformly from [0, T-1]
        t = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device) 
        t_expanded = t.reshape(-1, 1)
        eps = torch.randn_like(X)  
        alpha_hat_t = self.alphas_hat[t_expanded]
        x_t = x0_to_xt(X, alpha_hat_t, eps)  # go from x_0 to x_t in a single equation thanks to the step
        pred_eps = self(x_t, t / self.T, y) # predict the noise to transition from x_t to x_{t-1}
        loss = self.mse(eps, pred_eps) # compute the MSE between the predicted noise and the real noise
        # print("True", eps)
        # print("predicted", pred_eps)
        
        self.log(f"loss/{dataset}_loss", loss, on_step=True)

        self.iteration += 1
        return loss
    
    def eval_distr_matching(self, dataset):
        real_adata, generated_adata, reconstructed_adata = self.get_generated_and_reconstructed(dataset)
        self.metric_collector[dataset].compute_generation_metrics(real_adata, 
                                                                    generated_adata, 
                                                                    reconstructed_adata)
        self.log_dict(self.metric_collector[dataset])

    def generate(self, 
                 batch_size: Optional[int] = None, 
                 y: Optional[torch.Tensor] = None, 
                 z_t: Optional[torch.Tensor] = None,
                 T: Optional[int] = None,
                 get_intermediate_steps: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate sample of cells
        """
        # Generation time 
        T = T or self.T
        batch_size = batch_size or 1
        is_c_none = y is None
        
        # Optional conditioning
        if is_c_none:
            y = torch.zeros(batch_size, 
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
            t = torch.LongTensor([t] * batch_size).to(self.device)
            t_expanded = t.view(-1, 1)
            if is_c_none or not self.classifier_free:
                # compute unconditioned noise
                eps = self(z_t, t / T, y)  # predict via nn the noise
                print(eps)
            else:
                # compute class conditioned noise
                eps1 = (1 + self.w) * self(z_t, t / T, y)
                eps2 = self.w * self(z_t, t / T, y * 0)
                eps = eps1 - eps2
            
            alpha_t = self.alphas[t_expanded]
            z = torch.randn_like(z_t)
            alpha_hat_t = self.alphas_hat[t_expanded]
            # denoise step from x_t to x_{t-1} following the DDPM paper
            z_t = (z_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * eps) / (torch.sqrt(alpha_t)) + \
                  self.betas[t_expanded] * z
            
        if get_intermediate_steps:
            steps.append(z_t)
            
        if self.autoencoder_model != None:
            x_hat = self.autoencoder_model.decoder(z_t)
        else:
            x_hat = z_t
            
        if get_intermediate_steps:
            return x_hat, steps
        else:
            return x_hat
    
    def reconstruct(self, batch, T):
        # Encode X to latent space 
        X = batch["X"].to(self.device)
        if self.autoencoder_model != None:
            X = self.autoencoder_model.encoder(X)
        
        # Encode and concatenate variables
        y = self._featurize_batch_y(batch)
        # Generate at the end of the trajectory
        t = (T * torch.ones(X.shape[0], device=X.device)).long()
        t_expanded = t.view(-1, 1)
        eps = torch.randn_like(X)  # [bs, c, w, h]
        alpha_hat_t = self.alphas_hat[t_expanded] # get \hat{\alpha}_t
        x_t = x0_to_xt(X, alpha_hat_t, eps)  # go from x_0 to x_t in a single equation thanks to the step        
        
        # Generate observation from x_t 
        x_hat = self.generate(x_t.shape[0], 
                              y, 
                              x_t, 
                              T, 
                              get_intermediate_steps=False)
        return x_hat, x_t

    def configure_optimizers(self):
        parms_to_train = list(self.denoising_model.parameters())
        if self.task == "perturbation_modelling" and self.use_drugs:
            parms_to_train.extend(list(self.feature_embeddings["y_drug"].parameters()))
        if not self.one_hot_encode_features:
            for cov in self.feature_embeddings:
                if cov != "y_drug":
                    parms_to_train.extend(list(self.feature_embeddings[cov].embeddings.parameters()))
            
        optimizer_config = {'optimizer': self.optim(parms_to_train,
                                                    lr=self.learning_rate, 
                                                    weight_decay=self.weight_decay)}
        return optimizer_config

    def on_fit_start(self) -> None:
        self.betas = self.betas.to(self.device)
        self.betas_hat = self.betas_hat.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_hat = self.alphas_hat.to(self.device)
            
    def get_generated_and_reconstructed(self, dataset):
        """
        Compute metrics obtained from generating from random noise 
        """
        real_array = []
        generated_array = []
        reconstructed_array = []
              
        metadata = {i.strip("y_"):[] for i in self.feature_embeddings}
        if self.task == "perturbation_modelling" and self.use_drugs:
            metadata["dose"] = []
              
        for batch in self.metric_collector[dataset].dataloader:
            if self.task == "perturbation_modelling":
                # Get feature combination embedding to condition generation
                y = self._featurize_batch_y(batch)
                
                # Perform generation and reconstruction 
                X_generated = self.generate(batch["X"].shape[0],
                                                  y).detach().cpu().numpy()
                X_reconstructed = self.reconstruct(batch,
                                                    self.T)[0].detach().cpu().numpy()
                
                real_array.append(batch["X"].detach().cpu().numpy())
                generated_array.append(X_generated)
                reconstructed_array.append(X_reconstructed)
                
                for cat in batch["y"]:
                    if cat == "y_drug":
                        metadata["drug"] += batch["y"][cat][0].cpu().tolist()
                        metadata["dose"] += batch["y"][cat][1].cpu().tolist()
                    else:
                        metadata[cat.strip("y_")] += batch["y"][cat].cpu().tolist()      
                        
            else:
                raise NotImplementedError
        
        # Save the data 
        metadata = pd.DataFrame(metadata)
        real_adata = AnnData(X=np.concatenate(real_array, axis=0),
                             obs=metadata)
        generated_adata = AnnData(X=np.concatenate(generated_array, axis=0), 
                                  obs=metadata.copy())
        reconstructed_adata = AnnData(X=np.concatenate(reconstructed_array, axis=0), 
                                      obs=metadata.copy())
        return real_adata, generated_adata, reconstructed_adata
    
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
    