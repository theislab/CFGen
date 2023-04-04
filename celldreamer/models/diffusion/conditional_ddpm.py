"""
Implementation of the classifier free guidance for DDPM.
Paper: https://arxiv.org/abs/2207.12598
Code: https://github.com/Michedev/DDPMs-Pytorch/blob/72d621ea7b64793b82bc5cace3605b85dc5d0b03/model/classifier_free_ddpm.py
"""

import pytorch_lightning as pl
import torch
from pathlib import Path
from torch import nn
from typing import Callable, Union, Optional, List

from celldreamer.models.diffusion.variance_scheduler.abs_var_scheduler import Scheduler
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
                 variance_scheduler: 'Scheduler',  # default: cosine
                 optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
                 ):
        """
        :param denoising_module: The nn which computes the denoise step i.e. q(x_{t-1} | x_t, c)
        :param T: the amount of noising steps
        :param w: strength of class guidance, hyperparameter, paper suggests 0.3
        :param p_uncond: probability of train a batch without class conditioning
        :param number_of_genes: number of the genes in training observations 
        :param num_classes: number of classes
        :param logging_freq: frequency of logging loss function during training
        :param v: generative variance hyper-parameter
        :param classifier_free: whether to apply the classifier-free logic or not 
        :param variance_scheduler: the variance scheduler cited in DDPM paper. See folder variance_scheduler for practical implementation
        """
        assert 0.0 <= v <= 1.0, f'0.0 <= {v} <= 1.0'
        assert 0.0 <= w, f'0.0 <= {w}'
        assert 0.0 <= p_uncond <= 1.0, f'0.0 <= {p_uncond} <= 1.0'
        super().__init__()
        
        self.denoising_model = denoising_model
        self.autoencoder_model = autoencoder_model
        self.num_classes = self.denoising_model.num_classes
        self.n_covariates = n_covariates
        self.T = T
        self.w = w        
        self.v = v
        self.p_uncond = p_uncond
        self.number_of_genes = self.denoising_model.in_dim 
        self.task = task
        self.feature_embeddings = feature_embeddings
        
        self.var_scheduler = variance_scheduler
        self.alphas_hat: torch.FloatTensor = self.var_scheduler.get_alpha_hat().to(self.device)
        self.alphas: torch.FloatTensor = self.var_scheduler.get_alphas().to(self.device)
        self.betas = self.var_scheduler.get_betas().to(self.device)
        self.betas_hat = self.var_scheduler.get_betas_hat().to(self.device)
        self.classifier_free = classifier_free
        
        self.mse = nn.MSELoss()
        self.logging_freq = logging_freq
        self.iteration = 0
        self.gen_images = Path('training_gen_images')
        self.gen_images.mkdir(exist_ok=True)
        
        self.optim = optimizer

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        predict the score (noise) to transition from step t to t-1
        :param x: input image [bs, c, w, h]
        :param t: time step [bs]
        :param c:  class [bs, num_classes]
        :return: the predicted noise to transition from t to t-1
        """
        return self.denoising_model(x, t, c)

    def training_step(self, X, y):
        return self._step(X, y, 'train')

    def validation_step(self, batch, batch_idx):
        pass

    def _step(self, batch) -> torch.Tensor:
        """
        train/validation step of DDPM. The logic is mostly taken from the original DDPM paper,
        except for the class conditioning part.
        """
        X = batch["X"]
        y = []
                
        for feature_cat in batch["y"]:
            y_cat = self.feature_embeddings[feature_cat](batch["y"][feature_cat])
            y.append(y_cat)
            
        y = torch.cat(y, dim=1) # Concatenate covariate encodings 
        
        if self.classifier_free:
            # dummy flags that with probability p_uncond, we train without class conditioning
            is_class_cond = torch.rand(size=(X.shape[0],1), device=X.device) >= self.p_uncond
            y = y * is_class_cond.float()  # set to zero the batch elements not class conditioned
            
        t = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)  # sample t uniformly from [0, T-1]
        t_expanded = t.reshape(-1, 1)
        eps = torch.randn_like(X)  # [bs, c, w, h]
        alpha_hat_t = self.alphas_hat[t_expanded] # get \hat{\alpha}_t
        x_t = x0_to_xt(X, alpha_hat_t, eps)  # go from x_0 to x_t in a single equation thanks to the step
        pred_eps = self(x_t, t / self.T, y) # predict the noise to transition from x_t to x_{t-1}
        loss = self.mse(eps, pred_eps) # compute the MSE between the predicted noise and the real noise

        self.iteration += 1
        return loss
    
    def generate(self, batch_size: Optional[int] = None, c: Optional[torch.Tensor] = None, T: Optional[int] = None,
                 get_intermediate_steps: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate a new sample starting from pure random noise sampled from a normal standard distribution
        :param batch_size: the generated batch size
        :param c: the class conditional matrix [batch_size, num_classes]. By default, it will be deactivated by passing a matrix of full zeroes
        :param T: the number of generation steps. By default, it will be the number of steps of the training
        :param get_intermediate_steps: if true, it will all return the intermediate steps of the generation
        :return: the generated image or the list of intermediate steps
        """
        T = T or self.T
        batch_size = batch_size or 1
        is_c_none = c is None
        # Optional conditioning
        if is_c_none:
            c = torch.zeros(batch_size, self.num_classes, device=self.device)
        if get_intermediate_steps:
            steps = []
            
        # Starting random noise 
        z_t = torch.randn(batch_size, self.number_of_genes, device=self.device)
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
        return z_t if not get_intermediate_steps else steps

    def configure_optimizers(self):
        optimizer_config = {'optimizer': self.optim(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)}
        return optimizer_config

    def on_fit_start(self) -> None:
        self.betas = self.betas.to(self.device)
        self.betas_hat = self.betas_hat.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_hat = self.alphas_hat.to(self.device)
        

if __name__=="__main__":
    from celldreamer.models.diffusion.denoising_model import MLPTimeStep
    from celldreamer.models.base.variance_scheduler.cosine import CosineScheduler
    
    X = torch.randn(16, 19000)
    t = torch.rand(16)
    y = torch.rand(16, 3)
    
    
    m = MLPTimeStep(
            in_dim=19000,
            dims=[128, 64],
            time_embed_size=100,
            num_classes=3, 
            class_emb_size=100,
            dropout=0
            )
    
    ddpm = ConditionalGaussianDDPM(denoising_module=m, 
                 T=4_000,  # default: 4_000
                 w=0.3,  # default: 0.3
                 v=0.1, 
                 p_uncond=0.3, 
                 number_of_genes=19000,
                 num_classes=3,  # default: number of attributes / classes
                 logging_freq=1000,   
                 classifier_free=True, 
                 variance_scheduler=CosineScheduler(T=4_000)  # default: cosine
                 )
    
    pred = ddpm(X, y, t)
    print(pred)
    print(pred.shape)
