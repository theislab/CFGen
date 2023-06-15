# https://github.com/Michedev/DDPMs-Pytorch/blob/72d621ea7b64793b82bc5cace3605b85dc5d0b03/variance_scheduler/cosine.py

from math import pi

import torch
import torch.nn.functional as F

from celldreamer.models.diffusion.variance_scheduler.abs_var_scheduler import Scheduler

class CosineScheduler(Scheduler):
    def __init__(self, T: int = 1000, s: float = 0.008):
        # Get the range 
        self.T_range = (torch.arange(T + 1, dtype=torch.float64)) / T
        # Cumulative product of alphas
        self._alpha_cos = self.f(self.T_range, s)
        self._alpha_cos = self._alpha_cos / self._alpha_cos[0]
        # Set official alphas and betas
        self._betas = 1 - (self._alpha_cos[1:] / self._alpha_cos[:-1])
        self._betas = torch.clip(self._betas, 0, 0.999)
        self._alpha = 1 - self._betas
        self._alpha_hats = torch.cumprod(self._alpha, dim=0)
        self._alpha_hats_prev = F.pad(self._alpha_hats[:-1], (1, 0), value = 1.)
        self._posterior_variance = (1 - self._alpha_hats_prev) / (1 - self._alpha_hats) * self._betas
        
    def f(self, t: torch.Tensor, s: float):
        return torch.cos((t + s) / (1 + s) * pi * 0.5)**2

    def get_alpha_hat(self):
        return self._alpha_hats
    
    def get_alpha_hat_prev(self):
        return self._alpha_hats_prev

    def get_alphas(self):
        return self._alpha

    def get_betas(self):
        return self._betas

    def get_posterior_variance(self):
        return self._posterior_variance