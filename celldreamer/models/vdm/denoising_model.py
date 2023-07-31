import math
import numpy as np
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.init as init

from celldreamer.models.base.utils import MLP

def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    timesteps *= 1000.0  # In DDPM the time step is in [0, 1000], here [0, 1]
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)

@torch.no_grad()
def init_zero(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        torch.nn.init.zeros_(p)
    return module

class MLPTimeEmbedCond(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 time_embed_size: int,
                 batch_norm: bool = False,
                 dropout: bool=False,
                 dropout_p: float=0.0,
                 ):
        
        super().__init__()
        """
        Like ResBlockTimeEmbed, but without convolutional layers.
        Instead use linear layers.
        """ 
        # The feature net
        layers = []
        layers.append(nn.Linear(in_dim, out_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.SELU())
        if dropout:
            layers.append(nn.Dropout(p=dropout_p))
        self.net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(out_dim, out_dim)
        
        # Linear projection time 
        self.time_proj = nn.Linear(time_embed_size, out_dim)

    def forward(self, x, t):        
        x = self.net(x) + self.time_proj(t)
        return self.out_layer(x)

class MLPTimeStep(nn.Module):
    def __init__(
            self,
            in_dim: int, 
            hidden_dims: List[int],
            time_embed_size: int,
            dropout: bool = True,
            dropout_p: float = 0.0,
            batch_norm: bool = False,
            gamma_min: float = -13.3, 
            gamma_max: float = 5.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.time_embed_size = time_embed_size
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        
        # Neural network object
        dims = [in_dim, *hidden_dims, in_dim]
        
        layers = []
        for i in range(len(dims)-1):
            layers.append(MLPTimeEmbedCond(in_dim = dims[i],
                                            out_dim = dims[i+1],
                                            time_embed_size = time_embed_size,
                                            batch_norm = batch_norm,
                                            dropout = dropout,
                                            dropout_p = dropout_p))
        self.net = nn.Sequential(*layers)
        
        # Time embedding
        self.embed_conditioning = nn.Sequential(
            nn.Linear(self.time_embed_size, self.time_embed_size),
            nn.SELU(),
            nn.Linear(self.time_embed_size, self.time_embed_size),
            nn.SELU(),
        )

        # Initialize the parameters using He initialization
        self.apply(self._init_weights)

    def forward(self, x: torch.FloatTensor, g_t: torch.Tensor):
        # Expand gamma to batch size
        g_t = g_t.expand(x.shape[0])
        # Normalize gamma
        t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t_embedding = get_timestep_embedding(t, self.time_embed_size)
        # We will condition on time embedding.
        cond = self.embed_conditioning(t_embedding)
        
        # Encoder
        for layer in self.net:
            x = layer(x, cond)
            
        # Output
        return x

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.kaiming_uniform_(module.weight, mode='fan_in')
            if module.bias is not None:
                init.constant_(module.bias, 0.0)
                