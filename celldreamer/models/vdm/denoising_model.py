import math
import numpy as np
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.init as init

from celldreamer.models.base.utils import MLP
from celldreamer.models.base.utils import MLP, unsqueeze_right, kl_std_normal

def get_timestep_embedding(timesteps,
                           embedding_dim: int,
                           dtype=torch.float32,
                           max_timescale=10_000,
                           min_timescale=1):
    """
    Compute a sinusoidal embedding for a set of timesteps.

    Args:
        timesteps (torch.Tensor): Timesteps to be embedded.
        embedding_dim (int): Dimension of the embedding.
        dtype (torch.dtype, optional): Data type for the embedding. Defaults to torch.float32.
        max_timescale (int, optional): Maximum timescale value. Defaults to 10_000.
        min_timescale (int, optional): Minimum timescale value. Defaults to 1.

    Returns:
        torch.Tensor: Sinusoidal embedding for the given timesteps.
    """
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    timesteps *= 1000.0  # In DDPM the time step is in [0, 1000], here [0, 1]
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)

@torch.no_grad()
def init_zero(module: nn.Module) -> nn.Module:
    """
    Initialize module parameters to zero.

    Args:
        module (nn.Module): Module to be initialized.

    Returns:
        nn.Module: Initialized module.
    """
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
        """
        Multi-Layer Perceptron (MLP) with time embedding and conditioning.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            time_embed_size (int): Size of the time embedding.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            dropout (bool, optional): Whether to use dropout. Defaults to False.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__()

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
        """
        Forward pass of the MLPTimeEmbedCond.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
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
        """
        Multi-Layer Perceptron (MLP) with time step embedding.

        Args:
            in_dim (int): Input dimension.
            hidden_dims (List[int]): List of hidden layer dimensions.
            time_embed_size (int): Size of the time embedding.
            dropout (bool, optional): Whether to use dropout. Defaults to True.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            gamma_min (float, optional): Minimum value for normalization. Defaults to -13.3.
            gamma_max (float, optional): Maximum value for normalization. Defaults to 5.0.
        """
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
        """
        Forward pass of the MLPTimeStep.

        Args:
            x (torch.FloatTensor): Input tensor.
            g_t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Expand gamma to batch size
        g_t = g_t.expand(x.shape[0])
        # Normalize gamma
        t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        # Embed time 
        t_embedding = get_timestep_embedding(t, self.time_embed_size)
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

# Simple time MLP 

class SimpleMLPTimeStep(torch.nn.Module):
    def __init__(self,
                 in_dim, 
                 out_dim=None, 
                 w=64, 
                 time_varying=False, 
                 model_type="conditional_latent"):
        """
        Simple Multi-Layer Perceptron (MLP) with optional time variation.

        Args:
            in_dim (int): Input dimension.
            out_dim (int, optional): Output dimension. Defaults to None.
            w (int, optional): Hidden layer dimension. Defaults to 64.
            time_varying (bool, optional): Whether to use time variation. Defaults to False.
        """
        super().__init__()
        self.in_dim = in_dim
        self.time_varying = time_varying
        self.model_type = model_type
        
        if out_dim is None:
            out_dim = in_dim
            
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim + (1 if time_varying else 0) + (1 if self.model_type=="conditional_latent" else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, g_t=None, l=None):
        """
        Forward pass of the SimpleMLPTimeStep.

        Args:
            x (torch.Tensor): Input tensor.
            g_t (torch.Tensor): Time tensor.
            l (torch.Tensor): Size factor

        Returns:
            torch.Tensor: Output tensor.
        """
        # If g_t is not across all batch, repeat over the batch
        if self.time_varying:
            if g_t.shape[0] == 1:
                g_t = g_t.repeat((x.shape[0],) + (1,) * (g_t.ndim-1))
            if g_t.ndim != x.ndim:
                g_t = unsqueeze_right(g_t, x.ndim-g_t.ndim)
        
        if self.model_type=="conditional_latent":
            if l.ndim != l.ndim:
                l = unsqueeze_right(l, x.ndim-l.ndim)    
        
        if self.time_varying:
            x = torch.cat([x, g_t], dim=1)
        if self.model_type=="conditional_latent":
            x = torch.cat([x, l], dim=1)
        return self.net(x)
