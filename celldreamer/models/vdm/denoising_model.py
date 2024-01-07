import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from celldreamer.models.vdm.layer_utils import Linear
from celldreamer.models.base.utils import unsqueeze_right

# Util functions
def zero_init(module):
    """
    Initializes the weights and biases of a PyTorch module with zero values.

    Args:
        module (torch.nn.Module): PyTorch module for weight and bias initialization.

    Returns:
        torch.nn.Module: The input module with weights and biases initialized to zero.
    """
    nn.init.constant_(module.weight.data, 0)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias.data, 0)
    return module

def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    """
    Generates a sinusoidal embedding for a sequence of timesteps.

    Args:
        timesteps (torch.Tensor): 1-dimensional tensor representing the input timesteps.
        embedding_dim (int): Dimensionality of the embedding. It must be an even number.
        dtype (torch.dtype, optional): Data type for the resulting tensor. Default is torch.float32.
        max_timescale (float, optional): Maximum timescale value for the sinusoidal embedding. Default is 10,000.
        min_timescale (float, optional): Minimum timescale value for the sinusoidal embedding. Default is 1.

    Returns:
        torch.Tensor: Sinusoidal embedding tensor for the input timesteps with the specified embedding_dim.
    """
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

# ResNet MLP 
class MLPTimeStep(pl.LightningModule):
    def __init__(self, 
                 in_dim: int,
                 hidden_dim: int,
                 dropout_prob: int,
                 n_blocks: int, 
                 model_type: str,
                 gamma_min: float, 
                 gamma_max: float,
                 embed_gamma: bool,
                 size_factor_min: float, 
                 size_factor_max: float,
                 embed_size_factor: bool,
                 embedding_dim=None, 
                 normalization="layer"):
        
        super().__init__()
        
        # Gene expression dimension 
        self.in_dim = in_dim
        
        # The network downsizes the input multiple times 
        self.hidden_dim = hidden_dim * (2**n_blocks)
        
        self.model_type = model_type
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.embed_gamma = embed_gamma
        self.size_factor_min = size_factor_min
        self.size_factor_max = size_factor_max
        self.embed_size_factor = embed_size_factor 
        self.embedding_dim = embedding_dim
        
        # Time embedding network
        added_dimensions = 0
        if embed_gamma:
            self.embed_time = nn.Sequential(
                Linear(embedding_dim, embedding_dim * 4),  # Upsample embedding
                nn.SiLU(),
                Linear(embedding_dim * 4, embedding_dim * 4),
                nn.SiLU(),
            )
        else:
            added_dimensions += 1
            
        # Size factor embeddings 
        if model_type=="conditional_latent":
            if embed_size_factor:
                self.embed_size_factor = nn.Sequential(
                    Linear(embedding_dim, embedding_dim * 4),  # Upsample embedding
                    nn.SiLU(),
                    Linear(embedding_dim * 4, embedding_dim * 4),
                    nn.SiLU(),
                )
            else:
                added_dimensions += 1
        
        # Initial convolution
        self.net_in = Linear(in_dim, self.hidden_dim)

        # Down path: n_blocks blocks with a resnet block and maybe attention.
        self.down_blocks = []
        self.up_blocks = []
        for _ in range(n_blocks):
            self.down_blocks.append(ResnetBlock(in_dim=self.hidden_dim,
                                                     out_dim=self.hidden_dim // 2,
                                                     added_dimensions=added_dimensions,
                                                     dropout_prob=dropout_prob,
                                                     model_type=model_type, 
                                                     embedding_dim=embedding_dim * 4, 
                                                     normalization=normalization, 
                                                     embed_gamma=embed_gamma,
                                                     embed_size_factor=embed_size_factor))

            self.up_blocks.insert(-1, ResnetBlock(in_dim=self.hidden_dim // 2,
                                                        out_dim=self.hidden_dim,
                                                        added_dimensions=added_dimensions,
                                                        dropout_prob=dropout_prob,
                                                        model_type=model_type, 
                                                        embedding_dim=embedding_dim * 4, 
                                                        normalization=normalization, 
                                                        embed_gamma=embed_gamma,
                                                        embed_size_factor=embed_size_factor))
            self.hidden_dim = self.hidden_dim // 2
        
        # Set up blocks
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        self.net_out = nn.Sequential(
            nn.LayerNorm(hidden_dim * (2**n_blocks)) if normalization=="layer" else nn.BatchNorm1d(num_features=hidden_dim * (2**n_blocks)),
            nn.SiLU(),
            zero_init(Linear(hidden_dim * (2**n_blocks), in_dim)))

    def forward(self, x, g_t, l):
        # If time is unique (e.g during sampling) for all batch observations, repeat over batch dimension
        if g_t.shape[0] == 1:
            g_t = g_t.repeat((x.shape[0],) + (1,) * (g_t.ndim-1))
        
        # Size factor 
        if self.model_type=="conditional_latent":
            if self.embed_size_factor:
                l = l.squeeze()
                l = (l - self.size_factor_min) / (self.size_factor_max - self.size_factor_min)
                l = self.embed_size_factor(get_timestep_embedding(l, self.embedding_dim))
            else:
                if l.ndim != x.ndim:
                    l = unsqueeze_right(l, x.ndim-l.ndim)  
                
        # Get gamma to shape (B, ).
        t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        if self.embed_gamma:
            t = t.squeeze()
            t = self.embed_time(get_timestep_embedding(t, self.embedding_dim))
        else:
            if t.ndim != x.ndim:
                t = unsqueeze_right(t, x.ndim-t.ndim)

        # Embed x
        h = self.net_in(x)  
        for down_block in self.down_blocks:  # n_locks times
            h = down_block(h, t, l)
        for up_block in self.up_blocks:  
            h = up_block(h, t, l)
            
        pred = self.net_out(h)
        return pred + x

class ResnetBlock(nn.Module):
    """
    A block for a Multi-Layer Perceptron (MLP) with skip connection.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int, optional): Dimension of the output features. Defaults to None, in which case it's set equal to input_dim.
        condition_dim (int, optional): Dimension of the conditional input. Defaults to None.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.0.
        norm_groups (int, optional): Number of groups for layer normalization. Defaults to 32.
    """
    def __init__(
        self,
        in_dim,
        out_dim=None,
        added_dimensions=0,
        dropout_prob=0.0,
        model_type="conditional_latent", 
        embedding_dim=None, 
        normalization="layer", 
        embed_gamma=True,
        embed_size_factor=True):
        
        super().__init__()
        
        self.model_type = model_type
        
        self.embed_gamma = embed_gamma
        self.embed_size_factor = embed_size_factor
        self.embedding_dim = embedding_dim

        # Set output_dim to input_dim if not provided
        out_dim = in_dim if out_dim is None else out_dim

        self.out_dim = out_dim

        # First linear block with LayerNorm and SiLU activation
        self.net1 = nn.Sequential(
            nn.LayerNorm(in_dim) if normalization=="layer" else nn.BatchNorm1d(num_features=in_dim),
            nn.SiLU(),
            Linear(in_dim, out_dim))
        
        # Projections for conditions 
        if embed_gamma:
            self.cond_proj_gamma = zero_init(Linear(self.embedding_dim, out_dim, bias=False))
        if embed_size_factor:
            self.cond_proj_size_factor = zero_init(Linear(self.embedding_dim, out_dim, bias=False))

        # Second linear block with LayerNorm, SiLU activation, and optional dropout
        self.net2 = nn.Sequential(
            nn.LayerNorm(out_dim + added_dimensions) if normalization=="layer" else nn.BatchNorm1d(num_features=out_dim + added_dimensions),
            nn.SiLU(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            zero_init(Linear(out_dim + added_dimensions, out_dim)))

        # Linear projection for skip connection if input_dim and output_dim differ
        if in_dim != out_dim:
            self.skip_proj = Linear(in_dim, out_dim)

    def forward(self, x, t, l=None):
        """
        Forward pass of the MLP block.

        Args:
            x (torch.Tensor): Input features.
            condition (torch.Tensor, optional): Conditional input. Defaults to None.

        Returns:
            torch.Tensor: Output features.
        """
        # Forward pass through the first linear block
        h = self.net1(x)

        # Condition time and library size 
        if self.embed_gamma:
            t = self.cond_proj_gamma(t)
            h = h + t
            
        if self.model_type=="conditional_latent":
            if self.embed_size_factor:
                l = self.cond_proj_size_factor(l)
                h = h + l

        if not self.embed_gamma:
            h = torch.cat([h, t], dim=1)
            
        if self.model_type=="conditional_latent" and not self.embed_size_factor:
            h = torch.cat([h, l], dim=1)
                
        # Forward pass through the second linear block
        h = self.net2(h)

        # Linear projection for skip connection if input_dim and output_dim differ
        if x.shape[1] != self.out_dim:
            x = self.skip_proj(x)

        # Add skip connection to the output
        assert x.shape == h.shape
        
        return x + h
    
# Simple time MLP 
class SimpleMLPTimeStep(pl.LightningModule):
    def __init__(self,
                 in_dim, 
                 out_dim=None, 
                 w=64, 
                 model_type="conditional_latent"):
        """
        Simple Multi-Layer Perceptron (MLP) with optional time variation.

        Args:
            in_dim (int): Input dimension.
            out_dim (int, optional): Output dimension. Defaults to None.
            w (int, optional): Hidden layer dimension. Defaults to 64.
        """
        super().__init__()
        self.in_dim = in_dim
        self.model_type = model_type
        
        if out_dim is None:
            out_dim = in_dim
            
        self.net = torch.nn.Sequential(
            Linear(in_dim + 1 + (1 if self.model_type=="conditional_latent" else 0), w),
            nn.SELU(),
            Linear(w, w),
            torch.nn.SELU(),
            Linear(w, w),
            nn.SELU(),
            Linear(w, out_dim),
        )
        self.save_hyperparameters()

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
        if g_t.shape[0] == 1:
            g_t = g_t.repeat((x.shape[0],) + (1,) * (g_t.ndim-1))
        if g_t.ndim != x.ndim:
            g_t = unsqueeze_right(g_t, x.ndim-g_t.ndim)
        
        if self.model_type=="conditional_latent":
            if l.ndim != x.ndim:
                l = unsqueeze_right(l, x.ndim-l.ndim)    
        
        x = torch.cat([x, g_t], dim=1)
        if self.model_type=="conditional_latent":
            x = torch.cat([x, l], dim=1)
        return self.net(x) + x
