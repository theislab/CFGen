import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from celldreamer.models.fm.layer_utils import Linear
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
                 embed_time: bool,
                 size_factor_min: float, 
                 size_factor_max: float,
                 embed_size_factor: bool,
                 embedding_dim=128, 
                 normalization="layer", 
                 conditional=False,
                 embed_condition=False, 
                 n_cond=None):
        
        super().__init__()
        
        # Gene expression dimension 
        self.in_dim = in_dim
        
        # The network downsizes the input multiple times 
        self.hidden_dim = hidden_dim 
        
        # Initialize attributes 
        self.model_type = model_type
        self.embed_time = embed_time
        self.size_factor_min = size_factor_min
        self.size_factor_max = size_factor_max
        self.embed_size_factor = embed_size_factor 
        self.embedding_dim = embedding_dim
        self.conditional = conditional
        self.embed_condition = embed_condition
        
        added_dimensions = 0  # Incremented if not embedding conditioning variables 
        
        # Time embedding network
        if embed_time:
            self.time_embedder = nn.Sequential(
                Linear(embedding_dim, embedding_dim * 4),  # Upsample embedding
                nn.SiLU(),
                Linear(embedding_dim * 4, embedding_dim * 4))
        else:
            added_dimensions += 1
            
        # Size factor embeddings 
        if model_type=="conditional_latent":
            if embed_size_factor:
                self.size_factor_embedder = nn.Sequential(
                    Linear(embedding_dim, embedding_dim * 4),  # Upsample embedding
                    nn.SiLU(),
                    Linear(embedding_dim * 4, embedding_dim * 4))
            else:
                added_dimensions += 1
        
        # Covariate embedding
        if conditional:
            if embed_condition:
                self.condition_embedder = nn.Sequential(
                    Linear(n_cond, embedding_dim * 4),  # Upsample embedding
                    nn.SiLU(),
                    Linear(embedding_dim * 4, embedding_dim * 4))
            else:
                added_dimensions += 1
        
        # Initial convolution
        self.net_in = Linear(in_dim, self.hidden_dim)

        # Down path: n_blocks blocks with a resnet block and maybe attention.
        self.blocks = []

        # Dimensionality preserving Resnet in the bottleneck 
        for _ in range(n_blocks):
            self.blocks.append(ResnetBlock(in_dim=self.hidden_dim,
                                                out_dim=self.hidden_dim,
                                                added_dimensions=added_dimensions,
                                                dropout_prob=dropout_prob,
                                                model_type=model_type, 
                                                embedding_dim=embedding_dim * 4,  
                                                normalization=normalization, 
                                                embed_time=embed_time,
                                                embed_size_factor=embed_size_factor, 
                                                conditional=conditional,
                                                embed_condition=embed_condition))
        
        # Set up blocks
        self.blocks = nn.ModuleList(self.blocks)
        
        if normalization not in ["layer", "batch"]:
            self.net_out = nn.Sequential(
                nn.SiLU(),
                zero_init(Linear(self.hidden_dim, in_dim)))
        else:
            self.net_out = nn.Sequential(
                nn.LayerNorm(self.hidden_dim) if normalization=="layer" else nn.BatchNorm1d(num_features=self.hidden_dim),
                nn.SiLU(),
                zero_init(Linear(self.hidden_dim, in_dim)))    

    def forward(self, x, t, l, y):
        # If time is unique (e.g., during sampling) for all batch observations, repeat over the batch dimension
        if t.shape[0] == 1:
            t = t.repeat((x.shape[0],) + (1,) * (t.ndim-1))
        
        # Make a copy of time for using in time embeddings
        t_for_embeddings = t.clone().detach()

        # Size factor 
        if self.model_type == "conditional_latent":
            if self.embed_size_factor:
                l = l.squeeze()
                l = (l - self.size_factor_min) / (self.size_factor_max - self.size_factor_min)
                l = self.size_factor_embedder(get_timestep_embedding(l, self.embedding_dim))
            else:
                if l.ndim != x.ndim:
                    l = unsqueeze_right(l, x.ndim-l.ndim)  
                
        # Get time to shape (B, ).
        if self.embed_time:
            t_for_embeddings = t_for_embeddings.squeeze()
            t_for_embeddings = self.time_embedder(get_timestep_embedding(t_for_embeddings, self.embedding_dim))
        else:
            if t_for_embeddings.ndim != x.ndim:
                t_for_embeddings = unsqueeze_right(t_for_embeddings, x.ndim - t_for_embeddings.ndim)
                
        # Embed condition
        if self.conditional and self.embed_condition:
            y = self.condition_embedder(y)

        # Embed x
        h = self.net_in(x)  
        for block in self.blocks:  # n_blocks times
            h = block(h, t_for_embeddings, l, y)
        pred = self.net_out(h)
        return pred 

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
        normalization="batch", 
        embed_time=True,
        embed_size_factor=True, 
        conditional=True,
        embed_condition=True):
        
        super().__init__()
        
        self.model_type = model_type
        
        # Variables controlling if time and size factor should be embedded
        self.embed_time = embed_time
        self.embed_size_factor = embed_size_factor
        self.conditional = conditional
        self.embed_condition = embed_condition    
        self.embedding_dim = embedding_dim
    
        # Set output_dim to input_dim if not provided
        out_dim = in_dim if out_dim is None else out_dim

        self.out_dim = out_dim

        # First linear block with LayerNorm and SiLU activation
        if normalization not in ["layer", "batch"]:
            self.net1 = nn.Sequential(
                nn.SiLU(),
                Linear(in_dim, out_dim))          
        else:
            self.net1 = nn.Sequential(
                nn.LayerNorm(in_dim) if normalization=="layer" else nn.BatchNorm1d(num_features=in_dim),
                nn.SiLU(),
                Linear(in_dim, out_dim))
        
        # Projections for conditions 
        if embed_time:
            self.cond_proj_time = nn.Sequential(nn.SiLU(), Linear(self.embedding_dim, out_dim))
        if embed_size_factor and self.model_type=="conditional_latent":
            self.cond_proj_size_factor = nn.Sequential(nn.SiLU(), Linear(self.embedding_dim, out_dim))
        if embed_condition and self.conditional:
            self.cond_proj_covariate = nn.Sequential(nn.SiLU(), Linear(self.embedding_dim, out_dim))
            
        # Second linear block with LayerNorm, SiLU activation, and optional dropout
        if normalization not in ["layer", "batch"]:
            self.net2 = nn.Sequential(
                nn.SiLU(),
                *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
                zero_init(Linear(out_dim + added_dimensions, out_dim)))
        else:
            self.net2 = nn.Sequential(
                nn.LayerNorm(out_dim + added_dimensions) if normalization=="layer" else nn.BatchNorm1d(num_features=out_dim + added_dimensions),
                nn.SiLU(),
                *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
                zero_init(Linear(out_dim + added_dimensions, out_dim)))

        # Linear projection for skip connection if input_dim and output_dim differ
        if in_dim != out_dim:
            self.skip_proj = Linear(in_dim, out_dim)

    def forward(self, x, t, l, y, **args):
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
        if self.embed_time:
            t = self.cond_proj_time(t)
            h = h + t
            
        if self.model_type=="conditional_latent":
            if self.embed_size_factor:
                l = self.cond_proj_size_factor(l)
                h = h + l

        if self.embed_condition and self.conditional:
            y = self.cond_proj_covariate(y)
            h = h + y

        # Concateante if embedding is not the chosen option
        if not self.embed_time: 
            h = torch.cat([h, t], dim=1)
            
        if self.model_type=="conditional_latent" and not self.embed_size_factor:
            h = torch.cat([h, l], dim=1)
            
        if not self.embed_condition and self.conditional:
            h = torch.cat([h, y], dim=1)
                
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
                 w=512, 
                 model_type="conditional_latent", 
                 conditional=False, 
                 n_cond=None):
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
        self.conditional = conditional
        
        if out_dim is None:
            out_dim = in_dim
            
        self.net = torch.nn.Sequential(
            Linear(in_dim + 1 + (1 if self.model_type=="conditional_latent" else 0) + (n_cond if conditional else 0), w),
            nn.SELU(),
            Linear(w, w),
            nn.SELU(),
            Linear(w, w),
            nn.SELU(),
            Linear(w, out_dim),
        )
        self.save_hyperparameters()

    def forward(self, x, t, l, y, **args):
        """
        Forward pass of the SimpleMLPTimeStep.

        Args:
            x (torch.Tensor): Input tensor.
            g_t (torch.Tensor): Time tensor.
            l (torch.Tensor): Size factor

        Returns:
            torch.Tensor: Output tensor.
        """
        # If t is not repeated for all elements in the batch 
        if t.shape[0] == 1:
            t = t.repeat((x.shape[0],) + (1,) * (t.ndim-1))
        if t.ndim != x.ndim:
            t = unsqueeze_right(t, x.ndim-t.ndim)
        
        if self.model_type=="conditional_latent":
            if l.ndim != x.ndim:
                l = unsqueeze_right(l, x.ndim-l.ndim)    
        
        x = torch.cat([x, t], dim=1)
        if self.model_type=="conditional_latent":
            x = torch.cat([x, l], dim=1)
        if self.conditional:
            x = torch.cat([x, y], dim=1)
        return self.net(x) 
