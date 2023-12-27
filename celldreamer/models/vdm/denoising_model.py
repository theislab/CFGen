import pytorch_lightning as pl
import torch
from torch import nn

from celldreamer.models.base.utils import unsqueeze_right

# ResNet MLP 
class MLPTimeStep(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 hidden_dim: int,
                 dropout_prob: int,
                 n_blocks: int, 
                 model_type: str,
                 gamma_min: float, 
                 gamma_max: float,
                 time_varying=False):
        
        super().__init__()
        
        # Gene expression dimension 
        self.in_dim = in_dim
        
        # The network downsizes the input multiple times 
        self.hidden_dim = hidden_dim * (2**n_blocks)
        
        # Initialize the gammas
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.model_type = model_type
        self.time_varying = time_varying
        added_dimensions = (1 if time_varying else 0) + (1 if self.model_type=="conditional_latent" else 0)
        
        # Initial convolution
        self.net_in = nn.Linear(in_dim, self.hidden_dim)

        # Down path: n_blocks blocks with a resnet block and maybe attention.
        self.down_blocks = []
        self.up_blocks = []
        for _ in range(n_blocks):
            self.down_blocks.append(ResnetBlock(in_dim=self.hidden_dim,
                                                     out_dim=self.hidden_dim // 2,
                                                     added_dimensions=added_dimensions,
                                                     dropout_prob=dropout_prob,
                                                     time_varying=time_varying,
                                                     model_type=model_type))

            self.up_blocks.insert(-1, ResnetBlock(in_dim=self.hidden_dim // 2,
                                                        out_dim=self.hidden_dim,
                                                        added_dimensions=added_dimensions,
                                                        dropout_prob=dropout_prob,
                                                        time_varying=time_varying,
                                                        model_type=model_type))
            self.hidden_dim = self.hidden_dim // 2
        
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        self.net_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim * (2**n_blocks), in_dim))

    def forward(self, x, g_t, l):
        if self.time_varying:
            if g_t.shape[0] == 1:
                g_t = g_t.repeat((x.shape[0],) + (1,) * (g_t.ndim-1))
            if g_t.ndim != x.ndim:
                g_t = unsqueeze_right(g_t, x.ndim-g_t.ndim)
        
        if self.model_type=="conditional_latent":
            if l.ndim != x.ndim:
                l = unsqueeze_right(l, x.ndim-l.ndim)  
                
        # Get gamma to shape (B, ).
        t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)

        h = self.net_in(x)  # (B, embedding_dim, H, W)
        for down_block in self.down_blocks:  # n_blocks times
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
        time_varying=True,
        model_type="conditional_latent"):
        
        super().__init__()
        
        self.time_varying = time_varying
        self.model_type = model_type

        # Set output_dim to input_dim if not provided
        out_dim = in_dim if out_dim is None else out_dim

        self.out_dim = out_dim

        # First linear block with LayerNorm and SiLU activation
        self.net1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim))

        # Second linear block with LayerNorm, SiLU activation, and optional dropout
        self.net2 = nn.Sequential(
            nn.SiLU(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            nn.Linear(out_dim + added_dimensions, out_dim),
        )

        # Linear projection for skip connection if input_dim and output_dim differ
        if in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim)

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

        # Add conditional input if provided
        if self.time_varying:
            h = torch.cat([h, t], dim=1)
        if self.model_type=="conditional_latent":
            h = torch.cat([h, l], dim=1)

        # Forward pass through the second linear block
        h = self.net2(h)

        # Linear projection for skip connection if input_dim and output_dim differ
        if x.shape[1] != self.out_dim:
            x = self.skip_proj(x)

        # Add skip connection to the output
        assert x.shape == h.shape
        return x + h

    def _zero_init(self, module):
        nn.init.constant_(module.weight.data, 0)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
        return module
    
# Simple time MLP 
class SimpleMLPTimeStep(pl.LightningModule):
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
        if self.time_varying:
            if g_t.shape[0] == 1:
                g_t = g_t.repeat((x.shape[0],) + (1,) * (g_t.ndim-1))
            if g_t.ndim != x.ndim:
                g_t = unsqueeze_right(g_t, x.ndim-g_t.ndim)
        
        if self.model_type=="conditional_latent":
            if l.ndim != x.ndim:
                l = unsqueeze_right(l, x.ndim-l.ndim)    
        
        if self.time_varying:
            x = torch.cat([x, g_t], dim=1)
        if self.model_type=="conditional_latent":
            x = torch.cat([x, l], dim=1)
        return self.net(x)
