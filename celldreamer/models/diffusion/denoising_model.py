import math
import numpy as np
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.init as init

def positional_embedding_vector(t: int, dim: int) -> torch.FloatTensor:
    """
    Args:
        t (int): time step
        dim (int): embedding size
    Returns: the transformer sinusoidal positional embedding vector
    """
    two_i = 2 * torch.arange(0, dim)
    return torch.sin(t / torch.pow(10000, two_i / dim)).unsqueeze(0)


def timestep_embedding(t: torch.Tensor, dim: int):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb).to(t.device)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb

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
                 num_classes: int, 
                 class_emb_size: int, 
                 encode_class: float = False, 
                 conditional:bool=True,
                 dropout: bool=False,
                 p_dropout: float=0.0,
                 batch_norm: bool = False
                 ):
        
        super().__init__()
        """
        Like ResBlockTimeEmbed, but without convolutional layers.
        Instead use linear layers.
        """ 
        # Condition embedding
        self.conditional = conditional
        if self.conditional:
            if encode_class:
                self.linear_map_class = nn.Sequential(
                    nn.Linear(np.sum(list(num_classes.values())), class_emb_size)
                )
            else:
                self.linear_map_class = nn.Identity()
                class_emb_size = np.sum(list(num_classes.values()))
        else:
            class_emb_size = 0
            
        # Time embedding 
        self.time_embed_net = nn.Sequential(
            nn.Linear(time_embed_size, out_dim),
            nn.SELU(),
            nn.Linear(out_dim, out_dim))

        # The feature net
        layers = []
        layers.append(nn.Linear(in_dim, out_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.SELU())
        if dropout:
            layers.append(nn.Dropout(p=p_dropout))
    
        self.net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(out_dim, out_dim)

    def forward(self, x, time_embed, y):
        time_embed = self.time_embed_net(time_embed)
        if self.conditional:
            c = self.linear_map_class(y)
            x = torch.cat([x, c], dim=1)
        
        x = self.net(x) + time_embed
        return self.out_layer(x)

class MLPTimeStep(nn.Module):
    def __init__(
            self,
            in_dim: int, 
            hidden_dims: List[int],
            time_embed_size: int,
            num_classes: int, 
            class_emb_size: int,
            encode_class: float = False, 
            conditional:bool = True,
            dropout: bool = True,
            p_dropout: float = 0.0,
            batch_norm: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.time_embed_size = time_embed_size
        self.num_classes = num_classes
        self.class_emb_size = class_emb_size
        
        # Set up class conditioning
        if conditional:
            self.linear_map_class = nn.Identity()
            class_emb_size = np.sum(list(num_classes.values()))
        else:
            class_emb_size = 0
        
        # Neural network object
        channels = [in_dim, *hidden_dims, in_dim]
        channels = [dim+class_emb_size for dim in channels[:-1]]+[channels[-1]]
        
        layers = []
        for i in range(len(channels)-1):
            layers.append(MLPTimeEmbedCond(in_dim=channels[i], 
                                out_dim=channels[i+1],
                                time_embed_size=time_embed_size,
                                num_classes=num_classes, 
                                class_emb_size=class_emb_size, 
                                encode_class=encode_class, 
                                conditional=conditional, 
                                dropout=dropout,
                                p_dropout=p_dropout,
                                batch_norm=batch_norm))
        self.net = nn.Sequential(*layers)

        # Initialize the parameters using He initialization
        self.apply(self._init_weights)

    def forward(self, x: torch.FloatTensor, t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model
        Currently without self attention
        :param x:  input image
        :param t: time step
        :param c: class, not used, just for compatibility, I don't know why it should be there
        :return:
        """
        # Embed the time 
        time_embedding = timestep_embedding(t, self.time_embed_size)
        # Encoder
        for layer in self.net:
            x = layer(x, time_embedding, y)
        # Output
        return x

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.kaiming_uniform_(module.weight, mode='fan_in')
            if module.bias is not None:
                init.constant_(module.bias, 0.0)
                