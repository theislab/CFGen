import math
import numpy as np
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.init as init


# import tensorguard as tg


def positional_embedding_vector(t: int, dim: int) -> torch.FloatTensor:
    """
    Args:
        t (int): time step
        dim (int): embedding size
    Returns: the transformer sinusoidal positional embedding vector
    """
    two_i = 2 * torch.arange(0, dim)
    return torch.sin(t / torch.pow(10000, two_i / dim)).unsqueeze(0)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(timesteps.device)


@torch.no_grad()
def init_zero(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        torch.nn.init.zeros_(p)
    return module


class MLPTimeEmbedCond(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dims: int,
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

        # The net
        channels = [in_dim+class_emb_size+time_embed_size, *hidden_dims, in_dim]
        layers = []
        
        for i in range(len(channels)-2):
            layers.append(nn.Linear(channels[i], channels[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(p=p_dropout))
        layers.append(nn.Linear(channels[-2], channels[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x, time_embed, y):
        if self.conditional:
            c = self.linear_map_class(y)
            x = torch.cat([x, c], dim=1)
        
        x = torch.cat([x, time_embed], dim=1)
        x = self.net(x)
        return x


class MLPTimeStep(torch.nn.Sequential):
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

        # Neural network object
        self.model = MLPTimeEmbedCond(in_dim=in_dim, 
                            hidden_dims=hidden_dims,
                            time_embed_size=time_embed_size,
                            num_classes=num_classes, 
                            class_emb_size=class_emb_size, 
                            encode_class=encode_class, 
                            conditional=conditional, 
                            dropout=dropout,
                            p_dropout=p_dropout,
                            batch_norm=batch_norm
                            ) 

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
        x = self.model(x, time_embedding, y)

        # Output
        return x

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0.0)