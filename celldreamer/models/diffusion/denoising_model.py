import math
import numpy as np
from typing import List, Tuple, Optional, Callable

import torch
from torch import nn
from torch.nn import functional as F


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
                 in_channels: int,
                 out_channels: int,
                 time_embed_size: int,
                 p_dropout: float, 
                 num_classes: int, 
                 class_emb_size: int
                 ):
        super().__init__()
        """
        Like ResBlockTimeEmbed, but without convolutional layers.
        Instead use linear layers.
        """ 
        self.linear_map_class = nn.Sequential(
            nn.Linear(np.sum(list(num_classes.values())), class_emb_size),
            nn.ReLU(),
            nn.Linear(class_emb_size, class_emb_size)
        )
                
        self.net = nn.Sequential(
                        nn.Linear(in_channels + class_emb_size, out_channels),
                        nn.GELU(),
                        nn.Linear(out_channels, out_channels))
        
        self.l_embedding = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_embed_size, out_channels)
        )
        
        self.relu = nn.ReLU()
        
        self.out_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(out_channels, out_channels),
        )
        
        self.skip_connection = nn.Sequential(
                        nn.Linear(in_channels + class_emb_size, out_channels),
                        nn.GELU(),
                        nn.Linear(out_channels, out_channels))

    def forward(self, x, y, time_embed):
        c = self.linear_map_class(y)
        x = torch.cat([x, c], dim=1)
        h = self.net(x)
        time_embed = self.l_embedding(time_embed)
        h = self.relu(h + time_embed)
        return self.out_layer(h) + self.skip_connection(x)


class MLPTimeStep(torch.nn.Sequential):
    def __init__(
            self,
            in_dim: int, 
            dims: List[int],
            time_embed_size: int,
            num_classes: int, 
            class_emb_size: int,
            dropout: float = 0.0
    ):
        super().__init__()
        self.in_dim = in_dim
        self.time_embed_size = time_embed_size
        self.num_classes = num_classes
        self.class_emb_size = class_emb_size
        
        # Add the initial dimension to the channel list
        dims = [in_dim] + dims
        rev_dims = dims[::-1]

        self.encoder = nn.ModuleList([
                MLPTimeEmbedCond(in_channels=dims[i], 
                             out_channels=dims[i + 1],
                             time_embed_size=time_embed_size,
                             p_dropout=dropout, 
                             num_classes=num_classes, 
                             class_emb_size=class_emb_size) 
                for i in range(len(dims)-1)
                ])
 
        self.middle_block = MLPTimeEmbedCond(
            in_channels=dims[-1],
            out_channels=dims[-1],
            time_embed_size=time_embed_size,
            p_dropout=dropout,
            num_classes=num_classes, 
            class_emb_size=class_emb_size,
        )

        self.decoder = nn.ModuleList([
                MLPTimeEmbedCond(in_channels=rev_dims[i], 
                             out_channels=rev_dims[i + 1],
                             time_embed_size=time_embed_size,
                             p_dropout=dropout, 
                             num_classes=num_classes, 
                             class_emb_size=class_emb_size)
                for i in range(len(rev_dims)-1)
                ])

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_size, self.time_embed_size),
            nn.SiLU(),
            nn.Linear(self.time_embed_size, self.time_embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor, y: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model
        Currently without self attention
        :param x:  input image
        :param t: time step
        :param c: class, not used, just for compatibility, I don't know why it should be there
        :return:
        """
        # Embed the time 
        time_embedding = self.time_embed(timestep_embedding(t, self.time_embed_size))
        
        h = x

        # Encoder
        for encoder_layer in self.encoder:
            h = encoder_layer(h, y, time_embedding)
            h = self.dropout(h)

        # Middle block
        h = self.middle_block(h, y, time_embedding)

        # Decoder
        for decoder_block in self.decoder:
            h = decoder_block(h, y, time_embedding)

        # Output
        return h

if __name__=="__main__":
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
    
    pred = m(X, y, t)
    print(pred)
    print(pred.shape)
    