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
                 conditional: bool = True,
                 dropout: bool = False,
                 p_dropout: float = 0.0,
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
            # layers.append(nn.BatchNorm1d(out_dim)) this results in a dim error
            raise NotImplementedError
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

        out = self.net(x)
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
            conditional: bool = True,
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
        channels = [dim + class_emb_size for dim in channels[:-1]] + [channels[-1]]

        layers = []
        for i in range(len(channels) - 1):
            layers.append(MLPTimeEmbedCond(in_dim=channels[i],
                                           out_dim=channels[i + 1],
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


# UNET for images, adapted from https://github.com/Michedev/DDPMs-Pytorch/blob/master/model/unet.py


class ImageSelfAttention(nn.Module):

    def __init__(self, num_channels: int, num_heads: int = 1):
        """
        Args:
            num_channels (int): Number of channels in the input.
            num_heads (int, optional): Number of attention heads. Default: 1.

        Shape:
            - Input: :math:`(N, C, L)`
            - Output: :math:`(N, C, L)`

        Examples:
        >>> attention = ImageSelfAttention(3)
        >>> input = torch.randn(1, 3, 64)
        >>> output = attention(input)
        >>> output.shape
        torch.Size([1, 3, 64]) """

        super().__init__()
        self.channels = num_channels
        self.heads = num_heads

        self.attn_layer = nn.MultiheadAttention(num_channels, num_heads=num_heads)

    def forward(self, x):
        """

        :param x: tensor with shape [batch_size, channels, width, height]
        :return: the attention output applied to the image with the shape [batch_size, channels, width, height]
        """
        b, c, w, h = x.shape
        x = x.reshape(b, w * h, c)

        attn_output, _ = self.attn_layer(x, x, x)
        return attn_output.reshape(b, c, w, h)


class ResBlockTimeEmbedCond(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 time_embed_size: int,
                 p_dropout: float,
                 num_classes: int,
                 class_embed_size: int,
                 encode_class: float = False,
                 conditional: bool = True,
                 ):
        """
         Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of the convolution kernel
            stride (int): stride of the convolution
            padding (int): padding of the convolution
            time_embed_size (int): size of the time embedding
            p_dropout (float): dropout probability
        """
        super().__init__()

        # Condition embedding
        self.conditional = conditional
        self.class_emb_size = class_embed_size
        if self.conditional:
            if encode_class:
                self.linear_map_class = nn.Sequential(
                    nn.Linear(np.sum(list(num_classes.values())), self.class_emb_size)
                )
            else:
                self.linear_map_class = nn.Identity()
                class_emb_size = np.sum(list(num_classes.values()))
        else:
            class_emb_size = 0

        # Time embedding 
        self.time_embed_net = nn.Sequential(
            nn.Linear(time_embed_size, out_channels),
            nn.SELU(),
            nn.Linear(out_channels, out_channels))

        # The feature net
        num_groups_in = self.find_max_num_groups(in_channels)
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups_in, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
        self.relu = nn.ReLU()
        self.l_embedding = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_embed_size, out_channels)
        )
        num_groups_out = self.find_max_num_groups(out_channels)
        self.out_layer = nn.Sequential(
            nn.GroupNorm(num_groups_out, out_channels),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        )
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def find_max_num_groups(self, in_channels: int) -> int:
        for i in range(4, 0, -1):
            if in_channels % i == 0:
                return i
        raise Exception()

    def forward(self, x, time_embed, y):
        h = self.conv(x)
        time_embed = self.time_embed_net(time_embed)
        time_embed = time_embed.view(time_embed.shape[0], time_embed.shape[1], 1, 1)
        if self.conditonal:
            c = self.linear_map_class(y)
            x = torch.cat([x, c], dim=1)

        h = h + time_embed
        return self.out_layer(h) + self.skip_connection(x)


class UNetTimeStepClassSetConditioned(nn.Module):
    """
    UNet architecture with class and time embedding injected in every residual block, both in downsample and upsample.
    Both information are mapped via an 2-layers MLP to a fixed embedding size.
    After the third downsample block a self-attention layer is applied.
    """

    def __init__(self,
                 channels: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 paddings: List[int],
                 downsample: bool,
                 p_dropouts: List[float],
                 time_embed_size: int,
                 num_classes: dict,
                 class_embed_size: int,
                 assert_shapes: bool = True):
        super().__init__()
        assert len(channels) == (len(kernel_sizes) + 1) == (len(strides) + 1) == (len(paddings) + 1) == \
               (len(p_dropouts) + 1), f'{len(channels)} == {(len(kernel_sizes) + 1)} == ' \
                                      f'{(len(strides) + 1)} == {(len(paddings) + 1)} == \
                                                              {(len(p_dropouts) + 1)}'
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.assert_shapes = assert_shapes
        self.num_classes = num_classes
        self.time_embed_size = time_embed_size
        self.class_embed_size = class_embed_size
        self.downsample_blocks = nn.ModuleList([
            ResBlockTimeEmbedCond(in_channels=channels[i],
                                  out_channels=channels[i + 1],
                                  kernel_size=kernel_sizes[i],
                                  stride=strides[i],
                                  padding=paddings[i],
                                  time_embed_size=time_embed_size,
                                  p_dropout=p_dropouts[i],
                                  num_classes=num_classes,
                                  class_embed_size=class_embed_size,
                                  ) for i in range(len(channels) - 1)
        ])

        self.use_downsample = downsample
        self.downsample_op = nn.MaxPool2d(kernel_size=2)
        self.middle_block = ResBlockTimeEmbedCond(in_channels=channels[-1],
                                                  out_channels=channels[-1],
                                                  kernel_size=kernel_sizes[-1],
                                                  stride=strides[-1],
                                                  padding=paddings[-1],
                                                  time_embed_size=time_embed_size,
                                                  p_dropout=p_dropouts[-1],
                                                  num_classes=num_classes,
                                                  class_embed_size=class_embed_size,
                                                  )
        self.upsample_blocks = nn.ModuleList([
            ResBlockTimeEmbedCond(in_channels=(2 if i != 0 else 1) * channels[-i - 1],
                                  out_channels=channels[-i - 2],
                                  kernel_size=kernel_sizes[-i - 1],
                                  stride=strides[-i - 1],
                                  padding=paddings[-i - 1],
                                  time_embed_size=time_embed_size,
                                  p_dropout=p_dropouts[-i - 1],
                                  num_classes=num_classes,
                                  class_embed_size=class_embed_size,
                                  ) for i in range(len(channels) - 1)
        ])
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in p_dropouts])
        self.p_dropouts = p_dropouts
        self.self_attn = ImageSelfAttention(channels[2])
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_size, self.time_embed_size),
            nn.SiLU(),
            nn.Linear(self.time_embed_size, self.time_embed_size),
        )

    def forward(self, x: torch.FloatTensor, t: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_channels = x.shape[1]
        if self.assert_shapes:
            tg.guard(x, "B, C, W, H")
        if self.assert_shapes:
            tg.guard(c, "B, NUMCLASSES")
        time_embedding = self.time_embed(t)
        if self.assert_shapes:
            tg.guard(time_embedding, "B, TE")
        h = self.forward_unet(x, time_embedding, c)
        x_recon = h
        if self.assert_shapes:
            tg.guard(x_recon, "B, C, W, H")
        return x_recon

    def forward_unet(self, x, time_embedding, c):
        hs = []
        h = x
        for i, downsample_block in enumerate(self.downsample_blocks):
            h = downsample_block(h, time_embedding, c)
            if i == 2:
                h = self.self_attn(h)
            h = self.dropouts[i](h)
            if i != (len(self.downsample_blocks) - 1): hs.append(h)
            if self.use_downsample and i != (len(self.downsample_blocks) - 1):
                h = self.downsample_op(h)
        h = self.middle_block(h, time_embedding, c)
        for i, upsample_block in enumerate(self.upsample_blocks):
            if i != 0:
                h = torch.cat([h, hs[-i]], dim=1)
            h = upsample_block(h, time_embedding, c)
            if self.use_downsample and (i != (len(self.upsample_blocks) - 1)):
                h = F.interpolate(h, size=hs[-i - 1].shape[-1], mode='nearest')
        return h
