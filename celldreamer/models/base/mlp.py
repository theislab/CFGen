import math
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
    return torch.sin(t / torch.pow(10_000, two_i / dim)).unsqueeze(0)


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


class ResBlockTimeEmbed(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 time_embed_size: int, p_dropout: float):
        super().__init__()
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
        """
        Find the maximum number of groups that can be used for a given number of channels.
        :param in_channels: the number of channels
        :return: the maximum number of groups
        """
        for i in range(4, 0, -1):
            if in_channels % i == 0:
                return i
        raise Exception()

    def forward(self, x, time_embed):
        h = self.conv(x)
        time_embed = self.l_embedding(time_embed)
        time_embed = time_embed.view(time_embed.shape[0], time_embed.shape[1], 1, 1)
        h = h + time_embed
        return self.out_layer(h) + self.skip_connection(x)


class MLPTimeEmbed(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_embed_size: int,
                 p_dropout: float
                 ):
        super().__init__()
        """
        Like ResBlockTimeEmbed, but without convolutional layers.
        Instead use linear layers.
        """
        self.l_embedding = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_embed_size, out_channels)
        )
        self.relu = nn.ReLU()
        self.out_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(in_channels, out_channels),
        )
        self.skip_connection = nn.Linear(in_channels, out_channels)

    def forward(self, x, time_embed):
        time_embed = self.l_embedding(time_embed)
        h = self.relu(x + time_embed)
        return self.out_layer(h) + self.skip_connection(x)


class ImageSelfAttention(nn.Module):

    def __init__(self, num_channels: int, num_heads: int = 1):
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


class UNetTimeStep(nn.Module):

    def __init__(self, channels: List[int], kernel_sizes: List[int], strides: List[int], paddings: List[int],
                 downsample: bool, p_dropouts: List[float], time_embed_size: int):
        super().__init__()
        assert len(channels) == (len(kernel_sizes) + 1) == (len(strides) + 1) == (len(paddings) + 1) == \
               (len(p_dropouts) + 1), f'{len(channels)} == {(len(kernel_sizes) + 1)} == ' \
                                      f'{(len(strides) + 1)} == {(len(paddings) + 1)} == \
                                                              {(len(p_dropouts) + 1)}'
        self.channels = channels
        self.time_embed_size = time_embed_size
        self.downsample_blocks = nn.ModuleList([
            ResBlockTimeEmbed(channels[i], channels[i + 1], kernel_sizes[i], strides[i],
                              paddings[i], time_embed_size, p_dropouts[i]) for i in range(len(channels) - 1)
        ])

        self.use_downsample = downsample
        self.downsample_op = nn.MaxPool2d(kernel_size=2)
        self.middle_block = ResBlockTimeEmbed(channels[-1], channels[-1], kernel_sizes[-1], strides[-1],
                                              paddings[-1], time_embed_size, p_dropouts[-1])
        channels[0] *= 2 # because the output is the image plus the estimated variance coefficients
        self.upsample_blocks = nn.ModuleList([
            ResBlockTimeEmbed((2 if i != 0 else 1) * channels[-i - 1], channels[-i - 2], kernel_sizes[-i - 1],
                              strides[-i - 1],
                              paddings[-i - 1], time_embed_size, p_dropouts[-i - 1]) for i in range(len(channels) - 1)
        ])
        self.dropouts = nn.ModuleList([nn.Dropout2d(p) for p in p_dropouts])
        self.p_dropouts = p_dropouts
        self.self_attn = ImageSelfAttention(channels[3])
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_size, self.time_embed_size),
            nn.SiLU(),
            nn.Linear(self.time_embed_size, self.time_embed_size),
        )

    def forward(self, x: torch.FloatTensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_channels = x.shape[1]
        # tg.guard(x, "B, C, W, H")
        time_embedding = self.time_embed(timestep_embedding(t, self.time_embed_size))
        # tg.guard(time_embedding, "B, TE")
        hs = []
        h = x
        for i, downsample_block in enumerate(self.downsample_blocks):
            h = downsample_block(h, time_embedding)
            if i == 2:
                h = self.self_attn(h)
            h = self.dropouts[i](h)
            if i != (len(self.downsample_blocks) - 1): hs.append(h)
            if self.use_downsample and i != (len(self.downsample_blocks) - 1):
                h = self.downsample_op(h)
        h = self.middle_block(h, time_embedding)
        for i, upsample_block in enumerate(self.upsample_blocks):
            if i != 0:
                h = torch.cat([h, hs[-i]], dim=1)
            h = upsample_block(h, time_embedding)
            if self.use_downsample and (i != (len(self.upsample_blocks) - 1)):
                h = F.interpolate(h, size=hs[-i - 1].shape[-2:], mode='nearest')
        x_recon, v = h[:, :x_channels], h[:, x_channels:]
        # tg.guard(x_recon, "B, C, W, H")
        # tg.guard(v, "B, C, W, H")
        return x_recon, v


class MLPTimeStep(torch.nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: List[int],
            time_embed_size: int,
            inplace: Optional[bool] = True,
            dropout: float = 0.0,
    ):
        super().__init__()
        params = {} if inplace is None else {"inplace": inplace}
        self.time_embed_size = time_embed_size

        self.encoder = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

        self.middle_block = MLPTimeEmbed(
            in_channels=hidden_channels[-1],
            out_channels=hidden_channels[-1],
            time_embed_size=time_embed_size,
            p_dropout=dropout,
        )

        self.decoder = MLP(
            in_channels=hidden_channels[-1],
            hidden_channels=hidden_channels[::-1],
            dropout=dropout,
        )

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_size, self.time_embed_size),
            nn.SiLU(),
            nn.Linear(self.time_embed_size, self.time_embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor, t: torch.Tensor, c) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model
        Currently without self attention
        :param x:  input image
        :param t: time step
        :param c: class, not used, just for compatibility, I don't know why it should be there
        :return:
        """
        x_channels = x.shape[1]
        # tg.guard(x, "B, C, W, H")
        time_embedding = self.time_embed(timestep_embedding(t, self.time_embed_size))

        hs = []
        h = x

        # Encoder
        for i in range(len(self.encoder)):
            h = self.encoder[i](h)
            h = self.dropout(h)
            if i != (len(self.encoder) - 1):
                hs.append(h)
            if self.use_downsample and i != (len(self.encoder) - 1):
                h = self.downsample_op(h)

        # Middle block
        h = self.middle_block(h, time_embedding)

        # Decoder
        for i in range(len(self.decoder)):
            if i != 0:
                h = torch.cat([h, hs[-i]], dim=1)
            h = self.decoder[i](h)
            if self.use_downsample and (i != (len(self.decoder) - 1)):
                h = F.interpolate(h, size=hs[-i - 1].shape[-2:], mode='nearest')

        # Output
        x_recon, v = h[:, :x_channels], h[:, x_channels:]
        return x_recon, v


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.
    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    Implementation slightly adapted from https://pytorch.org/vision/main/generated/torchvision.ops.MLP.html
    (removed Dropout from last layer + log_api_usage call)
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: List[int],
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            inplace: Optional[bool] = True,
            bias: bool = True,
            dropout: float = 0.0,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim, eps=0.001))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))

        super().__init__(*layers)