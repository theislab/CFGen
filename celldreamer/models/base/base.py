# This is the BaseModel from Sfairazero
from typing import Callable, List, Optional
import torch
import torch.nn as nn


def is_last_layer(i, units):
    return i == (len(units) - 2)


def _get_norm_layer(batch_norm: bool, layer_norm: bool):
    if batch_norm:
        norm_layer = nn.BatchNorm1d
    elif layer_norm:
        norm_layer = nn.LayerNorm
    else:
        norm_layer = None

    return norm_layer


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
