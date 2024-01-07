import torch
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out, trunc_normal_

# Std of standard normal truncated to [-2, 2].
STD_TRUNC_NORMAL = 0.87962566103423978


def scaled_trunc_normal_(
    tensor: torch.Tensor, gain: float = 1.0, mode: str = "fan_in"
) -> torch.Tensor:
    r"""Fills the input `Tensor` with values according to the truncated
    normal distribution. Follows standard flax initialization.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denominator = fan_in
    elif mode == "fan_out":
        denominator = fan_out
    elif mode == "fan_avg":
        denominator = (fan_in + fan_out) / 2.0
    else:
        raise ValueError(f"Invalid mode for initializer: {mode}")

    var = gain / denominator
    scale = (var**0.5) / STD_TRUNC_NORMAL
    trunc_normal_(
        tensor, mean=0.0, std=1.0, a=-2.0, b=2.0
    )  # Var(tensor) = STD_TRUNC_NORMAL ** 2
    tensor.mul_(scale)  # Var(tensor) = var
    return tensor

@torch.no_grad()
def default_init(weight: torch.torch.Tensor, bias: torch.Tensor = None):
    """Initialize weight and bias with default values."""
    scaled_trunc_normal_(weight, mode="fan_in")
    if bias is not None:
        nn.init.zeros_(bias)


class CustomInitLayerMixin(nn.Module):
    """Mixin class for custom initialization."""

    def reset_parameters(self):
        """Reset parameters."""
        default_init(self.weight, self.bias)


class Conv2d(CustomInitLayerMixin, nn.Conv2d):
    """Conv2d with different default initialization."""

    pass


class Linear(CustomInitLayerMixin, nn.Linear):
    """Linear with different default initialization."""

    pass