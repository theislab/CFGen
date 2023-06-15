import math
import torch
from typing import Optional

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def x0_to_xt(x_0: torch.Tensor, alpha_hat_t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute x_t from x_0 using a closed form using theorem from original DDPM paper (Ho et al.)
    :param x_0: the image without noise
    :param alpha_hat_t: the cumulated variance schedule at time t
    :param eps: pure noise from N(0, 1)
    :return: the noised image x_t at step t
    """
    if eps is None:
        eps = torch.randn_like(x_0)
    return torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * eps

def identity(t, *args, **kwargs):
    return t
