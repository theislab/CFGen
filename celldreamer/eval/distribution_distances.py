import math
from typing import Union

import numpy as np
import torch

from celldreamer.eval.mmd import linear_mmd2, mix_rbf_mmd2, poly_mmd2
from celldreamer.eval.optimal_transport import wasserstein

def compute_distribution_distances(pred: torch.Tensor, true: Union[torch.Tensor, list]):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    """
    min_size = min(pred.shape[0], true.shape[0])
    
    names = [
        "1-Wasserstein",
        "2-Wasserstein",
        "Linear_MMD",
        "Poly_MMD"
    ]
    dists = []
    to_return = []
    w1 = wasserstein(pred, true, power=1)
    w2 = wasserstein(pred, true, power=2)
    pred_4_mmd = pred[:min_size]
    true_4_mmd = true[:min_size]
    mmd_linear = linear_mmd2(pred_4_mmd, true_4_mmd).item()
    mmd_poly = poly_mmd2(pred_4_mmd, true_4_mmd).item()
    dists.append((w1, w2, mmd_linear, mmd_poly))

    to_return.extend(np.array(dists).mean(axis=0))
    return dict(zip(names, to_return))
