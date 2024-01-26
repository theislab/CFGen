import math
from typing import Union

import numpy as np
import torch

from celldreamer.eval.mmd import linear_mmd2, mix_rbf_mmd2, poly_mmd2
from celldreamer.eval.optimal_transport import wasserstein

def compute_distances(pred, true):
    """computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred - true)).item()
    return mse, me, mae

def compute_distribution_distances(pred: torch.Tensor, true: Union[torch.Tensor, list]):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    """
    min_size = min(pred.shape[0], true.shape[0])
    
    NAMES = [
        "1-Wasserstein",
        "2-Wasserstein",
        "Linear_MMD",
        "Mean_MSE",
        "Mean_L2",
        "Mean_L1",
        "Median_MSE",
        "Median_L2",
        "Median_L1",
    ]
    is_jagged = isinstance(true, list)
    pred_is_jagged = isinstance(pred, list)
    dists = []
    to_return = []
    names = []
    filtered_names = [name for name in NAMES if not is_jagged or not name.endswith("MMD")]
    w1 = wasserstein(pred, true, power=1)
    w2 = wasserstein(pred, true, power=2)
    if not pred_is_jagged and not is_jagged:
        pred_4_mmd = pred[:min_size]
        true_4_mmd = true[:min_size]
        mmd_linear = linear_mmd2(pred_4_mmd, true_4_mmd).item()
    mean_dists = compute_distances(torch.mean(pred, dim=0), torch.mean(true, dim=0))
    median_dists = compute_distances(torch.median(pred, dim=0)[0], torch.median(true, dim=0)[0])
    dists.append((w1, w2, mmd_linear, *mean_dists, *median_dists))

    to_return.extend(np.array(dists).mean(axis=0))
    names.extend(filtered_names)
    return dict(zip(names, to_return))
