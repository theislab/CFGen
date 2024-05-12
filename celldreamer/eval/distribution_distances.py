import math
from typing import Union

import numpy as np
import torch
import sklearn 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

from celldreamer.eval.mmd import linear_mmd2, poly_mmd2
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

def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='l1', n_jobs=8)
    return dists

def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()
    
    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)
    
# def compute_knn_real_fake(X_real, X_fake, n_neighbors=5):
#     X = np.concatenate((X_real, X_fake), axis=0)
#     y = np.concatenate((np.ones(len(X_real)), np.zeros(len(X_fake))), axis=0)

#     # Initialize KNN classifier
#     knn = KNeighborsClassifier(n_neighbors=n_neighbors)

#     # Train the classifier
#     knn.fit(X, y)

#     # Evaluate the classifier
#     y_pred = knn.predict(X_fake)
#     auc = f1_score(np.ones(len(X_fake)), y_pred, average="macro")
#     return auc

def compute_knn_real_fake(X_real, X_fake, n_neighbors=5):
    X = np.concatenate((X_real, X_fake), axis=0)
    y = np.concatenate((np.ones(len(X_real)), np.zeros(len(X_fake))), axis=0)

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the classifier
    knn.fit(X, y)

    # Evaluate the classifier
    y_pred = knn.predict(X)
    auc = f1_score(y, y_pred, average="macro")
    return auc

def train_knn_real_data(adata_real, category_field, use_pca, n_neighbors=5):
    if not use_pca:
        X = adata_real.X  # Features
    else:
        X = adata_real.obsm["X_pca"] 
        
    y = adata_real.obs[category_field]  # Target variable

    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)  # You can adjust the number of neighbors

    # Fit the classifier to the training data
    knn.fit(X, y)
    return knn
