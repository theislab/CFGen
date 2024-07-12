import math
from typing import Union

import numpy as np
import torch
import sklearn 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from celldreamer.eval.mmd import linear_mmd2, poly_mmd2
from celldreamer.eval.optimal_transport import wasserstein


def compute_distribution_distances(pred: torch.Tensor, true: Union[torch.Tensor, list]):
    """
    Computes distances between predicted and true distributions.

    Args:
        pred (torch.Tensor): Predicted tensor of shape [batch, times, dims].
        true (Union[torch.Tensor, list]): True tensor of shape [batch, times, dims] or list of tensors of length times.

    Returns:
        dict: Dictionary containing the computed distribution distances.
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
    Computes pairwise distances between two datasets.

    Args:
        data_x (np.ndarray): Array of shape [N, feature_dim].
        data_y (np.ndarray, optional): Array of shape [N, feature_dim]. Defaults to None.

    Returns:
        np.ndarray: Array of shape [N, N] containing pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='l1', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Gets the k-th smallest value along the specified axis.

    Args:
        unsorted (np.ndarray): Unsorted array of any dimensionality.
        k (int): The k-th index.

    Returns:
        np.ndarray: Array containing k-th smallest values along the specified axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Computes distances to the k-th nearest neighbours.

    Args:
        input_features (np.ndarray): Array of shape [N, feature_dim].
        nearest_k (int): The number of nearest neighbours.

    Returns:
        np.ndarray: Distances to the k-th nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features (np.ndarray): Array of real features of shape [N, feature_dim].
        fake_features (np.ndarray): Array of fake features of shape [N, feature_dim].
        nearest_k (int): Number of nearest neighbours.

    Returns:
        dict: Dictionary containing precision, recall, density, and coverage.
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


def compute_knn_real_fake(X_real, X_fake, n_neighbors=5):
    """
    Computes F1 score using k-nearest neighbours classifier for real and fake data.

    Args:
        X_real (np.ndarray): Array of real features.
        X_fake (np.ndarray): Array of fake features.
        n_neighbors (int, optional): Number of neighbours. Defaults to 5.

    Returns:
        float: F1 score.
    """
    X = np.concatenate((X_real, X_fake), axis=0)
    y = np.concatenate((np.ones(len(X_real)), np.zeros(len(X_fake))), axis=0)

    # Initialize KNN classifier
    # knn = RandomForestClassifier()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the classifier
    knn.fit(X, y)

    # Evaluate the classifier
    y_pred = knn.predict(X)
    auc = f1_score(y, y_pred, average="macro")
    return auc


def train_knn_real_data(adata_real, category_field, use_pca, n_neighbors=5):
    """
    Trains a k-nearest neighbours classifier on real data.

    Args:
        adata_real (AnnData): Annotated Data object containing real data.
        category_field (str): The category field to be used as the target variable.
        use_pca (bool): Whether to use PCA-transformed data.
        n_neighbors (int, optional): Number of neighbours. Defaults to 5.

    Returns:
        KNeighborsClassifier: Trained KNN classifier.
    """
    if not use_pca:
        X = adata_real.X  # Features
    else:
        X = adata_real.obsm["X_pca"] 
        
    y = adata_real.obs[category_field]  # Target variable

    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)  # You can adjust the number of neighbors
    # knn = RandomForestClassifier()    

    # Fit the classifier to the training data
    knn.fit(X, y)
    return knn
