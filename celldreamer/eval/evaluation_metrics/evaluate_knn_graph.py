import numpy as np
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as skm
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd



# KNN similarity metric
def knn_graph_metric(X, X_simulated, k):
    # Extract X and categorical data from real and simulated datasets
    X_concat = np.concatenate((X, X_simulated))
    n_sim = X_simulated.shape[0]
    # TODO: add comparison in category from Till's code 

    # Build kNN graph for simulated data
    neigh = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    neigh.fit(X_concat)
    sim_indices = np.arange(X.shape[0], X_concat.shape[0])
    distances, indices = neigh.kneighbors(X_simulated)

    # Compute evaluation metrics
    real_counts = np.sum(indices[:, 1:] < X.shape[0], axis=1)
    prop_real = real_counts / k
    prop_sim = 1 - prop_real
    prop_expected = np.full(n_sim, 0.5)
    sim_entropy = -np.sum((prop_sim * np.log2(prop_sim)) + (prop_real * np.log2(prop_real)))

    # Return evaluation metrics
    return {
        'mean_prop_real': np.mean(prop_real),
        'mean_prop_sim': np.mean(prop_sim),
        'mean_prop_expected': np.mean(prop_expected),
        'sim_entropy': sim_entropy
    }
