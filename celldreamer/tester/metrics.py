import numpy as np
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as skm
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd

def get_unique_attr_combinations(adata):
    """
    Get a list of dictionaries of all unique combinations of categories
    :param adata: adata with categories in obs
    :return: list of dictionaries of all unique combinations of categories
    """
    # Get a list of dictionaries of all unique combinations of categories
    unique_categories = []
    # Loop over cells (rows) in adata
    for i, cell in adata.obs.iterrows():
        # Load the set of categories as dictionary with category name as key and category value as value
        categories = cell.to_dict()
        # If categories is not in unique_categories, append it
        if categories not in unique_categories:
            unique_categories.append(categories)
    return unique_categories


# 1) Reconstruction loss
def get_reconstruction_loss(adata, adata_sim):

    real_unique_categories = get_unique_attr_combinations(adata)
    sim_unique_categories = get_unique_attr_combinations(adata_sim)

    # get overlapping categories
    overlapping_categories = []
    for real_category in real_unique_categories:
        for sim_category in sim_unique_categories:
            if real_category == sim_category:
                overlapping_categories.append(real_category)

    print('Out of {} unique categories in real data, {} categories are also in simulated data'.format(len(real_unique_categories), len(overlapping_categories)))

    mse = []
    r2 = []
    explained_variance = []

    for category in overlapping_categories:
        for i, cell in adata.obs.iterrows():
            true_categories = cell.to_dict()
            # check if all categories are the same
            if all(true_categories[category] == value for category, value in category.items()):
                reference_cell = adata.X[int(i)]

        for i, cell in adata_sim.obs.iterrows():
            sim_categories = cell.to_dict()
            # check if all categories are the same
            if all(sim_categories[category] == value for category, value in category.items()):
                sim_cell = adata_sim.X[int(i)]
                mse.append(skm.mean_squared_error(reference_cell, sim_cell))
                r2.append(skm.r2_score(reference_cell, sim_cell))
                explained_variance.append(skm.explained_variance_score(reference_cell, sim_cell))

    # Take the mean of the reconstruction errors for each category
    mse = np.mean(mse)
    r2 = np.mean(r2)
    explained_variance = np.mean(explained_variance)
    return mse, r2, explained_variance


# 2) Merel's metric
def evaluate_knn_graph(adata, adata_sim, k, cat_vars):
    """
    Evaluate the quality of a simulated kNN graph by comparing it to the real kNN graph.
    cat_vars:
    mean_prop_real: the average proportion of real cells among the k nearest neighbors of each simulated cell. Higher values indicate better simulated data.
    mean_prop_sim: the average proportion of simulated cells among the k nearest neighbors of each simulated cell. Lower values indicate better simulated data.
    mean_prop_expected: the expected proportion of real cells among the k nearest neighbors of each simulated cell, assuming an equal number of real and simulated cells. This value is always 0.5, but we compute it here for completeness.
    sim_entropy: the entropy of the proportion of simulated cells among the k nearest neighbors of each simulated cell. This metric measures how uniformly distributed the simulated cells are around each simulated cell. Lower values indicate better simulated data.
    cat_match: the fraction of the k nearest neighbors of each simulated cell that have the same categorical data as the simulated cell. Higher values indicate better simulated data.
    :param adata:
    :param adata_sim:
    :param k:
    :param cat_vars:
    :return: mean_prop_real, mean_prop_sim, mean_prop_expected, sim_entropy, cat_match
    """
    # Extract X and categorical data from real and simulated datasets
    X = np.concatenate((adata.X, adata_sim.X))
    cat_data = np.concatenate((adata.obs[cat_vars].values, adata_sim.obs[cat_vars].values))
    n_sim = adata_sim.X.shape[0]

    # Build kNN graph for simulated data
    neigh = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    neigh.fit(X)
    sim_indices = np.arange(adata.X.shape[0], X.shape[0])
    distances, indices = neigh.kneighbors(adata_sim.X)

    # Compute evaluation metrics
    real_counts = np.sum(indices[:, 1:] < adata.X.shape[0], axis=1)
    prop_real = real_counts / k
    prop_sim = 1 - prop_real
    prop_expected = np.full(n_sim, 0.5)
    sim_entropy = -np.sum((prop_sim * np.log2(prop_sim)) + (prop_real * np.log2(prop_real)))

    # Compute fraction of neighbors with the same categorical data
    cat_match = np.mean(np.all(cat_data[indices[:, 1:]] == np.repeat(adata_sim.obs[cat_vars].values[:, np.newaxis, :], k, axis=1), axis=2))

    # Return evaluation metrics
    return {
        'mean_prop_real': np.mean(prop_real),
        'mean_prop_sim': np.mean(prop_sim),
        'mean_prop_expected': np.mean(prop_expected),
        'sim_entropy': sim_entropy,
        'cat_match': cat_match
    }

# 3) Coverage and Density
def evaluate_coverage(adata, adata_sim):
    """
    Evaluate the coverage and density of simulated cells by comparing them to the real cells.
    stats:
    ann_real: the average nearest neighbor distance for the real cells.
    ann_sim: the average nearest neighbor distance for the simulated cells.
    coverage_real: the proportion of high-dimensional space covered by the real cells.
    coverage_sim: the proportion of high-dimensional space covered by the simulated cells.
    :param adata:
    :param adata_sim:
    :return: stats
    """
    # Compute average nearest neighbor distance (ANN) for real and simulated cells
    kdt_real = cKDTree(adata.X)
    kdt_sim = cKDTree(adata_sim.X)
    ann_real = kdt_real.query(adata.X, k=2)[0][:, 1].mean()
    ann_sim = kdt_real.query(adata_sim.X, k=2)[0][:, 1].mean()

    # Compute coverage of real and simulated cells
    hull_real = ConvexHull(adata.X)
    hull_sim = ConvexHull(adata_sim.X)
    volume_real = hull_real.volume
    volume_sim = hull_sim.volume
    intersection = ConvexHull(np.concatenate((adata.X, adata_sim.X))).volume
    coverage_real = intersection / volume_real
    coverage_sim = intersection / volume_sim

    # Return evaluation metrics
    return {
        'ann_real': ann_real,
        'ann_sim': ann_sim,
        'coverage_real': coverage_real,
        'coverage_sim': coverage_sim
    }


def evaluate_clustering(adata, adata_sim, categories):
    """
    ari: Adjusted Rand Index (ARI), which measures the similarity between two clustering assignments.
    The ARI takes into account chance agreement and adjusts the raw agreement by comparing the clustering assignments
    to a random baseline. A higher ARI indicates better agreement between the clustering assignments.
    Evaluate the clustering of simulated cells by comparing it to the real clustering.
    :param adata: adata object containing real data
    :param adata_sim: adata object containing simulated data
    :param categories: list of categorical variables to use for clustering
    :return: ari
    """
    # Combine real and simulated data
    adata_combined = adata.concatenate(adata_sim)

    # Concatenate all categorical variables into a single variable
    combined_categories = pd.concat(
        [adata_combined.obs[c].astype(str) for c in categories],
        axis=1,
        join="inner",
    ).apply(lambda x: "".join(x), axis=1).values

    # Compute clustering on combined data
    kmeans = KMeans(n_clusters=len(np.unique(combined_categories))).fit(
        adata_combined.X
    )

    # Compute ARI between true and predicted clusters
    ari = adjusted_rand_score(combined_categories, kmeans.labels_)

    return {"ari": ari}
