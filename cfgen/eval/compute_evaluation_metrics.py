from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from cfgen.eval.distribution_distances import compute_distribution_distances

def process_labels(adata_original, adata_generated, category_field, categorical_obs=False):
    """
    Converts numeric labels to strings in the generated AnnData object.

    Args:
        adata_original (AnnData): Original AnnData object containing the real data.
        adata_generated (AnnData): Generated AnnData object to process labels for.
        category_field (str): Field name for the categorical labels.
        categorical_obs (bool, optional): Flag to indicate if the labels are categorical. Defaults to False.

    Returns:
        AnnData: Updated generated AnnData object with processed labels.
    """
    if not categorical_obs:
        label_unique = np.unique(adata_original.obs[category_field])
    else:
        label_unique = np.array(adata_original.obs[category_field].cat.categories)
    labels_dict = dict(zip(range(len(label_unique)), label_unique))
    adata_generated.obs[category_field] = [labels_dict[int(lab)] for lab in np.array(adata_generated.obs[category_field])]
    return adata_generated

def compute_evaluation_metrics(adata_real, 
                               adata_generated, 
                               category_field,
                               model_name,
                               nn=10, 
                               original_space=True, 
                               knn_pca=None, 
                               knn_data=None):  
    """
    Computes various evaluation metrics for generated data compared to real data.

    Args:
        adata_real (AnnData): Real AnnData object.
        adata_generated (AnnData): Generated AnnData object.
        category_field (str): Field name for the categorical labels.
        model_name (str): Name of the model being evaluated.
        nn (int, optional): Number of neighbors for KNN. Defaults to 10.
        original_space (bool, optional): Flag to indicate if the original feature space should be used. Defaults to True.
        knn_pca (KNeighborsClassifier, optional): KNN classifier trained on PCA-transformed data. Defaults to None.
        knn_data (KNeighborsClassifier, optional): KNN classifier trained on original data. Defaults to None.

    Returns:
        dict: Dictionary containing the computed evaluation metrics.
    """
    # Metric dictionary
    print(f"Evaluating for {model_name}")
    print("Real", adata_real.shape)
    print("Generated", adata_generated.shape)
    
    cell_type_metrics = {}

    # Compute Wasserstein distance and MMD metrics
    mmd_wasserstein = compute_distribution_distances(torch.tensor(adata_real.obsm["X_pca"]).float(), 
                                                     torch.tensor(adata_generated.obsm["X_pca"]).float())
    for metric in mmd_wasserstein:
        cell_type_metrics[metric + "_PCA"] = mmd_wasserstein[metric]
    
    return cell_type_metrics
