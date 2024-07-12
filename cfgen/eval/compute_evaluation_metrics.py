from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from cfgen.eval.distribution_distances import (compute_distribution_distances, 
                                                     compute_knn_real_fake, 
                                                     compute_prdc)

# Conditional dictionary for models
CONDITIONAL = {"scDiffusion": True, 
               "scgan": False, 
               "scvi": True, 
               "cfgen": True, 
               "activa": False, 
               "scrdit": False}

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
    
    # Compute KNN identity metrics
    auc_real_fake = compute_knn_real_fake(adata_real.X.A, 
                                          adata_generated.X.A, n_neighbors=nn)
    auc_real_fake_pca = compute_knn_real_fake(adata_real.obsm["X_pca"], 
                                              adata_generated.obsm["X_pca"], n_neighbors=nn)
    cell_type_metrics["KNN identity"] = auc_real_fake
    cell_type_metrics["KNN identity PCA"] = auc_real_fake_pca
     
    # Compute PRDC metrics in original space
    density_and_coverage = compute_prdc(adata_real.X.A, 
                                        adata_generated.X.A, 
                                        nearest_k=nn)
    for metric in density_and_coverage:
        cell_type_metrics[metric] = density_and_coverage[metric]

    # Compute PRDC metrics in PCA space
    density_and_coverage_pca = compute_prdc(adata_real.obsm["X_pca"], 
                                            adata_generated.obsm["X_pca"], 
                                            nearest_k=nn)
    for metric in density_and_coverage_pca:
        cell_type_metrics[metric + "_PCA"] = density_and_coverage_pca[metric]
    
    # Train and evaluate KNN classifier for cell type classification on original data
    if knn_data:
        y_pred = knn_data.predict(adata_generated.X.A)    
        accuracy = f1_score(np.array(adata_generated.obs[category_field]), y_pred, average="macro")
        cell_type_metrics["KNN category"] = accuracy
    
    # Train and evaluate KNN classifier for cell type classification on PCA data
    if knn_pca:
        y_pred = knn_pca.predict(adata_generated.obsm["X_pca"])
        accuracy = f1_score(adata_generated.obs[category_field], y_pred, average="macro")
        cell_type_metrics["KNN category PCA"] = accuracy
    
    return cell_type_metrics
