from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from celldreamer.eval.distribution_distances import (compute_distribution_distances, 
                                                     compute_knn_real_fake, 
                                                     train_knn_real_data,
                                                     compute_prdc)

CONDITIONAL = {"scDiffusion": True, 
               "scgan": False, 
               "scvi": True, 
               "celldreamer": True, 
               "activa": False, 
               "scrdit": False}

def process_labels(adata_original, adata_generated, category_field, categorical_obs=False):
    """
    If the labels are numeric, you convert them to strings 
    """
    if not categorical_obs:
        label_unique = np.unique(adata_original.obs[category_field])
    else:
        label_unique = np.array(adata_original.obs[category_field].cat.categories)
    labels_dict = dict(zip(range(len(label_unique)),label_unique))
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
    Compute metrics 
    """
    # Metric dict
    print(f"Evaluating for {model_name}")
    print("Real", adata_real.shape)
    print("Generated", adata_generated.shape)
    
    cell_type_metrics = {}

    # Wasserstein distance and MMD metrics 
    mmd_wasserstein = compute_distribution_distances(torch.tensor(adata_real.obsm["X_pca"]).float(), 
                                                         torch.tensor(adata_generated.obsm["X_pca"]).float())
    for metric in mmd_wasserstein:
        cell_type_metrics[metric+"_PCA"] = mmd_wasserstein[metric]
    
    # KNN metric 
    auc_real_fake = compute_knn_real_fake(adata_real.X.A, 
                                              adata_generated.X.A, n_neighbors=nn)
    auc_real_fake_pca = compute_knn_real_fake(adata_real.obsm["X_pca"], 
                                              adata_generated.obsm["X_pca"], n_neighbors=nn)
    cell_type_metrics["KNN identity"] = auc_real_fake
    cell_type_metrics["KNN identity PCA"] = auc_real_fake_pca
     
    # KNN cell type pca
    density_and_coverage = compute_prdc(adata_real.X.A, 
                                            adata_generated.X.A, 
                                            nearest_k=nn)
    for metric in density_and_coverage:
        cell_type_metrics[metric] = density_and_coverage[metric]

    density_and_coverage_pca = compute_prdc(adata_real.obsm["X_pca"], 
                        adata_generated.obsm["X_pca"], 
                        nearest_k=nn)
    for metric in density_and_coverage_pca:
        cell_type_metrics[metric+"_PCA"] = density_and_coverage_pca[metric]
    
      # Train cell type classification KNNs
    if knn_data:
        y_pred = knn_data.predict(adata_generated.X.A)    
        accuracy = f1_score(np.array(adata_generated.obs[category_field]), y_pred, average="macro")
        cell_type_metrics["KNN category"] = accuracy
    
    # KNN cell type pca
    if knn_pca:
        y_pred = knn_pca.predict(adata_generated.obsm["X_pca"])
        accuracy = f1_score(adata_generated.obs[category_field], y_pred, average="macro")
        cell_type_metrics["KNN category PCA"] = accuracy
    return cell_type_metrics      
