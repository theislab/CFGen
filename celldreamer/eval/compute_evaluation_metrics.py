from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from scib.metrics.lisi import ilisi_graph
from celldreamer.eval.distribution_distances import (compute_distribution_distances, 
                                                     compute_knn_real_fake, 
                                                     train_knn_real_data,
                                                     compute_prdc)


def compute_evaluation_metrics(adata_real, adata_generated, category_field):
    # Get unique cell type list
    cell_types_unique = np.unique(adata_real.obs[category_field])
    
    # Metrics per cell_type
    metrics = ["1-Wasserstein", 
               "2-Wasserstein",
               "Linear_MMD", 
               "Poly_MMD",
               "KNN identity",
               "KNN identity PCA",
               "KNN category", 
               "KNN category PCA",
               "precision", 
               "recall",
               "density",
               "coverage"]
    
    cell_type_metrics = {metric: [] for metric in metrics}
    # Train cell type classification KNNs
    knn_pca = train_knn_real_data(adata_real, category_field, use_pca=True)
    knn_data = train_knn_real_data(adata_real, category_field, use_pca=False)
    
    adata_generated.obsm["X_pca"] =  adata_generated.X.dot(adata_real.varm["PCs"])
    # Loop over cell type 
    for cell_type in cell_types_unique:
        # MMD and Wasserstein 
        adata_real_cell_type = adata_real[adata_real.obs[category_field]==cell_type]
        adata_generated_cell_type = adata_generated[adata_generated.obs[category_field]==cell_type]
        # adata_generated_cell_type.obsm["X_pca"] =  adata_generated_cell_type.X.dot(adata_real_cell_type.varm["PCs"])
        
        # MMD and Wasserstein
        mmd_wasserstein = compute_distribution_distances(torch.tensor(adata_real_cell_type.X.A), 
                                                         torch.tensor(adata_generated_cell_type.X.A))
        
        for metric in mmd_wasserstein:
            cell_type_metrics[metric].append(mmd_wasserstein[metric])
        
        # KNN identity data space 
        auc_real_fake = compute_knn_real_fake(adata_real_cell_type.X.A, 
                                              adata_generated_cell_type.X.A, n_neighbors=5)
        cell_type_metrics["KNN identity"].append(auc_real_fake)
        
        # KNN identity pca space 
        auc_real_fake = compute_knn_real_fake(adata_real_cell_type.obsm["X_pca"], 
                                              adata_generated_cell_type.obsm["X_pca"], n_neighbors=5)
        cell_type_metrics["KNN identity PCA"].append(auc_real_fake)
        
        # KNN cell type data
        y_pred = knn_data.predict(adata_generated_cell_type.X.A)    
        accuracy = accuracy_score(np.array(adata_generated_cell_type.obs[category_field]), y_pred)
        cell_type_metrics["KNN category"].append(accuracy)
        
        # KNN cell type pca
        y_pred = knn_pca.predict(adata_generated_cell_type.obsm["X_pca"])
        accuracy = accuracy_score(adata_generated_cell_type.obs[category_field], y_pred)
        cell_type_metrics["KNN category PCA"].append(accuracy)
        
        # KNN cell type pca
        density_and_coverage = compute_prdc(adata_real_cell_type.X, 
                               adata_generated_cell_type.X, 
                               nearest_k=5)
        for metric in density_and_coverage:
            cell_type_metrics[metric].append(density_and_coverage[metric])
    
    return cell_type_metrics       
        