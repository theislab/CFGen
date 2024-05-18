import scanpy as sc
from celldreamer.eval.distribution_distances import compute_distribution_distances
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import numpy as np

def normalize_and_compute_metrics(adata_real, adata_fake, layer, sparse=True):
    if layer!= None:
        if sparse:
            X_real = adata_real.obsm[layer].A
            X_fake = adata_fake.obsm[layer].A
        else:
            X_real = adata_real.obsm[layer]
            X_fake = adata_fake.obsm[layer]
    else:
        if sparse:
            X_real = adata_real.X.A
            X_fake = adata_fake.X.A
        else:
            X_real = adata_real.X
            X_fake = adata_fake.X
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_real = torch.tensor(scaler.fit_transform(X_real))
    normalized_generated = torch.tensor(scaler.fit_transform(X_fake))
    metrics = compute_distribution_distances(normalized_real, normalized_generated)
    return metrics, normalized_real, normalized_generated


def join_real_generated(adata_real, adata_generated, preprocess, classes_str, covariates):
    adata_real = adata_real.copy()
    adata_generated = adata_generated.copy()
    if preprocess:
        sc.pp.normalize_total(adata_real, target_sum=1e4)
        sc.pp.log1p(adata_real)
        sc.pp.normalize_total(adata_generated, target_sum=1e4)
        sc.pp.log1p(adata_generated)
        # sc.tl.pca(adata_real)
        # sc.tl.pca(adata_generated)
   
    # adata_generated.obsm["X_pca"]=adata_generated.X.dot(adata_real.varm["PCs"])
    
    adata_concat = sc.AnnData(X=np.concatenate([adata_real.X, 
                                                adata_generated.X], axis=0))
    # adata_concat.obsm["X_pca"] = np.concatenate([adata_real.obsm["X_pca"],
    #                                              adata_generated.obsm["X_pca"]], axis=0)
    sc.tl.pca(adata_concat)
    
    dataset_type = ["Real" for _ in range(adata_real.shape[0])] + ["Generated" for _ in range(adata_generated.shape[0])]
    dataset_type = pd.DataFrame(dataset_type)
    dataset_type.columns = ["dataset_type"]
    for cov in covariates:
        dataset_type[cov] = list(adata_real.obs[cov])+classes_str[cov]
    adata_concat.obs = dataset_type
    return adata_concat
