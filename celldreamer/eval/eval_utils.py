import scanpy as sc
from celldreamer.eval.distribution_distances import compute_distribution_distances
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch

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
