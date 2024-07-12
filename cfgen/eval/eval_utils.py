import scanpy as sc
from cfgen.eval.distribution_distances import compute_distribution_distances
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch

def normalize_and_compute_metrics(adata_real, adata_fake, layer, sparse=True):
    """
    Normalize data from AnnData objects and compute distribution distances between real and fake data.

    Args:
        adata_real (AnnData): Annotated Data object containing real data.
        adata_fake (AnnData): Annotated Data object containing fake data.
        layer (str or None): Key for the layer in `obsm` to use for data. If None, uses `X`.
        sparse (bool, optional): Whether the data is in sparse format. Default is True.

    Returns:
        tuple: A tuple containing:
            - dict: Metrics computed from the distribution distances.
            - torch.Tensor: Normalized real data.
            - torch.Tensor: Normalized fake data.
    """
    # Extract data from the specified layer or `X`
    if layer is not None:
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
    
    # Normalize the data to the range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_real = torch.tensor(scaler.fit_transform(X_real))
    normalized_generated = torch.tensor(scaler.fit_transform(X_fake))
    
    # Compute distribution distances
    metrics = compute_distribution_distances(normalized_real, normalized_generated)
    
    return metrics, normalized_real, normalized_generated
