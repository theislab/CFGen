"""
Evaluation utilities for comparing real vs. generated single-cell (unimodal) data.
 
This module turns the logic from `evaluate_metrics_unimodal.ipynb` into a
single reusable function, `evaluate_unimodal_generation`, that can be
imported anywhere in the repo, e.g.:
 
    from cfgen.eval.evaluate_metrics_unimodal import evaluate_unimodal_generation
 
    results_df, summary = evaluate_unimodal_generation(
        real_path="/path/to/real.h5ad",
        generated_path="/path/to/generated.h5ad",
        counts_layer="X_counts",
        cluster_key="clusters",
        modality_name="cfgen_rna",
    )
"""
 
from typing import Tuple
 
import numpy as np
import pandas as pd
import scanpy as sc
 
from cfgen.eval.compute_evaluation_metrics import compute_evaluation_metrics
 
 
def _add_to_dict(d: dict, metrics: dict) -> dict:
    """Append each value in `metrics` to the corresponding list in `d`."""
    for metric in metrics:
        if metric not in d:
            d[metric] = [metrics[metric]]
        else:
            d[metric] += [metrics[metric]]
    return d
 
 
def evaluate_unimodal_generation(
    real_path: str,
    generated_path: str,
    counts_layer: str = "X_counts",
    cluster_key: str = "clusters",
    modality_name: str = "cfgen_rna",
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    target_sum: float = 1e4,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute per-cell-type and aggregate generation-quality metrics
    (Wasserstein distances, MMD, etc.) between a real and a generated
    AnnData object.
 
    Parameters
    ----------
    real_path
        Path to the real (ground-truth) .h5ad file.
    generated_path
        Path to the generated .h5ad file.
    counts_layer
        Layer in `adata_real` holding raw counts, used both to restore
        `.X` before normalization and to compute HVGs.
    cluster_key
        Column in `.obs` identifying cell type / cluster labels. Must be
        present in both real and generated AnnData objects.
    modality_name
        Label passed to `compute_evaluation_metrics` to identify the
        modality being evaluated (kept configurable in case you evaluate
        more than RNA).
    n_top_genes
        Number of highly variable genes to select.
    n_pcs
        Number of principal components to compute on the real data;
        generated data is projected into this same PCA space.
    target_sum
        Target sum used for `sc.pp.normalize_total` on the real data.
 
    Returns
    -------
    results_df
        DataFrame with one row per cell type and one column per metric
        (plus a "ct" column identifying the cell type).
    summary
        Series with the mean of each metric across cell types
        (numeric columns only).
    """
    # --- Load data ---
    adata_real = sc.read_h5ad(real_path)
    adata_generated = sc.read_h5ad(generated_path)
 
    # --- Preprocess real data ---
    # Restore raw counts before HVG/normalization
    adata_real.X = adata_real.layers[counts_layer].copy()
 
    sc.pp.highly_variable_genes(
        adata_real,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        layer=counts_layer,
        subset=False,
    )
    vars_rna = adata_real.var.copy()
 
    sc.pp.normalize_total(adata_real, target_sum=target_sum)
    sc.pp.log1p(adata_real)
    sc.tl.pca(adata_real, n_comps=n_pcs)
 
    celltype_unique = np.unique(adata_real.obs[cluster_key])
    adata_real = adata_real[:, adata_real.var.highly_variable]
 
    # --- Align and project generated data into real PCA space ---
    adata_generated.var = vars_rna
    adata_generated = adata_generated[:, adata_generated.var.highly_variable]
    adata_generated.obsm["X_pca"] = (
        adata_generated.X.toarray().dot(adata_real.varm["PCs"])
    )
 
    # --- Compute metrics per cell type ---
    results: dict = {}
    for ct in celltype_unique:
        adata_real_ct = adata_real[adata_real.obs[cluster_key] == ct]
        adata_generated_ct = adata_generated[
            adata_generated.obs[cluster_key] == ct
        ]
 
        metrics_ct = compute_evaluation_metrics(
            adata_real_ct, adata_generated_ct, modality_name
        )
        metrics_ct["ct"] = ct
        results = _add_to_dict(results, metrics_ct)
 
    results_df = pd.DataFrame(results)
    summary = results_df.mean(numeric_only=True)
 
    return results_df, summary
 
    