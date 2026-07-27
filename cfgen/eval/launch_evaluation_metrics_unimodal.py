"""
Evaluation utilities for comparing real vs. generated single-cell (unimodal) data.

This module turns the logic from `evaluate_metrics_unimodal.ipynb` into
reusable functions:

    from cfgen.eval.evaluate_metrics_unimodal import (
        evaluate_unimodal_generation,       # compare two .h5ad files on disk
        generate_and_evaluate_unimodal,      # load ckpts, generate, compare -- all in memory
    )

    # Compare two existing files
    results_df, summary = evaluate_unimodal_generation(
        real_path="/path/to/real.h5ad",
        generated_path="/path/to/generated.h5ad",
        counts_layer="X_counts",
        cluster_key="clusters",
        modality_name="cfgen_rna",
    )

    # Generate from a trained model checkpoint and compare directly
    results_df, summary, adata_generated = generate_and_evaluate_unimodal(
        dataset_conf_path="../../configs/configs_sccfm/dataset/dentategyrus.yaml",
        encoder_conf_path="../../configs/configs_encoder/encoder/default.yaml",
        generative_model_conf_path="../../configs/configs_sccfm/generative_model/default.yaml",
        autoencoder_ckpt_path="../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_dentategyrus_final/checkpoints/last.ckpt",
        fm_ckpt_path="../../project_folder/experiments/cfgen_ckpt/train_fm_dentategyrus_final/last.ckpt",
    )
"""

import math
import random
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from omegaconf import OmegaConf

from cfgen.eval.compute_evaluation_metrics import compute_evaluation_metrics, process_labels
from cfgen.data.scrnaseq_loader import RNAseqLoader
from cfgen.models.base.encoder_model import EncoderModel
from cfgen.models.fm.fm import FM


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _add_to_dict(d: dict, metrics: dict) -> dict:
    """Append each value in `metrics` to the corresponding list in `d`."""
    for metric in metrics:
        if metric not in d:
            d[metric] = [metrics[metric]]
        else:
            d[metric] += [metrics[metric]]
    return d


def _preprocess_real(
    adata_real: AnnData,
    counts_layer: str,
    n_top_genes: int,
    n_pcs: int,
    target_sum: float,
):
    """Restore counts, compute HVGs, normalize/log1p, run PCA on the real data.

    Returns the processed (HVG-subset) adata_real, the full var table
    (with the `highly_variable` column) so it can be copied onto the
    generated data, and the per-gene mean (on the same HVG-subset,
    log-normalized expression `sc.tl.pca` was fit on). `sc.tl.pca` uses
    `zero_center=True` by default, i.e. it mean-centers each gene before
    computing PCs, but `adata.varm["PCs"]` only stores the rotation
    matrix -- projecting new data via a plain dot product requires
    subtracting this same mean first.
    """
    adata_real = adata_real.copy()
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

    adata_real = adata_real[:, adata_real.var.highly_variable]
    real_gene_mean = np.asarray(adata_real.X.mean(axis=0)).ravel()
    return adata_real, vars_rna, real_gene_mean


def _compare_real_generated(
    adata_real: AnnData,
    adata_generated: AnnData,
    counts_layer: str = "X_counts",
    cluster_key: str = "clusters",
    modality_name: str = "cfgen_rna",
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    target_sum: float = 1e4,
    generated_is_normalized: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Core comparison logic, operating on in-memory AnnData objects.

    Both `evaluate_unimodal_generation` (loads from disk) and
    `generate_and_evaluate_unimodal` (generates in memory) call this.

    `generated_is_normalized` should be set when `adata_generated.X` is
    already total-count-normalized and log1p-transformed (e.g. baselines
    such as scDiffusion that generate directly in that space), so it isn't
    double-normalized before projecting into the real data's PCA space.
    Whatever total-count scale the baseline's own output actually lands on
    is left as-is -- if its decoder doesn't reconstruct the normalized
    scale well, that's the baseline's own generative quality, not something
    for this eval code to correct.
    """
    adata_real, vars_rna, real_gene_mean = _preprocess_real(
        adata_real, counts_layer, n_top_genes, n_pcs, target_sum
    )
    celltype_unique = np.unique(adata_real.obs[cluster_key])

    adata_generated = adata_generated.copy()
    adata_generated.var = vars_rna
    if not generated_is_normalized:
        # Same total-count normalization as the real data, computed on the
        # full gene panel (before HVG subsetting) so size factors are
        # comparable.
        sc.pp.normalize_total(adata_generated, target_sum=target_sum)
        sc.pp.log1p(adata_generated)
    adata_generated = adata_generated[:, adata_generated.var.highly_variable]
    # Subtract the real data's per-gene mean before projecting: sc.tl.pca
    # mean-centers before computing PCs, but varm["PCs"] only holds the
    # rotation matrix, so a plain dot product here would leave real and
    # generated cells in two different (offset) coordinate systems.
    adata_generated.obsm["X_pca"] = (
        (adata_generated.X.toarray() - real_gene_mean).dot(adata_real.varm["PCs"])
    )

    results: dict = {}
    for ct in celltype_unique:
        adata_real_ct = adata_real[adata_real.obs[cluster_key] == ct]
        adata_generated_ct = adata_generated[
            adata_generated.obs[cluster_key] == ct
        ]

        # A single real (or generated) cell can't support a distributional
        # comparison -- Wasserstein/MMD would degenerate to a point-to-point
        # distance rather than measure distribution overlap -- so skip it.
        if adata_real_ct.n_obs <= 1 or adata_generated_ct.n_obs <= 1:
            print(
                f"Skipping cell type {ct!r}: {adata_real_ct.n_obs} real / "
                f"{adata_generated_ct.n_obs} generated cell(s) (need > 1 on both sides)"
            )
            continue

        metrics_ct = compute_evaluation_metrics(
            adata_real_ct, adata_generated_ct, modality_name
        )
        metrics_ct["ct"] = ct
        results = _add_to_dict(results, metrics_ct)

    results_df = pd.DataFrame(results)
    summary = results_df.mean(numeric_only=True)
    return results_df, summary


# --------------------------------------------------------------------------
# Public: compare two files on disk
# --------------------------------------------------------------------------

def evaluate_unimodal_generation(
    real_path: str,
    generated_path: str,
    counts_layer: str = "X_counts",
    cluster_key: str = "clusters",
    modality_name: str = "cfgen_rna",
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    target_sum: float = 1e4,
    generated_is_normalized: bool = False,
    generated_labels_are_codes: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute per-cell-type and aggregate generation-quality metrics
    (Wasserstein distances, MMD, etc.) between a real and a generated
    AnnData object loaded from disk.

    `generated_is_normalized` should be set for baselines (e.g.
    scDiffusion) that generate directly in normalized+log1p space, so
    that space isn't re-normalized before PCA projection.

    `generated_labels_are_codes` should be set when
    `adata_generated.obs[cluster_key]` holds integer class codes rather
    than the real string labels (again, e.g. scDiffusion) -- codes are
    decoded via `process_labels`, assuming the same
    `{0: label0, 1: label1, ...}` convention (sorted unique real labels)
    used when the baseline was conditioned on these classes.

    See module docstring for parameter details.
    """
    adata_real = sc.read_h5ad(real_path)
    adata_generated = sc.read_h5ad(generated_path)
    if generated_labels_are_codes:
        adata_generated = process_labels(
            adata_real, adata_generated, cluster_key, categorical_obs=False
        )
    return _compare_real_generated(
        adata_real,
        adata_generated,
        counts_layer=counts_layer,
        cluster_key=cluster_key,
        modality_name=modality_name,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        target_sum=target_sum,
        generated_is_normalized=generated_is_normalized,
    )


# --------------------------------------------------------------------------
# Public: load a trained model from checkpoints, generate, and evaluate
# --------------------------------------------------------------------------

def _generate_n_cells_for_class(
    generative_model,
    n_cells: int,
    class_id: int,
    log_size_factors_ct: torch.Tensor,
    cluster_key: str,
    theta_covariate: str,
    size_factor_covariate: str,
    conditioning_covariates: List[str],
    batch_size: int,
    n_sample_steps: int,
    device: str,
) -> torch.Tensor:
    """Generate exactly `n_cells` cells for a single cell type / class.

    `log_size_factors_ct` holds the (already log-transformed) size factors
    of the real cells of this class, one-to-one, so the generated cells are
    conditioned on the same size-factor distribution as their real
    counterparts. If `batch_size` does not evenly divide `n_cells`, the
    class/size-factor tensors are cycled to fill the last batch and the
    output is sliced back down to exactly `n_cells`.
    """
    if n_cells == 0:
        return torch.empty(0)

    eff_batch_size = min(batch_size, n_cells)
    repetitions = math.ceil(n_cells / eff_batch_size)
    total = eff_batch_size * repetitions

    # Cycle the real size factors / class id to fill `total` slots.
    reps = math.ceil(total / len(log_size_factors_ct))
    log_sf_padded = log_size_factors_ct.repeat(reps)[:total]
    class_tensor = torch.full((total,), class_id, dtype=torch.long)

    covariate_indices = {cluster_key: class_tensor}
    log_size_factor = {"rna": log_sf_padded.to(device).view(-1, 1)}

    X_generated = generative_model.batched_sample(
        batch_size=eff_batch_size,
        repetitions=repetitions,
        n_sample_steps=n_sample_steps,
        theta_covariate=theta_covariate,
        size_factor_covariate=size_factor_covariate,
        conditioning_covariates=conditioning_covariates,
        covariate_indices=covariate_indices,
        log_size_factor=log_size_factor,
    )
    X_generated = X_generated["rna"].to("cpu")
    return X_generated[:n_cells]


def generate_and_evaluate_unimodal(
    dataset_conf_path: str,
    encoder_conf_path: str,
    generative_model_conf_path: str,
    autoencoder_ckpt_path: str,
    fm_ckpt_path: str,
    real_adata_path: Optional[str] = None,
    counts_layer: str = "X_counts",
    cluster_key: str = "clusters",
    theta_covariate: str = "clusters",
    size_factor_covariate: str = "clusters",
    conditioning_covariates: Optional[List[str]] = None,
    guidance_weights: Optional[Dict[str, float]] = None,
    latent_dim: int = 50,
    is_binarized: bool = False,
    modality_list: Optional[List[str]] = None,
    modality_name: str = "cfgen_rna",
    batch_size: int = 350,
    n_sample_steps: int = 2,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    target_sum: float = 1e4,
    device: str = "cuda",
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, AnnData]:
    """
    Load a trained autoencoder + flow-matching checkpoint, generate exactly
    as many cells per cell type as the real (test) data contains, and
    evaluate the generated cells against the real ones in memory (no
    intermediate .h5ad is written).

    This mirrors the "Generate and save cells" section of
    `dentategyrus.ipynb`, but generates cluster-by-cluster so the
    generated cell-type composition matches the real data exactly, rather
    than sampling a fixed number of cells and hoping proportions line up.

    Parameters
    ----------
    dataset_conf_path, encoder_conf_path, generative_model_conf_path
        Paths to the OmegaConf yaml configs (dataset, encoder,
        generative model), same as loaded at the top of the notebook.
    autoencoder_ckpt_path
        Path to the trained `EncoderModel` checkpoint
        (`.../autoencoder_ckpt/.../checkpoints/last.ckpt`).
    fm_ckpt_path
        Path to the trained `FM` checkpoint
        (`.../cfgen_ckpt/.../last.ckpt`). Its `hyper_parameters` are used
        to reconstruct `denoising_model` and `feature_embeddings`.
    real_adata_path
        Path to the real data to compare against. If None, defaults to
        `dataset_conf.dataset_path` with "train" replaced by "test"
        (matching the notebook).
    counts_layer, cluster_key
        Same meaning as in `evaluate_unimodal_generation`.
    theta_covariate, size_factor_covariate, conditioning_covariates
        Passed through to `FM(...)` and `batched_sample(...)`. Default to
        `"clusters"` / `["clusters"]` as in the notebook -- change these
        if your model conditions on something else.
    guidance_weights
        Passed to `FM(...)`. Defaults to `{cluster_key: 1}`.
    latent_dim
        Encoder latent dim used to build `in_dim={"rna": latent_dim}` for
        `FM(...)`. Defaults to 50, matching the notebook.
    is_binarized, modality_list
        Passed to `FM(...)`. `modality_list` defaults to `["rna"]`.
    batch_size, n_sample_steps
        Passed to `generative_model.batched_sample(...)` per cell type.
        Number of repetitions is computed automatically so that exactly
        as many cells as present in the real data (for that cell type)
        are generated.
    n_top_genes, n_pcs, target_sum
        Passed through to the evaluation step (`_compare_real_generated`).
    device
        Device to run generation on (default "cuda").
    seed
        If given, seeds `random` and `torch` before sampling classes /
        generating, for reproducibility.

    Returns
    -------
    results_df
        DataFrame with one row per cell type and one column per metric.
    summary
        Series with the mean of each metric across cell types.
    adata_generated
        The generated AnnData (counts in `.layers[counts_layer]` and
        `.X`, cell type in `.obs[cluster_key]`), returned so it can be
        inspected or saved (`adata_generated.write_h5ad(...)`) if wanted.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    conditioning_covariates = conditioning_covariates or [cluster_key]
    guidance_weights = guidance_weights or {cluster_key: 1}
    modality_list = modality_list or ["rna"]

    # --- Load configs ---
    dataset_conf = OmegaConf.load(dataset_conf_path)
    encoder_conf = OmegaConf.load(encoder_conf_path)
    generative_model_config = OmegaConf.load(generative_model_conf_path)

    # --- Dataset (needed for in_dim / size_factor stats / id2cov) ---
    dataset = RNAseqLoader(
        dataset_conf.dataset_path,
        layer_key=dataset_conf.layer_key,
        covariate_keys=dataset_conf.covariate_keys,
        subsample_frac=dataset_conf.subsample_frac,
        normalization_type=dataset_conf.normalization_type,
        is_binarized=dataset_conf.is_binarized,
    )
    size_factor_statistics = {
        "mean": dataset.log_size_factor_mu,
        "sd": dataset.log_size_factor_sd,
    }
    gene_dim = {mod: dataset.X[mod].shape[1] for mod in dataset.X}

    # --- Encoder ---
    encoder_model = EncoderModel(
        in_dim=gene_dim,
        n_cat=None,
        conditioning_covariate=dataset_conf.theta_covariate,
        **encoder_conf,
    )
    encoder_model.load_state_dict(
        torch.load(autoencoder_ckpt_path, weights_only=False)["state_dict"]
    )

    # --- Flow-matching generative model ---
    ckpt = torch.load(fm_ckpt_path, weights_only=False)
    denoising_model = ckpt["hyper_parameters"]["denoising_model"]
    denoising_model.multimodal = False
    feature_embeddings = ckpt["hyper_parameters"]["feature_embeddings"]

    generative_model = FM(
        encoder_model=encoder_model,
        denoising_model=denoising_model,
        feature_embeddings=feature_embeddings,
        plotting_folder=None,
        in_dim={"rna": latent_dim},
        size_factor_statistics=size_factor_statistics,
        covariate_list=[cluster_key],
        theta_covariate=theta_covariate,
        size_factor_covariate=size_factor_covariate,
        is_binarized=is_binarized,
        modality_list=modality_list,
        guidance_weights=guidance_weights,
        **generative_model_config,
    )
    generative_model.load_state_dict(ckpt["state_dict"])
    generative_model.to(device)
    generative_model.eval()

    # --- Real (test) data ---
    if real_adata_path is None:
        real_adata_path = dataset_conf["dataset_path"].replace("train", "test")
    adata_real = sc.read_h5ad(real_adata_path)
    adata_real.X = adata_real.layers[counts_layer].copy()

    # --- Generate exactly as many cells per cell type as in the real data ---
    celltype_unique = np.unique(adata_real.obs[cluster_key])
    X_generated_chunks = []
    classes_str: List[str] = []
    skipped_celltypes: List[str] = []

    for ct in celltype_unique:
        mask = (adata_real.obs[cluster_key] == ct).values
        n_ct = int(mask.sum())

        # A single test cell can't support a distributional comparison
        # (Wasserstein/MMD degenerate to a point-to-point distance), and a
        # batch size of 1 breaks the ODE sampler's timestep embedding, so
        # skip these cell types entirely rather than generating for them.
        if n_ct <= 1:
            skipped_celltypes.append(ct)
            continue

        log_size_factors_ct = torch.log(
            torch.tensor(np.asarray(adata_real.layers[counts_layer][mask].sum(1)).ravel())
        ).float()

        class_id = dataset.id2cov[cluster_key][ct]
        X_ct = _generate_n_cells_for_class(
            generative_model=generative_model,
            n_cells=n_ct,
            class_id=class_id,
            log_size_factors_ct=log_size_factors_ct,
            cluster_key=cluster_key,
            theta_covariate=theta_covariate,
            size_factor_covariate=size_factor_covariate,
            conditioning_covariates=conditioning_covariates,
            batch_size=batch_size,
            n_sample_steps=n_sample_steps,
            device=device,
        )
        X_generated_chunks.append(X_ct.numpy())
        classes_str.extend([ct] * n_ct)

    if skipped_celltypes:
        print(
            f"Skipping {len(skipped_celltypes)} cell type(s) with <= 1 test "
            f"cell (no distributional comparison possible): {skipped_celltypes}"
        )
        adata_real = adata_real[~adata_real.obs[cluster_key].isin(skipped_celltypes)].copy()

    X_generated = np.concatenate(X_generated_chunks, axis=0)

    adata_generated = AnnData(
        X=X_generated,
        obs=pd.DataFrame({cluster_key: classes_str}),
    )
    adata_generated.layers[counts_layer] = adata_generated.X.copy()

    # --- Evaluate ---
    results_df, summary = _compare_real_generated(
        adata_real,
        adata_generated,
        counts_layer=counts_layer,
        cluster_key=cluster_key,
        modality_name=modality_name,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        target_sum=target_sum,
    )
    return results_df, summary, adata_generated
 