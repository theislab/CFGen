import scanpy as sc
from celldreamer.eval.compute_evaluation_metrics import process_labels, compute_evaluation_metrics
from scipy import sparse
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import f1_score
import mudata as mu
from muon import atac as ac

from celldreamer.eval.distribution_distances import train_knn_real_data

def add_to_dict(d, metrics):
    for metric in metrics:
        if metric not in d:
            d[metric] = [metrics[metric]]
        else:
            d[metric]+=[metrics[metric]]
    return d

def main(args):
    # Read real dataset 
    adata_real = mu.read(f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/datasets/processed/atac/pbmc/pbmc10k_multiome_test.h5mu")
    
    # RNA 
    adata_real_rna = adata_real["rna"]
    adata_real_rna.X = adata_real_rna.layers["X_counts"].copy()
    sc.pp.highly_variable_genes(adata_real_rna,
                                flavor="seurat_v3",
                                n_top_genes=2000,
                                layer="X_counts",
                                subset=False)
    sc.pp.normalize_total(adata_real_rna, target_sum=1e4)
    sc.pp.log1p(adata_real_rna)
    sc.tl.pca(adata_real_rna, n_comps=30)

    # ATAC 
    adata_real_atac = adata_real["atac"]
    adata_real_atac.obs["cell_type"] = adata_real_rna.obs["cell_type"]  # Harmonize annotation
    adata_real_atac.X = adata_real_atac.layers["X_counts"].copy()
    ac.pp.tfidf(adata_real_atac, scale_factor=1e4)
    sc.pp.highly_variable_genes(adata_real_atac, n_top_genes=10000, subset=False)
    sc.tl.pca(adata_real_rna, n_comps=30)
    
    del adata_real
    
    celltype_unique = np.unique(adata_real_rna.obs[args.category_name])  # unique cell type 
    vars_rna = adata_real_rna.var.copy()
    vars_atac = adata_real_atac.var.copy()
    adata_real_rna = adata_real_rna[:, adata_real_rna.var.highly_variable]
    adata_real_atac = adata_real_atac[:, adata_real_atac.var.highly_variable]
    
    # Get PCA score 
    knn_pca_rna = train_knn_real_data(adata_real_rna, args.category_name, use_pca=True, n_neighbors=args.nn)
    knn_data_rna = train_knn_real_data(adata_real_rna, args.category_name, use_pca=False, n_neighbors=args.nn)
    knn_pca_atac = train_knn_real_data(adata_real_atac, args.category_name, use_pca=True, n_neighbors=args.nn)
    knn_data_atac = train_knn_real_data(adata_real_atac, args.category_name, use_pca=False, n_neighbors=args.nn)

    # Will contain results
    results_celldreamer_atac = {}
    results_celldreamer_rna = {}
    results_peakvi_atac = {}
    results_multivi_atac = {}
    results_multivi_rna = {}
    results_scvi_rna = {}

    for i in range(3):
        # Dictionary with the read anndatas
        adatas = {}
        
        # Read fake datasets 
        adata_generated_path_celldreamer_rna = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/datasets/generated/pbmc10k_multimodal/generated_cells_{i}_rna.h5ad"
        adata_generated_celldreamer_rna = sc.read_h5ad(adata_generated_path_celldreamer_rna)
        adata_generated_celldreamer_rna.var = vars_rna
        adata_generated_celldreamer_rna = adata_generated_celldreamer_rna[:, adata_generated_celldreamer_rna.var.highly_variable]
        adata_generated_celldreamer_rna.obsm["X_pca"] = adata_generated_celldreamer_rna.X.A.dot(adata_real_rna.varm["PCs"])
        
        adata_generated_path_celldreamer_atac = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/datasets/generated/pbmc10k_multimodal/generated_cells_{i}_atac.h5ad"
        adata_generated_celldreamer_atac = sc.read_h5ad(adata_generated_path_celldreamer_atac)
        adata_generated_celldreamer_atac.var = vars_atac
        ac.pp.tfidf(adata_generated_celldreamer_atac, scale_factor=1e4)
        adata_generated_celldreamer_atac = adata_generated_celldreamer_atac[:, adata_generated_celldreamer_atac.var.highly_variable]
        adata_generated_celldreamer_atac.obsm["X_pca"] = adata_generated_celldreamer_atac.X.A.dot(adata_real_atac.varm["PCs"])
        
        adata_generated_path_peakvi_atac = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/baseline_experiments/peakvi/generated/pbmc10k_multimodal_{i}.h5ad"
        adata_generated_peakvi_atac = sc.read_h5ad(adata_generated_path_peakvi_atac)
        adata_generated_peakvi_atac.layers["X_counts"] = np.where(adata_generated_peakvi_atac.layers["X_counts"]>0.5, 1., 0.)
        adata_generated_peakvi_atac.obs["cell_type"] = adata_real_atac.obs["cell_type"]
        adata_generated_peakvi_atac.X = sparse.csr_matrix(adata_generated_peakvi_atac.layers["X_counts"])
        adata_generated_peakvi_atac.var = vars_atac
        ac.pp.tfidf(adata_generated_peakvi_atac, scale_factor=1e4)
        adata_generated_peakvi_atac = adata_generated_peakvi_atac[:, adata_generated_peakvi_atac.var.highly_variable]
        adata_generated_peakvi_atac.obsm["X_pca"] = adata_generated_peakvi_atac.X.A.dot(adata_real_atac.varm["PCs"])

        adata_generated_path_multivi_atac = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/baseline_experiments/multivi/pbmc/generated/atac/pbmc10k_multimodal_{i}.h5ad"
        adata_generated_multivi_atac = sc.read_h5ad(adata_generated_path_multivi_atac)
        adata_generated_multivi_atac.layers["X_counts"] = np.where(adata_generated_multivi_atac.layers["X_counts"]>0.5, 1., 0.)
        adata_generated_multivi_atac.X = sparse.csr_matrix(adata_generated_multivi_atac.X)
        process_labels(adata_real_atac, adata_generated_multivi_atac, args.category_name, categorical_obs=True)
        adata_generated_multivi_atac.var = vars_atac
        ac.pp.tfidf(adata_generated_multivi_atac, scale_factor=1e4)
        adata_generated_multivi_atac = adata_generated_multivi_atac[:, adata_generated_multivi_atac.var.highly_variable]
        adata_generated_multivi_atac.obsm["X_pca"] = adata_generated_multivi_atac.X.A.dot(adata_real_atac.varm["PCs"])

        adata_generated_path_multivi_rna = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/baseline_experiments/multivi/pbmc/generated/expression/pbmc10k_multimodal_{i}.h5ad"
        adata_generated_multivi_rna = sc.read_h5ad(adata_generated_path_multivi_rna)
        adata_generated_multivi_rna.X = sparse.csr_matrix(adata_generated_multivi_rna.X)
        process_labels(adata_real_rna, adata_generated_multivi_rna, args.category_name, categorical_obs=True)
        adata_generated_multivi_rna.var = vars_rna
        adata_generated_multivi_rna = adata_generated_multivi_rna[:, adata_generated_multivi_rna.var.highly_variable]
        adata_generated_multivi_rna.obsm["X_pca"] = adata_generated_multivi_rna.X.dot(adata_real_rna.varm["PCs"])

        adata_generated_path_scvi_rna = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/baseline_experiments/scvi/pbmc10k_multimodal/generated/pbmc_multimodal_{i}.h5ad"
        adata_generated_scvi_rna = sc.read_h5ad(adata_generated_path_scvi_rna)
        adata_generated_scvi_rna.X = sparse.csr_matrix(adata_generated_scvi_rna.X)        
        process_labels(adata_real_rna, adata_generated_scvi_rna, args.category_name, categorical_obs=True)
        adata_generated_scvi_rna.var = vars_rna
        adata_generated_scvi_rna = adata_generated_scvi_rna[:, adata_generated_scvi_rna.var.highly_variable]
        adata_generated_scvi_rna.obsm["X_pca"] = adata_generated_scvi_rna.X.dot(adata_real_rna.varm["PCs"])
    
        for ct in celltype_unique:
            adata_real_ct_atac = adata_real_atac[adata_real_atac.obs[args.category_name]==ct]
            adata_real_ct_rna = adata_real_rna[adata_real_rna.obs[args.category_name]==ct]
            adata_generated_celldreamer_rna_ct = adata_generated_celldreamer_rna[adata_generated_celldreamer_rna.obs[args.category_name]==ct]
            adata_generated_celldreamer_atac_ct = adata_generated_celldreamer_atac[adata_generated_celldreamer_atac.obs[args.category_name]==ct]
            adata_generated_peakvi_atac_ct = adata_generated_peakvi_atac[adata_generated_peakvi_atac.obs[args.category_name]==ct]
            adata_generated_multivi_atac_ct = adata_generated_multivi_atac[adata_generated_multivi_atac.obs[args.category_name]==ct]
            adata_generated_multivi_rna_ct = adata_generated_multivi_rna[adata_generated_multivi_rna.obs[args.category_name]==ct]
            adata_generated_scvi_rna_ct = adata_generated_scvi_rna[adata_generated_scvi_rna.obs[args.category_name]==ct]
            if len(adata_real_ct_atac) < 20:
                continue
            results_celldreamer_rna_ct = compute_evaluation_metrics(adata_real_ct_rna, 
                                                                        adata_generated_celldreamer_rna_ct, 
                                                                        args.category_name,
                                                                        "celldreamer_rna",
                                                                        nn=args.nn, 
                                                                        original_space=True, 
                                                                        knn_pca=knn_pca_rna, 
                                                                        knn_data=knn_data_rna)
            
            results_celldreamer_atac_ct = compute_evaluation_metrics(adata_real_ct_atac, 
                                                                        adata_generated_celldreamer_atac_ct,
                                                                        args.category_name, 
                                                                        "celldreamer_atac",
                                                                        nn=args.nn, 
                                                                        original_space=True,
                                                                        knn_pca=knn_pca_atac, 
                                                                        knn_data=knn_data_atac)

            results_peakvi_atac_ct = compute_evaluation_metrics(adata_real_ct_atac, 
                                                                adata_generated_peakvi_atac_ct, 
                                                                args.category_name, 
                                                                "peakvi_atac",
                                                                nn=args.nn, 
                                                                original_space=True, 
                                                                knn_pca=knn_pca_atac, 
                                                                knn_data=knn_data_atac)
            
            results_multivi_rna_ct = compute_evaluation_metrics(adata_real_ct_rna, 
                                                                    adata_generated_multivi_rna_ct, 
                                                                    args.category_name,
                                                                    "multivi_rna",
                                                                    nn=args.nn, 
                                                                    original_space=True, 
                                                                    knn_pca=knn_pca_rna, 
                                                                    knn_data=knn_data_rna)
            
            results_multivi_atac_ct = compute_evaluation_metrics(adata_real_ct_atac, 
                                                                    adata_generated_multivi_atac_ct,
                                                                    args.category_name, 
                                                                    "multivi_atac",
                                                                    nn=args.nn, 
                                                                    original_space=True,
                                                                    knn_pca=knn_pca_atac, 
                                                                    knn_data=knn_data_atac)
            
            results_scvi_rna_ct = compute_evaluation_metrics(adata_real_ct_rna, 
                                                                adata_generated_scvi_rna_ct, 
                                                                args.category_name,
                                                                "celldreamer_rna",
                                                                nn=args.nn, 
                                                                original_space=True, 
                                                                knn_pca=knn_pca_rna, 
                                                                knn_data=knn_data_rna)
                        
            
            results_celldreamer_rna_ct["ct"] = ct
            results_celldreamer_atac_ct["ct"] = ct
            results_peakvi_atac_ct["ct"] = ct
            results_multivi_rna_ct["ct"] = ct
            results_multivi_atac_ct["ct"] = ct
            results_scvi_rna_ct["ct"] = ct
            
            results_celldreamer_rna = add_to_dict(results_celldreamer_rna, results_celldreamer_rna_ct)
            results_celldreamer_atac = add_to_dict(results_celldreamer_atac, results_celldreamer_atac_ct)
            results_peakvi_atac = add_to_dict(results_peakvi_atac, results_peakvi_atac_ct)
            results_multivi_atac = add_to_dict(results_multivi_atac, results_multivi_atac_ct)
            results_multivi_rna = add_to_dict(results_multivi_rna, results_multivi_rna_ct)
            results_scvi_rna = add_to_dict(results_scvi_rna, results_scvi_rna_ct)
            
        
        results_celldreamer_rna["global_f1"] = f1_score(np.array(adata_generated_celldreamer_rna.obs[args.category_name]), 
                                                    y_pred = knn_data_rna.predict(adata_generated_celldreamer_rna.X.A), 
                                                    average="macro")
        results_celldreamer_atac["global_f1"] = f1_score(np.array(adata_generated_celldreamer_atac.obs[args.category_name]), 
                                                    y_pred = knn_data_atac.predict(adata_generated_celldreamer_atac.X.A), 
                                                    average="macro")
        results_peakvi_atac["global_f1"] = f1_score(np.array(adata_generated_peakvi_atac.obs[args.category_name]), 
                                                    y_pred = knn_data_atac.predict(adata_generated_peakvi_atac.X.A), 
                                                    average="macro")
        results_multivi_atac["global_f1"] = f1_score(np.array(adata_generated_multivi_atac.obs[args.category_name]), 
                                                    y_pred = knn_data_atac.predict(adata_generated_multivi_atac.X.A), 
                                                    average="macro")
        results_multivi_rna["global_f1"] = f1_score(np.array(adata_generated_multivi_rna.obs[args.category_name]), 
                                                    y_pred = knn_data_rna.predict(adata_generated_multivi_rna.X.A), 
                                                    average="macro")
        results_scvi_rna["global_f1"] = f1_score(np.array(adata_generated_scvi_rna.obs[args.category_name]), 
                                                    y_pred = knn_data_rna.predict(adata_generated_scvi_rna.X.A), 
                                                    average="macro")

    results_celldreamer_rna_df = pd.DataFrame(results_celldreamer_rna)
    results_celldreamer_atac_df = pd.DataFrame(results_celldreamer_atac)
    results_peakvi_atac_df = pd.DataFrame(results_peakvi_atac)
    results_multivi_rna_df = pd.DataFrame(results_multivi_rna)
    results_multivi_atac_df = pd.DataFrame(results_multivi_atac)
    results_scvi_rna_df = pd.DataFrame(results_scvi_rna)
    
    results_celldreamer_rna_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/metrics_multimodal/results_pbmc/results_celldreamer_rna")
    results_celldreamer_atac_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/metrics_multimodal/results_pbmc/results_celldreamer_atac")
    results_peakvi_atac_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/metrics_multimodal/results_pbmc/results_peakvi_atac")
    results_multivi_rna_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/metrics_multimodal/results_pbmc/results_multivi_rna")
    results_multivi_atac_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/metrics_multimodal/results_pbmc/results_multivi_atac")
    results_scvi_rna_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/metrics_multimodal/results_pbmc/results_scvi_rna")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Name of the category --> cell type
    parser.add_argument("--category_name", type=str, help="Category of AnnData containing the conditioning var")
    # The number of nearest neighbors
    parser.add_argument("--nn", type=int, help="Number of neighbors")
    args = parser.parse_args()
    main(args)
    