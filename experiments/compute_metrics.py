import scanpy as sc
from celldreamer.eval.compute_evaluation_metrics import compute_evaluation_metrics
from scipy import sparse
import pandas as pd
import numpy as np
import argparse

from celldreamer.eval.distribution_distances import train_knn_real_data

def add_to_dict(d, metrics):
    for metric in metrics:
        if metric not in d:
            d[metric] = [metrics[metric]]
        else:
            d[metric]+=[metrics[metric]]
    return d

def subsample_adata(adata, n_batches):
    idx = np.random.choice(range(len(adata)), n_batches)
    return adata[idx]

def evaluation(args, adata_real, adatas_generated, n_batches=None):
    if n_batches == None:
        results_celldreamer = compute_evaluation_metrics(adata_real, 
                                                         adatas_generated["celldreamer"], 
                                                         args.category_name,
                                                         "celldreamer",
                                                         nn=args.nn, 
                                                         original_space=True, 
                                                         knn_pca=None, 
                                                         knn_data=None)
        print("Finished celldreamer")
        results_activa = compute_evaluation_metrics(adata_real, 
                                                  adatas_generated["activa"],
                                                  args.category_name, 
                                                  "activa",
                                                  nn=args.nn, 
                                                  original_space=True,
                                                  knn_pca=None, 
                                                  knn_data=None)
        print("Finished scvi")
        results_scgan = compute_evaluation_metrics(adata_real, 
                                                    adatas_generated["scgan"], 
                                                    args.category_name, 
                                                    "scgan",
                                                    nn=args.nn, 
                                                    original_space=True, 
                                                    knn_pca=None, 
                                                    knn_data=None)
        
        print("Finished scvi")
        results_scrdit = compute_evaluation_metrics(adata_real, 
                                                    adatas_generated["scrdit"], 
                                                    args.category_name, 
                                                    "scrdit",
                                                    nn=args.nn, 
                                                    original_space=True, 
                                                    knn_pca=None, 
                                                    knn_data=None)
        print("Finished scDiff")
    else:
        adata_real = subsample_adata(adata_real, n_batches)
        results_celldreamer = compute_evaluation_metrics(adata_real, 
                                                         subsample_adata(adatas_generated["celldreamer"], n_batches), 
                                                         args.category_name,
                                                         "celldreamer",
                                                         nn=args.nn, 
                                                         original_space=True, 
                                                         knn_pca=None, 
                                                         knn_data=None)
        print("Finished celldreamer")
        results_activa = compute_evaluation_metrics(adata_real, 
                                                  subsample_adata(adatas_generated["activa"], n_batches), 
                                                  args.category_name, 
                                                  "activa",
                                                  nn=args.nn, 
                                                  original_space=True,
                                                  knn_pca=None, 
                                                  knn_data=None)
        print("Finished scvi")
        results_scgan = compute_evaluation_metrics(adata_real, 
                                                    subsample_adata(adatas_generated["scgan"], n_batches),
                                                    args.category_name, 
                                                    "scgan",
                                                    nn=args.nn, 
                                                    original_space=True,
                                                    knn_pca=None, 
                                                    knn_data=None)
        
        results_scrdit = compute_evaluation_metrics(adata_real, 
                                                    subsample_adata(adatas_generated["scrdit"], n_batches),
                                                    args.category_name, 
                                                    "scrdit",
                                                    nn=args.nn, 
                                                    original_space=True,
                                                    knn_pca=None, 
                                                    knn_data=None)
        print("Finished scDiff")
     
    return results_celldreamer, results_activa, results_scgan, results_scrdit

def main(args):
    # Read real dataset 
    adata_real = sc.read_h5ad(f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/datasets/processed_full_genome/{args.dataset_name}/{args.dataset_name}_test.h5ad")
    adata_real.X = adata_real.layers["X_counts"].copy()
    sc.pp.normalize_total(adata_real, target_sum=1e4)
    sc.pp.log1p(adata_real)
    
    vars = adata_real.var.copy()
    adata_real = adata_real[:, adata_real.var.highly_variable]
    sc.tl.pca(adata_real, n_comps=10)
    
    # Will contain results
    results_celldreamer = {}
    results_activa = {} 
    results_scgan = {}
    results_scrdit = {} 

    for i in range(3):
        # Dictionary with the read anndatas
        adatas = {}
        
        # Read fake datasets 
        adata_generated_path_celldreamer = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/datasets/generated/{args.dataset_name}/generated_cells_{i}.h5ad"
        adatas["celldreamer"] = sc.read_h5ad(adata_generated_path_celldreamer)

        adata_generated_path_activa = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/baseline_experiments/ACTIVA/generated/{args.dataset_name}/generated_cells_{i}.h5ad"
        adata_generated_activa = sc.read_h5ad(adata_generated_path_activa)
        adatas["activa"] = adata_generated_activa.copy()
        del adata_generated_activa
        
        adata_generated_path_scgan = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/baseline_experiments/scgan/model_runs/{args.dataset_name}/{args.dataset_name}_generated_{i}.h5ad"
        adata_generated_scgan = sc.read_h5ad(adata_generated_path_scgan)
        adatas["scgan"] = adata_generated_scgan
        del adata_generated_scgan
        
        adata_generated_path_scrdit = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/baseline_experiments/scRDiT/samples_h5ad/{args.dataset_name}/generated_cells_{i}.h5ad"
        adata_generated_scrdit = sc.read_h5ad(adata_generated_path_scrdit)
        adatas["scrdit"] = adata_generated_scrdit
        del adata_generated_scrdit
        
        # Keep only HVG
        for adata_name in adatas:
            if args.dataset_name == "hlca_core" and adata_name=="scgan":
                adatas[adata_name] = adatas[adata_name][:, adata_real.var.index]
            else:
                adatas[adata_name].var = vars
                adatas[adata_name] = adatas[adata_name][:, vars.highly_variable]
            adatas[adata_name].obsm["X_pca"] = adatas[adata_name].X.A.dot(adata_real.varm["PCs"])
        
        if args.batched:
            for _ in range(args.n_batches):
                results_celldreamer_b, results_activa_b, results_scgan_b, results_scrdit_b = evaluation(args, 
                                                                                     adata_real, 
                                                                                     adatas, 
                                                                                     n_batches=args.batch_size)
                results_celldreamer = add_to_dict(results_celldreamer, results_celldreamer_b)
                results_activa = add_to_dict(results_activa, results_activa_b)
                results_scgan = add_to_dict(results_scgan, results_scgan_b)
                results_scrdit = add_to_dict(results_scrdit, results_scrdit_b)
        else:
            results_celldreamer_i, results_activa_i, results_scgan_i, results_scrdit_i = evaluation(args, 
                                                                                                    adata_real, 
                                                                                                    adatas, 
                                                                                                    n_batches=None)
            results_celldreamer = add_to_dict(results_celldreamer, results_celldreamer_i)
            results_activa = add_to_dict(results_activa, results_activa_i)
            results_scgan = add_to_dict(results_scgan, results_scgan_i)
            results_scrdit = add_to_dict(results_scrdit, results_scrdit_i)
        
    results_celldreamer_df = pd.DataFrame(results_celldreamer)
    results_activa_df = pd.DataFrame(results_activa)
    results_scgan_df = pd.DataFrame(results_scgan)
    results_scrdit_df = pd.DataFrame(results_scrdit)
    
    results_celldreamer_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/results/{args.dataset_name}/celldreamer_{args.dataset_name}")
    results_activa_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/results/{args.dataset_name}/activa_{args.dataset_name}")
    results_scgan_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/results/{args.dataset_name}/scgan_{args.dataset_name}")
    results_scrdit_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/results/{args.dataset_name}/scrdit_{args.dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to be evaluated")
    parser.add_argument("--category_name", type=str, help="Category of AnnData containing the conditioning var")
    parser.add_argument("--batched", default=False, action='store_true', help="Whether to perform batched evaluation")
    parser.add_argument("--batch_size", type=int, default=None, help="The size of the batches")
    parser.add_argument("--n_batches", type=int, default=None, help="Number of batches to evaluate")
    parser.add_argument("--nn", type=int, help="Number of neighbors")
    args = parser.parse_args()
    main(args)
    