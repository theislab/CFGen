import scanpy as sc
from celldreamer.eval.compute_evaluation_metrics import process_labels, compute_evaluation_metrics
from scipy import sparse
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import f1_score

from celldreamer.eval.distribution_distances import train_knn_real_data

def add_to_dict(d, metrics):
    for metric in metrics:
        if metric not in d:
            d[metric] = [metrics[metric]]
        else:
            d[metric]+=[metrics[metric]]
    return d

def evaluation(args, adata_real, adatas_generated, knn_pca=None, knn_data=None):
    results_celldreamer = compute_evaluation_metrics(adata_real["disc"], 
                                                        adatas_generated["celldreamer"], 
                                                        args.category_name,
                                                        "celldreamer",
                                                        nn=args.nn, 
                                                        original_space=True, 
                                                        knn_pca=knn_pca["disc"], 
                                                        knn_data=knn_data["disc"])
    print("Finished celldreamer")
    results_scvi = compute_evaluation_metrics(adata_real["disc"], 
                                                adatas_generated["scvi"],
                                                args.category_name, 
                                                "scvi",
                                                nn=args.nn, 
                                                original_space=True,
                                                knn_pca=knn_pca["disc"], 
                                                knn_data=knn_data["disc"])
    print("Finished scvi")
    results_scdiff = compute_evaluation_metrics(adata_real["disc"], 
                                                adatas_generated["scDiffusion"], 
                                                args.category_name, 
                                                "scDiffusion",
                                                nn=args.nn, 
                                                original_space=True, 
                                                knn_pca=knn_pca["disc"], 
                                                knn_data=knn_data["disc"])
    print("Finished scDiff")
    return results_celldreamer, results_scvi, results_scdiff

def main(args):
    # Read real dataset 
    adata_real_cont = sc.read_h5ad(f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/datasets/processed_full_genome/{args.dataset_name}/{args.dataset_name}_test.h5ad")
    # Fixes for when dataset is neurips 
    adata_real_discr = adata_real_cont.copy()
    adata_real_discr.X = adata_real_discr.layers["X_counts"]
    sc.pp.normalize_total(adata_real_discr, target_sum=1e4)
    sc.pp.log1p(adata_real_discr)
    adata_real = {"cont": adata_real_cont,
                  "disc": adata_real_discr}
    del adata_real_cont
    del adata_real_discr
    
    celltype_unique = np.unique(adata_real["cont"].obs[args.category_name])  # unique cell type 
    n_obs = adata_real["cont"].shape[0]
    vars = adata_real["cont"].var.copy()
    adata_real = {mod: adata_real[mod][:, adata_real[mod].var.highly_variable] for mod in adata_real}  # Keep only highly variable 
    sc.tl.pca(adata_real["cont"], n_comps=30)
    sc.tl.pca(adata_real["disc"], n_comps=30)
    
    # Get PCA score 
    knn_pca_disc = train_knn_real_data(adata_real["disc"], args.category_name, use_pca=True, n_neighbors=args.nn)
    knn_data_disc = train_knn_real_data(adata_real["disc"], args.category_name, use_pca=False, n_neighbors=args.nn)
    knn_pca_cont = train_knn_real_data(adata_real["cont"], args.category_name, use_pca=True, n_neighbors=args.nn)
    knn_data_cont = train_knn_real_data(adata_real["cont"], args.category_name, use_pca=False, n_neighbors=args.nn)
    knn_data = {"cont": knn_data_cont, "disc": knn_data_disc}
    knn_pca = {"cont": knn_pca_cont, "disc": knn_pca_disc}
    del knn_pca_disc
    del knn_data_disc
    del knn_pca_cont
    del knn_data_cont
    
    # Will contain results
    results_celldreamer = {}
    results_scvi = {} 
    results_scdiff = {}

    for i in range(3):
        # Dictionary with the read anndatas
        adatas = {}
        
        # Read fake datasets 
        adata_generated_path_celldreamer = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/datasets/generated/{args.dataset_name}/generated_cells_{i}.h5ad"
        adatas["celldreamer"] = sc.read_h5ad(adata_generated_path_celldreamer)
        del adata_generated_path_celldreamer

        adata_generated_path_scvi = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/baseline_experiments/scvi/{args.dataset_name}/generated/{args.dataset_name}_{i}.h5ad"
        adata_generated_scvi = sc.read_h5ad(adata_generated_path_scvi)
        adata_generated_scvi.X = sparse.csr_matrix(adata_generated_scvi.X)  # convert to sparse
        adata_generated_scvi = process_labels(adata_real["disc"], adata_generated_scvi, args.category_name, categorical_obs=True)
        adatas["scvi"] = adata_generated_scvi.copy()
        del adata_generated_scvi
       
        adata_generated_path_scDiffusion = f"/home/icb/alessandro.palma/environment/celldreamer/project_folder/baseline_experiments/scDiffusion/generated/{args.dataset_name}/generated_cells_{i}.h5ad"
        adata_generated_scDiffusion = sc.read_h5ad(adata_generated_path_scDiffusion)[-n_obs:]
        adata_generated_scDiffusion = process_labels(adata_real["cont"], adata_generated_scDiffusion, args.category_name, categorical_obs=False)
        adatas["scDiffusion"] = adata_generated_scDiffusion.copy()
        del adata_generated_scDiffusion
        
        # Keep only HVG
        for adata_name in adatas:
            adatas[adata_name].var = vars
            adatas[adata_name] = adatas[adata_name][:, vars.highly_variable]
        
        adatas["celldreamer"].obsm["X_pca"] = adatas["celldreamer"].X.A.dot(adata_real["disc"].varm["PCs"])
        adatas["scvi"].obsm["X_pca"] = adatas["scvi"].X.A.dot(adata_real["disc"].varm["PCs"])
        adatas["scDiffusion"].obsm["X_pca"] = adatas["scDiffusion"].X.A.dot(adata_real["cont"].varm["PCs"])
    
        for ct in celltype_unique:
            adata_real_ct = {}
            adata_real_ct["cont"] = adata_real["cont"][adata_real["cont"].obs[args.category_name]==ct]
            adata_real_ct["disc"] = adata_real["disc"][adata_real["disc"].obs[args.category_name]==ct]
            
            adatas_ct = {model: adatas[model][adatas[model].obs[args.category_name]==ct] for model in adatas}
            
            freqs = np.array([len(adatas_ct[model]) for model in adatas_ct])
            if (freqs==0).any() or (freqs<20).any() or len(adata_real_ct["cont"])<20:
                continue
            
            if adata_real_ct["cont"].shape[0] > 3000:
                sc.pp.subsample(adata_real_ct["cont"], n_obs=3000)
                sc.pp.subsample(adata_real_ct["disc"], n_obs=3000)
                for model in adatas_ct:
                    if adatas_ct[model].shape[0] > 3000:
                        sc.pp.subsample(adatas_ct[model], n_obs=3000)

            results_celldreamer_ct, results_scvi_ct, results_scdiff_ct = evaluation(args, 
                                                                                    adata_real_ct, 
                                                                                    adatas_ct, 
                                                                                    knn_data=knn_data,
                                                                                    knn_pca=knn_pca)
            results_celldreamer_ct["ct"] = ct
            results_scvi_ct["ct"] = ct
            results_scdiff_ct["ct"] = ct
            
            results_celldreamer = add_to_dict(results_celldreamer, results_celldreamer_ct)
            results_scvi = add_to_dict(results_scvi, results_scvi_ct)
            results_scdiff = add_to_dict(results_scdiff, results_scdiff_ct)
        
        results_celldreamer["global_f1"] = f1_score(np.array(adatas["celldreamer"].obs[args.category_name]), 
                                                    y_pred = knn_data["disc"].predict(adatas["celldreamer"].X.A), 
                                                    average="macro")
        results_scvi["global_f1"] = f1_score(np.array(adatas["scvi"].obs[args.category_name]), 
                                                    y_pred = knn_data["disc"].predict(adatas["scvi"].X.A), 
                                                    average="macro")
        results_scdiff["global_f1"] = f1_score(np.array(adatas["scDiffusion"].obs[args.category_name]), 
                                                    y_pred = knn_data["disc"].predict(adatas["scDiffusion"].X.A), 
                                                    average="macro")

    results_celldreamer_df = pd.DataFrame(results_celldreamer)
    results_scvi_df = pd.DataFrame(results_scvi)
    results_scdiff_df = pd.DataFrame(results_scdiff)
    
    results_celldreamer_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/results_cell_type/{args.dataset_name}/celldreamer_{args.dataset_name}")
    results_scvi_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/results_cell_type/{args.dataset_name}/scvi_{args.dataset_name}")
    results_scdiff_df.to_csv(f"/home/icb/alessandro.palma/environment/celldreamer/experiments/results_cell_type/{args.dataset_name}/scdiff_{args.dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to be evaluated")
    parser.add_argument("--category_name", type=str, help="Category of AnnData containing the conditioning var")
    parser.add_argument("--nn", type=int, help="Number of neighbors")
    args = parser.parse_args()
    main(args)
    