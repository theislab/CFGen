import scanpy as sc
import numpy as np

def join_real_generated(adata_real, adata_generated, preprocess, classes_str, covariates):
    if preprocess:
        sc.pp.normalize_total(adata_real, target_sum=1e4)
        sc.pp.log1p(adata_real)
        sc.pp.normalize_total(adata_generated, target_sum=1e4)
        sc.pp.log1p(adata_generated)
    
    adata_generated.obsm["X_pca"]=adata_generated.X.dot(adata_real.varm["PCs"])
    
    adata_concat = sc.AnnData(X=np.concatenate([adata_real.X.A, 
                                                adata_generated.X.A], axis=0))
    adata_concat.obsm["X_pca"] = np.concatenate([adata_real.obsm.["X_pca"],
                                                 adata_generated.obsm.["X_pca"]], axis=0)
    
    dataset_type = ["Real" for _ in range(adata_real.shape[0])] + ["Generated" for _ in range(adata_generated.shape[0])]
    dataset_type = pd.DataFrame(dataset_type)
    dataset_type.columns = ["dataset_type"]
    for cov in covariates:
        dataset_type[cov] = list(adata_original.obs.clusters)+classes_str[cov]
    return adata_concat