import numpy as np
import scanpy as sc
import torch
from celldreamer.data.utils import indx

class RNAseqLoader:
    def __init__(
        self,
        data_path: str,
        covariate_keys=None,
        subsample_frac=1,
        use_pca=True, 
        n_dimensions=50
        ): 
        # Read adata
        data = sc.read(data_path)
        
        # Subsample if required
        if subsample_frac < 1:
            sc.pp.subsample(data, fraction=subsample_frac)
            
        # Transform genes to tensors
        if use_pca:
            self.genes = torch.Tensor(data.obsm["X_pca"])[:, :n_dimensions]
        else:
            self.genes = torch.Tensor(data.X)
        
        # Covariate to index
        self.id2cov = {}
        self.cov_values = {}
        self.covariate_names_unique = {}
        for cov_name in covariate_keys:
            cov = np.array(data.obs[cov_name])
            unique_cov = np.unique(cov)
            zip_cov_cat = dict(zip(unique_cov, 
                                   np.arange(len(unique_cov))))  
            self.id2cov[cov_name] = zip_cov_cat
            self.cov_values[cov_name] = torch.tensor([zip_cov_cat[c] for c in cov])
            self.covariate_names_unique[cov_name] = np.unique(cov)
            
    def __getitem__(self, i):
        X = self.genes[i]
        y = {"y_"+cov: self.cov_values[cov][i] for cov in self.cov_values}
        return ({"X": X, "y": y})

    def __len__(self):
        return len(self.genes) 
    