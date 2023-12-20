import numpy as np
import scanpy as sc
import torch
from celldreamer.data.utils import Scaler, normalize_expression, compute_size_factor_lognorm

class RNAseqLoader:
    """Class for RNAseq data loader."""
    def __init__(
        self,
        data_path: str,
        layer_key: str,
        covariate_keys=None,
        subsample_frac=1,
        use_pca=True,
        encoder_type="proportions", 
        target_max=1, 
        target_min=-1):
        """
        Initialize the RNAseqLoader.

        Args:
            data_path (str): Path to the data.
            layer_key (str): Layer key.
            covariate_keys (list, optional): List of covariate names. Defaults to None.
            subsample_frac (float, optional): Fraction of the dataset to use. Defaults to 1.
            use_pca (bool, optional): Use principal components for generation. Defaults to True.
            encoder_type (str, optional): Must be in (proportions, log_gexp, log_gexp_scaled).
            target_max (float, optional): Maximum value for scaling gene expression. Defaults to 1.
            target_min (float, optional): Minimum value for scaling gene expression. Defaults to 1.
        """
        # Read adata
        adata = sc.read(data_path)
        # Subsample if required
        if subsample_frac < 1:
            sc.pp.subsample(adata, fraction=subsample_frac)
        # Transform genes to tensors
        if use_pca:
            self.X = torch.Tensor(adata.obsm["X_pca"])
            self.X = (self.X - self.X.mean(1).unsqueeze(-1)) / self.X.std(1).unsqueeze(-1)
        else:
            if layer_key in adata.layers:
                self.X = torch.Tensor(adata.layers[layer_key].todense())
            else:
                self.X = torch.Tensor(adata.X.todense())
        
        # Get normalized gene expression 
        self.X_norm = normalize_expression(self.X, self.X.sum(1).unsqueeze(1), encoder_type)
        
        # Initialize scaler object 
        self.scaler = Scaler(target_max=target_max, target_min=target_min)
        self.scaler.fit(self.X_norm)
        
        # Compute mean and logvar of size factor
        self.log_size_factor_mu, self.log_size_factor_sd = compute_size_factor_lognorm(self.X)
        
        # Covariate to index
        self.id2cov = {}  # cov_name: dict_cov_2_id 
        self.Y_cov = {}   # cov: cov_ids
        for cov_name in covariate_keys:
            cov = np.array(adata.obs[cov_name])
            unique_cov = np.unique(cov)
            zip_cov_cat = dict(zip(unique_cov, np.arange(len(unique_cov))))  
            self.id2cov[cov_name] = zip_cov_cat
            self.Y_cov[cov_name] = torch.tensor([zip_cov_cat[c] for c in cov])
    
    def get_scaler(self):
        """Return the scaler object
        """
        return self.scaler
    
    def __getitem__(self, i):
        """
        Get item from the dataset.

        Args:
            i (int): Index.

        Returns:
            dict: Dictionary containing X (gene expression) and y (covariates).
        """
        X = self.X[i]
        X_norm = self.X_norm[i]
        y = {"y_" + cov: self.Y_cov[cov][i] for cov in self.Y_cov}
        return dict(X=X, X_norm=X_norm, y=y)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.X)
    