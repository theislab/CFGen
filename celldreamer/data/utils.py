import torch 

class Scaler:
    def __init__(self, target_min: int=-1, target_max: int=1):
        """Initialization function.

        Args:
            target_min (int): Minimum value for scaling.
            target_max (int): Maximum value for scaling.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_min = torch.tensor([target_min]).to(self.device)
        self.target_max = torch.tensor([target_max]).to(self.device)
    
    def fit(self, X):
        """Fit the scaler to the data.

        Args:
            X (torch.Tensor): Input data for fitting the scaler.
        """
        self.data_min = torch.min(X, dim=0, keepdim=True).values.to(self.device)
        self.data_max = torch.max(X, dim=0, keepdim=True).values.to(self.device)
        
    def scale(self, X, reverse=False):
        """Scale the input data.

        Args:
            X (torch.Tensor): Input data to be scaled.
            reverse (bool): If True, perform reverse scaling.

        Returns:
            torch.Tensor: Scaled data.
        """
        assert hasattr(self, "data_min"), "Run fit method on the dataset before calling transform"

        if not reverse:
            min_val, max_val = self.data_min, self.data_max
            new_min, new_max = self.target_min, self.target_max
        else:
            min_val, max_val = self.target_min, self.target_max
            new_min, new_max = self.data_min, self.data_max
        
        X_scaled = (X - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
        # Clip to the lower end
        if reverse:
            X_scaled = torch.clamp(X_scaled, min=new_min.to(X_scaled), max=new_max.to(X_scaled))
        return X_scaled

def normalize_expression(X, size_factor, encoder_type):
    """Normalize gene expression data based on the specified encoder type.

    Args:
        X (torch.Tensor): Input gene expression matrix.
        size_factor (torch.Tensor): Size factors for normalization.
        encoder_type (str): Type of encoder for normalization.

    Returns:
        torch.Tensor: Normalized gene expression data.
    """
    if encoder_type == "proportions":
        X = X / size_factor
    elif encoder_type == "log_gexp":
        X = torch.log1p(X)
    elif encoder_type == "log_gexp_scaled":
        X = torch.log1p(X / size_factor)
    else:
        raise NotImplementedError    
    return X

def compute_size_factor_lognorm(adata, layer, id2cov):
    """Compute the mean and variance of the log size factors.

    Args:
        X (torch.Tensor): Gene expression matrix.

    Returns:
        tuple: Mean and standard deviation of the log size factors.
    """
    # Each dictionary will contain one key for each covariate, associated to a torch value with log size factor 
    # mean/variance per category
    log_size_factors_mean, log_size_factors_sd = {}, {}
    
    for cov_name in id2cov:
        log_size_factors_mean_cov, log_size_factors_sd_cov = [], []
        # Iterate over the categories under the covariate
        for cov_cat in id2cov[cov_name]:
            adata_cov = adata[adata.obs[cov_name]==cov_cat]
            # Get log size factor of the gene expression of a certain 
            log_size_factors_cov = torch.log(torch.tensor(adata_cov.layers[layer].todense().sum(1)))
            mean, sd = log_size_factors_cov.mean(), log_size_factors_cov.std()
            # Append results 
            log_size_factors_mean_cov.append(mean)
            log_size_factors_sd_cov.append(sd)
        log_size_factors_mean[cov_name], log_size_factors_sd[cov_name] = torch.stack(log_size_factors_mean_cov), torch.stack(log_size_factors_sd_cov)
    return log_size_factors_mean, log_size_factors_sd
