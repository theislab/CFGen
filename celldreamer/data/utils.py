import torch 

class Scaler:
    def __init__(self, target_min: int=-1, target_max: int=1):
        """Initialization function.

        Args:
            target_min (int): Minimum value for scaling.
            target_max (int): Maximum value for scaling.
        """
        self.target_min = target_min
        self.target_max = target_max
    
    def fit(self, X):
        """Fit the scaler to the data.

        Args:
            X (torch.Tensor): Input data for fitting the scaler.
        """
        self.data_min, self.data_max = torch.min(X), torch.max(X)
        
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
            X_scaled = torch.clip(X_scaled, min=new_min)
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

def compute_size_factor_lognorm(X):
    """Compute the mean and variance of the log size factors.

    Args:
        X (torch.Tensor): Gene expression matrix.

    Returns:
        tuple: Mean and standard deviation of the log size factors.
    """
    log_size_factors = torch.log(X.sum(1))
    log_size_factors_mean = log_size_factors.mean()
    log_size_factors_sd = log_size_factors.std()
    return log_size_factors_mean, log_size_factors_sd

    