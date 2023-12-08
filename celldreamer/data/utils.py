import torch 

indx = lambda a, i: a[i] if a is not None else None

class Args(dict):
    """
    Wrapper around a dictiornary to make its keys callable as attributes
    """
    def __init__(self, *args, **kwargs):
        super(Args, self).__init__(*args, **kwargs)
        self.__dict__ = self

def compute_size_factor_lognorm(X):
    """Compute size factor variance and mean 

    Args:
        X (torch.tensor): gene expression matrix

    Returns:
        tuple: the mean and the variance of the log size factor  
    """
    log_size_factors = torch.log(X.sum(1))
    log_size_factors_mean = log_size_factors.mean()
    log_size_factors_sd = log_size_factors.std()
    return log_size_factors_mean, log_size_factors_sd
