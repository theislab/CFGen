import torch 

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
    elif encoder_type in ["log_gexp", "learnt_encoder", "learnt_autoencoder"]:
        X = torch.log1p(X)
    elif encoder_type == "log_gexp_scaled":
        X = torch.log1p(X / size_factor)
    else:
        raise NotImplementedError    
    return X
