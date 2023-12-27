import torch 
import pandas as pd
import torch.nn.functional as F
from celldreamer.models.base.utils import unsqueeze_right

class CellDecoder(torch.nn.Module):
    def __init__(self, encoder_type):
        super().__init__()
        self.encoder_type = encoder_type 
    
    def forward(self, X, size_factor):
        if self.encoder_type == "proportions":
            X = X * size_factor
        elif self.encoder_type == "log_gexp":
            X = torch.exp(X) - 1 
        elif self.encoder_type == "log_gexp_scaled":
            X = (torch.exp(X) - 1) * size_factor 
        else:
            raise NotImplementedError
        return X 
    