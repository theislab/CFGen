import torch 
import pandas as pd
import torch.nn.functional as F
from celldreamer.models.base.utils import unsqueeze_right

class CellEncoder(torch.nn.Module):
    def __init__(self, encoder_type):
        super().__init__()
        self.encoder_type = encoder_type 
    
    def encode(self, x, l):
        # Scale down 
        if l.ndim != x.ndim:
            l = unsqueeze_right(l, x.ndim-l.ndim) 
        # Propportion 
        x = x/l
        
        