import torch 
import pandas as pd
import torch.nn.functional as F
from celldreamer.models.base.utils import unsqueeze_right

class CellEncoder(torch.nn.Module):
    def __init__(self, encoder_type):
        super().__init__()
        self.encoder_type = encoder_type 
    
    def encode(self, x):
        l = x.sum(1, keepdim=True)  #TODO: check if right param
        # Proportion 
        x = x/l
        return x
    
# TO TRY:
# log1p + rescale, CDF approach dataset dependent, see which one converges faster