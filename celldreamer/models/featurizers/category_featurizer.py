import torch 
import pandas as pd
from celldreamer.paths import EMBEDDING_DIR
import torch.nn.functional as F

class CategoricalFeaturizer(torch.nn.Module):
    def __init__(self, n_cat, one_hot_encode_features, device, embedding_dimensions=None):
        super().__init__()
        self.n_cat = n_cat
        self.device = device
        self.one_hot_encode_features = one_hot_encode_features
        if not self.one_hot_encode_features:
            self.embeddings = torch.nn.Embedding(n_cat, embedding_dimensions).to(self.device)
    
    def forward(self, obs):
        """Extract features 

        Args:
            obs (torch.Tensor): the batch of observations 

        Returns:
            torch.Tensor: extracted embeddings 
        """
        obs = obs.to(self.device)
        if self.one_hot_encode_features: 
            return F.one_hot(obs, num_classes=self.n_cat).float()
        else:
            return self.embeddings(obs)
        