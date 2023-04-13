import torch 
import pandas as pd
from celldreamer.paths import EMBEDDING_DIR
from celldreamer.models.base.utils import MLP 

class DrugsFeaturizer(torch.nn.Module):
    def __init__(self, 
                 args, 
                 smiles,
                 device):
        """Wrapper around drug pre-trained features.
        Args:
            args (Args): dictionary with training arguments
            smiles (list): list of smiles in the order they should appear in the feature matrix
        """
        super().__init__()

        self.freeze_embeddings = args.freeze_embeddings
        self.feature_type = args.feature_type
        self.smiles = smiles
        self.device = device
        self.features = self._load_features()
        
        # Initialize doser 
        self.doser = MLP(in_channels = self.features.embedding_dim + 1,
                          hidden_channels = [args["doser_width"]] * args["doser_depth"]
                                            + [1],
                          norm_layer=None,
                          activation_layer=torch.nn.ReLU, 
                          inplace=True, 
                          bias=True, 
                          dropout=0.0)
    
    def forward(self, batch):
        """Given the SMILE IDs of a batch, collect the pre-trained features
        Args:
            batch_idx (Union[float, int, list, torch.Tensor]): The indices to extract from the matrix of pre-trained embeddings 
        Returns:
            torch.Tensor: features of the extracted batch ids 
        """
        batch_idx, dose = batch[0], batch[1]
        if type(batch_idx) != torch.Tensor:
            batch_idx = torch.tensor(batch_idx).long()
            
        if type(dose) != torch.Tensor:
            dose = torch.tensor(dose)
            
        batch_idx = batch_idx.to(self.device)
        drug_features = self.features(batch_idx)
        scaled_dosages = self.doser(torch.cat([drug_features, dose], dim=1))
        return drug_features @ scaled_dosages 

    def _load_features(self):
        """Load features from a pre-defined model.
        Returns:
            torch.Embedding: embedding matrix of pre-trained features
        """
        embeddings = pd.read_csv(EMBEDDING_DIR / self.feature_type / "data" / "embeddings.csv", index_col=0).loc[self.smiles].values
        emb = torch.tensor(embeddings, 
                           dtype=torch.float32, 
                           device=self.device)
        assert emb.shape[0] == len(self.smiles)
        return torch.nn.Embedding.from_pretrained(emb, freeze=self.freeze_embeddings)
    
    
if __name__ == '__main__':
    import numpy as np
    from celldreamer.data.utils import Args
    from celldreamer.paths import PERT_DATA_DIR
    from pathlib import Path 

    smiles = pd.read_csv(Path(PERT_DATA_DIR) / 'sciplex' / 'sciplex.smiles').SMILES
    smiles = np.sort(smiles)
    args = Args({'freeze_embeddings': True, 
                 'feature_type': 'None'})
    for model in ['ECFP', 'grover', 'MPNN']:
        args['feature_type'] = model
        f = DrugsFeaturizer(args, smiles, device='cpu')
         