from os.path import join
from typing import Dict, List
from pathlib import Path

from celldreamer.data.utils import Args

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from cellnet.datamodules import MerlinDataModule
from celldreamer.data.pert_loader import PertDataset
from celldreamer.models.featurizers.drug_featurizer import DrugsFeaturizer
from celldreamer.models.featurizers.category_featurizer import CategoricalFeaturizer

        
class CellDreamerEstimator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def init_datamodule(self):
        """
        Initialization of the data module
        """
        assert self.args.task in ["cell_generation", "perturbation_modelling"], f"The task {self.args.task} is not implemented"
        
        # initialize dataloaders for the different tasks 
        if self.args.task == "cell_generation":
            self.datamodule = MerlinDataModule(
                self.args.data_path,
                columns=self.args.categories,
                batch_size=self.args.batch_size,
                drop_last=self.args.drop_last)
            
        else:
            self.dataset = PertDataset(
                                    data=self.args.data_path,
                                    perturbation_key=self.args.perturbation_key,
                                    dose_key=self.args.dose_key,
                                    covariate_keys=self.args.covariate_keys,
                                    smiles_key=self.args.smile_keys,
                                    degs_key=self.args.degs_key,
                                    pert_category=self.args.pert_category,
                                    split_key=self.args.split_key,
                                    use_drugs_idx=True)
            
            # The keys of the data module can be called via datamodule.key
            self.datamodule = Args({"train_dataloader": torch.utils.data.DataLoader(
                                                        self.dataset.subset("train", "all"),
                                                        batch_size=self.args.batch_size,
                                                        shuffle=True,
                                                    ),
                                    "valid_dataloader": torch.utils.data.DataLoader(
                                                        self.dataset.subset("test", "all"),
                                                        batch_size=self.args.batch_size,
                                                        shuffle=True,
                                                    ),
                                    "test_dataloader": torch.utils.data.DataLoader(
                                                        self.dataset.subset("ood", "all"),
                                                        batch_size=self.args.batch_size,
                                                        shuffle=True,
                                    )})
            
            

    def init_feature_embeddings(self):
        """
        Initialize feature embeddings either for drugs or covariates 
        """
        assert self.args.task in ["cell_generation", "perturbation_modelling"], f"The task {self.args.task} is not implemented"
        
        if self.args.task == "perturbation_modelling":
            # ComPert will use the provided embedding, which is frozen during training
            self.feature_embeddings = DrugsFeaturizer(self.args,
                                                   self.dataset.canon_smiles_unique_sorted,
                                                   self.device)
        else:
            self.feature_embeddings = {}
            metadata_path = Path(self.args.data_path) / "categorical_lookup"
            for cat in self.args.categories:
                n_cat = len(pd.read_parquet(metadata_path / f"{cat}.parquet"))
                self.feature_embeddings[cat] = CategoricalFeaturizer(n_cat, 
                                                                     self.args.one_hot_encode_features, 
                                                                     self.device, 
                                                                     embedding_dimensions=self.args.embedding_dimensions)
    
    def init_model(self):
        pass
    
    def init_(self):
        pass
