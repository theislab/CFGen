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

from celldreamer.models.diffusion.denoising_model import MLPTimeStep
from celldreamer.models.diffusion.classifier_free_ddpm import ConditionalGaussianDDPM

class CellDreamerEstimator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def init_datamodule(self):
        """
        Initialization of the data module
        """
        assert self.args.task in ["cell_generation", "perturbation_modelling"], f"The task {self.args.task} is not implemented"
        
        # Initialize dataloaders for the different tasks 
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
            
            # The keys of the data module can be called via datamodule.key (aligned with the ones of scRNAseq)
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
        
        self.feature_embeddings = {}  # Contains the embedding class of multiple feature types
        num_classes = {}
        
        if self.args.task == "perturbation_modelling":
            # ComPert will use the provided embedding, which is frozen during training
            self.feature_embeddings["drug"] = DrugsFeaturizer(self.args,
                                                   self.dataset.canon_smiles_unique_sorted,
                                                   self.device)
            
            for cov, cov_names in self.dataset.covariate_names_unique.items():
                self.feature_embeddings[cov] = CategoricalFeaturizer(len(cov_names), 
                                                                        self.args.one_hot_encode_features, 
                                                                        self.device, 
                                                                        embedding_dimensions=self.args.embedding_dimensions)
                num_classes[cov] = len(cov_names)
            
        else:
            metadata_path = Path(self.args.metadata_path) / "categorical_lookup"
            for cat in self.args.categories:
                n_cat = len(pd.read_parquet(metadata_path / f"{cat}.parquet"))
                self.feature_embeddings[cat] = CategoricalFeaturizer(n_cat, 
                                                                     self.args.one_hot_encode_features, 
                                                                     self.device, 
                                                                     embedding_dimensions=self.args.embedding_dimensions)
                num_classes[cov] = n_cat
        
        self.args.denoising_module_kwargs["num_classes"] = num_classes
    
    def get_fixed_rna_model_params(self):
        if self.args.task == "perturbation_modelling":
            self.args.denoising_module_kwargs["in_dim"] = self.dataset.genes.shape[1]
            self.args.model_kwargs["n_covariates"] = len(self.dataset.covariate_names)
        else:
            self.args.denoising_module_kwargs["in_dim"] = len(pd.read_parquet(join(self.args.data_path, 'var.parquet')))
            self.args.model_kwargs["n_covariates"] = len(self.args.categories)
    
    def init_model(self):
        if self.args.generative_model == 'diffusion':
            if self.args.denoising_model == 'mlp':
                denoising_model = MLPTimeStep(**self.args.denoising_module_kwargs)
                self.model = ConditionalGaussianDDPM(
                    denoising_model=denoising_model,
                    **self.args.model_kwargs  # model_kwargs should contain the rest of the arguments
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
    def init_trainer(self):
        self.trainer = pl.Trainer(**self.args.trainer_kwargs)

    def _check_is_initialized(self):
        pass

    def find_lr(self):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass

    def predict(self) -> np.ndarray:
        pass
    
    def generate(self):
        pass
    