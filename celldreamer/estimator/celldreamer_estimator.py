from os.path import join
from typing import Dict, List
from pathlib import Path
from celldreamer.paths import TRAINING_FOLDER
from celldreamer.eval.metrics_collector import MetricsCollector
from celldreamer.data.utils import Args

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
import torch.nn.functional as F

# Restore once the dataset is ready
# from cellnet.datamodules import MerlinDataModule

from celldreamer.models.base.autoencoder import MLP_AutoEncoder
from celldreamer.data.pert_loader import PertDataset
from celldreamer.models.featurizers.drug_featurizer import DrugsFeaturizer
from celldreamer.models.featurizers.category_featurizer import CategoricalFeaturizer

from celldreamer.models.diffusion.denoising_model import MLPTimeStep
from celldreamer.models.diffusion.conditional_ddpm import ConditionalGaussianDDPM


class CellDreamerEstimator:
    def __init__(self, args):
        self.args = args
        
        # Create the training directory 
        if self.args.train:
            self.training_dir = TRAINING_FOLDER / self.args.experiment_name
            self.training_dir.mkdir(parents=True, exist_ok=True)
            print("Create the training folders...")
            if self.args.use_latent_repr:
                self.args["trainer_autoencoder_kwargs"]["default_root_dir"] = self.training_dir / "autoencoder"
            self.args["trainer_generative_kwargs"]["default_root_dir"] = self.training_dir / "generative"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Initialize data module...")
        self.init_datamodule()  # Initialize the data module 
        self.get_fixed_rna_model_params()  # Initialize the data derived model params 
        self.init_trainer()
        print("Initialize feature embeddings...")
        self.init_feature_embeddings()  # Initialize the feature embeddings 
        
        train_metric_collector = MetricsCollector(
                                    self.datamodule.train_dataloader, 
                                    self.args.task, 
                                    self.feature_embeddings)
        
        valid_metric_collector = MetricsCollector(
                                    self.datamodule.valid_dataloader, 
                                    self.args.task,
                                    self.feature_embeddings)
        
        self.metric_collector = {"train": train_metric_collector, 
                                    "valid": valid_metric_collector}
        
        print("Initialize model...")
        self.init_model()  # Initialize
    
    
    def init_datamodule(self):
        """
        Initialization of the data module
        """
        assert self.args.task in ["cell_generation", "perturbation_modelling"], f"The task {self.args.task} is not implemented"
        
        # Initialize dataloaders for the different tasks 
        if self.args.task == "cell_generation":
            self.dataset = None 
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
                            use_drugs=self.args.use_drugs)
            
            # The keys of the data module can be called via datamodule.key (aligned with the ones of scRNAseq)
            self.datamodule = Args({"train_dataloader": torch.utils.data.DataLoader(
                                                        self.dataset.subset("train", "all"),
                                                        batch_size=self.args.batch_size,
                                                        shuffle=True,
                                                        num_workers=8
                                                    ),
                                    "valid_dataloader": torch.utils.data.DataLoader(
                                                        self.dataset.subset("test", "all"),
                                                        batch_size=self.args.batch_size,
                                                        shuffle=False,
                                                        num_workers=8
                                                    ),
                                    "test_dataloader": torch.utils.data.DataLoader(
                                                        self.dataset.subset("ood", "all"),
                                                        batch_size=self.args.batch_size,
                                                        shuffle=False,
                                                        num_workers=8
                                    )})
    
    
    def get_fixed_rna_model_params(self):
        """Set the model parameters extracted from the data loader object
        """
        if self.args.task == "perturbation_modelling":
            if self.args.use_latent_repr: 
                self.args.autoencoder_kwargs["in_dim"] = self.dataset.genes.shape[1]
                self.args.denoising_module_kwargs["in_dim"] = self.args.autoencoder_kwargs["hidden_dim_encoder"][-1]
            else:
                self.args.denoising_module_kwargs["in_dim"] = self.dataset.genes.shape[1]
            self.args.generative_model_kwargs["n_covariates"] = len(self.dataset.covariate_names)
        else:
            self.args.denoising_module_kwargs["in_dim"] = len(pd.read_parquet(join(self.args.data_path, 'var.parquet')))
            self.args.generative_model_kwargs["n_covariates"] = len(self.args.categories)        
            
    def init_feature_embeddings(self):
        """
        Initialize feature embeddings either for drugs or covariates 
        """
        assert self.args.task in ["cell_generation", "perturbation_modelling"], f"The task {self.args.task} is not implemented"
        
        self.feature_embeddings = {}  # Contains the embedding class of multiple feature types
        num_classes = {}
        
        if self.args.task == "perturbation_modelling":
            if self.args.use_drugs:
                self.feature_embeddings["y_drug"] = DrugsFeaturizer(self.args,
                                                    self.dataset.canon_smiles_unique_sorted,
                                                    self.device)
                
                num_classes["y_drug"] = self.feature_embeddings["y_drug"].features.embedding_dim
            for cov, cov_names in self.dataset.covariate_names_unique.items():
                self.feature_embeddings["y_"+cov] = CategoricalFeaturizer(len(cov_names), 
                                                                        self.args.one_hot_encode_features, 
                                                                        self.device, 
                                                                        embedding_dimensions=self.args.embedding_dimensions)
                if self.args.one_hot_encode_features:
                    num_classes["y_"+cov] = len(cov_names)
                else:
                    num_classes["y_"+cov] = self.args.embedding_dimensions
                    
        else:
            metadata_path = Path(self.args.metadata_path) / "categorical_lookup"
            for cat in self.args.categories:
                n_cat = len(pd.read_parquet(metadata_path / f"{cat}.parquet"))
                self.feature_embeddings["y_"+cat] = CategoricalFeaturizer(n_cat, 
                                                                     self.args.one_hot_encode_features, 
                                                                     self.device, 
                                                                     embedding_dimensions=self.args.embedding_dimensions)
                if self.args.one_hot_encode_features:
                    num_classes["y_"+cov] = n_cat
                else:
                    num_classes["y_"+cov] = self.args.embedding_dimensions
                            
        self.args.denoising_module_kwargs["num_classes"] = num_classes


    def init_model(self):
        """Initialize the (optional) autoencoder and generative model 
        """
        if self.args.use_latent_repr:
            self.autoencoder = MLP_AutoEncoder(**self.args.autoencoder_kwargs)
        else:
            self.autoencoder = None 
        
        if self.args.generative_model == 'diffusion':
            if self.args.denoising_model == 'mlp':
                denoising_model = MLPTimeStep(**self.args.denoising_module_kwargs)
                self.generative_model = ConditionalGaussianDDPM(
                    denoising_model=denoising_model,
                    autoencoder_model=self.autoencoder,
                    feature_embeddings=self.feature_embeddings,
                    task=self.args.task,
                    metric_collector=self.metric_collector,  
                    **self.args.generative_model_kwargs  # model_kwargs should contain the rest of the arguments
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
    def init_trainer(self):
        if self.args.train_autoencoder:
            self.trainer_autoencoder = pl.Trainer(**self.args.trainer_autoencoder_kwargs, 
                                                  logger=CSVLogger(self.args["trainer_autoencoder_kwargs"]["default_root_dir"],
                                                    name=self.args.experiment_name))
            
        self.trainer_generative = pl.Trainer(**self.args.trainer_generative_kwargs, 
                                                  logger=CSVLogger(self.args["trainer_generative_kwargs"]["default_root_dir"],
                                                    name=self.args.experiment_name))

    def train(self):
        if self.args.use_latent_repr and self.args.train_autoencoder:
            # Fit autoencoder model 
            self.trainer_autoencoder.fit(
                self.autoencoder,
                train_dataloaders=self.datamodule.train_dataloader,
                val_dataloaders=self.datamodule.valid_dataloader,
                ckpt_path=None if not self.args.pretrained_autoencoder else self.args.checkpoint_autoencoder
            )
        
        self.trainer_generative.fit(
                self.generative_model,
                train_dataloaders=self.datamodule.train_dataloader,
                val_dataloaders=self.datamodule.valid_dataloader,
                ckpt_path=None if not self.args.pretrained_generative else self.args.checkpoint_generative
                )
        
    def validate(self, ckpt_path: str = None):
        self._check_is_initialized()
        return self.trainer_generative.validate(self.generative_model, 
                                                dataloaders=self.datamodule.valid_dataloader, 
                                                ckpt_path=None if not self.args.pretrained_generative else self.args.checkpoint_generative)
    