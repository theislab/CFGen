import os
from pathlib import Path
from celldreamer.eval.metrics_collector import MetricsCollector
from celldreamer.data.utils import Args

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from cellnet.datamodules import MerlinDataModule

# Restore once the dataset is ready

from celldreamer.paths import ROOT, TRAINING_FOLDER
from celldreamer.models.base.autoencoder import MLP_AutoEncoder
from celldreamer.data.pert_loader import PertDataset
from celldreamer.data.scrnaseq_loader import RNAseqLoader
from celldreamer.models.featurizers.drug_featurizer import DrugsFeaturizer
from celldreamer.models.featurizers.category_featurizer import CategoricalFeaturizer
from celldreamer.data.datamodules import HLCADataModule, ShapeColorDataModule
from celldreamer.models.diffusion.denoising_model import MLPTimeStep, UNetTimeStepClassSetConditioned
from celldreamer.models.diffusion.conditional_ddpm import ConditionalGaussianDDPM


class CellDreamerEstimator:
    def __init__(self, args):
        # Move to celldreamer directory
        self.autoencoder = None
        self.args = args
        
        # Read dataset
        self.data_path = Path(self.args.dataset_path) if self.args.dataset_path is not None else None
        
        # Initialize training directory         
        if self.args.train:
            self.training_dir = TRAINING_FOLDER / self.args.experiment_name
            print("Create the training folders...")
            self.training_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Initialize data module...")
        self.init_datamodule()  # Initialize the data module  
        self.get_fixed_model_params()  # Initialize the data derived model params 
        self.init_trainer()
        print("Initialize feature embeddings...")
        self.init_feature_embeddings()  # Initialize the feature embeddings 
        
        # Metric collection object 
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
        assert self.args.task in ["cell_generation", "perturbation_modelling", "toy_generation"], f"The task {self.args.task} is not implemented"
        
        # Initialize dataloaders for the different tasks 
        if self.args.task == "cell_generation":
            if self.args.atlas == "pbmc":
                self.dataset = RNAseqLoader(data_path=self.data_path,
                                    covariate_keys=self.args.covariate_keys,
                                    subsample_frac=self.args.subsample_frac, 
                                    use_pca=self.args.use_pca)

                train_data, test_data, valid_data = random_split(self.dataset, lengths=self.args.split_rates)
                self.datamodule = Args({"train_dataloader": torch.utils.data.DataLoader(
                                                            train_data,
                                                            batch_size=self.args.batch_size,
                                                            shuffle=True,
                                                            num_workers=8
                                                        ),
                                        "valid_dataloader": torch.utils.data.DataLoader(
                                                            valid_data,
                                                            batch_size=self.args.batch_size,
                                                            shuffle=False,
                                                            num_workers=8
                                                        ),
                                        "test_dataloader": torch.utils.data.DataLoader(
                                                            test_data,
                                                            batch_size=self.args.batch_size,
                                                            shuffle=False,
                                                            num_workers=8
                                        )})            
            elif self.args.atlas == "hlca":
                self.datamodule = HLCADataModule(
                    path=self.data_path,
                    batch_size=self.args.batch_size)
            elif self.args.atlas == "cellxgene":
                self.datamodule = MerlinDataModule(
                    self.data_path,
                    columns=self.args.categories,
                    batch_size=self.args.batch_size)
            else:
                raise NotImplementedError("The atlas {} is not implemented".format(self.args.atlas))
                
        elif self.args.task == "toy_generation":
            self.datamodule = ShapeColorDataModule(
                batch_size=self.args.batch_size,
                num_samples=self.args.num_samples,
            )
        
        elif self.args.task == "perturbation_modelling":
            self.dataset = PertDataset(
                            data_path=self.data_path,
                            perturbation_key=self.args.perturbation_key,
                            dose_key=self.args.dose_key,
                            covariate_keys=self.args.covariate_keys,
                            smiles_key=self.args.smile_keys,
                            degs_key=self.args.degs_key,
                            pert_category=self.args.pert_category,
                            split_key=self.args.split_key,
                            use_drugs=self.args.use_drugs, 
                            subsample_frac=self.args.subsample_frac, 
                            standardize=self.args.standardize)
            
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
        else:
            raise NotImplementedError("The task {} is not implemented".format(self.args.task))
    
    
    def get_fixed_model_params(self):
        """Set the model parameters extracted from the data loader object
        """
        """Set the model parameters extracted from the data loader object
        """
        if self.args.task == "perturbation_modelling":
            if self.args.use_latent_repr: 
                self.args.autoencoder_kwargs["in_dim"] = self.dataset.genes.shape[1]
                self.args.denoising_module_kwargs["in_dim"] = self.args.autoencoder_kwargs["hidden_dim_encoder"][-1]
            else:
                self.args.denoising_module_kwargs["in_dim"] = self.dataset.genes.shape[1]  # perform diffusion in gene dimension 
        elif self.args.task == "cell_generation":
            if self.args.atlas == "cellxgene":
                self.args.denoising_module_kwargs["in_dim"] = len(pd.read_parquet(join(self.args.data_path, 'var.parquet')))
                # self.args.generative_model_kwargs["n_covariates"] = len(self.args.categories)
            elif self.args.atlas == "hlca":
                self.args.denoising_module_kwargs["in_dim"] = 2000  # 2000 highly-variable genes
                # self.args.generative_model_kwargs["n_covariates"] = len(self.args.categories)
            elif self.args.atlas == "pbmc":
                raise NotImplementedError("The atlas {} is not implemented".format(self.args.atlas)) 
        elif self.args.task == "toy_generation":
            # self.args.generative_model_kwargs["n_covariates"] = len(self.args.categories)
            pass
        else:
            raise NotImplementedError(f"The task {self.args.task} is not implemented")

    
    def init_trainer(self):
        """
        Initialize 
        """
        # Callbacks for saving checkpoints 
        checkpoint_callback = ModelCheckpoint(dirpath=self.training_dir / "checkpoints", 
                                                        **self.args.checkpoint_kwargs)
        
        # Early stopping checkpoints 
        early_stopping_callbacks = EarlyStopping(**self.args.early_stopping_kwargs)
        
        # Logger settings 
        logger = WandbLogger(save_dir=self.training_dir, 
                                    **self.args.logger_kwargs) 
        
        if self.args.train_autoencoder:
            self.trainer_autoencoder = Trainer(callbacks=[checkpoint_callback, early_stopping_callbacks], 
                                                    default_root_dir=self.training_dir, 
                                                    logger=logger, 
                                                    **self.args.trainer_kwargs)
            
        self.trainer_generative = Trainer(callbacks=[checkpoint_callback, early_stopping_callbacks], 
                                            default_root_dir=self.training_dir, 
                                            logger=logger, 
                                            **self.args.trainer_kwargs)
        
    def init_feature_embeddings(self):
        """
        Initialize feature embeddings either for drugs or covariates 
        """
        assert self.args.task in ["cell_generation", "perturbation_modelling", "toy_generation"], f"The task {self.args.task} is not implemented"
        
        # Contains the embedding class of multiple feature types
        self.feature_embeddings = {}  
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
                                                                            embedding_dimensions=self.args.cov_embedding_dimensions)
                if self.args.one_hot_encode_features:
                    num_classes["y_"+cov] = len(cov_names)
                else:
                    num_classes["y_"+cov] = self.args.cov_embedding_dimensions
                    
        elif self.args.task == "cell_generation":
            if self.args.atlas == "pbmc":
                metadata_path = Path(self.args.metadata_path) / "categorical_lookup"
                for cat in self.args.categories:
                    # Categorical covariates are embedded using a one-hot encoding
                    n_cat = len(pd.read_parquet(metadata_path / f"{cat}.parquet"))
                    self.feature_embeddings["y_" + cat] = CategoricalFeaturizer(n_cat,
                                                                                self.args.one_hot_encode_features,
                                                                                self.device,
                                                                                embedding_dimensions=self.args.embedding_dimensions)
                    if self.args.one_hot_encode_features:
                        num_classes["y_" + cat] = n_cat
                    else:
                        num_classes["y_" + cat] = self.args.embedding_dimensions
            elif self.args.atlas == "hlca":
                for cat in self.args.categories:
                    n_cat = len(self.args.categories[cat])
                    self.feature_embeddings["y_" + cat] = CategoricalFeaturizer(n_cat,
                                                                                self.args.one_hot_encode_features,
                                                                                self.device,
                                                                                embedding_dimensions=self.args.embedding_dimensions)
                    if self.args.one_hot_encode_features:
                        num_classes["y_" + cat] = n_cat
                    else:
                        num_classes["y_" + cat] = self.args.embedding_dimensions

            else:
                raise NotImplementedError("The atlas {} is not implemented".format(self.args.atlas))

        elif self.args.task == "toy_generation":
            for cat in self.args.categories:
                n_cat = len(self.args.categories)
                self.feature_embeddings["y_" + cat] = CategoricalFeaturizer(len(self.args.categories[cat]),
                                                                            self.args.one_hot_encode_features,
                                                                            self.device,
                                                                            embedding_dimensions=self.args.embedding_dimensions)
                if self.args.one_hot_encode_features:
                    num_classes["y_" + cat] = n_cat
                else:
                    num_classes["y_" + cat] = self.args.embedding_dimensions
                    
        else:
            raise NotImplementedError(f"The task {self.args.task} is not implemented")
                            
        # Save number of classes                   
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
                denoising_model = MLPTimeStep(**self.args.denoising_module_kwargs).to(self.device)
                self.generative_model = ConditionalGaussianDDPM(
                    denoising_model=denoising_model,
                    autoencoder_model=self.autoencoder,
                    feature_embeddings=self.feature_embeddings,
                    task=self.args.task,
                    use_drugs=self.args.use_drugs if self.args.use_drugs else False,
                    one_hot_encode_features=self.args.one_hot_encode_features,
                    metric_collector=self.metric_collector,  
                    **self.args.generative_model_kwargs  # model_kwargs should contain the rest of the arguments
                )
                
            elif self.args.denoising_model == 'unet':
                denoising_model = UNetTimeStepClassSetConditioned(**self.args.denoising_module_kwargs)
                self.generative_model = ConditionalGaussianDDPM(
                    denoising_model=denoising_model,
                    autoencoder_model=self.autoencoder,
                    feature_embeddings=self.feature_embeddings,
                    task=self.args.task,
                    use_drugs=self.args.use_drugs if self.args.use_drugs else False,
                    one_hot_encode_features=self.args.one_hot_encode_features,
                    metric_collector=self.metric_collector,
                    **self.args.generative_model_kwargs  # model_kwargs should contain the rest of the arguments
                )
                
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            
    def _check_is_initialized(self):
        if not self.generative_model:
            raise RuntimeError('You need to call self.init_model before calling self.train')
        if not self.datamodule:
            raise RuntimeError('You need to call self.init_datamodule before calling self.train')
        if not self.trainer_generative:
            raise RuntimeError('You need to call self.init_trainer before calling self.train')

    def train(self):
        self._check_is_initialized()
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

    def test(self, ckpt_path: str = None):
        self._check_is_initialized()
        return self.trainer_generative.test(self.generative_model,
                                            dataloaders=self.datamodule.test_dataloader(),
                                            ckpt_path=None if not self.args.pretrained_generative else self.args.checkpoint_generative)
    