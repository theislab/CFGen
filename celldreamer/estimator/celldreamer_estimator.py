from pathlib import Path
from celldreamer.data.utils import Args

import torch
from torch.utils.data import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Restore once the dataset is ready

from celldreamer.paths import TRAINING_FOLDER
from celldreamer.models.base.autoencoder import AE
from celldreamer.data.scrnaseq_loader import RNAseqLoader
from celldreamer.models.featurizers.category_featurizer import CategoricalFeaturizer

from celldreamer.models.vdm.denoising_model import MLPTimeStep
from celldreamer.models.vdm.vdm import VDM

class CellDreamerEstimator:
    def __init__(self, args):
        # args is a dictionary containing the parameters 
        self.args = args
        
        # dataset path as Path object 
        self.data_path = Path(self.args.dataset_path)
        
        # Initialize training directory         
        if self.args.train:
            self.training_dir = TRAINING_FOLDER / self.args.experiment_name
            print("Create the training folders...")
            self.training_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Initialize data module...")
        self.init_datamodule()  # Initialize the data module  
        self.get_fixed_rna_model_params()  # Initialize the data derived model params 
        self.init_trainer()
        
        print("Initialize feature embeddings...")
        self.init_feature_embeddings()  # Initialize the feature embeddings 
        
        print("Initialize model...")
        self.init_model()  # Initialize
    
    def init_datamodule(self):
        """
        Initialization of the data module
        """        
        # Initialize dataloaders for the different tasks 
        if self.args.task == "cell_generation":
            self.dataset = RNAseqLoader(data_path=self.data_path,
                                covariate_keys=self.args.covariate_keys,
                                subsample_frac=self.args.subsample_frac, 
                                use_pca=self.args.use_pca, 
                                n_dimensions=self.args.n_dimensions, 
                                layer=self.args.layer)
            
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
    
    def get_fixed_rna_model_params(self):
        """Set the model parameters extracted from the data loader object
        """
        self.args.denoising_module_kwargs["in_dim"] = self.dataset.genes.shape[1]  # perform diffusion in gene dimension 

    def init_trainer(self):
        """
        Initialize Trainer
        """
        # Callbacks for saving checkpoints 
        checkpoint_callback = ModelCheckpoint(dirpath=self.training_dir / "checkpoints", 
                                                        **self.args.checkpoint_kwargs)
        
        # Early stopping checkpoints 
        early_stopping_callbacks = EarlyStopping(**self.args.early_stopping_kwargs)
        
        # Logger settings 
        logger = WandbLogger(save_dir=self.training_dir, 
                                    **self.args.logger_kwargs)
            
        self.trainer_generative = Trainer(callbacks=[checkpoint_callback, early_stopping_callbacks], 
                                            default_root_dir=self.training_dir, 
                                            logger=logger, 
                                            **self.args.trainer_kwargs)
            
    def init_feature_embeddings(self):
        """
        Initialize feature embeddings either for drugs or covariates 
        """
        # Contains the embedding class of multiple feature types
        self.feature_embeddings = {}  
        num_classes = {}
                
        for cov, cov_names in self.dataset.covariate_names_unique.items():
            self.feature_embeddings["y_"+cov] = CategoricalFeaturizer(len(cov_names), 
                                                                        self.args.one_hot_encode_features, 
                                                                        self.device, 
                                                                        embedding_dimensions=self.args.cov_embedding_dimensions)
            if self.args.one_hot_encode_features:
                num_classes["y_"+cov] = len(cov_names)
            else:
                num_classes["y_"+cov] = self.args.cov_embedding_dimensions

    def init_model(self):
        """Initialize the (optional) autoencoder and generative model 
        """
        if self.args.generative_model == 'diffusion':
            if self.args.denoising_model == 'mlp':
                denoising_model = MLPTimeStep(**self.args.denoising_module_kwargs).to(self.device)
                self.generative_model = VDM(
                    denoising_model=denoising_model,
                    feature_embeddings=self.feature_embeddings,
                    one_hot_encode_features=self.args.one_hot_encode_features,
                    **self.args.generative_model_kwargs  # model_kwargs should contain the rest of the arguments
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def train(self):
        self.trainer_generative.fit(
                self.generative_model,
                train_dataloaders=self.datamodule.train_dataloader,
                val_dataloaders=self.datamodule.valid_dataloader,
                ckpt_path=None if not self.args.pretrained_generative else self.args.checkpoint_generative
                )
        