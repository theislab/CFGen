import os
from pathlib import Path
import uuid
import torch
from torch.utils.data import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from celldreamer.paths import TRAINING_FOLDER
from celldreamer.data.scrnaseq_loader import RNAseqLoader
from celldreamer.models.base.encoder_model import EncoderModel
 
# Some general settings for the run
os.environ["WANDB__SERVICE_WAIT"] = "300"
torch.autograd.set_detect_anomaly(True)

class EncoderEstimator:
    """Class for training and using the CellDreamer model."""
    def __init__(self, args):
        """
        Initialize the CellDreamerEstimator.

        Args:
            args (Args): Configuration hyperparameters for the model.
        """
        # args is a dictionary containing the configuration hyperparameters 
        self.args = args
        
        # date and time to name run 
        self.unique_id = str(uuid.uuid4())
        
        # dataset path as Path object 
        self.data_path = Path(self.args.dataset.dataset_path)
        
        # Initialize training directory         
        self.training_dir = TRAINING_FOLDER / self.args.logger.project / self.unique_id
        print("Create the training folders...")
        self.training_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Initialize data module...")
        self.init_datamodule()  # Initialize the data module  
        self.get_fixed_rna_model_params()  # Initialize the data derived model params 
        self.init_trainer()
        
        print("Initialize model...")
        self.init_model()  # Initialize

    def init_datamodule(self):
        """
        Initialization of the data module
        """        
        # Initialize dataloaders for the different tasks 
        self.dataset = RNAseqLoader(data_path=self.data_path,
                                    layer_key=self.args.dataset.layer_key,
                                    covariate_keys=self.args.dataset.covariate_keys,
                                    subsample_frac=self.args.dataset.subsample_frac, 
                                    encoder_type=self.args.dataset.encoder_type,
                                    target_max=self.args.dataset.target_max, 
                                    target_min=self.args.dataset.target_min,
                                    multimodal=self.args.dataset.multimodal, 
                                    is_binarized=self.args.dataset.is_binarized)
        
        # Number of categories
        if self.args.encoder.covariate_specific_theta:
            self.n_cat = len(self.dataset.id2cov[self.args.dataset.conditioning_covariate])
        else:
            self.n_cat = None

        # Initialize the data loaders 
        self.train_data, self.valid_data = random_split(self.dataset,
                                                        lengths=self.args.dataset.split_rates)   
        
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=True,
                                                            num_workers=4, 
                                                            drop_last=True)
        
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=False,
                                                            num_workers=4, 
                                                            drop_last=True)
    
    def get_fixed_rna_model_params(self):
        """Set the model parameters extracted from the data loader object
        """
        if not self.dataset.multimodal:
            self.gene_dim = self.dataset.X.shape[1] 
        else:
            self.gene_dim = {mod: self.dataset.X[mod].shape[1] for mod in self.dataset.X}

    def init_trainer(self):
        """
        Initialize Trainer
        """
        # Callbacks for saving checkpoints 
        checkpoint_callback = ModelCheckpoint(dirpath=self.training_dir / "checkpoints", 
                                                **self.args.checkpoints)
        callbacks = [checkpoint_callback]
        
        # Early stopping checkpoints 
        if self.args.training_config.use_early_stopping:
            early_stopping_callbacks = EarlyStopping(**self.args.early_stopping)
            callbacks.append(early_stopping_callbacks)
        
        # Logger settings 
        self.logger = WandbLogger(save_dir=self.training_dir,
                                    name=self.unique_id, 
                                    **self.args.logger)
        
        self.trainer_generative = Trainer(callbacks=callbacks, 
                                          default_root_dir=self.training_dir, 
                                          logger=self.logger,
                                          **self.args.trainer)

    def init_model(self):
        """Initialize encoder model 
        """
        scaler = self.dataset.get_scaler()
        self.encoder_model = EncoderModel(in_dim=self.gene_dim,
                                          scaler=scaler, 
                                          n_cat=self.n_cat,
                                          conditioning_covariate=self.args.dataset.conditioning_covariate, 
                                          encoder_type=self.args.dataset.encoder_type,
                                          **self.args.encoder)
        print("Encoder architecture", self.encoder_model)

    def train(self):
        """
        Train the generative model using the provided trainer.
        """
        self.trainer_generative.fit(
            self.encoder_model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.valid_dataloader)
    
    def test(self):
        """
        Test the generative model.
        """
        self.trainer_generative.test(
            self.encoder_model,
            dataloaders=self.test_dataloader)
    