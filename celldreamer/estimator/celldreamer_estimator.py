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
from celldreamer.models.featurizers.category_featurizer import CategoricalFeaturizer
from celldreamer.models.fm.denoising_model import SimpleMLPTimeStep, MLPTimeStep
from celldreamer.models.fm.fm import FM

os.environ["WANDB__SERVICE_WAIT"] = "300"
torch.autograd.set_detect_anomaly(True)

class CellDreamerEstimator:
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
        self.plotting_dir = self.training_dir / "plots"
        print("Create the training folders...")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.plotting_dir.mkdir(exist_ok=True)

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
        self.dataset = RNAseqLoader(data_path=self.data_path,
                                    layer_key=self.args.dataset.layer_key,
                                    covariate_keys=self.args.dataset.covariate_keys,
                                    subsample_frac=self.args.dataset.subsample_frac, 
                                    encoder_type=self.args.dataset.encoder_type,
                                    target_max=self.args.dataset.target_max, 
                                    target_min=self.args.dataset.target_min)

        # Initialize the data loaders 
        self.train_data, self.test_data, self.valid_data = random_split(self.dataset,
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
        
        self.test_dataloader = torch.utils.data.DataLoader(self.test_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=False,
                                                            num_workers=4, 
                                                            drop_last=True)
    
    def get_fixed_rna_model_params(self):
        """Set the model parameters extracted from the data loader object
        """
        self.gene_dim = self.dataset.X.shape[1] 
        self.in_dim = self.gene_dim if self.args.dataset.encoder_type!="learnt_autoencoder" else self.args.generative_model.x0_from_x_kwargs["dims"][-1]

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
            
    def init_feature_embeddings(self):
        """
        Initialize feature embeddings either for drugs or covariates 
        """
        # Contains the embedding class of multiple feature types
        self.feature_embeddings = {}  
        self.num_classes = {}
                
        for cov, cov_names in self.dataset.id2cov.items():
            self.feature_embeddings[cov] = CategoricalFeaturizer(len(cov_names), 
                                                                    self.args.dataset.one_hot_encode_features, 
                                                                    self.device, 
                                                                    embedding_dimensions=self.args.dataset.cov_embedding_dimensions)
            if self.args.dataset.one_hot_encode_features:
                self.num_classes[cov] = len(cov_names)
            else:
                self.num_classes[cov] = self.args.dataset.cov_embedding_dimensions

    def init_model(self):
        """Initialize the (optional) autoencoder and generative model 
        """
        conditioning_cov = self.args.dataset.conditioning_covariate
        if self.args.denoising_module.denoising_net == "simple_mlp":
            denoising_model = SimpleMLPTimeStep(in_dim=self.in_dim, 
                                                out_dim=self.args.denoising_module.out_dim,
                                                w=self.args.denoising_module.w,
                                                model_type=self.args.denoising_module.model_type, 
                                                conditional=self.args.denoising_module.conditional, 
                                                n_cond=self.num_classes[conditioning_cov])
        else:
            denoising_model = MLPTimeStep(in_dim=self.in_dim, 
                                            hidden_dim=self.args.denoising_module.hidden_dim,
                                            dropout_prob=self.args.denoising_module.dropout_prob,
                                            n_blocks=self.args.denoising_module.n_blocks, 
                                            model_type=self.args.denoising_module.model_type, 
                                            embed_time=self.args.denoising_module.embed_time,
                                            size_factor_min=self.dataset.min_size_factor, 
                                            size_factor_max=self.dataset.max_size_factor,
                                            embed_size_factor=self.args.denoising_module.embed_size_factor, 
                                            embedding_dim=self.args.denoising_module.embedding_dim,
                                            normalization=self.args.denoising_module.normalization).to(self.device)
        
        print("Denoising model")
        print(denoising_model)
        
        size_factor_statistics = {"mean": self.dataset.log_size_factor_mu, 
                                  "sd": self.dataset.log_size_factor_sd}
        
        scaler = self.dataset.get_scaler()
        
        self.generative_model = FM(
            denoising_model=denoising_model,
            feature_embeddings=self.feature_embeddings,
            plotting_folder=self.plotting_dir,
            in_dim=self.gene_dim,
            size_factor_statistics=size_factor_statistics,
            scaler=scaler,
            encoder_type=self.args.dataset.encoder_type,
            conditioning_covariate=conditioning_cov,
            model_type=denoising_model.model_type, 
            **self.args.generative_model  # model_kwargs should contain the rest of the arguments
        )

    def train(self):
        """
        Train the generative model using the provided trainer.
        """
        self.trainer_generative.fit(
            self.generative_model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.valid_dataloader)
    
    def test(self):
        """
        Test the generative model.
        """
        self.trainer_generative.test(
            self.generative_model,
            dataloaders=self.test_dataloader)
    