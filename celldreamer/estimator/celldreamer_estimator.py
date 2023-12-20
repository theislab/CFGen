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
from celldreamer.models.vdm.denoising_model import SimpleMLPTimeStep
from celldreamer.models.vdm.vdm import VDM


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
                                    use_pca=self.args.dataset.use_pca, 
                                    encoder_type=self.args.dataset.encoder_type,
                                    target_max=self.args.dataset.target_max, 
                                    target_min=self.args.dataset.target_min)

        # Initialize the data loaders 
        self.train_data, self.test_data, self.valid_data = random_split(self.dataset, lengths=self.args.dataset.split_rates)   
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=True,
                                                            num_workers=4)
        
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=False,
                                                            num_workers=4)
        
        self.test_dataloader = torch.utils.data.DataLoader(self.test_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=False,
                                                            num_workers=4)
    
    def get_fixed_rna_model_params(self):
        """Set the model parameters extracted from the data loader object
        """
        self.in_dim = self.dataset.X.shape[1]  # perform diffusion in gene dimension 

    def init_trainer(self):
        """
        Initialize Trainer
        """
        # Callbacks for saving checkpoints 
        checkpoint_callback = ModelCheckpoint(dirpath=self.training_dir / "checkpoints", 
                                              **self.args.checkpoints)
        
        # Early stopping checkpoints 
        early_stopping_callbacks = EarlyStopping(**self.args.early_stopping)
        
        # Logger settings 
        logger = WandbLogger(save_dir=self.training_dir,
                             name=self.unique_id, 
                             **self.args.logger)
            
        self.trainer_generative = Trainer(callbacks=[checkpoint_callback, early_stopping_callbacks], 
                                          default_root_dir=self.training_dir, 
                                          logger=logger, 
                                          **self.args.trainer)
            
    def init_feature_embeddings(self):
        """
        Initialize feature embeddings either for drugs or covariates 
        """
        # Contains the embedding class of multiple feature types
        self.feature_embeddings = {}  
        self.num_classes = {}
                
        for cov, cov_names in self.dataset.id2cov.items():
            self.feature_embeddings["y_"+cov] = CategoricalFeaturizer(len(cov_names), 
                                                                      self.args.dataset.one_hot_encode_features, 
                                                                      self.device, 
                                                                      embedding_dimensions=self.args.dataset.cov_embedding_dimensions)
            if self.args.dataset.one_hot_encode_features:
                self.num_classes["y_"+cov] = len(cov_names)
            else:
                self.num_classes["y_"+cov] = self.args.dataset.cov_embedding_dimensions

    def init_model(self):
        """Initialize the (optional) autoencoder and generative model 
        """
        denoising_model = SimpleMLPTimeStep(in_dim=self.in_dim, 
                                            time_varying=True, 
                                            **self.args.denoising_module).to(self.device)
        
        size_factor_statistics = {"mean": self.dataset.log_size_factor_mu, 
                                  "sd": self.dataset.log_size_factor_sd}
        
        self.generative_model = VDM(
            denoising_model=denoising_model,
            feature_embeddings=self.feature_embeddings,
            plotting_folder=self.plotting_dir,
            in_dim=self.in_dim,
            size_factor_statistics=size_factor_statistics,
            scaler=self.dataset.get_scaler(),
            encoder_type=self.args.dataset.encoder_type,
            **self.args.generative_model  # model_kwargs should contain the rest of the arguments
        )

    def _check_is_initialized(self):
        if not self.generative_model:
            raise RuntimeError('You need to call self.init_model before calling self.train')
        if not self.datamodule:
            raise RuntimeError('You need to call self.init_datamodule before calling self.train')
        if not self.trainer_generative:
            raise RuntimeError('You need to call self.init_trainer before calling self.train')

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
    