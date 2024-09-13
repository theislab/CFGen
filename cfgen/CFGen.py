import uuid
from typing import Literal

from omegaconf import OmegaConf
from hydra import compose, initialize

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from cfgen.paths import TRAINING_FOLDER

from cfgen.data.scrnaseq_loader import RNAseqLoader
from cfgen.models.base.encoder_model import EncoderModel
from cfgen.models.featurizers.category_featurizer import CategoricalFeaturizer
from cfgen.models.fm.denoising_model import MLPTimeStep
from cfgen.models.fm.fm import FM


class CFGen:
    def __init__(self, adata, project: str="CFGen model"):
        self.adata = adata
        self.adata_set_up = False

        initialize(version_base=None, config_path="../configs/") # TODO check whether this is relative to file or working dir
        self.cfg_encoder = compose("configs_encoder/train.yaml")
        self.cfg_sccfm = compose("configs_sccfm/train.yaml")

        # to name the run
        self.unique_id = str(uuid.uuid4())


        # Initialize training/plotting dir
        self.training_dir = TRAINING_FOLDER / project / self.unique_id
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.plotting_dir = self.training_dir / "plots"
        self.plotting_dir.mkdir(exist_ok=True)


        # Set device for training 
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # TODO this is currently overwritten by trainer/default.yaml

        # TODO check if adata contains counts and warn if not

    def setup_encoder(self):
        
        checkpoint_callback = ModelCheckpoint(dirpath=self.training_dir / "encoder" / "checkpoints", 
                                                **self.cfg_encoder.checkpoints)
        callbacks = [checkpoint_callback]
        
        # Initialize the PyTorch Lightning trainer with the specified callbacks and logger
        self.trainer_encoder = Trainer(callbacks=callbacks, 
                                          default_root_dir=self.training_dir / "encoder", 
                                          **self.cfg_encoder.trainer)

        self.encoder_model = EncoderModel(in_dim=self.gene_dim,
                                            n_cat=self.n_cat,
                                            conditioning_covariate=self.theta_covariate,
                                            **self.cfg_encoder.encoder)


    def setup_sccfm(self):
        # Callbacks for saving checkpoints 
        checkpoint_callback = ModelCheckpoint(dirpath=self.training_dir / "sccfm" / "checkpoints", 
                                                **self.cfg_sccfm.checkpoints)
        callbacks = [checkpoint_callback]
        
        self.trainer_sccfm = Trainer(callbacks=callbacks, 
                                          default_root_dir=self.training_dir / "sccfm", 
                                          logger=False, # TODO this is needed to not crash lightning. Fix this properly
                                          **self.cfg_sccfm.trainer)
        

    def setup_anndata(self,
                      layer_key: str=None,
                      covariate_keys: list=None,
                      normalization_type: str="log_gexp",
                      is_binarized: bool=False,
                      conditioning_method: Literal["classic", "guided"] = None, # TODO implement
                      theta_covariate: str=None,
                      size_factor_covariate: str=None,
                      one_hot_encode_features: bool=False,
                      guidance_weights: dict=None):

        self.is_binarized = is_binarized
        self.theta_covariate = theta_covariate
        self.one_hot_encode_features = one_hot_encode_features
        self.size_factor_covariate = size_factor_covariate
        self.guidance_weights = guidance_weights

        if self.adata_set_up:
            raise Exception("AnnData object has already been set up")

        self.dataset = RNAseqLoader(self.adata,
                                    layer_key=layer_key,
                                    covariate_keys=covariate_keys,
                                    subsample_frac=1, 
                                    normalization_type=normalization_type,
                                    multimodal=True, # TODO remove
                                    is_binarized=self.is_binarized)


        if self.theta_covariate:
            self.n_cat = len(self.dataset.id2cov[self.theta_covariate])
        else:
            self.n_cat = None

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.cfg_encoder.training_config.batch_size,
                                                        shuffle=True,
                                                        num_workers=4, 
                                                        drop_last=True)

        self.get_fixed_rna_model_params()

        self.setup_encoder()

        self.setup_sccfm()

        self.init_feature_embeddings()  # Initialize the feature embeddings 

        self.init_flow_model()


    def init_flow_model(self):
        """Initialize the (optional) autoencoder and generative model 
        """
        size_factor_statistics = {"mean": {mod: self.dataset.log_size_factor_mu[mod] for mod in self.dataset.log_size_factor_mu}, 
                                    "sd": {mod: self.dataset.log_size_factor_sd[mod] for mod in self.dataset.log_size_factor_sd}}
                

        # Initialize the deoising model 
        denoising_model = MLPTimeStep(in_dim=sum(self.in_dim.values()), 
                                        hidden_dim=self.cfg_sccfm.denoising_module.hidden_dim,
                                        dropout_prob=self.cfg_sccfm.denoising_module.dropout_prob,
                                        n_blocks=self.cfg_sccfm.denoising_module.n_blocks, 
                                        size_factor_min=self.dataset.min_size_factor, 
                                        size_factor_max=self.dataset.max_size_factor,
                                        embed_size_factor=self.cfg_sccfm.denoising_module.embed_size_factor, 
                                        covariate_list=self.dataset.covariate_keys,
                                        embedding_dim=self.cfg_sccfm.denoising_module.embedding_dim,
                                        normalization=self.cfg_sccfm.denoising_module.normalization,
                                        conditional=self.cfg_sccfm.denoising_module.conditional, 
                                        multimodal=True, # TODO remove 
                                        is_binarized=self.is_binarized, 
                                        modality_list=self.modality_list, 
                                        guided_conditioning=self.cfg_sccfm.denoising_module.guided_conditioning).to(self.device)
        
        print("Denoising model", denoising_model)
            
        # Flow matching model
        self.flow_model = FM(
            encoder_model=self.encoder_model,
            denoising_model=denoising_model,
            feature_embeddings=self.feature_embeddings,
            plotting_folder=self.plotting_dir,
            in_dim=self.in_dim,
            size_factor_statistics=size_factor_statistics,
            covariate_list=self.dataset.covariate_keys, 
            theta_covariate=self.theta_covariate,
            size_factor_covariate=self.size_factor_covariate,
            multimodal=True, # TODO remove
            is_binarized=self.is_binarized,
            modality_list=self.modality_list,
            guidance_weights=self.guidance_weights,
            **self.cfg_sccfm.generative_model  # model_kwargs should contain the rest of the arguments
            )



    def init_feature_embeddings(self):
        """
        Initialize feature embeddings either for drugs or covariates 
        """
        # Contains the embedding class of multiple feature types
        self.feature_embeddings = {}  
        self.num_classes = {}
                
        for cov, cov_names in self.dataset.id2cov.items():
            self.feature_embeddings[cov] = CategoricalFeaturizer(len(cov_names), 
                                                                    self.one_hot_encode_features, 
                                                                    self.device, 
                                                                    embedding_dimensions=self.cfg_sccfm.denoising_module.embedding_dim)
            if self.one_hot_encode_features:
                self.num_classes[cov] = len(cov_names)
            else:
                self.num_classes[cov] = self.cfg_sccfm.denoising_module.embedding_dim


    def get_fixed_rna_model_params(self):                               
        self.gene_dim = {mod: self.dataset.X[mod].shape[1] for mod in self.dataset.X}
        self.modality_list = list(self.gene_dim.keys())
        self.in_dim = {}
        if not self.cfg_encoder.encoder.encoder_multimodal_joint_layers:  # Optional latent space shared between modalities
            for mod in self.dataset.X:
                self.in_dim[mod] = self.cfg_encoder.encoder.encoder_kwargs[mod]["dims"][-1]
        else:
            self.in_dim = self.cfg_encoder.encoder.encoder_multimodal_joint_layers["dims"][-1]


    def train(self):
        print("Training encoder")
        self.trainer_encoder.fit(self.encoder_model,
            train_dataloaders=self.dataloader)
        print("Training flow model")
        self.trainer_sccfm.fit(self.flow_model,
                    train_dataloaders=self.dataloader)
        

    def generate(self):
        pass

    def batch_translate(self):
        pass

    def get_representations(self):
        pass # TODO automatically write them into the AnnData after training?

    def save(self):
        pass

    def load(self):
        pass

    def __repr__(self):
        return "CFGen model" # TODO

