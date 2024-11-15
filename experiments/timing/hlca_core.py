import numpy as np
import pandas as pd
import yaml
import torch
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
from celldreamer.eval.eval_utils import normalize_and_compute_metrics
import scipy.sparse as sp

import muon as mu
from torch import nn
import scanpy as sc    

from celldreamer.data.scrnaseq_loader import RNAseqLoader
from celldreamer.models.featurizers.category_featurizer import CategoricalFeaturizer
from celldreamer.models.fm.fm import FM
from celldreamer.eval.optimal_transport import wasserstein
import random
from celldreamer.models.base.encoder_model import EncoderModel
from celldreamer.models.base.utils import unsqueeze_right

from celldreamer.paths import DATA_DIR

device  = "cuda" if torch.cuda.is_available() else "cpu"

sc.settings.figdir = 'figures'  # Directory to save figures
sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(6, 6)) 


dataset_config = {'dataset_path': DATA_DIR / 'processed' / 'atac' / 'neurips' / 'neurips_multiome_test.h5mu',
                    'layer_key': 'X_counts',
                    'covariate_keys': ['cell_type'],
                    'conditioning_covariate': 'cell_type',
                    'subsample_frac': 1,
                    'encoder_type': 'learnt_autoencoder',
                    'one_hot_encode_features': False,
                    'split_rates': [0.90, 0.10],
                    'cov_embedding_dimensions': 256, 
                    'multimodal': True,
                    'is_binarized': False}

data_path = dataset_config["dataset_path"]

dataset = RNAseqLoader(data_path = data_path,
                        layer_key=dataset_config["layer_key"],
                        covariate_keys=dataset_config["covariate_keys"],
                        subsample_frac=dataset_config["subsample_frac"], 
                        encoder_type=dataset_config["encoder_type"], 
                        multimodal=dataset_config["multimodal"],
                        is_binarized=dataset_config["is_binarized"])

encoder_config = {"x0_from_x_kwargs": 
                      {"rna": {"dims": [512, 300, 100],
                               "batch_norm": True,
                               "dropout": False,
                               "dropout_p": 0.0},
                       "atac": {"dims": [1024, 512, 100],
                               "batch_norm": True,
                               "dropout": False,
                               "dropout_p": 0.0}},
                    "learning_rate": 0.001,
                    "weight_decay": 0.00001,
                    "covariate_specific_theta": False,
                    "multimodal": True, 
                    "is_binarized": False,
                    "encoder_multimodal_joint_layers": None
                 }

state_dict_path = "/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_neurips_multimodal/9e32dc3f-46f5-4eca-849c-40072c9ef0b0/checkpoints/last.ckpt"
gene_dim = {mod: dataset.X[mod].shape[1] for mod in dataset.X}
modality_list = list(gene_dim.keys())
in_dim = {}

for mod in dataset.X:
    if dataset_config["encoder_type"]!="learnt_autoencoder":
        in_dim[mod] = gene_dim[mod]
    else:
        in_dim[mod] = encoder_config["x0_from_x_kwargs"][mod]["dims"][-1]
                    
size_factor_statistics = {"mean": {mod: dataset.log_size_factor_mu[mod] for mod in dataset.log_size_factor_mu}, 
                            "sd": {mod: dataset.log_size_factor_sd[mod] for mod in dataset.log_size_factor_sd}}

n_cat = len(dataset.id2cov["cell_type"])

encoder_model = EncoderModel(in_dim=gene_dim,
                              n_cat=n_cat,
                              conditioning_covariate=dataset_config["conditioning_covariate"], 
                              encoder_type=dataset_config["encoder_type"],
                              **encoder_config)


encoder_model.load_state_dict(torch.load(state_dict_path)["state_dict"])

encoder_model.eval()

generative_model_config = {'learning_rate': 0.0001,
                            'weight_decay': 0.00001,
                            'antithetic_time_sampling': True,
                            'sigma': 0.0001
                        }

ckpt = torch.load("/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/fm_resnet_autoencoder_neurips_multimodal/2989f065-c3db-49c0-806b-40ade02b2d02/checkpoints/last.ckpt")

denoising_model = ckpt["hyper_parameters"]["denoising_model"]
denoising_model

feature_embeddings = ckpt["hyper_parameters"]["feature_embeddings"]

generative_model = FM(
            encoder_model=encoder_model,
            denoising_model=denoising_model,
            feature_embeddings=feature_embeddings,
            plotting_folder=None,
            in_dim=in_dim,
            size_factor_statistics=size_factor_statistics,
            encoder_type=dataset_config["encoder_type"],
            conditioning_covariate=dataset_config["conditioning_covariate"],
            model_type=denoising_model.model_type, 
            multimodal=dataset_config["multimodal"],
            is_binarized=dataset_config["is_binarized"], 
            modality_list=modality_list,
            **generative_model_config  # model_kwargs should contain the rest of the arguments
            )

generative_model.load_state_dict(ckpt["state_dict"])
generative_model.to("cuda")

adata_original = mu.read(data_path)
adata_rna = adata_original.mod["rna"]
adata_atac = adata_original.mod["atac"]
adata_rna.obs["size_factor"]=adata_rna.X.A.sum(1)
adata_atac.obs["size_factor"]=adata_atac.X.A.sum(1)
X_rna = torch.tensor(adata_rna.layers["X_counts"].todense())
X_atac = torch.tensor(adata_atac.layers["X_counts"].todense())

saving_dir = DATA_DIR / "generated" / "neurips_multimodal"