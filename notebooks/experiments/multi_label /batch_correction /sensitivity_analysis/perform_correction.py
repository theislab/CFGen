import numpy as np
import pandas as pd
import yaml
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from celldreamer.eval.eval_utils import normalize_and_compute_metrics
import scipy.sparse as sp
from celldreamer.eval.eval_utils import join_real_generated

from torch import nn
import seaborn as sns
import scanpy as sc    

from celldreamer.data.scrnaseq_loader import RNAseqLoader
from celldreamer.models.featurizers.category_featurizer import CategoricalFeaturizer
from celldreamer.models.fm.fm import FM
from celldreamer.eval.optimal_transport import wasserstein
import random
from celldreamer.models.base.encoder_model import EncoderModel
from celldreamer.models.base.utils import unsqueeze_right
from celldreamer.paths import DATA_DIR
from scvi.distributions import NegativeBinomial

from torchdyn.core import NeuralODE
from celldreamer.models.fm.ode import torch_wrapper
from tqdm import tqdm 
from pathlib import Path

def perform_corrections(batch_key, bio_key, target_batch, generative_model, dataloader, encoder_model):
    X_inverted = []  # Inverted --> Corrected 
    X_corrected = []  # Final result 
    X_corrected_decoded = []  # Final result decoded 
    encoded_lab = {batch_key: [],
                      bio_key: []}  # Encode latent space ncoded labels 
    
    # Times to apply correction
    t = torch.linspace(0.0, 1.0, 2, device=generative_model.device)  # Time to data
    reverse_t = torch.linspace(1.0, 0.0, 2, device=generative_model.device)  # Time to noise 
    
    # Encode all batches in the dataloader 
    for batch in dataloader:
        with torch.no_grad():
            # Encode latent space 
            z = encoder_model.encode(batch)
            
            for cov in batch["y"]:
                encoded_lab[cov] += batch["y"][cov].tolist()
    
            # Get size factor
            log_size_factor = torch.log(batch["X"].sum(1))
    
            # Get condition embeddings
            y = {}
            for c in batch["y"]:
                y[c] = generative_model.feature_embeddings[c](batch["y"][c].cuda())
    
            # Go back to noise
            denoising_model_ode = torch_wrapper(generative_model.denoising_model, 
                                                log_size_factor, 
                                                y,
                                                guidance_weights=generative_model.guidance_weights,
                                                conditioning_covariates=[bio_key, batch_key], 
                                                unconditional=False)    
            
            node = NeuralODE(denoising_model_ode,
                                    solver="dopri5", 
                                    sensitivity="adjoint", 
                                    atol=1e-5, 
                                    rtol=1e-5)        
    
            z0 = node.trajectory(z, t_span=reverse_t)[-1]
            X_inverted.append(z0)

            y[batch_key] = generative_model.feature_embeddings[batch_key]((torch.ones(z0.shape[0]) * 3).long())
            
            # Decode noise with single batch
            denoising_model_ode = torch_wrapper(generative_model.denoising_model, 
                                                log_size_factor, 
                                                y,
                                                guidance_weights=generative_model.guidance_weights,
                                                conditioning_covariates=[bio_key, batch_key], 
                                                unconditional=False)    
    
            z1 = node.trajectory(z0, t_span=t)[-1]
            X_corrected.append(z1)
    
            # Now decode
            mu_hat = generative_model._decode(z1, torch.exp(log_size_factor).cuda().unsqueeze(1))
            distr = NegativeBinomial(mu=mu_hat, theta=torch.exp(encoder_model.theta))
            X_corrected_decoded.append(distr.sample())
    
    X_inverted = torch.cat(X_inverted, dim=0)
    X_corrected = torch.cat(X_corrected, dim=0)
    X_corrected_decoded = torch.cat(X_corrected_decoded, dim=0)
    obs = pd.DataFrame(encoded_lab)
    return X_inverted, X_corrected, X_corrected_decoded, obs
