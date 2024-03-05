import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from scvi.distributions import NegativeBinomial
from celldreamer.models.base.utils import MLP

class EncoderModel(pl.LightningModule):
    def __init__(self,
                 in_dim,
                 x0_from_x_kwargs, 
                 scaler, 
                 learning_rate, 
                 weight_decay, 
                 covariate_specific_theta, 
                 encoder_type, 
                 conditioning_covariate, 
                 n_cat=None):
        
        super().__init__()  
        self.in_dim = in_dim
        self.x0_from_x_kwargs = x0_from_x_kwargs
        self.scaler = scaler
        self.learning_rate = learning_rate 
        self.weight_decay = weight_decay
        self.covariate_specific_theta = covariate_specific_theta
        self.encoder_type = encoder_type 
        self.conditioning_covariate = conditioning_covariate 
        self.n_cat = n_cat 
        
        # Define the (log) inverse dispersion parameter (negative binomial)
        if not covariate_specific_theta:
            self.theta = torch.nn.Parameter(torch.randn(self.in_dim), requires_grad=True)
        else:
            self.theta = torch.nn.Parameter(torch.randn(n_cat, self.in_dim), requires_grad=True)
        
        # If the encoder is fixed, we just need an inverting decoder. If learnt, the decoding is simply the softmax operation 
        if encoder_type == "learnt_encoder":
            x0_from_x_kwargs["dims"] = [self.in_dim, *x0_from_x_kwargs["dims"], self.in_dim]
            self.x0_from_x = MLP(**x0_from_x_kwargs)
        elif encoder_type == "learnt_autoencoder":
            x0_from_x_kwargs["dims"] = [self.in_dim, *x0_from_x_kwargs["dims"]]  # Encoder params
            self.x0_from_x =  MLP(**x0_from_x_kwargs)
            x0_from_x_kwargs["dims"] = x0_from_x_kwargs["dims"][::-1] # Decoder params
            self.x_from_x0 = MLP(**x0_from_x_kwargs)
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
    
    def _step(self, batch, dataset_type):
        X = batch["X"].to(self.device)
        size_factor = X.sum(1).unsqueeze(1).to(self.device)
        y = batch["y"][self.conditioning_covariate].to(self.device)
        
        z = self.encode(batch)
        mu_hat = self.decode(z, size_factor)

        #see scvi tools
        if not self.covariate_specific_theta:
            px = NegativeBinomial(mu=mu_hat, theta=torch.exp(self.theta))
        else:
            px = NegativeBinomial(mu=mu_hat, theta=torch.exp(self.theta[y]))
            
        loss = - px.log_prob(X).sum(1).mean() # counts

        self.log(f'{dataset_type}/loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "valid")
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), 
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)
    
    def encode(self, batch):
        X_scaled = self.scaler.scale(batch["X_norm"].to(self.device), reverse=False)
        return self.x0_from_x(X_scaled)
    
    def decode(self, x, size_factor):
        if self.encoder_type == "learnt_autoencoder":
            x = self.x_from_x0(x)
        mu_hat = F.softmax(x, dim=1)    
        mu_hat = mu_hat * size_factor
        return mu_hat
    