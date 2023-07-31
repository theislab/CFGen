import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial 
from torch.distributions import Normal, kl_divergence
import pytorch_lightning as pl

from celldreamer.models.base.utils import MLP


class BaseAutoencoder(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        batch_norm,
        dropout,
        dropout_p,
        activation=torch.nn.ReLU,
        likelihood="nb",
    ):
        super(BaseAutoencoder, self).__init__()

        # Attributes
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.likelihood = likelihood
        self.latent_dim = hidden_dims[-1]

        # Encoder
        self.encoder_layers = MLP(
            hidden_dims=[in_dim, *hidden_dims[:-1]],
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_p=dropout_p,
            activation=activation,
        )

        # Decoder
        self.decoder_layers = MLP(
            hidden_dims=[*hidden_dims[::-1]],
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_p=dropout_p,
            activation=activation,
        )

        if likelihood == "gaussian":
            self.log_sigma = torch.nn.Parameter(torch.randn(self.in_dim))
            self.decoder_mu = torch.nn.Linear(hidden_dims[0], self.in_dim)

        elif likelihood == "nb":
            self.theta = torch.nn.Parameter(torch.randn(self.in_dim))
            self.decoder_mu = torch.nn.Linear(hidden_dims[0], self.in_dim)

        elif likelihood == "zinb":
            self.theta = torch.nn.Parameter(torch.randn(self.in_dim))
            self.decoder_mu = torch.nn.Linear(hidden_dims[0], self.in_dim)
            self.decoder_rho = torch.nn.Linear(hidden_dims[0], self.in_dim)

        else:
            raise NotImplementedError

    def encode(self, x):
        pass

    def decode(self, z, library_size):
        h = self.decoder_layers(z)

        if self.likelihood == "gaussian":
            mu = self.decoder_mu(h)
            return dict(mu=mu)

        elif self.likelihood == "nb":
            mu = self.decoder_mu(h)
            mu = F.softmax(mu, dim=-1)
            mu = mu*library_size.unsqueeze(-1)
            return dict(mu=mu)

        elif self.likelihood == "zinb":
            mu = self.decoder_mu(h)
            mu = F.softmax(mu, dim=-1)
            mu = mu*library_size.unsqueeze(-1)
            rho = self.decoder_rho(h)
            return dict(mu=mu, rho=rho)

        else:
            raise NotImplementedError

    def forward(self, batch):
        pass

    def reconstruction_loss(self, x, decoder_output):
        if self.likelihood == "gaussian":
            mu = decoder_output["mu"]
            distr = Normal(loc=mu, theta=torch.exp(self.log_sigma)*torch.eye(self.in_dim))
            recon_loss = -distr.log_prob(x).sum(-1)

        elif self.likelihood == "nb":
            mu = decoder_output["mu"]
            distr = NegativeBinomial(mu=mu, theta=torch.exp(self.theta))
            recon_loss = -distr.log_prob(x).sum(-1)

        elif self.likelihood == "zinb":
            mu, rho = decoder_output["mu"], decoder_output["rho"]
            distr = ZeroInflatedNegativeBinomial(mu=mu, theta=torch.exp(self.theta), zi_logits=rho)
            recon_loss = -distr.log_prob(x).sum(-1)

        else:
            raise NotImplementedError
        return recon_loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


class AE(BaseAutoencoder):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        batch_norm,
        dropout,
        dropout_p,
        activation=torch.nn.ELU,
        likelihood="nb",
    ):
        super(AE, self).__init__(
            in_dim,
            hidden_dims,
            batch_norm,
            dropout,
            dropout_p,
            activation,
            likelihood,
        )
        
        self.latent_layer = torch.nn.Linear(hidden_dims[-2], self.latent_dim)
    
    def encode(self, x):
        h = self.encoder_layers(x)
        z = self.latent_layer(h)
        return dict(z=z)
    
    def forward(self, batch):
        x = batch["X"]
        library_size = x.sum(1)
        x_log = torch.log(1 + x)  # for now log transf
        z = self.encode(x_log)["z"]
        return self.decode(z, library_size)
    
    def step(self, batch, prefix):
        x = batch["X"]
        decoder_output = self.forward(batch)
        loss = torch.mean(self.reconstruction_loss(x, decoder_output))
        self.log(f"{prefix}/loss", loss, prog_bar=True)
        if prefix == "train":
            return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")


class VAE(BaseAutoencoder):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        batch_norm,
        dropout,
        dropout_p,
        n_epochs: int,
        kl_warmup_fraction: 0.5,
        kl_weight: None, 
        activation=torch.nn.ReLU,
        likelihood:str="nb"  
    ):
        super(VAE, self).__init__(
            in_dim,
            hidden_dims,
            batch_norm,
            dropout,
            dropout_p,
            activation,
            likelihood,
        )

        # Latent space
        self.mu = nn.Linear(hidden_dims[-2], self.latent_dim)
        self.logvar = nn.Linear(hidden_dims[-2], self.latent_dim)
        if kl_weight==None:
            self.kl_weight = 0
            self.kl_weight_increase = 1/(kl_warmup_fraction*n_epochs)
            self.anneal_kl = True
        else:
            self.kl_weight = kl_weight
            self.anneal_kl = False
        
    def encode(self, x):
        h = self.encoder_layers(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return dict(z=z,
                    mu=mu,
                    logvar=logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, batch):
        x = batch["X"]
        library_size = x.sum(1)
        x_log = torch.log(1 + x)  # just for numerical stability
        z, mu, logvar = self.encode(x_log).values()
        return self.decode(z, library_size), mu, logvar

    def kl_divergence(self, mu, logvar):
        p =  Normal(mu, torch.sqrt(torch.exp(0.5 * logvar)))
        q = Normal(torch.zeros_like(mu), torch.ones_like(mu))
        kl_div = kl_divergence(p, q).sum(dim=-1)
        return kl_div
        
    def step(self, batch, prefix):
        x = batch["X"]
        decoder_output, mu, logvar = self.forward(batch)
        recon_loss = self.reconstruction_loss(x, decoder_output)
        kl_div = self.kl_divergence(mu, logvar)
        kl_weight = min([self.kl_weight, 1])

        loss = torch.mean(recon_loss + kl_weight * kl_div)
        dict_losses = {f"{prefix}/loss": loss,
                       f"{prefix}/kl": kl_div.mean(),
                       f"{prefix}/lik": recon_loss.mean()}
        self.log_dict(dict_losses, prog_bar=True)
        if prefix == "train":
            return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")
    
    def amortized_sampling(self, batch):
        z = self.encode(batch)["z"]
        return self.sample_decoder(z)

    def random_sampling(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim)
        return self.sample_decoder(z)
        
    def sample_decoder(self, z_batch):
        # Decode a z_output
        decoder_output = self.decode(z_batch)
        if self.likelihood == "gaussian":
            mu = decoder_output["mu"]
            distr = Normal(loc=mu, theta=torch.sqrt(torch.exp(self.log_sigma)))
            return distr.sample(z_batch)

        elif self.likelihood == "nb":
            mu = decoder_output["mu"]
            distr = NegativeBinomial(mu=mu, theta=torch.exp(self.theta))
            return distr.sample(z_batch)

        elif self.likelihood == "zinb":
            mu, rho = decoder_output["mu"], decoder_output["rho"]
            distr = ZeroInflatedNegativeBinomial(
                mu=mu, theta=torch.exp(self.theta), zi_logits=rho
            )
            return distr.sample(z_batch)
        else:
            raise NotImplementedError
    
    def on_train_epoch_end(self):
        if self.anneal_kl:
            self.kl_weight += self.kl_weight_increase
        else:
            self.kl_weight += 0
            