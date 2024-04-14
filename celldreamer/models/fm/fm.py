from typing import Literal
import numpy as np
from pathlib import Path

import torch
from torch import nn, linspace
import torch.nn.functional as F
from torch.distributions import Normal

import pytorch_lightning as pl

from scvi.distributions import NegativeBinomial
from torch.distributions import Poisson, Bernoulli
from celldreamer.models.base.cell_decoder import CellDecoder
from celldreamer.eval.evaluate import compute_umap_and_wasserstein
from celldreamer.models.base.utils import pad_t_like_x
from celldreamer.models.fm.ode import torch_wrapper
from celldreamer.models.fm.ot_sampler import OTPlanSampler

from torchdyn.core import NeuralODE

class FM(pl.LightningModule):
    def __init__(self,
                 encoder_model: nn.Module,
                 denoising_model: nn.Module,
                 feature_embeddings: dict, 
                 plotting_folder: Path,
                 in_dim: int,
                 size_factor_statistics: dict,
                 scaler, 
                 conditioning_covariate: str, 
                 model_type: str,
                 encoder_type: str = "fixed", 
                 learning_rate: float = 0.001, 
                 weight_decay: float = 0.0001, 
                 antithetic_time_sampling: bool = True, 
                 scaling_method: str = "log_normalization",  # Change int to str
                 sigma: float = 0.1, 
                 covariate_specific_theta: float = False, 
                 plot_and_eval_every=100, 
                 use_ot=True, 
                 multimodal=False, 
                 is_binarized=False, 
                 modality_list=None):
        """
        Flow matching for single-cell model. 
        
        Args:
            denoising_model (nn.Module): Denoising model.
            feature_embeddings (dict): Feature embeddings for covariates.
            x0_from_x_kwargs (dict): Arguments for the x0_from_x MLP.
            plotting_folder (Path): Folder for saving plots.
            in_dim (int): Number of genes.
            conditioning_covariate (str): Covariate controlling the size factor sampling.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            weight_decay (float, optional): Weight decay. Defaults to 0.0001.
            antithetic_time_sampling (bool, optional): Use antithetic time sampling. Defaults to True.
            scaling_method (str, optional): Scaling method for input data. Defaults to "log_normalization".
            sigma (float, optional): variance around straight path for flow matching objective.
        """
        super().__init__()
        
        self.encoder_model = encoder_model
        self.denoising_model = denoising_model.to(self.device)
        self.feature_embeddings = feature_embeddings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.in_dim = in_dim
        self.size_factor_statistics = size_factor_statistics
        self.scaler = scaler
        self.encoder_type = encoder_type
        self.antithetic_time_sampling = antithetic_time_sampling
        self.scaling_method = scaling_method
        self.plotting_folder = plotting_folder
        self.model_type = model_type
        self.conditioning_covariate = conditioning_covariate
        self.sigma = sigma
        self.covariate_specific_theta = covariate_specific_theta
        self.plot_and_eval_every = plot_and_eval_every
        self.use_ot = use_ot
        self.multimodal = multimodal
        self.is_binarized = is_binarized
        self.modality_list = modality_list
        
        # MSE lost for the Flow Matching algorithm 
        self.criterion = torch.nn.MSELoss()
                
        # Collection of testing observations for evaluation 
        if not self.multimodal:
            self.testing_outputs = []  
        else:
            self.testing_outputs = {mod: [] for mod in self.modality_list}
        
        # If the encoder is fixed, we just need an inverting decoder. If learnt, the decoding is simply the softmax operation 
        if encoder_type not in ["learnt_encoder", "learnt_autoencoder"]:
            self.cell_decoder = CellDecoder(self.encoder_type)
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
        
        # OT sampler
        if self.use_ot:
            self.ot_sampler = OTPlanSampler(method="exact")
    
    def training_step(self, batch, batch_idx):
        """
        Training step for VDM.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        return self._step(batch, dataset='train')

    def _step(self, batch, dataset: Literal['train', 'valid']):
        """
        Common step for training and validation.

        Args:
            batch: Batch data.
            dataset (Literal['train', 'valid']): Dataset type.

        Returns:
            torch.Tensor: Loss value.
        """
        # Collect observation and put onto device 
        x = batch["X"]  # counts
        if self.multimodal:
            x = {mod: x[mod].to(self.device) for mod in x}
        else:
            x = x.to(self.device)
        
        # Collect labels 
        y = batch["y"]
        y_fea = self.feature_embeddings[self.conditioning_covariate](y[self.conditioning_covariate])

        # Encode observations into the latent space
        if self.encoder_type in ["learnt_encoder", "learnt_autoencoder"]:
            with torch.no_grad():
                x0 = self.encoder_model.encode(batch)
                if self.multimodal:
                    x0 = torch.cat([x0[mod] for mod in self.modality_list], dim=1)
        else:
            x_scaled = self.scaler.scale(batch["X_norm"].to(self.device), reverse=False)
            x0 = x_scaled
            if self.multimodal:
                raise NotImplementedError

        # Quantify size factor 
        if not self.multimodal:
            size_factor = x.sum(1).unsqueeze(1)
            log_size_factor = torch.log(size_factor)
        else: 
            if self.is_binarized:
                # If binarized, the size factor is not required for atac 
                size_factor = x["rna"].sum(1).unsqueeze(1)
                log_size_factor = torch.log(size_factor)            
            else:
                size_factor = {mod: x[mod].sum(1).unsqueeze(1) for mod in self.modality_list}
                log_size_factor = {mod: torch.log(size_factor[mod]) for mod in self.modality_list}
        
        # Sample time 
        t = self._sample_times(x0.shape[0])  # B
        
        # Sample noise 
        z = self.sample_noise_like(x0)  # B x G
        
        # Get objective and perturbed observation
        t, x_t, u_t = self.sample_location_and_conditional_flow(z, x0, t)

        # Forward through the model 
        v_t = self.denoising_model(x_t, t, log_size_factor, y_fea)
        loss = self.criterion(u_t, v_t)  # (B, )
        
        # Save results
        metrics = {
            "batch_size": z.shape[0],
            f"{dataset}/loss": loss.mean()}
        self.log_dict(metrics, prog_bar=True)
        
        return loss.mean()
    
    # Private methods
    def _featurize_batch_y(self, batch):
        """
        Featurize all the covariates 

        Args:
            batch: Batch data.

        Returns:
            torch.Tensor: Featurized covariates.
        """
        y = []     
        for feature_cat in batch["y"]:
            y_cat = self.feature_embeddings[feature_cat](batch["y"][feature_cat])
            y.append(y_cat)
        y = torch.cat(y, dim=1).to(self.device)
        return y
    
    def _sample_times(self, batch_size):
        """
        Sample times, can be sampled to cover the 

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Sampled times.
        """
        if self.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
        else:
            times = torch.rand(batch_size, device=self.device)
        return times

    @torch.no_grad()
    def sample(self, batch_size, n_sample_steps, covariate, covariate_indices=None, log_size_factor=None):
        # Sample random noise 
        z = torch.randn((batch_size, self.denoising_model.in_dim), device=self.device)

        # Sample random classes from the sampling covariate 
        if covariate_indices==None:
            covariate_indices = torch.randint(0, self.feature_embeddings[covariate].n_cat, (batch_size,))
             
        # Sample size factor from the associated distribution
        if log_size_factor==None:
            # If size factor conditions the denoising, sample from the log-norm distribution. Else the size factor is None
            if self.multimodal and not self.is_binarized:
                log_size_factor = {}
                for mod in self.modality_list:
                    mean_size_factor, sd_size_factor = self.size_factor_statistics["mean"][mod][covariate], self.size_factor_statistics["sd"][mod][covariate]
                    mean_size_factor, sd_size_factor = mean_size_factor[covariate_indices], sd_size_factor[covariate_indices]
                    size_factor_dist = Normal(loc=mean_size_factor, scale=sd_size_factor)
                    log_size_factor_mod = size_factor_dist.sample().to(self.device).view(-1, 1)
                    log_size_factor[mod] = log_size_factor_mod
            else:
                mean_size_factor, sd_size_factor = self.size_factor_statistics["mean"][covariate], self.size_factor_statistics["sd"][covariate]
                mean_size_factor, sd_size_factor = mean_size_factor[covariate_indices], sd_size_factor[covariate_indices]
                size_factor_dist = Normal(loc=mean_size_factor, scale=sd_size_factor)
                log_size_factor = size_factor_dist.sample().to(self.device).view(-1, 1)
        
        # Featurize the covariate
        y = self.feature_embeddings[covariate](covariate_indices.cuda())

        # Generate 
        t = linspace(0.0, 1.0, n_sample_steps, device=self.device)
                
        denoising_model_ode = torch_wrapper(self.denoising_model, log_size_factor, y)    
        
        self.node = NeuralODE(denoising_model_ode,
                                solver="dopri5", 
                                sensitivity="adjoint", 
                                atol=1e-5, 
                                rtol=1e-5)        
        
        x0 = self.node.trajectory(z, t_span=t)[-1]
        
        # If multimodal, split the output to get separate z's
        if self.multimodal:
            x0 = torch.split(x0, [self.in_dim[d] for d in self.modality_list], dim=1)
            x0 = {mod: x0[i] for i, mod in enumerate(self.modality_list)}

        # Exponentiate log-size factor for decoding  
        if self.multimodal and not self.is_binarized:
            size_factor = {mod: torch.exp(log_size_factor[mod]) for mod in self.modality_list}
        else:
            size_factor = torch.exp(log_size_factor)
            
        # Decode to parameterize sampling distributions
        x = self._decode(x0, size_factor)

        # Sample from noise model
        if not self.multimodal:
            if not self.covariate_specific_theta:
                distr = NegativeBinomial(mu=x, theta=torch.exp(self.encoder_model.theta))
            else:
                distr = NegativeBinomial(mu=x, theta=torch.exp(self.encoder_model.theta[covariate_indices]))
            sample = distr.sample()
        else:
            sample = {}  # containing final samples 
            for mod in x:
                if mod=="rna":  
                    if not self.covariate_specific_theta:
                        distr = NegativeBinomial(mu=x[mod], theta=torch.exp(self.encoder_model.theta))
                    else:
                        distr = NegativeBinomial(mu=x[mod], theta=torch.exp(self.encoder_model.theta[covariate_indices]))
                else:  # if mod is atac
                    if not self.encoder_model.is_binarized:
                        distr = Poisson(rate=x[mod])
                    else:
                        distr = Bernoulli(probs=x[mod])
                sample[mod] = distr.sample() 
        return sample

    @torch.no_grad()
    def batched_sample(self, batch_size, repetitions, n_sample_steps, covariate, covariate_indices=None, log_size_factor=None):
        if not self.multimodal:
            total_samples = []
        else:
            total_samples = {mod:[] for mod in self.modality_list}
            
        # Covariate is same for all modalities 
        for i in range(repetitions): 
            covariate_indices_batch = covariate_indices[(i*batch_size):((i+1)*batch_size)] if covariate_indices != None else None
            # Input to the sampling pre-defined size factors if provided to the function 
            if not self.multimodal or (self.multimodal and self.is_binarized):
                log_size_factor_batch = log_size_factor[(i*batch_size):((i+1)*batch_size)] if log_size_factor != None else None 
            else:
                if log_size_factor != None:
                    log_size_factor_batch = {} 
                    for mod in self.modality_list:
                        log_size_factor_batch[mod] = log_size_factor[mod][(i*batch_size):((i+1)*batch_size)] 
                else:
                    log_size_factor_batch = None
            
            # Sample batch 
            X_samples = self.sample(batch_size, n_sample_steps, covariate, covariate_indices_batch, log_size_factor_batch)
                
            if not self.multimodal: 
                total_samples.append(X_samples.cpu())
            else:
                for mod in X_samples:
                    total_samples[mod].append(X_samples[mod].cpu())                
        
        # Concatenate observations in the samples 
        if not self.multimodal:
            return torch.cat(total_samples, dim=0)
        else:
            return {mod: torch.cat(total_samples[mod], dim=0) for mod in self.modality_list}                

    def _decode(self, z, size_factor):
        # Decode the rescaled z
        if self.encoder_type not in ["learnt_autoencoder", "learnt_encoder"]:
            if self.multimodal:
                raise NotImplementedError
            else:
                z = self.cell_decoder(self.scaler.scale(z, reverse=True), size_factor)
        else:
            if self.multimodal and self.is_binarized:
                size_factor = {"rna": size_factor}  # Compatibility with the decoder implementation for multimodal data 
            z = self.encoder_model.decode(z, size_factor)
        return z
    
    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        # Resample from OT coupling 
        if self.use_ot:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        # Sample time 
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        # Sample noise along straight line
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return t, xt, ut

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x1)
        if self.use_ot:
            mu_t = t * x1 + (1 - t) * x0
        else:
            mu_t = t * x1
        return mu_t
    
    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if self.use_ot:
            return self.sigma
        else:
            return 1 - (1 - self.sigma) * t
    
    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if self.use_ot:
            return x1 - x0
        else:
            t = t.unsqueeze(1)
            return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)

    def configure_optimizers(self):
        """
        Optimizer configuration 

        Returns:
            dict: Optimizer configuration.
        """
        params = list(self.parameters())
        
        if not self.feature_embeddings[self.conditioning_covariate].one_hot_encode_features:
            for cov in self.feature_embeddings:
                params += list(self.feature_embeddings[cov].parameters())
                 
        optimizer = torch.optim.AdamW(params, 
                                    self.learning_rate, 
                                    weight_decay=self.weight_decay)
        return optimizer

    def validation_step(self, batch, batch_idx):
        """
        Validation step for VDM.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        return self._step(batch, dataset='valid')
    
    def test_step(self, batch, batch_idx):
        """
        Training step for VDM.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        # Append the batches
        if not self.multimodal:
            self.testing_outputs.append(batch["X"].cpu())
        else:
            for mod in self.modality_list:
                self.testing_outputs[mod].append(batch["X"][mod].cpu())

    def on_test_epoch_end(self, *arg, **kwargs):
        self.compute_metrics_and_plots(dataset_type="test")
        if not self.multimodal:
            self.testing_outputs = []
        else:
            self.testing_outputs = {}

    @torch.no_grad()
    def compute_metrics_and_plots(self, dataset_type, *arg, **kwargs):
        """
        Concatenates all observations from the test data loader in a single dataset.

        Args:
            outputs: List of outputs from the test step.

        Returns:
            None
        """
        # Concatenate all test observations
        if not self.multimodal:
            testing_outputs = torch.cat(self.testing_outputs, dim=0)
        else:
            testing_outputs = {mod: torch.cat(self.testing_outputs[mod], dim=0) for mod in self.testing_outputs}
        
        # Plot UMAP of generated cells and real test cells
        wd = compute_umap_and_wasserstein(model=self, 
                                            batch_size=1000, 
                                            n_sample_steps=2, 
                                            plotting_folder=self.plotting_folder, 
                                            X_real=testing_outputs, 
                                            epoch=self.current_epoch,
                                            conditioning_covariate=self.conditioning_covariate)
        
        del testing_outputs
        metric_dict = {}
        for key in wd:
            metric_dict[f"{dataset_type}_{key}"] = wd[key]

        # Compute Wasserstein distance between real test set and generated data 
        self.log_dict(wd)
        return wd
    