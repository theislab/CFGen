from typing import Literal
import pytorch_lightning as pl
import torch
from torch import nn
from typing import Callable, Optional
from tqdm import tqdm
from functools import partial

from celldreamer.models.diffusion.variance_scheduler.cosine import CosineScheduler
from celldreamer.models.diffusion.diffusion_utils import extract, identity


class ConditionalGaussianDDPM(pl.LightningModule):
    """
    Implementation of "Classifier-Free Diffusion Guidance"
    """
    def __init__(self,
                 denoising_model: nn.Module,
                 autoencoder_model: nn.Module, 
                 feature_embeddings: dict, 
                 T: int,  
                 w: float,  
                 classifier_free: bool, 
                 p_uncond: float,
                 task: str, 
                 use_drugs: bool,
                 one_hot_encode_features: bool,
                 metric_collector, 
                 optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
                 variance_scheduler= CosineScheduler,  # default: cosine
                 learning_rate: float = 0.001, 
                 weight_decay: float = 0.0001, 
                 ):
        """
        :param denoising_model: The network which computes the denoise step, i.e., q(x_{t-1} | x_t, c)
        :param autoencoder_model: The network which computes the encoding step, i.e., q(z | x)
        :param feature_embeddings: The feature embeddings for each covariate
        :param T: The amount of noising steps
        :param w: strength of class guidance, hyperparemeter, paper suggests 0.3
        :param p_uncond: probability of training a batch without class conditioning
        :param task: task to train on, "perturbation_modelling", "cell_generation", "toy_generation"
        :param classifier_free: whether to apply the classifier-free logic or not
        :param metric_collector: metric collector object
        :param optimizer: optimizer to use
        :param variance_scheduler: scheduler for the variance
        :param learning_rate: learning rate
        :param weight_decay: weight decay
        """
        assert 0.0 <= w, f'0.0 <= {w}'
        assert 0.0 <= p_uncond <= 1.0, f'0.0 <= {p_uncond} <= 1.0'
        
        super().__init__()
        # Set device 
        
        # Denoising model and autoencoder (if required)
        self.denoising_model = denoising_model
        self.autoencoder_model = autoencoder_model
        self.metric_collector = metric_collector
        
        # Number of classes per covariate 
        self.num_classes = self.denoising_model.num_classes
        
        # Diffusion hyperparameters 
        self.T = T
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.p_uncond = p_uncond
        if isinstance(denoising_model, UNetTimeStepClassSetConditioned):
            self.in_dim = (32, 32, 3)  # UNetTimeStepClassSetConditioned is only for images, so we hardcode this
        elif isinstance(denoising_model, UNetTimeStepClassSetConditioned):
            self.in_dim = self.denoising_model.in_dim
        self.w = w        
        
        # Training hyperparameters
        self.task = task
        self.feature_embeddings = feature_embeddings
        self.classifier_free = classifier_free
        self.use_drugs = use_drugs
        self.one_hot_encode_features = one_hot_encode_features
        
        # Optimization 
        self.mse = nn.MSELoss()
        self.optim = optimizer
    
        # Initialize necessary values for variance schedule 
        self.var_scheduler = variance_scheduler(T = self.T)
        alphas_hat = self.var_scheduler.get_alpha_hat().to(self.device)
        alpha_hat_prev = self.var_scheduler.get_alpha_hat_prev().to(self.device)
        alphas = self.var_scheduler.get_alphas().to(self.device)
        betas = self.var_scheduler.get_betas().to(self.device)
        posterior_variance = self.var_scheduler.get_posterior_variance().to(self.device)    
        
        # Register all parameters in their buffer
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        # Standard values of the variance schedule 
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_hat)
        register_buffer('alphas_cumprod', alpha_hat_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_hat))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_hat))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_hat))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_hat))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_hat - 1))
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alpha_hat_prev) / (1. - alphas_hat))
        register_buffer('posterior_mean_coef2', (1. - alpha_hat_prev) * torch.sqrt(alphas) / (1. - alphas_hat))
    
    # Basic torch lightning setup
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        return self.denoising_model(x, t, y)

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'valid')
        
    def configure_optimizers(self):
        """
        Optimizer configuration 
        """
        parms_to_train = list(self.denoising_model.parameters())
        if self.task == "perturbation_modelling" and self.use_drugs:
            parms_to_train.extend(list(self.feature_embeddings["y_drug"].parameters()))
        if not self.one_hot_encode_features:
            for cov in self.feature_embeddings:
                if cov != "y_drug":
                    parms_to_train.extend(list(self.feature_embeddings[cov].embeddings.parameters()))
            
        optimizer_config = {'optimizer': self.optim(parms_to_train,
                                                    lr=self.learning_rate, 
                                                    weight_decay=self.weight_decay)}
        return optimizer_config
    
    # Forward
    def _step(self, batch, dataset: Literal['train', 'valid']) -> torch.Tensor:
        """
        train/validation step of DDPM. The logic is mostly taken from the original DDPM paper,
        except for the class conditioning part.
        """
        # Collect observation and optionally encode it 
        x = batch["X"].to(self.device)
        if self.autoencoder_model != None:
            x = self.autoencoder_model.encoder(x)
        
        # Define the classifier free strategy 
        y=self._featurize_batch_y(batch)
            
        # Sample t uniformly from [0, T]
        t = torch.randint(0, self.T, (x.shape[0],), device=x.device).long()
        loss = self.p_losses(x, t, y)
        self.log(f"loss/{dataset}_loss", loss, on_step=True)
        return loss

    def p_losses(self, x_start, t, y, noise = None):
        """
        Collect the MSE loss 
        """
        if noise == None:
            noise = torch.randn_like(x_start)  
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        model_out = self.denoising_model(x, t, y) 
        loss = self.mse(noise, model_out)
        return loss
        
    def q_sample(self, x_start, t, noise = None):
        """
        Sample from forward process     
        """
        if noise == None:
            noise = torch.randn_like(x_start)  
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    # Backward
    def sample(self, 
                batch_size: Optional[int] = None, 
                y: Optional[torch.Tensor] = None, 
                return_all_timesteps: bool = False, 
                clip_denoised: bool = True):
        """
        Sample generated cells
        """
        sample_fn = self.p_sample_loop 
        return sample_fn(batch_size, y, return_all_timesteps, clip_denoised)   
    
    @torch.no_grad()
    def ddim_sample(self, batch_size, y, return_all_timesteps = False, ddim_sampling_eta=0):
        batch_size, device, total_timesteps, eta = batch_size, self.device, self.T, ddim_sampling_eta

        times = torch.arange(-1, total_timesteps)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(batch_size, self.in_dim, device = device)
        xs = [x]

        x_start = None

        for time, time_next in tqdm(time_pairs):
            time_cond = torch.full((batch_size,), time, device = device, dtype = torch.long)
            pred_noise, x_start =list(self.model_predictions(x, time_cond, y, clip_x_start = True, rederive_pred_noise = True).values())

            if time_next < 0:
                x = x_start
                xs.append(x)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            xs.append(x)

        ret = x if not return_all_timesteps else torch.stack(xs, dim = 1)

        return ret

    
    def p_sample_loop(self, batch_size, y, return_all_timesteps, clip_denoised=True):
        """
        Loop to generate images
        """
        # Sample observation  
        x = torch.randn((batch_size, self.in_dim)).to(self.device)
        xs = [x]

        for t in reversed(range(0, self.T)):
            x, _ = self.p_sample(x, t, y, clip_denoised=clip_denoised)
            xs.append(x.cpu())
          
        ret = x if not return_all_timesteps else torch.stack(xs, dim = 0)
        return ret
    
    def p_sample(self, x, t, y, clip_denoised = True):
        """
        Sample from posterior
        """
        b = x.shape[0]
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, y = y, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    
    def p_mean_variance(self, x, t, y, clip_denoised = True):
        """
        Posterior mean and variance
        """
        preds = self.model_predictions(x, t, y)
        x_start = preds["x_start"]

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def model_predictions(self, x, t, y, clip_x_start = False, rederive_pred_noise = False):
        """
        Predict noise and starting point based on reverse diffusion  
        """
        # Predicted noise at a step
        pred_noise = self.denoising_model(x, t, y)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)
        
        if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
                
        return {"pred_noise": pred_noise, 
                "x_start": x_start}
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # Functions to perform direct jumps
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    # Private methods
    def _featurize_batch_y(self, batch):
        """
        Featurize all the covariates 
        """
        y = []     
        for feature_cat in batch["y"]:
            y_cat = self.feature_embeddings[feature_cat](batch["y"][feature_cat])
            y.append(y_cat)
        y = torch.cat(y, dim=1).to(self.device)
        return y
    