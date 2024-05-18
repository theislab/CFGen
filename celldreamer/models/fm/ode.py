import torch 

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model, l, y, guidance_weights, conditioning_covariates, unconditional=False):
        super().__init__()
        self.model = model  # Model class
        self.l = l  # log library size 
        self.y = y  # conditioning variable
        self.guidance_weights = guidance_weights
        self.conditioning_covariates = conditioning_covariates
        self.unconditional = unconditional
        self.guided_conditioning = model.guided_conditioning

    def forward(self, t, x, *args, **kwargs):
        # Repeat and concatenate 
        t = t.repeat(x.shape[0])[:, None]
        if self.unconditional or self.guided_conditioning:
            m_uncond = self.model(x, t, self.l, self.y, inference=True, unconditional=True, covariate=None)
            m = m_uncond.clone()
        if not self.unconditional:
            if self.guided_conditioning:
                for cov in self.conditioning_covariates:
                    m += self.guidance_weights[cov] * \
                        (self.model(x, t, self.l, self.y, inference=True, unconditional=False, covariate=cov) - m_uncond)
            
            if not self.guided_conditioning:
                m = self.model(x, t, self.l, self.y, inference=True, unconditional=False, covariate=None) 
        return m
