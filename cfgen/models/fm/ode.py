import torch 

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model, l, y, guidance_weights, conditioning_covariates, unconditional=False):
        """
        Initializes the torch_wrapper.

        Args:
            model: The model to be wrapped.
            l: Log library size, presumably a tensor.
            y: Conditioning variable, presumably a tensor.
            guidance_weights: Weights for attribute-based guiding, presumably a dictionary.
            conditioning_covariates: Names of the covariates used for the conditioning, presumably a list.
            unconditional (bool, optional): Flag for unconditional generation. Defaults to False.
        """
        super().__init__()
        self.model = model  # The model being wrapped
        self.l = l  # Log library size
        self.y = y  # Conditioning variable
        self.guidance_weights = guidance_weights  # Weights for attribute-based guiding
        self.conditioning_covariates = conditioning_covariates  # Covariate names for conditioning
        self.unconditional = unconditional  # Flag for unconditional generation
        self.guided_conditioning = model.guided_conditioning  # Model's guided conditioning flag

    def forward(self, t, x, *args, **kwargs):
        """
        Forward pass of the torch_wrapper.

        Args:
            t: Time tensor, will be repeated for each sample in the batch.
            x: Input tensor.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The output of the model after applying conditioning.
        """
        # Repeat and concatenate time tensor to match the batch size
        t = t.repeat(x.shape[0])[:, None]

        # Unconditional generation or guided conditioning
        if self.unconditional or self.guided_conditioning:
            m_uncond = self.model(x, t, self.l, self.y, inference=True, unconditional=True, covariate=None)
            m = m_uncond.clone()
        
        # Conditional generation
        if not self.unconditional:
            if self.guided_conditioning:
                # Apply guided conditioning using provided weights
                for cov in self.conditioning_covariates:
                    m += self.guidance_weights[cov] * \
                         (self.model(x, t, self.l, self.y, inference=True, unconditional=False, covariate=cov) - m_uncond)
            else:
                # Normal conditioning without guidance
                m = self.model(x, t, self.l, self.y, inference=True, unconditional=False, covariate=None)
        
        return m
