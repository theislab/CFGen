import torch 

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, l):
        super().__init__()
        self.model = model  # Model class
        self.l = l  # log library size 

    def forward(self, t, x, *args, **kwargs):
        t = t.repeat(x.shape[0])[:, None]
        m = self.model(x, t, self.l)
        return m