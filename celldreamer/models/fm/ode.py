import torch 

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model, l, y):
        super().__init__()
        self.model = model  # Model class
        self.l = l  # log library size 
        self.y = y  # conditioning variable

    def forward(self, t, x, *args, **kwargs):
        # Repeat and concatenate 
        t = t.repeat(x.shape[0])[:, None]
        m = self.model(x, t, self.l, self.y)
        return m
    