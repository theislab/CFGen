from typing import Callable, List, Optional
import torch
import torch.nn as nn

def unsqueeze_right(x, num_dims=1):
    """Unsqueezes the last `num_dims` dimensions of `x`."""
    return x.view(x.shape + (1,) * num_dims)

def kl_std_normal(mean_squared, var):
    """Gaussian KL divergence
    """
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)

class MLP(torch.nn.Module):
    def __init__(self, 
                 dims: list,
                 batch_norm: bool, 
                 dropout: bool, 
                 dropout_p: float, 
                 activation = torch.nn.SELU, 
                 final_activation = None):
        
        super(MLP, self).__init__()

        # Attributes 
        self.dims = dims
        self.batch_norm = batch_norm
        self.activation = activation

        # MLP 
        layers = []
        for i in range(len(self.dims[:-1])):
            block = []
            block.append(torch.nn.Linear(self.dims[i], self.dims[i+1]))
            if batch_norm: 
                block.append(torch.nn.BatchNorm1d(self.dims[i+1]))
            block.append(self.activation())
            if dropout:
                block.append(torch.nn.Dropout(dropout_p))
            layers.append(torch.nn.Sequential(*block))
        self.net = torch.nn.Sequential(*layers)
        
        if final_activation == "tanh":
            self.final_activation = torch.nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x):
        x = self.net(x)
        if not self.final_activation:
            return x
        else:
            return self.final_activation(x)
