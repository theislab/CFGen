import torch
from torch import nn

class FixedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        """
        Fixed linear schedule for a value between `gamma_min` and `gamma_max`.

        Args:
            gamma_min (float): Minimum value for the schedule.
            gamma_max (float): Maximum value for the schedule.
        """
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        """
        Compute the value of the schedule at a given time step `t`.

        Args:
            t (torch.Tensor): Time step.

        Returns:
            torch.Tensor: Value of the schedule at time step `t`.
        """
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t


class LearntLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        """
        Learned linear schedule for a value between `gamma_min` and `gamma_max`.

        Args:
            gamma_min (float): Minimum value for the schedule.
            gamma_max (float): Maximum value for the schedule.
        """
        super().__init__()
        self.b = nn.Parameter(torch.tensor(gamma_min))
        self.w = nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def forward(self, t):
        """
        Compute the value of the schedule at a given time step `t`.

        Args:
            t (torch.Tensor): Time step.

        Returns:
            torch.Tensor: Value of the schedule at time step `t`.
        """
        return self.b + self.w.abs() * t
    