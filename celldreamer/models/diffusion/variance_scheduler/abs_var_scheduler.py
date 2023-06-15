# https://github.com/Michedev/DDPMs-Pytorch/blob/72d621ea7b64793b82bc5cace3605b85dc5d0b03/variance_scheduler/abs_var_scheduler.py

from abc import abstractmethod, ABC


class Scheduler(ABC):

    @abstractmethod
    def get_alpha_hat(self):
        pass

    @abstractmethod
    def get_alphas(self):
        pass

    @abstractmethod
    def get_betas(self):
        pass

    @abstractmethod
    def get_posterior_variance(self):
        pass