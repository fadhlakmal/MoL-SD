import torch

from sd.objectives.base import Objective, TrainingPair, expand_like
from sd.samplers.ddim import DDIMSampler, scaled_linear_alphas_cumprod


class EpsilonObjective(Objective):
    """DDPM epsilon prediction with SD1.5's scaled-linear schedule (baseline)."""

    name = "epsilon"

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.alphas_cumprod = scaled_linear_alphas_cumprod(num_train_timesteps)

    def sample_t(self, batch_size, device, generator=None):
        return torch.randint(0, self.num_train_timesteps, (batch_size,), generator=generator, device=device)

    def prepare(self, x0, noise, t):
        ac = expand_like(self.alphas_cumprod.to(x0.device)[t], x0)
        return TrainingPair(
            x_t=ac.sqrt() * x0 + (1 - ac).sqrt() * noise,
            target=noise,
            model_t=t,
            t_unit=t.float() / (self.num_train_timesteps - 1),
        )

    def make_sampler(self):
        return DDIMSampler(self.num_train_timesteps)
