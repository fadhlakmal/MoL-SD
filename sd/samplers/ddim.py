import torch

from sd.samplers.base import Sampler


def scaled_linear_alphas_cumprod(
    num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012
) -> torch.Tensor:
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float64) ** 2
    return torch.cumprod(1.0 - betas, dim=0).float()


class DDIMSampler(Sampler):
    """Deterministic DDIM (eta=0) for epsilon-prediction models. 'leading' spacing with steps_offset=1, as SD1.5."""

    def __init__(self, num_train_timesteps: int = 1000, steps_offset: int = 1):
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.alphas_cumprod = scaled_linear_alphas_cumprod(num_train_timesteps)

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        ratio = self.num_train_timesteps // num_steps
        ts = (torch.arange(num_steps) * ratio).flip(0) + self.steps_offset
        self.model_timesteps = ts.clamp(max=self.num_train_timesteps - 1).to(device)
        self._ac = self.alphas_cumprod.to(device)

    def step(self, model_output: torch.Tensor, i: int, x: torch.Tensor) -> torch.Tensor:
        eps = model_output.to(x.dtype)
        a_t = self._ac[self.model_timesteps[i]]
        is_last = i == len(self.model_timesteps) - 1
        a_prev = torch.ones((), device=x.device) if is_last else self._ac[self.model_timesteps[i + 1]]

        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        return a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
