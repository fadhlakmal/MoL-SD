import torch

from molsd.samplers.base import Sampler


def shift_time(t: torch.Tensor, shift: float) -> torch.Tensor:
    """SD3 resolution-dependent timestep shift. Keeps 0 -> 0 and 1 -> 1; shift > 1 moves mass toward noise."""
    if shift == 1.0:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


class FlowEulerSampler(Sampler):
    """Euler integration of dx/dt = v from t=1 (noise) to t=0 (data)."""

    def __init__(self, shift: float = 1.0, num_train_timesteps: int = 1000):
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        sigmas = shift_time(torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32), self.shift)
        self.sigmas = sigmas.to(device)
        self.model_timesteps = self.sigmas[:-1] * self.num_train_timesteps

    def step(self, model_output: torch.Tensor, i: int, x: torch.Tensor) -> torch.Tensor:
        dt = self.sigmas[i + 1] - self.sigmas[i]
        return x + dt * model_output.to(x.dtype)
