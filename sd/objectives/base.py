from dataclasses import dataclass

import torch

from sd.samplers.base import Sampler


def expand_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1))).to(x.dtype)


@dataclass
class TrainingPair:
    x_t: torch.Tensor  # noisy latent fed to the model
    target: torch.Tensor  # regression target
    model_t: torch.Tensor  # (B,) timestep in the UNet's [0, 1000) scale
    t_unit: torch.Tensor  # (B,) noise level in [0, 1] (1 = pure noise), for logging


class Objective:
    """A diffusion training objective plus the sampler that inverts it."""

    name: str

    def sample_t(self, batch_size: int, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
        raise NotImplementedError

    def prepare(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> TrainingPair:
        raise NotImplementedError

    def loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE, shape (B,). Reduction is left to the caller so tasks/samples can be reweighted."""
        return (model_output.float() - target.float()).pow(2).flatten(1).mean(1)

    def make_sampler(self) -> Sampler:
        raise NotImplementedError
