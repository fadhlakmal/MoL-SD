import torch


class Sampler:
    """Common sampling interface.

    Usage:
        sampler.set_timesteps(n, device)
        x = sampler.init_noise(shape, generator, device)
        for i, t in enumerate(sampler.model_timesteps):
            out = model(sampler.scale_model_input(x, i), t, ...)
            x = sampler.step(out, i, x)
    `model_timesteps` are in the UNet's [0, 1000) scale.
    """

    model_timesteps: torch.Tensor

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        raise NotImplementedError

    def init_noise(self, shape, generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device)

    def scale_model_input(self, x: torch.Tensor, i: int) -> torch.Tensor:
        return x

    def step(self, model_output: torch.Tensor, i: int, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
