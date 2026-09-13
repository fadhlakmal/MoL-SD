import torch

from sd.objectives.base import Objective, TrainingPair, expand_like
from sd.samplers.flow_euler import FlowEulerSampler, shift_time


class FlowMatchingObjective(Objective):
    """SD3 rectified flow.

    x_t = (1 - t) * x0 + t * noise,  v = noise - x0,  t ~ logit-normal (or uniform), then shifted.
    t = 1 is pure noise, which lines up with SD1.5's timestep embedding when the UNet receives t * 1000.
    """

    name = "flow_matching"

    def __init__(
        self,
        timestep_sampling: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        shift: float = 1.0,
        num_train_timesteps: int = 1000,
    ):
        if timestep_sampling not in ("logit_normal", "uniform"):
            raise ValueError(f"unknown timestep_sampling: {timestep_sampling}")
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps

    def sample_t(self, batch_size, device, generator=None):
        if self.timestep_sampling == "logit_normal":
            u = torch.randn(batch_size, generator=generator, device=device) * self.logit_std + self.logit_mean
            u = torch.sigmoid(u)
        else:
            u = torch.rand(batch_size, generator=generator, device=device)
        return shift_time(u, self.shift)

    def prepare(self, x0, noise, t):
        te = expand_like(t, x0)
        return TrainingPair(
            x_t=(1.0 - te) * x0 + te * noise,
            target=noise - x0,
            model_t=t * self.num_train_timesteps,
            t_unit=t,
        )

    def make_sampler(self):
        return FlowEulerSampler(shift=self.shift, num_train_timesteps=self.num_train_timesteps)
