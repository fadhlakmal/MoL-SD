import torch

from sd.objectives import EpsilonObjective, FlowMatchingObjective
from sd.samplers.flow_euler import shift_time


def test_flow_matching_endpoints_and_target():
    obj = FlowMatchingObjective()
    x0, noise = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    pair = obj.prepare(x0, noise, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(pair.x_t[0], x0[0])  # t=0 is data
    torch.testing.assert_close(pair.x_t[1], noise[1])  # t=1 is noise
    torch.testing.assert_close(pair.target, noise - x0)
    torch.testing.assert_close(pair.model_t, torch.tensor([0.0, 1000.0]))


def test_flow_matching_t_sampling_in_range():
    for sampling in ("logit_normal", "uniform"):
        for shift in (1.0, 3.0):
            t = FlowMatchingObjective(timestep_sampling=sampling, shift=shift).sample_t(10_000, torch.device("cpu"))
            assert t.min() >= 0 and t.max() <= 1
    # shift > 1 pushes mass toward noise
    g = torch.Generator().manual_seed(0)
    t1 = FlowMatchingObjective(shift=1.0).sample_t(10_000, torch.device("cpu"), g).mean()
    g = torch.Generator().manual_seed(0)
    t3 = FlowMatchingObjective(shift=3.0).sample_t(10_000, torch.device("cpu"), g).mean()
    assert t3 > t1


def test_shift_time_endpoints_and_monotone():
    t = torch.linspace(0, 1, 101)
    for shift in (0.5, 1.0, 3.0):
        s = shift_time(t, shift)
        assert s[0] == 0 and torch.isclose(s[-1], torch.tensor(1.0))
        assert (s[1:] >= s[:-1]).all()


def test_flow_euler_with_oracle_velocity_recovers_data():
    for shift in (1.0, 3.0):
        obj = FlowMatchingObjective(shift=shift)
        sampler = obj.make_sampler()
        sampler.set_timesteps(7, torch.device("cpu"))
        x0 = torch.randn(1, 4, 8, 8)
        x = sampler.init_noise(x0.shape, torch.Generator().manual_seed(1), torch.device("cpu"))
        noise = x.clone()
        assert sampler.model_timesteps[0] == 1000  # starts at pure noise
        for i, _ in enumerate(sampler.model_timesteps):
            x = sampler.step(noise - x0, i, x)
        torch.testing.assert_close(x, x0, atol=1e-5, rtol=0)


def test_ddim_with_oracle_epsilon_recovers_data():
    obj = EpsilonObjective()
    sampler = obj.make_sampler()
    sampler.set_timesteps(10, torch.device("cpu"))
    x0 = torch.randn(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)
    x = obj.prepare(x0, noise, sampler.model_timesteps[:1]).x_t
    for i, _ in enumerate(sampler.model_timesteps):
        x = sampler.step(noise, i, x)
    torch.testing.assert_close(x, x0, atol=1e-4, rtol=0)
