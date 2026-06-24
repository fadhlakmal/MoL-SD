import torch
import numpy as np

class EulerDiscreteScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012):
        self.num_train_timesteps = num_train_timesteps
        
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=torch.float32) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.train_sigmas = self.sigmas.clone()
        
        self.init_noise_sigma = self.sigmas.max()
        self.timesteps = None

    def set_timesteps(self, num_inference_steps, device):
        timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)
        
        # sigmas = np.interp(timesteps, np.arange(0, len(self.sigmas)), self.sigmas.numpy())
        sigmas = np.interp(timesteps, np.arange(0, len(self.train_sigmas)), self.train_sigmas.cpu().numpy())
        sigmas = np.append(sigmas, 0.0) 
        self.sigmas = torch.from_numpy(sigmas).to(device)

    def scale_model_input(self, sample: torch.Tensor, timestep: int) -> torch.Tensor:
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]
        return sample / ((sigma ** 2 + 1) ** 0.5)

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        step_index = (self.timesteps == timestep).nonzero().item()
        
        sigma = self.sigmas[step_index]
        next_sigma = self.sigmas[step_index + 1]
        
        # x_t+1 = x_t + dt * (dx/dt)
        dt = next_sigma - sigma
        prev_sample = sample + model_output * dt
        
        return prev_sample