import torch
import numpy as np

class FlowMatchingScheduler:
    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = None

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        t = timesteps.float() / self.num_train_timesteps
        
        while len(t.shape) < len(original_samples.shape):
            t = t.unsqueeze(-1)
            
        noisy_samples = t * original_samples + (1.0 - t) * noise
        
        target_velocity = original_samples - noise
        
        return noisy_samples, target_velocity

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        step_size = self.num_train_timesteps / num_inference_steps
        timesteps = np.arange(0, self.num_train_timesteps, step_size)
        self.timesteps = torch.from_numpy(timesteps).to(device).float()

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, dt: float) -> torch.Tensor:
        # X_{t+dt} = X_t + v_\theta * dt_normalized
        dt_normalized = dt / self.num_train_timesteps
        next_sample = sample + model_output * dt_normalized
        return next_sample