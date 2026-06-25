import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from sd.engine.trainer import Trainer
from sd.pipelines.optimal_transport import FlowMatchingPipeline

class OTTrainer(Trainer):
    def __init__(self, config, unet, vae, clip, scheduler, dataloaders, optimizer, device):
        super().__init__(config, unet, vae, clip, scheduler, dataloaders, optimizer, device)
        self.pipeline = FlowMatchingPipeline(vae, clip, unet, scheduler)

    def compute_objective_loss(self, batch):
        """Computes the Rectified Flow / Optimal Transport loss for a task batch."""
        targets = batch["target"].to(self.device)
        conditions = batch["condition"].to(self.device)
        prompts = batch["text"]

        with torch.no_grad():
            target_latents = self.vae.encode(targets) * 0.18215
            condition_latents = self.vae.encode(conditions) * 0.18215
            encoder_hidden_states = self.clip(prompts, self.device)
        
        noise = torch.randn_like(target_latents)
        bsz = target_latents.shape[0]
        
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=self.device).long()

        noisy_latents, target_velocity = self.scheduler.add_noise(target_latents, noise, timesteps)
        
        unet_input = torch.cat([noisy_latents, condition_latents], dim=1)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            unet_input.requires_grad_(True)
            
            velocity_pred = checkpoint.checkpoint(
                self.unet,
                unet_input,
                timesteps,
                encoder_hidden_states,
                use_reentrant=False
            )
            
            # Target is (X_data - X_noise)
            loss = F.mse_loss(velocity_pred, target_velocity)
            
        return loss