import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm
from PIL import Image

from sd.utils.logger import WandbLogger
from sd.utils.checkpoint import CheckpointManager
from sd.pipelines.stable_diffusion import StableDiffusionPipeline

class Trainer:
    def __init__(self, config, unet, vae, clip, scheduler, dataloaders, optimizer, device):
        self.config = config
        self.device = device
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        
        self.logger = WandbLogger(project_name="mol-sd-finetune", run_name=config.experiment_name)
        self.checkpointer = CheckpointManager(save_dir=config.checkpoint_dir)
        self.pipeline = StableDiffusionPipeline(vae, clip, unet, scheduler)
        
        self.global_step = 0
        self.save_every_n_steps = config.training.save_every_n_steps

    def compute_objective_loss(self, batch):
        """Computes the MSE loss for a single condition batch."""
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

        noisy_latents = self.scheduler.add_noise(target_latents, noise, timesteps)
        unet_input = torch.cat([noisy_latents, condition_latents], dim=1)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            unet_input.requires_grad_(True)
            noise_pred = checkpoint.checkpoint(
                self.unet,
                unet_input,
                timesteps,
                encoder_hidden_states,
                use_reentrant=False
            )
            loss = F.mse_loss(noise_pred, noise)
            
        return loss

    def train(self):
        """Main training loop."""
        num_epochs = self.config.training.num_epochs
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            iterators = {name: iter(dl) for name, dl in self.dataloaders.items()}
            min_batches = min(len(dl) for dl in self.dataloaders.values())
            
            progress_bar = tqdm(range(min_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for _ in progress_bar:
                self.optimizer.zero_grad()
                
                total_loss = 0
                metrics = {}
                
                for task_name, task_config in self.config.data.tasks.items():
                    batch = next(iterators[task_name])
                    loss = self.compute_objective_loss(batch)
                    
                    weight = task_config.weight
                    scaled_loss = weight * loss
                    scaled_loss.backward()
                    
                    total_loss += scaled_loss.item()
                    metrics[f"loss_{task_name}"] = loss.item()
                
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
                self.optimizer.step()

                metrics["total_loss"] = total_loss
                metrics["lr"] = self.optimizer.param_groups[0]['lr']
                
                self.logger.log_metrics(metrics, step=self.global_step)
                progress_bar.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items() if "loss" in k})

                if self.save_every_n_steps > 0 and self.global_step > 0 and self.global_step % self.save_every_n_steps == 0:
                    self.save_and_evaluate()

                self.global_step += 1
                
        self.logger.finish()
        print("Training complete.")

    def save_and_evaluate(self):
        """Saves checkpoint and runs inference on a validation sample."""
        self.checkpointer.save(self.unet, self.optimizer, self.global_step, loss=0.0)
        self.unet.eval()
        
        first_task = list(self.dataloaders.keys())[0]
        val_batch = next(iter(self.dataloaders[first_task]))
        
        with torch.no_grad():
            image = self.pipeline.generate(
                prompt=val_batch["text"][0],
                condition_image=val_batch["condition"][0].unsqueeze(0).to(self.device),
                negative_prompt="blurry, distorted, low quality",
                height=self.config.data.size, 
                width=self.config.data.size,
                num_inference_steps=20, 
                cfg_scale=7.5, 
                device=self.device
            )
            caption = f"Step {self.global_step} | {first_task}: {val_batch['text'][0]}"
            self.logger.log_image(image, prompt=caption, step=self.global_step)
            
        self.unet.train()