import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

class FlowMatchingPipeline:
    def __init__(self, vae, text_encoder, unet, scheduler):
        self.vae = vae
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler

        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()

    @torch.no_grad()
    def generate(self, 
                 prompt: str, 
                 condition_image: torch.Tensor = None,
                 negative_prompt: str = "", 
                 height: int = 512, 
                 width: int = 512, 
                 num_inference_steps: int = 20, 
                 cfg_scale: float = 7.5, 
                 seed: int = None,
                 device: torch.device = torch.device("cuda")):
        
        self.vae.to(device)
        self.text_encoder.to(device)
        self.unet.to(device)

        cond_embeddings = self.text_encoder([prompt], device)        
        uncond_embeddings = self.text_encoder([negative_prompt], device)        
        context = torch.cat([uncond_embeddings, cond_embeddings])

        cond_latents = None
        if condition_image is not None:
            condition_image = condition_image.to(device)
            cond_latents = self.vae.encode(condition_image) * 0.18215
            cond_latents = torch.cat([cond_latents] * 2)

        shape = (1, 4, height // 8, width // 8) 
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
            
        latents = torch.randn(shape, generator=generator, device=device)
        self.scheduler.set_timesteps(num_inference_steps, device)
        dt = self.scheduler.num_train_timesteps / num_inference_steps

        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            
            if cond_latents is not None:
                latent_model_input = torch.cat([latent_model_input, cond_latents], dim=1)

            t_input = t.expand(latent_model_input.shape[0]).long()
            
            # Predict velocity
            velocity_pred = self.unet(latent_model_input, t_input, context)
            velocity_pred_uncond, velocity_pred_text = velocity_pred.chunk(2)
            
            # CFG on velocity vector
            velocity = velocity_pred_uncond + cfg_scale * (velocity_pred_text - velocity_pred_uncond)
            
            # Euler step
            latents = self.scheduler.step(model_output=velocity, timestep=t, sample=latents, dt=dt)

        latents = 1 / 0.18215 * latents
        image = self.vae(latents)
        image = (image / 2 + 0.5).clamp(0, 1) 
        image = image.to(torch.float32).cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype(np.uint8)
        
        return Image.fromarray(image[0])