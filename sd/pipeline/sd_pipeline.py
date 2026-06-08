import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

class StableDiffusionPipeline:
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
                 num_inference_steps: int = 50, 
                 cfg_scale: float = 7.5, 
                 seed: int = None,
                 device: torch.device = torch.device("cuda")):
        
        self.vae.to(device)
        self.text_encoder.to(device)
        self.unet.to(device)

        # encode text & setup cfg
        cond_embeddings = self.text_encoder([prompt], device)        
        uncond_embeddings = self.text_encoder([negative_prompt], device)        
        # Shape: (2, 77, 768)
        context = torch.cat([uncond_embeddings, cond_embeddings])

        cond_latents = None
        if condition_image is not None:
            condition_image = condition_image.to(device)
            cond_latents = self.vae.encode(condition_image) * 0.18215
            cond_latents = torch.cat([cond_latents] * 2)

        # init noise 
        shape = (1, 4, height // 8, width // 8) 
        
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
            
        latents = torch.randn(shape, generator=generator, device=device)        
        self.scheduler.set_timesteps(num_inference_steps, device)

        # diffusion loop (denoising)
        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            
            if cond_latents is not None:
                # Concat along channel dim: (2, 4, H, W) + (2, 4, H, W) -> (2, 8, H, W)
                latent_model_input = torch.cat([latent_model_input, cond_latents], dim=1)
            
            noise_pred = self.unet(latent_model_input, t, context)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.scheduler.step(model_output=noise_pred, timestep=t, sample=latents)

        # decode latents to img
        latents = 1 / 0.18215 * latents
        
        image = self.vae(latents)
        
        # PT Tensor (-1 to 1) -> PIL Img (0 to 255 RGB)
        image = (image / 2 + 0.5).clamp(0, 1) 
        image = image.cpu().permute(0, 2, 3, 1).numpy() 
        
        image = (image * 255).round().astype(np.uint8)
        pil_image = Image.fromarray(image[0])
        
        return pil_image