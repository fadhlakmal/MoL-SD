from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from sd.samplers.base import Sampler


@contextmanager
def eval_mode(*modules: nn.Module):
    """Temporarily switch modules to eval, restoring each module's previous mode afterwards."""
    states = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        yield
    finally:
        for m, was_training in zip(modules, states):
            m.train(was_training)


def to_pil(images: torch.Tensor) -> list[Image.Image]:
    """(B, 3, H, W) in [-1, 1] -> list of PIL images."""
    arr = ((images.float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
    arr = arr.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(np.ascontiguousarray(a)) for a in arr]


class ConditionalPipeline:
    """Text + optional image-condition generation for any (objective, sampler) pair.

    The condition is VAE-encoded (posterior mean) and concatenated to the noisy latent on the channel axis.
    """

    def __init__(self, unet: nn.Module, vae: nn.Module, text_encoder: nn.Module, sampler: Sampler):
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.sampler = sampler

    @torch.no_grad()
    def __call__(
        self,
        prompts: list[str],
        condition: torch.Tensor | None = None,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_steps: int = 28,
        cfg_scale: float = 5.0,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Returns decoded images (B, 3, H, W) in [-1, 1]. condition: (B, 3, H, W) in [-1, 1]."""
        device = next(self.unet.parameters()).device
        bsz = len(prompts)
        use_cfg = cfg_scale != 1.0
        autocast = torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32)

        with eval_mode(self.unet, self.vae, self.text_encoder), autocast:
            context = self.text_encoder(prompts)
            if use_cfg:
                context = torch.cat([self.text_encoder([negative_prompt] * bsz), context])

            cond_latents = None
            if condition is not None:
                cond_latents = self.vae.encode(condition.to(device), sample=False)
                if use_cfg:
                    cond_latents = torch.cat([cond_latents, cond_latents])

            self.sampler.set_timesteps(num_steps, device)
            f = self.vae.downsample_factor
            x = self.sampler.init_noise((bsz, self.vae.latent_channels, height // f, width // f), generator, device)

            for i, t in enumerate(self.sampler.model_timesteps):
                model_in = self.sampler.scale_model_input(x, i)
                if use_cfg:
                    model_in = torch.cat([model_in, model_in])
                if cond_latents is not None:
                    model_in = torch.cat([model_in, cond_latents], dim=1)

                out = self.unet(model_in, t, context).float()
                if use_cfg:
                    uncond, cond = out.chunk(2)
                    out = uncond + cfg_scale * (cond - uncond)
                x = self.sampler.step(out, i, x)

            images = self.vae.decode(x)
        return images.float()
