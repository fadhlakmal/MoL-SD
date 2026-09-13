"""Real-weight parity against diffusers. Run with: uv run pytest -m weights"""

import os

import pytest
import torch

from sd.models.loader import build_unet, build_vae

UNET = "weights/sd15_unet.safetensors"
VAE_W = "weights/sd15_vae.safetensors"
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pytestmark = [
    pytest.mark.weights,
    pytest.mark.skipif(not (os.path.exists(UNET) and os.path.exists(VAE_W)), reason="run scripts.convert_sd15 first"),
]


@torch.no_grad()
def test_unet_parity_sd15():
    from diffusers import UNet2DConditionModel

    ref = UNet2DConditionModel.from_pretrained(REPO, subfolder="unet").eval()
    ours = build_unet(UNET).eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 4, 32, 32, generator=g)
    ctx = torch.randn(1, 77, 768, generator=g)
    for t in (999.0, 500.0, 20.0):
        diff = (ours(x, t, ctx) - ref(x, t, ctx).sample).abs().max().item()
        assert diff < 1e-4, f"t={t}: max abs diff {diff}"


@torch.no_grad()
def test_vae_parity_sd15():
    from diffusers import AutoencoderKL

    ref = AutoencoderKL.from_pretrained(REPO, subfolder="vae").eval()
    ours = build_vae(VAE_W)
    img = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0)) * 2 - 1
    z = ours.encode(img, sample=False)
    assert (z / ours.scaling_factor - ref.encode(img).latent_dist.mean).abs().max() < 1e-4
    assert (ours.decode(z) - ref.decode(z / ours.scaling_factor).sample).abs().max() < 1e-4
